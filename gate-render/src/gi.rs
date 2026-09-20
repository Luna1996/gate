//! GI：**屏幕空间逐面 ReSTIR**（唯一的 GI 路径）。
//! 算法与 reservoir 布局见 `assets/shaders/voxel_raytrace/gi/screen.wesl`；Rust 侧只负责开
//! reservoir 双缓冲（每 GI 像素 `GI_RES_WORDS` 个 u32，随 GI 分辨率重建）、上传 uniform、
//! 建 bind group、每帧 ping-pong 换绑。
//! 着色 pass（`gi_main`）在 `dispatch_dda` 里排在主 pass 之前，一次派发完成
//! 「新鲜候选 → 时域复用 → 空间复用 → 着色」；没有独立的 GI 更新 / 复用 pass。
//!
//! 历史（为什么没有世界空间缓存了）：曾经有一条「逐 (体素,面) 辐照度缓存」的旁路
//! （哈希网格 + 每帧轮转更新 + `1/n` 运行平均收敛，与屏幕空间路径二选一）。它与射线求值
//! 构成**跨帧反馈环**（缓存值进射线、射线值又进缓存），表现是「像拿刷子一层层刷、稳态要等
//! 二十秒、有无 GI 的异常区」—— 三条都是「逐帧确定」的反面。已整条删除；GI 的输入现在只依赖
//! 几何与光照，因此**不存在「收敛」这件事**：第 1 帧就是稳态，差别只有噪声（由降噪链压）。

use bevy::render::render_resource::{
  BindGroupLayoutDescriptor, CachedComputePipelineId, ShaderType,
};
use glam::{Mat4, UVec2, UVec4, Vec4};

use crate::wesl_consts::gi_consts;

/// GI uniform（WESL `bindings.wesl` 的 `GiUniform` 逐字段镜像，字节一致）。
#[repr(C)]
#[derive(Debug, Default, Clone, Copy, ShaderType)]
pub struct GiUniform {
  /// x = **二次顶点太阳反弹**（1 = 开、0 = 关；菜单「渲染/RESTIR GI/太阳反弹」）、
  /// y = 保留（恒 0）、z = GI 增益、w = 保留（恒 0）
  pub params: Vec4,
  /// x = GI 开关（0/1）、yzw = 保留（恒 0）
  pub misc: Vec4,
  /// x = 保留（恒 0）、y = GI 分辨率除数（1 = 全分辨率、2 = 半分辨率、4 = 四分之一；**整数值的 f32**，
  /// 只被 `gi_main` 用来把本 pass 的像素下标换成 beam 纹理下标）、
  /// z = 世界几何自「写入上一帧 reservoir 的那一帧」起是否**逐位未变**（1 = 未变）⇒
  /// 跳过时域二次顶点键验证射线、w = **降噪质量档位**（0 = 关 / 1 = 低 / 2 = 中 / 3 = 高；
  /// 菜单「渲染/RESTIR GI/降噪质量」）：`gi_ss_main` 按它取 1/4 档的新鲜候选数
  /// （`GI_SS_CAND_N` vs `..._HQ`）与记忆窗（`GI_SS_M_CAP_K` vs `..._HQ`）——**这两项与分辨率档无关**。
  /// 时域/atrous 的派发与核半径不在本 pass 里，由 Rust 的 `denoise_plan` 决定。
  pub flags: Vec4,
  /// x = 自增帧号（精确 u32；yzw 恒 0）。所有整数帧逻辑（本帧的 RNG 种子混入、像素 hash）都用它：
  /// 帧号曾经以 f32 存在 `params.x`，超过 2^24 后无法表示连续整数 ⇒ 种子会偶发重复。
  pub seq: UVec4,
  /// 上一帧的相机矩阵（时域复用：把本帧主命中点重投影到上帧 GI 网格）。
  /// `prepare_gi` 每帧把上帧实际用过的那一份写进 uniform，再把当前帧的存下来 ⇒ 与上帧逐位一致。
  /// 只被 `prev_view_proj` 消费（重投影三维点不需要逆矩阵）。
  pub prev_view_proj: Mat4,
}

/// GI 档位（菜单「渲染/GI」）：`enabled` → uniform `misc.x`；`gi_div` → uniform `flags.y`。
#[derive(bevy::ecs::resource::Resource, Clone, Copy, Debug, PartialEq)]
pub struct GiSettings {
  /// GI 开关（关掉 = 整条 GI 链不派发，主 pass 只有太阳直射 + 天光兜底）。
  pub enabled: bool,
  /// GI **分辨率除数**：1 = 全分辨率、2 = 半分辨率、4 = 四分之一（GI 网格边长 = 渲染分辨率 ÷ 本值）。
  /// 不是开关：**每一档都跑 GI**，只是网格疏密与代价不同。
  /// 代价按 **GI 像素数**涨：射线、降噪（时域 + 5 轮 atrous）、reservoir/历史带宽全都 ∝ 1/除数²
  /// ⇒ 1/2 约是全分辨率的 1/4 时间、1/4 再降 4 倍。画质上 GI 是低频信号，1/4 几乎无差
  /// （回全分辨率由几何感知上采样兜住几何边界，见 `main.wesl` 的 `gi_upsample_joint`）。
  pub gi_div: u32,
  /// **降噪质量档**（菜单「渲染/RESTIR GI/降噪质量」）：把降噪链按成本分档，方便直接 A/B。
  /// 每档只比上一档多一件事，成本单调递增（数值的权威在 `gi/consts.wesl`，Rust 只决定派发与取值）；
  /// **与「分辨率」档正交**：下面这些项目对所有分辨率档一视同仁。
  ///
  /// | 档 | 时域累积 | atrous | 每像素候选数 | 记忆窗 | 比上一档多花什么 |
  /// |---|---|---|---|---|---|
  /// | 0 关 | ✗ | ✗ | 4 | 20 | —（基线：原始 GI + 几何感知上采样） |
  /// | 1 低 | ✓ | 5 轮 · 3×3（8 tap） | 4 | 20 | +1 时域 pass + 5 个 atrous pass |
  /// | 2 中 | ✓ | 5 轮 · 5×5（24 tap） | 4 | 20 | atrous 每轮 tap 8→24 |
  /// | 3 高 | ✓ | 5 轮 · 5×5 | 8 | 32 | **候选数翻倍 = GI 射线翻倍**（全链最贵的一项） |
  ///
  /// 档 0 时 `dda_main` 直接采样原始 GI（`gi_out`）——**完全不降噪**，也没有时域/atrous pass。
  /// 「候选数」与「记忆窗」只区分最高档（`screen.wesl` 读 `gi_u.flags.w`）；
  /// 「atrous 核半径」区分 0..=1 / 2..=3（写进降噪的小配置 buffer，见 `AuxTexCache::den_cfg`）；
  /// 「时域/atrous 是否派发、几轮」由 `denoise_plan` 决定（档 0 = 0 轮）。
  /// 「几何感知上采样」不属于降噪（那是把 GI 在几何边界上正确重建），所有档都保留。
  ///
  /// 与业界 NRD/RELAX 的对照见 README 的技术要点第 4 条：本链 = 时域累积（方差驱动权重 + 累积矩 +
  /// AABB 钳制）+ 5 轮迭代 atrous；**没有**独立的前滤 pass（等价物是时域内的共面邻域 mean ± K·σ
  /// 离群钳制）与 fast-history/history-fix。
  pub denoise: u32,
  /// **二次顶点的太阳反弹**（菜单「渲染/RESTIR GI/太阳反弹」）：**默认关**。
  /// GI 射线命中点（二次顶点）是否做一次太阳 NEE —— 朝太阳发一条阴影射线，把「被阳光照亮的
  /// 表面」这一路能量算进间接光。
  ///
  /// 关掉 ⇒ 二次顶点只剩天光项（`albedo · 天光 · AO / π`），**整条阴影射线都不发**。
  /// 实测 2K + 1/4 档：`gate_gi` 27.1 → 16.1ms（占该 pass 的 41%），全帧 31.6 → 20.4ms。
  /// 代价是阴影区/背光面失去「阳光经一次弹射照进来」的贡献 ⇒ 变暗变平；这是**能量层面的取舍**
  /// （不是采样层面的噪声取舍），降噪救不回来。静态场景实测画面差异很小。
  ///
  /// 为什么用"砍掉"而不是"缓存/查表"：那 11ms 里 90% 花在「>32 voxel 的长程遍历」上
  /// （短程只值 1.1ms），而太阳方向在昼夜循环下每帧变化 ⇒ 世界空间的太阳可见性缓存不成立
  /// （流式大世界更不成立）。
  pub sun_bounce: bool,
}

/// 「降噪质量」档的派发计划（`(是否跑降噪, atrous 轮数, atrous 核半径)`）。
/// 核半径的权威值在 `gi/consts.wesl`；轮数取 `GI_DEN_ATROUS_ITER`（档 0 = 0 轮 = 完全不跑）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DenoisePlan {
  /// 是否派发时域 + atrous（档 0 = false ⇒ `dda_main` 直接采样原始 GI）。
  pub on: bool,
  /// atrous 派发几轮（0 到 `GI_DEN_ATROUS_ITER`）。
  pub rounds: u32,
  /// atrous 单轮核半径（1 = 3×3，2 = 5×5）。
  pub radius: u32,
}

impl GiSettings {
  /// 菜单「渲染/GI/分辨率」的三个档位（网格除数）；**下标 = `switch_group` 的选中序号**。
  /// 菜单侧只认下标 ⇒ 档位增删必须改这里（`debug_menu.rs` 的观察者用同一份）。
  pub const DIV_CHOICES: [u32; 3] = [1, 2, 4];

  /// 菜单「渲染/GI/降噪质量」的档位数（0 = 关 ..= 3 = 高）；**下标 = 档位本身**。
  pub const DENOISE_TIERS: u32 = 4;

  /// 生效的分辨率除数（越界值钳回来）。
  pub fn div(&self) -> u32 {
    self.gi_div.clamp(1, 4)
  }

  /// 生效的降噪档位（越界值钳回来）。
  pub fn tier(&self) -> u32 {
    self.denoise.min(Self::DENOISE_TIERS - 1)
  }

  /// 本档的降噪派发计划（见 `DenoisePlan`；`gi/consts.wesl` 的 `GI_DEN_ATROUS_R*` / `_ITER` 是参数源）。
  pub fn denoise_plan(&self) -> DenoisePlan {
    let c = gi_consts();
    match self.tier() {
      // 关：一条降噪 pass 都不跑（`dda_main` 直接采样原始 GI）。
      0 => DenoisePlan { on: false, rounds: 0, radius: c.gi_den_atrous_r_fast },
      // 低：时域累积 + 5 轮 3×3（8 tap）atrous。
      1 => DenoisePlan { on: true, rounds: c.gi_den_atrous_iter, radius: c.gi_den_atrous_r_fast },
      // 中/高：atrous 换 5×5（24 tap）；高 额外动候选数与记忆窗（在 `screen.wesl` 里按档取）。
      _ => DenoisePlan { on: true, rounds: c.gi_den_atrous_iter, radius: c.gi_den_atrous_r },
    }
  }

  /// GI 网格尺寸 = 渲染分辨率 ÷ `div()`（逐轴向下取整，至少 1×1）。
  pub fn gi_size(&self, render_size: UVec2) -> UVec2 {
    let d = self.div();
    UVec2::new((render_size.x / d).max(1), (render_size.y / d).max(1))
  }
}

impl Default for GiSettings {
  fn default() -> Self {
    // 默认 1/4 档 + 「低」降噪档：与实测最划算的组合一致（时域 + 3×3 的 5 轮 atrous），
    // 想要更干净就往「中/高」拨，想量原始噪声与上限帧率就拨到「关」。
    // 太阳反弹默认关：实测画面差异细微（静态场景几乎看不出），代价却是 `gate_gi` 的 41%。
    Self { enabled: true, gi_div: 4, denoise: 1, sun_bounce: false }
  }
}

/// BG4 布局：uniform(0) + reservoir 双缓冲（20 = 本帧写、21 = 上帧读）。
/// 被 `gi_main`（读写 reservoir）与降噪前的着色共用。
pub fn gi_bg4_layout() -> BindGroupLayoutDescriptor {
  use bevy::render::render_resource::*;
  const C: ShaderStages = ShaderStages::COMPUTE;
  let buf = |binding: u32| BindGroupLayoutEntry {
    binding,
    visibility: C,
    ty: BindingType::Buffer {
      ty: BufferBindingType::Storage { read_only: false },
      has_dynamic_offset: false,
      min_binding_size: None,
    },
    count: None,
  };
  BindGroupLayoutDescriptor::new(
    "GiBg4",
    &[
      BindGroupLayoutEntry {
        binding: 0,
        visibility: C,
        ty: BindingType::Buffer {
          ty: BufferBindingType::Uniform,
          has_dynamic_offset: false,
          min_binding_size: Some(GiUniform::min_size()),
        },
        count: None,
      },
      buf(20),
      buf(21),
    ],
  )
}

/// BG5 写入侧布局：GI 的两张输出纹理（2 = gi、3 = cov）+ 降噪导引 buffer（6）。
/// 纹理尺寸 = GI 网格（渲染分辨率 ÷ `GiSettings.gi_div`），与全分辨率无关。
/// 采样侧（binding 4/5）在 `brickmap::dda` 里单独一份 layout，只给 `dda_main`。
pub fn gi_bg5_layout() -> BindGroupLayoutDescriptor {
  use bevy::render::render_resource::*;
  const C: ShaderStages = ShaderStages::COMPUTE;
  let store = |binding: u32, format: TextureFormat| BindGroupLayoutEntry {
    binding,
    visibility: C,
    ty: BindingType::StorageTexture {
      access: StorageTextureAccess::WriteOnly,
      format,
      view_dimension: TextureViewDimension::D2,
    },
    count: None,
  };
  BindGroupLayoutDescriptor::new(
    "GiBg5",
    &[
      // GI 网格分辨率下的 GI：rgb = gi·valid、a = valid
      store(2, TextureFormat::Rgba16Float),
      // GI 网格分辨率下的覆盖度：r = cov·valid、g = valid
      store(3, TextureFormat::Rg32Float),
      // 降噪导引（`gi_main` 写、两段降噪读；布局见 WESL `gi/consts.wesl`）
      BindGroupLayoutEntry {
        binding: 6,
        visibility: C,
        ty: BindingType::Buffer {
          ty: BufferBindingType::Storage { read_only: false },
          has_dynamic_offset: false,
          min_binding_size: None,
        },
        count: None,
      },
    ],
  )
}

/// 降噪时域段的布局：**只有 group(0) 一份**（binding 号见 `bindings.wesl`）。
/// 单 group 是刻意的：两段降噪都不需要相机矩阵（上帧像素坐标由 `gi_main` 写进导引）、
/// 也不需要 `grid_descs`（平面检验容差同样写在导引里）⇒ pipeline layout 不必拉进 bg1..bg4。
pub fn gi_den_temporal_layout() -> BindGroupLayoutDescriptor {
  use bevy::render::render_resource::*;
  const C: ShaderStages = ShaderStages::COMPUTE;
  let ro = |binding: u32| BindGroupLayoutEntry {
    binding,
    visibility: C,
    ty: BindingType::Buffer {
      ty: BufferBindingType::Storage { read_only: true },
      has_dynamic_offset: false,
      min_binding_size: None,
    },
    count: None,
  };
  let rw = |binding: u32| BindGroupLayoutEntry {
    binding,
    visibility: C,
    ty: BindingType::Buffer {
      ty: BufferBindingType::Storage { read_only: false },
      has_dynamic_offset: false,
      min_binding_size: None,
    },
    count: None,
  };
  let tex = |binding: u32| BindGroupLayoutEntry {
    binding,
    visibility: C,
    ty: BindingType::Texture {
      sample_type: TextureSampleType::Float { filterable: true },
      view_dimension: TextureViewDimension::D2,
      multisampled: false,
    },
    count: None,
  };
  BindGroupLayoutDescriptor::new(
    "GiDenTemporal",
    &[
      ro(10),  // 导引
      tex(11), // 本帧原始 GI（gi_out 的采样视图）
      ro(12),  // 历史（上帧）
      rw(13),  // 历史（本帧）
      BindGroupLayoutEntry {
        binding: 16, // 时域输出
        visibility: C,
        ty: BindingType::StorageTexture {
          access: StorageTextureAccess::WriteOnly,
          format: TextureFormat::Rgba16Float,
          view_dimension: TextureViewDimension::D2,
        },
        count: None,
      },
      rw(17), // φ（每像素亮度 range 尺度）
      ro(20), // 降噪小配置（`[0]` = atrous 核半径；降噪 pass 够不到 @group(4) 的 `gi_u`）
    ],
  )
}

/// 降噪空间段（迭代 atrous）的布局：group(0)，binding 14 = 采样输入、15 = 存储输出、10 = 导引、17 = φ。
/// Rust 每轮换绑 14/15（同一份 layout、同一个入口，见 `AuxTexCache::den_bg`）。
pub fn gi_den_atrous_layout() -> BindGroupLayoutDescriptor {
  use bevy::render::render_resource::*;
  const C: ShaderStages = ShaderStages::COMPUTE;
  BindGroupLayoutDescriptor::new(
    "GiDenAtrous",
    &[
      BindGroupLayoutEntry {
        binding: 10,
        visibility: C,
        ty: BindingType::Buffer {
          ty: BufferBindingType::Storage { read_only: true },
          has_dynamic_offset: false,
          min_binding_size: None,
        },
        count: None,
      },
      BindGroupLayoutEntry {
        binding: 14,
        visibility: C,
        ty: BindingType::Texture {
          sample_type: TextureSampleType::Float { filterable: true },
          view_dimension: TextureViewDimension::D2,
          multisampled: false,
        },
        count: None,
      },
      BindGroupLayoutEntry {
        binding: 15,
        visibility: C,
        ty: BindingType::StorageTexture {
          access: StorageTextureAccess::WriteOnly,
          format: TextureFormat::Rgba16Float,
          view_dimension: TextureViewDimension::D2,
        },
        count: None,
      },
      BindGroupLayoutEntry {
        binding: 17,
        visibility: C,
        ty: BindingType::Buffer {
          ty: BufferBindingType::Storage { read_only: false },
          has_dynamic_offset: false,
          min_binding_size: None,
        },
        count: None,
      },
      BindGroupLayoutEntry {
        binding: 20,
        visibility: C,
        ty: BindingType::Buffer {
          ty: BufferBindingType::Storage { read_only: true },
          has_dynamic_offset: false,
          min_binding_size: None,
        },
        count: None,
      },
    ],
  )
}

#[derive(bevy::ecs::resource::Resource)]
pub struct GiGpu {
  pub uniform: bevy::render::render_resource::UniformBuffer<GiUniform>,
  pub frame: u32,
  /// 上一帧 `gi_main` 实际用过的相机矩阵（uniform `prev_view_proj` 的来源）。
  /// 只在真正跑 GI 的帧更新 ⇒ 与上帧写入 reservoir 时用的矩阵逐位一致（时域重投影才准）。
  pub prev_view_proj: Mat4,
  /// reservoir 双缓冲的换绑状态：true ⇒ 本帧 `binding 20 = b`、`21 = a`（见 `prepare_gi`）。
  pub res_flip: bool,
  /// 世界几何修订号：**只在世界真的可能变了的那一帧**自增（全量上传 / palette 变化 / 收到脏盒）。
  /// `world_rev_gi` 要问的问题比「世界变了吗」更窄：「自上次跑 `gi_main` 起，射线求交的输入
  /// （体素占据）是否逐个字节没变」——只有那时才能确定上帧 reservoir 里的二次顶点键本帧仍逐位成立。
  pub world_rev: u32,
  /// 上一次真正跑 `gi_main` 的那一帧的 `world_rev`（那正是 reservoir 双缓冲里「上帧」的来源帧）
  /// ⇒ `world_rev == world_rev_gi` ⇔ 上帧 reservoir 存的二次顶点键在本帧仍然逐位成立。
  pub world_rev_gi: u32,
  /// 降噪 pipeline：`[0]` = 时域、`[1..6]` = atrous 第 1..5 轮（步长 1/2/4/8/16）。
  /// layout 只有 group(0) 一份（见 [`gi_den_temporal_layout`] / [`gi_den_atrous_layout`]）；
  /// 实际跑几轮由 `GI_DEN_ATROUS_ITER` 决定（1..=5，Rust 按它选 src→dst 链）。
  pub den_pipelines: [Option<CachedComputePipelineId>; 6],
}

#[derive(bevy::ecs::resource::Resource)]
pub struct GiBg4(pub bevy::render::render_resource::BindGroup);

/// GI 写入侧 bind group（`gi_main` 用）
#[derive(bevy::ecs::resource::Resource)]
pub struct GiBg5(pub bevy::render::render_resource::BindGroup);

/// GI 写入侧的占位纹理（1×1）：GI 缓冲未就绪时 BG5 仍须为 binding 2/3 提供视图。
/// 占位纹理不会被真正写入。
#[derive(bevy::ecs::resource::Resource, Default)]
struct GiPlaceholder {
  tex: Option<bevy::render::render_resource::Texture>,
  cov: Option<bevy::render::render_resource::Texture>,
  view: Option<bevy::render::render_resource::TextureView>,
  cov_view: Option<bevy::render::render_resource::TextureView>,
  /// BG4 binding 20/21 的占位（GI 缓冲未就绪时用；1 个 word，足够绑定，不会被读写）。
  res: Option<bevy::render::render_resource::Buffer>,
}

impl GiPlaceholder {
  fn views(
    &mut self,
    device: &bevy::render::renderer::RenderDevice,
  ) -> (&bevy::render::render_resource::TextureView, &bevy::render::render_resource::TextureView)
  {
    use bevy::render::render_resource::*;
    if self.tex.is_none() {
      let make = |label: &str, format: TextureFormat| {
        device.create_texture(&TextureDescriptor {
          label: Some(label),
          size: Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
          mip_level_count: 1,
          sample_count: 1,
          dimension: TextureDimension::D2,
          format,
          usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
          view_formats: &[],
        })
      };
      let t = make("gate_gi_placeholder", TextureFormat::Rgba16Float);
      let c = make("gate_gi_cov_placeholder", TextureFormat::Rg32Float);
      self.view = Some(t.create_view(&TextureViewDescriptor::default()));
      self.cov_view = Some(c.create_view(&TextureViewDescriptor::default()));
      self.tex = Some(t);
      self.cov = Some(c);
    }
    (self.view.as_ref().expect("刚创建"), self.cov_view.as_ref().expect("刚创建"))
  }

  /// BG4 binding 20/21 的占位 buffer（4 B）。
  fn res_buffer(
    &mut self,
    device: &bevy::render::renderer::RenderDevice,
  ) -> &bevy::render::render_resource::Buffer {
    use bevy::render::render_resource::{BufferDescriptor, BufferUsages};
    self.res.get_or_insert_with(|| {
      device.create_buffer(&BufferDescriptor {
        label: Some("gate_gi_res_placeholder"),
        size: 4,
        usage: BufferUsages::STORAGE,
        mapped_at_creation: false,
      })
    })
  }
}

pub struct GiPlugin;

impl bevy::app::Plugin for GiPlugin {
  fn build(&self, app: &mut bevy::app::App) {
    use bevy::ecs::schedule::IntoScheduleConfigs;
    app.init_resource::<GiSettings>();
    let Some(render_app) = app.get_sub_app_mut(bevy::render::RenderApp) else {
      return;
    };
    render_app
      .init_resource::<GiSettings>()
      .init_resource::<GiPlaceholder>()
      .add_systems(bevy::render::RenderStartup, init_gi_gpu)
      .add_systems(
        bevy::render::RenderStartup,
        queue_gi_pipelines.after(crate::brickmap::dda::init_dda_pipelines),
      )
      .add_systems(bevy::render::ExtractSchedule, extract_gi_settings)
      .add_systems(
        bevy::render::Render,
        prepare_gi
          .in_set(bevy::render::RenderSystems::PrepareBindGroups)
          .after(crate::brickmap::upload::prepare)
          .after(crate::brickmap::dda::prepare_dda_bind_groups),
      );
  }
}

fn init_gi_gpu(mut commands: bevy::ecs::system::Commands) {
  let c = gi_consts();
  bevy::log::info!(
    target: "gate",
    "GI：屏幕空间逐面 ReSTIR（reservoir {} word/像素 × 2 块 ping-pong）；\
     降噪导引 {} word/像素、历史 {} word/像素（双缓冲）、atrous 迭代 {} 轮",
    c.gi_res_words,
    c.gi_den_guide_words,
    c.gi_den_hist_words,
    c.gi_den_atrous_iter,
  );
  commands.insert_resource(GiGpu {
    uniform: bevy::render::render_resource::UniformBuffer::default(),
    frame: 0,
    // 首帧没有「上一帧」⇒ 恒等矩阵；此时 reservoir 两块都是零（M = 0）⇒ 复用一律判无效。
    prev_view_proj: Mat4::IDENTITY,
    res_flip: false,
    // 两者都从 0 起 ⇒ 首帧若恰好没有检测到任何变化，`skip_verify` 会是真；那时 reservoir 两块
    // 都是零（M = 0 ⇒ 一律判无效），跳过与否都不会接受任何历史 ⇒ 安全。
    world_rev: 0,
    world_rev_gi: 0,
    den_pipelines: [None; 6],
  });
}

fn queue_gi_pipelines(
  pipeline_cache: bevy::ecs::system::Res<bevy::render::render_resource::PipelineCache>,
  dda_shader: bevy::ecs::system::Res<crate::shader::DdaShaderHandle>,
  mut gpu: bevy::ecs::system::ResMut<GiGpu>,
) {
  use bevy::render::render_resource::ComputePipelineDescriptor;
  use std::borrow::Cow;
  // 降噪（两步四 pass）：不依赖 DdaPipelines —— layout 只有 group(0) 一份。
  if gpu.den_pipelines[0].is_some() {
    return;
  }
  let den_temporal_layout = gi_den_temporal_layout();
  let den_atrous_layout = gi_den_atrous_layout();
  let label = [
    "gate_gi_denoise_temporal",
    "gate_gi_denoise_atrous1",
    "gate_gi_denoise_atrous2",
    "gate_gi_denoise_atrous4",
    "gate_gi_denoise_atrous8",
    "gate_gi_denoise_atrous16",
  ];
  let entry = [
    "gi_denoise_temporal",
    "gi_denoise_atrous1",
    "gi_denoise_atrous2",
    "gi_denoise_atrous4",
    "gi_denoise_atrous8",
    "gi_denoise_atrous16",
  ];
  for i in 0..6 {
    let layout =
      if i == 0 { vec![den_temporal_layout.clone()] } else { vec![den_atrous_layout.clone()] };
    gpu.den_pipelines[i] = Some(pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
      label: Some(Cow::from(label[i])),
      layout,
      shader: dda_shader.0.clone(),
      entry_point: Some(Cow::from(entry[i])),
      ..Default::default()
    }));
  }
}

fn extract_gi_settings(
  mut commands: bevy::ecs::system::Commands,
  settings: Option<bevy::render::Extract<bevy::ecs::system::Res<GiSettings>>>,
) {
  commands.insert_resource(settings.map_or_else(GiSettings::default, |s| GiSettings {
    enabled: s.enabled,
    gi_div: s.div(),
    denoise: s.tier(),
    sun_bounce: s.sun_bounce,
  }));
}

#[allow(clippy::too_many_arguments)]
fn prepare_gi(
  mut commands: bevy::ecs::system::Commands,
  device: bevy::ecs::system::Res<bevy::render::renderer::RenderDevice>,
  queue: bevy::ecs::system::Res<bevy::render::renderer::RenderQueue>,
  pipeline_cache: bevy::ecs::system::Res<bevy::render::render_resource::PipelineCache>,
  settings: bevy::ecs::system::Res<GiSettings>,
  view: Option<bevy::ecs::system::Res<crate::brickmap::dda::DdaViewUniform>>,
  aux: Option<bevy::ecs::system::Res<crate::brickmap::dda::AuxTexCache>>,
  dirty: Option<bevy::ecs::system::Res<crate::brickmap::upload::BrickMapDirty>>,
  mut gi_ph: bevy::ecs::system::ResMut<GiPlaceholder>,
  mut gpu: bevy::ecs::system::ResMut<GiGpu>,
) {
  gpu.frame = gpu.frame.wrapping_add(1);

  // ---- 世界几何修订号（uniform `flags.z`）----
  // `skip_verify` 回答的是：「上帧 reservoir 里存的二次顶点键，在本帧是否仍然逐位成立」。
  // 成立的条件比「几何没变」更弱也更容易判：世界几何自**上次真正跑 `gi_main`** 起没变过。
  // 此时沿 `hist.dir` 重发的射线与当初写入 `hist.sk` 的那条**输入完全相同** —— 方向取自
  // reservoir 本身，原点由已校验逐位相等的主键唯一决定（同一体素同一面同一物体 ⇒ 同一
  // `p_voxel` / 同一逐体素法线 / 同一外推量）⇒ 光路求交结果必然逐位相同 ⇒ 那条**验证射线可以
  // 整条省掉**（`gi/screen.wesl` ①）。世界变过就照旧发那条射线验证。
  //
  // 前提（改动这里前先读）：`world_rev` 必须覆盖**一切可能改变 `world_raycast` 结果的输入**。
  // 今天覆盖 = 世界全量上传、palette 变化、以及任何脏盒（= 一切增量体素编辑）。运行时不存在别的
  // 几何变化源：物体变换只在建世界时设定，LOD / beam 只改遍历起点、不改最近命中。
  // 若将来加了「物体动画 / 运行时改变换」，必须让那条路径也自增 `world_rev`。
  let world_changed =
    dirty.as_ref().is_some_and(|d| d.full || d.palette_changed || !d.boxes.is_empty());
  if world_changed {
    gpu.world_rev = gpu.world_rev.wrapping_add(1);
  }
  let skip_verify = gpu.world_rev == gpu.world_rev_gi;

  // ---- uniform（字段与 WESL `GiUniform` 逐字段镜像）----
  let mut u = GiUniform::default();
  u.params = Vec4::new(
    if settings.sun_bounce { 1.0 } else { 0.0 },
    0.0,
    crate::consts::GI_GAIN,
    0.0,
  );
  u.misc = Vec4::new(if settings.enabled { 1.0 } else { 0.0 }, 0.0, 0.0, 0.0);
  u.flags = Vec4::new(
    0.0,
    settings.div() as f32,
    if skip_verify { 1.0 } else { 0.0 },
    // w = 降噪质量档位（0..=3）：`gi_ss_main` 按它取候选数与记忆窗（只有最高档不同）。
    settings.tier() as f32,
  );
  // 整数帧号走 u32 通道（`seq.x`）：`gpu.frame` 本就是 u32，不再经 `params.x` 的 f32 截断。
  u.seq = UVec4::new(gpu.frame, 0, 0, 0);
  // 上一帧相机矩阵（时域重投影）：写「上帧真正用过的那一份」，再把本帧存下来。
  u.prev_view_proj = gpu.prev_view_proj;
  *gpu.uniform.get_mut() = u;
  gpu.uniform.write_buffer(&device, &queue);
  // 只在真正会跑 `gi_main` 的帧更新「上一帧」⇒ 与上帧写 reservoir 时用的矩阵逐位一致
  // （GI 关掉一段时间再打开时，历史 reservoir 与 prev 矩阵都停留在最后一帧 GI，重投影仍自洽）。
  // 注意：**分辨率的任意取值都跑 GI** —— 这里只跟 `enabled` 走。
  let gi_runs = settings.enabled;
  if gi_runs && let Some(v) = view.as_ref() {
    gpu.prev_view_proj = v.view_proj;
    // 本帧的 reservoir 就是在当前 `world_rev` 下写出的 ⇒ 记下来，供下一帧判 `skip_verify`。
    gpu.world_rev_gi = gpu.world_rev;
  }

  // ---- BG4：uniform + reservoir 双缓冲（绑定号 0/20/21，必须显式给 entry）----
  use bevy::render::render_resource::{BindGroupEntry, BindingResource};
  let bg4_layout = pipeline_cache.get_bind_group_layout(&gi_bg4_layout());
  // 两块由 AuxTexCache 随 GI 分辨率一起创建；未就绪（首帧 / prepare 提前返回）时用 4 B 占位
  // （该帧不会派发 `gi_main`）。
  let (res_cur, res_prev) = match aux.as_ref().and_then(|a| a.gi_res_buffers()) {
    Some((a, b)) if gpu.res_flip => (b, a),
    Some((a, b)) => (a, b),
    None => {
      let p = gi_ph.res_buffer(&device);
      (p, p)
    }
  };
  let bg4 = device.create_bind_group(
    None,
    &bg4_layout,
    &[
      BindGroupEntry { binding: 0, resource: gpu.uniform.binding().expect("uniform 已写入") },
      BindGroupEntry { binding: 20, resource: res_cur.as_entire_binding() },
      BindGroupEntry { binding: 21, resource: res_prev.as_entire_binding() },
    ],
  );
  commands.insert_resource(GiBg4(bg4));
  // 换绑：本帧的写入目标成为下一帧的读源。
  if gi_runs {
    gpu.res_flip = !gpu.res_flip;
  }

  // ---- BG5：GI 的写入侧（绑定号 2/3/6）----
  // GI 视图来自 `crate::brickmap::dda::AuxTexCache`；未就绪时用 1×1 占位纹理（bind group 必须给全条目）。
  // 视图/ buffer 都先 clone 成句柄（`TextureView`/`Buffer` 都是 Arc 包装）⇒ 之后还能再借一次 `gi_ph`。
  let bg5_layout = pipeline_cache.get_bind_group_layout(&gi_bg5_layout());
  let (gi_view, gi_cov_view) = match aux.as_ref().and_then(|a| a.gi_write_views()) {
    Some((a, b)) => (a.clone(), b.clone()),
    None => {
      let (a, b) = gi_ph.views(&device);
      (a.clone(), b.clone())
    }
  };
  // binding 6 = 降噪导引（`gi_main` 写）；未就绪时用 4 B 占位（该帧不会派发 `gi_main`）。
  let guide = aux
    .as_ref()
    .and_then(|a| a.gi_guide_buffer())
    .cloned()
    .unwrap_or_else(|| gi_ph.res_buffer(&device).clone());
  let bg5 = device.create_bind_group(
    None,
    &bg5_layout,
    &[
      BindGroupEntry { binding: 2, resource: BindingResource::TextureView(&gi_view) },
      BindGroupEntry { binding: 3, resource: BindingResource::TextureView(&gi_cov_view) },
      BindGroupEntry { binding: 6, resource: guide.as_entire_binding() },
    ],
  );
  commands.insert_resource(GiBg5(bg5));
}
