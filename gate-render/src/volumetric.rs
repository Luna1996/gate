//! 体积散射（godray / 光轴）：`volumetric.wesl` 的驱动侧。
//!
//! 算法与成本模型写在 WESL 文件头，这里只说**落点与接线**：
//!   · `fog_main`（本模块的新 pipeline）：网格 = 渲染分辨率 ÷ `FogSettings.div`，每像素 1 条主射线
//!     （只为了 t_end）+ S 条阴影射线 ⇒ 写 `fog_out`（线性 HDR 散射辐亮度）与雾的降噪导引；
//!   · `fog_den_temporal`（本模块的新 pipeline，layout 只有 group(0)）：时域累积 + 写 φ；
//!   · **空间段直接复用 GI 的 `gi_denoise_atrous1/2/4/8/16` 入口与 pipeline**
//!     （`crate::gi::GiGpu::den_pipelines`，layout = group(0) 的 10/14/15/17/20）：
//!     把雾的导引 / 中间靶 / φ / 小配置绑到那几个号上即可 —— 不复制那份几百行的滤波实现，
//!     代价是**三处布局契约必须与 GI 对齐**（导引 word 布局、φ 的语义、历史 word 布局），
//!     三处都在 `volumetric.wesl` 里写明了。
//!   · `dda_main` 在**曝光之前**把雾（几何感知上采样后）加到自己那条射线的颜色上（group(6)）。
//!
//! 与 GI 的关系：**两条链完全独立**（各自的网格尺寸、资源、帧号、历史、enable 开关），
//! 唯一的共享是上面那条 atrous 入口与 `gi_den_same_plane` 判据（都是「同一份实现、多一个调用点」）。
//! 太阳会动（昼夜循环）⇒ 没有任何世界空间缓存；太阳**突变**时整块清雾的历史（见 `reset_history`）。

use bevy::render::render_resource::{BindGroupLayoutDescriptor, CachedComputePipelineId, ShaderType};
use glam::{Mat4, UVec2, UVec4, Vec4};

use crate::brickmap::dda::{DEN_ATROUS_CHAINS, DEN_ATROUS_ROUNDS, DDA_WORKGROUP_SIZE};
use crate::wesl_consts::gi_consts;

/// 雾的参数 uniform（WESL `bindings.wesl` 的 `FogUniform` 逐字段镜像，字节一致）。
#[repr(C)]
#[derive(Debug, Default, Clone, Copy, ShaderType)]
pub struct FogUniform {
  /// x = `σ_t`（消光，1/voxel）、y = 地面雾的高度衰减 `h`（1/voxel）、z = 各向异性 `g`、
  /// w = 天光环境散射系数。
  pub params: Vec4,
  /// x = 分辨率除数（整数 f32）、y = 太阳角径（rad；同时是阴影射线的锥角与太阳盘半径）、
  /// z = 阴影距离上限（voxel）、w = 每帧采样数 S。
  pub misc: Vec4,
  /// x = 光柱增益、y = 雾距（积分距离上限，voxel）、z = 太阳角径（rad，**与 `misc.y` 同值**：
  /// 一条给雾的阴影锥、一条给天空的太阳盘 —— 物理上是同一个量，只因为消费方在不同的 pass 才写两遍）、
  /// w = 日晕强度。
  pub misc2: Vec4,
  /// x = 帧号（抖动种子；yzw 恒 0）。
  pub seq: UVec4,
  /// 上一帧真正跑过 `fog_main` 的相机矩阵（时域重投影）。与 GI 同一条契约、但**各自独立**。
  pub prev_view_proj: Mat4,
}

/// 雾的档位（菜单「渲染/太阳」）：`enabled` → 是否派发整条链；`div`/`quality`/`shadow_dist` 是三组档位。
#[derive(bevy::ecs::resource::Resource, Clone, Copy, Debug, PartialEq)]
pub struct FogSettings {
  /// 总开关：关掉 ⇒ 整条链不派发，`dda_main` 的 group(6) 绑 1×1 零纹理（恒 +0，零成本）。
  pub enabled: bool,
  /// 分辨率除数：1 = 全分辨率、2 = 半分辨率、4 = 四分之一（雾网格边长 = 渲染分辨率 ÷ 本值）。
  /// 代价按**雾像素数**涨（每像素 1 条主射线 + S 条阴影射线）⇒ 1/2 是 1/4 的时间、1/4 再降 4 倍。
  /// 雾是低频场（光柱的尺度 ≫ 一个像素）⇒ 1/2 档 + 几何感知上采样几乎看不出差别。
  pub div: u32,
  /// 质量档（0 低 / 1 中（默认）/ 2 高 / 3 极高）：决定**每帧采样数**与**空间段**。
  /// | 档 | 每帧采样 S | 空间段 |
  /// |---|---|---|
  /// | 0 低 | 1 | 不跑（只有时域累积 + 上采样） |
  /// | 1 中 | 1 | 时域 + 5 轮 3×3 atrous |
  /// | 2 高 | 2 | 时域 + 5 轮 5×5 atrous |
  /// | 3 极高 | 4 | 同 2（射线翻倍 = 全链最贵的一项） |
  /// 档 0 用于量"原始噪声"，档 3 用于量上限画质。
  pub quality: u32,
  /// 雾的阴影射线的距离上限（voxel）：**这个效果唯一的成本闸门**（射线越短越便宜）。
  /// 超过这个距离的遮挡不再影响该采样点（远处的阴影本来对这段雾贡献极小）。
  pub shadow_dist: f32,
  /// 消光系数 `σ_t`（1/voxel）：`1/σ_t` = 一个 e-folding 的距离。
  /// 菜单滑杆直接给这个物理量（区间见 `assets/ui/debug_menu.toml`）。
  /// **不要为了"更明显"把它调大**：调大会把整个世界（含天空与远景）一起雾掉 —— 想让光柱更明显
  /// 用 [`FogSettings::beam_gain`]，想控制雾的"厚度"才用它。
  pub density: f32,
  /// Henyey-Greenstein 各向异性 `g`：正值 = 前向散射（**朝着太阳看最亮**，光柱的观感来源）。
  pub anisotropy: f32,
  /// 天光环境散射系数（乘 `sky_color`）：雾自身的"亮度地板"。
  /// **默认接近 0**：参考观感是「阳光照到的那段雾亮、其余地方黑」，环境项调大就会变成"起雾"
  /// （各向同性的乳白色蒙在整幅画面上）。
  pub ambient: f32,
  /// 太阳的**角径**（rad）：它同时是两件事、物理上就是同一个量 ——
  /// ① 阴影射线的锥角（一条射线 = 太阳盘上的一个样本 ⇒ 多帧累积后光柱边缘变软）；
  /// ② 天空里那颗**太阳盘**的半径（见 `volumetric.wesl::sky_primary`）。
  /// 0 = 硬边光柱 + 一个点状太阳；WESL 侧按 `FOG_CONE_MAX` 硬夹（≤3.4°）。
  pub sun_cone: f32,
  /// **光柱增益**：乘在太阳的散射项上。本实现只有单次散射 + 各向同性环境近似，
  /// 真实光柱里多重散射的份额可观（阳光被反复散射回视线）⇒ 丢掉的那部分能量在这里补。
  /// 它是"光柱有多亮"的主旋钮（与"雾有多厚"的 `density` 正交）。
  pub beam_gain: f32,
  /// **雾距**（voxel）：积分的距离上限。**没有它就会变成起雾** ——
  /// 一条射向天空的射线会一路积到 `e^{-σt·∞} = 0`，天空与远景被雾色 100% 替换；
  /// 截断之后透射率不再下降 ⇒ 远处仍是它自己的颜色，散射只来自雾距内的那一段。
  /// ⚠️ 本工程的场景尺度是 **≈700~800 voxel**（相机到注视点）⇒ 默认必须大于它（2048），
  /// 否则默认视角下光柱会整条落在雾距之外（看上去"没有雾"）。
  pub fog_dist: f32,
  /// **日晕强度**：太阳盘外那圈解析拖尾的系数（`sky_primary` 的 halo）。
  pub halo: f32,
  /// **地面雾的高度衰减** `h`（1/voxel）：密度沿视线 `exp(−h·dir.y·t)`，**相机高度处 ρ = 1**。
  /// `0` = 均匀介质（室外长视线会把 `1−e^{−σt·d}` 积到 1 ⇒ 整幅画面一层薄纱 = **雾霾天**）。
  /// 参考观感要求「抬头看天干净、只有近地那层空气有光柱」⇒ 这个旋钮是"利落"的开关，别设 0。
  /// 取值直觉：`0.01` ≈ 每 70 voxel 密度减半（默认）；`0.02` ≈ 每 35 voxel 减半（很贴地）。
  pub ground_fog: f32,
}

/// 「质量」档的派发计划（`(是否跑空间段, atrous 轮数, 核半径)`）。
/// 核半径的权威值在 `gi/consts.wesl`（复用 GI 的 atrous 入口 ⇒ 同样两个档）；档 0 = 一轮都不跑
/// （`dda_main` 直接采样时域累积的输出）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FogPlan {
  pub on: bool,
  pub rounds: u32,
  pub radius: u32,
}

impl FogSettings {
  /// 菜单「渲染/太阳/雾分辨率」的三个档位（**下标 = 选中序号**）。
  pub const DIV_CHOICES: [u32; 3] = [1, 2, 4];
  /// 菜单「渲染/太阳/雾质量」的档位数（0 = 低 ..= 3 = 极高）；**下标 = 档位本身**。
  pub const QUALITY_TIERS: u32 = 4;
  /// 菜单「渲染/太阳/阴影距离」的四个档位（voxel）；**下标 = 选中序号**。
  pub const SHADOW_CHOICES: [f32; 4] = [64.0, 128.0, 256.0, 512.0];

  /// 生效的分辨率除数（越界钳回来）。
  pub fn div(&self) -> u32 {
    self.div.clamp(1, 4)
  }

  /// 生效的质量档（越界钳回来）。
  pub fn tier(&self) -> u32 {
    self.quality.min(Self::QUALITY_TIERS - 1)
  }

  /// 每帧每像素的采样数 `S`（档位表见 [`FogSettings::quality`]）。
  /// **上限必须 ≤ WESL `FOG_SAMPLE_MAX`**（uniform 里还会被硬夹一次）。
  pub fn samples(&self) -> u32 {
    match self.tier() {
      0 | 1 => 1,
      2 => 2,
      _ => 4,
    }
  }

  /// 生效的阴影距离（下限 1 voxel = 不让它退化成"永远可见"）。
  pub fn shadow_dist(&self) -> f32 {
    self.shadow_dist.clamp(1.0, 4096.0)
  }

  /// 生效的雾距（下限 1 voxel）。
  pub fn fog_dist(&self) -> f32 {
    self.fog_dist.clamp(1.0, 8192.0)
  }

  /// 本档的派发计划（见 [`FogPlan`]）。
  pub fn plan(&self) -> FogPlan {
    let c = gi_consts();
    match self.tier() {
      // 低：只有时域累积 + 几何感知上采样（用来量原始噪声与上限帧率）。
      0 => FogPlan { on: false, rounds: 0, radius: c.gi_den_atrous_r_fast },
      // 中（默认）：时域 + 5 轮 3×3（8 tap）atrous。
      1 => FogPlan { on: true, rounds: c.gi_den_atrous_iter, radius: c.gi_den_atrous_r_fast },
      // 高/极高：atrous 换 5×5（24 tap）；极高的射线数翻两番（在 `samples()` 里）。
      _ => FogPlan { on: true, rounds: c.gi_den_atrous_iter, radius: c.gi_den_atrous_r },
    }
  }

  /// 雾网格尺寸 = 渲染分辨率 ÷ `div()`（逐轴向下取整，至少 1×1）。
  pub fn fog_size(&self, render_size: UVec2) -> UVec2 {
    let d = self.div();
    UVec2::new((render_size.x / d).max(1), (render_size.y / d).max(1))
  }
}

impl Default for FogSettings {
  /// 默认按「暗环境 + 清晰光柱」的参考观感给（对齐参考图：背景压黑、只有阳光照到的那段雾发亮、
  /// 远处与天空不被雾吃掉）：
  /// 开、**1/4 分辨率**（每帧 ≈ `雾像素数 × (1 主射线 + 1 阴影射线)`，1/4 档在 720p 下 ≈ `0.35 ms`；
  /// 光柱尺度远大于一个像素，1/4 档 + 几何感知上采样足够）、中档（S=1 + 时域 + 3×3 atrous）、
  /// 阴影距离 128、σ_t = 0.005、**地面雾 0.01**（≈每 70 voxel 密度减半：天空迅速干净、近地保留光柱）、
  /// 雾距 2048（场景尺度 ≈700~800 voxel ⇒ 必须覆盖整个场景）、光柱增益 8、
  /// 环境散射 0.02（近似关掉）、g = 0.55、太阳角径 0.9°、日晕 1.2。
  fn default() -> Self {
    Self {
      enabled: true,
      div: 4,
      quality: 1,
      shadow_dist: 128.0,
      density: 0.005,
      anisotropy: 0.55,
      ambient: 0.02,
      sun_cone: 0.0157,
      beam_gain: 8.0,
      fog_dist: 2048.0,
      halo: 1.2,
      ground_fog: 0.01,
    }
  }
}

// ============================================================================
// bind group layout
// ============================================================================

/// group(4) 的**写侧**（只给 `fog_main` 的 pipeline）：uniform(1) + 雾输出(2) + 导引(3)。
/// 绑定号 1/2/3 是 group(4) 空着的号（0 = `gi_u`、20/21 = reservoir）⇒ 不新开 group，
/// 沿用「binding 号在同一个模块里唯一」的约定（见 `bindings.wesl`）。
pub fn fog_write_layout() -> BindGroupLayoutDescriptor {
  use bevy::render::render_resource::*;
  const C: ShaderStages = ShaderStages::COMPUTE;
  BindGroupLayoutDescriptor::new(
    "FogWrite",
    &[
      BindGroupLayoutEntry {
        binding: 1,
        visibility: C,
        ty: BindingType::Buffer {
          ty: BufferBindingType::Uniform,
          has_dynamic_offset: false,
          min_binding_size: Some(FogUniform::min_size()),
        },
        count: None,
      },
      BindGroupLayoutEntry {
        binding: 2,
        visibility: C,
        ty: BindingType::StorageTexture {
          access: StorageTextureAccess::WriteOnly,
          format: TextureFormat::Rgba16Float,
          view_dimension: TextureViewDimension::D2,
        },
        count: None,
      },
      // 导引：同一份 shader 模块里这个 var 在别处（`fog_den_temporal` 的
      // @group(0) @binding(21)、`dda_main` 的 @group(6) @binding(3)）以 read 出现，
      // layout 的访问模式必须与**声明**一致（另两处是 read）⇒ 这一处写侧取 read_write。
      BindGroupLayoutEntry {
        binding: 3,
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

/// group(6) 的**读侧**（只给 `dda_main` 的 pipeline）：雾的参数(0) + 最终结果(1, 采样) + 导引(3, 只读)。
/// 参数 uniform 必须给：`dda_main` 的天空背景要走 `sky_primary`（太阳盘 + 日晕），
/// 而那条路径只有 group(6)（与 `@group(4) @binding(1)` 是同一块 buffer，见 `bindings.wesl`）。
/// 关掉雾时绑 1×1 的零纹理与 16 字节的零 buffer（uniform 照绑：太阳盘与雾的开关无关）。
pub fn fog_read_layout() -> BindGroupLayoutDescriptor {
  use bevy::render::render_resource::*;
  const C: ShaderStages = ShaderStages::COMPUTE;
  BindGroupLayoutDescriptor::new(
    "FogRead",
    &[
      BindGroupLayoutEntry {
        binding: 0,
        visibility: C,
        ty: BindingType::Buffer {
          ty: BufferBindingType::Uniform,
          has_dynamic_offset: false,
          min_binding_size: Some(FogUniform::min_size()),
        },
        count: None,
      },
      BindGroupLayoutEntry {
        binding: 1,
        visibility: C,
        ty: BindingType::Texture {
          sample_type: TextureSampleType::Float { filterable: true },
          view_dimension: TextureViewDimension::D2,
          multisampled: false,
        },
        count: None,
      },
      BindGroupLayoutEntry {
        binding: 3,
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

/// **时域累积**的 group(0)（binding 21..26，见 `bindings.wesl`）。
/// 只有 group(0) 一份是刻意的（与 GI 降噪同一套理由：不拉进 brickmap / 相机矩阵）。
pub fn fog_temporal_layout() -> BindGroupLayoutDescriptor {
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
  BindGroupLayoutDescriptor::new(
    "FogTemporal",
    &[
      ro(21), // 降噪导引（读侧；写侧在 group(4) binding 3）
      BindGroupLayoutEntry {
        binding: 22, // 本帧原始雾（fog_out 的采样视图）
        visibility: C,
        ty: BindingType::Texture {
          sample_type: TextureSampleType::Float { filterable: true },
          view_dimension: TextureViewDimension::D2,
          multisampled: false,
        },
        count: None,
      },
      ro(23), // 历史（上帧）
      rw(24), // 历史（本帧）
      BindGroupLayoutEntry {
        binding: 25, // 时域输出（atrous 第一轮的输入）
        visibility: C,
        ty: BindingType::StorageTexture {
          access: StorageTextureAccess::WriteOnly,
          format: TextureFormat::Rgba16Float,
          view_dimension: TextureViewDimension::D2,
        },
        count: None,
      },
      rw(26), // φ（供复用的 `gi_den_atrous` 读 group(0) binding 17）
    ],
  )
}

// ============================================================================
// GPU 状态与资源
// ============================================================================

/// 雾的全部 GPU 状态：uniform + 自己的资源（随分辨率重建）+ bind group + pipeline。
/// 资源**不放进** `AuxTexCache`（GI 的缓存）：雾的尺寸、生命周期、开关都与 GI 无关，
/// 混在一处会让两条链互相牵制；`dda_main` 侧只需要一个 group(6) 的 bind group（本模块给出）。
#[derive(bevy::ecs::resource::Resource, Default)]
pub struct FogGpu {
  pub uniform: bevy::render::render_resource::UniformBuffer<FogUniform>,
  /// 只在实际跑 `fog_main` 的帧自增（抖动种子；停发期间不动 ⇒ 重新打开时历史对不上，
  /// 由 `reset_history` 处理）。
  pub frame: u32,
  /// 上一帧跑过 `fog_main` 的相机矩阵（uniform 的来源）。
  pub prev_view_proj: Mat4,
  /// 上一帧的太阳方向（判定"太阳突变"⇒ 整块清历史，见 [`FogGpu::history_clear`]）。
  prev_sun: glam::Vec3,
  /// **本帧**是否要清雾的历史（太阳突变 / 雾刚被打开）：由 `prepare_fog` 每帧先清零、再按需置 1，
  /// 由同一帧稍后跑的 `dispatch_fog` 消费（只读）。**必须每帧清零** ——
  /// 否则一次突变之后每帧都清历史，时域累积等于被永久关掉（画面只剩当前帧的抖动）。
  history_clear: bool,
  /// 本帧雾是否在跑（`prepare_fog` 用它与上一帧对比 ⇒ 判定"刚打开"）。
  ran_last_frame: bool,
  /// 上一帧生效的 `render_size / div`（重建判定）。
  size: UVec2,
  /// 资源尺寸（= `FogSettings::fog_size`，与 `size` 一起变化）。
  tex: Option<bevy::render::render_resource::Texture>,
  tex_src: Option<bevy::render::render_resource::TextureView>,
  tex_dst: Option<bevy::render::render_resource::TextureView>,
  guide: Option<bevy::render::render_resource::Buffer>,
  hist: [Option<bevy::render::render_resource::Buffer>; 2],
  phi: Option<bevy::render::render_resource::Buffer>,
  /// 空间段的 4 张中间靶：`[0]` = 时域输出、`[1]`/`[2]` = ping-pong、`[3]` = 最终结果。
  dn: [Option<bevy::render::render_resource::Texture>; 4],
  dn_src: [Option<bevy::render::render_resource::TextureView>; 4],
  dn_dst: [Option<bevy::render::render_resource::TextureView>; 4],
  /// 空间段的小配置（`@group(0) @binding(20)`，1 个 word = atrous 核半径）。**雾自己一块**：
  /// 与 GI 共用会让两档互相覆盖。
  cfg: Option<bevy::render::render_resource::Buffer>,
  cfg_r: u32,
  /// 历史双缓冲的换绑状态（每次真正跑时域累积翻转一次）。
  hist_flip: bool,
  write_bg: Option<bevy::render::render_resource::BindGroup>,
  /// `dda_main` 的 group(6)（**恒有**：关掉雾时是占位版）。
  read_bg: Option<bevy::render::render_resource::BindGroup>,
  tmp_bg: Option<bevy::render::render_resource::BindGroup>,
  /// atrous 的 6 种 src→dst 组合（下标 +1 = `DEN_ATROUS_CHAINS` 的下标）。
  atrous_bg: [Option<bevy::render::render_resource::BindGroup>; 6],
  /// 占位（关掉雾时 read_bg 用）：1×1 零纹理 + 16 字节零 buffer。
  ph_tex: Option<bevy::render::render_resource::Texture>,
  ph_view: Option<bevy::render::render_resource::TextureView>,
  ph_buf: Option<bevy::render::render_resource::Buffer>,
  /// `[0]` = `fog_main`、`[1]` = `fog_den_temporal`（空间段复用 GI 的 pipeline，见文件头）。
  pipelines: [Option<CachedComputePipelineId>; 2],
}

/// 占位资源（关掉雾 / 首帧）：1×1 rgba16f 零纹理 + 16 字节零 buffer。
fn placeholder(
  device: &bevy::render::renderer::RenderDevice,
  gpu: &mut FogGpu,
) -> (bevy::render::render_resource::TextureView, bevy::render::render_resource::Buffer) {
  use bevy::render::render_resource::*;
  if gpu.ph_tex.is_none() {
    let t = device.create_texture(&TextureDescriptor {
      label: Some("gate_fog_placeholder"),
      size: Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
      mip_level_count: 1,
      sample_count: 1,
      dimension: TextureDimension::D2,
      format: TextureFormat::Rgba16Float,
      usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
      view_formats: &[],
    });
    gpu.ph_view = Some(t.create_view(&TextureViewDescriptor::default()));
    gpu.ph_tex = Some(t);
  }
  if gpu.ph_buf.is_none() {
    gpu.ph_buf = Some(device.create_buffer(&BufferDescriptor {
      label: Some("gate_fog_guide_placeholder"),
      size: 16,
      usage: BufferUsages::STORAGE,
      mapped_at_creation: false,
    }));
  }
  (
    gpu.ph_view.as_ref().expect("刚创建").clone(),
    gpu.ph_buf.as_ref().expect("刚创建").clone(),
  )
}

pub struct FogPlugin;

impl bevy::app::Plugin for FogPlugin {
  fn build(&self, app: &mut bevy::app::App) {
    use bevy::ecs::schedule::IntoScheduleConfigs;
    app.init_resource::<FogSettings>();
    let Some(render_app) = app.get_sub_app_mut(bevy::render::RenderApp) else {
      return;
    };
    render_app
      .init_resource::<FogSettings>()
      .init_resource::<FogGpu>()
      .add_systems(bevy::render::RenderStartup, init_fog_gpu)
      .add_systems(
        bevy::render::RenderStartup,
        queue_fog_pipelines.after(crate::brickmap::dda::init_dda_pipelines),
      )
      .add_systems(bevy::render::ExtractSchedule, extract_fog_settings)
      .add_systems(
        bevy::render::Render,
        prepare_fog.in_set(bevy::render::RenderSystems::PrepareBindGroups),
      );
  }
}

fn init_fog_gpu(mut commands: bevy::ecs::system::Commands) {
  let c = gi_consts();
  let d = FogSettings::default();
  bevy::log::info!(
    target: "gate",
    "体积散射（godray）：默认 {} / 1÷{} 分辨率 / 每帧每像素 {} 条采样射线（阴影距离 {} voxel）；\
     σt {}（e-folding ≈{} voxel）/ 地面雾 {}（每 {} voxel 减半）/ 雾距 {} / 光柱增益 {} / 环境散射 {} / g {} / 太阳角径 {}°；\
     空间段复用 GI 的 atrous 入口（{} 轮，核半径 质量档 {} / 快速档 {}）",
    if d.enabled { "开" } else { "关" },
    d.div(),
    d.samples(),
    d.shadow_dist(),
    d.density,
    if d.density > 0.0 { format!("{:.0}", 1.0 / d.density) } else { "∞".to_string() },
    d.ground_fog,
    if d.ground_fog > 0.0 { format!("{:.0}", 0.693 / d.ground_fog) } else { "∞".to_string() },
    d.fog_dist(),
    d.beam_gain,
    d.ambient,
    d.anisotropy,
    d.sun_cone.to_degrees(),
    c.gi_den_atrous_iter,
    c.gi_den_atrous_r,
    c.gi_den_atrous_r_fast,
  );
  commands.insert_resource(FogGpu::default());
}

fn queue_fog_pipelines(
  pipeline_cache: bevy::ecs::system::Res<bevy::render::render_resource::PipelineCache>,
  dda_shader: bevy::ecs::system::Res<crate::shader::DdaShaderHandle>,
  dda: bevy::ecs::system::Res<crate::brickmap::dda::DdaPipelines>,
  mut gpu: bevy::ecs::system::ResMut<FogGpu>,
) {
  use bevy::render::render_resource::ComputePipelineDescriptor;
  use std::borrow::Cow;
  if gpu.pipelines[0].is_some() {
    return;
  }
  // `fog_main`：要 brickmap（求交）+ grid descs + 光池 + 雾自己那一组；group(0) 复用 GI 的瘦版
  // （view uniform + beam depth —— 两者要的东西逐字相同，见 `dispatch_fog` 里绑 `aux.gi_bg0`）。
  let write_layouts = vec![
    dda.bg0_gi_layout.clone(),
    dda.bg1_layout.clone(),
    dda.bg2_layout.clone(),
    dda.bg3_layout.clone(),
    fog_write_layout(),
  ];
  gpu.pipelines[0] = Some(pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
    label: Some(Cow::from("gate_fog")),
    layout: write_layouts,
    shader: dda_shader.0.clone(),
    entry_point: Some(Cow::from("fog_main")),
    ..Default::default()
  }));
  gpu.pipelines[1] = Some(pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
    label: Some(Cow::from("gate_fog_denoise_temporal")),
    layout: vec![fog_temporal_layout()],
    shader: dda_shader.0.clone(),
    entry_point: Some(Cow::from("fog_den_temporal")),
    ..Default::default()
  }));
}

fn extract_fog_settings(
  mut commands: bevy::ecs::system::Commands,
  settings: Option<bevy::render::Extract<bevy::ecs::system::Res<FogSettings>>>,
) {
  commands.insert_resource(settings.map_or_else(FogSettings::default, |s| **s));
}

/// 每帧：写 uniform、必要时重建资源、建 bind group。
#[allow(clippy::too_many_arguments)] // Bevy render system：各资源逐一注入（与 `prepare_gi` 同一形态）
fn prepare_fog(
  mut commands: bevy::ecs::system::Commands,
  device: bevy::ecs::system::Res<bevy::render::renderer::RenderDevice>,
  queue: bevy::ecs::system::Res<bevy::render::renderer::RenderQueue>,
  pipeline_cache: bevy::ecs::system::Res<bevy::render::render_resource::PipelineCache>,
  settings: bevy::ecs::system::Res<FogSettings>,
  view: Option<bevy::ecs::system::Res<crate::brickmap::dda::DdaViewUniform>>,
  scale: Option<bevy::ecs::system::Res<crate::brickmap::dda::RenderScale>>,
  lighting: Option<bevy::ecs::system::Res<crate::lighting::LightingTheme>>,
  mut gpu: bevy::ecs::system::ResMut<FogGpu>,
) {
  use bevy::render::render_resource::*;
  let c = gi_consts();
  let plan = settings.plan();
  // 本帧是否**真的跑**雾（开关 + σt > 0）。两件事都以它为准：
  //   ① 帧号 / 上一帧矩阵 / 历史换绑只在真正跑的帧推进（与 GI 同一套语义）；
  //   ② group(6) 绑**谁** —— 不跑的时候必须绑占位零纹理，否则关掉之后那张纹理还留着最后一帧的雾，
  //      主 pass 会继续把它加进画面（「关不掉的雾」）。σt = 0 与开关同一条路径（两者都是"没有雾"）。
  let active = settings.enabled && settings.density > 0.0;

  // ---- 太阳突变判定（昼夜循环是缓慢的 ⇒ 不触发；菜单改方向是突变的 ⇒ 清历史，不拖影）----
  // 脉冲**每帧先清零**（消费方是同一帧稍后的 `dispatch_fog`）。
  gpu.history_clear = false;
  let sun = lighting
    .as_ref()
    .and_then(|l| l.sun.as_ref())
    .map(|s| glam::Vec3::from_array(s.dir).normalize_or_zero())
    .unwrap_or_default();
  let sun_jump = gpu.ran_last_frame && gpu.prev_sun != glam::Vec3::ZERO && sun != glam::Vec3::ZERO
    && gpu.prev_sun.dot(sun) < 0.9999;
  let just_enabled = active && !gpu.ran_last_frame;
  if sun_jump || just_enabled {
    gpu.history_clear = true;
    if sun_jump {
      bevy::log::info!(target: "gate", "太阳方向突变 ⇒ 清体积散射的历史（避免旧光照拖影）");
    }
  }
  gpu.prev_sun = sun;
  gpu.ran_last_frame = active;

  // ---- 帧号 / 上一帧矩阵（只在真正会跑 `fog_main` 的帧推进，与 GI 同一套语义）----
  if active {
    gpu.frame = gpu.frame.wrapping_add(1);
    if let Some(v) = view.as_ref() {
      gpu.prev_view_proj = v.view_proj;
    }
  }

  // ---- uniform ----
  let size = scale
    .as_deref()
    .map_or(crate::consts::VIEW_SIZE, |s| s.size);
  *gpu.uniform.get_mut() = FogUniform {
    params: Vec4::new(
      settings.density.max(0.0),
      settings.ground_fog.max(0.0),
      settings.anisotropy.clamp(-0.9, 0.9),
      settings.ambient.max(0.0),
    ),
    misc: Vec4::new(
      settings.div() as f32,
      settings.sun_cone.max(0.0),
      settings.shadow_dist(),
      settings.samples() as f32,
    ),
    // z 与 `misc.y` 同值（太阳角径）：一个给雾的阴影锥、一个给天空的太阳盘（见 `FogUniform::misc2`）。
    misc2: Vec4::new(
      settings.beam_gain.max(0.0),
      settings.fog_dist(),
      settings.sun_cone.max(0.0),
      settings.halo.max(0.0),
    ),
    seq: UVec4::new(gpu.frame, 0, 0, 0),
    prev_view_proj: gpu.prev_view_proj,
  };
  gpu.uniform.write_buffer(&device, &queue);

  // ---- 资源（随 分辨率 ÷ div 重建；wgpu 新建 buffer/纹理恒为零 ⇒ 历史 M = 0 = 无历史）----
  let fog_size = settings.fog_size(size);
  if gpu.tex.is_none() || gpu.size != fog_size {
    let px = fog_size.x as u64 * fog_size.y as u64;
    let make_tex = |label: &str| {
      device.create_texture(&TextureDescriptor {
        label: Some(label),
        size: Extent3d { width: fog_size.x, height: fog_size.y, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba16Float,
        // 既要被 `fog_main` / 时域 / atrous 写（storage），又要被采样（上采样 / atrous 的输入）
        usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
      })
    };
    let make_buf = |label: &str, bytes: u64| {
      device.create_buffer(&BufferDescriptor {
        label: Some(label),
        size: bytes.max(4),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
      })
    };
    let tex = make_tex("gate_fog");
    gpu.tex_src = Some(tex.create_view(&TextureViewDescriptor::default()));
    gpu.tex_dst = Some(tex.create_view(&TextureViewDescriptor::default()));
    gpu.tex = Some(tex);
    gpu.guide =
      Some(make_buf("gate_fog_guide", px * c.gi_den_guide_words as u64 * 4));
    gpu.hist = [
      Some(make_buf("gate_fog_hist_a", px * FOG_DEN_HIST_WORDS as u64 * 4)),
      Some(make_buf("gate_fog_hist_b", px * FOG_DEN_HIST_WORDS as u64 * 4)),
    ];
    gpu.phi = Some(make_buf("gate_fog_phi", px * 4));
    for (i, label) in ["gate_fog_tmp", "gate_fog_dn_a", "gate_fog_dn_b", "gate_fog_den"]
      .iter()
      .enumerate()
    {
      let t = make_tex(label);
      gpu.dn_src[i] = Some(t.create_view(&TextureViewDescriptor::default()));
      gpu.dn_dst[i] = Some(t.create_view(&TextureViewDescriptor::default()));
      gpu.dn[i] = Some(t);
    }
    gpu.size = fog_size;
    bevy::log::info!(
      target: "gate",
      "体积散射资源 → {}×{}（渲染 {}÷{}）；导引 {} word/像素、历史 {} word/像素 ×2",
      fog_size.x,
      fog_size.y,
      size.x,
      settings.div(),
      c.gi_den_guide_words,
      FOG_DEN_HIST_WORDS,
    );
  }
  if gpu.cfg.is_none() {
    gpu.cfg = Some(device.create_buffer(&BufferDescriptor {
      label: Some("gate_fog_den_cfg"),
      size: 16,
      usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
      mapped_at_creation: false,
    }));
  }
  if gpu.cfg_r != plan.radius {
    gpu.cfg_r = plan.radius;
    queue.write_buffer(
      gpu.cfg.as_ref().expect("刚创建"),
      0,
      &plan.radius.to_le_bytes(),
    );
    bevy::log::info!(
      target: "gate",
      "体积散射空间段 → {}（atrous 核半径 {}，每轮 {} 个 tap；每帧采样 {} 条）",
      if plan.on { format!("{} 轮 atrous", plan.rounds) } else { "关（只有时域 + 上采样）".into() },
      plan.radius,
      (2 * plan.radius + 1) * (2 * plan.radius + 1) - 1,
      settings.samples(),
    );
  }

  // ---- bind group（每帧重建；句柄都是 Arc，重建只是换绑）----
  let write_layout = pipeline_cache.get_bind_group_layout(&fog_write_layout());
  let read_layout = pipeline_cache.get_bind_group_layout(&fog_read_layout());
  let temporal_layout = pipeline_cache.get_bind_group_layout(&fog_temporal_layout());
  let atrous_layout =
    pipeline_cache.get_bind_group_layout(&crate::gi::gi_den_atrous_layout());
  let (ph_view, ph_buf) = placeholder(&device, &mut gpu);

  let guide = gpu.guide.clone();
  let tex_src = gpu.tex_src.clone();
  let tex_dst = gpu.tex_dst.clone();
  let phi = gpu.phi.clone();
  let cfg = gpu.cfg.clone();
  let dn_src: Vec<TextureView> = gpu.dn_src.iter().flatten().cloned().collect();
  let dn_dst: Vec<TextureView> = gpu.dn_dst.iter().flatten().cloned().collect();
  let hist_prev = gpu.hist[gpu.hist_flip as usize].clone();
  let hist_cur = gpu.hist[(!gpu.hist_flip) as usize].clone();

  if let (Some(guide), Some(tex_src), Some(tex_dst), Some(phi), Some(cfg), Some(hist_prev), Some(hist_cur)) =
    (guide, tex_src, tex_dst, phi, cfg, hist_prev, hist_cur)
    && dn_src.len() == 4
    && dn_dst.len() == 4
  {
    let write_bg = device.create_bind_group(
      None,
      &write_layout,
      &[
        BindGroupEntry { binding: 1, resource: gpu.uniform.binding().expect("uniform 已写入") },
        BindGroupEntry { binding: 2, resource: BindingResource::TextureView(&tex_dst) },
        BindGroupEntry { binding: 3, resource: guide.as_entire_binding() },
      ],
    );
    // group(6)：`dda_main` 采样的是**降噪后**的那张（质量低 = 不跑空间段时是原始 `fog_out`，
    // 语义一致：rgb = 线性散射辐亮度、a 恒 1）；**没在跑雾时（开关关 / σt = 0）绑占位零纹理** ——
    // 否则那一项会一直用最后一帧的雾（「关不掉的雾」）。
    let final_view = if !active {
      ph_view.clone()
    } else if plan.on {
      dn_src[3].clone()
    } else {
      tex_src.clone()
    };
    let read_bg = device.create_bind_group(
      None,
      &read_layout,
      &[
        // binding 0 = 雾的参数（`dda_main` 的天空要用它取太阳盘/日晕的参数）。
        BindGroupEntry { binding: 0, resource: gpu.uniform.binding().expect("uniform 已写入") },
        BindGroupEntry { binding: 1, resource: BindingResource::TextureView(&final_view) },
        BindGroupEntry { binding: 3, resource: guide.as_entire_binding() },
      ],
    );
    let tmp_bg = device.create_bind_group(
      None,
      &temporal_layout,
      &[
        BindGroupEntry { binding: 21, resource: guide.as_entire_binding() },
        BindGroupEntry { binding: 22, resource: BindingResource::TextureView(&tex_src) },
        BindGroupEntry { binding: 23, resource: hist_prev.as_entire_binding() },
        BindGroupEntry { binding: 24, resource: hist_cur.as_entire_binding() },
        BindGroupEntry { binding: 25, resource: BindingResource::TextureView(&dn_dst[0]) },
        BindGroupEntry { binding: 26, resource: phi.as_entire_binding() },
      ],
    );
    let mut atrous_bg: [Option<BindGroup>; 6] = Default::default();
    for (k, [s, d]) in DEN_ATROUS_CHAINS.iter().enumerate() {
      atrous_bg[k] = Some(device.create_bind_group(
        None,
        &atrous_layout,
        &[
          BindGroupEntry { binding: 10, resource: guide.as_entire_binding() },
          BindGroupEntry { binding: 14, resource: BindingResource::TextureView(&dn_src[*s]) },
          BindGroupEntry { binding: 15, resource: BindingResource::TextureView(&dn_dst[*d]) },
          BindGroupEntry { binding: 17, resource: phi.as_entire_binding() },
          BindGroupEntry { binding: 20, resource: cfg.as_entire_binding() },
        ],
      ));
    }
    gpu.write_bg = Some(write_bg);
    gpu.read_bg = Some(read_bg);
    gpu.tmp_bg = Some(tmp_bg);
    gpu.atrous_bg = atrous_bg;
  } else {
    // 资源还没就绪（首帧）⇒ 雾那一项用占位（恒 +0）；**参数 uniform 照绑**（太阳盘要它）。
    let read_bg = device.create_bind_group(
      None,
      &read_layout,
      &[
        BindGroupEntry { binding: 0, resource: gpu.uniform.binding().expect("uniform 已写入") },
        BindGroupEntry { binding: 1, resource: BindingResource::TextureView(&ph_view) },
        BindGroupEntry { binding: 3, resource: ph_buf.as_entire_binding() },
      ],
    );
    gpu.read_bg = Some(read_bg);
    gpu.write_bg = None;
    gpu.tmp_bg = None;
  }
  // 换绑：本帧的写入目标成为下一帧的读源（只在真正跑时域累积时翻转）。
  if active && plan.on {
    gpu.hist_flip = !gpu.hist_flip;
  }
  commands.insert_resource(FogReadBg(gpu.read_bg.clone()));
}

/// `dda_main` 的 group(6)（**恒有**：雾关掉时是占位版）。
#[derive(bevy::ecs::resource::Resource)]
pub struct FogReadBg(pub Option<bevy::render::render_resource::BindGroup>);

/// 体积散射在 `dispatch_dda` 里的三件套，**打包成一个 `SystemParam`**：
/// Bevy 的系统函数最多 16 个参数，而主 pass 的 dispatch 正好卡在边界上（拆成三个就超了）。
#[derive(bevy::ecs::system::SystemParam)]
pub(crate) struct FogRes<'w> {
  pub settings: Option<bevy::ecs::system::Res<'w, FogSettings>>,
  pub gpu: Option<bevy::ecs::system::Res<'w, FogGpu>>,
  pub read_bg: Option<bevy::ecs::system::Res<'w, FogReadBg>>,
}

/// 雾的历史每像素字数（**必须与 WESL `FOG_DEN_HIST_WORDS` 一致**）。
/// 权威值在 `volumetric.wesl`；这里只为了让 Rust 侧开 buffer 时读得出来 ——
/// 与 `gi_consts()` 那套解析不同（那几个常量是跨模块契约，这一个只在雾内部用）。
const FOG_DEN_HIST_WORDS: u32 = 6;

/// 派发整条雾链（`fog_main` → 时域 → 复用 GI 的 atrous）—— 由 `brickmap::dda::dispatch_dda`
/// 在**主 pass 之前**调用（主 pass 要采样它）。
/// group(0) 直接复用 GI pass 的 `gi_bg0`（view uniform + beam depth，两者要的东西逐字相同）；
/// group(1..3) 复用主 pass 的绑定；group(4) = 雾的写侧。
#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatch_fog(
  profiler: &mut crate::profiler::GpuProfilerRes,
  encoder: &mut bevy::render::render_resource::CommandEncoder,
  gpu: Option<&FogGpu>,
  settings: Option<&FogSettings>,
  gi_gpu: Option<&crate::gi::GiGpu>,
  pipeline_cache: &bevy::render::render_resource::PipelineCache,
  bg0: Option<&bevy::render::render_resource::BindGroup>,
  bg1: Option<&bevy::render::render_resource::BindGroup>,
  bg2: Option<&bevy::render::render_resource::BindGroup>,
  bg3: Option<&bevy::render::render_resource::BindGroup>,
) {
  let (Some(gpu), Some(s)) = (gpu, settings) else { return };
  if !s.enabled || s.density <= 0.0 {
    return;
  }
  let (Some(bg0), Some(bg1), Some(bg2), Some(bg3)) = (bg0, bg1, bg2, bg3) else {
    return;
  };
  let Some(write_bg) = gpu.write_bg.as_ref() else {
    return;
  };
  let Some(pipe) = gpu.pipelines[0].and_then(|id| pipeline_cache.get_compute_pipeline(id)) else {
    return;
  };
  let plan = s.plan();
  let gx = gpu.size.x.div_ceil(DDA_WORKGROUP_SIZE);
  let gy = gpu.size.y.div_ceil(DDA_WORKGROUP_SIZE);
  if gx == 0 || gy == 0 {
    return;
  }
  // 太阳突变 / 刚打开 ⇒ 整块清历史（一次 `clear_buffer`，比重建资源便宜得多）。
  if gpu.history_clear {
    for h in gpu.hist.iter().flatten() {
      encoder.clear_buffer(h, 0, None);
    }
  }
  crate::profiler::gpu_compute_pass(profiler, encoder, "gate_fog", |pass| {
    pass.set_pipeline(pipe);
    pass.set_bind_group(0, bg0, &[]);
    pass.set_bind_group(1, bg1, &[]);
    pass.set_bind_group(2, bg2, &[]);
    pass.set_bind_group(3, bg3, &[]);
    pass.set_bind_group(4, write_bg, &[]);
    pass.dispatch_workgroups(gx, gy, 1);
  });
  if !plan.on {
    return;
  }
  // ---- 时域累积（本模块的入口）----
  if let Some(p) = gpu.pipelines[1].and_then(|id| pipeline_cache.get_compute_pipeline(id))
    && let Some(bg) = gpu.tmp_bg.as_ref()
  {
    crate::profiler::gpu_compute_pass(profiler, encoder, "gate_fog_denoise_temporal", |pass| {
      pass.set_pipeline(p);
      pass.set_bind_group(0, bg, &[]);
      pass.dispatch_workgroups(gx, gy, 1);
    });
  }
  // ---- 空间段：**复用 GI 的 atrous pipeline**（同一份入口、同一份 layout，只换 bind group）----
  let rounds = plan.rounds.clamp(0, 5) as usize;
  if rounds == 0 {
    return;
  }
  let Some(gi_gpu) = gi_gpu else { return };
  let chain = &DEN_ATROUS_ROUNDS[rounds - 1];
  for i in 0..rounds {
    let Some(bg) = gpu.atrous_bg[chain[i]].as_ref() else { continue };
    let Some(pi) =
      gi_gpu.den_pipelines[i + 1].and_then(|id| pipeline_cache.get_compute_pipeline(id))
    else {
      continue;
    };
    let label = [
      "gate_fog_denoise_atrous1",
      "gate_fog_denoise_atrous2",
      "gate_fog_denoise_atrous4",
      "gate_fog_denoise_atrous8",
      "gate_fog_denoise_atrous16",
    ][i];
    crate::profiler::gpu_compute_pass(profiler, encoder, label, |pass| {
      pass.set_pipeline(pi);
      pass.set_bind_group(0, bg, &[]);
      pass.dispatch_workgroups(gx, gy, 1);
    });
  }
}
