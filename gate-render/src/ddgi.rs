//! DDGI（Dynamic Diffuse Global Illumination）——阶段一：世界空间探针烘焙 + 活跃探针筛选。
//!
//! 架构（严格对齐 Douglas Devlog #23 / Majercik 2019, 2021）：
//! - **嵌套级联 LOD，以相机为中心**：4 级 LOD 各自以相机为中心铺 16³ 网格（cell 边长
//!   16/32/64/128 voxel），覆盖范围逐级 ×2 且严格嵌套（LOD(l-1) 盒 ⊂ LOD(l) 盒）。
//!   相机移动使某级 origin 按该级 cell 对齐滚动时，只重烘「世界 cell 发生变化」的槽位
//!   （`ddgi_cell_id` 增量缓存），未变的槽位续龄。
//! - **烘焙（`ddgi_bake0..3`）**：世界数据变化（上传修订号自增）时才跑一次。逐 cell 沿 4³ 分裂树
//!   **BFS 找「最大的全空叶」并把探针放在其中心**（同级优先靠 cell 中心；全满 cell 无探针；
//!   全空 cell 居中）——即 Douglas 的探针放置启发式。结果写入 `ddgi_cell` storage buffer。
//!   按 LOD 拆成 4 个独立 compute pass（细→粗）：粗级要继承**本帧**细级的放置结果
//!   （Douglas 的 "down sample the generated data"），而 pass 边界才是内存屏障。
//! - **活跃判定（`ddgi_sort`）**：每帧逐 cell，读烘焙记录（不再重算树 BFS）；探针存在且
//!   「本 cell 或 6 邻接 cell 有体素」（或与非网格对齐物体 AABB 重叠）→ 活跃 → atomicAdd 进
//!   per-LOD worklist；同时刷新 age / slot_pos / meta。
//! - **seal**：按 LOD 把固定射线预算摊给活跃探针，写 cast/collect indirect args。
//!
//! 阶段二（cast）/ 阶段三（collect/着色）尚未接入；`irr/depth` 纹理沿用旧布局暂作占位。

use bevy::render::render_resource::{CachedComputePipelineId, ShaderType};
use glam::{IVec3, IVec4, UVec3, UVec4, Vec4};

/// 辐照度图每探针 4×4（= 16 纹素）。
///
/// 【为什么不用原版的 8×8】曾对齐 DDGI 原版试过 8×8 / 16×16（见 DEPTH_TEXELS 注释）：
/// 实测**画质没有明显改善**，代价却很实在（显存 ×4、collect 线程 ×4、帧轮换周期 ×4）。
/// 方向分辨率降低带来的模糊由 DDGI_ALPHA 的时间累积与纹素间插值承担。
/// 必须与 WGSL `DDGI_IRR_TEXELS` 一致。
pub const IRRADIANCE_TEXELS: u32 = 4;
/// 深度图每探针 8×8（= 64 纹素）。
///
/// 【为什么不用原版的 16×16】曾试过 16×16（每纹素 ~11°，现在 ~22°）：**实测漏光没有明显
/// 改善** —— 说明当时的漏光主因不在深度角分辨率，而在别处（射线方向未绑定纹素 / 借针跨墙 /
/// 级联硬切，见 dda.wgsl 对应注释）。代价则是深度图集 64MB → 256MB、collect 线程 80 → 320、
/// 帧轮换周期 ×4。故回退到 8×8。
/// 必须与 WGSL `DDGI_DEPTH_TEXELS` 一致。
pub const DEPTH_TEXELS: u32 = 8;
pub const PROBE_T_MAX: f32 = 8192.0;
/// 每帧射线总预算。WGSL `ddgi_seal` 把它**均分**给全部活跃探针（rpp = 预算 / 活跃数，
/// 钳在 [1, 256]），所以这是一条直接线性的画质/帧时旋钮：减半 → cast/collect 的帧时也
/// 大致减半，代价是每探针样本减半、图集噪声变大。
///
/// 【历史】曾试过 262144 / 1048576 来压制"探针晶格亮斑"与深度闸门抖动，**帧时涨了但问题
/// 没解决**（根因是探针网格对贴缝尺度欠采样 + 深度的角度均值偏差，不是射线数量），已退回
/// 131072。若要再动这条线，请先备好可验证的收益。
/// 必须与 WGSL `DDGI_RAY_BUDGET` 保持一致。
pub const DDGI_RAY_BUDGET: u32 = 131072;

pub const DDGI_LODS: u32 = 4;
/// 最细 LOD 的 cell 边长（voxel）。
pub const DDGI_BASE_CELL: i32 = 16;
/// 4 级 LOD cell 边长（voxel）。**不再是等比数列** —— LOD3 特意放大以换取覆盖范围。
///
/// 上限受 `ddgi_cell_state_sized` 支持（16/32/64/128/256）约束。前三级 [16,32,64] 等比：
/// 探针数 / 射线预算 / 显存全不变（dims 不变），只是把同样的探针铺在**更小的体积**上 ——
/// 近场探针间距 64cm→32cm。这是"探针晶格"伪影（GI 场的空间变化比探针网格更细时，
/// 三线性插值把每个探针自己的值暴露成 0.64m 周期的亮斑）最直接的降压手段。
///
/// 【LOD3 为什么是 256 而非 128】最粗级承担**覆盖兜底**：覆盖 = dims × cell，而窗口
/// **跟随相机** → 相机拉远时场景滑出窗口 → 覆盖外只能吃常量兜底（fallback），表现成
/// "有 GI / 无 GI"割裂。cell 128→256 使覆盖 82m → 164m，而**成本≈0**（dims 没动 →
/// 槽位数 / 射线预算 / 图集 / collect 全不变），代价只是最粗级探针间距 2.56m → 5.12m。
/// 注意：cell 不再等比后**不能从 cs 反推 lod**（`ddgi_place_probe` 现在显式接收 lod）。
pub const DDGI_LOD_CELL_SIZES: [i32; DDGI_LODS as usize] = [16, 32, 64, 256];
/// 各级 LOD 的 cell 维度（4 级相同）。**dims 不变**，靠 cell 变大来扩大覆盖（前三级 ×2、
/// LOD3 ×4，见上），形成严格嵌套的级联。水平 32 格、垂直 16 格（体素世界水平视野远大于垂直）。
/// cell [16,32,64,256] 时各级覆盖范围 = dims×cell：
/// 512×256×512 / 1024×512×1024 / 2048×1024×2048 / 8192×4096×8192 voxel
/// = 10.2×5.1×10.2 / 20.5×10.2×20.5 / 41×20.5×41 / **164×82×164 m**（半宽到 ±82m）。
/// 前三级严格嵌套；LOD3 因 cell 放大而跨得更大，嵌套关系仍成立（更大即包含）。
/// 每级 16384 槽 → 共 65536 槽（= 占位纹理容量，全部槽位可采样）。
pub const DDGI_LOD_DIMS: UVec3 = UVec3::new(32, 16, 32);
// 槽位映射是**世界锚定**的：shader 里 `slot = slot_base + (世界 cell 号 mod dims)`（见
// `ddgi_slot`）。因此「槽位 ↔ 世界 cell」的身份与相机无关 —— 相机滚动只会让「新进入窗口的
// 那条带」换掉世界 cell（旧数据本来就该丢），其余槽位保持自己的世界身份，图集不会因相机
// 移动而整体失效。旧版槽位是「相对相机窗口的格号」，滚一格就把整级所有槽位的世界 cell
// 全换掉 → 整级图集变成旧位置的读数 → 深度判定成片失败（Probe 大片红）+ 下一帧重写
// （大片绿），即「相机移动时的 GI 闪烁」。前提：`from_camera` 的原点必须是 cell 整数倍。

// indirect args / 计数器合一 buffer（word 布局与 WGSL DDGI_INDIR_* 对应）：
//   [0..16) cast args ×4 LOD；[16..32) collect args ×4 LOD；[32..36) rpp；[36..40) 活跃计数器
pub const DDGI_INDIRECT_BYTES: u64 = 256;
pub const DDGI_COUNTER_CLEAR_OFFSET: u64 = 36 * 4;
pub const DDGI_COUNTER_CLEAR_BYTES: u64 = 16;
pub const DDGI_WORKLIST_ITEM_BYTES: u64 = 16; // vec4(probe_pos.xyz, packed age|lod)
/// meta word = age(8 bit) | ENABLED<<8 | ACTIVE<<9；0 = 无探针哨兵。
pub const DDGI_META_ENABLED: u32 = 1 << 8;
/// ACTIVE：本帧需要投线（本 cell 或 6 邻接有体素/物体，且不在更细 LOD 覆盖内）。
pub const DDGI_META_ACTIVE: u32 = 1 << 9;

#[inline]
fn align_down(v: i32, a: i32) -> i32 {
  v.div_euclid(a) * a
}

/// 世界空间探针网格（4 级嵌套级联，各自独立原点，以相机为中心）。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DdgiWorldGrid {
  /// 各 LOD 的世界原点（voxel，按该级 cell 边长对齐）。
  pub lod_origins: [IVec3; DDGI_LODS as usize],
  /// 各 LOD 的 cell 维度（cell 单位）。
  pub lod_dims: [UVec3; DDGI_LODS as usize],
  /// 各 LOD 在全局 slot 数组中的起始下标。
  pub lod_slot_base: [u32; DDGI_LODS as usize],
  pub total_slots: u32,
}

impl DdgiWorldGrid {
  #[inline]
  pub fn lod_count(&self, lod: usize) -> u32 {
    let d = self.lod_dims[lod];
    d.x * d.y * d.z
  }

  #[inline]
  pub fn lod_cell_size(&self, lod: usize) -> i32 {
    DDGI_LOD_CELL_SIZES[lod]
  }

  /// 由相机世界坐标（voxel）推导 4 级嵌套级联。
  ///
  /// 每级：dims 固定为 `DDGI_LOD_DIMS`，cell 边长 = `DDGI_LOD_CELL_SIZES[lod]`；原点取
  /// `cam - half·cell` 并**按 cell 边长向下对齐**。这带来两个必须成立的不变量：
  /// - **相机恒在盒中心（偏差 < 1 个 cell）**：覆盖率最大化且不随相机漂移。旧版按
  ///   `cell×8` 对齐、再前移半步，导致相机在盒内游走（覆盖浮动）；现在不需要了。
  /// - **原点恒是 cell 的整数倍**：shader 里「世界 cell 号 = 原点 / cell」才能精确整除，
  ///   世界锚定的槽位映射（`slot = 世界 cell mod dims`）才成立。
  ///
  /// 因覆盖范围逐级 ×2，LOD(l-1) 盒严格包含于 LOD(l) 盒内。
  pub fn from_camera(camera_voxel: IVec3) -> Self {
    let mut out = Self::default();
    let dims = DDGI_LOD_DIMS;
    let half_cells = (DDGI_LOD_DIMS / 2).as_ivec3();
    let mut base = 0u32;
    for lod in 0..DDGI_LODS as usize {
      let cell = DDGI_LOD_CELL_SIZES[lod];
      let half = half_cells * cell;
      let origin = IVec3::new(
        align_down(camera_voxel.x - half.x, cell),
        align_down(camera_voxel.y - half.y, cell),
        align_down(camera_voxel.z - half.z, cell),
      );
      out.lod_origins[lod] = origin;
      out.lod_dims[lod] = dims;
      out.lod_slot_base[lod] = base;
      base += dims.x * dims.y * dims.z;
    }
    out.total_slots = base;
    out
  }

  #[inline]
  pub fn is_empty(&self) -> bool {
    self.total_slots == 0
  }
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, ShaderType)]
pub struct DdgiLod {
  /// xyz = 世界原点（voxel），w = cell 边长
  pub origin: IVec4,
  /// xyz = cell 维度，w = 全局 slot 起始下标
  pub dims: UVec4,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, bevy::ecs::resource::Resource, ShaderType)]
pub struct DdgiUniform {
  pub lods: [DdgiLod; 4],
  /// x = frame, y = debug mode, z = gain, w = 保留（恒 0）
  pub params: Vec4,
  /// x = shade GI, y = total slots
  pub misc: Vec4,
  /// 脏区（世界 voxel AABB）：xyz = min，w = 1 表示有效（0 = 本帧无脏区）
  pub dirty_min: Vec4,
  /// xyz = max（不含）；与 dirty_min 一起决定哪些 cell 强制重烘（局部编辑增量）
  pub dirty_max: Vec4,
}

pub fn ddgi_bg4_layout() -> bevy::render::render_resource::BindGroupLayoutDescriptor {
  use bevy::render::render_resource::*;
  const C: ShaderStages = ShaderStages::COMPUTE;
  let tex = |binding: u32, sample_type: TextureSampleType| BindGroupLayoutEntry {
    binding,
    visibility: C,
    ty: BindingType::Texture {
      sample_type,
      view_dimension: TextureViewDimension::D2Array,
      multisampled: false,
    },
    count: None,
  };
  let buf = |binding: u32, read_only: bool| BindGroupLayoutEntry {
    binding,
    visibility: C,
    ty: BindingType::Buffer {
      ty: BufferBindingType::Storage { read_only },
      has_dynamic_offset: false,
      min_binding_size: None,
    },
    count: None,
  };
  BindGroupLayoutDescriptor::new(
    "DdgiBg4",
    &[
      BindGroupLayoutEntry {
        binding: 0,
        visibility: C,
        ty: BindingType::Buffer {
          ty: BufferBindingType::Uniform,
          has_dynamic_offset: false,
          min_binding_size: Some(DdgiUniform::min_size()),
        },
        count: None,
      },
      // 1/2：irradiance / depth 图集（采样侧；cast 回读 GI、着色 ddgi_sample 用）
      tex(1, TextureSampleType::Float { filterable: true }),
      tex(2, TextureSampleType::Float { filterable: false }),
      // 3：烘焙输出（bake 写 / sort 读）4：age/flags（读写）5：indirect/counter（读写）
      // 6：worklist（读写）7：slot_pos（读写）8：cell_id（读写，滚动增量）
      // 9：cast 射线样本（cast 写 / collect 读）
      // 10：cell→slot 间接表（读写）—— 允许一个 cell 指向**邻近 cell 的探针**，
      //     这样"探针必须离表面足够远"和"每个采样点都有 8 个可用角"可以同时成立。
      //     初始化成"指向自身"时与旧行为逐位等价（见 create 处的 identity 填充）。
      buf(3, false),
      buf(4, false),
      buf(5, false),
      buf(6, false),
      buf(7, false),
      buf(8, false),
      buf(9, false),
      buf(10, false),
    ],
  )
}

/// BG5（仅 collect 用）：图集的**写入侧**。与 BG4 分离是硬性要求——同一纹理不能在同一
/// bind group / 同一 pass 内既作采样纹理又作存储纹理；collect 只写图集，其余 pass 只读。
pub fn ddgi_bg5_layout() -> bevy::render::render_resource::BindGroupLayoutDescriptor {
  use bevy::render::render_resource::*;
  const C: ShaderStages = ShaderStages::COMPUTE;
  let store = |binding: u32, format: TextureFormat| BindGroupLayoutEntry {
    binding,
    visibility: C,
    ty: BindingType::StorageTexture {
      access: StorageTextureAccess::WriteOnly,
      format,
      view_dimension: TextureViewDimension::D2Array,
    },
    count: None,
  };
  BindGroupLayoutDescriptor::new(
    "DdgiBg5",
    &[
      store(0, TextureFormat::Rgba16Float),
      // depth 图集也从单通道升到 Rgba16Float：.x = mean、.y = std（距离标准差），
      // 供采样侧做 Chebyshev 软遮挡（参考 Majercik/RTXGI）。R32Float 只有均值，
      // 只能做刀锋判定 → 深度一抖就"入选/落选"翻转（亮区边界伸缩）。
      store(1, TextureFormat::Rgba16Float),
    ],
  )
}

/// BG6（仅 seal 用）：dispatch 参数 buffer 的**写入侧**。
/// 与 BG4 分离是硬性要求：该 buffer 在 cast/collect pass 里要作 indirect 参数源，
/// 若同时被绑成 storage 会被 wgpu 判为 usage 冲突。
pub fn ddgi_bg6_layout() -> bevy::render::render_resource::BindGroupLayoutDescriptor {
  use bevy::render::render_resource::*;
  const C: ShaderStages = ShaderStages::COMPUTE;
  BindGroupLayoutDescriptor::new(
    "DdgiBg6",
    &[BindGroupLayoutEntry {
      binding: 0,
      visibility: C,
      ty: BindingType::Buffer {
        ty: BufferBindingType::Storage { read_only: false },
        has_dynamic_offset: false,
        min_binding_size: None,
      },
      count: None,
    }],
  )
}

#[derive(Debug, Clone, Copy)]
pub struct DdgiPipelines {
  /// per-LOD 烘焙入口（lod 0..DDGI_LODS，细→粗）。**必须逐级拆成独立 pass**：
  /// 粗级的探针放置要读本帧细级的 bake 输出，而同一 pass 内没有顺序保证，
  /// 只有 pass 边界才是内存屏障（见 WGSL `ddgi_bake_one`）。
  pub bake: [CachedComputePipelineId; DDGI_LODS as usize],
  pub sort: CachedComputePipelineId,
  pub seal: CachedComputePipelineId,
  pub cast: CachedComputePipelineId,
  pub collect: CachedComputePipelineId,
}

/// 图集的一侧（纹理 + D2Array 视图）
pub struct DdgiAtlas {
  pub tex: bevy::render::render_resource::Texture,
  pub view: bevy::render::render_resource::TextureView,
}

#[derive(bevy::ecs::resource::Resource)]
pub struct DdgiGpu {
  pub uniform: bevy::render::render_resource::UniformBuffer<DdgiUniform>,
  pub indirect: bevy::render::render_resource::Buffer,
  /// dispatch 参数（`INDIRECT`）。**必须与 `ddgi_indirect` 分开**：同一 pass 内一个 buffer 若
  /// 既作 storage 绑定又作 indirect 参数源，wgpu 会判定 usage 冲突（`STORAGE_READ_WRITE` 是独占用法）。
  /// 它只在 seal pass 里以 storage 绑定（BG6）；cast/collect 只把它当 indirect 源，不绑定。
  pub args: bevy::render::render_resource::Buffer,
  pub worklist: bevy::render::render_resource::Buffer,
  pub slot_pos: bevy::render::render_resource::Buffer,
  /// cell → slot 间接表（每 LOD 16384 项，u32）。初始化 = 指向自身；bake 可把"本格放不出
  /// 探针"的 cell 指向邻近 cell 的探针（见 WGSL `ddgi_slot`）。
  pub cell_slot: bevy::render::render_resource::Buffer,
  /// 烘焙输出：每 slot 一条 (flags | off_b)
  pub cell: bevy::render::render_resource::Buffer,
  /// 每 slot 已烘焙的世界 cell 键 + 有效标志（滚动增量烘焙）
  pub cell_id: bevy::render::render_resource::Buffer,
  /// 每帧 age / enabled
  pub meta: bevy::render::render_resource::Buffer,
  /// 图集 ping-pong：BG4 绑「当前」（上一帧 collect 写入的，供 cast 回读 + 着色采样），
  /// BG5 绑「目标」（本帧 collect 写入）。读写必须落在不同纹理 + 不同 bind group：
  /// wgpu 禁止同一 pass 内把同一纹理既当可写存储又当采样纹理。
  pub irr: [DdgiAtlas; 2],
  pub depth: [DdgiAtlas; 2],
  /// 采样侧索引（0/1）；dispatch_ddgi 在 collect 跑完后翻转
  pub parity: usize,
  /// cast 输出的射线样本（方向 + 命中距离）/(辐亮度 + 1)
  pub samples: bevy::render::render_resource::Buffer,
  pub frame: u32,
  pub grid: DdgiWorldGrid,
  pub total_slots: u32,
  /// 已消费的世界修订号（变化 → 需要重烘焙）
  pub last_revision: u64,
  /// 有一次烘焙请求已发出但**还没真正派发**（pipeline 未编译好 / 绑定组未就绪时
  /// `dispatch_ddgi` 会提前返回）。`prepare_ddgi` 一旦推进了 `grid`/`last_revision`，
  /// 这次请求就不会再被 `grid_changed || rev_changed` 重新检出 —— 必须靠这个标志把它
  /// 留到真正派发的那一帧，否则「启动时就打开 DDGI」会一次也不烘焙，整级图集恒空。
  pub bake_pending: bool,
  pub pipelines: Option<DdgiPipelines>,
}

#[derive(bevy::ecs::resource::Resource)]
pub struct DdgiBg4(pub bevy::render::render_resource::BindGroup);

/// collect 专用 bind group（图集写入侧 + 样本/间接参数读取）
#[derive(bevy::ecs::resource::Resource)]
pub struct DdgiBg5(pub bevy::render::render_resource::BindGroup);

/// seal 专用 bind group（dispatch 参数 buffer 写入侧）
#[derive(bevy::ecs::resource::Resource)]
pub struct DdgiBg6(pub bevy::render::render_resource::BindGroup);

/// 本帧是否需要跑探针烘焙（由 prepare_ddgi 写入，dispatch_ddgi 读取）
#[derive(bevy::ecs::resource::Resource, Clone, Copy, Default)]
pub struct DdgiBakeThisFrame(pub bool);

#[derive(bevy::ecs::resource::Resource, Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct DdgiStage(pub u8);

impl DdgiStage {
  pub const OFF: u8 = 0;
  pub const ACTIVE: u8 = 1;
  pub const CAST: u8 = 2;
  pub const FULL: u8 = 3;

  pub fn new(v: u8) -> Self {
    Self(v.min(Self::FULL))
  }
  /// `GATE_DDGI_STAGE=0..3` 覆盖启动阶段（与 GATE_NO_LOD / GATE_SKIP_CHUNKWALK 同风格）。
  ///
  /// **缺省 = FULL(3)**：DDGI 已稳定，默认开启（用户明确要求）。曾经缺省 Off、只能靠 UI
  /// 滑杆打开，`GATE_BENCH=1` 的无 UI 跑法因此拿不到 Full 的帧时数据 —— 现在反过来：
  /// 需要基准对比「DDGI=Off」时显式 `GATE_DDGI_STAGE=0`。
  fn from_env() -> Self {
    Self::new(
      std::env::var("GATE_DDGI_STAGE")
        .ok()
        .and_then(|s| s.trim().parse::<u8>().ok())
        .unwrap_or(Self::FULL),
    )
  }
  pub fn run_active(&self) -> bool {
    self.0 >= Self::ACTIVE
  }
  /// 是否跑阶段二/三（cast + collect）
  pub fn run_cast(&self) -> bool {
    self.0 >= Self::CAST
  }
  pub fn shade_gi(&self) -> bool {
    self.0 >= Self::FULL
  }
}

#[derive(bevy::ecs::resource::Resource, Clone, Copy, Debug, PartialEq)]
pub struct DdgiDebugSettings {
  pub mode: f32,
  pub gain: f32,
  pub probe_viz: bool,
  pub probe_viz_lod: f32,
  /// 借针搜索半径（格），对应 WGSL `params.w`：0 = 关闭借针（Douglas 原架构：
  /// 无针 cell 的插值角直接缺席），1 = ±1 邻域，2 = ±2 邻域（现状默认）。
  /// 仅作诊断 A/B：验证跨墙借针对室内墙角漏光的贡献。sort 每帧重写间接表，
  /// 拖动滑杆下一帧即生效，无需 rebake。
  pub borrow_radius: f32,
  /// Chebyshev 里 **std 项的信任系数**，对应 WGSL `misc.z`（0..1，默认 1 = 正常使用 std）。
  ///
  /// 拖到 0 = 完全忽略 std，`soft` 退回固定下限 `DDGI_DEPTH_SOFT_MIN`（硬判定：更能压漏光，
  /// 但过渡带变窄、动态时更易闪）。保留作 A/B 诊断用。
  ///
  /// 注：该系数最初是为确认一个已修复的缺陷而加 —— 射线方向当时是「Fibonacci 球 + 每帧
  /// 随机四元数整体重旋」，每个深度纹素跨帧收到的是全球随机方向，`std` 度量的是「20° 锥内
  /// 几何起伏」而非「同方向噪声」，墙角虚高 → 软漏光。现在射线已**绑定到深度纹素**（见
  /// dda.wgsl 的 cast「射线 ↔ 深度纹素绑定」），std 语义已正确。
  pub depth_soft_k: f32,
  /// 级联覆盖**之外**的天光兜底强度，对应 WGSL `misc.w`（0..1，默认 0.25）。
  ///
  /// 覆盖内的环境光由 DDGI 算出，覆盖外只能靠常量兜底 —— 两者强度不匹配时，级联盒边界
  /// 就是一条"亮 ↔ 暗"的硬边（相机拉远必然出现"有 GI / 无 GI 同屏"）。
  /// **不能直接用 `DDGI_SKY_AMBIENT` 调大**：它还兼作覆盖内无数据时的兜底，调大会让室内
  /// 凹角跟着变亮（漏光感）。所以覆盖外单独一个系数，运行时滑杆调到与覆盖内衔接为止。
  pub far_ambient: f32,
}

impl Default for DdgiDebugSettings {
  fn default() -> Self {
    Self {
      mode: 0.0,
      gain: 1.0,
      probe_viz: false,
      probe_viz_lod: 0.0,
      borrow_radius: 2.0,
      depth_soft_k: 1.0,
      far_ambient: 0.25,
    }
  }
}

pub struct DdgiPlugin;

impl bevy::app::Plugin for DdgiPlugin {
  fn build(&self, app: &mut bevy::app::App) {
    use bevy::ecs::schedule::IntoScheduleConfigs;
    use bevy::prelude::RenderGraph;
    let stage = DdgiStage::from_env();
    bevy::log::info!(
      target: "gate",
      "DDGI stage = {} ({}) —— 缺省 Full；GATE_DDGI_STAGE=0..3 可覆盖",
      stage.0,
      ["Off", "Active", "Cast", "Full"][stage.0.min(3) as usize],
    );
    app.insert_resource(stage);
    app.init_resource::<DdgiDebugSettings>();
    let Some(render_app) = app.get_sub_app_mut(bevy::render::RenderApp) else {
      return;
    };
    render_app
      .init_resource::<DdgiStage>()
      .init_resource::<DdgiDebugSettings>()
      .init_resource::<DdgiBakeThisFrame>()
      .add_systems(bevy::render::RenderStartup, init_ddgi_gpu)
      .add_systems(
        bevy::render::RenderStartup,
        queue_ddgi_pipelines.after(crate::brickmap::dda::init_dda_pipelines),
      )
      .add_systems(bevy::render::ExtractSchedule, extract_ddgi_settings)
      .add_systems(
        bevy::render::Render,
        prepare_ddgi
          .in_set(bevy::render::RenderSystems::PrepareBindGroups)
          .after(crate::brickmap::upload::prepare),
      )
      .add_systems(
        RenderGraph,
        dispatch_ddgi
          .in_set(bevy::render::renderer::RenderGraphSystems::Render)
          .before(crate::brickmap::dda::dispatch_dda),
      );
  }
}

fn dummy_sized_buffer(
  device: &bevy::render::renderer::RenderDevice,
  label: &str,
  size: u64,
) -> bevy::render::render_resource::Buffer {
  use bevy::render::render_resource::{BufferDescriptor, BufferUsages};
  device.create_buffer(&BufferDescriptor {
    label: Some(label.into()),
    size: size.max(4),
    usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
    mapped_at_creation: false,
  })
}

fn zero_storage_buffer(
  device: &bevy::render::renderer::RenderDevice,
  queue: &bevy::render::renderer::RenderQueue,
  label: &str,
  size: u64,
) -> bevy::render::render_resource::Buffer {
  let buf = dummy_sized_buffer(device, label, size);
  queue.write_buffer(&buf, 0, &vec![0u8; size.max(4) as usize]);
  buf
}

/// 容量不足时重建（并清零）。base 网格变化 = 世界窗口变化，低频。
fn ensure_storage_buffer(
  device: &bevy::render::renderer::RenderDevice,
  queue: &bevy::render::renderer::RenderQueue,
  buf: &mut bevy::render::render_resource::Buffer,
  label: &str,
  bytes: u64,
) {
  let bytes = bytes.max(4);
  if buf.size() >= bytes {
    return;
  }
  let new = dummy_sized_buffer(device, label, bytes);
  queue.write_buffer(&new, 0, &vec![0u8; bytes as usize]);
  *buf = new;
}

/// dispatch 参数 / 计数器 buffer：既要作为 storage（atomic）被 sort/seal 读写，
/// 又要（仅 args）作为 indirect dispatch 的参数源。两者必须是不同 buffer，见 `DdgiGpu::args`。
fn ddgi_indirect_buffer(
  device: &bevy::render::renderer::RenderDevice,
  queue: &bevy::render::renderer::RenderQueue,
  label: &str,
) -> bevy::render::render_resource::Buffer {
  use bevy::render::render_resource::{BufferDescriptor, BufferUsages};
  let buf = device.create_buffer(&BufferDescriptor {
    label: Some(label.into()),
    size: DDGI_INDIRECT_BYTES,
    usage: BufferUsages::STORAGE
      | BufferUsages::INDIRECT
      | BufferUsages::COPY_DST
      | BufferUsages::COPY_SRC,
    mapped_at_creation: false,
  });
  queue.write_buffer(&buf, 0, &vec![0u8; DDGI_INDIRECT_BYTES as usize]);
  buf
}

fn ddgi_array_view(
  tex: &bevy::render::render_resource::Texture,
) -> bevy::render::render_resource::TextureView {
  use bevy::render::render_resource::{TextureViewDescriptor, TextureViewDimension};
  tex.create_view(&TextureViewDescriptor {
    dimension: Some(TextureViewDimension::D2Array),
    ..Default::default()
  })
}

/// irradiance / depth 图集容量：= 各级 LOD 槽数总和 65536；每层 16×16 = 256 探针 → 256 层。
pub const DDGI_ATLAS_LAYERS: u32 = 256;
pub const DDGI_ATLAS_PROBES_PER_LAYER_AXIS: u32 = 16;
/// 阶段二射线样本缓冲：每样本 2×vec4 = (方向.xyz, 命中距离) + (辐亮度.xyz, 1)。
///
/// 样本下标 = **全局射线编号**（`ddgi_cast` 里 `si = tid * 2`），而 seal 保证全局射线
/// 总数 ≤ `DDGI_RAY_BUDGET`，所以槽数直接取预算即可。
pub const DDGI_SAMPLE_SLOTS: u32 = DDGI_RAY_BUDGET;
pub const DDGI_SAMPLE_BYTES: u64 = (DDGI_SAMPLE_SLOTS as u64) * 32;

fn ddgi_array_tex(
  device: &bevy::render::renderer::RenderDevice,
  label: &str,
  format: bevy::render::render_resource::TextureFormat,
  size: (u32, u32),
) -> bevy::render::render_resource::Texture {
  use bevy::render::render_resource::{
    Extent3d, TextureDescriptor, TextureDimension, TextureUsages,
  };
  device.create_texture(&TextureDescriptor {
    label: Some(label.into()),
    size: Extent3d {
      width: size.0,
      height: size.1,
      depth_or_array_layers: DDGI_ATLAS_LAYERS,
    },
    mip_level_count: 1,
    sample_count: 1,
    dimension: TextureDimension::D2,
    format,
    usage: TextureUsages::TEXTURE_BINDING
      | TextureUsages::STORAGE_BINDING
      | TextureUsages::COPY_DST
      | TextureUsages::COPY_SRC,
    view_formats: &[],
  })
}

/// 图集清零。Rgba16Float 的 0.0 是全 0 字节，直接写零块。
/// 必须清零：probe 一 enabled 就可能被采样，但它的图集纹素要等第一次 collect 才有效，
/// 否则会读到未初始化显存（NaN/垃圾污染 GI）。
fn zero_array_tex(
  queue: &bevy::render::renderer::RenderQueue,
  label: &str,
  tex: &bevy::render::render_resource::Texture,
  size: (u32, u32),
  bytes_per_texel: u32,
) {
  use bevy::render::render_resource::{
    Extent3d, Origin3d, TexelCopyBufferLayout, TexelCopyTextureInfo, TextureAspect,
  };
  let bytes = (size.0 * size.1 * DDGI_ATLAS_LAYERS * bytes_per_texel) as usize;
  bevy::log::info!(target: "gate", "DDGI 图集清零 {label}: {:.1}MB", bytes as f64 / (1 << 20) as f64);
  let data = vec![0u8; bytes];
  queue.write_texture(
    TexelCopyTextureInfo {
      texture: tex,
      mip_level: 0,
      origin: Origin3d::ZERO,
      aspect: TextureAspect::All,
    },
    &data,
    TexelCopyBufferLayout {
      offset: 0,
      bytes_per_row: Some(size.0 * bytes_per_texel),
      rows_per_image: Some(size.1),
    },
    Extent3d {
      width: size.0,
      height: size.1,
      depth_or_array_layers: DDGI_ATLAS_LAYERS,
    },
  );
}

fn init_ddgi_gpu(
  mut commands: bevy::ecs::system::Commands,
  device: bevy::ecs::system::Res<bevy::render::renderer::RenderDevice>,
  queue: bevy::ecs::system::Res<bevy::render::renderer::RenderQueue>,
) {
  use bevy::render::render_resource::TextureFormat;
  let irr_axis = DDGI_ATLAS_PROBES_PER_LAYER_AXIS * IRRADIANCE_TEXELS;
  let dep_axis = DDGI_ATLAS_PROBES_PER_LAYER_AXIS * DEPTH_TEXELS;
  let mk_atlas = |label: &str, format: TextureFormat, size: (u32, u32), bpt: u32| {
    let tex = ddgi_array_tex(&device, label, format, size);
    // 两侧都要清零：首帧 BG4 采样 a、collect 写 b；翻转后 a 才被写，不清零会读到未初始化显存。
    zero_array_tex(&queue, label, &tex, size, bpt);
    let view = ddgi_array_view(&tex);
    DdgiAtlas { tex, view }
  };
  let irr = [
    mk_atlas("ddgi_irr_a", TextureFormat::Rgba16Float, (irr_axis, irr_axis), 8),
    mk_atlas("ddgi_irr_b", TextureFormat::Rgba16Float, (irr_axis, irr_axis), 8),
  ];
  let depth = [
    mk_atlas("ddgi_depth_a", TextureFormat::Rgba16Float, (dep_axis, dep_axis), 8),
    mk_atlas("ddgi_depth_b", TextureFormat::Rgba16Float, (dep_axis, dep_axis), 8),
  ];

  let slot_pos = zero_storage_buffer(&device, &queue, "ddgi_slot_pos", 4096 * 16);
  let worklist = zero_storage_buffer(&device, &queue, "ddgi_worklist", 4096 * 16);
  let cell = zero_storage_buffer(&device, &queue, "ddgi_cell", 4096 * 4);
  let cell_id = zero_storage_buffer(&device, &queue, "ddgi_cell_id", 4096 * 16);
  let meta = zero_storage_buffer(&device, &queue, "ddgi_meta", 4096 * 4);
  // cell→slot 间接表：表长固定（= Σ 各级 dims 乘积），一次性建满 + 填成 **identity**
  // （`ddgi_slot` 返回绝对 slot 下标，所以 identity = [0,1,2,...]）。
  // identity 状态与"没有这张表"逐位等价 → 这一步本身不改画面，只建立通路（Step 1 第 1 小步）。
  let n_cells = (DDGI_LODS * DDGI_LOD_DIMS.x * DDGI_LOD_DIMS.y * DDGI_LOD_DIMS.z) as usize;
  let cell_slot = zero_storage_buffer(&device, &queue, "ddgi_cell_slot", (n_cells * 4) as u64);
  let mut identity = Vec::with_capacity(n_cells * 4);
  for i in 0..n_cells as u32 {
    identity.extend_from_slice(&i.to_le_bytes());
  }
  queue.write_buffer(&cell_slot, 0, &identity);
  let samples = zero_storage_buffer(&device, &queue, "ddgi_samples", DDGI_SAMPLE_BYTES);
  let indirect = ddgi_indirect_buffer(&device, &queue, "ddgi_indirect");
  let args = ddgi_indirect_buffer(&device, &queue, "ddgi_args");

  commands.insert_resource(DdgiGpu {
    uniform: bevy::render::render_resource::UniformBuffer::default(),
    indirect,
    args,
    worklist,
    slot_pos,
    cell_slot,
    cell,
    cell_id,
    meta,
    irr,
    depth,
    parity: 0,
    samples,
    frame: 0,
    grid: DdgiWorldGrid::default(),
    total_slots: 0,
    last_revision: u64::MAX,
    bake_pending: false,
    pipelines: None,
  });
}

fn queue_ddgi_pipelines(
  dda: Option<bevy::ecs::system::Res<crate::brickmap::dda::DdaPipelines>>,
  pipeline_cache: bevy::ecs::system::Res<bevy::render::render_resource::PipelineCache>,
  asset_server: bevy::ecs::system::Res<bevy::asset::AssetServer>,
  mut gpu: bevy::ecs::system::ResMut<DdgiGpu>,
) {
  use bevy::render::render_resource::{BindGroupLayoutDescriptor, ComputePipelineDescriptor, PipelineCache};
  use std::borrow::Cow;
  if gpu.pipelines.is_some() {
    return;
  }
  let Some(dda) = dda else {
    return;
  };
  let base = vec![
    dda.bg0_layout.clone(),
    dda.bg1_layout.clone(),
    dda.bg2_layout.clone(),
    dda.bg3_layout.clone(),
    ddgi_bg4_layout(),
  ];
  // collect 额外挂 BG5（图集写入侧）；seal 额外挂 BG6（dispatch 参数写入侧）。
  // 注意：layout 的**位置**就是 bind group 索引。base 已占 0..4，collect 的 BG5 正好落在 5；
  // seal 的参数侧在 @group(6)，而 @group(5) 已被 collect 占用（同一 WGSL 模块里同一个
  // group/binding 槽只能有一个变量），所以 seal 的 layout 在 5 号位留一个空的 layout。
  let mut collect_layout = base.clone();
  collect_layout.push(ddgi_bg5_layout());
  let mut seal_layout = base.clone();
  seal_layout.push(BindGroupLayoutDescriptor::new("DdgiBgEmpty5", &[]));
  seal_layout.push(ddgi_bg6_layout());
  let shader = asset_server.load(crate::brickmap::dda::DDA_SHADER_ASSET_PATH);
  let mk = |pipeline_cache: &PipelineCache,
            label: &'static str,
            entry: &'static str,
            layout: Vec<BindGroupLayoutDescriptor>| {
    pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
      label: Some(Cow::from(label)),
      layout,
      shader: shader.clone(),
      entry_point: Some(Cow::from(entry)),
      ..Default::default()
    })
  };
  gpu.pipelines = Some(DdgiPipelines {
    bake: [
      mk(&pipeline_cache, "gate_ddgi_bake0", "ddgi_bake0", base.clone()),
      mk(&pipeline_cache, "gate_ddgi_bake1", "ddgi_bake1", base.clone()),
      mk(&pipeline_cache, "gate_ddgi_bake2", "ddgi_bake2", base.clone()),
      mk(&pipeline_cache, "gate_ddgi_bake3", "ddgi_bake3", base.clone()),
    ],
    sort: mk(&pipeline_cache, "gate_ddgi_sort", "ddgi_sort", base.clone()),
    seal: mk(&pipeline_cache, "gate_ddgi_seal", "ddgi_seal", seal_layout),
    cast: mk(&pipeline_cache, "gate_ddgi_cast", "ddgi_cast", base.clone()),
    collect: mk(
      &pipeline_cache,
      "gate_ddgi_collect",
      "ddgi_collect",
      collect_layout,
    ),
  });
}

#[allow(clippy::too_many_arguments)]
fn dispatch_ddgi(
  mut ctx: bevy::render::renderer::RenderContext,
  bg0: Option<bevy::ecs::system::Res<crate::brickmap::dda::DdaBg0BindGroup>>,
  bg1: Option<bevy::ecs::system::Res<crate::brickmap::dda::DdaBg1BindGroup>>,
  bg2: Option<bevy::ecs::system::Res<crate::brickmap::dda::DdaBg2BindGroup>>,
  bg3: Option<bevy::ecs::system::Res<crate::brickmap::dda::DdaBg3BindGroup>>,
  bg4: Option<bevy::ecs::system::Res<DdgiBg4>>,
  bg5: Option<bevy::ecs::system::Res<DdgiBg5>>,
  bg6: Option<bevy::ecs::system::Res<DdgiBg6>>,
  bake: Option<bevy::ecs::system::Res<DdgiBakeThisFrame>>,
  stage: bevy::ecs::system::Res<DdgiStage>,
  dbg: Option<bevy::ecs::system::Res<DdgiDebugSettings>>,
  mut gpu: bevy::ecs::system::ResMut<DdgiGpu>,
  pipeline_cache: bevy::ecs::system::Res<bevy::render::render_resource::PipelineCache>,
  mut profiler: bevy::ecs::system::ResMut<crate::profiler::GpuProfilerRes>,
) {
  if !stage.run_active() && !dbg.map_or(false, |d| d.probe_viz) {
    return;
  }
  let (Some(bg0), Some(bg1), Some(bg2), Some(bg3), Some(bg4)) = (
    bg0.as_ref(),
    bg1.as_ref(),
    bg2.as_ref(),
    bg3.as_ref(),
    bg4.as_ref(),
  ) else {
    return;
  };
  let Some(pipes) = gpu.pipelines else {
    return;
  };
  let Some(p_sort) = pipeline_cache.get_compute_pipeline(pipes.sort) else {
    return;
  };
  let Some(p_seal) = pipeline_cache.get_compute_pipeline(pipes.seal) else {
    return;
  };
  if gpu.total_slots == 0 {
    return;
  }

  let set_bgs = |pass: &mut bevy::render::render_resource::ComputePass,
                 bg4: &bevy::render::render_resource::BindGroup| {
    pass.set_bind_group(0, &bg0.0, &[]);
    pass.set_bind_group(1, &bg1.0, &[]);
    pass.set_bind_group(2, &bg2.0, &[]);
    pass.set_bind_group(3, &bg3.0, &[]);
    pass.set_bind_group(4, bg4, &[]);
  };
  // 1D dispatch：WG=64，各 pass 覆盖 total_slots（bake 仅在修订号变化时跑）
  let wg_slots = gpu.total_slots.div_ceil(64).min(65535);

  // 烘焙：世界数据变化时重算探针位置（BFS 最大空叶）。
  // **按 LOD 逐级派发 4 个独立 pass（细→粗）**：粗级的放置要继承本帧细级的 bake 输出，
  // 同一 pass 内没有顺序保证，只有 pass 边界才是内存屏障（见 WGSL `ddgi_bake_one`）。
  if bake.map_or(false, |b| b.0) {
    let n_lods = DDGI_LODS as usize;
    let wg_per_lod = (DDGI_LOD_DIMS.x * DDGI_LOD_DIMS.y * DDGI_LOD_DIMS.z)
      .div_ceil(64)
      .min(65535);
    const LABELS: [&str; DDGI_LODS as usize] = [
      "gate_ddgi_bake0",
      "gate_ddgi_bake1",
      "gate_ddgi_bake2",
      "gate_ddgi_bake3",
    ];
    // 4 个入口必须**全部**就绪才开跑：否则会出现"细级烘了、粗级没烘"的半帧状态。
    let mut p_bake: [Option<&bevy::render::render_resource::ComputePipeline>; DDGI_LODS as usize] =
      [None; DDGI_LODS as usize];
    let mut ready = true;
    for lod in 0..n_lods {
      p_bake[lod] = pipeline_cache.get_compute_pipeline(pipes.bake[lod]);
      if p_bake[lod].is_none() {
        ready = false;
      }
    }
    if ready {
      for lod in 0..n_lods {
        let p = p_bake[lod].unwrap();
        crate::profiler::gpu_compute_pass(
          &mut profiler,
          ctx.command_encoder(),
          LABELS[lod],
          |pass| {
            pass.set_pipeline(p);
            set_bgs(pass, &bg4.0);
            pass.dispatch_workgroups(wg_per_lod, 1, 1);
          },
        );
      }
      // 真正派发过了才撤销挂起标志（否则保留它，等下一帧管线就绪再烘）
      gpu.bake_pending = false;
    }
  }
  // sort：读烘焙结果做活跃判定 + worklist 压缩 + age/slot_pos/meta 刷新（probe_viz 依赖其新鲜度）
  crate::profiler::gpu_compute_pass(
    &mut profiler,
    ctx.command_encoder(),
    "gate_ddgi_sort",
    |pass| {
      pass.set_pipeline(p_sort);
      set_bgs(pass, &bg4.0);
      pass.dispatch_workgroups(wg_slots, 1, 1);
    },
  );
  // seal：为 cast/collect 准备 per-LOD indirect args；只在需要阶段二时跑。
  if stage.run_active() {
    crate::profiler::gpu_compute_pass(
      &mut profiler,
      ctx.command_encoder(),
      "gate_ddgi_seal",
      |pass| {
        pass.set_pipeline(p_seal);
        set_bgs(pass, &bg4.0);
        if let Some(bg6) = bg6.as_ref() {
          pass.set_bind_group(6, &bg6.0, &[]);
        }
        pass.dispatch_workgroups(1, 1, 1);
      },
    );
  }
  // 阶段二 cast / 阶段三 collect：seal 已备好 per-LOD indirect args
  // （byte 0 = cast×4 LOD，byte 64 = collect×4 LOD）与 rpp（byte 128）。
  if stage.run_cast() {
    if let (Some(p_cast), Some(p_coll), Some(bg5)) = (
      pipeline_cache.get_compute_pipeline(pipes.cast),
      pipeline_cache.get_compute_pipeline(pipes.collect),
      bg5.as_ref(),
    ) {
      // cast / collect 各一次间接 dispatch（覆盖全部 LOD；LOD 由 shader 用 seal 写的前缀和还原）。
      // 注意两个 args 必须放在不同 buffer 段：cast 在 ddgi_args 词 0（字节 0），
      // collect 在词 16（字节 64）。
      crate::profiler::gpu_compute_pass(
        &mut profiler,
        ctx.command_encoder(),
        "gate_ddgi_cast",
        |pass| {
          pass.set_pipeline(p_cast);
          set_bgs(pass, &bg4.0);
          pass.dispatch_workgroups_indirect(&gpu.args, 0);
        },
      );
      crate::profiler::gpu_compute_pass(
        &mut profiler,
        ctx.command_encoder(),
        "gate_ddgi_collect",
        |pass| {
          pass.set_pipeline(p_coll);
          set_bgs(pass, &bg4.0);
          pass.set_bind_group(5, &bg5.0, &[]);
          pass.dispatch_workgroups_indirect(&gpu.args, 64);
        },
      );
      // 本帧写过图集 → 下一帧采样侧翻到刚写完的那一半
      gpu.parity ^= 1;
    }
  }
}

fn extract_ddgi_settings(
  mut commands: bevy::ecs::system::Commands,
  stage: Option<bevy::render::Extract<bevy::ecs::system::Res<DdgiStage>>>,
  debug: Option<bevy::render::Extract<bevy::ecs::system::Res<DdgiDebugSettings>>>,
) {
  let s = stage.map_or(DdgiStage::OFF, |s| s.0.min(DdgiStage::FULL));
  commands.insert_resource(DdgiStage(s));
  let dbg = debug.map_or_else(DdgiDebugSettings::default, |d| DdgiDebugSettings {
    mode: d.mode,
    gain: d.gain,
    probe_viz: d.probe_viz,
    probe_viz_lod: d.probe_viz_lod,
    borrow_radius: d.borrow_radius,
    depth_soft_k: d.depth_soft_k,
    far_ambient: d.far_ambient,
  });
  commands.insert_resource(dbg);
}

#[allow(clippy::too_many_arguments)]
fn prepare_ddgi(
  mut commands: bevy::ecs::system::Commands,
  device: bevy::ecs::system::Res<bevy::render::renderer::RenderDevice>,
  queue: bevy::ecs::system::Res<bevy::render::renderer::RenderQueue>,
  pipeline_cache: bevy::ecs::system::Res<bevy::render::render_resource::PipelineCache>,
  view: Option<bevy::ecs::system::Res<crate::brickmap::dda::DdaViewUniform>>,
  revision: Option<bevy::ecs::system::Res<crate::brickmap::upload::BrickMapRevision>>,
  dirty: Option<bevy::ecs::system::Res<crate::brickmap::upload::BrickMapDirty>>,
  stage: bevy::ecs::system::Res<DdgiStage>,
  dbg: bevy::ecs::system::Res<DdgiDebugSettings>,
  mut gpu: bevy::ecs::system::ResMut<DdgiGpu>,
) {
  // ---- 嵌套级联网格推导（相机中心 → 4 级 LOD 各自独立原点）----
  let cam = view
    .as_ref()
    .map(|v| v.cam_pos_voxel.truncate().floor().as_ivec3())
    .unwrap_or(IVec3::ZERO);
  let grid = DdgiWorldGrid::from_camera(cam);
  let total = grid.total_slots;
  let grid_changed = grid != gpu.grid;

  // ---- 推进网格/修订号状态（仅在本帧会跑 pass 时）----
  // 否则「关闭期间世界已更新/相机已移动」会被吞掉 → 之后打开时不会补烘。
  let rev = revision.map_or(0, |r| r.0);
  let will_run = stage.run_active() || dbg.probe_viz;
  let rev_changed = rev != gpu.last_revision;
  if will_run {
    gpu.grid = grid;
    gpu.total_slots = total;
    gpu.last_revision = rev;
  }
  // 相机滚动（grid 变化）→ 按「世界 cell 键」增量补烘；世界编辑 → 只失效脏区内的 cell
  // （bake 用 uniform 里的脏区 AABB 跳过 cell_id 键检查，不再整块清缓存）。
  // `gpu.bake_pending` 让请求**黏住**：本帧若因 pipeline 未编译好而没真正派发（见
  // dispatch_ddgi 的提前返回），下一帧仍会重试，而不是被「已推进的 grid/last_revision」
  // 悄悄吞掉（症状：启动时就打开 DDGI → 一次也不烘焙 → cast/collect 恒为 0.02ms）。
  let need_bake = (grid_changed || rev_changed || gpu.bake_pending) && total > 0 && will_run;
  gpu.bake_pending = need_bake;

  // ---- 脏区：本帧上传实际改动的世界 voxel AABB（max 不含）----
  // 全量上传 → ±1e9（等价于全部 cell 失效）；增量 → 改动 chunk 的合并包围盒。
  let (dirty_min, dirty_max, dirty_valid) = match dirty.as_ref() {
    Some(d) if d.full => (
      IVec3::splat(-1_000_000_000),
      IVec3::splat(1_000_000_000),
      1.0f32,
    ),
    Some(d) if d.min_voxel != d.max_voxel => (d.min_voxel, d.max_voxel, 1.0f32),
    _ => (IVec3::ZERO, IVec3::ZERO, 0.0f32),
  };

  // ---- 缓冲容量（槽数固定：4 LOD × 16³）----
  let s = total as u64;
  ensure_storage_buffer(&device, &queue, &mut gpu.cell, "ddgi_cell", s * 4);
  ensure_storage_buffer(&device, &queue, &mut gpu.meta, "ddgi_meta", s * 4);
  ensure_storage_buffer(&device, &queue, &mut gpu.slot_pos, "ddgi_slot_pos", s * 16);
  ensure_storage_buffer(&device, &queue, &mut gpu.worklist, "ddgi_worklist", s * 16);
  ensure_storage_buffer(&device, &queue, &mut gpu.cell_id, "ddgi_cell_id", s * 16);
  ensure_storage_buffer(&device, &queue, &mut gpu.samples, "ddgi_samples", DDGI_SAMPLE_BYTES);

  gpu.frame = gpu.frame.wrapping_add(1);

  // ---- uniform ----
  let mut u = DdgiUniform::default();
  for lod in 0..DDGI_LODS as usize {
    let cell = DDGI_LOD_CELL_SIZES[lod];
    let d = gpu.grid.lod_dims[lod];
    let o = gpu.grid.lod_origins[lod];
    u.lods[lod] = DdgiLod {
      origin: IVec4::new(o.x, o.y, o.z, cell),
      dims: UVec4::new(d.x, d.y, d.z, gpu.grid.lod_slot_base[lod]),
    };
  }
  // params: x=frame, y=debug mode, z=gain, w=借针半径（0=关闭，见 DdgiDebugSettings）
  u.params = Vec4::new(gpu.frame as f32, dbg.mode, dbg.gain, dbg.borrow_radius);
  u.misc = Vec4::new(
    if stage.shade_gi() { 1.0 } else { 0.0 },
    gpu.total_slots as f32,
    // z = Chebyshev std 信任系数（见 DdgiDebugSettings.depth_soft_k）
    dbg.depth_soft_k,
    // w = 级联覆盖外的天光兜底强度（见 DdgiDebugSettings.far_ambient）
    dbg.far_ambient,
  );
  u.dirty_min = Vec4::new(
    dirty_min.x as f32,
    dirty_min.y as f32,
    dirty_min.z as f32,
    dirty_valid,
  );
  u.dirty_max = Vec4::new(dirty_max.x as f32, dirty_max.y as f32, dirty_max.z as f32, 0.0);
  *gpu.uniform.get_mut() = u;
  gpu.uniform.write_buffer(&device, &queue);

  // 每帧清零活跃计数器（indirect words [36..40)）；indirect args 由 seal 当帧覆写。
  queue.write_buffer(
    &gpu.indirect,
    DDGI_COUNTER_CLEAR_OFFSET,
    &[0u8; DDGI_COUNTER_CLEAR_BYTES as usize],
  );

  // ---- BG4（图集采样侧 + 各 pass 共用缓冲）----
  // p = 采样侧：上一帧 collect 写入的那一半（cast 回读 + 着色采样都读它）。
  // 写入侧 1-p 只给 collect（BG5）。
  use bevy::render::render_resource::BindGroupEntries;
  let p = gpu.parity & 1;
  let bg4_layout = pipeline_cache.get_bind_group_layout(&ddgi_bg4_layout());
  let bg4 = device.create_bind_group(
    None,
    &bg4_layout,
    &BindGroupEntries::sequential((
      &gpu.uniform,
      &gpu.irr[p].view,
      &gpu.depth[p].view,
      gpu.cell.as_entire_binding(),
      gpu.meta.as_entire_binding(),
      gpu.indirect.as_entire_binding(),
      gpu.worklist.as_entire_binding(),
      gpu.slot_pos.as_entire_binding(),
      gpu.cell_id.as_entire_binding(),
      gpu.samples.as_entire_binding(),
      gpu.cell_slot.as_entire_binding(),
    )),
  );
  commands.insert_resource(DdgiBg4(bg4));

  // ---- BG5（collect 图集写入侧）----
  let bg5_layout = pipeline_cache.get_bind_group_layout(&ddgi_bg5_layout());
  let bg5 = device.create_bind_group(
    None,
    &bg5_layout,
    &BindGroupEntries::sequential((
      &gpu.irr[1 - p].view,
      &gpu.depth[1 - p].view,
    )),
  );
  commands.insert_resource(DdgiBg5(bg5));

  // ---- BG6（seal 的 dispatch 参数写入侧）----
  let bg6_layout = pipeline_cache.get_bind_group_layout(&ddgi_bg6_layout());
  let bg6 = device.create_bind_group(
    None,
    &bg6_layout,
    &BindGroupEntries::single(gpu.args.as_entire_binding()),
  );
  commands.insert_resource(DdgiBg6(bg6));
  commands.insert_resource(DdgiBakeThisFrame(need_bake));
}

#[cfg(test)]
mod tests {
  use super::*;

  /// 判定 LOD(l-1) 盒是否严格包含于 LOD(l) 盒（嵌套级联的核心不变式）。
  fn lod_aabb(g: &DdgiWorldGrid, lod: usize) -> (IVec3, IVec3) {
    let o = g.lod_origins[lod];
    let ext = IVec3::new(
      g.lod_dims[lod].x as i32,
      g.lod_dims[lod].y as i32,
      g.lod_dims[lod].z as i32,
    ) * DDGI_LOD_CELL_SIZES[lod];
    (o, o + ext)
  }

  #[test]
  fn cascade_is_nested_and_camera_centered() {
    let cam = IVec3::new(1000, -333, 7);
    let g = DdgiWorldGrid::from_camera(cam);
    // 4 级维度相同，cell 逐级 ×2 → 覆盖范围逐级 ×2
    for lod in 0..DDGI_LODS as usize {
      assert_eq!(g.lod_dims[lod], DDGI_LOD_DIMS);
    }
    // 相机落在每级盒内（居中，对齐误差 ≤ cell）
    for lod in 0..DDGI_LODS as usize {
      let (lo, hi) = lod_aabb(&g, lod);
      assert!(cam.cmpge(lo).all() && cam.cmplt(hi).all(), "lod {lod} 盒未包含相机");
    }
    // 相邻级严格嵌套：LOD(l-1) ⊂ LOD(l)
    for lod in 1..DDGI_LODS as usize {
      let (plo, phi) = lod_aabb(&g, lod - 1);
      let (lo, hi) = lod_aabb(&g, lod);
      assert!(plo.cmpge(lo).all() && phi.cmple(hi).all(), "lod {lod} 未包含 lod {}", lod - 1);
    }
    // 各级原点按自身 cell 对齐
    for lod in 0..DDGI_LODS as usize {
      let cell = DDGI_LOD_CELL_SIZES[lod];
      let o = g.lod_origins[lod];
      assert_eq!(o.x.rem_euclid(cell), 0);
      assert_eq!(o.y.rem_euclid(cell), 0);
      assert_eq!(o.z.rem_euclid(cell), 0);
    }
  }

  #[test]
  fn cascade_slot_layout_is_consistent() {
    let g = DdgiWorldGrid::from_camera(IVec3::ZERO);
    let per_lod = DDGI_LOD_DIMS.x * DDGI_LOD_DIMS.y * DDGI_LOD_DIMS.z;
    let mut acc = 0u32;
    for lod in 0..DDGI_LODS as usize {
      assert_eq!(g.lod_slot_base[lod], acc);
      assert_eq!(g.lod_count(lod), per_lod);
      acc += per_lod;
    }
    assert_eq!(g.total_slots, acc);
    assert!(!g.is_empty());
  }

  /// 世界锚定槽位映射的核心不变式：**同一个世界 cell，在相机移动前后映射到同一个槽位**
  /// （只要它还在窗口内）。shader 里 `ddgi_slot = slot_base + (世界 cell mod dims)` 就是
  /// 这个式子；这里用 Rust 复刻它，保证「槽位身份与世界绑定」这条性质不被 from_camera 改动
  /// （例如原点不再按 cell 对齐）悄悄破坏。
  #[test]
  fn world_cell_slot_identity_survives_camera_movement() {
    let slot_of = |g: &DdgiWorldGrid, lod: usize, wc: IVec3| -> u32 {
      let dims = g.lod_dims[lod].as_ivec3();
      let r = wc.rem_euclid(dims);
      g.lod_slot_base[lod]
        + (r.x + r.y * dims.x + r.z * dims.x * dims.y) as u32
    };
    // 同一世界 cell 在两处相机位置下都在窗口内 → 槽位必须相同
    let wc = IVec3::new(4096, 128, -2048);
    let g1 = DdgiWorldGrid::from_camera(wc + IVec3::new(0, 64, 0));
    let g2 = DdgiWorldGrid::from_camera(wc + IVec3::new(48, 96, -32));
    for lod in 0..DDGI_LODS as usize {
      assert_eq!(
        slot_of(&g1, lod, wc),
        slot_of(&g2, lod, wc),
        "lod {lod}: 相移动后同一世界 cell 的槽位变了"
      );
    }
    // 相机恒在盒中心（偏差 < 1 个 cell）：这条保证「原点 = 相机 - half·cell 后按 cell 对齐」
    // 不会退化，从而 shader 的「原点 / cell」整除成立。
    for cam in [IVec3::new(0, 0, 0), IVec3::new(-1, -7, 12345), wc] {
      let g = DdgiWorldGrid::from_camera(cam);
      for lod in 0..DDGI_LODS as usize {
        let cell = DDGI_LOD_CELL_SIZES[lod];
        let center = g.lod_origins[lod]
          + g.lod_dims[lod].as_ivec3() / 2 * cell;
        let off = (cam - center).abs();
        assert!(
          off.cmplt(IVec3::splat(cell)).all(),
          "lod {lod}: 相机偏离盒心 {off:?} ≥ 1 cell"
        );
      }
    }
  }
}
