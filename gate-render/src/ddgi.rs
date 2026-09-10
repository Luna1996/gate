//! DDGI 全局光照（Douglas Devlog #23 1:1 复刻）——全 GPU 烘焙 + 相机 LOD 滚动级联。
//!
//! ## 架构（2026-09-09 重做：CPU 零烘焙/零几何遍历）
//! - 探针放置每帧由 GPU active pass 遍历 con tree 现算（三态 Air/Mixed/Solid +
//!   BFS 最大空叶 offset）；CPU 不遍历体素树、不判 cell 三态、不摆探针。
//! - 槽身份 = 固定 (lod, cell_in_window)：slot = lod·4096 + (x + y·16 + z·256)，
//!   共 4 级 LOD × 4096 槽 = 16384 槽 = 64 纹理层。
//! - 4 级窗口以相机为锚、按半窗 8 cell 滚动（cell 边长 16/32/64/128 fine）；
//!   重叠带内 offset 匹配的探针沿用上帧 age（reuse bounds 见 [`LodWindow`]）。
//! - CPU 每帧只做：ping-pong 交换、由相机 fine 坐标算 4 级窗口 uniform、建 BG4。
//!
//! 载体 = 2D 纹理数组（irr rgba16f 128×128 / depth r32f 256×256 / meta r32uint
//! 16×16，各 64 层）双缓冲 ping-pong；光照数学（active 判定 / 射线投射 /
//! 投影更新 / 采样）全部在 WGSL 侧。
//!
//! ## WGSL 对齐表（改一处必改两处；wire 单测防漂移）
//! | Rust | WGSL | 值 | 语义 |
//! |---|---|---|---|
//! | IRRADIANCE_TEXELS | IRRADIANCE_TEXELS | 8 | irr oct 边长 |
//! | DEPTH_TEXELS | DEPTH_TEXELS | 16 | depth oct 边长 |
//! | PROBES_PER_LAYER_AXIS | PROBES_PER_LAYER_AXIS | 16 | 层内单轴探针数 |
//! | IRRADIANCE_LAYER_TEXELS | IRRADIANCE_LAYER_TEXELS | 128 | irr 层边长 |
//! | DEPTH_LAYER_TEXELS | DEPTH_LAYER_TEXELS | 256 | depth 层边长 |
//! | DDGI_LODS / DDGI_LOD_CELL_SIZES | DDGI_LODS / DDGI_LOD_CELL_SIZES | 4 / 16,32,64,128 | LOD 级数 / 各级 cell 边长 |
//! | SLOTS_PER_LOD / DDGI_TOTAL_SLOTS | 同名 | 4096 / 16384 | 每级槽数 / 总槽数 |
//! | DDGI_TOTAL_LAYERS | 同名 | 64 | 纹理数组总层数 |
//! | META_AGE_SHIFT | META_AGE_SHIFT | 15 | 元数据 age 位移 |
//! | PROBE_T_MAX | PROBE_T_MAX | 8192.0 | 射线远距 / depth 初值 |
//! | DDGI_RAY_BUDGET / DDGI_PROBE_BUDGET | 同名 | 65536 / 4096 | 射线预算 / 每帧探针钳制 |

use bevy::render::render_resource::ShaderType;
use glam::{IVec3, IVec4, UVec3, Vec3, Vec4};

// ============================================================================
// 常量（WGSL 侧逐字对齐）
// ============================================================================

/// 八面体 irradiance 边长（Majercik 2019 §3：8×8 texel/探针）
pub const IRRADIANCE_TEXELS: u32 = 8;
/// 八面体 depth 边长（Majercik 2019 §3：16×16 texel/探针）
pub const DEPTH_TEXELS: u32 = 16;
/// 探针射线 t_max / depth 纹理初值（世界窗口对角量级）
pub const PROBE_T_MAX: f32 = 8192.0;
/// Douglas Devlog #23：总射线数固定预算（摊派到本帧活跃探针）
pub const DDGI_RAY_BUDGET: u32 = 65536;
/// 每帧更新探针钳制上限（WGSL DDGI_PROBE_BUDGET 镜像；seal pass min(count, 此值)）
pub const DDGI_PROBE_BUDGET: u32 = 4096;

/// 层内单轴探针数（纹理数组每层 = 16×16 探针的 oct 图块网格）
pub const PROBES_PER_LAYER_AXIS: u32 = 16;
/// 每层探针数（16×16 = 256）
pub const PROBES_PER_LAYER: u32 = PROBES_PER_LAYER_AXIS * PROBES_PER_LAYER_AXIS;
/// irradiance 层边长（texel）= 16 探针 × 8 oct texel
pub const IRRADIANCE_LAYER_TEXELS: u32 = PROBES_PER_LAYER_AXIS * IRRADIANCE_TEXELS;
/// depth 层边长（texel）= 16 探针 × 16 oct texel
pub const DEPTH_LAYER_TEXELS: u32 = PROBES_PER_LAYER_AXIS * DEPTH_TEXELS;
/// LOD 级数（Douglas #23：four LODs）
pub const DDGI_LODS: u32 = 4;
/// 元数据 packed u32 中 age 的位移（offset 5bit/轴 ×3；age 在 [15..23)）
pub const META_AGE_SHIFT: u32 = 15;
/// meta 拷贝行距：wgpu COPY_BYTES_PER_ROW_ALIGNMENT = 256B。meta 行宽
/// 16 texel × 4B = 64B 不达标——所有 buffer↔meta 纹理拷贝必须按 256B 行距排布
pub const META_COPY_BPR: u32 = 256;
/// meta 拷贝行距对应的 word 数（256B / 4 = 64 word/行：前 16 word 数据 + pad）
const META_ROW_WORDS: usize = META_COPY_BPR as usize / 4;

// ============================================================================
// 相机滚动 LOD 窗口（Douglas #23 sort.glsl dispatch.base_position / reuse_bounds
// 镜像）：4 级 LOD，cell_size = [16,32,64,128]，每级 16³ cell 窗口以相机为锚、
// 随相机按半窗（8 cell）步进滚动；重叠带内 normalized offset 匹配的探针沿用 age。
// 纯算术、零几何遍历（CPU 不做任何烘焙）——每帧只由相机位置算窗口角与复用界。
// ============================================================================

/// 新架构 4 级 LOD cell 边长（fine 体素）
pub const DDGI_LOD_CELL_SIZES: [i32; DDGI_LODS as usize] = [16, 32, 64, 128];
/// 滚动级单轴 cell 数（论文默认 16³ 探针/级；与纹理层 16×16 对齐 → 每级 16 层）
pub const PROBES_PER_CASCADE_AXIS: u32 = 16;
/// 每级 LOD 的固定探针槽数（16³ cell 窗口 = 4096）
pub const SLOTS_PER_LOD: u32 =
  PROBES_PER_CASCADE_AXIS * PROBES_PER_CASCADE_AXIS * PROBES_PER_CASCADE_AXIS;
/// 全部 LOD 总槽数（4×4096 = 16384）
pub const DDGI_TOTAL_SLOTS: u32 = SLOTS_PER_LOD * DDGI_LODS;
/// 纹理数组总层数（每 256 槽/层 = 16×16；16384/256 = 64 层）
pub const DDGI_TOTAL_LAYERS: u32 = DDGI_TOTAL_SLOTS / PROBES_PER_LAYER;

/// 固定槽编码：slot = lod·4096 + (x + y·16 + z·256)，cell ∈ [0,16) 窗口内坐标。
/// 槽身份 = (lod, cell_in_window)，与探针世界位置/几何无关——滚动只改窗口原点，
/// 槽位不动，age 随 ping-pong 沿用。
#[inline]
pub fn ddgi_slot(lod: u32, cell: UVec3) -> u32 {
  let a = PROBES_PER_CASCADE_AXIS;
  lod * SLOTS_PER_LOD + cell.x + cell.y * a + cell.z * a * a
}

/// 窗口单轴 cell 数（= PROBES_PER_CASCADE_AXIS）
pub const DDGI_WINDOW_AXIS: i32 = PROBES_PER_CASCADE_AXIS as i32;
/// 滚动步长（cell）：半窗。相机在窗内 [BAND_LO, BAND_LO+STEP) = [4,12) 死区带
pub const DDGI_SCROLL_STEP: i32 = 8;
/// 相机在窗口内的最小 cell 下标（窗 16 → 死区 [4,12)，两侧各留 4 cell 余量）
pub const DDGI_BAND_LO: i32 = 4;

/// 单级 LOD 窗口描述（坐标均为 **本级 cell 单位** 的全局整数坐标；fine = ×cell_size）
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct LodWindow {
  pub cell_size: i32,
  /// 当前帧窗口角（本级 cell 全局坐标）
  pub origin: IVec3,
  /// 上一帧窗口角（ping-pong previous meta 寻址用）
  pub prev_origin: IVec3,
  /// 复用界（本级 cell 全局坐标）：current 与 previous 窗口重叠 [reuse_min, reuse_max)
  pub reuse_min: IVec3,
  pub reuse_max: IVec3,
}

/// 相机 fine 坐标 → 窗口角（本级 cell 全局坐标）。令相机相对窗口角落在
/// [BAND_LO, BAND_LO+STEP) = [4,12)：origin = STEP·⌊cam_cell/STEP⌋ − BAND_LO
/// （euclid 除法保证负坐标正确）。
#[inline]
pub fn lod_window_origin(cam_fine: Vec3, cell_size: i32) -> IVec3 {
  let cam_cell = (cam_fine / cell_size as f32).floor().as_ivec3();
  let f = |c: i32| c.div_euclid(DDGI_SCROLL_STEP) * DDGI_SCROLL_STEP - DDGI_BAND_LO;
  IVec3::new(f(cam_cell.x), f(cam_cell.y), f(cam_cell.z))
}

/// 计算单级窗口（含与上一帧的复用界）。`have_prev=false`（首帧）→ 复用界为空。
#[inline]
pub fn compute_lod_window(
  cam_fine: Vec3,
  cell_size: i32,
  prev_origin: IVec3,
  have_prev: bool,
) -> LodWindow {
  let origin = lod_window_origin(cam_fine, cell_size);
  let n = DDGI_WINDOW_AXIS;
  let (reuse_min, reuse_max) = if have_prev {
    let lo = origin.max(prev_origin);
    let hi = (origin + n).min(prev_origin + n);
    (lo, hi)
  } else {
    // 空复用界：WGSL all(cell>=min)&&all(cell<max) 在 min==max 时恒 false
    (origin, origin)
  };
  LodWindow {
    cell_size,
    origin,
    prev_origin,
    reuse_min,
    reuse_max,
  }
}

// ============================================================================
// 渲染侧：BG4 布局 + GPU 资源 + 每帧 prepare/dispatch（DdgiPlugin）
// ============================================================================

use bevy::asset::AssetServer;
use bevy::ecs::schedule::IntoScheduleConfigs;
use bevy::ecs::system::{Commands, Res, ResMut};
use bevy::prelude::RenderGraph;
use bevy::render::{
  Render, RenderApp, RenderStartup, RenderSystems,
  render_resource::{
    BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType,
    Buffer, BufferBindingType, BufferDescriptor, BufferUsages, CachedComputePipelineId,
    ComputePipelineDescriptor, Extent3d, MapMode, Origin3d, ShaderStages, StorageTextureAccess,
    TexelCopyBufferInfo, TexelCopyBufferLayout, TexelCopyTextureInfo, Texture, TextureAspect,
    TextureDescriptor, TextureDimension, TextureFormat, TextureSampleType, TextureUsages,
    TextureView, TextureViewDescriptor, TextureViewDimension, UniformBuffer,
  },
  renderer::{RenderDevice, RenderQueue},
};
use std::borrow::Cow;

/// 单级 LOD 窗口 uniform（dda.wgsl `DdgiLod` 逐字镜像，64B）
#[repr(C)]
#[derive(Debug, Default, Clone, Copy, ShaderType)]
pub struct DdgiLod {
  /// xyz = 窗口角「本级 cell 全局坐标」，w = cell_size（fine）
  pub origin: IVec4,
  /// xyz = 上一帧窗口角，w = pad
  pub prev_origin: IVec4,
  /// xyz = 复用界 min（含），w = pad
  pub reuse_min: IVec4,
  /// xyz = 复用界 max（不含），w = pad
  pub reuse_max: IVec4,
}

/// BG4 uniform（dda.wgsl `DdgiUniform` 逐字镜像）：4 级相机锚定滚动窗口 +
/// params = (frame, 调试 mode, gain, object bbox 数) + misc.x = DDGI 开关。
#[repr(C)]
#[derive(Debug, Default, Clone, Copy, bevy::ecs::resource::Resource, ShaderType)]
pub struct DdgiUniform {
  pub lods: [DdgiLod; 4],
  pub params: Vec4,
  pub misc: Vec4,
}

/// BG4 布局（dda.wgsl group(4) 12 binding 逐字镜像；改 shader 必同步此处）：
/// 0=uniform(DdgiUniform)
/// 1=irr_prev（texture_2d_array<f32> rgba16f，Float filterable）
/// 2=depth_prev（texture_2d_array<f32> r32f，Float 不可滤波）
/// 3=irr_next（storage rgba16float write）
/// 4=depth_next（storage r32float write）
/// 5=meta_prev（texture_2d_array<u32> r32uint）
/// 6=meta_next（storage r32uint write）
/// 7=dispatch（storage rw atomic<u32>：[0]=本帧活跃计数，[1]=seal 钳制值）
/// 8=objects（storage ro 物体 bbox；暂为空，params.w=0）
/// 9=samples（storage rw 射线样本）
/// 10=worklist（storage rw 本帧活跃 slot）
/// 11=slot_pos（storage rw slot→探针世界 xyz + w=lod）
pub fn ddgi_bg4_layout() -> BindGroupLayoutDescriptor {
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
      tex(1, TextureSampleType::Float { filterable: true }), // irr prev（rgba16f）
      tex(2, TextureSampleType::Float { filterable: false }), // depth prev（r32f）
      store(3, TextureFormat::Rgba16Float),                  // irr next
      store(4, TextureFormat::R32Float),                     // depth next
      tex(5, TextureSampleType::Uint),                       // meta prev（r32uint）
      store(6, TextureFormat::R32Uint),                      // meta next
      buf(7, false),                                         // dispatch（rw atomic）
      buf(8, true),                                          // objects（ro bbox）
      buf(9, false),                                         // samples（rw）
      buf(10, false),                                        // worklist（rw）
      buf(11, false),                                        // slot_pos（rw）
    ],
  )
}

/// DDGI compute 管线（dda.wgsl 五 entry；布局 = BG0-3（dda 复用）+ BG4）
#[derive(Debug, Clone, Copy)]
pub struct DdgiPipelines {
  pub clear: CachedComputePipelineId,
  pub active: CachedComputePipelineId,
  pub seal: CachedComputePipelineId,
  pub cast: CachedComputePipelineId,
  pub update: CachedComputePipelineId,
}

/// 渲染 world 持久 DDGI GPU 资源（RenderStartup 一次性建固定 64 层资源；
/// D7 双缓冲 ping-pong：prev 采样读 / next 更新写，帧末交换指针）
#[derive(bevy::ecs::resource::Resource)]
pub struct DdgiGpu {
  pub uniform: UniformBuffer<DdgiUniform>,
  /// 物体 bbox（storage ro；暂为空，params.w=0；runtime-sized 数组最小 1 元素）
  pub objects: Buffer,
  /// [count, seal, 0, 0]；[0] 由 ddgi_active atomicAdd、ddgi_clear 清零。
  /// **不能**兼 indirect——同 dispatch scope 内 STORAGE 与 INDIRECT 互斥（wgpu 验证）
  pub dispatch: Buffer,
  /// indirect 专用（INDIRECT + COPY_DST）：seal 后由 encoder 从 dispatch[1] 桥接
  pub indirect: Buffer,
  /// 本帧活跃 slot（atomicAdd 序号；DDGI_TOTAL_SLOTS 字）
  pub worklist: Buffer,
  /// 射线样本（DDGI_RAY_BUDGET×2 vec4：dir + radiance）
  pub samples: Buffer,
  /// slot → xyz=探针世界 fine 坐标，w=lod（active 写、cast/viz/sample 读）
  pub slot_pos: Buffer,
  pub irr_prev: Texture,
  pub irr_next: Texture,
  pub depth_prev: Texture,
  pub depth_next: Texture,
  pub meta_prev: Texture,
  pub meta_next: Texture,
  pub irr_prev_view: TextureView,
  pub irr_next_view: TextureView,
  pub depth_prev_view: TextureView,
  pub depth_next_view: TextureView,
  pub meta_prev_view: TextureView,
  pub meta_next_view: TextureView,
  /// 已推进帧号（prepare 自增）
  pub frame: u32,
  /// 上一帧各 LOD 窗口角（本级 cell 全局坐标；首帧 have_prev=false）
  pub prev_origins: [IVec3; 4],
  /// 是否已有上一帧窗口（false → 复用界为空）
  pub have_prev: bool,
  /// 五 entry 管线（DdaPipelines 就绪后排队一次）
  pub pipelines: Option<DdgiPipelines>,
  /// 诊断回读（每 120 帧三阶段：copy → map_async → 读+unmap；wgpu 规则：
  /// submit 时 buffer 不得处于 Pending/Mapped，三阶段保证每次 submit 时 Unmapped）
  pub readback: Buffer,
  pub readback_rx: std::sync::Mutex<Option<std::sync::mpsc::Receiver<()>>>,
  /// 0=idle 1=copied 2=mapped(等回调)
  pub readback_state: u32,
  /// 诊断回读采样层距（copy 时定格 = DDGI_TOTAL_LAYERS/8；解析侧据此还原层号）
  pub readback_step: u32,
}

#[derive(bevy::ecs::resource::Resource)]
pub struct DdgiBg4(pub BindGroup);

/// DDGI 运行时分阶段档位（main world 由 DebugView 档位 slider 写入；extract 拷到
/// render world）。三阶段逐级依赖，高档位包含所有低档位工作：
/// - [`DdgiStage::OFF`]（0）：全关。dispatch 链整体早退（省 GPU），trace shader
///   misc.x=0 跳过探针采样（gi=0）。
/// - [`DdgiStage::ACTIVE`]（1）：① 计算 Active Probe（copy/clear/active/seal）。
///   probe worklist/meta 产出，Probe Viz 可看探针放置结果；无射线、纹理不更新。
/// - [`DdgiStage::CAST`]（2）：② + 从 Active Probe 发射 RayQuery（cast）并把样本
///   投影积分进 irradiance/depth 纹理（update）；探针数据有效，但 voxel 着色仍不
///   采样（misc.x=0），可用调试模式可视化探针数据而不影响画面。
/// - [`DdgiStage::FULL`]（3）：③ + voxel 着色采样探针（trace shader misc.x=1，
///   完整 DDGI 间接光）。
///
/// 默认 0（关）。
#[derive(bevy::ecs::resource::Resource, Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct DdgiStage(pub u8);

impl DdgiStage {
  /// 0：全关
  pub const OFF: u8 = 0;
  /// 1：仅 Active Probe 计算
  pub const ACTIVE: u8 = 1;
  /// 2：Active + RayQuery 投射/积分
  pub const CAST: u8 = 2;
  /// 3：完整 DDGI（voxel 着色采样探针）
  pub const FULL: u8 = 3;

  /// 钳制到合法档位 [0, 3]
  pub fn new(v: u8) -> Self {
    Self(v.min(Self::FULL))
  }
  /// ① 是否运行 Active Probe 计算（档位 ≥ 1）
  pub fn run_active(&self) -> bool {
    self.0 >= Self::ACTIVE
  }
  /// ② 是否运行 RayQuery 投射 + 纹理积分（档位 ≥ 2）
  pub fn run_cast(&self) -> bool {
    self.0 >= Self::CAST
  }
  /// ③ voxel 着色是否采样探针（档位 ≥ 3）
  pub fn shade_gi(&self) -> bool {
    self.0 >= Self::FULL
  }
}

/// DDGI 运行时调试参数（主世界 DebugView UI 写入 → extract 拷到渲染世界 →
/// prepare_ddgi 每帧写进 DdgiUniform.params.y/.z）。
/// - mode：params.y，0=正常，1=GI 提亮，2=wsum 热度，3=选域 id，4=探针状态
/// - gain：params.z，诊断增益（调参/链路定位用）
/// - probe_viz：探针位置可视化开关（devlog #23 风格黄色方块）
#[derive(bevy::ecs::resource::Resource, Clone, Copy, Debug, PartialEq)]
pub struct DdgiDebugSettings {
  pub mode: f32,
  pub gain: f32,
  pub probe_viz: bool,
  /// probe viz 层级选择（0=全部, 1..=4=LOD0~3；新架构无 base 世界级网格）
  pub probe_viz_lod: f32,
}

impl Default for DdgiDebugSettings {
  fn default() -> Self {
    Self {
      mode: 0.0,
      gain: 1.0,
      probe_viz: false,
      probe_viz_lod: 0.0,
    }
  }
}

/// DDGI 插件：RenderStartup 一次性建固定 GPU 资源 + 管线；每帧 prepare（窗口
/// uniform + BG4）→ dispatch（五 pass 链在主 trace 之前编码）。
pub struct DdgiPlugin;

impl bevy::app::Plugin for DdgiPlugin {
  fn build(&self, app: &mut bevy::app::App) {
    app.init_resource::<DdgiStage>();
    app.init_resource::<DdgiDebugSettings>();
    let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
      return;
    };
    render_app
      .init_resource::<DdgiStage>()
      .init_resource::<DdgiDebugSettings>()
      .add_systems(RenderStartup, init_ddgi_gpu)
      // DDGI pipeline 排队必须在 DDA pipeline 之后（需要 dda.bg0-3 layouts）
      .add_systems(
        RenderStartup,
        queue_ddgi_pipelines.after(crate::brickmap::dda::init_dda_pipelines),
      )
      .add_systems(bevy::render::ExtractSchedule, extract_ddgi_settings)
      .add_systems(
        Render,
        prepare_ddgi.in_set(RenderSystems::PrepareBindGroups),
      )
      // 四 pass 链在主 trace（dispatch_dda）之前编码
      .add_systems(
        RenderGraph,
        dispatch_ddgi
          .in_set(bevy::render::renderer::RenderGraphSystems::Render)
          .before(crate::brickmap::dda::dispatch_dda),
      );
  }
}

/// 定长 storage buffer（usage = STORAGE|COPY_DST|COPY_SRC；wgpu 创建即零初始化）
fn dummy_sized_buffer(device: &RenderDevice, label: &str, size: u64) -> Buffer {
  device.create_buffer(&BufferDescriptor {
    label: Some(label.into()),
    size: size.max(4),
    usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
    mapped_at_creation: false,
  })
}

/// 定长 storage buffer 并显式零填（slot_pos / worklist 等内容敏感缓冲）
fn zero_storage_buffer(
  device: &RenderDevice,
  queue: &RenderQueue,
  label: &str,
  size: u64,
) -> Buffer {
  let buf = dummy_sized_buffer(device, label, size);
  queue.write_buffer(&buf, 0, &vec![0u8; size as usize]);
  buf
}

/// indirect dispatch 专用 buffer（INDIRECT + COPY_DST；与 storage 绑定互斥故独立）
fn dummy_indirect_buffer(device: &RenderDevice, label: &str) -> Buffer {
  device.create_buffer(&BufferDescriptor {
    label: Some(label.into()),
    size: 16,
    usage: BufferUsages::INDIRECT | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
    mapped_at_creation: false,
  })
}

/// 诊断 staging（16B dispatch + 16B indirect + 64B worklist + 8 irr 采样层 +
/// 1 depth 采样层 + 8 meta 采样层——黑探针普查用：逐探针 age × irr 非零分类）；
/// MAP_READ + COPY_DST
fn ddgi_readback_buffer(device: &RenderDevice) -> Buffer {
  device.create_buffer(&BufferDescriptor {
    label: Some("ddgi_readback".into()),
    size: 256u64
      + (8 * IRRADIANCE_LAYER_TEXELS * IRRADIANCE_LAYER_TEXELS * 8) as u64
      + (DEPTH_LAYER_TEXELS * DEPTH_LAYER_TEXELS * 4) as u64
      // meta 8 采样层：256B 行距 × 16 行/层（COPY_BYTES_PER_ROW_ALIGNMENT，
      // 行宽 64B 不达标 → Validation Error）
      + (8 * (META_COPY_BPR as u64 * PROBES_PER_LAYER_AXIS as u64))
      + 64,
    usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
    mapped_at_creation: false,
  })
}

fn ddgi_array_view(tex: &Texture) -> TextureView {
  tex.create_view(&TextureViewDescriptor {
    dimension: Some(TextureViewDimension::D2Array),
    ..Default::default()
  })
}

/// 纹理数组（usage = 采样读 + storage 写 + 拷贝读写；ping-pong 双份）
fn ddgi_array_tex(
  device: &RenderDevice,
  label: &str,
  format: TextureFormat,
  size: (u32, u32),
  layers: u32,
) -> Texture {
  device.create_texture(&TextureDescriptor {
    label: Some(label.into()),
    size: Extent3d {
      width: size.0,
      height: size.1,
      depth_or_array_layers: layers,
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

/// RenderStartup：一次性建固定 64 层资源（新架构无烘焙后重建——纹理/buffer 尺寸
/// 恒定 = 4 LOD × 4096 槽；meta/irr 靠 wgpu 零初始化，depth 两份填 PROBE_T_MAX）
fn init_ddgi_gpu(mut commands: Commands, device: Res<RenderDevice>, queue: Res<RenderQueue>) {
  // ---- 6 张纹理数组（irr 128² rgba16f / depth 256² r32f / meta 16² r32uint，各 64 层）----
  let mk_pair = |label: &str, format: TextureFormat, size: (u32, u32)| {
    let t0 = ddgi_array_tex(
      &device,
      &format!("{label}_a"),
      format,
      size,
      DDGI_TOTAL_LAYERS,
    );
    let t1 = ddgi_array_tex(
      &device,
      &format!("{label}_b"),
      format,
      size,
      DDGI_TOTAL_LAYERS,
    );
    let v0 = ddgi_array_view(&t0);
    let v1 = ddgi_array_view(&t1);
    (t0, v0, t1, v1)
  };
  let (irr_a, irr_av, irr_b, irr_bv) = mk_pair(
    "ddgi_irr",
    TextureFormat::Rgba16Float,
    (IRRADIANCE_LAYER_TEXELS, IRRADIANCE_LAYER_TEXELS),
  );
  let (dep_a, dep_av, dep_b, dep_bv) = mk_pair(
    "ddgi_depth",
    TextureFormat::R32Float,
    (DEPTH_LAYER_TEXELS, DEPTH_LAYER_TEXELS),
  );
  let (meta_a, meta_av, meta_b, meta_bv) = mk_pair(
    "ddgi_meta",
    TextureFormat::R32Uint,
    (PROBES_PER_LAYER_AXIS, PROBES_PER_LAYER_AXIS),
  );

  // depth prev/next 全 64 层初始化 PROBE_T_MAX（256×256×64 f32；行距 256×4B）；
  // meta/irr 靠 wgpu 零初始化即可（age=0 = 未激活，采样端门控）
  let dep_data = f32_bytes(&vec![
    PROBE_T_MAX;
    (DEPTH_LAYER_TEXELS * DEPTH_LAYER_TEXELS * DDGI_TOTAL_LAYERS)
      as usize
  ]);
  for tex in [&dep_a, &dep_b] {
    queue.write_texture(
      TexelCopyTextureInfo {
        texture: tex,
        mip_level: 0,
        origin: Origin3d::ZERO,
        aspect: TextureAspect::All,
      },
      &dep_data,
      TexelCopyBufferLayout {
        offset: 0,
        bytes_per_row: Some(DEPTH_LAYER_TEXELS * 4),
        rows_per_image: Some(DEPTH_LAYER_TEXELS),
      },
      Extent3d {
        width: DEPTH_LAYER_TEXELS,
        height: DEPTH_LAYER_TEXELS,
        depth_or_array_layers: DDGI_TOTAL_LAYERS,
      },
    );
  }

  // ---- 固定尺寸 buffer（槽身份恒定 = 16384，无烘焙后重建）----
  // slot_pos：16384 × vec4（xyz=探针世界 fine，w=lod）
  let slot_pos = zero_storage_buffer(
    &device,
    &queue,
    "ddgi_slot_pos",
    DDGI_TOTAL_SLOTS as u64 * 16,
  );
  // worklist：16384 × u32 slot
  let worklist = zero_storage_buffer(
    &device,
    &queue,
    "ddgi_worklist",
    DDGI_TOTAL_SLOTS as u64 * 4,
  );
  // samples：65536 射线 × 2（dir + radiance）× vec4
  let samples = zero_storage_buffer(
    &device,
    &queue,
    "ddgi_samples",
    DDGI_RAY_BUDGET as u64 * 2 * 16,
  );
  // dispatch：[count, seal, 0, 0]（[0] 每帧 clear 清零，这里再零填兜底）
  let dispatch = zero_storage_buffer(&device, &queue, "ddgi_dispatch", 16);
  // indirect：[x=seal 桥接覆写, y=1, z=1, pad=0]——y/z 一次性常驻
  let indirect = dummy_indirect_buffer(&device, "ddgi_indirect");
  queue.write_buffer(&indirect, 4, &1u32.to_le_bytes());
  queue.write_buffer(&indirect, 8, &1u32.to_le_bytes());
  // objects：空 bbox 数组占位（params.w=0；runtime-sized vec4 数组最小 1 元素 = 16B）
  let objects = dummy_sized_buffer(&device, "ddgi_objects", 16);

  commands.insert_resource(DdgiGpu {
    uniform: UniformBuffer::default(),
    objects,
    dispatch,
    indirect,
    worklist,
    samples,
    slot_pos,
    irr_prev: irr_a,
    irr_next: irr_b,
    depth_prev: dep_a,
    depth_next: dep_b,
    meta_prev: meta_a,
    meta_next: meta_b,
    irr_prev_view: irr_av,
    irr_next_view: irr_bv,
    depth_prev_view: dep_av,
    depth_next_view: dep_bv,
    meta_prev_view: meta_av,
    meta_next_view: meta_bv,
    frame: 0,
    prev_origins: [IVec3::ZERO; 4],
    have_prev: false,
    pipelines: None,
    readback: ddgi_readback_buffer(&device),
    readback_rx: std::sync::Mutex::new(None),
    readback_state: 0,
    readback_step: 1,
  });
}

/// 五 entry 管线排队（一次；BG0-3 复用 dda 布局 + BG4；layout 收 Descriptor）
fn queue_ddgi_pipelines(
  dda: Option<Res<crate::brickmap::dda::DdaPipelines>>,
  pipeline_cache: Res<bevy::render::render_resource::PipelineCache>,
  asset_server: Res<AssetServer>,
  mut gpu: ResMut<DdgiGpu>,
) {
  if gpu.pipelines.is_some() {
    return;
  }
  let Some(dda) = dda else {
    return;
  };
  let layout = vec![
    dda.bg0_layout.clone(),
    dda.bg1_layout.clone(),
    dda.bg2_layout.clone(),
    dda.bg3_layout.clone(),
    ddgi_bg4_layout(),
  ];
  let shader = asset_server.load(crate::brickmap::dda::DDA_SHADER_ASSET_PATH);
  let mk = |label: &'static str, entry: &'static str| {
    pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
      label: Some(Cow::from(label)),
      layout: layout.clone(),
      shader: shader.clone(),
      entry_point: Some(Cow::from(entry)),
      ..Default::default()
    })
  };
  gpu.pipelines = Some(DdgiPipelines {
    clear: mk("gate_ddgi_clear", "ddgi_clear"),
    active: mk("gate_ddgi_active", "ddgi_active"),
    seal: mk("gate_ddgi_seal", "ddgi_seal"),
    cast: mk("gate_ddgi_cast", "ddgi_cast"),
    update: mk("gate_ddgi_update", "ddgi_update"),
  });
}

/// 五 pass 编排（在 dispatch_dda 主 trace 之前执行）：
/// copy（prev→next 全 64 层重填，encoder 级无 pass 冲突）→ ①clear → ②active
/// （4×4×16 wg 覆盖 16×16 cell × 64 层）→ ③seal（1 wg 钳制）→ 桥接 copy →
/// ④cast（indirect）→ ⑤update（indirect）。
/// race 纪律：pass 间读写同 storage/texture 必须分 pass（wgpu 只在 pass 边界插
/// barrier）；clear→active 同 buffer 亦然。
#[allow(clippy::too_many_arguments)]
fn dispatch_ddgi(
  mut ctx: bevy::render::renderer::RenderContext,
  bg0: Option<Res<crate::brickmap::dda::DdaBg0BindGroup>>,
  bg1: Option<Res<crate::brickmap::dda::DdaBg1BindGroup>>,
  bg2: Option<Res<crate::brickmap::dda::DdaBg2BindGroup>>,
  bg3: Option<Res<crate::brickmap::dda::DdaBg3BindGroup>>,
  bg4: Option<Res<DdgiBg4>>,
  stage: Res<DdgiStage>,
  mut gpu: ResMut<DdgiGpu>,
  pipeline_cache: Res<bevy::render::render_resource::PipelineCache>,
  mut profiler: ResMut<crate::profiler::GpuProfilerRes>,
) {
  // 档位 0：整条 compute 链早退（省 GPU）。trace shader 侧由 misc.x 开关位 gi=0。
  // 档位 1：只跑 copy/clear/active/seal（① Active Probe）；
  // 档位 2：加跑桥接 copy/cast/update（② RayQuery + 积分）；
  // 档位 3：compute 同档位 2，voxel 着色采样由 prepare 写 misc.x=1 开启（③）。
  if !stage.run_active() {
    return;
  }
  if gpu.frame <= 5 {
    bevy::log::info!(
      "DISP_DDGI entry: frame={} has_bg4={} has_pipes={}",
      gpu.frame,
      bg4.is_some(),
      gpu.pipelines.is_some()
    );
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
  let (Some(p_clear), Some(p_active), Some(p_seal), Some(p_cast), Some(p_update)) = (
    pipeline_cache.get_compute_pipeline(pipes.clear),
    pipeline_cache.get_compute_pipeline(pipes.active),
    pipeline_cache.get_compute_pipeline(pipes.seal),
    pipeline_cache.get_compute_pipeline(pipes.cast),
    pipeline_cache.get_compute_pipeline(pipes.update),
  ) else {
    return;
  };

  // ---- copy：全 64 层 prev→next 三对整体重填（irr 128² / depth 256² / meta 16²）。
  // 本帧 prev = 上一帧 update 产物；next 经 copy 重填后只被本帧处理探针覆写。
  {
    let encoder = ctx.command_encoder();
    for (src, dst, size) in [
      (&gpu.irr_prev, &gpu.irr_next, IRRADIANCE_LAYER_TEXELS),
      (&gpu.depth_prev, &gpu.depth_next, DEPTH_LAYER_TEXELS),
      (&gpu.meta_prev, &gpu.meta_next, PROBES_PER_LAYER_AXIS),
    ] {
      encoder.copy_texture_to_texture(
        TexelCopyTextureInfo {
          texture: src,
          mip_level: 0,
          origin: Origin3d::ZERO,
          aspect: TextureAspect::All,
        },
        TexelCopyTextureInfo {
          texture: dst,
          mip_level: 0,
          origin: Origin3d::ZERO,
          aspect: TextureAspect::All,
        },
        Extent3d {
          width: size,
          height: size,
          depth_or_array_layers: DDGI_TOTAL_LAYERS,
        },
      );
    }
  }

  let set_bgs = |pass: &mut bevy::render::render_resource::ComputePass, bg4: &BindGroup| {
    pass.set_bind_group(0, &bg0.0, &[]);
    pass.set_bind_group(1, &bg1.0, &[]);
    pass.set_bind_group(2, &bg2.0, &[]);
    pass.set_bind_group(3, &bg3.0, &[]);
    pass.set_bind_group(4, bg4, &[]);
  };

  // ①clear（1 wg：dispatch[0] 清零）
  crate::profiler::gpu_compute_pass(
    &mut profiler,
    ctx.command_encoder(),
    "gate_ddgi_clear",
    |pass| {
      pass.set_pipeline(p_clear);
      set_bgs(pass, &bg4.0);
      pass.dispatch_workgroups(1, 1, 1);
    },
  );
  // ②active（@workgroup_size(4,4,4)：dispatch 4×4×16 → gid 覆盖 16×16 cell × 64 层
  // = 4 LOD × 16 层/级；每 (lod,cell) 1 线程遍历 con tree 现算探针放置）
  crate::profiler::gpu_compute_pass(
    &mut profiler,
    ctx.command_encoder(),
    "gate_ddgi_active",
    |pass| {
      pass.set_pipeline(p_active);
      set_bgs(pass, &bg4.0);
      pass.dispatch_workgroups(4, 4, 16);
    },
  );
  // ②.5 seal（1 wg）：全量 count → 钳制值 dispatch[1]（≤ DDGI_PROBE_BUDGET，
  // 规避 max_compute_workgroups_per_dimension = 65535 静默跳过——gate 特有规模坑）
  crate::profiler::gpu_compute_pass(
    &mut profiler,
    ctx.command_encoder(),
    "gate_ddgi_seal",
    |pass| {
      pass.set_pipeline(p_seal);
      set_bgs(pass, &bg4.0);
      pass.dispatch_workgroups(1, 1, 1);
    },
  );
  // ② RayQuery 段（档位 ≥ 2）：桥接 copy → cast（indirect）→ update（indirect）。
  // 档位 1 只算 Active Probe（probe worklist/meta 就绪，供 Probe Viz 目验），
  // 不发射线、不写 irradiance/depth 纹理。
  if stage.run_cast() {
    // dispatch[1]（min）→ indirect[0] 桥接（4B；indirect[1]=y=1 [2]=z=1 常驻——
    // copy 若带 dispatch[2..] 会把零值带进 y/z → 零 workgroup）
    {
      let encoder = ctx.command_encoder();
      encoder.copy_buffer_to_buffer(&gpu.dispatch, 4, &gpu.indirect, 0, 4);
    }
    // ③cast（indirect：x = dispatch[1] 钳制后本帧处理探针数）
    crate::profiler::gpu_compute_pass(
      &mut profiler,
      ctx.command_encoder(),
      "gate_ddgi_cast",
      |pass| {
        pass.set_pipeline(p_cast);
        set_bgs(pass, &bg4.0);
        pass.dispatch_workgroups_indirect(&gpu.indirect, 0);
      },
    );
    // ④update（indirect：同 count；1 wg = 1 探针）
    crate::profiler::gpu_compute_pass(
      &mut profiler,
      ctx.command_encoder(),
      "gate_ddgi_update",
      |pass| {
        pass.set_pipeline(p_update);
        set_bgs(pass, &bg4.0);
        pass.dispatch_workgroups_indirect(&gpu.indirect, 0);
      },
    );
  }

  // ---- 诊断回读三阶段（wgpu 规则：submit 时 buffer 必须 Unmapped，故 copy 与
  // map_async 分帧；每阶段一帧，回调完成后读+unmap 回 idle）----
  match gpu.readback_state {
    1 => {
      // 上一帧 copy 已 submit → 请求映射（本帧 submit 时 buffer 无被录命令）
      let (tx, rx) = std::sync::mpsc::channel();
      gpu.readback.slice(..).map_async(MapMode::Read, move |_| {
        let _ = tx.send(());
      });
      *gpu.readback_rx.lock().unwrap() = Some(rx);
      gpu.readback_state = 2;
    }
    2 => {
      let done = gpu
        .readback_rx
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|rx| rx.try_recv().ok())
        .is_some();
      if done {
        let data = gpu.readback.slice(..).get_mapped_range();
        let word = |i: usize| {
          u32::from_le_bytes([
            data[i * 4],
            data[i * 4 + 1],
            data[i * 4 + 2],
            data[i * 4 + 3],
          ])
        };
        let count = word(0);
        let indirect0 = word(4);
        let wl = [word(8), word(9), word(10), word(11)];
        // 8 个采样层（层距 = copy 时定格的 readback_step）：各层非零 texel 数
        let texels_per_layer = (IRRADIANCE_LAYER_TEXELS * IRRADIANCE_LAYER_TEXELS * 2) as usize;
        let mut summary = String::new();
        for k in 0..8usize {
          let base = 64 + k * texels_per_layer;
          let nz = (base..base + texels_per_layer)
            .filter(|&i| word(i) != 0)
            .count();
          let layer = k as u32 * gpu.readback_step;
          summary.push_str(&format!(" L{layer}:{nz}"));
        }
        // depth 采样层 0（tmax=8192 初值；EMA 拉低 = update 写入实证）
        let dep_base = 64 + 8 * texels_per_layer;
        let dep_texels = (DEPTH_LAYER_TEXELS * DEPTH_LAYER_TEXELS) as usize;
        let dep_lt = (dep_base..dep_base + dep_texels)
          .filter(|&i| f32::from_bits(word(i)) < 8000.0)
          .count();
        // 黑探针普查（同 8 采样层，2048 探针）：a0 = age=0（未激活，采样端已门控）；
        // blk = age≥1 且 irr 全零（被当有效探针采样 → 暗块/黑块直接来源）；ok = 正常。
        let meta_base =
          64 + 8 * texels_per_layer + (DEPTH_LAYER_TEXELS * DEPTH_LAYER_TEXELS) as usize;
        let (mut a0, mut blk, mut ok) = (0u32, 0u32, 0u32);
        let mut blk_ids = String::new();
        for k in 0..8usize {
          let layer = k as u32 * gpu.readback_step;
          let ibase = 64 + k * texels_per_layer;
          // meta 区按拷贝行距 256B（64 word/行）读回：层 = 16 行 × 64 word，
          // 探针 p 的 word 下标 = 行 p/16 × 64 + 列 p%16（其余为 pad 零）
          let mb = meta_base + k * (PROBES_PER_LAYER_AXIS as usize * META_ROW_WORDS);
          for p in 0..PROBES_PER_LAYER as usize {
            let age = (word(
              mb + (p / PROBES_PER_LAYER_AXIS as usize) * META_ROW_WORDS
                + (p % PROBES_PER_LAYER_AXIS as usize),
            ) >> META_AGE_SHIFT)
              & 0xFF;
            if age == 0 {
              a0 += 1;
              continue;
            }
            let px = p % PROBES_PER_LAYER_AXIS as usize;
            let py = p / PROBES_PER_LAYER_AXIS as usize;
            let mut nz = false;
            'scan: for ty in 0..IRRADIANCE_TEXELS as usize {
              let row = ibase + (((py * 8 + ty) * IRRADIANCE_LAYER_TEXELS as usize) + px * 8) * 2;
              for tx in 0..IRRADIANCE_TEXELS as usize {
                if word(row + tx * 2) != 0 || word(row + tx * 2 + 1) != 0 {
                  nz = true;
                  break 'scan;
                }
              }
            }
            if nz {
              ok += 1;
            } else {
              blk += 1;
              if blk <= 8 {
                blk_ids.push_str(&format!(
                  " {}@{}",
                  layer as u32 * PROBES_PER_LAYER + p as u32,
                  age
                ));
              }
            }
          }
        }
        drop(data);
        gpu.readback.unmap();
        gpu.readback_state = 0;
        let line = format!(
          "DDGI readback: frame={} dispatch0={} indirect0={} wl=[{},{},{},{}] irr:{summary} depth0_lt8k:{} census: a0={a0} blk={blk} ok={ok} blk_ids:[{blk_ids}]",
          gpu.frame, count, indirect0, wl[0], wl[1], wl[2], wl[3], dep_lt,
        );
        bevy::log::info!("{}", line);
      }
    }
    _ => {
      if gpu.frame % 120 == 0 {
        {
          let encoder = ctx.command_encoder();
          encoder.copy_buffer_to_buffer(&gpu.dispatch, 0, &gpu.readback, 0, 16);
          // 诊断：indirect[0]（桥接是否生效）+ worklist 头部（active 写入是否可见）
          encoder.copy_buffer_to_buffer(&gpu.indirect, 0, &gpu.readback, 16, 16);
          encoder.copy_buffer_to_buffer(&gpu.worklist, 0, &gpu.readback, 32, 64);
          // 8 个均匀采样层（层距 = 64/8 = 8）+ depth 层 0 + meta 同 8 层——
          // 读 **next**（本帧 update 刚写完，encoder 顺序在 update pass 之后）。
          let step = DDGI_TOTAL_LAYERS / 8;
          gpu.readback_step = step;
          for k in 0..8u32 {
            encoder.copy_texture_to_buffer(
              TexelCopyTextureInfo {
                texture: &gpu.irr_next,
                mip_level: 0,
                origin: Origin3d {
                  x: 0,
                  y: 0,
                  z: k * step,
                },
                aspect: TextureAspect::All,
              },
              TexelCopyBufferInfo {
                buffer: &gpu.readback,
                layout: TexelCopyBufferLayout {
                  offset: 256
                    + (k as u64) * (IRRADIANCE_LAYER_TEXELS * IRRADIANCE_LAYER_TEXELS * 8) as u64,
                  bytes_per_row: Some(IRRADIANCE_LAYER_TEXELS * 8),
                  rows_per_image: Some(IRRADIANCE_LAYER_TEXELS),
                },
              },
              Extent3d {
                width: IRRADIANCE_LAYER_TEXELS,
                height: IRRADIANCE_LAYER_TEXELS,
                depth_or_array_layers: 1,
              },
            );
          }
          // depth 采样层 0（tmax=8192 初值；EMA 拉低 = update 写入实证）
          encoder.copy_texture_to_buffer(
            TexelCopyTextureInfo {
              texture: &gpu.depth_next,
              mip_level: 0,
              origin: Origin3d::ZERO,
              aspect: TextureAspect::All,
            },
            TexelCopyBufferInfo {
              buffer: &gpu.readback,
              layout: TexelCopyBufferLayout {
                offset: 256 + (8 * IRRADIANCE_LAYER_TEXELS * IRRADIANCE_LAYER_TEXELS * 8) as u64,
                bytes_per_row: Some(DEPTH_LAYER_TEXELS * 4),
                rows_per_image: Some(DEPTH_LAYER_TEXELS),
              },
            },
            Extent3d {
              width: DEPTH_LAYER_TEXELS,
              height: DEPTH_LAYER_TEXELS,
              depth_or_array_layers: 1,
            },
          );
          // 黑探针普查：同 8 个采样层的 meta（逐探针 age；布局 = 16×16 u32/层）
          for k in 0..8u32 {
            encoder.copy_texture_to_buffer(
              TexelCopyTextureInfo {
                texture: &gpu.meta_next,
                mip_level: 0,
                origin: Origin3d {
                  x: 0,
                  y: 0,
                  z: k * step,
                },
                aspect: TextureAspect::All,
              },
              TexelCopyBufferInfo {
                buffer: &gpu.readback,
                layout: TexelCopyBufferLayout {
                  offset: 256
                    + (8 * IRRADIANCE_LAYER_TEXELS * IRRADIANCE_LAYER_TEXELS * 8) as u64
                    + (DEPTH_LAYER_TEXELS * DEPTH_LAYER_TEXELS * 4) as u64
                    + (k as u64) * (META_COPY_BPR as u64 * PROBES_PER_LAYER_AXIS as u64),
                  // 行距 256B 对齐（行宽 64B 违规 → Validation Error）
                  bytes_per_row: Some(META_COPY_BPR),
                  rows_per_image: Some(PROBES_PER_LAYER_AXIS),
                },
              },
              Extent3d {
                width: PROBES_PER_LAYER_AXIS,
                height: PROBES_PER_LAYER_AXIS,
                depth_or_array_layers: 1,
              },
            );
          }
        }
        gpu.readback_state = 1;
      }
    }
  }
}

/// main world DDGI 开关/调试参数 → render world（每帧 extract）。
/// 新架构 CPU 零烘焙：探针放置全部在 GPU active pass 逐帧遍历 con tree，
/// main world 无任何几何资源需要提取。
fn extract_ddgi_settings(
  mut commands: Commands,
  stage: Option<bevy::render::Extract<Res<DdgiStage>>>,
  debug: Option<bevy::render::Extract<Res<DdgiDebugSettings>>>,
) {
  // 主世界档位 → 渲染世界（prepare/dispatch 读渲染世界副本）；
  // 资源缺失/越界回退 = 0 关（与 DdgiStage::default()=OFF 一致）
  let s = stage.map_or(DdgiStage::OFF, |s| s.0.min(DdgiStage::FULL));
  commands.insert_resource(DdgiStage(s));
  // 主世界 DDGI 调试参数 → 渲染世界（prepare_ddgi 写进 uniform.params.y/.z）
  let dbg = debug.map_or_else(DdgiDebugSettings::default, |d| DdgiDebugSettings {
    mode: d.mode,
    gain: d.gain,
    probe_viz: d.probe_viz,
    probe_viz_lod: d.probe_viz_lod,
  });
  commands.insert_resource(dbg);
}

// --- 零依赖字节打包（f32 → le bytes；workspace 无 bytemuck 依赖）---
fn f32_bytes(v: &[f32]) -> Vec<u8> {
  let mut out = Vec::with_capacity(v.len() * 4);
  for c in v {
    out.extend_from_slice(&c.to_le_bytes());
  }
  out
}

/// 每帧 prepare：ping-pong 交换 → 相机 fine 坐标算 4 级滚动窗口 uniform →
/// 写 uniform buffer → 建唯一 BG4。纯算术 + 资源编排，零几何遍历。
#[allow(clippy::too_many_arguments)]
fn prepare_ddgi(
  mut commands: Commands,
  device: Res<RenderDevice>,
  queue: Res<RenderQueue>,
  pipeline_cache: Res<bevy::render::render_resource::PipelineCache>,
  view: Option<Res<crate::brickmap::dda::DdaViewUniform>>,
  stage: Res<DdgiStage>,
  dbg: Res<DdgiDebugSettings>,
  mut gpu: ResMut<DdgiGpu>,
) {
  // ---- D7 ping-pong 交换（无条件；两份纹理逐帧互换 prev/next 身份）----
  // 上一帧 next（已被 update 写入新值）变本帧 prev；旧 prev 由 dispatch_ddgi 的
  // copy pass 全 64 层重填后作为本帧 next 被 active/update 覆写。
  {
    let g = &mut *gpu;
    let DdgiGpu {
      irr_prev,
      irr_next,
      irr_prev_view,
      irr_next_view,
      depth_prev,
      depth_next,
      depth_prev_view,
      depth_next_view,
      meta_prev,
      meta_next,
      meta_prev_view,
      meta_next_view,
      ..
    } = g;
    std::mem::swap(irr_prev, irr_next);
    std::mem::swap(irr_prev_view, irr_next_view);
    std::mem::swap(depth_prev, depth_next);
    std::mem::swap(depth_prev_view, depth_next_view);
    std::mem::swap(meta_prev, meta_next);
    std::mem::swap(meta_prev_view, meta_next_view);
  }

  gpu.frame = gpu.frame.wrapping_add(1);

  // 相机 fine 坐标（render world DdaViewUniform；extract_camera_config 每帧插入）
  let cam_fine = view
    .map(|v| v.cam_pos_fine.truncate())
    .unwrap_or(Vec3::splat(32.0));

  // ---- 4 级 LOD 窗口：相机锚定、半窗 8 cell 滚动（纯算术，零几何查询）----
  let mut u = DdgiUniform::default();
  for lod in 0..DDGI_LODS as usize {
    let cell_size = DDGI_LOD_CELL_SIZES[lod];
    let w = compute_lod_window(cam_fine, cell_size, gpu.prev_origins[lod], gpu.have_prev);
    u.lods[lod] = DdgiLod {
      origin: IVec4::new(w.origin.x, w.origin.y, w.origin.z, w.cell_size),
      prev_origin: IVec4::new(w.prev_origin.x, w.prev_origin.y, w.prev_origin.z, 0),
      reuse_min: IVec4::new(w.reuse_min.x, w.reuse_min.y, w.reuse_min.z, 0),
      reuse_max: IVec4::new(w.reuse_max.x, w.reuse_max.y, w.reuse_max.z, 0),
    };
    gpu.prev_origins[lod] = w.origin;
  }
  gpu.have_prev = true;
  // params：x=frame, y=调试 mode, z=gain, w=object bbox 数（暂为 0）；
  // misc.x = ③ voxel 着色采样开关（仅档位 3 = FULL 置 1；trace shader 据此跳过
  // 探针采样 gi=0）。档位 1/2 compute 照跑但着色端不消费探针数据
  u.params = Vec4::new(gpu.frame as f32, dbg.mode, dbg.gain, 0.0);
  u.misc = Vec4::new(if stage.shade_gi() { 1.0 } else { 0.0 }, 0.0, 0.0, 0.0);
  *gpu.uniform.get_mut() = u;
  gpu.uniform.write_buffer(&device, &queue);

  // ---- 唯一 BG4（12 binding 逐字镜像 WGSL group(4)）----
  let bg4_layout = pipeline_cache.get_bind_group_layout(&ddgi_bg4_layout());
  let bg4 = device.create_bind_group(
    None,
    &bg4_layout,
    &BindGroupEntries::sequential((
      &gpu.uniform,
      &gpu.irr_prev_view,
      &gpu.depth_prev_view,
      &gpu.irr_next_view,
      &gpu.depth_next_view,
      &gpu.meta_prev_view,
      &gpu.meta_next_view,
      gpu.dispatch.as_entire_binding(),
      gpu.objects.as_entire_binding(),
      gpu.samples.as_entire_binding(),
      gpu.worklist.as_entire_binding(),
      gpu.slot_pos.as_entire_binding(),
    )),
  );
  commands.insert_resource(DdgiBg4(bg4));
}

// ============================================================================
// 单测：相机滚动窗口纯算术（Douglas #23 sort.glsl base_position / reuse_bounds）
// ============================================================================

#[cfg(test)]
mod tests {
  use super::*;

  /// T3：相机滚动窗口纯算术——相机恒在窗内 [4,12) 死区、origin 按 8 cell 步进、
  /// 负坐标 euclid 正确、静止全窗复用 / 滚 8 cell 半窗复用 / 首帧无复用。
  #[test]
  fn lod_window_camera_anchored_band_and_reuse() {
    for &cs in &DDGI_LOD_CELL_SIZES {
      // ① 相机在窗内相对坐标恒 ∈ [4,12)，遍历大范围（含负、跨步进边界）
      for cam_cell in -40..40 {
        let cam_fine = Vec3::splat(cam_cell as f32 * cs as f32 + cs as f32 * 0.37);
        let origin = lod_window_origin(cam_fine, cs);
        let rel = cam_cell - origin.x;
        assert!(
          (DDGI_BAND_LO..DDGI_BAND_LO + DDGI_SCROLL_STEP).contains(&rel),
          "cs={cs} cam_cell={cam_cell} origin={} rel={rel} 不在 [4,12)",
          origin.x
        );
        // origin+4 必为步长 8 的倍数（窗口按半窗对齐）
        assert_eq!((origin.x + DDGI_BAND_LO).rem_euclid(DDGI_SCROLL_STEP), 0);
        // fine 对齐 cell_size
        assert_eq!((origin.x * cs).rem_euclid(cs), 0);
      }
      // ② 静止 → 复用界 = 全窗 [origin, origin+16)
      let cam = Vec3::splat(100.0 * cs as f32);
      let o0 = lod_window_origin(cam, cs);
      let w = compute_lod_window(cam, cs, o0, true);
      assert_eq!(w.reuse_min, o0);
      assert_eq!(w.reuse_max, o0 + DDGI_WINDOW_AXIS);
      // ③ 向前滚 8 cell（相机 cell 100 → 108）→ 窗口角 92 → 100，重叠带厚 8
      let cam8 = cam + Vec3::splat(8.0 * cs as f32);
      let o8 = lod_window_origin(cam8, cs);
      assert_eq!(o8.x, o0.x + 8);
      let w8 = compute_lod_window(cam8, cs, o0, true);
      assert_eq!(w8.reuse_min.x, o0.x + 8);
      assert_eq!(w8.reuse_max.x, o0.x + 16);
      // ④ 首帧无 prev → 复用界空（min==max）
      let wf = compute_lod_window(cam, cs, IVec3::ZERO, false);
      assert_eq!(wf.reuse_min, wf.reuse_max);
    }
  }
}
