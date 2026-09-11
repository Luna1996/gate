//! DDGI（Dynamic Diffuse Global Illumination）——阶段一：世界空间探针烘焙 + 活跃探针筛选。
//!
//! 架构（严格对齐 Douglas Devlog #23 / Majercik 2019, 2021）：
//! - **嵌套级联 LOD，以相机为中心**：4 级 LOD 各自以相机为中心铺 16³ 网格（cell 边长
//!   16/32/64/128 voxel），覆盖范围逐级 ×2 且严格嵌套（LOD(l-1) 盒 ⊂ LOD(l) 盒）。
//!   相机移动使某级 origin 按该级 cell 对齐滚动时，只重烘「世界 cell 发生变化」的槽位
//!   （`ddgi_cell_id` 增量缓存），未变的槽位续龄。
//! - **烘焙（`ddgi_bake`）**：世界数据变化（上传修订号自增）时才跑一次。逐 cell 沿 4³ 分裂树
//!   **BFS 找「最大的全空叶」并把探针放在其中心**（同级优先靠 cell 中心；全满 cell 无探针；
//!   全空 cell 居中）——即 Douglas 的探针放置启发式。结果写入 `ddgi_cell` storage buffer。
//! - **活跃判定（`ddgi_sort`）**：每帧逐 cell，读烘焙记录（不再重算树 BFS）；探针存在且
//!   「本 cell 或 6 邻接 cell 有体素」（或与非网格对齐物体 AABB 重叠）→ 活跃 → atomicAdd 进
//!   per-LOD worklist；同时刷新 age / slot_pos / meta。
//! - **seal**：按 LOD 把固定射线预算摊给活跃探针，写 cast/collect indirect args。
//!
//! 阶段二（cast）/ 阶段三（collect/着色）尚未接入；`irr/depth` 纹理沿用旧布局暂作占位。

use bevy::render::render_resource::{CachedComputePipelineId, ShaderType};
use glam::{IVec3, IVec4, UVec3, UVec4, Vec4};

pub const IRRADIANCE_TEXELS: u32 = 8;
pub const DEPTH_TEXELS: u32 = 16;
pub const PROBE_T_MAX: f32 = 8192.0;
pub const DDGI_RAY_BUDGET: u32 = 65536;

pub const DDGI_LODS: u32 = 4;
/// 最细 LOD 的 cell 边长（voxel 单位）——Douglas 烘焙网格 base cell。
pub const DDGI_BASE_CELL: i32 = 16;
/// 4 级 LOD cell 边长（每级 ×2）。
pub const DDGI_LOD_CELL_SIZES: [i32; DDGI_LODS as usize] = [16, 32, 64, 128];
/// 各级 LOD 的 cell 维度（4 级相同）。每级 cell ×2 且维度不变 → 覆盖范围逐级 ×2，
/// 形成严格嵌套的级联。水平 32 格、垂直 16 格（体素世界水平视野远大于垂直），
/// 各级覆盖范围 = dims×cell：512×256×512 / 1024×512×1024 / 2048×1024×2048 / 4096×2048×4096。
/// 每级 16384 槽 → 共 65536 槽（= 占位纹理容量，全部槽位可采样）。
pub const DDGI_LOD_DIMS: UVec3 = UVec3::new(32, 16, 32);
/// 级联原点的对齐步长（单位 = 本级 cell 数）。
///
/// 探针位置只依赖世界内容、与相机无关；但槽位 ↔ 世界 cell 的映射依赖原点。原点若按
/// 「1 个 cell」对齐，2cm 体素下 LOD0（cell=16 voxel）每移动 32cm 就整体滚动一次 → 飞行时
/// 等于每帧全量重烘 16384 槽。按 8 个 cell 对齐把滚动频率降 8 倍；同时保证相机偏离盒心
/// ≤ 8 个 cell（半宽 16 个 cell）→ 前方仍有 ≥ 8 个 cell（LOD0 为 128 voxel = 2.56m）余量。
pub const DDGI_CASCADE_STEP_CELLS: i32 = 8;

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
  /// 每级：dims 固定为 `DDGI_LOD_DIMS`，cell 边长 = `16 << lod`；原点按 `cell ×
  /// DDGI_CASCADE_STEP_CELLS` 对齐后把相机放在盒中心附近（对齐误差 ≤ 步长）。因覆盖范围逐级
  /// ×2 且对齐误差远小于半宽，LOD(l-1) 盒严格包含于 LOD(l) 盒内。
  pub fn from_camera(camera_voxel: IVec3) -> Self {
    let mut out = Self::default();
    let dims = DDGI_LOD_DIMS;
    let half_cells = (DDGI_LOD_DIMS / 2).as_ivec3();
    let mut base = 0u32;
    for lod in 0..DDGI_LODS as usize {
      let cell = DDGI_LOD_CELL_SIZES[lod];
      let half = half_cells * cell;
      let step = cell * DDGI_CASCADE_STEP_CELLS;
      let origin = IVec3::new(
        align_down(camera_voxel.x, step) - half.x,
        align_down(camera_voxel.y, step) - half.y,
        align_down(camera_voxel.z, step) - half.z,
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
  /// x = frame, y = debug mode, z = gain, w = 物体数（非网格对齐 AABB 检查）
  pub params: Vec4,
  /// x = shade GI, y = total slots
  pub misc: Vec4,
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
      // 1/2：上一帧 irradiance / depth（阶段三着色采样用；阶段一未接入 cast/collect，
      // 纹理仅占位，索引空间与当前世界网格不一致，`ddgi_sample` 已做越界保护）
      tex(1, TextureSampleType::Float { filterable: true }),
      tex(2, TextureSampleType::Float { filterable: false }),
      // 3/4：本帧 irradiance / depth 写入目标（阶段二/三接入）
      store(3, TextureFormat::Rgba16Float),
      store(4, TextureFormat::R32Float),
      // 5：烘焙输出（bake 写 / sort 读）6：age/flags（读写）7：indirect/counter（读写）
      // 8：objects（只读）9：worklist（读写）10：slot_pos（读写）11：cell_id（读写，滚动增量）
      buf(5, false),
      buf(6, false),
      buf(7, false),
      buf(8, true),
      buf(9, false),
      buf(10, false),
      buf(11, false),
    ],
  )
}

#[derive(Debug, Clone, Copy)]
pub struct DdgiPipelines {
  pub bake: CachedComputePipelineId,
  pub sort: CachedComputePipelineId,
  pub seal: CachedComputePipelineId,
}

#[derive(bevy::ecs::resource::Resource)]
pub struct DdgiGpu {
  pub uniform: bevy::render::render_resource::UniformBuffer<DdgiUniform>,
  pub objects: bevy::render::render_resource::Buffer,
  pub indirect: bevy::render::render_resource::Buffer,
  pub worklist: bevy::render::render_resource::Buffer,
  pub slot_pos: bevy::render::render_resource::Buffer,
  /// 烘焙输出：每 slot 一条 (flags | off_b)
  pub cell: bevy::render::render_resource::Buffer,
  /// 每 slot 已烘焙的世界 cell 键 + 有效标志（滚动增量烘焙）
  pub cell_id: bevy::render::render_resource::Buffer,
  /// 每帧 age / enabled
  pub meta: bevy::render::render_resource::Buffer,
  pub irr_prev: bevy::render::render_resource::Texture,
  pub irr_next: bevy::render::render_resource::Texture,
  pub depth_prev: bevy::render::render_resource::Texture,
  pub depth_next: bevy::render::render_resource::Texture,
  pub irr_prev_view: bevy::render::render_resource::TextureView,
  pub irr_next_view: bevy::render::render_resource::TextureView,
  pub depth_prev_view: bevy::render::render_resource::TextureView,
  pub depth_next_view: bevy::render::render_resource::TextureView,
  pub frame: u32,
  pub grid: DdgiWorldGrid,
  pub total_slots: u32,
  /// 已消费的世界修订号（变化 → 需要重烘焙）
  pub last_revision: u64,
  pub pipelines: Option<DdgiPipelines>,
}

#[derive(bevy::ecs::resource::Resource)]
pub struct DdgiBg4(pub bevy::render::render_resource::BindGroup);

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
  pub fn run_active(&self) -> bool {
    self.0 >= Self::ACTIVE
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

pub struct DdgiPlugin;

impl bevy::app::Plugin for DdgiPlugin {
  fn build(&self, app: &mut bevy::app::App) {
    use bevy::ecs::schedule::IntoScheduleConfigs;
    use bevy::prelude::RenderGraph;
    app.init_resource::<DdgiStage>();
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

/// indirect args / 计数器合一 buffer：既要作为 storage（atomic）被 sort/seal 读写，
/// 又要在阶段二作为 indirect dispatch 参数源。
fn ddgi_indirect_buffer(
  device: &bevy::render::renderer::RenderDevice,
  queue: &bevy::render::renderer::RenderQueue,
) -> bevy::render::render_resource::Buffer {
  use bevy::render::render_resource::{BufferDescriptor, BufferUsages};
  let buf = device.create_buffer(&BufferDescriptor {
    label: Some("ddgi_indirect".into()),
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

/// 阶段二/三占位纹理（cast/collect 接入后重做布局）：容量 = 各级 LOD 槽数总和 65536，
/// 每层 16×16 = 256 探针 → 需要 256 层。
const DDGI_PLACEHOLDER_LAYERS: u32 = 256;
const DDGI_PLACEHOLDER_PROBES_PER_LAYER_AXIS: u32 = 16;

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
      depth_or_array_layers: DDGI_PLACEHOLDER_LAYERS,
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

fn init_ddgi_gpu(
  mut commands: bevy::ecs::system::Commands,
  device: bevy::ecs::system::Res<bevy::render::renderer::RenderDevice>,
  queue: bevy::ecs::system::Res<bevy::render::renderer::RenderQueue>,
) {
  use bevy::render::render_resource::TextureFormat;
  let mk_pair = |label: &str, format: TextureFormat, size: (u32, u32)| {
    let t0 = ddgi_array_tex(&device, &format!("{label}_a"), format, size);
    let t1 = ddgi_array_tex(&device, &format!("{label}_b"), format, size);
    let v0 = ddgi_array_view(&t0);
    let v1 = ddgi_array_view(&t1);
    (t0, v0, t1, v1)
  };
  let irr_axis = DDGI_PLACEHOLDER_PROBES_PER_LAYER_AXIS * IRRADIANCE_TEXELS;
  let dep_axis = DDGI_PLACEHOLDER_PROBES_PER_LAYER_AXIS * DEPTH_TEXELS;
  let (irr_a, irr_av, irr_b, irr_bv) =
    mk_pair("ddgi_irr", TextureFormat::Rgba16Float, (irr_axis, irr_axis));
  let (dep_a, dep_av, dep_b, dep_bv) =
    mk_pair("ddgi_depth", TextureFormat::R32Float, (dep_axis, dep_axis));

  let slot_pos = zero_storage_buffer(&device, &queue, "ddgi_slot_pos", 4096 * 16);
  let worklist = zero_storage_buffer(&device, &queue, "ddgi_worklist", 4096 * 16);
  let cell = zero_storage_buffer(&device, &queue, "ddgi_cell", 4096 * 4);
  let cell_id = zero_storage_buffer(&device, &queue, "ddgi_cell_id", 4096 * 16);
  let meta = zero_storage_buffer(&device, &queue, "ddgi_meta", 4096 * 4);
  let indirect = ddgi_indirect_buffer(&device, &queue);
  let objects = dummy_sized_buffer(&device, "ddgi_objects", 16);

  commands.insert_resource(DdgiGpu {
    uniform: bevy::render::render_resource::UniformBuffer::default(),
    objects,
    indirect,
    worklist,
    slot_pos,
    cell,
    cell_id,
    meta,
    irr_prev: irr_a,
    irr_next: irr_b,
    depth_prev: dep_a,
    depth_next: dep_b,
    irr_prev_view: irr_av,
    irr_next_view: irr_bv,
    depth_prev_view: dep_av,
    depth_next_view: dep_bv,
    frame: 0,
    grid: DdgiWorldGrid::default(),
    total_slots: 0,
    last_revision: u64::MAX,
    pipelines: None,
  });
}

fn queue_ddgi_pipelines(
  dda: Option<bevy::ecs::system::Res<crate::brickmap::dda::DdaPipelines>>,
  pipeline_cache: bevy::ecs::system::Res<bevy::render::render_resource::PipelineCache>,
  asset_server: bevy::ecs::system::Res<bevy::asset::AssetServer>,
  mut gpu: bevy::ecs::system::ResMut<DdgiGpu>,
) {
  use bevy::render::render_resource::{ComputePipelineDescriptor, PipelineCache};
  use std::borrow::Cow;
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
  let mk = |pipeline_cache: &PipelineCache, label: &'static str, entry: &'static str| {
    pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
      label: Some(Cow::from(label)),
      layout: layout.clone(),
      shader: shader.clone(),
      entry_point: Some(Cow::from(entry)),
      ..Default::default()
    })
  };
  gpu.pipelines = Some(DdgiPipelines {
    bake: mk(&pipeline_cache, "gate_ddgi_bake", "ddgi_bake"),
    sort: mk(&pipeline_cache, "gate_ddgi_sort", "ddgi_sort"),
    seal: mk(&pipeline_cache, "gate_ddgi_seal", "ddgi_seal"),
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
  bake: Option<bevy::ecs::system::Res<DdgiBakeThisFrame>>,
  stage: bevy::ecs::system::Res<DdgiStage>,
  dbg: Option<bevy::ecs::system::Res<DdgiDebugSettings>>,
  gpu: bevy::ecs::system::Res<DdgiGpu>,
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
  let Some(p_bake) = pipeline_cache.get_compute_pipeline(pipes.bake) else {
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

  // 烘焙：世界数据变化时重算探针位置（BFS 最大空叶）
  if bake.map_or(false, |b| b.0) {
    crate::profiler::gpu_compute_pass(
      &mut profiler,
      ctx.command_encoder(),
      "gate_ddgi_bake",
      |pass| {
        pass.set_pipeline(p_bake);
        set_bgs(pass, &bg4.0);
        pass.dispatch_workgroups(wg_slots, 1, 1);
      },
    );
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
        pass.dispatch_workgroups(1, 1, 1);
      },
    );
  }
  // 阶段二 cast / 阶段三 collect 待接入：seal 已在 indirect buffer 备好 per-LOD
  // dispatch 参数（offset 0 = cast×4 LOD，offset 64 = collect×4 LOD）与 rpp（offset 128）。
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
  // 相机滚动（grid 变化）→ 增量补烘；世界编辑（修订号变化）→ 失效缓存全量重烘
  let need_bake = (grid_changed || rev_changed) && total > 0 && will_run;

  // ---- 缓冲容量（槽数固定：4 LOD × 16³）----
  let s = total as u64;
  ensure_storage_buffer(&device, &queue, &mut gpu.cell, "ddgi_cell", s * 4);
  ensure_storage_buffer(&device, &queue, &mut gpu.meta, "ddgi_meta", s * 4);
  ensure_storage_buffer(&device, &queue, &mut gpu.slot_pos, "ddgi_slot_pos", s * 16);
  ensure_storage_buffer(&device, &queue, &mut gpu.worklist, "ddgi_worklist", s * 16);
  ensure_storage_buffer(&device, &queue, &mut gpu.cell_id, "ddgi_cell_id", s * 16);
  // 世界编辑 → 清空 cell_id 缓存令 bake 全量重算；纯滚动不清，bake 只补「世界 cell 变化」的槽位
  if need_bake && rev_changed {
    queue.write_buffer(&gpu.cell_id, 0, &vec![0u8; (s * 16).max(4) as usize]);
  }

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
  u.params = Vec4::new(gpu.frame as f32, dbg.mode, dbg.gain, 0.0);
  u.misc = Vec4::new(
    if stage.shade_gi() { 1.0 } else { 0.0 },
    gpu.total_slots as f32,
    0.0,
    0.0,
  );
  *gpu.uniform.get_mut() = u;
  gpu.uniform.write_buffer(&device, &queue);

  // 每帧清零活跃计数器（indirect words [36..40)）；indirect args 由 seal 当帧覆写。
  queue.write_buffer(
    &gpu.indirect,
    DDGI_COUNTER_CLEAR_OFFSET,
    &[0u8; DDGI_COUNTER_CLEAR_BYTES as usize],
  );

  // ---- BG4 ----
  use bevy::render::render_resource::BindGroupEntries;
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
      gpu.cell.as_entire_binding(),
      gpu.meta.as_entire_binding(),
      gpu.indirect.as_entire_binding(),
      gpu.objects.as_entire_binding(),
      gpu.worklist.as_entire_binding(),
      gpu.slot_pos.as_entire_binding(),
      gpu.cell_id.as_entire_binding(),
    )),
  );
  commands.insert_resource(DdgiBg4(bg4));
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

  #[test]
  fn cascade_origin_scrolls_by_step_not_by_cell() {
    // 一个步长内移动：各级原点都不动（不触发重烘）
    let a = DdgiWorldGrid::from_camera(IVec3::new(1, 1, 1));
    let b = DdgiWorldGrid::from_camera(IVec3::new(8, 8, 8));
    assert_eq!(a, b);
    // 越过 LOD0 的步长（cell 16 × 8 = 128）：LOD0 滚动；LOD1 步长 256 → 不动
    let c = DdgiWorldGrid::from_camera(IVec3::new(130, 1, 1));
    assert_ne!(a, c);
    assert_ne!(a.lod_origins[0], c.lod_origins[0]);
    assert_eq!(a.lod_origins[1], c.lod_origins[1]);
  }
}
