//! 逐体素直光可见性缓存（Douglas #19：每体素 1 条阴影射线 + hashmap 跨帧复用）。
//!
//! 方向光 vis 视角无关（只依赖几何）→ 缓存跨帧持久，收益来自跨帧复用；
//! 体素编辑后几何变化 → 按 `VolumeGrid.edit_generation` 整表清零（懒失效）。
//! WGSL 侧：`dda.wgsl` vis_key/vis_lookup/vis_insert/vis_sun_blocked（BG5）。
//!
//! 表：2^22 slots × u32 = 16MB，slot = tag30<<2 | state（0=空 1=遮挡 2=可见）。
//! tag = 32bit key hash 折叠 30bit——冲突误判率 ~1/2^30/查询，表现为极偶发的
//! 单像素光影错（无累积），编辑清零兜底。`GATE_NO_VIS_CACHE=1` 旁路（enabled=0，
//! shader 回退旧逐像素阴影射线）。

use bevy::ecs::schedule::IntoScheduleConfigs;
use bevy::ecs::system::{Commands, Res, ResMut};
use bevy::render::render_resource::{ShaderStages, ShaderType};
use bevy::render::{
  Extract, ExtractSchedule, Render, RenderStartup, RenderSystems,
  render_resource::{
    BindGroupLayoutDescriptor, BindGroupLayoutEntries, Buffer, BufferDescriptor, BufferUsages,
    UniformBuffer,
    binding_types::{storage_buffer_sized, uniform_buffer},
  },
  renderer::{RenderDevice, RenderQueue},
};

use crate::VoxelScene;

/// 表容量（2^22 slots × 4B = 16MB；demo 可见体素 ~几万，负载 <1%）
pub const VIS_CAP_SLOTS: u32 = 1 << 22;
pub const VIS_CAP_BYTES: u32 = VIS_CAP_SLOTS * 4;

/// BG5 meta uniform（WGSL VisCacheMeta 逐字镜像，16B）
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, ShaderType)]
pub struct VisCacheMeta {
  pub enabled: u32,
  pub capacity_mask: u32,
  pub _pad0: u32,
  pub _pad1: u32,
}

/// 渲染 world 持久缓存资源（RenderStartup 一次创建；表内容 GPU 侧积累）
#[derive(bevy::ecs::resource::Resource)]
pub struct VisCacheGpu {
  pub table: Buffer,
  pub norm: Buffer,
  pub meta: UniformBuffer<VisCacheMeta>,
  pub last_generation: u64,
  /// enabled 去重——变化才写 meta
  pub last_written: u32,
}

/// ExtractSchedule 产物：主世界体素编辑代数（prepare 比对触发清零）
#[derive(bevy::ecs::resource::Resource, Default)]
pub struct VisGenExtract(pub u64);

/// main world 开关（main.rs 按键切换）→ extract 进 render world → 每帧写 meta
/// （同轮 A/B 实验用：运行时切换免受热节流跨轮污染）
#[derive(bevy::ecs::resource::Resource, Default)]
pub struct VisCacheEnabled(pub bool);

/// BG5 layout：rw 可见性表 + rw normal 表 + meta uniform（WGSL 逐字对应）
pub fn vis_cache_bg5_layout() -> BindGroupLayoutDescriptor {
  BindGroupLayoutDescriptor::new(
    "DdaBg5",
    &BindGroupLayoutEntries::sequential(
      ShaderStages::COMPUTE,
      (
        storage_buffer_sized(false, None),     // @binding(0) vis_table（atomic u32 array）
        uniform_buffer::<VisCacheMeta>(false), // @binding(1) vis_meta
        storage_buffer_sized(false, None),     // @binding(2) vis_norm（64bit 槽 × 2 u32）
      ),
    ),
  )
}

fn init_vis_cache(
  mut commands: Commands,
  device: Res<RenderDevice>,
  queue: Res<RenderQueue>,
) {
  let bypass = std::env::var("GATE_NO_VIS_CACHE").is_ok();
  let mut meta = UniformBuffer::from(VisCacheMeta {
    enabled: u32::from(!bypass),
    capacity_mask: VIS_CAP_SLOTS - 1,
    _pad0: 0,
    _pad1: 0,
  });
  meta.write_buffer(&device, &queue);
  let table = device.create_buffer(&BufferDescriptor {
    label: Some("gate_vis_cache"),
    size: VIS_CAP_BYTES as u64,
    // COPY_DST：prepare 清零 write_buffer 必需
    usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    mapped_at_creation: false,
  });
  // per-voxel normal 表：64bit 槽 = (tag32, oct16×2) → 2× u32 slots
  let norm = device.create_buffer(&BufferDescriptor {
    label: Some("gate_vis_norm"),
    size: (VIS_CAP_SLOTS as u64) * 8,
    // COPY_DST：prepare 清零 write_buffer 必需
    usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    mapped_at_creation: false,
  });
  commands.insert_resource(VisCacheGpu {
    table,
    norm,
    meta,
    last_generation: 0,
    last_written: u32::from(!bypass),
  });
}

fn extract_vis_generation(
  mut commands: Commands,
  scene: Option<Extract<Res<VoxelScene>>>,
  enabled: Option<Extract<Res<VisCacheEnabled>>>,
) {
  let generation = scene.map(|s| s.volumes.main().edit_generation()).unwrap_or(0);
  commands.insert_resource(VisGenExtract(generation));
  commands
    .insert_resource(VisEnabledExtract(enabled.map(|e| e.0).unwrap_or(true)));
}

#[derive(bevy::ecs::resource::Resource, Default)]
pub struct VisEnabledExtract(pub bool);

/// 编辑代数变化 → 整表清零（16MB 一次性 DMA，编辑帧可接受；局部失效后置）。
/// enabled 每帧同步进 meta（运行时切换）。
fn prepare_vis_cache(
  mut gpu: Option<ResMut<VisCacheGpu>>,
  gen_extract: Option<Res<VisGenExtract>>,
  enabled: Option<Res<VisEnabledExtract>>,
  device: Res<RenderDevice>,
  queue: Res<RenderQueue>,
) {
  let (Some(gpu), Some(gen_extract)) = (gpu.as_mut(), gen_extract) else {
    return;
  };
  if gen_extract.0 != gpu.last_generation {
    bevy::log::info!("VisCache 清零: gen {} -> {}", gpu.last_generation, gen_extract.0);
    let zeros = vec![0u8; VIS_CAP_BYTES as usize];
    queue.write_buffer(&gpu.table, 0, &zeros);
    // normal 表 64bit 槽（空槽 = (0,0)，write 0 即空）
    let zeros_norm = vec![0u8; (VIS_CAP_SLOTS as usize) * 8];
    queue.write_buffer(&gpu.norm, 0, &zeros_norm);
    gpu.last_generation = gen_extract.0;
  }
  let enabled_now = enabled.map(|e| e.0).unwrap_or(true);
  let want = u32::from(enabled_now);
  if gpu.last_written != want {
    *gpu.meta.get_mut() = VisCacheMeta {
      enabled: want,
      capacity_mask: VIS_CAP_SLOTS - 1,
      _pad0: 0,
      _pad1: 0,
    };
    gpu.meta.write_buffer(&device, &queue);
    gpu.last_written = want;
  }
}

pub struct VisCachePlugin;

impl bevy::app::Plugin for VisCachePlugin {
  fn build(&self, app: &mut bevy::app::App) {
    app.add_systems(RenderStartup, init_vis_cache)
      .add_systems(ExtractSchedule, extract_vis_generation)
      .add_systems(Render, prepare_vis_cache.in_set(RenderSystems::PrepareResources));
  }
}
