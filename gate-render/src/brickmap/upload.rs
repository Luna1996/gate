//! 砖块图上传通道（Phase 1，Douglas 1:1）
//!
//! 裁决（遗留决策点一）：探测 `RenderDevice.limits().max_storage_buffer_binding_size`
//! 并在**首帧 info 日志打印路径选择**。默认使用**单 buffer 路径**（NVIDIA 1660/3070
//! 都 ≥2GB，最坏 b_struct < 1GB 阈值）。多 buffer 代码保留为 fallback
//! 结构（`BufferLayout::Multi` + 单测覆盖），运行期若 limit<1GB 则 warn 并退化为
//! 每帧全量上传，不 crash。
//!
//! 阶段拆分：
//! - `RenderStartup`：device/queue 就绪后创建 0 大小的 `GpuBrickMap` buffers（占位）
//! - `ExtractSchedule`（render sub-app 调度）：用 `Extract<ResMut<T>>` 访问主世界
//!   `VoxelScene(Volumes)`，按预算 drain 脏 chunk，做全量/增量 CPU 构建，将字节
//!   snapshot 写入 render world resource `UploadSnapshot`
//! - `RenderSystems::PrepareResources`：读 `UploadSnapshot` →
//!   * full：整块 `queue.write_buffer` 写 GPU buffer
//!   * incremental：仅写 Builder 记录的脏字节区间（struct/palette），
//!     典型单 chunk 编辑 → 窗口条目字 + 新树 append（KB 级），
//!     不再整块 DMA，帧率不掉
//! - info 打印 PROBE 结论 + UPLOAD[full|incremental]

use bevy::{
  log::{debug, info, warn},
  prelude::*,
  render::{
    Extract, ExtractSchedule, Render, RenderApp, RenderStartup, RenderSystems,
    render_resource::*,
    renderer::{RenderDevice, RenderQueue},
  },
};

use super::builder::{VolumesBuilder, VolumesSnapshot};
use super::wire::{
  BrickMapGlobals, CHUNK_COMP_WORDS, GridDesc, MARCH_MASK_WORDS, TREE_BASE, march_mask_lut_words,
};
use glam::{IVec3, UVec3};

// ----------------------------------------------------------------------------
// Limits + BufferLayout（CPU 单测覆盖单/多两模式）
// ----------------------------------------------------------------------------

pub const SINGLE_THRESHOLD_BYTES: u64 = (1 << 30) - 1;

#[derive(Debug, Clone, Copy)]
pub struct BindingLimits {
  pub max_storage_buffer_binding_size: u64,
}
impl BindingLimits {
  pub fn probe(device: &RenderDevice) -> Self {
    Self {
      max_storage_buffer_binding_size: device.limits().max_storage_buffer_binding_size,
    }
  }
  pub fn force_multi(&self) -> bool {
    self.max_storage_buffer_binding_size < SINGLE_THRESHOLD_BYTES
  }
}

#[derive(Debug, Clone)]
pub enum BufferLayout {
  Single,
  Multi { node_slices: usize },
}
impl BufferLayout {
  pub fn from_limits(limits: &BindingLimits) -> Self {
    if limits.force_multi() {
      let per = 64u64 << 20;
      let left = limits.max_storage_buffer_binding_size.min(per).max(4 << 20);
      let node_slices = ((TREE_BASE as u64).saturating_add(per) / left) as usize;
      Self::Multi {
        node_slices: node_slices.max(1),
      }
    } else {
      Self::Single
    }
  }
}

// ----------------------------------------------------------------------------
// 主世界 Resources
// ----------------------------------------------------------------------------

#[derive(Resource)]
pub struct VoxelScene {
  pub volumes: gate_voxel::Volumes,
  pub demo_force_full_rebuild: bool,
}

#[derive(Resource, Clone, Debug)]
pub struct UploadBudget {
  pub max_bytes_per_frame: usize,
  pub incremental: bool,
}
impl Default for UploadBudget {
  fn default() -> Self {
    Self {
      max_bytes_per_frame: 4 * 1024 * 1024,
      incremental: true,
    }
  }
}

/// 体素世界修订号（render world）：每完成一次真实上传（prepare 消费到 snapshot）自增。
/// 供依赖体素数据的下游 GPU pass（如 DDGI 探针烘焙）判定「世界是否变了」。
#[derive(Resource, Default, Clone, Copy, Debug, PartialEq, Eq)]
pub struct BrickMapRevision(pub u64);

/// DDGI 脏区（render world）：本帧上传**实际改动**的世界 voxel AABB（`max_voxel` 不含）。
///
/// `full = true` 表示全量上传（首帧 / force_full / 非增量）→ DDGI 需整体重烘；
/// 否则 [min, max) 是本次改动 chunk 的合并包围盒，DDGI 的 bake 只重烘与之相交的 cell。
/// 每帧由 extract 覆盖写入，保证不会残留上一帧的脏区。
#[derive(Resource, Default, Clone, Copy, Debug)]
pub struct BrickMapDirty {
  pub full: bool,
  pub min_voxel: IVec3,
  pub max_voxel: IVec3,
}

/// 主世界 Pending 资源：在主 world `Last` schedule 按预算 drain dirty，供只读提取
///
/// data_chunks / comp_chunks 元素 = `(volume_idx, coord)`：主世界 = 0，物体 = 1..N。
#[derive(Resource, Default)]
pub struct MainPending {
  pub force_full: bool,
  pub data_chunks: Vec<(usize, gate_voxel::ChunkCoord)>,
  pub comp_chunks: Vec<(usize, gate_voxel::ChunkCoord)>,
}

/// 单 chunk 增量重建的预算折中（典型树几十 KB~数 MB；256KB 经验值）
const PER_CHUNK_BYTES: usize = 256 * 1024;

/// 在主 world `Last` 阶段（晚于用户 Update 编辑）：按预算 drain dirty → MainPending
///
/// **关键**：进入时先清空 data_chunks / comp_chunks（上一帧的坐标已经在 ExtractSchedule
/// 被 mirror 消费）；否则坐标会逐帧累积并重复 update_chunk，带来巨量假增量上传。
/// force_full 同样在处理完一帧后复位。
///
/// Phase 3 统一：遍历 `scene.volumes.list` 的所有 volume（主世界 + 物体），
/// 每个 volume 独立 drain dirty，附带 volume_idx。
pub fn poll_pending(
  scene: Option<ResMut<VoxelScene>>,
  budget: Option<Res<UploadBudget>>,
  mut pending: ResMut<MainPending>,
) {
  let (Some(mut scene), Some(budget)) = (scene, budget) else {
    return;
  };
  pending.data_chunks.clear();
  pending.comp_chunks.clear();
  pending.force_full = false;
  if scene.demo_force_full_rebuild {
    pending.force_full = true;
    scene.demo_force_full_rebuild = false;
  }
  let budget_n = (budget.max_bytes_per_frame / PER_CHUNK_BYTES).clamp(1, 64);
  // 聚合所有 volume 的 backlog（主世界 + 物体）
  let mut total_data_backlog = 0usize;
  let mut total_comp_backlog = 0usize;
  for grid in scene.volumes.list.iter() {
    total_data_backlog = total_data_backlog.saturating_add(grid.dirty.data_dirty_count());
    total_comp_backlog = total_comp_backlog.saturating_add(grid.dirty.comp_dirty_count());
  }
  // 反推 backlog：Startup 首次构建有大量 chunk dirty，
  // 若按 budget_n 逐帧 drain，每帧 builder.update_chunk 会 CPU 阻塞冻结 Prepare
  // 全局调度 → BG1 绑定 / DDA dispatch 推迟 → 画面"只有 UI 全黑"。
  // 当 backlog > 3× 预算（即明显处于 Startup 批量构建积压，而不是零星增量编辑），
  // 一次性把 dirty 队列清空。这个判定不依赖任何外部 flag 时序，鲁棒。
  let (data_n, comp_n) = if total_data_backlog > budget_n * 3 {
    (total_data_backlog, total_comp_backlog.max(budget_n))
  } else {
    (budget_n, budget_n.min(total_comp_backlog.max(1)))
  };
  // 每个 volume 独立 drain（drain 内部按队列容量自截，data_n 是上限不是强制）
  for (vol_idx, grid) in scene.volumes.list.iter_mut().enumerate() {
    for c in grid.dirty.drain_data_budget(data_n) {
      pending.data_chunks.push((vol_idx, c));
    }
    for c in grid.dirty.drain_comp_budget(comp_n) {
      pending.comp_chunks.push((vol_idx, c));
    }
  }
}

// ----------------------------------------------------------------------------
// render world 资源
// ----------------------------------------------------------------------------

/// ExtractSchedule 用的 CPU builder / pending 状态（render world resource）
///
/// Phase 3 统一：`builder: Option<VolumesBuilder>` 持有 `Vec<BrickMapBuilder>`，
/// pending chunks 带 volume_idx。
#[derive(Resource, Default)]
pub struct BuilderMirror {
  pub builder: Option<VolumesBuilder>,
  pub pending_full: bool,
  pub pending_data_chunks: Vec<(usize, gate_voxel::ChunkCoord)>,
  pub pending_comp_chunks: Vec<(usize, gate_voxel::ChunkCoord)>,
}

/// ExtractSchedule 产出 → PrepareResources 消费（render world resource）
///
/// Phase 3 统一：`volumes: VolumesSnapshot`。full 模式带完整 `b_struct`/`b_palette`
/// 整块 DMA；incremental 模式不带整量字节（避免 100MB+ 级 memcpy 卡顿），
/// 改为 `struct_blobs`/`palette_blobs` 脏块（偏移+内容）逐块 write_buffer。
/// state/comp 仍是主世界 only（Phase 2 shader 重写后再扩展）。
#[derive(Resource, Clone)]
pub struct UploadSnapshot {
  pub volumes: VolumesSnapshot,
  pub state_bytes: Vec<u8>,
  pub comp_chunks: usize,
}

// ----------------------------------------------------------------------------
// 光照场（Douglas #15 的 AO fill + 「体素即光源」的发光密度 ε，共用一张 3D 纹理）
//
// 形态：16-voxel cube 网格（cell），相机中心、按 cell 向下对齐、世界锚定槽位
// （`slot = 世界 cell mod dims`，与 DDGI 的 `ddgi_slot` 同构）。这三个性质合起来
// 让它可流式：相机滚动只换「新进窗口的那条带」的世界 cell，其余槽位保持自己的身份，
// 不需要整幅重铺；世界编辑只重算脏 chunk。
//
// 纹理 Rgba16Unorm，dims = LIGHT_FIELD_DIM³：
//   .rgb = 发光密度 ε（Σ 发光强度 / cell 体积，0..1）→ cast 射线沿程累加 ε·L
//   .a   = 实心占比 AO fill（0.5 = 平面不压暗，≈0.75 = 夹角压暗）
// CPU 侧按 chunk 缓存 tally（每 chunk = 16³ = 4096 个 cell），见 [`LightChunk`]。
// ----------------------------------------------------------------------------

use super::dda::wgsl_consts::{LIGHT_FIELD_CELL, LIGHT_FIELD_DIM};

/// 光照场单个 chunk 的 CPU tally：16³ = 4096 个 16-voxel cell。
/// 本地序 `lx + ly*16 + lz*256`（lx/ly/lz 是 chunk 内的 cell 号，×16 即体素局部坐标）。
#[derive(Clone, Debug)]
pub struct LightChunk {
  /// 每 cell 的实心占比 ×255（0..=255）
  pub fill: Box<[u8]>,
  /// 每 cell 的发光密度 ε（0..1，= Σ 发光强度 / 4096）
  pub emit: Box<[f32]>,
}

/// 光照场 CPU 状态（render world）：按 chunk 的 tally 缓存 + 上次铺图的世界原点。
#[derive(Resource, Default)]
pub struct LightFieldCpu {
  pub cache: std::collections::HashMap<gate_voxel::ChunkCoord, LightChunk>,
  /// 上次铺图的世界原点（cell 单位，已按 cell 对齐）
  pub origin_cell: IVec3,
  /// 是否已铺过一次（首帧之前 cache 为空、origin 无意义）
  pub valid: bool,
}

/// extract → prepare：本帧要整幅重铺的光照场纹素字节
/// （Rgba16Unorm，dims³ × 8B，行主序 x 最快；bytes_per_row = dims×8 天然 256 对齐）。
#[derive(Resource, Clone)]
pub struct LightFieldUpdate {
  pub data: Vec<u8>,
}

/// 光照场窗口原点（cell 单位）：`align_down(cam - dims/2·cell, cell)`。
///
/// 与 WGSL `light_field_origin_voxel` 同一式子（shader 直接从 `view_u.cam_pos_voxel`
/// 推），两边必须逐字一致，否则 AO/发光会整体错位。
pub fn light_field_origin_cell(cam_voxel: glam::Vec3) -> IVec3 {
  let cell = LIGHT_FIELD_CELL as i32;
  let half = (LIGHT_FIELD_DIM as i32 / 2) * cell;
  let c = cam_voxel.floor().as_ivec3() - IVec3::splat(half);
  IVec3::new(c.x.div_euclid(cell), c.y.div_euclid(cell), c.z.div_euclid(cell))
}

/// 单个 chunk 的 16³ cell tally：同时产出 AO fill 与发光密度 ε（一次树查询供两者）。
///
/// 代价控制：cell 的三态 Air/Solid 一次查询即得（绝大多数 cell，Solid 还直接给出 palette
/// → ε 精确）；仅 Mixed 才展开 64 个 4³ 子块。Mixed 子块按 fill=32（半实心）近似，
/// ε 取 0 —— 发光块（灯）在调色板语义下是均质色，通常在 16³ 或 4³ 上就是 Solid，不受此近似影响。
fn build_light_chunk(tree: &gate_voxel::ChunkTree, palette: &gate_voxel::Palette) -> LightChunk {
  use gate_voxel::BrickState;
  let emis = |pal: u8| -> f32 { palette.get(pal).emissive as f32 / 255.0 };
  let mut fill = vec![0u8; 16 * 16 * 16].into_boxed_slice();
  let mut emit = vec![0f32; 16 * 16 * 16].into_boxed_slice();
  for lz in 0..16i32 {
    for ly in 0..16i32 {
      for lx in 0..16i32 {
        let bx = lx * 16;
        let by = ly * 16;
        let bz = lz * 16;
        // (实心体素数 0..4096, 发光强度之和 0..4096)
        let (v, e): (u32, f32) = match tree.get_brick_state(bx, by, bz, 2) {
          BrickState::Air => (0, 0.0),
          BrickState::Solid(pal) => (4096, 4096.0 * emis(pal)),
          BrickState::Mixed => {
            let (mut n, mut es) = (0u32, 0f32);
            for kk in 0..4i32 {
              for jj in 0..4i32 {
                for ii in 0..4i32 {
                  match tree.get_brick_state(bx + ii * 4, by + jj * 4, bz + kk * 4, 3) {
                    BrickState::Air => {}
                    BrickState::Solid(pal) => {
                      n += 64;
                      es += 64.0 * emis(pal);
                    }
                    BrickState::Mixed => n += 32,
                  }
                }
              }
            }
            (n, es)
          }
        };
        let i = (lx + ly * 16 + lz * 256) as usize;
        fill[i] = (v * 255 / 4096) as u8;
        emit[i] = e / 4096.0;
      }
    }
  }
  LightChunk { fill, emit }
}

/// P2.7 上传 CPU 耗时样本（render world 资源，由 prepare 每帧 insert_resource 覆盖。
/// render→main 同步由 P2.7 Task 2 gate-app sync_gpu_timings 内**通过 Arc<Mutex> 共享**，
/// 详见下方 [`UploadCpuSampleChannel`]。上传段的 GPU 拷贝在 submit 时发生，测不到——OQ-2 选 A。
#[derive(Resource, Clone, Copy, Debug, Default)]
pub struct UploadCpuSample {
  pub cpu_ms: f32,
  pub generation: u64,
}

/// render↔main 共享通道（Arc<Mutex>），与 bevy RenderDiagnosticsMutex 同款模式。
/// prepare（render world, PrepareResources）写入；sync_gpu_timings（main world, Update）读出。
#[derive(Resource, Clone, Debug, Default)]
pub struct UploadCpuSampleChannel(pub std::sync::Arc<std::sync::Mutex<Option<UploadCpuSample>>>);

/// GPU 资源（render world）：统一 struct/leaves/palette/comp/state + grid_descs + globals。
///
/// `grid_descs_buf`：Phase 3 新增，GridDesc 数组（144B/entry）——主世界 + 物体统一描述符，
/// shader `trace_scene` 遍历无 kind 分支。`grid_descs_count` 跟踪有效条目数。
/// `globals`：保留旧 BrickMapGlobals uniform（Phase 1 shader 字节兼容，重写后移除）。
/// `leaves`：恒空占位（Douglas 格式 palette 直存节点；BG1 binding(1) 布局占位必需）。
#[derive(Resource)]
pub struct GpuBrickMap {
  pub struct_buf: Buffer,
  pub leaves: Buffer,
  pub palette: Buffer,
  pub comp: Buffer,
  pub state: Buffer,
  pub grid_descs_buf: Buffer,
  pub grid_descs_count: u32,
  pub globals: UniformBuffer<BrickMapGlobals>,
  /// 主世界 chunk 窗口（chunk 单位）CPU 副本：DDGI 世界空间探针网格推导用
  pub main_window_origin: IVec3,
  pub main_window_dims: UVec3,
  /// 光照场（AO fill + 发光密度 ε）：Rgba16Unorm 3D 纹理 + **硬件三线性过滤**
  /// （Douglas #15："implemented as a single Hardware filtered texture read"）。
  /// 尺寸 = LIGHT_FIELD_DIM³，纹素 ↔ 一个 16-voxel cell；相机中心 + 世界锚定槽位，
  /// 由 extract 侧铺好后整幅重写（见 [`LightFieldUpdate`]）。
  /// 首次铺好前绑定 1³ 占位（采样恒 0 → AO=1、ε=0）。
  pub light_tex: Texture,
  pub light_view: TextureView,
  pub light_sampler: Sampler,
}

// ----------------------------------------------------------------------------
// RenderStartup system：创建占位 buffers（render world）
// ----------------------------------------------------------------------------

fn init_empty_gpu(device: Res<RenderDevice>, mut commands: Commands) {
  // COPY_SRC：扩容前缀拷贝（copy_buffer_to_buffer 旧→新）必需
  let make = |label: &str| -> Buffer {
    device.create_buffer(&BufferDescriptor {
      label: Some(label),
      size: 4,
      usage: BufferUsages::COPY_DST | BufferUsages::COPY_SRC | BufferUsages::STORAGE,
      mapped_at_creation: false,
    })
  };
  let globals = UniformBuffer::<BrickMapGlobals>::default();
  // 光照场占位：1³ 空纹理 + 线性过滤采样器（真实 32³ 在 prepare 里首次铺图时创建）
  let light_sampler = device.create_sampler(&SamplerDescriptor {
    label: Some("gate_light_field_sampler"),
    address_mode_u: AddressMode::ClampToEdge,
    address_mode_v: AddressMode::ClampToEdge,
    address_mode_w: AddressMode::ClampToEdge,
    mag_filter: FilterMode::Linear,
    min_filter: FilterMode::Linear,
    mipmap_filter: MipmapFilterMode::Nearest,
    ..Default::default()
  });
  let light_tex = device.create_texture(&TextureDescriptor {
    label: Some("gate_light_field"),
    size: Extent3d {
      width: 1,
      height: 1,
      depth_or_array_layers: 1,
    },
    mip_level_count: 1,
    sample_count: 1,
    dimension: TextureDimension::D3,
    format: TextureFormat::Rgba16Unorm,
    usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
    view_formats: &[],
  });
  let light_view = light_tex.create_view(&TextureViewDescriptor {
    dimension: Some(TextureViewDimension::D3),
    ..Default::default()
  });
  commands.insert_resource(GpuBrickMap {
    struct_buf: make("gate_struct"),
    leaves: make("gate_leaves"),
    palette: make("gate_palette"),
    comp: make("gate_comp"),
    state: make("gate_state"),
    grid_descs_buf: make("gate_grid_descs"),
    grid_descs_count: 0,
    globals,
    main_window_origin: IVec3::ZERO,
    main_window_dims: UVec3::ZERO,
    light_tex,
    light_view,
    light_sampler,
  });
}

// ----------------------------------------------------------------------------
// ExtractSchedule（render sub-app）：只读访问主 world 资源，CPU 构建 snapshot
//   Extract<T> 的 T 必须 ReadOnlySystemParam，所以全 Res<T>。
//   &VoxelScene.volumes 读视图 + MainPending.data_chunks(已 drained) → 就地 VolumesBuilder。
// ----------------------------------------------------------------------------

fn extract(
  mut commands: Commands,
  scene: Option<Extract<Res<VoxelScene>>>,
  budget: Option<Extract<Res<UploadBudget>>>,
  main_pending: Option<Extract<Res<MainPending>>>,
  camera: Option<Extract<Res<super::dda::DdaCameraConfig>>>,
  mut mirror: ResMut<BuilderMirror>,
  mut light_field: ResMut<LightFieldCpu>,
) {
  let (Some(scene), Some(budget), Some(main_pending)) = (scene, budget, main_pending) else {
    return;
  };
  if main_pending.force_full {
    mirror.pending_full = true;
  }
  mirror
    .pending_data_chunks
    .extend(main_pending.data_chunks.iter().copied());
  mirror
    .pending_comp_chunks
    .extend(main_pending.comp_chunks.iter().copied());

  let first = mirror.builder.is_none();
  let pending_full = std::mem::take(&mut mirror.pending_full);
  let mut pending_data: Vec<(usize, gate_voxel::ChunkCoord)> =
    std::mem::take(&mut mirror.pending_data_chunks);
  let _pending_comp: Vec<(usize, gate_voxel::ChunkCoord)> =
    std::mem::take(&mut mirror.pending_comp_chunks);
  let need_full = first || pending_full || !budget.incremental;
  let dirty_any = need_full || !pending_data.is_empty() || !_pending_comp.is_empty();

  // ---- DDGI 脏区：本帧实际重建的 chunk（仅主世界 volume 0）的合并 AABB ----
  // 全量上传 → full（DDGI 整体重烘）；增量 → 改动 chunk 的包围盒，bake 只跑相交 cell。
  // 无论有没有脏区都覆盖写入，避免上一帧的脏区残留导致重复重烘。
  let mut ddgi_dirty = BrickMapDirty::default();
  if dirty_any {
    if need_full {
      ddgi_dirty.full = true;
    } else {
      let (mut lo, mut hi): (Option<IVec3>, Option<IVec3>) = (None, None);
      for (vol_idx, c) in pending_data.iter().chain(_pending_comp.iter()) {
        if *vol_idx != 0 {
          continue;
        }
        let cl = c.0 * gate_voxel::CHUNK_SIZE;
        let ch = cl + IVec3::splat(gate_voxel::CHUNK_SIZE);
        lo = Some(lo.map_or(cl, |v| v.min(cl)));
        hi = Some(hi.map_or(ch, |v| v.max(ch)));
      }
      if let (Some(lo), Some(hi)) = (lo, hi) {
        ddgi_dirty.min_voxel = lo;
        ddgi_dirty.max_voxel = hi;
      }
    }
  }
  commands.insert_resource(ddgi_dirty);

  let volumes_ref = &scene.volumes;
  // ---- 光照场（相机中心 + 世界锚定 + 可流式）：窗口移动或脏 chunk 落入窗口才重铺 ----
  // 必须放在 `!dirty_any` 早退**之前**：相机滚动通常不带世界变化，但 AO/发光场得跟着相机走。
  // 铺图整幅重来（32³ = 256KB）只在相机跨过 16 体素边界或世界编辑时发生；tally 走缓存，
  // 新进窗口的 chunk 才需要读树（单个 chunk ≈3ms）。
  let dirty_main: Vec<gate_voxel::ChunkCoord> = if need_full {
    Vec::new() // full 会清空缓存，无需逐 chunk 剔除
  } else {
    pending_data
      .iter()
      .filter(|(v, _)| *v == 0)
      .map(|(_, c)| *c)
      .collect()
  };
  let cam_voxel = camera
    .as_ref()
    .map(|c| c.position_world)
    .unwrap_or(glam::Vec3::ZERO);
  if let Some(data) = update_light_field(
    &mut light_field,
    volumes_ref,
    cam_voxel,
    &dirty_main,
    need_full,
  ) {
    commands.insert_resource(LightFieldUpdate { data });
  }

  // 非首帧 + 非强制全量 + 无脏 chunk → 跳过体素构建/上传（省 CPU 构建 + PCIe）；
  // 光照场上面已经先行更新过（相机滚动不产生世界脏区）。
  if !dirty_any {
    return;
  }
  let builder = mirror
    .builder
    .get_or_insert_with(|| VolumesBuilder::new_unbuilt(volumes_ref));
  if need_full {
    *builder = VolumesBuilder::build_full(volumes_ref);
    pending_data.clear();
  } else {
    // 同步 volume 数量（新增物体追加 builder）+ 更新变换
    builder.sync(volumes_ref);
    for (vol_idx, c) in pending_data.drain(..) {
      builder.update_chunk(volumes_ref, vol_idx, c);
    }
  }
  let snapshot = builder.snapshot();
  // state/comp 暂仍主世界 only（Phase 2 shader 重写后再扩展到物体）
  let state_bytes = volumes_ref.main().state_table_bytes().to_vec();
  let comp_chunks = volumes_ref.main().comp_layer().len();
  commands.insert_resource(UploadSnapshot {
    volumes: snapshot,
    state_bytes,
    comp_chunks,
  });
}

/// 光照场增量更新（extract 侧）：相机中心 + 世界锚定槽位；窗口移动或脏 chunk 落入窗口时才铺。
///
/// - 窗口内缺失的 chunk → tally 进缓存；世界编辑只让脏 chunk 从缓存剔除后重算
/// - 窗口外的缓存条目 → 淘汰（缓存规模恒 ≈ 覆盖窗口的 chunk 数）
/// - 整幅铺图：dims³ 纹素，行主序 x 最快，Rgba16Unorm 8B/纹素（bytes_per_row = dims×8，
///   天然 256 对齐，故无需补行）
///
/// 世界锚定槽位的意义与 DDGI `ddgi_slot` 相同：相机滚动换掉的只是「新进窗口那条带」的
/// 世界 cell，其余纹素保持自己的身份 → 不会整幅错位/失效。
///
/// 返回 `None` = 本帧无需重铺（纹理沿用上一帧）。
fn update_light_field(
  lf: &mut LightFieldCpu,
  volumes: &gate_voxel::Volumes,
  cam_voxel: glam::Vec3,
  dirty_main: &[gate_voxel::ChunkCoord],
  force_full: bool,
) -> Option<Vec<u8>> {
  let dim = LIGHT_FIELD_DIM as i32;
  let cs = LIGHT_FIELD_CELL as i32;
  let origin_cell = light_field_origin_cell(cam_voxel);
  // 场窗口的世界 voxel AABB（max 不含）
  let win_lo = origin_cell * cs;
  let win_hi = win_lo + IVec3::splat(dim * cs);
  // 与窗口相交的 chunk 范围（chunk = CHUNK_SIZE 体素 = 16 cell）
  let csz = IVec3::splat(gate_voxel::CHUNK_SIZE);
  let c_lo = win_lo.div_euclid(csz);
  let c_hi = (win_hi - IVec3::ONE).div_euclid(csz);

  if force_full {
    lf.cache.clear();
  }
  let moved = !lf.valid || origin_cell != lf.origin_cell;
  // 脏 chunk 落在窗口内 → 剔除缓存，稍后重算（内容/palette 变了）
  let mut edited = false;
  for c in dirty_main {
    if c.0.cmpge(c_lo).all() && c.0.cmple(c_hi).all() {
      lf.cache.remove(c);
      edited = true;
    }
  }
  if !moved && !edited {
    return None;
  }

  // 补齐窗口内缺失的 chunk（世界上不存在的 chunk 不插缓存 → 该区域保持空气：fill 0 / ε 0）
  let grid = volumes.main();
  let palette = grid.palette();
  let t0 = std::time::Instant::now();
  let mut tallied = 0usize;
  for cz in c_lo.z..=c_hi.z {
    for cy in c_lo.y..=c_hi.y {
      for cx in c_lo.x..=c_hi.x {
        let coord = gate_voxel::ChunkCoord::new(cx, cy, cz);
        if lf.cache.contains_key(&coord) {
          continue;
        }
        if let Some(tree) = grid.chunk(coord) {
          lf.cache.insert(coord, build_light_chunk(tree, palette));
          tallied += 1;
        }
      }
    }
  }
  // 淘汰窗口外的条目
  lf.cache
    .retain(|c, _| c.0.cmpge(c_lo).all() && c.0.cmple(c_hi).all());

  // 整幅铺图：纹素下标 = **世界锚定槽位** `((wc mod dim) + dim) mod dim`（与 WGSL
  // `light_field_uv` 同一式子）。不能用 `wc − 窗口原点`：窗口原点是 cell 对齐而非 dim 对齐，
  // 两者差一个常量偏移 → 纹素与着色采样错位。
  let dimv = IVec3::splat(dim);
  let mut data = vec![0u8; (dim * dim * dim) as usize * 8];
  for (coord, t) in lf.cache.iter() {
    for lz in 0..16i32 {
      for ly in 0..16i32 {
        for lx in 0..16i32 {
          let wc = coord.0 * 16 + IVec3::new(lx, ly, lz);
          let rel = wc - origin_cell;
          if rel.cmplt(IVec3::ZERO).any() || rel.cmpge(dimv).any() {
            continue;
          }
          let si = (lx + ly * 16 + lz * 256) as usize;
          let r = wc.rem_euclid(dimv);
          let ti = (r.x + r.y * dim + r.z * dim * dim) as usize;
          // Rgba16Unorm：.rgb = ε、.a = fill（fill 是 0..255 → ×257 到 0..65535）
          let e = (t.emit[si].clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
          let a = (t.fill[si] as u32 * 257) as u16;
          let ob = ti * 8;
          data[ob..ob + 2].copy_from_slice(&e.to_le_bytes());
          data[ob + 2..ob + 4].copy_from_slice(&e.to_le_bytes());
          data[ob + 4..ob + 6].copy_from_slice(&e.to_le_bytes());
          data[ob + 6..ob + 8].copy_from_slice(&a.to_le_bytes());
        }
      }
    }
  }
  lf.origin_cell = origin_cell;
  lf.valid = true;
  bevy::log::debug!(
    target: "gate",
    "LIGHT FIELD: origin_cell={:?} tallied={} cached={} elapsed={:?}",
    origin_cell,
    tallied,
    lf.cache.len(),
    t0.elapsed()
  );
  Some(data)
}

// ----------------------------------------------------------------------------
// 辅助（u32 → u8 字节视图 + ensure/write buffer）
// ----------------------------------------------------------------------------

fn u8_of_u32(w: &[u32]) -> &[u8] {
  unsafe { std::slice::from_raw_parts(w.as_ptr() as *const u8, w.len() * 4) }
}

/// GridDesc 数组 → u8 字节视图（`#[repr(C)]` + 144B/entry，可直接 cast）
fn u8_of_grid_descs(descs: &[GridDesc]) -> &[u8] {
  unsafe {
    std::slice::from_raw_parts(
      descs.as_ptr() as *const u8,
      descs.len() * std::mem::size_of::<GridDesc>(),
    )
  }
}

/// GPU buffer 扩容尺寸策略（纯函数，单测覆盖）。
///
/// 旧策略 2× 翻倍的问题：b_struct 首次增长 ~10KB 就触发 163MB→327MB 翻倍 +
/// 整份 163MB PCIe 重写（首编辑 51ms 卡顿的 DMA 大头），且 VRAM 空耗近一倍。
/// 新策略：大 buffer（≥8MB，当前即 b_struct/b_leaves 大场景形态）按 32MB 水位
/// 对齐——扩容 DMA 量小、重建间隔 ≥32MB 增长；小 buffer 维持 2×（palette 等翻倍成本可忽略）。
fn grow_size(cap: u64, need: u64) -> u64 {
  const BIG: u64 = 8 << 20;
  const RESERVE: u64 = 32 << 20;
  if need >= BIG {
    need.div_ceil(RESERVE) * RESERVE
  } else {
    need.max(cap * 2).max(65536) // ≥64KB，2× amortize（原策略）
  }
}

/// 保证 buffer 能容纳 `bytes`，扩容时保留/写入新的整份内容（不丢旧字节）。
///
/// 之前的 bug（极限场景 GPU 端"全画面无体素"的根因）：
///   ensure() 只 `create_buffer(size = bytes)` 然后返回，新 buffer 全是 0，
///   之后 incremental 上传只 write_partial(dirty range)，164MB 里除了 dirty 的 6MB
///   其它区域（tile index 表 / leaves 指针 / l1/l2 表）都保持 0，tile entry = 0 被
///   WGSL sample_brickmap 当作空砖 → 全黑。
///
/// 修复：每次扩容（bytes > cur.size()）时创建更大 buffer（[`grow_size`] 水位策略）。
///
/// `prefix_valid`（增量镜像路径传 true）：bytes[0..cap) 与 GPU 现有内容一致
/// （CPU 镜像是权威拷贝、每帧脏区间全覆盖上传，故成立）→ 旧 buffer 前缀用
/// **GPU-GPU copy**（不占 PCIe，163MB <2ms）搬到新 buffer，只 write_buffer 新增
/// 尾部 [cap..need)——单次扩容 PCIe 从整份 163MB 降到增长量（10KB 级）。
/// full 重建等 GPU 旧内容不可信的场景传 false：整份 bytes 一次 DMA（原行为）。
fn ensure_with_copy(
  device: &RenderDevice,
  queue: &RenderQueue,
  cur: &mut Buffer,
  label: &str,
  bytes: &[u8],
  prefix_valid: bool,
) {
  let cap = cur.size();
  let need = bytes.len() as u64;
  if cap >= need {
    return;
  }
  let new_size = grow_size(cap, need);
  let new_buf = device.create_buffer(&BufferDescriptor {
    label: Some(label),
    size: new_size,
    // COPY_SRC：本 buffer 下次扩容时要作为前缀拷贝的源（缺它第二次 grow 必炸）
    usage: BufferUsages::COPY_DST | BufferUsages::COPY_SRC | BufferUsages::STORAGE,
    mapped_at_creation: false,
  });
  if prefix_valid && cap > 0 {
    // 前缀经 GPU-GPU 拷贝（独立 encoder submit，不占 PCIe）；随后 write_buffer 的
    // 内部 submit 与之保持提交顺序，先拷贝后写尾部。
    // COPY_BUFFER_ALIGNMENT=4；cap 恒为 words×4 或初始 4B，天然对齐。
    let mut enc = device.create_command_encoder(&CommandEncoderDescriptor {
      label: Some("gate_grow_prefix_copy"),
    });
    enc.copy_buffer_to_buffer(cur, 0, &new_buf, 0, cap);
    queue.submit([enc.finish()]);
    queue.write_buffer(&new_buf, cap, &bytes[cap as usize..need as usize]);
  } else if !bytes.is_empty() {
    queue.write_buffer(&new_buf, 0, bytes);
  }
  *cur = new_buf;
}

/// 整块写（full 模式 / state 等小 buffer）：GPU 旧内容不可信 → prefix_valid=false
fn write(device: &RenderDevice, queue: &RenderQueue, cur: &mut Buffer, label: &str, bytes: &[u8]) {
  ensure_with_copy(device, queue, cur, label, bytes, false);
  if !bytes.is_empty() {
    queue.write_buffer(cur, 0, bytes);
  }
}

/// 增量路径专用：保证 buffer 容量 ≥ `need_bytes`；扩容时创建新 buffer 并把旧内容
/// **GPU-GPU 拷贝**为前缀（不占 PCIe）。新增尾部 [cap..need) 不在这里写——
/// 调用方随后逐块 `write_buffer` 脏块，追加的树尾部恰好被脏块全覆盖。
fn ensure_capacity(
  device: &RenderDevice,
  queue: &RenderQueue,
  cur: &mut Buffer,
  label: &str,
  need_bytes: u64,
) {
  let cap = cur.size();
  if cap >= need_bytes {
    return;
  }
  let new_size = grow_size(cap, need_bytes);
  let new_buf = device.create_buffer(&BufferDescriptor {
    label: Some(label),
    size: new_size,
    // COPY_SRC：本 buffer 下次扩容时要作为前缀拷贝的源（缺它第二次 grow 必炸）
    usage: BufferUsages::COPY_DST | BufferUsages::COPY_SRC | BufferUsages::STORAGE,
    mapped_at_creation: false,
  });
  if cap > 0 {
    // COPY_BUFFER_ALIGNMENT=4；cap 恒为 words×4 或初始 4B，天然对齐。
    let mut enc = device.create_command_encoder(&CommandEncoderDescriptor {
      label: Some("gate_grow_prefix_copy"),
    });
    enc.copy_buffer_to_buffer(cur, 0, &new_buf, 0, cap);
    queue.submit([enc.finish()]);
  }
  *cur = new_buf;
}

// ----------------------------------------------------------------------------
// PrepareResources：snapshot → GPU 写；首帧探测 limits
// ----------------------------------------------------------------------------

/// 光照场整幅重铺：dims³ 纹素，行主序 x 最快，Rgba16Unorm 8B/纹素。
/// 尺寸固定（LIGHT_FIELD_DIM），仅在占位纹理不匹配时重建；此后每次只 write_texture。
fn upload_light_field(
  device: &RenderDevice,
  queue: &RenderQueue,
  gpu: &mut GpuBrickMap,
  data: &[u8],
) {
  let dim = LIGHT_FIELD_DIM;
  let want = Extent3d {
    width: dim,
    height: dim,
    depth_or_array_layers: dim,
  };
  let cur = gpu.light_tex.size();
  if cur.width != dim || cur.height != dim || cur.depth_or_array_layers != dim {
    gpu.light_tex = device.create_texture(&TextureDescriptor {
      label: Some("gate_light_field"),
      size: want,
      mip_level_count: 1,
      sample_count: 1,
      dimension: TextureDimension::D3,
      format: TextureFormat::Rgba16Unorm,
      usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
      view_formats: &[],
    });
    gpu.light_view = gpu.light_tex.create_view(&TextureViewDescriptor {
      dimension: Some(TextureViewDimension::D3),
      ..Default::default()
    });
  }
  // bytes_per_row = dim×8 = 256，天然满足 wgpu 的 256 对齐 → 无需补行
  debug_assert_eq!(data.len(), (dim as usize).pow(3) * 8);
  queue.write_texture(
    TexelCopyTextureInfo {
      texture: &gpu.light_tex,
      mip_level: 0,
      origin: Origin3d::ZERO,
      aspect: TextureAspect::All,
    },
    data,
    TexelCopyBufferLayout {
      offset: 0,
      bytes_per_row: Some(dim * 8),
      rows_per_image: Some(dim),
    },
    want,
  );
}

pub(crate) fn prepare(
  mut commands: Commands,
  snapshot: Option<ResMut<UploadSnapshot>>,
  light: Option<ResMut<LightFieldUpdate>>,
  mut gpu: ResMut<GpuBrickMap>,
  device: Res<RenderDevice>,
  queue: Res<RenderQueue>,
  sample_channel: Option<Res<UploadCpuSampleChannel>>,
  mut revision: ResMut<BrickMapRevision>,
) {
  // ---- 光照场（相机中心 + 世界锚定）：整幅重铺（Rgba16Unorm，dims³×8B）----
  // 放在 UploadSnapshot 早退之前：相机滚动只更新光照场、不带体素上传。
  if let Some(u) = light.as_ref() {
    upload_light_field(&device, &queue, &mut gpu, &u.data);
    commands.remove_resource::<LightFieldUpdate>();
  }
  let Some(snap) = snapshot else { return };
  let t0 = std::time::Instant::now();

  // ---- 首帧探测（只打一次日志）----
  static PROBED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
  let limits = BindingLimits::probe(&device);
  let layout = BufferLayout::from_limits(&limits);
  if !PROBED.swap(true, std::sync::atomic::Ordering::Relaxed) {
    let gb = limits.max_storage_buffer_binding_size as f64 / (1 << 30) as f64;
    info!(target: "gate",
      "PROBE: max_storage_buffer_binding_size = {gb:.2}GB → {}",
      match layout { BufferLayout::Single => "单 buffer 路径", BufferLayout::Multi{..} => "多 buffer fallback 路径" },
    );
    if limits.force_multi() {
      warn!(target: "gate",
        "GPU storage binding < 1GB，退化到全量上传（单 tile 170MB 会跨帧）——P2.3 多 buffer 切片代码保留，需要时接入"
      );
    }
  }

  let is_full = matches!(snap.volumes.mode_tag, "full" | "fallback_full");
  let comp_bytes = snap.comp_chunks * CHUNK_COMP_WORDS * 4;

  // P4：方向可达掩码 LUT（Douglas #18 Bitwise Masking）→ b_leaves。
  // 全局常量（8 octant × 64 入口格 × 2 u32 = 4KB），与 volume 无关；
  // 只在 buffer 尚未容纳时写一次，之后零 PCIe。
  {
    let lut_need = (MARCH_MASK_WORDS * 4) as u64;
    if gpu.leaves.size() < lut_need {
      let lut = march_mask_lut_words();
      write(
        &device,
        &queue,
        &mut gpu.leaves,
        "gate_leaves",
        u8_of_u32(&lut),
      );
    }
  }

  // 传输字节统计（日志用）
  let (struct_tx_bytes, palette_tx_bytes, grid_descs_tx_bytes);

  if is_full {
    // 全量：整块 DMA + ensure 保证 GPU 容量够
    let struct_bytes = u8_of_u32(&snap.volumes.b_struct);
    let palette_bytes = u8_of_u32(&snap.volumes.b_palette);
    let grid_descs_bytes = u8_of_grid_descs(&snap.volumes.grid_descs);
    struct_tx_bytes = struct_bytes.len();
    palette_tx_bytes = palette_bytes.len();
    grid_descs_tx_bytes = grid_descs_bytes.len();
    write(
      &device,
      &queue,
      &mut gpu.struct_buf,
      "gate_struct",
      struct_bytes,
    );
    write(
      &device,
      &queue,
      &mut gpu.palette,
      "gate_palette",
      palette_bytes,
    );
    write(
      &device,
      &queue,
      &mut gpu.grid_descs_buf,
      "gate_grid_descs",
      grid_descs_bytes,
    );
  } else {
    // 增量路径：snapshot 不再全量拼接 b_struct（旧实现每个脏帧 memcpy 100MB+，
    // 是编辑 spike 的根因）。这里只按总字节 ensure 容量（扩容走 GPU-GPU 前缀
    // 拷贝，不占 PCIe），随后逐脏块 write_buffer——每块 = chunk 窗口条目字 +
    // 新 append 的树尾部（KB~MB 级）；追加尾部恰好覆盖扩容后的新区域。
    // GridDesc：volume 数变化已由 snapshot 判 need_full；增量路径内容不变，
    // 仅 ensure 维持容量。leaves 恒空，跳过。
    ensure_capacity(
      &device,
      &queue,
      &mut gpu.struct_buf,
      "gate_struct",
      snap.volumes.struct_total_bytes as u64,
    );
    ensure_capacity(
      &device,
      &queue,
      &mut gpu.palette,
      "gate_palette",
      snap.volumes.palette_total_bytes as u64,
    );
    let grid_descs_bytes = u8_of_grid_descs(&snap.volumes.grid_descs);
    ensure_capacity(
      &device,
      &queue,
      &mut gpu.grid_descs_buf,
      "gate_grid_descs",
      grid_descs_bytes.len() as u64,
    );

    let mut s_tx = 0usize;
    for (off, payload) in &snap.volumes.struct_blobs {
      queue.write_buffer(&gpu.struct_buf, *off as u64, payload);
      s_tx += payload.len();
    }
    let mut p_tx = 0usize;
    for (off, payload) in &snap.volumes.palette_blobs {
      queue.write_buffer(&gpu.palette, *off as u64, payload);
      p_tx += payload.len();
    }
    struct_tx_bytes = s_tx;
    palette_tx_bytes = p_tx;
    grid_descs_tx_bytes = 0;
  }
  // state / comp 每次都整块写（state 4KB、comp 每 chunk 8KB，都很小）
  write(
    &device,
    &queue,
    &mut gpu.state,
    "gate_state",
    &snap.state_bytes,
  );
  // comp: 每 chunk 8KB 占位；build 后可能为 0 字节，ensure 至少 4B。
  // comp_bytes 只是预估上限；实际内容读 grid 时已经按真实 size 存。
  // comp 数据直接从 CPU 侧构建：UploadSnapshot 当前没带 comp 字节，
  // 这里用 gpu.comp size ≥ 预估的占位（历史行为：仅 buffer 大小对齐）。
  {
    let placeholder = vec![0u8; comp_bytes.max(4)];
    // comp 为占位通道（内容无意义，历史行为仅对齐 buffer 大小）；保留旧前缀即可
    ensure_with_copy(
      &device,
      &queue,
      &mut gpu.comp,
      "gate_comp",
      &placeholder,
      true,
    );
  }

  // globals：从主世界 GridDesc[0] 构造向后兼容 BrickMapGlobals（Phase 1 shader 字节兼容，
  // Phase 2 重写 dda.wgsl 后移除——届时 shader 走 grid_descs_buf，不再读 globals）。
  let main_desc = snap.volumes.grid_descs.first().copied().unwrap_or_default();
  let globals = BrickMapGlobals {
    index_origin_x: main_desc.index_origin_x,
    index_origin_y: main_desc.index_origin_y,
    index_origin_z: main_desc.index_origin_z,
    index_origin_w: 0,
    index_dims_x: main_desc.index_dims_x,
    index_dims_y: main_desc.index_dims_y,
    index_dims_z: main_desc.index_dims_z,
    index_dims_w: 0,
    tile_count: main_desc.chunk_count,
    // node_words / node_free_words 等字段在 VolumesBuilder 内部，未暴露；
    // shader 重写后这些字段不再使用，此处置 0 不影响 Phase 2 之后的路径。
    node_words: 0,
    node_free_words: 0,
    brick_slabs: 0,
    brick_free: 0,
    rejected_tiles: 0,
    grid_count: snap.volumes.grid_descs.len() as u32,
    _pad1: 0,
    _pad2: 0,
    _pad3: 0,
    _pad4: 0,
  };
  gpu.globals.set(globals);
  gpu.globals.write_buffer(&device, &queue);

  gpu.grid_descs_count = snap.volumes.grid_descs.len() as u32;
  // 主世界 chunk 窗口 CPU 副本（DDGI 世界探针网格推导）
  gpu.main_window_origin = IVec3::new(
    main_desc.index_origin_x,
    main_desc.index_origin_y,
    main_desc.index_origin_z,
  );
  gpu.main_window_dims = UVec3::new(
    main_desc.index_dims_x,
    main_desc.index_dims_y,
    main_desc.index_dims_z,
  );

  // 世界数据已更新 → 修订号自增（下游 GPU pass 据此触发重烘焙）
  revision.0 = revision.0.wrapping_add(1);

  let elapsed = t0.elapsed();
  let cpu_ms = elapsed.as_secs_f32() * 1000.0;
  let tx_bytes_total =
    (struct_tx_bytes + palette_tx_bytes + grid_descs_tx_bytes + snap.state_bytes.len()) as f64;
  let mb = tx_bytes_total / (1 << 20) as f64;
  // full: chunks = 所有 volume 的 chunk_count 之和；incremental: chunks = 本轮 dirty 数
  let chunks_show = if is_full {
    snap
      .volumes
      .grid_descs
      .iter()
      .map(|g| g.chunk_count as usize)
      .sum::<usize>()
  } else {
    snap.volumes.dirty_chunks
  };
  debug!(target: "gate",
    "UPLOAD[{}]: bytes={:.2}MB (s {}KB,pal {}KB,gd {}KB,state 4KB), chunks={}, comp={}KB, elapsed={:?}",
    snap.volumes.mode_tag, mb,
    struct_tx_bytes / 1024, palette_tx_bytes / 1024,
    grid_descs_tx_bytes / 1024, chunks_show, comp_bytes / 1024, elapsed,
  );
  // P2.7：写入共享通道（render↔main Arc<Mutex>，OQ-2 选 A 不提供 GPU 值）
  static SAMPLE_GEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
  let generation = SAMPLE_GEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
  let sample = UploadCpuSample { cpu_ms, generation };
  commands.insert_resource(sample);
  if let Some(ch) = sample_channel
    && let Ok(mut g) = ch.0.lock()
  {
    *g = Some(sample);
  }
  // VRAM 规模留档（v3.9.1 用户指令：2GB 内存预算断言取消，仅打印不拦截）
  let vram = gpu.struct_buf.size()
    + gpu.leaves.size()
    + gpu.palette.size()
    + gpu.comp.size()
    + gpu.state.size()
    + gpu.grid_descs_buf.size();
  if vram > 2u64 << 30 {
    bevy::log::warn!("GPU VRAM 超过旧预算线（仅提示）: {vram} bytes");
  }
  if !limits.force_multi() {
    debug_assert!(
      snap.volumes.struct_total_bytes as u64 <= limits.max_storage_buffer_binding_size,
      "b_struct {} bytes > binding limit {}",
      snap.volumes.struct_total_bytes,
      limits.max_storage_buffer_binding_size,
    );
  }
  debug!(target: "gate",
    "GpuBrickMap: struct_buf={}B leaves={}B palette={}B comp={}B state={}B grid_descs={}B(count={}) bind_group_ready=pending(P2.4)",
    gpu.struct_buf.size(), gpu.leaves.size(), gpu.palette.size(),
    gpu.comp.size(), gpu.state.size(), gpu.grid_descs_buf.size(), gpu.grid_descs_count,
  );
  // 消费完本帧 snapshot 必须移除；否则 prepare 每帧都读旧 snapshot → 140MB/帧假上传
  commands.remove_resource::<UploadSnapshot>();
}

// ----------------------------------------------------------------------------
// Plugin
// ----------------------------------------------------------------------------

/// 统一体素渲染上传插件（主世界 + 物体同一路径）
///
/// Phase 3 OBJ→Volume 统一后，OBJ 不再有独立 ObjScene/RenderObj/GpuObjPool 三段
/// 管道，而是作为 `Volumes.list[1..N]` 中的普通 `VolumeGrid`，走与主世界完全相同的
/// dirty → VolumesBuilder → UploadSnapshot 增量上传路径。
pub struct VolumePlugin;
impl Plugin for VolumePlugin {
  fn build(&self, app: &mut App) {
    // Render↔main 共享通道（UploadCpuSample）：插入同一个 Arc<Mutex> Resource 到两个世界
    let ch = UploadCpuSampleChannel::default();
    app
      .insert_resource(ch.clone())
      .init_resource::<MainPending>()
      .add_systems(Last, poll_pending);
    let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
      return;
    };
    render_app
      .insert_resource(ch)
      .init_resource::<BrickMapRevision>()
      .init_resource::<LightFieldCpu>()
      .insert_resource(BuilderMirror {
        pending_full: true,
        ..Default::default()
      })
      .add_systems(RenderStartup, init_empty_gpu)
      .add_systems(ExtractSchedule, extract)
      .add_systems(Render, prepare.in_set(RenderSystems::PrepareResources));
  }
}

// ----------------------------------------------------------------------------
// CPU 单测（不依赖 GPU）
// ----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
  use super::super::builder::BrickMapBuilder;
  use super::super::wire::{CHUNK_INDEX_WORDS, STATE_TOTAL_WORDS};
  use super::*;
  use gate_voxel::fill_box;

  /// 光照场窗口原点：按 cell 对齐（世界锚定槽位 `wc mod dim` 的铺图依赖它整除精确）
  /// + 相机恒落在盒中心（偏差 < 1 cell）。
  #[test]
  fn light_field_origin_is_cell_aligned_and_centered() {
    let cell = LIGHT_FIELD_CELL as i32;
    let dim = LIGHT_FIELD_DIM as i32;
    for cam in [
      glam::Vec3::new(0.0, 0.0, 0.0),
      glam::Vec3::new(1000.3, -333.7, 7.2),
      glam::Vec3::new(-1.5, -7.5, 12345.9),
    ] {
      let o_vox = light_field_origin_cell(cam) * cell;
      assert_eq!(o_vox.rem_euclid(IVec3::splat(cell)), IVec3::ZERO);
      let rel = cam.floor().as_ivec3() - o_vox;
      assert!(
        rel.cmpge(IVec3::ZERO).all() && rel.cmplt(IVec3::splat(dim * cell)).all(),
        "cam {cam:?} 不在场窗口内"
      );
      let d = (rel - IVec3::splat(dim / 2 * cell)).abs();
      assert!(d.cmplt(IVec3::splat(cell)).all(), "cam {cam:?} 偏离盒心 {d:?}");
    }
  }

  /// 光照场纹素 8B → bytes_per_row = dim×8；upload 依赖它天然满足 wgpu 的 256 对齐（不补行）。
  /// 并锁死「chunk 恰好是整数个 cell」（tally 的 chunk↔cell 换算依赖它）。
  #[test]
  fn light_field_row_alignment() {
    assert_eq!((LIGHT_FIELD_DIM * 8) % 256, 0);
    assert_eq!(gate_voxel::CHUNK_SIZE % LIGHT_FIELD_CELL as i32, 0);
    assert_eq!(gate_voxel::CHUNK_SIZE / LIGHT_FIELD_CELL as i32, 16);
  }

  fn emissive_grid() -> gate_voxel::Volumes {
    let mut g = gate_voxel::VolumeGrid::new();
    let mut e = gate_voxel::PaletteEntry::default();
    e.emissive = 255;
    g.palette_mut().set(1, e);
    // 单个 16³ 实心发光块 = 恰好一个 cell（世界 cell (24,24,24) = 体素 [384,400)）
    fill_box(&mut g, IVec3::splat(384), IVec3::splat(16), 1);
    gate_voxel::Volumes::new(g)
  }

  fn slot_texel(dim: i32, wc: IVec3) -> usize {
    let r = wc.rem_euclid(IVec3::splat(dim));
    (r.x + r.y * dim + r.z * dim * dim) as usize
  }

  /// tally：整块实心发光 → fill=255、ε=1；相邻空气 cell → 0。
  #[test]
  fn light_chunk_tally_fill_and_emissive() {
    let vol = emissive_grid();
    let tree = vol.main().chunk(gate_voxel::ChunkCoord::new(1, 1, 1)).unwrap();
    let t = build_light_chunk(tree, vol.main().palette());
    let solid = 8 + 8 * 16 + 8 * 256; // chunk-local cell (8,8,8)
    assert_eq!(t.fill[solid], 255);
    assert!((t.emit[solid] - 1.0).abs() < 1e-6);
    assert_eq!(t.fill[0], 0);
    assert_eq!(t.emit[0], 0.0);
  }

  /// 铺图必须用**世界锚定槽位** `wc mod dim`（与 WGSL `light_field_uv` 同式子）：
  /// 同一个世界 cell 在两处相机位置下都在窗口内时 → 同一个纹素、内容不变。
  /// 这正是「相机滚动只换新进那条带、其余纹素身份不变」（可流式）的代数形式。
  #[test]
  fn light_field_blit_is_world_anchored() {
    let dim = LIGHT_FIELD_DIM as i32;
    let cell = LIGHT_FIELD_CELL as i32;
    let vol = emissive_grid();
    let ti = slot_texel(dim, IVec3::splat(24));
    let read = |origin_cell: i32, lf: &mut LightFieldCpu| -> [u8; 8] {
      let cam = glam::Vec3::splat((origin_cell * cell + dim / 2 * cell) as f32);
      let data = update_light_field(lf, &vol, cam, &[], false).expect("窗口移动应重铺");
      assert_eq!(light_field_origin_cell(cam).x, origin_cell);
      let mut out = [0u8; 8];
      out.copy_from_slice(&data[ti * 8..ti * 8 + 8]);
      out
    };
    let mut lf = LightFieldCpu::default();
    let a = read(16, &mut lf);
    let b = read(24, &mut lf); // 窗口整体滚 8 个 cell，cell 24 仍在窗口内
    // Rgba16Unorm：ε = 1 → 65535、fill = 255 → 65535
    assert_eq!(a, [0xFFu8; 8], "发光 cell 未落在世界锚定槽位");
    assert_eq!(a, b, "相机移动后同一世界 cell 的纹素内容变了");
  }

  #[test]
  fn limits_select_layout() {
    let multi = BindingLimits {
      max_storage_buffer_binding_size: 128 * (1 << 20),
    };
    assert!(multi.force_multi());
    let single = BindingLimits {
      max_storage_buffer_binding_size: 2 * (1 << 30),
    };
    assert!(!single.force_multi());
    let _ml = BufferLayout::from_limits(&multi);
    let _sl = BufferLayout::from_limits(&single);
  }

  #[test]
  fn comp_state_api_on_grid() {
    let mut g = gate_voxel::VolumeGrid::new();
    let chunk = gate_voxel::ChunkCoord::new(0, 0, 0);
    // set_comp 挂 level 2 brick (bx,by,bz)；get_comp 查 voxel 所在 brick
    g.set_comp(chunk, 1, 1, 1, 0xABCD);
    let voxel = gate_voxel::VoxelCoord::new(16 + 5, 16 + 3, 16 + 2);
    assert_eq!(g.get_comp(voxel), 0xABCD);
    assert_eq!(g.get_comp(gate_voxel::VoxelCoord::new(1, 1, 1)), 0);
    g.set_state(7, 2, 0x42);
    assert_eq!(g.get_state(7, 2), 0x42);
    assert_eq!(g.get_state(99, 0), 0);
    assert_eq!(g.state_table_bytes().len(), 256 * 16);
  }

  #[test]
  fn wire_constants() {
    assert_eq!(CHUNK_COMP_WORDS * 4, 4096 * 2); // u16[4096] → 8KB/chunk
    assert_eq!(STATE_TOTAL_WORDS * 4, 4096); // 256×4×4B
  }

  #[test]
  fn grow_size_watermark_policy() {
    // 小 buffer：2×（原策略），下限 64KB
    assert_eq!(grow_size(4, 2048), 65536); // palette 首扩
    assert_eq!(grow_size(32768, 40000), 65536);
    assert_eq!(grow_size(40000, 50000), 80000); // 恰好 2×
    // 大 buffer：32MiB 水位对齐——首增 ~10KB 不再翻倍到 2×cap
    let cap160m: u64 = 163_798_052; // 156.2MiB（实机 full 后 b_struct）
    let need = cap160m + 10 * 1024; // 首次编辑真实增长 ~10KB
    let grown = grow_size(cap160m, need);
    assert_eq!(grown, 160 * 1024 * 1024); // → 160MiB（下一 32MiB 边界），非 2×=312MiB
    assert!(grown >= need);
    // 水位内的增长由 ensure_with_copy 的 cap>=need 早退拦截，不会进 grow_size；
    // 刚跨过边界 → 立即扩到下一档（amortized）
    assert_eq!(grow_size(grown, grown + 1024), 192 * 1024 * 1024);
    // 连续跨档（160MiB+33MiB=193MiB → 224MiB）
    assert_eq!(
      grow_size(grown, grown + 33 * 1024 * 1024),
      224 * 1024 * 1024
    );
  }

  #[test]
  fn buffer_data_roundtrip() {
    let mut g = gate_voxel::VolumeGrid::new();
    fill_box(&mut g, glam::IVec3::ZERO, glam::IVec3::splat(8), 1);
    g.set_state(5, 3, 0xCAFEBABE);
    let state = g.state_table_bytes();
    let off = 5 * 16 + 3 * 4; // entry 5 + field 3
    assert_eq!(state[off..off + 4], 0xCAFEBABEu32.to_le_bytes());
    let buffers = BrickMapBuilder::build_full(&g).buffers().clone();
    let bytes = u8_of_u32(&buffers.b_struct);
    assert_eq!(bytes.len(), buffers.b_struct.len() * 4);
    assert!(buffers.globals.tile_count >= 1);
    // chunk 窗口条目非零（Region ① 至少 1 个 chunk）
    assert!(
      !buffers.b_struct[..CHUNK_INDEX_WORDS]
        .iter()
        .all(|&w| w == 0)
    );
  }
}
