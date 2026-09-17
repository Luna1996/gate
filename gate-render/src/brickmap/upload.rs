//! 砖块图上传通道：主世界与物体同一路径（dirty → VolumesBuilder → UploadSnapshot → GPU）。
//! 阶段：`RenderStartup` 建占位 buffers；`ExtractSchedule` 按预算 drain 脏 chunk 构建 snapshot；
//! `PrepareResources` 整块或按脏字节区间写 GPU；binding limit < 1GB 时退化为每帧全量上传。

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

#[derive(Debug, Clone, Copy)]
pub struct BindingLimits {
  pub max_storage_buffer_binding_size: u64,
}
impl BindingLimits {
  pub fn probe(device: &RenderDevice) -> Self {
    Self { max_storage_buffer_binding_size: device.limits().max_storage_buffer_binding_size }
  }
  pub fn force_multi(&self) -> bool {
    self.max_storage_buffer_binding_size < crate::brickmap::consts::SINGLE_THRESHOLD_BYTES
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
      let per = crate::brickmap::consts::BUFFER_SLICE_TARGET;
      let left = limits
        .max_storage_buffer_binding_size
        .min(per)
        .max(crate::brickmap::consts::BUFFER_SLICE_MIN);
      let node_slices = ((TREE_BASE as u64).saturating_add(per) / left) as usize;
      Self::Multi { node_slices: node_slices.max(1) }
    } else {
      Self::Single
    }
  }
}

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
    Self { max_bytes_per_frame: crate::brickmap::consts::UPLOAD_BYTES_PER_FRAME, incremental: true }
  }
}

/// 体素世界修订号（render world）：每完成一次真实上传（prepare 消费到 snapshot）自增。
/// 供依赖体素数据的下游 GPU pass（如 DDGI 探针烘焙）判定「世界是否变了」。
#[derive(Resource, Default, Clone, Copy, Debug, PartialEq, Eq)]
pub struct BrickMapRevision(pub u64);

/// DDGI 脏区（render world）：本帧上传实际改动的世界 voxel AABB（`max_voxel` 不含）。
/// `full = true` = 全量上传（DDGI 整体重烘）；`full = false` 时 [min, max) 为改动 chunk 合并包围盒。
#[derive(Resource, Default, Clone, Copy, Debug)]
pub struct BrickMapDirty {
  pub full: bool,
  pub min_voxel: IVec3,
  pub max_voxel: IVec3,
}

/// 主世界 Pending 资源：在主 world `Last` schedule 按预算 drain dirty，供只读提取。
/// `data_chunks` / `comp_chunks` 元素 = `(volume_idx, coord)`：主世界 = 0，物体 = 1..N。
#[derive(Resource, Default)]
pub struct MainPending {
  pub force_full: bool,
  pub data_chunks: Vec<(usize, gate_voxel::ChunkCoord)>,
  pub comp_chunks: Vec<(usize, gate_voxel::ChunkCoord)>,
  /// 与 `data_chunks` 并行的编辑 AABB：`(volume_idx, coord, lo, hi)`，闭开世界 voxel 区间。
  pub data_aabbs: Vec<(usize, gate_voxel::ChunkCoord, IVec3, IVec3)>,
}

/// 在主 world `Last` 阶段按预算 drain dirty → `MainPending`，遍历所有 volume 附带 `volume_idx`。
/// 进入时清空 `data_chunks` / `comp_chunks` / `data_aabbs`；`force_full` 处理一帧后复位。
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
  pending.data_aabbs.clear();
  pending.force_full = false;
  if scene.demo_force_full_rebuild {
    pending.force_full = true;
    scene.demo_force_full_rebuild = false;
  }
  let budget_n =
    (budget.max_bytes_per_frame / crate::brickmap::consts::PER_CHUNK_BYTES).clamp(1, 64);

  let mut total_data_backlog = 0usize;
  let mut total_comp_backlog = 0usize;
  for grid in scene.volumes.list.iter() {
    total_data_backlog = total_data_backlog.saturating_add(grid.dirty.data_dirty_count());
    total_comp_backlog = total_comp_backlog.saturating_add(grid.dirty.comp_dirty_count());
  }

  let (data_n, comp_n) = if total_data_backlog > budget_n * 3 {
    (total_data_backlog, total_comp_backlog.max(budget_n))
  } else {
    (budget_n, budget_n.min(total_comp_backlog.max(1)))
  };

  for (vol_idx, grid) in scene.volumes.list.iter_mut().enumerate() {
    for c in grid.dirty.drain_data_budget(data_n) {
      pending.data_chunks.push((vol_idx, c));

      if let Some((lo, hi)) = grid.take_edit_aabb(c) {
        pending.data_aabbs.push((vol_idx, c, lo, hi));
      }
    }
    for c in grid.dirty.drain_comp_budget(comp_n) {
      pending.comp_chunks.push((vol_idx, c));
    }
  }
}

/// ExtractSchedule 用的 CPU builder / pending 状态（render world resource）
/// `builder: Option<VolumesBuilder>` 持有 `Vec<BrickMapBuilder>`；pending chunks 带 volume_idx。
#[derive(Resource, Default)]
pub struct BuilderMirror {
  pub builder: Option<VolumesBuilder>,
  pub pending_full: bool,
  pub pending_data_chunks: Vec<(usize, gate_voxel::ChunkCoord)>,
  /// 与 `pending_data_chunks` 并行的编辑 AABB（`(volume_idx, coord, lo, hi)`）
  pub pending_data_aabbs: Vec<(usize, gate_voxel::ChunkCoord, IVec3, IVec3)>,
  /// 各 volume 调色板上次同步的写版本（判断「只改材质」是否需上传）。
  pub palette_versions: Vec<u64>,
  pub pending_comp_chunks: Vec<(usize, gate_voxel::ChunkCoord)>,
}

/// ExtractSchedule 产出 → PrepareResources 消费（render world resource）。
/// full 模式带整块 `b_struct`/`b_palette`；incremental 模式走 `struct_blobs`/`palette_blobs` 脏块。
#[derive(Resource, Clone)]
pub struct UploadSnapshot {
  pub volumes: VolumesSnapshot,
  pub state_bytes: Vec<u8>,
  pub comp_chunks: usize,
}

// 光照场（AO fill）：16-voxel cell 网格，相机中心 + 世界锚定槽位，可流式。
// 纹理 Rgba16Unorm，dims = LIGHT_FIELD_DIM³：`.a` = 实心占比 AO fill，`.rgb` 恒 0。

use super::dda::wgsl_consts::{LIGHT_FIELD_CELL, LIGHT_FIELD_DIM};

/// 光照场单个 chunk 的 CPU tally：16³ = 4096 个 16-voxel cell。
/// 本地序 `lx + ly*16 + lz*256`（lx/ly/lz 是 chunk 内的 cell 号，×16 即体素局部坐标）。
#[derive(Clone, Debug)]
pub struct LightChunk {
  /// 每 cell 的实心占比 ×255（0..=255）
  pub fill: Box<[u8]>,
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

/// extract → prepare：本帧整幅重铺的光照场纹素字节（Rgba16Unorm，dims³ × 8B，行主序 x 最快）。
#[derive(Resource, Clone)]
pub struct LightFieldUpdate {
  pub data: Vec<u8>,
}

/// 光照场窗口原点（cell 单位）：`align_down(cam - dims/2·cell, cell)`。
/// 必须与 WGSL `light_field_origin_voxel` 逐字一致（shader 由 `view_u.cam_pos_voxel` 推）。
pub fn light_field_origin_cell(cam_voxel: glam::Vec3) -> IVec3 {
  let cell = LIGHT_FIELD_CELL as i32;
  let half = (LIGHT_FIELD_DIM as i32 / 2) * cell;
  let c = cam_voxel.floor().as_ivec3() - IVec3::splat(half);
  IVec3::new(c.x.div_euclid(cell), c.y.div_euclid(cell), c.z.div_euclid(cell))
}

/// 单个 chunk 的 16³ cell tally：产出 AO fill。
/// 仅 Mixed 展开 4³ 子块；Mixed 子块按半实心（fill=32）近似。
fn build_light_chunk(tree: &gate_voxel::ChunkTree) -> LightChunk {
  use gate_voxel::BrickState;
  let mut fill = vec![0u8; 16 * 16 * 16].into_boxed_slice();
  for lz in 0..16i32 {
    for ly in 0..16i32 {
      for lx in 0..16i32 {
        let bx = lx * 16;
        let by = ly * 16;
        let bz = lz * 16;

        let v: u32 = match tree.get_brick_state(bx, by, bz, 2) {
          BrickState::Air => 0,
          BrickState::Solid(_) => 4096,
          BrickState::Mixed => {
            let mut n = 0u32;
            for kk in 0..4i32 {
              for jj in 0..4i32 {
                for ii in 0..4i32 {
                  match tree.get_brick_state(bx + ii * 4, by + jj * 4, bz + kk * 4, 3) {
                    BrickState::Air => {}
                    BrickState::Solid(_) => n += 64,
                    BrickState::Mixed => n += 32,
                  }
                }
              }
            }
            n
          }
        };
        let i = (lx + ly * 16 + lz * 256) as usize;
        fill[i] = (v * 255 / 4096) as u8;
      }
    }
  }
  LightChunk { fill }
}

/// 上传 CPU 耗时样本（render world 资源，由 prepare 每帧覆盖）。
/// render→main 同步走 [`UploadCpuSampleChannel`]；GPU 拷贝发生在 submit 时，不计入此值。
#[derive(Resource, Clone, Copy, Debug, Default)]
pub struct UploadCpuSample {
  pub cpu_ms: f32,
  pub generation: u64,
}

/// render↔main 共享通道（`Arc<Mutex>`）。
/// prepare（render world, PrepareResources）写入；sync_gpu_timings（main world, Update）读出。
#[derive(Resource, Clone, Debug, Default)]
pub struct UploadCpuSampleChannel(pub std::sync::Arc<std::sync::Mutex<Option<UploadCpuSample>>>);

/// GPU 资源（render world）：统一 struct/leaves/palette/comp/state + grid_descs + globals。
/// `leaves` 存方向可达掩码 LUT（BG1 binding(1) 占位）；`grid_descs_count` 为有效条目数。
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
  /// 光照场（AO fill）：Rgba16Unorm 3D 纹理，硬件三线性过滤，仅 `.a` 有意义（尺寸 LIGHT_FIELD_DIM³）。
  pub light_tex: Texture,
  pub light_view: TextureView,
  pub light_sampler: Sampler,
}

fn init_empty_gpu(device: Res<RenderDevice>, mut commands: Commands) {
  // COPY_SRC：扩容时前缀拷贝（`copy_buffer_to_buffer`）必需。
  let make = |label: &str| -> Buffer {
    device.create_buffer(&BufferDescriptor {
      label: Some(label),
      size: 4,
      usage: BufferUsages::COPY_DST | BufferUsages::COPY_SRC | BufferUsages::STORAGE,
      mapped_at_creation: false,
    })
  };
  let globals = UniformBuffer::<BrickMapGlobals>::default();

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
    size: Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
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

// ExtractSchedule（render sub-app）：只读访问主 world 资源，CPU 构建 snapshot。
// `Extract<T>` 的 T 必须 ReadOnlySystemParam，全部用 `Res<T>`。

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
  mirror.pending_data_chunks.extend(main_pending.data_chunks.iter().copied());
  mirror.pending_data_aabbs.extend(main_pending.data_aabbs.iter().copied());
  mirror.pending_comp_chunks.extend(main_pending.comp_chunks.iter().copied());

  let first = mirror.builder.is_none();
  let pending_full = std::mem::take(&mut mirror.pending_full);
  let mut pending_data: Vec<(usize, gate_voxel::ChunkCoord)> =
    std::mem::take(&mut mirror.pending_data_chunks);
  let pending_aabbs: Vec<(usize, gate_voxel::ChunkCoord, IVec3, IVec3)> =
    std::mem::take(&mut mirror.pending_data_aabbs);
  let _pending_comp: Vec<(usize, gate_voxel::ChunkCoord)> =
    std::mem::take(&mut mirror.pending_comp_chunks);
  let need_full = first || pending_full || !budget.incremental;

  let palette_dirty = scene
    .volumes
    .list
    .iter()
    .enumerate()
    .any(|(i, g)| mirror.palette_versions.get(i).copied() != Some(g.palette().version()));
  let dirty_any =
    need_full || !pending_data.is_empty() || !_pending_comp.is_empty() || palette_dirty;

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

        let (cl, ch) = match pending_aabbs.iter().find(|(v, cc, ..)| *v == 0 && cc == c) {
          Some((_, _, alo, ahi)) => (*alo, *ahi),
          None => {
            let cl = c.0 * gate_voxel::CHUNK_SIZE;
            (cl, cl + IVec3::splat(gate_voxel::CHUNK_SIZE))
          }
        };
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

  let dirty_main: Vec<gate_voxel::ChunkCoord> = if need_full {
    Vec::new()
  } else {
    pending_data.iter().filter(|(v, _)| *v == 0).map(|(_, c)| *c).collect()
  };
  let cam_voxel = camera.as_ref().map(|c| c.position_world).unwrap_or(glam::Vec3::ZERO);
  if let Some(data) =
    update_light_field(&mut light_field, volumes_ref, cam_voxel, &dirty_main, need_full)
  {
    commands.insert_resource(LightFieldUpdate { data });
  }

  if !dirty_any {
    return;
  }
  let builder = mirror.builder.get_or_insert_with(|| VolumesBuilder::new_unbuilt(volumes_ref));
  if need_full {
    *builder = VolumesBuilder::build_full(volumes_ref);
    pending_data.clear();
  } else {
    builder.sync(volumes_ref);
    for (vol_idx, c) in pending_data.drain(..) {
      builder.update_chunk(volumes_ref, vol_idx, c);
    }
  }

  builder.sync_palettes(volumes_ref);
  let snapshot = builder.snapshot();

  mirror.palette_versions = scene.volumes.list.iter().map(|g| g.palette().version()).collect();

  let state_bytes = volumes_ref.main().state_table_bytes().to_vec();
  let comp_chunks = volumes_ref.main().comp_layer().len();
  commands.insert_resource(UploadSnapshot { volumes: snapshot, state_bytes, comp_chunks });
}

/// 光照场增量更新（extract 侧）：窗口移动或脏 chunk 落入窗口时才整幅重铺；返回 `None` = 无需重铺。
/// 输出 dims³ 纹素，行主序 x 最快，Rgba16Unorm 8B/纹素（bytes_per_row = dims×8 天然 256 对齐）。
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

  let win_lo = origin_cell * cs;
  let win_hi = win_lo + IVec3::splat(dim * cs);

  let csz = IVec3::splat(gate_voxel::CHUNK_SIZE);
  let c_lo = win_lo.div_euclid(csz);
  let c_hi = (win_hi - IVec3::ONE).div_euclid(csz);

  if force_full {
    lf.cache.clear();
  }
  let moved = !lf.valid || origin_cell != lf.origin_cell;

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

  let grid = volumes.main();
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
          lf.cache.insert(coord, build_light_chunk(tree));
          tallied += 1;
        }
      }
    }
  }

  lf.cache.retain(|c, _| c.0.cmpge(c_lo).all() && c.0.cmple(c_hi).all());

  // 纹素下标 = 世界锚定槽位 `((wc mod dim) + dim) mod dim`，必须与 WGSL `light_field_uv` 一致。
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
          // Rgba16Unorm：`.rgb` 恒 0（零初始化）；`.a` = AO fill（0..255 → ×257）。
          let ob = ti * 8;
          data[ob + 6..ob + 8].copy_from_slice(&((t.fill[si] as u32 * 257) as u16).to_le_bytes());
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

fn u8_of_u32(w: &[u32]) -> &[u8] {
  unsafe { std::slice::from_raw_parts(w.as_ptr() as *const u8, w.len() * 4) }
}

/// GridDesc 数组 → u8 字节视图（`#[repr(C)]` + 144B/entry，可直接 cast）
fn u8_of_grid_descs(descs: &[GridDesc]) -> &[u8] {
  unsafe { std::slice::from_raw_parts(descs.as_ptr() as *const u8, std::mem::size_of_val(descs)) }
}

/// GPU buffer 扩容尺寸策略（纯函数）：need ≥ 扩容阈值 → 32MB 对齐；否则 2× 增长（下限 64KB）。
fn grow_size(cap: u64, need: u64) -> u64 {
  let big = crate::brickmap::consts::BUFFER_GROW_BIG;
  let reserve = crate::brickmap::consts::BUFFER_GROW_RESERVE;
  if need >= big { need.div_ceil(reserve) * reserve } else { need.max(cap * 2).max(65536) }
}

/// 保证 buffer 能容纳 `bytes`，扩容时保留旧内容。
/// `prefix_valid=true` 时旧前缀走 GPU-GPU copy、仅 write 新增尾部；false 时整份一次 DMA。
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
    // COPY_SRC：本 buffer 扩容时作为前缀拷贝的源（必需）。
    usage: BufferUsages::COPY_DST | BufferUsages::COPY_SRC | BufferUsages::STORAGE,
    mapped_at_creation: false,
  });
  if prefix_valid && cap > 0 {
    // COPY_BUFFER_ALIGNMENT=4；cap 恒为 words×4 或初始 4B，天然对齐。
    let mut enc = device
      .create_command_encoder(&CommandEncoderDescriptor { label: Some("gate_grow_prefix_copy") });
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

/// 增量路径：保证 buffer 容量 ≥ `need_bytes`，扩容时新 buffer 前缀走 GPU-GPU 拷贝。
/// 新增尾部 [cap..need) 由调用方随后逐块 `write_buffer` 脏块覆盖。
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
    // COPY_SRC：本 buffer 扩容时作为前缀拷贝的源（必需）。
    usage: BufferUsages::COPY_DST | BufferUsages::COPY_SRC | BufferUsages::STORAGE,
    mapped_at_creation: false,
  });
  if cap > 0 {
    // COPY_BUFFER_ALIGNMENT=4；cap 恒为 words×4 或初始 4B，天然对齐。
    let mut enc = device
      .create_command_encoder(&CommandEncoderDescriptor { label: Some("gate_grow_prefix_copy") });
    enc.copy_buffer_to_buffer(cur, 0, &new_buf, 0, cap);
    queue.submit([enc.finish()]);
  }
  *cur = new_buf;
}

/// 光照场整幅重铺：dims³ 纹素（Rgba16Unorm 8B/纹素）；纹理尺寸不符时重建，尺寸相同则只 write_texture。
fn upload_light_field(
  device: &RenderDevice,
  queue: &RenderQueue,
  gpu: &mut GpuBrickMap,
  data: &[u8],
) {
  let dim = LIGHT_FIELD_DIM;
  let want = Extent3d { width: dim, height: dim, depth_or_array_layers: dim };
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

  // bytes_per_row = dim×8 = 256，满足 wgpu 256 对齐，无需补行。
  debug_assert_eq!(data.len(), (dim as usize).pow(3) * 8);
  queue.write_texture(
    TexelCopyTextureInfo {
      texture: &gpu.light_tex,
      mip_level: 0,
      origin: Origin3d::ZERO,
      aspect: TextureAspect::All,
    },
    data,
    TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(dim * 8), rows_per_image: Some(dim) },
    want,
  );
}

#[allow(clippy::too_many_arguments)]
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
  if let Some(u) = light.as_ref() {
    upload_light_field(&device, &queue, &mut gpu, &u.data);
    commands.remove_resource::<LightFieldUpdate>();
  }
  let Some(snap) = snapshot else { return };
  let t0 = std::time::Instant::now();

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

  // 方向可达掩码 LUT → b_leaves，与 volume 无关。
  {
    let lut_need = (MARCH_MASK_WORDS * 4) as u64;
    if gpu.leaves.size() < lut_need {
      let lut = march_mask_lut_words();
      write(&device, &queue, &mut gpu.leaves, "gate_leaves", u8_of_u32(&lut));
    }
  }

  let (struct_tx_bytes, palette_tx_bytes, grid_descs_tx_bytes);

  if is_full {
    let struct_bytes = u8_of_u32(&snap.volumes.b_struct);
    let palette_bytes = u8_of_u32(&snap.volumes.b_palette);
    let grid_descs_bytes = u8_of_grid_descs(&snap.volumes.grid_descs);
    struct_tx_bytes = struct_bytes.len();
    palette_tx_bytes = palette_bytes.len();
    grid_descs_tx_bytes = grid_descs_bytes.len();
    write(&device, &queue, &mut gpu.struct_buf, "gate_struct", struct_bytes);
    write(&device, &queue, &mut gpu.palette, "gate_palette", palette_bytes);
    write(&device, &queue, &mut gpu.grid_descs_buf, "gate_grid_descs", grid_descs_bytes);
  } else {
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

  write(&device, &queue, &mut gpu.state, "gate_state", &snap.state_bytes);

  {
    let placeholder = vec![0u8; comp_bytes.max(4)];

    ensure_with_copy(&device, &queue, &mut gpu.comp, "gate_comp", &placeholder, true);
  }

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

  gpu.main_window_origin =
    IVec3::new(main_desc.index_origin_x, main_desc.index_origin_y, main_desc.index_origin_z);
  gpu.main_window_dims =
    UVec3::new(main_desc.index_dims_x, main_desc.index_dims_y, main_desc.index_dims_z);

  revision.0 = revision.0.wrapping_add(1);

  let elapsed = t0.elapsed();
  let cpu_ms = elapsed.as_secs_f32() * 1000.0;
  let tx_bytes_total =
    (struct_tx_bytes + palette_tx_bytes + grid_descs_tx_bytes + snap.state_bytes.len()) as f64;
  let mb = tx_bytes_total / (1 << 20) as f64;

  let chunks_show = if is_full {
    snap.volumes.grid_descs.iter().map(|g| g.chunk_count as usize).sum::<usize>()
  } else {
    snap.volumes.dirty_chunks
  };
  debug!(target: "gate",
    "UPLOAD[{}]: bytes={:.2}MB (s {}KB,pal {}KB,gd {}KB,state 4KB), chunks={}, comp={}KB, elapsed={:?}",
    snap.volumes.mode_tag, mb,
    struct_tx_bytes / 1024, palette_tx_bytes / 1024,
    grid_descs_tx_bytes / 1024, chunks_show, comp_bytes / 1024, elapsed,
  );

  static SAMPLE_GEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
  let generation = SAMPLE_GEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
  let sample = UploadCpuSample { cpu_ms, generation };
  commands.insert_resource(sample);
  if let Some(ch) = sample_channel
    && let Ok(mut g) = ch.0.lock()
  {
    *g = Some(sample);
  }

  let vram = gpu.struct_buf.size()
    + gpu.leaves.size()
    + gpu.palette.size()
    + gpu.comp.size()
    + gpu.state.size()
    + gpu.grid_descs_buf.size();
  if vram > crate::brickmap::consts::VRAM_WARN_BYTES {
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

  // 消费完本帧 snapshot 必须移除。
  commands.remove_resource::<UploadSnapshot>();
}

/// 统一体素渲染上传插件：主世界与物体同一路径。
/// 物体是 `Volumes.list[1..N]` 的普通 `VolumeGrid`，走相同的 dirty → VolumesBuilder → UploadSnapshot 路径。
pub struct VolumePlugin;
impl Plugin for VolumePlugin {
  fn build(&self, app: &mut App) {
    let ch = UploadCpuSampleChannel::default();
    app.insert_resource(ch.clone()).init_resource::<MainPending>().add_systems(Last, poll_pending);
    let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
      return;
    };
    render_app
      .insert_resource(ch)
      .init_resource::<BrickMapRevision>()
      .init_resource::<LightFieldCpu>()
      .insert_resource(BuilderMirror { pending_full: true, ..Default::default() })
      .add_systems(RenderStartup, init_empty_gpu)
      .add_systems(ExtractSchedule, extract)
      .add_systems(Render, prepare.in_set(RenderSystems::PrepareResources));
  }
}
