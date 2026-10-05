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
use super::residency::{Level, Residency, ResidencyPolicy, want_level};
use super::wire::{
  BrickMapGlobals, CHUNK_COMP_WORDS, GridDesc, MARCH_MASK_WORDS, MaterialAsset, TREE_BASE,
  march_mask_lut_words,
};
use crate::pbr_texture::{METAL_DEMO_ID, PbrTextureSet, build_material_asset_table};
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
  pub interior_only_edit: bool,
  pub edit_in_flight: bool,
  pub residency_budget_bytes: usize,
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

#[derive(Resource, Default, Clone, Copy, Debug, PartialEq, Eq)]
pub struct BrickMapRevision(pub u64);

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DirtyBox {
  pub lo: IVec3,
  pub hi: IVec3,
}

impl DirtyBox {
  pub fn overlaps(&self, other: &Self) -> bool {
    self.lo.cmple(other.hi).all() && other.lo.cmple(self.hi).all()
  }

  pub fn union_with(&mut self, other: &Self) {
    self.lo = self.lo.min(other.lo);
    self.hi = self.hi.max(other.hi);
  }
}

#[derive(Resource, Default, Clone, Debug)]
pub struct BrickMapDirty {
  pub full: bool,
  pub palette_changed: bool,
  pub boxes: Vec<DirtyBox>,
}

pub fn world_dirty_box(t: gate_voxel::VolumeTransform, lo: IVec3, hi: IVec3) -> DirtyBox {
  let s = if t.scale.is_finite() && t.scale > 0.0 { t.scale } else { 1.0 };
  let mut mn = glam::Vec3::splat(f32::MAX);
  let mut mx = glam::Vec3::splat(f32::MIN);
  for &x in &[lo.x, hi.x] {
    for &y in &[lo.y, hi.y] {
      for &z in &[lo.z, hi.z] {
        let w = t.pos + t.rot * (glam::Vec3::new(x as f32, y as f32, z as f32) * s);
        mn = mn.min(w);
        mx = mx.max(w);
      }
    }
  }
  DirtyBox {
    lo: IVec3::new(mn.x.floor() as i32, mn.y.floor() as i32, mn.z.floor() as i32),
    hi: IVec3::new(mx.x.ceil() as i32, mx.y.ceil() as i32, mx.z.ceil() as i32),
  }
}

#[derive(Resource, Default)]
pub struct MainPending {
  pub force_full: bool,
  pub data_chunks: Vec<(usize, gate_voxel::ChunkCoord, gate_voxel::TreeDirty)>,
  pub comp_chunks: Vec<(usize, gate_voxel::ChunkCoord)>,
  pub data_aabbs: Vec<(usize, gate_voxel::ChunkCoord, IVec3, IVec3)>,
}

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
      let dirty = grid.chunk_mut(c).map(|t| t.take_dirty()).unwrap_or_default();
      pending.data_chunks.push((vol_idx, c, dirty));

      if let Some((lo, hi)) = grid.take_edit_aabb(c) {
        pending.data_aabbs.push((vol_idx, c, lo, hi));
      }
    }
    for c in grid.dirty.drain_comp_budget(comp_n) {
      pending.comp_chunks.push((vol_idx, c));
    }
  }
}

fn tick_voxel_dump_request(mut req: ResMut<VoxelDumpRequest>) {
  req.pending = req.pending.saturating_sub(1);
}

#[derive(
  Resource, Default, Clone, Copy, Debug, bevy::render::extract_resource::ExtractResource,
)]
#[extract_app(bevy::render::RenderApp)]
pub struct VoxelDumpRequest {
  pub pending: u8,
}

impl VoxelDumpRequest {
  const ARMED_FRAMES: u8 = 2;

  pub fn arm(&mut self) {
    self.pending = Self::ARMED_FRAMES;
  }
}

#[derive(Resource, Default)]
pub struct BuilderMirror {
  pub builder: Option<VolumesBuilder>,
  pub pending_full: bool,
  pub pending_data_chunks: Vec<(usize, gate_voxel::ChunkCoord, gate_voxel::TreeDirty)>,
  pub pending_data_aabbs: Vec<(usize, gate_voxel::ChunkCoord, IVec3, IVec3)>,
  pub palette_versions: Vec<u64>,
  pub palette_content_versions: Vec<u64>,
  pub pending_comp_chunks: Vec<(usize, gate_voxel::ChunkCoord)>,
}

#[derive(Resource, Clone)]
pub struct UploadSnapshot {
  pub volumes: VolumesSnapshot,
  pub state_bytes: Vec<u8>,
  pub comp_chunks: usize,
}

#[derive(Resource)]
pub struct ResidencyState {
  pub residency: Residency,
  pub policy: ResidencyPolicy,
  pub edited: Vec<gate_voxel::ChunkCoord>,
}

impl Default for ResidencyState {
  fn default() -> Self {
    Self { residency: Residency::new(), policy: ResidencyPolicy::DEFAULT, edited: Vec::new() }
  }
}

#[derive(Resource, Clone, Copy, Debug, Default)]
pub struct UploadCpuSample {
  pub cpu_ms: f32,
  pub generation: u64,
}

#[derive(Resource, Clone, Debug, Default)]
pub struct UploadCpuSampleChannel(pub std::sync::Arc<std::sync::Mutex<Option<UploadCpuSample>>>);

#[derive(Resource)]
pub struct GpuBrickMap {
  pub struct_buf: Buffer,
  pub leaves: Buffer,
  pub palette: Buffer,
  pub comp: Buffer,
  pub state: Buffer,
  pub lod_diag: Buffer,
  pub lod_req: Buffer,
  pub occ: Buffer,
  pub occ_ready: bool,
  pub grid_descs_buf: Buffer,
  pub grid_descs_count: u32,
  pub globals: UniformBuffer<BrickMapGlobals>,
  pub main_window_origin: IVec3,
  pub main_window_dims: UVec3,
  pub volume_windows: Vec<IVec3>,
  pub light_sampler: Sampler,
  pub blit_sampler: Sampler,
  pub material_assets: Buffer,
  pub material_assets_uploaded: bool,
  pub pbr_albedo_rough_tex: Texture,
  pub pbr_albedo_rough_view: TextureView,
  pub pbr_metal_tex: Texture,
  pub pbr_metal_view: TextureView,
  pub pbr_sampler: Sampler,
}

fn init_empty_gpu(device: Res<RenderDevice>, mut commands: Commands) {
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
    label: Some("gate_light_sampler"),
    address_mode_u: AddressMode::ClampToEdge,
    address_mode_v: AddressMode::ClampToEdge,
    address_mode_w: AddressMode::ClampToEdge,
    mag_filter: FilterMode::Linear,
    min_filter: FilterMode::Linear,
    mipmap_filter: MipmapFilterMode::Nearest,
    ..Default::default()
  });

  let blit_sampler = device.create_sampler(&SamplerDescriptor {
    label: Some("gate_dda_blit_sampler"),
    mag_filter: FilterMode::Linear,
    min_filter: FilterMode::Linear,
    ..Default::default()
  });

  let asset_slots = crate::wesl_consts::material_consts().material_asset_slots;
  let material_assets = device.create_buffer(&BufferDescriptor {
    label: Some("gate_material_assets"),
    size: asset_slots as u64 * std::mem::size_of::<MaterialAsset>() as u64,
    usage: BufferUsages::COPY_DST | BufferUsages::COPY_SRC | BufferUsages::STORAGE,
    mapped_at_creation: false,
  });
  let make_pbr_placeholder = |label: &str, format: TextureFormat| -> (Texture, TextureView) {
    let tex = device.create_texture(&TextureDescriptor {
      label: Some(label),
      size: Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
      mip_level_count: 1,
      sample_count: 1,
      dimension: TextureDimension::D2,
      format,
      usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
      view_formats: &[],
    });
    let view = tex.create_view(&TextureViewDescriptor {
      label: Some(label),
      dimension: Some(TextureViewDimension::D2Array),
      ..Default::default()
    });
    (tex, view)
  };
  let (pbr_albedo_rough_tex, pbr_albedo_rough_view) =
    make_pbr_placeholder("gate_pbr_albedo_rough_placeholder", TextureFormat::Rgba8Unorm);
  let (pbr_metal_tex, pbr_metal_view) =
    make_pbr_placeholder("gate_pbr_metal_placeholder", TextureFormat::R8Unorm);

  let pbr_sampler = crate::pbr_texture::create_pbr_sampler(&device);
  info!(
    target: "gate",
    "PBR 采样器（BG1 binding 8）: Repeat×3 / Linear(mag,min,mipmap) / anisotropy_clamp = {} \
     / lod_max_clamp = {}；mip 链 = {} 层（GPU_TEX_SIZE = {} → 1×1，CPU 盒式平均）；\
     与 light_samp（ClampToEdge / mipmap Nearest / 无 mip）分属两个 sampler",
    crate::pbr_texture::PBR_ANISOTROPY_CLAMP,
    (crate::pbr_texture::PBR_MIP_LEVELS_MAX - 1) as f32,
    crate::pbr_texture::PBR_MIP_LEVELS_MAX,
    crate::pbr_texture::GPU_TEX_SIZE,
  );

  let lod_diag = device.create_buffer(&BufferDescriptor {
    label: Some("gate_lod_diag"),
    size: (crate::brickmap::consts::LOD_DIAG_WORDS * 4) as u64,
    usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
    mapped_at_creation: false,
  });

  let lod_req = device.create_buffer(&BufferDescriptor {
    label: Some("gate_lod_req"),
    size: (crate::brickmap::consts::LOD_REQ_WORDS * 4) as u64,
    usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
    mapped_at_creation: false,
  });

  let occ = device.create_buffer(&BufferDescriptor {
    label: Some("gate_occ"),
    size: (crate::brickmap::consts::VOLUMES * crate::brickmap::builder::OCC_WORDS * 4) as u64,
    usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    mapped_at_creation: false,
  });

  commands.insert_resource(GpuBrickMap {
    struct_buf: make("gate_struct"),
    leaves: make("gate_leaves"),
    palette: make("gate_palette"),
    comp: make("gate_comp"),
    state: make("gate_state"),
    lod_diag,
    lod_req,
    occ,
    occ_ready: false,
    grid_descs_buf: make("gate_grid_descs"),
    grid_descs_count: 0,
    globals,
    main_window_origin: IVec3::ZERO,
    main_window_dims: UVec3::ZERO,
    volume_windows: Vec::new(),
    light_sampler,
    blit_sampler,
    material_assets,
    material_assets_uploaded: false,
    pbr_albedo_rough_tex,
    pbr_albedo_rough_view,
    pbr_metal_tex,
    pbr_metal_view,
    pbr_sampler,
  });
}

#[allow(clippy::too_many_arguments)]
fn extract(
  mut commands: Commands,
  scene: Option<Extract<Res<VoxelScene>>>,
  budget: Option<Extract<Res<UploadBudget>>>,
  main_pending: Option<Extract<Res<MainPending>>>,
  mut mirror: ResMut<BuilderMirror>,
  mut resid: ResMut<ResidencyState>,
  mut diag: Local<(f64, u32)>,
  mut split: Local<Option<crate::profiler::SplitDiag>>,
) {
  let _t = crate::profiler::SysTimer::new("EXTRACT 快照", &mut diag);
  let sd = split.get_or_insert_with(|| {
    crate::profiler::SplitDiag::new(&[
      "判脏", "sync", "sync_win", "update", "palette", "snapshot", "尾",
    ])
  });
  sd.start();
  let (Some(scene), Some(budget), Some(main_pending)) = (scene, budget, main_pending) else {
    return;
  };
  if main_pending.force_full {
    mirror.pending_full = true;
  }
  mirror.pending_data_chunks.extend(main_pending.data_chunks.iter().cloned());
  mirror.pending_data_aabbs.extend(main_pending.data_aabbs.iter().copied());
  mirror.pending_comp_chunks.extend(main_pending.comp_chunks.iter().copied());

  let first = mirror.builder.is_none();
  let pending_full = std::mem::take(&mut mirror.pending_full);
  let mut pending_data: Vec<(usize, gate_voxel::ChunkCoord, gate_voxel::TreeDirty)> =
    std::mem::take(&mut mirror.pending_data_chunks);
  let pending_aabbs: Vec<(usize, gate_voxel::ChunkCoord, IVec3, IVec3)> =
    std::mem::take(&mut mirror.pending_data_aabbs);
  let _pending_comp: Vec<(usize, gate_voxel::ChunkCoord)> =
    std::mem::take(&mut mirror.pending_comp_chunks);
  let need_full = first || pending_full || !budget.incremental;

  resid.edited.clear();
  resid.edited.extend(pending_data.iter().filter(|(v, ..)| *v == 0).map(|(_, c, _)| *c));

  let palette_dirty = scene
    .volumes
    .list
    .iter()
    .enumerate()
    .any(|(i, g)| mirror.palette_versions.get(i).copied() != Some(g.palette().version()));
  let palette_content_changed = scene.volumes.list.iter().enumerate().any(|(i, g)| {
    mirror.palette_content_versions.get(i).copied() != Some(g.palette().content_version())
  });
  let dirty_any =
    need_full || !pending_data.is_empty() || !_pending_comp.is_empty() || palette_dirty;

  let mut dirty_aabb =
    BrickMapDirty { palette_changed: palette_content_changed, ..Default::default() };
  if dirty_any {
    if need_full {
      dirty_aabb.full = true;
    } else if scene.interior_only_edit {
    } else {
      let mut acc: Vec<(usize, IVec3, IVec3)> = Vec::new();
      for (vol_idx, c) in
        pending_data.iter().map(|(v, c, _)| (v, c)).chain(_pending_comp.iter().map(|(v, c)| (v, c)))
      {
        let (cl, ch) = match pending_aabbs.iter().find(|(v, cc, ..)| v == vol_idx && cc == c) {
          Some((_, _, alo, ahi)) => (*alo, *ahi),
          None => {
            let cl = c.0 * gate_voxel::CHUNK_SIZE;
            (cl, cl + IVec3::splat(gate_voxel::CHUNK_SIZE))
          }
        };
        match acc.iter_mut().find(|(v, ..)| *v == *vol_idx) {
          Some((_, lo, hi)) => {
            *lo = (*lo).min(cl);
            *hi = (*hi).max(ch);
          }
          None => acc.push((*vol_idx, cl, ch)),
        }
      }
      dirty_aabb.boxes = acc
        .into_iter()
        .filter_map(|(vol_idx, lo, hi)| {
          scene.volumes.list.get(vol_idx).map(|g| world_dirty_box(g.transform(), lo, hi))
        })
        .collect();
    }
  }
  commands.insert_resource(dirty_aabb);

  let volumes_ref = &scene.volumes;

  let window_moved = mirror.builder.as_ref().is_some_and(|b| {
    volumes_ref
      .all()
      .iter()
      .enumerate()
      .any(|(i, g)| g.stream_window().is_some_and(|(o, d)| b.window_of(i) != Some((o, d))))
  });
  let builder_dirty = mirror.builder.as_ref().is_some_and(VolumesBuilder::has_dirty);

  if !dirty_any && !window_moved && !builder_dirty {
    sd.mark(0);
    sd.frame_end("EXTRACT");
    return;
  }
  sd.mark(0);
  let builder = mirror
    .builder
    .get_or_insert_with(|| VolumesBuilder::new_unbuilt(volumes_ref, scene.residency_budget_bytes));
  if need_full {
    *builder = VolumesBuilder::build_full(volumes_ref, scene.residency_budget_bytes);
    pending_data.clear();
  } else {
    let quiet = !scene.edit_in_flight
      && scene
        .volumes
        .list
        .iter()
        .all(|g| g.dirty.data_dirty_count() == 0 && g.dirty.comp_dirty_count() == 0);
    builder.sync(volumes_ref);
    sd.mark(1);
    builder.sync_windows(volumes_ref);
    sd.mark(2);
    for (vol_idx, c, dirty) in pending_data.drain(..) {
      builder.update_chunk(volumes_ref, vol_idx, c, &dirty, quiet);
    }
    sd.mark(3);
  }

  builder.sync_palettes(volumes_ref);
  sd.mark(4);
  let snapshot = builder.snapshot();
  sd.mark(5);

  mirror.palette_versions = scene.volumes.list.iter().map(|g| g.palette().version()).collect();
  mirror.palette_content_versions =
    scene.volumes.list.iter().map(|g| g.palette().content_version()).collect();

  let state_bytes = volumes_ref.main().state_table_bytes().to_vec();
  let comp_chunks = volumes_ref.main().comp_layer().len();
  commands.insert_resource(UploadSnapshot { volumes: snapshot, state_bytes, comp_chunks });
  sd.mark(6);
  sd.frame_end("EXTRACT");
}

fn u8_of_u32(w: &[u32]) -> &[u8] {
  unsafe { std::slice::from_raw_parts(w.as_ptr() as *const u8, w.len() * 4) }
}

fn u8_of_grid_descs(descs: &[GridDesc]) -> &[u8] {
  unsafe { std::slice::from_raw_parts(descs.as_ptr() as *const u8, std::mem::size_of_val(descs)) }
}

fn u8_of_material_assets(assets: &[MaterialAsset]) -> &[u8] {
  unsafe { std::slice::from_raw_parts(assets.as_ptr() as *const u8, std::mem::size_of_val(assets)) }
}

fn grow_size(cap: u64, need: u64, limit: u64) -> u64 {
  let big = crate::brickmap::consts::BUFFER_GROW_BIG;
  let reserve = crate::brickmap::consts::BUFFER_GROW_RESERVE;
  let sized =
    if need >= big { need.div_ceil(reserve) * reserve } else { need.max(cap * 2).max(65536) };
  sized.min(limit)
}

fn main_region_byte_cap(limit: u64, unbudgeted_bytes: u64) -> usize {
  limit
    .saturating_sub(unbudgeted_bytes)
    .saturating_sub(crate::brickmap::consts::BUFFER_GROW_RESERVE)
    .min(usize::MAX as u64) as usize
}

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
  let limit = device.limits().max_storage_buffer_binding_size;
  if need > limit {
    bevy::log::warn_once!(
      "显存 {label} 需要 {need} B，超过设备单次绑定上限 {limit} B ⇒ 调用方必须先缩小常驻池"
    );
  }
  let new_size = grow_size(cap, need, limit);
  let fits = new_size.min(need) as usize;
  if new_size <= cap {
    return;
  }
  let new_buf = device.create_buffer(&BufferDescriptor {
    label: Some(label),
    size: new_size,
    usage: BufferUsages::COPY_DST | BufferUsages::COPY_SRC | BufferUsages::STORAGE,
    mapped_at_creation: false,
  });
  if prefix_valid && cap > 0 {
    let mut enc = device
      .create_command_encoder(&CommandEncoderDescriptor { label: Some("gate_grow_prefix_copy") });
    enc.copy_buffer_to_buffer(cur, 0, &new_buf, 0, cap);
    queue.submit([enc.finish()]);
    if (cap as usize) < fits {
      queue.write_buffer(&new_buf, cap, &bytes[cap as usize..fits]);
    }
  } else if !bytes.is_empty() {
    queue.write_buffer(&new_buf, 0, &bytes[..fits]);
  }
  *cur = new_buf;
}

fn write(device: &RenderDevice, queue: &RenderQueue, cur: &mut Buffer, label: &str, bytes: &[u8]) {
  ensure_with_copy(device, queue, cur, label, bytes, false);
  if !bytes.is_empty() {
    let fits = (cur.size() as usize).min(bytes.len());
    queue.write_buffer(cur, 0, &bytes[..fits]);
  }
}

fn write_blob(queue: &RenderQueue, buf: &Buffer, off: u64, payload: &[u8]) -> bool {
  if off.saturating_add(payload.len() as u64) > buf.size() {
    return false;
  }
  queue.write_buffer(buf, off, payload);
  true
}

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
  let limit = device.limits().max_storage_buffer_binding_size;
  if need_bytes > limit {
    bevy::log::warn_once!(
      "显存 {label} 需要 {need_bytes} B，超过设备单次绑定上限 {limit} B ⇒ 调用方必须先缩小常驻池"
    );
  }
  let new_size = grow_size(cap, need_bytes, limit);
  if new_size <= cap {
    return;
  }
  let new_buf = device.create_buffer(&BufferDescriptor {
    label: Some(label),
    size: new_size,
    usage: BufferUsages::COPY_DST | BufferUsages::COPY_SRC | BufferUsages::STORAGE,
    mapped_at_creation: false,
  });
  if cap > 0 {
    let mut enc = device
      .create_command_encoder(&CommandEncoderDescriptor { label: Some("gate_grow_prefix_copy") });
    enc.copy_buffer_to_buffer(cur, 0, &new_buf, 0, cap);
    queue.submit([enc.finish()]);
  }
  debug!("显存 {label} 扩容 {cap} → {new_size} B（含前缀拷贝）");
  *cur = new_buf;
}

fn upload_material_assets(queue: &RenderQueue, set: Option<&PbrTextureSet>, gpu: &mut GpuBrickMap) {
  if gpu.material_assets_uploaded {
    return;
  }
  let Some(set) = set else {
    return;
  };
  let table = build_material_asset_table(set);
  queue.write_buffer(&gpu.material_assets, 0, u8_of_material_assets(&table));
  gpu.material_assets_uploaded = true;

  let layers = set.layers();
  info!(
    target: "gate",
    "材质资产表: 槽 0..{layers}: albedo_slot = roughmetal_slot = i、emissive/transmission/height \
     = MATERIAL_SLOT_NONE；其余槽位五个 *_slot 全 NONE（标量回退 = 中性灰 albedo(sRGB 128) + roughness 0.5 \
     + metallic 0 + specular 1.0(不调制电介质 F0) + emissive/transmission 0 + IOR 1.50）",
  );
  match set.slot_of(METAL_DEMO_ID) {
    Some(i) => info!(
      target: "gate",
      "材质资产表: 槽 {i}（{METAL_DEMO_ID}）metallic = 255（临时 demo 默认值）",
    ),
    None => warn!(
      target: "gate",
      "材质资产表: 贴图集里没有 `{METAL_DEMO_ID}` ⇒ 表里无金属槽；确认 assets/textures/pbr/{METAL_DEMO_ID}/ 存在",
    ),
  }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare(
  mut commands: Commands,
  snapshot: Option<ResMut<UploadSnapshot>>,
  pbr_set: Option<Res<PbrTextureSet>>,
  mut gpu: ResMut<GpuBrickMap>,
  device: Res<RenderDevice>,
  queue: Res<RenderQueue>,
  sample_channel: Option<Res<UploadCpuSampleChannel>>,
  mut revision: ResMut<BrickMapRevision>,
) {
  upload_material_assets(&queue, pbr_set.as_deref(), &mut gpu);
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
        "GPU storage binding < 1GB ⇒ 退化到全量上传（单 tile 170MB 跨帧）"
      );
    }
  }

      static RT_PROBED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
  if !RT_PROBED.swap(true, std::sync::atomic::Ordering::Relaxed) {
    use bevy::render::render_resource::WgpuFeatures;
    let f = device.features();
    let rt = f.contains(WgpuFeatures::EXPERIMENTAL_RAY_QUERY);
    let l = device.limits();
    info!(target: "gate",
      "PROBE: ray query = {} → {}；BLAS 图元上限 {} / 几何数 {}、TLAS 实例上限 {}",
      if rt { "有" } else { "无" },
      if rt { "硬件光追路径可用（软件 DDA 仍是回退与对照）" } else { "只有软件 DDA 路径" },
      l.max_blas_primitive_count,
      l.max_blas_geometry_count,
      l.max_tlas_instance_count,
    );
    if rt && l.max_tlas_instance_count == 0 {
      warn!(target: "gate",
        "ray query 已开但 TLAS 实例上限为 0 ⇒ BLAS/TLAS 不可用（驱动未给 RT 限额）"
      );
    }
  }

  let is_full = matches!(snap.volumes.mode_tag, "full" | "fallback_full");
    if let Some(occ) = snap.volumes.occ_all.as_ref() {
    queue.write_buffer(&gpu.occ, 0, u8_of_u32(occ));
    gpu.occ_ready = true;
  } else if !gpu.occ_ready {
    let n = crate::brickmap::consts::VOLUMES * crate::brickmap::builder::OCC_WORDS;
    let all = vec![u32::MAX; n];
    queue.write_buffer(&gpu.occ, 0, u8_of_u32(&all));
  }
  let comp_bytes = snap.comp_chunks * CHUNK_COMP_WORDS * 4;

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
    queue.write_buffer(&gpu.grid_descs_buf, 0, grid_descs_bytes);

    let mut s_tx = 0usize;
    let mut s_skipped = 0usize;
    for (off, payload) in &snap.volumes.struct_blobs {
      if write_blob(&queue, &gpu.struct_buf, *off as u64, payload) {
        s_tx += payload.len();
      } else {
        s_skipped += payload.len();
      }
    }
    if s_skipped > 0 {
      bevy::log::warn_once!(
        "增量上传 gate_struct：{s_skipped} B 脏块越过 buffer 末尾（容量 {}B）⇒ 本次跳过（该块在 GPU 上\
         保持旧内容）；树区超过设备单次绑定上限，先缩小常驻池（见 `plan_residency` 的绑定上限）",
        gpu.struct_buf.size()
      );
    }
    let mut p_tx = 0usize;
    let mut p_skipped = 0usize;
    for (off, payload) in &snap.volumes.palette_blobs {
      if write_blob(&queue, &gpu.palette, *off as u64, payload) {
        p_tx += payload.len();
      } else {
        p_skipped += payload.len();
      }
    }
    if p_skipped > 0 {
      bevy::log::warn_once!(
        "增量上传 gate_palette：{p_skipped} B 脏块越过 buffer 末尾（容量 {}B）⇒ 本次跳过",
        gpu.palette.size()
      );
    }
    struct_tx_bytes = s_tx;
    palette_tx_bytes = p_tx;
    grid_descs_tx_bytes = grid_descs_bytes.len();
  }

  write(&device, &queue, &mut gpu.state, "gate_state", &snap.state_bytes);

  {
        if gpu.comp.size() < comp_bytes.max(4) as u64 {
      let placeholder = vec![0u8; comp_bytes.max(4)];
      ensure_with_copy(&device, &queue, &mut gpu.comp, "gate_comp", &placeholder, true);
    }
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
  gpu.volume_windows = snap
    .volumes
    .grid_descs
    .iter()
    .map(|d| IVec3::new(d.index_origin_x, d.index_origin_y, d.index_origin_z))
    .collect();

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
    bevy::log::warn_once!("GPU VRAM {vram}B 超旧预算线（仅提示）");
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

  commands.remove_resource::<UploadSnapshot>();
}

const DUMP_MAGIC: u32 = 0x4458_4F56;
const DUMP_LAYOUT_VERSION: u32 = 2;
const DUMP_HEADER_WORDS: usize = 16;
const DUMP_VOLUME_WORDS: usize = 16;

fn dump_voxel_buffers(
  request: Option<Res<VoxelDumpRequest>>,
  mirror: Option<Res<BuilderMirror>>,
  gpu: Option<Res<GpuBrickMap>>,
  device: Res<RenderDevice>,
  queue: Res<RenderQueue>,
) {
  if request.is_none_or(|r| r.pending == 0) {
    return;
  }
  let (Some(mirror), Some(gpu)) = (mirror, gpu) else {
    warn!("数据转储：render 侧资源未就绪 → 忽略");
    return;
  };
  let Some(builder) = mirror.builder.as_ref() else {
    warn!("数据转储：CPU builder 尚未建立（还没有过一次上传）→ 忽略");
    return;
  };
  let buffers = builder.volume_buffers();
  let n = buffers.len();
  if n == 0 {
    warn!("数据转储：没有 volume → 忽略");
    return;
  }

  let layout_order: Vec<usize> = (1..n).chain(std::iter::once(0)).collect();
  let mut tree_base = vec![0u32; n];
  let mut palette_base = vec![0u32; n];
  let (mut struct_words, mut palette_words) = (0usize, 0usize);
  for &i in &layout_order {
    tree_base[i] = struct_words as u32;
    palette_base[i] = palette_words as u32;
    struct_words += buffers[i].b_struct.len();
    palette_words += buffers[i].b_palette.len();
  }
  let leaves_words = MARCH_MASK_WORDS;

  let mut head: Vec<u32> = Vec::with_capacity(DUMP_HEADER_WORDS + DUMP_VOLUME_WORDS * n);
  {
    let m = &buffers[0].globals;
    head.extend([
      DUMP_MAGIC,
      DUMP_LAYOUT_VERSION,
      n as u32,
      struct_words as u32,
      palette_words as u32,
      leaves_words as u32,
      m.index_origin_x as u32,
      m.index_origin_y as u32,
      m.index_origin_z as u32,
      m.index_dims_x,
      m.index_dims_y,
      m.index_dims_z,
      m.tile_count,
      m.node_words,
      m.node_free_words,
      m.rejected_tiles,
    ]);
  }
  for i in 0..n {
    let g = &buffers[i].globals;
    head.extend([
      tree_base[i],
      palette_base[i],
      buffers[i].b_struct.len() as u32,
      buffers[i].b_palette.len() as u32,
      g.index_origin_x as u32,
      g.index_origin_y as u32,
      g.index_origin_z as u32,
      g.index_dims_x,
      g.index_dims_y,
      g.index_dims_z,
      g.tile_count,
      g.node_words,
      g.node_free_words,
      g.rejected_tiles,
      0,
      0,
    ]);
  }

  let lut = march_mask_lut_words();
  let head_bytes = head.len() * 4;
  let struct_bytes = struct_words * 4;
  let palette_bytes = palette_words * 4;
  let leaves_bytes = leaves_words * 4;
  let total = head_bytes + struct_bytes + palette_bytes + leaves_bytes;
  let mut cpu: Vec<u8> = Vec::with_capacity(total);
  cpu.extend_from_slice(u8_of_u32(&head));
  for &i in &layout_order {
    cpu.extend_from_slice(u8_of_u32(&buffers[i].b_struct));
  }
  for &i in &layout_order {
    cpu.extend_from_slice(u8_of_u32(&buffers[i].b_palette));
  }
  cpu.extend_from_slice(u8_of_u32(&lut));

  let gpu_sizes = [
    ("struct", gpu.struct_buf.size(), struct_bytes as u64),
    ("palette", gpu.palette.size(), palette_bytes as u64),
    ("leaves", gpu.leaves.size(), leaves_bytes as u64),
  ];
  if let Some((label, have, need)) = gpu_sizes.iter().find(|(_, have, need)| have < need) {
    warn!(
      "数据转储：GPU {label} buffer 只有 {have}B < 布局 {need}B → 本次放弃（先看上传是否掉队）"
    );
    return;
  }
  let payload_bytes = total - head_bytes;
  let staging = device.create_buffer(&BufferDescriptor {
    label: Some("gate_voxel_dump_staging"),
    size: payload_bytes as u64,
    usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
    mapped_at_creation: false,
  });
  let mut enc = device
    .create_command_encoder(&CommandEncoderDescriptor { label: Some("gate_voxel_dump_readback") });
  enc.copy_buffer_to_buffer(&gpu.struct_buf, 0, &staging, 0, struct_bytes as u64);
  enc.copy_buffer_to_buffer(&gpu.palette, 0, &staging, struct_bytes as u64, palette_bytes as u64);
  enc.copy_buffer_to_buffer(
    &gpu.leaves,
    0,
    &staging,
    (struct_bytes + palette_bytes) as u64,
    leaves_bytes as u64,
  );
  queue.submit([enc.finish()]);

  let slice = staging.slice(..);
  let (tx, rx) = std::sync::mpsc::channel();
  device.map_buffer(&slice, MapMode::Read, move |r| {
    let _ = tx.send(r);
  });
  if let Err(e) = device.poll(PollType::wait_indefinitely()) {
    warn!("数据转储：等待 readback 失败 {e} → 本次放弃");
    staging.unmap();
    return;
  }
  match rx.recv_timeout(std::time::Duration::from_secs(5)) {
    Ok(Ok(())) => {}
    Ok(Err(e)) => {
      warn!("数据转储：readback 映射失败 {e} → 本次放弃");
      staging.unmap();
      return;
    }
    Err(e) => {
      warn!("数据转储：readback 超时/断线 {e} → 本次放弃");
      staging.unmap();
      return;
    }
  }
  let mut gpu_bytes: Vec<u8> = Vec::with_capacity(total);
  gpu_bytes.extend_from_slice(&cpu[..head_bytes]);
  match slice.get_mapped_range() {
    Ok(view) => gpu_bytes.extend_from_slice(&view),
    Err(e) => {
      warn!("数据转储：读取映射区间失败 {e:?} → 本次放弃");
      staging.unmap();
      return;
    }
  }
  staging.unmap();

  let dir = crate::paths::logs_dir();
  if let Err(e) = std::fs::create_dir_all(&dir) {
    warn!("数据转储：建目录失败 {}：{e}", dir.display());
    return;
  }
  let cpu_path = dir.join("voxel_dump_cpu.bin");
  let gpu_path = dir.join("voxel_dump_gpu.bin");
  if let Err(e) = std::fs::write(&cpu_path, &cpu) {
    warn!("数据转储：写 {} 失败 {e}", cpu_path.display());
    return;
  }
  if let Err(e) = std::fs::write(&gpu_path, &gpu_bytes) {
    warn!("数据转储：写 {} 失败 {e}", gpu_path.display());
    return;
  }
  info!(
    "数据转储 → {} / {}（各 {}KB = 头 {}B + struct {}KB + palette {}KB + leaves {}KB；volume={n}）",
    cpu_path.display(),
    gpu_path.display(),
    total / 1024,
    head_bytes,
    struct_bytes / 1024,
    palette_bytes / 1024,
    leaves_bytes / 1024,
  );
  match cpu.iter().zip(gpu_bytes.iter()).position(|(a, b)| a != b) {
    None if cpu.len() == gpu_bytes.len() => info!("数据转储 CPU/GPU 逐字节一致"),
    Some(i) => {
      warn!("数据转储 CPU/GPU 首异字节 @{i}：cpu={:#04x} gpu={:#04x}", cpu[i], gpu_bytes[i])
    }
    None => warn!("数据转储 CPU/GPU 长度不等：{}B vs {}B", cpu.len(), gpu_bytes.len()),
  }
}

const GAP_NEAR_CHUNKS: i32 = 3;

const FAR_INSTALL_PER_FRAME: usize = 24;

const LEDGER_SWEEP_FRAMES: u64 = 240;

const GPU_CHUNK_BYTES_EST: usize = 64 * 1024;
const CPU_CHUNK_BYTES_EST: usize = 64 * 1024;
const MIN_POOL_CHUNKS: usize = 64;

const CPU_FAR_CHUNK_BYTES_EST: usize = 32 * 1024;

const MIN_POOL_CHUNKS_FAR: usize = 512;

pub fn pool_capacity_chunks_far(budget_bytes: usize) -> usize {
  let target = crate::brickmap::consts::FAR_RESIDENT_TARGET;
  if budget_bytes == 0 {
    target
  } else {
    (budget_bytes / 4 / CPU_FAR_CHUNK_BYTES_EST).max(MIN_POOL_CHUNKS_FAR).min(target)
  }
}

pub fn pool_capacity_chunks(budget_bytes: usize) -> usize {
  if budget_bytes == 0 {
    usize::MAX
  } else {
    (budget_bytes / CPU_CHUNK_BYTES_EST).max(MIN_POOL_CHUNKS)
  }
}

const STATIC_EMPTY_SCAN_MAX: i64 = 64 * 1024;

fn gpu_pool_bytes(cap_chunks: usize) -> usize {
  if cap_chunks == usize::MAX { 0 } else { cap_chunks.saturating_mul(GPU_CHUNK_BYTES_EST) }
}

fn ledger_note(
  builder: &VolumesBuilder,
  residency: &mut Residency,
  c: gate_voxel::ChunkCoord,
  cam_now: glam::Vec3,
  streamed: bool,
  px: f32,
  frame: u64,
) {
  match builder.resident_bytes_of(0, c) {
    Some(bytes) => {
      if residency.resident_level(c).is_some() {
        residency.note_bytes(c, bytes);
      } else {
                                let lv = if streamed {
          let center = (c.0.as_vec3() + glam::Vec3::splat(0.5)) * gate_voxel::CHUNK_SIZE as f32;
          crate::brickmap::residency::raw_level((center - cam_now).length() * px)
        } else {
          gate_voxel::BRICK_FACTOR
        };
        residency.note_resident(c, bytes, lv, frame);
      }
    }
    None => residency.note_gone(c),
  }
}

fn plan_residency(
  scene: Option<Extract<Res<VoxelScene>>>,
  cam: Option<Extract<Res<crate::brickmap::dda::DdaCameraConfig>>>,
  mut mirror: ResMut<BuilderMirror>,
  mut state: ResMut<ResidencyState>,
  device: Res<RenderDevice>,
  mut pending: Local<Vec<gate_voxel::ChunkCoord>>,
  mut gap_last: Local<usize>,
  mut ledger_seq: Local<u64>,
  mut ledger_ready: Local<bool>,
  mut ledger_epoch: Local<u64>,
  mut ledger_cursor: Local<usize>,
  mut far_seq: Local<Vec<u64>>,
  mut empty_cursor: Local<Vec<usize>>,
  mut diag: Local<(f64, u32)>,
  mut split: Local<Option<crate::profiler::SplitDiag>>,
) {
  let _t = crate::profiler::SysTimer::new("RESID 常驻调度", &mut diag);
  let sd = split.get_or_insert_with(|| {
    crate::profiler::SplitDiag::new(&[
      "①账目",
      "①'空块",
      "②-",
      "④取候选",
      "④逐块",
      "④补缺",
      "④计划装",
      "⑤⑥远场",
      "⑦取证",
      "尾",
    ])
  });
  sd.start();
  let (Some(scene), Some(cam)) = (scene, cam) else { return };
  let Some(builder) = mirror.builder.as_mut() else { return };
  let grid = scene.volumes.main();
  let frame = state.residency.frame().wrapping_add(1);
  state.residency.tick(frame);
  let px = crate::brickmap::dda::px_ang(crate::consts::VIEW_SIZE.y as f32);

        let streamed = grid.stream_window().is_some();
  let seq = grid.resident_seq();
  let epoch = grid.resident_log_epoch();
  let changes = grid.resident_log_from(*ledger_cursor);
  let mut evicted = 0usize;
  let incremental = *ledger_ready
    && *ledger_epoch == epoch
    && changes.len() as u64 == seq.wrapping_sub(*ledger_seq)
    && frame % LEDGER_SWEEP_FRAMES != 0;
  let cam_now = cam.position_world;
  if incremental {
    for ch in changes {
      if ch.mounted {
        ledger_note(builder, &mut state.residency, ch.c, cam_now, streamed, px, frame);
        if !pending.contains(&ch.c) {
          pending.push(ch.c);
        }
      } else {
        state.residency.note_gone(ch.c);
        if builder.evict(0, ch.c) {
          evicted += 1;
        }
        pending.retain(|c| *c != ch.c);
      }
    }
  } else {
    for c in grid.chunk_coords().collect::<Vec<_>>() {
      ledger_note(builder, &mut state.residency, c, cam_now, streamed, px, frame);
    }
    pending.clear();
    pending.extend(grid.chunk_coords().filter(|c| !state.residency.is_resident(*c)));
    *ledger_epoch = epoch;
  }
  *ledger_cursor = grid.resident_log_len();
  *ledger_ready = true;
  *ledger_seq = seq;
  sd.mark(0);

    if empty_cursor.len() != scene.volumes.len() {
    empty_cursor.resize(scene.volumes.len(), 0);
  }
  for (vol, g) in scene.volumes.all().iter().enumerate() {
    if empty_cursor[vol] > g.empty_count() {
      empty_cursor[vol] = 0;
    }
    let fresh = g.empty_log_from(empty_cursor[vol]);
    if fresh.is_empty() {
      continue;
    }
    let n = fresh.iter().filter(|c| builder.note_empty(vol, **c)).count();
    empty_cursor[vol] = g.empty_count();
    bevy::log::debug!("EMPTY[vol{vol}] 哨兵 +{n}（已知空共 {}）", empty_cursor[vol]);
  }
  if grid.stream_window().is_none() {
    let (o, d) = builder.window(0);
    let slots = d.x as i64 * d.y as i64 * d.z as i64;
    if slots > 0 && slots <= STATIC_EMPTY_SCAN_MAX {
      let mut n = 0usize;
      for z in 0..d.z {
        for y in 0..d.y {
          for x in 0..d.x {
            let c = gate_voxel::ChunkCoord(o + IVec3::new(x, y, z));
            if grid.chunk(c).is_none() && builder.note_empty(0, c) {
              n += 1;
            }
          }
        }
      }
      if n > 0 {
        bevy::log::debug!("EMPTY[vol0] 静态世界补标 +{n}（窗口 {d} 槽）");
      }
    }
  }
  sd.mark(1);

  sd.mark(2);
      let cap_chunks = pool_capacity_chunks(scene.residency_budget_bytes);
  let want_bytes = gpu_pool_bytes(cap_chunks);
  let binding_cap = main_region_byte_cap(
    device.limits().max_storage_buffer_binding_size,
    builder.unbudgeted_struct_bytes() as u64,
  );
  let capped = if want_bytes == 0 { binding_cap } else { want_bytes.min(binding_cap) };
  if capped < want_bytes || want_bytes == 0 {
    let asked =
      if want_bytes == 0 { "不限".to_string() } else { format!("{} MB", want_bytes >> 20) };
    bevy::log::warn_once!(
      "RESID[!] 常驻预算被**绑定上限**压到 {} MB（请求 {}）⇒ 池容量随之变小；树区是一整块绑进去的，\
       超过设备的 max_storage_buffer_binding_size 会在 create_bind_group 判错。\
       要更多常驻得先把树区压小，或把 b_struct 分片成多个 binding",
      capped >> 20,
      asked
    );
  }
  state.policy.budget_bytes = capped;

  let cam_pos = cam.position_world;
  let cam_chunk = (cam_pos / gate_voxel::CHUNK_SIZE as f32).floor().as_ivec3();
  let mut wants: Vec<(gate_voxel::ChunkCoord, Level)> = Vec::new();
  let top = state.residency.nearest_top(cam_chunk, cap_chunks);
  let d_cut = if cap_chunks == usize::MAX {
    i32::MAX
  } else {
    top.last().map_or(i32::MAX, |c| crate::brickmap::residency::chunk_distance(cam_chunk, c.0))
  };
  sd.mark(3);
  for c in top {
    let Some(tree) = grid.chunk(c) else { continue };
    if tree.is_empty() {
      continue;
    }
    let cur = state.residency.resident_level(c).unwrap_or(gate_voxel::BRICK_FACTOR);
        if streamed && cur == gate_voxel::BRICK_FACTOR {
      continue;
    }
    let center = (c.0.as_vec3() + glam::Vec3::splat(0.5)) * gate_voxel::CHUNK_SIZE as f32;
    let dist = (center - cam_pos).length();
    let lv = want_level(dist, px, cur);
            let lv = if streamed { lv.min(cur) } else { lv };
    wants.push((c, lv));
  }
  sd.mark(4);
  for &c in pending.iter() {
    if crate::brickmap::residency::chunk_distance(cam_chunk, c.0) > d_cut {
      continue;
    }
    let Some(tree) = grid.chunk(c) else { continue };
    if tree.is_empty() {
      continue;
    }
    let center = (c.0.as_vec3() + glam::Vec3::splat(0.5)) * gate_voxel::CHUNK_SIZE as f32;
    let lv = crate::brickmap::residency::raw_level((center - cam_pos).length() * px);
    wants.push((c, lv));
  }
  sd.mark(5);

  let edited = std::mem::take(&mut state.edited);
  let must_keep: std::collections::HashSet<gate_voxel::ChunkCoord> =
    edited.iter().copied().collect();
  let pin_frames = state.policy.pin_frames;
  for &c in &edited {
    ledger_note(builder, &mut state.residency, c, cam_pos, streamed, px, frame);
    state.residency.note_edit(c, frame, pin_frames);
  }

  let plan = state.residency.plan(&state.policy, cam_chunk, wants.into_iter(), &must_keep);
  for c in plan.evict {
    if builder.evict(0, c) {
      state.residency.note_gone(c);
      evicted += 1;
      if !pending.contains(&c) {
        pending.push(c);
      }
    }
  }
  let mut installed = 0usize;
  let (mut t_new, mut t_up, mut t_down) = (0usize, 0usize, 0usize);
  let mut t_sample: Vec<(gate_voxel::ChunkCoord, Option<Level>, Level)> = Vec::new();
  for (c, level) in plan.install {
    {
      let cur = state.residency.resident_level(c);
      match cur {
        None => t_new += 1,
        Some(x) if level < x => t_up += 1,
        Some(_) => t_down += 1,
      }
      if t_sample.len() < 3 {
        t_sample.push((c, cur, level));
      }
    }
    let Some(tree) = grid.chunk(c) else { continue };
    let proxy;
    let tree = if level > gate_voxel::BRICK_FACTOR {
      proxy = tree.proxy(level);
      &proxy
    } else {
      tree
    };
    let ok = if builder.is_resident(0, c) {
      builder.relayout_resident_tree(0, c, tree)
    } else {
      builder.ensure_resident_tree(0, c, tree)
    };
    if ok {
      let bytes = builder.resident_bytes_of(0, c).unwrap_or(0);
      state.residency.note_resident(c, bytes, level, frame);
      installed += 1;
    }
  }
  pending.retain(|c| !state.residency.is_resident(*c) && grid.chunk(*c).is_some());
  sd.mark(6);
  if frame % LEDGER_SWEEP_FRAMES == 0 {
    for c in builder.resident_chunks(0) {
      if grid.chunk(c).is_none() {
        builder.evict(0, c);
        state.residency.note_gone(c);
        evicted += 1;
      }
    }
  }
  let (mut far_installed, mut far_evicted) = (0usize, 0usize);
  if far_seq.len() != scene.volumes.len() {
    far_seq.resize(scene.volumes.len(), u64::MAX);
  }
  for vol in 1..scene.volumes.len() {
    let far_grid = &scene.volumes.all()[vol];
    if !far_grid.is_far_level() {
      continue;
    }
    let fseq = far_grid.resident_seq();
    if far_seq[vol] == fseq && frame % LEDGER_SWEEP_FRAMES != 0 {
      continue;
    }
    far_seq[vol] = fseq;
    let quota = FAR_INSTALL_PER_FRAME.saturating_sub(far_installed);
    if quota > 0 {
      let mut todo: Vec<gate_voxel::ChunkCoord> =
        far_grid.chunk_coords().filter(|c| !builder.is_resident(vol, *c)).collect();
      if !todo.is_empty() {
        todo.sort_unstable_by_key(|c| (c.0.x, c.0.y, c.0.z));
        for c in todo.into_iter().take(quota) {
          if builder.ensure_resident(&scene.volumes, vol, c) {
            far_installed += 1;
          }
        }
      }
    }
    for c in builder.resident_chunks(vol) {
      if far_grid.chunk(c).is_none() {
        builder.evict(vol, c);
        far_evicted += 1;
      }
    }
  }
  sd.mark(7);
      if installed == 0 && evicted == 0 {
    let (mut gap_count, mut gap_nearest) = (0usize, None);
    for dy in -GAP_NEAR_CHUNKS..=GAP_NEAR_CHUNKS {
      for dz in -GAP_NEAR_CHUNKS..=GAP_NEAR_CHUNKS {
        for dx in -GAP_NEAR_CHUNKS..=GAP_NEAR_CHUNKS {
          let d = dx.abs().max(dy.abs()).max(dz.abs());
          let c = gate_voxel::ChunkCoord(cam_chunk + IVec3::new(dx, dy, dz));
          if grid.chunk(c).is_none() || state.residency.is_resident(c) {
            continue;
          }
          gap_count += 1;
          if gap_nearest.is_none_or(|(_, nd)| d < nd) {
            gap_nearest = Some((c.0, d));
          }
        }
      }
    }
    if gap_count != *gap_last {
      *gap_last = gap_count;
      if let Some((n, _)) = gap_nearest {
        warn!(
          "RESID[!!CPU 有 / GPU 无 {} chunk（{GAP_NEAR_CHUNKS} chunk 内最近 {n}）→ 画面上是空洞 / 齐平断口",
          gap_count
        );
      }
    }
  }
  let busy = installed + evicted + far_installed + far_evicted > 0;
  sd.mark(8);
  if !busy {
    sd.mark(9);
    sd.frame_end("RESID");
    return;
  }
  debug!(
    "INST[install {} 新增 {t_new} 升级 {t_up} 降级 {t_down}] 样本 {:?}",
    installed,
    t_sample.iter().map(|(c, cur, want)| (c.0.to_array(), *cur, *want)).collect::<Vec<_>>()
  );
  debug!(
    "RESID[resident {} {}KB 预算 {}MB install {} evict {} | 远场 {}块/装 {} evict {}]",
    state.residency.resident_count(),
    state.residency.resident_bytes() / 1024,
    state.policy.budget_bytes >> 20,
    installed,
    evicted,
    (1..scene.volumes.len()).map(|v| builder.resident_chunks(v).len()).sum::<usize>(),
    far_installed,
    far_evicted
  );
  sd.mark(9);
  sd.frame_end("RESID");
}

pub struct VolumePlugin;
impl Plugin for VolumePlugin {
  fn build(&self, app: &mut App) {
    let ch = UploadCpuSampleChannel::default();
    app
      .insert_resource(ch.clone())
      .init_resource::<MainPending>()
      .init_resource::<VoxelDumpRequest>()
      .add_plugins(
        bevy::render::extract_resource::ExtractResourcePlugin::<VoxelDumpRequest>::default(),
      )
      .add_systems(Last, (poll_pending, tick_voxel_dump_request));
    let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
      return;
    };
    render_app
      .insert_resource(ch)
      .init_resource::<BrickMapRevision>()
      .init_resource::<ResidencyState>()
      .insert_resource(BuilderMirror { pending_full: true, ..Default::default() })
      .add_systems(RenderStartup, init_empty_gpu)
      .add_systems(ExtractSchedule, (extract, plan_residency).chain())
      .add_systems(Render, prepare.in_set(RenderSystems::PrepareResources))
      .add_systems(Render, dump_voxel_buffers.after(prepare));
  }
}