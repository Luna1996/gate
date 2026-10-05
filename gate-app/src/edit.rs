use std::time::Instant;

use bevy::prelude::*;
use glam::IVec3;

use gate_render::brickmap::wire::pack_palette_entry;
use gate_render::{DdaCameraConfig, PbrTextureSet, VoxelScene, raycast};
use gate_voxel::{
  BRICK_FACTOR, BrickState, Displace, FillStats, LEVEL_EXTENT, PALETTE_INDEX_MAX, PaletteEntry,
  PaletteFlags, PaletteId, PbrOverrides, VolumeGrid, VoxelCoord, fill_box_displaced,
  fill_sphere_displaced,
};

use crate::{
  camera::{CameraMode, MouseLock, cursor_ray},
  consts::{
    DEMO_DISPLACE_TEX_SCALE, EDIT_BUDGET_MS, EDIT_DISPLACE_SIZE_MAX, EDIT_REACH, EDIT_REPEAT_SECS,
  },
  height_field::{MaterialDisplace, MaterialDisplaceCache},
};

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum BrushShape {
  #[default]
  Sphere,
  Cube,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BrushMaterial {
  pub color: [u8; 3],
  pub emissive: u8,
  pub transmission: u8,
  pub roughness: u8,
  pub metallic: u8,
  pub pbr: bool,
  pub asset_slot: u32,
}

impl Default for BrushMaterial {
  fn default() -> Self {
    Self {
      color: [0x96, 0x98, 0x9E],
      emissive: 0,
      transmission: 0,
      roughness: 128,
      metallic: 0,
      pbr: false,
      asset_slot: 0,
    }
  }
}

impl BrushMaterial {
  pub fn entry(&self) -> PaletteEntry {
    if self.pbr { self.pbr_entry() } else { self.plain_entry() }
  }

  fn plain_entry(&self) -> PaletteEntry {
    PaletteEntry {
      color: self.color,
      roughness: self.roughness,
      metallic: self.metallic,
      emissive: self.emissive,
      transmission: self.transmission,
      ..Default::default()
    }
  }

  fn pbr_entry(&self) -> PaletteEntry {
    PaletteEntry::pbr(
      self.asset_slot.min(u16::MAX as u32) as u16,
      PbrOverrides::default(),
      PaletteFlags::default(),
    )
  }

  pub fn hex(&self) -> String {
    format!("#{:02X}{:02X}{:02X}", self.color[0], self.color[1], self.color[2])
  }

  pub fn summary(&self) -> String {
    if !self.pbr {
      return format!(
        "平凡 {} 自发光={} 透射率={} 粗糙度={} 金属度={}",
        self.hex(),
        self.emissive,
        self.transmission,
        self.roughness,
        self.metallic
      );
    }
    format!("PBR 变体 asset={}", self.asset_slot)
  }
}

pub fn transparency_pct_to_transmission(pct: f32) -> u8 {
  ((pct.clamp(0.0, 100.0) / 100.0) * 255.0).round() as u8
}

pub fn smooth_pct_to_roughness(pct: f32) -> u8 {
  (((100.0 - pct.clamp(0.0, 100.0)) / 100.0) * 255.0).round() as u8
}

pub fn metal_toggle_to_metallic(on: bool) -> u8 {
  if on { 255 } else { 0 }
}

#[derive(Resource, Clone, Copy, Debug)]
pub struct EditSettings {
  pub shape: BrushShape,
  pub size: u32,
  pub offset: f32,
  pub mat: BrushMaterial,
}

impl Default for EditSettings {
  fn default() -> Self {
    Self { shape: BrushShape::Sphere, size: 3, offset: 1.5, mat: BrushMaterial::default() }
  }
}

fn material_slot(grid: &mut VolumeGrid, mat: BrushMaterial) -> PaletteId {
  let want = pack_palette_entry(&mat.entry());
  let mut existing = None;
  let mut free = None;
  {
    let pal = grid.palette();
    for i in 1u16..=PALETTE_INDEX_MAX {
      let id = PaletteId(i);
      if pal.occupied(id) {
        if existing.is_none() && pack_palette_entry(pal.get(id)) == want {
          existing = Some(id);
        }
      } else if free.is_none() {
        free = Some(id);
      }
      if existing.is_some() && free.is_some() {
        break;
      }
    }
  }
  if let Some(slot) = existing {
    return slot;
  }
  let slot = match free {
    Some(s) => s,
    None => {
      bevy::log::warn!("编辑材质无空调色板槽 → 复用 {PALETTE_INDEX_MAX}（覆盖该槽原材质）");
      PaletteId(PALETTE_INDEX_MAX)
    }
  };
  grid.palette_mut().set(slot, mat.entry());
  slot
}

fn brush_contains(shape: BrushShape, d: IVec3, r: i32) -> bool {
  match shape {
    BrushShape::Cube => d.x.abs() <= r && d.y.abs() <= r && d.z.abs() <= r,
    BrushShape::Sphere => d.length_squared() <= r.saturating_mul(r).saturating_add(r),
  }
}

fn brush_box_disjoint(shape: BrushShape, center: IVec3, r: i32, lo: IVec3, extent: i32) -> bool {
  let hi = lo + IVec3::splat(extent - 1);
  !brush_contains(shape, center.clamp(lo, hi) - center, r)
}

fn brush_box_inside(shape: BrushShape, center: IVec3, r: i32, lo: IVec3, extent: i32) -> bool {
  let e = extent - 1;
  for i in 0..8 {
    let c = IVec3::new(
      if i & 1 == 0 { 0 } else { e },
      if i & 2 == 0 { 0 } else { e },
      if i & 4 == 0 { 0 } else { e },
    );
    if !brush_contains(shape, lo + c - center, r) {
      return false;
    }
  }
  true
}

fn brush_radius(size: u32) -> i32 {
  size.saturating_sub(1).min(i32::MAX as u32) as i32
}

fn block_metric_range(shape: BrushShape, center: IVec3, blo: IVec3, extent: i32) -> (f32, f32) {
  let (mut dmin, mut dmax) = (0.0f32, 0.0f32);
  for a in 0..3 {
    let lo = (blo[a] - center[a]) as f32;
    let hi = lo + (extent - 1) as f32;
    let near = if lo <= 0.0 && hi >= 0.0 { 0.0 } else { lo.abs().min(hi.abs()) };
    let far = lo.abs().max(hi.abs());
    match shape {
      BrushShape::Cube => {
        dmin = dmin.max(near);
        dmax = dmax.max(far);
      }
      BrushShape::Sphere => {
        dmin += near * near;
        dmax += far * far;
      }
    }
  }
  (dmin, dmax)
}

fn stroke_hidden(grid: &VolumeGrid, shape: BrushShape, center: IVec3, r: i32) -> bool {
  let (d_lo, d_hi) = match shape {
    BrushShape::Cube => (r as f32, (r + 1) as f32),
    BrushShape::Sphere => {
      let rf = r as f32;
      (rf * rf + rf, (rf + 1.0) * (rf + 1.0) + rf)
    }
  };
  const EXT: i32 = 4;
  let reach = r.saturating_add(8);
  let lo0 = center.saturating_sub(IVec3::splat(reach)).div_euclid(IVec3::splat(EXT)) * EXT;
  let hi0 = center.saturating_add(IVec3::splat(reach));
  let mut x = lo0.x;
  while x <= hi0.x {
    let mut y = lo0.y;
    while y <= hi0.y {
      let mut z = lo0.z;
      while z <= hi0.z {
        let blo = IVec3::new(x, y, z);
        let (dmin, dmax) = block_metric_range(shape, center, blo, EXT);
        if dmin <= d_hi
          && dmax > d_lo
          && !matches!(grid.get_brick_state_extent(blo, EXT), BrickState::Solid(_))
        {
          return false;
        }
        z += EXT;
      }
      y += EXT;
    }
    x += EXT;
  }
  true
}

struct BrushJob {
  lo: IVec3,
  extent: i32,
}

pub(crate) struct PlainBrush {
  shape: BrushShape,
  center: IVec3,
  r: i32,
  palette: PaletteId,
  erase: bool,
  stack: Vec<BrushJob>,
  pub(crate) changed: usize,
  pub(crate) cpu: std::time::Duration,
  pub(crate) frames: u32,
}

impl PlainBrush {
  pub(crate) fn new(center: IVec3, shape: BrushShape, size: u32, palette: PaletteId) -> Self {
    let r = brush_radius(size);
    let span = r.saturating_mul(2).saturating_add(1);
    let start = LEVEL_EXTENT.iter().copied().find(|&e| e <= span).unwrap_or(1);
    let s = IVec3::splat(start);
    let b_lo = center.saturating_sub(IVec3::splat(r)).div_euclid(s) * s;
    let b_hi = center.saturating_add(IVec3::splat(r)).div_euclid(s) * s;
    let mut stack = Vec::new();
    let mut x = b_lo.x;
    while x <= b_hi.x {
      let mut y = b_lo.y;
      while y <= b_hi.y {
        let mut z = b_lo.z;
        while z <= b_hi.z {
          stack.push(BrushJob { lo: IVec3::new(x, y, z), extent: start });
          z += start;
        }
        y += start;
      }
      x += start;
    }
    stack.reverse();
    Self {
      shape,
      center,
      r,
      palette,
      erase: palette.is_air(),
      stack,
      changed: 0,
      cpu: std::time::Duration::ZERO,
      frames: 0,
    }
  }

  pub(crate) fn step(&mut self, grid: &mut VolumeGrid, budget: std::time::Duration) -> bool {
    let t0 = Instant::now();
    self.frames += 1;
    while let Some(job) = self.stack.pop() {
      self.one(grid, job);
      if t0.elapsed() >= budget {
        break;
      }
    }
    self.cpu += t0.elapsed();
    self.stack.is_empty()
  }

  pub(crate) fn is_erase(&self) -> bool {
    self.erase
  }

  fn one(&mut self, grid: &mut VolumeGrid, job: BrushJob) {
    let (lo, extent) = (job.lo, job.extent);
    if brush_box_disjoint(self.shape, self.center, self.r, lo, extent) {
      return;
    }
    if extent > 1 && brush_box_inside(self.shape, self.center, self.r, lo, extent) {
      match grid.get_brick_state_extent(lo, extent) {
        BrickState::Air => {
          if !self.erase {
            grid.fill_brick(lo, extent, self.palette);
            self.changed += (extent as usize).pow(3);
          }
          return;
        }
        BrickState::Solid(_) => {
          if self.erase {
            grid.fill_brick(lo, extent, self.palette);
            self.changed += (extent as usize).pow(3);
          }
          return;
        }
        BrickState::Mixed => {}
      }
    }
    if extent == 1 {
      if brush_contains(self.shape, lo - self.center, self.r) {
        let cur = grid.get_voxel(VoxelCoord::from_ivec3(lo)).unwrap_or(PaletteId::AIR);
        if cur.is_air() != self.erase && grid.set_voxel_ivec3(lo, self.palette).is_some() {
          self.changed += 1;
        }
      }
      return;
    }
    if extent == BRICK_FACTOR {
      let mut inside = 0u64;
      for i in 0..(BRICK_FACTOR * BRICK_FACTOR * BRICK_FACTOR) {
        let d = IVec3::new(
          i % BRICK_FACTOR,
          (i / BRICK_FACTOR) % BRICK_FACTOR,
          i / (BRICK_FACTOR * BRICK_FACTOR),
        );
        if brush_contains(self.shape, lo + d - self.center, self.r) {
          inside |= 1u64 << i;
        }
      }
      self.changed += grid.set_brick_voxels(lo, inside, self.palette) as usize;
      return;
    }
    let sub = extent / BRICK_FACTOR;
    for i in (0..BRICK_FACTOR * BRICK_FACTOR * BRICK_FACTOR).rev() {
      let d = IVec3::new(
        i % BRICK_FACTOR,
        (i / BRICK_FACTOR) % BRICK_FACTOR,
        i / (BRICK_FACTOR * BRICK_FACTOR),
      );
      self.stack.push(BrushJob { lo: lo + d * sub, extent: sub });
    }
  }
}

pub fn apply_brush(
  grid: &mut VolumeGrid,
  center: IVec3,
  shape: BrushShape,
  size: u32,
  palette: PaletteId,
) -> usize {
  let mut b = PlainBrush::new(center, shape, size, palette);
  let _ = b.step(grid, std::time::Duration::MAX);
  b.changed
}

struct BrushRun {
  voxels: usize,
  stats: Option<FillStats>,
  displace: String,
}

fn run_brush(
  grid: &mut VolumeGrid,
  center: IVec3,
  shape: BrushShape,
  size: u32,
  palette: PaletteId,
  pbr_set: Option<&PbrTextureSet>,
  cache: Option<&mut MaterialDisplaceCache>,
) -> BrushRun {
  let plain = |grid: &mut VolumeGrid, displace: String| BrushRun {
    voxels: apply_brush(grid, center, shape, size, palette),
    stats: None,
    displace,
  };
  if palette.is_air() {
    return plain(grid, "未位移（擦除路径）".to_string());
  }
  let entry = *grid.palette().get(palette);
  let (source, reason) = brush_displace_source(entry, size, pbr_set, cache);
  let Some((md, id)) = source else {
    return plain(grid, reason);
  };
  let f = md.displace_fn();
  let bound = md.bound();
  let Some(st) =
    fill_brush_displaced(grid, center, shape, size, palette, Displace { f: &f, bound })
  else {
    return plain(grid, "未位移（球 size=1：半径为 0，位移球要求 radius > 0）".to_string());
  };
  BrushRun {
    voxels: st.voxels,
    stats: Some(st),
    displace: format!(
      "已按材质 `{id}` 位移，幅度 {} 体素（峰-峰，偏置双向 ±{bound}，一张高度图铺 {DEMO_DISPLACE_TEX_SCALE} 体素）",
      md.amplitude()
    ),
  }
}

fn brush_displace_source<'a>(
  entry: PaletteEntry,
  size: u32,
  pbr_set: Option<&'a PbrTextureSet>,
  cache: Option<&'a mut MaterialDisplaceCache>,
) -> (Option<(&'a MaterialDisplace, &'a str)>, String) {
  if !entry.flags.contains(PaletteFlags::IS_PBR) {
    return (None, "未位移（平凡变体：槽里没有资产 ⇒ 没有高度图）".to_string());
  }
  if size > EDIT_DISPLACE_SIZE_MAX {
    bevy::log::warn!(
      "笔触位移跳过：size={size} > 上限 {EDIT_DISPLACE_SIZE_MAX} vx（位移壳层逐体素 ≈ O(size²)）\
       ⇒ 普通填充，无凹凸"
    );
    return (None, format!("未位移（size {size} > 上限 {EDIT_DISPLACE_SIZE_MAX}）"));
  }
  let Some(set) = pbr_set else {
    return (None, format!("未位移（PBR 贴图集未就绪：读不到资产槽 {} → id）", entry.pbr_asset()));
  };
  let Some(id) = entry_asset_id(&entry, set) else {
    return (None, format!("未位移（资产槽 {} 不在贴图集里）", entry.pbr_asset()));
  };
  let Some(cache) = cache else {
    return (None, "未位移（位移缓存资源缺失）".to_string());
  };
  match cache.get_or_load(id, DEMO_DISPLACE_TEX_SCALE) {
    Some(md) => (Some((md, id)), String::new()),
    None => {
      (None, format!("未位移（材质 `{id}` 的资产幅度 = 0 或高度图不可用，见上面的缓存日志）"))
    }
  }
}

fn entry_asset_id<'a>(entry: &PaletteEntry, set: &'a PbrTextureSet) -> Option<&'a str> {
  (entry.flags.contains(PaletteFlags::IS_PBR))
    .then(|| set.ids().get(entry.pbr_asset() as usize).map(String::as_str))
    .flatten()
}

fn fill_brush_displaced(
  grid: &mut VolumeGrid,
  center: IVec3,
  shape: BrushShape,
  size: u32,
  palette: PaletteId,
  disp: Displace<'_>,
) -> Option<FillStats> {
  let r = size.saturating_sub(1).min(i32::MAX as u32) as i32;
  match shape {
    BrushShape::Cube => {
      let extent = IVec3::splat(r.saturating_mul(2).saturating_add(1));
      Some(fill_box_displaced(grid, center - IVec3::splat(r), extent, palette, Some(disp)))
    }
    BrushShape::Sphere => {
      (r > 0).then(|| fill_sphere_displaced(grid, center, r, palette, Some(disp)))
    }
  }
}

fn stats_suffix(stats: Option<&FillStats>) -> String {
  match stats {
    Some(st) => format!(
      "（整块写 {} 块 = {} 体素 + 壳层逐体素 {} 格）",
      st.whole_bricks,
      st.whole_bricks * 64,
      st.shell_voxels
    ),
    None => String::new(),
  }
}

#[derive(Default)]
pub(crate) struct HoldRepeat {
  place: f32,
  erase: f32,
}

fn hold_repeat(just_pressed: bool, held: bool, acc: &mut f32, dt: f32) -> bool {
  if !held {
    *acc = 0.0;
    return false;
  }
  if just_pressed {
    *acc = 0.0;
    return true;
  }
  *acc += dt;
  if *acc < EDIT_REPEAT_SECS {
    return false;
  }
  *acc -= EDIT_REPEAT_SECS;
  true
}

struct StrokeLog {
  erase: bool,
  voxels: usize,
  center: IVec3,
  shape: BrushShape,
  size: u32,
  off: i32,
  pal: PaletteId,
  material: String,
  reason: String,
  stats: Option<FillStats>,
  cpu: std::time::Duration,
  frames: u32,
  fresh: bool,
}

impl StrokeLog {
  fn emit(&self) {
    if self.voxels == 0 {
      return;
    }
    let cost = if self.frames > 1 {
      format!("{:?} 跨{}帧", self.cpu, self.frames)
    } else {
      format!("{:?}", self.cpu)
    };
    let line = format!(
      "EDIT[{}] {}vx @({},{},{}) shape={:?} size={} offset={} slot={} material={} | {}{} | {cost}",
      if self.erase { "erase" } else { "place" },
      self.voxels,
      self.center.x,
      self.center.y,
      self.center.z,
      self.shape,
      self.size,
      self.off,
      self.pal,
      self.material,
      self.reason,
      stats_suffix(self.stats.as_ref()),
    );
    if self.fresh {
      bevy::log::info!("{line}");
    } else {
      bevy::log::debug!("{line}");
    }
  }
}

pub(crate) struct ActiveStroke {
  brush: PlainBrush,
  size: u32,
  off: i32,
  fresh: bool,
  material: String,
  reason: String,
}

impl ActiveStroke {
  fn to_log(&self) -> StrokeLog {
    StrokeLog {
      erase: self.brush.is_erase(),
      voxels: self.brush.changed,
      center: self.brush.center,
      shape: self.brush.shape,
      size: self.size,
      off: self.off,
      pal: self.brush.palette,
      material: self.material.clone(),
      reason: self.reason.clone(),
      stats: None,
      cpu: self.brush.cpu,
      frames: self.brush.frames,
      fresh: self.fresh,
    }
  }
}

fn brush_budget() -> std::time::Duration {
  std::time::Duration::from_secs_f32(EDIT_BUDGET_MS * 0.001)
}

fn brush_displaced(
  entry: PaletteEntry,
  size: u32,
  pbr_set: Option<&PbrTextureSet>,
  cache: Option<&mut MaterialDisplaceCache>,
) -> (bool, String) {
  let (source, reason) = brush_displace_source(entry, size, pbr_set, cache);
  (source.is_some(), reason)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn voxel_edit_input(
  mouse: Res<ButtonInput<MouseButton>>,
  time: Res<Time>,
  captured: Res<gate_ui::UiPointerCaptured>,
  intercepted: Res<gate_ui::MouseIntercepted>,
  windows: Query<&Window>,
  cfg: Res<DdaCameraConfig>,
  mode: Res<CameraMode>,
  lock: Res<MouseLock>,
  settings: Res<EditSettings>,
  pbr_set: Option<Res<PbrTextureSet>>,
  mut displace_cache: ResMut<MaterialDisplaceCache>,
  scene: Option<ResMut<VoxelScene>>,
  mut active: Local<Option<ActiveStroke>>,
  mut hold: Local<HoldRepeat>,
) {
  let Some(mut scene) = scene else { return };
  if *mode != CameraMode::Fly || scene.demo_force_full_rebuild {
    *active = None;
    return;
  }
  scene.edit_in_flight = active.is_some();
  if active.is_some() {
    let done = {
      let st = active.as_mut().expect("is_some 已判定");
      st.brush.step(scene.volumes.main_mut(), brush_budget())
    };
    if !done {
      return;
    }
    if let Some(st) = active.take() {
      st.to_log().emit();
    }
  }
  if captured.0 || intercepted.0 {
    return;
  }
  let dt = time.delta_secs();
  let place = hold_repeat(
    mouse.just_pressed(MouseButton::Left),
    mouse.pressed(MouseButton::Left),
    &mut hold.place,
    dt,
  );
  let erase = hold_repeat(
    mouse.just_pressed(MouseButton::Right),
    mouse.pressed(MouseButton::Right),
    &mut hold.erase,
    dt,
  );
  if !place && !erase {
    return;
  }
  let fresh = (place && mouse.just_pressed(MouseButton::Left))
    || (erase && mouse.just_pressed(MouseButton::Right));
  let Ok(window) = windows.single() else { return };
  let Some((origin, dir)) = cursor_ray(window, &cfg, lock.0) else {
    return;
  };
  let (shape, size) = (settings.shape, settings.size);
  let Some(hit) = raycast(&scene.volumes, origin, dir, EDIT_REACH) else {
    return;
  };
  if hit.obj_id != -1 {
    return;
  }
  let (hit, face) = (hit.voxel, hit.face);
  let off = settings.offset.max(0.0).round() as i32;
  let pal =
    if erase { PaletteId::AIR } else { material_slot(scene.volumes.main_mut(), settings.mat) };
  let center = if erase { hit - face * off } else { hit + face * off };
  let material = if erase { "-".to_string() } else { settings.mat.summary() };
  let entry = *scene.volumes.main().palette().get(pal);
  let (displaced, reason) =
    brush_displaced(entry, size, pbr_set.as_deref(), Some(&mut displace_cache));
    let hidden = !displaced && stroke_hidden(scene.volumes.main(), shape, center, brush_radius(size));
  let no_backlog = scene
    .volumes
    .list
    .iter()
    .all(|g| g.dirty.data_dirty_count() == 0 && g.dirty.comp_dirty_count() == 0);
  scene.interior_only_edit = hidden && no_backlog;
  let grid = scene.volumes.main_mut();
  if displaced {
    let t0 = Instant::now();
    let run =
      run_brush(grid, center, shape, size, pal, pbr_set.as_deref(), Some(&mut displace_cache));
    StrokeLog {
      erase,
      voxels: run.voxels,
      center,
      shape,
      size,
      off,
      pal,
      material,
      reason: run.displace,
      stats: run.stats,
      cpu: t0.elapsed(),
      frames: 1,
      fresh,
    }
    .emit();
    return;
  }
  let mut st = ActiveStroke {
    brush: PlainBrush::new(center, shape, size, pal),
    size,
    off,
    fresh,
    material,
    reason,
  };
  if st.brush.step(grid, brush_budget()) {
    st.to_log().emit();
  } else {
    *active = Some(st);
  }
}

pub(crate) fn edit_selftest(
  scene: Option<ResMut<VoxelScene>>,
  orbit: Res<gate_render::OrbitCamera>,
  settings: Res<EditSettings>,
  pbr_set: Option<Res<PbrTextureSet>>,
  mut displace_cache: ResMut<MaterialDisplaceCache>,
  mut frame: Local<u32>,
) {
  *frame += 1;
  let nth = match *frame {
    60 => 1,
    70 => 2,
    _ => return,
  };
  let Some(mut scene) = scene else { return };
  let dir = (orbit.target - orbit.eye()).normalize_or_zero();
  if dir.length_squared() < 1e-12 {
    bevy::log::warn!("EDIT SELFTEST: 相机朝向退化 → 跳过");
    return;
  }
  let (shape, size) = (settings.shape, settings.size);
  let Some(hit) = raycast(&scene.volumes, orbit.eye(), dir, EDIT_REACH) else {
    bevy::log::warn!("EDIT SELFTEST: 射线未命中体素 → 跳过");
    return;
  };
  if hit.obj_id != -1 {
    return;
  }
  let (hit, face, t) = (hit.voxel, hit.face, hit.t);
  scene.interior_only_edit = false;
  scene.edit_in_flight = false;
  let grid = scene.volumes.main_mut();
  let slot = material_slot(grid, settings.mat);
  let center = hit + face * settings.offset.max(0.0).round() as i32;
  let t0 = Instant::now();
  let run =
    run_brush(grid, center, shape, size, slot, pbr_set.as_deref(), Some(&mut displace_cache));
  let elapsed = t0.elapsed();
  bevy::log::info!(
    "EDIT SELFTEST 第{nth}笔 hit=({},{},{}) t={t:.1} → {}vx @({},{},{}) shape={:?} size={size} slot={slot} material={} | {}{} | {:?}",
    hit.x,
    hit.y,
    hit.z,
    run.voxels,
    center.x,
    center.y,
    center.z,
    shape,
    settings.mat.summary(),
    run.displace,
    stats_suffix(run.stats.as_ref()),
    elapsed,
  );
}