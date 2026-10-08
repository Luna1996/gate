use std::time::Instant;

use bevy::prelude::*;
use glam::{IVec3, Mat3, Vec3};

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
  Cylinder,
  Cone,
  Capsule,
  Torus,
  Random,
}

pub const BRUSH_SHAPES: [BrushShape; 6] = [
  BrushShape::Sphere,
  BrushShape::Cube,
  BrushShape::Cylinder,
  BrushShape::Cone,
  BrushShape::Capsule,
  BrushShape::Torus,
];

pub const SHAPE_CHOICES: [BrushShape; 7] = [
  BrushShape::Sphere,
  BrushShape::Cube,
  BrushShape::Cylinder,
  BrushShape::Cone,
  BrushShape::Capsule,
  BrushShape::Torus,
  BrushShape::Random,
];

impl BrushShape {
  pub fn from_index(i: usize) -> Self {
    SHAPE_CHOICES.get(i).copied().unwrap_or_default()
  }

  pub fn contains(self, d: IVec3, r: i32) -> bool {
    let r2 = r.saturating_mul(r);
    match self {
      Self::Sphere => d.length_squared() <= r2.saturating_add(r),
      Self::Cube => d.x.abs() <= r && d.y.abs() <= r && d.z.abs() <= r,
      Self::Cylinder => d.x * d.x + d.z * d.z <= r2 && d.y.abs() <= r,
      Self::Cone => {
        d.y >= -r && d.y <= r && {
          let k = r - d.y;
          4 * (d.x * d.x + d.z * d.z) <= k * k
        }
      }
      Self::Capsule => {
        let t = (r / 2).max(1);
        let dy = (d.y.abs() - (r - t).max(0)).max(0);
        d.x * d.x + d.z * d.z + dy * dy <= t * t
      }
      Self::Torus => {
        let t = (r / 4).max(1);
        let big = (r - t).max(0);
        let q = (d.x * d.x + d.z * d.z) as i64;
        let (big2, t2, dy2) = ((big * big) as i64, (t * t) as i64, (d.y * d.y) as i64);
        let lhs = q + big2 + dy2 - t2;
        lhs * lhs <= 4 * big2 * q
      }
      Self::Random => {
        debug_assert!(false, "Random 必须先经 burst() 解析成具体形状");
        Self::Sphere.contains(d, r)
      }
    }
  }

  pub fn is_convex(self) -> bool {
    !matches!(self, Self::Torus)
  }
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

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum EditTarget {
  #[default]
  World,
  Object,
}

#[derive(Resource, Clone, Copy, Debug)]
pub struct EditSettings {
  pub shape: BrushShape,
  pub size: u32,
  pub offset: f32,
  pub mat: BrushMaterial,
  pub target: EditTarget,
}

impl Default for EditSettings {
  fn default() -> Self {
    Self {
      shape: BrushShape::Sphere,
      size: 3,
      offset: 1.5,
      mat: BrushMaterial::default(),
      target: EditTarget::Object,
    }
  }
}

pub(crate) const RANDOM_SIZE_MIN: u32 = 10;
pub(crate) const RANDOM_SIZE_MAX: u32 = 30;

pub(crate) struct Rng(u64);

impl Default for Rng {
  fn default() -> Self {
    Self::new()
  }
}

impl Rng {
  pub(crate) const SEED: u64 = 0x5EED_5EED_5EED_5EED;

  pub(crate) fn new() -> Self {
    Self(Self::SEED)
  }
  fn next_u64(&mut self) -> u64 {
    self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = self.0;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
  }

  fn unit(&mut self) -> f32 {
    (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
  }

  fn int(&mut self, lo: i32, hi: i32) -> i32 {
    lo + (self.next_u64() % (hi - lo + 1) as u64) as i32
  }

  fn dir(&mut self) -> Vec3 {
    let d = Vec3::new(self.unit() * 2.0 - 1.0, self.unit() * 2.0 - 1.0, self.unit() * 2.0 - 1.0);
    if d.length_squared() < 1e-6 { Vec3::Y } else { d.normalize() }
  }
}

#[derive(Clone, Copy)]
pub(crate) struct Burst {
  pub shape: BrushShape,
  pub size: u32,
  pub mat: BrushMaterial,
  pub rot: Mat3,
}

impl Burst {
  pub(crate) fn fixed(shape: BrushShape, size: u32, mat: BrushMaterial) -> Self {
    Self { shape, size, mat, rot: Mat3::IDENTITY }
  }
}

pub(crate) fn burst(settings: &EditSettings, rng: &mut Rng) -> Burst {
  if settings.shape != BrushShape::Random {
    return Burst {
      shape: settings.shape,
      size: settings.size,
      mat: settings.mat,
      rot: Mat3::IDENTITY,
    };
  }
  let mut mat = settings.mat;
  mat.pbr = false;
  mat.color = [rng.int(48, 255) as u8, rng.int(48, 255) as u8, rng.int(48, 255) as u8];
  Burst {
    shape: BRUSH_SHAPES[rng.int(0, BRUSH_SHAPES.len() as i32 - 1) as usize],
    size: rng.int(RANDOM_SIZE_MIN as i32, RANDOM_SIZE_MAX as i32) as u32,
    mat,
    rot: Mat3::from_axis_angle(rng.dir(), rng.unit() * std::f32::consts::TAU),
  }
}

pub(crate) fn material_slot(grid: &mut VolumeGrid, mat: BrushMaterial) -> PaletteId {
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

fn brush_box_disjoint(shape: BrushShape, center: IVec3, r: i32, lo: IVec3, extent: i32) -> bool {
  let hi = lo + IVec3::splat(extent - 1);
  if shape == BrushShape::Torus {
    let t = (r / 4).max(1) as i64;
    let big = (r - t as i32).max(0) as i64;
    let span = |a: i64, b: i64| -> (i64, i64) {
      let near = if a > 0 {
        a
      } else if b < 0 {
        -b
      } else {
        0
      };
      (near, a.abs().max(b.abs()))
    };
    let (nx, fx) = span((lo.x - center.x) as i64, (hi.x - center.x) as i64);
    let (nz, fz) = span((lo.z - center.z) as i64, (hi.z - center.z) as i64);
    let (ny, _) = span((lo.y - center.y) as i64, (hi.y - center.y) as i64);
    let (rho_min2, rho_max2) = (nx * nx + nz * nz, fx * fx + fz * fz);
    let (inner, outer) = ((big - t).max(0), big + t);
    return ny > t || rho_min2 > outer * outer || rho_max2 < inner * inner;
  }
  !shape.contains(center.clamp(lo, hi) - center, r)
}

fn brush_box_inside(shape: BrushShape, center: IVec3, r: i32, lo: IVec3, extent: i32) -> bool {
  if !shape.is_convex() {
    return false;
  }
  let e = extent - 1;
  for i in 0..8 {
    let c = IVec3::new(
      if i & 1 == 0 { 0 } else { e },
      if i & 2 == 0 { 0 } else { e },
      if i & 4 == 0 { 0 } else { e },
    );
    if !shape.contains(lo + c - center, r) {
      return false;
    }
  }
  true
}

pub(crate) fn brush_radius(size: u32) -> i32 {
  size.saturating_sub(1).min(i32::MAX as u32) as i32
}

fn block_metric_range(shape: BrushShape, center: IVec3, blo: IVec3, extent: i32) -> (f32, f32) {
  let (mut dmin, mut dmax) = (0.0f32, 0.0f32);
  for a in 0..3 {
    let lo = (blo[a] - center[a]) as f32;
    let hi = lo + (extent - 1) as f32;
    let near = if lo <= 0.0 && hi >= 0.0 { 0.0 } else { lo.abs().min(hi.abs()) };
    let far = lo.abs().max(hi.abs());
    if shape == BrushShape::Cube {
      dmin = dmin.max(near);
      dmax = dmax.max(far);
    } else {
      dmin += near * near;
      dmax += far * far;
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
    _ => return false,
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
      if self.shape.contains(lo - self.center, self.r) {
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
        if self.shape.contains(lo + d - self.center, self.r) {
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
    _ => None,
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

impl HoldRepeat {
  pub(crate) fn erase_tick(&mut self, just_pressed: bool, held: bool, dt: f32) -> bool {
    hold_repeat(just_pressed, held, &mut self.erase, dt)
  }
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
  target: i32,
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
  mut rng: Local<Rng>,
) {
  let Some(mut scene) = scene else { return };
  if *mode != CameraMode::Fly
    || scene.demo_force_full_rebuild
    || settings.target != EditTarget::World
  {
    *active = None;
    scene.edit_in_flight = false;
    return;
  }
  scene.edit_in_flight = active.is_some();
  if active.is_some() {
    let target = active.as_ref().map_or(-1, |st| st.target);
    let Some(grid) = scene.volumes.volume_mut(target) else {
      *active = None;
      return;
    };
    let done = active.as_mut().expect("is_some 已判定").brush.step(grid, brush_budget());
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
  let b = burst(&settings, &mut rng);
  let (shape, size) = (b.shape, b.size);
  let Some(hit) = raycast(&scene.volumes, origin, dir, EDIT_REACH) else {
    return;
  };
  let target = hit.obj_id;
  let (voxel, face) = (hit.voxel, hit.face);
  let off = settings.offset.max(0.0).round() as i32;
  let pal = if erase {
    PaletteId::AIR
  } else {
    let Some(grid) = scene.volumes.volume_mut(target) else {
      return;
    };
    material_slot(grid, b.mat)
  };
  let center = if erase { voxel - face * off } else { voxel + face * off };
  let material = if erase { "-".to_string() } else { b.mat.summary() };
  let entry =
    scene.volumes.volume(target).map_or_else(PaletteEntry::default, |g| *g.palette().get(pal));
  let (displaced, reason) =
    brush_displaced(entry, size, pbr_set.as_deref(), Some(&mut displace_cache));
  let hidden = scene
    .volumes
    .volume(target)
    .is_some_and(|g| !displaced && stroke_hidden(g, shape, center, brush_radius(size)));
  let no_backlog = scene
    .volumes
    .list
    .iter()
    .all(|g| g.dirty.data_dirty_count() == 0 && g.dirty.comp_dirty_count() == 0);
  scene.interior_only_edit = hidden && no_backlog;
  let Some(grid) = scene.volumes.volume_mut(target) else {
    return;
  };
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
    target,
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
  let b = burst(&settings, &mut Rng::new());
  let (shape, size) = (b.shape, b.size);
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
  let slot = material_slot(grid, b.mat);
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
    b.mat.summary(),
    run.displace,
    stats_suffix(run.stats.as_ref()),
    elapsed,
  );
}
