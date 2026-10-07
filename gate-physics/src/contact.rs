use glam::{IVec3, Vec3};

use crate::classify::{ContactVoxels, DIRS6, Exposed, VoxelClass, x_slab};
use crate::field::{Field, world_aabb_of};

pub const VOXEL_ROUND: f32 = 0.15;

const REGION_EXTENT: i32 = 4;

const CONTACT_MARGIN: f32 = 0.05;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ContactPath {
  Corner,
  Edge,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Contact {
  pub point: Vec3,
  pub normal: Vec3,
  pub depth: f32,
  pub voxel_a: IVec3,
  pub voxel_b: IVec3,
  pub local_a: Vec3,
  pub local_b: Vec3,
  pub path: ContactPath,
}

impl Contact {
  pub fn key(&self) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for v in [self.voxel_a, self.voxel_b] {
      let u = v.as_uvec3();
      for c in [u.x, u.y, u.z] {
        h = (h ^ c as u64).wrapping_mul(0x100_0000_01b3);
      }
    }
    h
  }
}

#[derive(Clone, Debug, Default)]
pub struct Manifold {
  pub points: Vec<Contact>,
}

#[derive(Clone, Copy, Debug)]
pub struct ContactConfig {
  pub max_points: usize,
  pub spread: f32,
}

impl Default for ContactConfig {
  fn default() -> Self {
    Self { max_points: 96, spread: 2.0 }
  }
}

pub struct Probe<'a, 'b> {
  pub field: Field<'b>,
  pub vox: &'a ContactVoxels,
  pub local: (IVec3, IVec3),
  pub world: (Vec3, Vec3),
}

impl Probe<'_, '_> {
  #[inline]
  fn solid_at(&self, p: IVec3) -> bool {
    match self.vox.solid_bit(p) {
      Some(v) => v,
      None => self.field.solid_local(p),
    }
  }
}

pub fn manifold(a: &Field, b: &Field, cfg: &ContactConfig) -> Manifold {
  let (Some(a_local), Some(b_local)) = (a.local_bounds(), b.local_bounds()) else {
    return Manifold::default();
  };
  let a_vox = ContactVoxels::build(a, a_local);
  let b_vox = ContactVoxels::build(b, b_local);
  let pa =
    Probe { field: *a, vox: &a_vox, local: a_local, world: world_aabb_of(a_local, a.transform()) };
  let pb =
    Probe { field: *b, vox: &b_vox, local: b_local, world: world_aabb_of(b_local, b.transform()) };
  manifold_bounded(&pa, &pb, cfg)
}

pub fn manifold_bounded(a: &Probe, b: &Probe, cfg: &ContactConfig) -> Manifold {
  let mut out = Manifold::default();
  if a.world.0.max(b.world.0).cmpgt(a.world.1.min(b.world.1)).any() {
    return out;
  }
  let mut cand = Vec::new();
  if let Some((lo, hi)) = bounds_in(&a.field, a.local, b.world) {
    let dir_a = buried_axis(lo, hi, a.local.0, a.local.1);
    for p in x_slab(a.vox.corners(), lo, hi) {
      probe_pair(a, b, false, p, false, dir_a, &mut cand);
    }
    for p in x_slab(a.vox.edges(), lo, hi) {
      probe_pair(a, b, false, p, true, dir_a, &mut cand);
    }
  }
  if let Some((lo, hi)) = bounds_in(&b.field, b.local, a.world) {
    let dir_b = buried_axis(lo, hi, b.local.0, b.local.1);
    for p in x_slab(b.vox.corners(), lo, hi) {
      probe_pair(a, b, true, p, false, dir_b, &mut cand);
    }
  }
  select(&mut out, cand, cfg, a.field.transform().scale);
  out
}

fn probe_pair(
  a: &Probe,
  b: &Probe,
  swapped: bool,
  p: IVec3,
  edges_only: bool,
  body_dir: Vec3,
  out: &mut Vec<Contact>,
) {
  let (probe, other) = if swapped { (b, a) } else { (a, b) };
  let unit = probe.field.transform().scale;
  let eb = other.field.transform().scale / unit;
  let pa = p.as_vec3() + Vec3::splat(0.5);
  let q = other.field.to_local(probe.field.to_world(pa));
  //
  let r = 0.5 * (1.0 + eb) + CONTACT_MARGIN;
  let lo = (q - Vec3::splat(0.5 + r)).ceil().as_ivec3();
  let hi = (q - Vec3::splat(0.5 - r)).floor().as_ivec3();
  let n = hi - lo + IVec3::ONE;
  if n.cmple(IVec3::ZERO).any() {
    return;
  }
  let bits = if edges_only { other.vox.edge() } else { other.vox.solid() };
  let mut emit = |pb: IVec3| {
    let cb = probe.field.to_local(other.field.to_world(pb.as_vec3() + Vec3::splat(0.5)));
    let Some((mut n, mut depth, deep)) = rounded_pair(pa, cb, eb, body_dir) else { return };
    if deep && let Some(face) = face_axis(other, pb) {
      n = -(probe.field.transform().rot.transpose() * face);
      depth = pair_depth(cb - pa, eb, n).max(0.0);
    }
    let point = probe.field.to_world((pa + cb) * 0.5);
    let (voxel_a, voxel_b) = if swapped { (pb, p) } else { (p, pb) };
    out.push(Contact {
      point,
      normal: probe.field.transform().rot * (if swapped { -n } else { n }),
      depth: depth * unit,
      voxel_a,
      voxel_b,
      local_a: a.field.to_local(point),
      local_b: b.field.to_local(point),
      path: if edges_only { ContactPath::Edge } else { ContactPath::Corner },
    });
  };
  if let Some(base) = bits.cursor(lo, hi) {
    let (sx, syz) = bits.strides();
    for dz in 0..n.z as usize {
      for dy in 0..n.y as usize {
        let row = base + dz * syz + dy * sx;
        for dx in 0..n.x as usize {
          if bits.at(row + dx) {
            emit(lo + IVec3::new(dx as i32, dy as i32, dz as i32));
          }
        }
      }
    }
    return;
  }
  for z in lo.z..=hi.z {
    for y in lo.y..=hi.y {
      for x in lo.x..=hi.x {
        let pb = IVec3::new(x, y, z);
        let hit = if edges_only { other.vox.has_edge(pb) } else { other.solid_at(pb) };
        if hit {
          emit(pb);
        }
      }
    }
  }
}

fn buried_axis(clip_lo: IVec3, clip_hi: IVec3, probe_lo: IVec3, probe_hi: IVec3) -> Vec3 {
  let extent = clip_hi - clip_lo;
  let probe = probe_hi - probe_lo;
  let axis = if extent.x <= extent.y && extent.x <= extent.z {
    0
  } else if extent.y <= extent.z {
    1
  } else {
    2
  };
  let other_min = (0..3).filter(|&a| a != axis).map(|a| extent[a]).min().unwrap_or(0);
  if extent[axis] >= other_min || extent[axis] >= probe[axis] {
    return Vec3::ZERO;
  }
  let low_side = clip_lo[axis] <= probe_lo[axis] && clip_hi[axis] < probe_hi[axis];
  let mut v = Vec3::ZERO;
  v[axis] = if low_side { -1.0 } else { 1.0 };
  v
}

fn rounded_box(p: Vec3, r: f32, buried: Vec3) -> (f32, Vec3) {
  let q = p.abs() - Vec3::splat(1.0 - 2.0 * r);
  let outside = q.max(Vec3::ZERO);
  let d = outside.length() + q.max_element().min(0.0) - r;
  let n = if outside.length_squared() > 1e-12 {
    (outside * p.signum()).normalize_or_zero()
  } else if buried.length_squared() > 0.0 {
    buried
  } else {
    let axis = (0..3).max_by(|&i, &j| p[i].abs().total_cmp(&p[j].abs())).unwrap_or(0);
    let mut v = Vec3::ZERO;
    v[axis] = if p[axis] >= 0.0 { 1.0 } else { -1.0 };
    v
  };
  (d, n)
}

fn rounded_pair(pa: Vec3, pb: Vec3, eb: f32, body_dir: Vec3) -> Option<(Vec3, f32, bool)> {
  let d = pb - pa;
  let (da, na) = rounded_box(d, VOXEL_ROUND, body_dir);
  let (_, nb) = rounded_box(-d, VOXEL_ROUND, -body_dir);
  let n = if da <= 0.0 { na } else { -nb };
  if n.length_squared() <= 0.0 {
    return None;
  }
  let depth = pair_depth(d, eb, n);
  let deep = d.abs().max_element() <= 1.0 - 2.0 * VOXEL_ROUND;
  (depth > -CONTACT_MARGIN).then_some((n, depth.max(0.0), deep))
}

fn pair_depth(d: Vec3, eb: f32, n: Vec3) -> f32 {
  let support = VOXEL_ROUND + (0.5 - VOXEL_ROUND) * (n.x.abs() + n.y.abs() + n.z.abs());
  support * (1.0 + eb) - d.dot(n).abs()
}

fn face_axis(other: &Probe, p: IVec3) -> Option<Vec3> {
  let mut m = 0u8;
  for (i, d) in DIRS6.iter().enumerate() {
    if !other.solid_at(p + *d) {
      m |= 1 << i;
    }
  }
  if Exposed(m).class() != VoxelClass::Face {
    return None;
  }
  let bit = (0..6).find(|i| m & (1 << i) != 0)?;
  Some(other.field.transform().rot * DIRS6[bit].as_vec3())
}

fn select(out: &mut Manifold, mut cand: Vec<Contact>, cfg: &ContactConfig, unit: f32) {
  let mask = IVec3::splat(!(REGION_EXTENT - 1));
  for c in &mut cand {
    c.voxel_a &= mask;
    c.voxel_b &= mask;
  }
  cand.sort_by(|x, y| y.depth.total_cmp(&x.depth));
  let r2 = (cfg.spread * unit) * (cfg.spread * unit);
  for c in cand {
    if out.points.iter().any(|e| {
      (e.voxel_a == c.voxel_a && e.voxel_b == c.voxel_b)
        || (e.point - c.point).length_squared() < r2
    }) {
      continue;
    }
    out.points.push(c);
    if out.points.len() >= cfg.max_points {
      break;
    }
  }
}

fn bounds_in(
  field: &Field,
  field_local: (IVec3, IVec3),
  other_world: (Vec3, Vec3),
) -> Option<(IVec3, IVec3)> {
  let (mn, mx) = other_world;
  let mut lo = Vec3::splat(f32::MAX);
  let mut hi = Vec3::splat(f32::MIN);
  for i in 0..8 {
    let c = Vec3::new(
      if i & 1 == 0 { mn.x } else { mx.x },
      if i & 2 == 0 { mn.y } else { mx.y },
      if i & 4 == 0 { mn.z } else { mx.z },
    );
    let l = field.to_local(c);
    lo = lo.min(l);
    hi = hi.max(l);
  }
  let lo = (lo.floor().as_ivec3() - IVec3::ONE).max(field_local.0);
  let hi = (hi.ceil().as_ivec3() + IVec3::ONE).min(field_local.1);
  lo.cmple(hi).all().then_some((lo, hi))
}
