use glam::IVec3;

use gate_voxel::{BRICK_FACTOR, CHUNK_SIZE};

use crate::field::{Field, NodeFill};

pub const DIRS6: [IVec3; 6] = [
  IVec3::new(-1, 0, 0),
  IVec3::new(1, 0, 0),
  IVec3::new(0, -1, 0),
  IVec3::new(0, 1, 0),
  IVec3::new(0, 0, -1),
  IVec3::new(0, 0, 1),
];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum VoxelClass {
  Interior,
  Face,
  Edge,
  Corner,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Exposed(pub u8);

impl Exposed {
  pub fn full_axes(self) -> u32 {
    (0..3).filter(|&a| self.0 & (3 << (a * 2)) == 0).count() as u32
  }

  pub fn class(self) -> VoxelClass {
    match self.full_axes() {
      3 => VoxelClass::Interior,
      2 => VoxelClass::Face,
      1 => VoxelClass::Edge,
      _ => VoxelClass::Corner,
    }
  }
}

pub fn classify(field: &Field, p: IVec3) -> Exposed {
  let mut m = 0u8;
  for (i, d) in DIRS6.iter().enumerate() {
    if !field.solid_local(p + *d) {
      m |= 1 << i;
    }
  }
  Exposed(m)
}

#[derive(Clone, Debug, Default)]
pub struct VoxelBits {
  lo: IVec3,
  size: IVec3,
  words: Vec<u64>,
}

impl VoxelBits {
  fn covering(lo: IVec3, hi: IVec3) -> Self {
    let size = (hi - lo + IVec3::ONE).max(IVec3::ONE);
    let n = size.x as usize * size.y as usize * size.z as usize;
    Self { lo, size, words: vec![0; n.div_ceil(64)] }
  }

  #[inline]
  fn index(&self, p: IVec3) -> Option<usize> {
    let d = p - self.lo;
    if d.cmpge(IVec3::ZERO).all() && d.cmplt(self.size).all() {
      Some((d.x + (d.y + d.z * self.size.y) * self.size.x) as usize)
    } else {
      None
    }
  }

  #[inline]
  pub fn get(&self, p: IVec3) -> Option<bool> {
    let i = self.index(p)?;
    Some(self.words[i >> 6] & (1u64 << (i & 63)) != 0)
  }

  #[inline]
  pub fn cursor(&self, lo: IVec3, hi: IVec3) -> Option<usize> {
    if lo.cmpge(self.lo).all() && hi.cmplt(self.lo + self.size).all() {
      let d = lo - self.lo;
      Some((d.x + (d.y + d.z * self.size.y) * self.size.x) as usize)
    } else {
      None
    }
  }

  #[inline]
  pub fn strides(&self) -> (usize, usize) {
    let sx = self.size.x as usize;
    (sx, sx * self.size.y as usize)
  }

  #[inline]
  pub fn at(&self, i: usize) -> bool {
    self.words[i >> 6] & (1u64 << (i & 63)) != 0
  }

  #[inline]
  fn set(&mut self, p: IVec3) {
    if let Some(i) = self.index(p) {
      self.words[i >> 6] |= 1u64 << (i & 63);
    }
  }

  fn fill(&mut self, lo: IVec3, hi: IVec3) {
    let lo = lo.max(self.lo);
    let hi = hi.min(self.lo + self.size - IVec3::ONE);
    if lo.cmpgt(hi).any() {
      return;
    }
    let (sx, sy) = (self.size.x as i64, self.size.y as i64);
    for z in lo.z..=hi.z {
      for y in lo.y..=hi.y {
        let base = ((z - self.lo.z) as i64 * sy + (y - self.lo.y) as i64) * sx;
        let start = (base + (lo.x - self.lo.x) as i64) as usize;
        let end = (base + (hi.x - self.lo.x) as i64) as usize;
        self.set_range(start, end);
      }
    }
  }

  fn set_range(&mut self, start: usize, end: usize) {
    let (w0, b0) = (start >> 6, start & 63);
    let (w1, b1) = (end >> 6, end & 63);
    if w0 == w1 {
      self.words[w0] |= (!0u64 << b0) & (!0u64 >> (63 - b1));
      return;
    }
    self.words[w0] |= !0u64 << b0;
    self.words[w0 + 1..w1].fill(!0u64);
    self.words[w1] |= !0u64 >> (63 - b1);
  }
}

#[derive(Clone, Debug, Default)]
pub struct ContactVoxels {
  corners: Vec<IVec3>,
  edges: Vec<IVec3>,
  solid: VoxelBits,
  edge: VoxelBits,
}

impl ContactVoxels {
  pub fn build(field: &Field, (lo, hi): (IVec3, IVec3)) -> Self {
    let mut out = Self {
      solid: VoxelBits::covering(lo, hi),
      edge: VoxelBits::covering(lo, hi),
      ..Self::default()
    };
    let start = lo & IVec3::splat(!(CHUNK_SIZE - 1));
    let mut z = start.z;
    while z <= hi.z {
      let mut y = start.y;
      while y <= hi.y {
        let mut x = start.x;
        while x <= hi.x {
          walk(field, lo, hi, IVec3::new(x, y, z), CHUNK_SIZE, false, &mut out);
          x += CHUNK_SIZE;
        }
        y += CHUNK_SIZE;
      }
      z += CHUNK_SIZE;
    }
    sort_xyz(&mut out.corners);
    sort_xyz(&mut out.edges);
    out
  }

  #[inline]
  pub fn has_edge(&self, p: IVec3) -> bool {
    self.edge.get(p) == Some(true)
  }

  #[inline]
  pub fn solid_bit(&self, p: IVec3) -> Option<bool> {
    self.solid.get(p)
  }

  #[inline]
  pub fn solid(&self) -> &VoxelBits {
    &self.solid
  }

  #[inline]
  pub fn edge(&self) -> &VoxelBits {
    &self.edge
  }

  pub fn corners(&self) -> &[IVec3] {
    &self.corners
  }

  pub fn edges(&self) -> &[IVec3] {
    &self.edges
  }

  pub fn len(&self) -> usize {
    self.corners.len() + self.edges.len()
  }

  pub fn is_empty(&self) -> bool {
    self.len() == 0
  }
}

pub fn x_slab(list: &[IVec3], lo: IVec3, hi: IVec3) -> impl Iterator<Item = IVec3> + '_ {
  let s = list.partition_point(|v| v.x < lo.x);
  let e = list.partition_point(|v| v.x <= hi.x);
  list[s..e]
    .iter()
    .copied()
    .filter(move |v| v.y >= lo.y && v.y <= hi.y && v.z >= lo.z && v.z <= hi.z)
}

fn sort_xyz(list: &mut [IVec3]) {
  list.sort_unstable_by_key(|v| (v.x, v.y, v.z));
}

fn inside(p: IVec3, lo: IVec3, hi: IVec3) -> bool {
  p.cmpge(lo).all() && p.cmple(hi).all()
}

fn walk(
  field: &Field,
  lo: IVec3,
  hi: IVec3,
  origin: IVec3,
  extent: i32,
  solid_done: bool,
  out: &mut ContactVoxels,
) {
  if origin.cmpgt(hi).any() || (origin + IVec3::splat(extent - 1)).cmplt(lo).any() {
    return;
  }
  match field.node(origin, extent) {
    NodeFill::Air => {}
    NodeFill::Solid(_) => {
      if !solid_done {
        out.solid.fill(origin, origin + IVec3::splat(extent - 1));
      }
      if block_enclosed(field, origin, extent) {
        return;
      }
      if extent > 4 {
        let next = extent / BRICK_FACTOR;
        for cz in 0..BRICK_FACTOR {
          for cy in 0..BRICK_FACTOR {
            for cx in 0..BRICK_FACTOR {
              walk(field, lo, hi, origin + IVec3::new(cx, cy, cz) * next, next, true, out);
            }
          }
        }
      } else {
        classify_block(field, lo, hi, origin, true, out);
      }
    }
    NodeFill::Mixed if extent > 4 => {
      let next = extent / BRICK_FACTOR;
      for cz in 0..BRICK_FACTOR {
        for cy in 0..BRICK_FACTOR {
          for cx in 0..BRICK_FACTOR {
            walk(field, lo, hi, origin + IVec3::new(cx, cy, cz) * next, next, false, out);
          }
        }
      }
    }
    _ => classify_block(field, lo, hi, origin, solid_done, out),
  }
}

fn classify_block(
  field: &Field,
  lo: IVec3,
  hi: IVec3,
  origin: IVec3,
  solid_done: bool,
  out: &mut ContactVoxels,
) {
  let c = field.grid().block_solid_bits(origin);
  if c == 0 {
    return;
  }
  let face = |d: IVec3| field.grid().block_solid_bits(origin + d * 4);
  let (xm, xp) = (face(-IVec3::X), face(IVec3::X));
  let (ym, yp) = (face(-IVec3::Y), face(IVec3::Y));
  let (zm, zp) = (face(-IVec3::Z), face(IVec3::Z));
  for dz in 0..4 {
    for dy in 0..4 {
      for dx in 0..4 {
        if !solid_bit(c, dx, dy, dz) {
          continue;
        }
        let p = origin + IVec3::new(dx, dy, dz);
        if !inside(p, lo, hi) {
          continue;
        }
        if !solid_done {
          out.solid.set(p);
        }
        let mut e = 0u8;
        if !neighbor(c, xm, dx - 1, dy, dz) {
          e |= 1 << 0;
        }
        if !neighbor(c, xp, dx + 1, dy, dz) {
          e |= 1 << 1;
        }
        if !neighbor(c, ym, dx, dy - 1, dz) {
          e |= 1 << 2;
        }
        if !neighbor(c, yp, dx, dy + 1, dz) {
          e |= 1 << 3;
        }
        if !neighbor(c, zm, dx, dy, dz - 1) {
          e |= 1 << 4;
        }
        if !neighbor(c, zp, dx, dy, dz + 1) {
          e |= 1 << 5;
        }
        match Exposed(e).class() {
          VoxelClass::Corner => out.corners.push(p),
          VoxelClass::Edge => {
            out.edges.push(p);
            out.edge.set(p);
          }
          _ => {}
        }
      }
    }
  }
}

fn neighbor(c: u64, face: u64, x: i32, y: i32, z: i32) -> bool {
  match (x, y, z) {
    (-1, y, z) => solid_bit(face, 3, y, z),
    (4, y, z) => solid_bit(face, 0, y, z),
    (x, -1, z) => solid_bit(face, x, 3, z),
    (x, 4, z) => solid_bit(face, x, 0, z),
    (x, y, -1) => solid_bit(face, x, y, 3),
    (x, y, 4) => solid_bit(face, x, y, 0),
    _ => solid_bit(c, x, y, z),
  }
}

#[inline]
fn solid_bit(bits: u64, x: i32, y: i32, z: i32) -> bool {
  bits & (1u64 << (z * 16 + y * 4 + x)) != 0
}

fn block_enclosed(field: &Field, origin: IVec3, extent: i32) -> bool {
  for dz in -1..=1 {
    for dy in -1..=1 {
      for dx in -1..=1 {
        if dx == 0 && dy == 0 && dz == 0 {
          continue;
        }
        if !field.node(origin + IVec3::new(dx, dy, dz) * extent, extent).is_solid() {
          return false;
        }
      }
    }
  }
  true
}
