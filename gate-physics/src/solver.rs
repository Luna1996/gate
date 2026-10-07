use glam::Vec3;

use crate::body::BodySet;
use crate::contact::{Contact, ContactPath};

pub trait Constraint {
  fn prepare(&mut self, bodies: &BodySet);
  fn warm_start(&mut self, bodies: &mut BodySet, scale: f32);
  fn solve(&mut self, bodies: &mut BodySet);
  fn key(&self) -> u64;
  fn impulse(&self) -> [f32; 3];
  fn set_impulse(&mut self, v: [f32; 3]);
}

#[derive(Clone, Copy, Debug)]
pub struct ContactParams {
  pub friction: f32,
  pub restitution: f32,
  pub restitution_threshold: f32,
}

impl Default for ContactParams {
  fn default() -> Self {
    Self { friction: 0.6, restitution: 0.2, restitution_threshold: 50.0 }
  }
}

#[derive(Clone, Copy, Debug)]
pub struct ContactConstraint {
  pub ia: usize,
  pub ib: usize,
  key: u64,
  n: Vec3,
  t: [Vec3; 2],
  local_a: Vec3,
  local_b: Vec3,
  ra: Vec3,
  rb: Vec3,
  mass_n: f32,
  mass_t: [f32; 2],
  depth: f32,
  friction: f32,
  accum: [f32; 3],
  target_vn: f32,
  pseudo: f32,
  color: u8,
  path: ContactPath,
}

impl ContactConstraint {
  pub fn new(c: &Contact, ia: usize, ib: usize, p: &ContactParams) -> Self {
    let n = c.normal;
    let a = n.abs();
    let aux = if a.x <= a.y && a.x <= a.z {
      Vec3::X
    } else if a.y <= a.z {
      Vec3::Y
    } else {
      Vec3::Z
    };
    let t1 = n.cross(aux).normalize_or_zero();
    let t2 = n.cross(t1);
    Self {
      ia,
      ib,
      key: mix_pair(c.key(), ia, ib),
      n,
      t: [t1, t2],
      local_a: c.local_a,
      local_b: c.local_b,
      ra: Vec3::ZERO,
      rb: Vec3::ZERO,
      mass_n: 0.0,
      mass_t: [0.0; 2],
      depth: c.depth,
      friction: p.friction,
      accum: [0.0; 3],
      target_vn: 0.0,
      pseudo: 0.0,
      color: 0,
      path: c.path,
    }
  }

  pub fn normal(&self) -> Vec3 {
    self.n
  }

  pub fn path(&self) -> ContactPath {
    self.path
  }

  pub fn penetration(&self) -> f32 {
    self.depth
  }

  pub fn bouncing(&self) -> bool {
    self.target_vn != 0.0
  }

  pub fn reset_pseudo(&mut self) {
    self.pseudo = 0.0;
  }

  pub fn capture_restitution(&mut self, bodies: &BodySet, p: &ContactParams) {
    self.target_vn = 0.0;
    if p.restitution <= 0.0 {
      return;
    }
    let ra = bodies.rot[self.ia] * ((self.local_a - bodies.com[self.ia]) * bodies.scale[self.ia]);
    let rb = bodies.rot[self.ib] * ((self.local_b - bodies.com[self.ib]) * bodies.scale[self.ib]);
    let vn = relative_velocity(bodies, self.ia, self.ib, ra, rb).dot(self.n);
    if vn < -p.restitution_threshold {
      self.target_vn = -p.restitution * vn;
    }
  }

  pub fn solve_position(&mut self, bodies: &mut BodySet, beta: f32, h: f32) {
    let pen = self.depth - POSITION_SLOP;
    if pen <= 0.0 {
      return;
    }
    let target = beta * pen / h;
    let vn = pseudo_velocity(bodies, self.ia, self.ib, self.ra, self.rb).dot(self.n);
    let old = self.pseudo;
    self.pseudo = (old + self.mass_n * (target - vn)).max(0.0);
    let applied = self.pseudo - old;
    if applied != 0.0 {
      apply_pseudo(bodies, self.ia, self.ib, self.n * applied, self.ra, self.rb);
    }
  }
}

pub const POSITION_SLOP: f32 = 0.1;

pub const POSITION_BETA: f32 = 0.45;

impl Constraint for ContactConstraint {
  fn prepare(&mut self, bodies: &BodySet) {
    self.ra = bodies.rot[self.ia] * ((self.local_a - bodies.com[self.ia]) * bodies.scale[self.ia]);
    self.rb = bodies.rot[self.ib] * ((self.local_b - bodies.com[self.ib]) * bodies.scale[self.ib]);
    let k = bodies.effective_mass(self.ia, self.ra, self.n)
      + bodies.effective_mass(self.ib, self.rb, self.n);
    self.mass_n = if k > 1e-12 { 1.0 / k } else { 0.0 };
    for i in 0..2 {
      let k = bodies.effective_mass(self.ia, self.ra, self.t[i])
        + bodies.effective_mass(self.ib, self.rb, self.t[i]);
      self.mass_t[i] = if k > 1e-12 { 1.0 / k } else { 0.0 };
    }
  }

  fn warm_start(&mut self, bodies: &mut BodySet, scale: f32) {
    let p =
      (self.n * self.accum[0] + self.t[0] * self.accum[1] + self.t[1] * self.accum[2]) * scale;
    apply(bodies, self.ia, self.ib, p, self.ra, self.rb);
  }

  fn solve(&mut self, bodies: &mut BodySet) {
    {
      let vn = relative_velocity(bodies, self.ia, self.ib, self.ra, self.rb).dot(self.n);
      let old = self.accum[0];
      self.accum[0] = (old + self.mass_n * (self.target_vn - vn)).max(0.0);
      let applied = self.accum[0] - old;
      if applied != 0.0 {
        apply(bodies, self.ia, self.ib, self.n * applied, self.ra, self.rb);
      }
    }
    if self.friction > 0.0 && self.accum[0] > 0.0 {
      let max_f = self.friction * self.accum[0];
      let v = relative_velocity(bodies, self.ia, self.ib, self.ra, self.rb);
      let want = [
        self.accum[1] + self.mass_t[0] * -v.dot(self.t[0]),
        self.accum[2] + self.mass_t[1] * -v.dot(self.t[1]),
      ];
      let len = (want[0] * want[0] + want[1] * want[1]).sqrt();
      let scale = if len > max_f && len > 1e-12 { max_f / len } else { 1.0 };
      let dp = self.t[0] * (want[0] * scale - self.accum[1])
        + self.t[1] * (want[1] * scale - self.accum[2]);
      self.accum[1] = want[0] * scale;
      self.accum[2] = want[1] * scale;
      if dp != Vec3::ZERO {
        apply(bodies, self.ia, self.ib, dp, self.ra, self.rb);
      }
    }
  }

  fn key(&self) -> u64 {
    self.key
  }

  fn impulse(&self) -> [f32; 3] {
    self.accum
  }

  fn set_impulse(&mut self, v: [f32; 3]) {
    self.accum = v;
  }
}

fn relative_velocity(bodies: &BodySet, ia: usize, ib: usize, ra: Vec3, rb: Vec3) -> Vec3 {
  bodies.point_velocity(ib, rb) - bodies.point_velocity(ia, ra)
}

fn pseudo_velocity(bodies: &BodySet, ia: usize, ib: usize, ra: Vec3, rb: Vec3) -> Vec3 {
  bodies.pseudo_point_velocity(ib, rb) - bodies.pseudo_point_velocity(ia, ra)
}

pub fn color_by_body(cons: &mut [ContactConstraint], bodies: &BodySet) -> Vec<(usize, usize)> {
  const COLOR_WORDS: usize = 4;
  let mut used = vec![[0u64; COLOR_WORDS]; bodies.len()];
  for c in cons.iter_mut() {
    let (sta, stb) = (bodies.is_static(c.ia), bodies.is_static(c.ib));
    let mut blocked = [0u64; COLOR_WORDS];
    if !sta {
      for (b, u) in blocked.iter_mut().zip(&used[c.ia]) {
        *b |= *u;
      }
    }
    if !stb {
      for (b, u) in blocked.iter_mut().zip(&used[c.ib]) {
        *b |= *u;
      }
    }
    let color = blocked
      .iter()
      .enumerate()
      .find_map(|(i, &w)| (w != !0u64).then(|| i * 64 + (!w).trailing_zeros() as usize));
    let Some(color) = color else {
      debug_assert!(false, "颜色数超过 {}", COLOR_WORDS * 64);
      c.color = 0;
      continue;
    };
    let (w, bit) = (color / 64, 1u64 << (color % 64));
    if !sta {
      used[c.ia][w] |= bit;
    }
    if !stb {
      used[c.ib][w] |= bit;
    }
    c.color = color as u8;
  }
  cons.sort_unstable_by_key(|c| c.color);
  let mut ranges = Vec::new();
  let mut start = 0;
  while start < cons.len() {
    let mut end = start;
    while end < cons.len() && cons[end].color == cons[start].color {
      end += 1;
    }
    ranges.push((start, end));
    start = end;
  }
  ranges
}

fn apply(bodies: &mut BodySet, ia: usize, ib: usize, p: Vec3, ra: Vec3, rb: Vec3) {
  bodies.apply_impulse(ib, p, rb);
  bodies.apply_impulse(ia, -p, ra);
}

fn apply_pseudo(bodies: &mut BodySet, ia: usize, ib: usize, p: Vec3, ra: Vec3, rb: Vec3) {
  bodies.apply_pseudo_impulse(ib, p, rb);
  bodies.apply_pseudo_impulse(ia, -p, ra);
}

fn mix_pair(key: u64, ia: usize, ib: usize) -> u64 {
  let mut h = key ^ (ia as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
  h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
  h ^ (ib as u64).wrapping_mul(0x94D0_49BB_1331_11EB)
}

#[cfg(test)]
mod tests {
  use glam::{IVec3, Mat3};

  use super::*;
  use crate::body::MassProps;
  use crate::contact::Contact;

  fn contact(seed: i32) -> Contact {
    Contact {
      point: Vec3::ZERO,
      normal: Vec3::Y,
      depth: 0.1,
      voxel_a: IVec3::splat(seed),
      voxel_b: IVec3::ZERO,
      local_a: Vec3::ZERO,
      local_b: Vec3::ZERO,
      path: ContactPath::Corner,
    }
  }

  fn build(pairs: &[(usize, usize)]) -> (BodySet, Vec<ContactConstraint>) {
    let mut s = BodySet::default();
    s.push_static((IVec3::ZERO, IVec3::ZERO), 0);
    for _ in 1..=3 {
      let props = MassProps { mass: 1.0, com: Vec3::ZERO, inertia: Mat3::IDENTITY };
      s.push_dynamic(props, (IVec3::ZERO, IVec3::ZERO), 0, Vec3::ZERO, Mat3::IDENTITY, 1.0);
    }
    let cons = pairs
      .iter()
      .enumerate()
      .map(|(i, &(ia, ib))| {
        ContactConstraint::new(&contact(i as i32), ia, ib, &ContactParams::default())
      })
      .collect();
    (s, cons)
  }

  #[test]
  fn coloring_ignores_static_bodies_and_isolates_dynamic_sharing() {
    let (s, mut cons) = build(&[(0, 1), (0, 2), (0, 3), (1, 2), (2, 3)]);
    let ranges = color_by_body(&mut cons, &s);
    let ground: Vec<u8> = cons.iter().filter(|c| c.ia == 0).map(|c| c.color).collect();
    assert_eq!(ground, vec![0, 0, 0], "共享静态体的约束必须同色，否则并行度白白丢掉");
    for &(lo, hi) in &ranges {
      for i in lo..hi {
        for j in (i + 1)..hi {
          let (a, b) = (&cons[i], &cons[j]);
          let shares = [a.ia, a.ib].iter().any(|x| !s.is_static(*x) && (*x == b.ia || *x == b.ib));
          assert!(!shares, "同色区间内两个约束共享了动态体");
        }
      }
    }
    assert!(ranges.len() <= 4, "链式堆叠颜色数应很小，实为 {}", ranges.len());
  }
}
