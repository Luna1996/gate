use glam::{IVec3, Vec3};

use gate_voxel::{CHUNK_SIZE, ChunkCoord, ChunkTree, VolumeGrid};

use crate::field::chunk_bounds;

pub fn first_solid_local(
  grid: &VolumeGrid,
  origin: Vec3,
  dir: Vec3,
  t_cap: f32,
) -> Option<(IVec3, f32)> {
  if dir.length_squared() <= 0.0 || t_cap <= 0.0 {
    return None;
  }
  let d = dir;
  let (w_lo, w_hi) = chunk_bounds(grid)?;
  let (t_enter, t_exit) = slab(origin, d, w_lo, w_hi, 0.0, t_cap);
  if t_exit <= t_enter {
    return None;
  }
  let s = CHUNK_SIZE as f32;
  let start = origin + d * t_enter;
  let mut ci = (start / s).floor().as_ivec3();
  let mut t_max = [f32::INFINITY; 3];
  let mut t_delta = [f32::INFINITY; 3];
  for i in 0..3 {
    if d[i].abs() > 1e-30 {
      t_delta[i] = (s / d[i]).abs();
      let bnd = (ci[i] as f32 + if d[i] >= 0.0 { 1.0 } else { 0.0 }) * s;
      t_max[i] = ((bnd - origin[i]) / d[i]).max(t_enter);
    }
  }
  let span = ((w_hi - w_lo) / s).ceil().as_ivec3();
  let budget = (span.x.max(0) + span.y.max(0) + span.z.max(0)) as u32 * 3 + 16;
  let mut t_cur = t_enter;
  for _ in 0..budget {
    let t_exit_c = t_max[0].min(t_max[1]).min(t_max[2]).min(t_exit);
    if let Some(tree) = grid.chunk(ChunkCoord(ci))
      && let Some((v, t)) = step_in_chunk(tree, origin, d, t_cur, t_exit_c)
    {
      return Some((v + ci * CHUNK_SIZE, t));
    }
    if t_exit_c >= t_exit {
      return None;
    }
    let axis = if t_max[0] <= t_max[1] && t_max[0] <= t_max[2] {
      0
    } else if t_max[1] <= t_max[2] {
      1
    } else {
      2
    };
    t_cur = t_max[axis];
    t_max[axis] += t_delta[axis];
    ci[axis] += if d[axis] >= 0.0 { 1 } else { -1 };
  }
  None
}

fn step_in_chunk(tree: &ChunkTree, ro: Vec3, rd: Vec3, t0: f32, t1: f32) -> Option<(IVec3, f32)> {
  let p0 = ro + rd * t0;
  let mut v = [0i32; 3];
  let mut t_next = [f32::INFINITY; 3];
  let mut t_delta = [f32::INFINITY; 3];
  for i in 0..3 {
    v[i] = (p0[i].floor() as i32).clamp(0, CHUNK_SIZE - 1);
    if rd[i].abs() > 1e-30 {
      t_delta[i] = (1.0 / rd[i]).abs();
      let bnd = if rd[i] >= 0.0 { (v[i] + 1) as f32 } else { v[i] as f32 };
      t_next[i] = ((bnd - ro[i]) / rd[i]).max(t0);
    }
  }
  let mut t_cur = t0;
  loop {
    if tree.get_voxel(v[0], v[1], v[2]).is_some() {
      return Some((IVec3::new(v[0], v[1], v[2]), t_cur));
    }
    let axis = if t_next[0] <= t_next[1] && t_next[0] <= t_next[2] {
      0
    } else if t_next[1] <= t_next[2] {
      1
    } else {
      2
    };
    if t_next[axis] > t1 {
      return None;
    }
    t_cur = t_next[axis];
    v[axis] += if rd[axis] >= 0.0 { 1 } else { -1 };
    t_next[axis] += t_delta[axis];
    if v[axis] < 0 || v[axis] >= CHUNK_SIZE {
      return None;
    }
  }
}

fn slab(ro: Vec3, rd: Vec3, mn: Vec3, mx: Vec3, t0: f32, t1: f32) -> (f32, f32) {
  let mut t_enter = t0;
  let mut t_exit = t1;
  for i in 0..3 {
    if rd[i].abs() < 1e-30 {
      if ro[i] < mn[i] || ro[i] > mx[i] {
        return (1.0, 0.0);
      }
    } else {
      let ta = (mn[i] - ro[i]) / rd[i];
      let tb = (mx[i] - ro[i]) / rd[i];
      t_enter = t_enter.max(ta.min(tb));
      t_exit = t_exit.min(tb.max(ta));
    }
  }
  (t_enter, t_exit)
}
