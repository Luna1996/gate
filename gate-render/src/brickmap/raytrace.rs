use std::sync::OnceLock;

use glam::{IVec3, Vec3};

use gate_voxel::{CHUNK_SIZE, ChunkCoord, ChunkTree, VolumeGrid, VolumeTransform, Volumes};

use crate::brickmap::wire::{MARCH_MASK_ENTRIES, march_mask_lut_words};

const BUDGET: u32 = 65536;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RayHit {
  pub t: f32,
  pub pal: u16,
  pub voxel: IVec3,
  pub face: IVec3,
  pub normal: Vec3,
  pub obj_id: i32,
}

pub fn raycast(volumes: &Volumes, origin: Vec3, dir: Vec3, t_max: f32) -> Option<RayHit> {
  let mut best: Option<RayHit> = None;
  for grid in &volumes.list {
    let cap = best.as_ref().map_or(t_max, |b| b.t.min(t_max));
    let is_world = grid.obj_id == -1;
    if let Some(hit) = trace_volume(grid, grid.transform, is_world, origin, dir, cap)
      && best.as_ref().is_none_or(|b| hit.t < b.t)
    {
      best = Some(hit);
    }
  }
  best
}

pub fn raycast_objects(volumes: &Volumes, origin: Vec3, dir: Vec3, t_max: f32) -> Option<RayHit> {
  let mut best: Option<RayHit> = None;
  for grid in volumes.list.iter().skip(1) {
    if grid.is_far_level() {
      continue;
    }
    let cap = best.as_ref().map_or(t_max, |b| b.t.min(t_max));
    if let Some(hit) = trace_volume(grid, grid.transform, false, origin, dir, cap)
      && best.as_ref().is_none_or(|b| hit.t < b.t)
    {
      best = Some(hit);
    }
  }
  best
}

fn trace_volume(
  grid: &VolumeGrid,
  tr: VolumeTransform,
  is_world: bool,
  origin: Vec3,
  dir: Vec3,
  t_cap: f32,
) -> Option<RayHit> {
  let (ro, rd, t_hi_cap) = if is_world {
    (origin, dir, t_cap)
  } else {
    let (w_mn, w_mx) = tr.world_aabb();
    let (t_enter, t_exit) = slab_box(origin, dir, w_mn, w_mx, 0.0, t_cap);
    if t_exit <= t_enter.max(0.0) || t_enter >= t_cap {
      return None;
    }
    let wp = origin - tr.pos;
    let ro =
      Vec3::new(wp.dot(tr.rot.x_axis), wp.dot(tr.rot.y_axis), wp.dot(tr.rot.z_axis)) / tr.scale;
    let rd =
      Vec3::new(dir.dot(tr.rot.x_axis), dir.dot(tr.rot.y_axis), dir.dot(tr.rot.z_axis)) / tr.scale;
    (ro, rd, t_exit.min(t_cap))
  };
  let (lo, hi) = occupied_window(grid)?;
  let (tl_enter, tl_exit) = slab_box(ro, rd, lo, hi, 0.0, t_hi_cap);
  let tl0 = tl_enter.max(0.0);
  let tl1 = tl_exit.min(t_hi_cap);
  if tl1 <= tl0 {
    return None;
  }
  let ro_a = [ro.x, ro.y, ro.z];
  let rd_a = [rd.x, rd.y, rd.z];
  let sign = [
    if rd.x >= 0.0 { 1 } else { -1 },
    if rd.y >= 0.0 { 1 } else { -1 },
    if rd.z >= 0.0 { 1 } else { -1 },
  ];
  let delta = [
    if rd.x.abs() > 1e-30 { rd.x.abs().recip() } else { 1e30 },
    if rd.y.abs() > 1e-30 { rd.y.abs().recip() } else { 1e30 },
    if rd.z.abs() > 1e-30 { rd.z.abs().recip() } else { 1e30 },
  ];
  let delta_c =
    [delta[0] * CHUNK_SIZE as f32, delta[1] * CHUNK_SIZE as f32, delta[2] * CHUNK_SIZE as f32];
  let start = ro + rd * tl0;
  let mut ci = [
    (start.x / CHUNK_SIZE as f32).floor() as i32,
    (start.y / CHUNK_SIZE as f32).floor() as i32,
    (start.z / CHUNK_SIZE as f32).floor() as i32,
  ];
  let mut tmax_c = [1e30f32; 3];
  for i in 0..3 {
    if rd_a[i].abs() > 1e-30 {
      let side = if sign[i] >= 0 { 1.0 } else { 0.0 };
      let bnd = (ci[i] as f32 + side) * CHUNK_SIZE as f32;
      tmax_c[i] = ((bnd - ro_a[i]) / rd_a[i]).max(tl0);
    }
  }
  let mut t_enter_c = tl0;
  let mut entry_face = face_index_from_normal(-rd.normalize_or_zero());
  let span_c = ((hi - lo) / CHUNK_SIZE as f32).ceil();
  let budget = (span_c.x + span_c.y + span_c.z) as u32 * 3 + 16;
  for _ in 0..budget {
    let t_exit_c = tmax_c[0].min(tmax_c[1]).min(tmax_c[2]);
    if let Some(tree) = grid.chunk(ChunkCoord::new(ci[0], ci[1], ci[2])) {
      let s = CHUNK_SIZE as f32;
      let chunk_min = [ci[0] as f32 * s, ci[1] as f32 * s, ci[2] as f32 * s];
      let t1 = t_exit_c.min(tl1);
      if let Some((t, pal, face_id, v)) =
        trace_chunk(tree, chunk_min, ro_a, rd_a, sign, t_enter_c, t1, entry_face)
      {
        let voxel = IVec3::new(v[0], v[1], v[2]) + IVec3::new(ci[0], ci[1], ci[2]) * CHUNK_SIZE;
        let f = face_normal_from_index(face_id);
        let normal = if is_world { f } else { (tr.rot * f).normalize_or_zero() };
        return Some(RayHit { t, pal, voxel, face: f.as_ivec3(), normal, obj_id: grid.obj_id });
      }
    }
    if t_exit_c >= tl1 {
      break;
    }
    if tmax_c[0] <= tmax_c[1] && tmax_c[0] <= tmax_c[2] {
      t_enter_c = tmax_c[0];
      tmax_c[0] += delta_c[0];
      ci[0] += sign[0];
      entry_face = if sign[0] < 0 { 1 } else { 0 };
    } else if tmax_c[1] <= tmax_c[2] {
      t_enter_c = tmax_c[1];
      tmax_c[1] += delta_c[1];
      ci[1] += sign[1];
      entry_face = if sign[1] < 0 { 3 } else { 2 };
    } else {
      t_enter_c = tmax_c[2];
      tmax_c[2] += delta_c[2];
      ci[2] += sign[2];
      entry_face = if sign[2] < 0 { 5 } else { 4 };
    }
  }
  None
}

#[allow(clippy::too_many_arguments)]
fn trace_chunk(
  tree: &ChunkTree,
  chunk_min: [f32; 3],
  ro: [f32; 3],
  rd: [f32; 3],
  sign: [i32; 3],
  t0: f32,
  t1: f32,
  entry_face: u8,
) -> Option<(f32, u16, u8, [i32; 3])> {
  if t0 >= t1 {
    return None;
  }
  let ro_c = [ro[0] - chunk_min[0], ro[1] - chunk_min[1], ro[2] - chunk_min[2]];
  let inv_rd = [1.0 / rd[0], 1.0 / rd[1], 1.0 / rd[2]];
  let abs_inv_rd = [inv_rd[0].abs(), inv_rd[1].abs(), inv_rd[2].abs()];
  let oct =
    (sign[0] >= 0) as usize | (((sign[1] >= 0) as usize) << 1) | (((sign[2] >= 0) as usize) << 2);
  let lut = reach_lut();
  let p0 = [ro_c[0] + rd[0] * t0, ro_c[1] + rd[1] * t0, ro_c[2] + rd[2] * t0];
  let mut v = [
    (p0[0].floor() as i32).clamp(0, 255),
    (p0[1].floor() as i32).clamp(0, 255),
    (p0[2].floor() as i32).clamp(0, 255),
  ];
  let mut cur_t = t0;
  let mut face = entry_face;
  let mut level: u32 = 3;
  let mut budget = BUDGET;
  loop {
    if budget == 0 {
      return None;
    }
    budget -= 1;
    if level > 3 {
      return None;
    }
    let mut desc = tree.node_desc(v[0], v[1], v[2], (3 - level) as u8);
    loop {
      let idx = cell_index(&v, level);
      if level == 0 {
        if let Some(p) = tree.get_voxel(v[0], v[1], v[2]) {
          return Some((cur_t, p.get(), face, v));
        }
        break;
      }
      if desc.mask & (1u64 << idx) == 0 {
        if !desc.palette.is_air() {
          return Some((cur_t, desc.palette.get(), face, v));
        }
        break;
      }
      level -= 1;
      desc = tree.node_desc(v[0], v[1], v[2], (3 - level) as u8);
    }
    if desc.palette.is_air() {
      let entry_i = cell_index(&v, level);
      if desc.mask & lut[oct * MARCH_MASK_ENTRIES + entry_i] == 0 {
        level += 1;
        if level > 3 {
          return None;
        }
        desc = tree.node_desc(v[0], v[1], v[2], (3 - level) as u8);
      }
    }
    let log2 = level * 2;
    let s = 1i32 << log2;
    let mut side = [1e30f32; 3];
    for i in 0..3 {
      if rd[i].abs() > 1e-30 {
        let base = v[i] & !(s - 1);
        let boundary = if sign[i] >= 0 { base + s } else { base };
        side[i] = ((boundary as f32 - ro_c[i]) * inv_rd[i]).max(cur_t);
      }
    }
    let step_inc = [s as f32 * abs_inv_rd[0], s as f32 * abs_inv_rd[1], s as f32 * abs_inv_rd[2]];
    let mut step_axis: usize;
    let mut changed = false;
    loop {
      let mn = if side[0] <= side[1] && side[0] <= side[2] {
        0
      } else if side[1] <= side[2] {
        1
      } else {
        2
      };
      step_axis = mn;
      cur_t = side[mn];
      if cur_t >= t1 {
        return None;
      }
      let old_cell = (v[mn] >> log2) & 3;
      let cell_min = [v[0] & !(s - 1), v[1] & !(s - 1), v[2] & !(s - 1)];
      let p_step = [ro_c[0] + rd[0] * cur_t, ro_c[1] + rd[1] * cur_t, ro_c[2] + rd[2] * cur_t];
      for i in 0..3 {
        let pf = p_step[i].floor() as i32;
        v[i] = pf.clamp(cell_min[i], cell_min[i] + (s - 1));
      }
      v[mn] = if sign[mn] < 0 { cell_min[mn] - 1 } else { cell_min[mn] + s };
      side[mn] += step_inc[mn];
      face = (mn * 2) as u8 + if sign[mn] < 0 { 1 } else { 0 };
      let crossed = if sign[mn] >= 0 { old_cell == 3 } else { old_cell == 0 };
      if crossed {
        changed = true;
        break;
      }
      let idx = cell_index(&v, level);
      if level == 0 {
        if let Some(p) = tree.get_voxel(v[0], v[1], v[2]) {
          return Some((cur_t, p.get(), face, v));
        }
      } else if desc.mask & (1u64 << idx) != 0 {
        break;
      } else if !desc.palette.is_air() {
        return Some((cur_t, desc.palette.get(), face, v));
      }
    }
    if changed {
      level += 1;
      if level > 3 {
        return None;
      }
    }
    let positive = sign[step_axis] >= 0;
    let cur_log2 = level * 2;
    let m: u32 = 0xFFFF_FFFFu32.wrapping_shl(cur_log2);
    let vmin_u = v[step_axis] as u32;
    let comp = if positive { vmin_u & m } else { (vmin_u & m) | !m };
    let tz = comp.wrapping_add(if positive { 0 } else { 1 }).trailing_zeros();
    level = level.max(tz >> 1);
    if level > 3 {
      return None;
    }
    let mi = m as i32;
    let base = [v[0] & mi, v[1] & mi, v[2] & mi];
    let p = [ro_c[0] + rd[0] * cur_t, ro_c[1] + rd[1] * cur_t, ro_c[2] + rd[2] * cur_t];
    for i in 0..3 {
      let pf = p[i].floor() as i32;
      v[i] = pf.clamp(base[i], base[i] + !mi);
    }
    v[step_axis] = comp as i32;
  }
}

#[inline]
fn cell_index(v: &[i32; 3], level: u32) -> usize {
  let sh = level * 2;
  ((((v[2] >> sh) & 3) << 4) | (((v[1] >> sh) & 3) << 2) | ((v[0] >> sh) & 3)) as usize
}

fn occupied_window(grid: &VolumeGrid) -> Option<(Vec3, Vec3)> {
  let mut it = grid.chunk_coords();
  let first = it.next()?.0;
  let (mut lo, mut hi) = (first, first);
  for c in it {
    lo = lo.min(c.0);
    hi = hi.max(c.0);
  }
  let s = CHUNK_SIZE as f32;
  Some((lo.as_vec3() * s, (hi + IVec3::ONE).as_vec3() * s))
}

fn slab_box(ro: Vec3, rd: Vec3, mn: Vec3, mx: Vec3, t0: f32, t1: f32) -> (f32, f32) {
  let o = [ro.x, ro.y, ro.z];
  let d = [rd.x, rd.y, rd.z];
  let lo = [mn.x, mn.y, mn.z];
  let hi = [mx.x, mx.y, mx.z];
  let (mut t_enter, mut t_exit) = (t0, t1);
  for i in 0..3 {
    if d[i].abs() < 1e-30 {
      if o[i] < lo[i] || o[i] > hi[i] {
        return (1.0, 0.0);
      }
    } else {
      let ta = (lo[i] - o[i]) / d[i];
      let tb = (hi[i] - o[i]) / d[i];
      t_enter = t_enter.max(ta.min(tb));
      t_exit = t_exit.min(tb.max(ta));
    }
  }
  (t_enter, t_exit)
}

fn face_index_from_normal(n: Vec3) -> u8 {
  let ax = n.x.abs();
  let ay = n.y.abs();
  let az = n.z.abs();
  if ax >= ay.max(az) {
    if n.x >= 0.0 { 1 } else { 0 }
  } else if ay >= az {
    if n.y >= 0.0 { 3 } else { 2 }
  } else if n.z >= 0.0 {
    5
  } else {
    4
  }
}

fn face_normal_from_index(f: u8) -> Vec3 {
  match f {
    0 => -Vec3::X,
    1 => Vec3::X,
    2 => -Vec3::Y,
    3 => Vec3::Y,
    4 => -Vec3::Z,
    5 => Vec3::Z,
    _ => Vec3::ZERO,
  }
}

fn reach_lut() -> &'static [u64] {
  static LUT: OnceLock<Vec<u64>> = OnceLock::new();
  LUT.get_or_init(|| {
    march_mask_lut_words()
      .as_chunks::<2>()
      .0
      .iter()
      .map(|w| w[0] as u64 | ((w[1] as u64) << 32))
      .collect()
  })
}
