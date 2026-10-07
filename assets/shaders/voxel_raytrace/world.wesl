import package::bindings::{g, grid_descs, inst_bvh, lod_diag};
import package::brickmap::{Grid, sample_brickmap};
import package::common::{face_normal_from_index};
import package::trace::{MediumRay, UnifiedHit, make_grid, trace_grid, diag_ray, DIAG_NEAR_CALLS,
  DIAG_LEAF_ITER, DIAG_LEAF_MARCH};

fn vol_tint_color(h: WorldHit) -> vec3<f32> {
  if (!h.hit) {
    return vec3<f32>(0.02, 0.02, 0.03); 
  }
  var idx = 255u;
  let n = g.grid_count;
  for (var i: u32 = 0u; i < n; i = i + 1u) {
    if (grid_descs[i].palette_base == h.palette_base) {
      idx = i;
      break;
    }
  }
  
  
  switch (idx) {
    case 0u: { return vec3<f32>(0.05, 0.80, 0.10); }
    case 1u: { return vec3<f32>(0.90, 0.06, 0.06); }
    case 2u: { return vec3<f32>(0.10, 0.16, 0.90); }
    case 3u: { return vec3<f32>(0.90, 0.85, 0.05); }
    default: { return vec3<f32>(0.20, 0.90, 0.90); }
  }
}

fn voxel_normal(g: Grid, voxel: vec3<i32>, face_id: u32, face_n: vec3<f32>) -> vec3<f32> {
  let x = vec3<i32>(1, 0, 0);
  let y = vec3<i32>(0, 1, 0);
  let z = vec3<i32>(0, 0, 1);
  var d = vec3<f32>(0.0);
  d.x = f32(world_solid(g, voxel - x)) - f32(world_solid(g, voxel + x));
  d.y = f32(world_solid(g, voxel - y)) - f32(world_solid(g, voxel + y));
  d.z = f32(world_solid(g, voxel - z)) - f32(world_solid(g, voxel + z));
  
  var dw = d;
  if (g.obj_id >= 0) { dw = d.x * g.col0 + d.y * g.col1 + d.z * g.col2; }
  if (dot(dw, dw) > 0.0) { return normalize(dw); }
      return face_n;
}

fn voxel_center_world(g: Grid, voxel: vec3<i32>) -> vec3<f32> {
  let vc = vec3<f32>(voxel) + vec3<f32>(0.5);
  return g.pos + vec3<f32>(dot(vc, g.col0), dot(vc, g.col1), dot(vc, g.col2)) * g.scale;
}

fn surface_offset(g: Grid, n: vec3<f32>, eps: f32) -> f32 {
  return (0.5 / max(max(abs(n.x), abs(n.y)), max(abs(n.z), 1e-3)) + eps) * g.scale;
}

fn face_normal_world(g: Grid, face_id: u32) -> vec3<f32> {
  let nl = face_normal_from_index(face_id);
  return normalize(nl.x * g.col0 + nl.y * g.col1 + nl.z * g.col2);
}

struct SceneHit {
  uh: UnifiedHit,
  palette_base: u32,
}

fn coverage_r_of(g: Grid) -> f32 {
  return g.coverage_r;
}

fn level_cap(cov: f32, dir: vec3<f32>) -> f32 {
  return cov / max(length(dir.xz), 1e-4);
}

const BVH_HEADER: u32 = 4u;
const BVH_NODE_WORDS: u32 = 8u;
const BVH_STACK: u32 = 16u;

fn bvh_bits(w: u32) -> f32 {
  return bitcast<f32>(inst_bvh[w]);
}

fn bvh_slab(mn: vec3<f32>, mx: vec3<f32>, origin: vec3<f32>, inv: vec3<f32>) -> vec2<f32> {
  let a = (mn - origin) * inv;
  let b = (mx - origin) * inv;
  return vec2<f32>(
    max(max(min(a.x, b.x), min(a.y, b.y)), min(a.z, b.z)),
    min(min(max(a.x, b.x), max(a.y, b.y)), max(a.z, b.z)));
}

fn bvh_node_slab(idx: u32, origin: vec3<f32>, inv: vec3<f32>) -> vec2<f32> {
  let b = BVH_HEADER + idx * BVH_NODE_WORDS;
  let mn = vec3<f32>(bvh_bits(b), bvh_bits(b + 1u), bvh_bits(b + 2u));
  let mx = vec3<f32>(bvh_bits(b + 4u), bvh_bits(b + 5u), bvh_bits(b + 6u));
  return bvh_slab(mn, mx, origin, inv);
}

fn bvh_leaf(gi: Grid, origin: vec3<f32>, dir: vec3<f32>, t_hi_cap: f32, t_min: f32, depth_cap: u32,
            req: u32, best: ptr<function, SceneHit>, best_t: ptr<function, f32>,
            mc: ptr<function, MediumRay>, gap: ptr<function, f32>, marches: ptr<function, u32>) {
  var t_hi = min(*best_t, t_hi_cap);
  if (gi.coverage_r > 0.0) {
    t_hi = min(t_hi, level_cap(gi.coverage_r, dir));
  }
  if (t_min >= t_hi) {
    return;
  }
  *marches = *marches + 1u;
  let mhi = trace_grid(gi, origin, dir, t_hi, t_min, depth_cap, false, req, mc, gap);
  if (mhi.hit && mhi.t < *best_t) {
    *best_t = mhi.t;
    (*best).uh = mhi;
    (*best).palette_base = gi.palette_base;
  }
}

fn trace_near(origin: vec3<f32>, dir: vec3<f32>, t_cap: f32, t_min: f32, depth_cap: u32,
              req: u32, best: ptr<function, SceneHit>, best_t: ptr<function, f32>,
              mc: ptr<function, MediumRay>, gap: ptr<function, f32>) {
  let sampled = diag_ray(origin, dir);
  var leaf_it: u32 = 0u;
  var marches: u32 = 0u;
  if (inst_bvh[0u] == 0u) {
    return;
  }
  let order_base = inst_bvh[1u];
  let inv = 1.0 / dir;
  var stack: array<u32, BVH_STACK>;
  var sp: u32 = 0u;
  stack[0] = 0u;
  sp = 1u;
  loop {
    if (sp == 0u) {
      break;
    }
    sp = sp - 1u;
    let node = stack[sp];
    let base = BVH_HEADER + node * BVH_NODE_WORDS;
    let mn = vec3<f32>(bvh_bits(base), bvh_bits(base + 1u), bvh_bits(base + 2u));
    let mx = vec3<f32>(bvh_bits(base + 4u), bvh_bits(base + 5u), bvh_bits(base + 6u));
    let sl = bvh_slab(mn, mx, origin, inv);
    if (sl.x > sl.y || sl.y < t_min || sl.x > *best_t || sl.x > t_cap) {
      continue;
    }
    let count = inst_bvh[base + 7u];
    if (count == 0u) {
      let left = node + 1u;
      let right = inst_bvh[base + 3u];
      let sl_l = bvh_node_slab(left, origin, inv);
      let sl_r = bvh_node_slab(right, origin, inv);
      let ok_l = sl_l.x <= sl_l.y && sl_l.y >= t_min && sl_l.x <= *best_t && sl_l.x <= t_cap;
      let ok_r = sl_r.x <= sl_r.y && sl_r.y >= t_min && sl_r.x <= *best_t && sl_r.x <= t_cap;
      let l_first = sl_l.x <= sl_r.x;
      let near = select(right, left, l_first);
      let far = select(left, right, l_first);
      let far_ok = select(ok_l, ok_r, l_first);
      let near_ok = select(ok_r, ok_l, l_first);
      if (far_ok && sp + 2u <= BVH_STACK) {
        stack[sp] = far;
        sp = sp + 1u;
      }
      if (near_ok && sp + 1u <= BVH_STACK) {
        stack[sp] = near;
        sp = sp + 1u;
      }
      continue;
    }
    let first = order_base + inst_bvh[base + 3u];
            let tl = max(t_min, sl.x);
    let th = min(t_cap, sl.y);
    leaf_it = leaf_it + count;
    for (var k: u32 = 0u; k < count; k = k + 1u) {
      bvh_leaf(make_grid(inst_bvh[first + k]), origin, dir, th, tl, depth_cap, req,
               best, best_t, mc, gap, &marches);
    }
  }
  if (sampled) {
    atomicAdd(&lod_diag[DIAG_NEAR_CALLS], 1u);
    atomicAdd(&lod_diag[DIAG_LEAF_ITER], leaf_it);
    atomicAdd(&lod_diag[DIAG_LEAF_MARCH], marches);
  }
}

fn trace_far(origin: vec3<f32>, dir: vec3<f32>, t_cap: f32, t_min: f32, depth_cap: u32,
             req: u32, origin_xz: f32, shell: ptr<function, f32>,
             best: ptr<function, SceneHit>, best_t: ptr<function, f32>,
             mc: ptr<function, MediumRay>, gap: ptr<function, f32>) {
  let far_base = inst_bvh[2u];
  let far_count = inst_bvh[3u];
  for (var k: u32 = 0u; k < far_count; k = k + 1u) {
    if (req == 0u && origin_xz < *shell) {
      continue;
    }
    let gi = make_grid(inst_bvh[far_base + k]);
    var t_hi = min(*best_t, t_cap);
    if (gi.coverage_r > 0.0) {
      t_hi = min(t_hi, level_cap(gi.coverage_r, dir));
    }
    if (t_min >= t_hi) {
      *shell = max(*shell, coverage_r_of(gi));
      continue;
    }
    let mhi = trace_grid(gi, origin, dir, t_hi, t_min, depth_cap, false, req, mc, gap);
    if (mhi.hit && mhi.t < *best_t) {
      *best_t = mhi.t;
      (*best).uh = mhi;
      (*best).palette_base = gi.palette_base;
    }
    *shell = max(*shell, coverage_r_of(gi));
  }
}

fn trace_scene(g0: Grid, origin: vec3<f32>, dir: vec3<f32>, t_cap: f32, t_min: f32, depth_cap: u32,
               req: u32) -> SceneHit {
  var mc_off = MediumRay(false, g0.palette_base, 0u, vec3<f32>(1.0));
  var gap: f32 = 1.0e+30;
  var shell = coverage_r_of(g0);
  let cam_est = (g0.w_mn + g0.w_mx) * 0.5;
  let origin_xz = length((origin - cam_est).xz);
  var best_t = 1e+30;
  var g0_cap = t_cap;
  if (g0.coverage_r > 0.0) {
    g0_cap = min(t_cap, level_cap(g0.coverage_r, dir));
  }
  var best = SceneHit(
    trace_grid(g0, origin, dir, g0_cap, t_min, depth_cap, false, req, &mc_off, &gap), 0u);
  if (best.uh.hit) {
    best_t = best.uh.t;
    best.palette_base = g0.palette_base;
  }
  trace_near(origin, dir, t_cap, t_min, depth_cap, req, &best, &best_t, &mc_off, &gap);
  trace_far(origin, dir, t_cap, t_min, depth_cap, req, origin_xz, &shell, &best, &best_t,
            &mc_off, &gap);
  return best;
}

fn world_solid(g: Grid, voxel: vec3<i32>) -> bool {
  return sample_brickmap(g, voxel) != 0u;
}

fn camera_solid_pal(origin_voxel: vec3<f32>) -> u32 {
  return sample_brickmap(make_grid(0u), vec3<i32>(floor(origin_voxel)));
}

struct WorldRayQuery {
  t_min: f32,
  depth_cap: u32,
  
  req: u32,
}

fn world_cfg_full(t_min: f32, depth_cap: u32) -> WorldRayQuery {
  return WorldRayQuery(t_min, depth_cap, 0u);
}

fn world_cfg_primary(t_min: f32, depth_cap: u32) -> WorldRayQuery {
  return WorldRayQuery(t_min, depth_cap, 1u);
}

struct WorldHit {
  hit: bool,
  t: f32,               
  point: vec3<f32>,     
  pal: u32,
  n: vec3<f32>,         
  face_id: u32,         
  voxel: vec3<i32>,     
  obj_id: i32,          
  palette_base: u32,    
}

fn world_miss() -> WorldHit {
  return WorldHit(false, 0.0, vec3<f32>(0.0), 0u, vec3<f32>(0.0), 0u, vec3<i32>(0), -1, 0u);
}

fn world_raycast(g0: Grid, origin: vec3<f32>, dir: vec3<f32>, max_t: f32, cfg: WorldRayQuery) -> WorldHit {
  let s = trace_scene(g0, origin, dir, max_t, cfg.t_min, cfg.depth_cap, cfg.req);
  return WorldHit(
    s.uh.hit, s.uh.t, origin + dir * s.uh.t, s.uh.pal, s.uh.n,
    s.uh.face_id, s.uh.voxel, s.uh.obj_id, s.palette_base);
}

fn world_trace_medium(g: Grid, origin: vec3<f32>, dir: vec3<f32>, max_t: f32,
                      stop_on_air: bool, tint: ptr<function, vec3<f32>>) -> WorldHit {
  var mc = MediumRay(stop_on_air, g.palette_base, 0u, vec3<f32>(1.0));
  
  var gap: f32 = 1.0e+30;
  let h = trace_grid(g, origin, dir, max_t, 0.0, 3u, true, 0u, &mc, &gap);
  *tint = *tint * mc.tint;
  return WorldHit(
    h.hit, h.t, origin + dir * h.t, h.pal, h.n,
    h.face_id, h.voxel, h.obj_id, g.palette_base);
}

fn medium_one(gi: Grid, origin: vec3<f32>, dir: vec3<f32>, t_cap: f32, t_min: f32,
              stop_on_air: bool, best: ptr<function, WorldHit>, best_t: ptr<function, f32>,
              tint: ptr<function, vec3<f32>>) {
  let t_hi = min(*best_t, t_cap);
  if (t_min >= t_hi) {
    return;
  }
  var mc = MediumRay(stop_on_air, gi.palette_base, 0u, vec3<f32>(1.0));
  var gap: f32 = 1.0e+30;
  let h = trace_grid(gi, origin, dir, t_hi, t_min, 3u, true, 0u, &mc, &gap);
  *tint = *tint * mc.tint;
  if (h.hit && h.t < *best_t) {
    *best_t = h.t;
    *best = WorldHit(
      h.hit, h.t, origin + dir * h.t, h.pal, h.n, h.face_id, h.voxel, h.obj_id, gi.palette_base);
  }
}

fn medium_near(origin: vec3<f32>, dir: vec3<f32>, t_cap: f32, stop_on_air: bool,
               best: ptr<function, WorldHit>, best_t: ptr<function, f32>,
               tint: ptr<function, vec3<f32>>) {
  if (inst_bvh[0u] == 0u) {
    return;
  }
  let order_base = inst_bvh[1u];
  let inv = 1.0 / dir;
  var stack: array<u32, BVH_STACK>;
  var sp: u32 = 0u;
  stack[0] = 0u;
  sp = 1u;
  loop {
    if (sp == 0u) {
      break;
    }
    sp = sp - 1u;
    let node = stack[sp];
    let base = BVH_HEADER + node * BVH_NODE_WORDS;
    let mn = vec3<f32>(bvh_bits(base), bvh_bits(base + 1u), bvh_bits(base + 2u));
    let mx = vec3<f32>(bvh_bits(base + 4u), bvh_bits(base + 5u), bvh_bits(base + 6u));
    let sl = bvh_slab(mn, mx, origin, inv);
    if (sl.x > sl.y || sl.y < 0.0 || sl.x > *best_t || sl.x > t_cap) {
      continue;
    }
    let count = inst_bvh[base + 7u];
    if (count == 0u) {
      let left = node + 1u;
      let right = inst_bvh[base + 3u];
      let sl_l = bvh_node_slab(left, origin, inv);
      let sl_r = bvh_node_slab(right, origin, inv);
      let ok_l = sl_l.x <= sl_l.y && sl_l.y >= 0.0 && sl_l.x <= *best_t && sl_l.x <= t_cap;
      let ok_r = sl_r.x <= sl_r.y && sl_r.y >= 0.0 && sl_r.x <= *best_t && sl_r.x <= t_cap;
      let l_first = sl_l.x <= sl_r.x;
      let near = select(right, left, l_first);
      let far = select(left, right, l_first);
      let far_ok = select(ok_l, ok_r, l_first);
      let near_ok = select(ok_r, ok_l, l_first);
      if (far_ok && sp + 2u <= BVH_STACK) {
        stack[sp] = far;
        sp = sp + 1u;
      }
      if (near_ok && sp + 1u <= BVH_STACK) {
        stack[sp] = near;
        sp = sp + 1u;
      }
      continue;
    }
    let first = order_base + inst_bvh[base + 3u];
    let tl = max(0.0, sl.x);
    let th = min(t_cap, sl.y);
    for (var k: u32 = 0u; k < count; k = k + 1u) {
      medium_one(make_grid(inst_bvh[first + k]), origin, dir, th, tl, stop_on_air,
                 best, best_t, tint);
    }
  }
}

fn world_trace_medium_scene(origin: vec3<f32>, dir: vec3<f32>, max_t: f32, stop_on_air: bool,
                            tint: ptr<function, vec3<f32>>) -> WorldHit {
  let g0 = make_grid(0u);
  let cam_est = (g0.w_mn + g0.w_mx) * 0.5;
  let origin_xz = length((origin - cam_est).xz);
  var best = world_miss();
  var best_t = 1e+30;
  var shell = coverage_r_of(g0);
    medium_one(g0, origin, dir, max_t, 0.0, stop_on_air, &best, &best_t, tint);
  medium_near(origin, dir, max_t, stop_on_air, &best, &best_t, tint);
    let far_base = inst_bvh[2u];
  let far_count = inst_bvh[3u];
  for (var k: u32 = 0u; k < far_count; k = k + 1u) {
    if (origin_xz < shell) {
      continue;
    }
    let gi = make_grid(inst_bvh[far_base + k]);
    medium_one(gi, origin, dir, max_t, 0.0, stop_on_air, &best, &best_t, tint);
    shell = max(shell, coverage_r_of(gi));
  }
  return best;
}
