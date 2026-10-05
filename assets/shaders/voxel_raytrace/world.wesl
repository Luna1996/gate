import package::bindings::{g, grid_descs};
import package::brickmap::{Grid, sample_brickmap};
import package::common::{face_normal_from_index};
import package::trace::{MediumRay, UnifiedHit, grid_is_far, make_grid, trace_grid};







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


//





//



//


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



//






fn trace_scene(g0: Grid, origin: vec3<f32>, dir: vec3<f32>, t_cap: f32, t_min: f32, depth_cap: u32,
               req: u32) -> SceneHit {
  var mc_off = MediumRay(false, g0.palette_base, 0u, vec3<f32>(1.0));
  
  var gap: f32 = 1.0e+30;
  let n = g.grid_count;
  
  
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
  
  
  //
  
  
  
  
  var fb = select(1.0e+30, gap, !best.uh.hit);
  for (var i: u32 = 1u; i < n; i = i + 1u) {
    let gi = make_grid(i);
    let far = grid_is_far(i);
    
    
    
    
    
    
    
    
    if (far && req == 0u && origin_xz < shell) {
      continue;
    }
    
    
    
    
    
    
    
    let shell_cap = level_cap(shell, dir);
    
    let t_lo = select(t_min, select(max(t_min, min(shell_cap, fb)), t_min, origin_xz >= shell), far);
    
    
    if (far) {
      fb = 1.0e+30;
    }
    
    
    var t_hi = min(best_t, t_cap);
    if (gi.coverage_r > 0.0) {
      t_hi = min(t_hi, level_cap(gi.coverage_r, dir));
    }
    
    
    
    
    
    if (t_lo >= t_hi) {
      if (far) {
        shell = max(shell, coverage_r_of(gi));
      }
      continue;
    }
    let mhi = trace_grid(gi, origin, dir, t_hi, t_lo, depth_cap, false, req, &mc_off, &gap);
    if (mhi.hit && mhi.t < best_t) {
      best_t = mhi.t;
      best.uh = mhi;
      best.palette_base = gi.palette_base;
    }
    
    
    if (far) {
      shell = max(shell, coverage_r_of(gi));
    }
  }
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















fn world_trace_medium_scene(origin: vec3<f32>, dir: vec3<f32>, max_t: f32, stop_on_air: bool,
                            tint: ptr<function, vec3<f32>>) -> WorldHit {
  let g0 = make_grid(0u);
  let cam_est = (g0.w_mn + g0.w_mx) * 0.5;
  let origin_xz = length((origin - cam_est).xz);
  if (g0.coverage_r <= 0.0 || origin_xz < g0.coverage_r) {
    return world_trace_medium(g0, origin, dir, max_t, stop_on_air, tint);
  }
  let n = g.grid_count;
  var best = world_miss();
  var best_t = 1e+30;
  
  
  
  
  var shell = coverage_r_of(g0);
  for (var i: u32 = 0u; i < n; i = i + 1u) {
    let gi = make_grid(i);
    let far = grid_is_far(i);
    if (far && origin_xz < shell) {
      
      break;
    }
    var mc = MediumRay(stop_on_air, gi.palette_base, 0u, vec3<f32>(1.0));
    var gap: f32 = 1.0e+30;
    let h = trace_grid(gi, origin, dir, min(best_t, max_t), 0.0, 3u, true, 0u, &mc, &gap);
    *tint = *tint * mc.tint;
    if (h.hit && h.t < best_t) {
      best_t = h.t;
      best = WorldHit(
        h.hit, h.t, origin + dir * h.t, h.pal, h.n,
        h.face_id, h.voxel, h.obj_id, gi.palette_base);
    }
    if (far) {
      shell = max(shell, coverage_r_of(gi));
    }
  }
  return best;
}
