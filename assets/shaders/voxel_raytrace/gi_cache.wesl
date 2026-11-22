import package::bindings::{gi_u, light_u, light_cache_meta, view_u};
import package::common::{FLAGS_TRANSMISSIVE, MATERIAL_DEBUG_NONE, SurfaceLight, brdf_reflected, face_view_dir, fetch_material, palette_flags, sky_rgb};
import package::gi::common::{gi_cos_hemisphere, gi_key0, gi_key1, gi_key_ok, gi_mix, gi_rand2};
import package::gi::consts::{GI_PI, GI_SKY_AMBIENT, GI_T_MAX, SHADOW_SURFACE_EPS};
import package::light_cache::{LC_MAX_N, LC_NONE, lc_claim, lc_emit, lc_filter_claim, lc_filtered, lc_filtered_n, lc_filtered_store, lc_finish, lc_find, lc_hit_budget_ok, lc_hit_budget_take, lc_push_emit, lc_push_sun, lc_sun, lc_sun_n, sun_vis_samples};
import package::trace::{make_grid};
import package::world::{WorldHit, surface_offset, voxel_center_world, voxel_normal_fast, world_trace_medium_scene};

const GI_CACHE_MAX_N: u32 = 64u;
const GI_COARSE_DIST: f32 = 1200.0;
const GI_CACHE_BUDGET: u32 = 2048u;
const GI_CELL: i32 = 4;
const GI_CACHE_REACH: f32 = 4096.0;

fn gi_cache_budget_ok() -> bool {
  return atomicLoad(&light_cache_meta[1]) < GI_CACHE_BUDGET;
}

fn gi_cache_budget_take() {
  atomicAdd(&light_cache_meta[1], 1u);
}

fn gi_cache_sun_vis(origin: vec3<f32>, sun_dir: vec3<f32>,
                    self_voxel: vec3<i32>, self_obj: i32) -> vec3<f32> {
  var tint = vec3<f32>(1.0);
  let h = world_trace_medium_scene(origin, sun_dir, min(GI_T_MAX, GI_CACHE_REACH), false, false, &tint);
  if (!h.hit) { return tint; }
  if (all(h.voxel == self_voxel) && h.obj_id == self_obj) { return vec3<f32>(1.0); }
  return vec3<f32>(0.0);
}

fn gi_cache_hit(h: WorldHit, org: vec3<f32>, sun_dir: vec3<f32>, sun_c: vec3<f32>, seed: u32) -> vec3<f32> {
  let g = make_grid(u32(h.obj_id) + 1u);
  let p = voxel_center_world(g, h.voxel);
  let n = voxel_normal_fast(g, h.voxel, h.face_id, h.n);
  if ((palette_flags(h.palette_base, h.pal) & FLAGS_TRANSMISSIVE) != 0u) {
    return vec3<f32>(0.0);
  }
  let m = fetch_material(h.palette_base, h.pal, p, n, MATERIAL_DEBUG_NONE);
  let v = face_view_dir(org, p);
  let o = p + n * surface_offset(g, n, SHADOW_SURFACE_EPS);
  let k0 = gi_key0(h.voxel, h.obj_id);
  let k1 = gi_key1(h.voxel, h.face_id, h.obj_id);
  var vis = vec3<f32>(0.0);
  var known = false;
  var sun_n = 0u;
  var vis_sum = vec3<f32>(0.0);
  var vis_w = 0.0;
  let center = lc_find(k0, k1);
  if (center != LC_NONE) {
    sun_n = lc_sun_n(center);
    if (sun_n > 0u) {
      vis_sum = vis_sum + lc_sun(center) * 2.0;
      vis_w = vis_w + 2.0;
      known = true;
    }
  }
  let axis = h.face_id / 2u;
  let t1 = vec3<i32>(i32(axis == 1u), i32(axis == 2u), i32(axis == 0u));
  let t2 = vec3<i32>(i32(axis == 2u), i32(axis == 0u), i32(axis == 1u));
  let taps = array<vec3<i32>, 4u>(t1, -t1, t2, -t2);
  for (var i: u32 = 0u; i < 4u; i = i + 1u) {
    let vv = h.voxel + taps[i];
    let o = lc_find(gi_key0(vv, h.obj_id), gi_key1(vv, h.face_id, h.obj_id));
    if (o != LC_NONE && lc_sun_n(o) > 0u) {
      vis_sum = vis_sum + lc_sun(o);
      vis_w = vis_w + 1.0;
      known = true;
    }
  }
  if (vis_w > 0.0) { vis = vis_sum / vis_w; }
  if (sun_n < LC_MAX_N && lc_hit_budget_ok()) {
    let upd = lc_claim(k0, k1);
    if (upd != LC_NONE) {
      lc_hit_budget_take();
      let radius = max(light_u.lights[0].shape.y, 0.0);
      let warm = select(1u, 4u, sun_n < 4u);
      for (var i: u32 = 0u; i < warm; i = i + 1u) {
        lc_push_sun(upd, sun_vis_samples(o, sun_dir, radius, h.voxel, h.obj_id,
                                         seed + i * 0x9E3779B1u));
      }
      lc_finish(upd);
      vis = lc_sun(upd);
      known = true;
    }
  }
  if (!known) {
    vis = gi_cache_sun_vis(o, sun_dir, h.voxel, h.obj_id);
  }
  let amb = light_u.sky_color.xyz * GI_SKY_AMBIENT;
  let out = brdf_reflected(m, n, v, SurfaceLight(sun_dir, sun_c * vis, 1.0, amb, vec3<f32>(0.0)));
  if (any(out != out) || any(abs(out) > vec3<f32>(1.0e+20))) { return vec3<f32>(0.0); }
  return min(out, vec3<f32>(4.0));
}

fn gi_cache_ray(origin: vec3<f32>, n: vec3<f32>, sun_dir: vec3<f32>, sun_c: vec3<f32>,
                self_voxel: vec3<i32>, self_obj: i32, seed: u32) -> vec3<f32> {
  let u = gi_rand2(gi_mix(seed));
  let dir = gi_cos_hemisphere(n, u);
  var tint = vec3<f32>(1.0);
  let h = world_trace_medium_scene(origin, dir, min(GI_T_MAX, GI_CACHE_REACH), false, false, &tint);
  if (!h.hit || (all(h.voxel == self_voxel) && h.obj_id == self_obj)) {
    return sky_rgb() * GI_SKY_AMBIENT * tint * GI_PI;
  }
  return gi_cache_hit(h, origin, sun_dir, sun_c, seed) * tint * GI_PI;
}

fn gi_cell_key(voxel: vec3<i32>, obj_id: i32) -> vec2<u32> {
  let c = vec3<i32>(voxel.x / GI_CELL, voxel.y / GI_CELL, voxel.z / GI_CELL) * GI_CELL;
  let lod = u32(make_grid(u32(obj_id + 1)).scale * 16.0);
  let k0 = gi_key0(c, obj_id) ^ gi_mix(lod + 0x51ED270Bu);
  let k1 = gi_mix(gi_key1(c, 0u, obj_id) ^ lod ^ 0x9E3779B9u);
  return vec2<u32>(k0, k1);
}

fn gi_cache_read_cell(voxel: vec3<i32>, obj_id: i32) -> vec4<f32> {
  let c = vec3<i32>(voxel.x / GI_CELL, voxel.y / GI_CELL, voxel.z / GI_CELL) * GI_CELL;
  if (!gi_key_ok(c, obj_id)) { return vec4<f32>(0.0); }
  let k = gi_cell_key(voxel, obj_id);
  let off = lc_find(k.x, k.y);
  if (off == LC_NONE) { return vec4<f32>(0.0); }
  return lc_filtered(off);
}

fn gi_cache_read_raw_cell(voxel: vec3<i32>, obj_id: i32) -> vec4<f32> {
  let c = vec3<i32>(voxel.x / GI_CELL, voxel.y / GI_CELL, voxel.z / GI_CELL) * GI_CELL;
  if (!gi_key_ok(c, obj_id)) { return vec4<f32>(0.0); }
  let k = gi_cell_key(voxel, obj_id);
  let off = lc_find(k.x, k.y);
  if (off == LC_NONE) { return vec4<f32>(0.0); }
  return lc_emit(off);
}

fn gi_cache_filter(voxel: vec3<i32>, obj_id: i32) {
  if (!gi_key_ok(voxel, obj_id)) { return; }
  if (make_grid(u32(obj_id + 1)).scale > 1.01) { return; }
  let ck = gi_cell_key(voxel, obj_id);
  let off = lc_find(ck.x, ck.y);
  if (off == LC_NONE) { return; }
  if (!lc_filter_claim(off)) { return; }
  let cell = vec3<i32>(voxel.x / GI_CELL, voxel.y / GI_CELL, voxel.z / GI_CELL) * GI_CELL;
  var sum = vec3<f32>(0.0);
  var wsum = 0.0;
  for (var dz: i32 = -1; dz <= 1; dz = dz + 1) {
    for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
      for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
        let v = gi_cache_read_raw_cell(cell + vec3<i32>(dx, dy, dz) * GI_CELL, obj_id);
        if (v.w > 0.0) {
          let w = 1.0 / (1.0 + f32(abs(dx) + abs(dy) + abs(dz)));
          sum = sum + v.xyz * w;
          wsum = wsum + w;
        }
      }
    }
  }
  if (!(wsum > 0.0)) { return; }
  let cur = sum / wsum;
  let n = lc_filtered_n(off);
  let val = select(cur, mix(cur, lc_filtered(off).xyz, 0.85), n > 0u);
  lc_filtered_store(off, val, min(n + 1u, 64u));
}

fn gi_cache_read_world(voxel: vec3<i32>, obj_id: i32) -> vec4<f32> {
  let g = make_grid(u32(obj_id + 1));
  let p = voxel_center_world(g, voxel);
  let dist = length(p - view_u.cam_pos_voxel.xyz);
  if (dist > GI_COARSE_DIST) {
    return vec4<f32>(light_u.sky_color.xyz * GI_SKY_AMBIENT * 1.5, 1.0);
  }
  let fc = vec3<f32>(f32(voxel.x), f32(voxel.y), f32(voxel.z)) / f32(GI_CELL);
  let base = vec3<i32>(floor(fc));
  let fr = fc - floor(fc);
  var sum = vec3<f32>(0.0);
  var wsum = 0.0;
  var conf = 0.0;
  for (var dz: i32 = 0; dz <= 1; dz = dz + 1) {
    for (var dy: i32 = 0; dy <= 1; dy = dy + 1) {
      for (var dx: i32 = 0; dx <= 1; dx = dx + 1) {
        let v = gi_cache_read_cell((base + vec3<i32>(dx, dy, dz)) * GI_CELL, obj_id);
        if (!(v.w > 0.0)) { continue; }
        let w = select(1.0 - fr.x, fr.x, dx == 1)
              * select(1.0 - fr.y, fr.y, dy == 1)
              * select(1.0 - fr.z, fr.z, dz == 1);
        sum = sum + v.xyz * w;
        wsum = wsum + w;
        conf = max(conf, v.w);
      }
    }
  }
  if (wsum > 0.0) { return vec4<f32>(sum / wsum, conf); }
  return vec4<f32>(0.0);
}

fn gi_cache_update(off: u32, origin: vec3<f32>, n: vec3<f32>,
                   sun_dir: vec3<f32>, sun_c: vec3<f32>,
                   self_voxel: vec3<i32>, self_obj: i32, seed: u32, samples: u32) {
  for (var i: u32 = 0u; i < samples; i = i + 1u) {
    let rgb = gi_cache_ray(origin, n, sun_dir, sun_c, self_voxel, self_obj,
                           seed + i * 0x9E3779B1u);
    lc_push_emit(off, rgb);
  }
  lc_finish(off);
}
