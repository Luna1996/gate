import package::bindings::{BASE_FLAG_SHADOW, BASE_FLAG_SOFT_SHADOW, gi_u, light_u};
import package::common::{MATERIAL_DEBUG_NONE, MaterialParams, emissive_radiance, face_of_normal, fetch_material};
import package::brickmap::{Grid};
import package::gi::common::{gi_cos_hemisphere, gi_enabled, gi_mix, gi_rand2};
import package::gi::consts::{GI_FALLBACK, GI_LUMA, GI_NEE_P, GI_PI, GI_T_MAX, SHADOW_SURFACE_EPS};
import package::gi::irc::{IrcAddr, irc_accumulate, irc_addr, irc_lookup};
import package::gi::light::{lt_hit_weight, lt_nee, lt_nee_on};
import package::trace::{SECONDARY_REACH_CHUNKS, make_grid};
import package::volumetric::{sky_ambient};
import package::world::{WorldHit, surface_offset, voxel_center_world, voxel_normal_fast, world_trace_medium_scene};

const GI_REACH_T: f32 = SECONDARY_REACH_CHUNKS * 256.0;

struct GiSurface {
  g: Grid,
  p: vec3<f32>,
  n: vec3<f32>,
  m: MaterialParams,
  addr: IrcAddr,
  obj: u32,
  face: u32,
  voxel: vec3<i32>,
}

fn gi_surface(h: WorldHit, g: Grid, n: vec3<f32>, face: u32) -> GiSurface {
  let p = voxel_center_world(g, h.voxel);
  var nf = face;
  if (h.obj_id < 0) { nf = face_of_normal(n); }
  let m = fetch_material(h.palette_base, h.pal, p, n, g.voxel_per_meter, MATERIAL_DEBUG_NONE);
  return GiSurface(g, p, n, m, irc_addr(g, u32(h.obj_id + 1), h.voxel, p), u32(h.obj_id + 1), nf,
                   h.voxel);
}

fn gi_origin(s: GiSurface) -> vec3<f32> {
  return s.p + s.n * surface_offset(s.g, s.n, SHADOW_SURFACE_EPS);
}

fn gi_sun_c() -> vec3<f32> {
  return light_u.lights[0].color_intensity.xyz * light_u.lights[0].color_intensity.w;
}

fn gi_ndl(s: GiSurface) -> f32 {
  return max(dot(s.n, light_u.lights[0].kind_pos_dir.yzw), 0.0);
}

fn gi_sun_vis(s: GiSurface) -> f32 {
  if (gi_ndl(s) <= 0.0) { return 0.0; }
  var tint = vec3<f32>(1.0);
  let h = world_trace_medium_scene(gi_origin(s), light_u.lights[0].kind_pos_dir.yzw,
                                   min(GI_T_MAX, GI_REACH_T), false, false, &tint);
  if (!h.hit) { return dot(tint, GI_LUMA); }
  if (all(h.voxel == s.voxel) && h.obj_id == i32(s.obj) - 1) { return 1.0; }
  return 0.0;
}

fn gi_emission(s: GiSurface, recv_p: vec3<f32>, recv_n: vec3<f32>, to_recv: vec3<f32>,
               dist: f32) -> vec3<f32> {
  let le = emissive_radiance(s.m, gi_u.params.x);
  if (!(max(le.x, max(le.y, le.z)) > 0.0)) { return vec3<f32>(0.0); }
  let obj = select(s.obj, 0u, s.obj == 0u);
  let w = lt_hit_weight(s.voxel, s.face, obj, dot(le, GI_LUMA), recv_p, recv_n, to_recv, dist,
                        s.n, s.g.scale);
  return le * w;
}

fn gi_hit_radiance(s: GiSurface) -> vec3<f32> {
  let c = irc_lookup(s.addr, s.face);
  var sun = 1.0;
  var sun_tint = vec3<f32>(1.0);
  if ((light_u.g.base_flags & BASE_FLAG_SHADOW) != 0u) {
    if ((light_u.g.base_flags & BASE_FLAG_SOFT_SHADOW) != 0u) {
      sun = select(gi_sun_vis(s), c.sun, c.sun_ok);
      sun_tint = select(vec3<f32>(1.0), c.tint, c.sun_ok);
    } else {
      sun = gi_sun_vis(s);
    }
  }
  let hemi = 0.5 + 0.5 * max(s.n.y, 0.0);
  let cold = sky_ambient(s.n) * (GI_FALLBACK * hemi);
  let indirect = mix(cold, c.e, c.conf);
  let kd = 1.0 - s.m.metallic;
  return kd * s.m.albedo
    * (gi_sun_c() * sun_tint * (gi_ndl(s) * sun / GI_PI) + indirect);
}

fn gi_leaf_irradiance(s: GiSurface, seed: u32) -> vec3<f32> {
  var e = vec3<f32>(0.0);
  if (lt_nee_on() && gi_rand2(gi_mix(seed + 0x51ED270Bu)).x < GI_NEE_P) {
    e = lt_nee(gi_origin(s), s.n, gi_mix(seed + 0x68E31DA4u)) / (GI_PI * GI_NEE_P);
  }
  let dir = gi_cos_hemisphere(s.n, gi_rand2(seed));
  var tint = vec3<f32>(1.0);
  let h = world_trace_medium_scene(gi_origin(s), dir, min(GI_T_MAX, GI_REACH_T), false, false, &tint);
  var rad = sky_ambient(dir);
  if (h.hit) {
    let g2 = make_grid(u32(h.obj_id) + 1u);
    let n2 = voxel_normal_fast(g2, h.voxel, h.face_id, h.n);
    let s2 = gi_surface(h, g2, n2, h.face_id);
    rad = gi_emission(s2, s.p, s.n, dir, length(s2.p - s.p)) + s2.m.albedo * gi_hit_radiance(s2);
  }
  return e + rad * tint;
}

fn gi_leaf_indirect(s: GiSurface, seed: u32) -> vec3<f32> {
  let dir = gi_cos_hemisphere(s.n, gi_rand2(seed));
  var tint = vec3<f32>(1.0);
  let h = world_trace_medium_scene(gi_origin(s), dir, min(GI_T_MAX, GI_REACH_T), false, false, &tint);
  var rad = sky_ambient(dir);
  if (h.hit) {
    let g2 = make_grid(u32(h.obj_id) + 1u);
    let n2 = voxel_normal_fast(g2, h.voxel, h.face_id, h.n);
    let s2 = gi_surface(h, g2, n2, h.face_id);
    rad = gi_emission(s2, s.p, s.n, dir, length(s2.p - s.p)) + s2.m.albedo * gi_hit_radiance(s2);
  }
  return rad * tint;
}

fn gi_accumulate(s: GiSurface, seed: u32) {
  if (!gi_enabled()) { return; }
  let sample = gi_leaf_irradiance(s, seed);
  irc_accumulate(s.addr, s.n, sample, true, gi_rand2(gi_mix(seed + 0x9E3779B9u)));
}
