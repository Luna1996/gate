import package::bindings::{light_u};
import package::brickmap::{Grid};
import package::gi::common::{gi_mix, gi_rand2};
import package::gi::consts::{SHADOW_SURFACE_EPS, SUN_SLOTS, SUN_SPP};
import package::gi::irc::{IrcAddr, irc_cell_key, irc_cell_of, irc_sun_next, irc_sun_sample};
import package::world::{face_normal_world, surface_offset, world_trace_medium_scene};

const SUN_GOLDEN: f32 = 2.39996323;
const SUN_T_MAX: f32 = 3072.0;

struct SunRay {
  vis: f32,
  tint: vec3<f32>,
}

fn sun_dir() -> vec3<f32> {
  return normalize(light_u.lights[0].kind_pos_dir.yzw);
}

fn sun_radius() -> f32 {
  return clamp(light_u.lights[0].shape.y, 0.0, 0.5);
}

fn sun_grid_to_world(g: Grid, local: vec3<f32>) -> vec3<f32> {
  let s = local * g.scale;
  return g.pos + s.x * g.col0 + s.y * g.col1 + s.z * g.col2;
}

fn sun_cell_seed(a: IrcAddr, face: u32) -> u32 {
  let k = irc_cell_key(irc_cell_of(a.v, a.level, face, a.obj));
  return gi_mix(k.x ^ gi_mix(k.y + 0x9E3779B9u));
}

fn sun_cell_origin(g: Grid, a: IrcAddr, face: u32, n: vec3<f32>) -> vec3<f32> {
  if (a.level == 0u) {
    var p = vec3<f32>(a.v) + vec3<f32>(0.5);
    if (a.obj != 0u) { p = sun_grid_to_world(g, p); }
    return p + n * surface_offset(g, n, SHADOW_SURFACE_EPS);
  }
  let size = f32(1u << a.level);
  let snap = (a.v >> vec3<u32>(a.level)) << vec3<u32>(a.level);
  let local = vec3<f32>(snap) + vec3<f32>(size * 0.5);
  var p = local;
  var unit = 1.0;
  if (a.obj != 0u) {
    p = sun_grid_to_world(g, local);
    unit = g.scale;
  }
  let fnormal = face_normal_world(g, face);
  return p + fnormal * ((size * 0.5 + SHADOW_SURFACE_EPS) * unit);
}

fn sun_sample_dir(seed: u32, slot: u32) -> vec3<f32> {
  let l = sun_dir();
  let ang = sun_radius();
  if (ang <= 1e-5) { return l; }
  let j = reverseBits(slot) >> 28u;
  let r = sqrt((f32(j) + 0.5) / f32(SUN_SLOTS));
  let rot = gi_rand2(seed).x * 6.2831853;
  let phi = f32(j) * SUN_GOLDEN + rot;
  let axis = select(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 0.0), abs(l.z) > 0.9);
  let t = normalize(cross(axis, l));
  let b = cross(l, t);
  let tan_a = tan(ang);
  return normalize(l + t * (tan_a * r * cos(phi)) + b * (tan_a * r * sin(phi)));
}

fn sun_vis_dir(p: vec3<f32>, dir: vec3<f32>, voxel: vec3<i32>, obj_id: i32,
               level: u32) -> SunRay {
  var tint = vec3<f32>(1.0);
  let h = world_trace_medium_scene(p, dir, SUN_T_MAX, false, false, &tint);
  if (!h.hit) { return SunRay(1.0, tint); }
  if (level == 0u && all(h.voxel == voxel) && h.obj_id == obj_id) {
    return SunRay(1.0, vec3<f32>(1.0));
  }
  return SunRay(0.0, tint);
}

fn sun_gather_cell(g: Grid, a: IrcAddr, face: u32, n: vec3<f32>, voxel: vec3<i32>,
                   obj_id: i32) {
  let seed = sun_cell_seed(a, face);
  let p = sun_cell_origin(g, a, face, n);
  let base = irc_sun_next(a, face);
  for (var i = 0u; i < SUN_SPP; i = i + 1u) {
    let slot = (base + i) & (SUN_SLOTS - 1u);
    let ray = sun_vis_dir(p, sun_sample_dir(seed, slot), voxel, obj_id, a.level);
    irc_sun_sample(a, face, slot, ray.vis, ray.tint);
  }
}
