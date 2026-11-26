import package::bindings::{light_u};
import package::brickmap::{Grid};
import package::gi::common::{gi_mix, gi_rand2};
import package::gi::consts::{SHADOW_SURFACE_EPS, SUN_SLOTS, SUN_SPP};
import package::gi::irc::{IrcAddr, irc_cell_key, irc_cell_of, irc_sun_begin, irc_sun_cached, irc_sun_next, irc_sun_sample};
import package::world::{face_normal_world, surface_offset, world_trace_medium_scene};

const SUN_GOLDEN: f32 = 2.39996323;
const SUN_T_MAX: f32 = 3072.0;
// 满槽后每 16 帧刷新 1 槽，兜住流式加载/远处几何变化；必须为 2 的幂。
const SUN_CACHE_REFRESH: u32 = 16u;

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
  let cached = irc_sun_cached(a, face);
  if (cached != 0u) {
    if (cached == 1u) {
      let seed = sun_cell_seed(a, face);
      if (((gi_u.seq.x ^ seed) & (SUN_CACHE_REFRESH - 1u)) == 0u) {
        let p = sun_cell_origin(g, a, face, n);
        let slot = (gi_u.seq.x >> 4u) & (SUN_SLOTS - 1u);
        let ray = sun_vis_dir(p, sun_sample_dir(seed, slot), voxel, obj_id, a.level);
        irc_sun_sample(a, face, slot, ray.vis, ray.tint);
      }
    }
    return;
  }
  let seed = sun_cell_seed(a, face);
  let p = sun_cell_origin(g, a, face, n);
  let ok = irc_sun_begin(a, face);
  var base = 0u;
  // code 变化首帧补两倍方向；同一格的其他采样线程会继续填满空槽。
  var count = SUN_SPP * 2u;
  if (ok) {
    base = irc_sun_next(a, face);
    count = SUN_SPP;
  }
  for (var i = 0u; i < count; i = i + 1u) {
    let slot = (base + i) & (SUN_SLOTS - 1u);
    let ray = sun_vis_dir(p, sun_sample_dir(seed, slot), voxel, obj_id, a.level);
    irc_sun_sample(a, face, slot, ray.vis, ray.tint);
  }
}
