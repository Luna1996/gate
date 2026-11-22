import package::bindings::{gi_u, light_cache, light_cache_meta, light_u};
import package::gi::common::{gi_cos_hemisphere, gi_mix, gi_rand2};
import package::gi::consts::{GI_LUMA};
import package::gi::ray::{sun_transmittance};
import package::world::{world_trace_medium_scene};

const LC_WAYS: u32 = 8u;
const LC_WORDS: u32 = 16u;
const LC_KEY0: u32 = 0u;
const LC_KEY1: u32 = 1u;
const LC_STAMP: u32 = 2u;
const LC_REV: u32 = 3u;
const LC_SUN_N: u32 = 4u;
const LC_SUN: u32 = 5u;
const LC_AO_N: u32 = 8u;
const LC_AO: u32 = 9u;
const LC_EMIT_N: u32 = 10u;
const LC_EMIT: u32 = 11u;
const LC_ACCESS: u32 = 14u;

const LC_MAX_N: u32 = 64u;
const LC_SCALE: f32 = 1024.0;
const LC_BUDGET: u32 = 262144u;
const LC_HIT_BUDGET: u32 = 524288u;
const LC_AO_T: f32 = 12.0;
const LC_SUN_SAMPLES: u32 = 4u;
const LC_AO_SAMPLES: u32 = 4u;
const LC_NONE: u32 = 0xFFFFFFFFu;
const LC_BUSY: u32 = 0x80000000u;
const LC_FRAME_MASK: u32 = 0x7FFFFFFFu;

fn lc_rev() -> u32 {
  let sd = light_u.lights[0].kind_pos_dir.xyz;
  let q = vec3<i32>(round(sd * 128.0));
  let r = i32(round(light_u.lights[0].shape.y * 1024.0));
  let sun = u32(q.x) ^ (u32(q.y) << 1u) ^ (u32(q.z) << 2u) ^ (u32(r) << 3u);
  let content = u32(gi_u.params.w) ^ u32(gi_u.params.x);
  return content ^ gi_u.seq.z ^ gi_mix(sun);
}

fn lc_sets() -> u32 {
  return max(arrayLength(&light_cache) / (LC_WAYS * LC_WORDS), 1u);
}

fn lc_base(k0: u32, k1: u32) -> u32 {
  let h = gi_mix(k0 + gi_mix(k1 * 0x9E3779B1u));
  return (h & (lc_sets() - 1u)) * (LC_WAYS * LC_WORDS);
}

fn lc_stamp(off: u32) -> u32 {
  return atomicLoad(&light_cache[off + LC_STAMP]) & LC_FRAME_MASK;
}

fn lc_frame() -> u32 {
  return gi_u.seq.x & LC_FRAME_MASK;
}

fn lc_find(k0: u32, k1: u32) -> u32 {
  let base = lc_base(k0, k1);
  let rev = lc_rev();
  for (var w: u32 = 0u; w < LC_WAYS; w = w + 1u) {
    let off = base + w * LC_WORDS;
    if (atomicLoad(&light_cache[off + LC_STAMP]) == 0u) { continue; }
    if (atomicLoad(&light_cache[off + LC_KEY0]) != k0
        || atomicLoad(&light_cache[off + LC_KEY1]) != k1) { continue; }
    if (atomicLoad(&light_cache[off + LC_REV]) != rev) { return LC_NONE; }
    return off;
  }
  return LC_NONE;
}

fn lc_claim(k0: u32, k1: u32) -> u32 {
  let base = lc_base(k0, k1);
  let rev = lc_rev();
  let frame = lc_frame();
  let stamp_new = LC_BUSY | frame;
  var empty = LC_NONE;
  var oldest = LC_NONE;
  var oldest_stamp = 0xFFFFFFFFu;
  for (var w: u32 = 0u; w < LC_WAYS; w = w + 1u) {
    let off = base + w * LC_WORDS;
    let st = atomicLoad(&light_cache[off + LC_STAMP]);
    if (st == 0u) {
      if (empty == LC_NONE) { empty = off; }
      continue;
    }
    if ((st & LC_BUSY) != 0u) { continue; }
    if (atomicLoad(&light_cache[off + LC_KEY0]) == k0
        && atomicLoad(&light_cache[off + LC_KEY1]) == k1) {
      if (atomicLoad(&light_cache[off + LC_REV]) != rev) {
        atomicStore(&light_cache[off + LC_REV], rev);
        atomicStore(&light_cache[off + LC_SUN_N], 0u);
        atomicStore(&light_cache[off + LC_SUN], 0u);
        atomicStore(&light_cache[off + LC_SUN + 1u], 0u);
        atomicStore(&light_cache[off + LC_SUN + 2u], 0u);
        atomicStore(&light_cache[off + LC_AO_N], 0u);
        atomicStore(&light_cache[off + LC_AO], 0u);
        atomicStore(&light_cache[off + LC_EMIT_N], 0u);
        atomicStore(&light_cache[off + LC_EMIT], 0u);
        atomicStore(&light_cache[off + LC_EMIT + 1u], 0u);
        atomicStore(&light_cache[off + LC_EMIT + 2u], 0u);
      } else if ((st & LC_FRAME_MASK) == frame) {
        return LC_NONE;
      }
      let claim = atomicCompareExchangeWeak(&light_cache[off + LC_STAMP], st, stamp_new);
      if (claim.exchanged) {
        atomicStore(&light_cache[off + LC_ACCESS], frame);
        return off;
      }
      return LC_NONE;
    }
    let acc = max(st & LC_FRAME_MASK, atomicLoad(&light_cache[off + LC_ACCESS]));
    if (acc < oldest_stamp) {
      oldest_stamp = acc;
      oldest = off;
    }
  }
  let off = select(oldest, empty, empty != LC_NONE);
  if (off == LC_NONE) { return LC_NONE; }
  let st = atomicLoad(&light_cache[off + LC_STAMP]);
  let claim = atomicCompareExchangeWeak(&light_cache[off + LC_STAMP], st, stamp_new);
  if (!claim.exchanged) { return LC_NONE; }
  atomicStore(&light_cache[off + LC_KEY0], k0);
  atomicStore(&light_cache[off + LC_KEY1], k1);
  atomicStore(&light_cache[off + LC_REV], rev);
  atomicStore(&light_cache[off + LC_ACCESS], frame);
  atomicStore(&light_cache[off + LC_SUN_N], 0u);
  atomicStore(&light_cache[off + LC_SUN], 0u);
  atomicStore(&light_cache[off + LC_SUN + 1u], 0u);
  atomicStore(&light_cache[off + LC_SUN + 2u], 0u);
  atomicStore(&light_cache[off + LC_AO_N], 0u);
  atomicStore(&light_cache[off + LC_AO], 0u);
  atomicStore(&light_cache[off + LC_EMIT_N], 0u);
  atomicStore(&light_cache[off + LC_EMIT], 0u);
  atomicStore(&light_cache[off + LC_EMIT + 1u], 0u);
  atomicStore(&light_cache[off + LC_EMIT + 2u], 0u);
  return off;
}

fn lc_finish(off: u32) {
  atomicStore(&light_cache[off + LC_STAMP], lc_frame());
}

fn lc_budget_ok() -> bool {
  return atomicLoad(&light_cache_meta[0]) < LC_BUDGET;
}

fn lc_budget_take() {
  atomicAdd(&light_cache_meta[0], 1u);
}

fn lc_hit_budget_ok() -> bool {
  return atomicLoad(&light_cache_meta[2]) < LC_HIT_BUDGET;
}

fn lc_hit_budget_take() {
  atomicAdd(&light_cache_meta[2], 1u);
}

fn lc_full(off: u32) -> bool {
  return lc_sun_n(off) >= LC_MAX_N && lc_ao_n(off) >= LC_MAX_N;
}

fn lc_emit_n(off: u32) -> u32 {
  return atomicLoad(&light_cache[off + LC_EMIT_N]);
}

fn lc_push_emit(off: u32, v: vec3<f32>) {
  let n = atomicLoad(&light_cache[off + LC_EMIT_N]);
  var sum = vec3<f32>(
    f32(atomicLoad(&light_cache[off + LC_EMIT])),
    f32(atomicLoad(&light_cache[off + LC_EMIT + 1u])),
    f32(atomicLoad(&light_cache[off + LC_EMIT + 2u])),
  );
  var vv = max(v, vec3<f32>(0.0));
  if (n > 0u) {
    let mean = sum / (f32(n) * LC_SCALE);
    let cap = dot(mean, GI_LUMA) * 4.0 + 0.02;
    let lum = dot(vv, GI_LUMA);
    if (lum > cap && lum > 0.0) { vv = vv * (cap / lum); }
  }
  let s = min(vv, vec3<f32>(4.0)) * LC_SCALE;
  if (n < LC_MAX_N) {
    sum = sum + s;
    atomicStore(&light_cache[off + LC_EMIT_N], n + 1u);
  } else {
    sum = sum - sum / f32(n) + s;
  }
  atomicStore(&light_cache[off + LC_EMIT], u32(sum.x));
  atomicStore(&light_cache[off + LC_EMIT + 1u], u32(sum.y));
  atomicStore(&light_cache[off + LC_EMIT + 2u], u32(sum.z));
}

fn lc_emit(off: u32) -> vec4<f32> {
  let n = atomicLoad(&light_cache[off + LC_EMIT_N]);
  if (n == 0u) { return vec4<f32>(0.0); }
  let inv = 1.0 / (f32(n) * LC_SCALE);
  return vec4<f32>(
    f32(atomicLoad(&light_cache[off + LC_EMIT])) * inv,
    f32(atomicLoad(&light_cache[off + LC_EMIT + 1u])) * inv,
    f32(atomicLoad(&light_cache[off + LC_EMIT + 2u])) * inv,
    1.0,
  );
}

fn lc_filter_claim(off: u32) -> bool {
  let frame = lc_frame();
  let st = atomicLoad(&light_cache[off + LC_AO_N]);
  if (st == frame) { return false; }
  let c = atomicCompareExchangeWeak(&light_cache[off + LC_AO_N], st, frame);
  if (c.exchanged) {
    atomicStore(&light_cache[off + LC_ACCESS], frame);
  }
  return c.exchanged;
}

fn lc_filtered_store(off: u32, v: vec3<f32>, n: u32) {
  let s = clamp(v, vec3<f32>(0.0), vec3<f32>(4.0)) * LC_SCALE;
  atomicStore(&light_cache[off + LC_SUN], u32(s.x));
  atomicStore(&light_cache[off + LC_SUN + 1u], u32(s.y));
  atomicStore(&light_cache[off + LC_SUN + 2u], u32(s.z));
  atomicStore(&light_cache[off + LC_SUN_N], n);
}

fn lc_filtered_n(off: u32) -> u32 {
  return atomicLoad(&light_cache[off + LC_SUN_N]);
}

fn lc_filtered(off: u32) -> vec4<f32> {
  let n = atomicLoad(&light_cache[off + LC_SUN_N]);
  if (n == 0u) { return vec4<f32>(0.0); }
  let inv = 1.0 / LC_SCALE;
  return vec4<f32>(
    f32(atomicLoad(&light_cache[off + LC_SUN])) * inv,
    f32(atomicLoad(&light_cache[off + LC_SUN + 1u])) * inv,
    f32(atomicLoad(&light_cache[off + LC_SUN + 2u])) * inv,
    clamp(f32(n) / 4.0, 0.25, 1.0),
  );
}

fn lc_push_sun(off: u32, v: vec3<f32>) {
  let n = atomicLoad(&light_cache[off + LC_SUN_N]);
  var sum = vec3<f32>(
    f32(atomicLoad(&light_cache[off + LC_SUN])),
    f32(atomicLoad(&light_cache[off + LC_SUN + 1u])),
    f32(atomicLoad(&light_cache[off + LC_SUN + 2u])),
  );
  let s = clamp(v, vec3<f32>(0.0), vec3<f32>(1.0)) * LC_SCALE;
  if (n < LC_MAX_N) {
    sum = sum + s;
    atomicStore(&light_cache[off + LC_SUN_N], n + 1u);
  } else {
    sum = sum - sum / f32(n) + s;
  }
  atomicStore(&light_cache[off + LC_SUN], u32(sum.x));
  atomicStore(&light_cache[off + LC_SUN + 1u], u32(sum.y));
  atomicStore(&light_cache[off + LC_SUN + 2u], u32(sum.z));
}

fn lc_push_ao(off: u32, v: f32) {
  let n = atomicLoad(&light_cache[off + LC_AO_N]);
  var sum = f32(atomicLoad(&light_cache[off + LC_AO]));
  let s = clamp(v, 0.0, 1.0) * LC_SCALE;
  if (n < LC_MAX_N) {
    sum = sum + s;
    atomicStore(&light_cache[off + LC_AO_N], n + 1u);
  } else {
    sum = sum - sum / f32(n) + s;
  }
  atomicStore(&light_cache[off + LC_AO], u32(sum));
}

fn lc_sun_n(off: u32) -> u32 {
  return atomicLoad(&light_cache[off + LC_SUN_N]);
}

fn lc_ao_n(off: u32) -> u32 {
  return atomicLoad(&light_cache[off + LC_AO_N]);
}

fn lc_sun(off: u32) -> vec3<f32> {
  let n = atomicLoad(&light_cache[off + LC_SUN_N]);
  if (n == 0u) { return vec3<f32>(0.0); }
  let inv = 1.0 / (f32(n) * LC_SCALE);
  return vec3<f32>(
    f32(atomicLoad(&light_cache[off + LC_SUN])),
    f32(atomicLoad(&light_cache[off + LC_SUN + 1u])),
    f32(atomicLoad(&light_cache[off + LC_SUN + 2u])),
  ) * inv;
}

fn lc_ao(off: u32) -> f32 {
  let n = atomicLoad(&light_cache[off + LC_AO_N]);
  if (n == 0u) { return 1.0; }
  return f32(atomicLoad(&light_cache[off + LC_AO])) / (f32(n) * LC_SCALE);
}

fn sun_cone_dir(sun_dir: vec3<f32>, radius: f32, u: vec2<f32>) -> vec3<f32> {
  if (radius <= 0.0) { return sun_dir; }
  let cos_r = cos(clamp(radius, 0.0, 1.5));
  let cos_t = 1.0 - u.x * (1.0 - cos_r);
  let sin_t = sqrt(max(1.0 - cos_t * cos_t, 0.0));
  let phi = 6.28318530718 * u.y;
  let up = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(sun_dir.y) > 0.99);
  let t = normalize(cross(up, sun_dir));
  let b = cross(sun_dir, t);
  return normalize(t * (sin_t * cos(phi)) + b * (sin_t * sin(phi)) + sun_dir * cos_t);
}

fn sun_vis_samples(origin: vec3<f32>, sun_dir: vec3<f32>, radius: f32,
                   self_voxel: vec3<i32>, self_obj: i32, seed: u32) -> vec3<f32> {
  if (radius <= 0.0) {
    return sun_transmittance(origin, sun_dir, self_voxel, self_obj);
  }
  var sum = vec3<f32>(0.0);
  for (var i: u32 = 0u; i < LC_SUN_SAMPLES; i = i + 1u) {
    let u = gi_rand2(gi_mix(seed + i * 0x9E3779B1u));
    sum = sum + sun_transmittance(origin, sun_cone_dir(sun_dir, radius, u), self_voxel, self_obj);
  }
  return sum / f32(LC_SUN_SAMPLES);
}

fn ao_one(origin: vec3<f32>, n: vec3<f32>, u: vec2<f32>) -> f32 {
  let dir = gi_cos_hemisphere(n, u);
  var tint = vec3<f32>(1.0);
  let h = world_trace_medium_scene(origin, dir, LC_AO_T, false, false, &tint);
  if (!h.hit) { return 1.0; }
  let occ = clamp(1.0 - h.t / LC_AO_T, 0.0, 1.0);
  return 1.0 - occ * clamp(dot(tint, GI_LUMA), 0.0, 1.0);
}

fn ao_samples(origin: vec3<f32>, n: vec3<f32>, seed: u32) -> f32 {
  var sum = 0.0;
  for (var i: u32 = 0u; i < LC_AO_SAMPLES; i = i + 1u) {
    let u = gi_rand2(gi_mix(seed + i * 0x85EBCA77u));
    sum = sum + ao_one(origin, n, u);
  }
  return sum / f32(LC_AO_SAMPLES);
}
