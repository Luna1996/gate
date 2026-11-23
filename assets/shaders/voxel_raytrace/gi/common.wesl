import package::bindings::{gi_u, grid_descs, obj_pose_rev};
import package::common::{CHUNK_SIZE};
import package::gi::consts::{GI_AMBIENT_FLOOR, GI_AMBIENT_OPEN, GI_SPP_BOOST_N0, GI_SPP_BOOST_N1, GI_SPP_EASE_N, GI_SPP_MAX};
import package::volumetric::{sky_ambient};

const GI_KEY_LIMIT: i32 = 32768;
const GI_KEY_ORG_Q: u32 = 8192u;

fn gi_key_org(obj: i32) -> vec3<i32> {
  let d = grid_descs[u32(obj + 1)];
  let o = vec3<i32>(d.index_origin_x, d.index_origin_y, d.index_origin_z)
    * vec3<i32>(i32(CHUNK_SIZE));
  let q = vec3<i32>(i32(GI_KEY_ORG_Q));
  return (o / q) * q;
}

fn gi_pose_salt(obj: i32) -> u32 {
  if (obj < 0) { return 0u; }
  let i = u32(obj + 1);
  if (i >= arrayLength(&obj_pose_rev)) { return 0u; }
  return gi_mix(obj_pose_rev[i] * 0x9E3779B1u + 0x85EBCA77u) & 0xFFFFu;
}

fn gi_key_ok(v_world: vec3<i32>, obj: i32) -> bool {
  let v = v_world - gi_key_org(obj);
  return all(v > vec3<i32>(-GI_KEY_LIMIT)) && all(v < vec3<i32>(GI_KEY_LIMIT));
}

fn gi_key0(v_world: vec3<i32>, obj: i32) -> u32 {
  let v = v_world - gi_key_org(obj);
  return ((u32(v.x + 32768) & 0xFFFFu) | ((u32(v.y + 32768) & 0xFFFFu) << 16u))
    ^ gi_pose_salt(obj);
}

fn gi_key1(v_world: vec3<i32>, face: u32, obj: i32) -> u32 {
  let v = v_world - gi_key_org(obj);
  return ((u32(v.z + 32768) & 0xFFFFu)
    | ((face & 7u) << 16u)
    | ((u32(obj + 1) & 0x1FFFu) << 19u))
    ^ gi_pose_salt(obj);
}

fn gi_unkey_voxel(k0: u32, k1: u32, obj: i32) -> vec3<i32> {
  let s = gi_pose_salt(obj);
  return vec3<i32>(i32((k0 & 0xFFFFu) ^ s) - 32768,
                   i32(((k0 >> 16u) & 0xFFFFu) ^ s) - 32768,
                   i32((k1 & 0xFFFFu) ^ s) - 32768) + gi_key_org(obj);
}

fn gi_mix(x: u32) -> u32 {
  var h = x * 747796405u + 2891336453u;
  h = ((h >> ((h >> 28u) + 4u)) + h) * 277803737u;
  return (h >> 22u) + h;
}

fn gi_rand2(seed: u32) -> vec2<f32> {
  let a = gi_mix(seed);
  let b = gi_mix(a + 0x9E3779B9u);
  return vec2<f32>(f32(a), f32(b)) * (1.0 / 4294967296.0);
}

fn gi_cos_hemisphere(n: vec3<f32>, u: vec2<f32>) -> vec3<f32> {
  let r = sqrt(u.x);
  let phi = 6.2831853 * u.y;
  let axis = select(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 0.0), abs(n.z) > 0.9);
  let t = normalize(cross(axis, n));
  let b = cross(n, t);
  return normalize(t * (r * cos(phi)) + b * (r * sin(phi)) + n * sqrt(max(1.0 - u.x, 0.0)));
}

fn gi_enabled() -> bool {
  return gi_u.misc.x > 0.5;
}

fn gi_emit_gain() -> f32 {
  return gi_u.params.x;
}

fn gi_div() -> u32 {
  return max(u32(gi_u.flags.y), 1u);
}

fn gi_jitter(frame: u32, d: u32) -> vec2<u32> {
  if (d <= 1u) { return vec2<u32>(0u); }
  let bits = 2u * firstTrailingBit(d);
  let i = reverseBits(frame % (d * d)) >> (32u - bits);
  var j = vec2<u32>(0u);
  for (var b = 0u; b < 4u; b = b + 1u) {
    j.x = j.x | (((i >> (2u * b)) & 1u) << b);
    j.y = j.y | (((i >> (2u * b + 1u)) & 1u) << b);
  }
  return j & vec2<u32>(d - 1u);
}

fn gi_record_pixel(texel: vec2<u32>) -> vec2<u32> {
  let d = gi_div();
  return texel * d + gi_jitter(gi_u.seq.x, d);
}

// 缓存完全没有条目时的天光兜底：按法线取天空环境，覆盖率上来后退回 GI 缓存值。
// 用接近收敛量级的天光而不是一个极小的底值，避免新露出表面先闪黑再慢慢变亮。
fn gi_ambient_fallback(n: vec3<f32>, cov: f32) -> vec3<f32> {
  let hemi = 0.5 + 0.5 * max(n.y, 0.0);
  let gap = 1.0 - clamp(cov, 0.0, 1.0);
  let open = GI_AMBIENT_OPEN * hemi * gap * gap * gap * gap;
  return sky_ambient(n) * (open + GI_AMBIENT_FLOOR);
}

// 一个 GI texel 每帧的样本数：先补足「每帧至少一次落到缓存单元」，
// 未成熟的单元再按 n 加码，n 上来后回落，避免长期付高采样成本。
fn gi_sample_base() -> u32 {
  let per_cell = f32(gi_div()) / max(gi_u.params.y, 1.0);
  return u32(clamp(ceil(per_cell * per_cell), 1.0, f32(GI_SPP_MAX)));
}

fn gi_sample_boost(n: f32) -> u32 {
  if (n < GI_SPP_BOOST_N0) { return 8u; }
  if (n < GI_SPP_BOOST_N1) { return 4u; }
  if (n < GI_SPP_EASE_N) { return 2u; }
  return 4u;
}
