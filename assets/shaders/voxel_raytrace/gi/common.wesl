import package::bindings::{face_slots, face_slots_flat, gi_region_rev, gi_sec_slots};

import package::bindings::{grid_descs};
import package::common::{CHUNK_SIZE};

const GI_KEY_LIMIT: i32 = 32768;
const GI_KEY_ORG_Q: u32 = 8192u;

fn gi_key_org(obj: i32) -> vec3<i32> {
  let d = grid_descs[u32(obj + 1)];
  let o = vec3<i32>(d.index_origin_x, d.index_origin_y, d.index_origin_z)
    * vec3<i32>(i32(CHUNK_SIZE));
  let q = vec3<i32>(i32(GI_KEY_ORG_Q));
  return (o / q) * q;
}

fn gi_key_ok(v_world: vec3<i32>, obj: i32) -> bool {
  let v = v_world - gi_key_org(obj);
  return all(v > vec3<i32>(-GI_KEY_LIMIT)) && all(v < vec3<i32>(GI_KEY_LIMIT));
}

fn gi_key0(v_world: vec3<i32>, obj: i32) -> u32 {
  let v = v_world - gi_key_org(obj);
  return (u32(v.x + 32768) & 0xFFFFu) | ((u32(v.y + 32768) & 0xFFFFu) << 16u);
}

fn gi_key1(v_world: vec3<i32>, face: u32, obj: i32) -> u32 {
  let v = v_world - gi_key_org(obj);
  return (u32(v.z + 32768) & 0xFFFFu)
    | ((face & 7u) << 16u)
    | ((u32(obj + 1) & 0x1FFFu) << 19u);
}

fn gi_unkey_voxel(k0: u32, k1: u32, obj: i32) -> vec3<i32> {
  return vec3<i32>(i32(k0 & 0xFFFFu) - 32768,
                   i32((k0 >> 16u) & 0xFFFFu) - 32768,
                   i32(k1 & 0xFFFFu) - 32768) + gi_key_org(obj);
}

const GI_REGION_CHUNKS: u32 = 8u;
const GI_REGION_TABLE: u32 = 16u;

fn gi_local_rev(v_world: vec3<i32>, obj: i32) -> u32 {
  if (obj != -1) { return 0u; }
  if (arrayLength(&gi_region_rev) < GI_REGION_TABLE * GI_REGION_TABLE * GI_REGION_TABLE) {
    return 0u;
  }
  let rel = max(v_world - gi_key_org(obj), vec3<i32>(0));
  let chunk = vec3<u32>(rel) / vec3<u32>(CHUNK_SIZE);
  let cell = min(chunk / vec3<u32>(GI_REGION_CHUNKS), vec3<u32>(GI_REGION_TABLE - 1u));
  return gi_region_rev[cell.x + GI_REGION_TABLE * (cell.y + GI_REGION_TABLE * cell.z)];
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

fn gi_pack_dir(n: vec3<f32>) -> u32 {
  let l = abs(n.x) + abs(n.y) + abs(n.z);
  var o = n.xy / max(l, 1e-6);
  if (n.z < 0.0) {
    o = (vec2<f32>(1.0) - abs(o.yx)) * select(vec2<f32>(-1.0), vec2<f32>(1.0), o >= vec2<f32>(0.0));
  }
  return pack2x16float(o);
}

fn gi_unpack_dir(p: u32) -> vec3<f32> {
  let o = unpack2x16float(p);
  var n = vec3<f32>(o.x, o.y, 1.0 - abs(o.x) - abs(o.y));
  if (n.z < 0.0) {
    n = vec3<f32>(
      (1.0 - abs(n.y)) * select(-1.0, 1.0, n.x >= 0.0),
      (1.0 - abs(n.x)) * select(-1.0, 1.0, n.y >= 0.0),
      n.z,
    );
  }
  return normalize(n);
}

fn gi_cos_hemisphere(n: vec3<f32>, u: vec2<f32>) -> vec3<f32> {
  let r = sqrt(u.x);
  let phi = 6.2831853 * u.y;
  let axis = select(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 0.0), abs(n.z) > 0.9);
  let t = normalize(cross(axis, n));
  let b = cross(n, t);
  return normalize(t * (r * cos(phi)) + b * (r * sin(phi)) + n * sqrt(max(1.0 - u.x, 0.0)));
}

const FACE_WORDS: u32 = 16u;
const FACE_W_FLAG: u32 = 0u;
const FACE_W_K0: u32 = 1u;
const FACE_W_K1: u32 = 2u;
const FACE_W_TEXEL: u32 = 3u;
const FACE_W_PAL: u32 = 4u;
const FACE_W_COL: u32 = 5u;
const FACE_W_GI_N: u32 = 8u;
const FACE_W_GI: u32 = 9u;
const FACE_W_RAW_N: u32 = 12u;
const FACE_W_RAW: u32 = 13u;
const FACE_W_SUN: u32 = 12u;
const GI_ACC_SCALE: f32 = 4096.0;
const GI_ACC_CLAMP: f32 = 255.0;
const GI_ACC_MAX_N: u32 = 1024u;

fn face_slot_count() -> u32 {
  return arrayLength(&face_slots) / FACE_WORDS;
}

fn face_slot_of(k0: u32, k1: u32, ok: ptr<function, bool>) -> u32 {
  let n = face_slot_count();
  *ok = n >= 1u;
  if (!*ok) { return 0u; }
  return gi_mix(k0 + gi_mix(k1 * 0x9E3779B1u)) & (n - 1u);
}

fn face_slot_next(base: u32) -> u32 {
  return (base + 1u) & (face_slot_count() - 1u);
}

fn face_slot_matches(slot: u32, k0: u32, k1: u32) -> bool {
  let fe = slot * FACE_WORDS;
  let flag = atomicLoad(&face_slots[fe + FACE_W_FLAG]);
  let s0 = atomicLoad(&face_slots[fe + FACE_W_K0]);
  let s1 = atomicLoad(&face_slots[fe + FACE_W_K1]);
  return flag != 0u && s0 == k0 && s1 == k1;
}

fn face_slot_find(k0: u32, k1: u32, ok: ptr<function, bool>) -> u32 {
  var table_ok = false;
  let base = face_slot_of(k0, k1, &table_ok);
  *ok = false;
  if (!table_ok) { return 0u; }
  if (face_slot_matches(base, k0, k1)) {
    *ok = true;
    return base;
  }
  let second = face_slot_next(base);
  if (face_slot_matches(second, k0, k1)) {
    *ok = true;
    return second;
  }
  return 0u;
}

fn face_slot_count_flat() -> u32 {
  return arrayLength(&face_slots_flat) / FACE_WORDS;
}

fn face_slot_next_flat(base: u32) -> u32 {
  return (base + 1u) & (face_slot_count_flat() - 1u);
}

fn face_slot_matches_flat(slot: u32, k0: u32, k1: u32) -> bool {
  let fe = slot * FACE_WORDS;
  let flag = atomicLoad(&face_slots_flat[fe + FACE_W_FLAG]);
  let s0 = atomicLoad(&face_slots_flat[fe + FACE_W_K0]);
  let s1 = atomicLoad(&face_slots_flat[fe + FACE_W_K1]);
  return flag != 0u && s0 == k0 && s1 == k1;
}

fn face_slot_find_flat(k0: u32, k1: u32, ok: ptr<function, bool>) -> u32 {
  var table_ok = false;
  let base = face_slot_of_flat(k0, k1, &table_ok);
  *ok = false;
  if (!table_ok) { return 0u; }
  if (face_slot_matches_flat(base, k0, k1)) {
    *ok = true;
    return base;
  }
  let second = face_slot_next_flat(base);
  if (face_slot_matches_flat(second, k0, k1)) {
    *ok = true;
    return second;
  }
  return 0u;
}

fn face_slot_color(k0: u32, k1: u32, pal_out: ptr<function, u32>, sun_out: ptr<function, u32>,
                   ok: ptr<function, bool>) -> vec3<f32> {
  *pal_out = 0u;
  *sun_out = 0u;
  let slot = face_slot_find(k0, k1, ok);
  if (!*ok) { return vec3<f32>(0.0); }
  let fe = slot * FACE_WORDS;
  *pal_out = atomicLoad(&face_slots[fe + FACE_W_PAL]);
  *sun_out = atomicLoad(&face_slots[fe + FACE_W_SUN]);
  let c0 = atomicLoad(&face_slots[fe + FACE_W_COL]);
  let c1 = atomicLoad(&face_slots[fe + FACE_W_COL + 1u]);
  let c2 = atomicLoad(&face_slots[fe + FACE_W_COL + 2u]);
  return vec3<f32>(bitcast<f32>(c0), bitcast<f32>(c1), bitcast<f32>(c2));
}

const GI_SEC_WORDS: u32 = 5u;
const GI_SEC_W_K0: u32 = 0u;
const GI_SEC_W_K1: u32 = 1u;
const GI_SEC_W_RAD: u32 = 2u;

fn face_slot_of_flat(k0: u32, k1: u32, ok: ptr<function, bool>) -> u32 {
  let n = arrayLength(&face_slots_flat) / FACE_WORDS;
  *ok = n >= 1u;
  if (!*ok) { return 0u; }
  return gi_mix(k0 + gi_mix(k1 * 0x9E3779B1u)) & (n - 1u);
}

fn gi_sec_slot_of(k0: u32, k1: u32, ok: ptr<function, bool>) -> u32 {
  let n = arrayLength(&gi_sec_slots) / GI_SEC_WORDS;
  *ok = n >= 1u;
  if (!*ok) { return 0u; }
  return gi_mix(k0 + gi_mix(k1 * 0x9E3779B1u)) & (n - 1u);
}

fn gi_sec_key_masked(k0: u32, k1: u32, epoch: u32, pal: u32, rev: u32) -> vec2<u32> {
  return vec2<u32>(
    k0 + gi_mix(epoch * 0x9E3779B1u + rev * 0x85EBCA77u + 0x165667B1u),
    k1 + gi_mix(gi_mix(epoch + 0x85EBCA77u) + rev * 0x27D4EB2Fu + 0x27D4EB2Fu
                + pal * 0x9E3779B1u),
  );
}

fn gi_ring_offset(s: u32) -> vec2<i32> {
  if (s == 0u) { return vec2<i32>(1, 0); }
  if (s == 1u) { return vec2<i32>(-1, 0); }
  if (s == 2u) { return vec2<i32>(0, 1); }
  if (s == 3u) { return vec2<i32>(0, -1); }
  if (s == 4u) { return vec2<i32>(1, 1); }
  if (s == 5u) { return vec2<i32>(1, -1); }
  if (s == 6u) { return vec2<i32>(-1, 1); }
  return vec2<i32>(-1, -1);
}

fn gi_hist_search_offset(s: u32) -> vec2<i32> {
  if (s == 0u) { return vec2<i32>(0, 0); }
  return gi_ring_offset(s - 1u);
}