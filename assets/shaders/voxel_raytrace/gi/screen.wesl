import package::bindings::{gi_guide_out, gi_out, gi_res_cur, gi_res_prev, gi_u, lod_diag, light_u, view_u};
import package::common::{FLAGS_TRANSMISSIVE, SurfaceLight, brdf_reflected, face_of_normal, face_view_dir, fetch_material, palette_flags, sky_rgb};
import package::gi::common::{GI_ACC_CLAMP, gi_cos_hemisphere, gi_hist_search_offset, gi_in_edit_box, gi_key0, gi_key1, gi_key_ok, gi_local_rev, gi_mix, gi_pack_dir, gi_pose_salt, gi_rand2, gi_ring_offset, gi_unpack_dir, gi_unkey_voxel, gi_vox_k1};
import package::gi::consts::{GI_DEN_GUIDE_WORDS, GI_DEN_G_NOISE, GI_DEN_N_DOT, GI_EMIT_GAIN, GI_LUMA, GI_PI, GI_SKY_AMBIENT, GI_SKY_RADIANCE_SCALE, GI_SS_CAND_CLAMP_CEIL, GI_SS_CAND_CLAMP_K, GI_SS_CAND_N, GI_SS_CAND_N_HQ, GI_SS_M_CAP_K, GI_SS_REUSE_TAPS, SHADOW_SURFACE_EPS};
import package::gi::denoise::{gi_den_guide_store, gi_same_surface};
import package::gi::ray::{gi_bounce2, gi_face_shade_cached, gi_ray_radiance};
import package::gi::resample::{GI_KEY_FLAG_DIFFUSE, GI_KEY_FLAG_HIT, GI_KEY_FLAG_OPAQUE, gi_res_store_miss, gi_res_store_primary};
import package::gi::wal::gi_wal_merge;
import package::trace::make_grid;
import package::world::{WorldHit, surface_offset, voxel_center_world, voxel_normal_fast, world_trace_medium_scene};

const GI_RES_WORDS: u32 = 21u;
const GI_RES_W_FLAGS: u32 = 0u;
const GI_RES_W_PK: u32 = 1u;
const GI_RES_W_NRM: u32 = 3u;
const GI_RES_W_WSUM: u32 = 4u;
const GI_RES_W_M: u32 = 7u;
const GI_RES_W_SEC: u32 = 8u;
const GI_RES_W_SECN: u32 = 11u;
const GI_RES_W_DIST: u32 = 12u;
const GI_RES_W_HASH: u32 = 13u;
const GI_RES_W_TOKEN: u32 = 14u;
const GI_RES_W_SK0: u32 = 15u;
const GI_RES_W_SK1: u32 = 16u;
const GI_RES_W_PTOKEN: u32 = 17u;
const GI_RES_W_P: u32 = 18u;

const GI_RES_VALID: u32 = 1u;
const GI_RES_PK_OK: u32 = 2u;
const GI_RES_OWNED: u32 = 4u;
const GI_RES_SKY: u32 = 8u;

const GI_RECON_J_MIN: f32 = 0.3;
const GI_RECON_J_MAX: f32 = 2.5;

// The GI resampler owns the diffuse GI now: the old per-pixel reservoir only produces the
// G-buffer guide and the surface/key targets.
const GI_RES_FINAL: bool = true;

struct GiRes {
  w_sum: vec3<f32>,
  m: f32,
  pk0: u32,
  pk1: u32,
  flags: u32,
  nrm: vec3<f32>,
  sec: vec3<f32>,
  sec_n: vec3<f32>,
  dist: f32,
  hash: u32,
  token: u32,
  sk0: u32,
  sk1: u32,
  ptoken: u32,
  p: vec3<f32>,
}

struct GiSample {
  value: vec3<f32>,
  sec: vec3<f32>,
  sec_n: vec3<f32>,
  dist: f32,
  hash: u32,
  token: u32,
  sk0: u32,
  sk1: u32,
  sky: bool,
}

struct GiCand {
  value: vec3<f32>,
  sec: vec3<f32>,
  sec_n: vec3<f32>,
  dist: f32,
  hash: u32,
  token: u32,
  sk0: u32,
  sk1: u32,
  sky: bool,
  weight: f32,
  valid: bool,
}

fn gi_res_none() -> GiRes {
  return GiRes(vec3<f32>(0.0), 0.0, 0u, 0u, 0u, vec3<f32>(0.0, 0.0, 1.0),
               vec3<f32>(0.0), vec3<f32>(0.0, 0.0, 1.0), 0.0, 0u, 0u, 0u, 0u, 0u, vec3<f32>(0.0));
}

fn gi_res_load_cur(idx: u32) -> GiRes {
  let b = idx * GI_RES_WORDS;
  return GiRes(
    vec3<f32>(
      bitcast<f32>(gi_res_cur[b + GI_RES_W_WSUM]),
      bitcast<f32>(gi_res_cur[b + GI_RES_W_WSUM + 1u]),
      bitcast<f32>(gi_res_cur[b + GI_RES_W_WSUM + 2u]),
    ),
    bitcast<f32>(gi_res_cur[b + GI_RES_W_M]),
    gi_res_cur[b + GI_RES_W_PK],
    gi_res_cur[b + GI_RES_W_PK + 1u],
    gi_res_cur[b + GI_RES_W_FLAGS],
    gi_unpack_dir(gi_res_cur[b + GI_RES_W_NRM]),
    vec3<f32>(
      bitcast<f32>(gi_res_cur[b + GI_RES_W_SEC]),
      bitcast<f32>(gi_res_cur[b + GI_RES_W_SEC + 1u]),
      bitcast<f32>(gi_res_cur[b + GI_RES_W_SEC + 2u]),
    ),
    gi_unpack_dir(gi_res_cur[b + GI_RES_W_SECN]),
    bitcast<f32>(gi_res_cur[b + GI_RES_W_DIST]),
    gi_res_cur[b + GI_RES_W_HASH],
    gi_res_cur[b + GI_RES_W_TOKEN],
    gi_res_cur[b + GI_RES_W_SK0],
    gi_res_cur[b + GI_RES_W_SK1],
    gi_res_cur[b + GI_RES_W_PTOKEN],
    vec3<f32>(
      bitcast<f32>(gi_res_cur[b + GI_RES_W_P]),
      bitcast<f32>(gi_res_cur[b + GI_RES_W_P + 1u]),
      bitcast<f32>(gi_res_cur[b + GI_RES_W_P + 2u]),
    ),
  );
}

fn gi_res_load_prev(idx: u32) -> GiRes {  let b = idx * GI_RES_WORDS;
  return GiRes(
    vec3<f32>(
      bitcast<f32>(gi_res_prev[b + GI_RES_W_WSUM]),
      bitcast<f32>(gi_res_prev[b + GI_RES_W_WSUM + 1u]),
      bitcast<f32>(gi_res_prev[b + GI_RES_W_WSUM + 2u]),
    ),
    bitcast<f32>(gi_res_prev[b + GI_RES_W_M]),
    gi_res_prev[b + GI_RES_W_PK],
    gi_res_prev[b + GI_RES_W_PK + 1u],
    gi_res_prev[b + GI_RES_W_FLAGS],
    gi_unpack_dir(gi_res_prev[b + GI_RES_W_NRM]),
    vec3<f32>(
      bitcast<f32>(gi_res_prev[b + GI_RES_W_SEC]),
      bitcast<f32>(gi_res_prev[b + GI_RES_W_SEC + 1u]),
      bitcast<f32>(gi_res_prev[b + GI_RES_W_SEC + 2u]),
    ),
    gi_unpack_dir(gi_res_prev[b + GI_RES_W_SECN]),
    bitcast<f32>(gi_res_prev[b + GI_RES_W_DIST]),
    gi_res_prev[b + GI_RES_W_HASH],
    gi_res_prev[b + GI_RES_W_TOKEN],
    gi_res_prev[b + GI_RES_W_SK0],
    gi_res_prev[b + GI_RES_W_SK1],
    gi_res_prev[b + GI_RES_W_PTOKEN],
    vec3<f32>(
      bitcast<f32>(gi_res_prev[b + GI_RES_W_P]),
      bitcast<f32>(gi_res_prev[b + GI_RES_W_P + 1u]),
      bitcast<f32>(gi_res_prev[b + GI_RES_W_P + 2u]),
    ),
  );
}

fn gi_tap_head(idx: u32) -> vec3<u32> {
  let b = idx * GI_RES_WORDS;
  return vec3<u32>(gi_res_prev[b + GI_RES_W_FLAGS], gi_res_prev[b + GI_RES_W_PK],
                   gi_res_prev[b + GI_RES_W_PK + 1u]);
}

fn gi_res_head_prev(idx: u32) -> vec4<u32> {
  let b = idx * GI_RES_WORDS;
  return vec4<u32>(gi_res_prev[b + GI_RES_W_FLAGS], gi_res_prev[b + GI_RES_W_PK],
                   gi_res_prev[b + GI_RES_W_PK + 1u], gi_res_prev[b + GI_RES_W_NRM]);
}

fn gi_res_store_cur(idx: u32, r: GiRes) {
  let b = idx * GI_RES_WORDS;
  gi_res_cur[b + GI_RES_W_FLAGS] = r.flags;
  gi_res_cur[b + GI_RES_W_PK] = r.pk0;
  gi_res_cur[b + GI_RES_W_PK + 1u] = r.pk1;
  gi_res_cur[b + GI_RES_W_NRM] = gi_pack_dir(r.nrm);
  gi_res_cur[b + GI_RES_W_WSUM] = bitcast<u32>(r.w_sum.x);
  gi_res_cur[b + GI_RES_W_WSUM + 1u] = bitcast<u32>(r.w_sum.y);
  gi_res_cur[b + GI_RES_W_WSUM + 2u] = bitcast<u32>(r.w_sum.z);
  gi_res_cur[b + GI_RES_W_M] = bitcast<u32>(r.m);
  gi_res_cur[b + GI_RES_W_SEC] = bitcast<u32>(r.sec.x);
  gi_res_cur[b + GI_RES_W_SEC + 1u] = bitcast<u32>(r.sec.y);
  gi_res_cur[b + GI_RES_W_SEC + 2u] = bitcast<u32>(r.sec.z);
  gi_res_cur[b + GI_RES_W_SECN] = gi_pack_dir(r.sec_n);
  gi_res_cur[b + GI_RES_W_DIST] = bitcast<u32>(r.dist);
  gi_res_cur[b + GI_RES_W_HASH] = r.hash;
  gi_res_cur[b + GI_RES_W_TOKEN] = r.token;
  gi_res_cur[b + GI_RES_W_SK0] = r.sk0;
  gi_res_cur[b + GI_RES_W_SK1] = r.sk1;
  gi_res_cur[b + GI_RES_W_PTOKEN] = r.ptoken;
  gi_res_cur[b + GI_RES_W_P] = bitcast<u32>(r.p.x);
  gi_res_cur[b + GI_RES_W_P + 1u] = bitcast<u32>(r.p.y);
  gi_res_cur[b + GI_RES_W_P + 2u] = bitcast<u32>(r.p.z);
}

fn gi_res_store_sample(idx: u32, r: GiRes, s: GiSample) {
  var o = r;
  o.sec = s.sec;
  o.sec_n = s.sec_n;
  o.dist = s.dist;
  o.hash = s.hash;
  o.token = s.token;
  o.sk0 = s.sk0;
  o.sk1 = s.sk1;
  gi_res_store_cur(idx, o);
}

fn gi_lum(c: vec3<f32>) -> f32 {
  return dot(max(c, vec3<f32>(0.0)), GI_LUMA);
}

fn gi_key_token(k0: u32, k1: u32) -> u32 {
  let obj = i32((k1 >> 19u) & 0x1FFFu) - 1;
  let voxel = gi_unkey_voxel(k0, k1, obj);
  return gi_pose_salt(obj) ^ gi_local_rev(voxel, obj);
}

fn gi_ss_seed(coord: vec2<i32>, frame: u32) -> u32 {
  let ph = gi_mix(u32(coord.x) * 0x9E3779B1u + gi_mix(u32(coord.y) * 0x85EBCA77u + 0x165667B1u));
  return gi_mix(ph + gi_mix(frame * 0x9E3779B9u + 0x27D4EB2Fu));
}

fn gi_ss_reproject(p_world: vec3<f32>, size: vec2<u32>, ok: ptr<function, bool>) -> vec2<i32> {
  let clip = gi_u.prev_view_proj * vec4<f32>(p_world, 1.0);
  if (clip.w <= 1e-6) {
    *ok = false;
    return vec2<i32>(0);
  }
  let ndc = clip.xyz / clip.w;
  if (any(ndc < vec3<f32>(-1.0)) || any(ndc > vec3<f32>(1.0))) {
    *ok = false;
    return vec2<i32>(0);
  }
  let c = vec2<i32>(i32(floor((ndc.x * 0.5 + 0.5) * f32(size.x))),
                    i32(floor((0.5 - ndc.y * 0.5) * f32(size.y))));
  if (c.x < 0 || c.y < 0 || c.x >= i32(size.x) || c.y >= i32(size.y)) {
    *ok = false;
    return vec2<i32>(0);
  }
  *ok = true;
  return c;
}

fn gi_sample_ray(org: vec3<f32>, dir: vec3<f32>, sun_dir: vec3<f32>, sun_c: vec3<f32>,
                 sun_bounce: f32, epoch: u32, b2: f32, seed: u32) -> GiSample {
  var tint = vec3<f32>(1.0);
  let h = world_trace_medium_scene(org, dir, 4096.0, false, false, &tint);
  if (!h.hit) {
    let v = sky_rgb() * GI_SKY_RADIANCE_SCALE * tint;
    return GiSample(v, org + dir * 4096.0, vec3<f32>(0.0, 1.0, 0.0), 4096.0, 0u, 0u, 0u, 0u, true);
  }
  let g = make_grid(u32(h.obj_id) + 1u);
  let hp = voxel_center_world(g, h.voxel);
  let hn = voxel_normal_fast(g, h.voxel, h.face_id, h.n);
  let k0 = gi_key0(h.voxel, h.obj_id);
  let k1 = gi_key1(h.voxel, h.face_id, h.obj_id);
  let key_ok = gi_key_ok(h.voxel, h.obj_id);
  var l = gi_face_shade_cached(h, org, sun_dir, sun_c, sun_bounce, epoch, k0, k1, key_ok);
  if (b2 > 0.0) {
    l = l + gi_bounce2(h, org, sun_dir, sun_c, sun_bounce, epoch, seed, b2);
  }
  let token = gi_pose_salt(h.obj_id) ^ gi_local_rev(h.voxel, h.obj_id);
  let hash = gi_mix(k0 ^ gi_mix(k1));
  return GiSample(l * tint, hp, hn, max(length(hp - org), 1e-3), hash, token, k0, k1, false);
}

fn gi_pose_salt_ss(obj: i32) -> u32 {
  if (obj < 0) { return 0u; }
  return 0u;
}

fn gi_reconnect(p: vec3<f32>, n: vec3<f32>, s: GiRes) -> GiCand {
  var c = GiCand(s.w_sum / max(s.m, 1e-6), s.sec, s.sec_n, s.dist, s.hash, s.token, s.sk0, s.sk1, false, 0.0, false);
  if ((s.flags & GI_RES_VALID) == 0u) { return c; }
  if (gi_key_token(s.pk0, s.pk1) != s.ptoken) { return c; }
  if ((s.flags & GI_RES_SKY) == 0u && gi_key_token(s.sk0, s.sk1) != s.token) { return c; }
  let seg = s.sec - p;
  let d2 = dot(seg, seg);
  if (d2 < 1e-6) { return c; }
  let inv = inverseSqrt(d2);
  let d = seg * inv;
  let cn = dot(n, d);
  let co = dot(s.sec_n, -d);
  if (cn <= 1e-4 || co <= 1e-4) { return c; }
  let j = (cn / co) * (s.dist * s.dist / d2);
  if (j < GI_RECON_J_MIN || j > GI_RECON_J_MAX) { return c; }
  c.weight = gi_lum(c.value) * cn * j;
  c.valid = c.weight > 1e-6;
  return c;
}

fn gi_visible(p: vec3<f32>, n: vec3<f32>, sec: vec3<f32>, dist: f32) -> bool {
  let seg = sec - p;
  let d2 = dot(seg, seg);
  if (d2 < 1e-6) { return false; }
  let d = seg * inverseSqrt(d2);
  let eps = 0.01;
  var tint = vec3<f32>(1.0);
  let h = world_trace_medium_scene(p + n * eps, d, max(sqrt(d2) - eps * 2.0, 1e-3),
                                   false, false, &tint);
  return !h.hit;
}

fn gi_cand_own(s: GiSample, n: vec3<f32>, dir: vec3<f32>) -> GiCand {
  let cn = max(dot(n, dir), 1e-4);
  let w = gi_lum(s.value) * cn + 1e-5;
  return GiCand(s.value, s.sec, s.sec_n, s.dist, s.hash, s.token, s.sk0, s.sk1, s.sky, w, true);
}

fn gi_ss_main(coord: vec2<i32>, size: vec2<u32>, h: WorldHit, w_adapt: f32) {
  let idx = u32(coord.y) * size.x + u32(coord.x);
  let sun_dir = light_u.lights[0].kind_pos_dir.yzw;
  let sun_c = light_u.lights[0].color_intensity.xyz * light_u.lights[0].color_intensity.w;
  let sun_bounce = gi_u.params.x;
  let epoch = gi_u.seq.y;
  let b2_mult = gi_u.misc.w;

  if (!h.hit) {
    textureStore(gi_out, coord, vec4<f32>(0.0));
    gi_res_store_miss(coord);
    gi_res_store_cur(idx, gi_res_none());
    gi_den_guide_store(coord, size, false, 0u, 0u, vec3<f32>(0.0, 0.0, 1.0), false, vec2<i32>(0), false);
    return;
  }

  let gg = make_grid(u32(h.obj_id) + 1u);
  let p_voxel = voxel_center_world(gg, h.voxel);
  var n = h.n;
  if ((palette_flags(h.palette_base, h.pal) & FLAGS_TRANSMISSIVE) == 0u) {
    n = voxel_normal_fast(gg, h.voxel, h.face_id, h.n);
  }
  let npack = gi_pack_dir(n);
  let n_off = surface_offset(gg, n, SHADOW_SURFACE_EPS);
  let org = p_voxel + n * n_off;

  let pk0 = gi_key0(h.voxel, h.obj_id);
  let pk1 = gi_key1(h.voxel, h.face_id, h.obj_id);
  let pk_ok = gi_key_ok(h.voxel, h.obj_id);

  // G-buffer for the GI resampler: real surface, camera distance, path hash, instance slot
  // (grid index + 1), lifetime token and the world normal.
  let hh = gi_mix(pk0 ^ gi_mix(pk1));
  let hash16 = max(1u, (hh ^ (hh >> 16u)) & 0xFFFFu);
  let slot = u32(h.obj_id + 1) & 0xFFFFu;
  let token = gi_pose_salt(h.obj_id) ^ gi_local_rev(h.voxel, h.obj_id);
  let opaque = select(0u, GI_KEY_FLAG_OPAQUE,
    (palette_flags(h.palette_base, h.pal) & FLAGS_TRANSMISSIVE) == 0u);
  gi_res_store_primary(coord, h.point, length(h.point - view_u.cam_pos_voxel.xyz),
    hash16 | (slot << 16u), token, GI_KEY_FLAG_HIT | GI_KEY_FLAG_DIFFUSE | opaque, n);

  let frame = gi_u.seq.x;
  let seed = gi_ss_seed(coord, frame);

  var prev_ok = false;
  let prev_pc = gi_ss_reproject(h.point, size, &prev_ok);

  let affected = gi_in_edit_box(p_voxel);
  let since = gi_u.seq.x - gi_u.seq.w;
  let hist_reset_now = affected && (gi_u.flags.z <= 0.5);
  let hist_reset_recent = since < 32u;

  gi_den_guide_store(coord, size, true, pk0, pk1, n, prev_ok, prev_pc, hist_reset_recent);

  if (GI_RES_FINAL) {
    textureStore(gi_out, coord, vec4<f32>(0.0));
    gi_res_store_cur(idx, gi_res_none());
    return;
  }

  var hist_ok = false;
  var hist = gi_res_none();
  if (pk_ok && prev_ok) {
    for (var s = 0u; s <= 8u; s = s + 1u) {
      let pc = prev_pc + gi_hist_search_offset(s);
      if (pc.x < 0 || pc.y < 0 || pc.x >= i32(size.x) || pc.y >= i32(size.y)) { continue; }
      let nidx = u32(pc.y) * size.x + u32(pc.x);
      let head = gi_res_head_prev(nidx);
      if ((head.x & (GI_RES_VALID | GI_RES_PK_OK)) != (GI_RES_VALID | GI_RES_PK_OK)) { continue; }
      if (!gi_same_surface(head.y, head.z, head.w, pk0, pk1, npack)) { continue; }
      hist = gi_res_load_prev(nidx);
      hist_ok = true;
      break;
    }
  }

  let hq = u32(gi_u.flags.w) >= 3u;
  let base_n = select(GI_SS_CAND_N, GI_SS_CAND_N_HQ, hq);
  let share = max(u32(gi_u.misc.y), 1u);
  let res_k = f32(max(u32(gi_u.flags.y), 1u)) * f32(max(u32(gi_u.flags.y), 1u)) * (1.0 / 16.0);
  let dither = gi_rand2(gi_mix(seed + 0x2545F491u)).x;
  let rate = f32(base_n) / f32(share) * res_k * w_adapt;
  var cand_n = max(1u, u32(floor(rate + dither)));
  if (rate < 0.999) {
    let stride = max(u32(ceil(1.0 / max(rate, 1.0 / 64.0))), 1u);
    if (gi_mix(idx ^ (gi_u.seq.x * 0x9E3779B9u)) % stride != 0u) { cand_n = 0u; }
  }
  cand_n = min(cand_n, 4u);
  if ((!hist_ok || hist_reset_recent) && pk_ok) { cand_n = max(cand_n, 4u); }

  var sum_w = 0.0;
  var cand_count = 0u;
  var chosen = GiCand(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0, 1.0, 0.0), 0.0,
                      0u, 0u, 0u, 0u, true, 0.0, false);
  var own_best = GiCand(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0, 1.0, 0.0), 0.0,
                        0u, 0u, 0u, 0u, true, 0.0, false);
  var chosen_owned = false;

  for (var k = 0u; k < cand_n; k = k + 1u) {
    let cseed = gi_mix(seed + k * 0x9E3779B1u + 1u);
    let dir = gi_cos_hemisphere(n, gi_rand2(cseed));
    let b2 = select(0.0, b2_mult, gi_rand2(gi_mix(cseed + 0x27D4EB2Fu)).y < 1.0 / max(b2_mult, 1.0));
    let s = gi_sample_ray(org, dir, sun_dir, sun_c, sun_bounce, epoch, b2, cseed);
    var c = gi_cand_own(s, n, dir);
    if (c.valid && (!own_best.valid || c.weight > own_best.weight)) { own_best = c; }
    sum_w = sum_w + c.weight;
    if (c.valid) { cand_count = cand_count + 1u; }
    if (c.valid && gi_rand2(gi_mix(cseed + 0x9E3779B9u)).x * sum_w < c.weight) {
      chosen = c;
      chosen_owned = true;
    }
  }

  if (hist_ok && !hist_reset_now) {
    let c = gi_reconnect(p_voxel, n, hist);
    if (c.valid) {
      sum_w = sum_w + c.weight;
      cand_count = cand_count + 1u;
      if (gi_rand2(gi_mix(seed + 0x85EBCA77u)).x * sum_w < c.weight) {
        chosen = c;
        chosen_owned = false;
      }
    }
  }

  if (hist_ok || (prev_ok && pk_ok)) {
    let tb = prev_pc;
    for (var t = 0u; t < GI_SS_REUSE_TAPS; t = t + 1u) {
      let nc = tb + gi_ring_offset(t);
      if (nc.x < 0 || nc.y < 0 || nc.x >= i32(size.x) || nc.y >= i32(size.y)) { continue; }
      let nidx = u32(nc.y) * size.x + u32(nc.x);
      let head = gi_res_head_prev(nidx);
      if ((head.x & (GI_RES_VALID | GI_RES_PK_OK)) != (GI_RES_VALID | GI_RES_PK_OK)) { continue; }
      if (!gi_same_surface(head.y, head.z, head.w, pk0, pk1, npack)) { continue; }
      let ns = gi_res_load_prev(nidx);
      let c = gi_reconnect(p_voxel, n, ns);
      if (!c.valid) { continue; }
      sum_w = sum_w + c.weight;
      cand_count = cand_count + 1u;
      if (gi_rand2(gi_mix(seed + t * 0x27D4EB2Fu + 0x165667B1u)).x * sum_w < c.weight) {
        chosen = c;
        chosen_owned = false;
      }
    }
  }

  if (!chosen.valid) {
    textureStore(gi_out, coord, vec4<f32>(0.0));
    gi_res_store_cur(idx, gi_res_none());
    return;
  }

  if (!chosen_owned && !gi_visible(p_voxel, n, chosen.sec, chosen.dist)) {
    if (own_best.valid) {
      chosen = own_best;
      chosen_owned = true;
    } else {
      textureStore(gi_out, coord, vec4<f32>(0.0));
      gi_res_store_cur(idx, gi_res_none());
      return;
    }
  }

  var r = gi_res_none();
  r.pk0 = pk0;
  r.pk1 = pk1;
  r.nrm = n;
  r.flags = GI_RES_VALID | select(0u, GI_RES_PK_OK, pk_ok) | select(0u, GI_RES_OWNED, chosen_owned)
          | select(0u, GI_RES_SKY, chosen.sky);
  r.ptoken = gi_key_token(pk0, pk1);
  r.p = p_voxel;
  r.w_sum = chosen.value;
  r.m = 1.0;
  gi_res_store_sample(idx, r, GiSample(chosen.value, chosen.sec, chosen.sec_n, chosen.dist,
                                       chosen.hash, chosen.token, chosen.sk0, chosen.sk1, chosen.sky));

  let corr = sum_w / (max(f32(cand_count), 1.0) * max(chosen.weight, 1e-6));
  let out_v = min(chosen.value * corr * GI_PI, vec3<f32>(8.0));
  textureStore(gi_out, coord, vec4<f32>(out_v * gi_u.params.z, 1.0));
  if (gi_u.params.y > 0.5) {
    gi_wal_merge(h.voxel, h.obj_id, h.pal, pk0, pk1, r.w_sum * GI_PI, 1u);
  }
}

fn gi_ss_spatial(coord: vec2<i32>, size: vec2<u32>) {
  let idx = u32(coord.y) * size.x + u32(coord.x);
  let own = gi_res_load_cur(idx);
  if ((own.flags & GI_RES_VALID) == 0u) {
    textureStore(gi_out, coord, vec4<f32>(0.0));
    return;
  }
  let p = own.p;
  let n = own.nrm;
  let npack = gi_pack_dir(n);
  let pk0 = own.pk0;
  let pk1 = own.pk1;
  let own_val = own.w_sum / max(own.m, 1e-6);

  var chosen = gi_reconnect(p, n, own);
  var chosen_own = chosen.valid;
  if (!chosen.valid) {
    chosen = GiCand(own_val, own.sec, own.sec_n, own.dist, own.hash, own.token,
                    own.sk0, own.sk1, false, gi_lum(own_val) * 0.01 + 1e-5, true);
    chosen_own = true;
  };
  var sum_w = chosen.weight;
  var cnt = 1.0;

  let seed = gi_ss_seed(coord, gi_u.seq.x ^ 0xA5A5A5A5u);
  for (var i = 0u; i < 8u; i = i + 1u) {
    let rr = gi_rand2(gi_mix(seed + i * 0x9E3779B1u));
    let nc = clamp(coord + vec2<i32>(i32(rr.x * 17.0) - 8, i32(rr.y * 17.0) - 8),
                   vec2<i32>(0), vec2<i32>(i32(size.x) - 1, i32(size.y) - 1));
    if (all(nc == coord)) { continue; }
    let s = gi_res_load_cur(u32(nc.y) * size.x + u32(nc.x));
    if ((s.flags & GI_RES_VALID) == 0u) { continue; }
    if (!gi_same_surface(s.pk0, s.pk1, gi_pack_dir(s.nrm), pk0, pk1, npack)) { continue; }
    let c = gi_reconnect(p, n, s);
    if (!c.valid) { continue; }
    sum_w = sum_w + c.weight;
    cnt = cnt + 1.0;
    if (gi_rand2(gi_mix(seed + i * 0x27D4EB2Fu + 0x165667B1u)).x * sum_w < c.weight) {
      chosen = c;
      chosen_own = false;
    }
  }

  if (!chosen_own && !gi_visible(p, n, chosen.sec, chosen.dist)) {
    chosen = GiCand(own_val, own.sec, own.sec_n, own.dist, own.hash, own.token,
                    own.sk0, own.sk1, false, gi_lum(own_val) * 0.01 + 1e-5, true);
    chosen_own = true;
  }

  let corr = sum_w / (max(cnt, 1.0) * max(chosen.weight, 1e-6));
  let out_v = min(chosen.value * corr * GI_PI, vec3<f32>(8.0));
  textureStore(gi_out, coord, vec4<f32>(out_v * gi_u.params.z, 1.0));
}
