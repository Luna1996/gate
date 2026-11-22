// GI denoise: temporal reprojection followed by edge-aware a-trous over the resampler output.
//
// The temporal pass runs first so the stochastic resampler signal is accumulated before any
// spatial kernel touches it; the a-trous rounds then remove the remaining per-pixel variance
// and write the texture the shade samples. Signal alpha is validity (1 = data, 0 = none), not
// the history age: the age lives in the history buffer so the shade's coverage test stays 0..1.
// History:                          .rgb = accumulated irradiance (in the history buffer),
//                                   age/moments/surface data in the words next to it.
//
// Surface guides come from the same DDA pass that fills the resampler G-buffer: the per-pixel
// key/normal/token live in `gi_key_tex`, the primary world position and camera distance live in
// `gi_surface_tex`. Validation uses the key, the lifetime token and the centre surface's plane,
// so a stale tap can never cross an edit, a moved instance or a depth step.

import package::bindings::{gi_den_cfg, gi_dn_dst, gi_dn_src, gi_dn_tmp, gi_guide, gi_guide_out, gi_hist_cur, gi_hist_prev, gi_key_tex, gi_raw_tex, gi_surface_tex};
import package::gi::common::{gi_hist_search_offset, gi_pack_dir, gi_unpack_dir, gi_vox_k1};
import package::gi::consts::{
  GI_DEN_GUIDE_WORDS, GI_DEN_G_KEY0, GI_DEN_G_KEY1,
  GI_DEN_G_NRM, GI_DEN_G_PREV, GI_DEN_G_RESET, GI_DEN_HIST_WORDS, GI_DEN_N_DOT, GI_LUMA,
};
import package::gi::resample::{GI_KEY_FLAG_DIFFUSE};

// History word layout. Colour/moments are stored as bitcast f32 words so the readback probe can
// average them without another staging path.
const GI_DN_H_COL: u32 = 0u;
const GI_DN_H_AGE: u32 = 3u;
const GI_DN_H_M1: u32 = 4u;
const GI_DN_H_M2: u32 = 5u;
const GI_DN_H_KEY0: u32 = 6u;
const GI_DN_H_KEY1: u32 = 7u;
const GI_DN_H_TOKEN: u32 = 8u;
const GI_DN_H_NRM: u32 = 9u;
const GI_DN_H_POS: u32 = 12u; // words 12..14 = world position, 15 = pad

const GI_DN_MAX_FRAMES: f32 = 4096.0;
const GI_DN_CURRENT_MIN_WEIGHT: f32 = 0.0002;
const GI_DN_DEPTH_ABS: f32 = 0.5;
const GI_DN_DEPTH_REL: f32 = 0.005;
const GI_DN_LATERAL_MAX: f32 = 32.0;
const GI_DN_NORMAL_POWER: f32 = 64.0;
const GI_DN_RADIANCE_SIGMA: f32 = 4.0;
const GI_DN_RESET_SCALE: f32 = 0.25;

// --- guide helpers (also used by the DDA GI pass) -------------------------------------------

fn gi_den_valid_b(b: u32) -> bool {
  return (gi_guide[b + GI_DEN_G_KEY0] | gi_guide[b + GI_DEN_G_KEY1]) != 0u;
}

fn gi_den_key0_b(b: u32) -> u32 {
  return gi_guide[b + GI_DEN_G_KEY0];
}

fn gi_den_key1_b(b: u32) -> u32 {
  return gi_guide[b + GI_DEN_G_KEY1];
}

fn gi_den_guide_store(
  coord: vec2<i32>,
  size: vec2<u32>,
  valid: bool,
  k0: u32,
  k1: u32,
  n: vec3<f32>,
  prev_ok: bool,
  prev_px: vec2<i32>,
  reset: bool,
) {
  let idx = u32(coord.y) * size.x + u32(coord.x);
  let b = idx * GI_DEN_GUIDE_WORDS;
  if (!valid) {
    for (var i = 0u; i < GI_DEN_GUIDE_WORDS; i = i + 1u) { gi_guide_out[b + i] = 0u; }
    return;
  }
  gi_guide_out[b + GI_DEN_G_KEY0] = k0;
  gi_guide_out[b + GI_DEN_G_KEY1] = k1;
  gi_guide_out[b + GI_DEN_G_NRM] = gi_pack_dir(n);

  gi_guide_out[b + GI_DEN_G_RESET] = select(0u, 1u, reset);

  var code = 0xFFFFFFFFu;
  if (prev_ok) { code = u32(prev_px.x) | (u32(prev_px.y) << 16u); }
  gi_guide_out[b + GI_DEN_G_PREV] = code;
}

fn gi_den_same_plane(hk0: u32, hk1: u32, k0: u32, k1: u32) -> bool {
  let face = (k1 >> 16u) & 7u;
  if (face != ((hk1 >> 16u) & 7u)) { return false; }
  if ((k1 >> 19u) != (hk1 >> 19u)) { return false; }
  let axis = face >> 1u;
  if (axis == 0u) { return (k0 & 0xFFFFu) == (hk0 & 0xFFFFu); }
  if (axis == 1u) { return (k0 >> 16u) == (hk0 >> 16u); }
  return (k1 & 0xFFFFu) == (hk1 & 0xFFFFu);
}

fn gi_same_surface(hk0: u32, hk1: u32, hn: u32, k0: u32, k1: u32, n: u32) -> bool {
  if (!gi_den_same_plane(hk0, hk1, k0, k1)) { return false; }
  return dot(gi_unpack_dir(hn), gi_unpack_dir(n)) >= GI_DEN_N_DOT;
}

fn gi_near_surface(hk0: u32, hk1: u32, hn: u32, k0: u32, k1: u32, n: u32) -> bool {
  if ((k1 >> 19u) != (hk1 >> 19u)) { return false; }
  if (dot(gi_unpack_dir(hn), gi_unpack_dir(n)) < GI_DEN_N_DOT) { return false; }
  return (hk0 == k0 && gi_vox_k1(hk1) == gi_vox_k1(k1)) || gi_den_same_plane(hk0, hk1, k0, k1);
}

// --- shared denoise helpers -----------------------------------------------------------------

fn gi_dn_size() -> vec2<u32> {
  return vec2<u32>(max(gi_den_cfg[0u], 1u), max(gi_den_cfg[1u], 1u));
}

fn gi_dn_frame() -> u32 {
  return gi_den_cfg[2u];
}

fn gi_dn_radius() -> i32 {
  return max(i32(gi_den_cfg[3u]), 0);
}

fn gi_dn_luma(c: vec3<f32>) -> f32 {
  return dot(c, GI_LUMA);
}

fn gi_dn_hash(x: u32) -> u32 {
  var h = x;
  h = h ^ (h >> 16u);
  h = h * 0x7feb352du;
  h = h ^ (h >> 15u);
  h = h * 0x846ca68bu;
  h = h ^ (h >> 16u);
  return h;
}

fn gi_dn_surface(p: vec2<u32>) -> vec4<f32> {
  return textureLoad(gi_surface_tex, vec2<i32>(p), 0);
}

fn gi_dn_key(p: vec2<u32>) -> vec4<u32> {
  return textureLoad(gi_key_tex, vec2<i32>(p), 0);
}

fn gi_dn_key_ok(key: vec4<u32>) -> bool {
  return (key.z & GI_KEY_FLAG_DIFFUSE) != 0u;
}

fn gi_dn_hist_bit(b: u32, w: u32) -> f32 {
  return bitcast<f32>(gi_hist_prev[b + w]);
}

fn gi_dn_hist_store(b: u32, color: vec3<f32>, age: f32, m1: f32, m2: f32,
                    k0: u32, k1: u32, token: u32, nrm: u32, pos: vec3<f32>) {
  gi_hist_cur[b + GI_DN_H_COL] = bitcast<u32>(color.x);
  gi_hist_cur[b + GI_DN_H_COL + 1u] = bitcast<u32>(color.y);
  gi_hist_cur[b + GI_DN_H_COL + 2u] = bitcast<u32>(color.z);
  gi_hist_cur[b + GI_DN_H_AGE] = bitcast<u32>(age);
  gi_hist_cur[b + GI_DN_H_M1] = bitcast<u32>(m1);
  gi_hist_cur[b + GI_DN_H_M2] = bitcast<u32>(m2);
  gi_hist_cur[b + GI_DN_H_KEY0] = k0;
  gi_hist_cur[b + GI_DN_H_KEY1] = k1;
  gi_hist_cur[b + GI_DN_H_TOKEN] = token;
  gi_hist_cur[b + GI_DN_H_NRM] = nrm;
  gi_hist_cur[b + GI_DN_H_POS] = bitcast<u32>(pos.x);
  gi_hist_cur[b + GI_DN_H_POS + 1u] = bitcast<u32>(pos.y);
  gi_hist_cur[b + GI_DN_H_POS + 2u] = bitcast<u32>(pos.z);
}

fn gi_dn_hist_clear(b: u32) {
  for (var i = 0u; i < GI_DEN_HIST_WORDS; i = i + 1u) { gi_hist_cur[b + i] = 0u; }
}

fn gi_dn_valid_tap(p: vec2<i32>, size: vec2<u32>) -> bool {
  return p.x >= 0 && p.y >= 0 && p.x < i32(size.x) && p.y < i32(size.y);
}

fn gi_dn_kernel(offset: i32) -> f32 {
  let a = abs(offset);
  if (a == 0) { return 0.375; }
  if (a == 1) { return 0.25; }
  return 0.0625;
}

fn gi_dn_plane_tol(distance: f32) -> f32 {
  return max(GI_DN_DEPTH_ABS, abs(distance) * GI_DN_DEPTH_REL);
}

// --- temporal accumulation ------------------------------------------------------------------

fn gi_den_temporal(coord: vec2<i32>) {
  let size = gi_dn_size();
  let idx = u32(coord.y) * size.x + u32(coord.x);
  let gb = idx * GI_DEN_GUIDE_WORDS;
  let hb = idx * GI_DEN_HIST_WORDS;

  let key = gi_dn_key(vec2<u32>(coord));
  let guide_ok = gi_den_valid_b(gb);
  let cur = textureLoad(gi_raw_tex, coord, 0);
  let cur_ok = guide_ok && gi_dn_key_ok(key) && cur.a > 0.001;

  if (!cur_ok) {
    gi_dn_hist_clear(hb);
    textureStore(gi_dn_tmp, coord, vec4<f32>(0.0));
    return;
  }

  let k0 = gi_den_key0_b(gb);
  let k1 = gi_den_key1_b(gb);
  let n_c_packed = gi_guide[gb + GI_DEN_G_NRM];
  let n_c = gi_unpack_dir(n_c_packed);
  let token = key.y;
  let surf = gi_dn_surface(vec2<u32>(coord));
  let p_c = surf.xyz;
  let tol = gi_dn_plane_tol(surf.w);

  var color = cur.rgb;
  var m1 = gi_dn_luma(cur.rgb);
  var m2 = m1 * m1;

  var hist_found = false;
  var hist_age = 0.0;
  var hist_m1 = 0.0;
  var hist_m2 = 0.0;
  var hist_color = vec3<f32>(0.0);

  let code = gi_guide[gb + GI_DEN_G_PREV];
  if (code != 0xFFFFFFFFu) {
    let prev_px = vec2<i32>(i32(code & 0xFFFFu), i32((code >> 16u) & 0xFFFFu));
    for (var s = 0u; s <= 8u; s = s + 1u) {
      let px = prev_px.x + gi_hist_search_offset(s).x;
      let py = prev_px.y + gi_hist_search_offset(s).y;
      if (px < 0 || py < 0 || px >= i32(size.x) || py >= i32(size.y)) { continue; }
      let cand = (u32(py) * size.x + u32(px)) * GI_DEN_HIST_WORDS;
      let age = gi_dn_hist_bit(cand, GI_DN_H_AGE);
      if (age <= 0.0) { continue; }
      if (gi_hist_prev[cand + GI_DN_H_TOKEN] != token) { continue; }
      let hk0 = gi_hist_prev[cand + GI_DN_H_KEY0];
      let hk1 = gi_hist_prev[cand + GI_DN_H_KEY1];
      let hn = gi_hist_prev[cand + GI_DN_H_NRM];
      if (!gi_same_surface(hk0, hk1, hn, k0, k1, n_c_packed)) { continue; }
      let p_h = vec3<f32>(
        gi_dn_hist_bit(cand, GI_DN_H_POS),
        gi_dn_hist_bit(cand, GI_DN_H_POS + 1u),
        gi_dn_hist_bit(cand, GI_DN_H_POS + 2u),
      );
      let dp = p_h - p_c;
      if (abs(dot(dp, n_c)) > tol) { continue; }
      if (dot(dp, dp) > GI_DN_LATERAL_MAX * GI_DN_LATERAL_MAX) { continue; }
      hist_age = age;
      hist_m1 = gi_dn_hist_bit(cand, GI_DN_H_M1);
      hist_m2 = gi_dn_hist_bit(cand, GI_DN_H_M2);
      hist_color = vec3<f32>(
        gi_dn_hist_bit(cand, GI_DN_H_COL),
        gi_dn_hist_bit(cand, GI_DN_H_COL + 1u),
        gi_dn_hist_bit(cand, GI_DN_H_COL + 2u),
      );
      hist_found = true;
      break;
    }
  }

  if (hist_found) {
    let age_max = min(hist_age, GI_DN_MAX_FRAMES - 1.0);
    var w_hist = clamp(1.0 - 1.0 / max(age_max + 1.0, 1.0), 0.0, 1.0);
    if (gi_guide[gb + GI_DEN_G_RESET] != 0u) { w_hist = min(w_hist, GI_DN_RESET_SCALE); }
    let cur_w = max(1.0 - w_hist, GI_DN_CURRENT_MIN_WEIGHT);
    let l_in = gi_dn_luma(cur.rgb);
    color = mix(hist_color, cur.rgb, cur_w);
    m1 = mix(hist_m1, l_in, cur_w);
    m2 = mix(hist_m2, l_in * l_in, cur_w);
    hist_age = age_max + 1.0;
  }

  let age_out = min(max(hist_age, 1.0), GI_DN_MAX_FRAMES);
  gi_dn_hist_store(hb, color, age_out, m1, m2, k0, k1, token, n_c_packed, p_c);
  textureStore(gi_dn_tmp, coord, vec4<f32>(color, 1.0));
}

// --- edge-aware a-trous ---------------------------------------------------------------------

fn gi_den_atrous(coord: vec2<i32>, step: i32) {
  let size = gi_dn_size();
  let idx = u32(coord.y) * size.x + u32(coord.x);
  let gb = idx * GI_DEN_GUIDE_WORDS;
  let key = gi_dn_key(vec2<u32>(coord));
  if (!gi_den_valid_b(gb) || !gi_dn_key_ok(key)) {
    textureStore(gi_dn_dst, coord, vec4<f32>(0.0));
    return;
  }

  let c0 = textureLoad(gi_dn_src, coord, 0);
  if (!(c0.a > 0.001)) {
    textureStore(gi_dn_dst, coord, c0);
    return;
  }

  let k0 = gi_den_key0_b(gb);
  let k1 = gi_den_key1_b(gb);
  let n_packed = gi_guide[gb + GI_DEN_G_NRM];
  let n0 = gi_unpack_dir(n_packed);
  let surf = gi_dn_surface(vec2<u32>(coord));
  let p0 = surf.xyz;
  let tol = gi_dn_plane_tol(surf.w);
  let l0 = gi_dn_luma(c0.rgb);
  let r = gi_dn_radius();
  let st = max(step, 1);
  let use_luma = st > 1;

  var radiance_denom = 1.0;
  if (use_luma) {
    var lsum = 0.0;
    var lsum2 = 0.0;
    var cnt = 0.0;
    for (var my: i32 = -1; my <= 1; my = my + 1) {
      for (var mx: i32 = -1; mx <= 1; mx = mx + 1) {
        let p = coord + vec2<i32>(mx, my) * st;
        if (!gi_dn_valid_tap(p, size)) { continue; }
        let tb = (u32(p.y) * size.x + u32(p.x)) * GI_DEN_GUIDE_WORDS;
        if (!gi_den_valid_b(tb)) { continue; }
        let s = textureLoad(gi_dn_src, p, 0);
        if (!(s.a > 0.001)) { continue; }
        let l = gi_dn_luma(s.rgb);
        lsum = lsum + l;
        lsum2 = lsum2 + l * l;
        cnt = cnt + 1.0;
      }
    }
    var lstd = 0.0;
    if (cnt > 0.5) {
      let mean = lsum / cnt;
      lstd = sqrt(max(lsum2 / cnt - mean * mean, 0.0));
    }
    radiance_denom = GI_DN_RADIANCE_SIGMA * lstd + 0.01;
  }

  // Tap jitter is off by default: the filter runs after temporal accumulation, so a per-frame
  // jitter would resurface as static-scene flicker instead of being averaged away.
  let apply_jitter = gi_den_cfg[4u] != 0u && st > 1;
  let mask = u32(st - 1);
  let half = st >> 1;
  let seed = gi_dn_hash(
    u32(coord.x) ^ (u32(coord.y) << 16u)
    ^ (gi_dn_frame() * 0x68bc21ebu)
    ^ (u32(st) * 0x02e5be93u));

  var acc = c0.rgb * 0.140625;
  var wsum = 0.140625;
  for (var dy: i32 = -r; dy <= r; dy = dy + 1) {
    for (var dx: i32 = -r; dx <= r; dx = dx + 1) {
      if (dx == 0 && dy == 0) { continue; }
      var off = vec2<i32>(dx, dy) * st;
      if (apply_jitter) {
        let tap_hash = gi_dn_hash(seed + u32((dy + 2) * 5 + (dx + 2)));
        off = off + vec2<i32>(i32(tap_hash & mask), i32((tap_hash >> 16u) & mask))
          - vec2<i32>(half);
      }
      let p = coord + off;
      if (!gi_dn_valid_tap(p, size)) { continue; }
      let tb = (u32(p.y) * size.x + u32(p.x)) * GI_DEN_GUIDE_WORDS;
      if (!gi_den_valid_b(tb)) { continue; }
      if (!gi_den_same_plane(gi_den_key0_b(tb), gi_den_key1_b(tb), k0, k1)) { continue; }
      let s = textureLoad(gi_dn_src, p, 0);
      if (!(s.a > 0.001)) { continue; }
      let ps = gi_dn_surface(vec2<u32>(p));
      if (abs(dot(ps.xyz - p0, n0)) > tol) { continue; }

      let n1 = gi_unpack_dir(gi_guide[tb + GI_DEN_G_NRM]);
      let normal_w = pow(saturate(dot(n0, n1)), GI_DN_NORMAL_POWER);
      if (normal_w <= 1e-6) { continue; }

      var radiance_w = 1.0;
      if (use_luma) {
        radiance_w = exp(-abs(gi_dn_luma(s.rgb) - l0) / radiance_denom);
      }

      let w = gi_dn_kernel(dx) * gi_dn_kernel(dy) * normal_w * radiance_w;
      acc = acc + s.rgb * w;
      wsum = wsum + w;
    }
  }

  textureStore(gi_dn_dst, coord, vec4<f32>(acc / max(wsum, 1e-6), 1.0));
}
