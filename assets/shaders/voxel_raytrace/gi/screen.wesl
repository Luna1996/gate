import package::bindings::{face_slots, gi_out, gi_res_cur, gi_res_prev, gi_u, light_u};
import package::gi::common::{FACE_WORDS, FACE_W_FLAG, FACE_W_K0, FACE_W_K1, FACE_W_PAL, FACE_W_RAW, FACE_W_RAW_N, FACE_W_TEXEL, GI_ACC_CLAMP, GI_ACC_MAX_N, GI_ACC_SCALE, face_slot_find, face_slot_next, face_slot_of, gi_cos_hemisphere, gi_hist_search_offset, gi_key0, gi_key1, gi_key_ok, gi_mix, gi_pack_dir, gi_rand2, gi_ring_offset, gi_unpack_dir, gi_vox_k1};
import package::gi::consts::{GI_DEN_N_DOT, GI_HIST_SEARCH_TAPS, GI_PI, GI_SS_CAND_CLAMP_CEIL, GI_SS_CAND_CLAMP_K, GI_SS_CAND_N, GI_SS_CAND_N_HQ, GI_SS_M_CAP_K, GI_SS_M_CAP_K_HQ, GI_SS_M_CAP_REST_K, GI_SS_RAY_BATCH, GI_SS_REUSE_CLAMP_HI, GI_SS_REUSE_CLAMP_LO, GI_SS_REUSE_M_K, GI_SS_REUSE_M_K_STATIC, GI_SS_REUSE_TAPS, SHADOW_SURFACE_EPS};
import package::gi::denoise::{gi_den_guide_store, gi_den_same_plane, gi_same_surface_n};
import package::gi::ray::{gi_ray_radiance};
import package::gi::wal::{gi_wal_merge, gi_wal_sample};
import package::trace::{make_grid};
import package::world::{WorldHit, surface_offset, voxel_normal};

fn diag_texel_sampled(coord: vec2<i32>) -> bool {
  return ((((u32(coord.x) * 73856093u) ^ (u32(coord.y) * 19349663u)) & 63u) == 0u);
}
const GI_DIAG_STATIC: u32 = 17u;
const GI_DIAG_TAPS: u32 = 18u;
const GI_DIAG_HIT: u32 = 19u;
const GI_DIAG_PKFAIL: u32 = 20u;
const GI_DIAG_LONE: u32 = 21u;
const GI_DIAG_CLAMP: u32 = 22u;

const GI_RES_WORDS: u32 = 8u;
const GI_RES_W_FLAGS: u32 = 0u; const GI_RES_W_PK: u32 = 1u; const GI_RES_W_NRM: u32 = 3u;
const GI_RES_W_WSUM: u32 = 4u; const GI_RES_W_M: u32 = 7u; 
const GI_RES_VALID: u32 = 1u;
const GI_RES_PK_OK: u32 = 2u;
const GI_RES_LONE: u32 = 0x80000000u;

struct GiRes {
  w_sum: vec3<f32>,
  m: f32,
  pk0: u32,
  pk1: u32,
  flags: u32,
  nrm: vec3<f32>,
}

fn gi_res_none() -> GiRes {
  return GiRes(vec3<f32>(0.0), 0.0, 0u, 0u, 0u, vec3<f32>(0.0, 0.0, 1.0));
}

fn gi_res_load_prev(idx: u32) -> GiRes {
  let b = idx * GI_RES_WORDS;
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
  );
}

fn gi_tap_head(idx: u32) -> vec3<u32> {
  let b = idx * GI_RES_WORDS;
  return vec3<u32>(gi_res_prev[b + GI_RES_W_FLAGS], gi_res_prev[b + GI_RES_W_PK],
                   gi_res_prev[b + GI_RES_W_PK + 1u]);
}

fn gi_tap_merge(idx: u32) -> vec4<f32> {
  let b = idx * GI_RES_WORDS;
  return vec4<f32>(
    bitcast<f32>(gi_res_prev[b + GI_RES_W_WSUM]),
    bitcast<f32>(gi_res_prev[b + GI_RES_W_WSUM + 1u]),
    bitcast<f32>(gi_res_prev[b + GI_RES_W_WSUM + 2u]),
    bitcast<f32>(gi_res_prev[b + GI_RES_W_M]),
  );
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
}

fn gi_res_adopt(r: GiRes, h: GiRes) -> GiRes {
  var out = r;
  out.w_sum = r.w_sum + h.w_sum;
  out.m = r.m + h.m;
  return out;
}

fn gi_res_cap(r: GiRes, cand_n: u32, m_cap_k: u32) -> GiRes {
            let cap = f32(m_cap_k * max(cand_n, 1u));
  if (r.m <= cap) { return r; }
  let shrink = cap / r.m;
  var out = r;
  out.w_sum = r.w_sum * shrink;
  out.m = cap;
  return out;
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

fn gi_ss_main(coord: vec2<i32>, size: vec2<u32>, h: WorldHit) {
      if (diag_texel_sampled(coord)) {
    atomicAdd(&lod_diag[13u], 64u);
  }
  let idx = u32(coord.y) * size.x + u32(coord.x);
  let sun_dir = light_u.lights[0].kind_pos_dir.yzw;
  let sun_c = light_u.lights[0].color_intensity.xyz * light_u.lights[0].color_intensity.w;
    let sun_bounce = gi_u.params.x;
      let epoch = gi_u.seq.y;
      let b2_mult = gi_u.misc.w;

    if (!h.hit) {
    textureStore(gi_out, coord, vec4<f32>(0.0));
    gi_res_store_cur(idx, gi_res_none());
        gi_den_guide_store(coord, size, false, 0u, 0u, vec3<f32>(0.0, 0.0, 1.0), false, vec2<i32>(0));
    return;
  }

  let gg = make_grid(u32(h.obj_id) + 1u);
  let vc = vec3<f32>(h.voxel) + vec3<f32>(0.5);
  let p_voxel = gg.pos + vec3<f32>(dot(vc, gg.col0), dot(vc, gg.col1), dot(vc, gg.col2)) * gg.scale;
            let n = voxel_normal(gg, h.voxel, h.face_id, h.n);
      let n_off = surface_offset(gg, n, SHADOW_SURFACE_EPS);
  let org = p_voxel + n * n_off;

    let pk0 = gi_key0(h.voxel, h.obj_id);
  let pk1 = gi_key1(h.voxel, h.face_id, h.obj_id);
  let pk_ok = gi_key_ok(h.voxel, h.obj_id);

  let frame = gi_u.seq.x;
  let seed = gi_ss_seed(coord, frame);

              var prev_ok = false;
  let prev_pc = gi_ss_reproject(h.point, size, &prev_ok);
  gi_den_guide_store(coord, size, true, pk0, pk1, n, prev_ok, prev_pc);
              let px_static = prev_ok && prev_pc.x == coord.x && prev_pc.y == coord.y;

                      var wal_claimer = false;
  if (pk_ok) {
    var face_ok = false;
    let kv1 = gi_vox_k1(pk1);
    let fslot = face_slot_of(pk0, kv1, &face_ok);
    if (face_ok) {
      var fe = fslot * FACE_WORDS;
      var claimed =
        atomicCompareExchangeWeak(&face_slots[fe + FACE_W_FLAG], 0u, 1u).exchanged;
      if (!claimed) {
                        if (!face_slot_matches(fslot, pk0, kv1)) {
          fe = face_slot_next(fslot) * FACE_WORDS;
          claimed = atomicCompareExchangeWeak(&face_slots[fe + FACE_W_FLAG], 0u, 1u).exchanged;
        }
      }
      if (claimed) {
        wal_claimer = true;
        atomicStore(&face_slots[fe + FACE_W_K0], pk0);
        atomicStore(&face_slots[fe + FACE_W_K1], kv1);
        atomicStore(&face_slots[fe + FACE_W_TEXEL], idx);
        atomicStore(&face_slots[fe + FACE_W_PAL], h.pal);
      }
    }
  }

                                      var hist = gi_res_none();
  var hist_ok = false;
        if (pk_ok && prev_ok && gi_u.flags.z > 0.5) {
    for (var s = 0u; s <= GI_HIST_SEARCH_TAPS; s = s + 1u) {
      let pc = prev_pc + gi_hist_search_offset(s);
      if (pc.x < 0 || pc.y < 0 || pc.x >= i32(size.x) || pc.y >= i32(size.y)) { continue; }
      let nidx = u32(pc.y) * size.x + u32(pc.x);
            let head = gi_res_head_prev(nidx);
      if ((head.x & (GI_RES_VALID | GI_RES_PK_OK)) != (GI_RES_VALID | GI_RES_PK_OK)) { continue; }
      if (!gi_same_surface_n(head.y, head.z, gi_unpack_dir(head.w), pk0, pk1, n)) { continue; }
                  hist = gi_res_load_prev(nidx);
      hist_ok = true;
      break;
    }
  }
          let lone_prev = hist_ok && (hist.flags & GI_RES_LONE) != 0u;

                        let hq = u32(gi_u.flags.w) >= 3u;
  let base_n = select(GI_SS_CAND_N, GI_SS_CAND_N_HQ, hq);
  let cand_n0 = select(base_n, base_n * 2u, !hist_ok);
              if (hist_ok && diag_texel_sampled(coord)) {
    atomicAdd(&lod_diag[16u], 64u);
  }
              let share = max(u32(gi_u.misc.y), 1u);
                      let m_cap_base = select(GI_SS_M_CAP_K, GI_SS_M_CAP_K_HQ, hq);
  let err_tier = u32(gi_u.misc.z);
  var b_err = 1.0;
  if (err_tier > 0u) {
                                        let hi = select(select(1.40, 2.00, err_tier >= 2u), 4.00, err_tier >= 3u);
            let warm_m = f32(m_cap_base * cand_n0);
    b_err = select(hi, 1.0, hist_ok && hist.m >= warm_m);
  }
    let dither = gi_rand2(gi_mix(seed + 0x2545F491u)).x;
  var cand_n = max(1u, u32(floor(f32(cand_n0) * b_err / f32(share) + dither)));
                if (cand_n == 1u) {
    let frac = f32(cand_n0) * b_err / f32(share);
    if (frac < 0.999) {
      let stride = max(u32(ceil(1.0 / max(frac, 1.0 / 64.0))), 1u);
      if (gi_mix(idx ^ (gi_u.seq.x * 0x9E3779B9u)) % stride != 0u) {
        cand_n = 0u;
      }
    }
  }
                          const GI_SS_CLAIM_ONLY: u32 = 0u;
  if (GI_SS_CLAIM_ONLY != 0u && pk_ok && cand_n > 0u) {
    var fok2 = false;
        let fs2 = face_slot_find(pk0, pk1, &fok2);
    if (fok2) {
      let fe2 = fs2 * FACE_WORDS;
      if (atomicLoad(&face_slots[fe2 + FACE_W_TEXEL]) != idx) {
        cand_n = 0u;
      }
    }
  }
            let cap_mul = select(1u, GI_SS_M_CAP_REST_K, px_static || lone_prev);
  let m_cap_k = m_cap_base * max(cand_n0 / max(cand_n, 1u), 1u) * cap_mul;
  var r = gi_res_none();
              var clamped_any = false;
  var k = 0u;
  loop {
    if (k >= cand_n) { break; }
    let nb = min(GI_SS_RAY_BATCH, cand_n - k);
    var b_s: array<vec3<f32>, GI_SS_RAY_BATCH>;
    for (var i = 0u; i < nb; i = i + 1u) {
      let kk = k + i;
      let cseed = gi_mix(seed + kk * 0x9E3779B9u + 1u);
      let dir = gi_cos_hemisphere(n, gi_rand2(cseed));
                                    let b2 = select(0.0, b2_mult, gi_rand2(gi_mix(cseed + 0x27D4EB2Fu)).y < 1.0 / max(b2_mult, 1.0));
      b_s[i] = gi_ray_radiance(org, dir, sun_dir, sun_c, sun_bounce, epoch, b2, cseed);
    }
    for (var i = 0u; i < nb; i = i + 1u) {
            var c = b_s[i] * GI_PI;
                                          if (GI_SS_CAND_CLAMP_K > 0.0) {
        let m_now = r.m + hist.m;
        if (m_now > 0.5) {
          let cap = max(
            (r.w_sum + hist.w_sum) / m_now * GI_SS_CAND_CLAMP_K,
            vec3<f32>(GI_SS_CAND_CLAMP_CEIL),
          );
          if (c.x > cap.x || c.y > cap.y || c.z > cap.z) { clamped_any = true; }
          c = min(c, cap);
        }
      }
      r.w_sum = r.w_sum + c;
      r.m = r.m + 1.0;
    }
    k = k + nb;
  }

        r.pk0 = pk0;
  r.pk1 = pk1;
  r.flags = GI_RES_VALID | select(0u, GI_RES_PK_OK, pk_ok);
  r.nrm = n;
  if (hist_ok) { r = gi_res_adopt(r, hist); }
  r = gi_res_cap(r, cand_n, m_cap_k);
  gi_res_store_cur(idx, r);

                        if (gi_u.params.y > 0.5 && wal_claimer && r.m > 0.5) {
    gi_wal_merge(h.voxel, h.obj_id, h.pal, pk0, pk1, r.w_sum, u32(r.m));
  }

                                    let tb = prev_pc;
  var sk_sum = vec3<f32>(0.0);
  var sk_m = 0.0;
    var tap_used = 0u;
            let reuse_taps =
    select(GI_SS_REUSE_TAPS, 0u, !prev_ok || gi_u.flags.z <= 0.5 || r.m < 0.5);
      let est_self = r.w_sum / max(r.m, 1e-6);
  for (var t = 0u; t < reuse_taps; t = t + 1u) {
    let nc = tb + gi_ring_offset(t);
    if (nc.x < 0 || nc.y < 0 || nc.x >= i32(size.x) || nc.y >= i32(size.y)) { continue; }
    let nidx = u32(nc.y) * size.x + u32(nc.x);
    let head = gi_tap_head(nidx);
    if ((head.x & (GI_RES_VALID | GI_RES_PK_OK)) != (GI_RES_VALID | GI_RES_PK_OK)) { continue; }
        if (!gi_den_same_plane(head.y, head.z, pk0, pk1)) { continue; }
    if (dot(gi_unpack_dir(gi_res_prev[nidx * GI_RES_WORDS + GI_RES_W_NRM]), n) < GI_DEN_N_DOT) {
      continue;
    }
    let wm = gi_tap_merge(nidx);
                    let v = wm.xyz / max(wm.w, 1e-6);
    let vc = clamp(v, est_self * GI_SS_REUSE_CLAMP_LO, est_self * GI_SS_REUSE_CLAMP_HI);
    sk_sum = sk_sum + vc * wm.w;
    sk_m = sk_m + wm.w;
    tap_used = tap_used + 1u;
  }
      let lone_now = reuse_taps > 0u && tap_used == 0u;
            if (diag_texel_sampled(coord)) {
    atomicAdd(&lod_diag[GI_DIAG_STATIC], select(0u, 64u, px_static));
    atomicAdd(&lod_diag[GI_DIAG_TAPS], tap_used * 64u);
    atomicAdd(&lod_diag[GI_DIAG_HIT], 64u);
    atomicAdd(&lod_diag[GI_DIAG_PKFAIL], select(0u, 64u, !pk_ok));
    atomicAdd(&lod_diag[GI_DIAG_LONE], select(0u, 64u, lone_now));
    atomicAdd(&lod_diag[GI_DIAG_CLAMP], select(0u, 64u, clamped_any));
  }
        let lone_keep = select(lone_now, hist_ok && (hist.flags & GI_RES_LONE) != 0u, reuse_taps == 0u);
  gi_res_cur[idx * GI_RES_WORDS + GI_RES_W_FLAGS] =
    select(r.flags & 0x7FFFFFFFu, r.flags | GI_RES_LONE, lone_keep);

      var est = est_self;
  if (sk_m > 0.0) {
                                    let m_k = select(GI_SS_REUSE_M_K, GI_SS_REUSE_M_K_STATIC, px_static);
    let keep = min(1.0, m_k * r.m / max(sk_m, 1e-6));
    let denom = max(r.m + sk_m * keep, 1e-6);
    est = (r.w_sum + sk_sum * keep) / denom;
  }
  let gi = est * (gi_u.params.z / GI_PI);
        let has_evidence = (r.m + sk_m) > 0.0;
                        var gi_w = gi;
  var cov_w = select(0.0, 1.0, has_evidence);
  if (gi_u.params.y > 0.5) {
    let s = gi_wal_sample(h.voxel, h.obj_id, h.pal, pk1);
    if (s.w > 0.0) {
      gi_w = mix(gi, s.rgb * (gi_u.params.z / GI_PI), s.w);
      cov_w = max(cov_w, s.w);
    }
  }
                textureStore(gi_out, coord, vec4<f32>(gi_w, cov_w));

                  if (pk_ok && cov_w > 0.0) {
    var fok = false;
        let fslot = face_slot_find(pk0, gi_vox_k1(pk1), &fok);
    if (fok) {
      let fe = fslot * FACE_WORDS;
                  let q = clamp(gi_w, vec3<f32>(0.0), vec3<f32>(GI_ACC_CLAMP));
      let qv = vec3<u32>(round(q * GI_ACC_SCALE));
      var reserved = false;
      loop {
        let c = atomicLoad(&face_slots[fe + FACE_W_RAW_N]);
        if (c + 1u > GI_ACC_MAX_N) { break; }
        if (atomicCompareExchangeWeak(&face_slots[fe + FACE_W_RAW_N], c, c + 1u).exchanged) {
          reserved = true;
          break;
        }
      }
      if (reserved) {
        atomicAdd(&face_slots[fe + FACE_W_RAW], qv.x);
        atomicAdd(&face_slots[fe + FACE_W_RAW + 1u], qv.y);
        atomicAdd(&face_slots[fe + FACE_W_RAW + 2u], qv.z);
      }
    }
  }
}