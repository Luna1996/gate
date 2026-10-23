import package::bindings::{refl_cfg, refl_meta, refl_req, refl_sum, out_base_tex, BASE_FLAG_IMPLICIT_NORMAL, BASE_FLAG_SHADOW, BASE_FLAG_VOL_TINT, BEAM_BACKOFF, BEAM_DIV, beam_depth, dda_hit, eye_adapt, eye_adapt_ro, face_slots, face_slots_flat, fog_tex, gi_dn_dst, gi_dn_tmp, gi_guide, gi_guide_out, gi_out, gi_out_flat, gi_tex, gi_u, gi_wal, light_samp, light_u, lod_diag, out_base, out_tex, out_tex_src, view_u};
import package::common::{FLAGS_TRANSMISSIVE, MaterialParams, SurfaceLight, brdf_reflected, dielectric_f0, dither3, emissive_radiance, f0_of, face_view_dir, fetch_material, fresnel_schlick, linear_to_srgb, material_ior, palette_flags, palette_rgb_linear, sky_rgb, srgb_to_linear};
import package::gi::common::{FACE_WORDS, FACE_W_FLAG, FACE_W_GI, FACE_W_GI_N, FACE_W_K0, FACE_W_K1, FACE_W_PAL, FACE_W_RAW, FACE_W_RAW_N, FACE_W_TEXEL, face_slot_claim, face_slot_find, face_slot_find_flat, face_slot_of, gi_cos_hemisphere, gi_key0, gi_key1, gi_key_ok, gi_local_rev, gi_pose_salt, gi_unkey_voxel, gi_vox_k1, gi_mix, gi_pack_dir, gi_rand2, gi_unpack_dir, GI_ACC_CLAMP, GI_ACC_MAX_N, GI_ACC_SCALE};
import package::gi::consts::{GI_AMBIENT_FLOOR, GI_DEN_GUIDE_WORDS, GI_DEN_G_NOISE, GI_DEN_G_KEY0, GI_DEN_G_KEY1, GI_DEN_G_NRM, GI_DEN_N_DOT, GI_EMIT_GAIN, GI_LUMA, GI_PI, GI_SKY_AMBIENT, GI_SS_ADAPT_HI, GI_SS_ADAPT_LO, GI_SS_ADAPT_REF, GI_SS_CAND_CLAMP_CEIL, GI_SS_CAND_CLAMP_K, SHADOW_SURFACE_EPS};
import package::gi::denoise::{gi_den_atrous, gi_den_temporal, gi_same_surface};
import package::gi::ray::{gi_ray_radiance, sun_transmittance};
import package::gi::screen::{gi_ss_main};
import package::gi::wal::{gi_wal_merge, gi_wal_nmask, gi_wal_pool, gi_wal_slot, gi_wal_touch, GI_WAL_DIAG, GI_WAL_MATURE_N, GI_WAL_UPD_B, GI_WAL_UPD_R, GI_WAL_WORDS};
import package::brickmap::{sample_brickmap};
import package::trace::{LOD_DIAG, diag_sampled, make_grid, medium_of};
import package::volumetric::{godray_pass_blur_a, godray_pass_blur_b, godray_pass_mask, godray_sample_bilinear, sky_primary};
import package::world::{voxel_normal_fast, WorldHit, camera_solid_pal, face_normal_world, surface_offset, vol_tint_color, voxel_center_world, voxel_normal, world_cfg_full, world_cfg_primary, world_miss, world_raycast, world_trace_glass_path, world_trace_medium_scene};

@if(ray_query)
enable wgpu_ray_query;

@compute @workgroup_size(8, 8, 1)
fn beam_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let beam_size = textureDimensions(beam_depth);
  if (gid.x >= beam_size.x || gid.y >= beam_size.y) { return; }
  let px = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5))
         / vec2<f32>(f32(beam_size.x), f32(beam_size.y));
  let uv = vec2<f32>(px.x * 2.0 - 1.0, 1.0 - px.y * 2.0);
  let near_ndc = vec4<f32>(uv.x, uv.y, 0.0, 1.0);
  let far_ndc  = vec4<f32>(uv.x, uv.y, 1.0, 1.0);
  let near_world_h = view_u.inv_view_proj * near_ndc;
  let far_world_h  = view_u.inv_view_proj * far_ndc;
  let near_world = near_world_h.xyz / near_world_h.w;
  let far_world  = far_world_h.xyz  / far_world_h.w;
  let dir = normalize(far_world - near_world);
  let origin = view_u.cam_pos_voxel.xyz;
  let t_cap = length(far_world - near_world);
  let d_beam = 0.35 / (f32(BEAM_DIV) * max(view_u.lod.x, 1e-6));
  let t_cap_beam = min(t_cap, d_beam);

  let h = world_raycast(make_grid(0u), origin, dir, t_cap_beam, world_cfg_full(0.0, 3u));
  let t = select(t_cap_beam, h.t, h.hit);
  textureStore(beam_depth, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(t, 0.0, 0.0, 0.0));
}
fn dda_hit_record(gid: vec3<u32>, first: WorldHit, origin_voxel: vec3<f32>) {
  let d = max(u32(gi_u.flags.y), 1u);
  if (gid.x % d != d / 2u || gid.y % d != d / 2u) { return; }
  let texel = vec2<i32>(i32(gid.x / d), i32(gid.y / d));
  let hsz = vec2<f32>(textureDimensions(dda_hit));
  if (texel.x >= i32(hsz.x) || texel.y >= i32(hsz.y)) { return; }
  
  let gpx = (vec2<f32>(f32(texel.x), f32(texel.y)) + vec2<f32>(0.5))
          / vec2<f32>(f32(hsz.x), f32(hsz.y));
  let guv = vec2<f32>(gpx.x * 2.0 - 1.0, 1.0 - gpx.y * 2.0);
  let gc_near_h = view_u.inv_view_proj * vec4<f32>(guv.x, guv.y, 0.0, 1.0);
  let gc_far_h = view_u.inv_view_proj * vec4<f32>(guv.x, guv.y, 1.0, 1.0);
  let dir_c = normalize(gc_far_h.xyz / gc_far_h.w - gc_near_h.xyz / gc_near_h.w);
  let t_c = dot(first.point - origin_voxel, dir_c);
  let pack_w = ((gi_u.seq.x & 0xFFu) << 24u) | ((first.pal & 0x7FFFFFu) << 1u) | u32(first.hit);
  textureStore(dda_hit, texel,
               vec4<u32>(bitcast<u32>(select(0.0, t_c, first.hit)),
                         gi_key0(first.voxel, first.obj_id),
                         gi_key1(first.voxel, first.face_id, first.obj_id),
                         pack_w));
}
fn dda_hit_unpack(raw: vec4<u32>, origin_voxel: vec3<f32>, dir_voxel: vec3<f32>) -> WorldHit {
  let t = bitcast<f32>(raw.x);
  let obj = i32((raw.z >> 19u) & 0x1FFFu) - 1;
  let salt = gi_pose_salt(obj);
  let face = ((raw.z ^ salt) >> 16u) & 7u;
  let g = make_grid(u32(obj + 1));
  let voxel = gi_unkey_voxel(raw.y, raw.z, obj);
  let n = face_normal_world(g, face);
  return WorldHit(true, t, origin_voxel + dir_voxel * t, (raw.w >> 1u) & 0x7FFFFFu,
                  n, face, voxel, obj, g.palette_base);
}

const GLASS_EPS: f32 = 1e-3;

const GLASS_EXIT_BEND: bool = true;
const GLASS_QUALITY_SHIFT: u32 = 3u;
const GLASS_REFL_MIN: f32 = 0.02;
const GLASS_REFL_WEIGHT_MIN: f32 = 0.015;
const GLASS_REFL_AMBIENT: f32 = 0.15;
const PBR_REFLECTION_ENABLED: u32 = 1u;
const PBR_REFL_MIN_F0: f32 = 0.03;
const PBR_REFL_MAX_ROUGH: f32 = 0.05;
const PBR_REFL_CONE_K: f32 = 1.0;

fn refl_cone_dir(r: vec3<f32>, u: vec2<f32>, spread: f32) -> vec3<f32> {
  let axis = select(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 0.0), abs(r.z) > 0.9);
  let t = normalize(cross(axis, r));
  let b = cross(r, t);
  let rad = spread * sqrt(u.x);
  let phi = 6.2831853 * u.y;
  return normalize(r + t * (rad * cos(phi)) + b * (rad * sin(phi)));
}

const PBR_DEBUG_ASSET: u32 = 0xFFFFFFFFu;

var<private> dbg_gi: vec3<f32> = vec3<f32>(0.0);

struct GlassOut {
  ray: WorldHit,
  tint: vec3<f32>,
  refl: vec3<f32>,
  dir: vec3<f32>,
}

fn secondary_reach(max_t: f32) -> f32 {
  let rc = light_u.g.refl_reach;
  return select(min(max_t, f32(rc)), max_t, rc == 0xFFFFFFFFu);
}

fn secondary_on() -> bool {
  return light_u.g.refl_reach != 0u;
}

const REFL_NO_PX: u32 = 0xFFFFFFFFu;
const REFL_FAR_T: f32 = 1.0e9;
const REFL_KIND_PACKED: u32 = 1u;
const REFL_KIND_RAY: u32 = 0u;

fn pack_w10(w: vec3<f32>) -> u32 {
  let q = clamp(w, vec3<f32>(0.0), vec3<f32>(1.0)) * 1023.0 + 0.5;
  return u32(q.x) | (u32(q.y) << 10u) | (u32(q.z) << 20u);
}

fn unpack_w10(v: u32) -> vec3<f32> {
  return vec3<f32>(f32(v & 1023u), f32((v >> 10u) & 1023u), f32((v >> 20u) & 1023u)) * (1.0 / 1023.0);
}

fn queue_refl_ray(org: vec3<f32>, dir: vec3<f32>, w: vec3<f32>, px: u32) {
  if (px == REFL_NO_PX) { return; }
  let i = atomicAdd(&refl_meta[0], 1u);
  if (i >= refl_cfg.x) { return; }
  let o = i * 2u;
  refl_req[o] = vec4<u32>(bitcast<u32>(org.x), bitcast<u32>(org.y), bitcast<u32>(org.z),
                          (REFL_KIND_RAY << 30u) | pack_w10(w));
  refl_req[o + 1u] = vec4<u32>(bitcast<u32>(dir.x), bitcast<u32>(dir.y), bitcast<u32>(dir.z), px);
}

fn queue_refl_chain(k0: u32, k1: u32, pal: u32, pbase: u32, face_id: u32,
                     w: vec3<f32>, px: u32) {
  if (px == REFL_NO_PX || k0 == 0u || k1 == 0u) { return; }
  let i = atomicAdd(&refl_meta[0], 1u);
  if (i >= refl_cfg.x) { return; }
  let o = i * 2u;
  refl_req[o] = vec4<u32>(k0, k1, (REFL_KIND_PACKED << 30u) | pack_w10(w), pal);
  refl_req[o + 1u] = vec4<u32>(pbase, face_id, 0u, px);
}

fn trace_glass(origin_voxel: vec3<f32>, dir_voxel: vec3<f32>, max_t: f32,
               entry_n: vec3<f32>, entry_ior: f32, entry_v: vec3<f32>, qpx: u32,
               quality: u32) -> GlassOut {
  var tint = vec3<f32>(1.0);
  let f0 = dielectric_f0(entry_ior, 1.0);
  let fres = f0 + (1.0 - f0) * pow(1.0 - clamp(dot(entry_v, entry_n), 0.0, 1.0), 5.0);

  let r_org = origin_voxel + entry_n * GLASS_EPS;
  let r_dir = reflect(-entry_v, entry_n);
  if (secondary_on() && light_u.g.refl_tier > 0u && fres > GLASS_REFL_MIN
      && dot(vec3<f32>(fres), GI_LUMA) > GLASS_REFL_WEIGHT_MIN) {
    queue_refl_ray(r_org, r_dir, vec3<f32>(fres), qpx);
  }
  if (quality == 0u) {
    return GlassOut(world_miss(), vec3<f32>(1.0), vec3<f32>(0.0), dir_voxel);
  }

  let eta_in = 1.0 / entry_ior;
  let ci_in = clamp(dot(-dir_voxel, entry_n), 0.0, 1.0);
  let k_in = 1.0 - eta_in * eta_in * (1.0 - ci_in * ci_in);
  var dir_in = normalize(eta_in * dir_voxel + (eta_in * ci_in - sqrt(max(k_in, 0.0))) * entry_n);
  if (quality < 2u) { dir_in = dir_voxel; }
  let inner = world_trace_glass_path(origin_voxel + dir_in * GLASS_EPS, dir_in, max_t, true, &tint);
  if (!inner.hit || medium_of(inner.palette_base, inner.pal).a >= 1.0) {
    return GlassOut(inner, tint, vec3<f32>(0.0), dir_in);
  }

  let n_out = select(-inner.n, inner.n, dot(dir_in, inner.n) < 0.0);
  let ci_out = clamp(dot(-dir_in, n_out), 0.0, 1.0);
  let k_out = 1.0 - entry_ior * entry_ior * (1.0 - ci_out * ci_out);
  let dir_bent = normalize(entry_ior * dir_in + (entry_ior * ci_out - sqrt(max(k_out, 0.0))) * n_out);
  let dir_out = select(dir_in, dir_bent, GLASS_EXIT_BEND && quality >= 2u);
  let outer = world_trace_glass_path(inner.point + (dir_out - n_out) * GLASS_EPS, dir_out, max_t, false, &tint);
  return GlassOut(outer, tint, vec3<f32>(0.0), dir_out);
}

fn write_output(coord: vec2<i32>, col: vec3<f32>) {
  let exposure = bitcast<f32>(eye_adapt_ro[0u]);
  let exposed = col * select(1.0, exposure, exposure > 0.0);
  textureStore(out_base, coord, vec4<f32>(linear_to_srgb(exposed) + dither3(coord), 1.0));
}

const GI_UPSAMPLE_HOLE_R: i32 = 1;

fn gi_upsample_joint(
  guv: vec2<f32>,
  gsize: vec2<u32>,
  k0: u32,
  k1: u32,
  n: vec3<f32>,
  fallback: vec3<f32>,
) -> vec4<f32> {
  let g = guv * vec2<f32>(f32(gsize.x), f32(gsize.y));
  let base = floor(g - vec2<f32>(0.5));
  let f = g - vec2<f32>(0.5) - base;
  var acc = vec3<f32>(0.0);
  var acc_a = 0.0;
  var wsum = 0.0;
  for (var j: i32 = 0; j < 2; j = j + 1) {
    for (var i: i32 = 0; i < 2; i = i + 1) {
      let c = vec2<i32>(base) + vec2<i32>(i, j);
      if (c.x < 0 || c.y < 0 || c.x >= i32(gsize.x) || c.y >= i32(gsize.y)) { continue; }
      let bw = select(1.0 - f.x, f.x, i == 1) * select(1.0 - f.y, f.y, j == 1);
      if (bw <= 1e-6) { continue; }
      let tb = (u32(c.y) * gsize.x + u32(c.x)) * GI_DEN_GUIDE_WORDS;
      let gk0 = gi_guide_out[tb + GI_DEN_G_KEY0];
      let gk1 = gi_guide_out[tb + GI_DEN_G_KEY1];
      if ((gk0 | gk1) == 0u) { continue; }
      if (!gi_same_surface(gk0, gk1, gi_guide_out[tb + GI_DEN_G_NRM], k0, k1, n)) { continue; }
      let s = textureLoad(gi_tex, c, 0);
      acc = acc + s.rgb * bw;
      acc_a = acc_a + s.a * bw;
      wsum = wsum + bw;
    }
  }
  if (wsum > 0.0) { return vec4<f32>(acc / wsum, acc_a / wsum); }

  let nc = clamp(base + vec2<f32>(select(0.0, 1.0, f.x >= 0.5), select(0.0, 1.0, f.y >= 0.5)),
                 vec2<f32>(0.0), vec2<f32>(f32(gsize.x) - 1.0, f32(gsize.y) - 1.0));
  let nci = vec2<i32>(nc);
  let sn = textureLoad(gi_tex, nci, 0);
  if (sn.a > 0.5) { return sn; }

  var best_ok = false;
  var best_w = 0.0;
  var best_acc = vec3<f32>(0.0);
  var best_a = 0.0;
  var any_ok = false;
  var any_d2 = 0;
  var any = vec4<f32>(0.0);
  for (var dy: i32 = -GI_UPSAMPLE_HOLE_R; dy <= GI_UPSAMPLE_HOLE_R; dy = dy + 1) {
    for (var dx: i32 = -GI_UPSAMPLE_HOLE_R; dx <= GI_UPSAMPLE_HOLE_R; dx = dx + 1) {
      let c = nci + vec2<i32>(dx, dy);
      if (c.x < 0 || c.y < 0 || c.x >= i32(gsize.x) || c.y >= i32(gsize.y)) { continue; }
      let tb = (u32(c.y) * gsize.x + u32(c.x)) * GI_DEN_GUIDE_WORDS;
      let gk0 = gi_guide_out[tb + GI_DEN_G_KEY0];
      let gk1 = gi_guide_out[tb + GI_DEN_G_KEY1];
      if ((gk0 | gk1) == 0u) { continue; }
      let d2 = dx * dx + dy * dy;
      if (!any_ok || d2 < any_d2) {
        any_ok = true;
        any_d2 = d2;
        any = textureLoad(gi_tex, c, 0);
      }
      if (gi_same_surface(gk0, gk1, gi_guide_out[tb + GI_DEN_G_NRM], k0, k1, n)) {
        let w = 1.0 / (1.0 + f32(d2));
        let s = textureLoad(gi_tex, c, 0);
        best_acc = best_acc + s.rgb * w;
        best_a = best_a + s.a * w;
        best_w = best_w + w;
        best_ok = true;
      }
    }
  }
  if (best_ok) { return vec4<f32>(best_acc / best_w, best_a / best_w); }
  if (any_ok) { return any; }
  return vec4<f32>(fallback, 0.0);
}

fn voxel_screen_uv(p_voxel: vec3<f32>, fallback: vec2<f32>) -> vec2<f32> {
  let clip = view_u.view_proj * vec4<f32>(p_voxel, 1.0);
  if (clip.w <= 1e-6) { return fallback; }
  let ndc = clip.xyz / clip.w;
  return vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
}

fn gi_vox_gi(voxel: vec3<i32>, face_id: u32, obj_id: i32) -> vec4<f32> {
  var ok = false;
  let k0 = gi_key0(voxel, obj_id);
  let k1 = gi_vox_k1(gi_key1(voxel, face_id, obj_id));
  let s = face_slot_find(k0, k1, &ok);
  if (!ok) {
    if (LOD_DIAG == 1u) {
      atomicAdd(&lod_diag[45u], 1u);
      var ok2 = false;
      let fe0 = face_slot_of(k0, k1, &ok2) * FACE_WORDS;
      if (atomicLoad(&face_slots[fe0 + FACE_W_FLAG]) == 0u) {
        atomicAdd(&lod_diag[102u], 1u);
      } else if (atomicLoad(&face_slots[fe0 + FACE_W_K0]) == k0) {
        atomicAdd(&lod_diag[103u], 1u);
      } else {
        atomicAdd(&lod_diag[104u], 1u);
      }
    }
    return vec4<f32>(0.0);
  }
  let fe = s * FACE_WORDS;
  let n = atomicLoad(&face_slots[fe + FACE_W_GI_N]);
  if (n == 0u) {
    if (LOD_DIAG == 1u) { atomicAdd(&lod_diag[46u], 1u); }
    return vec4<f32>(0.0);
  }
  return vec4<f32>(
    f32(atomicLoad(&face_slots[fe + FACE_W_GI])),
    f32(atomicLoad(&face_slots[fe + FACE_W_GI + 1u])),
    f32(atomicLoad(&face_slots[fe + FACE_W_GI + 2u])),
    f32(n));
}

struct GiInput {
  use_face: bool,
  face_avg: vec3<f32>,
  face_cov: f32,
  uv: vec2<f32>,
}

fn refl_seed(voxel: vec3<i32>, face_id: u32, obj_id: i32) -> u32 {
  let rkey = face_id | ((u32(obj_id) + 1u) << 3u);
  return gi_mix(
    u32(voxel.x) * 0x9E3779B1u
    + gi_mix(u32(voxel.y) * 0x85EBCA77u
    + gi_mix(u32(voxel.z) * 0xC2B2AE3Du
    + gi_mix(rkey * 0x27D4EB2Fu + 0x165667B1u))));
}

fn refl_env_chain(p_voxel: vec3<f32>, n: vec3<f32>, v: vec3<f32>, rseed: u32, m: MaterialParams,
                  frustum_length: f32, n_off: f32) -> vec3<f32> {
  let sun_dir = light_u.lights[0].kind_pos_dir.yzw;
  let sun_c = light_u.lights[0].color_intensity.xyz * light_u.lights[0].color_intensity.w;
  let sky = light_u.sky_color.xyz;
  let mirror = reflect(-v, n);
  var rdir = refl_cone_dir(mirror, gi_rand2(rseed), PBR_REFL_CONE_K * m.roughness);

  if (dot(rdir, n) <= 0.0) { rdir = mirror; }
  let r_org = p_voxel + n * n_off;
  if (!secondary_on()) { return vec3<f32>(0.0); }
  let reach = secondary_reach(frustum_length);
  let tier = light_u.g.refl_tier;
  let nest = light_u.g.refl_nest;
  var refl_env = vec3<f32>(0.0);
  var chain_w = vec3<f32>(1.0);
  var chain_org = r_org;
  var chain_dir = rdir;
  for (var lvl = 0u; lvl <= nest; lvl = lvl + 1u) {
    var r_tint = vec3<f32>(1.0);
    let h = world_trace_medium_scene(chain_org, chain_dir, reach, false, &r_tint);
    if (!h.hit) {
      refl_env = refl_env
               + chain_w * r_tint
                 * select(sky_rgb(), sky_primary(chain_dir, sun_dir, sun_c), tier >= 2u);
      break;
    }
    let hgg = make_grid(u32(h.obj_id) + 1u);
    let hn = voxel_normal_fast(hgg, h.voxel, h.face_id, h.n);
    let hm = fetch_material(h.palette_base, h.pal, h.point, hn, PBR_DEBUG_ASSET);
    let hamb = sky * GI_SKY_AMBIENT;
    let hv = face_view_dir(chain_org, voxel_center_world(hgg, h.voxel));
    let h_start = h.point + hn * surface_offset(hgg, hn, SHADOW_SURFACE_EPS);
    var h_vis = vec3<f32>(1.0);

    if (tier >= 3u && (light_u.g.base_flags & BASE_FLAG_SHADOW) != 0u
        && dot(hn, sun_dir) > 0.0) {
      h_vis = sun_transmittance(h_start, sun_dir, h.voxel, h.obj_id);
    }
    let hf0 = f0_of(hm);
    let h_fade = clamp(1.0 - hm.roughness / PBR_REFL_MAX_ROUGH, 0.0, 1.0);
    let h_continue =
      lvl < nest && h_fade > 0.0
      && dot(hf0, GI_LUMA) >= PBR_REFL_MIN_F0;
    var h_local = brdf_reflected(hm, hn, hv,
                                 SurfaceLight(sun_dir, sun_c * h_vis, 1.0, hamb, vec3<f32>(0.0)))
                + emissive_radiance(hm, GI_EMIT_GAIN);
    if (h_continue) {
      h_local = h_local - hf0 * hamb;
    }
    refl_env = refl_env + chain_w * r_tint * h_local;
    if (!h_continue) { break; }
    let h_mirror_dir = reflect(-hv, hn);
    var h_dir = refl_cone_dir(h_mirror_dir, gi_rand2(gi_mix(rseed + lvl * 0x9E3779B1u)),
                              PBR_REFL_CONE_K * hm.roughness);
    if (dot(h_dir, hn) <= 0.0) { h_dir = h_mirror_dir; }
    chain_org = h_start;
    chain_dir = h_dir;
    chain_w = chain_w * fresnel_schlick(hf0, max(dot(hv, hn), 0.0)) * h_fade;
  }
  return refl_env;
}

fn shade_face(first: WorldHit, origin_voxel: vec3<f32>, ray_dir: vec3<f32>, frustum_length: f32,
              gi_in: GiInput, sun_dir: vec3<f32>, sky: vec3<f32>, sun_c: vec3<f32>,
              n_hint: vec3<f32>, n_hint_ok: bool,
              qpx: u32) -> vec3<f32> {
  var best = first;
  var through = vec3<f32>(1.0);
  var refl = vec3<f32>(0.0);
  var glass_col = vec3<f32>(0.0);
  var glass_w = 0.0;
  var sky_dir = ray_dir;
  let glass_q = (light_u.g.base_flags >> GLASS_QUALITY_SHIFT) & 3u;
  if (first.hit && (palette_flags(first.palette_base, first.pal) & FLAGS_TRANSMISSIVE) != 0u) {
    let fg = make_grid(u32(first.obj_id) + 1u);
    let fp = voxel_center_world(fg, first.voxel);
    let fvuv = voxel_screen_uv(fp, gi_in.uv);
    var fnn = n_hint;
    if (!n_hint_ok) { fnn = voxel_normal_fast(fg, first.voxel, first.face_id, first.n); }
    let f_off = surface_offset(fg, fnn, SHADOW_SURFACE_EPS);

    let f_face = fp + fnn * f_off;
    let fv = face_view_dir(origin_voxel, fp);
    let g = trace_glass(f_face, -fv, frustum_length, fnn, material_ior(first.palette_base, first.pal), fv, qpx,
                        glass_q);
    let fm = fetch_material(first.palette_base, first.pal, fp, fnn, PBR_DEBUG_ASSET);
    if (glass_q > 0u) {
      best = g.ray;
      through = g.tint;
      refl = g.refl;
      sky_dir = g.dir;
      glass_w = clamp(1.0 - fm.transmission, 0.0, 1.0);
    } else {
      glass_w = 1.0;
    }
    if (glass_w > 1.0 / 255.0) {
      let f_ndl = max(dot(fnn, sun_dir), 0.0);
      var f_sun = vec3<f32>(0.0);
      if (f_ndl > 0.0) {
        f_sun = sun_transmittance(f_face, sun_dir, first.voxel, first.obj_id);
      }
      var f_gi = vec3<f32>(0.0);
      var f_cov = 0.0;
      if (gi_in.use_face) {
        f_gi = gi_in.face_avg;
        f_cov = gi_in.face_cov;
      } else if (gi_u.misc.x > 0.5) {
        let fs = textureSampleLevel(gi_tex, light_samp, fvuv, 0.0);
        let fgw = max(fs.a, 1e-4);
        f_gi = fs.rgb / fgw;
        f_cov = clamp(fs.a, 0.0, 1.0);
      }
      let f_amb = sky * GI_SKY_AMBIENT * mix(1.0, GI_AMBIENT_FLOOR, f_cov);
      glass_col = brdf_reflected(fm, fnn, fv,
                                 SurfaceLight(sun_dir, sun_c * f_sun, 1.0, f_amb, f_gi))
                + emissive_radiance(fm, GI_EMIT_GAIN);
    }
  }
  var col = vec3<f32>(0.0);
  if (best.hit) {
    let gg = make_grid(u32(best.obj_id) + 1u);
    let p_voxel = voxel_center_world(gg, best.voxel);
    let vuv = voxel_screen_uv(p_voxel, gi_in.uv);
    if ((light_u.g.base_flags & BASE_FLAG_IMPLICIT_NORMAL) == 0u) {
      return fetch_material(best.palette_base, best.pal, best.point, best.n, PBR_DEBUG_ASSET).albedo;
    }
    let k0 = gi_key0(best.voxel, best.obj_id);
    let k1 = gi_key1(best.voxel, best.face_id, best.obj_id);
    if (gi_u.params.y > 0.5 && gi_key_ok(best.voxel, best.obj_id)) {
      gi_wal_touch(best.voxel, best.obj_id, best.pal, k0, k1);
    }
    var n = best.n;
    var n_reused = false;
    if (n_hint_ok) {
      n = n_hint;
      n_reused = true;
    } else if (u32(gi_u.flags.y) == 1u) {
      let gi_size = vec2<u32>(textureDimensions(gi_tex));
      let tp = vec2<i32>(floor(gi_in.uv * vec2<f32>(f32(gi_size.x), f32(gi_size.y))));
      if (tp.x >= 0 && tp.y >= 0 && tp.x < i32(gi_size.x) && tp.y < i32(gi_size.y)) {
        let gb = (u32(tp.y) * gi_size.x + u32(tp.x)) * GI_DEN_GUIDE_WORDS;
        if (gi_guide_out[gb + GI_DEN_G_KEY0] == k0 && gi_guide_out[gb + GI_DEN_G_KEY1] == k1) {
          n = gi_unpack_dir(gi_guide_out[gb + GI_DEN_G_NRM]);
          n_reused = true;
        }
      }
    }
    if (!n_reused) { n = voxel_normal_fast(gg, best.voxel, best.face_id, best.n); }
    var m = fetch_material(best.palette_base, best.pal, p_voxel, n, PBR_DEBUG_ASSET);
    let v = face_view_dir(origin_voxel, p_voxel);
    let n_off = surface_offset(gg, n, SHADOW_SURFACE_EPS);
    let ndl = max(dot(n, sun_dir), 0.0);
    var sun = vec3<f32>(0.0);
    if (ndl > 0.0) {
      sun = vec3<f32>(1.0);
      if ((light_u.g.base_flags & BASE_FLAG_SHADOW) != 0u) {
        sun = sun_transmittance(p_voxel + n * n_off, sun_dir, best.voxel, best.obj_id);
      }
    }
    let gi_on = gi_u.misc.x > 0.5;
    var gi = vec3<f32>(0.0);
    var cov = 0.0;
    if (gi_in.use_face) {
      gi = gi_in.face_avg;
      cov = gi_in.face_cov;
    } else if (gi_on) {
      var acc = gi_vox_gi(best.voxel, best.face_id, best.obj_id);
      acc = acc * vec4<f32>(2.0, 2.0, 2.0, 2.0);
      if (acc.w > 0.0) {
        var t1 = vec3<i32>(1, 0, 0);
        var t2 = vec3<i32>(0, 1, 0);
        if (best.face_id / 2u == 0u) {
          t1 = vec3<i32>(0, 1, 0);
          t2 = vec3<i32>(0, 0, 1);
        } else if (best.face_id / 2u == 1u) {
          t1 = vec3<i32>(1, 0, 0);
          t2 = vec3<i32>(0, 0, 1);
        }
        for (var i = 0u; i < 4u; i = i + 1u) {
          let s1 = select(-1, 1, (i & 1u) == 0u);
          let s2 = select(-1, 1, (i & 2u) == 0u);
          let nv = best.voxel + t1 * s1 + t2 * s2;
          if (!gi_key_ok(nv, best.obj_id)) { continue; }
          acc = acc + gi_vox_gi(nv, best.face_id, best.obj_id);
        }
      }
      if (acc.w > 0.0) {
        if (LOD_DIAG == 1u) { atomicAdd(&lod_diag[44u], 1u); }
        gi = acc.xyz / (acc.w * GI_ACC_SCALE);
        cov = 1.0;
      } else {
        if (LOD_DIAG == 1u) { atomicAdd(&lod_diag[42u], 1u); }
        let gup = vec2<u32>(textureDimensions(gi_tex));
        let s = gi_upsample_joint(vuv, gup, k0, k1, n, vec3<f32>(0.0));
        let gw = max(s.a, 1e-4);
        gi = s.rgb / gw;
        cov = clamp(s.a, 0.0, 1.0);
      }
      let gsize = vec2<u32>(textureDimensions(gi_tex));
      if (LOD_DIAG == 1u) {
        let tp = vec2<i32>(floor(vuv * vec2<f32>(gsize)));
        if (tp.x >= 0 && tp.y >= 0 && tp.x < i32(gsize.x) && tp.y < i32(gsize.y)) {
          let gb = (u32(tp.y) * gsize.x + u32(tp.x)) * GI_DEN_GUIDE_WORDS;
          let gk0 = gi_guide_out[gb + GI_DEN_G_KEY0];
          let gk1 = gi_guide_out[gb + GI_DEN_G_KEY1];
          if ((gk0 | gk1) != 0u) {
            atomicAdd(&lod_diag[select(101u, 100u, gk0 == k0 && gk1 == k1)], 1u);
          }
        }
      }
      if (LOD_DIAG == 1u) {
        let osz = textureDimensions(out_base);
        let gxp = qpx % osz.x;
        let gyp = qpx / osz.x;
        let gdv = max(u32(gi_u.flags.y), 1u);
        if ((gxp % gdv) == gdv / 2u && (gyp % gdv) == gdv / 2u) {
          let qtx = gxp / gdv;
          let qty = gyp / gdv;
          if (qtx < gsize.x && qty < gsize.y) {
            let qb = (qty * gsize.x + qtx) * GI_DEN_GUIDE_WORDS;
            let qk0 = gi_guide_out[qb + GI_DEN_G_KEY0];
            let qk1 = gi_guide_out[qb + GI_DEN_G_KEY1];
            if ((qk0 | qk1) != 0u) {
              let same_c = qk0 == k0 && gi_vox_k1(qk1) == gi_vox_k1(k1);
              atomicAdd(&lod_diag[select(111u, 110u, same_c)], 1u);
              let tpx = (vec2<f32>(f32(qtx), f32(qty)) + vec2<f32>(0.5)) / vec2<f32>(f32(gsize.x), f32(gsize.y));
              let tuv = vec2<f32>(tpx.x * 2.0 - 1.0, 1.0 - tpx.y * 2.0);
              let tnw4 = view_u.inv_view_proj * vec4<f32>(tuv.x, tuv.y, 0.0, 1.0);
              let tfw4 = view_u.inv_view_proj * vec4<f32>(tuv.x, tuv.y, 1.0, 1.0);
              let tdiff = tfw4.xyz / tfw4.w - tnw4.xyz / tnw4.w;
              var tt_min = 0.0;
              if (view_u.lod.z < 0.5) {
                let bsz = textureDimensions(beam_depth);
                let bxg = qtx / max(BEAM_DIV / gdv, 1u);
                let byg = qty / max(BEAM_DIV / gdv, 1u);
                tt_min = 1e+30;
                for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
                  for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
                    let sx = clamp(i32(bxg) + dx, 0, i32(bsz.x) - 1);
                    let sy = clamp(i32(byg) + dy, 0, i32(bsz.y) - 1);
                    tt_min = min(tt_min, textureLoad(beam_depth, vec2<i32>(sx, sy)).x);
                  }
                }
                tt_min = max(tt_min - BEAM_BACKOFF, 0.0);
              }
              let hr = world_raycast(make_grid(0u), view_u.cam_pos_voxel.xyz, normalize(tdiff), length(tdiff), world_cfg_full(tt_min, 3u));
              let rmatch = hr.hit && gi_key0(hr.voxel, hr.obj_id) == qk0
                && gi_vox_k1(gi_key1(hr.voxel, hr.face_id, hr.obj_id)) == gi_vox_k1(qk1);
              atomicAdd(&lod_diag[select(113u, 112u, rmatch)], 1u);
              let fmatch = first.hit && gi_key0(first.voxel, first.obj_id) == qk0
                && gi_vox_k1(gi_key1(first.voxel, first.face_id, first.obj_id)) == gi_vox_k1(qk1);
              atomicAdd(&lod_diag[select(115u, 114u, fmatch)], 1u);
              if (first.hit && (palette_flags(first.palette_base, first.pal) & FLAGS_TRANSMISSIVE) != 0u) {
                atomicAdd(&lod_diag[116u], 1u);
              }
              if (first.hit && hr.hit) {
                let dpt = length(first.point - hr.point);
                atomicAdd(&lod_diag[select(118u, 117u, dpt < 1.0)], 1u);
                atomicAdd(&lod_diag[select(120u, 119u, dot(first.n, hr.n) > 0.9)], 1u);
              }
            }
          }
        }
      }
    }
    let amb = sky * GI_SKY_AMBIENT * mix(1.0, GI_AMBIENT_FLOOR, cov);
    if (LOD_DIAG == 1u) { dbg_gi = gi; }
    col = brdf_reflected(m, n, v, SurfaceLight(sun_dir, sun_c * sun, 1.0, amb, gi))
        + emissive_radiance(m, GI_EMIT_GAIN);
    if (PBR_REFLECTION_ENABLED != 0u && light_u.g.refl_tier > 0u) {
      let tier = light_u.g.refl_tier;
      let f0r = f0_of(m);
      let f0_luma = dot(f0r, GI_LUMA);
      if (f0_luma >= PBR_REFL_MIN_F0 && m.roughness <= PBR_REFL_MAX_ROUGH) {
        let fade = clamp(1.0 - m.roughness / PBR_REFL_MAX_ROUGH, 0.0, 1.0);
        col = col - fade * f0r * (amb + gi);
        queue_refl_chain(gi_key0(best.voxel, best.obj_id), gi_key1(best.voxel, best.face_id, best.obj_id),
                         best.pal, best.palette_base, best.face_id,
                         fresnel_schlick(f0r, max(dot(v, n), 0.0)) * fade, qpx);
      }
    }
    col = col * through + refl;
  } else {
    return mix(glass_col, sky_primary(sky_dir, sun_dir, sun_c) * through + refl, 1.0 - glass_w);
  }
  return mix(glass_col, col, 1.0 - glass_w);
}

const DIAG_FACE_SKY: u32 = 8u;
const DIAG_MAIN_RAYS: u32 = 9u;
const DIAG_MAIN_NEAR: u32 = 10u;
const DIAG_MAIN_FAR: u32 = 11u;
const DIAG_MAIN_SKY: u32 = 12u;
const DIAG_GI_WAL_MATURE: u32 = 23u;
const DIAG_GI_WAL_FRESH: u32 = 24u;
const DIAG_WAL_FACES: u32 = 25u;
const DIAG_WAL_NSUM: u32 = 26u;
const DIAG_GI_MEAN_NEAR: u32 = 28u;
const DIAG_GI_MEAN_FAR: u32 = 29u;
const DIAG_LUMA_NEAR: u32 = 30u;
const DIAG_LUMA_FAR: u32 = 31u;

var<workgroup> cam_pal_wg: u32;

@compute @workgroup_size(8, 8, 1)
fn dda_main(@builtin(global_invocation_id) gid: vec3<u32>,
            @builtin(local_invocation_index) lid: u32) {
  if (lid == 0u) {
    cam_pal_wg = camera_solid_pal(view_u.cam_pos_voxel.xyz);
  }
  workgroupBarrier();
  let size = textureDimensions(out_base);
  if (gid.x >= size.x || gid.y >= size.y) { return; }
  let coord0 = vec2<i32>(i32(gid.x), i32(gid.y));
  let cam_pal = cam_pal_wg;
  if (cam_pal != 0u) {
    write_output(coord0, palette_rgb_linear(make_grid(0u).palette_base, cam_pal));
    return;
  }
  let px = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5)) / vec2<f32>(f32(size.x), f32(size.y));
  let uv = vec2<f32>(px.x * 2.0 - 1.0, 1.0 - px.y * 2.0);
  let near_ndc = vec4<f32>(uv.x, uv.y, 0.0, 1.0);
  let far_ndc  = vec4<f32>(uv.x, uv.y, 1.0, 1.0);
  let near_world_h = view_u.inv_view_proj * near_ndc;
  let far_world_h  = view_u.inv_view_proj * far_ndc;
  let near_world = near_world_h.xyz / near_world_h.w;
  let far_world  = far_world_h.xyz  / far_world_h.w;
  let diff_world = far_world - near_world;
  let frustum_length = length(diff_world);
  let dir_voxel = normalize(diff_world);
  let origin_voxel = view_u.cam_pos_voxel.xyz;
  var t_min = 0.0;
  if (view_u.lod.z < 0.5) {
    let beam_size = textureDimensions(beam_depth);
    let bx = gid.x / BEAM_DIV;
    let by = gid.y / BEAM_DIV;
    t_min = 1e+30;
    for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
      for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
        let sx = clamp(i32(bx) + dx, 0, i32(beam_size.x) - 1);
        let sy = clamp(i32(by) + dy, 0, i32(beam_size.y) - 1);
        let bt = textureLoad(beam_depth, vec2<i32>(sx, sy)).x;
        t_min = min(t_min, bt);
      }
    }
    t_min = max(t_min - BEAM_BACKOFF, 0.0);
  }
  let sun_dir = light_u.lights[0].kind_pos_dir.yzw;
  let sky = light_u.sky_color.xyz;
  let sun_c = light_u.lights[0].color_intensity.xyz * light_u.lights[0].color_intensity.w;
  let first = world_raycast(make_grid(0u), origin_voxel, dir_voxel, frustum_length,
                            world_cfg_primary(t_min, 3u));
  dda_hit_record(gid, first, origin_voxel);
  if (LOD_DIAG == 1u && first.hit) {
    let f2 = world_raycast(make_grid(0u), origin_voxel, dir_voxel, frustum_length, world_cfg_full(t_min, 3u));
    let same2 = f2.hit && all(f2.voxel == first.voxel) && f2.obj_id == first.obj_id;
    atomicAdd(&lod_diag[select(106u, 105u, same2)], 1u);
  }
  let need_n = (light_u.g.base_flags & BASE_FLAG_IMPLICIT_NORMAL) != 0u;
  var n_own = vec3<f32>(0.0);
  if (first.hit && need_n) {
    let first_xpar = (palette_flags(first.palette_base, first.pal) & FLAGS_TRANSMISSIVE) != 0u;
    let fgrid = make_grid(u32(first.obj_id) + 1u);
    n_own = select(voxel_normal_fast(fgrid, first.voxel, first.face_id, first.n), first.n, first_xpar);
  }
  let diag_pix = gid.x + gid.y * size.x;
  let diag_on = diag_sampled(diag_pix);
  if (diag_on) {
    atomicAdd(&lod_diag[DIAG_MAIN_RAYS], 1u);
    if (!first.hit) {
      atomicAdd(&lod_diag[DIAG_MAIN_SKY], 1u);
    } else if (first.obj_id < 0) {
      atomicAdd(&lod_diag[DIAG_MAIN_NEAR], 1u);
    } else {
      atomicAdd(&lod_diag[DIAG_MAIN_FAR], 1u);
    }
  }
  var col = vec3<f32>(0.0);
  if (diag_on && !first.hit) { atomicAdd(&lod_diag[DIAG_FACE_SKY], 1u); }
  col = shade_face(first, origin_voxel, dir_voxel, frustum_length,
                     GiInput(false, vec3<f32>(0.0), 0.0, px), sun_dir, sky, sun_c,
                     n_own, need_n, u32(gid.y) * size.x + u32(gid.x));
  if (diag_on && first.hit) {
    let q = u32(clamp(dot(dbg_gi, GI_LUMA), 0.0, 255.0) * 256.0);
    let p = u32(clamp(dot(col, GI_LUMA), 0.0, 255.0) * 256.0);
    if (first.obj_id < 0) {
      atomicAdd(&lod_diag[DIAG_GI_MEAN_NEAR], q);
      atomicAdd(&lod_diag[DIAG_LUMA_NEAR], p);
    } else {
      atomicAdd(&lod_diag[DIAG_GI_MEAN_FAR], q);
      atomicAdd(&lod_diag[DIAG_LUMA_FAR], p);
    }
  }
  col = col + godray_sample_bilinear(fog_tex, vec2<f32>(textureDimensions(fog_tex)), px);
  if ((light_u.g.base_flags & BASE_FLAG_VOL_TINT) != 0u) {
    col = vol_tint_color(first);
  }
  write_output(coord0, col);
}

@compute @workgroup_size(8, 8, 1)
fn dda_preview(@builtin(global_invocation_id) gid: vec3<u32>,
               @builtin(local_invocation_index) lid: u32) {
  if (lid == 0u) {
    cam_pal_wg = camera_solid_pal(view_u.cam_pos_voxel.xyz);
  }
  workgroupBarrier();
  let size = textureDimensions(out_base);
  if (gid.x >= size.x || gid.y >= size.y) { return; }
  let coord0 = vec2<i32>(i32(gid.x), i32(gid.y));
  if (cam_pal_wg != 0u) {
    write_output(coord0, palette_rgb_linear(make_grid(0u).palette_base, cam_pal_wg));
    return;
  }
  let px = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5)) / vec2<f32>(f32(size.x), f32(size.y));
  let uv = vec2<f32>(px.x * 2.0 - 1.0, 1.0 - px.y * 2.0);
  let near_world_h = view_u.inv_view_proj * vec4<f32>(uv.x, uv.y, 0.0, 1.0);
  let far_world_h = view_u.inv_view_proj * vec4<f32>(uv.x, uv.y, 1.0, 1.0);
  let near_world = near_world_h.xyz / near_world_h.w;
  let far_world = far_world_h.xyz / far_world_h.w;
  let diff_world = far_world - near_world;
  let frustum_length = length(diff_world);
  let dir_voxel = normalize(diff_world);
  let origin_voxel = view_u.cam_pos_voxel.xyz;
  var t_min = 0.0;
  if (view_u.lod.z < 0.5) {
    let beam_size = textureDimensions(beam_depth);
    let bx = gid.x / BEAM_DIV;
    let by = gid.y / BEAM_DIV;
    t_min = 1e+30;
    for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
      for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
        let sx = clamp(i32(bx) + dx, 0, i32(beam_size.x) - 1);
        let sy = clamp(i32(by) + dy, 0, i32(beam_size.y) - 1);
        t_min = min(t_min, textureLoad(beam_depth, vec2<i32>(sx, sy)).x);
      }
    }
    t_min = max(t_min - BEAM_BACKOFF, 0.0);
  }
  let first = world_raycast(make_grid(0u), origin_voxel, dir_voxel, frustum_length,
                            world_cfg_primary(t_min, 3u));
  dda_hit_record(gid, first, origin_voxel);
  let diag_pix = gid.x + gid.y * size.x;
  if (diag_sampled(diag_pix)) {
    atomicAdd(&lod_diag[DIAG_MAIN_RAYS], 1u);
    if (!first.hit) {
      atomicAdd(&lod_diag[DIAG_MAIN_SKY], 1u);
    } else if (first.obj_id < 0) {
      atomicAdd(&lod_diag[DIAG_MAIN_NEAR], 1u);
    } else {
      atomicAdd(&lod_diag[DIAG_MAIN_FAR], 1u);
    }
  }
  let sun_dir = light_u.lights[0].kind_pos_dir.yzw;
  let sky = light_u.sky_color.xyz;
  let sun_c = light_u.lights[0].color_intensity.xyz * light_u.lights[0].color_intensity.w;
  var col = sky_rgb();
  if (first.hit) {
    let gg = make_grid(u32(first.obj_id) + 1u);
    let n = voxel_normal_fast(gg, first.voxel, first.face_id, first.n);
    let albedo = palette_rgb_linear(first.palette_base, first.pal);
    col = albedo * (sky * GI_SKY_AMBIENT + sun_c * max(dot(n, sun_dir), 0.0));
  }
  write_output(coord0, col);
}

const FACE_ACC_WG: u32 = 64u;
const FACE_ACC_NONE: u32 = 0xFFFFFFFFu;

var<workgroup> fa_slot: array<u32, FACE_ACC_WG>;
var<workgroup> fa_val: array<vec3<u32>, FACE_ACC_WG>;

@compute @workgroup_size(8, 8, 1)
fn dda_face_accum(@builtin(global_invocation_id) gid: vec3<u32>,
                  @builtin(local_invocation_index) lid: u32) {
  var slot = FACE_ACC_NONE;
  var val = vec3<u32>(0u);
  let gsize = textureDimensions(gi_tex);
  if (gid.x < gsize.x && gid.y < gsize.y) {
    let texel = gid.y * gsize.x + gid.x;
    let gb = texel * GI_DEN_GUIDE_WORDS;
    let k0 = gi_guide_out[gb + GI_DEN_G_KEY0];
    let k1 = gi_guide_out[gb + GI_DEN_G_KEY1];
    if ((k0 | k1) != 0u) {
      var ok = false;
      let s = face_slot_claim(k0, gi_vox_k1(k1), &ok);
      if (ok) {
        let fe = s * FACE_WORDS;
        let guv = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5))
                / vec2<f32>(f32(gsize.x), f32(gsize.y));
        let smp = textureSampleLevel(gi_tex, light_samp, guv, 0.0);
        if (smp.a > 0.0) {
          let q = clamp(smp.rgb / max(smp.a, 1e-4), vec3<f32>(0.0), vec3<f32>(GI_ACC_CLAMP));
          val = vec3<u32>(round(q * GI_ACC_SCALE));
          slot = s;
        }
      }
    }
  }
  fa_slot[lid] = slot;
  fa_val[lid] = val;
  workgroupBarrier();
  if (slot == FACE_ACC_NONE) { return; }
  for (var j = 0u; j < lid; j = j + 1u) {
    if (fa_slot[j] == slot) { return; }
  }
  var sum = vec3<u32>(0u);
  var cnt = 0u;
  for (var j = 0u; j < FACE_ACC_WG; j = j + 1u) {
    if (fa_slot[j] == slot) {
      sum = sum + fa_val[j];
      cnt = cnt + 1u;
    }
  }
  let fe = slot * FACE_WORDS;
  loop {
    let c = atomicLoad(&face_slots[fe + FACE_W_GI_N]);
    if (c + cnt > GI_ACC_MAX_N) { return; }
    if (atomicCompareExchangeWeak(&face_slots[fe + FACE_W_GI_N], c, c + cnt).exchanged) { break; }
  }
  atomicAdd(&face_slots[fe + FACE_W_GI], sum.x);
  atomicAdd(&face_slots[fe + FACE_W_GI + 1u], sum.y);
  atomicAdd(&face_slots[fe + FACE_W_GI + 2u], sum.z);
}

@compute @workgroup_size(8, 8, 1)
fn dda_face_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let gsize = textureDimensions(gi_tex);
  if (gid.x >= gsize.x || gid.y >= gsize.y) { return; }
  let texel = gid.y * gsize.x + gid.x;
  let gb = texel * GI_DEN_GUIDE_WORDS;
  let k0 = gi_guide_out[gb + GI_DEN_G_KEY0];
  let k1 = gi_guide_out[gb + GI_DEN_G_KEY1];
  if ((k0 | k1) == 0u) { return; }
  var ok = false;
  let slot = face_slot_find(k0, gi_vox_k1(k1), &ok);
  if (!ok) { return; }
  let fe = slot * FACE_WORDS;
  if (atomicLoad(&face_slots[fe + FACE_W_TEXEL]) != texel) { return; }
  let obj_id = i32((k1 >> 19u) & 0x1FFFu) - 1;
  let voxel = gi_unkey_voxel(k0, k1, obj_id);
  if (diag_sampled(texel)) {
    atomicAdd(&lod_diag[DIAG_WAL_FACES], 1u);
    if (gi_u.params.y > 0.5) {
      let wp = gi_wal_pool(voxel, obj_id, atomicLoad(&face_slots[fe + FACE_W_PAL]), k1);
      if (wp.w > 0.0) {
        atomicAdd(&lod_diag[DIAG_WAL_NSUM], u32(min(wp.w, 1.0e7)));
        if (wp.w >= GI_WAL_MATURE_N) {
          atomicAdd(&lod_diag[DIAG_GI_WAL_MATURE], 1u);
        } else {
          atomicAdd(&lod_diag[DIAG_GI_WAL_FRESH], 1u);
        }
      }
    }
  }
  let gi_n = atomicLoad(&face_slots[fe + FACE_W_GI_N]);
  if (gi_n >= GI_ACC_MAX_N) {
    for (var i = 0u; i < 3u; i = i + 1u) {
      let w = fe + FACE_W_GI + i;
      atomicStore(&face_slots[w], atomicLoad(&face_slots[w]) >> 1u);
    }
    atomicStore(&face_slots[fe + FACE_W_GI_N], gi_n >> 1u);
  }
}

var<workgroup> gi_adapt_wg: array<f32, 64u>;
var<workgroup> gi_adapt_r: array<f32, 64u>;

@compute @workgroup_size(8, 8, 1)
fn gi_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let size = textureDimensions(gi_out);
  let in_range = gid.x < size.x && gid.y < size.y;
  let ccoord = vec2<i32>(i32(min(gid.x, size.x - 1u)), i32(min(gid.y, size.y - 1u)));
  let cidx = u32(ccoord.y) * size.x + u32(ccoord.x);
  let lid = (gid.y & 7u) * 8u + (gid.x & 7u);
  let pu = unpack2x16float(gi_guide_out[cidx * GI_DEN_GUIDE_WORDS + GI_DEN_G_NOISE]);
  var s = pu.x;
  if (!(s > 0.0)) { s = GI_SS_ADAPT_REF; }
  var rbase = pu.y;
  if (!(rbase > 0.0)) { rbase = 1.0; }
  gi_adapt_wg[lid] = s;
  gi_adapt_r[lid] = rbase;
  if (lid == 0u) { cam_pal_wg = camera_solid_pal(view_u.cam_pos_voxel.xyz); }
  workgroupBarrier();
  var acc_s = 0.0;
  var acc_r = 0.0;
  for (var i = 0u; i < 64u; i = i + 1u) {
    acc_s = acc_s + gi_adapt_wg[i];
    acc_r = acc_r + gi_adapt_r[i];
  }
  let w_raw = clamp(s / max(acc_s / 64.0, 1e-5), GI_SS_ADAPT_LO, GI_SS_ADAPT_HI);
  gi_adapt_wg[lid] = w_raw * rbase;
  workgroupBarrier();
  var acc_wr = 0.0;
  for (var i = 0u; i < 64u; i = i + 1u) { acc_wr = acc_wr + gi_adapt_wg[i]; }
  let w_adapt = w_raw * acc_r / max(acc_wr, 1e-5);
  if (!in_range) { return; }
  let coord = vec2<i32>(i32(gid.x), i32(gid.y));
  let px = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5))
         / vec2<f32>(f32(size.x), f32(size.y));
  let uv = vec2<f32>(px.x * 2.0 - 1.0, 1.0 - px.y * 2.0);
  let near_ndc = vec4<f32>(uv.x, uv.y, 0.0, 1.0);
  let far_ndc  = vec4<f32>(uv.x, uv.y, 1.0, 1.0);
  let near_world_h = view_u.inv_view_proj * near_ndc;
  let far_world_h  = view_u.inv_view_proj * far_ndc;
  let near_world = near_world_h.xyz / near_world_h.w;
  let far_world  = far_world_h.xyz  / far_world_h.w;
  let diff_world = far_world - near_world;
  let frustum_length = length(diff_world);
  let dir_voxel = normalize(diff_world);
  let origin_voxel = view_u.cam_pos_voxel.xyz;
  if (cam_pal_wg != 0u) {
    gi_ss_main(coord, size, world_miss(), w_adapt);
    return;
  }
  
  
  var h = world_miss();
  let raw = textureLoad(dda_hit, coord);
  let age_ok = (raw.w >> 24u) == (gi_u.seq.x & 0xFFu);
  let hit_ok = (raw.w & 1u) != 0u && ((raw.w >> 1u) & 0x7FFFFFu) != 0u;
  if (age_ok && hit_ok) {
    h = dda_hit_unpack(raw, origin_voxel, dir_voxel);
  } else if (age_ok) {
    let gi_div = max(u32(gi_u.flags.y), 1u);
    var t_min = 0.0;
    if (view_u.lod.z < 0.5) {
      let beam_size = textureDimensions(beam_depth);
      let bx = gid.x / max(BEAM_DIV / gi_div, 1u);
      let by = gid.y / max(BEAM_DIV / gi_div, 1u);
      t_min = 1e+30;
      for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
        for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
          let sx = clamp(i32(bx) + dx, 0, i32(beam_size.x) - 1);
          let sy = clamp(i32(by) + dy, 0, i32(beam_size.y) - 1);
          t_min = min(t_min, textureLoad(beam_depth, vec2<i32>(sx, sy)).x);
        }
      }
      t_min = max(t_min - BEAM_BACKOFF, 0.0);
    }
    h = world_raycast(make_grid(0u), origin_voxel, dir_voxel, frustum_length,
                      world_cfg_full(t_min, 3u));
  }
  gi_ss_main(coord, size, h, w_adapt);
}

@compute @workgroup_size(64, 1, 1)
fn gi_wal_update(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gi_u.params.y <= 0.5) { return; }
  if (GI_WAL_UPD_R == 0u) { return; }
  let frame = gi_u.seq.x;
  let slot = (frame * GI_WAL_UPD_B + gid.x) & gi_wal_nmask();
  let b = slot * GI_WAL_WORDS;
  let k0 = atomicLoad(&gi_wal[b]);
  let k1 = atomicLoad(&gi_wal[b + 1u]);
  if ((k0 | k1) == 0u) { return; }
  let stamp = atomicLoad(&gi_wal[b + 7u]) >> 1u;
  if (frame - stamp > 1u) { return; }
  let obj = i32((k1 >> 19u) & 0x1FFFu) - 1;
  let voxel = gi_unkey_voxel(k0, k1, obj);
  let face = (k1 >> 16u) & 7u;
  if (gi_wal_slot(voxel, face, obj, gi_wal_nmask()) != slot) { return; }
  let g = make_grid(u32(obj + 1));
  let pal = sample_brickmap(g, voxel);
  if (pal == 0u) { return; }
  let n_geo = face_normal_world(g, face);
  let org = voxel_center_world(g, voxel) + n_geo * surface_offset(g, n_geo, SHADOW_SURFACE_EPS);
  let n_old = bitcast<f32>(atomicLoad(&gi_wal[b + 6u]));
  let sum_old = vec3<f32>(bitcast<f32>(atomicLoad(&gi_wal[b + 3u])),
    bitcast<f32>(atomicLoad(&gi_wal[b + 4u])),
    bitcast<f32>(atomicLoad(&gi_wal[b + 5u])));
  let cap = max(sum_old / max(n_old, 1e-6) * GI_SS_CAND_CLAMP_K, vec3<f32>(GI_SS_CAND_CLAMP_CEIL));
  let sun_dir = light_u.lights[0].kind_pos_dir.yzw;
  let sun_c = light_u.lights[0].color_intensity.xyz * light_u.lights[0].color_intensity.w;
  let sun_bounce = gi_u.params.x;
  let epoch = gi_u.seq.y;
  let b2_mult = gi_u.misc.w;
  var sum = vec3<f32>(0.0);
  for (var i = 0u; i < GI_WAL_UPD_R; i = i + 1u) {
    let cseed = gi_mix(slot * 0x9E3779B1u + frame * 0x85EBCA77u + i * 0x27D4EB2Fu + 0x165667B1u);
    let dir = gi_cos_hemisphere(n_geo, gi_rand2(cseed));
    let b2 = select(0.0, b2_mult, gi_rand2(gi_mix(cseed + 0x27D4EB2Fu)).y < 1.0 / max(b2_mult, 1.0));
    var c = gi_ray_radiance(org, dir, sun_dir, sun_c, sun_bounce, epoch, b2, cseed) * GI_PI;
    if (n_old > 0.5) { c = min(c, cap); }
    sum = sum + c;
  }
  if (GI_WAL_DIAG == 1u) { atomicAdd(&lod_diag[121u], 1u); }
  gi_wal_merge(voxel, obj, pal, k0, k1, sum, GI_WAL_UPD_R, stamp);
}

@compute @workgroup_size(8, 8, 1)
fn gi_face_flatten(@builtin(global_invocation_id) gid: vec3<u32>) {
  let size = textureDimensions(gi_out_flat);
  if (gid.x >= size.x || gid.y >= size.y) { return; }
  let idx = gid.y * size.x + gid.x;
  let gb = idx * GI_DEN_GUIDE_WORDS;
  let k0 = gi_guide[gb + GI_DEN_G_KEY0];
  let k1 = gi_guide[gb + GI_DEN_G_KEY1];

  if ((k0 | k1) == 0u) { return; }
  var ok = false;
  let slot = face_slot_find_flat(k0, gi_vox_k1(k1), &ok);
  if (!ok) { return; }
  let fe = slot * FACE_WORDS;
  let n = atomicLoad(&face_slots_flat[fe + FACE_W_RAW_N]);
  if (n == 0u) { return; }
  let inv = 1.0 / (f32(n) * GI_ACC_SCALE);
  let flat = vec3<f32>(
    f32(atomicLoad(&face_slots_flat[fe + FACE_W_RAW])),
    f32(atomicLoad(&face_slots_flat[fe + FACE_W_RAW + 1u])),
    f32(atomicLoad(&face_slots_flat[fe + FACE_W_RAW + 2u]))) * inv;
  textureStore(gi_out_flat, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(flat, 1.0));
}

@compute @workgroup_size(8, 8, 1)
fn gi_denoise_temporal(@builtin(global_invocation_id) gid: vec3<u32>) {
  let size = textureDimensions(gi_dn_tmp);
  if (gid.x >= size.x || gid.y >= size.y) { return; }
  gi_den_temporal(size, vec2<i32>(i32(gid.x), i32(gid.y)));
}

@compute @workgroup_size(8, 8, 1)
fn gi_denoise_atrous1(@builtin(global_invocation_id) gid: vec3<u32>) {
  let size = textureDimensions(gi_dn_dst);
  if (gid.x >= size.x || gid.y >= size.y) { return; }
  gi_den_atrous(size, vec2<i32>(i32(gid.x), i32(gid.y)), 1);
}

@compute @workgroup_size(8, 8, 1)
fn gi_denoise_atrous2(@builtin(global_invocation_id) gid: vec3<u32>) {
  let size = textureDimensions(gi_dn_dst);
  if (gid.x >= size.x || gid.y >= size.y) { return; }
  gi_den_atrous(size, vec2<i32>(i32(gid.x), i32(gid.y)), 2);
}

@compute @workgroup_size(8, 8, 1)
fn gi_denoise_atrous4(@builtin(global_invocation_id) gid: vec3<u32>) {
  let size = textureDimensions(gi_dn_dst);
  if (gid.x >= size.x || gid.y >= size.y) { return; }
  gi_den_atrous(size, vec2<i32>(i32(gid.x), i32(gid.y)), 4);
}

@compute @workgroup_size(8, 8, 1)
fn gi_denoise_atrous8(@builtin(global_invocation_id) gid: vec3<u32>) {
  let size = textureDimensions(gi_dn_dst);
  if (gid.x >= size.x || gid.y >= size.y) { return; }
  gi_den_atrous(size, vec2<i32>(i32(gid.x), i32(gid.y)), 8);
}

@compute @workgroup_size(8, 8, 1)
fn gi_denoise_atrous16(@builtin(global_invocation_id) gid: vec3<u32>) {
  let size = textureDimensions(gi_dn_dst);
  if (gid.x >= size.x || gid.y >= size.y) { return; }
  gi_den_atrous(size, vec2<i32>(i32(gid.x), i32(gid.y)), 16);
}

@compute @workgroup_size(8, 8, 1)
fn godray_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  godray_pass_mask(gid);
}

@compute @workgroup_size(8, 8, 1)
fn godray_blur_a(@builtin(global_invocation_id) gid: vec3<u32>) {
  godray_pass_blur_a(gid);
}

@compute @workgroup_size(8, 8, 1)
fn godray_blur_b(@builtin(global_invocation_id) gid: vec3<u32>) {
  godray_pass_blur_b(gid);
}

const EYE_LUM_MIN_LOG2: f32 = -12.0;
const EYE_LUM_RANGE_LOG2: f32 = 24.0;
const EYE_HIST_BASE: u32 = 8u;
const EYE_HIST_BINS: u32 = 64u;
const EYE_PARAM_BASE: u32 = EYE_HIST_BASE + EYE_HIST_BINS;

fn eye_p(i: u32) -> f32 {
  return bitcast<f32>(atomicLoad(&eye_adapt[EYE_PARAM_BASE + i]));
}

var<workgroup> eye_hist: array<atomic<u32>, EYE_HIST_BINS>;

@compute @workgroup_size(256)
fn eye_adapt_histogram(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dims = vec2<i32>(textureDimensions(out_tex_src));
  for (var i = 0u; i < EYE_HIST_BINS; i = i + 1u) { atomicStore(&eye_hist[i], 0u); }
  workgroupBarrier();
  let stride = 16u;
  let axis = u32(dims.x / 16);
  let step = axis * u32(dims.y / 16);
  let bin_w = EYE_LUM_RANGE_LOG2 / f32(EYE_HIST_BINS);
  for (var k = gid.x; k < step; k = k + 256u) {
    let sx = i32((k % axis) * stride + stride / 2u);
    let sy = i32((k / axis) * stride + stride / 2u);
    if (sx >= dims.x || sy >= dims.y) { continue; }
    let c = textureLoad(out_tex_src, vec2<i32>(sx, sy), 0).rgb;
    let luma = dot(srgb_to_linear(c), GI_LUMA);
    let b = u32(clamp((log2(max(luma, 1e-5)) - EYE_LUM_MIN_LOG2) / bin_w, 0.0, f32(EYE_HIST_BINS - 1u)));
    atomicAdd(&eye_hist[b], 1u);
  }
  workgroupBarrier();
  for (var i = gid.x; i < EYE_HIST_BINS; i = i + 256u) {
    let v = atomicLoad(&eye_hist[i]);
    if (v > 0u) { atomicAdd(&eye_adapt[EYE_HIST_BASE + i], v); }
  }
}

@compute @workgroup_size(1)
fn eye_adapt_update() {
  var total = 0.0;
  for (var i = 0u; i < EYE_HIST_BINS; i = i + 1u) { total = total + f32(atomicLoad(&eye_adapt[EYE_HIST_BASE + i])); }
  if (total > 0.0) {
    let lo = total * 0.05;
    let hi = total * 0.95;
    let bin_w = EYE_LUM_RANGE_LOG2 / f32(EYE_HIST_BINS);
    var cum = 0.0;
    var acc = 0.0;
    var wsum = 0.0;
    for (var i = 0u; i < EYE_HIST_BINS; i = i + 1u) {
      let n = f32(atomicLoad(&eye_adapt[EYE_HIST_BASE + i]));
      let c0 = cum;
      cum = cum + n;
      let w = max(0.0, min(cum, hi) - max(c0, lo));
      acc = acc + (EYE_LUM_MIN_LOG2 + (f32(i) + 0.5) * bin_w) * w;
      wsum = wsum + w;
    }
    if (wsum > 0.0) {
      let avg_log2 = acc / wsum;
      atomicStore(&eye_adapt[4u], bitcast<u32>(avg_log2));
      let ev_cur = bitcast<f32>(atomicLoad(&eye_adapt[1u]));
      let ev_want = clamp(ev_cur + (log2(eye_p(4u)) - avg_log2), eye_p(1u), eye_p(0u));
      atomicStore(&eye_adapt[5u], bitcast<u32>(ev_want));
      let dt = clamp(bitcast<f32>(atomicLoad(&eye_adapt[3u])), 0.0, 0.25);
      let tau = select(eye_p(3u), eye_p(2u), ev_want > ev_cur);

      let k = select(0.0, 1.0 - exp(-dt / max(tau, 1e-3)), dt > 0.0);
      let ev_new = mix(ev_cur, ev_want, k);
      atomicStore(&eye_adapt[1u], bitcast<u32>(ev_new));
      let manual = bitcast<f32>(atomicLoad(&eye_adapt[2u]));
      let exp_val = exp2(clamp(ev_new + manual, eye_p(1u), eye_p(0u)));
      atomicStore(&eye_adapt[0u], bitcast<u32>(exp_val));
    }
  }
  for (var i = 0u; i < EYE_HIST_BINS; i = i + 1u) { atomicStore(&eye_adapt[EYE_HIST_BASE + i], 0u); }
}

fn refl_accum(px: u32, v: vec3<f32>) {
  if (px >= refl_cfg.y) { return; }

  let q = clamp(v, vec3<f32>(0.0), vec3<f32>(63.0)) * 65536.0 + 0.5;
  let o = px * 3u;
  atomicAdd(&refl_sum[o + 0u], u32(q.x));
  atomicAdd(&refl_sum[o + 1u], u32(q.y));
  atomicAdd(&refl_sum[o + 2u], u32(q.z));
}

fn refl_shade_ray(org: vec3<f32>, dir: vec3<f32>, tier: u32,
                  sun_dir: vec3<f32>, sun_c: vec3<f32>, sky: vec3<f32>) -> vec3<f32> {
  let g0w = make_grid(0u);
  let rhit = world_raycast(g0w, org, dir, REFL_FAR_T, world_cfg_full(0.0, 3u));
  var rc = select(sky_rgb(), sky_primary(dir, sun_dir, sun_c), tier >= 2u);
  if (rhit.hit) {
    let rgg = make_grid(u32(rhit.obj_id) + 1u);
    let r_vc = voxel_center_world(rgg, rhit.voxel);
    let rn = voxel_normal_fast(rgg, rhit.voxel, rhit.face_id, rhit.n);
    let mr = fetch_material(rhit.palette_base, rhit.pal, rhit.point, rn, PBR_DEBUG_ASSET);
    var r_vis = vec3<f32>(1.0);
    if (tier >= 3u && (light_u.g.base_flags & BASE_FLAG_SHADOW) != 0u
        && dot(rn, sun_dir) > 0.0) {
      r_vis = sun_transmittance(r_vc + rn * surface_offset(rgg, rn, SHADOW_SURFACE_EPS),
                                sun_dir, rhit.voxel, rhit.obj_id);
    }
    rc = brdf_reflected(mr, rn, face_view_dir(org, r_vc),
                        SurfaceLight(sun_dir, sun_c * r_vis, 1.0, sky * GLASS_REFL_AMBIENT, vec3<f32>(0.0)))
      + emissive_radiance(mr, GI_EMIT_GAIN);
  }
  return rc;
}

@compute @workgroup_size(64, 1, 1)
fn dda_refl(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= atomicLoad(&refl_meta[0]) || i >= refl_cfg.x) { return; }
  let a = refl_req[i * 2u];
  let b = refl_req[i * 2u + 1u];
  let px = b.w;
  let kind = (a.w >> 30u) & 3u;
  let w = unpack_w10(a.w & 0x3FFFFFFFu);
  let sun_dir = light_u.lights[0].kind_pos_dir.yzw;
  let sun_c = light_u.lights[0].color_intensity.xyz * light_u.lights[0].color_intensity.w;
  let sky = light_u.sky_color.xyz;
  let tier = light_u.g.refl_tier;
  var rad = vec3<f32>(0.0);
  if (kind == REFL_KIND_RAY) {
    let org = vec3<f32>(bitcast<f32>(a.x), bitcast<f32>(a.y), bitcast<f32>(a.z));
    let dir = normalize(vec3<f32>(bitcast<f32>(b.x), bitcast<f32>(b.y), bitcast<f32>(b.z)));
    rad = refl_shade_ray(org, dir, tier, sun_dir, sun_c, sky);
  } else {
    let k0 = a.x;
    let k1 = a.y;
    let obj_id = i32((k1 >> 19u) & 0x1FFFu) - 1;
    if (obj_id >= 0) {
      let face_id = b.y;
      let voxel = gi_unkey_voxel(k0, k1, obj_id);
      let gg = make_grid(u32(obj_id) + 1u);
      let p_voxel = voxel_center_world(gg, voxel);
      let hn = voxel_normal_fast(gg, voxel, face_id, face_normal_world(gg, face_id));
      let hm = fetch_material(b.x, a.w, p_voxel, hn, PBR_DEBUG_ASSET);
      let hv = face_view_dir(view_u.cam_pos_voxel.xyz, p_voxel);
      rad = w * refl_env_chain(p_voxel, hn, hv, refl_seed(voxel, face_id, obj_id), hm,
                               REFL_FAR_T, surface_offset(gg, hn, SHADOW_SURFACE_EPS));
    }
  }
  refl_accum(px, w * rad);
}

@compute @workgroup_size(8, 8, 1)
fn dda_refl_compose(@builtin(global_invocation_id) gid: vec3<u32>) {
  let size = textureDimensions(out_tex);
  if (gid.x >= size.x || gid.y >= size.y) { return; }
  let coord = vec2<i32>(i32(gid.x), i32(gid.y));
  let px = u32(gid.y) * size.x + u32(gid.x);
  let base = srgb_to_linear(textureLoad(out_base_tex, coord, 0).rgb);
  let o = px * 3u;
  let add = vec3<f32>(f32(atomicLoad(&refl_sum[o])),
                      f32(atomicLoad(&refl_sum[o + 1u])),
                      f32(atomicLoad(&refl_sum[o + 2u])))
          * (1.0 / 65536.0);
  let exposure = bitcast<f32>(eye_adapt_ro[0u]);
  let lin = base + add * select(1.0, exposure, exposure > 0.0);
  textureStore(out_tex, coord, vec4<f32>(linear_to_srgb(lin) + dither3(coord), 1.0));
}
