import package::bindings::{refl_cfg, refl_meta, refl_req, refl_sum, refl_first, refl_ray, out_base_tex, BASE_FLAG_IMPLICIT_NORMAL, BASE_FLAG_SHADOW, BASE_FLAG_VOL_TINT, BEAM_BACKOFF, BEAM_DIV, beam_depth, dda_hit, dda_vis, dda_nrm, dda_shadow_tex, glass_req, glass_meta, eye_adapt, eye_adapt_ro, fog_tex, gi_stats, gi_u, irc, light_u, lod_diag, out_base, out_tex, out_tex_src, view_u};
import package::common::{FLAGS_TRANSMISSIVE, MaterialParams, SurfaceLight, brdf_reflected, dielectric_f0, dither3, emissive_radiance, f0_of, face_of_normal, face_view_dir, fetch_material, fresnel_schlick, linear_to_srgb, material_ior, palette_flags, palette_rgb_linear, sky_rgb, srgb_to_linear};
import package::gi::common::{gi_ambient_fallback, gi_div, gi_emit_gain, gi_enabled, gi_key0, gi_key1, gi_key_ok, gi_mix, gi_pose_salt, gi_rand2, gi_record_pixel, gi_sample_base, gi_sample_boost, gi_unkey_voxel};
import package::gi::consts::{GI_LUMA, GI_PI, GI_SAMPLE_CLAMP, GI_SKY_AMBIENT, GI_SPP_MAX, GI_STAT_RAYS, GI_STAT_RES_SPATIAL, GI_STAT_RES_TEMPORAL, IRC_WORDS, SHADOW_SURFACE_EPS};
import package::gi::irc::{irc_accumulate, irc_addr, irc_display, irc_lookup, irc_resolve_slot};
import package::gi::light::{lt_hit_weight, lt_nee, lt_pick, lt_register, lt_sweep_slot};
import package::gi::ray::{gi_accumulate, gi_accumulate_sun, gi_leaf_indirect, gi_origin, gi_sun_vis, gi_surface};
import package::gi::restir::{Res, res_add_nb, res_add_pick, res_adopt, res_begin, res_direct, res_finish, res_load_cur, res_load_prev, res_load_tmp, res_none, res_store_cur, res_store_tmp};
import package::brickmap::{Grid, sample_brickmap};
import package::trace::{LOD_DIAG, diag_sampled, make_grid, medium_of};
import package::volumetric::{godray_pass_blur_a, godray_pass_blur_b, godray_pass_mask, godray_sample_bilinear, sky_primary};
import package::world::{medium_one, voxel_normal_fast, WorldHit, camera_solid_pal, face_normal_world, surface_offset, vol_tint_color, voxel_at_point, voxel_center_world, voxel_normal, world_cfg_full, world_cfg_primary, world_cfg_reflect, world_miss, world_raycast, world_trace_glass_path, world_trace_medium_scene};

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

@compute @workgroup_size(8, 8, 1)
fn dda_preview(@builtin(global_invocation_id) gid: vec3<u32>,
               @builtin(local_invocation_index) lid: u32) {
  if (lid == 0u) { cam_pal_wg = camera_solid_pal(view_u.cam_pos_voxel.xyz); }
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
  if (gi_enabled() && gi_u.params.y > 0.0) { dda_hit_record(gid, size, first, origin_voxel); }
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
    col = albedo * (sky * 0.3 + sun_c * max(dot(n, sun_dir), 0.0));
  }
  write_output(coord0, col);
}

struct FaceShadow {
  vis: vec3<f32>,
  ok: bool,
}

struct GlassOut {
  ray: WorldHit,
  tint: vec3<f32>,
  refl: vec3<f32>,
  dir: vec3<f32>,
}

const GLASS_EPS: f32 = 1e-3;

const GLASS_GI_FAR0: f32 = 1200.0;
const GLASS_GI_FAR1: f32 = 2600.0;

const GLASS_EXIT_BEND: bool = true;
const GLASS_QUALITY_SHIFT: u32 = 3u;
const GLASS_REFL_MIN: f32 = 0.02;
const GLASS_REFL_WEIGHT_MIN: f32 = 0.015;
const GLASS_REFL_AMBIENT: f32 = 0.15;
const PBR_REFLECTION_ENABLED: u32 = 1u;
const PBR_REFL_MIN_F0: f32 = 0.03;
const PBR_REFL_MAX_ROUGH: f32 = 0.05;
const PBR_REFL_CONE_K: f32 = 1.0;
const REFL_TIER_GI: u32 = 4u;
const FACE_W_REFL: u32 = 5u;

fn sun_transmittance(origin: vec3<f32>, sun_dir: vec3<f32>, self_voxel: vec3<i32>,
                     self_obj: i32) -> vec3<f32> {
  var tint = vec3<f32>(1.0);
  let h = world_trace_medium_scene(origin, sun_dir, min(8192.0, 3072.0), false, false, &tint);
  if (!h.hit) { return tint; }
  if (all(h.voxel == self_voxel) && h.obj_id == self_obj) { return vec3<f32>(1.0); }
  return vec3<f32>(0.0);
}

fn refl_cone_dir(r: vec3<f32>, u: vec2<f32>, spread: f32) -> vec3<f32> {
  let axis = select(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 0.0), abs(r.z) > 0.9);
  let t = normalize(cross(axis, r));
  let b = cross(r, t);
  let rad = spread * sqrt(u.x);
  let phi = 6.2831853 * u.y;
  return normalize(r + t * (rad * cos(phi)) + b * (rad * sin(phi)));
}

const PBR_DEBUG_ASSET: u32 = 0xFFFFFFFFu;

fn secondary_reach(max_t: f32) -> f32 {
  let rc = light_u.g.refl_reach;
  return select(min(max_t, f32(rc)), max_t, rc == 0xFFFFFFFFu);
}

fn secondary_on() -> bool {
  return light_u.g.refl_reach != 0u;
}

const REFL_NO_PX: u32 = 0xFFFFFFFFu;
const REFL_FIRST_NONE: u32 = 0xFFFFFFFFu;
const REFL_FIRST_WORDS: u32 = 4u;

fn refl_first_claim(k0: u32, k1: u32, px: u32) -> bool {
  let n = arrayLength(&refl_first) / REFL_FIRST_WORDS;
  if (n == 0u) { return true; }
  let mask = n - 1u;
  var s = gi_mix(k0 + gi_mix(k1 * 0x9E3779B1u)) & mask;
  for (var i = 0u; i < 8u; i = i + 1u) {
    let b = s * REFL_FIRST_WORDS;
    let st = atomicLoad(&refl_first[b + 3u]);
    if (st != 0u) {
      if (atomicLoad(&refl_first[b]) == k0 && atomicLoad(&refl_first[b + 1u]) == k1) {
        return false;
      }
      s = (s + 1u) & mask;
      continue;
    }
    if (atomicCompareExchangeWeak(&refl_first[b + 3u], 0u, 1u).exchanged) {
      atomicStore(&refl_first[b], k0);
      atomicStore(&refl_first[b + 1u], k1);
      atomicStore(&refl_first[b + 2u], px);
      atomicStore(&refl_first[b + 3u], 2u);
      return true;
    }
    if (atomicLoad(&refl_first[b]) == k0 && atomicLoad(&refl_first[b + 1u]) == k1) {
      return false;
    }
    s = (s + 1u) & mask;
  }
  return true;
}

fn refl_first_lookup(k0: u32, k1: u32) -> u32 {
  let n = arrayLength(&refl_first) / REFL_FIRST_WORDS;
  if (n == 0u) { return REFL_FIRST_NONE; }
  let mask = n - 1u;
  var s = gi_mix(k0 + gi_mix(k1 * 0x9E3779B1u)) & mask;
  for (var i = 0u; i < 8u; i = i + 1u) {
    let b = s * REFL_FIRST_WORDS;
    if (atomicLoad(&refl_first[b + 3u]) == 0u) { return REFL_FIRST_NONE; }
    if (atomicLoad(&refl_first[b]) == k0 && atomicLoad(&refl_first[b + 1u]) == k1) {
      return atomicLoad(&refl_first[b + 2u]);
    }
    s = (s + 1u) & mask;
  }
  return REFL_FIRST_NONE;
}
const REFL_FAR_T: f32 = 65536.0;
const REFL_GLASS_FAR_T: f32 = 32768.0;
const SEABED_SUN_CAP: bool = true;
const REFL_KIND_PACKED: u32 = 1u;
const REFL_KIND_RAY: u32 = 0u;

fn pack_w10(w: vec3<f32>) -> u32 {
  let q = clamp(w, vec3<f32>(0.0), vec3<f32>(1.0)) * 1023.0 + 0.5;
  return u32(q.x) | (u32(q.y) << 10u) | (u32(q.z) << 20u);
}

fn unpack_w10(v: u32) -> vec3<f32> {
  return vec3<f32>(f32(v & 1023u), f32((v >> 10u) & 1023u), f32((v >> 20u) & 1023u)) * (1.0 / 1023.0);
}

fn queue_refl_ray(org: vec3<f32>, dir: vec3<f32>, w: vec3<f32>, px: u32, k0: u32, k1: u32) {
  if (px == REFL_NO_PX) { return; }
  if (!refl_first_claim(k0, k1, px)) { return; }
  let i = atomicAdd(&refl_meta[0], 1u);
  if (i >= refl_cfg.x) { return; }
  let o = i * 3u;
  refl_req[o] = vec4<u32>(bitcast<u32>(org.x), bitcast<u32>(org.y), bitcast<u32>(org.z),
                          (REFL_KIND_RAY << 30u) | pack_w10(w));
  refl_req[o + 1u] = vec4<u32>(bitcast<u32>(dir.x), bitcast<u32>(dir.y), bitcast<u32>(dir.z), px);
  refl_req[o + 2u] = vec4<u32>(k0, k1, 1u, 0u);
}

fn queue_refl_chain(k0: u32, k1: u32, pal: u32, pbase: u32, face_id: u32,
                     w: vec3<f32>, px: u32, dedupe: bool) {
  if (px == REFL_NO_PX || k0 == 0u || k1 == 0u) { return; }
  if (dedupe && !refl_first_claim(k0, k1, px)) { return; }
  let i = atomicAdd(&refl_meta[0], 1u);
  if (i >= refl_cfg.x) { return; }
  let o = i * 3u;
  refl_req[o] = vec4<u32>(k0, k1, pal, (REFL_KIND_PACKED << 30u) | pack_w10(w));
  refl_req[o + 1u] = vec4<u32>(pbase, face_id, 0u, px);
  refl_req[o + 2u] = vec4<u32>(0u);
}

fn trace_glass(origin_voxel: vec3<f32>, dir_voxel: vec3<f32>, max_t: f32,
               entry_n: vec3<f32>, entry_ior: f32, entry_v: vec3<f32>, qpx: u32,
               quality: u32, allow_refl: bool, vk0: u32, vk1: u32) -> GlassOut {
  var tint = vec3<f32>(1.0);
  let f0 = dielectric_f0(entry_ior, 1.0);
  let fres = f0 + (1.0 - f0) * pow(1.0 - clamp(dot(entry_v, entry_n), 0.0, 1.0), 5.0);

  let r_org = origin_voxel + entry_n * GLASS_EPS;
  let r_dir = reflect(-entry_v, entry_n);
  if (allow_refl && secondary_on() && light_u.g.refl_tier > 0u
      && fres > GLASS_REFL_MIN
      && dot(vec3<f32>(fres), GI_LUMA) > GLASS_REFL_WEIGHT_MIN) {
    queue_refl_ray(r_org, r_dir, vec3<f32>(fres), qpx, vk0, vk1);
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

fn voxel_screen_uv(p_voxel: vec3<f32>, fallback: vec2<f32>) -> vec2<f32> {
  let clip = view_u.view_proj * vec4<f32>(p_voxel, 1.0);
  if (clip.w <= 1e-6) { return fallback; }
  let ndc = clip.xyz / clip.w;
  return vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
}

fn voxel_face_index(gg: Grid, n: vec3<f32>, obj_id: i32) -> u32 {
  let ln = select(n, vec3<f32>(dot(n, normalize(gg.col0)), dot(n, normalize(gg.col1)),
                               dot(n, normalize(gg.col2))), obj_id >= 0);
  return face_of_normal(ln);
}

fn gi_display_sky(h: WorldHit, n: vec3<f32>) -> vec3<f32> {
  let g = make_grid(u32(h.obj_id) + 1u);
  let face = voxel_face_index(g, n, h.obj_id);
  let s = gi_surface(h, g, n, face);
  let d = irc_display(s.addr, s.n);
  return mix(gi_ambient_fallback(n, 0.0), d.rgb, d.a);
}

fn refl_hit_gi_m(voxel: vec3<i32>, face_id: u32, obj_id: i32, pal: u32, pool_r: i32,
                 pool_st: i32, pos: vec3<f32>) -> vec4<f32> {
  let g = make_grid(u32(obj_id) + 1u);
  let n = face_normal_world(g, face_id);
  let s = gi_surface(WorldHit(true, 0.0, pos, pal, n, face_id, voxel, obj_id, g.palette_base),
                     g, n, face_id);
  let h = irc_lookup(s.addr, s.face);
  return vec4<f32>(h.e, max(h.conf, select(0.0, 1.0, h.sun_ok)));
}

fn refl_hit_gi(voxel: vec3<i32>, face_id: u32, obj_id: i32, pal: u32, pos: vec3<f32>) -> vec3<f32> {
  return refl_hit_gi_m(voxel, face_id, obj_id, pal, 3, 1, pos).rgb;
}

fn refl_gi_merge(h: WorldHit, seed: u32) -> bool {
  if (!gi_enabled()) { return false; }
  let g = make_grid(u32(h.obj_id) + 1u);
  let n = voxel_normal_fast(g, h.voxel, h.face_id, h.n);
  let s = gi_surface(h, g, n, h.face_id);
  let hit = irc_lookup(s.addr, s.face);
  if (hit.conf > 0.0 || hit.fresh) { return false; }
  gi_accumulate(s, seed, -1.0);
  return true;
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
                  frustum_length: f32, n_off: f32, merge_gi: bool) -> vec3<f32> {
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
    let h = world_trace_medium_scene(chain_org, chain_dir, reach, true, false, &r_tint);
    if (!h.hit) {
      refl_env = refl_env
               + chain_w * r_tint
                 * select(sky_rgb(), sky_primary(chain_dir, sun_dir, sun_c), tier >= 2u);
      break;
    }
    let hgg = make_grid(u32(h.obj_id) + 1u);
    let hn = voxel_normal_fast(hgg, h.voxel, h.face_id, h.n);
    var merged = false;
    if (merge_gi && lvl == 0u
        && (palette_flags(h.palette_base, h.pal) & FLAGS_TRANSMISSIVE) == 0u) {
      merged = refl_gi_merge(h, rseed);
    }
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
    var h_gi = vec3<f32>(0.0);
    if (tier >= REFL_TIER_GI) {
      h_gi = select(refl_hit_gi(h.voxel, h.face_id, h.obj_id, h.pal, h.point), vec3<f32>(0.0), merged);
    }
    var h_local = brdf_reflected(hm, hn, hv,
                                 SurfaceLight(sun_dir, sun_c * h_vis, 1.0, hamb, h_gi))
                + emissive_radiance(hm, gi_emit_gain());
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

const SUN_UNDERWATER_CAP: f32 = 3072.0;
fn sun_vis_capped(origin: vec3<f32>, sun_dir: vec3<f32>, self_voxel: vec3<i32>, self_obj: i32,
                  cap: f32) -> vec3<f32> {
  var tint = vec3<f32>(1.0);
  var best = world_miss();
  var best_t = 1.0e+30;
  medium_one(make_grid(0u), origin, sun_dir, min(cap, SUN_UNDERWATER_CAP), 0.0, false,
             &best, &best_t, &tint);
  if (!best.hit) { return tint; }
  if (all(best.voxel == self_voxel) && best.obj_id == self_obj) { return vec3<f32>(1.0); }
  return vec3<f32>(0.0);
}

fn shade_face(first: WorldHit, origin_voxel: vec3<f32>, ray_dir: vec3<f32>, frustum_length: f32,
              gi_uv: vec2<f32>, sun_dir: vec3<f32>, sky: vec3<f32>, sun_c: vec3<f32>,
              n_hint: vec3<f32>, n_hint_ok: bool,
              qpx: u32, best_out: ptr<function, WorldHit>, f_shadow: FaceShadow) -> vec3<f32> {
  var best = first;
  *best_out = first;
  var through = vec3<f32>(1.0);
  var refl = vec3<f32>(0.0);
  var glass_col = vec3<f32>(0.0);
  var glass_w = 0.0;
  var sky_dir = ray_dir;
  var entry_v = ray_dir;
  var water_miss = false;
  var water_surf = false;
  var refract_hit = false;
  var through_transmission = false;
  let glass_q = (light_u.g.base_flags >> GLASS_QUALITY_SHIFT) & 3u;
  if (first.hit && (palette_flags(first.palette_base, first.pal) & FLAGS_TRANSMISSIVE) != 0u) {
    let fg = make_grid(u32(first.obj_id) + 1u);
    let fvx = first.voxel;
    let fp = voxel_center_world(fg, fvx);
    var fnn = n_hint;
    if (!n_hint_ok) { fnn = voxel_normal_fast(fg, fvx, first.face_id, first.n); }
    let fm = fetch_material(first.palette_base, first.pal, fp, fnn, PBR_DEBUG_ASSET);
    water_miss = fm.roughness < 0.1;
    water_surf = water_miss || fm.ior < 1.4
      || (fm.albedo.b > fm.albedo.r * 1.25 && fm.albedo.b > fm.albedo.g * 1.05);
    if (water_miss && (first.face_id == 2u || first.face_id == 3u)) {
      fnn = face_normal_world(fg, first.face_id);
    }
    let f_off = surface_offset(fg, fnn, SHADOW_SURFACE_EPS);
    let f_face = fp + fnn * f_off;
    let fv = face_view_dir(origin_voxel, fp);
    entry_v = fv;
    let g = trace_glass(f_face, -fv, frustum_length, fnn,
                        material_ior(first.palette_base, first.pal), fv, qpx, glass_q, true,
                        gi_key0(first.voxel, first.obj_id), gi_key1(first.voxel, first.face_id, first.obj_id));
    if (glass_q > 0u) {
      best = g.ray;
      *best_out = g.ray;
      through = g.tint;
      refl = g.refl;
      sky_dir = g.dir;
      refract_hit = glass_q >= 2u;
      through_transmission = true;
      glass_w = clamp(1.0 - fm.transmission, 0.0, 1.0);
    } else {
      glass_w = 1.0;
    }
    if (glass_w > 1.0 / 255.0) {
      let f_ndl = max(dot(fnn, sun_dir), 0.0);
      var f_sun = select(vec3<f32>(0.0), f_shadow.vis, f_shadow.ok);
      if (!f_shadow.ok && f_ndl > 0.0) {
        f_sun = sun_transmittance(f_face, sun_dir, first.voxel, first.obj_id);
      }
      var f_gi = vec3<f32>(0.0);
      let f_amb = sky * GI_SKY_AMBIENT * 0.2;
      glass_col = brdf_reflected(fm, fnn, fv,
                                 SurfaceLight(sun_dir, sun_c * f_sun, 1.0, f_amb, f_gi))
                + emissive_radiance(fm, gi_emit_gain());
    }
  }
  var col = vec3<f32>(0.0);
  if (best.hit) {
    let gg = make_grid(u32(best.obj_id) + 1u);
    let p_voxel = voxel_center_world(gg, best.voxel);
    if ((light_u.g.base_flags & BASE_FLAG_IMPLICIT_NORMAL) == 0u) {
      return fetch_material(best.palette_base, best.pal, p_voxel, normalize(gg.col1), PBR_DEBUG_ASSET).albedo;
    }
    let k0 = gi_key0(best.voxel, best.obj_id);
    let k1 = gi_key1(best.voxel, best.face_id, best.obj_id);
    var n = best.n;
    if (n_hint_ok && !through_transmission) {
      n = n_hint;
    } else {
      n = voxel_normal_fast(gg, best.voxel, best.face_id, best.n);
    }
    let nf = voxel_face_index(gg, n, best.obj_id);
    let kv1 = gi_key1(best.voxel, nf, best.obj_id);
    let m = fetch_material(best.palette_base, best.pal, p_voxel, n, PBR_DEBUG_ASSET);
    if (gi_enabled() && gi_u.params.y > 0.0
        && (palette_flags(best.palette_base, best.pal) & FLAGS_TRANSMISSIVE) == 0u) {
      let le = emissive_radiance(m, gi_emit_gain());
      if (max(le.x, max(le.y, le.z)) > 0.0) {
        lt_register(best.voxel, nf, u32(best.obj_id + 1), dot(le, GI_LUMA));
      }
    }
    let v = face_view_dir(origin_voxel, p_voxel);
    let n_off = surface_offset(gg, n, SHADOW_SURFACE_EPS);
    let ndl = max(dot(n, sun_dir), 0.0);
    var sun = vec3<f32>(0.0);
    if (ndl > 0.0) {
      sun = vec3<f32>(1.0);
      if ((light_u.g.base_flags & BASE_FLAG_SHADOW) != 0u) {
        if (through_transmission && refract_hit && f_shadow.ok) {
          let k = clamp(abs(entry_v.y) / max(abs(sun_dir.y), 0.05), 0.02, 8.0);
          sun = f_shadow.vis * pow(max(through, vec3<f32>(1e-4)), vec3<f32>(k));
        } else {
          sun = sun_transmittance(p_voxel + n * n_off, sun_dir, best.voxel, best.obj_id);
        }
      }
    }
    var gi = vec3<f32>(0.0);
    var cov = 0.0;
    if (gi_enabled()) {
      let s = gi_surface(best, gg, n, nf);
      let d = irc_display(s.addr, s.n);
      gi = d.rgb;
      cov = d.a;
    }
    let amb = gi_ambient_fallback(n, cov);
    col = brdf_reflected(m, n, v, SurfaceLight(sun_dir, sun_c * sun, 1.0, amb, gi))
        + emissive_radiance(m, gi_emit_gain());
    if (PBR_REFLECTION_ENABLED != 0u && light_u.g.refl_tier > 0u) {
      let f0r = f0_of(m);
      let f0_luma = dot(f0r, GI_LUMA);
      if (f0_luma >= PBR_REFL_MIN_F0 && m.roughness <= PBR_REFL_MAX_ROUGH) {
        let fade = clamp(1.0 - m.roughness / PBR_REFL_MAX_ROUGH, 0.0, 1.0);
        col = col - fade * f0r * (amb + gi);
        queue_refl_chain(k0, kv1, best.pal, best.palette_base, nf,
                         fresnel_schlick(f0r, max(dot(v, n), 0.0)) * fade, qpx,
                         !through_transmission);
      }
    }
    col = col * through + refl;
  } else {
    var miss = sky_primary(sky_dir, sun_dir, sun_c) * through;
    if (water_miss) { miss = glass_col; }
    return mix(glass_col, miss + refl, 1.0 - glass_w);
  }
  return mix(glass_col, col, 1.0 - glass_w);
}

const DIAG_FACE_SKY: u32 = 8u;
const DIAG_MAIN_RAYS: u32 = 9u;
const DIAG_MAIN_NEAR: u32 = 10u;
const DIAG_MAIN_FAR: u32 = 11u;
const DIAG_MAIN_SKY: u32 = 12u;

var<workgroup> cam_pal_wg: u32;

fn dda_hit_record(gid: vec3<u32>, size: vec2<u32>, hit: WorldHit, origin_voxel: vec3<f32>) {
  let gid2 = vec2<u32>(u32(gid.x), u32(gid.y));
  let px = gi_record_pixel(gid2 / gi_div());
  if (px.x != gid2.x || px.y != gid2.y) { return; }
  let texel = px / gi_div();
  let hsz = textureDimensions(dda_hit);
  if (texel.x >= hsz.x || texel.y >= hsz.y) { return; }

  let gpx = (vec2<f32>(f32(px.x), f32(px.y)) + vec2<f32>(0.5))
          / vec2<f32>(f32(size.x), f32(size.y));
  let guv = vec2<f32>(gpx.x * 2.0 - 1.0, 1.0 - gpx.y * 2.0);
  let gc_near_h = view_u.inv_view_proj * vec4<f32>(guv.x, guv.y, 0.0, 1.0);
  let gc_far_h = view_u.inv_view_proj * vec4<f32>(guv.x, guv.y, 1.0, 1.0);
  let dir_c = normalize(gc_far_h.xyz / gc_far_h.w - gc_near_h.xyz / gc_near_h.w);
  let t_c = dot(hit.point - origin_voxel, dir_c);
  let pack_w = ((gi_u.seq.x & 0xFFu) << 24u) | ((hit.pal & 0x7FFFu) << 1u) | u32(hit.hit);
  textureStore(dda_hit, texel,
               vec4<u32>(bitcast<u32>(select(0.0, t_c, hit.hit)),
                         gi_key0(hit.voxel, hit.obj_id),
                         gi_key1(hit.voxel, hit.face_id, hit.obj_id),
                         pack_w));
}

fn dda_hit_unpack(raw: vec4<u32>, origin_voxel: vec3<f32>, dir_voxel: vec3<f32>) -> WorldHit {
  let t = bitcast<f32>(raw.x);
  let obj = i32((raw.z >> 19u) & 0x1FFFu) - 1;
  let salt = gi_pose_salt(obj);
  let face = ((raw.z ^ salt) >> 16u) & 7u;
  let g = make_grid(u32(obj + 1));
  var voxel = gi_unkey_voxel(raw.y, raw.z, obj);
  let n = face_normal_world(g, face);
  let point = origin_voxel + dir_voxel * t;
  if (obj >= 0) {
    voxel = voxel_at_point(g, point - n * (0.5 * g.scale));
  }
  return WorldHit(true, t, point, (raw.w >> 1u) & 0x7FFFu,
                  n, face, voxel, obj, g.palette_base);
}

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
  let need_n = (light_u.g.base_flags & BASE_FLAG_IMPLICIT_NORMAL) != 0u;
  var n_own = vec3<f32>(0.0);
  if (first.hit && need_n) {
    let fgrid = make_grid(u32(first.obj_id) + 1u);
    n_own = voxel_normal_fast(fgrid, first.voxel, first.face_id, first.n);
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
  var best_rec = first;
  col = shade_face(first, origin_voxel, dir_voxel, frustum_length, px, sun_dir, sky, sun_c,
                   n_own, need_n, u32(gid.y) * size.x + u32(gid.x),
                   &best_rec, FaceShadow(vec3<f32>(0.0), false));
  if (gi_enabled() && gi_u.params.y > 0.0) { dda_hit_record(gid, size, best_rec, origin_voxel); }
  if (!first.hit) {
    col = col + godray_sample_bilinear(fog_tex, vec2<f32>(textureDimensions(fog_tex)), px);
  }
  if ((light_u.g.base_flags & BASE_FLAG_VOL_TINT) != 0u) {
    col = vol_tint_color(first);
  }
  write_output(coord0, col);
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

fn refl_ray_accum(px: u32, v: vec3<f32>) {
  if (px >= refl_cfg.y) { return; }
  let q = clamp(v, vec3<f32>(0.0), vec3<f32>(63.0)) * 65536.0 + 0.5;
  let o = px * 3u;
  atomicAdd(&refl_ray[o + 0u], u32(q.x));
  atomicAdd(&refl_ray[o + 1u], u32(q.y));
  atomicAdd(&refl_ray[o + 2u], u32(q.z));
}

fn refl_shade_ray(org: vec3<f32>, dir: vec3<f32>, tier: u32,
                  sun_dir: vec3<f32>, sun_c: vec3<f32>, sky: vec3<f32>) -> vec3<f32> {
  let g0w = make_grid(0u);
  let rhit = world_raycast(g0w, org, dir, REFL_GLASS_FAR_T, world_cfg_reflect(0.0, 3u));
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
    var r_gi = vec3<f32>(0.0);
    if (tier >= REFL_TIER_GI) {
      r_gi = refl_hit_gi(rhit.voxel, rhit.face_id, rhit.obj_id, rhit.pal, rhit.point);
    }
    rc = brdf_reflected(mr, rn, face_view_dir(org, r_vc),
                        SurfaceLight(sun_dir, sun_c * r_vis, 1.0, sky * GLASS_REFL_AMBIENT, r_gi))
      + emissive_radiance(mr, gi_emit_gain());
  }
  return rc;
}

@compute @workgroup_size(64, 1, 1)
fn dda_refl(@builtin(global_invocation_id) gid: vec3<u32>,
            @builtin(num_workgroups) nwg: vec3<u32>) {
  let count = min(atomicLoad(&refl_meta[0]), refl_cfg.x);
  let stride = nwg.x * 64u;
  var i = gid.x;
  loop {
    if (i >= count) { break; }
    let a = refl_req[i * 3u];
    let b = refl_req[i * 3u + 1u];
    let aux = refl_req[i * 3u + 2u];
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
      let pal = a.z;
      let obj_id = i32((k1 >> 19u) & 0x1FFFu) - 1;
      let face_id = b.y;
      let voxel = gi_unkey_voxel(k0, k1, obj_id);
      let gg = make_grid(u32(obj_id + 1));
      let p_voxel = voxel_center_world(gg, voxel);
      let hn = voxel_normal_fast(gg, voxel, face_id, face_normal_world(gg, face_id));
      let hm = fetch_material(b.x, pal, p_voxel, hn, PBR_DEBUG_ASSET);
      let hv = face_view_dir(view_u.cam_pos_voxel.xyz, p_voxel);
      let rseed = refl_seed(voxel, face_id, obj_id);
      let merge_gi = tier >= REFL_TIER_GI && gi_enabled();
      rad = refl_env_chain(p_voxel, hn, hv, rseed, hm,
                           REFL_FAR_T, surface_offset(gg, hn, SHADOW_SURFACE_EPS), merge_gi);
    }
    if ((kind == REFL_KIND_RAY) && aux.z == 1u) { refl_ray_accum(px, w * rad); } else { refl_accum(px, w * rad); }
    i = i + stride;
  }
}

@compute @workgroup_size(8, 8, 1)
fn dda_refl_compose(@builtin(global_invocation_id) gid: vec3<u32>) {
  let size = textureDimensions(out_tex);
  if (gid.x >= size.x || gid.y >= size.y) { return; }
  let coord = vec2<i32>(i32(gid.x), i32(gid.y));
  let px = u32(gid.y) * size.x + u32(gid.x);
  if (atomicLoad(&refl_meta[0]) == 0u) {
    textureStore(out_tex, coord, textureLoad(out_base_tex, coord, 0));
    return;
  }
  let base = srgb_to_linear(textureLoad(out_base_tex, coord, 0).rgb);
  var px_src = px;
  var ray_px = REFL_FIRST_NONE;
  let vis = textureLoad(dda_vis, coord);
  let vpal = vis_pal(vis);
  if ((vis.w & 1u) != 0u && vpal != 0u) {
    let vobj = i32((vis.z >> 19u) & 0x1FFFu) - 1;
    let vgg = make_grid(u32(vobj) + 1u);
    if ((palette_flags(vgg.palette_base, vpal) & FLAGS_TRANSMISSIVE) == 0u) {
      let vdiff = split_ray(vec3<u32>(gid.x, gid.y, 0u), textureDimensions(dda_vis));
      let vh = dda_hit_unpack(vis, view_u.cam_pos_voxel.xyz, normalize(vdiff));
      if (vh.hit) {
        let hgg = make_grid(u32(vh.obj_id) + 1u);
        let vn = textureLoad(dda_nrm, coord).xyz;
        let vnf = voxel_face_index(hgg, vn, vh.obj_id);
        let fpx = refl_first_lookup(gi_key0(vh.voxel, vh.obj_id),
                                    gi_key1(vh.voxel, vnf, vh.obj_id));
        if (fpx != REFL_FIRST_NONE && fpx < refl_cfg.y) { px_src = fpx; }
      }
    } else {
      let vdiff = split_ray(vec3<u32>(gid.x, gid.y, 0u), textureDimensions(dda_vis));
      let vh = dda_hit_unpack(vis, view_u.cam_pos_voxel.xyz, normalize(vdiff));
      if (vh.hit) {
        let fpx = refl_first_lookup(gi_key0(vh.voxel, vh.obj_id),
                                    gi_key1(vh.voxel, vh.face_id, vh.obj_id));
        if (fpx != REFL_FIRST_NONE && fpx < refl_cfg.y) { ray_px = fpx; }
      }
    }
  }
  let o = px_src * 3u;
  var add = vec3<f32>(f32(atomicLoad(&refl_sum[o])),
                      f32(atomicLoad(&refl_sum[o + 1u])),
                      f32(atomicLoad(&refl_sum[o + 2u])))
          * (1.0 / 65536.0);
  if (ray_px != REFL_FIRST_NONE) {
    let ro = ray_px * 3u;
    add = add + vec3<f32>(f32(atomicLoad(&refl_ray[ro])),
                          f32(atomicLoad(&refl_ray[ro + 1u])),
                          f32(atomicLoad(&refl_ray[ro + 2u]))) * (1.0 / 65536.0);
  }
  let exposure = bitcast<f32>(eye_adapt_ro[0u]);
  let lin = base + add * select(1.0, exposure, exposure > 0.0);
  textureStore(out_tex, coord, vec4<f32>(linear_to_srgb(lin) + dither3(coord), 1.0));
}

const VIS_CAM: u32 = 0xFFFFFFFFu;
const VIS_SEQ_SHIFT: u32 = 24u;
const VIS_PAL_MASK: u32 = 0x7FFFFFu;

fn glass_push(px: u32) {
  let i = atomicAdd(&glass_meta[0], 1u);
  if (i >= arrayLength(&glass_req)) { return; }
  glass_req[i] = px;
}

fn vis_pack(hit: bool, pal: u32) -> u32 {
  return ((gi_u.seq.x & 0xFFu) << VIS_SEQ_SHIFT) | ((pal & VIS_PAL_MASK) << 1u) | u32(hit);
}

fn vis_pal(raw: vec4<u32>) -> u32 {
  return (raw.w >> 1u) & VIS_PAL_MASK;
}

fn vis_store_cam(coord: vec2<i32>, cam_pal: u32) {
  textureStore(dda_vis, coord, vec4<u32>(VIS_CAM, 0u, 0u, vis_pack(true, cam_pal)));
  textureStore(dda_nrm, coord, vec4<f32>(0.0, 0.0, 0.0, 0.0));
}

fn shadow_store(coord: vec2<i32>, vis: vec3<f32>) {
  textureStore(dda_shadow_tex, coord, vec4<f32>(vis, 1.0));
}

fn split_ray(gid: vec3<u32>, size: vec2<u32>) -> vec3<f32> {
  let px = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5)) / vec2<f32>(f32(size.x), f32(size.y));
  let uv = vec2<f32>(px.x * 2.0 - 1.0, 1.0 - px.y * 2.0);
  let near_world_h = view_u.inv_view_proj * vec4<f32>(uv.x, uv.y, 0.0, 1.0);
  let far_world_h = view_u.inv_view_proj * vec4<f32>(uv.x, uv.y, 1.0, 1.0);
  return far_world_h.xyz / far_world_h.w - near_world_h.xyz / near_world_h.w;
}

@compute @workgroup_size(8, 8, 1)
fn dda_visible(@builtin(global_invocation_id) gid: vec3<u32>,
               @builtin(local_invocation_index) lid: u32) {
  if (lid == 0u) {
    cam_pal_wg = camera_solid_pal(view_u.cam_pos_voxel.xyz);
  }
  workgroupBarrier();
  let size = textureDimensions(dda_vis);
  if (gid.x >= size.x || gid.y >= size.y) { return; }
  let coord = vec2<i32>(i32(gid.x), i32(gid.y));
  if (cam_pal_wg != 0u) {
    vis_store_cam(coord, cam_pal_wg);
    return;
  }
  let diff_world = split_ray(gid, size);
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
  var n = vec3<f32>(0.0);
  if (first.hit && (light_u.g.base_flags & BASE_FLAG_IMPLICIT_NORMAL) != 0u) {
    n = voxel_normal_fast(make_grid(u32(first.obj_id) + 1u), first.voxel, first.face_id, first.n);
    if ((palette_flags(first.palette_base, first.pal) & FLAGS_TRANSMISSIVE) != 0u) {
      n = first.n;
    }
  }
  textureStore(dda_nrm, coord, vec4<f32>(n, 0.0));
  textureStore(dda_vis, coord, vec4<u32>(
    select(0u, bitcast<u32>(first.t), first.hit),
    gi_key0(first.voxel, first.obj_id),
    gi_key1(first.voxel, first.face_id, first.obj_id),
    vis_pack(first.hit, first.pal)));
}

@compute @workgroup_size(8, 8, 1)
fn dda_shadow(@builtin(global_invocation_id) gid: vec3<u32>) {
  let size = textureDimensions(dda_shadow_tex);
  if (gid.x >= size.x || gid.y >= size.y) { return; }
  let coord = vec2<i32>(i32(gid.x), i32(gid.y));
  let raw = textureLoad(dda_vis, coord);
  let pal = vis_pal(raw);
  let hit = (raw.w & 1u) != 0u;
  let flags = light_u.g.base_flags;
  let normals_on = (flags & BASE_FLAG_IMPLICIT_NORMAL) != 0u;
  let shadow_on = normals_on && (flags & BASE_FLAG_SHADOW) != 0u;
  if (raw.x == VIS_CAM || !hit || pal == 0u) {
    shadow_store(coord, vec3<f32>(1.0));
    return;
  }
  let obj_id = i32((raw.z >> 19u) & 0x1FFFu) - 1;
  let g = make_grid(u32(obj_id + 1));
  let sun_dir = light_u.lights[0].kind_pos_dir.yzw;
  let face_id = ((raw.z ^ gi_pose_salt(obj_id)) >> 16u) & 7u;
  var voxel = gi_unkey_voxel(raw.y, raw.z, obj_id);
  if (obj_id >= 0) {
    let diff = split_ray(vec3<u32>(u32(coord.x), u32(coord.y), 0u), size);
    let dir = normalize(diff);
    let point = view_u.cam_pos_voxel.xyz + dir * bitcast<f32>(raw.x);
    voxel = voxel_at_point(g, point - face_normal_world(g, face_id) * (0.5 * g.scale));
  }
  if ((palette_flags(g.palette_base, pal) & FLAGS_TRANSMISSIVE) != 0u) {
    var vis = vec3<f32>(0.0);
    if (normals_on) {
      let n = textureLoad(dda_nrm, coord).xyz;
      if (dot(n, sun_dir) > 0.0) {
        let p = voxel_center_world(g, voxel);
        let off = surface_offset(g, n, SHADOW_SURFACE_EPS);
        vis = sun_transmittance(p + n * off, sun_dir, voxel, obj_id);
      }
    }
    shadow_store(coord, vis);
    glass_push(u32(coord.y) * size.x + u32(coord.x));
    return;
  }
  if (!shadow_on) {
    shadow_store(coord, vec3<f32>(1.0));
    return;
  }
  let n = textureLoad(dda_nrm, coord).xyz;
  if (dot(n, sun_dir) <= 0.0) {
    shadow_store(coord, vec3<f32>(0.0));
    return;
  }
  let p = voxel_center_world(g, voxel);
  let off = surface_offset(g, n, SHADOW_SURFACE_EPS);
  shadow_store(coord, sun_transmittance(p + n * off, sun_dir, voxel, obj_id));
}

fn shade_split(first: WorldHit, origin_voxel: vec3<f32>, ray_dir: vec3<f32>, uv: vec2<f32>, qpx: u32,
               n: vec3<f32>, sun_dir: vec3<f32>, sky: vec3<f32>, sun_c: vec3<f32>,
               sun_vis: vec3<f32>) -> vec3<f32> {
  if (!first.hit) { return sky_primary(ray_dir, sun_dir, sun_c); }
  let gg = make_grid(u32(first.obj_id) + 1u);
  let p_voxel = voxel_center_world(gg, first.voxel);
  if ((light_u.g.base_flags & BASE_FLAG_IMPLICIT_NORMAL) == 0u) {
    return fetch_material(first.palette_base, first.pal, p_voxel, normalize(gg.col1), PBR_DEBUG_ASSET).albedo;
  }
  let k0 = gi_key0(first.voxel, first.obj_id);
  let k1 = gi_key1(first.voxel, first.face_id, first.obj_id);
  let m = fetch_material(first.palette_base, first.pal, p_voxel, n, PBR_DEBUG_ASSET);
  let v = face_view_dir(origin_voxel, p_voxel);
  let nf = voxel_face_index(gg, n, first.obj_id);
  if (gi_enabled() && gi_u.params.y > 0.0
      && (palette_flags(first.palette_base, first.pal) & FLAGS_TRANSMISSIVE) == 0u) {
    let le = emissive_radiance(m, gi_emit_gain());
    if (max(le.x, max(le.y, le.z)) > 0.0) {
      lt_register(first.voxel, nf, u32(first.obj_id + 1), dot(le, GI_LUMA));
    }
  }
  let kv1 = gi_key1(first.voxel, nf, first.obj_id);
  var gi = vec3<f32>(0.0);
  var cov = 0.0;
  if (gi_enabled()) {
    let s = gi_surface(first, gg, n, nf);
    let d = irc_display(s.addr, s.n);
    gi = d.rgb;
    cov = d.a;
  }
  let amb = gi_ambient_fallback(n, cov);
  var col = brdf_reflected(m, n, v, SurfaceLight(sun_dir, sun_c * sun_vis, 1.0, amb, gi))
          + emissive_radiance(m, gi_emit_gain());
  if (PBR_REFLECTION_ENABLED != 0u && light_u.g.refl_tier > 0u) {
    let f0r = f0_of(m);
    let f0_luma = dot(f0r, GI_LUMA);
    if (f0_luma >= PBR_REFL_MIN_F0 && m.roughness <= PBR_REFL_MAX_ROUGH) {
      let fade = clamp(1.0 - m.roughness / PBR_REFL_MAX_ROUGH, 0.0, 1.0);
      col = col - fade * f0r * (amb + gi);
      queue_refl_chain(k0, kv1, first.pal, first.palette_base, nf,
                       fresnel_schlick(f0r, max(dot(v, n), 0.0)) * fade, qpx, true);
    }
  }
  return col;
}

@compute @workgroup_size(8, 8, 1)
fn dda_shade(@builtin(global_invocation_id) gid: vec3<u32>) {
  let size = textureDimensions(dda_vis);
  if (gid.x >= size.x || gid.y >= size.y) { return; }
  let coord = vec2<i32>(i32(gid.x), i32(gid.y));
  let raw = textureLoad(dda_vis, coord);
  let px = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5)) / vec2<f32>(f32(size.x), f32(size.y));
  if (raw.x == VIS_CAM) {
    write_output(coord, palette_rgb_linear(make_grid(0u).palette_base, vis_pal(raw)));
    return;
  }
  let diff_world = split_ray(gid, size);
  let dir_voxel = normalize(diff_world);
  let origin_voxel = view_u.cam_pos_voxel.xyz;
  let hit = (raw.w & 1u) != 0u;
  let pal = vis_pal(raw);
  var first = world_miss();
  if (hit && pal != 0u) {
    first = dda_hit_unpack(raw, origin_voxel, dir_voxel);
  }
  if (first.hit && (palette_flags(first.palette_base, pal) & FLAGS_TRANSMISSIVE) != 0u) { return; }
  let sun_dir = light_u.lights[0].kind_pos_dir.yzw;
  let sky = light_u.sky_color.xyz;
  let sun_c = light_u.lights[0].color_intensity.xyz * light_u.lights[0].color_intensity.w;
  let n = textureLoad(dda_nrm, coord).xyz;
  let sun_vis = textureLoad(dda_shadow_tex, coord).rgb;
  var col = shade_split(first, origin_voxel, dir_voxel, px, u32(gid.y) * size.x + u32(gid.x),
                        n, sun_dir, sky, sun_c, sun_vis);
  if (!first.hit) {
    col = col + godray_sample_bilinear(fog_tex, vec2<f32>(textureDimensions(fog_tex)), px);
  }
  if ((light_u.g.base_flags & BASE_FLAG_VOL_TINT) != 0u) { col = vol_tint_color(first); }
  write_output(coord, col);
  if (gi_enabled() && gi_u.params.y > 0.0) { dda_hit_record(gid, size, first, origin_voxel); }
}

@compute @workgroup_size(64, 1, 1)
fn dda_glass(@builtin(global_invocation_id) gid: vec3<u32>,
             @builtin(num_workgroups) nwg: vec3<u32>) {
  let count = min(atomicLoad(&glass_meta[0]), arrayLength(&glass_req));
  if (gid.x >= count) { return; }
  let size = textureDimensions(dda_vis);
  let stride = nwg.x * 64u;
  let sun_dir = light_u.lights[0].kind_pos_dir.yzw;
  let sky = light_u.sky_color.xyz;
  let sun_c = light_u.lights[0].color_intensity.xyz * light_u.lights[0].color_intensity.w;
  let need_n = (light_u.g.base_flags & BASE_FLAG_IMPLICIT_NORMAL) != 0u;
  let origin_voxel = view_u.cam_pos_voxel.xyz;
  var i = gid.x;
  loop {
    if (i >= count) { break; }
    let px = glass_req[i];
    i = i + stride;
    let coord = vec2<i32>(i32(px % size.x), i32(px / size.x));
    let raw = textureLoad(dda_vis, coord);
    let pal = vis_pal(raw);
    if (raw.x == VIS_CAM || (raw.w & 1u) == 0u || pal == 0u) { continue; }
    let pgid = vec3<u32>(u32(coord.x), u32(coord.y), 0u);
    let diff_world = split_ray(pgid, size);
    let frustum_length = length(diff_world);
    let dir_voxel = normalize(diff_world);
    let first = dda_hit_unpack(raw, origin_voxel, dir_voxel);
    if ((palette_flags(first.palette_base, first.pal) & FLAGS_TRANSMISSIVE) == 0u) { continue; }
    let uv = (vec2<f32>(f32(coord.x), f32(coord.y)) + vec2<f32>(0.5))
           / vec2<f32>(f32(size.x), f32(size.y));
    let n_hint = textureLoad(dda_nrm, coord).xyz;
    let f_shadow = FaceShadow(textureLoad(dda_shadow_tex, coord).rgb, need_n);
    var best = first;
    var col = shade_face(first, origin_voxel, dir_voxel, frustum_length, uv, sun_dir, sky, sun_c,
                     n_hint, need_n, u32(coord.y) * size.x + u32(coord.x), &best, f_shadow);
    if ((light_u.g.base_flags & BASE_FLAG_VOL_TINT) != 0u) { col = vol_tint_color(first); }
    write_output(coord, col);
    if (gi_enabled() && gi_u.params.y > 0.0) { dda_hit_record(pgid, size, best, origin_voxel); }
  }
}

@compute @workgroup_size(64, 1, 1)
fn gi_light_sweep(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= gi_u.seq.w) { return; }
  lt_sweep_slot(gid.x);
}

const RES_SPATIAL_TAPS: u32 = 0u;

@compute @workgroup_size(8, 8, 1)
fn gi_restir_temporal(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (!gi_enabled() || gi_u.params.y <= 0.0) { return; }
  let size = textureDimensions(dda_hit);
  if (gid.x >= size.x || gid.y >= size.y) { return; }
  let idx = gid.y * size.x + gid.x;
  let coord = vec2<i32>(i32(gid.x), i32(gid.y));
  let raw = textureLoad(dda_hit, coord);
  if ((raw.w & 1u) == 0u || ((raw.w >> 24u) != (gi_u.seq.x & 0xFFu))) {
    res_store_tmp(idx, res_none());
    return;
  }
  let px = gi_record_pixel(vec2<u32>(gid.x, gid.y));
  let gsz = vec2<f32>(textureDimensions(dda_vis));
  let p = (vec2<f32>(f32(px.x), f32(px.y)) + vec2<f32>(0.5)) / gsz;
  let ndc = vec2<f32>(p.x * 2.0 - 1.0, 1.0 - p.y * 2.0);
  let nw4 = view_u.inv_view_proj * vec4<f32>(ndc.x, ndc.y, 0.0, 1.0);
  let fw4 = view_u.inv_view_proj * vec4<f32>(ndc.x, ndc.y, 1.0, 1.0);
  let nw = nw4.xyz / nw4.w;
  let fw = fw4.xyz / fw4.w;
  let dir = normalize(fw - nw);
  let seed = gi_mix(u32(px.x) * 0x9E3779B1u
                    + gi_mix(u32(px.y) * 0x85EBCA77u + gi_u.seq.x * 0x27D4EB2Fu));
  var h = dda_hit_unpack(raw, view_u.cam_pos_voxel.xyz, dir);
  if (!h.hit) { res_store_tmp(idx, res_none()); return; }
  if ((palette_flags(h.palette_base, h.pal) & FLAGS_TRANSMISSIVE) != 0u) {
    let hm = fetch_material(h.palette_base, h.pal, h.point,
                            face_normal_world(make_grid(u32(h.obj_id) + 1u), h.face_id),
                            PBR_DEBUG_ASSET);
    if (hm.roughness < 0.1) { res_store_tmp(idx, res_none()); return; }
    var tint = vec3<f32>(1.0);
    let bh = world_trace_medium_scene(view_u.cam_pos_voxel.xyz, dir,
                                      length(fw - nw), false, false, &tint);
    if (!bh.hit) { res_store_tmp(idx, res_none()); return; }
    h = bh;
  }
  let g = make_grid(u32(h.obj_id) + 1u);
  let n = voxel_normal_fast(g, h.voxel, h.face_id, h.n);
  let s = gi_surface(h, g, n, h.face_id);
  var r = res_begin(s.p, s.n, gi_key0(h.voxel, h.obj_id),
                    gi_key1(h.voxel, h.face_id, h.obj_id), gi_u.seq.z);
  let clip = gi_u.prev_view_proj * vec4<f32>(h.point, 1.0);
  if (abs(clip.w) > 1e-6) {
    let pn = clip.xyz / clip.w;
    let uv = vec2<f32>(pn.x * 0.5 + 0.5, 0.5 - pn.y * 0.5);
    let pp = vec2<i32>(floor(uv * vec2<f32>(f32(size.x), f32(size.y))));
    if (pp.x >= 0 && pp.y >= 0 && pp.x < i32(size.x) && pp.y < i32(size.y)) {
      let prev = res_load_prev(u32(pp.y) * size.x + u32(pp.x));
      let m0 = r.m;
      r = res_adopt(r, prev, gi_rand2(seed).x);
      if (r.m > m0) { atomicAdd(&gi_stats[GI_STAT_RES_TEMPORAL], 1u); }
    }
  }
  let pick = lt_pick(s.p, s.n, gi_mix(seed + 0x68E31DA4u));
  var pick_vis = 0.0;
  if (pick.ok) {
    let origin = gi_origin(s);
    let d = pick.y - origin;
    let dist = length(d);
    var tint = vec3<f32>(1.0);
    let hit = world_trace_medium_scene(origin, d / max(dist, 1e-6),
                                       max(dist - SHADOW_SURFACE_EPS * g.scale, 0.0),
                                       false, false, &tint);
    pick_vis = select(1.0, 0.0, hit.hit);
  }
  r = res_add_pick(r, pick, pick_vis, gi_rand2(gi_mix(seed + 0x517CC1B7u)).x);
  r = res_finish(r);
  res_store_tmp(idx, r);
}

@compute @workgroup_size(8, 8, 1)
fn gi_restir_spatial(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (!gi_enabled() || gi_u.params.y <= 0.0) { return; }
  let size = textureDimensions(dda_hit);
  if (gid.x >= size.x || gid.y >= size.y) { return; }
  let idx = gid.y * size.x + gid.x;
  let coord = vec2<i32>(i32(gid.x), i32(gid.y));
  var r = res_load_tmp(idx);
  let offs = array<vec2<i32>, 8>(vec2<i32>(1, 0), vec2<i32>(-1, 0),
                                 vec2<i32>(0, 1), vec2<i32>(0, -1),
                                 vec2<i32>(1, 1), vec2<i32>(1, -1),
                                 vec2<i32>(-1, 1), vec2<i32>(-1, -1));
  let seed = gi_mix(idx * 0x9E3779B1u + gi_u.seq.x * 0x27D4EB2Fu);
  for (var i = 0u; r.valid && i < RES_SPATIAL_TAPS; i = i + 1u) {
    let nc = coord + offs[i];
    if (nc.x < 0 || nc.y < 0 || nc.x >= i32(size.x) || nc.y >= i32(size.y)) { continue; }
    let nb = res_load_tmp(u32(nc.y) * size.x + u32(nc.x));
    if (nb.rev != r.rev) { continue; }
    let m0 = r.m;
    r = res_add_nb(r, nb, gi_rand2(gi_mix(seed + i * 0x9E3779B9u)).x);
    if (r.m > m0) { atomicAdd(&gi_stats[GI_STAT_RES_SPATIAL], 1u); }
  }
  if (r.valid && r.rev != gi_u.seq.z) { r = res_none(); }
  res_store_cur(idx, res_finish(r));
}

@compute @workgroup_size(8, 8, 1)
fn gi_gather(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (!gi_enabled() || gi_u.params.y <= 0.0) { return; }
  let size = textureDimensions(dda_hit);
  if (gid.x >= size.x || gid.y >= size.y) { return; }
  let coord = vec2<i32>(i32(gid.x), i32(gid.y));
  let raw = textureLoad(dda_hit, coord);
  if ((raw.w & 1u) == 0u || ((raw.w >> 24u) != (gi_u.seq.x & 0xFFu))) { return; }
  let px = gi_record_pixel(vec2<u32>(gid.x, gid.y));
  let gsz = vec2<f32>(textureDimensions(dda_vis));
  let p = (vec2<f32>(f32(px.x), f32(px.y)) + vec2<f32>(0.5)) / gsz;
  let ndc = vec2<f32>(p.x * 2.0 - 1.0, 1.0 - p.y * 2.0);
  let nw4 = view_u.inv_view_proj * vec4<f32>(ndc.x, ndc.y, 0.0, 1.0);
  let fw4 = view_u.inv_view_proj * vec4<f32>(ndc.x, ndc.y, 1.0, 1.0);
  let nw = nw4.xyz / nw4.w;
  let fw = fw4.xyz / fw4.w;
  let dir = normalize(fw - nw);
  var h = dda_hit_unpack(raw, view_u.cam_pos_voxel.xyz, dir);
  if (!h.hit) { return; }
  let seed = gi_mix(u32(px.x) * 0x9E3779B1u + gi_mix(u32(px.y) * 0x85EBCA77u + gi_u.seq.x * 0x27D4EB2Fu));
  if ((palette_flags(h.palette_base, h.pal) & FLAGS_TRANSMISSIVE) != 0u) {
    let hm = fetch_material(h.palette_base, h.pal, h.point,
                            face_normal_world(make_grid(u32(h.obj_id) + 1u), h.face_id),
                            PBR_DEBUG_ASSET);
    if (hm.roughness < 0.1) { return; }
    var tint = vec3<f32>(1.0);
    let bh = world_trace_medium_scene(view_u.cam_pos_voxel.xyz, dir,
                                      length(fw - nw), false, false, &tint);
    if (!bh.hit) { return; }
    h = bh;
  }
  let g = make_grid(u32(h.obj_id) + 1u);
  let n = voxel_normal_fast(g, h.voxel, h.face_id, h.n);
  let s = gi_surface(h, g, n, h.face_id);
  let vis = textureLoad(dda_vis, vec2<i32>(px));
  var sun = -1.0;
  if ((vis.w & 1u) != 0u
      && gi_key0(h.voxel, h.obj_id) == vis.y && gi_key1(h.voxel, h.face_id, h.obj_id) == vis.z) {
    sun = dot(textureLoad(dda_shadow_tex, vec2<i32>(px)).rgb, GI_LUMA);
  }
  if (sun < 0.0) { sun = gi_sun_vis(s); }
  let maturity = irc_lookup(s.addr, s.face);
  var spp = min(gi_sample_base() * gi_sample_boost(maturity.n), GI_SPP_MAX);
  if (gi_u.flags.w > 0.5) { spp = min(gi_sample_base(), 1u); }
  let seed0 = gi_mix(seed + 0x3C6EF372u);
  atomicAdd(&gi_stats[GI_STAT_RAYS], spp);
  var indirect = vec3<f32>(0.0);
  for (var i = 0u; i < spp; i = i + 1u) {
    indirect = indirect + gi_leaf_indirect(s, gi_mix(seed0 + i * 0x9E3779B1u));
  }
  indirect = indirect / f32(spp);
  if (maturity.conf > 0.5) {
    let base = dot(maturity.e, GI_LUMA);
    let il = dot(indirect, GI_LUMA);
    if (base > 1e-5 && il > base * GI_SAMPLE_CLAMP) {
      indirect = indirect * (base * GI_SAMPLE_CLAMP / il);
    }
  }
  let rr = res_load_cur(gid.y * size.x + gid.x);
  var direct = vec3<f32>(0.0);
  if (rr.valid && rr.rev == gi_u.seq.z
      && rr.k0 == gi_key0(h.voxel, h.obj_id)
      && rr.k1 == gi_key1(h.voxel, h.face_id, h.obj_id)) {
    direct = res_direct(rr);
  }
  irc_accumulate(s.addr, s.n, direct + indirect, true, sun,
                 gi_rand2(gi_mix(seed + 0x9E3779B9u)));
}

@compute @workgroup_size(64, 1, 1)
fn gi_resolve(@builtin(global_invocation_id) gid: vec3<u32>) {
  let s = gid.x;
  if (s >= arrayLength(&irc) / IRC_WORDS) { return; }
  irc_resolve_slot(s);
}
