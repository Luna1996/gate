import package::bindings::{BASE_FLAG_IMPLICIT_NORMAL, BASE_FLAG_SHADOW, BASE_FLAG_VOL_TINT, BEAM_BACKOFF, BEAM_DIV, beam_depth, eye_adapt, eye_adapt_ro, face_slots, face_slots_flat, fog_tex, gi_dn_dst, gi_dn_tmp, gi_guide, gi_guide_out, gi_out, gi_out_flat, gi_tex, gi_u, light_samp, light_u, lod_diag, out_tex, out_tex_src, view_u};
import package::common::{FLAGS_TRANSMISSIVE, MaterialParams, SurfaceLight, brdf_reflected, dielectric_f0, emissive_radiance, f0_of, face_view_dir, fetch_material, fresnel_schlick, linear_to_srgb, material_ior, palette_flags, palette_rgb_linear, sky_rgb, srgb_to_linear};
import package::gi::common::{FACE_WORDS, FACE_W_COL, FACE_W_FLAG, FACE_W_GI, FACE_W_GI_N, FACE_W_K0, FACE_W_K1, FACE_W_PAL, FACE_W_RAW, FACE_W_RAW_N, FACE_W_SUN, FACE_W_TEXEL, face_slot_claim, face_slot_color, face_slot_find, face_slot_find_flat, gi_key0, gi_key1, gi_key_ok, gi_local_rev, gi_unkey_voxel, gi_vox_k1, gi_mix, gi_pack_dir, gi_rand2, gi_unpack_dir, GI_ACC_CLAMP, GI_ACC_MAX_N, GI_ACC_SCALE};
import package::gi::consts::{GI_AMBIENT_FLOOR, GI_DEN_GUIDE_WORDS, GI_DEN_G_NOISE, GI_DEN_G_KEY0, GI_DEN_G_KEY1, GI_DEN_G_NRM, GI_EMIT_GAIN, GI_LUMA, GI_PI, GI_SKY_AMBIENT, GI_SS_ADAPT_HI, GI_SS_ADAPT_LO, GI_SS_ADAPT_REF, SHADOW_SURFACE_EPS};
import package::gi::denoise::{gi_den_atrous, gi_den_temporal, gi_same_surface};
import package::gi::ray::{sun_transmittance};
import package::gi::screen::{gi_ss_main};
import package::gi::wal::{gi_wal_pool, GI_WAL_MATURE_N};
import package::trace::{diag_sampled, make_grid};
import package::volumetric::{godray_pass_blur_a, godray_pass_blur_b, godray_pass_mask, godray_sample_bilinear, sky_primary};
import package::world::{WorldHit, camera_solid_pal, face_normal_world, surface_offset, vol_tint_color, voxel_center_world, voxel_normal, world_cfg_full, world_cfg_primary, world_miss, world_raycast, world_trace_medium_scene};

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

const GLASS_EPS: f32 = 1e-3;
const GLASS_MAX_BOUNCE: u32 = 8u;
const GLASS_REFL_MIN: f32 = 0.02;
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
var<private> dbg_cov: f32 = 0.0;
var<private> dbg_sun: f32 = 0.0;
var<private> dbg_owner: f32 = 0.0;

struct GlassOut {
  ray: WorldHit,
  tint: vec3<f32>,
  refl: vec3<f32>,
  dir: vec3<f32>,
}

fn trace_glass(origin_voxel: vec3<f32>, dir_voxel: vec3<f32>, t_min: f32, max_t: f32,
               sun_dir: vec3<f32>, sky: vec3<f32>, sun_c: vec3<f32>) -> GlassOut {
  var org = origin_voxel;
  var dir = dir_voxel;
  var t_start = t_min;
  var in_medium = false;
  var tint = vec3<f32>(1.0);
  var refl = vec3<f32>(0.0);
  for (var bounce = 0u; bounce < GLASS_MAX_BOUNCE; bounce = bounce + 1u) {
    var seg = vec3<f32>(1.0);
    var best: WorldHit;
    if (in_medium) {
      best = world_trace_medium_scene(org, dir, max_t, true, &seg);
    } else {
      best = world_raycast(make_grid(0u), org, dir, max_t, world_cfg_primary(t_start, 3u));
    }
    tint = tint * seg;
    if (!best.hit) { return GlassOut(best, tint, refl, dir); }
    if ((palette_flags(best.palette_base, best.pal) & FLAGS_TRANSMISSIVE) == 0u) {
      return GlassOut(best, tint, refl, dir);
    }
    let p_hit = org + dir * best.t;
    let nn = voxel_normal(make_grid(0u), best.voxel, best.face_id, best.n);
    let ci = clamp(dot(-dir, nn), 0.0, 1.0);
    let ior = material_ior(best.palette_base, best.pal);
    let v = face_view_dir(view_u.cam_pos_voxel.xyz, voxel_center_world(make_grid(0u), best.voxel));
    if (!in_medium) {
      
      
      let f0 = dielectric_f0(ior, 1.0);
      let fres = f0 + (1.0 - f0) * pow(1.0 - clamp(dot(v, nn), 0.0, 1.0), 5.0);
      let r_tier = light_u.g.refl_tier;
      if (r_tier > 0u && fres > GLASS_REFL_MIN) {
        
        let rdir = reflect(-v, nn);
        let r_org = p_hit + nn * GLASS_EPS;
        let rhit = world_raycast(make_grid(0u), r_org, rdir,
                                 max_t, world_cfg_full(0.0, 3u));
        var rc = select(sky_rgb(), sky_primary(rdir, sun_dir, sun_c), r_tier >= 2u);
        if (rhit.hit) {
          let rgg = make_grid(u32(rhit.obj_id) + 1u);
          let rn = voxel_normal(rgg, rhit.voxel, rhit.face_id, rhit.n);
          let mr = fetch_material(rhit.palette_base, rhit.pal, rhit.point, rn, PBR_DEBUG_ASSET);
          var r_vis = vec3<f32>(1.0);
          if (r_tier >= 3u && (light_u.g.base_flags & BASE_FLAG_SHADOW) != 0u) {
            r_vis = sun_transmittance(
              voxel_center_world(rgg, rhit.voxel) + rn * surface_offset(rgg, rn, SHADOW_SURFACE_EPS),
              sun_dir, rhit.voxel, rhit.obj_id);
          }
          rc = brdf_reflected(mr, rn, face_view_dir(r_org, voxel_center_world(rgg, rhit.voxel)),
                              SurfaceLight(sun_dir, sun_c * r_vis, 1.0, sky * GLASS_REFL_AMBIENT, vec3<f32>(0.0)))
            + emissive_radiance(mr, GI_EMIT_GAIN);
        }
        refl = refl + tint * fres * rc;
      }
      let eta = 1.0 / ior;
      let k = 1.0 - eta * eta * (1.0 - ci * ci);
      dir = normalize(eta * dir + (eta * ci - sqrt(max(k, 0.0))) * nn);
      in_medium = true;
    } else {
      let eta = ior;
      let ci_out = clamp(dot(dir, nn), 0.0, 1.0);
      let k = 1.0 - eta * eta * (1.0 - ci_out * ci_out);
      if (k < 0.0) {
        dir = reflect(dir, nn);
      } else {
        dir = normalize(eta * dir - (eta * ci_out - sqrt(k)) * nn);
        in_medium = false;
      }
    }
    org = p_hit + dir * GLASS_EPS;
    t_start = 0.0;
  }
  return GlassOut(WorldHit(false, 0.0, vec3<f32>(0.0), 0u, vec3<f32>(0.0), 0u, vec3<i32>(0), 0, 0u),
                  tint, refl, dir);
}

fn write_output(coord: vec2<i32>, col: vec3<f32>) {
  let exposure = bitcast<f32>(eye_adapt_ro[0u]);
  let exposed = col * select(1.0, exposure, exposure > 0.0);
  textureStore(out_tex, coord, vec4<f32>(linear_to_srgb(exposed), 1.0));
}

const GI_UPSAMPLE_HOLE_R: i32 = 3;
const GI_FACE_POOL_R: i32 = 3;
const GI_FACE_POOL_MIN_N: u32 = 4u;

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
  let s = face_slot_find(gi_key0(voxel, obj_id), gi_vox_k1(gi_key1(voxel, face_id, obj_id)), &ok);
  if (!ok) { return vec4<f32>(0.0); }
  let fe = s * FACE_WORDS;
  let n = atomicLoad(&face_slots[fe + FACE_W_GI_N]);
  if (n == 0u) { return vec4<f32>(0.0); }
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
  let tier = light_u.g.refl_tier;
  let nest = light_u.g.refl_nest;
  var refl_env = vec3<f32>(0.0);
  var chain_w = vec3<f32>(1.0);
  var chain_org = r_org;
  var chain_dir = rdir;
  for (var lvl = 0u; lvl <= nest; lvl = lvl + 1u) {
    var r_tint = vec3<f32>(1.0);
    let h = world_trace_medium_scene(chain_org, chain_dir, frustum_length, false, &r_tint);
    if (!h.hit) {
      refl_env = refl_env
               + chain_w * r_tint
                 * select(sky_rgb(), sky_primary(chain_dir, sun_dir, sun_c), tier >= 2u);
      break;
    }
    let hgg = make_grid(u32(h.obj_id) + 1u);
    let hn = voxel_normal(hgg, h.voxel, h.face_id, h.n);
    let hm = fetch_material(h.palette_base, h.pal, h.point, hn, PBR_DEBUG_ASSET);
    let hamb = sky * GI_SKY_AMBIENT;
    let hv = face_view_dir(chain_org, voxel_center_world(hgg, h.voxel));
    let h_off = 0.5 / max(max(abs(hn.x), abs(hn.y)), max(abs(hn.z), 1e-3))
              + SHADOW_SURFACE_EPS;
    let h_start = h.point + hn * h_off;
    var h_vis = vec3<f32>(1.0);
    
    if (tier >= 3u && (light_u.g.base_flags & BASE_FLAG_SHADOW) != 0u) {
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
              n_hint: vec3<f32>, n_hint_ok: bool, sun_io: ptr<function, f32>) -> vec3<f32> {
  
  let sun_hint = *sun_io;
  *sun_io = 0.0;
  var best = first;
  var through = vec3<f32>(1.0);
  var refl = vec3<f32>(0.0);
  var glass_col = vec3<f32>(0.0);
  var glass_w = 0.0;
  if (first.hit && (palette_flags(first.palette_base, first.pal) & FLAGS_TRANSMISSIVE) != 0u) {
    let fg = make_grid(u32(first.obj_id) + 1u);
    let fp = voxel_center_world(fg, first.voxel);
    let fvuv = voxel_screen_uv(fp, gi_in.uv);
    let fnn = voxel_normal(fg, first.voxel, first.face_id, first.n);
    let f_off = surface_offset(fg, fnn, SHADOW_SURFACE_EPS);
    
    let f_face = fp + fnn * f_off;
    let fv = face_view_dir(origin_voxel, fp);
    let g = trace_glass(f_face, -fv, 0.0, frustum_length, sun_dir, sky, sun_c);
    best = g.ray;
    through = g.tint;
    refl = g.refl;
    let fm = fetch_material(first.palette_base, first.pal, fp, fnn, PBR_DEBUG_ASSET);
    glass_w = clamp(1.0 - fm.transmission, 0.0, 1.0);
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
    if (!n_reused) { n = voxel_normal(gg, best.voxel, best.face_id, best.n); }
    let m = fetch_material(best.palette_base, best.pal, p_voxel, n, PBR_DEBUG_ASSET);
    let v = face_view_dir(origin_voxel, p_voxel);
    let n_off = surface_offset(gg, n, SHADOW_SURFACE_EPS);
    let ndl = max(dot(n, sun_dir), 0.0);
    var sun = vec3<f32>(0.0);
    if (sun_hint >= 0.0) {
      sun = vec3<f32>(sun_hint);
    } else if (ndl > 0.0) {
      sun = vec3<f32>(1.0);
      if ((light_u.g.base_flags & BASE_FLAG_SHADOW) != 0u) {
        sun = sun_transmittance(p_voxel + n * n_off, sun_dir, best.voxel, best.obj_id);
      }
    }
    *sun_io = max(max(sun.x, sun.y), sun.z);
    let gi_on = gi_u.misc.x > 0.5;
    var gi = vec3<f32>(0.0);
    var cov = 0.0;
    if (gi_in.use_face) {
      gi = gi_in.face_avg;
      cov = gi_in.face_cov;
    } else if (gi_on) {
      var acc = gi_vox_gi(best.voxel, best.face_id, best.obj_id);
      acc = acc * vec4<f32>(2.0, 2.0, 2.0, 2.0);
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
      if (acc.w > 0.0) {
        gi = acc.xyz / (acc.w * GI_ACC_SCALE);
        cov = 1.0;
      } else {
        let gup = vec2<u32>(textureDimensions(gi_tex));
        let s = gi_upsample_joint(vuv, gup, k0, k1, n, vec3<f32>(0.0));
        let gw = max(s.a, 1e-4);
        gi = s.rgb / gw;
        cov = clamp(s.a, 0.0, 1.0);
      }
      let gsize = vec2<u32>(textureDimensions(gi_tex));
      let tp = vec2<i32>(floor(vuv * vec2<f32>(gsize)));
      if (tp.x >= 0 && tp.y >= 0 && tp.x < i32(gsize.x) && tp.y < i32(gsize.y)) {
        let gb = (u32(tp.y) * gsize.x + u32(tp.x)) * GI_DEN_GUIDE_WORDS;
        let gk0 = gi_guide_out[gb + GI_DEN_G_KEY0];
        let gk1 = gi_guide_out[gb + GI_DEN_G_KEY1];
        if ((gk0 | gk1) != 0u) {
          dbg_owner = select(0.5, 1.0, gk0 == k0 && gk1 == k1);
        }
      }
    }
    let amb = sky * GI_SKY_AMBIENT * mix(1.0, GI_AMBIENT_FLOOR, cov);
    dbg_gi = gi;
    dbg_cov = cov;
    dbg_sun = *sun_io;
    col = brdf_reflected(m, n, v, SurfaceLight(sun_dir, sun_c * sun, 1.0, amb, gi))
        + emissive_radiance(m, GI_EMIT_GAIN);
    if (PBR_REFLECTION_ENABLED != 0u && light_u.g.refl_tier > 0u) {
      let tier = light_u.g.refl_tier;
      let f0r = f0_of(m);
      let f0_luma = dot(f0r, GI_LUMA);
      if (f0_luma >= PBR_REFL_MIN_F0 && m.roughness <= PBR_REFL_MAX_ROUGH) {
        let refl_env = refl_env_chain(p_voxel, n, v,
                                      refl_seed(best.voxel, best.face_id, best.obj_id),
                                      m, frustum_length, n_off);
        let fade = clamp(1.0 - m.roughness / PBR_REFL_MAX_ROUGH, 0.0, 1.0);
        col = col - fade * f0r * (amb + gi)
            + fresnel_schlick(f0r, max(dot(v, n), 0.0)) * fade * refl_env;
      }
    }
    col = col * through + refl;
  } else {
    return mix(glass_col, sky_primary(ray_dir, sun_dir, sun_c) * through + refl, 1.0 - glass_w);
  }
  return mix(glass_col, col, 1.0 - glass_w);
}

const GI_FACE_FALLBACK: u32 = 1u;
const DIAG_FACE_EXACT: u32 = 3u;    
const DIAG_FACE_FB_HIT: u32 = 4u;   
const DIAG_FACE_FB_NOKEY: u32 = 5u; 
const DIAG_FACE_FB_MISS: u32 = 6u;  
const DIAG_FACE_INLINE: u32 = 7u;   
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

fn gi_cover_key(uv: vec2<f32>, key_out: ptr<function, vec2<u32>>,
                guide_off_out: ptr<function, u32>) -> bool {
  let gi_size = vec2<u32>(textureDimensions(gi_tex));
  let tp = vec2<i32>(floor(uv * vec2<f32>(f32(gi_size.x), f32(gi_size.y))));
  *guide_off_out = 0u;
  if (tp.x < 0 || tp.y < 0 || tp.x >= i32(gi_size.x) || tp.y >= i32(gi_size.y)) { return false; }
  let gb = (u32(tp.y) * gi_size.x + u32(tp.x)) * GI_DEN_GUIDE_WORDS;
  *guide_off_out = gb;
  *key_out = vec2<u32>(gi_guide_out[gb + GI_DEN_G_KEY0], gi_guide_out[gb + GI_DEN_G_KEY1]);
  return (*key_out).x != 0u || (*key_out).y != 0u;
}

var<workgroup> cam_pal_wg: u32;

@compute @workgroup_size(8, 8, 1)
fn dda_main(@builtin(global_invocation_id) gid: vec3<u32>,
            @builtin(local_invocation_index) lid: u32) {
  if (lid == 0u) {
    cam_pal_wg = camera_solid_pal(view_u.cam_pos_voxel.xyz);
  }
  workgroupBarrier();
  let size = textureDimensions(out_tex);
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
    n_own = voxel_normal(make_grid(u32(first.obj_id) + 1u), first.voxel, first.face_id, first.n);
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
  if (diag_on && first.hit) { atomicAdd(&lod_diag[DIAG_FACE_INLINE], 1u); }
  if (diag_on && !first.hit) { atomicAdd(&lod_diag[DIAG_FACE_SKY], 1u); }
  var sun_io = -1.0;
  col = shade_face(first, origin_voxel, dir_voxel, frustum_length,
                   GiInput(false, vec3<f32>(0.0), 0.0, px), sun_dir, sky, sun_c,
                   n_own, need_n, &sun_io);
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
  if (view_u.debug_mode.x > 0.5) {
    textureStore(out_tex, coord0, vec4<f32>(linear_to_srgb(dbg_gi), 1.0));
    return;
  }
  if (view_u.debug_mode.y > 0.5) {
    textureStore(out_tex, coord0, vec4<f32>(vec3<f32>(dbg_sun), 1.0));
    return;
  }
  if (view_u.debug_mode.w > 0.5) {
    var oc = vec3<f32>(0.0);
    if (dbg_owner > 0.9) {
      oc = vec3<f32>(0.0, 1.0, 0.0);
    } else if (dbg_owner > 0.4) {
      oc = vec3<f32>(1.0, 0.0, 0.0);
    }
    textureStore(out_tex, coord0, vec4<f32>(oc, 1.0));
    return;
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
  let g_nrm = gi_unpack_dir(gi_guide_out[gb + GI_DEN_G_NRM]);
  var ok = false;
  let slot = face_slot_find(k0, k1, &ok);
  if (!ok) { return; }
  let fe = slot * FACE_WORDS;
  if (atomicLoad(&face_slots[fe + FACE_W_TEXEL]) != texel) { return; }
  let face_id = (k1 >> 16u) & 7u;
  let obj_id = i32((k1 >> 19u) & 0x1FFFu) - 1;
  let voxel = gi_unkey_voxel(k0, k1, obj_id);
  let gg = make_grid(u32(obj_id) + 1u);
  let p_voxel = voxel_center_world(gg, voxel);
  let fnw = face_normal_world(gg, face_id);
  let first = WorldHit(true, 0.0, p_voxel, atomicLoad(&face_slots[fe + FACE_W_PAL]),
                       fnw, face_id, voxel, obj_id, gg.palette_base);
  if (diag_sampled(texel)) {
    atomicAdd(&lod_diag[DIAG_WAL_FACES], 1u);
    if (gi_u.params.y > 0.5) {
      let wp = gi_wal_pool(voxel, obj_id, first.pal, k1);
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
  let gi_uv = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5))
            / vec2<f32>(f32(gsize.x), f32(gsize.y));
  let ndc = vec2<f32>(gi_uv.x * 2.0 - 1.0, 1.0 - gi_uv.y * 2.0);
  let nw_h = view_u.inv_view_proj * vec4<f32>(ndc.x, ndc.y, 0.0, 1.0);
  let fw_h = view_u.inv_view_proj * vec4<f32>(ndc.x, ndc.y, 1.0, 1.0);
  let frustum_length = length(fw_h.xyz / fw_h.w - nw_h.xyz / nw_h.w);
  let sun_dir = light_u.lights[0].kind_pos_dir.yzw;
  let sky = light_u.sky_color.xyz;
  let sun_c = light_u.lights[0].color_intensity.xyz * light_u.lights[0].color_intensity.w;
  let gi_n = atomicLoad(&face_slots[fe + FACE_W_GI_N]);
  var gi_avg = vec3<f32>(0.0);
  if (gi_n > 0u) {
    let inv = 1.0 / (f32(gi_n) * GI_ACC_SCALE);
    gi_avg = vec3<f32>(
      f32(atomicLoad(&face_slots[fe + FACE_W_GI])),
      f32(atomicLoad(&face_slots[fe + FACE_W_GI + 1u])),
      f32(atomicLoad(&face_slots[fe + FACE_W_GI + 2u]))) * inv;
  }
  if (gi_n >= GI_ACC_MAX_N) {
    for (var i = 0u; i < 3u; i = i + 1u) {
      let w = fe + FACE_W_GI + i;
      atomicStore(&face_slots[w], atomicLoad(&face_slots[w]) >> 1u);
    }
    atomicStore(&face_slots[fe + FACE_W_GI_N], gi_n >> 1u);
  }
  if (gi_n <= GI_FACE_POOL_MIN_N) {
    var acc = vec3<f32>(0.0);
    var wsum = 0.0;
    for (var dy: i32 = -GI_FACE_POOL_R; dy <= GI_FACE_POOL_R; dy = dy + 1) {
      for (var dx: i32 = -GI_FACE_POOL_R; dx <= GI_FACE_POOL_R; dx = dx + 1) {
        let c = vec2<i32>(i32(gid.x) + dx, i32(gid.y) + dy);
        if (c.x < 0 || c.y < 0 || c.x >= i32(gsize.x) || c.y >= i32(gsize.y)) { continue; }
        let tb = (u32(c.y) * gsize.x + u32(c.x)) * GI_DEN_GUIDE_WORDS;
        let tk0 = gi_guide_out[tb + GI_DEN_G_KEY0];
        let tk1 = gi_guide_out[tb + GI_DEN_G_KEY1];
        if ((tk0 | tk1) == 0u) { continue; }
        if (!gi_same_surface(tk0, tk1, gi_guide_out[tb + GI_DEN_G_NRM], k0, k1, g_nrm)) { continue; }
        let v = textureLoad(gi_tex, c, 0);
        if (v.a <= 0.5) { continue; }
        acc = acc + v.rgb;
        wsum = wsum + 1.0;
      }
    }
    if (wsum > 0.5) { gi_avg = acc / wsum; }
  }
  var sun_io = -1.0;
  let col = shade_face(first, view_u.cam_pos_voxel.xyz, fnw, frustum_length,
                       GiInput(true, gi_avg, select(0.0, 1.0, gi_n > 0u), gi_uv),
                       sun_dir, sky, sun_c, g_nrm, true, &sun_io);
  atomicStore(&face_slots[fe + FACE_W_SUN], bitcast<u32>(sun_io));
  atomicStore(&face_slots[fe + FACE_W_COL], bitcast<u32>(col.x));
  atomicStore(&face_slots[fe + FACE_W_COL + 1u], bitcast<u32>(col.y));
  atomicStore(&face_slots[fe + FACE_W_COL + 2u], bitcast<u32>(col.z));
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

  let h = world_raycast(make_grid(0u), origin_voxel, dir_voxel, frustum_length,
                        world_cfg_full(t_min, 3u));
  gi_ss_main(coord, size, h, w_adapt);
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
  let slot = face_slot_find_flat(k0, k1, &ok);
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
