import package::bindings::{BEAM_BACKOFF, BEAM_DIV, DDGI_INDIR_CAST_BASE, DDGI_INDIR_COLLBASE_BASE, DDGI_INDIR_COLL_BASE, DDGI_INDIR_COLL_TOTAL, DDGI_INDIR_COUNT_BASE, DDGI_INDIR_RAYBASE_BASE, DDGI_INDIR_RAY_TOTAL, DDGI_INDIR_RPP_BASE, beam_depth, ddgi_args, ddgi_cell, ddgi_cell_id, ddgi_indirect, ddgi_lod0, ddgi_meta, ddgi_samples, ddgi_slot_pos, ddgi_u, ddgi_worklist, eye_adapt, eye_adapt_ro, gi_cov_out, gi_cov_tex, gi_out, gi_tex, light_samp, light_u, out_tex, out_tex_src, view_u};
import package::common::{linear_to_srgb, palette_albedo, palette_emissive, palette_roughness, palette_transmission, sky_rgb, srgb_to_linear};
import package::ddgi::bake::{ddgi_bake_one};
import package::ddgi::collect::{ddgi_collect_one};
import package::ddgi::consts::{DDGI_AGE_MAX, DDGI_AMBIENT_FLOOR, DDGI_BAKE_STAGGER, DDGI_BIAS_CELLS, DDGI_BIAS_MAX_VOXELS, DDGI_CASCADE_FADE_CELLS, DDGI_CAST_FLOOR, DDGI_CAST_SUN, DDGI_COLLECT_THREADS, DDGI_CONF_HI, DDGI_CONF_LO, DDGI_DEPTH_SWEEP, DDGI_DEPTH_TEXELS, DDGI_EMIT_GAIN, DDGI_LOD0_CELLS_PER_BRICK, DDGI_LOD0_SLOTS_PER_BRICK, DDGI_LOD_COUNT, DDGI_META_ACTIVE, DDGI_META_ENABLED, DDGI_META_NBR_VALID, DDGI_OCCL_END_BIAS, DDGI_PI, DDGI_RAY_BIAS, DDGI_RAY_BUDGET, DDGI_REFRESH_PERIOD, DDGI_SKIP_AGE, DDGI_SKY_AMBIENT, DDGI_SKY_RADIANCE_SCALE, DDGI_T_MAX, DDGI_USAGE_SCAN_MASK, DDGI_WL_IDX_MASK, DDGI_WL_LOD_SHIFT, SHADOW_SURFACE_EPS, ddgi_lod_param_idx};
import package::ddgi::helpers::{DDGI_REC_ENABLED, DDGI_REC_OCCUPIED, ddgi_air_cross, ddgi_cavity_narrow, ddgi_cell_in_window, ddgi_lod0_bricks_per_chunk, ddgi_lod0_claim_seen, ddgi_lod0_next, ddgi_lod0_seg_of, ddgi_lod0_slot_brick_off, ddgi_lod0_tile_key, ddgi_lod_count, ddgi_lod_slot_base, ddgi_meta_age, ddgi_meta_nbr, ddgi_meta_pack, ddgi_meta_with_nbr, ddgi_oct_decode, ddgi_rand2, ddgi_ray_rand, ddgi_rec_off, ddgi_slot, ddgi_slot_lod_of, ddgi_slot_world_cell};
import package::ddgi::sample::{ddgi_box_edge_dist, ddgi_dbg_cov, ddgi_dbg_dom, ddgi_dbg_edge, ddgi_dbg_fb, ddgi_dbg_occl_out, ddgi_dbg_rej, ddgi_dbg_rej_out, ddgi_dbg_wsum, ddgi_dbg_zero, ddgi_lod_contains, ddgi_sample, ddgi_sample_lod};
import package::ddgi::sort::{ddgi_box_hits_object, ddgi_nbr_vis_mask, ddgi_neighbor_occupied};
import package::lightfield::{light_field_ao};
import package::trace::{make_grid};
import package::world::{WorldHit, voxel_normal, world_cfg_full, world_raycast, world_solid, world_trace_medium};

// 全屏逐像素层次栈式 mask DDA（64-tree）：chunk 间 256³ A&W，chunk 内 4 层栈帧（256→64→16→4→1）。
// 常量/寻址/slot 打包须与 Rust `brickmap::dda::wgsl_consts`、`view.rs::get_voxel`、`wire.rs::encode_slot` 一致。

// beam 预 pass：全分辨率 / BEAM_DIV 的低分辨率保守 trace，输出每像素最近命中 t。
// 只行进到 min(t_cap, 0.35/(BEAM_DIV·px_ang))；miss 存该上限（非 t_cap）。
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

/// 玻璃折射率（常见玻璃 1.5）。
const GLASS_IOR: f32 = 1.5;
/// 垂直入射的 Fresnel 反射率 F0。
const GLASS_F0: f32 = 0.04;
/// 折射/反射后新射线起点沿新方向外推的距离（voxel），须大于浮点噪声且远小于 1 格。
const GLASS_EPS: f32 = 1e-3;
/// 一条主射线最多处理的玻璃界面数（入射/出射各算一次），不限介质段厚度。
const GLASS_MAX_BOUNCE: u32 = 8u;
/// Fresnel 反射率门槛：低于此值不追反射射线。
const GLASS_REFL_MIN: f32 = 0.02;
/// 反射命中的近似辐亮度：反射射线不追阴影、不采样 GI，用「阳光直照 × N·L + 常量天光」近似。
const GLASS_REFL_AMBIENT: f32 = 0.15;

/// 太阳方向的直光透过率（1 = 无遮挡，0 = 被不透明体素全挡），一次穿透 trace 走完整段介质。
/// `self_voxel`/`self_obj`：起点贴命中面外侧，豁免 t≈0 自命中。
fn sun_transmittance(origin: vec3<f32>, sun_dir: vec3<f32>, self_voxel: vec3<i32>, self_obj: i32) -> f32 {
  let sh = world_raycast(make_grid(0u), origin, sun_dir, 8192.0, world_cfg_full(0.0, 3u));
  if (!sh.hit) { return 1.0; }
  if (all(sh.voxel == self_voxel) && sh.obj_id == self_obj) { return 1.0; }
  if (palette_transmission(sh.palette_base, sh.pal) <= 0.0) { return 0.0; }
  var tint = vec3<f32>(1.0);
  let h = world_trace_medium(make_grid(0u), origin, sun_dir, 8192.0, false, &tint);
  let through = (tint.r + tint.g + tint.b) / 3.0;
  if (!h.hit) { return through; }
  if (all(h.voxel == self_voxel) && h.obj_id == self_obj) { return 1.0; }
  if (palette_transmission(h.palette_base, h.pal) <= 0.0) { return 0.0; }
  return through;
}

/// 状态机产物：最终命中的（不透明）表面 / 沿途累积的透射色 / 累积的反射项 / 当前射线方向。
struct GlassOut {
  ray: WorldHit,
  tint: vec3<f32>,
  refl: vec3<f32>,
  dir: vec3<f32>,
}

/// 玻璃状态机：每轮处理一个界面事件（入射面 Fresnel 分反射/折射，出射面折射回空气或全内反射，
/// 命中不透明实体则返回由调用方着色）；`t_min` 是 beam 预 pass 的保守起点，只对第一段有效。
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
      best = world_trace_medium(make_grid(0u), org, dir, max_t, true, &seg);
    } else {
      best = world_raycast(make_grid(0u), org, dir, max_t, world_cfg_full(t_start, 3u));
    }
    tint = tint * seg;
    if (!best.hit) { return GlassOut(best, tint, refl, dir); }
    let tau = palette_transmission(best.palette_base, best.pal);
    if (tau <= 0.0) { return GlassOut(best, tint, refl, dir); }
    let p_hit = org + dir * best.t;
    let nn = best.n;
    let ci = clamp(dot(-dir, nn), 0.0, 1.0);
    if (!in_medium) {
      let fres = GLASS_F0 + (1.0 - GLASS_F0) * pow(1.0 - ci, 5.0);
      if (fres > GLASS_REFL_MIN) {
        let rdir = reflect(dir, nn);
        let rhit = world_raycast(make_grid(0u), p_hit + nn * GLASS_EPS, rdir,
                                 max_t, world_cfg_full(0.0, 3u));
        var rc = sky_rgb();
        if (rhit.hit) {
          rc = palette_albedo(rhit.palette_base, rhit.pal)
            * (sun_c * max(dot(rhit.n, sun_dir), 0.0) + sky * GLASS_REFL_AMBIENT);
        }
        refl = refl + tint * fres * rc;
      }
      let eta = 1.0 / GLASS_IOR;
      let k = 1.0 - eta * eta * (1.0 - ci * ci);
      dir = normalize(eta * dir + (eta * ci - sqrt(max(k, 0.0))) * nn);
      in_medium = true;
    } else {
      let eta = GLASS_IOR;
      let k = 1.0 - eta * eta * (1.0 - ci * ci);
      if (k < 0.0) {
        dir = reflect(dir, nn);
      } else {
        dir = normalize(eta * dir + (eta * ci - sqrt(k)) * nn);
        in_medium = false;
      }
    }
    org = p_hit + dir * GLASS_EPS;
    t_start = 0.0;
  }
  return GlassOut(WorldHit(false, 0.0, vec3<f32>(0.0), 0u, vec3<f32>(0.0), 0u, vec3<i32>(0), 0, 0u),
                  tint, refl, dir);
}

/// 自动曝光 + sRGB 编码输出（天空 / 不透明 / 玻璃三条路径共用）。
/// 曝光系数必须在 sRGB 编码之前乘。
fn write_output(coord: vec2<i32>, col: vec3<f32>) {
  let exposure = bitcast<f32>(eye_adapt_ro[0u]);
  let exposed = col * select(1.0, exposure, exposure > 0.0);
  textureStore(out_tex, coord, vec4<f32>(linear_to_srgb(exposed), 1.0));
}

// DDA 主入口：命中 → 直光着色（硬阴影 + sky 环境 + emissive），miss → 纯色天空
@compute @workgroup_size(8, 8, 1)
fn dda_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let size = textureDimensions(out_tex);
  if (gid.x >= size.x || gid.y >= size.y) { return; }
  let coord0 = vec2<i32>(i32(gid.x), i32(gid.y));

  // 「屏幕使用」记账开关（本帧轮到的列带 ∪ 行带）：同时作为「是着色路径」标志传给 `ddgi_sample`
  // —— 认领细级探针、以及探针的「屏幕使用」粘滞记账都只在这条路径上做。
  let usage_frame = u32(ddgi_u.params.x);
  let track = select(
    0u,
    1u,
    ((gid.x + usage_frame) & DDGI_USAGE_SCAN_MASK) == 0u
      || ((gid.y + usage_frame) & DDGI_USAGE_SCAN_MASK) == 0u,
  );

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

  // lod.z > 0.5（`GATE_NO_BEAM=1`）关闭 beam，t_min = 0。
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
                            world_cfg_full(t_min, 3u));
  var best = first;
  var dir = dir_voxel;
  var through = vec3<f32>(1.0);
  var refl = vec3<f32>(0.0);
  if (first.hit && palette_transmission(first.palette_base, first.pal) > 0.0) {
    let g = trace_glass(origin_voxel, dir_voxel, t_min, frustum_length, sun_dir, sky, sun_c);
    best = g.ray;
    dir = g.dir;
    through = g.tint;
    refl = g.refl;
  }
  var col = sky_rgb() * through + refl;
  if (best.hit) {
    col = vec3<f32>(0.0);
    let alb = palette_albedo(best.palette_base, best.pal);
    let gg = make_grid(u32(best.obj_id) + 1u);
    let vc = vec3<f32>(best.voxel) + vec3<f32>(0.5);
    let p_voxel = gg.pos + vec3<f32>(dot(vc, gg.col0), dot(vc, gg.col1), dot(vc, gg.col2)) * gg.scale;
    let n = voxel_normal(gg, best.voxel, best.n);
    // 沿 n 外推 0.5/max|n| + SHADOW_SURFACE_EPS，使起点离开命中体素。
    let n_off = 0.5 / max(max(abs(n.x), abs(n.y)), max(abs(n.z), 1e-3)) + SHADOW_SURFACE_EPS;
    let ndl = max(dot(n, sun_dir), 0.0);
    var sun = 0.0;
    if (ndl > 0.0) {
      sun = sun_transmittance(p_voxel + n * n_off, sun_dir, best.voxel, best.obj_id);
    }
    let gi_on = ddgi_u.misc.x > 0.5;
    var gi = vec3<f32>(0.0);
    var cov = 0.0;
    // 半分辨率 GI 开关 = `ddgi_u.flags.y`；诊断档（`params.y > 0.5`）强制走逐像素内联路径。
    let gi_half = gi_on && ddgi_u.flags.y > 0.5 && ddgi_u.params.y < 0.5;
    if (gi_half) {
      // uv 必须按全分辨率尺寸（`size`）归一化，不能除以 GI 纹理自身尺寸。
      let guv = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5))
              / vec2<f32>(f32(size.x), f32(size.y));
      let s = textureSampleLevel(gi_tex, light_samp, guv, 0.0);
      let sc = textureSampleLevel(gi_cov_tex, light_samp, guv, 0.0);
      // premultiplied-valid 归一化：valid = 0（天空/介质）像素不参与插值。
      let gw = max(s.a, 1e-4);
      gi = s.rgb / gw;
      cov = sc.r / gw;
    } else if (gi_on) {
      gi = ddgi_sample(p_voxel + n * n_off, n, track, true) * ddgi_u.params.z / DDGI_PI;
      cov = ddgi_dbg_cov;
    }
    // AO 只乘到常量天光项（gi 已自带逐方向遮挡）。
    let ao = light_field_ao(p_voxel, best.obj_id);
    // 天光常量兜底权重 conf 用覆盖度 cov（非 wsum），只作用于天光项、不缩放 gi。
    let conf = smoothstep(DDGI_CONF_LO, DDGI_CONF_HI, cov);
    let amb_sky = sky * DDGI_SKY_AMBIENT;
    let outer_lod = DDGI_LOD_COUNT - 1u;
    let w_edge = max(f32(ddgi_u.lods[outer_lod].origin.w) * DDGI_CASCADE_FADE_CELLS, 1e-3);
    let t_edge = smoothstep(0.0, 1.0,
      clamp(ddgi_box_edge_dist(outer_lod, p_voxel) / w_edge, 0.0, 1.0));
    let amb_far = sky * ddgi_u.misc.w;
    // mix 实参顺序必须是 `mix(1.0, DDGI_AMBIENT_FLOOR, conf)`（conf=1 → FLOOR）。
    let amb = mix(amb_far, amb_sky, t_edge) * mix(1.0, DDGI_AMBIENT_FLOOR, conf);
    col = alb * (sun_c * ndl * sun + amb * ao + gi);
    // 太阳高光（Blinn-Phong 半程向量）：粗糙度 0 → 指数 256，1 → 4。
    {
      let rough = palette_roughness(best.palette_base, best.pal);
      let to_cam = -dir;
      let half_sum = sun_dir + to_cam;
      let half_vec = half_sum / max(length(half_sum), 1e-5);
      let spec_pow = mix(256.0, 4.0, rough);
      let spec = pow(max(dot(n, half_vec), 0.0), spec_pow) * sun * ndl;
      col = col + sun_c * alb * spec * mix(1.0, 0.15, rough);
    }
    col = col + alb * palette_emissive(best.palette_base, best.pal) * DDGI_EMIT_GAIN;
    if (ddgi_u.params.y > 0.5) {
      if (ddgi_u.params.y < 1.5) {
        // 1：GI 档 = 纯 gi ×8。
        col = alb * gi * 8.0;
      } else if (ddgi_u.params.y < 2.5) {
        // 2：wsum 档 = trilinear 权重和 ∈ [0,1] 的灰度（不吃 albedo）。
        col = vec3<f32>(clamp(ddgi_dbg_wsum, 0.0, 1.0));
      } else if (ddgi_u.params.y < 3.5) {
        // 3：包含该像素的壳所在 LOD + 1（d = 0 = 未被任何 LOD 盒包含）。
        let d = ddgi_dbg_dom;
        col = vec3<f32>(0.1, 0.8, 0.2);                          // 绿：未被任何 LOD 包含
        col = select(col, vec3<f32>(0.9, 0.2, 0.2), d > 0.5);    // 红：LOD0
        col = select(col, vec3<f32>(0.95, 0.85, 0.1), d > 1.5);  // 黄：LOD1
        col = select(col, vec3<f32>(0.2, 0.4, 0.9), d > 2.5);    // 蓝：LOD2
        col = select(col, vec3<f32>(0.9, 0.2, 0.9), d > 3.5);    // 品红：LOD3
        col = select(col, vec3<f32>(0.1, 0.9, 0.9), d > 4.5);    // 青：LOD4
      } else {
        // ≥4：Probe 档 = 8 角采样失败原因染色（颜色 ↔ ddgi_dbg_fb 阈值）。
        // fb > 4.5 青绿 / > 3.5 紫 / > 2.5 亮黄 = 前几级壳兜底命中；绿 = wsum > 0、白 = wsum > 0 但辐照度 ≈ 0。
        // 灰 = 未跑 sample_lod 或 8 角被窗口剔；青 = 无数据闸门；橙 = ENABLED；品红 = wn ≤ 0；红 = wd ≤ 0；蓝 = 线段求交遮挡。
        if (ddgi_dbg_fb > 4.5) {
          col = vec3<f32>(0.0, 0.6, 0.55);
        } else if (ddgi_dbg_fb > 3.5) {
          col = vec3<f32>(0.65, 0.35, 0.95);
        } else if (ddgi_dbg_fb > 2.5) {
          col = vec3<f32>(0.98, 0.98, 0.35);
        } else if (ddgi_dbg_wsum >= 1e-4) {
          col = select(
            vec3<f32>(0.15, 0.85, 0.25),
            vec3<f32>(0.95, 0.95, 0.95),
            ddgi_dbg_zero > 0.5,
          );
        } else {
          let r_np = f32(ddgi_dbg_rej_out.x);
          let r_wn = f32(ddgi_dbg_rej_out.y);
          let r_wd = f32(ddgi_dbg_rej_out.z);
          let r_ia = f32(ddgi_dbg_rej_out.w);
          let r_oc = f32(ddgi_dbg_occl_out);
          let rmax = max(r_oc, max(r_ia, max(r_wd, max(r_wn, r_np))));
          if (rmax <= 0.0) {
            col = vec3<f32>(0.5, 0.5, 0.5);
          } else if (r_oc >= rmax) {
            col = vec3<f32>(0.2, 0.35, 0.95);
          } else if (r_wd >= rmax) {
            col = vec3<f32>(0.9, 0.15, 0.15);
          } else if (r_wn >= rmax) {
            col = vec3<f32>(0.95, 0.15, 0.7);
          } else if (r_ia >= rmax) {
            col = vec3<f32>(0.1, 0.85, 0.95);
          } else {
            col = vec3<f32>(0.95, 0.6, 0.1);
          }
        }
      }
    }
    col = col * through + refl;
  }
  write_output(coord0, col);
}

// 半分辨率 GI（菜单开关 `ddgi_u.flags.y`）：GI 采样在 1/2 分辨率的独立 pass 完成，主 pass 双线性取用。
// 输出：gi_out.rgb = gi·valid、gi_out.a = valid；gi_cov_out.r = cov·valid、.g = valid；
// 主 pass 按 valid 归一化，天空/介质像素（valid = 0）不参与插值。
@compute @workgroup_size(8, 8, 1)
fn gi_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let size = textureDimensions(gi_out);
  if (gid.x >= size.x || gid.y >= size.y) { return; }
  let coord = vec2<i32>(i32(gid.x), i32(gid.y));

  // 「屏幕使用」记账开关：与 `dda_main` 同一套（本 pass 像素代表 2×2 全分辨率像素）。
  let usage_frame = u32(ddgi_u.params.x);
  let track = select(
    0u,
    1u,
    ((gid.x + usage_frame) & DDGI_USAGE_SCAN_MASK) == 0u
      || ((gid.y + usage_frame) & DDGI_USAGE_SCAN_MASK) == 0u,
  );

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

  // beam 纹理 = 本 pass 分辨率的 2×：beam 下标除 (BEAM_DIV/2)。
  var t_min = 0.0;
  if (view_u.lod.z < 0.5) {
    let beam_size = textureDimensions(beam_depth);
    let bx = gid.x / (BEAM_DIV / 2u);
    let by = gid.y / (BEAM_DIV / 2u);
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
  var gi = vec3<f32>(0.0);
  var cov = 0.0;
  var valid = 0.0;
  // 只处理「首命中即不透明」；着色点/法线/采样点与 `dda_main` 的不透明分支一致。
  if (h.hit && palette_transmission(h.palette_base, h.pal) <= 0.0) {
    let gg = make_grid(u32(h.obj_id) + 1u);
    let vc = vec3<f32>(h.voxel) + vec3<f32>(0.5);
    let p_voxel = gg.pos + vec3<f32>(dot(vc, gg.col0), dot(vc, gg.col1), dot(vc, gg.col2)) * gg.scale;
    let n = voxel_normal(gg, h.voxel, h.n);
    let n_off = 0.5 / max(max(abs(n.x), abs(n.y)), max(abs(n.z), 1e-3)) + SHADOW_SURFACE_EPS;
    gi = ddgi_sample(p_voxel + n * n_off, n, track, true) * ddgi_u.params.z / DDGI_PI;
    cov = ddgi_dbg_cov;
    valid = 1.0;
  }
  textureStore(gi_out, coord, vec4<f32>(gi * valid, valid));
  // storage texture 的 store 值恒为 vec4（rg32f 也不例外，多出的通道被丢弃）
  textureStore(gi_cov_out, coord, vec4<f32>(cov * valid, valid, 0.0, 0.0));
}

// 阶段一 seal：独立 pass（pass 边界保证计数器可见）。每 LOD 一个线程：读活跃探针数
// → 算 rpp（预算/活跃数，clamp 1..256）→ 写 cast/collect indirect args。
@compute @workgroup_size(5, 1, 1)
fn ddgi_seal(@builtin(local_invocation_id) lid: vec3<u32>) {
  let lod = lid.x;
  var total_active = 0u;
  for (var l = 0u; l < DDGI_LOD_COUNT; l = l + 1u) {
    total_active = total_active + atomicLoad(&ddgi_indirect[DDGI_INDIR_COUNT_BASE + l]);
  }
  let rpp = max(1u, min(256u, DDGI_RAY_BUDGET / max(total_active, 1u)));

  var ray_base = 0u;
  var coll_base = 0u;
  var total_ray = 0u;
  var total_coll = 0u;
  for (var l = 0u; l < DDGI_LOD_COUNT; l = l + 1u) {
    let cl = atomicLoad(&ddgi_indirect[DDGI_INDIR_COUNT_BASE + l]);
    let rc = select(0u, cl * rpp, cl > 0u);
    let cc = select(0u, cl * DDGI_COLLECT_THREADS, cl > 0u);
    if (l < lod) { ray_base = ray_base + rc; coll_base = coll_base + cc; }
    total_ray = total_ray + rc;
    total_coll = total_coll + cc;
  }
  atomicStore(&ddgi_indirect[DDGI_INDIR_RAYBASE_BASE + lod], ray_base);
  atomicStore(&ddgi_indirect[DDGI_INDIR_COLLBASE_BASE + lod], coll_base);
  atomicStore(&ddgi_indirect[DDGI_INDIR_RPP_BASE + lod], rpp);
  if (lod == 0u) {
    atomicStore(&ddgi_indirect[DDGI_INDIR_RAY_TOTAL], total_ray);
    atomicStore(&ddgi_indirect[DDGI_INDIR_COLL_TOTAL], total_coll);
    let cb = DDGI_INDIR_CAST_BASE;
    atomicStore(&ddgi_args[cb + 0u], min((total_ray + 63u) / 64u, 65535u));
    atomicStore(&ddgi_args[cb + 1u], 1u);
    atomicStore(&ddgi_args[cb + 2u], 1u);
    let kb = DDGI_INDIR_COLL_BASE;
    atomicStore(&ddgi_args[kb + 0u], min((total_coll + 63u) / 64u, 65535u));
    atomicStore(&ddgi_args[kb + 1u], 1u);
    atomicStore(&ddgi_args[kb + 2u], 1u);
  }
}

// 阶段一 烘焙：逐 cell 沿 4³ 分裂树找最大全空块（16³ → 8³ → 4³ → 体素粒度 `DDGI_PROBE_MIN_BLOCK`）、探针放其中心（同级取更靠近 cell 中心的块）；
// 全空 cell → 居中，全满 cell → 无探针。输出 ddgi_cell[slot] = flags(bit0 探针存在 / bit1 cell 有体素) | cell 内归一化偏移。
// 增量：ddgi_cell_id[slot] 记录已烘焙的世界 cell 键，键未变的槽位跳过。

// 5 个 per-LOD 入口：按 lod 0→4（细→粗）各派发一个独立 pass，pass 边界即内存屏障。
@compute @workgroup_size(64)
fn ddgi_bake0(@builtin(global_invocation_id) gid: vec3<u32>) { ddgi_bake_one(0u, gid.x); }
@compute @workgroup_size(64)
fn ddgi_bake1(@builtin(global_invocation_id) gid: vec3<u32>) { ddgi_bake_one(1u, gid.x); }
@compute @workgroup_size(64)
fn ddgi_bake2(@builtin(global_invocation_id) gid: vec3<u32>) { ddgi_bake_one(2u, gid.x); }
@compute @workgroup_size(64)
fn ddgi_bake3(@builtin(global_invocation_id) gid: vec3<u32>) { ddgi_bake_one(3u, gid.x); }
@compute @workgroup_size(64)
fn ddgi_bake4(@builtin(global_invocation_id) gid: vec3<u32>) { ddgi_bake_one(4u, gid.x); }

@compute @workgroup_size(64)
fn ddgi_sort(@builtin(global_invocation_id) gid: vec3<u32>) {
  let slot = gid.x;
  if (slot >= u32(ddgi_u.misc.y)) { return; }
  let lod = ddgi_slot_lod_of(slot);
  let idx = slot - ddgi_lod_slot_base(lod);
  // LOD0 未领段的槽位：反向表是空的（`g` 无意义，`ddgi_cell` 里可能还留着别的世界的记录）→ 跳过。
  if (lod == 0u && idx >= ddgi_lod0_next()) { return; }
  let rec = ddgi_cell[slot];
  let cs = ddgi_u.lods[lod].origin.w;
  let wc = ddgi_slot_world_cell(lod, idx);
  let cmin = wc * cs;

  let enabled = (rec & DDGI_REC_ENABLED) != 0u;
  if (!enabled) {
    ddgi_meta[slot] = 0u;
    ddgi_slot_pos[slot] = vec4<f32>(0.0, 0.0, 0.0, -1.0);
    return;
  }
  let off = vec3<f32>(ddgi_rec_off(rec)) / 255.0;
  let probe_pos = vec3<f32>(cmin) + off * f32(cs);

  var near = (rec & DDGI_REC_OCCUPIED) != 0u;
  if (!near) {
    near = ddgi_neighbor_occupied(lod, wc + vec3<i32>(-1, 0, 0))
      || ddgi_neighbor_occupied(lod, wc + vec3<i32>(1, 0, 0))
      || ddgi_neighbor_occupied(lod, wc + vec3<i32>(0, -1, 0))
      || ddgi_neighbor_occupied(lod, wc + vec3<i32>(0, 1, 0))
      || ddgi_neighbor_occupied(lod, wc + vec3<i32>(0, 0, -1))
      || ddgi_neighbor_occupied(lod, wc + vec3<i32>(0, 0, 1));
  }
  if (!near) {
    // 物体：主世界 brick tree 里没有它的体素，必须用 grid_descs 的世界 AABB 判。
    near = ddgi_box_hits_object(
      vec3<f32>(cmin),
      vec3<f32>(cmin) + vec3<f32>(f32(cs)),
    );
  }

  // age 继承：同 slot 且 world cell 未变 → 续龄（换 cell 时 bake 已清零）。
  let prev_meta = ddgi_meta[slot];
  var age = ddgi_meta_age(prev_meta);
  // 世界编辑（脏区）命中本 cell → 把年龄压回 1（图集数据来自编辑前的世界）。
  let cell_lo = vec3<f32>(cmin);
  let cell_hi = cell_lo + f32(cs);
  if (ddgi_u.dirty_min.w > 0.5
    && all(cell_hi > ddgi_u.dirty_min.xyz)
    && all(cell_lo < ddgi_u.dirty_max.xyz)) {
    // 按槽位号错峰：age = 1 + slot % DDGI_BAKE_STAGGER。
    age = 1u + (slot % DDGI_BAKE_STAGGER);
  }
  // 判活条件：有探针 && (本 cell 或 6 邻接有体素 || 与物体 AABB 相交)。
  let is_active = near;

  if (is_active) {
    // 本帧投线条件：used（上帧被采样）|| !converged（年龄不足）|| due（错峰强制刷新）。
    let usage = ddgi_cell_slot[slot];
    ddgi_cell_slot[slot] = select(0u, usage - 1u, usage > 0u);
    let used = usage > 0u;
    let frame = u32(ddgi_u.params.x);
    // 收敛 = 已把每个深度纹素都投过一遍（`age × rpp ≥ DDGI_DEPTH_SWEEP`）且至少投过 SKIP_AGE 次。
    // 少了前半条：rpp 小的时候探针会在只扫了半个八面体时停投，图集里那半边的方向既无深度记录
    // 又无辐照度覆盖度，采样侧的数据闸门会把它们全剔掉（表现为整片 GI 变黑）。
    // `rpp` 由 `seal` 写、本 pass 在其之前跑，所以读到的上一帧的值（rpp 逐帧变化极慢）。
    let rpp_prev = max(atomicLoad(&ddgi_indirect[DDGI_INDIR_RPP_BASE + lod]), 1u);
    let swept = age * rpp_prev >= DDGI_DEPTH_SWEEP;
    let converged = swept && age >= DDGI_SKIP_AGE[ddgi_lod_param_idx(lod)];
    let due = (frame + slot) % DDGI_REFRESH_PERIOD[ddgi_lod_param_idx(lod)] == 0u;
    if (used || !converged || due) {
      age = min(age + 1u, DDGI_AGE_MAX);
      let wslot = atomicAdd(&ddgi_indirect[DDGI_INDIR_COUNT_BASE + lod], 1u);
      // packed：低 8 位 age、中 19 位 cell 下标、bit27 起 lod（collect 用 cell 下标定位图集纹素）。
      let packed = age | (idx << 8u) | (lod << DDGI_WL_LOD_SHIFT);
      ddgi_worklist[ddgi_lod_slot_base(lod) + wslot] =
        vec4<f32>(probe_pos, bitcast<f32>(packed));
    }
  }

  // 邻接可见性掩码（空间混合门控）：只对活跃探针算；NBR_VALID = 0 消费侧按「被挡」处理。
  // 仅当本级可能变化时重算：换主/重建（bake 已清 NBR_VALID）或世界编辑（脏区 AABB 沿各轴外扩 cs）。
  var meta_word = ddgi_meta_pack(age, true, is_active);
  if (is_active) {
    var mask = ddgi_meta_nbr(prev_meta);
    let pad = vec3<f32>(f32(cs));
    let dirty_here = ddgi_u.dirty_min.w > 0.5
      && all(cell_hi + pad > ddgi_u.dirty_min.xyz)
      && all(cell_lo - pad < ddgi_u.dirty_max.xyz);
    if ((prev_meta & DDGI_META_NBR_VALID) == 0u || dirty_here) {
      mask = ddgi_nbr_vis_mask(probe_pos, vec3<f32>(cmin), f32(cs));
    }
    meta_word = ddgi_meta_with_nbr(meta_word, mask);
  }
  ddgi_meta[slot] = meta_word;
  ddgi_slot_pos[slot] = vec4<f32>(probe_pos, f32(lod));
}

// 阶段二 cast：worklist 活跃探针各投 rpp 条射线，方向 = 八面体深度纹素方向 + 纹素内抖动
// （纹素按 frame 轮换）；命中 → 出射辐亮度，miss → 天空色。输出 ddgi_samples（方向+命中距离 / 辐亮度）。
// 线程映射：全局 tid = ray_base[lod] + probe_idx × rpp + ray，一次 dispatch 覆盖全部 LOD。
@compute @workgroup_size(64)
fn ddgi_cast(@builtin(global_invocation_id) gid: vec3<u32>) {
  let tid = gid.x;
  if (tid >= atomicLoad(&ddgi_indirect[DDGI_INDIR_RAY_TOTAL])) { return; }
  var lod = 0u;
  for (var l = 1u; l < DDGI_LOD_COUNT; l = l + 1u) {
    if (tid >= atomicLoad(&ddgi_indirect[DDGI_INDIR_RAYBASE_BASE + l])) { lod = l; }
  }
  let rpp = atomicLoad(&ddgi_indirect[DDGI_INDIR_RPP_BASE + lod]);
  if (rpp == 0u) { return; }
  let local = tid - atomicLoad(&ddgi_indirect[DDGI_INDIR_RAYBASE_BASE + lod]);
  let probe_idx = local / rpp;
  let ray = local % rpp;
  let slot = ddgi_lod_slot_base(lod) + probe_idx;
  let probe_pos = ddgi_worklist[slot].xyz;
  let frame = u32(ddgi_u.params.x);
  // RNG 种子用探针的 cell 下标；时间维随机性由 frame 提供。
  let cell_idx = (bitcast<u32>(ddgi_worklist[slot].w) >> 8u) & DDGI_WL_IDX_MASK;

  // 射线↔深度纹素绑定：每条射线固定归属一个八面体纹素，纹素按 (frame·rpp + ray) 轮换（⌈N²/rpp⌉ 帧轮完一圈）。
  // 方向 = 纹素中心 + 纹素内抖动；同纹素跨帧同方向，Chebyshev 的 mean/std 才有效。
  let ndim = DDGI_DEPTH_TEXELS;
  let ntex = ndim * ndim;
  let tsel = (frame * rpp + ray) % ntex;
  let ttx = tsel % ndim;
  let tty = tsel / ndim;
  let jit = ddgi_rand2(ddgi_ray_rand(ddgi_lod_slot_base(lod) + cell_idx, frame, ray));
  let te = ((vec2<f32>(f32(ttx), f32(tty)) + jit) / f32(ndim)) * 2.0 - vec2<f32>(1.0);
  let dir = ddgi_oct_decode(te);
  let sh = world_raycast(make_grid(0u), probe_pos + dir * DDGI_RAY_BIAS, dir, DDGI_T_MAX, world_cfg_full(0.0, 3u));
  // miss → 天空色 × DDGI_SKY_RADIANCE_SCALE。
  var radiance = sky_rgb() * DDGI_SKY_RADIANCE_SCALE;
  var dist = DDGI_T_MAX;
  if (sh.hit) {
    dist = sh.t;
    let alb = palette_albedo(sh.palette_base, sh.pal);
    let hit_p = probe_pos + dir * dist;
    // 逐体素法线必须与着色侧一致（决定反弹辐照度的方向）。
    let n = voxel_normal(
      make_grid(u32(sh.obj_id) + 1u),
      sh.voxel,
      sh.n,
    );
    // 入射辐照度 E → 出射辐亮度 L = albedo·E/π（与着色侧同约定）。
    let gi = ddgi_sample(hit_p + n * 0.5, n, 0u, false);
    // 命中面的**直射太阳**（一次阴影射线）：GI 若只靠"射线逃逸到天空"取光，任何看不到天空的表面
    // ——草丛根部（被叶片挡住）、过道内墙、阴影里的地面——都会自洽地维持全黑，跟探针多密无关。
    // 起点外推量必须与着色侧同一套（半个 voxel + `SHADOW_SURFACE_EPS`）：`sh.t` 命中的是体素面，
    // 只退 1 个 eps 会从**体素内部**出发，阴影射线第一步就撞自己 ⇒ 这一项恒为 0（白耗一根射线）。
    // `DDGI_CAST_SUN = 0` 时整段被编译器折掉，连射线都不发。
    let sun_dir = light_u.lights[0].kind_pos_dir.yzw;
    let sun_c = light_u.lights[0].color_intensity.xyz * light_u.lights[0].color_intensity.w;
    let ndl = max(dot(n, sun_dir), 0.0);
    var sun = 0.0;
    if (DDGI_CAST_SUN > 0.0 && ndl > 0.0) {
      let n_off = 0.5 / max(max(abs(n.x), abs(n.y)), max(abs(n.z), 1e-3)) + SHADOW_SURFACE_EPS;
      sun = sun_transmittance(hit_p + n * n_off, sun_dir, sh.voxel, sh.obj_id);
    }
    // 命中发光体素 → 出射辐亮度含自身发射（radiance 直出，不吃 GI/入射）。
    radiance = alb * (gi / DDGI_PI + sky_rgb() * DDGI_CAST_FLOOR + sun_c * ndl * sun * DDGI_CAST_SUN)
      + alb * palette_emissive(sh.palette_base, sh.pal) * DDGI_EMIT_GAIN;
  }
  let si = tid * 2u;
  ddgi_samples[si] = vec4<f32>(dir, dist);
  ddgi_samples[si + 1u] = vec4<f32>(radiance, 1.0);
}

@compute @workgroup_size(64)
fn ddgi_collect(@builtin(global_invocation_id) gid: vec3<u32>) {
  let total_coll = atomicLoad(&ddgi_indirect[DDGI_INDIR_COLL_TOTAL]);
  let stride = 65535u * 64u;
  var tid = gid.x;
  loop {
    if (tid >= total_coll) { break; }
    ddgi_collect_one(tid);
    tid = tid + stride;
  }
}

@compute @workgroup_size(64)
fn probe_viz_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dot_size = u32(view_u.probe_viz_params.y);
  let id = gid.x;
  // x = 世界空间探针总槽数（Rust 每帧写入）
  if (id >= u32(view_u.probe_viz_params.x)) {
    return;
  }
  let lod = ddgi_slot_lod_of(id);
  // `probe_viz_params.z` = 要画的 LOD 下标（0..DDGI_LOD_COUNT-1）；≥ DDGI_LOD_COUNT = 全画。
  let sel = u32(view_u.probe_viz_params.z);
  if (sel < DDGI_LOD_COUNT && lod != sel) {
    return;
  }
  // 只可视化活跃探针（DDGI_META_ACTIVE）。
  if ((ddgi_meta[id] & DDGI_META_ACTIVE) == 0u) {
    return;
  }
  // 黄点画在探针实际位置。
  let pos = ddgi_slot_pos[id];
  let dot_col = vec4<f32>(1.0, 0.85, 0.0, 1.0);
  let clip = view_u.view_proj * vec4<f32>(pos.xyz, 1.0);
  if (clip.w <= 0.0) {
    return;
  }
  let ndc = clip.xyz / clip.w;
  if (ndc.x < -1.0 || ndc.x > 1.0 || ndc.y < -1.0 || ndc.y > 1.0 || ndc.z < 0.0 || ndc.z > 1.0) {
    return;
  }
  let to_probe = pos.xyz - view_u.cam_pos_voxel.xyz;
  let dist = length(to_probe);
  let hit = world_raycast(make_grid(0u), view_u.cam_pos_voxel.xyz, to_probe / dist, dist, world_cfg_full(0.0, 3u));
  if (hit.hit && hit.t < dist - 0.5) {
    return;
  }
  let dims = vec2<f32>(textureDimensions(out_tex));
  let px = vec2<f32>((ndc.x * 0.5 + 0.5) * dims.x, (1.0 - (ndc.y * 0.5 + 0.5)) * dims.y);
  let ci = vec2<i32>(px);
  let half = i32(dot_size) / 2;
  for (var dy = -half; dy <= half; dy++) {
    for (var dx = -half; dx <= half; dx++) {
      let p = vec2<i32>(ci.x + dx, ci.y + dy);
      if (p.x >= 0 && p.y >= 0 && u32(p.x) < u32(dims.x) && u32(p.y) < u32(dims.y)) {
        textureStore(out_tex, vec2<u32>(p), dot_col);
      }
    }
  }
}

// ---- 采样调试（`view_u.probe_dbg.w` = 1）：鼠标下体素「本来会采样的那 8 个 probe」----
// 只画一个体素，整段由单线程完成（`gid.x != 0` 直接退）。
// 颜色：白 = 采样点；绿 = 该角有探针且 probe→采样点 无遮挡（采样会计入）；
//       红 = 该角**被剔掉**（probe→采样点 被实体挡住，或该点本身被相机深度剔除）；
//       蓝 = 该角没有探针（未编址 / 无数据），画在格子名义中心。

/// 世界点 → (像素 x, 像素 y, 可见性)：z > 0.5 = 相机可见；0 ≤ z ≤ 0.5 = 被相机深度剔除；z < 0 = 出屏/背向。
fn probe_dbg_project(q: vec3<f32>, dims: vec2<f32>) -> vec3<f32> {
  let clip = view_u.view_proj * vec4<f32>(q, 1.0);
  if (clip.w <= 0.0) { return vec3<f32>(0.0, 0.0, -1.0); }
  let ndc = clip.xyz / clip.w;
  if (ndc.x < -1.0 || ndc.x > 1.0 || ndc.y < -1.0 || ndc.y > 1.0 || ndc.z < 0.0 || ndc.z > 1.0) {
    return vec3<f32>(0.0, 0.0, -1.0);
  }
  // 相机深度剔除：相机到该点之间被实体挡住（与 `probe_viz_main` 同一判据）。
  let to = q - view_u.cam_pos_voxel.xyz;
  let dist = length(to);
  let hit = world_raycast(
    make_grid(0u), view_u.cam_pos_voxel.xyz, to / max(dist, 1e-4), dist, world_cfg_full(0.0, 3u));
  let blocked = hit.hit && hit.t < dist - 0.5;
  let px = vec2<f32>((ndc.x * 0.5 + 0.5) * dims.x, (1.0 - (ndc.y * 0.5 + 0.5)) * dims.y);
  return vec3<f32>(px, select(1.0, 0.0, blocked));
}

/// 画一个方块点（`half` = 半径像素）。
fn probe_dbg_dot(px: vec2<f32>, half: i32, col: vec3<f32>, dims: vec2<f32>) {
  let ci = vec2<i32>(px);
  for (var dy = -half; dy <= half; dy = dy + 1) {
    for (var dx = -half; dx <= half; dx = dx + 1) {
      let p = vec2<i32>(ci.x + dx, ci.y + dy);
      if (p.x >= 0 && p.y >= 0 && u32(p.x) < u32(dims.x) && u32(p.y) < u32(dims.y)) {
        textureStore(out_tex, vec2<u32>(p), vec4<f32>(col, 1.0));
      }
    }
  }
}

/// probe 点：被相机深度剔除 → 红（仍画出来，便于定位「探针在那里但看不见」）；出屏 → 不画。
fn probe_dbg_mark(q: vec3<f32>, col: vec3<f32>, dims: vec2<f32>) {
  let pr = probe_dbg_project(q, dims);
  if (pr.z < 0.0) { return; }
  probe_dbg_dot(pr.xy, 5, select(vec3<f32>(1.0, 0.15, 0.15), col, pr.z > 0.5), dims);
}

/// `a → b` 的连线：逐采样点做相机深度剔除，被挡住的段不画（线自然断成可见段）。
fn probe_dbg_line(a: vec3<f32>, b: vec3<f32>, col: vec3<f32>, dims: vec2<f32>) {
  let len = distance(a, b);
  if (len <= 1e-4) { return; }
  let steps = u32(clamp(len, 4.0, 48.0));
  for (var i = 0u; i <= steps; i = i + 1u) {
    let q = mix(a, b, f32(i) / f32(steps));
    let pr = probe_dbg_project(q, dims);
    if (pr.z < 0.5) { continue; }
    probe_dbg_dot(pr.xy, 0, col, dims);
  }
}

/// 把 `v` 的低 `bits` 位画成一行点阵：低位在左、实心 = 1、暗灰 = 0。
/// 每位占 7px（块宽 5px + 间隔 2px），容得下截图采样 ±2px 的定位误差。
/// 给「采样调试」当数值示波器用 —— 截图放大即可直接读出各诊断量的二进制。
fn probe_dbg_bits(v: u32, x0: i32, y0: i32, bits: u32, dims: vec2<f32>) {
  for (var i = 0u; i < bits; i = i + 1u) {
    let on = ((v >> i) & 1u) != 0u;
    let col = vec4<f32>(select(vec3<f32>(0.1, 0.1, 0.1), vec3<f32>(1.0, 1.0, 1.0), on), 1.0);
    for (var dy = 0; dy < 9; dy = dy + 1) {
      for (var dx = 0; dx < 5; dx = dx + 1) {
        let px = vec2<i32>(x0 + i32(i) * 7 + dx, y0 + dy);
        if (px.x >= 0 && px.y >= 0 && u32(px.x) < u32(dims.x) && u32(px.y) < u32(dims.y)) {
          textureStore(out_tex, vec2<u32>(px), col);
        }
      }
    }
  }
}

/// 采样调试的两种判定色（绿 = 成功 / 红 = 失败）。
const DBG_GREEN: vec3<f32> = vec3<f32>(0.2, 1.0, 0.2);
const DBG_RED: vec3<f32> = vec3<f32>(1.0, 0.1, 0.1);

@compute @workgroup_size(64)
fn probe_dbg_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x != 0u) { return; }
  let dbg = view_u.probe_dbg;
  if (dbg.w < 0.5) { return; }
  let dims = vec2<f32>(textureDimensions(out_tex));
  let n = view_u.probe_dbg_n.xyz;
  // 采样点 = 体素中心 + 沿法线外推，与 `dda_main` 的 `n_off` 同式。
  let vc = dbg.xyz + vec3<f32>(0.5);
  let n_off = 0.5 / max(max(abs(n.x), abs(n.y)), max(abs(n.z), 1e-3)) + SHADOW_SURFACE_EPS;
  let p = vc + n * n_off;
  // 第一级包含 p 的 LOD：与 `ddgi_sample` 的循环同判据。
  var lod = DDGI_LOD_COUNT;
  for (var l = 0u; l < DDGI_LOD_COUNT; l = l + 1u) {
    if (ddgi_lod_contains(l, p)) { lod = l; break; }
  }
  probe_dbg_mark(p, vec3<f32>(1.0, 1.0, 1.0), dims);
  // ---- 采样点所在 8³ 瓦片的 LOD0 状态 ----
  // 瓦心标记三态：纯黄 = 已领段；灰白 = 试过但被拒（认领位已置却没段：门槛没过 / 容量满）；
  // 纯品红 = 从没试过（还没轮到它）。
  // 瓦内 8 个 slot 各画一点（画在"应该烘的 4³ cell 中心"）：灰 = 没烘过 / 橙 = 烘过没探针 /
  // 青 = 放下探针（ENABLED）/ 绿 = 本帧判活（ACTIVE，会发线）/ 红 = 烘的 cell 与应烘的不是同一个（寻址错位）。
  // 「有探针但一直是青」= 判活失败（`ddgi_sort` 的 near），「青→绿」才说明这一格会真出 GI。
  // 注意：`ddgi_lod0_seg_of` / `ddgi_lod0_tile_key` 收的都是 **4 voxel cell 坐标**。
  let lod0_cs = f32(ddgi_u.lods[0u].origin.w);
  let tile = f32(DDGI_LOD0_CELLS_PER_BRICK) * lod0_cs;
  let cell0 = vec3<i32>(floor(p / lod0_cs));
  let seg_same = ddgi_lod0_seg_of(cell0);
  let claimed = seg_same != DDGI_NO_PROBE;
  let br = vec3<i32>(floor(p / tile));
  let key0 = ddgi_lod0_tile_key(cell0);
  var mcol = vec3<f32>(1.0, 0.0, 1.0);
  if (claimed) {
    mcol = vec3<f32>(1.0, 1.0, 0.0);
  } else if (key0.z != 0u && ddgi_lod0_claim_seen(key0.x, key0.y)) {
    mcol = vec3<f32>(0.85, 0.85, 0.85);
  }
  probe_dbg_mark((vec3<f32>(br) + vec3<f32>(0.5)) * tile, mcol, dims);
  // （瓦内 8 个 slot 的状态点放到函数最后画：角点标记与它同位置，后画的才看得见。）
  // 版本标记：屏幕右下角一条青色小横条（换构建后能一眼确认跑的是新二进制）。
  for (var x = 0u; x < 16u; x = x + 1u) {
    for (var y = 0u; y < 3u; y = y + 1u) {
      textureStore(
        out_tex,
        vec2<u32>(u32(dims.x) - 24u + x, u32(dims.y) - 12u + y),
        vec4<f32>(0.0, 1.0, 1.0, 1.0),
      );
    }
  }
  // ---- 全局认领量：屏幕左下角横条，满宽 = LOD0 预留容量 ----
  {
    let total = max(f32(ddgi_lod_count(0u)), 1.0);
    let w = u32(clamp(f32(ddgi_lod0_next()) / total, 0.0, 1.0) * 500.0);
    for (var x = 0u; x < 504u; x = x + 1u) {
      let col = select(vec3<f32>(0.15, 0.15, 0.15), vec3<f32>(1.0, 0.9, 0.1), x < w);
      for (var y = 0u; y < 6u; y = y + 1u) {
        textureStore(out_tex, vec2<u32>(x + 8u, u32(dims.y) - 8u + y), vec4<f32>(col, 1.0));
      }
    }
  }
  // ---- LOD0 采样自检（白点上方 1 voxel，左右两个点）----
  // 左 = 严格路径 r1（`ddgi_sample_lod(..., 0, false, false, false)`，不放开背向/掠射）；
  // 右 = 完整阶梯 `ddgi_sample`（着色侧真正用的那条）。**只有右侧那点代表"这个像素有没有 GI"** ——
  // 过道里法线与探针方向近乎垂直，r1 本来就会被 `wn ≤ 0` 剔空，左点红是正常的、不是 bug。
  // 绿 = 拿到权重（wsum ≥ 1e-4），红 = 空手。
  let r_strict = ddgi_sample_lod(p, n, 0u, false, false, false, 0u);
  let rej1 = ddgi_dbg_rej;
  let w1 = r_strict.w;
  probe_dbg_mark(p + vec3<f32>(-0.9, 1.0, 0.0), select(DBG_RED, DBG_GREEN, w1 >= 1e-4), dims);
  let gi_full = ddgi_sample(p, n, 0u, false);
  let fb1 = ddgi_dbg_fb;
  let ws1 = ddgi_dbg_wsum;
  let cov1 = ddgi_dbg_cov;
  let dom1 = ddgi_dbg_dom;
  let zero1 = ddgi_dbg_zero;
  probe_dbg_mark(p + vec3<f32>(0.9, 1.0, 0.0), select(DBG_RED, DBG_GREEN, ws1 >= 1e-4), dims);
  // 第三个标记 = **认领几何判据**本身：绿 = 这点被判为"需要细级"（应当认领），红 = 判为开阔（不认领）。
  // 黑区里对着它看就知道是"判据认为不窄"还是"判据通过但认领/烘焙没成"。
  probe_dbg_mark(p + vec3<f32>(0.0, 1.0, -0.9),
    select(DBG_RED, DBG_GREEN, ddgi_cavity_narrow(make_grid(0u), p)), dims);
  // ---- 数值面板（截图放大即可读二进制，低位在左）；行序 = 下列调用顺序 ----
  //   0..1  ddgi_lod0_next（段分配游标）低16 / 高8
  //   2..5  严格路径 r1 被剔的角数 rej.x/y/z/w
  //   6     r1 的 wsum×1000
  //   7     完整阶梯的服务级 fb（0 主级 / 1 盒外 / 3 放宽wn / 4 背向 / 5 同侧场）
  //   8     完整阶梯 wsum×1000    9 覆盖度×255    10 命中级+1    11 zero 标志
  //   12..14 br.xyz（采样点所在瓦片）  15 seg_same
  //   16 key0.x（chunk 线性）  17 key0.y（chunk 内瓦片线性）
  //   18 g/npc（反向表里的 chunk 线性）  19 g%npc 低16   20..21 ddgi_cell_id[seg].xz
  {
    let slot0 = ddgi_lod_slot_base(0u) + seg_same;
    let id4 = ddgi_cell_id[slot0];
    let g = ddgi_lod0[ddgi_lod0_slot_brick_off() + seg_same];
    let nx = ddgi_lod0_next();
    let x0 = 16;
    let y0 = i32(dims.y) - 244;
    let st = 10;
    probe_dbg_bits(nx & 0xFFFFu, x0, y0, 16u, dims);
    probe_dbg_bits((nx >> 16u) & 0xFFu, x0, y0 + st, 8u, dims);
    probe_dbg_bits(rej1.x, x0, y0 + st * 2, 16u, dims);
    probe_dbg_bits(rej1.y, x0, y0 + st * 3, 16u, dims);
    probe_dbg_bits(rej1.z, x0, y0 + st * 4, 16u, dims);
    probe_dbg_bits(rej1.w, x0, y0 + st * 5, 16u, dims);
    probe_dbg_bits(u32(clamp(w1 * 1000.0, 0.0, 65535.0)), x0, y0 + st * 6, 16u, dims);
    probe_dbg_bits(u32(fb1), x0, y0 + st * 7, 8u, dims);
    probe_dbg_bits(u32(clamp(ws1 * 1000.0, 0.0, 65535.0)), x0, y0 + st * 8, 16u, dims);
    probe_dbg_bits(u32(clamp(cov1 * 255.0, 0.0, 255.0)), x0, y0 + st * 9, 8u, dims);
    probe_dbg_bits(u32(clamp(dom1, 0.0, 255.0)), x0, y0 + st * 10, 8u, dims);
    probe_dbg_bits(select(0u, 1u, zero1 > 0.5), x0, y0 + st * 11, 8u, dims);
    probe_dbg_bits(bitcast<u32>(br.x), x0, y0 + st * 12, 16u, dims);
    probe_dbg_bits(bitcast<u32>(br.y), x0, y0 + st * 13, 16u, dims);
    probe_dbg_bits(bitcast<u32>(br.z), x0, y0 + st * 14, 16u, dims);
    probe_dbg_bits(seg_same & 0xFFFFu, x0, y0 + st * 15, 16u, dims);
    probe_dbg_bits(key0.x, x0, y0 + st * 16, 16u, dims);
    probe_dbg_bits(key0.y, x0, y0 + st * 17, 16u, dims);
    probe_dbg_bits(g / ddgi_lod0_bricks_per_chunk(), x0, y0 + st * 18, 16u, dims);
    probe_dbg_bits((g % ddgi_lod0_bricks_per_chunk()) & 0xFFFFu, x0, y0 + st * 19, 16u, dims);
    probe_dbg_bits(bitcast<u32>(id4.x), x0, y0 + st * 20, 16u, dims);
    probe_dbg_bits(bitcast<u32>(id4.z), x0, y0 + st * 21, 16u, dims);
    // 22 = 完整阶梯拿到的 GI 亮度 ×1000（判断"有没有光"，而不是"有没有权重"）。
    let lum = dot(abs(gi_full), vec3<f32>(0.2126, 0.7152, 0.0722)) * 1000.0;
    probe_dbg_bits(u32(clamp(lum, 0.0, 65535.0)), x0, y0 + st * 22, 16u, dims);
  }
  // ---- 认领门槛现场判定（只读，不改状态）----
  //   0 位图：k=0..3 的 `ddgi_air_cross(floor(p)+step*k)` 结果（bit0 = k=0，即采样点本身）
  //   1 该瓦片认领位（1 = 已被试过）   2 该瓦片段基址低16（65535 = 未认领）   3..5 step+8
  {
    let gw = make_grid(0u);
    let step = vec3<i32>(n);
    let here = vec3<i32>(floor(p));
    var mask = 0u;
    for (var k = 0; k < 4; k = k + 1) {
      if (ddgi_air_cross(gw, here + step * k)) { mask = mask | (1u << u32(k)); }
    }
    let cell0b = vec3<i32>(floor(p / f32(ddgi_u.lods[0u].origin.w)));
    let keyb = ddgi_lod0_tile_key(cell0b);
    let seen = select(0u, 1u, keyb.z != 0u && ddgi_lod0_claim_seen(keyb.x, keyb.y));
    let x1 = 140;
    let y1 = i32(dims.y) - 244;
    let st = 10;
    probe_dbg_bits(mask, x1, y1, 4u, dims);
    probe_dbg_bits(seen, x1, y1 + st, 8u, dims);
    probe_dbg_bits(ddgi_lod0_seg_of(cell0b) & 0xFFFFu, x1, y1 + st * 2, 16u, dims);
    probe_dbg_bits(u32(step.x + 8), x1, y1 + st * 3, 8u, dims);
    probe_dbg_bits(u32(step.y + 8), x1, y1 + st * 4, 8u, dims);
    probe_dbg_bits(u32(step.z + 8), x1, y1 + st * 5, 8u, dims);
  }
  if (lod >= DDGI_LOD_COUNT) { return; }
  let cs = f32(ddgi_u.lods[lod].origin.w);
  // 与 `ddgi_sample_lod` 同一套：先沿法线外推，再以 cell 中心为三线性基。
  var ps = p + n * min(cs * DDGI_BIAS_CELLS, DDGI_BIAS_MAX_VOXELS);
  if (world_solid(make_grid(0u), vec3<i32>(floor(ps)))) { ps = p; }
  let wc0 = vec3<i32>(floor(ps / cs - vec3<f32>(0.5)));
  let g = make_grid(0u);
  let seg_cfg = world_cfg_full(0.0, 3u);
  for (var iz = 0; iz < 2; iz = iz + 1) {
    for (var iy = 0; iy < 2; iy = iy + 1) {
      for (var ix = 0; ix < 2; ix = ix + 1) {
        let wc = wc0 + vec3<i32>(ix, iy, iz);
        let nominal = (vec3<f32>(wc) + vec3<f32>(0.5)) * cs;
        let no_probe = vec3<f32>(0.25, 0.5, 1.0);
        if (!ddgi_cell_in_window(lod, wc)) {
          probe_dbg_mark(nominal, no_probe, dims);
          continue;
        }
        let slot = ddgi_slot(lod, wc);
        if ((ddgi_meta[slot] & DDGI_META_ENABLED) == 0u) {
          probe_dbg_mark(nominal, no_probe, dims);
          continue;
        }
        let probe = ddgi_slot_pos[slot].xyz;
        // 采样侧同一判据：probe → 采样点 的线段被实体挡住 ⇒ 本角在采样时会被剔掉。
        let to = ps - probe;
        let dist = length(to);
        let hit = world_raycast(
          g, probe, to / max(dist, 1e-4), max(dist - DDGI_OCCL_END_BIAS, 0.0), seg_cfg);
        let col = select(vec3<f32>(0.2, 1.0, 0.4), vec3<f32>(1.0, 0.15, 0.15), hit.hit);
        probe_dbg_line(probe, ps, col, dims);
        probe_dbg_mark(probe, col, dims);
      }
    }
  }
  // ---- 瓦内 8 个 LOD0 slot 的状态点（放最后画：与上面的角点标记同位置，后画的才看得见）----
  // 画在"该 slot 按寻址公式应该烘的 4³ cell 中心"：
  // 灰 = 没烘过 / 橙 = 烘过但没放下探针 / 青 = 放下探针（ENABLED）/ 绿 = 本帧判活（ACTIVE，会发线）/
  // 红 = 烘的 cell 与"应该烘的"不是同一个（反向表 / 寻址错位）。
  // 「一直青」= 有探针但判活失败；「青 → 绿」才说明这一格会真出 GI。
  if (claimed) {
    let base = ddgi_lod_slot_base(0u) + seg_same;
    let uc = u32(DDGI_LOD0_CELLS_PER_BRICK);
    for (var k = 0u; k < DDGI_LOD0_SLOTS_PER_BRICK; k = k + 1u) {
      let lc = vec3<i32>(i32(k % uc), i32((k / uc) % uc), i32(k / (uc * uc)));
      let want = br * DDGI_LOD0_CELLS_PER_BRICK + lc;
      let at = (vec3<f32>(want) + vec3<f32>(0.5)) * lod0_cs;
      let id4 = ddgi_cell_id[base + k];
      let rec = ddgi_cell[base + k];
      var col = vec3<f32>(0.5, 0.5, 0.5);
      if (id4.w != 0) {
        col = select(vec3<f32>(1.0, 0.5, 0.0), vec3<f32>(0.2, 1.0, 1.0), (rec & DDGI_REC_ENABLED) != 0u);
        if ((ddgi_meta[base + k] & DDGI_META_ACTIVE) != 0u) { col = DBG_GREEN; }
        if (any(id4.xyz != want)) { col = DBG_RED; }
      }
      probe_dbg_mark(at, col, dims);
    }
  }
}

// 自动曝光：pass1 `eye_adapt_histogram` 抽样 → log2 亮度 → 64 桶直方图；
// pass2 `eye_adapt_update` 取 5%~95% 百分位平均 + 时间平滑 → 写曝光系数（下一帧 `dda_main` 用）。
/// log2 亮度桶覆盖 [-12, +12]：24 档 / 64 桶 = 0.375 档每桶。
const EYE_LUM_MIN_LOG2: f32 = -12.0;
const EYE_LUM_RANGE_LOG2: f32 = 24.0;
const EYE_HIST_BINS: u32 = 64u;
/// 活参数区起始字（= 状态/调试区 8 + 直方图 64），由 Rust `EyeAdaptSettings` 上传。
/// 下标 = `eye_p(i)` 的实参，顺序即语义（改 Rust 结构体必须同步）：0 ev_max 1 ev_min 2 tau_brighten 3 tau_darken 4 key。
const EYE_PARAM_BASE: u32 = 8u + EYE_HIST_BINS;

fn eye_p(i: u32) -> f32 {
  return bitcast<f32>(atomicLoad(&eye_adapt[EYE_PARAM_BASE + i]));
}

/// WG 内共享直方图（原子类型只能位于 workgroup 地址空间）。
var<workgroup> eye_hist: array<atomic<u32>, 64>;

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
    let luma = dot(srgb_to_linear(c), vec3<f32>(0.2126, 0.7152, 0.0722));
    let b = u32(clamp((log2(max(luma, 1e-5)) - EYE_LUM_MIN_LOG2) / bin_w, 0.0, f32(EYE_HIST_BINS - 1u)));
    atomicAdd(&eye_hist[b], 1u);
  }
  workgroupBarrier();
  for (var i = gid.x; i < EYE_HIST_BINS; i = i + 256u) {
    let v = atomicLoad(&eye_hist[i]);
    if (v > 0u) { atomicAdd(&eye_adapt[8u + i], v); }
  }
}

@compute @workgroup_size(1)
fn eye_adapt_update() {
  var total = 0.0;
  for (var i = 0u; i < EYE_HIST_BINS; i = i + 1u) { total = total + f32(atomicLoad(&eye_adapt[8u + i])); }
  if (total > 0.0) {
    let lo = total * 0.05;
    let hi = total * 0.95;
    let bin_w = EYE_LUM_RANGE_LOG2 / f32(EYE_HIST_BINS);
    var cum = 0.0;
    var acc = 0.0;
    var wsum = 0.0;
    for (var i = 0u; i < EYE_HIST_BINS; i = i + 1u) {
      let n = f32(atomicLoad(&eye_adapt[8u + i]));
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
      // 与帧率无关的指数平滑
      let k = select(0.0, 1.0 - exp(-dt / max(tau, 1e-3)), dt > 0.0);
      let ev_new = mix(ev_cur, ev_want, k);
      atomicStore(&eye_adapt[1u], bitcast<u32>(ev_new));
      let manual = bitcast<f32>(atomicLoad(&eye_adapt[2u]));
      let exp_val = exp2(clamp(ev_new + manual, eye_p(1u), eye_p(0u)));
      atomicStore(&eye_adapt[0u], bitcast<u32>(exp_val));
    }
  }
  for (var i = 0u; i < EYE_HIST_BINS; i = i + 1u) { atomicStore(&eye_adapt[8u + i], 0u); }
}
