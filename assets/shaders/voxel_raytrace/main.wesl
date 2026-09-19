import package::bindings::{BEAM_BACKOFF, BEAM_DIV, beam_depth, eye_adapt, eye_adapt_ro, gi_cov_out, gi_cov_tex, gi_dn_dst, gi_dn_tmp, gi_out, gi_tex, gi_u, gi_vis_count, gi_vis_list, light_samp, light_u, out_tex, out_tex_src, view_u};
import package::common::{linear_to_srgb, palette_albedo, palette_emissive, palette_roughness, palette_transmission, sky_rgb, srgb_to_linear};
import package::gi::cache::{GI_CACHE_M_CAP, GI_CACHE_ON, GI_CACHE_RAYS, GI_CACHE_RAY_BIAS, GI_CACHE_REUSE, GI_CACHE_SLOTS, GI_CACHE_UPDATE_BUDGET, GI_CACHE_VISIBLE_CAPACITY, gi_accumulate, gi_cache_evict, gi_cache_peek, gi_cache_read, gi_cos_hemisphere, gi_current_generation, gi_dirty_hit, gi_entry_count, gi_entry_face, gi_entry_frame, gi_entry_generation, gi_entry_invalidate, gi_entry_m, gi_entry_nrm, gi_entry_obj, gi_entry_pos, gi_entry_reservoir_dir, gi_entry_reservoir_l, gi_entry_store_reservoir, gi_entry_valid, gi_entry_voxel, gi_entry_w_sum, gi_key0, gi_key1, gi_mix, gi_neighbor_entry, gi_rand2, gi_should_evict};
import package::gi::consts::{GI_AMBIENT_FLOOR, GI_CACHE_CONVERGED_N, GI_CACHE_VISIBLE_SHARE, GI_DEN_PLANE_K, GI_EMIT_GAIN, GI_LUMA, GI_PI, GI_SKY_AMBIENT, SHADOW_SURFACE_EPS};
import package::gi::denoise::{gi_den_atrous, gi_den_guide_store, gi_den_temporal};
import package::gi::ray::{gi_ray_radiance, sun_transmittance};
import package::gi::screen::{gi_ss_main, gi_ss_reproject};
import package::lightfield::{light_field_ao};
import package::trace::{make_grid};
import package::world::{WorldHit, voxel_normal, world_cfg_full, world_raycast, world_trace_medium};

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
    let gi_on = gi_u.misc.x > 0.5;
    var gi = vec3<f32>(0.0);
    var cov = 0.0;
    // GI 一律走缓冲通道（`gi_main` + 降噪写在 1/gi_div 分辨率的 GI 网格上，见 `gi/`）：
    // 这里只做一次双线性取值。uv 必须按**全分辨率尺寸**（`size`）归一化，不能除以 GI 纹理
    // 自身尺寸 —— 1:1（gi_div = 1）与 2:1（gi_div = 2）上采样都靠这条。
    // `gi_on = false` 时 gi/cov 恒 0 ⇒ 只有太阳直射 + 天光兜底。
    if (gi_on) {
      let guv = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5))
              / vec2<f32>(f32(size.x), f32(size.y));
      let s = textureSampleLevel(gi_tex, light_samp, guv, 0.0);
      let sc = textureSampleLevel(gi_cov_tex, light_samp, guv, 0.0);
      // premultiplied-valid 归一化：valid = 0（天空/介质）像素不参与插值。
      let gw = max(s.a, 1e-4);
      gi = s.rgb / gw;
      cov = sc.r / gw;
    }
    // AO 只乘到常量天光项（gi 已自带逐方向遮挡）。
    let ao = light_field_ao(p_voxel, best.obj_id);
    // cov 由缓存给出（命中 = 1、没有数据 = 0）：有数据处天光只占少量、无数据处天光兜底。
    // mix 实参顺序必须是 `mix(1.0, GI_AMBIENT_FLOOR, cov)`（cov = 1 → FLOOR）。
    let amb = sky * GI_SKY_AMBIENT * mix(1.0, GI_AMBIENT_FLOOR, cov);
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
    col = col + alb * palette_emissive(best.palette_base, best.pal) * GI_EMIT_GAIN;
    col = col * through + refl;
  }
  write_output(coord0, col);
}

// GI（菜单「渲染/GI」的分辨率档 `gi_u.flags.y` = 1 全分辨率 / 2 半分辨率）：GI 采样在本 pass 里
// 按该分辨率的网格完成，主 pass 按全分辨率 uv 双线性取用。
// 输出：gi_out.rgb = gi·valid、gi_out.a = valid；gi_cov_out.r = cov·valid、.g = valid；
// 主 pass 按 valid 归一化，天空/介质像素（valid = 0）不参与插值。
@compute @workgroup_size(8, 8, 1)
fn gi_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let size = textureDimensions(gi_out);
  if (gid.x >= size.x || gid.y >= size.y) { return; }
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

  // beam 纹理 = 全分辨率 / BEAM_DIV；本 pass 网格 = 全分辨率 / gi_div ⇒ 下标除 (BEAM_DIV / gi_div)。
  // gi_div = 1（全分辨率）→ 除 BEAM_DIV；gi_div = 2（半分辨率）→ 除 BEAM_DIV/2（= 旧版行为）。
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
  // ---- 两条路径二选一（`GI_CACHE_ON`，权威值在 `gi/cache.wesl`）----
  // false = **屏幕空间逐面 ReSTIR**（本步主路径，每帧完整估计 ⇒ 新露出像素当帧就有值）；
  // true  = 旧的世界空间逐 (体素,面) 缓存（A/B 对照，退役在第③步）。
  if (!GI_CACHE_ON) {
    gi_ss_main(coord, size, h);
    return;
  }
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
    let gs = gi_supply(h.voxel, h.face_id, h.obj_id, p_voxel + n * n_off, n);
    gi = gs.xyz;
    cov = gs.w;
    valid = 1.0;
    // 降噪导引（这条分支当前休眠 —— `GI_CACHE_ON = false` 时 `gi_main` 走 `gi_ss_main` 早返回；
    // 这里同步写一份，保证把开关翻回 true 时两段降噪读到的导引仍然自洽）。
    // 重投影用真实命中点（见 `gi/screen.wesl` 里同一处的说明）。
    var prev_ok = false;
    let pc = gi_ss_reproject(h.point, size, &prev_ok);
    gi_den_guide_store(coord, size, true, gi_key0(h.voxel), gi_key1(h.voxel, h.face_id, h.obj_id),
                       p_voxel, n, GI_DEN_PLANE_K * gg.scale, prev_ok, pc);
  } else {
    gi_den_guide_store(coord, size, false, 0u, 0u, vec3<f32>(0.0), vec3<f32>(0.0), 0.0, false,
                       vec2<i32>(0));
  }
  textureStore(gi_out, coord, vec4<f32>(gi * valid, valid));
  // storage texture 的 store 值恒为 vec4（rg32f 也不例外，多出的通道被丢弃）
  textureStore(gi_cov_out, coord, vec4<f32>(cov * valid, valid, 0.0, 0.0));
}

// ============================================================================
// GI 降噪（`gi/denoise.wesl`）：**时域累积** + **迭代 atrous** 两段，都跑在 GI 网格
// （= 渲染分辨率 / gi_div）上。
// 顺序（Rust `dispatch_dda`）：`gi_main` → `gi_denoise_temporal` → `gi_denoise_atrous1/2/4`
// （步长 1/2/4，次数 = `GI_DEN_ATROUS_ITER`）。最终结果落在 `gi_den`（`dda_main` 采样它，
// 原始 `gi_out` 保留作对照；`gi_cov_out` 全链不动）。
// ============================================================================

/// 时域累积：读本帧原始 GI + 导引 + 上帧历史 → 写累积色/矩/历史长度（双缓冲）+ φ。
@compute @workgroup_size(8, 8, 1)
fn gi_denoise_temporal(@builtin(global_invocation_id) gid: vec3<u32>) {
  let size = textureDimensions(gi_dn_tmp);
  if (gid.x >= size.x || gid.y >= size.y) { return; }
  gi_den_temporal(size, vec2<i32>(i32(gid.x), i32(gid.y)));
}

/// atrous 第 1 轮（步长 1）：源 = 时域输出，目标 = 见 Rust 侧的链。
@compute @workgroup_size(8, 8, 1)
fn gi_denoise_atrous1(@builtin(global_invocation_id) gid: vec3<u32>) {
  let size = textureDimensions(gi_dn_dst);
  if (gid.x >= size.x || gid.y >= size.y) { return; }
  gi_den_atrous(size, vec2<i32>(i32(gid.x), i32(gid.y)), 1);
}

/// atrous 第 2 轮（步长 2）。
@compute @workgroup_size(8, 8, 1)
fn gi_denoise_atrous2(@builtin(global_invocation_id) gid: vec3<u32>) {
  let size = textureDimensions(gi_dn_dst);
  if (gid.x >= size.x || gid.y >= size.y) { return; }
  gi_den_atrous(size, vec2<i32>(i32(gid.x), i32(gid.y)), 2);
}

/// atrous 第 3 轮（步长 4）。
@compute @workgroup_size(8, 8, 1)
fn gi_denoise_atrous4(@builtin(global_invocation_id) gid: vec3<u32>) {
  let size = textureDimensions(gi_dn_dst);
  if (gid.x >= size.x || gid.y >= size.y) { return; }
  gi_den_atrous(size, vec2<i32>(i32(gid.x), i32(gid.y)), 4);
}

// ============================================================================
// GI 路径（逐 (体素,面) 辐照度缓存，见 `gi/cache.wesl`）
// ============================================================================

/// 着色侧的 GI 取值：**只读缓存**（逐 (体素,面) 辐照度缓存）。
/// 条目未建 / 还没数据 → 返回 0，由着色侧既有的 `amb·ao`（天光常量 × AO）兜底。
/// 未建条目时缓存内部会顺手建条目并给出邻面播种值，所以大平面上第一帧就是平滑的。
fn gi_supply(
  voxel: vec3<i32>,
  face: u32,
  obj: i32,
  p_sample: vec3<f32>,
  n: vec3<f32>,
) -> vec4<f32> {
  if (!GI_CACHE_ON || gi_u.misc.x <= 0.5) { return vec4<f32>(0.0); }
  let c = gi_cache_read(voxel, face, obj, p_sample, n);
  return vec4<f32>(c.xyz * gi_u.params.z / GI_PI, c.w);
}

/// 缓存的更新 pass：**可见优先 + 收敛度优先**，每帧处理 `GI_CACHE_UPDATE_BUDGET` 个条目。
/// 预算切成两段（比例 = `GI_CACHE_VISIBLE_SHARE` %）：
///   · **可见段**消费着色侧上一帧 claim + append 的紧凑列表（`gi_vis_list`，即上一帧被看到的条目；
///     更新 pass 排在着色之前 ⇒ 有一帧延迟）；列表比可见段线程数长时取一个**轮换窗口**
///     （起点 `(frame × vis_threads) % cnt`，多帧内覆盖全部可见条目）；
///   · **轮转段**按帧号分片扫整表，覆盖远处 / 不可见条目（不饿死）；
///     两段里已收敛条目（`n ≥ GI_CACHE_CONVERGED_N`）都降频（`(idx + frame) % 4 != 0` 直接跳过），
///     把机会让给未收敛 / 新世代 / 脏区条目 —— 失效判定排在降频之前且重置后 `n = 0`，
///     所以世代过期与脏区条目永远优先且不降频。
/// 每个被选中的条目发 `GI_CACHE_RAYS` 条余弦加权半球射线（新鲜候选）+ `GI_CACHE_REUSE` 条**借用**射线
/// （取同平面邻条目 reservoir 里的方向，从本条目重发 = 重连接射线），按 RIS 合并成本帧估计
/// `Σ w / 候选数`（w = p̂/p）后喂给运行时累积（`gi_accumulate`）。借用让相邻条目的估计互相关联
/// ⇒ 大平面上的逐体素斑块被抹平；累积层保留 ⇒ 任何情况下都没有屏幕噪声。
/// 轮转起点由帧号推出 ⇒ 无跨帧游标、无竞态、丢帧不漏更新。
/// 两段都按 `[10]`（最近处理帧）去重：同一条目同帧只更新一次，不会累积两次。
/// 更新前先做「变化失效」判定：世代不符（太阳/天光/palette/世界全量变化）→ 就地重置；位置落在任一
/// 脏盒内（编辑体素，主世界 + 物体 volume，见 `gi_dirty_hit`）→ 就地重置。重置会连 reservoir 一起清空
/// （旧世代的方向/权重不可用）。
@compute @workgroup_size(64)
fn gi_cache_update(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= GI_CACHE_UPDATE_BUDGET) { return; }
  // 帧号取 `gi_u.seq.x`（精确 u32）：轮转起点、分片、去重、RNG 种子都必须是整数语义。
  let frame = gi_u.seq.x;

  // ---- 调度：预算切两段（可见优先 + 整表轮转）----
  let vis_threads = GI_CACHE_UPDATE_BUDGET * GI_CACHE_VISIBLE_SHARE / 100u;
  let rot_threads = GI_CACHE_UPDATE_BUDGET - vis_threads;
  var idx = 0u;
  // 本线程是否来自轮转段：只有轮转段做回收（可见段碰到的条目正要被看到，不该回收）。
  var from_rotation = false;
  if (i < vis_threads) {
    // 可见段：列表由着色 pass claim + append、Rust 每帧清零计数器 ⇒ 这里消费的是**上一帧**被看到的条目。
    let cnt = min(atomicLoad(&gi_vis_count[0u]), GI_CACHE_VISIBLE_CAPACITY);
    if (i >= cnt) { return; }
    // 轮换窗口：可见条目数可能远大于可见段线程数 ⇒ 起点随帧号推进（每帧前进 vis_threads 项，
    // cnt 帧内覆盖全部可见条目），而不是永远取前缀（那样列表尾部永远拿不到预算）。
    // cnt ≤ vis_threads 时 `(start + i) % cnt` 是 0..cnt-1 的一个排列，仍是一条不重复地覆盖全部。
    let start = (frame * vis_threads) % max(cnt, 1u);
    idx = gi_vis_list[(start + i) % max(cnt, 1u)];
  } else {
    // 轮转段：帧号分片（rot_threads × sweeps = GI_CACHE_SLOTS），覆盖远处 / 不可见条目。
    let sweeps = max(GI_CACHE_SLOTS / max(rot_threads, 1u), 1u);
    idx = ((frame % sweeps) * rot_threads + (i - vis_threads)) % GI_CACHE_SLOTS;
    from_rotation = true;
  }
  if (!gi_entry_valid(idx)) { return; }
  // 同帧去重：另一个分支已经处理过（`[10]` 已写成当前帧）就跳过。
  if (gi_entry_frame(idx) == frame) { return; }

  let gen = gi_current_generation();
  let pos = gi_entry_pos(idx);
  var invalid = gi_entry_generation(idx) != gen;
  if (!invalid) { invalid = gi_dirty_hit(pos); }
  if (invalid) { gi_entry_invalidate(idx, gen); }
  // 条目回收（只做在轮转段，见 `gi_should_evict`）：游标过水位后，把「久未露面」的条目清成空槽
  // —— 不回收的话游标只增不减，飞行十几屏新地形就耗光容量，新面永远建不出条目。
  // 分配游标耗尽时着色侧还能自己现场夺取一个可回收槽位（`gi_reclaim_slot`），这条只是提前腾地方。
  // 失效（世代过期 / 脏区）条目刚被就地重置、正要重收敛，**不回收**；已收敛但还年轻的条目也不回收。
  else if (from_rotation && gi_should_evict(idx, frame)) {
    gi_cache_evict(idx);
    return;
  }
  // 收敛降频：**两段都降频** —— 已收敛的可见条目每 4 帧才更新一次，否则预算被反复花在它们身上
  // （旧版可见段不降频时，「可见优先」实测反而比全轮转更慢，就是这个原因）；未收敛条目仍每帧更新。
  // 失效条目刚被重置成 `n = 0`，不会落进这里（世代过期 / 脏区永远优先且不降频）。
  if (gi_entry_count(idx) >= f32(GI_CACHE_CONVERGED_N)) {
    if ((idx + frame) % 4u != 0u) { return; }
  }

  let n = gi_entry_nrm(idx);
  let v = gi_entry_voxel(idx);
  let face = gi_entry_face(idx);
  let obj = gi_entry_obj(idx);
  let sun_dir = light_u.lights[0].kind_pos_dir.yzw;
  let sun_c = light_u.lights[0].color_intensity.xyz * light_u.lights[0].color_intensity.w;
  let org = pos + n * GI_CACHE_RAY_BIAS;

  // ---- RIS reservoir 的跨帧状态（失效 / 新建 = 0）----
  // w_sum / M：候选竞争的累计量，M 封顶后同比例缩回；邻居反读 `w_sum / M`（≈ 本条目的目标函数
  // 积分 Z）来算复用权重。sel_dir / sel_l：当前保留的样本（邻居只借方向，L 供邻居算源 pdf）。
  var w_sum = gi_entry_w_sum(idx);
  var m = gi_entry_m(idx);
  var sel_dir = gi_entry_reservoir_dir(idx);
  var sel_l = gi_entry_reservoir_l(idx);

  var acc = vec3<f32>(0.0);
  var cand = 0.0;
  for (var c = 0u; c < GI_CACHE_RAYS + GI_CACHE_REUSE; c = c + 1u) {
    var dir = vec3<f32>(0.0);
    // 权重系数 k：`w = p̂ / p = k · L`，由候选的**源 pdf** 决定（见两个分支）。
    var k = 1.0;
    if (c < GI_CACHE_RAYS) {
      // 新鲜候选：余弦加权方向 p = cosθ/π、p̂ = L·cosθ ⇒ w = π·L。
      // 随机数与旧版逐位一致（同 (条目, 帧, 序) 恒定 ⇒ 可复现、不逐帧抖动）。
      let seed = gi_mix(idx * 73856093u + frame * 19349663u + c * 83492791u + 1u);
      dir = gi_cos_hemisphere(n, gi_rand2(seed));
      k = GI_PI;
    } else {
      // 借用候选：邻条目 reservoir 的方向。邻位按条目下标轮转（`& 3u`），避免总是偏向同一侧；
      // 邻居没数据（未建 / 没样本 / 旧世代）就跳过这个候选。
      let nb = gi_neighbor_entry(v, face, obj, (c - GI_CACHE_RAYS + idx) & 3u);
      if (nb == 0xFFFFFFFFu) { continue; }
      // 源 pdf 取邻居 reservoir 的目标函数 p_nb(ω) ≈ lum(L_nb(ω))·cosθ / Z_nb：同一面 ⇒ cosθ 相同、
      // 约掉 ⇒ w = L·Z_nb / lum(L_nb(ω))。邻居样本恰好代表它自己的均值时退化为 π·L；样本偏亮时
      // （RIS 抽样的必然结果）这里会把它压回来，所以不会系统性过亮。只取「下压」侧（≤ 1）：
      // 邻居样本异常小时源 pdf 退化，否则权重会爆。
      let z_nb = gi_entry_w_sum(nb) / max(gi_entry_m(nb), 1.0);
      k = GI_PI * clamp(z_nb / max(GI_PI * dot(gi_entry_reservoir_l(nb), GI_LUMA), 1e-6), 0.0, 1.0);
      dir = gi_entry_reservoir_dir(nb);
    }
    // 单条射线的求值走**共用 helper**（`gi/ray.wesl`）：与屏幕空间新路径逐字同口径
    // （`E_store` 语义、除 π、二次顶点回读、太阳 NEE、自发光、逃逸天光）。
    let l = gi_ray_radiance(org, dir, sun_dir, sun_c).l;
    let w = l * k;
    let lum_w = dot(w, GI_LUMA);
    // 本帧估计 = Σ w / 候选数（RIS 的无偏写法：E[w] = ∫p̂ = π·⟨L⟩ = E_store）。
    acc = acc + w;
    cand = cand + 1.0;
    // 加权蓄水池抽样：候选 c 以「自己的权重 / 至今累计权重」成为 reservoir 样本。
    w_sum = w_sum + lum_w;
    m = m + 1.0;
    let u = gi_rand2(gi_mix(idx * 2654435761u + frame * 40503u + c * 2246822519u + 7u)).x;
    if (u * w_sum < lum_w) { sel_dir = dir; sel_l = l; }
  }
  // M 上限：w_sum 与 M 同比例缩回 ⇒ w_sum / M 不变（邻居看到的 Z 不变），只抑制历史对抽样的支配。
  if (m > f32(GI_CACHE_M_CAP)) {
    let shrink = f32(GI_CACHE_M_CAP) / m;
    w_sum = w_sum * shrink;
    m = f32(GI_CACHE_M_CAP);
  }
  gi_entry_store_reservoir(idx, sel_dir, sel_l, w_sum, m);
  // 一个候选都没有时（新鲜射线数为 0 才会发生）给 0，避免 0/0。
  gi_accumulate(idx, acc / max(cand, 1.0));
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
