import package::bindings::{BEAM_BACKOFF, BEAM_DIV, beam_depth, eye_adapt, eye_adapt_ro, gi_dn_dst, gi_dn_tmp, gi_guide_out, gi_out, gi_tex, gi_u, light_samp, light_u, out_tex, out_tex_src, view_u};
import package::common::{linear_to_srgb, palette_albedo, palette_emissive, palette_roughness, palette_transmission, sky_rgb, srgb_to_linear};
import package::gi::common::{gi_key0, gi_key1, gi_unpack_dir};
import package::gi::consts::{GI_AMBIENT_FLOOR, GI_DEN_G_KEY0, GI_DEN_G_KEY1, GI_DEN_G_NRM, GI_DEN_GUIDE_WORDS, GI_EMIT_GAIN, GI_SKY_AMBIENT, SHADOW_SURFACE_EPS};
import package::gi::denoise::{gi_den_atrous, gi_den_temporal, gi_same_surface};
import package::gi::ray::{sun_transmittance};
import package::gi::screen::{gi_ss_main};
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

/// **几何感知上采样**（joint bilateral）—— `dda_main` 取 GI 的唯一入口。
/// GI 跑在低于全分辨率的网格上（`GiSettings.gi_div` = 1 / 2 / 4），纯双线性会把**几何边界
/// 两侧**的值混在一起（棱边/墙角一侧的亮 GI 渗到另一侧），而这正是「GI 低于全分辨率」唯一
/// 真正丢掉的信息。档位越低（1/4）这一层越要紧：一个 GI texel 覆盖 4×4 个全分辨率像素。
/// 做法：取 GI 网格上最近的 2×2 texel，**只接受与本像素同一个面**的 texel
/// （`gi_same_surface`：平面归属靠键 ⇒ 薄板另一侧 / 隔一格的平行面不可能被接受；
/// 相似度靠**存的着色法线** ⇒ 凸棱上那一段倾斜的法线与本像素比仍在容差内 ⇒ 与着色一致地连起来），
/// 按双线性权重加权平均。退化顺序：一个都没接受 ⇒ **最近的 texel**（它至少在几何上覆盖本像素，
/// 比双线性更不容易渗色）⇒ 最近 texel 无有效数据 ⇒ 传进来的纯双线性结果。
/// 有效性判据 = 导引的「键非零」（与降噪侧一致）；最近 texel 那一档直接用纹理的 `a`（那里不插值，
/// `a` 恒是 0/1）。
/// `n` = 本像素的着色法线（`dda_main` 自己那条）—— 相似度那一条用它。
/// 代价：每全分辨率像素 4 次导引读（每次 3 word），**不加任何射线、不加 pass**。
fn gi_upsample_joint(
  guv: vec2<f32>,
  gsize: vec2<u32>,
  k0: u32,
  k1: u32,
  n: vec3<f32>,
  fallback: vec3<f32>,
) -> vec3<f32> {
  let g = guv * vec2<f32>(f32(gsize.x), f32(gsize.y));
  let base = floor(g - vec2<f32>(0.5));
  let f = g - vec2<f32>(0.5) - base;
  var acc = vec3<f32>(0.0);
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
      // 有效 texel 的 a 恒为 1（`gi_out` 与降噪链都写 `vec4(v, 1.0)`）⇒ `rgb` 就是 gi 本身。
      acc = acc + textureLoad(gi_tex, c, 0).rgb * bw;
      wsum = wsum + bw;
    }
  }
  if (wsum > 0.0) { return acc / wsum; }
  // 一个同面的 texel 都没有（凸角 / 贴天空的轮廓）：退回最近的 texel。
  let nc = clamp(base + vec2<f32>(select(0.0, 1.0, f.x >= 0.5), select(0.0, 1.0, f.y >= 0.5)),
                 vec2<f32>(0.0), vec2<f32>(f32(gsize.x) - 1.0, f32(gsize.y) - 1.0));
  let sn = textureLoad(gi_tex, vec2<i32>(nc), 0);
  if (sn.a > 0.5) { return sn.rgb; }
  return fallback;
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
    let k0 = gi_key0(best.voxel);
    let k1 = gi_key1(best.voxel, best.face_id, best.obj_id);
    // 着色法线（`voxel_normal`，主世界要探 4 个 / 物体 6 个邻域体素）：**先试导引复用，键相等就用它、不再自己算**。
    // 复用合法性由键给出：键逐位编码了精确体素 + 面号 + 物体，而 `voxel_normal` 是
    // `(命中体素, 可见面)` 的纯函数 ⇒ 键逐位相等 ⇒ 法线必然逐位相等。
    // 只在 gi_div = 1 时尝试：此时 GI 射线与本像素**逐位相同**，导引那格必然命中同一体素面，
    // 于是这里的复用替代掉整条法线。gi_div > 1 时导引那格来自另一条射线（更粗的网格、射向格中心），
    // 它命中的**体素**在多数像素上都与本像素不同（1/4 档实测复用率 ≈ 0，主 pass 时间无变化）
    // ⇒ 不做尝试，连那次必然失败的键读取都省掉。
    var n = best.n;
    var n_reused = false;
    if (u32(gi_u.flags.y) == 1u) {
      let gi_size = vec2<u32>(textureDimensions(gi_tex));
      if (gid.x < gi_size.x && gid.y < gi_size.y) {
        let gb = (gid.y * gi_size.x + gid.x) * GI_DEN_GUIDE_WORDS;
        if (gi_guide_out[gb + GI_DEN_G_KEY0] == k0 && gi_guide_out[gb + GI_DEN_G_KEY1] == k1) {
          n = gi_unpack_dir(gi_guide_out[gb + GI_DEN_G_NRM]);
          n_reused = true;
        }
      }
    }
    if (!n_reused) { n = voxel_normal(gg, best.voxel, best.face_id, best.n); }
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
    // 这里只做一次几何感知上采样（内部用 2×2 texel）。uv 必须按**全分辨率尺寸**（`size`）
    // 归一化，不能除以 GI 纹理自身尺寸 —— 1:1 / 2:1 / 4:1 上采样都靠这条。
    // `gi_on = false` 时 gi/cov 恒 0 ⇒ 只有太阳直射 + 天光兜底。
    if (gi_on) {
      let guv = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5))
              / vec2<f32>(f32(size.x), f32(size.y));
      let s = textureSampleLevel(gi_tex, light_samp, guv, 0.0);
      // premultiplied-valid 归一化：valid = 0（天空/介质）像素不参与插值。
      let gw = max(s.a, 1e-4);
      // 覆盖度 cov 直接取自 `a`：`gi_out.a = valid`，而独立的 cov 通道写的恒是「valid ? 1 : 0」
      // ⇒ `cov = valid / max(valid, 1e-4)` 与旧的 `sc.r / gw` 是同一个表达式（不再单开一张纹理）。
      // GI 值走**几何感知上采样**（纯双线性作 fallback）：不接受与本像素不同「面」的 texel
      // ⇒ 棱边 / 墙角不渗色、不糊（相似度那一条用本像素的着色法线 `n`）。cov 不参与
      // （它是 0/1 覆盖度，双线性归一化即可）。
      gi = gi_upsample_joint(guv, vec2<u32>(textureDimensions(gi_tex)), k0, k1, n, s.rgb / gw);
      cov = s.a / gw;
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

// GI（**唯一路径 = 屏幕空间逐面 ReSTIR**，见 `gi/screen.wesl`）。
// 本 pass 只负责：反投影 + beam 起点 + 主 trace（= G-Buffer），随后整条 GI 都交给 `gi_ss_main`
// （新鲜候选 → 时域复用 → 空间复用 → 着色）并写降噪导引。
// 输出：gi_out.rgb = gi·valid、gi_out.a = valid（覆盖度 cov 从 `a` 导出，见 `dda_main`）；
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
  // gi_div = 1 → 除 BEAM_DIV；gi_div = 2 → 除 BEAM_DIV/2；gi_div = 4 → 1:1（一个 GI 像素正好一个 beam 纹素）。
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
  // 天空 / 介质像素（首命中不是不透明体素）由 `gi_ss_main` 写 valid = 0 后早返回。
  gi_ss_main(coord, size, h);
}

// ============================================================================
// GI 降噪（`gi/denoise.wesl`）：**时域累积** + **迭代 atrous** 两段，都跑在 GI 网格
// （= 渲染分辨率 / gi_div）上。
// 顺序（Rust `dispatch_dda`）：`gi_main` → `gi_denoise_temporal` → `gi_denoise_atrous1/2/4/8/16`
// （步长 1/2/4/8/16 = NRD/RELAX 的 5 轮 a-trous；实际跑几轮 = `GI_DEN_ATROUS_ITER`，
// 中间靶只有 4 张 ⇒ 后面几轮靠 a↔b ping-pong）。最终结果落在 `gi_den`（`dda_main` 采样它，
// 原始 `gi_out` 保留作对照）。
// ============================================================================

/// 时域累积：读本帧原始 GI + 导引 + 上帧历史 → 写累积色/矩/历史长度（双缓冲）+ φ。
@compute @workgroup_size(8, 8, 1)
fn gi_denoise_temporal(@builtin(global_invocation_id) gid: vec3<u32>) {
  let size = textureDimensions(gi_dn_tmp);
  if (gid.x >= size.x || gid.y >= size.y) { return; }
  gi_den_temporal(size, vec2<i32>(i32(gid.x), i32(gid.y)));
}

/// atrous 第 1 轮（步长 1）。
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

/// atrous 第 4 轮（步长 8）。GI 是低频信号，大半径才不会伤到细节；
/// 硬拒判据（不是同一个面就 cut）与半径无关 ⇒ 大核也不会跨几何糊。
@compute @workgroup_size(8, 8, 1)
fn gi_denoise_atrous8(@builtin(global_invocation_id) gid: vec3<u32>) {
  let size = textureDimensions(gi_dn_dst);
  if (gid.x >= size.x || gid.y >= size.y) { return; }
  gi_den_atrous(size, vec2<i32>(i32(gid.x), i32(gid.y)), 8);
}

/// atrous 第 5 轮（步长 16）：足迹半径 = `GI_DEN_ATROUS_R` × 16 个 GI 像素。
@compute @workgroup_size(8, 8, 1)
fn gi_denoise_atrous16(@builtin(global_invocation_id) gid: vec3<u32>) {
  let size = textureDimensions(gi_dn_dst);
  if (gid.x >= size.x || gid.y >= size.y) { return; }
  gi_den_atrous(size, vec2<i32>(i32(gid.x), i32(gid.y)), 16);
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
