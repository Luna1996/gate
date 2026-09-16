import package::bindings::{BEAM_BACKOFF, BEAM_DIV, DDGI_INDIR_CAST_BASE, DDGI_INDIR_COLLBASE_BASE, DDGI_INDIR_COLL_BASE, DDGI_INDIR_COLL_TOTAL, DDGI_INDIR_COUNT_BASE, DDGI_INDIR_RAYBASE_BASE, DDGI_INDIR_RAY_TOTAL, DDGI_INDIR_RPP_BASE, beam_depth, ddgi_args, ddgi_cell, ddgi_indirect, ddgi_meta, ddgi_samples, ddgi_slot_pos, ddgi_u, ddgi_worklist, eye_adapt, eye_adapt_ro, gi_cov_out, gi_cov_tex, gi_out, gi_tex, light_samp, light_u, out_tex, out_tex_src, view_u};
import package::common::{linear_to_srgb, palette_albedo, palette_emissive, palette_roughness, palette_transmission, sky_rgb, srgb_to_linear};
import package::ddgi::bake::{ddgi_bake_one};
import package::ddgi::collect::{ddgi_collect_one};
import package::ddgi::consts::{DDGI_AGE_MAX, DDGI_AMBIENT_FLOOR, DDGI_BAKE_STAGGER, DDGI_CASCADE_FADE_CELLS, DDGI_CAST_FLOOR, DDGI_COLLECT_THREADS, DDGI_CONF_HI, DDGI_CONF_LO, DDGI_DEPTH_TEXELS, DDGI_EMIT_GAIN, DDGI_LOD_COUNT, DDGI_META_ACTIVE, DDGI_META_NBR_VALID, DDGI_PI, DDGI_RAY_BIAS, DDGI_RAY_BUDGET, DDGI_REFRESH_PERIOD, DDGI_SKIP_AGE, DDGI_SKY_AMBIENT, DDGI_SKY_RADIANCE_SCALE, DDGI_T_MAX, DDGI_USAGE_SCAN_MASK, DDGI_WL_IDX_MASK, DDGI_WL_LOD_SHIFT, SHADOW_SURFACE_EPS};
import package::ddgi::helpers::{DDGI_REC_ENABLED, DDGI_REC_OCCUPIED, ddgi_lod_slot_base, ddgi_meta_age, ddgi_meta_nbr, ddgi_meta_pack, ddgi_meta_with_nbr, ddgi_oct_decode, ddgi_rand2, ddgi_ray_rand, ddgi_rec_off, ddgi_slot_lod_of, ddgi_slot_world_cell};
import package::ddgi::sample::{ddgi_box_edge_dist, ddgi_dbg_cov, ddgi_dbg_dom, ddgi_dbg_edge, ddgi_dbg_fb, ddgi_dbg_rej_out, ddgi_dbg_wsum, ddgi_dbg_zero, ddgi_sample, ddgi_usage_track};
import package::ddgi::sort::{ddgi_box_hits_object, ddgi_nbr_vis_mask, ddgi_neighbor_occupied};
import package::lightfield::{light_field_ao};
import package::trace::{make_grid};
import package::world::{WorldHit, voxel_normal, world_cfg_full, world_raycast, world_trace_medium};

// ============================================================================
// DDA compute shader：全屏逐像素层次栈式 mask DDA（Douglas 64-tree）。chunk 间 256³
// A&W（窗口 entry=0 的空 chunk 一次 load 跨越）；chunk 内 4 层栈帧（256→64→16→4→1），mask 一次
// load，bit=1 分裂才下钻、bit=0 uniform 整格跳过/命中。各 binding 布局见 bindings.wesl。
// 不变量：顶部常量与 Rust `brickmap::dda::wgsl_consts` 一致（改一侧必须同步另一侧）；寻址链对应
// `brickmap/view.rs::get_voxel`；slot 打包对应 `brickmap/wire.rs::encode_slot`。
// ============================================================================

// ============================================================================
// beam 预 pass：低分辨率（全分辨率 / BEAM_DIV）**保守** trace，只输出最近命中 t。
// 设 D_safe = 0.35/(BEAM_DIV·px_ang)，beam 射线只行进到 min(t_cap, D_safe)：
// d ≤ D_safe 的体素投影必被 3×3 邻域某条 beam 命中，d > D_safe 的由主射线从 D_safe 起步覆盖，
// 故输出与无 beam 逐像素一致。miss 存 t_cap_beam（非 t_cap）。
// ============================================================================
@compute @workgroup_size(8, 8, 1)
fn beam_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let beam_size = textureDimensions(beam_depth);
  if (gid.x >= beam_size.x || gid.y >= beam_size.y) { return; }
  // 低分辨率像素中心 → 全分辨率 NDC（用 beam_size 归一化，保证覆盖全视锥）
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
  // D_safe = 0.35/(BEAM_DIV·px_ang)：0.35 ≈ 1/(2√2) 覆盖 45° 最坏菱形 + 旋转裕量。
  // d ≤ D_safe 的体素投影双向宽 ≥ 2·BEAM_DIV px → 全局 beam 网格必有一点落进主像素 3×3 邻域。
  let d_beam = 0.35 / (f32(BEAM_DIV) * max(view_u.lod.x, 1e-6));
  let t_cap_beam = min(t_cap, d_beam);

  // depth_cap=3：八叉树全深度（不限制深度，只限制行进距离 t_cap_beam）
  let h = world_raycast(make_grid(0u), origin, dir, t_cap_beam, world_cfg_full(0.0, 3u));
  // miss 存 t_cap_beam（非 t_cap）：邻域全 miss 时主射线起点 ≤ d_beam，
  // d_beam 之外的几何由主射线完整 trace。
  let t = select(t_cap_beam, h.t, h.hit);
  textureStore(beam_depth, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(t, 0.0, 0.0, 0.0));
}

// ============================================================================
// 玻璃（介质）：折射/反射常量 + 阴影透过率
// ============================================================================

/// 玻璃折射率。菜单没有 IOR 参数，取常见玻璃 1.5（对应垂直入射反射率 F0 ≈ 0.04）。
const GLASS_IOR: f32 = 1.5;
/// 垂直入射的 Fresnel 反射率 F0 = ((1-ior)/(1+ior))²
const GLASS_F0: f32 = 0.04;
/// 折射/反射后新射线起点沿新方向外推的距离（voxel）：必须大于界面上的浮点噪声，
/// 又必须远小于 1 格（否则会跳过相邻体素）。
const GLASS_EPS: f32 = 1e-3;
/// 一条主射线最多处理的玻璃**界面**数（入射 + 出射各算一次 → 8 可穿过 4 块玻璃）。
/// DDA 内的介质段本身不限厚度（穿透在 DDA 里做），这里限的只是界面事件次数。
const GLASS_MAX_BOUNCE: u32 = 8u;
/// Fresnel 反射率门槛：低于此值不追反射射线（省一次 trace）。
const GLASS_REFL_MIN: f32 = 0.02;
/// 反射命中的近似辐亮度：反射射线**不追阴影、不采样 GI**（每像素只多一次 trace），
/// 用「阳光直照 × N·L + 常量天光」近似，足够给出玻璃反光里的明暗层次。
const GLASS_REFL_AMBIENT: f32 = 0.15;

/// 太阳方向的直光透过率（1 = 无遮挡，0 = 被不透明体素全挡）。
///
/// 一次**穿透 trace** 走完整段介质（见 [`world_trace_medium`]：DDA 内跨 uniform 块）——
/// 玻璃背后的直射光与玻璃厚度无关地正确，代价与一次普通 trace 同阶，不再逐格重跑。
/// `self_voxel`/`self_obj`：起点贴在命中面外侧，斜向可能出现 t≈0 自命中，豁免之
/// （与旧版 select 里的自命中防护同义）。
fn sun_transmittance(origin: vec3<f32>, sun_dir: vec3<f32>, self_voxel: vec3<i32>, self_obj: i32) -> f32 {
  // **先用普通模式取第一个命中**：阴影射线是长距离射线（8192），而穿透模式下**每个 chunk
  // 步进**都要去读介质状态 —— 绝大多数场景根本没有玻璃，那是纯浪费。只有第一段就撞上玻璃
  // 时才切到穿透模式（[`world_trace_medium`]）走完整段并累积透射色。
  let sh = world_raycast(make_grid(0u), origin, sun_dir, 8192.0, world_cfg_full(0.0, 3u));
  if (!sh.hit) { return 1.0; }
  if (all(sh.voxel == self_voxel) && sh.obj_id == self_obj) { return 1.0; }
  // 不透明判定只读 palette word1 的透射字节（`transmission == 0` ⇔ 不透明度 == 1）：
  // `medium_of` 会多读 word0 并算 mix/染色，而这里只用得到 `.a`。
  if (palette_transmission(sh.palette_base, sh.pal) <= 0.0) { return 0.0; }   // 不透明实体遮挡
  // 第一段就是玻璃 → 穿透走完（含自命中豁免）
  var tint = vec3<f32>(1.0);
  let h = world_trace_medium(make_grid(0u), origin, sun_dir, 8192.0, false, &tint);
  let through = (tint.r + tint.g + tint.b) / 3.0;   // 只透介质 → 透射色平均亮度作标量衰减
  if (!h.hit) { return through; }
  if (all(h.voxel == self_voxel) && h.obj_id == self_obj) { return 1.0; }
  if (palette_transmission(h.palette_base, h.pal) <= 0.0) { return 0.0; }
  return through;
}

// ============================================================================
// 玻璃（介质）状态机 —— **独立函数**
// ============================================================================
//
// 为什么必须独立成函数：状态机要维护 6 个跨轮存活的变量（org / dir / t_start / in_medium /
// tint / refl）。它们若与着色代码同处 `dda_main` 的函数体，寄存器分配会把两边的活跃区间
// 叠加 —— 而**不透明像素占绝大多数、根本不进这条路径**。实测：仅把状态机内联在 `dda_main`
// 里，帧率就从 150 掉到 100 以下，而逻辑上只多了几条指令（瓶颈是 occupancy，不是指令数）。
// 放进独立函数、并把调用点收进"第一次命中就是介质"的分支后，不透明像素的活变量集合
// 回到透明功能之前。
//
// 每轮处理一个**界面事件**：
//   · 空气侧命中可穿透介质 → 入射面：Fresnel 分出反射（追一条）与折射（进入介质）
//   · 介质内命中可穿透介质 → 出射面：折射回空气；全内反射时只改方向、仍在介质内
//   · 命中不透明实体       → 返回它，交给调用方着色
// 介质**段内的推进**在 DDA 里一次性完成（[`world_trace_medium`]），不逐格重跑 trace。

/// 状态机产物：最终命中的（不透明）表面 / 沿途累积的透射色 / 累积的反射项 /
/// 最终那段射线的方向（着色算高光半程向量要用**当前**方向，折射后已不是初始方向）。
struct GlassOut {
  ray: WorldHit,
  tint: vec3<f32>,
  refl: vec3<f32>,
  dir: vec3<f32>,
}

/// 见上方状态机说明。`t_min` 是 beam 预 pass 给出的保守起点，只对第一段有效。
fn trace_glass(origin_voxel: vec3<f32>, dir_voxel: vec3<f32>, t_min: f32, max_t: f32,
               sun_dir: vec3<f32>, sky: vec3<f32>, sun_c: vec3<f32>) -> GlassOut {
  var org = origin_voxel;
  var dir = dir_voxel;
  var t_start = t_min;
  var in_medium = false;
  var tint = vec3<f32>(1.0);   // 沿途透射色（乘性累积）
  var refl = vec3<f32>(0.0);   // 界面反射项（已按当时的 tint 加权）
  for (var bounce = 0u; bounce < GLASS_MAX_BOUNCE; bounce = bounce + 1u) {
    // 本段的透射色（乘性）：穿透模式下由 DDA 累积；正常模式恒为 1
    var seg = vec3<f32>(1.0);
    var best: WorldHit;
    if (in_medium) {
      // 介质内：一次走完整段（stop_on_air=true ⇒ 停在出射面，或撞上不透明实体）
      best = world_trace_medium(make_grid(0u), org, dir, max_t, true, &seg);
    } else {
      best = world_raycast(make_grid(0u), org, dir, max_t, world_cfg_full(t_start, 3u));
    }
    tint = tint * seg;
    // 一路走到底 → 剩余 tint 由调用方乘到天空上
    if (!best.hit) { return GlassOut(best, tint, refl, dir); }
    let tau = palette_transmission(best.palette_base, best.pal);
    if (tau <= 0.0) { return GlassOut(best, tint, refl, dir); }   // 不透明实体 → 调用方着色
    // ---- 玻璃界面 ----
    let p_hit = org + dir * best.t;
    let nn = best.n;                           // 世界空间外向面法线
    let ci = clamp(dot(-dir, nn), 0.0, 1.0);   // 入射余弦
    if (!in_medium) {
      // 入射面：Fresnel 分反射/折射
      let fres = GLASS_F0 + (1.0 - GLASS_F0) * pow(1.0 - ci, 5.0);
      if (fres > GLASS_REFL_MIN) {
        // 反射项：只追一条、且反射射线不穿透（遇到玻璃按普通命中处理）→ 每像素至多多一次 trace
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
      // 折射进介质（空气 → 玻璃在物理上不会全反射，k 恒 > 0）
      let eta = 1.0 / GLASS_IOR;
      let k = 1.0 - eta * eta * (1.0 - ci * ci);
      dir = normalize(eta * dir + (eta * ci - sqrt(max(k, 0.0))) * nn);
      in_medium = true;
    } else {
      // 出射面：介质 → 空气；k < 0 = 全内反射（没有折射光，方向变但仍在介质内）
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
    t_start = 0.0;   // 折射后方向已变 ⇒ beam 预 pass 的 t_min 失效，必须清零
  }
  // 界面次数用尽：当作没命中（剩余 tint 落到天空），不做无界追踪
  return GlassOut(WorldHit(false, 0.0, vec3<f32>(0.0), 0u, vec3<f32>(0.0), 0u, vec3<i32>(0), 0, 0u),
                  tint, refl, dir);
}

/// 自动曝光 + 编码输出（天空 / 不透明 / 玻璃三条路径共用，避免把这段重复三遍）。
/// 曝光系数必须在 sRGB 编码**之前**乘 —— 暗场提亮才能用上全部 8bit 动态范围
/// （编码之后乘会把暗部的量化台阶放大成色带）。
fn write_output(coord: vec2<i32>, col: vec3<f32>) {
  let exposure = bitcast<f32>(eye_adapt_ro[0u]);
  let exposed = col * select(1.0, exposure, exposure > 0.0);
  textureStore(out_tex, coord, vec4<f32>(linear_to_srgb(exposed), 1.0));
}

// ============================================================================
// DDA 主入口：命中 → 直光着色（硬阴影 + sky 环境 + emissive），miss → 纯色天空
// ============================================================================
@compute @workgroup_size(8, 8, 1)
fn dda_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let size = textureDimensions(out_tex);
  if (gid.x >= size.x || gid.y >= size.y) { return; }
  let coord0 = vec2<i32>(i32(gid.x), i32(gid.y));

  // ---- 「屏幕使用」记账开关（工作集选择的唯一依据，见 DDGI_USAGE_STICKY）----
  // 采样带 = 本帧轮到的那一列 ∪ 那一行 → 每个像素每 MASK+1 帧至少被采到一次，
  // 全局写入量摊到每帧约 2/(MASK+1) 的像素。cast 里的 GI 回读恒置 0，不记账。
  let usage_frame = u32(ddgi_u.params.x);
  ddgi_usage_track = select(
    0u,
    1u,
    ((gid.x + usage_frame) & DDGI_USAGE_SCAN_MASK) == 0u
      || ((gid.y + usage_frame) & DDGI_USAGE_SCAN_MASK) == 0u,
  );

  // ---- 反投影：像素中心 (gid + 0.5) → NDC (u, v) ∈ [-1, 1] ----
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

  // ---- 场景 trace（beam 起点跳过空空间）+ 逐体素法线着色 ----
  // 取当前像素 3×3 beam 邻域的最小命中 t 作为起点，跳过空空间；
  // beam 像素 = floor(全分辨率像素 / BEAM_DIV)，邻域 clamp 到 beam 边界。
  // lod.z > 0.5（GATE_NO_BEAM=1）：关闭 beam，t_min = 0。
  var t_min = 0.0;
  if (view_u.lod.z < 0.5) {
    let beam_size = textureDimensions(beam_depth);
    let bx = gid.x / BEAM_DIV;
    let by = gid.y / BEAM_DIV;
    t_min = 1e+30;
    // 3×3 对称邻域：d_beam 内体素投影双向宽 ≥ 2·BEAM_DIV px → 最近 beam 网格点
    // 距主像素 ≤ ~6 px（最坏 45° 菱形 ≈ 5.7 px）→ ±1 beam 格覆盖。
    for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
      for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
        let sx = clamp(i32(bx) + dx, 0, i32(beam_size.x) - 1);
        let sy = clamp(i32(by) + dy, 0, i32(beam_size.y) - 1);
        let bt = textureLoad(beam_depth, vec2<i32>(sx, sy)).x;
        t_min = min(t_min, bt);
      }
    }
    // 保守回退：邻域 beam 可从侧面/远面命中主射线首命中体素，命中 t 晚 ~2.3 单位；
    // 不回退会把起点推过真实命中（薄壁/剪影穿墙）。见 BEAM_BACKOFF。
    t_min = max(t_min - BEAM_BACKOFF, 0.0);
  }
  let sun_dir = light_u.lights[0].kind_pos_dir.yzw;
  let sky = light_u.sky_color.xyz;
  let sun_c = light_u.lights[0].color_intensity.xyz * light_u.lights[0].color_intensity.w;
  // ---- 第一条射线 ----
  // 不透明表面（绝大多数像素）走**短路径**：一次 trace + 着色，不进入玻璃状态机 ——
  // 状态机那 6 个跨轮变量因此不参与不透明像素的寄存器分配（见 [`trace_glass`]）。
  let first = world_raycast(make_grid(0u), origin_voxel, dir_voxel, frustum_length,
                            world_cfg_full(t_min, 3u));
  var best = first;
  var dir = dir_voxel;           // 最终照到表面的射线方向（穿过玻璃时已被折射）
  var through = vec3<f32>(1.0);  // 沿途累积的透射色（无玻璃时恒为 1）
  var refl = vec3<f32>(0.0);     // 玻璃界面的反射项
  // 可穿透判定只读 palette word1 的透射字节（`transmission > 0` ⇔ 不透明度 < 1）——这是
  // 每个命中像素都要过的一步，省掉 `medium_of` 的第二次 palette load 与 mix。
  if (first.hit && palette_transmission(first.palette_base, first.pal) > 0.0) {
    // 第一次命中就是可穿透介质 → 交给状态机推进到"最终的不透明命中 / 走到底"
    let g = trace_glass(origin_voxel, dir_voxel, t_min, frustum_length, sun_dir, sky, sun_c);
    best = g.ray;
    dir = g.dir;
    through = g.tint;
    refl = g.refl;
  }
  // ---- 着色：天空 / 不透明表面 ----
  var col = sky_rgb() * through + refl;
  if (best.hit) {
    col = vec3<f32>(0.0);
    // ---- 逐体素着色 + 逐体素法线 ----
    // albedo/采样点/阴影射线按命中体素锚定（同体素同色）；法线取隐式逐体素法线（退化退面法线，
    // 见 voxel_normal）—— 它同时决定太阳 N·L、阴影射线/GI 采样点的外推方向、DDGI 方向采样。
    let alb = palette_albedo(best.palette_base, best.pal);
    // 体素中心 → 世界系（主世界 identity 直等于体素中心；物体经旋转/缩放变换）
    let gg = make_grid(u32(best.obj_id) + 1u);
    let vc = vec3<f32>(best.voxel) + vec3<f32>(0.5);
    let p_voxel = gg.pos + vec3<f32>(dot(vc, gg.col0), dot(vc, gg.col1), dot(vc, gg.col2)) * gg.scale;
    let n = voxel_normal(gg, best.voxel, best.n);
    // 沿 n 外推的距离必须保证离开命中体素：逐体素法线在棱边/圆角处是斜的，0.5·n 的分量
    // < 0.5 → 起点仍在体素内 → DDA 先命中所属体素的邻居（棱边被自己遮住 → 一圈暗边）。
    // 0.5/max|n| 即「从中心沿 n 走多远离开单位立方体」，任何方向都落在空气侧。
    let n_off = 0.5 / max(max(abs(n.x), abs(n.y)), max(abs(n.z), 1e-3)) + SHADOW_SURFACE_EPS;
    // 直光阴影：1 条太阳射线（途经玻璃按介质穿透累积透射色，见 sun_transmittance）。射线原点 =
    // p_voxel + n×n_off（贴空气侧）。
    // SHADOW_SURFACE_EPS 必需：+n×0.5 恰落在面平面上，-X/-Y/-Z 面坐标是整数 → DDA 的 floor
    // 落回体素自己 → t=0 自命中，而自命中防护把「命中自己」当无遮挡 → 这三向的面
    // 无论墙多厚都吃到太阳（封闭无光房间的明暗会完全跟着面朝向走）。
    // +X/+Y/+Z 面落在高侧边界，floor 到邻域空气格。
    let ndl = max(dot(n, sun_dir), 0.0);
    var sun = 0.0;
    if (ndl > 0.0) {
      sun = sun_transmittance(p_voxel + n * n_off, sun_dir, best.voxel, best.obj_id);
    }
    let gi_on = ddgi_u.misc.x > 0.5;
    var gi = vec3<f32>(0.0);
    // cov = 「本点周围有没有探针数据」的覆盖度，供 conf 用。内联路径由 `ddgi_sample` 发布到
    // `ddgi_dbg_cov`；半分辨率路径从 GI 缓冲里带出来（同一个量，只是按 2×2 像素合并过）。
    var cov = 0.0;
    // 半分辨率 GI（菜单开关 `flags.y`）：采样已在 `gi_main` 里按 1/2 分辨率算好，这里只双线性取用。
    // 诊断档（params.y > 0.5）强制走内联路径 —— 那些档要看逐像素的采样细节（wsum/cov/rej/dom）。
    let gi_half = gi_on && ddgi_u.flags.y > 0.5 && ddgi_u.params.y < 0.5;
    if (gi_half) {
      // uv 按**全分辨率**尺寸（`size` = out_tex 尺寸）归一化，**不能**除以 GI 纹理自己的尺寸：
      // GI 缓冲是 1/2 分辨率，除以它会得到 2× 拉开的坐标 ⇒ GI 相对几何整体错位/偏移。
      // 双线性取用自动混合本像素所属的 2×2 块（texel 中心落在 (2i+1)/full）。
      let guv = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5))
              / vec2<f32>(f32(size.x), f32(size.y));
      let s = textureSampleLevel(gi_tex, light_samp, guv, 0.0);
      let sc = textureSampleLevel(gi_cov_tex, light_samp, guv, 0.0);
      // premultiplied-valid 归一化：无数据（天空/介质）像素 valid = 0，不参与插值，
      // 否则 2×2 quad 里混进天空会沿几何边缘把 GI 拉黑。
      let gw = max(s.a, 1e-4);
      gi = s.rgb / gw;
      cov = sc.r / gw;
    } else if (gi_on) {
      // 采样点只沿 n 推到体素表面外侧（n_off），保证离开命中体素；随 cell 尺寸缩放的外推
      // 由 ddgi_sample_lod 内部按该 LOD 的 cs 施加（DDGI_BIAS_CELLS）。
      gi = ddgi_sample(p_voxel + n * n_off, n) * ddgi_u.params.z / DDGI_PI;
      cov = ddgi_dbg_cov;
    }
    // AO（Douglas #15）只作用于**常量天光**这一项：gi 已自带逐方向遮挡（探针射线自身
    // 就做了可见性判定），再乘一次 16³ 粗粒度填充率的 AO 属重复计算，且会随相机滚动
    // 重铺光照场而整片跳变。常量天光没有任何遮挡信息，AO 正是为它准备的。
    let ao = light_field_ao(p_voxel, best.obj_id);
    // 天光常量兜底：本像素不在任何级联盒内、或盒内但采样数据不可信时补上；数据可信时
    // 一律由 GI 提供（无光室内的探针辐照度≈0 → 画面接近全黑，室内外才有亮度差）。
    // conf = smoothstep(CONF_LO, CONF_HI, ddgi_dbg_cov)：判据用覆盖度 cov（本点有没有探针数据），
    // 而非 wsum（Σ wtri·wn·wd 另乘了两道照明闸门，会把「本就该黑的凹角」误判成数据不可信）；
    // conf 只补天光常量，不缩放 gi 的幅度。
    let conf = smoothstep(DDGI_CONF_LO, DDGI_CONF_HI, cov);
    let amb_sky = sky * DDGI_SKY_AMBIENT;
    // t_edge 是几何量（离盒边界多远），只决定用哪种兜底强度（盒外 amb_far / 盒内 amb_sky）；
    // 补多少完全由 conf 决定 → 有数据时 amb 不受 Far amb 影响。
    // 它只依赖 p 与最外层 LOD 盒（纯函数），与走哪条 GI 采样路径无关 ⇒ 这里本地重算，
    // 不再依赖 `ddgi_dbg_edge`（半分辨率档下那个私有量不会被赋值）。与 `ddgi_sample` 内的
    // 式子逐字一致：clamp(盒外距离 / (cs × 过渡带宽), 0, 1)。
    let outer_lod = DDGI_LOD_COUNT - 1u;
    let w_edge = max(f32(ddgi_u.lods[outer_lod].origin.w) * DDGI_CASCADE_FADE_CELLS, 1e-3);
    let t_edge = smoothstep(0.0, 1.0,
      clamp(ddgi_box_edge_dist(outer_lod, p_voxel) / w_edge, 0.0, 1.0));
    let amb_far = sky * ddgi_u.misc.w;
    // 下限不取 0：conf=1 时归零会让无光室内恒为纯黑，而 auto exposure 对恒 0 无效（0·s = 0）；
    // DDGI_AMBIENT_FLOOR 是"多次反弹/天光渗透"的常量替身。
    // mix 实参顺序必须是 (1.0, DDGI_AMBIENT_FLOOR, conf)：conf=1 → FLOOR（只留地板），
    // conf=0 → 1.0（满额兜底，与 DDGI 关闭时一致）；写成 mix(FLOOR, 1.0, conf) 语义相反。
    let amb = mix(amb_far, amb_sky, t_edge) * mix(1.0, DDGI_AMBIENT_FLOOR, conf);
    col = alb * (sun_c * ndl * sun + amb * ao + gi);
    // ---- 光滑度 → 太阳高光（Blinn-Phong 半程向量）----
    // 粗糙度 0 = 镜面（指数 256：窄而亮），1 = 完全粗糙（指数 4：宽而弱）。
    // 用隐式逐体素法线 n，棱边/圆角上的高光自然过渡到平滑；乘 `sun` 让阴影里不出高光。
    // 峰值再按粗糙度衰减（粗糙面把高光能量摊到更宽的方向上，峰值必然更低）。
    {
      let rough = palette_roughness(best.palette_base, best.pal);
      let to_cam = -dir;   // 用**当前**射线方向：折射后 dir 已不是初始方向
      let half_sum = sun_dir + to_cam;
      // 半程向量在 L≈-V 时退化 → 除零；夹住长度
      let half_vec = half_sum / max(length(half_sum), 1e-5);
      let spec_pow = mix(256.0, 4.0, rough);
      let spec = pow(max(dot(n, half_vec), 0.0), spec_pow) * sun * ndl;
      col = col + sun_c * alb * spec * mix(1.0, 0.15, rough);
    }
    // 自发光体素：radiance 直出（不吃方向、不吃阴影）——「每个体素都能是光源」的着色侧。
    col = col + alb * palette_emissive(best.palette_base, best.pal) * DDGI_EMIT_GAIN;
    if (ddgi_u.params.y > 0.5) {
      if (ddgi_u.params.y < 1.5) {
        // GI 档 = 纯 gi ×8（便于观察），刻意不加任何常量天光项：本档里出现漏光 ⇒ 在 gi 侧；
        // 本档干净而只有 Normal 档漏光 ⇒ 在 amb（天光常量 × AO）或 sun 侧。
        col = alb * gi * 8.0;
      } else if (ddgi_u.params.y < 2.5) {
        // wsum 档 = 采样置信度的灰度直方图（1:1，不吃 albedo）。
        // wsum = Σ wtri·wn·wd 是 trilinear 权重和 ∈ [0,1]，开阔表面约 0.3-0.6 → 直接当灰度。
        col = vec3<f32>(clamp(ddgi_dbg_wsum, 0.0, 1.0));
      } else if (ddgi_u.params.y < 3.5) {
        // d = 「包含该像素的壳」所在 LOD + 1（纯几何包含，与采样是否成功无关）；
        // d = 0 表示没有任何 LOD 盒包含该像素 —— 这是覆盖率问题，不是剔除问题。
        // 4 级给出 4 种可分辨颜色。
        let d = ddgi_dbg_dom;
        col = vec3<f32>(0.1, 0.8, 0.2);                          // 绿：未被任何 LOD 包含
        col = select(col, vec3<f32>(0.9, 0.2, 0.2), d > 0.5);    // 红：LOD0
        col = select(col, vec3<f32>(0.95, 0.85, 0.1), d > 1.5);  // 黄：LOD1
        col = select(col, vec3<f32>(0.2, 0.4, 0.9), d > 2.5);    // 蓝：LOD2
        col = select(col, vec3<f32>(0.9, 0.2, 0.9), d > 3.5);    // 品红：LOD3
      } else {
        // Probe 档：先看数据有没有取到，再看 8 个角被哪道闸门剔掉（颜色 ↔ ddgi_dbg_fb 阈值）。
        //   fb > 4.5 青绿 = 本级壳 r1/r2/r3' 全失败，改用同侧探针场插值（ddgi_sample 的 r4）兜住；
        //   fb > 3.5 紫 = 背向闸门放开的末级兜底 r3' 成功 → GI 有效；fb > 2.5 亮黄 = 壳内 8 角被
        //     wn 全剔（薄几何），用同级放宽 wn 兜住 → GI 有效（不退到粗级，故不会跨墙漏光）。
        // 其余按 wsum / 最大剔除原因染色：绿 = wsum > 0 且辐照度非 0；白 = wsum > 0 但辐照度≈0
        //（过闸探针图集为空）；灰 = 没跑过 sample_lod 或 8 角全被窗口/ENABLED 剔掉；
        // 青 = 无数据闸门；橙 = ENABLED；品红 = wn <= 0；红 = wd <= 0。
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
          let rmax = max(r_ia, max(r_wd, max(r_wn, r_np)));
          if (rmax <= 0.0) {
            col = vec3<f32>(0.5, 0.5, 0.5);
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
    // 不透明实体：表面亮度按沿途透射色衰减、叠上玻璃界面反射项后收工
    col = col * through + refl;
  }
  write_output(coord0, col);
}

// ============================================================================
// 半分辨率 GI（菜单开关 `ddgi_u.flags.y`）：把 GI 采样从主 pass 搬到 1/2 分辨率的独立 pass。
//
// 为什么可以：GI 是**低频量**（探针间距 = cell 16~128 voxel，主 pass 只是对 8 个探针做三线性
// 混合 + 逐角精确可见性），而 `ddgi_sample` 的成本（8 角闸门循环 + 逐角线段求交 + 图集取样）
// 是主 pass 最大的一块。按 2×2 像素合并采样、主 pass 双线性取用 ⇒ 这块成本按 1/4 削减。
//
// 精度取舍（关掉开关即回到逐像素精确路径）：
//   · GI 的可见性/接触阴影分辨率减半（薄结构边缘变软、跨 2 像素的几何缝隙可能被平均掉）；
//   · 介质（玻璃）像素不产出 GI：`ddgi_sample` 的折射路径不可复现（要走完整玻璃状态机），
//     这里只对「首命中即不透明」的像素采样 ⇒ 玻璃像素落到缓冲里 valid=0 的位置，
//     采样侧退回 conf=0 的天光兜底（本场景无透射材质，实际无差异）。
//
// 输出（写入侧在 BG5）：
//   gi_out.rgb = gi·valid、gi_out.a = valid
//   gi_cov_out.r = cov·valid、gi_cov_out.g = valid
//   ⇒ 主 pass 双线性后按 valid 归一化：天空/介质像素（valid = 0）不参与插值。
// ============================================================================
@compute @workgroup_size(8, 8, 1)
fn gi_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let size = textureDimensions(gi_out);
  if (gid.x >= size.x || gid.y >= size.y) { return; }
  let coord = vec2<i32>(i32(gid.x), i32(gid.y));

  // ---- 「屏幕使用」记账开关：与 dda_main 同一套（本 pass 的像素代表 2×2 全分辨率像素，
  // 列带 ∪ 行带的抽样比例因此不变；见 DDGI_USAGE_SCAN_MASK）----
  let usage_frame = u32(ddgi_u.params.x);
  ddgi_usage_track = select(
    0u,
    1u,
    ((gid.x + usage_frame) & DDGI_USAGE_SCAN_MASK) == 0u
      || ((gid.y + usage_frame) & DDGI_USAGE_SCAN_MASK) == 0u,
  );

  // ---- 反投影：本 pass 像素中心（半分辨率）→ NDC ----
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

  // ---- beam 起点跳过空空间 ----
  // beam 纹理 = 全分辨率 / BEAM_DIV(=4) = 本 pass 的 /2 ⇒ 本 pass 1 个 beam 纹素覆盖 2×2 本 pass
  // 像素，故下标要再除 (BEAM_DIV/2)。邻域/回退与 `beam_main` 的 d_beam 约定同源（BEAM_BACKOFF）。
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
  // 只处理「首命中即不透明」：着色点/法线/采样点的推导与 `dda_main` 的不透明分支逐字一致。
  if (h.hit && palette_transmission(h.palette_base, h.pal) <= 0.0) {
    let gg = make_grid(u32(h.obj_id) + 1u);
    let vc = vec3<f32>(h.voxel) + vec3<f32>(0.5);
    let p_voxel = gg.pos + vec3<f32>(dot(vc, gg.col0), dot(vc, gg.col1), dot(vc, gg.col2)) * gg.scale;
    let n = voxel_normal(gg, h.voxel, h.n);
    let n_off = 0.5 / max(max(abs(n.x), abs(n.y)), max(abs(n.z), 1e-3)) + SHADOW_SURFACE_EPS;
    gi = ddgi_sample(p_voxel + n * n_off, n) * ddgi_u.params.z / DDGI_PI;
    cov = ddgi_dbg_cov;
    valid = 1.0;
  }
  textureStore(gi_out, coord, vec4<f32>(gi * valid, valid));
  // storage texture 的 store 值恒为 vec4（rg32f 也不例外，多出的通道被丢弃）
  textureStore(gi_cov_out, coord, vec4<f32>(cov * valid, valid, 0.0, 0.0));
}


// ============================================================================
// 阶段一 seal：独立 pass（pass 边界保证计数器可见）。4 线程各管一个 LOD：读活跃探针数
// → 算 rpp（预算/活跃数，clamp 1..256）→ 写 cast/collect indirect args。
// ============================================================================
@compute @workgroup_size(4, 1, 1)
fn ddgi_seal(@builtin(local_invocation_id) lid: vec3<u32>) {
  let lod = lid.x;
  // 每线程各自读 4 个 LOD 的活跃计数并算前缀和 —— 计数都在全局内存里，无需 workgroup 同步；
  // thread 0 另写出总线程数与唯一一份 indirect args。
  // rpp 全局统一：射线预算在全部活跃探针间均分。
  var total_active = 0u;
  for (var l = 0u; l < DDGI_LOD_COUNT; l = l + 1u) {
    total_active = total_active + atomicLoad(&ddgi_indirect[DDGI_INDIR_COUNT_BASE + l]);
  }
  // 下限 1：rpp ≥ 1 是硬性的（每个活跃探针每帧至少 1 条射线）。故
  // total_ray ≤ max(total_active, DDGI_RAY_BUDGET)，其上界是总槽位数，
  // samples 缓冲按 `max(预算, 总槽位)` 定容（见 Rust `ddgi_sample_bytes`）。
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
    // 每探针 rpp 射线、每探针 320 纹素线程，WG 均为 64 → 两次 indirect dispatch，y=z=1
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

// ============================================================================
// 阶段一 烘焙：逐 cell 沿 4³ 分裂树 BFS 找「最大的全空叶」并把探针放其中心（同级取更靠近
// cell 中心的空叶）；全空 cell → 居中，全满 cell → 无探针。输出 ddgi_cell[slot] =
// flags(bit0 探针存在 / bit1 cell 有体素) | cell 内归一化偏移，sort 每帧只读该记录。
// 增量：ddgi_cell_id[slot] 记录已烘焙的世界 cell 键，键未变的槽位直接跳过（续龄）。
// ============================================================================

// 4 个 per-LOD 入口：Rust 侧按 lod 0→3（细→粗）各派发一个独立 compute pass，
// pass 边界即内存屏障 ⇒ 粗级能读到本帧细级刚写好的 ddgi_cell / ddgi_cell_id。
@compute @workgroup_size(64)
fn ddgi_bake0(@builtin(global_invocation_id) gid: vec3<u32>) { ddgi_bake_one(0u, gid.x); }
@compute @workgroup_size(64)
fn ddgi_bake1(@builtin(global_invocation_id) gid: vec3<u32>) { ddgi_bake_one(1u, gid.x); }
@compute @workgroup_size(64)
fn ddgi_bake2(@builtin(global_invocation_id) gid: vec3<u32>) { ddgi_bake_one(2u, gid.x); }
@compute @workgroup_size(64)
fn ddgi_bake3(@builtin(global_invocation_id) gid: vec3<u32>) { ddgi_bake_one(3u, gid.x); }

@compute @workgroup_size(64)
fn ddgi_sort(@builtin(global_invocation_id) gid: vec3<u32>) {
  let slot = gid.x;
  if (slot >= u32(ddgi_u.misc.y)) { return; }
  let lod = ddgi_slot_lod_of(slot);
  let idx = slot - ddgi_lod_slot_base(lod);
  let rec = ddgi_cell[slot];
  let cs = ddgi_u.lods[lod].origin.w;
  let wc = ddgi_slot_world_cell(lod, idx);
  let cmin = wc * cs;

  let enabled = (rec & DDGI_REC_ENABLED) != 0u;
  if (!enabled) {
    // 本格没有探针：清 meta 与 slot_pos 即可。
    ddgi_meta[slot] = 0u;
    ddgi_slot_pos[slot] = vec4<f32>(0.0, 0.0, 0.0, -1.0);
    return;
  }
  let off = vec3<f32>(ddgi_rec_off(rec)) / 255.0;
  let probe_pos = vec3<f32>(cmin) + off * f32(cs);

  // ---- 「本 cell 附近有没有几何」：活跃判定的输入（只对 enabled 的格计算）----
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

  // ---- age 继承（同 slot 且 world cell 未变 → 续龄；换 cell 时 bake 已清零）----
  // prev_meta 一次读出复用：下面判"邻接可见性掩码是否已在有效期内"也要它（省一次 buffer load）。
  let prev_meta = ddgi_meta[slot];
  var age = ddgi_meta_age(prev_meta);
  // ---- 世界编辑（脏区）→ 把年龄压回 1：长窗口不牺牲编辑响应 ----
  // `dirty_min.w` 只在"本帧真的上传了世界数据"那一帧为 1（无脏区时 min == max ⇒ w = 0），
  // 故不会误触发、也不会常驻为真。压低 age ⇒ 该探针回到 `ddgi_age_alpha` 的 1/age 等权斜坡，
  // 并因 `!converged` 进入本帧 worklist。判据与 bake 的脏区判定逐字一致（cell 的 voxel AABB
  // 与脏区 AABB 相交），保证"被重烘的 cell"与"被重置 age 的 cell"是同一批。
  let cell_lo = vec3<f32>(cmin);
  let cell_hi = cell_lo + f32(cs);
  if (ddgi_u.dirty_min.w > 0.5
      && all(cell_hi > ddgi_u.dirty_min.xyz)
      && all(cell_lo < ddgi_u.dirty_max.xyz)) {
    // 按槽位号错峰（与 bake 换主路径同一套做法，见 bake.wesl 的 DDGI_BAKE_STAGGER 注释）：
    // 脏区是一个 chunk 级 AABB，同帧把整片探针的 age 都压到 1，会让它们在下一帧**一起**以
    // alpha=1/2 换成"单帧高方差估计" ⇒ 一整块间接光同帧偏移，视觉上就是"编辑后闪一下"。
    // 错开后每帧只有 1/N 的探针发生跳变，打散空间一致性；平均跟随速度不变（age 仍从下往上跑满斜坡）。
    age = 1u + (slot % DDGI_BAKE_STAGGER);
  }
  // 判活条件：有探针 && (本 cell 或 6 邻接有体素 || 与物体 AABB 相交)。
  let is_active = near;

  if (is_active) {
    // ---- 本帧要不要投线：按「屏幕会不会用到它」（见 consts 的 DDGI_USAGE_STICKY）----
    // 三个入选项：① used = 上一帧被着色采样过（漏桶计数 > 0）；② !converged = 年龄不足
    // （新探针 / 相机滚动换主），让它尽快成型；③ due = 按槽位号错峰的强制刷新，
    // 兜住从没上过屏的探针。记账在此读后衰减（-1/帧）。
    let usage = ddgi_cell_slot[slot];
    ddgi_cell_slot[slot] = select(0u, usage - 1u, usage > 0u);
    let used = usage > 0u;
    let frame = u32(ddgi_u.params.x);
    let converged = age >= DDGI_SKIP_AGE[lod];
    let due = (frame + slot) % DDGI_REFRESH_PERIOD[lod] == 0u;
    if (used || !converged || due) {
      age = min(age + 1u, DDGI_AGE_MAX);
      let wslot = atomicAdd(&ddgi_indirect[DDGI_INDIR_COUNT_BASE + lod], 1u);
      // worklist 是压缩后的列表，其下标（wslot）只是排名，不是 cell 下标。packed：低 8 位 age、
      // 中 19 位「探针在本级网格里的 cell 下标」、bit27 起 lod —— collect 必须用 cell 下标
      // 才能把数据写进正确的图集纹素（见 DDGI_WL_IDX_MASK）。
      let packed = age | (idx << 8u) | (lod << DDGI_WL_LOD_SHIFT);
      ddgi_worklist[ddgi_lod_slot_base(lod) + wslot] =
        vec4<f32>(probe_pos, bitcast<f32>(packed));
    }
  }

  // ---- 邻接可见性掩码（空间混合门控）：只对活跃探针算（非活跃者不投线、图集不更新，
  // 也不会作为混来源被邻居采用）。非活跃路径不置 NBR_VALID → 消费侧按"未算过"处理（不做门控）。
  //
  // 【只在本级可能变了时重算】掩码 = f(探针实际位置, 世界几何, cs)，而这三者的变化时机都是已知的：
  //   · 换主 / LOD0 池重分配 / 网格重建 → bake 重写该槽 meta（丢掉 NBR_VALID）⇒ 这里 !valid 自动触发；
  //   · 世界编辑 → 该 cell 内探针可能被 `ddgi_place_probe` 搬家（位置变），几何也可能挡住/放通某条轴
  //     向射线 ⇒ 按脏区 AABB 判。**判据必须把 cell 盒沿各轴外扩 cs**：射线长 cs、起点在 cell 内，
  //     故线段 AABB ⊆ cell 盒 ⊕ cs；只用 cell 盒会漏掉"贴 chunk 边界的 cell"朝外那一侧
  //     （那一侧的射线落进相邻 chunk，编辑恰在那里时掩码会陈旧）。
  // 其余帧直接沿用上一帧掩码 —— 6 条射线 × 每个活跃探针 × 每帧就是 sort 的全部成本
  // （实测 0.44ms → 0.06ms）。DDGI 网格锚定世界 AABB、不随相机换主，所以相机移动不会误触发重算。
  var meta_word = ddgi_meta_pack(age, true, is_active);
  if (is_active) {
    var mask = ddgi_meta_nbr(prev_meta);
    // 外扩量 = 本级 cell 边长（cs 是 i32，见 `DdgiLod.origin: IVec4` 的 .w）
    let pad = vec3<f32>(f32(cs));
    let dirty_here = ddgi_u.dirty_min.w > 0.5
      && all(cell_hi + pad > ddgi_u.dirty_min.xyz)
      && all(cell_lo - pad < ddgi_u.dirty_max.xyz);
    if ((prev_meta & DDGI_META_NBR_VALID) == 0u || dirty_here) {
      mask = ddgi_nbr_vis_mask(probe_pos, f32(cs));
    }
    meta_word = ddgi_meta_with_nbr(meta_word, mask);
  }
  ddgi_meta[slot] = meta_word;
  ddgi_slot_pos[slot] = vec4<f32>(probe_pos, f32(lod));
}

// ============================================================================
// 阶段二 cast（Douglas pass #2）：worklist 里的活跃探针各投 rpp 条射线。
// 射线方向 = 绑定的八面体深度纹素方向 + 纹素内抖动（纹素按 frame 轮换覆盖）。
// 命中体素 → 该处上一帧 GI 乘 albedo 得出射辐亮度；命中天空 → 天空色。
// 输出 ddgi_samples（方向 + 命中距离）/（辐亮度）。
// 线程映射：全局 tid = ray_base[lod] + probe_idx × rpp + ray（ray_base 由 ddgi_seal 算），
// 一次 dispatch 覆盖全部 LOD。
// ============================================================================
@compute @workgroup_size(64)
fn ddgi_cast(@builtin(global_invocation_id) gid: vec3<u32>) {
  let tid = gid.x;
  if (tid >= atomicLoad(&ddgi_indirect[DDGI_INDIR_RAY_TOTAL])) { return; }
  // 全局 tid → LOD：ray_base 单调递增，取最后一个 base ≤ tid 的那级
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
  // RNG 种子用探针自己的 cell 下标（而非 worklist 排名：排名每帧会变，会让同一探针的
  // 射线方向逐帧跳变）。时间维的随机性由 frame 提供。
  let cell_idx = (bitcast<u32>(ddgi_worklist[slot].w) >> 8u) & DDGI_WL_IDX_MASK;

  // 【射线 ↔ 深度纹素绑定】每条射线固定归属一个八面体深度纹素，纹素按
  // (frame·rpp + ray) 轮换 → 每帧覆盖 rpp 个纹素、⌈N²/rpp⌉ 帧轮完一圈（N=8 → 64 个）；方向 =
  // 该纹素中心 + 纹素内抖动（幅度限制在纹素内，避免固定方向产生 banding）。
  // 每个纹素的样本跨帧同属一个方向，mean/std 才是 Chebyshev 需要的「该方向的最近距离 + 噪声」。
  let ndim = DDGI_DEPTH_TEXELS;
  let ntex = ndim * ndim;
  let tsel = (frame * rpp + ray) % ntex;
  let ttx = tsel % ndim;
  let tty = tsel / ndim;
  let jit = ddgi_rand2(ddgi_ray_rand(ddgi_lod_slot_base(lod) + cell_idx, frame, ray));
  let te = ((vec2<f32>(f32(ttx), f32(tty)) + jit) / f32(ndim)) * 2.0 - vec2<f32>(1.0);
  let dir = ddgi_oct_decode(te);
  let sh = world_raycast(make_grid(0u), probe_pos + dir * DDGI_RAY_BIAS, dir, DDGI_T_MAX, world_cfg_full(0.0, 3u));
  // miss 分支注入天空色 × DDGI_SKY_RADIANCE_SCALE（置 0 = GI 里没有天空贡献，用于确认
  // "封闭室内还亮着"是否由射线逃逸到天空引起）。命中分支的 `alb·gi/π` 是自反馈：
  // `DDGI_CAST_FLOOR = 0` ⇒ 其自身不含任何照明量。
  // 世界之外不分方向一律注入天空纯色（纯色环境，无"地面/虚空"之分）。
  var radiance = sky_rgb() * DDGI_SKY_RADIANCE_SCALE;
  var dist = DDGI_T_MAX;
  if (sh.hit) {
    dist = sh.t;
    let alb = palette_albedo(sh.palette_base, sh.pal);
    let hit_p = probe_pos + dir * dist;
    // 与 dda_main 同一套逐体素法线（退化退面法线）：反弹辐照度的方向必须与着色侧一致，
    // 否则曲面上的间接光会与直射/环境光"对不上"。代价是每条 GI 射线 6 次 world_solid。
    let n = voxel_normal(
      make_grid(u32(sh.obj_id) + 1u),
      sh.voxel,
      sh.n,
    );
    // 入射辐照度 E → 出射辐亮度 L = albedo·E/π（与着色侧同约定）。albedo·sky·CAST_FLOOR
    // 作下限，避免首帧 GI 全 0 时反馈回路死锁在 0；该下限必须小，否则封闭房间也被抬亮。
    let gi = ddgi_sample(hit_p + n * 0.5, n);
    // 命中发光体素 → 出射辐亮度含自身发射（radiance 直出，不吃 GI/入射）。
    radiance = alb * (gi / DDGI_PI + sky_rgb() * DDGI_CAST_FLOOR)
      + alb * palette_emissive(sh.palette_base, sh.pal) * DDGI_EMIT_GAIN;
  }
  // 样本下标 = 全局射线编号（tid 本身就是「该 LOD 起始射线号 + 本探针号×rpp + 射线号」）
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
  let sel = u32(view_u.probe_viz_params.z);
  if (sel != 0u && lod != sel - 1u) {
    return;
  }
  // 只可视化「活跃」探针（DDGI_META_ACTIVE），而非所有烘焙探针
  if ((ddgi_meta[id] & DDGI_META_ACTIVE) == 0u) {
    return;
  }
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

// ============================================================================
// 自动曝光（结构对齐 UE/Unity 的 EyeAdaptation）：
//   pass1 `eye_adapt_histogram`：out_tex 按 1/16 步长抽样 → sRGB 还原 → log2 亮度 → 64 桶直方图
//   pass2 `eye_adapt_update`：取 5%~95% 百分位区间求平均（剔除亮窗/太阳这类离群值）
//                             → 反馈 + 分方向时间平滑 → 写曝光系数（下一帧 `dda_main` 用）
// 统计在"曝光已应用之后"的颜色上，反馈环把百分位平均亮度压到目标中灰。
// ============================================================================
/// log2 亮度桶覆盖 [-12, +12]：24 档 / 64 桶 = 0.375 档每桶，远细于 8bit 量化，够用。
const EYE_LUM_MIN_LOG2: f32 = -12.0;
const EYE_LUM_RANGE_LOG2: f32 = 24.0;
const EYE_HIST_BINS: u32 = 64u;
/// 活参数区起始字（= 状态/调试区 8 + 直方图 64），由 Rust `EyeAdaptSettings` 上传。
/// 下标 = `eye_p(i)` 的实参，顺序即语义，改 Rust 结构体时必须同步这里：
///   0 ev_max  1 ev_min  2 tau_brighten  3 tau_darken  4 key
const EYE_PARAM_BASE: u32 = 8u + EYE_HIST_BINS;

fn eye_p(i: u32) -> f32 {
  return bitcast<f32>(atomicLoad(&eye_adapt[EYE_PARAM_BASE + i]));
}

/// WG 内共享直方图。必须是模块级 workgroup 变量：WGSL 规定原子类型只能出现在
/// storage / workgroup 地址空间，函数内的局部 var 不能是 atomic。
var<workgroup> eye_hist: array<atomic<u32>, 64>;

@compute @workgroup_size(256)
fn eye_adapt_histogram(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dims = vec2<i32>(textureDimensions(out_tex_src));
  // 先局部原子累加，最后一次性写全局（全局原子量压到 64/帧）。
  for (var i = 0u; i < EYE_HIST_BINS; i = i + 1u) { atomicStore(&eye_hist[i], 0u); }
  workgroupBarrier();
  // 抽样：x/y 各每 16 像素取一个中心点；256 线程按 grid-stride 覆盖全部抽样点。
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
    // 5%~95% 百分位裁剪：按累计计数取区间内的加权平均 log2 亮度
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
      // 反馈：把"当前（已含曝光）的平均亮度"往目标中灰拉 → 还差多少档
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
  // 直方图清零，留给下一帧累加（本 pass 与 pass1 分属不同 pass，pass 边界即屏障）
  for (var i = 0u; i < EYE_HIST_BINS; i = i + 1u) { atomicStore(&eye_adapt[8u + i], 0u); }
}
