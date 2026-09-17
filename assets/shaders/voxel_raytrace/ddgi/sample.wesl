import package::bindings::{ddgi_cell, ddgi_depth, ddgi_irr, ddgi_meta, ddgi_slot_pos, ddgi_u};
import package::ddgi::consts::{DDGI_BIAS_CELLS, DDGI_BIAS_MAX_VOXELS, DDGI_CASCADE_BLEND, DDGI_CASCADE_FADE_CELLS, DDGI_DEPTH_SOFT_MIN, DDGI_DEPTH_TEXELS, DDGI_FIELD_MIN_W, DDGI_IRR_TEXELS, DDGI_LOD_COUNT, DDGI_META_ACTIVE, DDGI_META_ENABLED, DDGI_MIN_SAMPLE_AGE, DDGI_NORMAL_BIAS, DDGI_OCCL_END_BIAS, DDGI_OCCL_W_MIN, DDGI_TEXEL_MIN_COVERAGE};
import package::ddgi::helpers::{DDGI_REC_OCCUPIED, ddgi_cell_in_window, ddgi_depth_coord, ddgi_irr_coord, ddgi_meta_age, ddgi_oct_encode, ddgi_origin_cell, ddgi_slot, ddgi_usage_mark};
import package::trace::{make_grid};
import package::world::{world_cfg_full, world_raycast, world_solid};

/// 辐照度图集纹素：`.w` = 覆盖度（被 collect 写过就向 1 靠拢，用于区分「探针活过」与「从未投线」）。
fn ddgi_irr_fetch(id: u32, tx: i32, ty: i32) -> vec4<f32> {
  let c = ddgi_irr_coord(id, u32(tx), u32(ty));
  return textureLoad(ddgi_irr, vec2<i32>(vec2<u32>(c.y, c.z)), i32(c.x), 0);
}
/// 八面体双线性采样中与探针无关的那一半：4 个纹素坐标 + 两轴插值权重（8 角循环不变量）。
struct DdgiIrrTaps {
  x0: i32,
  y0: i32,
  x1: i32,
  y1: i32,
  fx: vec4<f32>,
  fy: vec4<f32>,
}
fn ddgi_irr_taps(d: vec3<f32>) -> DdgiIrrTaps {
  let s = f32(DDGI_IRR_TEXELS);
  let e = ddgi_oct_encode(d) * 0.5 + vec2<f32>(0.5);
  let g2 = e * s - vec2<f32>(0.5);
  let g0 = floor(g2);
  let f = g2 - g0;
  // 上界 = DDGI_IRR_TEXELS-1（本探针自身的最后一个纹素）。
  let tmax = i32(DDGI_IRR_TEXELS) - 1;
  return DdgiIrrTaps(
    clamp(i32(g0.x), 0, tmax),
    clamp(i32(g0.y), 0, tmax),
    clamp(i32(g0.x) + 1, 0, tmax),
    clamp(i32(g0.y) + 1, 0, tmax),
    vec4<f32>(vec3<f32>(f.x), f.x),
    vec4<f32>(vec3<f32>(f.y), f.y),
  );
}
/// 用已算好的纹素坐标/权重对探针 `id` 取 4 个纹素做双线性。
fn ddgi_irr_sample(id: u32, t: DdgiIrrTaps) -> vec4<f32> {
  let c00 = ddgi_irr_fetch(id, t.x0, t.y0);
  let c10 = ddgi_irr_fetch(id, t.x1, t.y0);
  let c01 = ddgi_irr_fetch(id, t.x0, t.y1);
  let c11 = ddgi_irr_fetch(id, t.x1, t.y1);
  return mix(mix(c00, c10, t.fx), mix(c01, c11, t.fx), t.fy);
}
fn ddgi_depth_fetch(id: u32, x: i32, y: i32) -> vec2<f32> {
  let c = ddgi_depth_coord(id, u32(x), u32(y));
  return textureLoad(ddgi_depth, vec2<i32>(vec2<u32>(c.y, c.z)), i32(c.x), 0).xy;
}
// 返回 vec2(mean, std)：该方向最近距离的均值与其标准差（图集 .x/.y）。
// 在最近邻 2×2 邻域取平均；仅统计 mean>0 的纹素，四个都无数据才回 0（采样侧剔除）。
fn ddgi_depth_sample(id: u32, d: vec3<f32>) -> vec2<f32> {
  let s = f32(DDGI_DEPTH_TEXELS);
  let e = ddgi_oct_encode(d) * 0.5 + vec2<f32>(0.5);
  let g2 = e * s - vec2<f32>(0.5);
  // 上界 = DDGI_DEPTH_TEXELS-1（本探针自身的最后一个纹素）。
  let tmax = i32(DDGI_DEPTH_TEXELS) - 1;
  let x0 = clamp(i32(floor(g2.x)), 0, tmax);
  let y0 = clamp(i32(floor(g2.y)), 0, tmax);
  let x1 = clamp(x0 + 1, 0, tmax);
  let y1 = clamp(y0 + 1, 0, tmax);
  let v00 = ddgi_depth_fetch(id, x0, y0);
  let v10 = ddgi_depth_fetch(id, x1, y0);
  let v01 = ddgi_depth_fetch(id, x0, y1);
  let v11 = ddgi_depth_fetch(id, x1, y1);
  var acc = vec2<f32>(0.0);
  var n = 0.0;
  if (v00.x > 0.0) { acc = acc + v00; n = n + 1.0; }
  if (v10.x > 0.0) { acc = acc + v10; n = n + 1.0; }
  if (v01.x > 0.0) { acc = acc + v01; n = n + 1.0; }
  if (v11.x > 0.0) { acc = acc + v11; n = n + 1.0; }
  return select(vec2<f32>(0.0), acc / max(n, 1.0), n > 0.0);
}
/// p 到 LOD(lod) 盒外边界面距离：盒内为正、盒外为负（六个面取最小）。
fn ddgi_box_edge_dist(lod: u32, p: vec3<f32>) -> f32 {
  let L = ddgi_u.lods[lod];
  let lo = vec3<f32>(L.origin.xyz);
  let hi = lo + vec3<f32>(L.dims.xyz) * f32(L.origin.w);
  return min(
    min(min(p.x - lo.x, hi.x - p.x), min(p.y - lo.y, hi.y - p.y)),
    min(p.z - lo.z, hi.z - p.z),
  );
}
// p 是否落在 LOD(lod) 的「壳」内：在 lod 盒内、且不在更细一级 lod-1 盒内。
fn ddgi_lod_contains(lod: u32, p: vec3<f32>) -> bool {
  let L = ddgi_u.lods[lod];
  let o = vec3<f32>(L.origin.xyz);
  let ext = vec3<f32>(L.dims.xyz) * f32(L.origin.w);
  if (!(all(p >= o) && all(p < o + ext))) { return false; }
  if (lod == 0u) { return true; }
  let Lo = ddgi_u.lods[lod - 1u];
  let lo_f = vec3<f32>(Lo.origin.xyz);
  let hi_f = lo_f + vec3<f32>(Lo.dims.xyz) * f32(Lo.origin.w);
  return !(all(p >= lo_f) && all(p < hi_f));
}

/// 采样 LOD(lod) 的 8 角三线性，返回 vec4(辐照度.xyz, 权重和)。
/// 三个开关：`clamp_cells` 越界 cell 号 clamp 进本级窗口；`relax_normal` 下限抬到 0.25；`allow_backface` 放开 wn 符号闸门。
fn ddgi_sample_lod(
  p: vec3<f32>,
  n: vec3<f32>,
  lod: u32,
  clamp_cells: bool,
  relax_normal: bool,
  allow_backface: bool,
) -> vec4<f32> {
  let L = ddgi_u.lods[lod];
  let cs = f32(L.origin.w);
  // 主世界网格，逐角几何遮挡的数据源；构造一次复用给全部 8 个角。
  let gw = make_grid(0u);
  // 逐角几何遮挡配置（完整层次 DDA）：t_min = 0 从探针位置起测；max_t 由调用点按「线段长 − DDGI_OCCL_END_BIAS」传入。
  let occl_cfg = world_cfg_full(0.0, 3u);
  // 探针标称位置在 cell 中心 → cell 坐标先平移半格再 floor（三线性基）。
  // 采样点沿法线外推 min(cs × DDGI_BIAS_CELLS, DDGI_BIAS_MAX_VOXELS)；仍落在实体体素内则退回 p。
  var ps = p + n * min(cs * DDGI_BIAS_CELLS, DDGI_BIAS_MAX_VOXELS);
  if (world_solid(gw, vec3<i32>(floor(ps)))) { ps = p; }
  // 用世界 cell 号（槽位 = 世界 cell 对网格维度取模），与烘焙/判活侧同一套世界身份。
  let wc_f = ps / cs - vec3<f32>(0.5);
  let wc0 = vec3<i32>(floor(wc_f));
  let fr = wc_f - floor(wc_f);
  let o_cell = ddgi_origin_cell(lod);
  let di = vec3<i32>(L.dims.xyz);
  let wc_hi = o_cell + di - vec3<i32>(1);
  let irr_taps = ddgi_irr_taps(n);
  // clamp 域 = 本级空间盒；LOD0 再叠「chunk 已领段」判定（空 chunk 的 cell 被剔除）。
  var total = vec3<f32>(0.0);
  var wsum = 0.0;
  // `cov`：只累加「有可用数据」的角，不含照明闸门。
  var cov = 0.0;
  // 「cell 放不下探针」的角的名义权重和，`cov` 的分母修正项。
  var cov_solid = 0.0;
  // 「本点一侧辐照度场」的分子/分母累加器。
  var fld = vec3<f32>(0.0);
  var fw = 0.0;
  ddgi_dbg_rej = vec4<u32>(0u, 0u, 0u, 0u);
  ddgi_dbg_occl = 0u;
  for (var iz = 0; iz < 2; iz = iz + 1) {
    for (var iy = 0; iy < 2; iy = iy + 1) {
      for (var ix = 0; ix < 2; ix = ix + 1) {
        var wc = wc0 + vec3<i32>(ix, iy, iz);
        // 权重只依赖循环下标与 `fr`；`cov_solid` 分母修正要用它。
        let wl = vec3<f32>(
          select(1.0 - fr.x, fr.x, ix == 1),
          select(1.0 - fr.y, fr.y, iy == 1),
          select(1.0 - fr.z, fr.z, iz == 1),
        );
        let wtri = wl.x * wl.y * wl.z;
        if (clamp_cells) {
          wc = clamp(wc, o_cell, wc_hi);
          if (!ddgi_cell_in_window(lod, wc)) {
            ddgi_dbg_rej.x = ddgi_dbg_rej.x + 1u;
            continue;
          }
        } else if (!ddgi_cell_in_window(lod, wc)) {
          ddgi_dbg_rej.x = ddgi_dbg_rej.x + 1u;
          continue;
        }
        let slot = ddgi_slot(lod, wc);
        let mw = ddgi_meta[slot];
        if ((mw & DDGI_META_ENABLED) == 0u) {
          // 只有 OCCUPIED=1（cell 放不下探针）才计入 cov_solid 分母修正。
          if ((ddgi_cell[slot] & DDGI_REC_OCCUPIED) != 0u) { cov_solid = cov_solid + wtri; }
          ddgi_dbg_rej.x = ddgi_dbg_rej.x + 1u;
          continue;
        }
        // ACTIVE 是逐帧量（只决定本帧是否发新射线），不作采样闸门。
        // 年龄 < DDGI_MIN_SAMPLE_AGE 的角跳过（图集里是旧位置 / 未覆盖读数）。
        if (ddgi_meta_age(mw) < DDGI_MIN_SAMPLE_AGE) { ddgi_dbg_rej.w = ddgi_dbg_rej.w + 1u; continue; }
        // 插值基 = 晶格名义位置上的三线性基（权重场连续），非探针实际位置的 tent。
        // 探针实际位置仍参与 wn/wd（朝向/可见性用真实几何）。
        if (wtri <= 1e-6) { continue; }
        // ---- 记账：候选角（有数据、判活、年龄够、权重非零）即算「本帧被屏幕用到」，不要求过可见性闸门。 ----
        // 只记紧贴本像素的角（与基础角格差 ≤ 2 格）。
        if (ddgi_usage_track != 0u && all(abs(wc - wc0) <= vec3<i32>(2))) { ddgi_usage_mark(slot); }
        let probe = ddgi_slot_pos[slot].xyz;
        let to = ps - probe;
        let dist = length(to);
        let dir = to / max(dist, 1e-4);
        // ===== 「这个角的数据」是否可用 =====
        // `wn ≤ 0`（背面）与 `wd` 小（被挡）是「本角照明无效」：本角既不进 total/wsum，也不进 cov。
        let dtex = ddgi_depth_sample(slot, dir);
        // 本点方向辐照度；`.w` = 该纹素的覆盖度（判断探针是否活）。
        let irr4 = ddgi_irr_sample(slot, irr_taps);
        // 深度无记录 = 无遮挡（wd 保持 1），不算「没有数据」。
        // 唯一剔除条件：探针从未被写过（覆盖度 < DDGI_TEXEL_MIN_COVERAGE）。
        if (dtex.x <= 0.0 && irr4.w < DDGI_TEXEL_MIN_COVERAGE) {
          ddgi_dbg_rej.w = ddgi_dbg_rej.w + 1u;
          continue;
        }
        // `cov` 必须在所有照明闸门（wn / wd / ε / 求交）之前累加：它衡量的是「这片探针有没有数据」。
        cov = cov + wtri;
        // ===== 「照明是否可用」判定 =====
        // 背向闸门用未 clamp 的 `wn_raw`；除 `allow_backface`（r3'）外一律剔除。
        let wn_raw = dot(n, -dir) / DDGI_NORMAL_BIAS;
        if (!allow_backface && wn_raw <= 0.0) {
          ddgi_dbg_rej.y = ddgi_dbg_rej.y + 1u;
          continue;
        }
        let wn = clamp(wn_raw, 0.0, 1.0);
        let wn_w = select(wn, max(wn, 0.25), relax_normal);
        // ---- 遮挡：Chebyshev 软判定 wd = soft²/(soft² + delta²) ----
        let d_vis = dist - max(cs * 0.02, dist * 0.005);
        let mean = dtex.x;
        var wd = 1.0;
        if (mean > 0.0 && d_vis > mean) {
          // [诊断] `misc.z` = std 项信任系数（默认 1，0 = 忽略 dtex.y 退成固定 2% 硬判定）。
          let soft = max(dtex.y * ddgi_u.misc.z, DDGI_DEPTH_SOFT_MIN);
          let delta = d_vis - mean;
          wd = (soft * soft) / (soft * soft + delta * delta);
        }
        if (wd <= 1e-3) { ddgi_dbg_rej.z = ddgi_dbg_rej.z + 1u; continue; }
        let irr = irr4.xyz;
        // ===== 权重阈值 ε + 逐角几何求交（一次测、两处用） =====
        let w = wtri * wn_w * wd;
        // 求交条件：`w ≥ ε`（进 total/wsum）或同侧角 `wn_raw > 0`（r4 需要）。
        let need_test = w >= DDGI_OCCL_W_MIN || wn_raw > 0.0;
        var occluded = false;
        // 【量测开关】`flags.x > 0.5`（Rust 侧 `GATE_NO_OCCL=1`）：整段跳过逐角几何求交，8 角全部按"未遮挡"处理。
        if (need_test && dist > 0.0 && ddgi_u.flags.x < 0.5) {
          // max_t = 线段长 − DDGI_OCCL_END_BIAS；`dist == 0` 时不求交。
          occluded = world_raycast(gw, probe, to / dist, dist - DDGI_OCCL_END_BIAS, occl_cfg).hit;
        }
        // ---- 同侧辐照度场（供末级兜底 r4）：只收 `wn_raw > 0` 且通过精确求交的角，余弦权重用 clamp 后 `wn`。 ----
        if (wn_raw > 0.0 && !occluded) {
          let wc_c = wn * wtri;
          fld = fld + irr4.xyz * wc_c;
          fw = fw + wc_c;
        }
        // `w < ε` ⇒ 丢弃（不计入 total/wsum）。
        if (occluded) {
          ddgi_dbg_occl = ddgi_dbg_occl + 1u;
          continue;
        }
        if (w < DDGI_OCCL_W_MIN) { continue; }
        total = total + irr * w;
        wsum = wsum + w;
      }
    }
  }
  let ok = wsum >= 1e-4;
  let avg = select(vec3<f32>(0.0), total / max(wsum, 1e-6), ok);
  ddgi_dbg_zero = select(0.0, 1.0, ok && max(max(abs(avg.x), abs(avg.y)), abs(avg.z)) < 1e-5);
  ddgi_dbg_cov = cov / max(1.0 - cov_solid, 0.25);
  ddgi_field = fld;
  ddgi_field_w = fw;
  return vec4<f32>(select(vec3<f32>(0.0), total / wsum, ok), wsum);
}
var<private> ddgi_dbg_wsum: f32;
/// 1 = 本次 invoke 是着色（`dda_main`），0 = cast 的 GI 回读（`ddgi_cast`），由调用方在入口设置。
var<private> ddgi_usage_track: u32;
/// 「有可用数据」的角的名义权重覆盖度 Σ wtri（不含 wn/wd 照明闸门），天光兜底的唯一判据。
/// 分母已扣 cov_solid（下限 0.25）；着色侧用它决定是否退回常量天光。
var<private> ddgi_dbg_cov: f32;
// 本点一侧辐照度场 Σ wtri·wn·irr / Σ wtri·wn（只收 `wn_raw > 0` 且过精确求交的角），供末级兜底 r4。
var<private> ddgi_field: vec3<f32>;
var<private> ddgi_field_w: f32;
// 0 = 没有任何 LOD 盒包含该像素；lod+1 = 落在 LOD(lod) 的壳内（纯几何包含）。
var<private> ddgi_dbg_dom: f32;
/// 到最外层级联盒边界的距离，归一到 [0,1]：0 = 盒外/压边界，1 = 深入盒内 ≥ 过渡带宽。
/// 着色侧用它平滑混合「覆盖内的 GI」与「覆盖外的天光兜底」。
var<private> ddgi_dbg_edge: f32;
// 兜底来源（也对应 Probe 档颜色）：0 = 主级成功；1 = 像素不在任何级联盒内（满额 amb_far）；
// 3 = 同级放宽 wn 成功（亮黄）；4 = 背向放开 + 精确求交成功（紫）；5 = 同侧场插值（青绿）。
var<private> ddgi_dbg_fb: f32;
// 1 = 权重和 > 0 但辐照度 ≈ 0（探针在、闸门过、图集从未被写）。
var<private> ddgi_dbg_zero: f32;
// 8 角剔除原因计数：x = 窗口外 / ENABLED=0，y = `wn ≤ 0`（背面），
// z = Chebyshev `wd ≤ 1e-3`，w = 年龄不足 / 深度无记录且覆盖度不足（逐角求交不计这里）。
var<private> ddgi_dbg_rej: vec4<u32>;
var<private> ddgi_dbg_rej_out: vec4<u32>;
// 被逐角线段求交判为遮挡的角数（与 `rej.z` 的 Chebyshev 代理量分开计数）。
var<private> ddgi_dbg_occl: u32;
var<private> ddgi_dbg_occl_out: u32;

/// 级联混合：p 落在混合带内时与相邻 LOD 按 t = 0.5·(1 − d/W) 混合。
/// 外侧（离开更细级盒）用 `clamp_cells=true`；内侧（靠近本级外边界）正常采样；伙伴无数据则退回主级。
fn ddgi_cascade_blend(p: vec3<f32>, n: vec3<f32>, lod: u32, main_c: vec3<f32>) -> vec3<f32> {
  let W = f32(ddgi_u.lods[lod].origin.w) * DDGI_CASCADE_BLEND;
  if (W <= 0.0) { return main_c; }
  // 伙伴会覆写全局诊断量：先存主级的，返回前恢复。
  let dbg_rej = ddgi_dbg_rej_out;
  let dbg_occl = ddgi_dbg_occl_out;
  let dbg_wsum = ddgi_dbg_wsum;
  let dbg_dom = ddgi_dbg_dom;
  let dbg_fb = ddgi_dbg_fb;
  let dbg_cov = ddgi_dbg_cov;
  let dbg_zero = ddgi_dbg_zero;
  var out = main_c;
  if (lod > 0u) {
    let d = -ddgi_box_edge_dist(lod - 1u, p);
    if (d >= 0.0 && d < W) {
      let rf = ddgi_sample_lod(p, n, lod - 1u, true, false, false);
      if (rf.w >= 1e-4) {
        out = mix(main_c, rf.xyz, 0.5 * (1.0 - d / W));
      }
    }
  }
  if (lod + 1u < DDGI_LOD_COUNT) {
    let d = ddgi_box_edge_dist(lod, p);
    if (d >= 0.0 && d < W) {
      let rc = ddgi_sample_lod(p, n, lod + 1u, false, false, false);
      if (rc.w >= 1e-4) {
        out = mix(main_c, rc.xyz, 0.5 * (1.0 - d / W));
      }
    }
  }
  ddgi_dbg_rej_out = dbg_rej;
  ddgi_dbg_occl_out = dbg_occl;
  ddgi_dbg_wsum = dbg_wsum;
  ddgi_dbg_dom = dbg_dom;
  ddgi_dbg_fb = dbg_fb;
  ddgi_dbg_cov = dbg_cov;
  ddgi_dbg_zero = dbg_zero;
  return out;
}

fn ddgi_sample(p: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
  let outer = DDGI_LOD_COUNT - 1u;
  let w_edge = f32(ddgi_u.lods[outer].origin.w) * DDGI_CASCADE_FADE_CELLS;
  ddgi_dbg_edge = clamp(ddgi_box_edge_dist(outer, p) / max(w_edge, 1e-3), 0.0, 1.0);
  var in_cascade = false;
  // 主级那一次带出的「同侧辐照度场」副本，供 r4 用。
  var field_irr_shell = vec3<f32>(0.0);
  var field_w_shell = 0.0;
  // 主级（严格闸门）的覆盖度副本：全失败时决定天光兜底须用它。
  var cov_strict_shell = 0.0;
  for (var lod = 0u; lod < DDGI_LOD_COUNT; lod = lod + 1u) {
    if (ddgi_lod_contains(lod, p)) {
      in_cascade = true;
      let r = ddgi_sample_lod(p, n, lod, false, false, false);
      ddgi_dbg_rej_out = ddgi_dbg_rej;
      ddgi_dbg_occl_out = ddgi_dbg_occl;
      ddgi_dbg_wsum = r.w;
      ddgi_dbg_dom = f32(lod) + 1.0;
      ddgi_dbg_fb = 0.0;
      cov_strict_shell = ddgi_dbg_cov;
      field_irr_shell = ddgi_field;
      field_w_shell = ddgi_field_w;
      if (r.w >= 1e-4) {
        return ddgi_cascade_blend(p, n, lod, r.xyz);
      }
      // r2 = 同一级放宽 wn（下限 0.25）重试（不 clamp、不换级）。
      let rej_shell = ddgi_dbg_rej;
      let occl_shell = ddgi_dbg_occl;
      let r2 = ddgi_sample_lod(p, n, lod, false, true, false);
      if (r2.w >= 1e-4) {
        ddgi_dbg_wsum = r2.w;
        ddgi_dbg_fb = 3.0;
        return ddgi_cascade_blend(p, n, lod, r2.xyz);
      }
      // ---- 末级兜底 r3'：放开 wn 符号闸门，只保留 Chebyshev wd + 逐角线段求交两个精确判据 ----
      let r3 = ddgi_sample_lod(p, n, lod, false, true, true);
      if (r3.w >= 1e-4) {
        ddgi_dbg_wsum = r3.w;
        ddgi_dbg_fb = 4.0;
        return ddgi_cascade_blend(p, n, lod, r3.xyz);
      }
      // 全失败时把 cov / rej / occl 恢复成本级严格路径的值。
      ddgi_dbg_cov = cov_strict_shell;
      ddgi_dbg_rej = rej_shell;
      ddgi_dbg_occl = occl_shell;
      // 嵌套级联下 p 至多落在一级壳内。
      break;
    }
  }
  if (!in_cascade) {
    // 像素不在任何级联盒内：值 0、cov 0 ⇒ 着色侧给满额 `amb_far`（随 `ddgi_dbg_edge` 连续过渡）。
    // `ddgi_dbg_rej` 保持全 0（没跑过 sample_lod）。
    ddgi_dbg_fb = 1.0;
    ddgi_dbg_wsum = 0.0;
    ddgi_dbg_cov = 0.0;
    return vec3<f32>(0.0);
  }
  // ---- 末级兜底 r4：用「本点一侧的探针场插值」（只按同侧余弦权重），并把 cov 置 1。 ----
  // 触发：主级壳内 r1/r2/r3' 全失败，但 `field_w_shell ≥ DDGI_FIELD_MIN_W`。
  if (field_w_shell >= DDGI_FIELD_MIN_W) {
    ddgi_dbg_fb = 5.0;
    ddgi_dbg_wsum = field_w_shell;
    ddgi_dbg_cov = 1.0;
    return field_irr_shell / field_w_shell;
  }
  // 场也拿不到 ⇒ 退回常量天光；`cov` 取本级严格值。
  ddgi_dbg_cov = cov_strict_shell;
  return vec3<f32>(0.0);
}
