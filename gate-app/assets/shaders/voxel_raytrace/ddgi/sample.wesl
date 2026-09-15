import package::bindings::{ddgi_cell, ddgi_depth, ddgi_irr, ddgi_meta, ddgi_slot_pos, ddgi_u};
import package::ddgi::consts::{DDGI_BIAS_CELLS, DDGI_CASCADE_BLEND, DDGI_CASCADE_FADE_CELLS, DDGI_DEPTH_SOFT_MIN, DDGI_DEPTH_TEXELS, DDGI_FIELD_MIN_W, DDGI_IRR_TEXELS, DDGI_LOD_COUNT, DDGI_META_ACTIVE, DDGI_META_ENABLED, DDGI_MIN_SAMPLE_AGE, DDGI_NORMAL_BIAS, DDGI_OCCL_END_BIAS, DDGI_OCCL_W_MIN, DDGI_TEXEL_MIN_COVERAGE};
import package::ddgi::helpers::{DDGI_REC_OCCUPIED, ddgi_cell_in_window, ddgi_depth_coord, ddgi_irr_coord, ddgi_meta_age, ddgi_oct_encode, ddgi_origin_cell, ddgi_slot, ddgi_usage_mark};
import package::trace::{make_grid};
import package::world::{world_cfg_full, world_raycast};

fn ddgi_irr_fetch(id: u32, tx: i32, ty: i32) -> vec4<f32> {
  let c = ddgi_irr_coord(id, u32(tx), u32(ty));
  return textureLoad(ddgi_irr, vec2<i32>(vec2<u32>(c.y, c.z)), i32(c.x), 0);
}
/// 返回 vec4(辐照度.xyz, 覆盖度)。覆盖度 = 该纹素被 collect 写过的程度（每写一次向 1 靠拢），
/// 用于区分"探针真的活过"与"从未被投线"（见 `ddgi_sample_lod` 里深度无记录的分支）。
/// 八面体双线性采样中「与探针无关」的那一半：4 个纹素坐标 + 两个轴的插值权重。
/// `ddgi_sample_lod` 的 8 角循环里采样方向 `n` 是循环不变量（只有 slot 随角变），故这半边
/// （`ddgi_oct_encode`、floor、clamp）提到循环外算一次即可。
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
  // 上界必须是**本探针自己的**最后一个纹素（DDGI_IRR_TEXELS-1）：tx 一旦取到
  // DDGI_IRR_TEXELS 就跨进了**相邻探针**的纹素 0（宽 45° 的八面体楔形，插值权重可达 0.5）
  // ⇒ 该楔形取到的是别人的辐照度。
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
// 返回 vec2(mean, std)：距离均值与该均值的标准差（图集 .x/.y）。
// 在原方向最近邻之外取 2×2 邻域取平均：单纹素读数逐帧抖动，4 点平均压噪并抹掉纹素边界硬跳变。
// 全 0（无数据）纹素不参与；四个都无数据才回 0（采样侧按「无数据」剔除）。
fn ddgi_depth_sample(id: u32, d: vec3<f32>) -> vec2<f32> {
  let s = f32(DDGI_DEPTH_TEXELS);
  let e = ddgi_oct_encode(d) * 0.5 + vec2<f32>(0.5);
  let g2 = e * s - vec2<f32>(0.5);
  // 上界必须是**本探针自己的**最后一个纹素（DDGI_DEPTH_TEXELS-1）：tx 一旦取到
  // DDGI_DEPTH_TEXELS 就跨进相邻探针的纹素 0，2×2 平均会把"别人的深度"掺进来
  // ⇒ 该八面体楔形的 mean/std 变错值、wd 整体偏 0 或整体偏 1。
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
/// p 到 LOD(lod) 盒**外边界面**的距离：盒内为正、盒外为负（取六个面中最小者）。
/// 级联混合带的判据用它 —— 见 ddgi_cascade_blend。
fn ddgi_box_edge_dist(lod: u32, p: vec3<f32>) -> f32 {
  let L = ddgi_u.lods[lod];
  let lo = vec3<f32>(L.origin.xyz);
  let hi = lo + vec3<f32>(L.dims.xyz) * f32(L.origin.w);
  return min(
    min(min(p.x - lo.x, hi.x - p.x), min(p.y - lo.y, hi.y - p.y)),
    min(p.z - lo.z, hi.z - p.z),
  );
}
// p 是否落在 LOD(lod) 的「壳」内：在 LOD(lod) 盒内、且不在更细一级 LOD 盒内。
// 嵌套级联下任意点恰好属于一级 → **主采样**只有一个 LOD（保证只在探针真正活跃的区域取数）；
// 只有落在混合带内（见 ddgi_cascade_blend）时才额外采相邻一级做过渡。
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

// clamp_cells = true：把越界的世界 cell 号 clamp 进本级窗口。**只剩级联混合的外侧过渡带用它**
// （p 刚离开更细一级的盒 → 那级在 p 处无覆盖，只能取最近的边界格做过渡）。
// relax_normal = true：把背向探针闸门 wn 的下限抬到 0.25 —— 只抬**掠射**角的权重
// （`wn_raw > 0` 这个符号闸门照旧生效）。仅供 r2/r3' 两条重试路径使用。
// allow_backface = true：**完全放开 wn 的符号闸门**（背向探针也参与），只保留两个精确判据
// （Chebyshev wd + 逐角线段求交）来过滤。仅供 `ddgi_sample` 的末级兜底 r3' 使用；r1/r2/
// 级联混合一律 false。
// ⚠️ cov 与 allow_backface 正交：`cov` 在**照明闸门之前**累加（只问"周围有没有探针数据"，
// 不问朝向），所以无论 wn 闸门开不开，返回的 `cov` 都一样。消费侧需知道：`ddgi_sample` 在
// r1/r2/r3' 全失败时会把 `ddgi_dbg_cov` 恢复成本级那一份（`cov_strict`）。
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
  // 主世界网格（逐角几何遮挡用 trace 数据源）。本函数可能被级联混合/兜底调用多次，
  // 在此构造一次即可复用给全部 8 个角，避免每个角各读一次 grid_descs。
  let gw = make_grid(0u);
  // 逐角几何遮挡配置（world_raycast 完整层次 DDA，与主可见性 trace_scene 同一套遍历）。
  // t_min = 0：从探针位置本身起测。探针由放置规则放在空区内（cell 内最大空子区域的中心）⇒
  // t > 0 处任何一次命中都是真遮挡，不需要起点偏置（起步跳过会漏掉紧贴探针的 1~2 体素薄墙）。
  // max_t 在调用点按「线段长 − DDGI_OCCL_END_BIAS」传入，不设距离上限。
  let occl_cfg = world_cfg_full(0.0, 3u);
  // 探针的**标称位置在 cell 中心**（bake 取 center = cmin + cs*0.5；只有中心是实心时才搬到
  // 最近空叶），所以相邻两个探针位于 (c+0.5)·cs 与 (c+1.5)·cs —— 三线性插值必须先把 cell
  // 坐标**平移半格**再 floor。
  // 采样点沿法线再外推一点（见 DDGI_BIAS_CELLS）：让前后判定不落在临界值上；调用处已用 0.5
  // 体素把点从「命中体素中心」推到体素表面。
  let ps = p + n * (cs * DDGI_BIAS_CELLS);
  // 直接用**世界 cell 号**（不再经过窗口原点）：槽位 = 世界 cell 对网格维度取模，
  // 所以采样侧与烘焙/判活侧用的是同一套世界身份。
  let wc_f = ps / cs - vec3<f32>(0.5);
  let wc0 = vec3<i32>(floor(wc_f));
  let fr = wc_f - floor(wc_f);
  let o_cell = ddgi_origin_cell(lod);
  let di = vec3<i32>(L.dims.xyz);
  // clamp 的上界（含端点）= o_cell + di - 1：只依赖 lod，是 8 角循环的不变量。
  let wc_hi = o_cell + di - vec3<i32>(1);
  // 辐照度采样的方向 = 循环不变量 n（只有 slot 随角变）→ 八面体纹素坐标/权重循环外算一次。
  let irr_taps = ddgi_irr_taps(n);
  // 退化 clamp 的域 = 本级空间盒；LOD0 还要再叠 cell_in_window 的"chunk 已领段"判定
  // （clamp 落到空 chunk 的 cell 会被剔除）。
  var total = vec3<f32>(0.0);
  var wsum = 0.0;
  // 覆盖度：只累加"有可用数据"的角（见下面的分界注释），与 wsum 的区别是本函数的核心。
  var cov = 0.0;
  // "该角所在 cell 根本放不下探针"（全实心，或自由空间全是 <4 体素宽的窄缝 → bake 没放探针）
  // 的名义权重和。它是 cov 的**分母修正项**：这类角由几何决定，不代表"本点没有数据"。
  var cov_solid = 0.0;
  // 「本点一侧辐照度场」的两个累加器（见 ddgi_field 的说明）。
  var fld = vec3<f32>(0.0);
  var fw = 0.0;
  ddgi_dbg_rej = vec4<u32>(0u, 0u, 0u, 0u);
  for (var iz = 0; iz < 2; iz = iz + 1) {
    for (var iy = 0; iy < 2; iy = iy + 1) {
      for (var ix = 0; ix < 2; ix = ix + 1) {
        var wc = wc0 + vec3<i32>(ix, iy, iz);
        // 三线性权重先算：它只依赖循环下标与 `fr`（与 `wc` 无关），下面"cell 放不下探针"的
        // 分母修正需要用到它。为什么用晶格名义位置的基而不是探针实际位置上的 tent：
        // 见下面 `wtri <= 1e-6` 处的说明。
        let wl = vec3<f32>(
          select(1.0 - fr.x, fr.x, ix == 1),
          select(1.0 - fr.y, fr.y, iy == 1),
          select(1.0 - fr.z, fr.z, iz == 1),
        );
        let wtri = wl.x * wl.y * wl.z;
        if (clamp_cells) {
          wc = clamp(wc, o_cell, wc_hi);
          // LOD0：clamp 到的 cell 可能落在没领段的 chunk（空 chunk）→ 与"窗口外"同样剔除。
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
          // "这个角没有探针" ≠ "本点没有数据"：cell 放不下探针时 ENABLED=0，原因全是几何的（全实心，
          // 或自由空间只有 <4 体素宽的窄缝），与"本点附近有没有可用数据"无关。把它计进 cov 会让
          // conf 取决于三线性 stencil 恰好有几个角落在墙体内 ⇒ 只从 cov 的**分母**里剔除
          // （见结尾的 `cov / max(1 - cov_solid, 0.25)`），且只有 OCCUPIED=1 才剔除：
          // 纯空气但没放针的角（远离几何的 cell）是真的没数据。
          if ((ddgi_cell[slot] & DDGI_REC_OCCUPIED) != 0u) { cov_solid = cov_solid + wtri; }
          ddgi_dbg_rej.x = ddgi_dbg_rej.x + 1u;
          continue;
        }
        // ACTIVE 是**逐帧**量（sort 每帧重判），只用来决定"本帧要不要发新射线"，不作采样闸门：
        // 用它当闸门会让同一像素的角在"有数据/没数据"之间逐帧翻转。
        // 年龄太小（刚换过世界 cell / 刚编辑）→ 图集里是旧位置或未覆盖的读数，跳过
        if (ddgi_meta_age(mw) < DDGI_MIN_SAMPLE_AGE) { ddgi_dbg_rej.w = ddgi_dbg_rej.w + 1u; continue; }
        // 插值基 = **晶格名义位置**上的三线性基（用上面的 `fr`），不是探针实际位置上的 tent。
        // 名义基在 8 个角上权重和恒为 1，且在 cell 边界处该角权重恰好归零 ⇒ 权重场连续、无跳变
        // （tent 的支撑一旦超过半格，越过 cell 边界时"离开 stencil 的角"权重不归零却被丢弃，
        // 权重场会在每个 cell 边界跳变）。
        // 探针的**实际位置**依旧参与 wn/wd —— 朝向与可见性必须用真实几何关系。
        if (wtri <= 1e-6) { continue; }
        // ---- 记账：本探针的数据**本帧会被屏幕用到**（工作集选择的唯一依据）----
        // 位置在"所有几何/状态闸门之后、图集取样之前"：只要这个角是候选（探针有数据、本帧判活、
        // 年龄够、三线性权重非零）就算被用到，**不要求它过后面那些可见性闸门**（wn/wd/遮挡）——
        // 否则数据坏掉的探针会被永久拒绝 → 永远拿不到射线 → 永远修不好（自锁）。
        // 只记「紧贴本像素的角」（角格离本像素的基础角格 ≤ 2 格）：它们确实会被显示；不记则只靠
        // `due` 零星刷新、`age` 涨不上去（大增益），且 ping-pong 一侧长期陈旧 ⇒ 逐帧跳变。
        if (ddgi_usage_track != 0u && all(abs(wc - wc0) <= vec3<i32>(2))) { ddgi_usage_mark(slot); }
        let probe = ddgi_slot_pos[slot].xyz;
        // 不去重：基是名义三线性基（8 角和 = 1），同一份辐照度若落在两个角上，本就该拿这两个角的
        // 权重之和 —— 那正是"两个角的代表是同一个值"的正确插值结果；去重会把它压成其中一个角的
        // 基值（等于把该值摆回那个角点），晶格伪影会回来。
        let to = ps - probe;
        let dist = length(to);
        let dir = to / max(dist, 1e-4);
        // ================= 「这个角的**数据**是否可用」=================
        // 天光常量兜底（cov）必须由"本点周围有没有数据"决定，**不能**由 wsum 决定
        // （wsum 额外乘了照明闸门，凹角/远处粗级的存活权重天然低）。
        // 两类闸门分开对待：wn ≤ 0（探针在背面）是"该数据对本点照明无效"、不是"没有数据"；
        // wd 小（被墙挡住）是"本点本来就该暗"的正确结果。
        // 用 wsum 判的后果是室内墙角漏光；把 wn ≤ 0 当成"没数据"的后果是远处整片纯黑。
        let dtex = ddgi_depth_sample(slot, dir);
        // 本点方向的辐照度（.w = 该纹素的覆盖度：被 collect 写过就 →1），用于判断"探针到底活没活"。
        let irr4 = ddgi_irr_sample(slot, irr_taps);
        // 深度"没有记录" = 没有遮挡物，不是"没有数据"：深度图集逐纹素轮换写入（每个探针每帧只写
        // rpp 个纹素），所以启动后几十帧里绝大部分纹素仍是 0 —— 把"纹素是 0"当成"无数据"剔除，
        // 那些像素会一个角都不剩。
        // 探针**活着**（辐照度纹素被写过 → 覆盖度 ≥ 阈值）而该方向没有深度记录 ⇒ 无遮挡 ⇒ wd = 1。
        // 唯一必须保留的剔除是"探针从来没被写过"（覆盖度 ≈ 0）：否则空探针会以 0 辐照度满权重
        // 参与平均，把整片拉黑。
        if (dtex.x <= 0.0 && irr4.w < DDGI_TEXEL_MIN_COVERAGE) {
          ddgi_dbg_rej.w = ddgi_dbg_rej.w + 1u;
          continue;
        }
        // ★ 覆盖度在**这里**累加：`cov` 回答的是「本点周围有没有探针数据」，与朝向/可见性无关。
        // 这一行**必须**在所有照明闸门（wn / wd / ε / 线段求交）之前，否则 cov 会变成
        // "表面朝向的函数"（同一堵墙相邻两面一个 conf=1、一个 conf=0 ⇒ 亮度差几十倍、边界硬边）。
        cov = cov + wtri;
        // ============ 往下是「照明是否可用」判定 + 照明决策 ============
        // 背向探针闸门：**所有路径一律剔除**（含放宽路径）。放宽（wn 下限 0.25）是为薄几何的
        // **掠射**探针准备的（探针在表面侧方，dot ≈ 0），不是为**背向**探针（探针在表面背后）。
        // 必须先用未 clamp 的 dot 判定：`clamp(dot, 0, 1)` 把 dot≈0（掠射）与 dot<0（背向）抹成
        // 同一个 0，放宽路径会把墙后/箱体内部的探针（辐照度 ≈ 0）也拉进平均 ⇒ 像素被压成暗斑。
        // 唯一例外 `allow_backface`（末级兜底 r3'）：它放开符号闸门，但保留精确判据
        // （Chebyshev wd + 逐角线段求交），权重仍用地板 0.25。
        let wn_raw = dot(n, -dir) / DDGI_NORMAL_BIAS;
        if (!allow_backface && wn_raw <= 0.0) {
          // 背向角：探针落在表面背面 ⇒ 数据对本点的**照明**无效 ⇒ 不参与加权平均。
          // 这只是照明闸门，不是"数据缺失"：cov 已在上面（照明闸门之前）累加。
          ddgi_dbg_rej.y = ddgi_dbg_rej.y + 1u;
          continue;
        }
        let wn = clamp(wn_raw, 0.0, 1.0);
        // 朝向与遮挡全部交给照明闸门（wn / wd / ε / 线段求交）：它们照常把角排除出 total/wsum，
        // 只是不再借 cov 去关掉天光兜底（cov 只回答"有没有数据"）。
        // 8 角全在背面时由 r2 → r3' → r4 兜住；同侧辐照度场在 `w` 算完后累加（与求交共用结果）。
        let wn_w = select(wn, max(wn, 0.25), relax_normal);
        // ---- 遮挡：Chebyshev 软判定（对齐 Majercik / RTXGI）----
        // wd = soft²/(soft² + delta²)：深度越不确定（std 越大）过渡越宽，只有深度一致（真是一堵墙）
        // 时才接近硬判定。前提是 dtex.y 真的是"这个 mean 有多不确定"（collect 存的是样本序列的
        // std）；刀锋式判定在 dtex 抖动时会让探针在入选/落选之间翻 ⇒ 亮区边界伸缩。
        // d_vis 减一个小偏置：贴在深度记录面上的采样点不因浮点误差被判全额遮挡。偏置不能大 ——
        // 它等价于"允许点比最近表面再近 bias"，会在贴面/接缝留一条漏带。
        let d_vis = dist - max(cs * 0.02, dist * 0.005);
        let mean = dtex.x;
        var wd = 1.0;
        // mean == 0 ⇒ 该方向**没有深度记录**（从未命中过表面）⇒ 无遮挡，wd 保持 1
        // （能走到这里说明探针本身是活的，见上面的覆盖度判定）。
        if (mean > 0.0 && d_vis > mean) {
          // [诊断] std 项的信任系数（misc.z，默认 1）：拖到 0 则忽略 dtex.y、退成固定 2% 的硬判定
          // （更能压漏光，但过渡带变窄、动态时更易闪）；1 = 正常使用。
          let soft = max(dtex.y * ddgi_u.misc.z, DDGI_DEPTH_SOFT_MIN);
          let delta = d_vis - mean;
          wd = (soft * soft) / (soft * soft + delta * delta);
        }
        // wd 极小 = 实质遮挡：跳过该角（不做 0/1 硬剔除，避免临界翻转）。
        if (wd <= 1e-3) { ddgi_dbg_rej.z = ddgi_dbg_rej.z + 1u; continue; }
        let irr = irr4.xyz;
        // 不做"值太小 ⇒ 剔除"的闸门：`DDGI_CAST_FLOOR = 0` 时"已写入但真暗"的纹素值本来就是 0，
        // 按值过滤会把"真的暗"当成"没数据"剔掉（值在阈值附近抖动时探针会逐帧在入选/落选间翻，
        // 而归一化平均对集合变化敏感 ⇒ 画面闪、亮区边界伸缩）。
        // "这个方向有没有被采样过"由**深度纹素**负责，但"没有记录"只表示**没有遮挡物**
        // （只有探针本身也是死的才剔除，见上面的覆盖度判定）。真的暗方向本来就该以 ~0 参与平均。
        //
        // ================= 权重阈值 ε + 逐角几何求交（一次测、两处用）=================
        // 角的最终权重 w = wtri * wn_w * wd（**不依赖**求交结果，故先算）。
        let w = wtri * wn_w * wd;
        // ---- 谁要做求交 ----
        //   ① `w ≥ ε` 的存活角：要进 total/wsum，必须验遮挡；
        //   ② **同侧角**（`wn_raw > 0`）：r4 的同侧场需要知道它是否真看得见本点。
        //   其余角（ε 之下**且**非同侧 —— 只有 r3' 里被放开的背向弱角）跳过求交。
        // ε 之下的同侧角也必须测：`wd`（Chebyshev）与 `ε` 都是代理/阈值判据，而线段求交是
        // **几何事实** —— 代理量可以放宽，事实不能。这也是"细小处漏光"的正解。
        let need_test = w >= DDGI_OCCL_W_MIN || wn_raw > 0.0;
        var occluded = false;
        if (need_test && dist > 0.0) {
          // END_BIAS 从 max_t 扣除：停在采样点前，既不自相交也不吃掉贴接收面的薄墙。
          // 线段 = ps - probe，与上面的 `to` 是**同一个表达式**（其间 probe/ps 都是不可变 let）
          // ⇒ 直接复用 `to`/`dist`。
          // dist == 0（探针与采样点重合的退化情形）无方向可言 → 不求交（防除零）。
          // 求交走完整层次 DDA：max_t = 线段长（无距离上限），薄墙判定与主 trace 一致；
          // 超 65536 步安全网才回退保守可见。
          occluded = world_raycast(gw, probe, to / dist, dist - DDGI_OCCL_END_BIAS, occl_cfg).hit;
        }
        // ---- 同侧辐照度场（供末级兜底 r4）----
        // 只收「探针与法线**同侧**（`wn_raw > 0`）+ **通过精确求交**」的角。余弦权重用 clamp 后的
        // `wn`，于是绕角掠射的角（wn≈0）权重自然趋 0，不会把墙外/邻室的光搬进来。
        // 必须在求交之后累加：r4 要回答"本点周围确实看得见的同侧探针辐照度是多少"，
        // 掺进被求交证明挡住的角 = 把隔墙/凹槽外的亮值搬进来。
        if (wn_raw > 0.0 && !occluded) {
          let wc_c = wn * wtri;
          fld = fld + irr4.xyz * wc_c;
          fw = fw + wc_c;
        }
        // 【ε 的语义】`w < ε` ⇒ 该角当作 0 **丢弃**：不计入 total/wsum（误差被 ε 界定，最坏只是
        // 少贡献一点弱光）。"不贡献"必须与"不检测"一致 —— 若权重低就跳过检测却**照旧保留权重**，
        // 跳过的角仍以原权重参与平均 ⇒ 等于绕过几何遮挡。
        // 现在求交对同侧角一律做了（见上），所以这里只决定"要不要计入 total/wsum"。
        // （诊断：被求交判为遮挡的角一律计入 `ddgi_dbg_rej.z`，含 ε 之下的同侧弱角。）
        if (occluded) {
          ddgi_dbg_rej.z = ddgi_dbg_rej.z + 1u;
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
  // 权重和 > 0 但辐照度 ≈ 0 → 探针存在、过了全部闸门，只是图集纹素从未被 collect 写过。
  // 必须与「被剔除」区分开：两者的着色结果都是黑，但根因完全不同。
  ddgi_dbg_zero = select(0.0, 1.0, ok && max(max(abs(avg.x), abs(avg.y)), abs(avg.z)) < 1e-5);
  // 覆盖度单独带出（返回值只有 4 个分量，且 .w 已被 wsum 占用）—— 着色侧用它决定
  // "要不要退回常量天光"。
  // 【分母修正】剔除"cell 放不下探针"的角（见上面 ENABLED 处的推导）：cov 的语义是
  // "在**可能**有数据的角里，有多少真的有数据"，而不是"8 个角里有多少个落在空气里"。
  // 下限 0.25 防极端（只剩 1~2 个可放针的角时，别把单个角的存在放大成"满覆盖"）。
  // 恒有 cov ≤ 1 - cov_solid ⇒ 商 ≤ 1，不需要再 clamp 上界。
  ddgi_dbg_cov = cov / max(1.0 - cov_solid, 0.25);
  // 同侧辐照度场（与可见性无关）也单独带出 —— 供 ddgi_sample 的末级兜底 r4 用，
  // 见 ddgi_field 的说明。
  ddgi_field = fld;
  ddgi_field_w = fw;
  return vec4<f32>(select(vec3<f32>(0.0), total / wsum, ok), wsum);
}
var<private> ddgi_dbg_wsum: f32;
/// 1 = 本次 invoke 是**着色**（`dda_main`），0 = cast 的 GI 回读（`ddgi_cast`）——由调用方在
/// 入口设置。只用来决定 `ddgi_sample_lod` 要不要给"屏幕使用"计数记账（cast 里记账会形成
/// "射线多的探针拿到更多射线"的正反馈）。
var<private> ddgi_usage_track: u32;
// 「有可用数据的角」的名义权重覆盖度（Σ wtri，不含 wn/wd 两道照明闸门），天光常量兜底的
// 唯一判据（见 ddgi_sample_lod 里 cov 的注释）。
// 末级全失败时 ddgi_sample 会把它恢复成主级那一份严格值 cov_strict，让"8 角全在背面"的像素
// 退回常量天光，而不是掉进"GI=0 且 conf=1 ⇒ 纯黑点"。分母已修正：全实心 / 窄缝的角不计入分母。
var<private> ddgi_dbg_cov: f32;
// 「与可见性无关」的**本点一侧辐照度场**插值结果 —— Σ wtri·wn·irr 与 Σ wtri·wn，
// 只累加 `wn_raw > 0`（探针与法线**同侧**）且通过精确求交的角。由每次 `ddgi_sample_lod` 末尾
// 发布；`ddgi_sample` 在主级那一次之后立刻存副本，供"所有 LOD、所有路径都没给出值"的像素做
// 末级兜底 r4。
var<private> ddgi_field: vec3<f32>;
var<private> ddgi_field_w: f32;
// 0 = 没有任何 LOD 盒包含该像素；lod+1 = 该像素落在这一级的「壳」内。
// 语义是**纯几何包含**，不被采样成功与否影响。
var<private> ddgi_dbg_dom: f32;
/// 0 = 在最外层级联盒外（或压着边界），1 = 深入盒内 ≥ 过渡带宽。
/// 着色侧用它把"覆盖内的 GI"与"覆盖外的天光兜底"**平滑混合**（见 DDGI_CASCADE_FADE_CELLS）。
var<private> ddgi_dbg_edge: f32;
// 1 = 像素不在任何级联盒内（无探针身份：值恒 0、cov = 0 ⇒ 着色侧给满额 amb_far）
// 3 = 同级放宽 wn 后成功（亮黄）；4 = 背向放开 + 精确求交后成功（紫）；5 = 同侧场插值（青绿）
var<private> ddgi_dbg_fb: f32;
// 1 = 权重和 > 0 但辐照度 ≈ 0（探针在、闸门过、图集空）
var<private> ddgi_dbg_zero: f32;
var<private> ddgi_dbg_rej: vec4<u32>;
var<private> ddgi_dbg_rej_out: vec4<u32>;

/// 级联混合：主级 `lod` 已给出结果 `main_c`，若 p 落在混合带内则与相邻级按权重混合，
/// 消除 LOD 盒边界处「跨过一步就换一套光照」的硬边（见 DDGI_CASCADE_BLEND）。
///
/// 两个方向各自独立判断；**边界两侧都恰好停在 0.5/0.5** → 跨边界连续：
///   ① **外侧**（p 刚离开更细一级 LOD(lod-1) 的盒）：与更细级混合，必须 `clamp_cells=true`
///      取边界格的数据；**绝不放宽背向闸门**（放宽会把隔墙探针拉进平均 ⇒ 漏光）。
///   ② **内侧**（p 靠近本级盒的外边界）：与更粗一级 LOD(lod+1) 混合；盒严格嵌套 ⇒ p 本就在
///      其盒内，正常采样、不 clamp、不放宽。
/// 权重 t = 0.5·(1 − d/W)：d=0（恰在边界）时 0.5，d ≥ W（出了混合带）时 0。
/// 伙伴取不到数据（wsum < 1e-4）时退回纯主级 —— 混合只负责平滑，不改变「谁有数据」。
fn ddgi_cascade_blend(p: vec3<f32>, n: vec3<f32>, lod: u32, main_c: vec3<f32>) -> vec3<f32> {
  let W = f32(ddgi_u.lods[lod].origin.w) * DDGI_CASCADE_BLEND;
  if (W <= 0.0) { return main_c; }
  // 混合伙伴会覆写这批全局诊断量 → 先存主级的，返回前恢复（Probe/Domain 档才不被搅乱）
  let dbg_rej = ddgi_dbg_rej_out;
  let dbg_wsum = ddgi_dbg_wsum;
  let dbg_dom = ddgi_dbg_dom;
  let dbg_fb = ddgi_dbg_fb;
  let dbg_cov = ddgi_dbg_cov;
  let dbg_zero = ddgi_dbg_zero;
  var out = main_c;
  if (lod > 0u) {
    // ① 外侧：p 在 LOD(lod-1) 盒外（距离为正）且落在混合带内
    let d = -ddgi_box_edge_dist(lod - 1u, p);
    if (d >= 0.0 && d < W) {
      let rf = ddgi_sample_lod(p, n, lod - 1u, true, false, false);
      if (rf.w >= 1e-4) {
        out = mix(main_c, rf.xyz, 0.5 * (1.0 - d / W));
      }
    }
  }
  if (lod + 1u < DDGI_LOD_COUNT) {
    // ② 内侧：p 在 LOD(lod) 盒内（距离为正）且靠近其外边界
    let d = ddgi_box_edge_dist(lod, p);
    if (d >= 0.0 && d < W) {
      let rc = ddgi_sample_lod(p, n, lod + 1u, false, false, false);
      if (rc.w >= 1e-4) {
        out = mix(main_c, rc.xyz, 0.5 * (1.0 - d / W));
      }
    }
  }
  ddgi_dbg_rej_out = dbg_rej;
  ddgi_dbg_wsum = dbg_wsum;
  ddgi_dbg_dom = dbg_dom;
  ddgi_dbg_fb = dbg_fb;
  ddgi_dbg_cov = dbg_cov;
  ddgi_dbg_zero = dbg_zero;
  return out;
}

fn ddgi_sample(p: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
  // 到**最外层**级联盒边界的距离，归一到 [0,1]：0 = 盒外/压边界，1 = 深入盒内。
  // 着色侧用它平滑混合"覆盖内 GI ↔ 覆盖外天光兜底"（见 DDGI_CASCADE_FADE_CELLS）。
  // 必须**无条件**计算 —— 盒外的点采样必然失败，但仍需要一个**连续**的信号来决定补多少天光。
  let outer = DDGI_LOD_COUNT - 1u;
  let w_edge = f32(ddgi_u.lods[outer].origin.w) * DDGI_CASCADE_FADE_CELLS;
  ddgi_dbg_edge = clamp(ddgi_box_edge_dist(outer, p) / max(w_edge, 1e-3), 0.0, 1.0);
  var in_cascade = false;
  // 主级（壳）那一次采样带出的「同侧辐照度场」副本 —— 末级兜底 r4 用它（见函数末尾）。
  // 必须在循环外持有：r4 的判定在壳循环之外，而 r2/r3'/级联混合都会重新发布 ddgi_field。
  var field_irr_shell = vec3<f32>(0.0);
  var field_w_shell = 0.0;
  // 主级（严格闸门）算出的覆盖度副本 —— 同样必须在循环外持有：后面 r2/r3'/级联都会覆写
  // `ddgi_dbg_cov`，而"全失败"时决定天光兜底的那份覆盖度必须来自主级。
  var cov_strict_shell = 0.0;
  for (var lod = 0u; lod < DDGI_LOD_COUNT; lod = lod + 1u) {
    if (ddgi_lod_contains(lod, p)) {
      in_cascade = true;
      let r = ddgi_sample_lod(p, n, lod, false, false, false);
      // 记账在下面所有能给出显示值的路径里都要发生（同级放宽 r2 / 背向放开 r3' / 级联混合伙伴 /
      // 末级同侧场兜底 r4）—— 它们给出的值都会被显示。
      // `ddgi_sample_lod` 内部只记紧贴像素的角（≤2 格），不会把远处角落拉进工作集。
      ddgi_dbg_rej_out = ddgi_dbg_rej;
      ddgi_dbg_wsum = r.w;
      ddgi_dbg_dom = f32(lod) + 1.0;
      ddgi_dbg_fb = 0.0;
      // 主级（严格闸门）算出的覆盖度 —— 后面 r2/r3' 会覆写 `ddgi_dbg_cov`，所以这里先存下来：
      // 末级全失败时要用**它**决定天光兜底。
      cov_strict_shell = ddgi_dbg_cov;
      // 同时存下主级带出的「同侧辐照度场」（r2/r3' 会重新发布，见 ddgi_field 说明）。
      field_irr_shell = ddgi_field;
      field_w_shell = ddgi_field_w;
      if (r.w >= 1e-4) {
        // 主级成功 → 过一遍级联混合（不在混合带内则原样返回）
        return ddgi_cascade_blend(p, n, lod, r.xyz);
      }
      // 壳内 8 角被 wn 背向闸门**全剔**的最常见来源是薄几何（线缆/薄板/薄边/薄梁）：贴着细长
      // 物件时，最近的空叶都在**侧面**（上面/下面/两侧），dir 与命中面法线近乎垂直 → wn≈0 →
      // 被 `wn <= 0` 硬剔 → 壳内一个角都不剩。
      // 这种情形优先用**同一级**放宽 wn（下限 0.25）重试，而不是直接退到粗级 LOD：
      //   · 同级探针只在 32~64cm 外，深度图在这个尺度可信 ⇒ 放宽也不会让隔墙探针漏进来；
      //   · 粗级探针在 1.3~2.6m 外且多为掠射方向 → **超出深度图的角度分辨率** → wd 判不出
      //     遮挡，而归一化平均下一个入选探针就能把整片点亮成跟着粗级 cell 走的亮条纹；
      //   · 仍是同一个 cell 邻域（不 clamp、不换级），位置连续、不产生块状跳变。
      // 放宽只抬掠射角（wn_raw > 0 但很小）的权重；wn_raw <= 0 的背向探针在任何路径下都被剔除。
      let rej_shell = ddgi_dbg_rej; // 诊断用：保留未放宽时的剔除统计
      let r2 = ddgi_sample_lod(p, n, lod, false, true, false);
      if (r2.w >= 1e-4) {
        ddgi_dbg_wsum = r2.w;
        ddgi_dbg_fb = 3.0; // 3 = 同级放宽 wn 后成功（Probe 档亮黄）
        return ddgi_cascade_blend(p, n, lod, r2.xyz);
      }
      // ---- 末级兜底 r3'：**背向闸门放开 + 只留精确判据**（治"纯黑点"）----
      // r1/r2 都失败时像素会在 Probe 档显示品红（= 8 个角全被 wn 背向闸门剔掉），它缺的是一个
      // **能用的探针**，于是这里再给一次机会。
      // r3' 把 wn 的**符号闸门**整个放开（allow_backface=true），只保留两个**精确**判据：
      //   ① Chebyshev（wd，该探针自己的深度图）② 逐角线段求交（探针→本点，几何精确）。
      // 于是"探针在表面背面、但线段上完全没有几何"的角能给出有物理依据的值（薄几何两侧、掠射、
      // 法线退化处）；被墙/箱体真正挡住的背向角会被线段求交剔掉。权重用 r2 的地板 0.25。
      // r1/r2/级联混合一律保持严格 ⇒ 正常像素一字不变。
      let r3 = ddgi_sample_lod(p, n, lod, false, true, true);
      if (r3.w >= 1e-4) {
        ddgi_dbg_wsum = r3.w;
        ddgi_dbg_fb = 4.0; // 4 = 背向放开后成功（Probe 档紫）
        return ddgi_cascade_blend(p, n, lod, r3.xyz);
      }
      // 【覆盖度用**本级的严格值**】`ddgi_dbg_cov = cov_strict_shell` 保证"全失败"的像素按
      // "本级壳到底有没有可用数据"决定天光兜底 —— 不能留用放宽/放开那两次调用写的 cov。
      // `rej_shell` 同理由：Probe 档要看的是**本级严格路径为什么失败**，不是放宽后的统计。
      ddgi_dbg_cov = cov_strict_shell;
      ddgi_dbg_rej = rej_shell;
      // 嵌套级联下 p 至多落在一级的壳内，无需继续向上试
      break;
    }
  }
  if (!in_cascade) {
    // 像素不在任何级联盒内（`ddgi_dbg_dom == 0`）：本点没有任何探针身份。
    // 值返回 0、`cov` 置 0 ⇒ conf = 0 ⇒ 着色侧给出**满额 `amb_far`** —— 这正是「覆盖外天光
    // 兜底」的设计语义，且随 `ddgi_dbg_edge` 连续过渡（见 DDGI_CASCADE_FADE_CELLS）。
    // `ddgi_dbg_rej` 保持全 0（没有跑过任何 sample_lod）→ 直方图显示灰 = "没跑过"，不会误导。
    ddgi_dbg_fb = 1.0;
    ddgi_dbg_wsum = 0.0;
    ddgi_dbg_cov = 0.0;
    return vec3<f32>(0.0);
  }
  // ---- 末级兜底 r4：**本点一侧的探针场插值** ----
  // 走到这里 = 本级壳的 r1/r2/r3' **全失败**。若主级还有"同侧且有数据"的角（`field_w_shell`
  // 够大，见 DDGI_FIELD_MIN_W），说明本点周围**确实有代表它的探针** —— 只是那些角全都过不了
  // 严格判据（`wd` 判为被挡 / 权重阈值 ε / 恰被半遮挡），这正是"犄角旮旯"里的配置：表现为
  // **Probe 档品红 + 纯黑点**（GI=0 而 cov 已满 ⇒ conf=1 ⇒ 常量天光也不生效）。
  // 这里改用只按同侧余弦权重的同级插值，量级与周边一致。
  // 【不漏光】① 只累加 `wn_raw > 0`（探针与法线**同侧**）的角 —— 8 角全在背面时
  // `field_w_shell = 0`，本分支不生效，仍走"cov=0 ⇒ 常量天光"；② 权重用余弦 `wn`，
  // 绕角掠射的角权重趋 0，再由 DDGI_FIELD_MIN_W 挡住"只有掠射角"的情形；
  // ③ **场的分子只收通过精确线段求交的角** —— 否则"同侧、权重不小、却被墙挡住"的角（细缝/
  // 凹槽/薄板两侧）也会进分子，把光塞回本该暗的体素。
  // 【conf】这里同时把 `ddgi_dbg_cov` 置 1：已经拿到"本点一侧的场"= 有数据，
  // 再叠常量天光会重复计入天空那一份（见 cov 的语义）。
  if (field_w_shell >= DDGI_FIELD_MIN_W) {
    ddgi_dbg_fb = 5.0; // 5 = 同侧场插值兜底（Probe 档青绿）
    ddgi_dbg_wsum = field_w_shell;
    ddgi_dbg_cov = 1.0;
    return field_irr_shell / field_w_shell;
  }
  // 连"本点一侧的场"都拿不到（`field_w_shell` 太小 = 只有绕角掠射的角，或 8 角全在背面）
  // ⇒ 退回常量天光。这里把 `ddgi_dbg_cov` 定成**本级的严格值**：conf =
  // smoothstep(DDGI_CONF_LO, DDGI_CONF_HI, 它) 于是给出连续的兜底强度。
  ddgi_dbg_cov = cov_strict_shell;
  return vec3<f32>(0.0);
}
