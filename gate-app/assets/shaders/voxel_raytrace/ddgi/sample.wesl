import package::bindings::{ddgi_depth, ddgi_irr, ddgi_meta, ddgi_slot_pos, ddgi_u};
import package::ddgi::consts::{DDGI_BIAS_CELLS, DDGI_CASCADE_BLEND, DDGI_CASCADE_FADE_CELLS, DDGI_DEPTH_SOFT_MIN, DDGI_DEPTH_TEXELS, DDGI_IRR_TEXELS, DDGI_LOD_COUNT, DDGI_META_ACTIVE, DDGI_META_ENABLED, DDGI_MIN_SAMPLE_AGE, DDGI_NORMAL_BIAS, DDGI_OCCL_END_BIAS, DDGI_OCCL_W_MIN, DDGI_TEXEL_MIN_COVERAGE};
import package::ddgi::helpers::{ddgi_cell_in_window, ddgi_depth_coord, ddgi_irr_coord, ddgi_meta_age, ddgi_oct_encode, ddgi_origin_cell, ddgi_slot, ddgi_usage_mark};
import package::trace::{make_grid};
import package::world::{world_cfg_full, world_raycast};

fn ddgi_irr_fetch(id: u32, tx: i32, ty: i32) -> vec4<f32> {
  let c = ddgi_irr_coord(id, u32(tx), u32(ty));
  return textureLoad(ddgi_irr, vec2<i32>(vec2<u32>(c.y, c.z)), i32(c.x), 0);
}
/// 返回 vec4(辐照度.xyz, 覆盖度)。**覆盖度** = 该纹素被 collect 写过的程度（每写一次向 1
/// 靠拢）。采样侧用它区分"探针真的活过"与"探针从未被投线" —— 见 `ddgi_sample_lod` 里
/// 「深度没有记录 = 没有遮挡物」那段：只有活着的探针才允许在深度纹素为 0 时按无遮挡处理。
/// 八面体双线性采样的「与探针无关」那一半：4 个纹素坐标 + 两个轴的插值权重。
/// `ddgi_sample_lod` 的 8 角循环里**采样方向 `n` 是循环不变量**（只有 slot 随角变），
/// 故这半边（含 `ddgi_oct_encode` 的除法与分支、floor、clamp）提到循环外算一次即可。
/// 与原来逐角各算一遍是**同一批表达式的同一次求值**，数值逐位不变。
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
  // 上界必须是**本探针自己的**最后一个纹素（DDGI_IRR_TEXELS-1），不能写死 7：
  // 常量从 8 降到 4 之后，`p.x*DDGI_IRR_TEXELS + tx` 里 tx 一旦取到 4 就跨进了
  // **相邻探针**的纹素 0（45° 宽的一个八面体楔形，插值权重可达 0.5）——
  // 于是整个楔形方向取到的是别人的辐照度，在表面上表现为**跟着探针投影走的大三角锯齿**。
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
/// 用已算好的纹素坐标/权重对探针 `id` 取 4 个纹素做双线性。权重与 tap 顺序与改造前一致。
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
// 在原方向最近邻之外再取 2×2 邻域**取平均**：深度图每个纹素是 rpp 条射线估出来的，
// 射线方向逐帧重随机 → 单纹素读数逐帧抖动；4 点平均把噪声压 ~2×，同时抹掉最近邻在
// 纹素边界上的硬跳变。平均后的方差用 std 的平方近似（只用于软遮挡过渡宽度）。
// 全 0（无数据）纹素不参与；四个都无数据才回 0（采样侧按「无数据」剔除）。
fn ddgi_depth_sample(id: u32, d: vec3<f32>) -> vec2<f32> {
  let s = f32(DDGI_DEPTH_TEXELS);
  let e = ddgi_oct_encode(d) * 0.5 + vec2<f32>(0.5);
  let g2 = e * s - vec2<f32>(0.5);
  // 上界必须是**本探针自己的**最后一个纹素（DDGI_DEPTH_TEXELS-1），不能写死 15：
  // 常量从 16 降到 8 之后，`p.x*DDGI_DEPTH_TEXELS + tx` 里 tx 一旦取到 8 就跨进了
  // **相邻探针**的纹素 0 —— 2×2 平均会把这个"别人的深度"掺进来，让那一整个八面体楔形
  // 方向的 mean/std 变成错值 → Chebyshev 的 wd 在这个楔形里整体偏 0 或整体偏 1 →
  // 探针在楔形内外**成片入选/落选**，而归一化平均对集合变化敏感 → 表面上一块块
  // 「跟着探针投影走的三角亮/暗锯齿」。
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

// clamp_cells = true：把越界的世界 cell 号 clamp 进本级窗口（级联之外的退化采样）。
// 级联只覆盖相机周围有限体积；越界时若直接判空，相机上方/远方的表面会整片无 GI。
// clamp 之后用最靠近的可用探针给一个粗粒度估计 —— 空间上仍随位置变化，只是精度粗。
// relax_normal = true：把背向探针闸门 wn 的下限抬到 0.25。**只在像素不在任何级联盒内时**
// 使用；壳内回退到更粗 LOD 时必须为 false，否则隔墙探针会被放进平均 → 漏光。
fn ddgi_sample_lod(
  p: vec3<f32>,
  n: vec3<f32>,
  lod: u32,
  clamp_cells: bool,
  relax_normal: bool,
) -> vec4<f32> {
  let L = ddgi_u.lods[lod];
  let cs = f32(L.origin.w);
  // 主世界网格（逐角几何遮挡用 trace 数据源）。本函数可能被级联混合/兜底调用多次，
  // 在此构造一次即可复用给全部 8 个角，避免每个角各读一次 grid_descs。
  let gw = make_grid(0u);
  // 逐角几何遮挡配置（world_raycast 完整层次 DDA，与主可见性 trace_scene 同一套遍历）。
  // t_min = 0：从探针位置本身起测。探针放置规则（取 cell 内最大空子区域、探针放在该空区
  // 中心）保证探针必位于空区，因此 t > 0 处任何一次命中都是**真遮挡**，不需要起点偏置；
  // 原先的 START_BIAS=0.5 体素起步跳过会漏掉紧贴探针的薄墙（1~2 体素厚）⇒ 该角被误判为
  // 「可见」⇒ 墙上出现零星方块状漏光。
  // max_t 在调用点按「线段长 − END_BIAS」传入，不设距离上限。
  let occl_cfg = world_cfg_full(0.0, 3u);
  // 探针的**标称位置在 cell 中心**（bake 取 center = cmin + cs*0.5；中心是实心时才搬到
  // 最近空叶），所以相邻两个探针位于 (c+0.5)·cs 与 (c+1.5)·cs —— 三线性插值必须在这两者
  // 之间做，即先把 cell 坐标**平移半格**再 floor。
  // 旧版直接 floor((p-o)/cs)：取到的是「包含 p 的 cell 与它 +1 的 cell」，这两个探针整体
  // 比 p 所在区间偏 +半格。后果按轴向最明显：对朝向 -x/-y/-z 的面，当 p 落在 cell 的
  // 前半格时，沿该轴的**两个候选探针都在 p 的正方向一侧** → 8 个角全部落在表面背面 →
  // wn 背向剔除把它们全剔光 → 整块按 cell 分块的无 GI 区域（Probe 品红、GI 全黑），
  // 且与法线正确与否无关（这就是换逐面法线后表现完全不变的原因）。
  // 采样点沿法线再外推一点（见 DDGI_BIAS_CELLS）：让采样点明确落在表面外侧，前后的
  // 判定不落在临界值上。调用处已用 0.5 体素把点从「命中体素中心」推到体素表面。
  let ps = p + n * (cs * DDGI_BIAS_CELLS);
  // 直接用**世界 cell 号**（不再经过窗口原点）：槽位是「世界 cell 对网格维度取模」，
  // 所以采样侧与烘焙/判活侧用的是同一套世界身份。
  let wc_f = ps / cs - vec3<f32>(0.5);
  let wc0 = vec3<i32>(floor(wc_f));
  let fr = wc_f - floor(wc_f);
  let o_cell = ddgi_origin_cell(lod);
  let di = vec3<i32>(L.dims.xyz);
  // clamp 的上界（含端点）= o_cell + di - 1：只依赖 lod，是 8 角循环的不变量 → 提到循环外。
  // （原式 `o_cell + di - vec3<i32>(1)` 逐角重算；两式整数运算逐位等价。）
  let wc_hi = o_cell + di - vec3<i32>(1);
  // 辐照度采样的方向 = 循环不变量 n（只有 slot 随角变）→ 八面体纹素坐标/权重循环外算一次。
  let irr_taps = ddgi_irr_taps(n);
  // 退化 clamp 的域就是本级空间盒（LOD0 的下面的 cell_in_window 还会再叠"chunk 已领段"
  // 的判定 → clamp 落到空 chunk 的 cell 会被剔除，退回更粗一级）。
  var total = vec3<f32>(0.0);
  var wsum = 0.0;
  // 覆盖度：只累加「有可用数据」的角（见下面的分界注释）。与 wsum 的区别是本函数的核心 ——
  var cov = 0.0;
  // 含**背向角**的覆盖度（名义权重和，同样不含 wn/wd 闸门）。背向角现在一律剔除、不进 cov，
  // 但已删除的「末级背向兜底（r3）」那次调用在开关=0 时会把背向角也计入它末尾覆写的
  // ddgi_dbg_cov。为了让删 r3 后的 ddgi_dbg_cov 与当时逐像素一致，这里单独记下，由
  // ddgi_sample 在 r3 原位补回（见那里的说明）。
  var cov_all = 0.0;
  ddgi_dbg_rej = vec4<u32>(0u, 0u, 0u, 0u);
  for (var iz = 0; iz < 2; iz = iz + 1) {
    for (var iy = 0; iy < 2; iy = iy + 1) {
      for (var ix = 0; ix < 2; ix = ix + 1) {
        var wc = wc0 + vec3<i32>(ix, iy, iz);
        if (clamp_cells) {
          wc = clamp(wc, o_cell, wc_hi);
          // LOD0：clamp 到的 cell 可能落在**没领段**的 chunk（空 chunk）→ 与"窗口外"同样剔除。
          // LOD1~3 的 clamp 域就是规则网格，此判定恒真，不改变原行为。
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
        if ((mw & DDGI_META_ENABLED) == 0u) { ddgi_dbg_rej.x = ddgi_dbg_rej.x + 1u; continue; }
        // 【已删闸门】这里原本有一条 `(mw & DDGI_META_ACTIVE) == 0 → 剔除`。
        // 它的本意是"别让图集里没数据的探针以 0 辐照度满权重参与平均"，但 ACTIVE 是
        // **逐帧**量（由 sort 的 near/worklist 每帧重判）⇒ 同一像素的角会在"有数据/没数据"
        // 之间逐帧翻转：Probe 档表现为浅绿(粗级兜住)↔品红(wn 闸门)高频闪，Normal 档里
        // 同源地让 GI 逐帧抖动。而"图集里到底有没有数据"是**持久**的，判据在下面就够：
        // 下面的纹理覆盖度闸门（irr4.w）—— 配 collect 的「首次写入热启动」后，写过一次
        // 该纹素覆盖度就到 1，所以这条持久闸门是可靠的。
        // （ACTIVE 仍由 sort 维护，但它只用来决定"这帧要不要发新射线"，不再当采样闸门。）
        // 年龄太小（刚换过世界 cell / 刚编辑）→ 图集里是旧位置或未覆盖的读数，跳过
        if (ddgi_meta_age(mw) < DDGI_MIN_SAMPLE_AGE) { ddgi_dbg_rej.w = ddgi_dbg_rej.w + 1u; continue; }
        // 插值基 = **晶格名义位置**上的三线性基（用上面的 `fr`），不是探针实际位置上的 tent。
        //
        // 【为什么必须用名义基】tent 的支撑一旦超过半格（旧 DDGI_SHARE_K = 2），越过一个
        // cell 边界时「离开 stencil 的那个角」权重仍有 ~0.5，但它已不再被枚举 → 这一份
        // 贡献被**整体丢弃**。两侧归一化后的结果因此相差 ~0.33·Δirr（相邻探针辐照度之差），
        // 也就是重量场在每个 cell 边界上有一个硬跳变：表现就是**跟着探针晶格走的放射状
        // 三角锯齿，周期 = cell 边长** ⇒ 粗级周期大 2×、离得稍远就格外显眼（wsum 档能直接
        // 看到同一图案，因为那是权重场本身在跳）。
        //
        // 名义三线性基的性质正好相反：8 个角上恒好和为 1，且在 cell 边界处该角的权重恰好
        // 归零 → 权重场连续、无跳变。探针的**实际位置**依旧参与 wn/wd（朝向与可见性必须用
        // 真实几何关系 —— 这正是 Douglas 说的那一步：
        //   "we fetch the probes' offsets within their cells and we compute additional blending
        //    factors based upon which probes have line of sight visibility to the voxel and which
        //    probes are in front of the voxel versus which probes are behind the voxel."
        // 而"哪个值算在哪个格点上"是插值的事，与探针在格内的偏移无关）。
        //
        // 顺带：每个探针拿到的都是它所属角的名义权重（旧的"借针"共享机制与 DDGI_SHARE_K
        // 都已删除，不再需要放大支撑半径）。
        let wl = vec3<f32>(
          select(1.0 - fr.x, fr.x, ix == 1),
          select(1.0 - fr.y, fr.y, iy == 1),
          select(1.0 - fr.z, fr.z, iz == 1),
        );
        let wtri = wl.x * wl.y * wl.z;
        if (wtri <= 1e-6) { continue; }
        // ---- 记账：本探针的数据**本帧会被屏幕用到**（工作集选择的唯一依据）----
        // 位置在"所有几何/状态闸门之后、图集取样之前"：只要这个角是**候选**（探针有数据、
        // 本帧判活、年龄够、三线性权重非零），就算被用到 —— 注意**不要求它过后面那些
        // 可见性闸门**（wn/wd/遮挡）。否则数据坏掉的探针会被永久拒绝 → 永远拿不到射线 →
        // 永远修不好（自锁）。见 DDGI_USAGE_STICKY / ddgi_usage_mark。
        if (ddgi_usage_track != 0u) { ddgi_usage_mark(slot); }
        let probe = ddgi_slot_pos[slot].xyz;
        // 【不去重】基是名义三线性基（8 角和 = 1）：同一份辐照度若落在两个角上，本就该拿
        // 这两个角的权重之和 —— 那正是"两个角的代表是同一个值"的正确插值结果。去重反而会把
        // 它压成其中一个角的基值，等于把该值又摆回了那个角点，晶格伪影会原样回来。
        // （旧版是因为"支撑半径按探针实际位置算 + 借针共享"才需要去重；借针已删，基的语义未变，
        //   去重仍不必要。）
        let to = ps - probe;
        let dist = length(to);
        let dir = to / max(dist, 1e-4);
        // ================= 「这个角的**数据**是否可用」=================
        // 上面过的闸门（窗口 / ENABLED / ACTIVE / age / wtri>0）加上「该方向深度纹素有记录」
        // 和「探针在表面正面（wn > 0）」共同回答：**本点是"有可用数据"还是"根本没有"**。
        // 天光常量兜底必须由这个问题决定（cov），**不能**由 wsum 决定。
        // 两类闸门必须分开对待（这就是 cov 在下面背向闸门之后才累加的原因）：
        //   · wn ≤ 0（探针在表面**背面**）：该探针的数据对本点**无效** → 算「无数据」。
        //   · wd 小（被墙挡住）：是"本点本来就该暗"的正确结果 → **仍算有数据**。
        // 用 wsum 判的后果是室内墙角漏光（wd 被误判成"没数据" → 凹角被补上天光）；
        // 而把 wn 也当成"有数据"的后果是远处整片纯黑（8 角全在背面时兜底不生效、
        // 与周边形成"有 GI / 无 GI"硬边）。
        let dtex = ddgi_depth_sample(slot, dir);
        // 本点方向的辐照度（.w = 该纹素的**覆盖度**：被 collect 写过就 →1）。提前取，用于下面
        // 判断"探针到底活没活"。
        let irr4 = ddgi_irr_sample(slot, irr_taps);
        // 【深度"没有记录" = 没有遮挡物，不是"没有数据"】
        // 深度图集是**逐纹素轮换**写入的：每个探针每帧只写 rpp 个纹素（rpp 还可以低到 1），
        // 所以启动后几十帧里绝大部分纹素仍是 0。旧版把"纹素是 0"直接当成"无数据"剔除该角 →
        // 那些像素**一个角都不剩** → 只剩 0.05 的常量天光 → **整片纯黑**（Probe 档青色）；
        // 且 rpp 越小（探针越多）、越远（再叠加 can_skip_update 的 1/64 节流）收敛越慢，
        // 黑区就越大、越久 —— 与实测完全吻合。
        // 正确语义：探针**活着**（辐照度纹素被写过 → 覆盖度 ≥ 阈值）而该方向没有深度记录
        // ⇒ 那个方向从未命中过任何表面 ⇒ 无遮挡 ⇒ wd = 1（见下面遮挡判定的分支）。
        // 唯一必须保留的剔除是"探针从来没被写过"（覆盖度 ≈ 0）：否则一个空探针会以 0 辐照度
        // 满权重参与平均，把整片拉黑 —— 那正是早期"白↔绿逐帧翻转"的成因（见 collect 注释）。
        if (dtex.x <= 0.0 && irr4.w < DDGI_TEXEL_MIN_COVERAGE) {
          ddgi_dbg_rej.w = ddgi_dbg_rej.w + 1u;
          continue;
        }
        // 注意：**cov 不在这里累加** —— 还要先过下面的背向闸门，见那里的说明。
        // ============ 往下是「数据是否可用」判定 + 照明决策 ============
        // 背向探针闸门：**所有路径一律剔除**（含放宽路径）。放宽（wn 下限 0.25）是为薄几何的
        // **掠射**探针准备的（探针在表面侧方，dot ≈ 0），不是为**背向**探针（探针在表面背后）。
        //
        // 【为什么必须先用未 clamp 的 dot 判定】`clamp(dot, 0, 1)` 把 dot≈0（掠射）与 dot<0
        // （背向）抹成同一个 0，放宽路径的 `max(wn, 0.25)` 于是把背向探针也拉进平均。而墙后/
        // 箱体内部的探针辐照度 ≈0（那里本来无光 —— 例如这个大圆柱是个罐体，背面就是罐内），
        // 一旦入选就把像素压成**暗斑**；哪几个角入选又随 cell 邻域摆动 → 斑块跟着晶格走、
        // 随相机跳变。这正是「GI 模式里也有暗斑」的来源。
        // 【背向兜底（原 r3）已删除】不再有任何路径允许背向角参与加权平均。
        let wn_raw = dot(n, -dir) / DDGI_NORMAL_BIAS;
        if (wn_raw <= 0.0) {
          ddgi_dbg_rej.y = ddgi_dbg_rej.y + 1u;
          // 背向角只计入「含背向角」的覆盖度（复刻删 r3 前开关=0 的 ddgi_dbg_cov），不进 cov。
          cov_all = cov_all + wtri;
          continue;
        }
        let wn = clamp(wn_raw, 0.0, 1.0);
        // 到这里 = 本角数据**对本点可用**（探针在正面 + 该方向有深度记录）→ 计入覆盖度。
        // 【为什么 wn 要排除、wd 不用】wn ≤ 0 = 探针落在表面**背面**，它的数据对本点无效，
        // 属于「没有可用数据」→ 必须让 cov 反映出来：否则 8 角全在背面时 cov 依然满 →
        // 天光兜底不生效 → 整片纯黑、与周边形成硬边（"有 GI / 无 GI"割裂）。
        // 而 wd（被墙挡住）是"本点本来就该暗"的**正确结果**，不是数据缺失 → 仍算有数据；
        // 早期用 wsum 当置信度正是把 wd 也误判成"没数据"，凹角被补天光 → 室内漏光。
        cov = cov + wtri;
        cov_all = cov_all + wtri;
        let wn_w = select(wn, max(wn, 0.25), relax_normal);
        // ---- 遮挡：Chebyshev 软判定（对齐 Majercik / RTXGI）----
        // 旧版是 `clamp((dtex-dist)/bias+1)` + `<=0 剔除` 的**刀锋**判定：dtex 抖一点，
        // 探针就在「入选/落选」之间翻 → 亮区边界伸缩（静止相机也闪）；而要把 bias 调大到
        // 不翻，就必然大面积漏光。Chebyshev 用 mean + 方差：**深度越不确定（std 越大）
        // 过渡越宽**，只有深度一致（真是一堵墙）时才接近硬判定 → 同时兼顾抗漏与抗闪。
        // 【前提】dtex.y 必须真的是"这个 mean 有多不确定"。旧版在 collect 侧存的是**本帧**
        // std，而该纹素每帧平均只摊到 ~0.1 条射线（大多数帧只有一条）→ 本帧 std 恒为 0 →
        // soft 永远塌到 0.02·mean 的刀锋，上面那句"过渡越宽"从来没生效过（静止也闪的根因）。
        // 现在 collect 存的是样本序列的 std（见那里的注释），这里不用改判据。
        // d_vis 减一个小偏置：贴在该深度图记录的那个面上的采样点，不应因浮点误差落到
        // mean 外侧被判成全额遮挡。**这个偏置不能大**：它等价于"允许点比最近表面再近
        // bias"，会在每处贴面/接缝制造一条 bias 宽的漏带（旧值 cs*0.1 = LOD0 3.2 体素，
        // 与深度均值偏大叠加就是可见亮带）。改成 cs*0.05，让 std/Softness 去吸收抖动。
        let d_vis = dist - max(cs * 0.02, dist * 0.005);
        let mean = dtex.x;
        var wd = 1.0;
        // mean == 0 ⇒ 该方向**没有深度记录**（从未命中过表面）⇒ 无遮挡 → wd 保持 1。
        // （能走到这里说明探针本身是活的，见上面的覆盖度判定。）
        if (mean > 0.0 && d_vis > mean) {
          // [诊断] std 项的信任系数（misc.z，默认 1）：拖到 0 则完全忽略 dtex.y、退成
          // 固定 2% 的硬判定；1 = 正常使用。射线绑定深度纹素后 dtex.y 已是「同一方向
          // 深度读数的跨帧噪声」（此前度量的是「20° 锥内几何变化的离散度」，见 cast 注释），
          // 该滑杆保留作 A/B：0 更能压漏光，但过渡带变窄、动态时更易闪。
          let soft = max(dtex.y * ddgi_u.misc.z, DDGI_DEPTH_SOFT_MIN);
          let delta = d_vis - mean;
          wd = (soft * soft) / (soft * soft + delta * delta);
        }
        // wd 极小 = 实质遮挡：保留统计与提前退出（不再做 0/1 硬剔除，避免临界翻转）
        if (wd <= 1e-3) { ddgi_dbg_rej.z = ddgi_dbg_rej.z + 1u; continue; }
        let irr = irr4.xyz;
        // 【旧闸门已删】这里曾经有一道 `max(irr) < 1e-4 → 剔除`，
        // 理由是"已写入的纹素恒 > 0（radiance ≥ albedo·sky·DDGI_CAST_FLOOR）"。但
        // **DDGI_CAST_FLOOR 后来被改成 0**（室内要能压黑），这个前提就不成立了：射线打在
        // 全黑表面 / 暗室里的**已写入**纹素，值本来就是 0 —— 用值阈值会把"真的暗"当成
        // "没数据"剔掉。
        //
        // 为什么它会造成**静态也闪 / 亮区边界伸缩**：纹素值每帧按 age 斜率做 alpha 混合，
        // 而 rpp 只有几条射线 → 值本来就在抖动。值在 1e-4 附近抖动时，该探针会逐帧在
        // "入选/落选"之间翻，而归一化平均（total/wsum）对**集合变化**很敏感（低 wsum 的
        // 像素尤其：1~2 个探针就占满结果）→ 画面闪、亮区边界伸缩。
        // Domain/Probe 档看不出来：它们只看几何归属与闸门直方图，不看这个连续值。
        //
        // "这个方向有没有被采样过"由**深度纹素**负责，但"没有记录"只表示**没有遮挡物**
        // （只有探针本身也是死的才剔除，见上面的覆盖度判定）。真的暗方向本来就该以 ~0 参与平均。
        //
        // ================= 权重阈值：弱权重角直接**丢弃** =================
        // 角的最终权重 w = wtri * wn_w * wd（本计算**不依赖**下面的遮挡求交结果，故先算）。
        // w < ε ⇒ 把该角当作 0 **丢弃**：跳过遮挡检测、不计入 total/wsum。
        // 【正确语义 / 为什么结构上不可能漏光】误差被 ε 死死界定 —— 最坏只是少贡献一点弱光。
        // 注意"丢弃"必须与"不检测"一致：旧门槛的写法是"权重低 ⇒ 不测，但权重**照旧保留**"✗，
        // 那样跳过的角仍以原权重参与 total/wsum 平均 → 等于绕过几何遮挡 → 墙体接缝/薄墙处
        // 的弱权重角照样漏光（这就是旧门槛漏光的根因）。这里"不测 == 不贡献" ✓。
        let w = wtri * wn_w * wd;
        if (w < DDGI_OCCL_W_MIN) { continue; }
        // ---- 追加：逐角几何遮挡（真·线段可见性）----
        // wn/wd 都是代理量（见 DDGI_OCCL_* 常量顶注）→ 这里对**每个角**再补一次
        // 「探针→采样点」线段求交。命中实体 ⇒ 该角权重直接置 0：既不进 total，也不算进
        // wsum（wsum 归一化 → 它就是 wsum 调试图看到的那套权重，用户据此验收）。
        // 【cov 已在上面累加、与本判定无关】被 ε 丢弃的角仍算「有数据」（cov 在 wn 闸门之后、
        // 本权重判定之前累加）→ 不改变 conf/天光兜底，远场亮度/灰度与无条件版一致。
        // 只有通过 ε 阈值的**存活角**才做求交 —— 强权重角才是漏光主因，弱角已由 ε 界定。
        // 求交走完整层次 DDA：max_t = 线段长（无距离上限），薄墙判定与主 trace
        // 一致；超 65536 步安全网才回退保守可见（正常线段远不会触发）。
        // 线段 = ps - probe，与上面的 `to` 是**同一个表达式**（probe / ps 都是不可变 let，其间无改写）
        // → 直接复用 `to` / `dist`（dist = length(to) = length(seg)），省一次 vec3 相减与一次 length。
        // `to / dist` 与原来的 `seg / seg_len` 是同一批操作数的同一次求值 → 逐位相同（注：不是 dir，
        // dir 的分母是 max(dist,1e-4)，在 dist<1e-4 的退化情形与这里刻意不同，保持原样）。
        // dist == 0（探针与采样点重合的退化情形）无方向可言 → 跳过求交、权重不变（防除零）。
        // 背向角与 w < ε 的弱权重角已在上面 `continue` 剔除 → 走到这里的角 wn_w > 0 且 w ≥ ε。
        if (dist > 0.0) {
          // END_BIAS 从 max_t 扣除：停在采样点前，既不自相交也不吃掉贴接收面的薄墙。
          if (world_raycast(gw, probe, to / dist, dist - DDGI_OCCL_END_BIAS, occl_cfg).hit) {
            ddgi_dbg_rej.z = ddgi_dbg_rej.z + 1u;
            continue;
          }
        }
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
  // "要不要退回常量天光"，见 ddgi_dbg_cov 声明处与 ddgi_sample_lod 的分界注释。
  ddgi_dbg_cov = cov;
  ddgi_dbg_cov_all = cov_all;
  return vec4<f32>(select(vec3<f32>(0.0), total / wsum, ok), wsum);
}
var<private> ddgi_dbg_wsum: f32;
/// 1 = 本次 invoke 是**着色**（`dda_main`），0 = cast 的 GI 回读（`ddgi_cast`）——由调用方在
/// 入口设置。只用来决定 `ddgi_sample_lod` 要不要给"屏幕使用"计数记账（见 `ddgi_usage_mark`：
/// cast 里记账会形成"射线多的探针拿到更多射线"的正反馈）。
var<private> ddgi_usage_track: u32;
// 「有可用数据的角」的名义权重覆盖度（Σ wtri，**不含** wn/wd 两道照明闸门）。
// 它是天光常量兜底的唯一判据 —— 见 ddgi_sample_lod 里 cov 的注释。
var<private> ddgi_dbg_cov: f32;
// 同上，但**把背向角也计入**。背向角现在一律剔除、不进 ddgi_dbg_cov；此量仅用于在
// ddgi_sample 的「末级背向兜底（原 r3）」原位复刻删 r3 前开关=0 覆写过的 ddgi_dbg_cov，
// 保证逐像素一致（见 ddgi_sample 壳内分支与 ddgi_sample_lod 的 cov_all 说明）。
var<private> ddgi_dbg_cov_all: f32;
// 0 = 没有任何 LOD 盒包含该像素；lod+1 = 该像素落在这一级的「壳」内。
// 语义是**纯几何包含**，不被采样成功与否影响（旧版在 fallback 里把它清零，导致
// 「越界」和「壳内但无数据」在 Domain 模式里都是绿色，无法区分）。
var<private> ddgi_dbg_dom: f32;
/// 0 = 在最外层级联盒外（或压着边界），1 = 深入盒内 ≥ 过渡带宽。
/// 着色侧用它把"覆盖内的 GI"与"覆盖外的天光兜底"**平滑混合**（见 DDGI_CASCADE_FADE_CELLS）。
var<private> ddgi_dbg_edge: f32;
// 1 = 本像素走了退化 fallback（clamp 到最粗一级）
var<private> ddgi_dbg_fb: f32;
// 1 = 权重和 > 0 但辐照度 ≈ 0（探针在、闸门过、图集空）
var<private> ddgi_dbg_zero: f32;
var<private> ddgi_dbg_rej: vec4<u32>;
var<private> ddgi_dbg_rej_out: vec4<u32>;

/// 级联混合：主级 `lod` 已给出结果 `main_c`，若 p 落在混合带内则与相邻级按权重混合，
/// 消除 LOD 盒边界处「跨过一步就换一套光照」的硬边（见 DDGI_CASCADE_BLEND）。
///
/// 两个方向各自独立判断；**边界两侧都恰好停在 0.5/0.5** → 跨边界连续：
///   ① **外侧**（p 刚离开更细一级 LOD(lod-1) 的盒）：与更细级混合。更细级在盒外，必须
///      `clamp_cells=true` 取边界格的数据；`relax_normal=false` —— **绝不放宽背向闸门**，
///      放宽会把隔墙探针拉进平均（漏光）。
///   ② **内侧**（p 靠近本级盒的外边界）：与更粗一级 LOD(lod+1) 混合。更粗级在 p 处本来
///      就在其盒内（盒严格嵌套）→ 正常采样，不 clamp、不放宽。
/// 权重 t = 0.5·(1 − d/W)：d=0（恰在边界）时 0.5，d ≥ W（出了混合带）时 0。
/// ⇒ 边界内侧 = 0.5·LOD(k) + 0.5·LOD(k+1)，外侧 = 0.5·LOD(k+1) + 0.5·clamp(LOD(k))，
///   两者近似相等 ⇒ 跨边界亮度连续。
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
      let rf = ddgi_sample_lod(p, n, lod - 1u, true, false);
      if (rf.w >= 1e-4) {
        out = mix(main_c, rf.xyz, 0.5 * (1.0 - d / W));
      }
    }
  }
  if (lod + 1u < DDGI_LOD_COUNT) {
    // ② 内侧：p 在 LOD(lod) 盒内（距离为正）且靠近其外边界
    let d = ddgi_box_edge_dist(lod, p);
    if (d >= 0.0 && d < W) {
      let rc = ddgi_sample_lod(p, n, lod + 1u, false, false);
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
  // 必须**无条件**计算 —— 盒外的点采样必然失败，但仍需要一个**连续**的覆盖度信号来决定
  // 补多少天光；旧版靠布尔的 ddgi_dbg_dom 判断，边界处不连续。
  let outer = DDGI_LOD_COUNT - 1u;
  let w_edge = f32(ddgi_u.lods[outer].origin.w) * DDGI_CASCADE_FADE_CELLS;
  ddgi_dbg_edge = clamp(ddgi_box_edge_dist(outer, p) / max(w_edge, 1e-3), 0.0, 1.0);
  var in_cascade = false;
  var shell_lod = DDGI_LOD_COUNT;
  for (var lod = 0u; lod < DDGI_LOD_COUNT; lod = lod + 1u) {
    if (ddgi_lod_contains(lod, p)) {
      in_cascade = true;
      shell_lod = lod;
      let r = ddgi_sample_lod(p, n, lod, false, false);
      // 记账只在**这一次（主级 = 本像素所在壳的 LOD）**里发生：后面还有 ①同级放宽重试
      // （r2，同一批槽位，重复记账无意义）、②级联混合伙伴、③"逐级退粗"兜底 —— 后两者各自
      // 再走 8 个角，且兜底那条会用 `clamp_cells` 把角 clamp 到本级窗口（可能离屏幕很远）。
      // 不关掉的话"被用到"的集合会被稀释成几万个 → `rpp` 上不去（实测 rpp 只到 2）。
      ddgi_usage_track = 0u;
      ddgi_dbg_rej_out = ddgi_dbg_rej;
      ddgi_dbg_wsum = r.w;
      ddgi_dbg_dom = f32(lod) + 1.0;
      ddgi_dbg_fb = 0.0;
      if (r.w >= 1e-4) {
        // 主级成功 → 过一遍级联混合（不在混合带内则原样返回）
        return ddgi_cascade_blend(p, n, lod, r.xyz);
      }
      // 壳内 8 角被 wn 背向闸门**全剔**的最常见来源是薄几何（线缆/薄板/薄边/薄梁）：
      // 贴着细长物件时，最近的空叶都在**侧面**（上面/下面/两侧），dir 与命中面法线近乎
      // 垂直 → wn≈0 → 被 `wn <= 0` 硬剔 → 壳内一个角都不剩。
      // 对这种情形优先用**同一级**放宽 wn（下限 0.25）重试，而不是直接退到粗级 LOD：
      //   · 同级探针只在 32~64cm 外，16×16 深度图在这个尺度上可信 → 即使放宽 wn 也不会
      //     让隔墙探针漏进来；
      //   · 粗级探针在 1.3~2.6m 外、且多为掠射方向 → **超出深度图的角度分辨率** → 掠射
      //     方向上的薄墙在深度图里"看起来"在更远处 → wd 判不出遮挡。而 ddgi_sample 返回的是
      //     **归一化**平均（一个探针入选就占满结果）→ 一个隔墙粗级探针就能把整片点亮成
      //     跟着粗级 cell 走的亮条纹 —— 封闭空间里那些"细条"就是这么来的。
      //   · 仍是同一个 cell 邻域（不 clamp、不换级），位置连续、不产生块状跳变。
      //   · 放宽**只抬掠射探针的权重**（wn_raw > 0 但很小）；wn_raw <= 0 的背向探针在任何
      //     路径下都被剔除 —— 它们是墙后/箱体内部的探针（辐照度≈0），拉进来会把像素压成
      //     「跟着 cell 晶格走、随相机跳变」的暗斑（见 ddgi_sample_lod 的闸门注释）。
      let rej_shell = ddgi_dbg_rej; // 诊断用：保留未放宽时的剔除统计
      let r2 = ddgi_sample_lod(p, n, lod, false, true);
      if (r2.w >= 1e-4) {
        ddgi_dbg_wsum = r2.w;
        ddgi_dbg_fb = 3.0; // 3 = 同级放宽 wn 后成功（Probe 档亮黄）
        return ddgi_cascade_blend(p, n, lod, r2.xyz);
      }
      // ---- 末级背向兜底（原 r3）已删除 ----
      // 原代码在此再调一次 ddgi_sample_lod(..., allow_backface=true)，把「探针在表面背面」的角
      // 也按 |wn| 放进加权平均，只为避免「8 角全背向 → 整片纯黑」。背向探针对漫反射半球本不该
      // 贡献辐照度，现改为**所有路径一律剔除**（见 ddgi_sample_lod 的背向闸门）→ 这里直接落到
      // 下面的「逐级退粗」兜底。
      // 【与删 r3 前「开关=0」逐像素一致】那次调用在开关=0 时背向角权重恒 0、wsum 必然 < 1e-4，
      // 返回值被丢弃；但它末尾的 `ddgi_dbg_cov = cov` 会把**背向角**也算进覆盖度。若不复刻，
      // 这些像素的 cov 会少掉背向角那一份 → conf 变小 → 天光兜底变多（远场发灰）。最近一次调用
      // （上面的 r2）同时算出的 cov_all 正是当时的覆写值，补回它。
      ddgi_dbg_cov = ddgi_dbg_cov_all;
      ddgi_dbg_rej = rej_shell;
      // 嵌套级联下 p 至多落在一级的壳内，无需继续向上试
      break;
    }
  }
  // 壳内 8 个角全被剔除（或像素在级联之外）→ **逐级退到更粗的 LOD 再试**，而不是直接
  // clamp 到最粗一级的边界：
  //   ① 粗一级的探针间距更大、其标称位置与 p 的切平面关系不同，往往还有可用数据；
  //   ② clamp 到最粗一级的边界时，那圈边界探针远离几何、基本判不了活（图集恒 0）——
  //      这正是之前「整片发黑」的来源，而不是壳本身有问题。
  // 每级仍然照常过全部门（未投线的探针照样排除），所以不会因此漏光。
  // ddgi_dbg_dom 不再清零 —— 它如实标出「该像素根本不在任何级联盒内」（绿）。
  var start = 0u;
  if (in_cascade) { start = shell_lod + 1u; }
  // clamp/放宽只留给「不在任何壳内」的退化路径：
  //   ① 壳内回退（in_cascade）：盒子严格嵌套 → p 必然也在更粗级的窗口内，无需 clamp；
  //      且**不放宽 wn** —— 否则背向/隔墙探针也会参与平均，正是漏光的来源之一。
  //   ② 壳外（!in_cascade）：越界 cell 必须 clamp 到最粗一级窗口才取得到数据，并放宽 wn。
  let relax_normal = !in_cascade;
  for (var lod = start; lod < DDGI_LOD_COUNT; lod = lod + 1u) {
    let r = ddgi_sample_lod(p, n, lod, true, relax_normal);
    if (r.w >= 1e-4) {
      if (!in_cascade) {
        // 不在任何壳内：退化结果与剔除统计都归它
        ddgi_dbg_fb = 1.0;
        ddgi_dbg_wsum = r.w;
        ddgi_dbg_rej_out = ddgi_dbg_rej;
      } else {
        // 壳没给出数据，但更粗一级 LOD 兜住了 → GI 有效（只是精度粗）。单独标记，Probe
        // 模式用另一种颜色表示「粗级兜底」，而不是当成异常报警 —— 相机滚动时新进入窗口的
        // 那条带就是这种情况。
        ddgi_dbg_fb = 2.0;
        ddgi_dbg_wsum = r.w;
      }
      return r.xyz;
    }
  }
  // 全部失败：in_cascade 时**保留壳的剔除统计**（不被退化调用覆盖），否则 Probe 模式永远
  // 显示退化路径的直方图（那圈探针本来就全是 near=false），定位不到壳为什么失败。
  if (!in_cascade) {
    ddgi_dbg_fb = 1.0;
    ddgi_dbg_rej_out = ddgi_dbg_rej;
  }
  return vec3<f32>(0.0);
}
