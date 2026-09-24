import package::bindings::{gi_den_cfg, gi_den_phi, gi_dn_dst, gi_dn_src, gi_dn_tmp, gi_guide, gi_guide_out, gi_hist_cur, gi_hist_prev, gi_raw_tex};
import package::gi::common::{gi_hist_search_offset, gi_pack_dir, gi_unpack_dir};
import package::gi::consts::{GI_DEN_CLAMP_K, GI_DEN_G_KEY0, GI_DEN_G_KEY1, GI_DEN_G_NRM, GI_DEN_G_PREV, GI_DEN_GUIDE_WORDS, GI_DEN_H_COL, GI_DEN_H_KEY0, GI_DEN_H_KEY1, GI_DEN_H_M, GI_DEN_H_M1, GI_DEN_H_M2, GI_DEN_H_NRM, GI_DEN_HIST_WORDS, GI_DEN_LUMA_FLOOR_REL, GI_DEN_LUMA_K, GI_DEN_M_MAX, GI_DEN_NBR_R, GI_DEN_N_DOT, GI_DEN_SIGMA_ALPHA, GI_DEN_STAB_FRAMES, GI_DEN_STAB_MAX, GI_DEN_W_HIST_MIN, GI_HIST_SEARCH_TAPS, GI_LUMA};

// ============================================================================
// GI 降噪（SVGF 家族两段）：**时域累积**（`gi_den_temporal`）+ **迭代 atrous 空间滤波**
// （`gi_den_atrous`，步长 1/2/4/8/16 = NRD/RELAX 的 5 轮；实际轮数 = `GI_DEN_ATROUS_ITER`）。
//
// 与 NRD/RELAX/FidelityFX 的对照（每一项都对着行业里成熟的处方）：
//   · ✅ **离群抑制**（NRD 的 pre-blur pass）：不单开一趟，并进时域 pass —— 对输入做一次
//        「同面邻域 mean ± K·σ」钳制（见 ③′）。缺 pre-blur 的代价只是省了一趟 dispatch。
//   · ✅ **累积矩**（NRD 的 m1/m2）：历史里存的是**按与色相同权重累积**的亮度一/二阶矩，
//        `σ_hist = sqrt(M2 − M1²)` 给出时间上平滑的噪声水平，与本帧观测 σ 取 max 后用于
//        时域权重与 φ ⇒ 权重不再逐帧抖。
//   · ✅ **history-fix**（`GI_HIST_SEARCH_TAPS`）：NRD 在重投影落点邻域里找一条可用的历史；
//        这里做的是同一件事 —— 判据（面键 + 着色法线）**不变**，只是落点不通过时继续在它的
//        3×3 邻域里按同一套判据找。为什么必须有：判据不通过 ⇒ `M = 1`、历史权重 0 ⇒ 本帧原始估计
//        直接上屏，而"落点那个 texel 采到了相邻台阶"在斜面上是常态（一个体素面常常只占一个 texel）。
//   · ✅ **短历史用模糊兜**（`GI_DEN_STAB_FRAMES` / `GI_DEN_STAB_MAX`）：NRD 的 `blurRadiusScale`
//        / FidelityFX 的「低时域样本数 ⇒ 加大空间滤波」—— `M` 小的像素把 atrous 步长放大，
//        用宽模糊代替生噪点（NRD 原话："If history gets discarded we see blur instead of noise"）。
//        **接受与否仍是二值的**，代价由此项补偿，而不是靠一段被信过的短历史。
//   · ✅ **限制历史长度**（`GI_DEN_M_MAX` = 32 帧，NRD `maxAccumulatedFrameNum` 量级）：
//        τ 就是「光照 / 远处几何渐变」的响应时间。
//   · ✅ **变化即重置**（NRD 的 `accumulation speed = 1.0 if a disocclusion is detected`）：
//        由 **reservoir 侧**承担（`gi/screen.wesl` 文件头 ⑥ —— 世界一变就整帧不复用历史，
//        因为那里的判据只能证明"上帧那个面还在"）。**降噪侧不重置**：它的 ⑤ AABB 钳制会把历史
//        拉回本帧邻域的范围，重置只会让编辑帧露出未平滑的输入、并让短历史稳定化整屏生效一帧
//        （症状：放置体素时 GI 闪一下）。
//   · ✅ **宽核统计**（`GI_DEN_NBR_R` = 2 ⇒ 5×5）：钳制与本帧输入都要求邻域非空，
//        而斜面上邻域常常只剩它自己；FidelityFX 为此用 17×17（可分离两趟）取矩。
//
// 输入 / 输出（GI 网格，尺寸 = 渲染分辨率 / gi_div：1/1、1/2 或 1/4）：
//   · `gi_raw_tex`（= `gi_out` 的采样视图）：本帧原始 GI，rgb = gi·valid、a = valid；
//   · 导引 `gi_guide_out`/`gi_guide`：`gi_main` 写（`dda_main` 也读它做几何感知上采样），
//     只有**面键（2 word；有效 = 键非零，没有独立有效位）** 与**上帧重投影像素坐标**
//     （1，`gi_u.prev_view_proj` 投出来的；`0xFFFFFFFF` = 无）—— 所有「同不同面」的判定
//     都靠位域比较，不存世界位置 / 法线 / 容差（见 `GI_DEN_GUIDE_WORDS` 的说明）；
//   · 历史 `gi_hist_prev`/`gi_hist_cur`（双缓冲，每像素 `GI_DEN_HIST_WORDS` 字）；
//   · φ（每像素一个字）= `GI_DEN_LUMA_K` × max(噪声尺度 σ_use, 相对下限)，供 atrous 的亮度 range 权重；
//   · 输出：`gi_dn_tmp`（时域结果）→ atrous 5 轮 ping-pong → `gi_den`（`dda_main` 采样）。
//     atrous 的输入是**采样视图**、输出是**存储视图**，Rust 每轮换绑同一份 layout 的两个号
//     ⇒ 一个入口跑任意轮数（见 `bindings.wesl` 的 14/15）。
//
// 三条硬约束怎么保证：
//  ① **不引入 ramp**：历史失效（重投影出屏 / 不同平面 / 上帧 M = 0）时**不**退回 0 也不沿用旧值，
//     而是直接用本帧原始估计（`M = 1`）。本帧原始估计与稳态估计是同一个积分的无偏估计
//     ⇒ 亮度不随时间爬升，第 1 帧 ≈ 稳态（差别只有噪声，靠空间段压）。
//  ② **不跨薄板 / 墙缝**：唯一的判据是 `gi_den_same_plane`（面号 + 物体相等、沿面法线轴的
//     体素坐标相等；「键逐位相等」是它的特例）。薄板两侧面号不同 / 法线轴坐标差 1 ⇒ 精确拒掉，
//     与缩放无关、与**着色法线**无关（`voxel_normal` 在凸棱上会给出斜的梯度法线，任何用它的
//     几何判据在那里都会静默变味 —— 所以这里一律不用法线）。
//  ③ **逐体素平坦风味**：atrous 里**同一面键的 tap 给满权重 1**（同面内被强拉平），
//     同平面但不同体素面才走亮度权重 ⇒ 同一个面里的像素趋于同值，而不是被大核糊成渐变。
// ============================================================================

/// 导引 [word 址]：本像素是否有效（首命中不透明体素）。
/// 判据 = **键非零** —— 切**不要**改成「读某个有效位」：`GI_DEN_G_KEY1` 里 face 占 13..15
/// 三个 bit，`face = 4 (-z)` 的 0x8000 会与 bit15 撞车（见 `gi/consts.wesl` 的说明）。
fn gi_den_valid_b(b: u32) -> bool {
  return (gi_guide[b + GI_DEN_G_KEY0] | gi_guide[b + GI_DEN_G_KEY1]) != 0u;
}

/// 导引 [word 址] → 面键 word0。
fn gi_den_key0_b(b: u32) -> u32 {
  return gi_guide[b + GI_DEN_G_KEY0];
}

/// 导引 [word 址] → 面键 word1（**原样**返回，不抹任何位 ⇒ 与 `gi_key1` 的输出可直接逐位比较）。
fn gi_den_key1_b(b: u32) -> u32 {
  return gi_guide[b + GI_DEN_G_KEY1];
}

/// 导引 [word 址] → 着色法线（八面体解包）。`voxel_normal` 的梯度法线在凸棱上是斜的，
/// 这里就是那条判据用的「着色法线」。
fn gi_den_nrm_b(b: u32) -> vec3<f32> {
  return gi_unpack_dir(gi_guide[b + GI_DEN_G_NRM]);
}

/// 写导引（`gi_ss_main` 调用）。
/// `valid = false` ⇒ 整格清零（天空像素：两段降噪 + `dda_main` 的上采样都跳过）。
/// `n` = **着色法线**（`voxel_normal` 的结果）—— 在 `gi_main` 里算一次存下来，后面所有判据都读它，
/// **不再重算**（那个梯度要探 6 个邻域体素）。
/// `prev_px` / `prev_ok` = **真实命中点**（`h.point`）用 `gi_u.prev_view_proj` 投到**上帧** GI 网格的
/// 像素坐标。必须用命中点而不是体素中心：静止相机下前者正好映回本像素，后者会偏 1~2 像素 ⇒
/// 约 1/3 的像素每帧都因「上帧那个像素不是同一个面」而丢掉历史（实测接受率只有 68%）。
fn gi_den_guide_store(
  coord: vec2<i32>,
  size: vec2<u32>,
  valid: bool,
  k0: u32,
  k1: u32,
  n: vec3<f32>,
  prev_ok: bool,
  prev_px: vec2<i32>,
) {
  let idx = u32(coord.y) * size.x + u32(coord.x);
  let b = idx * GI_DEN_GUIDE_WORDS;
  if (!valid) {
    for (var i = 0u; i < GI_DEN_GUIDE_WORDS; i = i + 1u) { gi_guide_out[b + i] = 0u; }
    return;
  }
  gi_guide_out[b + GI_DEN_G_KEY0] = k0;
  gi_guide_out[b + GI_DEN_G_KEY1] = k1;
  gi_guide_out[b + GI_DEN_G_NRM] = gi_pack_dir(n);
  // 上帧像素坐标（GI 网格，u16 足够：渲染上限 4096）：x | y<<16；无 = 0xFFFFFFFF。
  var code = 0xFFFFFFFFu;
  if (prev_ok) { code = u32(prev_px.x) | (u32(prev_px.y) << 16u); }
  gi_guide_out[b + GI_DEN_G_PREV] = code;
}

/// **平面归属判据**（纯位域比较，不看任何浮点）：三条全等号 ——
///   1. `face` 相等 —— face 是**射线进入的轴向面**（DDA 直接给出，0..5 = −x/+x/−y/+y/−z/+z）
///      ⇒ 相等即「同一轴向、同一朝向」；
///   2. `obj+1` 相等 —— 同一个 volume；
///   3. **沿面法线轴的那一个体素坐标相等**（`axis = face >> 1`）。
/// 为什么这就够：面 `f = 2a(+轴)` 的平面是 `p[a] = v[a] + 1`、`f = 2a+1(−轴)` 是 `p[a] = v[a]`
/// ⇒ 两个**同面号**体素共面 ⟺ `v[a] == v'[a]`；另外两轴（切向）随便差多少都仍共面。
/// 1 体素薄板另一侧是**面号不同**（+x 对 −x）或法线轴坐标差 1 ⇒ 精确拒掉 ⇒ 不跨薄板 / 墙缝，
/// 连「同朝向、相隔一格的平行面」（1 格缝隙两侧）也被这条拒掉（法线相同但轴坐标不同）。
/// **它只看 DDA 的入射面**，因此不受 `voxel_normal` 的梯度法线影响（凸棱上那条法线是斜的，
/// 拿它去算平面距离会自相矛盾 —— 平面归属必须用这个轴对齐的面）。
fn gi_den_same_plane(hk0: u32, hk1: u32, k0: u32, k1: u32) -> bool {
  let face = (k1 >> 13u) & 7u;
  if (face != ((hk1 >> 13u) & 7u)) { return false; }
  if ((k1 >> 16u) != (hk1 >> 16u)) { return false; }
  let axis = face >> 1u;
  if (axis == 0u) { return (k0 & 0x1FFFu) == (hk0 & 0x1FFFu); }
  if (axis == 1u) { return ((k0 >> 13u) & 0x1FFFu) == ((hk0 >> 13u) & 0x1FFFu); }
  return (k1 & 0x1FFFu) == (hk1 & 0x1FFFu);
}

/// **全管线唯一的「同一个面」判据** = 平面归属（`gi_den_same_plane`，键）**且** 着色法线相似。
/// 两条各司其职、缺一不可：
/// · **平面归属**必须用 DDA 入射面（键）—— 它精确、与缩放无关，且能拒掉「同朝向但相隔一格的
///   平行面」（1 格缝隙两侧）与「薄板另一侧」；
/// · **相似度**必须用**着色法线**（`voxel_normal` 的梯度法线）—— 因为画面里「看起来是不是同一个
///   面」由着色法线决定：凸棱上它被邻域平均成 45° 斜向，那段画面是平滑过渡的。只用面键会把
///   这段切碎（着色平滑、GI 却是碎的 ⇒ 棱边一圈噪）。容差 `GI_DEN_N_DOT` 就是容忍这种倾斜。
/// 两个法线都从**存的**着色法线里读（`gi_main` 算一次），任何地方都**不重算**。
/// 这里是**已解包**法线的版本（reservoir 里的法线是解包后的 `vec3`）。
fn gi_same_surface_n(hk0: u32, hk1: u32, hn: vec3<f32>, k0: u32, k1: u32, n: vec3<f32>) -> bool {
  if (!gi_den_same_plane(hk0, hk1, k0, k1)) { return false; }
  return dot(hn, n) >= GI_DEN_N_DOT;
}

/// 同上，但另一侧的法线是**打包的 f16x2 一字**（导引 / 历史 / tap 缓冲里的存法）。
/// `hk0/hk1/hn` = 另一侧（历史像素 / tap / 上采样 texel）的键与法线；`k0/k1/n` = 本像素的。
fn gi_same_surface(hk0: u32, hk1: u32, hn: u32, k0: u32, k1: u32, n: vec3<f32>) -> bool {
  return gi_same_surface_n(hk0, hk1, gi_unpack_dir(hn), k0, k1, n);
}

/// 时域累积（每 GI 像素一次）。步骤：
/// ① 本帧邻域（半径 `GI_DEN_NBR_R`）两套统计（**必须分开**，见下）：
///    · **同面键** tap（同一个体素同一个面）→ 亮度一阶/二阶矩 → σ_n = **纯噪声**（只用于算 σ）；
///    · **同一个面** tap（`gi_same_surface` 通过，可跨体素）→ 颜色 mean/std → 本帧输入 + 钳制范围；
/// ② 输入 `c_in`：面跨多个 texel（n3 ≥ 2）取同面键均值，否则（面不跨 texel）取**同一面均值**，
///    再过一次 `mean ± K·std` 的离群钳制；
/// ③ 历史校验 = 导引里的上帧像素坐标 → 取那条历史的键与法线，与本帧**同一个面**
///    （`gi_same_surface`）且 `M > 0`。**这里不检测「世界变了吗」**：光照 / 几何一变，⑤ 的 AABB
///    钳制会把历史色拉回**本帧邻域**的范围 ⇒ 响应足够快；而"当帧重置"反而会让编辑帧露出未平滑的
///    输入、并让 atrous 的短历史稳定化（`GI_DEN_STAB_*`）整屏生效一帧（症状：放置体素时 GI 闪一下）。
/// ④ 噪声尺度 `σ_use = max(本帧同面键 σ_n, 累积矩 σ_hist)`；历史权重
///    `w = min(clamp(σ_α/σ_use, W_MIN, 1), (M−1)/M, 1/(1+mo))`，末项 `mo` = 本像素命中点在上帧的
///    落点与本像素的屏幕距离（运动自适应：位移越大，这条历史越可能来自面上另一个位置）；
/// ⑤ 历史色先钳到同一面邻域的 `mean ± K·std`（anti-ghosting）；
/// ⑥ 写回本帧色 + **累积矩**（`mix(hist, 本帧输入, 1−w)`）+ 本帧键 + 法线 + M（历史双缓冲，
///    Rust 每帧换绑）。
///
/// 为什么 σ 的**空间**那一套必须取**同面键**：同一个面内的像素是同一个积分，**差异只可能是噪声**；
/// 而共面邻域跨体素，差异里混着真实信号（逐体素风味）。用后者算 σ 会把「本来就很干净的面」
/// 因为邻居体素不同而判定成「噪声大」⇒ 历史权重被压低（实测静态逐帧时间 σ 只降到 42~47%）。
/// **输入**则相反：面跨多个 texel 时取**同面键邻域的均值**（面内像素接到的输入几乎相同 ⇒
/// 累积出来的历史也几乎相同 ⇒ 面内空间 σ 塌掉，而且不跨面）；**面不跨 texel 时**（低分辨率档）
/// 同面键邻域只剩它自己 ⇒ 改用**共面邻域均值**（同一个平面的相邻体素是同一个积分，
/// 差异只有噪声）⇒ 输入噪声降 √cnt，时域权重与 atrous 的 range 权重才回到正常量级。
fn gi_den_temporal(size: vec2<u32>, coord: vec2<i32>) {
  let x = i32(coord.x);
  let y = i32(coord.y);
  let idx = u32(y) * size.x + u32(x);
  let gb = idx * GI_DEN_GUIDE_WORDS;
  let hb = idx * GI_DEN_HIST_WORDS;
  if (!gi_den_valid_b(gb)) {
    for (var i = 0u; i < GI_DEN_HIST_WORDS; i = i + 1u) { gi_hist_cur[hb + i] = 0u; }
    gi_den_phi[idx] = 0.0;
    textureStore(gi_dn_tmp, coord, vec4<f32>(0.0));
    return;
  }
  let k0 = gi_den_key0_b(gb);
  let k1 = gi_den_key1_b(gb);
  // 本像素的**着色法线**：`gi_main` 已经算好写进导引，这里直接读（不重算那个梯度）。
  let n_c = gi_den_nrm_b(gb);
  let c_raw = textureLoad(gi_raw_tex, coord, 0).rgb;

  // ---- ① 两套邻域统计（半径 `GI_DEN_NBR_R`；两个集合都靠 `gi_same_surface` 判，
  // **同面键是它的特例**。邻域必须够大：斜面上「一个体素面 ≈ 一个 texel」⇒ 半径 1 时 `cnt`
  // 常常停在 1 ⇒ ③′ 钳制与 ③″ 全都不触发，历史色没护栏、输入退回单 texel 原始噪声）----
  var n3 = 0.0; // 同面键 tap 数（同一个体素同一个面：`σ` 只用这一套）
  var m3 = 0.0;
  var q3 = 0.0;
  var c3 = vec3<f32>(0.0); // 同面键 tap 的 rgb 和（本帧的「面均值」输入，见 ③）
  var cnt = 0.0; // 同一个面的 tap 数（含切向偏移的相邻体素面、法线倾斜相近的凸棱体素）
  var csum = vec3<f32>(0.0);
  var csum2 = vec3<f32>(0.0);
  let nr = i32(GI_DEN_NBR_R);
  for (var dy: i32 = -nr; dy <= nr; dy = dy + 1) {
    for (var dx: i32 = -nr; dx <= nr; dx = dx + 1) {
      let tx = x + dx;
      let ty = y + dy;
      if (tx < 0 || ty < 0 || tx >= i32(size.x) || ty >= i32(size.y)) { continue; }
      let tb = (u32(ty) * size.x + u32(tx)) * GI_DEN_GUIDE_WORDS;
      // 与 atrous 同一套两级读：平面归属只用键（`gi_same_surface` 的两半，判据一字不改）。
      let tk0 = gi_guide[tb + GI_DEN_G_KEY0];
      let tk1 = gi_guide[tb + GI_DEN_G_KEY1];
      if ((tk0 | tk1) == 0u) { continue; }
      if (!gi_den_same_plane(tk0, tk1, k0, k1)) { continue; }
      // 法线用的也是**存的**着色法线（`gi_main` 算一次），只在平面归属通过后才解包。
      if (dot(gi_den_nrm_b(tb), n_c) < GI_DEN_N_DOT) { continue; }
      let c = textureLoad(gi_raw_tex, vec2<i32>(tx, ty), 0).rgb;
      if (tk0 == k0 && tk1 == k1) {
        let l = dot(c, GI_LUMA);
        m3 = m3 + l;
        q3 = q3 + l * l;
        c3 = c3 + c;
        n3 = n3 + 1.0;
      }
      csum = csum + c;
      csum2 = csum2 + c * c;
      cnt = cnt + 1.0;
    }
  }
  let mu3 = m3 / max(n3, 1.0);
  let mo2 = q3 / max(n3, 1.0); // 同面键邻域的亮度二阶矩（与 `mu3` 配对，见下面的 σ）
  // 面内样本 < 2 ⇒ σ = 0（面内本来就没有空间信息可丢；此时靠下面的**同面均值**与累积矩）。
  let sigma = select(0.0, sqrt(max(mo2 - mu3 * mu3, 0.0)), n3 > 1.5);
  let cmean = csum / max(cnt, 1.0);
  let cstd = sqrt(max(csum2 / max(cnt, 1.0) - cmean * cmean, vec3<f32>(0.0)));
  // 参考亮度取「同面邻域均值」与「同面键均值」的较大者（同面 tap 为 0 时退化成本像素自己的亮度）。
  let l_ref = max(dot(cmean, GI_LUMA), mu3);
  // 权重与 φ 实际用的噪声尺度（默认 = 本帧同面键 σ；拿到历史后会被**累积矩**抬起来，见 `sigma_use`）。
  var sigma_use = sigma;
  // ③ 本帧输入 = **同面键邻域的均值**（同一个面内的像素是同一个积分 ⇒ 先面内平均再进时域累积）。
  var c_in = select(c_raw, c3 / max(n3, 1.0), n3 > 1.5);
  // ③″ **面不跨 texel 时改用同一面邻域均值**（低分辨率档下几乎所有面都如此；1/2 档只有远景小面）：
  // 同面键邻域里只剩它自己 ⇒ 上面那条「先面内平均」整条失效，输入退化成单个 texel 的原始噪声
  // （σ_in 大约 ×3）⇒ `w_var = σ_α/σ_in` 塌向下限、**时域累积被关掉**，只剩空间滤波兜 ⇒
  // 与改造前同款的「雪花屏」。同一个面上相邻体素描述的是**同一个积分**（差异只有噪声）⇒
  // 取同面均值把 σ_in 降 √cnt 倍。**面不跨 texel 时不存在「面内梯度」问题**（一个面就一个 texel）
  // ⇒ 这一条不会破坏「逐体素平坦」。
  if (n3 <= 1.5 && cnt > 1.5) { c_in = cmean; }
  // ③′ **离群抑制**（NRD pre-blur 的角色，热插在本 pass 里而不是单开一趟）：
  // GI 的原始估计是重尾分布（4 条候选里偶尔有一条打到强光面 ⇒ 离群亮点）。不压掉的话
  // 它会在时域里被反复加权、再被 atrous 摊成一片，表现为「偶发亮斑 + 逐帧闪烁」。
  // 用**共面邻域**的均值 ± K × 标准差做一次钳制即可；③ / ③″ 的均值本身已是空间平均，
  // 两者叠加后离群值进不了累积。
  if (cnt > 1.5) {
    c_in = clamp(c_in, cmean - GI_DEN_CLAMP_K * cstd, cmean + GI_DEN_CLAMP_K * cstd);
  }
  // 累积矩的**观测 = 本帧实际输入** `c_in`（面内 / 共面平均 + 钳制之后）：NRD 的 m1/m2 量的
  // 就是输入信号 ⇒ `σ_hist` 是**输入的帧间波动**，与 `σ_α` 直接可比（而不是某个中间量）。
  // 未与历史融合的初值 = 本帧观测；融合在 ② 里做，因为要用到实际的历史权重。
  let l_in = dot(c_in, GI_LUMA);
  var m1_acc = l_in;
  var m2_acc = l_in * l_in;

  // ---- ② 历史校验（同一个面）+ ⑤ AABB 钳制 + ④ 历史权重 ----
  var color = c_in;
  var m = 1.0;
  let code = gi_guide[gb + GI_DEN_G_PREV];
  if (code != 0xFFFFFFFFu) {
    let prev_px = vec2<i32>(i32(code & 0xFFFFu), i32((code >> 16u) & 0xFFFFu));
    // 历史校验 = **同一个面**（`gi_same_surface`：平面归属靠键 + 着色法线相似，
    // 历史里的法线是上帧那个像素自己的 —— 导引是单缓冲，读不到上帧的法线）。
    // 不通过 ⇒ `M = 1` 且色 = 本帧的同面均值（**本帧重建**：不拖影、不糊，也不回退到 0 / 旧值）。
    // 薄板两侧 / 墙缝两侧面号不同 ⇒ 一律不接历史。
    //
    // **历史搜索**（`GI_HIST_SEARCH_TAPS`，与 reservoir 侧同一张偏移表、同一个顺序）：判据不变，改的
    // 只是"去哪个像素找"。重投影落点那个上帧 texel 可能采到**相邻台阶**（斜面上一个体素面常常只占
    // 一个 texel）⇒ 判据不通过；而本面在上一帧确实有人采过，就在落点旁边。正中恒排第一 ⇒ 静止相机
    // 下行为与"只认落点"逐位一致。接不上时 `M = 1`、历史权重为 0 ⇒ 本帧原始估计直接上屏。
    var hb = 0u;
    var h_m = 0.0;
    var found = false;
    for (var s = 0u; s <= GI_HIST_SEARCH_TAPS; s = s + 1u) {
      let px = prev_px.x + gi_hist_search_offset(s).x;
      let py = prev_px.y + gi_hist_search_offset(s).y;
      if (px < 0 || py < 0 || px >= i32(size.x) || py >= i32(size.y)) { continue; }
      let cand_b = (u32(py) * size.x + u32(px)) * GI_DEN_HIST_WORDS;
      let cand_m = bitcast<f32>(gi_hist_prev[cand_b + GI_DEN_H_M]);
      if (cand_m <= 0.0) { continue; }
      if (!gi_same_surface(gi_hist_prev[cand_b + GI_DEN_H_KEY0],
                           gi_hist_prev[cand_b + GI_DEN_H_KEY1],
                           gi_hist_prev[cand_b + GI_DEN_H_NRM], k0, k1, n_c)) { continue; }
      hb = cand_b;
      h_m = cand_m;
      found = true;
      break;
    }
    if (found) {
      m = min(h_m + 1.0, f32(GI_DEN_M_MAX));
      let hcol = vec3<f32>(
        bitcast<f32>(gi_hist_prev[hb + GI_DEN_H_COL]),
        bitcast<f32>(gi_hist_prev[hb + GI_DEN_H_COL + 1u]),
        bitcast<f32>(gi_hist_prev[hb + GI_DEN_H_COL + 2u]),
      );
      var clamped = hcol;
      if (cnt > 1.5) {
        clamped = clamp(hcol, cmean - GI_DEN_CLAMP_K * cstd, cmean + GI_DEN_CLAMP_K * cstd);
      }
      // σ 取**累积矩**与本帧观测的较大者：累积矩把多帧的观测平均掉 ⇒ σ 不再逐帧抖
      // （逐帧抖会让下面的历史权重跟着抖，表现为持续的细颗粒闪动）；
      // 取 max 保证「噪声突然变大」（去遮挡、光照阶跃）时仍然按本帧的观测走，不迟钝。
      let h_m1 = bitcast<f32>(gi_hist_prev[hb + GI_DEN_H_M1]);
      let h_m2 = bitcast<f32>(gi_hist_prev[hb + GI_DEN_H_M2]);
      let sigma_hist = sqrt(max(h_m2 - h_m1 * h_m1, 0.0));
      sigma_use = max(sigma, sigma_hist);
      let w_var = clamp(GI_DEN_SIGMA_ALPHA / max(sigma_use, 1e-5), GI_DEN_W_HIST_MIN, 1.0);
      // WHY: 接受判据只证明「同一个面」、不证明「同一个位置」（面键的切向坐标不进判据），
      // 其近似误差随屏幕位移放大 —— 位移越大，这条历史越可能来自面上另一个位置（拖影来源）。
      // mo = 本像素命中点在上帧的落点（`prev_px`）与本像素的屏幕距离 ⇒ 上限 1/(1+mo)：
      // 记忆窗 ≈ 1+mo 帧。相机静止 ⇒ prev_px == coord ⇒ mo = 0 ⇒ 因子为 1（静止画面的
      // 累积长度与噪声不变）。
      let dx = f32(prev_px.x - x);
      let dy = f32(prev_px.y - y);
      let mo = sqrt(dx * dx + dy * dy);
      // 历史上限：M 帧以来最多信 (M−1)/M ⇒ 稳态也保留 1/M 的新信息（不无限累积）。
      let w = min(min(w_var, (m - 1.0) / m), 1.0 / (1.0 + mo));
      color = mix(c_in, clamped, w);
      // 矩按**与色完全相同的权重**累积（历史权重 = w ⇒ 本帧观测权重 = 1 − w）⇒
      // `M2 − M1²` 就是「本像素输入在累积窗口内的波动」，量纲与 `σ_α` 可比。
      m1_acc = mix(h_m1, l_in, 1.0 - w);
      m2_acc = mix(h_m2, l_in * l_in, 1.0 - w);
    }
  }

  // φ = `GI_DEN_LUMA_K` × max(噪声尺度, **相对下限**)：下限见 `GI_DEN_LUMA_FLOOR_REL`。
  // WHY: 噪声尺度取「同面键 σ / 累积矩」；**同面键邻域退化时（`n3 ≤ 1.5`：面比一个 texel 还小，
  // 低分辨率档的斜面与小面几乎全是这种）改用共面邻域逐通道 std 的亮度投影 `dot(cstd, GI_LUMA)`**。
  // 那条相对下限只是最后一道地板，不能当主尺度：它是"参考亮度的固定百分比"，而 GI 的**相对**波动
  // 随场景变暗而变大（暗处逃逸到天空的射线的辐亮度远高于任何被照到的面）⇒ 亮处 `|Δluma|` 尚在
  // 下限之内、暗处早已远超 ⇒ range 权重 `exp(-|Δluma|/φ)` 塌到 0、五轮 atrous 空转、raw 噪声直通。
  // 用实测离散度当尺度 ⇒ 与"这一帧实际有多大差异"同量纲，与场景明暗无关。
  // 只影响本像素自己的 atrous 权重：时域的历史权重仍走 `sigma_use`（`σ_α/σ_use`），不受本行影响。
  let sigma_phi = select(dot(cstd, GI_LUMA), sigma_use, n3 > 1.5);
  gi_den_phi[idx] = GI_DEN_LUMA_K * max(max(sigma_phi, GI_DEN_LUMA_FLOOR_REL * l_ref), 1e-5);

  // ---- ⑥ 写历史（本帧**累积后**的色 + 累积矩 + 本帧键 + 法线 + M）与输出 ----
  gi_hist_cur[hb + GI_DEN_H_COL] = bitcast<u32>(color.x);
  gi_hist_cur[hb + GI_DEN_H_COL + 1u] = bitcast<u32>(color.y);
  gi_hist_cur[hb + GI_DEN_H_COL + 2u] = bitcast<u32>(color.z);
  gi_hist_cur[hb + GI_DEN_H_M1] = bitcast<u32>(m1_acc);
  gi_hist_cur[hb + GI_DEN_H_M2] = bitcast<u32>(m2_acc);
  gi_hist_cur[hb + GI_DEN_H_M] = bitcast<u32>(m);
  gi_hist_cur[hb + GI_DEN_H_KEY0] = k0;
  gi_hist_cur[hb + GI_DEN_H_KEY1] = k1;
  // 法线原样转发本像素的（从导引读到的那个）—— 下一帧的接受判据要它。
  gi_hist_cur[hb + GI_DEN_H_NRM] = gi_guide[gb + GI_DEN_G_NRM];
  textureStore(gi_dn_tmp, coord, vec4<f32>(color, 1.0));
}

/// 迭代 atrous（`step` = 本轮步长 1/2/4/8/16，核半径 = `GI_DEN_ATROUS_R`）。
/// 权重完全按「几何 + 相似度」正确性给：
///  1. **不是同一个面 ⇒ 硬拒**（`gi_same_surface`）：平面归属靠键（薄板另一侧 / 墙缝另一侧 /
///     相隔一格的平行面都在这里精确拒掉），相似度靠**着色法线**（凸棱上被平均成 45° 斜向的那些
///     体素与相邻的同面体素只差 0.707 ⇒ 仍在容差内 ⇒ 那一段画面连成一片，与着色一致）；
///  2. **同面键 ⇒ 满权重 1**（`w = 1`，连亮度权重都不看：同一个面内的差异只可能是噪声）
///     —— 这是「逐体素平坦风味」的来源（同面像素被强拉平）；
/// 3. 同一个面但不是同键（切向偏移 / 两侧倾斜度不同）⇒ `dot(n) × exp(-|Δluma| / φ)`，
///     φ = `GI_DEN_LUMA_K` × 本像素的噪声尺度（时域 pass 写进 `gi_den_phi`：同面键 σ / 累积矩；
///     同面键邻域退化时取共面邻域实测离散度，见 `gi_den_temporal` 的 ④ 段说明）：那里的差异可能是
///     真实信号（比如同一地板上被墙挡住的一段），所以留一个亮度门；而「同一个面」保证了几何上是
///     同一个积分。
/// 核是 `(2R+1)²-1` 个 tap（R 由 Rust 按菜单「高质量降噪」写进 `gi_den_cfg[0]`：
/// 2 ⇒ 5×5 的 24 tap、1 ⇒ 3×3 的 8 tap）：同一足迹下采样越密 ⇒ 残余噪声越低。
/// 4. **短历史的像素把步长放大**（`GI_DEN_STAB_FRAMES` / `GI_DEN_STAB_MAX`）：`M` 小 ⇒ 同一批 tap
///     落得更远 ⇒ 用模糊代替生噪点（NRD 的 `blurRadiusScale`）；`M` 大了倍率回到 1 ⇒ 稳态不变。
fn gi_den_atrous(size: vec2<u32>, coord: vec2<i32>, step: i32) {
  let x = i32(coord.x);
  let y = i32(coord.y);
  let idx = u32(y) * size.x + u32(x);
  let gb = idx * GI_DEN_GUIDE_WORDS;
  if (!gi_den_valid_b(gb)) {
    textureStore(gi_dn_dst, coord, vec4<f32>(0.0));
    return;
  }
  // 核半径来自降噪自己的小配置 buffer（降噪 pass 只有 group(0)，够不到 `gi_u`，见 `bindings.wesl`）。
  let r = i32(gi_den_cfg[0u]);
  // **短历史稳定化**（`GI_DEN_STAB_FRAMES` / `GI_DEN_STAB_MAX`）：本帧的历史长度由时域 pass 写进
  // `gi_hist_cur`（两个降噪 pass 共用同一份 group(0) ⇒ 这里读得到）。`M` 小 ⇒ 步长放大 ⇒
  // 用**更宽的模糊**代替生噪点；`M ≥ GI_DEN_STAB_FRAMES` ⇒ 倍率 1 ⇒ 与稳定像素的路径逐位相同。
  let hb = idx * GI_DEN_HIST_WORDS;
  let m_hist = bitcast<f32>(gi_hist_cur[hb + GI_DEN_H_M]);
  let stab = clamp(GI_DEN_STAB_FRAMES / max(m_hist, 1.0), 1.0, GI_DEN_STAB_MAX);
  let st = max(1, i32(round(f32(step) * stab)));
  let k0 = gi_den_key0_b(gb);
  let k1 = gi_den_key1_b(gb);
  let n_c = gi_den_nrm_b(gb);
  let phi = max(gi_den_phi[idx], 1e-5);
  let c0 = textureLoad(gi_dn_src, coord, 0).rgb;
  let l0 = dot(c0, GI_LUMA);
  var acc = c0;
  var wsum = 1.0;
  for (var dy: i32 = -r; dy <= r; dy = dy + 1) {
    for (var dx: i32 = -r; dx <= r; dx = dx + 1) {
      if (dx == 0 && dy == 0) { continue; }
      let tx = x + dx * st;
      let ty = y + dy * st;
      if (tx < 0 || ty < 0 || tx >= i32(size.x) || ty >= i32(size.y)) { continue; }
      let tb = (u32(ty) * size.x + u32(tx)) * GI_DEN_GUIDE_WORDS;
      // 导引**两级读**：平面归属只用两个键字（被它拒掉的 tap 不再读法线那个字）。大核的多数 tap
      // 都在这一步退出（薄板另一侧、隔一格的平行面、天空都被 `gi_den_same_plane` 精确拒掉）⇒
      // 每帧省下这部分 tap 的 1/3 导引流量，且不改变任何判据（`gi_same_surface_n` = 同一判据的两半）。
      let tk0 = gi_guide[tb + GI_DEN_G_KEY0];
      let tk1 = gi_guide[tb + GI_DEN_G_KEY1];
      if ((tk0 | tk1) == 0u) { continue; }
      if (!gi_den_same_plane(tk0, tk1, k0, k1)) { continue; }
      let tn = gi_den_nrm_b(tb);
      if (dot(tn, n_c) < GI_DEN_N_DOT) { continue; }
      let s = textureLoad(gi_dn_src, vec2<i32>(tx, ty), 0).rgb;
      // 同面键 ⇒ 满权重 1（同一面内的差异只可能是噪声，见文件头 ③）；
      // 同一个面但不是同键（切向偏移 / 法线倾斜度不同）⇒ 法线点积 × 亮度 range 权重：
      // 那里的差异可能是真实信号（例如同一地板上的明暗两段、凸棱两侧的倾斜差）。
      var w = 1.0;
      if (!(tk0 == k0 && tk1 == k1)) {
        w = clamp(dot(tn, n_c), 0.0, 1.0) * exp(-abs(dot(s, GI_LUMA) - l0) / phi);
      }
      acc = acc + s * w;
      wsum = wsum + w;
    }
  }
  textureStore(gi_dn_dst, coord, vec4<f32>(acc / max(wsum, 1e-6), 1.0));
}
