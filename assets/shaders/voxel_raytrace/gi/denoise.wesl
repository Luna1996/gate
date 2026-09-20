import package::bindings::{gi_den_phi, gi_dn_dst, gi_dn_src, gi_dn_tmp, gi_guide, gi_guide_out, gi_hist_cur, gi_hist_prev, gi_raw_tex};
import package::gi::common::{gi_pack_dir, gi_unpack_dir};
import package::gi::consts::{GI_DEN_CLAMP_K, GI_DEN_G_KEY0, GI_DEN_G_KEY1, GI_DEN_G_NRM, GI_DEN_G_POS, GI_DEN_G_PREV, GI_DEN_G_TOL, GI_DEN_G_VALID, GI_DEN_GUIDE_WORDS, GI_DEN_H_COL, GI_DEN_H_KEY0, GI_DEN_H_KEY1, GI_DEN_H_M, GI_DEN_H_M1, GI_DEN_H_M2, GI_DEN_HIST_WORDS, GI_DEN_LUMA_K, GI_DEN_M_MAX, GI_DEN_N_DOT, GI_DEN_SIGMA_ALPHA, GI_DEN_W_HIST_MIN, GI_LUMA};

// ============================================================================
// GI 降噪（SVGF 家族两段）：**时域累积**（`gi_den_temporal`）+ **迭代 atrous 空间滤波**
// （`gi_den_atrous`，步长 1/2/4 各跑一次）。
//
// 输入 / 输出（GI 网格，尺寸 = 渲染分辨率 / gi_div：全分辨率或 1/2）：
//   · `gi_raw_tex`（= `gi_out` 的采样视图）：本帧原始 GI，rgb = gi·valid、a = valid；
//   · 导引 `gi_guide_out`/`gi_guide`：`gi_main` 写，面键（2 word，bit15 = 有效位）、命中体素中心
//     世界位置（3）、法线（八面体 f16x2）、**上帧重投影像素坐标**（1，`gi_u.prev_view_proj` 投出来
//     的；`0xFFFFFFFF` = 无）、平面检验容差（1，= `GI_DEN_PLANE_K` × 该体素世界边长）；
//   · 历史 `gi_hist_prev`/`gi_hist_cur`（双缓冲，每像素 `GI_DEN_HIST_WORDS` 字）；
//   · φ（每像素一个字）= `GI_DEN_LUMA_K` × 本帧噪声 σ，供 atrous 的亮度 range 权重；
//   · 输出：`gi_dn_tmp`（时域结果）→ atrous 三轮 ping-pong → `gi_den`（`dda_main` 采样）。
//     atrous 的输入是**采样视图**、输出是**存储视图**，Rust 每轮换绑同一份 layout 的两个号
//     ⇒ 一个入口跑任意轮数（见 `bindings.wesl` 的 14/15）。
//
// 三条硬约束怎么保证：
//  ① **不引入 ramp**：历史失效（重投影出屏 / 面键不等 / 上帧 M = 0）时**不**退回 0 也不沿用旧值，
//     而是直接用本帧原始估计（`M = 1`）。本帧原始估计与稳态估计是同一个积分的无偏估计
//     ⇒ 亮度不随时间爬升，第 1 帧 ≈ 稳态（差别只有噪声，靠空间段压）。
//  ② **不跨薄板 / 墙缝**：判据是**精确面键逐位相等**（体素,面,物体；薄板两侧是两个键），
//     加上法线相似 + 平面深度检验（世界位置取的是命中**体素中心** ⇒ 同一平面上相邻体素的内积
//     恰好为 0，而 1 体素薄板另一侧 = 1 个体素边长 > 容差）。
//  ③ **逐体素平坦风味**：atrous 里**同一面键的 tap 给满权重 1**（同面内被强拉平），
//     跨面才走几何 × 亮度权重 ⇒ 同一个面里的像素趋于同值，而不是被大核糊成渐变。
// ============================================================================

/// 导引 [word 址]：本像素是否有效（首命中不透明体素）。
fn gi_den_valid_b(b: u32) -> bool {
  return (gi_guide[b + GI_DEN_G_KEY1] & GI_DEN_G_VALID) != 0u;
}

/// 导引 [word 址] → 面键 word0。
fn gi_den_key0_b(b: u32) -> u32 {
  return gi_guide[b + GI_DEN_G_KEY0];
}

/// 导引 [word 址] → 面键 word1（剥掉有效位 ⇒ 与 `gi_key1` 的输出可直接逐位比较）。
fn gi_den_key1_b(b: u32) -> u32 {
  return gi_guide[b + GI_DEN_G_KEY1] & 0xFFFF7FFFu;
}

/// 导引 [word 址] → 命中体素中心的世界位置。
fn gi_den_pos_b(b: u32) -> vec3<f32> {
  return vec3<f32>(
    bitcast<f32>(gi_guide[b + GI_DEN_G_POS]),
    bitcast<f32>(gi_guide[b + GI_DEN_G_POS + 1u]),
    bitcast<f32>(gi_guide[b + GI_DEN_G_POS + 2u]),
  );
}

/// 导引 [word 址] → 世界空间法线（八面体解包）。
fn gi_den_nrm_b(b: u32) -> vec3<f32> {
  return gi_unpack_dir(gi_guide[b + GI_DEN_G_NRM]);
}

/// 写导引（`gi_main` / `gi_ss_main` 调用）。
/// `valid = false` ⇒ 整格清零（天空 / 介质像素：两段降噪都跳过，与 `gi_out` 的 valid = 0 一致）。
/// `tol` = 平面深度检验容差（`GI_DEN_PLANE_K` × 命中体素的世界边长；写侧算，见 `gi/consts.wesl`）。
/// `prev_px` / `prev_ok` = **真实命中点**（`h.point`）用 `gi_u.prev_view_proj` 投到**上帧** GI 网格的
/// 像素坐标。必须用命中点而不是体素中心：静止相机下前者正好映回本像素，后者会偏 1~2 像素 ⇒
/// 约 1/3 的像素每帧都因「上帧那个像素不是同一个面」而丢掉历史（实测接受率只有 68%）。
fn gi_den_guide_store(
  coord: vec2<i32>,
  size: vec2<u32>,
  valid: bool,
  k0: u32,
  k1: u32,
  pos: vec3<f32>,
  n: vec3<f32>,
  tol: f32,
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
  gi_guide_out[b + GI_DEN_G_KEY1] = k1 | GI_DEN_G_VALID;
  gi_guide_out[b + GI_DEN_G_POS] = bitcast<u32>(pos.x);
  gi_guide_out[b + GI_DEN_G_POS + 1u] = bitcast<u32>(pos.y);
  gi_guide_out[b + GI_DEN_G_POS + 2u] = bitcast<u32>(pos.z);
  gi_guide_out[b + GI_DEN_G_NRM] = gi_pack_dir(n);
  // 上帧像素坐标（GI 网格，u16 足够：渲染上限 4096）：x | y<<16；无 = 0xFFFFFFFF。
  var code = 0xFFFFFFFFu;
  if (prev_ok) { code = u32(prev_px.x) | (u32(prev_px.y) << 16u); }
  gi_guide_out[b + GI_DEN_G_PREV] = code;
  gi_guide_out[b + GI_DEN_G_TOL] = bitcast<u32>(tol);
}

/// 平面深度检验（SVGF 的写法）：`|(p_nbr − p_center)·n_center| < tol`。
/// `p` 取命中**体素中心**（不是射线命中点）⇒ 同一平面上的相邻体素内积**恒为 0**；
/// 1 体素薄板的另一侧 = 1 个世界边长 > tol（= 0.5 个边长）⇒ 被这条精确拒掉，
/// 而且与距离无关（远景薄板不会因为像素足迹变大而漏）。
fn gi_den_plane_ok(p_c: vec3<f32>, n_c: vec3<f32>, p_n: vec3<f32>, tol: f32) -> bool {
  return abs(dot(p_n - p_c, n_c)) < tol;
}

/// 时域累积（每 GI 像素一次）。步骤：
/// ① 本帧 3×3 邻域两套统计（**必须分开**，见下）：
///    · **同面键** tap（同一个体素同一个面）→ 亮度一阶/二阶矩 → σ_n = **纯噪声**；
///    · **共面** tap（法线相似 + 平面检验通过，可跨体素）→ 颜色 mean/std → AABB 钳制范围；
/// ② 历史校验 = 导引里的上帧像素坐标 → 取那条历史的**面键**与本帧**逐位相等**且 `M > 0`；
/// ③ 历史权重 `w = min(clamp(σ_α/σ_n, W_MIN, 1), (M−1)/M)`；
/// ④ 历史色先钳到共面邻域的 `mean ± K·std`（anti-ghosting）；
/// ⑤ 写回本帧色 + 本帧矩 + 本帧键 + M（历史双缓冲 ping-pong，Rust 每帧换绑）。
///
/// 为什么 σ 必须取**同面键**那一套：同一个面内的像素是同一个积分，**差异只可能是噪声**；
/// 而共面邻域跨体素，差异里混着真实信号（逐体素风味）。用后者算 σ 会把「本来就很干净的面」
/// 因为邻居体素不同而判定成「噪声大」⇒ 历史权重被压低（实测静态逐帧时间 σ 只降到 42~47%）。
/// 本帧输入同样取**同面键邻域的均值**（`c_in`）而不是中心像素：面内像素接到的输入几乎相同 ⇒
/// 累积出来的历史也几乎相同 ⇒ 面内空间 σ 塌掉，而且不跨面（只合同面键）。
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
  let p_c = gi_den_pos_b(gb);
  let n_c = gi_den_nrm_b(gb);
  let tol = bitcast<f32>(gi_guide[gb + GI_DEN_G_TOL]);
  let c_raw = textureLoad(gi_raw_tex, coord, 0).rgb;

  // ---- ① 两套邻域统计 ----
  var n3 = 0.0; // 同面键 tap 数（面内）
  var m3 = 0.0;
  var q3 = 0.0;
  var c3 = vec3<f32>(0.0); // 同面键 tap 的 rgb 和（本帧的「面均值」输入，见 ③）
  var cnt = 0.0; // 共面 tap 数（可跨体素）
  var csum = vec3<f32>(0.0);
  var csum2 = vec3<f32>(0.0);
  for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
    for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
      let tx = x + dx;
      let ty = y + dy;
      if (tx < 0 || ty < 0 || tx >= i32(size.x) || ty >= i32(size.y)) { continue; }
      let tb = (u32(ty) * size.x + u32(tx)) * GI_DEN_GUIDE_WORDS;
      if (!gi_den_valid_b(tb)) { continue; }
      let c = textureLoad(gi_raw_tex, vec2<i32>(tx, ty), 0).rgb;
      if (gi_den_key0_b(tb) == k0 && gi_den_key1_b(tb) == k1) {
        let l = dot(c, GI_LUMA);
        m3 = m3 + l;
        q3 = q3 + l * l;
        c3 = c3 + c;
        n3 = n3 + 1.0;
      }
      if (dot(gi_den_nrm_b(tb), n_c) >= GI_DEN_N_DOT
          && gi_den_plane_ok(p_c, n_c, gi_den_pos_b(tb), tol)) {
        csum = csum + c;
        csum2 = csum2 + c * c;
        cnt = cnt + 1.0;
      }
    }
  }
  let mu3 = m3 / max(n3, 1.0);
  // 面内样本 < 2（半分辨率下体素面常只占 1 个像素）⇒ σ = 0 = 纯累积（面内本来就没有空间信息可丢）。
  let sigma = select(0.0, sqrt(max(q3 / max(n3, 1.0) - mu3 * mu3, 0.0)), n3 > 1.5);
  let cmean = csum / max(cnt, 1.0);
  let cstd = sqrt(max(csum2 / max(cnt, 1.0) - cmean * cmean, vec3<f32>(0.0)));
  gi_den_phi[idx] = GI_DEN_LUMA_K * max(sigma, 1e-5);
  // ③ 本帧输入 = **同面键邻域的均值**（同一个面内的像素是同一个积分 ⇒ 先面内平均再进时域累积）。
  // 单像素面（n3 = 1）退化成中心像素本身 ⇒ 不改变旧行为。
  var c_in = select(c_raw, c3 / max(n3, 1.0), n3 > 1.5);
  // ③′ **离群抑制**（NRD pre-blur 的角色，热插在本 pass 里而不是单开一趟）：
  // GI 的原始估计是重尾分布（4 条候选里偶尔有一条打到强光面 ⇒ 离群亮点）。不压掉的话
  // 它会在时域里被反复加权、再被 atrous 摊成一片，表现为「偶发亮斑 + 逐帧闪烁」。
  // 用**共面邻域**的均值 ± K × 标准差做一次钳制即可；同面键均值（`c_in` 的来源）本身
  // 已经是面内平均，两者叠加后离群值进不了累积。
  if (cnt > 1.5) {
    c_in = clamp(c_in, cmean - GI_DEN_CLAMP_K * cstd, cmean + GI_DEN_CLAMP_K * cstd);
  }

  // ---- ② 历史校验（精确面键，逐位相等）+ ④ AABB 钳制 + ③ 历史权重 ----
  var color = c_in;
  var m = 1.0;
  let code = gi_guide[gb + GI_DEN_G_PREV];
  if (code != 0xFFFFFFFFu) {
    let px = i32(code & 0xFFFFu);
    let py = i32((code >> 16u) & 0xFFFFu);
    if (px < i32(size.x) && py < i32(size.y)) {
      let hpb = (u32(py) * size.x + u32(px)) * GI_DEN_HIST_WORDS;
      let h_m = bitcast<f32>(gi_hist_prev[hpb + GI_DEN_H_M]);
      // 面键逐位相等（体素 + 面 + 物体）：薄板两侧 / 墙缝两侧是两个键 ⇒ 一律不接历史。
      // 不等 ⇒ `M = 1` 且色 = 本帧的面均值（**本帧重建**：不拖影、不糊，也不回退到 0 / 旧值）。
      if (h_m > 0.0 && gi_hist_prev[hpb + GI_DEN_H_KEY0] == k0
          && gi_hist_prev[hpb + GI_DEN_H_KEY1] == k1) {
        m = min(h_m + 1.0, f32(GI_DEN_M_MAX));
        let hcol = vec3<f32>(
          bitcast<f32>(gi_hist_prev[hpb + GI_DEN_H_COL]),
          bitcast<f32>(gi_hist_prev[hpb + GI_DEN_H_COL + 1u]),
          bitcast<f32>(gi_hist_prev[hpb + GI_DEN_H_COL + 2u]),
        );
        var clamped = hcol;
        if (cnt > 1.5) {
          clamped = clamp(hcol, cmean - GI_DEN_CLAMP_K * cstd, cmean + GI_DEN_CLAMP_K * cstd);
        }
        let w_var = clamp(GI_DEN_SIGMA_ALPHA / max(sigma, 1e-5), GI_DEN_W_HIST_MIN, 1.0);
        // 历史上限：M 帧以来最多信 (M−1)/M ⇒ 稳态也保留 1/M 的新信息（不无限累积）。
        let w = min(w_var, (m - 1.0) / m);
        color = mix(c_in, clamped, w);
      }
    }
  }

  // ---- ⑤ 写历史（本帧**累积后**的色 + 本帧矩 + 本帧键 + M）与输出 ----
  gi_hist_cur[hb + GI_DEN_H_COL] = bitcast<u32>(color.x);
  gi_hist_cur[hb + GI_DEN_H_COL + 1u] = bitcast<u32>(color.y);
  gi_hist_cur[hb + GI_DEN_H_COL + 2u] = bitcast<u32>(color.z);
  gi_hist_cur[hb + GI_DEN_H_M1] = bitcast<u32>(mu3);
  gi_hist_cur[hb + GI_DEN_H_M2] = bitcast<u32>(q3 / max(n3, 1.0));
  gi_hist_cur[hb + GI_DEN_H_M] = bitcast<u32>(m);
  gi_hist_cur[hb + GI_DEN_H_KEY0] = k0;
  gi_hist_cur[hb + GI_DEN_H_KEY1] = k1;
  textureStore(gi_dn_tmp, coord, vec4<f32>(color, 1.0));
}

/// 迭代 atrous（`step` = 本轮步长，1/2/4）。权重完全按「几何 + 平面」正确性给：
///  1. **同面键 ⇒ 满权重 1**（`w = 1`，连亮度权重都不看：同一个面内的差异只可能是噪声）
///     —— 这是「逐体素平坦风味」的来源（同面像素被强拉平）；
///  2. 不同键：法线相似（`dot ≥ GI_DEN_N_DOT`）否则硬拒；
///  3. 不同键：平面深度检验（`gi_den_plane_ok`）—— 1 体素薄板的另一侧在这里被拒；
///  4. 不同键：亮度 range 权重 `exp(-|Δluma| / φ)`，φ = `GI_DEN_LUMA_K` × 本帧 σ。
/// 硬拒（2/3）用 `continue`（不是把权重压小）：大核模糊永远不该跨过一条真实的几何边界。
fn gi_den_atrous(size: vec2<u32>, coord: vec2<i32>, step: i32) {
  let x = i32(coord.x);
  let y = i32(coord.y);
  let idx = u32(y) * size.x + u32(x);
  let gb = idx * GI_DEN_GUIDE_WORDS;
  if (!gi_den_valid_b(gb)) {
    textureStore(gi_dn_dst, coord, vec4<f32>(0.0));
    return;
  }
  let k0 = gi_den_key0_b(gb);
  let k1 = gi_den_key1_b(gb);
  let p_c = gi_den_pos_b(gb);
  let n_c = gi_den_nrm_b(gb);
  let tol = bitcast<f32>(gi_guide[gb + GI_DEN_G_TOL]);
  let phi = max(gi_den_phi[idx], 1e-5);
  let c0 = textureLoad(gi_dn_src, coord, 0).rgb;
  let l0 = dot(c0, GI_LUMA);
  var acc = c0;
  var wsum = 1.0;
  for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
    for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
      if (dx == 0 && dy == 0) { continue; }
      let tx = x + dx * step;
      let ty = y + dy * step;
      if (tx < 0 || ty < 0 || tx >= i32(size.x) || ty >= i32(size.y)) { continue; }
      let tb = (u32(ty) * size.x + u32(tx)) * GI_DEN_GUIDE_WORDS;
      if (!gi_den_valid_b(tb)) { continue; }
      let s = textureLoad(gi_dn_src, vec2<i32>(tx, ty), 0).rgb;
      var w = 1.0;
      if (!(gi_den_key0_b(tb) == k0 && gi_den_key1_b(tb) == k1)) {
        let nd = dot(gi_den_nrm_b(tb), n_c);
        if (nd < GI_DEN_N_DOT) { continue; }
        if (!gi_den_plane_ok(p_c, n_c, gi_den_pos_b(tb), tol)) { continue; }
        w = clamp(nd, 0.0, 1.0) * exp(-abs(dot(s, GI_LUMA) - l0) / phi);
      }
      acc = acc + s * w;
      wsum = wsum + w;
    }
  }
  textureStore(gi_dn_dst, coord, vec4<f32>(acc / max(wsum, 1e-6), 1.0));
}
