import package::bindings::{DDGI_INDIR_COLLBASE_BASE, DDGI_INDIR_RAYBASE_BASE, DDGI_INDIR_RPP_BASE, ddgi_depth, ddgi_depth_out, ddgi_indirect, ddgi_irr, ddgi_irr_out, ddgi_meta, ddgi_samples, ddgi_slot_pos, ddgi_u, ddgi_worklist};
import package::ddgi::consts::{DDGI_ALPHA, DDGI_BLEND_SELF, DDGI_COLLECT_THREADS, DDGI_DEPTH_ALPHA, DDGI_DEPTH_TEXELS, DDGI_IRR_TEXELS, DDGI_LOD_COUNT, DDGI_META_ENABLED, DDGI_META_NBR_VALID, DDGI_PI, DDGI_TEXEL_MIN_COVERAGE, DDGI_T_MAX, DDGI_WL_IDX_MASK};
import package::ddgi::helpers::{ddgi_age_alpha, ddgi_cell_in_window, ddgi_depth_coord, ddgi_irr_coord, ddgi_lod_slot_base, ddgi_meta_nbr, ddgi_nbr_bit_lod0, ddgi_oct_texel_dir, ddgi_slot_cell, ddgi_slot_own, ddgi_slot_world_cell};

// ---- 空间混合（RTXGI ProbeBlending 的简化）----
// 采样端 gi = Σ(irr·w)/Σw，w = wtri·wn·wd。`wn` 会把"探针落在背面"的那半边剔掉，于是
// 同一个 cell 内权重在「探针处 ≈1 ↔ cell 角 ≈0.5」之间变化；归一化后，加权平均会把
// **每个探针自己的辐照度**顶出来 → 与探针间距同周期的晶格亮斑（墙角/接缝亮带的来源）。
// 相邻探针的辐照度越接近，这个重分配越看不出来 → 把 6 个网格邻居的同名纹素混进来。
fn ddgi_irr_nbr(id: u32, tx: u32, ty: u32) -> vec4<f32> {
  let c = ddgi_irr_coord(id, tx, ty);
  return textureLoad(ddgi_irr, vec2<i32>(vec2<u32>(c.y, c.z)), i32(c.x), 0);
}
/// 6 个轴向邻居的同名纹素按探针**实际间距**加权求和；返回 (加权和, 权重和)。
/// 只接受同 LOD 网格内、本帧 ENABLED 且**该纹素有数据**的邻居。
///
/// 【为什么不能按值过滤】旧版是 `max(v) < 1e-4 → 跳过`。`DDGI_CAST_FLOOR` 改成 0 之后
/// "已写入但真黑"与"从未写过"的值**都是 0**，按值判会把**暗邻居整批踢掉** —— 而暗邻居
/// 恰恰是稀释亮邻居的唯一手段。后果：一颗落点在墙另一侧、看得到天光的探针，四周的暗邻居
/// 全部不参与平均 ⇒ 它完全不被稀释 ⇒ 在自己的投影处形成一个亮斑（室内墙角/接缝的亮边；
/// 粗级 cell 更大、更多邻居是实心格或未成型 ⇒ 更严重，对上"离越远越明显"）。
/// 现在按第 4 通道的**覆盖度**判"有没有数据"（见 DDGI_TEXEL_MIN_COVERAGE）。
fn ddgi_probe_blend_irr(lod: u32, pidx: u32, tx: u32, ty: u32) -> vec4<f32> {
  let dims = ddgi_u.lods[lod].dims.xyz;
  let base = ddgi_lod_slot_base(lod);
  let cs = f32(ddgi_u.lods[lod].origin.w);
  let self_pos = ddgi_slot_pos[base + pidx].xyz;
  var acc = vec3<f32>(0.0);
  var wacc = 0.0;
  if (lod == 0u) {
    // LOD0 是 chunk 段编址：**槽位线性邻居不再是空间邻居**（跨 chunk 会跳到别的段）→
    // 改在世界 cell 空间里取 6 个面邻居。邻居集与 LOD1~3 的 stride 版完全对应（同为 ±1 cell），
    // 只是跨 chunk 时用 `ddgi_slot_own` 现算槽位。
    let wc = ddgi_slot_world_cell(0u, pidx);
    let self_meta = ddgi_meta[base + pidx];
    let self_nbr = ddgi_meta_nbr(self_meta);
    // NBR_VALID 未置位（非活跃探针 / 掩码未算）→ 不做门控，退回改造前行为。
    let self_gate = (self_meta & DDGI_META_NBR_VALID) != 0u;
    for (var a = 0u; a < 6u; a = a + 1u) {
      var off = vec3<i32>(1, 0, 0);
      if (a == 1u) { off = vec3<i32>(-1, 0, 0); }
      else if (a == 2u) { off = vec3<i32>(0, 1, 0); }
      else if (a == 3u) { off = vec3<i32>(0, -1, 0); }
      else if (a == 4u) { off = vec3<i32>(0, 0, 1); }
      else if (a == 5u) { off = vec3<i32>(0, 0, -1); }
      let nwc = wc + off;
      if (!ddgi_cell_in_window(0u, nwc)) { continue; }
      let nslot = ddgi_slot_own(0u, nwc);
      if (nslot == base + pidx) { continue; }
      // ---- 邻接可见性门控（与 LOD1~3 分支同一套语义，位序映射见 ddgi_nbr_bit_lod0）----
      let sbit = ddgi_nbr_bit_lod0(a);
      let nmeta = ddgi_meta[nslot];
      if (self_gate) {
        if ((self_nbr & (1u << sbit)) == 0u) { continue; }
        // 反向也判（邻居掩码已算时才判）：本方向通、反向被挡属薄墙，按被挡处理。
        if ((nmeta & DDGI_META_NBR_VALID) != 0u
            && (ddgi_meta_nbr(nmeta) & (1u << (sbit ^ 1u))) == 0u) { continue; }
      }
      if ((nmeta & DDGI_META_ENABLED) == 0u) { continue; }
      let v = ddgi_irr_nbr(nslot, tx, ty);
      if (v.w < DDGI_TEXEL_MIN_COVERAGE) { continue; }
      let npos = ddgi_slot_pos[nslot].xyz;
      let w = max(0.0, 1.0 - length(npos - self_pos) / (cs * 2.0));
      acc = acc + v.xyz * w;
      wacc = wacc + w;
    }
    return vec4<f32>(acc, wacc);
  }
  let cell = ddgi_slot_cell(lod, pidx);
  let self_meta = ddgi_meta[base + pidx];
  let self_nbr = ddgi_meta_nbr(self_meta);
  let self_gate = (self_meta & DDGI_META_NBR_VALID) != 0u;
  for (var a = 0u; a < 3u; a = a + 1u) {
    let stride = select(select(1u, dims.x, a == 1u), dims.x * dims.y, a == 2u);
    let extent = select(select(dims.x, dims.y, a == 1u), dims.z, a == 2u);
    let coord = select(select(cell.x, cell.y, a == 1u), cell.z, a == 2u);
    for (var s = 0u; s < 2u; s = s + 1u) {
      let plus = s == 1u;
      if (plus && coord + 1u >= extent) { continue; }
      if (!plus && coord == 0u) { continue; }
      // ---- 邻接可见性门控：本探针向该邻居方向的线段被几何挡住 → 不混 ----
      // 位序 = a*2 + (plus ? 1 : 0)（0=-X,1=+X,2=-Y,3=+Y,4=-Z,5=+Z）。
      // 【为什么必须门控】混合完全不看可见性，一个「能看到天空」的探针会把它的天空辐照度
      // 抹给隔墙的暗探针 → 封闭房间的凹角整片漏光（且逐帧向外扩散成平滑梯度）。
      // 反向也判：邻居的掩码里「指向本探针」那一位（bit^1）为 0 → 同样不混，
      // 避免单向不对称（本方向通、反向被挡的薄墙情形）。
      // 掩码未算（NBR_VALID=0）的一侧不参与判定 → 退回改造前行为，不会误关混合。
      let sbit = a * 2u + select(0u, 1u, plus);
      let nslot = base + select(pidx - stride, pidx + stride, plus);
      let nmeta = ddgi_meta[nslot];
      if (self_gate) {
        if ((self_nbr & (1u << sbit)) == 0u) { continue; }
        if ((nmeta & DDGI_META_NBR_VALID) != 0u
            && (ddgi_meta_nbr(nmeta) & (1u << (sbit ^ 1u))) == 0u) { continue; }
      }
      if ((nmeta & DDGI_META_ENABLED) == 0u) { continue; }
      let v = ddgi_irr_nbr(nslot, tx, ty);
      if (v.w < DDGI_TEXEL_MIN_COVERAGE) { continue; }
      let npos = ddgi_slot_pos[nslot].xyz;
      let w = max(0.0, 1.0 - length(npos - self_pos) / (cs * 2.0));
      acc = acc + v.xyz * w;
      wacc = wacc + w;
    }
  }
  return vec4<f32>(acc, wacc);
}

// ============================================================================
// 阶段三 collect（Douglas pass #3）：射线样本积成图集。
//   · irradiance 4×4：按「样本方向 · 纹素方向」半球余弦加权（来自任何方向的射线都能贡献）。
//   · depth 8×8：**只吃绑定到本纹素的射线**（见 cast 的「射线 ↔ 深度纹素绑定」）→
//     mean/std 是「该方向最近距离」的均值与其噪声，Chebyshev 判定成立的前提。
// 每探针 DDGI_COLLECT_THREADS 个线程：texel < 16 → irradiance 4×4；否则 depth 8×8。
// 与上一帧做时域混合（hysteresis）；探针刚唤醒（age ≤ 1）时直接写入，避免残留旧世界位置的数据。
// 读 ddgi_irr/ddgi_depth（BG4 采样侧），写 ddgi_irr_out/ddgi_depth_out（BG5 写入侧）。
// ============================================================================
// 一次 dispatch 覆盖全部 LOD：全局 tid → coll_base[lod] + probe_idx × 80 + texel。
// ============================================================================
/// 单线程：处理「一个探针的一个纹素」。
fn ddgi_collect_one(tid: u32) {
  var lod = 0u;
  for (var l = 1u; l < DDGI_LOD_COUNT; l = l + 1u) {
    if (tid >= atomicLoad(&ddgi_indirect[DDGI_INDIR_COLLBASE_BASE + l])) { lod = l; }
  }
  let rpp = atomicLoad(&ddgi_indirect[DDGI_INDIR_RPP_BASE + lod]);
  if (rpp == 0u) { return; }
  let local = tid - atomicLoad(&ddgi_indirect[DDGI_INDIR_COLLBASE_BASE + lod]);
  let probe_idx = local / DDGI_COLLECT_THREADS;
  let texel = local % DDGI_COLLECT_THREADS;
  let slot = ddgi_lod_slot_base(lod) + probe_idx;
  let packed = bitcast<u32>(ddgi_worklist[slot].w);
  let age = packed & 0xFFu;
  // 图集纹素按「探针的 cell 下标」寻址，而 slot 只是压缩列表里的排名 ——
  // 用错会把整张图集写成错位置换（表现：大面积发黑、只有局部有 GI）。
  // ⚠️ 这里必须用 19 位掩码：16 位会把 LOD0（34 万 cell）的下标截断 → 见 DDGI_WL_IDX_MASK。
  let atlas_slot = ddgi_lod_slot_base(lod) + ((packed >> 8u) & DDGI_WL_IDX_MASK);
  // 该探针的首条射线在全局射线空间里的编号 = 本 LOD 起始射线号 + 探针号×rpp
  let si = atomicLoad(&ddgi_indirect[DDGI_INDIR_RAYBASE_BASE + lod]) + probe_idx * rpp;

  if (texel < DDGI_IRR_TEXELS * DDGI_IRR_TEXELS) {
    // ---- irradiance：E = π · Σ(w·L) / Σw，w = max(0, dot(texel_dir, ray_dir)) ----
    let tx = texel % DDGI_IRR_TEXELS;
    let ty = texel / DDGI_IRR_TEXELS;
    let td = ddgi_oct_texel_dir(tx, ty, DDGI_IRR_TEXELS);
    var sum = vec3<f32>(0.0);
    var wsum = 0.0;
    for (var i = 0u; i < rpp; i = i + 1u) {
      let s = ddgi_samples[(si + i) * 2u];
      let w = max(0.0, dot(td, s.xyz));
      sum = sum + w * ddgi_samples[(si + i) * 2u + 1u].xyz;
      wsum = wsum + w;
    }
    if (wsum > 1e-5) {
      let irr_new = DDGI_PI * sum / wsum;
      // 先空间混合（和 6 邻居同名纹素），再时域混合 —— 见 ddgi_probe_blend_irr 注释
      let nb = ddgi_probe_blend_irr(lod, atlas_slot - ddgi_lod_slot_base(lod), tx, ty);
      let irr_s = (irr_new * DDGI_BLEND_SELF + nb.xyz) / (DDGI_BLEND_SELF + nb.w);
      let cc = ddgi_irr_coord(atlas_slot, tx, ty);
      let idx = vec2<i32>(vec2<u32>(cc.y, cc.z));
      let prev = textureLoad(ddgi_irr, idx, i32(cc.x), 0);
      // 年龄越小时向新样本靠拢越多（1/age 等权滑动平均）→ 新探针几帧内成型；
      // 年龄大了自动切回 DDGI_ALPHA，稳态脉动不变。见 ddgi_age_alpha。
      let a = ddgi_age_alpha(age, DDGI_ALPHA);
      // .w = 该纹素的**覆盖度**：每被写过一次就向 1 靠拢一次；从未写过则恒为 0。
      // 只有它能把"已写入但真黑"与"从未写过"分开（两者的值都是 0）—— 空间混合需要这个
      // 信号才能拿暗邻居稀释亮邻居，见 DDGI_TEXEL_MIN_COVERAGE / ddgi_probe_blend_irr。
      textureStore(
        ddgi_irr_out,
        idx,
        i32(cc.x),
        vec4<f32>(mix(prev.xyz, irr_s, a), mix(prev.w, 1.0, a)),
      );
    } else {
      // 本帧没有属于本纹素的射线 → 把上一帧读数**原样搬到本帧缓冲**。
      // 不搬的话 ping-pong 会让该纹素在 A/B 两个缓冲之间交替：采样每帧读到的是
      // 「另一个缓冲里两帧前的值」→ 图集在「有数据 ↔ 0」之间逐帧翻转 → wsum 处处
      // 高频闪（实测：Normal 档几乎看不出、wsum 档整个场景都在跳）。
      // 旧实现「射线全球随机撒」几乎每帧都有射线落到每个纹素，掩盖了这个 bug；
      // 改为「射线绑定纹素 + 帧轮换」（见 cast）后每纹素每 ⌈64/rpp⌉ 帧才写一次，才暴露。
      let cc = ddgi_irr_coord(atlas_slot, tx, ty);
      let idx = vec2<i32>(vec2<u32>(cc.y, cc.z));
      textureStore(ddgi_irr_out, idx, i32(cc.x), textureLoad(ddgi_irr, idx, i32(cc.x), 0));
    }
  } else {
    // ---- depth：本纹素绑定射线的**命中**距离统计 ----
    // 两条铁律：① 只统计**属于本纹素**的射线（绑定，见 cast）；② 只累计**命中**射线。
    // ② 的理由：早期版本把未命中（dist = DDGI_T_MAX = 8192）也加进均值，只要有射线看到
    // 天空，均值就被抬到数百且回落极慢 → dtex 系统性偏大 → 遮挡判定几乎恒为「可见」
    // → 深度闸门形同虚设 → 严重漏光。只累计命中后 dtex 逼近「该方向最近表面距离」；
    // 该方向一条都没命中 → 无遮挡物 → 不更新（保留上次命中均值；从未命中则恒 0 =
    // 无数据，采样侧按「无数据」剔除该角）。
    let t = texel - DDGI_IRR_TEXELS * DDGI_IRR_TEXELS;
    let tx = t % DDGI_DEPTH_TEXELS;
    let ty = t / DDGI_DEPTH_TEXELS;
    let frame = u32(ddgi_u.params.x);
    let ntex = DDGI_DEPTH_TEXELS * DDGI_DEPTH_TEXELS;
    var dsum = 0.0;
    var dsum2 = 0.0;
    var cnt = 0u;
    for (var i = 0u; i < rpp; i = i + 1u) {
      // 【与 cast 同一套绑定】本纹素只吃「属于它」的射线：(frame·rpp + i) % ntex == t。
      // 本帧可能对应 0 条（rpp < ntex 且未轮到）、1 条（rpp == ntex）或若干条（rpp > ntex）。
      // 这些射线的方向同属本纹素（只是纹素内抖动不同）→ 下面的 mean/mean2 就是
      // 「该方向最近表面距离」的均值与**噪声**方差，正是 Chebyshev 需要的量。
      // 旧版按 20° 锥权重把附近方向的射线也混进来，mean 被远处命中系统性抬高 →
      // 隔墙采样点被判「可见」→ 漏光（详见 cast 里的注释）。
      if ((frame * rpp + i) % ntex != t) { continue; }
      let s = ddgi_samples[(si + i) * 2u];
      // 未命中（dist = DDGI_T_MAX）不参与：本纹素存的是「该方向最近表面距离」，
      // 朝开阔方向的射线根本没有表面 → 不更新（保留上次命中均值；从未命中则恒为 0，
      // 采样侧按「无数据」剔除该角）。
      if (s.w < DDGI_T_MAX) {
        dsum = dsum + s.w;
        dsum2 = dsum2 + s.w * s.w;
        cnt = cnt + 1u;
      }
    }
    if (cnt > 0u) {
      let mean = dsum / f32(cnt);
      let mean2 = dsum2 / f32(cnt);
      let cc = ddgi_depth_coord(atlas_slot, tx, ty);
      let idx = vec2<i32>(vec2<u32>(cc.y, cc.z));
      let prev = textureLoad(ddgi_depth, idx, i32(cc.x), 0).xy;
      // 深度也走同样的年龄斜坡：新探针的深度图第一帧只摊到约 1 条射线/纹素，噪声极大，
      // 若沿用 0.03 的慢混合，遮挡判定会带着错误读数持续 ~1 秒（表现是"漏光/发暗慢慢退"）。
      let a = ddgi_age_alpha(age, DDGI_DEPTH_ALPHA);
      // 【为什么存的是「距离均值的跨帧序列 std」而不是本帧 std】
      // 绑定后每个纹素每 ⌈64/rpp⌉ 帧才拿到一条射线（rpp≈4 → 16 帧），绝大多数帧里
      // cnt 至多为 1 → 本帧 std 恒等于 0 → 采样侧
      // `soft = max(dtex.y, mean·0.02)` 就退化成 0.02·mean 的**刀锋**。
      // 而 mean 是单样本估计、跨帧方向抖动 → wd 在 ~1 与 ~0 之间翻 → 该角被
      // `wd <= 1e-3` 整块剔掉又放回 → wsum（以及跟着它走的光影）逐帧跳、亮区边界伸缩，
      // 相机静止也照样闪。Chebyshev 的过渡宽度本该由"深度读数有多不确定"决定，而不是
      // 一个恒定的 2%。
      // 做法：同时 EMA「距离」与「距离²」→ 存的是**落进本纹素的样本序列的 std**，
      // 即 mean 的真实不确定度尺度（同一纹素内抖动方向落在不同距离上的离散度）。
      // 注意 pm2 必须由上一帧的 (mean, std) 反推 —— 直接存 mean² 会在 f16 里溢出
      // （最远 8192 → 8192² = 6.7e7 ≫ f16 上限 65504）。中间量都是 f32，最后只存 std。
      let pm2 = prev.x * prev.x + prev.y * prev.y;
      let em = mix(prev.x, mean, a);
      let em2 = mix(pm2, mean2, a);
      let dep_std = sqrt(max(em2 - em * em, 0.0));
      // 存储纹素的 store 值类型是 vec4<f32>（naga 校验要求）：.x = mean、.y = std
      textureStore(
        ddgi_depth_out,
        idx,
        i32(cc.x),
        vec4<f32>(em, dep_std, 0.0, 0.0),
      );
    } else {
      // 同 irradiance：本帧无样本时把上一帧读数搬到本帧缓冲（含"滚动换主"——旧读数来自
      // 紧邻的上一任世界 cell，是有效近似，不能清零；见 bake 的换主说明）。
      let cc = ddgi_depth_coord(atlas_slot, tx, ty);
      let idx = vec2<i32>(vec2<u32>(cc.y, cc.z));
      textureStore(ddgi_depth_out, idx, i32(cc.x), textureLoad(ddgi_depth, idx, i32(cc.x), 0));
    }
  }
}

/// collect 入口：**grid-stride** 循环覆盖全部工作项。
///
/// 【为什么不能只派发 ceil(total_coll / 64) 个 workgroup】seal 把 collect 的 WG 数写成
/// `min((total_coll + 63) / 64, 65535)`（65535 = `dispatchWorkgroupsIndirect` 的单维上限）。
/// 而 `total_coll = Σ 活跃探针 × 80`。旧版隐含前提「活跃数 ≤ 槽数 65536」在 dims 改成按
/// 世界 AABB 算后失效 —— 槽位 39 万、活跃数上一个量级 → WG 被钳到 65535（= 4.19M 线程
/// = 52428 个探针），**排在后面的 LOD / 槽位永远不被 collect** → 图集恒空 → 成片无 GI，
/// 且黑区范围随活跃数变化而移动（与实测吻合）。
/// grid-stride 让「派发上限」与「覆盖完整性」解耦：每线程按 stride 走完自己的份额。
/// stride 必须等于派发上限（65535 × 64），见 Rust `dispatch_ddgi` 与 seal 的 `min(.., 65535)`。
