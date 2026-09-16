import package::bindings::{DDGI_INDIR_COLLBASE_BASE, DDGI_INDIR_RAYBASE_BASE, DDGI_INDIR_RPP_BASE, ddgi_depth, ddgi_depth_out, ddgi_indirect, ddgi_irr, ddgi_irr_out, ddgi_meta, ddgi_samples, ddgi_slot_pos, ddgi_u, ddgi_worklist};
import package::ddgi::consts::{DDGI_ALPHA, DDGI_BLEND_SELF, DDGI_COLLECT_THREADS, DDGI_DEPTH_ALPHA, DDGI_DEPTH_CONV_SAMPLES, DDGI_DEPTH_SOFT_MIN, DDGI_DEPTH_TEXELS, DDGI_FAST_ALPHA, DDGI_FAST_BRIGHT_MIN, DDGI_FAST_DEPTH_STEP, DDGI_FAST_STEP, DDGI_IRR_TEXELS, DDGI_LOD_COUNT, DDGI_META_ENABLED, DDGI_META_NBR_VALID, DDGI_PI, DDGI_TEXEL_MIN_COVERAGE, DDGI_T_MAX, DDGI_WL_IDX_MASK};
import package::ddgi::helpers::{ddgi_age_alpha, ddgi_cell_in_window, ddgi_depth_coord, ddgi_irr_coord, ddgi_lod_slot_base, ddgi_maxc, ddgi_meta_nbr, ddgi_nbr_bit_lod0, ddgi_oct_texel_dir, ddgi_slot_cell, ddgi_slot_own, ddgi_slot_world_cell};

// ---- 空间混合（RTXGI ProbeBlending 的简化）----
// 采样端 gi = Σ(irr·w)/Σw，w = wtri·wn·wd。`wn` 剔掉"探针落在背面"的那半边后，归一化会把
// 每个探针自己的辐照度顶出来 → 与探针间距同周期的晶格亮斑（墙角/接缝亮带）。
// 把 6 个网格邻居的同名纹素混进来可压低它。
fn ddgi_irr_nbr(id: u32, tx: u32, ty: u32) -> vec4<f32> {
  let c = ddgi_irr_coord(id, tx, ty);
  return textureLoad(ddgi_irr, vec2<i32>(vec2<u32>(c.y, c.z)), i32(c.x), 0);
}
/// 6 个轴向邻居的同名纹素按探针**实际间距**加权求和；返回 (加权和, 权重和)。
/// 只接受同 LOD 网格内、本帧 ENABLED 且**该纹素有数据**的邻居。
/// "有数据"必须按第 4 通道的覆盖度判（DDGI_TEXEL_MIN_COVERAGE）：DDGI_CAST_FLOOR = 0 时
/// "已写入但真黑"与"从未写过"的值都是 0，按值过滤会把暗邻居整批踢掉 —— 而暗邻居正是稀释
/// 亮邻居的唯一手段。
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
    // NBR_VALID 未置位（非活跃探针 / 掩码未算）→ 不做门控。
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
      // 必须门控：混合本身不看可见性，能看到天空的探针会把天空辐照度抹给隔墙的暗探针
      // ⇒ 封闭房间凹角整片漏光。反向也判（邻居掩码的 bit^1），避免薄墙造成的单向不对称；
      // 两侧掩码未算（NBR_VALID=0）时不参与判定，不会误关混合。
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
// 阶段三 collect：射线样本积成图集。
//   · irradiance 4×4：按「样本方向 · 纹素方向」半球余弦加权（任何方向的射线都能贡献）。
//   · depth 8×8：只吃绑定到本纹素的射线 ⇒ mean/std 是"该方向最近距离"的均值与噪声。
// 每探针 DDGI_COLLECT_THREADS 个线程：texel < DDGI_IRR_TEXELS² → irradiance，否则 depth。
// 读 ddgi_irr/ddgi_depth（BG4 采样侧），写 ddgi_irr_out/ddgi_depth_out（BG5 写入侧）；
// 一次 dispatch 覆盖全部 LOD：全局 tid → coll_base[lod] + probe_idx × DDGI_COLLECT_THREADS + texel。
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
  // 图集纹素按「探针的 cell 下标」寻址，而 slot 只是压缩列表里的排名，两者不可混用。
  // 掩码必须是 DDGI_WL_IDX_MASK（19 位）：16 位会截断 LOD0（34 万 cell）的下标。
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
      // a = ddgi_age_alpha(age, DDGI_ALPHA)：age 越小向新样本靠拢越多（1/age 等权平均），
      // age 到上限后自动切到 DDGI_ALPHA 的长窗口（见 ddgi_age_alpha）。
      // 该纹素从未被写过（覆盖度 0）时直接取新样本（a = 1）：否则 rpp 小时它每 ⌈64/rpp⌉ 帧才
      // 更新一次，从 0 慢慢爬会长期停在假黑；覆盖度也因此一写即到 1，成为持久的"有没有数据"判据。
      // .w = 该纹素的覆盖度（每写一次向 1 靠拢，从未写过恒 0），只有它能把"已写入但真黑"与
      // "从未写过"分开（两者值都是 0），见 DDGI_TEXEL_MIN_COVERAGE。
      var a = select(ddgi_age_alpha(age, DDGI_ALPHA), 1.0, prev.w <= 0.0);
      // 大变化快速跟随（见 consts 的 DDGI_FAST_*）：长窗 1/3000 会把「发光体被删掉」这件事
      // 记住几十分钟 —— 脏区重烘只覆盖被编辑的体素，不覆盖被它照亮过的探针。只在"亮量级"
      // 触发，抬到 1/8 后 ~30 次更新收敛完；暗区不触发，长窗的噪声免疫不变。
      let dmax = max(abs(irr_s.x - prev.x), max(abs(irr_s.y - prev.y), abs(irr_s.z - prev.z)));
      if (max(ddgi_maxc(prev.xyz), ddgi_maxc(irr_s)) > DDGI_FAST_BRIGHT_MIN
          && dmax > DDGI_FAST_STEP) {
        a = max(a, DDGI_FAST_ALPHA);
      }
      textureStore(
        ddgi_irr_out,
        idx,
        i32(cc.x),
        vec4<f32>(mix(prev.xyz, irr_s, a), mix(prev.w, 1.0, a)),
      );
    } else {
      // 本帧没有属于本纹素的射线 → 把上一帧读数原样搬到本帧缓冲。
      // 不搬的话 ping-pong 下该纹素在 A/B 两缓冲间交替，采样每帧读到的是"另一个缓冲里的旧值"
      // ⇒ 图集在"有数据 ↔ 0"之间逐帧翻转、wsum 高频闪。
      let cc = ddgi_irr_coord(atlas_slot, tx, ty);
      let idx = vec2<i32>(vec2<u32>(cc.y, cc.z));
      textureStore(ddgi_irr_out, idx, i32(cc.x), textureLoad(ddgi_irr, idx, i32(cc.x), 0));
    }
  } else {
    // ---- depth：本纹素绑定射线的命中距离统计 ----
    // 只统计属于本纹素的射线，且只累计命中（dist < DDGI_T_MAX）：本纹素存的是"该方向最近表面
    // 距离"，混进未命中（天空）会把均值抬到数百且回落极慢 ⇒ 遮挡判定几乎恒为"可见"。
    // 本方向一条都没命中 → 不更新（保留上次命中均值；从未命中则恒 0 = 无数据，采样侧剔除该角）。
    let t = texel - DDGI_IRR_TEXELS * DDGI_IRR_TEXELS;
    let tx = t % DDGI_DEPTH_TEXELS;
    let ty = t / DDGI_DEPTH_TEXELS;
    let frame = u32(ddgi_u.params.x);
    let ntex = DDGI_DEPTH_TEXELS * DDGI_DEPTH_TEXELS;
    var dsum = 0.0;
    var dsum2 = 0.0;
    var cnt = 0u;
    for (var i = 0u; i < rpp; i = i + 1u) {
      // 【与 cast 同一套绑定】本纹素只吃属于它的射线：(frame·rpp + i) % ntex == t。
      // 本帧可能对应 0 条（rpp < ntex 且未轮到）、1 条（rpp == ntex）或若干条（rpp > ntex）；
      // 这些射线方向同属本纹素（只是纹素内抖动不同）⇒ 下面的 mean/mean2 是该方向最近表面距离
      // 的均值与噪声方差，正是 Chebyshev 需要的量（不可按角度锥把邻近方向混进来）。
      if ((frame * rpp + i) % ntex != t) { continue; }
      let s = ddgi_samples[(si + i) * 2u];
      // 未命中（dist = DDGI_T_MAX）不参与：本纹素存的是"该方向最近表面距离"。
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
      let prev4 = textureLoad(ddgi_depth, idx, i32(cc.x), 0);
      let prev = prev4.xy;
      // 收敛时钟 = 纹素自己的更新次数（存 .z，Rgba16Float 整数精确，见 DDGI_DEPTH_CONV_SAMPLES）：
      // 前 DDGI_DEPTH_CONV_SAMPLES 次更新是 1/n 的等权平均 —— 与辐照度用 age 同思路，只是时钟
      // 换成"这个纹素积累了多少样本"。
      // alpha 取 age 项与 1/n_upd 的较大者：换主/编辑时 age 被压回 1，计数却是跨换主继承的
      // ⇒ 只有 age 项能在那几帧把 alpha 抬回大值、快速跟上新位置。
      let n_upd = min(prev4.z + 1.0, DDGI_DEPTH_CONV_SAMPLES);
      var a = max(ddgi_age_alpha(age, DDGI_DEPTH_ALPHA), 1.0 / n_upd);
      // 深度侧同款快速跟随（见 consts 的 DDGI_FAST_*）：几何大跳变（编辑 / 移除发光体）会把
      // "该方向最近表面距离"整体挪走，1/255 的地板会让它滞留很久 ⇒ Chebyshev 可见性过期
      // （留下的是"假阴影"而不是假亮光）。判据用 2×std 兜住"样本本身就散"的纹素，再加绝对下限。
      if (abs(mean - prev.x) > max(2.0 * max(prev.y, DDGI_DEPTH_SOFT_MIN), DDGI_FAST_DEPTH_STEP)) {
        a = max(a, DDGI_FAST_ALPHA);
      }
      // 存的是「落进本纹素的样本序列 std」而不是本帧 std：绝大多数帧里 cnt ≤ 1 ⇒ 本帧 std 恒 0，
      // 会让 Chebyshev 的过渡宽度塌成固定比例的刀锋。
      // 做法：同时 EMA「距离」与「距离²」，std 由两者反推。pm2 必须由上一帧的 (mean, std) 反推
      // —— 直接存 mean² 会在 f16 里溢出（最远 8192 → 8192²  f16 上限 65504）。
      let pm2 = prev.x * prev.x + prev.y * prev.y;
      let em = mix(prev.x, mean, a);
      let em2 = mix(pm2, mean2, a);
      let dep_std = sqrt(max(em2 - em * em, 0.0));
      // 存储纹素的 store 值类型是 vec4<f32>（naga 校验要求）：
      // .x = mean、.y = std、.z = 本纹素累计更新次数（收敛时钟，采样侧不读）
      textureStore(
        ddgi_depth_out,
        idx,
        i32(cc.x),
        vec4<f32>(em, dep_std, n_upd, 0.0),
      );
    } else {
      // 同 irradiance：本帧无样本时把上一帧读数搬到本帧缓冲。换主后旧读数来自紧邻的上一任
      // 世界 cell，是有效近似，不能清零。
      let cc = ddgi_depth_coord(atlas_slot, tx, ty);
      let idx = vec2<i32>(vec2<u32>(cc.y, cc.z));
      textureStore(ddgi_depth_out, idx, i32(cc.x), textureLoad(ddgi_depth, idx, i32(cc.x), 0));
    }
  }
}

/// collect 入口：**grid-stride** 循环覆盖全部工作项。
/// seal 把 collect 的 WG 数钳在 `dispatchWorkgroupsIndirect` 的单维上限 65535，而
/// `total_coll = Σ 活跃探针 × DDGI_COLLECT_THREADS` 远大于 65535 × 64 ⇒ 只派发一轮会漏掉
/// 排在后面的 LOD / 槽位（图集恒空、成片无 GI）。grid-stride 让派发上限与覆盖完整性解耦。
/// stride 必须等于派发上限（65535 × 64），见 Rust `dispatch_ddgi` 与 seal 的 `min(.., 65535)`。
