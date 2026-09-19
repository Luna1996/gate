import package::bindings::{DDGI_INDIR_COLLBASE_BASE, DDGI_INDIR_RAYBASE_BASE, DDGI_INDIR_RPP_BASE, ddgi_depth, ddgi_depth_out, ddgi_indirect, ddgi_irr, ddgi_irr_out, ddgi_meta, ddgi_samples, ddgi_slot_pos, ddgi_u, ddgi_worklist};
import package::ddgi::consts::{DDGI_ALPHA, DDGI_BLEND_SELF, DDGI_CHUNK_LOD, DDGI_COLLECT_THREADS, DDGI_DEPTH_ALPHA, DDGI_DEPTH_CONV_SAMPLES, DDGI_DEPTH_TEXELS, DDGI_IRR_TEXELS, DDGI_LOD_COUNT, DDGI_META_ENABLED, DDGI_META_NBR_VALID, DDGI_PI, DDGI_TEXEL_MIN_COVERAGE, DDGI_T_MAX, DDGI_WL_IDX_MASK};
import package::ddgi::helpers::{ddgi_age_alpha, ddgi_cell_in_window, ddgi_depth_coord, ddgi_irr_coord, ddgi_lod_slot_base, ddgi_meta_nbr, ddgi_nbr_bit_chunk, ddgi_oct_texel_dir, ddgi_slot_cell, ddgi_slot_own, ddgi_slot_world_cell};

// ---- 空间混合：把 6 个网格邻居的同名纹素混进来。 ----
fn ddgi_irr_nbr(id: u32, tx: u32, ty: u32) -> vec4<f32> {
  let c = ddgi_irr_coord(id, tx, ty);
  return textureLoad(ddgi_irr, vec2<i32>(vec2<u32>(c.y, c.z)), i32(c.x), 0);
}
/// 6 个轴向邻居的同名纹素按探针实际间距加权求和；返回 (加权和, 权重和)。
/// 只接受同 LOD、双向邻接可见性掩码已算且不撞几何、本帧 ENABLED 且覆盖度 ≥ DDGI_TEXEL_MIN_COVERAGE 的邻居（未算掩码按被挡处理）。
fn ddgi_probe_blend_irr(lod: u32, pidx: u32, tx: u32, ty: u32) -> vec4<f32> {
  let dims = ddgi_u.lods[lod].dims.xyz;
  let base = ddgi_lod_slot_base(lod);
  let cs = f32(ddgi_u.lods[lod].origin.w);
  let self_pos = ddgi_slot_pos[base + pidx].xyz;
  var acc = vec3<f32>(0.0);
  var wacc = 0.0;
  if (lod == DDGI_CHUNK_LOD) {
    // chunk 级是 chunk 段编址：在世界 cell 空间取 6 个面邻居。
    let wc = ddgi_slot_world_cell(DDGI_CHUNK_LOD, pidx);
    let self_meta = ddgi_meta[base + pidx];
    let self_nbr = ddgi_meta_nbr(self_meta);
    // 本探针掩码是否已算；未算则整段不混。
    let self_gate = (self_meta & DDGI_META_NBR_VALID) != 0u;
    for (var a = 0u; a < 6u; a = a + 1u) {
      var off = vec3<i32>(1, 0, 0);
      if (a == 1u) { off = vec3<i32>(-1, 0, 0); }
      else if (a == 2u) { off = vec3<i32>(0, 1, 0); }
      else if (a == 3u) { off = vec3<i32>(0, -1, 0); }
      else if (a == 4u) { off = vec3<i32>(0, 0, 1); }
      else if (a == 5u) { off = vec3<i32>(0, 0, -1); }
      let nwc = wc + off;
      if (!ddgi_cell_in_window(DDGI_CHUNK_LOD, nwc)) { continue; }
      let nslot = ddgi_slot_own(DDGI_CHUNK_LOD, nwc);
      if (nslot == base + pidx) { continue; }
      // 邻接可见性门控：掩码未算的探针按「被挡」处理。
      let sbit = ddgi_nbr_bit_chunk(a);
      let nmeta = ddgi_meta[nslot];
      if (!self_gate || (self_nbr & (1u << sbit)) == 0u) { continue; }
      if ((nmeta & DDGI_META_NBR_VALID) == 0u
          || (ddgi_meta_nbr(nmeta) & (1u << (sbit ^ 1u))) == 0u) { continue; }
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
      // 邻接可见性门控：位序 = a*2 + (plus ? 1 : 0)，反向位也判；掩码未算按「被挡」处理。
      let sbit = a * 2u + select(0u, 1u, plus);
      let nslot = base + select(pidx - stride, pidx + stride, plus);
      let nmeta = ddgi_meta[nslot];
      if (!self_gate || (self_nbr & (1u << sbit)) == 0u) { continue; }
      if ((nmeta & DDGI_META_NBR_VALID) == 0u
          || (ddgi_meta_nbr(nmeta) & (1u << (sbit ^ 1u))) == 0u) { continue; }
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
// 阶段三 collect：射线样本积成图集。读 ddgi_irr/ddgi_depth，写 ddgi_irr_out/ddgi_depth_out。
// 每探针 DDGI_COLLECT_THREADS 线程：texel < DDGI_IRR_TEXELS² → irradiance（半球余弦加权），其余 → depth。
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
  // 图集纹素按探针的 cell 下标寻址，slot 只是压缩列表排名，两者不可混用。
  // 掩码必须用 DDGI_WL_IDX_MASK（19 位）。
  let atlas_slot = ddgi_lod_slot_base(lod) + ((packed >> 8u) & DDGI_WL_IDX_MASK);
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
      // 先空间混合（6 邻居同名纹素）再时域混合。
      let nb = ddgi_probe_blend_irr(lod, atlas_slot - ddgi_lod_slot_base(lod), tx, ty);
      let irr_s = (irr_new * DDGI_BLEND_SELF + nb.xyz) / (DDGI_BLEND_SELF + nb.w);
      let cc = ddgi_irr_coord(atlas_slot, tx, ty);
      let idx = vec2<i32>(vec2<u32>(cc.y, cc.z));
      let prev = textureLoad(ddgi_irr, idx, i32(cc.x), 0);
      // `a = ddgi_age_alpha(age, DDGI_ALPHA)`；该纹素覆盖度为 0（从未写过）时取 a = 1。
      // `.w` = 图集纹素的覆盖度（每写一次向 1 靠拢，从未写过恒 0），用于区分「已写入但真黑」与「从未写过」。
      let a = select(ddgi_age_alpha(age, DDGI_ALPHA), 1.0, prev.w <= 0.0);
      textureStore(
        ddgi_irr_out,
        idx,
        i32(cc.x),
        vec4<f32>(mix(prev.xyz, irr_s, a), mix(prev.w, 1.0, a)),
      );
    } else {
      // 本帧无属于本纹素的射线 → 把上一帧读数搬到本帧缓冲（ping-pong 要求）。
      let cc = ddgi_irr_coord(atlas_slot, tx, ty);
      let idx = vec2<i32>(vec2<u32>(cc.y, cc.z));
      textureStore(ddgi_irr_out, idx, i32(cc.x), textureLoad(ddgi_irr, idx, i32(cc.x), 0));
    }
  } else {
    // ---- depth：本纹素绑定射线的命中距离统计 ----
    // 只累计命中（dist < DDGI_T_MAX）；无命中则不更新（从未命中恒 0 = 无数据，采样侧剔除该角）。
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
      // mean/mean2 是该方向最近表面距离的均值与方差（Chebyshev 用），不混入邻近方向。
      if ((frame * rpp + i) % ntex != t) { continue; }
      let s = ddgi_samples[(si + i) * 2u];
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
      // `.z` = 该纹素的收敛时钟（累计更新次数，封顶 DDGI_DEPTH_CONV_SAMPLES）。
      // alpha 取 age 项与 1/n_upd 的较大者。
      let n_upd = min(prev4.z + 1.0, DDGI_DEPTH_CONV_SAMPLES);
      let a = max(ddgi_age_alpha(age, DDGI_DEPTH_ALPHA), 1.0 / n_upd);
      // `.y` 是样本序列的 EMA std（不是本帧 std），由「距离」与「距离²」两个 EMA 反推。
      // pm2 = prev.x² + prev.y²。
      let pm2 = prev.x * prev.x + prev.y * prev.y;
      let em = mix(prev.x, mean, a);
      let em2 = mix(pm2, mean2, a);
      let dep_std = sqrt(max(em2 - em * em, 0.0));
      // 图集通道：`.x` = mean、`.y` = std、`.z` = 收敛时钟（采样侧不读）。
      textureStore(
        ddgi_depth_out,
        idx,
        i32(cc.x),
        vec4<f32>(em, dep_std, n_upd, 0.0),
      );
    } else {
      // 同 irradiance：本帧无样本时搬上一帧读数到本帧缓冲。
      let cc = ddgi_depth_coord(atlas_slot, tx, ty);
      let idx = vec2<i32>(vec2<u32>(cc.y, cc.z));
      textureStore(ddgi_depth_out, idx, i32(cc.x), textureLoad(ddgi_depth, idx, i32(cc.x), 0));
    }
  }
}

/// collect 入口：grid-stride 循环覆盖全部工作项。
/// stride 必须等于派的 WG 上限（65535 × 64），与 Rust 侧 `dispatch_ddgi` / seal 的 `min(.., 65535)` 同步。
