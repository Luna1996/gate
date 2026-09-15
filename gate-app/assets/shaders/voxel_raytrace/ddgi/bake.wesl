import package::bindings::{ddgi_cell, ddgi_cell_id, ddgi_meta, ddgi_u, g};
import package::brickmap::{Grid};
import package::ddgi::consts::{DDGI_BAKE_STAGGER};
import package::ddgi::helpers::{DDGI_REC_ENABLED, ddgi_brick_state, ddgi_cell_in_window, ddgi_cell_state_sized, ddgi_lod_cell_size, ddgi_lod_count, ddgi_lod_slot_base, ddgi_meta_pack, ddgi_rec_off, ddgi_rec_pack, ddgi_slot_own, ddgi_slot_world_cell};
import package::ddgi::sort::{ddgi_box_hits_object, ddgi_push_out_of_objects};
import package::trace::{make_grid};

/// 对齐的 8³ 子块是否**完全空**（= 它的 8 个 4³ 子节点全空）。
///
/// brickmap 的分裂因子是 4（256³→64³→16³→4³→1³），没有原生 8³ 节点，所以这里聚合 2³ 个 4³
/// 节点。8³ 正是 Douglas 八叉树里 16³ 的下一层（字幕："traverses downward until it finds a
/// totally empty leaf"）；加上它才能让"优先选**最大的**空子区域"在 16³ 与 4³ 之间不漏档。
fn ddgi_block8_empty(g: Grid, o: vec3<i32>) -> bool {
  for (var k = 0; k < 2; k = k + 1) {
    for (var j = 0; j < 2; j = j + 1) {
      for (var i = 0; i < 2; i = i + 1) {
        if (ddgi_brick_state(g, o + vec3<i32>(i, j, k) * 4, 3u) != 0u) { return false; }
      }
    }
  }
  return true;
}

// 在 [cmin, cmin+cs) 内按 16³ → 8³ → 4³ 逐级找靠中心的空叶；返回 (p, found)。
// 全部找不到（格内自由空间全是 <4 体素宽的窄缝）→ **返回 found = false，不放探针**。
/// 规则**严格对齐 Douglas #23 字幕**：
///   · "one of the **largest** subnodes is selected" —— 从粗到细，**第一个**含"完全空子块"的
///     层级胜出；
///   · "places the probe at the center of that subnode" —— 探针放在该子块中心；
///   · 同级并列时取**离 cell 中心最近**的（Douglas："a certain order that prioritizes subnodes
///     near the center of the cell"）；
///   · "for a cell that was totally full of voxels, no probe would be generated"（调用方已挡）。
/// 惰性求值：只有上一级完全找不到空块时才展开下一级；16³/8³/4³ 都只在**混合 16³** 子块内展开
/// （全空 16³ 已在上一级命中、全满 16³ 必然没有空子块），避免整格扫描的指数开销。
/// **没有净距门槛**（曾加过 4 体素硬门槛，见上方注释）；层级只下探到 4³ ⇒ 格内自由空间若
/// 全是 <4 体素宽的窄缝，本格**没有探针**（见函数尾部的说明）。
fn ddgi_place_probe(g: Grid, cmin: vec3<i32>, cs: i32, center: vec3<f32>, lod: u32) -> vec4<f32> {
  let n16 = max(cs / 16, 1);
  // ---- Step 2：粗级 LOD **优先继承细级探针的位置** ----
  // Douglas："I then down sample the generated data to pick out probe positions for higher
  // levels of detail or LODs." —— 粗级应该是细级的**低分辨率版本**，而不是同一片空间里的
  // 另一套独立数据；后者会让跨级边界像"换了一套光照"（LOD1 的锯齿/跳变就是这么来的）。
  //
  // 细级 cell 边长 = cs/2 ⇒ 本 cell 正好覆盖 `wc*2 + (0|1)³` 这 8 个细级 cell。取其中
  // 离本 cell 中心**最近**的那个已放置探针的位置，直接作为本格的探针。找不到（细级盒之外
  // 的那一圈）才退回下面原有的放置启发式。
  // `wc` 用 center 反推（floor，负数也正确），与 `ddgi_slot_world_cell` 的约定一致。
  // 【lod 必须由调用方传入，不能从 cs 反推】旧版写 `let my_lod = log2(cs/16)`，隐含
  // 「cell = 16 << lod」的等比关系；一旦某一级的 cell 不再等比（例如最粗级放大到 256 以换取
  // 覆盖范围），反推就会越界或指错细级。现在直接用调用方的 lod。
  if (lod > 0u) {
    let flod = lod - 1u;
    let fcs = ddgi_lod_cell_size(flod);
    let wc = vec3<i32>(floor(center / f32(cs)));
    var best_d2 = 1e30;
    var best_p = vec3<f32>(0.0);
    var found = false;
    for (var k = 0; k < 2; k = k + 1) {
      for (var j = 0; j < 2; j = j + 1) {
        for (var i = 0; i < 2; i = i + 1) {
          let fwc = wc * 2 + vec3<i32>(i, j, k);
          // 读**细级本帧的 bake 输出**（ddgi_cell / ddgi_cell_id），而不是 ddgi_slot_pos：
          // 位置必须与细级**同一帧**自洽，否则相机滚动时粗级会继承到"上一个世界 cell 的
          // 探针位置"（可能落在实体内部）→ 粗级边界跳变/锯齿。bake 已拆成 per-LOD 的 4 个
          // pass（细→粗），pass 边界就是内存屏障，这里读到的正是本帧细级刚算出的结果。
          // 细级该 cell 必须**可编址**：LOD0 下 = 它所属 chunk 已领段（未领段时 ddgi_slot_own
          // 返回 0；下面的 cell_id 比对虽也能过滤掉，但显式挡掉更清晰，避免读无关槽位）。
          if (!ddgi_cell_in_window(flod, fwc)) { continue; }
          let fslot = ddgi_slot_own(flod, fwc);
          // 该槽位必须**当前确实覆盖 fwc**（细级窗口外 / 环面绕到别的世界 cell → 跳过），
          // 且本帧 bake 已写过它（cell_id.w != 0）。
          if (ddgi_cell_id[fslot].w == 0 || any(ddgi_cell_id[fslot].xyz != fwc)) { continue; }
          let frec = ddgi_cell[fslot];
          if ((frec & DDGI_REC_ENABLED) == 0u) { continue; }
          let fp = vec3<f32>(fwc * fcs) + vec3<f32>(ddgi_rec_off(frec)) / 255.0 * f32(fcs);
          let d2 = dot(fp - center, fp - center);
          if (d2 < best_d2) {
            best_d2 = d2;
            best_p = fp;
            found = true;
          }
        }
      }
    }
    if (found) {
      return vec4<f32>(best_p, 1.0);
    }
  }
  // 轮 1（16³ 及以上）：从大到小找**最大的全空子块**（对齐 Douglas #23："in a breadth first
  // manner so that one of the **largest** subnodes is selected"），同级内取离 cell 中心最近的。
  //
  // 【为什么不能固定 16³ 粒度】旧实现只试 16³，且取「离 center 最近」——对「薄板地面 +
  // 下方也是空气」的 cell，center 附近若落在地表**下方**，被选中的空块也在下方 → 探针落到
  // 地下 → 对朝上的地面，8 个角的 wn = dot(n, 指向探针) 全 ≤ 0 → 被硬剔（Probe 档品红）
  // → 该片 GI 纯黑、与周边形成硬边（"有 GI / 无 GI"割裂）。优先取**更大的**空块能让探针
  // 落在"更开阔、离薄板更远"的一侧，因为同级内更大的空块其中心离表面更远。
  // cs 本身也在列表里 ⇒ 全空 cell 的"最大空叶"就是整格 → 探针落在正中（Douglas："for a
  // totally empty cell the probe would end up right in the center"）。
  let sizes = array<i32, 3>(64, 32, 16);
  for (var si = 0; si < 3; si = si + 1) {
    let sz = sizes[si];
    if (sz > cs) { continue; }
    let n = max(cs / sz, 1);
    var b_d2 = 1e30;
    var b_p = vec3<f32>(0.0);
    var found = false;
    for (var k = 0; k < n; k = k + 1) {
      for (var j = 0; j < n; j = j + 1) {
        for (var i = 0; i < n; i = i + 1) {
          let sub = cmin + vec3<i32>(i, j, k) * sz;
          if (ddgi_cell_state_sized(g, sub, sz) == 0u) {
            let p = vec3<f32>(sub) + vec3<f32>(f32(sz) * 0.5);
            let d2 = dot(p - center, p - center);
            if (d2 < b_d2) { b_d2 = d2; b_p = p; found = true; }
          }
        }
      }
    }
    if (found) { return vec4<f32>(b_p, 1.0); }
  }
  // 轮 2（8³）：无空 16³ → 在**混合 16³** 内找最近的「全空 8³」（8³ 是 16³ 的下一层）。
  var b8_d2 = 1e30;
  var b8_p = vec3<f32>(0.0);
  var f8 = false;
  for (var k = 0; k < n16; k = k + 1) {
    for (var j = 0; j < n16; j = j + 1) {
      for (var i = 0; i < n16; i = i + 1) {
        let sub16 = cmin + vec3<i32>(i, j, k) * 16;
        if (ddgi_cell_state_sized(g, sub16, 16) == 2u) {
          for (var kk = 0; kk < 2; kk = kk + 1) {
            for (var jj = 0; jj < 2; jj = jj + 1) {
              for (var ii = 0; ii < 2; ii = ii + 1) {
                let sub8 = sub16 + vec3<i32>(ii, jj, kk) * 8;
                if (ddgi_block8_empty(g, sub8)) {
                  let p = vec3<f32>(sub8) + vec3<f32>(4.0);
                  let d2 = dot(p - center, p - center);
                  if (d2 < b8_d2) { b8_d2 = d2; b8_p = p; f8 = true; }
                }
              }
            }
          }
        }
      }
    }
  }
  if (f8) { return vec4<f32>(b8_p, 1.0); }
  // 轮 3（4³）：无空 8³ → 在混合 16³ 内找最近的「全空 4³」。**无净距门槛**——全空 4³ 的中心
  // 离最近表面 ≥2 体素，直接采用。
  var b4_d2 = 1e30;
  var b4_p = vec3<f32>(0.0);
  var f4 = false;
  for (var k = 0; k < n16; k = k + 1) {
    for (var j = 0; j < n16; j = j + 1) {
      for (var i = 0; i < n16; i = i + 1) {
        let sub16 = cmin + vec3<i32>(i, j, k) * 16;
        if (ddgi_cell_state_sized(g, sub16, 16) == 2u) {
          for (var kk = 0; kk < 4; kk = kk + 1) {
            for (var jj = 0; jj < 4; jj = jj + 1) {
              for (var ii = 0; ii < 4; ii = ii + 1) {
                let sub4 = sub16 + vec3<i32>(ii, jj, kk) * 4;
                if (ddgi_brick_state(g, sub4, 3u) == 0u) {
                  let p = vec3<f32>(sub4) + vec3<f32>(2.0);
                  let d2 = dot(p - center, p - center);
                  if (d2 < b4_d2) { b4_d2 = d2; b4_p = p; f4 = true; }
                }
              }
            }
          }
        }
      }
    }
  }
  if (f4) { return vec4<f32>(b4_p, 1.0); }
  // ---- 到这里就**不放探针**（原「轮 4：1³ 保底档」已删）----
  // 旧版在没有任何 ≥4³ 全空子块时，还会逐 1³ 体素找一个空位（ddgi_leaf16）把探针塞进去，
  // 理由是"格内只要还有空体素，就一定放得出探针"。那个不变式在这里**主动放弃**：
  // 能满足"连 4³ 空区都没有"的 cell，其空腔必然是 **1~3 体素宽的窄缝/接缝**（两个物体之间的
  // 缝、薄板内部的空腔）。这种缝里进不了光 ⇒ 塞进去的探针辐照度恒 ≈0 —— 它不会变暗变亮，
  // 就是**恒 0**（射线全打在贴着的墙上，自反馈回路没有天空那道非零种子，见 ddgi_cast 的
  // DDGI_CAST_FLOOR 注释）。而它是"本点同侧"的合法角 ⇒ 缝两侧的像素会稳定地采到它
  // ⇒ 表现就是**纯黑窄带**（Probe 档：r4 同侧场兜底 = 青绿，它也只能把这份 ≈0 传下去）。
  // 不放探针之后：该 cell 的 `ENABLED=0`，采样侧按"无探针"剔除（Probe 橙），像素改用
  // **周围 cell 的探针**——它们落在开阔空间里，值是对的 ⇒ 窄缝表面与周边同亮度（1 体素宽的
  // 缝本来就只是建模接缝，其表面视觉上就等同于周边表面）。
  // 【覆盖性】三线性 stencil 跨 8 个 cell，个别 cell 无针不会留洞；只有"整片 8 个 cell 都
  // 没有 4³ 空区"才会没数据 —— 那意味着该处被实体包住，本来也不可见（不可见处由常量天光兜底）。
  return vec4<f32>(vec3<f32>(0.0), 0.0);
}

/// 单级烘焙：lod 由**派发该 pass 时确定**（见下面 4 个入口）。idx = 该级内的槽位下标。
///
/// 为什么按级拆成 4 个 compute pass，而不是一次 dispatch 覆盖全部槽位：
/// Step 2 的「粗级继承细级探针位置」要读**本帧**细级的 bake 输出（ddgi_cell / ddgi_cell_id）。
/// 同一个 pass 内没有顺序保证（连续 dispatch 也不构成屏障），粗级可能读到细级尚未写入的值；
/// 只有 pass 边界才是内存屏障。所以按细→粗拆成 4 个 pass，逐级建立依赖。
fn ddgi_bake_one(lod: u32, idx: u32) {
  if (idx >= ddgi_lod_count(lod)) { return; }
  let slot = ddgi_lod_slot_base(lod) + idx;
  let cs = ddgi_u.lods[lod].origin.w;
  // 本槽位「当前」覆盖的世界 cell（世界锚定环面映射的逆）。
  let wc = ddgi_slot_world_cell(lod, idx);
  let cmin = wc * cs;
  // 增量判据：
  //   ① 相机滚动 → 只有「世界 cell 键变了」的槽位重烘（键相同即续用，保留 meta 续龄）；
  //   ② 世界编辑 → 脏区 AABB 内的 cell 强制重烘（脏区 = CPU 上传时改动 chunk 的合并包围盒，
  //      不再是整块清缓存，所以一次小编辑只重算受影响的 cell）。
  let cell_lo = vec3<f32>(cmin);
  let cell_hi = cell_lo + f32(cs);
  let dirty = ddgi_u.dirty_min.w > 0.5
    && all(cell_hi > ddgi_u.dirty_min.xyz)
    && all(cell_lo < ddgi_u.dirty_max.xyz);
  let wcell = cmin / cs;
  let prev = ddgi_cell_id[slot];
  // 「同一世界 cell」= 本槽位没有换主。这是区分「编辑重烘」与「滚动换主」的**唯一**判据：
  // 槽位是世界锚定的环面映射（见 ddgi_slot_own 注释），相机没滚过本格时，编辑世界也**不会**
  // 改变 wcell。所以 wcell 未变 ⇒ 本次重烘只可能是「世界编辑」，不是「槽位换主」。
  let same_cell = prev.w != 0 && all(prev.xyz == wcell);
  if (!dirty && same_cell) { return; }
  ddgi_cell_id[slot] = vec4<i32>(wcell, 1);
  // 【保留】换主与编辑**都**保留图集历史（age / meta）。「同一 cell 只因世界编辑被重烘」**保留**
  // 探针的逻辑身份、位置最多在 cell 内微调，图集历史仍然有效。保留它，时域平均就能把新光照
  // **渐入**；反之若在这里清零，sort 会走直写一帧高方差估计、采样侧又因 MIN_SAMPLE_AGE 短暂跳过
  // 本针退到粗级 LOD —— 编辑后附近十几帧的光影闪烁正是这条链造成的。对齐 Douglas：编辑不重置
  // 累积量。
  // 【换主同样不丢弃图集历史】旧版在这里 `= 0u`（age 归零）→ 采样侧跳过 + collect 清零 →
  // 滚动时那一圈 cell 整片无数据 → 相机一移动就冒黑斑。但**滚动换主 = 平移一格**，新 cell
  // 与旧 cell 是紧邻邻居，旧读数是**极好的近似** —— 这与上面「编辑保留历史、靠 EMA 渐入」
  // 是同一个判断，没理由两种处理。真正的"内容已改变"只来自世界编辑。
  //
  // 【换主：age = 1 + 相位抖动 —— 目前观感最好的一档】取值实验（2026-09-12）走了五种：
  //   · age = SKIP_AGE   → converged=true → 探针**停止投线** → 旧值永久残留 → 灰色尾巴（最差）
  //   · age = SKIP_AGE-1 → 保证投线、alpha≈0.03，但换主当帧仍有可见跳变（浅灰带）
  //   · age = 1          → 无残留、收敛最快，但整圈换主的探针同帧全部直写 → 一条完整的浅灰带
  //   · age = 1+slot%N（本版）→ 把"直写"按**槽位号**错开到 N 帧：任何一帧只有 1/N 的换主探针
  //     在直写，且它们在空间上是散点（槽位号连续 ⇒ 世界 cell 相邻 ⇒ 取模后相位分散）→
  //     不再形成完整的一条带。
  //   · 【失败，别再试】解耦出 RESIDENT 位、让换主后走"稳态档慢收敛" —— 实测**更差**。原因：
  //     换主同时**改变了探针位置**（`ddgi_place_probe` 按新 cell 重算），而图集里还是旧位置的
  //     值；**位置变了、值不变 ⇒ 误差被"慢"放大而不是被抹平**。方向必须是让值**快速**跟上新
  //     位置（即论文说的 "speed transitions"），不是把过渡拉长。
  // age ∈ [1, N] 全部 < DDGI_SKIP_AGE（最小 12）→ converged 恒 false → **保证每帧投线** ✓
  // 注意：换主**不能清零图集**（那是黑斑的来源，见上一条注释）。
  // enabled/active 乐观置真：不等 sort 重建，避免"有图集却被判不活跃"的当帧空档。
  if (!same_cell) {
    ddgi_meta[slot] = ddgi_meta_pack(1u + (slot % DDGI_BAKE_STAGGER), true, true);
  }
  let g = make_grid(0u);
  let st = ddgi_cell_state_sized(g, cmin, cs);
  // 物体（非主世界 volume）不在主世界 brick tree 里 → 单独判：cell 与任一物体相交就
  // 当作「有体素」，否则物体所在的 cell 会被判成纯空气（探针永不判活 → 图集恒 0）。
  let obj_hit = ddgi_box_hits_object(cell_lo, cell_hi);
  if (st == 1u) {
    // 全满：无探针，但仍标记「有体素」供邻接 cell 判活
    ddgi_cell[slot] = ddgi_rec_pack(0u, 1u, vec3<u32>(0u));
    return;
  }
  // ---- 探针只在「离表面不超过 1 格」的 cell 里生成 ----
  // 【为什么必须有这条】Douglas 的探针网格是**每个模型一份**（字幕："I split every model up
  // into a grid"），所以空旷空气里根本不存在 cell。我们是在**整个世界 AABB** 上铺一张规则
  // 网格（为了世界锚定、相机移动不换主），于是开阔天空/场地里会长出**几十万个永远判不活**
  // （`near=false`）的悬空探针：白占图集、白扫 sort，而且 Probe Viz 看起来完全不像 Douglas
  // （大片悬空探针，而不是贴着几何的一层壳）。
  // 判据 = 本 cell 或 6 邻接 cell 有体素/物体 —— 与 sort 的活跃判定**同一个条件**，于是
  // "生成了 ⇒ 必定判活"，不再有"有探针却永远不投线"的槽位。
  // 【覆盖性】像素的 8 个角格都落在「其所属 cell 的 ±1 邻域」内：角格要么自己就是含表面的
  // 格，要么与含表面的格 6 邻接（斜向角格也至少有一个 6 邻接面格）→ 一定满足本条 →
  // 不会出现"该有针的地方没针"。
  let one = vec3<f32>(f32(cs));
  let obj_near = obj_hit || ddgi_box_hits_object(cell_lo - one, cell_hi + one);
  var near_surface = st != 0u || obj_near;
  if (!near_surface) {
    near_surface = ddgi_cell_state_sized(g, cmin + vec3<i32>(-cs, 0, 0), cs) != 0u
      || ddgi_cell_state_sized(g, cmin + vec3<i32>(cs, 0, 0), cs) != 0u
      || ddgi_cell_state_sized(g, cmin + vec3<i32>(0, -cs, 0), cs) != 0u
      || ddgi_cell_state_sized(g, cmin + vec3<i32>(0, cs, 0), cs) != 0u
      || ddgi_cell_state_sized(g, cmin + vec3<i32>(0, 0, -cs), cs) != 0u
      || ddgi_cell_state_sized(g, cmin + vec3<i32>(0, 0, cs), cs) != 0u;
  }
  if (!near_surface) {
    // 纯空气且一格之内没有任何几何 → 不生成探针。OCCUPIED 记 0（本格确实没有体素）。
    ddgi_cell[slot] = ddgi_rec_pack(0u, 0u, vec3<u32>(0u));
    return;
  }
  let occupied = select(0u, 1u, st != 0u || obj_hit);
  let center = vec3<f32>(cmin) + f32(cs) * 0.5;
  // 放置规则（严格对齐 Douglas #23）：见 `ddgi_place_probe` —— 从粗到细取「第一个完全空的
  // 子块」的中心；同级并列取离 cell 中心最近的。全空 cell 的最大空叶就是整格 → 探针落在正中
  // （Douglas："for a totally empty cell the probe would end up right in the center"）。
  // 全满 cell 已在上面 `st == 1u` 处返回、不放探针。
  //
  // 【为什么删掉了"中心捷径"】旧版有一条捷径：cell 中心落在完全空的 16³ 子块内就直接用中心。
  // 对 LOD0（cs=16 ⇒ 子块 = 整格）它恰好等价于本函数轮 1 的"整格全空 → 居中"，删掉不改结果；
  // 对 LOD1~3（cs≥32）它的 `off = center_v - sub16` 恒为 0 → 旧净距门槛下**从不触发**，于是
  // 粗级总能先走 `ddgi_place_probe` 的"继承细级探针位置"。若只去掉净距项而不删捷径，LOD1~3
  // 会开始走捷径、绕开继承 → 粗级不再与细级自洽。所以整条删掉，放置只留 `ddgi_place_probe`
  // 这一条 Douglas 规则。
  let r = ddgi_place_probe(g, cmin, cs, center, lod);
  if (r.w <= 0.5) { ddgi_cell[slot] = ddgi_rec_pack(0u, occupied, vec3<u32>(0u)); return; }
  var p = r.xyz;
  // 主世界的 place_probe 看不到物体体素：探针可能正落在物体内部（射线起点即实心，
  // 首命中 t≈0，辐亮度/深度全错）→ 推到物体 AABB 外。
  p = ddgi_push_out_of_objects(p);
  let u = clamp((p - vec3<f32>(cmin)) / f32(cs), vec3<f32>(0.0), vec3<f32>(1.0));
  let off_b = vec3<u32>(clamp(round(u * 255.0), vec3<f32>(0.0), vec3<f32>(255.0)));
  ddgi_cell[slot] = ddgi_rec_pack(1u, occupied, off_b);
}
