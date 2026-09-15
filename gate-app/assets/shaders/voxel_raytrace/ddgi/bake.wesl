import package::bindings::{ddgi_cell, ddgi_cell_id, ddgi_meta, ddgi_u, g};
import package::brickmap::{Grid};
import package::ddgi::consts::{DDGI_BAKE_STAGGER};
import package::ddgi::helpers::{DDGI_REC_ENABLED, ddgi_brick_state, ddgi_cell_in_window, ddgi_cell_state_sized, ddgi_lod_cell_size, ddgi_lod_count, ddgi_lod_slot_base, ddgi_meta_pack, ddgi_rec_off, ddgi_rec_pack, ddgi_slot_own, ddgi_slot_world_cell};
import package::ddgi::sort::{ddgi_box_hits_object, ddgi_push_out_of_objects};
import package::trace::{make_grid};

/// 对齐的 8³ 子块是否**完全空**（= 它的 8 个 4³ 子节点全空）。
///
/// brickmap 的分裂因子是 4（256³→64³→16³→4³→1³），没有原生 8³ 节点 ⇒ 这里聚合 2³ 个 4³ 节点。
/// 8³ 是 16³ 与 4³ 之间的一档，缺它则"优先选最大的空子区域"会在两级之间漏档。
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
/// 返回 (探针位置, found)。调用前提：调用方已挡住全满 cell（全满 cell 不放探针）。
/// 逐级找**第一个**含"完全空子块"的层级，探针放在该子块中心；同级并列取离 cell 中心最近的。
/// lod 必须由调用方传入，不可从 cs 反推（各级 cell 边长不保证等比）。
fn ddgi_place_probe(g: Grid, cmin: vec3<i32>, cs: i32, center: vec3<f32>, lod: u32) -> vec4<f32> {
  let n16 = max(cs / 16, 1);
  // ---- Step 2：粗级 LOD 优先继承细级探针的位置 ----
  // 细级 cell 边长 = cs/2 ⇒ 本 cell 覆盖 wc*2 + (0|1)³ 这 8 个细级 cell，取其中离本 cell
  // 中心最近的已放置探针位置（找不到才退回下面的启发式）。wc 由 center 反推（floor，负数也
  // 正确）；lod 必须由调用方传入，各级 cell 边长不保证等比、不能从 cs 反推。
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
          // 读细级本帧的 bake 输出（ddgi_cell / ddgi_cell_id）：bake 按 per-LOD 分 pass，
          // pass 边界才是内存屏障 ⇒ 这里读到的正是本帧细级刚算出的结果；改用 ddgi_slot_pos
          // 会拿到与细级不同帧的位置（相机滚动时可能落在实体内部）。
          // 细级该 cell 必须可编址（LOD0 下 = 其 chunk 已领段，否则 ddgi_slot_own 返回 0）。
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
  // 轮 1（sz = 64/32/16）：从大到小取**第一个**含全空子块的粒度，同级内取离 cell 中心最近的。
  // 必须从大到小：空子块越大，其中心离表面越远 —— 否则（如"薄板地面 + 下方也是空气"的 cell）
  // 探针会落在地表下方，对朝上的地面 8 个角的 wn 全 ≤ 0 而被硬剔，该片 GI 纯黑。
  // cs 本身也在 sz 列表里 ⇒ 全空 cell 的最大空叶即整格，探针落在正中。
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
  // ---- 三级都没找到全空子块 → 不放探针：返回 w = 0，该 cell ENABLED = 0，采样侧按"无探针"
  // 剔除该角、改用周围 cell 的探针。格内自由空间全是 <4 体素宽的窄缝/接缝（进不了光，塞进去的
  // 探针辐照度恒 ≈0，会把缝两侧稳定压成纯黑窄带）。
  // 覆盖性：三线性 stencil 跨 8 个 cell，个别 cell 无针不会留洞。
  return vec4<f32>(vec3<f32>(0.0), 0.0);
}

/// 单级烘焙：lod 由派发该 pass 时确定，idx = 该级内的槽位下标。
/// 按细→粗拆成 4 个 pass：Step 2 要读本帧细级的 bake 输出，而同一 pass 内没有顺序保证
/// （连续 dispatch 也不构成屏障），只有 pass 边界才是内存屏障。
fn ddgi_bake_one(lod: u32, idx: u32) {
  if (idx >= ddgi_lod_count(lod)) { return; }
  let slot = ddgi_lod_slot_base(lod) + idx;
  let cs = ddgi_u.lods[lod].origin.w;
  // 本槽位「当前」覆盖的世界 cell（世界锚定环面映射的逆）。
  let wc = ddgi_slot_world_cell(lod, idx);
  let cmin = wc * cs;
  // 重烘判据：本槽位换主（世界 cell 键变了）或落在脏区 AABB 内（dirty_min.w > 0.5 表示本帧
  // 确有世界数据上传，脏区 = 本帧改动 chunk 的合并包围盒）；否则本帧直接跳过、保留旧值。
  let cell_lo = vec3<f32>(cmin);
  let cell_hi = cell_lo + f32(cs);
  let dirty = ddgi_u.dirty_min.w > 0.5
    && all(cell_hi > ddgi_u.dirty_min.xyz)
    && all(cell_lo < ddgi_u.dirty_max.xyz);
  let wcell = cmin / cs;
  let prev = ddgi_cell_id[slot];
  // 槽位是世界锚定的环面映射 ⇒ 相机没滚过本格时，编辑世界**不会**改变 wcell。
  // 故 wcell 未变（且 prev.w != 0）即"本槽位没有换主"，本次重烘只可能出自世界编辑。
  let same_cell = prev.w != 0 && all(prev.xyz == wcell);
  if (!dirty && same_cell) { return; }
  ddgi_cell_id[slot] = vec4<i32>(wcell, 1);
  // 换主与编辑**都**保留图集历史（age / meta）：清零会让 sort 直写一帧高方差估计、采样侧又因
  // MIN_SAMPLE_AGE 跳过本针 ⇒ 黑斑与闪烁。滚动换主只是平移一格，新 cell 与旧 cell 紧邻，旧读数
  // 是很好的近似，靠 EMA 渐入即可。
  // 换主时 age = 1 + slot % DDGI_BAKE_STAGGER：把"当帧直写"按槽位号错开到 N 帧，避免整圈换主
  // 探针同帧全部直写形成一条完整的浅灰带（槽位号连续 ⇒ 世界 cell 相邻 ⇒ 取模后相位分散）。
  // age ∈ [1, N] 恒 < DDGI_SKIP_AGE（最小 12）⇒ converged 为假 ⇒ 保证每帧投线。
  // enabled/active 乐观置真，不等 sort 重建，避免"有图集却被判不活跃"的当帧空档。
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
  // ---- 探针只在「本 cell 或 6 邻接 cell 有体素/物体」的 cell 里生成 ----
  // 世界网格铺满整个世界 AABB（为世界锚定），不设这条判据就会在开阔天空里生出几十万个永远
  // 判不活的悬空探针（白占图集、白扫 sort）。
  // 判据与 sort 的活跃判定同一个条件 ⇒ "生成了就必定判活"。
  // 覆盖性：像素的 8 个角格都落在"其所属 cell 的 ±1 邻域"内（角格要么自己含表面，要么与含
  // 表面的格 6 邻接）⇒ 不会出现该有针的地方没针。
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
  // 放置规则见 `ddgi_place_probe`：从粗到细取第一个完全空子块的中心，同级并列取离 cell 中心
  // 最近的；全空 cell 的最大空叶即整格 ⇒ 探针落在正中。全满 cell 已在上面 `st == 1u` 处返回。
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
