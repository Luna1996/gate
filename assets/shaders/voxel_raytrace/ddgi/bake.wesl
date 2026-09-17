import package::bindings::{ddgi_cell, ddgi_cell_id, ddgi_meta, ddgi_u, g};
import package::brickmap::{Grid};
import package::ddgi::consts::{DDGI_BAKE_STAGGER, DDGI_PROBE_MIN_BLOCK};
import package::ddgi::helpers::{DDGI_REC_ENABLED, ddgi_brick_state, ddgi_cell_in_window, ddgi_cell_state_sized, ddgi_lod_cell_size, ddgi_lod_count, ddgi_lod_slot_base, ddgi_meta_pack, ddgi_rec_off, ddgi_rec_pack, ddgi_slot_own, ddgi_slot_world_cell};
import package::ddgi::sort::{ddgi_box_hits_object, ddgi_push_out_of_objects};
import package::trace::{make_grid};
import package::world::{world_solid};

/// 体素粒度的 `sz³` 全空块（不要求对齐 4 voxel 网格）；探针放置的最后一档。
fn ddgi_block_empty(g: Grid, o: vec3<i32>, sz: i32) -> bool {
  for (var k = 0; k < sz; k = k + 1) {
    for (var j = 0; j < sz; j = j + 1) {
      for (var i = 0; i < sz; i = i + 1) {
        if (world_solid(g, o + vec3<i32>(i, j, k))) { return false; }
      }
    }
  }
  return true;
}

/// 对齐的 8³ 子块是否完全空。
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

/// 体素粒度的最后一档：在 [cmin, cmin+cs) 内找离 `center` 最近的 `DDGI_PROBE_MIN_BLOCK³` 全空块，
/// 返回 (位置, found)。窄过道截面对齐不上 4 voxel 网格，只靠这一档。
fn ddgi_place_fine(g: Grid, cmin: vec3<i32>, cs: i32, center: vec3<f32>) -> vec4<f32> {
  let msz = DDGI_PROBE_MIN_BLOCK;
  let n16 = max(cs / 16, 1);
  let step2 = max(cs / 16, 1);
  // 快速路径：`center` 自身所在的块（clamp 进 cell 内）—— 绝大多数调用就是它，省掉整轮扫描。
  let base = clamp(
    vec3<i32>(floor(center)) - vec3<i32>(msz / 2),
    cmin,
    cmin + vec3<i32>(cs - msz),
  );
  if (ddgi_block_empty(g, base, msz)) {
    return vec4<f32>(vec3<f32>(base) + vec3<f32>(f32(msz) * 0.5), 1.0);
  }
  var bm_d2 = 1e30;
  var bm_p = vec3<f32>(0.0);
  var fm = false;
  for (var k = 0; k < n16; k = k + 1) {
    for (var j = 0; j < n16; j = j + 1) {
      for (var i = 0; i < n16; i = i + 1) {
        let sub16 = cmin + vec3<i32>(i, j, k) * 16;
        if (ddgi_cell_state_sized(g, sub16, 16) != 2u) { continue; }
        for (var kk = 0; kk + msz <= 16; kk = kk + step2) {
          for (var jj = 0; jj + msz <= 16; jj = jj + step2) {
            for (var ii = 0; ii + msz <= 16; ii = ii + step2) {
              let sub = sub16 + vec3<i32>(ii, jj, kk);
              // 中心体素先测：缝/洞里的候选一次淘汰。
              if (world_solid(g, sub + vec3<i32>(msz / 2))) { continue; }
              if (!ddgi_block_empty(g, sub, msz)) { continue; }
              let p = vec3<f32>(sub) + vec3<f32>(f32(msz) * 0.5);
              let d2 = dot(p - center, p - center);
              if (d2 < bm_d2) { bm_d2 = d2; bm_p = p; fm = true; }
            }
          }
        }
      }
    }
  }
  return vec4<f32>(select(vec3<f32>(0.0), bm_p, fm), select(0.0, 1.0, fm));
}

/// 在 [cmin, cmin+cs) 内按 16³ → 8³ → 4³ → DDGI_PROBE_MIN_BLOCK³ 找靠 `center` 的空块，返回 (探针位置, found)，全找不到则 found=false。
/// 调用前提：调用方已挡住全满 cell；lod 必须由调用方传入，不可从 cs 反推。
fn ddgi_place_probe(g: Grid, cmin: vec3<i32>, cs: i32, center: vec3<f32>, lod: u32) -> vec4<f32> {
  let n16 = max(cs / 16, 1);
  // ---- Step 2：粗级 LOD 优先继承细级探针的位置 ----
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
          // 读细级本帧的 bake 输出（ddgi_cell / ddgi_cell_id）：bake 按 per-LOD 分 pass，pass 边界是内存屏障。
          // 细级该 cell 必须可编址（LOD0 下 = 其 chunk 已领段）。
          if (!ddgi_cell_in_window(flod, fwc)) { continue; }
          let fslot = ddgi_slot_own(flod, fwc);
          // 该槽位必须当前确实覆盖 fwc，且本帧 bake 已写过它（cell_id.w != 0）。
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
  // 轮 1（sz = 64/32/16）：从大到小取第一个含全空子块的粒度，同级内取离 cell 中心最近者。
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
  // 轮 2（8³）：在混合 16³ 内找最近的全空 8³。
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
  // 轮 3（4³）：在混合 16³ 内找最近的全空 4³。
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
  // 轮 4（体素粒度）：4³ 要求块对齐 4 voxel 网格，窄过道截面对齐不上 → 交给 `ddgi_place_fine`。
  // 都没找到 → w = 0，该 cell ENABLED = 0（比 DDGI_PROBE_MIN_BLOCK 更窄的缝不放探针）。
  return ddgi_place_fine(g, cmin, cs, center);
}

/// 单级烘焙：lod 由派发该 pass 时确定，idx = 该级内的槽位下标。
/// 按细→粗拆成 4 个 pass（同一 pass 内无顺序保证，只有 pass 边界是内存屏障）。
fn ddgi_bake_one(lod: u32, idx: u32) {
  if (idx >= ddgi_lod_count(lod)) { return; }
  let slot = ddgi_lod_slot_base(lod) + idx;
  let cs = ddgi_u.lods[lod].origin.w;
  let wc = ddgi_slot_world_cell(lod, idx);
  let cmin = wc * cs;
  // 重烘判据：本槽位换主，或落在脏区 AABB 内（dirty_min.w > 0.5 = 本帧有世界数据上传）。
  let cell_lo = vec3<f32>(cmin);
  let cell_hi = cell_lo + f32(cs);
  let dirty = ddgi_u.dirty_min.w > 0.5
    && all(cell_hi > ddgi_u.dirty_min.xyz)
    && all(cell_lo < ddgi_u.dirty_max.xyz);
  let wcell = cmin / cs;
  let prev = ddgi_cell_id[slot];
  // wcell 未变（且 prev.w != 0）即本槽位没有换主。
  let same_cell = prev.w != 0 && all(prev.xyz == wcell);
  if (!dirty && same_cell) { return; }
  ddgi_cell_id[slot] = vec4<i32>(wcell, 1);
  // 换主与编辑都保留图集历史（age / meta）。
  // 换主时 age = 1 + slot % DDGI_BAKE_STAGGER（∈ [1, N]）；enabled/active 置真。
  if (!same_cell || dirty) {
    ddgi_meta[slot] = ddgi_meta_pack(1u + (slot % DDGI_BAKE_STAGGER), true, true);
  }
  let g = make_grid(0u);
  let st = ddgi_cell_state_sized(g, cmin, cs);
  // 物体不在主世界 brick tree 里 → 单独判：cell 与任一物体相交就当作「有体素」。
  let obj_hit = ddgi_box_hits_object(cell_lo, cell_hi);
  if (st == 1u) {
    // 全满：无探针，但标记「有体素」供邻接 cell 判活。
    ddgi_cell[slot] = ddgi_rec_pack(0u, 1u, vec3<u32>(0u));
    return;
  }
  // ---- 探针只在本 cell 或 6 邻接 cell 有体素/物体时生成 ----
  // 判据与 sort 的活跃判定同一个条件（生成即判活）。
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
    // 纯空气且一格内无几何 → 不生成探针，OCCUPIED = 0。
    ddgi_cell[slot] = ddgi_rec_pack(0u, 0u, vec3<u32>(0u));
    return;
  }
  let occupied = select(0u, 1u, st != 0u || obj_hit);
  let center = vec3<f32>(cmin) + f32(cs) * 0.5;
  // 探针放置在从粗到细第一个完全空子块的中心（同级并列取最近者）。
  let r = ddgi_place_probe(g, cmin, cs, center, lod);
  if (r.w <= 0.5) { ddgi_cell[slot] = ddgi_rec_pack(0u, occupied, vec3<u32>(0u)); return; }
  var p = r.xyz;
  // 探针可能落在物体内部 → 推到物体 AABB 外。
  p = ddgi_push_out_of_objects(p);
  let u = clamp((p - vec3<f32>(cmin)) / f32(cs), vec3<f32>(0.0), vec3<f32>(1.0));
  let off_b = vec3<u32>(clamp(round(u * 255.0), vec3<f32>(0.0), vec3<f32>(255.0)));
  ddgi_cell[slot] = ddgi_rec_pack(1u, occupied, off_b);
}
