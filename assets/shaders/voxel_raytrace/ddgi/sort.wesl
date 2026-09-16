import package::bindings::{ddgi_cell, g, grid_descs};
import package::ddgi::consts::{DDGI_META_NBR_MASK};
import package::ddgi::helpers::{DDGI_REC_OCCUPIED, ddgi_cell_in_window, ddgi_nbr_dir, ddgi_slot_own};
import package::trace::{make_grid};
import package::world::{world_cfg_full, world_raycast};

// ============================================================================
// 阶段一 sort：探针活跃性判定 + worklist 压缩。1D dispatch，每线程一个 (lod, 世界 cell) slot。
//   ① 读烘焙记录 ddgi_cell[slot]（探针是否存在 / 本 cell 是否有体素 / cell 内偏移）
//   ② 活跃 = 有探针 && (本 cell 有体素 || 6 邻接 cell 任一有体素 || 与物体 AABB 重叠)
//      且不落在更细 LOD 覆盖范围内（避免各 LOD 重复投线）
//   ③ 本帧投线：used（最近被屏幕采样过）∪ !converged（新探针/换主）∪ due（按槽位错峰刷新）
//   ④ 幸存者 atomicAdd 进 per-LOD worklist（item = 探针世界坐标 + age/lod）
//   ⑤ 全 slot 写 meta（age/enabled）与 slot_pos
// ============================================================================
// ============================================================================
// 非主世界 volume（物体）辅助。
// 物体的体素只在 grid_descs[i≥1]（各自独立的 brick tree + 变换）里，主世界 brick tree
// 完全没有它们 ⇒ 判占用/判活/把探针推出物体都必须单独遍历物体 AABB。
// ============================================================================
/// 世界 AABB [lo, hi) 是否与任一物体的世界 AABB 相交（cell 粒度判占用/判活）
fn ddgi_box_hits_object(lo: vec3<f32>, hi: vec3<f32>) -> bool {
  for (var i = 1u; i < g.grid_count; i = i + 1u) {
    let d = grid_descs[i];
    if (all(hi > d.aabb_min.xyz) && all(lo < d.aabb_max.xyz)) { return true; }
  }
  return false;
}
/// 把点推到所有物体 AABB 之外：沿「离得最近的那个面」推 1 voxel。
/// 探针落在物体内时射线起点就在实心里（首命中 t≈0）→ 该探针的辐亮度/深度全错。
fn ddgi_push_out_of_objects(p0: vec3<f32>) -> vec3<f32> {
  var p = p0;
  for (var i = 1u; i < g.grid_count; i = i + 1u) {
    let d = grid_descs[i];
    let mn = d.aabb_min.xyz;
    let mx = d.aabb_max.xyz;
    if (all(p >= mn) && all(p < mx)) {
      let dlo = p - mn;
      let dhi = mx - p;
      let near_lo = min(dlo, dhi);
      if (near_lo.x <= near_lo.y && near_lo.x <= near_lo.z) {
        p.x = select(mn.x - 1.0, mx.x + 1.0, dhi.x < dlo.x);
      } else if (near_lo.y <= near_lo.z) {
        p.y = select(mn.y - 1.0, mx.y + 1.0, dhi.y < dlo.y);
      } else {
        p.z = select(mn.z - 1.0, mx.z + 1.0, dhi.z < dlo.z);
      }
    }
  }
  return p;
}

/// 邻接 cell 是否有体素（cell 用**世界 cell 号**；不在本级窗口内视为无）
fn ddgi_neighbor_occupied(lod: u32, wc: vec3<i32>) -> bool {
  if (!ddgi_cell_in_window(lod, wc)) { return false; }
  return (ddgi_cell[ddgi_slot_own(lod, wc)] & DDGI_REC_OCCUPIED) != 0u;
}

/// 6 邻接可见性掩码：从探针位置沿 6 个轴向各走 1 个 cell（长度 = cs），**不撞几何**则该位为 1。
/// 空间混合（`ddgi_probe_blend_irr`）用它门控：跨墙的那一位被清掉，墙外探针的天空辐照度不会
/// 被抹到墙内。
/// 起点必须是探针本身（探针由 `ddgi_place_probe` 放在空叶内 ⇒ t>0 的任何命中都是真遮挡；
/// 沿轴内缩会漏掉紧贴探针的薄墙）。终点取邻居 cell 中心，故长度 = cs —— 实心邻居的射线会在
/// 其内部命中，该位为 0（被遮挡）。
fn ddgi_nbr_vis_mask(probe_pos: vec3<f32>, cs: f32) -> u32 {
  let gw = make_grid(0u);
  let cfg = world_cfg_full(0.0, 3u);
  var mask: u32 = 0u;
  for (var a = 0u; a < 6u; a = a + 1u) {
    let hit = world_raycast(gw, probe_pos, ddgi_nbr_dir(a), cs, cfg);
    if (!hit.hit) { mask = mask | (1u << a); }
  }
  return mask & DDGI_META_NBR_MASK;
}

