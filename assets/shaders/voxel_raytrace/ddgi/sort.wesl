import package::bindings::{ddgi_cell, ddgi_u, g, grid_descs};
import package::ddgi::consts::{DDGI_META_NBR_MASK};
import package::ddgi::helpers::{DDGI_REC_OCCUPIED, ddgi_cell_in_window, ddgi_cell_state_sized, ddgi_nbr_dir, ddgi_slot_own};
import package::trace::{make_grid};
import package::world::{world_cfg_full, world_raycast};

// ============================================================================
// 阶段一 sort：探针活跃性判定 + worklist 压缩。1D dispatch，每线程一个 (lod, 世界 cell) slot。
// 活跃 = 有探针 && (本 cell 或 6 邻接有体素 || 与物体 AABB 重叠) 且不落在更细 LOD 覆盖范围内。
// ============================================================================
// ============================================================================
// 非主世界 volume（物体）辅助：物体体素只在 grid_descs[i≥1]（独立 brick tree + 变换）里，须单独遍历其 AABB。
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

/// 邻接 cell 是否有体素（cell 用世界 cell 号）。
/// 规则网格级：查邻接槽位的 OCCUPIED 位（那些级铺满，邻接 cell 一定有槽位）。
/// LOD0 是逐砖认领的：过道 cell 的邻接（墙）常常落在**没认领**的砖里，槽位查不到 ⇒ 必须兜底直接
/// 查世界几何，否则纯空气的过道 cell 永远判不活（表现 = 有探针但不发线、GI 全黑）。
fn ddgi_neighbor_occupied(lod: u32, wc: vec3<i32>) -> bool {
  if (ddgi_cell_in_window(lod, wc))
    && (ddgi_cell[ddgi_slot_own(lod, wc)] & DDGI_REC_OCCUPIED) != 0u {
    return true;
  }
  if (lod != 0u) { return false; }
  let cs = ddgi_u.lods[0u].origin.w;
  return ddgi_cell_state_sized(make_grid(0u), wc * cs, cs) != 0u;
}

/// 6 邻接可见性掩码：从探针位置沿 6 轴向各走向邻居 cell 中心，不撞几何则该位为 1。
/// 起点是探针本身、终点是邻居 cell 中心；长度必须按探针实际偏移算，不能恒取 cs。
fn ddgi_nbr_vis_mask(probe_pos: vec3<f32>, cmin: vec3<f32>, cs: f32) -> u32 {
  let gw = make_grid(0u);
  let cfg = world_cfg_full(0.0, 3u);
  let center = cmin + vec3<f32>(cs * 0.5);
  var mask: u32 = 0u;
  for (var a = 0u; a < 6u; a = a + 1u) {
    let seg = center + ddgi_nbr_dir(a) * cs - probe_pos;
    let len = length(seg);
    if (len <= 1e-4) { continue; }
    let hit = world_raycast(gw, probe_pos, seg / len, len, cfg);
    if (!hit.hit) { mask = mask | (1u << a); }
  }
  return mask & DDGI_META_NBR_MASK;
}
