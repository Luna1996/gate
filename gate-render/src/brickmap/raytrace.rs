//! CPU 侧**唯一**的体素射线求交入口：[`raycast`]。
//!
//! 单根射线，**没有 beam 预 pass**：beam 的收益来自"相邻像素共享一个最近命中 t"，
//! CPU 侧一次只有一根射线（编辑落笔 / 拾取），预 pass 只会白烧一遍遍历。
//!
//! 与 GPU `trace.wesl` 同构的四层结构：
//! 1. **局部 AABB slab 剔除**：物体先过世界 AABB，再把射线变换到局部系；
//! 2. **chunk 间 256³ A&W 步进**：`chunk()` 无表项 = 整块空气，直接跳过（不展开树）；
//! 3. **chunk 内层次栈式 mask DDA**：节点掩码拿在手里（`NodeDesc`），4³ 子块之间步进**零重查**；
//!    跨 brick 后用 `firstTrailingBit` 一次跳到最粗可行层；
//! 4. **方向可达掩码 LUT**（Douglas #18）：`palette == 0` 的节点用 `mask & reach` 整砖剔除（LUT 是可达集的
//!    保守超集 ⇒ 绝不漏命中）；uniform 子块整块命中 / 整块跳过。
//!
//! 数据源是**权威** `VolumeGrid`（编辑层那份），不是 GPU 的序列化 buffer —— 调用方不必先
//! `build_full`（castle.vox 的 87MB 树序列化 ≈ 1.7s，每次点击都付不起）。
//! 代价：节点描述按需从根下钻（`ChunkTree::node_desc`，O(深度)；GPU 侧那一步是一次 buffer 读，同阶）。
//! 仅在**跨层**时下钻；同一节点内的 4³ 子块步进只查手里那份 mask，与 shader 的"零 load"口径一致。

use std::sync::OnceLock;

use glam::{IVec3, Vec3};

use gate_voxel::{CHUNK_SIZE, ChunkCoord, ChunkTree, VolumeGrid, VolumeTransform, Volumes};

use crate::brickmap::wire::{MARCH_MASK_ENTRIES, march_mask_lut_words};

/// 迭代安全网（与 GPU `trace.wesl` 的 budget 同值）：只有退化射线会撞到它。
const BUDGET: u32 = 65536;

/// 命中记录（`t` 为世界标尺；`voxel` / `face` 是该 volume 的局部系）
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RayHit {
  /// 自 `origin` 沿 `dir` 的距离（voxel 单位；`dir` 需归一化）
  pub t: f32,
  /// 命中体素的调色板槽
  pub pal: u16,
  /// 命中体素（该 volume 的局部体素坐标）
  pub voxel: IVec3,
  /// 入面法线（轴对齐整数，体积局部系）：指向射线来向 ⇒ `voxel + face` 是前方那格
  pub face: IVec3,
  /// 入面法线（世界系；物体命中已过变换）
  pub normal: Vec3,
  /// `-1` = 主世界；`0..` = 物体（`Volumes::list[1..]` 下标）
  pub obj_id: i32,
}

/// 单根射线 × 场景：逐 volume 求最近命中（同 GPU `trace_scene`），命中后收紧后续 volume 的 t 上限。
/// `dir` 需归一化（`t` 才是 voxel 距离）；`t_max` 可取 `f32::INFINITY`（世界外的部分由各 volume 的
/// 占用窗口 AABB 截掉）。
pub fn raycast(volumes: &Volumes, origin: Vec3, dir: Vec3, t_max: f32) -> Option<RayHit> {
  let mut best: Option<RayHit> = None;
  for grid in &volumes.list {
    let cap = best.as_ref().map_or(t_max, |b| b.t.min(t_max));
    let is_world = grid.obj_id == -1;
    if let Some(hit) = trace_volume(grid, grid.transform, is_world, origin, dir, cap)
      && best.as_ref().is_none_or(|b| hit.t < b.t)
    {
      best = Some(hit);
    }
  }
  best
}

/// 单 volume 求交：世界 AABB（仅物体）→ 局部占用窗口 slab → chunk 间 A&W → chunk 内层次 DDA。
fn trace_volume(
  grid: &VolumeGrid,
  tr: VolumeTransform,
  is_world: bool,
  origin: Vec3,
  dir: Vec3,
  t_cap: f32,
) -> Option<RayHit> {
  // 主世界 = identity：射线即世界射线；物体：先世界 AABB 预剔除，再变换到局部系
  // （`rd` 含 1/scale ⇒ t 标尺两侧一致，返回的 t 就是世界距离）。
  let (ro, rd, t_hi_cap) = if is_world {
    (origin, dir, t_cap)
  } else {
    let (w_mn, w_mx) = tr.world_aabb();
    let (t_enter, t_exit) = slab_box(origin, dir, w_mn, w_mx, 0.0, t_cap);
    if t_exit <= t_enter.max(0.0) || t_enter >= t_cap {
      return None;
    }
    let wp = origin - tr.pos;
    let ro =
      Vec3::new(wp.dot(tr.rot.x_axis), wp.dot(tr.rot.y_axis), wp.dot(tr.rot.z_axis)) / tr.scale;
    let rd =
      Vec3::new(dir.dot(tr.rot.x_axis), dir.dot(tr.rot.y_axis), dir.dot(tr.rot.z_axis)) / tr.scale;
    (ro, rd, t_exit.min(t_cap))
  };
  // ---- 局部占用窗口（voxel AABB）：有 chunk 的区块外包。世界之外恒为空气 ----
  let (lo, hi) = occupied_window(grid)?;
  let (tl_enter, tl_exit) = slab_box(ro, rd, lo, hi, 0.0, t_hi_cap);
  let tl0 = tl_enter.max(0.0);
  let tl1 = tl_exit.min(t_hi_cap);
  if tl1 <= tl0 {
    return None;
  }
  let ro_a = [ro.x, ro.y, ro.z];
  let rd_a = [rd.x, rd.y, rd.z];
  let sign = [
    if rd.x >= 0.0 { 1 } else { -1 },
    if rd.y >= 0.0 { 1 } else { -1 },
    if rd.z >= 0.0 { 1 } else { -1 },
  ];
  // 单轴 t 增量（voxel）；零分量给 1e30 ⇒ 该轴永不被选中
  let delta = [
    if rd.x.abs() > 1e-30 { rd.x.abs().recip() } else { 1e30 },
    if rd.y.abs() > 1e-30 { rd.y.abs().recip() } else { 1e30 },
    if rd.z.abs() > 1e-30 { rd.z.abs().recip() } else { 1e30 },
  ];
  let delta_c =
    [delta[0] * CHUNK_SIZE as f32, delta[1] * CHUNK_SIZE as f32, delta[2] * CHUNK_SIZE as f32];
  // ---- chunk 间 256³ A&W（起点推进到窗口入口：外面那段一定打不到东西）----
  let start = ro + rd * tl0;
  let mut ci = [
    (start.x / CHUNK_SIZE as f32).floor() as i32,
    (start.y / CHUNK_SIZE as f32).floor() as i32,
    (start.z / CHUNK_SIZE as f32).floor() as i32,
  ];
  let mut tmax_c = [1e30f32; 3];
  for i in 0..3 {
    if rd_a[i].abs() > 1e-30 {
      let side = if sign[i] >= 0 { 1.0 } else { 0.0 };
      let bnd = (ci[i] as f32 + side) * CHUNK_SIZE as f32;
      tmax_c[i] = ((bnd - ro_a[i]) / rd_a[i]).max(tl0);
    }
  }
  let mut t_enter_c = tl0;
  // 首 chunk 进入面：`-rd` 的主轴（相机贴面 / 自窗口外射入；嵌在实体内属 UB）
  let mut entry_face = face_index_from_normal(-rd.normalize_or_zero());
  let span_c = ((hi - lo) / CHUNK_SIZE as f32).ceil();
  let budget = (span_c.x + span_c.y + span_c.z) as u32 * 3 + 16;
  for _ in 0..budget {
    let t_exit_c = tmax_c[0].min(tmax_c[1]).min(tmax_c[2]);
    // 窗口外 / 无 chunk 表项 = 空气块，不展开
    if let Some(tree) = grid.chunk(ChunkCoord::new(ci[0], ci[1], ci[2])) {
      let s = CHUNK_SIZE as f32;
      let chunk_min = [ci[0] as f32 * s, ci[1] as f32 * s, ci[2] as f32 * s];
      let t1 = t_exit_c.min(tl1);
      if let Some((t, pal, face_id, v)) =
        trace_chunk(tree, chunk_min, ro_a, rd_a, sign, t_enter_c, t1, entry_face)
      {
        let voxel = IVec3::new(v[0], v[1], v[2]) + IVec3::new(ci[0], ci[1], ci[2]) * CHUNK_SIZE;
        let f = face_normal_from_index(face_id);
        let normal = if is_world { f } else { (tr.rot * f).normalize_or_zero() };
        return Some(RayHit { t, pal, voxel, face: f.as_ivec3(), normal, obj_id: grid.obj_id });
      }
    }
    if t_exit_c >= tl1 {
      break;
    }
    if tmax_c[0] <= tmax_c[1] && tmax_c[0] <= tmax_c[2] {
      t_enter_c = tmax_c[0];
      tmax_c[0] += delta_c[0];
      ci[0] += sign[0];
      entry_face = if sign[0] < 0 { 1 } else { 0 };
    } else if tmax_c[1] <= tmax_c[2] {
      t_enter_c = tmax_c[1];
      tmax_c[1] += delta_c[1];
      ci[1] += sign[1];
      entry_face = if sign[1] < 0 { 3 } else { 2 };
    } else {
      t_enter_c = tmax_c[2];
      tmax_c[2] += delta_c[2];
      ci[2] += sign[2];
      entry_face = if sign[2] < 0 { 5 } else { 4 };
    }
  }
  None
}

/// 单 chunk 内整数体素层级 DDA（镜像 WGSL `trace_chunk_opaque`）。
/// `chunk_min` = chunk 原点（局部 voxel）；`[t0, t1]` 与返回 t 为 ro 系绝对 t；`entry_face` = 进入面。
/// 返回 `(t, palette, face_id, chunk 局部体素)`。
#[allow(clippy::too_many_arguments)]
fn trace_chunk(
  tree: &ChunkTree,
  chunk_min: [f32; 3],
  ro: [f32; 3],
  rd: [f32; 3],
  sign: [i32; 3],
  t0: f32,
  t1: f32,
  entry_face: u8,
) -> Option<(f32, u16, u8, [i32; 3])> {
  // 擦边退化（射线只蹭到 chunk 边界）
  if t0 >= t1 {
    return None;
  }
  let ro_c = [ro[0] - chunk_min[0], ro[1] - chunk_min[1], ro[2] - chunk_min[2]];
  let inv_rd = [1.0 / rd[0], 1.0 / rd[1], 1.0 / rd[2]];
  let abs_inv_rd = [inv_rd[0].abs(), inv_rd[1].abs(), inv_rd[2].abs()];
  // 方向 octant（bit0/1/2 = x/y/z 正，零分量按正 = 保守）：与 LUT 编码一致
  let oct =
    (sign[0] >= 0) as usize | (((sign[1] >= 0) as usize) << 1) | (((sign[2] >= 0) as usize) << 2);
  let lut = reach_lut();
  let p0 = [ro_c[0] + rd[0] * t0, ro_c[1] + rd[1] * t0, ro_c[2] + rd[2] * t0];
  let mut v = [
    (p0[0].floor() as i32).clamp(0, 255),
    (p0[1].floor() as i32).clamp(0, 255),
    (p0[2].floor() as i32).clamp(0, 255),
  ];
  let mut cur_t = t0;
  let mut face = entry_face;
  let mut level: u32 = 3; // 3 = chunk 根（子块 64³）… 0 = 4³ 节点（子块 = 1³ 体素）
  let mut budget = BUDGET;
  loop {
    if budget == 0 {
      return None;
    }
    budget -= 1;
    if level > 3 {
      return None;
    }
    // ---- traverse：从当前 level 下钻到 v 处内容；只有跨层才重取节点描述 ----
    let mut desc = tree.node_desc(v[0], v[1], v[2], (3 - level) as u8);
    loop {
      let idx = cell_index(&v, level);
      if level == 0 {
        // 叶层：1³ 体素直接问树（GPU 侧读的是 inline 半字，判定口径相同）
        if let Some(p) = tree.get_voxel(v[0], v[1], v[2]) {
          return Some((cur_t, p.get(), face, v));
        }
        break;
      }
      if desc.mask & (1u64 << idx) == 0 {
        // 统一子块：色 = 节点 palette（0 = 空气）
        if !desc.palette.is_air() {
          return Some((cur_t, desc.palette.get(), face, v));
        }
        break;
      }
      level -= 1;
      desc = tree.node_desc(v[0], v[1], v[2], (3 - level) as u8);
    }
    // ---- 整砖 LUT 跳过（仅 palette==0 节点安全：这类节点里只有 mask bit=1 的子块可能有实体）----
    if desc.palette.is_air() {
      let entry_i = cell_index(&v, level);
      if desc.mask & lut[oct * MARCH_MASK_ENTRIES + entry_i] == 0 {
        level += 1;
        if level > 3 {
          return None;
        }
        // 提级后必须重取掩码：内层步进按新 level 的子块粒度判内容（GPU 侧那份来自栈里的上一层）
        desc = tree.node_desc(v[0], v[1], v[2], (3 - level) as u8);
      }
    }
    // ---- v 处为空气：当前 level brick 内 DDA ----
    let log2 = level * 2;
    let s = 1i32 << log2; // 子块边长 voxel：1/4/16/64
    // side 距离：v 对齐到 s 的基址；正向 → 基址+s，负向 → 基址
    let mut side = [1e30f32; 3];
    for i in 0..3 {
      if rd[i].abs() > 1e-30 {
        let base = v[i] & !(s - 1);
        let boundary = if sign[i] >= 0 { base + s } else { base };
        side[i] = ((boundary as f32 - ro_c[i]) * inv_rd[i]).max(cur_t);
      }
    }
    let step_inc = [s as f32 * abs_inv_rd[0], s as f32 * abs_inv_rd[1], s as f32 * abs_inv_rd[2]];
    let mut step_axis: usize;
    let mut changed = false;
    // 内层步进只查手里这份 `desc`（同一节点内零重取）
    loop {
      let mn = if side[0] <= side[1] && side[0] <= side[2] {
        0
      } else if side[1] <= side[2] {
        1
      } else {
        2
      };
      step_axis = mn;
      cur_t = side[mn];
      if cur_t >= t1 {
        return None; // 段内再无子块可入
      }
      let old_cell = (v[mn] >> log2) & 3;
      // 步进即对齐：v 恒 = `cur_t` 处的真实体素。格基址每步从 v 重算；步进轴取新格边界，
      // 其余轴按射线位置钳在本格内。
      let cell_min = [v[0] & !(s - 1), v[1] & !(s - 1), v[2] & !(s - 1)];
      let p_step = [ro_c[0] + rd[0] * cur_t, ro_c[1] + rd[1] * cur_t, ro_c[2] + rd[2] * cur_t];
      for i in 0..3 {
        let pf = p_step[i].floor() as i32;
        v[i] = pf.clamp(cell_min[i], cell_min[i] + (s - 1));
      }
      v[mn] = if sign[mn] < 0 { cell_min[mn] - 1 } else { cell_min[mn] + s };
      side[mn] += step_inc[mn];
      face = (mn * 2) as u8 + if sign[mn] < 0 { 1 } else { 0 };
      // 跨出 brick（4 子块）？正向往 3→外、负向往 0→外
      let crossed = if sign[mn] >= 0 { old_cell == 3 } else { old_cell == 0 };
      if crossed {
        changed = true;
        break;
      }
      let idx = cell_index(&v, level);
      if level == 0 {
        if let Some(p) = tree.get_voxel(v[0], v[1], v[2]) {
          return Some((cur_t, p.get(), face, v));
        }
      } else if desc.mask & (1u64 << idx) != 0 {
        break; // 分裂子块 → 回 traverse 下钻
      } else if !desc.palette.is_air() {
        return Some((cur_t, desc.palette.get(), face, v)); // 统一实体
      }
      // 空气子块 → 回循环头重选轴
    }
    if changed {
      level += 1;
      if level > 3 {
        return None;
      }
    }
    // ---- firstTrailingBit 层级自适应跨级跳 ----
    // 步进轴新坐标的尾随零位 = 对齐 run 长度：正向 comp = 对齐基址，负向 comp = 区域尾址 +1；
    // tz >> 1 = 可一次跨步到的最粗 level。
    let positive = sign[step_axis] >= 0;
    let cur_log2 = level * 2;
    let m: u32 = 0xFFFF_FFFFu32.wrapping_shl(cur_log2);
    let vmin_u = v[step_axis] as u32; // i32→u32 环绕（负值由公式自然处理）
    let comp = if positive { vmin_u & m } else { (vmin_u & m) | !m };
    let tz = comp.wrapping_add(if positive { 0 } else { 1 }).trailing_zeros();
    level = level.max(tz >> 1);
    if level > 3 {
      return None; // 跨出 chunk
    }
    // 对齐快照：v 钳到 cur_t 处当前 level 区域内，步进轴取精确边界整数
    let mi = m as i32;
    let base = [v[0] & mi, v[1] & mi, v[2] & mi];
    let p = [ro_c[0] + rd[0] * cur_t, ro_c[1] + rd[1] * cur_t, ro_c[2] + rd[2] * cur_t];
    for i in 0..3 {
      let pf = p[i].floor() as i32;
      v[i] = pf.clamp(base[i], base[i] + !mi);
    }
    v[step_axis] = comp as i32;
  }
}

/// 当前 level 子块在其父 brick 内的线性下标（`z*16 + y*4 + x`，与 LUT 的入口编码一致）
#[inline]
fn cell_index(v: &[i32; 3], level: u32) -> usize {
  let sh = level * 2;
  ((((v[2] >> sh) & 3) << 4) | (((v[1] >> sh) & 3) << 2) | ((v[0] >> sh) & 3)) as usize
}

/// volume 的占用窗口（局部体素 AABB：min / max(exclusive)）= 有 chunk 的区块外包；空 volume → None。
fn occupied_window(grid: &VolumeGrid) -> Option<(Vec3, Vec3)> {
  let mut it = grid.chunk_coords();
  let first = it.next()?.0;
  let (mut lo, mut hi) = (first, first);
  for c in it {
    lo = lo.min(c.0);
    hi = hi.max(c.0);
  }
  let s = CHUNK_SIZE as f32;
  Some((lo.as_vec3() * s, (hi + IVec3::ONE).as_vec3() * s))
}

/// slab 法射线 × AABB：返回 `(t_enter, t_exit)`；`t_exit < t_enter` 表示不相交（含平行且在外）。
fn slab_box(ro: Vec3, rd: Vec3, mn: Vec3, mx: Vec3, t0: f32, t1: f32) -> (f32, f32) {
  let o = [ro.x, ro.y, ro.z];
  let d = [rd.x, rd.y, rd.z];
  let lo = [mn.x, mn.y, mn.z];
  let hi = [mx.x, mx.y, mx.z];
  let (mut t_enter, mut t_exit) = (t0, t1);
  for i in 0..3 {
    if d[i].abs() < 1e-30 {
      if o[i] < lo[i] || o[i] > hi[i] {
        return (1.0, 0.0);
      }
    } else {
      let ta = (lo[i] - o[i]) / d[i];
      let tb = (hi[i] - o[i]) / d[i];
      t_enter = t_enter.max(ta.min(tb));
      t_exit = t_exit.min(tb.max(ta));
    }
  }
  (t_enter, t_exit)
}

/// 法线 → 面号（取最大分量轴；镜像 WGSL `face_normal_from_index` 的逆）
fn face_index_from_normal(n: Vec3) -> u8 {
  let ax = n.x.abs();
  let ay = n.y.abs();
  let az = n.z.abs();
  if ax >= ay.max(az) {
    if n.x >= 0.0 { 1 } else { 0 }
  } else if ay >= az {
    if n.y >= 0.0 { 3 } else { 2 }
  } else if n.z >= 0.0 {
    5
  } else {
    4
  }
}

/// 面号 0..5 → ±轴单位向量（镜像 WGSL `face_normal_from_index`）
fn face_normal_from_index(f: u8) -> Vec3 {
  match f {
    0 => -Vec3::X,
    1 => Vec3::X,
    2 => -Vec3::Y,
    3 => Vec3::Y,
    4 => -Vec3::Z,
    5 => Vec3::Z,
    _ => Vec3::ZERO,
  }
}

/// 方向可达掩码 LUT（8 octant × 64 入口格 × u64）：与 GPU 同一张表（`wire::march_mask_lut_words`
/// 生成后上传 `b_leaves`），此处按 u64 重排一次并缓存 —— 表是纯组合量，与场景无关。
fn reach_lut() -> &'static [u64] {
  static LUT: OnceLock<Vec<u64>> = OnceLock::new();
  LUT.get_or_init(|| {
    march_mask_lut_words()
      .as_chunks::<2>()
      .0
      .iter()
      .map(|w| w[0] as u64 | ((w[1] as u64) << 32))
      .collect()
  })
}

#[cfg(test)]
mod tests {
  use super::*;
  use gate_voxel::{PaletteId, VoxelCoord, fill_box, fill_sphere};
  use glam::Mat3;

  /// 测试内参考实现：**无层次 / 无 LUT / 无跨级跳**的逐体素 A&W —— 优化版的对照物。
  /// 起点格嵌在实体内属 UB（`raycast` 那时把 `-rd` 主轴当进入面），故本测试的起点一律取空气格。
  fn brute(
    main: &VolumeGrid,
    origin: Vec3,
    dir: Vec3,
    max_dist: f32,
  ) -> Option<(IVec3, IVec3, f32)> {
    let mut v =
      IVec3::new(origin.x.floor() as i32, origin.y.floor() as i32, origin.z.floor() as i32);
    let step = IVec3::new(
      if dir.x >= 0.0 { 1 } else { -1 },
      if dir.y >= 0.0 { 1 } else { -1 },
      if dir.z >= 0.0 { 1 } else { -1 },
    );
    let inv = Vec3::new(1.0 / dir.x, 1.0 / dir.y, 1.0 / dir.z);
    let mut t_max = Vec3::new(
      ((v.x + if step.x > 0 { 1 } else { 0 }) as f32 - origin.x) * inv.x,
      ((v.y + if step.y > 0 { 1 } else { 0 }) as f32 - origin.y) * inv.y,
      ((v.z + if step.z > 0 { 1 } else { 0 }) as f32 - origin.z) * inv.z,
    );
    let t_delta = Vec3::new(inv.x.abs(), inv.y.abs(), inv.z.abs());
    let mut face = IVec3::ZERO;
    let mut t = 0.0f32;
    for _ in 0..(3.0 * max_dist) as u32 + 3 {
      if !main.get_voxel(VoxelCoord::from_ivec3(v)).unwrap_or(PaletteId::AIR).is_air() {
        return Some((v, face, t));
      }
      let axis = if t_max.x <= t_max.y && t_max.x <= t_max.z {
        0
      } else if t_max.y <= t_max.z {
        1
      } else {
        2
      };
      t = t_max[axis];
      if t > max_dist {
        return None;
      }
      v[axis] += step[axis];
      t_max[axis] += t_delta[axis];
      face = IVec3::ZERO;
      face[axis] = -step[axis];
    }
    None
  }

  /// 确定性伪随机（互素线性同余，不用 rand）
  fn lcg(state: &mut u32) -> f32 {
    *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    (*state >> 8) as f32 / (1u32 << 24) as f32
  }

  /// 跨 3 个 chunk（含负坐标）+ 实心块内空腔 + 悬浮球（斜掠 / 擦边）
  fn scene() -> Volumes {
    let mut g = VolumeGrid::new();
    fill_box(&mut g, IVec3::new(0, 0, 0), IVec3::new(200, 40, 40), 1);
    fill_box(&mut g, IVec3::new(270, 0, 0), IVec3::new(20, 40, 40), 2);
    fill_box(&mut g, IVec3::new(-60, -60, -60), IVec3::new(2, 120, 120), 3);
    fill_sphere(&mut g, IVec3::new(220, 120, 30), 20, 4);
    // 实心块里的空腔：射线会穿进去再穿出来（uniform 空气节点 + LUT 剔除的用武之地）
    fill_box(&mut g, IVec3::new(60, 10, 10), IVec3::new(30, 10, 10), 0);
    Volumes::new(g)
  }

  /// `ChunkTree::node_desc` 的覆盖不变量（抽样扫描，步长 7）：**实心体素**在每一层要么落在
  /// 分裂子块里（bit=1，继续下钻），要么被该节点的 uniform 色覆盖（palette ≠ 空气）。
  /// 遍历的正确性完全押在这个不变量上，故单独钉一条。
  #[test]
  fn node_desc_covers_every_solid_voxel() {
    let vols = scene();
    let g = vols.main();
    let mut solid = 0usize;
    for cc in [ChunkCoord::new(-1, 0, -1), ChunkCoord::new(0, 0, 0), ChunkCoord::new(1, 0, 0)] {
      let Some(tree) = g.chunk(cc) else { continue };
      for z in (0..256).step_by(7) {
        for y in (0..256).step_by(7) {
          for x in (0..256).step_by(7) {
            if tree.get_voxel(x, y, z).is_none() {
              continue; // 空气
            }
            solid += 1;
            for level in 0..=3u8 {
              let d = tree.node_desc(x, y, z, level);
              let idx = cell_index(&[x, y, z], 3 - level as u32);
              assert!(
                d.mask & (1u64 << idx) != 0 || !d.palette.is_air(),
                "实心体素 ({x},{y},{z}) 在 level {level} 既无分裂子块也无 uniform 色"
              );
            }
          }
        }
      }
    }
    assert!(solid > 100, "抽样里实心体素太少（{solid}），测试没测到东西");
  }

  /// 主验收：优化版（层次 + LUT + 跨级跳）与暴力逐体素扫描**逐条射线**一致（体素 / 入面 / t）。
  /// 射线朝实心目标 + 抖动 ⇒ 命中 / 穿透实心块内空腔 / 擦边三种情形都覆盖；再单独来一组轴对齐射线
  /// （`inv = ∞` 的退化分支）。
  #[test]
  fn matches_brute_force_on_deterministic_rays() {
    let vols = scene();
    let main = vols.main();
    let max_dist = 800.0f32;
    let origins = [
      Vec3::new(150.5, 100.5, 20.5),
      Vec3::new(220.5, 200.5, 30.5),
      Vec3::new(-100.5, -100.5, -100.5),
      Vec3::new(300.5, 150.5, 20.5),
    ];
    // 目标点：大块内部、实心块里的空腔之外、第二块、薄墙、悬浮球
    let targets = [
      Vec3::new(20.0, 20.0, 20.0),
      Vec3::new(150.0, 20.0, 20.0),
      Vec3::new(280.0, 20.0, 20.0),
      Vec3::new(-59.0, 0.0, 0.0),
      Vec3::new(220.0, 130.0, 30.0),
    ];
    let mut state = 0x1234_5678u32;
    let mut hits = 0usize;
    let mut rays = 0usize;
    let mut check = |origin: Vec3, dir: Vec3| {
      if dir == Vec3::ZERO {
        return;
      }
      rays += 1;
      match (brute(main, origin, dir, max_dist), raycast(&vols, origin, dir, max_dist)) {
        (None, None) => {}
        (Some((v, f, t)), Some(hit)) => {
          hits += 1;
          assert_eq!(hit.voxel, v, "origin={origin} dir={dir}");
          assert_eq!(hit.face, f, "origin={origin} dir={dir} voxel={v}");
          // t 只比到相对 1e-4：长射线跨数百格 / 多层，层次版的逐级 side 距离与逐格累加会有浮点累积差
          let tol = 1e-3 + 1e-4 * t.abs();
          assert!((hit.t - t).abs() <= tol, "origin={origin} dir={dir} t: {t} vs {}", hit.t);
        }
        (w, g) => panic!("命中不一致 origin={origin} dir={dir}: brute={w:?} cpu={g:?}"),
      }
    };
    for origin in origins {
      for target in targets {
        for _ in 0..12 {
          let jitter =
            Vec3::new(lcg(&mut state) - 0.5, lcg(&mut state) - 0.5, lcg(&mut state) - 0.5) * 40.0;
          check(origin, (target + jitter - origin).normalize_or_zero());
        }
      }
      for dir in [Vec3::X, Vec3::NEG_X, Vec3::Y, Vec3::NEG_Y, Vec3::Z, Vec3::NEG_Z] {
        check(origin, dir);
      }
    }
    assert!(hits > rays / 3, "对照集里命中太少（{hits}/{rays}），测试没测到东西");
  }

  /// 射程无界（∞）+ 入面契约：`voxel + face` 是前方那格空气（放置落点）。
  #[test]
  fn reach_is_unbounded_and_face_points_back() {
    let mut g = VolumeGrid::new();
    g.set_voxel_ivec3(IVec3::new(500, 0, 0), PaletteId(1));
    let vols = Volumes::new(g);
    let origin = Vec3::new(-1000.5, 0.5, 0.5);
    let hit = raycast(&vols, origin, Vec3::X, f32::INFINITY).expect("无界射程应命中");
    assert_eq!(hit.voxel, IVec3::new(500, 0, 0));
    assert_eq!(hit.face, IVec3::new(-1, 0, 0));
    assert_eq!(hit.normal, -Vec3::X);
    assert!((hit.t - 1500.5).abs() < 1e-3, "t={}", hit.t);
    assert!(vols.main().get_voxel(VoxelCoord::from_ivec3(hit.voxel + hit.face)).is_none());
    // 空世界 / 背离世界：世界外恒为空气 ⇒ None（不会空转到步数上限）
    let empty = Volumes::new(VolumeGrid::new());
    assert!(raycast(&empty, Vec3::ZERO, Vec3::X, f32::INFINITY).is_none());
    assert!(raycast(&vols, origin, Vec3::NEG_X, f32::INFINITY).is_none());
  }

  /// 物体：世界 AABB + 局部变换 + 局部 DDA —— t 保持世界标尺，法线转回世界系。
  #[test]
  fn object_hit_is_transformed() {
    let mut vols = Volumes::new(VolumeGrid::new());
    vols.add_object(Vec3::new(1000.0, 0.0, 0.0), Mat3::IDENTITY, 2.0);
    vols.object_mut(0).unwrap().set_voxel_ivec3(IVec3::ZERO, PaletteId(5));
    let hit = raycast(&vols, Vec3::new(900.5, 0.5, 0.5), Vec3::X, 500.0).expect("物体应命中");
    assert_eq!(hit.obj_id, 0);
    assert_eq!(hit.voxel, IVec3::ZERO, "局部体素坐标");
    assert_eq!(hit.face, IVec3::new(-1, 0, 0));
    assert_eq!(hit.normal, -Vec3::X);
    // 局部 [0,1) × scale 2 ⇒ 世界 [1000, 1002)
    assert!((hit.t - 99.5).abs() < 1e-3, "t={}", hit.t);
  }
}
