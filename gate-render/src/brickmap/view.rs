//! Brick Tree 软件遍历器（Phase 1，Douglas mask DDA 的 CPU 参考实现）
//!
//! 独立实现 wire.rs §b_struct 寻址链——刻意不共享 builder 内部代码，
//! 等价性测试的意义就在于两套实现互为对照（共享即自证）。
//! 语义与 `VolumeGrid::get_voxel` 严格一致：Some(palette 1..=255) / None（空）。
//!
//! 寻址链（3 步，全部 storage load）：
//! ① chunk 窗口：fine → chunk（floor div 256）→ entry（0 = 无 chunk）
//! ② chunk base → 根节点 fixed（mask_lo/mask_hi/palette，8B）
//! ③ mask bit 测试 → popcount 定位 child offset → 下钻（每层 1 次 load，
//!    uniform 子块零额外 load——palette 直存父节点 fixed 字）

use glam::IVec3;

use super::wire::{BrickMapBuffers, CHUNK_INDEX_CAP, CHUNK_SIZE};

/// 稠密 chunk 窗口线性位置；窗口外返回 None
fn chunk_index_pos(origin: IVec3, dims: IVec3, chunk: IVec3) -> Option<usize> {
  let rel = chunk - origin;
  if rel.cmplt(IVec3::ZERO).any() || rel.cmpge(dims).any() {
    return None;
  }
  Some(
    (rel.x
      + rel.y * CHUNK_INDEX_CAP as i32
      + rel.z * CHUNK_INDEX_CAP as i32 * CHUNK_INDEX_CAP as i32) as usize,
  )
}

/// 读一个节点的 (mask, palette_u32)。`node` 为 b_struct 内绝对字址。
#[inline]
fn read_node(b_struct: &[u32], node: usize) -> (u64, u32) {
  let lo = b_struct[node] as u64;
  let hi = b_struct[node + 1] as u64;
  (hi << 32 | lo, b_struct[node + 2])
}

/// 砖块图只读视图：持有与 GPU buffer 字节一致的缓冲区引用
pub struct BrickMapView<'a> {
  b_struct: &'a [u32],
  origin: IVec3,
  dims: IVec3,
}

impl<'a> BrickMapView<'a> {
  pub fn new(buffers: &'a BrickMapBuffers) -> Self {
    let g = &buffers.globals;
    Self {
      b_struct: &buffers.b_struct,
      origin: IVec3::new(g.index_origin_x, g.index_origin_y, g.index_origin_z),
      dims: IVec3::new(
        g.index_dims_x as i32,
        g.index_dims_y as i32,
        g.index_dims_z as i32,
      ),
    }
  }

  /// 从裸字切片构造（obj pool 拼接采样用：切片 = 某物体 b_struct 的连续窗口）
  pub fn from_parts(b_struct: &'a [u32], origin: IVec3, dims: IVec3) -> Self {
    Self {
      b_struct,
      origin,
      dims,
    }
  }

  /// chunk 窗口查找 → DFS 树绝对字基址（0 = 无 chunk）
  fn chunk_base(&self, chunk: IVec3) -> Option<usize> {
    let ip = chunk_index_pos(self.origin, self.dims, chunk)?;
    let entry = self.b_struct[ip];
    if entry == 0 {
      return None;
    }
    Some(entry as usize - 1)
  }

  /// mask DDA 逐层下钻读单个最细格（1³）体素，O(分裂层数) = 最多 4 层
  pub fn get_voxel(&self, fine: IVec3) -> Option<u8> {
    let chunk = fine.div_euclid(IVec3::splat(CHUNK_SIZE));
    let base = self.chunk_base(chunk)?;
    let mut local = fine.rem_euclid(IVec3::splat(CHUNK_SIZE));
    let mut node = base;
    let mut extent = CHUNK_SIZE;
    loop {
      let (mask, pal) = read_node(self.b_struct, node);
      if mask == 0 {
        // uniform leaf（含 level 4 Uniform）：palette 0 = AIR
        return (pal != 0).then_some(pal as u8);
      }
      let child_extent = extent >> 2;
      let ix = local.x / child_extent;
      let iy = local.y / child_extent;
      let iz = local.z / child_extent;
      let ci = (iz * 16 + iy * 4 + ix) as u64;
      let bit = 1u64 << ci;
      if mask & bit == 0 {
        // uniform 子块：颜色 = 本节点 palette_u32（零额外 load）
        return (pal != 0).then_some(pal as u8);
      }
      let slot = (mask & (bit - 1)).count_ones() as usize;
      // child offset = chunk 内相对字址 → 绝对 = base + offset
      node = base + self.b_struct[node + 3 + slot] as usize;
      local = IVec3::new(
        local.x - ix * child_extent,
        local.y - iy * child_extent,
        local.z - iz * child_extent,
      );
      extent = child_extent;
    }
  }

  /// cell 灭占用查询（两级 DDA 粗步专用）：cc 为 cell 坐标（1 单位 = 16 fine）。
  /// 语义：false ⇒ get_voxel 对该 cell 内全部 16³ fine 位置都返回 None。
  ///
  /// 走树到 level 2（16³）粒度：
  /// - 中途 mask=0（uniform）→ occupied = palette != 0
  /// - mask bit=0（uniform 子块）→ occupied = palette != 0
  /// - 到达 extent=16 的 Split 节点 → occupied = true
  ///   （ChunkTree merge 不变式：Split ⇒ 64 槽颜色不全同 ⇒ 至少一槽非 AIR）
  pub fn cell_occupied(&self, cc: IVec3) -> bool {
    // cell 的最小角 fine → chunk + local（cell 恒不跨 chunk：16 | 256）
    let fine_min = cc * 16;
    let chunk = fine_min.div_euclid(IVec3::splat(CHUNK_SIZE));
    let Some(base) = self.chunk_base(chunk) else {
      return false;
    };
    let mut local = fine_min.rem_euclid(IVec3::splat(CHUNK_SIZE));
    let mut node = base;
    let mut extent = CHUNK_SIZE;
    loop {
      let (mask, pal) = read_node(self.b_struct, node);
      if mask == 0 {
        return pal != 0;
      }
      if extent == 16 {
        // 已到目标 brick 粒度且是 Split → 内部必有非 AIR 体素
        return true;
      }
      let child_extent = extent >> 2;
      let ix = local.x / child_extent;
      let iy = local.y / child_extent;
      let iz = local.z / child_extent;
      let ci = (iz * 16 + iy * 4 + ix) as u64;
      let bit = 1u64 << ci;
      if mask & bit == 0 {
        return pal != 0;
      }
      let slot = (mask & (bit - 1)).count_ones() as usize;
      node = base + self.b_struct[node + 3 + slot] as usize;
      local = IVec3::new(
        local.x - ix * child_extent,
        local.y - iy * child_extent,
        local.z - iz * child_extent,
      );
      extent = child_extent;
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use gate_voxel::{VolumeGrid, fill_box};

  /// 各级 child 线性索引与 gate-voxel coords.rs `child_linear_idx` 同构
  /// （独立实现对照：z*16 + y*4 + x）
  #[test]
  fn child_linear_idx_matches_gate_voxel() {
    use gate_voxel::child_linear_idx;
    for x in 0..4i64 {
      for y in 0..4i64 {
        for z in 0..4i64 {
          assert_eq!(
            (z * 16 + y * 4 + x) as u32,
            child_linear_idx(x as i32, y as i32, z as i32)
          );
        }
      }
    }
  }

  /// 负坐标 chunk 定位：fine -1 → chunk -1（div_euclid），local 255（rem_euclid）
  #[test]
  fn negative_fine_chunk_lookup() {
    let f = IVec3::new(-1, -257, 256);
    assert_eq!(f.div_euclid(IVec3::splat(CHUNK_SIZE)), IVec3::new(-1, -2, 1));
    assert_eq!(
      f.rem_euclid(IVec3::splat(CHUNK_SIZE)),
      IVec3::new(255, 255, 0)
    );
  }

  /// view 走 1 chunk 场景的完整读回（builder 等价性测试的主体在 builder.rs）
  #[test]
  fn view_reads_single_chunk_scene() {
    let mut grid = VolumeGrid::new();
    fill_box(&mut grid, IVec3::new(0, 0, 0), IVec3::splat(16), 3);
    let b = crate::brickmap::BrickMapBuilder::build_full(&grid);
    let v = BrickMapView::new(b.buffers());
    assert_eq!(v.get_voxel(IVec3::new(0, 0, 0)), Some(3));
    assert_eq!(v.get_voxel(IVec3::new(15, 15, 15)), Some(3));
    assert_eq!(v.get_voxel(IVec3::new(16, 0, 0)), None);
    assert_eq!(v.get_voxel(IVec3::new(-1, 0, 0)), None);
    assert!(v.cell_occupied(IVec3::ZERO));
    assert!(!v.cell_occupied(IVec3::new(1, 0, 0)));
  }
}
