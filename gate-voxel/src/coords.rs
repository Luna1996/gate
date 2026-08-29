//! 坐标系与层级常量（P1.1）
//!
//! 分辨率体系（grill-me v3 裁决）：
//! - L0 基元胞 4cm，Tile 边长 32 基元胞 = 128cm
//! - 细分 4 级：L1 2cm / L2 1cm / L3 0.5cm / L4 0.25cm
//! - 世界位置以**最细格点（0.25cm）为单位的 i32** 表达，无边界常量（沙盒无界）
//!
//! 任何层级的体素立方都完整落在单个基元胞内（sub 网格与基元胞对齐），
//! 跨 Tile 的大块编辑由 batch_edit 在上层拆分。

use glam::IVec3;

/// 细分层级：0 = 4cm 基元胞（最粗），4 = 0.25cm（最细）
pub type Level = u8;

/// 最深细分层级
pub const MAX_LEVEL: Level = 4;

/// Tile 边长（基元胞数）：32³ = 32768 基元胞 / Tile
pub const TILE_CELLS: i32 = 32;

/// 基元胞边长（最细格数）：16³ = 4096 槽 @ L4
pub const SUB_PER_CELL: i32 = 16;

/// Tile 边长（最细格数）：32 × 16 = 512
pub const TILE_SUB: i32 = TILE_CELLS * SUB_PER_CELL;

/// 每层级的边长（最细格数）：L0=16, L1=8, L2=4, L3=2, L4=1
pub const LEVEL_SUB_EXTENT: [i32; 5] = [16, 8, 4, 2, 1];

/// 每层级的槽位数（该层一张完整表覆盖整个基元胞）
pub const LEVEL_SLOT_COUNT: [usize; 5] = [1, 8, 64, 512, 4096];

/// 层级对应的槽表边长：L1=2, L2=4, L3=8, L4=16（usize：直接参与槽索引线性运算）
pub const LEVEL_TABLE_AXIS: [usize; 5] = [1, 2, 4, 8, 16];

/// Tile 网格坐标（i32³，无界）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileCoord(pub IVec3);

impl TileCoord {
  pub fn new(x: i32, y: i32, z: i32) -> Self {
    Self(IVec3::new(x, y, z))
  }
}

/// 体素位置：Tile 内基元胞 + 层级 + 层内子坐标
///
/// 一个 VoxelPos 唯一定位一个体素立方，边长 = `LEVEL_SUB_EXTENT[level]` 最细格
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VoxelPos {
  pub tile: TileCoord,
  /// 基元胞在 Tile 内坐标，各分量 0..32
  pub cell: IVec3,
  pub level: Level,
  /// 层内子坐标，各分量 0..LEVEL_TABLE_AXIS[level]
  pub sub: IVec3,
}

impl VoxelPos {
  /// 从最细格坐标 + 目标层级构造（向下取整到该层网格）
  pub fn from_fine(fine: IVec3, level: Level) -> Self {
    debug_assert!(level <= MAX_LEVEL);
    // 欧氏除法保证负坐标正确落在邻接 Tile
    let tile = fine.div_euclid(IVec3::splat(TILE_SUB));
    let in_tile = fine.rem_euclid(IVec3::splat(TILE_SUB));
    let cell = in_tile.div_euclid(IVec3::splat(SUB_PER_CELL));
    let in_cell = in_tile.rem_euclid(IVec3::splat(SUB_PER_CELL));
    let shift = MAX_LEVEL - level;
    let sub = in_cell >> shift;
    Self {
      tile: TileCoord(tile),
      cell,
      level,
      sub,
    }
  }

  /// 体素立方的最小角（最细格坐标）
  pub fn fine_min(&self) -> IVec3 {
    let shift = MAX_LEVEL - self.level;
    self.tile.0 * TILE_SUB + self.cell * SUB_PER_CELL + (self.sub << shift)
  }

  /// 体素立方边长（最细格数）
  pub fn fine_extent(&self) -> i32 {
    LEVEL_SUB_EXTENT[self.level as usize]
  }

  /// 同区域换层级（sub 重新按新层级解析）
  pub fn with_level(&self, level: Level) -> Self {
    Self::from_fine(self.fine_min(), level)
  }

  /// 基元胞在 Tile 内的线性索引（x + y*32 + z*32²）
  pub fn cell_index(&self) -> u16 {
    (self.cell.x + self.cell.y * TILE_CELLS + self.cell.z * TILE_CELLS * TILE_CELLS) as u16
  }

  /// 该层级子槽在层级表内的线性索引（x + y*axis + z*axis²）
  pub fn slot_index(&self) -> usize {
    let axis = LEVEL_TABLE_AXIS[self.level as usize] as i32;
    (self.sub.x + self.sub.y * axis + self.sub.z * axis * axis) as usize
  }

  /// 父级槽索引（level 0 无父级，返回自身 slot）
  pub fn parent_slot_index(&self) -> Option<usize> {
    if self.level == 0 {
      return None;
    }
    let parent = Self::from_fine(self.fine_min(), self.level - 1);
    Some(parent.slot_index())
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn from_fine_positive_roundtrip() {
    let pos = VoxelPos::from_fine(IVec3::new(300, 200, 100), 2);
    assert_eq!(pos.tile, TileCoord(IVec3::ZERO));
    assert_eq!(
      pos.fine_min(),
      IVec3::new(300, 200, 100).div_euclid(IVec3::splat(4)) * 4
    );
    assert_eq!(pos.fine_extent(), 4);
  }

  #[test]
  fn negative_coords_fall_into_neighbor_tile() {
    // fine x = -1 → 前一个 tile 的最后一个格
    let pos = VoxelPos::from_fine(IVec3::new(-1, 0, 0), 4);
    assert_eq!(pos.tile.0.x, -1);
    assert_eq!(pos.cell.x, 31);
    assert_eq!(pos.sub.x, 15);
    assert_eq!(pos.fine_min().x, -1);
  }

  #[test]
  fn with_level_resolves_same_region() {
    let fine = IVec3::new(513, -30, 77); // 跨 tile 边界
    let l4 = VoxelPos::from_fine(fine, 4);
    let l1 = l4.with_level(1);
    assert_eq!(l1.level, 1);
    assert_eq!(l1.fine_extent(), 8);
    // L1 立方必须包含原 L4 点
    let min = l1.fine_min();
    assert!(fine.cmpge(min).all() && fine.cmple(min + IVec3::splat(l1.fine_extent())).all());
    assert_eq!(l1.tile, l4.tile);
    assert_eq!(l1.cell, l4.cell);
  }

  #[test]
  fn slot_index_linear_layout() {
    // cell 0 内 L1 sub (1,1,0)：fine = sub << 3 = (8,8,0)
    let pos = VoxelPos::from_fine(IVec3::new(8, 8, 0), 1);
    assert_eq!(pos.sub, IVec3::new(1, 1, 0));
    assert_eq!(pos.slot_index(), 1 + 2);
    // L4: axis=16
    let pos4 = VoxelPos::from_fine(IVec3::new(3, 0, 0), 4);
    assert_eq!(pos4.slot_index(), 3);
  }

  #[test]
  fn cell_index_saturates_u16() {
    let max = VoxelPos::from_fine(IVec3::new(511, 511, 511), 4);
    assert_eq!(max.cell_index(), 32 * 32 * 32 - 1);
  }
}
