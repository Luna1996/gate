//! 坐标系与层级常量（Phase 0，Douglas Brick Tree 对齐）
//!
//! 统一后坐标体系：
//! - **VoxelCoord**：最细格（1³），i32³ 世界坐标，无边界
//! - **ChunkCoord**：chunk（256³ 体素）分层 HashMap key
//! - **BrickCoord**：Brick Tree 内 brick（level 指定分裂深度，256³ / 4^level 体素）
//!
//! 分裂树层级：256 → 64 → 16 → 4 → 1（4 层分裂，4³=64 分裂因子）
//! - Level 0: 256³（chunk 整体 uniform leaf 时）
//! - Level 1: 64³
//! - Level 2: 16³ ← 组件粒度（= DDGI probe cell = gate cell）
//! - Level 3: 4³
//! - Level 4: 1³ ← 编辑最细粒度

use glam::IVec3;

/// chunk 边长（体素）：256³
pub const CHUNK_SIZE: i32 = 256;

/// 分裂因子（4³ = 64 子块）
pub const BRICK_FACTOR: i32 = 4;

/// 最大分裂深度：ceil(log₄(256)) = 4
pub const MAX_LEVEL: u8 = 4;

/// 每层级 brick 边长（体素）：[256, 64, 16, 4, 1]
pub const LEVEL_EXTENT: [i32; 5] = [256, 64, 16, 4, 1];

/// Brick Tree 内 4³ 子块的线性索引（z*16 + y*4 + x）
#[inline]
pub fn child_linear_idx(local_x: i32, local_y: i32, local_z: i32) -> u32 {
  debug_assert!((0..BRICK_FACTOR).contains(&local_x));
  debug_assert!((0..BRICK_FACTOR).contains(&local_y));
  debug_assert!((0..BRICK_FACTOR).contains(&local_z));
  (local_z * BRICK_FACTOR * BRICK_FACTOR + local_y * BRICK_FACTOR + local_x) as u32
}

/// Chunk 网格坐标（i32³，无界）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ChunkCoord(pub IVec3);

impl ChunkCoord {
  pub fn new(x: i32, y: i32, z: i32) -> Self {
    Self(IVec3::new(x, y, z))
  }
}

impl Ord for ChunkCoord {
  fn cmp(&self, other: &Self) -> std::cmp::Ordering {
    (self.0.x, self.0.y, self.0.z).cmp(&(other.0.x, other.0.y, other.0.z))
  }
}

impl PartialOrd for ChunkCoord {
  fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
    Some(self.cmp(other))
  }
}

/// 世界体素坐标（1³ 最细格，i32³）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VoxelCoord {
  pub x: i32,
  pub y: i32,
  pub z: i32,
}

impl VoxelCoord {
  pub fn new(x: i32, y: i32, z: i32) -> Self {
    Self { x, y, z }
  }

  pub fn from_ivec3(v: IVec3) -> Self {
    Self {
      x: v.x,
      y: v.y,
      z: v.z,
    }
  }

  pub fn to_ivec3(self) -> IVec3 {
    IVec3::new(self.x, self.y, self.z)
  }

  /// 对应的 chunk 坐标（欧氏除法保证负坐标落到邻接 chunk）
  pub fn chunk(&self) -> ChunkCoord {
    let v = IVec3::new(self.x, self.y, self.z);
    ChunkCoord(v.div_euclid(IVec3::splat(CHUNK_SIZE)))
  }

  /// 在所在 chunk 内的本地坐标（0..255 每分量，负坐标正确）
  pub fn in_chunk(&self) -> IVec3 {
    let v = IVec3::new(self.x, self.y, self.z);
    v.rem_euclid(IVec3::splat(CHUNK_SIZE))
  }

  /// 映射到 chunk 内指定 level 的 brick 索引（各分量 0..4^level）
  /// 返回值 = (chunk_coord, level_brick_x, level_brick_y, level_brick_z)
  pub fn to_level_brick(&self, level: u8) -> (ChunkCoord, u32, u32, u32) {
    let chunk = self.chunk();
    let extent = LEVEL_EXTENT[level as usize];
    let local = self.in_chunk();
    let bx = (local.x / extent) as u32;
    let by = (local.y / extent) as u32;
    let bz = (local.z / extent) as u32;
    (chunk, bx, by, bz)
  }
}

/// Brick Tree 内 brick 的坐标
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BrickCoord {
  /// 分裂深度（0..MAX_LEVEL）
  pub level: u8,
  pub chunk: ChunkCoord,
  /// 该层级 brick 在 chunk 内的索引（各分量 0..4^level）
  pub x: u32,
  pub y: u32,
  pub z: u32,
}

impl BrickCoord {
  pub fn new(level: u8, chunk: ChunkCoord, x: u32, y: u32, z: u32) -> Self {
    Self {
      level,
      chunk,
      x,
      y,
      z,
    }
  }

  /// brick 边长（体素）
  pub fn size(&self) -> i32 {
    LEVEL_EXTENT[self.level as usize]
  }

  /// brick 覆盖的体素世界最小角（VoxelCoord）
  pub fn voxel_min(&self) -> VoxelCoord {
    let extent = self.size();
    let offset_x = self.x as i32 * extent;
    let offset_y = self.y as i32 * extent;
    let offset_z = self.z as i32 * extent;
    VoxelCoord::new(
      self.chunk.0.x * CHUNK_SIZE + offset_x,
      self.chunk.0.y * CHUNK_SIZE + offset_y,
      self.chunk.0.z * CHUNK_SIZE + offset_z,
    )
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn chunk_coord_positive() {
    let v = VoxelCoord::new(100, 200, 300);
    assert_eq!(v.chunk(), ChunkCoord::new(0, 0, 1));
  }

  #[test]
  fn chunk_coord_negative() {
    let v = VoxelCoord::new(-1, 0, 0);
    assert_eq!(v.chunk(), ChunkCoord::new(-1, 0, 0));
    assert_eq!(v.in_chunk().x, 255);
  }

  #[test]
  fn level_2_brick_aligns() {
    let v = VoxelCoord::new(256 + 16 * 3 + 5, 0, 0);
    let (chunk, bx, by, bz) = v.to_level_brick(2);
    assert_eq!(chunk, ChunkCoord::new(1, 0, 0));
    assert_eq!(bx, 3);
    assert_eq!(by, 0);
    assert_eq!(bz, 0);
    let brick = BrickCoord::new(2, chunk, bx, by, bz);
    assert_eq!(brick.size(), 16);
    assert_eq!(brick.voxel_min().x, 256 + 48);
  }

  #[test]
  fn level_4_voxel_aligns() {
    // level 4 = 1³，精确到体素
    let v = VoxelCoord::new(256 + 100, 50, -10);
    let (chunk, bx, _by, bz) = v.to_level_brick(4);
    assert_eq!(chunk, ChunkCoord::new(1, 0, -1));
    let local = v.in_chunk();
    assert_eq!(bx, local.x as u32);
    // z=-10 在 chunk z=-1 里的 local z = -10 - (-1*256) = 246
    assert_eq!(bz, local.z as u32);
  }

  #[test]
  fn child_linear_idx_layout() {
    // (x=0, y=0, z=0) → 0
    assert_eq!(child_linear_idx(0, 0, 0), 0);
    // (x=3, y=3, z=3) → 3*16 + 3*4 + 3 = 63
    assert_eq!(child_linear_idx(3, 3, 3), 63);
    // (x=1, y=2, z=0) → 0*16 + 2*4 + 1 = 9
    assert_eq!(child_linear_idx(1, 2, 0), 9);
  }

  #[test]
  fn level_extent_chain() {
    // 256 / 4 = 64, 64 / 4 = 16, 16 / 4 = 4, 4 / 4 = 1
    assert_eq!(LEVEL_EXTENT, [256, 64, 16, 4, 1]);
  }
}
