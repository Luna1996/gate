//! Volume + VolumeGrid：主世界 / 独立物体的统一容器（Phase 0）
//!
//! VolumeGrid = 主世界容器（一个 Volume + 无界 chunk HashMap）
//! Volume = 任意体素 volume（主世界 = identity transform，独立物体 = 任意 transform）

use std::collections::HashMap;

use glam::IVec3;

use crate::chunk_tree::ChunkTree;
use crate::coords::{CHUNK_SIZE, ChunkCoord, VoxelCoord};
use crate::dirty::DirtyTracker;
use crate::palette::{AIR_INDEX, Palette};

/// 组件层：level 2 brick = 16³ = 4096 体素 = 一个组件 cell
/// 每 chunk = (256/16)³ = 16³ = 4096 个 level 2 brick = 4096 个 u16
pub const COMP_BRICKS_PER_CHUNK: usize = 16 * 16 * 16;
pub const COMP_BRICK_EXTENT: i32 = 16;

/// 主世界容器 = VolumeGrid = 一个 Volume + VolumeCoord
///
/// Phase 0 只有主世界（VolumeGrid 是唯一 Volume）。Phase 1+ 扩展为 Vec<Volume>
/// 支持独立物体（任意数量、任意 transform）。
#[derive(Debug, Default)]
pub struct VolumeGrid {
  chunks: HashMap<ChunkCoord, ChunkTree>,
  palette: Palette,
  pub dirty: DirtyTracker,
  comp_layer: HashMap<ChunkCoord, Box<[u16; COMP_BRICKS_PER_CHUNK]>>,
  state_table: Vec<[u32; 4]>,
  pub state_dirty: bool,
}

/// 一次编辑产生的脏区域（Phase 1 上传管道用）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirtyEdit {
  pub chunk: ChunkCoord,
}

impl VolumeGrid {
  pub fn new() -> Self {
    Self {
      chunks: HashMap::new(),
      palette: Palette::default(),
      dirty: DirtyTracker::new(),
      comp_layer: HashMap::new(),
      state_table: vec![[0u32; 4]; 256],
      state_dirty: true,
    }
  }

  pub fn palette(&self) -> &Palette {
    &self.palette
  }

  pub fn palette_mut(&mut self) -> &mut Palette {
    &mut self.palette
  }

  pub fn chunk(&self, coord: ChunkCoord) -> Option<&ChunkTree> {
    self.chunks.get(&coord)
  }

  pub fn chunk_mut(&mut self, coord: ChunkCoord) -> Option<&mut ChunkTree> {
    self.chunks.get_mut(&coord)
  }

  pub fn chunk_coords(&self) -> impl Iterator<Item = ChunkCoord> + '_ {
    self.chunks.keys().copied()
  }

  pub fn chunk_count(&self) -> usize {
    self.chunks.len()
  }

  // =========================================================================
  // 体素查询
  // =========================================================================

  pub fn get_voxel(&self, voxel: VoxelCoord) -> Option<u8> {
    let chunk = voxel.chunk();
    let tree = self.chunks.get(&chunk)?;
    let local = voxel.in_chunk();
    tree.get_voxel(local.x, local.y, local.z)
  }

  /// 查询指定 level 的 brick 是否 uniform 同色
  pub fn get_uniform(&self, voxel: VoxelCoord, level: u8) -> Option<u8> {
    let chunk = voxel.chunk();
    let tree = self.chunks.get(&chunk)?;
    let local = voxel.in_chunk();
    tree.get_uniform(local.x, local.y, local.z, level)
  }

  // =========================================================================
  // 体素编辑
  // =========================================================================

  pub fn set_voxel(&mut self, voxel: VoxelCoord, palette: u8) -> Option<DirtyEdit> {
    let chunk = voxel.chunk();
    let local = voxel.in_chunk();

    let tree = self.chunks.entry(chunk).or_insert_with(ChunkTree::empty);
    if tree.set_voxel(local.x, local.y, local.z, palette) {
      self.dirty.mark_data(chunk);
      Some(DirtyEdit { chunk })
    } else {
      None
    }
  }

  pub fn set_voxel_ivec3(&mut self, pos: IVec3, palette: u8) -> Option<DirtyEdit> {
    self.set_voxel(VoxelCoord::from_ivec3(pos), palette)
  }

  pub fn clear_voxel(&mut self, voxel: VoxelCoord) -> Option<DirtyEdit> {
    let chunk = voxel.chunk();
    let local = voxel.in_chunk();
    let tree = self.chunks.get_mut(&chunk)?;
    if tree.clear_voxel(local.x, local.y, local.z) {
      // 如果 chunk 全空，可以删除
      if tree.is_empty() {
        self.chunks.remove(&chunk);
      }
      self.dirty.mark_data(chunk);
      Some(DirtyEdit { chunk })
    } else {
      None
    }
  }

  pub fn batch_edit(&mut self, ops: impl IntoIterator<Item = (VoxelCoord, u8)>) -> usize {
    let mut applied = 0;
    for (voxel, palette) in ops {
      if self.set_voxel(voxel, palette).is_some() {
        applied += 1;
      }
    }
    applied
  }

  // =========================================================================
  // 组件层（level 2 brick = 16³ per cell）
  // =========================================================================

  /// 写指定 ChunkCoord 下 level 2 brick (bx, by, bz) 的组件 ID
  pub fn set_comp(&mut self, chunk: ChunkCoord, bx: u32, by: u32, bz: u32, comp_id: u16) {
    let arr = self
      .comp_layer
      .entry(chunk)
      .or_insert_with(|| Box::new([0u16; COMP_BRICKS_PER_CHUNK]));
    let idx = (bx as usize)
      + (by as usize) * COMP_BRICK_EXTENT as usize
      + (bz as usize) * COMP_BRICK_EXTENT as usize * COMP_BRICK_EXTENT as usize;
    if arr[idx] != comp_id {
      arr[idx] = comp_id;
      self.dirty.mark_comp(chunk);
    }
  }

  /// 读指定 world voxel 所在 level 2 brick 的组件 ID
  pub fn get_comp(&self, voxel: VoxelCoord) -> u16 {
    let chunk = voxel.chunk();
    let local = voxel.in_chunk();
    let bx = (local.x / COMP_BRICK_EXTENT) as usize;
    let by = (local.y / COMP_BRICK_EXTENT) as usize;
    let bz = (local.z / COMP_BRICK_EXTENT) as usize;
    let arr = match self.comp_layer.get(&chunk) {
      Some(a) => a,
      None => return 0,
    };
    arr[bx + by * COMP_BRICK_EXTENT as usize + bz * COMP_BRICK_EXTENT as usize * COMP_BRICK_EXTENT as usize]
  }

  pub fn comp_layer(&self) -> &HashMap<ChunkCoord, Box<[u16; COMP_BRICKS_PER_CHUNK]>> {
    &self.comp_layer
  }

  // =========================================================================
  // StateTable
  // =========================================================================

  pub fn set_state(&mut self, id: u8, word: usize, value: u32) {
    debug_assert!(word < 4);
    if id as usize >= self.state_table.len() {
      self.state_table.resize(id as usize + 1, [0u32; 4]);
    }
    self.state_table[id as usize][word] = value;
    self.state_dirty = true;
  }

  pub fn get_state(&self, id: u8, word: usize) -> u32 {
    debug_assert!(word < 4);
    self
      .state_table
      .get(id as usize)
      .map(|a| a[word])
      .unwrap_or(0)
  }

  pub fn state_table_bytes(&self) -> &[u8] {
    let slice = &self.state_table[..];
    unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, slice.len() * 16) }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn set_get_single_voxel() {
    let mut grid = VolumeGrid::new();
    assert_eq!(grid.get_voxel(VoxelCoord::new(5, 6, 7)), None);
    assert!(grid.set_voxel(VoxelCoord::new(5, 6, 7), 1).is_some());
    assert_eq!(grid.get_voxel(VoxelCoord::new(5, 6, 7)), Some(1));
  }

  #[test]
  fn cross_chunk_editing() {
    let mut grid = VolumeGrid::new();
    // chunk 0: (255,255,255) 和 chunk 1: (256,0,0)
    assert!(grid.set_voxel(VoxelCoord::new(255, 255, 255), 1).is_some());
    assert!(grid.set_voxel(VoxelCoord::new(256, 0, 0), 2).is_some());
    assert_eq!(grid.chunk_count(), 2);
    assert_eq!(grid.get_voxel(VoxelCoord::new(255, 255, 255)), Some(1));
    assert_eq!(grid.get_voxel(VoxelCoord::new(256, 0, 0)), Some(2));
  }

  #[test]
  fn negative_coords_cross_chunk() {
    let mut grid = VolumeGrid::new();
    assert!(grid.set_voxel(VoxelCoord::new(-1, 0, 0), 3).is_some());
    assert_eq!(grid.get_voxel(VoxelCoord::new(-1, 0, 0)), Some(3));
    // 负 chunk
    let chunk = VoxelCoord::new(-1, 0, 0).chunk();
    assert_eq!(chunk, ChunkCoord::new(-1, 0, 0));
  }

  #[test]
  fn clear_removes_chunk_when_empty() {
    let mut grid = VolumeGrid::new();
    grid.set_voxel(VoxelCoord::new(10, 10, 10), 5);
    assert_eq!(grid.chunk_count(), 1);
    assert!(grid.clear_voxel(VoxelCoord::new(10, 10, 10)).is_some());
    assert_eq!(grid.chunk_count(), 0);
  }

  #[test]
  fn batch_edit_dedupes_dirty() {
    let mut grid = VolumeGrid::new();
    let ops: Vec<(VoxelCoord, u8)> = (0..100)
      .map(|i| (VoxelCoord::new(i, 0, 0), 1u8))
      .collect();
    assert_eq!(grid.batch_edit(ops), 100);
    assert_eq!(grid.dirty.data_dirty_count(), 1); // 全在 chunk (0,0,0)
  }
}
