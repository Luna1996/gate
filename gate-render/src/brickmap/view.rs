//! 砖块图软件遍历器（P2.2）：wire 格式缓冲 → 体素读回
//!
//! 独立实现 docs/brickmap.md §3 寻址链——刻意不共享 builder 内部代码，
//! 等价性测试的意义就在于两套实现互为对照（共享即自证）。
//! 语义与 `TileGrid::get_voxel` 严格一致：Some(palette 1..=255) / None（空）。
//! 同时作为 P2.4 GPU DDA 着色器的 CPU 参考实现（同路径逐级下钻）。

use gate_voxel::{MAX_LEVEL, VoxelPos};
use glam::IVec3;

use super::wire::{
  BITMAP_BASE, BRICK_SLAB_WORDS, BrickMapBuffers, CELL_DIR_WORDS, DIR_BASE, HDR_HAS_BRICK,
  HDR_HAS_L1, HDR_HAS_L2, HDR_HAS_L3, HDR_UNIFORM_MASK, SLOT_TABLE_WORDS, SLOT_TAG_BRANCH,
  SLOT_TAG_EMPTY, SLOT_TAG_LEAF, TILE_BITMAP_WORDS, TILE_INDEX_CAP, slot_palette, slot_tag,
  unpack_slot_word,
};

/// 层级槽线性索引（axis³ 表：x + y*axis + z*axis²）
fn slot_at(sub: IVec3, axis: i32) -> usize {
  (sub.x + sub.y * axis + sub.z * axis * axis) as usize
}

/// TileIndex 线性位置（独立实现，与 builder 各写各的，测试互为对照）
fn index_pos(origin: IVec3, dims: IVec3, tile: IVec3) -> Option<usize> {
  let rel = tile - origin;
  if rel.cmplt(IVec3::ZERO).any() || rel.cmpge(dims).any() {
    return None;
  }
  Some((rel.x + rel.y * TILE_INDEX_CAP + rel.z * TILE_INDEX_CAP * TILE_INDEX_CAP) as usize)
}

/// 砖块图只读视图：持有与 GPU buffer 字节一致的缓冲区引用
pub struct BrickMapView<'a> {
  b_struct: &'a [u32],
  b_leaves: &'a [u32],
  origin: IVec3,
  dims: IVec3,
}

impl<'a> BrickMapView<'a> {
  pub fn new(buffers: &'a BrickMapBuffers) -> Self {
    let g = &buffers.globals;
    Self {
      b_struct: &buffers.b_struct,
      b_leaves: &buffers.b_leaves,
      origin: IVec3::new(g.index_origin_x, g.index_origin_y, g.index_origin_z),
      dims: IVec3::new(
        g.index_dims_x as i32,
        g.index_dims_y as i32,
        g.index_dims_z as i32,
      ),
    }
  }

  /// 寻址链五步读回单个最细格（0.25cm）体素，O(1)
  pub fn get_voxel(&self, fine: IVec3) -> Option<u8> {
    let pos = VoxelPos::from_fine(fine, MAX_LEVEL);
    // ① TileIndex：0 = 空条目
    let ip = index_pos(self.origin, self.dims, pos.tile.0)?;
    let entry = self.b_struct[ip];
    if entry == 0 {
      return None;
    }
    let slot = (entry - 1) as usize;
    // ② TileBitmaps：基元胞占用位（0 = 整胞无数据）
    let ci = pos.cell_index() as usize;
    let bmp = BITMAP_BASE + slot * TILE_BITMAP_WORDS + ci / 32;
    if self.b_struct[bmp] >> (ci % 32) & 1 == 0 {
      return None;
    }
    // ③ CellDirs：胞节点绝对字偏移（0 = 空）
    let abs = self.b_struct[DIR_BASE + slot * CELL_DIR_WORDS + ci] as usize;
    if abs == 0 {
      return None;
    }
    // ④ CellNode 依序下钻（表按 l1 → l2 → l3 → brick 定序，无内部指针）
    self.walk_node(abs, pos.sub)
  }

  fn walk_node(&self, abs: usize, sub: IVec3) -> Option<u8> {
    let hdr = self.b_struct[abs];
    // uniform 胞：bits 0-7 = palette（0 = AIR，恰作空哨兵）
    if hdr & HDR_UNIFORM_MASK != 0 {
      return Some((hdr & HDR_UNIFORM_MASK) as u8);
    }
    let mut p = abs + 1;

    // L1 表（2³ = 8 槽）；非 uniform 必有 l1（tile.rs 规范型不变式 1）
    if hdr & HDR_HAS_L1 == 0 {
      return None;
    }
    let idx = slot_at(sub >> 3, 2);
    match self.slot(p, idx) {
      (SLOT_TAG_EMPTY, _) => return None,
      (SLOT_TAG_LEAF, pal) => return Some(pal),
      (SLOT_TAG_BRANCH, _) => {}
      _ => {
        debug_assert!(false, "非法 L1 槽 tag (node {abs:#x})");
        return None;
      }
    }
    p += SLOT_TABLE_WORDS[1];

    // L2 表（4³ = 64 槽）
    if hdr & HDR_HAS_L2 == 0 {
      debug_assert!(false, "L1 Branch 但 L2 表缺失 (node {abs:#x})");
      return None;
    }
    let idx = slot_at(sub >> 2, 4);
    match self.slot(p, idx) {
      (SLOT_TAG_EMPTY, _) => return None,
      (SLOT_TAG_LEAF, pal) => return Some(pal),
      (SLOT_TAG_BRANCH, _) => {}
      _ => {
        debug_assert!(false, "非法 L2 槽 tag (node {abs:#x})");
        return None;
      }
    }
    p += SLOT_TABLE_WORDS[2];

    // L3 表（8³ = 512 槽）
    if hdr & HDR_HAS_L3 == 0 {
      debug_assert!(false, "L2 Branch 但 L3 表缺失 (node {abs:#x})");
      return None;
    }
    let idx = slot_at(sub >> 1, 8);
    match self.slot(p, idx) {
      (SLOT_TAG_EMPTY, _) => return None,
      (SLOT_TAG_LEAF, pal) => return Some(pal),
      (SLOT_TAG_BRANCH, _) => {}
      _ => {
        debug_assert!(false, "非法 L3 槽 tag (node {abs:#x})");
        return None;
      }
    }
    p += SLOT_TABLE_WORDS[3];

    // L4 brick：字存 slab 号 + 1；palette 0 = 空（GPU 不存 brick 掩码，§3.4）
    if hdr & HDR_HAS_BRICK == 0 {
      debug_assert!(false, "L3 Branch 但 brick 缺失 (node {abs:#x})");
      return None;
    }
    let slab = (self.b_struct[p] - 1) as usize;
    let idx = slot_at(sub, 16);
    let word = self.b_leaves[slab * BRICK_SLAB_WORDS + (idx >> 2)];
    let pal = (word >> ((idx & 3) * 8)) as u8;
    (pal != 0).then_some(pal)
  }

  /// 槽表第 idx 槽解包为 (tag, palette)
  fn slot(&self, table: usize, idx: usize) -> (u16, u8) {
    let s = unpack_slot_word(self.b_struct[table + (idx >> 1)], idx & 1);
    (slot_tag(s), slot_palette(s))
  }

  /// cell 级占用查询（两级 DDA 粗步专用）：只走寻址链 ①TileIndex + ②TileBitmaps，
  /// 不下钻 node。cc 为 cell 坐标（1 单位 = 16 fine，同 VoxelPos::cell 粒度）。
  /// 语义：false ⇒ get_voxel 对该 cell 内全部 16³ fine 位置都返回 None。
  /// 成本：2 次 b_struct load（vs get_voxel 全链 4~10 次）。
  pub fn cell_occupied(&self, cc: IVec3) -> bool {
    // tile = cc >> 5（32 cell/tile，算术右移 = floor 除法，负坐标正确）；
    // in-tile cell = cc & 31（欧氏余数）
    let tile = cc >> 5;
    let it = cc & 31;
    let Some(ip) = index_pos(self.origin, self.dims, tile) else {
      return false;
    };
    let entry = self.b_struct[ip];
    if entry == 0 {
      return false;
    }
    let slot = (entry - 1) as usize;
    let ci = (it.z * 1024 + it.y * 32 + it.x) as usize;
    let bmp = BITMAP_BASE + slot * TILE_BITMAP_WORDS + ci / 32;
    self.b_struct[bmp] >> (ci % 32) & 1 != 0
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  /// 各级槽索引线性布局与 coords.rs `VoxelPos::slot_index` 同构（独立实现对照）
  #[test]
  fn slot_index_matches_voxel_pos() {
    for x in 0..16i32 {
      for y in 0..16i32 {
        for z in 0..16i32 {
          let in_cell = IVec3::new(x, y, z);
          for level in 1..=4u8 {
            let shift = (4 - level) as i32;
            let axis = 1i32 << level; // L1=2, L2=4, L3=8, L4=16
            let pos = VoxelPos::from_fine(in_cell, level);
            assert_eq!(pos.sub, in_cell >> shift, "level {level} sub");
            assert_eq!(
              slot_at(in_cell >> shift, axis),
              pos.slot_index(),
              "level {level} slot"
            );
          }
        }
      }
    }
  }
}
