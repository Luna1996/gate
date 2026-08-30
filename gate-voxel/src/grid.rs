//! TileGrid：无界体素世界容器（P1.4）
//!
//! `HashMap<TileCoord, Tile>` 稀疏存储 + 调色板 + 脏标记联动。
//! 编辑 API 以最细格坐标（0.25cm 单位 i32³）+ 层级定位，跨 Tile 自动拆分。

use std::collections::HashMap;

use glam::IVec3;

use crate::coords::{Level, MAX_LEVEL, TileCoord, VoxelPos};
use crate::dirty::DirtyTracker;
use crate::palette::{AIR_INDEX, Palette};
use crate::tile::Tile;

/// 一次编辑产生的脏区域（P2.3 按此增量更新 GPU 砖块图）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirtyEdit {
  pub tile: TileCoord,
  pub level: Level,
  // 编辑操作粒度可再扩展；P1 先以 Tile 为上传粒度
}

#[derive(Debug, Default)]
pub struct TileGrid {
  /// Box 载荷：Tile 含 4KB occupancy，若内联进桶，百万 tile 时桶数组膨胀至 GB 级
  /// 且 rehash 反复搬运（P1.7 实测 12.3GB → 装箱后 ~24B/entry）
  tiles: HashMap<TileCoord, Box<Tile>>,
  palette: Palette,
  pub dirty: DirtyTracker,
  /// 元件层（P2.3 通道，P6 真数据）：每 tile u16[32768] = 组件 ID，空 tile 不占内存
  comp_layer: HashMap<TileCoord, Box<[u16; 32768]>>,
  /// 状态表（P2.3 通道，P3.3 占位呼吸，P6 模拟真数据）：
  /// Vec<[u32; 4]> 按元件 ID 下标；[类型, 状态值, 相位/时间戳, 预留]
  state_table: Vec<[u32; 4]>,
  /// state_table 全表脏（P3.3 每帧呼吸）—— 全 4KB 上传，不做增量
  pub state_dirty: bool,
}

/// 内存用量核算（P1.7 极限测试的预算断言依据）
///
/// 全部为**深尺寸**：Tile 内联（occupancy 4KB）+ 堆上哈希表/层级表/brick + comp/state。
/// 估算策略偏保守（只高不低）；不含分配器元数据与 false sharing。
#[derive(Debug, Clone, Copy)]
pub struct MemoryUsage {
  pub tile_count: usize,
  pub cell_count: usize,
  /// Tile 内联部分合计（含 occupancy 4KB/Tile）
  pub tile_inline_bytes: usize,
  /// Tile 堆上部分合计
  pub tile_heap_bytes: usize,
  /// 脏标记结构合计
  pub dirty_bytes: usize,
  /// 元件层合计（64KB × 已写 tile 数）
  pub comp_bytes: usize,
  /// 状态表合计（16B × 条目数）
  pub state_bytes: usize,
}

impl MemoryUsage {
  pub fn total_bytes(&self) -> usize {
    self.tile_inline_bytes
      + self.tile_heap_bytes
      + self.dirty_bytes
      + self.comp_bytes
      + self.state_bytes
  }
}

impl TileGrid {
  pub fn new() -> Self {
    Self {
      tiles: HashMap::new(),
      palette: Palette::default(),
      dirty: DirtyTracker::new(),
      comp_layer: HashMap::new(),
      state_table: vec![[0u32; 4]; 256], // 预留 256 条目 = u16 组件 ID 低 8bit（MVP）
      state_dirty: true,
    }
  }

  pub fn palette(&self) -> &Palette {
    &self.palette
  }

  pub fn palette_mut(&mut self) -> &mut Palette {
    &mut self.palette
  }

  pub fn tile(&self, coord: TileCoord) -> Option<&Tile> {
    self.tiles.get(&coord).map(|t| &**t)
  }

  /// 元件层数据引用（P2.3 上传用）
  pub fn comp_layer(&self) -> &HashMap<TileCoord, Box<[u16; 32768]>> {
    &self.comp_layer
  }
  /// StateTable 字节视图（P2.3 上传用）：4×u32 平铺，256 条目 × 16B = 4096B
  pub fn state_table_bytes(&self) -> &[u8] {
    let slice = &self.state_table[..];
    let byte_len = slice.len() * 16;
    unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, byte_len) }
  }
  /// StateTable 条目读写（P3.3 演示用，P6 真数据）
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
  /// 写单个基元胞的组件 ID（P6 洪泛接）
  pub fn set_comp(&mut self, tile: TileCoord, cell_idx: u16, comp_id: u16) {
    let arr = self
      .comp_layer
      .entry(tile)
      .or_insert_with(|| Box::new([0u16; 32768]));
    if arr[cell_idx as usize] != comp_id {
      arr[cell_idx as usize] = comp_id;
      self.dirty.mark_comp(tile);
    }
  }
  /// 读单个基元胞的组件 ID；未写的 tile/cell = 0（AIR/无元件）
  pub fn get_comp(&self, tile: TileCoord, cell_idx: u16) -> u16 {
    self
      .comp_layer
      .get(&tile)
      .map(|a| a[cell_idx as usize])
      .unwrap_or(0)
  }

  /// 全部 tile 坐标枚举（P2.2 构建器 / P2.3 上传遍历用）
  pub fn tile_coords(&self) -> impl Iterator<Item = TileCoord> + '_ {
    self.tiles.keys().copied()
  }

  pub fn tile_count(&self) -> usize {
    self.tiles.len()
  }

  /// 内存用量（深尺寸，估算偏保守；预算断言用）
  pub fn memory_usage(&self) -> MemoryUsage {
    let tile_inline = std::mem::size_of::<Tile>();
    let tile_count = self.tiles.len();
    let cell_count: usize = self.tiles.values().map(|t| t.cell_count()).sum();
    let tile_heap: usize = self.tiles.values().map(|t| t.heap_bytes()).sum();
    // 外层桶（现仅 TileCoord + Box 指针）+ Tile 载荷（Box 目标，连续堆块）
    let outer = self.tiles.capacity() * (std::mem::size_of::<(TileCoord, Box<Tile>)>() + 1) / 7 * 8;
    // comp_layer：每个已写 tile 64KB
    let comp_bytes = self.comp_layer.len() * std::mem::size_of::<[u16; 32768]>();
    // state_table：Vec<[u32; 4]> 连续堆块
    let state_bytes = self.state_table.capacity() * std::mem::size_of::<[u32; 4]>();
    MemoryUsage {
      tile_count,
      cell_count,
      tile_inline_bytes: tile_inline * tile_count + outer,
      tile_heap_bytes: tile_heap,
      dirty_bytes: self.dirty.heap_bytes(),
      comp_bytes,
      state_bytes,
    }
  }

  /// 包含点的最深叶颜色；空 = None（AIR）
  pub fn get_voxel(&self, fine: IVec3) -> Option<u8> {
    let pos = VoxelPos::from_fine(fine, MAX_LEVEL);
    self.tiles.get(&pos.tile)?.get_leaf(&pos)
  }

  /// 指定层级区域全同色查询（编辑器预览/拾取用）
  pub fn get_uniform(&self, fine: IVec3, level: Level) -> Option<u8> {
    let pos = VoxelPos::from_fine(fine, level);
    self.tiles.get(&pos.tile)?.get_uniform(&pos)
  }

  /// 写体素（跨 Tile 自动定位；palette 索引 0 = AIR 禁止，清体素用 clear_voxel）
  pub fn set_voxel(&mut self, fine: IVec3, level: Level, palette: u8) -> Option<DirtyEdit> {
    debug_assert!(palette != AIR_INDEX, "palette 0 is AIR; use clear_voxel");
    let pos = VoxelPos::from_fine(fine, level);
    let tile = self.tiles.entry(pos.tile).or_default();
    if tile.set_leaf(&pos, palette) {
      self.dirty.mark_data(pos.tile);
      Some(DirtyEdit {
        tile: pos.tile,
        level,
      })
    } else {
      None
    }
  }

  /// 清除体素
  pub fn clear_voxel(&mut self, fine: IVec3, level: Level) -> Option<DirtyEdit> {
    let pos = VoxelPos::from_fine(fine, level);
    let tile = self.tiles.get_mut(&pos.tile)?;
    if tile.clear_voxel(&pos) {
      tile.remove_cell_if_empty(pos.cell_index());
      self.dirty.mark_data(pos.tile);
      Some(DirtyEdit {
        tile: pos.tile,
        level,
      })
    } else {
      None
    }
  }

  /// 批量编辑：单次脏标记去重，返回实际生效的编辑数
  pub fn batch_edit(&mut self, ops: impl IntoIterator<Item = (IVec3, Level, u8)>) -> usize {
    let mut applied = 0;
    for (fine, level, palette) in ops {
      if self.set_voxel(fine, level, palette).is_some() {
        applied += 1;
      }
    }
    applied
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn set_get_across_tiles() {
    let mut grid = TileGrid::new();
    // 正负两侧跨 tile
    let a = IVec3::new(5, 5, 5);
    let b = IVec3::new(-5, 0, 0); // tile (-1,0,0)
    let c = IVec3::new(513, 513, 513); // tile (1,1,1)
    assert_eq!(
      grid.set_voxel(a, 4, 1),
      Some(DirtyEdit {
        tile: TileCoord::new(0, 0, 0),
        level: 4
      })
    );
    assert_eq!(
      grid.set_voxel(b, 2, 2),
      Some(DirtyEdit {
        tile: TileCoord::new(-1, 0, 0),
        level: 2
      })
    );
    assert_eq!(
      grid.set_voxel(c, 0, 3),
      Some(DirtyEdit {
        tile: TileCoord::new(1, 1, 1),
        level: 0
      })
    );
    assert_eq!(grid.tile_count(), 3);
    assert_eq!(grid.get_voxel(a), Some(1));
    assert_eq!(grid.get_voxel(b), Some(2));
    // L0 写覆盖一个基元胞（4cm = 16 最细格，fine 512..528）
    assert_eq!(grid.get_voxel(IVec3::new(527, 527, 527)), Some(3));
    // 同 tile 不同基元胞 → 未写 = AIR
    assert_eq!(grid.get_voxel(IVec3::new(600, 600, 600)), None);
  }

  #[test]
  fn batch_and_dirty_pipeline() {
    let mut grid = TileGrid::new();
    let ops = (0..100).map(|i| (IVec3::new(i, 0, 0), 4u8, 1u8));
    assert_eq!(grid.batch_edit(ops), 100);
    // 100 个 fine 点全在 tile (0,0,0) → 脏去重为 1
    assert_eq!(grid.dirty.data_dirty_count(), 1);
    let drained = grid.dirty.drain_data_budget(16);
    assert_eq!(drained.len(), 1);
    assert_eq!(grid.dirty.data_dirty_count(), 0);
    // 同位置重写同色 → 无脏
    grid.batch_edit((0..100).map(|i| (IVec3::new(i, 0, 0), 4u8, 1u8)));
    assert_eq!(grid.dirty.data_dirty_count(), 0);
  }

  #[test]
  fn tile_coords_enumeration() {
    let mut grid = TileGrid::new();
    grid.set_voxel(IVec3::new(5, 5, 5), 4, 1);
    grid.set_voxel(IVec3::new(-5, 0, 0), 4, 1);
    let mut coords: Vec<_> = grid.tile_coords().collect();
    coords.sort();
    assert_eq!(coords.len(), grid.tile_count());
    assert_eq!(
      coords,
      vec![TileCoord::new(-1, 0, 0), TileCoord::new(0, 0, 0)]
    );
  }

  #[test]
  fn clear_removes_cell_when_empty() {
    let mut grid = TileGrid::new();
    grid.set_voxel(IVec3::new(10, 10, 10), 4, 1);
    assert!(grid.clear_voxel(IVec3::new(10, 10, 10), 4).is_some());
    assert_eq!(grid.get_voxel(IVec3::new(10, 10, 10)), None);
    // 基元胞收缩干净：占用位与 cell 表项都消失
    let t = grid.tile(TileCoord::new(0, 0, 0)).unwrap();
    assert!(t.cell(0).is_none());
    assert_eq!(t.occupancy, [0; 512]);
  }
}
