//! CPU 砖块图构建器（Phase 1 + Phase 3 统一）
//!
//! 单 volume 路径（[`BrickMapBuilder`]）：`VolumeGrid` → wire 格式（wire.rs §b_struct 契约）：
//! - 全量构建 [`BrickMapBuilder::build_full`]：ChunkCoord 排序 + Rayon 并行
//!   `ChunkTree::serialize()` + 树区顺序 append（字节级确定性）
//! - 增量更新 [`BrickMapBuilder::update_chunk`]：单 chunk 重序列化 append +
//!   窗口条目改指新基址；旧树字节作废（计入 `globals.node_free_words`，
//!   全量重建归零压缩）——append-only，无原地修改（紧凑 DFS 格式编辑即失效）
//!
//! 多 volume 路径（[`VolumesBuilder`]，Phase 3 OBJ→Volume 统一）：
//! 持有 `Vec<BrickMapBuilder>`（主世界 + 物体），输出统一 `b_struct` + `b_palette`
//! + `GridDesc` 数组。各 volume 的 b_struct 顺序拼接，`GridDesc.tree_base`
//! 指向统一 buffer 内的绝对字基址。增量脏区间按 `tree_base` 偏移后传给 GPU
//! partial write；任一前置 volume 增长导致后续 tree_base 漂移 → 自动降级全量。
//!
//! 分配纪律：树区 append-only bump，零空闲链。旧版 pow2 桶/槽位/slab 空闲链
//! 全删除：新格式 palette 直存节点、chunk 树尺寸随内容任意变化，原地复用
//! 得不偿失（chunk 256³ 重建序列化 ≈ 数百 KB，PCIe 追加写远快于碎片整理）。

use std::collections::HashMap;

use gate_voxel::{ChunkCoord, VolumeGrid, VolumeTransform, Volumes};
use glam::{IVec3, Vec4};

use super::wire::{CHUNK_SIZE, GridDesc, pack_palette_entry};
use rayon::prelude::*;

use super::wire::{BrickMapBuffers, BrickMapGlobals, CHUNK_INDEX_CAP, PALETTE_WORDS, TREE_BASE};

/// 稠密 chunk 窗口线性位置（stride = CHUNK_INDEX_CAP，与 view.rs 同构）；窗口外 None
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

/// ChunkCoord 确定性排序键（IVec3 无 Ord，展开为分量元组）
fn coord_key(c: ChunkCoord) -> (i32, i32, i32) {
  (c.0.x, c.0.y, c.0.z)
}

/// chunk 是否有可渲染内容（HashMap 里可能残留空树，防御性排除）
fn chunk_has_content(grid: &VolumeGrid, c: ChunkCoord) -> bool {
  grid.chunk(c).is_some_and(|t| !t.is_empty())
}

/// 增量更新结果（调用方日志/测试用）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkUpdate {
  /// chunk 树重建完成（含新建）
  Rebuilt,
  /// chunk 已无数据，窗口条目清零
  Released,
  /// 窗口内但两侧都无内容（grid 无此 chunk 或本就未渲染）
  Unchanged,
  /// 超出稠密索引窗口（不渲染，构建时计数 rejected_tiles）
  OutsideWindow,
}

/// 砖块图构建器：持有与 GPU buffer 字节一致的持久状态
///
/// chunk 窗口（origin/dims，chunk 单位）构造时一次性确定；此后出现的窗口外
/// 新 chunk 不渲染（[`ChunkUpdate::OutsideWindow`]），全量重建可扩窗。
pub struct BrickMapBuilder {
  buffers: BrickMapBuffers,
  /// 已渲染 chunk → (树区绝对字基址, 树字数)
  base_of: HashMap<ChunkCoord, (usize, usize)>,
  origin: IVec3,
  dims: IVec3,
  rejected_chunks: u32,
  /// 作废树字节累计（增量 append 后旧树；全量重建归零）
  garbage_words: usize,
  /// 增量更新脏字节区间列表：每项 (lo_byte, hi_byte) 闭开，字对齐。
  /// 每次增量 = 1 个窗口条目字 + 1 段树 append，区间天然分离。
  dirty_struct: Vec<(usize, usize)>,
  dirty_palette: bool,
}

/// 脏字节区间列表（prepare 按此逐项 write_buffer 部分写 GPU）。
/// 空列表 = 对应 buffer 完全未修改，跳过写。palette 2KB 整块写。
#[derive(Debug, Default, Clone)]
pub struct DirtyRanges {
  pub struct_ranges: Vec<(usize, usize)>,
  pub palette_changed: bool,
}

impl BrickMapBuilder {
  // --- Dirty range tracking：记录被写的字节范围（闭开 [lo, hi)），字对齐 ---
  #[inline]
  fn mark_struct_words(&mut self, start_word: usize, count_words: usize) {
    if count_words == 0 {
      return;
    }
    let lo = start_word * 4;
    let hi = (start_word + count_words) * 4;
    if let Some(last) = self.dirty_struct.last_mut()
      && lo <= last.1
      && hi >= last.0
    {
      last.0 = last.0.min(lo);
      last.1 = last.1.max(hi);
      return;
    }
    self.dirty_struct.push((lo, hi));
  }

  /// 由 grid 包围盒确定 chunk 窗口（min - 1 起，跨度 +3 封顶 64³），不序列化内容
  ///
  /// 随后可逐 [`Self::update_chunk`] 累积内容（渐进式初载；等价性测试亦走此路径）。
  pub fn new_unbuilt(grid: &VolumeGrid) -> Self {
    let (origin, dims, rejected) = compute_window(grid);
    let mut b = Self {
      buffers: BrickMapBuffers {
        // Region ① 稠密 chunk 窗口（1MB）+ 空树区
        b_struct: vec![0; TREE_BASE],
        b_palette: vec![0; PALETTE_WORDS],
        globals: BrickMapGlobals {
          index_origin_x: origin.x,
          index_origin_y: origin.y,
          index_origin_z: origin.z,
          index_origin_w: 0,
          index_dims_x: dims.x as u32,
          index_dims_y: dims.y as u32,
          index_dims_z: dims.z as u32,
          index_dims_w: 0,
          tile_count: 0,
          node_words: 0,
          node_free_words: 0,
          brick_slabs: 0,
          brick_free: 0,
          rejected_tiles: rejected as u32,
          grid_count: 0,
          _pad1: 0,
          _pad2: 0,
          _pad3: 0,
          _pad4: 0,
        },
      },
      base_of: HashMap::new(),
      origin,
      dims,
      rejected_chunks: rejected as u32,
      garbage_words: 0,
      dirty_struct: Vec::new(),
      dirty_palette: false,
    };
    b.write_palette(grid);
    b
  }

  /// 全量构建（初始化/兜底）：确定性 + Rayon 并行序列化
  ///
  /// chunk 按 ChunkCoord 升序 append；并行仅化序列化（纯函数），append 顺序不变。
  pub fn build_full(grid: &VolumeGrid) -> Self {
    let mut b = Self::new_unbuilt(grid);
    let mut coords: Vec<ChunkCoord> = grid
      .chunk_coords()
      .filter(|&c| chunk_index_pos(b.origin, b.dims, c.0).is_some() && chunk_has_content(grid, c))
      .collect();
    coords.sort_by_key(|&c| coord_key(c));

    let blobs: Vec<Vec<u32>> = coords
      .par_iter()
      .map(|&c| grid.chunk(c).expect("chunk_has_content 已过滤").serialize())
      .collect();
    for (c, words) in coords.iter().zip(blobs) {
      b.append_chunk(*c, &words);
    }
    b.refresh_globals();
    // build_full 的调用方以整块写消费产物（mode_tag="full"），不消费脏区间；
    // 若不丢弃，append 累积的全场景 mark 会留存到后续增量路径，
    // 第一次 take_dirty_ranges 会带出全场景假区间。
    let _ = b.take_dirty_ranges();
    b
  }

  /// 逐 chunk 增量重建（DirtyTracker 吐出的每个 coord 调一次）：
  /// 重序列化 append + 窗口条目改指；旧树字节作废计入 node_free_words
  pub fn update_chunk(&mut self, grid: &VolumeGrid, coord: ChunkCoord) -> ChunkUpdate {
    if chunk_index_pos(self.origin, self.dims, coord.0).is_none() {
      return ChunkUpdate::OutsideWindow;
    }
    let has = chunk_has_content(grid, coord);
    let out = match self.base_of.get(&coord).copied() {
      None if !has => ChunkUpdate::Unchanged,
      None => {
        let words = grid.chunk(coord).expect("chunk_has_content").serialize();
        self.append_chunk(coord, &words);
        ChunkUpdate::Rebuilt
      }
      Some((_, old_words)) if !has => {
        self.release_chunk(coord, old_words);
        ChunkUpdate::Released
      }
      Some((_, old_words)) => {
        let words = grid.chunk(coord).expect("chunk_has_content").serialize();
        self.garbage_words += old_words;
        self.append_chunk(coord, &words);
        ChunkUpdate::Rebuilt
      }
    };
    self.refresh_globals();
    out
  }

  /// 调色板整表重铺（256 条全量 2KB，无增量必要）
  pub fn write_palette(&mut self, grid: &VolumeGrid) {
    for (i, [a, b]) in self
      .buffers
      .b_palette
      .as_chunks_mut::<2>()
      .0
      .iter_mut()
      .enumerate()
    {
      let [x, y] = pack_palette_entry(grid.palette().get(i as u8));
      *a = x;
      *b = y;
    }
    self.dirty_palette = true;
  }

  pub fn buffers(&self) -> &BrickMapBuffers {
    &self.buffers
  }

  /// 取走累积的增量脏字节区间列表（并重置）。
  ///
  /// 全量构建路径 (`build_full`) 不需要它：调用方应以 mode_tag=full 整块上传。
  pub fn take_dirty_ranges(&mut self) -> DirtyRanges {
    DirtyRanges {
      struct_ranges: std::mem::take(&mut self.dirty_struct),
      palette_changed: std::mem::take(&mut self.dirty_palette),
    }
  }

  /// chunk 当前树基址（b_struct 内绝对字址；None = 未渲染）
  pub fn chunk_base(&self, coord: ChunkCoord) -> Option<usize> {
    self.base_of.get(&coord).map(|&(b, _)| b)
  }

  pub fn origin(&self) -> IVec3 {
    self.origin
  }

  pub fn dims(&self) -> IVec3 {
    self.dims
  }

  // ---- 内部：append / release ----

  /// append 一个 chunk 的序列化树 + 写窗口条目（幂等覆盖同 chunk 旧条目）
  fn append_chunk(&mut self, coord: ChunkCoord, words: &[u32]) {
    let base = self.buffers.b_struct.len();
    let ip =
      chunk_index_pos(self.origin, self.dims, coord.0).expect("append_chunk 只接受窗口内 chunk");
    self.buffers.b_struct[ip] = base as u32 + 1;
    self.buffers.b_struct.extend_from_slice(words);
    self.base_of.insert(coord, (base, words.len()));
    self.mark_struct_words(ip, 1);
    self.mark_struct_words(base, words.len());
  }

  /// chunk 变空：条目清零 + 旧树作废
  fn release_chunk(&mut self, coord: ChunkCoord, old_words: usize) {
    let ip =
      chunk_index_pos(self.origin, self.dims, coord.0).expect("release_chunk 只接受窗口内 chunk");
    self.garbage_words += old_words;
    self.buffers.b_struct[ip] = 0;
    self.base_of.remove(&coord);
    self.mark_struct_words(ip, 1);
  }

  fn refresh_globals(&mut self) {
    let g = &mut self.buffers.globals;
    g.tile_count = self.base_of.len() as u32;
    g.node_words = (self.buffers.b_struct.len() - TREE_BASE) as u32;
    g.node_free_words = self.garbage_words as u32;
    g.brick_slabs = 0;
    g.brick_free = 0;
    g.rejected_tiles = self.rejected_chunks;
  }
}

// ============================================================================
// VolumesBuilder：多 volume 统一构建器（Phase 3 OBJ→Volume 统一）
// ============================================================================

/// u32 字切片 → 本机字节序 u8 Vec（wire 按小端直存，x86/ARM 均 LE；同 u8_of_u32 约定）
fn words_to_bytes(words: &[u32]) -> Vec<u8> {
  let mut v = Vec::with_capacity(words.len() * 4);
  // SAFETY: &[u32] → &[u8] 等长重解释，仅用于立即拷贝进 v
  v.extend_from_slice(unsafe {
    std::slice::from_raw_parts(words.as_ptr() as *const u8, words.len() * 4)
  });
  v
}

/// 多 volume 统一快照（与 GPU buffer 字节一一对应）
#[derive(Debug, Clone)]
pub struct VolumesSnapshot {
  /// full 模式：所有 volume 的 b_struct 顺序拼接；incremental 模式为空（内容走 struct_blobs）
  pub b_struct: Vec<u32>,
  /// full 模式：所有 volume 的 b_palette 顺序拼接；incremental 模式为空
  pub b_palette: Vec<u32>,
  /// 每 volume 一个 GridDesc（tree_base/palette_base 指向上述统一 buffer）
  pub grid_descs: Vec<GridDesc>,
  /// "full" = 整块写；"incremental" = 仅写脏块
  pub mode_tag: &'static str,
  /// incremental 模式：b_struct 脏块（统一 buffer 内字节偏移 + 内容）；full 为空
  pub struct_blobs: Vec<(usize, Vec<u8>)>,
  /// incremental 模式：b_palette 脏块（统一 buffer 内字节偏移 + 内容）；full 为空
  pub palette_blobs: Vec<(usize, Vec<u8>)>,
  /// 统一 b_struct 总字节数（两种模式都有效；buffer 容量 ensure 用）
  pub struct_total_bytes: usize,
  /// 统一 b_palette 总字节数
  pub palette_total_bytes: usize,
  /// 本轮更新的 dirty chunk 总数（跨所有 volume；日志用）
  pub dirty_chunks: usize,
}

/// 多 volume 统一构建器：持有 `Vec<BrickMapBuilder>`，输出统一 buffer + GridDesc 数组
///
/// 各 volume 的 `BrickMapBuilder` 独立维护 dirty 跟踪和增量 append。
/// `snapshot()` 顺序拼接各 volume 的 b_struct/b_palette，生成 GridDesc 数组。
/// 增量路径：若任一前置 volume 增长导致后续 tree_base 漂移，自动降级为全量。
pub struct VolumesBuilder {
  builders: Vec<BrickMapBuilder>,
  /// 每 volume 的变换（缓存自 Volumes，用于 GridDesc 生成）
  transforms: Vec<VolumeTransform>,
  /// 上一次 snapshot 的 tree_bases（字偏移）；空 = 首帧 → 强制全量
  prev_tree_bases: Vec<u32>,
  /// 上一次 snapshot 的 palette_bases（字偏移）
  prev_palette_bases: Vec<u32>,
  /// 强制全量标志（新增 volume /手动请求）
  force_full: bool,
}

impl VolumesBuilder {
  /// 全量构建所有 volume（初始化 / 兜底）
  pub fn build_full(volumes: &Volumes) -> Self {
    let mut builders = Vec::with_capacity(volumes.len());
    let mut transforms = Vec::with_capacity(volumes.len());
    for grid in volumes.all() {
      builders.push(BrickMapBuilder::build_full(grid));
      transforms.push(grid.transform());
    }
    Self {
      builders,
      transforms,
      prev_tree_bases: Vec::new(),
      prev_palette_bases: Vec::new(),
      force_full: true,
    }
  }

  /// 空构造（渐进式：先 new_unbuilt，再 update_chunk 累积）
  pub fn new_unbuilt(volumes: &Volumes) -> Self {
    let mut builders = Vec::with_capacity(volumes.len());
    let mut transforms = Vec::with_capacity(volumes.len());
    for grid in volumes.all() {
      builders.push(BrickMapBuilder::new_unbuilt(grid));
      transforms.push(grid.transform());
    }
    Self {
      builders,
      transforms,
      prev_tree_bases: Vec::new(),
      prev_palette_bases: Vec::new(),
      force_full: true,
    }
  }

  /// 同步 volume 数量（新增 volume 时追加 builder）+ 更新变换
  pub fn sync(&mut self, volumes: &Volumes) {
    while self.builders.len() < volumes.all().len() {
      let idx = self.builders.len();
      let grid = &volumes.all()[idx];
      self.builders.push(BrickMapBuilder::build_full(grid));
      self.transforms.push(grid.transform());
      self.force_full = true;
    }
    for (i, grid) in volumes.all().iter().enumerate() {
      self.transforms[i] = grid.transform();
    }
  }

  /// 逐 chunk 增量更新指定 volume
  pub fn update_chunk(
    &mut self,
    volumes: &Volumes,
    volume_idx: usize,
    coord: ChunkCoord,
  ) -> ChunkUpdate {
    self.builders[volume_idx].update_chunk(&volumes.all()[volume_idx], coord)
  }

  /// 标记全量重建（下帧 snapshot 走 full 路径）
  pub fn force_full(&mut self) {
    self.force_full = true;
  }

  /// 取走统一快照：拼接所有 volume 的 b_struct/b_palette + 生成 GridDesc 数组
  ///
  /// **布局策略**：物体 (1..N) 先放，主世界 (0) 后放。主世界编辑是高频常见路径，
  /// 放在尾部 → 其 b_struct 增长不漂移任何前置 volume 的 tree_base → 增量上传。
  /// GridDesc 数组仍按 volume 索引顺序 [0, 1, 2, ...]（主世界 = 0），tree_base
  /// 指向统一 buffer 内的实际位置。
  pub fn snapshot(&mut self) -> VolumesSnapshot {
    let n = self.builders.len();
    // b_struct 布局序：物体 1..N 先，主世界 0 后（主世界编辑不漂移物体）
    let layout_order: Vec<usize> = (1..n).chain(std::iter::once(0)).collect();

    // tree_bases[i] / palette_bases[i] = volume i 在统一 buffer 内的字基址。
    // 只累加字数，不拼接字节——全量拼接在增量帧是 100MB+ 级 memcpy（帧卡顿根因）。
    let mut tree_bases = vec![0u32; n];
    let mut palette_bases = vec![0u32; n];
    let mut struct_total_words = 0usize;
    let mut palette_total_words = 0usize;
    for &i in &layout_order {
      let buffers = self.builders[i].buffers();
      tree_bases[i] = struct_total_words as u32;
      palette_bases[i] = palette_total_words as u32;
      struct_total_words += buffers.b_struct.len();
      palette_total_words += buffers.b_palette.len();
    }

    // GridDesc 数组按 volume 索引顺序（0=主世界，1..N=物体）
    let mut grid_descs = Vec::with_capacity(n);
    for i in 0..n {
      let buffers = self.builders[i].buffers();
      let g = &buffers.globals;
      let tr = self.transforms[i];
      let origin = IVec3::new(g.index_origin_x, g.index_origin_y, g.index_origin_z);
      let dims = IVec3::new(
        g.index_dims_x as i32,
        g.index_dims_y as i32,
        g.index_dims_z as i32,
      );
      let mut desc = GridDesc::from_transform(
        tr.pos,
        tr.rot,
        tr.scale,
        tree_bases[i],
        palette_bases[i],
        g.tile_count,
        origin,
        dims,
      );
      if i == 0 {
        // 主世界（identity：局部=世界）：AABB = chunk 窗口范围（fine 单位）。
        // from_transform 默认给 [0,256]³·scale——窗口 origin 可为负且 dims 巨大，
        // 默认盒会把窗口绝大部分 slab 剔除 → 全屏只渲染 chunk(0,0,0) 附近一小块。
        desc.aabb_min = Vec4::new(
          (origin.x * CHUNK_SIZE as i32) as f32,
          (origin.y * CHUNK_SIZE as i32) as f32,
          (origin.z * CHUNK_SIZE as i32) as f32,
          0.0,
        );
        desc.aabb_max = Vec4::new(
          ((origin.x + dims.x) * CHUNK_SIZE as i32) as f32,
          ((origin.y + dims.y) * CHUNK_SIZE as i32) as f32,
          ((origin.z + dims.z) * CHUNK_SIZE as i32) as f32,
          0.0,
        );
      }
      grid_descs.push(desc);
    }

    // 漂移检测：volume 数变化 / 任一 tree_base 或 palette_base 变化 → 全量
    let bases_shifted = self.prev_tree_bases.len() != tree_bases.len()
      || self
        .prev_tree_bases
        .iter()
        .zip(tree_bases.iter())
        .any(|(p, c)| p != c)
      || self
        .prev_palette_bases
        .iter()
        .zip(palette_bases.iter())
        .any(|(p, c)| p != c);
    let need_full = self.force_full || bases_shifted;

    let dirty_chunks: usize = self.builders.iter().map(|b| b.dirty_struct.len()).sum();

    let mut b_struct = Vec::new();
    let mut b_palette = Vec::new();
    let mut struct_blobs = Vec::new();
    let mut palette_blobs = Vec::new();

    if need_full {
      // 全量路径：丢弃各 builder 的脏区间（整块写覆盖），拼接完整字节
      for b in &mut self.builders {
        let _ = b.take_dirty_ranges();
      }
      for &i in &layout_order {
        let buffers = self.builders[i].buffers();
        b_struct.extend_from_slice(&buffers.b_struct);
        b_palette.extend_from_slice(&buffers.b_palette);
      }
    } else {
      // 增量路径：只把各 builder 的脏区间内容拷贝成独立字节块（KB~MB 级），
      // 偏移按 tree_base/palette_base 平移到统一 buffer。prepare 逐块 write_buffer。
      for i in 0..n {
        let dr = self.builders[i].take_dirty_ranges();
        let tb = tree_bases[i] as usize;
        let pb = palette_bases[i] as usize;
        let buffers = self.builders[i].buffers();
        for (lo, hi) in dr.struct_ranges {
          let (lw, hw) = (lo / 4, hi / 4);
          struct_blobs.push((tb * 4 + lo, words_to_bytes(&buffers.b_struct[lw..hw])));
        }
        if dr.palette_changed {
          palette_blobs.push((pb * 4, words_to_bytes(&buffers.b_palette)));
        }
      }
    }

    self.prev_tree_bases = tree_bases;
    self.prev_palette_bases = palette_bases;
    self.force_full = false;

    VolumesSnapshot {
      b_struct,
      b_palette,
      grid_descs,
      mode_tag: if need_full { "full" } else { "incremental" },
      struct_blobs,
      palette_blobs,
      struct_total_bytes: struct_total_words * 4,
      palette_total_bytes: palette_total_words * 4,
      dirty_chunks,
    }
  }

  /// 单 volume 的 chunk 基址（调试/DDA 参考用）
  pub fn chunk_base(&self, volume_idx: usize, coord: ChunkCoord) -> Option<usize> {
    self.builders[volume_idx].chunk_base(coord)
  }

  pub fn len(&self) -> usize {
    self.builders.len()
  }

  pub fn is_empty(&self) -> bool {
    self.builders.is_empty()
  }
}

/// 窗口计算：原点 = 最小非空 chunk - 1（±1 chunk 余量），跨度 = max - min + 3，
/// 封顶 64³（CHUNK_INDEX_CAP）
fn compute_window(grid: &VolumeGrid) -> (IVec3, IVec3, usize) {
  let mut min = IVec3::splat(i32::MAX);
  let mut max = IVec3::splat(i32::MIN);
  let mut any = false;
  for c in grid.chunk_coords() {
    if chunk_has_content(grid, c) {
      any = true;
      min = min.min(c.0);
      max = max.max(c.0);
    }
  }
  if !any {
    return (IVec3::ZERO, IVec3::ZERO, 0);
  }
  let origin = min - IVec3::ONE;
  let dims = (max - min + IVec3::splat(3)).min(IVec3::splat(CHUNK_INDEX_CAP as i32));
  let rejected = grid
    .chunk_coords()
    .filter(|&c| chunk_has_content(grid, c) && chunk_index_pos(origin, dims, c.0).is_none())
    .count();
  (origin, dims, rejected)
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::brickmap::BrickMapView;
  use gate_voxel::{fill_box, fill_sphere};

  /// view 与 grid 在给定区域逐最细格一致（stride 控制采样密度）
  fn assert_view_matches(
    grid: &VolumeGrid,
    b: &BrickMapBuilder,
    lo: IVec3,
    hi: IVec3,
    stride: i32,
  ) {
    let view = BrickMapView::new(b.buffers());
    let mut z = lo.z;
    while z < hi.z {
      let mut y = lo.y;
      while y < hi.y {
        let mut x = lo.x;
        while x < hi.x {
          let f = IVec3::new(x, y, z);
          assert_eq!(
            view.get_voxel(f),
            grid.get_voxel(gate_voxel::VoxelCoord::from_ivec3(f)),
            "fine {f:?} 不一致"
          );
          x += stride;
        }
        y += stride;
      }
      z += stride;
    }
  }

  #[test]
  fn empty_grid_builds_empty_buffers() {
    let grid = VolumeGrid::new();
    let b = BrickMapBuilder::build_full(&grid);
    let g = &b.buffers().globals;
    assert_eq!(g.tile_count, 0);
    assert_eq!(g.index_dims_x, 0);
    assert_eq!(g.index_dims_y, 0);
    assert_eq!(g.index_dims_z, 0);
    assert_eq!(g.node_words, 0);
    assert_eq!(g.node_free_words, 0);
    assert_eq!(g.brick_slabs, 0);
    assert_eq!(g.rejected_tiles, 0);
    assert_eq!(b.buffers().b_struct.len(), TREE_BASE);
    assert_eq!(b.buffers().b_palette.len(), PALETTE_WORDS);
    // 调色板 0 条目 = AIR 全零
    assert_eq!(&b.buffers().b_palette[..2], &[0, 0]);
    let v = BrickMapView::new(b.buffers());
    assert_eq!(v.get_voxel(IVec3::new(123, -456, 789)), None);
    assert_eq!(v.get_voxel(IVec3::ZERO), None);
  }

  #[test]
  fn single_voxel_window_and_readback() {
    let mut grid = VolumeGrid::new();
    grid.set_voxel_ivec3(IVec3::new(5, 6, 7), 3);
    let b = BrickMapBuilder::build_full(&grid);
    {
      let g = &b.buffers().globals;
      assert_eq!(g.tile_count, 1);
      assert_eq!(
        g.node_words as usize,
        b.buffers().b_struct.len() - TREE_BASE
      );
      assert!(g.node_words > 0, "单体素分裂树非空");
      // 单 chunk 窗口：min-1 起 +3
      assert_eq!(
        (g.index_origin_x, g.index_origin_y, g.index_origin_z),
        (-1, -1, -1)
      );
      assert_eq!((g.index_dims_x, g.index_dims_y, g.index_dims_z), (3, 3, 3));
      // 窗口条目 = 树基址 + 1
      let ip = chunk_index_pos(b.origin(), b.dims(), IVec3::ZERO).unwrap();
      let entry = b.buffers().b_struct[ip] as usize;
      assert_eq!(entry, TREE_BASE + 1);
      assert_eq!(b.chunk_base(ChunkCoord::new(0, 0, 0)), Some(TREE_BASE));
    }
    let v = BrickMapView::new(b.buffers());
    assert_eq!(v.get_voxel(IVec3::new(5, 6, 7)), Some(3));
    assert_eq!(v.get_voxel(IVec3::new(6, 6, 7)), None);
    assert_eq!(
      v.get_voxel(IVec3::new(255, 255, 255)),
      None,
      "同 chunk 邻域空"
    );
    assert_eq!(
      v.get_voxel(IVec3::new(256, 0, 0)),
      None,
      "相邻 chunk 窗口内无树"
    );
    assert_eq!(v.get_voxel(IVec3::new(-1, 0, 0)), None);
    assert!(v.cell_occupied(IVec3::new(0, 0, 0)));
    assert!(!v.cell_occupied(IVec3::new(1, 0, 0)));
  }

  #[test]
  fn multichunk_negative_coords_view_equivalence() {
    // 跨 chunk（256 边界）+ 负坐标 chunk
    let mut grid = VolumeGrid::new();
    fill_box(&mut grid, IVec3::new(0, 0, 0), IVec3::new(300, 16, 16), 1);
    fill_sphere(&mut grid, IVec3::new(300, 128, 128), 40, 2);
    fill_box(&mut grid, IVec3::new(-10, 4, 4), IVec3::new(20, 8, 8), 4);
    fill_box(&mut grid, IVec3::new(-300, -300, -300), IVec3::splat(64), 5);
    let b = BrickMapBuilder::build_full(&grid);
    let g = &b.buffers().globals;
    // 内容 chunk：x∈{-2,-1,0,1}，y/z∈{-2..0}：窗口必含负分量
    assert!(g.tile_count >= 4, "应覆盖多个 chunk（含负坐标）");
    assert!(g.index_origin_x < 0 && g.index_origin_y < 0);
    assert_view_matches(
      &grid,
      &b,
      IVec3::new(-310, -310, -310),
      IVec3::splat(310),
      7,
    );
  }

  #[test]
  fn full_vs_incremental_byte_identical() {
    let mut grid = VolumeGrid::new();
    fill_box(&mut grid, IVec3::new(0, 0, 0), IVec3::new(300, 32, 32), 1);
    fill_sphere(&mut grid, IVec3::new(280, 100, 100), 30, 2);
    fill_box(&mut grid, IVec3::new(-6, 2, 2), IVec3::new(4, 4, 4), 4);
    grid.set_voxel_ivec3(IVec3::new(600, 600, 600), 7);

    let full = BrickMapBuilder::build_full(&grid);

    // 增量累积：同窗口空起，逐 chunk 重建（排序确定性顺序）
    let mut inc = BrickMapBuilder::new_unbuilt(&grid);
    let mut coords: Vec<ChunkCoord> = grid
      .chunk_coords()
      .filter(|&c| chunk_has_content(&grid, c))
      .collect();
    coords.sort_by_key(|&c| coord_key(c));
    for c in coords {
      assert_eq!(inc.update_chunk(&grid, c), ChunkUpdate::Rebuilt);
    }

    assert_eq!(
      full.buffers().b_struct,
      inc.buffers().b_struct,
      "b_struct 字节级一致"
    );
    assert_eq!(full.buffers().b_palette, inc.buffers().b_palette);
    assert_eq!(full.buffers().globals, inc.buffers().globals);
    for c in grid.chunk_coords() {
      assert_eq!(full.chunk_base(c), inc.chunk_base(c));
    }
  }

  #[test]
  fn full_build_leaves_no_stale_dirty_marks() {
    let mut grid = VolumeGrid::new();
    fill_box(&mut grid, IVec3::ZERO, IVec3::splat(64), 1);
    fill_box(&mut grid, IVec3::new(300, 0, 0), IVec3::splat(64), 2);
    let mut b = BrickMapBuilder::build_full(&grid);
    let stale = b.take_dirty_ranges();
    assert!(
      stale.struct_ranges.is_empty() && !stale.palette_changed,
      "build_full 不得残留脏区间"
    );
    // 一次真实增量后，脏区间只含该 chunk 自身（条目字 + 新树 << 树区总量）
    grid.set_voxel_ivec3(IVec3::new(3, 3, 3), 5);
    assert_eq!(
      b.update_chunk(&grid, ChunkCoord::new(0, 0, 0)),
      ChunkUpdate::Rebuilt
    );
    let d = b.take_dirty_ranges();
    let total: usize = d.struct_ranges.iter().map(|&(a, hi)| hi - a).sum();
    assert!(total > 0, "增量应有脏区间");
    // 2 个 chunk 树 + 窗口条目远小于 Region ① 的 1MB
    assert!(
      total < 2 * 1024 * 1024,
      "增量脏区间应远小于 1MB 窗口，实际 {total}B"
    );
  }

  #[test]
  fn update_chunk_edit_release_garbage_and_rebuild() {
    let mut grid = VolumeGrid::new();
    fill_box(&mut grid, IVec3::new(0, 0, 0), IVec3::new(32, 32, 32), 1);
    grid.set_voxel_ivec3(IVec3::new(260, 10, 10), 2); // chunk (1,0,0)
    let mut b = BrickMapBuilder::build_full(&grid);
    let base0 = b.chunk_base(ChunkCoord::new(0, 0, 0)).unwrap();
    let base1 = b.chunk_base(ChunkCoord::new(1, 0, 0)).unwrap();
    // 排序 append：(0,0,0) 先于 (1,0,0) → 各树字数可由基址差反推
    let words0 = base1 - base0;
    let words1 = grid
      .chunk(ChunkCoord::new(1, 0, 0))
      .expect("chunk (1,0,0) 存在")
      .serialize()
      .len();

    // 编辑 1：既有 chunk 加体素 → append 新树，旧树作废
    assert!(grid.set_voxel_ivec3(IVec3::new(3, 3, 3), 5).is_some());
    assert_eq!(
      b.update_chunk(&grid, ChunkCoord::new(0, 0, 0)),
      ChunkUpdate::Rebuilt
    );
    assert_eq!(
      b.buffers().globals.node_free_words as usize,
      words0,
      "garbage += chunk(0,0,0) 旧树"
    );
    let v = BrickMapView::new(b.buffers());
    assert_eq!(v.get_voxel(IVec3::new(3, 3, 3)), Some(5));
    assert_eq!(v.get_voxel(IVec3::new(4, 3, 3)), Some(1)); // 背景仍在
    assert_eq!(v.get_voxel(IVec3::new(260, 10, 10)), Some(2));

    // 编辑 2：窗口内新 chunk (2,0,0)
    grid.set_voxel_ivec3(IVec3::new(2 * 256 + 7, 7, 7), 6);
    assert_eq!(
      b.update_chunk(&grid, ChunkCoord::new(2, 0, 0)),
      ChunkUpdate::Rebuilt
    );
    assert_eq!(
      BrickMapView::new(b.buffers()).get_voxel(IVec3::new(2 * 256 + 7, 7, 7)),
      Some(6)
    );
    assert_eq!(b.buffers().globals.tile_count, 3);

    // 编辑 3：清空 chunk (1,0,0) → 条目清零 + garbage
    let free_before = b.buffers().globals.node_free_words as usize;
    assert!(
      grid
        .clear_voxel(gate_voxel::VoxelCoord::new(260, 10, 10))
        .is_some()
    );
    assert_eq!(
      b.update_chunk(&grid, ChunkCoord::new(1, 0, 0)),
      ChunkUpdate::Released
    );
    assert_eq!(b.chunk_base(ChunkCoord::new(1, 0, 0)), None);
    assert_eq!(b.buffers().globals.tile_count, 2);
    assert_eq!(
      BrickMapView::new(b.buffers()).get_voxel(IVec3::new(260, 10, 10)),
      None
    );
    let free_after = b.buffers().globals.node_free_words as usize;
    assert_eq!(free_after - free_before, words1, "garbage += 旧树字数");

    // 编辑 4：窗口外 chunk → 不渲染不崩溃
    grid.set_voxel_ivec3(IVec3::new(50 * 256, 0, 0), 9);
    assert_eq!(
      b.update_chunk(&grid, ChunkCoord::new(50, 0, 0)),
      ChunkUpdate::OutsideWindow
    );
    assert_eq!(
      BrickMapView::new(b.buffers()).get_voxel(IVec3::new(50 * 256, 0, 0)),
      None
    );
  }

  #[test]
  fn reject_beyond_index_cap() {
    let mut grid = VolumeGrid::new();
    grid.set_voxel_ivec3(IVec3::new(5, 5, 5), 1);
    grid.set_voxel_ivec3(IVec3::new(200 * 256, 5, 5), 1); // chunk 跨度 201 > 64
    let b = BrickMapBuilder::build_full(&grid);
    let g = &b.buffers().globals;
    assert_eq!(g.index_dims_x, CHUNK_INDEX_CAP as u32);
    assert_eq!(g.rejected_tiles, 1);
    assert_eq!(g.tile_count, 1);
    let v = BrickMapView::new(b.buffers());
    assert_eq!(v.get_voxel(IVec3::new(5, 5, 5)), Some(1));
    assert_eq!(v.get_voxel(IVec3::new(200 * 256, 5, 5)), None);
  }

  #[test]
  fn fuzz_updates_match_semantics() {
    // 确定性伪随机编辑序列：增量重建后语义 == grid（覆盖 garbage/条目改指路径）
    let mut grid = VolumeGrid::new();
    fill_box(&mut grid, IVec3::new(0, 0, 0), IVec3::new(32, 32, 32), 1);
    fill_sphere(&mut grid, IVec3::new(-100, -100, -100), 20, 2);
    let mut b = BrickMapBuilder::build_full(&grid);

    // 编辑区跨 chunk 0 与负 chunk（各轴 ±300）
    let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut rnd = move || {
      seed ^= seed << 13;
      seed ^= seed >> 7;
      seed ^= seed << 17;
      seed
    };
    for step in 0..400u64 {
      let f = IVec3::new(
        (rnd() % 600) as i32 - 300,
        (rnd() % 600) as i32 - 300,
        (rnd() % 600) as i32 - 300,
      );
      if rnd() & 1 == 0 {
        grid.set_voxel_ivec3(f, (rnd() % 254 + 1) as u8);
      } else {
        grid.clear_voxel(gate_voxel::VoxelCoord::from_ivec3(f));
      }
      if step % 3 == 0 {
        for c in grid.dirty.drain_data_budget(16) {
          b.update_chunk(&grid, c);
        }
      }
      if step % 50 == 0 {
        for c in grid.dirty.drain_data_budget(1000) {
          b.update_chunk(&grid, c);
        }
        assert_view_matches(&grid, &b, IVec3::new(-20, -20, -20), IVec3::splat(20), 1);
      }
    }
    for c in grid.dirty.drain_data_budget(1000) {
      b.update_chunk(&grid, c);
    }
    assert_view_matches(&grid, &b, IVec3::new(-20, -20, -20), IVec3::splat(20), 1);
    // 终态与全量重建语义一致（布局可不同：增量有 garbage，内容必须一致）
    let fresh = BrickMapBuilder::build_full(&grid);
    assert_view_matches(&grid, &b, IVec3::splat(-300), IVec3::splat(300), 5);
    let (a, b2) = (
      BrickMapView::new(fresh.buffers()),
      BrickMapView::new(b.buffers()),
    );
    for z in -20..20 {
      for y in -20..20 {
        for x in -20..20 {
          let f = IVec3::new(x, y, z);
          assert_eq!(a.get_voxel(f), b2.get_voxel(f), "终态 {f:?} 不一致");
        }
      }
    }
  }

  // ===========================================================================
  // VolumesBuilder 测试（Phase 3 统一构建器）
  // ===========================================================================

  use gate_voxel::Volumes;
  use glam::{Mat3, Vec3, Vec4};

  /// 构造主世界 + 1 物体的 Volumes
  fn make_volumes(main_pal: u8, obj_pal: u8) -> Volumes {
    let mut main = VolumeGrid::new();
    fill_box(&mut main, IVec3::ZERO, IVec3::splat(32), main_pal);
    let mut vols = Volumes::new(main);
    let mut obj = VolumeGrid::new_object(0, Vec3::new(500.0, 0.0, 0.0), Mat3::IDENTITY, 1.0);
    fill_box(&mut obj, IVec3::ZERO, IVec3::splat(16), obj_pal);
    vols.list.push(obj);
    vols
  }

  #[test]
  fn volumes_builder_single_volume_matches_brickmap() {
    let mut main = VolumeGrid::new();
    fill_box(&mut main, IVec3::ZERO, IVec3::splat(32), 1);
    let vols = Volumes::new(main);

    let mut vb = VolumesBuilder::build_full(&vols);
    let snap = vb.snapshot();
    // 单 volume → GridDesc[0] tree_base=0, palette_base=0, identity transform
    assert_eq!(snap.grid_descs.len(), 1);
    let g = &snap.grid_descs[0];
    assert_eq!(g.tree_base, 0);
    assert_eq!(g.palette_base, 0);
    assert_eq!(g.pos_scale, Vec4::new(0.0, 0.0, 0.0, 1.0));
    assert_eq!(g.rot0, Vec4::new(1.0, 0.0, 0.0, 0.0));
    assert_eq!(g.chunk_count, 1);
    // b_struct = 主世界 builder 的 b_struct
    let single = BrickMapBuilder::build_full(&vols.list[0]);
    assert_eq!(snap.b_struct, single.buffers().b_struct);
    assert_eq!(snap.b_palette, single.buffers().b_palette);
    // 首帧 = full
    assert_eq!(snap.mode_tag, "full");
  }

  #[test]
  fn volumes_builder_two_volumes_layout_and_offsets() {
    let vols = make_volumes(1, 2);
    let mut vb = VolumesBuilder::build_full(&vols);
    // 先取一次 full snapshot 建立 prev_bases
    let snap0 = vb.snapshot();
    assert_eq!(snap0.mode_tag, "full");
    assert_eq!(snap0.grid_descs.len(), 2);

    // 布局：物体(1) 先，主世界(0) 后
    let obj_builder = BrickMapBuilder::build_full(&vols.list[1]);
    let main_builder = BrickMapBuilder::build_full(&vols.list[0]);

    // GridDesc[0] = 主世界，tree_base = 物体 b_struct 长度（主世界在尾部）
    let g0 = &snap0.grid_descs[0];
    assert_eq!(g0.tree_base as usize, obj_builder.buffers().b_struct.len());
    assert_eq!(g0.palette_base as usize, PALETTE_WORDS);
    // identity transform
    assert_eq!(g0.pos_scale, Vec4::new(0.0, 0.0, 0.0, 1.0));

    // GridDesc[1] = 物体，tree_base = 0（物体在头部）
    let g1 = &snap0.grid_descs[1];
    assert_eq!(g1.tree_base, 0);
    assert_eq!(g1.palette_base, 0);
    // 物体变换
    assert_eq!(g1.pos_scale, Vec4::new(500.0, 0.0, 0.0, 1.0));

    // b_struct = 物体 b_struct + 主世界 b_struct
    let mut expected = obj_builder.buffers().b_struct.clone();
    expected.extend_from_slice(&main_builder.buffers().b_struct);
    assert_eq!(snap0.b_struct, expected);

    // b_palette = 物体 palette + 主世界 palette
    let mut pal_expected = obj_builder.buffers().b_palette.clone();
    pal_expected.extend_from_slice(&main_builder.buffers().b_palette);
    assert_eq!(snap0.b_palette, pal_expected);
  }

  #[test]
  fn volumes_builder_main_edit_stays_incremental() {
    let vols = make_volumes(1, 2);
    let mut vb = VolumesBuilder::build_full(&vols);
    // 首帧 full
    let _ = vb.snapshot();

    // 编辑主世界（尾部 volume）→ tree_base 不漂移 → incremental
    let mut vols = vols;
    vols.main_mut().set_voxel_ivec3(IVec3::new(5, 5, 5), 7);
    let chunks = vols.main_mut().dirty.drain_data_budget(100);
    for c in chunks {
      vb.update_chunk(&vols, 0, c);
    }
    let snap = vb.snapshot();
    assert_eq!(
      snap.mode_tag, "incremental",
      "主世界编辑（尾部）不应漂移物体 tree_base"
    );
    // 增量脏块应非空
    assert!(!snap.struct_blobs.is_empty());
    // 增量模式不带整量字节
    assert!(snap.b_struct.is_empty() && snap.b_palette.is_empty());
    // GridDesc tree_base 未变
    assert_eq!(snap.grid_descs[1].tree_base, 0, "物体 tree_base 不变");
  }

  #[test]
  fn volumes_builder_object_edit_forces_full() {
    let vols = make_volumes(1, 2);
    let mut vb = VolumesBuilder::build_full(&vols);
    // 首帧 full
    let _ = vb.snapshot();

    // 编辑物体（头部 volume）→ 主世界 tree_base 漂移 → full
    let mut vols = vols;
    vols
      .object_mut(0)
      .unwrap()
      .set_voxel_ivec3(IVec3::new(5, 5, 5), 9);
    let chunks = vols.object_mut(0).unwrap().dirty.drain_data_budget(100);
    for c in chunks {
      vb.update_chunk(&vols, 1, c);
    }
    let snap = vb.snapshot();
    assert_eq!(
      snap.mode_tag, "full",
      "物体编辑（头部）漂移主世界 tree_base → 全量"
    );
    // 增量脏块为空（full 路径逐块上传无意义）
    assert!(snap.struct_blobs.is_empty());
  }

  #[test]
  fn volumes_builder_sync_adds_new_volume() {
    let mut main = VolumeGrid::new();
    fill_box(&mut main, IVec3::ZERO, IVec3::splat(16), 1);
    let mut vols = Volumes::new(main);
    let mut vb = VolumesBuilder::build_full(&vols);
    let _ = vb.snapshot();

    // 新增物体
    let id = vols.add_object(Vec3::new(300.0, 0.0, 0.0), Mat3::IDENTITY, 1.0);
    vols.object_mut(id).unwrap().set_voxel_ivec3(IVec3::ZERO, 5);
    vb.sync(&vols);
    assert_eq!(vb.len(), 2);

    let snap = vb.snapshot();
    assert_eq!(snap.mode_tag, "full", "新增 volume → full");
    assert_eq!(snap.grid_descs.len(), 2);
    // 物体 GridDesc
    let g1 = &snap.grid_descs[1];
    assert_eq!(g1.pos_scale, Vec4::new(300.0, 0.0, 0.0, 1.0));
    assert_eq!(g1.tree_base, 0, "物体在头部");
    // 主世界 GridDesc
    let g0 = &snap.grid_descs[0];
    assert!(g0.tree_base > 0, "主世界在物体之后");
  }

  #[test]
  fn volumes_builder_incremental_dirty_ranges_offset_correctly() {
    let vols = make_volumes(1, 2);
    let mut vb = VolumesBuilder::build_full(&vols);
    let _ = vb.snapshot(); // 首帧 full

    // 主世界编辑 → incremental，dirty 区间应在主世界 tree_base 之后
    let mut vols = vols;
    vols.main_mut().set_voxel_ivec3(IVec3::new(10, 10, 10), 3);
    let chunks = vols.main_mut().dirty.drain_data_budget(100);
    for c in chunks {
      vb.update_chunk(&vols, 0, c);
    }
    let snap = vb.snapshot();
    assert_eq!(snap.mode_tag, "incremental");
    // 主世界 tree_base = 物体 b_struct 长度
    let main_tree_base = snap.grid_descs[0].tree_base as usize * 4;
    // 所有脏块偏移应 ≥ main_tree_base（主世界在尾部）且不越界
    for (off, payload) in &snap.struct_blobs {
      assert!(
        *off >= main_tree_base,
        "blob 偏移 {} 应 ≥ 主世界 tree_base {}",
        off,
        main_tree_base
      );
      assert!(!payload.is_empty(), "blob 非空");
      assert!(
        off + payload.len() <= snap.struct_total_bytes,
        "blob 尾部 {} 不越界 {}",
        off + payload.len(),
        snap.struct_total_bytes
      );
    }
    assert!(!snap.struct_blobs.is_empty());
  }
}
