//! CPU 砖块图构建器（P2.2）
//!
//! `TileGrid` → wire 格式（docs/brickmap.md §2/§3 契约）：
//! - 全量构建 [`BrickMapBuilder::build_full`]：TileCoord 排序 + Rayon 分批序列化 +
//!   顺序放置（字节级确定性，§5）
//! - 增量更新 [`BrickMapBuilder::update_tile`]：逐 Tile 重建（§5 协议：
//!   分配新块 → 序列化写入 → 释放旧块 → 重铺目录/位图）
//!
//! 分配纪律（§5 的落地细化）：
//! - NodeStream：新块无空闲时从 bump 区**精确尺寸**分配；释放块按 hdr 反推精确
//!   尺寸进 pow2 字桶空闲链（1..512，10 桶）；复用时取同桶 FIFO 首个适配块（确定性）
//! - BrickPool：slab 定长 1024 字，FIFO 空闲链；b_leaves 只增不缩（全量重建兜底时压缩）
//! - Tile 槽位：FIFO 空闲链复用；槽存在 ⟺ 上次更新时该 tile 有 ≥1 占用基元胞
//!
//! 全量与增量共用同一序列化/放置路径，且首建路径全部走精确 bump ⇒
//! 「全量构建 vs 逐 Tile 增量累积」字节级一致（§8.1 等价性测试）。
//!
//! P2.3 上传优化：增量更新时不再整块 140MB PCIe DMA。每次放置/释放/调色板写，
//! Builder 记录三份 buffer 的**脏字节区间** `[lo, hi)`（闭开，字对齐），
//! 调用方 `take_dirty_ranges()` 取出后交给 Prepare 阶段做 write_buffer(offset, slice)
//! 部分写。典型 palette swap 单 Tile 场景：struct 脏 ≈132KB（bitmap 1KB+dirs 128KB+
//! node stream 局部），leaves/palette 往往无脏（palette 索引变 palette 色不变）。

use std::collections::{HashMap, VecDeque};

use gate_voxel::{Brick, Cell, Slot, Tile, TileCoord, TileGrid};
use glam::IVec3;
use rayon::prelude::*;

use super::wire::{
  BITMAP_BASE, BRICK_SLAB_WORDS, BrickMapBuffers, BrickMapGlobals, CELL_DIR_WORDS, DIR_BASE,
  HDR_HAS_BRICK, HDR_HAS_L1, HDR_HAS_L2, HDR_HAS_L3, NODE_STREAM_BASE, PALETTE_WORDS,
  SLOT_TABLE_WORDS, SLOT_TAG_BRANCH, SLOT_TAG_EMPTY, SLOT_TAG_LEAF, TILE_BITMAP_WORDS,
  TILE_INDEX_CAP, encode_slot, pack_palette_entry, pack_slot_pair,
};

/// NodeStream 最大的胞节点字数（hdr + l1 4 + l2 32 + l3 256 + brick 1）
const MAX_NODE_WORDS: usize = 1 + 4 + 32 + 256 + 1;
/// pow2 空闲桶数（1, 2, 4, ..., 512）
const BUCKETS: usize = 10;
/// 并行序列化的单批字节预算（控制峰值临时内存；单 Tile 超预算时独占一批）
const BATCH_BUDGET_WORDS: usize = 16 << 20; // 64 MB

/// TileIndex 条目线性位置；窗口外返回 None
fn index_pos(origin: IVec3, dims: IVec3, coord: TileCoord) -> Option<usize> {
  let rel = coord.0 - origin;
  if rel.cmplt(IVec3::ZERO).any() || rel.cmpge(dims).any() {
    return None;
  }
  Some((rel.x + rel.y * TILE_INDEX_CAP + rel.z * TILE_INDEX_CAP * TILE_INDEX_CAP) as usize)
}

/// TileCoord 确定性排序键（IVec3 无 Ord，展开为分量元组）
fn coord_key(c: TileCoord) -> (i32, i32, i32) {
  (c.0.x, c.0.y, c.0.z)
}

/// CellNode 字数（1 + 各存在表；与 `write_cell` 严格一致）
fn cell_words(cell: &Cell) -> usize {
  1 + cell.l1.is_some() as usize * SLOT_TABLE_WORDS[1]
    + cell.l2.is_some() as usize * SLOT_TABLE_WORDS[2]
    + cell.l3.is_some() as usize * SLOT_TABLE_WORDS[3]
    + cell.l4.is_some() as usize
}

/// brick 字在胞内的相对字位置（表按 l1 → l2 → l3 → brick 定序）
fn brick_word_rel(cell: &Cell) -> u32 {
  (1 + cell.l1.is_some() as usize * SLOT_TABLE_WORDS[1]
    + cell.l2.is_some() as usize * SLOT_TABLE_WORDS[2]
    + cell.l3.is_some() as usize * SLOT_TABLE_WORDS[3]) as u32
}

fn enc_slot(s: &Slot) -> u16 {
  match s {
    Slot::Empty => encode_slot(SLOT_TAG_EMPTY, 0),
    Slot::Leaf(p) => encode_slot(SLOT_TAG_LEAF, *p),
    Slot::Branch => encode_slot(SLOT_TAG_BRANCH, 0),
  }
}

/// 序列化单个基元胞到 `out`（长度必须等于 `cell_words`）
///
/// uniform 胞只写 hdr（bits 0-7 = palette）；非 uniform 写 hdr + 依序存在的表链。
/// brick 字写 0 占位，放置阶段补 slab+1。
fn write_cell(cell: &Cell, out: &mut [u32]) {
  if let Some(p) = cell.uniform {
    out[0] = p as u32;
    return;
  }
  let mut hdr = 0u32;
  let mut pos = 1;
  if let Some(t) = &cell.l1 {
    hdr |= HDR_HAS_L1;
    for k in 0..SLOT_TABLE_WORDS[1] {
      out[pos + k] = pack_slot_pair(enc_slot(&t[2 * k]), enc_slot(&t[2 * k + 1]));
    }
    pos += SLOT_TABLE_WORDS[1];
  }
  if let Some(t) = &cell.l2 {
    hdr |= HDR_HAS_L2;
    for k in 0..SLOT_TABLE_WORDS[2] {
      out[pos + k] = pack_slot_pair(enc_slot(&t[2 * k]), enc_slot(&t[2 * k + 1]));
    }
    pos += SLOT_TABLE_WORDS[2];
  }
  if let Some(t) = &cell.l3 {
    hdr |= HDR_HAS_L3;
    for k in 0..SLOT_TABLE_WORDS[3] {
      out[pos + k] = pack_slot_pair(enc_slot(&t[2 * k]), enc_slot(&t[2 * k + 1]));
    }
    pos += SLOT_TABLE_WORDS[3];
  }
  if cell.l4.is_some() {
    hdr |= HDR_HAS_BRICK;
    out[pos] = 0; // 放置时补 slab+1
  }
  out[0] = hdr;
}

/// CPU Brick → 4KB slab 内容：槽 i 的 palette 字节进字 i/4 的字节道 i%4（小端）
///
/// 占用位未置位的槽写 0（CPU `clear()` 只清掩码会留陈旧 palette 值；
/// GPU 不存 brick 掩码，palette 0 即空，见 §3.4）
fn pack_brick(b: &Brick) -> [u32; BRICK_SLAB_WORDS] {
  let mut out = [0u32; BRICK_SLAB_WORDS];
  for (wi, w) in b.occupancy.iter().enumerate() {
    let mut bits = *w;
    while bits != 0 {
      let bit = bits.trailing_zeros() as usize;
      bits &= bits - 1;
      let slot = wi * 64 + bit;
      out[slot >> 2] |= (b.palette[slot] as u32) << ((slot & 3) * 8);
    }
  }
  out
}

/// 单 Tile 序列化产物（纯函数输出，与放置解耦 ⇒ Rayon 可并行）
#[derive(Debug, Clone, PartialEq, Eq)]
struct TileBlob {
  /// NodeStream 内容：各胞紧排（放置时逐胞搬运进分配块）
  node: Vec<u32>,
  /// TileBitmaps 槽内容（u64 占用位图小端拆为 u32，位布局逐位相同）
  bitmap: [u32; TILE_BITMAP_WORDS],
  spans: Vec<Span>,
  /// L4 brick slab 内容（顺序 = spans 中 brick 引用的 slab 下标）
  slabs: Vec<[u32; BRICK_SLAB_WORDS]>,
}

/// 一个基元胞在 `node` 中的位置与分配信息
#[derive(Debug, Clone, PartialEq, Eq)]
struct Span {
  cell: u16,
  /// `node` 内起始字
  start: u32,
  words: u32,
  /// Some((胞内 brick 字相对位置, slab 下标))
  brick: Option<(u32, u32)>,
}

/// 序列化一个 Tile（纯函数：输出只依赖 tile 内容，确定性）
fn serialize_tile(tile: &Tile) -> TileBlob {
  let mut blob = TileBlob {
    node: Vec::new(),
    bitmap: [0; TILE_BITMAP_WORDS],
    spans: Vec::new(),
    slabs: Vec::new(),
  };
  // 占用位图：u64 → 双 u32（小端拆分，bit i = cell_index i 逐位保持）
  for (k, w) in tile.occupancy.iter().enumerate() {
    blob.bitmap[2 * k] = *w as u32;
    blob.bitmap[2 * k + 1] = (*w >> 32) as u32;
  }
  // 基元胞按 cell_index 升序（确定性）
  for (wi, w) in tile.occupancy.iter().enumerate() {
    let mut bits = *w;
    while bits != 0 {
      let bit = bits.trailing_zeros();
      bits &= bits - 1;
      let idx = (wi * 64 + bit as usize) as u16;
      let cell = tile
        .cells
        .get(&idx)
        .expect("occupancy bit 置位 implies cell 存在（tile.rs 不变式）");
      let start = blob.node.len() as u32;
      let words = cell_words(cell) as u32;
      let brick = cell.l4.as_ref().map(|b| {
        let slab = blob.slabs.len() as u32;
        blob.slabs.push(pack_brick(b));
        (brick_word_rel(cell), slab)
      });
      blob.node.resize(start as usize + words as usize, 0);
      write_cell(cell, &mut blob.node[start as usize..]);
      blob.spans.push(Span {
        cell: idx,
        start,
        words,
        brick,
      });
    }
  }
  blob
}

/// Tile 体积估算（字数）：序列化分批预算用（廉价：只看表存在标志）
fn tile_blob_words(tile: &Tile) -> usize {
  tile
    .cells
    .values()
    .map(|c| cell_words(c) + c.l4.is_some() as usize * BRICK_SLAB_WORDS)
    .sum()
}

/// 增量更新结果（调用方日志/测试用）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TileUpdate {
  /// 胞内容重建完成（含新建槽）
  Rebuilt,
  /// Tile 已无数据，槽位释放
  Released,
  /// 窗口内但两侧都无内容（grid 无此 tile 或本就无槽）
  Unchanged,
  /// 超出稠密索引窗口（沙盒语义：不渲染，构建时已计数告警）
  OutsideWindow,
}

/// 砖块图构建器：持有与 GPU buffer 字节一致的持久状态（P2.3 直接上传）
///
/// 窗口（index_origin/index_dims）在构造时一次性确定；此后出现的窗口外新 tile
/// 不渲染（[`TileUpdate::OutsideWindow`]），全量重建可扩窗。
pub struct BrickMapBuilder {
  buffers: BrickMapBuffers,
  /// 已渲染 tile 的槽位表
  slot_of: HashMap<TileCoord, u32>,
  /// 释放待复用的槽位（FIFO）
  free_slots: VecDeque<u32>,
  next_slot: u32,
  /// NodeStream 已用字数（相对 NODE_STREAM_BASE）
  node_len: usize,
  /// pow2 字桶空闲链：bucket i 存 (相对偏移, 精确字数)，块尺寸 ∈ (2^i/2, 2^i]
  free_buckets: [VecDeque<(u32, u32)>; BUCKETS],
  /// 空闲 slab（FIFO）
  free_slabs: VecDeque<u32>,
  slab_count: u32,
  origin: IVec3,
  dims: IVec3,
  rejected_tiles: u32,
  /// 增量更新脏字节区间列表：每项 (lo_byte, hi_byte) 闭开，字对齐。
  /// 不合并成单区间，避免「DIR_BASE(≈32MB 起点) + bump_node(≈末尾)」两端都脏时
  /// union 出整块 140MB 的假区间。prepare 阶段逐项 write_partial 即可。
  dirty_struct: Vec<(usize, usize)>,
  dirty_leaves: Vec<(usize, usize)>,
  dirty_palette: bool,
}

/// 三个 buffer 的脏字节区间列表（prepare 按此逐项部分写 GPU）。
/// 空列表 = 对应 buffer 完全未修改，跳过写。palette 2048B 整块写。
#[derive(Debug, Default, Clone)]
pub struct DirtyRanges {
  pub struct_ranges: Vec<(usize, usize)>,
  pub leaves_ranges: Vec<(usize, usize)>,
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
    // 与前一条相接/重叠就合并，控制列表项数（place_tile 内 span 多但集中 → 1 条 per 节点）
    if let Some(last) = self.dirty_struct.last_mut() {
      if lo <= last.1 && hi >= last.0 {
        last.0 = last.0.min(lo);
        last.1 = last.1.max(hi);
        return;
      }
    }
    self.dirty_struct.push((lo, hi));
  }
  #[inline]
  fn mark_leaves_words(&mut self, start_word: usize, count_words: usize) {
    if count_words == 0 {
      return;
    }
    let lo = start_word * 4;
    let hi = (start_word + count_words) * 4;
    if let Some(last) = self.dirty_leaves.last_mut() {
      if lo <= last.1 && hi >= last.0 {
        last.0 = last.0.min(lo);
        last.1 = last.1.max(hi);
        return;
      }
    }
    self.dirty_leaves.push((lo, hi));
  }

  /// 由 grid 包围盒确定索引窗口（min - 1 起，跨度 +3 封顶 128³），不序列化内容
  ///
  /// 随后可逐 [`Self::update_tile`] 累积内容（渐进式初载；§8.1 等价性测试亦走此路径）。
  /// bbox 只统计有占用基元胞的 tile（空 tile 不占窗口/槽位）。
  pub fn new_unbuilt(grid: &TileGrid) -> Self {
    let (origin, dims, rejected) = compute_window(grid);
    let mut b = Self {
      buffers: BrickMapBuffers {
        b_struct: vec![0; NODE_STREAM_BASE],
        b_leaves: Vec::new(),
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
          _pad0: 0,
          _pad1: 0,
          _pad2: 0,
          _pad3: 0,
          _pad4: 0,
        },
      },
      slot_of: HashMap::new(),
      free_slots: VecDeque::new(),
      next_slot: 0,
      node_len: 0,
      free_buckets: Default::default(),
      free_slabs: VecDeque::new(),
      slab_count: 0,
      origin,
      dims,
      rejected_tiles: rejected as u32,
      dirty_struct: Vec::new(),
      dirty_leaves: Vec::new(),
      dirty_palette: false,
    };
    b.write_palette(grid);
    b
  }

  /// 全量构建（初始化/兜底）：确定性 + Rayon 并行分块序列化
  ///
  /// Tile 按 TileCoord 升序放置；分批仅并行化序列化（纯函数），放置顺序不变。
  pub fn build_full(grid: &TileGrid) -> Self {
    let mut b = Self::new_unbuilt(grid);
    let mut coords: Vec<TileCoord> = grid
      .tile_coords()
      .filter(|&c| {
        index_pos(b.origin, b.dims, c).is_some() && grid.tile(c).is_some_and(|t| t.cell_count() > 0)
      })
      .collect();
    coords.sort_by_key(|&c| coord_key(c));

    let mut i = 0;
    while i < coords.len() {
      // 按体积预算分批（单 Tile 必整批：最坏全深度 Tile ~170MB > 预算）
      let mut j = i;
      let mut budget = 0usize;
      while j < coords.len() {
        let w = tile_blob_words(grid.tile(coords[j]).unwrap());
        if j > i && budget + w > BATCH_BUDGET_WORDS {
          break;
        }
        budget += w;
        j += 1;
      }
      let batch = &coords[i..j];
      let blobs: Vec<TileBlob> = batch
        .par_iter()
        .map(|&c| serialize_tile(grid.tile(c).unwrap()))
        .collect();
      for (c, blob) in batch.iter().zip(blobs) {
        b.place_tile(*c, &blob);
      }
      i = j;
    }
    b.refresh_globals();
    b
  }

  /// 逐 Tile 增量重建（§5 协议；DirtyTracker 吐出的每个 coord 调一次）
  pub fn update_tile(&mut self, grid: &TileGrid, coord: TileCoord) -> TileUpdate {
    if index_pos(self.origin, self.dims, coord).is_none() {
      return TileUpdate::OutsideWindow;
    }
    let has_cells = grid.tile(coord).is_some_and(|t| t.cell_count() > 0);
    let out = match self.slot_of.get(&coord).copied() {
      None if !has_cells => TileUpdate::Unchanged,
      None => {
        let blob = serialize_tile(grid.tile(coord).unwrap());
        self.place_tile(coord, &blob);
        TileUpdate::Rebuilt
      }
      Some(slot) if !has_cells => {
        self.release_slot(coord, slot);
        TileUpdate::Released
      }
      Some(_) => {
        let blob = serialize_tile(grid.tile(coord).unwrap());
        self.place_tile(coord, &blob);
        TileUpdate::Rebuilt
      }
    };
    self.refresh_globals();
    out
  }

  /// 调色板整表重铺（P2.3 的独立更新通道之一；256 条全量 2KB，无增量必要）
  pub fn write_palette(&mut self, grid: &TileGrid) {
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
      leaves_ranges: std::mem::take(&mut self.dirty_leaves),
      palette_changed: std::mem::take(&mut self.dirty_palette),
    }
  }

  /// tile 当前占用的槽位（P2.3 上传定位/测试断言用）
  pub fn slot_of_coord(&self, coord: TileCoord) -> Option<u32> {
    self.slot_of.get(&coord).copied()
  }

  pub fn origin(&self) -> IVec3 {
    self.origin
  }

  pub fn dims(&self) -> IVec3 {
    self.dims
  }

  // ---- 内部：放置与释放 ----

  /// 放置一个 tile 的序列化内容（有旧内容则按 §5 先分配后释放）
  fn place_tile(&mut self, coord: TileCoord, blob: &TileBlob) {
    let (slot, _existed) = self.ensure_slot(coord);
    let dir_base = DIR_BASE + slot as usize * CELL_DIR_WORDS;

    // 旧块快照（新块分配不会碰未释放的旧块；快照后才能在释放阶段按 hdr 反推尺寸）
    let old: Vec<u32> = if _existed {
      self.buffers.b_struct[dir_base..dir_base + CELL_DIR_WORDS]
        .iter()
        .copied()
        .filter(|&v| v != 0)
        .collect()
    } else {
      Vec::new()
    };

    // 新块分配 + 拷贝（先分配后释放，§5 步骤 1）
    let mut new_dirs: Vec<(u16, u32)> = Vec::with_capacity(blob.spans.len());
    for span in &blob.spans {
      let off = self.alloc_node(span.words);
      let base = NODE_STREAM_BASE + off as usize;
      self.buffers.b_struct[base..base + span.words as usize]
        .copy_from_slice(&blob.node[span.start as usize..(span.start + span.words) as usize]);
      self.mark_struct_words(base, span.words as usize);
      if let Some((rel, slab_i)) = span.brick {
        let slab = self.alloc_slab();
        let sb = slab as usize * BRICK_SLAB_WORDS;
        self.buffers.b_leaves[sb..sb + BRICK_SLAB_WORDS]
          .copy_from_slice(&blob.slabs[slab_i as usize]);
        self.mark_leaves_words(sb, BRICK_SLAB_WORDS);
        self.buffers.b_struct[base + rel as usize] = slab + 1;
        self.mark_struct_words(base + rel as usize, 1);
      }
      new_dirs.push((span.cell, (NODE_STREAM_BASE + off as usize) as u32));
    }

    // 释放旧块（§5 步骤 3；hdr 反推精确尺寸，brick 字带出 slab）
    // 注意：free_node_block 仅修改空闲链元数据，不写 b_struct/b_leaves，
    // 因此这里不需要 mark_*（被 free 覆盖的内存已经在新块里写过/或下一帧分配再写）。
    for v in old {
      self.free_node_block(v);
    }

    // 重铺目录 + 位图（§5 步骤 2 的 CPU 侧部分）
    for (ci, v) in &new_dirs {
      self.buffers.b_struct[dir_base + *ci as usize] = *v;
    }
    if !new_dirs.is_empty() {
      // ci 范围 0..CELL_DIR_WORDS，取最宽（实际不会覆盖全 CELL_DIR_WORDS，但保守标记）
      self.mark_struct_words(dir_base, CELL_DIR_WORDS);
    }
    let bmp_base = BITMAP_BASE + slot as usize * TILE_BITMAP_WORDS;
    self.buffers.b_struct[bmp_base..bmp_base + TILE_BITMAP_WORDS].copy_from_slice(&blob.bitmap);
    self.mark_struct_words(bmp_base, TILE_BITMAP_WORDS);
  }

  /// 槽位就绪：无则分配（FIFO 复用优先），写 TileIndex 条目；返回 (槽, 是否已有槽)
  fn ensure_slot(&mut self, coord: TileCoord) -> (u32, bool) {
    if let Some(&s) = self.slot_of.get(&coord) {
      return (s, true);
    }
    let s = self.free_slots.pop_front().unwrap_or_else(|| {
      let s = self.next_slot;
      self.next_slot += 1;
      s
    });
    self.slot_of.insert(coord, s);
    let ip = index_pos(self.origin, self.dims, coord).expect("place_tile 只接受窗口内 tile");
    self.buffers.b_struct[ip] = s + 1;
    self.mark_struct_words(ip, 1);
    (s, false)
  }

  /// 释放 tile 的全部块 + 槽位（tile 变空时）
  fn release_slot(&mut self, coord: TileCoord, slot: u32) {
    let dir_base = DIR_BASE + slot as usize * CELL_DIR_WORDS;
    let old: Vec<u32> = self.buffers.b_struct[dir_base..dir_base + CELL_DIR_WORDS]
      .iter()
      .copied()
      .filter(|&v| v != 0)
      .collect();
    // （free_node_block 不改 buffer，只改空闲链，无需 mark_*）
    for v in old {
      self.free_node_block(v);
    }
    self.buffers.b_struct[dir_base..dir_base + CELL_DIR_WORDS].fill(0);
    self.mark_struct_words(dir_base, CELL_DIR_WORDS);
    let bmp_base = BITMAP_BASE + slot as usize * TILE_BITMAP_WORDS;
    self.buffers.b_struct[bmp_base..bmp_base + TILE_BITMAP_WORDS].fill(0);
    self.mark_struct_words(bmp_base, TILE_BITMAP_WORDS);
    let ip = index_pos(self.origin, self.dims, coord).expect("release_slot 只接受窗口内 tile");
    self.buffers.b_struct[ip] = 0;
    self.mark_struct_words(ip, 1);
    self.slot_of.remove(&coord);
    self.free_slots.push_back(slot);
  }

  /// NodeStream 分配：同桶 FIFO 首个适配块，否则 bump 精确尺寸（确定性）
  fn alloc_node(&mut self, words: u32) -> u32 {
    debug_assert!(
      (1..=MAX_NODE_WORDS as u32).contains(&words),
      "胞节点字数越界: {words}"
    );
    let b = bucket_of(words);
    let bucket = &mut self.free_buckets[b];
    if let Some(i) = bucket.iter().position(|&(_, sz)| sz >= words) {
      let (off, _) = bucket.remove(i).expect("position 命中");
      return off;
    }
    let off = self.node_len as u32;
    let old_total = self.buffers.b_struct.len();
    self.node_len += words as usize;
    self
      .buffers
      .b_struct
      .resize(NODE_STREAM_BASE + self.node_len, 0);
    // 新增区 [old_total, new_total) 也属于本次分配（之后 place_tile 会再写 span.words
    // 的具体位置并 mark，这里兜底避免边界漏标）
    let new_total = self.buffers.b_struct.len();
    if new_total > old_total {
      self.mark_struct_words(old_total, new_total - old_total);
    }
    off
  }

  /// 释放胞节点块：hdr 反推精确尺寸；brick 标志连带释放 slab
  fn free_node_block(&mut self, abs: u32) {
    let hdr = self.buffers.b_struct[abs as usize];
    let mut words = 1usize;
    let mut brick_pos = None;
    if hdr & HDR_HAS_L1 != 0 {
      words += SLOT_TABLE_WORDS[1];
    }
    if hdr & HDR_HAS_L2 != 0 {
      words += SLOT_TABLE_WORDS[2];
    }
    if hdr & HDR_HAS_L3 != 0 {
      words += SLOT_TABLE_WORDS[3];
    }
    if hdr & HDR_HAS_BRICK != 0 {
      brick_pos = Some(abs as usize + words);
      words += 1;
    }
    debug_assert!(
      words <= MAX_NODE_WORDS,
      "hdr 反推块尺寸越界: {words} (hdr {hdr:#x})"
    );
    if let Some(pos) = brick_pos {
      let slab = self.buffers.b_struct[pos] - 1;
      self.free_slabs.push_back(slab);
    }
    let b = bucket_of(words as u32).min(BUCKETS - 1);
    self.free_buckets[b].push_back((abs - NODE_STREAM_BASE as u32, words as u32));
  }

  fn alloc_slab(&mut self) -> u32 {
    if let Some(s) = self.free_slabs.pop_front() {
      return s;
    }
    let s = self.slab_count;
    let old_total = self.buffers.b_leaves.len();
    self.slab_count += 1;
    self
      .buffers
      .b_leaves
      .resize(self.slab_count as usize * BRICK_SLAB_WORDS, 0);
    let new_total = self.buffers.b_leaves.len();
    if new_total > old_total {
      self.mark_leaves_words(old_total, new_total - old_total);
    }
    s
  }

  fn refresh_globals(&mut self) {
    let g = &mut self.buffers.globals;
    g.tile_count = self.slot_of.len() as u32;
    g.node_words = self.node_len as u32;
    g.node_free_words = self
      .free_buckets
      .iter()
      .map(|q| q.iter().map(|&(_, sz)| sz).sum::<u32>())
      .sum::<u32>();
    g.brick_slabs = self.slab_count;
    g.brick_free = self.free_slabs.len() as u32;
    g.rejected_tiles = self.rejected_tiles;
  }
}

/// pow2 桶下标：1→0, 2→1, 4→2, ..., 512→9
fn bucket_of(words: u32) -> usize {
  words.next_power_of_two().trailing_zeros() as usize
}

/// 窗口计算：原点 = 最小非空 tile - 1（±1 tile 余量），跨度 = max - min + 3，封顶 128³
fn compute_window(grid: &TileGrid) -> (IVec3, IVec3, usize) {
  let mut min = IVec3::splat(i32::MAX);
  let mut max = IVec3::splat(i32::MIN);
  let mut any = false;
  for c in grid.tile_coords() {
    if grid.tile(c).is_some_and(|t| t.cell_count() > 0) {
      any = true;
      min = min.min(c.0);
      max = max.max(c.0);
    }
  }
  if !any {
    return (IVec3::ZERO, IVec3::ZERO, 0);
  }
  let origin = min - IVec3::ONE;
  let dims = (max - min + IVec3::splat(3)).min(IVec3::splat(TILE_INDEX_CAP));
  let rejected = grid
    .tile_coords()
    .filter(|&c| {
      grid.tile(c).is_some_and(|t| t.cell_count() > 0) && index_pos(origin, dims, c).is_none()
    })
    .count();
  (origin, dims, rejected)
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::brickmap::BrickMapView;
  use gate_voxel::{MAX_LEVEL, fill_box, fill_sphere};

  /// view 与 grid 在给定区域逐最细格一致（stride 控制采样密度）
  fn assert_view_matches(grid: &TileGrid, b: &BrickMapBuilder, lo: IVec3, hi: IVec3, stride: i32) {
    let view = BrickMapView::new(b.buffers());
    let mut z = lo.z;
    while z < hi.z {
      let mut y = lo.y;
      while y < hi.y {
        let mut x = lo.x;
        while x < hi.x {
          let f = IVec3::new(x, y, z);
          assert_eq!(view.get_voxel(f), grid.get_voxel(f), "fine {f:?} 不一致");
          x += stride;
        }
        y += stride;
      }
      z += stride;
    }
  }

  #[test]
  fn empty_grid_builds_empty_buffers() {
    let grid = TileGrid::new();
    let b = BrickMapBuilder::build_full(&grid);
    let g = &b.buffers().globals;
    assert_eq!(g.tile_count, 0);
    assert_eq!(g.index_dims_x, 0);
    assert_eq!(g.index_dims_y, 0);
    assert_eq!(g.index_dims_z, 0);
    assert_eq!(g.node_words, 0);
    assert_eq!(g.brick_slabs, 0);
    assert_eq!(g.rejected_tiles, 0);
    assert_eq!(b.buffers().b_struct.len(), NODE_STREAM_BASE);
    assert_eq!(b.buffers().b_leaves.len(), 0);
    assert_eq!(b.buffers().b_palette.len(), PALETTE_WORDS);
    // 调色板 0 条目 = AIR 全零
    assert_eq!(&b.buffers().b_palette[..2], &[0, 0]);
    let v = BrickMapView::new(b.buffers());
    assert_eq!(v.get_voxel(IVec3::new(123, -456, 789)), None);
    assert_eq!(v.get_voxel(IVec3::ZERO), None);
  }

  #[test]
  fn single_cell_and_full_uniform_tile() {
    // 单个 L0 体素
    let mut grid = TileGrid::new();
    grid.set_voxel(IVec3::new(5, 6, 7), 0, 3);
    let b = BrickMapBuilder::build_full(&grid);
    {
      let g = &b.buffers().globals;
      assert_eq!(g.tile_count, 1);
      assert_eq!(g.node_words, 1, "uniform 胞 = 1 字");
      assert_eq!(g.index_origin_x, -1);
      assert_eq!(g.index_origin_y, -1);
      assert_eq!(g.index_origin_z, -1);
      assert_eq!(g.index_dims_x, 3);
      assert_eq!(g.index_dims_y, 3);
      assert_eq!(g.index_dims_z, 3);
    }
    let v = BrickMapView::new(b.buffers());
    for f in [IVec3::ZERO, IVec3::new(15, 15, 15), IVec3::new(5, 6, 7)] {
      assert_eq!(v.get_voxel(f), Some(3), "整胞 uniform 覆盖 {f:?}");
    }
    for f in [
      IVec3::new(16, 0, 0),
      IVec3::new(-1, 0, 0),
      IVec3::splat(512),
    ] {
      assert_eq!(v.get_voxel(f), None, "{f:?} 应为空");
    }

    // 整 tile 全 uniform（32768 胞 × 1 字）
    let mut grid = TileGrid::new();
    for z in 0..32 {
      for y in 0..32 {
        for x in 0..32 {
          grid.set_voxel(IVec3::new(x, y, z) * 16, 0, 9);
        }
      }
    }
    let b = BrickMapBuilder::build_full(&grid);
    let g = &b.buffers().globals;
    assert_eq!(g.tile_count, 1);
    assert_eq!(g.node_words, 32_768);
    assert_eq!(g.brick_slabs, 0);
    assert_view_matches(&grid, &b, IVec3::ZERO, IVec3::splat(512), 16);
    // 全部 32768 胞角点逐一验证（uniform 覆盖整胞）
    let v = BrickMapView::new(b.buffers());
    for cz in 0..32i32 {
      for cy in 0..32i32 {
        for cx in 0..32i32 {
          let f = IVec3::new(cx, cy, cz) * 16;
          assert_eq!(v.get_voxel(f), Some(9));
          assert_eq!(v.get_voxel(f + IVec3::splat(15)), Some(9));
        }
      }
    }
    assert_eq!(v.get_voxel(IVec3::splat(512)), None, "相邻 tile 应为空");
  }

  #[test]
  fn multires_scene_view_equivalence() {
    // 多分辨率混合 + 跨 tile + 负坐标 tile（§8.1 场景覆盖）
    let mut grid = TileGrid::new();
    fill_box(&mut grid, IVec3::new(0, 0, 0), IVec3::new(32, 16, 16), 0, 1);
    fill_sphere(&mut grid, IVec3::new(40, 8, 8), 6, 2, 2);
    fill_sphere(&mut grid, IVec3::new(100, 40, 40), 5, 4, 3);
    // L1 块落入负 tile
    fill_box(&mut grid, IVec3::new(-10, 4, 4), IVec3::new(8, 8, 8), 1, 4);
    // 孤立最细体素
    for p in [
      IVec3::new(70, 20, 3),
      IVec3::new(-20, -3, 9),
      IVec3::new(200, 300, 100),
    ] {
      grid.set_voxel(p, MAX_LEVEL, 5);
    }
    let b = BrickMapBuilder::build_full(&grid);
    let g = &b.buffers().globals;
    // 内容覆盖 tile (0,0,0) / (-1,0,0) / (-1,-1,0)：窗口原点必含负分量
    assert!(g.tile_count >= 3, "应覆盖多个 tile（含负坐标）");
    assert!(
      g.index_origin_x < 0 && g.index_origin_y < 0,
      "窗口应延伸到负坐标 tile"
    );
    assert!(g.brick_slabs > 0, "L4 球应产生 brick slab");
    // 主内容区 stride 1（~35 万点），远端孤立点 stride 1 小盒
    assert_view_matches(
      &grid,
      &b,
      IVec3::new(-24, -8, 0),
      IVec3::new(110, 48, 48),
      1,
    );
    assert_view_matches(
      &grid,
      &b,
      IVec3::new(190, 290, 90),
      IVec3::new(210, 310, 110),
      1,
    );
  }

  #[test]
  fn cross_tile_sphere() {
    // 球心贴近 tile 角落边界 → 跨 4~8 个 tile
    let mut grid = TileGrid::new();
    fill_sphere(&mut grid, IVec3::new(508, 500, 500), 30, 3, 6);
    let b = BrickMapBuilder::build_full(&grid);
    assert!(b.buffers().globals.tile_count >= 4);
    assert_view_matches(
      &grid,
      &b,
      IVec3::new(470, 460, 460),
      IVec3::new(546, 536, 536),
      1,
    );
  }

  #[test]
  fn full_vs_incremental_byte_identical() {
    let mut grid = TileGrid::new();
    fill_box(&mut grid, IVec3::new(0, 0, 0), IVec3::new(32, 32, 32), 0, 1);
    fill_sphere(&mut grid, IVec3::new(520, 30, 30), 12, 2, 2);
    fill_sphere(&mut grid, IVec3::new(30, 30, 30), 8, 4, 3);
    fill_box(&mut grid, IVec3::new(-6, 2, 2), IVec3::new(4, 4, 4), 1, 4);
    grid.set_voxel(IVec3::new(600, 600, 600), 0, 7);

    let full = BrickMapBuilder::build_full(&grid);

    // 增量累积：同窗口空起，逐 tile 重建（排序确定性顺序）
    let mut inc = BrickMapBuilder::new_unbuilt(&grid);
    let mut coords: Vec<TileCoord> = grid
      .tile_coords()
      .filter(|&c| grid.tile(c).is_some_and(|t| t.cell_count() > 0))
      .collect();
    coords.sort_by_key(|&c| coord_key(c));
    for c in coords {
      assert_eq!(inc.update_tile(&grid, c), TileUpdate::Rebuilt);
    }

    assert_eq!(
      full.buffers().b_struct,
      inc.buffers().b_struct,
      "b_struct 字节级一致"
    );
    assert_eq!(
      full.buffers().b_leaves,
      inc.buffers().b_leaves,
      "b_leaves 字节级一致"
    );
    assert_eq!(full.buffers().b_palette, inc.buffers().b_palette);
    assert_eq!(full.buffers().globals, inc.buffers().globals);
    for c in grid.tile_coords() {
      assert_eq!(full.slot_of_coord(c), inc.slot_of_coord(c));
    }
  }

  #[test]
  fn update_tile_edit_release_and_slot_reuse() {
    let mut grid = TileGrid::new();
    fill_box(&mut grid, IVec3::new(0, 0, 0), IVec3::new(32, 32, 32), 0, 1);
    grid.set_voxel(IVec3::new(512 + 10, 10, 10), 0, 2); // tile (1,0,0)
    let mut b = BrickMapBuilder::build_full(&grid);
    let slot1 = b.slot_of_coord(TileCoord::new(1, 0, 0)).unwrap();

    // 编辑 1：既有 tile 加 L4 体素（uniform 胞打散路径）
    assert!(grid.set_voxel(IVec3::new(3, 3, 3), MAX_LEVEL, 5).is_some());
    assert_eq!(
      b.update_tile(&grid, TileCoord::new(0, 0, 0)),
      TileUpdate::Rebuilt
    );
    let v = BrickMapView::new(b.buffers());
    assert_eq!(v.get_voxel(IVec3::new(3, 3, 3)), Some(5));
    assert_eq!(v.get_voxel(IVec3::new(4, 3, 3)), Some(1)); // 背景仍在

    // 编辑 2：窗口内新 tile (2,0,0)
    grid.set_voxel(IVec3::new(2 * 512 + 7, 7, 7), 2, 6);
    assert_eq!(
      b.update_tile(&grid, TileCoord::new(2, 0, 0)),
      TileUpdate::Rebuilt
    );
    let v = BrickMapView::new(b.buffers());
    assert_eq!(v.get_voxel(IVec3::new(2 * 512 + 7, 7, 7)), Some(6));
    assert_eq!(b.buffers().globals.tile_count, 3);

    // 编辑 3：清空 tile (1,0,0) → 槽释放
    grid.clear_voxel(IVec3::new(512 + 10, 10, 10), 0);
    assert_eq!(
      b.update_tile(&grid, TileCoord::new(1, 0, 0)),
      TileUpdate::Released
    );
    assert_eq!(b.slot_of_coord(TileCoord::new(1, 0, 0)), None);
    assert_eq!(b.buffers().globals.tile_count, 2);
    assert_eq!(
      BrickMapView::new(b.buffers()).get_voxel(IVec3::new(512 + 10, 10, 10)),
      None
    );

    // 编辑 4：tile (1,0,0) 复活 → FIFO 复用刚释放的槽
    grid.set_voxel(IVec3::new(512 + 20, 20, 20), 1, 8);
    assert_eq!(
      b.update_tile(&grid, TileCoord::new(1, 0, 0)),
      TileUpdate::Rebuilt
    );
    assert_eq!(
      b.slot_of_coord(TileCoord::new(1, 0, 0)),
      Some(slot1),
      "FIFO 复用"
    );
    assert_eq!(
      BrickMapView::new(b.buffers()).get_voxel(IVec3::new(512 + 20, 20, 20)),
      Some(8)
    );

    // 编辑 5：窗口外 tile → 不渲染不崩溃
    grid.set_voxel(IVec3::new(50 * 512, 0, 0), 0, 9);
    assert_eq!(
      b.update_tile(&grid, TileCoord::new(50, 0, 0)),
      TileUpdate::OutsideWindow
    );
    assert_eq!(
      BrickMapView::new(b.buffers()).get_voxel(IVec3::new(50 * 512, 0, 0)),
      None
    );
  }

  #[test]
  fn reject_beyond_index_cap() {
    let mut grid = TileGrid::new();
    grid.set_voxel(IVec3::new(5, 5, 5), 0, 1);
    grid.set_voxel(IVec3::new(200 * 512, 5, 5), 0, 1); // 跨度 202 > 128
    let b = BrickMapBuilder::build_full(&grid);
    let g = &b.buffers().globals;
    assert_eq!(g.index_dims_x, TILE_INDEX_CAP as u32);
    assert_eq!(g.rejected_tiles, 1);
    let v = BrickMapView::new(b.buffers());
    assert_eq!(v.get_voxel(IVec3::new(5, 5, 5)), Some(1));
    assert_eq!(v.get_voxel(IVec3::new(200 * 512, 5, 5)), None);
  }

  #[test]
  fn fuzz_updates_match_semantics() {
    // 确定性伪随机编辑序列：增量重建后语义 == grid（覆盖槽复用/空闲链/打散路径）
    let mut grid = TileGrid::new();
    fill_box(&mut grid, IVec3::new(0, 0, 0), IVec3::new(32, 32, 32), 0, 1);
    fill_sphere(&mut grid, IVec3::new(-200, -200, -200), 20, 4, 2);
    let mut b = BrickMapBuilder::build_full(&grid);

    // 编辑区跨 tile 0 与负 tile（各轴 ±300 覆盖 8 个 tile 邻域）
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
      let level = (rnd() % 5) as u8;
      if rnd() & 1 == 0 {
        grid.set_voxel(f, level, (rnd() % 254 + 1) as u8);
      } else {
        grid.clear_voxel(f, level);
      }
      if step % 3 == 0 {
        for c in grid.dirty.drain_data_budget(16) {
          b.update_tile(&grid, c);
        }
      }
      if step % 50 == 0 {
        for c in grid.dirty.drain_data_budget(1000) {
          b.update_tile(&grid, c);
        }
        // 跨 tile 边界小盒 stride 1 验证
        assert_view_matches(&grid, &b, IVec3::new(-20, -20, -20), IVec3::splat(20), 1);
      }
    }
    for c in grid.dirty.drain_data_budget(1000) {
      b.update_tile(&grid, c);
    }
    assert_view_matches(&grid, &b, IVec3::new(-20, -20, -20), IVec3::splat(20), 1);
    // 终态与全量重建语义一致（布局可因复用不同，内容必须一致；大区 stride 5）
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

  #[test]
  fn worst_tile_full_depth_stress() {
    use std::time::{Duration, Instant};
    // P2.2 极限：单 Tile 全深度（32768 胞 × 294 字 + 32768 slab）≈ 170MB GPU
    let mut grid = TileGrid::new();
    for cz in 0..32 {
      for cy in 0..32 {
        for cx in 0..32 {
          let fine = IVec3::new(cx, cy, cz) * 16;
          let p = ((cx + cy * 32 + cz * 1024) % 255 + 1) as u8;
          grid.set_voxel(fine, MAX_LEVEL, p);
        }
      }
    }
    let t0 = Instant::now();
    let b = BrickMapBuilder::build_full(&grid);
    let build = t0.elapsed();

    let g = &b.buffers().globals;
    assert_eq!(g.tile_count, 1);
    // 精确 bump：node = 32768 × 294 字（无 pow2 padding）
    assert_eq!(g.node_words as usize, 32_768 * 294);
    assert_eq!(g.brick_slabs, 32_768);
    assert_eq!(g.node_free_words, 0);
    assert_eq!(g.brick_free, 0);
    assert_eq!(
      b.buffers().b_leaves.len(),
      32_768 * BRICK_SLAB_WORDS,
      "slab 池 128MB"
    );
    // VRAM 预算断言（§7：b_struct + b_leaves + palette ≤ 2GB）
    let vram =
      (b.buffers().b_struct.len() + b.buffers().b_leaves.len() + b.buffers().b_palette.len()) * 4;
    println!(
      "P2.2 全深度单Tile: build={build:?}, node={} MB, slabs=128 MB, GPU 合计={} MB / 预算 2048 MB",
      g.node_words * 4 / 1_048_576,
      vram / 1_048_576,
    );
    assert!(
      build < Duration::from_secs(30),
      "全深度构建超预算: {build:?}"
    );
    assert!(vram <= 2 << 30, "单 tile VRAM 超预算: {vram} B");
    // 读回抽查（对角胞 + 中心胞；每胞 1 个 L4 体素 = 其 0.25cm 格）
    let v = BrickMapView::new(b.buffers());
    for cz in [0usize, 31] {
      for cy in [0usize, 31] {
        for cx in [0usize, 31] {
          let fine = IVec3::new(cx as i32, cy as i32, cz as i32) * 16;
          let p = ((cx + cy * 32 + cz * 1024) % 255 + 1) as u8;
          assert_eq!(v.get_voxel(fine), Some(p), "({cx},{cy},{cz}) 读回");
          assert_eq!(v.get_voxel(fine + IVec3::ONE), None, "邻格应空");
        }
      }
    }
  }
}
