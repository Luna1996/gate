//! CPU 砖块图构建器：`VolumeGrid` → wire 格式。
//! [`BrickMapBuilder`]（单 volume）：**每个 chunk 一个树块**（块首 = 根节点地址），块内是节点
//! arena —— 全量构建按 ChunkCoord 排序并行 `serialize_with_layout()` 后顺序安装；增量更新按
//! [`TreeDirty`] **只重写动过的节点**（字节没变就不写不标脏），节点在块内原地增删。
//! 一次编辑的上传量由此从"整棵 chunk 树"降到"路径上的几个节点"（几十~几百字节）。
//! 块级空闲段（[`FreeRuns`]）复用被释放的树块，空闲过半时压实（搬块、只改窗口条目）。
//! [`VolumesBuilder`] 拼接各 b_struct 后按 `tree_base` 偏移上传。

use std::collections::HashMap;

use gate_voxel::{
  ChunkCoord, ChunkTree, NODE_OFFSET_NONE, NodeLayout, NodeView, PALETTE_INDEX_MAX, PaletteId,
  ROOT_WIRE_WORDS, TreeDirty, VolumeGrid, VolumeTransform, Volumes, pack_palette_word,
};
use glam::{IVec3, Vec4};

use super::wire::{GridDesc, LEAF_INLINE_WORDS, NODE_FIXED_WORDS, pack_palette_entry};
use rayon::prelude::*;

use super::wire::{
  BrickMapBuffers, BrickMapGlobals, CHUNK_INDEX_CAP, PALETTE_BYTES_PER_ENTRY, PALETTE_WORDS,
  TREE_BASE,
};

/// 稠密 chunk 窗口线性位置（stride = `CHUNK_INDEX_CAP`，与 WGSL `trace.wesl` 的窗口寻址同构）；窗口外 None。
///
/// **M6**：窗口是**跟着相机走**的（`infinite_cubes` 每帧把它钉在相机为中心的 64³ chunk 上），
/// 移动时靠 [`BrickMapBuilder::set_window`] **平移索引区**（条目存的是块相对地址、与相位无关 ⇒ 只是搬家）
/// —— 于是"走得再远"也不需要丢掉 CPU chunk / 重传树块。
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

/// ChunkCoord 确定性排序键（展开为分量元组）。
fn coord_key(c: ChunkCoord) -> (i32, i32, i32) {
  (c.0.x, c.0.y, c.0.z)
}

/// 空闲段表：按起点升序、相邻自动合并，单位 = 字。
///
/// 两个使用者：① 全局（[`BrickMapBuilder::free`]）分配 / 回收**整块 chunk 树**；
/// ② 块内（[`ChunkSlot::free`]，偏移相对块首）分配 / 回收**单个节点块**。
///
/// 存在意义：增量更新若只靠追加，高水位会单调上升（空闲字节从不复用）⇒ buffer 周期性扩容并
/// 整份拷贝旧内容。first-fit 复用后高水位稳定在"并发存活总量"附近。
#[derive(Default, Debug)]
struct FreeRuns {
  runs: Vec<(usize, usize)>,
}

impl FreeRuns {
  /// first-fit 取一段 ≥ `n` 的空闲，返回段首（从段首切走 `n` 字）。
  fn alloc(&mut self, n: usize) -> Option<usize> {
    let i = self.runs.iter().position(|&(_, len)| len >= n)?;
    let (start, len) = self.runs[i];
    if len == n {
      self.runs.remove(i);
    } else {
      self.runs[i] = (start + n, len - n);
    }
    Some(start)
  }

  /// 若 `[start, start+len)` 完全落在某段空闲内则切走它（原地扩容用），否则不动并返回 false。
  fn take_range(&mut self, start: usize, len: usize) -> bool {
    if len == 0 {
      return true;
    }
    let Some(i) = self.runs.iter().position(|&(s, l)| s <= start && start + len <= s + l) else {
      return false;
    };
    let (s, l) = self.runs.remove(i);
    self.free(s, start - s);
    self.free(start + len, s + l - start - len);
    true
  }

  /// 归还 `[start, start+len)`（与前后相邻段合并）。
  fn free(&mut self, start: usize, len: usize) {
    if len == 0 {
      return;
    }
    let i = self.runs.partition_point(|&(s, _)| s < start);
    if i > 0 && self.runs[i - 1].0 + self.runs[i - 1].1 == start {
      self.runs[i - 1].1 += len;
      if i < self.runs.len() && self.runs[i - 1].0 + self.runs[i - 1].1 == self.runs[i].0 {
        let (_, l2) = self.runs.remove(i);
        self.runs[i - 1].1 += l2;
      }
      return;
    }
    if i < self.runs.len() && start + len == self.runs[i].0 {
      self.runs[i].0 = start;
      self.runs[i].1 += len;
      return;
    }
    self.runs.insert(i, (start, len));
  }

  /// 空闲总字数
  fn words(&self) -> usize {
    self.runs.iter().map(|&(_, l)| l).sum()
  }
}

/// chunk 是否有可渲染内容（排除残留空树）。
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

/// 节点槽未分配的哨兵
const NODE_NONE: u32 = u32::MAX;

/// 节点槽（索引 = 树节点 id）：节点在**块内**的字偏移 + 当前占用字数。
/// 不存 wire 层 —— 层号由调用方给（`TreeDirty` 的层 / 递归深度），节点 id 与层的对应关系恒定。
#[derive(Debug, Clone, Copy)]
struct NodeSlot {
  off: u32,
  words: u16,
}

impl NodeSlot {
  const UNASSIGNED: Self = Self { off: NODE_NONE, words: 0 };
}

/// 单 chunk 的树块：块首（= 根节点地址）+ 块内节点 arena。
///
/// 为什么是「块 + 块内 arena」而不是全局节点池：wire 的子块指针以**根地址**为基准
/// （shader 的 `chunk_base`），一个 chunk 的节点只能整体搬 —— 块内偏移全不变 ⇒ 搬块只需改
/// 窗口条目 1 个字，节点内容一字不动。块首由全局 [`FreeRuns`] 分配，块内节点由 `free` 分配。
#[derive(Debug)]
struct ChunkSlot {
  /// 块首（`b_struct` 内字址），恒等于根节点地址
  base: usize,
  /// 块容量（字；含末尾余量，见 [`BrickMapBuilder::install_blob`]）
  cap: usize,
  /// 块内空闲段（偏移相对 `base`）
  free: FreeRuns,
  /// 节点槽表（索引 = 节点 id）
  nodes: Vec<NodeSlot>,
}

impl ChunkSlot {
  /// 已占用字数（含根节点的固定预留区）
  fn used(&self) -> usize {
    self.cap - self.free.words()
  }
}

/// 节点在块内占用的字数（wire 形态）：根固定 [`ROOT_WIRE_WORDS`]（掩码增减不改根的字数 ⇒ 根永不搬迁）。
#[inline]
fn wire_words_of(id: u32, level: u8, mask: u64) -> usize {
  if id == 0 {
    ROOT_WIRE_WORDS
  } else if mask == 0 {
    NODE_FIXED_WORDS
  } else if level == 3 {
    NODE_FIXED_WORDS + LEAF_INLINE_WORDS
  } else {
    NODE_FIXED_WORDS + mask.count_ones() as usize
  }
}

/// 读 `b_struct` 里某节点的 64 位掩码
#[inline]
fn read_mask(buf: &[u32], at: usize) -> u64 {
  (buf[at + 1] as u64) << 32 | buf[at] as u64
}

/// 砖块图构建器：持有与 GPU buffer 字节一致的持久状态。
/// chunk 窗口（origin/dims，chunk 单位）构造时确定；窗口外新 chunk 不渲染，全量重建可扩窗。
pub struct BrickMapBuilder {
  buffers: BrickMapBuffers,
  /// 已渲染 chunk → 树块
  chunks: HashMap<ChunkCoord, ChunkSlot>,
  origin: IVec3,
  dims: IVec3,
  rejected_chunks: u32,
  /// 全局块级空闲段（字，`b_struct` 内绝对地址）
  free: FreeRuns,
  /// 增量更新脏字节区间列表：每项 `(lo_byte, hi_byte)` 闭开，字对齐。
  dirty_struct: Vec<(usize, usize)>,
  /// 待上传的调色板脏槽闭区间（同一帧多次 `write_palette` 取并集）；None = 无变动。
  dirty_palette: Option<(u16, u16)>,
  /// 调色板同步游标：本 builder 上次同步时调色板的写版本；None = 从未同步过。
  palette_synced_at: Option<u64>,
  /// **树区预留字数**（M8，0 = 不预留）：见 [`BrickMapBuilder::new_unbuilt_reserved`] 的说明。
  /// 预留区让 `b_struct` 的**长度恒定** ⇒ 排在后面的 volume 的 `tree_base` 不漂移。
  reserve: usize,
}

/// 脏字节区间列表（prepare 按此逐项 write_buffer 部分写 GPU）；空 = 未修改。
#[derive(Debug, Default, Clone)]
pub struct DirtyRanges {
  pub struct_ranges: Vec<(usize, usize)>,
  /// 调色板脏槽闭区间（槽号，非字节）；None = 本次无槽变动
  pub palette_range: Option<(u16, u16)>,
}

impl BrickMapBuilder {
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

  /// 由 grid 包围盒确定 chunk 窗口（min - 1 起，跨度 +3 封顶 64³），不序列化内容。
  /// 随后可逐 [`Self::update_chunk`] 累积内容（渐进式初载）。
  pub fn new_unbuilt(grid: &VolumeGrid) -> Self {
    Self::new_unbuilt_reserved(grid, 0)
  }

  /// 同 [`Self::new_unbuilt`]，但**预占一段固定长度的树区**（`reserve_words`）。
  ///
  /// WHY（M8）：[`VolumesBuilder::snapshot`] 的 `bases_shifted` 以"各 volume 的 `b_struct` 长度"为判据；
  /// 排在**主世界之前**的远场级一变长，主世界的 `tree_base` 就漂移 ⇒ 降级**全量快照**（把四个 volume
  /// 拼一遍 = 580 MB 的 memcpy + 上传，实测 `UPLOAD[full] … elapsed=81–200ms`，`extract` 107–227 ms/帧
  /// ⇒ 帧率掉到个位数）。预留一段够用满池的固定区 ⇒ 长度恒定 ⇒ 布局不漂移（§8 的"用增长余量摊薄"）。
  ///
  /// CONSTRAINT: 预留区**只能被块级分配用掉**；用满后 `alloc_block` 会照旧追加（真变长一次，罕见）。
  pub fn new_unbuilt_reserved(grid: &VolumeGrid, reserve_words: usize) -> Self {
    let (origin, dims, rejected) = compute_window(grid);
    let mut b = Self {
      buffers: BrickMapBuffers {
        b_struct: vec![0; TREE_BASE + reserve_words],
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
      chunks: HashMap::new(),
      origin,
      dims,
      rejected_chunks: rejected as u32,
      free: FreeRuns::default(),
      dirty_struct: Vec::new(),
      dirty_palette: None,
      palette_synced_at: None,
      reserve: reserve_words,
    };
    // 预留区登记为**空闲**：块级分配走 first-fit，会优先把它切走
    if reserve_words > 0 {
      b.free.free(TREE_BASE, reserve_words);
    }
    b.write_palette(grid);
    b
  }

  /// 全量构建（初始化 / 兜底）：确定性 + Rayon 并行序列化。
  /// chunk 按 ChunkCoord 升序安装；序列化并行，安装顺序不变（块地址按安装序递增）。
  pub fn build_full(grid: &VolumeGrid) -> Self {
    Self::build_full_reserved(grid, 0)
  }

  /// 同 [`Self::build_full`]，但带树区预留（见 [`Self::new_unbuilt_reserved`]）。
  pub fn build_full_reserved(grid: &VolumeGrid, reserve_words: usize) -> Self {
    let mut b = Self::new_unbuilt_reserved(grid, reserve_words);
    let mut coords: Vec<ChunkCoord> = grid
      .chunk_coords()
      .filter(|&c| chunk_index_pos(b.origin, b.dims, c.0).is_some() && chunk_has_content(grid, c))
      .collect();
    coords.sort_by_key(|&c| coord_key(c));

    let blobs: Vec<(Vec<u32>, NodeLayout)> = coords
      .par_iter()
      .map(|&c| grid.chunk(c).expect("chunk_has_content 已过滤").serialize_with_layout())
      .collect();
    for (c, (blob, layout)) in coords.iter().zip(blobs) {
      b.install_blob(*c, blob, layout);
    }
    b.refresh_globals();

    let _ = b.take_dirty_ranges();
    b
  }

  /// 逐 chunk 增量更新。
  ///
  /// `dirty` = 该 chunk 自上次上传以来的**节点级**改动（主 world 在 `Last` 阶段与脏 chunk 一起取走）。
  /// `reset`（身份空间换过）或本来没有块 ⇒ 释放旧块后全量重装；否则只重写 `dirty.nodes` 里的节点。
  /// `allow_compact` = 现在是不是"安静时刻"（见 [`Self::compact`]）；由调用方（`extract`）判定。
  pub fn update_chunk(
    &mut self,
    grid: &VolumeGrid,
    coord: ChunkCoord,
    dirty: &TreeDirty,
    allow_compact: bool,
  ) -> ChunkUpdate {
    if chunk_index_pos(self.origin, self.dims, coord.0).is_none() {
      return ChunkUpdate::OutsideWindow;
    }
    let out = if !chunk_has_content(grid, coord) {
      if self.chunks.contains_key(&coord) {
        self.release_chunk(coord);
        ChunkUpdate::Released
      } else {
        ChunkUpdate::Unchanged
      }
    } else {
      let tree = grid.chunk(coord).expect("chunk_has_content");
      if self.chunks.contains_key(&coord) && !dirty.reset {
        self.apply_node_dirty(coord, tree, dirty);
      } else {
        // 新 chunk / 身份空间作废：旧块（若有）归还，整棵重装
        self.release_chunk(coord);
        self.install_chunk(coord, tree);
      }
      ChunkUpdate::Rebuilt
    };
    // 树区过半是空闲段 ⇒ 压实：否则碎片会累积到"没有足够大的连续段"从而只能追加，高水位长期膨胀。
    // 只在安静时刻做（见 `allow_compact` 的说明）。（带预留的 volume 在 `compact` 里直接跳过。）
    if allow_compact && self.free.words() * 2 > self.buffers.b_struct.len() - TREE_BASE {
      self.compact();
    }
    self.refresh_globals();

    // palette 脏槽必须与触发它的那次体素编辑同一帧上传。
    self.write_palette(grid);
    out
  }

  /// 调色板增量铺：把自上次同步以来变化的槽写进 CPU 镜像并累积脏区间。
  /// 多消费者安全：用调色板写版本判断是否同步；首次同步必然全量，非首次拿不到脏区间也退回全量。
  pub fn write_palette(&mut self, grid: &VolumeGrid) {
    let version = grid.palette().version();
    if self.palette_synced_at == Some(version) {
      return;
    }
    let range = match self.palette_synced_at {
      None => {
        let _ = grid.palette().take_dirty();
        None
      }

      Some(_) => grid.palette().take_dirty(),
    };
    let (lo, hi) = range.unwrap_or((0, PALETTE_INDEX_MAX));
    for i in lo..=hi {
      let [a, b] = pack_palette_entry(grid.palette().get(PaletteId(i)));
      self.buffers.b_palette[i as usize * 2] = a;
      self.buffers.b_palette[i as usize * 2 + 1] = b;
    }
    self.palette_synced_at = Some(version);

    self.dirty_palette = Some(match self.dirty_palette {
      None => (lo, hi),
      Some((d0, d1)) => (d0.min(lo), d1.max(hi)),
    });
  }

  pub fn buffers(&self) -> &BrickMapBuffers {
    &self.buffers
  }

  /// 取走累积的增量脏字节区间列表（并重置）。
  pub fn take_dirty_ranges(&mut self) -> DirtyRanges {
    DirtyRanges {
      struct_ranges: std::mem::take(&mut self.dirty_struct),
      palette_range: self.dirty_palette.take(),
    }
  }

  /// 还有没有**未上传**的改动（[`Self::take_dirty_ranges`] 会清空它）。
  ///
  /// 给"本帧该不该出快照"用：常驻调度（`upload::plan_residency`）的 evict / install 也走标脏，
  /// 但它在 `extract`（唯一的出快照点）之后 ⇒ 那一份改动要到**下一帧**才被取走，`extract` 必须
  /// 因此知道"虽然没有脏 chunk、窗口也没动，但 builder 还有待上传的东西"。
  pub fn has_dirty(&self) -> bool {
    !self.dirty_struct.is_empty() || self.dirty_palette.is_some()
  }

  /// chunk 当前树基址（b_struct 内绝对字址；None = 未渲染）
  pub fn chunk_base(&self, coord: ChunkCoord) -> Option<usize> {
    self.chunks.get(&coord).map(|s| s.base)
  }

  // ---- 常驻管理（M3）：唤醒 / 换出 / 记账 ----------------------------------------------------
  //
  // CPU 树是权威（`VolumeGrid`），GPU 块是派生缓存 ⇒ "换出"只是归还 GPU 块，"唤醒"是从 CPU 树
  // 重新整块序列化（约 0.9 ms/chunk，见 `docs/editable-gigavoxel.md` §3.5）。两者的决策在
  // [`super::residency::Residency`]，这里只提供落实动作与记账。

  /// 该 chunk 现在有没有 GPU 块（= 是否常驻）。空 chunk 永远不常驻。
  pub fn is_resident(&self, coord: ChunkCoord) -> bool {
    self.chunks.contains_key(&coord)
  }

  /// 常驻块占用的**字数**（含块内余量）× 4 = 字节。常驻预算按它算。
  pub fn resident_words(&self) -> usize {
    self.chunks.values().map(|s| s.cap).sum()
  }

  /// 单个 chunk 的常驻字节（None = 未常驻）。
  pub fn resident_bytes_of(&self, coord: ChunkCoord) -> Option<usize> {
    self.chunks.get(&coord).map(|s| s.cap * 4)
  }

  /// 当前常驻的 chunk 列表（常驻调度的反向同步用：CPU 侧没了 ⇒ 归还这些块）。
  pub fn resident_chunks(&self) -> Vec<ChunkCoord> {
    self.chunks.keys().copied().collect()
  }

  /// **唤醒**：把 CPU 树整块装进 GPU。返回是否真的发生了安装。
  /// 已常驻 / CPU 侧为空 / 不在窗口内 ⇒ 什么都不做。
  pub fn ensure_resident(&mut self, grid: &VolumeGrid, coord: ChunkCoord) -> bool {
    if !chunk_has_content(grid, coord) {
      return false;
    }
    let tree = grid.chunk(coord).expect("chunk_has_content 已判非空");
    self.ensure_resident_tree(coord, tree)
  }

  /// **唤醒**（给定树）：调用方决定装全树还是 proxy 树（[`ChunkTree::proxy`]）；其余语义同
  /// [`Self::ensure_resident`]。
  pub fn ensure_resident_tree(&mut self, coord: ChunkCoord, tree: &ChunkTree) -> bool {
    if self.chunks.contains_key(&coord) || tree.is_empty() {
      return false;
    }
    if chunk_index_pos(self.origin, self.dims, coord.0).is_none() {
      return false;
    }
    self.install_chunk(coord, tree);
    self.refresh_globals();
    true
  }

  /// **换出**：归还 GPU 块（CPU 树不动 ⇒ 之后可再唤醒）。返回是否真的释放了。
  pub fn evict(&mut self, coord: ChunkCoord) -> bool {
    if !self.chunks.contains_key(&coord) {
      return false;
    }
    self.release_chunk(coord);
    self.refresh_globals();
    true
  }

  pub fn origin(&self) -> IVec3 {
    self.origin
  }

  pub fn dims(&self) -> IVec3 {
    self.dims
  }

  /// 窗口条目字址（调用方保证 coord 在窗口内）
  #[inline]
  fn window_word(&self, coord: ChunkCoord) -> usize {
    chunk_index_pos(self.origin, self.dims, coord.0).expect("窗口内 chunk")
  }

  /// 本 volume 当前的窗口（供"窗口是否移动了"的判断，见 [`VolumesBuilder::sync_windows`]）
  pub fn window(&self) -> (IVec3, IVec3) {
    (self.origin, self.dims)
  }

  /// **M6 · 窗口平移**：把窗口钉到 `(origin, dims)`（流式世界每帧钉在相机中心）。相位一变，
  /// 所有 chunk 的槽位跟着变，但**窗口条目存的是块相对地址**（`base + 1`，与相位无关）⇒ 只需把
  /// 条目**搬家**：不重传任何树块、不动 CPU 侧内容、不重新序列化。
  ///
  /// 代价 = 索引区（64³ chunk = 1 MB）整体标脏重写 —— 这就是"走得再远也不会整块重定"的实现。
  fn set_window(&mut self, origin: IVec3, dims: IVec3) {
    if self.origin == origin && self.dims == dims {
      return;
    }
    let old = (self.origin, self.dims);
    // **两阶段**：先把所有条目读出来（掉出窗口的丢弃），再把旧索引区整体清零，最后按新槽位写回。
    // CONSTRAINT: **不能边读边搬** —— 平移是**循环位移**，某个 chunk 的新槽位可能正是另一个 chunk 的
    // 旧槽位；边搬会互相覆盖（表现为画面里出现"别处 chunk 的几何"这种错位巨块）。
    let mut keep: Vec<(usize, u32)> = Vec::with_capacity(self.chunks.len());
    let mut dropped: Vec<ChunkCoord> = Vec::new();
    for c in self.chunks.keys().copied().collect::<Vec<_>>() {
      let Some(a) = chunk_index_pos(old.0, old.1, c.0) else {
        dropped.push(c);
        continue;
      };
      let entry = self.buffers.b_struct[a];
      match chunk_index_pos(origin, dims, c.0) {
        Some(b) => keep.push((b, entry)),
        None => dropped.push(c),
      }
    }
    // 索引区是**定长**的 `TREE_BASE` 字（64³ 槽，与窗口大小无关；窗口只是它被使用的子盒）
    self.buffers.b_struct[..TREE_BASE].fill(0);
    for (b, entry) in keep {
      self.buffers.b_struct[b] = entry;
    }
    self.origin = origin;
    self.dims = dims;
    // 掉出窗口的**必须真的释放**（块归还全局空闲段 + 从 `self.chunks` 移除），不能只丢条目 ——
    // 留下的"没有槽位的块"会让 [`Self::compact`] 逐块重指条目时 `window_word` panic（高速移动后崩）。
    // CONSTRAINT: 必须在 `self.origin/dims` 换过之后调 —— `release_chunk` 靠新窗口判"没有条目可清"。
    for c in &dropped {
      self.release_chunk(*c);
    }
    {
      let g = &mut self.buffers.globals;
      g.index_origin_x = origin.x;
      g.index_origin_y = origin.y;
      g.index_origin_z = origin.z;
      g.index_dims_x = dims.x as u32;
      g.index_dims_y = dims.y as u32;
      g.index_dims_z = dims.z as u32;
    }
    // 整个索引区都要重传（旧的要让 GPU 忘掉、新的要写上）
    self.mark_struct_words(0, TREE_BASE);
    bevy::log::debug!("WINDOW 平移 {} → {origin} dims {dims}（掉了 {} 个出门的）", old.0, dropped.len());
  }

  /// 从全局空闲段取一个 ≥`cap` 字的块（找不到就追加到高水位）。
  fn alloc_block(&mut self, cap: usize) -> usize {
    match self.free.alloc(cap) {
      Some(start) => start,
      None => {
        let start = self.buffers.b_struct.len();
        self.buffers.b_struct.resize(start + cap, 0);
        start
      }
    }
  }

  /// 全量安装一个 chunk 的树（新块 + 窗口条目 + 节点槽表）
  fn install_chunk(&mut self, coord: ChunkCoord, tree: &ChunkTree) {
    let (blob, layout) = tree.serialize_with_layout();
    self.install_blob(coord, blob, layout);
  }

  fn install_blob(&mut self, coord: ChunkCoord, blob: Vec<u32>, layout: NodeLayout) {
    let need = blob.len();
    // 块留 25% 余量（且至少容得下一个最大节点）：后续编辑的字数变化优先在块内解决，
    // 免得动不动搬整块（搬块 = 整棵 chunk 重传）。
    let cap = need + need / 4 + ROOT_WIRE_WORDS + 16;
    let base = self.alloc_block(cap);
    self.buffers.b_struct[base..base + need].copy_from_slice(&blob);
    let mut free = FreeRuns::default();
    free.free(need, cap - need);
    // 节点槽表同样留 25% 余量：树每编辑一次就可能多几个节点，槽表若刚好卡在长度上，
    // 头一次增长要 realloc + 填满整表（几十万槽 = 毫秒级）。
    let mut nodes: Vec<NodeSlot> = Vec::with_capacity(layout.len() + layout.len() / 4 + 64);
    nodes.extend(layout.iter().enumerate().map(|(id, &(off, level))| {
      if off == NODE_OFFSET_NONE {
        NodeSlot::UNASSIGNED
      } else {
        NodeSlot {
          off,
          words: wire_words_of(id as u32, level, read_mask(&blob, off as usize)) as u16,
        }
      }
    }));
    let ip = self.window_word(coord);
    self.buffers.b_struct[ip] = base as u32 + 1;
    self.chunks.insert(coord, ChunkSlot { base, cap, free, nodes });
    self.mark_struct_words(ip, 1);
    self.mark_struct_words(base, need);
  }

  /// chunk 变空 / 身份空间作废：块归还全局空闲段 + 窗口条目清零（无块时无操作）
  fn release_chunk(&mut self, coord: ChunkCoord) {
    let Some(slot) = self.chunks.remove(&coord) else { return };
    self.free.free(slot.base, slot.cap);
    // 掉出窗口的 chunk 没有条目（[`Self::set_window`] 已把整个索引区清零）⇒ 不写条目
    let Some(ip) = chunk_index_pos(self.origin, self.dims, coord.0) else { return };
    self.buffers.b_struct[ip] = 0;
    self.mark_struct_words(ip, 1);
  }

  /// 增量重写一个 chunk 里动过的节点（[`TreeDirty`] 保证按层降序 = 自底向上：
  /// 子节点先定址，父节点的指针表才写得对）。
  fn apply_node_dirty(&mut self, coord: ChunkCoord, tree: &ChunkTree, dirty: &TreeDirty) {
    let mut slot = self.chunks.remove(&coord).expect("调用方保证有块");
    if slot.nodes.len() < tree.node_capacity() {
      slot.nodes.resize(tree.node_capacity(), NodeSlot::UNASSIGNED);
    }
    for &(level, id) in &dirty.nodes {
      let Some(view) = tree.node_view(id) else {
        // 协议外的情形（身份空间失效却没带 reset）⇒ 退回全量重装，不做静默跳过
        self.chunks.insert(coord, slot);
        self.release_chunk(coord);
        self.install_chunk(coord, tree);
        return;
      };
      self.rewrite_node(coord, &mut slot, tree, level, id, view);
    }
    self.chunks.insert(coord, slot);
  }

  /// 重编码一个节点并就地更新：**字节完全相同就不写、不标脏**（[`TreeDirty`] 会把路径节点全报上来，
  /// 靠这一步把"报多了"过滤掉）。字数变了优先原地（缩 → 尾部归还；增 → 紧跟其后正好空闲才扩），
  /// 否则在块内搬一个位置（父节点必然也在 dirty 表里，会随之改指针）。
  fn rewrite_node(
    &mut self,
    coord: ChunkCoord,
    slot: &mut ChunkSlot,
    tree: &ChunkTree,
    level: u8,
    id: u32,
    view: NodeView<'_>,
  ) {
    let new_words = wire_words_of(id, level, view.mask);
    let old = slot.nodes[id as usize];

    // ---- 新字节：子块指针 = 子节点**当前**块内偏移 ----
    let mut out: Vec<u32> = Vec::with_capacity(new_words);
    out.push(view.mask as u32);
    out.push((view.mask >> 32) as u32);
    // palette word：低 16 = tile 色、高 16 = 叶代表值（M2）—— 与全量路径同一个打包函数
    out.push(pack_palette_word(view.palette, view.rep));
    if view.mask != 0 {
      if level == 3 {
        out.extend_from_slice(&tree.node_inline_words(id).expect("level 3 分裂节点有 inline"));
      } else {
        for &c in view.children {
          out.push(slot.nodes[c as usize].off);
        }
      }
    }
    // 根块的预留区（[`ROOT_WIRE_WORDS`]）大于实际内容（3 + popcount）：定址按预留区算、
    // 写只写内容长度 —— 这样掩码增减不会动根的位置（见 `wire_words_of`）。
    let content = out.len();
    debug_assert!(content == new_words || id == 0, "节点 {id} 的编码字数与定址口径不符");

    // ---- 变 uniform（掩码清零）⇒ 旧子块整棵释放 ----
    //
    // CONSTRAINT: 只在这个状态下释放。掩码只增不减（`child_or_create` 只置位，缩位只发生在整节点
    // 变 uniform 时），而"变 uniform 的节点其子块必然也已是 uniform"、uniform 节点恒 3 字且
    // 原地留驻 ⇒ 老 blob 里的子块偏移此刻仍然有效（换成"按偏移差集释放"就会把已搬走的活子块
    // 的旧地址当成死块释放 —— 那个地址可能已分给别人）。
    if old.off != NODE_NONE && level < 3 && view.mask == 0 {
      let old_at = slot.base + old.off as usize;
      let old_mask = read_mask(&self.buffers.b_struct, old_at);
      for slot_i in 0..old_mask.count_ones() as usize {
        let c = self.buffers.b_struct[old_at + NODE_FIXED_WORDS + slot_i];
        Self::free_subtree(&self.buffers.b_struct, slot, c, level + 1);
      }
    }

    // ---- 定址 ----
    // 缩小（含字数不变）：原地留驻，**尾部必须归还**空闲段 —— 紧挨着的空闲段合到一起，下一次
    // 增长才能原地扩回来（同一个 4³ brick 反复"合并 → 再分裂"是高频场景：level 3 在 3 ↔ 35 字之间
    // 摆动，不还尾部就会攒出一堆填不上的洞，把高水位一路顶上去）。
    let off = if old.off == NODE_NONE {
      self.alloc_node(coord, slot, new_words)
    } else if new_words <= old.words as usize {
      slot.free.free(old.off as usize + new_words, old.words as usize - new_words);
      old.off
    } else if slot
      .free
      .take_range(old.off as usize + old.words as usize, new_words - old.words as usize)
    {
      old.off // 原地扩容：紧跟其后正好是空闲段
    } else {
      slot.free.free(old.off as usize, old.words as usize);
      self.alloc_node(coord, slot, new_words)
    };

    // ---- 写 ----
    let at = slot.base + off as usize;
    if off != old.off || self.buffers.b_struct[at..at + content] != out[..] {
      self.buffers.b_struct[at..at + content].copy_from_slice(&out);
      self.mark_struct_words(at, content);
    }
    slot.nodes[id as usize] = NodeSlot { off, words: new_words as u16 };
  }

  /// 释放一个节点块**及其全部后代**（子块关系读**旧 blob**，与缓存无关；根不在此路径上，
  /// 故字数口径不必特判 id = 0）。字数 = 内容字数（与 [`Self::rewrite_node`] 的归还口径一致）。
  fn free_subtree(buf: &[u32], slot: &mut ChunkSlot, off: u32, level: u8) {
    let at = slot.base + off as usize;
    let mask = read_mask(buf, at);
    if mask != 0 && level < 3 {
      for i in 0..mask.count_ones() as usize {
        Self::free_subtree(buf, slot, buf[at + NODE_FIXED_WORDS + i], level + 1);
      }
    }
    slot.free.free(off as usize, wire_words_of(1, level, mask));
  }

  /// 块内分配一个字数为 `words` 的节点；块内没有足够连续空闲时先把块换大。
  fn alloc_node(&mut self, coord: ChunkCoord, slot: &mut ChunkSlot, words: usize) -> u32 {
    if let Some(off) = slot.free.alloc(words) {
      return off as u32;
    }
    self.grow_block(coord, slot);
    slot.free.alloc(words).expect("块扩容量恒 ≥ 单个节点最大字数") as u32
  }

  /// 块内空间不足：换一个更大的块（整块搬过去；块内偏移不变 ⇒ 节点内容一字不动），旧块归还全局空闲段。
  /// 整块字节都换了位置 ⇒ **必须整块重传** —— 只在块的几何增长时发生。
  fn grow_block(&mut self, coord: ChunkCoord, slot: &mut ChunkSlot) {
    let old_cap = slot.cap;
    let new_cap = old_cap + old_cap / 2 + ROOT_WIRE_WORDS + 1;
    let new_base = self.alloc_block(new_cap);
    self.buffers.b_struct.copy_within(slot.base..slot.base + old_cap, new_base);
    self.free.free(slot.base, old_cap);
    let used = slot.used();
    slot.base = new_base;
    slot.cap = new_cap;
    slot.free.free(old_cap, new_cap - old_cap);
    let ip = self.window_word(coord);
    self.buffers.b_struct[ip] = new_base as u32 + 1;
    self.mark_struct_words(ip, 1);
    self.mark_struct_words(new_base, used);
  }

  /// 压实树区：把存活块紧排到 `TREE_BASE` 之后、丢掉全部空闲段、窗口条目重指。
  /// 块内节点存的是**相对块首**的偏移 ⇒ 搬块不需要重写块内容，只改窗口条目与块首。
  /// 只在空闲过半、且处于安静时刻时调用（见 [`Self::update_chunk`]）；**只标"搬动过的块"的目标段**
  /// （没动的块在 GPU 上本来就是对的），相邻段由 `mark_struct_words` 自动并成 1~2 段。
  fn compact(&mut self) {
    // CONSTRAINT（M8）：**带预留的 volume（远场级）直接跳过** —— 它的 `b_struct` 长度必须恒定，
    // 否则排在其后的 volume 的 `tree_base` 漂移 ⇒ `bases_shifted` 全量重传（实测 577 MB / 81–200 ms/帧）。
    // 而压实**只会**把长度压小 ⇒ 对预留区毫无收益（预留区一开场就是空闲的 ⇒ 判据恒真 ⇒ 每帧白搬一遍块）。
    if self.reserve > 0 {
      return;
    }
    let mut items: Vec<(ChunkCoord, usize, usize)> =
      self.chunks.iter().map(|(&c, s)| (c, s.base, s.cap)).collect();
    // CONSTRAINT: 必须按**旧基址**升序搬（不是按坐标）—— 压实只会前移，升序搬运才能保证"目标段"
    // 永不覆盖尚未搬运的段（旧布局是分配序，与坐标序不一致）。
    items.sort_by_key(|&(_, old_base, _)| old_base);
    let mut cursor = TREE_BASE;
    let mut moved = 0usize;
    for (coord, old_base, cap) in items {
      if old_base != cursor {
        self.buffers.b_struct.copy_within(old_base..old_base + cap, cursor);
        let ip = self.window_word(coord);
        self.chunks.get_mut(&coord).expect("窗口内 chunk").base = cursor;
        self.buffers.b_struct[ip] = cursor as u32 + 1;
        self.mark_struct_words(ip, 1);
        self.mark_struct_words(cursor, cap);
        moved += cap;
      }
      cursor += cap;
    }
    self.buffers.b_struct.truncate(cursor);
    self.free = FreeRuns::default();
    bevy::log::debug!(
      "COMPACT 树区 长{}字 → {}字（搬 {moved} 字 / {} 块）",
      self.buffers.globals.node_words,
      cursor - TREE_BASE,
      self.chunks.len()
    );
  }

  fn refresh_globals(&mut self) {
    let g = &mut self.buffers.globals;
    g.tile_count = self.chunks.len() as u32;
    g.node_words = (self.buffers.b_struct.len() - TREE_BASE) as u32;
    g.node_free_words = self.free.words() as u32;
    g.brick_slabs = 0;
    g.brick_free = 0;
    g.rejected_tiles = self.rejected_chunks;
  }
}

/// u32 字切片 → 本机字节序 u8 `Vec`（wire 按小端直存）。
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

/// 多 volume 统一构建器：持有 `Vec<BrickMapBuilder>`，输出统一 buffer + GridDesc 数组。
/// 各 volume 独立维护 dirty 跟踪；增量下任一前置 volume 增长使 tree_base 漂移则自动降级全量。
pub struct VolumesBuilder {
  builders: Vec<BrickMapBuilder>,
  /// 每 volume 的变换（缓存自 Volumes，用于 GridDesc 生成）
  transforms: Vec<VolumeTransform>,
  /// 每 volume 是不是**远场级**（M8）：进 `GridDesc::grid_flags` 的 [`GRID_FLAG_FAR`]，
  /// 唯一消费者是 shader 的分壳裁剪（`trace.wesl::grid_is_far`）。
  far: Vec<bool>,
  /// 上一次 snapshot 的 tree_bases（字偏移）；空 = 首帧 → 强制全量
  prev_tree_bases: Vec<u32>,
  /// 上一次 snapshot 的 palette_bases（字偏移）
  prev_palette_bases: Vec<u32>,
  /// 强制全量标志（新增 volume /手动请求）
  force_full: bool,
}

impl VolumesBuilder {
  /// 该 volume 的树区预留（M8）：远场级固定预留 [`FAR_TREE_RESERVE_WORDS`]（长度恒定 ⇒ 主世界的
  /// `tree_base` 不漂移 ⇒ 不降级全量重传），主世界与普通物体为 0。
  fn reserve_of(grid: &gate_voxel::VolumeGrid) -> usize {
    if grid.is_far_level() {
      super::consts::FAR_TREE_RESERVE_WORDS
    } else {
      0
    }
  }

  /// 全量构建所有 volume（初始化 / 兜底）
  pub fn build_full(volumes: &Volumes) -> Self {
    let mut builders = Vec::with_capacity(volumes.len());
    let mut transforms = Vec::with_capacity(volumes.len());
    let mut far = Vec::with_capacity(volumes.len());
    for grid in volumes.all() {
      builders.push(BrickMapBuilder::build_full_reserved(grid, Self::reserve_of(grid)));
      transforms.push(grid.transform());
      far.push(grid.is_far_level());
    }
    Self {
      builders,
      transforms,
      far,
      prev_tree_bases: Vec::new(),
      prev_palette_bases: Vec::new(),
      force_full: true,
    }
  }

  /// 空构造（渐进式：先 new_unbuilt，再 update_chunk 累积）
  pub fn new_unbuilt(volumes: &Volumes) -> Self {
    let mut builders = Vec::with_capacity(volumes.len());
    let mut transforms = Vec::with_capacity(volumes.len());
    let mut far = Vec::with_capacity(volumes.len());
    for grid in volumes.all() {
      builders.push(BrickMapBuilder::new_unbuilt_reserved(grid, Self::reserve_of(grid)));
      transforms.push(grid.transform());
      far.push(grid.is_far_level());
    }
    Self {
      builders,
      transforms,
      far,
      prev_tree_bases: Vec::new(),
      prev_palette_bases: Vec::new(),
      force_full: true,
    }
  }

  /// 逐 volume 同步调色板（版本门控；无变化时全部空操作）。
  /// 用途：菜单改材质参数时几何不脏但调色板槽内容变了，须在此补一次同步。
  pub fn sync_palettes(&mut self, volumes: &Volumes) {
    for (i, grid) in volumes.all().iter().enumerate() {
      if let Some(b) = self.builders.get_mut(i) {
        b.write_palette(grid);
      }
    }
  }

  pub fn sync(&mut self, volumes: &Volumes) {
    while self.builders.len() < volumes.all().len() {
      let idx = self.builders.len();
      let grid = &volumes.all()[idx];
      self.builders.push(BrickMapBuilder::build_full_reserved(grid, Self::reserve_of(grid)));
      self.transforms.push(grid.transform());
      self.far.push(grid.is_far_level());
      self.force_full = true;
    }
    for (i, grid) in volumes.all().iter().enumerate() {
      self.transforms[i] = grid.transform();
      self.far[i] = grid.is_far_level();
    }
  }

  /// 逐 chunk 增量更新指定 volume（`dirty` / `allow_compact` 见 [`BrickMapBuilder::update_chunk`]）
  pub fn update_chunk(
    &mut self,
    volumes: &Volumes,
    volume_idx: usize,
    coord: ChunkCoord,
    dirty: &TreeDirty,
    allow_compact: bool,
  ) -> ChunkUpdate {
    self.builders[volume_idx].update_chunk(&volumes.all()[volume_idx], coord, dirty, allow_compact)
  }

  /// 某个 volume 当前的窗口（`None` = 该 volume 还没建起来）
  pub fn window_of(&self, volume_idx: usize) -> Option<(IVec3, IVec3)> {
    self.builders.get(volume_idx).map(BrickMapBuilder::window)
  }

  /// 还有没有未上传的改动（见 [`BrickMapBuilder::has_dirty`]）
  pub fn has_dirty(&self) -> bool {
    self.builders.iter().any(BrickMapBuilder::has_dirty)
  }

  /// **M6**：把每个 volume 的窗口对齐到 `grid.stream_window()`（流式世界每帧钉在相机中心）。
  /// 相位变了就**平移索引区**（[`BrickMapBuilder::set_window`]）—— 只搬条目，不重传树块。
  pub fn sync_windows(&mut self, volumes: &Volumes) {
    for (i, g) in volumes.all().iter().enumerate() {
      if let Some((o, d)) = g.stream_window() {
        self.builders[i].set_window(o, d);
      }
    }
  }

  /// 标记全量重建（下帧 snapshot 走 full 路径）
  pub fn force_full(&mut self) {
    self.force_full = true;
  }

  /// 取走统一快照：拼接所有 volume 的 b_struct/b_palette + 生成 GridDesc 数组。
  /// 布局：物体 (1..N) 先放、主世界 (0) 后放（主世界增长不漂移前置 tree_base）；GridDesc 按 volume 索引序。
  pub fn snapshot(&mut self) -> VolumesSnapshot {
    let n = self.builders.len();

    let layout_order: Vec<usize> = (1..n).chain(std::iter::once(0)).collect();

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

    let mut grid_descs = Vec::with_capacity(n);
    for i in 0..n {
      let buffers = self.builders[i].buffers();
      let g = &buffers.globals;
      let tr = self.transforms[i];
      let origin = IVec3::new(g.index_origin_x, g.index_origin_y, g.index_origin_z);
      let dims = IVec3::new(g.index_dims_x as i32, g.index_dims_y as i32, g.index_dims_z as i32);
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
      // **世界 AABB 一律按窗口算**（主世界 / 远场级 / 物体同一口径）：`from_transform` 给的是
      // 局部 `[0,256]³` 的 AABB（"单 chunk 物体"的形态），而主世界与远场级的局部范围是它们的窗口。
      // 这里覆盖成窗口 AABB 后，shader 的 AABB 预剔除对三级远场都成立（否则远场级整级会被裁掉）。
      let (mn, mx) = super::wire::window_world_aabb(tr, origin, dims);
      desc.aabb_min = Vec4::new(mn.x, mn.y, mn.z, 0.0);
      desc.aabb_max = Vec4::new(mx.x, mx.y, mx.z, 0.0);
      // M8：远场级标记（分壳裁剪的开关，见 `trace.wesl::grid_is_far`）
      desc.grid_flags = if self.far[i] { super::wire::GRID_FLAG_FAR } else { 0 };
      grid_descs.push(desc);
    }

    let bases_shifted = self.prev_tree_bases.len() != tree_bases.len()
      || self.prev_tree_bases.iter().zip(tree_bases.iter()).any(|(p, c)| p != c)
      || self.prev_palette_bases.iter().zip(palette_bases.iter()).any(|(p, c)| p != c);
    let need_full = self.force_full || bases_shifted;

    let dirty_chunks: usize = self.builders.iter().map(|b| b.dirty_struct.len()).sum();

    let mut b_struct = Vec::new();
    let mut b_palette = Vec::new();
    let mut struct_blobs = Vec::new();
    let mut palette_blobs = Vec::new();

    if need_full {
      for b in &mut self.builders {
        let _ = b.take_dirty_ranges();
      }
      for &i in &layout_order {
        let buffers = self.builders[i].buffers();
        b_struct.extend_from_slice(&buffers.b_struct);
        b_palette.extend_from_slice(&buffers.b_palette);
      }
    } else {
      for i in 0..n {
        let dr = self.builders[i].take_dirty_ranges();
        let tb = tree_bases[i] as usize;
        let pb = palette_bases[i] as usize;
        let buffers = self.builders[i].buffers();
        for (lo, hi) in dr.struct_ranges {
          let (lw, hw) = (lo / 4, hi / 4);
          struct_blobs.push((tb * 4 + lo, words_to_bytes(&buffers.b_struct[lw..hw])));
        }
        if let Some((lo, hi)) = dr.palette_range {
          let w0 = lo as usize * 2;
          let w1 = hi as usize * 2 + 2;
          palette_blobs.push((
            pb * 4 + lo as usize * PALETTE_BYTES_PER_ENTRY,
            words_to_bytes(&buffers.b_palette[w0..w1]),
          ));
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

  // ---- 常驻管理（M3，逐 volume 分发）-------------------------------------------------------

  /// 该 volume 的 chunk 当前是否常驻（有 GPU 块）。
  pub fn is_resident(&self, vol_idx: usize, coord: ChunkCoord) -> bool {
    self.builders.get(vol_idx).is_some_and(|b| b.is_resident(coord))
  }

  /// 该 volume 常驻块占用的**字数**（×4 = 字节；预算按它算）。
  pub fn resident_words(&self, vol_idx: usize) -> usize {
    self.builders.get(vol_idx).map_or(0, BrickMapBuilder::resident_words)
  }

  /// 该 volume 里单个 chunk 的常驻字节（None = 未常驻）。
  pub fn resident_bytes_of(&self, vol_idx: usize, coord: ChunkCoord) -> Option<usize> {
    self.builders.get(vol_idx).and_then(|b| b.resident_bytes_of(coord))
  }

  /// 该 volume 当前常驻的 chunk 列表。
  pub fn resident_chunks(&self, vol_idx: usize) -> Vec<ChunkCoord> {
    self.builders.get(vol_idx).map(BrickMapBuilder::resident_chunks).unwrap_or_default()
  }

  /// 唤醒：按**给定的树**（全树或 proxy）整块装进 GPU（见 [`BrickMapBuilder::ensure_resident_tree`]）。
  pub fn ensure_resident_tree(
    &mut self,
    vol_idx: usize,
    coord: ChunkCoord,
    tree: &ChunkTree,
  ) -> bool {
    self.builders.get_mut(vol_idx).is_some_and(|b| b.ensure_resident_tree(coord, tree))
  }

  /// 唤醒：从该 volume 的 CPU 树整块装进 GPU（**近场不截断**）。远场级走这一条 ——
  /// 它们本身已经是粗档（每格 `FAR_GRAIN` 级体素），没有"再截断一层"可做的（见 `upload::plan_residency`）。
  pub fn ensure_resident(&mut self, volumes: &Volumes, vol_idx: usize, coord: ChunkCoord) -> bool {
    let Some(b) = self.builders.get_mut(vol_idx) else { return false };
    let Some(g) = volumes.all().get(vol_idx) else { return false };
    b.ensure_resident(g, coord)
  }

  /// 换出：归还 GPU 块（CPU 树不动 ⇒ 之后可再唤醒）。
  pub fn evict(&mut self, vol_idx: usize, coord: ChunkCoord) -> bool {
    self.builders.get_mut(vol_idx).is_some_and(|b| b.evict(coord))
  }

  /// 逐 volume 的 wire 字节状态（按 volume 索引序，非拼接序）：供调试转储（`upload::dump_voxel_buffers`）
  /// 把"CPU 认为该上传什么"整份取出；内容与 `snapshot()` 上传的字节同源。
  pub fn volume_buffers(&self) -> Vec<&BrickMapBuffers> {
    self.builders.iter().map(|b| b.buffers()).collect()
  }

  pub fn len(&self) -> usize {
    self.builders.len()
  }

  pub fn is_empty(&self) -> bool {
    self.builders.is_empty()
  }
}

/// 窗口计算：原点 = 最小非空 chunk - 1（±1 chunk 余量），跨度 = max - min + 3，封顶 64³（CHUNK_INDEX_CAP）。
fn compute_window(grid: &VolumeGrid) -> (IVec3, IVec3, usize) {
  // 流式世界（`infinite_cubes`）：窗口由 `VolumeGrid::set_stream_window` **钉死** —— 它由
  // `infinite_cubes::stream_chunks` 每帧钉在"相机为中心"上（M6：走得再远也只是平移索引区，见
  // `BrickMapBuilder::set_window`）。常驻集是相机周围的环（半径 ≪ 半窗宽）⇒ 内容一律在窗口内，
  // **不按内容扩张**（扩张会让 origin 随内容漂移，平移就无从谈起）。
  if let Some((o, d)) = grid.stream_window() {
    return (o, d, 0);
  }
  let (mut min, mut max, mut any) = (IVec3::splat(i32::MAX), IVec3::splat(i32::MIN), false);
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
  use super::super::wire::CHUNK_SIZE;

  /// **M6**：窗口平移 —— 条目跟着 chunk **搬家**、树块**原地不动**（"走得再远也不整块重定"的实现）。
  ///
  /// 关键用例：两个**相邻** chunk + 窗口**反向**平移 ⇒ 甲的"新槽位"正是乙的"旧槽位"（循环位移）。
  /// 边读边搬会互相覆盖（画面上出现"别处 chunk 的几何"这种错位巨块）⇒ 必须两阶段。
  #[test]
  fn set_window_translates_index_entries() {
    let mut grid = VolumeGrid::new();
    let (c, c2) = (ChunkCoord(IVec3::new(4, 0, 0)), ChunkCoord(IVec3::new(5, 0, 0)));
    for x in 0..8 {
      for y in 0..8 {
        grid.set_voxel_ivec3(c.0 * CHUNK_SIZE + IVec3::new(x, y, 0), PaletteId(1));
        grid.set_voxel_ivec3(c2.0 * CHUNK_SIZE + IVec3::new(x, y, 0), PaletteId(2));
      }
    }
    let mut b = build_and_drain(&mut grid);
    let (o0, d0) = b.window();
    let pos0 = b.window_word(c);
    let (entry0, entry1) = (b.buffers().b_struct[pos0], b.buffers().b_struct[pos0 + 1]);
    assert!(entry0 != 0 && entry1 != 0, "两个 chunk 都该装进窗口");
    let base = entry0 as usize - 1;
    let block = b.buffers().b_struct[base..base + 8].to_vec();

    // **反向**平移 1 chunk：甲的新槽位 = 乙的旧槽位（循环位移，naive 边搬必坏）
    let o1 = o0 - IVec3::new(1, 0, 0);
    b.set_window(o1, d0);
    assert_eq!(b.window(), (o1, d0));
    let pos1 = b.window_word(c);
    assert_eq!(pos1, pos0 + 1, "前提：正好撞上乙的旧槽位");
    assert_eq!(b.buffers().b_struct[pos1], entry0, "甲的条目搬到（原乙的）新槽位");
    assert_eq!(b.buffers().b_struct[pos1 + 1], entry1, "乙的条目也搬对了（没被甲覆盖）");
    assert_eq!(b.buffers().b_struct[pos0], 0, "腾出来的槽位清零");
    assert_eq!(&b.buffers().b_struct[base..base + 8], &block[..], "树块原地不动");

    // 再平移 3 chunk ⇒ 甲掉出窗口：只清条目（GPU 视作空气），**CPU 侧内容保留**
    let o2 = o1 + IVec3::new(3, 0, 0);
    assert!(chunk_index_pos(o2, d0, c.0).is_none(), "前提：甲已在窗口外");
    b.set_window(o2, d0);
    assert_eq!(b.buffers().b_struct[pos1], 0, "掉出窗口的条目清空");
    assert!(grid.chunk(c).is_some(), "CPU 侧 chunk 不受窗口平移影响");
    assert_ne!(b.window_word(c2), 0, "仍在窗口内的 chunk 条目照旧跟着走");

    // 掉出窗口的必须**真的释放**（块归还空闲段 + 从 `chunks` 移除），不能只丢条目：留下的
    // "没有槽位的块"会让 [`BrickMapBuilder::compact`] 逐块重指条目时 panic（实测高速移动后崩）。
    let words_before = b.buffers().b_struct.len();
    b.compact();
    assert!(
      b.buffers().b_struct.len() < words_before,
      "掉出窗口的块该被回收：{words_before} → {}",
      b.buffers().b_struct.len()
    );
  }

  /// **wire → GPU 一致性**：`install_blob` 之后，`b_struct` 里那个 4³ 值块的 32 个 inline 半字
  /// 必须逐格等于 CPU 写入的槽号。
  ///
  /// 把"逐体素取色"的数据链验到 GPU 缓冲（CPU 树 → wire → `b_struct`）：任何一环把块内合并成
  /// uniform、或写漏 inline，着色端就只能拿到一个节点色 ⇒ 画面变成"一个 4³ 块一个色"。
  /// 读法与 `trace.wesl` 的层次 DDA **同式**：节点 3 字 = mask_lo + mask_hi + palette，
  /// 子块指针存**相对根的字偏移**（根恒占 3 + 64 字）。
  #[test]
  fn gpu_blob_carries_per_voxel_palette_in_a_brick() {
    const N_FIXED: usize = 3;
    let mut grid = VolumeGrid::new();
    for i in 0..64i32 {
      let p = IVec3::new(i % 4, (i / 4) % 4, i / 16);
      grid.set_voxel_ivec3(p, PaletteId((i + 1) as u16));
    }
    let b = build_and_drain(&mut grid);
    let words = &b.buffers().b_struct;
    let ip = b.window_word(ChunkCoord(IVec3::ZERO));
    let base = words[ip] as usize - 1;
    // 逐层走 cell (0,0,0)：`pop` = 该格之前的置位数 = 0 ⇒ 取指针表第 0 项
    let mut addr = base;
    for lv in 0..3 {
      assert_ne!(words[addr] | words[addr + 1], 0, "第 {lv} 层应已分裂（cell 0 有内容）");
      addr = base + words[addr + N_FIXED] as usize;
    }
    let mask = (words[addr] as u64) | ((words[addr + 1] as u64) << 32);
    assert_eq!(mask.count_ones(), 64, "4³ 值块应 64 格全置位（否则整块一色）");
    for i in 0..64usize {
      let w = words[addr + N_FIXED + (i >> 1)];
      assert_eq!((w >> ((i & 1) * 16)) & 0xFFFF, (i + 1) as u32, "GPU 第 {i} 格槽号");
    }
  }

  /// 空闲段：first-fit、相邻合并（前向 / 后向 / 双向）、跨洞不合并
  #[test]
  fn free_runs_first_fit_and_coalesce() {
    let mut f = FreeRuns::default();
    assert_eq!(f.alloc(4), None, "空表无段可分配");
    f.free(100, 8);
    f.free(200, 8);
    f.free(108, 4); // 与 100..108 前向相邻 ⇒ 合并成 100..112
    assert_eq!(f.words(), 20);
    assert_eq!(f.alloc(12), Some(100), "first-fit 命中第一段够大的");
    assert_eq!(f.words(), 8);
    assert_eq!(f.alloc(9), None, "剩 8 字凑不出 9");
    f.free(300, 4);
    f.free(304, 4); // 与 300..304 后向相邻 ⇒ 合并成 300..308
    assert_eq!(f.alloc(8), Some(200), "更靠前的段够 8 字");
    assert_eq!(f.alloc(4), Some(300));
    assert_eq!(f.alloc(4), Some(304), "切走 4 字后剩余段从 304 起");
    assert_eq!(f.words(), 0);
    // 中间有洞则各自独立，凑不出连续 8 字
    f.free(500, 4);
    f.free(510, 4);
    assert_eq!(f.alloc(8), None);
    // take_range：只在完全落在某段空闲内时切走（原地扩容用），并把两侧余量归还
    assert!(!f.take_range(506, 4), "跨在两段之间 ⇒ 拒绝");
    f.free(600, 10);
    assert!(!f.take_range(606, 8), "跨出段尾 ⇒ 拒绝");
    assert!(f.take_range(604, 4), "完全落在段内 ⇒ 切走");
    assert_eq!(f.words(), 4 + 4 + 4 + 2, "500..504 / 510..514 / 600..604 / 608..610");
    assert!(f.take_range(604, 0), "空请求恒成功");
  }

  /// 铺一块 16³ 实体（树里有 Split 层）+ 一个 8³ 方块到相邻 chunk
  fn two_chunk_grid() -> (VolumeGrid, ChunkCoord, ChunkCoord) {
    let mut grid = VolumeGrid::new();
    for x in 0..16 {
      for y in 0..16 {
        for z in 0..16 {
          grid.set_voxel_ivec3(IVec3::new(x, y, z), PaletteId(1));
        }
      }
    }
    let c1 = ChunkCoord(IVec3::new(1, 0, 0));
    for x in 0..8 {
      for y in 0..8 {
        for z in 0..8 {
          grid.set_voxel_ivec3(c1.0 * CHUNK_SIZE + IVec3::new(x, y, z), PaletteId(2));
        }
      }
    }
    (grid, ChunkCoord(IVec3::ZERO), c1)
  }

  /// 取走一个 chunk 的节点级改动（与主 world `poll_pending` 同一口径）
  fn take_dirty_of(grid: &mut VolumeGrid, c: ChunkCoord) -> TreeDirty {
    grid.chunk_mut(c).map(|t| t.take_dirty()).unwrap_or_default()
  }

  /// 建场景时的标记由首帧安装消费（生产路径：`poll_pending` 的首次 drain 带上 `reset`）。
  /// 测试里等价地先 drain 一遍，让后续断言只看得见新编辑。
  fn build_and_drain(grid: &mut VolumeGrid) -> BrickMapBuilder {
    let b = BrickMapBuilder::build_full(grid);
    let coords: Vec<ChunkCoord> = grid.chunk_coords().collect();
    for c in coords {
      let _ = take_dirty_of(grid, c);
    }
    b
  }

  /// wire 子树 ↔ `ChunkTree` 子树**逐节点**精确比对（掩码 / uniform 色 / inline / 子块指针），
  /// 返回比对的节点数。读的就是 shader 会读的那些字节 ⇒ 不用抽样、也不会漏掉坏节点。
  fn assert_wire_node(
    buf: &[u32],
    base: usize,
    off: usize,
    tree: &ChunkTree,
    id: u32,
    level: u8,
  ) -> usize {
    let view = tree.node_view(id).expect("grid 侧节点必 live");
    let at = base + off;
    assert_eq!(read_mask(buf, at), view.mask, "节点 {id}（层 {level}）掩码不符");
    assert_eq!(
      buf[at + 2],
      pack_palette_word(view.palette, view.rep),
      "节点 {id}（层 {level}）的 palette word（tile 色 + 叶代表值）不符"
    );
    if view.mask == 0 {
      return 1;
    }
    if level == 3 {
      let inline = tree.node_inline_words(id).expect("level 3 分裂节点有 inline");
      assert_eq!(
        &buf[at + NODE_FIXED_WORDS..at + NODE_FIXED_WORDS + LEAF_INLINE_WORDS],
        &inline[..],
        "节点 {id} 的 inline 字不符"
      );
      return 1;
    }
    let mut n = 1;
    for (slot, &child) in view.children.iter().enumerate() {
      let child_off = buf[at + NODE_FIXED_WORDS + slot] as usize;
      n += assert_wire_node(buf, base, child_off, tree, child, level + 1);
    }
    n
  }

  /// 每个有内容的 chunk：wire 字节解出来的树必须与 grid 的权威树**逐节点相同**
  ///（复用 / 原地增删 / 压实搬运 / 块扩容之后都成立）
  fn assert_wire_matches_grid(b: &BrickMapBuilder, grid: &VolumeGrid, what: &str) {
    let mut nodes = 0usize;
    for c in grid.chunk_coords().filter(|&c| chunk_has_content(grid, c)) {
      let tree = grid.chunk(c).expect("chunk_has_content");
      let base = b.chunk_base(c).unwrap_or_else(|| panic!("{what}：{c:?} 有内容但窗口无块"));
      assert_eq!(
        b.buffers().b_struct[b.window_word(c)] as usize,
        base + 1,
        "{what}：{c:?} 的窗口条目未指向块首"
      );
      if tree.is_uniform_root() {
        assert_eq!(read_mask(&b.buffers().b_struct, base), 0, "{what}：{c:?} 单色根掩码应为 0");
        assert_eq!(
          b.buffers().b_struct[base + 2],
          pack_palette_word(tree.root_palette(), PaletteId::AIR),
          "{what}：{c:?} 单色根的 palette word 不符（代表值字段恒 0）"
        );
        nodes += 1;
      } else {
        nodes += assert_wire_node(&b.buffers().b_struct, base, 0, tree, 0, 0);
      }
    }
    assert!(nodes > 0, "{what}：没有可比对的节点");
  }

  /// 反复编辑同一个 chunk：节点字数在"分裂 / 合回"间摆动 ⇒ 块内空闲段原地复用，高水位不动
  ///（没有块内 arena 时，每次编辑都要追加一整棵新树 ⇒ 高水位线性增长）。
  #[test]
  fn repeated_edits_reuse_freed_space_in_block() {
    let (mut grid, c0, _) = two_chunk_grid();
    let mut b = build_and_drain(&mut grid);
    let hw0 = b.buffers().b_struct.len();
    // 改 / 改回同一个格：块在 Uniform 与 Split 间切换（节点字数摆动）
    let p = IVec3::new(3, 3, 3);
    let mut hw_seq = Vec::new();
    for i in 0..20 {
      grid.set_voxel_ivec3(p, if i % 2 == 0 { PaletteId(2) } else { PaletteId(1) });
      let dirty = take_dirty_of(&mut grid, c0);
      assert_eq!(b.update_chunk(&grid, c0, &dirty, true), ChunkUpdate::Rebuilt);
      assert_wire_matches_grid(&b, &grid, &format!("第 {i} 次编辑"));
      hw_seq.push(b.buffers().b_struct.len());
    }
    let tail = &hw_seq[5..];
    assert!(
      tail.iter().all(|&h| h == tail[0]),
      "高水位应在头几次编辑后稳定在块内解决，实际序列 {hw_seq:?}"
    );
    assert!(
      hw_seq[19] <= hw0 + ROOT_WIRE_WORDS,
      "20 次编辑最多只该用掉初始余量：{hw0} → {}",
      hw_seq[19]
    );
  }

  /// **M8**：远场级的树区**长度恒定** —— 这是"不降级全量重传"的前提。
  ///
  /// `VolumesBuilder::snapshot` 的 `bases_shifted` 以各 volume 的 `b_struct` 长度为判据；远场级排在
  /// 主世界**之前** ⇒ 远场一变长，主世界的 `tree_base` 就漂移 ⇒ 全量快照（实测 `UPLOAD[full]
  /// bytes=577MB elapsed=81–200ms`，`extract` 107–227 ms/帧 ⇒ 帧率掉到个位数）。
  /// 预留区必须同时扛住**装块**与**压实**（压实原来会把尾巴截掉）。
  #[test]
  fn far_reserve_keeps_tree_region_length_stable() {
    let mut grid = VolumeGrid::new();
    let coords: Vec<ChunkCoord> =
      (0..12).map(|i| ChunkCoord(IVec3::new(i % 4, i / 4, 0))).collect();
    for c in &coords {
      for x in 0..8 {
        for y in 0..8 {
          for z in 0..8 {
            grid.set_voxel_ivec3(c.0 * CHUNK_SIZE + IVec3::new(x, y, z), PaletteId(1));
          }
        }
      }
    }
    let reserve = 1 << 20; // 4 M 字（够这 12 块的 1.5 万倍余量）
    let mut b = BrickMapBuilder::new_unbuilt_reserved(&grid, reserve);
    let len0 = b.buffers().b_struct.len();
    assert_eq!(len0, TREE_BASE + reserve, "预留区应当场占住");
    for c in &coords {
      let dirty = take_dirty_of(&mut grid, *c);
      b.update_chunk(&grid, *c, &dirty, true);
      assert_eq!(
        b.buffers().b_struct.len(),
        len0,
        "装块必须被预留区吃掉，不许改长度（否则主世界 tree_base 漂移 ⇒ 全量重传）"
      );
    }
    b.compact();
    assert_eq!(b.buffers().b_struct.len(), len0, "压实必须保住预留区长度");
    assert_eq!(b.window(), (b.origin, b.dims), "压实后窗口不变");
  }

  /// chunk 清空后空闲段占满树区 ⇒ 触发压实：高水位回落，存活 chunk 的树跟着前移且内容不变
  #[test]
  fn compaction_shrinks_tree_region() {
    let mut grid = VolumeGrid::new();
    let (c0, c1) = (ChunkCoord(IVec3::ZERO), ChunkCoord(IVec3::new(1, 0, 0)));
    // c0：16³ 棋盘格 —— 每个 4³ brick 都是 Mixed，树远大于 c1（实体块会被合并成 uniform）
    let checker = |x: i32, y: i32, z: i32| (x + y + z) % 2 == 0;
    for x in 0..16 {
      for y in 0..16 {
        for z in 0..16 {
          if checker(x, y, z) {
            grid.set_voxel_ivec3(IVec3::new(x, y, z), PaletteId(1));
          }
        }
      }
    }
    // c1：8³ 实心块
    for x in 0..8 {
      for y in 0..8 {
        for z in 0..8 {
          grid.set_voxel_ivec3(c1.0 * CHUNK_SIZE + IVec3::new(x, y, z), PaletteId(2));
        }
      }
    }
    let mut b = build_and_drain(&mut grid);
    let hw0 = b.buffers().b_struct.len();
    let c1_base_before = b.chunk_base(c1).expect("窗口内");
    assert!(
      grid.chunk(c0).expect("chunk").serialize().len()
        > grid.chunk(c1).expect("chunk").serialize().len(),
      "前提：待清空的 c0 树更大（否则空闲占不到一半，不该压实）"
    );
    // 擦空 c0（棋盘格逐格擦 ⇒ 每层向上合并，最终整 chunk 空）
    for x in 0..16 {
      for y in 0..16 {
        for z in 0..16 {
          if checker(x, y, z) {
            grid.set_voxel_ivec3(IVec3::new(x, y, z), PaletteId::AIR);
          }
        }
      }
    }
    let dirty = take_dirty_of(&mut grid, c0);
    assert_eq!(b.update_chunk(&grid, c0, &dirty, true), ChunkUpdate::Released);
    assert_eq!(b.chunk_base(c0), None, "清空后窗口条目应清零");
    assert!(
      b.buffers().b_struct.len() < hw0,
      "压实后树区应变短（{hw0} → {}）",
      b.buffers().b_struct.len()
    );
    assert_eq!(b.chunk_base(c1), Some(TREE_BASE), "存活 chunk 应被前移到树区起点");
    assert_ne!(b.chunk_base(c1), Some(c1_base_before));
    assert_wire_matches_grid(&b, &grid, "压实后");
    assert_eq!(b.buffers().globals.node_free_words, 0, "压实后不应残留空闲段");
  }

  /// 笔触进行中（`allow_compact = false`）**不压实**：同一场景下先不压（树区不缩），安静后再压（缩）。
  #[test]
  fn compaction_deferred_while_editing() {
    let mut grid = VolumeGrid::new();
    let (c0, c1) = (ChunkCoord(IVec3::ZERO), ChunkCoord(IVec3::new(1, 0, 0)));
    let checker = |x: i32, y: i32, z: i32| (x + y + z) % 2 == 0;
    for x in 0..16 {
      for y in 0..16 {
        for z in 0..16 {
          if checker(x, y, z) {
            grid.set_voxel_ivec3(IVec3::new(x, y, z), PaletteId(1));
          }
        }
      }
    }
    for x in 0..8 {
      for y in 0..8 {
        for z in 0..8 {
          grid.set_voxel_ivec3(c1.0 * CHUNK_SIZE + IVec3::new(x, y, z), PaletteId(2));
        }
      }
    }
    let mut b = build_and_drain(&mut grid);
    let hw0 = b.buffers().b_struct.len();
    for x in 0..16 {
      for y in 0..16 {
        for z in 0..16 {
          if checker(x, y, z) {
            grid.set_voxel_ivec3(IVec3::new(x, y, z), PaletteId::AIR);
          }
        }
      }
    }
    // 笔触进行中：空闲已过半但不压 ⇒ 树区长度不变
    let dirty = take_dirty_of(&mut grid, c0);
    assert_eq!(b.update_chunk(&grid, c0, &dirty, false), ChunkUpdate::Released);
    assert_eq!(b.buffers().b_struct.len(), hw0, "笔触进行中不该压实");
    assert_wire_matches_grid(&b, &grid, "延迟压实后（未压）");
    // 安静了：下一次更新顺手压实 ⇒ 树区变短、存活 chunk 前移
    let dirty = take_dirty_of(&mut grid, c1);
    assert_eq!(b.update_chunk(&grid, c1, &dirty, true), ChunkUpdate::Rebuilt);
    assert!(b.buffers().b_struct.len() < hw0, "安静时应压实");
    assert_eq!(b.chunk_base(c1), Some(TREE_BASE), "存活 chunk 应被前移到树区起点");
    assert_wire_matches_grid(&b, &grid, "延迟压实后（已压）");
    assert_eq!(b.buffers().globals.node_free_words, 0);
  }

  /// 压力：随机切换体素（含清空 / 重建）200 步，每步逐节点核对 wire 字节
  #[test]
  fn incremental_layout_survives_random_edits() {
    let (mut grid, c0, c1) = two_chunk_grid();
    let mut b = build_and_drain(&mut grid);
    let mut seed = 0x1234_5678u32;
    let mut rnd = move || {
      seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
      (seed >> 8) as usize
    };
    for step in 0..200 {
      let coord = if rnd() % 2 == 0 { c0 } else { c1 };
      let off = IVec3::new((rnd() % 24) as i32, (rnd() % 24) as i32, (rnd() % 24) as i32);
      let p = coord.0 * CHUNK_SIZE + off;
      let v = if rnd() % 3 == 0 { PaletteId::AIR } else { PaletteId((rnd() % 3) as u16 + 1) };
      grid.set_voxel_ivec3(p, v);
      let dirty = take_dirty_of(&mut grid, coord);
      b.update_chunk(&grid, coord, &dirty, true);
      assert_wire_matches_grid(&b, &grid, &format!("随机第 {step} 步"));
    }
  }

  /// **本改动的验收指标**：一次单格编辑的上传量 = 路径上的几个节点（几十~几百字节），
  /// 而不是整棵 chunk 树（旧实现是 ~1.4MB/笔）。
  #[test]
  fn single_voxel_edit_uploads_only_the_path() {
    let (mut grid, c0, _) = two_chunk_grid();
    let mut b = build_and_drain(&mut grid);
    let _ = b.take_dirty_ranges();

    grid.set_voxel_ivec3(IVec3::new(5, 5, 5), PaletteId(3));
    let dirty = take_dirty_of(&mut grid, c0);
    assert!(!dirty.reset, "单格编辑不该重置身份空间");
    assert!(dirty.nodes.len() <= 8, "dirty 表应只有路径节点，实际 {:?}", dirty.nodes);
    assert_eq!(b.update_chunk(&grid, c0, &dirty, true), ChunkUpdate::Rebuilt);

    let dr = b.take_dirty_ranges();
    let bytes: usize = dr.struct_ranges.iter().map(|(lo, hi)| hi - lo).sum();
    assert!(
      bytes <= 512,
      "单格编辑应只重写路径节点（≤512B），实际 {bytes}B：{:?}",
      dr.struct_ranges
    );
    assert_wire_matches_grid(&b, &grid, "单格编辑后");
  }

  /// 整 chunk 擦空再重建：Released（块归还）→ Rebuilt（重新安装），全程 wire 与 grid 一致
  #[test]
  fn chunk_becomes_empty_then_rebuilds() {
    let (mut grid, _, c1) = two_chunk_grid();
    let mut b = build_and_drain(&mut grid);
    let origin = c1.0 * CHUNK_SIZE;
    for x in 0..8 {
      for y in 0..8 {
        for z in 0..8 {
          grid.set_voxel_ivec3(origin + IVec3::new(x, y, z), PaletteId::AIR);
        }
      }
    }
    // `allow_compact = false`：这里要看"块有没有归还到全局空闲段"，压实会把它清空
    let dirty = take_dirty_of(&mut grid, c1);
    assert_eq!(b.update_chunk(&grid, c1, &dirty, false), ChunkUpdate::Released);
    assert_eq!(b.chunk_base(c1), None, "擦空后窗口条目应清零");
    assert!(b.buffers().globals.node_free_words > 0, "块应被归还到全局空闲段");

    for x in 0..8 {
      for y in 0..8 {
        for z in 0..8 {
          grid.set_voxel_ivec3(origin + IVec3::new(x, y, z), PaletteId(2));
        }
      }
    }
    // 512 格一次重填会撞上"节点表上限 ⇒ 改用整棵重建"，这正是设计意图
    let dirty = take_dirty_of(&mut grid, c1);
    assert_eq!(b.update_chunk(&grid, c1, &dirty, true), ChunkUpdate::Rebuilt);
    assert_wire_matches_grid(&b, &grid, "重建后");
  }

  /// M3：换出 → 唤醒是**无损**的（唤醒重建的 wire 与 grid 逐节点相同）。
  #[test]
  fn evict_then_wake_is_lossless() {
    let (mut grid, _, c1) = two_chunk_grid();
    let mut b = build_and_drain(&mut grid);
    let words_all = b.resident_words();
    assert!(b.is_resident(c1));

    assert!(b.evict(c1), "常驻的 chunk 应能换出");
    assert!(!b.is_resident(c1), "换出后窗口条目应清零");
    assert_eq!(b.chunk_base(c1), None);
    assert!(b.resident_words() < words_all, "换出应释放字数");
    assert!(!b.evict(c1), "重复换出应无操作");
    // 注意：这里**不能**用 `assert_wire_matches_grid` —— 它要求"每个有内容的 chunk 都有块"，
    // 与"故意换出"天然冲突（换出后窗口条目就该是 0）。

    // CPU 树不动 ⇒ 唤醒就是整块重装，且重建的 wire 必须与 grid 一致
    assert!(b.ensure_resident(&grid, c1), "有内容的 chunk 应能唤醒");
    assert!(b.is_resident(c1));
    assert_eq!(b.resident_words(), words_all, "唤醒后字数应回到原值");
    assert!(!b.ensure_resident(&grid, c1), "已常驻 ⇒ 唤醒应无操作");
    assert_wire_matches_grid(&b, &grid, "唤醒后");
  }

  /// M3：空 chunk（CPU 侧无内容）永远不常驻 —— 唤醒它是无操作。
  #[test]
  fn empty_chunk_is_never_resident() {
    let (mut grid, c0, c1) = two_chunk_grid();
    let mut b = build_and_drain(&mut grid);
    assert!(b.is_resident(c0) && b.is_resident(c1), "有内容的 chunk 建树后即常驻");
    let empty = ChunkCoord(IVec3::new(3, 0, 0));
    assert!(!b.is_resident(empty), "空 chunk 没有块");
    assert!(!b.ensure_resident(&grid, empty), "对空 chunk 唤醒应无操作");
  }
}
