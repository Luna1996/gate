//! CPU 砖块图构建器：`VolumeGrid` → wire 格式（wire.rs §b_struct 契约）。
//!
//! [`BrickMapBuilder`]（单 volume）：全量构建按 ChunkCoord 排序 + Rayon 并行
//! `ChunkTree::serialize()` + 顺序 append（字节级确定性）；增量
//! [`BrickMapBuilder::update_chunk`] 重序列化 append + 窗口条目改指新基址，旧树字节作废计入
//! `globals.node_free_words`（全量重建归零）——append-only，无原地修改。
//!
//! [`VolumesBuilder`]（多 volume）：持有 `Vec<BrickMapBuilder>`，各 volume 的 b_struct 顺序拼接，
//! `GridDesc.tree_base` 指向统一 buffer 内的绝对字基址；增量脏区间按 `tree_base` 偏移后传给 GPU
//! partial write，任一前置 volume 增长导致后续 tree_base 漂移 → 自动降级全量。

use std::collections::HashMap;

use gate_voxel::{ChunkCoord, PALETTE_INDEX_MAX, PaletteId, VolumeGrid, VolumeTransform, Volumes};
use glam::{IVec3, Vec4};

use super::wire::{CHUNK_SIZE, GridDesc, pack_palette_entry};
use rayon::prelude::*;

use super::wire::{
  BrickMapBuffers, BrickMapGlobals, CHUNK_INDEX_CAP, PALETTE_BYTES_PER_ENTRY, PALETTE_WORDS,
  TREE_BASE,
};

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
  /// 待上传的调色板脏槽闭区间（同一帧多次 `write_palette` 取并集）；None = 无变动。
  /// 表本体 512KB，故不再整表重铺（见 [`Self::write_palette`]）。
  dirty_palette: Option<(u16, u16)>,
  /// 调色板同步游标：本 builder 上次同步时调色板的写版本；None = 从未同步过。
  /// 用它而非"脏区间是否为空"来判断，因为同一张表可能有多个消费者（见 [`Self::write_palette`]）。
  palette_synced_at: Option<u64>,
}

/// 脏字节区间列表（prepare 按此逐项 write_buffer 部分写 GPU）。
/// 空列表 = 对应 buffer 完全未修改，跳过写。
#[derive(Debug, Default, Clone)]
pub struct DirtyRanges {
  pub struct_ranges: Vec<(usize, usize)>,
  /// 调色板脏槽闭区间（槽号，非字节）；None = 本次无槽变动
  pub palette_range: Option<(u16, u16)>,
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
      dirty_palette: None,
      palette_synced_at: None,
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
    // 顺带铺 palette 的**脏槽**（材质是"用时才写进调色板"的，见 gate-app/src/edit.rs）：
    // 必须与触发它的那次体素编辑**同一帧**上传，否则新放的体素会以槽位上一任材质的颜色出现。
    // 表本体 2^16 条 = 512KB，不能像原来 256 条（2KB）那样整表重铺，故只写变动槽区间。
    self.write_palette(grid);
    out
  }

  /// 调色板增量铺：把 grid 侧"自本 builder 上次同步以来变化"的槽写进 CPU 镜像并累积脏区间。
  ///
  /// **多消费者安全**：同一张表可能被多个 builder 消费（渲染世界每 volume 一个，外加
  /// gate-app 启动期的诊断 builder —— 见 `gate-app/src/scene.rs` 的 `BrickMapBuilder::build_full`）。
  /// 脏区间只能被取走一次，所以判断"要不要同步"用调色板自己的写版本，而不是脏区间是否为空；
  /// 且**首次同步必然全量**（镜像初始全零），非首次拿不到脏区间时也退回全量，绝不静默跳过。
  /// 首次全量铺满 512KB，此后每次材质新增最多几个槽。
  pub fn write_palette(&mut self, grid: &VolumeGrid) {
    let version = grid.palette().version();
    if self.palette_synced_at == Some(version) {
      return; // 自本 builder 上次同步以来调色板未变
    }
    let range = match self.palette_synced_at {
      None => {
        // 首次同步：本镜像全零，必须整表铺；顺手把脏区间消费掉（内容已被整表覆盖）
        let _ = grid.palette().take_dirty();
        None
      }
      // 非首次：拿到脏区间就只铺那一段；被别人取走（None）则退回全量
      Some(_) => grid.palette().take_dirty(),
    };
    let (lo, hi) = range.unwrap_or((0, PALETTE_INDEX_MAX));
    for i in lo..=hi {
      let [a, b] = pack_palette_entry(grid.palette().get(PaletteId(i)));
      self.buffers.b_palette[i as usize * 2] = a;
      self.buffers.b_palette[i as usize * 2 + 1] = b;
    }
    self.palette_synced_at = Some(version);
    // 累积进"待上传槽区间"（同一帧多次 update_chunk 取并集）
    self.dirty_palette = Some(match self.dirty_palette {
      None => (lo, hi),
      Some((d0, d1)) => (d0.min(lo), d1.max(hi)),
    });
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
      palette_range: self.dirty_palette.take(),
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
// VolumesBuilder：多 volume 统一构建器
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

  /// 取走统一快照：拼接所有 volume 的 b_struct/b_palette + 生成 GridDesc 数组。
  ///
  /// 布局策略：物体 (1..N) 先放、主世界 (0) 后放，使主世界 b_struct 增长不漂移任何前置
  /// volume 的 tree_base（→ 可走增量上传）；GridDesc 数组仍按 volume 索引顺序
  /// [0, 1, 2, ...]（主世界 = 0），tree_base 指向统一 buffer 内的实际位置。
  pub fn snapshot(&mut self) -> VolumesSnapshot {
    let n = self.builders.len();
    // b_struct 布局序：物体 1..N 先，主世界 0 后（主世界编辑不漂移物体）
    let layout_order: Vec<usize> = (1..n).chain(std::iter::once(0)).collect();

    // tree_bases[i] / palette_bases[i] = volume i 在统一 buffer 内的字基址。
    // 只累加字数，不拼接字节。
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
      if i == 0 {
        // 主世界（identity：局部=世界）：AABB = chunk 窗口范围（voxel 单位）。
        // from_transform 默认给 [0,256]³·scale——窗口 origin 可为负且 dims 巨大，
        // 默认盒会把窗口绝大部分 slab 剔除 → 全屏只渲染 chunk(0,0,0) 附近一小块。
        desc.aabb_min = Vec4::new(
          (origin.x * CHUNK_SIZE) as f32,
          (origin.y * CHUNK_SIZE) as f32,
          (origin.z * CHUNK_SIZE) as f32,
          0.0,
        );
        desc.aabb_max = Vec4::new(
          ((origin.x + dims.x) * CHUNK_SIZE) as f32,
          ((origin.y + dims.y) * CHUNK_SIZE) as f32,
          ((origin.z + dims.z) * CHUNK_SIZE) as f32,
          0.0,
        );
      }
      grid_descs.push(desc);
    }

    // 漂移检测：volume 数变化 / 任一 tree_base 或 palette_base 变化 → 全量
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
        if let Some((lo, hi)) = dr.palette_range {
          // 脏槽闭区间 → 统一 palette buffer 内的字节偏移 + 该区间字节（每槽 8B = 2 字）
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
