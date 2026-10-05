use std::collections::{HashMap, HashSet};

use gate_voxel::{
  ChunkCoord, ChunkTree, NODE_OFFSET_NONE, NodeLayout, NodeView, PALETTE_INDEX_MAX, PaletteId,
  ROOT_WIRE_WORDS, TreeDirty, VolumeGrid, VolumeTransform, Volumes, pack_palette_word,
};
use glam::{IVec3, Vec4};

use super::wire::{GridDesc, LEAF_INLINE_WORDS, NODE_FIXED_WORDS, pack_palette_entry};
use rayon::prelude::*;

use super::consts::INDEX_ENTRY_EMPTY;
use super::wire::{
  BrickMapBuffers, BrickMapGlobals, CHUNK_INDEX_CAP, PALETTE_BYTES_PER_ENTRY, PALETTE_WORDS,
  TREE_BASE,
};

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

fn chunk_rel(origin: IVec3, dims: IVec3, chunk: IVec3) -> Option<IVec3> {
  let rel = chunk - origin;
  if rel.cmplt(IVec3::ZERO).any() || rel.cmpge(dims).any() {
    return None;
  }
  Some(rel)
}

const OCC_GROUP: i32 = 4;
const OCC_GROUPS: usize = 16 * 16 * 16;
pub const OCC_WORDS: usize = OCC_GROUPS * 2;

#[inline]
fn occ_bit(rel: IVec3) -> usize {
  let g = (rel.x >> 2) as usize + ((rel.y >> 2) as usize) * 16 + ((rel.z >> 2) as usize) * 256;
  let w = (rel.x & 3) as usize | (((rel.y & 3) as usize) << 2) | (((rel.z & 3) as usize) << 4);
  g * 64 + w
}

const _: () = assert!(OCC_GROUP == 4, "组内的位打包按 4×4×4 写死（`>> 2` 与 `& 3`）");
const _: () = assert!(
  OCC_WORDS * 32 == (CHUNK_INDEX_CAP as usize).pow(3),
  "位图必须逐 chunk 一位地覆盖整个索引区（64³ 槽）"
);
const _: () = assert!(OCC_WORDS == 8192, "trace.wesl::OCC_WORDS 必须同值");

fn coord_key(c: ChunkCoord) -> (i32, i32, i32) {
  (c.0.x, c.0.y, c.0.z)
}

#[derive(Default, Debug)]
struct FreeRuns {
  runs: Vec<(usize, usize)>,
}

impl FreeRuns {
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

  fn words(&self) -> usize {
    self.runs.iter().map(|&(_, l)| l).sum()
  }
}

fn chunk_has_content(grid: &VolumeGrid, c: ChunkCoord) -> bool {
  grid.chunk(c).is_some_and(|t| !t.is_empty())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkUpdate {
  Rebuilt,
  Released,
  Unchanged,
  OutsideWindow,
}

const NODE_NONE: u32 = u32::MAX;

#[inline]
fn slot_words(buf: &[u32], base: usize, id: u32, level: u8, off: u32) -> usize {
  if off == NODE_NONE {
    return 0;
  }
  wire_words_of(id, level, read_mask(buf, base + off as usize))
}

const NODE_SLOT_NONE: (u32, u8) = (NODE_OFFSET_NONE, 0);
const _: () = assert!(NODE_NONE == NODE_OFFSET_NONE);

#[derive(Debug)]
struct ChunkSlot {
  base: usize,
  cap: usize,
  free: FreeRuns,
  nodes: gate_voxel::NodeLayout,
  aabbs: crate::brickmap::rt::ChunkAabbs,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidentEvent {
  pub c: ChunkCoord,
  pub mounted: bool,
}

const RESIDENT_LOG_CAP: usize = 1 << 20;

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

#[inline]
fn read_mask(buf: &[u32], at: usize) -> u64 {
  (buf[at + 1] as u64) << 32 | buf[at] as u64
}

#[inline]
fn gather_aabbs(tree: &ChunkTree, c: ChunkCoord) -> crate::brickmap::rt::ChunkAabbs {
  if crate::brickmap::consts::RT_RAY_QUERY {
    crate::brickmap::rt::gather_chunk_aabbs(tree, (c.0 * gate_voxel::CHUNK_SIZE).as_vec3(), 1.0)
  } else {
    crate::brickmap::rt::ChunkAabbs::default()
  }
}

pub struct BrickMapBuilder {
  buffers: BrickMapBuffers,
  chunks: HashMap<ChunkCoord, ChunkSlot>,
  origin: IVec3,
  dims: IVec3,
  rejected_chunks: u32,
  free: FreeRuns,
  dirty_struct: Vec<(usize, usize)>,
  dirty_palette: Option<(u16, u16)>,
  palette_synced_at: Option<u64>,
  reserve: usize,
  region_chunks: usize,
  empty: HashSet<ChunkCoord>,
  scratch_blob: Vec<u32>,
  scratch_layout: NodeLayout,
  resident_log: Vec<ResidentEvent>,
  resident_epoch: u64,
  resident_seq: u64,
  occ: Vec<u32>,
  occ_dirty: bool,
}

#[derive(Debug, Default, Clone)]
pub struct DirtyRanges {
  pub struct_ranges: Vec<(usize, usize)>,
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

  pub fn new_unbuilt(grid: &VolumeGrid) -> Self {
    Self::new_unbuilt_sized(grid, 0, 0)
  }

  pub fn new_unbuilt_sized(grid: &VolumeGrid, reserve_words: usize, region_chunks: usize) -> Self {
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
      region_chunks: region_chunks.max(1),
      empty: HashSet::new(),
      scratch_blob: Vec::new(),
      scratch_layout: NodeLayout::new(),
      resident_log: Vec::new(),
      resident_epoch: 0,
      resident_seq: 0,
      occ: vec![0; OCC_WORDS],
      occ_dirty: true,
    };
    if reserve_words > 0 {
      b.free.free(TREE_BASE, reserve_words);
    }
    b.write_palette(grid);
    let empties: Vec<ChunkCoord> = grid.empty_log_from(0).to_vec();
    for c in empties {
      b.note_empty(c);
    }
    b
  }

  pub fn build_full(grid: &VolumeGrid) -> Self {
    Self::build_full_sized(grid, 0, 0)
  }

  pub fn build_full_sized(grid: &VolumeGrid, reserve_words: usize, region_chunks: usize) -> Self {
    let mut b = Self::new_unbuilt_sized(grid, reserve_words, region_chunks);
    let mut coords: Vec<ChunkCoord> = grid
      .chunk_coords()
      .filter(|&c| chunk_index_pos(b.origin, b.dims, c.0).is_some() && chunk_has_content(grid, c))
      .collect();
    coords.sort_by_key(|&c| coord_key(c));

    let blobs: Vec<(Vec<u32>, NodeLayout, crate::brickmap::rt::ChunkAabbs)> = coords
      .par_iter()
      .map(|&c| {
        let tree = grid.chunk(c).expect("chunk_has_content 已过滤");
        let aabbs = gather_aabbs(tree, c);
        let (blob, layout) = tree.serialize_with_layout();
        (blob, layout, aabbs)
      })
      .collect();
    for (c, (blob, layout, aabbs)) in coords.iter().zip(blobs) {
      b.install_blob(*c, &blob, layout, aabbs);
    }
    b.refresh_globals();

    let _ = b.take_dirty_ranges();
    b
  }

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
        self.release_chunk(coord);
        self.install_chunk(coord, tree);
      }
      ChunkUpdate::Rebuilt
    };
    if allow_compact && self.free.words() * 2 > self.buffers.b_struct.len() - TREE_BASE {
      self.compact();
    }
    self.refresh_globals();

    self.write_palette(grid);
    out
  }

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

  pub fn take_dirty_ranges(&mut self) -> DirtyRanges {
    DirtyRanges {
      struct_ranges: std::mem::take(&mut self.dirty_struct),
      palette_range: self.dirty_palette.take(),
    }
  }

  pub fn has_dirty(&self) -> bool {
    !self.dirty_struct.is_empty() || self.dirty_palette.is_some()
  }

  pub fn chunk_base(&self, coord: ChunkCoord) -> Option<usize> {
    self.chunks.get(&coord).map(|s| s.base)
  }

  //

  pub fn is_resident(&self, coord: ChunkCoord) -> bool {
    self.chunks.contains_key(&coord)
  }

  pub fn resident_words(&self) -> usize {
    self.chunks.values().map(|s| s.cap).sum()
  }

  pub fn resident_bytes_of(&self, coord: ChunkCoord) -> Option<usize> {
    self.chunks.get(&coord).map(|s| s.cap * 4)
  }

  pub fn resident_chunks(&self) -> Vec<ChunkCoord> {
    self.chunks.keys().copied().collect()
  }

  pub fn aabbs_of(&self, coord: ChunkCoord) -> Option<&crate::brickmap::rt::ChunkAabbs> {
    self.chunks.get(&coord).map(|s| &s.aabbs)
  }

  fn note_resident_event(&mut self, c: ChunkCoord, mounted: bool) {
    if self.resident_log.len() >= RESIDENT_LOG_CAP {
      self.resident_log.clear();
      self.resident_epoch = self.resident_epoch.wrapping_add(1);
    }
    self.resident_log.push(ResidentEvent { c, mounted });
    self.resident_seq = self.resident_seq.wrapping_add(1);
  }

  pub fn resident_epoch(&self) -> u64 {
    self.resident_epoch
  }

  pub fn resident_seq(&self) -> u64 {
    self.resident_seq
  }

  pub fn resident_log_len(&self) -> usize {
    self.resident_log.len()
  }

  pub fn resident_log_from(&self, cursor: usize) -> &[ResidentEvent] {
    let from = cursor.min(self.resident_log.len());
    &self.resident_log[from..]
  }

  fn occ_set(&mut self, coord: ChunkCoord) {
    if let Some(rel) = chunk_rel(self.origin, self.dims, coord.0) {
      let b = occ_bit(rel);
      self.occ[b >> 5] |= 1u32 << (b & 31);
      self.occ_dirty = true;
    }
  }

  fn occ_clear(&mut self, coord: ChunkCoord) {
    if let Some(rel) = chunk_rel(self.origin, self.dims, coord.0) {
      let b = occ_bit(rel);
      self.occ[b >> 5] &= !(1u32 << (b & 31));
      self.occ_dirty = true;
    }
  }

  fn occ_rebuild(&mut self) {
    self.occ.fill(0);
    let cs: Vec<ChunkCoord> = self.chunks.keys().copied().collect();
    for c in cs {
      self.occ_set(c);
    }
    self.occ_dirty = true;
  }

  pub fn occ_len(&self) -> usize {
    self.occ.len()
  }

  pub fn occ_words(&self) -> &[u32] {
    &self.occ
  }

  pub fn take_occ_dirty(&mut self) -> bool {
    std::mem::take(&mut self.occ_dirty)
  }

  pub fn occ_dirty_pending(&self) -> bool {
    self.occ_dirty
  }

  pub fn ensure_resident(&mut self, grid: &VolumeGrid, coord: ChunkCoord) -> bool {
    if !chunk_has_content(grid, coord) {
      return false;
    }
    let tree = grid.chunk(coord).expect("chunk_has_content 已判非空");
    self.ensure_resident_tree(coord, tree)
  }

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

  pub fn note_empty(&mut self, coord: ChunkCoord) -> bool {
    if self.chunks.contains_key(&coord) || !self.empty.insert(coord) {
      return false;
    }
    let Some(ip) = chunk_index_pos(self.origin, self.dims, coord.0) else {
      return false;
    };
    self.buffers.b_struct[ip] = INDEX_ENTRY_EMPTY;
    self.mark_struct_words(ip, 1);
    true
  }

  #[inline]
  fn window_word(&self, coord: ChunkCoord) -> usize {
    chunk_index_pos(self.origin, self.dims, coord.0).expect("窗口内 chunk")
  }

  pub fn window(&self) -> (IVec3, IVec3) {
    (self.origin, self.dims)
  }

  fn set_window(&mut self, origin: IVec3, dims: IVec3) {
    if self.origin == origin && self.dims == dims {
      return;
    }
    let old = (self.origin, self.dims);
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
    self.buffers.b_struct[..TREE_BASE].fill(0);
    for (b, entry) in keep {
      self.buffers.b_struct[b] = entry;
    }
    self.origin = origin;
    self.dims = dims;
    for c in self.empty.iter().copied().collect::<Vec<_>>() {
      if self.chunks.contains_key(&c) {
        continue;
      }
      if let Some(b) = chunk_index_pos(origin, dims, c.0) {
        self.buffers.b_struct[b] = INDEX_ENTRY_EMPTY;
      }
    }
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
    self.mark_struct_words(0, TREE_BASE);
    self.occ_rebuild();
    bevy::log::debug!(
      "WINDOW 平移 {} → {origin} dims {dims}（掉了 {} 个出门的）",
      old.0,
      dropped.len()
    );
  }

  fn alloc_block(&mut self, cap: usize) -> usize {
    if let Some(start) = self.free.alloc(cap) {
      return start;
    }
    self.grow_region(cap);
    if let Some(start) = self.free.alloc(cap) {
      return start;
    }
    let start = self.buffers.b_struct.len();
    self.buffers.b_struct.resize(start + cap, 0);
    start
  }

  fn grow_region(&mut self, block_cap: usize) {
    const REGION_CHUNKS_CAP: usize = 32768;
    const PER_CHUNK_WORDS_CAP: usize = 65536;
    const REGION_WORDS_CAP: usize = 512 * 1024 * 1024;
    let used = self.buffers.b_struct.len().saturating_sub(TREE_BASE);
    let per_chunk = (used / self.chunks.len().max(1)).max(block_cap);
    //
    let region_chunks = self.region_chunks.min(REGION_CHUNKS_CAP);
    let per_chunk = per_chunk.min(PER_CHUNK_WORDS_CAP);
    let target = (TREE_BASE + region_chunks.saturating_mul(per_chunk)).min(REGION_WORDS_CAP);
    //
    //
    //
    if self.reserve > 0 && target > self.buffers.b_struct.len() {
      let from = self.buffers.b_struct.len();
      bevy::log::debug!(
        "树区长度一次顶到位：{from} → {target} 字（块 {}、每块估 {per_chunk} 字）",
        self.chunks.len()
      );
      self.buffers.b_struct.resize(target, 0);
      self.free.free(from, target - from);
      return;
    }
    if target <= self.buffers.b_struct.capacity() {
      return;
    }
    let grown = self.buffers.b_struct.capacity().saturating_mul(2).max(TREE_BASE + (1 << 20));
    let want = grown.min(target);
    let extra = want.saturating_sub(self.buffers.b_struct.len());
    if extra > 0 {
      self.buffers.b_struct.reserve(extra);
    }
    bevy::log::debug!(
      "树区容量拨到 {} 字（旧高水位 {used} 字 / {} 块，每块估 {per_chunk} 字）",
      target - TREE_BASE,
      self.chunks.len(),
    );
  }

  fn install_chunk(&mut self, coord: ChunkCoord, tree: &ChunkTree) {
    self.install_chunk_impl(coord, tree, true);
  }

  fn install_chunk_quiet(&mut self, coord: ChunkCoord, tree: &ChunkTree) {
    self.install_chunk_impl(coord, tree, false);
  }

  fn install_chunk_impl(&mut self, coord: ChunkCoord, tree: &ChunkTree, emit: bool) {
    let mut blob = std::mem::take(&mut self.scratch_blob);
    let mut layout = std::mem::take(&mut self.scratch_layout);
    tree.serialize_into(&mut blob, &mut layout);
    let nodes = layout.clone();
    let aabbs = gather_aabbs(tree, coord);
    if emit {
      self.install_blob(coord, &blob, nodes, aabbs);
    } else {
      self.install_blob_inner(coord, &blob, nodes, aabbs);
    }
    self.scratch_blob = blob;
    self.scratch_layout = layout;
  }

  pub fn relayout_resident_tree(&mut self, coord: ChunkCoord, tree: &ChunkTree) -> bool {
    if !self.chunks.contains_key(&coord) || tree.is_empty() {
      return false;
    }
    if chunk_index_pos(self.origin, self.dims, coord.0).is_none() {
      return false;
    }
    self.release_chunk_inner(coord);
    self.install_chunk_quiet(coord, tree);
    self.refresh_globals();
    true
  }

  fn install_blob(
    &mut self,
    coord: ChunkCoord,
    blob: &[u32],
    nodes: gate_voxel::NodeLayout,
    aabbs: crate::brickmap::rt::ChunkAabbs,
  ) {
    self.install_blob_inner(coord, blob, nodes, aabbs);
    self.note_resident_event(coord, true);
  }

  fn install_blob_inner(
    &mut self,
    coord: ChunkCoord,
    blob: &[u32],
    nodes: gate_voxel::NodeLayout,
    aabbs: crate::brickmap::rt::ChunkAabbs,
  ) {
    let need = blob.len();
    let cap = need + need / 4 + ROOT_WIRE_WORDS + 16;
    let base = self.alloc_block(cap);
    self.buffers.b_struct[base..base + need].copy_from_slice(blob);
    let mut free = FreeRuns::default();
    free.free(need, cap - need);
    let ip = self.window_word(coord);
    self.buffers.b_struct[ip] = base as u32 + 1;
    self.chunks.insert(coord, ChunkSlot { base, cap, free, nodes, aabbs });
    self.occ_set(coord);
    self.mark_struct_words(ip, 1);
    self.mark_struct_words(base, need);
  }

  fn release_chunk(&mut self, coord: ChunkCoord) {
    if self.release_chunk_inner(coord) {
      self.note_resident_event(coord, false);
    }
  }

  fn release_chunk_inner(&mut self, coord: ChunkCoord) -> bool {
    let Some(slot) = self.chunks.remove(&coord) else { return false };
    self.occ_clear(coord);
    self.free.free(slot.base, slot.cap);
    let Some(ip) = chunk_index_pos(self.origin, self.dims, coord.0) else { return true };
    self.buffers.b_struct[ip] = if self.empty.contains(&coord) { INDEX_ENTRY_EMPTY } else { 0 };
    self.mark_struct_words(ip, 1);
    true
  }

  fn apply_node_dirty(&mut self, coord: ChunkCoord, tree: &ChunkTree, dirty: &TreeDirty) {
    let mut slot = self.chunks.remove(&coord).expect("调用方保证有块");
    if slot.nodes.len() < tree.node_capacity() {
      slot.nodes.resize(tree.node_capacity(), NODE_SLOT_NONE);
    }
    for &(level, id) in &dirty.nodes {
      let Some(view) = tree.node_view(id) else {
        self.chunks.insert(coord, slot);
        self.release_chunk(coord);
        self.install_chunk(coord, tree);
        return;
      };
      self.rewrite_node(coord, &mut slot, tree, level, id, view);
    }
    self.chunks.insert(coord, slot);
  }

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
    let (old_off, old_level) = slot.nodes[id as usize];
    let old_words = slot_words(&self.buffers.b_struct, slot.base, id, old_level, old_off);

    let mut out: Vec<u32> = Vec::with_capacity(new_words);
    out.push(view.mask as u32);
    out.push((view.mask >> 32) as u32);
    out.push(pack_palette_word(view.palette, view.rep));
    if view.mask != 0 {
      if level == 3 {
        out.extend_from_slice(&tree.node_inline_words(id).expect("level 3 分裂节点有 inline"));
      } else {
        for &c in view.children {
          out.push(slot.nodes[c as usize].0);
        }
      }
    }
    let content = out.len();
    debug_assert!(content == new_words || id == 0, "节点 {id} 的编码字数与定址口径不符");

    //
    if old_off != NODE_NONE && level < 3 && view.mask == 0 {
      let old_at = slot.base + old_off as usize;
      let old_mask = read_mask(&self.buffers.b_struct, old_at);
      for slot_i in 0..old_mask.count_ones() as usize {
        let c = self.buffers.b_struct[old_at + NODE_FIXED_WORDS + slot_i];
        Self::free_subtree(&self.buffers.b_struct, slot, c, level + 1);
      }
    }

    let off = if old_off == NODE_NONE {
      self.alloc_node(coord, slot, new_words)
    } else if new_words <= old_words {
      slot.free.free(old_off as usize + new_words, old_words - new_words);
      old_off
    } else if slot.free.take_range(old_off as usize + old_words, new_words - old_words) {
      old_off
    } else {
      slot.free.free(old_off as usize, old_words);
      self.alloc_node(coord, slot, new_words)
    };

    let at = slot.base + off as usize;
    if off != old_off || self.buffers.b_struct[at..at + content] != out[..] {
      self.buffers.b_struct[at..at + content].copy_from_slice(&out);
      self.mark_struct_words(at, content);
    }
    slot.nodes[id as usize] = (off, level);
  }

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

  fn alloc_node(&mut self, coord: ChunkCoord, slot: &mut ChunkSlot, words: usize) -> u32 {
    if let Some(off) = slot.free.alloc(words) {
      return off as u32;
    }
    self.grow_block(coord, slot);
    slot.free.alloc(words).expect("块扩容量恒 ≥ 单个节点最大字数") as u32
  }

  fn grow_block(&mut self, coord: ChunkCoord, slot: &mut ChunkSlot) {
    let old_cap = slot.cap;
    let new_cap = old_cap + old_cap / 2 + ROOT_WIRE_WORDS + 1;
    let new_base = self.alloc_block(new_cap);
    self.buffers.b_struct.copy_within(slot.base..slot.base + old_cap, new_base);
    self.free.free(slot.base, old_cap);
    slot.base = new_base;
    slot.cap = new_cap;
    slot.free.free(old_cap, new_cap - old_cap);
    let ip = self.window_word(coord);
    self.buffers.b_struct[ip] = new_base as u32 + 1;
    self.mark_struct_words(ip, 1);
    self.mark_struct_words(new_base, old_cap);
  }

  fn compact(&mut self) {
    if self.reserve > 0 {
      return;
    }
    let mut items: Vec<(ChunkCoord, usize, usize)> =
      self.chunks.iter().map(|(&c, s)| (c, s.base, s.cap)).collect();
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

fn words_to_bytes(words: &[u32]) -> Vec<u8> {
  let mut v = Vec::with_capacity(words.len() * 4);
  v.extend_from_slice(unsafe {
    std::slice::from_raw_parts(words.as_ptr() as *const u8, words.len() * 4)
  });
  v
}

#[derive(Debug, Clone)]
pub struct VolumesSnapshot {
  pub b_struct: Vec<u32>,
  pub b_palette: Vec<u32>,
  pub grid_descs: Vec<GridDesc>,
  pub mode_tag: &'static str,
  pub struct_blobs: Vec<(usize, Vec<u8>)>,
  pub palette_blobs: Vec<(usize, Vec<u8>)>,
  pub struct_total_bytes: usize,
  pub palette_total_bytes: usize,
  pub dirty_chunks: usize,
  pub occ_all: Option<Vec<u32>>,
}

pub struct VolumesBuilder {
  builders: Vec<BrickMapBuilder>,
  transforms: Vec<VolumeTransform>,
  far: Vec<bool>,
  coverage: Vec<f32>,
  budget_bytes: usize,
  prev_tree_bases: Vec<u32>,
  prev_palette_bases: Vec<u32>,
  force_full: bool,
}

impl VolumesBuilder {
  fn reserve_of(grid: &gate_voxel::VolumeGrid) -> usize {
    if grid.is_far_level() { super::consts::FAR_TREE_RESERVE_WORDS } else { 0 }
  }

  fn region_chunks_of(grid: &gate_voxel::VolumeGrid, budget_bytes: usize) -> usize {
    if grid.is_far_level() {
      return crate::brickmap::consts::FAR_POOL_CHUNKS;
    }
    if grid.stream_window().is_some() {
      if budget_bytes == 0 {
        crate::brickmap::consts::FAR_POOL_CHUNKS
      } else {
        super::upload::pool_capacity_chunks(budget_bytes)
      }
    } else {
      grid.chunk_count().max(1)
    }
  }

  pub fn build_full(volumes: &Volumes, budget_bytes: usize) -> Self {
    let mut builders = Vec::with_capacity(volumes.len());
    let mut transforms = Vec::with_capacity(volumes.len());
    let mut far = Vec::with_capacity(volumes.len());
    let mut coverage = Vec::with_capacity(volumes.len());
    for grid in volumes.all() {
      builders.push(BrickMapBuilder::build_full_sized(
        grid,
        Self::reserve_of(grid),
        Self::region_chunks_of(grid, budget_bytes),
      ));
      transforms.push(grid.transform());
      far.push(grid.is_far_level());
      coverage.push(grid.coverage_r());
    }
    Self {
      builders,
      transforms,
      far,
      coverage,
      budget_bytes,
      prev_tree_bases: Vec::new(),
      prev_palette_bases: Vec::new(),
      force_full: true,
    }
  }

  pub fn new_unbuilt(volumes: &Volumes, budget_bytes: usize) -> Self {
    let mut builders = Vec::with_capacity(volumes.len());
    let mut transforms = Vec::with_capacity(volumes.len());
    let mut far = Vec::with_capacity(volumes.len());
    let mut coverage = Vec::with_capacity(volumes.len());
    for grid in volumes.all() {
      builders.push(BrickMapBuilder::new_unbuilt_sized(
        grid,
        Self::reserve_of(grid),
        Self::region_chunks_of(grid, budget_bytes),
      ));
      transforms.push(grid.transform());
      far.push(grid.is_far_level());
      coverage.push(grid.coverage_r());
    }
    Self {
      builders,
      transforms,
      far,
      coverage,
      budget_bytes,
      prev_tree_bases: Vec::new(),
      prev_palette_bases: Vec::new(),
      force_full: true,
    }
  }

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
      self.builders.push(BrickMapBuilder::build_full_sized(
        grid,
        Self::reserve_of(grid),
        Self::region_chunks_of(grid, self.budget_bytes),
      ));
      self.transforms.push(grid.transform());
      self.far.push(grid.is_far_level());
      self.coverage.push(grid.coverage_r());
      self.force_full = true;
    }
    for (i, grid) in volumes.all().iter().enumerate() {
      self.transforms[i] = grid.transform();
      self.far[i] = grid.is_far_level();
      self.coverage[i] = grid.coverage_r();
    }
  }

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

  pub fn window_of(&self, volume_idx: usize) -> Option<(IVec3, IVec3)> {
    self.builders.get(volume_idx).map(BrickMapBuilder::window)
  }

  pub fn has_dirty(&self) -> bool {
    self.builders.iter().any(BrickMapBuilder::has_dirty)
  }

  pub fn sync_windows(&mut self, volumes: &Volumes) {
    for (i, g) in volumes.all().iter().enumerate() {
      if let Some((o, d)) = g.stream_window() {
        self.builders[i].set_window(o, d);
      }
    }
  }

  pub fn force_full(&mut self) {
    self.force_full = true;
  }

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
      let (mn, mx) = super::wire::window_world_aabb(tr, origin, dims);
      desc.aabb_min = Vec4::new(mn.x, mn.y, mn.z, 0.0);
      desc.aabb_max = Vec4::new(mx.x, mx.y, mx.z, 0.0);
      desc.grid_flags = if self.far[i] { super::wire::GRID_FLAG_FAR } else { 0 };
      desc.coverage_r = self.coverage[i];
      grid_descs.push(desc);
    }

    let bases_shifted = self.prev_tree_bases.len() != tree_bases.len()
      || self.prev_tree_bases.iter().zip(tree_bases.iter()).any(|(p, c)| p != c)
      || self.prev_palette_bases.iter().zip(palette_bases.iter()).any(|(p, c)| p != c);
    let need_full = self.force_full || bases_shifted;
    if bases_shifted {
      let info: Vec<String> = self
        .builders
        .iter()
        .enumerate()
        .map(|(i, b)| {
          let len = b.buffers.b_struct.len();
          let used = len.saturating_sub(TREE_BASE);
          format!("v{i} 树区 {used} 字/预留 {}（{} 块）", b.reserve, b.chunks.len())
        })
        .collect();
      bevy::log::debug!("BASES 漂移 → 全量快照：{}", info.join(" | "));
    }

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
      occ_all: {
        if self.builders.iter().any(BrickMapBuilder::occ_dirty_pending) {
          let mut v = Vec::with_capacity(self.builders.len() * OCC_WORDS);
          for b in self.builders.iter_mut() {
            b.take_occ_dirty();
            v.extend_from_slice(b.occ_words());
          }
          Some(v)
        } else {
          None
        }
      },
    }
  }

  pub fn chunk_base(&self, volume_idx: usize, coord: ChunkCoord) -> Option<usize> {
    self.builders[volume_idx].chunk_base(coord)
  }

  pub fn is_resident(&self, vol_idx: usize, coord: ChunkCoord) -> bool {
    self.builders.get(vol_idx).is_some_and(|b| b.is_resident(coord))
  }

  pub fn resident_words(&self, vol_idx: usize) -> usize {
    self.builders.get(vol_idx).map_or(0, BrickMapBuilder::resident_words)
  }

  pub fn unbudgeted_struct_bytes(&self) -> usize {
    let mut words = 0usize;
    for (i, b) in self.builders.iter().enumerate() {
      if i == 0 {
        words += b.buffers.b_struct.len().saturating_sub(b.resident_words());
      } else {
        words += b.buffers.b_struct.len();
      }
    }
    words * 4
  }

  pub fn resident_bytes_of(&self, vol_idx: usize, coord: ChunkCoord) -> Option<usize> {
    self.builders.get(vol_idx).and_then(|b| b.resident_bytes_of(coord))
  }

  pub fn resident_chunks(&self, vol_idx: usize) -> Vec<ChunkCoord> {
    self.builders.get(vol_idx).map(BrickMapBuilder::resident_chunks).unwrap_or_default()
  }

  pub fn note_empty(&mut self, vol_idx: usize, coord: ChunkCoord) -> bool {
    self.builders.get_mut(vol_idx).is_some_and(|b| b.note_empty(coord))
  }

  pub fn window(&self, vol_idx: usize) -> (IVec3, IVec3) {
    self.builders.get(vol_idx).map(BrickMapBuilder::window).unwrap_or((IVec3::ZERO, IVec3::ZERO))
  }

  pub fn ensure_resident_tree(
    &mut self,
    vol_idx: usize,
    coord: ChunkCoord,
    tree: &ChunkTree,
  ) -> bool {
    self.builders.get_mut(vol_idx).is_some_and(|b| b.ensure_resident_tree(coord, tree))
  }

  pub fn relayout_resident_tree(
    &mut self,
    vol_idx: usize,
    coord: ChunkCoord,
    tree: &ChunkTree,
  ) -> bool {
    self.builders.get_mut(vol_idx).is_some_and(|b| b.relayout_resident_tree(coord, tree))
  }

  pub fn aabbs_of(
    &self,
    vol_idx: usize,
    coord: ChunkCoord,
  ) -> Option<&crate::brickmap::rt::ChunkAabbs> {
    self.builders.get(vol_idx).and_then(|b| b.aabbs_of(coord))
  }

  pub fn resident_epoch(&self, vol_idx: usize) -> u64 {
    self.builders.get(vol_idx).map_or(0, BrickMapBuilder::resident_epoch)
  }

  pub fn resident_seq(&self, vol_idx: usize) -> u64 {
    self.builders.get(vol_idx).map_or(0, BrickMapBuilder::resident_seq)
  }

  pub fn resident_log_len(&self, vol_idx: usize) -> usize {
    self.builders.get(vol_idx).map_or(0, BrickMapBuilder::resident_log_len)
  }

  pub fn resident_log_from(&self, vol_idx: usize, cursor: usize) -> &[ResidentEvent] {
    self.builders.get(vol_idx).map_or(&[], |b| b.resident_log_from(cursor))
  }

  pub fn ensure_resident(&mut self, volumes: &Volumes, vol_idx: usize, coord: ChunkCoord) -> bool {
    let Some(b) = self.builders.get_mut(vol_idx) else { return false };
    let Some(g) = volumes.all().get(vol_idx) else { return false };
    b.ensure_resident(g, coord)
  }

  pub fn evict(&mut self, vol_idx: usize, coord: ChunkCoord) -> bool {
    self.builders.get_mut(vol_idx).is_some_and(|b| b.evict(coord))
  }

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

fn compute_window(grid: &VolumeGrid) -> (IVec3, IVec3, usize) {
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