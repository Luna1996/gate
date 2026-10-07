use super::coords::{BRICK_FACTOR, CHUNK_SIZE, LEVEL_EXTENT, child_linear_idx};
use crate::palette::{PALETTE_BITS, PaletteId};

pub const LEAF_VOXELS_PER_WORD: usize = 32 / PALETTE_BITS as usize;
pub const LEAF_INLINE_WORDS: usize = 64 / LEAF_VOXELS_PER_WORD;
const LEAF_VOXELS: usize = LEAF_INLINE_WORDS * LEAF_VOXELS_PER_WORD;
const REP_TALLY_CAP: usize = 8;

const NODE_FIXED_WORDS: usize = 3;
pub const ROOT_WIRE_WORDS: usize = NODE_FIXED_WORDS + 64;

pub const NODE_OFFSET_NONE: u32 = u32::MAX;
const DIRTY_NODE_CAP: usize = 4096;
pub type NodeLayout = Vec<(u32, u8)>;

#[derive(Debug, Clone, Copy)]
pub struct NodeView<'a> {
  pub mask: u64,
  pub palette: PaletteId,
  pub rep: PaletteId,
  pub children: &'a [u32],
}

#[derive(Debug, Default, Clone)]
pub struct TreeDirty {
  pub reset: bool,
  pub nodes: Vec<(u8, u32)>,
}

#[derive(Debug, Clone, Copy)]
struct Leaf<'a> {
  mask: u64,
  palette: PaletteId,
  inline: &'a [u32; LEAF_INLINE_WORDS],
}

impl Leaf<'_> {
  #[inline]
  fn get(&self, i: u32) -> PaletteId {
    if (self.mask & (1u64 << i)) == 0 {
      return self.palette;
    }
    let w = self.inline[(i >> 1) as usize];
    PaletteId(((w >> ((i & 1) * 16)) & 0xFFFF) as u16)
  }

  #[inline]
  fn rep(&self) -> PaletteId {
    leaf_rep_palette(self.mask, self.palette, self.inline)
  }

  fn tile_color(&self) -> Option<PaletteId> {
    let mut target = if self.mask == u64::MAX { None } else { Some(self.palette) };
    let mut m = self.mask;
    while m != 0 {
      let i = m.trailing_zeros();
      m &= m - 1;
      let w = self.inline[(i >> 1) as usize];
      let c = PaletteId(((w >> ((i & 1) * 16)) & 0xFFFF) as u16);
      match target {
        None => target = Some(c),
        Some(t) if t != c => return None,
        _ => {}
      }
    }
    target
  }
}

struct NodePath {
  ids: [usize; LEVEL_EXTENT.len()],
  len: usize,
}

impl NodePath {
  fn new() -> Self {
    Self { ids: [0; LEVEL_EXTENT.len()], len: 0 }
  }

  fn push(&mut self, id: usize) {
    debug_assert!(self.len < self.ids.len(), "路径长度不该超过 brick 层数");
    self.ids[self.len] = id;
    self.len += 1;
  }

  fn iter_rev(&self) -> impl Iterator<Item = usize> + '_ {
    self.ids[..self.len].iter().rev().copied()
  }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NodeKind {
  Uniform,
  Split,
  Leaf,
}

#[derive(Debug, Clone, Copy)]
struct Node {
  mask: u64,
  off: u32,
  palette: PaletteId,
  kind: NodeKind,
}

impl Node {
  #[inline]
  fn uniform(palette: PaletteId) -> Self {
    Self { mask: 0, off: 0, palette, kind: NodeKind::Uniform }
  }
}

const _: () = assert!(std::mem::size_of::<Node>() == 16);

#[derive(Debug, Clone, Default)]
struct ChildPool {
  words: Vec<u32>,
  free: Vec<(u32, u32)>,
}

impl ChildPool {
  fn alloc(&mut self, n: usize) -> u32 {
    let cap = slot_cap(n);
    if cap == 0 {
      return 0;
    }
    if let Some(i) = self.free.iter().position(|&(_, len)| len as usize >= cap) {
      let (off, len) = self.free[i];
      if len as usize == cap {
        self.free.swap_remove(i);
      } else {
        self.free[i] = (off + cap as u32, len - cap as u32);
      }
      return off;
    }
    let off = self.words.len() as u32;
    self.words.resize(self.words.len() + cap, 0);
    off
  }

  fn free(&mut self, off: u32, n: usize) {
    let cap = slot_cap(n);
    if cap == 0 {
      return;
    }
    self.free.push((off, cap as u32));
  }

  #[inline]
  fn slice(&self, off: u32, n: usize) -> &[u32] {
    &self.words[off as usize..off as usize + n]
  }

  #[inline]
  fn slice_mut(&mut self, off: u32, n: usize) -> &mut [u32] {
    &mut self.words[off as usize..off as usize + n]
  }

  fn insert(&mut self, off: u32, n: usize, slot: usize, value: u32) -> u32 {
    if slot_cap(n + 1) == slot_cap(n) {
      let s = self.slice_mut(off, n + 1);
      s.copy_within(slot..n, slot + 1);
      s[slot] = value;
      return off;
    }
    let mut tmp = [0u32; 64];
    tmp[..n].copy_from_slice(self.slice(off, n));
    tmp.copy_within(slot..n, slot + 1);
    tmp[slot] = value;
    let new_off = self.alloc(n + 1);
    self.slice_mut(new_off, n + 1).copy_from_slice(&tmp[..n + 1]);
    self.free(off, n);
    new_off
  }

  fn clear(&mut self) {
    self.words.clear();
    self.free.clear();
  }
}

#[inline]
fn slot_cap(n: usize) -> usize {
  if n == 0 { 0 } else { n.next_power_of_two() }
}

static EMPTY_LEAF: [u32; LEAF_INLINE_WORDS] = [0; LEAF_INLINE_WORDS];

#[inline]
fn child_slot(mask: u64, i: u32) -> u32 {
  debug_assert!(i < 64 && (mask & (1u64 << i)) != 0, "child_slot 要求 bit=1");
  (mask & ((1u64 << i) - 1)).count_ones()
}

#[inline]
pub fn pack_palette_word(palette: PaletteId, rep: PaletteId) -> u32 {
  palette.get() as u32 | ((rep.get() as u32) << 16)
}

pub fn leaf_rep_palette(
  mask: u64,
  palette: PaletteId,
  inline: &[u32; LEAF_INLINE_WORDS],
) -> PaletteId {
  let mut tally = [(0u16, 0u8); REP_TALLY_CAP];
  let mut used = 0usize;
  let mut first = 0u16;
  for i in 0..LEAF_VOXELS as u32 {
    let v = if mask & (1u64 << i) == 0 {
      palette.get()
    } else {
      let w = inline[(i >> 1) as usize];
      ((w >> ((i & 1) * 16)) & 0xFFFF) as u16
    };
    if v == 0 {
      continue;
    }
    if first == 0 {
      first = v;
    }
    match tally[..used].iter_mut().find(|(id, _)| *id == v) {
      Some(slot) => slot.1 = slot.1.saturating_add(1),
      None if used < REP_TALLY_CAP => {
        tally[used] = (v, 1);
        used += 1;
      }
      None => return PaletteId(first),
    }
  }
  if used == 0 {
    return PaletteId::AIR;
  }
  let mut best = (u16::MAX, 0u8);
  for &(id, w) in &tally[..used] {
    if w > best.1 || (w == best.1 && id < best.0) {
      best = (id, w);
    }
  }
  PaletteId(best.0)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BrickState {
  Air,
  Solid(PaletteId),
  Mixed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NodeDesc {
  pub mask: u64,
  pub palette: PaletteId,
}

#[inline]
fn brick_state_of(palette: PaletteId) -> BrickState {
  if palette.is_air() { BrickState::Air } else { BrickState::Solid(palette) }
}

#[inline]
fn solid(palette: PaletteId) -> Option<PaletteId> {
  if palette.is_air() { None } else { Some(palette) }
}

#[inline]
pub fn level_of_extent(extent: i32) -> u8 {
  LEVEL_EXTENT
    .iter()
    .position(|&e| e == extent)
    .unwrap_or_else(|| panic!("extent 必须是 brick 粒度 {LEVEL_EXTENT:?} 之一（got {extent}）"))
    as u8
}

#[derive(Debug, Clone)]
pub struct ChunkTree {
  nodes: Vec<Node>,
  children: ChildPool,
  leaves: Vec<[u32; LEAF_INLINE_WORDS]>,
  root_palette: PaletteId,
  identity_reset: bool,
  dirty_nodes: Vec<(u8, u32)>,
}

impl ChunkTree {
  pub fn empty() -> Self {
    Self {
      nodes: Vec::new(),
      children: ChildPool::default(),
      leaves: Vec::new(),
      root_palette: PaletteId::AIR,
      identity_reset: false,
      dirty_nodes: Vec::new(),
    }
  }

  pub fn uniform(palette: PaletteId) -> Self {
    Self { root_palette: palette, ..Self::empty() }
  }

  #[inline]
  fn mark(&mut self, extent: i32, id: usize) {
    if extent <= 1 {
      return;
    }
    if self.dirty_nodes.len() >= DIRTY_NODE_CAP {
      self.dirty_nodes.clear();
      self.identity_reset = true;
      return;
    }
    self.dirty_nodes.push((level_of_extent(extent), id as u32));
  }

  #[inline]
  fn mark_identity_reset(&mut self) {
    self.identity_reset = true;
  }

  pub fn take_dirty(&mut self) -> TreeDirty {
    let reset = std::mem::take(&mut self.identity_reset);
    let mut nodes = std::mem::take(&mut self.dirty_nodes);
    if reset {
      nodes.clear();
    } else {
      nodes.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
      nodes.dedup();
    }
    TreeDirty { reset, nodes }
  }

  pub fn mark_replaced(&mut self) {
    self.mark_identity_reset();
  }

  pub fn node_capacity(&self) -> usize {
    self.nodes.len()
  }

  pub const NODE_BYTES: usize = std::mem::size_of::<Node>();

  pub fn heap_bytes(&self) -> usize {
    self.nodes.len() * Self::NODE_BYTES
      + self.children.words.capacity() * std::mem::size_of::<u32>()
      + self.leaves.capacity() * LEAF_INLINE_WORDS * std::mem::size_of::<u32>()
      + self.dirty_nodes.capacity() * std::mem::size_of::<(u8, u32)>()
  }

  pub fn root_palette(&self) -> PaletteId {
    self.root_palette
  }

  pub fn is_uniform_root(&self) -> bool {
    self.nodes.is_empty()
  }

  pub fn node_view(&self, id: u32) -> Option<NodeView<'_>> {
    let n = self.nodes.get(id as usize)?;
    Some(NodeView {
      mask: n.mask,
      palette: n.palette,
      rep: self.node_rep(id as usize),
      children: self.node_children(*n),
    })
  }

  #[inline]
  fn node_rep(&self, id: usize) -> PaletteId {
    match self.nodes.get(id) {
      Some(n) if n.kind == NodeKind::Leaf => self.leaf(id).rep(),
      _ => PaletteId::AIR,
    }
  }

  pub fn node_inline_words(&self, id: u32) -> Option<[u32; LEAF_INLINE_WORDS]> {
    let n = *self.nodes.get(id as usize)?;
    match n.kind {
      NodeKind::Leaf => Some(*self.leaf_words(n)),
      _ => None,
    }
  }

  #[inline]
  fn node_mask_palette(&self, idx: usize) -> (u64, PaletteId) {
    let n = self.nodes[idx];
    (n.mask, n.palette)
  }

  #[inline]
  fn leaf_words(&self, n: Node) -> &[u32; LEAF_INLINE_WORDS] {
    if n.mask == 0 { &EMPTY_LEAF } else { &self.leaves[n.off as usize] }
  }

  #[inline]
  fn leaf(&self, id: usize) -> Leaf<'_> {
    let n = self.nodes[id];
    Leaf { mask: n.mask, palette: n.palette, inline: self.leaf_words(n) }
  }

  #[inline]
  fn leaf_set(&mut self, id: usize, i: u32, palette: PaletteId) {
    if self.nodes[id].mask == 0 {
      self.leaves.push([0; LEAF_INLINE_WORDS]);
      self.nodes[id].off = self.leaves.len() as u32 - 1;
    }
    let slot = (i >> 1) as usize;
    let shift = (i & 1) * 16;
    let w = &mut self.leaves[self.nodes[id].off as usize][slot];
    *w = (*w & !(0xFFFF << shift)) | ((palette.get() as u32) << shift);
    self.nodes[id].mask |= 1u64 << i;
  }

  fn open_leaf(&mut self, idx: usize) {
    match self.nodes[idx].kind {
      NodeKind::Leaf => return,
      NodeKind::Uniform => {}
      NodeKind::Split => unreachable!("层 3 恒为值块"),
    }
    let palette = self.nodes[idx].palette;
    self.nodes[idx] = Node { mask: 0, off: 0, palette, kind: NodeKind::Leaf };
  }

  #[inline]
  fn node_children(&self, n: Node) -> &[u32] {
    match n.kind {
      NodeKind::Split => self.children.slice(n.off, n.mask.count_ones() as usize),
      _ => &[],
    }
  }

  #[inline]
  fn release_child_block(&mut self, idx: usize) {
    let n = self.nodes[idx];
    if n.kind == NodeKind::Split {
      self.children.free(n.off, n.mask.count_ones() as usize);
    }
  }

  #[inline]
  fn child_at(&self, n: Node, cell: u32) -> usize {
    self.children.slice(n.off, n.mask.count_ones() as usize)[child_slot(n.mask, cell) as usize]
      as usize
  }

  pub fn serialize(&self) -> Vec<u32> {
    self.serialize_with_layout().0
  }

  pub fn serialize_with_layout(&self) -> (Vec<u32>, NodeLayout) {
    let mut out = Vec::with_capacity(self.nodes.len() * 3 + 4096);
    let mut layout: NodeLayout = Vec::new();
    self.serialize_into(&mut out, &mut layout);
    (out, layout)
  }

  pub fn serialize_into(&self, out: &mut Vec<u32>, layout: &mut NodeLayout) {
    out.clear();
    layout.clear();
    layout.resize(self.nodes.len().max(1), (NODE_OFFSET_NONE, 0));
    let (mask, palette) = match self.nodes.first() {
      None => (0u64, self.root_palette),
      Some(n) => (n.mask, n.palette),
    };
    out.push(mask as u32);
    out.push((mask >> 32) as u32);
    out.push(pack_palette_word(palette, self.node_rep(0)));
    layout[0] = (0, 0);
    out.resize(ROOT_WIRE_WORDS, 0);
    match self.nodes.first() {
      None => {}
      Some(n) => match n.kind {
        NodeKind::Uniform => {}
        NodeKind::Leaf => out.extend_from_slice(self.leaf_words(*n)),
        NodeKind::Split => {
          let child_extent = CHUNK_SIZE / BRICK_FACTOR;
          let count = n.mask.count_ones() as usize;
          for slot in 0..count {
            let child = self.children.slice(n.off, count)[slot] as usize;
            out[NODE_FIXED_WORDS + slot] = out.len() as u32;
            self.serialize_node(child, child_extent, out, layout);
          }
        }
      },
    }
  }

  fn serialize_node(&self, idx: usize, extent: i32, out: &mut Vec<u32>, layout: &mut NodeLayout) {
    layout[idx] = (out.len() as u32, level_of_extent(extent));
    let n = self.nodes[idx];
    out.push(n.mask as u32);
    out.push((n.mask >> 32) as u32);
    out.push(pack_palette_word(n.palette, self.node_rep(idx)));
    if n.mask == 0 {
      return;
    }
    if n.kind == NodeKind::Leaf {
      out.extend_from_slice(self.leaf_words(n));
      return;
    }
    let child_extent = extent / BRICK_FACTOR;
    let num_children = n.mask.count_ones() as usize;
    let offsets_start = out.len();
    out.resize(out.len() + num_children, 0);
    let mut m = n.mask;
    let mut slot = 0usize;
    while m != 0 {
      m &= m - 1;
      out[offsets_start + slot] = out.len() as u32;
      let child_idx = self.children.slice(n.off, num_children)[slot] as usize;
      slot += 1;
      self.serialize_node(child_idx, child_extent, out, layout);
    }
  }

  pub fn len_words(&self) -> usize {
    let Some(n) = self.nodes.first() else {
      return ROOT_WIRE_WORDS;
    };
    let mut total = ROOT_WIRE_WORDS;
    match n.kind {
      NodeKind::Uniform => {}
      NodeKind::Leaf => total += LEAF_INLINE_WORDS,
      NodeKind::Split => {
        let count = n.mask.count_ones() as usize;
        for slot in 0..count {
          total += self.count_node(self.children.slice(n.off, count)[slot] as usize);
        }
      }
    }
    total
  }

  fn count_node(&self, idx: usize) -> usize {
    let n = self.nodes[idx];
    let mut total = NODE_FIXED_WORDS;
    if n.mask == 0 {
      return total;
    }
    if n.kind == NodeKind::Leaf {
      return total + LEAF_INLINE_WORDS;
    }
    let count = n.mask.count_ones() as usize;
    total += count;
    for slot in 0..count {
      total += self.count_node(self.children.slice(n.off, count)[slot] as usize);
    }
    total
  }

  pub fn get_voxel(&self, local_x: i32, local_y: i32, local_z: i32) -> Option<PaletteId> {
    if self.nodes.is_empty() {
      return solid(self.root_palette);
    }
    self.get_at(
      local_x.clamp(0, CHUNK_SIZE - 1),
      local_y.clamp(0, CHUNK_SIZE - 1),
      local_z.clamp(0, CHUNK_SIZE - 1),
      0,
      CHUNK_SIZE,
    )
  }

  pub fn block_solid_bits(&self, local_x: i32, local_y: i32, local_z: i32) -> u64 {
    if self.nodes.is_empty() {
      return if self.root_palette.is_air() { 0 } else { u64::MAX };
    }
    self.block_solid_bits_at(
      local_x.clamp(0, CHUNK_SIZE - 1),
      local_y.clamp(0, CHUNK_SIZE - 1),
      local_z.clamp(0, CHUNK_SIZE - 1),
      0,
      CHUNK_SIZE,
    )
  }

  fn block_solid_bits_at(&self, x: i32, y: i32, z: i32, idx: usize, extent: i32) -> u64 {
    let n = self.nodes[idx];
    let uniform = || if n.palette.is_air() { 0 } else { u64::MAX };
    match n.kind {
      NodeKind::Uniform => uniform(),
      NodeKind::Leaf => {
        let l = self.leaf(idx);
        let mut bits = 0u64;
        for i in 0u32..64 {
          if !l.get(i).is_air() {
            bits |= 1u64 << i;
          }
        }
        bits
      }
      NodeKind::Split => {
        if n.mask == 0 {
          return uniform();
        }
        let child_extent = extent / BRICK_FACTOR;
        let (ix, iy, iz) = (
          (x / child_extent).clamp(0, BRICK_FACTOR - 1),
          (y / child_extent).clamp(0, BRICK_FACTOR - 1),
          (z / child_extent).clamp(0, BRICK_FACTOR - 1),
        );
        let cell = child_linear_idx(ix, iy, iz);
        if (n.mask & (1u64 << cell)) == 0 {
          return uniform();
        }
        let child = self.child_at(n, cell);
        self.block_solid_bits_at(
          x - ix * child_extent,
          y - iy * child_extent,
          z - iz * child_extent,
          child,
          child_extent,
        )
      }
    }
  }

  fn get_at(&self, x: i32, y: i32, z: i32, idx: usize, extent: i32) -> Option<PaletteId> {
    let n = self.nodes[idx];
    match n.kind {
      NodeKind::Uniform => solid(n.palette),
      NodeKind::Leaf => solid(self.leaf(idx).get(child_linear_idx(x, y, z))),
      NodeKind::Split => {
        if n.mask == 0 {
          return solid(n.palette);
        }
        let child_extent = extent / BRICK_FACTOR;
        let (ix, iy, iz) = (
          (x / child_extent).clamp(0, BRICK_FACTOR - 1),
          (y / child_extent).clamp(0, BRICK_FACTOR - 1),
          (z / child_extent).clamp(0, BRICK_FACTOR - 1),
        );
        let cell = child_linear_idx(ix, iy, iz);
        if (n.mask & (1u64 << cell)) == 0 {
          return solid(n.palette);
        }
        let child = self.child_at(n, cell);
        self.get_at(
          x - ix * child_extent,
          y - iy * child_extent,
          z - iz * child_extent,
          child,
          child_extent,
        )
      }
    }
  }

  pub fn get_uniform(
    &self,
    local_x: i32,
    local_y: i32,
    local_z: i32,
    level: u8,
  ) -> Option<PaletteId> {
    let query_extent = LEVEL_EXTENT[level as usize];
    if self.nodes.is_empty() {
      return solid(self.root_palette);
    }
    self.get_uniform_at(
      local_x.clamp(0, CHUNK_SIZE - 1),
      local_y.clamp(0, CHUNK_SIZE - 1),
      local_z.clamp(0, CHUNK_SIZE - 1),
      query_extent,
      0,
      CHUNK_SIZE,
    )
  }

  pub fn get_brick_state(&self, local_x: i32, local_y: i32, local_z: i32, level: u8) -> BrickState {
    let query_extent = LEVEL_EXTENT[level as usize];
    if self.nodes.is_empty() {
      return brick_state_of(self.root_palette);
    }
    self.brick_state_at(
      local_x.clamp(0, CHUNK_SIZE - 1),
      local_y.clamp(0, CHUNK_SIZE - 1),
      local_z.clamp(0, CHUNK_SIZE - 1),
      query_extent,
      0,
      CHUNK_SIZE,
    )
  }

  pub fn node_desc(&self, local_x: i32, local_y: i32, local_z: i32, level: u8) -> NodeDesc {
    if self.nodes.is_empty() {
      return NodeDesc { mask: 0, palette: self.root_palette };
    }
    let (mut x, mut y, mut z) = (local_x, local_y, local_z);
    let mut idx = 0usize;
    let mut extent = CHUNK_SIZE;
    for _ in 0..level.min(3) {
      let (mask, palette) = self.node_mask_palette(idx);
      if mask == 0 {
        return NodeDesc { mask: 0, palette };
      }
      let child_extent = extent / BRICK_FACTOR;
      let (cx, cy, cz) = (
        (x / child_extent).clamp(0, BRICK_FACTOR - 1),
        (y / child_extent).clamp(0, BRICK_FACTOR - 1),
        (z / child_extent).clamp(0, BRICK_FACTOR - 1),
      );
      let cell = child_linear_idx(cx, cy, cz);
      if (mask & (1u64 << cell)) == 0 {
        return NodeDesc { mask: 0, palette };
      }
      let node = self.nodes[idx];
      idx = match node.kind {
        NodeKind::Split => self.child_at(node, cell),
        _ => unreachable!("层 3 之下不再下钻"),
      };
      x -= cx * child_extent;
      y -= cy * child_extent;
      z -= cz * child_extent;
      extent = child_extent;
    }
    let (mask, palette) = self.node_mask_palette(idx);
    NodeDesc { mask, palette }
  }

  fn brick_state_at(
    &self,
    x: i32,
    y: i32,
    z: i32,
    query_extent: i32,
    idx: usize,
    cur_extent: i32,
  ) -> BrickState {
    let (mask, palette) = self.node_mask_palette(idx);
    let node = self.nodes[idx];
    if node.kind == NodeKind::Leaf {
      let l = self.leaf(idx);
      return if query_extent >= BRICK_FACTOR {
        l.tile_color().map_or(BrickState::Mixed, brick_state_of)
      } else {
        brick_state_of(l.get(child_linear_idx(x, y, z)))
      };
    }
    if cur_extent <= query_extent {
      return self.aggregate_node_state(mask, palette, idx);
    }
    if mask == 0 {
      return brick_state_of(palette);
    }
    let child_extent = cur_extent / BRICK_FACTOR;
    let ix = (x / child_extent).clamp(0, BRICK_FACTOR - 1);
    let iy = (y / child_extent).clamp(0, BRICK_FACTOR - 1);
    let iz = (z / child_extent).clamp(0, BRICK_FACTOR - 1);
    let child_i = child_linear_idx(ix, iy, iz);
    if (mask & (1u64 << child_i)) == 0 {
      return brick_state_of(palette);
    }
    let child = self.child_at(node, child_i);
    self.brick_state_at(
      x - ix * child_extent,
      y - iy * child_extent,
      z - iz * child_extent,
      query_extent,
      child,
      child_extent,
    )
  }

  fn aggregate_node_state(&self, mask: u64, palette: PaletteId, idx: usize) -> BrickState {
    if mask == 0 {
      return brick_state_of(palette);
    }
    let node = self.nodes[idx];
    if node.kind != NodeKind::Split {
      return brick_state_of(palette);
    }
    let children = self.node_children(node);
    let mut seen: Option<BrickState> = None;
    for i in 0u32..64 {
      let bit = 1u64 << i;
      let st = if (mask & bit) != 0 {
        self.node_state(children[child_slot(mask, i) as usize] as usize)
      } else {
        brick_state_of(palette)
      };
      match (seen, st) {
        (None, s) => seen = Some(s),
        (Some(prev), s) if prev == s => {}
        (Some(_), _) => return BrickState::Mixed,
      }
    }
    seen.unwrap_or(brick_state_of(palette))
  }

  fn node_state(&self, idx: usize) -> BrickState {
    let node = self.nodes[idx];
    match node.kind {
      NodeKind::Uniform => brick_state_of(node.palette),
      NodeKind::Leaf => self.leaf(idx).tile_color().map_or(BrickState::Mixed, brick_state_of),
      NodeKind::Split => {
        if node.mask == 0 {
          return brick_state_of(node.palette);
        }
        self.aggregate_node_state(node.mask, node.palette, idx)
      }
    }
  }

  fn get_uniform_at(
    &self,
    x: i32,
    y: i32,
    z: i32,
    query_extent: i32,
    idx: usize,
    cur_extent: i32,
  ) -> Option<PaletteId> {
    let (mask, palette) = self.node_mask_palette(idx);
    let node = self.nodes[idx];
    if node.kind == NodeKind::Leaf {
      let l = self.leaf(idx);
      return if query_extent >= BRICK_FACTOR {
        l.tile_color().and_then(solid)
      } else {
        solid(l.get(child_linear_idx(x, y, z)))
      };
    }
    if cur_extent <= query_extent {
      if mask == 0 {
        return solid(palette);
      }
      let mut first_color: Option<PaletteId> = None;
      let mut all_same = true;
      let children = self.node_children(node);
      for i in 0u32..64 {
        let bit = 1u64 << i;
        let c: Option<PaletteId> = if (mask & bit) != 0 {
          self.first_uniform_color(children[child_slot(mask, i) as usize] as usize)
        } else {
          Some(palette)
        };
        {
          let c = c?;
          match first_color {
            None => first_color = Some(c),
            Some(f) if f != c => {
              all_same = false;
              break;
            }
            _ => {}
          }
        }
      }
      return if all_same { first_color } else { None };
    }

    let child_extent = cur_extent / BRICK_FACTOR;
    if mask == 0 {
      return solid(palette);
    }
    let ix = (x / child_extent).clamp(0, BRICK_FACTOR - 1);
    let iy = (y / child_extent).clamp(0, BRICK_FACTOR - 1);
    let iz = (z / child_extent).clamp(0, BRICK_FACTOR - 1);
    let child_i = child_linear_idx(ix, iy, iz);
    if (mask & (1u64 << child_i)) == 0 {
      return solid(palette);
    }
    let child = self.child_at(node, child_i);
    self.get_uniform_at(
      x - ix * child_extent,
      y - iy * child_extent,
      z - iz * child_extent,
      query_extent,
      child,
      child_extent,
    )
  }

  fn first_uniform_color(&self, idx: usize) -> Option<PaletteId> {
    let node = self.nodes[idx];
    match node.kind {
      NodeKind::Uniform => solid(node.palette),
      NodeKind::Leaf => self.leaf(idx).tile_color().and_then(solid),
      NodeKind::Split => {
        if node.mask == 0 {
          return solid(node.palette);
        }
        self.first_uniform_color(self.child_at(node, node.mask.trailing_zeros()))
      }
    }
  }

  pub fn fill_brick(&mut self, local: [i32; 3], extent: i32, palette: PaletteId) -> bool {
    assert!(
      LEVEL_EXTENT.contains(&extent),
      "extent 必须是 brick 粒度 {LEVEL_EXTENT:?} 之一（got {extent}）"
    );
    for (i, &v) in local.iter().enumerate() {
      assert!(
        v >= 0 && v % extent == 0 && v + extent <= CHUNK_SIZE,
        "brick 必须对齐且在 chunk 内（local[{i}]={v} extent={extent}）"
      );
    }
    if extent == 1 {
      return self.set_voxel(local[0], local[1], local[2], palette);
    }

    if !palette.is_air() {
      let level = level_of_extent(extent);
      if self.get_uniform(local[0], local[1], local[2], level) == Some(palette) {
        return false;
      }
    }
    self.fill_recursive(local, extent, palette, None, CHUNK_SIZE);
    true
  }

  fn fill_recursive(
    &mut self,
    x: [i32; 3],
    extent: i32,
    palette: PaletteId,
    idx: Option<usize>,
    cur_extent: i32,
  ) {
    if extent == cur_extent {
      if cur_extent == CHUNK_SIZE {
        self.nodes.clear();
        self.children.clear();
        self.leaves.clear();
        self.mark_identity_reset();
        self.root_palette = palette;
      } else {
        let i = idx.expect("非 root 层 idx 必为 Some");
        self.release_child_block(i);
        self.nodes[i] = Node::uniform(palette);
        self.mark(cur_extent, i);
      }
      return;
    }

    let mut node_idx = idx;
    let p = self.ensure_split(&mut node_idx);
    self.mark(cur_extent, p);

    let child_extent = cur_extent / BRICK_FACTOR;
    let ix = x[0] / child_extent;
    let iy = x[1] / child_extent;
    let iz = x[2] / child_extent;
    let child_i = child_linear_idx(ix, iy, iz);

    let child = self.child_or_create(p, child_i, child_extent == BRICK_FACTOR);
    let next = [x[0] - ix * child_extent, x[1] - iy * child_extent, x[2] - iz * child_extent];
    self.fill_recursive(next, extent, palette, Some(child), child_extent);

    self.try_merge(p);
  }

  pub fn set_voxel(
    &mut self,
    local_x: i32,
    local_y: i32,
    local_z: i32,
    palette: PaletteId,
  ) -> bool {
    let current = self.get_voxel(local_x, local_y, local_z);
    let want = solid(palette);
    if current == want {
      return false;
    }
    self.set_recursive(local_x, local_y, local_z, palette, None, CHUNK_SIZE);
    true
  }

  fn set_recursive(
    &mut self,
    x: i32,
    y: i32,
    z: i32,
    palette: PaletteId,
    idx: Option<usize>,
    extent: i32,
  ) {
    if extent == BRICK_FACTOR {
      let id = idx.expect("4³ 值块必已由父层创建");
      self.open_leaf(id);
      let cell = child_linear_idx(x, y, z);
      self.leaf_set(id, cell, palette);
      self.mark(BRICK_FACTOR, id);
      self.try_merge(id);
      return;
    }

    let mut node_idx = idx;
    let p = self.ensure_split(&mut node_idx);
    self.mark(extent, p);

    let child_extent = extent / BRICK_FACTOR;
    let ix = (x / child_extent).clamp(0, BRICK_FACTOR - 1);
    let iy = (y / child_extent).clamp(0, BRICK_FACTOR - 1);
    let iz = (z / child_extent).clamp(0, BRICK_FACTOR - 1);
    let child_i = child_linear_idx(ix, iy, iz);
    let next_x = x - ix * child_extent;
    let next_y = y - iy * child_extent;
    let next_z = z - iz * child_extent;

    let child = self.child_or_create(p, child_i, child_extent == BRICK_FACTOR);
    self.set_recursive(next_x, next_y, next_z, palette, Some(child), child_extent);

    self.try_merge(p);
  }

  fn split_uniform(&mut self, idx: usize, old_palette: PaletteId) {
    self.nodes[idx] = Node { mask: 0, off: 0, palette: old_palette, kind: NodeKind::Split };
  }

  fn child_or_create(&mut self, p: usize, cell: u32, leaf_child: bool) -> usize {
    let parent = self.nodes[p];
    if parent.kind != NodeKind::Split {
      unreachable!("child_or_create 只接受 Split 节点");
    }
    let bit = 1u64 << cell;
    if (parent.mask & bit) != 0 {
      return self.child_at(parent, cell);
    }
    let new = self.nodes.len() as u32;
    let kind = if leaf_child { NodeKind::Leaf } else { NodeKind::Split };
    self.nodes.push(Node { mask: 0, off: 0, palette: parent.palette, kind });
    let slot = child_slot(parent.mask | bit, cell) as usize;
    let off = self.children.insert(parent.off, parent.mask.count_ones() as usize, slot, new);
    self.nodes[p].mask |= bit;
    self.nodes[p].off = off;
    new as usize
  }

  fn ensure_split(&mut self, idx: &mut Option<usize>) -> usize {
    if let Some(i) = *idx {
      if self.nodes[i].kind == NodeKind::Uniform {
        let p = self.nodes[i].palette;
        self.split_uniform(i, p);
      }
      return i;
    }
    if !self.nodes.is_empty() {
      *idx = Some(0);
      return 0;
    }
    let cur = self.root_palette;
    let ni = self.nodes.len();
    self.nodes.push(Node { mask: 0, off: 0, palette: cur, kind: NodeKind::Split });
    *idx = Some(ni);
    ni
  }

  fn ensure_leaf(&mut self, idx: &mut Option<usize>) -> usize {
    let i = idx.expect("4³ 值块必已由父层创建");
    self.open_leaf(i);
    i
  }

  fn descend_to(&mut self, x: [i32; 3], target_extent: i32) -> (usize, NodePath) {
    debug_assert!(target_extent >= BRICK_FACTOR);
    let mut path = NodePath::new();
    let mut idx: Option<usize> = None;
    let mut cur = CHUNK_SIZE;
    let mut p = x;
    loop {
      let node = if cur == target_extent && target_extent == BRICK_FACTOR {
        self.ensure_leaf(&mut idx)
      } else {
        self.ensure_split(&mut idx)
      };
      self.mark(cur, node);
      path.push(node);
      if cur == target_extent {
        return (node, path);
      }
      let child_extent = cur / BRICK_FACTOR;
      let (ix, iy, iz) = (p[0] / child_extent, p[1] / child_extent, p[2] / child_extent);
      let child_i = child_linear_idx(ix, iy, iz);
      idx = Some(self.child_or_create(node, child_i, child_extent == BRICK_FACTOR));
      p = [p[0] - ix * child_extent, p[1] - iy * child_extent, p[2] - iz * child_extent];
      cur = child_extent;
    }
  }

  pub fn set_brick_voxels(&mut self, local: [i32; 3], inside: u64, palette: PaletteId) -> u32 {
    for (i, &v) in local.iter().enumerate() {
      assert!(
        v >= 0 && v % BRICK_FACTOR == 0 && v + BRICK_FACTOR <= CHUNK_SIZE,
        "brick 必须对齐且在 chunk 内（local[{i}]={v}）"
      );
    }
    if inside == 0 {
      return 0;
    }
    let erase = palette.is_air();
    let (node, path) = self.descend_to(local, BRICK_FACTOR);
    if self.nodes[node].kind != NodeKind::Leaf {
      unreachable!("descend_to 已把 4³ 目标归一成值块");
    }

    let mut write = 0u64;
    {
      let leaf = self.leaf(node);
      let mut m = inside;
      while m != 0 {
        let i = m.trailing_zeros();
        m &= m - 1;
        if leaf.get(i).is_air() != erase {
          write |= 1u64 << i;
        }
      }
      if write == 0 {
        return 0;
      }
    }
    let mut m = write;
    while m != 0 {
      let i = m.trailing_zeros();
      m &= m - 1;
      self.leaf_set(node, i, palette);
    }
    let changed = write.count_ones();

    if self.leaf(node).tile_color().is_some() {
      for anc in path.iter_rev() {
        self.try_merge(anc);
      }
    }
    changed
  }

  fn try_merge(&mut self, idx: usize) {
    let node = self.nodes[idx];
    match node.kind {
      NodeKind::Uniform => return,
      NodeKind::Leaf => {
        if let Some(p) = self.leaf(idx).tile_color() {
          self.nodes[idx] = Node::uniform(p);
        }
        return;
      }
      NodeKind::Split => {}
    }
    let (mask, palette) = (self.nodes[idx].mask, self.nodes[idx].palette);
    if mask == 0 {
      return;
    }
    let children = self.node_children(self.nodes[idx]);
    let mut target = if mask == u64::MAX { None } else { Some(palette) };
    let mut m = mask;
    let mut slot = 0usize;
    while m != 0 {
      m &= m - 1;
      let child = children[slot] as usize;
      slot += 1;
      let cn = self.nodes[child];
      let p = match cn.kind {
        NodeKind::Uniform => cn.palette,
        NodeKind::Leaf => match self.leaf(child).tile_color() {
          Some(p) => p,
          None => return,
        },
        NodeKind::Split => return,
      };
      match target {
        None => target = Some(p),
        Some(t) if t != p => return,
        _ => {}
      }
    }
    let merged = target.expect("mask != 0 ⇒ 至少有一个置位子块");

    if idx == 0 {
      self.nodes.clear();
      self.children.clear();
      self.leaves.clear();
      self.mark_identity_reset();
      self.root_palette = merged;
    } else {
      self.release_child_block(idx);
      self.nodes[idx] = Node::uniform(merged);
    }
  }

  pub fn clear_voxel(&mut self, local_x: i32, local_y: i32, local_z: i32) -> bool {
    self.set_voxel(local_x, local_y, local_z, PaletteId::AIR)
  }

  pub fn proxy(&self, keep_extent: i32) -> Self {
    let keep = keep_extent.max(BRICK_FACTOR);
    let mut out = Self::empty();
    match self.nodes.first() {
      None => out.root_palette = self.root_palette,
      Some(n) if n.kind == NodeKind::Uniform => out.root_palette = n.palette,
      Some(_) if CHUNK_SIZE <= keep => out.root_palette = self.rep_of(0),
      Some(_) => {
        let root = self.copy_proxy(0, CHUNK_SIZE, keep, &mut out);
        debug_assert_eq!(root, 0, "根必须落在 nodes[0]");
      }
    }
    out
  }

  fn copy_proxy(&self, id: usize, extent: i32, keep: i32, out: &mut ChunkTree) -> usize {
    let my = out.nodes.len();
    if extent <= keep {
      out.nodes.push(Node::uniform(self.rep_of(id)));
      return my;
    }
    let node = self.nodes[id];
    if node.kind != NodeKind::Split {
      if node.kind != NodeKind::Uniform {
        unreachable!("节点只有三种形态");
      }
      out.nodes.push(Node::uniform(node.palette));
      return my;
    }
    out.nodes.push(Node::uniform(PaletteId::AIR));
    let child_extent = extent / BRICK_FACTOR;
    let count = node.mask.count_ones() as usize;
    let src = self.children.slice(node.off, count);
    let dst = out.children.alloc(count);
    for (slot, &c) in src.iter().enumerate() {
      let mapped = self.copy_proxy(c as usize, child_extent, keep, out) as u32;
      out.children.slice_mut(dst, count)[slot] = mapped;
    }
    out.nodes[my] =
      Node { mask: node.mask, off: dst, palette: node.palette, kind: NodeKind::Split };
    my
  }

  fn rep_of(&self, id: usize) -> PaletteId {
    let node = self.nodes[id];
    match node.kind {
      NodeKind::Uniform => node.palette,
      NodeKind::Leaf => self.leaf(id).rep(),
      NodeKind::Split => {
        if node.mask != u64::MAX && !node.palette.is_air() {
          return node.palette;
        }
        for &c in self.node_children(node) {
          let r = self.rep_of(c as usize);
          if !r.is_air() {
            return r;
          }
        }
        PaletteId::AIR
      }
    }
  }

  pub fn is_empty(&self) -> bool {
    self.nodes.is_empty() && self.root_palette.is_air()
  }

  pub fn compact(&mut self) {
    if self.nodes.len() <= 1 {
      return;
    }
    self.mark_identity_reset();
    let mut idx_map = vec![u32::MAX; self.nodes.len()];
    let mut order: Vec<usize> = Vec::with_capacity(self.nodes.len());
    let mut stack: Vec<usize> = vec![0];
    while let Some(old) = stack.pop() {
      if idx_map[old] != u32::MAX {
        continue;
      }
      idx_map[old] = order.len() as u32;
      order.push(old);
      let node = self.nodes[old];
      if node.kind == NodeKind::Split {
        for &c in self.children.slice(node.off, node.mask.count_ones() as usize) {
          stack.push(c as usize);
        }
      }
    }

    let mut new_nodes: Vec<Node> = Vec::with_capacity(order.len());
    let mut new_children = ChildPool::default();
    let mut new_leaves: Vec<[u32; LEAF_INLINE_WORDS]> = Vec::with_capacity(self.leaves.len());
    for &old in &order {
      let mut node = self.nodes[old];
      match node.kind {
        NodeKind::Split => {
          let count = node.mask.count_ones() as usize;
          let src = self.children.slice(node.off, count);
          let dst = new_children.alloc(count);
          for slot in 0..count {
            new_children.slice_mut(dst, count)[slot] = idx_map[src[slot] as usize];
          }
          node.off = dst;
        }
        NodeKind::Leaf => {
          if node.mask == 0 {
            node.off = 0;
          } else {
            new_leaves.push(self.leaves[node.off as usize]);
            node.off = (new_leaves.len() - 1) as u32;
          }
        }
        NodeKind::Uniform => {}
      }
      new_nodes.push(node);
    }
    self.nodes = new_nodes;
    self.children = new_children;
    self.leaves = new_leaves;
  }

  pub fn node_count(&self) -> usize {
    self.nodes.len()
  }
}
