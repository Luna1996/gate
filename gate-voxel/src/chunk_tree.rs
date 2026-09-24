//! ChunkTree：Douglas Brick Tree（分裂因子 4³=64，u64 occupancy mask per non-leaf，
//! 紧凑 child offset，uniform tile 自适应）。
//!
//! 形态照 VDB / GVDB（Museth 2013；Hoetzlein 2016）的"值块与拓扑分离、按层定长"：
//! - 层 0-2 = **拓扑节点**（`Node::Split`：子块掩码 + tile 色 + 紧凑 child 表）；
//! - 层 3 = **定长密集值块**（[`Node::Leaf`]：64 格 inline 值表 + 活跃掩码）。这一层**不是**
//!   "每个 1³ 体素一个节点"—— 那样一次 4³ 批量写要付 64 次节点插写与分配（实测占球笔触 CPU 的 59%），
//!   改成值块后同一笔只剩一次 ~140B 写，且与 wire 的层 3 逐位同构。
//!
//! 序列化层把树摊成 `Vec<u32>`（wire）：根节点恒占固定字数（见 [`ROOT_WIRE_WORDS`]），其余节点地址任意
//! —— wire 的子块指针以**根地址**为基准，故增量编辑可以只重写动过的那几个节点（节点级改动走
//! [`ChunkTree::take_dirty`]，消费方是 `gate-render` 的树块 arena）。
//! Level 链 256 → 64 → 16 → 4 → 1；材质索引是 `PaletteId`（16 位，容量 2^16），节点 tile 色与叶层逐体素色
//! 同宽，二者都从同一 u32 字段解包（见 `pack_node_palette`）。

use super::coords::{BRICK_FACTOR, CHUNK_SIZE, LEVEL_EXTENT, child_linear_idx};
use crate::palette::{PALETTE_BITS, PaletteId};

/// 叶父层每 u32 字装的体素数（= 32 位 / 材质索引位宽）
pub const LEAF_VOXELS_PER_WORD: usize = 32 / PALETTE_BITS as usize;
/// 叶父层（level 3）inline 字数：4³ = 64 体素，每字装 [`LEAF_VOXELS_PER_WORD`] 个
pub const LEAF_INLINE_WORDS: usize = 64 / LEAF_VOXELS_PER_WORD;

/// 非叶节点 fixed 字数（mask_lo / mask_hi / palette）
const NODE_FIXED_WORDS: usize = 3;
/// 根节点固定 wire 字数 = fixed 3 字 + 64 槽子块指针表。
///
/// CONSTRAINT: 根地址必须恒定 —— wire 的子块指针以**根地址**为基准（shader 的 `chunk_base`），
/// 根一搬迁全体指针都要重算。预留满 64 槽 ⇒ 掩码增减不再改变根的字数，根永不移动。
pub const ROOT_WIRE_WORDS: usize = NODE_FIXED_WORDS + 64;

/// 逐节点 layout 里"该 id 不参与 wire"的哨兵
pub const NODE_OFFSET_NONE: u32 = u32::MAX;
/// 两次 [`ChunkTree::take_dirty`] 之间累计的节点改动上限（超过即改用整棵重建，见 `ChunkTree::mark`）。
/// 定在 4096：一笔 size≈17 的球（≈1.6 万体素 / 250 个 4³ 值块）标记数约 1000 ⇒ 仍走节点级路径；
/// 再大的批量填充才退回整棵重建（那次重建的量级固定，而逐节点表会涨到几十万项）。
const DIRTY_NODE_CAP: usize = 4096;
/// 逐节点 wire 布局（索引 = 节点 id，值 = (`blob` 内首字偏移, wire 层 0..=3)）
pub type NodeLayout = Vec<(u32, u8)>;

/// wire 节点视图（`gate-render` 做节点级增量寻址用；子节点 id 是不透明身份）
#[derive(Debug, Clone, Copy)]
pub struct NodeView<'a> {
  /// 分裂掩码：bit=1 的子块有独立节点，bit=0 的子块 = `palette`。层 3 = 活跃掩码
  /// （bit=1 的体素色在 inline 值表里，bit=0 的 = `palette`）
  pub mask: u64,
  /// tile 色 / 值块的默认色（0 = 空气）
  pub palette: PaletteId,
  /// 紧凑子节点表（按 mask 位序）；层 3 的值块没有子节点（内容走 [`ChunkTree::node_inline_words`]）
  pub children: &'a [u32],
}

/// 自上次 [`ChunkTree::take_dirty`] 以来的**节点级**改动（供 wire 层只重写动过的节点）
#[derive(Debug, Default, Clone)]
pub struct TreeDirty {
  /// true = 树的节点身份空间被整体换过（根被合并清空 / [`ChunkTree::compact`] / 外部整体替换）
  /// ⇒ 调用方必须先丢弃该 chunk 的全部节点映射再全量重建。
  pub reset: bool,
  /// 被写过的 wire 节点 `(wire 层 0..=3, 节点 id)`，已按层降序（自底向上：子地址先定，
  /// 父节点的指针表才写得对）去重。可能**多报**（路径上的节点即使字节没变也在表里）——
  /// 消费方按"编码后与原字节比较"决定是否真写。
  pub nodes: Vec<(u8, u32)>,
}

/// 4³ 值块（wire 层 3）：值表与活跃掩码分离的**定长**叶子 —— 形态同 VDB 的 `LeafNode`
///（`mLeafDAT` 值表 + `mValueMask` 活跃掩码）/ GVDB 的 brick。
///
/// `inline` 的位布局与 wire 逐位相同（每字 2 体素 × 16 位索引）；bit=0 的体素不入表（其色 = `palette`）。
/// 定长 ⇒ 写入不分配、不搬动、不进节点表；一次 4³ 批量写 = 一次 ~140B 写。
#[derive(Debug, Clone)]
struct Leaf {
  /// 活跃掩码：bit=1 的体素色在 `inline` 里，bit=0 = `palette`
  mask: u64,
  /// 未置位体素的色（= 该 4³ brick 的 tile 色）
  palette: PaletteId,
  /// 每字 2 体素 × 16 位索引（`i` 的槽 = `inline[i >> 1]` 的 `(i & 1)` 半字）
  inline: [u32; LEAF_INLINE_WORDS],
}

impl Leaf {
  /// 等值值块（掩码 0 ⇒ 整 brick 同色，与 wire 的 3 字形态等价）
  fn uniform(palette: PaletteId) -> Self {
    Self { mask: 0, palette, inline: [0; LEAF_INLINE_WORDS] }
  }

  /// 第 `i` 格的色（`i = z*16 + y*4 + x`）
  #[inline]
  fn get(&self, i: u32) -> PaletteId {
    if (self.mask & (1u64 << i)) == 0 {
      return self.palette;
    }
    let w = self.inline[(i >> 1) as usize];
    PaletteId(((w >> ((i & 1) * 16)) & 0xFFFF) as u16)
  }

  /// 写第 `i` 格并置活跃位
  #[inline]
  fn set(&mut self, i: u32, palette: PaletteId) {
    let slot = (i >> 1) as usize;
    let shift = (i & 1) * 16;
    self.inline[slot] =
      (self.inline[slot] & !(0xFFFF << shift)) | ((palette.get() as u32) << shift);
    self.mask |= 1u64 << i;
  }

  /// 全 64 格同色 → Some(色)。
  /// 有未置位格 ⇒ 目标色 = `palette`（未置位格的颜色来源），置位格必须都等于它；
  /// 掩码满（没有未置位格）⇒ 取第一个置位格的色 —— 与拓扑节点的同款特例，漏了它会把实心砖报成 Mixed。
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

/// 根 → 目标节点的路径（[`ChunkTree::descend_to`] 的返回）：定长数组，不分配 ——
/// 4³ 批量写是笔触热路径，每次下钻都 `Vec::with_capacity` 是一笔可观固定开销。
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

  /// 自底向上（目标节点 → 根）
  fn iter_rev(&self) -> impl Iterator<Item = usize> + '_ {
    self.ids[..self.len].iter().rev().copied()
  }
}

/// 结构化节点：层 0-2 是拓扑节点，层 3 是 [`Leaf`] 值块（用 [`Node::Uniform`] 表示等值 brick）。
#[derive(Debug, Clone)]
enum Node {
  /// 整 brick（任意层）同一 palette —— 即 VDB 的 "tile"
  Uniform(PaletteId),
  /// 拓扑节点（层 0-2）：mask bit=1 的子块有独立节点，bit=0 = uniform（色 = `palette`）
  Split {
    mask: u64,
    /// uniform 子块的默认 palette
    palette: PaletteId,
    /// 紧凑 child 表（按 mask 位序）：children[j] = 第 j 个 bit=1 子块的 nodes 下标，
    /// j = popcount(mask & (bit_i - 1))，len == popcount(mask)。
    children: Vec<u32>,
  },
  /// 4³ 值块（层 3）
  Leaf(Leaf),
}

/// mask 中子块位序 i → 紧凑表下标（GPU DDA 同款 popcount 定位，O(1)）
#[inline]
fn child_slot(mask: u64, i: u32) -> u32 {
  debug_assert!(i < 64 && (mask & (1u64 << i)) != 0, "child_slot 要求 bit=1");
  (mask & ((1u64 << i) - 1)).count_ones()
}

/// 节点 palette word 打包：bit0..15 = tile 色；bit16..31 恒 0。
///
/// CONSTRAINT: 高 16 位（wire 里的旧字段「LOD 子树多数色」，配合 `consts::DDA_LOD` 的远场早停）
/// **没有消费方** —— shader 侧所有节点读取点都 `& 0xFFFF` 屏蔽高半字，`view_u.lod.y` 也没有读取点。
/// 它的逐节点递归 + 64² 计票曾占序列化 88% 的耗时，故整段移除。
#[inline]
fn pack_node_palette(palette: PaletteId) -> u32 {
  palette.get() as u32
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BrickState {
  /// 整 brick 空气（palette 0）
  Air,
  /// 整 brick 同一非零 palette
  Solid(PaletteId),
  /// 空气与实体混合（或多种颜色混合）
  Mixed,
}

/// brick 节点描述：64bit 子块分裂掩码 + tile 色。
/// 语义 = GPU `b_struct` 节点前两字段（`mask_lo/mask_hi` + palette 低 16 位）：
/// `mask` bit=1 的子块有独立节点，bit=0 的子块与该节点同色 = `palette`。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NodeDesc {
  pub mask: u64,
  pub palette: PaletteId,
}

/// palette → brick 三态（palette 0 = AIR）
#[inline]
fn brick_state_of(palette: PaletteId) -> BrickState {
  if palette.is_air() { BrickState::Air } else { BrickState::Solid(palette) }
}

/// palette → Option（空气 → None），查询路径的公共归一
#[inline]
fn solid(palette: PaletteId) -> Option<PaletteId> {
  if palette.is_air() { None } else { Some(palette) }
}

/// brick 边长 → 层级（`LEVEL_EXTENT` 的下标）；非 brick 粒度即 panic。
#[inline]
pub fn level_of_extent(extent: i32) -> u8 {
  LEVEL_EXTENT
    .iter()
    .position(|&e| e == extent)
    .unwrap_or_else(|| panic!("extent 必须是 brick 粒度 {LEVEL_EXTENT:?} 之一（got {extent}）"))
    as u8
}

/// Douglas Brick Tree 1:1（分裂树，4³=64 分裂因子，u64 mask）
#[derive(Debug, Clone)]
pub struct ChunkTree {
  nodes: Vec<Node>,
  root_palette: PaletteId,
  /// 节点身份空间被整体换过（根被合并清空 / [`Self::compact`]）
  identity_reset: bool,
  /// 自上次 [`Self::take_dirty`] 以来被写过的 wire 节点 `(层, id)`（可能含重复）
  dirty_nodes: Vec<(u8, u32)>,
}

impl ChunkTree {
  /// 空 chunk：root uniform AIR（palette=0）
  pub fn empty() -> Self {
    Self {
      nodes: Vec::new(),
      root_palette: PaletteId::AIR,
      identity_reset: false,
      dirty_nodes: Vec::new(),
    }
  }

  /// 全 uniform palette chunk
  pub fn uniform(palette: PaletteId) -> Self {
    Self { root_palette: palette, ..Self::empty() }
  }

  /// 标一个 wire 节点被写过（`extent` = 该节点的 brick 边长；wire 只有层 0..3）
  ///
  /// 累计超过 [`DIRTY_NODE_CAP`] 就改成"身份空间作废"（整棵重建）：一次批量填充会逐格标路径，
  /// 无上限时这张表会涨到几十万项（排序与逐节点编码都是白烧），而整棵重建的量级反而固定。
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

  /// 整棵树的节点身份空间作废（调用方须丢弃全部节点映射）：根被合并清空 / [`Self::compact`]。
  #[inline]
  fn mark_identity_reset(&mut self) {
    self.identity_reset = true;
  }

  /// 取走自上次调用以来的节点级改动（见 [`TreeDirty`]）。
  pub fn take_dirty(&mut self) -> TreeDirty {
    let reset = std::mem::take(&mut self.identity_reset);
    let mut nodes = std::mem::take(&mut self.dirty_nodes);
    if reset {
      nodes.clear();
    } else {
      // 层降序 = 自底向上（子地址先定，父节点的指针表才写得对）；再按 id 排序使重复项相邻
      nodes.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
      nodes.dedup();
    }
    TreeDirty { reset, nodes }
  }

  /// 外部整体替换该 chunk 的树（`VolumeGrid::mount_chunk_tree`）后调用：让消费方全量重建。
  pub fn mark_replaced(&mut self) {
    self.mark_identity_reset();
  }

  /// 节点 id 空间上限（消费方的节点槽表按它扩容）
  pub fn node_capacity(&self) -> usize {
    self.nodes.len()
  }

  pub fn root_palette(&self) -> PaletteId {
    self.root_palette
  }

  /// 树里没有节点 ⇒ 整 chunk 是 `root_palette` 单色（含 AIR）
  pub fn is_uniform_root(&self) -> bool {
    self.nodes.is_empty()
  }

  /// wire 节点视图；`id` 越界（已被合并成 uniform 后回收）→ None
  pub fn node_view(&self, id: u32) -> Option<NodeView<'_>> {
    let (mask, palette, children): (u64, PaletteId, &[u32]) = match self.nodes.get(id as usize)? {
      Node::Uniform(p) => (0, *p, &[]),
      Node::Leaf(l) => (l.mask, l.palette, &[]),
      Node::Split { mask, palette, children } => (*mask, *palette, children),
    };
    Some(NodeView { mask, palette, children })
  }

  /// wire 层 3（4³ 值块）节点的 32 字 inline 值表。非值块节点（含越界）→ None。
  pub fn node_inline_words(&self, id: u32) -> Option<[u32; LEAF_INLINE_WORDS]> {
    match self.nodes.get(id as usize)? {
      Node::Leaf(l) => Some(l.inline),
      _ => None,
    }
  }

  /// 节点的 `(掩码, tile 色)` —— 查询与写入路径的公共取值（值块把活跃掩码当掩码看）
  #[inline]
  fn node_mask_palette(&self, idx: usize) -> (u64, PaletteId) {
    match &self.nodes[idx] {
      Node::Uniform(p) => (0, *p),
      Node::Leaf(l) => (l.mask, l.palette),
      Node::Split { mask, palette, .. } => (*mask, *palette),
    }
  }

  /// DFS 紧凑序列化（上传 GPU struct buffer）
  pub fn serialize(&self) -> Vec<u32> {
    self.serialize_with_layout().0
  }

  /// 序列化 + 逐节点字偏移（索引 = 节点 id；见 [`NodeLayout`]）。
  ///
  /// 根节点固定占 [`ROOT_WIRE_WORDS`] 字（3 fixed + 满 64 槽指针表），其余节点从其后紧排 ——
  /// 根地址恒等于 `blob` 首字，故 wire 里所有子块指针（= 相对根的字偏移）在增量重写中始终有效。
  pub fn serialize_with_layout(&self) -> (Vec<u32>, NodeLayout) {
    // 预估容量：每节点 3 字 + 指针/inline。不做精确预估，只为省掉增长期的反复重分配+拷贝。
    let mut out = Vec::with_capacity(self.nodes.len() * 3 + 4096);
    let mut layout: NodeLayout = vec![(NODE_OFFSET_NONE, 0); self.nodes.len().max(1)];
    let (mask, palette) = match self.nodes.first() {
      None => (0u64, self.root_palette),
      Some(n) => match n {
        Node::Uniform(p) => (0u64, *p),
        Node::Leaf(l) => (l.mask, l.palette),
        Node::Split { mask, palette, .. } => (*mask, *palette),
      },
    };
    out.push(mask as u32);
    out.push((mask >> 32) as u32);
    out.push(pack_node_palette(palette));
    layout[0] = (0, 0);
    // 根**恒**占满预留区（即便 mask=0 只用到前 3 字）：根的字数不随掩码变化 ⇒ 永不搬迁。
    out.resize(ROOT_WIRE_WORDS, 0);
    match self.nodes.first() {
      None | Some(Node::Uniform(_)) => {}
      Some(Node::Leaf(l)) => out.extend_from_slice(&l.inline),
      Some(Node::Split { children, .. }) => {
        let child_extent = CHUNK_SIZE / BRICK_FACTOR;
        for (slot, &child) in children.iter().enumerate() {
          out[NODE_FIXED_WORDS + slot] = out.len() as u32;
          self.serialize_node(child as usize, child_extent, &mut out, &mut layout);
        }
      }
    }
    (out, layout)
  }

  /// DFS 序列化一个节点（层 0-2 紧凑指针表 / 层 3 值块 inline）
  fn serialize_node(&self, idx: usize, extent: i32, out: &mut Vec<u32>, layout: &mut NodeLayout) {
    layout[idx] = (out.len() as u32, level_of_extent(extent));
    let (mask, palette, children): (u64, PaletteId, &[u32]) = match &self.nodes[idx] {
      Node::Uniform(p) => (0, *p, &[]),
      Node::Leaf(l) => (l.mask, l.palette, &[]),
      Node::Split { mask, palette, children } => (*mask, *palette, children),
    };
    // 写 3 words: mask_lo, mask_hi, palette(bit0..15，高 16 位恒 0 —— 见 `pack_node_palette`)
    out.push(mask as u32);
    out.push((mask >> 32) as u32);
    out.push(pack_node_palette(palette));
    if mask == 0 {
      return;
    }
    if let Node::Leaf(l) = &self.nodes[idx] {
      // 值块：3 字 + 32 字 inline（与 CPU 侧逐位同构，序列化 = 一次拷贝）
      out.extend_from_slice(&l.inline);
      return;
    }
    let child_extent = extent / BRICK_FACTOR;
    // 子块按**置位**遍历（不是 0..64 扫全 64 位）：密集 chunk 的节点通常只有 ≤8 个置位，
    // 这里每节点省下 ~60 次空转；紧凑表下标 = 第几个置位 ⇒ 顺带用计数器代替 popcount。
    let num_children = mask.count_ones() as usize;
    let offsets_start = out.len();
    out.resize(out.len() + num_children, 0);
    let mut m = mask;
    let mut slot = 0usize;
    while m != 0 {
      m &= m - 1;
      // 子块地址 = 写完指针表后的当前位置（DFS 紧排 ⇒ 只有走到才可知）
      out[offsets_start + slot] = out.len() as u32;
      let child_idx = children[slot] as usize;
      slot += 1;
      self.serialize_node(child_idx, child_extent, out, layout);
    }
  }

  pub fn len_words(&self) -> usize {
    self.serialize().len()
  }

  /// 序列化 buffer（上传 GPU 用）
  pub fn nodes(&self) -> Vec<u32> {
    self.serialize()
  }

  /// 查询指定体素（1³，level 4）的 palette；未设置返回 None
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

  /// 下钻取值：`(x,y,z)` 为当前节点内坐标，`extent` = 当前节点边长
  fn get_at(&self, x: i32, y: i32, z: i32, idx: usize, extent: i32) -> Option<PaletteId> {
    match &self.nodes[idx] {
      Node::Uniform(p) => solid(*p),
      Node::Leaf(l) => solid(l.get(child_linear_idx(x, y, z))),
      Node::Split { mask, palette, children } => {
        if *mask == 0 {
          return solid(*palette);
        }
        let child_extent = extent / BRICK_FACTOR;
        let (ix, iy, iz) = (
          (x / child_extent).clamp(0, BRICK_FACTOR - 1),
          (y / child_extent).clamp(0, BRICK_FACTOR - 1),
          (z / child_extent).clamp(0, BRICK_FACTOR - 1),
        );
        let cell = child_linear_idx(ix, iy, iz);
        if (*mask & (1u64 << cell)) == 0 {
          return solid(*palette);
        }
        let child = children[child_slot(*mask, cell) as usize] as usize;
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

  /// uniform 查询：level L 的 brick 是否全同色
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

  /// 体素 `(x,y,z)` 所在的 `LEVEL_EXTENT[level]` 粒度 brick 的节点描述（见 [`NodeDesc`]）：
  /// level 0 = 256³（chunk 根，子块 64³）… 3 = 4³（值块，子块 = 1³ 体素，即 DDA 的最内层）。
  /// 更粗的祖先已是 uniform / 该子块在更粗层就 uniform → `mask = 0` 且 `palette` = 那一层的色。
  /// 遍历侧按「节点掩码常驻」用：同一节点内跨 4³ 子块步进不必重查。
  pub fn node_desc(&self, local_x: i32, local_y: i32, local_z: i32, level: u8) -> NodeDesc {
    if self.nodes.is_empty() {
      return NodeDesc { mask: 0, palette: self.root_palette };
    }
    // 下钻全程用**节点内**坐标（与 `get_at` 同一手法）：每层减去该 4³ 子块的偏移
    let (mut x, mut y, mut z) = (local_x, local_y, local_z);
    let mut idx = 0usize;
    let mut extent = CHUNK_SIZE;
    // 从根下钻到目标层：LEVEL_EXTENT 下标 0(256³) → level
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
      idx = match &self.nodes[idx] {
        Node::Split { children, .. } => children[child_slot(mask, cell) as usize] as usize,
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
    if let Node::Leaf(l) = &self.nodes[idx] {
      // 值块：查询粒度 = 整 brick ⇒ 三态；= 单格 ⇒ 该格自身三态
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
    let child = match &self.nodes[idx] {
      Node::Split { children, .. } => children[child_slot(mask, child_i) as usize] as usize,
      _ => unreachable!(),
    };
    self.brick_state_at(
      x - ix * child_extent,
      y - iy * child_extent,
      z - iz * child_extent,
      query_extent,
      child,
      child_extent,
    )
  }

  /// 聚合拓扑节点完整区域的 64 子块 → 三态；任一子块与已见态冲突 → Mixed。
  fn aggregate_node_state(&self, mask: u64, palette: PaletteId, idx: usize) -> BrickState {
    if mask == 0 {
      return brick_state_of(palette);
    }
    let children: &[u32] = match &self.nodes[idx] {
      Node::Split { children, .. } => children.as_slice(),
      _ => return brick_state_of(palette),
    };
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

  /// 单节点完整区域三态（与 `aggregate_node_state` 互递归，Mixed 早退）。
  fn node_state(&self, idx: usize) -> BrickState {
    match &self.nodes[idx] {
      Node::Uniform(p) => brick_state_of(*p),
      Node::Leaf(l) => l.tile_color().map_or(BrickState::Mixed, brick_state_of),
      Node::Split { mask, palette, .. } => {
        if *mask == 0 {
          return brick_state_of(*palette);
        }
        self.aggregate_node_state(*mask, *palette, idx)
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
    if let Node::Leaf(l) = &self.nodes[idx] {
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
      let children = match &self.nodes[idx] {
        Node::Split { children, .. } => children.as_slice(),
        _ => unreachable!(),
      };
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
    let child = match &self.nodes[idx] {
      Node::Split { children, .. } => children[child_slot(mask, child_i) as usize] as usize,
      _ => unreachable!(),
    };
    self.get_uniform_at(
      x - ix * child_extent,
      y - iy * child_extent,
      z - iz * child_extent,
      query_extent,
      child,
      child_extent,
    )
  }

  /// 子树的首个 uniform 色：全子树同色 → Some，否则 None（值块 / 空 mask 之外都往下取第一个置位子块）
  fn first_uniform_color(&self, idx: usize) -> Option<PaletteId> {
    match &self.nodes[idx] {
      Node::Uniform(p) => solid(*p),
      Node::Leaf(l) => l.tile_color().and_then(solid),
      Node::Split { mask, children, palette, .. } => {
        if *mask == 0 {
          return solid(*palette);
        }
        let i = mask.trailing_zeros();
        self.first_uniform_color(children[child_slot(*mask, i) as usize] as usize)
      }
    }
  }

  /// 填充对齐 brick（extent ∈ `LEVEL_EXTENT`），一次调用只沿路径创建 ≤depth 个节点（lazy split）。
  /// 返回是否实际修改。`extent = 1` 走单格写（值块里的一格）。
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
        // 整 chunk uniform：规范形 = 空 nodes + root_palette。
        self.nodes.clear();
        self.mark_identity_reset();
        self.root_palette = palette;
      } else {
        let i = idx.expect("非 root 层 idx 必为 Some");
        // 4³ 目标也归一成 uniform（wire 3 字，与等值值块等价，且可被父层 merge）
        self.nodes[i] = Node::Uniform(palette);
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

  /// 设置单个 1³ 体素的 palette（palette=0 = 清除）；返回是否实际修改。
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
      // 4³ 值块：一次半字写 + 置活跃位（零分配）
      let id = idx.expect("4³ 值块必已由父层创建");
      let cell = child_linear_idx(x, y, z);
      if let Node::Uniform(p) = self.nodes[id] {
        self.nodes[id] = Node::Leaf(Leaf::uniform(p));
      }
      match &mut self.nodes[id] {
        Node::Leaf(l) => l.set(cell, palette),
        _ => unreachable!("层 3 恒为值块"),
      }
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

  /// 把 uniform 节点变成分裂节点（lazy：mask=0 全 uniform，子节点按需创建）。
  fn split_uniform(&mut self, idx: usize, old_palette: PaletteId) {
    self.nodes[idx] = Node::Split { mask: 0, palette: old_palette, children: Vec::new() };
  }

  /// 取节点 `p` 的第 `cell` 个子节点；没有就按 `p` 的 tile 色补一个（lazy）。
  /// `leaf_child` = 子节点该建 4³ 值块（父层为 16³）还是分裂节点（父层 ≥ 64³）。
  /// CONSTRAINT: `p` 必须是 Split 节点（`None` 表示"整块同色"，调用方须先 [`Self::ensure_split`]）。
  fn child_or_create(&mut self, p: usize, cell: u32, leaf_child: bool) -> usize {
    let (mask, palette) = match &self.nodes[p] {
      Node::Split { mask, palette, .. } => (*mask, *palette),
      _ => unreachable!("child_or_create 只接受 Split 节点"),
    };
    let bit = 1u64 << cell;
    if (mask & bit) != 0 {
      let children = match &self.nodes[p] {
        Node::Split { children, .. } => children,
        _ => unreachable!(),
      };
      return children[child_slot(mask, cell) as usize] as usize;
    }
    let new = self.nodes.len() as u32;
    let child = if leaf_child {
      Node::Leaf(Leaf::uniform(palette))
    } else {
      Node::Split { mask: 0, palette, children: Vec::new() }
    };
    self.nodes.push(child);
    if let Node::Split { mask, children, .. } = &mut self.nodes[p] {
      let slot = child_slot(*mask | bit, cell) as usize;
      children.insert(slot, new);
      *mask |= bit;
    }
    new as usize
  }

  /// 把 `idx` 归一成"可以挂子节点"的 Split 节点下标：
  /// `Some(i)` 且是 Uniform ⇒ lazy split；`None` 且树非空 ⇒ 根（下标 0，恒为 Split）；
  /// `None` 且树空 ⇒ 新建根。返回值即归一后的下标。
  fn ensure_split(&mut self, idx: &mut Option<usize>) -> usize {
    if let Some(i) = *idx {
      if let Node::Uniform(p) = &self.nodes[i] {
        let p = *p;
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
    self.nodes.push(Node::Split { mask: 0, palette: cur, children: Vec::new() });
    *idx = Some(ni);
    ni
  }

  /// 把 `idx` 归一成 4³ 值块（层 3）：`Uniform(p)` ⇒ 就地展开成等值值块（lazy，掩码 0）。
  fn ensure_leaf(&mut self, idx: &mut Option<usize>) -> usize {
    let i = idx.expect("4³ 值块必已由父层创建");
    if let Node::Uniform(p) = self.nodes[i] {
      self.nodes[i] = Node::Leaf(Leaf::uniform(p));
    }
    i
  }

  /// 沿路径下钻到 `target_extent`（∈ `LEVEL_EXTENT`）的节点：lazy 补齐中间层，
  /// 返回 `(该节点下标, 路径[根..=该节点])` —— 路径用来自底向上 `try_merge`。
  /// `target_extent` 允许是 4（4³ 值块）；1 由调用方归到 4 处理。
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

  /// 按 **4³ brick** 批量写体素：一次下钻 + 一次值块写 + 沿路径一次向上合并。
  ///
  /// `local` = brick 最小角（对齐 [`BRICK_FACTOR`]）；`inside` 的 bit `i = z*16 + y*4 + x`
  /// （x→y→z 密集序）标出"落在笔触形状内"的体素。写入口径与逐格 [`Self::set_voxel`] 的调用方
  /// **逐格过滤**完全一致：`palette` 非空 ⇒ 只填空气格；`palette` 为 AIR ⇒ 只挖实体格
  /// （`inside` 之外的格子保持原值）。返回实际改变的体素数。
  ///
  /// 代价与"笔触覆盖的体素数"无关：值块是定长密集表，写入只是 64 位内的一次掩码/半字更新 + 一次
  /// ~140B 写，无分配、无节点增删（逐格写要付 64 次下钻 + 64 次节点插写）。
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
    let leaf = match &mut self.nodes[node] {
      Node::Leaf(l) => l,
      _ => unreachable!("descend_to 已把 4³ 目标归一成值块"),
    };

    // 一遍扫出"该写哪些位"：只读值块当前色（口径同逐格 set_voxel 的调用方过滤）
    let mut write = 0u64;
    let mut m = inside;
    while m != 0 {
      let i = m.trailing_zeros();
      m &= m - 1;
      if leaf.get(i).is_air() != erase {
        write |= 1u64 << i;
      }
    }
    if write == 0 {
      return 0; // 往空气里放 / 挖空气：无改动（与逐格口径一致）
    }
    let mut m = write;
    while m != 0 {
      let i = m.trailing_zeros();
      m &= m - 1;
      leaf.set(i, palette);
    }
    let changed = write.count_ones();

    // 自底向上合并：**只在值块本身变成等值砖时才需要** —— 树守恒（可达节点都已规范化），
    // 本次写只可能让"这条路径"上的节点变 uniform；值块非等值时祖先必然也合不了 ⇒ 整条路径省掉。
    // 逐格写走 `set_recursive`，每条路径本来就只有一层，照旧每层试一次。
    if leaf.tile_color().is_some() {
      for anc in path.iter_rev() {
        self.try_merge(anc);
      }
    }
    changed
  }

  /// 尝试把一个节点合回 uniform（**所有**子块同色时成立）：
  /// - 4³ 值块 ⇒ 看活跃位上的色是否都等于 tile 色（VDB 的 "leaf 变 tile"）；
  /// - 拓扑节点 ⇒ 只遍历**置位**（不是扫全 64 位）：未置位部分的色 = 本节点 palette，故"全同色"⟺
  ///   每个置位子节点都是 `Uniform(palette)`；掩码满（没有未置位部分）时才退化成"置位子节点彼此同色"。
  fn try_merge(&mut self, idx: usize) {
    match &self.nodes[idx] {
      Node::Uniform(_) => return,
      Node::Leaf(l) => {
        if let Some(p) = l.tile_color() {
          self.nodes[idx] = Node::Uniform(p);
        }
        return;
      }
      Node::Split { .. } => {}
    }
    let (mask, palette, children) = match &self.nodes[idx] {
      Node::Split { mask, palette, children } => (*mask, *palette, children.as_slice()),
      _ => unreachable!(),
    };
    if mask == 0 {
      return;
    }
    // 目标色：有未置位位 ⇒ palette；掩码满 ⇒ 取第一个置位子块色
    let mut target = if mask == u64::MAX { None } else { Some(palette) };
    let mut m = mask;
    let mut slot = 0usize;
    while m != 0 {
      m &= m - 1;
      let child = children[slot] as usize;
      slot += 1;
      let p = match &self.nodes[child] {
        Node::Uniform(p) => *p,
        Node::Leaf(l) => match l.tile_color() {
          Some(p) => p,
          None => return,
        },
        Node::Split { .. } => return,
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
      self.mark_identity_reset();
      self.root_palette = merged;
    } else {
      self.nodes[idx] = Node::Uniform(merged);
    }
  }

  pub fn clear_voxel(&mut self, local_x: i32, local_y: i32, local_z: i32) -> bool {
    self.set_voxel(local_x, local_y, local_z, PaletteId::AIR)
  }

  /// 整个 chunk 是否 uniform AIR（空）
  pub fn is_empty(&self) -> bool {
    self.nodes.is_empty() && self.root_palette.is_air()
  }

  /// GC：重建 nodes Vec 只保留 root 可达节点，回收被 merge / 覆盖废弃的索引。
  /// 迭代 DFS 将可达节点 move 到连续新 Vec 并重写 children 索引。
  pub fn compact(&mut self) {
    if self.nodes.len() <= 1 {
      return;
    }
    self.mark_identity_reset();
    let mut new_nodes: Vec<Node> = Vec::with_capacity(self.nodes.len());
    let mut idx_map = vec![u32::MAX; self.nodes.len()];
    let mut stack: Vec<usize> = vec![0];
    while let Some(old) = stack.pop() {
      if idx_map[old] != u32::MAX {
        continue;
      }
      idx_map[old] = new_nodes.len() as u32;

      let node = std::mem::replace(&mut self.nodes[old], Node::Uniform(PaletteId::AIR));
      match node {
        Node::Uniform(p) => new_nodes.push(Node::Uniform(p)),
        Node::Leaf(l) => new_nodes.push(Node::Leaf(l)),
        Node::Split { mask, palette, children } => {
          for &c in &children {
            stack.push(c as usize);
          }
          new_nodes.push(Node::Split { mask, palette, children });
        }
      }
    }

    for n in new_nodes.iter_mut() {
      if let Node::Split { children, .. } = n {
        for c in children.iter_mut() {
          *c = idx_map[*c as usize];
        }
      }
    }
    self.nodes = new_nodes;
  }

  pub fn node_count(&self) -> usize {
    self.nodes.len()
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  /// 节点 wire 内容快照（`id` → `(层, 掩码, tile 色, 子 id, 层 3 的 inline 值表)`）。
  /// **不含地址** —— 只回答"这个节点的 wire 内容变了没 / 它在第几层"。
  type WireEntry = (u8, u64, PaletteId, Vec<u32>, Option<[u32; LEAF_INLINE_WORDS]>);

  fn wire_snapshot(t: &ChunkTree) -> std::collections::HashMap<u32, WireEntry> {
    let mut map = std::collections::HashMap::new();
    if t.is_uniform_root() {
      map.insert(0, (0, 0, t.root_palette(), Vec::new(), None));
      return map;
    }
    let mut stack = vec![(0u32, 0u8)];
    while let Some((id, level)) = stack.pop() {
      let Some(v) = t.node_view(id) else { continue };
      let inline = if level == 3 { t.node_inline_words(id) } else { None };
      map.insert(id, (level, v.mask, v.palette, v.children.to_vec(), inline));
      if level < 3 {
        for &c in v.children {
          stack.push((c, level + 1));
        }
      }
    }
    map
  }

  /// 只比较快照里"wire 内容"部分（层号另判）
  fn same_content(a: &WireEntry, b: &WireEntry) -> bool {
    a.1 == b.1 && a.2 == b.2 && a.3 == b.3 && a.4 == b.4
  }

  /// 节点级 dirty 协议（wire 层增量重写的正确性前提）：一次编辑后，
  /// **所有 wire 内容变了的节点**都必须出现在 `take_dirty` 里 —— 宁可多报（多一次字节比较），
  /// 不可漏报（漏报 = GPU 侧静默不一致）。
  #[test]
  fn dirty_nodes_cover_every_changed_wire_node() {
    let mut t = ChunkTree::empty();
    let _ = t.take_dirty();
    let mut seed = 0x5EED_1234u32;
    let mut rnd = move || {
      seed ^= seed << 13;
      seed ^= seed >> 17;
      seed ^= seed << 5;
      seed
    };
    for step in 0..400 {
      let before = wire_snapshot(&t);
      let (x, y, z) = ((rnd() % 256) as i32, (rnd() % 256) as i32, (rnd() % 256) as i32);
      match step % 3 {
        2 => {
          let b = [x & !3, y & !3, z & !3];
          t.set_brick_voxels(b, (rnd() as u64) & (rnd() as u64), PaletteId((rnd() % 4) as u16 + 1));
        }
        _ => {
          t.set_voxel(x, y, z, PaletteId((rnd() % 4) as u16 + 1));
        }
      }
      let dirty = t.take_dirty();
      let after = wire_snapshot(&t);
      if dirty.reset {
        // 根被合并成 uniform：身份空间换过 ⇒ 消费方整体重建，内容比较无意义
        assert!(dirty.nodes.is_empty(), "step {step}: reset 时不该再带节点表");
        continue;
      }
      let dirty_set: std::collections::HashSet<u32> =
        dirty.nodes.iter().map(|&(_, id)| id).collect();
      for (id, content) in &after {
        let unchanged = before.get(id).is_some_and(|b| same_content(b, content));
        assert!(
          unchanged || dirty_set.contains(id),
          "step {step}: 节点 {id} 的 wire 内容变了但不在 dirty 表里"
        );
      }
      // 报的层号必须与节点当前所在的层一致（层错 ⇒ 值块 / 指针表形态选错）
      for &(level, id) in &dirty.nodes {
        if let Some(cur) = after.get(&id) {
          assert_eq!(cur.0, level, "step {step}: 节点 {id} 报的层 ≠ 实际层");
        }
      }
    }
  }

  /// layout 表 ↔ blob 的一致（wire 层按 layout 原地重写节点的前提）：
  /// `layout[id]` 处的 3 字 = 该节点的 mask / 色；拓扑节点的指针表**逐槽**指向子节点的 `layout[child]`。
  #[test]
  fn layout_offsets_match_blob() {
    let mut t = ChunkTree::empty();
    for x in 0..64 {
      for z in 0..64 {
        t.set_voxel(x, 0, z, PaletteId(2)); // 地板（跨多个 64³ 子块）
      }
    }
    for y in 0..8 {
      t.set_voxel(4, y, 4, PaletteId(3)); // 柱子
    }
    t.set_brick_voxels([64, 0, 64], 0x0F0F_0F0F_0F0F_0F0F, PaletteId(4));
    let (blob, layout) = t.serialize_with_layout();
    assert_eq!(layout[0], (0, 0), "根恒在 blob 首字、恒为 level 0");
    let mut stack = vec![(0u32, 0u8)];
    let mut seen = 0usize;
    while let Some((id, level)) = stack.pop() {
      let v = t.node_view(id).expect("可达节点必 live");
      let (off, lv) = layout[id as usize];
      assert_ne!(off, NODE_OFFSET_NONE, "id {id} 是 wire 节点但 layout 缺失");
      assert_eq!(lv, level, "id {id} 的 layout 层号不符");
      let o = off as usize;
      assert_eq!((blob[o + 1] as u64) << 32 | blob[o] as u64, v.mask, "id {id} 掩码不符");
      assert_eq!(blob[o + 2] & 0xFFFF, v.palette.get() as u32, "id {id} 的 tile 色不符");
      seen += 1;
      if v.mask == 0 {
        continue;
      }
      if level == 3 {
        assert_eq!(
          t.node_inline_words(id).expect("层 3 值块有 inline").as_slice(),
          &blob[o + 3..o + 3 + LEAF_INLINE_WORDS],
          "id {id} 的 inline 值表不符"
        );
      } else {
        for (slot, &c) in v.children.iter().enumerate() {
          assert_eq!(
            blob[o + 3 + slot],
            layout[c as usize].0,
            "id {id} 的第 {slot} 个子块指针未指向 layout"
          );
          stack.push((c, level + 1));
        }
      }
    }
    assert!(seen > 10, "树太小（{seen} 个 wire 节点），没测到东西");
  }

  /// 逐格参照（口径 = [`ChunkTree::set_brick_voxels`] 的文档）：`mask` 内、且"该写"才写。
  fn per_voxel_reference(t: &mut ChunkTree, b: [i32; 3], mask: u64, palette: PaletteId) -> u32 {
    let erase = palette.is_air();
    let mut changed = 0u32;
    for i in 0..64u32 {
      if (mask >> i) & 1 == 0 {
        continue;
      }
      let (x, y, z) = (b[0] + (i % 4) as i32, b[1] + ((i / 4) % 4) as i32, b[2] + (i / 16) as i32);
      let cur = t.get_voxel(x, y, z).unwrap_or(PaletteId::AIR);
      if cur.is_air() == erase {
        continue;
      }
      if t.set_voxel(x, y, z, palette) {
        changed += 1;
      }
    }
    changed
  }

  /// 批量写与逐格写结果必须**逐字相同**（序列化字节 + 64 格取值 + 改动计数）：
  /// 掩码覆盖空 / 满 / 单轴 / 随机稀疏，palette 覆盖"放置"与"擦除"，底子含地板与柱子（三种 brick 三态都遇到）。
  #[test]
  fn brick_voxels_matches_per_voxel_writes() {
    let base = |t: &mut ChunkTree| {
      for x in 0..64 {
        for z in 0..64 {
          t.set_voxel(x, 0, z, PaletteId(2)); // 地板
        }
      }
      for y in 0..8 {
        t.set_voxel(4, y, 4, PaletteId(3)); // 柱子
      }
    };
    let mut seed = 0x9E37_79B9u32;
    let mut rnd = move || {
      seed ^= seed << 13;
      seed ^= seed >> 17;
      seed ^= seed << 5;
      seed
    };
    let (bx, by, bz) = (4, 0, 4); // 与地板/柱子相交的 4³ 砖
    for case in 0..24 {
      let mask = match case {
        0 => 0,
        1 => u64::MAX,
        2 => 0x0000_0000_0000_000F,
        3 => 0xFFFF_0000_0000_0000,
        _ => (rnd() as u64) & (rnd() as u64),
      };
      for palette in [PaletteId(5), PaletteId::AIR] {
        let (mut a, mut b) = (ChunkTree::empty(), ChunkTree::empty());
        base(&mut a);
        base(&mut b);
        let ea = a.set_brick_voxels([bx, by, bz], mask, palette);
        let eb = per_voxel_reference(&mut b, [bx, by, bz], mask, palette);
        assert_eq!(ea, eb, "case {case} {palette:?}：改动计数不一致");
        for i in 0..64u32 {
          let p = (bx + (i % 4) as i32, by + ((i / 4) % 4) as i32, bz + (i / 16) as i32);
          assert_eq!(
            a.get_voxel(p.0, p.1, p.2),
            b.get_voxel(p.0, p.1, p.2),
            "case {case} {palette:?}：体素 {i} 不一致"
          );
        }
        assert_eq!(a.serialize(), b.serialize(), "case {case} {palette:?}：序列化字节不一致");
      }
    }
  }
}
