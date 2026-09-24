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
//! 同宽，二者都从同一 u32 字段解包（见 [`pack_palette_word`]）。该字的高 16 位是**叶块代表值**
//! （M2，见 [`leaf_rep_palette`]）：shader 的叶级 LOD 拿它当命中材质，非叶节点恒 0。

use super::coords::{BRICK_FACTOR, CHUNK_SIZE, LEVEL_EXTENT, child_linear_idx};
use crate::palette::{PALETTE_BITS, PaletteId};

/// 叶父层每 u32 字装的体素数（= 32 位 / 材质索引位宽）
pub const LEAF_VOXELS_PER_WORD: usize = 32 / PALETTE_BITS as usize;
/// 叶父层（level 3）inline 字数：4³ = 64 体素，每字装 [`LEAF_VOXELS_PER_WORD`] 个
pub const LEAF_INLINE_WORDS: usize = 64 / LEAF_VOXELS_PER_WORD;
/// 叶块（level 3）体素数
const LEAF_VOXELS: usize = LEAF_INLINE_WORDS * LEAF_VOXELS_PER_WORD;
/// 叶代表值计数表的容量（块内**材质种类**数上限）：超过就退回"首个非空"（见 [`leaf_rep_palette`]）。
const REP_TALLY_CAP: usize = 8;

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
  /// **叶块代表值**（M2）：wire 的 palette word 高 16 位（bit16..31）在 CPU 侧的取值 ——
  /// 叶块 = [`leaf_rep_palette`]（块内加权众数），非叶节点 = [`PaletteId::AIR`]（wire 里恒 0）。
  pub rep: PaletteId,
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

  /// 本块的代表值（M2，见 [`leaf_rep_palette`]）：wire 的节点 palette word 高 16 位就填它。
  #[inline]
  fn rep(&self) -> PaletteId {
    leaf_rep_palette(self.mask, self.palette, &self.inline)
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

/// 节点 palette word 打包：bit0..15 = tile 色；bit16..31 = **叶块代表值**（M2；非叶 = AIR）。
///
/// 高 16 位曾是 wire 的旧字段「LOD 子树多数色」（逐节点递归计票占序列化 88% 的耗时，已整段移除，
/// 见 `docs/editable-gigavoxel.md` §9 的「M2 的一处事实修正」），现由 M2 复用为叶块代表值。
/// 读侧**所有 uniform 色的读取点都只取低 16 位**（shader 里逐处 `& 0xFFFF`）⇒ 填高半字对既有
/// 语义零影响；唯一读高半字的是叶级 LOD（`trace.wesl::leaf_lod_pal`）。
#[inline]
pub fn pack_palette_word(palette: PaletteId, rep: PaletteId) -> u32 {
  palette.get() as u32 | ((rep.get() as u32) << 16)
}

/// 叶块（4³）的**代表值**：块内按体素数加权的众数槽号 —— 出现次数最多的那个材质；平手取槽号小者；
/// 全空气 ⇒ 0（= 无代表，读侧退回旧口径"块内首个非空体素色"，故 0 只可能来自旧数据）。
///
/// **为什么存槽号、不存平均色**（M2 的语义决定）：槽号让材质属性照旧（命中直接进 `fetch_material`）、
/// 调色板改色自动跟随、也不必另立"命中 = 颜色"的第二条表示；众数 = 这块**看起来最像**的那个材质
/// （旧口径取"首个非空体素"，可能只是一粒别的色 ⇒ 整块 4 像素被染成它）。
///
/// 成本：一次 64 格扫描 + 一张 ≤ [`REP_TALLY_CAP`] 项的小表（常见块只有 2~4 种材质）⇒ 编辑路径每块
/// 百纳秒量级；块内材质种类爆表（噪声块）直接退回"首个非空"，不为精确众数付二次扫描。
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
    Some(NodeView { mask, palette, rep: self.node_rep(id as usize), children })
  }

  /// 节点的 wire 代表值（M2）：叶块 = [`leaf_rep_palette`]，其余（含越界 id）= AIR。
  #[inline]
  fn node_rep(&self, id: usize) -> PaletteId {
    match self.nodes.get(id) {
      Some(Node::Leaf(l)) => l.rep(),
      _ => PaletteId::AIR,
    }
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
    out.push(pack_palette_word(palette, self.node_rep(0)));
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
    // 写 3 words: mask_lo, mask_hi, palette（bit0..15 = tile 色、bit16..31 = 叶代表值 —— `pack_palette_word`）
    out.push(mask as u32);
    out.push((mask >> 32) as u32);
    out.push(pack_palette_word(palette, self.node_rep(idx)));
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

  // ---- proxy 树（M3）------------------------------------------------------------------------

  /// **proxy 树**：把 `keep_extent` 及以下的细节整体塌成「子树代表色」，保留 `keep_extent` 以上的
  /// 几何与拓扑。`keep_extent <= BRICK_FACTOR` ⇒ 与整棵树等价（不截断）。
  ///
  /// 用途：常驻预算紧张时用 proxy 顶替整棵 chunk（`brickmap::builder::ensure_resident` 的另一种来源）。
  ///
  /// CONSTRAINT: **不挖洞** —— 只要子树里有任何实体，塌出来的就是**实体色**（只有全空气子树才塌成空气）。
  /// 所以 proxy 的实心集合是原树的**超集**，外扩最多一个 `keep_extent`（与"停止下钻"的误差同量级）。
  /// CONSTRAINT: proxy 树**只用于安装、不参与编辑** ⇒ 不保证 `try_merge` 意义上的规范化
  /// （塌缩可能让"与父同色的 Uniform 子块"重新出现）。
  pub fn proxy(&self, keep_extent: i32) -> Self {
    let keep = keep_extent.max(BRICK_FACTOR);
    let mut out = Self {
      nodes: Vec::new(),
      root_palette: PaletteId::AIR,
      identity_reset: false,
      dirty_nodes: Vec::new(),
    };
    match self.nodes.first() {
      None => out.root_palette = self.root_palette,
      Some(Node::Uniform(p)) => out.root_palette = *p,
      // 整 chunk 也塌掉 ⇒ 结果必须落在 `root_palette` 上（"uniform 根不占节点"的表示约定）。
      Some(_) if CHUNK_SIZE <= keep => out.root_palette = self.rep_of(0),
      Some(_) => {
        let root = self.copy_proxy(0, CHUNK_SIZE, keep, &mut out.nodes);
        debug_assert_eq!(root, 0, "根必须落在 nodes[0]");
      }
    }
    out
  }

  /// 递归复制到 `out`（`extent <= keep` 时塌成代表色）；返回新树里的节点下标。
  fn copy_proxy(&self, id: usize, extent: i32, keep: i32, out: &mut Vec<Node>) -> usize {
    let my = out.len();
    if extent <= keep {
      out.push(Node::Uniform(self.rep_of(id)));
      return my;
    }
    let Node::Split { mask, palette, children } = &self.nodes[id] else {
      // keep ≥ BRICK_FACTOR ⇒ 叶层（extent = 4）必已在上面的分支塌缩；到这里的只可能是 Uniform。
      let Node::Uniform(p) = self.nodes[id] else { unreachable!("节点只有三种形态") };
      out.push(Node::Uniform(p));
      return my;
    };
    let (mask, palette) = (*mask, *palette);
    out.push(Node::Uniform(PaletteId::AIR)); // 占位：子节点必须先落位
    let child_extent = extent / BRICK_FACTOR;
    let mapped: Vec<u32> = children
      .iter()
      .map(|&c| self.copy_proxy(c as usize, child_extent, keep, out) as u32)
      .collect();
    out[my] = Node::Split { mask, palette, children: mapped };
    my
  }

  /// 子树代表色（0 = 全空气）。与 shader 的叶级 LOD **同一口径**：叶层 = 叶块代表值（块内加权
  /// 众数，M2 的 [`leaf_rep_palette`]，也就是 wire 高 16 位那个字段）；上层 = 先看本节点 `palette`
  /// （存在未置位格且非空 ⇒ 用它），否则按子块序找第一个非空的子树。
  fn rep_of(&self, id: usize) -> PaletteId {
    match &self.nodes[id] {
      Node::Uniform(p) => *p,
      Node::Leaf(l) => l.rep(),
      Node::Split { mask, palette, children } => {
        if *mask != u64::MAX && !palette.is_air() {
          return *palette;
        }
        for &c in children {
          let r = self.rep_of(c as usize);
          if !r.is_air() {
            return r;
          }
        }
        PaletteId::AIR
      }
    }
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
      assert_eq!(
        blob[o + 2],
        pack_palette_word(v.palette, v.rep),
        "id {id} 的 palette word（tile 色 + 叶代表值）不符"
      );
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

  /// M2：叶代表值 = 块内**按体素数加权的众数**；平手取槽号小者；全空气 = 0；
  /// 材质种类超过 [`REP_TALLY_CAP`] 时退回"首个非空"（与读侧旧口径一致）。
  #[test]
  fn leaf_rep_is_weighted_mode() {
    let put = |inline: &mut [u32; LEAF_INLINE_WORDS], i: u32, v: u16| {
      let (slot, shift) = ((i >> 1) as usize, (i & 1) * 16);
      inline[slot] = (inline[slot] & !(0xFFFF << shift)) | ((v as u32) << shift);
    };
    // 前 `n` 格 = `va`，其余 = `vb`；`pal` 只有当 `mask` 有 0 位时才是块内真实存在的色
    let build = |n: u32, va: u16, vb: u16| {
      let (mut inline, mut mask) = ([0u32; LEAF_INLINE_WORDS], 0u64);
      for i in 0..64u32 {
        put(&mut inline, i, if i < n { va } else { vb });
        mask |= 1u64 << i;
      }
      (mask, inline)
    };

    let (mask, inline) = build(40, 7, 3);
    assert_eq!(leaf_rep_palette(mask, PaletteId::AIR, &inline), PaletteId(7), "体素数多者胜");
    let (mask, inline) = build(32, 9, 2);
    assert_eq!(leaf_rep_palette(mask, PaletteId::AIR, &inline), PaletteId(2), "平手取槽号小者");
    // 全空气
    assert_eq!(leaf_rep_palette(0, PaletteId::AIR, &[0; LEAF_INLINE_WORDS]), PaletteId::AIR);
    // 掩码 0 ⇒ 64 格全是 tile 色
    assert_eq!(leaf_rep_palette(0, PaletteId(5), &[0; LEAF_INLINE_WORDS]), PaletteId(5));
    // 8 格 inline(4) + 56 格 tile(9)：tile 色也是候选，且体素数占优
    let mut inline = [0u32; LEAF_INLINE_WORDS];
    let mut mask = 0u64;
    for i in 0..8u32 {
      put(&mut inline, i, 4);
      mask |= 1u64 << i;
    }
    assert_eq!(leaf_rep_palette(mask, PaletteId(9), &inline), PaletteId(9), "未置位格的 tile 色要计入");
    // 9 种材质（超过计数表容量）⇒ 退回"首个非空"
    let mut inline = [0u32; LEAF_INLINE_WORDS];
    let mut mask = 0u64;
    for i in 0..64u32 {
      put(&mut inline, i, (i % 9 + 1) as u16);
      mask |= 1u64 << i;
    }
    assert_eq!(leaf_rep_palette(mask, PaletteId::AIR, &inline), PaletteId(1), "爆表退回首个非空");
  }

  /// M2：叶块的 wire 代表值写进 palette word 的高 16 位（= 加权众数，**不是**"首个非空"——
  /// 本块首格特意放了一粒别的色），非叶节点那个字段恒 0。
  #[test]
  fn wire_palette_word_carries_leaf_rep() {
    let mut t = ChunkTree::empty();
    t.set_brick_voxels([0, 0, 0], u64::MAX, PaletteId(2)); // 整块 = 2
    t.set_voxel(0, 0, 0, PaletteId(3)); // 首格 = 3（旧口径会选它）
    let (blob, layout) = t.serialize_with_layout();
    let mut leaves = 0;
    for (id, &(off, level)) in layout.iter().enumerate() {
      if off == NODE_OFFSET_NONE {
        continue; // 不可达节点
      }
      let word = blob[off as usize + 2];
      if level == 3 {
        assert_eq!(word >> 16, 2, "id {id} 的叶代表值应为加权众数 2（高半字 = {}）", word >> 16);
        leaves += 1;
      } else {
        assert_eq!(word >> 16, 0, "id {id} 在第 {level} 层，代表值字段应恒 0");
      }
    }
    assert_eq!(leaves, 1, "本用例只有一条到叶的路径");
  }

  /// M2：proxy 的塌缩色与叶级 LOD **同一口径** —— 混合叶块塌成**众数**色（不是"首个非空"那一粒）。
  #[test]
  fn proxy_collapse_uses_leaf_rep() {
    let mut t = ChunkTree::empty();
    t.set_brick_voxels([0, 0, 0], u64::MAX, PaletteId(2)); // 整块 = 2
    t.set_voxel(0, 0, 0, PaletteId(3)); // 首格 = 3（旧口径会选它）
    let p = t.proxy(BRICK_FACTOR); // 塌 4³ 及以下 ⇒ 该块成一个 uniform 色
    for x in 0..4 {
      for y in 0..4 {
        for z in 0..4 {
          assert_eq!(p.get_voxel(x, y, z), Some(PaletteId(2)), "proxy 塌缩色应为众数 2");
        }
      }
    }
  }

  /// M3：proxy **不挖洞** —— 原来实体的体素在 proxy 里仍然实体（只有全空气的子树才塌成空气），
  /// 且原来空的地方不会长出实体；同时节点数应显著下降（塌缩确实发生了）。
  #[test]
  fn proxy_has_no_holes() {
    let mut t = ChunkTree::empty();
    // 16³ 混合色实心块（三种 palette 交错，保证内部有 Split）+ 一个孤立体素
    for x in 0..16 {
      for y in 0..16 {
        for z in 0..16 {
          let p = match (x + y + z) % 3 {
            0 => PaletteId(1),
            1 => PaletteId(2),
            _ => PaletteId(3),
          };
          t.set_voxel(x, y, z, p);
        }
      }
    }
    t.set_voxel(200, 3, 40, PaletteId(3));
    let full_nodes = t.nodes.len();

    for keep in [16, 64] {
      let p = t.proxy(keep);
      for x in 0..16 {
        for y in 0..16 {
          for z in 0..16 {
            assert!(p.get_voxel(x, y, z).is_some(), "keep={keep}：原实体 ({x},{y},{z}) 在 proxy 里丢了");
          }
        }
      }
      assert!(p.get_voxel(200, 3, 40).is_some(), "keep={keep}：孤立实体丢了");
      assert!(p.get_voxel(64, 64, 64).is_none(), "keep={keep}：全空气区不该长出实体");
      assert!(p.nodes.len() < full_nodes, "keep={keep}：塌缩后节点数应减少");
    }

    // keep = 256 = "整 chunk 单色"这一档：整块本来就该填成同一个代表色（含原本是空气的位置）。
    let whole = t.proxy(CHUNK_SIZE);
    let rep = whole.get_voxel(0, 0, 0).expect("整块有实体 ⇒ 代表色非空");
    assert!(whole.nodes.is_empty(), "整 chunk 单色应落在 root_palette 上（零节点）");
    for p in [(0, 0, 0), (64, 64, 64), (200, 3, 40), (255, 255, 255)] {
      assert_eq!(whole.get_voxel(p.0, p.1, p.2), Some(rep), "整 chunk 档应为同一个代表色");
    }
  }

  /// M3：proxy 只塌 `keep_extent` **及以下**；`keep_extent = BRICK_FACTOR` 等价于不截断。
  /// 用一个"半个 16³ 块实心"的形状验证外扩范围：塌缩后该 16³ 块整体变实心，但相邻块不受影响。
  #[test]
  fn proxy_collapses_only_at_or_below_keep_extent() {
    let mut t = ChunkTree::empty();
    for x in 0..16 {
      for y in 0..8 {
        for z in 0..16 {
          t.set_voxel(x, y, z, PaletteId(1)); // 只用 16³ 块的下半
        }
      }
    }
    // 未截断：语义逐点不变
    let same = t.proxy(BRICK_FACTOR);
    for x in 0..16 {
      for y in 0..16 {
        for z in 0..16 {
          assert_eq!(
            same.get_voxel(x, y, z),
            t.get_voxel(x, y, z),
            "keep=4 应等价于不截断"
          );
        }
      }
    }
    // 截断到 16³：本块上半（原本是空气）随塌缩变实心；相邻块仍为空气
    let p = t.proxy(16);
    assert!(p.get_voxel(0, 12, 0).is_some(), "同块内应被代表色填实（不挖洞的另一面）");
    assert!(p.get_voxel(16, 0, 0).is_none(), "相邻 16³ 块不该受影响");
    assert!(p.get_voxel(0, 0, 200).is_none(), "远处仍为空气");
  }
}
