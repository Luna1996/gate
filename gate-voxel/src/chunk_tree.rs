//! ChunkTree：Douglas Brick Tree（分裂因子 4³=64，u64 occupancy mask per non-leaf，
//! 紧凑 child offset，uniform leaf 自适应）。
//! 编辑层用结构化节点（`Node` enum）；序列化层 DFS flatten 成 `Vec<u32>` GPU buffer；Level 链 256 → 64 → 16 → 4 → 1。
//! 材质索引是 `PaletteId`（16 位，容量 2^16）；节点 uniform 色与叶层逐体素色同宽，二者都从同一 u32 字段解包（见 `pack_pal_lod`）。

use super::coords::{BRICK_FACTOR, CHUNK_SIZE, LEVEL_EXTENT, child_linear_idx};
use crate::palette::{PALETTE_BITS, PaletteId};

/// 叶父层每 u32 字装的体素数（= 32 位 / 材质索引位宽）
pub const LEAF_VOXELS_PER_WORD: usize = 32 / PALETTE_BITS as usize;
/// 叶父层（level 3）inline 字数：4³ = 64 体素，每字装 [`LEAF_VOXELS_PER_WORD`] 个
pub const LEAF_INLINE_WORDS: usize = 64 / LEAF_VOXELS_PER_WORD;

/// 结构化节点（编辑用）；紧凑 child 表与 GPU wire 格式同构，只存 mask bit=1 的子块（按位序）。
#[derive(Debug, Clone)]
enum Node {
  /// uniform leaf：整 brick 同一 palette
  Uniform(PaletteId),
  /// 分裂节点：mask bit=1 的子块有独立节点，bit=0 = uniform（颜色 = palette）
  Split {
    mask: u64,
    /// 紧凑 child 表（按 mask 位序）：children[j] = 第 j 个 bit=1 子块的 nodes 下标，
    /// j = popcount(mask & (bit_i - 1))，len == popcount(mask)。
    children: Vec<u32>,
    /// uniform 子块的默认 palette（mask bit=0 时子块颜色 = 此值）
    palette: PaletteId,
  },
}

/// mask 中子块位序 i → 紧凑表下标（GPU DDA 同款 popcount 定位，O(1)）
#[inline]
fn child_slot(mask: u64, i: u32) -> u32 {
  debug_assert!(i < 64 && (mask & (1u64 << i)) != 0, "child_slot 要求 bit=1");
  (mask & ((1u64 << i) - 1)).count_ones()
}

/// 节点 palette word 打包：bit0..15 = uniform 子块色，bit16..31 = LOD 子树多数色。
/// 两半都是 `PaletteId`，与 shader `b_struct[node+2]` 的解包一致。
#[inline]
fn pack_pal_lod(palette: PaletteId, lod: PaletteId) -> u32 {
  palette.get() as u32 | ((lod.get() as u32) << 16)
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
}

impl ChunkTree {
  /// 空 chunk：root uniform AIR（palette=0）
  pub fn empty() -> Self {
    Self { nodes: Vec::new(), root_palette: PaletteId::AIR }
  }

  /// 全 uniform palette chunk
  pub fn uniform(palette: PaletteId) -> Self {
    Self { nodes: Vec::new(), root_palette: palette }
  }

  /// DFS 紧凑序列化（上传 GPU struct buffer）
  pub fn serialize(&self) -> Vec<u32> {
    let mut out = Vec::new();
    if self.nodes.is_empty() {
      out.push(0);
      out.push(0);
      out.push(pack_pal_lod(self.root_palette, self.root_palette));
    } else {
      self.serialize_node(Some(0), CHUNK_SIZE, &mut out);
    }
    out
  }

  /// DFS 序列化（上传 GPU struct buffer）：上层（level 0-2）紧凑格式，叶父层（level 3）inline 32 word。
  fn serialize_node(&self, idx: Option<usize>, extent: i32, out: &mut Vec<u32>) {
    let (mask, palette) = match idx {
      None => (0u64, self.root_palette),
      Some(i) => match &self.nodes[i] {
        Node::Uniform(p) => (0u64, *p),
        Node::Split { mask, palette, .. } => (*mask, *palette),
      },
    };
    let lod = self.node_lod(idx);

    // 写 3 words: mask_lo, mask_hi, palette(bit0..15) | lod(bit16..31)
    out.push(mask as u32);
    out.push((mask >> 32) as u32);
    out.push(pack_pal_lod(palette, lod));

    if mask == 0 {
      return;
    }

    let child_extent = extent / BRICK_FACTOR;

    if child_extent == 1 {
      // 叶父层（level 3）：inline 32 word，2 体素/word；读端 b_struct[node + 3 + (child_idx >> 1)]
      // 的第 (child_idx & 1) 个半字，bit=0 体素 = 0（AIR）。
      let inline_start = out.len();
      out.resize(out.len() + LEAF_INLINE_WORDS, 0);
      for i in 0u32..64 {
        let bit = 1u64 << i;
        if (mask & bit) != 0 {
          let child_idx = match idx {
            Some(p) => match &self.nodes[p] {
              Node::Split { children, .. } => children[child_slot(mask, i) as usize] as usize,
              _ => unreachable!(),
            },
            None => unreachable!(),
          };
          let pal = match &self.nodes[child_idx] {
            Node::Uniform(p) => *p,
            Node::Split { .. } => unreachable!(),
          };
          out[inline_start + (i >> 1) as usize] |= (pal.get() as u32) << ((i & 1) * 16);
        }
      }
    } else {
      // 内部层（level 0-2）：紧凑 child offset 表（只存 mask bit=1 的子块）
      let num_children = mask.count_ones() as usize;
      let offsets_start = out.len();
      out.resize(out.len() + num_children, 0);
      let mut slot = 0usize;
      for i in 0u32..64 {
        let bit = 1u64 << i;
        if (mask & bit) != 0 {
          let child_offset = out.len() as u32;
          out[offsets_start + slot] = child_offset;
          slot += 1;
          let child_idx = match idx {
            Some(p) => match &self.nodes[p] {
              Node::Split { children, .. } => children[child_slot(mask, i) as usize] as usize,
              _ => unreachable!(),
            },
            None => unreachable!(),
          };
          self.serialize_node(Some(child_idx), child_extent, out);
        }
      }
    }
  }

  /// 节点 LOD 代表色 = 子树「实体多数色」（排除空气计票；无实体 → AIR）；bit=0 子块 = 本节点 uniform palette。
  /// 并列取较大调色板值。
  fn node_lod(&self, idx: Option<usize>) -> PaletteId {
    match idx {
      None => self.root_palette,
      Some(i) => match &self.nodes[i] {
        Node::Uniform(p) => *p,
        Node::Split { mask, palette, children } => {
          let mut lods = [PaletteId::AIR; 64];
          for k in 0u32..64 {
            lods[k as usize] = if *mask & (1u64 << k) != 0 {
              let child = children[child_slot(*mask, k) as usize] as usize;
              self.node_lod(Some(child))
            } else {
              *palette
            };
          }
          let mut best = PaletteId::AIR;
          let mut best_n = 0usize;
          for v in lods {
            if v.is_air() {
              continue;
            }
            let n = lods.iter().filter(|&&x| x == v).count();
            if n > best_n || (n == best_n && v > best) {
              best_n = n;
              best = v;
            }
          }
          best
        }
      },
    }
  }

  pub fn len_words(&self) -> usize {
    self.serialize().len()
  }

  /// 序列化 buffer（上传 GPU 用）
  pub fn nodes(&self) -> Vec<u32> {
    self.serialize()
  }

  pub fn node_mask(&self, _idx: u32) -> u64 {
    0
  }

  pub fn node_palette(&self, _idx: u32) -> u32 {
    0
  }

  /// 查询指定体素（1³，level 4）的 palette；未设置返回 None
  pub fn get_voxel(&self, local_x: i32, local_y: i32, local_z: i32) -> Option<PaletteId> {
    if self.nodes.is_empty() {
      return solid(self.root_palette);
    }
    self.get_at(local_x, local_y, local_z, Some(0), CHUNK_SIZE)
  }

  fn get_at(&self, x: i32, y: i32, z: i32, idx: Option<usize>, extent: i32) -> Option<PaletteId> {
    let (mask, palette) = match idx {
      Some(i) => match &self.nodes[i] {
        Node::Uniform(p) => return solid(*p),
        Node::Split { mask, palette, .. } => (*mask, *palette),
      },
      None => (0u64, self.root_palette),
    };

    if mask == 0 {
      return solid(palette);
    }

    let child_extent = extent / BRICK_FACTOR;
    let ix = (x / child_extent).clamp(0, BRICK_FACTOR - 1);
    let iy = (y / child_extent).clamp(0, BRICK_FACTOR - 1);
    let iz = (z / child_extent).clamp(0, BRICK_FACTOR - 1);
    let child_i = child_linear_idx(ix, iy, iz);
    let bit_i = 1u64 << child_i;

    if (mask & bit_i) == 0 {
      return solid(palette);
    }

    let p = idx.unwrap();
    let child_idx = match &self.nodes[p] {
      Node::Split { children, .. } => children[child_slot(mask, child_i) as usize] as usize,
      _ => unreachable!(),
    };
    let next_x = x - ix * child_extent;
    let next_y = y - iy * child_extent;
    let next_z = z - iz * child_extent;

    self.get_at(next_x, next_y, next_z, Some(child_idx), child_extent)
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
    self.get_uniform_at(local_x, local_y, local_z, query_extent, Some(0), CHUNK_SIZE)
  }

  pub fn get_brick_state(&self, local_x: i32, local_y: i32, local_z: i32, level: u8) -> BrickState {
    let query_extent = LEVEL_EXTENT[level as usize];
    if self.nodes.is_empty() {
      return brick_state_of(self.root_palette);
    }
    self.brick_state_at(local_x, local_y, local_z, query_extent, Some(0), CHUNK_SIZE)
  }

  fn brick_state_at(
    &self,
    x: i32,
    y: i32,
    z: i32,
    query_extent: i32,
    idx: Option<usize>,
    cur_extent: i32,
  ) -> BrickState {
    let (mask, palette) = match idx {
      None => (0u64, self.root_palette),
      Some(i) => match &self.nodes[i] {
        Node::Uniform(p) => return brick_state_of(*p),
        Node::Split { mask, palette, .. } => (*mask, *palette),
      },
    };

    if cur_extent <= query_extent {
      return self.aggregate_node_state(mask, palette, idx);
    }

    let child_extent = cur_extent / BRICK_FACTOR;
    if mask == 0 {
      return brick_state_of(palette);
    }
    let ix = (x / child_extent).clamp(0, BRICK_FACTOR - 1);
    let iy = (y / child_extent).clamp(0, BRICK_FACTOR - 1);
    let iz = (z / child_extent).clamp(0, BRICK_FACTOR - 1);
    let child_i = child_linear_idx(ix, iy, iz);
    let bit_i = 1u64 << child_i;

    if (mask & bit_i) == 0 {
      return brick_state_of(palette);
    }

    let p = idx.unwrap();
    let child_idx = match &self.nodes[p] {
      Node::Split { children, .. } => children[child_slot(mask, child_i) as usize] as usize,
      _ => unreachable!(),
    };
    self.brick_state_at(
      x - ix * child_extent,
      y - iy * child_extent,
      z - iz * child_extent,
      query_extent,
      Some(child_idx),
      child_extent,
    )
  }

  /// 聚合节点完整区域的 64 子块 → 三态；任一子块与已见态冲突 → Mixed。
  fn aggregate_node_state(&self, mask: u64, palette: PaletteId, idx: Option<usize>) -> BrickState {
    let mut seen: Option<BrickState> = None;
    let children: &[u32] = match idx {
      Some(i) => match &self.nodes[i] {
        Node::Split { children, .. } => children.as_slice(),
        _ => unreachable!(),
      },
      None => return brick_state_of(palette),
    };
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
      Node::Split { mask, palette, .. } => {
        if *mask == 0 {
          // lazy Split：mask=0 即整节点 uniform。
          return brick_state_of(*palette);
        }
        self.aggregate_node_state(*mask, *palette, Some(idx))
      }
    }
  }

  fn get_uniform_at(
    &self,
    x: i32,
    y: i32,
    z: i32,
    query_extent: i32,
    idx: Option<usize>,
    cur_extent: i32,
  ) -> Option<PaletteId> {
    let (mask, palette) = match idx {
      None => (0u64, self.root_palette),
      Some(i) => match &self.nodes[i] {
        Node::Uniform(p) => return solid(*p),
        Node::Split { mask, palette, .. } => (*mask, *palette),
      },
    };

    if cur_extent <= query_extent {
      if mask == 0 {
        return solid(palette);
      }

      let mut first_color: Option<PaletteId> = None;
      let mut all_same = true;
      let children = match idx {
        Some(i) => match &self.nodes[i] {
          Node::Split { children, .. } => children.as_slice(),
          _ => unreachable!(),
        },
        None => unreachable!(),
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
    let bit_i = 1u64 << child_i;

    if (mask & bit_i) == 0 {
      return solid(palette);
    }

    let p = idx.unwrap();
    let child_idx = match &self.nodes[p] {
      Node::Split { children, .. } => children[child_slot(mask, child_i) as usize] as usize,
      _ => unreachable!(),
    };
    self.get_uniform_at(
      x - ix * child_extent,
      y - iy * child_extent,
      z - iz * child_extent,
      query_extent,
      Some(child_idx),
      child_extent,
    )
  }

  fn first_uniform_color(&self, idx: usize) -> Option<PaletteId> {
    match &self.nodes[idx] {
      Node::Uniform(p) => solid(*p),
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
  /// 返回是否实际修改。
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
        self.root_palette = palette;
      } else {
        self.nodes[idx.expect("非 root 层 idx 必为 Some")] = Node::Uniform(palette);
      }
      return;
    }

    let mut node_idx = idx;
    if let Some(i) = node_idx {
      if let Node::Uniform(p) = &self.nodes[i] {
        let old = *p;
        self.split_uniform(i, old);
      }
    } else if !self.nodes.is_empty() {
      node_idx = Some(0);
    }
    if node_idx.is_none() {
      let cur = self.root_palette;
      let split_idx = self.nodes.len();
      self.nodes.push(Node::Split { mask: 0, palette: cur, children: Vec::new() });
      node_idx = Some(split_idx);
    }

    let p = node_idx.unwrap();
    let child_extent = cur_extent / BRICK_FACTOR;
    let ix = x[0] / child_extent;
    let iy = x[1] / child_extent;
    let iz = x[2] / child_extent;
    let child_i = child_linear_idx(ix, iy, iz);

    let (child_idx, parent_pal, cur_mask) = match &self.nodes[p] {
      Node::Split { mask, palette, children } => {
        let bit = 1u64 << child_i;
        let ci = if (mask & bit) != 0 {
          Some(children[child_slot(*mask, child_i) as usize] as usize)
        } else {
          None
        };
        (ci, *palette, *mask)
      }
      _ => unreachable!(),
    };
    let child_idx = match child_idx {
      Some(ci) => ci,
      None => {
        let new = self.nodes.len() as u32;
        self.nodes.push(Node::Uniform(parent_pal));
        if let Node::Split { mask, children, .. } = &mut self.nodes[p] {
          let slot = child_slot(cur_mask | (1u64 << child_i), child_i) as usize;
          children.insert(slot, new);
          *mask |= 1u64 << child_i;
        }
        new as usize
      }
    };
    let next = [x[0] - ix * child_extent, x[1] - iy * child_extent, x[2] - iz * child_extent];
    self.fill_recursive(next, extent, palette, Some(child_idx), child_extent);

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
    let cur_palette = match idx {
      None => self.root_palette,
      Some(i) => match &self.nodes[i] {
        Node::Uniform(p) => *p,
        Node::Split { palette, .. } => *palette,
      },
    };

    if extent == 1 {
      let new_pal = palette;
      if let Some(i) = idx {
        self.nodes[i] = Node::Uniform(new_pal);
      } else {
        self.root_palette = new_pal;
      }
      return;
    }

    let mut node_idx = idx;
    if let Some(i) = node_idx {
      if let Node::Uniform(_) = &self.nodes[i] {
        self.split_uniform(i, cur_palette);
      }
    } else if !self.nodes.is_empty() {
      node_idx = Some(0);
    }

    if node_idx.is_none() {
      let split_idx = self.nodes.len();
      self.nodes.push(Node::Split { mask: 0, palette: cur_palette, children: Vec::new() });
      node_idx = Some(split_idx);
    }

    let p = node_idx.unwrap();
    let child_extent = extent / BRICK_FACTOR;
    let ix = (x / child_extent).clamp(0, BRICK_FACTOR - 1);
    let iy = (y / child_extent).clamp(0, BRICK_FACTOR - 1);
    let iz = (z / child_extent).clamp(0, BRICK_FACTOR - 1);
    let child_i = child_linear_idx(ix, iy, iz);
    let next_x = x - ix * child_extent;
    let next_y = y - iy * child_extent;
    let next_z = z - iz * child_extent;

    let (child_idx, parent_pal, cur_mask) = match &self.nodes[p] {
      Node::Split { mask, palette, children } => {
        let bit = 1u64 << child_i;
        let ci = if (mask & bit) != 0 {
          Some(children[child_slot(*mask, child_i) as usize] as usize)
        } else {
          None
        };
        (ci, *palette, *mask)
      }
      _ => unreachable!(),
    };
    let child_idx = match child_idx {
      Some(ci) => ci,
      None => {
        let new = self.nodes.len() as u32;
        self.nodes.push(Node::Uniform(parent_pal));
        if let Node::Split { mask, children, .. } = &mut self.nodes[p] {
          let slot = child_slot(cur_mask | (1u64 << child_i), child_i) as usize;
          children.insert(slot, new);
          *mask |= 1u64 << child_i;
        }
        new as usize
      }
    };
    self.set_recursive(next_x, next_y, next_z, palette, Some(child_idx), child_extent);

    self.try_merge(p);
  }

  /// 把 uniform 节点变成 split 节点（lazy：mask=0 全 uniform，子节点按需创建）。
  fn split_uniform(&mut self, idx: usize, old_palette: PaletteId) {
    self.nodes[idx] = Node::Split { mask: 0, palette: old_palette, children: Vec::new() };
  }

  fn try_merge(&mut self, idx: usize) {
    let (mask, palette) = match &self.nodes[idx] {
      Node::Split { mask, palette, .. } => (*mask, *palette),
      _ => return,
    };

    if mask == 0 {
      return;
    }

    let mut first_color: Option<PaletteId> = None;
    let mut all_uniform_same = true;
    {
      let children = match &self.nodes[idx] {
        Node::Split { children, .. } => children,
        _ => unreachable!(),
      };
      for i in 0u32..64 {
        let bit = 1u64 << i;
        let c: Option<PaletteId> = if (mask & bit) != 0 {
          match &self.nodes[children[child_slot(mask, i) as usize] as usize] {
            Node::Uniform(p) => Some(*p),
            Node::Split { .. } => {
              all_uniform_same = false;
              break;
            }
          }
        } else {
          Some(palette)
        };
        match c {
          None => {}
          Some(c) => match first_color {
            None => first_color = Some(c),
            Some(f) if f != c => {
              all_uniform_same = false;
              break;
            }
            _ => {}
          },
        }
      }
    }

    if all_uniform_same {
      let merged = first_color.unwrap_or(PaletteId::AIR);
      if idx == 0 {
        self.nodes.clear();
        self.root_palette = merged;
      } else {
        self.nodes[idx] = Node::Uniform(merged);
      }
    }
  }

  pub fn clear_voxel(&mut self, local_x: i32, local_y: i32, local_z: i32) -> bool {
    self.set_voxel(local_x, local_y, local_z, PaletteId::AIR)
  }

  /// 整个 chunk 是否 uniform AIR（空）
  pub fn is_empty(&self) -> bool {
    self.nodes.is_empty() && self.root_palette.is_air()
  }

  /// GC：重建 nodes Vec 只保留 root 可达节点，回收被 merge 废弃的索引。
  /// 迭代 DFS 将可达节点 move 到连续新 Vec 并重写 children 索引。
  pub fn compact(&mut self) {
    if self.nodes.len() <= 1 {
      return;
    }
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
    self.nodes.iter().filter(|n| matches!(n, Node::Split { .. })).count()
  }

  pub fn leaf_count(&self) -> usize {
    let ser = self.serialize();
    let mut leaves = 0;
    let mut idx = 0;
    while idx + 1 < ser.len() {
      let mask = (ser[idx + 1] as u64) << 32 | ser[idx] as u64;
      if mask == 0 {
        leaves += 1;
        idx += 3;
      } else {
        idx += 3 + mask.count_ones() as usize;
      }
    }
    leaves
  }
}
