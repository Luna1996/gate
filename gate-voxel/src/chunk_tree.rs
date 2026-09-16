//! ChunkTree：Douglas Brick Tree（分裂因子 4³=64，u64 occupancy mask per non-leaf，
//! 紧凑 child offset，uniform leaf 自适应）。
//!
//! 编辑层用结构化节点（Node enum）；序列化层 DFS flatten 成 `Vec<u32>` GPU buffer。
//! Level 链：256 → 64 → 16 → 4 → 1。
//!
//! 材质索引是 [`PaletteId`]（16 位，容量 2^16）：**节点 uniform 色与叶层逐体素色同宽**，
//! 因为 GPU 侧两者都从同一个 u32 字段解包（见 [`pack_pal_lod`]）。

use super::coords::{BRICK_FACTOR, CHUNK_SIZE, LEVEL_EXTENT, child_linear_idx};
use crate::palette::{PALETTE_BITS, PaletteId};

/// 叶父层每 u32 字装的体素数（= 32 位 / 材质索引位宽）
pub const LEAF_VOXELS_PER_WORD: usize = 32 / PALETTE_BITS as usize;
/// 叶父层（level 3）inline 字数：4³ = 64 体素，每字装 [`LEAF_VOXELS_PER_WORD`] 个
pub const LEAF_INLINE_WORDS: usize = 64 / LEAF_VOXELS_PER_WORD;

/// 结构化节点（编辑用）
///
/// 内存约束：N=10 场景树节点量级在百万。紧凑 child 表与 GPU wire 格式同构：
/// 只存 mask bit=1 的子块（按位序）。
#[derive(Debug, Clone)]
enum Node {
  /// uniform leaf：整 brick 同一 palette
  Uniform(PaletteId),
  /// 分裂节点：mask bit=1 的子块有独立节点，bit=0 = uniform（颜色 = palette）
  Split {
    mask: u64,
    /// 紧凑 child 表（按 mask 位序）：children[j] = 第 j 个 bit=1 子块的
    /// nodes 下标；j = popcount(mask & (bit_i - 1))。len() == popcount(mask)
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

/// 节点 palette word 打包：**bit0..15 = uniform 子块色，bit16..31 = LOD 子树多数色**。
///
/// 两半都是 [`PaletteId`]（16 位）—— 节点色与体素色必须同宽，否则"整块同色"的节点
/// 表达不了全部材质。参见 `wire.rs` 注释与 shader `b_struct[node+2]` 的解包。
///
/// 【现状】shader 侧目前只读低 16 位（uniform 色）；早停走 `depth_cap` 并用 `b.pal`
/// 近似，尚未读 LOD 半区。LOD 半区为将来的距离 LOD（同一棵树早停出图）预留。
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

/// brick **边长** → 层级（[`LEVEL_EXTENT`] 的下标）。非 brick 粒度即编程错误，直接 panic。
///
/// 编辑侧按边长思考（64³ 的整块填充），而树查询接口按 level 索引 —— 这里集中做一次映射，
/// 免得每个调用方各写一遍 `position()`。
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
  // =========================================================================
  // 构造
  // =========================================================================

  /// 空 chunk：root uniform AIR（palette=0）
  pub fn empty() -> Self {
    Self { nodes: Vec::new(), root_palette: PaletteId::AIR }
  }

  /// 全 uniform palette chunk
  pub fn uniform(palette: PaletteId) -> Self {
    Self { nodes: Vec::new(), root_palette: palette }
  }

  // =========================================================================
  // 序列化：DFS flatten 成 Vec<u32>
  // =========================================================================

  /// DFS 紧凑序列化（上传 GPU struct buffer）
  pub fn serialize(&self) -> Vec<u32> {
    let mut out = Vec::new();
    if self.nodes.is_empty() {
      // root uniform（没有 Split 节点）
      out.push(0); // mask low
      out.push(0); // mask high
      out.push(pack_pal_lod(self.root_palette, self.root_palette));
    } else {
      self.serialize_node(Some(0), CHUNK_SIZE, &mut out);
    }
    out
  }

  /// DFS 序列化（上传 GPU struct buffer）。
  ///
  /// 上层（level 0-2）紧凑格式；叶父层（level 3）inline **32 word**，2 体素/word
  /// （每 16 位 = 一个 palette，child_idx 低 1 位选半字）。
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
      // 叶父层（level 3）：inline 32 word，2 体素/word（每 16 位 = 一个 palette）。
      // 读端：b_struct[node + 3 + (child_idx >> 1)] 的第 (child_idx & 1) 个半字。
      // bit=0 体素 = 0（AIR）。
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

  /// 节点 LOD 代表色 = 子树「实体多数色」（排除空气计票；子树无实体 → AIR）。
  /// bit=0 子块 = 本节点 uniform palette（=0 空气不计票），bit=1 子块递归子节点 lod。
  ///
  /// 用途：距离 LOD —— 远距时不下钻到叶子，停在该层直接用本值出图（Douglas 原设计：
  /// 「stop at the selected level of detail and then use the materials stored there」）。
  /// shader 侧**尚未接入**：目前远距早停走 `depth_cap` + 节点 uniform 色近似。
  ///
  /// 实现**零分配**：只把 ≤64 个子块的代表色收进定长数组，再两两计数取众数。
  /// （旧版用 `[u16; 256]` 直方图按调色板值索引；容量到 2^16 后那会是 128KB 栈数组。）
  /// 并列时取**较大**的调色板值，与旧版 `counts.iter().max_by_key`（返回最后一个最大项）
  /// 的取向一致；但**全空气子树现在正确返回 AIR**（旧版会返回 255，见下方注释）。
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

  // =========================================================================
  // 节点级工具（pub 便于 VolumeGrid 做 chunk 清理检查）
  // =========================================================================

  /// 序列化 buffer（上传 GPU 用）
  pub fn nodes(&self) -> Vec<u32> {
    self.serialize()
  }

  pub fn node_mask(&self, _idx: u32) -> u64 {
    // 占位：本层不暴露序列化内部
    0
  }

  pub fn node_palette(&self, _idx: u32) -> u32 {
    0
  }

  // =========================================================================
  // 查询
  // =========================================================================

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

    // mask == 0 就是 uniform（idx=None 也是 uniform）
    if mask == 0 {
      return solid(palette);
    }

    // 分裂节点：定位子块
    let child_extent = extent / BRICK_FACTOR;
    let ix = (x / child_extent).clamp(0, BRICK_FACTOR - 1);
    let iy = (y / child_extent).clamp(0, BRICK_FACTOR - 1);
    let iz = (z / child_extent).clamp(0, BRICK_FACTOR - 1);
    let child_i = child_linear_idx(ix, iy, iz);
    let bit_i = 1u64 << child_i;

    if (mask & bit_i) == 0 {
      // uniform 子块，颜色 = parent palette
      return solid(palette);
    }

    // 分裂子块：下钻
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
      // 到达查询粒度：聚合整个节点的 64 子块三态
      return self.aggregate_node_state(mask, palette, idx);
    }

    // cur_extent > query_extent：下钻
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

  /// 聚合节点完整区域的 64 子块 → 三态（到达查询粒度时调用）
  ///
  /// 早退：任一子块与已见态冲突 → Mixed。bit=0 子块 = 父 palette 的 uniform。
  fn aggregate_node_state(&self, mask: u64, palette: PaletteId, idx: Option<usize>) -> BrickState {
    // None = 未定；Some(st) = 已见唯一态；再遇异态 → Mixed
    let mut seen: Option<BrickState> = None;
    let children: &[u32] = match idx {
      Some(i) => match &self.nodes[i] {
        Node::Split { children, .. } => children.as_slice(),
        _ => unreachable!(),
      },
      None => return brick_state_of(palette), // 空 nodes + root palette
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

  /// 单节点完整区域三态（不限查询窗口；与 aggregate_node_state 互递归，Mixed 早退）
  fn node_state(&self, idx: usize) -> BrickState {
    match &self.nodes[idx] {
      Node::Uniform(p) => brick_state_of(*p),
      Node::Split { mask, palette, .. } => {
        if *mask == 0 {
          // lazy Split：mask=0 即整节点 uniform（防御 trailing_zeros(0) 越界）
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
      // 分裂节点：检查所有子块
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

    // cur_extent > query_extent：下钻
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
          // lazy Split：整节点 uniform（防御 trailing_zeros(0)=64 越界）
          return solid(*palette);
        }
        let i = mask.trailing_zeros();
        // mask bit=1 ⇔ 紧凑表有该子块（不变式），直接取首个 bit=1 子块颜色
        self.first_uniform_color(children[child_slot(*mask, i) as usize] as usize)
      }
    }
  }

  // =========================================================================
  // 编辑
  // =========================================================================

  /// 填充对齐 brick（extent ∈ LEVEL_EXTENT）：O(depth) 树路径写入。
  ///
  /// 一次调用只沿路径创建 ≤depth 个节点（lazy split），比逐体素 [`Self::set_voxel`]
  /// 少 64× 节点创建。返回：是否实际修改。
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
    // 只读预检查：非零 palette 且 brick 已 uniform 同色 → noop
    //（palette=0 时 get_uniform 的 None 语义与「非 uniform」歧义，跳过预检查）
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
        // 整 chunk uniform：规范形 = 空 nodes + root_palette（is_empty 语义一致）
        self.nodes.clear();
        self.root_palette = palette;
      } else {
        self.nodes[idx.expect("非 root 层 idx 必为 Some")] = Node::Uniform(palette);
      }
      return;
    }

    // 确保 current 是 Split（与 set_recursive 同一套 root/split 逻辑）
    let mut node_idx = idx;
    if let Some(i) = node_idx {
      if let Node::Uniform(p) = &self.nodes[i] {
        let old = *p;
        self.split_uniform(i, old);
      }
    } else if !self.nodes.is_empty() {
      // root Split 已存在（之前编辑过），从 nodes[0] 开始
      node_idx = Some(0);
    }
    if node_idx.is_none() {
      // root uniform（nodes 为空）→ 创建 split root（lazy：mask=0，紧凑表为空）
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
    // lazy split：mask bit=0 → 子块 uniform（= 本节点 palette），
    // 首次编辑才创建节点 + 置 bit + 紧凑表插入
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
    // 回溯：try merge
    self.try_merge(p);
  }

  /// 设置单个 1³ 体素的 palette（palette=0 = 清除）
  /// 返回：是否实际修改
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

    // 最细 level：直接写
    if extent == 1 {
      let new_pal = palette;
      if let Some(i) = idx {
        self.nodes[i] = Node::Uniform(new_pal);
      } else {
        self.root_palette = new_pal;
      }
      return;
    }

    // uniform leaf → 先 split（仅当当前节点是 Uniform）
    let mut node_idx = idx;
    if let Some(i) = node_idx {
      if let Node::Uniform(_) = &self.nodes[i] {
        self.split_uniform(i, cur_palette);
      }
    } else if !self.nodes.is_empty() {
      // 已经有 root Split 了（之前编辑过），从 nodes[0] 开始
      node_idx = Some(0);
    }

    // 如果是 root uniform（idx=None, nodes 为空），需要把 root 变成 split
    if node_idx.is_none() {
      // 创建 split root（lazy：mask=0，紧凑表为空，子节点按需创建）
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

    // lazy split：mask bit=0 → 子块 uniform（= 本节点 palette），
    // 首次编辑才创建节点 + 置 bit + 紧凑表插入
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

    // 回溯：try merge
    self.try_merge(p);
  }

  /// 把一个 uniform 节点变成 split 节点（lazy：mask=0 全 uniform，子节点按需创建）。
  ///
  /// mask bit=0 → uniform leaf（颜色 = 父节点 palette），不占内存；只有真正被编辑的
  /// 子块才置 bit + 建节点。
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

    // 阶段 1：只读扫描（零分配）
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
    } // children 不可变借用结束

    // 阶段 2：可变写
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

  /// GC：重建 nodes Vec 只保留 root 可达的有效节点，回收废弃索引。
  ///
  /// split + try_merge 会留下被 merge 掉的子节点（仍在 Vec 中占位，容量只增不减）。
  /// 迭代 DFS 把可达节点 **move** 到连续新 Vec（零 clone）并重写 children 索引；
  /// O(n) 时间 + O(n) 临时空间，编辑完成后调一次即可。
  pub fn compact(&mut self) {
    if self.nodes.len() <= 1 {
      return; // 空 root 或单节点，无废弃
    }
    let mut new_nodes: Vec<Node> = Vec::with_capacity(self.nodes.len());
    let mut idx_map = vec![u32::MAX; self.nodes.len()];
    let mut stack: Vec<usize> = vec![0];
    while let Some(old) = stack.pop() {
      if idx_map[old] != u32::MAX {
        continue; // 已访问（树形无重访，防御性保留）
      }
      idx_map[old] = new_nodes.len() as u32;
      // 占位替换 + move：Split 的 children 随节点搬入新 Vec，零堆分配
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
    // 重写 children 索引：old → new
    for n in new_nodes.iter_mut() {
      if let Node::Split { children, .. } = n {
        for c in children.iter_mut() {
          *c = idx_map[*c as usize];
        }
      }
    }
    self.nodes = new_nodes;
  }

  // =========================================================================
  // 统计
  // =========================================================================

  pub fn node_count(&self) -> usize {
    self.nodes.iter().filter(|n| matches!(n, Node::Split { .. })).count()
  }

  pub fn leaf_count(&self) -> usize {
    // 用序列化来算（保证正确）
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
