//! ChunkTree：Douglas Brick Tree 1:1 复刻（Phase 0 核心）
//!
//! 分裂树，分裂因子 4³=64，u64 occupancy mask per non-leaf，
//! 紧凑 child offset，uniform leaf 自适应。
//!
//! Phase 0 实现：
//! - 编辑层用结构化节点（Node enum），保证 set/get/merge 逻辑正确
//! - 序列化层 DFS flatten 成 `Vec<u32>` GPU buffer
//! - 每次编辑后重新 flatten（Phase 0 简化，性能之后再优化）
//!
//! Level 链：256 → 64 → 16 → 4 → 1（4 次分裂到 1³ 体素）

use super::coords::{child_linear_idx, BRICK_FACTOR, CHUNK_SIZE, LEVEL_EXTENT};

/// 结构化节点（编辑用）
#[derive(Debug, Clone)]
enum Node {
  /// uniform leaf：整 brick 同一 palette
  Uniform(u8),
  /// 分裂节点：64 个子块，mask bit=1 表示子块被分裂
  Split {
    mask: u64,
    /// 64 子块：每个是 uniform 颜色（palette）或分裂节点索引
    /// None 表示这个子块和 parent palette 相同（uniform，不存独立子节点）
    children: Vec<Option<usize>>,
    /// uniform 子块的默认 palette（mask bit=0 时子块颜色 = 此值）
    palette: u8,
  },
}

/// Douglas Brick Tree 1:1（分裂树，4³=64 分裂因子，u64 mask）
#[derive(Debug, Clone)]
pub struct ChunkTree {
  nodes: Vec<Node>,
  root_palette: u8,
}

impl ChunkTree {
  // =========================================================================
  // 构造
  // =========================================================================

  /// 空 chunk：root uniform AIR（palette=0）
  pub fn empty() -> Self {
    Self {
      nodes: Vec::new(),
      root_palette: 0,
    }
  }

  /// 全 uniform palette chunk
  pub fn uniform(palette: u8) -> Self {
    Self {
      nodes: Vec::new(),
      root_palette: palette,
    }
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
      out.push(self.root_palette as u32);
    } else {
      self.serialize_node(Some(0), self.root_palette, CHUNK_SIZE, &mut out);
    }
    out
  }

  fn serialize_node(
    &self,
    idx: Option<usize>,
    default_palette: u8,
    extent: i32,
    out: &mut Vec<u32>,
  ) {
    let (mask, palette_u32) = match idx {
      None => (0u64, self.root_palette as u32),
      Some(i) => match &self.nodes[i] {
        Node::Uniform(p) => (0u64, *p as u32),
        Node::Split { mask, palette, .. } => (*mask, *palette as u32),
      },
    };

    // 写 3 words: mask_lo, mask_hi, palette
    out.push(mask as u32);
    out.push((mask >> 32) as u32);
    out.push(palette_u32);

    if mask == 0 {
      return;
    }

    // 写 child offset 表（只存 mask bit=1 的子块）
    let child_extent = extent / BRICK_FACTOR;
    let num_children = mask.count_ones() as usize;
    // 先占位，后面回填 offset
    let offsets_start = out.len();
    out.resize(out.len() + num_children, 0);

    // 递归写子节点（DFS 顺序）
    let mut slot = 0usize;
    for i in 0u32..64 {
      let bit = 1u64 << i;
      if (mask & bit) != 0 {
        let child_offset = out.len() as u32;
        out[offsets_start + slot] = child_offset;
        slot += 1;

        // 这个子块的 palette：children[i] 是分裂节点索引，或 None
        // children[i] 的 default_palette 取决于它是 split 还是 uniform
        let child_idx = match idx {
          Some(p) => match &self.nodes[p] {
            Node::Split { children, .. } => children[i as usize],
            _ => unreachable!(),
          },
          None => {
            // root split 后也应该走 Split 分支
            // 但 idx=None 时 mask=0 已经 return 了，所以不会到这里
            unreachable!()
          }
        };
        let child_default = match child_idx {
          Some(ci) => match &self.nodes[ci] {
            Node::Uniform(p) => *p,
            Node::Split { palette, .. } => *palette,
          },
          None => {
            // mask bit=1 但没有子节点索引？不应该
            default_palette
          }
        };
        self.serialize_node(child_idx, child_default, child_extent, out);
      }
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
    // Phase 0 不暴露序列化内部
    0
  }

  pub fn node_palette(&self, _idx: u32) -> u32 {
    0
  }

  // =========================================================================
  // 查询
  // =========================================================================

  /// 查询指定体素（1³，level 4）的 palette；未设置返回 None
  pub fn get_voxel(&self, local_x: i32, local_y: i32, local_z: i32) -> Option<u8> {
    if self.nodes.is_empty() {
      return if self.root_palette == 0 { None } else { Some(self.root_palette) };
    }
    self.get_at(local_x, local_y, local_z, Some(0), self.root_palette, CHUNK_SIZE)
  }

  fn get_at(
    &self,
    x: i32,
    y: i32,
    z: i32,
    idx: Option<usize>,
    default_palette: u8,
    extent: i32,
  ) -> Option<u8> {
    let (mask, palette) = match idx {
      Some(i) => match &self.nodes[i] {
        Node::Uniform(p) => return if *p == 0 { None } else { Some(*p) },
        Node::Split { mask, palette, .. } => (*mask, *palette),
      },
      None => (0u64, self.root_palette),
    };

    // mask == 0 就是 uniform（idx=None 也是 uniform）
    if mask == 0 {
      return if palette == 0 { None } else { Some(palette) };
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
      return if palette == 0 { None } else { Some(palette) };
    }

    // 分裂子块：下钻
    let p = idx.unwrap();
    let child_idx = match &self.nodes[p] {
      Node::Split { children, .. } => children[child_i as usize],
      _ => unreachable!(),
    };
    let next_x = x - ix * child_extent;
    let next_y = y - iy * child_extent;
    let next_z = z - iz * child_extent;

    self.get_at(
      next_x, next_y, next_z, child_idx, palette, child_extent,
    )
  }

  /// uniform 查询：level L 的 brick 是否全同色
  pub fn get_uniform(&self, local_x: i32, local_y: i32, local_z: i32, level: u8) -> Option<u8> {
    let query_extent = LEVEL_EXTENT[level as usize];
    if self.nodes.is_empty() {
      return if self.root_palette == 0 { None } else { Some(self.root_palette) };
    }
    self.get_uniform_at(local_x, local_y, local_z, query_extent, Some(0), self.root_palette, CHUNK_SIZE)
  }

  fn get_uniform_at(
    &self,
    x: i32,
    y: i32,
    z: i32,
    query_extent: i32,
    idx: Option<usize>,
    default_palette: u8,
    cur_extent: i32,
  ) -> Option<u8> {
    let (mask, palette) = match idx {
      None => (0u64, self.root_palette),
      Some(i) => match &self.nodes[i] {
        Node::Uniform(p) => return if *p == 0 { None } else { Some(*p) },
        Node::Split { mask, palette, .. } => (*mask, *palette),
      },
    };

    if cur_extent <= query_extent {
      if mask == 0 {
        return if palette == 0 { None } else { Some(palette) };
      }
      // 分裂节点：检查所有子块
      let mut first_color: Option<u8> = None;
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
        let c: Option<u8> = if (mask & bit) != 0 {
          match children[i as usize] {
            Some(ci) => self.first_uniform_color(ci),
            None => Some(palette),
          }
        } else {
          Some(palette)
        };
        match c {
          None => return None,
          Some(c) => match first_color {
            None => first_color = Some(c),
            Some(f) if f != c => { all_same = false; break; }
            _ => {}
          }
        }
      }
      return if all_same { first_color } else { None };
    }

    // cur_extent > query_extent：下钻
    let child_extent = cur_extent / BRICK_FACTOR;
    if mask == 0 {
      return if palette == 0 { None } else { Some(palette) };
    }
    let ix = (x / child_extent).clamp(0, BRICK_FACTOR - 1);
    let iy = (y / child_extent).clamp(0, BRICK_FACTOR - 1);
    let iz = (z / child_extent).clamp(0, BRICK_FACTOR - 1);
    let child_i = child_linear_idx(ix, iy, iz);
    let bit_i = 1u64 << child_i;

    if (mask & bit_i) == 0 {
      return if palette == 0 { None } else { Some(palette) };
    }

    let p = idx.unwrap();
    let child_idx = match &self.nodes[p] {
      Node::Split { children, .. } => children[child_i as usize],
      _ => unreachable!(),
    };
    self.get_uniform_at(
      x - ix * child_extent,
      y - iy * child_extent,
      z - iz * child_extent,
      query_extent,
      child_idx,
      palette,
      child_extent,
    )
  }

  fn first_uniform_color(&self, idx: usize) -> Option<u8> {
    match &self.nodes[idx] {
      Node::Uniform(p) => if *p == 0 { None } else { Some(*p) },
      Node::Split { mask, children, palette, .. } => {
        let i = mask.trailing_zeros() as usize;
        let bit = 1u64 << i;
        if (mask & bit) != 0 {
          match children[i] {
            Some(ci) => self.first_uniform_color(ci),
            None => if *palette == 0 { None } else { Some(*palette) },
          }
        } else {
          if *palette == 0 { None } else { Some(*palette) }
        }
      }
    }
  }

  // =========================================================================
  // 编辑
  // =========================================================================

  /// 设置单个 1³ 体素的 palette（palette=0 = 清除）
  /// 返回：是否实际修改
  pub fn set_voxel(&mut self, local_x: i32, local_y: i32, local_z: i32, palette: u8) -> bool {
    let current = self.get_voxel(local_x, local_y, local_z);
    let want = if palette == 0 { None } else { Some(palette) };
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
    palette: u8,
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
      // 创建 split root
      let split_idx = self.nodes.len();
      self.nodes.push(Node::Split { mask: 0, palette: self.root_palette, children: vec![None; 64] });
      // 创建 64 个 Uniform(cur_palette) 子节点
      let start = self.nodes.len();
      for _ in 0..64 {
        self.nodes.push(Node::Uniform(cur_palette));
      }
      // 更新 split_idx 的 mask + children
      let split = &mut self.nodes[split_idx];
      if let Node::Split { mask, children, palette } = split {
        *mask = 0xFFFFFFFFFFFFFFFF;
        *palette = cur_palette;
        for i in 0..64 {
          children[i] = Some(start + i);
        }
      }
      node_idx = Some(split_idx);
    }

    let p = node_idx.unwrap();
    let (mask, cur_pal, children) = match &self.nodes[p] {
      Node::Split { mask, palette, children } => (*mask, *palette, children.clone()),
      _ => unreachable!(),
    };

    let child_extent = extent / BRICK_FACTOR;
    let ix = (x / child_extent).clamp(0, BRICK_FACTOR - 1);
    let iy = (y / child_extent).clamp(0, BRICK_FACTOR - 1);
    let iz = (z / child_extent).clamp(0, BRICK_FACTOR - 1);
    let child_i = child_linear_idx(ix, iy, iz);
    let next_x = x - ix * child_extent;
    let next_y = y - iy * child_extent;
    let next_z = z - iz * child_extent;

    // mask bit=0 且 children[child_i] = None → 这个子块还不存在（uniform = parent palette）
    // 需要创建 split node 并 replace
    let child_idx = children[child_i as usize];
    self.set_recursive(next_x, next_y, next_z, palette, child_idx, child_extent);

    // 更新 children 里可能新增的节点引用
    // （如果 uniform leaf 被 split 了，children[child_i] 现在应该指向新节点）
    let _ = mask; // suppress unused

    // 回溯：try merge
    self.try_merge(p);
  }

  /// 把一个 uniform 节点变成 split 节点（64 个 uniform 子节点 = old_palette）
  fn split_uniform(&mut self, idx: usize, old_palette: u8) {
    // 创建 64 个 Uniform 子节点
    let start = self.nodes.len();
    for _ in 0..64 {
      self.nodes.push(Node::Uniform(old_palette));
    }
    let mut children = vec![None; 64];
    for i in 0..64 {
      children[i] = Some(start + i);
    }
    self.nodes[idx] = Node::Split {
      mask: 0xFFFFFFFFFFFFFFFF,
      palette: old_palette,
      children,
    };
  }

  fn try_merge(&mut self, idx: usize) {
    let split = match &self.nodes[idx] {
      Node::Split { mask, children, palette } => (*mask, children.clone(), *palette),
      _ => return,
    };
    let (mask, children, palette) = split;

    if mask == 0 {
      return;
    }

    let mut first_color: Option<u8> = None;
    let mut all_uniform_same = true;

    for i in 0u32..64 {
      let bit = 1u64 << i;
      let c: Option<u8> = if (mask & bit) != 0 {
        match children[i as usize] {
          Some(ci) => match &self.nodes[ci] {
            Node::Uniform(p) => Some(*p),
            Node::Split { .. } => { all_uniform_same = false; break; }
          },
          None => Some(palette),
        }
      } else {
        Some(palette)
      };
      match c {
        None => {}
        Some(c) => match first_color {
          None => first_color = Some(c),
          Some(f) if f != c => { all_uniform_same = false; break; }
          _ => {}
        }
      }
    }

    if all_uniform_same {
      let merged = first_color.unwrap_or(0);
      if idx == 0 {
        self.nodes.clear();
        self.root_palette = merged;
      } else {
        self.nodes[idx] = Node::Uniform(merged);
      }
    }
  }

  pub fn clear_voxel(&mut self, local_x: i32, local_y: i32, local_z: i32) -> bool {
    self.set_voxel(local_x, local_y, local_z, 0)
  }

  /// 整个 chunk 是否 uniform AIR（空）
  pub fn is_empty(&self) -> bool {
    self.nodes.is_empty() && self.root_palette == 0
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

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn empty_chunk_is_uniform_air() {
    let t = ChunkTree::empty();
    let ser = t.serialize();
    assert_eq!(ser.len(), 3);
    assert_eq!(ser[0], 0); // mask low
    assert_eq!(ser[1], 0); // mask high
    assert_eq!(ser[2], 0); // palette
    assert_eq!(t.node_count(), 0);
  }

  #[test]
  fn uniform_chunk_readback() {
    let t = ChunkTree::uniform(42);
    assert_eq!(t.get_voxel(100, 50, 200), Some(42));
    assert_eq!(t.serialize()[2], 42);
  }

  #[test]
  fn set_single_voxel_readback() {
    let mut t = ChunkTree::empty();
    assert!(t.set_voxel(100, 50, 200, 5));
    assert_eq!(t.get_voxel(100, 50, 200), Some(5));
    assert_eq!(t.get_voxel(99, 50, 200), None);
    assert_eq!(t.get_voxel(101, 50, 200), None);
  }

  #[test]
  fn set_same_color_is_noop() {
    let mut t = ChunkTree::empty();
    assert!(t.set_voxel(5, 5, 5, 3));
    assert!(!t.set_voxel(5, 5, 5, 3));
    assert_eq!(t.get_voxel(5, 5, 5), Some(3));
  }

  #[test]
  fn clear_voxel_reads_none() {
    let mut t = ChunkTree::empty();
    t.set_voxel(5, 5, 5, 7);
    assert_eq!(t.get_voxel(5, 5, 5), Some(7));
    assert!(t.clear_voxel(5, 5, 5));
    assert_eq!(t.get_voxel(5, 5, 5), None);
  }

  #[test]
  fn multiple_voxels_different_colors() {
    let mut t = ChunkTree::empty();
    t.set_voxel(10, 0, 0, 1);
    t.set_voxel(200, 200, 200, 2);
    t.set_voxel(128, 128, 128, 3);
    assert_eq!(t.get_voxel(10, 0, 0), Some(1));
    assert_eq!(t.get_voxel(200, 200, 200), Some(2));
    assert_eq!(t.get_voxel(128, 128, 128), Some(3));
  }

  #[test]
  fn merge_after_clear_all_children() {
    let mut t = ChunkTree::empty();
    for i in 0..64u32 {
      let x = (i % 4) as i32;
      let y = ((i / 4) % 4) as i32;
      let z = (i / 16) as i32;
      t.set_voxel(x, y, z, (i + 1) as u8);
    }
    for i in 0..64u32 {
      let x = (i % 4) as i32;
      let y = ((i / 4) % 4) as i32;
      let z = (i / 16) as i32;
      t.clear_voxel(x, y, z);
    }
    // 整块应该回到 uniform AIR
    let ser = t.serialize();
    assert_eq!(ser[0], 0);
    assert_eq!(ser[1], 0);
    assert_eq!(ser[2], 0);
  }

  #[test]
  fn uniform_query_level_2() {
    let mut t = ChunkTree::empty();
    for z in 0..16i32 {
      for y in 0..16i32 {
        for x in 0..16i32 {
          t.set_voxel(x, y, z, 42);
        }
      }
    }
    assert_eq!(t.get_uniform(0, 0, 0, 2), Some(42));
    assert_eq!(t.get_uniform(16, 0, 0, 2), None);
  }

  #[test]
  fn empty_uniform_returns_none() {
    let t = ChunkTree::empty();
    assert_eq!(t.get_uniform(0, 0, 0, 0), None);
    assert_eq!(t.get_uniform(0, 0, 0, 4), None);
  }

  #[test]
  fn node_layout_roundtrip() {
    let mut t = ChunkTree::empty();
    t.set_voxel(100, 100, 100, 7);
    let ser = t.serialize();
    assert!(ser.len() >= 3);
    // root 应该是 split
    let mask = (ser[1] as u64) << 32 | ser[0] as u64;
    assert_eq!(mask, 0xFFFFFFFFFFFFFFFF);
    // 有 64 个 child offset
    assert!(ser.len() >= 3 + 64);
  }

  #[test]
  fn large_uniform_area_stays_coarse() {
    // 填 64³ 区域 [0,64)³，期望 level 1 child (0,0,0) merge 成 Uniform(9)，
    // 但 root 还是 Split（其他 3/4 的 chunk 是 AIR）
    let mut t = ChunkTree::empty();
    for z in 0..64i32 {
      for y in 0..64i32 {
        for x in 0..64i32 {
          t.set_voxel(x, y, z, 9);
        }
      }
    }
    // [0,64)³ 内所有点 = 9
    assert_eq!(t.get_voxel(0, 0, 0), Some(9));
    assert_eq!(t.get_voxel(63, 63, 63), Some(9));
    // 相邻区域 = None
    assert_eq!(t.get_voxel(64, 0, 0), None);
    // uniform 查询 level 1 brick at (0,0,0) = Uniform(9)
    assert_eq!(t.get_uniform(0, 0, 0, 1), Some(9));
    // level 2 brick at (0,0,0) 也是 Uniform(9)（它在 64³ 内）
    assert_eq!(t.get_uniform(0, 0, 0, 2), Some(9));
  }
}
