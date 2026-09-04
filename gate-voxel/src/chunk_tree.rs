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

use super::coords::{BRICK_FACTOR, CHUNK_SIZE, LEVEL_EXTENT, child_linear_idx};

/// 结构化节点（编辑用）
///
/// 内存约束：N=10 场景树节点量级在百万，每节点字节数直接决定能否启动
/// （64 槽 Vec<Option<usize>> = 552B/节点曾致 5.9GB OOM）。紧凑 child 表
/// 与 GPU wire 格式同构：只存 mask bit=1 的子块（按位序）。
#[derive(Debug, Clone)]
enum Node {
  /// uniform leaf：整 brick 同一 palette
  Uniform(u8),
  /// 分裂节点：mask bit=1 的子块有独立节点，bit=0 = uniform（颜色 = palette）
  Split {
    mask: u64,
    /// 紧凑 child 表（按 mask 位序）：children[j] = 第 j 个 bit=1 子块的
    /// nodes 下标；j = popcount(mask & (bit_i - 1))。len() == popcount(mask)
    children: Vec<u32>,
    /// uniform 子块的默认 palette（mask bit=0 时子块颜色 = 此值）
    palette: u8,
  },
}

/// mask 中子块位序 i → 紧凑表下标（GPU DDA 同款 popcount 定位，O(1)）
#[inline]
fn child_slot(mask: u64, i: u32) -> u32 {
  debug_assert!(i < 64 && (mask & (1u64 << i)) != 0, "child_slot 要求 bit=1");
  (mask & ((1u64 << i) - 1)).count_ones()
}

/// 节点 palette word 打包：低字节 = uniform 子块色（既有 wire 语义），
/// 高字节 = LOD 子树多数色（GPU 八叉树早停用，dda.wgsl 解码 (w>>8)&0xFF）
#[inline]
fn pack_pal_lod(palette: u8, lod: u8) -> u32 {
  palette as u32 | ((lod as u32) << 8)
}

/// brick 三态（R3-10 DDGI 探针烘焙：cell 16³ = level 2 brick）
///
/// `get_uniform` 的 `None` 语义同时涵盖「uniform 空气」与「含空气混合」，
/// 探针烘焙需要区分 Air（居中放探针）/ Solid（无探针）/ Mixed（BFS 找最大空叶），
/// 故单独提供三态查询。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BrickState {
  /// 整 brick 空气（palette 0）
  Air,
  /// 整 brick 同一非零 palette
  Solid(u8),
  /// 空气与实体混合（或多种颜色混合）
  Mixed,
}

/// palette → brick 三态（palette 0 = AIR）
#[inline]
fn brick_state_of(palette: u8) -> BrickState {
  if palette == 0 {
    BrickState::Air
  } else {
    BrickState::Solid(palette)
  }
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
      out.push(pack_pal_lod(self.root_palette, self.root_palette));
    } else {
      self.serialize_node(Some(0), CHUNK_SIZE, &mut out);
    }
    out
  }

  /// DFS 序列化（上传 GPU struct buffer）。
  ///
  /// **wire v2**（2026-09-04 DDA 加速）：**叶父层（level 3）inline 64 palette**，
  /// 消除 level 4 叶节点（3 word → 0）和 child_addr indirection（2 load → 1）。
  /// 上层（level 0-2）保持紧凑格式（省空间，稀疏节点不膨胀 64×）。
  fn serialize_node(&self, idx: Option<usize>, extent: i32, out: &mut Vec<u32>) {
    let (mask, palette_u32) = match idx {
      None => (0u64, self.root_palette as u32),
      Some(i) => match &self.nodes[i] {
        Node::Uniform(p) => (0u64, *p as u32),
        Node::Split { mask, palette, .. } => (*mask, *palette as u32),
      },
    };
    let lod = self.node_lod(idx);

    // 写 3 words: mask_lo, mask_hi, palette(low) | lod(high)
    out.push(mask as u32);
    out.push((mask >> 32) as u32);
    out.push(pack_pal_lod(palette_u32 as u8, lod));

    if mask == 0 {
      return;
    }

    let child_extent = extent / BRICK_FACTOR;

    if child_extent == 1 {
      // 叶父层（level 3）：inline 64 palette word（非紧凑，bit=0 位填 0）
      // 消除 level 4 叶节点；GPU 读 b_struct[node + 3 + child_idx] 直取 palette
      let inline_start = out.len();
      out.resize(out.len() + 64, 0);
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
          let (pal, lod_val) = match &self.nodes[child_idx] {
            Node::Uniform(p) => (*p, *p),
            Node::Split { .. } => unreachable!(),
          };
          out[inline_start + i as usize] = pack_pal_lod(pal, lod_val);
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

  /// 节点 LOD 代表色 = 子树「实体多数色」（排除空气计票；子树无实体 → 0）。
  /// bit=0 子块 = 本节点 uniform palette（=0 空气不计票）；bit=1 子块 = 子节点
  /// lod 递归。uniform 节点 = 自身色（空气 → 0）。
  /// GPU 侧 lod!=0 即早停：区域含实体就按多数固体色整块出图——触发条件保证
  /// 区域投影 <1px，剪影/颜色误差 ≤ 子块边长 = 亚像素。排除空气是关键：
  /// 地形薄表面区域空气占多数，若含空气计票则 lod 恒 0 永不早停。
  fn node_lod(&self, idx: Option<usize>) -> u8 {
    match idx {
      None => self.root_palette,
      Some(i) => match &self.nodes[i] {
        Node::Uniform(p) => *p,
        Node::Split { mask, palette, children } => {
          let mut counts = [0u16; 256];
          for i in 0u32..64 {
            let c = if *mask & (1u64 << i) != 0 {
              let child = children[child_slot(*mask, i) as usize] as usize;
              self.node_lod(Some(child))
            } else {
              *palette
            };
            if c != 0 {
              counts[c as usize] += 1;
            }
          }
          counts
            .iter()
            .enumerate()
            .max_by_key(|(_, c)| **c)
            .map(|(i, _)| i as u8)
            .unwrap_or(0)
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
      return if self.root_palette == 0 {
        None
      } else {
        Some(self.root_palette)
      };
    }
    self.get_at(local_x, local_y, local_z, Some(0), CHUNK_SIZE)
  }

  fn get_at(&self, x: i32, y: i32, z: i32, idx: Option<usize>, extent: i32) -> Option<u8> {
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
      Node::Split { children, .. } => children[child_slot(mask, child_i) as usize] as usize,
      _ => unreachable!(),
    };
    let next_x = x - ix * child_extent;
    let next_y = y - iy * child_extent;
    let next_z = z - iz * child_extent;

    self.get_at(next_x, next_y, next_z, Some(child_idx), child_extent)
  }

  /// uniform 查询：level L 的 brick 是否全同色
  pub fn get_uniform(&self, local_x: i32, local_y: i32, local_z: i32, level: u8) -> Option<u8> {
    let query_extent = LEVEL_EXTENT[level as usize];
    if self.nodes.is_empty() {
      return if self.root_palette == 0 {
        None
      } else {
        Some(self.root_palette)
      };
    }
    self.get_uniform_at(local_x, local_y, local_z, query_extent, Some(0), CHUNK_SIZE)
  }

  /// brick 三态查询（DDGI 探针烘焙）：Air / Solid(palette) / Mixed
  ///
  /// `local_*` = chunk 内 fine 坐标（brick 最小角），`level` ∈ 0..5 对应
  /// `LEVEL_EXTENT` = [256, 64, 16, 4, 1]。
  pub fn get_brick_state(&self, local_x: i32, local_y: i32, local_z: i32, level: u8) -> BrickState {
    let query_extent = LEVEL_EXTENT[level as usize];
    if self.nodes.is_empty() {
      return if self.root_palette == 0 {
        BrickState::Air
      } else {
        BrickState::Solid(self.root_palette)
      };
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
  fn aggregate_node_state(&self, mask: u64, palette: u8, idx: Option<usize>) -> BrickState {
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
          // lazy Split：整节点 uniform（防御 trailing_zeros(0) 同款哨兵）
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
          self.first_uniform_color(children[child_slot(mask, i) as usize] as usize)
        } else {
          Some(palette)
        };
        match c {
          None => return None,
          Some(c) => match first_color {
            None => first_color = Some(c),
            Some(f) if f != c => {
              all_same = false;
              break;
            }
            _ => {}
          },
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

  fn first_uniform_color(&self, idx: usize) -> Option<u8> {
    match &self.nodes[idx] {
      Node::Uniform(p) => {
        if *p == 0 {
          None
        } else {
          Some(*p)
        }
      }
      Node::Split {
        mask,
        children,
        palette,
        ..
      } => {
        if *mask == 0 {
          // lazy Split：整节点 uniform（防御 trailing_zeros(0)=64 越界）
          return if *palette == 0 { None } else { Some(*palette) };
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

  /// 填充对齐 brick（extent ∈ LEVEL_EXTENT）：树路径 O(depth) 写入。
  ///
  /// 大体积均匀填充专用：一次调用只沿路径创建 ≤depth 个节点（lazy split），
  /// 比逐体素 [`Self::set_voxel`] 少 64× 节点创建。返回：是否实际修改。
  pub fn fill_brick(&mut self, local: [i32; 3], extent: i32, palette: u8) -> bool {
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
    if palette != 0 {
      let level = LEVEL_EXTENT
        .iter()
        .position(|&e| e == extent)
        .expect("LEVEL_EXTENT.contains 已保证") as u8;
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
    palette: u8,
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
      self.nodes.push(Node::Split {
        mask: 0,
        palette: cur,
        children: Vec::new(),
      });
      node_idx = Some(split_idx);
    }

    let p = node_idx.unwrap();
    let child_extent = cur_extent / BRICK_FACTOR;
    let ix = x[0] / child_extent;
    let iy = x[1] / child_extent;
    let iz = x[2] / child_extent;
    let child_i = child_linear_idx(ix, iy, iz) as u32;
    // lazy split：mask bit=0 → 子块 uniform（= 本节点 palette），
    // 首次编辑才创建节点 + 置 bit + 紧凑表插入
    let (child_idx, parent_pal, cur_mask) = match &self.nodes[p] {
      Node::Split {
        mask,
        palette,
        children,
      } => {
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
    let next = [
      x[0] - ix * child_extent,
      x[1] - iy * child_extent,
      x[2] - iz * child_extent,
    ];
    self.fill_recursive(next, extent, palette, Some(child_idx), child_extent);
    // 回溯：try merge
    self.try_merge(p);
  }

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
      // 创建 split root（lazy：mask=0，紧凑表为空，子节点按需创建）
      let split_idx = self.nodes.len();
      self.nodes.push(Node::Split {
        mask: 0,
        palette: cur_palette,
        children: Vec::new(),
      });
      node_idx = Some(split_idx);
    }

    let p = node_idx.unwrap();
    let child_extent = extent / BRICK_FACTOR;
    let ix = (x / child_extent).clamp(0, BRICK_FACTOR - 1);
    let iy = (y / child_extent).clamp(0, BRICK_FACTOR - 1);
    let iz = (z / child_extent).clamp(0, BRICK_FACTOR - 1);
    let child_i = child_linear_idx(ix, iy, iz) as u32;
    let next_x = x - ix * child_extent;
    let next_y = y - iy * child_extent;
    let next_z = z - iz * child_extent;

    // lazy split：mask bit=0 → 子块 uniform（= 本节点 palette），
    // 首次编辑才创建节点 + 置 bit + 紧凑表插入
    let (child_idx, parent_pal, cur_mask) = match &self.nodes[p] {
      Node::Split {
        mask,
        palette,
        children,
      } => {
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
    self.set_recursive(
      next_x,
      next_y,
      next_z,
      palette,
      Some(child_idx),
      child_extent,
    );

    // 回溯：try merge
    self.try_merge(p);
  }

  /// 把一个 uniform 节点变成 split 节点（lazy：mask=0 全 uniform，子节点按需创建）
  ///
  /// Douglas 语义（devlog #17）：mask bit=0 → uniform leaf（颜色 = 父节点
  /// palette_u32），**不占内存**。只有真正被编辑的子块才置 bit + 建节点。
  /// 旧实现（SPLIT_ALL + 64 个同色 Uniform 子节点）每 4³ 块浪费 65 节点
  /// （≈36KB），大场景编辑内存爆炸 ~65×。
  fn split_uniform(&mut self, idx: usize, old_palette: u8) {
    self.nodes[idx] = Node::Split {
      mask: 0,
      palette: old_palette,
      children: Vec::new(),
    };
  }

  fn try_merge(&mut self, idx: usize) {
    let (mask, palette) = match &self.nodes[idx] {
      Node::Split { mask, palette, .. } => (*mask, *palette),
      _ => return,
    };

    if mask == 0 {
      return;
    }

    // 阶段 1：只读扫描（零分配；旧实现 clone 64 槽 children Vec，
    // 每次 512B × 百万级调用 = 巨量分配流量拖慢编辑 + 内存高水位）
    let mut first_color: Option<u8> = None;
    let mut all_uniform_same = true;
    {
      let children = match &self.nodes[idx] {
        Node::Split { children, .. } => children,
        _ => unreachable!(),
      };
      for i in 0u32..64 {
        let bit = 1u64 << i;
        let c: Option<u8> = if (mask & bit) != 0 {
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

  /// GC：重建 nodes Vec 只保留 root 可达的有效节点，回收废弃索引
  ///
  /// 编辑（set_voxel/fill_brick/clear）过程中 split + try_merge 会留下被
  /// merge 掉的子节点（索引变废但仍在 Vec，容量只增不减）。长时间编辑后
  /// nodes 膨胀——本方法 DFS 标记可达节点，compact 到连续新 Vec，
  /// 重写 children 索引。O(n) 时间 + O(n) 临时空间，编辑完成后调一次即可。
  pub fn compact(&mut self) {
    if self.nodes.len() <= 1 {
      return; // 空 root 或单节点，无废弃
    }
    let mut new_nodes: Vec<Node> = Vec::with_capacity(self.nodes.len());
    let mut idx_map = vec![usize::MAX; self.nodes.len()];
    self.collect_reachable(Some(0), &mut new_nodes, &mut idx_map);
    // 重写 children 索引：old → new
    for n in new_nodes.iter_mut() {
      if let Node::Split { children, .. } = n {
        for c in children.iter_mut() {
          *c = idx_map[*c as usize] as u32;
        }
      }
    }
    self.nodes = new_nodes;
  }

  fn collect_reachable(&self, idx: Option<usize>, out: &mut Vec<Node>, map: &mut Vec<usize>) {
    let old = match idx {
      Some(i) => i,
      None => return,
    };
    if map[old] != usize::MAX {
      return; // 已访问（防环）
    }
    let new_idx = out.len();
    map[old] = new_idx;
    out.push(self.nodes[old].clone());
    if let Node::Split { children, .. } = &self.nodes[old] {
      for &c in children.iter() {
        self.collect_reachable(Some(c as usize), out, map);
      }
    }
  }

  // =========================================================================
  // 统计
  // =========================================================================

  pub fn node_count(&self) -> usize {
    self
      .nodes
      .iter()
      .filter(|n| matches!(n, Node::Split { .. }))
      .count()
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
    // palette word：低字节 = uniform 色，高字节 = lod（uniform 节点 lod = 自身色）
    let w = t.serialize()[2];
    assert_eq!(w & 0xFF, 42);
    assert_eq!((w >> 8) & 0xFF, 42);
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

  /// R3-10 DDGI：三态查询（get_uniform 的 None 无法区分 Air/Mixed，此处锁语义）
  #[test]
  fn brick_state_three_way() {
    // 空 chunk = Air（各级一致）
    let t = ChunkTree::empty();
    assert_eq!(t.get_brick_state(0, 0, 0, 2), BrickState::Air);
    assert_eq!(t.get_brick_state(16, 16, 16, 2), BrickState::Air);
    // 全 chunk uniform 实体：层级 0..4 全 Solid
    let t = ChunkTree::uniform(5);
    for level in 0..5u8 {
      assert_eq!(t.get_brick_state(0, 0, 0, level), BrickState::Solid(5));
    }
    // 16³ cell 内单体素：该 cell Mixed，邻 cell Air，64³ 父 brick 也 Mixed
    let mut t = ChunkTree::empty();
    t.set_voxel(5, 6, 7, 3);
    assert_eq!(t.get_brick_state(0, 0, 0, 2), BrickState::Mixed);
    assert_eq!(t.get_brick_state(16, 0, 0, 2), BrickState::Air);
    assert_eq!(t.get_brick_state(0, 0, 0, 1), BrickState::Mixed);
    // 4³ 子砖粒度：含体素的 4³ brick Mixed，全空 4³ brick Air
    assert_eq!(t.get_brick_state(4, 4, 4, 3), BrickState::Mixed);
    assert_eq!(t.get_brick_state(0, 0, 0, 3), BrickState::Air);
    // 整 4³ brick 填充：该 brick Solid，兄弟 Air
    t.fill_brick([16, 0, 0], 4, 9);
    assert_eq!(t.get_brick_state(16, 0, 0, 3), BrickState::Solid(9));
    assert_eq!(t.get_brick_state(0, 0, 0, 3), BrickState::Air);
    // 清掉后回到 Air（merge 生效）
    for dz in 0..4 {
      for dy in 0..4 {
        for dx in 0..4 {
          t.clear_voxel(16 + dx, dy, dz);
        }
      }
    }
    assert_eq!(t.get_brick_state(16, 0, 0, 3), BrickState::Air);
    // 256³ 全 chunk 填充（fill_brick 整 chunk 规范形）→ level 0 Solid
    t.fill_brick([0, 0, 0], 256, 4);
    assert_eq!(t.get_brick_state(0, 0, 0, 0), BrickState::Solid(4));
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
    // root 是 split：只有 (100,100,100) 所在 64³ 子块的 bit=1（lazy split，
    // mask bit=0 子块 = uniform AIR——Douglas #17 语义）
    let mask = (ser[1] as u64) << 32 | ser[0] as u64;
    assert_ne!(mask, 0);
    assert_eq!(mask.count_ones(), 1, "lazy split：单 bit 而非 SPLIT_ALL");
    // wire v2 混合格式：levels 0-2 紧凑（3+1 offset=4 words/层，单 child），
    // level 3 叶父层 inline palette（3+64=67 words，不递归到 level 4 叶节点）
    // 总 = 4 + 4 + 4 + 67 = 79 words
    assert_eq!(ser.len(), 79);
    // 序列化后读回语义不变
    assert_eq!(t.get_voxel(100, 100, 100), Some(7));
    assert_eq!(t.get_voxel(0, 0, 0), None);
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
