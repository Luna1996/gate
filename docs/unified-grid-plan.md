# 全量 1:1 复刻 Douglas Brick Tree 方案

> 状态：规划稿（v4 — chunk 256³ + 自适应叶子 + 分裂方向修正）。对标 Douglas #17/#18/#22/#23。
> 与 docs/brickmap.md（当前数据结构设计，换轨后标注废弃）、docs/todo/README.md（施工队列表）同级。

**v1（2026-09-02 早）**：统一 buffer + shader，保持 gate 数据结构 → 已推翻
**v2（2026-09-02 晚）**：全量换轨，错把分裂方向当合并方向 → 已推翻
**v3（2026-09-02 深夜）**：修正分裂方向，但 chunk 保持 128³ → 数学上无法得到 16³ 层
**v4（2026-09-03 凌晨）**：chunk 改为 256³，层级 256→64→16→4→1（4 层分裂），16³=level 2 天然对齐 DDGI probe 和 gate cell

---

## 0. Douglas Brick Tree 精确规格（Devlog #17/#18/#22/#23 原文还原）

### 0.1 核心确认（有原文证据）

| # | 架构点 | 原文来源 | 引用 |
|---|---|---|---|
| 1 | **分裂树**（不是合并树） | #17 / #18 | #17: "each parent Cube into 64 smaller cubes so each axis is **subdivided** by four"；#18: "tree is divided into 4x4x4 regions" |
| 2 | **分支因子 4³ = 64** | #17 / #22 | #17: "64 smaller cubes"；#22: "64 tree" |
| 3 | **Occupancy mask = u64** | #17 | "64-bit number where each bit represents whether the child node in this tree is **subdivided or not**" |
| 4 | **mask bit 语义** | #17 | bit=1 → 子块继续分裂（有自己的 u64 mask）；bit=0 → uniform leaf（空或实，palette 存父节点） |
| 5 | **DDA 零负载步进** | #17 | "mask 进寄存器 → bitwise AND 测试子块位 → popcount 定位子节点 offset → **最多 10+ 步零显存 load**" |
| 6 | **chunk 分层** | #23 | "readding the chunk system" |
| 7 | **统一 contree 格式** | #22 | "4x4x4 groups... I took my contree implementation from my old engine and I ripped out the per voxel normal data"（保留旧引擎树结构，只删法线） |
| 8 | **无 per-voxel normal** | #22 | "I didn't need the per voxel normal information... ripped out the per voxel normal data" |
| 9 | **DDGI BFS 向下** | #23 | "starts at the tree level of a single cell and then it **traverses downward** until it finds a totally empty leaf"（从某个层级分裂到叶子） |

### 0.2 chunk 大小：256³（v4 裁决）

Douglas 早期 devlog（#2/#3/#4/#7）明确用 256³ chunk。#17 换 Brick Tree 后没明确说，但 #23 DDGI probe cell = 16³，天然对齐 Brick Tree level 2 当且仅当 chunk = 256³。

**gate 当前 Tile = 128³ → 改为 256³**。这与 Douglas 早期 chunk 大小一致，且让 16³ 成为 level 2 天然层。

### 0.3 层级结构（分裂方向）

```
Chunk（256³ 体素 = 整个 Brick Tree 的体积）
│
├── Root node（Level 0）：整个 256³ 体积，u64 mask
│   bit i = 1 → 子块 i 继续分裂
│   bit i = 0 → uniform leaf（整个 64³ 子块同 palette 或全空）
│   子块索引：z*16 + y*4 + x（4³ → z*16 + y*4 + x 线性化）
│   子块边长：256 / 4^0 = 256（但 root 本身就是 256³，不分裂时整 chunk uniform）
│
├── Level 1：子块边长 = 256/4 = **64**
│   bit i = 1 → 子块继续分裂
│   bit i = 0 → uniform leaf（64³ brick，较大的 uniform 块）
│   体素数：262,144 / 子块
│
├── Level 2：子块边长 = 64/4 = **16** ← ★ DDGI probe cell 对齐层 + gate cell 对齐层
│   bit i = 1 → 子块继续分裂
│   bit i = 0 → uniform leaf（16³ brick = 4096 体素 = gate cell = DDGI probe cell）
│   体素数：4096 / 子块
│
├── Level 3：子块边长 = 16/4 = **4**
│   bit i = 1 → 子块继续分裂
│   bit i = 0 → uniform leaf（4³ brick = 64 体素）
│   体素数：64 / 子块
│
├── Level 4：子块边长 = 4/4 = **1** ← 最大分裂深度（叶子最小 = 1³）
│   mask bit 全 0（无法再分裂）
│
└── **没有固定最小叶子大小**
    uniform leaf 停在哪一层是自适应的：
    - 大平面 → 停在 Level 1（64³）或 Level 2（16³）
    - 小结构 → 分裂到 Level 3（4³）或 Level 4（1³）
    - 整块 chunk 空 → root mask = 0（整 chunk uniform 空，一个 u64 搞定）
```

**分裂深度 = ceil(log₄(256)) = 4 层**（256 → 64 → 16 → 4 → 1），和 Douglas 说的"最多 10+ 步零 load"一致（4 层分裂 + 每层 64 子块测试）。

### 0.4 GPU Brick Tree 存储格式（1:1 复刻）

```wgsl
// 固定部分（8 bytes = 2 words）
struct BrickTreeNodeFixed {
  mask: u64,          // 64-bit occupancy
  palette_u32: u32,   // mask=0 时 uniform palette; mask!=0 时第一个 child 在 struct_buf 内的下标起点
}

// 可变部分：当 mask != 0 时，紧接 popcount(mask) 个 u32 child offset
// 每个 child offset = 子节点在 struct_buf 内的下标

// child_offset 计算（纯寄存器操作）
fn brick_child_offset(parent_mask: u64, i: u32) -> u32 {
  let bit_i = 1u64 << i;
  // popcount 前 i bit 中的 1 → 子节点在紧凑表中的位置
  return countOneBits(parent_mask & (bit_i - 1u64));
}

// 给定父节点在 struct_buf 内的下标 + 子块索引 i，求子节点下标
fn brick_child_node_idx(parent_idx: u32, parent_mask: u64, i: u32) -> u32 {
  let child_slot = brick_child_offset(parent_mask, i);
  // 父节点 fixed 部分占 2 words，之后紧跟 popcount(parent_mask) 个 child offset
  let child_offset_word = 2u32 + child_slot;
  return parent_idx + child_offset_word;  // child offset 存在父节点 fixed 之后
}
```

**紧凑存储**：
- 空块（mask bit=0）**零开销**——不需要任何内存分配
- 只有被分裂的子块才占空间
- uniform leaf → 直接读父节点 palette_u32，**不需要额外 leaves buffer**

### 0.5 DDA 步进算法（1:1 复刻 Douglas #17/#18）

```wgsl
// 在 Brick Tree 某一层的某节点里，DDA 步进一格
fn bricktree_dda_step(
    cur_node_mask: u64,            // 当前节点 mask
    cur_node_idx: u32,             // 当前节点在 struct_buf 内的下标
    cur_brick_size: f32,           // 当前 brick 边长（体素）
    child_local_pos: vec3<i32>,    // 射线起点在当前 brick 内的 4³ 子块坐标（0..3 每轴）
) -> (hit_uniform: bool, next_node_idx: u32, child_brick_size: f32, palette: u32) {

  // 3D → 1D 子块索引
  let i = child_local_pos.z * 16u32 + child_local_pos.y * 4u32 + child_local_pos.x;
  let bit_i = 1u64 << i;

  if ((cur_node_mask & bit_i) == 0u64) {
    // 这个子块 uniform leaf → 直接返回 palette（在当前节点 fixed.palette_u32）
    return (hit_uniform = true, next_node_idx = 0u32,
            child_brick_size = cur_brick_size * 0.25,
            palette = 0u32);  // palette 由调用者从 cur_node_idx.fixed.palette_u32 取
  }

  // 这个子块被分裂 → 下钻到子节点
  let child_slot = countOneBits(cur_node_mask & (bit_i - 1u64));
  let child_node_idx = cur_node_idx + 2u32 + child_slot;
  return (hit_uniform = false, next_node_idx = child_node_idx,
          child_brick_size = cur_brick_size * 0.25,
          palette = 0u32);
}
```

**关键优势**：
- 每层**单次 storage load**（读 mask + palette_u32 = 8 bytes）
- mask 进寄存器后，64 子块的 occupancy 测试 = **纯 bitwise AND**
- popcount = **单 cycle 寄存器操作**（AMDGCN/Intel GPU）
- uniform leaf → palette 直接在当前节点 → **无额外 leaves buffer load**
- 零额外 load 步数 = 63 × 分裂层数（4 层）= **最多 252 步零 load**

---

## 1. 当前 gate vs Douglas Brick Tree（v4 最终）

| 维度 | gate v5 当前 | Douglas Brick Tree（v4） | 差异 |
|---|---|---|---|
| **chunk 大小** | 128³ | **256³** | chunk 变大 8 倍 |
| **树方向** | 分裂（root=Tile，down=cell→1³） | **分裂（root=Chunk256³，down=1³）** | 相同 |
| **分裂因子** | 2³（八叉树） | **4³（64-tree）** | 核心差异 |
| **层级** | 5 level（2³ 递归） | **4 level（4³ 递归）** | 不同深度 |
| **Occupancy** | TileBitmaps + CellDirs | **u64 mask per non-leaf** | 大幅简化 |
| **Uniform leaf 存储** | b_leaves（额外 buffer） | **palette 存父节点 fixed** | 删 leaves buffer |
| **空体素开销** | CellDirs 128KB 固定 | **mask bit=0 → 零开销** | 删 CellDirs |
| **DDA 链** | 五步链 Tile→Bitmap→Dirs→Node→Slot→Brick | **mask+palette 单次 load → bitwise AND** | 渲染核心 |
| **buffer** | 两组（BG0 + BG2） | **一组（所有 volume+chunk 共享）** | GPU 布局 |
| **组件 ID 粒度** | Cell（16³，Tile 内 level 2） | **BrickCoord{level:2}（256³ chunk 内 level 2 = 16³）** | 相同 ✅ |
| **DDGI 对齐** | N/A | **level 2 = 16³ = probe cell** | 天然对齐 |
| **VRAM** | ≤ 2.2GB | **≤ 350MB** | 省 ~1.85GB |
| **每步 load** | 2~5 | **1** | 渲染加速 |

---

## 2. 统一后的数据结构

### 2.1 Volume + Chunk 分层

```rust
/// 全场景 = Vec<Volume>
pub struct Volume {
  /// chunk 分层（保留 gate TileGrid 概念，chunk = 256³）
  pub chunks: HashMap<ChunkCoord, ChunkTree>,
  pub comp_layer: Option<CompLayer>,
  pub dirty: DirtyTracker,
  pub transform: Transform,  // 主世界 identity，物体任意变换

  /// GPU 侧 GridDesc
  pub grid_desc: GridDesc,
}

/// 主世界 = Volume[0]（chunks 无界 HashMap）
/// 独立物体 = Volume[1..N]（通常 1~少数几个 chunk）
```

### 2.2 ChunkTree（Douglas Brick Tree，1:1 复刻）

```rust
/// 每个 chunk（256³ 体素）内的 Brick Tree
/// 分裂树：root=256³ → level1=64³ → level2=16³ → level3=4³ → level4=1³
pub struct ChunkTree {
  /// DFS 序紧凑 BrickTreeNode 数组
  /// 每个节点 = 2 words fixed + popcount(mask) words child offset
  pub nodes: Vec<u32>,
  /// 256 条 palette
  pub palette: Vec<u32>,
}

/// u64 mask + palette 是核心。child offset 表紧跟 fixed 部分之后。
/// mask = 0 → uniform leaf，palette_u32 有效
/// mask != 0 → 分裂节点，child offset 表紧跟

impl ChunkTree {
  /// chunk 边长（固定 256）
  pub const CHUNK_SIZE: u32 = 256;
  /// 分裂因子（固定 4）
  pub const BRICK_FACTOR: u32 = 4;
  /// 最大分裂深度 = ceil(log₄(256)) = 4
  pub const MAX_DEPTH: u32 = 4;

  /// uniform leaf 可能的体素边长
  /// level 0 = 256（整 chunk uniform）
  /// level 1 = 64
  /// level 2 = 16 ← gate cell 对齐
  /// level 3 = 4
  /// level 4 = 1（最细）

  /// 给定父节点 mask 和子块索引 i（0..63），求紧凑 child offset
  pub fn child_slot(mask: u64, i: u32) -> u32 {
    let bit_i = 1u64 << i;
    (mask & (bit_i - 1u64)).count_ones()
  }

  /// 给定父节点下标 + mask + 子块索引，求子节点在 nodes 数组内的下标
  pub fn child_node_idx(parent_idx: u32, parent_mask: u64, i: u32) -> u32 {
    let child_slot = Self::child_slot(parent_mask, i);
    parent_idx + 2 + child_slot  // fixed 2 words + child_slot
  }
}
```

### 2.3 坐标体系

```rust
/// Brick Tree 内 brick 的坐标
pub struct BrickCoord {
  /// 分裂深度（0 = root=256³，1=64³，2=16³，3=4³，4=1³）
  pub level: u8,
  pub chunk: ChunkCoord,
  pub x: u32, pub y: u32, pub z: u32,  // 该层级 brick 在 chunk 内的索引
}

impl BrickCoord {
  /// brick 边长（体素）= CHUNK_SIZE / 4^level
  pub fn size(&self) -> u32 {
    256 >> (self.level * 2)  // 2^8 / 2^(level*2) = 2^(8 - 2*level)
  }
}

/// 体素坐标（世界空间）
pub struct WorldCoord {
  pub x: i32, pub y: i32, pub z: i32,
}

impl WorldCoord {
  /// 映射到 chunk 坐标（256³）
  pub fn chunk(&self) -> ChunkCoord {
    ChunkCoord {
      x: self.x.div_euclid(256),
      y: self.y.div_euclid(256),
      z: self.z.div_euclid(256),
    }
  }

  /// 映射到 chunk 内 level 2 brick（16³ = gate cell = DDGI probe cell）
  pub fn cell_brick(&self) -> BrickCoord {
    let chunk = self.chunk();
    let local = (IVec3::new(self.x, self.y, self.z) % 256).abs();
    BrickCoord {
      level: 2,
      chunk,
      x: (local.x as u32) >> 4,
      y: (local.y as u32) >> 4,
      z: (local.z as u32) >> 4,
    }
  }
}
```

### 2.4 组件 ID 层（挂 BrickCoord{level:2}）

gate cell = 16³ = Brick Tree level 2 = 天然对齐。

```rust
pub struct CompLayer {
  pub components: HashMap<BrickCoord, ComponentInstance>,
}
```

u16 per 16³ brick。洪泛算法 DFS on level 2 brick 间连通边。和 gate 当前 cell 粒度完全相同，**算法零改动**，只是 key 类型从 `CellCoord` 换成 `BrickCoord{level:2}`。

### 2.5 GPU struct buffer 格式

**单一 buffer**（所有 volume + chunk 共享）：

```
struct_buf：
├── 段 0: GridDesc 数组（N × 128B，N = volume 数）
├── 段 1: Brick Tree nodes（所有 chunk 的 nodes Vec 按 DFS 序拼接）
│         每个 node = 2 words fixed + popcount(mask) words child offset
└── 段 2: palette（所有 volume 的 palette 数组，256 × 4B / volume）
```

**删除的 buffer**：TileIndex（8MB）、TileBitmaps（4MB）、CellDirs（128MB）、b_leaves（~1GB）。

### 2.6 GridDesc（128B）

```rust
#[repr(C)]
#[derive(ShaderType, Clone, Copy)]
pub struct GridDesc {
  // ---- 变换（Mat4 列向量，64B）----
  pub pos_scale: Vec4,   // xyz = 世界原点，w = scale
  pub rot0: Vec4,        // 世界→局部（Mat3 列向量，第四分量 = 0）
  pub rot1: Vec4,
  pub rot2: Vec4,

  // ---- 世界 AABB（32B）----
  pub aabb_min: Vec4,
  pub aabb_max: Vec4,

  // ---- Brick Tree 数据基址（16B）----
  pub tree_base: u32,       // 本 volume 第一个 chunk 的 Brick Tree 根节点下标
  pub tree_depth: u32,      // 最大分裂深度 = 4（所有 chunk 相同）
  pub chunk_count: u32,     // 本 volume 的 chunk 数量
  pub palette_base: u32,    // palette 在 struct_buf 内的字偏移

  // ---- 预留（16B）----
  pub _pad: [u32; 4],
}
```

---

## 3. DDA 主循环（1:1 复刻 Douglas）

```wgsl
struct Grid {
  pos_scale: vec4<f32>,
  rot0: vec4<f32>, rot1: vec4<f32>, rot2: vec4<f32>,
  aabb_min: vec3<f32>, aabb_max: vec3<f32>,
  tree_base: u32,
  tree_depth: u32,
  palette_base: u32,
  obj_id: i32,
}

fn trace_brick_tree(g: Grid, ray_origin: vec3<f32>, ray_dir: vec3<f32>) -> UniformHit {
  // 1. 顶层 AABB slab（世界空间 → 局部空间 → 与 AABB 求交）
  // ...（和当前 trace_scene slab 相同）
  if (!hit_aabb) { return UniformHit(false, 0u32); }

  // 2. 遍历每个 chunk 的 Brick Tree
  var cur_node_idx: u32 = g.tree_base;
  var cur_level: u32 = 0u32;
  var cur_brick_size: f32 = 256.0;  // chunk 边长 = 256³

  while (cur_level <= g.tree_depth) {
    // 单次 storage load：读当前节点 fixed（8 bytes = 2 words）
    let mask_u32_0 = g.tree_buf[cur_node_idx];
    let mask_u32_1 = g.tree_buf[cur_node_idx + 1u32];  // palette_u32

    let mask: u64 = bitcast<u64>(vec2<u32>(mask_u32_0, mask_u32_1));
    let palette: u32 = g.tree_buf[cur_node_idx + 2u32];  // fixed 第 3 个字 = palette_u32

    if (mask == 0u64) {
      // uniform leaf：整个 brick 同 palette
      // 如果是背景（palette == 0），表示全空，跳过
      if (palette == 0u32) {
        // 空体素 → 继续 DDA
      }
      return UniformHit(true, palette);
    }

    // mask != 0：有子块被分裂
    let child_size: f32 = cur_brick_size * 0.25;

    // DDA 步进：在当前 brick 内沿射线前进
    // 4³ = 64 个子块，每个子块是 child_size 边长的立方体
    // 子块索引 = z*16 + y*4 + x
    // DDA 步进算法：Douglas #18 描述的 ray-AABB per-child stepping
    // （和 gate 当前 trace_scene slab 的 DDA 法相同）
    //
    // 关键优化：mask 进寄存器后，64 个子块的 occupancy 测试 = 纯 bitwise AND
    // 零额外 storage load

    // 计算射线在当前 brick 内的起点和 DDA 参数
    let (t_x, t_y, t_z, cur_pos) = dda_setup(cur_brick_size, ray_origin, ray_dir);

    // 主步进循环（最多 64 步 = 每层每个子块测试）
    var hit_next_level = false;
    while (step < 64u32) {
      let i = cur_pos.z * 16u32 + cur_pos.y * 4u32 + cur_pos.x;
      let bit_i = 1u64 << i;

      if ((mask & bit_i) == 0u64) {
        // uniform 子块 → 检查射线是否与该子块 AABB 相交
        if (dda_hit_child(cur_pos, t_x, t_y, t_z, child_size)) {
          // 命中！这个 uniform 子块的 palette = 当前节点的 palette_u32
          if (palette != 0u32) {
            return UniformHit(true, palette);
          }
        }
      } else {
        // 这个子块被分裂 → 下钻
        let child_slot = countOneBits(mask & (bit_i - 1u64));
        cur_node_idx = cur_node_idx + 3u32 + child_slot;  // fixed 3 words + child_slot
        cur_level += 1u32;
        cur_brick_size = child_size;
        hit_next_level = true;
        break;  // 跳到外层 while 继续
      }

      // DDA 步进到下一个子块
      dda_step(&mut cur_pos, &mut t_x, &mut t_y, &mut t_z, &mut step);
    }

    if (!hit_next_level) { break; }
  }

  return UniformHit(false, 0u32);
}
```

---

## 4. 电路组件迁移（零改动）

| 项 | gate v5 | Douglas Brick Tree v4 |
|---|---|---|
| 组件 ID 粒度 | Cell = 16³ | **BrickCoord{level:2} = 16³** ✅ 相同 |
| comp_layer key | CellCoord | **BrickCoord{level:2}**（类型变，粒度同） |
| StateTable | 挂 TileGrid | 挂 Volume |
| 洪泛算法 | DFS on Cell 连通边 | **DFS on level 2 brick 连通边**（算法零改动） |

**对齐链条**：chunk 256³ → level 2 = 16³ → gate cell = 16³ → DDGI probe cell = 16³。

---

## 5. Phase 规划

### Phase 0：ChunkTree + 坐标 + 编辑 API（gate-voxel 重写）

**目标**：替换 Tile 内可变叶八叉树为 Douglas Brick Tree（分裂树 + u64 mask + uniform leaf）。chunk 从 128³ → 256³。

| 文件 | 改动 | 说明 |
|---|---|---|
| `gate-voxel/src/coords.rs` | **重写** | ChunkCoord（256³）+ BrickCoord + WorldCoord；删除 CellCoord |
| `gate-voxel/src/brick.rs` | **删除** | Cell + Slot + BrickPool（~800 行） |
| `gate-voxel/src/tile.rs` | **删除** | Tile 内部结构；TileCoord 概念合并进 ChunkCoord |
| `gate-voxel/src/grid.rs` | **重写为 volume.rs** | TileGrid → Volume（chunks: HashMap<ChunkCoord, ChunkTree>） |
| `gate-voxel/src/chunk_tree.rs` | **新增** | ChunkTree + DFS build + 编辑 API（体素 → 分裂 uniform leaf → 自底向上合并） |
| `gate-voxel/src/volume.rs` | **新增** | Volume 容器：chunks + comp_layer + dirty + transform |

**验收**：
1. ChunkTree 单元测试：build_full（uniform/稀疏/多材质）→ DFS 序列化正确
2. 编辑测试：修改 uniform leaf → 分裂 → 正确分裂出子节点 → 自底向上 merge
3. 坐标测试：WorldCoord → ChunkCoord → BrickCoord{level:2} 正确映射

**风险**：高。分裂树编辑 API（体素级修改 → 分裂 uniform leaf → 向上 merge）比 gate 当前 cell 编辑更复杂。

### Phase 1：GPU struct 打包 + GridDesc

**目标**：BrickTree → struct buffer（紧凑 child offset）+ GridDesc 数组。

| 文件 | 改动 |
|---|---|
| `gate-render/src/brickmap/builder.rs` | **重写**：ChunkTree.nodes Vec<u32> 直接上传（不需要二次打包，已经是 DFS 序紧凑格式） |
| `gate-render/src/brickmap/wire.rs` | **重写**：常量 CHUNK_SIZE=256, BRICK_FACTOR=4, MAX_DEPTH=4 |
| `gate-render/src/brickmap/globals.rs` | **重写**：BrickMapGlobals → GridDesc |

**验收**：CPU 构建的 ChunkTree 序列化 → GPU 上读回 mask + child offset → 正确还原 DFS 树结构。

### Phase 2：Shader Brick Tree DDA + 统一入口

**目标**：WGSL 彻底改写为 Douglas mask DDA。

| 文件 | 改动 |
|---|---|
| `dda.wgsl` | **最大重写点**：五步链 → Douglas mask DDA（§3 算法）；删除 cell_occupied / sample_brickmap / obj_cell_occupied / obj_sample_voxel |
| `gate-render/src/brickmap/dda.rs` | **重写**：trace_scene → 遍历所有 volume GridDesc → 每个 volume 内 Brick Tree DDA |
| `gate-render/src/brickmap/obj.rs` | **删除** |

**验收**：DDA 等价性测试（world_only / occlusion / rotation / scale）全绿；帧预算比五步链快。

### Phase 3：上传管道泛化

**目标**：统一 dirty → 增量上传。

| 文件 | 改动 |
|---|---|
| `upload.rs` | **重写**：三段管道从 TileGrid 依赖 → Volume dirty 依赖 |
| `plugin.rs` | ObjPlugin → VolumePlugin |
| `mod.rs` | BrickMap + ObjPool → VolumeRenderModule |

**验收**：主世界编辑 / 物体编辑 → 同一路径 → 正确渲染更新。

---

## 6. 内存预算

### 6.1 gate v5 当前

| 项 | 尺寸 |
|---|---|
| TileIndex | 8MB |
| TileBitmaps | 4MB |
| CellDirs | 128MB |
| NodeStream | ≤ 1GB |
| b_leaves | ≤ 1GB |
| **合计** | **≤ 2.2GB** |

### 6.2 Douglas Brick Tree v4

| 项 | 尺寸 | 说明 |
|---|---|---|
| TileIndex | **删** | Brick Tree mask 取代 |
| TileBitmaps + CellDirs | **删** | mask + popcount 取代 |
| b_leaves | **删** | palette 直接存 BrickTreeNode fixed |
| Brick Tree nodes | ~200MB | **紧凑 child offset**，只存 mask 非零子块 |
| GridDesc | N × 128B | N=1000 → 128KB |
| palette | (N+1) × 1KB | |
| **合计** | **≤ 350MB** | |

**省 ~1.85GB VRAM**，给 DDGI（~500MB）+ G-Buffer（~300MB）腾空间。

---

## 7. 替换清单

### 新增
- `gate-voxel/src/chunk_tree.rs` — ChunkTree + DFS build + 编辑 API
- `gate-voxel/src/volume.rs` — Volume 容器

### 删除
- `gate-voxel/src/brick.rs` — Cell/Slot/BrickPool
- `gate-voxel/src/tile.rs` — Tile 内部结构
- `gate-render/src/brickmap/obj.rs` — ObjPool + ObjDesc
- `docs/brickmap.md` — Phase 0 后标注废弃

### 重写
- `gate-voxel/src/coords.rs` — ChunkCoord + BrickCoord + WorldCoord
- `gate-voxel/src/grid.rs` → 合并进 volume.rs
- `gate-render/src/brickmap/builder.rs` — 直接上传 ChunkTree.nodes
- `gate-render/src/brickmap/upload.rs` — Volume dirty 依赖
- `gate-render/src/brickmap/dda.rs` — trace_scene 入口
- `gate-render/src/brickmap/wire.rs` — 常量
- `gate-render/src/brickmap/globals.rs` — GridDesc
- `dda.wgsl` — **最大重写点**

---

## 8. 决策记录

| # | 决策 | 日期 |
|---|---|---|
| 1 | 分裂树（root=chunk大 → 叶子小），分裂因子 4³ = 64 | 2026-09-03 |
| 2 | chunk = **256³**（Douglas 早期 chunk 大小，DDGI probe cell 对齐 level 2 = 16³） | 2026-09-03 |
| 3 | 最大分裂深度 = 4 层（256→64→16→4→1） | 2026-09-03 |
| 4 | uniform leaf 自适应（无固定最小叶子大小） | 2026-09-03 |
| 5 | u64 mask per non-leaf，child offset 紧凑（只存被分裂子块） | 2026-09-03 |
| 6 | Tile 分层保留（chunk = 256³ = 旧 Tile 8 倍大） | 2026-09-03 |
| 7 | 组件 ID 挂 BrickCoord{level:2}（16³ = gate cell） | 2026-09-03 |
| 8 | b_leaves / CellDirs / TileBitmaps / TileIndex 全部删除 | 2026-09-03 |
| 9 | shader 统一 GridDesc + 无 kind 分支 | 2026-09-03 |
| 10 | 编辑精度 = 1³（分裂到 level 4），组件粒度 = 16³（level 2） | 2026-09-03 |
