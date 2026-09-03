# Phase 0 工作日志 — gate-voxel crate Brick Tree 重写

**日期**: 2026-09-03
**状态**: ✅ Phase 0 + Phase 1 完成（workspace 114 tests 全绿）

---

## 目标

1:1 复刻 Douglas Brick Tree 架构（分裂树 + 4³ 分裂因子 + u64 mask + 紧凑 child offset + DFS 紧凑序列化），替换 gate 旧的可变叶八叉树。

---

## 文件清单

### gate-voxel crate（Phase 0 边界，纯逻辑，唯一外部依赖 glam）

| 文件 | 操作 | 说明 |
|------|------|------|
| `src/coords.rs` | **重写** | ChunkCoord(Vec3 包装+HashMap key)、VoxelCoord(1³ world i32+chunk/in_chunk/to_level_brick)、BrickCoord(level+chunk+xyz)、常量 CHUNK_SIZE=256/BRICK_FACTOR=4/MAX_LEVEL=4/LEVEL_EXTENT=[256,64,16,4,1] |
| `src/chunk_tree.rs` | **新增** | Douglas Brick Tree 核心实现（结构化 Node 编辑 + DFS 紧凑序列化） |
| `src/volume.rs` | **新增** | VolumeGrid 主世界容器（ChunkCoord→ChunkTree HashMap + comp_layer + state_table + DirtyTracker 集成） |
| `src/dirty.rs` | **重写** | DirtyTracker 改用 ChunkCoord + FIFO 预算队列（mark_data/mark_comp/drain_budget） |
| `src/scene.rs` | **重写** | fill_box/fill_sphere/draw_text 改用 VolumeGrid |
| `src/lib.rs` | **重写** | 导出 ChunkCoord/VoxelCoord/BrickCoord/ChunkTree/VolumeGrid/DirtyTracker/Palette/fill_box 等 |
| `src/tile.rs` | **删除** | 旧 Cell/Slot/Brick/Tile（~830 行） |
| `src/grid.rs` | **删除** | 旧 TileGrid（227 行） |
| `src/stress.rs` | **删除** | 旧极限测试（全引用 Tile/TileGrid） |

### 文档

| 文件 | 操作 | 说明 |
|------|------|------|
| `docs/unified-grid-plan.md` | **新增** | Phase 0 v4 最终规划主文档 |
| `docs/decisions.md` | **更新** | ADR-0007 v4 版 |
| `docs/todo/` | **新增** | Phase 0 任务分解 |
| `docs/brickmap.md` | **更新** | 标注旧数据结构待废弃 |

### 其他 crate（已改但**不编译**，旧 API 引用待 Phase 1）

gate-render 引用了已删除的旧类型：
- `gate_voxel::{Brick, Cell, Slot, Tile, TileCoord, TileGrid, VoxelPos, MAX_LEVEL, SUB_PER_CELL}`
- 文件：`brickmap/builder.rs`, `brickmap/view.rs`, `brickmap/upload.rs`, `brickmap/dda.rs`, `lighting.rs`, `brickmap/obj.rs`

gate-app 也有改动但不在 Phase 0 边界内。

---

## ChunkTree 核心设计

### 双层架构

```
编辑层（结构化）                  序列化层（紧凑 GPU buffer）
Node::Uniform(u8)            →  [mask_lo, mask_hi, palette]  (3 words)
Node::Split {                →  [mask_lo, mask_hi, palette, child_offsets...]
  mask: u64,                       (3 + popcount(mask) words)
  palette: u8,                    child_offsets[i] = child node 绝对下标
  children: Vec<Option<usize>>   DFS 顺序 flatten
}
```

### 为什么双层？

紧凑 DFS 格式在**编辑时无法原地修改**：child offset 表位置偏移会连锁破坏上层引用。所以编辑用结构化 Node enum（可变、可 clone、popcount mask 直接定位 children slot），每次变更后重新 DFS flatten 成 GPU buffer。

Phase 0 简化：编辑只改结构化节点，serialize() 只在上传 GPU 前调用。性能开销之后再优化（可加脏标记延迟 serialize）。

### Node enum

```rust
enum Node {
  Uniform(u8),                    // mask=0 → 整 brick 同一 palette
  Split {                         // mask!=0 → 分裂节点
    mask: u64,                    // bit=1 → 子块分裂有 child node; bit=0 → uniform(= parent palette)
    palette: u8,                  // uniform 子块默认色 + GPU DDA 零负载步进用
    children: Vec<Option<usize>>, // 64 slots，mask bit=1 时指向 child node
  },
}
```

### root 特殊处理

ChunkTree 有 `root_palette: u8` + `nodes: Vec<Node>` 两个字段：
- **nodes 为空** = root uniform，颜色 = root_palette（节省一个 Uniform 节点）
- **nodes 非空** = nodes[0] 是 root Split 或 Uniform

merge 到 root 时：`nodes.clear()` + 更新 root_palette。

### set_voxel 流程

```
1. get_voxel 只读检查 → 颜色未变 return false
2. set_recursive(idx=None, extent=256):
   a. nodes 为空 → 创建 Split root + 64 Uniform(cur_palette) 子节点，node_idx=Some(split_idx)
   b. nodes[idx] 是 Uniform → split_uniform(idx, old_palette)：
      改写为 Split{mask=SPLIT_ALL, palette=old_palette} + append 64 Uniform(old_palette)
   c. 计算 child_i = child_linear_idx(ix,iy,iz)
   d. children[child_i] 可能是 None（mask bit=0）或 Some(child_idx)
      - None → 先 split_uniform（父节点 Split 的 children[child_i] 从 None → Split{...} → append 64 Uniform）
      - Some(child_idx) → 直接下钻
   e. 递归 set_recursive(child_idx, child_extent)
   f. 回溯：try_merge(idx) —— 所有 children 是 Uniform 且同色 → merge 成 Uniform
```

### try_merge 逻辑

```
for each of 64 child slot i:
  mask bit=1 → 读 children[i]，必须 Uniform，颜色 = Uniform.palette
  mask bit=0 → 颜色 = Split.palette（uniform 子块默认色）
  颜色全部相同且无 Split child → all_uniform_same = true

all_uniform_same → merge:
  idx == 0 → nodes.clear(), root_palette = merged_color
  其他 → nodes[idx] = Uniform(merged_color)
```

### 关键约束

- **split_uniform_leaf 总是 SPLIT_ALL**：简化编辑逻辑，消除 mask bit=0 分裂子块的歧义
- **内存只增不减**：merge 只改上层 Split → Uniform，被 merge 的子节点还在 nodes Vec 里（索引变废）
- **serialize() DFS flatten**：从 root 开始，mask bit=1 的 child 递归写入并记录绝对下标到 offset 表

---

## 测试覆盖（31 tests 全绿）

### coords（6 tests）
- chunk_coord_positive / chunk_coord_negative（ChunkCoord 从 VoxelCoord chunk() 计算）
- level_extent_chain（256→64→16→4→1）
- level_2_brick_aligns（16³ = 组件粒度）
- level_4_voxel_aligns（1³ 体素精度）
- child_linear_idx_layout（(x,y,z) → z*16+y*4+x）

### chunk_tree（11 tests）
- empty_chunk_is_uniform_air（nodes=Vec::new(), root_palette=0）
- uniform_chunk_readback（nodes=Vec::new(), root_palette=42）
- set_single_voxel_readback（分裂 4 层到 level 4 Uniform）
- set_same_color_is_noop（只读前置检查）
- clear_voxel_reads_none（写→清→读 None）
- multiple_voxels_different_colors（多点不同色共存）
- merge_after_clear_all_children（64 体素→清光→root merge uniform）
- uniform_query_level_2（16³ brick 全同色）
- empty_uniform_returns_none
- node_layout_roundtrip（serialize 输出 3 + 64 words + child offsets）
- large_uniform_area_stays_coarse（64³ 同色区域 merge 到 level 1/2）

### dirty（2 tests）
- dedup_and_fifo
- data_and_comp_independent

### palette（3 tests）
- air_slot_is_reserved / entry_size_is_8_bytes / flags_compose

### scene（4 tests）
- box_readback / box_cross_chunk / sphere_center_in_solid / text_glyph_count

### volume（5 tests）
- set_get_single_voxel / cross_chunk_editing / negative_coords_cross_chunk / clear_removes_chunk_when_empty / batch_edit_dedupes_dirty

---

## Level 链

```
chunk = 256³ → level 0 (extent=256)
brick = 64³  → level 1 (extent=64)
brick = 16³  → level 2 (extent=16) ← DDGI probe cell / 组件 cell 天然对齐
brick = 4³   → level 3 (extent=4)
brick = 1³   → level 4 (extent=1) ← 编辑精度
```

分裂因子 4³=64，4 次分裂到 1³。

---

## 序列化 buffer 布局（GPU 上传）

```
nodes[i]       = mask low 32 bits
nodes[i + 1]   = mask high 32 bits  → u64 = occupancy mask
nodes[i + 2]   = palette_u32        (mask=0: uniform color; mask!=0: uniform 子块默认色 + DDA 用)
nodes[i+3 .. +3+popcount(mask)] = child_offsets[] (绝对下标，紧凑只存 mask bit=1 的子块)
```

GPU DDA 步进：
```
mask = (mask_hi << 32) | mask_lo       // 进寄存器
bit  = 1u64 << child_idx               // 定位子块
if (mask & bit) == 0 → palette_u32     // uniform 子块，零额外 load
else child_offset = nodes[idx + 3 + popcount(mask & (bit-1))]  // 单 cycle popcount
```

---

## 待办 Phase 1+

1. ~~**gate-render/gate-app 适配**：把 TileGrid/Brick/Cell/Slot/VoxelPos 等旧 API 全部替换为 VolumeGrid/ChunkCoord/VoxelCoord/ChunkTree.serialize()~~ ✅ Phase 1 完成
2. **GPU struct buffer 上传管道**：DirtyTracker.drain_data_budget → serialize 成 Vec<u32> → writeBuffer + indirect draw
3. **DDGI probe cell pipeline**：level 2 brick（16³）天然对齐 probe cell
4. **内存 GC/compaction**：Phase 0 split 只增不减，长时间编辑后 nodes Vec 会膨胀
5. **性能优化**：每次 set_voxel 后全量 DFS serialize 太奢侈，可加脏标记延迟

---

## Phase 1 适配总结（2026-09-03）

### 已迁移文件
- `gate-render/src/brickmap/wire.rs`：b_struct 布局重写（Region ① chunk 窗口 1MB + Region ② DFS 树 append，b_leaves 删除保留空 Vec）
- `gate-render/src/brickmap/view.rs`：BrickMapView mask DDA 软件遍历器（独立寻址链，与 builder 互为对照）
- `gate-render/src/brickmap/builder.rs`：VolumeGrid→serialize append + 增量 update_chunk + DirtyRanges
- `gate-render/src/brickmap/upload.rs`：VoxelScene/Mirror/poll/extract/prepare 迁移
- `gate-render/src/brickmap/obj.rs`：pack_obj_pool + rebase_obj_window（物体局部空间 [0,256)³）+ cpu_reference_object_ray 适配
- `gate-render/src/brickmap/dda.rs`：wgsl_consts 新镜像 + 测试适配
- `gate-render/src/lighting.rs`：测试 fill_bricks 替换 fill_box（修 160MB 崩溃）
- `gate-render/tests/p29_limits.rs`：63 chunk 替换 64（compute_window 1 chunk 边距使 64 连续 chunk 必丢 1）
- `gate-app/src/main.rs`：TileGrid→VolumeGrid、fill_box level 参数删除、大体积改 fill_bricks

### 关键修复
- **obj.rs 窗口条目 rebase bug**：世界 builder 的窗口带 1 chunk 生长边距（origin=min-1），1-chunk 物体的窗口条目落在 ip=(1,1,1) 而非 0。`rebase_obj_window` 打包时把条目搬到 origin=(0,0,0) 视角（树区绝对字址不变，纯搬家）
- **fill_box 内存爆炸**：512³ per-voxel 分裂 = 1.34亿体素 × 65 节点 → 160MB+ 崩溃。`fill_bricks(grid, min, extent, e=16, palette)` brick 级树路径写入，零逐体素分裂开销
- **p29_limits 窗口边距**：compute_window 固定留 1 chunk 生长边距，64 个连续 chunk 必丢 1 → 改 63 chunk（百万级口径仍满足：2.06M）

### 测试统计
- gate-voxel lib: 31 passed
- gate-render lib: 46 passed
- gate-render p29_limits: 7 passed
- gate-app: 1 passed
- gate-render 其他测试: 29 passed
- **合计: 114 passed / 0 failed**

---

## CPU 编辑层性能约束（2026-09-03 确立，违反即启动卡死/内存爆炸）

N=10 场景构建 78s 卡死 + 5.9GB OOM → 全部修复后 N=10 实测：1264 chunks 场景 53.4s 构建 / 树输出 247MB / build_full 415ms / 稳定渲染：

1. **lazy split 是 Douglas 语义的一部分**（devlog #17）：split_uniform 必须 `Split { mask: 0, children 全 None }`，编辑落到哪个子块才置 bit + 建 1 节点。禁止「SPLIT_ALL + 64 个同色 Uniform 子节点」写法（每 4³ 块 65 节点 ≈ 36KB，65× 内存浪费）；`first_uniform_color` 需防御 mask=0（trailing_zeros(0)=64 越界）
2. **Node 用紧凑 child 表（与 GPU wire 同构）**：`children: Vec<u32>` 只存 mask bit=1 的子块（`child_slot(mask,i) = popcount(mask & (bit-1))` 定位，O(1)），禁用 `Vec<Option<usize>>` 64 槽（552B/节点 → 40B，14×）。插入用 `children.insert(slot, idx)`（壳区域槽位少，O(n) 可忽略）
3. **编辑热路径禁止 clone children Vec**：try_merge 用两阶段（只读扫描 + 可变写），每次 clone 64 槽（512B）× 百万级调用 = 巨量分配流量
4. **大体积球/盒填充必须 4³ 分块**：fill_sphere/fill_box 整块命中走 fill_brick（O(depth)），仅边缘壳逐体素；纯逐体素 set_voxel 对大球是数亿次 O(depth) 调用
5. **场景尺寸参数必须 4 对齐**：terrain_h 输出 `& !3`、snow_line=40、河床层高 4 的倍数——非对齐高度使每列顶部边缘块退化为逐体素 1³ 编辑 → 4³ Split 碎片化，树序列化输出 1935MB → 247MB（8×）。阶梯化 4 级符合像素风
6. **demo 场景坐标必须随 EXT_FINE 缩放校验**：世界外的循环体素/簇会导致 placed 永不达标 → 计数器 i32 溢出 panic（已加簇中心超界 continue）

分节耗时（N=10，5120²）：地形 47.5s / 浮空岛 5.4s / 其余 <0.6s / compact_all 2.7s / build_full 0.42s。剩余大头是地形列循环的 400 万级 fill_brick 调用（CPU 编辑层 40B/节点紧凑表已达标；再优化需 batch/延迟 serialize，属 Phase 4+ 范畴）。

---

## 已知限制

- Phase 0 serialize() 每次调用全量 DFS flatten——编辑频繁但 serialize 很少（只在上传 GPU 前调用），实际开销可控
- CPU 编辑层 Node ≈ 552B/节点（children: Vec<Option<usize>> 64×8B），约为 GPU wire 格式（3 words + 4B/child）的 35×。当前量级可接受；若后期 CPU 侧内存成为瓶颈可压缩 children 布局
- 无 per-voxel normal（Douglas #22 也没了，符合 1:1 复刻）
- VRAM 从 ~2GB 降到 ~350MB（TileIndex/CellDirs/TileBitmaps/b_leaves 全删）

---

## 明天继续的切入点

打开 gate-render/src/brickmap/upload.rs 第 84 行（`pub grid: gate_voxel::TileGrid`），开始把 upload mirror 从 TileGrid/TileCoord 迁移到 VolumeGrid/ChunkCoord。ChunkTree.serialize() 已经能输出正确的紧凑 GPU buffer。
