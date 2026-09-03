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

（已完成，切入点保留备查）upload mirror 迁移 VolumeGrid/ChunkCoord 已随 Phase 1 结束。

---

## 帧率根因与层次栈式 mask DDA（2026-09-03）

**现象**：Brick Tree 上屏后 200+ fps → ~30 fps（6.7× 下降）。

**根因（数 storage load 账定位，非架构/正确性问题）**：移植期遍历器把 Brick Tree 当**点查询结构**用——cell_occupied 每个 16³ 粗步从根重走 DFS（~9 load/粗步），sample_brickmap 每细步从根走 4 层（9-13 load/步）；旧五步链位图粗步仅 2 load。天空射线跨 5632 世界 ≈ 352 粗步：旧 ~700 load vs 新 ~3170 load，综合 ~6× 与实测吻合。

**修法（WGSL + CPU 逐字镜像）**：层次栈式 mask DDA（devlog #17/#18 遍历范式）：
- chunk 间 256³ A&W（ci=floor(start/256)，delta_c=delta*256，窗口外 chunk=空气直接步进，空 chunk entry=0 仅 1 load 跨越；chunk budget=(dx+dy+dz)*3+16）
- chunk 内 4 层栈帧（256→64→16→4→1）：节点 mask 一次 load 进寄存器，4³=64 子块间 A&W 步进只查 bit（零 load）；bit=0 uniform 子块整格跳过/整格命中（palette 直决）；bit=1 popcount 定位 child offset 压栈下钻；level 3 分裂读 leaf palette；t_out>=t_exit 弹栈
- 栈帧字段：node_addr/node_min/cell(0..3)³/tmax 三轴/t_enter/t_exit/face_id；`sub = 64 >> (level*2)`
- child 定位：child_idx=cz*16+cy*4+cx，mask 拆 lo/hi 两 u32，slot=countOneBits(低于 idx 位)
- face_id 0..5=±xyz（穿入面）；face_normal_from_index：0→-X,1→+X,2→-Y,3→+Y,4→-Z,5→+Z
- **t 标尺**：trace_chunk/trace_volume_tree 内全部 ro 系绝对 t（不用相对标尺）；物体路径 ro/rd 已烘焙 1/scale（WGSL `/ g.scale`，CPU `/ tr.scale`），局部射线参数 = 世界射线参数，物体 ray 直接调 trace_volume_tree，无 tl0 偏移还原
- **budget = 65536**（WGSL/CPU 同值）：几何上界 = 对角射线穿全分裂 chunk 节点读数 ~25k 量级；真实场景每 chunk 几十~几百次迭代；纯防挂死安全网

**代码落点**：WGSL `dda.wgsl` TreeFrame/init_tree_frame/trace_chunk/trace_grid（chunk_budget）；CPU 镜像 `dda.rs` TreeHit/TreeFrameCpu/init_tree_frame_cpu/trace_chunk_cpu/trace_volume_tree/cpu_reference_dda_ray_tree；多 volume 三函数（object_ray_unified/trace_volumes/volumes_occluded）切树路径；sample_brickmap 点查询保留给 implicit normals；旧两级 DDA 保留（旧等价测试仍用）。

**验证**：
- 新等价性测试 `tree_traversal_equivalence_300_rays`：跨 chunk/负坐标场景（4 特征体），8 轴平行射线（含起点在体内 UB）+ 300 随机射线（球壳心 (128,32,128) r32..1500，state=20260904）；新层次遍历 vs 逐体素 full DDA 参考：hit/miss + palette 严格一致、|t| 容差 1.0 fine、命中面法线 n·d < 0
- cargo test --workspace：**116 passed / 0 failed / 0 警告**；wgsl naga parse+validate 通过
- 后台启动正常：1264 chunks / 247MB 树 / build_full 513ms / 无 wgpu validation 错误
- 帧率目验待用户 F5（目标恢复 200fps 量级；预期天空射线 ~90 load vs 旧 ~3170 load）

**naga 陷阱（本次实测新增）**：`active` 是 WGSL 保留关键字（改 axis_on）；struct/函数必须先声明后使用（Grid 前移到 sample_brickmap 前、FineHit 前移到 trace_chunk 前）。

### 层次遍历虚影 bug 修复（2026-09-03 下午）

**现象**：demo 场景实心体（浮空岛叠球等）被射线「穿透」成噪声虚影，测试场景（小坐标 build_full）却全绿。

**定位（CPU 侧复现 → 插桩）**：新增 `tree_equivalence_demo_like_edit_compact` 测试（地形分层柱 16×8×16 非 brick 对齐 + fill_bricks(e=16) + clear_voxel 空气孔洞 + 倒锥叠球 + 交替 palette 编辑 + compact_all）——rand26 射线 CPU 树遍历与逐体素参考分歧：t=37.7 虚假命中 pal 7（地形色，命中点在地形上方 10 fine 空气里）。

**根因（浮点边界 → 越界回绕）**：层次遍历内层帧的 tmax 用 `tmax += delta*sub` 累计，与下钻时父帧保存的 t_exit（同一几何边界的另一条计算路径）差 1 ulp。步进判定 `t_out >= t_exit` 差一步该弹栈却做了步进 → **cell 越界（-1）** → `child_idx = cz*16+cy*4+cx` 负数回绕成别的子块 bit（-1 → idx 23，恰是下方地形列）→ 下钻进零厚度退化帧（t_enter==t_exit）→ 擦边假命中。小坐标测试场景 ulp 误差不足触发；demo 5632 世界大量触发 → 满屏噪声。

**修复（WGSL + CPU 逐字镜像同步，三处）**：
1. 推进循环：步进后 cell 越界（<0 或 >3）→ 视为节点耗尽弹栈（禁止越界索引）
2. 零厚度擦边不下钻/不命中（`cell_exit > t_enter` 才下钻；uniform 子块与 leaf 命中同样要求 `t_enter < cell_exit`）——与逐体素点查语义一致（不入内部不命中）
3. trace_chunk 入口 t0>=t1 早退 miss（chunk 角擦边）

**验证**：demo 复刻测试 300 随机射线 + 10 轴射线全过；workspace 117 测试全绿；实机启动无 wgpu 报错。

### 世界规模运行时可调（2026-09-03 下午，快速调试档）

正确性调试期启动 55s 不可接受 → 世界规模改为运行时可调：
- `GATE_TILES` env（默认 **2** = 快速调试档，启动构建 **4.45s**，12.5×；`GATE_TILES=10` = 完整压测场景）
- 场景结构性坐标全部改为以 `EXT_FINE_HALF`（世界中心）为锚：城堡/大道/河/森林/物体芯片 A·B·C/初始相机；const → LazyLock
- 城堡外 криstal 簇与 tile(2,0,0) 热点带越界守卫（小世界自动跳过）
- N=2 实测：80 chunks / 99MB 树 / build_full 203ms

### grow 扩容 usage bug + 芯片摆放（2026-09-03）

- **`ensure_with_copy` 二次扩容必炸**：grow 路径新建 buffer 的 usage 缺 `COPY_SRC`（初始创建有），第二次扩容时旧 buffer（上次 grow 产物）作为 `copy_buffer_to_buffer` 源 → validation error 退出。小场景（99MB buffer）+ 120 帧编辑触发生长，很快踩中。修复：grow 新建 buffer usage 补 `COPY_SRC`。实机 100s 多轮编辑零错误。
- 芯片三枚拿到城堡区平地（y=16），三角错开避让大道/边界/彼此，各给不同 yaw+俯仰（25°/160°/-75° + 15°/-30°/40°）与缩放（2.0/1.3/2.2）。
- 帧率仍 ~30：PresentMode 已是 AutoNoVsync（非 vsync 砍半），待 GPU 计时剖析主/阴影射线占比。

### 帧率剖析与修复：30 → ~56 fps（2026-09-03 晚）

**方法**：FrameTimeDiagnosticsPlugin + LogDiagnosticsPlugin 落日志（fps/frame_time + gate_dda_compute/elapsed_gpu GPU 耗时），配合 env 诊断开关（GATE_SKIP_SHADOW / GATE_SKIP_IMPLN / GATE_SKIP_CHUNKWALK / GATE_SKYOUT / GATE_MAKEGRID_ONLY / GATE_CAM=sky）逐层二分。**关键教训：诊断跑出的 fps 必须先确认 shader 编译成功**（naga parse 失败时 compute 管线静默缺失 → 黑屏假高速）。

**二分数据（gate_dda_compute GPU 耗时，1.44M 像素 overview）**：
- 完整路径 31.3ms ≈ frame_time 31.5ms → 100% 在 DDA compute pass
- 跳阴影射线 26.1ms（阴影占 ~5ms）；跳 implicit normal 6 邻域 ≈ 0（免费）
- 纯天空输出（跳全部 trace）0.05ms → 遍历框架本身占 ~全部
- CPU 镜像计数：典型射线仅 ~13-31 次迭代 → **不是迭代步数问题，是 WGSL 函数调用开销**

**根因**：**naga/驱动不内联 WGSL 函数**——每个函数调用（含大/小参数、含返回 struct）实测 ~µs 级开销。slab_box 每 pixel 被调 8 次（4 volume × 2 slab）= 1150 万次调用/帧 → 仅此一项 ~14-17ms。

**修复（全部实测验证）**：
1. slab_box 内联进 trace_grid（世界 + 局部两个 slab）——**17ms → 2.75ms**（skip-chunkwalk 模式对照）
2. init_tree_frame 内联进 trace_chunk 两处调用点（根帧 + 下钻）——23.4 → 17.7ms
3. face_index_from_normal 内联进 trace_grid（entry_face 计算）
4. trace_grid 改名 trace_grid_idx：参数从 15 字段 Grid 结构体改为 `idx: u32` 直读 grid_descs（make_grid 仅在命中时构造 best_grid 用）
- **反例存档**：4 独立变量静态分支栈版实测 2× 慢（59.8 vs 27.6ms）——勿再试
- **陷阱存档**：PowerShell Set-Content -Encoding utf8 会给文件加 BOM，naga 拒绝（"\u{feff}" parse error）；WGSL 里误写 Rust 的 `#[allow]` 属性 = parse error

**战果**：31.5ms（30fps）→ **17.8ms（55-56 fps，GPU-bound）**。等价性测试全绿（117 tests）。

**遗留（下一杠杆，2026-09-04 定向修正）**：skip-chunkwalk（纯 GridDesc 读 + slab，全 miss）稳态仍有 ~16.6ms GPU 且与负载弱相关（偶发 0.07ms 毛刺）——真实遍历仅占 ~1.2ms，底噪是大头。基准对齐：Douglas 1660 Ti 主+阴影全程 7ms。下一刀（1:1 最终方案 #17→#23 内，#7 光栅系 G-buffer/pass 拆分已废弃不学）：①二分矩阵重测（skyout/makegrid_only/skip-chunkwalk/full）定位 floor 层；②lazy descriptor 读（先 slab 8 字、chunk-walk 字段 slab 通过后才读——射线不用的数据不预读，与按需读树同构）；③若仍高上 #18 方向位掩码 LUT 预过滤（他实测 100→80ms）。
