# 砖块图（GPU Brick Map）设计文档 — P2.1

> 状态：评审稿（先文档后代码）。本文档是 P2.2 构建器 / P2.3 上传通道 / P2.4 DDA 的实现契约。
> 数据源事实以 `gate-voxel` 源码为准（coords.rs / tile.rs / grid.rs / dirty.rs / palette.rs）。

## 1. 目标与非目标

**目标**

- 与 CPU 可变叶八叉树**同构**的 GPU 线性布局（NanoVDB 式，无指针、纯 u32 索引）
- 5 级分辨率完整保留：L0 4cm / L1 2cm / L2 1cm / L3 0.5cm / L4 0.25cm
- 编辑增量上传：以 `DirtyEdit { tile }` 为粒度，逐 Tile 重建，帧预算内分批
- 工作间/关卡场景 VRAM ≤ 2GB（砖块图 + G-Buffer，v3.1 预算语义）
- DDA 友好：空区域跳过以 u32 位掩码字为单位（一次 load 跳过 32 个基元胞）

**非目标（本期不做）**

- 元件层 / StateTable：`Tile.comp_layer` 当前为 None；P6 后状态更新走独立 buffer，不触碰砖块图
- 流式加载 / 大世界分页（P9 范畴）
- 100 万非空 Tile 的**渲染**：数据层沙盒稳健性已在 P1.7 验证；GPU 渲染按 §7 上限截断并告警
- GI / 阴影 / 发光（P3+）：本结构只输出 G-Buffer（pos / normal / palette_idx）

## 2. 结构总览：CPU ↔ GPU 同构映射

| CPU（gate-voxel） | GPU（线性化后） | 说明 |
|---|---|---|
| `HashMap<TileCoord, Box<Tile>>` | TileIndex（稠密 u32 网格）+ TileBitmaps（槽位池） | 稀疏→稠密索引，见 §3.1 |
| `Tile.occupancy: [u64; 512]` | TileBitmaps 槽：`u32[1024]`（4KB） | **位布局逐位相同**，bit i = cell_index i |
| `Tile.cells: HashMap<u16, Cell>` | CellDirs 槽：`u32[32768]`，cell_index 直接寻址 | 0 = 空胞；见 §3.2 |
| `Cell.uniform: Option<u8>` | CellNode hdr bits 0-7（0 = 非 uniform） | AIR=0 恰好当哨兵 |
| `Cell.l1/l2/l3: [Slot; N]` | CellNode 内联槽表（u16 槽 × 2 打包进 u32） | 顺序定长排列，无需内部指针 |
| `Slot::Empty / Leaf(u8) / Branch` | u16：tag(bits 8-15) + palette(bits 0-7) | tag: 0=Empty 1=Leaf 2=Branch |
| `Cell.l4: Option<Box<Brick>>` | has_brick 标志 + brick_index → BrickPool | 见 §3.4 |
| `Brick { occupancy, palette: [u8; 4096] }` | BrickPool slab：`u32[1024]` = 4096B 纯调色板 | **GPU 不存 brick 掩码**：palette 0 即空 |
| `Palette: [PaletteEntry; 256]`（8B/条，repr(C) 有测试断言） | b_palette buffer 原样 memcpy | 2KB |

关键取舍：**Branch 槽不携带指针**。同胞内表链按固定顺序排列（hdr → l1 → l2 → l3 → brick_idx），下一级表地址 = 胞基址 + 前置表尺寸和（由 hdr 标志位可算）。胞内寻址零指针，最大胞节点 294 字（1176B）。

## 3. 寻址链（DDA 视角的五步下钻）

```
射线位置（最细格 fine，i32³）
 → ① TileIndex[tile_pos]      稠密网格，1 次 load；0 = 空 tile（128cm 步进跳过）
 → ② TileBitmaps[slot]        4cm 基元胞 DDA；空区按 u32 字跳过（1 load 跳 32 胞）
 → ③ CellDirs[slot][cell_idx] 命中占用胞后取胞节点；0 = 空（理论不可达，assert）
 → ④ CellNode                 uniform → 命中；否则逐级槽表子 DDA（L1 2cm → L4 0.25cm）
 → ⑤ BrickPool[brick_index]   L3 Branch → 0.25cm 步进查 palette 字节，非 0 = 命中
```

### 3.1 TileIndex：稠密 tile 网格

- 维度 = 场景包围盒（CPU 侧统计非空 TileCoord 的 min/max，构建时定，含 ±1 tile 余量），三元组进 `b_globals`
- 条目 = 0（空）或 tile_slot+1（TileBitmaps/CellDirs 槽号）
- 上限 `TILE_INDEX_CAP = 128³`（2M 条目 = 8MB）。超出包围盒上限的 tile **不渲染**，启动时 `warn!` 一次性上报（沙盒语义：不崩溃 + 数字上报）
- slot 分配：TileBitmaps / CellDirs 池顺序分配，上限 `TILE_CAP = 1024`（工作间 ~320 tile 含余量的 3 倍以上）

### 3.2 CellDirs：直寻胞目录

- 每 tile 固定 `u32[32768]`（128KB），下标 = `cell_index`（与 CPU `cell_index()` 同一线性序：x + y·32 + z·32²）
- **不做 rank 压缩**：占位虚耗换取 O(1) 直寻与逐 Tile 原地重建（增量更新无需重排）。代价恒定：`TILE_CAP × 128KB = 128MB`，写死在预算表里
- TileBitmaps 的存在意义就是让 DDA 在空胞上**不碰** CellDirs（位掩码字跳过），故虚耗不转化为带宽

### 3.3 CellNode 编码（NodeStream 内，u32 对齐）

```text
CellNode:
  word0  hdr:
    bits 0-7   uniform_palette（0 = 非 uniform，后接表链）
    bit  8     has_l1    bit 9  has_l2    bit 10 has_l3    bit 11 has_brick
    bits 12-31 保留（0）
  [has_l1]    4   u32   l1 槽表（8 槽 × u16）
  [has_l2]    32  u32   l2 槽表（64 槽 × u16）
  [has_l3]    256 u32   l3 槽表（512 槽 × u16）
  [has_brick] 1   u32   brick_index + 1

u16 槽（word 内低 16 位 = 偶数槽，高 16 位 = 奇数槽）：
  bits 0-7   palette（Leaf 载荷；Empty/Branch 时为 0）
  bits 8-15  tag：0 = Empty，1 = Leaf，2 = Branch
```

字数：uniform 胞 1 字（4B）；全深度胞 1+4+32+256+1 = 294 字（1176B）。
规范型不变式与 CPU 一致（`tile.rs` 头注释）：uniform ⟺ 无表链；Branch ⟹ 下级表存在；构建器照搬 CPU Canonical 形态，**不做 GPU 侧再折叠**。

### 3.4 BrickPool：L4 砖块 slab

- slab = `u32[1024]` = 4096 字节，恰好 4096 个 L4 槽的 palette 字节；槽 i → 字 `i/4` 的字节道 `i%4`（小端）
- **不存占用掩码**：palette 0 = AIR 即空（构建器负责把 CPU brick 中掩码未置位的槽写 0，CPU `clear()` 只清掩码会留陈旧 palette 值）
- slab 定长 → 空闲链零碎片（§5）

## 4. Buffer 布局（storage buffer 分区）

| Buffer | 类型 | 分区 | 预算（最坏） |
|---|---|---|---|
| `b_struct` | `array<u32>` | ① TileIndex（≤8MB）② TileBitmaps（4MB）③ CellDirs（128MB）④ NodeStream（可变） | ~750MB |
| `b_leaves` | `array<u32>` | BrickPool slabs | 按热点，典型 ≤256MB |
| `b_palette` | `array<u32>` | 512 u32（256 × PaletteEntry 原样） | 2KB |
| `b_globals` | uniform | 包围盒 / index 维度 / 各池计数与上限 / 帧号（P2.4+ 相机矩阵） | <1KB |

**⚠ wgpu 限制风险（P2.3 开工当天先用 3 行探测代码验证）**：wgpu `Limits::default()` 为 `max_buffer_size = 256MiB`、`max_storage_buffer_binding_size = 128MiB`，而 `b_struct` 最坏 ~750MB。对策：创建 Device 时请求提升限制（`request_device` 用 adapter limits；GTX 1660 / Turing 桌面 Vulkan 可达 GB 级）。若 1660 实测拿不到，则把 NodeStream 拆成多 buffer（分区偏移进 globals），设计不变、寻址加一层。**此项列为 P2.3 第一验收点。**

## 5. 增量更新协议（P2.3 消费）

**粒度**：`DirtyTracker::drain_data_budget` 吐出的 TileCoord（FIFO 去重已就绪）。每帧预算以**字节**计（初值 4MB/帧），由上传侧换算为 Tile 个数。

**逐 Tile 重建**（Tile 上所有胞重新序列化，粒度粗但简单正确。成本两极：均匀为主 tile 亚毫秒级 CPU + ~百 KB 上传；**全深度 tile = 32768 胞 × 5.2KB ≈ 170MB 上传**——单帧 4MB 预算下需 ~42 帧，是 P4.2 同帧编辑闭环的延迟上界。对策（本期不做，触发点 = P4.2 实测编辑延迟超标）：`DirtyEdit` 粒度扩展到胞级增量上传，P1 已预留扩展位）：

1. 从 NodeStream / BrickPool 空闲结构分配新块，序列化该 tile 全部胞（Rayon 单任务）
2. 经 staging buffer 拷入 GPU；写 TileBitmaps 槽与 CellDirs 槽
3. 释放旧块：CellNode 变长块按 **pow2 字桶空闲链**（1..512 字，约 10 桶，FIFO → 分配确定性）；brick slab 定长 → slab 空闲链
4. NodeStream 碎片可观测（free/total 比率进日志）；**分配失败的兜底 = 全量重建**（永远正确，作为安全网，而非碎片整理器）

**全量构建**（初始化/兜底）：按 TileCoord 排序逐 Tile 构建（确定性），Rayon 并行分块（P2.2）。首次加载为阻塞式，不受帧预算约束。

**确定性**：FIFO 空闲链 + 排序后 tile 顺序 ⇒ 全量构建字节级可复现 ⇒ 可写 golden 等价测试（§8）。

## 6. DDA 遍历草案（P2.4 消费）

Amanatides & Woo 步进为骨架，四级网格嵌套（128cm → 4cm → 表级 → 0.25cm）：

1. **射线预裁剪**：射线 × 场景包围盒（Majercik 2018 ray-box，拾取/碰撞将来复用同一实现）；空场景早退
2. **Tile 级 DDA**（128cm）：TileIndex 一次 load；空则步进。步长=tile 边长，坐标换算在 fine 空间做（×400 到 0.25cm 格，f32 整数精确范围内，bbox 相对化保精度）
3. **基元胞 DDA**（4cm）：TileBitmaps 位测试；空区按 u32 字跳过（一次 load 跳 32 胞）
4. **胞内判定**：CellNode 取出。uniform → 命中，法线 = 进入面。非 uniform → 子 DDA：
   - L1（2cm）→ L2（1cm）→ L3（0.5cm）：逐级 A&W 步进查槽；Leaf = 命中；Empty = 该级跳过；Branch = 下钻
   - L3 Branch → brick 内 0.25cm 步进，palette 字节非 0 = 命中
   - 射线离胞 → 回到第 3 步继续（子 DDA 可能 miss：胞占用不代表射线碰到其体素）
5. 输出 G-Buffer：world pos（f32×3）、normal（进入面）、palette_idx + flags

## 7. 内存预算表（工作间最坏情形，VRAM）

以 P1.7 工作间实测规模为基准（4M 基元胞 / ~320 tile 含余量）：

| 项 | 最坏情形 | 字节 |
|---|---|---|
| TileIndex | 128³ 封顶 | 8MB |
| TileBitmaps | 1024 tile | 4MB |
| CellDirs | 1024 × 128KB（恒定虚耗，写死） | 128MB |
| NodeStream — 均匀胞主导（实测场景形态） | 4M × 8B（dir 4B + hdr 4B） | 32MB |
| NodeStream — L2 精细化满铺（理论最坏） | 4M × 152B（hdr4+l1 16+l2 128+dir 4） | 608MB |
| NodeStream — L4 热点 | 全深度胞 5.2KB；预算内 ~20 万胞（≈12.8m³ 满深度体量） | ~1GB |
| BrickPool | 随热点 | ≤1GB |
| G-Buffer（P2.4） | 1080p × ~3 目标 | ~50MB |
| **合计** | | **≤2GB ✓（L2 满铺 + 适度热点并存时 1.7GB 留余）** |

超预算的触发与对策（按优先级）：
1. NodeStream 换 **掩码+紧排载荷**（NanoVDB value/child mask 式，l2 全叶 80B vs 128B，省 ~40%）——构建器与 DDA 各加 rank，实现复杂度中
2. CellDirs 换尺寸分级空闲链（变长目录，省 128MB 恒定项）
3. CellDirs 换 GPU 开放寻址哈希（key = tile_slot<<16 | cell_idx）——省最多，DDA 每步加一次探测

### 7.1 极限能力换算（结构上限 vs 预算上限）

结构上限 = 寻址可达；预算上限 = 2GB 内真实可实现。**DDA 吞吐 / 帧率极限不在本表**（P2.4/P2.9 实测项）。

| 维度 | 极限值 | 卡在哪 |
|---|---|---|
| 最细分辨率 | 0.25cm（L4） | 结构固定 5 级 |
| 渲染非空 Tile | 1024 | TILE_CAP 池 |
| 渲染包围盒跨度 | ~164m（128³ 稠密索引） | TILE_INDEX_CAP |
| 可寻址基元胞 | 3350 万 = 2147m³（≈12.9m 立方） | 1024 × 32768 |
| 均匀 L0（4cm）体素 | 3350 万全可实现，268MB | 寻址先到顶，内存富余 |
| L2 全异色（1cm 满铺） | ~1200 万胞 ≈ 762m³ | 2GB（152B/胞） |
| L4 满深度热点 | ~20 万胞 ≈ 13m³ ≈ 8.4 亿最细体素位 | 2GB（5.2KB/胞） |
| 最坏单 Tile | 全 L4：1.34 亿细槽 / 170MB GPU | 单 tile 结构 |
| 全深度 tile 编辑上传 | 170MB/次（≈42 帧 @ 4MB 预算） | 逐 Tile 重建粒度，见 §5 |
| 调色板 | 256 材质 | u8 |
| 数据层（CPU 沙盒） | 100 万 tile 不崩溃（实测 4.2GB RAM）；渲染截断 1024 + 告警 | v3.1 语义 |

工作间设计点核对：8×8×4m = 256m³，距 L2 满铺极限 762m³ 余量 3×；混合形态（L2 满 + 20 万胞热点）≈1.8GB ✓。

## 8. 测试与验收（先于/伴随 P2.2-2.4 代码）

1. **构建器等价性**（P2.2，纯 CPU 无 wgpu 依赖，wire 格式类型放 `gate-render::brickmap`）：
   - 全量构建 vs 逐 Tile 增量累积 → 字节级一致（依赖 §5 确定性）
   - 构建后按 §3 寻址链软件遍历，与 `TileGrid::get_voxel` 全场景逐点比对（scene.rs 三种场景 + 多分辨率混合）
   - 空网格 / 单胞 / 全 uniform 大块 / 跨 tile 球体等边界场景
2. **预算断言**（P2.3）：构建典型 + 最坏场景后，各池字节量 ≤ §7 表；超限即 fail 并打印明细
3. **增量正确性**（P2.3）：随机编辑序列（确定性种子）→ 每步增量重建后软件遍历结果 == 全量重建结果
4. **上屏验收**（P2.5）：scene.rs 场景纯色直出，颜色与调色板一致（sRGB→linear 换算按 ADR-0002 契约）
5. **极限探测**（P2.4）：`--nocapture` 打印逐帧 DDA 步数分布与 G-Buffer 生成耗时（CI 宽松上界）

## 9. 已拒绝的替代方案（防重复论证）

| 方案 | 拒绝理由 | 重议触发点 |
|---|---|---|
| DAG 压缩（Kämpe 2013） | 只读结构，海量编辑是死穴 | 无（项目层面排除） |
| uniform 胞展开为 4KB brick（Teardown 式全 brick） | 工作间 4M 胞 × 4KB = 16GB | 无 |
| u32 槽表（tag+payload 单字） | l2 满铺内存 ×2（296B vs 152B/胞） | 若 u16 打包被证明出瓶颈（不太可能） |
| ropes 邻域指针（Laine & Karras 2010） | P2 规模下 A&W 裸步进足够；rope 表增维护成本 | 1660 实测 DDA 超帧预算时 |
| Brick 内再存占用掩码 | palette 0 即空，掩码是 512B 纯冗余 | 无 |

## 10. 遗留决策点

- [x] **P2.3 已裁决：wgpu 提升限制探测（§4 风险项）——决定 NodeStream 单/多 buffer**
  - 方案：首帧 PrepareResources 阶段通过 `RenderDevice.limits().max_storage_buffer_binding_size` 读取后端真实值；**阈值 = 1GB**：≥1GB ⇒ `BufferLayout::Single`（整块 NodeStream 单 storage buffer，代码路径简单）；<1GB ⇒ `BufferLayout::Multi { node_slices }`（按 word 切片分段绑定兜底，不做 P4.2 sub-range 精细）
  - 实测（RTX 3070 移动版，驱动 610.88，Vulkan 后端，wgpu 0.19，Bevy 0.19.1）：`max_storage_buffer_binding_size = 2147483648 B = 2.00 GB` → **走 Single 路径**（Single 也是 CPU 单测默认路径）
  - 双路径 CPU 单测：`limits_select_layout` 覆盖三条分支（≥1GB / 1B~1GB-1 / 0）
  - 日志探针：首帧 INFO `PROBE: max_storage_buffer_binding_size = X.XXGB → 单 buffer 路径`（实测 PROBE 出现 1 次，与最终接受日志一致）
  - 预算护栏：prepare 末尾 `debug_assert!(total_bytes ≤ 2*1024^3)`，VRAM ≤2GB（v3.1 决策）在 dev 构建硬性校验
- [ ] P2.4 开工前：G-Buffer 目标格式定稿（pos 用 world 还是 bbox 相对；normal 八面体编码 or 直存）——写 ADR

## 11. 实现期决策记录（P2.2 + P2.3 落地细化）

P2.2 已完成（builder.rs + wire.rs + view.rs，13 测试全绿）。P2.3 交付 upload.rs（410 行 + 4 CPU 单测）+ gate-app demo 场景。实现中把 §4/§5 的契约细化为：

1.  **NodeStream 精确 bump**：新块无空闲时按精确字数分配（无 pow2 padding）；释放块按 hdr
    反推精确尺寸进 pow2 字桶（1..512，10 桶）；复用取同桶 FIFO 首个适配块。 ⇒ 全量首建
    全走精确 bump，与增量累积字节级一致（§8.1 已测：`full_vs_incremental_byte_identical`）。
2.  **并行分批预算 64MB**：Rayon 只并行化纯函数序列化（TileBlob），放置仍按 TileCoord
    升序串行——字节确定性不受并行影响；单 Tile 最坏 ~170MB > 预算，必整批独占。
3.  **uniform 哨兵**：hdr bits 0-7 = 0 表示非 uniform；uniform palette 0（AIR）写入后
    恰好表现为空胞，读取侧语义自然正确，无需特判。
4.  **slot 槽位 FIFO 复用**：槽存在 ⟺ 上次更新时 tile 有占用胞；释放进 FIFO，复活 tile 复用。
5.  **TileCoord 增补 Ord**（分量字典序）：构建器确定性排序的基础数据类型增强。
6.  **view.rs 独立实现寻址链**：刻意不共享 builder 内部代码（共享即自证），等价性测试
    才有效；同时作为 P2.4 GPU DDA 的 CPU 参考实现。
7.  **实测参考**（开发机，CI 上界 ×10）：全深度单 Tile（32768 胞全 L4，node 36MB + slab
    128MB）全量构建 219ms（上界 30s）；GPU 合计 304MB / 2048MB 预算。
8.  **P2.3 #1 · World-cross 三段管道**：主 world `Last` → render world `ExtractSchedule` → render
    world `PrepareResources`。原因：TileGrid 是大对象，dirty drain 需要 `&mut`；ExtractSchedule 只允许
    `Extract<Res<T>>` 只读拿主 world 资源；所以主 Last `poll_pending` 先 `drain_data_budget(n)` 产出**轻量**
    `MainPending { force_full, data_tiles: Vec<TileCoord>, comp_tiles }`（坐标复制 ≤200B）；
    ExtractSchedule 只读 clone 进 render-world `BuilderMirror`，CPU 构建 BrickMapBuffers
    snapshot（~140MB 首帧，增量按需），`commands.insert_resource(UploadSnapshot)`；
    PrepareResources 再消费 snapshot，`queue.write_buffer()` 整块写 GPU。**三段解耦**
    主 world 的所有权需求。
9.  **P2.3 #2 · Extract ReadOnly 约束**：`Extract<ResMut<T>>` 在 Bevy 0.19 非法（T 必须是
    `ReadOnlySystemParam`，`ResMut` 不是）。因此主 world 的可变操作（drain dirty、清空
    force_full）全部塞进主 `Last` 的 `poll_pending`，render ExtractSchedule 只读。
10. **P2.3 #3 · poll_pending 在 Last schedule 的位置**：Bevy 0.19 调度顺序是
    主 App: First→…→**Last** → 才是渲染子 App 的 ExtractSchedule→Render sets。
    所以同一帧内，主 Last 的 poll_pending **一定先于** ExtractSchedule，
    `MainPending` 刚 drain 的新坐标对 ExtractSchedule 立即可见（无滞后一帧）。
11. **P2.3 #4 · poll_pending 入口强制清空**：BUG 层 1：上一轮的 data_tiles 没清会被
    反复 clone 进 mirror。修：poll_pending 函数最开头先执行
    `pending.force_full = false; pending.data_tiles.clear(); pending.comp_tiles.clear()`，
    再从 dirty 队列重新 drain。保证每帧 pending 内容 **= 本帧净变更**。
12. **P2.3 #5 · UploadSnapshot 消费即 remove**：BUG 层 2：render world 插入的
    UploadSnapshot 是 Resource，默认永存。若 prepare 消费后不 remove，下帧仍为
    `Some` → 读旧字节 → 每帧 ~140MB 假上传（首版 demo 实测 fps 44 < 60，PCIe ~5.6GB/s）。
    修：prepare 末尾 `commands.remove_resource::<UploadSnapshot>()`；同时 extract
    加 `dirty_any = need_full || !pending_data.is_empty() || !pending_comp.is_empty()`
    短路 return，不生产 snapshot → 无脏帧 GPU 完全零触碰。
13. **P2.3 #6 · encase 0.12.1 uniform 数组 stride 限制→全部 scalar**：
    `UniformBuffer<BrickMapGlobals>::write_buffer()` 底层走 encase 0.12.1
    （Bevy 0.19.1 lock），encase uniform 模式下对 **Rust fixed-size `[u32/i32; N]`** 会
    触发 `array stride must be a multiple of 16 (current stride: 4)` 断言 panic。
    即使 `[u32;4]` 总大小=16B 仍然断言（encase 按 **element stride**=4 判断）。
    Workaround：**把 BrickMapGlobals 内所有数组拆成命名 scalar**：
    `index_origin_x/y/z/w (i32×4)`, `index_dims_x/y/z/w (u32×4)`, `_pad0.._pad4 (u32×5)`。
    80B / repr(C)，ShaderType derive 通过，UniformBuffer write 正常。
    P2.3 期间修了 3 版（[T;3]→[T;4] 仍 panic→拆命名 pad 仍 panic→全拆 scalar final）。
14. **P2.3 #7 · 默认走 Single 布局**：首帧 PROBE 若返回异常 / 误读，默认分支走 Single
    并打日志告警；Multi 分段绑定是 <1GB 古董后端（≤2020 DX12 WARP/emulated）兜底，
    开发主流 GPU（RTX 3070 / 1660 / Arc A750 等）均 ≥2GB 绑定限额。
15. **P2.3 #8 · VRAM 2GB debug_assert 护栏**：prepare 计算完 `struct+leaves+palette+
    comp+state` 字节和后 `debug_assert!(total ≤ 2*1024*1024*1024)`。符合 TODO.md v3.1
    资源预算条款；release 构建降级为 warn 日志（不 panic，允许沙盒边界场景继续）。
16. **P2.3 #9 · MainPending → BuilderMirror clone 轻量语义**：MainPending 只放
    `TileCoord`（i32×4 = 16B/tile）Vec 和 bool，不是数据拷贝；Mirror 持有 `Option<BrickMapBuilder>`
    （完整 node/leaves/palette allocator），下次 build_full/update_tile 直接在**同一 allocator**
    上增量操作——保证 node_words / node_free_words 等全局计数跨帧连续。
17. **P2.3 #10 · PCIe 整块上传，sub-range 精细推迟 P4.2**：prepare 使用
    `queue.write_buffer(buf, 0, whole_bytes)` 整块重写 5 个 buffer。按 2GB 上限 +
    PCIe 4.0 x16 ≈32GB/s，单帧最坏 62.5ms；P4.2 可按 TileBlob offset 改为 sub-range
    write 进一步降 PCIe 占用，本期不做（验证通道闭环优先）。
18. **P2.3 #11 · 五 buffer + UniformBuffer<BrickMapGlobals> 统一资源**：
    `GpuBrickMap { struct_buf, leaves, palette, comp, state: Buffer, globals: UniformBuffer<BrickMapGlobals> }`。
    comp/state 即使没脏也分配 placeholder（4B buffer）以避免 render pipeline binding
    检查缺资源；globals 单独走 UniformBuffer 因为 encase 有 padding 约束，storage buffer
    无此限制（所以 comp/state 可以是 raw bytes）。
19. **P2.3 #12 · 首帧 force_full 与 demo_force_full_rebuild 语义拆分**：`VoxelScene.
    demo_force_full_rebuild` 只在 Startup frame=1 时塞一次 `MainPending.force_full=true`；
    `BuilderMirror.pending_full` 初始值 = true（Render app 启动时独立初始化）；
    双保险保证第一帧一定会 build_full 并产生 UploadSnapshot，不会受渲染子 app vs 主 app
    初始化顺序影响。
20. **P2.3 #13 · wgpu #9213 初始帧 VUIDs 接受**：启动时 `VUID-VkPresentInfoKHR-pImageIndices-01430`
    与 `VUID-vkAcquireNextImageKHR-semaphore-01286` 各 1 条已知上游 wgpu issue，
    无运行时危害（画面、FPS、GPU 资源都正常）。最终接受日志保留、不做规避 hack。
