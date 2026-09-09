# DDGI 探针烘焙 GPU 化 + 相机 LOD 级联（Douglas #23 1:1）- 产品需求文档

## Overview
- **Summary**: 将 DDGI 探针的"烘焙放置"从 CPU 整体迁移到 GPU compute：逐 cell 在 WGSL 内遍历体素 con tree 判定三态（Air/Mixed/Solid）并 BFS 找最大空叶确定探针 offset；探针槽位改为按 `(lod, cell_in_window)` 固定索引；4 级 LOD 窗口以相机为锚、随相机滚动，`outside_lower_grid` 做近密远疏划分，reuse bounds 沿用上帧 age。删除 CPU 侧全部探针烘焙几何遍历与 positions/cell_index/cell_flags 间接寻址。
- **Purpose**: ① 满足硬约束"CPU 侧不做任何烘焙/光照几何计算"，消除编辑/移动时 CPU bake 风暴（曾 150ms/帧）；② 1:1 复刻 Douglas Devlog #23 sort.glsl 的真实架构（`get_ddgi_probe` GPU 逐帧放置 + `dispatch.lod/base_position/reuse bounds` 相机滚动级联），根除此前因"CPU 烘焙固定网格 vs GPU 运行时 active 判定"三套几何遍历分歧导致的红点/粉块；③ 近密远疏相对相机，移动时探针窗口跟随且收敛结果复用。
- **Target Users**: gate 引擎开发者（DDGI GI 功能维护者）；最终受益为 gate-app 运行时 GI 质量与帧率。

## Goals
- CPU 侧零探针几何：不遍历体素树、不判 cell 三态、不摆探针位置、不建 CascadeManager 网格；每帧只上传体素树（b_struct，已有）+ 填窗口 uniform + 编排 dispatch。
- GPU active pass 内逐 cell 执行 `get_ddgi_probe(lod, cell)`：读 b_struct 判 Air/Mixed/Solid，Air/Mixed 放置探针（cell 原点 + 量化 offset），Solid 无探针。
- 4 级相机锚定 LOD 窗口：cell 边长 16/32/64/128 fine，每级 16³ cell 窗口；窗口原点随相机；`outside_lower_grid()` 使粗 LOD 仅在更细 LOD 窗口覆盖范围外存活 → 相机同心近密远疏。
- 探针固定槽 `(lod, cell_in_window)`：meta/irr/depth 纹理按 lod 分区、cell 线性索引；探针世界坐标 = `cell_world_origin + offset*cell_size` 现算，不存 positions 数组。
- 窗口滚动 reuse：重叠 cell（offset 与上帧一致且在 reuse bounds 内）沿用 age；新滚入 cell age=0。
- 采样端：着色点 → 选 LOD → 网格坐标三线性取 8 cell → 从各 cell meta 读 offset 重建探针世界坐标 → wn（背面）/wd（深度 chevron）加权，Douglas line-of-sight + front/behind。
- active 判定不做深度剔除（深度/背面剔除仅在采样阶段）。

## Non-Goals
- 不改体素树（b_struct / SVDAG brickmap）本身的数据结构与 DDA 主遍历。
- 不改八面体 irradiance/depth 编码、SH/EMA 更新数学、固定射线预算分摊机制（cast/update 内核沿用）。
- 不做非网格对象（object buckets）移动体的探针特殊放置之外的新功能；object bbox 第三条 near 判定保留现状。
- 不引入 TCP/回读调试；不做 f16 打包等后续优化。
- 不改变直光/阴影路径。

## Background & Context
- Douglas sort.glsl（截图1/2/3，`docs/screenshots/23/代码1..3.png`）权威逻辑：
  - `ivec3 cell = ivec3(gl_GlobalInvocationID) + dispatch.base_position;`
  - `DdgiProbeLocus probe = get_ddgi_probe(dispatch.lod, cell, object_buckets);`
  - `DdgiProbeFlags flags = ddgi_probe_locus_flags(probe); set_probe_flags(...); load_adjacent_probes(cell); barrier();`
  - `if ((flags & ENABLED) && outside_lower_grid()) { vec3 cell_center=...; if (probe_near_surface(cell_center, flags)) { ...reuse bounds + offset 匹配沿用 age... worklist_insert; } }`
  - `imageStore(next_ddgi_probes, texel, ...)` 在 if 块外（inactive 写 age=0）。
  - `probe_near_surface = (flags&NO_SURFACES)==0 || (6面邻接 flags 按位 AND & NO_SURFACES)==0 || probe_near_objects`。
- 字幕原文："four LODs which are increasingly less detailed in resolution but cover more of the world"；"per model baking step ... whenever new voxel data is uploaded"（= GPU 上 `get_ddgi_probe` 几何遍历，非 CPU）；采样 "line of sight visibility ... in front of ... versus behind ... weighted average over all eight probes"。
- 复用窗口 `reuse_min_bound/reuse_max_bound` + ping-pong `previous/next` + `normalized_offset == previous.offset` 仅在窗口滚动时需要 → 证明 LOD 窗口随相机移动。
- gate 现状（错误架构）：CPU [bake_probe_grid](file:///c:/code/repo.rust/gate/gate-render/src/ddgi.rs#L337)/[bake_cascade_grid](file:///c:/code/repo.rust/gate/gate-render/src/ddgi.rs) 遍历 [probe_position_in_cell](file:///c:/code/repo.rust/gate/gate-render/src/ddgi.rs#L408) 摆探针、填 positions/cell_index/cell_flags/active；CascadeManager 以相机初位烘焙后冻结；GPU active 读上传的 cell_flags。base 为世界级静态大网格 + 4 相机窗口（cell 32/64/128/256）。
- GPU 已有树遍历原语：[sample_brickmap](file:///c:/code/repo.rust/gate/gate-app/assets/shaders/dda.wgsl#L226)（fine→chunk→DFS mask 下钻，返回 palette，0=Air）与 trace_chunk 节点头逻辑（mask_lo/mask_hi/palette + popcount child 定位）可复用来查任意 brick 三态。
- 纹理现状：每层 16×16=256 探针槽（PROBES_PER_LAYER）；meta packed u32 = offset(5bit/轴 ×3) + age(8bit)，offset 量化 = cell 内 fine×2（[0,32)）。4 LOD × 16³ cell = 4×4096 = 16384 槽 = 64 纹理层。
- WGSL 常量与 Rust 常量有 wire 单测防漂移（改常量须双侧同步）。

## Functional Requirements
- **FR-1（GPU 放置探针）**: WGSL 新增 `ddgi_get_probe(lod, cell_window_origin_fine, cell_size) -> struct{enabled, no_surfaces, offset_fine2}`：对 cell 覆盖区域判三态；Solid → enabled=false；Air → offset=cell 中心；Mixed → 沿树 BFS 找最大空叶，offset=空叶中心（同层取靠 cell 中心最近），与 CPU `probe_position_in_cell` 语义逐字一致。需要一个"给定 fine 原点 + 树层级查 brick 三态（Air/Solid/Mixed）"的 GPU helper（复用节点头 mask/palette）。
- **FR-2（相机 LOD 窗口 uniform）**: CPU 每帧由相机 fine 坐标计算 4 级窗口原点（对齐到各自 cell_size）、cell_size、reuse bounds（相对上帧窗口），写入 uniform；不做任何几何查询。
- **FR-3（active pass 重写）**: 逐 lod × 逐 cell（4³ workgroup）一个线程：cell 世界原点 = window_origin + cell*cell_size；调 FR-1 得 flags/offset；halo 共享内存载邻 cell flags（OOB 邻居 = NO_SURFACES）；`outside_lower_grid` 门控；`probe_near_surface` 三条件；reuse 沿用上帧 age；命中 → atomicAdd worklist（槽 = lod*4096 + cell_linear）；函数末尾单一 meta textureStore（inactive 写 age=0，Solid/非 ENABLED 为免写分支）。
- **FR-4（固定槽纹理寻址）**: meta/irr/depth 按 `slot = lod*4096 + (x + y*16 + z*256)` 索引；层 = slot/256、层内 (x,y)=slot%256 展开。删除全局 probe id 空间、positions、cell_index、cell_flags、all_ci buffer 及其 binding。
- **FR-5（cast/update 按槽）**: worklist 元素为 slot（含 lod）；cast 由 slot → lod/cell → 现算探针世界坐标（cell 原点 + meta offset）发射射线；update 写回该 slot 的 irr/depth 层；射线预算仍固定 RAY_BUDGET 分摊到本帧 active slot 数。
- **FR-6（采样重建探针坐标）**: `ddgi_sample_dom` 按着色点选 LOD，网格坐标三线性取 8 cell slot；从各 slot meta 读 offset 重建探针世界坐标；wn/wd 加权不变；多级联 fallback 改为"按到相机距离/覆盖选择 lod"，删除 all_ci 反查与 dom 优先级链。
- **FR-7（viz）**: probe viz 遍历 4 lod × 4096 slot，读 meta age>0 且 enabled，按 slot 现算世界坐标画统一色点；删除 pos.w 红/黄与 positions 依赖。
- **FR-8（CPU 清理）**: 删除 ProbeGrid 烘焙、CascadeManager::new/bake_*、pack_probe_positions、cascade_ci_global、build_meta_texture_data 等 CPU 几何/打包路径；DdgiGpu/CascadeGpu 精简为窗口 uniform + 纹理 + worklist/dispatch/samples；prepare 仅做 ping-pong、uniform 写入、dispatch 参数。
- **FR-9（体素编辑刷新）**: 体素编辑后无需 CPU 重烘；GPU 每帧 get_ddgi_probe 读最新 b_struct，探针放置自动跟随几何变化（age 复用由 offset 匹配自然失效）。

## Non-Functional Requirements
- **NFR-1（性能）**: 相机静止与移动时均无 CPU 侧每帧几何尖峰；prepare_ddgi CPU 耗时应为 O(窗口 uniform 写入) 常量级；GPU active pass 每 cell 树遍历成本受控（Solid/Air 早退，Mixed 才下钻），帧时相对现状不回退（目验 FPS 均值不低于现状基线）。
- **NFR-2（等价性）**: GPU `ddgi_get_probe` 的三态判定与空叶 offset 必须与 CPU `probe_position_in_cell` 在随机场景上数值一致（offset 量化后相同），以 fuzz/等价测试锁定。
- **NFR-3（可维护性）**: WGSL 常量与 Rust 常量继续由 wire 单测防漂移；2 空格缩进；删除代码不留死函数/死 binding。
- **NFR-4（视觉正确性）**: release 目验无红点（GPU/CPU active 分歧）、背面无成片粉块（wn 误剔除）；GI 随相机移动平滑、无明显滚动跳变。

## Constraints
- **技术**: Rust + Bevy + wgpu/WGSL；WGSL 无 subgroup，worklist 用 atomicAdd；树为 4³ 细分 brickmap（CHUNK_SIZE=256，LEVEL_EXTENT=[256,64,16,4,1]）；meta offset 5bit/轴量化（×2）。
- **业务/用户硬约束**: CPU 侧禁止任何烘焙/光照几何计算；不自研扩展、1:1 复刻 Douglas；不凭印象复刻（以 sort.glsl 截图 + 字幕为权威）；禁止 TCP 调试；2 空格缩进；中文回复。
- **依赖**: b_struct/palette 数据已在 GPU；DdgiDebugSettings（mode/gain/lod 过滤）UI 已存在；相机 fine 坐标在 extract 可得。
- **流程**: 修改 dda.wgsl / ddgi.rs 后须跑 wgsl_compile、gate-render lib/测试、gate-app bin 测试、cargo fmt/check。

## Assumptions
- LOD 数 = 4，cell 边长 = 16/32/64/128 fine（字幕 "four LODs"，base grid 16³ 为最细 LOD）；每级窗口 16³ cell。最粗覆盖半径 = 16×128 = 2048 fine。若目验覆盖不足，再评估加第 5 级（256）——本 spec 先锁 4 级。
- "per model baking" 对应 GPU `get_ddgi_probe` 几何遍历（主世界 identity grid + object grids）；本 spec 先对主世界 grid 实现，object grid 探针放置沿用同一 helper（多 grid 循环）。
- 纹理总层预算 64 层（4×16）在现有纹理尺寸内可容纳（现状 base 世界级 + 4 级联已开更多层）。
- 窗口原点对齐 cell_size，reuse bounds = 新旧窗口重叠区间（cell 单位）。

## Acceptance Criteria

### AC-1: CPU 侧无探针烘焙几何
- **Type**: `rule`
- **Given**: gate-render 源码
- **When**: 检查 ddgi.rs 及相关
- **Then**: 不存在对体素树的 cell 三态遍历/探针摆放（bake_probe_grid/bake_cascade_grid/probe_position_in_cell/region_has_voxels/probe_is_active/CascadeManager 网格烘焙等已删除或不再被调用）；prepare 每帧不做几何查询。
- **Pass Condition**: Grep 无上述函数定义/调用；prepare_ddgi 内仅 uniform/buffer/dispatch 写入；`cargo check` 通过。
- **Evidence**: Grep 输出 + cargo check 结果。

### AC-2: GPU get_ddgi_probe 与 CPU 参考等价
- **Type**: `rule`
- **Given**: 随机/固定体素场景的 cell 集合
- **When**: 对比 WGSL `ddgi_get_probe`（或其 CPU 镜像参考）与旧 CPU `probe_position_in_cell` 的 enabled/no_surfaces/offset
- **Then**: 三态与量化 offset 完全一致
- **Pass Condition**: 新增等价/fuzz 测试（CPU 参考 vs WGSL 逻辑镜像，≥ 数百随机 cell）全绿。
- **Evidence**: 测试输出。

### AC-3: 相机 LOD 窗口滚动与 reuse
- **Type**: `rule`
- **Given**: 相机在场景中移动
- **When**: 窗口原点随相机更新
- **Then**: 重叠且 offset 不变的 cell age 沿用（不重置为 0），新滚入 cell age=0；探针窗口视觉上跟随相机。
- **Pass Condition**: WGSL reuse 逻辑与 sort.glsl 一致（reuse bounds + offset 匹配）；release 目验移动时近场探针分布持续贴合表面、无整窗 age 清零导致的周期性变暗。
- **Evidence**: 代码对照 + release 运行目验。

### AC-4: 近密远疏 LOD 划分
- **Type**: `rule`
- **Given**: 相机周围 4 级窗口
- **When**: active pass 执行 outside_lower_grid
- **Then**: 细 LOD 覆盖近场密探针，粗 LOD 仅在细 LOD 窗口外存活；同一空间不重复 active 多级探针。
- **Pass Condition**: probe viz（或 Domain 调试）显示以相机为中心的同心壳层：近场 cell=16 密、外围逐级变疏；无内外层重叠亮斑。
- **Evidence**: release 目验截图。

### AC-5: 无红点/无成片背面粉块
- **Type**: `rubric`
- **Dimension**: GI 探针/采样视觉正确性
- **Scale**: 1-5
- **Anchors**: 1 = 大量红点或背面粉块/黑块；3 = 偶发边界瑕疵，整体可用；5 = 各视角探针贴合表面、背面无成片粉块、GI 平滑无漏光网格。
- **Pass Threshold**: >= 4
- **Evidence**: release 多视角（正面/背面/俯视/移动）Probe 诊断与 GI 模式截图。

### AC-6: 全测试与编译通过
- **Type**: `rule`
- **Given**: 重构完成
- **When**: 运行 wgsl_compile、gate-render lib、gate-app bin、fmt/check
- **Then**: 全部通过（基线：wgsl 1、gate-render lib 56+、gate-app 6）
- **Pass Condition**: 各命令 test result ok，无 error；cargo fmt 无 diff。
- **Evidence**: 命令输出。

### AC-7: CPU 帧时无烘焙尖峰
- **Type**: `rubric`
- **Dimension**: 移动/编辑时 CPU prepare 耗时
- **Scale**: 1-5
- **Anchors**: 1 = 移动/编辑周期性百 ms 卡顿；3 = 偶发可感小尖峰；5 = 移动与静止帧时一致、无烘焙尖峰。
- **Pass Threshold**: >= 4
- **Evidence**: FPS 面板/帧时日志在相机移动与体素编辑时的表现。

## Open Questions
- [ ] LOD 是否确为 4 级（16/32/64/128）还是保留 5 级（含 256）？规格先锁 4 级，目验覆盖不足再议。
- [ ] object grid（非主世界网格）的探针放置是否本期一并多 grid 循环，还是先主世界、object bbox 第三条兜底？倾向先主世界 + 现有 object bbox 判定。
