# DDGI 探针烘焙 GPU 化 + 相机 LOD 级联 - 实现计划

> 说明：本重构寻址模型整体切换（全局 probe id → `(lod, cell)` 固定槽），WGSL 侧
> active/cast/update/sample/viz 与 Rust 侧 buffer/uniform 强耦合，按依赖顺序串行推进；
> 每个垂直切片自带可验证 TR。常量双侧（dda.wgsl / ddgi.rs）同步，wire 单测防漂移。

## Task 1: WGSL brick 三态查询 helper + CPU 镜像
- **Status**: `pending`
- **Priority**: high
- **Depends On**: None
- **Description**:
  - 在 dda.wgsl 新增 `ddgi_brick_state(g: Grid, origin_fine: vec3<i32>, level: u32) -> u32`：返回 0=Air / 1=Solid / 2=Mixed。复用 sample_brickmap/trace_chunk 的节点头逻辑：定位 chunk → 读对应层级节点 mask_lo/mask_hi/palette → 该层级 4³ 子块（或对齐 region）全 0 且 palette=0 → Air；mask 全 1（或全实心 palette）→ Solid；否则 Mixed。需支持 cell_size 16/32/64/128（对应树 level 2 及 2×2×2 组合 / level 1 / 2×2×2 / chunk root）。
  - gate-voxel/ddgi.rs 侧保留/新增一个 CPU 参考函数（纯树查询，非烘焙摆放）供等价测试调用。
- **Acceptance Criteria Addressed**: AC-2
- **Test Requirements**:
  - `rule` TR-1.1: WGSL helper 对 Air/Solid/Mixed 三种 region 返回值正确；wgsl_compile 通过。Evidence: wgsl_compile 测试。
  - `rubric` TR-1.2: 三态逻辑逐字镜像 gate-voxel chunk_tree.rs `brick_state_at`/`aggregate_node_state`（代码评审对照：mask/palette/popcount/inline 原语与已被 DDA fuzz 验证的 sample_brickmap 同源）；scale 1-5；anchors 1=镜像偏差大，3=大体一致偶发疏漏，5=逐条对照一致；threshold >=4；Evidence: 评审记录 + release 目验探针落点。
  - 注：用户决策（2026-09-09）不写 CPU 镜像 fuzz——三态归约极简且树读取原语已被 trace_chunk_cpu 暴力 fuzz 覆盖，靠 wgsl_compile + 评审 + 目验兜底。

## Task 2: WGSL get_ddgi_probe（三态 + BFS 最大空叶 offset）+ 等价测试
- **Status**: `pending`
- **Priority**: high
- **Depends On**: Task 1
- **Description**:
  - dda.wgsl 新增 `ddgi_get_probe(g, cell_origin_fine, cell_size) -> ProbePlace { enabled: bool, no_surfaces: bool, off2: vec3<u32> }`：逐字复刻 CPU [probe_position_in_cell](file:///c:/code/repo.rust/gate/gate-render/src/ddgi.rs#L408)——cell 三态；Solid → enabled=false；Air → offset=cell 中心(off2=16×2? 量化 cell 内 fine×2)；Mixed → 从 cell 所在树层 BFS 下钻，同层取靠 cell 中心最近的最大空叶中心，off2 = 空叶中心相对 cell 原点的 fine 偏移 ×2（[0,32) 5bit）。
  - CPU 侧以现有 probe_position_in_cell 为参考（Task 9 删除前保留供测试）。
- **Acceptance Criteria Addressed**: AC-2, FR-1
- **Test Requirements**:
  - `rule` TR-2.1: GPU 逻辑（CPU 镜像 WGSL 算法）与 probe_position_in_cell 在随机 cell（Air/Mixed/Solid、跨 chunk、cell 16/32/64/128）上 enabled/no_surfaces/off2 完全一致，≥300 cell。Evidence: 等价/fuzz 测试。
  - `rule` TR-2.2: wgsl_compile 通过。Evidence: 测试输出。

## Task 3: 相机 LOD 窗口 uniform（纯算术，无几何）
- **Status**: `pending`
- **Priority**: high
- **Depends On**: None
- **Description**:
  - 定义 4 级 LOD 窗口参数（每级：cell_size=16/32/64/128、window_origin_fine（相机 fine 对齐 cell_size，使相机位于窗口中央）、reuse bounds（相对上帧窗口原点的 cell 偏移/重叠区间））。
  - Rust 每帧（extract/prepare）由相机 fine 坐标纯算术计算这些值写入 uniform/buffer；不查体素。WGSL 侧 DdgiUniform/DdgiDomains 增加 lod 窗口数组（4 × {origin, cell_size, reuse_lo, reuse_hi}）。
  - 删除旧"base 世界级静态网格"uniform 语义，grid_origin/grid_dims 改为 lod0 窗口或移入 lod 数组。
- **Acceptance Criteria Addressed**: FR-2, AC-3
- **Test Requirements**:
  - `rule` TR-3.1: 窗口原点对齐 cell_size 且相机落在窗口 [4,12] cell 中央带；移动相机 origin 阶梯式跟随。Evidence: 单元测试（纯函数窗口原点计算）。
  - `rule` TR-3.2: Rust ShaderType 与 WGSL struct 字段/偏移一致（wire/布局断言）。Evidence: 编译 + 现有 wire 测试。

## Task 4: 固定槽 (lod, cell) 纹理寻址
- **Status**: `pending`
- **Priority**: high
- **Depends On**: Task 3
- **Description**:
  - 定义 slot = lod*4096 + (x + y*16 + z*256)（每 lod 16³ cell）。WGSL 改 ddgi_meta_coord/ddgi_irr_coord/ddgi_depth_coord 为按 slot 映射（layer=slot/256，层内 (x,y)=slot%256 展开）。
  - 纹理层预算 = 4×16=64 层；Rust 侧纹理/layer 分配改为 4 lod 固定 64 层（删除 base 世界级可变层数 + 级联偏移 id_base 逻辑）。
- **Acceptance Criteria Addressed**: FR-4
- **Test Requirements**:
  - `rule` TR-4.1: slot↔(layer, in-layer x,y) 双向映射正确且落在 64 层内；Rust 与 WGSL 公式一致。Evidence: wire/单元测试 + wgsl_compile。

## Task 5: active pass 重写（get_probe + halo + outside_lower + reuse + worklist slot）
- **Status**: `pending`
- **Priority**: high
- **Depends On**: Task 2, Task 3, Task 4
- **Description**:
  - 重写 ddgi_active：dispatch 覆盖 4 lod × 16³ cell（或逐 lod dispatch）。每线程：cell 世界原点 = window_origin[lod] + cell*cell_size；调 ddgi_get_probe 得 flags/off2；halo workgroup 数组载本 lod 邻 cell flags（域内 get_probe、OOB=NO_SURFACES）；`outside_lower_grid(lod, cell_world_center)`（本 cell 中心不在更细 lod 窗口覆盖内）门控；probe_near_surface 三条件（own/6 邻 AND/object bbox）；reuse：cell 在 reuse bounds 且 off2 与 previous meta offset 一致 → 沿用 age，否则 age=0；命中 atomicAdd worklist（写 slot）；函数末尾单一 meta textureStore（pack off2+age；inactive age=0；Solid/非 ENABLED 免写）。
- **Acceptance Criteria Addressed**: FR-3, AC-3, AC-4
- **Test Requirements**:
  - `rule` TR-5.1: active 逻辑与 sort.glsl 截图逐条对照（flags 来源、halo AND、outside、reuse、imageStore 位置）一致；wgsl_compile 通过。Evidence: 代码评审对照 + 编译。
  - `rule` TR-5.2: worklist 计数 = 本帧 active slot 数，seal 钳制 ≤ PROBE_BUDGET；无重复 slot。Evidence: 逻辑审查 + 运行无越界。

## Task 6: cast / update 按 slot + 现算探针坐标
- **Status**: `pending`
- **Priority**: high
- **Depends On**: Task 5
- **Description**:
  - worklist 元素改为 slot（u32，含 lod）。cast：slot → lod/cell → 从 meta 读 off2 → 探针世界坐标 = window_origin[lod] + cell*cell_size + off2/2（fine）；发射 Fibonacci 射线（预算分摊不变）；端点着色/深度写回该 slot 的 irr/depth 层。update EMA/SH 数学不变，仅寻址改 slot。
  - 删除 ddgi_positions 读取。
- **Acceptance Criteria Addressed**: FR-5
- **Test Requirements**:
  - `rule` TR-6.1: cast 探针坐标 = active 放置坐标一致（同源 off2/window）；样本缓冲按本帧处理序号 wid.x 寻址不越界。Evidence: 代码审查 + wgsl_compile + 运行无亮/暗发散。

## Task 7: 采样 ddgi_sample_dom 改 slot + offset 重建 + lod 选择
- **Status**: `pending`
- **Priority**: high
- **Depends On**: Task 4, Task 6
- **Description**:
  - 重写采样：着色点 p + normal n → 选 lod（按到相机距离/覆盖：细 lod 窗口内用细，否则逐级外）；在该 lod 网格坐标三线性取 8 cell slot；各 slot 从 meta 读 enabled/age/off2 重建探针世界坐标；wn（dot(n, p-probe)）/wd（depth chevron，bias=cell_size*0.25）加权；8 探针加权平均。删除 all_ci/cell_index 反查、dom 优先级链、ddgi_dom_contains。
  - 保留调试 rejection 统计（wn/wd/age/np）用于粉块诊断。
- **Acceptance Criteria Addressed**: FR-6, AC-5
- **Test Requirements**:
  - `rule` TR-7.1: 三线性 8 cell 选取与 offset 重建坐标正确；wn/wd 公式不变；wgsl_compile 通过。Evidence: 编译 + 代码对照 Douglas 采样段。
  - `rubric` TR-7.2: 采样视觉正确性；scale 1-5；anchors 1=成片粉块/黑块/漏光，3=偶发边界瑕疵，5=各视角 GI 平滑无粉块；threshold >=4；Evidence: release 多视角截图。

## Task 8: probe viz 改 slot
- **Status**: `pending`
- **Priority**: medium
- **Depends On**: Task 5
- **Description**:
  - probe_viz_main 遍历 4 lod × 4096 slot，读 meta age>0 且 enabled，按 slot→lod/cell + off2 现算世界坐标，画统一黄色点；删除 positions/pos.w 依赖与 padding 槽判断（Solid slot 自然 age=0/未 enabled 跳过）。LOD 过滤滑条按 lod 生效。
- **Acceptance Criteria Addressed**: FR-7, AC-4
- **Test Requirements**:
  - `rule` TR-8.1: viz 只画 active slot、坐标与 cast 同源；wgsl_compile 通过。Evidence: 编译 + 目验同心壳层。

## Task 9: CPU 删除烘焙链路 + buffer/binding 精简
- **Status**: `pending`
- **Priority**: high
- **Depends On**: Task 5, Task 6, Task 7, Task 8
- **Description**:
  - 删除/停用：ProbeGrid 烘焙（bake_probe_grid/bake_cascade_grid/bake_cascade_cell/probe_position_in_cell 仅供 Task2 测试参考后移除）、region_has_voxels/probe_is_active、CascadeManager 网格烘焙（new/grids/metas）、pack_probe_positions/cascade_ci_global/build_meta_texture_data、positions/cell_index/cell_flags/all_ci buffer 及其 BG4 binding、active: Vec<bool>。
  - DdgiGpu/CascadeGpu 精简为：lod 窗口 uniform、meta/irr/depth ping-pong 纹理（64 层）、worklist/dispatch(indirect)/samples buffer。prepare_ddgi 仅：ping-pong 交换、写窗口 uniform、写 dispatch 间接参数、按几何 edit generation 置 age 失效（如需）。
  - 同步删除 WGSL 对应 binding（positions/cell_index/cell_flags/all_ci/ddgi_domains 旧域链）与 ddgi.rs BG4 layout。
  - 修复/更新受影响单元测试（移除 active/positions 断言，改为 slot/窗口断言）。
- **Acceptance Criteria Addressed**: FR-8, FR-9, AC-1
- **Test Requirements**:
  - `rule` TR-9.1: Grep 无 bake/probe_position/region_has_voxels/CascadeManager 网格烘焙残留；无 positions/cell_index/cell_flags buffer 与 binding。Evidence: Grep + cargo check。
  - `rule` TR-9.2: 体素编辑后不触发 CPU 重烘，GPU 下帧自动放置新探针。Evidence: 代码路径审查（无 bake 调用）+ 目验编辑后 GI 恢复。

## Task 10: 全量验证与目验
- **Status**: `pending`
- **Priority**: high
- **Depends On**: Task 9
- **Description**:
  - 跑 `cargo test -p gate-render --test wgsl_compile --release`、`cargo test -p gate-render --release`（lib + 集成）、`cargo test -p gate-app --release --bin gate-app`、`cargo fmt -p gate-render -p gate-app`、`cargo check -p gate-render --release --all-targets`。
  - release 运行：正面/背面/俯视/移动相机/体素编辑，Probe 诊断（看红点/粉块/同心 LOD 壳层）+ GI 模式 + 帧时。
- **Acceptance Criteria Addressed**: AC-5, AC-6, AC-7
- **Test Requirements**:
  - `rule` TR-10.1: 所有测试/编译/fmt 通过，无 error。Evidence: 命令输出。
  - `rubric` TR-10.2: 视觉正确性（无红点/背面粉块、LOD 近密远疏、移动平滑）；scale 1-5；threshold >=4；Evidence: 多视角截图。
  - `rubric` TR-10.3: CPU 帧时无烘焙尖峰（移动/编辑）；scale 1-5；threshold >=4；Evidence: FPS/帧时。
