# GATE — GPU-Accelerated Tile Engine

> GPU 稀疏体素（Douglas Brick Tree，256³ chunk）+ 计算着色器层次 DDA 光追 + 世界锚定 DDGI 全局光照 + 自研 bevy_ui 工具链。
> **当前分支 `wip/ddgi-v1`**：DDGI v1 / 光照场（AO）/ 自动曝光 / 体素编辑 / 可持久化调试菜单均已落地；默认场景为 MagicaVoxel `nuke.vox`。

---

## 0. 快速开始

```powershell
# 工具链：Rust stable（rust-toolchain.toml，MSVC toolchain）+ Vulkan 显卡驱动
cargo run -p gate-app                     # 默认场景 GATE_SCENE=vox → assets/vox/nuke.vox
cargo run -p gate-app --features profile  # 性能剖析：Tracy GUI 连接进程（CPU span + GPU pass 同时间线）
cargo clippy --workspace --all-targets -- -D warnings
```

> `assets/vox/nuke.vox` 被 `.gitignore` 排除（体积大），新克隆的仓库里没有它。备选：
> `$env:GATE_SCENE="demo"` 跑程序化「极限场景」，或自备一份 `.vox` 放进 `assets/vox/`。

**操作（默认「幽灵飞行」模式，无碰撞）**

- `WASD` 平移 / `Space` 升 / `Shift` 降 / `Ctrl` 切低高速档（缺省 128 → 256 voxel/s；低速档基础速度在菜单可调）
- 右键拖拽 = 转头（两种相机模式共享 yaw/pitch，切换时视线连续）
- 左键 = 放置笔触；右键**单击**（按下到释放位移 < 4px）= 擦除。形状 / 大小 / 材质在菜单「游戏/编辑」
- `F3` 开关左上角调试菜单；右上角 FPS 覆盖层与组件展示窗默认隐藏（菜单「视频/FPS」「界面/showcase」）

**相机模式**

- 默认 **Fly**（幽灵飞行）；菜单「玩家/相机/相机模式」切到 **Orbit** 后：
  中键拖拽平移 / 滚轮对数缩放（`Shift` 细调档）/ 左键拾取重设注视点

**菜单持久化**

- 初始值 `assets/ui/debug_menu.toml`（随包发布）；运行期改动在退出时写回 `<安装根>/data/ui/debug_menu.toml`，
  下次启动优先读它（读不到才回退 assets 版，新节点由 `merge_defaults` 回填）

---

## 1. 项目现状

| 模块 | 状态 | 说明 |
|---|---|---|
| **体素核心（gate-voxel）** | ✅ 生产可用 | Douglas Brick Tree：`HashMap<ChunkCoord, ChunkTree>`，每 chunk 256³ voxel，分裂因子 4³（256 → 64 → 16 → 4 → 1）；非叶节点 u64 占用掩码 + 紧凑 child 偏移表；`Node::Uniform` 自适应叶；16 位材质索引（65536 槽，8B/条）；三级查询 `get_voxel` / `get_brick_state` / `fill_brick`（O(深度) 整块写）；`try_merge` + `compact()` DFS GC |
| **多 volume** | ✅ 生产可用 | `Volumes`：`list[0]` = 主世界（`obj_id = -1`），`add_object` 注册物体（`VolumeTransform { pos, rot, scale }`）；同一 dirty → builder → upload 路径；GPU 侧为统一 `GridDesc` 数组（144B/条），shader `trace_scene` 无分支遍历 |
| **组件层 / 状态表** | ✅ 数据通路可用 | `comp_layer`：每 chunk 4096 个 16³ 组件 ID（u16）；`StateTable`：256 条 × 4×u32；随 dirty 双通道（data / comp）分别上传。**尚无逐帧模拟驱动**（仅 demo 场景写测试值） |
| **GPU 上传** | ✅ 生产可用 | `b_struct`（64³ 稠密 chunk 窗口 + 各 chunk DFS 序列化树）+ `b_palette`（512KB/volume）+ `globals`；脏区增量部分写（struct 字区间 + palette 槽区间）；扩容 `ensure_with_copy`（GPU-GPU 前缀拷贝）；backlog > 3× 预算时一次性刷新，避免逐帧阻塞 Prepare；日志 `UPLOAD[full\|incremental]` |
| **DDA 光追** | ✅ 生产可用 | WESL 包（`assets/shaders/voxel_raytrace/`）启动时读盘编译；层次栈式 mask DDA（节点掩码常驻寄存器，4³ 子块间步进零 load；`firstTrailingBit` 跨级跳）；方向可达掩码 LUT（Douglas #18 Bitwise Masking）辅助剔除；beam 低分辨率最近命中断面预 pass；局部 AABB slab 剔除 |
| **DDGI** | ✅ 生产可用 | 4 级世界锚定级联（cell 16/32/64/128 voxel，严格嵌套）；LOD0 按 chunk 从探针池领固定 4096 槽段；探针放在「最大全空叶」中心；每探针 4×4 辐照度 + 8×8 深度（均值/方差/更新数）图集，时域 EMA + 6 邻域空间混合 + Chebyshev 软遮挡；活跃探针 worklist + indirect `cast`/`collect`；增量重烘只覆盖 dirty AABB 命中的 cell |
| **光照场（AO）** | ✅ 生产可用 | 相机中心、世界锚定的 32³ × 16 voxel 网格（`Rgba16Unorm`，硬件三线性），.a = AO fill 直接乘进命中着色；发光走「命中直出自身颜色 + 进 GI」，不再走发光密度通道 |
| **材质与介质** | ✅ 生产可用 | `PaletteEntry { color, roughness, emissive, transmission }`；`transmission > 0` 走玻璃状态机（折射/透射 + 太阳透射率，`trace_glass`）；表面法线与命中体素由整数 DDA 精确产出（禁「命中点 ± 半法线」启发式重建） |
| **自动曝光** | ✅ 生产可用 | UE EyeAdaptation 式：1/16 抽样 → 64 桶 log2 亮度直方图 → 5%~95% 百分位均值 → 分方向时间平滑（变亮/变暗常数分开）；菜单「渲染/曝光」可调 EV± / tau / key |
| **体素编辑** | ✅ 生产可用 | 幽灵模式左键放置 / 右键单击擦除；球 / 立方笔触按 brick 粒度整块写入（整块全在笔触内 → 一次 O(深度) 写，落成 uniform 上级节点）；材质按**内容去重**落调色板槽（改材质不影响旧体素）；编辑 AABB 同时驱动增量上传 + DDGI 重烘 + 光照场重算；`EDIT[place\|erase]` 日志 |
| **渲染管线** | ✅ 生产可用 | **无 render node**：extract / prepare / dispatch / blit 全部系统级显式调度；`blit.wgsl` 双入口 `fs_main` / `fs_fxaa`（FXAA 3.11 移植）；半分辨率 `RenderScale` + 线性上采样；MSAA 强制关闭 |
| **调试菜单 + i18n** | ✅ 生产可用 | gate-ui 的 TOML 可序列化 `DebugWindow`（9 种行控件）+ `MenuActionEvent` 观察者；5 个顶层页（视频 / 渲染 / 玩家 / 游戏 / 界面，游戏页下含编辑与世界两个子页）；文案全走 i18n key（`assets/locales/zh-CN.yml` 编译期 codegen，缺键回落中文）；「游戏/世界」可扫 `assets/vox/*.vox` 选择模型并**热重载世界**（DDGI AABB / LOD0 chunk 集 / 光照场随之重建） |
| **世界标签** | ✅ 生产可用 | `WorldAnchor`：世界坐标 → 屏幕像素 UI 标签（距离缩放、CJK 字体延迟解析） |
| **性能剖析** | ✅ 生产可用 | `--features profile`：wgpu-profiler GPU pass 时间戳（Tracy 时间线）+ tracing span → Tracy CPU zone 桥；非 profile 构建零成本 |

### 已知限制 / 待办（不阻塞当前开发）

- **没有自动化测试与 CI**：workspace 0 个 `#[test]`（仅 `vendor/parley` 除外），回归靠手工验收 + 日志。
- **文档缺口**：代码注释引用的 `docs/brickmap.md`、`docs/decisions.md`（ADR-0001/0002/0005）、`docs/ui-dark-theme.md` 尚未落盘；`docs/` 目前只有 Douglas 开发日志转录（`docs/douglas/`）与 DDGI 截图（`docs/screenshots/`）。
- **正式 sim tick 未实现**：StateTable 只有数据通路与上传，没有逐帧模拟系统驱动它。
- **WorldAnchor 不做体素遮挡判断**（永远绘制在最上层）。
- **`assets/lighting/dark_lab.ron` 暂无代码引用**：当前只加载 `day_outdoor.ron`。
- **wgpu Vulkan 首帧 VUID 报错**（上游已知问题，仅首 1-2 帧 swapchain 时序）：`LogPlugin` filter 静默 `wgpu_hal::vulkan::instance` 与 `surface` 两层。

---

## 2. 技术路线总览

```
[Bevy 主 world]
  Startup : scene::setup —— 读 lighting/*.ron、建 VolumeGrid（GATE_SCENE=vox 默认 / demo 程序化）、
            算 DDGI 世界 AABB + LOD0 chunk 集、初始化 OrbitCamera / FlyCamera / CameraMode / UploadBudget
  Update  : 相机链（模式对齐 → 转头 → 各模式输入 → 拾取 → build_camera_config）→ 体素编辑
            → 调试菜单 / 组件展示窗 / FPS 覆盖层 / 相机信息文本
  Last    : poll_pending —— UploadBudget（4MB/帧）× DirtyTracker → MainPending
      ↓ ExtractSchedule（main → render world）
  extract        : VolumesBuilder 增量/全量构建 → UploadSnapshot + BrickMapDirty(AABB) + LightFieldUpdate
  extract_camera : DdaCameraConfig → DdaViewUniform
  extract_ddgi   : DdgiStage / DdgiDebugSettings / DdgiWorldAabb / DdgiLod0Chunks / 曝光活参数
      ↓ render world
  RenderStartup    : init_dda_pipelines / init_empty_gpu / init_ddgi_gpu / queue_ddgi_pipelines
  PrepareResources : prepare_upload（struct/palette/comp/state/grid_descs + 光照场 3D 纹理 + 扩容）
  PrepareBindGroups: prepare_dda_bind_groups → prepare_ddgi
  RenderGraph::Render（dispatch 顺序）:
      ddgi_bake0..3（细→粗，pass 边界即屏障）→ ddgi_sort → ddgi_seal
      → ddgi_cast（indirect）→ ddgi_collect（indirect，网格跨越）
      → beam（1/4 分辨率最近命中断面预 pass）
      → gi（半分辨率 GI 预 pass，可开关）
      → dda_main（主可见 pass，trace + 着色 → out_tex）
      → eye_histogram + eye_update（自动曝光）→ probe_viz（可选）
  Core2d::PostProcess: blit_dda_view（fs_main | fs_fxaa，线性上采样 + sRGB cancel → ViewTarget）
```

### 关键技术要点（为什么快）

1. **层次掩码 DDA**：每个非叶节点一次读入 u64 掩码进寄存器，该节点 4³=64 个子块之间步进**零 load**；
   `mask bit=0` 的 uniform 子块整格命中或整格跳过；跨 brick 后用 `firstTrailingBit` 一次跳到最粗可行层。
2. **方向可达掩码 LUT（b_leaves，Douglas #18）**：8 octant × 64 入口格 × 2 字的保守可达集，
   `occupancy & reach` 在进入子块前剔除；LUT 是真实可达集的**超集**，绝不漏命中。
3. **beam 预 pass**：1/4 分辨率先求「最近命中 t」，主 pass 从该 t 起步 —— 近场空空间不产生步进。
4. **世界锚定 DDGI**：探针槽位 = 世界 cell mod dims（LOD1~3）/ chunk 固定段（LOD0），相机移动不换主、不闪烁；
   编辑只重烘 dirty AABB 命中的 cell。DDGI chunk 段基址**一经分配不再改变**（基址变 = 图集整段错位）。
5. **脏区增量上传**：struct 字区间 + palette 槽区间局部写；全量路径只在首帧 / 换世界 / 树基址漂移时触发。
6. **半分辨率 + FXAA**：渲染内部分辨率 = 窗口物理像素 ÷ factor（菜单「视频/半分辨率」），blit 线性上采样；
   关掉 FXAA 时只是换一条 fragment 入口，无分支代价。

---

## 3. 模块架构（4 个 crate）

```
gate-voxel/        纯逻辑核心（零渲染依赖；依赖仅 glam + rayon）
                   - coords.rs      ChunkCoord / VoxelCoord / BrickCoord + CHUNK_SIZE / BRICK_FACTOR /
                                    MAX_LEVEL / LEVEL_EXTENT / child_linear_idx
                   - chunk_tree.rs  ChunkTree（Douglas Brick Tree）：Block 掩码分裂 / 紧凑 child 偏移 /
                                    Uniform 叶 / try_merge / compact GC / serialize（GPU wire 字数）
                   - volume.rs      VolumeGrid（chunk HashMap + palette + comp_layer + state_table +
                                    dirty + 编辑 AABB）+ Volumes 容器 + VolumeTransform
                   - palette.rs     Palette（65536 槽 × 8B）+ PaletteEntry + 脏槽区间 / 版本号
                   - dirty.rs       DirtyTracker：data / comp 双通道按 ChunkCoord 的队列与预算 drain
                   - scene.rs       几何帮助：fill_box / fill_bricks / fill_sphere / draw_text（5×7 点阵）

gate-render/       渲染与 wire 契约（CPU 侧；GPU 状态全部在 render world 系统里建）
                   - brickmap/
                       wire.rs     字节契约：常量、pack_palette_entry、BrickMapGlobals、GridDesc(144B)、
                                   方向可达掩码 LUT、BrickMapBuffers（b_struct/b_palette/globals）
                       view.rs     BrickMapView 纯读端寻址链（get_voxel / cell_occupied / chunk_base）
                       builder.rs  BrickMapBuilder（单 volume）/ VolumesBuilder（多 volume）+
                                   DirtyRanges（struct 字区间 + palette 槽区间）+ snapshot（自动降级全量）
                       upload.rs   poll_pending / extract / prepare / init_empty_gpu；ensure_with_copy 扩容；
                                   光照场烘焙与上传；UPLOAD[full|incremental] 日志；VolumePlugin
                       dda.rs      全部 CPU DDA 参考实现（brute / AABB-skip / 两级 cell / 层次栈式 /
                                   多 volume trace）、OrbitCamera / DdaCameraConfig、RenderScale /
                                   PostFxSettings / EyeAdaptSettings、全部 BG layout + pipeline +
                                   prepare_dda_bind_groups / dispatch_dda / blit_dda_view
                   - ddgi.rs        世界网格 / chunk 段池 / DdgiUniform / bake·sort·seal·cast·collect 调度
                   - lighting.rs    LightingTheme（RON）+ LightPoolUniform wire 契约（BG3）
                   - shader.rs      启动时 wesl-rs 编译 WESL 包 → Bevy Shader 资产
                   - wesl_consts.rs 从 .wesl 源码解析跨语言 u32/f32 常量（DDGI 等的**唯一权威**），启动 fail-fast
                   - profiler.rs    wgpu-profiler / Tracy 集成（feature = "profile"；非该 feature 为零成本壳）
                   - responsive.rs  窗口 resize → 渲染目标原地重建 + RenderScale 跟随
                   - paths.rs       install_root / assets_dir / logs_dir / data_dir

gate-ui/           自研 bevy_ui 组件库 + 调试菜单 + 世界标签
                   - theme.rs       UiTheme（RON 资产 + 内置暗色默认）、令牌（颜色/圆角/间距/字号）、UiScale
                   - icon.rs        FontAwesome 字形（菜单标题栏按钮与箭头）
                   - i18n.rs        UiTranslator：key → 文案解析器（切语言后整体重解析）
                   - capture.rs     UI 指针捕获 / 命中测试（**不是**截图工具）
                   - world_anchor.rs WorldAnchor：世界坐标 → 屏幕像素标签（距离缩放、延迟文本）
                   - widgets/       16 个组件：Panel / Label / Button / Slider / Checkbox / ToggleSwitch /
                                    Dropdown / Plot（折线）/ List（环形日志）/ ScrollView / Splitter /
                                    Grid / Table / TabView / TextInput / Tooltip
                   - menu/          TOML 可序列化菜单模型（9 种行控件）+ DebugWindow 容器 +
                                    menu_system（唯一交互驱动）+ MenuActionEvent

gate-app/          Demo 应用入口
                   - main.rs       插件装配、窗口/日志/i18n 初始化、系统注册、环境变量开关
                   - scene.rs      setup + 程序化极限场景 build_demo_scene + reload_world 换世界 +
                                   lod0_needed_chunks（DDGI LOD0 段分配集）
                   - camera.rs     CameraMode（Orbit|Fly）/ FlyCamera / 输入系统 / cursor_ray / 拾取 recenter
                   - edit.rs       EditSettings + BrushShape/BrushMaterial + raycast_main + 笔触施加与输入
                   - vox_scene.rs  MagicaVoxel .vox 导入（vox-rs）+ scan_vox_models 模型发现
                   - debug_menu.rs 菜单树默认定义 + 状态应用/落盘 + FPS 覆盖层 + 相机信息 + F3 开关
                   - showcase.rs   右上角组件展示窗（控件画廊 + 交互事件日志）
                   - tracy_layer.rs tracing span → Tracy CPU zone（feature = "profile"）

assets/            运行期只读资源（与可写 logs//data/ 分离，见 gate-render/src/paths.rs）
                   - shaders/    blit.wgsl + voxel_raytrace/ WESL 包（main.wesl 入口 + bindings /
                                 common / brickmap / trace / world / lightfield / ddgi/*，
                                 启动时读盘编译，改 shader 需重启）
                   - ui/         theme.ron 暗色主题令牌；debug_menu.toml 菜单初始值
                   - locales/    zh-CN.yml（编译期 codegen 进二进制，运行期不读）
                   - fonts/      MapleMono-NF-CN-Regular.ttf（CJK）+ fa-solid-900.ttf（图标）
                   - lighting/   day_outdoor.ron（当前唯一被加载的主题；dark_lab.ron 暂无引用）
                   - vox/        nuke.vox（默认场景，gitignore）

logs/  data/       运行期可写目录（gitignore）：logs/latest.log、data/ui/debug_menu.toml
dist/              打包产物（bash package.sh 生成，gitignore）
```

---

## 4. 常量体系

**坐标系约定**：1 voxel = 2cm。DDA 内所有坐标（ray origin/dir、cell、tmax）、
`DdaCameraConfig::position_world`、`GridDesc` 的世界 AABB 与 chunk 窗口全部为 **voxel 单位**；
chunk 窗口原点/尺寸为 chunk 单位（×256 即 voxel）。

### 体素 / 树（gate-voxel::coords，渲染侧在 brickmap::wire 同步一份）

| 符号 | 值 | 含义 |
|---|---|---|
| `CHUNK_SIZE` | 256 voxel（= 5.12m） | chunk 边长；存储 / dirty / DDGI 段分配的共同粒度 |
| `BRICK_FACTOR` | 4 | 分裂因子：每节点 4³ = 64 子块 |
| `MAX_LEVEL` / `LEVEL_EXTENT` | 4 / `[256, 64, 16, 4, 1]` | 树层级边长（voxel）；level 2 = 16³ 组件 / DDGI cell 粒度 |
| `PALETTE_ENTRY_COUNT` / `PALETTE_BITS` | 65536 / 16 | 材质索引位宽（`0 = AIR`），索引上限 65535 |
| `PaletteEntry` | 8 B | color\[3\] + roughness + emissive + transmission + flags（`#[repr(C)]`，整表 512KB/volume） |
| `LEAF_INLINE_WORDS` / `LEAF_VOXELS_PER_WORD` | 32 / 2 | level 3 叶父层 inline 存储：32 字，每字 2 个 16 位半字 |
| `CHUNK_COMP_WORDS` | 2048 | comp_layer 每 chunk 字数（4096 个 u16 组件 ID 打包） |
| `STATE_ENTRY_COUNT` / `STATE_WORDS_PER_ENTRY` | 256 / 4 | StateTable 条目与每条目字数 |

### GPU wire（b_struct / b_palette）

| 符号 | 值 | 含义 |
|---|---|---|
| `NODE_FIXED_WORDS` | 3 | 每节点 `[mask_lo, mask_hi, palette_u32]` + popcount(mask) 个 child 偏移 |
| `CHUNK_INDEX_CAP` / `CHUNK_INDEX_WORDS` | 64 / 262144 | 稠密 chunk 窗口（64³ 槽 = 1MB）；窗口外的 chunk 被拒绝并计数 |
| `TREE_BASE` | 262144 | chunk 树区起始字偏移 |
| `PALETTE_WORDS` | 131072 | 调色板字数（2^16 × 2 u32）/volume |
| `MARCH_MASK_*` | 8 octant × 64 入口 × 2 字 = 1024 字 | 方向可达掩码 LUT（b_leaves，4KB） |

### 渲染 / DDGI / 光照场

| 符号 | 值 | 含义 |
|---|---|---|
| `VIEW_SIZE` | 1280×720 | 初始渲染分辨率（窗口内部分辨率 = 物理像素 ÷ `RenderScale.factor`） |
| DDA workgroup | 8×8 | `dda_main` / `beam_main` / `gi_main` 工作组边长（须与 WESL 一致） |
| beam / GI 分辨率 | 1/4 / 1/2 | beam depth 纹理；半分辨率 GI 缓冲（premultiplied valid 格式） |
| DDGI 级联 | 4 级，cell `[16, 32, 64, 128]` voxel | 世界 AABB 锚定、严格嵌套；网格外扩 `DDGI_GRID_MARGIN = 16` voxel |
| LOD0 chunk 段 | 4096 槽/chunk | (256/16)³；池高水位定容，段基址分配后不变 |
| DDGI 图集 | 4×4 irr + 8×8 depth / 探针 | 每层 40² 探针（`DDGI_PROBES_PER_LAYER_AXIS`），272 层 |
| `DDGI_RAY_BUDGET` | 65536 射线/帧 | 摊给活跃探针（worklist 驱动 indirect dispatch） |
| 刷新周期 / 跳过年龄 | `(65, 97, 129, 161)` / `(32, 24, 16, 12)` | 逐 LOD 的探针刷新节奏（WESL `ddgi/consts.wesl`） |
| 光照场 | 32³ cell × 16 voxel | `Rgba16Unorm` 3D 纹理，.a = AO fill；世界覆盖 512 voxel = ±5.12m |
| `SHADOW_SURFACE_EPS` | 1/32 voxel | 阴影/二次射线起点沿法线外推（防自命中，与 WGSL 同步） |

> **跨语言常量的权威在 WESL 源码**：`gate-render/src/wesl_consts.rs` 启动时解析 `ddgi/consts.wesl` 等文件，
> Rust 侧不再各留副本（不一致即启动 fail-fast）。改 DDGI 常量请改 `.wesl`。

---

## 5. 不变量与踩坑（回归必看）

以下每一条都已固化在对应代码注释里，改动相关模块时先核对：

1. **存储 buffer 扩容必须带内容**：`ensure_with_copy` 做 GPU-GPU 前缀拷贝再写尾部；直接重建会把
   chunk 索引表清零 → 全屏空（经典回归）。
2. **上传别逐帧硬扛 backlog**：`poll_pending` 在 backlog > 3× 预算时一次性刷新，否则每次 Prepare 被长阻塞。
3. **树编辑走整块路径**：笔触用 `fill_brick`（O(深度)）而不是逐体素 `set_voxel`（31³ 笔触 = 近 3 万次树下降
   + 沿途 `try_merge`）。
4. **DDGI chunk 段基址一经分配不再改变**：基址变 = 该 chunk 全部探针换槽位 → 图集整段错位 → 闪烁；
   释放走空闲链表 LIFO 复用。
5. **DDA cell 步进必须整数增量维护**：不得用 `floor(origin + dir·t)` 重算（t 恰在边界时 floor 会取到
   穿越前/后的胞 → 对角胞漏检，与 brute-force 不等价）。
6. **bind group 必须从索引 0 起成前缀设置**：自动曝光两个入口的布局因此重复挂 8 份；
   半分辨率 GI 的**采样视图**必须挂 group(5)（挂 BG0 会与 collect 的存储写入在同一 pass 撞 usage）。
7. **blit 采样器必须 Linear**：半分辨率上采样与 FXAA 亚像素偏移都依赖它；Nearest 会让 FXAA 整体空转
   （现象 = "开了没变化"）。
8. **UI 指针捕获闸门**：相机拖拽 / 滚轮 / 编辑射线都要查 `UiPointerCaptured` + `MouseIntercepted`，
   否则操作菜单会同时驱动相机；文本输入框编辑态（`TextInputFocus`）还要挡键盘。
9. **`scale_factor_override = 1.0` 必须每帧重设**：winit 在 resize / 跨显示器时会把 scale factor 刷回
   OS DPI，导致 UI 1px 边框抗锯齿发虚、文字模糊。
10. **UI 必须等字体资产加载完成再 spawn**：否则 TextPipeline 会把字形缓存进不含 CJK 的默认 slot →
    之后即使 override 也是方框。
11. **发光只走「命中直出 + GI」**：光照场的发光密度通道已移除（`Rgba16Unorm.rgb` 恒 0），别再往 .rgb 塞东西。

---

## 6. 环境变量速查

| 变量 | 取值 | 作用 |
|---|---|---|
| `GATE_ROOT` | 路径 | 强制安装根（`assets/`、`logs/`、`data/` 的父目录） |
| `GATE_SCENE` | `vox`（默认）/ `demo` | 默认 `.vox` 场景 / 程序化极限场景 |
| `GATE_TILES` | 2..10（默认 2） | demo 场景规模（1 tile = 512 voxel） |
| `GATE_CAM` | `sky` | 相机朝天空（纯 miss 基准） |
| `GATE_BENCH` | `1` | 失焦后台跑帧（Continuous 更新） |
| `GATE_ORBIT` | `1` | 相机自动边转边平移（配 `GATE_BENCH` 读移动中逐 pass 帧时） |
| `GATE_EDIT_SELFTEST` | `1` | 第 60 帧自动刷一次笔触，无鼠标走通编辑链路 |
| `GATE_RES_SCALE` | ≥1（默认 1） | 渲染内部分辨率降采样倍数（**仅初值**，运行期以菜单为准） |
| `GATE_DDGI_STAGE` | 0..3（默认 3） | DDGI 阶段停靠：0=Off / 1=Active / 2=Cast / 3=Full（基准对比用） |
| `GATE_NO_BEAM` / `GATE_NO_LUT` / `GATE_NO_LOD` / `GATE_NO_EYE_ADAPT` | `1` | 关 beam / 关方向掩码剔除 / 关远场 LOD / 关自动曝光 |
| `GATE_SKYOUT` / `GATE_MAKEGRID_ONLY` / `GATE_SKIP_CHUNKWALK` | `1` | 诊断：只出天空 / 只建 grid 不 trace / 跳过 chunk 步进 |

---

## 7. 质量门禁（本地自测 before commit）

```powershell
# 本地跑法（VS Code → 终端 → 运行任务，或直接敲命令）
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo build --workspace
```

- **`cargo fmt --check`**：代码风格（rustfmt.toml：Google 风，2 空格缩进，edition 2024）。
- **`cargo clippy --workspace --all-targets -- -D warnings`**：lint 零警告。
- **`cargo build --workspace`**：0 error。

> 无 CI：以上三条 + 下面的人工验收靠提交前手动跑。

**手工验收（`cargo run -p gate-app`）**

- 默认 `nuke.vox` 场景出画；`WASD` 飞行与右键转头流畅；`F3` 菜单显隐正常。
- 菜单逐项生效：DDGI 开关 / 诊断模式 / 探针可视化 / 半分辨率 GI；半分辨率 / FXAA / 垂直同步；
  曝光参数（改完 stderr 有 `eye adapt 参数 → GPU` 一行）。
- 左键放置、右键单击擦除 → stderr 出现 `EDIT[...]` 与 `UPLOAD[incremental]`（增量部分写：`bytes` 远小于全量上传）。
- 「游戏/世界/重载世界」换模型后画面整块刷新（DDGI / 光照场跟随重建，无残留旧几何）。
- 长时间运行无 1 秒以上的增量上传长耗时（`UPLOAD[incremental]` 的 elapsed 应在毫秒级）。

**性能剖析（可选）**

```powershell
cargo run -p gate-app --features profile   # 启动后用 Tracy GUI 连接进程
```

逐 pass 的 GPU 均值每 2 秒打印一行；Tracy 时间线上 CPU span（tracing 桥）与 GPU pass 同一帧轴。

---

## 8. 打包发布

```sh
bash package.sh     # release 构建 → 组装便携目录（不压缩）
# 产物：dist/gate-<版本>-win64/{gate-app.exe, assets/}
```

- **产物形态**：`gate-app.exe` 与 `assets/` **同级**，双击即用；整个目录放到任何位置都行。要发 zip/7z 自己压。
- **路径怎么找**（`gate-render/src/paths.rs`，`install_root()` 一套规则覆盖两种形态）：
  1. `GATE_ROOT` 环境变量 → 直接当安装根（自定义安装位置 / 测试）；
  2. exe 同目录存在 `assets/`（便携发布形态）→ 安装根 = exe 所在目录；
  3. 否则 = 源码树根（`cargo run` / F5 时 exe 在 `target/<profile>/`，走这条）。
  于是 `assets/`（只读）、`logs/`、`data/`（可写）在开发与发布下都指向同一套相对位置。
- **必须随包发**：`assets/` 全部内容（字体、`blit.wgsl`、WESL 源码——启动时读盘编译、`ui/theme.ron`、
  `ui/debug_menu.toml`、`lighting/*.ron`、`vox/*.vox`）。
- **无需随包**：`assets/locales/*.yml`（编译期 codegen 进 exe）、`logs/`、`data/`（首次启动自建）。
- **不写安装目录**：日志 → `<安装根>/logs/latest.log`；菜单状态 → `<安装根>/data/ui/debug_menu.toml`
  （启动时优先读它，没有才用 `assets/ui/debug_menu.toml` 初版）。安装到只读目录（如 Program Files）时用
  `GATE_ROOT` 把可写数据挪到别处。
- **注意**：`assets/vox/nuke.vox` 被 gitignore，脚本只在缺失时 WARN，不会阻断打包——发布前自备该文件。
