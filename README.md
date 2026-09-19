# GATE — GPU-Accelerated Tile Engine

> GPU 稀疏体素（Douglas Brick Tree，256³ chunk）+ 计算着色器层次 DDA 光追 + 世界空间 GI（逐 (体素,面) 辐照度缓存）+ 自研 bevy_ui 工具链。
> **当前分支 `restir-gi`**：GI 缓存 / 光照场（AO）/ 自动曝光 / 体素编辑 / 可持久化调试菜单均已落地；默认场景为 MagicaVoxel `nuke.vox`。

---

## 0. 快速开始

```powershell
# 工具链：Rust stable（rust-toolchain.toml，MSVC toolchain）+ Vulkan 显卡驱动
cargo run -p gate-app                     # 默认场景 = assets/vox/nuke.vox（启动场景/规模见 gate-app/src/consts.rs）
cargo run -p gate-app --features profile  # 性能剖析：Tracy GUI 连接进程（CPU span + GPU pass 同时间线）
cargo clippy --workspace --all-targets -- -D warnings
```

> `assets/vox/nuke.vox` 被 `.gitignore` 排除（体积大），新克隆的仓库里没有它。备选：
> 把 `gate-app/src/consts.rs` 的 `STARTUP_DEMO_SCENE` 改成 `true` 跑程序化「极限场景」，或自备一份 `.vox` 放进 `assets/vox/`。

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
| **GI（逐 (体素,面) 辐照度缓存）** | ✅ 生产可用 | 世界空间、相机无关的间接光存储：键 = (体素坐标, 面, 物体)，**面**在键里 ⇒ 薄板两侧天然是两个条目、不跨面插值 ⇒ 不漏光；取值 = 命中体素**自己**的条目（不插值）⇒ 没有级联接缝。存储 = 哈希网格 + 开放寻址按需创建（1M 条目 / 8M 桶），无 atlas / 段池 / 认领位图 / LOD。更新 = `gi_cache_update` 每帧整表轮转 64K 条目，每条目发余弦加权真实光路 + 二次顶点 NEE（回读 + 直射太阳），运行时平均（`E ← E + (c−E)/(n+1)`）⇒ 收敛后确定、无屏幕噪声；新条目从同面同平面 4 邻播种。 |
| **光照场（AO）** | ✅ 生产可用 | 相机中心、世界锚定的 32³ × 16 voxel 网格（`Rgba16Unorm`，硬件三线性），.a = AO fill 直接乘进命中着色；发光走「命中直出自身颜色 + 进 GI」，不再走发光密度通道 |
| **材质与介质** | ✅ 生产可用 | `PaletteEntry { color, roughness, emissive, transmission }`；`transmission > 0` 走玻璃状态机（折射/透射 + 太阳透射率，`trace_glass`）；表面法线与命中体素由整数 DDA 精确产出（禁「命中点 ± 半法线」启发式重建） |
| **自动曝光** | ✅ 生产可用 | UE EyeAdaptation 式：1/16 抽样 → 64 桶 log2 亮度直方图 → 5%~95% 百分位均值 → 分方向时间平滑（变亮/变暗常数分开）；菜单「渲染/曝光」可调 EV± / tau / key |
| **体素编辑** | ✅ 生产可用 | 幽灵模式左键放置 / 右键单击擦除；球 / 立方笔触按 brick 粒度整块写入（整块全在笔触内 → 一次 O(深度) 写，落成 uniform 上级节点）；材质按**内容去重**落调色板槽（改材质不影响旧体素）；编辑 AABB 同时驱动增量上传 + 光照场重算；`EDIT[place\|erase]` 日志 |
| **渲染管线** | ✅ 生产可用 | **无 render node**：extract / prepare / dispatch / blit 全部系统级显式调度；`blit.wgsl` 双入口 `fs_main` / `fs_fxaa`（FXAA 3.11 移植）；半分辨率 `RenderScale` + 线性上采样；MSAA 强制关闭 |
| **调试菜单 + i18n** | ✅ 生产可用 | gate-ui 的 TOML 可序列化 `DebugWindow`（9 种行控件）+ `MenuActionEvent` 观察者；5 个顶层页（视频 / 渲染 / 玩家 / 游戏 / 界面，游戏页下含编辑与世界两个子页）；文案全走 i18n key（`assets/locales/zh-CN.yml` 编译期 codegen，缺键回落中文）；「游戏/世界」可扫 `assets/vox/*.vox` 选择模型并**热重载世界**（光照场随之重建） |
| **世界标签** | ✅ 生产可用 | `WorldAnchor`：世界坐标 → 屏幕像素 UI 标签（距离缩放、CJK 字体延迟解析） |
| **性能剖析** | ✅ 生产可用 | `--features profile`：wgpu-profiler GPU pass 时间戳（Tracy 时间线）+ tracing span → Tracy CPU zone 桥；非 profile 构建零成本 |

### 已知限制 / 待办（不阻塞当前开发）

- **没有自动化测试与 CI**：workspace 0 个 `#[test]`（仅 `vendor/parley` 除外），回归靠手工验收 + 日志。
- **文档缺口**：代码注释引用的 `docs/brickmap.md`、`docs/decisions.md`（ADR-0001/0002/0005）、`docs/ui-dark-theme.md` 尚未落盘；`docs/` 目前只有 Douglas 开发日志转录（`docs/douglas/`）与早期光照截图（`docs/screenshots/`）。
- **正式 sim tick 未实现**：StateTable 只有数据通路与上传，没有逐帧模拟系统驱动它。
- **WorldAnchor 不做体素遮挡判断**（永远绘制在最上层）。
- **`assets/lighting/dark_lab.ron` 暂无代码引用**：当前只加载 `day_outdoor.ron`。
- **wgpu Vulkan 首帧 VUID 报错**（上游已知问题，仅首 1-2 帧 swapchain 时序）：`LogPlugin` filter 静默 `wgpu_hal::vulkan::instance` 与 `surface` 两层。

---

## 2. 技术路线总览

```
[Bevy 主 world]
  Startup : scene::setup —— 读 lighting/*.ron、建 VolumeGrid（默认 vox / demo 程序化，见 consts）、
            初始化 OrbitCamera / FlyCamera / CameraMode / UploadBudget
  Update  : 相机链（模式对齐 → 转头 → 各模式输入 → 拾取 → build_camera_config）→ 体素编辑
            → 调试菜单 / 组件展示窗 / FPS 覆盖层 / 相机信息文本
  Last    : poll_pending —— UploadBudget（4MB/帧）× DirtyTracker → MainPending
      ↓ ExtractSchedule（main → render world）
  extract        : VolumesBuilder 增量/全量构建 → UploadSnapshot + BrickMapDirty(AABB) + LightFieldUpdate
  extract_camera : DdaCameraConfig → DdaViewUniform
  extract_gi     : GiSettings（GI 开关 / 半分辨率）+ 曝光活参数
      ↓ render world
  RenderStartup    : init_dda_pipelines / init_empty_gpu / init_gi_gpu / queue_gi_pipelines
  PrepareResources : prepare_upload（struct/palette/comp/state/grid_descs + 光照场 3D 纹理 + 扩容）
  PrepareBindGroups: prepare_dda_bind_groups → prepare_gi
  RenderGraph::Render（dispatch 顺序）:
      gi_cache_update（可见优先 + 整表轮转：真实光路 + 二次顶点 NEE + 运行时平均）
      → beam（1/4 分辨率最近命中断面预 pass）
      → gi（半分辨率 GI 预 pass，可开关）
      → dda_main（主可见 pass，trace + 着色 → out_tex）
      → eye_histogram + eye_update（自动曝光）
  Core2d::PostProcess: blit_dda_view（fs_main | fs_fxaa，线性上采样 + sRGB cancel → ViewTarget）
```

### 关键技术要点（为什么快）

1. **层次掩码 DDA**：每个非叶节点一次读入 u64 掩码进寄存器，该节点 4³=64 个子块之间步进**零 load**；
   `mask bit=0` 的 uniform 子块整格命中或整格跳过；跨 brick 后用 `firstTrailingBit` 一次跳到最粗可行层。
2. **方向可达掩码 LUT（b_leaves，Douglas #18）**：8 octant × 64 入口格 × 2 字的保守可达集，
   `occupancy & reach` 在进入子块前剔除；LUT 是真实可达集的**超集**，绝不漏命中。
3. **beam 预 pass**：1/4 分辨率先求「最近命中 t」，主 pass 从该 t 起步 —— 近场空空间不产生步进。
4. **世界空间 GI 缓存**：条目键 = (体素坐标, 面, 物体)，世界锚定 ⇒ 相机移动不换主、不闪烁；
   取值 = 命中体素自己的条目、不插值 ⇒ 没有级联接缝；面在键里 ⇒ 薄板/墙缝不漏光。
   更新 = 可见优先（着色侧 `atomicExchange` 精确 claim + append 紧凑列表，数量超线程数时取轮换窗口）
   ＋ 按帧号整表轮转（`GI_CACHE_SLOTS / 预算` 帧扫完一遍）⇒ 无跨帧游标、无竞态、丢帧不漏更新；
   变化失效 = 世代号（太阳/天光/palette/世界全量）＋ N 个世界 voxel 脏盒（主世界 + 物体 volume）。
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
                   - gi.rs          GI 缓存（GiUniform / GiSettings / GiGpu / BG4·BG5 布局 /
                                   prepare_gi / dispatch_gi：世代 + 脏盒失效、可见 claim、
                                   整表轮转更新 pass 的调度）
                   - lighting.rs    LightingTheme（RON）+ LightPoolUniform wire 契约（BG3）
                   - shader.rs      启动时 wesl-rs 编译 WESL 包 → Bevy Shader 资产
                   - wesl_consts.rs 从 .wesl 源码解析跨语言 u32 常量（GI 缓存容量等的**唯一权威**），启动 fail-fast
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
                   - scene.rs      setup + 程序化极限场景 build_demo_scene + reload_world 换世界
                   - camera.rs     CameraMode（Orbit|Fly）/ FlyCamera / 输入系统 / cursor_ray / 拾取 recenter
                   - edit.rs       EditSettings + BrushShape/BrushMaterial + raycast_main + 笔触施加与输入
                   - vox_scene.rs  MagicaVoxel .vox 导入（vox-rs）+ scan_vox_models 模型发现
                   - debug_menu.rs 菜单树默认定义 + 状态应用/落盘 + FPS 覆盖层 + 相机信息 + F3 开关
                   - showcase.rs   右上角组件展示窗（控件画廊 + 交互事件日志）
                   - tracy_layer.rs tracing span → Tracy CPU zone（feature = "profile"）

assets/            运行期只读资源（与可写 logs//data/ 分离，见 gate-render/src/paths.rs）
                   - shaders/    blit.wgsl + voxel_raytrace/ WESL 包（main.wesl 入口 + bindings /
                                 common / brickmap / trace / world / lightfield / gi/*，
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
| `CHUNK_SIZE` | 256 voxel（= 5.12m） | chunk 边长；存储 / dirty 的共同粒度 |
| `BRICK_FACTOR` | 4 | 分裂因子：每节点 4³ = 64 子块 |
| `MAX_LEVEL` / `LEVEL_EXTENT` | 4 / `[256, 64, 16, 4, 1]` | 树层级边长（voxel）；level 2 = 16³ 组件粒度 |
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

### 渲染 / GI / 光照场

| 符号 | 值 | 含义 |
|---|---|---|
| `VIEW_SIZE` | 1280×720 | 初始渲染分辨率（窗口内部分辨率 = 物理像素 ÷ `RenderScale.factor`） |
| DDA workgroup | 8×8 | `dda_main` / `beam_main` / `gi_main` 工作组边长（须与 WESL 一致）；`gi_cache_update` 为 64 |
| beam / GI 分辨率 | 1/4 / 1/2 | beam depth 纹理；半分辨率 GI 缓冲（premultiplied valid 格式） |
| `GI_CACHE_SLOTS` | 1,048,576 条目 | 条目表容量（每条 `GI_CACHE_ENTRY_WORDS` × 4B = 72B ⇒ 72MB）；满则不再新建条目 |
| `GI_CACHE_BUCKETS` | 8,388,608 桶 | 哈希网格桶数（2 的幂，×4B = 32MB）；桶内存「条目下标 + 1」，0 = 空 |
| `GI_CACHE_ENTRY_WORDS` | 18 | 条目布局：`[0,1]` 键、`[2,4]` 世界位置、`[5]` 世界法线（八面体 f16x2）、`[6,8]` E、`[9]` 已累积样本数、`[10]` 最近**处理**（更新/轮转，同帧去重用）的帧、`[11]` 世代号、`[12]` reservoir 方向、`[13,15]` reservoir 样本的 L、`[16]` reservoir `w_sum`、`[17]` reservoir 候选数 M |
| `GI_CACHE_RAYS` | 2 | 每条目每帧发射的**新鲜**余弦加权半球射线数 |
| `GI_CACHE_REUSE` | 2 | 每条目每帧额外的**借用**射线数（RIS 复用：借同平面邻条目 reservoir 的方向，从本条目重发） |
| `GI_CACHE_M_CAP` | 32 | RIS 候选数上限 M：超过则把 `w_sum` 与 M 同比例缩回（`w_sum / M` 不变），抑制历史对抽样的支配 |
| `GI_CACHE_UPDATE_BUDGET` | 65536 条目/帧 | 更新 pass 线程数（可见段 + 轮转段的总预算） |
| `GI_CACHE_VISIBLE_CAPACITY` | 1,048,576 项 | 可见条目紧凑列表容量（×4B = 4MB）；着色侧 claim 后 append、更新 pass 消费，溢出即丢弃 |
| `GI_CACHE_DIRTY_BOXES` | 8 | 单帧生效的脏盒数上限（主世界 + 各物体 volume 的编辑各一盒）；超出退化为「自增世代」全量失效 |
| `GI_CACHE_PROBE` | 8 | 开放寻址最大探测步数 |
| `GI_CACHE_RAY_BIAS` | 0.5 voxel | 更新射线起点沿条目法线的外推（防第一步撞自己） |
| 光照场 | 32³ cell × 16 voxel | `Rgba16Unorm` 3D 纹理，.a = AO fill；世界覆盖 512 voxel = ±5.12m |
| `SHADOW_SURFACE_EPS` | 1/32 voxel | 阴影/二次射线起点沿法线外推（防自命中，与 WGSL 同步） |

### 4.1 GI 着色 / 光路旋钮（都在 `gi/consts.wesl`）

| 符号 | 值 | 含义 |
|---|---|---|
| `GI_PI` | 3.14159265 | 辐照度 ↔ 辐亮度换算（缓存存 `E`，着色侧乘增益后除 π） |
| `GI_T_MAX` | 8192 voxel | 更新射线 / 阴影射线的 t 上限 |
| `GI_SKY_RADIANCE_SCALE` | 1.0 | 更新射线 miss（逃逸到天空）时注入的天空辐亮度系数 |
| `GI_SUN_BOUNCE` | 1.0 | 更新射线命中面的**直射太阳**项增益（GI 唯一的直射光源）；0 = 整段折掉、连阴影射线都不发 |
| `GI_EMIT_GAIN` | 1.0 | 命中自发光体素时的辐亮度增益 |
| `GI_SKY_AMBIENT` | 0.05 | 天光环境项：缓存没有数据（`cov = 0`）时的兜底强度 |
| `GI_AMBIENT_FLOOR` | 0.01 | 天光地板：`cov = 1`（有数据）时保留 k 倍 —— `amb = sky · GI_SKY_AMBIENT · mix(1.0, k, cov)` |
| `GI_CACHE_VISIBLE_SHARE` | 50（%） | 更新预算里「可见优先」占比（50 = 1/2）：先处理着色侧上一帧 claim + append 的可见条目（数量超过可见段线程数时取轮换窗口：起点 `(frame × vis_threads) % cnt`）；置 0 关闭（全部走轮转） |
| `GI_CACHE_CONVERGED_N` | 64 | 「已收敛」阈值（`n ≥ 本值`）：**可见段与轮转段**都每 4 帧才更新一次；旧世代/脏区条目 `n` 被重置 ⇒ 永远优先且不降频 |
| `GI_CACHE_DIRTY_MARGIN_VOXELS` | 2 voxel | 脏盒失效余量基准：盒 0（主世界，scale = 1）用它；物体 volume 的盒按 `max(本值, 本值 × scale)` 放大（权威值在 `gi/consts.wesl`，Rust 经 `wesl_consts.rs` 解析） |

两条硬约束：

1. **GI 的直射光源只有两项**：更新射线命中面的太阳项 + 射线逃逸到天空。命中面的入射项与直射项都必须带 1/π
   （否则"太阳反弹"会比"缓存多弹跳"亮 π 倍，能量不自洽）。删掉太阳项 ⇒ 任何看不到天空的表面
   （草根、过道内墙）只能自洽维持全黑。
2. **太阳阴影射线的起点必须跟着色侧同一套外推**（半个 voxel + `SHADOW_SURFACE_EPS`）：只退 1 个 eps 会从
   **命中体素内部**出发，第一步就自命中 ⇒ 该项恒为 0，白耗一根射线。

> **跨语言常量的权威在 WESL 源码**：`gate-render/src/wesl_consts.rs` 启动时解析 WESL 包 `gi/`
> （`cache.wesl` 的容量/布局 + `consts.wesl` 的旋钮），Rust 侧不再各留副本（不一致即启动 fail-fast）。
> 改缓存容量 / 调度比例 / 脏区余量基准请改 `.wesl`。

> **帧号是精确 u32**：`GiUniform.seq.x`（`vec4<u32>`）承载自增帧号，所有整数帧逻辑（轮转起点、
> 条目 `[10]`「本帧已处理」去重、可见 claim、RNG 种子混入）都只用它 —— 帧号曾以 f32 存在
> `params.x`，超过 2^24 后 f32 不能表示连续整数，上述判断会偶发失效。`params.x` 保留为恒 0 占位。

---

## 5. 不变量与踩坑（回归必看）

以下每一条都已固化在对应代码注释里，改动相关模块时先核对：

1. **存储 buffer 扩容必须带内容**：`ensure_with_copy` 做 GPU-GPU 前缀拷贝再写尾部；直接重建会把
   chunk 索引表清零 → 全屏空（经典回归）。
2. **上传别逐帧硬扛 backlog**：`poll_pending` 在 backlog > 3× 预算时一次性刷新，否则每次 Prepare 被长阻塞。
3. **树编辑走整块路径**：笔触用 `fill_brick`（O(深度)）而不是逐体素 `set_voxel`（31³ 笔触 = 近 3 万次树下降
   + 沿途 `try_merge`）。
4. **GI 缓存条目容量满即停止新建**：`gi_alloc` 里 `atomicAdd` 出的游标 ≥ `GI_CACHE_SLOTS` 就返回失败
   （当作"没有条目"），不会越界写。桶是「键先写、桶后 CAS」⇒ 抢输的孤儿条目只会短暂命中不到，不会读错。
5. **DDA cell 步进必须整数增量维护**：不得用 `floor(origin + dir·t)` 重算（t 恰在边界时 floor 会取到
   穿越前/后的胞 → 对角胞漏检，与 brute-force 不等价）。
6. **bind group 必须从索引 0 起成前缀设置**：自动曝光两个入口的布局因此重复挂 8 份；
   半分辨率 GI 的**采样视图**必须挂 group(5)（同一 pass 内同一张纹理不能既采样又作存储写入）。
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

## 6. 可调常量（旋钮）

没有环境变量开关：所有可调项都是编译期常量，每个目录一个 `consts.rs`（贴着使用方）。

| 文件 | 内容 |
|---|---|
| `gate-app/src/consts.rs` | 启动场景 / demo 规模 / 相机（FOV、裁剪面、灵敏度、飞行速度）/ 编辑笔触 / 菜单与展示窗 |
| `gate-render/src/consts.rs` | 光照增益 / 响应式尺寸 / profiler 周期 / GI 增益 |
| `gate-render/src/brickmap/consts.rs` | DDA（分辨率、beam、FOV、诊断开关）/ 上传预算与缓冲策略 |
| `gate-ui/src/consts.rs` | `UiScale` 自适应基准 |
| `gate-ui/src/menu/consts.rs` | 菜单容器布局与动画 |
| `gate-ui/src/widgets/consts.rs` | widget 尺寸与手感 |

- 取值：`const` 直接读（改值后重新编译）。要让某项运行期可调，把它挪进 Bevy 资源（`RenderScale` / `EyeAdaptSettings` / `GiSettings` 是现成例子），再由 debug_menu 的节点 + 回调写它。
- WESL 侧常量仍是 shader 的权威值（`wesl_consts.rs` 启动时解析同一份源码），不在 Rust `consts.rs` 里重复。

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
- 菜单逐项生效：GI 开关 / 半分辨率 GI；半分辨率 / FXAA / 垂直同步；
  曝光参数（改完 stderr 有 `eye adapt 参数 → GPU` 一行）。
- 左键放置、右键单击擦除 → stderr 出现 `EDIT[...]` 与 `UPLOAD[incremental]`（增量部分写：`bytes` 远小于全量上传）。
- 「游戏/世界/重载世界」换模型后画面整块刷新（光照场跟随重建，无残留旧几何）。
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
  1. exe 同目录存在 `assets/`（便携发布形态）→ 安装根 = exe 所在目录；
  2. 否则 = 源码树根（`cargo run` / F5 时 exe 在 `target/<profile>/`，走这条）。
  于是 `assets/`（只读）、`logs/`、`data/`（可写）在开发与发布下都指向同一套相对位置。
- **必须随包发**：`assets/` 全部内容（字体、`blit.wgsl`、WESL 源码——启动时读盘编译、`ui/theme.ron`、
  `ui/debug_menu.toml`、`lighting/*.ron`、`vox/*.vox`）。
- **无需随包**：`assets/locales/*.yml`（编译期 codegen 进 exe）、`logs/`、`data/`（首次启动自建）。
- **不写安装目录**：日志 → `<安装根>/logs/latest.log`；菜单状态 → `<安装根>/data/ui/debug_menu.toml`
  （启动时优先读它，没有才用 `assets/ui/debug_menu.toml` 初版）。
- **注意**：`assets/vox/nuke.vox` 被 gitignore，脚本只在缺失时 WARN，不会阻断打包——发布前自备该文件。
