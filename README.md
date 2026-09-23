# GATE — GPU-Accelerated Tile Engine

> GPU 稀疏体素（Douglas Brick Tree，256³ chunk）+ 计算着色器层次 DDA 光追 + 屏幕空间 ReSTIR GI + 自研 bevy_ui 工具链。
> **当前分支 `restir-gi`**：ReSTIR GI / 光照场（AO）/ 自动曝光 / 体素编辑 / 可持久化调试菜单均已落地；默认场景为 MagicaVoxel `nuke.vox`。

---

## 0. 快速开始

```powershell
# 工具链：Rust stable（rust-toolchain.toml，MSVC toolchain）+ Vulkan 显卡驱动
# 全 workspace 禁止 debug 构建（各 crate 的 build.rs 各内联同一份判据）：所有 cargo 命令一律加 --release
cargo run --release -p gate-app                     # 默认场景 = assets/vox/nuke.vox（启动场景/规模见 gate-app/src/consts.rs）
cargo run --release -p gate-app --features profile  # 性能剖析：Tracy GUI 连接进程（CPU span + GPU pass 同时间线）
cargo clippy --release --workspace --all-targets -- -D warnings
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

- UI 结构与控件缺省值固定在 `assets/ui/debug_menu.toml`（随包发布，每次启动都读）；
  运行期改动过的控件值 + 窗口位置/收起/停留路径 + 相机姿态在退出时写回 `<安装根>/data/config.toml`，
  下次启动读它覆盖缺省值（配置里没有的控件用缺省，结构里没有的路径忽略）

---

## 1. 项目现状

| 模块 | 状态 | 说明 |
|---|---|---|
| **体素核心（gate-voxel）** | ✅ 生产可用 | Douglas Brick Tree：`HashMap<ChunkCoord, ChunkTree>`，每 chunk 256³ voxel，分裂因子 4³（256 → 64 → 16 → 4 → 1）；非叶节点 u64 占用掩码 + 紧凑 child 偏移表；`Node::Uniform` 自适应叶；16 位材质索引（65536 槽，8B/条）；三级查询 `get_voxel` / `get_brick_state` / `fill_brick`（O(深度) 整块写）；`try_merge` + `compact()` DFS GC |
| **多 volume** | ✅ 生产可用 | `Volumes`：`list[0]` = 主世界（`obj_id = -1`），`add_object` 注册物体（`VolumeTransform { pos, rot, scale }`）；同一 dirty → builder → upload 路径；GPU 侧为统一 `GridDesc` 数组（144B/条），shader `trace_scene` 无分支遍历 |
| **组件层 / 状态表** | ✅ 数据通路可用 | `comp_layer`：每 chunk 4096 个 16³ 组件 ID（u16）；`StateTable`：256 条 × 4×u32；随 dirty 双通道（data / comp）分别上传。**尚无逐帧模拟驱动**（仅 demo 场景写测试值） |
| **GPU 上传** | ✅ 生产可用 | `b_struct`（64³ 稠密 chunk 窗口 + 各 chunk DFS 序列化树）+ `b_palette`（512KB/volume）+ `globals`；脏区增量部分写（struct 字区间 + palette 槽区间）；扩容 `ensure_with_copy`（GPU-GPU 前缀拷贝）；backlog > 3× 预算时一次性刷新，避免逐帧阻塞 Prepare；日志 `UPLOAD[full\|incremental]` |
| **DDA 光追** | ✅ 生产可用 | WESL 包（`assets/shaders/voxel_raytrace/`）启动时读盘编译；层次栈式 mask DDA（节点掩码常驻寄存器，4³ 子块间步进零 load；`firstTrailingBit` 跨级跳）；方向可达掩码 LUT（Douglas #18 Bitwise Masking）辅助剔除；beam 低分辨率最近命中断面预 pass；局部 AABB slab 剔除 |
| **GI（屏幕空间 ReSTIR）** | ✅ 生产可用 | 逐像素一个 reservoir（每 GI 像素 `GI_RES_WORDS` 个 word × 2 块 ping-pong），`gi_main` 一次派发完成「新鲜候选（4 条余弦；「高」档 8 条；去遮挡再翻倍）→ 时域复用 → 空间复用 → 着色」。复用判据 = **同一个面**（平面归属靠面键，相似度靠**着色法线** `dot ≥ GI_DEN_N_DOT`）⇒ 薄板/墙缝不漏光，而凸棱上被梯度法线平均过的那一段仍与相邻同面体素连成一片。**空间复用零射线**（8 tap 只并上一帧邻居的累计量），且**只作用在本帧估计、不回写时域历史**。辐亮度 = 单次弹射（命中面按「太阳直射 + 天光×AO」着色后除 π，miss 取天光）⇒ 输入只依赖几何与光照、逐帧确定，第 1 帧即稳态。降噪 = 时域累积（输入在「面不跨 texel」时取**同一面邻域均值**做预平均 + 方差驱动历史权重 + 累积矩给出的平滑噪声尺度 + AABB 钳制 + 离群抑制，上限 96 帧）→ 5 轮 atrous（同面键满权重、其他走法线点积 × 亮度权重）；回全分辨率用**几何感知上采样**。菜单「渲染/ReSTIR GI」开关 + 分辨率档（1/1、1/2、1/4；每档都跑 GI，只是网格疏密不同）+「降噪质量」档（关/低/中/高，**与分辨率档正交**：关 = 一条降噪 pass 都不跑、直接采样原始 GI；低 = 时域 + 5 轮 3×3；中 = atrous 换 5×5；高 = 再把每像素候选数翻倍即 GI 射线翻倍 + 记忆窗 20→32 帧）。此外：**① 逐面合并**（同一个可见体素面的全部 texel 的候选是**同一个积分**的 i.i.d. 样本 ⇒ `gi_main` 定点累加、`gi_face_flatten` 在**降噪之前**把整面换成均值，方差 ÷ texel 数）；**② 二次顶点缓存跨帧持久**（槽里的键带 epoch 掩码：几何 / 太阳 / 天光 / 光照场窗口任一变化即整表自失效 ⇒ 不再每帧 `clear_buffer`）；**④ 二次弹射**（菜单「光照/二次弹射」：二次顶点再加一跳的间接光，稀疏档按 `1/4` 概率带上并乘 `4` 补齐期望 ⇒ 无偏）。 |
| **光照场（AO）** | ✅ 生产可用 | 相机中心、世界锚定的 32³ × 16 voxel 网格（`Rgba16Unorm`，硬件三线性），.a = AO fill 直接乘进命中着色（间接项 / 天光项共 4 个调用点，唯一出口是 `light_field_ao`）；**菜单「渲染/基础/AO」可整条关掉**（系数恒 1.0，且 CPU 重铺与上传一并停掉）；发光走「命中直出自身颜色 + 进 GI」，不再走发光密度通道 |
| **材质与介质** | ✅ 生产可用 | `PaletteEntry { color, roughness, emissive, transmission }`；`transmission > 0` 走玻璃状态机（折射/透射 + 太阳透射率，`trace_glass`）；表面法线与命中体素由整数 DDA 精确产出（禁「命中点 ± 半法线」启发式重建） |
| **自动曝光** | ✅ 生产可用 | UE EyeAdaptation 式：1/16 抽样 → 64 桶 log2 亮度直方图 → 5%~95% 百分位均值 → 分方向时间平滑（变亮/变暗常数分开）；菜单「渲染/曝光」可调 EV± / tau / key |
| **体素编辑** | ✅ 生产可用 | 幽灵模式左键放置 / 右键单击擦除；球 / 立方笔触按 brick 粒度整块写入（整块全在笔触内 → 一次 O(深度) 写，落成 uniform 上级节点）；材质按**内容去重**落调色板槽（改材质不影响旧体素）；编辑 AABB 同时驱动增量上传 + 光照场重算；`EDIT[place\|erase]` 日志 |
| **渲染管线** | ✅ 生产可用 | **无 render node**：extract / prepare / dispatch / blit 全部系统级显式调度；`blit.wgsl` 双入口 `fs_main` / `fs_fxaa`（FXAA 3.11 移植）；半分辨率 `RenderScale` + 线性上采样；MSAA 强制关闭 |
| **调试菜单 + i18n** | ✅ 生产可用 | gate-ui 的 TOML 可序列化 `DebugWindow`（9 种行控件）+ `MenuActionEvent` 观察者；5 个顶层页（视频 / 渲染 / 玩家 / 游戏 / 界面，游戏页下含编辑与世界两个子页）；文案全走 i18n key（`assets/locales/zh-CN.yml` 编译期 codegen，缺键回落中文）；「游戏/世界」可扫 `assets/vox/*.vox` 选择模型并**热重载世界**（光照场随之重建） |
| **世界标签** | ✅ 生产可用 | `WorldAnchor`：世界坐标 → 屏幕像素 UI 标签（距离缩放、CJK 字体延迟解析） |
| **性能剖析** | ✅ 生产可用 | `--features profile`：wgpu-profiler GPU pass 时间戳（Tracy 时间线）+ tracing span → Tracy CPU zone 桥；非 profile 构建零成本 |

### 已知限制 / 待办（不阻塞当前开发）

- **没有自动化测试与 CI**：workspace 0 个 `#[test]`，回归靠手工验收 + 日志。
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
  extract_gi     : GiSettings（GI 开关 / 分辨率除数）+ 曝光活参数
      ↓ render world
  RenderStartup    : init_dda_pipelines / init_empty_gpu / init_gi_gpu / queue_gi_pipelines
  PrepareResources : prepare_upload（struct/palette/comp/state/grid_descs + 光照场 3D 纹理 + 扩容）
  PrepareBindGroups: prepare_dda_bind_groups → prepare_gi
  RenderGraph::Render（dispatch 顺序）:
      beam（1/4 分辨率最近命中断面预 pass）
      → gi（GI 预 pass：反投影 + G-Buffer + ReSTIR 取样，可开关）
      → gi_face_flatten（① 逐面合并：把每个 GI texel 的估计换成"它所在那个可见体素面"的均值）
      → gi_denoise_temporal → gi_denoise_atrous1/2/4（时域累积 + 迭代 atrous）
      → dda_face_accum → dda_face（逐面 GI 均值 + 逐面着色一次，写回槽）
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
4. **屏幕空间 ReSTIR GI**：逐像素一个 reservoir，每帧「新鲜候选 → 时域复用 → 空间复用 → 着色」，
   再走降噪：**SVGF 家族**（时域累积：方差驱动历史权重 + 累积矩 + AABB 钳制、上限 32 帧）→
   5 轮迭代 atrous（步长 1/2/4/8/16）。NRD / RELAX / FidelityFX 的处方已对标闭环：
   离群抑制并进时域 pass（`mean ± K·σ`）；**history-fix** = `GI_HIST_SEARCH_TAPS`（落点不接受时
   在它的 3×3 邻域里继续按**同一套判据**找历史，见 5g）；**短历史用模糊兜** = `GI_DEN_STAB_FRAMES`
   （`M` 小的像素把 atrous 步长放大 ⇒ NRD 的 "blur instead of noise"）；**历史长度上限** =
   `GI_DEN_M_MAX`（NRD `maxAccumulatedFrameNum` 量级，见 5i）；**宽核统计** = `GI_DEN_NBR_R`
   （2 ⇒ 5×5：钳制与本帧输入都要求邻域非空）；**变化即重置** = 世界的体素占据一变，**reservoir** 该帧
   不复用任何历史（NRD 的 `accumulation speed = 1.0 on change`；降噪侧不重置，靠 ⑤ 的 AABB 钳制跟随，
   见 5j）。接受与否仍是**二值**的 —— 短历史的代价由上面那条模糊补偿，而不是靠一段被信过的短历史。
   核半径（5×5 的 24 tap / 3×3 的 8 tap）与「是否跑降噪」由菜单「降噪质量」档（关/低/中/高）定，
   该档**与分辨率档正交**。复用判据是**同一个面**：
   **平面归属**用面键（面号 + 物体 + 沿面法线轴的体素坐标相等），只在 DDA 入射面上比 —— 精确、
   与缩放无关 ⇒ 薄板 / 墙缝 / 隔一格的平行面都不漏光；**相似度**用**着色法线**（`dot ≥ GI_DEN_N_DOT`）
   —— 容忍凸棱上被梯度法线平均出来的 45° 倾斜，与「画面看起来是连续的一段」一致。
   时域接受在判据之外还做一次**邻域搜索**：重投影落点不接受时，按 `GI_HIST_SEARCH_TAPS`
   在它的 3×3 邻域里继续找（正中恒排第一 ⇒ 静止相机逐位不变）。修的是"**重投影落在哪个 texel**"
   这个定位问题，不是放宽几何条件 —— 斜面上一个体素面常常只占一个 texel，"包含落点的那个 texel"
   有相当概率采到了相邻台阶。
   着色法线只在 `gi_main` 算**一次**（导引 / reservoir / 历史三处存它），`dda_main` 在 `gi_div = 1`
   时直接读它 ⇒ 全管线不重复算这个隐式法线。
   **GI 的输入只依赖几何与光照**（单次弹射，辐亮度不含任何跨帧累积量）⇒ 第 1 帧就是稳态，
   没有「收敛」这件事，也没有相机运动相关的延迟。代价是丢掉二次以上的弹射。
   GI 回全分辨率时用**几何感知上采样**（joint bilateral：只接受与命中面同一个面的
   GI texel）⇒ 棱边 / 墙角不渗色（纯双线性会混边界两侧）。
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
                   - gi.rs          GI（GiUniform / GiSettings / GiGpu / BG4·BG5 布局 /
                                   prepare_gi：reservoir 双缓冲换绑 + uniform 上传 + 几何修订号）
                   - lighting.rs    LightingTheme（RON）+ LightPoolUniform wire 契约（BG3）
                   - shader.rs      启动时 wesl-rs 编译 WESL 包 → Bevy Shader 资产
                   - wesl_consts.rs 从 .wesl 源码解析跨语言 u32 常量（buffer 布局的**唯一权威**），启动 fail-fast
                   - profiler.rs    wgpu-profiler / Tracy 集成（feature = "profile"；非该 feature 为零成本壳）
                   - responsive.rs  窗口 resize → 渲染目标原地重建 + RenderScale 跟随
                   - paths.rs       install_root / assets_dir / logs_dir / data_dir

gate-ui/           自研 bevy_ui 组件库 + 调试菜单 + 世界标签
                   - theme.rs       UiTheme（RON 资产 + 内置暗色默认）、令牌（颜色/圆角/间距/字号）、UiScale
                   - icon.rs        FontAwesome 字形（菜单标题栏按钮与箭头）
                   - i18n.rs        UiTranslator：key → 文案解析器（切语言后整体重解析）
                   - capture.rs     UI 指针门控（**不是**截图工具）：`UiPointerCaptured` 由 picking
                                    悬停映射得出、`MouseIntercepted` 由 `MouseIntercept` 节点的
                                    悬停态得出（半透明遮罩等「吞掉鼠标」语义挂 `MouseIntercept`）
                   - pointer.rs     指针交互基座：picking 的 `Hovered`（悬停）+ 观察者维护的
                                    `Pressed`（按住）→ 控件侧 `UiInteract`（None/Hovered/Pressed）；
                                    `UiInteractBundle` 是交互控件根三件套（含 `Pickable` 命中标记）
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
                   - debug_menu.rs 菜单结构/缺省值加载 + 配置值合并 + 状态应用 + FPS 覆盖层 + 相机信息 + F3 开关
                   - config.rs     <安装根>/data/config.toml 读写（菜单窗口/控件值 + 相机姿态）+ 退出落盘
                   - showcase.rs   右上角组件展示窗（控件画廊 + 交互事件日志）
                   - tracy_layer.rs tracing span → Tracy CPU zone（feature = "profile"）

assets/            运行期只读资源（与可写 logs//data/ 分离，见 gate-render/src/paths.rs）
                   - shaders/    blit.wgsl + voxel_raytrace/ WESL 包（main.wesl 入口 + bindings /
                                 common / brickmap / trace / world / lightfield / gi/*，
                                 启动时读盘编译，改 shader 需重启）
                   - ui/         theme.ron 暗色主题令牌；debug_menu.toml 菜单结构与控件缺省值
                   - locales/    zh-CN.yml（编译期 codegen 进二进制，运行期不读）
                   - fonts/      MapleMono-NF-CN-Regular.ttf（CJK）+ fa-solid-900.ttf（图标）
                   - lighting/   day_outdoor.ron（当前唯一被加载的主题；dark_lab.ron 暂无引用）
                   - vox/        nuke.vox（默认场景，gitignore）

logs/  data/       运行期可写目录（gitignore）：logs/latest.log、data/config.toml
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
| DDA workgroup | 8×8 | `dda_main` / `beam_main` / `gi_main` / 降噪 4 个入口的工作组边长（须与 WESL 一致） |
| beam / GI 分辨率 | 1/4 / 1/2 | beam depth 纹理（1/4 全分辨率）；GI 缓冲档位 `GiSettings.gi_div` = 1 / 2 / 4（菜单「渲染/GI/分辨率」，默认 1/2；premultiplied valid 格式） |
| `GI_RES_WORDS` | 8 | 每 GI 像素的 reservoir 字数（布局见 `gi/screen.wesl`：状态位 + 面键 2 + **着色法线** + 累计量 rgb + `M`）。**只有累计量与判据** —— 曾经的"代表样本"（方向 / 辐亮度 / 二次顶点键）随逐样本校验一起删掉了，见 5j |
| 光照场 | 32³ cell × 16 voxel | `Rgba16Unorm` 3D 纹理，.a = AO fill；世界覆盖 512 voxel = ±5.12m。**菜单「渲染/基础/AO」清零 `base_flags` 的 bit2 即整条关掉**（`light_field_ao` 恒返回 1.0，CPU 重铺与上传也跳过） |
| `SHADOW_SURFACE_EPS` | 1/32 voxel | 阴影/二次射线起点沿法线外推（防自命中，与 WGSL 同步） |

### 4.1 GI 着色 / 光路旋钮（都在 `gi/consts.wesl`）

| 符号 | 值 | 含义 |
|---|---|---|
| `GI_PI` | 3.14159265 | 辐照度 ↔ 辐亮度换算（reservoir 存 π·L，着色侧乘增益后除 π） |
| `GI_T_MAX` | 8192 voxel | GI 射线 / 阴影射线的 t 上限 |
| `GI_SKY_RADIANCE_SCALE` | 1.0 | GI 射线 miss（逃逸到天空）时注入的天空辐亮度系数 |
| `GI_SUN_BOUNCE` | 1.0 | 二次顶点命中面的**直射太阳**项增益（GI 的直射光源）；0 = 该项折掉、连阴影射线都不发 |
| `GI_EMIT_GAIN` | 1.0 | 命中自发光体素时的辐亮度增益 |
| `GI_SKY_AMBIENT` | 0.05 | 天光环境项：GI 无数据（`cov = 0`）时的兜底强度；也是二次顶点唯一的天空项入口 |
| `GI_AMBIENT_FLOOR` | 0.01 | 天光地板：`cov = 1`（GI 有数据）时保留 k 倍 —— `amb = sky · GI_SKY_AMBIENT · mix(1.0, k, cov)` |
| `GI_SS_CAND_N` | 4 | 每像素每帧的**新鲜**余弦候选数 · 「降噪质量」档 0/1/2（每条含 ≤1 条太阳 NEE 阴影射线），GI 的代价几乎全在这里。代价按「每屏幕面积的射线数 = 候选数 / 除数²」计 |
| `GI_SS_CAND_N_HQ` | 8 | 每像素新鲜候选数 · 「降噪质量」**档 3（高）**：射线翻倍（全链最贵的一项）。**与分辨率档无关**（1/1 档也翻倍——那是极限画质的对照点）。1/4 + 8 = 0.5 条/屏幕像素（仍是 1/2 档的一半），每 texel 噪声 σ/2.83 |
| `GI_SS_M_CAP_K` / `_HQ` | 20 / 32 | reservoir 候选数上限 = 本值 × 本帧新鲜候选数（超过则同比例缩回，`w_sum / M` 不变）⇒ **滑动窗口 = 本值帧**（与候选数、去遮挡 boost 都无关）。几何变化不吃这个窗（判据精确），只对**光照**变化滞后（32 帧 ≈ 0.5s@60fps）。本项**零性能成本**；`_HQ` 属「降噪质量」档 3 |
| `GI_SS_REUSE_TAPS` | 8 | 空间复用 tap 数（3×3 环；同一个面 ⇒ **只并累计量，零射线**） |
| `GI_HIST_SEARCH_TAPS` | 8 | **历史搜索**的候选 tap 数（0 = 关，8 = 落点周围满 3×3；正中恒为第一个候选）：重投影落点不接受时，按**同一套判据**在邻域里继续找。静止 / 纯相机运动下**零额外射线**（世界几何逐位未变 ⇒ 验证射线整条省掉） |
| `GI_DEN_M_MAX` | 32 | 时域累积的历史长度上限 M（帧）⇒ 稳态历史权重上限 31/32，τ ≈ 0.53s@60fps，而 τ 就是「光照 / 远处几何**渐变**」的响应时间。**这是「拖影 ↔ 静态闪烁」的主旋钮**：本工程曾取 96（τ ≈ 1.6s、残余闪烁 ≈ 9.3%），现取 32（≈16.2%）对齐 NRD / RTX Remix 的量级（见 5i） |
| `GI_DEN_NBR_R` | 2 | 时域统计的**邻域半径**（`(2R+1)²` tap，含中心）：`2` = 5×5（25 tap）/ `1` = 3×3（9 tap）。钳制、σ、本帧输入**全都要求邻域非空**，而斜面上「一个体素面 ≈ 一个 texel」⇒ 取 1 时 `cnt` 停在 1、钳制整段不执行 ⇒ 历史色没有护栏（拖影）。FidelityFX 为此用 17×17 取矩 |
| `GI_DEN_STAB_FRAMES` / `_MAX` | 8.0 / 3.0 | **短历史稳定化**：atrous 有效步长 × `clamp(8/M, 1, 3)` ⇒ `M = 1` 的像素步长 ×3、用宽模糊代替生噪点（NRD 的 `blurRadiusScale`）；`M ≥ 8` ⇒ ×1 ⇒ **稳态画面逐位不变**。零额外 tap |
| `GI_DEN_ATROUS_ITER` | 5 | 迭代 atrous 轮数（步长 1/2/4/8/16；= NRD/RELAX 的迭代数） |
| `GI_DEN_ATROUS_R` / `_FAST` | 2 / 1 | atrous 单轮核半径：`(2R+1)²-1` 个 tap（「降噪质量」中/高 档 5×5 的 24 tap / 关/低 档 3×3 的 8 tap）。足迹半径 = `R × 16` 个 GI 像素。**由 Rust 按菜单档位写进降噪的小配置 buffer**（`@group(0) @binding(20)`，降噪 pass 只有 group(0)、够不到 `gi_u`） |
| `GI_DEN_CLAMP_K` | 2.5 | AABB 钳制系数（本帧输入 = NRD pre-blur 的离群抑制；历史色 = anti-ghosting） |
| `GI_DEN_N_DOT` | 0.7 | **着色法线相似度的容差**（`dot(n_a,n_b) ≥ 本值` 才算同一个面）。`voxel_normal` 是梯度法线，取值离散（1 / 0.816 / 0.707 / 0.577 / 0 / …）⇒ 0.7 = 同轴与「朝同轴倾斜 45°」（凸棱）都算同一个面，凸角（0.577）与垂直面（0）拒掉；0.9 等于只认轴对齐 |
| `GI_DEN_LUMA_FLOOR_REL` | 0.05 | φ 的相对下限（× 邻域平均亮度）：没有它，小面上 σ = 0 ⇒ φ ≈ 0 ⇒ 大核空转 ⇒ 运动时远处大面起雪花 |

**GI 链上的四处结构性事实**（细节都在 `gi/*.wesl` 与 `main.wesl` 的注释里）：

1. **① 逐面合并**：同一个**可见体素面**的全部 texel，GI 射线起点/法线/外推量完全相同（`gi_ss_main`
   的 `org`），只有种子不同 ⇒ 它们的候选是**同一个积分的 i.i.d. 样本**，面内平均是纯降噪（方差 ÷ texel 数）。
   链路：`gi_ss_main` 末尾把本 texel 的估计**定点累加**进逐面表 → 新 pass `gi_face_flatten` 把该面均值
   写回 `gi_out`（只改写"被本面认领且两键逐位相等"的 texel）→ 降噪链拿到的是"整个面一个值"的输入。
2. **② 二次顶点缓存跨帧持久**：`gi_sec_slots` 槽里的键带 **epoch 掩码**（`gi_sec_key_masked`），
   epoch = 几何修订号 + 太阳方向/色×强度 + 天光色 + 「太阳反弹」开关 + **光照场窗口原点**
   （= `gi_secondary_shade` 的全部输入，`GiGpu::epoch` 逐项比对后自增）⇒ 任一项变了整张表自失效，
   **不再每帧 `clear_buffer`**（2K + 1/4 档约省 42MB/帧的写带宽），静态场景下那个面的着色
   （含它那条太阳 NEE）只在 epoch 变化时重算。
3. **④ 二次弹射（菜单「光照/二次弹射」）**：二次顶点上再发一条余弦射线，按 `kD·albedo·L2` 把它的入射
   辐亮度叠回二次顶点的出射辐亮度 —— 即"多一跳"的间接光（互反射 / 彩色渗色）。「稀疏」档 = 每个候选
   **独立地**以 `1/4` 概率带上、带上时乘 `4` ⇒ 期望无偏、额外射线 = 候选数的 `1/4`；
   第三条顶点的着色走上面那张**同一个**缓存（含它的太阳 NEE）⇒ 不随候选数增长。
4. **⑤ 时域接受 = 判据 + 搜索**：判据（面键 + 着色法线相似，即 5c）**一个字不改**；重投影落点不
   接受时，按 `GI_HIST_SEARCH_TAPS` 在它的 3×3 邻域里继续找（`gi/common.wesl::gi_hist_search_offset`，
   reservoir 侧与降噪侧**共用同一张偏移表**；正中恒排第一 ⇒ 静止相机下与"只认落点"逐位一致）。
   **为什么必须有**：不接受 ⇒ `M = 1`、降噪侧历史权重为 `0` ⇒ 本帧 2~4 条射线的原始估计直接上屏；
   而"落点那个 texel 采到了相邻台阶"在斜面上是常态（一个体素面常常只占一个 texel）。

两条硬约束：

1. **GI 的入射辐亮度只有两项**：二次顶点命中面的太阳项 + 射线逃逸到天空（外加自发光）。
   命中面的入射项与直射项都必须带 1/π（与 `dda_main` 的口径一致，能量才自洽）。
   删掉太阳项 ⇒ 任何看不到天空的表面（草根、过道内墙）只能自洽维持全黑。
2. **太阳阴影射线的起点必须跟着色侧同一套外推**（半个 voxel + `SHADOW_SURFACE_EPS`）：只退 1 个 eps 会从
   **命中体素内部**出发，第一步就自命中 ⇒ 该项恒为 0，白耗一根射线。

> **跨语言常量的权威在 WESL 源码**：`gate-render/src/wesl_consts.rs` 启动时解析 WESL 包 `gi/`
> （`screen.wesl` 的 `GI_RES_WORDS` + `consts.wesl` 的 buffer 布局与轮数），Rust 侧不再各留副本
> （不一致即启动 fail-fast）。改 reservoir 布局 / atrous 轮数请改 `.wesl`。

> **帧号是精确 u32**：`GiUniform.seq.x`（`vec4<u32>`）承载自增帧号，整数帧逻辑（RNG 种子混入、
> 像素 hash）只用它 —— 帧号曾以 f32 存在 `params.x`，超过 2^24 后 f32 不能表示连续整数，
> 种子会偶发重复。`params.x` 保留为恒 0 占位。

---

## 5. 不变量与踩坑（回归必看）

以下每一条都已固化在对应代码注释里，改动相关模块时先核对：

1. **存储 buffer 扩容必须带内容**：`ensure_with_copy` 做 GPU-GPU 前缀拷贝再写尾部；直接重建会把
   chunk 索引表清零 → 全屏空（经典回归）。
2. **上传别逐帧硬扛 backlog**：`poll_pending` 在 backlog > 3× 预算时一次性刷新，否则每次 Prepare 被长阻塞。
3. **树编辑走整块路径**：笔触用 `fill_brick`（O(深度)）而不是逐体素 `set_voxel`（31³ 笔触 = 近 3 万次树下降
   + 沿途 `try_merge`）。
4. **空间复用的结果绝不回写时域历史**（`gi/screen.wesl` 文件头 ②）：空间复用读的是**上一帧**邻居的
   reservoir；把「新鲜 + 时域 + 空间」一起存回去，下一帧的历史里就混进了邻居的历史 ⇒ 时域链被横向
   污染（相关性、拖影、移动物体「前沿带」）。正确写法是：存回 reservoir 的只有「新鲜 + 时域」，
   空间合并只作用在本帧的着色估计上。
5. **DDA cell 步进必须整数增量维护**：不得用 `floor(origin + dir·t)` 重算（t 恰在边界时 floor 会取到
   穿越前/后的胞 → 对角胞漏检，与 brute-force 不等价）。
5b. **降噪导引的「有效」判据是「键非零」，不许另设有效位**：`GI_DEN_G_KEY1` 里 face 占 **13..15 三个
   bit**，`face = 4 (-z)` 就是 `0x8000`——把有效位放 bit15 会和面号撞车，读侧再抹掉它就会把 `-z`
   读成 `-x`、`+z` 读成 `+x`，于是在体素凸角处把两个**互相垂直**的面当成同一个面（棱边渗色）。
5c. **「同一个面」= 面键（平面归属）+ 着色法线相似（相似度），两条各司其职**：平面归属必须用
   **DDA 入射面**（键，精确、与缩放无关，能拒掉薄板另一侧 / 隔一格的平行面）；相似度必须用
   **着色法线**（`voxel_normal` 的梯度法线 —— 凸棱上它是 45° 斜向、画面在那一段是连续的）。
   两件事不能混用：**拿斜的着色法线去算「平面距离」是自相矛盾的**（它描述的不是那个轴对齐的面），
   反过来拿面法线去判「看起来像不像同一个面」就会把凸棱那一段切碎。
5d. **隐式法线（`voxel_normal`）只算一次**：`gi_main` 算完写进导引 / reservoir / 历史，下游一律读；
   `dda_main` 在 `gi_div = 1` 时直接读导引里那一条（同一像素同一条射线 ⇒ 逐位相同），
   `gi_div > 1` 时导引那格是另一条射线，只能自己算。**不要在任何降噪 pass 里重算它**
   （那要探 6 个邻域体素，而且降噪 pass 的 layout 里根本没有 brickmap 绑定）。
5e. **① 只"多赚"，不改对错**：`gi_face_flatten` 只改写"槽被本面认领、两键逐位相等"的 texel；
   撞键 / 未认领的那条路一律退化成逐 texel，**只损失收益**。
5f. **② 的 epoch 必须覆盖 `gi_secondary_shade` 的全部输入**：那张表**已经不清空**了 ⇒ 漏一项就会把
   过期光照当成本帧的值。新增任何"被二次顶点读取的光源 / 几何 / 光照场量"时，先把 `GiEpochKey`（`gi.rs`）
   补上；**不要**改成哈希（逐项比对才看得出是哪一项变了）。
5g. **历史搜索只改「去哪儿找」，不改判据**：`GI_HIST_SEARCH_TAPS` 的每个候选一律仍走
   `gi_same_surface`（面键 + 着色法线，同 5c），且**正中必须排第一**（静止相机下要与"只认落点"
   逐位一致）。reservoir 侧与降噪侧共用 `gi/common.wesl::gi_hist_search_offset` —— 两边顺序不一致，
   "哪条历史被接上"就会分叉。它**不是**放宽几何条件的入口：跨面 / 跨格混样本仍然只有空间复用那条
   明写在案的路。
5h. **空间复用并进来的累计量必须有上限**（`GI_SS_REUSE_M_K`）：邻居的 `M` 上限与我们**相同**
   （`GI_SS_M_CAP_K` × 本帧候选数），而 `GI_SS_REUSE_TAPS` 个 tap 合计可达它的数倍 ⇒ 不设限时
   本帧那几条新鲜样本被完全淹没，画面被"**上一帧 + 屏幕偏移一格**"的值支配（症状：相机一动就拖影、
   暗块像钉在屏幕上）。上限按 `w_sum` / `M` **同比例**施加 ⇒ 邻居的估计值不变，只压它的权重。
   另注：`hist_ok = false` 会**同时**触发「去遮挡」候选翻倍 ⇒ 任何"把历史判得更严"的改动（哪怕是
   临时诊断）都会成倍放大射线开销，别把它当免费手段。
5i. **「拖影 ↔ 静态闪烁」的取舍落在 `GI_DEN_M_MAX`（时域历史长度上限）上**：稳态权重 =
   (M−1)/M ⇒ **时间常数 τ = M 帧**，而 τ 就是光照 / 远处几何**渐变**的响应时间（`σ_hist` 对渐变
   不敏感，每帧变化小 ⇒ `w_var` 一路是 1，只剩这一项拦着）；静态残余闪烁 ∝ 1/√M。取 32
   （τ ≈ 0.53s、闪烁 ≈ 16.2%）对齐 NRD / RTX Remix 的量级（它们的历史长度是几帧~几十帧，
   本工程曾取 96 ⇒ τ ≈ 1.6s）。**几何**变化不吃这个窗（面键判据精确），**光照阶跃**也不吃
   （σ 突增 ⇒ `w_var` 掉下来）。
5j. **世界几何变了 ⇒ 该帧 reservoir 不复用任何历史**（门 = `gi_u.flags.z`，见 `gi/screen.wesl` 文件头 ⑥；
   信息取自 `world_rev`，覆盖一切增量体素编辑 / palette 变化 / 全量上传）。为什么要这么粗：reservoir 存的是
   `w_sum` / `M` 这条**累加量**，而验证射线只重发**被选中的那一个代表样本**的光路 —— 代表样本没变
   **不代表**累加量里别的样本还有效。症状：擦掉一块自发光体素 ⇒ 接收面上"打到它"的候选全部失效，
   但代表样本（例如一条逃向天空的方向）仍然成立 ⇒ 整条累加量照收 ⇒ 那束光只能靠记忆窗淡出
   （**相机不动时看不出任何即时变化**）。这条粗门也正是 `GI_RES_WORDS` 能从 14 缩到 8 的原因
   （逐样本校验、`gi_ray_secondary_key`、代表样本字段全部删除）。
   **降噪侧不要跟着重置**：它的 ⑤ AABB 钳制会把历史色拉回本帧邻域的范围 ⇒ 响应足够快；重置反而会让
   编辑帧露出未平滑的输入、并让 atrous 的短历史稳定化（`GI_DEN_STAB_*`）整屏生效一帧
   （症状：放置体素时 GI 闪一下）。
6. **bind group 必须从索引 0 起成前缀设置**：自动曝光两个入口的布局因此重复挂 8 份；
   半分辨率 GI 的**采样视图**必须挂 group(5)（同一 pass 内同一张纹理不能既采样又作存储写入）。
7. **blit 采样器必须 Linear**：半分辨率上采样与 FXAA 亚像素偏移都依赖它；Nearest 会让 FXAA 整体空转
   （现象 = "开了没变化"）。
8. **UI 指针捕获闸门**：相机拖拽 / 滚轮 / 编辑射线都要查 `UiPointerCaptured` + `MouseIntercepted`，
   否则操作菜单会同时驱动相机；文本输入框编辑态（`TextInputFocus`）还要挡键盘。
   两者的判定在 `gate-ui/src/capture.rs`：UI picking 开了 `UiPickingSettings::require_markers`
   （GateUiPlugin 里），要求**相机挂 `UiPickingCamera`、节点挂 `Pickable`** 才参与命中
   ⇒ 装饰节点对命中不可见，与旧 `FocusPolicy`（只有 `Block` 才拦命中）等价。
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
cargo clippy --release --workspace --all-targets -- -D warnings
cargo build --release --workspace
```

- **全部加 `--release`**：`gate-app/build.rs` 禁止 debug 构建，dev profile 下 `build` / `run` / `check` / `clippy` / `test` 一律直接失败（无逃生开关）。
- **`cargo fmt --check`**：代码风格（rustfmt.toml：Google 风，2 空格缩进，edition 2024）。
- **`cargo clippy --release --workspace --all-targets -- -D warnings`**：lint 零警告。
- **`cargo build --release --workspace`**：0 error。

> 无 CI：以上三条 + 下面的人工验收靠提交前手动跑。

**手工验收（`cargo run --release -p gate-app`）**

- 默认 `nuke.vox` 场景出画；`WASD` 飞行与右键转头流畅；`F3` 菜单显隐正常。
- 菜单逐项生效：GI 开关 / GI 分辨率档（1/1、1/2、1/4）/ **降噪质量档（关、低、中、高；与分辨率档正交）**
  （切档时 stderr 有 `GI 降噪质量 → …` 与 `GI 降噪档位 → …` 两行）/ **「光照/二次弹射」（关/稀疏/全，
  stderr `GI 二次弹射 → …`）**；
  半分辨率 / FXAA / 垂直同步；
  曝光参数（改完 stderr 有 `eye adapt 参数 → GPU` 一行）。
- 左键放置、右键单击擦除 → stderr 出现 `EDIT[...]` 与 `UPLOAD[incremental]`（增量部分写：`bytes` 远小于全量上传）。
- 「游戏/世界/重载世界」换模型后画面整块刷新（光照场跟随重建，无残留旧几何）。
- 长时间运行无 1 秒以上的增量上传长耗时（`UPLOAD[incremental]` 的 elapsed 应在毫秒级）。

**性能剖析（可选）**

```powershell
cargo run --release -p gate-app --features profile   # 启动后用 Tracy GUI 连接进程
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
- **不写安装目录**：日志 → `<安装根>/logs/latest.log`；持久化配置（菜单控件值/窗口 + 相机姿态）→
  `<安装根>/data/config.toml`（菜单结构与缺省值每次从 `assets/ui/debug_menu.toml` 读）。
- **注意**：`assets/vox/nuke.vox` 被 gitignore，脚本只在缺失时 WARN，不会阻断打包——发布前自备该文件。

## 9. 其他

### 行数统计
```cloc --include-lang=Rust,WGSL --force-lang=WGSL,wesl ./assets/shaders ./gate-app ./gate-render ./gate-ui ./gate-voxel```