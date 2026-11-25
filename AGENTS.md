# AGENTS.md

## 构建 / 运行

- **本仓库禁止 debug 构建**：`gate-*/build.rs` 会在 `PROFILE=debug` 时 panic。使用 `build-fast` / `balance` / `run-fast`；不再维护自定义 `release` profile。
- 默认 Build Task（`Ctrl+Shift+B`）是 `build-gate-app-fast`：
  ```bash
  cargo build --profile build-fast -p gate-app
  ```
  产物 `target/build-fast/gate-app.exe`。
- 三种构建 profile：`build-fast` 是默认与性能测试档，`balance` 暂保留，`run-fast` 仅用于最终打包：

  **`build-fast`：编译优先，默认档（F5 `gate-app (build-fast)` → `build-gate-app-fast`）**
  ```bash
  cargo build --profile build-fast -p gate-app
  ```
  `lto=false` + `codegen-units=16` + incremental；产物 `target/build-fast/gate-app.exe`。改 `gate-app` 约 5 秒，改 `gate-render` 约 6 秒。正确性验证、性能优化、FPS / 帧时间测量、基准与性能回归都用这个。

  **`balance`：编译 / 运行平衡（F5 `gate-app (balance)` → `build-gate-app-balance`）**
  ```bash
  cargo build --profile balance -p gate-app
  ```
  Thin LTO + `codegen-units=4`；产物 `target/balance/gate-app.exe`。当前保留备用，不用于日常或性能测试。

  **`run-fast`：运行优先（F5 `gate-app (run-fast)` → `build-gate-app-run-fast`）**
  ```bash
  cargo build --profile run-fast -p gate-app
  ```
  Fat LTO + `codegen-units=1`；产物 `target/run-fast/gate-app.exe`。构建可达十几分钟，只用于最终打包。

- **选择规则**：渲染正确性、UI、逻辑、普通功能验证，以及性能优化、FPS / 帧时间测量、基准与性能回归，一律用 `build-fast`；`balance` 暂保留；`run-fast` 只用于最终打包。实测三种构建对帧率几乎没有差异，不要把构建 profile 当作帧率优化手段。
- PGO 尚未接入；后续如需尝试，作为一次性发布流程处理，不纳入日常构建。
- 三套产物目录互不复用：`target/build-fast`、`target/balance`、`target/run-fast`；不要混用二进制。
- 迭代时不要直接 `cargo run --release`；用 `cargo run --profile build-fast -p gate-app`，或构建后运行 `target/build-fast/gate-app.exe`。
- 发布打包：`bash package.sh` 默认使用 `run-fast`（Fat LTO）；其他 profile 用 `GATE_PROFILE=balance bash package.sh`。
- 快速验证：`cargo check --profile build-fast -p gate-app`；格式：`cargo fmt --all -- --check`。

## 验证

- 项目默认不新增测试；不要新增 `tests/` 文件或 `#[test]`。正确性通过 `cargo check/build` 和实际运行验证。
- WESL 在 `gate-render/build.rs` 做**编译期校验**：编译 WESL、用 naga 校验 WGSL，并检查所有 compute 入口存在。改 `.wesl` 后构建会触发该检查。
- `cargo clippy --profile build-fast --workspace --all-targets --all-features -- -D warnings` 必须保持全清；新增代码不得引入 lint。
- 禁止用 `#[allow(...)]` / `#[expect(...)]` 抑制 lint；必须修掉根因。参数过多用 `#[derive(SystemParam)]` 分组，类型复杂用类型别名。

## 工作区结构

- `gate-app`：唯一二进制入口 `src/main.rs`。场景构建 `scene.rs`；菜单/配置 `debug_menu.rs`、`config.rs`；体素编辑 `edit.rs`；MC 地图 `mc/`；无限方块/流式 `infinite_cubes.rs`。
- `gate-render`：渲染核心。`GateRenderPlugin` 在 `lib.rs`；DDA、brickmap、反射与 GI 的 GPU 资源主要在 `src/brickmap/`，其中 `brickmap/dda.rs` 是手写 bind group/pipeline 布局的大文件；GI 设置与资源在 `src/gi.rs`；运行时 WESL 编译在 `shader.rs`。
- `gate-voxel`：`VolumeGrid`、调色板、chunk tree、编辑 AABB、`Volumes`。
- `gate-ui`：Debug Menu、控件、主题；菜单树内容在 `assets/ui/debug_menu.toml`。
- `gate-physics`：物理。
- WESL 源码全部在 `assets/shaders/voxel_raytrace/`，不在任何 crate 的 `src/` 内。

## WESL / 渲染约定

- WESL 在**运行时**编译（`gate-render/src/shader.rs`），同时在 `gate-render/build.rs` 的构建阶段校验 naga WGSL 合法性和入口列表。改 `.wesl` 会触发 build script 并重编 `gate-render`/`gate-app`（`build-fast` 下约 10 秒），比 Rust 逻辑改动轻；重启程序会加载新 shader。**不要**保留「管线就绪前渲染预览 / 逐 pass 上线」的过渡路径：`dda_main`/`dda_preview` 已删除，`DdaReady` 门控保证全部用到的管线（按当前设置）就绪前不渲染、不 blit，首帧即完整画面。
- 主模块（`main.wesl`）全部 compute 管线经 `gate-render/src/pipe_cache.rs` 直连 wgpu 创建，挂持久化 `wgpu::PipelineCache`（blob 在 `data/pipeline-cache.bin`，`DdaReady` 后回写）；bevy `PipelineCache` 是设备不支持 `PIPELINE_CACHE` 时的回退路径，两条路径都要能跑。启动变慢先看 `PIPE {入口} 编译` 日志定位是哪个入口的驱动编译涨了。
- `gi_accumulate` 只允许出现在自带采样上下文的入口（`gi_gather` / `gi_merge_drain`）。把它内联进其他逐 texel 入口（`dda_refl`、`gi_gather` 等）会把该入口的驱动编译时间推高一个数量级（NVIDIA 驱动对「采样机器 × 逐 texel 循环」的组合有编译病理）。镜面 GI 补样（`refl_gi_merge`）因此走 `gi_merge_req` 请求队列，由 `gi_merge_drain` 消费，采样语义与时机不变。
- 项目只保留软件 DDA；`RT_RAY_QUERY`、TLAS/BLAS 和 `@if(ray_query)` 分支已移除，不要重新引入。
- `.wesl` 里的部分常量是 Rust 的资源布局真源：`gate-render/src/wesl_consts.rs` 会从源码解析 `const NAME: u32 = <字面量>u;`，缺失会 panic。不要把这些常量只改在 Rust 侧，也不要用非字面量表达式。
- 新增 WESL compute 入口时，同步 `gate-render/build.rs` 的 `ENTRY_POINTS` 列表。
- bind group layout、binding 编号和 pipeline group 向量是**手写**的（主要在 `brickmap/dda.rs`）。naga 校验只覆盖 shader 本身，不校验 wgpu bind group；改 binding 后必须启动程序验证。
- 反射请求二进制布局有历史坑：两种请求都把 `kind|weight` 放在 `.w`；packed chain 的 `pal` 在 `.z`，第二行是 `(pbase, face_id, 0, px)`。改布局必须同时改读写两端。
- 「镜面质量」档位固定为 `0=关, 1=直射, 2=天空, 3=阴影, 4=GI`。改档位时同步 `ReflectionSettings::TIERS`、shader 的 `REFL_TIER_GI`、`assets/ui/debug_menu.toml` 和 `assets/locales/zh-CN.toml`。
- **逐体素着色 = 一个体素六个面同一个平色**（不是逐面平色）：材质与 GI 都量化到体素，同体素的任意可见面共用一个颜色；不要在着色里引入逐像素 / 逐面差异，否则画风会被打碎。
- 隐式法相（`voxel_normal` / `voxel_normal_fast`）在梯度退化时按**体素自身**的邻域选轴（`degenerate_axis_normal`：空气邻居最多的轴，平票 Y > X > Z，双侧都空气取正侧）：一 voxel 厚的墙/板在 castle 这类模型里到处都是，不能回退 `+Y`（竖墙会去查「朝上」GI 面，整片发黑），也不能按命中面取法线（同一体素的不同可见面会变色，破坏「一个体素六个面同一个平色」）。GI 显示与累积（`irc_display` / `irc_accumulate`）另外对零法线兜底 `+Y`，避免 NaN 与全黑。
- GI 是**世界空间辐照度缓存**（Rust：`src/gi.rs`；WESL：`assets/shaders/voxel_raytrace/gi/`）。条目按「体素坐标（世界/物体本地） × 面轴 × 屏幕足迹分级」哈希，值是 **E/π**（余弦加权平均辐亮度）；显示端每个体素按中心取一次值、同体素各面按余弦加权混合，所以画面上不会出现屏幕空间网格或重投影拖影。
  - 采样由 1/N 分辨率的 GI 网格驱动（`gi_gather`）：静止时每 texel 每帧按缓存成熟度自适应 1–8 个样本，相机移动时降回 1 个以保帧率。缓存单元的屏幕尺寸必须 ≥ 一个 GI texel（`irc_addr` 取 `ceil`，足迹不小于 `div`），否则一个 texel 覆盖多个单元、只有其中一个能拿到样本。`dda_hit` 每帧只由抖动像素写入，块内表面按 16 帧循环轮流被采样。`irc_claim` 在桶满时按最旧 stamp 原位淘汰（最近 4 帧内读写的槽位受保护），不能直接丢弃新样本，否则移动后新表面要等全局年龄回收才开始收敛。`gi_resolve` 把本帧样本折进条目、按下一条的规则与父级混合，`gi_light_sweep` 维护光源瓦片。缓存条目同时写本层与三级父层（`IRC_CHAIN`），新露出的表面立刻能取到平滑值。`gi/restir.wesl` 的**时域与空域复用都必须用精确体素键**（`k0/k1`，即同对象 + 同体素 + 同命中面）：reservoir 里的 `vis` 是在采样位置测的，只有同键才成立。按「同对象 + 法线夹角 + 平面距离」放宽会让 1 voxel 厚墙两侧共用退化法线、相邻体素互判同面，室内窗光/自发光可见性被搬到外立面，夜间表现为成片暖色光斑且长时间不散。相机移动时同一点的重投影仍落在同一体素，复用不会整段失效；复用率下降应重算 `vis`，不要放宽键。空域 tap 目前关闭（`RES_SPATIAL_TAPS = 0`）：把邻居的选中样本按权重 1 再计一次的期望是 `Σc²/Σc` 而非 `Σc/m`，夜间会把亮样本放大成过亮与斑块；直接合并 `sum/m` 又会让 `m` 每帧几何增长并很快触顶，早期噪声被冻结。恢复空域复用前先实现有界权重、有界历史的合并。
  - 显示值 = 父级插值 + `w`·（本格均值 − 父级插值）（`irc_resolve_slot`）。`w` 取两条证据的较大者：本格与父级之差的 `z²`（`IRC_SIG_LO`..`IRC_SIG_HI` 平滑放行），以及父级记下的子格离散度 `het/(het+se²)`（兄弟格把各自的 `d²`、`se²` 推给父级，平均 `d²` 超出 `IRC_HET_K`·平均 `se²` 的部分才算真实离散，用来放行棱和接触阴影）。不显著的格子显示父级，画面噪声由样本多 4×/16×/64× 的上层决定。**不要**改回按样本数放行（`n/(n+N)` 或单格 `d²/(d²+c·se²)`）：相机靠近时整屏换到更细一层，新格只有几十到几百个样本，每样本噪声会直接显示成逐体素椒盐，停下后要几秒才散。每样本相对方差沿层级池化（样本少时用父级的估计，`IRC_RV_K`），且「当前每样本方差」（给子格当先验）与「均值内样本的平均方差」（算 `se²`）分开记：相机一停每样本方差立刻变小，但均值仍由移动期的样本构成。
  - 条目只按全局年龄（`EVICT_FRAMES`）回收；所在桶空位不足 2 个时才提前回收 `IRC_EVICT_PRESSURE` 帧未触达的条目。**不要**按相机是否移动缩短回收窗口：离屏 1 s 的表面被清掉后，转回来整面从零开始。
  - 改缓存布局 / 常量必须同步三处：`gi/consts.wesl`（权威字面量）、`src/wesl_consts.rs`（解析）、`src/gi.rs` 的 `gi_bg5_layout()` 与缓冲分配。
  - 屏幕空间补值只允许出现在「缓存完全没有条目」的兜底上，且必须落在同一套世界空间核上（按同体素/同面判定），否则棱上会明暗交替。
- 自发光是**真实光源**：`gi/light.wesl` 维护世界空间光源瓦片（发光面按 8×8 面片瓦片化）与按功率的层级树；`lt_nee` 做 NEE，命中按平衡启发式与 BRDF 采样合并，同一块发光面不会被算两次。可见自发光面在着色阶段就注册进光源瓦片，进入新区域时 NEE 从首帧起生效（否则要等 GI 射线偶然命中才注册，暗处会先出现严重 firefly）。**不要**引入「同屏最多 N 个光源」的列表或上限；容量只受 `LT_SLOTS` 约束（超出部分退化为 BRDF 命中，仍然可见）。`res_add_pick` 里落空的抽样（背向、超距、发光面已失效）是贡献为 0 的合法样本，必须计入 `m`，`sum/m` 才无偏；漏计会把直接光按 `1/命中率` 放大，每帧抽样数越多越明显。
- 镜面同样逐体素着色（一个体素一个反射值）：反射射线从体素中心出发，同一体素的像素共用同一方向与权重；不要引入逐像素 / 逐面差异或屏幕空间共享。镜面 GI 直接读世界缓存 `refl_hit_gi`（未成熟时按命中点补一条样本），不要再引入屏幕空间兜底（会把屏幕边缘的 GI 串进镜中）。

## 运行期数据 / 资源

- `assets/` 只读、随仓库分发；`data/`、`logs/`、`*.vox`、`*.log` 被 `.gitignore` 排除。
- `data/config.toml` 保存菜单、相机和窗口状态；要重置用户配置就删除它。不要提交 `data/`。
- `data/pipeline-cache.bin` 是驱动管线缓存 blob（机器/驱动绑定，wgpu 自带失效校验）；删除它下次启动就是全量冷编译。
- 资源根目录由 `gate-render/src/paths.rs` 决定：exe 同目录有 `assets/` 就用 exe 目录，否则回退到仓库根。直接运行 `target/.../gate-app.exe` 也能找到仓库 `assets/`。
- 默认场景名是 `nuke`，依赖 `assets/vox/nuke.vox`；该文件被 gitignore，新克隆可能没有。可在 `data/config.toml` 改 `game/world/model`，或把 `STARTUP_DEMO_SCENE` 改成 `true`。
- MC 地图默认路径硬编码为 `C:\game\Greenfield v0.5.4\...`；用 `GATE_MC_MAP` / `GATE_MC_ASSETS` 覆盖。`data/lod/mc_map.lod` 是生成缓存。
- 常用环境变量：`GATE_LOG`（日志过滤）、`GATE_BENCH`（`orbit` / `fly` / `roam`；`roam` 以存档机位为基准按系统时钟每 32 s 循环「前冲往返 → 停 → 退回 → 快速转头 → 停」，用来量移动噪声与停下后的收敛，配置存 `data/config.roam.toml`）、`GATE_PHYS`、`GATE_PHYS_THREADS`、`GATE_GI`、`GATE_EDIT_SELFTEST`、`GATE_TRACY`（`0` = 不建 Tracy zone，仍保留 GPU 逐 pass 均值日志）、`GATE_GI_PROBE`（每 2 s 打一条 `GISTAT`）。

## 日志规范

- 日志是**最后手段**：能用返回值、类型、断言或测试表达的信息，不要打日志。
- 只写关键对象、符号和值；用 `→` 表动作、`!` 表失败、`[]` 表标签、`vx`/`ms` 表单位，不写完整句子。
  - 例：`info!("size → 8")`、`info!("EDIT[place] 27vx @(12,4,9) 3.1ms")`
- 同一事件只在一处记录；启动、初始化、编译期的一次性提示用 `*_once!`。
- 级别按**实际触发频率**决定：
  - `info!`：离散用户动作、失败/降级、启动里程碑。
  - `debug!`：逐帧/周期信息、诊断细节、成功路径内部步骤。
  - 默认 `info` 过滤下若一分钟超过一行，降到 `debug!`。
- 失败和降级必须保留，并写清触发输入、撞到的上限、回退成了什么；已登记的验收日志不得降级。
- **不要改**：`tracing` / Tracy 的 zone 名与 pass 名、`panic!` / `expect()` 文案、`assets/locales/*.toml` 的 UI 文案。
- 新增日志前先找同类，能合并就不新开。`INFO` 只服务当前这次运行排查；运行轨迹用 `debug!`，性能用 `--features profile` 下的 Tracy。

## 注释 / 输出规范

- 默认不写注释；只有用户明确要求，或代码确实需要解释约束/原因/不变量时才写。
- 注释必须客观、简洁、面向未来维护者；不写实现流水账。
- 禁止写：踩坑记录、调试历史、失败尝试、情绪化表达；第一人称（我、我们、你）；对话残留（用户说、模型说、之前试过、大模型提示）。
- 注释中禁止出现这些词：踩坑、坑、血泪、教训、千万别、不能改、否则会、之前、曾经、试过、报错、注意这里。
- 实现/修复类任务的最终输出只给代码或最简改动说明；不要输出无关教程、长篇 Markdown 或对话残留。

## UI 文案 / i18n

- 中文文案表在 `assets/locales/zh-CN.toml`，由 rust-i18n 在**编译期**生成进二进制。每语言一个文件；加语言 = 加文件 + 调 `rust_i18n::set_locale`。表头 `_version = 1` 是布局版本。
- key 采用「模块.控件」语义路径，调用点写 `t!("模块.控件")`；**key 必须带引号**，否则 TOML 会把裸点号解析成嵌套表。
- 菜单 `label` 最多约 **4 个汉字或 8 个字母**；超限换词。
- 占位符 `%{name}` 原样替换；定长对齐的数字由调用点先 `format!` 好再传入。
- 只收**可见 UI 文案**；日志、技术标识符（LOD / DDA）、单位（v/s、vx）、材质预设名保持英文。
- `*.tip` 是 Markdown（由 gate-ui tooltip 渲染）：多行/列表/引用用 `"""…"""`，续行顶格（缩进 4 空格会变成代码块）；行内 `code` 标标识符与数字，**粗体**标关键结论。只写让人明白的最低限度，不抄实测数字、不写历史对比、不复述调用点注释。
- DebugMenu 的文案 key 在本表 `menu.*` 段；菜单树 `assets/ui/debug_menu.toml` 存 key，由 `gate-ui::UiTranslator` 解析，切语言后整棵树重解析（`sync_ui_locale`）。
- 改本表需**重新编译**（proc macro 编译期内联）；`gate-app/build.rs` 已登记 `rerun-if-changed`。
- 新增/修改菜单项时同步：`assets/ui/debug_menu.toml`、`gate-app/src/debug_menu.rs` 回调、locale 文案。

## 工具链

- Rust stable 1.98（`rust-toolchain.toml`）、edition 2024、Bevy `=0.20.0-rc.1`、WESL 0.5。不要套用更新版本 Bevy/WGPU 的 API 假设。
- rustfmt 配置：2 空格缩进、`use_small_heuristics = "Max"`。

## 优化方法

- 针对硬件的优化：例如我们之前将大shader pass拆分为小pass，以及各种缓存优化（cache miss）
- 空间换时间：典型的有各种LUT查表，我们的DDA算法中就用到了
- 并行计算：例如我们的物理引擎中就用到了并行计算
- 算法优化：利用数学等价或近似简化复杂运算