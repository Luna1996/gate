# 决策记录（ADR）

> 每条 ADR：背景 → 决策 → 后果。状态只有两种：**已接受** / **已否决**（否决项保留理由，防止反复）。
> 与 TODO.md「已定关键决策」表对应，此处存放工程细节与版本对齐记录。

## 索引

| ADR | 标题 | 状态 | 日期 |
|---|---|---|---|
| [ADR-0001](#adr-0001bevy-版本锁定p02) | Bevy 版本锁定 | 已接受 | 2026-08-29 |
| [ADR-0002](#adr-0002bevy-019-渲染挂载模式与开发环境基线p03-spike-结论) | Bevy 0.19 渲染挂载模式与开发环境基线 | 已接受 | 2026-08-29 |
| [ADR-0003](#adr-0003viewtarget-直写的挂载点与格式msaa-契约p03-阶段-b) | ViewTarget 直写的挂载点与格式/MSAA 契约 | 已接受 | 2026-08-29 |
| [ADR-0004](#adr-0004决策存放原则与工程基线p05p06) | 决策存放原则与工程基线（日志/CI） | 已接受 | 2026-08-29 |
| [ADR-0005](#adr-0005parley-090-本地补丁cjk-文本分段vendorparley) | parley 0.9.0 本地补丁——CJK 文本分段 | 已接受 | 2026-08-30 |

---

## ADR-0001：Bevy 版本锁定（P0.2）

**状态**：已接受
**日期**：2026-08-29

### 背景

Bevy 约每 3 个月发布一次破坏性变更版本，render graph API（自定义 render node 挂载点，本项目最高优先风险项 P0.3 依赖它）是重灾区。周期内追新会让 P0.3 的 spike 反复报废。

### 决策

- **Bevy `=0.19.1`**（workspace `Cargo.toml` 精确锁定，crates.io 2026-08-13 发布，当前最新 stable）
- 工具链 `channel = "stable"`（`rust-toolchain.toml`），当前 **rustc 1.98.0**（2026-08-18），Bevy 0.19 使用 edition 2024，要求 Rust ≥1.85
- 周期内**不追新**：Bevy 0.20 发布后不升级，除非遇到无法绕过的 bug 或安全漏洞
- 需要升级时的流程：读官方 migration guide → 评估 render graph API 变更面 → 单独开 spike 分支验证 P0.3 链路 → 通过后才合入主分支

### 版本对齐记录（Bevy 0.19.1 实际携带）

| 组件 | 版本 | 说明 |
|---|---|---|
| bevy | 0.19.1 | 精确锁定 |
| wgpu | 29.0.4 | 自定义 compute 管线的底层 API |
| winit | 0.30.13 | 窗口/输入 |
| glam | 0.32.1 | 数学库（shader 外的 CPU 侧矩阵计算） |

### Bevy 0.19 feature 体系变更记录

0.19 重组了 feature 树（相对 0.18），以下是踩坑记录，升级时先对照：

- `bevy_input` **不再是** bevy 顶层 feature（错误信息会建议 `bevy_winit`，误导）；键盘/鼠标输入由 `keyboard` / `mouse` 子 feature 提供，且 `bevy_winit` 已传递依赖
- 默认 feature = `2d` + `3d` + `audio` + `ui` 四大组的组合，本项目用 `default-features = false` 手工精选最小集
- `scene` 组（BSN 新场景系统）暂不启用——项目不用 Bevy 场景序列化
- 启用集见根 `Cargo.toml`：`bevy_winit` / `bevy_render` / `bevy_core_pipeline` / `bevy_post_process` / `bevy_asset` / `bevy_log` / `bevy_state` / `multi_threaded` / `std` / `default_font` / `reflect_auto_register` / `async_executor`
- P3 评估复用后处理链时可能需补 `tonemapping_luts`；P7 UI 阶段补 `bevy_ui` + `ui_api` 组；P10 .vox 导入需 `png`（`bevy_image` 已传递引入）

### 后果

- ✅ P0.3 spike 目标环境稳定，render graph API 学习成果整个周期有效
- ✅ Cargo.lock 可复现构建，CI 无漂移
- ⚠️ 依赖的第三方 crate（如未来加 egui）必须选兼容 bevy 0.19 的版本线
- ⚠️ 版本锁定 ≠ 冻结：子依赖补丁版可经 `cargo update` 浮动（当前策略：仅显式执行，不自动）

---

## ADR-0002：Bevy 0.19 渲染挂载模式与开发环境基线（P0.3 spike 结论）

**状态**：已接受
**日期**：2026-08-29

### 背景

P0.3 spike 目标是打通"全屏 compute → 上屏"。原计划按 render graph node 模式实现，但 0.19 API 已变（#22144 "Replace RenderGraph with systems"）。

### 决策一：渲染挂载模式

Bevy 0.19 中**没有自定义 render node 概念**，正确姿势：

- `RenderGraph` 是一个 **schedule**（`bevy::render::renderer::RenderGraph`，注意顶层 re-export 是 private 的，必须从 `renderer` 模块导入）
- GPU 命令录制 = 往 `RenderGraph` schedule 加普通 system，`RenderContext` 可直接作为系统参数注入
- 排序锚点：`core_pipeline::schedule::camera_driver`，自定义 system 用 `.before(camera_driver)` 保证先于相机 pass
- 管线创建：`RenderStartup` schedule + `pipeline_cache.queue_compute_pipeline(...)`，shader 走 asset 加载（异步），`get_compute_pipeline` 返回 None 时跳过该帧
- bind group：`RenderSystems::PrepareBindGroups` set 内创建（`BindGroupLayoutDescriptor` 直接存 Resource，`pipeline_cache.get_bind_group_layout`）
- ViewTarget（阶段 B 用）：`get_color_attachment()` / `post_process_write()` / `main_texture_format()`；官方参考示例 `examples/shader_advanced/custom_post_processing.rs`（0.19 中已不在 shader/ 目录）

### 决策二：GPU 后端——Vulkan 为默认（附 wgpu 已知 bug 记录），DX12 为后备

- **EOS Overlay 注册表残留**（HKCU\...\Vulkan\ImplicitLayers 指向不存在的 JSON）已清理，loader 报错消失
- **wgpu Vulkan present VUID 错误**（VUID-01430 layout UNDEFINED + VUID-01286 semaphore signaled，仅启动前几帧，运行期干净）：**wgpu 上游已知 bug**，见 gfx-rs/wgpu#9213 / #6748；修复 PR #9361 尚未合并，wgpu 29.0.4（Bevy 0.19.1 锁定）必然携带。与 Steam/EOS overlay layer 无关（已做单变量二分验证）
- 处置：**接受 + 跟踪上游**。已实验将 wgpu patch 到修复 PR#9361（third_party/wgpu，commit 494a4e1 + 版本号提至 29.0.4 + `[workspace] exclude = ["third_party"]` 解决嵌套 workspace 吸收问题，cargo#12154），结果 **bevy_render 0.19.1 编译失败**：PR 为 breaking change（`SurfaceTexture::present` 移至 `Queue::present`、`NoopBackendOptions`/`RequestAdapterOptions` 加新字段）。fork bevy_render 修 3 处调用点的成本远超收益，撤销 patch。#9361 合并进 stable 后随 Bevy 升级自然消失
- DX12 后备：无 Vulkan validation 检查路径，无此错误（`WGPU_BACKEND=dx12`）
- 已知小尾巴：Vulkan 模式窗口关闭后退出阶段报 `STATUS_STACK_BUFFER_OVERRUN`（teardown 顺序，不影响运行，跟踪）

### 决策三：dev profile 优化（必加）

workspace 必须包含 Bevy 官方推荐配置，否则 debug 构建慢 10-50 倍，症状为窗口迟迟不开、主线程卡到 Windows 判定无响应：

```toml
[profile.dev]
opt-level = 1
[profile.dev.package."*"]
opt-level = 3
```

### 踩坑记录

1. **WGSL `target` 是保留字**——storage texture 变量命名避开（改用 `out_tex`）；shader 编译错误在 `pipeline_cache` 的 ERROR 日志里，not asset server
2. **ShaderType 对齐**：Rust 侧 `Vec2`（8B）与 WGSL `vec4<f32>`（16B）不匹配 → "min_binding_size, which is 8" 验证错误。uniform 结构体字段必须与 WGSL 声明大小对齐（Vec4 对 vec4）
3. **asset 路径**：0.19 相对 crate 目录解析 asset，与 F5 的 cwd（workspaceRoot）不一致。统一用 `AssetPlugin { file_path: concat!(env!("CARGO_MANIFEST_DIR"), "/assets") }` 锚定
4. **"rustc ICE"（后经 ADR-0003 修正定性）**：codegen 阶段 panic，实为沙箱拦截 incremental 目录写入（Access denied 伪装），非编译器 bug。最终处置：`incremental = false`（见 ADR-0003 决策四）

### 后果

- ✅ 最高优先风险项已证活：compute → storage texture → 上屏全链路 OK（阶段 A：Sprite 中转；阶段 B：ViewTarget 直写，见 ADR-0003）
- ✅ 0.19 渲染 API 摸底完成，P2 全屏 DDA pass 的挂载路径明确
- ✅ Vulkan 与 DX12 双后端验证通过，默认开发环境 Vulkan
- ✅ 阶段 B（ViewTarget 直写）完成，Sprite 中转已移除——架构定型：compute → storage texture → blit → ViewTarget

---

## ADR-0003：ViewTarget 直写的挂载点与格式/MSAA 契约（P0.3 阶段 B）

**状态**：已接受
**日期**：2026-08-29

### 背景

阶段 A 用 Sprite 显示 compute 输出只是临时方案。阶段 B 目标：blit 直写 Bevy ViewTarget，Sprite 链路彻底移除。过程中三次踩坑（格式、MSAA、挂载点），逐一记录。

### 决策一：blit 挂载点——Core2d 的 PostProcess set，而非 RenderGraph schedule

- **`after(camera_driver)` 挂载无效**：`camera_driver` system 内部同步运行各相机的 Core2d/Core3d 子 schedule，**surface copy（upscaling）在相机图内部就完成了**。排在 camera_driver 之后的 system 写 ViewTarget main texture，输出永远上不了屏（症状：灰屏但零 validation 错误，日志显示 blit 正常 running）
- 正确挂法：`render_app.add_systems(Core2d, blit_view.in_set(Core2dSystems::PostProcess))`——Core2d schedule 的 set 链为 `Prepass → MainPass → EarlyPostProcess → PostProcess`，upscaling 排在 PostProcess set 之后 → blit 必然在 clear 之后、上屏 copy 之前
- 导入路径：`Core2d` 与 `Core2dSystems` 都在 `bevy::core_pipeline::schedule`（`core_2d` 模块对 Core2d 只是私有 re-import）
- **规则**：写 ViewTarget 的 pass 一律挂 Core2d PostProcess set；RenderGraph schedule 只放 camera_driver 之前的工作（compute 预计算等）

### 决策二：格式契约——ViewTarget 是 Rgba8UnormSrgb

- Bevy ViewTarget 主纹理默认格式 **`Rgba8UnormSrgb`**（Camera2d 非 HDR），自定义 render pipeline 的 `ColorTargetState.format` 必须匹配，否则 validation fail → bevy error handler 主动退出（症状：启动即崩 0xc0000409）
- 内容语义：sRGB target 上硬件做 linear→sRGB 编码。源纹理（storage texture）存的是"设计显示值"时，fragment 里先做 **sRGB→linear** 转换再输出，两次变换抵消 → 屏幕像素字节 = 源纹理字节（调试读回直觉一致）
- P2 管线注意：将来 DDA 输出线性光照值时，硬件编码恰好正确（gamma correct），无需手动转换——两阶段语义不同，勿混用

### 决策三：MSAA——0.19 中是 per-view Component

- Bevy 0.19 的 `Msaa` 是 **Component**（挂在相机实体上），不再是全局 Resource（`insert_resource(Msaa::Off)` 编译都过不了）
- 默认 4x → ViewTarget color attachment 是 4 采样纹理，自定义管线 `multisample.count = 1` 与之冲突 → validation fail 崩溃
- 本项目管线不用 MSAA：相机 spawn 时挂 `Msaa::Off`（`(Camera2d, Msaa::Off)`）

### 决策四：沙箱假 ICE——关闭 incremental

- 症状：rustc codegen 阶段 panic（表现同编译器 ICE），且随代码修改**反复**出现
- 真相：Trae 沙箱拦截 `target/debug/incremental` 目录写入（`pre-lto-bitcode ... Access is denied (os error 5)`），rustc 写文件失败 panic，伪装成 ICE。此前"cargo clean 恢复"只是碰巧
- 处置：`[profile.dev] incremental = false` + 启动时清理残留 incremental 目录。项目规模小，全量编译代价可忽略
- 教训：**"ICE" 先看 panic 消息尾部有无 IO error / Access denied**，别急着怪编译器

### 调试方法论沉淀

- CLIXML 包装（PowerShell 2>&1）会吞日志尾部：跑 Bevy 应用用 `| Out-File run.log -Encoding utf8` 重定向再 Read
- 灰屏但无错误 ≠ 没渲染：用一次性日志（`std::sync::Once` + `info!`）确认 pass 真的在跑，再排查时序

### 后果

- ✅ 架构定型：compute（RenderGraph schedule，camera_driver 前）→ blit（Core2d PostProcess）→ 上屏，P2 的 DDA pass 直接套用此骨架
- ✅ 阶段 A 的 Sprite 路径代码删除，无中转
- ⚠️ 多相机时 `Query<&ViewTarget>.single()` 需重审（当前单相机）；PostProcess set 内与其他 post system 的顺序依赖将来 tonemapping 接入时再梳理（P3.5）

---

## ADR-0004：决策存放原则与工程基线（P0.5/P0.6）

**状态**：已接受
**日期**：2026-08-29

### 决策一：决策存放原则（防止双源漂移）

- **TODO.md「已定关键决策」表是产品/架构决策的唯一真源**（live 文档，随 grill-me 裁决更新）
- **decisions.md 只存工程细节与版本对齐记录**（怎么实现、踩了什么坑、API 长什么样），不复述 TODO 表内容
- 大决策若被推翻：改 TODO 表 + 对应 ADR 标注修订（参考 ADR-0002 踩坑 4 被 ADR-0003 修正的先例）
- 架构级大决策（数据结构/电路语义/模拟位置/画质基线等）的完整论证见 TODO.md v3 及 grill-me 会话记录，此处不复制

### 决策二：日志基线（P0.5）

- **代码默认 INFO 全开，不静音任何 crate**（用户明确要求：不掩盖问题）。注意 `LogPlugin::default()` 的 filter 是 `"wgpu=error,naga=warn"`——静音 wgpu 是默认行为，必须显式覆盖为 `"info"`
- 临时降噪走环境变量 `RUST_LOG`（launch.json / 终端 env），不在代码里做
- 帧时间统计：`FrameTimeDiagnosticsPlugin` + `LogDiagnosticsPlugin`（每秒输出 fps/frame_time 到 INFO）。后续扩展：P2.7 GPU timestamp 接 diagnostics 通道，P7 接 UI 面板

### 决策三：CI 基线（P0.4）

- 本地一键：`scripts/ci.ps1`（fmt --check → clippy `-D warnings` → build → test），提交前必跑
- GitHub Actions：`.github/workflows/ci.yml`（windows-latest，rust-cache），推 GitHub 即生效；同分支并发取消
- clippy 例外：Bevy render system 参数即依赖注入，参数多属常态——`#[allow(clippy::too_many_arguments)]`（Bevy 源码同款），不用配置 lint 阈值

### 后果

- ✅ P0 全部完成（0.1~0.6），进入 P1 体素数据层
- ✅ 决策双源风险消除：TODO 表 = what/why，ADR = how/坑
- ⚠️ ADR 索引表需随新增 ADR 手动维护（低成本，可接受）

---

## ADR-0005：parley 0.9.0 本地补丁——CJK 文本分段（vendor/parley）

**状态**：已接受
**日期**：2026-08-30

### 背景

gate-ui（2.7a）引入中文 UI 文本后，每次排版向 stderr 刷：

```text
ICU4X data error: No segmentation model for complex script: Chinese/Japanese
```

且整段中文被视为单一不可断行单元（word 边界回退到整段末尾）。

根因链（Bevy 0.19.1 → bevy_text 0.19.1 → parley 0.9.0 → icu_segmenter 2.3.0）：

1. parley 0.9.0 用 `LineSegmenter/WordSegmenter::new_for_non_complex_scripts` 构造分段器，
   复杂文字载荷（my/km/lo/th/ja）全 None；
2. CJK 文本进入 word 分段器 complex 路径 → `select(ChineseOrJapanese)` 返回 None →
   icu_provider `with_display_context`（`logging` feature 开启）打出 `log::warn!`——**不是 panic**，
   是警告刷屏 + 断行降级；
3. baked `compiled_data` 里**本来就带有 cjdict**（中文/日文词典约 2MB，`segmenter_dictionary_auto_v1`
   "und/cjdict"），只是从未被加载；
4. parley 0.9 无任何 feature 可启用复杂文字支持；bevy_text 0.19.1 锁定 parley 0.9，
   无法升级到带修复的版本。

注意区分：行分段器侧上游**刻意**不加载 cjdict（line.rs `load_dictionary` 注释：UAX #14 的
ID 类断行属性允许汉字间断行），行侧 CJK 行为本来就正确；出问题的只有 word 分段器侧
（CJK 分词必须依赖词典）。

### 决策

- vendor parley 0.9.0（registry 提取物原样复制）到 `vendor/parley`，
  根 Cargo.toml `[patch.crates-io] parley = { path = "vendor/parley" }` 接管；
- 唯一改动点 `src/analysis/mod.rs`：`word_segmenter` 改用
  `WordSegmenter::new_dictionary`（加载 SEA 词典 + cjdict），`line_segmenter` 三个
  WordBreak 分支改用 `LineSegmenter::new_dictionary`（顺带修复泰/缅/高棉/老挝断行，
  CJK 行侧行为不变）；
- const 构造改为运行时构造（`new_dictionary` 非 const；每次调用 5 次静态 zerotrie 查找，
  开销可忽略，返回值仍借用 `'static` 数据）；
- 放 `vendor/` 而非 `third_party/`（后者被 .gitignore 排除，补丁必须可提交）；
  workspace `exclude` 增加 `"vendor"` 防止被吸收为成员（同 ADR-0002 的 cargo#12154 坑）。

### 踩坑记录

- 报错无 panic 包装、应用照常 60fps 运行，容易被当成无害噪音——实际断行行为已降级；
- `DataError::with_display_context` 在 `logging` feature 下即 `log::warn!`，这就是打印源
  （错误本身在 select() 里被丢弃、优雅回退）；
- registry 原地改源码的方式在 `cargo update`/registry 重提取后会静默丢失，必须走
  `[patch]` + vendor；
- **中文方框是字体缺失**，与 parley/ICU4X 无关——警告刷屏和方框是两个独立问题，
  曾短暂误判为同一根因而误撤 vendor 补丁，已恢复。

### 后果

- ✅ 实机验证：运行 12s ICU4X 警告 0 次（修复前主题加载后立即刷屏），CJK 断行正确
- ✅ workspace 98 测试全绿，parley 公开 API 未变，bevy_text 无感
- ⚠️ 二进制增大：cjdict ≈2MB + SEA 词典被链接器保留
- ⚠️ parley 升级或 bevy_text 改绑更高版本时，vendor 补丁需人工重放