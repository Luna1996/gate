# DLSS SR 运行时交接文档（完成后删除本文件）

## 当前状态

- **gnu 构建（`build-gate-app-fast-gnu`）全绿**：shader 侧（mvec/depth/jitter）+ dda.rs + responsive 全部编译并过 naga 校验。
- **msvc+dlss 构建（F5 默认路径）差 5 个编译错误**，全部集中在 `gate-render/src/dlss.rs`，见下。

## 剩余编译错误与修法

1. `error: unresolved import ... RenderScale`（dlss.rs ~215 行用到但 import 被删）→ 把 `RenderScale` 加回 `use crate::brickmap::dda::{...}` 列表。
2. `error: field .0 of RenderQueue is private`（`&queue.0`，sync_dlss_ctx 里 DlssSuperResolution::new 需要 `&wgpu::Queue`）→ 见下面「堵点」。
3. `error: field .0 of RenderAdapter is private`（`&adapter.0`，dispatch_dlss_eval 里 `sr.dlss.render(params, encoder, &adapter.0)` 需要 `&wgpu::Adapter`）→ 同「堵点」。
4. `error: field dlss_mvec_view/dlss_depth_view of AuxTexCache is private`（dlss.rs 228 行）→ dda.rs 里这两个字段加 `pub(crate)`（同 struct 里其他 view 字段的可见性模式）。
5. `error: RenderGraphSystems: ScheduleLabel not satisfied`（dlss.rs ~280 `.add_systems(bevy::render::renderer::RenderGraphSystems::Render, ...)`）→ 要用 **schedule 标签** `RenderGraph`：正确写法参照 dda.rs 586 行：`.add_systems(bevy::render::renderer::RenderGraph, dispatch_dlss_eval.in_set(bevy::render::renderer::RenderGraphSystems::Render).after(crate::brickmap::dda::dispatch_dda))`（`RenderGraph` 是 ScheduleLabel，从 `bevy::render::renderer` 导入）。

## 堵点：rc.1 的 RenderQueue/RenderAdapter 没有公开访问器

调研结论（别重复查）：
- `bevy_render-0.20.0-rc.1` 里 `RenderQueue(Queue)`、`RenderAdapter(Adapter)` 都是**私有元组字段**，全 crate 无 `wgpu_queue()/wgpu_adapter()` 访问器（`RenderDevice::wgpu_device()` 存在，queue/adapter 没有）。
- NGX 需要：`DlssSuperResolution::new(..., &Queue)`（创建时内部 submit）、`dlss.render(params, encoder, &Adapter)`（纹理转 NGX 句柄）。

候选解法（按推荐顺序，先试 a）：
- **a. 查 bevy 0.20 final 的 `bevy_anti_arg::dlss` 怎么拿**：final 里有 JMS55 写的 DlssInitPlugin/SR 上下文，很可能 final 的 bevy_render 加了访问器（或 field 改 pub）。如果 final 有 → 最干净的出路是把 bevy 依赖从 `=0.20.0-rc.1` 升到 `=0.20.0`（final 已发布，Breaking 很小：rc.1→final 主要是补齐，AGENTS 工具链栏要同步）。
- **b. 若不想升级**：`wgpu-hal vulkan` 的 `CreateDeviceCallbackArgs` 里若有 raw queue（或 open 结果可从回调拿到）→ 在 DlssInitPlugin 的 create_device 回调里把 raw `wgpu::Queue` 捕获进一个资源，绕开 bevy 访问器。先读 `wgpu-hal-30.0.1/src/vulkan/mod.rs` 里 `CreateDeviceCallbackArgs` 的字段定义再判断可行性。
- **c. 下策**：`[patch.crates-io]` 本地 fork bevy_render 把两个字段改 `pub(crate)` → 重、脏，不推荐。

## 已完成的实现事实（画质验证阶段会用到）

- **mvec 约定**：输出像素单位、Y 向下；`cur - pre`；`cur = ((c.x/w)*0.5+0.5)*W, (0.5-(c.y/w)*0.5)*H`；静止世界 `p_prev == p`（相机重投影吸收）；移动物体目前会拖影——**prev 位姿表是二阶段任务**（上传 grid_descs 的镜像缓冲，shade 里查 prev grid 重投影）。
- **depth**：命中点 `clip.w`（视空间线性 z）；未命中=远点（`cam+dir*1e5`）+ `z=1e30`。
- **jitter**：halton(2)/halton(3) 序列，相位数 `8*ratio²`，`DlssFrame::jitter()` 与 dlss_wgpu 的 `suggested_jitter` 一致；进 `DdaViewUniform.dlss.xy`（新字段，uniform 布局已加），`dlss.zw` = 输出分辨率（0=关）。eval 时把 jitter 原样传 `InJitterOffset`。
- **eval 顺序**：`ctx.add_command_buffer` 会先 flush 主 encoder → 提交序恒为 compute → DLSS eval → blit（已验证 bevy 源码）。
- **reset**：上下文新建置位 + `view_proj` 逐元素 |Δ|>0.5 判传送。
- **exposure**：Automatic（二阶段再换 Manual 喂 eye_adapt_ro 1×1 纹理）。
- **FXAA**：DLSS 激活时自动关（blit_dda_view 里 `!dlss.active`）。
- **布局**：mvec=`rg16float`、depth=`r32float`，BG0 **perf 变体**新 binding 35/36（base bg0 无；shade/glass 才写）。纹理在 `AuxTexCache`，随 render 尺寸重建。
- **尺寸流**：responsive.rs 里 DLSS 开 → `render = window*ratio(mode)`（DLAA=1.0/Q=2/3/B=0.577/P=1/2/U=1/3），`DlssFrame.out_size=window`；关 → 走 factor 菜单。`DlssSettings` 默认 Off，无菜单也安全。

## 待办（按顺序）

1. 修上面 5 个编译错（堵点按 a→b→c）→ `tools/build-dlss.ps1` 全绿。
2. 冒烟：msvc exe 跑起来，日志应出现 `DLSS 探测：支持超分`；DLSS 上下文只在档位非 Off 时建（现在无菜单，可临时改 `DlssSettings::default` 的 mode 或用 config 注入验证）。
3. 菜单三件套（AGENTS 规范）：`assets/ui/debug_menu.toml` 加 `render/dlss/mode` select（关/DLAA/质量/平衡/性能/超性能）；`gate-app/src/debug_menu.rs` 回调（参照 `render/ray/tier` 的写法，label ≤4 汉字）；`assets/locales/zh-CN.toml` 补 `menu.render.dlss.*` 文案 + tip（Markdown，"""多行"""）。
4. 画质验证：先 DLAA（输出=内部，纯 AA 最容易判对错），再 2K Performance；若鬼影/抖动查 mvec 方向与 jitter 补偿；对比参考可用 run-gate.ps1 截图。
5. 二阶段（可另开）：物体 prev 位姿表（物理动态物体 mvec）、Manual exposure、DLSS-RR 评估。
6. 收尾：`cargo fmt --all`、clippy `-D warnings` 全清（含 msvc target）、AGENTS.md 十六节/优化方案.md 记录、`git rm` 本文件、commit。

## 快速命令

```bash
# DLSS 构建（F5 等价）
powershell -NoProfile -ExecutionPolicy Bypass -File tools/build-dlss.ps1 -Profile build-fast
# gnu 构建（无 DLSS）
cargo build --profile build-fast -p gate-app
# 冒烟
./target/x86_64-pc-windows-msvc/build-fast/gate-app.exe
```
