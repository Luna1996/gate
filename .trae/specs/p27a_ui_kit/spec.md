# 2.7a UI 组件库基座（gate-ui）+ 响应式与世界空间 - 产品需求规范（Spec）

> 修订 r2（2026-08-30）：按用户要求纳入 **任意分辨率/比例适配（UI + 场景）** 与 **世界空间 UI**，原 r1 两条 Non-Goal 取消。

## Overview
- **Summary**：新建 workspace 成员 `gate-ui`（纯 bevy_ui 之上的自研组件库，唯一依赖 bevy + serde/ron）。三层结构：**主题令牌**（`UiTheme` RON 数据资产，缺失回退内置暗色默认）→ **widget 层**（Panel/Label/Button/Slider/Checkbox/Plot 折线图/滚动列表 + **WorldAnchor 世界空间标注**）→ **输入门控**（`UiPointerCaptured`）。同任务打通**全链路响应式**：窗口任意 resize 下 UI 布局不破、场景渲染分辨率/相机 aspect 实时跟随（gate-render VIEW_SIZE 常量 → 资源 + 纹理重建链）。gate-app 挂 demo 性能面板 + 世界空间标注实机验收。
- **Purpose**：统一 UI 框架裁决（2026-08-30）的落地基座 + 补齐分辨率适配短板（2.6 遗留的 aspect 锁定假设）。
- **Target Users**：开发者（性能面板/观测工具）+ 玩家（后续 HUD/菜单同框架同主题）。

## Goals
- 复用 bevy_ui 0.19 已有能力（Taffy、圆角/边框/半透明、Interaction），自研只做**主题令牌层 + widget 层**，不重造布局引擎
- 风格 = Modern UI mod 参照：暗色、半透明圆角面板、扁平控件、强调色点缀、清晰排版
- 主题令牌数据资产化（对齐"光照主题为数据资产"哲学）：改 RON 换肤，不改代码
- **任意分辨率/比例**：窗口 resize 后 UI 布局自适应（相对单位 + 锚定），场景渲染正确（渲染纹理重建 + aspect 实时重算，画面无拉伸变形、无裁切错位）
- **世界空间 UI**：3D 锚点标注（如元件信息卡）每帧投影到屏幕空间，由 gate-ui widget 渲染——完整复用主题/widget/门控；遮挡检测复用 CPU DDA（v0 可选）
- Plot 折线图：环形缓冲 + CPU 光栅化 + ImageNode，逻辑可 headless 单测

## Non-Goals
- 不做 IME / 文本输入框（P4+ 编辑器需要时追加）
- 不做动画系统（v0 hover 即时切换视觉态，缓动后置）
- 不做字体资产分发：`font_path: Option<String>`，缺省回退 `default_font`（CJK 由用户放入 assets 后在 theme.ron 指定）
- 世界空间 UI 不做 3D mesh billboard 渲染路径（文本/控件不进 3D 材质管线）——统一走投影锚定方案，保持单渲染路径

## 功能需求

### FR-1 主题令牌 `UiTheme`（数据资产 + 响应式参数）
- 颜色组：`panel_bg`（带 alpha）/ `panel_border` / `text` / `text_muted` / `accent` / `accent_hover` / `accent_pressed` / `success` / `warning` / `danger`
- 度量组：`corner_radius` / `border_width` / `spacing`（xs/sm/md/lg）/ `font_size`（sm/md/lg）/ `ui_scale: f32`（全局 UI 缩放，接到 Bevy `UiScale` 资源；`auto_fit_ui_scale: bool` 时按窗口高度/720 推导）
- `font_path: Option<String>`
- 资产：`assets/ui/theme.ron`；加载失败/缺失 → 内置暗色默认 + `warn!`（不 panic，不静默）

### FR-2 widget 层（retained spawn API，响应式约定）
- 全部为 `gate_ui::widgets::*` spawn 函数 + 状态组件 + observer 事件；不引入宏 DSL
- **响应式布局约定**：容器级尺寸/边距优先 `Val::Percent` / `Val::VMin` 语义（bevy_ui Val::Vw/Vh 若 0.19 支持则用之，否则 Percent + UiScale 兜底）；控件本体（按钮/滑杆）允许固定像素——面板锚定边缘（PositionType::Absolute + Percent 边距），窗口任意尺寸下不越界不重叠
- `panel` / `label` / `button`（Interaction 状态机 + UiClick）/ `slider`（SliderValue + clamp/step + 拖动）/ `checkbox`（Checked 切换）/ `plot`（FR-4）/ `list`（固定容量环形列表；滚动裁剪能力核实后决定 v0 是否带滚动，结论记 review）

### FR-3 世界空间 UI：WorldAnchor 投影锚定
- 组件 `WorldAnchor { pos_fine: Vec3, visible: bool }`（挂任意 UI 节点）
- system 每帧：用 `DdaCameraConfig.inv_view_proj`（已有，2.6 交付）把 pos_fine 投影到 NDC → 屏幕像素 → 写回节点 `Node` 的绝对定位（left/top）；NDC 超出 [-1,1]（视锥外）→ `Visibility::Hidden`
- 距离衰减：可选 `scale_with_distance`（投影时按 1/w 缩放节点，用 UiScale 局部等价实现或直接缩放 font_size/尺寸 token）
- 遮挡检测（v0 可选，Open Question OQ-2）：锚点→相机射线走 CPU DDA（复用 `cpu_reference_dda_ray`），被体素遮挡 → Hidden
- 多锚点聚合、面板内嵌控件跟随等复杂形态后置 P7.3；v0 交付：锚点 + label + 可见性 + 距离缩放

### FR-4 Plot 折线图
- `PlotData { samples: 环形缓冲(N), y_domain: Fixed|Auto, line_color, grid: bool }` + `push(f32)`
- CPU 光栅化 rgba8 Image（如 256×64，尺寸随面板宽高自适应重建）+ ImageNode + 网格线/极值标注
- 纯函数化，headless 单测断言像素

### FR-5 场景响应式（gate-render/gate-app 侧）
- `VIEW_SIZE` 常量 → `RenderScale` 资源（当前渲染分辨率），gradient/DDA 目标纹理 resize 时原地重建（`Assets<Image>::get_mut + resize`，Handle 不变，引用方零改动）
- `GradientUniforms.size` 每帧跟随资源；blit 为归一化 UV 采样，无需改
- `DdaCameraConfig` aspect 动态化：`from_orbit` 已参数化（2.6 交付），新增 resize 监听 system——窗口尺寸变化 → 重算 perspective_rh（fov_y 固定、aspect 更新）→ inv_view_proj 同步
- gate-app 解锁 `resizable: true`（撤销 2.6 的锁定）
- 渲染内部分辨率策略 v0：跟随窗口物理像素（1080p 基线不变；性能余量见 NFR-3）

### FR-6 输入门控 `UiPointerCaptured`
- 任一 `Interaction ∈ {Hovered, Pressed}` → true；Update 每帧重算
- `orbit_camera_input` 开头检查：captured → 跳过旋转/平移/缩放（滚轮一并吞）

## 非功能需求
- NFR-1 测试：主题解析/回退、Plot 环形缓冲/域/光栅化、slider clamp/step、门控翻转、投影锚定数学（已知 view_proj 下锚点→屏幕坐标断言、视锥外隐藏）、resize 后 aspect 重算正确性——全部 headless 单测；CI fmt/clippy/test 全绿
- NFR-2 依赖纪律：gate-ui 仅依赖 bevy（workspace 锁 =0.19.1，新增 `bevy_ui` feature）+ serde/ron；不引入 egui/lunex 等第三方 UI
- NFR-3 性能：demo 面板开启帧率无可感知回退（对照 2.6 基线 60fps）；Plot 光栅化 μs 量级；**resize 重建不炸帧**（纹理重建单帧完成，预留 16.7ms 预算断言进 2.9 极限测试）
- NFR-4 "不隐藏日志"：主题回退、纹理重建、投影隐藏均 info!/warn! 可观测

## 假设与约束
- bevy_ui 0.19 圆角/边框/Val::Vw/Vh/ScrollPosition 能力以动工核实为准（OQ-1）
- 场景适配改动跨 crate（gate-render + gate-app），随本任务一并交付——2.6 review 中"aspect 恒定"假设由本任务解除
- 世界空间 UI 依赖 2.6 的 `inv_view_proj` 精度（求逆误差 1.2e-4，锚定像素级够用）

## 验收标准
- AC-1：workspace 接线（members + bevy_ui feature + gate-ui 骨架），全 workspace 编译零警告
- AC-2：theme.ron 加载生效；删除/破坏回退默认 + warn 日志（两情形实测）
- AC-3：六类 widget spawn 结构 + 交互状态机 headless 断言通过
- AC-4：Plot 环形顶替/域计算/光栅化像素断言通过
- AC-5：**响应式实机**——窗口任意拉伸（≥3 种尺寸/比例），UI 布局不破不越界；场景无拉伸变形（地板网格线仍平行）、渲染正确跟随；2.7a 前后帧率对照无回退
- AC-6：**世界空间实机**——体素场景至少 2 个锚点标注，相机旋转/缩放时标注跟随锚点、转到背面时隐藏（或遮挡隐藏，若 OQ-2 交付）
- AC-7：hover UI 拖拽时轨道相机静止，离开恢复
- AC-8：fmt/clippy -D warnings 清零；workspace 测试 ≥70 绿（58 + gate-ui/响应式新增 ≥12）

## Open Questions
- OQ-1：bevy_ui 0.19 滚动裁剪（ScrollPosition/overflow clip）与 Val::Vw/Vh 支持度——Task 2/5 核实，结论记 review
- OQ-2：世界空间遮挡检测（CPU DDA）是否随 v0 交付——Task 6 评估成本，若交付则 AC-6 升级为遮挡隐藏
- OQ-3：Plot 重绘节流（每帧 vs 脏标记）——Task 3 实测定，默认每帧
