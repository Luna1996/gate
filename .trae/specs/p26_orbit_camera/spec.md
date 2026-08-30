# P2.6 轨道相机 + 视图矩阵实时传递 - 产品需求规范（Spec）

## Overview
- **Summary**：将 P2.4 的静态 `DdaCameraConfig::build_static()`（Startup 注入一次，永不变化）替换为**轨道相机（orbit camera）**：`OrbitCamera { target, distance, yaw, pitch }` 资源 + gate-app 输入 system（右键拖拽旋转 / 中键拖拽平移 / 滚轮缩放），每帧 Update 由 orbit 参数重算 `DdaCameraConfig`（view_proj / inv_view_proj / position_world），经既有 ExtractSchedule → `DdaViewUniform` → PrepareBindGroups `write_buffer` 链路实时传给 DDA 着色器。**WGSL 与渲染链路零改动**（uniform 每帧已重写，本次只换数据来源）。
- **Purpose**：兑现 TODO 2.6「Bevy 相机接入：轨道相机（平移/旋转/缩放），视图矩阵实时传递」+ 清偿 P0.3 遗留项「Bevy 相机视图矩阵 → system 可见的 uniform」+ P2.4 review §3.2-2 遗留「Dda 硬编码 Mat4 → P2.6 同步」。这是 2.8 验收「相机自由飞行」的前置。
- **Target Users**：开发者（自由视角检查场景/热点/增量编辑区域；P4 编辑闭环的视角基础）。

## Goals
- 轨道相机三操作全可用：旋转（orbit）、平移（pan）、缩放（zoom），操作下一帧画面即生效（≤1 帧延迟）
- 初始视图与 P2.4 `build_static()` 完全等价（eye fine (700,560,700) / target (260,120,260) / fov 60° / aspect 1280:720 / near 1.0 / far 4000），切换瞬间画面零跳变
- 全部数学在 gate-render 单测覆盖：from_eye 往返一致、clamp 边界、from_orbit 与 build_static 矩阵逐元素一致
- 实机 500 帧连续操作无 panic，VUID 仍仅 wgpu#9213 两条初始

## Non-Goals
- **不做窗口 resize 适配**：窗口 `resizable: false` 锁定 1280×720（aspect 恒定，DDA 输出纹理与 ViewTarget 尺寸绑定，resize 拉伸变形问题不在本期；P2.7 性能面板或后续需要时再开）
- **不做惯性/阻尼**（P7.4 相机打磨）：本期直接 1:1 增量映射
- **不做键盘飞行（WASD）/ first-person 模式**：轨道相机已满足 2.8「自由飞行」验收（围绕场景自由观察）
- **不接管 Camera2d 实体**：Camera2d 仍为空壳（ViewTarget 时序载体），DDA 消费自算 Mat4（P2.4 spec Assumption-4 既定方案）
- **不做焦点跟随/命中点拾取**（P4.1 CPU 拾取后才有）
- **不触 gate-render 之外任何渲染改动**：dda.wgsl / blit.wgsl / 插件调度时序全部不动

## Background & Context
1. **既有数据链路**（P2.4 已验证，本次只换源头）：main world `DdaCameraConfig`（Resource）→ `extract_camera_config`（ExtractSchedule，`Extract<Res<>>` 跨 world）→ render world `DdaViewUniform`（ShaderType 96B：mat4 + vec4 cam_pos_fine + vec4 pad）→ `prepare_dda_bind_groups` 每帧 `UniformBuffer::write_buffer` → BG0 binding → `dda_main` 反投影射线
2. **静态参数现状**：`build_static()` = perspective_rh(60°, 1280/720, 1.0, 4000) × look_at_rh(eye=(700,560,700), target=(260,120,260), up=Y)；`DdaCameraConfig { view_proj, inv_view_proj, position_world }`
3. **坐标约定**：世界单位 = fine 单位（0.25cm）；右手系 +Y 上，`look_at_rh` 朝 -Z；DDA 着色器 dir 不缩放
4. **输入基础设施**：Bevy 0.19 `ButtonInput<MouseButton>` + `AccumulatedMouseMotion` + `AccumulatedMouseScroll`（Update 阶段可读，帧内增量已累计）；winit 鼠标事件由 DefaultPlugins 提供
5. **窗口现状**：`Window { resolution: VIEW_SIZE.into(), present_mode: AutoNoVsync }`，未显式设 resizable（winit 默认 true——本期显式关闭）

## Functional Requirements
- **FR-1 · OrbitCamera 资源**（gate-render `dda.rs` 定义，gate-app 持有并操作）：
  - 字段：`target: Vec3`（注视点，fine）、`distance: f32`（相机到 target 距离，fine）、`yaw: f32`（绕 +Y 方位角，rad）、`pitch: f32`（仰角，rad）
  - `fn from_eye(eye: Vec3, target: Vec3) -> Self`：offset = eye − target；distance = offset.length()；pitch = asin(offset.y / distance)；yaw = atan2(offset.x, offset.z)
  - 眼位重建公式（from_orbit 用）：`eye = target + distance * (sin_yaw * cos_pitch, sin_pitch, cos_yaw * cos_pitch)`
- **FR-2 · DdaCameraConfig::from_orbit(orbit, fov_y, aspect, near, far)**：按 FR-1 公式算 eye → `perspective_rh(fov_y, aspect, near, far) × look_at_rh(eye, target, Y)` → view_proj / inv_view_proj / position_world = eye。fov/aspect/near/far 为参数（默认沿用 60° / 1280:720 / 1.0 / 4000），不写死在 orbit 里
- **FR-3 · 输入映射**（gate-app Update system `orbit_camera_input`）：
  - **右键按住拖拽** = 旋转：`yaw -= dx * ROT_SPEED; pitch += dy * ROT_SPEED`，ROT_SPEED = 0.005 rad/px；pitch 夹到 `±89°`（1.5533 rad）
  - **中键按住拖拽** = 平移：pan 向量 = `(right * (-dx) + up * dy) * PAN_PER_PX`，`right/up` 由 yaw/pitch 正交基给出，`PAN_PER_PX = distance * 2*tan(fov_y/2) / VIEW_SIZE.y`（视觉 1:1 抓取）；target += pan
  - **滚轮** = 缩放：`distance *= exp(-scroll_line * ZOOM_LOG_SPEED)`，ZOOM_LOG_SPEED = 0.35/行（乘法缩放，各距离档手感一致）
  - clamp：pitch ∈ ±89°；distance ∈ [32, 8000]（近不至于穿进体素内部失稳，远可看 tile(2,0,0) 全景）；yaw 不 clamp（自由旋转）
  - 三操作互斥：同帧多键按下时以 旋转 > 平移 > 缩放 优先级取一（拖拽类不叠加，滚轮可与拖拽共存）
- **FR-4 · 每帧管线**：`orbit_camera_input`（Update）改 `OrbitCamera` → 同 system 内 `ResMut<DdaCameraConfig>` 直接 `*cfg = DdaCameraConfig::from_orbit(...)`（不复制矩阵代码，from_orbit 是唯一矩阵构造点；build_static 保留但仅测试使用，或删除——由实现取简洁者，若删除则单测直接用 from_orbit(from_eye(...)) 对照历史常量）
- **FR-5 · 初始状态**：Startup 注入 `OrbitCamera::from_eye(Vec3::new(700,560,700), Vec3::new(260,120,260))` + `DdaCameraConfig::from_orbit(...)`（同参数），**替换** `build_static()` 调用点
- **FR-6 · 窗口锁定**：`Window { resizable: false, .. }`（aspect 恒定 1280:720）
- **FR-7 · 导出**：`OrbitCamera` 经 `brickmap/mod.rs` + `lib.rs` re-export（gate-app use）

## Non-Functional Requirements
- **NFR-1 · 新增单测 ≥3**（gate-render）：① from_eye ↔ 眼位重建往返（误差 <1e-4）② pitch/distance clamp 生效 ③ from_orbit(from_eye(build_static 参数)) 的 view_proj/inv_view_proj/position_world 与 build_static 逐元素一致（<1e-5，回归保护）
- **NFR-2 · CI 护栏**：fmt/clippy(-D warnings)/test 全绿；现有 55 测试语义不变
- **NFR-3 · 实机**：500 帧 + 拖拽/缩放/平移操作无 panic；VUID 仅 wgpu#9213 两条；旋转停止后画面稳定（无漂移 = 累计输入用完即清）
- **NFR-4 · 响应性**：输入到画面生效 ≤1 帧（Update 改资源 → 同帧 Extract → 同帧 PrepareBindGroups write_buffer → 本帧 dispatch）

## Constraints
- **Technical 1**：DDA 着色器 uniform 布局不变（DdaViewUniform 字段不动），WGSL 零改动
- **Technical 2**：输入 system 放 gate-app（main world）；gate-render 不依赖 bevy_input（保持渲染 crate 纯净）
- **Technical 3**：NoVsync 下拖拽增量小而频，AccumulatedMouseMotion 帧内累计天然帧率无关，无需 per-event 处理
- **Dependencies**：P2.4 全部链路（本任务不改任何渲染代码路径）
- **Business**：键鼠首发（决策表既定）；手柄 P12.4 后置

## Assumptions
1. Bevy 0.19 存在 `AccumulatedMouseMotion` / `AccumulatedMouseScroll`（0.15 引入并沿用；若 API 名有出入，fallback = Local<(Vec2, f32)> 手写累积，语义不变）
2. `look_at_rh` up=Y 在 pitch→±89° 附近仍稳定（clamp 89° 留 1° 余量，不出现 up 与 view 向量共线）
3. Camera2d 空壳方案沿用（P2.4 spec Q3 既定裁决），Camera2d 的 Transform 不动（默认 (0,0,0) 朝 -Z），Core2d PostProcess 时序不受 orbit 影响

## Acceptance Criteria

### AC-1: 数学单测（from_eye 往返 / clamp / 矩阵等价）
- **Type**: rule
- **Given**: gate-render dda.rs 新增 OrbitCamera 及单测
- **When**: `cargo test -p gate-render --lib orbit`
- **Then**: ① from_eye((700,560,700),(260,120,260)) → from_orbit 重建 eye 误差 <1e-4 ② pitch=95° 输入被夹到 89°、distance=1 输入被夹到 32 ③ from_orbit 与 build_static（同参数）三字段逐元素 <1e-5
- **Pass Condition**: 新增 ≥3 测试全绿
- **Evidence**: CI 测试输出

### AC-2: 初始画面零跳变
- **Type**: rule
- **Given**: 替换后首帧 Startup 注入 from_orbit 初始配置
- **When**: 启动对比 P2.4/2.5 基线截图
- **Then**: 构图一致（同一 eye/target/fov），仅因 WGSL 无改动而完全相同的画面
- **Pass Condition**: 启动截图与 2.5 基线截图构图一致
- **Evidence**: 启动截图（docs/ 留档）

### AC-3: 三操作实机生效
- **Type**: rule
- **Given**: `cargo run -p gate-app` 运行中
- **When**: 右键拖拽旋转 / 中键拖拽平移 / 滚轮缩放
- **Then**: 画面视角实时变化且方向正确（拖右 = 场景左移；上滚 = 拉近或拉远按 ZOOM_LOG_SPEED 符号定义验证）；停止操作画面静止
- **Pass Condition**: 旋转后/缩放后/平移后三张截图与初始构图明显不同且符合操作语义
- **Evidence**: 操作截图 ×3（review 人工核实）

### AC-4: 实机稳定性
- **Type**: rule
- **Given**: 持续操作 ≥500 帧（含旋转+缩放+平移混操作）
- **When**: 检查运行日志
- **Then**: 无 panic；VUID 仅 wgpu#9213 两条；UPLOAD[incremental] 周期日志持续（编辑系统不受相机影响）
- **Pass Condition**: 日志 grep 通过
- **Evidence**: run log 留档

### AC-5: fmt / clippy / test CI 护栏
- **Type**: rule
- **Given**: 全部改动落盘
- **When**: `cargo fmt --all --check` + `cargo clippy --workspace --all-targets -- -D warnings` + `cargo test --workspace`
- **Then**: 三条 exit 0，≥58 tests 绿
- **Pass Condition**: exit 0 全清
- **Evidence**: 命令输出

### AC-6: 手感评分
- **Type**: rubric
- **Dimension**: 拖拽灵敏度、缩放平滑度、平移跟手度（开发自测）
- **Scale**: 1-5
- **Anchors**: 1 = 拖不动或一拖飞出场景；3 = 可用但需要适应；5 = 灵敏度适中、缩放各档一致、平移 1:1 跟手
- **Pass Threshold**: ≥4
- **Evidence**: review 自测记录（灵敏度常量如有调整附最终值）

## Open Questions
- [x] Q1: 相机实体是否接管 ViewTarget？否——Camera2d 空壳沿用（Assumption 3，P2.4 Q3 既定）
- [x] Q2: 是否支持键盘飞行？否（Non-Goals；2.8「相机自由飞行」以轨道三操作验收）
- [x] Q3: 光标捕获（pointer lock）？否——增量累计方案无需锁定光标，拖出窗口边界仅暂停旋转（松键复位），体验可接受
