# 2.7a UI 组件库基座 + 响应式与世界空间 - 任务分解（r2）

## Task 1: workspace 接线 + 主题令牌（FR-1 / AC-1 / AC-2）
- [ ] 根 Cargo.toml：members += gate-ui；workspace.dependencies.bevy features += "bevy_ui"；workspace.dependencies += gate-ui、serde、ron
- [ ] gate-ui/Cargo.toml + src/lib.rs 骨架（GateUiPlugin 注册入口）
- [ ] `theme.rs`：UiTheme（颜色组 + 度量组 + ui_scale/auto_fit_ui_scale + font_path）+ 内置暗色默认 + UiThemeAsset(RON) + 加载 system（成功插入/失败回退 + warn!）+ UiScale 资源联动
- [ ] assets/ui/theme.ron 示例文件
- [ ] 单测：RON 解析、缺文件回退、坏文件回退
- **验证**：cargo test -p gate-ui（≥3 绿）

## Task 2: 基础 widget（FR-2 / AC-3）
- [ ] `widgets/mod.rs`：spawn API（parent + tokens，返回 Entity）
- [ ] panel / label（纯视觉，token 驱动）
- [ ] button：Interaction 状态机 + UiClick observer 事件 + hover/pressed 视觉态
- [ ] slider：SliderValue(f32) + 拖动 + clamp/step + accent 填充视觉
- [ ] checkbox：Checked(bool) 切换 + 勾选视觉
- [ ] 滚动列表：固定容量环形列表；核实 ScrollPosition/overflow clip（OQ-1，结论记 review）
- [ ] 单测：spawn 结构断言（headless App）+ slider clamp/step + checkbox 翻转
- **验证**：cargo test -p gate-ui（累计 ≥8 绿）

## Task 3: Plot 折线图（FR-4 / AC-4）
- [ ] `plot.rs`：PlotData（环形缓冲 + Fixed/Auto 域）+ push API
- [ ] 纯函数光栅化：网格线 + Bresenham 折线 → rgba8 Image
- [ ] plot widget spawn：ImageNode + 极值文本 + 每帧重绘 system（节流策略按 OQ-3 实测定）
- [ ] 单测：环形顶替、域计算、光栅化像素断言（固定域已知两点连线中点命中）
- **验证**：cargo test -p gate-ui（累计 ≥11 绿）

## Task 4: 输入门控（FR-6 / AC-7）
- [ ] `capture.rs`：UiPointerCaptured 资源 + Interaction 全集扫描 system（每帧重算）
- [ ] gate-app：orbit_camera_input 开头 gate 检查（captured → 跳过旋转/平移/缩放）
- [ ] 单测：门控翻转逻辑
- **验证**：cargo test -p gate-ui（累计 ≥12 绿）

## Task 5: 场景响应式（FR-5 / AC-5）
- [ ] gate-render：VIEW_SIZE 常量 → `RenderScale` 资源（保留 VIEW_SIZE 兼容导出或全量替换调用点：gradient.rs/dda.rs/main.rs）
- [ ] resize 监听 system：Window Resized → Assets<Image> get_mut + resize（gradient + DDA 目标纹理，Handle 不变）+ RenderScale 更新 + GradientUniforms.size 跟随
- [ ] DdaCameraConfig aspect 动态化：resize → from_orbit 重算（fov_y 固定，orbit 资源持有），inv_view_proj 同步更新
- [ ] gate-app：resizable: true（撤销 2.6 锁定）；orbit_camera_input 中 aspect 参数改为读当前窗口
- [ ] 单测：已知窗口尺寸变化 → aspect/inv_view_proj 重算正确（headless）
- **验证**：cargo build 全 workspace 零警告

## Task 6: 世界空间 UI（FR-3 / AC-6）
- [ ] `world_anchor.rs`：WorldAnchor 组件 + 投影 system（inv_view_proj → NDC → 屏幕像素 → Node 绝对定位；视锥外 Hidden）
- [ ] 距离缩放：scale_with_distance 选项
- [ ] OQ-2 评估：CPU DDA 遮挡检测（复用 cpu_reference_dda_ray）成本——可交付则做，否则记 review 后置 P7.3
- [ ] gate-app demo：红塔顶 + GATE 牌 2 个锚点标注
- [ ] 单测：已知 view_proj 下锚点→屏幕坐标断言 + 视锥外隐藏 + 距离缩放单调性
- **验证**：cargo test -p gate-ui（累计 ≥15 绿）

## Task 7: CI 护栏（NFR-1 / AC-8）
- [ ] cargo fmt --all
- [ ] cargo clippy --workspace --all-targets -- -D warnings
- [ ] cargo test --workspace（≥70 绿）
- **验证**：三条 exit 0

## Task 8: 实机验收 + 收尾（AC-2/5/6/7 + NFR-3）
- [ ] 响应式实测：脚本拖拽窗口 ≥3 种尺寸/比例截图（UI 不破 + 场景无变形）；与 SendInput 拖拽窗口边缘
- [ ] 世界空间实测：相机旋转/缩放，锚点标注跟随 + 转出视锥隐藏截图 ×2
- [ ] UI 交互实测：hover 面板拖拽相机静止；slider/checkbox/button 生效
- [ ] 帧率对照 2.6 基线（60fps 无回退；resize 单帧无炸帧）
- [ ] review.md（AC 逐项证据 + OQ 结论）+ TODO.md 勾选 2.7a
- **验证**：截图 + run log 留档 spec 目录
