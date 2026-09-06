---
name: "gate-ui-interaction-debug"
description: "gate-ui（自研 bevy_ui 0.19 组件库）输入穿透/指针门控/显隐/滚轮类问题排查：对照引擎 focus.rs 命中语义定位。当 UI 操作漏到 3D 场景（相机旋转/缩放/拾取穿透）、hover/点击无响应、隐藏面板仍交互、滚动方向异常时调用。"
---

# gate-ui 交互/输入/显隐问题排查工作流

适用于 `gate-ui/src/**`（widgets、capture.rs）与 `gate-app/src`（相机输入消费者）。
核心方法：**gate 的门控/显隐逻辑必须与 bevy_ui 0.19 引擎自身的命中/焦点语义逐条对齐**——
不要猜，直接读引擎源码对照。

## 铁律：引擎语义优先

bevy 0.19 UI 源码（registry 内，随版本固定）：
- 焦点/Interaction：`bevy_ui-*/src/focus.rs`（`ui_focus_system` + `clip_check_recursive`）
- 拾取：`bevy_ui-*/src/picking_backend.rs`（`ui_picking`）
- 节点命中：`bevy_ui-*/src/ui_node.rs`（`ComputedNode::contains_point`）
- 变换：`bevy_ui-*/src/ui_transform.rs`（`UiGlobalTransform`，节点中心 = transform 平移点，范围 ±size/2）

registry 路径：`$env:USERPROFILE\.cargo\registry\src\index.crates.io-*\bevy_ui-<ver>\src\`
（版本以 `cargo metadata` 为准，当前 bevy 0.19.1）。

### 命中判定（focus.rs / picking_backend.rs 共同规则）

光标算"在 UI 上"当且仅当存在一个节点同时满足：

1. `InheritedVisibility` 为 true（隐藏节点不可交互；注意这是**每帧由
   `VisibilityPropagate` 系统从 `Visibility` 组件计算**的，不是 Visibility 本身）
2. `ComputedNode.size != Vec2::ZERO`（Display::None / 未布局完 = 零尺寸）
3. `computed_node.contains_point(ui_global_transform, physical_cursor)`（物理像素空间，
   含圆角半径判定）
4. `clip_check_recursive(cursor, entity, ...)` 通过（光标被任一 `Overflow::clip()` 祖先
   裁掉则不算；`OverrideClip` 组件可豁免）
5. 节点**没有** `FocusPolicy::Pass`（Pass = 显式穿透，继续找下层；默认 Block = 阻挡）

遍历按 UiStack 从顶到底，遇到 Block 即停。

### 关键陷阱（都踩过）

- **无 `Interaction` 组件的节点照样命中并阻挡**：focus.rs 对所有可见节点做命中测试，
  但只给挂了 `Interaction` 组件的节点写 Hovered/Pressed 状态。面板背景、容器、label
  命中时不产生任何 Interaction 变化。→ **门控系统不能扫描 `Interaction != None`**，
  必须自己做命中测试（gate-ui `capture.rs` 已复刻：`UiPointerCaptured` 用
  `Query<(Entity,&ComputedNode,&UiGlobalTransform,Option<&InheritedVisibility>,Option<&FocusPolicy>)>`
  + `contains_point` + `clip_check_recursive`，这些都是 bevy_ui pub API）。
- **坐标系**：`Window::physical_cursor_position()` 返回物理像素，与
  ComputedNode/UiGlobalTransform 同空间；`cursor_position()` 是逻辑像素（要 ×scale_factor）。
  getter 还会做窗口边界裁剪（光标在窗口外/负坐标 → None）。
- **`Visibility` 三态语义**：`Visible`=强制可见（无视祖先）、`Hidden`=强制隐藏、
  `Inherited`=跟随父级。面板整体隐藏用根节点 `Visibility::Hidden`；叠放的子页面/
  勾选标记用 `Visibility::Inherited`（Hidden 时会浮在场景上，已踩过两次）。
- **滚轮事件**：bevy 0.19 中 `MouseWheel` 是 **Message 不是 Event**，读用
  `MessageReader<MouseWheel>`；连续累积量用 `Res<AccumulatedMouseScroll>`。
  方向约定：`ev.y > 0` = 滚轮向上滚（内容应下移、scroll 偏移趋向 0）；Pixel 单位
  （触控板）按 16px=1 行折算，与相机缩放口径一致。
- **`InheritedVisibility::default()` = HIDDEN**（bool 默认 false）。裸 App 单测里
  没有 VisibilityPropagate 系统，必须显式插 `InheritedVisibility::VISIBLE`；
  真机由传播系统计算。
- **门控消费者**：gate-app 所有 3D 鼠标输入（`orbit_camera_input` 右键旋转/中键平移/
  滚轮缩放、`left_click_pick_recenter`）开头必须 `if captured.0 { ... }` 整块跳过。
  新增鼠标交互系统时照此办理。

## 单测套路（capture.rs / scroll_view.rs 已示范）

裸 `App::new()` + `add_systems(Update, system)`，手工构造数据，`app.update()` 驱动：

```rust
// 节点：ComputedNode{size} + UiGlobalTransform::from_translation(中心) + InheritedVisibility::VISIBLE
// 窗口：app.world_mut().spawn((Window::default(), PrimaryWindow));
// 设光标：window.set_physical_cursor_position(Some(DVec2::new(x,y)));
// 断言：app.world().resource::<UiPointerCaptured>().0
```

覆盖场景：无光标 / 节点内 / 节点外 / Hidden / FocusPolicy::Pass。
注意 `World::query_*` 取数据用 `&mut World`（`world_mut()`），查询迭代用 `&World`。

## 排错 checklist

1. 输入穿透到 3D？→ 确认门控系统是否真的做了命中测试（不是扫 Interaction）；
   可疑面板节点是否有 ComputedNode/UiGlobalTransform（布局后才生成）。
2. hover/点击无响应？→ 节点是否挂 `Interaction`；viewport 是否 `FocusPolicy::Block`；
   是否被 `Overflow::clip()` 祖先裁掉（ScrollView 内容超出 viewport 部分不命中）。
3. 隐藏后还能点/悬浮？→ 根节点是否 `Visibility::Hidden`；叠放子节点是否误用
   `Visible`（应 `Inherited`）。
4. 滚轮方向反？→ 对照 `ev.y > 0 = 上滚 = 内容下移`；scroll 偏移正负约定。
5. 改完跑 `cargo test -p gate-ui`（组件库全部单测，含 capture/scroll/widget），
   再 release 构建真机目测（UI 行为无法靠日志判定）。
