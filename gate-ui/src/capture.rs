//! capture：UI 指针门控。
//!
//! 直接对 UI 节点树做命中测试，规则与 bevy_ui 0.19 `ui_focus_system` 一致：光标落在
//! 「可见 + 非零尺寸 + 祖先裁剪区内」的 UI 节点上即视为捕获（每帧重算）；显式
//! [`FocusPolicy::Pass`] 的节点不捕获，交互穿透到下层。
//!
//! 轨道相机等游戏输入 system 开头检查 [`UiPointerCaptured`]，捕获时跳过一切鼠标操作。

use bevy::prelude::*;
use bevy::ui::{
  ComputedNode, FocusPolicy, Node, OverrideClip, UiGlobalTransform, clip_check_recursive,
};
use bevy::window::PrimaryWindow;

/// 指针是否被 UI 捕获（每帧重算：光标在任一可见 UI 节点上即为 true）
#[derive(Resource, Default, Debug, Clone, Copy, PartialEq)]
pub struct UiPointerCaptured(pub bool);

/// 鼠标拦截标记：挂在任意 UI 节点上（控件根节点即可，其后代一并被覆盖）。
///
/// 语义 = 「悬停其上的鼠标事件（按键、移动、滚轮）全部吞掉」：本节点及祖先若为
/// [`FocusPolicy::Block`]（组件库只在*可交互根节点*上显式设置；注意 `Node` 的
/// `FocusPolicy` 默认值是 [`FocusPolicy::Pass`]，给装饰性子节点误设 Block 会在命中链上截断，
/// 让父节点收不到 hover）会终止 bevy_ui 的命中链，下层 UI 收不到 hover；
/// 同时 [`MouseIntercepted`] 置真，3D 场景输入侧据此跳过（见 gate-app 相机输入）。
#[derive(Component, Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct MouseIntercept;

/// 指针是否落在带 [`MouseIntercept`] 的节点上（每帧重算）
#[derive(Resource, Default, Debug, Clone, Copy, PartialEq)]
pub struct MouseIntercepted(pub bool);

/// UI 节点树命中测试（每帧重算；set_if_neq 仅在翻转时触发 change 检测）
///
/// 判定条件与 bevy_ui `ui_focus_system` / picking backend 逐条对齐：
/// - `InheritedVisibility` 为真（隐藏的节点不可交互；无该组件的节点跳过）
/// - `ComputedNode.size != 0`（Display::None 布局为零尺寸）
/// - `ComputedNode::contains_point`（节点变换后的物理像素矩形；圆角按实际半径）
/// - `clip_check_recursive`（光标被任一 `Overflow::clip()` 祖先裁掉则不算命中）
/// - `FocusPolicy::Pass` 的节点本身不捕获（下层 Block 节点仍可命中）
///
/// 面板背景/容器/label 等节点没有 `Interaction`，故不能只统计 Interaction 状态。
#[allow(clippy::type_complexity)] // Bevy system：多组件查询签名固有
pub fn ui_pointer_capture_system(
  windows: Query<&Window, With<PrimaryWindow>>,
  nodes: Query<(
    Entity,
    &ComputedNode,
    &UiGlobalTransform,
    Option<&InheritedVisibility>,
    Option<&FocusPolicy>,
    Option<&MouseIntercept>,
  )>,
  clipping: Query<(&ComputedNode, &UiGlobalTransform, &Node)>,
  child_of: Query<&ChildOf, Without<OverrideClip>>,
  mut captured: ResMut<UiPointerCaptured>,
  mut intercepted: ResMut<MouseIntercepted>,
) {
  let mut over_ui = false;
  let mut over_intercept = false;
  if let Ok(window) = windows.single()
    && let Some(cursor) = window.physical_cursor_position()
  {
    for (entity, node, transform, vis, focus, intercept) in &nodes {
      if over_ui && over_intercept {
        break; // 两个标志都已确定，无需继续遍历
      }
      if !vis.is_some_and(|v| v.get()) {
        continue;
      }
      if node.size() == Vec2::ZERO {
        continue;
      }
      // FocusPolicy::Pass = 显式让交互穿透（本组件库全部用 Block）
      if focus == Some(&FocusPolicy::Pass) {
        continue;
      }
      if node.contains_point(*transform, cursor)
        && clip_check_recursive(cursor, entity, &clipping, &child_of)
      {
        over_ui = true;
        if intercept.is_some() {
          over_intercept = true;
        }
      }
    }
  }
  captured.set_if_neq(UiPointerCaptured(over_ui));
  intercepted.set_if_neq(MouseIntercepted(over_intercept));
}
