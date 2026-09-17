//! capture：UI 指针门控。
//! 直接对 UI 节点树做命中测试（规则同 bevy_ui `ui_focus_system`），每帧重算；显式 `FocusPolicy::Pass` 的节点不捕获。
//! 游戏输入 system 开头检查 `UiPointerCaptured`，捕获时跳过鼠标操作。

use bevy::prelude::*;
use bevy::ui::{
  ComputedNode, FocusPolicy, Node, OverrideClip, UiGlobalTransform, clip_check_recursive,
};
use bevy::window::PrimaryWindow;

/// 指针是否被 UI 捕获（每帧重算：光标在任一可见 UI 节点上即为 true）
#[derive(Resource, Default, Debug, Clone, Copy, PartialEq)]
pub struct UiPointerCaptured(pub bool);

/// 鼠标拦截标记：挂在 UI 控件根节点（其后代一并被覆盖）；语义 = 悬停其上的鼠标事件全部吞掉。
/// 终止 bevy_ui 命中链；`Node` 的 `FocusPolicy` 默认 `Pass`，给装饰性子节点误设 `Block` 会截断命中链、父节点收不到 hover。
#[derive(Component, Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct MouseIntercept;

/// 指针是否落在带 `MouseIntercept` 的节点上（每帧重算）
#[derive(Resource, Default, Debug, Clone, Copy, PartialEq)]
pub struct MouseIntercepted(pub bool);

/// UI 节点树命中测试（每帧重算）：命中 = 可见 + 非零尺寸 + 在祖先裁剪区内 + 非 `FocusPolicy::Pass`。
/// `set_if_neq` 仅在翻转时触发 change 检测。
#[allow(clippy::type_complexity)]
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
        break;
      }
      if !vis.is_some_and(|v| v.get()) {
        continue;
      }
      if node.size() == Vec2::ZERO {
        continue;
      }
      // Pass = 显式穿透（本组件库全用 Block）
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
