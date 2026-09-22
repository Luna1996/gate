//! capture：UI 指针门控。
//! 事实来源 = picking 的悬停映射（`HoverMap`）：指针落在任一 **UI 节点**上即捕获；
//! 带 `MouseIntercept` 的节点（控件根及其后代）被悬停时吞掉鼠标事件。
//! 注意 picking 的「窗口后端」会把 window 实体也塞进悬停映射（order = -inf，指针在窗口上恒成立），
//! 所以判定必须过滤出真正的 UI 节点（`ComputedNode`），不能只看映射非空。
//! 游戏输入 system 开头检查 `UiPointerCaptured`，捕获时跳过鼠标操作。

use bevy::picking::hover::{HoverMap, Hovered};
use bevy::picking::pointer::PointerId;
use bevy::prelude::*;
use bevy::ui::ComputedNode;

/// 指针是否被 UI 捕获（每帧重算：悬停映射里有 UI 节点 = 光标下有 UI）
#[derive(Resource, Default, Debug, Clone, Copy, PartialEq)]
pub struct UiPointerCaptured(pub bool);

/// 鼠标拦截标记：挂在 UI 控件根节点（其后代一并被覆盖）；语义 = 悬停其上的鼠标事件全部吞掉。
/// 需同时在同节点挂 `Hovered`（见 `UiInteractBundle` 或手动 spawn），本系统按它判定悬停。
#[derive(Component, Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct MouseIntercept;

/// 指针是否落在带 `MouseIntercept` 的节点上（每帧重算）
#[derive(Resource, Default, Debug, Clone, Copy, PartialEq)]
pub struct MouseIntercepted(pub bool);

/// Shift 是否被 UI 占用（滑杆拖动中按住 Shift = 精细调值，见 `widgets::slider`）。
/// 场景输入须忽略 Shift：否则同一个按键会把相机往下降。
#[derive(Resource, Default, Debug, Clone, Copy, PartialEq, Eq)]
pub struct UiShiftCaptured(pub bool);

/// 指针门控（每帧重算）。`set_if_neq` 仅在翻转时触发 change 检测。
pub fn ui_pointer_capture_system(
  hover_map: Res<HoverMap>,
  ui_nodes: Query<(), With<ComputedNode>>,
  intercepts: Query<&Hovered, With<MouseIntercept>>,
  mut captured: ResMut<UiPointerCaptured>,
  mut intercepted: ResMut<MouseIntercepted>,
) {
  let over_ui =
    hover_map.get(&PointerId::Mouse).is_some_and(|hits| hits.keys().any(|e| ui_nodes.contains(*e)));
  let over_intercept = intercepts.iter().any(|h| h.get());
  captured.set_if_neq(UiPointerCaptured(over_ui));
  intercepted.set_if_neq(MouseIntercepted(over_intercept));
}
