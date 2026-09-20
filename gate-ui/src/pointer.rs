//! pointer：gate-ui 的指针交互基座。
//! bevy 0.20 起 `Interaction` 已变成 crate 私有的废弃别名（外部无法命名），指针对交互的事实来源改为 picking：
//! - **悬停** = [`Hovered`]（picking 在 `PreUpdate` 维护；对后代也成立 ⇒ CSS `:hover` 语义，
//!   所以装饰性子节点（标签/图标/滑块 thumb）挡住命中也不会让控件根丢悬停态）；
//! - **按住** = [`Pressed`]（本模块的观察者在 `PointerPress` / `PointerRelease` / `PointerDragEnd` /
//!   `PointerCancel` 上维护；事件沿 `ChildOf` 冒泡 ⇒ 命中装饰子节点时也能落到控件根）。
//!
//! 控件侧统一用 [`UiInteract::of`] 把两个来源合成一个三态值（形状与旧 `Interaction` 一致），
//! 再配 [`UiInteractPrev`] 做「上一帧按住 + 本帧在节点上释放 = 点击」的判定。
//!
//! **谁能被命中**：GateUiPlugin 把 `UiPickingSettings::require_markers` 打开 ⇒ UI 拾取需要两侧都显式标记：
//! **相机**挂 `UiPickingCamera`（渲染 UI 的那个相机，见 gate-app 的 `scene::setup`）、**节点**挂 `Pickable`。
//! 其余节点（标签/图标/结构容器等装饰节点）对 picking 不可见 —— 这正是旧模型里
//! 「只有 `FocusPolicy::Block` 的节点才拦命中」的等价物：交互控件根靠 [`UiInteractBundle`] 自动带上
//! `Pickable`，非交互但要拦命中的节点（如面板表面）显式挂它；压在其他控件之上的浮层
//! （tooltip 层 / 下拉浮层容器 / 禁用行）挂 `Pickable::IGNORE` = 不拦也不响应（旧 `Pass`）。

use bevy::picking::Pickable;
use bevy::picking::events::{PointerCancel, PointerDragEnd, PointerPress, PointerRelease};
use bevy::picking::hover::Hovered;
use bevy::picking::pointer::PointerButton;
use bevy::prelude::*;
use bevy::ui::Pressed;

/// 控件侧的交互态（只读派生值，不是组件；由 [`UiInteract::of`] 逐帧合成）
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum UiInteract {
  #[default]
  None,
  Hovered,
  Pressed,
}

impl UiInteract {
  /// 由 picking 的两个事实来源合成；按住优先于悬停（拖出控件仍算按住）。
  pub fn of(hovered: &Hovered, pressed: bool) -> Self {
    if pressed {
      Self::Pressed
    } else if hovered.get() {
      Self::Hovered
    } else {
      Self::None
    }
  }

  /// 悬停或按住（等价旧 `inter != Interaction::None`）
  pub fn is_active(self) -> bool {
    self != Self::None
  }
}

/// 交互控件根标记：`PointerPress` 冒泡链的终点，只有挂它的节点会拿到 `Pressed`。
#[derive(Component, Clone, Copy, Debug, Default)]
pub struct UiInteractable;

/// 上一帧交互态（click = 上一帧 `Pressed` + 本帧 `Hovered`，即「按下后在同一个节点上释放」）
#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UiInteractPrev(pub UiInteract);

/// 交互控件根三件套：标记 + picking 悬停态 + 上一帧态 + 命中标记（拾取后端只认挂了 `Pickable` 的节点）。
#[derive(Bundle, Clone, Copy, Debug, Default)]
pub struct UiInteractBundle {
  pub interactable: UiInteractable,
  pub hovered: Hovered,
  pub prev: UiInteractPrev,
  pub pickable: Pickable,
}

/// 左键按下 → 控件根挂 `Pressed`；未命中控件根则继续冒泡给祖先控件（内层控件优先）。
pub(crate) fn ui_pointer_press(
  mut ev: On<PointerPress>,
  mut commands: Commands,
  q: Query<Has<Pressed>, With<UiInteractable>>,
) {
  if ev.button != PointerButton::Primary {
    return;
  }
  if let Ok(pressed) = q.get(ev.entity) {
    ev.propagate(false);
    if !pressed {
      commands.entity(ev.entity).insert(Pressed);
    }
  }
}

/// 左键释放 → 摘掉控件根的 `Pressed`（`query` 未命中 = 该节点不是控件根，不拦冒泡）。
pub(crate) fn ui_pointer_release(
  mut ev: On<PointerRelease>,
  mut commands: Commands,
  q: Query<(), With<UiInteractable>>,
) {
  if ev.button != PointerButton::Primary {
    return;
  }
  if q.contains(ev.entity) {
    ev.propagate(false);
    commands.entity(ev.entity).remove::<Pressed>();
  }
}

/// 拖拽结束（按下后移动过，释放点在别处）→ 同样摘掉 `Pressed`
pub(crate) fn ui_pointer_drag_end(
  mut ev: On<PointerDragEnd>,
  mut commands: Commands,
  q: Query<(), With<UiInteractable>>,
) {
  if ev.button != PointerButton::Primary {
    return;
  }
  if q.contains(ev.entity) {
    ev.propagate(false);
    commands.entity(ev.entity).remove::<Pressed>();
  }
}

/// 指针被取消（窗口失焦 / 指针消失）→ 摘掉 `Pressed`，避免卡在按住态
pub(crate) fn ui_pointer_cancel(
  mut ev: On<PointerCancel>,
  mut commands: Commands,
  q: Query<(), With<UiInteractable>>,
) {
  if q.contains(ev.entity) {
    ev.propagate(false);
    commands.entity(ev.entity).remove::<Pressed>();
  }
}
