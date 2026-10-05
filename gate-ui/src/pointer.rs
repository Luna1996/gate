use bevy::picking::Pickable;
use bevy::picking::events::{PointerCancel, PointerDragEnd, PointerPress, PointerRelease};
use bevy::picking::hover::Hovered;
use bevy::picking::pointer::PointerButton;
use bevy::prelude::*;
use bevy::ui::Pressed;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum UiInteract {
  #[default]
  None,
  Hovered,
  Pressed,
}

impl UiInteract {
  pub fn of(hovered: &Hovered, pressed: bool) -> Self {
    if pressed {
      Self::Pressed
    } else if hovered.get() {
      Self::Hovered
    } else {
      Self::None
    }
  }

  pub fn is_active(self) -> bool {
    self != Self::None
  }
}

#[derive(Component, Clone, Copy, Debug, Default)]
pub struct UiInteractable;

#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UiInteractPrev(pub UiInteract);

#[derive(Bundle, Clone, Copy, Debug, Default)]
pub struct UiInteractBundle {
  pub interactable: UiInteractable,
  pub hovered: Hovered,
  pub prev: UiInteractPrev,
  pub pickable: Pickable,
}

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
