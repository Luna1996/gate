use bevy::picking::hover::{HoverMap, Hovered};
use bevy::picking::pointer::PointerId;
use bevy::prelude::*;
use bevy::ui::ComputedNode;

#[derive(Resource, Default, Debug, Clone, Copy, PartialEq)]
pub struct UiPointerCaptured(pub bool);

#[derive(Component, Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct MouseIntercept;

#[derive(Resource, Default, Debug, Clone, Copy, PartialEq)]
pub struct MouseIntercepted(pub bool);

#[derive(Resource, Default, Debug, Clone, Copy, PartialEq, Eq)]
pub struct UiShiftCaptured(pub bool);

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
