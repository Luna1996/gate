use bevy::picking::Pickable;
use bevy::picking::hover::Hovered;
use bevy::prelude::*;
use bevy::ui::{ComputedNode, UiGlobalTransform};
use bevy::window::PrimaryWindow;

use super::markdown::{MarkdownConfig, MarkdownView, markdown_root, markdown_set_text};
use super::{color_of, px};
use crate::theme::UiTheme;
use crate::widgets::consts::{TOOLTIP_DELAY, TOOLTIP_MARGIN, TOOLTIP_MAX_W, TOOLTIP_OFFSET};

const TOOLTIP_Z: i32 = 1000;

#[derive(Component, Clone, Debug, PartialEq, Eq)]
pub struct Tooltip {
  pub text: String,
}

impl Tooltip {
  pub fn new(text: impl Into<String>) -> Self {
    Self { text: text.into() }
  }
}

#[derive(Component, Debug)]
pub struct TooltipLayer;

#[derive(Component, Debug)]
pub struct TooltipText;

#[derive(Resource, Default)]
pub struct TooltipLayerEntity(pub Option<Entity>);

#[derive(Default)]
pub struct TooltipHoverState {
  entity: Option<Entity>,
  elapsed: f32,
  pin: Option<(Entity, Vec2)>,
}

#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn tooltip_system(
  mut commands: Commands,
  theme: Option<Res<UiTheme>>,
  windows: Query<&Window, With<PrimaryWindow>>,
  mut layer_res: ResMut<TooltipLayerEntity>,
  time: Res<Time>,
  mut state: Local<TooltipHoverState>,
  mut layer_q: Query<(&mut Node, Option<&ComputedNode>), With<TooltipLayer>>,
  view_q: Query<(Entity, &MarkdownView), With<TooltipText>>,
  owners: Query<(
    Entity,
    &Tooltip,
    Option<&Hovered>,
    &ComputedNode,
    &UiGlobalTransform,
    Option<&InheritedVisibility>,
  )>,
) {
  let Some(theme) = theme else { return };
  let Ok(window) = windows.single() else { return };
  let physical = window.physical_cursor_position();

  let mut best: Option<(Entity, &Tooltip)> = None;
  let mut best_area = f32::MAX;
  if let Some(cursor) = physical {
    for (e, tip, hover, node, transform, vis) in &owners {
      if !vis.is_some_and(|v| v.get()) || node.size() == Vec2::ZERO {
        continue;
      }
      let hovered = match hover {
        Some(h) => h.get(),
        None => node.contains_point(*transform, cursor),
      };
      if !hovered {
        continue;
      }
      let area = node.size().x * node.size().y;
      if area < best_area {
        best_area = area;
        best = Some((e, tip));
      }
    }
  }

  let Some((e, tip)) = best else {
    state.entity = None;
    state.elapsed = 0.0;
    state.pin = None;
    if let Ok((mut node, _)) = layer_q.single_mut() {
      node.display = Display::None;
    }
    return;
  };
  if state.entity != Some(e) {
    state.entity = Some(e);
    state.elapsed = 0.0;
  }
  state.elapsed += time.delta_secs();
  if state.elapsed < TOOLTIP_DELAY {
    return;
  }
  let text = tip.text.as_str();
  ensure_layer(&mut commands, &mut layer_res, &theme);
  if let Ok((e, view)) = view_q.single()
    && view.source() != text
  {
    markdown_set_text(&mut commands, e, text);
  }
  let sf = window.scale_factor().max(f32::EPSILON);
  let cursor = physical.unwrap_or_default() / sf;
  let anchor = match state.pin {
    Some((pinned, p)) if pinned == e => p,
    _ => {
      let p = cursor + TOOLTIP_OFFSET;
      state.pin = Some((e, p));
      p
    }
  };
  let (w, h) = (window.width(), window.height());
  let size =
    layer_q.single_mut().ok().and_then(|(_, c)| c.map(|n| n.size() / sf)).unwrap_or_default();
  let max_x = (w - size.x - TOOLTIP_MARGIN).max(TOOLTIP_MARGIN);
  let max_y = (h - size.y - TOOLTIP_MARGIN).max(TOOLTIP_MARGIN);
  if let Ok((mut node, _)) = layer_q.single_mut() {
    node.display = Display::Flex;
    node.left = px(anchor.x.min(max_x));
    node.top = px(anchor.y.min(max_y));
  }
}

fn ensure_layer(commands: &mut Commands, res: &mut TooltipLayerEntity, theme: &UiTheme) -> Entity {
  if let Some(e) = res.0
    && commands.get_entity(e).is_ok()
  {
    return e;
  }
  let c = &theme.colors;
  let m = &theme.metrics;
  let e = commands
    .spawn((
      Name::new("ui-tooltip"),
      TooltipLayer,
      Node {
        position_type: PositionType::Absolute,
        display: Display::None,
        max_width: px(TOOLTIP_MAX_W),
        padding: UiRect {
          left: px(m.spacing.sm),
          right: px(m.spacing.sm),
          top: px(m.spacing.xs),
          bottom: px(m.spacing.xs),
        },
        border: UiRect::all(px(m.border_width)),
        ..default()
      },
      BackgroundColor(color_of(&c.surface_overlay)),
      BorderColor::all(color_of(&c.border_strong)),
      Pickable::IGNORE,
      GlobalZIndex(TOOLTIP_Z),
    ))
    .with_children(|p| {
      p.spawn((
        Name::new("ui-tooltip-text"),
        TooltipText,
        markdown_root(theme, &MarkdownConfig { text: String::new(), dense: true, max_width: None }),
      ));
    })
    .id();
  res.0 = Some(e);
  e
}
