use std::ops::Deref;

use bevy::ecs::message::MessageReader;
use bevy::input::mouse::{MouseScrollUnit, MouseWheel};
use bevy::picking::hover::Hovered;
use bevy::prelude::*;
use bevy::ui::{ComputedNode, Overflow, Pressed};

use super::{UiCtx, color_of, px};
use crate::pointer::{UiInteract, UiInteractBundle};
use crate::widgets::consts::SCROLL_SPEED;

#[derive(Component, Debug, Default)]
pub struct ScrollView {
  pub scroll_y: f32,
}

#[derive(Component, Debug, Default)]
pub struct ScrollViewport;

#[derive(Component, Debug, Default)]
pub struct ScrollContent;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ScrollViewHandle {
  pub entity: Entity,
  pub content: Entity,
}

impl Deref for ScrollViewHandle {
  type Target = Entity;
  fn deref(&self) -> &Entity {
    &self.entity
  }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScrollConfig {
  pub height: Val,
}

impl Default for ScrollConfig {
  fn default() -> Self {
    Self { height: Val::Auto }
  }
}

pub fn scroll_view(
  ctx: &UiCtx,
  parent: &mut ChildSpawner,
  config: ScrollConfig,
) -> ScrollViewHandle {
  let c = &ctx.theme.colors;
  let m = &ctx.theme.metrics;
  let mut content_e = None;
  let vp_e = parent
    .spawn((
      Name::new("ui-scroll-view"),
      ScrollView::default(),
      ScrollViewport,
      UiInteractBundle::default(),
      Node {
        width: Val::Percent(100.0),
        height: config.height,
        overflow: Overflow::clip(),
        border: UiRect::all(px(m.border_width)),
        ..default()
      },
      BackgroundColor(Color::NONE),
      BorderColor::all(color_of(&c.border)),
    ))
    .with_children(|vp| {
      content_e = Some(
        vp.spawn((
          Name::new("ui-scroll-content"),
          ScrollContent,
          Node {
            position_type: PositionType::Absolute,
            top: Val::Px(0.0),
            left: Val::Px(0.0),
            right: Val::Px(0.0),
            flex_direction: FlexDirection::Column,
            row_gap: px(m.spacing.xs),
            padding: UiRect::all(px(m.spacing.sm)),
            ..default()
          },
        ))
        .id(),
      );
    })
    .id();
  ScrollViewHandle { entity: vp_e, content: content_e.expect("scroll content spawned") }
}

#[allow(clippy::type_complexity)]
pub fn scroll_view_system(
  mut scroll_reader: MessageReader<MouseWheel>,
  mut q_vp: Query<
    (&mut ScrollView, &Hovered, Has<Pressed>, &Children, &ComputedNode),
    With<ScrollViewport>,
  >,
  mut q_content: Query<(&mut Node, &ComputedNode), With<ScrollContent>>,
) {
  let mut lines = 0.0;
  for ev in scroll_reader.read() {
    match ev.unit {
      MouseScrollUnit::Line => lines += ev.y,
      MouseScrollUnit::Pixel => lines += ev.y / 16.0,
    }
  }
  if lines == 0.0 {
    return;
  }
  let delta = lines * SCROLL_SPEED;
  for (mut sv, hovered, pressed, children, vp_computed) in &mut q_vp {
    let inter = UiInteract::of(hovered, pressed);
    if !inter.is_active() {
      continue;
    }
    let vp_h = vp_computed.size.y;
    if vp_h <= 0.0 {
      continue;
    }
    for child in children.iter() {
      let Ok((mut content_node, content_computed)) = q_content.get_mut(child) else {
        continue;
      };
      let content_h = content_computed.size.y;
      sv.scroll_y += delta;
      let max_scroll = (content_h - vp_h).max(0.0);
      sv.scroll_y = sv.scroll_y.clamp(-max_scroll, 0.0);
      content_node.top = Val::Px(sv.scroll_y);
    }
  }
}
