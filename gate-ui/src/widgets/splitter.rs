use bevy::prelude::*;

use super::{UiCtx, color_of, px};

#[derive(Component, Debug, Default)]
pub struct Splitter;

pub fn splitter(ctx: &UiCtx, parent: &mut ChildSpawner) -> Entity {
  let horizontal = parent.world().get::<Node>(parent.target_entity()).is_none_or(|n| {
    matches!(n.flex_direction, FlexDirection::Column | FlexDirection::ColumnReverse)
  });
  let (width, height, name) = if horizontal {
    (Val::Auto, px(1.0), "ui-splitter-h")
  } else {
    (px(1.0), Val::Auto, "ui-splitter-v")
  };
  let c = &ctx.theme.colors;
  parent
    .spawn((
      Name::new(name),
      Splitter,
      Node { width, height, align_self: AlignSelf::Stretch, flex_shrink: 0.0, ..default() },
      BackgroundColor(color_of(&c.border)),
    ))
    .id()
}
