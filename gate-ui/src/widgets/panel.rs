use std::ops::Deref;

use bevy::picking::Pickable;
use bevy::prelude::*;

use super::{UiCtx, color_of, px};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PanelSurface {
  #[default]
  Card,
  Hud,
  Elevated,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PanelHandle(pub Entity);

impl Deref for PanelHandle {
  type Target = Entity;
  fn deref(&self) -> &Entity {
    &self.0
  }
}

impl From<PanelHandle> for Entity {
  fn from(h: PanelHandle) -> Entity {
    h.0
  }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct PanelConfig {
  pub surface: PanelSurface,
}

pub fn panel(ctx: &UiCtx, parent: &mut ChildSpawner, config: PanelConfig) -> PanelHandle {
  let c = &ctx.theme.colors;
  let (bg, border) = match config.surface {
    PanelSurface::Card => (color_of(&c.surface_card), color_of(&c.border)),
    PanelSurface::Hud => (color_of(&c.surface_card_hud), color_of(&c.border)),
    PanelSurface::Elevated => (color_of(&c.surface_elevated), color_of(&c.border_subtle)),
  };
  let m = &ctx.theme.metrics;
  let e = parent
    .spawn((
      Name::new("ui-panel"),
      Node {
        flex_direction: FlexDirection::Column,
        row_gap: px(m.spacing.sm),
        padding: UiRect {
          left: px(m.spacing.lg),
          right: px(m.spacing.lg),
          top: px(m.spacing.md),
          bottom: px(m.spacing.md),
        },
        border: UiRect::all(px(m.border_width)),
        ..default()
      },
      BackgroundColor(bg),
      BorderColor::all(border),
      Pickable::default(),
    ))
    .id();
  PanelHandle(e)
}
