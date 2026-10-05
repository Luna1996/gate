use std::ops::Deref;

use bevy::prelude::*;

use super::{PanelSurface, UiCtx, color_of, px};

#[derive(Component, Debug, Default)]
pub struct UiGrid;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GridHandle(pub Entity);

impl Deref for GridHandle {
  type Target = Entity;
  fn deref(&self) -> &Entity {
    &self.0
  }
}

impl From<GridHandle> for Entity {
  fn from(h: GridHandle) -> Entity {
    h.0
  }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GridConfig {
  pub columns: u16,
  pub row_height: Option<f32>,
}

impl Default for GridConfig {
  fn default() -> Self {
    Self { columns: 2, row_height: None }
  }
}

pub fn grid(ctx: &UiCtx, parent: &mut ChildSpawner, config: GridConfig) -> GridHandle {
  let c = &ctx.theme.colors;
  let m = &ctx.theme.metrics;
  let columns = config.columns.max(1);
  let e = parent
    .spawn((
      Name::new("ui-grid"),
      UiGrid,
      Node {
        display: Display::Grid,
        grid_template_columns: vec![RepeatedGridTrack::fr(columns, 1.0)],
        grid_auto_rows: config.row_height.map(|h| vec![GridTrack::px(h)]).unwrap_or_default(),
        row_gap: px(m.border_width),
        column_gap: px(m.border_width),
        width: Val::Percent(100.0),
        border: UiRect::all(px(m.border_width)),
        ..default()
      },
      BackgroundColor(color_of(&c.border)),
      BorderColor::all(color_of(&c.border)),
    ))
    .id();
  GridHandle(e)
}

pub fn grid_cell(ctx: &UiCtx, parent: &mut ChildSpawner, surface: PanelSurface) -> Entity {
  let c = &ctx.theme.colors;
  let bg = match surface {
    PanelSurface::Card | PanelSurface::Hud => &c.surface_card,
    PanelSurface::Elevated => &c.surface_elevated,
  };
  parent
    .spawn((
      Name::new("ui-grid-cell"),
      Node { padding: UiRect::all(px(ctx.theme.metrics.spacing.sm)), ..default() },
      BackgroundColor(color_of(bg)),
    ))
    .id()
}
