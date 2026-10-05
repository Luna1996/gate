use std::ops::Deref;

use bevy::prelude::*;

use super::{UiCtx, color_of, label_bundle, px};

#[derive(Component, Debug, Default)]
pub struct UiTable;

#[derive(Component, Debug, Default)]
pub struct TableCell;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TableHandle(pub Entity);

impl Deref for TableHandle {
  type Target = Entity;
  fn deref(&self) -> &Entity {
    &self.0
  }
}

impl From<TableHandle> for Entity {
  fn from(h: TableHandle) -> Entity {
    h.0
  }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TableConfig {
  pub headers: Vec<String>,
  pub rows: Vec<Vec<String>>,
}

pub fn table(ctx: &UiCtx, parent: &mut ChildSpawner, config: TableConfig) -> TableHandle {
  let c = &ctx.theme.colors;
  let m = &ctx.theme.metrics;
  let ncols = config.headers.len().max(1);
  let e = parent
    .spawn((
      Name::new("ui-table"),
      UiTable,
      Node {
        flex_direction: FlexDirection::Column,
        width: Val::Percent(100.0),
        border: UiRect::all(px(m.border_width)),
        ..default()
      },
      BorderColor::all(color_of(&c.border)),
    ))
    .with_children(|root| {
      root
        .spawn((
          Name::new("ui-table-header"),
          Node {
            flex_direction: FlexDirection::Row,
            padding: UiRect::all(px(m.spacing.sm)),
            border: UiRect::bottom(px(m.border_width)),
            ..default()
          },
          BackgroundColor(color_of(&c.surface_elevated)),
          BorderColor::all(color_of(&c.border)),
        ))
        .with_children(|row| {
          for h in &config.headers {
            row.spawn((
              TableCell,
              Node { flex_grow: 1.0, flex_basis: Val::Px(0.0), ..default() },
              label_bundle(ctx, h.clone(), m.font_size.sm, color_of(&c.text_primary)),
            ));
          }
        });
      for (i, row_data) in config.rows.iter().enumerate() {
        let bg = if i % 2 == 0 { color_of(&c.surface_card) } else { color_of(&c.surface_base) };
        root
          .spawn((
            Name::new("ui-table-row"),
            Node {
              flex_direction: FlexDirection::Row,
              padding: UiRect::all(px(m.spacing.sm)),
              border: UiRect::bottom(px(m.border_width)),
              ..default()
            },
            BackgroundColor(bg),
            BorderColor::all(color_of(&c.border_subtle)),
          ))
          .with_children(|row| {
            for cell in row_data.iter().take(ncols) {
              row.spawn((
                TableCell,
                Node { flex_grow: 1.0, flex_basis: Val::Px(0.0), ..default() },
                label_bundle(ctx, cell.clone(), m.font_size.sm, color_of(&c.text_body)),
              ));
            }
            for _ in row_data.len()..ncols {
              row.spawn((
                TableCell,
                Node { flex_grow: 1.0, flex_basis: Val::Px(0.0), ..default() },
                label_bundle(ctx, String::new(), m.font_size.sm, color_of(&c.text_body)),
              ));
            }
          });
      }
    })
    .id();
  TableHandle(e)
}
