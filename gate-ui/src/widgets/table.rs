//! table：简易数据表格（表头 + 斑马纹行，每列 flex_grow 等宽）。
//!
//! 无排序/选择/虚拟滚动；展示用。行高靠 padding，列宽靠 flex_grow:1 均分。

use std::ops::Deref;

use bevy::prelude::*;

use super::{UiCtx, color_of, label_bundle, px};

/// 表格根节点标记
#[derive(Component, Debug, Default)]
pub struct UiTable;

/// 表格单元标记
#[derive(Component, Debug, Default)]
pub struct TableCell;

/// 表格句柄（Deref 到根实体 Entity）
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

/// 表格配置（全部字段进 Config；Default = 空表）
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TableConfig {
  /// 表头列名
  pub headers: Vec<String>,
  /// 数据行（每行列数不足表头时自动补空单元格）
  pub rows: Vec<Vec<String>>,
}

/// 创建表格。`headers` 为表头列名，`rows` 为数据行（每行列数应与 headers 一致）。
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
      // ---- 表头行 ----
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
              Node {
                flex_grow: 1.0,
                flex_basis: Val::Px(0.0),
                ..default()
              },
              label_bundle(ctx, h.clone(), m.font_size.sm, color_of(&c.text_primary)),
            ));
          }
        });
      // ---- 数据行（斑马纹）----
      for (i, row_data) in config.rows.iter().enumerate() {
        let bg = if i % 2 == 0 {
          color_of(&c.surface_card)
        } else {
          color_of(&c.surface_base)
        };
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
                Node {
                  flex_grow: 1.0,
                  flex_basis: Val::Px(0.0),
                  ..default()
                },
                label_bundle(ctx, cell.clone(), m.font_size.sm, color_of(&c.text_body)),
              ));
            }
            // 列数不足时填空单元格，保持列对齐
            for _ in row_data.len()..ncols {
              row.spawn((
                TableCell,
                Node {
                  flex_grow: 1.0,
                  flex_basis: Val::Px(0.0),
                  ..default()
                },
                label_bundle(ctx, String::new(), m.font_size.sm, color_of(&c.text_body)),
              ));
            }
          });
      }
    })
    .id();
  TableHandle(e)
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::theme::default_theme;

  fn s(v: &[&str]) -> Vec<String> {
    v.iter().map(|x| x.to_string()).collect()
  }

  #[test]
  fn table_spawn_structure() {
    let theme = default_theme();
    let ctx = UiCtx::new(&theme, None);
    let mut app = App::new();
    let root = app.world_mut().spawn_empty().id();
    let mut tbl = None;
    app.world_mut().entity_mut(root).with_children(|p| {
      tbl = Some(table(
        &ctx,
        p,
        TableConfig {
          headers: s(&["name", "value"]),
          rows: vec![s(&["a", "1"]), s(&["b", "2"])],
        },
      ));
    });
    let e = *tbl.expect("table spawned");
    let w = app.world();
    assert!(w.get::<UiTable>(e).is_some());
    let children = w.get::<Children>(e).unwrap();
    assert_eq!(children.len(), 3, "header + 2 data rows");
    // 表头有 2 个单元格
    let header = w.get::<Children>(children[0]).unwrap();
    assert_eq!(header.len(), 2);
  }

  #[test]
  fn table_pads_short_rows() {
    let theme = default_theme();
    let ctx = UiCtx::new(&theme, None);
    let mut app = App::new();
    let root = app.world_mut().spawn_empty().id();
    let mut tbl = None;
    app.world_mut().entity_mut(root).with_children(|p| {
      tbl = Some(table(
        &ctx,
        p,
        TableConfig {
          headers: s(&["a", "b", "c"]),
          rows: vec![s(&["x"])],
        },
      ));
    });
    let e = *tbl.unwrap();
    let w = app.world();
    let children = w.get::<Children>(e).unwrap();
    let row = w.get::<Children>(children[1]).unwrap();
    assert_eq!(row.len(), 3, "short row padded to header column count");
  }
}
