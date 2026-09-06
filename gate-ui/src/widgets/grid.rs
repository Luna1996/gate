//! grid：网格布局容器（bevy_ui 原生 CSS Grid + gap 填色格线）。
//!
//! 格线方案 = **gap 填色 trick**（等效 HTML table 的 `border-collapse: collapse`）：
//! 容器背景 = `border` 令牌色，`row_gap/column_gap = border_width`（1px），
//! cell 用不透明表面背景盖住容器底色，透出的部分即格线。
//! 相比"相邻 cell 各画 border"的约定：线宽恒 1px 不叠加、十字交叉天然连通
//! （就是同一块背景）、外框由容器 border 补齐且颜色一致。
//!
//! 为什么不用 splitter 拼网格：flex 的 gap 是轨道间空隙，线元素活在兄弟流里，
//! 无法得知并抵消自己在 gap 流中的位置，端点永远无法保证相接。
//! 网格线需求一律用本组件；独立分割线仍用 [`super::splitter`]。

use std::ops::Deref;

use bevy::prelude::*;

use super::{PanelSurface, UiCtx, color_of, px};

/// 网格根节点标记
#[derive(Component, Debug, Default)]
pub struct UiGrid;

/// 网格句柄（Deref 到根实体 Entity）
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

/// 网格配置（Default = 2 列等宽、行高 auto）
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GridConfig {
  /// 列数（等宽 1fr 轨道）
  pub columns: u16,
  /// 行高；None → auto（由内容撑开）
  pub row_height: Option<f32>,
}

impl Default for GridConfig {
  fn default() -> Self {
    Self { columns: 2, row_height: None }
  }
}

/// 网格容器：N 列等宽 + 1px 连通格线（gap 填色）。
/// 直接子项即 cell（每项占一格，按行自动流入）。
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
      // Node.border 只是布局内缩，可见描边必须配 BorderColor 组件（与 panel/table 一致）
      BorderColor::all(color_of(&c.border)),
    ))
    .id();
  GridHandle(e)
}

/// 网格 cell：不透明表面背景（遮住容器底色成格），无 border、带 sm 内边距。
/// 必须作为 [`grid`] 的直接子项；[`PanelSurface::Hud`] 半透明会露格线，按 Card 处理。
pub fn grid_cell(ctx: &UiCtx, parent: &mut ChildSpawner, surface: PanelSurface) -> Entity {
  let c = &ctx.theme.colors;
  let bg = match surface {
    PanelSurface::Card | PanelSurface::Hud => &c.surface_card,
    PanelSurface::Elevated => &c.surface_elevated,
  };
  parent
    .spawn((
      Name::new("ui-grid-cell"),
      Node {
        padding: UiRect::all(px(ctx.theme.metrics.spacing.sm)),
        ..default()
      },
      BackgroundColor(color_of(bg)),
    ))
    .id()
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::theme::default_theme;

  #[test]
  fn grid_spawn_structure() {
    let theme = default_theme();
    let ctx = UiCtx::new(&theme, None);
    let mut app = App::new();
    let root = app.world_mut().spawn_empty().id();

    let mut h = None;
    app.world_mut().entity_mut(root).with_children(|p| {
      h = Some(grid(&ctx, p, GridConfig { columns: 3, row_height: Some(28.0) }));
    });
    // row_height: None → 空（auto 行）
    let mut h2 = None;
    app.world_mut().entity_mut(root).with_children(|p| {
      h2 = Some(grid(&ctx, p, GridConfig::default()));
    });
    // columns 下限 1
    let mut h3 = None;
    app.world_mut().entity_mut(root).with_children(|p| {
      h3 = Some(grid(&ctx, p, GridConfig { columns: 0, ..default() }));
    });
    let e = h.unwrap();
    let w = app.world();

    let n = w.get::<Node>(*e).unwrap();
    assert_eq!(n.display, Display::Grid);
    assert_eq!(n.grid_template_columns.len(), 1, "single repeated track");
    assert_eq!(n.row_gap, px(theme.metrics.border_width), "gap = border_width");
    assert_eq!(n.column_gap, px(theme.metrics.border_width), "gap = border_width");
    assert_eq!(n.border, UiRect::all(px(theme.metrics.border_width)), "outer frame");
    assert_eq!(
      w.get::<BorderColor>(*e).unwrap(),
      &BorderColor::all(color_of(&theme.colors.border)),
      "outer frame stroke (Node.border alone renders nothing)"
    );
    assert_eq!(
      w.get::<BackgroundColor>(*e).unwrap().0,
      color_of(&theme.colors.border),
      "container bg = border token (bleeds through gaps)"
    );
    // 行高进隐式轨道
    assert_eq!(n.grid_auto_rows, vec![GridTrack::px(28.0)]);

    let n = w.get::<Node>(*h2.unwrap()).unwrap();
    assert!(n.grid_auto_rows.is_empty(), "no row_height means auto rows");
    let n = w.get::<Node>(*h3.unwrap()).unwrap();
    assert_eq!(n.grid_template_columns.len(), 1, "clamped to at least 1 column");
  }

  #[test]
  fn grid_cell_surfaces() {
    let theme = default_theme();
    let ctx = UiCtx::new(&theme, None);
    let mut app = App::new();
    let root = app.world_mut().spawn_empty().id();

    let mut cells = Vec::new();
    app.world_mut().entity_mut(root).with_children(|p| {
      cells.push(grid_cell(&ctx, p, PanelSurface::Card));
      cells.push(grid_cell(&ctx, p, PanelSurface::Elevated));
      cells.push(grid_cell(&ctx, p, PanelSurface::Hud));
    });
    let w = app.world();
    let n = w.get::<Node>(cells[0]).unwrap();
    assert_eq!(n.padding, UiRect::all(px(theme.metrics.spacing.sm)));
    assert_eq!(n.border, UiRect::DEFAULT, "cell carries no border");
    assert_eq!(
      w.get::<BackgroundColor>(cells[0]).unwrap().0,
      color_of(&theme.colors.surface_card),
      "Card cell"
    );
    assert_eq!(
      w.get::<BackgroundColor>(cells[1]).unwrap().0,
      color_of(&theme.colors.surface_elevated),
      "Elevated cell"
    );
    assert_eq!(
      w.get::<BackgroundColor>(cells[2]).unwrap().0,
      color_of(&theme.colors.surface_card),
      "Hud falls back to Card (cell must be opaque)"
    );
  }
}
