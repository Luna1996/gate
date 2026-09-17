//! grid：网格布局容器（bevy_ui 原生 CSS Grid + gap 填色格线）。
//! 格线方案：容器背景 = `border` 令牌色，`row_gap/column_gap = border_width`（1px），cell 用不透明
//! 表面遮住底色，透出部分即格线（恒 1px、十字连通，外框由容器 border 补齐）；独立分割线用 `splitter`。

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
      // Node.border 仅布局内缩，可见描边须配 BorderColor 组件（与 panel/table 一致）
      BorderColor::all(color_of(&c.border)),
    ))
    .id();
  GridHandle(e)
}

/// 网格 cell：不透明表面背景（遮住容器底色成格），无 border、带 sm 内边距。必须作为 `grid` 的
/// 直接子项；cell 须不透明，故 `PanelSurface::Hud` 按 Card 处理。
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
