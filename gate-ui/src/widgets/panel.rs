//! panel：半透明圆角面板容器（Modern UI 风格基座）。
//!
//! 响应式约定：面板定位由调用方设置（PositionType::Absolute + Percent 边距锚定边缘），
//! 本函数只给视觉令牌与 flex 纵向布局。

use bevy::prelude::*;
use bevy::ui::FocusPolicy;

use super::{UiCtx, color_of, px};

/// 面板容器（视觉 token 全部来自主题）
pub fn panel(ctx: &UiCtx, parent: &mut ChildSpawner) -> Entity {
  let c = &ctx.theme.colors;
  let m = &ctx.theme.metrics;
  parent
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
        border_radius: BorderRadius::all(px(m.corner_radius)),
        ..default()
      },
      BackgroundColor(color_of(&c.panel_bg)),
      BorderColor::all(color_of(&c.panel_border)),
      FocusPolicy::Block,
    ))
    .id()
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::theme::default_theme;

  #[test]
  fn panel_spawn_structure() {
    let theme = default_theme();
    let ctx = UiCtx::new(&theme, None);
    let mut app = App::new();
    let root = app.world_mut().spawn_empty().id();
    let mut child = None;
    app.world_mut().entity_mut(root).with_children(|p| {
      child = Some(panel(&ctx, p));
    });
    let e = child.expect("panel spawned");
    let w = app.world();
    assert!(w.get::<Node>(e).is_some(), "panel has Node");
    assert!(w.get::<BackgroundColor>(e).is_some(), "panel has bg");
    assert!(w.get::<FocusPolicy>(e).is_some(), "panel blocks focus");
  }
}
