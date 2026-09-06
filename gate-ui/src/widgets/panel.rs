//! panel：圆角面板容器（暗色表面令牌，docs/ui-dark-theme.md §5.1）。
//!
//! 层级靠表面色与 1px 边框表达，不用阴影：
//! - [`panel`]：L1 不透明卡片（菜单/设置/模态本体）
//! - [`panel_hud`]：L1 半透明卡片（浮在 3D 体素场景上的 HUD/调试 overlay，alpha 90%）
//! - [`panel_elevated`]：L2 抬升嵌块（卡片内分区）
//!
//! 响应式约定：面板定位由调用方设置（PositionType::Absolute + Percent 边距锚定边缘），
//! 本函数只给视觉令牌与 flex 纵向布局。

use std::ops::Deref;

use bevy::prelude::*;
use bevy::ui::FocusPolicy;

use super::{UiCtx, color_of, px};

/// 面板表面档（层级靠表面色与 1px 边框表达，不用阴影）
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PanelSurface {
  /// L1 不透明卡片（菜单/设置/模态本体）
  #[default]
  Card,
  /// L1 HUD 卡片（浮在 3D 体素场景上的 HUD/调试 overlay）
  Hud,
  /// L2 抬升嵌块（卡片内分区）
  Elevated,
}

/// 面板句柄（Deref 到根实体 Entity）
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

/// 面板配置（Default = Card 档）
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct PanelConfig {
  pub surface: PanelSurface,
}

/// 主题面板容器（flex 纵向布局；定位由调用方设置）
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
      FocusPolicy::Block,
    ))
    .id();
  PanelHandle(e)
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
      child = Some(panel(&ctx, p, PanelConfig::default()));
    });
    let e = *child.expect("panel spawned");
    let w = app.world();
    assert!(w.get::<Node>(e).is_some(), "panel has Node");
    assert!(w.get::<BackgroundColor>(e).is_some(), "panel has bg");
    assert!(w.get::<FocusPolicy>(e).is_some(), "panel blocks focus");
    assert_eq!(
      w.get::<BackgroundColor>(e).unwrap().0,
      color_of(&theme.colors.surface_card),
      "default surface = Card (L1 surface token)"
    );
  }

  #[test]
  fn hud_panel_is_opaque() {
    let theme = default_theme();
    let ctx = UiCtx::new(&theme, None);
    let mut app = App::new();
    let root = app.world_mut().spawn_empty().id();
    let mut child = None;
    app.world_mut().entity_mut(root).with_children(|p| {
      child = Some(panel(&ctx, p, PanelConfig { surface: PanelSurface::Hud }));
    });
    let e = *child.expect("hud panel spawned");
    let bg = app
      .world()
      .get::<BackgroundColor>(e)
      .unwrap()
      .0
      .to_srgba();
    assert!(bg.alpha > 0.99, "hud panel is opaque (no translucency)");
  }
}
