//! panel：面板容器（暗色表面令牌）。
//! 层级靠表面色 + 1px 边框表达，无阴影；表面档由 `PanelSurface` 选定（Card/Hud/Elevated）。
//! 定位由调用方设置（PositionType::Absolute + Percent 边距锚边），本函数只给视觉令牌与 flex 纵向布局。

use std::ops::Deref;

use bevy::prelude::*;
use bevy::ui::FocusPolicy;

use super::{UiCtx, color_of, px};

/// 面板表面档（决定背景与边框令牌）
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
