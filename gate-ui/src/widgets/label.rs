//! label：主题字号/颜色的文本标签（docs/ui-dark-theme.md §5.6）。
//!
//! 变体由 [`LabelStyle`] preset enum 表达：字号 + 颜色永远成对来自主题令牌，
//! 不可能拼出非法组合（如 faint 档 <18px 违反对比度规则）。
//! 禁止纯白：一律走主题令牌。

use std::ops::Deref;

use bevy::prelude::*;

use super::{UiCtx, color_of, spawn_label};

/// 文本预设档：四档正文（primary/body/muted/faint）+ 四档语义色
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum LabelStyle {
  /// 正文（text_body，字号 md）
  #[default]
  Body,
  /// 次要/说明（text_muted，字号 sm）
  Muted,
  /// 面板标题 / KPI 读数（text_primary，字号 lg）
  Title,
  /// 大号弱化（text_faint，字号 lg；faint 档对比度只允许 ≥18px）
  FaintLg,
  /// 语义色：强调文本（accent_text，字号 sm）
  Accent,
  /// 语义色：成功（success，字号 sm）
  Success,
  /// 语义色：警告（warning，字号 sm）
  Warning,
  /// 语义色：危险（danger，字号 sm）
  Danger,
}

impl LabelStyle {
  /// (字号档, 颜色令牌)
  fn tokens(
    self,
  ) -> (
    fn(&crate::theme::FontSize) -> f32,
    fn(&crate::theme::ThemeColors) -> &crate::theme::HexColor,
  ) {
    match self {
      Self::Body => (|fs| fs.md, |c| &c.text_body),
      Self::Muted => (|fs| fs.sm, |c| &c.text_muted),
      Self::Title => (|fs| fs.lg, |c| &c.text_primary),
      Self::FaintLg => (|fs| fs.lg, |c| &c.text_faint),
      Self::Accent => (|fs| fs.sm, |c| &c.accent_text),
      Self::Success => (|fs| fs.sm, |c| &c.success),
      Self::Warning => (|fs| fs.sm, |c| &c.warning),
      Self::Danger => (|fs| fs.sm, |c| &c.danger),
    }
  }
}

/// 标签句柄（Deref 到文本实体 Entity）
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LabelHandle(pub Entity);

impl Deref for LabelHandle {
  type Target = Entity;
  fn deref(&self) -> &Entity {
    &self.0
  }
}

impl From<LabelHandle> for Entity {
  fn from(h: LabelHandle) -> Entity {
    h.0
  }
}

/// 标签配置（全部字段进 Config；Default = 空文本 + Body 档）
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LabelConfig {
  pub text: String,
  pub style: LabelStyle,
}

/// 主题文本标签
pub fn label(ctx: &UiCtx, parent: &mut ChildSpawner, config: LabelConfig) -> LabelHandle {
  let fs = &ctx.theme.metrics.font_size;
  let (size, color) = config.style.tokens();
  let e = spawn_label(ctx, parent, config.text, size(fs), color_of(&color(&ctx.theme.colors)));
  LabelHandle(e)
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::theme::default_theme;

  #[test]
  fn label_spawn_structure() {
    let theme = default_theme();
    let ctx = UiCtx::new(&theme, None);
    let mut app = App::new();
    let root = app.world_mut().spawn_empty().id();
    let mut child = None;
    app.world_mut().entity_mut(root).with_children(|p| {
      child = Some(label(&ctx, p, LabelConfig { text: "hello".into(), ..default() }));
    });
    let e = *child.expect("label spawned");
    let w = app.world();
    let text = w.get::<Text>(e).expect("label has Text");
    assert_eq!(text.0.as_str(), "hello");
    assert!(w.get::<bevy::ui::widget::Label>(e).is_some());
    let tf = w.get::<TextFont>(e).expect("label has TextFont");
    assert_eq!(
      tf.font_size,
      bevy::text::FontSize::Px(theme.metrics.font_size.md)
    );
    assert_eq!(
      w.get::<TextColor>(e).unwrap().0,
      color_of(&theme.colors.text_body),
      "Body style uses text_body token"
    );
  }

  #[test]
  fn label_styles_map_to_token_pairs() {
    let theme = default_theme();
    let ctx = UiCtx::new(&theme, None);
    let mut app = App::new();
    let root = app.world_mut().spawn_empty().id();
    let mut got = Vec::new();
    app.world_mut().entity_mut(root).with_children(|p| {
      got.push(*label(&ctx, p, LabelConfig { text: "t".into(), style: LabelStyle::Title }));
      got.push(*label(&ctx, p, LabelConfig { text: "m".into(), style: LabelStyle::Muted }));
      got.push(*label(&ctx, p, LabelConfig { text: "f".into(), style: LabelStyle::FaintLg }));
      got.push(*label(&ctx, p, LabelConfig { text: "s".into(), style: LabelStyle::Success }));
    });
    let w = app.world();
    assert_eq!(
      w.get::<TextColor>(got[0]).unwrap().0,
      color_of(&theme.colors.text_primary)
    );
    assert_eq!(
      w.get::<TextFont>(got[0]).unwrap().font_size,
      bevy::text::FontSize::Px(theme.metrics.font_size.lg)
    );
    assert_eq!(
      w.get::<TextColor>(got[1]).unwrap().0,
      color_of(&theme.colors.text_muted)
    );
    assert_eq!(
      w.get::<TextFont>(got[2]).unwrap().font_size,
      bevy::text::FontSize::Px(theme.metrics.font_size.lg),
      "FaintLg stays >=18px"
    );
    assert_eq!(
      w.get::<TextColor>(got[3]).unwrap().0,
      color_of(&theme.colors.success)
    );
  }
}
