//! label：主题字号/颜色的文本标签。

use bevy::prelude::*;

use super::{UiCtx, color_of, spawn_label};

/// 文本标签（字号 = theme.metrics.font_size.md，颜色 = theme.colors.text）
pub fn label(ctx: &UiCtx, parent: &mut ChildSpawner, text: impl Into<String>) -> Entity {
  let fs = &ctx.theme.metrics.font_size;
  spawn_label(
    ctx,
    parent,
    text.into(),
    fs.md,
    color_of(&ctx.theme.colors.text),
  )
}

/// 次要文本（text_muted，字号 sm）
pub fn label_muted(ctx: &UiCtx, parent: &mut ChildSpawner, text: impl Into<String>) -> Entity {
  let fs = &ctx.theme.metrics.font_size;
  spawn_label(
    ctx,
    parent,
    text.into(),
    fs.sm,
    color_of(&ctx.theme.colors.text_muted),
  )
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
      child = Some(label(&ctx, p, "hello"));
    });
    let e = child.expect("label spawned");
    let w = app.world();
    let text = w.get::<Text>(e).expect("label has Text");
    assert_eq!(text.0.as_str(), "hello");
    assert!(w.get::<bevy::ui::widget::Label>(e).is_some());
    let tf = w.get::<TextFont>(e).expect("label has TextFont");
    assert_eq!(
      tf.font_size,
      bevy::text::FontSize::Px(theme.metrics.font_size.md)
    );
  }
}
