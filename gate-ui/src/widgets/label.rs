use std::ops::Deref;

use bevy::prelude::*;
use bevy::text::TextLayoutInfo;
use bevy::text::{FontFeatures, FontSmoothing, FontStyle, FontVariations, FontWeight, FontWidth};
use bevy::ui::Overflow;

use super::{UiCtx, color_of, dim_color};
use crate::widgets::consts::ELLIPSIS;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum LabelStyle {
  #[default]
  Body,
  Muted,
  Title,
  FaintLg,
  Accent,
  Success,
  Warning,
  Danger,
}

type SizeFn = fn(&crate::theme::FontSize) -> f32;
type ColorFn = fn(&crate::theme::ThemeColors) -> &crate::theme::HexColor;

impl LabelStyle {
  fn tokens(self) -> (SizeFn, ColorFn) {
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LabelHandle(pub Entity);

impl Deref for LabelHandle {
  type Target = Entity;
  fn deref(&self) -> &Entity {
    &self.0
  }
}

impl From<LabelHandle> for Entity {
  fn from(h: LabelHandle) -> Self {
    h.0
  }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum LabelOverflow {
  #[default]
  Clip,
  MiddleEllipsis,
}

#[derive(Component, Clone, Debug)]
pub struct EllipsisText {
  pub full: String,
  last_avail: f32,
}

impl EllipsisText {
  pub fn new(full: impl Into<String>) -> Self {
    Self { full: full.into(), last_avail: 0.0 }
  }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct LabelConfig {
  pub text: String,
  pub style: LabelStyle,
  pub size: Option<f32>,
  pub color: Option<Color>,
  pub font: FontAttrs,
  pub disabled: bool,
  pub overflow: LabelOverflow,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct FontAttrs {
  pub weight: Option<FontWeight>,
  pub width: Option<FontWidth>,
  pub style: Option<FontStyle>,
  pub smoothing: Option<FontSmoothing>,
  pub features: Option<FontFeatures>,
  pub variations: Option<FontVariations>,
}

impl FontAttrs {
  pub fn apply(&self, font: &mut TextFont) {
    if let Some(v) = self.weight {
      font.weight = v;
    }
    if let Some(v) = self.width {
      font.width = v;
    }
    if let Some(v) = self.style {
      font.style = v;
    }
    if let Some(v) = self.smoothing {
      font.font_smoothing = v;
    }
    if let Some(v) = self.features.clone() {
      font.font_features = v;
    }
    if let Some(v) = self.variations.clone() {
      font.font_variations = v;
    }
  }
}

pub fn label(ctx: &UiCtx, parent: &mut ChildSpawner, config: LabelConfig) -> LabelHandle {
  let fs = &ctx.theme.metrics.font_size;
  let (size_token, color_token) = config.style.tokens();
  let size = config.size.unwrap_or_else(|| size_token(fs));
  let color = config.color.unwrap_or_else(|| color_of(color_token(&ctx.theme.colors)));
  let color = if config.disabled { dim_color(color) } else { color };
  let mut ec =
    parent.spawn(super::label_bundle_attrs(ctx, config.text.clone(), size, color, config.font));
  if config.overflow == LabelOverflow::MiddleEllipsis {
    ec.insert((
      EllipsisText::new(config.text),
      bevy::text::TextLayout::no_wrap(),
      Node { overflow: Overflow::clip(), ..default() },
    ));
  }
  LabelHandle(ec.id())
}

pub fn label_ellipsis_system(
  mut q: Query<(&mut Text, &mut EllipsisText, &ComputedNode, &TextLayoutInfo)>,
) {
  for (mut text, mut el, node, info) in &mut q {
    let avail = node.size().x;
    if avail <= 0.0 || el.full.is_empty() {
      continue;
    }
    if (avail - el.last_avail).abs() > 1.0 {
      el.last_avail = avail;
      if text.0 != el.full {
        text.0 = el.full.clone();
      }
      continue;
    }
    if info.size.x <= avail {
      continue;
    }
    let shown = middle_ellipsis(&el.full, avail, info.size.x);
    if text.0 != shown {
      text.0 = shown;
    }
  }
}

pub fn middle_ellipsis(full: &str, avail: f32, full_w: f32) -> String {
  let chars: Vec<char> = full.chars().collect();
  if full_w <= 0.0 || chars.len() <= 1 || avail >= full_w {
    return full.to_string();
  }
  let keep = ((chars.len() as f32) * (avail / full_w) * 0.98).floor().max(1.0) as usize;
  if keep >= chars.len() {
    return full.to_string();
  }
  let head = keep.div_ceil(2);
  let tail = keep - head;
  let mut s: String = chars[..head].iter().collect();
  s.push_str(ELLIPSIS);
  if tail > 0 {
    s.extend(&chars[chars.len() - tail..]);
  }
  s
}
