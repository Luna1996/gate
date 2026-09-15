//! label：主题字号/颜色的文本标签（docs/ui-dark-theme.md §5.6）。
//!
//! 变体由 [`LabelStyle`] preset enum 表达：字号 + 颜色永远成对来自主题令牌，拼不出非法组合
//! （如 faint 档 <18px 违反对比度规则）。禁止纯白：一律走主题令牌。

use std::ops::Deref;

use bevy::prelude::*;
use bevy::text::TextLayoutInfo;
use bevy::ui::Overflow;

use super::{UiCtx, color_of, dim_color};

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

/// 字号档取用器 / 颜色令牌取用器
type SizeFn = fn(&crate::theme::FontSize) -> f32;
type ColorFn = fn(&crate::theme::ThemeColors) -> &crate::theme::HexColor;

impl LabelStyle {
  /// (字号档, 颜色令牌)
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
  fn from(h: LabelHandle) -> Self {
    h.0
  }
}

/// 超宽文本的处理方式（label 基础能力）
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum LabelOverflow {
  /// 截断在节点矩形内（默认；配合调用方自行设置的 `Overflow::clip`）
  #[default]
  Clip,
  /// 中间省略：`head...tail`，宽度由节点实际排布宽度决定（见 [`EllipsisText`]）
  MiddleEllipsis,
}

/// 中间省略状态（挂在文本实体上；`full` 是完整原文，`Text` 里放截断结果）。
///
/// 测量用 bevy_text 布局结果（`TextLayoutInfo.size`）与节点实际宽度
/// （`ComputedNode.size`）比较，两者同为「缩放后物理 px」；本工程把窗口
/// `scale_factor` 固定为 1.0（见 gate-app `enforce_integer_scale_factor`），故与逻辑 px 等值。
#[derive(Component, Clone, Debug)]
pub struct EllipsisText {
  /// 完整原文（截断永远从这里重算，不叠加）
  pub full: String,
  /// 上次生效的可用宽度（变化 → 用 `full` 重排一次）
  last_avail: f32,
}

impl EllipsisText {
  pub fn new(full: impl Into<String>) -> Self {
    Self { full: full.into(), last_avail: 0.0 }
  }
}

/// 省略符
pub const ELLIPSIS: &str = "...";

/// 标签配置（全部字段进 Config；Default = 空文本 + Body 档、不禁用）
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LabelConfig {
  pub text: String,
  pub style: LabelStyle,
  /// true = 禁用态：文本颜色经 [`dim_color`] 降亮一档（不可交互，纯视觉）
  pub disabled: bool,
  /// 超宽处理（见 [`LabelOverflow`]）
  pub overflow: LabelOverflow,
}

/// 主题文本标签
pub fn label(ctx: &UiCtx, parent: &mut ChildSpawner, config: LabelConfig) -> LabelHandle {
  let fs = &ctx.theme.metrics.font_size;
  let (size, color) = config.style.tokens();
  let color = color_of(color(&ctx.theme.colors));
  let color = if config.disabled { dim_color(color) } else { color };
  let mut ec = parent.spawn(super::label_bundle(ctx, config.text.clone(), size(fs), color));
  if config.overflow == LabelOverflow::MiddleEllipsis {
    // NoWrap：中间省略要「整行自然宽度」，换行测量值会等于可用宽度而永不触发省略；
    // 节点自身 clip，省略收敛前的一两帧不会溢出
    ec.insert((
      EllipsisText::new(config.text),
      bevy::text::TextLayout::no_wrap(),
      Node { overflow: Overflow::clip(), ..default() },
    ));
  }
  LabelHandle(ec.id())
}

/// 中间省略：可用宽度放不下时改成 `head...tail`；宽度变化时用原文重排。
///
/// 截断永远从 [`EllipsisText::full`] 重算，故「窗口变宽 → 恢复原文」也成立。
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
      continue; // 下一帧按新宽度重算
    }
    if info.size.x <= avail {
      continue; // 放得下
    }
    let shown = middle_ellipsis(&el.full, avail, info.size.x);
    if text.0 != shown {
      text.0 = shown;
    }
  }
}

/// 按宽度比生成中间省略文本（纯函数；`full_w = 0` 时原样返回）
pub fn middle_ellipsis(full: &str, avail: f32, full_w: f32) -> String {
  let chars: Vec<char> = full.chars().collect();
  if full_w <= 0.0 || chars.len() <= 1 || avail >= full_w {
    return full.to_string();
  }
  // 目标保留字数按宽度比缩放，再留 2% 余量避免比目标宽度更宽
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
    assert_eq!(tf.font_size, bevy::text::FontSize::Px(theme.metrics.font_size.md));
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
      got.push(*label(
        &ctx,
        p,
        LabelConfig { text: "t".into(), style: LabelStyle::Title, ..default() },
      ));
      got.push(*label(
        &ctx,
        p,
        LabelConfig { text: "m".into(), style: LabelStyle::Muted, ..default() },
      ));
      got.push(*label(
        &ctx,
        p,
        LabelConfig { text: "f".into(), style: LabelStyle::FaintLg, ..default() },
      ));
      got.push(*label(
        &ctx,
        p,
        LabelConfig { text: "s".into(), style: LabelStyle::Success, ..default() },
      ));
    });
    let w = app.world();
    assert_eq!(w.get::<TextColor>(got[0]).unwrap().0, color_of(&theme.colors.text_primary));
    assert_eq!(
      w.get::<TextFont>(got[0]).unwrap().font_size,
      bevy::text::FontSize::Px(theme.metrics.font_size.lg)
    );
    assert_eq!(w.get::<TextColor>(got[1]).unwrap().0, color_of(&theme.colors.text_muted));
    assert_eq!(
      w.get::<TextFont>(got[2]).unwrap().font_size,
      bevy::text::FontSize::Px(theme.metrics.font_size.lg),
      "FaintLg stays >=18px"
    );
    assert_eq!(w.get::<TextColor>(got[3]).unwrap().0, color_of(&theme.colors.success));
  }

  #[test]
  fn middle_ellipsis_keeps_head_and_tail() {
    // 宽度刚好够 → 原样
    assert_eq!(middle_ellipsis("/debug/overlay", 200.0, 100.0), "/debug/overlay");
    // 只够一半 → head + ... + tail
    let s = middle_ellipsis("/debug/overlay", 50.0, 100.0);
    assert!(s.contains("..."), "{s}");
    assert!(s.starts_with('/'), "head kept: {s}");
    assert!(s.ends_with("lay"), "tail kept: {s}");
    assert!(s.chars().count() < "/debug/overlay".chars().count());
    // 极窄 → 至少保留一个字符（不 panic、不为空）
    let tiny = middle_ellipsis("/debug/overlay", 1.0, 100.0);
    assert!(tiny.contains("..."));
    // 退化输入原样返回
    assert_eq!(middle_ellipsis("/", 1.0, 100.0), "/");
    assert_eq!(middle_ellipsis("/debug", 100.0, 0.0), "/debug");
  }
}
