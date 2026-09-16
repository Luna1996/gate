//! label：文本标签，字号/颜色/字体属性各自独立可设。
//!
//! 两条取值路径，同一个标签上可混用（每个属性互不影响）：
//! 1. **预设档** [`LabelStyle`]：字号 + 颜色**成对**取自主题令牌，是默认路径，
//!    保证拼不出违反对比度的组合（如 faint 档 <18px）；
//! 2. **逐属性覆盖**：[`LabelConfig`] 的 `size`（字号）/ `color`（颜色）/ `font`（[`FontAttrs`]：
//!    字重、字宽、倾斜、抗锯齿、OpenType 特性、可变字体轴）都是 `Option`，填哪项覆盖哪项，
//!    没填的仍走预设档 —— 例如「Muted 的次要灰 + 正文档字号」直接写
//!    `LabelConfig { style: LabelStyle::Muted, size: Some(fs.md), ..default() }`。
//!
//! 覆盖值不走令牌、也不做合规校验（调用方自担）；禁止纯白一类硬约束只由预设档保证。

use std::ops::Deref;

use bevy::prelude::*;
use bevy::text::TextLayoutInfo;
use bevy::text::{FontFeatures, FontSmoothing, FontStyle, FontVariations, FontWeight, FontWidth};
use bevy::ui::Overflow;

use super::{UiCtx, color_of, dim_color};

/// 文本预设档：四档正文（primary/body/muted/faint）+ 四档语义色。
///
/// 每档 = 字号 + 颜色**成对**取自主题令牌，作为 [`LabelConfig`] 的默认值；
/// 想单独改字号或颜色时用 `LabelConfig::size` / `LabelConfig::color` 覆盖其中一项。
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

/// 标签配置：**预设档给默认值，每个属性都能单独覆盖**。
///
/// Default = 空文本 + Body 档 + 不覆盖任何属性 + 不禁用。
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LabelConfig {
  pub text: String,
  /// 预设档：提供字号与颜色的默认值（被下面的 `size` / `color` 覆盖时以覆盖值为准）
  pub style: LabelStyle,
  /// 字号覆盖（逻辑 px）；None = 用预设档字号
  pub size: Option<f32>,
  /// 颜色覆盖；None = 用预设档颜色
  pub color: Option<Color>,
  /// 其余字体属性逐项覆盖（见 [`FontAttrs`]）
  pub font: FontAttrs,
  /// true = 禁用态：文本颜色经 [`dim_color`] 降亮一档（不可交互，纯视觉）
  pub disabled: bool,
  /// 超宽处理（见 [`LabelOverflow`]）
  pub overflow: LabelOverflow,
}

/// 可逐项覆盖的 `TextFont` 属性（每项 `None` = 用 bevy 的默认值）。
///
/// 不含 `font`（字体句柄由 [`UiCtx`] 统一给主题字体）与 `font_size`（走 [`LabelConfig::size`]）。
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FontAttrs {
  /// 字重（仅可变字重字体生效）
  pub weight: Option<FontWeight>,
  /// 字宽（压缩 / 扩展）
  pub width: Option<FontWidth>,
  /// 字形倾斜（normal / italic / oblique）
  pub style: Option<FontStyle>,
  /// 抗锯齿方式
  pub smoothing: Option<FontSmoothing>,
  /// OpenType 特性（连字 / 数字样式等）
  pub features: Option<FontFeatures>,
  /// 可变字体轴
  pub variations: Option<FontVariations>,
}

impl FontAttrs {
  /// 把填了的项写进 `TextFont`（未填的保持原值）
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

/// 主题文本标签（字号/颜色/字体属性各自独立取值，见模块文档）
pub fn label(ctx: &UiCtx, parent: &mut ChildSpawner, config: LabelConfig) -> LabelHandle {
  let fs = &ctx.theme.metrics.font_size;
  let (size_token, color_token) = config.style.tokens();
  let size = config.size.unwrap_or_else(|| size_token(fs));
  let color = config.color.unwrap_or_else(|| color_of(color_token(&ctx.theme.colors)));
  let color = if config.disabled { dim_color(color) } else { color };
  let mut ec =
    parent.spawn(super::label_bundle_attrs(ctx, config.text.clone(), size, color, config.font));
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
