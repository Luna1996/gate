//! text_input：单行文本输入框（纯文本模式 / 数字模式）。
//!
//! - **纯文本模式**：点击进入编辑态，键盘输入字符、Backspace/Delete 删除、左右键移光标，
//!   Enter/Escape/失焦提交。编辑态由 [`TextInputFocus`] 资源对外广播（3D 场景输入据此屏蔽键盘）。
//! - **数字模式**：在框内按住左键水平拖拽直接调值（每 [`NUMBER_DRAG_PX_PER_STEP`] 逻辑 px 一个
//!   step，超过 [`DRAG_THRESHOLD_PX`] 才算拖拽）；同样可点击进入编辑态手输，提交时按
//!   `min/max/step/decimals` 归一。
//!
//! 真源 = [`TextInputValue`] 组件（显示文本）；事件 [`TextInputChanged`] 只在**提交**（回车/失焦）
//! 或**拖拽调值**时发出，避免打字中途把半成品值推给调用方。

use std::ops::Deref;

use bevy::input::keyboard::{Key, KeyboardInput};
use bevy::input::mouse::AccumulatedMouseMotion;
use bevy::prelude::*;
use bevy::ui::{FocusPolicy, Interaction};

use super::{UiCtx, UiDisabled, color_of, dim_color, px};
use crate::theme::UiTheme;

/// 数字模式拖拽灵敏度：每 N 逻辑 px = 1 个 step
pub const NUMBER_DRAG_PX_PER_STEP: f32 = 4.0;
/// 「点击」与「拖拽」的位移判定阈值（逻辑 px）
pub const DRAG_THRESHOLD_PX: f32 = 3.0;
/// 编辑态光标字符（仅显示态插入，不写进值）
pub const CARET_CHAR: char = '|';

/// 输入框根标记
#[derive(Component, Debug, Default)]
pub struct TextInputRoot;

/// 输入框显示文本子实体标记
#[derive(Component, Debug, Default)]
pub struct TextInputText;

/// 输入模式（挂在输入框根实体上）
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub enum TextInputKind {
  /// 纯文本（任意可见字符）
  Text,
  /// 数字（支持左键拖拽调值；提交时按 min/max/step 归一）
  Number { min: f32, max: f32, step: f32, decimals: usize },
}

impl TextInputKind {
  /// 数值归一：clamp + 按 step 吸附（step <= 0 视为连续）
  pub fn normalize(&self, v: f32) -> f32 {
    let Self::Number { min, max, step, .. } = *self else {
      return v;
    };
    let c = v.clamp(min, max);
    let snapped = if step > 0.0 { min + ((c - min) / step).round() * step } else { c };
    snapped.clamp(min, max)
  }

  /// 数值 → 显示文本
  pub fn format(&self, v: f32) -> String {
    match *self {
      Self::Text => String::new(),
      Self::Number { decimals, .. } => format!("{v:.*}", decimals),
    }
  }

  /// 数值模式的初值（文本解析失败时的回退）
  pub fn min(&self) -> f32 {
    match *self {
      Self::Text => 0.0,
      Self::Number { min, .. } => min,
    }
  }

  /// 文本 → 数值（数字模式；解析失败 → None）
  pub fn parse(&self, s: &str) -> Option<f32> {
    match *self {
      Self::Text => None,
      Self::Number { .. } => s.trim().parse::<f32>().ok().map(|v| self.normalize(v)),
    }
  }
}

/// 输入框句柄（Deref 到根实体 Entity）
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TextInputHandle(pub Entity);

impl Deref for TextInputHandle {
  type Target = Entity;
  fn deref(&self) -> &Entity {
    &self.0
  }
}

impl From<TextInputHandle> for Entity {
  fn from(h: TextInputHandle) -> Self {
    h.0
  }
}

/// 输入框配置（全部字段进 Config；Default = 空文本 + 纯文本模式、不禁用）
#[derive(Clone, Debug, PartialEq)]
pub struct TextInputConfig {
  pub text: String,
  pub kind: TextInputKind,
  pub disabled: bool,
}

impl Default for TextInputConfig {
  fn default() -> Self {
    Self { text: String::new(), kind: TextInputKind::Text, disabled: false }
  }
}

/// 输入值（真源；显示态可能会叠加光标字符）
#[derive(Component, Clone, Debug, PartialEq, Eq, Default)]
pub struct TextInputValue(pub String);

/// 输入框交互/编辑状态
#[derive(Component, Clone, Copy, Debug, Default, PartialEq)]
pub struct TextInputState {
  /// 编辑态（键盘输入落到本框）
  pub editing: bool,
  /// 光标位置（字符下标，0..=len）
  pub caret: usize,
  /// 本次按下是否起始于本框
  armed: bool,
  /// 是否已进入数字拖拽
  dragging: bool,
  /// 拖拽累计位移（逻辑 px）
  accum: f32,
  /// 拖拽起始值
  start_value: f32,
}

/// 输入值变化事件（提交或拖拽调值；EntityEvent，target = 输入框根实体）
#[derive(EntityEvent, Clone, Debug, PartialEq)]
pub struct TextInputChanged {
  pub entity: Entity,
  /// 归一后的值文本（数字模式已 clamp/step；纯文本模式为原文）
  pub text: String,
}

/// 当前处于编辑态的输入框（None = 无）。
///
/// 3D 场景的键盘输入（飞行/档位切换）必须检查本资源：否则打字会同时驱动相机。
#[derive(Resource, Default, Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextInputFocus(pub Option<Entity>);

/// 主题输入框（宽度由父容器决定；最小宽度 56px）
pub fn text_input(
  ctx: &UiCtx,
  parent: &mut ChildSpawner,
  config: TextInputConfig,
) -> TextInputHandle {
  let m = &ctx.theme.metrics;
  let (bg, border, text_color) = input_colors(ctx.theme, false, Interaction::None);
  let (bg, border, text_color) = if config.disabled {
    (dim_color(bg), dim_color(border), dim_color(text_color))
  } else {
    (bg, border, text_color)
  };
  let mut ec = parent.spawn((
    Name::new("ui-text-input"),
    TextInputRoot,
    Interaction::default(),
    TextInputState::default(),
    TextInputValue(config.text.clone()),
    config.kind,
    Node {
      min_width: px(56.0),
      // 与行内其它控件同高（滑杆/切换组 = 24）：文字高 + 上下内距 + 上下边框
      height: px(m.font_size.md + m.spacing.sm + m.border_width * 2.0),
      padding: UiRect::horizontal(px(m.spacing.xs)),
      border: UiRect::all(px(m.border_width)),
      align_items: AlignItems::Center,
      overflow: Overflow::clip(),
      ..default()
    },
    BackgroundColor(bg),
    BorderColor::all(border),
    // 输入框吞掉鼠标事件（拖拽调值 / 点击聚焦都不外泄到 3D 场景）
    FocusPolicy::Block,
    crate::capture::MouseIntercept,
  ));
  if config.disabled {
    ec.insert(UiDisabled);
  }
  ec.with_children(|root| {
    root.spawn((
      Name::new("ui-text-input-text"),
      TextInputText,
      Text::new(config.text),
      TextFont {
        font: ctx.font_source(),
        font_size: bevy::text::FontSize::Px(m.font_size.md),
        ..default()
      },
      TextColor(text_color),
      TextLayout { linebreak: bevy::text::LineBreak::NoWrap, ..default() },
    ));
  });
  TextInputHandle(ec.id())
}

/// (背景, 边框, 文字) 配色：编辑态 → 强调边框；hover → 强边框；否则普通边框
fn input_colors(theme: &UiTheme, editing: bool, inter: Interaction) -> (Color, Color, Color) {
  let c = &theme.colors;
  let bg = color_of(&c.surface_elevated);
  let border = if editing {
    color_of(&c.accent_text)
  } else if inter != Interaction::None {
    color_of(&c.border_strong)
  } else {
    color_of(&c.border)
  };
  (bg, border, color_of(&c.text_primary))
}

/// 指针交互：点击聚焦（失焦提交）、数字模式拖拽调值
#[allow(clippy::type_complexity)] // Bevy system：多组件查询签名固有
pub fn text_input_pointer_system(
  mut commands: Commands,
  mouse: Res<ButtonInput<MouseButton>>,
  motion: Res<AccumulatedMouseMotion>,
  mut focus: ResMut<TextInputFocus>,
  mut q: Query<(
    Entity,
    &Interaction,
    &TextInputKind,
    &mut TextInputValue,
    &mut TextInputState,
    Has<UiDisabled>,
  )>,
) {
  let pressed = mouse.just_pressed(MouseButton::Left);
  let held = mouse.pressed(MouseButton::Left);
  let released = mouse.just_released(MouseButton::Left);
  if !pressed && !held && !released {
    return;
  }
  for (e, inter, kind, mut value, mut st, disabled) in &mut q {
    if disabled {
      continue;
    }
    // 落在别的控件上的按下 → 提交并退出编辑态
    if pressed && *inter == Interaction::None && st.editing {
      st.editing = false;
      if focus.0 == Some(e) {
        focus.0 = None;
      }
      commands.trigger(TextInputChanged { entity: e, text: value.0.clone() });
      continue;
    }
    if pressed {
      st.armed = *inter != Interaction::None;
      st.accum = 0.0;
      st.dragging = false;
      st.start_value = kind.parse(&value.0).unwrap_or_else(|| kind.min());
    }
    if held && st.armed {
      st.accum += motion.delta.x;
      if !st.dragging
        && matches!(kind, TextInputKind::Number { .. })
        && st.accum.abs() > DRAG_THRESHOLD_PX
      {
        st.dragging = true;
        st.editing = false;
        if focus.0 == Some(e) {
          focus.0 = None;
        }
      }
      if st.dragging {
        let steps = st.accum / NUMBER_DRAG_PX_PER_STEP;
        let delta = match *kind {
          TextInputKind::Number { step, .. } => steps * step,
          TextInputKind::Text => 0.0,
        };
        let v = kind.normalize(st.start_value + delta);
        let text = kind.format(v);
        if value.0 != text {
          value.0 = text;
          commands.trigger(TextInputChanged { entity: e, text: value.0.clone() });
        }
      }
    }
    if released && st.armed {
      if st.dragging {
        st.dragging = false;
        commands.trigger(TextInputChanged { entity: e, text: value.0.clone() });
      } else {
        // 单击：进入编辑态并把光标放到末尾
        st.editing = true;
        st.caret = value.0.chars().count();
        focus.0 = Some(e);
      }
      st.armed = false;
    }
  }
}

/// 键盘编辑：字符输入 / 删除 / 移光标 / 提交（Enter、Escape、Tab）
pub fn text_input_keyboard_system(
  mut commands: Commands,
  mut focus: ResMut<TextInputFocus>,
  mut keys: MessageReader<KeyboardInput>,
  mut q: Query<(Entity, &TextInputKind, &mut TextInputValue, &mut TextInputState)>,
) {
  let Some(fe) = focus.0 else {
    keys.clear();
    return;
  };
  let Ok((e, kind, mut value, mut st)) = q.get_mut(fe) else {
    focus.0 = None;
    keys.clear();
    return;
  };
  let mut chars: Vec<char> = value.0.chars().collect();
  let mut dirty = false;
  let mut commit = false;
  for ev in keys.read() {
    if ev.state != bevy::input::ButtonState::Pressed {
      continue;
    }
    match &ev.logical_key {
      Key::Backspace => {
        if st.caret > 0 && st.caret <= chars.len() {
          chars.remove(st.caret - 1);
          st.caret -= 1;
          dirty = true;
        }
      }
      Key::Delete => {
        if st.caret < chars.len() {
          chars.remove(st.caret);
          dirty = true;
        }
      }
      Key::ArrowLeft => st.caret = st.caret.saturating_sub(1),
      Key::ArrowRight => st.caret = (st.caret + 1).min(chars.len()),
      Key::Home => st.caret = 0,
      Key::End => st.caret = chars.len(),
      Key::Enter | Key::Escape | Key::Tab => commit = true,
      _ => {
        if let Some(t) = ev.text.as_ref() {
          for ch in t.chars().filter(|c| !c.is_control()) {
            let at = st.caret.min(chars.len());
            chars.insert(at, ch);
            st.caret = at + 1;
            dirty = true;
          }
        }
      }
    }
  }
  if dirty {
    value.0 = chars.into_iter().collect();
    st.caret = st.caret.min(value.0.chars().count());
  }
  if commit {
    st.editing = false;
    focus.0 = None;
    // 数字模式提交时归一（非法输入回退到当前文本）
    if let Some(v) = kind.parse(&value.0) {
      value.0 = kind.format(v);
    }
    commands.trigger(TextInputChanged { entity: e, text: value.0.clone() });
  }
}

/// 视觉：显示文本（编辑态叠加光标）+ 边框/文字配色
#[allow(clippy::type_complexity)] // Bevy system：多组件查询签名固有
pub fn text_input_visual_system(
  theme: Option<Res<UiTheme>>,
  mut q: Query<(
    &TextInputValue,
    &TextInputState,
    &TextInputKind,
    &Interaction,
    &mut BackgroundColor,
    &mut BorderColor,
    &Children,
    Has<UiDisabled>,
  )>,
  mut q_text: Query<(&mut Text, &mut TextColor)>,
) {
  let Some(theme) = theme else { return };
  for (value, st, kind, inter, mut bg, mut border, children, disabled) in &mut q {
    let (target_bg, target_border, target_text) = input_colors(&theme, st.editing, *inter);
    let (target_bg, target_border, target_text) = if disabled {
      (dim_color(target_bg), dim_color(target_border), dim_color(target_text))
    } else {
      (target_bg, target_border, target_text)
    };
    if bg.0 != target_bg {
      bg.0 = target_bg;
    }
    let target_border = BorderColor::all(target_border);
    if *border != target_border {
      *border = target_border;
    }
    // 编辑态显示光标（不写进真源值）；数字模式拖拽中显示归一后的文本
    let shown = if st.editing {
      let mut s: Vec<char> = value.0.chars().collect();
      let at = st.caret.min(s.len());
      s.insert(at, CARET_CHAR);
      s.into_iter().collect()
    } else if value.0.is_empty() && matches!(kind, TextInputKind::Text) {
      String::new()
    } else {
      value.0.clone()
    };
    for child in children.iter() {
      if let Ok((mut text, mut color)) = q_text.get_mut(child) {
        if text.0 != shown {
          text.0 = shown.clone();
        }
        if color.0 != target_text {
          color.0 = target_text;
        }
      }
    }
  }
}
