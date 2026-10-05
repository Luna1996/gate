use std::ops::Deref;

use bevy::input::keyboard::{Key, KeyboardInput};
use bevy::input::mouse::AccumulatedMouseMotion;
use bevy::picking::hover::Hovered;
use bevy::prelude::*;
use bevy::ui::Pressed;

use super::{UiCtx, UiDisabled, color_of, dim_color, px};
use crate::pointer::{UiInteract, UiInteractBundle};
use crate::theme::UiTheme;
use crate::widgets::consts::{CARET_CHAR, DRAG_THRESHOLD_PX, NUMBER_DRAG_PX_PER_STEP};

#[derive(Component, Debug, Default)]
pub struct TextInputRoot;

#[derive(Component, Debug, Default)]
pub struct TextInputText;

#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub enum TextInputKind {
  Text,
  Number { min: f32, max: f32, step: f32, decimals: usize },
}

impl TextInputKind {
  pub fn normalize(&self, v: f32) -> f32 {
    let Self::Number { min, max, step, .. } = *self else {
      return v;
    };
    let c = v.clamp(min, max);
    let snapped = if step > 0.0 { min + ((c - min) / step).round() * step } else { c };
    snapped.clamp(min, max)
  }

  pub fn format(&self, v: f32) -> String {
    match *self {
      Self::Text => String::new(),
      Self::Number { decimals, .. } => format!("{v:.*}", decimals),
    }
  }

  pub fn min(&self) -> f32 {
    match *self {
      Self::Text => 0.0,
      Self::Number { min, .. } => min,
    }
  }

  pub fn parse(&self, s: &str) -> Option<f32> {
    match *self {
      Self::Text => None,
      Self::Number { .. } => s.trim().parse::<f32>().ok().map(|v| self.normalize(v)),
    }
  }
}

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

#[derive(Component, Clone, Debug, PartialEq, Eq, Default)]
pub struct TextInputValue(pub String);

#[derive(Component, Clone, Copy, Debug, Default, PartialEq)]
pub struct TextInputState {
  pub editing: bool,
  pub caret: usize,
  armed: bool,
  dragging: bool,
  accum: f32,
  start_value: f32,
}

#[derive(EntityEvent, Clone, Debug, PartialEq)]
pub struct TextInputChanged {
  pub entity: Entity,
  pub text: String,
}

#[derive(Resource, Default, Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextInputFocus(pub Option<Entity>);

pub fn text_input(
  ctx: &UiCtx,
  parent: &mut ChildSpawner,
  config: TextInputConfig,
) -> TextInputHandle {
  let m = &ctx.theme.metrics;
  let (bg, border, text_color) = input_colors(ctx.theme, false, UiInteract::None);
  let (bg, border, text_color) = if config.disabled {
    (dim_color(bg), dim_color(border), dim_color(text_color))
  } else {
    (bg, border, text_color)
  };
  let mut ec = parent.spawn((
    Name::new("ui-text-input"),
    TextInputRoot,
    UiInteractBundle::default(),
    TextInputState::default(),
    TextInputValue(config.text.clone()),
    config.kind,
    Node {
      min_width: px(56.0),
      height: px(m.font_size.md + m.spacing.sm + m.border_width * 2.0),
      padding: UiRect::horizontal(px(m.spacing.xs)),
      border: UiRect::all(px(m.border_width)),
      align_items: AlignItems::Center,
      overflow: Overflow::clip(),
      ..default()
    },
    BackgroundColor(bg),
    BorderColor::all(border),
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

fn input_colors(theme: &UiTheme, editing: bool, inter: UiInteract) -> (Color, Color, Color) {
  let c = &theme.colors;
  let bg = color_of(&c.surface_elevated);
  let border = if editing {
    color_of(&c.accent_text)
  } else if inter.is_active() {
    color_of(&c.border_strong)
  } else {
    color_of(&c.border)
  };
  (bg, border, color_of(&c.text_primary))
}

#[allow(clippy::type_complexity)]
pub fn text_input_pointer_system(
  mut commands: Commands,
  mouse: Res<ButtonInput<MouseButton>>,
  motion: Res<AccumulatedMouseMotion>,
  mut focus: ResMut<TextInputFocus>,
  mut q: Query<(
    Entity,
    &Hovered,
    Has<Pressed>,
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
  for (e, hovered, is_pressed, kind, mut value, mut st, disabled) in &mut q {
    let inter = UiInteract::of(hovered, is_pressed);
    if disabled {
      continue;
    }
    if pressed && inter == UiInteract::None && st.editing {
      st.editing = false;
      if focus.0 == Some(e) {
        focus.0 = None;
      }
      commands.trigger(TextInputChanged { entity: e, text: value.0.clone() });
      continue;
    }
    if pressed {
      st.armed = inter.is_active();
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
        st.editing = true;
        st.caret = value.0.chars().count();
        focus.0 = Some(e);
      }
      st.armed = false;
    }
  }
}

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
    if let Some(v) = kind.parse(&value.0) {
      value.0 = kind.format(v);
    }
    commands.trigger(TextInputChanged { entity: e, text: value.0.clone() });
  }
}

#[allow(clippy::type_complexity)]
pub fn text_input_visual_system(
  theme: Option<Res<UiTheme>>,
  mut q: Query<(
    &TextInputValue,
    &TextInputState,
    &TextInputKind,
    &Hovered,
    Has<Pressed>,
    &mut BackgroundColor,
    &mut BorderColor,
    &Children,
    Has<UiDisabled>,
  )>,
  mut q_text: Query<(&mut Text, &mut TextColor)>,
) {
  let Some(theme) = theme else { return };
  for (value, st, kind, hovered, pressed, mut bg, mut border, children, disabled) in &mut q {
    let inter = UiInteract::of(hovered, pressed);
    let (target_bg, target_border, target_text) = input_colors(&theme, st.editing, inter);
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
