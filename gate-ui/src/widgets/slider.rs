use std::ops::Deref;

use bevy::picking::hover::Hovered;
use bevy::prelude::*;
use bevy::ui::{Pressed, RelativeCursorPosition};

use super::{UiCtx, UiDisabled, color_of, dim_color, px};
use crate::pointer::{UiInteract, UiInteractBundle};
use crate::theme::UiTheme;
use crate::widgets::consts::{
  SLIDER_FINE_SCALE, THUMB_INSET, THUMB_SIZE, THUMB_SIZE_DRAG, TRACK_HEIGHT,
};

#[derive(Component, Debug, Default)]
pub struct UiSlider;

#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct SliderRange {
  pub min: f32,
  pub max: f32,
}

impl Default for SliderRange {
  fn default() -> Self {
    Self { min: 0.0, max: 1.0 }
  }
}

#[derive(Component, Clone, Copy, Debug, PartialEq, Default)]
pub struct SliderStep(pub Option<f32>);

#[derive(Component, Clone, Copy, Debug, PartialEq, Default)]
pub struct SliderValue(pub f32);

#[derive(Component, Debug, Default)]
pub struct SliderFill;

#[derive(Component, Debug, Default)]
pub struct SliderTrack;

#[derive(Component, Debug, Default)]
pub struct SliderThumb;

#[derive(Component, Debug, Default)]
pub struct SliderThumbSlot;

#[derive(Component, Clone, Copy, Debug, Default, PartialEq)]
pub struct SliderDrag {
  fine: bool,
  anchor_cursor_x: f32,
  anchor_value: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SliderHandle(pub Entity);

impl Deref for SliderHandle {
  type Target = Entity;
  fn deref(&self) -> &Entity {
    &self.0
  }
}

impl From<SliderHandle> for Entity {
  fn from(h: SliderHandle) -> Entity {
    h.0
  }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SliderConfig {
  pub min: f32,
  pub max: f32,
  pub value: f32,
  pub step: Option<f32>,
  pub disabled: bool,
}

impl Default for SliderConfig {
  fn default() -> Self {
    Self { min: 0.0, max: 1.0, value: 0.0, step: None, disabled: false }
  }
}

#[derive(EntityEvent, Clone, Copy, Debug, PartialEq)]
pub struct SliderValueChanged {
  pub entity: Entity,
  pub value: f32,
}

pub fn clamp_step(value: f32, min: f32, max: f32, step: Option<f32>) -> f32 {
  let v = value.clamp(min, max);
  match step {
    Some(s) if s > 0.0 => min + ((v - min) / s).round() * s,
    _ => v,
  }
  .clamp(min, max)
}

pub fn slider(ctx: &UiCtx, parent: &mut ChildSpawner, config: SliderConfig) -> SliderHandle {
  let c = &ctx.theme.colors;
  let m = &ctx.theme.metrics;
  let v = clamp_step(config.value, config.min, config.max, config.step);
  let norm = normalize(v, config.min, config.max);
  let track_bg = color_of(&c.surface_overlay);
  let fill_bg = color_of(&c.accent_fill_hover);
  let thumb_bg = color_of(&c.surface_top);
  let thumb_border = color_of(&c.border_strong);
  let (track_bg, fill_bg, thumb_bg, thumb_border) = if config.disabled {
    (dim_color(track_bg), dim_color(fill_bg), dim_color(thumb_bg), dim_color(thumb_border))
  } else {
    (track_bg, fill_bg, thumb_bg, thumb_border)
  };
  let mut ec = parent.spawn((
    Name::new("ui-slider"),
    UiSlider,
    UiInteractBundle::default(),
    RelativeCursorPosition::default(),
    SliderDrag::default(),
    SliderRange { min: config.min, max: config.max },
    SliderStep(config.step),
    SliderValue(v),
    Node { min_width: px(120.0), height: px(24.0), align_items: AlignItems::Center, ..default() },
  ));
  ec.with_children(|root| {
    root
      .spawn((
        Name::new("ui-slider-track"),
        SliderTrack,
        Node { flex_grow: 1.0, height: px(TRACK_HEIGHT), ..default() },
        BackgroundColor(track_bg),
      ))
      .with_children(|track| {
        track.spawn((
          Name::new("ui-slider-fill"),
          SliderFill,
          Node {
            position_type: PositionType::Absolute,
            left: Val::ZERO,
            top: Val::ZERO,
            bottom: Val::ZERO,
            width: Val::Percent(norm * 100.0),
            ..default()
          },
          BackgroundColor(fill_bg),
        ));
        track
          .spawn((
            Name::new("ui-slider-thumb-slot"),
            SliderThumbSlot,
            Node {
              position_type: PositionType::Absolute,
              left: Val::ZERO,
              right: Val::ZERO,
              top: Val::ZERO,
              bottom: Val::ZERO,
              margin: UiRect::horizontal(px(THUMB_INSET)),
              ..default()
            },
          ))
          .with_children(|slot| {
            slot.spawn((
              Name::new("ui-slider-thumb"),
              SliderThumb,
              Node {
                position_type: PositionType::Absolute,
                left: Val::Percent(norm * 100.0),
                top: Val::Percent(50.0),
                margin: UiRect {
                  left: px(-THUMB_SIZE / 2.0),
                  top: px(-THUMB_SIZE / 2.0),
                  ..default()
                },
                width: px(THUMB_SIZE),
                height: px(THUMB_SIZE),
                border: UiRect::all(px(m.border_width)),
                ..default()
              },
              BackgroundColor(thumb_bg),
              BorderColor::all(thumb_border),
            ));
          });
      });
  });
  if config.disabled {
    ec.insert(UiDisabled);
  }
  SliderHandle(ec.id())
}

fn normalize(v: f32, min: f32, max: f32) -> f32 {
  if (max - min).abs() < f32::EPSILON { 0.0 } else { ((v - min) / (max - min)).clamp(0.0, 1.0) }
}

#[allow(clippy::type_complexity)]
pub fn slider_drag_system(
  mut commands: Commands,
  keys: Res<ButtonInput<KeyCode>>,
  mut shift_captured: ResMut<crate::capture::UiShiftCaptured>,
  mut q: Query<(
    Entity,
    &Hovered,
    Has<Pressed>,
    &RelativeCursorPosition,
    &ComputedNode,
    &SliderRange,
    &SliderStep,
    &mut SliderValue,
    &mut SliderDrag,
    Has<UiDisabled>,
  )>,
) {
  let shift = keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight);
  let mut fine_drag = false;
  for (e, hovered, pressed, rcp, node, range, step, mut val, mut drag, disabled) in &mut q {
    let inter = UiInteract::of(hovered, pressed);
    if disabled || inter != UiInteract::Pressed {
      drag.fine = false;
      continue;
    }
    fine_drag = shift;
    let Some(n) = rcp.normalized else { continue };
    let root_w = node.size().x;
    let cursor_x = (n.x + 0.5) * root_w;
    let pad_l = node.padding.min_inset.x;
    let pad_r = node.padding.max_inset.x;
    let travel_w = root_w - pad_l - pad_r - THUMB_SIZE;
    let span = range.max - range.min;
    let target = if travel_w <= 0.0 {
      range.min
    } else if shift {
      if !drag.fine {
        drag.fine = true;
        drag.anchor_cursor_x = cursor_x;
        drag.anchor_value = val.0;
      }
      drag.anchor_value + (cursor_x - drag.anchor_cursor_x) / travel_w * span * SLIDER_FINE_SCALE
    } else {
      drag.fine = false;
      let norm = ((cursor_x - pad_l - THUMB_SIZE / 2.0) / travel_w).clamp(0.0, 1.0);
      range.min + norm * span
    };
    let v = clamp_step(target, range.min, range.max, step.0);
    if (val.0 - v).abs() > f32::EPSILON {
      val.0 = v;
      commands.trigger(SliderValueChanged { entity: e, value: v });
    }
  }
  shift_captured.set_if_neq(crate::capture::UiShiftCaptured(fine_drag));
}

#[allow(clippy::type_complexity)]
pub fn slider_visual_system(
  theme: Option<Res<UiTheme>>,
  mut q_root: Query<
    (&SliderValue, &SliderRange, &Hovered, Has<Pressed>, &Children, Has<UiDisabled>),
    With<UiSlider>,
  >,
  mut q_track: Query<
    &mut BackgroundColor,
    (With<SliderTrack>, Without<SliderFill>, Without<SliderThumb>),
  >,
  mut fills: Query<(&mut Node, &mut BackgroundColor), (With<SliderFill>, Without<SliderThumb>)>,
  mut thumbs: Query<(&mut Node, &mut BackgroundColor, &mut BorderColor), With<SliderThumb>>,
  q_track_children: Query<&Children, Without<UiSlider>>,
  q_slot_children: Query<&Children, With<SliderThumbSlot>>,
) {
  let Some(theme) = theme else { return };
  let c = &theme.colors;
  let track_bg = color_of(&c.surface_overlay);
  let fill_bg = color_of(&c.accent_fill_hover);
  let thumb_bg = color_of(&c.surface_top);
  let thumb_border = color_of(&c.border_strong);
  for (val, range, hovered, pressed, children, disabled) in &mut q_root {
    let inter = UiInteract::of(hovered, pressed);
    let pct = normalize(val.0, range.min, range.max) * 100.0;
    let thumb_size =
      if !disabled && inter == UiInteract::Pressed { THUMB_SIZE_DRAG } else { THUMB_SIZE };
    let (track_bg, fill_bg, thumb_bg, thumb_border) = if disabled {
      (dim_color(track_bg), dim_color(fill_bg), dim_color(thumb_bg), dim_color(thumb_border))
    } else {
      (track_bg, fill_bg, thumb_bg, thumb_border)
    };
    for child in children.iter() {
      if let Ok(tc) = q_track_children.get(child) {
        if let Ok(mut bg) = q_track.get_mut(child)
          && bg.0 != track_bg
        {
          bg.0 = track_bg;
        }
        for sub in tc.iter() {
          if let Ok((mut node, mut bg)) = fills.get_mut(sub) {
            node.width = Val::Percent(pct);
            if bg.0 != fill_bg {
              bg.0 = fill_bg;
            }
          }
          let Ok(sc) = q_slot_children.get(sub) else { continue };
          for thumb in sc.iter() {
            if let Ok((mut node, mut bg, mut bc)) = thumbs.get_mut(thumb) {
              node.left = Val::Percent(pct);
              node.top = Val::Percent(50.0);
              node.width = px(thumb_size);
              node.height = px(thumb_size);
              node.margin =
                UiRect { left: px(-thumb_size / 2.0), top: px(-thumb_size / 2.0), ..default() };
              let border = BorderColor::all(thumb_border);
              if bg.0 != thumb_bg {
                bg.0 = thumb_bg;
              }
              if *bc != border {
                *bc = border;
              }
            }
          }
        }
      }
    }
  }
}
