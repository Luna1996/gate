use std::ops::Deref;

use bevy::picking::hover::Hovered;
use bevy::prelude::*;
use bevy::ui::{Checked, Pressed};

use super::{UiCtx, UiDisabled, color_of, dim_color, px, spawn_label};
use crate::pointer::{UiInteract, UiInteractBundle, UiInteractPrev};
use crate::theme::UiTheme;
use crate::widgets::consts::{TRACK_H, TRACK_W};

#[derive(Component, Debug, Default)]
pub struct ToggleSwitch;

#[derive(Component, Debug, Default)]
pub struct ToggleTrack;

#[derive(Component, Debug, Default)]
pub struct ToggleKnob;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ToggleSwitchHandle(pub Entity);

impl Deref for ToggleSwitchHandle {
  type Target = Entity;
  fn deref(&self) -> &Entity {
    &self.0
  }
}

impl From<ToggleSwitchHandle> for Entity {
  fn from(h: ToggleSwitchHandle) -> Entity {
    h.0
  }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ToggleSwitchConfig {
  pub text: Option<String>,
  pub checked: bool,
  pub disabled: bool,
}

#[derive(EntityEvent, Clone, Copy, Debug, PartialEq)]
pub struct ToggleSwitchToggled {
  pub entity: Entity,
  pub checked: bool,
}

pub fn toggle_switch(
  ctx: &UiCtx,
  parent: &mut ChildSpawner,
  config: ToggleSwitchConfig,
) -> ToggleSwitchHandle {
  let c = &ctx.theme.colors;
  let m = &ctx.theme.metrics;
  let track_bg = color_of(&c.surface_elevated);
  let track_border = color_of(&c.border);
  let knob_off = color_of(&c.text_muted);
  let (track_bg, track_border, knob_off) = if config.disabled {
    (dim_color(track_bg), dim_color(track_border), dim_color(knob_off))
  } else {
    (track_bg, track_border, knob_off)
  };
  let mut ec = parent.spawn((
    Name::new("ui-toggle-switch"),
    ToggleSwitch,
    UiInteractBundle::default(),
    Node { align_items: AlignItems::Center, column_gap: px(m.spacing.sm), ..default() },
  ));
  ec.with_children(|root| {
    root
      .spawn((
        Name::new("ui-toggle-track"),
        ToggleTrack,
        Node {
          width: px(TRACK_W),
          height: px(TRACK_H),
          border: UiRect::all(px(m.border_width)),
          border_radius: BorderRadius::all(px(m.corner_radius_sm)),
          ..default()
        },
        BackgroundColor(track_bg),
        BorderColor::all(track_border),
      ))
      .with_children(|track| {
        track.spawn((
          Name::new("ui-toggle-knob"),
          ToggleKnob,
          Node {
            position_type: PositionType::Absolute,
            width: px(TRACK_H - m.border_width * 2.0),
            height: px(TRACK_H - m.border_width * 2.0),
            ..default()
          },
          BackgroundColor(knob_off),
        ));
      });
    if let Some(t) = config.text {
      let text_color = color_of(&c.text_body);
      let text_color = if config.disabled { dim_color(text_color) } else { text_color };
      spawn_label(ctx, root, t, m.font_size.md, text_color);
    }
  });
  if config.disabled {
    ec.insert(UiDisabled);
  }
  if config.checked {
    ec.insert(Checked);
  }
  ToggleSwitchHandle(ec.id())
}

type ToggleQuery = (
  Entity,
  &'static Hovered,
  Has<Pressed>,
  &'static mut UiInteractPrev,
  Has<Checked>,
  &'static Children,
  Has<UiDisabled>,
);
type TrackData = (&'static mut BackgroundColor, &'static mut BorderColor, &'static Children);
type TrackFilter = (With<ToggleTrack>, Without<ToggleSwitch>);
type KnobData = (&'static mut Node, &'static mut BackgroundColor);
type KnobFilter = (With<ToggleKnob>, Without<ToggleTrack>);

pub fn toggle_switch_state_system(
  mut commands: Commands,
  theme: Option<Res<UiTheme>>,
  mut q: Query<ToggleQuery, With<ToggleSwitch>>,
  mut tracks: Query<TrackData, TrackFilter>,
  mut knobs: Query<KnobData, KnobFilter>,
) {
  let Some(theme) = theme else { return };
  let c = &theme.colors;
  let accent = color_of(&c.accent_fill);
  let elevated = color_of(&c.surface_elevated);
  let border = color_of(&c.border);
  let border_strong = color_of(&c.border_strong);
  let knob_off = color_of(&c.text_muted);
  let knob_on = color_of(&c.text_primary);
  let left_on = px(TRACK_W - TRACK_H);
  for (e, hovered, pressed, mut prev, checked, children, disabled) in &mut q {
    let inter = UiInteract::of(hovered, pressed);
    if !disabled && prev.0 == UiInteract::Pressed && inter == UiInteract::Hovered {
      if checked {
        commands.entity(e).remove::<Checked>();
      } else {
        commands.entity(e).insert(Checked);
      }
      commands.trigger(ToggleSwitchToggled { entity: e, checked: !checked });
    }
    prev.0 = inter;
    let hovered = !disabled && inter == UiInteract::Hovered;
    let (target_bg, target_border) = if checked {
      (accent, accent)
    } else if hovered {
      (elevated, border_strong)
    } else {
      (elevated, border)
    };
    let target_left = if checked { left_on } else { px(0.0) };
    let target_knob = if checked { knob_on } else { knob_off };
    let (target_bg, target_border, target_knob) = if disabled {
      (dim_color(target_bg), dim_color(target_border), dim_color(target_knob))
    } else {
      (target_bg, target_border, target_knob)
    };
    for child in children.iter() {
      if let Ok((mut bg, mut bc, track_children)) = tracks.get_mut(child) {
        if bg.0 != target_bg {
          bg.0 = target_bg;
        }
        if bc.top != target_border {
          *bc = BorderColor::all(target_border);
        }
        for knob in track_children.iter() {
          if let Ok((mut node, mut knob_bg)) = knobs.get_mut(knob) {
            if node.left != target_left {
              node.left = target_left;
            }
            if knob_bg.0 != target_knob {
              knob_bg.0 = target_knob;
            }
          }
        }
      }
    }
  }
}
