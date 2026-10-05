use std::ops::Deref;

use bevy::picking::hover::Hovered;
use bevy::prelude::*;
use bevy::ui::{Pressed, UiTransform};

use super::{UiCtx, UiDisabled, color_of, dim_color, px, spawn_label};
use crate::pointer::{UiInteract, UiInteractBundle, UiInteractPrev};
use crate::theme::UiTheme;
use crate::widgets::consts::PRESSED_SCALE;

#[derive(EntityEvent, Clone, Copy, Debug, PartialEq)]
pub struct UiClick {
  pub entity: Entity,
}

#[derive(Component, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ButtonVariant {
  #[default]
  Primary,
  Secondary,
  Ghost,
  Danger,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ButtonHandle(pub Entity);

impl Deref for ButtonHandle {
  type Target = Entity;
  fn deref(&self) -> &Entity {
    &self.0
  }
}

impl From<ButtonHandle> for Entity {
  fn from(h: ButtonHandle) -> Entity {
    h.0
  }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ButtonConfig {
  pub text: String,
  pub variant: ButtonVariant,
  pub disabled: bool,
}

pub fn button(ctx: &UiCtx, parent: &mut ChildSpawner, config: ButtonConfig) -> ButtonHandle {
  let m = &ctx.theme.metrics;
  let (bg, border, text_color) = variant_colors(ctx.theme, config.variant, UiInteract::None);
  let (bg, border, text_color) = if config.disabled {
    (dim_color(bg), dim_color(border), dim_color(text_color))
  } else {
    (bg, border, text_color)
  };
  let mut ec = parent.spawn((
    Name::new("ui-button"),
    config.variant,
    UiInteractBundle::default(),
    Node {
      padding: UiRect {
        left: px(m.spacing.md),
        right: px(m.spacing.md),
        top: px(m.spacing.sm),
        bottom: px(m.spacing.sm),
      },
      border: UiRect::all(px(m.border_width)),
      align_items: AlignItems::Center,
      justify_content: JustifyContent::Center,
      ..default()
    },
    BackgroundColor(bg),
    BorderColor::all(border),
    UiTransform::default(),
  ));
  if config.disabled {
    ec.insert(UiDisabled);
  }
  let e = ec
    .with_children(|b| {
      spawn_label(ctx, b, config.text, m.font_size.md, text_color);
    })
    .id();
  ButtonHandle(e)
}

fn variant_colors(
  theme: &UiTheme,
  variant: ButtonVariant,
  inter: UiInteract,
) -> (Color, Color, Color) {
  let c = &theme.colors;
  let none = Color::NONE;
  match variant {
    ButtonVariant::Primary => {
      let bg = match inter {
        UiInteract::Pressed => color_of(&c.accent_fill_pressed),
        UiInteract::Hovered => color_of(&c.accent_fill_hover),
        UiInteract::None => color_of(&c.accent_fill),
      };
      (bg, bg, color_of(&c.text_primary))
    }
    ButtonVariant::Danger => {
      let bg = color_of(&c.danger_fill);
      let border = match inter {
        UiInteract::Hovered => color_of(&c.danger),
        _ => color_of(&c.danger_fill),
      };
      (bg, border, color_of(&c.text_primary))
    }
    ButtonVariant::Secondary => {
      let bg = match inter {
        UiInteract::Pressed => color_of(&c.surface_card),
        UiInteract::Hovered => color_of(&c.surface_overlay),
        UiInteract::None => color_of(&c.surface_elevated),
      };
      let border = match inter {
        UiInteract::None => color_of(&c.border),
        _ => color_of(&c.border_strong),
      };
      (bg, border, color_of(&c.text_body))
    }
    ButtonVariant::Ghost => {
      let bg = match inter {
        UiInteract::Pressed => color_of(&c.surface_card),
        UiInteract::Hovered => color_of(&c.surface_elevated),
        UiInteract::None => none,
      };
      let text = match inter {
        UiInteract::None => color_of(&c.text_muted),
        _ => color_of(&c.text_body),
      };
      (bg, none, text)
    }
  }
}

type ButtonQuery = (
  Entity,
  &'static Hovered,
  Has<Pressed>,
  &'static mut UiInteractPrev,
  &'static mut BackgroundColor,
  &'static mut BorderColor,
  &'static mut UiTransform,
  &'static ButtonVariant,
  &'static Children,
  Has<UiDisabled>,
);

pub fn button_state_system(
  mut commands: Commands,
  theme: Option<Res<UiTheme>>,
  mut q: Query<ButtonQuery>,
  mut q_text: Query<&mut TextColor>,
) {
  let Some(theme) = theme else { return };
  for (e, hovered, pressed, mut prev, mut bg, mut border, mut ui_t, variant, children, disabled) in
    &mut q
  {
    let inter = UiInteract::of(hovered, pressed);
    if disabled {
      let (target_bg, target_border, target_text) =
        variant_colors(&theme, *variant, UiInteract::None);
      let target_bg = dim_color(target_bg);
      let target_border = dim_color(target_border);
      let target_text = dim_color(target_text);
      if bg.0 != target_bg {
        bg.0 = target_bg;
      }
      let target_border = BorderColor::all(target_border);
      if *border != target_border {
        *border = target_border;
      }
      if (ui_t.scale.x - 1.0).abs() > f32::EPSILON {
        ui_t.scale = Vec2::splat(1.0);
      }
      for child in children.iter() {
        if let Ok(mut tc) = q_text.get_mut(child)
          && tc.0 != target_text
        {
          tc.0 = target_text;
        }
      }
      prev.0 = UiInteract::None;
      continue;
    }
    if prev.0 == UiInteract::Pressed && inter == UiInteract::Hovered {
      commands.trigger(UiClick { entity: e });
    }
    prev.0 = inter;
    let (target_bg, target_border, target_text) = variant_colors(&theme, *variant, inter);
    if bg.0 != target_bg {
      bg.0 = target_bg;
    }
    let target_border = BorderColor::all(target_border);
    if *border != target_border {
      *border = target_border;
    }
    let target_scale = if inter == UiInteract::Pressed { PRESSED_SCALE } else { 1.0 };
    if (ui_t.scale.x - target_scale).abs() > f32::EPSILON {
      ui_t.scale = Vec2::splat(target_scale);
    }
    for child in children.iter() {
      if let Ok(mut tc) = q_text.get_mut(child)
        && tc.0 != target_text
      {
        tc.0 = target_text;
      }
    }
  }
}
