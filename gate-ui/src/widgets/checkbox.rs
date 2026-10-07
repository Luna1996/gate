use std::ops::Deref;

use bevy::picking::hover::Hovered;
use bevy::prelude::*;
use bevy::ui::{Checkable, Checked, Pressed};

use super::{UiCtx, UiDisabled, color_of, dim_color, px, spawn_label};
use crate::pointer::{UiInteract, UiInteractBundle, UiInteractPrev};
use crate::theme::UiTheme;
use crate::widgets::consts::BOX_SIZE;

#[derive(Component, Debug, Default)]
pub struct CheckboxBox;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CheckboxHandle(pub Entity);

impl Deref for CheckboxHandle {
  type Target = Entity;
  fn deref(&self) -> &Entity {
    &self.0
  }
}

impl From<CheckboxHandle> for Entity {
  fn from(h: CheckboxHandle) -> Entity {
    h.0
  }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CheckboxConfig {
  pub text: Option<String>,
  pub checked: bool,
  pub disabled: bool,
}

#[derive(EntityEvent, Clone, Copy, Debug, PartialEq)]
pub struct CheckboxToggled {
  pub entity: Entity,
  pub checked: bool,
}

pub fn checkbox(ctx: &UiCtx, parent: &mut ChildSpawner, config: CheckboxConfig) -> CheckboxHandle {
  let c = &ctx.theme.colors;
  let m = &ctx.theme.metrics;
  let box_bg = color_of(&c.surface_elevated);
  let box_border = color_of(&c.border);
  let (box_bg, box_border) =
    if config.disabled { (dim_color(box_bg), dim_color(box_border)) } else { (box_bg, box_border) };
  let mut ec = parent.spawn((
    Name::new("ui-checkbox"),
    UiInteractBundle::default(),
    Checkable,
    Node { align_items: AlignItems::Center, column_gap: px(m.spacing.sm), ..default() },
  ));
  ec.with_children(|root| {
    root
      .spawn((
        Name::new("ui-checkbox-box"),
        CheckboxBox,
        Node {
          width: px(BOX_SIZE),
          height: px(BOX_SIZE),
          border: UiRect::all(px(m.border_width)),
          border_radius: BorderRadius::all(px(m.corner_radius_sm)),
          align_items: AlignItems::Center,
          justify_content: JustifyContent::Center,
          ..default()
        },
        BackgroundColor(box_bg),
        BorderColor::all(box_border),
      ))
      .with_children(|box_node| {
        box_node.spawn((
          Name::new("ui-checkbox-mark"),
          Node { width: px(BOX_SIZE * 0.5), height: px(BOX_SIZE * 0.5), ..default() },
          BackgroundColor(color_of(&c.text_primary)),
          Visibility::Hidden,
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
  CheckboxHandle(ec.id())
}

type CheckboxQuery = (
  Entity,
  &'static Hovered,
  Has<Pressed>,
  &'static mut UiInteractPrev,
  Has<Checked>,
  &'static Children,
  Has<UiDisabled>,
);

pub fn checkbox_state_system(
  mut commands: Commands,
  theme: Option<Res<UiTheme>>,
  mut q: Query<CheckboxQuery, With<Checkable>>,
  mut boxes: Query<(&mut BackgroundColor, &mut BorderColor, &Children), With<CheckboxBox>>,
  mut marks: Query<&mut Visibility, Without<CheckboxBox>>,
) {
  let Some(theme) = theme else { return };
  let c = &theme.colors;
  let accent = color_of(&c.accent_fill);
  let elevated = color_of(&c.surface_elevated);
  let border = color_of(&c.border);
  let border_strong = color_of(&c.border_strong);
  for (e, hovered, pressed, mut prev, checked, children, disabled) in &mut q {
    let inter = UiInteract::of(hovered, pressed);
    if !disabled && prev.0 == UiInteract::Pressed && inter == UiInteract::Hovered {
      if checked {
        commands.entity(e).remove::<Checked>();
      } else {
        commands.entity(e).insert(Checked);
      }
      commands.trigger(CheckboxToggled { entity: e, checked: !checked });
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
    let (target_bg, target_border) = if disabled {
      (dim_color(target_bg), dim_color(target_border))
    } else {
      (target_bg, target_border)
    };
    for child in children.iter() {
      if let Ok((mut bg, mut bc, box_children)) = boxes.get_mut(child) {
        if bg.0 != target_bg {
          bg.0 = target_bg;
        }
        if bc.top != target_border {
          *bc = BorderColor::all(target_border);
        }
        for mark in box_children.iter() {
          if let Ok(mut vis) = marks.get_mut(mark) {
            let target_vis = if checked { Visibility::Inherited } else { Visibility::Hidden };
            if *vis != target_vis {
              *vis = target_vis;
            }
          }
        }
      }
    }
  }
}
