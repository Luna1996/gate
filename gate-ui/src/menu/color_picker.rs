use bevy::picking::Pickable;
use bevy::picking::hover::Hovered;
use bevy::prelude::*;
use bevy::ui::{ComputedNode, Pressed, UiGlobalTransform};
use bevy::window::PrimaryWindow;

use super::consts::{PALETTE_CELL, PALETTE_EDGE, PALETTE_MARK_BORDER, WINDOW_MARGIN};
use super::items::{MenuColorSwatch, MenuItem, MenuRole};
use super::model::MenuNode;
use super::window::{DebugMenu, split_path};
use crate::capture::MouseIntercept;
use crate::pointer::{UiInteract, UiInteractBundle, UiInteractPrev};
use crate::theme::{ThemeFont, ThemeMetrics, UiTheme};
use crate::widgets::dropdown::anchor_rect;
use crate::widgets::{TextInputValue, UiCtx, UiDisabled, color_of, px};

const COLOR_PICKER_Z: i32 = 900;

const PALETTE_DARK_V: f32 = 0.25;
const PALETTE_TINT_S: f32 = 0.1;

#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ColorPickerOpen(pub bool);

#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub struct ColorPickerPopup {
  pub owner: Entity,
  pub backdrop: Entity,
}

#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub struct ColorPickerBackdrop {
  pub owner: Entity,
}

#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub struct ColorPickerCell {
  pub owner: Entity,
  pub index: usize,
}

pub fn palette_color(index: usize) -> Color {
  let row = index / PALETTE_EDGE;
  let col = index % PALETTE_EDGE;
  let u = col as f32 / (PALETTE_EDGE - 1) as f32;
  if row == 0 {
    let v = 1.0 - u;
    return Color::srgb(v, v, v);
  }
  let hue = (220.0 + (row - 1) as f32 * (240.0 / (PALETTE_EDGE - 2) as f32)) % 360.0;
  let (saturation, value) = if u <= 0.5 {
    (PALETTE_TINT_S + (1.0 - PALETTE_TINT_S) * (u * 2.0), 1.0)
  } else {
    (1.0, 1.0 - (1.0 - PALETTE_DARK_V) * ((u - 0.5) * 2.0))
  };
  Color::hsv(hue, saturation, value)
}

fn hex_of(color: Color) -> String {
  let c = color.to_srgba();
  let ch = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
  format!("{:02X}{:02X}{:02X}", ch(c.red), ch(c.green), ch(c.blue))
}

fn norm_hex(hex: &str) -> Option<String> {
  crate::parse_hex_color(hex).map(|[r, g, b, _]| format!("{r:02X}{g:02X}{b:02X}"))
}

fn marker_color(bg: Color) -> Color {
  let c = bg.to_srgba();
  if 0.299 * c.red + 0.587 * c.green + 0.114 * c.blue > 0.6 { Color::BLACK } else { Color::WHITE }
}

fn palette_size(m: &ThemeMetrics) -> Vec2 {
  let pad = m.spacing.sm + m.border_width;
  let edge = PALETTE_CELL * PALETTE_EDGE as f32 + pad * 2.0;
  Vec2::splat(edge)
}

fn popup_pos(anchor: (Vec2, Vec2), size: Vec2, win: Option<Vec2>, gap: f32) -> Vec2 {
  let (pos, anchor_size) = anchor;
  let mut p = Vec2::new(pos.x + anchor_size.x - size.x, pos.y + anchor_size.y + gap);
  if let Some(win) = win {
    p.x = p.x.clamp(WINDOW_MARGIN, (win.x - size.x - WINDOW_MARGIN).max(WINDOW_MARGIN));
    p.y = p.y.clamp(WINDOW_MARGIN, (win.y - size.y - WINDOW_MARGIN).max(WINDOW_MARGIN));
  }
  p
}

fn ctx_of<'a>(
  theme: &'a Option<Res<UiTheme>>,
  font: &'a Option<Res<ThemeFont>>,
) -> Option<UiCtx<'a>> {
  let theme = &**theme.as_ref()?;
  Some(UiCtx::new(theme, font.as_ref().and_then(|f| f.handle.as_ref())))
}

fn spawn_popup(
  commands: &mut Commands,
  ctx: &UiCtx,
  owner: Entity,
  anchor: (Vec2, Vec2),
  win: Option<Vec2>,
) {
  let m = &ctx.theme.metrics;
  let c = &ctx.theme.colors;
  let size = palette_size(m);
  let pos = popup_pos(anchor, size, win, m.spacing.xs);
  let backdrop = commands
    .spawn((
      Name::new("ui-color-picker-backdrop"),
      ColorPickerBackdrop { owner },
      Node {
        position_type: PositionType::Absolute,
        left: px(0.0),
        top: px(0.0),
        width: Val::Percent(100.0),
        height: Val::Percent(100.0),
        ..default()
      },
      BackgroundColor(Color::NONE),
      GlobalZIndex(COLOR_PICKER_Z - 1),
      MouseIntercept,
      UiInteractBundle::default(),
    ))
    .id();
  let popup = commands
    .spawn((
      Name::new("ui-color-picker-popup"),
      ColorPickerPopup { owner, backdrop },
      Node {
        position_type: PositionType::Absolute,
        left: px(pos.x),
        top: px(pos.y),
        width: px(size.x),
        height: px(size.y),
        flex_direction: FlexDirection::Column,
        border: UiRect::all(px(m.border_width)),
        padding: UiRect::all(px(m.spacing.sm)),
        ..default()
      },
      BackgroundColor(color_of(&c.surface_elevated)),
      BorderColor::all(color_of(&c.border)),
      Pickable::IGNORE,
      GlobalZIndex(COLOR_PICKER_Z),
    ))
    .id();
  for row in 0..PALETTE_EDGE {
    let row_e = commands
      .spawn((
        Name::new("ui-color-picker-row"),
        ChildOf(popup),
        Node { height: px(PALETTE_CELL), flex_direction: FlexDirection::Row, ..default() },
      ))
      .id();
    for col in 0..PALETTE_EDGE {
      let index = row * PALETTE_EDGE + col;
      let bg = palette_color(index);
      commands.spawn((
        Name::new("ui-color-picker-cell"),
        ChildOf(row_e),
        ColorPickerCell { owner, index },
        UiInteractBundle::default(),
        Node {
          flex_basis: px(0.0),
          flex_grow: 1.0,
          height: Val::Percent(100.0),
          border: UiRect::all(px(PALETTE_MARK_BORDER)),
          ..default()
        },
        BackgroundColor(bg),
        BorderColor::all(bg),
        MouseIntercept,
      ));
    }
  }
}

fn close_popup(
  commands: &mut Commands,
  owner: Entity,
  popups: &Query<(Entity, &ColorPickerPopup)>,
  open: &mut ColorPickerOpen,
) {
  for (e, p) in popups.iter() {
    if p.owner == owner {
      commands.entity(e).despawn();
      commands.entity(p.backdrop).despawn();
    }
  }
  open.0 = false;
}

#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn color_picker_system(
  mut commands: Commands,
  windows: Query<&Window, With<PrimaryWindow>>,
  theme: Option<Res<UiTheme>>,
  font: Option<Res<ThemeFont>>,
  mut q_swatches: Query<
    (
      Entity,
      &MenuColorSwatch,
      &Hovered,
      Has<Pressed>,
      &mut UiInteractPrev,
      &mut ColorPickerOpen,
      &ComputedNode,
      &UiGlobalTransform,
      Option<&InheritedVisibility>,
      Has<UiDisabled>,
    ),
    (Without<ColorPickerCell>, Without<ColorPickerBackdrop>, Without<ColorPickerPopup>),
  >,
  mut q_cells: Query<
    (&Hovered, Has<Pressed>, &mut UiInteractPrev, &ColorPickerCell),
    (Without<MenuColorSwatch>, Without<ColorPickerBackdrop>, Without<ColorPickerPopup>),
  >,
  mut q_backdrops: Query<
    (&Hovered, Has<Pressed>, &mut UiInteractPrev, &ColorPickerBackdrop),
    (Without<MenuColorSwatch>, Without<ColorPickerCell>, Without<ColorPickerPopup>),
  >,
  q_popups: Query<(Entity, &ColorPickerPopup)>,
  mut q_inputs: Query<(&MenuItem, &mut TextInputValue)>,
) {
  let win_size = windows.single().ok().map(|w| Vec2::new(w.width(), w.height()));
  let sf = windows.single().ok().map(|w| w.scale_factor()).unwrap_or(1.0);

  let mut picked: Option<(Entity, usize)> = None;
  for (hovered, pressed, mut prev, cell) in &mut q_cells {
    let inter = UiInteract::of(hovered, pressed);
    if prev.0 == UiInteract::Pressed && inter == UiInteract::Hovered {
      picked = Some((cell.owner, cell.index));
    }
    prev.0 = inter;
  }

  let mut clicked_outside: Option<Entity> = None;
  for (hovered, pressed, mut prev, bd) in &mut q_backdrops {
    let inter = UiInteract::of(hovered, pressed);
    if prev.0 == UiInteract::Pressed && inter == UiInteract::Hovered {
      clicked_outside = Some(bd.owner);
    }
    prev.0 = inter;
  }

  for (e, swatch, hovered, pressed, mut prev, mut open, node, xform, visible, disabled) in
    &mut q_swatches
  {
    let inter = UiInteract::of(hovered, pressed);
    let clicked_self = prev.0 == UiInteract::Pressed && inter == UiInteract::Hovered;
    prev.0 = inter;
    if disabled {
      continue;
    }
    if let Some((owner, index)) = picked
      && owner == e
    {
      let hex = hex_of(palette_color(index));
      for (item, mut value) in &mut q_inputs {
        if item.role == MenuRole::Color && item.path == swatch.path && value.0 != hex {
          value.0.clone_from(&hex);
        }
      }
      close_popup(&mut commands, e, &q_popups, &mut open);
      continue;
    }
    if clicked_outside == Some(e) {
      close_popup(&mut commands, e, &q_popups, &mut open);
      continue;
    }
    if clicked_self {
      if open.0 {
        close_popup(&mut commands, e, &q_popups, &mut open);
      } else if let Some(ctx) = ctx_of(&theme, &font) {
        spawn_popup(&mut commands, &ctx, e, anchor_rect(node, xform, sf), win_size);
        open.0 = true;
      }
    } else if open.0 && !visible.map(|v| v.get()).unwrap_or(true) {
      close_popup(&mut commands, e, &q_popups, &mut open);
    }
  }
  for (popup, p) in &q_popups {
    if q_swatches.get(p.owner).is_err() {
      commands.entity(popup).despawn();
      commands.entity(p.backdrop).despawn();
    }
  }
}

#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn color_picker_visual_system(
  theme: Option<Res<UiTheme>>,
  windows: Query<&Window, With<PrimaryWindow>>,
  q_anchors: Query<
    (&ComputedNode, &UiGlobalTransform, Option<&InheritedVisibility>, &MenuColorSwatch),
    Without<ColorPickerPopup>,
  >,
  q_menu: Query<&DebugMenu>,
  mut q_popups: Query<(&ColorPickerPopup, &mut Node)>,
  mut q_cells: Query<(&ColorPickerCell, &Hovered, Has<Pressed>, &mut BorderColor)>,
) {
  let Some(theme) = theme else { return };
  let sf = windows.single().ok().map(|w| w.scale_factor()).unwrap_or(1.0);
  let win_size = windows.single().ok().map(|w| Vec2::new(w.width(), w.height()));
  let size = palette_size(&theme.metrics);
  let border_strong = color_of(&theme.colors.border_strong);

  for (popup, mut node) in &mut q_popups {
    let Ok((anchor_node, anchor_xform, visible, swatch)) = q_anchors.get(popup.owner) else {
      continue;
    };
    if !visible.map(|v| v.get()).unwrap_or(true) {
      continue;
    }
    let pos = popup_pos(
      anchor_rect(anchor_node, anchor_xform, sf),
      size,
      win_size,
      theme.metrics.spacing.xs,
    );
    if node.left != px(pos.x) {
      node.left = px(pos.x);
    }
    if node.top != px(pos.y) {
      node.top = px(pos.y);
    }
    let current =
      q_menu.iter().next().and_then(|m| m.model.node(&split_path(&swatch.path))).and_then(|n| {
        match n {
          MenuNode::Color { hex, .. } => norm_hex(hex),
          _ => None,
        }
      });
    for (cell, hovered, pressed, mut border) in &mut q_cells {
      if cell.owner != popup.owner {
        continue;
      }
      let bg = palette_color(cell.index);
      let selected = current.as_deref() == Some(hex_of(bg).as_str());
      let target = BorderColor::all(if selected {
        marker_color(bg)
      } else if UiInteract::of(hovered, pressed).is_active() {
        border_strong
      } else {
        bg
      });
      if *border != target {
        *border = target;
      }
    }
  }
}
