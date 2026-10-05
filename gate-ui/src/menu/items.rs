use bevy::picking::Pickable;
use bevy::picking::hover::Hovered;
use bevy::prelude::*;
use bevy::text::{Justify, LineBreak, TextLayout as BevyTextLayout};

use super::color_picker::ColorPickerOpen;
use super::consts::*;
use super::model::{InputField, MenuNode};
use crate::capture::MouseIntercept;
use crate::icon::Icon;
use crate::pointer::{UiInteract, UiInteractBundle};
use crate::widgets::{
  DropdownConfig, LabelConfig, LabelOverflow, LabelStyle, SliderConfig, TextInputConfig,
  TextInputKind, ToggleSwitchConfig, Tooltip, UiCtx, UiDisabled, color_of, dropdown, label, px,
  slider, spawn_icon, text_input, toggle_switch,
};

#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub enum MenuRole {
  SubMenu,
  Button(usize),
  Slider,
  SwitchOption(usize),
  Dropdown,
  Toggle,
  Input(usize),
  Color,
  Text,
}

#[derive(Component, Clone, Debug, PartialEq)]
pub struct MenuItem {
  pub path: String,
  pub role: MenuRole,
}

#[derive(Component, Debug, Default)]
pub struct MenuOptionButton;

#[derive(Component, Debug, Default)]
pub struct MenuSubMenuRow;

#[derive(Component, Clone, Debug, PartialEq)]
pub struct MenuSliderValue {
  pub path: String,
}

#[derive(Component, Clone, Debug, PartialEq)]
pub struct MenuTextValue {
  pub path: String,
}

#[derive(Component, Clone, Debug, PartialEq)]
pub struct MenuColorSwatch {
  pub path: String,
}

#[derive(Component, Clone, Copy, Debug, Default, PartialEq)]
pub struct MenuPressPrev(pub UiInteract);

fn base_row<'a, 'w>(parent: &'a mut ChildSpawner<'w>, name: &str) -> EntityWorldMut<'a> {
  parent.spawn((
    Name::new(name.to_string()),
    Node {
      width: Val::Percent(100.0),
      height: px(ITEM_H),
      flex_direction: FlexDirection::Row,
      align_items: AlignItems::Center,
      padding: UiRect::horizontal(px(ITEM_PAD)),
      ..default()
    },
    BackgroundColor(Color::NONE),
    Hovered::default(),
    Pickable::default(),
  ))
}

fn fixed_label(
  ctx: &UiCtx,
  parent: &mut ChildSpawner,
  key: &str,
  width: f32,
  style: LabelStyle,
  justify: Justify,
) -> Entity {
  let e = *label(
    ctx,
    parent,
    LabelConfig {
      text: ctx.text(key),
      style,
      overflow: LabelOverflow::MiddleEllipsis,
      ..default()
    },
  );
  let mut ec = parent.world_mut().entity_mut(e);
  ec.insert((
    crate::i18n::I18nKey::new(key),
    BevyTextLayout { justify, linebreak: LineBreak::NoWrap },
  ));
  if let Some(mut n) = ec.get_mut::<Node>() {
    n.width = px(width);
  }
  e
}

fn grow_label(ctx: &UiCtx, parent: &mut ChildSpawner, key: &str, style: LabelStyle) -> Entity {
  let e = *label(ctx, parent, LabelConfig { text: ctx.text(key), style, ..default() });
  parent
    .world_mut()
    .entity_mut(e)
    .insert((crate::i18n::I18nKey::new(key), Node { flex_grow: 1.0, ..default() }));
  e
}

fn grow_in_row(world: &mut World, e: Entity) {
  if let Some(mut n) = world.get_mut::<Node>(e) {
    n.flex_grow = 1.0;
  }
}

fn option_button(
  ctx: &UiCtx,
  parent: &mut ChildSpawner,
  key: &str,
  path: &str,
  role: MenuRole,
  first: bool,
) -> Entity {
  let m = &ctx.theme.metrics;
  let mut ec = parent.spawn((
    Name::new("menu-option-button"),
    MenuOptionButton,
    MenuItem { path: path.to_string(), role },
    UiInteractBundle::default(),
    MenuPressPrev::default(),
    Node {
      flex_basis: px(0.0),
      flex_grow: 1.0,
      height: Val::Percent(100.0),
      align_items: AlignItems::Center,
      justify_content: JustifyContent::Center,
      border: UiRect {
        left: px(if first { m.border_width } else { 0.0 }),
        right: px(m.border_width),
        top: px(m.border_width),
        bottom: px(m.border_width),
      },
      ..default()
    },
    BackgroundColor(color_of(&ctx.theme.colors.surface_elevated)),
    BorderColor::all(color_of(&ctx.theme.colors.border)),
    MouseIntercept,
  ));
  ec.with_children(|b| {
    b.spawn((
      Name::new("menu-option-text"),
      bevy::ui::widget::Label,
      crate::i18n::I18nKey::new(key),
      Text::new(ctx.text(key)),
      TextFont {
        font: ctx.font_source(),
        font_size: bevy::text::FontSize::Px(m.font_size.sm),
        ..default()
      },
      TextColor(color_of(&ctx.theme.colors.text_muted)),
      BevyTextLayout { linebreak: LineBreak::NoWrap, ..default() },
    ));
  });
  ec.id()
}

pub(crate) fn spawn_item(
  ctx: &UiCtx,
  parent: &mut ChildSpawner,
  node: &MenuNode,
  parent_path: &str,
) -> Entity {
  let path = join_path(parent_path, node.id());
  let row = match node {
    MenuNode::SubMenu { label: key, .. } => sub_menu_row(ctx, parent, key, &path),
    MenuNode::Buttons { label: key, items, .. } => buttons_row(ctx, parent, key, items, &path),
    MenuNode::Slider { label: key, value, min, max, step, decimals, disabled, .. } => slider_row(
      ctx,
      parent,
      key,
      SliderConfig {
        min: *min,
        max: *max,
        value: *value,
        step: (*step > 0.0).then_some(*step),
        disabled: *disabled,
      },
      *decimals,
      &path,
    ),
    MenuNode::SwitchGroup { label: key, options, .. } => {
      switch_group_row(ctx, parent, key, options, &path)
    }
    MenuNode::Dropdown { label: key, options, selected, .. } => {
      dropdown_row(ctx, parent, key, options, *selected, &path)
    }
    MenuNode::Toggle { label: key, checked, disabled, .. } => {
      toggle_row(ctx, parent, key, *checked, *disabled, &path)
    }
    MenuNode::Input { label: key, fields, .. } => input_row(ctx, parent, key, fields, &path),
    MenuNode::Color { label: key, hex, disabled, .. } => {
      color_row(ctx, parent, key, hex, &path, *disabled)
    }
    MenuNode::Text { text: key, .. } => text_row(ctx, parent, key, &path),
  };
  if let Some(key) = node.tooltip() {
    parent
      .world_mut()
      .entity_mut(row)
      .insert((Tooltip::new(ctx.text(key)), crate::i18n::I18nKey::new(key)));
  }
  row
}

pub fn join_path(parent: &str, child: &str) -> String {
  if parent.is_empty() { child.to_string() } else { format!("{parent}/{child}") }
}

fn sub_menu_row(ctx: &UiCtx, parent: &mut ChildSpawner, key: &str, path: &str) -> Entity {
  let mut ec = base_row(parent, "menu-sub-menu");
  let row = ec.id();
  ec.insert((
    MenuItem { path: path.to_string(), role: MenuRole::SubMenu },
    MenuSubMenuRow,
    UiInteractBundle::default(),
    MenuPressPrev::default(),
    MouseIntercept,
  ));
  ec.with_children(|r| {
    grow_label(ctx, r, key, LabelStyle::Body);
    spawn_icon(
      ctx,
      r,
      Icon::ChevronRight.glyph(),
      TITLE_ICON_SIZE,
      color_of(&ctx.theme.colors.text_muted),
    );
  });
  row
}

fn buttons_row(
  ctx: &UiCtx,
  parent: &mut ChildSpawner,
  key: &str,
  items: &[String],
  path: &str,
) -> Entity {
  let mut ec = base_row(parent, "menu-buttons");
  let row = ec.id();
  ec.with_children(|r| {
    if !key.is_empty() {
      fixed_label(ctx, r, key, LEFT_COL_W, LabelStyle::Body, Justify::Left);
    }
    let mut group = r.spawn(Node {
      flex_grow: 1.0,
      height: px(CTRL_H),
      flex_direction: FlexDirection::Row,
      column_gap: px(BUTTON_GAP),
      ..default()
    });
    group.with_children(|g| {
      for (i, name) in items.iter().enumerate() {
        option_button(ctx, g, name, path, MenuRole::Button(i), i == 0);
      }
    });
  });
  row
}

fn slider_row(
  ctx: &UiCtx,
  parent: &mut ChildSpawner,
  key: &str,
  config: SliderConfig,
  decimals: u32,
  path: &str,
) -> Entity {
  let value = config.value;
  let disabled = config.disabled;
  let mut ec = base_row(parent, "menu-slider");
  let row = ec.id();
  ec.with_children(|r| {
    let name = fixed_label(ctx, r, key, LEFT_COL_W, LabelStyle::Body, Justify::Left);
    let se = *slider(ctx, r, config);
    grow_in_row(r.world_mut(), se);
    r.world_mut()
      .entity_mut(se)
      .insert(MenuItem { path: path.to_string(), role: MenuRole::Slider });
    let ve = *label(
      ctx,
      r,
      LabelConfig {
        text: format!("{value:.*}", decimals as usize),
        style: LabelStyle::Muted,
        ..default()
      },
    );
    r.world_mut().entity_mut(ve).insert((
      MenuSliderValue { path: path.to_string() },
      Node { width: px(RIGHT_COL_W), ..default() },
      BevyTextLayout { justify: Justify::Right, ..default() },
    ));
    if disabled {
      for e in [name, ve] {
        dim_text(r.world_mut(), e);
      }
    }
  });
  row
}

fn dim_text(world: &mut World, e: Entity) {
  if let Some(mut c) = world.get_mut::<TextColor>(e) {
    c.0 = crate::widgets::dim_color(c.0);
  }
}

fn switch_group_row(
  ctx: &UiCtx,
  parent: &mut ChildSpawner,
  key: &str,
  options: &[String],
  path: &str,
) -> Entity {
  let mut ec = base_row(parent, "menu-switch-group");
  let row = ec.id();
  ec.with_children(|r| {
    if !key.is_empty() {
      fixed_label(ctx, r, key, LEFT_COL_W, LabelStyle::Body, Justify::Left);
    }
    let mut group = r.spawn(Node {
      flex_grow: 1.0,
      height: px(CTRL_H),
      flex_direction: FlexDirection::Row,
      column_gap: px(0.0),
      ..default()
    });
    group.with_children(|g| {
      for (i, name) in options.iter().enumerate() {
        option_button(ctx, g, name, path, MenuRole::SwitchOption(i), i == 0);
      }
    });
  });
  row
}

fn dropdown_row(
  ctx: &UiCtx,
  parent: &mut ChildSpawner,
  key: &str,
  options: &[String],
  selected: usize,
  path: &str,
) -> Entity {
  let mut ec = base_row(parent, "menu-dropdown");
  let row = ec.id();
  ec.with_children(|r| {
    fixed_label(ctx, r, key, LEFT_COL_W, LabelStyle::Body, Justify::Left);
    let de =
      *dropdown(ctx, r, DropdownConfig { options: options.to_vec(), selected, disabled: false });
    grow_in_row(r.world_mut(), de);
    r.world_mut()
      .entity_mut(de)
      .insert(MenuItem { path: path.to_string(), role: MenuRole::Dropdown });
  });
  row
}

fn toggle_row(
  ctx: &UiCtx,
  parent: &mut ChildSpawner,
  key: &str,
  checked: bool,
  disabled: bool,
  path: &str,
) -> Entity {
  let mut ec = base_row(parent, "menu-toggle");
  let row = ec.id();
  ec.with_children(|r| {
    let name = grow_label(ctx, r, key, LabelStyle::Body);
    let te = *toggle_switch(ctx, r, ToggleSwitchConfig { text: None, checked, disabled });
    r.world_mut()
      .entity_mut(te)
      .insert(MenuItem { path: path.to_string(), role: MenuRole::Toggle });
    if disabled {
      dim_text(r.world_mut(), name);
    }
  });
  row
}

fn input_row(
  ctx: &UiCtx,
  parent: &mut ChildSpawner,
  key: &str,
  fields: &[InputField],
  path: &str,
) -> Entity {
  let mut ec = base_row(parent, "menu-input");
  let row = ec.id();
  ec.with_children(|r| {
    fixed_label(ctx, r, key, LEFT_COL_W, LabelStyle::Body, Justify::Left);
    for (i, f) in fields.iter().enumerate() {
      if !f.label.is_empty() {
        let fe = *label(
          ctx,
          r,
          LabelConfig { text: ctx.text(&f.label), style: LabelStyle::Muted, ..default() },
        );
        r.world_mut().entity_mut(fe).insert(crate::i18n::I18nKey::new(&f.label));
      }
      let ie = *text_input(
        ctx,
        r,
        TextInputConfig { text: f.text.clone(), kind: f.kind(), disabled: false },
      );
      grow_in_row(r.world_mut(), ie);
      r.world_mut()
        .entity_mut(ie)
        .insert(MenuItem { path: path.to_string(), role: MenuRole::Input(i) });
    }
  });
  row
}

fn color_row(
  ctx: &UiCtx,
  parent: &mut ChildSpawner,
  key: &str,
  hex: &str,
  path: &str,
  disabled: bool,
) -> Entity {
  let m = &ctx.theme.metrics;
  let mut ec = base_row(parent, "menu-color");
  let row = ec.id();
  ec.with_children(|r| {
    let name = fixed_label(ctx, r, key, LEFT_COL_W, LabelStyle::Body, Justify::Left);
    let ie = *text_input(
      ctx,
      r,
      TextInputConfig { text: hex.to_string(), kind: TextInputKind::Text, disabled },
    );
    grow_in_row(r.world_mut(), ie);
    r.world_mut().entity_mut(ie).insert(MenuItem { path: path.to_string(), role: MenuRole::Color });
    let swatch = r
      .spawn((
        Name::new("menu-color-swatch"),
        MenuColorSwatch { path: path.to_string() },
        ColorPickerOpen::default(),
        UiInteractBundle::default(),
        Node {
          width: px(RIGHT_COL_W),
          height: px(CTRL_H),
          border: UiRect::all(px(m.border_width)),
          margin: UiRect::left(px(m.spacing.sm)),
          ..default()
        },
        BackgroundColor(
          crate::parse_hex_color(hex)
            .map_or(Color::NONE, |[r_, g, b, a]| Color::srgba_u8(r_, g, b, a)),
        ),
        BorderColor::all(color_of(&ctx.theme.colors.border)),
        MouseIntercept,
      ))
      .id();
    if disabled {
      dim_text(r.world_mut(), name);
      r.world_mut().entity_mut(swatch).insert(UiDisabled);
    }
  });
  row
}

fn text_row(ctx: &UiCtx, parent: &mut ChildSpawner, key: &str, path: &str) -> Entity {
  let mut ec = base_row(parent, "menu-text");
  let row = ec.id();
  ec.insert(MenuItem { path: path.to_string(), role: MenuRole::Text });
  ec.with_children(|r| {
    fixed_label(ctx, r, key, LEFT_COL_W, LabelStyle::Body, Justify::Left);
    let ve =
      *label(ctx, r, LabelConfig { text: String::new(), style: LabelStyle::Body, ..default() });
    r.world_mut().entity_mut(ve).insert((
      MenuTextValue { path: path.to_string() },
      Node { flex_grow: 1.0, ..default() },
      BevyTextLayout { justify: Justify::Right, ..default() },
    ));
  });
  row
}
