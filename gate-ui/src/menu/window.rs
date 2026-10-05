use bevy::picking::Pickable;
use bevy::picking::hover::Hovered;
use bevy::prelude::*;
use bevy::ui::{Checked, Pressed};
use bevy::window::PrimaryWindow;

use super::consts::*;
use super::items::{
  MenuColorSwatch, MenuItem, MenuPressPrev, MenuRole, MenuSliderValue, spawn_item,
};
use super::model::{MenuFile, MenuNode};
use crate::capture::MouseIntercept;
use crate::icon::{Icon, IconFont};
use crate::pointer::{UiInteract, UiInteractBundle};
use crate::theme::{ThemeFont, UiTheme};
use crate::widgets::{
  DropdownValue, LabelConfig, LabelOverflow, LabelStyle, SliderValue, TextInputValue, UiCtx,
  color_of, dim_color, label, px, spawn_icon,
};

#[derive(Component, Clone, Copy, Debug, Default, PartialEq)]
pub struct MenuDrag(pub Option<Vec2>);

#[derive(Component)]
pub struct DebugMenu {
  pub model: MenuFile,
  pub path: Vec<String>,
  pub collapsed: bool,
  dragging: bool,
  go: Option<Vec<String>>,
  back: bool,
  reset_pos: bool,
  toggle_collapse: bool,
}

#[derive(Component)]
pub struct MenuPager {
  pub current: Entity,
  pub outgoing: Option<Entity>,
  dir: f32,
  t: f32,
  start_off: f32,
}

#[derive(Component, Clone, Copy, Debug, PartialEq)]
struct MenuCollapse {
  t: f32,
  from: f32,
  to: f32,
  collapsed: bool,
}

#[derive(Component)]
pub struct MenuParts {
  pub title_bar: Entity,
  pub back_btn: Entity,
  pub back_icon: Entity,
  pub path_label: Entity,
  pub reset_btn: Entity,
  pub reset_icon: Entity,
  pub collapse_btn: Entity,
  pub collapse_icon: Entity,
  pub viewport: Entity,
}

#[derive(Component, Debug, Default)]
pub struct DebugMenuRoot;

#[derive(Component, Clone, Debug, PartialEq)]
pub struct MenuPage {
  pub path: Vec<String>,
}

#[derive(EntityEvent, Clone, Debug, PartialEq)]
pub struct MenuActionEvent {
  pub entity: Entity,
  pub path: String,
  pub action: MenuAction,
}

#[derive(Clone, Debug, PartialEq)]
pub enum MenuAction {
  Button(usize),
  Select(usize),
  Toggle(bool),
  Value(f32),
  Text(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DebugMenuHandle {
  pub root: Entity,
}

#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
enum TitleAction {
  Back,
  ResetPosition,
  Collapse,
}

pub fn spawn_debug_menu(world: &mut World, ctx: &UiCtx, model: MenuFile) -> DebugMenuHandle {
  let mut model = model;
  model.sanitize();
  let c = &ctx.theme.colors;
  let m = &ctx.theme.metrics;
  let path = model.window.path.clone();
  let collapsed = model.window.collapsed;
  let (x, y) = (model.window.x, model.window.y);
  let root_path = display_path(&model, &path, |k| ctx.text(k));

  let root = world
    .spawn((
      Name::new("debug-menu"),
      DebugMenuRoot,
      Node {
        position_type: PositionType::Absolute,
        left: px(x),
        top: px(y),
        width: px(PAGE_W + m.border_width * 2.0),
        flex_direction: FlexDirection::Column,
        ..default()
      },
      BackgroundColor(Color::NONE),
    ))
    .id();

  let mut title_bar_e = Entity::PLACEHOLDER;
  let (
    mut back_e,
    mut back_icon,
    mut path_l,
    mut reset_e,
    mut reset_icon,
    mut collapse_e,
    mut collapse_icon,
  ) = (
    Entity::PLACEHOLDER,
    Entity::PLACEHOLDER,
    Entity::PLACEHOLDER,
    Entity::PLACEHOLDER,
    Entity::PLACEHOLDER,
    Entity::PLACEHOLDER,
    Entity::PLACEHOLDER,
  );
  world.entity_mut(root).with_children(|r| {
    let tb = r
      .spawn((
        Name::new("menu-title-bar"),
        Node {
          width: Val::Percent(100.0),
          height: px(TITLE_BAR_H),
          flex_direction: FlexDirection::Row,
          align_items: AlignItems::Center,
          border: UiRect {
            left: px(m.border_width),
            right: px(m.border_width),
            top: px(0.0),
            bottom: px(m.border_width),
          },
          ..default()
        },
        BackgroundColor(color_of(&c.surface_elevated)),
        BorderColor {
          top: Color::NONE,
          right: color_of(&c.border),
          bottom: color_of(&c.border),
          left: color_of(&c.border),
        },
        UiInteractBundle::default(),
        MouseIntercept,
      ))
      .id();
    title_bar_e = tb;
    r.world_mut().entity_mut(tb).with_children(|bar| {
      (back_e, back_icon) = icon_button(ctx, bar, TitleAction::Back, Icon::ChevronLeft);
      path_l = *label(
        ctx,
        bar,
        LabelConfig {
          text: root_path.clone(),
          style: LabelStyle::Muted,
          size: Some(m.font_size.md),
          overflow: LabelOverflow::MiddleEllipsis,
          ..default()
        },
      );
      if let Some(mut n) = bar.world_mut().get_mut::<Node>(path_l) {
        n.flex_grow = 1.0;
        n.margin = UiRect::horizontal(px(m.spacing.sm));
      }
      (reset_e, reset_icon) = icon_button(ctx, bar, TitleAction::ResetPosition, Icon::UndoAlt);
      (collapse_e, collapse_icon) = icon_button(ctx, bar, TitleAction::Collapse, Icon::AngleUp);
    });
  });

  let viewport = world
    .spawn((
      Name::new("menu-viewport"),
      Node {
        width: Val::Percent(100.0),
        flex_direction: FlexDirection::Column,
        overflow: Overflow::clip(),
        ..default()
      },
      BackgroundColor(Color::NONE),
      UiInteractBundle::default(),
      MouseIntercept,
    ))
    .id();
  world.entity_mut(root).add_child(viewport);

  let page = build_page(world, ctx, viewport, &model, &path);

  world.entity_mut(root).insert((
    DebugMenu {
      model,
      path,
      collapsed,
      dragging: false,
      go: None,
      back: false,
      reset_pos: false,
      toggle_collapse: false,
    },
    MenuPager { current: page, outgoing: None, dir: 1.0, t: 1.0, start_off: 0.0 },
    MenuDrag::default(),
    MenuCollapse { t: 1.0, from: 0.0, to: 0.0, collapsed },
    MenuParts {
      title_bar: title_bar_e,
      back_btn: back_e,
      back_icon,
      path_label: path_l,
      reset_btn: reset_e,
      reset_icon,
      collapse_btn: collapse_e,
      collapse_icon,
      viewport,
    },
  ));
  DebugMenuHandle { root }
}

fn icon_button(
  ctx: &UiCtx,
  parent: &mut ChildSpawner,
  action: TitleAction,
  icon: Icon,
) -> (Entity, Entity) {
  let c = &ctx.theme.colors;
  let mut ec = parent.spawn((
    Name::new("menu-title-button"),
    action,
    UiInteractBundle::default(),
    MenuPressPrev::default(),
    Node {
      width: px(TITLE_BAR_H),
      height: px(TITLE_BAR_H),
      align_items: AlignItems::Center,
      justify_content: JustifyContent::Center,
      ..default()
    },
    BackgroundColor(Color::NONE),
    MouseIntercept,
  ));
  let mut icon_e = Entity::PLACEHOLDER;
  ec.with_children(|b| {
    icon_e = spawn_icon(ctx, b, icon.glyph(), TITLE_ICON_SIZE, color_of(&c.text_body));
  });
  (ec.id(), icon_e)
}

fn build_page(
  world: &mut World,
  ctx: &UiCtx,
  viewport: Entity,
  model: &MenuFile,
  path: &[String],
) -> Entity {
  let c = &ctx.theme.colors;
  let m = &ctx.theme.metrics;
  let page = world
    .spawn((
      Name::new("menu-page"),
      MenuPage { path: path.to_vec() },
      Node {
        width: px(page_outer_w(m.border_width)),
        flex_direction: FlexDirection::Column,
        border: UiRect {
          left: px(m.border_width),
          right: px(m.border_width),
          top: px(0.0),
          bottom: px(m.border_width),
        },
        ..default()
      },
      BackgroundColor(color_of(&c.surface_card_hud)),
      BorderColor {
        top: Color::NONE,
        right: color_of(&c.border),
        bottom: color_of(&c.border),
        left: color_of(&c.border),
      },
    ))
    .id();
  let path_str = path.join("/");
  let items: Vec<MenuNode> = model.children_of(path).to_vec();
  world.entity_mut(page).with_children(|p| {
    for node in items.iter() {
      spawn_item(ctx, p, node, &path_str);
    }
  });
  world.entity_mut(viewport).add_child(page);
  page
}

fn clip_nodes<'a>(model: &'a MenuFile, path: &'a [String]) -> Vec<&'a MenuNode> {
  let mut ok: Vec<&MenuNode> = Vec::new();
  let mut cur = model.items.as_slice();
  for seg in path {
    let Some(node) = cur.iter().find(|n| n.id() == seg) else {
      break;
    };
    ok.push(node);
    cur = node.children();
  }
  ok
}

pub(crate) fn clip_path(model: &MenuFile, path: &[String]) -> String {
  let ok: Vec<&str> = clip_nodes(model, path).iter().map(|n| n.id()).collect();
  if ok.is_empty() { "/".to_string() } else { format!("/{}", ok.join("/")) }
}

fn display_path(model: &MenuFile, path: &[String], translate: impl Fn(&str) -> String) -> String {
  let ok: Vec<String> = clip_nodes(model, path).iter().map(|n| translate(n.label())).collect();
  if ok.is_empty() { "/".to_string() } else { format!("/{}", ok.join("/")) }
}

type CtxParts =
  (UiTheme, Option<Handle<Font>>, Option<Handle<Font>>, Option<crate::i18n::TranslatorFn>);

fn ctx_from_world(world: &World) -> Option<CtxParts> {
  let theme = world.get_resource::<UiTheme>()?.clone();
  let font = world.get_resource::<ThemeFont>().and_then(|f| f.handle.clone());
  let icon = world.get_resource::<IconFont>().and_then(|f| f.handle.clone());
  let translate = world.get_resource::<crate::i18n::UiTranslator>().and_then(|t| t.handle());
  Some((theme, font, icon, translate))
}

#[allow(clippy::too_many_lines)]
pub fn menu_system(world: &mut World) {
  let mut q_roots = world.query_filtered::<Entity, With<DebugMenu>>();
  let Ok(root) = q_roots.single(world) else { return };
  let Some(parts) = world.get::<MenuParts>(root).map(|p| MenuParts {
    title_bar: p.title_bar,
    back_btn: p.back_btn,
    back_icon: p.back_icon,
    path_label: p.path_label,
    reset_btn: p.reset_btn,
    reset_icon: p.reset_icon,
    collapse_btn: p.collapse_btn,
    collapse_icon: p.collapse_icon,
    viewport: p.viewport,
  }) else {
    return;
  };

  let mut actions: Vec<(TitleAction, bool)> = Vec::new();
  for (e, action) in [
    (parts.back_btn, TitleAction::Back),
    (parts.reset_btn, TitleAction::ResetPosition),
    (parts.collapse_btn, TitleAction::Collapse),
  ] {
    let clicked = poll_click(world, e);
    actions.push((action, clicked));
  }
  for (action, clicked) in actions {
    if !clicked {
      continue;
    }
    match action {
      TitleAction::Back => {
        let at_root = world.get::<DebugMenu>(root).is_some_and(|m| m.path.is_empty());
        if !at_root && let Some(mut m) = world.get_mut::<DebugMenu>(root) {
          m.back = true;
        }
      }
      TitleAction::ResetPosition => {
        if let Some(mut m) = world.get_mut::<DebugMenu>(root) {
          m.reset_pos = true;
        }
      }
      TitleAction::Collapse => {
        if let Some(mut m) = world.get_mut::<DebugMenu>(root) {
          m.toggle_collapse = true;
        }
      }
    };
  }

  let mut clicks: Vec<(String, MenuRole)> = Vec::new();
  {
    let mut q = world.query::<(Entity, &MenuItem, &Hovered, Has<Pressed>)>();
    let hits: Vec<(Entity, String, MenuRole)> = q
      .iter(world)
      .filter(|(_, _, hovered, pressed)| UiInteract::of(hovered, *pressed).is_active())
      .map(|(e, item, _, _)| (e, item.path.clone(), item.role))
      .collect();
    for (e, path, role) in hits {
      if poll_click(world, e) {
        clicks.push((path, role));
      }
    }
  }

  let mut value_changes: Vec<(String, MenuRole, MenuValue)> = Vec::new();
  {
    let mut q = world.query::<(
      &MenuItem,
      Option<&SliderValue>,
      Has<Checked>,
      Option<&TextInputValue>,
      Option<&DropdownValue>,
    )>();
    let snapshot: Vec<ControlSnapshot> = q
      .iter(world)
      .map(|(item, sv, checked, tv, dv)| {
        (
          item.path.clone(),
          item.role,
          sv.map(|v| v.0),
          Some(checked),
          tv.map(|t| t.0.clone()),
          dv.map(|d| d.0),
        )
      })
      .collect();
    for (path, role, sv, checked, tv, dv) in snapshot {
      match role {
        MenuRole::Slider => {
          if let Some(v) = sv {
            value_changes.push((path, role, MenuValue::Value(v)));
          }
        }
        MenuRole::Toggle => {
          if let Some(b) = checked {
            value_changes.push((path, role, MenuValue::Checked(b)));
          }
        }
        MenuRole::Input(_) | MenuRole::Color => {
          if let Some(t) = tv {
            value_changes.push((path, role, MenuValue::Text(t)));
          }
        }
        MenuRole::Dropdown => {
          if let Some(i) = dv {
            value_changes.push((path, role, MenuValue::Index(i)));
          }
        }
        _ => {}
      }
    }
  }

  let mut menu = world.get_mut::<DebugMenu>(root).expect("DebugMenu 存在");
  let mut events: Vec<MenuActionEvent> = Vec::new();
  for (path, role) in clicks {
    let path_segs = split_path(&path);
    match role {
      MenuRole::SubMenu => {
        menu.go = Some(path_segs);
      }
      MenuRole::Button(i) => events.push(ev(root, &path, MenuAction::Button(i))),
      MenuRole::SwitchOption(i) => {
        if let Some(MenuNode::SwitchGroup { selected, .. }) = menu.model.node_mut(&path_segs)
          && *selected != i
        {
          *selected = i;
          events.push(ev(root, &path, MenuAction::Select(i)));
        }
      }
      _ => {}
    }
  }
  for (path, role, value) in value_changes {
    let path_segs = split_path(&path);
    let Some(node) = menu.model.node_mut(&path_segs) else { continue };
    match (node, role, value) {
      (MenuNode::Slider { value: cur, min, max, .. }, _, MenuValue::Value(v)) => {
        let v = v.clamp(*min, *max);
        if (*cur - v).abs() > f32::EPSILON {
          *cur = v;
          events.push(ev(root, &path, MenuAction::Value(v)));
        }
      }
      (MenuNode::Toggle { checked, .. }, _, MenuValue::Checked(b)) => {
        if *checked != b {
          *checked = b;
          events.push(ev(root, &path, MenuAction::Toggle(b)));
        }
      }
      (MenuNode::Input { fields, .. }, MenuRole::Input(i), MenuValue::Text(t)) => {
        if let Some(f) = fields.get_mut(i)
          && f.text != t
        {
          f.text = t.clone();
          events.push(ev(root, &path, MenuAction::Text(t)));
        }
      }
      (MenuNode::Color { hex, .. }, MenuRole::Color, MenuValue::Text(t)) => {
        if *hex != t {
          *hex = t.clone();
          events.push(ev(root, &path, MenuAction::Text(t)));
        }
      }
      (MenuNode::Dropdown { selected, .. }, _, MenuValue::Index(i)) if *selected != i => {
        *selected = i;
        events.push(ev(root, &path, MenuAction::Select(i)));
      }
      _ => {}
    }
  }
  if menu.toggle_collapse {
    menu.toggle_collapse = false;
    menu.collapsed = !menu.collapsed;
  }
  if menu.reset_pos {
    menu.reset_pos = false;
    menu.model.window.x = DEFAULT_WINDOW_POS.x;
    menu.model.window.y = DEFAULT_WINDOW_POS.y;
  }
  if menu.back {
    menu.back = false;
    if !menu.path.is_empty() {
      let mut p = menu.path.clone();
      p.pop();
      menu.go = Some(p);
    }
  }

  for e in events {
    world.trigger(e);
  }

  let nav = {
    let menu = world.get::<DebugMenu>(root).expect("DebugMenu 存在");
    menu.go.clone()
  };
  if let Some(target) = nav {
    let target = clip_path_segments(world, root, &target);
    start_navigation(world, root, parts.viewport, &target);
  }

  advance_pager(world, root, parts.viewport);

  advance_collapse(world, root, parts.viewport);

  refresh_visuals(world, root, &parts);

  drag_and_clamp(world, root, &parts);
}

enum MenuValue {
  Value(f32),
  Checked(bool),
  Text(String),
  Index(usize),
}

type ControlSnapshot = (String, MenuRole, Option<f32>, Option<bool>, Option<String>, Option<usize>);

fn ev(root: Entity, path: &str, action: MenuAction) -> MenuActionEvent {
  MenuActionEvent { entity: root, path: path.to_string(), action }
}

pub(crate) fn split_path(path: &str) -> Vec<String> {
  path.split('/').filter(|s| !s.is_empty()).map(|s| s.to_string()).collect()
}

fn clip_path_segments(world: &mut World, root: Entity, path: &[String]) -> Vec<String> {
  let model = &world.get::<DebugMenu>(root).expect("DebugMenu 存在").model;
  let mut ok: Vec<String> = Vec::new();
  let mut cur = model.items.as_slice();
  for seg in path {
    let Some(node) = cur.iter().find(|n| n.id() == seg) else { break };
    ok.push(seg.clone());
    cur = node.children();
  }
  ok
}

fn poll_click(world: &mut World, e: Entity) -> bool {
  let hovered = world.get::<Hovered>(e).copied().unwrap_or_default();
  let pressed = world.get::<Pressed>(e).is_some();
  let inter = UiInteract::of(&hovered, pressed);
  let prev = world.get::<MenuPressPrev>(e).copied().unwrap_or_default().0;
  if let Some(mut p) = world.get_mut::<MenuPressPrev>(e) {
    p.0 = inter;
  }
  prev == UiInteract::Pressed && inter == UiInteract::Hovered
}

fn start_navigation(world: &mut World, root: Entity, viewport: Entity, target: &[String]) {
  let dir = {
    let menu = world.get::<DebugMenu>(root).expect("DebugMenu 存在");
    if target.len() >= menu.path.len() { 1.0 } else { -1.0 }
  };
  let Some((theme, font, icon, translate)) = ctx_from_world(world) else { return };
  let ctx =
    UiCtx::new(&theme, font.as_ref()).with_icon_font(icon.as_ref()).with_translate(translate);
  let model = world.get::<DebugMenu>(root).expect("DebugMenu 存在").model.clone();
  let new_page = build_page(world, &ctx, viewport, &model, target);
  let (old_page, stale, from_left) = {
    let p = world.get::<MenuPager>(root).expect("MenuPager 存在");
    let old_page = p.current;
    let from_left = world.get::<Node>(old_page).map_or(0.0, |n| match n.left {
      Val::Px(v) => v,
      _ => 0.0,
    });
    (old_page, p.outgoing, from_left)
  };
  if let Some(e) = stale
    && e != old_page
    && world.get_entity(e).is_ok()
  {
    world.despawn(e);
  }
  let old_h = logical_height(world, old_page);
  let step = page_outer_w(ctx.theme.metrics.border_width);
  for (e, left) in [(old_page, from_left), (new_page, from_left + dir * step)] {
    if let Some(mut n) = world.get_mut::<Node>(e) {
      n.position_type = PositionType::Absolute;
      n.left = px(left);
      n.top = px(0.0);
    }
  }
  if let Some(mut n) = world.get_mut::<Node>(viewport) {
    n.height = px(old_h);
  }
  {
    let mut menu = world.get_mut::<DebugMenu>(root).expect("DebugMenu 存在");
    menu.path = target.to_vec();
    menu.go = None;
  }
  debug!("菜单导航 → {}", clip_path(&model, target));
  if let Some(mut p) = world.get_mut::<MenuPager>(root) {
    p.outgoing = Some(old_page);
    p.current = new_page;
    p.dir = dir;
    p.start_off = from_left;
    p.t = 0.0;
  }
}

fn advance_pager(world: &mut World, root: Entity, viewport: Entity) {
  let (mut t, dir, current, outgoing, start_off) = {
    let Some(p) = world.get::<MenuPager>(root) else { return };
    (p.t, p.dir, p.current, p.outgoing, p.start_off)
  };
  if outgoing.is_none() {
    if let Some(mut n) = world.get_mut::<Node>(current) {
      n.position_type = PositionType::Relative;
      n.left = px(0.0);
    }
    if let Some(mut n) = world.get_mut::<Node>(viewport) {
      n.height = Val::Auto;
    }
    return;
  }
  let dt = world.resource::<Time>().delta_secs();
  t = (t + dt / PAGE_ANIM_SECS).min(1.0);
  let p = ease_in_out(t);
  let step =
    page_outer_w(world.get_resource::<UiTheme>().map_or(0.0, |th| th.metrics.border_width));
  let off = start_off * (1.0 - p);
  let old = outgoing.expect("outgoing 存在");
  if let Some(mut n) = world.get_mut::<Node>(old) {
    n.left = px(off - dir * p * step);
  }
  if let Some(mut n) = world.get_mut::<Node>(current) {
    n.left = px(off + dir * (1.0 - p) * step);
  }
  let h = logical_height(world, old).max(logical_height(world, current));
  if let Some(mut n) = world.get_mut::<Node>(viewport) {
    n.height = px(h);
  }
  if t >= 1.0 {
    world.despawn(old);
    if let Some(mut p) = world.get_mut::<MenuPager>(root) {
      p.outgoing = None;
      p.t = 1.0;
    }
    if let Some(mut n) = world.get_mut::<Node>(current) {
      n.position_type = PositionType::Relative;
      n.left = px(0.0);
    }
    if let Some(mut n) = world.get_mut::<Node>(viewport) {
      n.height = Val::Auto;
    }
  } else if let Some(mut p) = world.get_mut::<MenuPager>(root) {
    p.t = t;
  }
}

fn logical_height(world: &World, e: Entity) -> f32 {
  world.get::<ComputedNode>(e).map(|n| n.size().y * n.inverse_scale_factor).unwrap_or(0.0)
}

fn page_outer_w(border_width: f32) -> f32 {
  PAGE_W + border_width * 2.0
}

fn page_height(world: &World, page: Entity) -> f32 {
  let rows = world.get::<Children>(page).map_or(0.0, |c| c.len() as f32 * ITEM_H);
  let border = world.get::<Node>(page).map_or(0.0, |n| {
    let px_of = |v: Val| if let Val::Px(v) = v { v } else { 0.0 };
    px_of(n.border.top) + px_of(n.border.bottom)
  });
  rows + border
}

fn advance_collapse(world: &mut World, root: Entity, viewport: Entity) {
  let Some(state) = world.get::<MenuCollapse>(root).copied() else { return };
  let collapsed = world.get::<DebugMenu>(root).is_some_and(|m| m.collapsed);

  if state.t >= 1.0 && state.collapsed == collapsed {
    let display = if collapsed { Display::None } else { Display::Flex };
    if let Some(mut n) = world.get_mut::<Node>(viewport)
      && n.display != display
    {
      n.display = display;
    }
    return;
  }
  if state.t >= 1.0 {
    let from = if state.collapsed { 0.0 } else { logical_height(world, viewport) };
    let to = if collapsed {
      0.0
    } else {
      world.get::<MenuPager>(root).map_or(0.0, |p| page_height(world, p.current))
    };
    if let Some(mut n) = world.get_mut::<Node>(viewport) {
      n.display = Display::Flex;
      n.height = px(from);
    }
    if let Some(mut s) = world.get_mut::<MenuCollapse>(root) {
      *s = MenuCollapse { t: 0.0, from, to, collapsed };
    }
    return;
  }
  let dt = world.resource::<Time>().delta_secs();
  let t = (state.t + dt / PAGE_ANIM_SECS).min(1.0);
  let h = state.from + (state.to - state.from) * ease_in_out(t);
  if let Some(mut n) = world.get_mut::<Node>(viewport) {
    n.height = px(h);
  }
  if let Some(mut s) = world.get_mut::<MenuCollapse>(root) {
    s.t = t;
  }
}

fn ease_in_out(t: f32) -> f32 {
  if t < 0.5 { 4.0 * t * t * t } else { 1.0 - (-2.0 * t + 2.0).powi(3) / 2.0 }
}

fn refresh_visuals(world: &mut World, root: Entity, parts: &MenuParts) {
  let (collapsed, path_text, back_disabled) = {
    let menu = world.get::<DebugMenu>(root).expect("DebugMenu 存在");
    let i18n = world.get_resource::<crate::i18n::UiTranslator>();
    let path_text = match i18n {
      Some(t) => display_path(&menu.model, &menu.path, |k| t.resolve(k)),
      None => display_path(&menu.model, &menu.path, |k| k.to_string()),
    };
    (menu.collapsed, path_text, menu.path.is_empty())
  };
  let Some(theme) = world.get_resource::<UiTheme>() else { return };
  let c = &theme.colors;
  let (surface_elevated, surface_overlay, accent_fill, accent_text, border, border_strong) = (
    color_of(&c.surface_elevated),
    color_of(&c.surface_overlay),
    color_of(&c.accent_fill),
    color_of(&c.accent_text),
    color_of(&c.border),
    color_of(&c.border_strong),
  );
  let (text_primary, text_body, text_muted) =
    (color_of(&c.text_primary), color_of(&c.text_body), color_of(&c.text_muted));

  if let Some(mut t) = world.get_mut::<Text>(parts.path_label)
    && t.0 != path_text
  {
    t.0 = path_text.clone();
  }
  if let Some(mut el) = world.get_mut::<crate::widgets::EllipsisText>(parts.path_label)
    && el.full != path_text
  {
    el.full = path_text;
  }
  for (btn, icon, disabled) in [
    (parts.back_btn, parts.back_icon, back_disabled),
    (parts.reset_btn, parts.reset_icon, false),
    (parts.collapse_btn, parts.collapse_icon, false),
  ] {
    let hovered = world.get::<Hovered>(btn).copied().unwrap_or_default();
    let pressed = world.get::<Pressed>(btn).is_some();
    let inter = UiInteract::of(&hovered, pressed);
    let (bg, fg) = if disabled {
      (Color::NONE, dim_color(text_body))
    } else {
      match inter {
        UiInteract::Pressed => (accent_fill, accent_text),
        UiInteract::Hovered => (surface_overlay, text_primary),
        UiInteract::None => (Color::NONE, text_body),
      }
    };
    if let Some(mut b) = world.get_mut::<BackgroundColor>(btn)
      && b.0 != bg
    {
      b.0 = bg;
    }
    let pickable = if disabled { Pickable::IGNORE } else { Pickable::default() };
    if world.get::<Pickable>(btn) != Some(&pickable) {
      world.entity_mut(btn).insert(pickable);
    }
    if let Some(mut t) = world.get_mut::<TextColor>(icon)
      && t.0 != fg
    {
      t.0 = fg;
    }
  }
  let glyph = if collapsed { Icon::AngleDown } else { Icon::AngleUp }.glyph();
  if let Some(mut t) = world.get_mut::<Text>(parts.collapse_icon)
    && t.0 != glyph
  {
    t.0 = glyph.to_string();
  }

  let mut q = world.query::<(Entity, &MenuItem, &Hovered, Has<Pressed>, Option<&Children>)>();
  let rows: Vec<(Entity, String, MenuRole, UiInteract, Vec<Entity>)> = q
    .iter(world)
    .map(|(e, item, hovered, pressed, ch)| {
      (
        e,
        item.path.clone(),
        item.role,
        UiInteract::of(hovered, pressed),
        ch.map(|c| c.iter().collect()).unwrap_or_default(),
      )
    })
    .collect();
  drop(q);
  let mut shared_edges: Vec<(Entity, usize, Entity, u8, Color)> = Vec::new();
  for (e, path, role, inter, children) in rows {
    let hovered = inter.is_active();
    let (bg, border, fg, prio) = match role {
      MenuRole::SubMenu => {
        (if hovered { surface_elevated } else { Color::NONE }, Color::NONE, text_body, 0)
      }
      MenuRole::SwitchOption(i) => {
        let selected = {
          let menu = world.get::<DebugMenu>(root).expect("DebugMenu 存在");
          matches!(menu.model.node(&split_path(&path)), Some(MenuNode::SwitchGroup { selected, .. }) if *selected == i)
        };
        if selected {
          (accent_fill, accent_text, text_primary, 2)
        } else if hovered {
          (surface_overlay, border_strong, text_body, 1)
        } else {
          (surface_elevated, border, text_muted, 0)
        }
      }
      MenuRole::Button(_) => {
        if hovered {
          (surface_overlay, border_strong, text_body, 1)
        } else {
          (surface_elevated, border, text_muted, 0)
        }
      }
      _ => continue,
    };
    if let Some(mut b) = world.get_mut::<BackgroundColor>(e)
      && b.0 != bg
    {
      b.0 = bg;
    }
    if border != Color::NONE
      && let Some(mut b) = world.get_mut::<BorderColor>(e)
    {
      let left = match role {
        MenuRole::SwitchOption(i) if i > 0 => Color::NONE,
        MenuRole::Button(i) if i > 0 => Color::NONE,
        _ => border,
      };
      let target = BorderColor { left, right: border, top: border, bottom: border };
      if *b != target {
        *b = target;
      }
    }
    if let MenuRole::SwitchOption(i) | MenuRole::Button(i) = role
      && let Some(group) = world.get::<ChildOf>(e).map(ChildOf::parent)
    {
      shared_edges.push((group, i, e, prio, border));
    }
    for child in children {
      if let Some(mut t) = world.get_mut::<TextColor>(child)
        && t.0 != fg
      {
        t.0 = fg;
      }
    }
  }
  shared_edges.sort_by_key(|(group, i, ..)| (*group, *i));
  for pair in shared_edges.windows(2) {
    let (group_a, i_a, btn_a, prio_a, _) = pair[0];
    let (group_b, i_b, _, prio_b, border_b) = pair[1];
    if group_a != group_b || i_b != i_a + 1 || prio_b <= prio_a {
      continue;
    }
    if let Some(mut b) = world.get_mut::<BorderColor>(btn_a)
      && b.right != border_b
    {
      b.right = border_b;
    }
  }

  let mut q = world.query::<(Entity, &MenuSliderValue)>();
  let labels: Vec<(Entity, String)> = q.iter(world).map(|(e, v)| (e, v.path.clone())).collect();
  drop(q);
  for (e, path) in labels {
    let text = {
      let menu = world.get::<DebugMenu>(root).expect("DebugMenu 存在");
      match menu.model.node(&split_path(&path)) {
        Some(MenuNode::Slider { value, decimals, .. }) => format!("{value:.*}", *decimals as usize),
        _ => continue,
      }
    };
    if let Some(mut t) = world.get_mut::<Text>(e)
      && t.0 != text
    {
      t.0 = text;
    }
  }

  let mut q = world.query::<(Entity, &MenuItem)>();
  let rows: Vec<(Entity, String, MenuRole)> =
    q.iter(world).map(|(e, item)| (e, item.path.clone(), item.role)).collect();
  drop(q);
  for (control, path, role) in rows {
    if !matches!(role, MenuRole::Slider | MenuRole::Color | MenuRole::Toggle) {
      continue;
    }
    let disabled = world
      .get::<DebugMenu>(root)
      .expect("DebugMenu 存在")
      .model
      .node(&split_path(&path))
      .is_some_and(MenuNode::disabled);
    let Some(row) = world.get::<ChildOf>(control).map(ChildOf::parent) else { continue };
    let Some(children) = world.get::<Children>(row).map(|c| c.iter().collect::<Vec<Entity>>())
    else {
      continue;
    };
    for child in children {
      let base = if world.get::<MenuSliderValue>(child).is_some() { text_muted } else { text_body };
      let target = if disabled { dim_color(base) } else { base };
      if let Some(mut tc) = world.get_mut::<TextColor>(child)
        && tc.0 != target
      {
        tc.0 = target;
      }
    }
  }

  let mut q = world.query::<(Entity, &MenuColorSwatch)>();
  let swatches: Vec<(Entity, String)> = q.iter(world).map(|(e, v)| (e, v.path.clone())).collect();
  drop(q);
  for (e, path) in swatches {
    let (color, disabled) = {
      let menu = world.get::<DebugMenu>(root).expect("DebugMenu 存在");
      match menu.model.node(&split_path(&path)) {
        Some(MenuNode::Color { hex, disabled, .. }) => (
          crate::parse_hex_color(hex)
            .map_or(Color::NONE, |[r, g, b, a]| Color::srgba_u8(r, g, b, a)),
          *disabled,
        ),
        _ => continue,
      }
    };
    let color = if disabled { crate::widgets::dim_color(color) } else { color };
    if let Some(mut b) = world.get_mut::<BackgroundColor>(e)
      && b.0 != color
    {
      b.0 = color;
    }
  }
}

fn drag_and_clamp(world: &mut World, root: Entity, parts: &MenuParts) {
  let (just_pressed, pressed, released) = {
    let mouse = world.resource::<ButtonInput<MouseButton>>();
    (
      mouse.just_pressed(MouseButton::Left),
      mouse.pressed(MouseButton::Left),
      mouse.just_released(MouseButton::Left),
    )
  };
  let cursor = {
    let mut q = world.query_filtered::<&Window, With<PrimaryWindow>>();
    q.iter(world).next().and_then(|w| w.cursor_position())
  };
  if just_pressed {
    let inter_of = |e: Entity| {
      let hovered = world.get::<Hovered>(e).copied().unwrap_or_default();
      UiInteract::of(&hovered, world.get::<Pressed>(e).is_some())
    };
    let on_title = inter_of(parts.title_bar) == UiInteract::Pressed;
    let on_blank = inter_of(parts.viewport) == UiInteract::Pressed;
    if (on_title || on_blank)
      && let Some(mut m) = world.get_mut::<DebugMenu>(root)
    {
      m.dragging = true;
    }
  }
  if released && let Some(mut m) = world.get_mut::<DebugMenu>(root) {
    m.dragging = false;
  }
  let dragging = world.get::<DebugMenu>(root).is_some_and(|m| m.dragging);
  let last = world.get::<MenuDrag>(root).and_then(|d| d.0);
  if dragging
    && pressed
    && let (Some(cur), Some(prev)) = (cursor, last)
    && let Some(mut m) = world.get_mut::<DebugMenu>(root)
  {
    let d = cur - prev;
    if d != Vec2::ZERO {
      m.model.window.x += d.x;
      m.model.window.y += d.y;
    }
  }
  if let Some(mut drag) = world.get_mut::<MenuDrag>(root)
    && drag.0 != cursor
  {
    drag.0 = cursor;
  }
  let win = {
    let mut q = world.query_filtered::<&Window, With<PrimaryWindow>>();
    q.iter(world).next().map(|w| (w.width(), w.height()))
  };
  let Some((win_w, win_h)) = win else { return };
  let menu_size = world
    .get::<ComputedNode>(root)
    .map(|n| n.size() * n.inverse_scale_factor)
    .unwrap_or(Vec2::ZERO);
  {
    let Some(mut m) = world.get_mut::<DebugMenu>(root) else { return };
    let max_x = (win_w - menu_size.x - WINDOW_MARGIN).max(WINDOW_MARGIN);
    let max_y = (win_h - menu_size.y - WINDOW_MARGIN).max(WINDOW_MARGIN);
    m.model.window.x = m.model.window.x.clamp(WINDOW_MARGIN, max_x);
    m.model.window.y = m.model.window.y.clamp(WINDOW_MARGIN, max_y);
    let (x, y) = (m.model.window.x, m.model.window.y);
    if let Some(mut n) = world.get_mut::<Node>(root) {
      n.left = px(x);
      n.top = px(y);
    }
  }
}
