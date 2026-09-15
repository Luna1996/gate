//! DebugWindow 容器：标题栏 + 拖拽 + ViewPager 分页动画 + 收起/展开。
//!
//! 结构：
//! ```text
//! root (DebugMenu, 绝对定位)
//! ├── title_bar (返回 | 路径 | 重置位置 | 收起/展开)
//! ── viewport (overflow clip, 背景透明)
//!     ── page × 1..2（切换动画期间新旧两页并排滑动）
//! ```
//!
//! [`menu_system`] 是唯一的交互/布局驱动（独占系统）：读控件状态 → 写回模型
//! （[`MenuFile`] 永远是真源）→ 发 [`MenuActionEvent`] → 刷新视觉 → 推进分页动画。

use bevy::input::mouse::AccumulatedMouseMotion;
use bevy::prelude::*;
use bevy::ui::{Checked, FocusPolicy, Interaction};
use bevy::window::PrimaryWindow;

use super::consts::*;
use super::items::{
  MenuColorSwatch, MenuItem, MenuPressPrev, MenuRole, MenuSliderValue, spawn_item,
};
use super::model::{MenuFile, MenuNode};
use crate::capture::MouseIntercept;
use crate::icon::{Icon, IconFont};
use crate::theme::{ThemeFont, UiTheme};
use crate::widgets::{
  LabelConfig, LabelOverflow, LabelStyle, SliderValue, TextInputValue, UiCtx, color_of, label, px,
  spawn_icon,
};

/// 窗口拖拽的上一帧光标位置（逻辑 px；`None` = 指针不在窗口内）。
///
/// 不用 `AccumulatedMouseMotion`（依赖 winit 的原始设备事件，合成/远程输入下可能恒为 0），
/// 直接用光标位置差 —— 与鼠标事件来源无关，行为稳定。
#[derive(Component, Clone, Copy, Debug, Default, PartialEq)]
pub struct MenuDrag(pub Option<Vec2>);

/// 菜单根（DebugWindow）
#[derive(Component)]
pub struct DebugMenu {
  /// 模型（层级 + 全部控件状态；持久化直接取这份）
  pub model: MenuFile,
  /// 当前节点 id 路径（空 = 根）
  pub path: Vec<String>,
  pub collapsed: bool,
  /// 窗口拖拽中
  dragging: bool,
  /// 待进入的路径
  go: Option<Vec<String>>,
  /// 待返回上级
  back: bool,
  /// 待重置窗口位置
  reset_pos: bool,
  /// 待收起/展开
  toggle_collapse: bool,
}

/// 分页动画状态
#[derive(Component)]
pub struct MenuPager {
  /// 当前页实体
  pub current: Entity,
  /// 切换动画中的旧页（动画结束即销毁）
  pub outgoing: Option<Entity>,
  /// 方向：+1 = 进入下级（新页自右入），-1 = 返回上级
  dir: f32,
  /// 进度 0..1
  t: f32,
}

/// 容器各部位实体句柄（menu_system 定位用）
#[derive(Component)]
pub struct MenuParts {
  pub title_bar: Entity,
  pub back_btn: Entity,
  pub path_label: Entity,
  pub reset_btn: Entity,
  pub collapse_btn: Entity,
  pub collapse_icon: Entity,
  pub viewport: Entity,
}

/// 窗口根标记（外部按此定位菜单；也是唯一根）
#[derive(Component, Debug, Default)]
pub struct DebugMenuRoot;

/// 菜单页（一页 = 一个菜单节点的子项列表）
#[derive(Component, Clone, Debug, PartialEq)]
pub struct MenuPage {
  pub path: Vec<String>,
}

/// 菜单整体事件（统一回调接口：一个事件 + 路径 + 动作）
#[derive(EntityEvent, Clone, Debug, PartialEq)]
pub struct MenuActionEvent {
  pub entity: Entity,
  /// 节点 id 路径（如 "render/ddgi/probe_viz"）
  pub path: String,
  pub action: MenuAction,
}

/// 菜单动作
#[derive(Clone, Debug, PartialEq)]
pub enum MenuAction {
  /// 按钮组第 i 个按钮被点击
  Button(usize),
  /// 切换组选中第 i 项
  Select(usize),
  /// 开关项翻转
  Toggle(bool),
  /// 滑动条值变化
  Value(f32),
  /// 输入框/颜色文本变化（提交后）
  Text(String),
}

/// 菜单句柄
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DebugMenuHandle {
  pub root: Entity,
}

/// 标题栏图标按钮动作
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
enum TitleAction {
  Back,
  ResetPosition,
  Collapse,
}

/// 建菜单容器 + 按模型建出当前路径的一页；返回根实体句柄
pub fn spawn_debug_menu(world: &mut World, ctx: &UiCtx, model: MenuFile) -> DebugMenuHandle {
  let mut model = model;
  model.sanitize();
  let c = &ctx.theme.colors;
  let m = &ctx.theme.metrics;
  let path = model.window.path.clone();
  let collapsed = model.window.collapsed;
  let (x, y) = (model.window.x, model.window.y);
  let root_path = clip_path(&model, &path);

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
        border: UiRect::all(px(m.border_width)),
        ..default()
      },
      BackgroundColor(color_of(&c.surface_card_hud)),
      BorderColor::all(color_of(&c.border)),
    ))
    .id();

  // ---- 标题栏 ----
  let mut title_bar_e = Entity::PLACEHOLDER;
  let (mut back_e, mut path_l, mut reset_e, mut collapse_e, mut collapse_icon) = (
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
          border: UiRect::bottom(px(m.border_width)),
          ..default()
        },
        BackgroundColor(color_of(&c.surface_elevated)),
        BorderColor { bottom: color_of(&c.border), ..BorderColor::DEFAULT },
        Interaction::default(),
        // 标题栏空白处可拖动窗口
        FocusPolicy::Block,
        MouseIntercept,
      ))
      .id();
    title_bar_e = tb;
    r.world_mut().entity_mut(tb).with_children(|bar| {
      back_e = icon_button(ctx, bar, TitleAction::Back, Icon::ChevronLeft);
      // 路径标签：占满中间，过长中间省略
      path_l = *label(
        ctx,
        bar,
        LabelConfig {
          text: root_path.clone(),
          style: LabelStyle::Muted,
          overflow: LabelOverflow::MiddleEllipsis,
          ..default()
        },
      );
      bar.world_mut().entity_mut(path_l).insert(Node { flex_grow: 1.0, ..default() });
      reset_e = icon_button(ctx, bar, TitleAction::ResetPosition, Icon::UndoAlt);
      collapse_e = icon_button(ctx, bar, TitleAction::Collapse, Icon::Compress);
      collapse_icon = bar
        .world_mut()
        .get::<Children>(collapse_e)
        .and_then(|ch| ch.first().copied())
        .unwrap_or(Entity::PLACEHOLDER);
    });
  });

  // ---- 内容视口 ----
  let viewport = world
    .spawn((
      Name::new("menu-viewport"),
      Node {
        width: Val::Percent(100.0),
        flex_direction: FlexDirection::Column,
        overflow: Overflow::clip(),
        // 背景透明 + clip 子元素
        ..default()
      },
      BackgroundColor(Color::NONE),
      Interaction::default(),
      FocusPolicy::Block,
      MouseIntercept,
    ))
    .id();
  world.entity_mut(root).add_child(viewport);

  // ---- 当前页 ----
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
    MenuPager { current: page, outgoing: None, dir: 1.0, t: 1.0 },
    MenuDrag::default(),
    MenuParts {
      title_bar: title_bar_e,
      back_btn: back_e,
      path_label: path_l,
      reset_btn: reset_e,
      collapse_btn: collapse_e,
      collapse_icon,
      viewport,
    },
  ));
  DebugMenuHandle { root }
}

/// 标题栏图标按钮（无文字，只有 FontAwesome 字形）
fn icon_button(ctx: &UiCtx, parent: &mut ChildSpawner, action: TitleAction, icon: Icon) -> Entity {
  let c = &ctx.theme.colors;
  let mut ec = parent.spawn((
    Name::new("menu-title-button"),
    action,
    Interaction::default(),
    MenuPressPrev::default(),
    Node {
      width: px(TITLE_ICON_W),
      height: px(TITLE_BAR_H),
      align_items: AlignItems::Center,
      justify_content: JustifyContent::Center,
      ..default()
    },
    BackgroundColor(Color::NONE),
    FocusPolicy::Block,
    MouseIntercept,
  ));
  ec.with_children(|b| {
    spawn_icon(ctx, b, icon.glyph(), TITLE_ICON_SIZE, color_of(&c.text_body));
  });
  ec.id()
}

/// 按模型建一页（节点子树 + 直接挂到 viewport 下）
fn build_page(
  world: &mut World,
  ctx: &UiCtx,
  viewport: Entity,
  model: &MenuFile,
  path: &[String],
) -> Entity {
  let page = world
    .spawn((
      Name::new("menu-page"),
      MenuPage { path: path.to_vec() },
      Node { width: px(PAGE_W), flex_direction: FlexDirection::Column, ..default() },
      BackgroundColor(Color::NONE),
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

/// 剪掉无效路径段（TOML 被手改后仍能建出 UI）
pub(crate) fn clip_path(model: &MenuFile, path: &[String]) -> String {
  let mut ok: Vec<&str> = Vec::new();
  let mut cur = model.items.as_slice();
  for seg in path {
    let Some(node) = cur.iter().find(|n| n.id() == seg) else {
      break;
    };
    ok.push(node.id());
    cur = node.children();
  }
  if ok.is_empty() { "/".to_string() } else { format!("/{}", ok.join("/")) }
}

/// 世界 → UiCtx（导航建页时需要；主题/字体/解析器句柄 clone 一次，成本可忽略）
fn ctx_from_world(world: &World) -> Option<(UiTheme, Option<Handle<Font>>, Option<Handle<Font>>, Option<crate::i18n::TranslatorFn>)> {
  let theme = world.get_resource::<UiTheme>()?.clone();
  let font = world.get_resource::<ThemeFont>().and_then(|f| f.handle.clone());
  let icon = world.get_resource::<IconFont>().and_then(|f| f.handle.clone());
  let translate = world.get_resource::<crate::i18n::UiTranslator>().and_then(|t| t.handle());
  Some((theme, font, icon, translate))
}

/// 菜单主系统：交互 → 模型 → 事件 → 视觉 → 分页动画 → 窗口约束（独占系统）。
#[allow(clippy::too_many_lines)] // 单一职责但步骤线性，拆开反而割裂状态
pub fn menu_system(world: &mut World) {
  let mut q_roots = world.query_filtered::<Entity, With<DebugMenu>>();
  let Ok(root) = q_roots.single(world) else { return };
  let Some(parts) = world.get::<MenuParts>(root).map(|p| MenuParts {
    title_bar: p.title_bar,
    back_btn: p.back_btn,
    path_label: p.path_label,
    reset_btn: p.reset_btn,
    collapse_btn: p.collapse_btn,
    collapse_icon: p.collapse_icon,
    viewport: p.viewport,
  }) else {
    return;
  };

  // ---------- 1. 标题栏按钮 ----------
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
      TitleAction::Back => world.get_mut::<DebugMenu>(root).map(|mut m| m.back = true),
      TitleAction::ResetPosition => {
        world.get_mut::<DebugMenu>(root).map(|mut m| m.reset_pos = true)
      }
      TitleAction::Collapse => {
        world.get_mut::<DebugMenu>(root).map(|mut m| m.toggle_collapse = true)
      }
    };
  }

  // ---------- 2. 菜单项点击（子菜单 / 按钮组 / 切换组） ----------
  let mut clicks: Vec<(String, MenuRole)> = Vec::new();
  {
    let mut q = world.query::<(Entity, &MenuItem, &Interaction)>();
    let hits: Vec<(Entity, String, MenuRole)> = q
      .iter(world)
      .filter(|(_, _, inter)| matches!(inter, Interaction::Hovered | Interaction::Pressed))
      .map(|(e, item, _)| (e, item.path.clone(), item.role))
      .collect();
    for (e, path, role) in hits {
      if poll_click(world, e) {
        clicks.push((path, role));
      }
    }
  }

  // ---------- 3. 控件值同步（滑杆 / 开关 / 输入框 / 颜色） ----------
  let mut value_changes: Vec<(String, MenuRole, MenuValue)> = Vec::new();
  {
    let mut q =
      world.query::<(&MenuItem, Option<&SliderValue>, Has<Checked>, Option<&TextInputValue>)>();
    let snapshot: Vec<(String, MenuRole, Option<f32>, Option<bool>, Option<String>)> = q
      .iter(world)
      .map(|(item, sv, checked, tv)| {
        (item.path.clone(), item.role, sv.map(|v| v.0), Some(checked), tv.map(|t| t.0.clone()))
      })
      .collect();
    for (path, role, sv, checked, tv) in snapshot {
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
        _ => {}
      }
    }
  }

  // ---------- 4. 写回模型 + 发事件 ----------
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
      _ => {}
    }
  }
  // 收起/展开
  if menu.toggle_collapse {
    menu.toggle_collapse = false;
    menu.collapsed = !menu.collapsed;
  }
  // 重置窗口位置
  if menu.reset_pos {
    menu.reset_pos = false;
    menu.model.window.x = DEFAULT_WINDOW_POS.x;
    menu.model.window.y = DEFAULT_WINDOW_POS.y;
  }
  // 返回上级
  if menu.back {
    menu.back = false;
    if !menu.path.is_empty() {
      let mut p = menu.path.clone();
      p.pop();
      menu.go = Some(p);
    }
  }
  drop(menu);

  for e in events {
    world.trigger(e);
  }

  // ---------- 5. 导航 ----------
  let nav = {
    let menu = world.get::<DebugMenu>(root).expect("DebugMenu 存在");
    menu.go.clone()
  };
  if let Some(target) = nav {
    let target = clip_path_segments(world, root, &target);
    start_navigation(world, root, parts.viewport, &target);
  }

  // ---------- 6. 分页动画 ----------
  advance_pager(world, root, parts.viewport);

  // ---------- 7. 视觉刷新 ----------
  refresh_visuals(world, root, &parts);

  // ---------- 8. 窗口拖拽 + 边界约束 ----------
  drag_and_clamp(world, root, &parts);
}

/// 控件值快照
enum MenuValue {
  Value(f32),
  Checked(bool),
  Text(String),
}

fn ev(root: Entity, path: &str, action: MenuAction) -> MenuActionEvent {
  MenuActionEvent { entity: root, path: path.to_string(), action }
}

/// 路径字符串 → 段
pub(crate) fn split_path(path: &str) -> Vec<String> {
  path.split('/').filter(|s| !s.is_empty()).map(|s| s.to_string()).collect()
}

/// 只保留模型里真实存在的路径段
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

/// 读 Interaction 并把 prev 推进一帧；返回是否「按下后在同节点上释放」（= 点击）
fn poll_click(world: &mut World, e: Entity) -> bool {
  let inter = world.get::<Interaction>(e).copied().unwrap_or(Interaction::None);
  let prev = world.get::<MenuPressPrev>(e).copied().unwrap_or_default().0;
  if let Some(mut p) = world.get_mut::<MenuPressPrev>(e) {
    p.0 = inter;
  }
  prev == Interaction::Pressed && inter == Interaction::Hovered
}

/// 开始一次页面切换：建新页、旧页脱流、启动动画
fn start_navigation(world: &mut World, root: Entity, viewport: Entity, target: &[String]) {
  let dir = {
    let menu = world.get::<DebugMenu>(root).expect("DebugMenu 存在");
    if target.len() >= menu.path.len() { 1.0 } else { -1.0 }
  };
  let Some((theme, font, icon, translate)) = ctx_from_world(world) else { return };
  let ctx = UiCtx::new(&theme, font.as_ref())
    .with_icon_font(icon.as_ref())
    .with_translate(translate);
  let model = world.get::<DebugMenu>(root).expect("DebugMenu 存在").model.clone();
  let new_page = build_page(world, &ctx, viewport, &model, target);
  let old_page = world.get::<MenuPager>(root).expect("MenuPager 存在").current;
  // 旧页高度作为动画起始高度（新页布局完成后下一帧再取 max）
  let old_h = logical_height(world, old_page);
  for (e, left) in [(old_page, 0.0f32), (new_page, dir * PAGE_W)] {
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
  info!("菜单导航 → {}", clip_path(&model, target));
  if let Some(mut p) = world.get_mut::<MenuPager>(root) {
    p.outgoing = Some(old_page);
    p.current = new_page;
    p.dir = dir;
    p.t = 0.0;
  }
}

/// 推进分页动画：两页并排滑动；窗口内容高度取两者较大者
fn advance_pager(world: &mut World, root: Entity, viewport: Entity) {
  let (mut t, dir, current, outgoing) = {
    let Some(p) = world.get::<MenuPager>(root) else { return };
    (p.t, p.dir, p.current, p.outgoing)
  };
  if outgoing.is_none() {
    // 静止态：当前页走文档流，视口高度 auto（窗口高度随内容）
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
  let old = outgoing.expect("outgoing 存在");
  if let Some(mut n) = world.get_mut::<Node>(old) {
    n.left = px(-dir * p * PAGE_W);
  }
  if let Some(mut n) = world.get_mut::<Node>(current) {
    n.left = px(dir * (1.0 - p) * PAGE_W);
  }
  // 内容高度 = 两页较高者（页面为绝对定位，高度取上帧布局结果）
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

/// 节点逻辑高度（ComputedNode 是物理 px；乘 inverse_scale_factor 得逻辑 px）
fn logical_height(world: &World, e: Entity) -> f32 {
  world.get::<ComputedNode>(e).map(|n| n.size().y * n.inverse_scale_factor).unwrap_or(0.0)
}

/// 缓动（ease-in-out cubic）
fn ease_in_out(t: f32) -> f32 {
  if t < 0.5 { 4.0 * t * t * t } else { 1.0 - (-2.0 * t + 2.0).powi(3) / 2.0 }
}

/// 视觉刷新：路径 / 收起态 / 子菜单与选项高亮 / 滑杆数值 / 色块
fn refresh_visuals(world: &mut World, root: Entity, parts: &MenuParts) {
  let (collapsed, display_path, back_display) = {
    let menu = world.get::<DebugMenu>(root).expect("DebugMenu 存在");
    (
      menu.collapsed,
      clip_path(&menu.model, &menu.path),
      if menu.path.is_empty() { Display::None } else { Display::Flex },
    )
  };
  // 色令牌按需取（避免每帧 clone 整份主题）
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

  // 路径标签（显示 id 路径；根 = "/"）+ 返回图标可见性
  if let Some(mut n) = world.get_mut::<Node>(parts.back_btn)
    && n.display != back_display
  {
    n.display = back_display;
  }
  if let Some(mut t) = world.get_mut::<Text>(parts.path_label)
    && t.0 != display_path
  {
    t.0 = display_path.clone();
  }
  if let Some(mut el) = world.get_mut::<crate::widgets::EllipsisText>(parts.path_label)
    && el.full != display_path
  {
    el.full = display_path;
  }
  // 收起/展开：视口显隐 + 图标切换
  let vis = if collapsed { Visibility::Hidden } else { Visibility::Inherited };
  if let Some(mut v) = world.get_mut::<Visibility>(parts.viewport)
    && *v != vis
  {
    *v = vis;
  }
  let glyph = if collapsed { Icon::Expand } else { Icon::Compress }.glyph();
  if let Some(mut t) = world.get_mut::<Text>(parts.collapse_icon)
    && t.0 != glyph
  {
    t.0 = glyph.to_string();
  }

  // 选项按钮 / 子菜单行：选中态 + hover 高亮
  let mut q = world.query::<(Entity, &MenuItem, &Interaction, Option<&Children>)>();
  let rows: Vec<(Entity, String, MenuRole, Interaction, Vec<Entity>)> = q
    .iter(world)
    .map(|(e, item, inter, ch)| {
      (e, item.path.clone(), item.role, *inter, ch.map(|c| c.iter().collect()).unwrap_or_default())
    })
    .collect();
  drop(q);
  for (e, path, role, inter, children) in rows {
    let hovered = inter != Interaction::None;
    let (bg, border, fg) = match role {
      MenuRole::SubMenu => {
        (if hovered { surface_elevated } else { Color::NONE }, Color::NONE, text_body)
      }
      MenuRole::SwitchOption(i) => {
        let selected = {
          let menu = world.get::<DebugMenu>(root).expect("DebugMenu 存在");
          matches!(menu.model.node(&split_path(&path)), Some(MenuNode::SwitchGroup { selected, .. }) if *selected == i)
        };
        if selected {
          (accent_fill, accent_text, text_primary)
        } else if hovered {
          (surface_overlay, border_strong, text_body)
        } else {
          (surface_elevated, border, text_muted)
        }
      }
      MenuRole::Button(_) => {
        if hovered {
          (surface_overlay, border_strong, text_body)
        } else {
          (surface_elevated, border, text_muted)
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
      // 切换组首项以外的左边框为 0（与前一项的右边框合成 1px，无空隙并排）
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
    for child in children {
      if let Some(mut t) = world.get_mut::<TextColor>(child)
        && t.0 != fg
      {
        t.0 = fg;
      }
    }
  }

  // 滑杆数值标签
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

  // 颜色色块（跟随模型 hex；非法 hex → 透明）
  let mut q = world.query::<(Entity, &MenuColorSwatch)>();
  let swatches: Vec<(Entity, String)> = q.iter(world).map(|(e, v)| (e, v.path.clone())).collect();
  drop(q);
  for (e, path) in swatches {
    let color = {
      let menu = world.get::<DebugMenu>(root).expect("DebugMenu 存在");
      match menu.model.node(&split_path(&path)) {
        Some(MenuNode::Color { hex, .. }) => crate::parse_hex_color(hex)
          .map_or(Color::NONE, |[r, g, b, a]| Color::srgba_u8(r, g, b, a)),
        _ => continue,
      }
    };
    if let Some(mut b) = world.get_mut::<BackgroundColor>(e)
      && b.0 != color
    {
      b.0 = color;
    }
  }
}

/// 拖拽窗口（标题栏 / 内容空白处起拖）+ 限制在系统窗口内
fn drag_and_clamp(world: &mut World, root: Entity, parts: &MenuParts) {
  let (just_pressed, pressed, released) = {
    let mouse = world.resource::<ButtonInput<MouseButton>>();
    (
      mouse.just_pressed(MouseButton::Left),
      mouse.pressed(MouseButton::Left),
      mouse.just_released(MouseButton::Left),
    )
  };
  // 光标逻辑位置（指针不在窗口内 → None）
  let cursor = {
    let mut q = world.query_filtered::<&Window, With<PrimaryWindow>>();
    q.iter(world).next().and_then(|w| w.cursor_position())
  };
  if just_pressed {
    let on_title =
      world.get::<Interaction>(parts.title_bar).is_some_and(|i| *i == Interaction::Pressed);
    let on_blank =
      world.get::<Interaction>(parts.viewport).is_some_and(|i| *i == Interaction::Pressed);
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
  // 边界约束：整窗不出系统窗口
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
    drop(m);
    if let Some(mut n) = world.get_mut::<Node>(root) {
      n.left = px(x);
      n.top = px(y);
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::menu::model::{InputField, WindowState, toggle};

  fn model() -> MenuFile {
    MenuFile {
      window: WindowState::default(),
      items: vec![toggle("fps", "FPS", false), toggle("vsync", "垂直同步", true)],
    }
  }

  /// 最小可交互 app：主题 + 时间 + 鼠标/窗口 + widget 状态机 + 菜单主系统
  fn test_app(theme: &crate::theme::UiTheme) -> App {
    let mut app = App::new();
    app.insert_resource(theme.clone());
    app.insert_resource(Time::<()>::default());
    app.init_resource::<ButtonInput<MouseButton>>();
    app.init_resource::<AccumulatedMouseMotion>();
    app.init_resource::<crate::capture::UiPointerCaptured>();
    app.init_resource::<crate::capture::MouseIntercepted>();
    app.init_resource::<crate::widgets::TextInputFocus>();
    app.init_resource::<crate::widgets::TooltipLayerEntity>();
    app.init_resource::<bevy::ecs::message::Messages<bevy::input::keyboard::KeyboardInput>>();
    app.world_mut().spawn((Window::default(), PrimaryWindow));
    app.add_systems(
      Update,
      (
        crate::widgets::toggle_switch_state_system,
        crate::widgets::slider_drag_system,
        crate::widgets::slider_visual_system,
        crate::widgets::text_input_pointer_system,
        crate::widgets::text_input_keyboard_system,
        crate::widgets::text_input_visual_system,
        crate::widgets::label_ellipsis_system,
        crate::i18n::i18n_refresh_system,
        menu_system,
      ),
    );
    app
  }

  /// 模拟一次「按下 → 释放 → 沉淀一帧」（Interaction 由 bevy_ui 的真实 ui_focus_system 驱动，
  /// 这里直接写组件模拟同一条边沿；沉淀帧用于让 widget 用 `Commands` 落地的状态（如
  /// toggle 的 `Checked` 插入）被菜单系统读到）
  fn click(app: &mut App, e: Entity) {
    app.world_mut().get_mut::<Interaction>(e).unwrap().set_if_neq(Interaction::Pressed);
    app.update();
    app.world_mut().get_mut::<Interaction>(e).unwrap().set_if_neq(Interaction::Hovered);
    app.update();
    app.update();
  }

  fn find_item(app: &mut App, path: &str, role: MenuRole) -> Entity {
    let mut q = app.world_mut().query::<(Entity, &MenuItem)>();
    let hits: Vec<(Entity, MenuItem)> = q.iter(app.world()).map(|(e, i)| (e, i.clone())).collect();
    hits
      .into_iter()
      .find(|(_, i)| i.path == path && i.role == role)
      .map(|(e, _)| e)
      .unwrap_or_else(|| panic!("找不到菜单项 {path} {role:?}"))
  }

  /// (模型, 当前路径, 是否收起)
  fn menu_state(app: &mut App, root: Entity) -> (MenuFile, Vec<String>, bool) {
    let m = app.world().get::<DebugMenu>(root).expect("DebugMenu");
    (m.model.clone(), m.path.clone(), m.collapsed)
  }

  #[test]
  fn spawn_builds_root_title_and_one_page() {
    let theme = crate::theme::default_theme();
    let ctx = UiCtx::new(&theme, None);
    let mut app = App::new();
    let h = spawn_debug_menu(app.world_mut(), &ctx, model());
    let w = app.world_mut();
    assert!(w.get::<DebugMenu>(h.root).is_some());
    let menu = w.get::<DebugMenu>(h.root).unwrap();
    assert!(menu.path.is_empty(), "根节点");
    let pages: Vec<Entity> = w.query_filtered::<Entity, With<MenuPage>>().iter(w).collect();
    assert_eq!(pages.len(), 1, "根页只有一页");
    // 两个开关项
    let items: Vec<&MenuItem> = w.query::<&MenuItem>().iter(w).collect();
    assert_eq!(items.len(), 2);
  }

  #[test]
  fn toggle_click_flips_model_and_emits_event() {
    use std::sync::{Arc, Mutex};
    let theme = crate::theme::default_theme();
    let mut app = test_app(&theme);
    let seen = Arc::new(Mutex::new(Vec::<(String, MenuAction)>::new()));
    let sink = seen.clone();
    app.add_observer(move |ev: On<MenuActionEvent>| {
      sink.lock().unwrap().push((ev.path.clone(), ev.action.clone()));
    });
    let h = spawn_debug_menu(app.world_mut(), &UiCtx::new(&theme, None), model());
    let e = find_item(&mut app, "fps", MenuRole::Toggle);
    assert!(!app.world().get::<Checked>(e).is_some_and(|_| true), "初始为关");

    click(&mut app, e);

    assert!(app.world().get::<Checked>(e).is_some(), "点击后 toggle 变开");
    let (m, _, _) = menu_state(&mut app, h.root);
    assert!(
      matches!(m.node(&split_path("fps")), Some(MenuNode::Toggle { checked, .. }) if *checked),
      "模型 checked 同步为 true"
    );
    let got = seen.lock().unwrap().clone();
    assert_eq!(got, vec![("fps".to_string(), MenuAction::Toggle(true))], "发出统一事件");
  }

  #[test]
  fn submenu_click_navigates_and_back_returns() {
    let theme = crate::theme::default_theme();
    let mut app = test_app(&theme);
    let model = MenuFile {
      window: WindowState::default(),
      items: vec![super::super::model::sub_menu(
        "video",
        "视频",
        vec![toggle("vsync", "垂直同步", true)],
      )],
    };
    let h = spawn_debug_menu(app.world_mut(), &UiCtx::new(&theme, None), model);
    let row = find_item(&mut app, "video", MenuRole::SubMenu);

    click(&mut app, row);
    // 动画播完（0.22s）
    for _ in 0..20 {
      app.world_mut().resource_mut::<Time>().advance_by(std::time::Duration::from_millis(20));
      app.update();
    }
    let (m, path, _) = menu_state(&mut app, h.root);
    assert_eq!(path, vec!["video".to_string()], "进入子菜单");
    assert!(m.window.path.is_empty(), "模型里的路径不随导航改（只在保存时写）");
    // 页面已是子页内容
    assert!(find_item(&mut app, "video/vsync", MenuRole::Toggle) != Entity::PLACEHOLDER);

    // 返回
    let back = app.world().get::<MenuParts>(h.root).unwrap().back_btn;
    click(&mut app, back);
    for _ in 0..20 {
      app.world_mut().resource_mut::<Time>().advance_by(std::time::Duration::from_millis(20));
      app.update();
    }
    let (_, path, _) = menu_state(&mut app, h.root);
    assert!(path.is_empty(), "回到根");
  }

  #[test]
  fn collapse_and_reset_position_buttons_work() {
    let theme = crate::theme::default_theme();
    let mut app = test_app(&theme);
    let h = spawn_debug_menu(app.world_mut(), &UiCtx::new(&theme, None), model());
    let parts = {
      let p = app.world().get::<MenuParts>(h.root).unwrap();
      (p.collapse_btn, p.reset_btn, p.viewport)
    };
    click(&mut app, parts.0);
    app.update();
    let (_, _, collapsed) = menu_state(&mut app, h.root);
    assert!(collapsed, "收起");
    assert_eq!(
      *app.world().get::<Visibility>(parts.2).unwrap(),
      Visibility::Hidden,
      "收起后内容区隐藏"
    );
    // 移动窗口后重置位置
    app.world_mut().get_mut::<DebugMenu>(h.root).unwrap().model.window.x = 300.0;
    click(&mut app, parts.1);
    app.update();
    let (m, _, _) = menu_state(&mut app, h.root);
    assert_eq!(m.window.x, DEFAULT_WINDOW_POS.x, "重置到默认位置");
  }

  #[test]
  fn switch_group_select_updates_model() {
    use std::sync::{Arc, Mutex};
    let theme = crate::theme::default_theme();
    let mut app = test_app(&theme);
    let seen = Arc::new(Mutex::new(Vec::<MenuAction>::new()));
    let sink = seen.clone();
    app.add_observer(move |ev: On<MenuActionEvent>| sink.lock().unwrap().push(ev.action.clone()));
    let model = MenuFile {
      window: WindowState::default(),
      items: vec![super::super::model::switch_group("mode", "模式", &["轨道", "自由"], 0)],
    };
    let h = spawn_debug_menu(app.world_mut(), &UiCtx::new(&theme, None), model);
    let second = find_item(&mut app, "mode", MenuRole::SwitchOption(1));
    click(&mut app, second);
    let (m, _, _) = menu_state(&mut app, h.root);
    assert!(
      matches!(m.node(&split_path("mode")), Some(MenuNode::SwitchGroup { selected, .. }) if *selected == 1),
      "选中下标写回模型"
    );
    assert_eq!(*seen.lock().unwrap(), vec![MenuAction::Select(1)]);
    // 选中项的视觉：背景切到强调填充
    assert_eq!(
      app.world().get::<BackgroundColor>(second).unwrap().0,
      color_of(&theme.colors.accent_fill),
      "选中项高亮"
    );
  }

  #[test]
  fn slider_value_writes_back_to_model() {
    use std::sync::{Arc, Mutex};
    let theme = crate::theme::default_theme();
    let mut app = test_app(&theme);
    let seen = Arc::new(Mutex::new(Vec::<MenuAction>::new()));
    let sink = seen.clone();
    app.add_observer(move |ev: On<MenuActionEvent>| sink.lock().unwrap().push(ev.action.clone()));
    let model = MenuFile {
      window: WindowState::default(),
      items: vec![super::super::model::slider("gain", "增益", 1.0, 0.0, 4.0, 0.5, 1, None)],
    };
    let h = spawn_debug_menu(app.world_mut(), &UiCtx::new(&theme, None), model);
    let e = find_item(&mut app, "gain", MenuRole::Slider);
    app.world_mut().get_mut::<SliderValue>(e).unwrap().0 = 3.0;
    app.update();
    let (m, _, _) = menu_state(&mut app, h.root);
    assert!(
      matches!(m.node(&split_path("gain")), Some(MenuNode::Slider { value, .. }) if (*value - 3.0).abs() < 1e-6),
      "滑杆值写回模型"
    );
    assert_eq!(*seen.lock().unwrap(), vec![MenuAction::Value(3.0)]);
  }

  #[test]
  fn input_text_writes_back_to_model() {
    let theme = crate::theme::default_theme();
    let mut app = test_app(&theme);
    let model = MenuFile {
      window: WindowState::default(),
      items: vec![super::super::model::input(
        "size",
        "大小",
        vec![InputField::number("", "3", 1.0, 16.0, 1.0, 0)],
      )],
    };
    let h = spawn_debug_menu(app.world_mut(), &UiCtx::new(&theme, None), model);
    let e = find_item(&mut app, "size", MenuRole::Input(0));
    app.world_mut().get_mut::<TextInputValue>(e).unwrap().0 = "9".into();
    app.update();
    let (m, _, _) = menu_state(&mut app, h.root);
    match m.node(&split_path("size")) {
      Some(MenuNode::Input { fields, .. }) => assert_eq!(fields[0].text, "9"),
      other => panic!("{other:?}"),
    }
  }

  #[test]
  fn title_bar_drag_moves_window_within_bounds() {
    let theme = crate::theme::default_theme();
    let mut app = test_app(&theme);
    let h = spawn_debug_menu(app.world_mut(), &UiCtx::new(&theme, None), model());
    let parts = app.world().get::<MenuParts>(h.root).unwrap();
    let title = parts.title_bar;

    // 光标落在标题栏上
    let set_cursor = |app: &mut App, p: Option<(f32, f32)>| {
      let mut q = app.world_mut().query_filtered::<&mut Window, With<PrimaryWindow>>();
      let mut w = q.single_mut(app.world_mut()).unwrap();
      w.set_physical_cursor_position(p.map(|(x, y)| bevy::math::DVec2::new(x as f64, y as f64)));
    };
    set_cursor(&mut app, Some((100.0, 15.0)));
    app.update();
    // 按下（模拟 ui_focus 把标题栏置为 Pressed）
    app.world_mut().resource_mut::<ButtonInput<MouseButton>>().press(MouseButton::Left);
    app.world_mut().get_mut::<Interaction>(title).unwrap().set_if_neq(Interaction::Pressed);
    app.update();
    // 光标右移 60、下移 40 → 窗口同量跟随
    set_cursor(&mut app, Some((160.0, 55.0)));
    app.update();
    let (m, _, _) = menu_state(&mut app, h.root);
    assert!(
      (m.window.x - (DEFAULT_WINDOW_POS.x + 60.0)).abs() < 0.5
        && (m.window.y - (DEFAULT_WINDOW_POS.y + 40.0)).abs() < 0.5,
      "窗口跟随光标：{:?}",
      (m.window.x, m.window.y)
    );
    // Node 同步
    let node = app.world().get::<Node>(h.root).unwrap();
    assert_eq!(node.left, px(m.window.x));
    assert_eq!(node.top, px(m.window.y));

    // 往左上拖出界 → 钳在窗口内（不小于 0 + MARGIN）
    set_cursor(&mut app, Some((-500.0, -500.0)));
    app.update();
    let (m, _, _) = menu_state(&mut app, h.root);
    assert!(m.window.x >= WINDOW_MARGIN && m.window.y >= WINDOW_MARGIN, "不出左/上边界");
    // 松开 → 停止跟随
    app.world_mut().resource_mut::<ButtonInput<MouseButton>>().release(MouseButton::Left);
    app.world_mut().get_mut::<Interaction>(title).unwrap().set_if_neq(Interaction::Hovered);
    app.update();
    let before = menu_state(&mut app, h.root).0.window;
    set_cursor(&mut app, Some((400.0, 300.0)));
    app.update();
    let after = menu_state(&mut app, h.root).0.window;
    assert_eq!((before.x, before.y), (after.x, after.y), "松开后不再跟随");
  }

  #[test]
  fn menu_labels_follow_locale_switch() {
    let theme = crate::theme::default_theme();
    let mut app = test_app(&theme);
    app.insert_resource(crate::i18n::UiTranslator::new(|k| match k {
      "menu.fps" => "帧率".to_string(),
      "menu.vsync" => "垂直同步".to_string(),
      other => other.to_string(),
    }));
    let model = MenuFile {
      window: WindowState::default(),
      items: vec![toggle("fps", "menu.fps", false), toggle("vsync", "menu.vsync", true)],
    };
    let translate = app.world().resource::<crate::i18n::UiTranslator>().handle();
    spawn_debug_menu(
      app.world_mut(),
      &UiCtx::new(&theme, None).with_translate(translate),
      model,
    );
    let label_text = |app: &mut App, key: &str| -> String {
      let mut q = app.world_mut().query::<(&crate::i18n::I18nKey, &Text)>();
      q.iter(app.world())
        .find(|(k, _)| k.0 == key)
        .map(|(_, t)| t.0.clone())
        .unwrap_or_else(|| panic!("找不到 keyed 文本 {key}"))
    };
    assert_eq!(label_text(&mut app, "menu.fps"), "帧率", "spawn 时按 key 解析");
    assert_eq!(label_text(&mut app, "menu.vsync"), "垂直同步");

    // 切语言：换解析器（内部 bump 版本）→ 菜单文本下帧跟着变
    app.world_mut().resource_mut::<crate::i18n::UiTranslator>().set(|k| match k {
      "menu.fps" => "FPS".to_string(),
      "menu.vsync" => "VSync".to_string(),
      other => other.to_string(),
    });
    app.update();
    assert_eq!(label_text(&mut app, "menu.fps"), "FPS");
    assert_eq!(label_text(&mut app, "menu.vsync"), "VSync");
  }

  #[test]
  fn ease_is_monotonic_and_bounded() {
    assert!((ease_in_out(0.0)).abs() < 1e-6);
    assert!((ease_in_out(1.0) - 1.0).abs() < 1e-6);
    assert!((ease_in_out(0.5) - 0.5).abs() < 1e-6);
    assert!(ease_in_out(0.3) < ease_in_out(0.7));
  }

  #[test]
  fn clip_path_stops_at_unknown_segment() {
    let m = MenuFile {
      window: WindowState::default(),
      items: vec![super::super::model::sub_menu("a", "A", vec![toggle("b", "B", false)])],
    };
    assert_eq!(clip_path(&m, &[]), "/");
    assert_eq!(clip_path(&m, &["a".into(), "b".into()]), "/a/b");
    assert_eq!(clip_path(&m, &["a".into(), "zzz".into()]), "/a");
    assert_eq!(clip_path(&m, &["zzz".into()]), "/");
  }

  #[test]
  fn path_split_and_join_roundtrip() {
    use super::super::items::join_path;
    assert_eq!(split_path("render/ddgi"), vec!["render".to_string(), "ddgi".to_string()]);
    assert!(split_path("/").is_empty());
    assert_eq!(join_path("", "a"), "a");
    assert_eq!(join_path("a", "b"), "a/b");
  }

  #[test]
  fn sub_menu_row_is_clickable_button() {
    use super::super::items::MenuSubMenuRow;
    let theme = crate::theme::default_theme();
    let mut app = App::new();
    let root = app.world_mut().spawn_empty().id();
    let node = super::super::model::sub_menu("child", "子菜单", vec![]);
    app.world_mut().entity_mut(root).with_children(|p| {
      spawn_item(&UiCtx::new(&theme, None), p, &node, "");
    });
    let w = app.world_mut();
    let e = w.query_filtered::<Entity, With<MenuSubMenuRow>>().iter(w).next().expect("子菜单行");
    assert_eq!(*w.get::<FocusPolicy>(e).unwrap(), FocusPolicy::Block);
    assert!(w.get::<MouseIntercept>(e).is_some());
    let item = w.get::<MenuItem>(e).unwrap();
    assert_eq!(item.path, "child");
    assert_eq!(item.role, MenuRole::SubMenu);
  }
}
