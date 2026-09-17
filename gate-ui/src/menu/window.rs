//! DebugWindow 容器：标题栏 + 拖拽 + ViewPager 分页动画 + 收起/展开（高度缓动，时长 `PAGE_ANIM_SECS`）。
//! 结构：root（`DebugMenu`，绝对定位）→ title_bar + viewport（overflow clip，收起后 `Display::None`）→ page × 1..2。
//! `menu_system` 是唯一的交互/布局驱动（独占系统）：读控件状态 → 写回模型 → 发 `MenuActionEvent` → 刷新视觉 → 推进分页动画。

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
  DropdownValue, LabelConfig, LabelOverflow, LabelStyle, SliderValue, TextInputValue, UiCtx,
  color_of, dim_color, label, px, spawn_icon,
};

/// 窗口拖拽的上一帧光标位置（逻辑 px；`None` = 指针不在窗口内）。
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
  /// 本段动画的公共起点偏移（逻辑 px）：上一段被打断时接续它当前的偏移，否则 0
  start_off: f32,
}

/// 收起/展开动画状态（视口高度缓动；时长 `PAGE_ANIM_SECS`）
#[derive(Component, Clone, Copy, Debug, PartialEq)]
struct MenuCollapse {
  /// 进度 0..=1（1 = 动画结束）
  t: f32,
  /// 起始高度（逻辑 px）
  from: f32,
  /// 目标高度（逻辑 px；收起 = 0）
  to: f32,
  /// 目标态：true = 收起
  collapsed: bool,
}

/// 容器各部位实体句柄（menu_system 定位用）
#[derive(Component)]
pub struct MenuParts {
  pub title_bar: Entity,
  pub back_btn: Entity,
  /// 返回按钮的图标（禁用态置灰用）
  pub back_icon: Entity,
  pub path_label: Entity,
  pub reset_btn: Entity,
  /// 重置位置按钮的图标（交互配色用）
  pub reset_icon: Entity,
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
  /// 切换组选中第 i 项（下拉框选中第 i 项同此）
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
  let root_path = display_path(&model, &path, |k| ctx.text(k));

  // 容器不绘制（纯透明、无边框，只做布局与命中）：底色与边框交给标题栏与各页画
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
          // 左右+底边框；顶边不画
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
        Interaction::default(),
        // 标题栏空白处可拖动窗口
        FocusPolicy::Block,
        MouseIntercept,
      ))
      .id();
    title_bar_e = tb;
    r.world_mut().entity_mut(tb).with_children(|bar| {
      (back_e, back_icon) = icon_button(ctx, bar, TitleAction::Back, Icon::ChevronLeft);
      // 路径标签：占满中间，过长中间省略；颜色用 Muted 档，字号单独提到正文档（12 → 14）
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
      // 只补字段，不能整个 Node 覆盖（中间省略依赖 label 自带 Node 的 overflow: clip）；左右 margin 留间距
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
    // 收起动画静止态起步（收起态由模型带入，首帧直接落成 Display::None）
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

/// 标题栏图标按钮（无文字，只有 FontAwesome 字形）；返回 (按钮, 图标)
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
    Interaction::default(),
    MenuPressPrev::default(),
    // 命中区 = 与标题栏同高的正方形（点击范围铺满整格）
    Node {
      width: px(TITLE_BAR_H),
      height: px(TITLE_BAR_H),
      align_items: AlignItems::Center,
      justify_content: JustifyContent::Center,
      ..default()
    },
    BackgroundColor(Color::NONE),
    FocusPolicy::Block,
    MouseIntercept,
  ));
  let mut icon_e = Entity::PLACEHOLDER;
  ec.with_children(|b| {
    icon_e = spawn_icon(ctx, b, icon.glyph(), TITLE_ICON_SIZE, color_of(&c.text_body));
  });
  (ec.id(), icon_e)
}

/// 按模型建一页（节点子树 + 直接挂到 viewport 下）；页面自带底色与边框（容器不画）。
/// 上边框不画——与标题栏底边框重叠会变 2px 粗线。
fn build_page(
  world: &mut World,
  ctx: &UiCtx,
  viewport: Entity,
  model: &MenuFile,
  path: &[String],
) -> Entity {
  // 页面外框宽 = 内容宽 + 左右各 1px 边框；内容宽恒为 PAGE_W
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

/// 沿 id 路径收集模型里真实存在的节点（无效段处截断）
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

/// 剪掉无效路径段（TOML 被手改后仍能建出 UI）；显示 id 路径（日志用）
pub(crate) fn clip_path(model: &MenuFile, path: &[String]) -> String {
  let ok: Vec<&str> = clip_nodes(model, path).iter().map(|n| n.id()).collect();
  if ok.is_empty() { "/".to_string() } else { format!("/{}", ok.join("/")) }
}

/// 标题栏路径文本：各段取节点 label 的译文（如 `/渲染/曝光`）；无效段截断；根 = `/`
fn display_path(model: &MenuFile, path: &[String], translate: impl Fn(&str) -> String) -> String {
  let ok: Vec<String> = clip_nodes(model, path).iter().map(|n| translate(n.label())).collect();
  if ok.is_empty() { "/".to_string() } else { format!("/{}", ok.join("/")) }
}

/// 世界 → UiCtx（导航建页时需要；主题/字体/解析器句柄 clone）
fn ctx_from_world(
  world: &World,
) -> Option<(UiTheme, Option<Handle<Font>>, Option<Handle<Font>>, Option<crate::i18n::TranslatorFn>)>
{
  let theme = world.get_resource::<UiTheme>()?.clone();
  let font = world.get_resource::<ThemeFont>().and_then(|f| f.handle.clone());
  let icon = world.get_resource::<IconFont>().and_then(|f| f.handle.clone());
  let translate = world.get_resource::<crate::i18n::UiTranslator>().and_then(|t| t.handle());
  Some((theme, font, icon, translate))
}

/// 菜单主系统：交互 → 模型 → 事件 → 视觉 → 分页动画 → 窗口约束（独占系统）。
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
      // 根节点无上级：返回按钮禁用，点击忽略
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
  drop(menu);

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

/// 控件值快照
enum MenuValue {
  Value(f32),
  Checked(bool),
  Text(String),
  /// 下拉框选中的下标
  Index(usize),
}

/// 每帧控件值快照行：(path, 角色, 滑杆值, 开关态, 输入文本, 下拉选中下标)
type ControlSnapshot = (String, MenuRole, Option<f32>, Option<bool>, Option<String>, Option<usize>);

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
  // 旧页直接销毁，不等收尾动画（p.outgoing 将被覆盖）
  if let Some(e) = stale
    && e != old_page
    && world.get_entity(e).is_ok()
  {
    world.despawn(e);
  }
  // 旧页高度作动画起始高度（新页布局完成后下一帧再取 max）
  let old_h = logical_height(world, old_page);
  let step = page_outer_w(ctx.theme.metrics.border_width);
  // 被打断时旧页已偏离中线，从其当前偏移接续滑动
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
  info!("菜单导航 → {}", clip_path(&model, target));
  if let Some(mut p) = world.get_mut::<MenuPager>(root) {
    p.outgoing = Some(old_page);
    p.current = new_page;
    p.dir = dir;
    p.start_off = from_left;
    p.t = 0.0;
  }
}

/// 推进分页动画：两页并排滑动；窗口内容高度取两者较大者
fn advance_pager(world: &mut World, root: Entity, viewport: Entity) {
  let (mut t, dir, current, outgoing, start_off) = {
    let Some(p) = world.get::<MenuPager>(root) else { return };
    (p.t, p.dir, p.current, p.outgoing, p.start_off)
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
  let step =
    page_outer_w(world.get_resource::<UiTheme>().map_or(0.0, |th| th.metrics.border_width));
  // 公共起点偏移随进度回零；t=1 时新页落在 0、旧页完全出窗
  let off = start_off * (1.0 - p);
  let old = outgoing.expect("outgoing 存在");
  if let Some(mut n) = world.get_mut::<Node>(old) {
    n.left = px(off - dir * p * step);
  }
  if let Some(mut n) = world.get_mut::<Node>(current) {
    n.left = px(off + dir * (1.0 - p) * step);
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

/// 页面外框宽度（内容宽 `PAGE_W` + 左右各 1px 边框）；滑动切换时两页紧贴的步距。
fn page_outer_w(border_width: f32) -> f32 {
  PAGE_W + border_width * 2.0
}

/// 页面完整高度：纯行列表、行高固定 `ITEM_H`（见 `items::base_row`）+ 页面自身的上下边框
fn page_height(world: &World, page: Entity) -> f32 {
  let rows = world.get::<Children>(page).map_or(0.0, |c| c.len() as f32 * ITEM_H);
  let border = world.get::<Node>(page).map_or(0.0, |n| {
    let px_of = |v: Val| if let Val::Px(v) = v { v } else { 0.0 };
    px_of(n.border.top) + px_of(n.border.bottom)
  });
  rows + border
}

/// 收起/展开：视口高度缓动（时长 `PAGE_ANIM_SECS`）；收起后视口 `Display::None` 完全脱离布局，展开涨到页面完整高度后交回 `Val::Auto`。
fn advance_collapse(world: &mut World, root: Entity, viewport: Entity) {
  let Some(state) = world.get::<MenuCollapse>(root).copied() else { return };
  let collapsed = world.get::<DebugMenu>(root).is_some_and(|m| m.collapsed);

  if state.t >= 1.0 && state.collapsed == collapsed {
    // 静止态：收起 → 内容不参与布局
    let display = if collapsed { Display::None } else { Display::Flex };
    if let Some(mut n) = world.get_mut::<Node>(viewport)
      && n.display != display
    {
      n.display = display;
    }
    return;
  }
  if state.t >= 1.0 {
    // 目标态翻转 → 起一段新动画：起点 = 视口当前高度（收起态下为 0）
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

/// 缓动（ease-in-out cubic）
fn ease_in_out(t: f32) -> f32 {
  if t < 0.5 { 4.0 * t * t * t } else { 1.0 - (-2.0 * t + 2.0).powi(3) / 2.0 }
}

/// 视觉刷新：路径 / 返回按钮禁用态 / 收起图标 / 子菜单与选项高亮 / 滑杆数值 / 色块
fn refresh_visuals(world: &mut World, root: Entity, parts: &MenuParts) {
  let (collapsed, path_text, back_disabled) = {
    let menu = world.get::<DebugMenu>(root).expect("DebugMenu 存在");
    let i18n = world.get_resource::<crate::i18n::UiTranslator>();
    let path_text = match i18n {
      Some(t) => display_path(&menu.model, &menu.path, |k| t.resolve(k)),
      None => display_path(&menu.model, &menu.path, |k| k.to_string()),
    };
    // 根路径无上级可返回：按钮置灰（禁用态），位置照常占住
    (menu.collapsed, path_text, menu.path.is_empty())
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

  // 路径标签（显示译文路径；根 = "/"）
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
  // 标题栏图标按钮：hover / 按下 配色；根节点的返回按钮为禁用态（置灰 + 交互穿透，标题栏照常可拖）
  for (btn, icon, disabled) in [
    (parts.back_btn, parts.back_icon, back_disabled),
    (parts.reset_btn, parts.reset_icon, false),
    (parts.collapse_btn, parts.collapse_icon, false),
  ] {
    let inter = world.get::<Interaction>(btn).copied().unwrap_or(Interaction::None);
    let (bg, fg) = if disabled {
      (Color::NONE, dim_color(text_body))
    } else {
      match inter {
        Interaction::Pressed => (accent_fill, accent_text),
        Interaction::Hovered => (surface_overlay, text_primary),
        Interaction::None => (Color::NONE, text_body),
      }
    };
    if let Some(mut b) = world.get_mut::<BackgroundColor>(btn)
      && b.0 != bg
    {
      b.0 = bg;
    }
    // 禁用态让指针穿过整格；图标节点须保持 Node 默认 `FocusPolicy::Pass`
    let focus = if disabled { FocusPolicy::Pass } else { FocusPolicy::Block };
    if let Some(mut f) = world.get_mut::<FocusPolicy>(btn)
      && *f != focus
    {
      *f = focus;
    }
    if let Some(mut t) = world.get_mut::<TextColor>(icon)
      && t.0 != fg
    {
      t.0 = fg;
    }
  }
  // 收起/展开图标：向上 = 收起，向下 = 展开
  let glyph = if collapsed { Icon::AngleDown } else { Icon::AngleUp }.glyph();
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
  // 并排按钮的共享竖边（见本循环后的归属修正）：(父容器, 序号, 按钮, 高亮优先级, 边框色)
  let mut shared_edges: Vec<(Entity, usize, Entity, u8, Color)> = Vec::new();
  for (e, path, role, inter, children) in rows {
    let hovered = inter != Interaction::None;
    // 末位 = 高亮优先级（0 常态 / 1 悬停 / 2 选中）：只有并排按钮用，决定共享竖边归谁
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
    // 并排按钮全部入表（含首个）：相邻两项间那 1px 竖线由前一项的右边框画出，
    // 入表不全会让共享竖线漏改（首个项没入表 → 第 2 项选中时左边缘仍缺）。
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
  // 相邻两项间 1px 竖线由前一项的右边框画出；后一项高亮更强时，前一项右边框取其颜色。
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
