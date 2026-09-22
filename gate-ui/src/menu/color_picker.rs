//! 调色板浮窗：点颜色行的色块展开一格离散色板，选中格子把 HEX 写回该行的输入框。
//! 模态逻辑与 `widgets::dropdown` 的列表浮窗同款：浮层 = 顶层实体（无父节点 + `GlobalZIndex`），
//! 下面垫一层全屏透明遮罩 —— 只在「点浮层外」时关闭（该次点击不再下传）；位置每帧从色块锚点重算并钳进窗口。
//! 色板布局见 `palette_color`：`PALETTE_EDGE` × `PALETTE_EDGE` 方阵 —— 横轴 = 明度（左淡右暗），
//! 纵轴 = 色相（第 0 行灰阶，其余每行一个色相）。

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

/// 浮层层深：高于所有面板/菜单，低于 tooltip（1000）；与下拉浮层同档（两者不会同时展开）
const COLOR_PICKER_Z: i32 = 900;

/// 色板暗端（最右列）的明度（1.0 = 纯色）
const PALETTE_DARK_V: f32 = 0.25;
/// 色板淡端（最左列）的饱和度（只留一点色味，几乎近白；与灰阶行的白格同列）
const PALETTE_TINT_S: f32 = 0.1;

/// 色块展开态（挂在色块根上）
#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ColorPickerOpen(pub bool);

/// 浮层根标记（顶层实体；`owner` = 所属色块，`backdrop` = 同一生存期的遮罩）
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub struct ColorPickerPopup {
  pub owner: Entity,
  pub backdrop: Entity,
}

/// 浮层遮罩标记（全屏透明顶层实体，压在浮层之下；`owner` = 所属色块）。
/// 存在期间吞掉所有指针事件：点击它 = 关闭色板，且不触发其他 UI / 场景逻辑。
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub struct ColorPickerBackdrop {
  pub owner: Entity,
}

/// 色板里的一个格子（`owner` = 所属色块，`index` = 0..`palette_len`，颜色由 `palette_color` 给出）
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub struct ColorPickerCell {
  pub owner: Entity,
  pub index: usize,
}

/// 色板第 `index` 格的颜色。两根轴分工明确：
/// **横轴 = 明度**（与第 0 行的灰阶同向：左淡 → 右暗），**纵轴 = 色相**（第 0 行灰阶，其余每行一个色相）。
/// 这样最淡的一列正压在灰阶的白格下方、最暗的一列压在黑格下方 —— 色相块和灰阶行共用同一根明度轴，
/// 不会出现「最淡的一行被甩到网格最底下、和白色隔了一整块」这种断裂。
/// 每行沿横轴分两段且**每段只动一维**：淡侧只降饱和度（往白里混），暗侧只降明度（往黑里混）——
/// 两维一起动（如 `s: 1→0.25` 同时 `v: 0.25→1`）会把整行压成一片发灰的中间调，既没有纯色也没有亮的纯色。
pub fn palette_color(index: usize) -> Color {
  let row = index / PALETTE_EDGE;
  let col = index % PALETTE_EDGE;
  // 明度轴参数：0 = 最淡（最左），0.5 = 纯色，1 = 最暗（最右）
  let u = col as f32 / (PALETTE_EDGE - 1) as f32;
  if row == 0 {
    let v = 1.0 - u;
    return Color::srgb(v, v, v);
  }
  // 色相：起于蓝（220°）绕 240° 到绿（100°），与参考色板同序（蓝→紫→品红→红→橙→黄→绿）
  let hue = (220.0 + (row - 1) as f32 * (240.0 / (PALETTE_EDGE - 2) as f32)) % 360.0;
  let (saturation, value) = if u <= 0.5 {
    (PALETTE_TINT_S + (1.0 - PALETTE_TINT_S) * (u * 2.0), 1.0)
  } else {
    (1.0, 1.0 - (1.0 - PALETTE_DARK_V) * ((u - 0.5) * 2.0))
  };
  Color::hsv(hue, saturation, value)
}

/// 颜色 → `RRGGBB`（大写）：写回输入框与选中判定共用的唯一形式
fn hex_of(color: Color) -> String {
  let c = color.to_srgba();
  let ch = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
  format!("{:02X}{:02X}{:02X}", ch(c.red), ch(c.green), ch(c.blue))
}

/// 模型里的 hex（任意大小写 / 可带 `#`）→ 归一化的 `RRGGBB`；非法 → None
fn norm_hex(hex: &str) -> Option<String> {
  crate::parse_hex_color(hex).map(|[r, g, b, _]| format!("{r:02X}{g:02X}{b:02X}"))
}

/// 格子底色偏亮时选中框取黑，否则取白（灰阶首位那种近白格子也能看清框）
fn marker_color(bg: Color) -> Color {
  let c = bg.to_srgba();
  if 0.299 * c.red + 0.587 * c.green + 0.114 * c.blue > 0.6 { Color::BLACK } else { Color::WHITE }
}

/// 浮窗外框尺寸：色板方阵 + 四周各 `spacing.sm` 内衬 + 四周各 `border_width` 边框。
/// 内衬与边框都取自主题 ⇒ 与容器 `Node` 的 `padding`/`border` 同源，尺寸不会和实际内容对不上。
fn palette_size(m: &ThemeMetrics) -> Vec2 {
  let pad = m.spacing.sm + m.border_width;
  let edge = PALETTE_CELL * PALETTE_EDGE as f32 + pad * 2.0;
  Vec2::splat(edge)
}

/// 浮窗位置：右缘对齐色块右缘、顶边落在色块下缘下方 `gap`；再按 `WINDOW_MARGIN` 钳进窗口
fn popup_pos(anchor: (Vec2, Vec2), size: Vec2, win: Option<Vec2>, gap: f32) -> Vec2 {
  let (pos, anchor_size) = anchor;
  let mut p = Vec2::new(pos.x + anchor_size.x - size.x, pos.y + anchor_size.y + gap);
  if let Some(win) = win {
    p.x = p.x.clamp(WINDOW_MARGIN, (win.x - size.x - WINDOW_MARGIN).max(WINDOW_MARGIN));
    p.y = p.y.clamp(WINDOW_MARGIN, (win.y - size.y - WINDOW_MARGIN).max(WINDOW_MARGIN));
  }
  p
}

/// 从资源装配 `UiCtx`（展开时现场建浮层用；主题未就绪 → None）
fn ctx_of<'a>(
  theme: &'a Option<Res<UiTheme>>,
  font: &'a Option<Res<ThemeFont>>,
) -> Option<UiCtx<'a>> {
  let theme = &**theme.as_ref()?;
  Some(UiCtx::new(theme, font.as_ref().and_then(|f| f.handle.as_ref())))
}

/// 建浮窗：容器 + `PALETTE_EDGE` 行 × `PALETTE_EDGE` 格；另建一层全屏透明遮罩（浮层之下）。
/// 格子无外边距、外框即（与底色同色的）`PALETTE_MARK_BORDER` 边框 ⇒ 相邻格子严丝合缝，选中时改边框色即得方框标记。
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
  // 命令上下文用显式 `ChildOf` 组装：commands 版 spawner 拿不到 world 版 `ChildSpawner`
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
      // 压在所有面板/菜单之上、浮层之下
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
        // 边框规格与其它控件一致（`border_width`）；色板与边框之间留一圈内衬（贴着边框太挤）
        border: UiRect::all(px(m.border_width)),
        padding: UiRect::all(px(m.spacing.sm)),
        ..default()
      },
      BackgroundColor(color_of(&c.surface_elevated)),
      BorderColor::all(color_of(&c.border)),
      // 容器只定位不吃命中（格子各自 Block）
      Pickable::IGNORE,
      GlobalZIndex(COLOR_PICKER_Z),
    ))
    .id();
  for row in 0..PALETTE_EDGE {
    let row_e = commands
      .spawn((
        Name::new("ui-color-picker-row"),
        ChildOf(popup),
        Node {
          // 行高定死 = 格子边长（浮窗高就是按它算的），避免百分比高度在 flex 解析里的不确定
          height: px(PALETTE_CELL),
          flex_direction: FlexDirection::Row,
          ..default()
        },
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
        // 边框同底色 = 看不见的分隔（占位不占视觉），选中/悬停时由 visual 改色
        BorderColor::all(bg),
        MouseIntercept,
      ));
    }
  }
}

/// 关闭：销毁该色块的浮层（连同其遮罩）+ 复位展开态
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

/// 指针交互：点色块展开/收起 / 点格子写回 HEX 输入框并关闭 / 点遮罩关闭（不再有其他关闭路径）。
/// 写回的是 HEX 输入框（该行的真源）—— 模型与 `MenuActionEvent` 走 `menu_system` 的既有链路，故本系统须排在它之前。
#[allow(clippy::type_complexity, clippy::too_many_arguments)] // Bevy system：多组件查询/入参签名固有
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

  // ---------- 1. 格子：点到哪个 ----------
  let mut picked: Option<(Entity, usize)> = None;
  for (hovered, pressed, mut prev, cell) in &mut q_cells {
    let inter = UiInteract::of(hovered, pressed);
    if prev.0 == UiInteract::Pressed && inter == UiInteract::Hovered {
      picked = Some((cell.owner, cell.index));
    }
    prev.0 = inter;
  }

  // ---------- 2. 遮罩：点击色板外 ----------
  let mut clicked_outside: Option<Entity> = None;
  for (hovered, pressed, mut prev, bd) in &mut q_backdrops {
    let inter = UiInteract::of(hovered, pressed);
    if prev.0 == UiInteract::Pressed && inter == UiInteract::Hovered {
      clicked_outside = Some(bd.owner);
    }
    prev.0 = inter;
  }

  // ---------- 3. 色块：选中格子 / 展开 / 收起 ----------
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
      // 菜单被隐藏 → 浮窗与遮罩一并回收
      close_popup(&mut commands, e, &q_popups, &mut open);
    }
  }
  // 色块已销毁 → 浮层（连同遮罩）立即回收，不留孤儿
  for (popup, p) in &q_popups {
    if q_swatches.get(p.owner).is_err() {
      commands.entity(popup).despawn();
      commands.entity(p.backdrop).despawn();
    }
  }
}

/// 视觉：浮窗跟随色块（窗口内钳制）+ 格子的悬停/选中边框（选中框 = 该行 HEX 对应的那一格）。
#[allow(clippy::type_complexity, clippy::too_many_arguments)] // Bevy system：多组件查询/入参签名固有
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
      continue; // 菜单隐藏时冻结位置（关闭由 color_picker_system 负责）
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
    // 当前色 = 模型里的 hex（非法 / 不在色板上 → 没有选中框）
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
