//! dropdown：单行下拉框（点击展开；选中项与控件原位重合，其余选项自上方/下方缓动展开）。
//! 框体与 `text_input` 同款（底/边框/高度/内距/字号），右侧多一个箭头图标。
//! 展开浮层是顶层实体（无父节点 + `GlobalZIndex`，不进调用方 UI 树），位置每帧从控件锚点重算；
//! 浮层下垫一层全屏透明遮罩：列表只在「点击列表外」时关闭（该次点击不再下传）；
//! 列表整体按 `WINDOW_MARGIN` 钳制在窗口内（优先于「选中项与控件重合」）；
//! 滚轮在「未展开且悬停控件」或「展开中任意位置」时上下移动选中项，浮层播放滑动动画。

use std::ops::Deref;

use bevy::ecs::message::MessageReader;
use bevy::input::mouse::{MouseScrollUnit, MouseWheel};
use bevy::picking::Pickable;
use bevy::picking::hover::Hovered;
use bevy::prelude::*;
use bevy::text::LineBreak;
use bevy::ui::{ComputedNode, Pressed, UiGlobalTransform};
use bevy::window::PrimaryWindow;

use super::{
  FontAttrs, LabelConfig, LabelStyle, UiCtx, UiDisabled, color_of, dim_color, label,
  label_bundle_attrs, px, spawn_icon,
};
use crate::capture::MouseIntercept;
use crate::icon::{Icon, IconFont};
use crate::menu::consts::WINDOW_MARGIN;
use crate::pointer::{UiInteract, UiInteractBundle, UiInteractPrev};
use crate::theme::{ThemeFont, UiTheme};
use crate::widgets::consts::{DROPDOWN_ANIM_SECS, DROPDOWN_ARROW_SIZE, DROPDOWN_SCROLL_SPEED};

/// 浮层层深：高于所有面板/菜单，低于 tooltip（1000）
const DROPDOWN_Z: i32 = 900;

/// 下拉框控件根标记
#[derive(Component, Debug, Default)]
pub struct DropdownRoot;

/// 关闭态显示文本子实体标记
#[derive(Component, Debug, Default)]
pub struct DropdownText;

/// 框内右侧箭头图标子实体标记
#[derive(Component, Debug, Default)]
pub struct DropdownArrow;

/// 展开浮层根标记（顶层实体；`owner` = 所属控件）
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct DropdownPopup {
  pub owner: Entity,
  /// 同一生存期的遮罩实体（随浮层一并销毁）
  pub backdrop: Entity,
  /// 展开进度 0..=1
  pub t: f32,
  /// 选中项视觉下标：向真源缓动（= 滚动动画）；与真源相等时选中项与控件逐像素重合
  pub sel: f32,
}

/// 浮层遮罩标记（全屏透明顶层实体，压在浮层之下；`owner` = 所属控件）。
/// 存在期间吞掉所有指针事件：点击它 = 关闭列表，且不触发其他 UI / 场景逻辑。
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub struct DropdownBackdrop {
  pub owner: Entity,
}

/// 浮层里的一个选项（`owner` = 所属控件，`index` = 选项下标）
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct DropdownOption {
  pub owner: Entity,
  pub index: usize,
}

/// 选项文案列表（i18n key；未注入解析器时 key 原样显示，直接写字面量同样成立）
#[derive(Component, Clone, Debug, PartialEq, Eq, Default)]
pub struct DropdownOptions(pub Vec<String>);

/// 选中下标（真源）
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct DropdownValue(pub usize);

/// 交互状态
#[derive(Component, Clone, Copy, Debug, PartialEq, Default)]
pub struct DropdownState {
  /// 展开中（浮层存活期间恒为 true）
  pub open: bool,
  /// 滚轮累计增量（格；满 1 格移动一项，余数留到下一帧）
  pub wheel: f32,
}

/// 选中项变化事件（EntityEvent，target = 控件根实体；由点击选项或滚轮改选触发）
#[derive(EntityEvent, Clone, Copy, Debug, PartialEq)]
pub struct DropdownChanged {
  pub entity: Entity,
  pub value: usize,
}

/// 下拉框句柄（Deref 到根实体 Entity）
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DropdownHandle(pub Entity);

impl Deref for DropdownHandle {
  type Target = Entity;
  fn deref(&self) -> &Entity {
    &self.0
  }
}

impl From<DropdownHandle> for Entity {
  fn from(h: DropdownHandle) -> Self {
    h.0
  }
}

/// 下拉框配置（全部字段进 Config；Default = 空选项 + 选中 0、不禁用）
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct DropdownConfig {
  /// 选项文案（i18n key / 字面量）
  pub options: Vec<String>,
  /// 选中下标（越界自动钳到末项）
  pub selected: usize,
  pub disabled: bool,
}

/// 主题下拉框（宽度由父容器决定；最小宽度 56px，与输入框一致）
pub fn dropdown(ctx: &UiCtx, parent: &mut ChildSpawner, config: DropdownConfig) -> DropdownHandle {
  let m = &ctx.theme.metrics;
  let selected = config.selected.min(config.options.len().saturating_sub(1));
  let key = config.options.get(selected).cloned().unwrap_or_default();
  let (bg, border, text_color) = dropdown_colors(ctx.theme, false, UiInteract::None);
  let (bg, border, text_color) = if config.disabled {
    (dim_color(bg), dim_color(border), dim_color(text_color))
  } else {
    (bg, border, text_color)
  };
  let mut ec = parent.spawn((
    Name::new("ui-dropdown"),
    DropdownRoot,
    DropdownState::default(),
    DropdownValue(selected),
    DropdownOptions(config.options.clone()),
    UiInteractBundle::default(),
    Node {
      min_width: px(56.0),
      // 与输入框同高（文字高 + 上下内距 + 上下边框）
      height: px(m.font_size.md + m.spacing.sm + m.border_width * 2.0),
      flex_direction: FlexDirection::Row,
      align_items: AlignItems::Center,
      padding: UiRect::horizontal(px(m.spacing.xs)),
      border: UiRect::all(px(m.border_width)),
      overflow: Overflow::clip(),
      ..default()
    },
    BackgroundColor(bg),
    BorderColor::all(border),
    // 控件吞掉鼠标事件（点击展开 / 悬停不穿透到菜单行与 3D 场景）
    MouseIntercept,
  ));
  if config.disabled {
    ec.insert(UiDisabled);
  }
  ec.with_children(|root| {
    let t = *label(
      ctx,
      root,
      LabelConfig {
        text: ctx.text(&key),
        style: LabelStyle::Body,
        size: Some(m.font_size.md),
        ..default()
      },
    );
    root.world_mut().entity_mut(t).insert((
      DropdownText,
      crate::i18n::I18nKey::new(&key),
      TextColor(text_color),
      TextLayout { linebreak: LineBreak::NoWrap, ..default() },
      // 占满剩余宽度：文本左对齐、箭头贴右（只补字段，不覆盖 label 自带 Node）
      Node { flex_grow: 1.0, ..default() },
    ));
    let a = spawn_icon(
      ctx,
      root,
      Icon::ChevronDown.glyph(),
      DROPDOWN_ARROW_SIZE,
      color_of(&ctx.theme.colors.text_muted),
    );
    root.world_mut().entity_mut(a).insert(DropdownArrow);
  });
  DropdownHandle(ec.id())
}

/// (背景, 边框, 文字) 配色：与输入框同档；展开态 = 输入框编辑态（强调边框），hover 提亮一档
fn dropdown_colors(theme: &UiTheme, open: bool, inter: UiInteract) -> (Color, Color, Color) {
  let c = &theme.colors;
  let border = if open {
    color_of(&c.accent_text)
  } else if inter.is_active() {
    color_of(&c.border_strong)
  } else {
    color_of(&c.border)
  };
  (color_of(&c.surface_elevated), border, color_of(&c.text_primary))
}

/// 缓动（ease-in-out cubic；与菜单分页动画同一曲线）
fn ease_in_out(t: f32) -> f32 {
  if t < 0.5 { 4.0 * t * t * t } else { 1.0 - (-2.0 * t + 2.0).powi(3) / 2.0 }
}

/// 锚点矩形（逻辑 px 左上角 + 尺寸）；`ComputedNode`/`UiGlobalTransform` 是物理 px，按 `scale_factor` 折算为逻辑 px。
/// 浮层定位共用（下拉浮层 / 调色板浮窗）。
pub(crate) fn anchor_rect(
  node: &ComputedNode,
  xform: &UiGlobalTransform,
  scale_factor: f32,
) -> (Vec2, Vec2) {
  let sf = scale_factor.max(f32::EPSILON);
  let size = node.size() / sf;
  (xform.translation / sf - size * 0.5, size)
}

/// 从资源装配 `UiCtx`（展开时现场建浮层用；主题未就绪 → None）
fn ctx_of<'a>(
  theme: &'a Option<Res<UiTheme>>,
  font: &'a Option<Res<ThemeFont>>,
  icon: &'a Option<Res<IconFont>>,
  i18n: &'a Option<Res<crate::i18n::UiTranslator>>,
) -> Option<UiCtx<'a>> {
  let theme = &**theme.as_ref()?;
  Some(
    UiCtx::new(theme, font.as_ref().and_then(|f| f.handle.as_ref()))
      .with_icon_font(icon.as_ref().and_then(|f| f.handle.as_ref()))
      .with_translate(i18n.as_ref().and_then(|t| t.handle())),
  )
}

/// 建展开浮层：容器 = 锚点控件矩形；选项绝对定位（动画由 `dropdown_visual_system` 推进）。
/// 选项与控件同高/同内距/同字号：`t = 1` 时选中项与关闭态控件逐像素重合。
/// 另建一层全屏透明遮罩（浮层之下）：列表存活期间吞掉所有指针事件，点击它只关列表。
fn spawn_popup(
  commands: &mut Commands,
  ctx: &UiCtx,
  owner: Entity,
  options: &DropdownOptions,
  selected: usize,
  rect: (Vec2, Vec2),
) {
  let m = &ctx.theme.metrics;
  let c = &ctx.theme.colors;
  let (pos, size) = rect;
  let selected = selected.min(options.0.len().saturating_sub(1));
  // 命令上下文用显式 `ChildOf` 组装：commands 版 spawner 拿不到 world 版 `ChildSpawner`
  let backdrop = commands
    .spawn((
      Name::new("ui-dropdown-backdrop"),
      DropdownBackdrop { owner },
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
      GlobalZIndex(DROPDOWN_Z - 1),
      MouseIntercept,
      UiInteractBundle::default(),
    ))
    .id();
  let popup = commands
    .spawn((
      Name::new("ui-dropdown-popup"),
      DropdownPopup { owner, backdrop, t: 0.0, sel: selected as f32 },
      Node {
        position_type: PositionType::Absolute,
        left: px(pos.x),
        top: px(pos.y),
        width: px(size.x),
        height: px(size.y),
        ..default()
      },
      BackgroundColor(Color::NONE),
      // 容器只有控件那么大，自身不吃命中（选项各自 Block）
      Pickable::IGNORE,
      GlobalZIndex(DROPDOWN_Z),
    ))
    .id();
  for (i, key) in options.0.iter().enumerate() {
    let option = commands
      .spawn((
        Name::new("ui-dropdown-option"),
        ChildOf(popup),
        DropdownOption { owner, index: i },
        UiInteractBundle::default(),
        Node {
          position_type: PositionType::Absolute,
          // 动画每帧写 top = (i - sel) · h · ease(t)；t = 0 时全部叠在锚点矩形上
          left: px(0.0),
          top: px(0.0),
          width: Val::Percent(100.0),
          height: px(size.y),
          align_items: AlignItems::Center,
          padding: UiRect::horizontal(px(m.spacing.xs)),
          border: UiRect {
            left: px(m.border_width),
            right: px(m.border_width),
            // 纵向相邻共享 1px：首项画上边框，其余靠前一项的下边框
            top: px(if i == 0 { m.border_width } else { 0.0 }),
            bottom: px(m.border_width),
          },
          overflow: Overflow::clip(),
          ..default()
        },
        BackgroundColor(color_of(&c.surface_elevated)),
        BorderColor::all(color_of(&c.border)),
        ZIndex(if i == selected { 2 } else { 1 }),
        MouseIntercept,
      ))
      .id();
    // 选项文本与控件同字号/同内距 → 选中项与关闭态控件文本完全重叠
    let text = commands
      .spawn((
        label_bundle_attrs(
          ctx,
          ctx.text(key),
          m.font_size.md,
          color_of(&c.text_body),
          FontAttrs::default(),
        ),
        ChildOf(option),
      ))
      .id();
    commands.entity(text).insert((
      crate::i18n::I18nKey::new(key),
      TextLayout { linebreak: LineBreak::NoWrap, ..default() },
      Node { flex_grow: 1.0, ..default() },
    ));
  }
}

/// 关闭：销毁该控件的浮层（连同其遮罩）+ 复位展开态
fn close_popup(
  commands: &mut Commands,
  owner: Entity,
  popups: &Query<(Entity, &DropdownPopup)>,
  state: &mut DropdownState,
) {
  for (e, p) in popups.iter() {
    if p.owner == owner {
      commands.entity(e).despawn();
      commands.entity(p.backdrop).despawn();
    }
  }
  state.open = false;
}

/// 指针交互：点控件展开 / 点选项选中 / 点遮罩关闭（不再有其他关闭路径）。
/// 滚轮：未展开时需悬停控件、展开时任意位置；上/下滚 = 选中项上移/下移（满一格动一项）。
#[allow(clippy::type_complexity, clippy::too_many_arguments)] // Bevy system：多组件查询/入参签名固有
pub fn dropdown_system(
  mut commands: Commands,
  mut wheel: MessageReader<MouseWheel>,
  windows: Query<&Window, With<PrimaryWindow>>,
  theme: Option<Res<UiTheme>>,
  font: Option<Res<ThemeFont>>,
  icon: Option<Res<IconFont>>,
  i18n: Option<Res<crate::i18n::UiTranslator>>,
  mut q_roots: Query<
    (
      Entity,
      &Hovered,
      Has<Pressed>,
      &mut UiInteractPrev,
      &mut DropdownState,
      &DropdownOptions,
      &mut DropdownValue,
      &ComputedNode,
      &UiGlobalTransform,
      Option<&InheritedVisibility>,
      Has<UiDisabled>,
    ),
    (With<DropdownRoot>, Without<DropdownOption>),
  >,
  mut q_options: Query<
    (&Hovered, Has<Pressed>, &mut UiInteractPrev, &DropdownOption),
    (Without<DropdownRoot>, Without<DropdownPopup>),
  >,
  mut q_backdrops: Query<
    (Entity, &Hovered, Has<Pressed>, &mut UiInteractPrev, &DropdownBackdrop),
    (Without<DropdownRoot>, Without<DropdownOption>),
  >,
  q_popups: Query<(Entity, &DropdownPopup)>,
) {
  let sf = windows.single().ok().map(|w| w.scale_factor()).unwrap_or(1.0);

  // ---------- 0. 滚轮增量：Line 一格；Pixel（触控板）按 16px 行高折算 ----------
  let mut lines = 0.0;
  for ev in wheel.read() {
    match ev.unit {
      MouseScrollUnit::Line => lines += ev.y,
      MouseScrollUnit::Pixel => lines += ev.y / 16.0,
    }
  }

  // ---------- 1. 选项：点击判定 ----------
  let mut clicked: Option<(Entity, usize)> = None;
  for (hovered, pressed, mut prev, opt) in &mut q_options {
    let inter = UiInteract::of(hovered, pressed);
    if prev.0 == UiInteract::Pressed && inter == UiInteract::Hovered {
      clicked = Some((opt.owner, opt.index));
    }
    prev.0 = inter;
  }

  // ---------- 2. 遮罩：点击列表外 ----------
  let mut clicked_outside: Option<Entity> = None;
  for (_, hovered, pressed, mut prev, bd) in &mut q_backdrops {
    let inter = UiInteract::of(hovered, pressed);
    if prev.0 == UiInteract::Pressed && inter == UiInteract::Hovered {
      clicked_outside = Some(bd.owner);
    }
    prev.0 = inter;
  }

  // ---------- 3. 控件：选项选中 / 展开 / 收起 / 滚轮改选 ----------
  for (
    e,
    hovered,
    pressed,
    mut prev,
    mut state,
    options,
    mut value,
    node,
    xform,
    visible,
    disabled,
  ) in &mut q_roots
  {
    let inter = UiInteract::of(hovered, pressed);
    let clicked_self = prev.0 == UiInteract::Pressed && inter == UiInteract::Hovered;
    prev.0 = inter;
    if disabled {
      continue;
    }
    // 点选项 = 选中（写真源 + 通知）；点当前项 = 只关闭
    if let Some((owner, index)) = clicked
      && owner == e
    {
      if value.0 != index {
        value.0 = index;
        commands.trigger(DropdownChanged { entity: e, value: index });
      }
      close_popup(&mut commands, e, &q_popups, &mut state);
      continue;
    }
    // 点列表外（遮罩）→ 只关闭
    if clicked_outside == Some(e) {
      close_popup(&mut commands, e, &q_popups, &mut state);
      continue;
    }
    // 滚轮改选（满一格动一项，余数留到下一帧）
    if lines != 0.0 && (state.open || inter.is_active()) {
      state.wheel += lines;
      let steps = state.wheel.trunc();
      if steps != 0.0 {
        state.wheel -= steps;
        let count = options.0.len();
        if count > 0 {
          // 滚轮上（steps > 0）→ 下标减
          let next = (value.0 as i32 - steps as i32).clamp(0, count as i32 - 1) as usize;
          if next != value.0 {
            value.0 = next;
            commands.trigger(DropdownChanged { entity: e, value: next });
          }
        }
      }
    }
    if clicked_self {
      if state.open {
        close_popup(&mut commands, e, &q_popups, &mut state);
      } else if let Some(ctx) = ctx_of(&theme, &font, &icon, &i18n) {
        spawn_popup(&mut commands, &ctx, e, options, value.0, anchor_rect(node, xform, sf));
        state.open = true;
      }
    } else if state.open && !visible.map(|v| v.get()).unwrap_or(true) {
      // 菜单被隐藏 → 浮层与遮罩一并回收
      close_popup(&mut commands, e, &q_popups, &mut state);
    }
  }
  // 锚点控件已销毁 → 浮层（连同遮罩）立即回收，不留孤儿
  for (popup, p) in &q_popups {
    if q_roots.get(p.owner).is_err() {
      commands.entity(popup).despawn();
      commands.entity(p.backdrop).despawn();
    }
  }
}

/// 视觉：关闭态文本/配色 + 浮层跟随锚点（窗口内钳制）+ 展开/滚动动画 + 选项配色（每帧重算）
#[allow(clippy::type_complexity, clippy::too_many_arguments)] // Bevy system：多组件查询/入参签名固有
pub fn dropdown_visual_system(
  theme: Option<Res<UiTheme>>,
  i18n: Option<Res<crate::i18n::UiTranslator>>,
  time: Res<Time>,
  windows: Query<&Window, With<PrimaryWindow>>,
  mut q_roots: Query<
    (
      &DropdownValue,
      &DropdownOptions,
      &Hovered,
      Has<Pressed>,
      &DropdownState,
      &Children,
      &mut BackgroundColor,
      &mut BorderColor,
      Has<UiDisabled>,
    ),
    (With<DropdownRoot>, Without<DropdownOption>),
  >,
  q_anchors: Query<
    (&ComputedNode, &UiGlobalTransform, Option<&InheritedVisibility>),
    With<DropdownRoot>,
  >,
  mut q_texts: Query<(&mut Text, &mut crate::i18n::I18nKey), With<DropdownText>>,
  q_arrow: Query<&DropdownArrow>,
  mut q_colors: Query<&mut TextColor>,
  mut q_popups: Query<(&mut DropdownPopup, &mut Node), Without<DropdownOption>>,
  mut q_options: Query<
    (
      &DropdownOption,
      &Hovered,
      Has<Pressed>,
      &mut Node,
      &mut BackgroundColor,
      &mut BorderColor,
      &Children,
    ),
    Without<DropdownPopup>,
  >,
) {
  let Some(theme) = theme else { return };
  let c = &theme.colors;
  let window = windows.single().ok();
  let sf = window.map(|w| w.scale_factor()).unwrap_or(1.0);
  // 窗口逻辑尺寸（与 `anchor_rect` 同坐标系），列表钳制用
  let win_size = window.map(|w| Vec2::new(w.width(), w.height()));
  let dt = time.delta_secs();
  let surface_elevated = color_of(&c.surface_elevated);
  let surface_overlay = color_of(&c.surface_overlay);
  let border = color_of(&c.border);
  let border_strong = color_of(&c.border_strong);
  let text_primary = color_of(&c.text_primary);
  let text_body = color_of(&c.text_body);
  let text_muted = color_of(&c.text_muted);

  // ---------- 1. 控件：关闭态文本 + 配色（展开/hover/禁用） ----------
  for (value, options, hovered, pressed, state, children, mut bg, mut border_c, disabled) in
    &mut q_roots
  {
    let inter = UiInteract::of(hovered, pressed);
    let selected = value.0.min(options.0.len().saturating_sub(1));
    let key = options.0.get(selected).cloned().unwrap_or_default();
    let shown = match i18n.as_ref() {
      Some(t) => t.resolve(&key),
      None => key.clone(),
    };
    let (target_bg, target_border, target_text) = dropdown_colors(&theme, state.open, inter);
    let (target_bg, target_border, target_text, arrow_text) = if disabled {
      (
        dim_color(target_bg),
        dim_color(target_border),
        dim_color(target_text),
        dim_color(text_muted),
      )
    } else {
      (target_bg, target_border, target_text, text_muted)
    };
    if bg.0 != target_bg {
      bg.0 = target_bg;
    }
    let target_border = BorderColor::all(target_border);
    if *border_c != target_border {
      *border_c = target_border;
    }
    for child in children.iter() {
      if let Ok((mut text, mut k)) = q_texts.get_mut(child) {
        if k.0 != key {
          k.0 = key.clone();
        }
        if text.0 != shown {
          text.0 = shown.clone();
        }
        if let Ok(mut col) = q_colors.get_mut(child)
          && col.0 != target_text
        {
          col.0 = target_text;
        }
        continue;
      }
      if q_arrow.get(child).is_ok()
        && let Ok(mut col) = q_colors.get_mut(child)
        && col.0 != arrow_text
      {
        col.0 = arrow_text;
      }
    }
  }

  // ---------- 2. 浮层：跟随锚点（窗口内钳制）+ 展开/滚动动画 + 选项配色 ----------
  for (mut popup, mut node) in &mut q_popups {
    let Ok((anchor_node, anchor_xform, visible)) = q_anchors.get(popup.owner) else { continue };
    if !visible.map(|v| v.get()).unwrap_or(true) {
      continue; // 菜单隐藏时冻结位置（关闭由 dropdown_system 负责）
    }
    let (selected, count) =
      q_roots.get(popup.owner).map(|r| (r.0.0, r.1.0.len())).unwrap_or((0, 0));
    let (mut pos, size) = anchor_rect(anchor_node, anchor_xform, sf);
    // 整份列表（n 项首尾相接，选中项与控件重合）必须留在窗口内并留 `WINDOW_MARGIN`；
    // 该规则优先于「选中项与控件重合」：贴边时整份列表平移，选中项不再对齐控件。
    if let Some(win) = win_size {
      let list_h = size.y * count as f32;
      let list_top = pos.y - selected as f32 * size.y;
      let top = list_top.clamp(WINDOW_MARGIN, (win.y - WINDOW_MARGIN - list_h).max(WINDOW_MARGIN));
      pos.y = top + selected as f32 * size.y;
      pos.x = pos.x.clamp(WINDOW_MARGIN, (win.x - size.x - WINDOW_MARGIN).max(WINDOW_MARGIN));
    }
    if node.left != px(pos.x) {
      node.left = px(pos.x);
    }
    if node.top != px(pos.y) {
      node.top = px(pos.y);
    }
    if node.width != px(size.x) {
      node.width = px(size.x);
    }
    if node.height != px(size.y) {
      node.height = px(size.y);
    }
    popup.t = (popup.t + dt / DROPDOWN_ANIM_SECS).min(1.0);
    let p = ease_in_out(popup.t);
    // 视觉下标向真源匀速靠拢（滚轮改选时的滑动动画）
    let target = selected as f32;
    let d = target - popup.sel;
    if d != 0.0 {
      let step = DROPDOWN_SCROLL_SPEED * dt;
      popup.sel = if step >= d.abs() { target } else { popup.sel + d.signum() * step };
    }
    for (opt, hovered, pressed, mut opt_node, mut opt_bg, mut opt_border, children) in
      &mut q_options
    {
      let inter = UiInteract::of(hovered, pressed);
      if opt.owner != popup.owner {
        continue;
      }
      // 选中项 (i - sel) = 0 → 与控件矩形重合；其余按缓动从它上下滑出
      let top = px((opt.index as f32 - popup.sel) * size.y * p);
      if opt_node.top != top {
        opt_node.top = top;
      }
      if opt_node.height != px(size.y) {
        opt_node.height = px(size.y);
      }
      // 配色：hover 提亮；选中项与关闭态控件同款（底/边框/主文本色）
      let hovered = inter.is_active();
      let (target_bg, target_border, target_text) = if hovered {
        (surface_overlay, border_strong, text_primary)
      } else if opt.index == selected {
        (surface_elevated, border, text_primary)
      } else {
        (surface_elevated, border, text_body)
      };
      if opt_bg.0 != target_bg {
        opt_bg.0 = target_bg;
      }
      let target_border = BorderColor::all(target_border);
      if *opt_border != target_border {
        *opt_border = target_border;
      }
      for child in children.iter() {
        if let Ok(mut col) = q_colors.get_mut(child)
          && col.0 != target_text
        {
          col.0 = target_text;
        }
      }
    }
  }
}
