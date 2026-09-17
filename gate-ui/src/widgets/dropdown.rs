//! dropdown：单行下拉框（点击展开；选中项与控件原位重合，其余选项自上方/下方缓动展开）。
//! 框体与 `text_input` 同款（底/边框/高度/内距/字号），右侧多一个箭头图标。
//! 展开浮层是顶层实体（无父节点 + `GlobalZIndex`，不进调用方 UI 树），位置每帧从控件锚点重算。

use std::ops::Deref;

use bevy::prelude::*;
use bevy::text::LineBreak;
use bevy::ui::{ComputedNode, FocusPolicy, Interaction, UiGlobalTransform};
use bevy::window::PrimaryWindow;

use super::{
  FontAttrs, InteractionPrev, LabelConfig, LabelStyle, UiCtx, UiDisabled, color_of, dim_color,
  icon_bundle, label, label_bundle_attrs, px, spawn_icon,
};
use crate::capture::MouseIntercept;
use crate::icon::{Icon, IconFont};
use crate::theme::{ThemeFont, UiTheme};

/// 展开/收起动画时长（秒）
pub const DROPDOWN_ANIM_SECS: f32 = 0.2;
/// 下拉箭头字号（px）
const DROPDOWN_ARROW_SIZE: f32 = 12.0;
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
  /// 展开进度 0..=1
  pub t: f32,
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
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct DropdownState {
  /// 展开中（浮层存活期间恒为 true）
  pub open: bool,
}

/// 选中项变化事件（EntityEvent，target = 控件根实体，由用户点击选项触发）
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
  let (bg, border, text_color) = dropdown_colors(ctx.theme, false, Interaction::None);
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
    Interaction::default(),
    InteractionPrev::default(),
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
    FocusPolicy::Block,
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
fn dropdown_colors(theme: &UiTheme, open: bool, inter: Interaction) -> (Color, Color, Color) {
  let c = &theme.colors;
  let border = if open {
    color_of(&c.accent_text)
  } else if inter != Interaction::None {
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
fn anchor_rect(node: &ComputedNode, xform: &UiGlobalTransform, scale_factor: f32) -> (Vec2, Vec2) {
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
/// 选项与控件同高/同内距/同字号：`t = 1` 时选中项与关闭态控件逐像素重合；选中项 `ZIndex` 最高。
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
  let popup = commands
    .spawn((
      Name::new("ui-dropdown-popup"),
      DropdownPopup { owner, t: 0.0 },
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
      FocusPolicy::Pass,
      GlobalZIndex(DROPDOWN_Z),
    ))
    .id();
  for (i, key) in options.0.iter().enumerate() {
    let option = commands
      .spawn((
        Name::new("ui-dropdown-option"),
        ChildOf(popup),
        DropdownOption { owner, index: i },
        Interaction::default(),
        InteractionPrev::default(),
        Node {
          position_type: PositionType::Absolute,
          // 动画每帧写 top = (i - selected) · h · ease(t)；t = 0 时全部叠在锚点矩形上
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
        FocusPolicy::Block,
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
    // 选中项也画箭头图标（与控件展开态一致）
    if i == selected {
      commands.spawn((
        icon_bundle(ctx, Icon::ChevronDown.glyph(), DROPDOWN_ARROW_SIZE, color_of(&c.text_muted)),
        ChildOf(option),
        DropdownArrow,
      ));
    }
  }
}

/// 关闭：销毁该控件的浮层 + 复位展开态
fn close_popup(
  commands: &mut Commands,
  owner: Entity,
  popups: &Query<(Entity, &DropdownPopup)>,
  state: &mut DropdownState,
) {
  for (e, p) in popups.iter() {
    if p.owner == owner {
      commands.entity(e).despawn();
    }
  }
  state.open = false;
}

/// 指针交互：点控件展开 / 点选项选中 / 鼠标移出列表取消。
/// 「鼠标在列表内」= 落在任一选项矩形内（相邻选项间距 ≤ 高度 → 并集连续）；指针离开窗口同样视为移出。
#[allow(clippy::type_complexity, clippy::too_many_arguments)] // Bevy system：多组件查询/入参签名固有
pub fn dropdown_system(
  mut commands: Commands,
  windows: Query<&Window, With<PrimaryWindow>>,
  theme: Option<Res<UiTheme>>,
  font: Option<Res<ThemeFont>>,
  icon: Option<Res<IconFont>>,
  i18n: Option<Res<crate::i18n::UiTranslator>>,
  mut q_roots: Query<
    (
      Entity,
      &Interaction,
      &mut InteractionPrev,
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
    (
      Entity,
      &Interaction,
      &mut InteractionPrev,
      &DropdownOption,
      &ComputedNode,
      &UiGlobalTransform,
    ),
    (Without<DropdownRoot>, Without<DropdownPopup>),
  >,
  q_popups: Query<(Entity, &DropdownPopup)>,
) {
  let cursor = windows.single().ok().and_then(|w| w.physical_cursor_position());
  let sf = windows.single().ok().map(|w| w.scale_factor()).unwrap_or(1.0);

  // ---------- 1. 选项：点击判定 + 命中收集 ----------
  let mut clicked: Option<(Entity, usize)> = None;
  let mut hits: Vec<(Entity, bool)> = Vec::new();
  for (_, inter, mut prev, opt, node, xform) in &mut q_options {
    if prev.0 == Interaction::Pressed && *inter == Interaction::Hovered {
      clicked = Some((opt.owner, opt.index));
    }
    prev.0 = *inter;
    hits.push((opt.owner, cursor.is_some_and(|c| node.contains_point(*xform, c))));
  }
  let inside_list = |owner: Entity| hits.iter().any(|(o, hit)| *o == owner && *hit);

  // ---------- 2. 控件：选项选中 / 展开 / 收起 / 锚点已消失的浮层回收 ----------
  for (e, inter, mut prev, mut state, options, mut value, node, xform, visible, disabled) in
    &mut q_roots
  {
    let clicked_self = prev.0 == Interaction::Pressed && *inter == Interaction::Hovered;
    prev.0 = *inter;
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
    if clicked_self {
      if state.open {
        close_popup(&mut commands, e, &q_popups, &mut state);
      } else if let Some(ctx) = ctx_of(&theme, &font, &icon, &i18n) {
        spawn_popup(&mut commands, &ctx, e, options, value.0, anchor_rect(node, xform, sf));
        state.open = true;
      }
    } else if state.open && (!visible.map(|v| v.get()).unwrap_or(true) || !inside_list(e)) {
      // 鼠标移出列表（或菜单被隐藏）→ 取消选择
      close_popup(&mut commands, e, &q_popups, &mut state);
    }
  }
  // 锚点控件已销毁 → 浮层立即回收，不留孤儿
  for (popup, p) in &q_popups {
    if q_roots.get(p.owner).is_err() {
      commands.entity(popup).despawn();
    }
  }
}

/// 视觉：关闭态文本/配色 + 浮层跟随锚点 + 展开动画 + 选项配色（每帧重算）
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
      &Interaction,
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
    (&DropdownOption, &Interaction, &mut Node, &mut BackgroundColor, &mut BorderColor, &Children),
    Without<DropdownPopup>,
  >,
) {
  let Some(theme) = theme else { return };
  let c = &theme.colors;
  let sf = windows.single().ok().map(|w| w.scale_factor()).unwrap_or(1.0);
  let dt = time.delta_secs();
  let surface_elevated = color_of(&c.surface_elevated);
  let surface_overlay = color_of(&c.surface_overlay);
  let border = color_of(&c.border);
  let border_strong = color_of(&c.border_strong);
  let text_primary = color_of(&c.text_primary);
  let text_body = color_of(&c.text_body);
  let text_muted = color_of(&c.text_muted);

  // ---------- 1. 控件：关闭态文本 + 配色（展开/hover/禁用） ----------
  for (value, options, inter, state, children, mut bg, mut border_c, disabled) in &mut q_roots {
    let selected = value.0.min(options.0.len().saturating_sub(1));
    let key = options.0.get(selected).cloned().unwrap_or_default();
    let shown = match i18n.as_ref() {
      Some(t) => t.resolve(&key),
      None => key.clone(),
    };
    let (target_bg, target_border, target_text) = dropdown_colors(&theme, state.open, *inter);
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

  // ---------- 2. 浮层：跟随锚点 + 展开动画 + 选项配色 ----------
  for (mut popup, mut node) in &mut q_popups {
    let Ok((anchor_node, anchor_xform, visible)) = q_anchors.get(popup.owner) else { continue };
    if !visible.map(|v| v.get()).unwrap_or(true) {
      continue; // 菜单隐藏时冻结位置（关闭由 dropdown_system 负责）
    }
    let (pos, size) = anchor_rect(anchor_node, anchor_xform, sf);
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
    let selected = q_roots.get(popup.owner).map(|r| r.0.0).unwrap_or(0);
    for (opt, inter, mut opt_node, mut opt_bg, mut opt_border, children) in &mut q_options {
      if opt.owner != popup.owner {
        continue;
      }
      // 选中项 (i - selected) = 0 → 恒与控件矩形重合；其余按缓动从它背后滑出
      let top = px((opt.index as f32 - selected as f32) * size.y * p);
      if opt_node.top != top {
        opt_node.top = top;
      }
      if opt_node.height != px(size.y) {
        opt_node.height = px(size.y);
      }
      // 配色：hover 提亮；选中项与关闭态控件同款（底/边框/主文本色）
      let hovered = *inter != Interaction::None;
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
