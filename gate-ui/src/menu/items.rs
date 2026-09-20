//! 菜单通用组件：DebugWindow 列表里的 9 种列表单项。
//! 一行分「左|中|右」三部分（左列 `LEFT_COL_W`、右列 `RIGHT_COL_W` 固定且不相等，中间占余下）。
//! 交互行由 `menu_system` 驱动：写回 `super::model::MenuFile` 并对外发 `super::MenuActionEvent`，本文件只建节点与静态文案。

use bevy::prelude::*;
use bevy::text::{Justify, LineBreak, TextLayout as BevyTextLayout};

use super::consts::*;
use super::model::{InputField, MenuNode};
use crate::capture::MouseIntercept;
use crate::icon::Icon;
use crate::pointer::{UiInteract, UiInteractBundle};
use crate::widgets::{
  DropdownConfig, LabelConfig, LabelOverflow, LabelStyle, SliderConfig, TextInputConfig,
  TextInputKind, ToggleSwitchConfig, Tooltip, UiCtx, color_of, dropdown, label, px, slider,
  spawn_icon, text_input, toggle_switch,
};

/// 菜单项在模型里的角色（决定 menu_system 如何读写值）
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub enum MenuRole {
  /// 子菜单整行（点击进入下级）
  SubMenu,
  /// 按钮组第 i 个按钮
  Button(usize),
  /// 滑动条
  Slider,
  /// 切换组第 i 个选项
  SwitchOption(usize),
  /// 下拉框
  Dropdown,
  /// 开关项
  Toggle,
  /// 输入框第 i 个字段
  Input(usize),
  /// 颜色选择器的 HEX 输入框
  Color,
  /// 纯文本（只读展示）
  Text,
}

/// 菜单项身份：`path` = 节点 id 路径（root 下的第一段即根子项 id）
#[derive(Component, Clone, Debug, PartialEq)]
pub struct MenuItem {
  pub path: String,
  pub role: MenuRole,
}

/// 菜单按钮（切换组选项 / 按钮组按钮）标记：视觉由 menu_system 刷新
#[derive(Component, Debug, Default)]
pub struct MenuOptionButton;

/// 子菜单行标记
#[derive(Component, Debug, Default)]
pub struct MenuSubMenuRow;

/// 滑杆数值标签（`path` = 所属滑杆项的 id 路径）
#[derive(Component, Clone, Debug, PartialEq)]
pub struct MenuSliderValue {
  pub path: String,
}

/// 纯文本行的值标签（`path` = 所属文本项的 id 路径；左列名称之外的中间部分由它显示）
#[derive(Component, Clone, Debug, PartialEq)]
pub struct MenuTextValue {
  pub path: String,
}

/// 颜色预览色块（`path` = 所属颜色项的 id 路径）
#[derive(Component, Clone, Debug, PartialEq)]
pub struct MenuColorSwatch {
  pub path: String,
}

/// 菜单行的按压状态（menu_system 独占；与 button widget 的 `UiInteractPrev` 不共享）
#[derive(Component, Clone, Copy, Debug, Default, PartialEq)]
pub struct MenuPressPrev(pub UiInteract);

/// 一行的通用外壳（宽 100%、高 `ITEM_H`、横向排列、无分割线）
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
  ))
}

/// 固定宽度 + 指定对齐的文本（左名称 / 右数值共用）；`key` = i18n key（无解析器 → 原样显示）。
/// 超宽走中间省略；`label()` 已挂 clip Node，故这里只补 `width` 字段，不覆盖 Node。
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

/// 占满剩余宽度的文本（`key` = i18n key）
fn grow_label(ctx: &UiCtx, parent: &mut ChildSpawner, key: &str, style: LabelStyle) -> Entity {
  let e = *label(ctx, parent, LabelConfig { text: ctx.text(key), style, ..default() });
  parent
    .world_mut()
    .entity_mut(e)
    .insert((crate::i18n::I18nKey::new(key), Node { flex_grow: 1.0, ..default() }));
  e
}

/// 让建好的控件根节点在行内占满剩余宽度。
/// 只改 `flex_grow` 字段——整体覆盖 `Node` 会连自带的 `height`/`padding`/`min_width`/`overflow` 一起抹掉。
fn grow_in_row(world: &mut World, e: Entity) {
  if let Some(mut n) = world.get_mut::<Node>(e) {
    n.flex_grow = 1.0;
  }
}

/// 一行按钮（切换组选项 / 按钮组按钮）：等高、等宽（flex_grow）、无圆角
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
      // flex_basis 0 + flex_grow：各按钮等宽（不设 basis 则按文字宽度分配）
      flex_basis: px(0.0),
      flex_grow: 1.0,
      // 高度由按钮组容器（CTRL_H）决定，上下留白由容器在行内居中保证
      height: Val::Percent(100.0),
      align_items: AlignItems::Center,
      justify_content: JustifyContent::Center,
      // 首项画左边框，其余靠前一项右边框，相邻合成 1px
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

/// 建一个菜单项（返回行的根实体）。
/// 提示（`MenuNode::tooltip`）挂在整行上，悬停行内任意位置都出提示。
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
    MenuNode::Slider { label: key, value, min, max, step, decimals, .. } => slider_row(
      ctx,
      parent,
      key,
      // step <= 0 = 连续（菜单模型用 0 表达「无档位」）
      SliderConfig {
        min: *min,
        max: *max,
        value: *value,
        step: (*step > 0.0).then_some(*step),
        ..default()
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
    MenuNode::Toggle { label: key, checked, .. } => toggle_row(ctx, parent, key, *checked, &path),
    MenuNode::Input { label: key, fields, .. } => input_row(ctx, parent, key, fields, &path),
    MenuNode::Color { label: key, hex, .. } => color_row(ctx, parent, key, hex, &path),
    MenuNode::Text { text: key, .. } => text_row(ctx, parent, key, &path),
  };
  // 提示也是 i18n key：解析后挂整行，保留 key 供语言切换重解析
  if let Some(key) = node.tooltip() {
    parent
      .world_mut()
      .entity_mut(row)
      .insert((Tooltip::new(ctx.text(key)), crate::i18n::I18nKey::new(key)));
  }
  row
}

/// 路径拼接（root = ""）
pub fn join_path(parent: &str, child: &str) -> String {
  if parent.is_empty() { child.to_string() } else { format!("{parent}/{child}") }
}

/// 子菜单：文字左对齐 + 右侧向右箭头，整行为按钮（hover 高亮由 menu_system 刷新）
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

/// 按钮组：等宽按钮（有名称时 左|中右，无名称时占满整行）
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
      // 控件高 = CTRL_H，行的 align_items: Center 让它上下各留 (ITEM_H-CTRL_H)/2
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

/// 滑动条：「左|中|右」名称 / 滑杆 / 数值（右侧数值颜色暗一档）
fn slider_row(
  ctx: &UiCtx,
  parent: &mut ChildSpawner,
  key: &str,
  config: SliderConfig,
  decimals: u32,
  path: &str,
) -> Entity {
  let value = config.value;
  let mut ec = base_row(parent, "menu-slider");
  let row = ec.id();
  ec.with_children(|r| {
    fixed_label(ctx, r, key, LEFT_COL_W, LabelStyle::Body, Justify::Left);
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
      // 不能带 NoWrap：bevy_ui 对 NoWrap 用无界宽度排版，Justify::Right 会失效
      BevyTextLayout { justify: Justify::Right, ..default() },
    ));
  });
  row
}

/// 切换组：无空隙并排按钮（有名称时 左|中右）
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
      // 控件高 = CTRL_H，行的 align_items: Center 让它上下各留 (ITEM_H-CTRL_H)/2
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

/// 下拉框：「左|中右」名称 + 下拉控件（样式同输入框，框内右侧带展开箭头）
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

/// 开关项：「左中|右」文字 + toggle
fn toggle_row(
  ctx: &UiCtx,
  parent: &mut ChildSpawner,
  key: &str,
  checked: bool,
  path: &str,
) -> Entity {
  let mut ec = base_row(parent, "menu-toggle");
  let row = ec.id();
  ec.with_children(|r| {
    grow_label(ctx, r, key, LabelStyle::Body);
    let te = *toggle_switch(ctx, r, ToggleSwitchConfig { text: None, checked, disabled: false });
    r.world_mut()
      .entity_mut(te)
      .insert(MenuItem { path: path.to_string(), role: MenuRole::Toggle });
  });
  row
}

/// 输入框：「左|中|右」名称 + 一个或多个等宽输入框
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
      // 字段前缀同样是 i18n key
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

/// 颜色选择器：「左|中|右」名称 / HEX 输入框 / 色块
fn color_row(ctx: &UiCtx, parent: &mut ChildSpawner, key: &str, hex: &str, path: &str) -> Entity {
  let m = &ctx.theme.metrics;
  let mut ec = base_row(parent, "menu-color");
  let row = ec.id();
  ec.with_children(|r| {
    fixed_label(ctx, r, key, LEFT_COL_W, LabelStyle::Body, Justify::Left);
    let ie = *text_input(
      ctx,
      r,
      TextInputConfig { text: hex.to_string(), kind: TextInputKind::Text, disabled: false },
    );
    grow_in_row(r.world_mut(), ie);
    r.world_mut().entity_mut(ie).insert(MenuItem { path: path.to_string(), role: MenuRole::Color });
    r.spawn((
      Name::new("menu-color-swatch"),
      MenuColorSwatch { path: path.to_string() },
      Node {
        width: px(RIGHT_COL_W),
        height: px(CTRL_H),
        border: UiRect::all(px(m.border_width)),
        // 与左侧 HEX 输入框留出间距
        margin: UiRect::left(px(m.spacing.sm)),
        ..default()
      },
      BackgroundColor(
        crate::parse_hex_color(hex)
          .map_or(Color::NONE, |[r_, g, b, a]| Color::srgba_u8(r_, g, b, a)),
      ),
      BorderColor::all(color_of(&ctx.theme.colors.border)),
    ));
  });
  row
}

/// 纯文本行：「左|中右」名称 + 值（右侧对齐到行右缘）；值由调用方运行期改写（见 `MenuTextValue`），初值为空串。
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
      // 不能带 NoWrap（同滑杆数值标签）：bevy_ui 对 NoWrap 用无界宽度排版，Justify::Right 失效
      BevyTextLayout { justify: Justify::Right, ..default() },
    ));
  });
  row
}
