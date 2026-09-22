//! 基础 widget 层：retained spawn API（token 驱动），三件套统一约定。
//! `XxxConfig` 收全部参数（布局字段除外，调用方拿 Handle 后自行改 `Node`）；`XxxHandle` Deref 到根实体 `Entity`。
//! 仅交互型 widget 发名词/-ed 事件（组件为真源）；`splitter` 无 Config；容器尺寸优先 `Val::Percent`/`Val::Vw/Vh`。

pub mod button;
pub mod checkbox;
pub mod consts;
pub mod dropdown;
pub mod grid;
pub mod label;
pub mod list;
pub mod markdown;
pub mod panel;
pub mod plot;
pub mod scroll_view;
pub mod slider;
pub mod splitter;
pub mod tab_view;
pub mod table;
pub mod text_input;
pub mod toggle_switch;
pub mod tooltip;

pub use button::{ButtonConfig, ButtonHandle, ButtonVariant, UiClick, button, button_state_system};
pub use checkbox::{
  CheckboxBox, CheckboxConfig, CheckboxHandle, CheckboxToggled, checkbox, checkbox_state_system,
};
pub use dropdown::{
  DropdownArrow, DropdownBackdrop, DropdownChanged, DropdownConfig, DropdownHandle, DropdownOption,
  DropdownOptions, DropdownPopup, DropdownRoot, DropdownState, DropdownText, DropdownValue,
  dropdown, dropdown_system, dropdown_visual_system,
};
pub use grid::{GridConfig, GridHandle, UiGrid, grid, grid_cell};
pub use label::{
  EllipsisText, FontAttrs, LabelConfig, LabelHandle, LabelOverflow, LabelStyle, label,
  label_ellipsis_system, middle_ellipsis,
};
pub use list::{ListConfig, ListHandle, RingList, list, ring_list_sync_system};
pub use markdown::{MarkdownConfig, MarkdownHandle, MarkdownView, markdown, markdown_set_text};
pub use panel::{PanelConfig, PanelHandle, PanelSurface, panel};
pub use plot::{
  PlotCanvas, PlotConfig, PlotData, PlotDomain, PlotExtents, PlotHandle, PlotLayout, PlotYAxis,
  blank_plot_image, plot, plot_redraw_system,
};
pub use scroll_view::{
  ScrollConfig, ScrollContent, ScrollView, ScrollViewHandle, ScrollViewport, scroll_view,
  scroll_view_system,
};
pub use slider::{
  SliderConfig, SliderDrag, SliderHandle, SliderRange, SliderStep, SliderThumb, SliderValue,
  SliderValueChanged, UiSlider, clamp_step, slider, slider_drag_system, slider_visual_system,
};
pub use splitter::{Splitter, splitter};
pub use tab_view::{
  TabButton, TabChanged, TabConfig, TabContent, TabView, TabViewHandle, tab_view, tab_view_system,
};
pub use table::{TableCell, TableConfig, TableHandle, UiTable, table};
pub use text_input::{
  TextInputChanged, TextInputConfig, TextInputFocus, TextInputHandle, TextInputKind, TextInputRoot,
  TextInputState, TextInputText, TextInputValue, text_input, text_input_keyboard_system,
  text_input_pointer_system, text_input_visual_system,
};
pub use toggle_switch::{
  ToggleKnob, ToggleSwitch, ToggleSwitchConfig, ToggleSwitchHandle, ToggleSwitchToggled,
  ToggleTrack, toggle_switch, toggle_switch_state_system,
};
pub use tooltip::{Tooltip, TooltipLayer, TooltipLayerEntity, TooltipText, tooltip_system};

use bevy::log::warn;
use bevy::prelude::*;
use bevy::text::FontSource;
use bevy::ui::widget::Label;

use crate::theme::{HexColor, UiTheme};
use crate::widgets::consts::DISABLED_DIM;

/// Disabled 态标记组件（挂在 widget 根节点）。状态机检测到它时跳过交互逻辑（不发事件、不翻转状态），
/// 并把配色经 `dim_color` 降亮。
#[derive(Component, Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct UiDisabled;

/// 将颜色暗一档（disabled 态通用）：线性空间下 RGB 乘 `DISABLED_DIM`，alpha 不变。
pub fn dim_color(color: Color) -> Color {
  let lin = color.to_linear();
  let mut v = lin.to_vec4();
  v[0] *= DISABLED_DIM;
  v[1] *= DISABLED_DIM;
  v[2] *= DISABLED_DIM;
  Color::linear_rgba(v[0], v[1], v[2], v[3])
}

/// spawn 上下文：主题令牌 + 字体（None → SystemUi 系统字体）
pub struct UiCtx<'a> {
  pub theme: &'a UiTheme,
  pub font: Option<&'a Handle<Font>>,
  /// 图标字体（FontAwesome Free Solid）；None → 图标回退 `Self::font`
  pub icon_font: Option<&'a Handle<Font>>,
  /// 文案解析器（key → 当前语言文案）；None → key 原样当文案
  pub translate: Option<crate::i18n::TranslatorFn>,
}

impl<'a> UiCtx<'a> {
  pub fn new(theme: &'a UiTheme, font: Option<&'a Handle<Font>>) -> Self {
    Self { theme, font, icon_font: None, translate: None }
  }

  /// 补上图标字体（菜单容器的标题栏/子菜单箭头用）
  pub fn with_icon_font(mut self, icon_font: Option<&'a Handle<Font>>) -> Self {
    self.icon_font = icon_font;
    self
  }

  /// 补上文案解析器（取 `crate::i18n::UiTranslator::handle`）；控件文案字段写 key 时经 `Self::text` 解析
  pub fn with_translate(mut self, translate: Option<crate::i18n::TranslatorFn>) -> Self {
    self.translate = translate;
    self
  }

  /// key → 当前语言文案（未注入解析器 → 原样返回，方便 TOML/测试直接写字面量）
  pub fn text(&self, key: &str) -> String {
    match &self.translate {
      Some(f) => f(key),
      None => key.to_string(),
    }
  }

  /// TextFont.font 来源：显式 ThemeFont → 自定义字体；否则 `FontSource::default()`（`AssetId::<Font>::default()`
  /// slot，GateUiPlugin 加载主题字体后覆盖）。所有 gate-ui 文本统一走主题字体。
  pub fn font_source(&self) -> FontSource {
    match self.font {
      Some(h) => FontSource::from(h),
      None => FontSource::default(),
    }
  }

  /// 图标字体的来源（未配置 → 回退正文来源）
  pub fn icon_source(&self) -> FontSource {
    match self.icon_font {
      Some(h) => FontSource::from(h),
      None => self.font_source(),
    }
  }
}

/// HexColor → Color（运行时构造的非法 hex 回退白色并 warn，不 panic）
pub fn color_of(hc: &HexColor) -> Color {
  hc.to_color().unwrap_or_else(|| {
    warn!("invalid hex {:?} in runtime value → white", hc.0);
    Color::WHITE
  })
}

/// 方便构造 Val::Px
pub fn px(v: f32) -> Val {
  Val::Px(v)
}

/// 标签 bundle（button/checkbox 文本、list 条目、plot 极值文本共用；字体属性全默认）
pub(crate) fn label_bundle(ctx: &UiCtx, text: String, size: f32, color: Color) -> impl Bundle {
  label_bundle_attrs(ctx, text, size, color, label::FontAttrs::default())
}

/// 标签 bundle（逐属性覆盖版：`attrs` 里 None 的项保持 `TextFont` 默认值）。`attrs` 须按值传入。
pub(crate) fn label_bundle_attrs(
  ctx: &UiCtx,
  text: String,
  size: f32,
  color: Color,
  attrs: label::FontAttrs,
) -> impl Bundle {
  let mut font =
    TextFont { font: ctx.font_source(), font_size: bevy::text::FontSize::Px(size), ..default() };
  attrs.apply(&mut font);
  (Name::new("ui-label"), Label, Text::new(text), font, TextColor(color), TextLayout::default())
}

/// 图标文本 bundle（FontAwesome 字形；字号单独给，不受 LabelStyle 档位约束）
pub(crate) fn icon_bundle(ctx: &UiCtx, glyph: &str, size: f32, color: Color) -> impl Bundle {
  (
    Name::new("ui-icon"),
    Label,
    Text::new(glyph.to_string()),
    TextFont { font: ctx.icon_source(), font_size: bevy::text::FontSize::Px(size), ..default() },
    TextColor(color),
    TextLayout::default(),
  )
}

/// 图标文本实体（内部复用：标题栏/子菜单箭头）
pub(crate) fn spawn_icon(
  ctx: &UiCtx,
  parent: &mut ChildSpawner,
  glyph: &str,
  size: f32,
  color: Color,
) -> Entity {
  parent.spawn(icon_bundle(ctx, glyph, size, color)).id()
}

/// 标签文本实体（内部复用：button/checkbox 文本）
pub(crate) fn spawn_label(
  ctx: &UiCtx,
  parent: &mut ChildSpawner,
  text: String,
  size: f32,
  color: Color,
) -> Entity {
  parent.spawn(label_bundle(ctx, text, size, color)).id()
}

/// 标签文本实体（commands 版：list 同步重建等 commands 上下文）
pub(crate) fn spawn_label_cmd(
  ctx: &UiCtx,
  commands: &mut Commands,
  parent: Entity,
  text: String,
  size: f32,
  color: Color,
) -> Entity {
  commands.spawn((label_bundle(ctx, text, size, color), ChildOf(parent))).id()
}
