//! FR-2 基础 widget 层：retained spawn API（token 驱动），三件套统一约定。
//!
//! ## 组件 API 约定（v2）
//!
//! 每个 widget 由三件套组成，全部自由函数、无 builder：
//!
//! 1. **`XxxConfig`**：全部参数进 Config（含必填如 text），`..default()` 补齐可选字段。
//!    布局字段（宽高/边距）不进 Config——调用方拿到 Handle 后自行改 `Node`。
//!    预设档（label 字号/颜色、panel 表面）用 enum 表达，颜色字号永远成对来自令牌。
//! 2. **`XxxHandle`**：spawn 返回类型，Deref 到根实体 `Entity`（零开销新类型）。
//!    多实体 widget 用具名字段：`ScrollViewHandle { entity, content }`、
//!    `TabViewHandle { entity, contents }`。不挂 sugar 方法，后续按需再加。
//! 3. **事件**：仅交互型 widget（button/slider/checkbox/tab_view）发事件；
//!    展示型（label/panel/plot/list/table/splitter）不发。组件永远是真源，
//!    事件只是通知——主动读值仍可 Query。命名：动作用名词（`UiClick`）、
//!    变化用 -ed 过去式（`SliderValueChanged`/`CheckboxToggled`/`TabChanged`）。
//!
//! 豁免：`splitter` 无任何配置项，不设 Config；`row`/`column`/`spacer` 等布局
//! 容器不纳入本层，直接用 bevy_ui 原生 `Node`。
//!
//! 响应式布局约定（FR-2）：容器级尺寸/边距优先 `Val::Percent` / `Val::Vw/Vh`；
//! 控件本体（按钮/滑杆）允许固定像素——面板锚定边缘，窗口任意尺寸下不越界不重叠。

pub mod button;
pub mod checkbox;
pub mod grid;
pub mod label;
pub mod list;
pub mod panel;
pub mod plot;
pub mod scroll_view;
pub mod slider;
pub mod splitter;
pub mod tab_view;
pub mod table;
pub mod toggle_switch;

pub use button::{
  ButtonConfig, ButtonHandle, ButtonVariant, InteractionPrev, UiClick, button, button_state_system,
};
pub use checkbox::{
  CheckboxBox, CheckboxConfig, CheckboxHandle, CheckboxToggled, checkbox, checkbox_state_system,
};
pub use grid::{GridConfig, GridHandle, UiGrid, grid, grid_cell};
pub use label::{LabelConfig, LabelHandle, LabelStyle, label};
pub use list::{ListConfig, ListHandle, RingList, list, ring_list_sync_system};
pub use panel::{PanelConfig, PanelHandle, PanelSurface, panel};
pub use plot::{
  PLOT_H, PLOT_W, PlotCanvas, PlotConfig, PlotData, PlotDomain, PlotExtents, PlotHandle,
  PlotLayout, PlotYAxis, blank_plot_image, plot, plot_redraw_system,
};
pub use scroll_view::{
  ScrollConfig, ScrollContent, ScrollView, ScrollViewHandle, ScrollViewport, scroll_view,
  scroll_view_system,
};
pub use slider::{
  SliderConfig, SliderHandle, SliderRange, SliderStep, SliderThumb, SliderValue,
  SliderValueChanged, UiSlider, clamp_step, slider, slider_drag_system, slider_visual_system,
};
pub use splitter::{Splitter, splitter};
pub use tab_view::{
  TabButton, TabChanged, TabConfig, TabContent, TabView, TabViewHandle, tab_view, tab_view_system,
};
pub use table::{TableCell, TableConfig, TableHandle, UiTable, table};
pub use toggle_switch::{
  ToggleKnob, ToggleSwitch, ToggleSwitchConfig, ToggleSwitchHandle, ToggleSwitchToggled,
  ToggleTrack, toggle_switch, toggle_switch_state_system,
};

use bevy::log::warn;
use bevy::prelude::*;
use bevy::text::FontSource;
use bevy::ui::widget::Label;

use crate::theme::{HexColor, UiTheme};

/// Disabled 态标记组件。挂在 widget 根节点上表示该控件不可交互、视觉暗一档。
///
/// 各 widget 状态机统一约定：检测到 `UiDisabled` 时跳过交互逻辑（不发事件、
/// 不翻转状态），并把当前配色经 [`dim_color`] 降亮后输出。
#[derive(Component, Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct UiDisabled;

/// Disabled 态降亮系数：RGB 通道乘以此值，保持 alpha。
/// 0.55 给出清晰的"灰化"观感，与主题 5 层表面的亮度差相当（约暗一档）。
const DISABLED_DIM: f32 = 0.55;

/// 将颜色暗一档（disabled 态通用）：线性空间下 RGB 乘 [`DISABLED_DIM`]，alpha 不变。
///
/// 用线性空间而非 sRGB 直接相乘，保证感知亮度均匀下降（sRGB 直接乘会偏暗）。
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
}

impl<'a> UiCtx<'a> {
  pub fn new(theme: &'a UiTheme, font: Option<&'a Handle<Font>>) -> Self {
    Self { theme, font }
  }

  /// TextFont.font 来源：
  /// - 显式 ThemeFont → 自定义字体（MapleMono 等含 CJK 的 TTF，通常作为项目唯一字体）
  /// - 否则 → `FontSource::default()`（= AssetId::<Font>::default() slot，由
  ///   GateUiPlugin 在主题字体加载后覆盖；即使 theme.font_path 是 None，
  ///   这条也将命中 Bevy 内置 FiraMono，结果与 FontSource::Handle(Handle::default()) 一致）。
  ///   之前用 `FontSource::SystemUi`——它会退到操作系统 UI 字体，字形与项目字体不一致，
  ///   且在英文系统上可能没有 CJK 字形。除非显式指定其他字体，所有 gate-ui 文本统一走
  ///   主题字体，满足"除非额外指定，全部用同一字体"的需求。
  pub fn font_source(&self) -> FontSource {
    match self.font {
      Some(h) => FontSource::from(h),
      None => FontSource::default(),
    }
  }
}

/// HexColor → Color（运行时构造的非法 hex 回退白色并 warn，不 panic）
pub fn color_of(hc: &HexColor) -> Color {
  hc.to_color().unwrap_or_else(|| {
    warn!(
      "invalid hex color {:?} in runtime value, fallback white",
      hc.0
    );
    Color::WHITE
  })
}

/// 方便构造 Val::Px
pub fn px(v: f32) -> Val {
  Val::Px(v)
}

/// 标签 bundle（button/checkbox 文本、list 条目、plot 极值文本共用）
pub(crate) fn label_bundle(ctx: &UiCtx, text: String, size: f32, color: Color) -> impl Bundle {
  (
    Name::new("ui-label"),
    Label,
    Text::new(text),
    TextFont {
      font: ctx.font_source(),
      font_size: bevy::text::FontSize::Px(size),
      ..default()
    },
    TextColor(color),
    TextLayout::default(),
  )
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
  commands
    .spawn((label_bundle(ctx, text, size, color), ChildOf(parent)))
    .id()
}
