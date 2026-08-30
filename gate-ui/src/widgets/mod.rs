//! FR-2 基础 widget 层：retained spawn API（UiCtx + ChildSpawner → Entity），token 驱动。
//!
//! 响应式布局约定（FR-2）：容器级尺寸/边距优先 `Val::Percent` / `Val::Vw/Vh`；
//! 控件本体（按钮/滑杆）允许固定像素——面板锚定边缘，窗口任意尺寸下不越界不重叠。

pub mod button;
pub mod checkbox;
pub mod label;
pub mod list;
pub mod panel;
pub mod plot;
pub mod slider;

pub use button::{InteractionPrev, UiClick, button, button_state_system};
pub use checkbox::{CheckboxBox, checkbox, checkbox_state_system};
pub use label::{label, label_muted};
pub use list::{RingList, list, ring_list_sync_system};
pub use panel::panel;
pub use plot::{
  PLOT_H, PLOT_W, PlotCanvas, PlotData, PlotDomain, PlotExtents, blank_plot_image, plot,
  plot_redraw_system,
};
pub use slider::{
  SliderRange, SliderStep, SliderThumb, SliderValue, UiSlider, clamp_step, slider,
  slider_drag_system, slider_visual_system,
};

use bevy::log::warn;
use bevy::prelude::*;
use bevy::text::FontSource;
use bevy::ui::widget::Label;

use crate::theme::{HexColor, UiTheme};

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
  /// 之前用 `FontSource::SystemUi`——它会退到操作系统 UI 字体，字形与项目字体不一致，
  /// 且在英文系统上可能没有 CJK 字形。除非显式指定其他字体，所有 gate-ui 文本统一走
  /// 主题字体，满足"除非额外指定，全部用同一字体"的需求。
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
