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

#[derive(Component, Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct UiDisabled;

pub fn dim_color(color: Color) -> Color {
  let lin = color.to_linear();
  let mut v = lin.to_vec4();
  v[0] *= DISABLED_DIM;
  v[1] *= DISABLED_DIM;
  v[2] *= DISABLED_DIM;
  Color::linear_rgba(v[0], v[1], v[2], v[3])
}

pub struct UiCtx<'a> {
  pub theme: &'a UiTheme,
  pub font: Option<&'a Handle<Font>>,
  pub icon_font: Option<&'a Handle<Font>>,
  pub translate: Option<crate::i18n::TranslatorFn>,
}

impl<'a> UiCtx<'a> {
  pub fn new(theme: &'a UiTheme, font: Option<&'a Handle<Font>>) -> Self {
    Self { theme, font, icon_font: None, translate: None }
  }

  pub fn with_icon_font(mut self, icon_font: Option<&'a Handle<Font>>) -> Self {
    self.icon_font = icon_font;
    self
  }

  pub fn with_translate(mut self, translate: Option<crate::i18n::TranslatorFn>) -> Self {
    self.translate = translate;
    self
  }

  pub fn text(&self, key: &str) -> String {
    match &self.translate {
      Some(f) => f(key),
      None => key.to_string(),
    }
  }

  pub fn font_source(&self) -> FontSource {
    match self.font {
      Some(h) => FontSource::from(h),
      None => FontSource::default(),
    }
  }

  pub fn icon_source(&self) -> FontSource {
    match self.icon_font {
      Some(h) => FontSource::from(h),
      None => self.font_source(),
    }
  }
}

pub fn color_of(hc: &HexColor) -> Color {
  hc.to_color().unwrap_or_else(|| {
    warn!("invalid hex {:?} in runtime value → white", hc.0);
    Color::WHITE
  })
}

pub fn px(v: f32) -> Val {
  Val::Px(v)
}

pub(crate) fn label_bundle(ctx: &UiCtx, text: String, size: f32, color: Color) -> impl Bundle {
  label_bundle_attrs(ctx, text, size, color, label::FontAttrs::default())
}

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

pub(crate) fn spawn_icon(
  ctx: &UiCtx,
  parent: &mut ChildSpawner,
  glyph: &str,
  size: f32,
  color: Color,
) -> Entity {
  parent.spawn(icon_bundle(ctx, glyph, size, color)).id()
}

pub(crate) fn spawn_label(
  ctx: &UiCtx,
  parent: &mut ChildSpawner,
  text: String,
  size: f32,
  color: Color,
) -> Entity {
  parent.spawn(label_bundle(ctx, text, size, color)).id()
}

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
