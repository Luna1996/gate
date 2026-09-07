//! gate-ui：bevy_ui 之上的自研组件库基座（2.7a）。
//!
//! 三层结构：主题令牌（theme）→ widget 层（widgets）→ 输入门控（capture，Task 4）。
//! 只依赖 bevy + serde/ron，不引入第三方 UI 框架。

pub mod capture;
pub mod theme;
pub mod widgets;
pub mod world_anchor;

pub use capture::{UiPointerCaptured, ui_pointer_capture_system};
pub use world_anchor::{
  AnchorBaseFont, AnchorCamera, WorldAnchor, anchor_distance_scale, project_to_screen,
  world_anchor_label, world_anchor_system,
};

pub use theme::{
  AUTOFIT_BASE_HEIGHT, FontSize, GateUiPlugin, HexColor, Spacing, THEME_ASSET_PATH, ThemeColors,
  ThemeFont, ThemeMetrics, UiTheme, UiThemeHandle, UiThemeState, autofit_ui_scale, default_theme,
  parse_hex_color, parse_theme_ron,
};
pub use widgets::{
  ButtonConfig, ButtonHandle, ButtonVariant, CheckboxBox, CheckboxConfig, CheckboxHandle,
  CheckboxToggled, GridConfig, GridHandle, InteractionPrev, LabelConfig, LabelHandle, LabelStyle,
  ListConfig, ListHandle, PanelConfig, PanelHandle, PanelSurface, PlotCanvas, PlotConfig, PlotData,
  PlotDomain, PlotExtents, PlotHandle, PlotLayout, PlotYAxis, RingList, ScrollConfig,
  ScrollContent, ScrollView, ScrollViewHandle, ScrollViewport, SliderConfig, SliderHandle,
  SliderRange, SliderStep, SliderThumb, SliderValue, SliderValueChanged, TabButton, TabChanged,
  TabConfig, TabContent, TabView, TabViewHandle, TableCell, TableConfig, TableHandle, ToggleKnob,
  ToggleSwitch, ToggleSwitchConfig, ToggleSwitchHandle, ToggleSwitchToggled, ToggleTrack, UiClick,
  UiCtx, UiGrid, UiSlider, UiTable, blank_plot_image, button, button_state_system, checkbox,
  checkbox_state_system, clamp_step, color_of, grid, grid_cell, label, list, panel, plot,
  plot_redraw_system, px, ring_list_sync_system, scroll_view, scroll_view_system, slider,
  slider_drag_system, slider_visual_system, splitter, tab_view, tab_view_system, table,
  toggle_switch, toggle_switch_state_system,
};
