//! gate-ui：bevy_ui 之上的自研组件库基座。
//!
//! 三层结构：主题令牌（theme）→ widget 层（widgets）→ 输入门控（capture）。
//! 只依赖 bevy + serde/ron，不引入第三方 UI 框架。

pub mod capture;
pub mod i18n;
pub mod icon;
pub mod menu;
pub mod theme;
pub mod widgets;
pub mod world_anchor;

pub use capture::{MouseIntercept, MouseIntercepted, UiPointerCaptured, ui_pointer_capture_system};
pub use i18n::{I18nKey, UiTranslator, i18n_refresh_system};
pub use icon::{Icon, IconFont};
pub use menu::{
  DebugMenu, DebugMenuHandle, DebugMenuRoot, InputField, MenuAction, MenuActionEvent, MenuFile,
  MenuItem, MenuNode, MenuRole, MenuTextValue, WindowState, menu_model, menu_selected, menu_system,
  menu_text, menu_toggle, menu_value, spawn_debug_menu,
};
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
  CheckboxToggled, DROPDOWN_ANIM_SECS, DropdownArrow, DropdownChanged, DropdownConfig,
  DropdownHandle, DropdownOption, DropdownOptions, DropdownPopup, DropdownRoot, DropdownState,
  DropdownText, DropdownValue, ELLIPSIS, EllipsisText, FontAttrs, GridConfig, GridHandle,
  InteractionPrev, LabelConfig, LabelHandle, LabelOverflow, LabelStyle, ListConfig, ListHandle,
  PanelConfig, PanelHandle, PanelSurface, PlotCanvas, PlotConfig, PlotData, PlotDomain,
  PlotExtents, PlotHandle, PlotLayout, PlotYAxis, RingList, ScrollConfig, ScrollContent,
  ScrollView, ScrollViewHandle, ScrollViewport, SliderConfig, SliderHandle, SliderRange,
  SliderStep, SliderThumb, SliderValue, SliderValueChanged, TabButton, TabChanged, TabConfig,
  TabContent, TabView, TabViewHandle, TableCell, TableConfig, TableHandle, TextInputChanged,
  TextInputConfig, TextInputFocus, TextInputHandle, TextInputKind, TextInputRoot, TextInputState,
  TextInputText, TextInputValue, ToggleKnob, ToggleSwitch, ToggleSwitchConfig, ToggleSwitchHandle,
  ToggleSwitchToggled, ToggleTrack, Tooltip, TooltipLayer, TooltipLayerEntity, TooltipText,
  UiClick, UiCtx, UiGrid, UiSlider, UiTable, blank_plot_image, button, button_state_system,
  checkbox, checkbox_state_system, clamp_step, color_of, dropdown, dropdown_system,
  dropdown_visual_system, grid, grid_cell, label, label_ellipsis_system, list, middle_ellipsis,
  panel, plot, plot_redraw_system, px, ring_list_sync_system, scroll_view, scroll_view_system,
  slider, slider_drag_system, slider_visual_system, splitter, tab_view, tab_view_system, table,
  text_input, text_input_keyboard_system, text_input_pointer_system, text_input_visual_system,
  toggle_switch, toggle_switch_state_system, tooltip_system,
};
