use bevy::app::{Plugin, PreStartup, Update};
use bevy::asset::io::Reader;
use bevy::asset::{
  Asset, AssetApp, AssetEvent, AssetId, AssetLoader, Assets, Handle, LoadContext, LoadState,
};
use bevy::log::{debug, warn};
use bevy::prelude::*;
use bevy::ui::UiScale;
use bevy::window::{PrimaryWindow, Window};
use serde::Deserialize;

use crate::consts::AUTOFIT_BASE_HEIGHT;

pub const THEME_ASSET_PATH: &str = "ui/theme.ron";

#[derive(Clone, Debug, PartialEq)]
pub struct HexColor(pub String);

pub fn parse_hex_color(s: &str) -> Option<[u8; 4]> {
  let s = s.trim();
  let body = s.strip_prefix('#').unwrap_or(s);
  let (rgb, a) = match body.len() {
    6 => (body, 255u8),
    8 => (&body[..6], u8::from_str_radix(&body[6..8], 16).ok()?),
    _ => return None,
  };
  if !rgb.bytes().all(|b| b.is_ascii_hexdigit()) {
    return None;
  }
  let r = u8::from_str_radix(&rgb[0..2], 16).ok()?;
  let g = u8::from_str_radix(&rgb[2..4], 16).ok()?;
  let b = u8::from_str_radix(&rgb[4..6], 16).ok()?;
  Some([r, g, b, a])
}

impl HexColor {
  pub fn to_color(&self) -> Option<Color> {
    parse_hex_color(&self.0).map(|[r, g, b, a]| Color::srgba_u8(r, g, b, a))
  }
}

impl<'de> Deserialize<'de> for HexColor {
  fn deserialize<D>(d: D) -> Result<Self, D::Error>
  where
    D: serde::Deserializer<'de>,
  {
    let s = String::deserialize(d)?;
    if parse_hex_color(&s).is_some() {
      Ok(HexColor(s))
    } else {
      Err(serde::de::Error::custom(format!("invalid hex color {s:?} (expect RRGGBB or RRGGBBAA)")))
    }
  }
}

#[derive(Deserialize, Clone, Debug, PartialEq)]
#[serde(default)]
pub struct ThemeColors {
  pub surface_base: HexColor,
  pub surface_card: HexColor,
  pub surface_elevated: HexColor,
  pub surface_overlay: HexColor,
  pub surface_top: HexColor,
  pub surface_card_hud: HexColor,
  pub scrim: HexColor,
  pub border_subtle: HexColor,
  pub border: HexColor,
  pub border_strong: HexColor,
  pub text_primary: HexColor,
  pub text_body: HexColor,
  pub text_muted: HexColor,
  pub text_faint: HexColor,
  pub accent_fill: HexColor,
  pub accent_fill_hover: HexColor,
  pub accent_fill_pressed: HexColor,
  pub accent_text: HexColor,
  pub success: HexColor,
  pub success_fill: HexColor,
  pub warning: HexColor,
  pub warning_fill: HexColor,
  pub danger: HexColor,
  pub danger_fill: HexColor,
}

impl Default for ThemeColors {
  fn default() -> Self {
    Self {
      surface_base: HexColor("09090B".into()),
      surface_card: HexColor("131316".into()),
      surface_elevated: HexColor("1A1A20".into()),
      surface_overlay: HexColor("222228".into()),
      surface_top: HexColor("2A2A31".into()),
      surface_card_hud: HexColor("131316".into()),
      scrim: HexColor("09090B".into()),
      border_subtle: HexColor("232329".into()),
      border: HexColor("2A2A31".into()),
      border_strong: HexColor("3F3F46".into()),
      text_primary: HexColor("FAFAFA".into()),
      text_body: HexColor("D4D4D8".into()),
      text_muted: HexColor("A1A1AA".into()),
      text_faint: HexColor("71717A".into()),
      accent_fill: HexColor("2A2A31".into()),
      accent_fill_hover: HexColor("3F3F46".into()),
      accent_fill_pressed: HexColor("1A1A20".into()),
      accent_text: HexColor("FAFAFA".into()),
      success: HexColor("D4D4D8".into()),
      success_fill: HexColor("1A1A20".into()),
      warning: HexColor("A1A1AA".into()),
      warning_fill: HexColor("131316".into()),
      danger: HexColor("FAFAFA".into()),
      danger_fill: HexColor("3F3F46".into()),
    }
  }
}

#[derive(Deserialize, Clone, Debug, PartialEq)]
#[serde(default)]
pub struct ThemeMetrics {
  pub corner_radius: f32,
  pub corner_radius_sm: f32,
  pub border_width: f32,
  pub spacing: Spacing,
  pub font_size: FontSize,
}

impl Default for ThemeMetrics {
  fn default() -> Self {
    Self {
      corner_radius: 0.0,
      corner_radius_sm: 0.0,
      border_width: 1.0,
      spacing: Spacing::default(),
      font_size: FontSize::default(),
    }
  }
}

#[derive(Deserialize, Clone, Debug, PartialEq)]
#[serde(default)]
pub struct Spacing {
  pub xs: f32,
  pub sm: f32,
  pub md: f32,
  pub lg: f32,
}

impl Default for Spacing {
  fn default() -> Self {
    Self { xs: 4.0, sm: 8.0, md: 12.0, lg: 16.0 }
  }
}

#[derive(Deserialize, Clone, Debug, PartialEq)]
#[serde(default)]
pub struct FontSize {
  pub sm: f32,
  pub md: f32,
  pub lg: f32,
}

impl Default for FontSize {
  fn default() -> Self {
    Self { sm: 12.0, md: 14.0, lg: 18.0 }
  }
}

#[derive(Asset, Resource, TypePath, Deserialize, Clone, Debug, PartialEq)]
#[serde(default)]
pub struct UiTheme {
  pub colors: ThemeColors,
  pub metrics: ThemeMetrics,
  pub ui_scale: f32,
  pub auto_fit_ui_scale: bool,
  pub font_path: Option<String>,
  pub icon_font_path: Option<String>,
}

impl Default for UiTheme {
  fn default() -> Self {
    Self {
      colors: ThemeColors::default(),
      metrics: ThemeMetrics::default(),
      ui_scale: 1.0,
      auto_fit_ui_scale: false,
      font_path: None,
      icon_font_path: None,
    }
  }
}

pub fn default_theme() -> UiTheme {
  UiTheme::default()
}

pub fn parse_theme_ron(src: &str) -> Result<UiTheme, ron::error::SpannedError> {
  ron::de::from_str(src)
}

pub fn autofit_ui_scale(ui_scale: f32, auto_fit: bool, window_height: f32) -> f32 {
  if auto_fit {
    let h = window_height.min(4096.0);
    ui_scale * (h / AUTOFIT_BASE_HEIGHT).clamp(0.25, 4.0)
  } else {
    ui_scale
  }
}

#[derive(Debug)]
pub enum RonLoadError {
  Io(std::io::Error),
  Ron(ron::error::SpannedError),
}

impl std::fmt::Display for RonLoadError {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      RonLoadError::Io(e) => write!(f, "io error: {e}"),
      RonLoadError::Ron(e) => write!(f, "ron parse error: {e}"),
    }
  }
}

impl std::error::Error for RonLoadError {
  fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
    match self {
      RonLoadError::Io(e) => Some(e),
      RonLoadError::Ron(e) => Some(e),
    }
  }
}

impl From<std::io::Error> for RonLoadError {
  fn from(e: std::io::Error) -> Self {
    Self::Io(e)
  }
}

impl From<ron::error::SpannedError> for RonLoadError {
  fn from(e: ron::error::SpannedError) -> Self {
    Self::Ron(e)
  }
}

#[derive(TypePath)]
pub struct RonThemeLoader;

impl AssetLoader for RonThemeLoader {
  type Asset = UiTheme;
  type Settings = ();
  type Error = RonLoadError;

  async fn load(
    &self,
    reader: &mut dyn Reader,
    _settings: &(),
    _load_context: &mut LoadContext<'_>,
  ) -> Result<UiTheme, Self::Error> {
    let mut bytes = Vec::new();
    reader.read_to_end(&mut bytes).await?;
    ron::de::from_bytes(&bytes).map_err(RonLoadError::Ron)
  }

  fn extensions(&self) -> &[&str] {
    &["ron"]
  }
}

#[derive(Resource)]
pub struct UiThemeHandle(pub Handle<UiTheme>);

#[derive(Resource, Default)]
pub struct UiThemeState {
  pub applied: bool,
  pub fell_back: bool,
}

fn ui_theme_request(mut commands: Commands, server: Option<Res<AssetServer>>) {
  let Some(server) = server else { return };
  let handle: Handle<UiTheme> = server.load(THEME_ASSET_PATH);
  commands.insert_resource(UiThemeHandle(handle));
}

fn ui_theme_resolve(
  mut evr: MessageReader<AssetEvent<UiTheme>>,
  assets: Option<Res<Assets<UiTheme>>>,
  handle: Option<Res<UiThemeHandle>>,
  server: Option<Res<AssetServer>>,
  mut state: ResMut<UiThemeState>,
  mut commands: Commands,
) {
  let (Some(assets), Some(handle), Some(server)) = (assets, handle, server) else {
    return;
  };
  let hid = handle.0.id();
  for ev in evr.read() {
    if let AssetEvent::LoadedWithDependencies { id } = ev
      && id == &hid
      && !state.applied
      && let Some(theme) = assets.get(*id)
    {
      commands.insert_resource(theme.clone());
      state.applied = true;
      debug!("theme → {THEME_ASSET_PATH}");
    }
  }
  if !state.applied
    && !state.fell_back
    && let LoadState::Failed(err) = server.load_state(hid)
  {
    commands.insert_resource(default_theme());
    state.fell_back = true;
    warn!("theme load failed ({err}) → built-in dark default");
  }
}

fn ui_scale_autofit(
  theme: Option<Res<UiTheme>>,
  windows: Query<&Window, With<PrimaryWindow>>,
  mut ui_scale: ResMut<UiScale>,
) {
  let Some(theme) = theme else { return };
  let Ok(w) = windows.single() else { return };
  let target = autofit_ui_scale(theme.ui_scale, theme.auto_fit_ui_scale, w.height());
  if (ui_scale.0 - target).abs() > f32::EPSILON {
    ui_scale.0 = target;
    debug!("ui_scale → {target:.3} [window h {}]", w.height());
  }
}

#[derive(Resource, Default)]
pub struct ThemeFont {
  pub path: Option<String>,
  pub handle: Option<Handle<Font>>,
}

fn ui_theme_font_load(
  theme: Option<Res<UiTheme>>,
  server: Option<Res<AssetServer>>,
  mut font: ResMut<ThemeFont>,
) {
  let (Some(theme), Some(server)) = (theme, server) else {
    return;
  };
  if !theme.is_changed() {
    return;
  }
  let Some(path) = theme.font_path.as_ref() else {
    return;
  };
  if font.path.as_ref() == Some(path) {
    return;
  }
  font.path = Some(path.clone());
  font.handle = Some(server.load::<Font>(path));
  debug!("theme font → {path}");
}

fn ui_theme_font_install_default(
  server: Option<Res<AssetServer>>,
  font: Res<ThemeFont>,
  mut fonts: ResMut<Assets<Font>>,
  mut done: Local<bool>,
) {
  if *done {
    return;
  }
  let Some(handle) = font.handle.clone() else {
    return;
  };
  let Some(server) = server else {
    return;
  };
  match server.load_state(handle.id()) {
    LoadState::Loaded => {}
    LoadState::Failed(_) => {
      warn!("theme font {:?} load failed → default slot not overridden (CJK boxes)", font.path);
      *done = true;
      return;
    }
    _ => return,
  }
  let Some(font_asset) = fonts.get(&handle).cloned() else {
    return;
  };
  match fonts.insert(AssetId::<Font>::default(), font_asset) {
    Ok(()) => {
      debug!("default font → {:?}", font.path);
      *done = true;
    }
    Err(e) => warn!("default font override insert failed ({e:?}) → retry next frame"),
  }
}

#[derive(Default)]
pub struct GateUiPlugin;

impl Plugin for GateUiPlugin {
  fn build(&self, app: &mut App) {
    app
      .init_asset::<UiTheme>()
      .init_resource::<UiScale>()
      .init_resource::<UiThemeState>()
      .init_resource::<ThemeFont>()
      .init_resource::<crate::icon::IconFont>()
      .init_resource::<crate::capture::UiPointerCaptured>()
      .init_resource::<crate::capture::MouseIntercepted>()
      .init_resource::<crate::capture::UiShiftCaptured>()
      .init_resource::<crate::widgets::TextInputFocus>()
      .init_resource::<crate::widgets::TooltipLayerEntity>()
      .init_resource::<crate::i18n::UiTranslator>()
      .init_resource::<crate::world_anchor::AnchorCamera>()
      .register_asset_loader(RonThemeLoader)
      .insert_resource(bevy::ui::picking_backend::UiPickingSettings { require_markers: true })
      .add_observer(crate::pointer::ui_pointer_press)
      .add_observer(crate::pointer::ui_pointer_release)
      .add_observer(crate::pointer::ui_pointer_drag_end)
      .add_observer(crate::pointer::ui_pointer_cancel)
      .add_systems(PreStartup, ui_theme_request)
      .add_systems(
        Update,
        (
          (
            ui_theme_resolve,
            ui_theme_font_load,
            ui_theme_font_install_default.after(ui_theme_font_load),
            crate::icon::icon_font_load.after(ui_theme_resolve),
            crate::icon::icon_font_report.after(crate::icon::icon_font_load),
            ui_scale_autofit,
          ),
          (
            crate::widgets::button_state_system,
            crate::widgets::checkbox_state_system,
            crate::widgets::toggle_switch_state_system,
            crate::widgets::slider_drag_system,
            crate::widgets::slider_visual_system,
            crate::widgets::ring_list_sync_system,
            crate::widgets::plot_redraw_system,
            crate::widgets::scroll_view_system,
            crate::widgets::tab_view_system,
            (crate::widgets::dropdown_system, crate::widgets::dropdown_visual_system).chain(),
          ),
          (
            crate::widgets::label_ellipsis_system,
            crate::widgets::text_input_pointer_system,
            crate::widgets::text_input_keyboard_system,
            crate::widgets::text_input_visual_system,
            crate::widgets::tooltip_system,
          ),
          (
            (crate::menu::color_picker_system, crate::menu::color_picker_visual_system)
              .chain()
              .before(crate::menu::menu_system),
            (
              crate::i18n::i18n_refresh_system,
              crate::capture::ui_pointer_capture_system,
              crate::menu::menu_system,
            ),
          ),
          (
            crate::world_anchor::world_anchor_apply_text.after(ui_theme_font_install_default),
            crate::world_anchor::world_anchor_system,
          ),
        ),
      );
  }
}
