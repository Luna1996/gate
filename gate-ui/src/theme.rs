//! 主题令牌：`UiTheme` RON 数据资产 + 内置暗色默认 + 加载/回退接线。
//!
//! 改 RON 换肤，不改代码；加载失败/缺失 → 内置暗色默认 + `warn!`（不 panic，不静默）。

use bevy::app::{Plugin, PreStartup, Update};
use bevy::asset::io::Reader;
use bevy::asset::{
  Asset, AssetApp, AssetEvent, AssetId, AssetLoader, Assets, Handle, LoadContext, LoadState,
};
use bevy::log::{info, warn};
use bevy::prelude::*;
use bevy::ui::UiScale;
use bevy::window::{PrimaryWindow, Window};
use serde::Deserialize;

/// 主题资产路径（相对资源根 `assets/`，见 gate-render `paths`）
pub const THEME_ASSET_PATH: &str = "ui/theme.ron";
/// auto_fit 基准高度：窗口高 720 时 UiScale == ui_scale
pub const AUTOFIT_BASE_HEIGHT: f32 = 720.0;

// ---------- 颜色令牌 ----------

/// 十六进制颜色："RRGGBB"（alpha=FF）或 "RRGGBBAA"
#[derive(Clone, Debug, PartialEq)]
pub struct HexColor(pub String);

/// 解析 hex → `[r, g, b, a]`；非法输入返回 None
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
  /// 转 bevy Color（sRGB）；非法 hex → None（仅运行时构造的 HexColor 可能非法）
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

// ---------- 主题结构 ----------

/// 颜色组（docs/ui-dark-theme.md §2）。全部带 alpha。
///
/// 暗色令牌：5 层表面（亮度即海拔）+ HUD 变体 + 3 档边框 + 4 档文字 + 全灰度强调/语义色。
#[derive(Deserialize, Clone, Debug, PartialEq)]
#[serde(default)]
pub struct ThemeColors {
  // ---- 表面 5 层（L0 基底 → L4 顶层，每层亮约 3-4%）----
  pub surface_base: HexColor,
  pub surface_card: HexColor,
  pub surface_elevated: HexColor,
  pub surface_overlay: HexColor,
  pub surface_top: HexColor,
  /// HUD 半透明卡片（浮在 3D 场景上，alpha 下限 90%）
  pub surface_card_hud: HexColor,
  /// 模态遮罩
  pub scrim: HexColor,
  // ---- 边框 3 档（1px）----
  pub border_subtle: HexColor,
  pub border: HexColor,
  pub border_strong: HexColor,
  // ---- 文字 4 档（禁纯白）----
  pub text_primary: HexColor,
  pub text_body: HexColor,
  pub text_muted: HexColor,
  /// 仅限 ≥18px 大号弱化标签
  pub text_faint: HexColor,
  // ---- 强调色（全灰度：主操作靠亮度差区分，无任何彩色）----
  /// 主按钮填充（最亮表面，靠亮度突出主操作）
  pub accent_fill: HexColor,
  /// 主按钮 hover 填充（提亮一档）
  pub accent_fill_hover: HexColor,
  /// 主按钮按压填充（压暗）
  pub accent_fill_pressed: HexColor,
  /// 强调文字/图标/折线（主文本色，最高对比）
  pub accent_text: HexColor,
  // ---- 语义色（全灰度：仅靠亮度差区分，无彩色）----
  /// success 文字 = 正文灰
  pub success: HexColor,
  /// success 填充 = 抬升表面
  pub success_fill: HexColor,
  /// warning 文字 = 说明灰
  pub warning: HexColor,
  /// warning 填充 = 卡片表面
  pub warning_fill: HexColor,
  /// danger 文字 = 主文本色
  pub danger: HexColor,
  /// danger 填充 = 强边框灰（最深灰表示危险态）
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
      // HUD 面板不透明（与卡片同色，靠边框与场景分隔）
      surface_card_hud: HexColor("131316".into()),
      scrim: HexColor("09090B".into()),
      border_subtle: HexColor("232329".into()),
      border: HexColor("2A2A31".into()),
      border_strong: HexColor("3F3F46".into()),
      text_primary: HexColor("FAFAFA".into()),
      text_body: HexColor("D4D4D8".into()),
      text_muted: HexColor("A1A1AA".into()),
      text_faint: HexColor("71717A".into()),
      // 强调全灰度：主操作靠最亮表面突出
      accent_fill: HexColor("2A2A31".into()),
      accent_fill_hover: HexColor("3F3F46".into()),
      accent_fill_pressed: HexColor("1A1A20".into()),
      accent_text: HexColor("FAFAFA".into()),
      // 语义全灰度
      success: HexColor("D4D4D8".into()),
      success_fill: HexColor("1A1A20".into()),
      warning: HexColor("A1A1AA".into()),
      warning_fill: HexColor("131316".into()),
      danger: HexColor("FAFAFA".into()),
      danger_fill: HexColor("3F3F46".into()),
    }
  }
}

/// 度量组（docs/ui-dark-theme.md §3）
#[derive(Deserialize, Clone, Debug, PartialEq)]
#[serde(default)]
pub struct ThemeMetrics {
  /// 容器圆角：面板、按钮、弹窗（rounded-lg）
  pub corner_radius: f32,
  /// 小控件圆角：checkbox、slider thumb、标签 chip
  pub corner_radius_sm: f32,
  pub border_width: f32,
  pub spacing: Spacing,
  pub font_size: FontSize,
}

impl Default for ThemeMetrics {
  fn default() -> Self {
    Self {
      // 无圆角：硬边界面，靠 1px 边框与亮度差分层级
      corner_radius: 0.0,
      corner_radius_sm: 0.0,
      border_width: 1.0,
      spacing: Spacing::default(),
      font_size: FontSize::default(),
    }
  }
}

/// 间距令牌 xs/sm/md/lg
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

/// 字号令牌 sm/md/lg
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

/// 主题令牌资产（RON）。字段缺失回退内置默认；整份解析失败整体回退。
///
/// 运行时应用后以 Resource 形式存在（`Res<UiTheme>`），同时是 Asset 供 AssetServer 管理。
#[derive(Asset, Resource, TypePath, Deserialize, Clone, Debug, PartialEq)]
#[serde(default)]
pub struct UiTheme {
  pub colors: ThemeColors,
  pub metrics: ThemeMetrics,
  /// 全局 UI 缩放，接到 Bevy `UiScale` 资源
  pub ui_scale: f32,
  /// true 时按窗口高度/AUTOFIT_BASE_HEIGHT 推导总缩放
  pub auto_fit_ui_scale: bool,
  /// 字体资产路径；None → bevy default_font
  pub font_path: Option<String>,
  /// 图标字体（FontAwesome Free Solid）资产路径；None → 图标回退正文字体
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

/// 内置暗色默认主题（Modern UI 参照）
pub fn default_theme() -> UiTheme {
  UiTheme::default()
}

/// RON 解析纯函数（system 接线只做加载与回退）
pub fn parse_theme_ron(src: &str) -> Result<UiTheme, ron::error::SpannedError> {
  ron::de::from_str(src)
}

/// UiScale 推导：auto_fit 时按窗口高度缩放。
/// 窗口高上限钳 4096（平台可能上报退化矩形），缩放系数再钳 [0.25, 4.0] 防极端。
pub fn autofit_ui_scale(ui_scale: f32, auto_fit: bool, window_height: f32) -> f32 {
  if auto_fit {
    let h = window_height.min(4096.0);
    ui_scale * (h / AUTOFIT_BASE_HEIGHT).clamp(0.25, 4.0)
  } else {
    ui_scale
  }
}

// ---------- RON 资产加载器（bevy 0.19 无内置 RonAssetPlugin，自写） ----------

/// RON 加载错误（IO 读取 + RON 解析）
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

/// UiTheme 的 RON 资产加载器（bevy 0.19 无内置 RonAssetPlugin；具体类型避免泛型 TypePath 边界）
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

// ---------- Bevy 接线 ----------

/// 主题资产句柄（PreStartup 请求加载后插入）
#[derive(Resource)]
pub struct UiThemeHandle(pub Handle<UiTheme>);

/// 主题应用状态（防重复日志/重复插入）
#[derive(Resource, Default)]
pub struct UiThemeState {
  pub applied: bool,
  pub fell_back: bool,
}

fn ui_theme_request(mut commands: Commands, server: Option<Res<AssetServer>>) {
  // headless 测试环境无 AssetServer 时跳过
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
  // 成功路径：LoadedWithDependencies → 插入运行时主题
  for ev in evr.read() {
    if let AssetEvent::LoadedWithDependencies { id } = ev
      && id == &hid
      && !state.applied
      && let Some(theme) = assets.get(*id)
    {
      commands.insert_resource(theme.clone());
      state.applied = true;
      info!("ui theme loaded from {THEME_ASSET_PATH}");
    }
  }
  // 失败路径：load_state Failed（文件缺失 / RON 解析失败 / hex 非法都汇于此）
  if !state.applied
    && !state.fell_back
    && let LoadState::Failed(err) = server.load_state(hid)
  {
    commands.insert_resource(default_theme());
    state.fell_back = true;
    warn!("ui theme load failed ({err}), falling back to built-in dark default");
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
    info!("UiScale -> {target:.3} (window h = {})", w.height());
  }
}

/// 主题字体资源（theme.font_path 加载结果；None → widget 层用 FontSource::SystemUi）
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
  info!("ui theme font loading: {path}");
}

/// 主题字体资产加载成功后，clone 到 `AssetId::<Font>::default()` 的 storage slot。
///
/// 使所有不显式指定字体的文本（`FontSource::default()` / `TextFont::default()` /
/// `FontSource::Handle(Handle::default())`）都用主题字体，而非 Bevy 内置 FiraMono-subset
/// （该字体不含 CJK 字形 → 中文方框）。只执行一次；headless 无 AssetServer 时跳过。
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
      warn!(
        "theme font {:?} failed to load; default font slot not overridden (CJK will be boxes)",
        font.path
      );
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
      info!(
        "default font overridden with theme font {:?} — all implicit TextFont now use CJK font",
        font.path
      );
      *done = true;
    }
    Err(e) => warn!("default font override insert failed ({e:?}); retrying next frame"),
  }
}

/// gate-ui 插件：主题资产注册 + UiScale 联动 + 各 widget 状态机注册。
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
      .init_resource::<crate::widgets::TextInputFocus>()
      .init_resource::<crate::widgets::TooltipLayerEntity>()
      .init_resource::<crate::i18n::UiTranslator>()
      .init_resource::<crate::world_anchor::AnchorCamera>()
      .register_asset_loader(RonThemeLoader)
      .add_systems(PreStartup, ui_theme_request)
      .add_systems(
        Update,
        (
          // 主题/字体/缩放接线
          (
            ui_theme_resolve,
            ui_theme_font_load,
            ui_theme_font_install_default.after(ui_theme_font_load),
            crate::icon::icon_font_load.after(ui_theme_resolve),
            crate::icon::icon_font_report.after(crate::icon::icon_font_load),
            ui_scale_autofit,
          ),
          // 各 widget 状态机
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
            // 下拉框：交互（开/选/关）在前，视觉（跟随锚点 + 展开动画）在后
            (crate::widgets::dropdown_system, crate::widgets::dropdown_visual_system).chain(),
          ),
          // 文本相关（省略号/输入框）+ 悬浮提示
          (
            crate::widgets::label_ellipsis_system,
            crate::widgets::text_input_pointer_system,
            crate::widgets::text_input_keyboard_system,
            crate::widgets::text_input_visual_system,
            crate::widgets::tooltip_system,
          ),
          // 语言切换后重解析 keyed 文本 + 输入门控 + 菜单容器/分页动画
          (
            crate::i18n::i18n_refresh_system,
            crate::capture::ui_pointer_capture_system,
            crate::menu::menu_system,
          ),
          (
            crate::world_anchor::world_anchor_apply_text.after(ui_theme_font_install_default),
            crate::world_anchor::world_anchor_system,
          ),
        ),
      );
  }
}
