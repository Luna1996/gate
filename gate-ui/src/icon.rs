//! 图标：FontAwesome 字形码点 + 主题图标字体接线。
//! 码点常量来自 `font-awesome` crate（FA 5 Free Solid），字体文件由 `UiTheme::icon_font_path` 指定。
//! 仓库内字体 `assets/fonts/fa-solid-900.ttf`；码点须与字体版本一致。

use bevy::asset::LoadState;
use bevy::log::{debug, warn};
use bevy::prelude::*;
use bevy::text::FontSource;
use font_awesome::strs;

use crate::theme::UiTheme;

/// 图标集：只登记本仓库用到的字形（新增图标在此加一项，值直接取 crate 常量）。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Icon {
  /// 标题栏返回（上一级菜单）
  ChevronLeft,
  /// 子菜单项右侧「进入下级」指示
  ChevronRight,
  /// 标题栏重置位置
  UndoAlt,
  /// 标题栏收起（内容向上收拢）
  AngleUp,
  /// 标题栏展开（内容向下展开）
  AngleDown,
  /// 下拉框右侧「展开列表」指示
  ChevronDown,
}

impl Icon {
  /// 码点（单字符字符串，直接进 `Text`）
  pub fn glyph(self) -> &'static str {
    match self {
      Self::ChevronLeft => strs::CHEVRON_LEFT,
      Self::ChevronRight => strs::CHEVRON_RIGHT,
      Self::UndoAlt => strs::UNDO_ALT,
      Self::AngleUp => strs::ANGLE_UP,
      Self::AngleDown => strs::ANGLE_DOWN,
      Self::ChevronDown => strs::CHEVRON_DOWN,
    }
  }
}

/// 图标字体资源（`theme.icon_font_path` 的加载结果；未配置 → 图标回退主题正文字体）
#[derive(Resource, Default)]
pub struct IconFont {
  pub path: Option<String>,
  pub handle: Option<Handle<Font>>,
}

impl IconFont {
  /// 图标文本的字体来源：已加载 → 图标字体；否则 → 主题正文字体（字形大概率缺失，仅兜底）
  pub fn font_source(&self) -> FontSource {
    match self.handle.as_ref() {
      Some(h) => FontSource::from(h),
      None => FontSource::default(),
    }
  }
}

/// 图标字体资产加载（`theme.icon_font_path` 变化时重新加载）
pub(crate) fn icon_font_load(
  theme: Option<Res<UiTheme>>,
  server: Option<Res<AssetServer>>,
  mut font: ResMut<IconFont>,
) {
  let (Some(theme), Some(server)) = (theme, server) else {
    return;
  };
  if !theme.is_changed() {
    return;
  }
  let Some(path) = theme.icon_font_path.as_ref() else {
    return;
  };
  if font.path.as_ref() == Some(path) {
    return;
  }
  font.path = Some(path.clone());
  font.handle = Some(server.load::<Font>(path));
  debug!("icon font → {path}");
}

/// 图标字体加载失败告警（一次）；成功则静默
pub(crate) fn icon_font_report(
  server: Option<Res<AssetServer>>,
  font: Res<IconFont>,
  mut reported: Local<bool>,
) {
  if *reported {
    return;
  }
  let (Some(server), Some(handle)) = (server, font.handle.as_ref()) else {
    return;
  };
  if let LoadState::Failed(e) = server.load_state(handle.id()) {
    *reported = true;
    warn!("icon font {:?} load failed ({e}) → icons not rendered", font.path);
  }
}
