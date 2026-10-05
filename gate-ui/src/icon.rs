use bevy::asset::LoadState;
use bevy::log::{debug, warn};
use bevy::prelude::*;
use bevy::text::FontSource;
use font_awesome::strs;

use crate::theme::UiTheme;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Icon {
  ChevronLeft,
  ChevronRight,
  UndoAlt,
  AngleUp,
  AngleDown,
  ChevronDown,
}

impl Icon {
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

#[derive(Resource, Default)]
pub struct IconFont {
  pub path: Option<String>,
  pub handle: Option<Handle<Font>>,
}

impl IconFont {
  pub fn font_source(&self) -> FontSource {
    match self.handle.as_ref() {
      Some(h) => FontSource::from(h),
      None => FontSource::default(),
    }
  }
}

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
