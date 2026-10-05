use std::collections::BTreeMap;
use std::path::PathBuf;

use bevy::prelude::*;
use serde::{Deserialize, Serialize};

use gate_ui::{DebugMenu, MenuValue, WindowState};

use crate::camera::{CameraMode, CameraPose, FlyCamera};

pub const CONFIG_PATH: &str = "config.toml";

pub fn path() -> PathBuf {
  gate_render::data_dir().join(CONFIG_PATH)
}

#[derive(Resource, Serialize, Deserialize, Clone, Debug, Default)]
#[serde(default)]
pub struct Config {
  pub window: WindowState,
  pub menu: BTreeMap<String, MenuValue>,
  pub camera: Option<CameraPose>,
}

impl Config {
  pub fn load() -> Self {
    let path = path();
    let src = match std::fs::read_to_string(&path) {
      Ok(s) => s,
      Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
        info!("config 无 {} → 缺省", path.display());
        return Self::default();
      }
      Err(e) => {
        warn!("config 读取失败 {e} → 缺省");
        return Self::default();
      }
    };
    match toml::from_str::<Self>(&src) {
      Ok(c) => {
        debug!("config ← {}", path.display());
        c
      }
      Err(e) => {
        warn!("config 解析失败 {e} → 缺省");
        Self::default()
      }
    }
  }

  pub fn save(&self) {
    let path = path();
    let Ok(src) = toml::to_string_pretty(self) else {
      warn!("config 序列化失败 → 未保存");
      return;
    };
    if let Some(dir) = path.parent()
      && let Err(e) = std::fs::create_dir_all(dir)
    {
      warn!("config 目录创建失败 {e} → 未保存");
      return;
    }
    match std::fs::write(&path, src) {
      Ok(()) => debug!("config → {}", path.display()),
      Err(e) => warn!("config 写入失败 {e}"),
    }
  }
}

pub(crate) fn save_config_on_exit(
  mut exit: MessageReader<AppExit>,
  config: Res<Config>,
  q_menu: Query<&DebugMenu>,
  mode: Res<CameraMode>,
  orbit: Res<gate_render::OrbitCamera>,
  fly: Res<FlyCamera>,
  mut saved: Local<bool>,
) {
  if *saved || exit.read().count() == 0 {
    return;
  }
  *saved = true;
  let mut out = config.clone();
  out.camera = Some(CameraPose::capture(*mode, &orbit, &fly));
  if let Ok(menu) = q_menu.single() {
    let mut window = menu.model.window.clone();
    window.path = menu.path.clone();
    window.collapsed = menu.collapsed;
    out.window = window;
    out.menu = menu.model.values();
  }
  out.save();
}
