use std::collections::BTreeMap;
use std::path::PathBuf;

use bevy::prelude::*;
use serde::{Deserialize, Serialize};

use gate_ui::{DebugMenu, MenuValue, WindowState};

use crate::camera::{CameraMode, CameraPose, FlyCamera};

pub const CONFIG_PATH: &str = "config.toml";

const MENU_UNITS_VERSION: u32 = 4;

pub fn name() -> String {
  match crate::consts::bench_config_tag() {
    Some(tag) => format!("config.{tag}.toml"),
    None => CONFIG_PATH.to_owned(),
  }
}

pub fn path() -> PathBuf {
  gate_render::data_dir().join(name())
}

fn base_path() -> PathBuf {
  gate_render::data_dir().join(CONFIG_PATH)
}

fn write_text(path: &std::path::Path, src: &str) -> bool {
  if let Some(dir) = path.parent()
    && let Err(e) = std::fs::create_dir_all(dir)
  {
    warn!("config 目录创建失败 {e} → 未写入");
    return false;
  }
  match std::fs::write(path, src) {
    Ok(()) => true,
    Err(e) => {
      warn!("config 写入失败 {e}");
      false
    }
  }
}

#[derive(Resource, Serialize, Deserialize, Clone, Debug)]
pub struct Config {
  #[serde(default)]
  pub menu_units: u32,
  #[serde(default)]
  pub window: WindowState,
  #[serde(default)]
  pub menu: BTreeMap<String, MenuValue>,
  #[serde(default)]
  pub camera: Option<CameraPose>,
}

impl Default for Config {
  fn default() -> Self {
    Self {
      menu_units: MENU_UNITS_VERSION,
      window: WindowState::default(),
      menu: BTreeMap::new(),
      camera: None,
    }
  }
}

impl Config {
  pub fn load() -> Self {
    let path = path();
    let src = match std::fs::read_to_string(&path) {
      Ok(s) => s,
      Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
        let name = name();
        let Some(seed) =
          (name != CONFIG_PATH).then(|| std::fs::read_to_string(base_path()).ok()).flatten()
        else {
          info!("config 无 {} → 缺省", path.display());
          return Self::default();
        };
        if write_text(&path, &seed) {
          info!("config[{}] 无 → 复制 {} 建文件", name, CONFIG_PATH);
        }
        seed
      }
      Err(e) => {
        warn!("config 读取失败 {e} → 缺省");
        return Self::default();
      }
    };
    match toml::from_str::<Self>(&src) {
      Ok(mut c) => {
        if c.migrate_menu_units() {
          c.save();
        }
        info!("config ← {}（menu_units={}）", path.display(), c.menu_units);
        c
      }
      Err(e) => {
        warn!("config 解析失败 {e} → 缺省");
        Self::default()
      }
    }
  }

  fn migrate_menu_units(&mut self) -> bool {
    let old = self.menu_units;
    if old >= MENU_UNITS_VERSION {
      return false;
    }
    let vpm = self.migration_voxel_per_meter();
    let reference_vpm = gate_voxel::DEFAULT_VOXEL_PER_METER;
    if old < 2 {
      if let Some(MenuValue::Number(v)) = self.menu.get_mut("render/sky/air/start") {
        *v = ((*v / vpm as f64) * 100.0).round() / 100.0;
      }
      if let Some(MenuValue::Number(v)) = self.menu.get_mut("render/sky/air/density") {
        *v = ((*v * vpm as f64 / reference_vpm as f64) * 100.0).round() / 100.0;
      }
    }
    if old == 2 {
      for (path, decimals) in
        [("game/edit/place/size", 0usize), ("game/edit/place/offset", 1), ("player/camera/pos", 2)]
      {
        if let Some(MenuValue::Text(t)) = self.menu.get_mut(path)
          && let Ok(v) = t.trim().parse::<f32>()
        {
          *t = format!("{:.*}", decimals, v * reference_vpm);
        }
      }
      if let Some(MenuValue::Number(v)) = self.menu.get_mut("render/sky/air/density") {
        *v = ((*v * vpm as f64 / reference_vpm as f64) * 100.0).round() / 100.0;
      }
    }
    if old == 3 {
      for (path, decimals) in
        [("game/edit/place/size", 0usize), ("game/edit/place/offset", 1), ("player/camera/pos", 2)]
      {
        if let Some(MenuValue::Text(t)) = self.menu.get_mut(path)
          && let Ok(v) = t.trim().parse::<f32>()
        {
          *t = format!("{:.*}", decimals, v * vpm);
        }
      }
      let chunk_scale = vpm as f64 / gate_voxel::CHUNK_SIZE as f64;
      for path in [
        "game/world/load_radius",
        "game/world/unload_radius",
        "game/world/coarse_radius",
        "game/world/coarse_height",
      ] {
        if let Some(MenuValue::Number(v)) = self.menu.get_mut(path) {
          *v = (*v * chunk_scale).round().max(0.0);
        }
      }
    }
    self.menu_units = MENU_UNITS_VERSION;
    info!(
      "config 菜单单位 → v{old} → v{MENU_UNITS_VERSION}（物理/光照=米，编辑/坐标=voxel，流式=chunk；按 {vpm} vx/m 换算）"
    );
    true
  }

  fn migration_voxel_per_meter(&self) -> f32 {
    let name = match self.menu.get("game/world/model") {
      Some(MenuValue::Text(name)) => name.as_str(),
      _ => "nuke",
    };
    if name == crate::mc::GREEN_FIELD {
      crate::mc::VOXELS_PER_BLOCK as f32
    } else {
      crate::vox_scene::voxel_per_meter(name)
    }
  }

  pub fn save(&self) {
    let path = path();
    let Ok(src) = toml::to_string_pretty(self) else {
      warn!("config 序列化失败 → 未保存");
      return;
    };
    if write_text(&path, &src) {
      debug!("config → {}", path.display());
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
  snapshot(&config, &q_menu, *mode, &orbit, &fly).save();
}

fn snapshot(
  config: &Config,
  q_menu: &Query<&DebugMenu>,
  mode: CameraMode,
  orbit: &gate_render::OrbitCamera,
  fly: &FlyCamera,
) -> Config {
  let mut out = config.clone();
  out.camera = Some(CameraPose::capture(mode, orbit, fly));
  if let Ok(menu) = q_menu.single() {
    let mut window = menu.model.window.clone();
    window.path = menu.path.clone();
    window.collapsed = menu.collapsed;
    out.window = window;
    out.menu = menu.model.values();
  }
  out
}
