//! 可写配置 `<安装根>/data/config.toml`：**所有**跨启动持久化的设置（DebugMenu 窗口状态 +
//! 控件值 + 相机姿态）。菜单的**结构与控件缺省值**来自只读资源 `assets/ui/debug_menu.toml`
//! （每次启动都读一遍，见 `crate::debug_menu::load_menu`），本文件只存运行期改动过的值。

use std::collections::BTreeMap;
use std::path::PathBuf;

use bevy::prelude::*;
use serde::{Deserialize, Serialize};

use gate_ui::{DebugMenu, MenuValue, WindowState};

use crate::camera::{CameraMode, CameraPose, FlyCamera};

/// 配置文件名（相对 `data_dir`）
pub const CONFIG_PATH: &str = "config.toml";

/// 配置文件路径 `<安装根>/data/config.toml`
pub fn path() -> PathBuf {
  gate_render::data_dir().join(CONFIG_PATH)
}

/// 跨启动持久化的全部设置
#[derive(Resource, Serialize, Deserialize, Clone, Debug, Default)]
#[serde(default)]
pub struct Config {
  /// DebugMenu 窗口位置 / 收起 / 停留路径
  pub window: WindowState,
  /// DebugMenu 控件值（节点 id 路径 → 值；结构里没有的路径在套用时被忽略）
  pub menu: BTreeMap<String, MenuValue>,
  /// 相机姿态（首启 / 无该节 → None = 用场景默认机位）
  pub camera: Option<CameraPose>,
}

impl Config {
  /// 读 `data/config.toml`；缺失（首次启动）/ 读取或解析失败 → 全缺省 + 日志。
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

  /// 写 `data/config.toml`（目录不存在则创建）。
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

/// 退出前把配置写盘（AppExit 那一帧执行一次）：菜单窗口状态 + 控件值 + 相机姿态。
/// 从启动时读入的 `Config` 出发只覆盖本次运行确定有的部分（菜单未建出来时保留原值）。
/// 须排在 `bevy_window::ExitSystems` 之后（AppExit 由它写入）。
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
    // 运行期的停留路径 / 收起态只在组件上，存盘前回填进窗口状态
    let mut window = menu.model.window.clone();
    window.path = menu.path.clone();
    window.collapsed = menu.collapsed;
    out.window = window;
    out.menu = menu.model.values();
  }
  out.save();
}
