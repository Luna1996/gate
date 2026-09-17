//! 运行期路径定位：`assets/`（只读资源）、`logs/`、`data/`（可写）。
//! 根目录由 [`install_root`] 判定：便携发布 = exe 同目录（存在 `assets/`）；开发 = 源码树根；`GATE_ROOT` 可强制覆盖。

use std::path::PathBuf;

/// 安装根（`assets/`、`logs/`、`data/` 的父目录）
pub fn install_root() -> PathBuf {
  if let Ok(root) = std::env::var("GATE_ROOT") {
    return PathBuf::from(root);
  }
  if let Some(dir) = std::env::current_exe().ok().and_then(|exe| exe.parent().map(PathBuf::from))
    && dir.join("assets").is_dir()
  {
    return dir;
  }
  PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
}

/// 只读资源根（Bevy `AssetPlugin::file_path` 用的就是它）
pub fn assets_dir() -> PathBuf {
  install_root().join("assets")
}

/// 日志目录（可写；每次启动截断重写 `latest.log`）
pub fn logs_dir() -> PathBuf {
  install_root().join("logs")
}

/// 运行期数据目录（可写；菜单状态等需要跨启动保留、但不属于源码资源的文件）
pub fn data_dir() -> PathBuf {
  install_root().join("data")
}

/// voxel raytrace WESL 包目录（内含 `main.wesl` 与各子模块；启动时读盘编译）
pub fn dda_wesl_dir() -> PathBuf {
  assets_dir().join("shaders").join("voxel_raytrace")
}
