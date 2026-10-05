use std::path::PathBuf;

pub fn install_root() -> PathBuf {
  if let Some(dir) = std::env::current_exe().ok().and_then(|exe| exe.parent().map(PathBuf::from))
    && dir.join("assets").is_dir()
  {
    return dir;
  }
  PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
}

pub fn assets_dir() -> PathBuf {
  install_root().join("assets")
}

pub fn logs_dir() -> PathBuf {
  install_root().join("logs")
}

pub fn data_dir() -> PathBuf {
  install_root().join("data")
}

pub fn dda_wesl_dir() -> PathBuf {
  assets_dir().join("shaders").join("voxel_raytrace")
}
