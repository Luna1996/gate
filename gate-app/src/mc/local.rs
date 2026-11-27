use std::path::PathBuf;
use std::sync::OnceLock;

use serde::Deserialize;

const FILE: &str = "mc.local.toml";

#[derive(Deserialize)]
struct File {
  map: Option<String>,
  assets: Option<String>,
  models: Option<String>,
  vanilla: Option<String>,
}

pub struct Paths {
  pub map: PathBuf,
  pub assets: PathBuf,
  pub models: Option<PathBuf>,
  pub vanilla: Option<PathBuf>,
}

static CACHE: OnceLock<Result<Paths, String>> = OnceLock::new();

pub fn get() -> Result<&'static Paths, &'static str> {
  CACHE.get_or_init(load).as_ref().map_err(String::as_str)
}

fn load() -> Result<Paths, String> {
  let path = gate_render::data_dir().join(FILE);
  let text = std::fs::read_to_string(&path).map_err(|e| {
    format!(
      "MC 本地配置! {} 读取失败 {e}；复制 data/mc.local.example.toml 为 data/mc.local.toml 并填写",
      path.display()
    )
  })?;
  let file: File =
    toml::from_str(&text).map_err(|e| format!("MC 本地配置! {} 解析失败 {e}", path.display()))?;
  let map = required("GATE_GREEN_FIELD", file.map, "map", &path)?;
  let assets = required("GATE_MC_ASSETS", file.assets, "assets", &path)?;
  if !map.is_dir() {
    return Err(format!("MC 本地配置! map {} 不是目录", map.display()));
  }
  if !assets.exists() {
    return Err(format!("MC 本地配置! assets {} 不存在", assets.display()));
  }
  let models = optional("GATE_MC_MODELS", file.models);
  let vanilla = optional("GATE_MC_VANILLA", file.vanilla);
  Ok(Paths { map, assets, models, vanilla })
}

fn required(
  env: &str,
  value: Option<String>,
  key: &str,
  path: &std::path::Path,
) -> Result<PathBuf, String> {
  std::env::var_os(env)
    .map(PathBuf::from)
    .or_else(|| value.map(PathBuf::from))
    .ok_or_else(|| format!("MC 本地配置! {} 缺 {key}", path.display()))
}

fn optional(env: &str, value: Option<String>) -> Option<PathBuf> {
  let path = std::env::var_os(env).map(PathBuf::from).or_else(|| value.map(PathBuf::from))?;
  if path.exists() {
    return Some(path);
  }
  bevy::log::warn!(target: "gate", "MC 本地配置! {env} {} 不存在 ⇒ 忽略", path.display());
  None
}
