use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::SystemTime;

use serde::Deserialize;

const FILE: &str = "mc.local.toml";
const SAVE_PREFIX: &str = "greenfield";

#[derive(Deserialize)]
struct File {
  mc: Option<String>,
  map: Option<String>,
  packs: Option<Vec<String>>,
  vanilla: Option<String>,
}

pub struct Paths {
  pub mc: PathBuf,
  pub map: PathBuf,
  pub stack: Vec<PathBuf>,
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
  let mc = std::env::var_os("GATE_MC_VERSION")
    .map(PathBuf::from)
    .or_else(|| file.mc.map(PathBuf::from))
    .ok_or_else(|| format!("MC 本地配置! {} 缺 mc", path.display()))?;
  if !mc.is_dir() {
    return Err(format!("MC 本地配置! mc {} 不是目录", mc.display()));
  }
  let map = match std::env::var_os("GATE_GREEN_FIELD")
    .map(PathBuf::from)
    .or_else(|| file.map.map(PathBuf::from))
  {
    Some(path) => path,
    None => find_save(&mc)?,
  };
  let vanilla = std::env::var_os("GATE_MC_VANILLA")
    .map(PathBuf::from)
    .or_else(|| file.vanilla.map(PathBuf::from))
    .or_else(|| find_jar(&mc));
  let mut stack = Vec::new();
  if let Some(jar) = vanilla {
    stack.push(jar);
  }
  let packs = match file.packs {
    Some(list) if !list.is_empty() => list.into_iter().map(PathBuf::from).collect(),
    _ => pack_stack(&mc),
  };
  stack.extend(packs);
  if let Some(extra) = std::env::var_os("GATE_MC_ASSETS") {
    stack.push(PathBuf::from(extra));
  }
  stack.reverse();
  Ok(Paths { mc, map, stack })
}

fn find_save(mc: &Path) -> Result<PathBuf, String> {
  let saves = mc.join("saves");
  let mut found: Vec<(SystemTime, PathBuf)> = std::fs::read_dir(&saves)
    .map_err(|e| format!("MC 存档! {} 读取失败 {e}", saves.display()))?
    .flatten()
    .filter(|e| e.path().is_dir())
    .filter(|e| e.file_name().to_string_lossy().to_lowercase().starts_with(SAVE_PREFIX))
    .map(|e| {
      let path = e.path();
      let time = std::fs::metadata(path.join("level.dat"))
        .and_then(|m| m.modified())
        .or_else(|_| e.metadata().and_then(|m| m.modified()))
        .unwrap_or(SystemTime::UNIX_EPOCH);
      (time, path)
    })
    .collect();
  found.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| b.1.cmp(&a.1)));
  found
    .into_iter()
    .next()
    .map(|(_, path)| path)
    .ok_or_else(|| format!("MC 存档! {} 下没有以 `Greenfield` 开头的存档", saves.display()))
}

fn find_jar(mc: &Path) -> Option<PathBuf> {
  let mut best: Option<(u64, PathBuf)> = None;
  let own = mc.file_name().and_then(|n| n.to_str()).map(|n| mc.join(format!("{n}.jar")));
  if let Some(path) = own.filter(|p| p.is_file()) {
    return Some(path);
  }
  for entry in std::fs::read_dir(mc).ok()?.flatten() {
    let path = entry.path();
    if !path.is_file() || path.extension().is_none_or(|e| !e.eq_ignore_ascii_case("jar")) {
      continue;
    }
    let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
    if best.as_ref().is_none_or(|(s, _)| size > *s) {
      best = Some((size, path));
    }
  }
  best.map(|(_, path)| path)
}

fn pack_stack(mc: &Path) -> Vec<PathBuf> {
  let dir = mc.join("resourcepacks");
  let mut entries: Vec<PathBuf> = std::fs::read_dir(&dir)
    .into_iter()
    .flatten()
    .flatten()
    .map(|e| e.path())
    .filter(|p| p.is_dir() || p.extension().is_some_and(|e| e.eq_ignore_ascii_case("zip")))
    .collect();
  entries.sort();
  let requested = options_packs(mc);
  let mut stack: Vec<PathBuf> = Vec::new();
  for name in &requested {
    let Some(name) = name.strip_prefix("file/") else { continue };
    match resolve_pack(&entries, name) {
      Some(path) => {
        if !stack.contains(&path) {
          stack.push(path);
        }
      }
      None => {
        bevy::log::warn!(target: "gate", "MC 资源包! options.txt 引用 `{name}` 不存在 ⇒ 跳过");
      }
    }
  }
  for path in entries {
    if !stack.contains(&path) {
      stack.push(path);
    }
  }
  stack
}

fn options_packs(mc: &Path) -> Vec<String> {
  let Ok(text) = std::fs::read_to_string(mc.join("options.txt")) else {
    return Vec::new();
  };
  for line in text.lines() {
    let Some(list) = line.strip_prefix("resourcePacks:") else { continue };
    if let Ok(packs) = serde_json::from_str::<Vec<String>>(list.trim()) {
      return packs;
    }
  }
  Vec::new()
}

fn resolve_pack(entries: &[PathBuf], name: &str) -> Option<PathBuf> {
  let lower = name.to_lowercase();
  if let Some(path) = find_pack(entries, |n| n == lower) {
    return Some(path);
  }
  let stem = lower.strip_suffix(".zip").unwrap_or(&lower).to_string();
  if let Some(path) = find_pack(entries, |n| n.trim_end_matches(".zip") == stem.as_str()) {
    return Some(path);
  }
  let head = stem.split('.').next().unwrap_or(&stem);
  (head.len() >= 4).then(|| find_pack(entries, |n| n.starts_with(head))).flatten()
}

fn find_pack(entries: &[PathBuf], pred: impl Fn(&str) -> bool) -> Option<PathBuf> {
  let mut hits = entries
    .iter()
    .filter(|p| p.file_name().is_some_and(|n| pred(&n.to_string_lossy().to_lowercase())));
  let first = hits.next()?;
  hits.next().is_none().then(|| first.clone())
}
