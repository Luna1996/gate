use std::collections::HashMap;
use std::io::{Cursor, Read};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use zip::ZipArchive;

pub type Json = serde_json::Value;

pub struct Texture {
  pub path: String,
  pub w: u32,
  pub h: u32,
  rgba: Vec<u8>,
  pub avg: [u8; 3],
  pub opaque: bool,
}

impl Texture {
  pub fn new(path: String, w: u32, h: u32, rgba: Vec<u8>) -> Self {
    let (mut acc, mut n) = ([0u64; 3], 0u64);
    for px in rgba.as_chunks::<4>().0 {
      if px[3] > 0 {
        acc[0] += px[0] as u64;
        acc[1] += px[1] as u64;
        acc[2] += px[2] as u64;
        n += 1;
      }
    }
    let d = n.max(1);
    let avg = [(acc[0] / d) as u8, (acc[1] / d) as u8, (acc[2] / d) as u8];
    let opaque = rgba.as_chunks::<4>().0.iter().all(|px| px[3] == 255);
    Self { path, w, h, rgba, avg, opaque }
  }

  pub fn pixels(&self) -> &[u8] {
    &self.rgba
  }

  pub fn sample(&self, u: f32, v: f32) -> [u8; 4] {
    let x = ((u.fract() + 1.0).fract() * self.w as f32) as u32;
    let y = ((v.fract() + 1.0).fract() * self.h as f32) as u32;
    let (x, y) = (x.min(self.w - 1), y.min(self.h - 1));
    let o = ((y * self.w + x) * 4) as usize;
    match self.rgba.get(o..o + 4) {
      Some(p) => [p[0], p[1], p[2], p[3]],
      None => [0, 0, 0, 0],
    }
  }
}

pub struct Assets {
  layers: Vec<Layer>,
  json: Mutex<HashMap<String, Option<Arc<Json>>>>,
  tex: Mutex<HashMap<String, Option<Arc<Texture>>>>,
  missing: Mutex<std::collections::BTreeSet<String>>,
}

enum Layer {
  Dir { root: PathBuf },
  Zip { base: String, archive: Mutex<ZipArchive<std::fs::File>> },
}

impl Layer {
  fn open(path: &Path) -> Option<Self> {
    if path.is_dir() {
      return Some(Self::Dir { root: dir_root(path) });
    }
    if !path.is_file() {
      return None;
    }
    let file = std::fs::File::open(path).ok()?;
    let mut archive = ZipArchive::new(file).ok()?;
    let base = zip_base(&mut archive);
    Some(Self::Zip { base, archive: Mutex::new(archive) })
  }

  fn kind(&self) -> &'static str {
    match self {
      Self::Dir { .. } => "目录",
      Self::Zip { .. } => "zip",
    }
  }

  fn read(&self, rel: &str) -> Option<Vec<u8>> {
    match self {
      Self::Dir { root } => std::fs::read(root.join(rel)).ok(),
      Self::Zip { base, archive } => {
        let name = format!("{base}{rel}");
        let mut a = archive.lock().unwrap_or_else(|e| e.into_inner());
        let mut f = a.by_name(&name).ok()?;
        let mut out = Vec::with_capacity(f.size() as usize);
        f.read_to_end(&mut out).ok()?;
        Some(out)
      }
    }
  }

  fn collect_files(&self, dir: &str, rel: &str, ext: &str, out: &mut Vec<String>) {
    match self {
      Self::Dir { root } => collect_files_dir(&root.join(dir), rel, ext, out),
      Self::Zip { base, archive } => {
        let prefix = format!("{base}{dir}/");
        let suffix = format!(".{ext}");
        let mut a = archive.lock().unwrap_or_else(|e| e.into_inner());
        for i in 0..a.len() {
          let Ok(entry) = a.by_index(i) else { continue };
          let Some(name) = entry.name().strip_prefix(&prefix) else { continue };
          if let Some(stem) = name.strip_suffix(&suffix) {
            out.push(format!("{rel}/{stem}"));
          }
        }
      }
    }
  }
}

impl Assets {
  pub fn new(stack: &[PathBuf]) -> Self {
    let mut layers = Vec::new();
    let mut labels = Vec::new();
    for path in stack {
      match Layer::open(path) {
        Some(layer) => {
          labels.push(format!("{}（{}）", path.display(), layer.kind()));
          layers.push(layer);
        }
        None => {
          bevy::log::warn!(target: "gate", "MC 资源! 打不开 {} ⇒ 该层跳过", path.display());
        }
      }
    }
    bevy::log::info!(target: "gate", "MC 资源栈: {}", labels.join(" → "));
    Self {
      layers,
      json: Mutex::new(HashMap::new()),
      tex: Mutex::new(HashMap::new()),
      missing: Mutex::new(Default::default()),
    }
  }

  pub fn blockstate(&self, block: &str) -> Option<Arc<Json>> {
    self.json_of(&format!("blockstates/{block}"), block)
  }

  pub fn model(&self, path: &str) -> Option<Arc<Json>> {
    self.json_of(&format!("models/{path}"), path)
  }

  pub fn texture(&self, path: &str) -> Option<Arc<Texture>> {
    self.texture_inner(path, true)
  }

  pub fn texture_probe(&self, path: &str) -> Option<Arc<Texture>> {
    self.texture_inner(path, false)
  }

  fn texture_inner(&self, path: &str, note: bool) -> Option<Arc<Texture>> {
    let path = safe_rel(path)?;
    let key = path.to_string();
    if let Some(hit) = self.tex.lock().unwrap_or_else(|e| e.into_inner()).get(&key) {
      if hit.is_none() && note {
        self.note_missing(&key);
      }
      return hit.clone();
    }
    let loaded = self
      .read(&format!("textures/{path}.png"))
      .and_then(|bytes| decode_png(&bytes, &key))
      .map(Arc::new);
    if loaded.is_none() && note {
      self.note_missing(&key);
    }
    let mut g = self.tex.lock().unwrap_or_else(|e| e.into_inner());
    g.insert(key, loaded.clone());
    loaded
  }

  pub fn block_texture_paths(&self) -> Vec<String> {
    let mut models = Vec::new();
    for layer in &self.layers {
      layer.collect_files("models", "models", "json", &mut models);
    }
    let mut out = Vec::new();
    for rel in models {
      let Some(path) = rel.strip_prefix("models/") else { continue };
      let Some(json) = self.model(path) else { continue };
      let Some(textures) = json.get("textures").and_then(|v| v.as_object()) else { continue };
      for value in textures.values() {
        let Some(name) = value.as_str() else { continue };
        let name = name.strip_prefix("minecraft:").unwrap_or(name);
        if name.starts_with("block/") {
          out.push(name.to_string());
        }
      }
    }
    out.sort();
    out.dedup();
    out
  }

  pub fn missing_names(&self) -> Vec<String> {
    self.missing.lock().unwrap_or_else(|e| e.into_inner()).iter().cloned().collect()
  }

  fn read(&self, rel: &str) -> Option<Vec<u8>> {
    self.layers.iter().find_map(|layer| layer.read(rel))
  }

  fn json_of(&self, rel: &str, name: &str) -> Option<Arc<Json>> {
    let key = rel.to_string();
    if let Some(hit) = self.json.lock().unwrap_or_else(|e| e.into_inner()).get(&key) {
      return hit.clone();
    }
    let loaded = self
      .read(&format!("{rel}.json"))
      .and_then(|s| serde_json::from_slice::<Json>(&s).ok())
      .map(Arc::new);
    if loaded.is_none() {
      self.note_missing(name);
    }
    let mut g = self.json.lock().unwrap_or_else(|e| e.into_inner());
    g.insert(key, loaded.clone());
    loaded
  }

  fn note_missing(&self, name: &str) {
    if self.missing.lock().unwrap_or_else(|e| e.into_inner()).insert(name.to_string()) {
      bevy::log::debug!(target: "gate", "MC 资产缺失 {name}");
    }
  }
}

fn dir_root(root: &Path) -> PathBuf {
  let nested = root.join("assets").join("minecraft");
  if nested.is_dir() { nested } else { root.to_path_buf() }
}

fn zip_base(archive: &mut ZipArchive<std::fs::File>) -> String {
  for i in 0..archive.len() {
    let Ok(entry) = archive.by_index(i) else { continue };
    let name = entry.name();
    if name.starts_with("assets/minecraft/blockstates/")
      || name.starts_with("assets/minecraft/models/")
      || name.starts_with("assets/minecraft/textures/")
    {
      return "assets/minecraft/".to_string();
    }
    if name.starts_with("blockstates/")
      || name.starts_with("models/")
      || name.starts_with("textures/")
    {
      return String::new();
    }
  }
  String::new()
}

fn safe_rel(path: &str) -> Option<&str> {
  let p = path.strip_prefix("minecraft:").unwrap_or(path);
  if p.is_empty() || p.starts_with('/') || p.contains("..") || p.contains('\\') {
    return None;
  }
  if !p.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b'/' | b'-')) {
    return None;
  }
  Some(p)
}

fn collect_files_dir(dir: &Path, rel: &str, ext: &str, out: &mut Vec<String>) {
  let Ok(entries) = std::fs::read_dir(dir) else { return };
  for entry in entries.flatten() {
    let path = entry.path();
    let Some(name) = path.file_name().and_then(|n| n.to_str()) else { continue };
    if path.is_dir() {
      collect_files_dir(&path, &format!("{rel}/{name}"), ext, out);
    } else if path.extension().is_some_and(|e| e.eq_ignore_ascii_case(ext)) {
      let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or(name);
      out.push(format!("{rel}/{stem}"));
    }
  }
}

fn decode_png(bytes: &[u8], id: &str) -> Option<Texture> {
  let mut dec = png::Decoder::new(Cursor::new(bytes));
  dec.set_transformations(png::Transformations::normalize_to_color8());
  let mut reader = dec.read_info().ok()?;
  let mut buf = vec![0u8; reader.output_buffer_size()?];
  let info = reader.next_frame(&mut buf).ok()?;
  let (w, h) = (info.width, info.height);
  let channels = match info.color_type {
    png::ColorType::Rgb => 3,
    png::ColorType::Rgba => 4,
    png::ColorType::Grayscale => 1,
    png::ColorType::GrayscaleAlpha => 2,
    png::ColorType::Indexed => return None,
  };
  let frame_h = if h > w && h % w == 0 { w } else { h };
  let src = &buf[..info.buffer_size()];
  let mut rgba = Vec::with_capacity((w * frame_h * 4) as usize);
  for y in 0..frame_h {
    for x in 0..w {
      let o = ((y * w + x) * channels as u32) as usize;
      let px = src.get(o..o + channels as usize)?;
      let (r, g, b, a) = match channels {
        4 => (px[0], px[1], px[2], px[3]),
        3 => (px[0], px[1], px[2], 255),
        2 => (px[0], px[0], px[0], px[1]),
        _ => (px[0], px[0], px[0], 255),
      };
      rgba.extend_from_slice(&[r, g, b, a]);
    }
  }
  Some(Texture::new(id.to_string(), w, frame_h, rgba))
}
