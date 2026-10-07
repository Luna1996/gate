use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

pub type Json = serde_json::Value;

pub struct Texture {
  pub w: u32,
  pub h: u32,
  rgba: Vec<u8>,
  pub avg: [u8; 3],
}

impl Texture {
  pub fn new(w: u32, h: u32, rgba: Vec<u8>) -> Self {
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
    Self { w, h, rgba, avg }
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
  root: PathBuf,
  json: Mutex<HashMap<String, Option<Arc<Json>>>>,
  tex: Mutex<HashMap<String, Option<Arc<Texture>>>>,
  missing: Mutex<std::collections::BTreeSet<String>>,
}

impl Assets {
  pub fn new(root: impl Into<PathBuf>) -> Self {
    Self {
      root: root.into(),
      json: Mutex::new(HashMap::new()),
      tex: Mutex::new(HashMap::new()),
      missing: Mutex::new(Default::default()),
    }
  }

  pub fn root(&self) -> &Path {
    &self.root
  }

  pub fn blockstate(&self, block: &str) -> Option<Arc<Json>> {
    self.json_of(&format!("blockstates/{block}"), block)
  }

  pub fn model(&self, path: &str) -> Option<Arc<Json>> {
    self.json_of(&format!("models/{path}"), path)
  }

  pub fn texture(&self, path: &str) -> Option<Arc<Texture>> {
    let path = safe_rel(path)?;
    let key = path.to_string();
    if let Some(hit) = self.tex.lock().unwrap_or_else(|e| e.into_inner()).get(&key) {
      return hit.clone();
    }
    let file = self.root.join(format!("textures/{path}.png"));
    let loaded = decode_png(&file).map(Arc::new);
    if loaded.is_none() {
      self.note_missing(&key);
    }
    let mut g = self.tex.lock().unwrap_or_else(|e| e.into_inner());
    g.insert(key, loaded.clone());
    loaded
  }

  pub fn missing_names(&self) -> Vec<String> {
    self.missing.lock().unwrap_or_else(|e| e.into_inner()).iter().cloned().collect()
  }

  fn json_of(&self, rel: &str, name: &str) -> Option<Arc<Json>> {
    let key = rel.to_string();
    if let Some(hit) = self.json.lock().unwrap_or_else(|e| e.into_inner()).get(&key) {
      return hit.clone();
    }
    let loaded = std::fs::read_to_string(self.root.join(format!("{rel}.json")))
      .ok()
      .and_then(|s| serde_json::from_str::<Json>(&s).ok())
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

fn decode_png(path: &Path) -> Option<Texture> {
  let file = std::fs::File::open(path).ok()?;
  let mut dec = png::Decoder::new(std::io::BufReader::new(file));
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
  Some(Texture::new(w, frame_h, rgba))
}
