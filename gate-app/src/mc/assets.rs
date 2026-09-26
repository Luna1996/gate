//! **MC 资源包资产**：blockstate / model 的 JSON 与 block 贴图的 PNG（16×16）。
//!
//! 只做两件事：
//! - **读**：`<root>/blockstates/<名>.json`、`<root>/models/<路径>.json`、`<root>/textures/<路径>.png`。
//!   目录结构 = 资源包（覆盖）+ 客户端 jar（兜底）解出来的一份，见 `docs/mc_map.md` §3。
//! - **缓存 + 并发**：worker 线程共享一个 `&Assets` ⇒ 两张表都进 `Mutex`；**失败也缓存**
//!   （`None`），否则一张缺失的贴图会被每个区块重新 `open + 解码` 一次。
//!
//! 名字一律**去命名空间**（`minecraft:block/stone` → `block/stone`）：解包时已把两份来源合并进同一棵树，
//! 命名空间只剩路径含义；同时挡住 `..` / 绝对路径（外部资源包的名字不可信）。
//!
//! ## 贴图的两处细节
//! - **动画贴图**：`<名>.png` 可能是"竖着叠 N 帧"的长图（`water_still` 16×512、`fire_0` 16×… ）。
//!   本仓不做动画，只取**第一帧**（`h > w && h % w == 0` ⇒ 有效高度 = `w`）；
//! - **`avg`**：整张贴图非全透明 texel 的平均色（给"元素内部"与粗档当代表色用）。

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

pub type Json = serde_json::Value;

/// 一张解码好的 block 贴图（RGBA8，行序 = PNG 行序，v 向下）
pub struct Texture {
  /// 有效宽度（texel）
  pub w: u32,
  /// 有效高度（texel）= 第一帧的高度
  pub h: u32,
  rgba: Vec<u8>,
  /// 非全透明 texel 的平均色
  pub avg: [u8; 3],
}

impl Texture {
  /// 从 RGBA8 字节建一张（行序 = 行优先，v 向下）；`avg` = 非全透明 texel 的平均色
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
    let d = n.max(1); // 全透明贴图 ⇒ 均色取 0（不除零）
    let avg = [(acc[0] / d) as u8, (acc[1] / d) as u8, (acc[2] / d) as u8];
    Self { w, h, rgba, avg }
  }

  /// `(u, v)` ∈ [0,1) 处的 texel 颜色（**最近邻**：16×16 的方块贴图做双线性只会糊掉边界）
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

/// 资产根目录（`blockstates/` + `models/` + `textures/` 的父目录）
pub struct Assets {
  root: PathBuf,
  json: Mutex<HashMap<String, Option<Arc<Json>>>>,
  tex: Mutex<HashMap<String, Option<Arc<Texture>>>>,
  /// 缺失的名字（诊断；每个名字只记一次）
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

  /// `<root>/blockstates/<方块名>.json`（方块名 = 去命名空间的 `oak_stairs` 这种）
  pub fn blockstate(&self, block: &str) -> Option<Arc<Json>> {
    self.json_of(&format!("blockstates/{block}"), block)
  }

  /// `<root>/models/<路径>.json`（路径 = `block/oak_slab` 这种，与 MC 引用同形）
  pub fn model(&self, path: &str) -> Option<Arc<Json>> {
    self.json_of(&format!("models/{path}"), path)
  }

  /// `<root>/textures/<路径>.png`（路径 = `block/stone` 这种）
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

  /// 已记下的缺失名（诊断用）
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

/// 相对路径白名单：只允许 `[a-z0-9_./-]` 且不含 `..`（外部资源包的名字不可信）
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

/// PNG → RGBA8。`normalize_to_color8()` 让灰度 / 调色板 / 16 位统一成 8 位 RGB(A)，
/// 后面就不用逐种颜色型写分支。
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
    png::ColorType::Indexed => return None, // normalize_to_color8 已展开索引色
  };
  // 动画贴图：只取第一帧（竖排 N 帧，每帧高 = 宽）
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

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn relative_paths_reject_escapes_and_namespaces() {
    assert_eq!(safe_rel("minecraft:block/stone"), Some("block/stone"));
    assert_eq!(safe_rel("block/oak_slab"), Some("block/oak_slab"));
    assert_eq!(safe_rel("../../etc/passwd"), None);
    assert_eq!(safe_rel("/abs"), None);
    assert_eq!(safe_rel("block/sto ne"), None);
    assert_eq!(safe_rel(""), None);
  }

  /// 资产根缺目录时全部返回 `None` 且只记一次缺失（不会 panic、不会每次重读盘）
  #[test]
  fn missing_assets_are_none_and_cached() {
    let a = Assets::new(std::env::temp_dir().join("gate_mc_assets_none"));
    assert!(a.blockstate("stone").is_none());
    assert!(a.model("block/cube_all").is_none());
    assert!(a.texture("block/stone").is_none());
    assert_eq!(a.missing_names().len(), 3);
    assert!(a.texture("block/stone").is_none());
    assert_eq!(a.missing_names().len(), 3, "同一个名字只记一次");
  }

  /// **真资产冒烟**（`#[ignore]`）：解码出生点区块用到的几张关键贴图 —— 这是"贴图真能读进来、
  /// 尺寸与均色合理"的直接证据（`grass_block_top` 是灰度贴图，均色应接近中性灰偏亮）。
  /// 跑法：`cargo test --release -p gate-app --bin gate-app -- --ignored --nocapture real_assets`
  #[test]
  #[ignore = "需要外部资产目录：设 GATE_MC_ASSETS，或放默认路径"]
  fn real_assets_decode() {
    let root = std::env::var("GATE_MC_ASSETS")
      .unwrap_or_else(|_| r"C:\game\Greenfield v0.5.4\assets_mc".to_string());
    let a = Assets::new(&root);
    for name in [
      "block/stone",
      "block/white_terracotta",
      "block/oak_planks",
      "block/grass_block_top",
      "block/water_still",
      "block/glass",
      "block/light_gray_stained_glass",
    ] {
      let t = a.texture(name).unwrap_or_else(|| panic!("{name} 应能解码"));
      let c = t.sample(0.5, 0.5);
      println!("{name:<32} {}×{} 中心 texel {:?} 均色 {:?}", t.w, t.h, c, t.avg);
      assert!(t.w >= 16 || t.w == 0, "{name} 宽度异常");
    }
    println!("缺失资产 = {:?}", a.missing_names());
    assert!(a.missing_names().is_empty(), "这几张都不该缺");
  }
}
