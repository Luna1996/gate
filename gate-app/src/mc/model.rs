use std::collections::HashMap;
use std::sync::Arc;

use serde_json::Value as J;

use super::assets::{Assets, Texture};
use super::world::BlockState;

pub const FACE_NAMES: [&str; 6] = ["down", "up", "north", "south", "west", "east"];
pub const FACE_PRIORITY: [usize; 6] = [1, 0, 4, 5, 2, 3];

pub struct Face {
  pub tex: Arc<Texture>,
  pub uv: [f32; 4],
  pub rot: u16,
  pub tint: Option<u8>,
}

pub struct Element {
  pub from: [f32; 3],
  pub to: [f32; 3],
  pub faces: [Option<Face>; 6],
}

#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct Rot {
  pub x: u16,
  pub y: u16,
}

pub struct SubModel {
  pub elements: Vec<Element>,
  pub rot: Rot,
}

pub fn resolve(assets: &Assets, st: &BlockState) -> Option<Vec<SubModel>> {
  if super::world::is_air(&st.name) {
    return None;
  }
  let mut subs = Vec::new();
  if let Some(bs) = assets.blockstate(st.short_name()) {
    if let Some(vars) = bs.get("variants").and_then(J::as_object) {
      for (key, val) in vars {
        if cond_matches(key, st) {
          subs.extend(applies(assets, val));
          break;
        }
      }
    }
    if let Some(parts) = bs.get("multipart").and_then(J::as_array) {
      for part in parts {
        if !part.get("when").map(|w| when_matches(w, st)).unwrap_or(true) {
          continue;
        }
        if let Some(apply) = part.get("apply") {
          subs.extend(applies(assets, apply));
        }
      }
    }
  }
  if subs.is_empty() {
    subs.extend(fallback(assets, st));
  }
  (!subs.is_empty()).then_some(subs)
}

fn applies(assets: &Assets, v: &J) -> Vec<SubModel> {
  match v {
    J::Array(items) => items.iter().take(1).filter_map(|i| one(assets, i)).collect(),
    other => one(assets, other).into_iter().collect(),
  }
}

fn one(assets: &Assets, v: &J) -> Option<SubModel> {
  let (path, rot) = match v {
    J::String(s) => (s.clone(), Rot::default()),
    J::Object(o) => {
      (o.get("model")?.as_str()?.to_string(), Rot { x: deg(o.get("x")), y: deg(o.get("y")) })
    }
    _ => return None,
  };
  let md = load_model(assets, &path, 0)?;
  let elements = bake(assets, &md)?;
  Some(SubModel { elements, rot })
}

fn deg(v: Option<&J>) -> u16 {
  let d = v.and_then(J::as_i64).unwrap_or(0).rem_euclid(360);
  (((d + 45) / 90 * 90) % 360) as u16
}

#[derive(Default)]
struct RawModel {
  textures: HashMap<String, String>,
  elements: Option<Vec<J>>,
}

fn load_model(assets: &Assets, path: &str, depth: usize) -> Option<RawModel> {
  if depth > 8 {
    return None;
  }
  let j = assets.model(strip_ns(path))?;
  let mut m = match j.get("parent").and_then(J::as_str) {
    Some(p) => load_model(assets, p, depth + 1)?,
    None => RawModel::default(),
  };
  if let Some(t) = j.get("textures").and_then(J::as_object) {
    for (k, v) in t {
      if let Some(s) = v.as_str() {
        m.textures.insert(k.clone(), s.to_string());
      }
    }
  }
  if let Some(e) = j.get("elements").and_then(J::as_array) {
    m.elements = Some(e.clone());
  }
  Some(m)
}

fn bake(assets: &Assets, md: &RawModel) -> Option<Vec<Element>> {
  let raw = md.elements.as_ref()?;
  let mut out = Vec::with_capacity(raw.len());
  for e in raw {
    let from = vec3(e.get("from"))?;
    let to = vec3(e.get("to"))?;
    let mut faces: [Option<Face>; 6] = Default::default();
    if let Some(f) = e.get("faces").and_then(J::as_object) {
      for (i, name) in FACE_NAMES.iter().enumerate() {
        let Some(fv) = f.get(*name) else { continue };
        let Some(tex_name) = resolve_var(&md.textures, fv.get("texture").and_then(J::as_str)?)
        else {
          continue;
        };
        let Some(tex) = assets.texture(&tex_name) else { continue };
        let uv = match fv.get("uv").and_then(J::as_array).and_then(|a| quad(a)) {
          Some(uv) => uv,
          None => default_uv(i, from, to),
        };
        faces[i] = Some(Face {
          tex,
          uv,
          rot: (fv.get("rotation").and_then(J::as_u64).unwrap_or(0) as u16) % 360,
          tint: fv.get("tintindex").and_then(J::as_i64).map(|v| v.clamp(0, 255) as u8),
        });
      }
    }
    if faces.iter().all(Option::is_none) {
      continue;
    }
    out.push(Element { from, to, faces });
  }
  (!out.is_empty()).then_some(out)
}

fn fallback(assets: &Assets, st: &BlockState) -> Option<SubModel> {
  let short = st.short_name();
  let tex = assets
    .texture(&format!("block/{short}"))
    .or_else(|| assets.texture(&format!("block/{short}_still")))?;
  let faces: [Option<Face>; 6] = std::array::from_fn(|i| {
    Some(Face { tex: tex.clone(), uv: default_uv(i, [0.0; 3], [16.0; 3]), rot: 0, tint: Some(0) })
  });
  Some(SubModel {
    elements: vec![Element { from: [0.0; 3], to: [16.0; 3], faces }],
    rot: Rot::default(),
  })
}

fn vec3(v: Option<&J>) -> Option<[f32; 3]> {
  let a = v?.as_array()?;
  if a.len() < 3 {
    return None;
  }
  Some([a[0].as_f64()? as f32, a[1].as_f64()? as f32, a[2].as_f64()? as f32])
}

fn quad(a: &[J]) -> Option<[f32; 4]> {
  if a.len() < 4 {
    return None;
  }
  Some([a[0].as_f64()? as f32, a[1].as_f64()? as f32, a[2].as_f64()? as f32, a[3].as_f64()? as f32])
}

fn resolve_var(map: &HashMap<String, String>, name: &str) -> Option<String> {
  let mut cur = name.to_string();
  for _ in 0..8 {
    let Some(key) = cur.strip_prefix('#') else { return Some(strip_ns(&cur).to_string()) };
    cur = map.get(key)?.clone();
  }
  None
}

fn strip_ns(p: &str) -> &str {
  p.split_once(':').map(|(_, r)| r).unwrap_or(p)
}

pub fn default_uv(f: usize, from: [f32; 3], to: [f32; 3]) -> [f32; 4] {
  let [x0, y0, z0] = from;
  let [x1, y1, z1] = to;
  match f {
    0 => [x0, 16.0 - z1, x1, 16.0 - z0],
    1 => [x0, z0, x1, z1],
    2 => [16.0 - x1, 16.0 - y1, 16.0 - x0, 16.0 - y0],
    3 => [x0, 16.0 - y1, x1, 16.0 - y0],
    4 => [z0, 16.0 - y1, z1, 16.0 - y0],
    _ => [16.0 - z1, 16.0 - y1, 16.0 - z0, 16.0 - y0],
  }
}

pub fn cond_matches(cond: &str, st: &BlockState) -> bool {
  let c = cond.trim();
  if c.is_empty() {
    return true;
  }
  c.split(',').all(|part| match part.trim().split_once('=') {
    Some((k, v)) => v.split('|').any(|alt| st.prop(k.trim()) == Some(alt.trim())),
    None => false,
  })
}

fn when_matches(w: &J, st: &BlockState) -> bool {
  match w {
    J::String(s) => cond_matches(s, st),
    J::Object(o) => {
      if let Some(alts) = o.get("OR").and_then(J::as_array) {
        return alts.iter().any(|x| when_matches(x, st));
      }
      if o.is_empty() {
        return true;
      }
      o.iter().all(|(k, v)| {
        let want = match v {
          J::String(s) => s.clone(),
          other => other.to_string(),
        };
        want.split('|').any(|alt| st.prop(k) == Some(alt))
      })
    }
    _ => false,
  }
}
