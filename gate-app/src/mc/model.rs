//! **blockstate / model → 元素盒**：把 MC 的形状描述翻成"哪些 1/16 格里是实体、每格取哪个 texel"。
//!
//! 覆盖的规则（`docs/mc_map.md` §5 第 2 条）：
//! - `variants`：键 = `性质=值` 用 `,` 连（AND），值可用 `|` 给多个候选；按方块的 `Properties` 匹配；
//! - `multipart`：逐个 `when` 判定（`when` 可以是字符串、或 `{"OR": [...]}`），命中就把 `apply` 叠上去；
//! - `parent` 继承链：**父在前、子覆盖**（子有 `elements` 就整段替换，`textures` 逐键覆盖）；
//! - 贴图变量：`#all` 这类引用按 `textures` 表逐跳解析（带深度上限），叶子是 `block/stone` 这种路径；
//! - 面的 uv：缺省按元素盒的六个面推（与 MC 的 `FaceBakery` 同一套公式），显式 uv 原样取；
//! - **块级旋转**（`variants`/`multipart` 的 `x`/`y`）：记在 [`SubModel::rot`] 上，由 [`super::voxel`]
//!   在**体素层面**整体旋转 —— 那样 uv 就不必跟着旋转换算（先按未旋转的模型空间出 texel，再旋格）。
//!
//! 不覆盖（首版明确的取舍）：元素自带的 `rotation`（植物的 45° 交叉面 → 退化成轴对齐薄片）、
//! `uvlock`、`display`、`ambientocclusion`。见 `docs/mc_map.md` §6。

use std::collections::HashMap;
use std::sync::Arc;

use serde_json::Value as J;

use super::assets::{Assets, Texture};
use super::world::BlockState;

/// 面的方向名与下标（与 MC 一致；也是 [`Element::faces`] 的下标）
pub const FACE_NAMES: [&str; 6] = ["down", "up", "north", "south", "west", "east"];
/// 体素化时的**面优先序**：同一个格同时贴着两个面时取谁。
/// 取"上"优先是权衡后的定：上表面在俯视里最常被看到，代价是侧面最上一行的格取了顶面 texel
/// （`grass_block` 的侧面顶行本来就是草色 ⇒ 这条偏差几乎不可见）。
pub const FACE_PRIORITY: [usize; 6] = [1, 0, 4, 5, 2, 3];

/// 一个面：贴图 + texel 空间的 uv 矩形（0..16）+ 面内旋转 + 染色档
pub struct Face {
  pub tex: Arc<Texture>,
  /// `[u1, v1, u2, v2]`，texel 单位（0..16）
  pub uv: [f32; 4],
  /// 面内旋转（0 / 90 / 180 / 270）
  pub rot: u16,
  /// `tintindex`（有值 ⇒ 该面要按生物群系染色，见 [`super::voxel`]）
  pub tint: Option<u8>,
}

/// 一个元素盒（模型空间 0..16，单位 = 1/16 方块 = 1 个体素）
pub struct Element {
  pub from: [f32; 3],
  pub to: [f32; 3],
  pub faces: [Option<Face>; 6],
}

/// 块级旋转（blockstate 的 `x` / `y`，已归一到 0/90/180/270）
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct Rot {
  pub x: u16,
  pub y: u16,
}

/// 一个"子模型"：`variants` 命中的那一条 / `multipart` 的一个 `apply`
pub struct SubModel {
  pub elements: Vec<Element>,
  pub rot: Rot,
}

/// blockstate → 子模型列表（`None` = 这个方块渲染不出任何东西）
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
          break; // variants 只取命中一条（正常数据里恰好一条匹配）
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
    // 兜底整块：没有 blockstate（水 / 熔岩这类流体）或形状整段缺失时，用同名贴图铺一个实心方块
    subs.extend(fallback(assets, st));
  }
  (!subs.is_empty()).then_some(subs)
}

/// `apply` / variant 值 → 子模型：可以是字符串（模型路径），也可以是数组（等概率取一个；本仓取第一个）
fn applies(assets: &Assets, v: &J) -> Vec<SubModel> {
  match v {
    J::Array(items) => items.iter().take(1).filter_map(|i| one(assets, i)).collect(),
    other => one(assets, other).into_iter().collect(),
  }
}

fn one(assets: &Assets, v: &J) -> Option<SubModel> {
  let (path, rot) = match v {
    J::String(s) => (s.clone(), Rot::default()),
    J::Object(o) => (
      o.get("model")?.as_str()?.to_string(),
      Rot {
        x: deg(o.get("x")),
        y: deg(o.get("y")),
      },
    ),
    _ => return None,
  };
  let md = load_model(assets, &path, 0)?;
  let elements = bake(assets, &md)?;
  Some(SubModel { elements, rot })
}

/// 角度归一到 0/90/180/270
fn deg(v: Option<&J>) -> u16 {
  let d = v.and_then(J::as_i64).unwrap_or(0).rem_euclid(360);
  (((d + 45) / 90 * 90) % 360) as u16
}

/// 模型的继承链（父在前）+ 贴图变量表
#[derive(Default)]
struct RawModel {
  textures: HashMap<String, String>,
  elements: Option<Vec<J>>,
}

/// 读一个模型并沿 `parent` 上溯（深度上限 8：资源包偶尔有环）
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

/// 展开成 `Element`：解析贴图变量、补缺省 uv。**一张贴图缺失 ⇒ 只丢那个面**（其余面照画）。
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
        let Some(tex_name) = resolve_var(&md.textures, fv.get("texture").and_then(J::as_str)?) else {
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

/// 兜底：整块 + 同名贴图（`block/<名>`，退一步 `block/<名>_still`）。
/// `tintindex` 记 0 ⇒ 走 [`super::voxel::tint_of`] 的按名染色（水才会是蓝的）。
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

/// `#变量` → 贴图路径（逐跳解析，深度上限 8）
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

/// 面的**缺省 uv**（texel 单位）：与 MC `FaceBakery` 同一套 —— 只要元素盒的角点。
/// 注意 north / east / down 的 u 或 v 是**反向**的（`u1 > u2`），插值时按"位置比例 0→1"走即可。
pub fn default_uv(f: usize, from: [f32; 3], to: [f32; 3]) -> [f32; 4] {
  let [x0, y0, z0] = from;
  let [x1, y1, z1] = to;
  match f {
    0 => [x0, 16.0 - z1, x1, 16.0 - z0],           // down
    1 => [x0, z0, x1, z1],                        // up
    2 => [16.0 - x1, 16.0 - y1, 16.0 - x0, 16.0 - y0], // north
    3 => [x0, 16.0 - y1, x1, 16.0 - y0],          // south
    4 => [z0, 16.0 - y1, z1, 16.0 - y0],          // west
    _ => [16.0 - z1, 16.0 - y1, 16.0 - z0, 16.0 - y0], // east
  }
}

/// variant 键 / `when` 字符串的匹配：`,` 连的多个条件是 AND，值可用 `|` 给候选。
/// 空串 = 无条件成立（MC 的"默认"变体）。
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

/// `multipart` 的 `when`（`select` 由方块状态给）：
/// - 字符串：`性质=值` 用 `,` 连（AND）；
/// - `{"性质": "值"}`：全部成立的 AND（MC 的 `StatePropertiesPredicate` 就是这个形状）；
/// - `{"OR": [ ... ]}`：数组元素再按上面两条判，任一成立即可（**数组元素是对象，不是字符串**）。
fn when_matches(w: &J, st: &BlockState) -> bool {
  match w {
    J::String(s) => cond_matches(s, st),
    J::Object(o) => {
      if let Some(alts) = o.get("OR").and_then(J::as_array) {
        return alts.iter().any(|x| when_matches(x, st));
      }
      if o.is_empty() {
        return true; // `"when": {}` = 无条件
      }
      o.iter().all(|(k, v)| {
        let want = match v {
          J::String(s) => s.clone(),
          other => other.to_string(),
        };
        // 值可以给候选（`"facing": "north|south"`）
        want.split('|').any(|alt| st.prop(k) == Some(alt))
      })
    }
    _ => false,
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn st(name: &str, props: &[(&str, &str)]) -> BlockState {
    BlockState {
      name: format!("minecraft:{name}"),
      props: props.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect(),
    }
  }

  #[test]
  fn variant_conditions_are_sets_with_alternatives() {
    let s = st("oak_stairs", &[("facing", "east"), ("half", "bottom"), ("shape", "straight")]);
    assert!(cond_matches("", &s), "空键 = 默认变体");
    assert!(cond_matches("facing=east", &s));
    assert!(cond_matches("facing=east,half=bottom", &s));
    assert!(cond_matches("facing=east|west,half=bottom", &s), "值可给候选");
    assert!(!cond_matches("facing=west", &s));
    assert!(!cond_matches("facing=east,half=top", &s));
    assert!(!cond_matches("facing", &s), "非 k=v 的键不匹配");
  }

  #[test]
  fn multipart_when_supports_or() {
    let s = st("oak_fence", &[("north", "true"), ("south", "false")]);
    let w: J = serde_json::json!({"OR": [{"north": "false"}, {"south": "true"}]});
    assert!(!when_matches(&w, &s));
    let w: J = serde_json::json!({"OR": [{"north": "true"}, {"south": "true"}]});
    assert!(when_matches(&w, &s), "OR 任一成立");
    assert!(when_matches(&J::String("north=true".into()), &s));
  }

  /// 缺省 uv 必须与 MC 的公式一致（三条容易写反的：down 的 v、north 的 u、east 的 u）
  #[test]
  fn default_uvs_match_vanilla_formula() {
    let (from, to) = ([0.0, 0.0, 0.0], [16.0, 16.0, 16.0]);
    assert_eq!(default_uv(1, from, to), [0.0, 0.0, 16.0, 16.0], "up：u←x, v←z");
    assert_eq!(default_uv(0, from, to), [0.0, 0.0, 16.0, 16.0], "down 整块时与 up 同");
    assert_eq!(default_uv(2, from, to), [0.0, 0.0, 16.0, 16.0], "north 整块时也是全幅");
    // 台阶的下半块：正面的 v 落在下半张（v 8..16），与 vanilla `slab` / `stairs` 的写法一致
    let (from, to) = ([0.0, 0.0, 0.0], [16.0, 8.0, 16.0]);
    assert_eq!(default_uv(2, from, to), [0.0, 8.0, 16.0, 16.0]);
    assert_eq!(default_uv(0, from, to), [0.0, 0.0, 16.0, 16.0], "down 只看 x/z");
  }

  #[test]
  fn degrees_normalize_to_quarters() {
    assert_eq!(deg(Some(&J::from(90))), 90);
    assert_eq!(deg(Some(&J::from(270))), 270);
    assert_eq!(deg(Some(&J::from(-90))), 270);
    assert_eq!(deg(Some(&J::from(360))), 0);
    assert_eq!(deg(None), 0);
  }

  #[test]
  fn texture_variables_chase_to_a_path() {
    let mut m = HashMap::new();
    m.insert("all".to_string(), "#side".to_string());
    m.insert("side".to_string(), "minecraft:block/stone".to_string());
    assert_eq!(resolve_var(&m, "#all").as_deref(), Some("block/stone"));
    assert_eq!(resolve_var(&m, "minecraft:block/dirt").as_deref(), Some("block/dirt"));
    assert_eq!(resolve_var(&m, "#看不到"), None);
    // 环：不会死循环
    let mut cyc = HashMap::new();
    cyc.insert("a".to_string(), "#b".to_string());
    cyc.insert("b".to_string(), "#a".to_string());
    assert_eq!(resolve_var(&cyc, "#a"), None);
  }
}
