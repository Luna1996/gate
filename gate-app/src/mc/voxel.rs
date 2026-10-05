use std::sync::Arc;

use gate_voxel::{PaletteEntry, PaletteFlags, PaletteId};

use super::assets::Assets;
use super::material::Pool;
use super::model::{self, Face, Rot, SubModel};
use super::world::BlockState;

pub const CELLS: usize = 4096;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mat {
  pub roughness: u8,
  pub emissive: u8,
  pub transmission: u8,
  pub metallic: u8,
}

impl Mat {
  fn entry(self, color: [u8; 3]) -> PaletteEntry {
    let mut flags = PaletteFlags::default();
    if self.transmission > 0 {
      flags = flags.union(PaletteFlags::TRANSMISSIVE);
    }
    PaletteEntry {
      color,
      roughness: self.roughness,
      emissive: self.emissive,
      transmission: self.transmission,
      flags,
      metallic: self.metallic,
    }
  }
}

#[derive(Debug, Clone)]
pub enum Fill {
  Brick { off: [i32; 3], extent: i32, id: PaletteId },
}

impl Fill {
  pub fn id(&self) -> PaletteId {
    let Fill::Brick { id, .. } = self;
    *id
  }
}

pub struct BlockPlan {
  pub rep: PaletteId,
  pub fills: Vec<Fill>,
}

pub fn cells_of(assets: &Assets, pool: &Pool, st: &BlockState) -> Option<Box<[PaletteId; CELLS]>> {
  let subs: Vec<SubModel> = model::resolve(assets, st)?;
  let mat = material_of(st);
  let tint = tint_of(st);
  let mut cells = [PaletteId::AIR; CELLS];
  for sub in &subs {
    let base = raster(&sub.elements, mat, tint, pool);
    let rot = match sub.rot {
      Rot { x: 0, y: 0 } => base,
      r => rotate(&base, r),
    };
    for i in 0..CELLS {
      if !rot[i].is_air() {
        cells[i] = rot[i];
      }
    }
  }
  (!cells.iter().all(|c| c.is_air())).then_some(Box::new(cells))
}

pub fn plan(assets: &Assets, pool: &Pool, st: &BlockState) -> Option<Arc<BlockPlan>> {
  let cells = cells_of(assets, pool, st)?;
  let rep = rep_of(&cells[..]).unwrap_or(PaletteId::AIR);
  Some(Arc::new(BlockPlan { rep, fills: fills_of(&cells) }))
}

#[inline]
pub fn idx(x: i32, y: i32, z: i32) -> usize {
  (y * 256 + z * 16 + x) as usize
}

fn raster(
  elements: &[model::Element],
  mat: Mat,
  tint: Option<[u8; 3]>,
  pool: &Pool,
) -> [PaletteId; CELLS] {
  let mut cells = [PaletteId::AIR; CELLS];
  for el in elements {
    let (lo, hi) = spans(el.from, el.to);
    if (0..3).any(|a| hi[a] <= lo[a]) {
      continue;
    }
    let avg = element_avg(el, tint).map(|c| pool.intern(mat.entry(c)));
    for y in lo[1]..hi[1] {
      for z in lo[2]..hi[2] {
        for x in lo[0]..hi[0] {
          let touching =
            [y == lo[1], y + 1 == hi[1], z == lo[2], z + 1 == hi[2], x == lo[0], x + 1 == hi[0]];
          let face =
            model::FACE_PRIORITY.iter().copied().find(|f| touching[*f] && el.faces[*f].is_some());
          let id = match face {
            Some(f) => {
              match sample_face(el.faces[f].as_ref().expect("已判非空"), f, [x, y, z], el, tint)
              {
                Some(c) => pool.intern(mat.entry(c)),
                None => continue,
              }
            }
            None => match avg {
              Some(id) => id,
              None => continue,
            },
          };
          cells[idx(x, y, z)] = id;
        }
      }
    }
  }
  cells
}

fn span3(v: [f32; 3]) -> [i32; 3] {
  std::array::from_fn(|i| v[i].round() as i32)
}

fn spans(from: [f32; 3], to: [f32; 3]) -> ([i32; 3], [i32; 3]) {
  let mut lo = span3(from);
  let mut hi = span3(to);
  for i in 0..3 {
    hi[i] = hi[i].max(lo[i] + 1);
    lo[i] = lo[i].clamp(0, 16);
    hi[i] = hi[i].clamp(0, 16);
  }
  (lo, hi)
}

fn sample_face(
  f: &Face,
  dir: usize,
  p: [i32; 3],
  el: &model::Element,
  tint: Option<[u8; 3]>,
) -> Option<[u8; 3]> {
  let tint = if f.tint.is_some() { tint } else { None };
  let (lo, hi) = spans(el.from, el.to);
  let (au, av) = match dir {
    0 | 1 => (0usize, 2usize),
    2 | 3 => (0, 1),
    _ => (2, 1),
  };
  let fu = frac(p[au], lo[au], hi[au]);
  let fv = frac(p[av], lo[av], hi[av]);
  let uv = f.uv;
  let (mut u, mut v) = (lerp(uv[0], uv[2], fu), lerp(uv[1], uv[3], fv));
  let (lu, hu) = (uv[0].min(uv[2]), uv[0].max(uv[2]));
  let (lv, hv) = (uv[1].min(uv[3]), uv[1].max(uv[3]));
  (u, v) = match f.rot {
    90 => (lu + (hv - v), lv + (u - lu)),
    180 => (hu - (u - lu), hv - (v - lv)),
    270 => (lu + (v - lv), lv + (hu - u)),
    _ => (u, v),
  };
  let texel = f.tex.sample(u.clamp(0.0, 15.999) / 16.0, v.clamp(0.0, 15.999) / 16.0);
  if texel[3] == 0 {
    return None;
  }
  let mut c = [texel[0], texel[1], texel[2]];
  if let Some(t) = tint {
    for i in 0..3 {
      c[i] = ((c[i] as u32 * t[i] as u32) / 255) as u8;
    }
  }
  Some(c)
}

fn frac(c: i32, lo: i32, hi: i32) -> f32 {
  let span = (hi - lo) as f32;
  if span <= 0.0 { 0.5 } else { ((c as f32 + 0.5) - lo as f32) / span }
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
  a + (b - a) * t
}

fn element_avg(el: &model::Element, tint: Option<[u8; 3]>) -> Option<[u8; 3]> {
  let (mut acc, mut n) = ([0u32; 3], 0u32);
  for f in el.faces.iter().flatten() {
    let mut c = f.tex.avg;
    if let (Some(t), true) = (tint, f.tint.is_some()) {
      for i in 0..3 {
        c[i] = ((c[i] as u32 * t[i] as u32) / 255) as u8;
      }
    }
    for i in 0..3 {
      acc[i] += c[i] as u32;
    }
    n += 1;
  }
  (n > 0).then(|| std::array::from_fn(|i| (acc[i] / n) as u8))
}

pub fn fills_of(cells: &[PaletteId; CELLS]) -> Vec<Fill> {
  let mut out = Vec::new();
  let mut buf = [PaletteId::AIR; 64];
  for by in 0..4 {
    for bz in 0..4 {
      for bx in 0..4 {
        let off = [bx * 4, by * 4, bz * 4];
        for z in 0..4 {
          for y in 0..4 {
            for x in 0..4 {
              buf[(z * 16 + y * 4 + x) as usize] = cells[idx(off[0] + x, off[1] + y, off[2] + z)];
            }
          }
        }
        if let Some(id) = rep_of(&buf) {
          out.push(Fill::Brick { off, extent: 4, id });
        }
      }
    }
  }
  if out.len() == 64
    && out.iter().all(|f| matches!(f, Fill::Brick { id, .. } if *id == out[0].id()))
  {
    return vec![Fill::Brick { off: [0, 0, 0], extent: 16, id: out[0].id() }];
  }
  out
}

pub fn rep_of(cells: &[PaletteId]) -> Option<PaletteId> {
  let mut tally: Vec<(PaletteId, u32)> = Vec::with_capacity(8);
  for c in cells.iter().filter(|c| !c.is_air()) {
    match tally.iter_mut().find(|(id, _)| id == c) {
      Some(slot) => slot.1 += 1,
      None => tally.push((*c, 1)),
    }
  }
  tally.into_iter().max_by_key(|(id, n)| (*n, std::cmp::Reverse(id.0))).map(|(id, _)| id)
}

pub fn rep_of_surface(cells: &[PaletteId], per_layer: usize) -> Option<PaletteId> {
  if per_layer == 0 {
    return None;
  }
  debug_assert_eq!(cells.len() % per_layer, 0, "每层方块数必须整除总格数");
  for y in (0..cells.len() / per_layer).rev() {
    let layer = &cells[y * per_layer..(y + 1) * per_layer];
    let solid = layer.iter().filter(|c| !c.is_air()).count();
    if solid * 2 >= per_layer {
      if let Some(r) = rep_of(layer) {
        return Some(r);
      }
    }
  }
  rep_of(cells)
}

fn rotate(src: &[PaletteId; CELLS], r: Rot) -> [PaletteId; CELLS] {
  let step = |s: &[PaletteId; CELLS], f: fn(i32, i32, i32) -> (i32, i32, i32)| {
    let mut out = [PaletteId::AIR; CELLS];
    for y in 0..16 {
      for z in 0..16 {
        for x in 0..16 {
          let (a, b, c) = f(x, y, z);
          out[idx(a, b, c)] = s[idx(x, y, z)];
        }
      }
    }
    out
  };
  let t = match r.x {
    90 => step(src, |x, y, z| (x, 15 - z, y)),
    180 => step(src, |x, y, z| (x, 15 - y, 15 - z)),
    270 => step(src, |x, y, z| (x, z, 15 - y)),
    _ => *src,
  };
  match r.y {
    90 => step(&t, |x, y, z| (15 - z, y, x)),
    180 => step(&t, |x, y, z| (15 - x, y, 15 - z)),
    270 => step(&t, |x, y, z| (z, y, 15 - x)),
    _ => t,
  }
}

pub fn material_of(st: &BlockState) -> Mat {
  let n = st.short_name();
  let lit = st.prop("lit").map(|v| v == "true").unwrap_or(false);
  let mut m = Mat { roughness: 235, emissive: 0, transmission: 0, metallic: 0 };
  if is_light(n, lit) {
    m.emissive = 255;
    m.roughness = 200;
  }
  if n.contains("glass") || n.ends_with("_pane") {
    m.transmission = 190;
    m.roughness = 40;
  }
  if n.contains("ice") {
    m.transmission = 120;
    m.roughness = 70;
  }
  if matches!(n, "water" | "flowing_water" | "bubble_column") {
    m.transmission = 190;
    m.roughness = 20;
  }
  if matches!(n, "honey_block" | "slime_block") {
    m.transmission = 150;
    m.roughness = 40;
  }
  if matches!(n, "lava" | "flowing_lava") {
    m.emissive = 255;
  }
  if is_metal(n) {
    m.metallic = 255;
    m.roughness = 70;
  }
  m
}

fn is_light(n: &str, lit: bool) -> bool {
  const ALWAYS: [&str; 20] = [
    "glowstone",
    "sea_lantern",
    "shroomlight",
    "torch",
    "lantern",
    "campfire",
    "end_rod",
    "beacon",
    "conduit",
    "magma_block",
    "froglight",
    "lava",
    "fire",
    "crying_obsidian",
    "respawn_anchor",
    "jack_o_lantern",
    "amethyst_cluster",
    "glow_lichen",
    "sea_pickle",
    "enchanting_table",
  ];
  if ALWAYS.iter().any(|k| n.contains(k)) {
    return true;
  }
  lit && matches!(n, "redstone_lamp" | "redstone_torch" | "candle")
}

fn is_metal(n: &str) -> bool {
  if n.ends_with("_ore") || n.contains("raw_") {
    return true;
  }
  const WORDS: [&str; 6] = ["iron", "gold", "copper", "netherite", "anvil", "lodestone"];
  if WORDS.iter().any(|k| n.contains(k)) {
    return true;
  }
  matches!(n, "cauldron" | "hopper" | "chain" | "blast_furnace" | "observer" | "piston")
    || n.ends_with("_bars")
}

pub fn tint_of(st: &BlockState) -> Option<[u8; 3]> {
  let n = st.short_name();
  if matches!(n, "water" | "flowing_water" | "bubble_column") {
    return Some([63, 118, 228]);
  }
  if n == "spruce_leaves" {
    return Some([97, 153, 97]);
  }
  if n == "birch_leaves" {
    return Some([128, 167, 85]);
  }
  if n.ends_with("_leaves") || n == "lily_pad" || n == "vine" {
    return Some([119, 171, 47]);
  }
  if n == "grass_block"
    || n == "grass"
    || n == "tall_grass"
    || n == "fern"
    || n == "large_fern"
    || n == "sugar_cane"
    || n.ends_with("_grass")
  {
    return Some([145, 189, 89]);
  }
  None
}