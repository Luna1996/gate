//! **元素盒 → 16³ 体素色**，以及"怎么把这些色写进 chunk 树"的**写入计划**。
//!
//! 对齐关系（`docs/mc_map.md` §1）：1 方块 = 16³ 体素，而元素盒的坐标单位正是 1/16 方块 ⇒
//! **一个元素盒角点就落在一个体素边界上，不需要重采样**：`from`/`to` 直接当体素下标用。
//!
//! 逐 texel 上色的规则：
//! - 每个格取**离它最近的那个面**的 texel（贴不到任何面 = 元素内部 ⇒ 取该元素的均色，
//!   内部不可见，取均色还能让"整块同色"的方块在树里大量合并）；
//! - 同时贴着两个面时按 [`model::FACE_PRIORITY`] 取（上 > 下 > 西 > 东 > 北 > 南）；
//! - texel 的 α = 0 ⇒ **不写**（挖空：树叶的洞、玻璃中央）—— 是"跳过"而不是"写空气"，
//!   免得后面的元素把前面元素已经填好的格擦掉；
//! - 面的 uv 由 [`model::default_uv`] 或模型里写的 `uv` 给，按格心比例插值到 texel，再按
//!   `rotation` 在面内转（0/90/180/270）；
//! - `tintindex` 存在时按方块名取固定生物群系色（[`tint_of`]）—— 本仓不读 `Biomes`。
//!
//! **块级旋转**（`x`/`y`）在**体素层面**整体转：先按未旋转的模型空间出 texel，再旋转 16³ 数组
//! （90 的倍数 ⇒ 格到格，无插值）。这样 uv 就不必跟着旋转去换算。
//!
//! **写入计划** = 把 16³ 的色阵列按 `Detail` 摊成"对齐 brick 的写操作"：
//! `Full` → 逐 texel（`4³` 等值砖走 [`gate_voxel::VolumeGrid::fill_brick`]，混合砖走
//! [`gate_voxel::VolumeGrid::set_brick_cells`]，两者都是"一次下钻写一整砖"）；
//! `Fine` → 每 `4³` 砖一个代表色；`Coarse` 及以上 → 整块一个代表色（由调用方按卷聚合）。

use std::sync::Arc;

use gate_voxel::{PaletteEntry, PaletteFlags, PaletteId};

use super::assets::Assets;
use super::material::Pool;
use super::model::{self, Face, Rot, SubModel};
use super::world::BlockState;

/// 一个方块的体素数（16³）
pub const CELLS: usize = 4096;

/// 材质的物理参数（按方块名推；见 [`material_of`]）
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

/// 一次 brick 级写入（恒为等值砖：`4³` 或整块 `16`）
#[derive(Debug, Clone)]
pub enum Fill {
  /// `extent³` 的等值砖（`extent` ∈ {4, 16}）
  Brick { off: [i32; 3], extent: i32, id: PaletteId },
}

impl Fill {
  /// 这一条写的槽号
  pub fn id(&self) -> PaletteId {
    let Fill::Brick { id, .. } = self;
    *id
  }
}

/// 一个方块状态的**写入计划**（按状态缓存；worker 只读）。
///
/// 只有一档粒度：每个 `4³` 砖一个代表色（8 cm）。**逐 texel 那一档实测不可用** —— 见
/// [`super::source`] 里 `Detail::Full` 的说明（一层满实体 section 的 CPU 树 20 MB、78 ms）。
pub struct BlockPlan {
  /// 代表色 = 非空气格里最多的那个（粗档与将来的远场用）
  pub rep: PaletteId,
  /// 写入列表（`4³` 砖，全是等值砖）
  pub fills: Vec<Fill>,
}

/// 逐 texel 的 16³ 槽号（已按块级旋转摆正）。`None` = 这个方块渲染不出东西。
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
    // 只覆盖非空气：后一个元素（如草方块顶面的 overlay）的透明 texel 不该擦掉下面的土
    for i in 0..CELLS {
      if !rot[i].is_air() {
        cells[i] = rot[i];
      }
    }
  }
  (!cells.iter().all(|c| c.is_air())).then_some(Box::new(cells))
}

/// 出计划（`None` = 这个方块不用画，例如空气 / 缺方块状态的未知方块）。
///
/// 逐 texel 的 16³ 阵列**只在建计划时算一次**（按方块状态缓存），落成的写操作是 8 cm 档
/// ⇒ 一个方块的产出成本与"它的贴图有几个色"无关，只与"有几个砖是混合的"有关。
pub fn plan(assets: &Assets, pool: &Pool, st: &BlockState) -> Option<Arc<BlockPlan>> {
  let cells = cells_of(assets, pool, st)?;
  let rep = rep_of(&cells[..]).unwrap_or(PaletteId::AIR);
  Some(Arc::new(BlockPlan { rep, fills: fills_of(&cells) }))
}

/// 反解一份计划回 16³ 色阵列（测试用：验证"计划 ≡ 逐砖多数色"）
#[cfg(test)]
fn rebuild(fills: &[Fill]) -> [PaletteId; CELLS] {
  let mut out = [PaletteId::AIR; CELLS];
  for f in fills {
    let Fill::Brick { off, extent, id } = f;
    for y in 0..*extent {
      for z in 0..*extent {
        for x in 0..*extent {
          out[idx(off[0] + x, off[1] + y, off[2] + z)] = *id;
        }
      }
    }
  }
  out
}

/// 16³ 阵列的线性下标（**x 最密**：`y*256 + z*16 + x`，与区块 NBT 的 `BlockStates` 同序）
#[inline]
pub fn idx(x: i32, y: i32, z: i32) -> usize {
  (y * 256 + z * 16 + x) as usize
}

/// 一个元素盒在**未旋转**的模型空间里出 16³ 槽号
fn raster(
  elements: &[model::Element],
  mat: Mat,
  tint: Option<[u8; 3]>,
  pool: &Pool,
) -> [PaletteId; CELLS] {
  let mut cells = [PaletteId::AIR; CELLS];
  for el in elements {
    // 元素盒可能越出 [0,16]（挂牌、藤蔓、火焰这类模型的角点会超出方块范围）⇒ 裁到方块内
    let (lo, hi) = spans(el.from, el.to);
    if (0..3).any(|a| hi[a] <= lo[a]) {
      continue; // 与方块无交
    }
    // 元素内部（贴不到任何面）的色：各面贴图均色的平均
    let avg = element_avg(el, tint).map(|c| pool.intern(mat.entry(c)));
    for y in lo[1]..hi[1] {
      for z in lo[2]..hi[2] {
        for x in lo[0]..hi[0] {
          let touching = [
            y == lo[1],
            y + 1 == hi[1],
            z == lo[2],
            z + 1 == hi[2],
            x == lo[0],
            x + 1 == hi[0],
          ];
          let face = model::FACE_PRIORITY
            .iter()
            .copied()
            .find(|f| touching[*f] && el.faces[*f].is_some());
          let id = match face {
            Some(f) => match sample_face(el.faces[f].as_ref().expect("已判非空"), f, [x, y, z], el, tint) {
              Some(c) => pool.intern(mat.entry(c)),
              None => continue, // texel 全透明 ⇒ 挖空
            },
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

/// `f32` 角点 → 体素区间 `[lo, hi)`（贴边界；零厚度退化成 1 格厚）
fn span3(v: [f32; 3]) -> [i32; 3] {
  std::array::from_fn(|i| v[i].round() as i32)
}

/// `from`/`to` → 逐轴区间（零厚度补成 1 格）
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

/// 取一个格在某个面上的 texel 色（`None` = 全透明）。
/// **染色只在面带 `tintindex` 时生效**（一个方块的某些面染色、某些面不染是 MC 的正常写法）。
fn sample_face(
  f: &Face,
  dir: usize,
  p: [i32; 3],
  el: &model::Element,
  tint: Option<[u8; 3]>,
) -> Option<[u8; 3]> {
  let tint = if f.tint.is_some() { tint } else { None };
  let (lo, hi) = spans(el.from, el.to);
  // 面在哪个轴上取 u / v（与 `model::default_uv` 的公式配套）
  let (au, av) = match dir {
    0 | 1 => (0usize, 2usize), // y 法线：u←x, v←z
    2 | 3 => (0, 1),           // z 法线：u←x, v←y
    _ => (2, 1),               // x 法线：u←z, v←y
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

/// 格心在该轴上的位置比例（0..1）
fn frac(c: i32, lo: i32, hi: i32) -> f32 {
  let span = (hi - lo) as f32;
  if span <= 0.0 { 0.5 } else { ((c as f32 + 0.5) - lo as f32) / span }
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
  a + (b - a) * t
}

/// 元素内部色 = 各面贴图均色的平均（带染色的面才染色）
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

/// 把一个方块的 16³ 色阵列摊成 `4³` 砖上的写操作：**每砖取非空气里最多的那个色**。
///
/// 这就是"8 cm 档"：粗到 4 个体素（8 cm）之后，砖内本来也分不出更细的结构；代价从"每砖一张值表"
/// 降到"每砖一个节点"（能并进父层）。逐 texel 版见 `docs/mc_map.md` §6 的实测记录。
///
/// 收尾的**收缩**很关键：64 个砖同色（量化后石/陶土/混凝土都是这样）⇒ 一条整块 `extent 16` 写，
/// 产出从 64 次下钻降到 1 次。
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
  if out.len() == 64 && out.iter().all(|f| matches!(f, Fill::Brick { id, .. } if *id == out[0].id())) {
    return vec![Fill::Brick { off: [0, 0, 0], extent: 16, id: out[0].id() }];
  }
  out
}

/// 非空气格里出现最多的色（全空气 → `None`）
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

/// 体素层面的块级旋转（`x` 先、`y` 后；见 `docs/mc_map.md` §5）
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

/// 按方块名推物理参数（`docs/mc_map.md` §5 的启发式表）。
/// 不读任何方块表：名字里能看出的类别（自发光 / 玻璃 / 金属 / 冰）各给一组固定值，其余按粗糙石材。
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
  // 只有点亮时才自发光（关着的红石灯 / 蜡烛 / 红石火把是不发光的）
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

/// 生物群系染色（`docs/mc_map.md` §6：**不读 `Biomes`**，取 MC 的默认平原色）。
/// `tintindex` 只用来判"这个面要不要染色"，具体色只按方块名给。
pub fn tint_of(st: &BlockState) -> Option<[u8; 3]> {
  let n = st.short_name();
  if matches!(n, "water" | "flowing_water" | "bubble_column") {
    return Some([63, 118, 228]); // MC 的默认水色
  }
  if n == "spruce_leaves" {
    return Some([97, 153, 97]);
  }
  if n == "birch_leaves" {
    return Some([128, 167, 85]);
  }
  if n.ends_with("_leaves") || n == "lily_pad" || n == "vine" {
    return Some([119, 171, 47]); // 树叶（默认平原）
  }
  if n == "grass_block" || n == "grass" || n == "tall_grass" || n == "fern" || n == "large_fern"
    || n == "sugar_cane" || n.ends_with("_grass")
  {
    return Some([145, 189, 89]); // 草（默认平原）
  }
  None
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::mc::assets::Texture;
  use crate::mc::model::{Element, FACE_NAMES};

  fn solid_tex(c: [u8; 4]) -> Arc<Texture> {
    let mut rgba = Vec::with_capacity(16 * 16 * 4);
    for _ in 0..16 * 16 {
      rgba.extend_from_slice(&c);
    }
    Arc::new(Texture::new(16, 16, rgba))
  }

  /// 整块六面同色 ⇒ 16³ 全同槽，且计划**收缩成一条整块写**
  #[test]
  fn uniform_cube_is_a_single_block_write() {
    let tex = solid_tex([200, 30, 40, 255]);
    let el = Element {
      from: [0.0; 3],
      to: [16.0; 3],
      faces: std::array::from_fn(|_| {
        Some(Face { tex: tex.clone(), uv: [0.0, 0.0, 16.0, 16.0], rot: 0, tint: None })
      }),
    };
    let pool = Pool::new();
    let cells = raster(&[el], Mat { roughness: 235, emissive: 0, transmission: 0, metallic: 0 }, None, &pool);
    assert!(cells.iter().all(|c| *c == cells[0] && !c.is_air()), "整块应同色");
    assert_eq!(cells[idx(7, 7, 7)], cells[idx(0, 15, 0)], "表面与内部同色（都取自同一张贴图）");
    let fills = fills_of(&cells);
    assert_eq!(fills.len(), 1, "64 个砖同色 ⇒ 收缩成一条整块写：{fills:?}");
    assert!(matches!(fills[0], Fill::Brick { off: [0, 0, 0], extent: 16, .. }));
    assert_eq!(rebuild(&fills), cells);
  }

  /// **8 cm 档的语义**：每个 `4³` 砖整砖一个色（非空气里最多的），且只在该砖有实体时才写
  #[test]
  fn per_brick_plan_takes_the_majority_color() {
    let a = solid_tex([200, 30, 40, 255]);
    let b = solid_tex([10, 220, 30, 255]);
    // 下 6 格 a、上 10 格 b：分界 6 不是 4 的倍数 ⇒ by=1 那层砖是混合砖
    let mk = |y0: f32, y1: f32, tex: Arc<Texture>| Element {
      from: [0.0, y0, 0.0],
      to: [16.0, y1, 16.0],
      faces: std::array::from_fn(|_| {
        Some(Face { tex: tex.clone(), uv: [0.0, 0.0, 16.0, 16.0], rot: 0, tint: None })
      }),
    };
    let pool = Pool::new();
    let cells = raster(
      &[mk(0.0, 6.0, a), mk(6.0, 16.0, b)],
      Mat { roughness: 235, emissive: 0, transmission: 0, metallic: 0 },
      None,
      &pool,
    );
    let fills = fills_of(&cells);
    assert_eq!(fills.len(), 64, "16³ = 64 个 4³ 砖，每砖一个色");
    let bc = rebuild(&fills);
    for by in 0..4 {
      for bz in 0..4 {
        for bx in 0..4 {
          let c0 = bc[idx(bx * 4, by * 4, bz * 4)];
          for z in 0..4 {
            for y in 0..4 {
              for x in 0..4 {
                assert_eq!(bc[idx(bx * 4 + x, by * 4 + y, bz * 4 + z)], c0, "每砖内必须同色");
              }
            }
          }
        }
      }
    }
    // y=4..8 那层砖 32 个 a / 32 个 b（平手）⇒ 取先认领的那个（a）
    let (a, b) = (cells[idx(0, 0, 0)], cells[idx(0, 15, 0)]);
    assert_ne!(a, b);
    assert_eq!(bc[idx(0, 4, 0)], a, "平手时取槽号小者");
    assert_eq!(bc[idx(0, 12, 0)], b, "上半砖是 b");
    // 半空块的砖不该被填实：只画下半 ⇒ 上面 8 格的砖一个都不写
    let half = raster(
      &[mk(0.0, 6.0, solid_tex([1, 2, 3, 255]))],
      Mat { roughness: 235, emissive: 0, transmission: 0, metallic: 0 },
      None,
      &pool,
    );
    let hf = fills_of(&half);
    assert_eq!(hf.len(), 32, "只写 y<8 的 32 个砖");
    assert!(rebuild(&hf)[idx(8, 12, 8)].is_air(), "上半应留空");
  }

  /// 透明 texel 挖空
  #[test]
  fn cutout_texels_are_holes() {
    let mut rgba = vec![0u8; 16 * 16 * 4];
    for y in 0..16 {
      for x in 0..8 {
        let o = ((y * 16 + x) * 4) as usize;
        rgba[o..o + 4].copy_from_slice(&[1, 2, 3, 255]);
      }
    }
    let tex = Arc::new(Texture::new(16, 16, rgba));
    let el = Element {
      from: [0.0; 3],
      to: [16.0; 3],
      faces: std::array::from_fn(|_| {
        Some(Face { tex: tex.clone(), uv: [0.0, 0.0, 16.0, 16.0], rot: 0, tint: None })
      }),
    };
    let pool = Pool::new();
    let m = Mat { roughness: 235, emissive: 0, transmission: 0, metallic: 0 };
    let cells = raster(&[el], m, None, &pool);
    // up 面：u←x ⇒ 左半边（x<8）实体、右半边（x≥8）全透明 ⇒ 洞
    assert!(!cells[idx(2, 15, 8)].is_air(), "顶面左半边实体");
    assert!(cells[idx(14, 15, 8)].is_air(), "顶面右半边（α=0）应挖空");
    // west 面：u←z ⇒ 洞落在 z≥8 那一半
    assert!(!cells[idx(0, 8, 2)].is_air(), "西面 z<8 实体");
    assert!(cells[idx(0, 8, 14)].is_air(), "西面 z≥8 应挖空");
  }

  /// 体素层面的块级旋转：90° 绕 Y 把 +X 面转到 +Z（与 `oak_log` 的 `axis=x` 参照同源）
  #[test]
  fn rotations_are_bijective_quarter_turns() {
    let mut src = [PaletteId::AIR; CELLS];
    src[idx(15, 3, 4)] = PaletteId(7); // 东面某格
    let r = rotate(&src, Rot { x: 0, y: 90 });
    assert_eq!(r[idx(11, 3, 15)], PaletteId(7), "y=90：(x,z)→(15-z,x)");
    assert_eq!(r.iter().filter(|c| !c.is_air()).count(), 1, "旋转不得增删实体");
    // 转四次回到原样
    let mut t = src;
    for _ in 0..4 {
      t = rotate(&t, Rot { x: 0, y: 90 });
    }
    assert_eq!(t, src, "四个 90° 应复原");
    // x 与 y 都转：先 x 后 y（顺序影响结果，钉住它）
    let a = rotate(&src, Rot { x: 90, y: 90 });
    let b = rotate(&rotate(&src, Rot { x: 90, y: 0 }), Rot { x: 0, y: 90 });
    assert_eq!(a, b, "必须是先 x 后 y");
  }

  /// `FACE_PRIORITY` 必须覆盖全部 6 个面且不重复
  #[test]
  fn face_priority_is_a_permutation() {
    let mut s = model::FACE_PRIORITY.to_vec();
    s.sort_unstable();
    assert_eq!(s, (0..6).collect::<Vec<_>>());
    assert_eq!(FACE_NAMES.len(), 6);
  }

  /// 材质启发式：自发光 / 玻璃 / 金属 / 冰各挡住一类，普通石块不动
  #[test]
  fn material_heuristics_by_name() {
    let st = |n: &str, props: &[(&str, &str)]| BlockState {
      name: format!("minecraft:{n}"),
      props: props.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect(),
    };
    assert_eq!(material_of(&st("glowstone", &[])).emissive, 255);
    assert_eq!(material_of(&st("redstone_lamp", &[("lit", "false")])).emissive, 0, "没点亮不发光");
    assert_eq!(material_of(&st("redstone_lamp", &[("lit", "true")])).emissive, 255);
    assert_eq!(material_of(&st("light_gray_stained_glass", &[])).transmission, 190);
    assert_eq!(material_of(&st("glass_pane", &[])).transmission, 190, "玻璃板也算玻璃");
    assert_eq!(material_of(&st("iron_block", &[])).metallic, 255);
    assert_eq!(material_of(&st("coal_ore", &[])).metallic, 255);
    assert_eq!(material_of(&st("blue_ice", &[])).transmission, 120);
    let plain = material_of(&st("white_terracotta", &[]));
    assert_eq!(plain, Mat { roughness: 235, emissive: 0, transmission: 0, metallic: 0 });
  }

  /// 染色：草 / 树叶 / 水各一色，石头不染
  #[test]
  fn tints_are_fixed_biome_defaults() {
    let st = |n: &str| BlockState { name: format!("minecraft:{n}"), props: Default::default() };
    assert_eq!(tint_of(&st("grass_block")), Some([145, 189, 89]));
    assert_eq!(tint_of(&st("oak_leaves")), Some([119, 171, 47]));
    assert_eq!(tint_of(&st("water")), Some([63, 118, 228]));
    assert_eq!(tint_of(&st("stone")), None);
  }
}
