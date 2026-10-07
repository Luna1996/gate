use std::collections::HashMap;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Mutex, RwLock};

use glam::IVec3;
use rayon::prelude::*;

use super::summary::SOLID_MIN_PERMILLE;
use super::voxel;
use super::world::{self, SECTION_VOLUME, SECTIONS_PER_CHUNK, World};
use gate_voxel::PaletteId;

const MAGIC: &[u8; 8] = b"GATELOD1";
const VERSION: u32 = 2;
pub const CELL: i32 = 16;
pub const FINE_CELL: i32 = 4;
const MAX_CELL_SECTIONS: usize = 64;
const FINE_PER_AXIS: i32 = CELL / FINE_CELL;
const FINE_TOTAL: usize = (FINE_PER_AXIS * FINE_PER_AXIS * FINE_PER_AXIS) as usize;
const FINE_AIR: u16 = u16::MAX;
const PROGRESS_COLS: usize = 4096;

const FORMAT_TAG: u64 = 4;

pub fn stamp(map_dir: &Path) -> u64 {
  let mut h = std::collections::hash_map::DefaultHasher::new();
  FORMAT_TAG.hash(&mut h);
  map_dir.hash(&mut h);
  if let Ok(m) = fs::metadata(map_dir.join("level.dat")) {
    m.len().hash(&mut h);
    if let Ok(t) = m.modified() {
      t.hash(&mut h);
    }
  }
  h.finish()
}

pub fn path_for(world_name: &str) -> PathBuf {
  gate_render::data_dir().join("lod").join(format!("{world_name}.lod"))
}

pub struct File {
  pub stamp: u64,
  pub col_min: (i32, i32),
  pub dims: (i32, i32),
  pub sec_y: i32,
  pub names: Vec<String>,
  mask: Vec<u16>,
  offs: Vec<u32>,
  body: Vec<u8>,
  fine_offs: Vec<u32>,
  fine_body: Vec<u8>,
}

impl File {
  fn col_off(&self, i: usize) -> usize {
    self.offs[i] as usize
  }

  pub fn heap_bytes(&self) -> usize {
    self.body.len()
      + self.fine_body.len()
      + self.mask.len() * 2
      + self.offs.len() * 4
      + self.fine_offs.len() * 4
      + self.names.iter().map(|n| n.len() + 24).sum::<usize>()
  }

  pub fn read(path: &Path) -> Result<Self, String> {
    let raw = fs::read(path).map_err(|e| format!("读 {} 失败：{e}", path.display()))?;
    let mut c = Cursor { b: &raw, i: 0 };
    if c.take(8)? != MAGIC.as_slice() {
      return Err("魔数不符".into());
    }
    let version = c.u32()?;
    if version != VERSION {
      return Err(format!("版本 {version} ≠ {VERSION}（布局变了，重新构建）"));
    }
    let stamp = c.u64()?;
    let col_min = (c.i32()?, c.i32()?);
    let dims = (c.i32()?, c.i32()?);
    let sec_y = c.i32()?;
    let name_count = c.u32()? as usize;
    let mut names = Vec::with_capacity(name_count);
    for _ in 0..name_count {
      let n = c.u32()? as usize;
      names
        .push(String::from_utf8(c.take(n)?.to_vec()).map_err(|e| format!("名字不是 UTF-8：{e}"))?);
    }
    if dims.0 <= 0 || dims.1 <= 0 || !(1..=SECTIONS_PER_CHUNK).contains(&sec_y) {
      return Err(format!("范围不合法：{}×{} 列、{sec_y} 节", dims.0, dims.1));
    }
    let cols = dims.0 as usize * dims.1 as usize;
    let mut mask = Vec::with_capacity(cols);
    for _ in 0..cols {
      mask.push(c.u16()?);
    }
    let mut offs = Vec::with_capacity(cols);
    for _ in 0..cols {
      offs.push(c.u32()?);
    }
    let body_len = c.u32()? as usize;
    let body = c.take(body_len)?.to_vec();
    let want: u32 = mask.iter().map(|m| m.count_ones()).sum();
    if body.len() != want as usize * 4 {
      return Err(format!("条目区 {} B ≠ 掩码推出来的 {} B", body.len(), want * 4));
    }
    let mut fine_offs = Vec::with_capacity(want as usize);
    for _ in 0..want {
      fine_offs.push(c.u32()?);
    }
    let fine_len = c.u32()? as usize;
    let fine_body = c.take(fine_len)?.to_vec();
    let mut at = 0usize;
    for (k, &off) in fine_offs.iter().enumerate() {
      if off as usize != at {
        return Err(format!("细格偏移 [{k}] = {off} ≠ 走到 {at}（区间不连续）"));
      }
      let mut cells = 0usize;
      while cells < FINE_TOTAL {
        if at + 3 > fine_body.len() {
          return Err("细格区截断".into());
        }
        let run = fine_body[at + 2] as usize;
        if run == 0 || cells + run > FINE_TOTAL {
          return Err(format!("细格 RLE 长度非法：{run}（已 {cells}/{FINE_TOTAL}）"));
        }
        cells += run;
        at += 3;
      }
    }
    if at != fine_body.len() {
      return Err(format!("细格区多出 {} B（掩码只推到 {at}）", fine_body.len() - at));
    }
    Ok(Self { stamp, col_min, dims, sec_y, names, mask, offs, body, fine_offs, fine_body })
  }

  pub fn write(&self, path: &Path) -> Result<usize, String> {
    let mut out: Vec<u8> = Vec::with_capacity(self.heap_bytes() + 64);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&self.stamp.to_le_bytes());
    out.extend_from_slice(&self.col_min.0.to_le_bytes());
    out.extend_from_slice(&self.col_min.1.to_le_bytes());
    out.extend_from_slice(&self.dims.0.to_le_bytes());
    out.extend_from_slice(&self.dims.1.to_le_bytes());
    out.extend_from_slice(&self.sec_y.to_le_bytes());
    out.extend_from_slice(&(self.names.len() as u32).to_le_bytes());
    for n in &self.names {
      out.extend_from_slice(&(n.len() as u32).to_le_bytes());
      out.extend_from_slice(n.as_bytes());
    }
    for m in &self.mask {
      out.extend_from_slice(&m.to_le_bytes());
    }
    for o in &self.offs {
      out.extend_from_slice(&o.to_le_bytes());
    }
    out.extend_from_slice(&(self.body.len() as u32).to_le_bytes());
    out.extend_from_slice(&self.body);
    for o in &self.fine_offs {
      out.extend_from_slice(&o.to_le_bytes());
    }
    out.extend_from_slice(&(self.fine_body.len() as u32).to_le_bytes());
    out.extend_from_slice(&self.fine_body);
    if let Some(dir) = path.parent()
      && let Err(e) = fs::create_dir_all(dir)
    {
      return Err(format!("建目录 {} 失败：{e}", dir.display()));
    }
    fs::write(path, &out).map_err(|e| format!("写 {} 失败：{e}", path.display()))?;
    Ok(out.len())
  }
}

struct Cursor<'a> {
  b: &'a [u8],
  i: usize,
}

impl<'a> Cursor<'a> {
  fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
    let end = self.i.checked_add(n).ok_or("长度溢出")?;
    if end > self.b.len() {
      return Err(format!("文件截断：要 {} B，只剩 {} B", n, self.b.len() - self.i));
    }
    let s = &self.b[self.i..end];
    self.i = end;
    Ok(s)
  }
  fn u16(&mut self) -> Result<u16, String> {
    Ok(u16::from_le_bytes(self.take(2)?.try_into().expect("刚好 2 B")))
  }
  fn u32(&mut self) -> Result<u32, String> {
    Ok(u32::from_le_bytes(self.take(4)?.try_into().expect("刚好 4 B")))
  }
  fn u64(&mut self) -> Result<u64, String> {
    Ok(u64::from_le_bytes(self.take(8)?.try_into().expect("刚好 8 B")))
  }
  fn i32(&mut self) -> Result<i32, String> {
    Ok(i32::from_le_bytes(self.take(4)?.try_into().expect("刚好 4 B")))
  }
}

pub struct View {
  f: File,
  rep: Vec<Option<PaletteId>>,
}

impl View {
  pub fn new(f: File, resolve: impl Fn(&str) -> Option<PaletteId>) -> Self {
    let rep = f.names.iter().map(|n| resolve(n)).collect();
    Self { f, rep }
  }

  pub fn heap_bytes(&self) -> usize {
    self.f.heap_bytes() + self.rep.len() * std::mem::size_of::<Option<PaletteId>>()
  }

  pub fn missing(&self) -> usize {
    self.rep.iter().filter(|r| r.is_none()).count()
  }

  pub fn names(&self) -> usize {
    self.f.names.len()
  }

  pub fn cols(&self) -> usize {
    self.f.dims.0 as usize * self.f.dims.1 as usize
  }

  pub fn sections(&self) -> u32 {
    self.f.mask.iter().map(|m| m.count_ones()).sum()
  }

  pub fn supports(&self, cell_blocks: i32) -> bool {
    cell_blocks == FINE_CELL || (cell_blocks >= CELL && cell_blocks % CELL == 0)
  }

  fn entry(&self, cx: i32, y: i32, cz: i32) -> Option<(usize, usize, usize)> {
    if !(0..self.f.sec_y).contains(&y) {
      return None;
    }
    let (ix, iz) = (cx - self.f.col_min.0, cz - self.f.col_min.1);
    if ix < 0 || iz < 0 || ix >= self.f.dims.0 || iz >= self.f.dims.1 {
      return None;
    }
    let col = (iz * self.f.dims.0 + ix) as usize;
    let mask = self.f.mask[col];
    if mask & (1 << y) == 0 {
      return None;
    }
    let k = (mask & ((1u16 << y) - 1)).count_ones() as usize;
    Some((col, k, self.f.col_off(col) / 4 + k))
  }

  fn section(&self, cx: i32, y: i32, cz: i32) -> Option<(PaletteId, u32)> {
    let (col, k, _) = self.entry(cx, y, cz)?;
    let off = self.f.col_off(col) + k * 4;
    let name = u16::from_le_bytes([self.f.body[off], self.f.body[off + 1]]) as usize;
    let solid = u16::from_le_bytes([self.f.body[off + 2], self.f.body[off + 3]]) as u32;
    self.rep.get(name).copied().flatten().map(|r| (r, solid))
  }

  fn fine(&self, k_global: usize, l: usize) -> Option<PaletteId> {
    let mut at = *self.f.fine_offs.get(k_global)? as usize;
    let mut seen = 0usize;
    loop {
      let name = u16::from_le_bytes([self.f.fine_body[at], self.f.fine_body[at + 1]]);
      let run = self.f.fine_body[at + 2] as usize;
      if l < seen + run {
        if name == FINE_AIR {
          return None;
        }
        return self.rep.get(name as usize).copied().flatten();
      }
      seen += run;
      at += 3;
    }
  }

  pub fn cell(&self, cell: IVec3, cell_blocks: i32) -> Option<PaletteId> {
    if cell_blocks == FINE_CELL {
      let sec = cell.div_euclid(IVec3::splat(CELL));
      let (_, _, k) = self.entry(sec.x, sec.y, sec.z)?;
      let l = cell.rem_euclid(IVec3::splat(CELL)) / FINE_CELL;
      let idx = (l.x + FINE_PER_AXIS * l.z + FINE_PER_AXIS * FINE_PER_AXIS * l.y) as usize;
      return self.fine(k, idx);
    }
    let n = cell_blocks / CELL;
    if n <= 0 {
      return None;
    }
    let base = cell.div_euclid(IVec3::splat(CELL));
    if n == 1 {
      let (rep, _) = self.section(base.x, base.y, base.z)?;
      return Some(rep);
    }
    let per_layer = (n * n) as usize;
    debug_assert!(
      (n * n * n) as usize <= MAX_CELL_SECTIONS,
      "一格的节数 {n}³ 超过栈缓冲上限 —— `FAR_SCALES` 是不是加了更大的级？"
    );
    let mut reps = [PaletteId::AIR; MAX_CELL_SECTIONS];
    let mut len = 0usize;
    for dy in 0..n {
      for dz in 0..n {
        for dx in 0..n {
          reps[len] =
            self.section(base.x + dx, base.y + dy, base.z + dz).map_or(PaletteId::AIR, |(r, _)| r);
          len += 1;
        }
      }
    }
    voxel::rep_of_surface(&reps[..len], per_layer)
  }
}

#[derive(Debug, Clone, Copy)]
pub struct BuildStats {
  pub cols: usize,
  pub sections: usize,
  pub bytes: usize,
  pub secs: f64,
}

pub fn build(
  world: &World,
  out: &Path,
  cancel: &AtomicBool,
  progress: impl Fn(usize, usize) + Sync,
) -> Result<(BuildStats, File), String> {
  let t0 = std::time::Instant::now();
  let (col_min, dims, regions) = region_bounds(world.dir())?;
  let cols = (dims.0 * dims.1) as usize;
  bevy::log::info!(
    "LOD 构建：{} 个 region、{}×{} 列（chunk {}..{}）→ {}",
    regions,
    dims.0,
    dims.1,
    col_min.0,
    col_min.0 + dims.0,
    out.display()
  );
  let names = Mutex::new(Interner::default());
  let done = AtomicUsize::new(0);
  let per_col: Vec<Option<Vec<SecCol>>> = (0..cols)
    .into_par_iter()
    .map(|i| {
      if cancel.load(Ordering::Relaxed) {
        return None;
      }
      let cx = col_min.0 + (i as i32) % dims.0;
      let cz = col_min.1 + (i as i32) / dims.0;
      let out = column(world, &names, cx, cz);
      let n = done.fetch_add(1, Ordering::Relaxed) + 1;
      if n.is_multiple_of(PROGRESS_COLS) {
        progress(n, cols);
      }
      out
    })
    .collect();
  if cancel.load(Ordering::Relaxed) {
    return Err("已取消".into());
  }
  let names = names.into_inner().unwrap_or_else(|e| e.into_inner());
  let mut mask = Vec::with_capacity(cols);
  let mut offs = Vec::with_capacity(cols);
  let mut body: Vec<u8> = Vec::new();
  let mut fine_offs: Vec<u32> = Vec::new();
  let mut fine_body: Vec<u8> = Vec::new();
  let mut sections = 0usize;
  for col in &per_col {
    let mut m = 0u16;
    if let Some(items) = col {
      for it in items {
        m |= 1 << it.y;
      }
    }
    mask.push(m);
    offs.push(body.len() as u32);
    if let Some(items) = col {
      for it in items {
        body.extend_from_slice(&it.rep.to_le_bytes());
        body.extend_from_slice(&it.solid.to_le_bytes());
        fine_offs.push(fine_body.len() as u32);
        for &(name, len) in &it.fine {
          fine_body.extend_from_slice(&name.to_le_bytes());
          fine_body.push(len);
        }
        sections += 1;
      }
    }
  }
  let f = File {
    stamp: stamp(world.dir()),
    col_min,
    dims,
    sec_y: SECTIONS_PER_CHUNK,
    names: names.list,
    mask,
    offs,
    body,
    fine_offs,
    fine_body,
  };
  let bytes = f.write(out)?;
  let stats = BuildStats { cols, sections, bytes, secs: t0.elapsed().as_secs_f64() };
  Ok((stats, f))
}

type RegionBounds = ((i32, i32), (i32, i32), usize);

fn region_bounds(dir: &Path) -> Result<RegionBounds, String> {
  let rdir = dir.join("region");
  let rd = fs::read_dir(&rdir).map_err(|e| format!("读 {} 失败：{e}", rdir.display()))?;
  let (mut lo, mut hi, mut n) = ((i32::MAX, i32::MAX), (i32::MIN, i32::MIN), 0usize);
  for e in rd.flatten() {
    let name = e.file_name().to_string_lossy().into_owned();
    let Some(t) = name.strip_prefix("r.").and_then(|s| s.strip_suffix(".mca")) else { continue };
    let mut it = t.split('.');
    let (Some(x), Some(z)) = (
      it.next().and_then(|s| s.parse::<i32>().ok()),
      it.next().and_then(|s| s.parse::<i32>().ok()),
    ) else {
      continue;
    };
    lo = (lo.0.min(x), lo.1.min(z));
    hi = (hi.0.max(x), hi.1.max(z));
    n += 1;
  }
  if n == 0 {
    return Err(format!("{} 里没有 r.X.Z.mca", rdir.display()));
  }
  let dims = ((hi.0 - lo.0 + 1) * world::REGION_CHUNKS, (hi.1 - lo.1 + 1) * world::REGION_CHUNKS);
  Ok(((lo.0 * world::REGION_CHUNKS, lo.1 * world::REGION_CHUNKS), dims, n))
}

struct SecCol {
  y: u8,
  rep: u16,
  solid: u16,
  fine: Vec<(u16, u8)>,
}

fn column(world: &World, names: &Mutex<Interner>, cx: i32, cz: i32) -> Option<Vec<SecCol>> {
  let chunk = world.chunk(cx, cz)?;
  let mut buf: Vec<u16> = Vec::new();
  let mut counts: Vec<u32> = Vec::new();
  let mut out: Vec<SecCol> = Vec::new();
  for y in 0..SECTIONS_PER_CHUNK {
    let Some(sec) = chunk.section(y).filter(|s| !s.is_empty_layer()) else { continue };
    sec.unpack_into(&mut buf);
    let ids: Vec<u16> = {
      let mut g = names.lock().unwrap_or_else(|e| e.into_inner());
      sec
        .palette
        .iter()
        .map(|st| if world::is_air(&st.name) { FINE_AIR } else { g.intern(&st.key()) })
        .collect()
    };
    let fine = fine_cells(&ids, &buf, &mut counts);
    let mut cells = [FINE_AIR; FINE_TOTAL];
    {
      let mut i = 0usize;
      for &(name, len) in &fine {
        for _ in 0..len {
          cells[i] = name;
          i += 1;
        }
      }
      debug_assert_eq!(i, FINE_TOTAL, "细格 RLE 的长度和必须恰好是 64");
    }
    let layer_len = (FINE_PER_AXIS * FINE_PER_AXIS) as usize;
    let mut rep = FINE_AIR;
    for ly in (0..FINE_PER_AXIS as usize).rev() {
      let layer = &cells[ly * layer_len..(ly + 1) * layer_len];
      let occupied = layer.iter().filter(|&&n| n != FINE_AIR).count();
      if occupied * 2 < layer_len {
        continue;
      }
      rep = majority_name(layer);
      break;
    }
    if rep == FINE_AIR {
      rep = majority_name(&cells);
    }
    let solid = buf
      .iter()
      .filter(|&&pi| sec.palette.get(pi as usize).is_some_and(|st| !world::is_air(&st.name)))
      .count() as u16;
    let fine_any = fine.iter().any(|&(n, _)| n != FINE_AIR);
    if !fine_any && solid as u32 * 1000 < SOLID_MIN_PERMILLE * SECTION_VOLUME as u32 {
      continue;
    }
    out.push(SecCol { y: y as u8, rep, solid, fine });
  }
  (!out.is_empty()).then_some(out)
}

fn majority(counts: &[u32]) -> Option<(u32, usize)> {
  let mut best: Option<(u32, usize)> = None;
  for (i, &c) in counts.iter().enumerate() {
    if c == 0 {
      continue;
    }
    if best.is_none_or(|(bc, bi)| c > bc || (c == bc && i < bi)) {
      best = Some((c, i));
    }
  }
  best
}

fn majority_name(cells: &[u16]) -> u16 {
  let (mut best, mut best_n) = (FINE_AIR, 0usize);
  for &n in cells.iter().filter(|&&n| n != FINE_AIR) {
    let c = cells.iter().filter(|&&m| m == n).count();
    if c > best_n || (c == best_n && n < best) {
      best_n = c;
      best = n;
    }
  }
  best
}

fn fine_cells(ids: &[u16], buf: &[u16], tally: &mut Vec<u32>) -> Vec<(u16, u8)> {
  let pi_at =
    |bx: i32, by: i32, bz: i32| -> usize { buf[(by * 256 + bz * 16 + bx) as usize] as usize };
  let solid_at = |pi: usize| ids.get(pi).is_some_and(|&id| id != FINE_AIR);
  let mut runs: Vec<(u16, u8)> = Vec::with_capacity(8);
  for gy in 0..FINE_PER_AXIS {
    for gz in 0..FINE_PER_AXIS {
      for gx in 0..FINE_PER_AXIS {
        let mut solid = 0u32;
        for y in 0..4 {
          for z in 0..4 {
            for x in 0..4 {
              if solid_at(pi_at(gx * 4 + x, gy * 4 + y, gz * 4 + z)) {
                solid += 1;
              }
            }
          }
        }
        let mut rep = FINE_AIR;
        if solid * 1000 >= SOLID_MIN_PERMILLE * 64 {
          let mut top: Option<i32> = None;
          for y in (0..4).rev() {
            let mut n = 0u32;
            for z in 0..4 {
              for x in 0..4 {
                if solid_at(pi_at(gx * 4 + x, gy * 4 + y, gz * 4 + z)) {
                  n += 1;
                }
              }
            }
            if n * 2 >= 16 {
              top = Some(y);
              break;
            }
          }
          tally.clear();
          tally.resize(ids.len(), 0);
          match top {
            Some(y) => {
              for z in 0..4 {
                for x in 0..4 {
                  let pi = pi_at(gx * 4 + x, gy * 4 + y, gz * 4 + z);
                  if solid_at(pi) {
                    tally[pi] += 1;
                  }
                }
              }
            }
            None => {
              for y in 0..4 {
                for z in 0..4 {
                  for x in 0..4 {
                    let pi = pi_at(gx * 4 + x, gy * 4 + y, gz * 4 + z);
                    if solid_at(pi) {
                      tally[pi] += 1;
                    }
                  }
                }
              }
            }
          }
          rep = majority(tally).map_or(FINE_AIR, |(_, bi)| ids[bi]);
        }
        match runs.last_mut() {
          Some((n, len)) if *n == rep && *len < u8::MAX => *len += 1,
          _ => runs.push((rep, 1)),
        }
      }
    }
  }
  debug_assert_eq!(runs.iter().map(|&(_, l)| l as usize).sum::<usize>(), FINE_TOTAL);
  runs
}

#[derive(Default)]
struct Interner {
  map: HashMap<String, u16>,
  list: Vec<String>,
}

impl Interner {
  fn intern(&mut self, key: &str) -> u16 {
    if let Some(&i) = self.map.get(key) {
      return i;
    }
    let i = self.list.len() as u16;
    self.list.push(key.to_string());
    self.map.insert(key.to_string(), i);
    i
  }
}

pub enum Unavailable {
  Missing,
  Bad(String),
  Stale,
}

impl Unavailable {
  pub fn reason(&self, path: &Path) -> String {
    let p = path.display();
    match self {
      Self::Missing => format!("无 {p}"),
      Self::Bad(e) => format!("{p} 不可用：{e}"),
      Self::Stale => format!("{p} 是别的地图 / 存档改过了 → 作废"),
    }
  }
}

pub fn load(path: &Path, want_stamp: u64) -> Result<File, Unavailable> {
  let f = match File::read(path) {
    Ok(f) => f,
    Err(_) if !path.exists() => return Err(Unavailable::Missing),
    Err(e) => return Err(Unavailable::Bad(e)),
  };
  if f.stamp != want_stamp {
    return Err(Unavailable::Stale);
  }
  Ok(f)
}

pub type Cell = std::sync::Arc<RwLock<Option<std::sync::Arc<View>>>>;
