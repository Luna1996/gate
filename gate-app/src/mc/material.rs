use std::collections::HashMap;
use std::sync::Mutex;

use gate_voxel::{PaletteEntry, PaletteId};

const SLOT_LIMIT: u16 = 65_000;

#[derive(Default)]
struct Inner {
  map: HashMap<PaletteEntry, u16>,
  log: Vec<(PaletteId, PaletteEntry)>,
  next: u16,
  quant: HashMap<[u8; 3], u16>,
  overflow: usize,
}

pub struct Pool {
  inner: Mutex<Inner>,
  bits: u8,
}

impl Default for Pool {
  fn default() -> Self {
    Self::new()
  }
}

impl Pool {
  pub fn new() -> Self {
    Self::quantized(5)
  }

  pub fn quantized(bits: u8) -> Self {
    Self { inner: Mutex::new(Inner { next: 1, ..Default::default() }), bits: bits.clamp(3, 8) }
  }

  pub fn intern(&self, e: PaletteEntry) -> PaletteId {
    let e = PaletteEntry { color: quant(e.color, self.bits), ..e };
    let mut g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(i) = g.map.get(&e) {
      return PaletteId(*i);
    }
    if g.next < SLOT_LIMIT {
      let id = g.next;
      g.next += 1;
      g.map.insert(e, id);
      g.log.push((PaletteId(id), e));
      return PaletteId(id);
    }
    let q = quantize(e.color);
    if let Some(i) = g.quant.get(&q) {
      return PaletteId(*i);
    }
    let best = g
      .log
      .iter()
      .min_by_key(|(_, o)| color_dist2(o.color, e.color))
      .map(|(i, _)| *i)
      .unwrap_or(PaletteId(1));
    g.quant.insert(q, best.0);
    g.overflow += 1;
    PaletteId(best.0)
  }

  pub fn log_from(&self, from: usize) -> Vec<(PaletteId, PaletteEntry)> {
    let g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
    if from >= g.log.len() {
      return Vec::new();
    }
    g.log[from..].to_vec()
  }

  pub fn len(&self) -> usize {
    self.inner.lock().unwrap_or_else(|e| e.into_inner()).log.len()
  }

  pub fn overflow_count(&self) -> usize {
    self.inner.lock().unwrap_or_else(|e| e.into_inner()).overflow
  }
}

fn quantize(c: [u8; 3]) -> [u8; 3] {
  [c[0] >> 5, c[1] >> 5, c[2] >> 5]
}

fn quant(c: [u8; 3], bits: u8) -> [u8; 3] {
  if bits >= 8 {
    return c;
  }
  let mask = 0xFFu8 << (8 - bits);
  [c[0] & mask, c[1] & mask, c[2] & mask]
}

fn color_dist2(a: [u8; 3], b: [u8; 3]) -> u32 {
  let d = |x: u8, y: u8| {
    let v = x as i32 - y as i32;
    (v * v) as u32
  };
  d(a[0], b[0]) + d(a[1], b[1]) + d(a[2], b[2])
}