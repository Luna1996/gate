use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

pub const PALETTE_ENTRY_COUNT: usize = 65_536;

pub const PALETTE_BITS: u32 = 16;

pub const PALETTE_INDEX_MAX: u16 = (PALETTE_ENTRY_COUNT - 1) as u16;

const OCCUPIED_WORDS: usize = PALETTE_ENTRY_COUNT / 64;

const _: () = assert!(OCCUPIED_WORDS * 64 == PALETTE_ENTRY_COUNT);

const _: () = assert!(PALETTE_ENTRY_COUNT == 1usize << PALETTE_BITS);
const _: () = assert!(PALETTE_ENTRY_COUNT == PALETTE_INDEX_MAX as usize + 1);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(transparent)]
pub struct PaletteId(pub u16);

impl PaletteId {
  pub const AIR: Self = Self(0);

  #[inline]
  pub fn get(self) -> u16 {
    self.0
  }

  #[inline]
  pub fn is_air(self) -> bool {
    self.0 == 0
  }
}

impl From<u16> for PaletteId {
  fn from(v: u16) -> Self {
    Self(v)
  }
}

impl From<u8> for PaletteId {
  fn from(v: u8) -> Self {
    Self(v as u16)
  }
}

impl From<i32> for PaletteId {
  fn from(v: i32) -> Self {
    assert!(
      (0..=PALETTE_INDEX_MAX as i32).contains(&v),
      "调色板索引越界：{v}（合法范围 0..={PALETTE_INDEX_MAX}）"
    );
    Self(v as u16)
  }
}

impl std::fmt::Display for PaletteId {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "{}", self.0)
  }
}

impl From<PaletteId> for u16 {
  fn from(v: PaletteId) -> Self {
    v.0
  }
}

impl From<PaletteId> for u32 {
  fn from(v: PaletteId) -> Self {
    v.0 as u32
  }
}

impl From<PaletteId> for usize {
  fn from(v: PaletteId) -> Self {
    v.0 as usize
  }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct PaletteEntry {
  pub color: [u8; 3],
  pub roughness: u8,
  pub emissive: u8,
  pub transmission: u8,
  pub flags: PaletteFlags,
  pub metallic: u8,
}

impl Default for PaletteEntry {
  fn default() -> Self {
    Self {
      color: [0, 0, 0],
      roughness: 255,
      emissive: 0,
      transmission: 0,
      flags: PaletteFlags::default(),
      metallic: 0,
    }
  }
}

impl PaletteEntry {
  pub fn pbr(asset: u16, ov: PbrOverrides, flags: PaletteFlags) -> Self {
    Self {
      color: [ov.roughness, ov.metallic, ov.emissive],
      roughness: ov.transmission,
      emissive: (asset & 0xFF) as u8,
      transmission: (asset >> 8) as u8,
      flags: flags.union(PaletteFlags::IS_PBR),
      metallic: ov.specular,
    }
  }

  #[inline]
  pub fn pbr_asset(&self) -> u16 {
    (self.emissive as u16) | ((self.transmission as u16) << 8)
  }

  #[inline]
  pub fn pbr_overrides(&self) -> PbrOverrides {
    PbrOverrides {
      roughness: self.color[0],
      metallic: self.color[1],
      emissive: self.color[2],
      transmission: self.roughness,
      specular: self.metallic,
    }
  }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PbrOverrides {
  pub roughness: u8,
  pub metallic: u8,
  pub emissive: u8,
  pub transmission: u8,
  pub specular: u8,
}

pub fn override_byte(value: f32) -> u8 {
  (1.0 + (value.clamp(0.0, 1.0) * 254.0).round()) as u8
}

pub fn override_value(byte: u8) -> Option<f32> {
  if byte == 0 { None } else { Some(f32::from(byte - 1) / 254.0) }
}

pub fn slider_to_override(value: f32, min: f32, max: f32) -> u8 {
  let span = max - min;
  if span <= 0.0 || value <= min {
    return 0;
  }
  override_byte((value - min) / span)
}

pub fn inverted_pct_to_override(pct: f32) -> u8 {
  if pct <= 0.0 {
    return 0;
  }
  override_byte((100.0 - pct.clamp(0.0, 100.0)) / 100.0)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct PaletteFlags(pub u8);

impl PaletteFlags {
  pub const LOCKED: Self = Self(1 << 0);
  pub const INPUT_PORT: Self = Self(1 << 1);
  pub const OUTPUT_PORT: Self = Self(1 << 2);
  pub const HOLOGRAM: Self = Self(1 << 3);
  pub const IS_PBR: Self = Self(1 << 4);
  pub const TRANSMISSIVE: Self = Self(1 << 5);

  pub fn contains(self, other: Self) -> bool {
    self.0 & other.0 == other.0
  }

  pub fn union(self, other: Self) -> Self {
    Self(self.0 | other.0)
  }
}

pub struct Palette {
  entries: Box<[PaletteEntry; PALETTE_ENTRY_COUNT]>,
  used: Box<[u64; OCCUPIED_WORDS]>,
  dirty: Mutex<Option<(u16, u16)>>,
  version: AtomicU64,
  content_version: AtomicU64,
}

impl Clone for Palette {
  fn clone(&self) -> Self {
    Self {
      entries: self.entries.clone(),
      used: self.used.clone(),
      dirty: Mutex::new(Some((0, PALETTE_INDEX_MAX))),
      version: AtomicU64::new(0),
      content_version: AtomicU64::new(0),
    }
  }
}

impl PartialEq for Palette {
  fn eq(&self, other: &Self) -> bool {
    self.entries == other.entries
  }
}

impl Eq for Palette {}

impl std::fmt::Debug for Palette {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_struct("Palette")
      .field("entries", &PALETTE_ENTRY_COUNT)
      .field("dirty", &self.dirty.lock().map(|d| *d).unwrap_or(None))
      .finish()
  }
}

impl Default for Palette {
  fn default() -> Self {
    Self::new()
  }
}

impl Palette {
  pub fn new() -> Self {
    Self {
      entries: Box::new([PaletteEntry::default(); PALETTE_ENTRY_COUNT]),
      used: Box::new([0u64; OCCUPIED_WORDS]),
      dirty: Mutex::new(Some((0, PALETTE_INDEX_MAX))),
      version: AtomicU64::new(1),
      content_version: AtomicU64::new(0),
    }
  }

  #[inline]
  pub fn get(&self, idx: PaletteId) -> &PaletteEntry {
    &self.entries[idx.0 as usize]
  }

  pub fn set(&mut self, idx: PaletteId, entry: PaletteEntry) {
    assert!(!idx.is_air(), "index 0 is reserved for air");
    let i = idx.0 as usize;
    let was_used = (self.used[i / 64] >> (i % 64)) & 1 == 1;
    let changed = self.entries[i] != entry;
    self.entries[i] = entry;
    self.used[i / 64] |= 1u64 << (i % 64);
    self.mark_dirty(idx.0);
    self.version.fetch_add(1, Ordering::Release);
    if was_used && changed {
      self.content_version.fetch_add(1, Ordering::Release);
    }
  }

  #[inline]
  pub fn version(&self) -> u64 {
    self.version.load(Ordering::Acquire)
  }

  #[inline]
  pub fn content_version(&self) -> u64 {
    self.content_version.load(Ordering::Acquire)
  }

  fn mark_dirty(&self, idx: u16) {
    let mut d = self.dirty.lock().unwrap_or_else(|e| e.into_inner());
    *d = Some(match *d {
      None => (idx, idx),
      Some((lo, hi)) => (lo.min(idx), hi.max(idx)),
    });
  }

  pub fn take_dirty(&self) -> Option<(u16, u16)> {
    self.dirty.lock().unwrap_or_else(|e| e.into_inner()).take()
  }

  #[inline]
  pub fn is_air(&self, idx: PaletteId) -> bool {
    idx.is_air()
  }

  #[inline]
  pub fn occupied(&self, idx: PaletteId) -> bool {
    !idx.is_air() && (self.used[idx.0 as usize / 64] >> (idx.0 % 64)) & 1 == 1
  }

  #[inline]
  pub fn is_empty_slot(&self, idx: PaletteId) -> bool {
    !self.occupied(idx)
  }
}
