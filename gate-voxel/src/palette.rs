//! 调色板（P1.3）：u8 索引 × 256 条目，材质参数 + 视觉/语义标志位
//!
//! grill-me v3 裁决：256 色（u8 索引）够用；特殊体素走「调色板视觉变体 /
//! CPU 校验标志 / 侧表数据」三级分流，不摊平进体素词。

/// 调色板条目：RGB 颜色 + PBR 简化参数 + 标志位
///
/// 体积对齐考虑（未来直接进 GPU uniform/storage buffer）：
/// 3B color + 1B roughness + 1B emissive + 1B transmission + 1B flags = 7B，
/// 补 1B padding = 8B/条目，256 条 = 2KB
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(C)]
pub struct PaletteEntry {
  /// sRGB 颜色（P2 管线在 shader 侧转 linear）
  pub color: [u8; 3],
  /// 粗糙度 0（镜面）..255（完全粗糙）
  pub roughness: u8,
  /// 发光强度 0..255；发光颜色 = color × 强度（P3.2 消费）
  pub emissive: u8,
  /// 透射率 0（不透明）..255（全透）；玻璃/LED 外壳用
  pub transmission: u8,
  /// 视觉变体 + 语义标志（见 [`PaletteFlags`]）
  pub flags: PaletteFlags,
  _pad: u8,
}

/// 标志位：手写 bit 常量，不引 bitflags crate（最小依赖偏好）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PaletteFlags(pub u8);

impl PaletteFlags {
  pub const LOCKED: Self = Self(1 << 0); // 关卡锁定体素（不可编辑，P3.6 全息视觉变体）
  pub const INPUT_PORT: Self = Self(1 << 1); // 输入端口（P5 电路边界条件）
  pub const OUTPUT_PORT: Self = Self(1 << 2); // 输出端口
  pub const HOLOGRAM: Self = Self(1 << 3); // 全息渲染变体

  pub fn contains(self, other: Self) -> bool {
    self.0 & other.0 == other.0
  }

  pub fn union(self, other: Self) -> Self {
    Self(self.0 | other.0)
  }
}

/// 调色板：固定 256 槽，索引 0 保留为「空/空气」语义
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Palette {
  entries: Box<[PaletteEntry; 256]>,
}

/// 索引 0 = 空（体素词里的 0 天然表示未占用，见 tile.rs）
pub const AIR_INDEX: u8 = 0;

impl Default for Palette {
  fn default() -> Self {
    Self::new()
  }
}

impl Palette {
  pub fn new() -> Self {
    Self {
      entries: Box::new([PaletteEntry::default(); 256]),
    }
  }

  pub fn get(&self, idx: u8) -> &PaletteEntry {
    &self.entries[idx as usize]
  }

  pub fn get_mut(&mut self, idx: u8) -> &mut PaletteEntry {
    &mut self.entries[idx as usize]
  }

  /// 写入条目，返回分配到的索引（AIR 槽不可占用）
  pub fn set(&mut self, idx: u8, entry: PaletteEntry) {
    assert!(idx != AIR_INDEX, "index 0 is reserved for air");
    self.entries[idx as usize] = entry;
  }

  pub fn is_air(&self, idx: u8) -> bool {
    idx == AIR_INDEX
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn entry_size_is_8_bytes() {
    assert_eq!(size_of::<PaletteEntry>(), 8);
  }

  #[test]
  fn air_slot_is_reserved() {
    let mut p = Palette::new();
    let r = std::panic::catch_unwind(move || {
      p.set(AIR_INDEX, PaletteEntry::default());
    });
    assert!(r.is_err());
  }

  #[test]
  fn flags_compose() {
    let f = PaletteFlags::LOCKED.union(PaletteFlags::INPUT_PORT);
    assert!(f.contains(PaletteFlags::LOCKED));
    assert!(f.contains(PaletteFlags::INPUT_PORT));
    assert!(!f.contains(PaletteFlags::OUTPUT_PORT));
  }
}
