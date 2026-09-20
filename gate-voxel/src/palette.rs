//! 调色板：u16 索引 × 65536 条目，材质参数 + 视觉/语义标志位。
//! 索引 0 保留给空气，最多 65535 种材质；表本体 65536 × 8B = 512KB/volume，与体素数量无关。
//! 节点 uniform 色与叶层逐体素色同宽（16 位，见 `chunk_tree.rs` 的 `pack_pal_lod`）。

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

/// 调色板条目数（索引 0 保留为空气，可用材质数 = 本值 - 1）
pub const PALETTE_ENTRY_COUNT: usize = 65_536;

/// 材质索引位宽（wire 格式一等参数）。改这里必须同步 shader 的解包掩码/位移。
pub const PALETTE_BITS: u32 = 16;

/// 最大合法索引
pub const PALETTE_INDEX_MAX: u16 = (PALETTE_ENTRY_COUNT - 1) as u16;

/// 槽占用位图的字数（每字 64 槽）
const OCCUPIED_WORDS: usize = PALETTE_ENTRY_COUNT / 64;

const _: () = assert!(OCCUPIED_WORDS * 64 == PALETTE_ENTRY_COUNT);

const _: () = assert!(PALETTE_ENTRY_COUNT == 1usize << PALETTE_BITS);
const _: () = assert!(PALETTE_ENTRY_COUNT == PALETTE_INDEX_MAX as usize + 1);

/// 调色板索引（新类型，区别于裸 `u16`）；取值 0..=2^16-1，其中 0 = 空气。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(transparent)]
pub struct PaletteId(pub u16);

impl PaletteId {
  /// 空气（体素数据 0）。
  pub const AIR: Self = Self(0);

  #[inline]
  pub fn get(self) -> u16 {
    self.0
  }

  /// 是否空气
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

/// 场景作者便利：小槽号字面量（如 `2`）默认推断为 `u8`。
impl From<u8> for PaletteId {
  fn from(v: u8) -> Self {
    Self(v as u16)
  }
}

/// 场景作者便利：无类型约束的整数字面量回落 `i32`；越界 panic。
impl From<i32> for PaletteId {
  fn from(v: i32) -> Self {
    assert!(
      (0..=PALETTE_INDEX_MAX as i32).contains(&v),
      "调色板索引越界：{v}（合法范围 0..={PALETTE_INDEX_MAX}）"
    );
    Self(v as u16)
  }
}

/// 日志用（材质号直接打印）
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

/// 调色板条目：8B/条目，65536 条 = 512KB，可直接进 GPU buffer。
/// **这 8B 按 [`PaletteFlags::IS_PBR`] 复用为两种变体**（`docs/PLAN.md` D1 tagged union）：
/// - 平凡变体（`IS_PBR = 0`）：本结构体字段就是全部 payload（`color` + `roughness` + `emissive` +
///   `transmission` + `metallic`），字节布局与改动前**逐位相同**（只有原 `_pad` 变成有语义的 `metallic`）；
/// - PBR 变体（`IS_PBR = 1`）：这 8B 被解释为 `asset: u16` + 5 个标量覆盖，本结构体字段不再被读
///   （打包入口是 `gate-render/src/brickmap/wire.rs::pack_palette_entry_pbr`）。
///
/// 布局的写侧权威见 `wire.rs::pack_palette_entry`，读侧见 `common.wesl` 的 `palette_*`。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(C)]
pub struct PaletteEntry {
  /// sRGB 颜色（管线在 shader 侧转 linear）
  pub color: [u8; 3],
  /// 粗糙度 0（镜面）..255（完全粗糙）
  pub roughness: u8,
  /// 发光强度 0..255；发光颜色 = color × 强度
  pub emissive: u8,
  /// 透射率 0（不透明）..255（全透）；玻璃/LED 外壳用
  pub transmission: u8,
  /// 视觉变体 + 语义标志（见 [`PaletteFlags`]）
  pub flags: PaletteFlags,
  /// 金属度 0（非金属/电介质）..255（金属）；默认 0。
  /// 原先这个字节是废弃的 `_pad`（恒 0）⇒ 默认值下打包结果与改动前**逐位相同**。
  /// 贴图驱动时它是 rough-metal 贴图的 B 通道（8 bit 连续量），本字节是**无贴图时的回退值**。
  pub metallic: u8,
}

/// 标志位：手写 bit 常量（不引 bitflags crate）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PaletteFlags(pub u8);

impl PaletteFlags {
  pub const LOCKED: Self = Self(1 << 0); // 关卡锁定体素（不可编辑，全息视觉变体）
  pub const INPUT_PORT: Self = Self(1 << 1); // 输入端口（电路边界条件）
  pub const OUTPUT_PORT: Self = Self(1 << 2); // 输出端口
  pub const HOLOGRAM: Self = Self(1 << 3); // 全息渲染变体
  /// 变体位（D1）：置 1 ⇒ 这 8B payload 按 **PBR 变体**（`asset: u16` + 标量覆盖）解释。
  /// 写入侧由 `wire.rs::pack_palette_entry_pbr` 保证置上（`pack_palette_entry` 恒不置）。
  pub const IS_PBR: Self = Self(1 << 4);
  /// DDA 热路径位：置 1 = 该槽是**可穿透介质**。`trace.wesl::medium_of` 在 DDA 内逐体素调用，
  /// 只读这一位（与 transmission 同在 word1 ⇒ 零额外读取）—— PBR 变体里 transmission 所在字节
  /// 属于 `asset`，不能再按字节判介质。
  /// **由写入侧维护**：平凡变体 = `transmission > 0`（见 `wire.rs::pack_palette_entry`）；
  /// PBR 变体由调用方决定（只有它知道资产是不是透射材质）。
  /// 改动前 bit5 空闲且恒 0 ⇒ 旧条目的介质行为不变。
  pub const TRANSMISSIVE: Self = Self(1 << 5);

  pub fn contains(self, other: Self) -> bool {
    self.0 & other.0 == other.0
  }

  pub fn union(self, other: Self) -> Self {
    Self(self.0 | other.0)
  }
}

/// 调色板：固定 65536 槽，索引 0 保留为空/空气。
/// 写入只经 `set` 并标脏；`take_dirty` 取走脏槽区间（只能取一次），拿不到者退回全量铺。
pub struct Palette {
  entries: Box<[PaletteEntry; PALETTE_ENTRY_COUNT]>,
  /// 槽占用位图（bit=1 = 该槽被 `set` 写过），8KB；占用与条目内容无关。
  used: Box<[u64; OCCUPIED_WORDS]>,
  /// 脏槽闭区间（含两端）；None = 自上次取走以来无变化
  dirty: Mutex<Option<(u16, u16)>>,
  /// 写版本：每次 `set` 自增，消费者据此判断是否需同步；`new()` 从 1 起，0 = 克隆出的新副本（视为全表待上传）。
  version: AtomicU64,
}

/// 克隆出的副本一律视为全表待上传（`version` = 0）。
impl Clone for Palette {
  fn clone(&self) -> Self {
    Self {
      entries: self.entries.clone(),
      used: self.used.clone(),
      dirty: Mutex::new(Some((0, PALETTE_INDEX_MAX))),
      version: AtomicU64::new(0),
    }
  }
}

/// 只比较条目内容（脏标记是传输状态，不参与相等性）
impl PartialEq for Palette {
  fn eq(&self, other: &Self) -> bool {
    self.entries == other.entries
  }
}

impl Eq for Palette {}

/// 只打印条目数，不把 512KB 表内容打出来
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
  /// 新建全空表（全表标脏，首次须完整上传）。
  pub fn new() -> Self {
    Self {
      entries: Box::new([PaletteEntry::default(); PALETTE_ENTRY_COUNT]),
      used: Box::new([0u64; OCCUPIED_WORDS]),
      dirty: Mutex::new(Some((0, PALETTE_INDEX_MAX))),
      version: AtomicU64::new(1),
    }
  }

  #[inline]
  pub fn get(&self, idx: PaletteId) -> &PaletteEntry {
    &self.entries[idx.0 as usize]
  }

  /// 写入条目（AIR 槽不可占用，违规 panic）并标脏该槽
  pub fn set(&mut self, idx: PaletteId, entry: PaletteEntry) {
    assert!(!idx.is_air(), "index 0 is reserved for air");
    self.entries[idx.0 as usize] = entry;
    self.used[idx.0 as usize / 64] |= 1u64 << (idx.0 % 64);
    self.mark_dirty(idx.0);
    self.version.fetch_add(1, Ordering::Release);
  }

  /// 当前写版本（供消费者判断是否已同步）。
  #[inline]
  pub fn version(&self) -> u64 {
    self.version.load(Ordering::Acquire)
  }

  /// 标脏单个槽（区间取并集）
  fn mark_dirty(&self, idx: u16) {
    let mut d = self.dirty.lock().unwrap_or_else(|e| e.into_inner());
    *d = Some(match *d {
      None => (idx, idx),
      Some((lo, hi)) => (lo.min(idx), hi.max(idx)),
    });
  }

  /// 取走脏槽闭区间（None = 无变化）并清零；只能被一个消费者拿到，拿不到者应退回全量铺。
  pub fn take_dirty(&self) -> Option<(u16, u16)> {
    self.dirty.lock().unwrap_or_else(|e| e.into_inner()).take()
  }

  #[inline]
  pub fn is_air(&self, idx: PaletteId) -> bool {
    idx.is_air()
  }

  /// 该槽是否已被写过（占用位图；与条目内容无关）
  #[inline]
  pub fn occupied(&self, idx: PaletteId) -> bool {
    !idx.is_air() && (self.used[idx.0 as usize / 64] >> (idx.0 % 64)) & 1 == 1
  }

  /// 空槽判据：从未被 `set` 写过（新材质认领槽位据此挑选）。
  #[inline]
  pub fn is_empty_slot(&self, idx: PaletteId) -> bool {
    !self.occupied(idx)
  }
}
