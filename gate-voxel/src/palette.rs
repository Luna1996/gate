//! 调色板：u16 索引 × 65536 条目，材质参数 + 视觉/语义标志位。
//!
//! 特殊体素走「调色板视觉变体 / CPU 校验标志 / 侧表数据」三级分流，不摊平进体素数据。
//!
//! **容量**：索引 16 位 → 单个 volume 最多 65536 种材质（0 保留给空气）。容量与体素
//! 载荷宽度是同一件事的两个端面：树里"整块同色"存在节点上、逐体素色存在叶层 inline，
//! 两处都是 16 位，故节点与体素必须同宽（见 `chunk_tree.rs` 的 `pack_pal_lod`）。
//! 表本体 65536 × 8B = 512KB/volume，与体素数量无关；多 volume 共享同一张表由
//! 渲染侧的 `palette_base` 指针天然支持。

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

/// 调色板条目数（索引 0 保留为空气，故可用材质数 = 本值 - 1）
pub const PALETTE_ENTRY_COUNT: usize = 65_536;

/// 材质索引位宽。**这是 wire 格式的一等参数**：节点 uniform 色（3 字头的第 3 字低半区）
/// 与叶父层逐体素色都按本宽度打包，改这里必须同步 shader 的解包掩码/位移。
pub const PALETTE_BITS: u32 = 16;

/// 最大合法索引
pub const PALETTE_INDEX_MAX: u16 = (PALETTE_ENTRY_COUNT - 1) as u16;

// 容量与位宽必须是同一个数（编译期校验，避免两处常量漂移）
const _: () = assert!(PALETTE_ENTRY_COUNT == 1usize << PALETTE_BITS);
const _: () = assert!(PALETTE_ENTRY_COUNT == PALETTE_INDEX_MAX as usize + 1);

/// 调色板索引（新类型：把"材质索引"与裸 `u16` 区分开，避免位宽再变时的静默截断）
///
/// 语义上只用 0..=2^16-1，其中 0 = 空气（见 [`PaletteId::AIR`]）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(transparent)]
pub struct PaletteId(pub u16);

impl PaletteId {
  /// 空气（体素数据里的 0 天然表示未占用）
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

/// 场景作者便利：`fill_box(..., 2)` 这类小槽号字面量默认推断为 `u8`
impl From<u8> for PaletteId {
  fn from(v: u8) -> Self {
    Self(v as u16)
  }
}

/// 场景作者便利：无类型约束的整数字面量会回落 `i32`，故也收它（越界直接 panic，
/// 不做静默截断 —— 槽号写错属于编程错误，应在构造场景时就炸出来）
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

/// 调色板条目：RGB 颜色 + PBR 简化参数 + 标志位
///
/// 体积对齐：3B color + 1B roughness + 1B emissive + 1B transmission + 1B flags = 7B，
/// 补 1B padding = 8B/条目，65536 条 = 512KB（可直接进 GPU buffer）。
///
/// GPU 侧当前只解包 `color` 与 `emissive`；`roughness`/`transmission`/`flags` 已随
/// 条目上传但尚无消费方，留给后续材质扩展（见 `wire.rs::pack_palette_entry`）。
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
  _pad: u8,
}

/// 标志位：手写 bit 常量（不引 bitflags crate）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PaletteFlags(pub u8);

impl PaletteFlags {
  pub const LOCKED: Self = Self(1 << 0); // 关卡锁定体素（不可编辑，全息视觉变体）
  pub const INPUT_PORT: Self = Self(1 << 1); // 输入端口（电路边界条件）
  pub const OUTPUT_PORT: Self = Self(1 << 2); // 输出端口
  pub const HOLOGRAM: Self = Self(1 << 3); // 全息渲染变体

  pub fn contains(self, other: Self) -> bool {
    self.0 & other.0 == other.0
  }

  pub fn union(self, other: Self) -> Self {
    Self(self.0 | other.0)
  }
}

/// 调色板：固定 65536 槽，索引 0 保留为「空/空气」语义
///
/// **脏槽追踪**：表本体 512KB 无法像原来 2KB 那样每次编辑整表重铺，故记录"自上次
/// 取走以来被写过的槽区间"。渲染侧上传后调 [`Palette::take_dirty`] 取走并清零。
/// 只提供 [`Palette::set`] 一条写路径（不暴露 `get_mut`），保证任何写入都会标脏。
///
/// **多消费者**：同一张表可能被多个 builder 消费（渲染世界每 volume 一个 + gate-app
/// 启动期的诊断 builder）。脏区间只能被取走一次，故额外维护单调递增的 [`Palette::version`]
/// 供消费者自行对齐"我需要同步吗"；拿不到脏区间的消费者退回全量铺（见 builder
/// `write_palette`），不会静默丢数据。
///
/// 脏标记用 `Mutex` 而非 `Cell`：`Palette` 内含于 `VolumeGrid`，而 `VoxelScene` 是 Bevy
/// `Resource`（要求 `Sync`），`Cell` 不满足。锁只在校验/取走时短暂持有，且调色板写入
/// 只发生在"新材质首次使用"这类低频路径，开销可忽略。
pub struct Palette {
  entries: Box<[PaletteEntry; PALETTE_ENTRY_COUNT]>,
  /// 脏槽闭区间（含两端）；None = 自上次取走以来无变化
  dirty: Mutex<Option<(u16, u16)>>,
  /// 写版本：每次 `set` 自增。消费者记住自己同步过的版本，用来判断是否需要同步
  /// （不依赖脏区间是否被别人取走）。`new()` 从 **1** 起，故 **0 = 克隆出来的新副本**
  /// （任何消费者都不可能已经同步过它，见 `Clone`）。
  version: AtomicU64,
}

/// 克隆出的副本一律视为**全表待上传**：副本若被挂到别的 volume 上，其 GPU 侧尚无内容。
/// 版本给 **0**（合法版本 ≥ 1），保证任何已同步过的消费者都会重新全量铺一次，不会被跳过。
impl Clone for Palette {
  fn clone(&self) -> Self {
    Self {
      entries: self.entries.clone(),
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
  /// 新建全空表（全表标脏：首次上传必须把 512KB 完整送上去）
  pub fn new() -> Self {
    Self {
      entries: Box::new([PaletteEntry::default(); PALETTE_ENTRY_COUNT]),
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
    self.mark_dirty(idx.0);
    self.version.fetch_add(1, Ordering::Release);
  }

  /// 当前写版本（供消费者对齐"我同步过了吗"）
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

  /// 取走脏槽区间（闭区间；None = 无变化）并清零。
  ///
  /// **只能被一个消费者拿到**：拿不到的消费者应退回全量铺（别静默跳过）。
  pub fn take_dirty(&self) -> Option<(u16, u16)> {
    self.dirty.lock().unwrap_or_else(|e| e.into_inner()).take()
  }

  #[inline]
  pub fn is_air(&self, idx: PaletteId) -> bool {
    idx.is_air()
  }

  /// 条目全零判据：空槽（编辑材质的槽位分配据此认领）
  #[inline]
  pub fn is_empty_slot(&self, idx: PaletteId) -> bool {
    !idx.is_air() && self.entries[idx.0 as usize] == PaletteEntry::default()
  }
}
