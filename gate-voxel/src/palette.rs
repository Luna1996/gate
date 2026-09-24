//! 调色板：u16 索引 × 65536 条目，材质参数 + 视觉/语义标志位。
//! 索引 0 保留给空气，最多 65535 种材质；表本体 65536 × 8B = 512KB/volume，与体素数量无关。
//! 节点 uniform 色与叶层逐体素色同宽（16 位，见 `chunk_tree.rs` 的 `pack_node_palette`）。

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
/// - PBR 变体（`IS_PBR = 1`）：这 8B 被解释为 `asset: u16` + 5 个标量覆盖
///   （构造入口是 [`PaletteEntry::pbr`]）。
///
/// 布局的写侧权威见 `wire.rs::pack_palette_entry`，读侧见 `common.wesl` 的 `palette_*`。
///
/// ## PBR 变体下的字段对应表（**关键**：字段名只是 8B 的"位置标签"，语义随变体变）
///
/// PBR 变体的字节位置完全自由（D1：平凡变体的位置被向后兼容钉死，PBR 变体按最省的方式排），
/// 于是同一个 8B 结构体在 PBR 变体下**每个字段都另有含义**：
///
/// | 字段（`PaletteEntry` 的 8B 视图） | 所属 word / bits | PBR 变体下的含义 |
/// |---|---|---|
/// | `color[0]` | word0 bits 0..7 | `roughness` 覆盖 |
/// | `color[1]` | word0 bits 8..15 | `metallic` 覆盖 |
/// | `color[2]` | word0 bits 16..23 | `emissive` 覆盖 |
/// | `roughness` | word0 bits 24..31 | `transmission` 覆盖 |
/// | `emissive` | word1 bits 0..7 | `asset: u16` 的**低**字节 |
/// | `transmission` | word1 bits 8..15 | `asset: u16` 的**高**字节 |
/// | `flags` | word1 bits 16..23 | flags（含 `IS_PBR` 变体位） |
/// | `metallic` | word1 bits 24..31 | `specular` 覆盖（语义同 glTF `KHR_materials_specular`） |
///
/// 覆盖的编码见 [`PbrOverrides`]（`0` = 不覆盖）；`asset` 是 `MaterialAsset` 全局表的槽号，
/// 分工是「资产级给物理基值、槽级只给调节量」（见 D1「F0 的唯一来源规则」）。
///
/// **谁来写这 5 个覆盖字节：只有场景资产** —— `.vox` 的 MATL 元数据（`_rough` / `_metal` / `_emit`）
/// 由 `gate-app/src/vox_scene.rs::matl_to_entry` 落进来。**编辑器画笔一律全 0**（"一刀切"：
/// 用 PBR 就用资产与贴图那一份，菜单控件整行置灰，见 `gate-app/src/edit.rs::BrushMaterial`）。
///
/// **读侧（shader / 本仓任何按字段读的地方）不得对 PBR 变体套用平凡语义**：变体判定只认
/// `flags` 的 `IS_PBR` 位（`common.wesl::fetch_material` 就是这么分派的）。
///
/// `PartialEq`（内容去重的判据，见 `gate-app/src/edit.rs::material_slot`）在**两个变体上都与
/// `pack_palette_entry` 的输出一一对应**：打包是单射（每个字段都落在固定的 bit 段里、无重叠、
/// 无被丢弃的字段）⇒ 等字段 ⟺ 等 8B payload。
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

impl PaletteEntry {
  /// 构造 **PBR 变体**条目：**只填字段、不打包**（打包只有一处 —— `wire.rs::pack_palette_entry`
  /// 按 `IS_PBR` 分派；本函数与它严格互逆，字段对应表见类型文档）。
  ///
  /// - `asset` 是 `MaterialAsset` 全局表的槽号（`u16` ⇒ 最多 65536 个资产，D1）；本函数把它**拆成
  ///   `emissive`（低字节）/ `transmission`（高字节）两个"位置标签"**——这两个字段在平凡变体里
  ///   是发光/透射，在 PBR 变体里只是 `asset` 的两个字节，**不要**按平凡语义去读它们；
  /// - `IS_PBR` 由本函数保证置上（`pack_palette_entry` 据此分派）；
  /// - `TRANSMISSIVE` 由**调用方**决定（D1：只有调用方知道所选资产是不是透射材质）；
  ///   平凡变体那条"`transmission > 0` ⇒ 介质"的自动推断规则**不适用于** PBR 变体。
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

  /// PBR 变体的 `asset`（把 `emissive` / `transmission` 两个字节拼回）；[`Self::pbr`] 的逆。
  /// 平凡变体下无意义（那两个字段是发光/透射）。
  #[inline]
  pub fn pbr_asset(&self) -> u16 {
    (self.emissive as u16) | ((self.transmission as u16) << 8)
  }

  /// PBR 变体的 5 个槽级覆盖；[`Self::pbr`] 的逆（平凡变体下无意义）。
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

/// PBR 变体的逐实例标量覆盖（`docs/PLAN.md` D1）。
/// - `0` = **不覆盖**（用材质资产的值）；
/// - `1..=255` = 覆盖为 `(v − 1) / 254`（`1` 因此能表达"完全镜面"这种 0 值，而不与"不覆盖"撞码）。
///
/// 定义在本 crate（而不是 `gate-render`）的理由：它是这 8B 的**语义**、不是打包细节 ——
/// `PaletteEntry::pbr` 要吃它，而 `gate-voxel` 是纯逻辑层、不能被 `gate-render` 反向依赖（硬约束 8）。
/// 编码的读写镜像在 `wire.rs::pack_palette_entry` 与 `common.wesl::override_value`（三处一致）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PbrOverrides {
  pub roughness: u8,
  pub metallic: u8,
  pub emissive: u8,
  pub transmission: u8,
  /// 语义同 glTF `KHR_materials_specular`：只调制**电介质**的 F0，对金属无效（D1「F0 的唯一来源规则」）。
  pub specular: u8,
}

// ============================================================================
// 槽级覆盖的编码 ↔ UI 滑杆的映射（MT7-1 的纯函数，带单元测试）
// 这些是"编码"本身的一部分（与 `PbrOverrides` 同一层），故放在本 crate：
// gate-app 的菜单动作只做「滑杆值 → 覆盖字节」，不另立第二套编码。
// ============================================================================

/// 参数值 `value ∈ [0, 1]` → 覆盖字节：`1 + round(value × 254)` ∈ `1..=255`。
/// `1..=255` 解码回 `(v−1)/254` ⇒ `1` 精确表达 0（完全镜面 / 完全不透明），`255` 表达 1。
pub fn override_byte(value: f32) -> u8 {
  (1.0 + (value.clamp(0.0, 1.0) * 254.0).round()) as u8
}

/// 覆盖字节的逆解码：`0` → `None`（不覆盖），`1..=255` → `Some((v−1)/254)`。
/// 与 `common.wesl::override_value`（返回 −1 哨兵）同义，只是 Rust 侧用 `Option` 表达。
pub fn override_value(byte: u8) -> Option<f32> {
  if byte == 0 { None } else { Some(f32::from(byte - 1) / 254.0) }
}

/// 「滑杆值越大 ⇒ 参数越大」的控件（metallic / specular / 自发光）→ 覆盖字节。
/// **最低档（`value <= min`）= 不覆盖**（用资产的值）——所以滑杆的行程被"让"出一档来给
/// "不覆盖"，这也是唯一能在单个滑杆上表达三态（不覆盖 / 覆盖为 0 / 覆盖为 1）的做法。
pub fn slider_to_override(value: f32, min: f32, max: f32) -> u8 {
  let span = max - min;
  if span <= 0.0 || value <= min {
    return 0;
  }
  override_byte((value - min) / span)
}

/// 「滑杆值越大 ⇒ 参数越小」的控件（光滑度 = 100 − roughness、不透明度 = 100 − transmission）
/// → 覆盖字节：参数值 = `(100 − pct) / 100`。最低档（`pct <= 0`）同上 = **不覆盖**。
///
/// 单测 `inverted_slider_keeps_direction_and_reaches_zero` 钉住这一档落差：
/// `pct = 0` 是"不覆盖"（不是"参数 = 1"），`pct = 100` 给出字节 `1` = 参数精确 0（完全镜面 / 全透）。
pub fn inverted_pct_to_override(pct: f32) -> u8 {
  if pct <= 0.0 {
    return 0;
  }
  override_byte((100.0 - pct.clamp(0.0, 100.0)) / 100.0)
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
  /// 写入侧由 [`PaletteEntry::pbr`] 保证置上；`wire.rs::pack_palette_entry` 按这一位**分派**
  /// （置上则按 PBR 布局打包，否则走平凡路径、逐位不变）。
  pub const IS_PBR: Self = Self(1 << 4);
  /// DDA 热路径位：置 1 = 该槽是**可穿透介质**。`trace.wesl::medium_of` 在 DDA 内逐体素调用，
  /// 只读这一位（与 transmission 同在 word1 ⇒ 零额外读取）—— PBR 变体里 transmission 所在字节
  /// 属于 `asset`，不能再按字节判介质。
  /// **由写入侧维护**：平凡变体 = `transmission > 0`（`wire.rs::pack_palette_entry` 自动推断）；
  /// PBR 变体由调用方决定、打包函数**原样透传**（只有调用方知道资产是不是透射材质）。
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

#[cfg(test)]
mod tests {
  use super::*;

  /// MT7-1 的映射函数必须与 `PbrOverrides` 的编码逐点一致（`0` = 不覆盖 / `1..=255` = `(v−1)/254`）。
  #[test]
  fn override_byte_matches_d1_encoding() {
    assert_eq!(override_byte(0.0), 1, "参数 0 必须编码成字节 1（而不是 0 = 不覆盖）");
    assert_eq!(override_byte(1.0), 255);
    assert_eq!(override_byte(0.5), 128);
    assert_eq!(override_value(override_byte(0.5)), Some(127.0 / 254.0));
    // 越界钳位：滑杆理论上给不出 > 1，但钳位是语义的一部分（不 panic、不环绕）
    assert_eq!(override_byte(-1.0), 1);
    assert_eq!(override_byte(9.0), 255);
  }

  /// 覆盖编码的逆：`0` = 不覆盖（`None`），其余给出 `(v−1)/254`。
  #[test]
  fn override_value_is_inverse_of_byte() {
    assert_eq!(override_value(0), None);
    assert_eq!(override_value(1), Some(0.0));
    assert_eq!(override_value(255), Some(1.0));
    for b in 0..=255u8 {
      match override_value(b) {
        None => assert_eq!(b, 0),
        Some(v) => assert_eq!(override_byte(v), b, "字节 {b} 往返不一致"),
      }
    }
  }

  /// 「不覆盖」必须在**最低档**就能到达（MT7-1 的验收之一：滑杆要能表达"不覆盖"）。
  #[test]
  fn lowest_notch_means_no_override() {
    assert_eq!(slider_to_override(0.0, 0.0, 100.0), 0, "metallic 滑杆最低档 = 不覆盖");
    assert_eq!(slider_to_override(0.0, 0.0, 255.0), 0, "自发光滑杆 0 = 不覆盖");
    assert_eq!(inverted_pct_to_override(0.0), 0, "光滑度/透明度最低档 = 不覆盖");
    // 最低档之外的行程仍覆盖满 0..1（除去为"不覆盖"让出的那一档）
    assert_eq!(slider_to_override(100.0, 0.0, 100.0), 255);
    assert_eq!(inverted_pct_to_override(100.0), 1, "光滑度 100% ⇒ roughness 覆盖为精确 0（镜面）");
  }

  /// 反向滑杆（光滑度 / 透明度）必须保持平凡变体的方向：pct 越大 ⇒ 参数越小。
  #[test]
  fn inverted_slider_keeps_direction_and_reaches_zero() {
    let rough = |pct: f32| override_value(inverted_pct_to_override(pct));
    let g50 = rough(50.0).unwrap();
    let g100 = rough(100.0).unwrap();
    let g1 = rough(1.0).unwrap();
    assert_eq!(g100, 0.0);
    assert!(g1 > g50 && g50 > g100, "粗糙度必须随「光滑度」单调不增：{g1} > {g50} > {g100}");
    // 50% 约等于旧的 `smooth_pct_to_roughness(50) = 128` 归一化值（128/255 ≈ 0.502）
    assert!((g50 - 0.502).abs() < 0.01);
  }

  /// `PaletteEntry::pbr` 的字段落位必须与 D1 的字节布局表一致（`pbr_asset` / `pbr_overrides` 是它的逆）。
  #[test]
  fn pbr_constructor_field_layout() {
    let ov = PbrOverrides { roughness: 1, metallic: 2, emissive: 3, transmission: 4, specular: 5 };
    let e = PaletteEntry::pbr(0xBEEF, ov, PaletteFlags::TRANSMISSIVE);
    assert_eq!(e.color, [1, 2, 3], "color[0..3] = roughness / metallic / emissive 覆盖");
    assert_eq!(e.roughness, 4, "roughness 字段 = transmission 覆盖");
    assert_eq!(e.metallic, 5, "metallic 字段 = specular 覆盖");
    assert_eq!(e.emissive, 0xEF, "emissive 字段 = asset 低字节");
    assert_eq!(e.transmission, 0xBE, "transmission 字段 = asset 高字节");
    assert_eq!(e.pbr_asset(), 0xBEEF);
    assert_eq!(e.pbr_overrides(), ov);
    assert!(e.flags.contains(PaletteFlags::IS_PBR), "IS_PBR 由构造器保证置上");
    assert!(e.flags.contains(PaletteFlags::TRANSMISSIVE), "其余位原样保留");
  }
}
