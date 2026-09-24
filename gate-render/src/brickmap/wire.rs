//! Brick Tree wire 格式：常量、编码函数、全局参数；CPU 构建器与 GPU shader 之间的字节契约。
//! b_struct = Region ① 稠密 chunk 窗口 + Region ② 各 chunk 的**树块**（块首 = 根节点地址，
//! 块内是节点 arena；节点地址 = 块首 + 块内偏移）；`PALETTE_BITS`=16 同宽约束两侧一致。
//!
//! 节点字节契约（CPU `gate-voxel::ChunkTree::serialize_with_layout` 写、shader `trace.wesl` /
//! `brickmap.wesl` 读）：
//! ```text
//! [chunk 窗口] entry = 树块首（b_struct 内本 volume 字址）+ 1；0 = 无此 chunk
//! [节点]  +0/+1 = mask_lo/mask_hi（64 位子块占用）；+2 = uniform 子块色（低 16 位）
//!         有掩码时 +3 起：内部层 = 紧凑子块指针表（指针 = **相对根节点**的字偏移，按 mask 位序）
//!                        叶父层（4³）= 32 字 inline（每字 2 体素 × 16 位索引，0 = AIR）
//! ```
//! CONSTRAINT: 根节点恒占 `gate_voxel::ROOT_WIRE_WORDS` 字（3 + 满 64 槽指针表）—— 指针以根地址
//! 为基准，根一搬迁全体指针都要重算，故根的字数不随掩码变化（见 `ChunkTree::serialize_with_layout`）。

use gate_voxel::{PALETTE_BITS, PALETTE_ENTRY_COUNT, PaletteEntry, PaletteFlags};
use glam::{IVec3, Mat3, Vec3, Vec4};

/// PBR 变体的逐实例标量覆盖 —— **定义在 `gate-voxel`**（`PaletteEntry::pbr` 要吃它，而
/// `gate-voxel` 是纯逻辑层、不能被本 crate 反向依赖），这里只做**转发**以保持既有导入路径可用
/// （`wire::PbrOverrides`；编码与语义见 `gate_voxel::PbrOverrides` 的文档）。
pub use gate_voxel::PbrOverrides;

/// chunk 边长（voxel 单位）：256³
pub const CHUNK_SIZE: i32 = 256;
/// 分裂因子（每轴 4，共 64 子块）
pub const BRICK_FACTOR: i32 = 4;
/// 最大分裂深度：256 → 64 → 16 → 4 → 1
pub const MAX_LEVEL: u32 = 4;
/// 每 chunk 64 子块的节点 fixed 字数（mask_lo + mask_hi + palette）
pub const NODE_FIXED_WORDS: usize = 3;

/// 稠密 chunk 窗口每轴上限（64 chunk × 256 = 16384 voxel 覆盖半径）
pub const CHUNK_INDEX_CAP: usize = 64;
/// Region ① 字数：64³ = 262144 字 = 1MB
pub const CHUNK_INDEX_WORDS: usize = CHUNK_INDEX_CAP * CHUNK_INDEX_CAP * CHUNK_INDEX_CAP;
/// Region ②（chunk 树区）起始字偏移
pub const TREE_BASE: usize = CHUNK_INDEX_WORDS;

/// 调色板字数（2^16 条 × 2 u32 = 512KB/volume）
pub const PALETTE_WORDS: usize = PALETTE_ENTRY_COUNT * 2;
/// 叶父层每 u32 字装的体素数（= 2）
pub const LEAF_VOXELS_PER_WORD: usize = gate_voxel::LEAF_VOXELS_PER_WORD;
/// 叶父层（level 3，64 体素）inline 字数 = 32
pub const LEAF_INLINE_WORDS: usize = gate_voxel::LEAF_INLINE_WORDS;
/// 每槽字节数（8B；[`pack_palette_entry`] 输出 2 个 u32）
pub const PALETTE_BYTES_PER_ENTRY: usize = 8;

// 位宽一致性（编译期）：叶层每字体素数必须整除 64
const _: () = assert!(64 % LEAF_VOXELS_PER_WORD == 0);
const _: () = assert!(32 % PALETTE_BITS == 0);
/// comp_layer 每 chunk 字数（4096 个 u16 打包为 2048 个 u32）
pub const CHUNK_COMP_WORDS: usize = gate_voxel::COMP_BRICKS_PER_CHUNK / 2;
/// StateTable 条目的字数（4×u32/条目）
pub const STATE_WORDS_PER_ENTRY: usize = 4;
/// StateTable MVP 条目数（u16 组件 ID 低 8bit）
pub const STATE_ENTRY_COUNT: usize = 256;
/// StateTable 总字数（256×4 = 1024 = 4KB）
pub const STATE_TOTAL_WORDS: usize = STATE_ENTRY_COUNT * STATE_WORDS_PER_ENTRY;

/// PaletteEntry（8B，repr(C)）→ 2 个 u32（小端字节序打包）—— **唯一的 palette 打包点**，
/// 按 `flags::IS_PBR` 分派两种变体（`docs/PLAN.md` D1 tagged union；写侧权威）。
///
/// **平凡变体（`IS_PBR = 0`）—— 逐位布局**（读侧见 `common.wesl` 的 `palette_*`、
/// 介质读见 `trace.wesl::medium_of`）：
/// ```text
/// word0 = color.r | color.g<<8 | color.b<<16 | roughness<<24
/// word1 = emissive | transmission<<8 | flags<<16 | metallic<<24
/// ```
/// `metallic` 落在原先恒 0 的 `_pad` 字节上、默认 0 ⇒ **与改动前逐位相同**。`TRANSMISSIVE` 位由本分支维护。
///
/// **PBR 变体（`IS_PBR = 1`）—— 逐位布局**（构造入口 `gate_voxel::PaletteEntry::pbr`，
/// 字段对应表见该类型文档；平凡变体的字节位置被向后兼容钉死，PBR 变体的位置完全自由）：
/// ```text
/// word0 = roughness覆盖 | metallic覆盖<<8 | emissive覆盖<<16 | transmission覆盖<<24
/// word1 = asset:u16 | flags<<16 | specular覆盖<<24
/// ```
/// 本分支**不做任何推断**：`TRANSMISSIVE` 由调用方置/不置（只有它知道所选资产是不是透射材质），
/// `IS_PBR` 由 `PaletteEntry::pbr` 保证置上（否则分派不到这里）。
/// 覆盖的编码（`0` = 不覆盖 / `1..=255` = `(v−1)/254`）见 `gate_voxel::PbrOverrides`。
pub fn pack_palette_entry(e: &PaletteEntry) -> [u32; 2] {
  if e.flags.contains(PaletteFlags::IS_PBR) {
    // PBR 变体：8B 的每一字节都只是"位置标签"（`PaletteEntry` 的字段名在此变体下另有含义）。
    // 这里逐字节搬运、不重排 —— 顺序即 D1 的 PBR 布局表；与 `PaletteEntry::pbr` 严格互逆。
    return [
      e.color[0] as u32
        | (e.color[1] as u32) << 8
        | (e.color[2] as u32) << 16
        | (e.roughness as u32) << 24,
      e.emissive as u32
        | (e.transmission as u32) << 8
        | (e.flags.0 as u32) << 16
        | (e.metallic as u32) << 24,
    ];
  }
  let mut flags = e.flags.0;
  if e.transmission > 0 {
    // 介质位由**写入侧**维护（D1）：`medium_of` 在 DDA 内逐体素调用，改读本字节的 bit5
    // （与 transmission 同在 word1 ⇒ 零额外读取），不再读 transmission 字节 ——
    // PBR 变体里那个字节属于 `asset`。改动前 bit5 空闲且恒 0，故旧条目行为不变。
    flags |= PaletteFlags::TRANSMISSIVE.0;
  }
  [
    e.color[0] as u32
      | (e.color[1] as u32) << 8
      | (e.color[2] as u32) << 16
      | (e.roughness as u32) << 24,
    e.emissive as u32
      | (e.transmission as u32) << 8
      | (flags as u32) << 16
      | (e.metallic as u32) << 24,
  ]
}

/// PBR 变体（`flags::IS_PBR = 1`）的 8B 打包 —— [`pack_palette_entry`] 的**薄包装**
/// （实现只有一份：构造 `PaletteEntry::pbr` 后走同一个分派点，避免两份打包逻辑漂移）。
/// `flags` 里的 `TRANSMISSIVE` 由**调用方**决定，本函数不推断（见 `pack_palette_entry`）。
pub fn pack_palette_entry_pbr(asset: u16, ov: PbrOverrides, flags: PaletteFlags) -> [u32; 2] {
  pack_palette_entry(&PaletteEntry::pbr(asset, ov, flags))
}

/// `MaterialAsset` 各 `*_slot` 的「无贴图」哨兵：该通道退回 `albedo_rough` / `emissive_metal` /
/// `transmission_ior` 里的标量值。**0 不是哨兵**（层 0 可以被真实贴图占用）——判据是 `u32::MAX`。
pub const MATERIAL_SLOT_NONE: u32 = 0xFFFF_FFFF;

/// 材质资产条目：PBR 变体的 palette 槽按 `asset: u16` 索引本表。
/// **全局一张表**（所有 volume 共用，不是 per-volume；`asset` 是全局下标），大小 `MATERIAL_ASSET_SLOTS` 项。
/// 一个条目 = 「一组 PBR 贴图集槽位 + 无贴图时的标量回退值」；贴图形态是 2D 贴图集 + triplanar（D3），
/// 槽位指向 `texture_2d_array` 的层（层数上限 `MATERIAL_TEX_SLOTS`）。
///
/// 全 `u32` 字段（不用 `u8`/`u16`）+ `repr(C)` ⇒ 无对齐填充坑，storage array stride = 32B。
/// **无 normal 层**：法线是几何的函数（由 `voxel_normal` 隐式给出），凹凸走 MT6 的真实体素几何（D2）。
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, bevy::render::render_resource::ShaderType)]
pub struct MaterialAsset {
  /// albedo 贴图槽（sRGB，采样后转 linear）；`MATERIAL_SLOT_NONE` ⇒ 取 `albedo_rough` 的颜色
  pub albedo_slot: u32,
  /// rough-metal 贴图槽，**glTF ORM 布局：G = Roughness、B = Metalness**（Poly Haven 的 `arm` 就是它）；
  /// `MATERIAL_SLOT_NONE` ⇒ roughness 取 `albedo_rough`、metallic 取 `emissive_metal`
  pub roughmetal_slot: u32,
  /// emissive 贴图槽（乘 `emissive_metal` 的 emissive 强度）；`MATERIAL_SLOT_NONE` ⇒ 只用标量强度
  pub emissive_slot: u32,
  /// transmission 贴图槽（可选）；`MATERIAL_SLOT_NONE` ⇒ 只用 `transmission_ior` 的标量透射率
  pub transmission_slot: u32,
  /// 高度（位移）贴图槽 —— **仅 CPU 侧读**：MT6 的 CSG 位移在体素化那一刻采样一次，
  /// 由 `gate-app` 解码成普通 CPU 高度场（`gate-voxel` 零渲染依赖，硬约束 8）；
  /// **shader 不读本项**（位移产物就是普通体素，凹凸靠真实几何，不做法线贴图混合）。
  pub height_slot: u32,
  /// 标量回退值：`color.rgb(sRGB) | roughness<<24`，布局同平凡变体 word0 ⇒ 无贴图时逐位等价于平凡材质
  pub albedo_rough: u32,
  /// 标量回退值：`emissive | metallic<<8 | specular<<16 | displacement_amplitude<<24`（specular 语义同 glTF
  /// `KHR_materials_specular`，只调制电介质 F0；无 specular 贴图时它就是槽级覆盖之外的资产基值）。
  ///
  /// ⚠️ **bits 24..31（D1 里预留的"保留"字节）自 MT8-5 起有语义 = 位移幅度**（单位 **体素**，
  /// 0 = 不位移，8 = **峰-峰** 8 体素 ⇒ 偏置 0.5 下上下各 ±4；语义表见 `docs/PLAN.md` §3 D2 与
  /// `gate-app/src/height_field.rs::HeightField::displace_fn`）。读写走
  /// [`Self::displacement_amplitude`] / [`Self::with_displacement_amplitude`]，**不要手写移位**。
  /// **只被 CPU 读**：位移发生在体素化那一刻（`gate-app` 把高度图解码成普通 CPU 高度场 +
  /// 闭包交给 `gate_voxel`，硬约束 8），产物就是普通体素 ⇒ **shader 不读本字节**
  /// （`common.wesl` 只解 emissive/metallic/specular 三个低字节）。
  pub emissive_metal: u32,
  /// `transmission | ior_x100<<16`：透射率 + IOR ×100（u16 ⇒ IOR 0..655.35，玻璃 1.5 → 150）。
  /// IOR 是**资产级的物理基值**，同时服务玻璃折射与电介质 F0 = ((IOR−1)/(IOR+1))²
  /// （硬约束 9：F0 只有一个来源；逐槽调节走 PBR 变体的 `specular` 覆盖）。
  pub transmission_ior: u32,
}

const _: () = assert!(std::mem::size_of::<MaterialAsset>() == 32);

/// [`MaterialAsset::emissive_metal`] 里**位移幅度**字节的位偏移（MT8-5）—— 就是 D1 预留的
/// "保留"字节（bits 24..31），**布局一个字都没动**（本条与下面那条编译期断言就是取证）。
pub const MATERIAL_DISPLACE_AMPLITUDE_SHIFT: u32 = 24;
/// 位移幅度字节的掩码（8 bit ⇒ 0..=255 **体素**；0 = 不位移）。
pub const MATERIAL_DISPLACE_AMPLITUDE_MASK: u32 = 0xFF;

// 位域自检（编译期）：位移幅度必须是 `emissive_metal` 的**最高一个字节**（24..31），
// 且不越出 32B 条目的字段边界 —— 改布局时这里先炸，不会静默串到别的字段上。
const _: () = assert!(MATERIAL_DISPLACE_AMPLITUDE_SHIFT + 8 == 32);
const _: () =
  assert!(MATERIAL_DISPLACE_AMPLITUDE_MASK << MATERIAL_DISPLACE_AMPLITUDE_SHIFT == 0xFF00_0000);

impl MaterialAsset {
  /// 该材质的**位移幅度**（**体素**，峰-峰；`0` = 不位移）—— MT8-5 的读出侧。
  ///
  /// 语义："材质自带高度图 ⇒ CSG 表面按材质自动出凹凸"（决策 B = 甲，`docs/PLAN.md` §4b MT8-5）：
  /// 体素化之前读这一个字节决定"要不要位移 / 位移多少"（8 ⇒ 偏置 0.5 下上下各 ±4）。
  /// 消费方见 `gate-app/src/height_field.rs::MaterialDisplace`。
  pub fn displacement_amplitude(&self) -> u8 {
    ((self.emissive_metal >> MATERIAL_DISPLACE_AMPLITUDE_SHIFT) & MATERIAL_DISPLACE_AMPLITUDE_MASK)
      as u8
  }

  /// 链式设置位移幅度（只动那一个字节，其余 31B 逐位保留）。
  /// 与 [`Self::displacement_amplitude`] 严格互逆：`a.with_displacement_amplitude(n).displacement_amplitude() == n`。
  pub fn with_displacement_amplitude(mut self, amplitude: u8) -> Self {
    self.emissive_metal = (self.emissive_metal
      & !(MATERIAL_DISPLACE_AMPLITUDE_MASK << MATERIAL_DISPLACE_AMPLITUDE_SHIFT))
      | ((amplitude as u32) << MATERIAL_DISPLACE_AMPLITUDE_SHIFT);
    self
  }
}

// 字段名与整体布局须与 shaders/voxel_raytrace/ 的 Globals struct 字节兼容；
// index_origin/dims 与 tile_count 均为 chunk 语义（×256 voxel）。

/// GPU 全局 uniform；整体 80B。
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, bevy::render::render_resource::ShaderType)]
pub struct BrickMapGlobals {
  /// chunk 窗口原点（chunk 单位，可为负）
  pub index_origin_x: i32,
  pub index_origin_y: i32,
  pub index_origin_z: i32,
  pub index_origin_w: i32,
  /// chunk 窗口尺寸（chunk 单位，≤ CHUNK_INDEX_CAP）
  pub index_dims_x: u32,
  pub index_dims_y: u32,
  pub index_dims_z: u32,
  pub index_dims_w: u32,
  /// 有内容的 chunk 数（字段名沿用 WGSL 侧 `tile_count`，保持字节兼容）
  pub tile_count: u32,
  /// b_struct 树区字数（不含 Region ① chunk 窗口）
  pub node_words: u32,
  /// 作废树区累计字数（增量 append 后未压缩的旧字节；全量重建归零）
  pub node_free_words: u32,
  /// 恒 0（字段保留以维持 WGSL 字节兼容）
  pub brick_slabs: u32,
  pub brick_free: u32,
  /// 超出 chunk 窗口被拒绝的 chunk 数
  pub rejected_tiles: u32,
  /// grid_descs 有效条目数（主世界 + 物体）
  pub grid_count: u32,
  pub _pad1: u32,
  pub _pad2: u32,
  pub _pad3: u32,
  pub _pad4: u32,
}

// 主世界与物体统一描述符：`world = pos + rot · (local · scale)`（rot 为 glam Mat3 列；主世界 = identity）。
// std140：6×Vec4 + 12×u32 = 144B，storage array stride 144B（16B 对齐）。
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Default, bevy::render::render_resource::ShaderType)]
pub struct GridDesc {
  pub pos_scale: Vec4,
  pub rot0: Vec4,
  pub rot1: Vec4,
  pub rot2: Vec4,

  pub aabb_min: Vec4,
  pub aabb_max: Vec4,

  pub tree_base: u32,
  pub tree_depth: u32,
  pub chunk_count: u32,
  pub palette_base: u32,

  pub index_origin_x: i32,
  pub index_origin_y: i32,
  pub index_origin_z: i32,
  pub _pad0: u32,
  pub index_dims_x: u32,
  pub index_dims_y: u32,
  pub index_dims_z: u32,
  pub _pad1: u32,
}

impl GridDesc {
  /// 主世界默认 GridDesc（identity transform，由 builder 填充 tree_base/palette_base/window）
  pub const IDENTITY: Self = Self {
    pos_scale: Vec4::new(0.0, 0.0, 0.0, 1.0),
    rot0: Vec4::new(1.0, 0.0, 0.0, 0.0),
    rot1: Vec4::new(0.0, 1.0, 0.0, 0.0),
    rot2: Vec4::new(0.0, 0.0, 1.0, 0.0),
    aabb_min: Vec4::ZERO,
    aabb_max: Vec4::ZERO,
    tree_base: 0,
    tree_depth: MAX_LEVEL,
    chunk_count: 0,
    palette_base: 0,
    index_origin_x: 0,
    index_origin_y: 0,
    index_origin_z: 0,
    _pad0: 0,
    index_dims_x: 0,
    index_dims_y: 0,
    index_dims_z: 0,
    _pad1: 0,
  };

  /// 从 VolumeTransform + tree/palette 基址构造
  #[allow(clippy::too_many_arguments)]
  pub fn from_transform(
    pos: Vec3,
    rot: Mat3,
    scale: f32,
    tree_base: u32,
    palette_base: u32,
    chunk_count: u32,
    origin: IVec3,
    dims: IVec3,
  ) -> Self {
    let (mn, mx) = transform_aabb(pos, rot, scale);
    Self {
      pos_scale: pos.extend(scale),
      rot0: rot.x_axis.extend(0.0),
      rot1: rot.y_axis.extend(0.0),
      rot2: rot.z_axis.extend(0.0),
      aabb_min: mn.extend(0.0),
      aabb_max: mx.extend(0.0),
      tree_base,
      tree_depth: MAX_LEVEL,
      chunk_count,
      palette_base,
      index_origin_x: origin.x,
      index_origin_y: origin.y,
      index_origin_z: origin.z,
      _pad0: 0,
      index_dims_x: dims.x as u32,
      index_dims_y: dims.y as u32,
      index_dims_z: dims.z as u32,
      _pad1: 0,
    }
  }
}

/// LUT octant 数：射线方向符号组合。编码同 shaders/voxel_raytrace/ `dir_mask`：
/// bit0/1/2 = x/y/z 正（零分量按正处理 = 保守）。
pub const MARCH_MASK_OCTANTS: usize = 8;
/// LUT 入口格数：4³ brick 内的 DDA 起始格。编码同 shaders/voxel_raytrace/ `child_idx`：`z*16 + y*4 + x`。
pub const MARCH_MASK_ENTRIES: usize = 64;
/// 每入口格掩码字数（64-bit 子块占用 → 2×u32）
pub const MARCH_MASK_WORDS_PER_ENTRY: usize = 2;
/// LUT 总字数：8 × 64 × 2 = 1024 u32 = 4KB
pub const MARCH_MASK_WORDS: usize =
  MARCH_MASK_OCTANTS * MARCH_MASK_ENTRIES * MARCH_MASK_WORDS_PER_ENTRY;

/// 生成方向可达掩码 LUT（低 64 bit 有效）。
/// `lut[octant][entry]` = 从入口格 `entry` 出发、方向符号 `octant` 的射线可能经过的子块集合（保守超集）。
pub fn march_mask_lut_words() -> Vec<u32> {
  let mut out = vec![0u32; MARCH_MASK_WORDS];
  for oct in 0..MARCH_MASK_OCTANTS {
    let pos = [oct & 1 != 0, (oct >> 1) & 1 != 0, (oct >> 2) & 1 != 0];
    for entry in 0..MARCH_MASK_ENTRIES {
      let e = [(entry & 3) as i32, ((entry >> 2) & 3) as i32, ((entry >> 4) & 3) as i32];
      let mut mask = 0u64;
      for p in 0..MARCH_MASK_ENTRIES {
        let q = [(p & 3) as i32, ((p >> 2) & 3) as i32, ((p >> 4) & 3) as i32];
        let ok =
          [0, 1, 2].iter().all(|&i| if pos[i] { q[i] >= e[i] - 1 } else { q[i] <= e[i] + 1 });
        if ok {
          mask |= 1u64 << p;
        }
      }
      let base =
        entry * MARCH_MASK_WORDS_PER_ENTRY + oct * MARCH_MASK_ENTRIES * MARCH_MASK_WORDS_PER_ENTRY;
      out[base] = mask as u32;
      out[base + 1] = (mask >> 32) as u32;
    }
  }
  out
}

/// 局部 [0,256]³·scale 经旋转平移后的世界 AABB
fn transform_aabb(pos: Vec3, rot: Mat3, scale: f32) -> (Vec3, Vec3) {
  let mut mn = Vec3::splat(f32::MAX);
  let mut mx = Vec3::splat(f32::MIN);
  for &x in &[0.0_f32, 256.0] {
    for &y in &[0.0, 256.0] {
      for &z in &[0.0, 256.0] {
        let local = Vec3::new(x, y, z) * scale;
        let w = pos + rot * local;
        mn = mn.min(w);
        mx = mx.max(w);
      }
    }
  }
  (mn, mx)
}

/// 构建产物：与 GPU buffer 字节一一对应的内容（原样上传）
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BrickMapBuffers {
  /// Region ① chunk 窗口 + Region ② 各 chunk 树块
  pub b_struct: Vec<u32>,
  pub b_palette: Vec<u32>,
  pub globals: BrickMapGlobals,
}

#[cfg(test)]
mod tests {
  use super::*;

  /// MT8-5 的位域取证：位移幅度 = `emissive_metal` 的 bits 24..31（D1 预留的"保留"字节），
  /// 读写助手严格互逆、**不动其余 31B**，且 `MaterialAsset` 仍是 32B（布局未变）。
  #[test]
  fn displacement_amplitude_lives_in_the_reserved_byte() {
    assert_eq!(std::mem::size_of::<MaterialAsset>(), 32, "MT8-5 不改 32B 布局");
    // 默认资产（未设幅度）= 0；把它放在 D1 的位置上（`emissive | metallic<<8 | specular<<16`）
    let a = MaterialAsset { emissive_metal: 0x11 | 0x22 << 8 | 0x33 << 16, ..Default::default() };
    assert_eq!(a.displacement_amplitude(), 0);
    // 8 体素落进最高字节、低三字节逐位不变（可换算的字面量：0x08332211）
    let b = a.with_displacement_amplitude(8);
    assert_eq!(b.emissive_metal, 0x0833_2211);
    assert_eq!(b.displacement_amplitude(), 8);
    // 覆盖已有值（只改那一个字节）
    assert_eq!(b.with_displacement_amplitude(4).emissive_metal, 0x0433_2211);
    // 上界 255 不越界到别的字段（条目的字节数不变）
    assert_eq!(b.with_displacement_amplitude(255).displacement_amplitude(), 255);
  }

  /// 平凡变体的**逐位向后兼容**取证：三个字面量取自 `docs/PLAN.md §8`（MT1 的回归表），
  /// 它们是在 MT1 改动**之后**实测到的值 —— 本测试把"打包逐位不变"钉死在回归测试里。
  #[test]
  fn plain_variant_packing_is_bit_for_bit_unchanged() {
    // 纯色不透明：flags = LOCKED；metallic 默认 0（原 `_pad`）⇒ 与改动前完全一致
    let opaque = PaletteEntry {
      color: [0xC8, 0x64, 0x32],
      roughness: 0x80,
      flags: PaletteFlags::LOCKED,
      ..Default::default()
    };
    assert_eq!(pack_palette_entry(&opaque), [0x803264C8, 0x00010000]);

    // 玻璃 transmission = 200：唯一差异是 bit21（word1 的 flags 字节 bit5 = TRANSMISSIVE 0→1）
    let glass = PaletteEntry { color: [0x0A, 0x14, 0x1E], transmission: 200, ..Default::default() };
    assert_eq!(pack_palette_entry(&glass), [0x001E140A, 0x0020C800]);

    // 发光玻璃（HOLOGRAM + transmission = 7）：同上，只有介质位是"新"的
    let hologram = PaletteEntry {
      color: [0xFF, 0x80, 0x00],
      roughness: 0x28,
      emissive: 0x5A,
      transmission: 7,
      flags: PaletteFlags::HOLOGRAM,
      ..Default::default()
    };
    assert_eq!(pack_palette_entry(&hologram), [0x280080FF, 0x0028075A]);
  }

  /// 平凡变体的介质位维护规则：`transmission > 0 ⟺ TRANSMISSIVE = 1`（D1 / MT5-0 的写侧保证）。
  #[test]
  fn plain_variant_maintains_transmissive_bit() {
    let mut e = PaletteEntry { transmission: 1, ..Default::default() };
    assert_ne!(pack_palette_entry(&e)[1] & 0x0020_0000, 0);
    e.transmission = 0;
    assert_eq!(pack_palette_entry(&e)[1] & 0x0020_0000, 0);
    // 调用方显式置位但 transmission = 0：本分支**不**清除它（只做或运算，与改动前一致）
    e.flags = PaletteFlags::TRANSMISSIVE;
    assert_ne!(pack_palette_entry(&e)[1] & 0x0020_0000, 0);
  }

  /// `IS_PBR = 0` 时**不**看 PBR 字段：即便字段长得像 PBR payload，也一律按平凡布局打包。
  #[test]
  fn plain_variant_ignores_pbr_payload() {
    let e = PaletteEntry {
      color: [1, 2, 3],
      roughness: 4,
      emissive: 5,
      transmission: 6,
      flags: PaletteFlags::LOCKED,
      metallic: 7,
    };
    assert_eq!(pack_palette_entry(&e), [0x04030201, 0x07210605]);
  }

  /// PBR 变体：「构造 → 打包 → 期望 u32」的对照取证（D1 的 PBR 字节布局表）。
  #[test]
  fn pbr_variant_packing_matches_d1_layout() {
    // 覆盖 = {一档全满的 roughness、完全金属、不覆盖 emissive、不覆盖 transmission、不调制 specular}
    let ov =
      PbrOverrides { roughness: 1, metallic: 255, emissive: 0, transmission: 0, specular: 255 };
    // asset = 10（`metal_plate`）；flags 只带 IS_PBR（0x10）⇒ 不是介质
    let e = PaletteEntry::pbr(10, ov, PaletteFlags::default());
    assert_eq!(e.flags.0, 0x10, "构造器置上 IS_PBR");
    // word0 = 1 | 255<<8 | 0<<16 | 0<<24 = 0x0000FF01
    // word1 = 10 | (0x10)<<16 | 255<<24 = 0xFF10000A
    assert_eq!(pack_palette_entry(&e), [0x0000_FF01, 0xFF10_000A]);
    // 薄包装走的是同一条实现（不再有第二份打包逻辑）
    assert_eq!(pack_palette_entry_pbr(10, ov, PaletteFlags::default()), [0x0000_FF01, 0xFF10_000A]);

    // `TRANSMISSIVE` 在 PBR 变体里是**调用方**的决定，打包原样透传（0x20 ⇒ word1 的 flags 字节 0x30）
    let e_trans = PaletteEntry::pbr(10, ov, PaletteFlags::TRANSMISSIVE);
    assert_eq!(pack_palette_entry(&e_trans), [0x0000_FF01, 0xFF30_000A]);
  }

  /// PBR 变体的严格互逆：打包出的两个 word 按 D1 的布局手写解包，必须逐字段还原出
  /// `PaletteEntry::pbr` 的入参（= 报告里那张"构造 → 打包 → 期望 u32"表的可执行版本）。
  #[test]
  fn pbr_variant_pack_unpack_round_trip() {
    let ov =
      PbrOverrides { roughness: 1, metallic: 128, emissive: 7, transmission: 200, specular: 64 };
    let flags = PaletteFlags::LOCKED;
    let asset: u16 = 0x0ABC;
    let src = PaletteEntry::pbr(asset, ov, flags);
    let [w0, w1] = pack_palette_entry(&src);

    // 手写解包（顺序照 D1：word0 = roughness/metallic/emissive/transmission 覆盖）
    let unpacked = PbrOverrides {
      roughness: (w0 & 0xFF) as u8,
      metallic: ((w0 >> 8) & 0xFF) as u8,
      emissive: ((w0 >> 16) & 0xFF) as u8,
      transmission: ((w0 >> 24) & 0xFF) as u8,
      specular: ((w1 >> 24) & 0xFF) as u8,
    };
    let unpacked_flags = ((w1 >> 16) & 0xFF) as u8;
    let unpacked_asset = (w1 & 0xFFFF) as u16;

    assert_eq!(unpacked, ov);
    assert_eq!(unpacked_asset, asset);
    assert_eq!(unpacked_flags, flags.union(PaletteFlags::IS_PBR).0);
    // 结构的 `pbr_asset` / `pbr_overrides` 与手写解包一致（两条读回路径不得分叉）
    assert_eq!(src.pbr_asset(), asset);
    assert_eq!(src.pbr_overrides(), ov);
  }
}
