//! Brick Tree wire 格式：常量、编码函数、全局参数；CPU 构建器与 GPU shader 之间的字节契约。
//! b_struct = Region ① 稠密 chunk 窗口 + Region ② 各 chunk 的 DFS 树；`PALETTE_BITS`=16 同宽约束两侧一致。

use gate_voxel::{PALETTE_BITS, PALETTE_ENTRY_COUNT, PaletteEntry, PaletteFlags};
use glam::{IVec3, Mat3, Vec3, Vec4};

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

/// PaletteEntry（8B，repr(C)）→ 2 个 u32（小端字节序打包）—— **平凡变体**（`flags::IS_PBR = 0`）。
/// 逐位布局（`docs/PLAN.md` D1；写侧权威，读侧见 `common.wesl` 的 `palette_*`、介质读见 `trace.wesl::medium_of`）：
/// ```text
/// word0 = color.r | color.g<<8 | color.b<<16 | roughness<<24
/// word1 = emissive | transmission<<8 | flags<<16 | metallic<<24
/// ```
/// `metallic` 落在原先恒 0 的 `_pad` 字节上、默认 0 ⇒ **与改动前逐位相同**。`TRANSMISSIVE` 位由本函数维护。
pub fn pack_palette_entry(e: &PaletteEntry) -> [u32; 2] {
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

/// PBR 变体的逐实例标量覆盖。`0` = 不覆盖（用材质资产的值）；`1..=255` = 覆盖为 `(v-1)/254`
/// （`1` 因此能表达"完全镜面"这种 0 值，而不与"不覆盖"撞码）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PbrOverrides {
  pub roughness: u8,
  pub metallic: u8,
  pub emissive: u8,
  pub transmission: u8,
  /// 语义同 glTF `KHR_materials_specular`：只调制**电介质**的 F0，对金属无效（D1「F0 的唯一来源规则」）。
  pub specular: u8,
}

/// PBR 变体（`flags::IS_PBR = 1`）的 8B 打包。布局（`docs/PLAN.md` D1；平凡变体的字节位置被向后兼容钉死，
/// PBR 变体的位置完全自由，故按最省的方式排）：
/// ```text
/// word0 = roughness 覆盖 | metallic 覆盖<<8 | emissive 覆盖<<16 | transmission 覆盖<<24
/// word1 = asset:u16 | flags<<16 | specular 覆盖<<24
/// ```
/// `flags` 里的 `TRANSMISSIVE` 由**调用方**决定 —— 只有调用方知道该资产是不是透射材质
/// （可能来自资产的 transmission 贴图/标量），本函数不推断。`IS_PBR` 则由本函数保证置上。
pub fn pack_palette_entry_pbr(asset: u16, ov: PbrOverrides, flags: PaletteFlags) -> [u32; 2] {
  [
    ov.roughness as u32
      | (ov.metallic as u32) << 8
      | (ov.emissive as u32) << 16
      | (ov.transmission as u32) << 24,
    asset as u32
      | ((flags.union(PaletteFlags::IS_PBR).0 as u32) << 16)
      | (ov.specular as u32) << 24,
  ]
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
  /// 标量回退值：`emissive | metallic<<8 | specular<<16 | 保留<<24`（specular 语义同 glTF
  /// `KHR_materials_specular`，只调制电介质 F0；无 specular 贴图时它就是槽级覆盖之外的资产基值）
  pub emissive_metal: u32,
  /// `transmission | ior_x100<<16`：透射率 + IOR ×100（u16 ⇒ IOR 0..655.35，玻璃 1.5 → 150）。
  /// IOR 是**资产级的物理基值**，同时服务玻璃折射与电介质 F0 = ((IOR−1)/(IOR+1))²
  /// （硬约束 9：F0 只有一个来源；逐槽调节走 PBR 变体的 `specular` 覆盖）。
  pub transmission_ior: u32,
}

const _: () = assert!(std::mem::size_of::<MaterialAsset>() == 32);

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
  /// Region ① chunk 窗口 + Region ② 各 chunk DFS 树
  pub b_struct: Vec<u32>,
  pub b_palette: Vec<u32>,
  pub globals: BrickMapGlobals,
}
