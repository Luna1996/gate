//! Brick Tree wire 格式：常量、编码函数、全局参数。本模块是 CPU 构建器（builder.rs）与 GPU
//! shader 之间的字节契约，纯数据变换、零渲染依赖，可脱离渲染运行时单测。
//!
//! b_struct：`[0 .. CHUNK_INDEX_WORDS)` 为稠密 chunk 窗口（CHUNK_INDEX_CAP³），
//! entry = chunk DFS 树绝对字基址 + 1（0 = 无此 chunk）；其后为各 chunk 的 DFS 序列化树，
//! 节点 = [mask_lo, mask_hi, palette_u32] + popcount(mask) 个 child offset（chunk 内相对字址）。
//! mask bit=1 → 子块被分裂（child offset 有效）；bit=0 → uniform 子块，颜色 = 该节点
//! palette_u32（零额外 load），palette=0 = AIR。
//!
//! level 3 叶父层例外：inline 16 word，4 体素/word，读端按 child_idx 低 2 位选字节。
//! b_leaves 存放方向可达掩码 LUT（Douglas #18 Bitwise Masking），见 [`march_mask_lut_words`]。

use gate_voxel::PaletteEntry;
use glam::{IVec3, Mat3, Vec3, Vec4};

// ============ 层级常量（与 gate-voxel coords.rs 一致）============

/// chunk 边长（voxel 单位）：256³
pub const CHUNK_SIZE: i32 = 256;
/// 分裂因子（4³ = 64 子块）
pub const BRICK_FACTOR: i32 = 4;
/// 最大分裂深度：256 → 64 → 16 → 4 → 1
pub const MAX_LEVEL: u32 = 4;
/// 每 chunk 64 子块的节点 fixed 字数（mask_lo + mask_hi + palette）
pub const NODE_FIXED_WORDS: usize = 3;

// ============ b_struct Region ①：稠密 chunk 窗口 ============

/// 稠密 chunk 窗口每轴上限（64 chunk × 256 = 16384 voxel 覆盖半径）
pub const CHUNK_INDEX_CAP: usize = 64;
/// Region ① 字数：64³ = 262144 字 = 1MB
pub const CHUNK_INDEX_WORDS: usize = CHUNK_INDEX_CAP * CHUNK_INDEX_CAP * CHUNK_INDEX_CAP;
/// Region ②（chunk 树区）起始字偏移
pub const TREE_BASE: usize = CHUNK_INDEX_WORDS;

// ============ palette / comp / state ============

/// 调色板字数（256 条 × 2 u32）
pub const PALETTE_WORDS: usize = 256 * 2;
/// comp_layer 每 chunk 字数（u16[4096] → 每 2 字打包成 u32 = 2048）
pub const CHUNK_COMP_WORDS: usize = gate_voxel::COMP_BRICKS_PER_CHUNK / 2;
/// StateTable 条目的字数（4×u32/条目）
pub const STATE_WORDS_PER_ENTRY: usize = 4;
/// StateTable MVP 条目数（u16 组件 ID 低 8bit）
pub const STATE_ENTRY_COUNT: usize = 256;
/// StateTable 总字数（256×4 = 1024 = 4KB）
pub const STATE_TOTAL_WORDS: usize = STATE_ENTRY_COUNT * STATE_WORDS_PER_ENTRY;

// ============ PaletteEntry 打包 ============

/// PaletteEntry（8B，repr(C)）→ 2 个 u32（小端字节序打包）
pub fn pack_palette_entry(e: &PaletteEntry) -> [u32; 2] {
  [
    e.color[0] as u32
      | (e.color[1] as u32) << 8
      | (e.color[2] as u32) << 16
      | (e.roughness as u32) << 24,
    e.emissive as u32 | (e.transmission as u32) << 8 | (e.flags.0 as u32) << 16,
  ]
}

// ============ GPU 全局参数 ============
//
// 字段名与整体布局须与 shaders/voxel_raytrace/ 的 Globals struct 字节兼容；
// index_origin/dims 与 tile_count 均为 chunk 语义（×256 voxel）。

/// 三轴字段全部拆成具名 scalar（x/y/z/w）、尾部填充同理：encase 0.12.1 在 uniform 模式下对
/// Rust fixed-size `[i32/u32; N]` 断言 "array stride must be a multiple of 16"。
/// 整体字节数 = 16+16+(7×4)+(5×4) = 32+28+20 = 80B（std140 允许末尾非 16 对齐）。
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

// ============ GridDesc（主世界与物体统一描述符，144B）============
//
// 主世界与物体走同一 GridDesc 数组，shader `trace_scene` 遍历数组无 kind 分支。
//
// 字段语义：
// - pos_scale/rot0/rot1/rot2：`world = pos + rot · (local · scale)`，
//   rot 列向量与 glam Mat3 一致（x_axis/y_axis/z_axis = 列）；主世界 = identity。
// - aabb_min/max：局部 [0,256]³·scale 经变换后的世界外包盒（CPU 预算，剔除用）。
// - tree_base：本 volume 的 b_struct 在 struct_buf 内的字基址（含 chunk 窗口段）。
// - tree_depth：最大分裂深度 = 4（256→64→16→4→1）。
// - chunk_count：本 volume 的 chunk 数（主世界可能 N，物体 = 1）。
// - palette_base：本 volume 的 palette 在 palette_buf 内的字基址。
// - index_origin/dims：本 volume 的稠密 chunk 窗口（chunk 单位）；物体 dims=(1,1,1)。
//
// std140 布局：6×Vec4(96B) + 4×u32(16B) + 4×i32(16B) + 4×u32(16B) = 144B
// storage buffer array stride = 144B（16B 对齐 ✓）。
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Default, bevy::render::render_resource::ShaderType)]
pub struct GridDesc {
  // ---- 变换（64B）----
  pub pos_scale: Vec4,
  pub rot0: Vec4,
  pub rot1: Vec4,
  pub rot2: Vec4,
  // ---- 世界 AABB（32B）----
  pub aabb_min: Vec4,
  pub aabb_max: Vec4,
  // ---- Brick Tree 数据基址（16B）----
  pub tree_base: u32,
  pub tree_depth: u32,
  pub chunk_count: u32,
  pub palette_base: u32,
  // ---- chunk 窗口（32B；物体 dims=(1,1,1) origin=(0,0,0)）----
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
  #[allow(clippy::too_many_arguments)] // 逐字段展开，语义即参数名（wire 契约）
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
    // 局部 [0,256]³·scale 经旋转平移后的世界 AABB 外包
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

// ============ 方向可达掩码 LUT（Douglas #18 Bitwise Masking）============

/// LUT octant 数：射线方向符号组合。编码与 shaders/voxel_raytrace/ `dir_mask` 一致：
/// bit0 = x 正方向、bit1 = y 正、bit2 = z 正（正 = 1，零分量按正处理 = 保守）。
pub const MARCH_MASK_OCTANTS: usize = 8;
/// LUT 入口格数：4³ brick 内的 DDA 起始格。编码与 shaders/voxel_raytrace/ `child_idx` 一致：
/// `z*16 + y*4 + x`。
pub const MARCH_MASK_ENTRIES: usize = 64;
/// 每入口格掩码字数（64-bit 子块占用 → 2×u32）
pub const MARCH_MASK_WORDS_PER_ENTRY: usize = 2;
/// LUT 总字数：8 × 64 × 2 = 1024 u32 = 4KB
pub const MARCH_MASK_WORDS: usize =
  MARCH_MASK_OCTANTS * MARCH_MASK_ENTRIES * MARCH_MASK_WORDS_PER_ENTRY;

/// 生成方向可达掩码 LUT（与 octo-release `march_masks` 同构，低 64 bit 有效）。
///
/// `lut[octant][entry]` = 从 brick 内入口格 `entry` 出发、方向符号 = `octant` 的
/// 射线**可能经过**的子块集合（64-bit，bit i = 子块 `x + y*4 + z*16`）。
///
/// 条件：octant 分量正 → `p_i ≥ e_i − 1`，负 → `p_i ≤ e_i + 1`（±1 是浮点边界裕量，
/// 使掩码恒为真实可达集的**保守超集**）。故 shader 端 `occupancy & reach` 剔除绝不漏真实
/// 命中（不穿墙）。
///
/// 子块是否含实体的判定归 shader（本 LUT 只答"几何上能否经过"）：mask bit=1 = 分裂 ≠ 实体，
/// uniform 子块色 = 节点 palette，只有 palette==0（空气）节点才可用 `mask & reach` 剔除
/// （见 shaders/voxel_raytrace/ trace_chunk）。
pub fn march_mask_lut_words() -> Vec<u32> {
  let mut out = vec![0u32; MARCH_MASK_WORDS];
  for oct in 0..MARCH_MASK_OCTANTS {
    let pos = [oct & 1 != 0, (oct >> 1) & 1 != 0, (oct >> 2) & 1 != 0];
    for entry in 0..MARCH_MASK_ENTRIES {
      let e = [
        (entry & 3) as i32,
        ((entry >> 2) & 3) as i32,
        ((entry >> 4) & 3) as i32,
      ];
      let mut mask = 0u64;
      for p in 0..MARCH_MASK_ENTRIES {
        let q = [(p & 3) as i32, ((p >> 2) & 3) as i32, ((p >> 4) & 3) as i32];
        let ok = [0, 1, 2].iter().all(|&i| {
          if pos[i] {
            q[i] >= e[i] - 1
          } else {
            q[i] <= e[i] + 1
          }
        });
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

#[cfg(test)]
mod tests {
  use super::*;
  use gate_voxel::PaletteFlags;

  #[test]
  fn region_layout_matches_design() {
    assert_eq!(CHUNK_SIZE, 256);
    assert_eq!(BRICK_FACTOR, 4);
    assert_eq!(MAX_LEVEL, 4);
    assert_eq!(NODE_FIXED_WORDS, 3);
    assert_eq!(CHUNK_INDEX_CAP, 64);
    assert_eq!(CHUNK_INDEX_WORDS, 64 * 64 * 64);
    assert_eq!(TREE_BASE, CHUNK_INDEX_WORDS);
    assert_eq!(PALETTE_WORDS, 512);
    assert_eq!(CHUNK_COMP_WORDS, 2048);
    assert_eq!(STATE_TOTAL_WORDS * 4, 4096);
  }

  #[test]
  fn globals_layout_is_76_bytes() {
    // 与 WGSL Globals 字节兼容：19 个 u32 = 76B，
    // encase 写 UniformBuffer 时整体 round 到 16B 对齐（80B）
    let _ = BrickMapGlobals::default();
    assert_eq!(
      std::mem::size_of::<BrickMapGlobals>(),
      19 * 4,
      "19 个 u32 字段 = 76B"
    );
  }

  #[test]
  fn grid_desc_layout_is_144_bytes() {
    // 6×Vec4(96) + 4×u32(16) + 4×i32(16) + 4×u32(16) = 144B
    // storage buffer array stride = 144B（16B 对齐 ✓，std140 兼容）
    assert_eq!(std::mem::size_of::<GridDesc>(), 144, "GridDesc 必须 144B");
    // IDENTITY 常量健全
    let id = GridDesc::IDENTITY;
    assert_eq!(id.pos_scale.w, 1.0, "IDENTITY scale=1");
    assert_eq!(id.rot0.x, 1.0, "IDENTITY rot0.x=1");
    assert_eq!(id.tree_depth, MAX_LEVEL, "IDENTITY tree_depth=MAX_LEVEL");
  }

  #[test]
  fn grid_desc_from_transform_computes_aabb() {
    // scale 2 物体：局部 [0,256]³ → 世界 [0,512]³ @ origin (100,0,0)
    let g = GridDesc::from_transform(
      Vec3::new(100.0, 0.0, 0.0),
      Mat3::IDENTITY,
      2.0,
      100,
      200,
      1,
      IVec3::ZERO,
      IVec3::splat(1),
    );
    assert_eq!(g.pos_scale, Vec4::new(100.0, 0.0, 0.0, 2.0));
    // AABB = [100,0,0]..[612,512,512]
    assert!((g.aabb_min.truncate() - Vec3::new(100.0, 0.0, 0.0)).length() < 1e-4);
    assert!((g.aabb_max.truncate() - Vec3::new(612.0, 512.0, 512.0)).length() < 1e-4);
    assert_eq!(g.tree_base, 100);
    assert_eq!(g.palette_base, 200);
    assert_eq!(g.chunk_count, 1);
    assert_eq!(g.index_dims_x, 1);
  }

  #[test]
  fn palette_entry_pack_layout() {
    // _pad 私有，跨 crate 用 default + 逐字段赋值
    let mut e = PaletteEntry::default();
    e.color = [1, 2, 3];
    e.roughness = 4;
    e.emissive = 5;
    e.transmission = 6;
    e.flags = PaletteFlags(7);
    assert_eq!(
      pack_palette_entry(&e),
      [
        1 | (2 << 8) | (3 << 16) | (4 << 24),
        5 | (6 << 8) | (7 << 16)
      ]
    );
  }

  // ---- 方向可达掩码 LUT ----

  /// 读 LUT 单项（octant × entry → u64 掩码），布局与 shader 端
  /// `b_leaves[oct*128 + entry*2 ..]` 一致
  fn lut_get(lut: &[u32], oct: usize, entry: usize) -> u64 {
    let base =
      oct * MARCH_MASK_ENTRIES * MARCH_MASK_WORDS_PER_ENTRY + entry * MARCH_MASK_WORDS_PER_ENTRY;
    lut[base] as u64 | ((lut[base + 1] as u64) << 32)
  }

  fn entry_index(x: i32, y: i32, z: i32) -> usize {
    (z * 16 + y * 4 + x) as usize
  }

  #[test]
  fn march_mask_lut_size_and_self_reach() {
    let lut = march_mask_lut_words();
    assert_eq!(lut.len(), MARCH_MASK_WORDS);
    assert_eq!(MARCH_MASK_WORDS, 1024, "8 octant × 64 entry × 2 u32 = 4KB");
    // 入口格自身恒可达（p_i ≥ e_i − 1 含 p=e；p_i ≤ e_i + 1 同理）
    for oct in 0..MARCH_MASK_OCTANTS {
      for entry in 0..MARCH_MASK_ENTRIES {
        assert_ne!(
          lut_get(&lut, oct, entry) & (1u64 << entry),
          0,
          "oct={oct} entry={entry} 自身必须可达"
        );
      }
    }
  }

  #[test]
  fn march_mask_lut_corner_entries() {
    let lut = march_mask_lut_words();
    // octant 7 = (+,+,+)（bit1=正）。entry (0,0,0) → p_i ≥ −1 恒真 → 全 64 格可达
    assert_eq!(lut_get(&lut, 7, entry_index(0, 0, 0)), u64::MAX);
    // entry (3,3,3) (+,+,+) → p_i ≥ 2 → 恰 {2,3}³ 8 格
    let mut want = 0u64;
    for z in 2..4 {
      for y in 2..4 {
        for x in 2..4 {
          want |= 1u64 << entry_index(x, y, z);
        }
      }
    }
    assert_eq!(lut_get(&lut, 7, entry_index(3, 3, 3)), want);
    // octant 0 = (−,−,−)，entry (0,0,0) → p_i ≤ 1 → {0,1}³ 8 格
    let mut want_lo = 0u64;
    for z in 0..2 {
      for y in 0..2 {
        for x in 0..2 {
          want_lo |= 1u64 << entry_index(x, y, z);
        }
      }
    }
    assert_eq!(lut_get(&lut, 0, entry_index(0, 0, 0)), want_lo);
  }

  #[test]
  fn march_mask_lut_monotone_and_bounded() {
    let lut = march_mask_lut_words();
    // 单调性：+轴入口沿正向推进 → 条件 p_i ≥ e_i−1 收紧 → 掩码缩小（子集）；
    // −轴条件 p_i ≤ e_i+1 放宽 → 掩码扩大（超集）。按轴符号分别验证。
    for oct in 0..MARCH_MASK_OCTANTS {
      let x_pos = oct & 1 != 0;
      for y in 0..4 {
        for z in 0..4 {
          let mut prev = if x_pos { u64::MAX } else { 0u64 };
          for x in 0..4 {
            let m = lut_get(&lut, oct, entry_index(x, y, z));
            if x_pos {
              assert_eq!(m & prev, m, "oct={oct} +x 入口未单调缩小 @ ({x},{y},{z})");
            } else {
              assert_eq!(m | prev, m, "oct={oct} −x 入口未单调扩大 @ ({x},{y},{z})");
            }
            prev = m;
          }
        }
      }
    }
    // 保守下界：任一 (oct, entry) 可达集 ≥ 8 格（每轴至少 {e−1..3} 或 {0..e+1} 取 2 格）
    for oct in 0..MARCH_MASK_OCTANTS {
      for entry in 0..MARCH_MASK_ENTRIES {
        assert!(
          lut_get(&lut, oct, entry).count_ones() >= 8,
          "oct={oct} entry={entry} 可达集异常小（过激进剔除风险）"
        );
      }
    }
  }

  #[test]
  fn march_mask_lut_axis_aligned_exact() {
    // 纯 +x 方向（oct bit0=1, bit1/2=0 → 符号 (+,−,−)）从 (0,y,z) 出发的精确可达集：
    // x 轴向走遍 0..4，y/z 向 ≤ e+1 裕量内。抽查 entry (0, 3, 3)：
    // 可达 = 全 x × y ≤ 3（恒真）× z ≤ 3（恒真）→ 全 64 格
    let lut = march_mask_lut_words();
    // (+,−,−) = bit0=1, bit1=0, bit2=0 → oct 1
    assert_eq!(lut_get(&lut, 1, entry_index(0, 3, 3)), u64::MAX);
    // 同方向 entry (3, 0, 0)：x ≥ 2 且 y ≤ 1 且 z ≤ 1 → 2×2×2 = 8 格
    let mut want = 0u64;
    for z in 0..2 {
      for y in 0..2 {
        for x in 2..4 {
          want |= 1u64 << entry_index(x, y, z);
        }
      }
    }
    assert_eq!(lut_get(&lut, 1, entry_index(3, 0, 0)), want);
  }
}
