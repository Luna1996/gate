//! Brick Tree wire 格式：常量、编码函数、全局参数（Phase 1，Douglas 1:1）
//!
//! 本模块是 CPU 构建器（builder.rs）与 GPU shader（Phase 2 重写 dda.wgsl）之间的
//! 字节契约。纯数据变换、零渲染依赖，可脱离渲染运行时单测。
//!
//! ## b_struct 布局（Douglas Brick Tree 紧凑 DFS 格式）
//!
//! ```text
//! b_struct:
//! ├── [0 .. CHUNK_INDEX_WORDS)         稠密 chunk 窗口（CHUNK_INDEX_CAP³）
//! │     entry = chunk DFS 树绝对字基址 + 1（0 = 无此 chunk）
//! └── [CHUNK_INDEX_WORDS ..)           各 chunk 的 DFS 序列化树（append bump）
//!       每 chunk 树（ChunkTree::serialize() 原样）：
//!       node = [mask_lo, mask_hi, palette_u32] + popcount(mask) 个 child offset
//!       child offset = chunk 内相对字址（shader 端加 chunk base 转绝对）
//! ```
//!
//! mask bit=1 → 子块被分裂（child offset 有效）；bit=0 → uniform 子块，
//! 颜色 = 该节点 palette_u32（零额外 load）。palette=0 = AIR。
//!
//! b_leaves 已删除（Douglas 格式 palette 直存节点 fixed 字）——字段保留空 Vec
//! 以维持 GpuBrickMap/obj 打包结构稳定，Phase 2 shader 重写时移除。

use gate_voxel::PaletteEntry;
use glam::{IVec3, Mat3, Vec3, Vec4};

// ============ 层级常量（与 gate-voxel coords.rs 一致）============

/// chunk 边长（体素/fine 单位）：256³（Douglas 早期 chunk 大小）
pub const CHUNK_SIZE: i32 = 256;
/// 分裂因子（4³ = 64 子块）
pub const BRICK_FACTOR: i32 = 4;
/// 最大分裂深度：256 → 64 → 16 → 4 → 1
pub const MAX_LEVEL: u32 = 4;
/// 每 chunk 64 子块的节点 fixed 字数（mask_lo + mask_hi + palette）
pub const NODE_FIXED_WORDS: usize = 3;

// ============ b_struct Region ①：稠密 chunk 窗口 ============

/// 稠密 chunk 窗口每轴上限（64 chunk × 256 = 16384 fine 覆盖半径）
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

// ============ PaletteEntry 打包（不变）============

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
// **字段名与旧版逐字相同（80B 布局不变）**——Phase 1 保持 dda.wgsl 的 Globals
// struct 字节兼容（shader 逻辑 Phase 2 重写，本阶段画面为空属预期）。
// 语义升级：index_origin/dims 从 tile 窗口（×512 fine）变为 **chunk 窗口**
// （×256 fine）；tile_count 语义变为 chunk_count。

/// encase 0.12.1 在 uniform 模式下对 Rust fixed-size `[i32/u32; N]` 断言
/// "array stride must be a multiple of 16"（按 element stride=4 判，而非按 array
/// stride=16），所以三轴字段全部**拆成具名 scalar**（x/y/z/w），尾部填充同理。
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
  /// 有内容的 chunk 数（旧 tile_count 字段名保留 = WGSL 字节兼容）
  pub tile_count: u32,
  /// b_struct 树区字数（不含 Region ① chunk 窗口）
  pub node_words: u32,
  /// 作废树区累计字数（增量 append 后未压缩的旧字节；全量重建归零）
  pub node_free_words: u32,
  /// 0（b_leaves 已删除；字段保留 = WGSL 字节兼容）
  pub brick_slabs: u32,
  pub brick_free: u32,
  /// 超出 chunk 窗口被拒绝的 chunk 数
  pub rejected_tiles: u32,
  pub _pad0: u32,
  pub _pad1: u32,
  pub _pad2: u32,
  pub _pad3: u32,
  pub _pad4: u32,
}

// ============ GridDesc（Phase 2+3 统一描述符，144B）============
//
// 替代 BrickMapGlobals + ObjDesc：主世界与物体走同一 GridDesc 数组，
// shader `trace_scene` 遍历数组无 kind 分支（§2.6 计划扩展为 144B 含 chunk 窗口）。
//
// 字段语义：
// - pos_scale/rot0/rot1/rot2：`world = pos + rot · (local · scale)`，
//   rot 列向量与 glam Mat3 一致（x_axis/y_axis/z_axis = 列）；主世界 = identity。
// - aabb_min/max：局部 [0,256]³·scale 经变换后的世界外包盒（CPU 预算，剔除用）。
// - tree_base：本 volume 的 b_struct 在 struct_buf 内的字基址（含 chunk 窗口段）。
// - tree_depth：Douglas Brick Tree 最大分裂深度 = 4（256→64→16→4→1）。
// - chunk_count：本 volume 的 chunk 数（主世界可能 N，物体 v1=1）。
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

/// 局部 [0,256]³·scale 经旋转平移后的世界 AABB（与 obj.rs 旧 world_aabb 同型）
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

/// 构建产物：与 GPU buffer 字节一一对应的内容（P2.3 原样上传）
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BrickMapBuffers {
  /// Region ① chunk 窗口 + Region ② 各 chunk DFS 树
  pub b_struct: Vec<u32>,
  /// 恒空（Douglas 格式无 leaves buffer；字段保留维持打包结构稳定）
  pub b_leaves: Vec<u32>,
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
    // WGSL Globals 字节兼容（Phase 1 shader 不重写）：19 个 u32 = 76B，
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
    // Phase 2+3 统一描述符：6×Vec4(96) + 4×u32(16) + 4×i32(16) + 4×u32(16) = 144B
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
}
