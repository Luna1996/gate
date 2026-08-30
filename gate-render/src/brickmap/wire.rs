//! 砖块图 wire 格式：常量、编码函数、全局参数（docs/brickmap.md 的实现契约）
//!
//! 本模块是 CPU 构建器（builder.rs）与 GPU shader（P2.4）之间的字节契约。
//! 纯数据变换、零渲染依赖，可脱离渲染运行时单测。

use gate_voxel::PaletteEntry;

/// Tile 池上限：TileBitmaps / CellDirs 的槽位数（预算表 §7 的 128MB 恒定项来源）
pub const TILE_CAP: usize = 1024;
/// 稠密 TileIndex 每轴上限（128³ = 2M 条目 = 8MB）
pub const TILE_INDEX_CAP: i32 = 128;
/// Tile 内基元胞数：32³
pub const CELLS_PER_TILE: usize = 32 * 32 * 32;
/// TileBitmaps 每槽字数（32768 bit / 32）
pub const TILE_BITMAP_WORDS: usize = CELLS_PER_TILE / 32;
/// CellDirs 每槽字数（每基元胞 1 u32，直寻）
pub const CELL_DIR_WORDS: usize = CELLS_PER_TILE;
/// BrickPool 每 slab 字数（4096 字节调色板）
pub const BRICK_SLAB_WORDS: usize = 4096 / 4;
/// 调色板字数（256 条 × 2 u32）
pub const PALETTE_WORDS: usize = 256 * 2;

/// Region ① TileIndex：定长 128³ 条目
pub const INDEX_WORDS: usize = 128 * 128 * 128;
/// Region ② TileBitmaps：TILE_CAP 槽 × 占用位图
pub const BITMAP_REGION_WORDS: usize = TILE_CAP * TILE_BITMAP_WORDS;
/// Region ③ CellDirs：TILE_CAP 槽 × 直寻目录
pub const DIR_REGION_WORDS: usize = TILE_CAP * CELL_DIR_WORDS;
/// Region ④ NodeStream 起始字偏移（b_struct 定长前缀 ①+②+③ ≈ 140MB）
pub const NODE_STREAM_BASE: usize = INDEX_WORDS + BITMAP_REGION_WORDS + DIR_REGION_WORDS;

/// Region ② 起始字偏移（TileIndex 之后）
pub const BITMAP_BASE: usize = INDEX_WORDS;
/// Region ③ 起始字偏移（TileIndex + TileBitmaps 之后）
pub const DIR_BASE: usize = INDEX_WORDS + BITMAP_REGION_WORDS;

/// 每层级槽表的字数（L1 8 槽 / L2 64 槽 / L3 512 槽，u16 两两打包；L0/L4 无槽表）
pub const SLOT_TABLE_WORDS: [usize; 5] = [0, 4, 32, 256, 0];

// --- comp_layer / StateTable wire 常量（P2.3 元件层通道）---

/// comp_layer 每 tile 字数（u16[32768] → 每 2 字打包成 u32 = 32768 / 2 = 16384）
pub const TILE_COMP_WORDS: usize = CELLS_PER_TILE / 2;
/// StateTable 条目的字数（4×u32/条目）
pub const STATE_WORDS_PER_ENTRY: usize = 4;
/// StateTable MVP 条目数（u16 组件 ID 低 8bit）
pub const STATE_ENTRY_COUNT: usize = 256;
/// StateTable 总字数（256×4 = 1024 = 4KB）
pub const STATE_TOTAL_WORDS: usize = STATE_ENTRY_COUNT * STATE_WORDS_PER_ENTRY;

/// CellNode hdr：bits 0-7 uniform palette（0 = 非 uniform，AIR 恰作哨兵）
pub const HDR_UNIFORM_MASK: u32 = 0xff;
/// CellNode hdr：表链存在标志（表按 l1 → l2 → l3 → brick 定序排列，无内部指针）
pub const HDR_HAS_L1: u32 = 1 << 8;
pub const HDR_HAS_L2: u32 = 1 << 9;
pub const HDR_HAS_L3: u32 = 1 << 10;
pub const HDR_HAS_BRICK: u32 = 1 << 11;

/// u16 槽 tag（bits 8-15）
pub const SLOT_TAG_EMPTY: u16 = 0;
pub const SLOT_TAG_LEAF: u16 = 1;
pub const SLOT_TAG_BRANCH: u16 = 2;

/// 编码一个槽：tag 进高位，palette 进低位
pub const fn encode_slot(tag: u16, palette: u8) -> u16 {
  (tag << 8) | palette as u16
}

/// 两个 u16 槽打包进一个 u32（低 16 位 = 偶数槽）
pub const fn pack_slot_pair(lo: u16, hi: u16) -> u32 {
  lo as u32 | (hi as u32) << 16
}

/// 取槽字中的第 lane（0/1）个槽
pub const fn unpack_slot_word(word: u32, lane: usize) -> u16 {
  (word >> (16 * lane)) as u16
}

/// 槽 tag 提取
pub const fn slot_tag(slot: u16) -> u16 {
  slot >> 8
}

/// 槽 palette 提取
pub const fn slot_palette(slot: u16) -> u8 {
  slot as u8
}

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

/// GPU 全局参数（P2.3 进 uniform buffer）。
///
/// encase 0.12.1 在 uniform 模式下对 Rust fixed-size `[i32/u32; N]` 断言
/// "array stride must be a multiple of 16"（按 element stride=4 判，而非按 array
/// stride=16），所以三轴字段全部**拆成具名 scalar**（x/y/z/w），尾部填充同理。
/// 整体字节数 = 16+16+(7×4)+(5×4) = 32+28+20 = 80B（std140 允许末尾非 16 对齐，
/// encase 会在写 UniformBuffer 时把整体尺寸 round 到 16B 对齐；CPU 侧字节保持
/// 80B 独立契约，需要时写死 assert）。
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, bevy::render::render_resource::ShaderType)]
pub struct BrickMapGlobals {
  pub index_origin_x: i32,
  pub index_origin_y: i32,
  pub index_origin_z: i32,
  pub index_origin_w: i32,
  pub index_dims_x: u32,
  pub index_dims_y: u32,
  pub index_dims_z: u32,
  pub index_dims_w: u32,
  pub tile_count: u32,
  pub node_words: u32,
  pub node_free_words: u32,
  pub brick_slabs: u32,
  pub brick_free: u32,
  pub rejected_tiles: u32,
  pub _pad0: u32,
  pub _pad1: u32,
  pub _pad2: u32,
  pub _pad3: u32,
  pub _pad4: u32,
}

/// 构建产物：与 GPU buffer 字节一一对应的内容（P2.3 原样上传）
///
/// b_struct = 定长前缀（index 8MB + bitmaps 4MB + dirs 128MB）+ NodeStream 尾部。
/// b_leaves 的 slab i = `[i*1024, (i+1)*1024)`（P2.3 上传到预分配池的同号 slab）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BrickMapBuffers {
  pub b_struct: Vec<u32>,
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
    assert_eq!(CELLS_PER_TILE, 32768);
    assert_eq!(TILE_BITMAP_WORDS, 1024);
    assert_eq!(CELL_DIR_WORDS, 32768);
    assert_eq!(BRICK_SLAB_WORDS, 1024);
    assert_eq!(PALETTE_WORDS, 512);
    assert_eq!(INDEX_WORDS, 128 * 128 * 128);
    assert_eq!(BITMAP_REGION_WORDS, TILE_CAP * TILE_BITMAP_WORDS);
    assert_eq!(DIR_REGION_WORDS, TILE_CAP * CELL_DIR_WORDS);
    // 定长前缀 ≈ 140MB（预算表 §7 恒定项）：index 8MB + bitmaps 4MB + dirs 128MB
    assert_eq!(NODE_STREAM_BASE, 36_700_160);
    assert_eq!(BITMAP_BASE, INDEX_WORDS);
    assert_eq!(DIR_BASE, INDEX_WORDS + BITMAP_REGION_WORDS);
    assert_eq!(SLOT_TABLE_WORDS, [0, 4, 32, 256, 0]);
  }

  #[test]
  fn slot_encoding_roundtrip() {
    let w = pack_slot_pair(
      encode_slot(SLOT_TAG_LEAF, 0xab),
      encode_slot(SLOT_TAG_BRANCH, 0),
    );
    assert_eq!(unpack_slot_word(w, 0), encode_slot(SLOT_TAG_LEAF, 0xab));
    assert_eq!(unpack_slot_word(w, 1), encode_slot(SLOT_TAG_BRANCH, 0));
    assert_eq!(slot_tag(encode_slot(SLOT_TAG_LEAF, 7)), SLOT_TAG_LEAF);
    assert_eq!(slot_palette(encode_slot(SLOT_TAG_LEAF, 7)), 7);
    assert_eq!(slot_tag(encode_slot(SLOT_TAG_EMPTY, 0)), SLOT_TAG_EMPTY);
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
