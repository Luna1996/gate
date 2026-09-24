//! gate-voxel：权威体素数据层 + Douglas Brick Tree。
//! 纯逻辑 crate，零渲染依赖，唯一外部依赖 `glam`。
//! Brick Tree = 4³ 分裂因子 + u64 mask + 紧凑 child offset + 自适应 uniform leaf。

pub mod chunk_tree;
pub mod coords;
pub mod dirty;
pub mod palette;
pub mod produce;
pub mod scene;
pub mod volume;

pub use coords::{
  BRICK_FACTOR, BrickCoord, CHUNK_SIZE, ChunkCoord, LEVEL_EXTENT, MAX_LEVEL, VoxelCoord,
  child_linear_idx,
};

pub use chunk_tree::{
  BrickState, ChunkTree, LEAF_INLINE_WORDS, LEAF_VOXELS_PER_WORD, NODE_OFFSET_NONE, NodeDesc,
  NodeLayout, NodeView, ROOT_WIRE_WORDS, TreeDirty, leaf_rep_palette, pack_palette_word,
};

pub use dirty::DirtyTracker;

pub use produce::{ChunkProducer, ChunkSource, Detail};

pub use volume::{
  COMP_BRICK_EXTENT, COMP_BRICKS_PER_CHUNK, DirtyEdit, VolumeGrid, VolumeTransform, Volumes,
};

pub use palette::{
  PALETTE_BITS, PALETTE_ENTRY_COUNT, PALETTE_INDEX_MAX, Palette, PaletteEntry, PaletteFlags,
  PaletteId, PbrOverrides, inverted_pct_to_override, override_byte, override_value,
  slider_to_override,
};

pub use scene::{
  Displace, DisplaceFn, FillStats, draw_text, fill_box, fill_box_displaced, fill_bricks,
  fill_bricks_displaced, fill_sphere, fill_sphere_displaced, text_size,
};
