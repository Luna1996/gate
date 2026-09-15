//! gate-voxel：权威体素数据层 + Douglas Brick Tree。
//!
//! 纯逻辑 crate，零渲染依赖，唯一外部依赖 `glam`。
//! Brick Tree = 4³ 分裂因子 + u64 mask + 紧凑 child offset + 自适应 uniform leaf。

pub mod chunk_tree;
pub mod coords;
pub mod dirty;
pub mod palette;
pub mod scene;
pub mod volume;

// ============ 核心坐标 ============
pub use coords::{
  BRICK_FACTOR, BrickCoord, CHUNK_SIZE, ChunkCoord, LEVEL_EXTENT, MAX_LEVEL, VoxelCoord,
  child_linear_idx,
};

// ============ Brick Tree ============
pub use chunk_tree::{BrickState, ChunkTree};

// ============ 脏追踪 ============
pub use dirty::DirtyTracker;

// ============ Volume 容器 ============
pub use volume::{
  COMP_BRICK_EXTENT, COMP_BRICKS_PER_CHUNK, DirtyEdit, VolumeGrid, VolumeTransform, Volumes,
};

// ============ 调色板 ============
pub use palette::{AIR_INDEX, Palette, PaletteEntry, PaletteFlags};

// ============ 场景构造 ============
pub use scene::{draw_text, fill_box, fill_bricks, fill_sphere, text_size};
