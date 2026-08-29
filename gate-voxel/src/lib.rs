//! gate-voxel：权威体素数据层 + 洪泛 + 模拟（纯逻辑，零渲染依赖）

pub mod coords;
pub mod dirty;
pub mod grid;
pub mod palette;
pub mod scene;
pub mod tile;

#[cfg(test)]
mod stress;

pub use coords::{Level, MAX_LEVEL, TILE_CELLS, TILE_SUB, TileCoord, VoxelPos};
pub use dirty::DirtyTracker;
pub use grid::{DirtyEdit, TileGrid};
pub use palette::{AIR_INDEX, Palette, PaletteEntry, PaletteFlags};
pub use scene::{draw_text, fill_box, fill_sphere, text_size};
pub use tile::{Brick, Cell, Slot, Tile};
