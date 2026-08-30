//! 砖块图 CPU 侧（P2.2）：wire 字节契约 + 构建器 + 软件遍历器 + DDA（P2.4）
//!
//! - [`wire`]：CPU 构建器与 GPU shader（P2.4）之间的字节级契约，纯数据零渲染依赖
//! - [`BrickMapBuilder`]：`TileGrid` → wire 格式；全量构建（Rayon 分批）+ 逐 Tile 增量（§5 协议）
//! - [`BrickMapView`]：软件遍历器，独立实现寻址链，等价性测试互为对照
//! - [`dda`]：P2.4 DDA 主可见性 pass（WGSL compute + Core2d blit 上屏）

mod builder;
pub mod dda;
pub mod upload;
mod view;
pub mod wire;

pub use builder::{BrickMapBuilder, TileUpdate};
pub use dda::{
  DdaCameraConfig, DdaImages, DdaViewUniform, OrbitCamera, cpu_dda_ascii_grid_32x32,
  cpu_reference_dda_ray, cpu_reference_dda_ray_aabb_skip, create_dda_image,
};
pub use upload::{
  BindingLimits, BrickMapUploadPlugin, BufferLayout, BuilderMirror, GpuBrickMap, UploadBudget,
  UploadCpuSample, UploadCpuSampleChannel, UploadSnapshot, VoxelScene,
};
pub use view::BrickMapView;
pub use wire::{BrickMapBuffers, BrickMapGlobals};
