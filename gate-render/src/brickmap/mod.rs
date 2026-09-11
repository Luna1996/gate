//! 砖块图 CPU 侧（Phase 1，Douglas 1:1）：wire 字节契约 + 构建器 + 软件遍历器 + DDA
//!
//! - [`wire`]：CPU 构建器与 GPU shader（Phase 2 重写）之间的字节级契约，纯数据零渲染依赖
//! - [`BrickMapBuilder`]：`VolumeGrid` → wire 格式；全量构建（Rayon 分批）+ 逐 chunk 增量（append-only）
//! - [`BrickMapView`]：软件遍历器，独立实现寻址链，等价性测试互为对照
//! - [`dda`]：DDA 主可见性 pass（WGSL compute + Core2d blit 上屏）

mod builder;
pub mod dda;
pub mod upload;
mod view;
pub mod wire;

pub use builder::{BrickMapBuilder, ChunkUpdate, DirtyRanges, VolumesBuilder, VolumesSnapshot};
pub use dda::{
  DdaCameraConfig, DdaHit, DdaImages, DdaViewUniform, DebugNormals, OrbitCamera, RenderScale,
  TreeHit, VIEW_SIZE, VolumeHit, cpu_dda_ascii_grid_32x32, cpu_reference_dda_ray,
  cpu_reference_dda_ray_aabb_skip, cpu_reference_dda_ray_tree, cpu_reference_dda_ray_two_level,
  cpu_reference_trace_volumes, cpu_reference_volumes_occluded, create_dda_image,
};
pub use upload::{
  BindingLimits, BrickMapRevision, BufferLayout, BuilderMirror, GpuBrickMap, UploadBudget,
  UploadCpuSample, UploadCpuSampleChannel, UploadSnapshot, VolumePlugin, VoxelScene,
};
pub use view::BrickMapView;
pub use wire::{BrickMapBuffers, BrickMapGlobals, GridDesc};
