//! 砖块图 CPU 侧：wire 字节契约 + 构建器 + 软件遍历器 + DDA pass。
//! [`wire`] 是 CPU 构建器与 GPU shader 的字节级契约；[`BrickMapView`] 独立实现寻址链。

mod builder;
pub mod consts;
pub mod dda;
pub mod upload;
mod view;
pub mod wire;

pub use builder::{BrickMapBuilder, ChunkUpdate, DirtyRanges, VolumesBuilder, VolumesSnapshot};
pub use dda::{
  DdaCameraConfig, DdaHit, DdaImages, DdaViewUniform, DebugNormals, EyeAdaptSettings, OrbitCamera,
  PostFxSettings, RenderScale, TreeHit, VolumeHit, cpu_dda_ascii_grid_32x32,
  cpu_reference_dda_ray, cpu_reference_dda_ray_aabb_skip, cpu_reference_dda_ray_tree,
  cpu_reference_dda_ray_two_level, cpu_reference_trace_volumes, cpu_reference_volumes_occluded,
  create_dda_image,
};
pub use upload::{
  BindingLimits, BrickMapRevision, BufferLayout, BuilderMirror, GpuBrickMap, UploadBudget,
  UploadCpuSample, UploadCpuSampleChannel, UploadSnapshot, VolumePlugin, VoxelScene,
};
pub use view::BrickMapView;
pub use wire::{BrickMapBuffers, BrickMapGlobals, GridDesc};
