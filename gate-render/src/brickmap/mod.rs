//! 砖块图 CPU 侧：wire 字节契约 + 构建器 + DDA pass。
//! [`wire`] 是 CPU 构建器与 GPU shader 的字节级契约；[`raytrace`] 是 CPU 侧唯一的射线求交入口
//! （跑在权威 `VolumeGrid` 上，不经序列化）。

mod builder;
pub mod consts;
pub mod dda;
pub mod raytrace;
pub mod residency;
pub mod upload;
pub mod wire;

pub use builder::{BrickMapBuilder, ChunkUpdate, DirtyRanges, VolumesBuilder, VolumesSnapshot};
pub use dda::{
  DdaCameraConfig, DdaImages, DdaViewUniform, DebugNormals, EyeAdaptSettings, OrbitCamera,
  PostFxSettings, RenderScale, create_dda_image,
};
pub use raytrace::{RayHit, raycast};
pub use upload::{
  BindingLimits, BrickMapRevision, BufferLayout, BuilderMirror, GpuBrickMap, UploadBudget,
  UploadCpuSample, UploadCpuSampleChannel, UploadSnapshot, VolumePlugin, VoxelDumpRequest,
  VoxelScene,
};
pub use wire::{BrickMapBuffers, BrickMapGlobals, GridDesc};
