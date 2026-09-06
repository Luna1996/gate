pub mod brickmap;
pub mod ddgi;
pub mod lighting;
mod responsive;

use bevy::prelude::*;

pub use brickmap::{
  BindingLimits, BrickMapBuffers, BrickMapBuilder, BrickMapGlobals, BrickMapView, BufferLayout,
  BuilderMirror, ChunkUpdate, DdaCameraConfig, DdaImages, DdaViewUniform, DebugNormals,
  DirtyRanges, GpuBrickMap, GridDesc, OrbitCamera, UploadBudget, UploadCpuSample,
  UploadCpuSampleChannel, UploadSnapshot, VolumeHit, VolumesBuilder, VolumesSnapshot, VoxelScene,
  cpu_dda_ascii_grid_32x32, cpu_reference_dda_ray, cpu_reference_dda_ray_aabb_skip,
  cpu_reference_dda_ray_two_level, cpu_reference_trace_volumes, cpu_reference_volumes_occluded,
  create_dda_image,
};
pub use brickmap::{RenderScale, VIEW_SIZE};
pub use lighting::{
  DirLightCfg, EMISSIVE_EMIT_GAIN, LightDesc, LightGlobals, LightPoolUniform, LightingTheme,
  MAX_LIGHTS, SHADOW_BIAS, SHADOW_DIR_T_MAX, SkyCfg, build_light_pool, parse_lighting_ron,
};
pub use responsive::{ResponsivePlugin, resize_render_targets};

pub struct GateRenderPlugin;

impl Plugin for GateRenderPlugin {
  fn build(&self, app: &mut App) {
    // Devlog 23：DDGI 探针 pass 已下线（ddgi 模块保留——数学/布局供下一代 DDGI 复用，
    // 重新接线时再挂回）；octo hashmap 光照链（vis_cache/direct/gi/denoise）已整体拆除。
    app.add_plugins((
      brickmap::upload::VolumePlugin,
      brickmap::dda::BrickMapDdaPlugin,
    ));
  }
}
