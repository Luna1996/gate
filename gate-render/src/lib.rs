pub mod brickmap;
pub mod consts;
pub mod gi;
pub mod lighting;
pub mod paths;
pub mod profiler;
mod responsive;
pub mod shader;
pub mod wesl_consts;

use bevy::prelude::*;

pub use brickmap::{
  BindingLimits, BrickMapBuffers, BrickMapBuilder, BrickMapGlobals, BrickMapView, BufferLayout,
  BuilderMirror, ChunkUpdate, DdaCameraConfig, DdaImages, DdaViewUniform, DebugNormals,
  DirtyRanges, EyeAdaptSettings, GpuBrickMap, GridDesc, OrbitCamera, UploadBudget, UploadCpuSample,
  UploadCpuSampleChannel, UploadSnapshot, VolumeHit, VolumesBuilder, VolumesSnapshot, VoxelScene,
  cpu_dda_ascii_grid_32x32, cpu_reference_dda_ray, cpu_reference_dda_ray_aabb_skip,
  cpu_reference_dda_ray_two_level, cpu_reference_trace_volumes, cpu_reference_volumes_occluded,
  create_dda_image,
};
pub use brickmap::{PostFxSettings, RenderScale};
pub use consts::VIEW_SIZE;
pub use lighting::{
  DirLightCfg, LightDesc, LightGlobals, LightPoolUniform, LightingTheme, MAX_LIGHTS, SkyCfg,
  build_light_pool, parse_lighting_ron,
};
pub use paths::{assets_dir, data_dir, dda_wesl_dir, install_root, logs_dir};
pub use responsive::{ResponsivePlugin, resize_render_targets};

pub struct GateRenderPlugin;

impl Plugin for GateRenderPlugin {
  fn build(&self, app: &mut App) {
    app.add_plugins((
      brickmap::upload::VolumePlugin,
      brickmap::dda::BrickMapDdaPlugin,
      gi::GiPlugin,
      profiler::GateProfilerPlugin,
    ));
  }
}
