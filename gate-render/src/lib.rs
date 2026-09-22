pub mod brickmap;
pub mod consts;
pub mod gi;
pub mod lighting;
pub mod paths;
pub mod pbr_texture;
pub mod profiler;
mod responsive;
pub mod shader;
pub mod sky;
pub mod volumetric;
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
pub use pbr_texture::{PBR_TEXTURE_DIR, PbrTextureSet, PbrTexturesPlugin};
pub use responsive::{ResponsivePlugin, resize_render_targets};
pub use sky::{SkyPlugin, SkySettings, sun_altitude_deg};
pub use volumetric::{FogPlugin, FogSettings};

pub struct GateRenderPlugin;

impl Plugin for GateRenderPlugin {
  fn build(&self, app: &mut App) {
    app.add_plugins((
      brickmap::upload::VolumePlugin,
      brickmap::dda::BrickMapDdaPlugin,
      gi::GiPlugin,
      // 径向模糊（光柱）：屏幕空间径向模糊的 pass / 资源 / 档位（菜单「渲染/天空/径向模糊」）
      volumetric::FogPlugin,
      // 天象（时间 → 太阳 / 月亮 / 天空色）：每帧把推导结果写进 `LightingTheme`（菜单「渲染/天空」）
      sky::SkyPlugin,
      profiler::GateProfilerPlugin,
      // PBR 贴图集（MT2-1）：扫描 assets/textures/pbr/ → 两张 texture_2d_array（只加载，不绑定）
      pbr_texture::PbrTexturesPlugin,
    ));
  }
}
