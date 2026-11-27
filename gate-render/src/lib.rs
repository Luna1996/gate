pub mod brickmap;
pub mod cloud_texture;
pub mod consts;
#[cfg(feature = "dlss")]
pub mod dlss;
pub mod gi;
pub mod lighting;
pub mod mc_texture;
pub mod paths;
pub mod pbr_texture;
pub mod pipe_cache;
pub mod profiler;
mod responsive;
pub mod shader;
pub mod sky;
pub mod volumetric;
pub mod wesl_consts;

use bevy::prelude::*;

pub use brickmap::dda::{DlssMode, DlssSettings};
pub use brickmap::{
  BindingLimits, BrickMapBuffers, BrickMapBuilder, BrickMapGlobals, BufferLayout, BuilderMirror,
  ChunkUpdate, DdaCameraConfig, DdaImages, DdaViewUniform, DirtyRanges, EyeAdaptSettings,
  GpuBrickMap, GridDesc, OrbitCamera, RayHit, UploadBudget, UploadCpuSample,
  UploadCpuSampleChannel, UploadSnapshot, VolumesBuilder, VolumesSnapshot, VoxelDumpRequest,
  VoxelScene, create_dda_image, pool_capacity_chunks, pool_capacity_chunks_far, raycast,
  raycast_objects,
};
pub use brickmap::{PostFxSettings, RenderScale, TonemapMode};
pub use cloud_texture::{CloudTexturePlugin, CloudTextureSet};
pub use consts::VIEW_SIZE;
pub use lighting::{
  AtmosphereSettings, BaseSettings, CloudSettings, DirLightCfg, LightDesc, LightGlobals,
  LightPoolUniform, LightingTheme, MAX_LIGHTS, ReflectionSettings, SkyCfg, build_light_pool,
  parse_lighting_ron,
};
pub use mc_texture::{
  MC_ASSET_BASE, MC_TEX_MIP_LEVELS, MC_TEX_SIZE, McTextureSet, texture_array_image,
};
pub use paths::{assets_dir, data_dir, dda_wesl_dir, install_root, logs_dir};
pub use pbr_texture::{PBR_TEXTURE_DIR, PbrTextureSet, PbrTexturesPlugin, material_ids};
pub use profiler::{ChunkUseFeed, LodRequest, LodRequestFeed, LodUse};
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
      volumetric::FogPlugin,
      sky::SkyPlugin,
      profiler::GateProfilerPlugin,
      pbr_texture::PbrTexturesPlugin,
      cloud_texture::CloudTexturePlugin,
    ));
    #[cfg(feature = "dlss")]
    app.add_plugins(dlss::DlssRenderPlugin);
  }
}
