use bevy::{
  asset::RenderAssetUsages,
  image::Image,
  prelude::*,
  render::{extract_resource::ExtractResource, render_resource::*},
};
use glam::camera::{rh::proj, rh::view};
use std::ops::Mul;

pub const BLIT_SHADER_ASSET_PATH: &str = "shaders/blit.wgsl";

pub const DDA_WORKGROUP_SIZE: u32 = 8;

#[derive(Resource, Clone, Copy, Debug, PartialEq, ExtractResource)]
#[extract_app(bevy::render::RenderApp)]
pub struct RenderScale {
  pub size: UVec2,

  pub factor: u32,
}

impl RenderScale {
  pub const SCALE_CHOICES: [u32; 4] = [1, 2, 3, 4];
}

impl Default for RenderScale {
  fn default() -> Self {
    Self { size: crate::consts::VIEW_SIZE, factor: 1 }
  }
}

#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq, ExtractResource)]
#[extract_app(bevy::render::RenderApp)]
pub struct PostFxSettings {
  pub fxaa: bool,
}

#[derive(Resource, Clone, Copy)]
pub struct DdaCameraConfig {
  pub view_proj: Mat4,
  pub inv_view_proj: Mat4,
  pub position_world: Vec3,

  pub forward: Vec3,
}

impl DdaCameraConfig {
  pub fn build_static() -> Self {
    let eye = Vec3::new(700.0, 560.0, 700.0);
    let target = Vec3::new(260.0, 120.0, 260.0);
    let up = Vec3::Y;
    let aspect = crate::consts::VIEW_SIZE.x as f32 / crate::consts::VIEW_SIZE.y as f32;
    let fovy = 60.0_f32.to_radians();
    let near = 1.0;
    let far = 4000.0;
    let proj = proj::directx::perspective(fovy, aspect, near, far);
    let view = view::look_at_mat4(eye, target, up);
    let view_proj = proj.mul(view);
    let inv_view_proj = view_proj.inverse();
    Self { view_proj, inv_view_proj, position_world: eye, forward: (target - eye).normalize() }
  }
}

#[derive(Resource, Clone, Copy, Default, bevy::render::extract_resource::ExtractResource)]
#[extract_app(bevy::render::RenderApp)]
pub struct DebugNormals(pub u32);

pub const PITCH_LIMIT: f32 = 89.0_f32.to_radians();
pub const DIST_MIN: f32 = 32.0;

#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct OrbitCamera {
  pub target: Vec3,
  pub distance: f32,
  pub yaw: f32,
  pub pitch: f32,
}

impl OrbitCamera {
  pub fn from_eye(eye: Vec3, target: Vec3) -> Self {
    let offset = eye - target;
    let distance = offset.length();
    let pitch = offset.y.atan2(offset.xz().length());
    let yaw = offset.x.atan2(offset.z);
    Self { target, distance, yaw, pitch }
  }

  pub fn eye(&self) -> Vec3 {
    let (sin_yaw, cos_yaw) = self.yaw.sin_cos();
    let (sin_pitch, cos_pitch) = self.pitch.sin_cos();
    self.target + self.distance * Vec3::new(sin_yaw * cos_pitch, sin_pitch, cos_yaw * cos_pitch)
  }

  pub fn clamp(&mut self) {
    self.pitch = self.pitch.clamp(-PITCH_LIMIT, PITCH_LIMIT);
    self.distance = self.distance.max(DIST_MIN);
  }
}

impl DdaCameraConfig {
  pub fn from_eye_forward(
    eye: Vec3,
    forward: Vec3,
    fov_y: f32,
    aspect: f32,
    near: f32,
    far: f32,
  ) -> Self {
    let f = forward.normalize();
    let view = view::look_at_mat4(eye, eye + f, Vec3::Y);
    let proj = proj::directx::perspective(fov_y, aspect, near, far);
    let view_proj = proj.mul(view);
    Self { view_proj, inv_view_proj: view_proj.inverse(), position_world: eye, forward: f }
  }

  pub fn from_orbit(orbit: &OrbitCamera, fov_y: f32, aspect: f32, near: f32, far: f32) -> Self {
    let eye = orbit.eye();
    let view = view::look_at_mat4(eye, orbit.target, Vec3::Y);
    let proj = proj::directx::perspective(fov_y, aspect, near, far);
    let view_proj = proj.mul(view);
    Self {
      view_proj,
      inv_view_proj: view_proj.inverse(),
      position_world: eye,
      forward: (orbit.target - eye).normalize_or_zero(),
    }
  }
}

#[derive(Resource, Clone, Copy, ShaderType)]
pub struct DdaViewUniform {
  pub view_proj: Mat4,
  pub inv_view_proj: Mat4,
  pub cam_pos_voxel: Vec4,

  pub debug_mode: Vec4,

  pub lod: Vec4,
}

use crate::brickmap::consts::{
  DDA_BEAM, DDA_CHUNKWALK, DDA_DIR_LUT, DDA_LOD, DDA_MAKEGRID_ONLY, DDA_SKY_ONLY, EYE_ADAPT,
};

pub fn px_ang(render_h: f32) -> f32 {
  2.0 * (crate::brickmap::consts::DDA_FOV_Y * 0.5).tan() / render_h.max(1.0)
}

impl DdaViewUniform {
  pub fn from_cfg(cfg: &DdaCameraConfig, debug_mode: u32, render_h: f32) -> Self {
    let px_ang = px_ang(render_h);
    Self {
      view_proj: cfg.view_proj,
      inv_view_proj: cfg.inv_view_proj,
      cam_pos_voxel: cfg.position_world.extend(1.0),
      debug_mode: Vec4::new(
        (debug_mode == 4) as u32 as f32,
        (debug_mode == 5) as u32 as f32,
        if DDA_CHUNKWALK { 0.0 } else { 2.0 },
        if DDA_SKY_ONLY {
          2.0
        } else if DDA_MAKEGRID_ONLY {
          4.0
        } else if debug_mode == 6 {
          1.0
        } else {
          0.0
        },
      ),
      lod: Vec4::new(
        px_ang,
        DDA_LOD as u32 as f32,
        !DDA_BEAM as u32 as f32,
        !DDA_DIR_LUT as u32 as f32,
      ),
    }
  }
}

#[derive(Resource, Clone, ExtractResource)]
#[extract_app(bevy::render::RenderApp)]
pub struct DdaImages {
  pub target: Handle<Image>,
}

pub fn create_dda_image(images: &mut Assets<Image>) -> Handle<Image> {
  let mut image = Image::new_target_texture(
    crate::consts::VIEW_SIZE.x,
    crate::consts::VIEW_SIZE.y,
    TextureFormat::Rgba8Unorm,
    None,
  );
  image.asset_usage = RenderAssetUsages::RENDER_WORLD;
  image.texture_descriptor.usage = TextureUsages::STORAGE_BINDING
    | TextureUsages::TEXTURE_BINDING
    | TextureUsages::COPY_SRC
    | TextureUsages::COPY_DST;
  images.add(image)
}

pub mod wgsl_consts {
  pub const CHUNK_SIZE: u32 = 256;
  pub const BRICK_FACTOR: u32 = 4;
  pub const MAX_LEVEL: u32 = 4;

  pub const NODE_FIXED_WORDS: u32 = 3;
  pub const CHUNK_INDEX_CAP: u32 = 64;
  pub const CHUNK_INDEX_WORDS: u32 = 262_144;
  pub const TREE_BASE: u32 = 262_144;

  pub const PALETTE_WORDS: u32 = crate::brickmap::wire::PALETTE_WORDS as u32;

  pub const LEAF_INLINE_WORDS: u32 = crate::brickmap::wire::LEAF_INLINE_WORDS as u32;
  pub const LEAF_VOXELS_PER_WORD: u32 = crate::brickmap::wire::LEAF_VOXELS_PER_WORD as u32;
  pub const CHUNK_COMP_WORDS: u32 = 2048;
  pub const STATE_ENTRY_COUNT: u32 = 256;
  pub const STATE_WORDS_PER_ENTRY: u32 = 4;
  pub const STATE_TOTAL_WORDS: u32 = 1024;
  pub const SHADOW_BIAS: f32 = crate::consts::SHADOW_BIAS;
  pub const SHADOW_DIR_T_MAX: f32 = crate::consts::SHADOW_DIR_T_MAX;
  pub const EMISSIVE_EMIT_GAIN: f32 = crate::consts::EMISSIVE_EMIT_GAIN;

  pub const SHADOW_SURFACE_EPS: f32 = 0.03125;
}

use bevy::{
  core_pipeline::schedule::{Core2d, Core2dSystems, camera_driver},
  render::{
    Render, RenderApp, RenderStartup, RenderSystems,
    render_asset::RenderAssets,
    render_resource::{
      BindGroup, BindGroupEntries, BindGroupEntry, BindGroupLayoutDescriptor,
      BindGroupLayoutEntries, BindingResource, CachedComputePipelineId, CachedRenderPipelineId,
      ColorTargetState, ColorWrites, ComputePipelineDescriptor, Extent3d, FilterMode,
      FragmentState, PipelineCache, RenderPassDescriptor, SamplerBindingType, ShaderStages,
      StorageTextureAccess, TextureDescriptor, TextureDimension, TextureFormat, TextureSampleType,
      TextureUsages, TextureViewDescriptor, UniformBuffer, VertexState,
      binding_types::{
        sampler, storage_buffer_read_only_sized, storage_buffer_sized, texture_2d,
        texture_2d_array, texture_storage_2d, uniform_buffer,
      },
    },
    renderer::{RenderContext, RenderDevice, RenderQueue},
    texture::GpuImage,
    view::ViewTarget,
  },
};

use std::borrow::Cow;

use super::upload::GpuBrickMap;
use crate::lighting::{LightPoolUniform, LightingTheme, build_light_pool};

#[derive(Resource)]
pub(crate) struct DdaBg0BindGroup(pub(crate) BindGroup);
#[derive(Resource)]
pub(crate) struct DdaBg1BindGroup(pub(crate) BindGroup);
#[derive(Resource)]
pub(crate) struct DdaBg2BindGroup(pub(crate) BindGroup);
#[derive(Resource)]
pub(crate) struct DdaBg3BindGroup(pub(crate) BindGroup);
#[derive(Resource)]
struct DdaBlitBindGroup(BindGroup);

#[derive(Resource)]
pub(crate) struct LightPoolGpu(UniformBuffer<LightPoolUniform>);

#[derive(Resource, Default)]
pub(crate) struct AuxTexCache {
  texture: Option<Texture>,
  size: UVec2,
  gi_tex: Option<Texture>,
  gi_view: Option<TextureView>,
  gi_size: UVec2,
  gi_res_a: Option<Buffer>,
  gi_res_b: Option<Buffer>,
  gi_bg0: Option<BindGroup>,

  gi_read_bg: Option<BindGroup>,

  gi_guide: Option<Buffer>,

  face_slots: Option<Buffer>,

  gi_sec_slots: Option<Buffer>,

  gi_hist: [Option<Buffer>; 2],

  gi_phi: Option<Buffer>,

  gi_dn: [Option<Texture>; 4],

  gi_dn_src: [Option<TextureView>; 4],

  gi_dn_dst: [Option<TextureView>; 4],

  den_bg: [Option<BindGroup>; 7],

  den_cfg: Option<Buffer>,
  den_cfg_r: u32,

  den_flip: bool,

  gi_flatten_bg: Option<BindGroup>,

  wal: Option<Buffer>,

  region_table: Option<Buffer>,
}

pub(crate) const DEN_ATROUS_CHAINS: [[usize; 2]; 6] =
  [[0, 1], [1, 2], [2, 3], [0, 3], [1, 3], [2, 1]];

pub(crate) const DEN_ATROUS_ROUNDS: [[usize; 5]; 5] =
  [[3, 0, 0, 0, 0], [0, 4, 0, 0, 0], [0, 1, 2, 0, 0], [0, 1, 5, 4, 0], [0, 1, 5, 1, 2]];

impl AuxTexCache {
  pub(crate) fn gi_write_view(&self) -> Option<&TextureView> {
    self.gi_view.as_ref()
  }

  pub(crate) fn wal_buffer(&self) -> Option<&Buffer> {
    self.wal.as_ref()
  }

  pub(crate) fn region_buffer(&self) -> Option<&Buffer> {
    self.region_table.as_ref()
  }

  pub(crate) fn gi_res_buffers(&self) -> Option<(&Buffer, &Buffer)> {
    Some((self.gi_res_a.as_ref()?, self.gi_res_b.as_ref()?))
  }

  pub(crate) fn gi_guide_buffer(&self) -> Option<&Buffer> {
    self.gi_guide.as_ref()
  }

  pub(crate) fn face_slots_buffer(&self) -> Option<&Buffer> {
    self.face_slots.as_ref()
  }

  pub(crate) fn gi_sec_slots_buffer(&self) -> Option<&Buffer> {
    self.gi_sec_slots.as_ref()
  }

  pub(crate) fn gi_bg0(&self) -> Option<&BindGroup> {
    self.gi_bg0.as_ref()
  }
}

#[derive(Resource)]
#[allow(dead_code)]
pub(crate) struct DdaPipelines {
  pub(crate) bg0_layout: BindGroupLayoutDescriptor,

  pub(crate) bg0_gi_layout: BindGroupLayoutDescriptor,

  pub(crate) gi_read_layout: BindGroupLayoutDescriptor,
  pub(crate) bg1_layout: BindGroupLayoutDescriptor,
  pub(crate) bg2_layout: BindGroupLayoutDescriptor,
  pub(crate) bg3_layout: BindGroupLayoutDescriptor,
  blit_layout: BindGroupLayoutDescriptor,

  eye_layout: BindGroupLayoutDescriptor,
  pub(crate) compute_pipeline: CachedComputePipelineId,

  pub(crate) face_pipeline: CachedComputePipelineId,

  pub(crate) face_accum_pipeline: CachedComputePipelineId,
  pub(crate) beam_pipeline: CachedComputePipelineId,

  pub(crate) gi_pipeline: CachedComputePipelineId,

  eye_histogram_pipeline: CachedComputePipelineId,
  eye_update_pipeline: CachedComputePipelineId,
  blit_pipeline: CachedRenderPipelineId,

  blit_fxaa_pipeline: CachedRenderPipelineId,
}

#[derive(bevy::ecs::resource::Resource)]
pub struct EyeAdaptGpu {
  pub buf: Option<Buffer>,

  pub last: Option<std::time::Instant>,
  pub bg: Option<BindGroup>,

  pub settings: EyeAdaptSettings,

  pub settings_dirty: bool,
}

impl Default for EyeAdaptGpu {
  fn default() -> Self {
    Self {
      buf: None,
      last: None,
      bg: None,
      settings: EyeAdaptSettings::default(),
      settings_dirty: true,
    }
  }
}

fn sync_eye_adapt_settings(eye_set: Option<Res<EyeAdaptSettings>>, mut eye: ResMut<EyeAdaptGpu>) {
  let Some(s) = eye_set else {
    return;
  };
  if !s.is_changed() || eye.settings == *s {
    return;
  }
  eye.settings = *s;
  eye.settings_dirty = true;
}

#[derive(Resource, Clone, Copy, Debug, PartialEq, ExtractResource)]
#[extract_app(bevy::render::RenderApp)]
pub struct EyeAdaptSettings {
  pub enabled: bool,

  pub ev_max: f32,

  pub ev_min: f32,

  pub tau_brighten: f32,

  pub tau_darken: f32,

  pub key: f32,
}

impl Default for EyeAdaptSettings {
  fn default() -> Self {
    Self { enabled: true, ev_max: 3.0, ev_min: -3.0, tau_brighten: 2.0, tau_darken: 1.0, key: 0.18 }
  }
}

impl EyeAdaptSettings {
  pub fn startup() -> Self {
    Self { enabled: EYE_ADAPT, ..Self::default() }
  }
}

const EYE_PARAM_WORD: u64 = 72;

const EYE_PARAM_OFFSET: u64 = EYE_PARAM_WORD * 4;

fn eye_param_bytes(s: EyeAdaptSettings) -> [u8; 20] {
  let mut out = [0u8; 20];
  for (i, v) in [s.ev_max, s.ev_min, s.tau_brighten, s.tau_darken, s.key].iter().enumerate() {
    out[i * 4..i * 4 + 4].copy_from_slice(&v.to_bits().to_le_bytes());
  }
  out
}

pub struct BrickMapDdaPlugin;

impl Plugin for BrickMapDdaPlugin {
  fn build(&self, app: &mut App) {
    crate::shader::build_dda_shader(app);

    app.add_plugins((
      bevy::render::extract_resource::ExtractResourcePlugin::<DdaImages>::default(),
      bevy::render::extract_resource::ExtractResourcePlugin::<RenderScale>::default(),
      bevy::render::extract_resource::ExtractResourcePlugin::<PostFxSettings>::default(),
      bevy::render::extract_resource::ExtractResourcePlugin::<LightingTheme>::default(),
      bevy::render::extract_resource::ExtractResourcePlugin::<crate::lighting::ReflectionSettings>::default(),
      bevy::render::extract_resource::ExtractResourcePlugin::<crate::lighting::BaseSettings>::default(),
      bevy::render::extract_resource::ExtractResourcePlugin::<EyeAdaptSettings>::default(),
      crate::responsive::ResponsivePlugin,
    ));
    app.insert_resource(EyeAdaptSettings::startup());
    app.init_resource::<crate::lighting::ReflectionSettings>();
    app.init_resource::<crate::lighting::BaseSettings>();
    let dda_shader = app.world().resource::<crate::shader::DdaShaderHandle>().clone();
    let dda_shader_rt = app.world().resource::<crate::shader::DdaShaderRtHandle>().clone();
    let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
      return;
    };
    render_app.insert_resource(dda_shader);
    render_app.insert_resource(dda_shader_rt);
    render_app
      .add_systems(bevy::render::ExtractSchedule, extract_camera_config)
      .add_systems(RenderStartup, init_dda_pipelines)
      .add_systems(Render, sync_eye_adapt_settings.in_set(RenderSystems::PrepareResources))
      .add_systems(Render, prepare_rt_scene.in_set(RenderSystems::PrepareResources))
      .add_systems(
        Render,
        prepare_dda_bind_groups
          .in_set(RenderSystems::PrepareBindGroups)
          .after(sync_eye_adapt_settings)
          .after(super::upload::prepare),
      )
      .add_systems(
        RenderGraph,
        sync_rt_scene
          .in_set(bevy::render::renderer::RenderGraphSystems::Render)
          .before(dispatch_dda),
      )
      .add_systems(
        RenderGraph,
        dispatch_dda
          .in_set(bevy::render::renderer::RenderGraphSystems::Render)
          .before(camera_driver),
      )
      .add_systems(Core2d, blit_dda_view.in_set(Core2dSystems::PostProcess));
  }
}

fn extract_camera_config(
  mut commands: bevy::ecs::system::Commands,
  cfg: Option<bevy::render::Extract<bevy::ecs::system::Res<crate::brickmap::DdaCameraConfig>>>,
  debug: Option<bevy::render::Extract<bevy::ecs::system::Res<crate::brickmap::DebugNormals>>>,
  scale: Option<bevy::render::Extract<bevy::ecs::system::Res<RenderScale>>>,
) {
  let Some(cfg) = cfg else { return };
  let debug_mode = debug.map(|d| d.0).unwrap_or(0);
  let render_h = scale.map(|s| s.size.y as f32).unwrap_or(crate::consts::VIEW_SIZE.y as f32);
  let uniform = DdaViewUniform::from_cfg(&cfg, debug_mode, render_h);
  commands.insert_resource(uniform);
}

pub(crate) fn init_dda_pipelines(
  mut commands: Commands,
  asset_server: Res<AssetServer>,
  dda_shader: Res<crate::shader::DdaShaderHandle>,
  dda_shader_rt: Res<crate::shader::DdaShaderRtHandle>,
  pipeline_cache: Res<PipelineCache>,
  render_device: Res<RenderDevice>,
) {
  let rt = super::rt::rt_enabled(&render_device);
  commands.insert_resource(super::rt::RtScene::new(rt, &render_device));

  let bg0 = BindGroupLayoutDescriptor::new(
    "DdaBg0",
    &BindGroupLayoutEntries::sequential(
      ShaderStages::COMPUTE,
      (
        texture_storage_2d(TextureFormat::Rgba8Unorm, StorageTextureAccess::WriteOnly),
        uniform_buffer::<DdaViewUniform>(false),
        texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::ReadWrite),
        storage_buffer_read_only_sized(false, None),
      ),
    ),
  );

  let gi_read = BindGroupLayoutDescriptor::new(
    "DdaBg5GiRead",
    &[
      BindGroupLayoutEntry {
        binding: 4,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Texture {
          sample_type: TextureSampleType::Float { filterable: true },
          view_dimension: TextureViewDimension::D2,
          multisampled: false,
        },
        count: None,
      },
      BindGroupLayoutEntry {
        binding: 6,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
          ty: BufferBindingType::Storage { read_only: false },
          has_dynamic_offset: false,
          min_binding_size: None,
        },
        count: None,
      },
      BindGroupLayoutEntry {
        binding: 8,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
          ty: BufferBindingType::Storage { read_only: false },
          has_dynamic_offset: false,
          min_binding_size: None,
        },
        count: None,
      },
      BindGroupLayoutEntry {
        binding: 10,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
          ty: BufferBindingType::Storage { read_only: true },
          has_dynamic_offset: false,
          min_binding_size: None,
        },
        count: None,
      },
      BindGroupLayoutEntry {
        binding: 11,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
          ty: BufferBindingType::Storage { read_only: false },
          has_dynamic_offset: false,
          min_binding_size: None,
        },
        count: None,
      },
    ],
  );

  let bg0_gi = BindGroupLayoutDescriptor::new(
    "DdaBg0Gi",
    &[
      BindGroupLayoutEntry {
        binding: 1,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
          ty: BufferBindingType::Uniform,
          has_dynamic_offset: false,
          min_binding_size: Some(DdaViewUniform::min_size()),
        },
        count: None,
      },
      BindGroupLayoutEntry {
        binding: 2,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::StorageTexture {
          access: StorageTextureAccess::ReadWrite,
          format: TextureFormat::R32Float,
          view_dimension: TextureViewDimension::D2,
        },
        count: None,
      },
    ],
  );

  let mut bg1_entries: Vec<BindGroupLayoutEntry> = BindGroupLayoutEntries::sequential(
    ShaderStages::COMPUTE,
    (
      storage_buffer_read_only_sized(false, None),
      storage_buffer_read_only_sized(false, None),
      storage_buffer_read_only_sized(false, None),
      uniform_buffer::<super::wire::BrickMapGlobals>(false),
      sampler(SamplerBindingType::Filtering),
      storage_buffer_read_only_sized(false, None),
      texture_2d_array(TextureSampleType::Float { filterable: true }),
      texture_2d_array(TextureSampleType::Float { filterable: true }),
      sampler(SamplerBindingType::Filtering),
      storage_buffer_sized(false, None),
      storage_buffer_sized(false, None),
    ),
  )
  .to_vec();

  bg1_entries.push(BindGroupLayoutEntry {
    binding: 12,
    visibility: ShaderStages::COMPUTE,
    ty: BindingType::Buffer {
      ty: BufferBindingType::Storage { read_only: true },
      has_dynamic_offset: false,
      min_binding_size: None,
    },
    count: None,
  });

  bg1_entries.push(BindGroupLayoutEntry {
    binding: 13,
    visibility: ShaderStages::COMPUTE,
    ty: BindingType::Buffer {
      ty: BufferBindingType::Storage { read_only: true },
      has_dynamic_offset: false,
      min_binding_size: None,
    },
    count: None,
  });

  if rt {
    bg1_entries.push(BindGroupLayoutEntry {
      binding: 11,
      visibility: ShaderStages::COMPUTE,
      ty: BindingType::AccelerationStructure { vertex_return: false },
      count: None,
    });
  }
  let bg1 = BindGroupLayoutDescriptor::new("DdaBg1", &bg1_entries);

  let bg2 = BindGroupLayoutDescriptor::new(
    "DdaBg2",
    &BindGroupLayoutEntries::sequential(
      ShaderStages::COMPUTE,
      (storage_buffer_read_only_sized(false, None), storage_buffer_read_only_sized(false, None)),
    ),
  );

  let bg3 = BindGroupLayoutDescriptor::new(
    "DdaBg3",
    &BindGroupLayoutEntries::sequential(
      ShaderStages::COMPUTE,
      (uniform_buffer::<LightPoolUniform>(false),),
    ),
  );

  let blit = BindGroupLayoutDescriptor::new(
    "DdaBlit",
    &BindGroupLayoutEntries::sequential(
      ShaderStages::FRAGMENT,
      (
        texture_2d(TextureSampleType::Float { filterable: true }),
        sampler(SamplerBindingType::Filtering),
      ),
    ),
  );

  let eye = BindGroupLayoutDescriptor::new(
    "DdaBgEye",
    &BindGroupLayoutEntries::sequential(
      ShaderStages::COMPUTE,
      (
        texture_2d(TextureSampleType::Float { filterable: true }),
        storage_buffer_sized(false, None),
      ),
    ),
  );

  let dda_shader = if rt { dda_shader_rt.0.clone() } else { dda_shader.0.clone() };
  let layouts =
    vec![bg0.clone(), bg1.clone(), bg2.clone(), bg3.clone(), crate::gi::gi_bg4_layout()];

  let dda_layouts = {
    let mut v = layouts.clone();
    v.push(gi_read.clone());
    v.push(crate::volumetric::fog_read_layout());
    v
  };

  let dda_layouts_face = dda_layouts.clone();
  let dda_layouts_accum = dda_layouts.clone();

  let eye_layouts = vec![eye.clone(); 8];
  let compute = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
    label: Some(Cow::from("gate_dda_compute")),
    layout: dda_layouts,
    shader: dda_shader.clone(),
    entry_point: Some(Cow::from("dda_main")),
    ..default()
  });

  let face = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
    label: Some(Cow::from("gate_dda_face")),
    layout: dda_layouts_face,
    shader: dda_shader.clone(),
    entry_point: Some(Cow::from("dda_face_main")),
    ..default()
  });

  let face_accum = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
    label: Some(Cow::from("gate_dda_face_accum")),
    layout: dda_layouts_accum,
    shader: dda_shader.clone(),
    entry_point: Some(Cow::from("dda_face_accum")),
    ..default()
  });

  let beam = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
    label: Some(Cow::from("gate_beam")),
    layout: layouts.clone(),
    shader: dda_shader.clone(),
    entry_point: Some(Cow::from("beam_main")),
    ..default()
  });

  let gi_layouts = vec![
    bg0_gi.clone(),
    bg1.clone(),
    bg2.clone(),
    bg3.clone(),
    crate::gi::gi_bg4_layout(),
    crate::gi::gi_bg5_layout(),
  ];
  let gi = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
    label: Some(Cow::from("gate_gi")),
    layout: gi_layouts,
    shader: dda_shader.clone(),
    entry_point: Some(Cow::from("gi_main")),
    ..default()
  });

  let eye_histogram = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
    label: Some(Cow::from("gate_eye_histogram")),
    layout: eye_layouts.clone(),
    shader: dda_shader.clone(),
    entry_point: Some(Cow::from("eye_adapt_histogram")),
    ..default()
  });
  let eye_update = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
    label: Some(Cow::from("gate_eye_update")),
    layout: eye_layouts.clone(),
    shader: dda_shader.clone(),
    entry_point: Some(Cow::from("eye_adapt_update")),
    ..default()
  });

  let blit_shader = asset_server.load(BLIT_SHADER_ASSET_PATH);
  let blit_make = |label: &str, entry: &str| {
    pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
      label: Some(Cow::from(label.to_string())),
      layout: vec![blit.clone()],
      vertex: VertexState {
        shader: blit_shader.clone(),
        entry_point: Some(Cow::from("vs_main")),
        ..default()
      },
      fragment: Some(FragmentState {
        shader: blit_shader.clone(),
        entry_point: Some(Cow::from(entry.to_string())),
        targets: vec![Some(ColorTargetState {
          format: TextureFormat::Rgba8UnormSrgb,
          blend: None,
          write_mask: ColorWrites::ALL,
        })],
        ..default()
      }),
      ..default()
    })
  };
  let blit_pipeline = blit_make("gate_dda_blit", "fs_main");
  let blit_fxaa_pipeline = blit_make("gate_dda_blit_fxaa", "fs_fxaa");

  commands.insert_resource(DdaPipelines {
    bg0_layout: bg0,
    bg0_gi_layout: bg0_gi,
    gi_read_layout: gi_read,
    bg1_layout: bg1,
    bg2_layout: bg2,
    bg3_layout: bg3,
    blit_layout: blit,
    eye_layout: eye,
    compute_pipeline: compute,
    face_pipeline: face,
    face_accum_pipeline: face_accum,
    beam_pipeline: beam,
    gi_pipeline: gi,
    eye_histogram_pipeline: eye_histogram,
    eye_update_pipeline: eye_update,
    blit_pipeline,
    blit_fxaa_pipeline,
  });
  commands.insert_resource(LightPoolGpu(UniformBuffer::default()));
  commands.insert_resource(AuxTexCache::default());
  commands.insert_resource(EyeAdaptGpu::default());
}

#[derive(bevy::ecs::system::SystemParam)]
pub(crate) struct DdaTune<'w> {
  pub gi: Option<Res<'w, crate::gi::GiSettings>>,

  pub refl: Option<Res<'w, crate::lighting::ReflectionSettings>>,

  pub base: Option<Res<'w, crate::lighting::BaseSettings>>,

  pub rt: Option<Res<'w, super::rt::RtScene>>,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare_dda_bind_groups(
  mut commands: Commands,
  pipelines: Res<DdaPipelines>,
  gpu_images: Res<RenderAssets<GpuImage>>,
  mut eye: ResMut<EyeAdaptGpu>,
  images: Option<Res<DdaImages>>,
  view_uniform: Option<Res<DdaViewUniform>>,
  gpu_brickmap: Option<Res<GpuBrickMap>>,
  pbr_set: Option<Res<crate::pbr_texture::PbrTextureSet>>,
  lighting: Option<Res<LightingTheme>>,
  light_gpu: Option<ResMut<LightPoolGpu>>,
  render_device: Res<RenderDevice>,
  pipeline_cache: Res<PipelineCache>,
  queue: Res<RenderQueue>,
  scale: Res<RenderScale>,
  tune: DdaTune,
  mut beam_cache: ResMut<AuxTexCache>,
) {
  let Some(images) = images else {
    bevy::log::info_once!("DDA prepare: no DdaImages");
    return;
  };
  let Some(view_uniform) = view_uniform else {
    bevy::log::info_once!("DDA prepare: no DdaViewUniform");
    return;
  };
  let Some(gpu) = gpu_brickmap else {
    bevy::log::info_once!("DDA prepare: no GpuBrickMap");
    return;
  };
  let Some(tex_view) = gpu_images.get(&images.target) else {
    bevy::log::info_once!("DDA prepare: GpuImage not ready");
    return;
  };

  let mut u = UniformBuffer::from(*view_uniform);
  u.write_buffer(&render_device, &queue);

  let bg0_layout = pipeline_cache.get_bind_group_layout(&pipelines.bg0_layout);
  let eye_layout = pipeline_cache.get_bind_group_layout(&pipelines.eye_layout);
  let bg1_layout = pipeline_cache.get_bind_group_layout(&pipelines.bg1_layout);
  let bg2_layout = pipeline_cache.get_bind_group_layout(&pipelines.bg2_layout);
  let bg3_layout = pipeline_cache.get_bind_group_layout(&pipelines.bg3_layout);
  let blit_layout = pipeline_cache.get_bind_group_layout(&pipelines.blit_layout);

  let beam_div = crate::brickmap::consts::BEAM_DIV;
  let beam_size = UVec2::new(scale.size.x.div_ceil(beam_div), scale.size.y.div_ceil(beam_div));
  if beam_cache.texture.is_none() || beam_cache.size != beam_size {
    let tex = render_device.create_texture(&TextureDescriptor {
      label: Some("gate_beam_depth"),
      size: Extent3d { width: beam_size.x, height: beam_size.y, depth_or_array_layers: 1 },
      mip_level_count: 1,
      sample_count: 1,
      dimension: TextureDimension::D2,
      format: TextureFormat::R32Float,
      usage: TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC,
      view_formats: &[],
    });
    beam_cache.texture = Some(tex);
    beam_cache.size = beam_size;
  }
  let beam_tex = beam_cache.texture.as_ref().expect("beam texture not created");
  let beam_view = beam_tex.create_view(&TextureViewDescriptor::default());

  let gi_size = tune.gi.as_deref().copied().unwrap_or_default().gi_size(scale.size);
  if beam_cache.gi_tex.is_none() || beam_cache.gi_size != gi_size {
    let make = |label: &str, format: TextureFormat| {
      render_device.create_texture(&TextureDescriptor {
        label: Some(label),
        size: Extent3d { width: gi_size.x, height: gi_size.y, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format,

        usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
      })
    };
    let gi_tex = make("gate_gi", TextureFormat::Rgba16Float);
    beam_cache.gi_view = Some(gi_tex.create_view(&TextureViewDescriptor::default()));
    beam_cache.gi_tex = Some(gi_tex);
    beam_cache.gi_size = gi_size;

    let res_bytes =
      gi_size.x as u64 * gi_size.y as u64 * crate::wesl_consts::gi_consts().gi_res_words as u64 * 4;
    let make_res = |label: &str| {
      render_device.create_buffer(&BufferDescriptor {
        label: Some(label),
        size: res_bytes.max(4),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
      })
    };
    beam_cache.gi_res_a = Some(make_res("gate_gi_res_a"));
    beam_cache.gi_res_b = Some(make_res("gate_gi_res_b"));

    let c = crate::wesl_consts::gi_consts();
    let px = gi_size.x as u64 * gi_size.y as u64;
    let make_buf = |label: &str, bytes: u64| {
      render_device.create_buffer(&BufferDescriptor {
        label: Some(label),
        size: bytes.max(4),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
      })
    };
    beam_cache.gi_guide = Some(make_buf("gate_gi_guide", px * c.gi_den_guide_words as u64 * 4));

    let face_slots_n = (px * 2).next_power_of_two().clamp(256, 1u64 << 21);
    beam_cache.face_slots =
      Some(make_buf("gate_face_slots", face_slots_n * c.face_words as u64 * 4));

    beam_cache.gi_sec_slots =
      Some(make_buf("gate_gi_sec_slots", face_slots_n * c.gi_sec_words as u64 * 4));
    beam_cache.gi_hist = [
      Some(make_buf("gate_gi_hist_a", px * c.gi_den_hist_words as u64 * 4)),
      Some(make_buf("gate_gi_hist_b", px * c.gi_den_hist_words as u64 * 4)),
    ];
    beam_cache.gi_phi = Some(make_buf("gate_gi_den_phi", px * 4));
    for (i, label) in
      ["gate_gi_dn_tmp", "gate_gi_dn_a", "gate_gi_dn_b", "gate_gi_den"].iter().enumerate()
    {
      let t = make(label, TextureFormat::Rgba16Float);
      beam_cache.gi_dn_src[i] = Some(t.create_view(&TextureViewDescriptor::default()));
      beam_cache.gi_dn_dst[i] = Some(t.create_view(&TextureViewDescriptor::default()));
      beam_cache.gi_dn[i] = Some(t);
    }
  }

  if beam_cache.wal.is_none() {
    beam_cache.wal = Some(render_device.create_buffer(&BufferDescriptor {
      label: Some("gate_gi_wal"),
      size: crate::gi::WAL_SLOTS * crate::gi::WAL_WORDS * 4,
      usage: BufferUsages::STORAGE,
      mapped_at_creation: false,
    }));
  }

  if beam_cache.region_table.is_none() {
    beam_cache.region_table = Some(render_device.create_buffer(&BufferDescriptor {
      label: Some("gate_gi_region_rev"),
      size: crate::gi::REGION_TABLE_BYTES,
      usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
      mapped_at_creation: false,
    }));
  }

  let gi_view = beam_cache.gi_view.as_ref().expect("gi view not created").clone();

  const EYE_WORDS: u64 = 80;
  if eye.buf.is_none() {
    let b = render_device.create_buffer(&BufferDescriptor {
      label: Some("dda_eye_adapt"),
      size: EYE_WORDS * 4,
      usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
      mapped_at_creation: false,
    });
    let mut init = [0u8; (EYE_WORDS * 4) as usize];
    init[..4].copy_from_slice(&1.0f32.to_bits().to_le_bytes());
    queue.write_buffer(&b, 0, &init);

    queue.write_buffer(&b, EYE_PARAM_OFFSET, &eye_param_bytes(eye.settings));
    eye.buf = Some(b);
  }

  let eye_buf = eye.buf.clone().expect("刚插入");

  let now = std::time::Instant::now();
  if eye.settings.enabled {
    let dt = eye.last.map_or(1.0 / 60.0, |t| now.duration_since(t).as_secs_f32());
    queue.write_buffer(&eye_buf, 12, &dt.clamp(0.0, 0.25).to_bits().to_le_bytes());
  }
  eye.last = Some(now);

  if std::mem::take(&mut eye.settings_dirty) {
    let s = eye.settings;
    queue.write_buffer(&eye_buf, EYE_PARAM_OFFSET, &eye_param_bytes(s));

    if !s.enabled {
      queue.write_buffer(&eye_buf, 0, &1.0f32.to_bits().to_le_bytes());
    }

    bevy::log::debug!(
      target: "gate",
      "eye adapt → GPU: {} EV+ {:.2} / EV- {:.2} / tau+ {:.2}s / tau- {:.2}s / key {:.3}",
      if s.enabled { "on" } else { "off" },
      s.ev_max,
      s.ev_min,
      s.tau_brighten,
      s.tau_darken,
      s.key,
    );
  }

  let bg0 = render_device.create_bind_group(
    None,
    &bg0_layout,
    &BindGroupEntries::sequential((
      &tex_view.texture_view,
      &u,
      &beam_view,
      eye_buf.as_entire_binding(),
    )),
  );

  let gi_read_layout = pipeline_cache.get_bind_group_layout(&pipelines.gi_read_layout);

  let den_plan = match tune.gi.as_ref() {
    Some(g) => g.denoise_plan(),
    None => crate::gi::GiSettings::default().denoise_plan(),
  };
  let gi_den_view = if den_plan.on {
    beam_cache.gi_dn_src[3].as_ref().expect("降噪输出视图未创建").clone()
  } else {
    gi_view.clone()
  };
  let gi_guide = beam_cache.gi_guide.as_ref().expect("降噪导引 buffer 未创建").clone();

  let face_slots = beam_cache.face_slots.as_ref().expect("逐面去重表 buffer 未创建").clone();

  let wal = beam_cache.wal.as_ref().expect("WAL 表 buffer 未创建").clone();

  let region_table = beam_cache.region_table.as_ref().expect("区域修订表 buffer 未创建").clone();
  let gi_read_bg = render_device.create_bind_group(
    None,
    &gi_read_layout,
    &[
      BindGroupEntry { binding: 4, resource: BindingResource::TextureView(&gi_den_view) },
      BindGroupEntry { binding: 6, resource: gi_guide.as_entire_binding() },
      BindGroupEntry { binding: 8, resource: face_slots.as_entire_binding() },
      BindGroupEntry { binding: 10, resource: region_table.as_entire_binding() },
      BindGroupEntry { binding: 11, resource: wal.as_entire_binding() },
    ],
  );
  beam_cache.gi_read_bg = Some(gi_read_bg);

  let gi_flatten_layout = pipeline_cache.get_bind_group_layout(&crate::gi::gi_flatten_layout());
  let gi_flatten_bg = render_device.create_bind_group(
    None,
    &gi_flatten_layout,
    &[
      BindGroupEntry { binding: 10, resource: gi_guide.as_entire_binding() },
      BindGroupEntry { binding: 18, resource: face_slots.as_entire_binding() },
      BindGroupEntry { binding: 19, resource: BindingResource::TextureView(&gi_view) },
    ],
  );
  beam_cache.gi_flatten_bg = Some(gi_flatten_bg);

  let den_r = den_plan.radius;
  if beam_cache.den_cfg.is_none() {
    beam_cache.den_cfg = Some(render_device.create_buffer(&BufferDescriptor {
      label: Some("gate_gi_den_cfg"),
      size: 16,
      usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
      mapped_at_creation: false,
    }));
  }
  if beam_cache.den_cfg_r != den_r {
    beam_cache.den_cfg_r = den_r;
    let cfg = beam_cache.den_cfg.as_ref().expect("刚创建");
    queue.write_buffer(cfg, 0, &den_r.to_le_bytes());
    bevy::log::debug!(
      target: "gate",
      "GI 降噪档位 → {}（atrous 核半径 {}，每轮 {} tap）",
      if den_plan.on {
        format!("{} 轮 atrous", den_plan.rounds)
      } else {
        "关（直接采样原始 GI）".to_string()
      },
      den_r,
      (2 * den_r + 1) * (2 * den_r + 1) - 1,
    );
  }
  let den_cfg = beam_cache.den_cfg.as_ref().expect("刚创建").clone();

  let den_runs = tune.gi.as_ref().is_some_and(|g| g.enabled);
  let (prev_i, cur_i) = if beam_cache.den_flip { (1usize, 0usize) } else { (0usize, 1usize) };
  {
    let guide = beam_cache.gi_guide.as_ref().expect("导引 buffer 未创建").clone();
    let phi = beam_cache.gi_phi.as_ref().expect("φ buffer 未创建").clone();
    let hist_prev = beam_cache.gi_hist[prev_i].as_ref().expect("历史 buffer 未创建").clone();
    let hist_cur = beam_cache.gi_hist[cur_i].as_ref().expect("历史 buffer 未创建").clone();
    let dn_src: Vec<TextureView> =
      beam_cache.gi_dn_src.iter().map(|v| v.as_ref().expect("降噪纹理未创建").clone()).collect();
    let dn_dst: Vec<TextureView> =
      beam_cache.gi_dn_dst.iter().map(|v| v.as_ref().expect("降噪纹理未创建").clone()).collect();
    let temporal_layout =
      pipeline_cache.get_bind_group_layout(&crate::gi::gi_den_temporal_layout());
    let atrous_layout = pipeline_cache.get_bind_group_layout(&crate::gi::gi_den_atrous_layout());
    beam_cache.den_bg[0] = Some(render_device.create_bind_group(
      None,
      &temporal_layout,
      &[
        BindGroupEntry { binding: 10, resource: guide.as_entire_binding() },
        BindGroupEntry { binding: 11, resource: BindingResource::TextureView(&gi_view) },
        BindGroupEntry { binding: 12, resource: hist_prev.as_entire_binding() },
        BindGroupEntry { binding: 13, resource: hist_cur.as_entire_binding() },
        BindGroupEntry { binding: 16, resource: BindingResource::TextureView(&dn_dst[0]) },
        BindGroupEntry { binding: 17, resource: phi.as_entire_binding() },
        BindGroupEntry { binding: 20, resource: den_cfg.as_entire_binding() },
      ],
    ));
    for (k, [s, d]) in DEN_ATROUS_CHAINS.iter().enumerate() {
      beam_cache.den_bg[k + 1] = Some(render_device.create_bind_group(
        None,
        &atrous_layout,
        &[
          BindGroupEntry { binding: 10, resource: guide.as_entire_binding() },
          BindGroupEntry { binding: 13, resource: hist_cur.as_entire_binding() },
          BindGroupEntry { binding: 14, resource: BindingResource::TextureView(&dn_src[*s]) },
          BindGroupEntry { binding: 15, resource: BindingResource::TextureView(&dn_dst[*d]) },
          BindGroupEntry { binding: 17, resource: phi.as_entire_binding() },
          BindGroupEntry { binding: 20, resource: den_cfg.as_entire_binding() },
        ],
      ));
    }
  }
  if den_runs {
    beam_cache.den_flip = !beam_cache.den_flip;
  }

  let bg0_gi_layout = pipeline_cache.get_bind_group_layout(&pipelines.bg0_gi_layout);
  let gi_bg0 = render_device.create_bind_group(
    None,
    &bg0_gi_layout,
    &[
      BindGroupEntry { binding: 1, resource: u.binding().expect("view uniform 已写入") },
      BindGroupEntry { binding: 2, resource: BindingResource::TextureView(&beam_view) },
    ],
  );
  beam_cache.gi_bg0 = Some(gi_bg0);

  let eye_bg = render_device.create_bind_group(
    None,
    &eye_layout,
    &BindGroupEntries::sequential((&tex_view.texture_view, eye_buf.as_entire_binding())),
  );
  eye.bg = Some(eye_bg);

  let globals_bind = gpu.globals.binding().expect(
    "GpuBrickMap.globals uniform buffer 未初始化（RenderStartup init_empty_gpu 应默认构造）",
  );

  let pbr_albedo = pbr_set.as_deref().and_then(|s| gpu_images.get(s.albedo_rough()));
  let pbr_metal = pbr_set.as_deref().and_then(|s| gpu_images.get(s.metal()));
  if pbr_albedo.is_none() || pbr_metal.is_none() {
    bevy::log::info_once!(
      target: "gate",
      "DDA prepare: PBR 贴图数组未就绪（贴图集或 GpuImage）⇒ BG1 binding 6/7 先绑 1×1×1 占位\
       （资产表 binding 5 若也未上传就是零初始化占位，长度已够）；不 panic"
    );
  }
  let pbr_albedo_view =
    pbr_albedo.map_or_else(|| gpu.pbr_albedo_rough_view.clone(), |i| i.texture_view.clone());
  let pbr_metal_view =
    pbr_metal.map_or_else(|| gpu.pbr_metal_view.clone(), |i| i.texture_view.clone());
  let bg1 = {
    let mut entries: Vec<BindGroupEntry> = BindGroupEntries::sequential((
      gpu.struct_buf.as_entire_binding(),
      gpu.leaves.as_entire_binding(),
      gpu.palette.as_entire_binding(),
      globals_bind,
      &gpu.light_sampler,
      gpu.material_assets.as_entire_binding(),
      &pbr_albedo_view,
      &pbr_metal_view,
      &gpu.pbr_sampler,
      gpu.lod_diag.as_entire_binding(),
      gpu.lod_req.as_entire_binding(),
    ))
    .to_vec();

    entries.push(BindGroupEntry { binding: 12, resource: gpu.occ.as_entire_binding() });

    entries.push(BindGroupEntry { binding: 13, resource: gpu.struct_buf_p1.as_entire_binding() });

    if let Some(rt) = tune.rt.as_deref().filter(|r| r.is_enabled()) {
      entries.push(BindGroupEntry {
        binding: 11,
        resource: BindingResource::AccelerationStructure(rt.tlas()),
      });
    }
    render_device.create_bind_group(None, &bg1_layout, &entries)
  };

  let bg2 = render_device.create_bind_group(
    None,
    &bg2_layout,
    &BindGroupEntries::sequential((
      gpu.grid_descs_buf.as_entire_binding(),
      gpu.inst_bvh_buf.as_entire_binding(),
    )),
  );

  let Some(lighting) = lighting else {
    bevy::log::info_once!("DDA prepare: no LightingTheme");
    return;
  };
  let Some(mut lp) = light_gpu else {
    bevy::log::info_once!("DDA prepare: no LightPoolGpu");
    return;
  };
  *lp.0.get_mut() = build_light_pool(&lighting);

  lp.0.get_mut().g.refl_tier = tune.refl.as_ref().map_or(0, |r| r.tier());
  lp.0.get_mut().g.refl_nest = tune.refl.as_ref().map_or(0, |r| r.nest());

  lp.0.get_mut().g.base_flags =
    tune.base.as_ref().map_or(crate::lighting::BaseSettings::default().flags(), |b| b.flags());
  lp.0.write_buffer(&render_device, &queue);
  let bg3 = render_device.create_bind_group(None, &bg3_layout, &BindGroupEntries::single(&lp.0));

  let blit_sampler = render_device.create_sampler(&SamplerDescriptor {
    label: Some("gate_dda_blit_sampler"),
    mag_filter: FilterMode::Linear,
    min_filter: FilterMode::Linear,
    ..default()
  });
  let blit_bg = render_device.create_bind_group(
    None,
    &blit_layout,
    &BindGroupEntries::sequential((&tex_view.texture_view, &blit_sampler)),
  );

  commands.insert_resource(DdaBg0BindGroup(bg0));
  commands.insert_resource(DdaBg1BindGroup(bg1));
  commands.insert_resource(DdaBg2BindGroup(bg2));
  commands.insert_resource(DdaBg3BindGroup(bg3));
  commands.insert_resource(DdaBlitBindGroup(blit_bg));
}

const RT_INSERT_PER_FRAME: u32 = 1024;

#[derive(Default)]
struct RtSyncCursor {
  ready: bool,

  cursor: usize,
  seq: u64,
  epoch: u64,

  stat: u32,
}

fn prepare_rt_scene(
  mut rt: ResMut<super::rt::RtScene>,
  mirror: Res<super::upload::BuilderMirror>,
  device: Res<RenderDevice>,
  queue: Res<RenderQueue>,
  mut cur: Local<RtSyncCursor>,
) {
  if !rt.is_enabled() {
    return;
  }
  rt.tick();
  let Some(builder) = mirror.builder.as_ref() else { return };
  let len = builder.resident_log_len(0);
  let seq = builder.resident_seq(0);
  let epoch = builder.resident_epoch(0);

  let consistent = cur.ready
    && cur.epoch == epoch
    && cur.cursor <= len
    && (len as u64).wrapping_sub(cur.cursor as u64) == seq.wrapping_sub(cur.seq);
  let mut budget = RT_INSERT_PER_FRAME;

  let mut still: Vec<gate_voxel::ChunkCoord> = Vec::new();
  for c in std::mem::take(&mut rt.deferred) {
    if budget == 0 {
      still.push(c);
      continue;
    }
    match builder.aabbs_of(0, c) {
      Some(a) if !a.is_empty() && !rt.contains(c) => {
        rt.insert_chunk(&device, &queue, c, a);
        budget -= 1;
      }

      _ => {}
    }
  }
  rt.deferred = still;

  if consistent {
    for ev in builder.resident_log_from(0, cur.cursor) {
      if !ev.mounted {
        rt.remove_chunk(ev.c);
        continue;
      }
      if budget == 0 {
        rt.deferred.push(ev.c);
        continue;
      }
      if let Some(a) = builder.aabbs_of(0, ev.c)
        && !a.is_empty()
        && !rt.contains(ev.c)
      {
        rt.insert_chunk(&device, &queue, ev.c, a);
        budget -= 1;
      }
    }
  } else {
    let resident: std::collections::HashSet<gate_voxel::ChunkCoord> =
      builder.resident_chunks(0).into_iter().collect();
    for c in rt.chunks().collect::<Vec<_>>() {
      if !resident.contains(&c) {
        rt.remove_chunk(c);
      }
    }
    for &c in &resident {
      if budget == 0 {
        rt.deferred.push(c);
        continue;
      }
      if let Some(a) = builder.aabbs_of(0, c)
        && !a.is_empty()
        && !rt.contains(c)
      {
        rt.insert_chunk(&device, &queue, c, a);
        budget -= 1;
      }
    }
  }
  cur.cursor = len;
  cur.seq = seq;
  cur.epoch = epoch;
  cur.ready = true;

  if rt.slots_exhausted() {
    bevy::log::warn_once!(
      "RT: TLAS 实例槽用尽（{} 个）⇒ 之后挂载的块没有加速结构",
      super::rt::RT_MAX_INSTANCES
    );
  }
  cur.stat = cur.stat.wrapping_add(1);
  if cur.stat.is_multiple_of(RT_STAT_FRAMES) {
    bevy::log::info!(
      target: "gate",
      "RT: 实例 {}（常驻 {} 块）/ 欠账 {}",
      rt.instance_count(),
      builder.resident_chunks(0).len(),
      rt.deferred.len(),
    );

    if !rt.deferred.is_empty() {
      bevy::log::warn!(
        target: "gate",
        "RT[!] 本帧 {} 块没有加速结构 ⇒ 这些块在 RT 段上不可见；应恒为 0，非零要查挂载速率",
        rt.deferred.len()
      );
    }
  }
}

const RT_STAT_FRAMES: u32 = 240;

fn sync_rt_scene(mut ctx: RenderContext, mut rt: ResMut<super::rt::RtScene>) {
  if !rt.is_enabled() {
    return;
  }
  rt.record(ctx.command_encoder());
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatch_dda(
  mut ctx: RenderContext,
  bg0: Option<Res<DdaBg0BindGroup>>,
  bg1: Option<Res<DdaBg1BindGroup>>,
  bg2: Option<Res<DdaBg2BindGroup>>,
  bg3: Option<Res<DdaBg3BindGroup>>,
  bg4: Option<Res<crate::gi::GiBg4>>,
  bg5: Option<Res<crate::gi::GiBg5>>,
  eye: Option<Res<EyeAdaptGpu>>,
  gi: Option<Res<crate::gi::GiSettings>>,
  gi_gpu: Option<Res<crate::gi::GiGpu>>,
  fog: crate::volumetric::FogRes,
  aux: Option<Res<AuxTexCache>>,
  pipeline_cache: Res<PipelineCache>,
  pipelines: Res<DdaPipelines>,
  scale: Res<RenderScale>,
  mut profiler: ResMut<crate::profiler::GpuProfilerRes>,
) {
  let (Some(bg0), Some(bg1), Some(bg2), Some(bg3), Some(bg4)) =
    (bg0.as_ref(), bg1.as_ref(), bg2.as_ref(), bg3.as_ref(), bg4.as_ref())
  else {
    bevy::log::debug_once!("DDA dispatch: bind groups missing");
    return;
  };

  let dda_pipe = pipeline_cache.get_compute_pipeline(pipelines.compute_pipeline).or_else(|| {
    bevy::log::debug_once!("DDA dispatch: dda pipeline not ready");
    None
  });
  let beam_pipe = pipeline_cache.get_compute_pipeline(pipelines.beam_pipeline);

  let fog_read = fog.read_bg.as_ref().and_then(|b| b.0.as_ref());

  let gx = scale.size.x.div_ceil(DDA_WORKGROUP_SIZE);
  let gy = scale.size.y.div_ceil(DDA_WORKGROUP_SIZE);

  let bx = scale.size.x.div_ceil(4).div_ceil(crate::brickmap::consts::WORKGROUP_SIZE);
  let by = scale.size.y.div_ceil(4).div_ceil(crate::brickmap::consts::WORKGROUP_SIZE);

  if DDA_BEAM && let Some(beam_pipe) = beam_pipe {
    crate::profiler::gpu_compute_pass(&mut profiler, ctx.command_encoder(), "gate_beam", |pass| {
      pass.set_pipeline(beam_pipe);
      pass.set_bind_group(0, &bg0.0, &[]);
      pass.set_bind_group(1, &bg1.0, &[]);
      pass.set_bind_group(2, &bg2.0, &[]);
      pass.set_bind_group(3, &bg3.0, &[]);
      pass.set_bind_group(4, &bg4.0, &[]);
      pass.dispatch_workgroups(bx, by, 1);
    });
  }

  if gi.as_ref().is_some_and(|g| g.enabled)
    && let Some(aux) = aux.as_ref()
    && let Some(gi_bg0) = aux.gi_bg0.as_ref()
    && let Some(bg5) = bg5.as_ref()
    && let Some(gi_pipe) = pipeline_cache.get_compute_pipeline(pipelines.gi_pipeline)
  {
    let gx = aux.gi_size.x.div_ceil(DDA_WORKGROUP_SIZE);
    let gy = aux.gi_size.y.div_ceil(DDA_WORKGROUP_SIZE);

    if let Some(fs) = aux.face_slots_buffer() {
      ctx.command_encoder().clear_buffer(fs, 0, None);
    }
    crate::profiler::gpu_compute_pass(&mut profiler, ctx.command_encoder(), "gate_gi", |pass| {
      pass.set_pipeline(gi_pipe);
      pass.set_bind_group(0, gi_bg0, &[]);
      pass.set_bind_group(1, &bg1.0, &[]);
      pass.set_bind_group(2, &bg2.0, &[]);
      pass.set_bind_group(3, &bg3.0, &[]);
      pass.set_bind_group(4, &bg4.0, &[]);
      pass.set_bind_group(5, &bg5.0, &[]);
      pass.dispatch_workgroups(gx, gy, 1);
    });
  }

  if gi.as_ref().is_some_and(|g| g.enabled)
    && let Some(aux) = aux.as_ref()
    && let Some(gi_gpu) = gi_gpu.as_ref()
    && let Some(bg) = aux.gi_flatten_bg.as_ref()
    && let Some(pipe) =
      gi_gpu.flatten_pipeline.and_then(|id| pipeline_cache.get_compute_pipeline(id))
  {
    let gx = aux.gi_size.x.div_ceil(DDA_WORKGROUP_SIZE);
    let gy = aux.gi_size.y.div_ceil(DDA_WORKGROUP_SIZE);
    crate::profiler::gpu_compute_pass(
      &mut profiler,
      ctx.command_encoder(),
      "gate_gi_face_flatten",
      |pass| {
        pass.set_pipeline(pipe);
        pass.set_bind_group(0, bg, &[]);
        pass.dispatch_workgroups(gx, gy, 1);
      },
    );
  }

  if let Some(cur) = gi.as_ref()
    && cur.enabled
    && let Some(aux) = aux.as_ref()
    && let Some(gi_gpu) = gi_gpu.as_ref()
  {
    let plan = cur.denoise_plan();
    let n = plan.rounds.clamp(0, 5) as usize;
    if n > 0 {
      let gx = aux.gi_size.x.div_ceil(DDA_WORKGROUP_SIZE);
      let gy = aux.gi_size.y.div_ceil(DDA_WORKGROUP_SIZE);
      let rounds = &DEN_ATROUS_ROUNDS[n - 1];
      if let Some(bg) = aux.den_bg[0].as_ref()
        && let Some(pipe) =
          gi_gpu.den_pipelines[0].and_then(|id| pipeline_cache.get_compute_pipeline(id))
      {
        crate::profiler::gpu_compute_pass(
          &mut profiler,
          ctx.command_encoder(),
          "gate_gi_denoise_temporal",
          |pass| {
            pass.set_pipeline(pipe);
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(gx, gy, 1);
          },
        );
      }
      for i in 0..n {
        let Some(bg) = aux.den_bg[rounds[i] + 1].as_ref() else {
          continue;
        };
        let Some(pipe) =
          gi_gpu.den_pipelines[i + 1].and_then(|id| pipeline_cache.get_compute_pipeline(id))
        else {
          continue;
        };
        let label = [
          "gate_gi_denoise_atrous1",
          "gate_gi_denoise_atrous2",
          "gate_gi_denoise_atrous4",
          "gate_gi_denoise_atrous8",
          "gate_gi_denoise_atrous16",
        ][i];
        crate::profiler::gpu_compute_pass(&mut profiler, ctx.command_encoder(), label, |pass| {
          pass.set_pipeline(pipe);
          pass.set_bind_group(0, bg, &[]);
          pass.dispatch_workgroups(gx, gy, 1);
        });
      }
    }
  }

  if gi.as_ref().is_some_and(|g| g.enabled)
    && let Some(aux) = aux.as_ref()
  {
    let gx = aux.gi_size.x.div_ceil(DDA_WORKGROUP_SIZE);
    let gy = aux.gi_size.y.div_ceil(DDA_WORKGROUP_SIZE);
    if let Some(accum_pipe) = pipeline_cache.get_compute_pipeline(pipelines.face_accum_pipeline) {
      crate::profiler::gpu_compute_pass(
        &mut profiler,
        ctx.command_encoder(),
        "gate_dda_face_accum",
        |pass| {
          pass.set_pipeline(accum_pipe);
          pass.set_bind_group(0, &bg0.0, &[]);
          pass.set_bind_group(1, &bg1.0, &[]);
          pass.set_bind_group(2, &bg2.0, &[]);
          pass.set_bind_group(3, &bg3.0, &[]);
          pass.set_bind_group(4, &bg4.0, &[]);
          if let Some(gi_read) = aux.gi_read_bg.as_ref() {
            pass.set_bind_group(5, gi_read, &[]);
          }
          if let Some(fog_read) = fog_read {
            pass.set_bind_group(6, fog_read, &[]);
          }
          pass.dispatch_workgroups(gx, gy, 1);
        },
      );
    }
  }

  crate::volumetric::dispatch_fog(
    &mut profiler,
    ctx.command_encoder(),
    fog.gpu.as_deref(),
    fog.settings.as_deref(),
    &pipeline_cache,
    aux.as_ref().and_then(|a| a.gi_bg0()),
    Some(&bg1.0),
    Some(&bg2.0),
    Some(&bg3.0),
  );

  if fog_read.is_none() {
    bevy::log::debug_once!("DDA dispatch: 光柱 group(6) 未就绪（本帧跳过主 pass）");
    return;
  }
  if let Some(dda_pipe) = dda_pipe {
    crate::profiler::gpu_compute_pass(
      &mut profiler,
      ctx.command_encoder(),
      "gate_dda_trace",
      |pass| {
        pass.set_pipeline(dda_pipe);
        pass.set_bind_group(0, &bg0.0, &[]);
        pass.set_bind_group(1, &bg1.0, &[]);
        pass.set_bind_group(2, &bg2.0, &[]);
        pass.set_bind_group(3, &bg3.0, &[]);
        pass.set_bind_group(4, &bg4.0, &[]);

        if let Some(gi_read) = aux.as_ref().and_then(|a| a.gi_read_bg.as_ref()) {
          pass.set_bind_group(5, gi_read, &[]);
        }

        if let Some(fog_read) = fog_read {
          pass.set_bind_group(6, fog_read, &[]);
        }
        pass.dispatch_workgroups(gx, gy, 1);
      },
    );
  }

  if eye.as_ref().is_some_and(|e| e.settings.enabled)
    && let Some(eye_bg) = eye.as_ref().and_then(|e| e.bg.as_ref())
    && let Some(h) = pipeline_cache.get_compute_pipeline(pipelines.eye_histogram_pipeline)
    && let Some(u) = pipeline_cache.get_compute_pipeline(pipelines.eye_update_pipeline)
  {
    crate::profiler::gpu_compute_pass(
      &mut profiler,
      ctx.command_encoder(),
      "gate_eye_histogram",
      |pass| {
        pass.set_pipeline(h);

        for i in 0..8u32 {
          pass.set_bind_group(i, eye_bg, &[]);
        }
        pass.dispatch_workgroups(1, 1, 1);
      },
    );
    crate::profiler::gpu_compute_pass(
      &mut profiler,
      ctx.command_encoder(),
      "gate_eye_update",
      |pass| {
        pass.set_pipeline(u);
        for i in 0..8u32 {
          pass.set_bind_group(i, eye_bg, &[]);
        }
        pass.dispatch_workgroups(1, 1, 1);
      },
    );
  }
}

#[cfg_attr(not(feature = "profile"), allow(unused_variables, unused_mut))]
#[allow(clippy::too_many_arguments)]
fn blit_dda_view(
  mut ctx: RenderContext,
  views: Query<&ViewTarget>,
  blit_bg: Option<Res<DdaBlitBindGroup>>,
  post: Option<Res<PostFxSettings>>,
  scale: Option<Res<RenderScale>>,
  pipeline_cache: Res<PipelineCache>,
  pipelines: Res<DdaPipelines>,
  mut profiler: ResMut<crate::profiler::GpuProfilerRes>,
) {
  let (Some(bg), Ok(target)) = (blit_bg.as_ref(), views.single()) else {
    bevy::log::debug_once!("DDA blit: bg or ViewTarget missing");
    return;
  };

  let downscaled = scale.as_ref().is_some_and(|s| s.factor != 1);
  let id = if post.as_ref().is_some_and(|p| p.fxaa) && !downscaled {
    pipelines.blit_fxaa_pipeline
  } else {
    pipelines.blit_pipeline
  };
  let Some(pipe) = pipeline_cache.get_render_pipeline(id) else {
    bevy::log::debug_once!("DDA blit: blit pipeline not ready");
    return;
  };

  #[cfg(feature = "profile")]
  if let Some(profiler) = crate::profiler::profiler_mut(&mut profiler) {
    let mut encoder_scope = profiler.scope("gate_dda_blit", ctx.command_encoder());
    let mut pass = encoder_scope.scoped_render_pass(
      "gate_dda_blit",
      RenderPassDescriptor {
        label: Some("gate_dda_blit"),
        color_attachments: &[Some(target.get_color_attachment())],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        ..default()
      },
    );
    pass.set_pipeline(pipe);
    pass.set_bind_group(0, &bg.0, &[]);
    pass.draw(0..3, 0..1);
    return;
  }
  let mut pass = ctx
    .command_encoder()
    .begin_render_pass(&RenderPassDescriptor {
      label: Some("gate_dda_blit"),
      color_attachments: &[Some(target.get_color_attachment())],
      depth_stencil_attachment: None,
      timestamp_writes: None,
      occlusion_query_set: None,
      ..default()
    })
    .forget_lifetime();
  pass.set_pipeline(pipe);
  pass.set_bind_group(0, &bg.0, &[]);
  pass.draw(0..3, 0..1);
}
