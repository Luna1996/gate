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

#[derive(Resource, Clone, Copy, Debug, Default, ExtractResource)]
#[extract_app(bevy::render::RenderApp)]
pub struct DlssFrame {
  pub active: bool,
  pub out_size: UVec2,
  pub frame: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DlssMode {
  #[default]
  Off,
  Dlaa,
  Quality,
  Balanced,
  Performance,
  Ultra,
}

impl DlssMode {
  pub const CHOICES: [DlssMode; 6] =
    [Self::Off, Self::Dlaa, Self::Quality, Self::Balanced, Self::Performance, Self::Ultra];

  pub fn ratio(self) -> f32 {
    match self {
      Self::Off | Self::Dlaa => 1.0,
      Self::Quality => 2.0 / 3.0,
      Self::Balanced => 0.577,
      Self::Performance => 0.5,
      Self::Ultra => 1.0 / 3.0,
    }
  }
}

#[derive(Resource, Clone, Copy, Default, ExtractResource)]
#[extract_app(bevy::render::RenderApp)]
pub struct DlssSettings {
  pub mode: DlssMode,
  pub supported: bool,
}

#[derive(Resource, Default)]
pub struct DlssBlitSource(pub Option<std::sync::Arc<TextureView>>);

#[derive(Resource, Default)]
pub struct DdaColorView(pub Option<std::sync::Arc<TextureView>>);

fn advance_dlss_frame(mut frame: ResMut<DlssFrame>, settings: Res<DlssSettings>) {
  frame.active = settings.mode != DlssMode::Off;
  frame.frame = frame.frame.wrapping_add(1);
}

fn halton(mut i: u32, base: u32) -> f32 {
  let mut f = 1.0;
  let mut r = 0.0;
  while i > 0 {
    f /= base as f32;
    r += f * (i % base) as f32;
    i /= base;
  }
  r
}

impl DlssFrame {
  pub fn jitter(&self, render_size: UVec2) -> Vec2 {
    if !self.active || self.out_size.x == 0 || render_size.x == 0 {
      return Vec2::ZERO;
    }
    let ratio = self.out_size.x as f32 / render_size.x as f32;
    let phase_count = (8.0 * ratio * ratio).max(1.0) as u32;
    let i = self.frame % phase_count;
    Vec2::new(halton(i, 2) - 0.5, halton(i, 3) - 0.5)
  }
}

impl Default for RenderScale {
  fn default() -> Self {
    Self { size: crate::consts::VIEW_SIZE, factor: 1 }
  }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TonemapMode {
  Off,
  #[default]
  AgX,
  Aces,
  Neutral,
}

impl TonemapMode {
  pub const COUNT: usize = 4;

  pub fn from_index(i: usize) -> Self {
    match i {
      1 => Self::AgX,
      2 => Self::Aces,
      3 => Self::Neutral,
      _ => Self::Off,
    }
  }

  pub fn as_u32(self) -> u32 {
    match self {
      Self::Off => 0,
      Self::AgX => 1,
      Self::Aces => 2,
      Self::Neutral => 3,
    }
  }
}

#[derive(Resource, Clone, Copy, Debug, PartialEq, Eq, ExtractResource)]
#[extract_app(bevy::render::RenderApp)]
pub struct PostFxSettings {
  pub fxaa: bool,
  pub tonemap: TonemapMode,
}

impl Default for PostFxSettings {
  fn default() -> Self {
    Self { fxaa: false, tonemap: TonemapMode::AgX }
  }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, ShaderType)]
pub struct PostUniform {
  pub mode: u32,
  pub bloom: f32,
  pub threshold: f32,
  pub pad: f32,
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

  pub lod: Vec4,

  pub dlss: Vec4,
}

use crate::brickmap::consts::{DDA_BEAM, DDA_DIR_LUT, DDA_LOD, EYE_ADAPT};

pub fn px_ang(render_h: f32) -> f32 {
  2.0 * (crate::brickmap::consts::DDA_FOV_Y * 0.5).tan() / render_h.max(1.0)
}

impl DdaViewUniform {
  pub fn from_cfg(cfg: &DdaCameraConfig, render_h: f32) -> Self {
    let px_ang = px_ang(render_h);
    Self {
      view_proj: cfg.view_proj,
      inv_view_proj: cfg.inv_view_proj,
      cam_pos_voxel: cfg.position_world.extend(1.0),
      lod: Vec4::new(
        px_ang,
        DDA_LOD as u32 as f32,
        !DDA_BEAM as u32 as f32,
        !DDA_DIR_LUT as u32 as f32,
      ),
      dlss: Vec4::ZERO,
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
      BindGroup, BindGroupEntries, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
      BindGroupLayoutEntries, BindingResource, CachedComputePipelineId, CachedRenderPipelineId,
      ColorTargetState, ColorWrites, ComputePipeline, ComputePipelineDescriptor, Extent3d,
      FilterMode, FragmentState, PipelineCache, RenderPassDescriptor, SamplerBindingType,
      ShaderStages, StorageTextureAccess, TextureDescriptor, TextureDimension, TextureFormat,
      TextureSampleType, TextureUsages, TextureViewDescriptor, UniformBuffer, VertexState,
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
pub struct DdaBlitBindGroup(pub BindGroup);

#[derive(Resource)]
pub(crate) struct LightPoolGpu(UniformBuffer<LightPoolUniform>);

#[derive(Resource, Default)]
pub(crate) struct AuxTexCache {
  texture: Option<Texture>,
  size: UVec2,
  gi_size: UVec2,
  dda_hit: Option<Texture>,
  dda_hit_view: Option<TextureView>,

  gi_bg0: Option<BindGroup>,

  refl_write_bg: Option<BindGroup>,
  refl_read_bg: Option<BindGroup>,
  refl_compose_bg0: Option<BindGroup>,
  refl_cap: u64,
  refl_req: Option<Buffer>,
  refl_sum: Option<Buffer>,
  refl_ray: Option<Buffer>,
  refl_first: Option<Buffer>,
  refl_meta: Option<Buffer>,
  refl_cfg: Option<Buffer>,
  glass_req: Option<Buffer>,
  glass_meta: Option<Buffer>,
  out_base: Option<Texture>,
  out_base_view: Option<TextureView>,
  out_base_size: UVec2,
  hdr_scene: Option<Texture>,
  hdr_scene_view: Option<TextureView>,
  bloom_a: Option<Texture>,
  bloom_a_view: Option<TextureView>,
  bloom_b: Option<Texture>,
  bloom_b_view: Option<TextureView>,
  post_sampler: Option<Sampler>,
  post_buf: Option<UniformBuffer<PostUniform>>,
  post_size: UVec2,
  post_down_bg: Option<BindGroup>,
  post_blur_h_bg: Option<BindGroup>,
  post_blur_v_bg: Option<BindGroup>,
  post_tonemap_bg: Option<BindGroup>,

  perf_vis: Option<Texture>,
  perf_vis_view: Option<TextureView>,
  perf_nrm: Option<Texture>,
  perf_nrm_view: Option<TextureView>,
  perf_shadow: Option<Texture>,
  perf_shadow_view: Option<TextureView>,
  dlss_mvec: Option<Texture>,
  pub(crate) dlss_mvec_view: Option<TextureView>,
  dlss_depth: Option<Texture>,
  pub(crate) dlss_depth_view: Option<TextureView>,
  perf_size: UVec2,
  perf_bg0: Option<BindGroup>,
}

impl AuxTexCache {
  pub(crate) fn glass_meta_buffer(&self) -> Option<&Buffer> {
    self.glass_meta.as_ref()
  }

  pub(crate) fn refl_meta_buffer(&self) -> Option<&Buffer> {
    self.refl_meta.as_ref()
  }

  pub(crate) fn post_ready(&self) -> bool {
    self.post_down_bg.is_some()
      && self.post_blur_h_bg.is_some()
      && self.post_blur_v_bg.is_some()
      && self.post_tonemap_bg.is_some()
  }
}

#[derive(Clone)]
pub enum ComputePipe {
  Bevy(CachedComputePipelineId),
  Raw(std::sync::Arc<std::sync::OnceLock<ComputePipeline>>),
}

impl ComputePipe {
  pub fn get<'a>(&'a self, cache: &'a PipelineCache) -> Option<&'a ComputePipeline> {
    match self {
      ComputePipe::Bevy(id) => cache.get_compute_pipeline(*id),
      ComputePipe::Raw(p) => p.get(),
    }
  }
}

pub(crate) fn raw_compute_pipe(
  device: &RenderDevice,
  cache: &PipelineCache,
  gate: &crate::pipe_cache::GatePipeCache,
  label: &'static str,
  layouts: &[BindGroupLayoutDescriptor],
  entry: &'static str,
) -> Option<ComputePipe> {
  let module = gate.module.clone()?;
  let pc = gate.pc.clone()?;
  let bgs: Vec<BindGroupLayout> = layouts.iter().map(|d| cache.get_bind_group_layout(d)).collect();
  let slot = std::sync::Arc::new(std::sync::OnceLock::new());
  let slot_out = slot.clone();
  let device = device.wgpu_device().clone();
  std::thread::Builder::new()
    .name(format!("gate-pipe-{entry}"))
    .spawn(move || {
      let t0 = std::time::Instant::now();
      let bg_refs: Vec<Option<&wgpu::BindGroupLayout>> = bgs.iter().map(|b| Some(&**b)).collect();
      let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &bg_refs,
        immediate_size: 0,
      });
      let pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        module: &module,
        entry_point: Some(entry),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: Some(&pc),
      });
      info!(target: "gate", "PIPE {entry} 编译 {:.1}s", t0.elapsed().as_secs_f32());
      let _ = slot_out.set(ComputePipeline::from(pipe));
    })
    .expect("管线编译线程创建失败");
  Some(ComputePipe::Raw(slot))
}

#[derive(Resource)]
pub(crate) struct DdaPipelines {
  pub(crate) bg0_layout: BindGroupLayoutDescriptor,

  pub(crate) bg0_gi_layout: BindGroupLayoutDescriptor,

  pub(crate) bg1_layout: BindGroupLayoutDescriptor,
  pub(crate) bg2_layout: BindGroupLayoutDescriptor,
  pub(crate) bg3_layout: BindGroupLayoutDescriptor,
  blit_layout: BindGroupLayoutDescriptor,

  eye_layout: BindGroupLayoutDescriptor,
  bg0_perf_layout: BindGroupLayoutDescriptor,
  visible_pipeline: ComputePipe,
  sun_pipeline: ComputePipe,
  shadow_pipeline: ComputePipe,
  shade_pipeline: ComputePipe,
  glass_pipeline: ComputePipe,
  refl_write_layout: BindGroupLayoutDescriptor,
  refl_read_layout: BindGroupLayoutDescriptor,
  refl_pipeline: ComputePipe,
  compose_pipeline: ComputePipe,
  post_layout: BindGroupLayoutDescriptor,
  bloom_down_pipeline: ComputePipe,
  bloom_h_pipeline: ComputePipe,
  bloom_v_pipeline: ComputePipe,
  tonemap_pipeline: ComputePipe,

  pub(crate) beam_pipeline: ComputePipe,

  eye_histogram_pipeline: ComputePipe,
  eye_update_pipeline: ComputePipe,
  pub blit_pipeline: CachedRenderPipelineId,

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
      bevy::render::extract_resource::ExtractResourcePlugin::<crate::lighting::AtmosphereSettings>::default(),
      bevy::render::extract_resource::ExtractResourcePlugin::<crate::lighting::CloudSettings>::default(),
      bevy::render::extract_resource::ExtractResourcePlugin::<EyeAdaptSettings>::default(),
      bevy::render::extract_resource::ExtractResourcePlugin::<DlssFrame>::default(),
      bevy::render::extract_resource::ExtractResourcePlugin::<DlssSettings>::default(),
      crate::responsive::ResponsivePlugin,
    ));
    app.add_systems(bevy::app::Update, advance_dlss_frame);
    app.insert_resource(EyeAdaptSettings::startup());
    app.init_resource::<crate::lighting::ReflectionSettings>();
    app.init_resource::<crate::lighting::BaseSettings>();
    app.init_resource::<crate::lighting::AtmosphereSettings>();
    app.init_resource::<crate::lighting::CloudSettings>();
    app.init_resource::<DlssFrame>();
    app.init_resource::<DlssSettings>();
    let dda_shader = app.world().resource::<crate::shader::DdaShaderHandle>().clone();
    let dda_source = app.world().resource::<crate::shader::DdaShaderSource>().clone();
    let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
      return;
    };
    render_app.insert_resource(dda_shader);
    render_app.insert_resource(dda_source);
    render_app.init_resource::<DlssBlitSource>();
    render_app.init_resource::<DdaColorView>();
    render_app
      .add_systems(bevy::render::ExtractSchedule, extract_camera_config)
      .add_systems(
        RenderStartup,
        (crate::pipe_cache::init_gate_pipe_cache, init_dda_pipelines).chain(),
      )
      .add_systems(Render, sync_eye_adapt_settings.in_set(RenderSystems::PrepareResources))
      .add_systems(
        Render,
        prepare_dda_bind_groups
          .in_set(RenderSystems::PrepareBindGroups)
          .after(sync_eye_adapt_settings)
          .after(super::upload::prepare),
      )
      .add_systems(
        RenderGraph,
        frame_bracket_head.in_set(bevy::render::renderer::RenderGraphSystems::Render),
      )
      .add_systems(
        RenderGraph,
        dispatch_dda
          .in_set(bevy::render::renderer::RenderGraphSystems::Render)
          .before(camera_driver),
      )
      .add_systems(
        Core2d,
        frame_bracket_tail.in_set(Core2dSystems::PostProcess).after(blit_dda_view),
      )
      .add_systems(Core2d, blit_dda_view.in_set(Core2dSystems::PostProcess));
  }
}

fn extract_camera_config(
  mut commands: bevy::ecs::system::Commands,
  cfg: Option<bevy::render::Extract<bevy::ecs::system::Res<crate::brickmap::DdaCameraConfig>>>,
  scale: Option<bevy::render::Extract<bevy::ecs::system::Res<RenderScale>>>,
  dlss: Option<bevy::render::Extract<bevy::ecs::system::Res<DlssFrame>>>,
) {
  let Some(cfg) = cfg else { return };
  let render_h =
    scale.as_deref().map(|s| s.size.y as f32).unwrap_or(crate::consts::VIEW_SIZE.y as f32);
  let mut uniform = DdaViewUniform::from_cfg(&cfg, render_h);
  if let Some(frame) = dlss.as_deref() {
    let render_w = scale.as_deref().map(|s| s.size).unwrap_or(crate::consts::VIEW_SIZE);
    let j = frame.jitter(render_w);
    uniform.dlss = j.extend(frame.out_size.x as f32).extend(frame.out_size.y as f32);
  }
  commands.insert_resource(uniform);
}

fn post_layout() -> BindGroupLayoutDescriptor {
  let texture = |binding: u32| BindGroupLayoutEntry {
    binding,
    visibility: ShaderStages::COMPUTE,
    ty: BindingType::Texture {
      sample_type: TextureSampleType::Float { filterable: true },
      view_dimension: TextureViewDimension::D2,
      multisampled: false,
    },
    count: None,
  };
  let storage = |binding: u32, format: TextureFormat| BindGroupLayoutEntry {
    binding,
    visibility: ShaderStages::COMPUTE,
    ty: BindingType::StorageTexture {
      access: StorageTextureAccess::WriteOnly,
      format,
      view_dimension: TextureViewDimension::D2,
    },
    count: None,
  };
  BindGroupLayoutDescriptor::new(
    "DdaPost",
    &[
      texture(40),
      texture(41),
      storage(42, TextureFormat::Rgba16Float),
      storage(43, TextureFormat::Rgba8Unorm),
      BindGroupLayoutEntry {
        binding: 44,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Sampler(SamplerBindingType::Filtering),
        count: None,
      },
      BindGroupLayoutEntry {
        binding: 45,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
          ty: BufferBindingType::Uniform,
          has_dynamic_offset: false,
          min_binding_size: Some(PostUniform::min_size()),
        },
        count: None,
      },
    ],
  )
}

pub(crate) fn init_dda_pipelines(
  mut commands: Commands,
  asset_server: Res<AssetServer>,
  render_device: Res<RenderDevice>,
  dda_shader: Res<crate::shader::DdaShaderHandle>,
  gate: Res<crate::pipe_cache::GatePipeCache>,
  pipeline_cache: Res<PipelineCache>,
) {
  let mut bg0_entries: Vec<BindGroupLayoutEntry> = BindGroupLayoutEntries::sequential(
    ShaderStages::COMPUTE,
    (
      texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::WriteOnly),
      uniform_buffer::<DdaViewUniform>(false),
      texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::ReadWrite),
      storage_buffer_read_only_sized(false, None),
      texture_storage_2d(TextureFormat::Rgba32Uint, StorageTextureAccess::ReadWrite),
    ),
  )
  .to_vec();
  for (binding, format) in [
    (30u32, TextureFormat::Rgba32Uint),
    (31, TextureFormat::Rgba16Float),
    (32, TextureFormat::Rgba16Float),
  ] {
    bg0_entries.push(BindGroupLayoutEntry {
      binding,
      visibility: ShaderStages::COMPUTE,
      ty: BindingType::StorageTexture {
        access: StorageTextureAccess::ReadWrite,
        format,
        view_dimension: TextureViewDimension::D2,
      },
      count: None,
    });
  }
  let bg0 = BindGroupLayoutDescriptor::new("DdaBg0", &bg0_entries);

  let mut bg0_perf_entries: Vec<BindGroupLayoutEntry> = BindGroupLayoutEntries::sequential(
    ShaderStages::COMPUTE,
    (
      texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::WriteOnly),
      uniform_buffer::<DdaViewUniform>(false),
      texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::ReadWrite),
      storage_buffer_read_only_sized(false, None),
      texture_storage_2d(TextureFormat::Rgba32Uint, StorageTextureAccess::ReadWrite),
    ),
  )
  .to_vec();
  for (binding, format) in [
    (30u32, TextureFormat::Rgba32Uint),
    (31, TextureFormat::Rgba16Float),
    (32, TextureFormat::Rgba16Float),
  ] {
    bg0_perf_entries.push(BindGroupLayoutEntry {
      binding,
      visibility: ShaderStages::COMPUTE,
      ty: BindingType::StorageTexture {
        access: StorageTextureAccess::ReadWrite,
        format,
        view_dimension: TextureViewDimension::D2,
      },
      count: None,
    });
  }
  bg0_perf_entries.push(BindGroupLayoutEntry {
    binding: 33,
    visibility: ShaderStages::COMPUTE,
    ty: BindingType::Buffer {
      ty: BufferBindingType::Storage { read_only: false },
      has_dynamic_offset: false,
      min_binding_size: None,
    },
    count: None,
  });
  bg0_perf_entries.push(BindGroupLayoutEntry {
    binding: 34,
    visibility: ShaderStages::COMPUTE,
    ty: BindingType::Buffer {
      ty: BufferBindingType::Storage { read_only: false },
      has_dynamic_offset: false,
      min_binding_size: None,
    },
    count: None,
  });
  bg0_perf_entries.push(BindGroupLayoutEntry {
    binding: 35,
    visibility: ShaderStages::COMPUTE,
    ty: BindingType::StorageTexture {
      access: StorageTextureAccess::WriteOnly,
      format: TextureFormat::Rg16Float,
      view_dimension: TextureViewDimension::D2,
    },
    count: None,
  });
  bg0_perf_entries.push(BindGroupLayoutEntry {
    binding: 36,
    visibility: ShaderStages::COMPUTE,
    ty: BindingType::StorageTexture {
      access: StorageTextureAccess::WriteOnly,
      format: TextureFormat::R32Float,
      view_dimension: TextureViewDimension::D2,
    },
    count: None,
  });
  let bg0_perf = BindGroupLayoutDescriptor::new("DdaBg0Perf", &bg0_perf_entries);

  let mut bg0_gi_entries: Vec<BindGroupLayoutEntry> = Vec::new();
  bg0_gi_entries.push(BindGroupLayoutEntry {
    binding: 1,
    visibility: ShaderStages::COMPUTE,
    ty: BindingType::Buffer {
      ty: BufferBindingType::Uniform,
      has_dynamic_offset: false,
      min_binding_size: Some(DdaViewUniform::min_size()),
    },
    count: None,
  });
  bg0_gi_entries.push(BindGroupLayoutEntry {
    binding: 2,
    visibility: ShaderStages::COMPUTE,
    ty: BindingType::StorageTexture {
      access: StorageTextureAccess::ReadWrite,
      format: TextureFormat::R32Float,
      view_dimension: TextureViewDimension::D2,
    },
    count: None,
  });
  for (binding, format) in [
    (4u32, TextureFormat::Rgba32Uint),
    (30, TextureFormat::Rgba32Uint),
    (31, TextureFormat::Rgba16Float),
    (32, TextureFormat::Rgba16Float),
  ] {
    bg0_gi_entries.push(BindGroupLayoutEntry {
      binding,
      visibility: ShaderStages::COMPUTE,
      ty: BindingType::StorageTexture {
        access: StorageTextureAccess::ReadWrite,
        format,
        view_dimension: TextureViewDimension::D2,
      },
      count: None,
    });
  }
  let bg0_gi = BindGroupLayoutDescriptor::new("DdaBg0Gi", &bg0_gi_entries);

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

  bg1_entries.push(BindGroupLayoutEntry {
    binding: 14,
    visibility: ShaderStages::COMPUTE,
    ty: BindingType::Texture {
      sample_type: TextureSampleType::Float { filterable: true },
      view_dimension: TextureViewDimension::D2Array,
      multisampled: false,
    },
    count: None,
  });

  bg1_entries.push(BindGroupLayoutEntry {
    binding: 15,
    visibility: ShaderStages::COMPUTE,
    ty: BindingType::Sampler(SamplerBindingType::Filtering),
    count: None,
  });

  bg1_entries.push(BindGroupLayoutEntry {
    binding: 16,
    visibility: ShaderStages::COMPUTE,
    ty: BindingType::Texture {
      sample_type: TextureSampleType::Float { filterable: true },
      view_dimension: TextureViewDimension::D2,
      multisampled: false,
    },
    count: None,
  });

  bg1_entries.push(BindGroupLayoutEntry {
    binding: 17,
    visibility: ShaderStages::COMPUTE,
    ty: BindingType::Sampler(SamplerBindingType::Filtering),
    count: None,
  });

  bg1_entries.push(BindGroupLayoutEntry {
    binding: 18,
    visibility: ShaderStages::COMPUTE,
    ty: BindingType::Texture {
      sample_type: TextureSampleType::Float { filterable: true },
      view_dimension: TextureViewDimension::D3,
      multisampled: false,
    },
    count: None,
  });

  bg1_entries.push(BindGroupLayoutEntry {
    binding: 19,
    visibility: ShaderStages::COMPUTE,
    ty: BindingType::Texture {
      sample_type: TextureSampleType::Float { filterable: true },
      view_dimension: TextureViewDimension::D2,
      multisampled: false,
    },
    count: None,
  });

  let bg1 = BindGroupLayoutDescriptor::new("DdaBg1", &bg1_entries);

  let bg2 = BindGroupLayoutDescriptor::new(
    "DdaBg2",
    &BindGroupLayoutEntries::sequential(
      ShaderStages::COMPUTE,
      (
        storage_buffer_read_only_sized(false, None),
        storage_buffer_read_only_sized(false, None),
        storage_buffer_read_only_sized(false, None),
      ),
    ),
  );

  let bg3 = BindGroupLayoutDescriptor::new(
    "DdaBg3",
    &BindGroupLayoutEntries::sequential(
      ShaderStages::COMPUTE,
      (uniform_buffer::<LightPoolUniform>(false),),
    ),
  );

  let mut refl_all_entries: Vec<BindGroupLayoutEntry> = Vec::new();
  for (binding, rw) in [(10u32, true), (11, true)] {
    refl_all_entries.push(BindGroupLayoutEntry {
      binding,
      visibility: ShaderStages::COMPUTE,
      ty: BindingType::Buffer {
        ty: BufferBindingType::Storage { read_only: !rw },
        has_dynamic_offset: false,
        min_binding_size: None,
      },
      count: None,
    });
  }
  refl_all_entries.push(BindGroupLayoutEntry {
    binding: 12,
    visibility: ShaderStages::COMPUTE,
    ty: BindingType::Buffer {
      ty: BufferBindingType::Uniform,
      has_dynamic_offset: false,
      min_binding_size: None,
    },
    count: None,
  });
  refl_all_entries.push(BindGroupLayoutEntry {
    binding: 23,
    visibility: ShaderStages::COMPUTE,
    ty: BindingType::Buffer {
      ty: BufferBindingType::Storage { read_only: false },
      has_dynamic_offset: false,
      min_binding_size: None,
    },
    count: None,
  });
  refl_all_entries.push(BindGroupLayoutEntry {
    binding: 24,
    visibility: ShaderStages::COMPUTE,
    ty: BindingType::Buffer {
      ty: BufferBindingType::Storage { read_only: false },
      has_dynamic_offset: false,
      min_binding_size: None,
    },
    count: None,
  });
  let refl_write = BindGroupLayoutDescriptor::new("DdaBg7ReflWrite", &refl_all_entries);
  let mut refl_read_entries = refl_all_entries.clone();
  refl_read_entries.push(BindGroupLayoutEntry {
    binding: 20,
    visibility: ShaderStages::COMPUTE,
    ty: BindingType::StorageTexture {
      access: StorageTextureAccess::WriteOnly,
      format: TextureFormat::Rgba16Float,
      view_dimension: TextureViewDimension::D2,
    },
    count: None,
  });
  refl_read_entries.push(BindGroupLayoutEntry {
    binding: 21,
    visibility: ShaderStages::COMPUTE,
    ty: BindingType::Texture {
      sample_type: TextureSampleType::Float { filterable: true },
      view_dimension: TextureViewDimension::D2,
      multisampled: false,
    },
    count: None,
  });
  refl_read_entries.push(BindGroupLayoutEntry {
    binding: 22,
    visibility: ShaderStages::COMPUTE,
    ty: BindingType::Buffer {
      ty: BufferBindingType::Storage { read_only: false },
      has_dynamic_offset: false,
      min_binding_size: None,
    },
    count: None,
  });
  let refl_read = BindGroupLayoutDescriptor::new("DdaBg7ReflRead", &refl_read_entries);

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

  let dda_shader = dda_shader.0.clone();
  let layouts =
    vec![bg0.clone(), bg1.clone(), bg2.clone(), bg3.clone(), crate::gi::gi_bg0_layout()];

  let dda_layouts = {
    let mut v = layouts.clone();
    v.push(crate::gi::gi_bg5_layout());
    v.push(crate::volumetric::fog_read_layout());
    v.push(refl_write.clone());
    v
  };

  let eye_layouts = vec![eye.clone(); 8];

  let dda_layouts_perf = {
    let mut v = dda_layouts.clone();
    v[0] = bg0_perf.clone();
    v
  };
  let refl_layouts = {
    let mut v = dda_layouts.clone();
    v[7] = refl_read.clone();
    v
  };
  let pipe = |label: &'static str,
              layout: Vec<BindGroupLayoutDescriptor>,
              entry: &'static str|
   -> ComputePipe {
    if let Some(raw) =
      raw_compute_pipe(&render_device, &pipeline_cache, &gate, label, &layout, entry)
    {
      return raw;
    }
    ComputePipe::Bevy(pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
      label: Some(Cow::from(label)),
      layout,
      shader: dda_shader.clone(),
      entry_point: Some(Cow::from(entry)),
      ..default()
    }))
  };
  let visible = pipe("gate_dda_visible", dda_layouts_perf.clone(), "dda_visible");
  let sun = pipe("gate_sun_gather", dda_layouts_perf.clone(), "sun_gather");
  let shadow = pipe("gate_dda_shadow", dda_layouts_perf.clone(), "dda_shadow");
  let shade = pipe("gate_dda_shade", dda_layouts_perf.clone(), "dda_shade");
  let glass = pipe("gate_dda_glass", dda_layouts_perf, "dda_glass");

  let refl = pipe("gate_dda_refl", refl_layouts.clone(), "dda_refl");
  let compose = pipe("gate_dda_refl_compose", refl_layouts, "dda_refl_compose");

  let post = post_layout();
  let empty = BindGroupLayoutDescriptor::new("DdaPostEmpty", &[]);
  let post_layouts = vec![
    empty.clone(),
    empty.clone(),
    empty.clone(),
    empty.clone(),
    empty.clone(),
    empty.clone(),
    empty.clone(),
    post.clone(),
  ];
  let bloom_down = pipe("gate_bloom_down", post_layouts.clone(), "bloom_down");
  let bloom_h = pipe("gate_bloom_blur_h", post_layouts.clone(), "bloom_blur_h");
  let bloom_v = pipe("gate_bloom_blur_v", post_layouts.clone(), "bloom_blur_v");
  let tonemap = pipe("gate_post_tonemap", post_layouts, "post_tonemap");

  let beam = pipe("gate_beam", layouts, "beam_main");

  let eye_histogram = pipe("gate_eye_histogram", eye_layouts.clone(), "eye_adapt_histogram");
  let eye_update = pipe("gate_eye_update", eye_layouts, "eye_adapt_update");

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
    bg1_layout: bg1,
    bg2_layout: bg2,
    bg3_layout: bg3,
    blit_layout: blit,
    eye_layout: eye,
    bg0_perf_layout: bg0_perf,
    visible_pipeline: visible,
    sun_pipeline: sun,
    shadow_pipeline: shadow,
    shade_pipeline: shade,
    glass_pipeline: glass,
    refl_write_layout: refl_write,
    refl_read_layout: refl_read,
    refl_pipeline: refl,
    compose_pipeline: compose,
    post_layout: post,
    bloom_down_pipeline: bloom_down,
    bloom_h_pipeline: bloom_h,
    bloom_v_pipeline: bloom_v,
    tonemap_pipeline: tonemap,
    beam_pipeline: beam,
    eye_histogram_pipeline: eye_histogram,
    eye_update_pipeline: eye_update,
    blit_pipeline,
    blit_fxaa_pipeline,
  });
  commands.insert_resource(LightPoolGpu(UniformBuffer::default()));
  commands.insert_resource(AuxTexCache::default());
  commands.insert_resource(EyeAdaptGpu::default());
  commands.insert_resource(DdaReady::default());
}

#[derive(bevy::ecs::system::SystemParam)]
pub(crate) struct DdaTune<'w> {
  pub gi: Option<Res<'w, crate::gi::GiSettings>>,

  pub refl: Option<Res<'w, crate::lighting::ReflectionSettings>>,

  pub base: Option<Res<'w, crate::lighting::BaseSettings>>,

  pub atmo: Option<Res<'w, crate::lighting::AtmosphereSettings>>,

  pub cloud: Option<Res<'w, crate::lighting::CloudSettings>>,

  pub post: Option<Res<'w, PostFxSettings>>,
}

#[derive(bevy::ecs::system::SystemParam)]
pub(crate) struct PrepareDdaBindGroupsParams<'w, 's> {
  commands: Commands<'w, 's>,
  pipelines: Res<'w, DdaPipelines>,
  gpu_images: Res<'w, RenderAssets<GpuImage>>,
  eye: ResMut<'w, EyeAdaptGpu>,
  images: Option<Res<'w, DdaImages>>,
  view_uniform: Option<Res<'w, DdaViewUniform>>,
  gpu_brickmap: Option<Res<'w, GpuBrickMap>>,
  pbr_set: Option<Res<'w, crate::pbr_texture::PbrTextureSet>>,
  mc_set: Option<Res<'w, crate::mc_texture::McTextureSet>>,
  cloud_set: Option<Res<'w, crate::cloud_texture::CloudTextureSet>>,
  cloud_gpu: Option<ResMut<'w, crate::cloud::CloudGpu>>,
  lighting: Option<Res<'w, LightingTheme>>,
  light_gpu: Option<ResMut<'w, LightPoolGpu>>,
  render_device: Res<'w, RenderDevice>,
  pipeline_cache: Res<'w, PipelineCache>,
  queue: Res<'w, RenderQueue>,
  scale: Res<'w, RenderScale>,
  tune: DdaTune<'w>,
  beam_cache: ResMut<'w, AuxTexCache>,
  dlss_src: Res<'w, DlssBlitSource>,
}

pub(crate) fn prepare_dda_bind_groups(p: PrepareDdaBindGroupsParams) {
  let PrepareDdaBindGroupsParams {
    mut commands,
    pipelines,
    gpu_images,
    mut eye,
    images,
    view_uniform,
    gpu_brickmap,
    pbr_set,
    mc_set,
    cloud_set,
    mut cloud_gpu,
    lighting,
    light_gpu,
    render_device,
    pipeline_cache,
    queue,
    scale,
    tune,
    mut beam_cache,
    dlss_src,
  } = p;
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
  if beam_cache.dda_hit.is_none() || beam_cache.gi_size != gi_size {
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
    let hit_tex = make("gate_dda_hit", TextureFormat::Rgba32Uint);
    beam_cache.dda_hit_view = Some(hit_tex.create_view(&TextureViewDescriptor::default()));
    beam_cache.dda_hit = Some(hit_tex);
    beam_cache.gi_size = gi_size;
  }

  if beam_cache.out_base.is_none() || beam_cache.out_base_size != scale.size {
    let refl_px = scale.size.x as u64 * scale.size.y as u64;
    let tex = render_device.create_texture(&TextureDescriptor {
      label: Some("gate_out_base"),
      size: Extent3d { width: scale.size.x, height: scale.size.y, depth_or_array_layers: 1 },
      mip_level_count: 1,
      sample_count: 1,
      dimension: TextureDimension::D2,
      format: TextureFormat::Rgba16Float,
      usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
      view_formats: &[],
    });
    beam_cache.out_base_view = Some(tex.create_view(&TextureViewDescriptor::default()));
    beam_cache.out_base = Some(tex);
    beam_cache.out_base_size = scale.size;

    let refl_cap = refl_px * 2;
    let make_buf = |label: &str, bytes: u64| {
      render_device.create_buffer(&BufferDescriptor {
        label: Some(label),
        size: bytes.max(4),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
      })
    };
    beam_cache.refl_req = Some(make_buf("gate_refl_req", refl_cap * 12 * 4));
    beam_cache.refl_sum = Some(make_buf("gate_refl_sum", refl_px * 3 * 4));
    beam_cache.refl_ray = Some(make_buf("gate_refl_ray", refl_px * 3 * 4));
    let refl_slots = refl_px.next_power_of_two().clamp(1024, 1u64 << 21);
    beam_cache.refl_first = Some(make_buf("gate_refl_first", refl_slots * 4 * 4));
    beam_cache.refl_meta = Some(make_buf("gate_refl_meta", 16));
    let cfg_buf = render_device.create_buffer(&BufferDescriptor {
      label: Some("gate_refl_cfg"),
      size: 16,
      usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
      mapped_at_creation: false,
    });
    let mut b = [0u8; 16];
    b[0..4].copy_from_slice(&(refl_cap as u32).to_le_bytes());
    b[4..8].copy_from_slice(&(refl_px as u32).to_le_bytes());
    queue.write_buffer(&cfg_buf, 0, &b);
    beam_cache.refl_cfg = Some(cfg_buf);
    beam_cache.refl_cap = refl_cap;
    beam_cache.glass_req = Some(make_buf("gate_glass_req", refl_px * 4));
    beam_cache.glass_meta = Some(make_buf("gate_glass_meta", 16));
  }

  if beam_cache.hdr_scene.is_none() || beam_cache.post_size != scale.size {
    let make = |label: &str, width: u32, height: u32| {
      render_device.create_texture(&TextureDescriptor {
        label: Some(label),
        size: Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba16Float,
        usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
      })
    };
    let hdr = make("gate_hdr_scene", scale.size.x, scale.size.y);
    beam_cache.hdr_scene_view = Some(hdr.create_view(&TextureViewDescriptor::default()));
    beam_cache.hdr_scene = Some(hdr);
    let bloom_w = scale.size.x.div_ceil(2).max(1);
    let bloom_h = scale.size.y.div_ceil(2).max(1);
    let a = make("gate_bloom_a", bloom_w, bloom_h);
    beam_cache.bloom_a_view = Some(a.create_view(&TextureViewDescriptor::default()));
    beam_cache.bloom_a = Some(a);
    let b = make("gate_bloom_b", bloom_w, bloom_h);
    beam_cache.bloom_b_view = Some(b.create_view(&TextureViewDescriptor::default()));
    beam_cache.bloom_b = Some(b);
    beam_cache.post_size = scale.size;
  }

  if beam_cache.post_sampler.is_none() {
    beam_cache.post_sampler = Some(render_device.create_sampler(&SamplerDescriptor {
      label: Some("gate_post_sampler"),
      mag_filter: FilterMode::Linear,
      min_filter: FilterMode::Linear,
      mipmap_filter: MipmapFilterMode::Nearest,
      ..default()
    }));
  }

  if beam_cache.post_buf.is_none() {
    beam_cache.post_buf = Some(UniformBuffer::from(PostUniform::default()));
  }

  let post_settings = tune.post.as_deref().copied().unwrap_or_default();
  let bloom_strength = if post_settings.tonemap == TonemapMode::Off { 0.0 } else { 0.08 };
  if let Some(post_buf) = beam_cache.post_buf.as_mut() {
    *post_buf.get_mut() = PostUniform {
      mode: post_settings.tonemap.as_u32(),
      bloom: bloom_strength,
      threshold: 1.0,
      pad: 0.0,
    };
    post_buf.write_buffer(&render_device, &queue);
  }

  if beam_cache.perf_vis.is_none() || beam_cache.perf_size != scale.size {
    let make = |label: &str, format: TextureFormat| {
      render_device.create_texture(&TextureDescriptor {
        label: Some(label),
        size: Extent3d { width: scale.size.x, height: scale.size.y, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format,
        usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
      })
    };
    let vis = make("gate_dda_vis", TextureFormat::Rgba32Uint);
    beam_cache.perf_vis_view = Some(vis.create_view(&TextureViewDescriptor::default()));
    beam_cache.perf_vis = Some(vis);
    let nrm = make("gate_dda_nrm", TextureFormat::Rgba16Float);
    beam_cache.perf_nrm_view = Some(nrm.create_view(&TextureViewDescriptor::default()));
    beam_cache.perf_nrm = Some(nrm);
    let shadow = make("gate_dda_shadow", TextureFormat::Rgba16Float);
    beam_cache.perf_shadow_view = Some(shadow.create_view(&TextureViewDescriptor::default()));
    beam_cache.perf_shadow = Some(shadow);
    let mvec = make("gate_dlss_mvec", TextureFormat::Rg16Float);
    beam_cache.dlss_mvec_view = Some(mvec.create_view(&TextureViewDescriptor::default()));
    beam_cache.dlss_mvec = Some(mvec);
    let depth = make("gate_dlss_depth", TextureFormat::R32Float);
    beam_cache.dlss_depth_view = Some(depth.create_view(&TextureViewDescriptor::default()));
    beam_cache.dlss_depth = Some(depth);
    beam_cache.perf_size = scale.size;
  }

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

  let mut bg0_entries_vec: Vec<BindGroupEntry> = BindGroupEntries::sequential((
    beam_cache.out_base_view.as_ref().expect("out_base view 未建"),
    &u,
    &beam_view,
    eye_buf.as_entire_binding(),
    beam_cache.dda_hit_view.as_ref().expect("dda_hit view 未建"),
  ))
  .to_vec();
  bg0_entries_vec.push(BindGroupEntry {
    binding: 30,
    resource: BindingResource::TextureView(
      beam_cache.perf_vis_view.as_ref().expect("dda vis view 未建"),
    ),
  });
  bg0_entries_vec.push(BindGroupEntry {
    binding: 31,
    resource: BindingResource::TextureView(
      beam_cache.perf_nrm_view.as_ref().expect("dda nrm view 未建"),
    ),
  });
  bg0_entries_vec.push(BindGroupEntry {
    binding: 32,
    resource: BindingResource::TextureView(
      beam_cache.perf_shadow_view.as_ref().expect("dda shadow view 未建"),
    ),
  });
  let bg0 = render_device.create_bind_group(None, &bg0_layout, &bg0_entries_vec);

  let gi_bg0 = {
    let layout = pipeline_cache.get_bind_group_layout(&pipelines.bg0_gi_layout);
    render_device.create_bind_group(
      None,
      &layout,
      &[
        BindGroupEntry { binding: 1, resource: u.binding().expect("view uniform 已写入") },
        BindGroupEntry { binding: 2, resource: BindingResource::TextureView(&beam_view) },
        BindGroupEntry {
          binding: 4,
          resource: BindingResource::TextureView(
            beam_cache.dda_hit_view.as_ref().expect("dda_hit view 未建"),
          ),
        },
        BindGroupEntry {
          binding: 30,
          resource: BindingResource::TextureView(
            beam_cache.perf_vis_view.as_ref().expect("dda vis view 未建"),
          ),
        },
        BindGroupEntry {
          binding: 31,
          resource: BindingResource::TextureView(
            beam_cache.perf_nrm_view.as_ref().expect("dda nrm view 未建"),
          ),
        },
        BindGroupEntry {
          binding: 32,
          resource: BindingResource::TextureView(
            beam_cache.perf_shadow_view.as_ref().expect("dda shadow view 未建"),
          ),
        },
      ],
    )
  };
  beam_cache.gi_bg0 = Some(gi_bg0);

  let perf_bg0 =
    if let (Some(vis), Some(nrm), Some(sh), Some(greq), Some(gmeta), Some(mvec), Some(dep)) = (
      beam_cache.perf_vis_view.as_ref(),
      beam_cache.perf_nrm_view.as_ref(),
      beam_cache.perf_shadow_view.as_ref(),
      beam_cache.glass_req.as_ref(),
      beam_cache.glass_meta.as_ref(),
      beam_cache.dlss_mvec_view.as_ref(),
      beam_cache.dlss_depth_view.as_ref(),
    ) {
      let layout = pipeline_cache.get_bind_group_layout(&pipelines.bg0_perf_layout);
      let mut entries: Vec<BindGroupEntry> = BindGroupEntries::sequential((
        beam_cache.out_base_view.as_ref().expect("out_base view 未建"),
        &u,
        &beam_view,
        eye_buf.as_entire_binding(),
        beam_cache.dda_hit_view.as_ref().expect("dda_hit view 未建"),
      ))
      .to_vec();
      entries.push(BindGroupEntry { binding: 30, resource: BindingResource::TextureView(vis) });
      entries.push(BindGroupEntry { binding: 31, resource: BindingResource::TextureView(nrm) });
      entries.push(BindGroupEntry { binding: 32, resource: BindingResource::TextureView(sh) });
      entries.push(BindGroupEntry { binding: 33, resource: greq.as_entire_binding() });
      entries.push(BindGroupEntry { binding: 34, resource: gmeta.as_entire_binding() });
      entries.push(BindGroupEntry { binding: 35, resource: BindingResource::TextureView(mvec) });
      entries.push(BindGroupEntry { binding: 36, resource: BindingResource::TextureView(dep) });
      Some(render_device.create_bind_group(None, &layout, &entries))
    } else {
      None
    };
  beam_cache.perf_bg0 = perf_bg0;

  let (refl_write_bg, refl_read_bg) = if let (
    Some(req),
    Some(sum),
    Some(ray),
    Some(first),
    Some(meta),
    Some(cfg),
    Some(base_view),
  ) = (
    beam_cache.refl_req.as_ref(),
    beam_cache.refl_sum.as_ref(),
    beam_cache.refl_ray.as_ref(),
    beam_cache.refl_first.as_ref(),
    beam_cache.refl_meta.as_ref(),
    beam_cache.refl_cfg.as_ref(),
    beam_cache.out_base_view.as_ref(),
  ) {
    let wl = pipeline_cache.get_bind_group_layout(&pipelines.refl_write_layout);
    let rl = pipeline_cache.get_bind_group_layout(&pipelines.refl_read_layout);
    let wbg = render_device.create_bind_group(
      None,
      &wl,
      &[
        BindGroupEntry { binding: 10, resource: req.as_entire_binding() },
        BindGroupEntry { binding: 11, resource: meta.as_entire_binding() },
        BindGroupEntry { binding: 12, resource: cfg.as_entire_binding() },
        BindGroupEntry { binding: 23, resource: first.as_entire_binding() },
        BindGroupEntry { binding: 24, resource: ray.as_entire_binding() },
      ],
    );
    let rbg = render_device.create_bind_group(
      None,
      &rl,
      &[
        BindGroupEntry { binding: 10, resource: req.as_entire_binding() },
        BindGroupEntry { binding: 11, resource: meta.as_entire_binding() },
        BindGroupEntry { binding: 12, resource: cfg.as_entire_binding() },
        BindGroupEntry { binding: 23, resource: first.as_entire_binding() },
        BindGroupEntry { binding: 24, resource: ray.as_entire_binding() },
        BindGroupEntry {
          binding: 20,
          resource: BindingResource::TextureView(
            beam_cache.hdr_scene_view.as_ref().expect("hdr scene view"),
          ),
        },
        BindGroupEntry { binding: 21, resource: BindingResource::TextureView(base_view) },
        BindGroupEntry { binding: 22, resource: sum.as_entire_binding() },
      ],
    );
    (Some(wbg), Some(rbg))
  } else {
    (None, None)
  };

  let post_bgs = if let (Some(hdr), Some(bloom_a), Some(bloom_b), Some(sampler), Some(post_buf)) = (
    beam_cache.hdr_scene_view.as_ref(),
    beam_cache.bloom_a_view.as_ref(),
    beam_cache.bloom_b_view.as_ref(),
    beam_cache.post_sampler.as_ref(),
    beam_cache.post_buf.as_ref(),
  ) {
    let layout = pipeline_cache.get_bind_group_layout(&pipelines.post_layout);
    let post_u = post_buf.binding().expect("post uniform");
    let target = &tex_view.texture_view;
    let make = |src0: &TextureView, src1: &TextureView, dst: &TextureView| {
      render_device.create_bind_group(
        None,
        &layout,
        &[
          BindGroupEntry { binding: 40, resource: BindingResource::TextureView(src0) },
          BindGroupEntry { binding: 41, resource: BindingResource::TextureView(src1) },
          BindGroupEntry { binding: 42, resource: BindingResource::TextureView(dst) },
          BindGroupEntry { binding: 43, resource: BindingResource::TextureView(target) },
          BindGroupEntry { binding: 44, resource: BindingResource::Sampler(sampler) },
          BindGroupEntry { binding: 45, resource: post_u.clone() },
        ],
      )
    };
    (
      Some(make(hdr, hdr, bloom_a)),
      Some(make(bloom_a, hdr, bloom_b)),
      Some(make(bloom_b, hdr, bloom_a)),
      Some(make(hdr, bloom_a, bloom_b)),
    )
  } else {
    (None, None, None, None)
  };
  beam_cache.post_down_bg = post_bgs.0;
  beam_cache.post_blur_h_bg = post_bgs.1;
  beam_cache.post_blur_v_bg = post_bgs.2;
  beam_cache.post_tonemap_bg = post_bgs.3;

  let mut compose_entries: Vec<BindGroupEntry> = BindGroupEntries::sequential((
    beam_cache.bloom_a_view.as_ref().expect("bloom view 未建"),
    &u,
    &beam_view,
    eye_buf.as_entire_binding(),
    beam_cache.dda_hit_view.as_ref().expect("dda_hit view 未建"),
  ))
  .to_vec();
  compose_entries.push(BindGroupEntry {
    binding: 30,
    resource: BindingResource::TextureView(
      beam_cache.perf_vis_view.as_ref().expect("dda vis view 未建"),
    ),
  });
  compose_entries.push(BindGroupEntry {
    binding: 31,
    resource: BindingResource::TextureView(
      beam_cache.perf_nrm_view.as_ref().expect("dda nrm view 未建"),
    ),
  });
  compose_entries.push(BindGroupEntry {
    binding: 32,
    resource: BindingResource::TextureView(
      beam_cache.perf_shadow_view.as_ref().expect("dda shadow view 未建"),
    ),
  });
  let compose_bg0 = render_device.create_bind_group(None, &bg0_layout, &compose_entries);
  beam_cache.refl_compose_bg0 = Some(compose_bg0);
  beam_cache.refl_write_bg = refl_write_bg;
  beam_cache.refl_read_bg = refl_read_bg;

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
  let mc_albedo = mc_set.as_deref().and_then(|s| gpu_images.get(s.albedo()));
  let mc_albedo_view =
    mc_albedo.map_or_else(|| gpu.mc_albedo_view.clone(), |i| i.texture_view.clone());
  let cloud = cloud_set.as_deref().and_then(|s| gpu_images.get(s.weather()));
  let cloud_view = cloud.map_or_else(|| gpu.cloud_view.clone(), |i| i.texture_view.clone());
  let cloud_noise = cloud_set.as_deref().and_then(|s| gpu_images.get(s.noise()));
  let cloud_noise_view =
    cloud_noise.map_or_else(|| gpu.cloud_noise_view.clone(), |i| i.texture_view.clone());
  let cloud_cover = tune.cloud.as_deref().map_or(1.0, |c| c.cover.max(0.0));
  let cloud_time = tune.cloud.as_deref().map_or(0.0, |c| c.time);
  if let Some(cloud_gpu) = cloud_gpu.as_mut() {
    cloud_gpu.prepare(
      &render_device,
      &queue,
      scale.size,
      view_uniform.view_proj,
      cloud_time,
      cloud_cover,
    );
  }
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

    entries.push(BindGroupEntry {
      binding: 14,
      resource: BindingResource::TextureView(&mc_albedo_view),
    });

    entries
      .push(BindGroupEntry { binding: 15, resource: BindingResource::Sampler(&gpu.mc_sampler) });

    entries
      .push(BindGroupEntry { binding: 16, resource: BindingResource::TextureView(&cloud_view) });

    entries
      .push(BindGroupEntry { binding: 17, resource: BindingResource::Sampler(&gpu.cloud_sampler) });

    entries.push(BindGroupEntry {
      binding: 18,
      resource: BindingResource::TextureView(&cloud_noise_view),
    });

    let cloud_out_view = cloud_gpu.as_ref().and_then(|c| c.hist_view()).unwrap_or(&gpu.cloud_view);
    entries
      .push(BindGroupEntry { binding: 19, resource: BindingResource::TextureView(cloud_out_view) });

    render_device.create_bind_group(None, &bg1_layout, &entries)
  };

  let bg2 = render_device.create_bind_group(
    None,
    &bg2_layout,
    &BindGroupEntries::sequential((
      gpu.grid_descs_buf.as_entire_binding(),
      gpu.inst_bvh_buf.as_entire_binding(),
      gpu.obj_pose_rev_buf.as_entire_binding(),
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
  let soft_angle = tune
    .base
    .as_ref()
    .map_or(crate::lighting::BaseSettings::default().soft_angle_deg, |b| b.soft_angle_deg);
  lp.0.get_mut().lights[0].shape.y = soft_angle.clamp(0.0, 5.0).to_radians();
  let vpm = gpu.main_voxel_per_meter.max(1e-6);

  lp.0.get_mut().g.refl_tier = tune.refl.as_ref().map_or(0, |r| r.tier());
  lp.0.get_mut().g.refl_nest = tune.refl.as_ref().map_or(0, |r| r.nest());
  lp.0.get_mut().g.refl_reach = tune.refl.as_ref().map_or(0, |r| r.reach_voxels(vpm));

  lp.0.get_mut().g.base_flags =
    tune.base.as_ref().map_or(crate::lighting::BaseSettings::default().flags(), |b| b.flags());
  let atmo = tune.atmo.as_deref().copied().unwrap_or_default();
  lp.0.get_mut().g.exposure_pad = Vec4::new(
    atmo.density.max(0.0) / vpm,
    atmo.start.max(0.0) * vpm,
    atmo.scatter.max(0.0),
    atmo.anisotropy.clamp(-0.95, 0.95),
  );
  lp.0.write_buffer(&render_device, &queue);
  let bg3 = render_device.create_bind_group(None, &bg3_layout, &BindGroupEntries::single(&lp.0));

  let blit_sampler = render_device.create_sampler(&SamplerDescriptor {
    label: Some("gate_dda_blit_sampler"),
    mag_filter: FilterMode::Linear,
    min_filter: FilterMode::Linear,
    ..default()
  });
  let blit_view = dlss_src.0.as_ref().map_or(&tex_view.texture_view, |v| v.as_ref());
  let blit_bg = render_device.create_bind_group(
    None,
    &blit_layout,
    &BindGroupEntries::sequential((blit_view, &blit_sampler)),
  );

  commands.insert_resource(DdaColorView(Some(std::sync::Arc::new(tex_view.texture_view.clone()))));

  commands.insert_resource(DdaBg0BindGroup(bg0));
  commands.insert_resource(DdaBg1BindGroup(bg1));
  commands.insert_resource(DdaBg2BindGroup(bg2));
  commands.insert_resource(DdaBg3BindGroup(bg3));
  commands.insert_resource(DdaBlitBindGroup(blit_bg));
}

fn frame_bracket_head(
  mut ctx: RenderContext,
  mut profiler: ResMut<crate::profiler::GpuProfilerRes>,
) {
  crate::profiler::gpu_encoder_scope(
    &mut profiler,
    ctx.command_encoder(),
    "gate_frame_head",
    |_| (),
  );
}

fn frame_bracket_tail(
  mut ctx: RenderContext,
  mut profiler: ResMut<crate::profiler::GpuProfilerRes>,
) {
  crate::profiler::gpu_encoder_scope(
    &mut profiler,
    ctx.command_encoder(),
    "gate_frame_tail",
    |_| (),
  );
}

static WARM_T0: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
static WARM_VIS: std::sync::OnceLock<()> = std::sync::OnceLock::new();
static WARM_SHADOW: std::sync::OnceLock<()> = std::sync::OnceLock::new();
static WARM_SHADE: std::sync::OnceLock<()> = std::sync::OnceLock::new();
static WARM_GLASS: std::sync::OnceLock<()> = std::sync::OnceLock::new();
static WARM_REFL: std::sync::OnceLock<()> = std::sync::OnceLock::new();
static WARM_BEAM: std::sync::OnceLock<()> = std::sync::OnceLock::new();
static WARM_GI: std::sync::OnceLock<()> = std::sync::OnceLock::new();
static WARM_EYE: std::sync::OnceLock<()> = std::sync::OnceLock::new();
static WARM_ALL: std::sync::OnceLock<()> = std::sync::OnceLock::new();

fn warm_mark(slot: &'static std::sync::OnceLock<()>, label: &str, ready: bool) {
  if ready && slot.set(()).is_ok() {
    let t0 = *WARM_T0.get_or_init(std::time::Instant::now);
    bevy::log::info!(target: "gate", "WARM {label} 就绪 @{:.1}s", t0.elapsed().as_secs_f32());
  }
}

struct PerfBinds<'a> {
  bg0: &'a BindGroup,
  bg1: &'a BindGroup,
  bg2: &'a BindGroup,
  bg3: &'a BindGroup,
  bg4: &'a BindGroup,
  bg5: &'a BindGroup,
  fog_read: Option<&'a BindGroup>,
  refl_write: Option<&'a BindGroup>,
}

fn bind_perf_groups(pass: &mut ComputePass<'_>, b: &PerfBinds<'_>) {
  pass.set_bind_group(0, b.bg0, &[]);
  pass.set_bind_group(1, b.bg1, &[]);
  pass.set_bind_group(2, b.bg2, &[]);
  pass.set_bind_group(3, b.bg3, &[]);
  pass.set_bind_group(4, b.bg4, &[]);
  pass.set_bind_group(5, b.bg5, &[]);
  if let Some(bg) = b.fog_read {
    pass.set_bind_group(6, bg, &[]);
  }
  if let Some(bg) = b.refl_write {
    pass.set_bind_group(7, bg, &[]);
  }
}

#[derive(bevy::ecs::system::SystemParam)]
pub(crate) struct DispatchDdaParams<'w, 's> {
  ctx: RenderContext<'w, 's>,
  bg0: Option<Res<'w, DdaBg0BindGroup>>,
  bg1: Option<Res<'w, DdaBg1BindGroup>>,
  bg2: Option<Res<'w, DdaBg2BindGroup>>,
  bg3: Option<Res<'w, DdaBg3BindGroup>>,
  bg4: Option<Res<'w, crate::gi::GiGpu>>,
  eye: Option<Res<'w, EyeAdaptGpu>>,
  gi: Option<Res<'w, crate::gi::GiSettings>>,
  fog: crate::volumetric::FogRes<'w>,
  cloud: Option<Res<'w, crate::cloud::CloudGpu>>,
  aux: Option<Res<'w, AuxTexCache>>,
  pipeline_cache: Res<'w, PipelineCache>,
  pipelines: Res<'w, DdaPipelines>,
  gate: Res<'w, crate::pipe_cache::GatePipeCache>,
  scale: Res<'w, RenderScale>,
  ready: ResMut<'w, DdaReady>,
  profiler: ResMut<'w, crate::profiler::GpuProfilerRes>,
}

#[derive(bevy::ecs::resource::Resource, Default)]
pub(crate) struct DdaReady(pub bool);

struct DdaReadyCtx<'a> {
  pipeline_cache: &'a PipelineCache,
  pipelines: &'a DdaPipelines,
  fog: Option<&'a crate::volumetric::FogGpu>,
  bg4: Option<&'a crate::gi::GiGpu>,
  fog_on: bool,
  gi_on: bool,
  eye_on: bool,
  refl_ok: bool,
  post_ok: bool,
}

fn dda_pipelines_ready(cx: &DdaReadyCtx<'_>) -> bool {
  let comp = |p: &ComputePipe| p.get(cx.pipeline_cache).is_some();
  let gi_ok = cx.bg4.is_some_and(|bg| {
    let all = |x: &Option<ComputePipe>| x.as_ref().is_some_and(&comp);
    all(&bg.light_pipeline)
      && all(&bg.gather_pipeline)
      && all(&bg.resolve_pipeline)
      && all(&bg.restir_temporal_pipeline)
      && all(&bg.restir_spatial_pipeline)
      && all(&bg.merge_pipeline)
  });
  comp(&cx.pipelines.visible_pipeline)
    && comp(&cx.pipelines.sun_pipeline)
    && comp(&cx.pipelines.shadow_pipeline)
    && comp(&cx.pipelines.shade_pipeline)
    && comp(&cx.pipelines.glass_pipeline)
    && (!DDA_BEAM || comp(&cx.pipelines.beam_pipeline))
    && cx.bg4.is_some_and(|bg| bg.resolve_pipeline.as_ref().is_some_and(&comp))
    && cx.pipeline_cache.get_render_pipeline(cx.pipelines.blit_pipeline).is_some()
    && cx.pipeline_cache.get_render_pipeline(cx.pipelines.blit_fxaa_pipeline).is_some()
    && (!cx.fog_on || cx.fog.is_some_and(|f| f.pipelines_ready(cx.pipeline_cache)))
    && (!cx.gi_on || gi_ok)
    && (!cx.refl_ok || comp(&cx.pipelines.refl_pipeline) && comp(&cx.pipelines.compose_pipeline))
    && cx.post_ok
    && comp(&cx.pipelines.bloom_down_pipeline)
    && comp(&cx.pipelines.bloom_h_pipeline)
    && comp(&cx.pipelines.bloom_v_pipeline)
    && comp(&cx.pipelines.tonemap_pipeline)
    && (!cx.eye_on
      || comp(&cx.pipelines.eye_histogram_pipeline) && comp(&cx.pipelines.eye_update_pipeline))
}

pub(crate) fn dispatch_dda(p: DispatchDdaParams) {
  let DispatchDdaParams {
    mut ctx,
    bg0,
    bg1,
    bg2,
    bg3,
    bg4,
    eye,
    gi,
    fog,
    cloud,
    aux,
    pipeline_cache,
    pipelines,
    gate,
    scale,
    mut ready,
    mut profiler,
  } = p;
  WARM_T0.get_or_init(std::time::Instant::now);
  let (Some(bg0), Some(bg1), Some(bg2), Some(bg3), Some(bg4)) =
    (bg0.as_ref(), bg1.as_ref(), bg2.as_ref(), bg3.as_ref(), bg4.as_ref())
  else {
    bevy::log::debug_once!("DDA dispatch: bind groups missing");
    return;
  };
  let Some(gi_b0) = bg4.bg0.as_ref() else {
    bevy::log::debug_once!("DDA dispatch: GI uniform bind group missing");
    return;
  };
  let Some(gi_view_bg) = aux.as_ref().and_then(|a| a.gi_bg0.as_ref()) else {
    bevy::log::debug_once!("DDA dispatch: GI view bind group missing");
    return;
  };
  let Some(b5) = bg4.irc_bind_group() else {
    bevy::log::debug_once!("DDA dispatch: GI cache bind group missing");
    return;
  };

  let beam_pipe = pipelines.beam_pipeline.get(&pipeline_cache);

  warm_mark(&WARM_BEAM, "beam", beam_pipe.is_some());
  warm_mark(
    &WARM_GI,
    "gi",
    bg4.gather_pipeline.as_ref().is_some_and(|p| p.get(&pipeline_cache).is_some()),
  );
  warm_mark(&WARM_EYE, "eye_hist", pipelines.eye_histogram_pipeline.get(&pipeline_cache).is_some());

  let fog_read = fog.read_bg.as_ref().and_then(|b| b.0.as_ref());

  let gx = scale.size.x.div_ceil(DDA_WORKGROUP_SIZE);
  let gy = scale.size.y.div_ceil(DDA_WORKGROUP_SIZE);
  let sun_div = crate::wesl_consts::gi_consts().sun_div.max(1);
  let sx = scale.size.x.div_ceil(sun_div).div_ceil(DDA_WORKGROUP_SIZE);
  let sy = scale.size.y.div_ceil(sun_div).div_ceil(DDA_WORKGROUP_SIZE);

  let bx = scale.size.x.div_ceil(4).div_ceil(crate::brickmap::consts::WORKGROUP_SIZE);
  let by = scale.size.y.div_ceil(4).div_ceil(crate::brickmap::consts::WORKGROUP_SIZE);

  let refl_ok = match aux.as_ref() {
    Some(a) => {
      a.refl_sum
        .as_ref()
        .zip(a.refl_meta.as_ref())
        .map(|(sum, meta)| {
          ctx.command_encoder().clear_buffer(sum, 0, None);
          ctx.command_encoder().clear_buffer(meta, 0, None);
          if let Some(first) = a.refl_first.as_ref() {
            ctx.command_encoder().clear_buffer(first, 0, None);
          }
          if let Some(ray) = a.refl_ray.as_ref() {
            ctx.command_encoder().clear_buffer(ray, 0, None);
          }
          true
        })
        .unwrap_or(false)
        && a.refl_write_bg.is_some()
        && a.refl_read_bg.is_some()
        && a.refl_cap > 0
    }
    None => false,
  };
  let refl_write_bg = aux.as_ref().and_then(|a| a.refl_write_bg.as_ref());
  let refl_read_bg = aux.as_ref().and_then(|a| a.refl_read_bg.as_ref());
  let refl_compose_bg0 = aux.as_ref().and_then(|a| a.refl_compose_bg0.as_ref());
  let refl_cap = aux.as_ref().map(|a| a.refl_cap).unwrap_or(0);
  let glass_ok = match aux.as_ref().and_then(|a| a.glass_meta.as_ref()) {
    Some(meta) => {
      ctx.command_encoder().clear_buffer(meta, 0, None);
      true
    }
    None => false,
  };

  if DDA_BEAM && let Some(beam_pipe) = beam_pipe {
    crate::profiler::gpu_compute_pass(&mut profiler, ctx.command_encoder(), "gate_beam", |pass| {
      pass.set_pipeline(beam_pipe);
      pass.set_bind_group(0, &bg0.0, &[]);
      pass.set_bind_group(1, &bg1.0, &[]);
      pass.set_bind_group(2, &bg2.0, &[]);
      pass.set_bind_group(3, &bg3.0, &[]);
      pass.set_bind_group(4, gi_b0, &[]);
      pass.dispatch_workgroups(bx, by, 1);
    });
  }

  crate::volumetric::dispatch_fog(
    &mut profiler,
    ctx.command_encoder(),
    fog.gpu.as_deref(),
    fog.settings.as_deref(),
    &pipeline_cache,
    [Some(gi_view_bg), Some(&bg1.0), Some(&bg2.0), Some(&bg3.0)],
  );

  let main_ready = fog_read.is_some();
  if !main_ready {
    bevy::log::debug_once!("DDA dispatch: 光柱 group(6) 未就绪（本帧跳过主 pass）");
  }

  let perf_bg0 = aux.as_ref().and_then(|a| a.perf_bg0.as_ref());
  let perf_pipes = (
    pipelines.visible_pipeline.get(&pipeline_cache),
    pipelines.sun_pipeline.get(&pipeline_cache),
    pipelines.shadow_pipeline.get(&pipeline_cache),
    pipelines.shade_pipeline.get(&pipeline_cache),
    pipelines.glass_pipeline.get(&pipeline_cache),
  );
  warm_mark(&WARM_VIS, "visible", perf_pipes.0.is_some());
  warm_mark(&WARM_SHADOW, "shadow", perf_pipes.2.is_some());
  warm_mark(&WARM_SHADE, "shade", perf_pipes.3.is_some());
  warm_mark(&WARM_GLASS, "glass", perf_pipes.4.is_some());
  warm_mark(&WARM_REFL, "refl", pipelines.refl_pipeline.get(&pipeline_cache).is_some());
  let perf_ready = main_ready && refl_ok && glass_ok && perf_bg0.is_some();

  let gi_on = gi.as_ref().is_some_and(|g| g.enabled);
  let fog_on = fog.settings.as_deref().is_some_and(|s| s.enabled && s.strength() > 0.0);
  let eye_on = eye.as_ref().is_some_and(|e| e.settings.enabled);
  let all_ready = perf_ready
    && perf_pipes.0.is_some()
    && perf_pipes.1.is_some()
    && perf_pipes.2.is_some()
    && perf_pipes.3.is_some()
    && perf_pipes.4.is_some()
    && dda_pipelines_ready(&DdaReadyCtx {
      pipeline_cache: &pipeline_cache,
      pipelines: &pipelines,
      fog: fog.gpu.as_deref(),
      bg4: Some(&**bg4),
      fog_on,
      gi_on,
      eye_on,
      refl_ok,
      post_ok: aux.as_ref().is_some_and(|a| a.post_ready()),
    });
  warm_mark(&WARM_ALL, "全部", all_ready);
  *ready = DdaReady(all_ready);
  if all_ready {
    crate::pipe_cache::save_once(&gate);
  }
  if !all_ready {
    return;
  }
  if let (Some(vis_pipe), Some(sun_pipe), Some(shadow_pipe), Some(shade_pipe), Some(glass_pipe)) =
    perf_pipes
    && let Some(pbg0) = perf_bg0
  {
    crate::profiler::gpu_compute_pass(
      &mut profiler,
      ctx.command_encoder(),
      "gate_dda_visible",
      |pass| {
        pass.set_pipeline(vis_pipe);
        bind_perf_groups(
          pass,
          &PerfBinds {
            bg0: pbg0,
            bg1: &bg1.0,
            bg2: &bg2.0,
            bg3: &bg3.0,
            bg4: gi_b0,
            bg5: b5,
            fog_read,
            refl_write: refl_write_bg,
          },
        );
        pass.dispatch_workgroups(gx, gy, 1);
      },
    );
    crate::profiler::gpu_compute_pass(
      &mut profiler,
      ctx.command_encoder(),
      "gate_sun_gather",
      |pass| {
        pass.set_pipeline(sun_pipe);
        bind_perf_groups(
          pass,
          &PerfBinds {
            bg0: pbg0,
            bg1: &bg1.0,
            bg2: &bg2.0,
            bg3: &bg3.0,
            bg4: gi_b0,
            bg5: b5,
            fog_read,
            refl_write: refl_write_bg,
          },
        );
        pass.dispatch_workgroups(sx, sy, 1);
      },
    );
    crate::profiler::gpu_compute_pass(
      &mut profiler,
      ctx.command_encoder(),
      "gate_dda_shadow",
      |pass| {
        pass.set_pipeline(shadow_pipe);
        bind_perf_groups(
          pass,
          &PerfBinds {
            bg0: pbg0,
            bg1: &bg1.0,
            bg2: &bg2.0,
            bg3: &bg3.0,
            bg4: gi_b0,
            bg5: b5,
            fog_read,
            refl_write: refl_write_bg,
          },
        );
        pass.dispatch_workgroups(gx, gy, 1);
      },
    );
    if let Some(cloud) = cloud.as_ref()
      && let Some(fog_read) = fog_read
    {
      cloud.dispatch(
        &mut profiler,
        ctx.command_encoder(),
        &pipeline_cache,
        [
          Some(&bg0.0),
          Some(&bg1.0),
          Some(&bg2.0),
          Some(&bg3.0),
          Some(gi_b0),
          Some(b5),
          Some(fog_read),
        ],
      );
    }
    crate::profiler::gpu_compute_pass(
      &mut profiler,
      ctx.command_encoder(),
      "gate_dda_shade",
      |pass| {
        pass.set_pipeline(shade_pipe);
        bind_perf_groups(
          pass,
          &PerfBinds {
            bg0: pbg0,
            bg1: &bg1.0,
            bg2: &bg2.0,
            bg3: &bg3.0,
            bg4: gi_b0,
            bg5: b5,
            fog_read,
            refl_write: refl_write_bg,
          },
        );
        pass.dispatch_workgroups(gx, gy, 1);
      },
    );
    let glass_groups = (scale.size.x * scale.size.y).div_ceil(64);
    crate::profiler::gpu_compute_pass(
      &mut profiler,
      ctx.command_encoder(),
      "gate_dda_glass",
      |pass| {
        pass.set_pipeline(glass_pipe);
        bind_perf_groups(
          pass,
          &PerfBinds {
            bg0: pbg0,
            bg1: &bg1.0,
            bg2: &bg2.0,
            bg3: &bg3.0,
            bg4: gi_b0,
            bg5: b5,
            fog_read,
            refl_write: refl_write_bg,
          },
        );
        pass.dispatch_workgroups(glass_groups, 1, 1);
      },
    );
  }

  if gi_on {
    let gi_size = aux.as_ref().map(|a| a.gi_size).unwrap_or(UVec2::ONE);
    let ggx = gi_size.x.div_ceil(DDA_WORKGROUP_SIZE);
    let ggy = gi_size.y.div_ceil(DDA_WORKGROUP_SIZE);
    if let Some(stats) = bg4.stats_buffer() {
      ctx.command_encoder().clear_buffer(stats, 0, None);
    }
    if let Some(ev) = bg4.lt_event_buffer() {
      ctx.command_encoder().clear_buffer(ev, 0, None);
    }
    {
      let _ = gi_b0;
      if let Some(pipe) = bg4.light_pipeline.as_ref().and_then(|p| p.get(&pipeline_cache)) {
        let groups = (crate::gi::LT_SLOTS as u32).div_ceil(64);
        crate::profiler::gpu_compute_pass(
          &mut profiler,
          ctx.command_encoder(),
          "gate_gi_light",
          |pass| {
            pass.set_pipeline(pipe);
            pass.set_bind_group(0, gi_view_bg, &[]);
            pass.set_bind_group(1, &bg1.0, &[]);
            pass.set_bind_group(2, &bg2.0, &[]);
            pass.set_bind_group(3, &bg3.0, &[]);
            pass.set_bind_group(4, gi_b0, &[]);
            pass.set_bind_group(5, b5, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
          },
        );
      }
      if let Some(pipe) = bg4.restir_temporal_pipeline.as_ref().and_then(|p| p.get(&pipeline_cache))
      {
        crate::profiler::gpu_compute_pass(
          &mut profiler,
          ctx.command_encoder(),
          "gate_gi_restir_temporal",
          |pass| {
            pass.set_pipeline(pipe);
            pass.set_bind_group(0, gi_view_bg, &[]);
            pass.set_bind_group(1, &bg1.0, &[]);
            pass.set_bind_group(2, &bg2.0, &[]);
            pass.set_bind_group(3, &bg3.0, &[]);
            pass.set_bind_group(4, gi_b0, &[]);
            pass.set_bind_group(5, b5, &[]);
            pass.dispatch_workgroups(ggx, ggy, 1);
          },
        );
      }
      if let Some(pipe) = bg4.restir_spatial_pipeline.as_ref().and_then(|p| p.get(&pipeline_cache))
      {
        crate::profiler::gpu_compute_pass(
          &mut profiler,
          ctx.command_encoder(),
          "gate_gi_restir_spatial",
          |pass| {
            pass.set_pipeline(pipe);
            pass.set_bind_group(0, gi_view_bg, &[]);
            pass.set_bind_group(1, &bg1.0, &[]);
            pass.set_bind_group(2, &bg2.0, &[]);
            pass.set_bind_group(3, &bg3.0, &[]);
            pass.set_bind_group(4, gi_b0, &[]);
            pass.set_bind_group(5, b5, &[]);
            pass.dispatch_workgroups(ggx, ggy, 1);
          },
        );
      }
      if let Some(pipe) = bg4.gather_pipeline.as_ref().and_then(|p| p.get(&pipeline_cache)) {
        crate::profiler::gpu_compute_pass(
          &mut profiler,
          ctx.command_encoder(),
          "gate_gi_gather",
          |pass| {
            pass.set_pipeline(pipe);
            pass.set_bind_group(0, gi_view_bg, &[]);
            pass.set_bind_group(1, &bg1.0, &[]);
            pass.set_bind_group(2, &bg2.0, &[]);
            pass.set_bind_group(3, &bg3.0, &[]);
            pass.set_bind_group(4, gi_b0, &[]);
            pass.set_bind_group(5, b5, &[]);
            pass.dispatch_workgroups(ggx, ggy, 1);
          },
        );
      }
      if let Some(pipe) = bg4.merge_pipeline.as_ref().and_then(|p| p.get(&pipeline_cache)) {
        let groups = crate::gi::GI_MERGE_SLOTS.div_ceil(64);
        crate::profiler::gpu_compute_pass(
          &mut profiler,
          ctx.command_encoder(),
          "gate_gi_merge_drain",
          |pass| {
            pass.set_pipeline(pipe);
            pass.set_bind_group(0, gi_view_bg, &[]);
            pass.set_bind_group(1, &bg1.0, &[]);
            pass.set_bind_group(2, &bg2.0, &[]);
            pass.set_bind_group(3, &bg3.0, &[]);
            pass.set_bind_group(4, gi_b0, &[]);
            pass.set_bind_group(5, b5, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
          },
        );
        if let Some(merge) = bg4.gi_merge.as_ref() {
          ctx.command_encoder().clear_buffer(merge, 0, Some(4));
        }
      }
      if let Some((src, dst, bytes)) = bg4.reservoir_copy() {
        ctx.command_encoder().copy_buffer_to_buffer(src, 0, dst, 0, bytes);
      }
    }
  }

  if let Some(pipe) = bg4.resolve_pipeline.as_ref().and_then(|p| p.get(&pipeline_cache)) {
    let groups = bg4.slots().div_ceil(64);
    crate::profiler::gpu_compute_pass(
      &mut profiler,
      ctx.command_encoder(),
      "gate_gi_resolve",
      |pass| {
        pass.set_pipeline(pipe);
        pass.set_bind_group(0, gi_view_bg, &[]);
        pass.set_bind_group(1, &bg1.0, &[]);
        pass.set_bind_group(2, &bg2.0, &[]);
        pass.set_bind_group(3, &bg3.0, &[]);
        pass.set_bind_group(4, gi_b0, &[]);
        pass.set_bind_group(5, b5, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
      },
    );
  }

  if refl_ok
    && let (Some(rr), Some(cb0)) = (refl_read_bg, refl_compose_bg0)
    && let Some(pipe) = pipelines.refl_pipeline.get(&pipeline_cache)
  {
    let groups = ((refl_cap as u32).div_ceil(64)).clamp(1, 1024);
    crate::profiler::gpu_compute_pass(
      &mut profiler,
      ctx.command_encoder(),
      "gate_dda_refl",
      |pass| {
        pass.set_pipeline(pipe);
        pass.set_bind_group(0, cb0, &[]);
        pass.set_bind_group(1, &bg1.0, &[]);
        pass.set_bind_group(2, &bg2.0, &[]);
        pass.set_bind_group(3, &bg3.0, &[]);
        pass.set_bind_group(4, gi_b0, &[]);
        pass.set_bind_group(5, b5, &[]);
        if let Some(fog_read) = fog_read {
          pass.set_bind_group(6, fog_read, &[]);
        }
        pass.set_bind_group(7, rr, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
      },
    );
  }

  if refl_ok
    && let Some(rr) = refl_read_bg
    && let Some(cb0) = refl_compose_bg0
    && let Some(pipe) = pipelines.compose_pipeline.get(&pipeline_cache)
  {
    let gxc = gx.max(1);
    let gyc = gy.max(1);
    crate::profiler::gpu_compute_pass(
      &mut profiler,
      ctx.command_encoder(),
      "gate_dda_refl_compose",
      |pass| {
        pass.set_pipeline(pipe);
        pass.set_bind_group(0, cb0, &[]);
        pass.set_bind_group(1, &bg1.0, &[]);
        pass.set_bind_group(2, &bg2.0, &[]);
        pass.set_bind_group(3, &bg3.0, &[]);
        pass.set_bind_group(4, gi_b0, &[]);
        pass.set_bind_group(5, b5, &[]);
        if let Some(fog_read) = fog_read {
          pass.set_bind_group(6, fog_read, &[]);
        }
        pass.set_bind_group(7, rr, &[]);
        pass.dispatch_workgroups(gxc, gyc, 1);
      },
    );
  }

  let post_bgs = aux.as_ref().and_then(|a| {
    Some((
      a.post_down_bg.as_ref()?,
      a.post_blur_h_bg.as_ref()?,
      a.post_blur_v_bg.as_ref()?,
      a.post_tonemap_bg.as_ref()?,
    ))
  });
  if let Some((down, blur_h, blur_v, tonemap)) = post_bgs {
    let half_x = scale.size.x.div_ceil(2).div_ceil(DDA_WORKGROUP_SIZE).max(1);
    let half_y = scale.size.y.div_ceil(2).div_ceil(DDA_WORKGROUP_SIZE).max(1);
    if let Some(pipe) = pipelines.bloom_down_pipeline.get(&pipeline_cache) {
      crate::profiler::gpu_compute_pass(
        &mut profiler,
        ctx.command_encoder(),
        "gate_bloom_down",
        |pass| {
          pass.set_pipeline(pipe);
          pass.set_bind_group(7, down, &[]);
          pass.dispatch_workgroups(half_x, half_y, 1);
        },
      );
    }
    if let Some(pipe) = pipelines.bloom_h_pipeline.get(&pipeline_cache) {
      crate::profiler::gpu_compute_pass(
        &mut profiler,
        ctx.command_encoder(),
        "gate_bloom_blur_h",
        |pass| {
          pass.set_pipeline(pipe);
          pass.set_bind_group(7, blur_h, &[]);
          pass.dispatch_workgroups(half_x, half_y, 1);
        },
      );
    }
    if let Some(pipe) = pipelines.bloom_v_pipeline.get(&pipeline_cache) {
      crate::profiler::gpu_compute_pass(
        &mut profiler,
        ctx.command_encoder(),
        "gate_bloom_blur_v",
        |pass| {
          pass.set_pipeline(pipe);
          pass.set_bind_group(7, blur_v, &[]);
          pass.dispatch_workgroups(half_x, half_y, 1);
        },
      );
    }
    if let Some(pipe) = pipelines.tonemap_pipeline.get(&pipeline_cache) {
      crate::profiler::gpu_compute_pass(
        &mut profiler,
        ctx.command_encoder(),
        "gate_post_tonemap",
        |pass| {
          pass.set_pipeline(pipe);
          pass.set_bind_group(7, tonemap, &[]);
          pass.dispatch_workgroups(gx.max(1), gy.max(1), 1);
        },
      );
    }
  }

  if eye.as_ref().is_some_and(|e| e.settings.enabled)
    && let Some(eye_bg) = eye.as_ref().and_then(|e| e.bg.as_ref())
    && let Some(h) = pipelines.eye_histogram_pipeline.get(&pipeline_cache)
    && let Some(u) = pipelines.eye_update_pipeline.get(&pipeline_cache)
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

#[derive(bevy::ecs::system::SystemParam)]
struct BlitDdaParams<'w, 's> {
  ctx: RenderContext<'w, 's>,
  views: Query<'w, 's, &'static ViewTarget>,
  ready: Res<'w, DdaReady>,
  blit_bg: Option<Res<'w, DdaBlitBindGroup>>,
  post: Option<Res<'w, PostFxSettings>>,
  dlss: Res<'w, DlssFrame>,
  pipeline_cache: Res<'w, PipelineCache>,
  pipelines: Res<'w, DdaPipelines>,
  #[cfg(feature = "profile")]
  profiler: ResMut<'w, crate::profiler::GpuProfilerRes>,
}

#[cfg_attr(not(feature = "profile"), allow(unused_variables, unused_mut))]
fn blit_dda_view(p: BlitDdaParams) {
  let BlitDdaParams {
    mut ctx,
    views,
    ready,
    blit_bg,
    post,
    dlss,
    pipeline_cache,
    pipelines,
    #[cfg(feature = "profile")]
    mut profiler,
  } = p;
  if !ready.0 {
    return;
  }
  let (Some(bg), Ok(target)) = (blit_bg.as_ref(), views.single()) else {
    bevy::log::debug_once!("DDA blit: bg or ViewTarget missing");
    return;
  };

  let fxaa = post.as_ref().is_some_and(|p| p.fxaa) && !dlss.active;
  let id = if fxaa { pipelines.blit_fxaa_pipeline } else { pipelines.blit_pipeline };
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
