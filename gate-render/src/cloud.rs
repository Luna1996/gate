use bevy::{
  ecs::resource::Resource,
  render::{
    render_resource::{
      AddressMode, BindGroup, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType,
      BufferBindingType, ComputePipelineDescriptor, Extent3d, FilterMode, PipelineCache,
      SamplerBindingType, SamplerDescriptor, ShaderStages, ShaderType, StorageTextureAccess,
      TextureDescriptor, TextureDimension, TextureFormat, TextureSampleType, TextureUsages,
      TextureView, TextureViewDescriptor, TextureViewDimension, UniformBuffer,
    },
    renderer::{RenderDevice, RenderQueue},
  },
};
use glam::{Mat4, UVec2, Vec4};

use crate::brickmap::dda::ComputePipe;

#[derive(ShaderType, Default, Clone, Copy)]
pub struct CloudUniform {
  pub prev_view_proj: Mat4,
  pub params: Vec4,
}

#[derive(Resource, Default)]
pub struct CloudGpu {
  pub uniform: UniformBuffer<CloudUniform>,
  size: UVec2,
  out_tex: Option<bevy::render::render_resource::Texture>,
  out_view: Option<TextureView>,
  hist_tex: Option<bevy::render::render_resource::Texture>,
  hist_view: Option<TextureView>,
  sampler: Option<bevy::render::render_resource::Sampler>,
  bg: Option<BindGroup>,
  layout: Option<bevy::render::render_resource::BindGroupLayout>,
  pub pipeline: Option<ComputePipe>,
  frame: u32,
  prev_view_proj: Mat4,
}

impl CloudGpu {
  pub fn out_view(&self) -> Option<&TextureView> {
    self.out_view.as_ref()
  }

  pub fn pipeline_ready(&self, cache: &PipelineCache) -> bool {
    self.pipeline.as_ref().is_some_and(|p| p.get(cache).is_some())
  }

  pub fn prepare(
    &mut self,
    device: &RenderDevice,
    queue: &RenderQueue,
    size: UVec2,
    view_proj: Mat4,
    time: f32,
    cover: f32,
  ) {
    let half = UVec2::new((size.x / 2).max(1), (size.y / 2).max(1));
    let resized = self.size != half || self.out_tex.is_none();
    if resized {
      let make = |label: &'static str| {
        device.create_texture(&TextureDescriptor {
          label: Some(label),
          size: Extent3d { width: half.x, height: half.y, depth_or_array_layers: 1 },
          mip_level_count: 1,
          sample_count: 1,
          dimension: TextureDimension::D2,
          format: TextureFormat::Rgba16Float,
          usage: TextureUsages::STORAGE_BINDING
            | TextureUsages::TEXTURE_BINDING
            | TextureUsages::COPY_SRC
            | TextureUsages::COPY_DST,
          view_formats: &[],
        })
      };
      let out = make("gate_cloud_out");
      let hist = make("gate_cloud_hist");
      self.out_view = Some(out.create_view(&TextureViewDescriptor::default()));
      self.hist_view = Some(hist.create_view(&TextureViewDescriptor::default()));
      self.out_tex = Some(out);
      self.hist_tex = Some(hist);
      self.size = half;
      self.prev_view_proj = view_proj;
      self.frame = 0;
      self.sampler = Some(device.create_sampler(&SamplerDescriptor {
        label: Some("gate_cloud_history_sampler"),
        address_mode_u: AddressMode::ClampToEdge,
        address_mode_v: AddressMode::ClampToEdge,
        address_mode_w: AddressMode::ClampToEdge,
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        mipmap_filter: bevy::render::render_resource::MipmapFilterMode::Nearest,
        ..Default::default()
      }));
    }

    let valid = if self.frame == 0 { 0.0 } else { 1.0 };
    self.uniform.get_mut().prev_view_proj = self.prev_view_proj;
    self.uniform.get_mut().params = Vec4::new(time, cover.max(0.0), self.frame as f32, valid);
    self.uniform.write_buffer(device, queue);
    self.prev_view_proj = view_proj;
    self.frame = self.frame.wrapping_add(1);

    if let (Some(out), Some(hist), Some(sampler), Some(layout)) =
      (self.out_view.as_ref(), self.hist_view.as_ref(), self.sampler.as_ref(), self.layout.as_ref())
    {
      let uniform = self.uniform.binding().expect("cloud uniform");
      self.bg = Some(device.create_bind_group(
        None,
        layout,
        &[
          bevy::render::render_resource::BindGroupEntry { binding: 50, resource: uniform },
          bevy::render::render_resource::BindGroupEntry {
            binding: 51,
            resource: bevy::render::render_resource::BindingResource::TextureView(out),
          },
          bevy::render::render_resource::BindGroupEntry {
            binding: 52,
            resource: bevy::render::render_resource::BindingResource::TextureView(hist),
          },
          bevy::render::render_resource::BindGroupEntry {
            binding: 53,
            resource: bevy::render::render_resource::BindingResource::Sampler(sampler),
          },
        ],
      ));
    }
  }

  pub(crate) fn dispatch(
    &self,
    profiler: &mut crate::profiler::GpuProfilerRes,
    encoder: &mut bevy::render::render_resource::CommandEncoder,
    cache: &PipelineCache,
    groups: [Option<&BindGroup>; 7],
  ) {
    let (Some(pipe), Some(bg), Some(out), Some(hist)) = (
      self.pipeline.as_ref().and_then(|p| p.get(cache)),
      self.bg.as_ref(),
      self.out_tex.as_ref(),
      self.hist_tex.as_ref(),
    ) else {
      return;
    };
    let [Some(g0), Some(g1), Some(g2), Some(g3), Some(g4), Some(g5), Some(g6)] = groups else {
      return;
    };
    let gx = self.size.x.div_ceil(8);
    let gy = self.size.y.div_ceil(8);
    if gx == 0 || gy == 0 {
      return;
    }
    crate::profiler::gpu_compute_pass(profiler, encoder, "gate_cloud", |pass| {
      pass.set_pipeline(pipe);
      pass.set_bind_group(0, g0, &[]);
      pass.set_bind_group(1, g1, &[]);
      pass.set_bind_group(2, g2, &[]);
      pass.set_bind_group(3, g3, &[]);
      pass.set_bind_group(4, g4, &[]);
      pass.set_bind_group(5, g5, &[]);
      pass.set_bind_group(6, g6, &[]);
      pass.set_bind_group(7, bg, &[]);
      pass.dispatch_workgroups(gx, gy, 1);
    });
    encoder.copy_texture_to_texture(
      out.as_image_copy(),
      hist.as_image_copy(),
      Extent3d { width: self.size.x, height: self.size.y, depth_or_array_layers: 1 },
    );
  }
}

pub fn cloud_layout() -> BindGroupLayoutDescriptor {
  const C: ShaderStages = ShaderStages::COMPUTE;
  BindGroupLayoutDescriptor::new(
    "GateCloud",
    &[
      BindGroupLayoutEntry {
        binding: 50,
        visibility: C,
        ty: BindingType::Buffer {
          ty: BufferBindingType::Uniform,
          has_dynamic_offset: false,
          min_binding_size: Some(CloudUniform::min_size()),
        },
        count: None,
      },
      BindGroupLayoutEntry {
        binding: 51,
        visibility: C,
        ty: BindingType::StorageTexture {
          access: StorageTextureAccess::WriteOnly,
          format: TextureFormat::Rgba16Float,
          view_dimension: TextureViewDimension::D2,
        },
        count: None,
      },
      BindGroupLayoutEntry {
        binding: 52,
        visibility: C,
        ty: BindingType::Texture {
          sample_type: TextureSampleType::Float { filterable: true },
          view_dimension: TextureViewDimension::D2,
          multisampled: false,
        },
        count: None,
      },
      BindGroupLayoutEntry {
        binding: 53,
        visibility: C,
        ty: BindingType::Sampler(SamplerBindingType::Filtering),
        count: None,
      },
    ],
  )
}

pub struct CloudPlugin;

impl bevy::app::Plugin for CloudPlugin {
  fn build(&self, app: &mut bevy::app::App) {
    use bevy::ecs::schedule::IntoScheduleConfigs;
    let Some(render_app) = app.get_sub_app_mut(bevy::render::RenderApp) else {
      return;
    };
    render_app.init_resource::<CloudGpu>().add_systems(
      bevy::render::RenderStartup,
      queue_cloud_pipeline
        .after(crate::brickmap::dda::init_dda_pipelines)
        .after(crate::pipe_cache::init_gate_pipe_cache),
    );
  }
}

fn queue_cloud_pipeline(
  device: bevy::ecs::system::Res<RenderDevice>,
  cache: bevy::ecs::system::Res<PipelineCache>,
  shader: bevy::ecs::system::Res<crate::shader::DdaShaderHandle>,
  gate: bevy::ecs::system::Res<crate::pipe_cache::GatePipeCache>,
  dda: bevy::ecs::system::Res<crate::brickmap::dda::DdaPipelines>,
  mut gpu: bevy::ecs::system::ResMut<CloudGpu>,
) {
  use std::borrow::Cow;
  if gpu.pipeline.is_some() {
    return;
  }
  gpu.layout = Some(cache.get_bind_group_layout(&cloud_layout()));
  let layouts = vec![
    dda.bg0_layout.clone(),
    dda.bg1_layout.clone(),
    dda.bg2_layout.clone(),
    dda.bg3_layout.clone(),
    crate::gi::gi_bg0_layout(),
    crate::gi::gi_bg5_layout(),
    crate::volumetric::fog_read_layout(),
    cloud_layout(),
  ];
  if let Some(raw) = crate::brickmap::dda::raw_compute_pipe(
    &device,
    &cache,
    &gate,
    "gate_cloud",
    &layouts,
    "cloud_main",
  ) {
    gpu.pipeline = Some(raw);
    return;
  }
  gpu.pipeline = Some(ComputePipe::Bevy(cache.queue_compute_pipeline(ComputePipelineDescriptor {
    label: Some(Cow::from("gate_cloud")),
    layout: layouts,
    shader: shader.0.clone(),
    entry_point: Some(Cow::from("cloud_main")),
    ..Default::default()
  })));
}
