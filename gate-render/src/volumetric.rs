use bevy::render::render_resource::{
  BindGroupLayoutDescriptor, CachedComputePipelineId, ShaderType,
};
use glam::{Mat4, UVec2, Vec2, Vec4};

use crate::brickmap::dda::DDA_WORKGROUP_SIZE;

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, ShaderType)]
pub struct FogUniform {
  pub params: Vec4,
  pub misc: Vec4,
  pub misc2: Vec4,
  pub misc3: Vec4,
}

#[derive(bevy::ecs::resource::Resource, Clone, Copy, Debug, PartialEq)]
pub struct FogSettings {
  pub enabled: bool,
  pub body: bool,
  pub strength: f32,
  pub decay: f32,
  pub focus: f32,
  pub sun_cone: f32,
  pub halo: f32,
}

pub const FOG_DIV: u32 = 4;

impl FogSettings {
  pub fn strength(&self) -> f32 {
    self.strength.max(0.0)
  }

  pub fn decay(&self) -> f32 {
    self.decay.clamp(0.05, 1.0)
  }

  pub fn focus(&self) -> f32 {
    self.focus.clamp(1.0, 64.0)
  }

  pub fn grid_size(&self, render_size: UVec2) -> UVec2 {
    UVec2::new((render_size.x / FOG_DIV).max(1), (render_size.y / FOG_DIV).max(1))
  }
}

impl Default for FogSettings {
  fn default() -> Self {
    Self {
      enabled: true,
      body: true,
      strength: 0.5,
      decay: 0.6,
      focus: 20.0,
      sun_cone: 2.4_f32.to_radians(),
      halo: 0.5,
    }
  }
}

fn fog_layout(name: &'static str, slots: &[(u32, FogSlot)]) -> BindGroupLayoutDescriptor {
  use bevy::render::render_resource::*;
  const C: ShaderStages = ShaderStages::COMPUTE;
  let entries: Vec<BindGroupLayoutEntry> = slots
    .iter()
    .map(|(binding, slot)| BindGroupLayoutEntry {
      binding: *binding,
      visibility: C,
      ty: match slot {
        FogSlot::Uniform => BindingType::Buffer {
          ty: BufferBindingType::Uniform,
          has_dynamic_offset: false,
          min_binding_size: Some(FogUniform::min_size()),
        },
        FogSlot::Sample => BindingType::Texture {
          sample_type: TextureSampleType::Float { filterable: false },
          view_dimension: TextureViewDimension::D2,
          multisampled: false,
        },
        FogSlot::Write => BindingType::StorageTexture {
          access: StorageTextureAccess::WriteOnly,
          format: TextureFormat::Rgba16Float,
          view_dimension: TextureViewDimension::D2,
        },
      },
      count: None,
    })
    .collect();
  BindGroupLayoutDescriptor::new(name, &entries)
}

#[derive(Clone, Copy)]
enum FogSlot {
  Uniform,
  Sample,
  Write,
}

pub fn fog_layout_mask() -> BindGroupLayoutDescriptor {
  fog_layout("FogGodrayMask", &[(1, FogSlot::Uniform), (3, FogSlot::Write)])
}

pub fn fog_layout_blur_a() -> BindGroupLayoutDescriptor {
  fog_layout(
    "FogGodrayBlurA",
    &[(21, FogSlot::Uniform), (22, FogSlot::Sample), (25, FogSlot::Write)],
  )
}

pub fn fog_layout_blur_b() -> BindGroupLayoutDescriptor {
  fog_layout(
    "FogGodrayBlurB",
    &[(21, FogSlot::Uniform), (24, FogSlot::Sample), (23, FogSlot::Write)],
  )
}

pub fn fog_read_layout() -> BindGroupLayoutDescriptor {
  use bevy::render::render_resource::*;
  const C: ShaderStages = ShaderStages::COMPUTE;
  BindGroupLayoutDescriptor::new(
    "FogGodrayRead",
    &[
      BindGroupLayoutEntry {
        binding: 0,
        visibility: C,
        ty: BindingType::Buffer {
          ty: BufferBindingType::Uniform,
          has_dynamic_offset: false,
          min_binding_size: Some(FogUniform::min_size()),
        },
        count: None,
      },
      BindGroupLayoutEntry {
        binding: 1,
        visibility: C,
        ty: BindingType::Texture {
          sample_type: TextureSampleType::Float { filterable: false },
          view_dimension: TextureViewDimension::D2,
          multisampled: false,
        },
        count: None,
      },
    ],
  )
}

#[derive(Default)]
struct FogLayouts {
  mask: Option<bevy::render::render_resource::BindGroupLayout>,
  blur_a: Option<bevy::render::render_resource::BindGroupLayout>,
  blur_b: Option<bevy::render::render_resource::BindGroupLayout>,
  read: Option<bevy::render::render_resource::BindGroupLayout>,
}

#[derive(bevy::ecs::resource::Resource, Default)]
pub struct FogGpu {
  pub uniform: bevy::render::render_resource::UniformBuffer<FogUniform>,
  size: UVec2,
  tex_a: Option<bevy::render::render_resource::Texture>,
  a_src: Option<bevy::render::render_resource::TextureView>,
  a_dst: Option<bevy::render::render_resource::TextureView>,
  tex_b: Option<bevy::render::render_resource::Texture>,
  b_src: Option<bevy::render::render_resource::TextureView>,
  b_dst: Option<bevy::render::render_resource::TextureView>,
  mask_bg: Option<bevy::render::render_resource::BindGroup>,
  blur_a_bg: Option<bevy::render::render_resource::BindGroup>,
  blur_b_bg: Option<bevy::render::render_resource::BindGroup>,
  read_bg: Option<bevy::render::render_resource::BindGroup>,
  ph_tex: Option<bevy::render::render_resource::Texture>,
  ph_view: Option<bevy::render::render_resource::TextureView>,
  pipelines: [Option<CachedComputePipelineId>; 3],
  layouts: FogLayouts,
}

fn placeholder(
  device: &bevy::render::renderer::RenderDevice,
  gpu: &mut FogGpu,
) -> bevy::render::render_resource::TextureView {
  use bevy::render::render_resource::*;
  if gpu.ph_tex.is_none() {
    let t = device.create_texture(&TextureDescriptor {
      label: Some("gate_godray_placeholder"),
      size: Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
      mip_level_count: 1,
      sample_count: 1,
      dimension: TextureDimension::D2,
      format: TextureFormat::Rgba16Float,
      usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
      view_formats: &[],
    });
    gpu.ph_view = Some(t.create_view(&TextureViewDescriptor::default()));
    gpu.ph_tex = Some(t);
  }
  gpu.ph_view.as_ref().expect("刚创建").clone()
}

fn project_sun(view_proj: Mat4, cam_pos: glam::Vec3, to_sun: glam::Vec3) -> (Vec2, f32) {
  const SUN_PROJECT_DIST: f32 = 4096.0;
  if to_sun.length_squared() < 1e-6 {
    return (Vec2::new(0.5, 0.5), 0.0);
  }
  let p = cam_pos + to_sun * SUN_PROJECT_DIST;
  let clip = view_proj * p.extend(1.0);
  if clip.w <= 1e-3 {
    return (Vec2::new(0.5, 0.5), 0.0);
  }
  let ndc = clip.truncate() / clip.w;
  let uv = Vec2::new(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
  let uv = Vec2::new(uv.x.clamp(-0.25, 1.25), uv.y.clamp(-0.25, 1.25));
  let t = (clip.w / SUN_PROJECT_DIST / 0.25).clamp(0.0, 1.0);
  (uv, t * t * (3.0 - 2.0 * t))
}

pub struct FogPlugin;

impl bevy::app::Plugin for FogPlugin {
  fn build(&self, app: &mut bevy::app::App) {
    use bevy::ecs::schedule::IntoScheduleConfigs;
    app.init_resource::<FogSettings>();
    let Some(render_app) = app.get_sub_app_mut(bevy::render::RenderApp) else {
      return;
    };
    render_app
      .init_resource::<FogSettings>()
      .init_resource::<FogGpu>()
      .add_systems(bevy::render::RenderStartup, init_fog_gpu)
      .add_systems(
        bevy::render::RenderStartup,
        queue_fog_pipelines.after(crate::brickmap::dda::init_dda_pipelines),
      )
      .add_systems(bevy::render::ExtractSchedule, extract_fog_settings)
      .add_systems(
        bevy::render::Render,
        prepare_fog.in_set(bevy::render::RenderSystems::PrepareBindGroups),
      );
  }
}

fn init_fog_gpu(mut commands: bevy::ecs::system::Commands) {
  let d = FogSettings::default();
  bevy::log::info!(
    target: "gate",
    "光柱（godray）：默认 {} / 1÷{} 网格（固定）/ 强度 {} / 衰减 {} / 集中度 {} / 天体盘角径 {}° / 光晕 {}",
    if d.enabled { "开" } else { "关" },
    FOG_DIV,
    d.strength(),
    d.decay(),
    d.focus(),
    d.sun_cone.to_degrees(),
    d.halo,
  );
  commands.insert_resource(FogGpu::default());
}

fn queue_fog_pipelines(
  pipeline_cache: bevy::ecs::system::Res<bevy::render::render_resource::PipelineCache>,
  dda_shader: bevy::ecs::system::Res<crate::shader::DdaShaderHandle>,
  dda: bevy::ecs::system::Res<crate::brickmap::dda::DdaPipelines>,
  mut gpu: bevy::ecs::system::ResMut<FogGpu>,
) {
  use bevy::render::render_resource::ComputePipelineDescriptor;
  use std::borrow::Cow;
  if gpu.pipelines[0].is_some() {
    return;
  }
  gpu.pipelines[0] = Some(pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
    label: Some(Cow::from("gate_godray")),
    layout: vec![
      dda.bg0_gi_layout.clone(),
      dda.bg1_layout.clone(),
      dda.bg2_layout.clone(),
      dda.bg3_layout.clone(),
      fog_layout_mask(),
    ],
    shader: dda_shader.0.clone(),
    entry_point: Some(Cow::from("godray_main")),
    ..Default::default()
  }));
  for (i, (entry, layout)) in
    [("godray_blur_a", fog_layout_blur_a()), ("godray_blur_b", fog_layout_blur_b())]
      .into_iter()
      .enumerate()
  {
    gpu.pipelines[i + 1] = Some(pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
      label: Some(Cow::from(match i {
        0 => "gate_godray_blur_a",
        _ => "gate_godray_blur_b",
      })),
      layout: vec![layout],
      shader: dda_shader.0.clone(),
      entry_point: Some(Cow::from(entry)),
      ..Default::default()
    }));
  }
}

fn extract_fog_settings(
  mut commands: bevy::ecs::system::Commands,
  settings: Option<bevy::render::Extract<bevy::ecs::system::Res<FogSettings>>>,
) {
  commands.insert_resource(settings.map_or_else(FogSettings::default, |s| **s));
}

#[allow(clippy::too_many_arguments)]
fn prepare_fog(
  mut commands: bevy::ecs::system::Commands,
  device: bevy::ecs::system::Res<bevy::render::renderer::RenderDevice>,
  queue: bevy::ecs::system::Res<bevy::render::renderer::RenderQueue>,
  pipeline_cache: bevy::ecs::system::Res<bevy::render::render_resource::PipelineCache>,
  settings: bevy::ecs::system::Res<FogSettings>,
  view: Option<bevy::ecs::system::Res<crate::brickmap::dda::DdaViewUniform>>,
  scale: Option<bevy::ecs::system::Res<crate::brickmap::dda::RenderScale>>,
  lighting: Option<bevy::ecs::system::Res<crate::lighting::LightingTheme>>,
  mut gpu: bevy::ecs::system::ResMut<FogGpu>,
) {
  use bevy::render::render_resource::*;
  let sun = lighting
    .as_ref()
    .and_then(|l| l.sun.as_ref())
    .map(|s| -glam::Vec3::from_array(s.dir).normalize_or_zero())
    .unwrap_or_default();
  let (view_proj, cam_pos) = view
    .as_deref()
    .map_or((Mat4::IDENTITY, glam::Vec3::ZERO), |v| (v.view_proj, v.cam_pos_voxel.truncate()));
  let (sun_uv, facing) = project_sun(view_proj, cam_pos, sun);

  let size = scale.as_deref().map_or(crate::consts::VIEW_SIZE, |s| s.size);
  *gpu.uniform.get_mut() = FogUniform {
    params: Vec4::new(settings.decay(), 0.0, 0.0, 0.0),
    misc: Vec4::new(0.0, settings.focus(), settings.strength(), facing),
    misc2: Vec4::new(
      if settings.body { 1.0 } else { 0.0 },
      0.0,
      settings.sun_cone.max(0.0),
      settings.halo.max(0.0),
    ),
    misc3: Vec4::new(sun_uv.x, sun_uv.y, 0.0, 0.0),
  };
  gpu.uniform.write_buffer(&device, &queue);

  let grid = settings.grid_size(size);
  if gpu.tex_a.is_none() || gpu.size != grid {
    let make_tex = |label: &str| {
      device.create_texture(&TextureDescriptor {
        label: Some(label),
        size: Extent3d { width: grid.x, height: grid.y, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba16Float,
        usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
      })
    };
    let a = make_tex("gate_godray_a");
    gpu.a_src = Some(a.create_view(&TextureViewDescriptor::default()));
    gpu.a_dst = Some(a.create_view(&TextureViewDescriptor::default()));
    gpu.tex_a = Some(a);
    let b = make_tex("gate_godray_b");
    gpu.b_src = Some(b.create_view(&TextureViewDescriptor::default()));
    gpu.b_dst = Some(b.create_view(&TextureViewDescriptor::default()));
    gpu.tex_b = Some(b);
    gpu.size = grid;
    bevy::log::debug!(
      target: "gate",
      "光柱资源 → {}×{}（渲染 {}÷{}）；两张中间靶 rgba16f",
      grid.x,
      grid.y,
      size.x,
      FOG_DIV,
    );
  }

  let (mask_layout, blur_a_layout, blur_b_layout, read_layout) = {
    let l = &mut gpu.layouts;
    (
      l.mask
        .get_or_insert_with(|| pipeline_cache.get_bind_group_layout(&fog_layout_mask()))
        .clone(),
      l.blur_a
        .get_or_insert_with(|| pipeline_cache.get_bind_group_layout(&fog_layout_blur_a()))
        .clone(),
      l.blur_b
        .get_or_insert_with(|| pipeline_cache.get_bind_group_layout(&fog_layout_blur_b()))
        .clone(),
      l.read
        .get_or_insert_with(|| pipeline_cache.get_bind_group_layout(&fog_read_layout()))
        .clone(),
    )
  };
  let ph_view = placeholder(&device, &mut gpu);
  if let (Some(a_src), Some(a_dst), Some(b_src), Some(b_dst)) =
    (gpu.a_src.clone(), gpu.a_dst.clone(), gpu.b_src.clone(), gpu.b_dst.clone())
  {
    let uniform = gpu.uniform.binding().expect("uniform 已写入");
    let mask_bg = device.create_bind_group(
      None,
      &mask_layout,
      &[
        BindGroupEntry { binding: 1, resource: uniform.clone() },
        BindGroupEntry { binding: 3, resource: BindingResource::TextureView(&a_dst) },
      ],
    );
    let blur_a_bg = device.create_bind_group(
      None,
      &blur_a_layout,
      &[
        BindGroupEntry { binding: 21, resource: uniform.clone() },
        BindGroupEntry { binding: 22, resource: BindingResource::TextureView(&a_src) },
        BindGroupEntry { binding: 25, resource: BindingResource::TextureView(&b_dst) },
      ],
    );
    let blur_b_bg = device.create_bind_group(
      None,
      &blur_b_layout,
      &[
        BindGroupEntry { binding: 21, resource: uniform.clone() },
        BindGroupEntry { binding: 24, resource: BindingResource::TextureView(&b_src) },
        BindGroupEntry { binding: 23, resource: BindingResource::TextureView(&a_dst) },
      ],
    );
    let final_view =
      if settings.enabled && settings.strength() > 0.0 { a_src.clone() } else { ph_view.clone() };
    let read_bg = device.create_bind_group(
      None,
      &read_layout,
      &[
        BindGroupEntry { binding: 0, resource: uniform },
        BindGroupEntry { binding: 1, resource: BindingResource::TextureView(&final_view) },
      ],
    );
    gpu.mask_bg = Some(mask_bg);
    gpu.blur_a_bg = Some(blur_a_bg);
    gpu.blur_b_bg = Some(blur_b_bg);
    gpu.read_bg = Some(read_bg);
  } else {
    gpu.mask_bg = None;
    gpu.blur_a_bg = None;
    gpu.blur_b_bg = None;
    gpu.read_bg = Some(device.create_bind_group(
      None,
      &read_layout,
      &[
        BindGroupEntry { binding: 0, resource: gpu.uniform.binding().expect("uniform 已写入") },
        BindGroupEntry { binding: 1, resource: BindingResource::TextureView(&ph_view) },
      ],
    ));
  }
  commands.insert_resource(FogReadBg(gpu.read_bg.clone()));
}

#[derive(bevy::ecs::resource::Resource)]
pub struct FogReadBg(pub Option<bevy::render::render_resource::BindGroup>);

#[derive(bevy::ecs::system::SystemParam)]
pub(crate) struct FogRes<'w> {
  pub settings: Option<bevy::ecs::system::Res<'w, FogSettings>>,
  pub gpu: Option<bevy::ecs::system::Res<'w, FogGpu>>,
  pub read_bg: Option<bevy::ecs::system::Res<'w, FogReadBg>>,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatch_fog(
  profiler: &mut crate::profiler::GpuProfilerRes,
  encoder: &mut bevy::render::render_resource::CommandEncoder,
  gpu: Option<&FogGpu>,
  settings: Option<&FogSettings>,
  pipeline_cache: &bevy::render::render_resource::PipelineCache,
  bg0: Option<&bevy::render::render_resource::BindGroup>,
  bg1: Option<&bevy::render::render_resource::BindGroup>,
  bg2: Option<&bevy::render::render_resource::BindGroup>,
  bg3: Option<&bevy::render::render_resource::BindGroup>,
) {
  let (Some(gpu), Some(s)) = (gpu, settings) else { return };
  if !s.enabled || s.strength() <= 0.0 {
    return;
  }
  let (Some(bg0), Some(bg1), Some(bg2), Some(bg3)) = (bg0, bg1, bg2, bg3) else {
    return;
  };
  let Some(mask_bg) = gpu.mask_bg.as_ref() else {
    return;
  };
  let gx = gpu.size.x.div_ceil(DDA_WORKGROUP_SIZE);
  let gy = gpu.size.y.div_ceil(DDA_WORKGROUP_SIZE);
  if gx == 0 || gy == 0 {
    return;
  }
  if let Some(pipe) = gpu.pipelines[0].and_then(|id| pipeline_cache.get_compute_pipeline(id)) {
    crate::profiler::gpu_compute_pass(profiler, encoder, "gate_godray", |pass| {
      pass.set_pipeline(pipe);
      pass.set_bind_group(0, bg0, &[]);
      pass.set_bind_group(1, bg1, &[]);
      pass.set_bind_group(2, bg2, &[]);
      pass.set_bind_group(3, bg3, &[]);
      pass.set_bind_group(4, mask_bg, &[]);
      pass.dispatch_workgroups(gx, gy, 1);
    });
  }
  for (i, (label, bg)) in
    [("gate_godray_blur_a", gpu.blur_a_bg.as_ref()), ("gate_godray_blur_b", gpu.blur_b_bg.as_ref())]
      .into_iter()
      .enumerate()
  {
    let Some(bg) = bg else { continue };
    let Some(pipe) = gpu.pipelines[i + 1].and_then(|id| pipeline_cache.get_compute_pipeline(id))
    else {
      continue;
    };
    crate::profiler::gpu_compute_pass(profiler, encoder, label, |pass| {
      pass.set_pipeline(pipe);
      pass.set_bind_group(0, bg, &[]);
      pass.dispatch_workgroups(gx, gy, 1);
    });
  }
}
