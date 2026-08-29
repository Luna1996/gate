// P0.3 阶段 B：全屏 compute 渐变 → storage texture → blit 直写 ViewTarget
// 链路：compute dispatch（camera_driver 前）→ 渲染图相机 core pass → blit pass 写 view target → 上屏
// Sprite 中转已移除

use bevy::{
  asset::RenderAssetUsages,
  core_pipeline::schedule::{Core2d, Core2dSystems, camera_driver},
  prelude::*,
  render::{
    Render, RenderApp, RenderStartup, RenderSystems,
    extract_resource::{ExtractResource, ExtractResourcePlugin},
    render_asset::RenderAssets,
    render_resource::{
      binding_types::{texture_2d, texture_storage_2d, uniform_buffer},
      *,
    },
    renderer::{RenderContext, RenderDevice, RenderGraph, RenderQueue},
    texture::GpuImage,
    view::ViewTarget,
  },
};

use std::borrow::Cow;

pub const SHADER_ASSET_PATH: &str = "shaders/gradient.wgsl";
pub const BLIT_SHADER_ASSET_PATH: &str = "shaders/blit.wgsl";
pub const VIEW_SIZE: UVec2 = UVec2::new(1280, 720);
const WORKGROUP_SIZE: u32 = 8;

pub struct GradientPlugin;

impl Plugin for GradientPlugin {
  fn build(&self, app: &mut App) {
    app.add_plugins((
      ExtractResourcePlugin::<GradientImages>::default(),
      ExtractResourcePlugin::<GradientUniforms>::default(),
    ));

    let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
      return;
    };
    render_app
      .add_systems(RenderStartup, init_gradient_pipeline)
      .add_systems(
        Render,
        prepare_bind_group.in_set(RenderSystems::PrepareBindGroups),
      )
      // compute 在 camera_driver 前写 storage texture（阶段 A 验证过的时序）
      .add_systems(RenderGraph, dispatch_gradient.before(camera_driver))
      // blit 挂在 Core2d 的 PostProcess set：MainPass（含 clear）之后、
      // upscaling（上屏 copy）之前。camera_driver 之后挂载无效——
      // surface copy 在 camera_driver 内部的相机图末尾就完成了
      .add_systems(Core2d, blit_view.in_set(Core2dSystems::PostProcess));
  }
}

/// compute 写入目标纹理（main world 创建，自动提取进 render world）
#[derive(Resource, Clone, ExtractResource)]
pub struct GradientImages {
  pub target: Handle<Image>,
}

#[derive(Resource, Clone, ExtractResource, ShaderType)]
pub struct GradientUniforms {
  pub size: Vec4, // xy = viewport 尺寸, zw = 0（16B 对齐）
}

#[derive(Resource)]
struct GradientImageBindGroup(BindGroup);

/// blit 采样同一纹理（bind group 独立，layout 不同）
#[derive(Resource)]
struct BlitBindGroup(BindGroup);

#[derive(Resource)]
struct GradientPipeline {
  layout: BindGroupLayoutDescriptor,
  pipeline: CachedComputePipelineId,
  blit_layout: BindGroupLayoutDescriptor,
  blit_pipeline: CachedRenderPipelineId,
}

fn init_gradient_pipeline(
  mut commands: Commands,
  asset_server: Res<AssetServer>,
  pipeline_cache: Res<PipelineCache>,
) {
  let layout = BindGroupLayoutDescriptor::new(
    "GradientLayout",
    &BindGroupLayoutEntries::sequential(
      ShaderStages::COMPUTE,
      (
        texture_storage_2d(TextureFormat::Rgba8Unorm, StorageTextureAccess::WriteOnly),
        uniform_buffer::<GradientUniforms>(false),
      ),
    ),
  );

  let shader = asset_server.load(SHADER_ASSET_PATH);
  let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
    layout: vec![layout.clone()],
    shader,
    entry_point: Some(Cow::from("gradient")),
    ..default()
  });

  // blit：纹理采样（float，免 sampler）+ 全屏三角 fragment 输出
  // 目标格式 = Bevy view target 主纹理格式（Camera2d 非 HDR → Rgba8UnormSrgb）
  let blit_layout = BindGroupLayoutDescriptor::new(
    "GradientBlitLayout",
    // textureLoad 用法，不参与过滤 → filterable: false
    &BindGroupLayoutEntries::single(
      ShaderStages::FRAGMENT,
      texture_2d(TextureSampleType::Float { filterable: false }),
    ),
  );
  let blit_shader = asset_server.load(BLIT_SHADER_ASSET_PATH);
  let blit_pipeline = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
    label: Some("gate_blit_pipeline".into()),
    layout: vec![blit_layout.clone()],
    vertex: VertexState {
      shader: blit_shader.clone(),
      entry_point: Some(Cow::from("vs_main")),
      ..default()
    },
    fragment: Some(FragmentState {
      shader: blit_shader,
      entry_point: Some(Cow::from("fs_main")),
      targets: vec![Some(ColorTargetState {
        format: TextureFormat::Rgba8UnormSrgb,
        blend: None,
        write_mask: ColorWrites::ALL,
      })],
      ..default()
    }),
    ..default()
  });

  commands.insert_resource(GradientPipeline {
    layout,
    pipeline,
    blit_layout,
    blit_pipeline,
  });
}

// Bevy render system 参数即依赖注入，8 个参数属常态（Bevy 源码同款 allow）
#[allow(clippy::too_many_arguments)]
fn prepare_bind_group(
  mut commands: Commands,
  pipeline: Res<GradientPipeline>,
  gpu_images: Res<RenderAssets<GpuImage>>,
  images: Res<GradientImages>,
  uniforms: Res<GradientUniforms>,
  render_device: Res<RenderDevice>,
  pipeline_cache: Res<PipelineCache>,
  queue: Res<RenderQueue>,
) {
  let Some(view) = gpu_images.get(&images.target) else {
    return;
  };

  let mut uniform_buffer = UniformBuffer::from(uniforms.into_inner());
  uniform_buffer.write_buffer(&render_device, &queue);

  let bind_group = render_device.create_bind_group(
    None,
    &pipeline_cache.get_bind_group_layout(&pipeline.layout),
    &BindGroupEntries::sequential((&view.texture_view, &uniform_buffer)),
  );

  let blit_bind_group = render_device.create_bind_group(
    None,
    &pipeline_cache.get_bind_group_layout(&pipeline.blit_layout),
    &BindGroupEntries::single(&view.texture_view),
  );

  commands.insert_resource(GradientImageBindGroup(bind_group));
  commands.insert_resource(BlitBindGroup(blit_bind_group));
}

fn dispatch_gradient(
  mut render_context: RenderContext,
  bind_group: Option<Res<GradientImageBindGroup>>,
  pipeline_cache: Res<PipelineCache>,
  pipeline: Res<GradientPipeline>,
) {
  let Some(bind_group) = bind_group.as_ref() else {
    return;
  };
  // shader 尚未加载 / pipeline 未编译完成时跳过本帧
  let Some(pipeline) = pipeline_cache.get_compute_pipeline(pipeline.pipeline) else {
    return;
  };

  let mut pass = render_context
    .command_encoder()
    .begin_compute_pass(&ComputePassDescriptor::default());
  pass.set_bind_group(0, &bind_group.0, &[]);
  pass.set_pipeline(pipeline);
  pass.dispatch_workgroups(
    VIEW_SIZE.x / WORKGROUP_SIZE,
    VIEW_SIZE.y / WORKGROUP_SIZE,
    1,
  );
}

/// 全屏 blit：storage texture → view target（相机 surface）
/// 挂载在 Core2d 的 PostProcess set：MainPass（含 clear）后、upscaling 上屏前
fn blit_view(
  mut render_context: RenderContext,
  views: Query<&ViewTarget>,
  blit_bind_group: Option<Res<BlitBindGroup>>,
  pipeline_cache: Res<PipelineCache>,
  pipeline: Res<GradientPipeline>,
) {
  let (Some(bind_group), Ok(target)) = (blit_bind_group.as_ref(), views.single()) else {
    return;
  };
  let Some(pipeline) = pipeline_cache.get_render_pipeline(pipeline.blit_pipeline) else {
    return;
  };

  let pass = render_context
    .command_encoder()
    .begin_render_pass(&RenderPassDescriptor {
      label: Some("gate_blit"),
      color_attachments: &[Some(target.get_color_attachment())],
      depth_stencil_attachment: None,
      timestamp_writes: None,
      occlusion_query_set: None,
      ..default()
    });
  let mut pass = pass.forget_lifetime();
  pass.set_pipeline(pipeline);
  pass.set_bind_group(0, &bind_group.0, &[]);
  pass.draw(0..3, 0..1);
}

/// 创建渐变目标纹理（RENDER_WORLD 专用，storage + 采样双用途）
pub fn create_gradient_image(images: &mut Assets<Image>) -> Handle<Image> {
  let mut image =
    Image::new_target_texture(VIEW_SIZE.x, VIEW_SIZE.y, TextureFormat::Rgba8Unorm, None);
  image.asset_usage = RenderAssetUsages::RENDER_WORLD;
  image.texture_descriptor.usage = TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;
  images.add(image)
}
