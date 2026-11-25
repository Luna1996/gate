use bevy::prelude::*;
use bevy::render::{
  Render, RenderApp, RenderStartup,
  render_resource::{
    Extent3d, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages, TextureView,
    TextureViewDescriptor,
  },
  renderer::{RenderAdapter, RenderContext, RenderDevice, RenderQueue},
};
use glam::Mat4;

use crate::brickmap::dda::{
  AuxTexCache, DdaColorView, DdaViewUniform, DlssBlitSource, DlssFrame, DlssMode, DlssSettings,
  RenderScale,
};

fn ngx_project_id() -> uuid::Uuid {
  uuid::Uuid::from_u128(0x6632_5bfe_724e_491b_a4b1_e6ce_bcd0_92e2)
}

#[derive(Resource, Clone, Default)]
pub struct DlssProbe(pub std::sync::Arc<std::sync::Mutex<dlss_wgpu::FeatureSupport>>);

fn sync_dlss_supported(mut settings: ResMut<DlssSettings>, probe: Res<DlssProbe>) {
  let supported = probe.0.lock().expect("DLSS 探测锁").super_resolution_supported;
  if settings.supported != supported {
    settings.supported = supported;
  }
}

pub struct DlssInitPlugin;

impl Plugin for DlssInitPlugin {
  fn build(&self, app: &mut App) {
    let probe =
      DlssProbe(std::sync::Arc::new(std::sync::Mutex::new(dlss_wgpu::FeatureSupport::default())));
    let mut settings = bevy::render::renderer::raw_vulkan_init::RawVulkanInitSettings::default();
    let id = ngx_project_id();
    let p_instance = probe.0.clone();
    unsafe {
      settings.add_create_instance_callback(move |args, _| {
        let before = args.extensions.len();
        if let Err(e) = dlss_wgpu::register_instance_extensions(
          id,
          args,
          &mut p_instance.lock().expect("DLSS 探测锁"),
        ) {
          bevy::log::error!("DLSS 实例扩展注册 ! {e}");
        }
        let added: Vec<String> =
          args.extensions[before..].iter().map(|c| c.to_string_lossy().into_owned()).collect();
        bevy::log::info!(target: "gate", "DLSS 实例扩展 → {:?}", added);
      });
      let p_device = probe.0.clone();
      settings.add_create_device_callback(move |args, adapter, _| {
        let before = args.extensions.len();
        if let Err(e) = dlss_wgpu::register_device_extensions(
          id,
          args,
          adapter,
          &mut p_device.lock().expect("DLSS 探测锁"),
        ) {
          bevy::log::error!("DLSS 设备扩展注册 ! {e}");
        }
        let added: Vec<String> =
          args.extensions[before..].iter().map(|c| c.to_string_lossy().into_owned()).collect();
        bevy::log::info!(target: "gate", "DLSS 设备扩展 → {:?}", added);
      });
    }
    app.insert_resource(settings);
    app.add_systems(bevy::app::Update, sync_dlss_supported);
    app.insert_resource(probe);
  }
}

struct SrCtx {
  dlss: dlss_wgpu::super_resolution::DlssSuperResolution,
  mode: DlssMode,
  out_size: UVec2,
  out_view: std::sync::Arc<TextureView>,
}

#[derive(Resource, Default)]
pub struct DlssGpu {
  sdk: Option<std::sync::Arc<std::sync::Mutex<dlss_wgpu::DlssSdk>>>,
  ctx: Option<SrCtx>,
  reset: bool,
  prev_vp: Option<Mat4>,
}

fn perf_quality(mode: DlssMode) -> dlss_wgpu::DlssPerfQualityMode {
  match mode {
    DlssMode::Off | DlssMode::Dlaa => dlss_wgpu::DlssPerfQualityMode::Dlaa,
    DlssMode::Quality => dlss_wgpu::DlssPerfQualityMode::Quality,
    DlssMode::Balanced => dlss_wgpu::DlssPerfQualityMode::Balanced,
    DlssMode::Performance => dlss_wgpu::DlssPerfQualityMode::Performance,
    DlssMode::Ultra => dlss_wgpu::DlssPerfQualityMode::UltraPerformance,
  }
}

unsafe extern "system" {
  fn NVSDK_NGX_VULKAN_Shutdown1(device: *mut core::ffi::c_void) -> i32;
}

fn init_dlss_sdk(mut commands: Commands, device: Res<RenderDevice>, probe: Option<Res<DlssProbe>>) {
  let Some(probe) = probe else { return };
  let supported = probe.0.lock().expect("DLSS 探测锁").super_resolution_supported;
  info!(target: "gate", "DLSS 探测：{}", if supported { "支持超分" } else { "不支持（保持原生 blit）" });
  let sdk = if !supported {
    None
  } else if std::env::var_os("GATE_DLSS_SKIP_SDK").is_some() {
    warn!(target: "gate", "DLSS SDK 创建被 GATE_DLSS_SKIP_SDK 跳过");
    None
  } else {
    match dlss_wgpu::DlssSdk::new(ngx_project_id(), device.wgpu_device().clone()) {
      Ok(sdk) => Some(sdk),
      Err(e) => {
        if matches!(e, dlss_wgpu::DlssError::FeatureNotSupported) {
          use ash::vk::Handle as _;
          let hal = unsafe { device.wgpu_device().as_hal::<wgpu::hal::api::Vulkan>() };
          if let Some(hal) = hal {
            let vk = hal.raw_device().handle();
            unsafe { NVSDK_NGX_VULKAN_Shutdown1(vk.as_raw() as *mut core::ffi::c_void) };
          }
        }
        error!(target: "gate", "DLSS SDK 创建 ! {e}");
        None
      }
    }
  };
  commands.insert_resource(DlssGpu { sdk, ..Default::default() });
}

fn sync_dlss_ctx(
  mut commands: Commands,
  settings: Res<DlssSettings>,
  frame: Res<DlssFrame>,
  mut gpu: ResMut<DlssGpu>,
  device: Res<RenderDevice>,
  queue: Res<RenderQueue>,
) {
  let want = frame.active && frame.out_size.x > 0 && frame.out_size.y > 0 && gpu.sdk.is_some();
  if !want {
    if gpu.ctx.is_some() {
      gpu.ctx = None;
      commands.insert_resource(DlssBlitSource(None));
    }
    return;
  }
  if gpu.ctx.as_ref().is_some_and(|c| c.mode == settings.mode && c.out_size == frame.out_size) {
    let view = gpu.ctx.as_ref().expect("ctx 已检查").out_view.clone();
    commands.insert_resource(DlssBlitSource(Some(view)));
    return;
  }
  let Some(sdk) = gpu.sdk.clone() else { return };
  let out = [frame.out_size.x, frame.out_size.y];
  match dlss_wgpu::super_resolution::DlssSuperResolution::new(
    out,
    perf_quality(settings.mode),
    dlss_wgpu::DlssFeatureFlags::AutoExposure
      | dlss_wgpu::DlssFeatureFlags::LowResolutionMotionVectors,
    sdk,
    device.wgpu_device(),
    &queue,
  ) {
    Ok(dlss) => {
      let tex = device.create_texture(&TextureDescriptor {
        label: Some("gate_dlss_out"),
        size: Extent3d { width: out[0], height: out[1], depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
      });
      let view = tex.create_view(&TextureViewDescriptor::default());
      let rr = dlss.render_resolution();
      info!(
        target: "gate",
        "DLSS 上下文 → {}×{}（档位 {:?}，建议内部渲染 {}×{}）",
        out[0], out[1], settings.mode, rr[0], rr[1]
      );
      let view = std::sync::Arc::new(view);
      gpu.ctx =
        Some(SrCtx { dlss, mode: settings.mode, out_size: frame.out_size, out_view: view.clone() });
      gpu.reset = true;
      gpu.prev_vp = None;
      commands.insert_resource(DlssBlitSource(Some(view)));
    }
    Err(e) => {
      error!(target: "gate", "DLSS 上下文创建 ! {e}（保持原生 blit）");
      commands.insert_resource(DlssBlitSource(None));
    }
  }
}

static EVAL_FAIL: std::sync::OnceLock<()> = std::sync::OnceLock::new();

fn dispatch_dlss_eval(
  mut ctx: RenderContext,
  mut gpu: ResMut<DlssGpu>,
  color: Option<Res<DdaColorView>>,
  aux: Option<Res<AuxTexCache>>,
  view: Option<Res<DdaViewUniform>>,
  scale: Res<RenderScale>,
  adapter: Res<RenderAdapter>,
) {
  let (Some(color), Some(aux), Some(view)) =
    (color.as_deref().map(|c| c.0.clone()), aux.as_deref(), view.as_deref().copied())
  else {
    return;
  };
  let Some(mut sr) = gpu.ctx.take() else { return };
  let Some(color) = color else {
    gpu.ctx = Some(sr);
    return;
  };
  let (Some(mvec), Some(depth)) = (aux.dlss_mvec_view.as_ref(), aux.dlss_depth_view.as_ref())
  else {
    gpu.ctx = Some(sr);
    return;
  };
  let mut reset = gpu.reset;
  if let Some(prev) = gpu.prev_vp
    && view
      .view_proj
      .to_cols_array()
      .iter()
      .zip(prev.to_cols_array())
      .any(|(a, b)| (a - b).abs() > 0.5)
  {
    reset = true;
  }
  gpu.prev_vp = Some(view.view_proj);
  let mv_scale =
    [-(scale.size.x as f32 / sr.out_size.x as f32), -(scale.size.y as f32 / sr.out_size.y as f32)];
  let params = dlss_wgpu::super_resolution::DlssSuperResolutionRenderParameters {
    color: &color,
    depth,
    motion_vectors: mvec,
    exposure: dlss_wgpu::super_resolution::DlssSuperResolutionExposure::Automatic,
    bias: None,
    dlss_output: &sr.out_view,
    reset,
    jitter_offset: [-view.dlss.x, -view.dlss.y],
    partial_texture_size: Some([scale.size.x, scale.size.y]),
    motion_vector_scale: Some(mv_scale),
  };
  gpu.reset = false;
  match sr.dlss.render(params, ctx.command_encoder(), &adapter) {
    Ok(cb) => ctx.add_command_buffer(cb),
    Err(e) => {
      if EVAL_FAIL.set(()).is_ok() {
        error!(target: "gate", "DLSS eval ! {e}（后续帧不再重试直到档位切换）");
      }
    }
  }
  gpu.ctx = Some(sr);
}

pub struct DlssRenderPlugin;

impl Plugin for DlssRenderPlugin {
  fn build(&self, app: &mut App) {
    let probe = app.world().get_resource::<DlssProbe>().map(|p| DlssProbe(p.0.clone()));
    let Some(render_app) = app.get_sub_app_mut(RenderApp) else { return };
    if let Some(probe) = probe {
      render_app.insert_resource(probe);
    }
    render_app
      .add_systems(RenderStartup, init_dlss_sdk)
      .add_systems(
        Render,
        sync_dlss_ctx
          .in_set(bevy::render::RenderSystems::PrepareResources)
          .before(crate::brickmap::dda::prepare_dda_bind_groups),
      )
      .add_systems(
        RenderGraph,
        dispatch_dlss_eval
          .in_set(bevy::render::renderer::RenderGraphSystems::Render)
          .after(crate::brickmap::dda::dispatch_dda)
          .before(bevy::core_pipeline::schedule::camera_driver),
      );
  }
}
