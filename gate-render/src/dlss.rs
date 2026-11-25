//! DLSS 集成（cargo feature `dlss`）。
//! NGX 链接库只有 MSVC 格式：本 feature 的构建需 `--target x86_64-pc-windows-msvc`。
use bevy::prelude::*;
use bevy::render::{
    Render, RenderApp, RenderStartup,
    renderer::{RenderAdapter, RenderContext, RenderDevice, RenderQueue},
    render_resource::{
        Extent3d, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
        TextureView, TextureViewDescriptor,
    },
};
use glam::Mat4;

use crate::brickmap::dda::{
    AuxTexCache, DdaColorView, DdaViewUniform, DlssBlitSource, DlssFrame, DlssMode, DlssSettings,
};

/// NGX 应用标识（固定常量）。
fn ngx_project_id() -> uuid::Uuid {
  uuid::Uuid::from_u128(0x6741_5445_444c_5353_0001_0000_0000_0001)
}

/// NGX 探测结果的共享端：实例/设备创建回调在渲染初始化时填充，
/// 渲染侧在 RenderStartup / PrepareResources 读取。
#[derive(Resource, Clone, Default)]
pub struct DlssProbe(pub std::sync::Arc<std::sync::Mutex<dlss_wgpu::FeatureSupport>>);

/// NGX 能力探测结果快照（`dlss_wgpu::FeatureSupport` 的值语义）。
#[derive(Resource, Clone, Default)]
pub struct DlssSupport {
  pub super_resolution: bool,
  pub ray_reconstruction: bool,
}

pub fn probe_from(fs: &dlss_wgpu::FeatureSupport) -> DlssSupport {
  DlssSupport {
    super_resolution: fs.super_resolution_supported,
    ray_reconstruction: fs.ray_reconstruction_supported,
  }
}

/// 在 RenderPlugin 完成渲染初始化之前注册 NGX 需要的实例/设备扩展。
/// 探测不通过时后续 SR 上下文不创建，渲染保持现有 blit 路径。
pub struct DlssInitPlugin;

impl Plugin for DlssInitPlugin {
  fn build(&self, app: &mut App) {
    let probe = DlssProbe(std::sync::Arc::new(std::sync::Mutex::new(
      dlss_wgpu::FeatureSupport::default(),
    )));
    let mut settings =
      bevy::render::renderer::raw_vulkan_init::RawVulkanInitSettings::default();
    let id = ngx_project_id();
    let p_instance = probe.0.clone();
    // SAFETY: 回调只向实例追加 NGX 声明所需的扩展；探测失败时 dlss_wgpu
    // 仅把能力位置 false，不删改实例本身的特性。
    unsafe {
      settings.add_create_instance_callback(move |args, _| {
        if let Err(e) = dlss_wgpu::register_instance_extensions(
          id,
          args,
          &mut p_instance.lock().expect("DLSS 探测锁"),
        ) {
          bevy::log::error!("DLSS 实例扩展注册 ! {e}");
        }
      });
      let p_device = probe.0.clone();
      settings.add_create_device_callback(move |args, adapter, _| {
        if let Err(e) = dlss_wgpu::register_device_extensions(
          id,
          args,
          adapter,
          &mut p_device.lock().expect("DLSS 探测锁"),
        ) {
          bevy::log::error!("DLSS 设备扩展注册 ! {e}");
        }
      });
    }
    app.insert_resource(settings);
    if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
      render_app.insert_resource(DlssProbe(probe.0.clone()));
    }
    app.insert_resource(probe);
  }
}

// ---------------------------------------------------------------------------

struct SrCtx {
  dlss: dlss_wgpu::super_resolution::DlssSuperResolution,
  mode: DlssMode,
  out_size: UVec2,
  out_view: std::sync::Arc<TextureView>,
}

/// 渲染端 DLSS 状态：NGX SDK 与活跃 SR 上下文。
#[derive(Resource, Default)]
pub struct DlssGpu {
  pub sdk: Option<std::sync::Arc<std::sync::Mutex<dlss_wgpu::DlssSdk>>>,
  pub ctx: Option<SrCtx>,
  pub reset: bool,
  pub prev_vp: Option<Mat4>,
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

/// 渲染初始化完成后按探测结果创建 NGX SDK（SR 上下文按档位/窗口延迟建）。
fn init_dlss_sdk(
  mut commands: Commands,
  device: Res<RenderDevice>,
  probe: Option<Res<DlssProbe>>,
) {
  let Some(probe) = probe else { return };
  let supported = probe.0.lock().expect("DLSS 探测锁").super_resolution_supported;
  info!(target: "gate", "DLSS 探测：{}", if supported { "支持超分" } else { "不支持（保持原生 blit）" });
  if !supported {
    return;
  }
  let sdk = match dlss_wgpu::DlssSdk::new(ngx_project_id(), device.wgpu_device().clone()) {
    Ok(sdk) => sdk,
    Err(e) => {
      error!(target: "gate", "DLSS SDK 创建 ! {e}");
      return;
    }
  };
  commands.insert_resource(DlssGpu { sdk: Some(sdk), ..Default::default() });
}

/// 每帧同步 SR 上下文与 blit 源：档位或输出尺寸变化时重建
/// （DlssSuperResolution 的 Drop 会等设备空闲，只发生在切换帧）。
fn sync_dlss_ctx(
  mut commands: Commands,
  settings: Res<DlssSettings>,
  frame: Res<DlssFrame>,
  mut gpu: ResMut<DlssGpu>,
  device: Res<RenderDevice>,
  queue: Res<RenderQueue>,
) {
  let want = frame.active
    && frame.out_size.x > 0
    && frame.out_size.y > 0
    && gpu.sdk.is_some();
  if !want {
    if gpu.ctx.is_some() {
      gpu.ctx = None;
      commands.insert_resource(DlssBlitSource(None));
    }
    return;
  }
  if gpu.ctx.is_some_and(|c| c.mode == settings.mode && c.out_size == frame.out_size) {
    let view = gpu.ctx.as_ref().expect("ctx 已检查").out_view.clone();
    commands.insert_resource(DlssBlitSource(Some(view)));
    return;
  }
  let Some(sdk) = gpu.sdk.clone() else { return };
  let out = [frame.out_size.x, frame.out_size.y];
  match dlss_wgpu::super_resolution::DlssSuperResolution::new(
    out,
    perf_quality(settings.mode),
    dlss_wgpu::DlssFeatureFlags::empty(),
    sdk,
    device.wgpu_device(),
    &queue.0,
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
      gpu.ctx = Some(SrCtx { dlss, mode: settings.mode, out_size: frame.out_size, out_view: view.clone() });
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

/// compose 之后、blit 之前执行 DLSS 超分；eval 命令缓冲经
/// `RenderContext::add_command_buffer` 追加（先 flush 主 encoder，顺序恒定）。
fn dispatch_dlss_eval(
  mut ctx: RenderContext,
  frame: Res<DlssFrame>,
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
  let (Some(mvec), Some(depth)) = (aux.dlss_mvec_view.as_ref(), aux.dlss_depth_view.as_ref()) else {
    gpu.ctx = Some(sr);
    return;
  };
  // 相机跳变 ⇒ 时域历史清零
  let mut reset = gpu.reset;
  if let Some(prev) = gpu.prev_vp {
    if view.view_proj.to_cols_array().iter().zip(prev.to_cols_array()).any(|(a, b)| (a - b).abs() > 0.5)
    {
      reset = true;
    }
  }
  gpu.prev_vp = Some(view.view_proj);
  let params = dlss_wgpu::super_resolution::DlssSuperResolutionRenderParameters {
    color: &color,
    depth,
    motion_vectors: mvec,
    exposure: dlss_wgpu::super_resolution::DlssSuperResolutionExposure::Automatic,
    bias: None,
    dlss_output: &sr.out_view,
    reset,
    jitter_offset: [view.dlss.x, view.dlss.y],
    partial_texture_size: Some([scale.size.x, scale.size.y]),
    motion_vector_scale: None,
  };
  gpu.reset = false;
  match sr.dlss.render(params, ctx.command_encoder(), &adapter.0) {
    Ok(cb) => ctx.add_command_buffer(cb),
    Err(e) => {
      if EVAL_FAIL.set(()).is_ok() {
        error!(target: "gate", "DLSS eval ! {e}（后续帧不再重试直到档位切换）");
      }
    }
  }
  gpu.ctx = Some(sr);
}

/// DLSS 渲染侧插件（feature `dlss`）。
pub struct DlssRenderPlugin;

impl Plugin for DlssRenderPlugin {
  fn build(&self, app: &mut App) {
    let Some(render_app) = app.get_sub_app_mut(RenderApp) else { return };
    render_app
      .add_systems(RenderStartup, init_dlss_sdk)
      .add_systems(
        Render,
        sync_dlss_ctx
          .in_set(bevy::render::RenderSystems::PrepareResources)
          .before(crate::brickmap::dda::prepare_dda_bind_groups),
      )
      .add_systems(
        bevy::render::renderer::RenderGraphSystems::Render,
        dispatch_dlss_eval.after(crate::brickmap::dda::dispatch_dda),
      );
  }
}
