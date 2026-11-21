use bevy::render::render_resource::TextureView;

pub struct RrInputs<'a> {
  pub diffuse_albedo: &'a TextureView,
  pub specular_albedo: &'a TextureView,
  pub normals: &'a TextureView,
  pub color: &'a TextureView,
  pub depth: &'a TextureView,
  pub motion_vectors: &'a TextureView,
  pub specular_guide: &'a TextureView,
  pub output: &'a TextureView,
  pub reset: bool,
  pub jitter: [f32; 2],
}

#[cfg(feature = "dlss")]
mod imp {
  use std::sync::{Arc, Mutex};

  use bevy::prelude::*;
  use bevy::render::renderer::raw_vulkan_init::RawVulkanInitSettings;
  use dlss_wgpu::ray_reconstruction::{
    DlssRayReconstruction, DlssRayReconstructionDepthMode, DlssRayReconstructionRoughnessMode,
  };
  use dlss_wgpu::{
    DlssFeatureFlags, DlssPerfQualityMode, DlssSdk, FeatureSupport, register_device_extensions,
    register_instance_extensions,
  };
  use uuid::Uuid;

  const PROJECT_ID: &str = "6f1d2a43-6f1a-4c8e-9f2b-8a0e7c5d3b11";

  #[derive(Resource, Clone)]
  pub struct DlssSupport(Arc<Mutex<FeatureSupport>>);

  #[derive(Resource, Clone)]
  pub struct DlssSdkRes(pub Arc<Mutex<DlssSdk>>);

  #[derive(Resource, Clone)]
  pub struct RrHook(pub Arc<Mutex<DlssRayReconstruction>>);

  impl RrHook {
    pub fn render(
      &self,
      inputs: crate::dlss::RrInputs<'_>,
      encoder: &mut wgpu::CommandEncoder,
      adapter: &wgpu::Adapter,
    ) -> Option<wgpu::CommandBuffer> {
      use dlss_wgpu::ray_reconstruction::{
        DlssRayReconstructionRenderParameters, DlssRayReconstructionSpecularGuide,
      };
      let params = DlssRayReconstructionRenderParameters {
        diffuse_albedo: inputs.diffuse_albedo,
        specular_albedo: inputs.specular_albedo,
        normals: inputs.normals,
        roughness: None,
        color: inputs.color,
        depth: inputs.depth,
        motion_vectors: inputs.motion_vectors,
        specular_guide: DlssRayReconstructionSpecularGuide::SpecularMotionVectors(
          inputs.specular_guide,
        ),
        transparency_overlay: None,
        color_before_transparency: None,
        screen_space_subsurface_scattering_guide: None,
        depth_of_field_guide: None,
        responsivity_mask: None,
        alpha: None,
        dlss_output_alpha: None,
        dlss_output: inputs.output,
        reset: inputs.reset,
        jitter_offset: inputs.jitter,
        partial_texture_size: None,
        motion_vector_scale: None,
      };
      let mut rr = self.0.lock().expect("DLSS RR lock");
      match rr.render(params, encoder, adapter) {
        Ok(cmd) => Some(cmd),
        Err(e) => {
          error!("DLSS RR render failed: {e}");
          None
        }
      }
    }
  }

  impl DlssSupport {
    pub fn super_resolution(&self) -> bool {
      self.0.lock().map(|s| s.super_resolution_supported).unwrap_or(false)
    }

    pub fn ray_reconstruction(&self) -> bool {
      self.0.lock().map(|s| s.ray_reconstruction_supported).unwrap_or(false)
    }
  }

  pub fn install(app: &mut App) {
    let project_id = Uuid::parse_str(PROJECT_ID).expect("DLSS project id");
    let support = Arc::new(Mutex::new(FeatureSupport::default()));
    let instance_support = Arc::clone(&support);
    let device_support = Arc::clone(&support);
    let mut settings = RawVulkanInitSettings::default();

    unsafe {
      settings.add_create_instance_callback(move |args, _| {
        let mut support = instance_support.lock().expect("DLSS support lock");
        if let Err(e) = register_instance_extensions(project_id, args, &mut support) {
          error!("DLSS instance extension registration failed: {e}");
        }
      });
      settings.add_create_device_callback(move |args, adapter, _| {
        let mut support = device_support.lock().expect("DLSS support lock");
        if let Err(e) = register_device_extensions(project_id, args, adapter, &mut support) {
          error!("DLSS device extension registration failed: {e}");
        }
      });
    }

    app.insert_resource(settings);
    app.insert_resource(DlssSupport(support));
    app.add_systems(Startup, log_support);
  }

  fn log_support(support: Option<Res<DlssSupport>>) {
    if let Some(support) = support {
      info!(
        "DLSS SR supported: {} / RR supported: {}",
        if support.super_resolution() { "yes" } else { "no" },
        if support.ray_reconstruction() { "yes" } else { "no" },
      );
    }
  }

  pub fn init_render(app: &mut App) {
    let Some(render_app) = app.get_sub_app_mut(bevy::render::RenderApp) else { return };
    render_app.add_systems(bevy::render::Render, init_sdk);
  }

  fn init_sdk(
    mut commands: Commands,
    device: Res<bevy::render::renderer::RenderDevice>,
    queue: Res<bevy::render::renderer::RenderQueue>,
    existing: Option<Res<DlssSdkRes>>,
  ) {
    if existing.is_some() {
      return;
    }
    let project_id = Uuid::parse_str(PROJECT_ID).expect("DLSS project id");
    match DlssSdk::new(project_id, device.wgpu_device().clone()) {
      Ok(sdk) => {
        info!("DLSS SDK initialized");
        match DlssRayReconstruction::new(
          [crate::consts::VIEW_SIZE.x, crate::consts::VIEW_SIZE.y],
          DlssPerfQualityMode::Dlaa,
          DlssFeatureFlags::empty(),
          DlssRayReconstructionRoughnessMode::Packed,
          DlssRayReconstructionDepthMode::Linear,
          sdk.clone(),
          device.wgpu_device(),
          &queue,
        ) {
          Ok(rr) => {
            info!("DLSS RR initialized");
            commands.insert_resource(RrHook(Arc::new(Mutex::new(rr))));
          }
          Err(e) => warn!("DLSS RR initialization failed: {e}"),
        }
        commands.insert_resource(DlssSdkRes(sdk));
      }
      Err(e) => error!("DLSS SDK initialization failed: {e}"),
    }
  }
}

#[cfg(not(feature = "dlss"))]
mod imp {
  use bevy::prelude::App;

  #[derive(bevy::prelude::Resource, Clone)]
  pub struct RrHook;

  impl RrHook {
    pub fn render(
      &self,
      _inputs: crate::dlss::RrInputs<'_>,
      _encoder: &mut wgpu::CommandEncoder,
      _adapter: &wgpu::Adapter,
    ) -> Option<wgpu::CommandBuffer> {
      None
    }
  }

  pub fn install(_app: &mut App) {}
  pub fn init_render(_app: &mut App) {}
}

#[cfg(not(feature = "dlss"))]
pub use imp::RrHook;
pub use imp::init_render;
pub use imp::install;
#[cfg(feature = "dlss")]
pub use imp::{DlssSdkRes, DlssSupport, RrHook};
