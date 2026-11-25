//! DLSS 集成（cargo feature `dlss`）。
//! NGX 链接库只有 MSVC 格式：本 feature 的构建需 `--target x86_64-pc-windows-msvc`。
use bevy::prelude::*;
use bevy::render::renderer::raw_vulkan_init::RawVulkanInitSettings;

/// NGX 应用标识（固定常量）。
fn ngx_project_id() -> uuid::Uuid {
  uuid::Uuid::from_u128(0x6741_5445_444c_5353_0001_0000_0000_0001)
}

/// NGX 探测结果的共享端：实例/设备创建回调在渲染初始化线程里填充，
/// 渲染侧（SR 上下文创建）在 RenderStartup 读取。
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
    let mut settings = RawVulkanInitSettings::default();
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
    app.insert_resource(probe);
  }
}
