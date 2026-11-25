use bevy::prelude::*;
use bevy::render::renderer::RenderDevice;

use crate::shader::DdaShaderSource;

#[derive(Resource, Clone, Default)]
pub struct GatePipeCache {
  pub pc: Option<std::sync::Arc<wgpu::PipelineCache>>,
  pub module: Option<std::sync::Arc<wgpu::ShaderModule>>,
}

pub fn cache_path() -> std::path::PathBuf {
  crate::paths::data_dir().join("pipeline-cache.bin")
}

pub(crate) fn init_gate_pipe_cache(
  mut commands: Commands,
  render_device: Res<RenderDevice>,
  source: Option<Res<DdaShaderSource>>,
) {
  let Some(source) = source else { return };
  if !render_device.features().contains(wgpu::Features::PIPELINE_CACHE) {
    info_once!(target: "gate", "管线磁盘缓存：设备不支持 PIPELINE_CACHE → bevy 异步路径");
    commands.insert_resource(GatePipeCache::default());
    return;
  }
  let blob = std::fs::read(cache_path()).ok();
  if let Some(b) = blob.as_ref() {
    info!(target: "gate", "管线磁盘缓存：载入 {} KB（{}）", b.len() / 1024, cache_path().display());
  }
  let pc = unsafe {
    render_device.wgpu_device().create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
      label: Some("gate-pipe-cache"),
      data: blob.as_deref(),
      fallback: true,
    })
  };
  let t0 = std::time::Instant::now();
  let module = render_device.wgpu_device().create_shader_module(wgpu::ShaderModuleDescriptor {
    label: Some("gate-dda-wgsl"),
    source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(&source.0[..])),
  });
  info!(target: "gate", "WGSL 模块 → wgpu：{:.1}s", t0.elapsed().as_secs_f32());
  commands.insert_resource(GatePipeCache {
    pc: Some(std::sync::Arc::new(pc)),
    module: Some(std::sync::Arc::new(module)),
  });
}

static SAVED: std::sync::OnceLock<()> = std::sync::OnceLock::new();

pub(crate) fn save_once(cache: &GatePipeCache) {
  if SAVED.set(()).is_err() {
    return;
  }
  let Some(pc) = cache.pc.as_ref() else { return };
  let Some(data) = pc.get_data() else {
    error!(target: "gate", "管线磁盘缓存：get_data → None，跳过回写");
    return;
  };
  info!(target: "gate", "管线磁盘缓存：回写 {} KB → {}", data.len() / 1024, cache_path().display());
  let path = cache_path();
  std::thread::Builder::new()
    .name("gate-pipe-cache-save".to_string())
    .spawn(move || {
      let tmp = path.with_extension("bin.tmp");
      if let Err(e) = std::fs::write(&tmp, &data).and_then(|_| std::fs::rename(&tmp, &path)) {
        error!(target: "gate", "管线磁盘缓存回写 ! {e}");
      }
    })
    .expect("管线缓存回写线程创建失败");
}
