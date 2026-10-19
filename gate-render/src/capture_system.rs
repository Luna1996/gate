use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use bevy::{
  prelude::*,
  render::{
    RenderApp,
    render_resource::*,
    renderer::{RenderContext, RenderDevice},
    view::ViewTarget,
  },
};

use crate::brickmap::dda::{DdaBlitBindGroup, DdaPipelines};

#[derive(Resource)]
pub struct DdaCapture {
  path: String,
  frame_target: u32,
  frame: u32,
  extent: Option<Extent3d>,
  tex: Option<Texture>,
  tex_view: Option<TextureView>,
  buf: Option<Buffer>,
  row_bytes: u32,
  copied: bool,
  submit_started: bool,
  ready: Arc<AtomicBool>,
  done: bool,
}

impl DdaCapture {
  pub fn from_env() -> Option<Self> {
    let path = std::env::var("GATE_SHOT").ok().filter(|s| !s.is_empty())?;
    let frame_target =
      std::env::var("GATE_SHOT_FRAME").ok().and_then(|s| s.parse().ok()).unwrap_or(60u32);
    Some(Self {
      path,
      frame_target,
      frame: 0,
      extent: None,
      tex: None,
      tex_view: None,
      buf: None,
      row_bytes: 0,
      copied: false,
      submit_started: false,
      ready: Arc::new(AtomicBool::new(false)),
      done: false,
    })
  }
}

pub struct CapturePlugin;

impl Plugin for CapturePlugin {
  fn build(&self, _app: &mut App) {}

  fn finish(&self, app: &mut App) {
    let Some(cap) = DdaCapture::from_env() else { return };
    let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
      bevy::log::warn!("截图未启用：找不到 RenderApp");
      return;
    };
    bevy::log::info!("截图已启用 → {} (第 {} 帧)", cap.path, cap.frame_target);
    render_app.insert_resource(cap);
  }
}

pub fn capture_dda_view(
  mut ctx: RenderContext,
  mut cap: Option<ResMut<DdaCapture>>,
  views: Query<&ViewTarget>,
  blit_bg: Option<Res<DdaBlitBindGroup>>,
  pipeline_cache: Res<PipelineCache>,
  pipelines: Res<DdaPipelines>,
  render_device: Res<RenderDevice>,
) {
  let Some(mut cap) = cap.as_mut() else { return };
  if cap.done {
    return;
  }
  cap.frame += 1;
  if cap.frame < cap.frame_target {
    return;
  }

  if cap.copied {
    if !cap.submit_started {
      cap.submit_started = true;
      let flag = Arc::clone(&cap.ready);
      let Some(buf) = cap.buf.as_ref() else {
        cap.done = true;
        return;
      };
      let bytes = buf.size();
      let res = buf.slice(..).map_async(wgpu::MapMode::Read, move |r| match r {
        Ok(_) => flag.store(true, Ordering::SeqCst),
        Err(e) => {
          bevy::log::error!("截图回读失败（{bytes} 字节）: {e:?}");
          flag.store(true, Ordering::SeqCst);
        }
      });
      drop(buf);
      let _ = res;
      let _ = render_device.wgpu_device().poll(wgpu::PollType::wait_indefinitely());
    }
    let Some(buf) = cap.buf.as_ref() else {
      cap.done = true;
      return;
    };
    if !cap.ready.load(Ordering::SeqCst) {
      if cap.frame > cap.frame_target + 600 {
        bevy::log::warn!("截图：回读超时 → 放弃");
        cap.done = true;
      }
      return;
    }
    let Some(ext) = cap.extent else {
      cap.done = true;
      return;
    };
    let w = ext.width;
    let h = ext.height;
    let row = cap.row_bytes as usize;
    let mapped = buf.slice(..).get_mapped_range();
    let Ok(view) = mapped else {
      bevy::log::error!("截图：get_mapped_range 失败 → 放弃");
      cap.done = true;
      return;
    };
    let mut rgba = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h as usize {
      let src = y * row;
      rgba.extend_from_slice(&view[src..src + w as usize * 4]);
    }
    drop(view);
    buf.unmap();
    let path = std::path::PathBuf::from(&cap.path);
    match crate::capture::write_png(&path, w, h, &rgba) {
      Ok(()) => bevy::log::info!("截图已写出 → {} ({}x{})", path.display(), w, h),
      Err(e) => bevy::log::error!("截图写盘失败 {}: {e}", path.display()),
    }
    cap.tex = None;
    cap.tex_view = None;
    cap.buf = None;
    cap.done = true;
    return;
  }

  let (Some(bg), Ok(view)) = (blit_bg.as_ref(), views.single()) else {
    if cap.frame == cap.frame_target {
      bevy::log::warn!("截图：bind group 或 view 缺失");
    }
    return;
  };
  let Some(pipe) = pipeline_cache.get_render_pipeline(pipelines.blit_pipeline) else {
    if cap.frame == cap.frame_target {
      bevy::log::warn!("截图：blit pipeline 未就绪");
    }
    return;
  };

  if cap.extent.is_none() {
    let ext = view.main_texture().size();
    if ext.width == 0 || ext.height == 0 {
      return;
    }
    cap.row_bytes = (ext.width * 4).div_ceil(256) * 256;
    let bytes = u64::from(cap.row_bytes) * u64::from(ext.height);
    let tex = render_device.create_texture(&TextureDescriptor {
      label: Some("gate_dda_capture"),
      size: ext,
      mip_level_count: 1,
      sample_count: 1,
      dimension: TextureDimension::D2,
      format: TextureFormat::Rgba8UnormSrgb,
      usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
      view_formats: &[],
    });
    cap.tex_view = Some(tex.create_view(&TextureViewDescriptor::default()));
    cap.tex = Some(tex);
    cap.buf = Some(render_device.create_buffer(&BufferDescriptor {
      label: Some("gate_dda_capture_readback"),
      size: bytes,
      usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
      mapped_at_creation: false,
    }));
    cap.extent = Some(ext);
  }
  let (Some(tex), Some(tex_view), Some(buffer), Some(ext)) =
    (cap.tex.as_ref(), cap.tex_view.as_ref(), cap.buf.as_ref(), cap.extent)
  else {
    cap.done = true;
    return;
  };

  {
    let mut pass = ctx
      .command_encoder()
      .begin_render_pass(&RenderPassDescriptor {
        label: Some("gate_dda_capture"),
        color_attachments: &[Some(RenderPassColorAttachment {
          view: tex_view,
          depth_slice: None,
          resolve_target: None,
          ops: Operations { load: LoadOp::Clear(wgpu::Color::BLACK), store: StoreOp::Store },
        })],
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
  ctx.command_encoder().copy_texture_to_buffer(
    TexelCopyTextureInfo {
      texture: tex,
      mip_level: 0,
      origin: Origin3d::ZERO,
      aspect: TextureAspect::All,
    },
    TexelCopyBufferInfo {
      buffer,
      layout: TexelCopyBufferLayout {
        offset: 0,
        bytes_per_row: Some(cap.row_bytes),
        rows_per_image: Some(ext.height),
      },
    },
    ext,
  );
  cap.copied = true;
}