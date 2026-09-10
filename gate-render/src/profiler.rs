//! GPU 帧剖析：wgpu-profiler 接入（cargo feature = "profile" 时启用）。
//!
//! profile 构建下：
//! - [`RenderStartup`] 建 `GpuProfiler`（tracy 模式：GPU zone 直送 Tracy，
//!   要求 gate-app main 最早期已 `tracy_client::Client::start()`；Tracy 未运行时
//!   创建失败 → profiler = None，所有 scope 空转，渲染不受影响）；
//! - resolve 系统挂 Render→Submit 之间：独立 encoder 调 `resolve_queries` 后 push
//!   PendingCommandBuffers（与 bevy 自带 RenderDiagnosticsPlugin 同模式，保证 resolve
//!   在所有带 query 的 command buffer 之后、同一次 submit 执行）；
//! - finish 系统挂 Finish 集（submit 已完成）：`end_frame` + `process_finished_frame`；
//! - 各 dispatch 系统经 [`gpu_compute_pass`] 打 compute pass scope；render pass
//!   （blit）经 [`profiler_mut`] 取 profiler 手动开 scoped_render_pass。
//!
//! 非 profile 构建：本模块为零依赖空壳——资源是 unit、helper 等价于
//! `begin_compute_pass`，渲染路径零开销。
//!
//! 设备特性：profile 构建需要 wgpu timestamp 特性，由 gate-app 经
//! [`timestamp_wgpu_features`] 写入 WgpuSettings。

use bevy::prelude::*;
use bevy::render::render_resource::{CommandEncoder, ComputePass, ComputePassDescriptor};
#[cfg(feature = "profile")]
use bevy::render::renderer::{
  PendingCommandBuffers, RenderAdapter, RenderDevice, RenderGraphSystems, RenderQueue,
};

/// render world 资源：包 wgpu-profiler（profile feature 关闭时为 unit 资源）。
#[derive(Resource, Default)]
pub(crate) struct GpuProfilerRes {
  #[cfg(feature = "profile")]
  profiler: Option<wgpu_profiler::GpuProfiler>,
}

/// 取 profiler 可变引用（render pass 无闭包 helper，调用方手动 scope）。
#[cfg(feature = "profile")]
pub(crate) fn profiler_mut(res: &mut GpuProfilerRes) -> Option<&mut wgpu_profiler::GpuProfiler> {
  res.profiler.as_mut()
}

/// profile 构建需要的 wgpu 设备特性（gate-app 写入 WgpuSettings.features）。
#[cfg(feature = "profile")]
pub fn timestamp_wgpu_features() -> bevy::render::render_resource::WgpuFeatures {
  wgpu_profiler::GpuProfiler::ALL_WGPU_TIMER_FEATURES
}

/// 在一个 compute pass 外打 GPU scope（profile 关闭 = 直接 begin_compute_pass）。
///
/// profile 下 = encoder scope（同 label）→ scoped_compute_pass（pass 自身时间戳），
/// body 拿到 [`ComputePass`] 录命令；pass/scope 随 block 结束自动关闭。
#[cfg(feature = "profile")]
pub(crate) fn gpu_compute_pass<T>(
  res: &mut GpuProfilerRes,
  encoder: &mut CommandEncoder,
  label: &str,
  mut body: impl FnMut(&mut ComputePass<'_>) -> T,
) -> T {
  if let Some(profiler) = res.profiler.as_mut() {
    let mut encoder_scope = profiler.scope(label, encoder);
    let mut pass = encoder_scope.scoped_compute_pass(label);
    body(&mut pass)
  } else {
    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
      label: Some(label),
      ..default()
    });
    body(&mut pass)
  }
}

/// 非 profile 构建：直接 begin_compute_pass（与 profile 分支同 body 形态）。
#[cfg(not(feature = "profile"))]
pub(crate) fn gpu_compute_pass<T>(
  _res: &mut GpuProfilerRes,
  encoder: &mut CommandEncoder,
  label: &str,
  mut body: impl FnMut(&mut ComputePass<'_>) -> T,
) -> T {
  let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
    label: Some(label),
    ..default()
  });
  body(&mut pass)
}

/// 渲染剖析插件（profile feature 关闭时仅注册空资源）。
pub(crate) struct GateProfilerPlugin;

impl Plugin for GateProfilerPlugin {
  fn build(&self, app: &mut App) {
    let Some(render_app) = app.get_sub_app_mut(bevy::render::RenderApp) else {
      return;
    };
    render_app.init_resource::<GpuProfilerRes>();
    #[cfg(feature = "profile")]
    {
      render_app.add_systems(bevy::render::RenderStartup, init_gpu_profiler);
      render_app.add_systems(
        bevy::render::renderer::RenderGraph,
        (
          resolve_profiler_queries
            .after(RenderGraphSystems::Render)
            .before(RenderGraphSystems::Submit),
          finish_profiler_frame.in_set(RenderGraphSystems::Finish),
        ),
      );
    }
  }
}

/// RenderStartup：建 GpuProfiler（tracy 模式）。
#[cfg(feature = "profile")]
fn init_gpu_profiler(
  device: Res<RenderDevice>,
  queue: Res<RenderQueue>,
  adapter: Res<RenderAdapter>,
  mut res: ResMut<GpuProfilerRes>,
) {
  // new_with_tracy_client 内部要求 tracy client 已 start（Client::running()）；
  // gate-app 在 main 最早期启动 client。失败（Tracy 未运行）则空转。
  let backend = adapter.get_info().backend;
  match wgpu_profiler::GpuProfiler::new_with_tracy_client(
    wgpu_profiler::GpuProfilerSettings::default(),
    backend,
    device.wgpu_device(),
    &queue,
  ) {
    Ok(profiler) => {
      bevy::log::info!("wgpu-profiler 就绪：GPU zone 直送 Tracy（backend={backend:?}）");
      res.profiler = Some(profiler);
    }
    Err(e) => {
      bevy::log::warn!("wgpu-profiler 初始化失败（{e:?}）；GPU scope 空转");
    }
  }
}

/// Render→Submit 之间：resolve 本帧全部 timestamp query。
///
/// 独立 encoder push PendingCommandBuffers → 与 dispatch 系统的 command buffers 同一次
/// queue.submit，按 push 序 resolve 在所有 query 写入之后执行。
#[cfg(feature = "profile")]
fn resolve_profiler_queries(
  mut res: ResMut<GpuProfilerRes>,
  device: Res<RenderDevice>,
  mut pending: ResMut<PendingCommandBuffers>,
) {
  let Some(profiler) = res.profiler.as_mut() else {
    return;
  };
  let mut encoder =
    device.create_command_encoder(&bevy::render::render_resource::CommandEncoderDescriptor {
      label: Some("wgpu_profiler_resolve"),
    });
  profiler.resolve_queries(&mut encoder);
  pending.push_encoder(encoder);
}

/// Finish 集（submit 已完成）：结束本帧并处理已就绪帧（tracy 模式自动上报）。
#[cfg(feature = "profile")]
fn finish_profiler_frame(mut res: ResMut<GpuProfilerRes>, queue: Res<RenderQueue>) {
  let Some(profiler) = res.profiler.as_mut() else {
    return;
  };
  if let Err(e) = profiler.end_frame() {
    bevy::log::warn_once!("wgpu-profiler end_frame 失败：{e:?}");
  }
  // tracy 模式下结果直接上报 Tracy；返回值（Option<Vec<GpuTimerQueryResult>>）无需消费
  profiler.process_finished_frame(queue.get_timestamp_period());
}
