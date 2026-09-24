//! GPU 帧剖析：wgpu-profiler 接入（cargo feature = "profile" 时启用）。
//! tracy 模式要求 gate-app main 最早期已 `tracy_client::Client::start()`；
//! device 需 wgpu timestamp 特性；非 profile 构建为零依赖空壳。
//! 另含**呈现帧计数**（[`FramePace`]）：跨 feature 恒定存在，理由见它的说明。

use bevy::prelude::*;
use bevy::render::render_resource::{
  BufferDescriptor, BufferUsages, CommandEncoder, ComputePass, ComputePassDescriptor, MapMode,
  PollType,
};
use bevy::render::renderer::{RenderDevice, RenderQueue};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(feature = "profile")]
use bevy::render::renderer::{PendingCommandBuffers, RenderAdapter};

/// **呈现侧的帧计数**（诊断用；跨 feature 恒定存在）。
///
/// 为什么需要它：`Time::delta` 量的是**主循环**的节奏，而本工程是 pipelined rendering ——
/// 主循环不必等渲染线程（实测能跑出 1ms 的主循环帧），于是"主循环 FPS"会显示成
/// 66 ↔ 792 的锯齿，而真正被提交呈现的帧是稳定的一串（profiler 的逐 pass 计数可见：
/// 2.0s 内 122 帧 = 61fps）。
/// 本计数在**渲染世界**每帧自增（`RenderGraphSystems::Finish`，与 profiler 收尾同集），
/// 主世界读差值算速率 ⇒ 覆盖层显示的就是真实呈现节奏。
/// 两个世界共享同一个 `Arc` ⇒ 不需要 `ExtractResource`（见 `GateProfilerPlugin::build`）。
#[derive(Resource, Clone, Default)]
pub struct FramePace {
  /// 已提交/呈现的帧数（渲染世界每帧 +1）
  pub presented: Arc<AtomicU64>,
}

/// 渲染世界：每帧自增（`GateProfilerPlugin` 注册在 `RenderGraphSystems::Finish`）。
fn tick_frame_pace(pace: Res<FramePace>) {
  pace.presented.fetch_add(1, Ordering::Relaxed);
}

/// render world 资源：包 wgpu-profiler（profile feature 关闭时为 unit 资源）。
#[derive(Resource, Default)]
pub(crate) struct GpuProfilerRes {
  #[cfg(feature = "profile")]
  profiler: Option<wgpu_profiler::GpuProfiler>,
  #[cfg(feature = "profile")]
  report: PassReport,
}

/// 逐 pass GPU 耗时聚合器：按 label 累计，每 [`crate::consts::REPORT_PERIOD_SECS`] 秒落一行平均耗时日志。
#[cfg(feature = "profile")]
#[derive(Default)]
struct PassReport {
  /// label → 累计秒数
  acc: std::collections::BTreeMap<String, f64>,
  frames: u32,
  last: Option<std::time::Instant>,
}

#[cfg(feature = "profile")]
impl PassReport {
  fn push(&mut self, results: &[wgpu_profiler::GpuTimerQueryResult]) {
    for r in results {
      if let Some(t) = &r.time {
        *self.acc.entry(r.label.clone()).or_insert(0.0) += t.end - t.start;
      }
    }
    self.frames += 1;
    let now = std::time::Instant::now();
    let start = *self.last.get_or_insert(now);
    let dt = now.duration_since(start).as_secs_f32();
    if dt < crate::consts::REPORT_PERIOD_SECS || self.frames == 0 {
      return;
    }
    let n = self.frames as f64;
    let ms = |s: f64| s / n * 1000.0;
    let mut line = format!("GPU[{dt:.1}s x{}] ", self.frames);
    let mut total = 0.0;
    for (label, s) in &self.acc {
      total += ms(*s);
      line.push_str(&format!("{label}={:.2}ms ", ms(*s)));
    }
    bevy::log::info!("GPU 逐 pass 均值（共 {total:.2}ms/frame）：{line}");
    self.acc.clear();
    self.frames = 0;
    self.last = Some(now);
  }
}

/// 取 profiler 可变引用。
#[cfg(feature = "profile")]
pub(crate) fn profiler_mut(res: &mut GpuProfilerRes) -> Option<&mut wgpu_profiler::GpuProfiler> {
  res.profiler.as_mut()
}

/// profile 构建需要的 wgpu 设备特性（gate-app 写入 WgpuSettings.features）。
#[cfg(feature = "profile")]
pub fn timestamp_wgpu_features() -> bevy::render::render_resource::WgpuFeatures {
  wgpu_profiler::GpuProfiler::ALL_WGPU_TIMER_FEATURES
}

/// 在一个 compute pass 外打 GPU scope（profile 下含 pass 时间戳；pass/scope 随 block 结束自动关闭）。
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
    let mut pass =
      encoder.begin_compute_pass(&ComputePassDescriptor { label: Some(label), ..default() });
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
  let mut pass =
    encoder.begin_compute_pass(&ComputePassDescriptor { label: Some(label), ..default() });
  body(&mut pass)
}

/// 渲染剖析插件（profile feature 关闭时仅注册空资源）。
pub(crate) struct GateProfilerPlugin;

impl Plugin for GateProfilerPlugin {
  fn build(&self, app: &mut App) {
    // 呈现帧计数：**两个世界共享同一个 `Arc`**（主世界读、渲染世界每帧自增）。
    // 必须在取 render_app 之前插进主世界（渲染世界那份下面一起给）。
    let pace = FramePace::default();
    app.insert_resource(pace.clone());
    let Some(render_app) = app.get_sub_app_mut(bevy::render::RenderApp) else {
      return;
    };
    render_app.init_resource::<GpuProfilerRes>();
    render_app.insert_resource(pace);
    render_app.add_systems(
      bevy::render::renderer::RenderGraph,
      tick_frame_pace.in_set(bevy::render::renderer::RenderGraphSystems::Finish),
    );
    // 叶级 LOD 诊断读回（M0）：只在 `consts::LOD_DIAG` 打开时注册 ⇒ 关闭时零成本、零日志
    // （须与 `trace.wesl::LOD_DIAG` 同时打开才有数，见该常量的说明）。
    if crate::brickmap::consts::LOD_DIAG {
      render_app.add_systems(
        bevy::render::renderer::RenderGraph,
        report_lod_diag.in_set(bevy::render::renderer::RenderGraphSystems::Finish),
      );
    }
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
  // 第二参数是 bevy 0.20 新增的 encoder label（只在 `trace` feature 下消费）
  pending.push_encoder(encoder, "wgpu_profiler_resolve");
}

/// Finish 集（submit 已完成）：结束本帧并处理已就绪帧（tracy 模式自动上报）。
#[cfg(feature = "profile")]
fn finish_profiler_frame(mut res: ResMut<GpuProfilerRes>, queue: Res<RenderQueue>) {
  let period = queue.get_timestamp_period();
  let results = res.profiler.as_mut().and_then(|profiler| {
    if let Err(e) = profiler.end_frame() {
      bevy::log::warn_once!("wgpu-profiler end_frame 失败：{e:?}");
    }
    profiler.process_finished_frame(period)
  });

  if let Some(results) = results {
    res.report.push(&results);
  }
}

/// 叶级 LOD 诊断计数器（M0）的读回：每 [`crate::consts::REPORT_PERIOD_SECS`] 秒把 `gpu.lod_diag`
/// 拷进 staging 并**同步**读回，落一行 `DIAG[...]`（见 `docs/editable-gigavoxel.md` §9）。
///
/// WHY 同步阻塞（自建 encoder + `map_buffer` + `poll(wait_indefinitely)`）：与
/// `brickmap::upload::dump_voxel_buffers` 同一取舍 —— 诊断是离散动作，跨帧状态机
/// （arm → 下帧 map → 再下帧读）比"submit 后等结果"复杂得多，而这里每 `REPORT_PERIOD_SECS` 才付一次。
/// 调度在 `RenderGraphSystems::Finish`（本帧已提交）⇒ 读到的是本帧的值（累积计数，差一帧无影响）。
///
/// 计数器是**累积**的（shader 只加不清）⇒ 用 `wrapping_sub` 求窗口差值：既不需要清零，也没有
/// "清零写 vs GPU 写"的竞态；`u32` 回绕也由 `wrapping_sub` 自然处理（窗口内增量远小于 2³²）。
fn report_lod_diag(
  device: Res<RenderDevice>,
  queue: Res<RenderQueue>,
  gpu: Option<Res<crate::brickmap::upload::GpuBrickMap>>,
  mut period: Local<Option<std::time::Instant>>,
  mut prev: Local<Option<[u32; crate::brickmap::consts::LOD_DIAG_WORDS]>>,
) {
  const WORDS: usize = crate::brickmap::consts::LOD_DIAG_WORDS;
  let Some(gpu) = gpu else { return };
  let now = std::time::Instant::now();
  let due = period
    .is_none_or(|t| now.duration_since(t).as_secs_f32() >= crate::consts::REPORT_PERIOD_SECS);
  if !due {
    return;
  }
  *period = Some(now);

  let bytes = (WORDS * 4) as u64;
  let staging = device.create_buffer(&BufferDescriptor {
    label: Some("gate_lod_diag_staging"),
    size: bytes,
    usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
    mapped_at_creation: false,
  });
  let mut enc = device.create_command_encoder(&bevy::render::render_resource::CommandEncoderDescriptor {
    label: Some("gate_lod_diag_readback"),
  });
  enc.copy_buffer_to_buffer(&gpu.lod_diag, 0, &staging, 0, bytes);
  queue.submit([enc.finish()]);

  let slice = staging.slice(..);
  let (tx, rx) = std::sync::mpsc::channel();
  device.map_buffer(&slice, MapMode::Read, move |r| {
    let _ = tx.send(r);
  });
  if device.poll(PollType::wait_indefinitely()).is_err() {
    warn!("诊断读回：等待失败 → 本窗口跳过");
    staging.unmap();
    return;
  }
  if !matches!(rx.recv_timeout(std::time::Duration::from_secs(5)), Ok(Ok(()))) {
    warn!("诊断读回：映射超时 → 本窗口跳过");
    staging.unmap();
    return;
  }
  let mut cur = [0u32; WORDS];
  if let Ok(view) = slice.get_mapped_range() {
    for (i, w) in cur.iter_mut().enumerate() {
      let o = i * 4;
      *w = u32::from_le_bytes([view[o], view[o + 1], view[o + 2], view[o + 3]]);
    }
  }
  staging.unmap();

  let p = *prev.get_or_insert(cur);
  let d: [u32; WORDS] = std::array::from_fn(|i| cur[i].wrapping_sub(p[i]));
  *prev = Some(cur);
  // 槽位含义见 `trace.wesl::DIAG_*`：0 = 采样叶入口，1 = 其中叶级 LOD 拦下的，2 = 非法早停。
  let entries = d[0].max(1);
  info!(
    "DIAG[leaf_in {} lod_stop {} {:.1}% illegal {}]",
    d[0],
    d[1],
    d[1] as f32 * 100.0 / entries as f32,
    d[2]
  );
}
