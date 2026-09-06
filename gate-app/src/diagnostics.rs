//! GATE_BENCH=1 诊断：逐帧 GPU pass 时间日志（logs/gpu_frame.log）。

use bevy::prelude::*;

use crate::debug_overlay::FPS_REFRESH_SECS;

/// 逐帧 GPU pass 时间日志（仅 GATE_BENCH=1 装配）。
/// render world 的 time_span 每帧由 RenderDiagnosticsPlugin 同步到主世界
/// DiagnosticsStore；读 latest value 攒批 0.25s 刷 logs/gpu_frame.log。
/// 列：elapsed_secs,wall_ms,frame_gpu_ms,trace_gpu_ms,ddgi_gpu_ms,direct_gpu_ms。
/// frame_gpu = ddgi+trace+direct+blit 四 pass span 之和（GPU 帧真实工作量的近似：
/// pass 间 gap 与 span 外开销未计）。不能做跨系统嵌套 span —— bevy_render
/// open_spans 按 thread_id 分栈，并行 executor 下 begin/end 落不同线程会 panic。
pub(crate) fn gpu_frame_log(
  time: Res<Time>,
  store: Option<Res<bevy::diagnostic::DiagnosticsStore>>,
  mut log_file: Local<Option<std::fs::File>>,
  mut buf: Local<String>,
  mut acc: Local<f32>,
) {
  if log_file.is_none() {
    *log_file = std::fs::File::create(concat!(
      env!("CARGO_MANIFEST_DIR"),
      "/logs/gpu_frame.log"
    ))
    .ok();
  }
  let dt = time.delta_secs();
  // DiagnosticPath::new 要求 'static（Cow<'static, str>）→ 路径用字面值常量
  let gpu_ms = |store: Option<&bevy::diagnostic::DiagnosticsStore>, path: &'static str| -> f64 {
    store
      .and_then(|s| s.get(&bevy::diagnostic::DiagnosticPath::new(path)))
      .and_then(|d| d.value())
      .unwrap_or(-1.0)
  };
  let trace = gpu_ms(store.as_deref(), "render/gate_dda_trace/elapsed_gpu");
  let ddgi = gpu_ms(store.as_deref(), "render/gate_ddgi_update/elapsed_gpu");
  let direct = gpu_ms(store.as_deref(), "render/gate_direct_light/elapsed_gpu");
  let gi_ind = gpu_ms(store.as_deref(), "render/gate_gi_indirect/elapsed_gpu");
  let gi_den = gpu_ms(store.as_deref(), "render/gate_gi_denoise/elapsed_gpu");
  let blit = gpu_ms(store.as_deref(), "render/gate_dda_blit/elapsed_gpu");
  // 注意：gate_gi_denoise 每帧 4 个 pass 同名 span，diagnostic 给的是累加值（≈4×单次）
  let gi = if gi_ind < 0.0 && gi_den < 0.0 {
    -1.0
  } else {
    gi_ind.max(0.0) + gi_den.max(0.0)
  };
  let parts = [trace, ddgi, direct, if gi >= 0.0 { gi } else { -1.0 }, blit];
  let frame = if parts.iter().any(|&v| v >= 0.0) {
    parts.iter().filter(|&&v| v >= 0.0).sum()
  } else {
    -1.0
  };
  use std::fmt::Write as _;
  let _ = writeln!(
    *buf,
    "{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3}",
    time.elapsed_secs(),
    dt * 1000.0,
    frame,
    trace,
    ddgi,
    direct,
    gi
  );
  *acc += dt;
  if *acc >= FPS_REFRESH_SECS
    && let Some(f) = log_file.as_mut()
    && !buf.is_empty()
  {
    use std::io::Write as _;
    let _ = f.write_all(buf.as_bytes());
    buf.clear();
    *acc = 0.0;
  }
}
