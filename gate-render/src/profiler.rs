use bevy::prelude::*;
use bevy::render::render_resource::{
  Buffer, BufferDescriptor, BufferUsages, CommandEncoder, ComputePass, ComputePassDescriptor,
  MapMode, PollType,
};
#[cfg(feature = "profile")]
use bevy::render::renderer::{PendingCommandBuffers, RenderAdapter};
use bevy::render::renderer::{RenderDevice, RenderQueue};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Resource, Clone, Default)]
pub struct FramePace {
  pub presented: Arc<AtomicU64>,
}

fn tick_frame_pace(pace: Res<FramePace>) {
  pace.presented.fetch_add(1, Ordering::Relaxed);
}

#[derive(Resource, Default)]
pub(crate) struct DiagFrameGap {
  prev_end: Option<std::time::Instant>,
  busy0: Option<std::time::Instant>,
  acc: f64,
  acc_busy: f64,
  n: u32,
}

fn diag_gap_begin(mut st: ResMut<DiagFrameGap>) {
  st.busy0 = Some(std::time::Instant::now());
}

pub struct SysTimer<'a> {
  t0: std::time::Instant,
  label: &'static str,
  acc: &'a mut (f64, u32),
}

impl<'a> SysTimer<'a> {
  pub fn new(label: &'static str, acc: &'a mut (f64, u32)) -> Self {
    Self { t0: std::time::Instant::now(), label, acc }
  }
}

impl Drop for SysTimer<'_> {
  fn drop(&mut self) {
    self.acc.0 += self.t0.elapsed().as_secs_f64();
    self.acc.1 += 1;
    if self.acc.1 >= 60 {
      bevy::log::debug!(target: "gate", "{} {:.2} ms/帧", self.label, self.acc.0 / self.acc.1 as f64 * 1000.0);
      self.acc.0 = 0.0;
      self.acc.1 = 0;
    }
  }
}

pub const SPLIT_DIAG: bool = false;

pub struct SplitDiag {
  names: &'static [&'static str],
  acc: Vec<f64>,
  n: u32,
  t0: std::time::Instant,
}

impl SplitDiag {
  pub fn new(names: &'static [&'static str]) -> Self {
    Self { names, acc: vec![0.0; names.len()], n: 0, t0: std::time::Instant::now() }
  }

  pub fn mark(&mut self, i: usize) {
    if !SPLIT_DIAG {
      return;
    }
    let now = std::time::Instant::now();
    self.acc[i] += now.duration_since(self.t0).as_secs_f64();
    self.t0 = now;
  }

  pub fn start(&mut self) {
    if !SPLIT_DIAG {
      return;
    }
    self.t0 = std::time::Instant::now();
  }

  pub fn frame_end(&mut self, label: &str) {
    if !SPLIT_DIAG {
      return;
    }
    self.n += 1;
    if self.n >= 60 {
      let mut line = String::new();
      for (name, s) in self.names.iter().zip(&self.acc) {
        line.push_str(&format!("{name}={:.2} ", s / self.n as f64 * 1000.0));
      }
      bevy::log::debug!(target: "gate", "SPLIT {label} ms/帧: {line}");
      self.acc.iter_mut().for_each(|v| *v = 0.0);
      self.n = 0;
    }
    self.t0 = std::time::Instant::now();
  }
}

fn diag_gap_end(mut st: ResMut<DiagFrameGap>) {
  let now = std::time::Instant::now();
  let (Some(t0), Some(b0)) = (st.prev_end, st.busy0) else {
    st.prev_end = Some(now);
    return;
  };
  st.acc += now.duration_since(t0).as_secs_f64();
  st.acc_busy += now.duration_since(b0).as_secs_f64();
  st.n += 1;
  st.prev_end = Some(now);
  if st.n >= 60 {
    let n = st.n as f64;
    bevy::log::debug!(target: "gate",
      "RENDER 帧周期 {:.1} ms（{:.1} fps）：渲染世界自身 {:.1} ms，等别处 {:.1} ms",
      st.acc / n * 1000.0, n / st.acc, st.acc_busy / n * 1000.0, (st.acc - st.acc_busy) / n * 1000.0);
    st.acc = 0.0;
    st.acc_busy = 0.0;
    st.n = 0;
  }
}

#[derive(Resource, Default)]
pub(crate) struct GpuProfilerRes {
  #[cfg(feature = "profile")]
  profiler: Option<wgpu_profiler::GpuProfiler>,
  #[cfg(feature = "profile")]
  report: PassReport,
}

#[cfg(feature = "profile")]
#[derive(Default)]
struct PassReport {
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

#[cfg(feature = "profile")]
pub(crate) fn profiler_mut(res: &mut GpuProfilerRes) -> Option<&mut wgpu_profiler::GpuProfiler> {
  res.profiler.as_mut()
}

#[cfg(feature = "profile")]
pub fn timestamp_wgpu_features() -> bevy::render::render_resource::WgpuFeatures {
  wgpu_profiler::GpuProfiler::ALL_WGPU_TIMER_FEATURES
}

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

#[derive(Resource, Default)]
pub(crate) struct DiagMainFrame {
  t0: Option<std::time::Instant>,
  acc: f64,
  n: u32,
}

fn main_frame_begin(mut st: ResMut<DiagMainFrame>) {
  st.t0 = Some(std::time::Instant::now());
}

fn main_frame_end(mut st: ResMut<DiagMainFrame>) {
  let Some(t0) = st.t0.take() else { return };
  st.acc += t0.elapsed().as_secs_f64();
  st.n += 1;
  if st.n >= 60 {
    bevy::log::debug!(target: "gate", "MAIN 主世界调度 {:.2} ms/帧", st.acc / st.n as f64 * 1000.0);
    st.acc = 0.0;
    st.n = 0;
  }
}

pub(crate) struct GateProfilerPlugin;

impl Plugin for GateProfilerPlugin {
  fn build(&self, app: &mut App) {
    app.init_resource::<DiagMainFrame>();
    app.add_systems(bevy::prelude::First, main_frame_begin);
    app.add_systems(bevy::prelude::Last, main_frame_end);
    let pace = FramePace::default();
    app.insert_resource(pace.clone());
    let req_feed = LodRequestFeed::default();
    app.insert_resource(req_feed.clone());
    let use_feed = ChunkUseFeed::default();
    app.insert_resource(use_feed.clone());
    let Some(render_app) = app.get_sub_app_mut(bevy::render::RenderApp) else {
      return;
    };
    render_app.init_resource::<GpuProfilerRes>();
    render_app.insert_resource(pace);
    render_app.insert_resource(req_feed);
    render_app.insert_resource(use_feed);
    render_app.init_resource::<DiagFrameGap>();
    render_app.add_systems(
      bevy::render::renderer::RenderGraph,
      diag_gap_begin.in_set(bevy::render::renderer::RenderGraphSystems::Begin),
    );
    render_app.add_systems(
      bevy::render::renderer::RenderGraph,
      diag_gap_end.in_set(bevy::render::renderer::RenderGraphSystems::Finish),
    );
    render_app.add_systems(
      bevy::render::renderer::RenderGraph,
      tick_frame_pace.in_set(bevy::render::renderer::RenderGraphSystems::Finish),
    );
    let trace = crate::wesl_consts::trace_consts();
    if trace.lod_diag != 0 {
      render_app.add_systems(
        bevy::render::renderer::RenderGraph,
        report_lod_diag.in_set(bevy::render::renderer::RenderGraphSystems::Finish),
      );
    }
    if trace.req_enable != 0 {
      render_app.add_systems(
        bevy::render::renderer::RenderGraph,
        report_lod_requests.in_set(bevy::render::renderer::RenderGraphSystems::Finish),
      );
    }
    #[cfg(feature = "profile")]
    {
      render_app.add_systems(bevy::render::RenderStartup, init_gpu_profiler);
      render_app.add_systems(
        bevy::render::renderer::RenderGraph,
        (
          resolve_profiler_queries
            .after(bevy::render::renderer::RenderGraphSystems::Render)
            .before(bevy::render::renderer::RenderGraphSystems::Submit),
          finish_profiler_frame.in_set(bevy::render::renderer::RenderGraphSystems::Finish),
        ),
      );
    }
  }
}

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
  pending.push_encoder(encoder, "wgpu_profiler_resolve");
}

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
  let due =
    period.is_none_or(|t| now.duration_since(t).as_secs_f32() >= crate::consts::REPORT_PERIOD_SECS);
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
  let mut enc =
    device.create_command_encoder(&bevy::render::render_resource::CommandEncoderDescriptor {
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
  let entries = d[0].max(1);
  let face_total = (d[3] + d[4] + d[5] + d[6] + d[7] + d[8]).max(1);
  let pct = |v: u32| v as f32 * 100.0 / face_total as f32;
  info!(
    "DIAG[leaf_in {} lod_stop {} {:.1}% illegal {} | 面查表 精确 {}({:.1}%) 回退 {}({:.1}%) \
     回退不可用 {}({:.1}%) 回退槽失配 {}({:.1}%) 内联 {}({:.1}%) 天空 {}({:.1}%)]",
    d[0],
    d[1],
    d[1] as f32 * 100.0 / entries as f32,
    d[2],
    d[3],
    pct(d[3]),
    d[4],
    pct(d[4]),
    d[5],
    pct(d[5]),
    d[6],
    pct(d[6]),
    d[7],
    pct(d[7]),
    d[8],
    pct(d[8]),
  );
  //
  let main_total = (d[9]).max(1);
  let mpct = |v: u32| v as f32 * 100.0 / main_total as f32;
  info!(
    "DIAG[主射线 {} 条（1/64 采样）| 近场 {} {:.1}% 远场 {} {:.1}% 天空 {} {:.1}%]",
    d[9],
    d[10],
    mpct(d[10]),
    d[11],
    mpct(d[11]),
    d[12],
    mpct(d[12]),
  );
  //
  let gi_texels = (d[13]).max(1);
  let gi_hits = (d[19]).max(1);
  info!(
    "DIAG[GI {} texel|候选射线 {}（{:.1}/texel）| NEE 阴影 {}（{:.1}/texel）| 命中历史 {:.1}%| \
     静止 {:.1}% 并 tap {:.1}| 键越界 {:.1}% 孤立 {:.1}% 钳制 {:.1}%]",
    d[13],
    d[14],
    d[14] as f32 / gi_texels as f32,
    d[15],
    d[15] as f32 / gi_texels as f32,
    d[16] as f32 * 100.0 / gi_hits as f32,
    d[17] as f32 * 100.0 / gi_hits as f32,
    d[18] as f32 / gi_hits as f32,
    d[20] as f32 * 100.0 / gi_hits as f32,
    d[21] as f32 * 100.0 / gi_hits as f32,
    d[22] as f32 * 100.0 / gi_hits as f32,
  );
  if d[23] > 0 || d[24] > 0 {
    let faces = (d[25]).max(1) as f32;
    info!(
      "DIAG[WAL 面 成熟 {:.1}% 未熟 {:.1}%（占被着色面 {}，平均池 Σn {:.0}）]",
      d[23] as f32 * 100.0 / faces,
      d[24] as f32 * 100.0 / faces,
      d[25],
      d[26] as f32 / faces,
    );
  }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LodRequest {
  pub vol: u8,
  pub chunk: IVec3,
  pub votes: u32,
  pub level: u8,
}

#[derive(Resource, Clone, Default)]
pub struct LodRequestFeed(pub Arc<std::sync::Mutex<Vec<LodRequest>>>);

impl LodRequestFeed {
  pub fn peek(&self) -> Vec<LodRequest> {
    self.0.lock().unwrap_or_else(|e| e.into_inner()).clone()
  }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LodUse {
  pub vol: u8,
  pub chunk: IVec3,
  pub stamp: u32,
}

#[derive(Resource, Clone, Default)]
pub struct ChunkUseFeed(pub Arc<std::sync::Mutex<Vec<LodUse>>>);

impl ChunkUseFeed {
  pub fn peek(&self) -> Vec<LodUse> {
    self.0.lock().unwrap_or_else(|e| e.into_inner()).clone()
  }
}

fn req_rel(key: u32) -> IVec3 {
  IVec3::new((key & 63) as i32, ((key >> 6) & 63) as i32, ((key >> 12) & 63) as i32)
}

struct ReqReadback {
  staging: Buffer,
  pending: bool,
  rx:
    Option<std::sync::mpsc::Receiver<Result<(), bevy::render::render_resource::BufferAsyncError>>>,
  words: Vec<u32>,
  submitted: u64,
  delivered: u64,
}

impl ReqReadback {
  fn new(device: &RenderDevice, bytes: u64) -> Self {
    let staging = device.create_buffer(&BufferDescriptor {
      label: Some("gate_lod_req_staging"),
      size: bytes,
      usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
      mapped_at_creation: false,
    });
    Self {
      staging,
      pending: false,
      rx: None,
      words: vec![0u32; crate::brickmap::consts::LOD_REQ_WORDS],
      submitted: 0,
      delivered: 0,
    }
  }

  fn fetch(&mut self, device: &RenderDevice) -> bool {
    if !self.pending {
      return false;
    }
    let _ = device.poll(bevy::render::render_resource::PollType::Poll);
    match self.rx.as_ref().map(|rx| rx.try_recv()) {
      Some(Ok(Ok(()))) => {
        let slice = self.staging.slice(..);
        if let Ok(view) = slice.get_mapped_range() {
          let (head, mid, _tail) = unsafe { view.align_to::<u32>() };
          let n = crate::brickmap::consts::LOD_REQ_WORDS;
          if head.is_empty() && mid.len() >= n {
            self.words.copy_from_slice(&mid[..n]);
          } else {
            for (i, w) in self.words.iter_mut().enumerate() {
              let o = i * 4;
              *w = u32::from_le_bytes([view[o], view[o + 1], view[o + 2], view[o + 3]]);
            }
          }
        }
        self.staging.unmap();
        self.pending = false;
        self.delivered += 1;
        true
      }
      Some(Ok(Err(_))) | Some(Err(std::sync::mpsc::TryRecvError::Disconnected)) => {
        warn!("REQ 读回：映射失败 → 本趟丢弃");
        self.staging.unmap();
        self.pending = false;
        self.rx = None;
        false
      }
      Some(Err(std::sync::mpsc::TryRecvError::Empty)) => false,
      None => false,
    }
  }

  fn submit(&mut self, device: &RenderDevice, queue: &RenderQueue, src: &Buffer, bytes: u64) {
    let mut enc =
      device.create_command_encoder(&bevy::render::render_resource::CommandEncoderDescriptor {
        label: Some("gate_lod_req_readback"),
      });
    enc.copy_buffer_to_buffer(src, 0, &self.staging, 0, bytes);
    queue.submit([enc.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    let slice = self.staging.slice(..);
    device.map_buffer(&slice, MapMode::Read, move |r| {
      let _ = tx.send(r);
    });
    self.rx = Some(rx);
    self.pending = true;
    self.submitted += 1;
  }
}

fn report_lod_requests(
  device: Res<RenderDevice>,
  queue: Res<RenderQueue>,
  gpu: Option<Res<crate::brickmap::upload::GpuBrickMap>>,
  feed: Option<Res<LodRequestFeed>>,
  use_feed: Option<Res<ChunkUseFeed>>,
  mut prev: Local<Option<[u32; 2]>>,
  mut counts: Local<Vec<u32>>,
  mut min_levels: Local<Vec<u32>>,
  mut touched: Local<Vec<u32>>,
  mut rb: Local<Option<ReqReadback>>,
  mut log_at: Local<Option<std::time::Instant>>,
  mut diag: Local<(f64, u32)>,
) {
  use crate::brickmap::consts::{LOD_REQ_WORDS, REQ_BASE, REQ_CAP, USE_BASE, USE_WORDS, VOLUMES};
  let Some(gpu) = gpu else { return };
  let bytes = (LOD_REQ_WORDS * 4) as u64;
  let Some(r) = rb.as_mut() else {
    *rb = Some(ReqReadback::new(&device, bytes));
    return;
  };
  let fresh = r.fetch(&device);
  if !r.pending {
    r.submit(&device, &queue, &gpu.lod_req, bytes);
  }
  if !fresh {
    return;
  }
  let _t = SysTimer::new("REQ 读回+合并", &mut diag);
  let now = std::time::Instant::now();
  let log_now =
    log_at.is_none_or(|t| now.duration_since(t).as_secs_f32() >= crate::consts::REPORT_PERIOD_SECS);
  if log_now {
    *log_at = Some(now);
  }
  let words = &r.words;
  let (count, tick) = (words[0], words[2]);

  let prev_tick = prev.map(|p| p[1]).unwrap_or(tick);
  let span = tick.wrapping_sub(prev_tick);
  let mut used: Vec<LodUse> = Vec::new();
  let nv = gpu.volume_windows.len().min(VOLUMES);
  if span > 0 {
    for vol in 0..nv {
      let origin = gpu.volume_windows[vol];
      for i in 0..USE_WORDS {
        let s = words[USE_BASE + vol * USE_WORDS + i];
        if s == 0 {
          continue;
        }
        let d = tick.wrapping_sub(s);
        if d > 0 && d <= span {
          used.push(LodUse { vol: vol as u8, chunk: origin + req_rel(i as u32), stamp: s });
        }
      }
    }
  }
  let used_n = used.len();
  if let Some(f) = use_feed.as_ref() {
    *f.0.lock().unwrap_or_else(|e| e.into_inner()) = used;
  }

  let new_reqs = match *prev {
    None => 0,
    Some(p) => count.wrapping_sub(p[0]),
  };
  *prev = Some([count, tick]);
  let lost = new_reqs.saturating_sub(REQ_CAP as u32);
  let n = (new_reqs as usize).min(REQ_CAP);
  let set_feed = |list: Vec<LodRequest>| {
    if let Some(feed) = feed.as_ref() {
      *feed.0.lock().unwrap_or_else(|e| e.into_inner()) = list;
    }
  };
  if n == 0 {
    set_feed(Vec::new());
    if log_now {
      debug!("REQ[本窗口 0 条请求；用途戳 {used_n} chunk]");
    }
    return;
  }

  let cells = VOLUMES * USE_WORDS;
  let counts = &mut *counts;
  if counts.len() != cells {
    *counts = vec![0u32; cells];
  }
  let min_lv = &mut *min_levels;
  if min_lv.len() != cells {
    *min_lv = vec![3u32; cells];
  }
  for &k in touched.iter() {
    counts[k as usize] = 0;
    min_lv[k as usize] = 3;
  }
  touched.clear();
  let mut slot = (count.wrapping_sub(1) as usize) % REQ_CAP;
  for _ in 0..n {
    let w = words[REQ_BASE + slot];
    let key = ((w >> 21) & 3) as usize * USE_WORDS + (w & 0x3_FFFF) as usize;
    if counts[key] == 0 {
      touched.push(key as u32);
    }
    counts[key] = counts[key].saturating_add(1);
    let lv = (w >> 18) & 3;
    if lv < min_lv[key] {
      min_lv[key] = lv;
    }
    if slot == 0 {
      slot = REQ_CAP - 1;
    } else {
      slot -= 1;
    }
  }
  let mut top: Vec<(usize, u32)> =
    touched.iter().map(|&k| (k as usize, counts[k as usize])).filter(|(_, v)| *v > 0).collect();
  let distinct = top.len();
  top.sort_unstable_by_key(|(k, v)| (std::cmp::Reverse(*v), *k));
  //
  let mut per_vol = [0usize; VOLUMES];
  top.retain(|(k, _)| {
    let v = k / USE_WORDS;
    if v < VOLUMES && per_vol[v] < crate::brickmap::consts::REQ_FEED_MAX {
      per_vol[v] += 1;
      true
    } else {
      false
    }
  });
  let merged: Vec<LodRequest> = top
    .iter()
    .map(|(key, votes)| {
      let vol = key / USE_WORDS;
      let origin = gpu.volume_windows.get(vol).copied().unwrap_or(IVec3::ZERO);
      LodRequest {
        vol: vol as u8,
        chunk: origin + req_rel((key % USE_WORDS) as u32),
        votes: *votes,
        level: min_lv[*key].min(3) as u8,
      }
    })
    .collect();
  if log_now {
    let shown: Vec<String> = merged
      .iter()
      .take(6)
      .map(|r| format!("v{}({},{},{})×{}", r.vol, r.chunk.x, r.chunk.y, r.chunk.z, r.votes))
      .collect();
    info!(
      "REQ[去重 {distinct} chunk、取 {} 条（按卷配额，最热 {}）；本窗口 {new_reqs} 条、超容丢失 {lost}；\
       用途戳 {used_n} chunk；读回 {}/{} 趟]",
      top.len(),
      shown.join(" "),
      r.delivered,
      r.submitted,
    );
  }
  set_feed(merged);
}
