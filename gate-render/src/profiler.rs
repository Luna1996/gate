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

// 渲染帧周期拆解（**唯一的"帧率限制在哪一侧"量具**）：`Begin` 记起点、`Finish` 记终点 ⇒
//   · 「自身」= 渲染图自己花的时间（extract / prepare 在 `Begin` 之前，故不在此列）；
//   · 「等别处」= 周期 − 自身 = 渲染世界停在调度之外的时间（等主世界 / 等 present）。
// WHY 必须有它：GPU 逐 pass 之和 + 进程 CPU% 都正常、但帧率很烂时，只看这两项会把人引到错误的方向
// （本仓实测过一次：`gate_gi` 已降到 1.1 ms、GPU 共 5 ms、CPU 11%，帧率却只有 9 fps ——
// 真凶是主世界的 ray-guided 装载路径，靠这一行走查出来）。用 `debug!`：它是周期量、排查时才开。
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
    // M4 请求通道：主世界留一份（`infinite_cubes::stream_chunks` 读需求），渲染世界共享同一份
    // （`report_lod_requests` 写）。两个世界都插入 ⇒ 关掉开关时它就是一张空表。
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
    // 渲染帧周期拆解（**"帧率限制在哪一侧"的唯一量具**，见资源注释）。
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
    // 叶级 LOD 诊断读回（M0）：只在 `trace.wesl::LOD_DIAG` 打开时注册 ⇒ 关闭时零成本、零日志。
    // 开关从 `.wesl` 源码解析（单一来源），Rust 侧不另抄一份。
    let trace = crate::wesl_consts::trace_consts();
    if trace.lod_diag != 0 {
      render_app.add_systems(
        bevy::render::renderer::RenderGraph,
        report_lod_diag.in_set(bevy::render::renderer::RenderGraphSystems::Finish),
      );
    }
    // M4 ray-guided 请求读回：同样只看 `trace.wesl::REQ_ENABLE` 这一个开关。
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

/// 一条 **ray-guided 请求**（**合并后**的一条 = 一个 chunk）：几票 = 有多少条采样主射线要它。
///
/// 编码侧是 shader 的 `trace.wesl::req_push`（请求字 = 窗口相对下标 + 档位），合并与窗口下标 →
/// 绝对坐标的还原都在 [`report_lod_requests`] 里做（只有那里拿得到 `main_window_origin`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LodRequest {
  /// 请求的 chunk（**绝对** chunk 坐标）
  pub chunk: IVec3,
  /// 票数：主射线里有多少条撞在"这个 chunk 不在 GPU 上"上 —— 越大的越该先补（消费端按它排）
  pub votes: u32,
  /// 射线报回的**最少需要多细**（`trace.wesl::req_level` 的下标：0 = 全分辨率、1 = 16³、2 = 64³、
  /// 3 = 整 chunk）。同一个 chunk 被多档请求过时取**最细**的那个。
  ///
  /// WHY 要带着它走：档位原本在消费端**一律按距离**给（`detail_at`），于是"射线只想要 16³ 的远处大块"
  /// 也会按近处规则升到全分辨率 ⇒ 反复细化、几何内容反复变（GI 时域永远接不上）。论文的口径是
  /// **refinement 由渲染结果给**，所以档位必须来自请求。
  pub level: u8,
}

/// **最近一批 ray-guided 请求**（渲染世界写、主世界读）。
///
/// 跨世界手法与 [`crate::brickmap::upload::UploadCpuSampleChannel`] 同一套：`ExtractResource` 只搬
/// "变化过的资源"，而这条通道每 `REPORT_PERIOD_SECS` 换一批 ⇒ 用 `Arc<Mutex<..>>` 两个世界共享同一份。
///
/// 空表 = 没有需求（`trace.wesl::REQ_ENABLE` 关着，或这一窗口没人看缺块）⇒ 消费端退回纯半径启发式。
///
/// 表是**快照**（每 `REPORT_PERIOD_SECS` 整份替换，不是队列）⇒ 见 [`Self::peek`]。
#[derive(Resource, Clone, Default)]
pub struct LodRequestFeed(pub Arc<std::sync::Mutex<Vec<LodRequest>>>);

impl LodRequestFeed {
  /// **看一眼**本批请求（**不清空**）。表是快照 ⇒ 每个消费者都该看到同一份，不该有"谁先读谁拿走"。
  ///
  /// WHY 不用 `std::mem::take`：GPU 侧（`plan_residency`）与 CPU 侧（`stream_chunks`）**都要**这份
  /// 需求，而两个消费者里只有一个能 take 成功 ⇒ 另一个永远读到空表（需求通道形同虚设）。
  pub fn peek(&self) -> Vec<LodRequest> {
    self.0.lock().unwrap_or_else(|e| e.into_inner()).clone()
  }
}

/// 一条**用途戳**（论文 §III.A 的 usage stamp）：主射线最近看到过这个 chunk。
///
/// 与 [`LodRequest`] 的差别是结构性的：请求 = "缺了，要装"（离散、稀有），用途 = "在看着"（连续、海量）
/// ⇒ 用途必须走**稠密表 + `atomicMax`**（`trace.wesl::req_use`），不能走环记录 —— 后者会被投票打爆。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LodUse {
  /// 被看到的 chunk（**绝对** chunk 坐标）
  pub chunk: IVec3,
  /// 用途戳（单调计数器，只在同一块缓冲内比较先后）
  pub stamp: u32,
}

/// **最近一批用途戳**（渲染世界写、两个世界都读）。
///
/// 消费者：`brickmap::upload::plan_residency` 把它喂给 `Residency::note_use`（LRU 的"最近使用"），
/// 以及主世界的 `infinite_cubes::stream_chunks`（它自己的 CPU 侧保留策略按同一个信号裁）。
/// 空表 = 这一窗口没人看（或 `trace.wesl::REQ_ENABLE` 关着）⇒ 消费端各自退回"无信号"路径。
#[derive(Resource, Clone, Default)]
pub struct ChunkUseFeed(pub Arc<std::sync::Mutex<Vec<LodUse>>>);

impl ChunkUseFeed {
  /// **看一眼**本批用途戳（**不清空**）—— 理由同 [`LodRequestFeed::peek`]：GPU 侧与 CPU 侧都要它。
  pub fn peek(&self) -> Vec<LodUse> {
    self.0.lock().unwrap_or_else(|e| e.into_inner()).clone()
  }
}

/// chunk 相对窗口下标的解包（`trace.wesl::req_push` 的位域：3 × 6 位）
fn req_rel(key: u32) -> IVec3 {
  IVec3::new((key & 63) as i32, ((key >> 6) & 63) as i32, ((key >> 12) & 63) as i32)
}

/// **M4 ray-guided 请求的读回**（`docs/editable-gigavoxel.md` §4 M4）：每 [`crate::consts::REPORT_PERIOD_SECS`]
/// 把 `gpu.lod_req`（环缓冲 + 用途戳表）拷进 staging 同步读回：用途戳整批发给
/// [`ChunkUseFeed`]（LRU 的"最近使用"），请求**合并**后落一行 `REQ[...]`（新增条数 → 去重后的
/// chunk 数 → 最热的几个 → 超容丢失条数）并发给 [`LodRequestFeed`]。
///
/// 与 [`report_lod_diag`] 同一取舍（自建 encoder + `map_buffer` + `poll` 等待）与同一注册条件
/// （`trace.wesl::REQ_ENABLE` 非 0；Rust 经 [`crate::wesl_consts::trace_consts`] 读同一份源码）。
///
/// 与诊断计数器的差别：那是累积量（只加不清）⇒ 读差值；这里是**环缓冲** ⇒ 差值只用来算"本窗口新增
/// 了几条"，字面值本身要解码（见 `trace.wesl::req_push`），且只解释**最近**那批（跨窗口累积超过容量
/// 时尾部就是全部有意义的样本）。
fn report_lod_requests(
  device: Res<RenderDevice>,
  queue: Res<RenderQueue>,
  gpu: Option<Res<crate::brickmap::upload::GpuBrickMap>>,
  feed: Option<Res<LodRequestFeed>>,
  use_feed: Option<Res<ChunkUseFeed>>,
  mut period: Local<Option<std::time::Instant>>,
  mut prev: Local<Option<[u32; 2]>>,
  // 合并用的稠密 64³ 计数器（1 M 字 = 4 MB）：只在首次分配，之后每窗 `fill(0)` 复用
  mut counts: Local<Vec<u32>>,
  // 同上尺寸的"每格最细请求档位"（与 `counts` 同一趟填）
  mut min_levels: Local<Vec<u32>>,
) {
  use crate::brickmap::consts::{LOD_REQ_WORDS, REQ_BASE, REQ_CAP, USE_BASE, USE_WORDS};
  let Some(gpu) = gpu else { return };
  let now = std::time::Instant::now();
  let due = period
    .is_none_or(|t| now.duration_since(t).as_secs_f32() >= crate::consts::REPORT_PERIOD_SECS);
  if !due {
    return;
  }
  *period = Some(now);

  let bytes = (LOD_REQ_WORDS * 4) as u64;
  let staging = device.create_buffer(&BufferDescriptor {
    label: Some("gate_lod_req_staging"),
    size: bytes,
    usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
    mapped_at_creation: false,
  });
  let mut enc =
    device.create_command_encoder(&bevy::render::render_resource::CommandEncoderDescriptor {
      label: Some("gate_lod_req_readback"),
    });
  enc.copy_buffer_to_buffer(&gpu.lod_req, 0, &staging, 0, bytes);
  queue.submit([enc.finish()]);

  let slice = staging.slice(..);
  let (tx, rx) = std::sync::mpsc::channel();
  device.map_buffer(&slice, MapMode::Read, move |r| {
    let _ = tx.send(r);
  });
  if device.poll(PollType::wait_indefinitely()).is_err()
    || !matches!(rx.recv_timeout(std::time::Duration::from_secs(5)), Ok(Ok(())))
  {
    warn!("REQ 读回：等待 / 映射失败 → 本窗口跳过");
    staging.unmap();
    return;
  }
  let mut words = vec![0u32; LOD_REQ_WORDS];
  if let Ok(view) = slice.get_mapped_range() {
    for (i, w) in words.iter_mut().enumerate() {
      let o = i * 4;
      *w = u32::from_le_bytes([view[o], view[o + 1], view[o + 2], view[o + 3]]);
    }
  }
  staging.unmap();

  // `[1]` 保留（见 `trace.wesl::REQ_RESERVED`：那个"溢出计数"是累计量，读不出"这一窗口丢没丢"）
  let (count, tick) = (words[0], words[2]);

  // ---- ① 用途戳（LRU 的"最近使用"信号；**先发**，因为它与请求无关：没有缺块时它也照样有值）----
  // 表是稠密的 ⇒ 扫满 `USE_WORDS` 格（262144 次顺序读，每窗一次 ≈ 0.1 ms），只挑戳落在
  // **区间 `(上次读的戳, 本次读的戳]`** 里的 —— 陈旧条目天然落在区间外，所以表**不需要清零**。
  let prev_tick = prev.map(|p| p[1]).unwrap_or(tick);
  let span = tick.wrapping_sub(prev_tick);
  let mut used: Vec<LodUse> = Vec::new();
  if span > 0 {
    for i in 0..USE_WORDS {
      let s = words[USE_BASE + i];
      if s == 0 {
        continue;
      }
      let d = tick.wrapping_sub(s);
      if d > 0 && d <= span {
        used.push(LodUse { chunk: gpu.main_window_origin + req_rel(i as u32), stamp: s });
      }
    }
  }
  let used_n = used.len();
  if let Some(f) = use_feed.as_ref() {
    *f.0.lock().unwrap_or_else(|e| e.into_inner()) = used;
  }

  let new_reqs = match *prev {
    // 首个窗口只建立基线：倒推"最近 N 条"要靠差值
    None => 0,
    Some(p) => count.wrapping_sub(p[0]),
  };
  *prev = Some([count, tick]);
  // 环满不代表丢数据：环指针是**累计**的，`count ≥ REQ_CAP` 之后每条新请求都会覆盖一条最旧的，
  // 而消费端本来也只读最新 `REQ_CAP` 条 ⇒ 只有"**这一窗口**的新增条数超过环容量"才真的丢。
  let lost = new_reqs.saturating_sub(REQ_CAP as u32);
  let n = (new_reqs as usize).min(REQ_CAP);
  // 本窗口**没有任何请求**（都装好了 / 没人在看缺块）⇒ 需求表清空：消费端退回纯半径启发式。
  // 用途戳已经在上面发过了（它与请求是两条独立的信号）。
  let set_feed = |list: Vec<LodRequest>| {
    if let Some(feed) = feed.as_ref() {
      *feed.0.lock().unwrap_or_else(|e| e.into_inner()) = list;
    }
  };
  if n == 0 {
    set_feed(Vec::new());
    debug!("REQ[本窗口 0 条请求；用途戳 {used_n} chunk]");
    return;
  }

  // 合并：同一 chunk 的多条请求合成一条（票数 = 有多少条**采样射线**要它）。
  // 用**稠密 64³ 计数器**（键就是窗口相对下标，18 位 ⇒ 正好 64³）而不是 HashMap：一整屏的采样射线
  // 有几十万条事件，哈希表要 20–30 ms（每 2 s 抖一下），稠密数组是 1 M 次顺序写（~1–2 ms）。
  let counts = &mut *counts;
  if counts.len() != 64 * 64 * 64 {
    *counts = vec![0u32; 64 * 64 * 64];
  }
  counts.fill(0);
  // 每格的**最细**请求档位（`min` 合并；初值 3 = 最粗 ⇒ 任何请求都会把它压低）
  let min_lv = &mut *min_levels;
  if min_lv.len() != 64 * 64 * 64 {
    *min_lv = vec![3u32; 64 * 64 * 64];
  }
  min_lv.fill(3);
  for i in 0..n {
    // 环缓冲：最近一条在 `count - 1` 处，往前倒推
    let slot = count.wrapping_sub(1).wrapping_sub(i as u32) as usize % REQ_CAP;
    let w = words[REQ_BASE + slot];
    let key = (w & 0x3_FFFF) as usize;
    counts[key] = counts[key].saturating_add(1);
    let lv = (w >> 18) & 3;
    if lv < min_lv[key] {
      min_lv[key] = lv;
    }
  }
  let mut top: Vec<(usize, u32)> =
    counts.iter().enumerate().filter(|(_, v)| **v > 0).map(|(k, v)| (k, *v)).collect();
  let distinct = top.len();
  top.sort_unstable_by_key(|(k, v)| (std::cmp::Reverse(*v), *k));
  // 只把**票数最高的前 `REQ_FEED_MAX` 条**喂给消费端（见该常量的说明）：需求表要的是"最想要的那
  // 一批"，而带一万多条进主线程会让每帧的拷贝 + 排序变成几十 ms 的尖峰。日志里的条数仍报真实的
  // 去重总数（`distinct`），所以"信息量够不够"照旧看得见。
  top.truncate(crate::brickmap::consts::REQ_FEED_MAX);
  let merged: Vec<LodRequest> = top
    .iter()
    .map(|(key, votes)| LodRequest {
      chunk: gpu.main_window_origin + req_rel(*key as u32),
      votes: *votes,
      level: min_lv[*key].min(3) as u8,
    })
    .collect();
  let shown: Vec<String> = merged
    .iter()
    .take(6)
    .map(|r| format!("({},{},{})×{}", r.chunk.x, r.chunk.y, r.chunk.z, r.votes))
    .collect();
  set_feed(merged);
  info!(
    "REQ[去重 {distinct} chunk、取前 {}（最热 {}）；本窗口 {new_reqs} 条、超容丢失 {lost}；用途戳 {used_n} chunk]",
    top.len(),
    shown.join(" ")
  );
}
