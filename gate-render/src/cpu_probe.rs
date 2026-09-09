//! 每帧真实 CPU 工作时长测量（排除 vsync / present 帧队列反压等待）。
//!
//! bevy 0.19 的一帧在主线程上严格串行：主 world schedule → render 子 app
//! ExtractSchedule → Render schedule（集链：ExtractCommands→PrepareMeshes→
//! CreateViews→Specialize→PrepareViews→Queue→PhaseSort→Prepare→Render[
//! render_system 内跑 RenderGraph：Begin→Render→Submit→Finish，之后 present]）。
//!
//! vsync（FIFO present mode）与 GPU 帧队列反压的阻塞按 wgpu 帧节拍设计应发生在
//! prepare_windows 的 `surface.get_current_texture()`（RenderSystems::PrepareViews；
//! bevy 官方注释："get_current_texture can take a long time if the GPU workload is
//! the performance bottleneck"，bevy_render src/view/window/mod.rs）；queue.submit /
//! 为实证等待位置，每帧打 6 个时间戳并分段回传：
//! - seg_main：主世界 First → ExtractSchedule（主世界 schedule 全部 CPU）
//! - seg_pre：ExtractSchedule 末 → CreateViews（extract 命令应用 + acquire 前 prepare）
//! - seg_acq：CreateViews → Queue（Specialize + PrepareViews；**acquire 等待应在此**）
//! - seg_prep：Queue → RenderGraph Begin（Queue/PhaseSort/Prepare 集）
//! - seg_graph：RenderGraph Begin → Submit（命令录制）
//! - seg_submit：Submit → Finish（命令提交）
//!
//! CPU 工作量（折线）= seg_main+seg_pre+seg_prep+seg_graph+seg_submit（不含 seg_acq）。
//! 各段 ms 一并经 mpsc 通道回传主世界（fps_line_feed try_recv，晚 1 帧）。

use std::sync::mpsc::{Sender, channel};
use std::time::Instant;

use bevy::prelude::*;
use bevy::render::{Extract, ExtractSchedule, Render, RenderApp, RenderSystems};
use bevy::render::renderer::{RenderGraph, RenderGraphSystems};

/// 主世界：本帧 CPU 工作起点（First 集系统每帧刷新）。
#[derive(Resource)]
pub struct CpuFrameStart(pub Instant);

/// 主世界：render 侧回传的每帧 CPU 分段时长（ms；-1 = 该段未采到）。
/// Receiver 非 Sync，照 ddgi readback 模式套 `Mutex<Option<…>>`。
#[derive(Resource)]
pub struct CpuWorkReport(pub std::sync::Mutex<Option<std::sync::mpsc::Receiver<CpuTiming>>>);

/// 每帧 CPU 分段测量（ms）。
#[derive(Clone, Copy)]
pub struct CpuTiming {
  /// 折线口径总 CPU 工作量 = seg_main+seg_pre+seg_prep+seg_graph+seg_submit（不含 seg_acq）
  pub total: f32,
  /// First → ExtractSchedule（主世界 schedule 全部 CPU）
  pub seg_main: f32,
  /// ExtractSchedule 末 → CreateViews（extract 命令应用 + acquire 前 prepare）
  pub seg_pre: f32,
  /// CreateViews → Queue（含 PrepareViews acquire 阻塞，**不计入 total**）
  pub seg_acq: f32,
  /// Queue → RenderGraph Begin（acquire 后的 prepare CPU）
  pub seg_prep: f32,
  /// Begin → Submit（RenderGraph 命令录制 CPU）
  pub seg_graph: f32,
  /// Submit → Finish（命令提交 CPU）
  pub seg_submit: f32,
}

/// render 世界：跨系统测量状态（6 段时间戳 + 回传通道）。
#[derive(Resource)]
struct CpuProbeState {
  tx: Sender<CpuTiming>,
  t0: Option<Instant>,
  t_e: Option<Instant>,
  t1: Option<Instant>,
  t2: Option<Instant>,
  t3: Option<Instant>,
  t4: Option<Instant>,
}

/// 主世界 First：帧首打点。
fn mark_frame_start(mut stamp: ResMut<CpuFrameStart>) {
  stamp.0 = Instant::now();
}

/// ExtractSchedule：主世界帧起点直接写入 render world 状态（ExtractSchedule 在
/// render world 上运行，资源可直接访问，不经 commands）+ 打 extract 点。
fn extract_frame_start(
  start: Option<Extract<Res<CpuFrameStart>>>,
  mut state: ResMut<CpuProbeState>,
) {
  let now = Instant::now();
  // 临时探针：First→extract 的跨度在 extract 内就地计算（不经状态机），
  // 用于定位 t0 滞后来源（~1ms = 戳新鲜；~16ms = mark_frame_start 有问题）。
  if let Some(s) = start {
    state.t0 = Some(s.0);
    eprintln!("[cpu_probe] first->extract = {:.3} ms", (now - s.0).as_secs_f32() * 1000.0);
  }
  state.t_e = Some(now);
}

/// CreateViews 集（下一个集 PrepareViews 才 acquire）：t1。
fn mark_create_views(mut state: ResMut<CpuProbeState>) {
  if state.t0.is_some() {
    state.t1 = Some(Instant::now());
  }
}

/// Queue 集（PrepareViews/acquire 已完成）：t2。
fn mark_queue(mut state: ResMut<CpuProbeState>) {
  if state.t1.is_some() {
    state.t2 = Some(Instant::now());
  }
}

/// RenderGraph Begin 集：t3（RenderGraph 内最前）。
fn mark_graph_begin(mut state: ResMut<CpuProbeState>) {
  if state.t2.is_some() {
    state.t3 = Some(Instant::now());
  }
}

/// RenderGraph Submit 集（命令提交后）：t4。
fn mark_submit(mut state: ResMut<CpuProbeState>) {
  if state.t3.is_some() {
    state.t4 = Some(Instant::now());
  }
}

/// RenderGraph Finish 集：t5，分段合计回传主世界。
fn finish_and_report(mut state: ResMut<CpuProbeState>) {
  if let (Some(t0), Some(te), Some(t1), Some(t2), Some(t3), Some(t4)) =
    (state.t0, state.t_e, state.t1, state.t2, state.t3, state.t4)
  {
    let t5 = Instant::now();
    let seg_main = (te - t0).as_secs_f32() * 1000.0;
    let seg_pre = (t1 - te).as_secs_f32() * 1000.0;
    let seg_acq = (t2 - t1).as_secs_f32() * 1000.0;
    let seg_prep = (t3 - t2).as_secs_f32() * 1000.0;
    let seg_graph = (t4 - t3).as_secs_f32() * 1000.0;
    let seg_submit = (t5 - t4).as_secs_f32() * 1000.0;
    let total = seg_main + seg_pre + seg_prep + seg_graph + seg_submit;
    let _ = state.tx.send(CpuTiming {
      total,
      seg_main,
      seg_pre,
      seg_acq,
      seg_prep,
      seg_graph,
      seg_submit,
    });
  }
  // 下一帧重新打点（首帧/无 RenderGraph 帧不回传）
  state.t0 = None;
  state.t_e = None;
  state.t1 = None;
  state.t2 = None;
  state.t3 = None;
  state.t4 = None;
}

pub struct CpuProbePlugin;

impl Plugin for CpuProbePlugin {
  fn build(&self, app: &mut App) {
    let (tx, rx) = channel::<CpuTiming>();
    app.insert_resource(CpuFrameStart(Instant::now()));
    app.insert_resource(CpuWorkReport(std::sync::Mutex::new(Some(rx))));
    app.add_systems(First, mark_frame_start);

    let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
      return;
    };
    render_app.insert_resource(CpuProbeState {
      tx,
      t0: None,
      t_e: None,
      t1: None,
      t2: None,
      t3: None,
      t4: None,
    });
    render_app.add_systems(ExtractSchedule, extract_frame_start);
    render_app.add_systems(Render, mark_create_views.in_set(RenderSystems::CreateViews));
    render_app.add_systems(Render, mark_queue.in_set(RenderSystems::Queue));
    render_app.add_systems(RenderGraph, mark_graph_begin.in_set(RenderGraphSystems::Begin));
    render_app.add_systems(RenderGraph, mark_submit.in_set(RenderGraphSystems::Submit));
    render_app.add_systems(RenderGraph, finish_and_report.in_set(RenderGraphSystems::Finish));
  }
}
