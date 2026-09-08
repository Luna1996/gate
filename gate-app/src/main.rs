//! gate 演示应用入口：插件装配 + 系统注册。
//!
//! 代码按功能模块拆分，本文件只保留装配与一次性 UI 编排：
//! - [`scene`]：Startup 场景搭建（vox/demo 场景 + 相机初始机位 + brickmap 诊断）
//! - [`camera`]：轨道相机输入、左键拾取 recenter、调试开关（V 可见性缓存）
//! - [`debug_overlay`]：左上角 FPS/帧时折线 HUD + fps.log/frame_time.log 落盘，
//!   F3 切换显隐（默认显示）
//! - [`showcase`]：右上角 gate-ui 组件展示窗（交互观察者 + 演示折线喂数）
//! - [`diagnostics`]：GATE_BENCH=1 逐帧 GPU pass 时间日志（logs/gpu_frame.log）
//! - [`vox_scene`]：MagicaVoxel .vox 场景导入

mod camera;
mod debug_overlay;
mod diagnostics;
mod scene;
mod showcase;
mod vox_scene;

use bevy::{
  asset::{AssetPlugin, LoadState},
  log::LogPlugin,
  prelude::*,
  window::{PresentMode, Window},
};

use gate_render::VIEW_SIZE;
use gate_ui::{ThemeFont, UiCtx, UiTheme};

use camera::{left_click_pick_recenter, orbit_camera_input, probe_click_inspect};
use debug_overlay::{DemoUiRoot, debug_overlay_toggle, fps_line_feed, spawn_debug_view};
use diagnostics::gpu_frame_log;
use scene::setup;
use showcase::{showcase_demo_system, spawn_showcase};

/// 以 crate 目录为锚的 assets 路径，F5 / 终端启动行为一致
pub const ASSETS_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/assets");

/// 程序侧日志落盘（方案 3）：LogPlugin custom_layer 追加无色文件层，
/// 每次启动截断重写（永远最新一轮）；默认 stderr 彩色层保留不变
pub const LOG_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/logs/latest.log");

fn main() {
  // 【诊断】GATE_BENCH=1：静默后台（窗口不可见）+ vsync Fifo + 渲染诊断 + 逐帧帧时日志
  let bench = std::env::var("GATE_BENCH").as_deref() == Ok("1");
  let mut app = App::new();
  app.add_plugins(
    DefaultPlugins
      .set(WindowPlugin {
        primary_window: Some(Window {
          resolution: VIEW_SIZE.into(),
          // 【诊断】GATE_BENCH=1：① Fifo vsync（帧时测量用真实墙钟时间，不吃
          // immediate 模式的 frame pacing 伪影；高刷屏 vblank 6.9ms < trace 帧时，
          // 不会封顶掩盖差异）；② visible=false 静默后台运行，不抢前台焦点
          focused: false,
          present_mode: PresentMode::AutoVsync,
          resizable: true, // 2.7a FR-5：解锁任意 resize（渲染目标 + aspect 由响应式系统跟随）
          ..default()
        }),
        ..default()
      })
      .set(AssetPlugin {
        file_path: ASSETS_PATH.into(),
        ..default()
      })
      .set(LogPlugin {
        // info 基线 + 两条定向屏蔽：
        // 1) wgpu_hal::vulkan::instance = off：屏蔽 wgpu 29.0.4 启动时的 VUID 错误——
        //    已知 bug（wgpu#9213/#9361，VUID-VkPresentInfoKHR-pImageIndices-01430 和
        //    VUID-vkAcquireNextImageKHR-semaphore-01286），仅首 1-2 帧的 swapchain
        //    时序异常，不影响渲染功能与画面正确性；wgpu 升级后自动恢复。
        // 2) wgpu_hal::vulkan::surface = off：同来源的 surface 层偶发错误。
        //    其余 wgpu/winit/bevy_render 的错误照常打印，避免掩盖真实问题。
        filter: "info,\
          wgpu_hal::vulkan::instance=off,\
          wgpu_hal::vulkan::surface=off"
          .into(),
        // 日志落盘层（方案 3）：无色文件层，写 logs/latest.log；
        // 默认 stderr 彩色层不受影响，终端 / LLDB log 照常有色输出
        custom_layer: |_app| {
          use bevy::log::BoxedLayer;
          let path = std::path::Path::new(LOG_PATH);
          std::fs::create_dir_all(path.parent().expect("LOG_PATH 必有父目录")).ok()?;
          let file = std::fs::File::create(path).ok()?;
          let (writer, guard) = tracing_appender::non_blocking(file);
          std::mem::forget(guard);
          let offset = time::UtcOffset::current_local_offset().unwrap_or(time::UtcOffset::UTC);
          let timer = tracing_subscriber::fmt::time::OffsetTime::new(
            offset,
            time::format_description::well_known::Rfc3339,
          );
          Some(Box::new(
            tracing_subscriber::fmt::layer()
              .with_ansi(false)
              .with_timer(timer)
              .with_writer(writer),
          ) as BoxedLayer)
        },
        ..default()
      }),
  );
  // P2.7：渲染诊断（Bevy 0.19 非默认装配，仅 tracing-tracy feature 才自动加）——
  // 装配后 DiagnosticsStore 才存在：左上角折线图推真实 GPU 帧时（gate_frame span），
  // GATE_BENCH=1 时另加逐秒均值打印与 logs/gpu_frame.log。
  // 未装配时 gate-render 的 span 走 Option<&T> no-op，不影响渲染。
  app.add_plugins(bevy::render::diagnostic::RenderDiagnosticsPlugin);
  if bench {
    app.add_plugins(bevy::diagnostic::LogDiagnosticsPlugin::default());
  }
  app
    .add_plugins(gate_render::GateRenderPlugin)
    .add_plugins(gate_ui::GateUiPlugin)
    // 【诊断】GATE_BENCH=1：后台/失焦窗口也用 Continuous 更新（Bevy 默认失焦切
    // reactive_low_power 60Hz，后台跑帧时 fps.log 会被 60 封顶，无法比较 trace 变体）
    .insert_resource(bevy::winit::WinitSettings {
      focused_mode: bevy::winit::UpdateMode::Continuous,
      unfocused_mode: if bench {
        bevy::winit::UpdateMode::Continuous
      } else {
        bevy::winit::UpdateMode::reactive_low_power(std::time::Duration::from_secs_f64(1.0 / 60.0))
      },
      ..Default::default()
    })
    .add_systems(Startup, setup)
    .add_systems(
      Update,
      (
        orbit_camera_input,
        // Shift+左键探针点查（必须在 left_click_pick_recenter 之前拦截左键）
        probe_click_inspect
          .after(orbit_camera_input)
          .before(left_click_pick_recenter),
        left_click_pick_recenter.after(orbit_camera_input),
        demo_ui_setup,
        // 先推帧时长样本，gate-ui 的 plot_redraw_system 同帧再重绘折线图
        fps_line_feed.before(gate_ui::plot_redraw_system),
        // 右上角组件展示窗：交互事件日志 / slider 实时值 / 演示折线喂数
        showcase_demo_system,
        // F3 切换左上角 debug overlay 显隐（默认显示）
        debug_overlay_toggle,
      ),
    );
  // 【诊断】GATE_BENCH=1：逐帧记录 GPU pass 时间到 logs/gpu_frame.log
  // （elapsed,wall_ms,trace_gpu_ms,ddgi_gpu_ms），定位帧时波动是 GPU 还是 CPU/present
  if bench {
    app.add_systems(Update, gpu_frame_log);
  }
  app.run();
}

/// 主题就绪后 spawn 一次（RON 成功或回退默认都会插入 UiTheme 资源）
///
/// one-shot 语义 = **存在性守卫**：spawn 的根节点挂 [`DemoUiRoot`]，查到即跳过，
/// 无平行 done 状态。命令 apply 后 marker 当帧生效，守卫最迟下一帧命中。
///
/// 等待条件：主题字体资产已加载为 `LoadState::Loaded`。若 theme.font_path = None
/// 则用 SystemUi fallback（不等待，直接 spawn）。必须等字体到位再 spawn UI 标签，
/// 否则 TextPipeline 在 FiraMono-subset（Bevy 内置 default slot，不含 CJK）上生成
/// 字形缓存，即使之后字体 override default slot，已缓存的 atlas 条目仍回退 CJK
/// 字形为方框（parley 的字形请求不因为 Assets slot 变化而自动失效）。
/// FPS 行纯 ASCII，但统一遵守字体就绪判定，避免以后换字符串时又出方框。
fn demo_ui_setup(
  theme: Option<Res<UiTheme>>,
  font: Option<Res<ThemeFont>>,
  server: Option<Res<AssetServer>>,
  mut commands: Commands,
  q_spawned: Query<(), With<DemoUiRoot>>,
) {
  if !q_spawned.is_empty() {
    return;
  }
  let Some(theme) = theme else {
    return;
  };
  let font = match (server.as_ref(), font.as_ref(), theme.font_path.as_ref()) {
    (Some(srv), Some(f), Some(_)) => match &f.handle {
      Some(h) if matches!(srv.load_state(h.id()), LoadState::Loaded) => Some(h.clone()),
      _ => return,
    },
    (_, _, None) => None,
    _ => return,
  };
  let theme = theme.clone();

  commands.queue(move |world: &mut World| {
    let ctx = UiCtx::new(&theme, font.as_ref());
    // 左上角调试 overlay + 右上角组件展示窗：同一主题上下文，一次性生成
    spawn_debug_view(world, &ctx);
    spawn_showcase(world, &ctx);
  });
}
