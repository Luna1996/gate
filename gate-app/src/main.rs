//! gate 演示应用入口：插件装配 + 系统注册。
//!
//! 代码按功能模块拆分，本文件只保留装配与一次性 UI 编排：
//! - [`scene`]：Startup 场景搭建（vox/demo 场景 + 相机初始机位 + brickmap 诊断）
//! - [`camera`]：轨道相机输入、左键拾取 recenter、调试开关（V 可见性缓存）
//! - [`debug_overlay`]：左上角 FPS 读数 HUD，F3 切换显隐（默认显示）
//! - [`showcase`]：右上角 gate-ui 组件展示窗（交互观察者 + 演示折线喂数）
//! - [`vox_scene`]：MagicaVoxel .vox 场景导入
//!
//! 性能剖析：`cargo run --features profile` 后用 Tracy GUI 连接（CPU zones +
//! wgpu-profiler GPU zones 同时间线）；详见 gate-render profiler 模块。

mod camera;
mod debug_overlay;
mod edit;
mod scene;
mod showcase;
#[cfg(feature = "profile")]
mod tracy_layer;
mod vox_scene;

use bevy::{
  asset::{AssetPlugin, LoadState},
  log::LogPlugin,
  prelude::*,
  window::{PresentMode, PrimaryWindow, Window, WindowResolution},
};

use gate_render::VIEW_SIZE;
use gate_ui::{ThemeFont, UiCtx, UiTheme};

use camera::{
  build_camera_config, camera_look_input, fly_camera_input, left_click_pick_recenter,
  orbit_camera_input, sync_camera_mode_switch,
};
use debug_overlay::{DemoUiRoot, debug_overlay_toggle, fps_line_feed, spawn_debug_view};
use edit::voxel_edit_input;
use scene::setup;
use showcase::{showcase_demo_system, spawn_showcase};

/// 以 crate 目录为锚的 assets 路径，F5 / 终端启动行为一致
pub const ASSETS_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/assets");

/// 程序侧日志落盘（方案 3）：LogPlugin custom_layer 追加无色文件层，
/// 每次启动截断重写（永远最新一轮）；默认 stderr 彩色层保留不变
pub const LOG_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/logs/latest.log");

fn main() {
  // 【剖析】profile feature：main 最早期启动 Tracy 客户端——wgpu-profiler 的
  // new_with_tracy_client（RenderStartup）与 TracyLayer 都要求 Client 已 running。
  // guard 保活到 main 结束（drop 时 flush 残留 zone）。Tracy GUI 未连接时
  // zone 被丢弃，不阻塞、不报错。
  #[cfg(feature = "profile")]
  let _tracy_client = tracy_client::Client::start();

  // 【诊断】GATE_BENCH=1：静默后台（窗口不可见）+ vsync Fifo
  let bench = std::env::var("GATE_BENCH").as_deref() == Ok("1");
  let mut app = App::new();
  app.add_plugins(
    DefaultPlugins
      .set(WindowPlugin {
        primary_window: Some(Window {
          // scale_factor_override=1.0：强制 1 逻辑像素 = 1 物理像素，避免 OS 显示缩放
          // （125%/150%）产生非整数物理像素，导致 UI 1px 边框抗锯齿成 2px、文字模糊。
          // 3D 渲染不受影响（DDA 目标固定 VIEW_SIZE，blit 到窗口）。
          resolution: WindowResolution::new(VIEW_SIZE.x, VIEW_SIZE.y)
            .with_scale_factor_override(1.0),
          // 默认 Fifo 硬垂直同步（与 DebugView「VSync」开关默认开同步；vblank 墙钟
          // 节拍稳定帧时，GATE_BENCH=1 帧时测量也用同一模式）。开关关闭 → AutoNoVsync
          // 不封顶测裸 GPU 吞吐；bevy_render 检测 present_mode 变化自动重配 swapchain。
          // focused=false：启动不抢前台焦点（静默后台）
          focused: false,
          present_mode: PresentMode::Fifo,
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
        filter: "info,wgpu=debug,wgpu_core=debug,\
          wgpu_hal::vulkan::instance=off,\
          wgpu_hal::vulkan::surface=off"
          .into(),
        // 附加层：无色文件层（logs/latest.log；stderr 彩色层不受影响）；
        // profile feature 再加 TracyLayer（bevy trace span → Tracy CPU zone）
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
          let file_layer = tracing_subscriber::fmt::layer()
            .with_ansi(false)
            .with_timer(timer)
            .with_writer(writer);
          #[cfg(feature = "profile")]
          {
            use tracing_subscriber::Layer as _;
            Some(Box::new(crate::tracy_layer::TracyLayer.and_then(file_layer)) as BoxedLayer)
          }
          #[cfg(not(feature = "profile"))]
          {
            Some(Box::new(file_layer) as BoxedLayer)
          }
        },
        ..default()
      })
      // 【剖析】profile feature：WgpuSettings 加 wgpu timestamp 特性（wgpu-profiler
      // GPU zone 必需；DX12/Vulkan 全支持；缺特性时 scope 静默空转不报错）。
      // Bevy 0.19 的 WgpuSettings 不再是 Resource，经 RenderPlugin.render_creation 注入。
      .set(bevy::render::RenderPlugin {
        render_creation: profile_render_creation(),
        ..default()
      }),
  );
  app
    .add_plugins(gate_render::GateRenderPlugin)
    .add_plugins(gate_ui::GateUiPlugin)
    // 体素编辑设置（形状/大小/材质/材质→调色板槽缓存）；DebugView 的 Edit tab 是它的视图
    .init_resource::<edit::EditSettings>()
    // 【诊断】GATE_BENCH=1：后台/失焦窗口也用 Continuous 更新（Bevy 默认失焦切
    // reactive_low_power 60Hz，后台跑帧时 fps 会被 60 封顶）
    .insert_resource(bevy::winit::WinitSettings {
      focused_mode: bevy::winit::UpdateMode::Continuous,
      unfocused_mode: if bench {
        bevy::winit::UpdateMode::Continuous
      } else {
        bevy::winit::UpdateMode::reactive_low_power(std::time::Duration::from_secs_f64(1.0 / 60.0))
      },
    })
    .add_systems(Startup, setup)
    .add_systems(
      Update,
      (
        // 相机链：模式切换对齐 → 转头（共享）→ 两套模式各自输入 → 左键拾取 → 唯一矩阵构造点
        // → 体素编辑（要读本帧的 cfg，所以排在矩阵构造之后）。
        // 必须严格串行：它们都读写 OrbitCamera/FlyCamera/DdaCameraConfig，且 build_camera_config
        // 需要拿到同帧的输入结果（旧实现把矩阵重建放在 orbit_camera_input 末尾，拆出来后才有多模式）。
        (
          sync_camera_mode_switch,
          camera_look_input,
          orbit_camera_input,
          fly_camera_input,
          left_click_pick_recenter,
          build_camera_config,
          voxel_edit_input,
        )
          .chain(),
        demo_ui_setup,
        // 先推帧时长样本，gate-ui 的 plot_redraw_system 同帧再重绘折线图
        fps_line_feed.before(gate_ui::plot_redraw_system),
        // 右上角组件展示窗：交互事件日志 / slider 实时值 / 演示折线喂数
        showcase_demo_system,
        // F3 切换左上角 debug overlay 显隐（默认显示）
        debug_overlay_toggle,
        // 每帧强制 scale_factor=1.0：resize/换显示器时 winit 会重设 OS DPI 值，
        // override 可能被覆盖；重设保证 UI 1:1 物理像素、不随 DPI 缩放
        enforce_integer_scale_factor,
      ),
    );
  // 【剖析】profile feature：主世界每帧一个 Tracy frame mark（CPU/GPU zone 归帧）
  #[cfg(feature = "profile")]
  app.add_systems(Update, tracy_frame_mark);
  // 【诊断】GATE_EDIT_SELFTEST=1：无鼠标输入地走通一次"编辑 → 增量上传"链路
  if std::env::var("GATE_EDIT_SELFTEST").as_deref() == Ok("1") {
    app.add_systems(Update, edit::edit_selftest);
  }
  // 【诊断】GATE_ORBIT=1：相机自动绕目标转 → 复现"相机移动中"的帧时问题（配 GATE_BENCH=1 读逐 pass）
  if std::env::var("GATE_ORBIT").as_deref() == Ok("1") {
    app.add_systems(Update, camera::auto_orbit_system);
  }
  app.run();
}

/// profile 构建：WgpuSettings 开 wgpu timestamp 特性（wgpu-profiler GPU zone 必需）
/// 包进 RenderCreation 供 RenderPlugin 使用；非 profile 构建：默认设置。
#[cfg(feature = "profile")]
fn profile_render_creation() -> bevy::render::settings::RenderCreation {
  bevy::render::settings::WgpuSettings {
    features: gate_render::profiler::timestamp_wgpu_features(),
    ..default()
  }
  .into()
}

#[cfg(not(feature = "profile"))]
fn profile_render_creation() -> bevy::render::settings::RenderCreation {
  bevy::render::settings::RenderCreation::default()
}

/// 每帧 Tracy frame mark（Tracy 主时间线的帧边界；GPU zone 同样归帧）。
#[cfg(feature = "profile")]
fn tracy_frame_mark() {
  if let Some(client) = tracy_client::Client::running() {
    client.frame_mark();
  }
}

/// 每帧强制主窗口 scale_factor_override = 1.0。
///
/// winit 在窗口 resize / 跨显示器移动时会触发 `ScaleFactorChanged`，把
/// `Window.scale_factor` 刷成 OS DPI 值（125%/150% → 1.25/1.5）。每帧重设
/// override 保证 1 逻辑像素 = 1 物理像素，UI 不随系统 DPI 缩放、边框/文字不糊。
fn enforce_integer_scale_factor(mut q: Query<&mut Window, With<PrimaryWindow>>) {
  let Ok(mut w) = q.single_mut() else {
    return;
  };
  if w.resolution.scale_factor_override() != Some(1.0) {
    w.resolution.set_scale_factor_override(Some(1.0));
  }
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
