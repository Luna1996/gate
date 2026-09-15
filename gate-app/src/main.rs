//! gate 演示应用入口：插件装配 + 系统注册。
//!
//! 模块：[`scene`] 场景搭建 / [`camera`] 相机与输入 / [`debug_overlay`] 左上角 FPS HUD /
//! [`showcase`] 右上角组件展示窗 / [`vox_scene`] MagicaVoxel .vox 导入。
//! 性能剖析：`--features profile` 启动后用 Tracy GUI 连接（CPU 与 wgpu-profiler GPU zone 同时间线）。

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

// i18n 文案表：编译期把 gate-app/locales/*.yml codegen 进二进制（`t!` 查表零 IO）。
// fallback = 缺省语言：某语言缺键时回落中文，不会把裸 key 显示到 UI。
// 加语言 = locales/<locale>.yml + Cargo.toml 的 available-locales；运行期切语言 = rust_i18n::set_locale。
rust_i18n::i18n!("locales", fallback = "zh-CN");

/// 缺省语言（`rust_i18n::set_locale` 的入参）
pub const DEFAULT_LOCALE: &str = "zh-CN";

/// 以 crate 目录为锚的 assets 路径，F5 / 终端启动行为一致
pub const ASSETS_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/assets");

/// 日志文件路径：LogPlugin custom_layer 追加的无色文件层在此落盘，
/// 每次启动截断重写；stderr 彩色层不受影响
pub const LOG_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/logs/latest.log");

fn main() {
  // profile feature：Tracy 客户端必须在最早期启动 —— wgpu-profiler 的
  // new_with_tracy_client（RenderStartup）与 TracyLayer 都要求 Client 已 running；
  // guard 保活到 main 结束（drop 时 flush 残留 zone）。
  #[cfg(feature = "profile")]
  let _tracy_client = tracy_client::Client::start();

  // i18n：把当前语言定成缺省中文。必须在任何 `t!` 求值之前 —— UI 由 demo_ui_setup
  // 启动时一次性 spawn，文本生成后不再重算。
  rust_i18n::set_locale(DEFAULT_LOCALE);

  // GATE_BENCH=1：失焦后台跑帧（配合窗口 focused=false 与 Fifo）
  let bench = std::env::var("GATE_BENCH").as_deref() == Ok("1");
  let mut app = App::new();
  app.add_plugins(
    DefaultPlugins
      .set(WindowPlugin {
        primary_window: Some(Window {
          // scale_factor_override=1.0：强制 1 逻辑像素 = 1 物理像素，避免 OS 显示缩放
          // （125%/150%）下 UI 1px 边框抗锯齿成 2px、文字模糊；3D 不受影响
          // （DDA 目标固定 VIEW_SIZE，再 blit 到窗口）。
          resolution: WindowResolution::new(VIEW_SIZE.x, VIEW_SIZE.y)
            .with_scale_factor_override(1.0),
          // Fifo 硬垂直同步（与 DebugView「VSync」开关默认开一致）；关闭 → AutoNoVsync
          // 不封顶测裸 GPU 吞吐，bevy_render 检测 present_mode 变化后重配 swapchain。
          // focused=false：启动不抢前台焦点。
          focused: false,
          present_mode: PresentMode::Fifo,
          resizable: true, // resize 后渲染目标与 aspect 由响应式系统跟随
          ..default()
        }),
        ..default()
      })
      .set(AssetPlugin { file_path: ASSETS_PATH.into(), ..default() })
      .set(LogPlugin {
        // info 基线 + 定向屏蔽：wgpu_hal::vulkan 的 instance / surface 层会打印
        // wgpu 已知 bug 的 VUID 错误（仅首 1-2 帧 swapchain 时序异常，不影响画面正确性）；
        // 其余 wgpu/winit/bevy_render 错误照常打印。
        filter: "info,wgpu=debug,wgpu_core=debug,\
          wgpu_hal::vulkan::instance=off,\
          wgpu_hal::vulkan::surface=off"
          .into(),
        // 附加层：无色文件层写 logs/latest.log（stderr 彩色层不受影响）；
        // profile feature 再叠加 TracyLayer（tracing span → Tracy CPU zone）
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
          let file_layer =
            tracing_subscriber::fmt::layer().with_ansi(false).with_timer(timer).with_writer(writer);
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
      // profile feature：WgpuSettings 开 wgpu timestamp 特性（wgpu-profiler 的
      // GPU zone 必需，DX12/Vulkan 均支持，缺特性时 scope 静默空转）。
      // Bevy 0.19 的 WgpuSettings 不再是 Resource，须经 RenderPlugin.render_creation 注入。
      .set(bevy::render::RenderPlugin { render_creation: profile_render_creation(), ..default() }),
  );
  app
    .add_plugins(gate_render::GateRenderPlugin)
    .add_plugins(gate_ui::GateUiPlugin)
    // 体素编辑设置（形状/大小/材质/材质→调色板槽缓存）；DebugView 的 Edit tab 是它的视图
    .init_resource::<edit::EditSettings>()
    // GATE_BENCH=1：失焦窗口也用 Continuous 更新（Bevy 默认失焦切 reactive_low_power
    // 60Hz，后台跑帧时 fps 会被封顶到 60）
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
        // 相机链：模式切换对齐 → 转头（两模式共享）→ 各自输入 → 左键拾取 →
        // 矩阵构造（build_camera_config）→ 体素编辑。
        // 必须严格串行：都读写 OrbitCamera/FlyCamera/DdaCameraConfig，且 build_camera_config
        // 需要同帧的输入结果；体素编辑读本帧 cfg，故排在矩阵构造之后。
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
        // 每帧强制 scale_factor=1.0：resize/换显示器时 winit 会重设 OS DPI 值并覆盖
        // override，重设保证 UI 恒为 1 物理像素/逻辑像素
        enforce_integer_scale_factor,
      ),
    );
  // profile feature：主世界每帧一个 Tracy frame mark（CPU/GPU zone 归帧）
  #[cfg(feature = "profile")]
  app.add_systems(Update, tracy_frame_mark);
  // GATE_EDIT_SELFTEST=1：无鼠标输入地走通一次"编辑 → 增量上传"链路
  if std::env::var("GATE_EDIT_SELFTEST").as_deref() == Ok("1") {
    app.add_systems(Update, edit::edit_selftest);
  }
  // GATE_ORBIT=1：相机自动绕目标旋转并平移（配 GATE_BENCH=1 读逐 pass 帧时）
  if std::env::var("GATE_ORBIT").as_deref() == Ok("1") {
    app.add_systems(Update, camera::auto_orbit_system);
  }
  app.run();
}

/// profile 构建：WgpuSettings 开 wgpu timestamp 特性（wgpu-profiler GPU zone 必需），
/// 包进 RenderCreation 供 RenderPlugin 使用。
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
/// winit 在 resize / 跨显示器移动时会触发 `ScaleFactorChanged`，把
/// `Window.scale_factor` 刷成 OS DPI 值（125%/150% → 1.25/1.5）；重设 override
/// 保证 1 逻辑像素 = 1 物理像素。
fn enforce_integer_scale_factor(mut q: Query<&mut Window, With<PrimaryWindow>>) {
  let Ok(mut w) = q.single_mut() else {
    return;
  };
  if w.resolution.scale_factor_override() != Some(1.0) {
    w.resolution.set_scale_factor_override(Some(1.0));
  }
}

/// 主题就绪后 spawn 一次（RON 成功或回退默认都会插入 UiTheme 资源）。
///
/// one-shot = 存在性守卫：spawn 的根节点挂 [`DemoUiRoot`]，查到即跳过；命令 apply
/// 后 marker 当帧生效，守卫最迟下一帧命中。
///
/// 等待条件：`theme.font_path` 非 None 时字体资产须已 `LoadState::Loaded`，否则
/// TextPipeline 会在不含 CJK 的默认 slot 上生成字形缓存，之后即使 override default
/// slot，已缓存的 atlas 条目光栅化仍是方框。
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
