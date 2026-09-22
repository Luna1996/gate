//! gate 演示应用入口：插件装配 + 系统注册；`--features profile` 用 Tracy GUI 剖析（CPU 与 GPU zone 同时间线）。
//! 模块：`scene` 场景搭建 / `camera` 相机与输入 / `debug_menu` 左上角调试菜单（gate-ui menu 容器）/
//! `showcase` 右上角组件展示窗 / `vox_scene` MagicaVoxel .vox 导入。

mod camera;
mod config;
mod consts;
mod debug_menu;
mod edit;
mod height_field;
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
use rust_i18n::t;

use gate_render::VIEW_SIZE;
use gate_ui::{ThemeFont, UiCtx, UiTheme};

use camera::{
  build_camera_config, camera_look_input, fly_camera_input, left_click_pick_recenter,
  orbit_camera_input, sync_camera_mode_switch,
};
use config::{Config, save_config_on_exit};
use debug_menu::{
  DebugUiRoot, FpsOverlayVisible, FpsWindow, camera_info_tick, debug_menu_toggle, fps_overlay_tick,
  spawn_debug_menu_ui, sync_edit_menu, sync_sky_menu, sync_ui_locale, sync_video_menu,
};
use edit::voxel_edit_input;
use scene::setup;
use showcase::{showcase_demo_system, spawn_showcase};

// i18n 文案表：编译期把 assets/locales/*.yml codegen 进二进制（`t!` 查表零 IO）。
// fallback = 缺省语言（缺键回落中文）；加语言 = assets/locales/<locale>.yml + Cargo.toml available-locales。
rust_i18n::i18n!("../assets/locales", fallback = "zh-CN");

/// 缺省语言（`rust_i18n::set_locale` 的入参）
pub const DEFAULT_LOCALE: &str = "zh-CN";

/// 只读资源根（Bevy `AssetPlugin::file_path`）：开发时 = 源码树 `assets/`，
/// 便携发布时 = exe 同目录 `assets/`（见 `gate_render::paths`）
pub fn assets_dir() -> std::path::PathBuf {
  gate_render::assets_dir()
}

/// 日志文件路径（可写目录，与资源分离）：<安装根>/logs/latest.log，
/// 每次启动截断重写；stderr 彩色层不受影响
pub fn log_path() -> std::path::PathBuf {
  gate_render::logs_dir().join("latest.log")
}

fn main() {
  // profile feature：Tracy 客户端必须在最早期启动，new_with_tracy_client 与 TracyLayer 都要求已 running；
  // guard 保活到 main 结束（drop 时 flush 残留 zone）。
  #[cfg(feature = "profile")]
  let _tracy_client = tracy_client::Client::start();

  // i18n：把当前语言定成缺省中文；必须在任何 `t!` 求值之前（UI 文本只生成一次，之后不重算）。
  rust_i18n::set_locale(DEFAULT_LOCALE);

  // 失焦后台跑帧开关见 `consts::BENCH_UNFOCUSED`
  let bench = consts::BENCH_UNFOCUSED;
  let mut app = App::new();
  app.add_plugins(
    DefaultPlugins
      .set(WindowPlugin {
        primary_window: Some(Window {
          // scale_factor_override=1.0：强制 1 逻辑像素 = 1 物理像素。
          resolution: WindowResolution::new(VIEW_SIZE.x, VIEW_SIZE.y)
            .with_scale_factor_override(1.0),
          // 启动即聚焦：未聚焦走下面的 `reactive_low_power(1/60)` ⇒ 整个 app 被帽在 60Hz
          // （看性能数据必须保持聚焦）。
          focused: true,
          present_mode: PresentMode::Fifo,
          resizable: true, // resize 后渲染目标/aspect 由响应式系统跟随
          ..default()
        }),
        ..default()
      })
      .set(AssetPlugin {
        // 运行期解析（非编译期常量）：便携发布时指向 exe 同目录的 assets/
        file_path: assets_dir().to_string_lossy().into_owned(),
        ..default()
      })
      .set(LogPlugin {
        // info 基线 + 定向屏蔽 wgpu_hal::vulkan 的 instance/surface 层 VUID 报错
        filter: "info,wgpu=debug,wgpu_core=debug,\
          wgpu_hal::vulkan::instance=off,\
          wgpu_hal::vulkan::surface=off"
          .into(),
        // 附加层：无色文件层写 <安装根>/logs/latest.log（stderr 彩色层不受影响）；
        // profile feature 再叠加 TracyLayer（tracing span → Tracy CPU zone）
        custom_layer: |_app| {
          use bevy::log::BoxedLayer;
          let path = log_path();
          std::fs::create_dir_all(path.parent().expect("log_path 必有父目录")).ok()?;
          let file = std::fs::File::create(&path).ok()?;
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
      // profile feature：经 RenderPlugin.render_creation 注入 WgpuSettings，开 wgpu timestamp 特性
      // （wgpu-profiler GPU zone 必需）。
      .set(bevy::render::RenderPlugin { render_creation: profile_render_creation(), ..default() }),
  );
  app
    .add_plugins(gate_render::GateRenderPlugin)
    .add_plugins(gate_ui::GateUiPlugin)
    // 体素编辑设置（形状/大小/材质）；DebugMenu 的「游戏/编辑」是它的视图
    .init_resource::<edit::EditSettings>()
    // MT8-5：笔触位移按材质 id 缓存的高度场（首次落笔解码 ≈22ms，之后解码耗时 0）
    .init_resource::<height_field::MaterialDisplaceCache>()
    // DebugMenu 相关：FPS 覆盖层显隐 + 1s 帧时长滚动窗口
    .init_resource::<FpsOverlayVisible>()
    .init_resource::<FpsWindow>()
    // 跨启动配置（<安装根>/data/config.toml）：菜单窗口/控件值 + 相机姿态；读一次供各处共享
    .insert_resource(Config::load())
    // UI 文案解析器：菜单树存 i18n key，gate-ui 经它解析（切语言后 sync_ui_locale 触发重解析）
    .insert_resource(gate_ui::UiTranslator::new(|key| t!(key).to_string()))
    // 失焦窗口也用 Continuous 更新（默认失焦为 reactive_low_power 60Hz）
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
        // 相机链顺序固定：模式切换对齐 → 转头 → 各自输入 → 左键拾取 → build_camera_config → 体素编辑。
        // 必须严格串行：都读写 OrbitCamera/FlyCamera/DdaCameraConfig；体素编辑读本帧 cfg，故在矩阵构造之后。
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
        debug_ui_setup,
        // 右上角 FPS 覆盖层（开关打开时每帧刷新）+ 纯文本行的相机信息
        (fps_overlay_tick, camera_info_tick),
        // 「天空」页：自动流逝推进的时刻回写进菜单（面板显示的就是画面里的时刻）
        sync_sky_menu,
        // 「编辑」页：PBR 变体开着时把材质控件整行置灰（参数全由资产/贴图决定）
        sync_edit_menu,
        // 「视频」页：像素大小 ≠ 1 时把「抗锯齿」整行置灰（那些档位下不启用 FXAA，开关值保留）
        sync_video_menu,
        // 语言切换 → 菜单/UI 文案整体重解析
        sync_ui_locale,
        // 右上角组件展示窗：交互事件日志 / slider 实时值 / 演示折线喂数
        showcase_demo_system,
        // F3 切换 DebugMenu 显隐（默认显示）
        debug_menu_toggle,
        // 每帧强制 scale_factor=1.0：winit 在 resize/换显示器时会重设 OS DPI 值并覆盖 override
        enforce_integer_scale_factor,
      ),
    )
    // 退出前落盘：全部持久化设置写进 <安装根>/data/config.toml（菜单窗口/控件值 + 相机姿态）；
    // 须排在 bevy_window 的 ExitSystems 之后（AppExit 由它写入）。
    .add_systems(Last, save_config_on_exit.after(bevy::window::ExitSystems));
  // profile feature：主世界每帧一个 Tracy frame mark（CPU/GPU zone 归帧）
  #[cfg(feature = "profile")]
  app.add_systems(Update, tracy_frame_mark);
  // 无鼠标输入地走通一次"编辑 → 增量上传"链路
  if consts::EDIT_SELFTEST {
    app.add_systems(Update, edit::edit_selftest);
  }
  // 相机自动绕目标旋转并平移（配 BENCH_UNFOCUSED 读移动中的逐 pass 帧时）
  if consts::AUTO_ORBIT {
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

/// 每帧强制主窗口 scale_factor_override = 1.0（winit 在 resize/跨显示器时会用 OS DPI 值覆盖它）。
fn enforce_integer_scale_factor(mut q: Query<&mut Window, With<PrimaryWindow>>) {
  let Ok(mut w) = q.single_mut() else {
    return;
  };
  if w.resolution.scale_factor_override() != Some(1.0) {
    w.resolution.set_scale_factor_override(Some(1.0));
  }
}

/// 主题就绪后 spawn 一次（`UiTheme` 存在即跳过，根节点挂 `DebugUiRoot` 作守卫）。
/// 字体资产须已 `LoadState::Loaded` 才 spawn；图标字体同理。
fn debug_ui_setup(
  theme: Option<Res<UiTheme>>,
  font: Option<Res<ThemeFont>>,
  icon: Option<Res<gate_ui::IconFont>>,
  server: Option<Res<AssetServer>>,
  mut commands: Commands,
  q_spawned: Query<(), With<DebugUiRoot>>,
) {
  if !q_spawned.is_empty() {
    return;
  }
  let Some(theme) = theme else {
    return;
  };
  let loaded = |h: &Option<Handle<Font>>| match (server.as_ref(), h) {
    (Some(srv), Some(h)) => matches!(srv.load_state(h.id()), LoadState::Loaded),
    (None, _) => true,
    _ => false,
  };
  let font = match (font.as_ref(), theme.font_path.as_ref()) {
    (Some(f), Some(_)) => {
      if !loaded(&f.handle) {
        return;
      }
      f.handle.clone()
    }
    (_, None) => None,
    _ => return,
  };
  let icon_font = match (icon.as_ref(), theme.icon_font_path.as_ref()) {
    (Some(i), Some(_)) => {
      if !loaded(&i.handle) {
        return;
      }
      i.handle.clone()
    }
    _ => None,
  };
  let theme = theme.clone();

  commands.queue(move |world: &mut World| {
    // 图标字体挂进 ctx：菜单标题栏按钮与子菜单箭头用 FA 字形
    let ctx = UiCtx::new(&theme, font.as_ref()).with_icon_font(icon_font.as_ref());
    // 左上角调试菜单 + 右上角组件展示窗：同一主题上下文，一次性生成
    world.spawn((Name::new("debug-ui-root"), DebugUiRoot));
    spawn_debug_menu_ui(world, &ctx);
    spawn_showcase(world, &ctx);
  });
}
