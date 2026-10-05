mod camera;
mod config;
mod consts;
mod debug_menu;
mod edit;
mod height_field;
mod infinite_cubes;
mod mc;
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
  MouseLock, apply_mouse_lock, build_camera_config, camera_look_input, fly_camera_input,
  free_look_input, left_click_pick_recenter, orbit_camera_input, spawn_crosshair,
  sync_camera_mode_switch, sync_crosshair, toggle_mouse_lock,
};
use config::{Config, save_config_on_exit};
use debug_menu::{
  DebugUiRoot, FpsOverlayVisible, FpsWindow, camera_info_tick, debug_menu_toggle, fps_overlay_tick,
  lod_state_tick, spawn_debug_menu_ui, sync_edit_menu, sync_sky_menu, sync_ui_locale,
  sync_video_menu,
};
use edit::voxel_edit_input;
use scene::setup;
use showcase::{showcase_demo_system, spawn_showcase};

rust_i18n::i18n!("../assets/locales", fallback = "zh-CN");

pub const DEFAULT_LOCALE: &str = "zh-CN";

pub fn assets_dir() -> std::path::PathBuf {
  gate_render::assets_dir()
}

pub fn log_path() -> std::path::PathBuf {
  gate_render::logs_dir().join("latest.log")
}

fn main() {
  #[cfg(feature = "profile")]
  let _tracy_client = tracy_client::Client::start();

  rust_i18n::set_locale(DEFAULT_LOCALE);

  let bench = consts::BENCH_UNFOCUSED || consts::bench();
  let mut app = App::new();
  app.add_plugins(
    DefaultPlugins
      .set(WindowPlugin {
        primary_window: Some(Window {
          resolution: WindowResolution::new(VIEW_SIZE.x, VIEW_SIZE.y)
            .with_scale_factor_override(1.0),
          focused: true,
          present_mode: PresentMode::Fifo,
          resizable: true,
          ..default()
        }),
        ..default()
      })
      .set(AssetPlugin { file_path: assets_dir().to_string_lossy().into_owned(), ..default() })
      .set(LogPlugin {
        filter: std::env::var("GATE_LOG")
          .unwrap_or_else(|_| consts::DEFAULT_LOG_FILTER.to_string()),
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
      .set(bevy::render::RenderPlugin { render_creation: profile_render_creation(), ..default() }),
  );
  app
    .add_plugins(gate_render::GateRenderPlugin)
    .add_plugins(gate_ui::GateUiPlugin)
    .init_resource::<edit::EditSettings>()
    .init_resource::<MouseLock>()
    .init_resource::<height_field::MaterialDisplaceCache>()
    .init_resource::<infinite_cubes::Streaming>()
    .init_resource::<FpsOverlayVisible>()
    .init_resource::<FpsWindow>()
    .insert_resource(Config::load())
    .insert_resource(gate_ui::UiTranslator::new(|key| t!(key).to_string()))
    .insert_resource(bevy::winit::WinitSettings {
      focused_mode: bevy::winit::UpdateMode::Continuous,
      unfocused_mode: if bench {
        bevy::winit::UpdateMode::Continuous
      } else {
        bevy::winit::UpdateMode::reactive_low_power(std::time::Duration::from_secs_f64(1.0 / 60.0))
      },
    })
    .add_systems(Startup, (setup, spawn_crosshair))
    .add_systems(
      Update,
      (
        (
          sync_camera_mode_switch,
          toggle_mouse_lock,
          apply_mouse_lock,
          camera_look_input,
          free_look_input,
          orbit_camera_input,
          fly_camera_input,
          left_click_pick_recenter,
          build_camera_config,
          voxel_edit_input,
        )
          .chain(),
        sync_crosshair,
        debug_ui_setup,
        (fps_overlay_tick, camera_info_tick, lod_state_tick),
        sync_sky_menu,
        sync_edit_menu,
        sync_video_menu,
        sync_ui_locale,
        showcase_demo_system,
        debug_menu_toggle,
        enforce_integer_scale_factor,
      ),
    )
    .add_systems(Last, save_config_on_exit.after(bevy::window::ExitSystems));
  #[cfg(feature = "profile")]
  app.add_systems(Update, tracy_frame_mark);
  app.add_systems(Update, infinite_cubes::stream_chunks);
  if consts::EDIT_SELFTEST {
    app.add_systems(Update, edit::edit_selftest);
  }
  if consts::AUTO_ORBIT || consts::bench_orbit() {
    app.add_systems(Update, camera::auto_orbit_system);
  }
  if consts::bench_fly() {
    app.add_systems(
      Update,
      camera::auto_fly_system.after(camera::fly_camera_input).before(camera::build_camera_config),
    );
  }
  app.run();
}

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

#[cfg(feature = "profile")]
fn tracy_frame_mark() {
  if let Some(client) = tracy_client::Client::running() {
    client.frame_mark();
  }
}

fn enforce_integer_scale_factor(mut q: Query<&mut Window, With<PrimaryWindow>>) {
  let Ok(mut w) = q.single_mut() else {
    return;
  };
  if w.resolution.scale_factor_override() != Some(1.0) {
    w.resolution.set_scale_factor_override(Some(1.0));
  }
}

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
    let ctx = UiCtx::new(&theme, font.as_ref()).with_icon_font(icon_font.as_ref());
    world.spawn((Name::new("debug-ui-root"), DebugUiRoot));
    spawn_debug_menu_ui(world, &ctx);
    spawn_showcase(world, &ctx);
  });
}
