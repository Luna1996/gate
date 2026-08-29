use bevy::{asset::AssetPlugin, image::Image, prelude::*, window::Window};

use gate_render::{GradientImages, GradientUniforms, VIEW_SIZE, create_gradient_image};

/// 以 crate 目录为锚的 assets 路径，F5 / 终端启动行为一致
pub const ASSETS_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/assets");

fn main() {
  App::new()
    .add_plugins(
      DefaultPlugins
        .set(WindowPlugin {
          primary_window: Some(Window {
            resolution: VIEW_SIZE.into(),
            ..default()
          }),
          ..default()
        })
        .set(AssetPlugin {
          file_path: ASSETS_PATH.into(),
          ..default()
        }),
    )
    .add_plugins(gate_render::GateRenderPlugin)
    .add_systems(Startup, setup)
    .run();
}

fn setup(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
  let handle = create_gradient_image(&mut images);

  // 相机提供 ViewTarget（blit 直写目标）；Sprite 中转已移除（阶段 B）
  // Msaa::Off：ViewTarget 非 MSAA，blit 管线采样数匹配（体素光追管线不用 MSAA）
  commands.spawn((Camera2d, Msaa::Off));

  commands.insert_resource(GradientImages { target: handle });
  commands.insert_resource(GradientUniforms {
    size: VIEW_SIZE.as_vec2().extend(0.0).extend(0.0),
  });
}
