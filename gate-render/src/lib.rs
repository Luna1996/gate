mod gradient;

use bevy::prelude::*;

pub use gradient::{GradientImages, GradientUniforms, VIEW_SIZE, create_gradient_image};

pub struct GateRenderPlugin;

impl Plugin for GateRenderPlugin {
  fn build(&self, app: &mut App) {
    app.add_plugins(gradient::GradientPlugin);
  }
}
