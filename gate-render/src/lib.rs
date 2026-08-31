pub mod brickmap;
mod gradient;
mod responsive;

use bevy::prelude::*;

pub use brickmap::{
    BindingLimits, BrickMapBuffers, BrickMapBuilder, BrickMapGlobals, BrickMapUploadPlugin,
    DirtyRanges, GpuMovPool, MovDesc, MovGlobals, MovHit, MovObject, MovPlugin, MovPoolPacked,
    MovScene, OBJ_WORLD,
    BrickMapView, BufferLayout, BuilderMirror, DdaCameraConfig, DdaImages, DdaViewUniform,
    GpuBrickMap, OrbitCamera, TileUpdate, UploadBudget, UploadCpuSample, UploadCpuSampleChannel,
    UploadSnapshot, VoxelScene, cpu_dda_ascii_grid_32x32, cpu_reference_dda_ray,
    cpu_reference_dda_ray_aabb_skip, cpu_reference_object_ray, cpu_reference_trace_scene,
    create_dda_image, pack_mov_pool,
};
pub use gradient::{
    GradientImages, GradientUniforms, RenderScale, VIEW_SIZE, create_gradient_image,
};
pub use responsive::{ResponsivePlugin, resize_render_targets};

pub struct GateRenderPlugin;

impl Plugin for GateRenderPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            gradient::GradientPlugin,
            BrickMapUploadPlugin,
            brickmap::dda::BrickMapDdaPlugin,
            brickmap::mov::MovPlugin,
        ));
    }
}
