use bevy::prelude::*;

pub const SHADOW_BIAS: f32 = 0.5;
pub const SHADOW_DIR_T_MAX: f32 = 8192.0;
pub const EMISSIVE_EMIT_GAIN: f32 = 4.0;

pub const MIN_DIM: u32 = 64;
pub const MAX_DIM: u32 = 4096;
pub const RENDER_SCALE: u32 = 1;

pub const REPORT_PERIOD_SECS: f32 = 2.0;

pub const GI_GAIN: f32 = 1.0;

pub const VIEW_SIZE: UVec2 =
  UVec2::new(crate::brickmap::consts::VIEW_W, crate::brickmap::consts::VIEW_H);

pub const MINECRAFT_SKY: [f32; 3] = [120.0 / 255.0, 167.0 / 255.0, 1.0];
