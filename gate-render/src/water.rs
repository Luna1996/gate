use bevy::prelude::*;
use bevy::render::extract_resource::ExtractResource;
use bevy::render::render_resource::ShaderType;
use glam::{Vec2, Vec3, Vec4};

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, ShaderType)]
pub struct WaterUniform {
  pub origin: Vec4,
  pub params: Vec4,
  pub color: Vec4,
  pub flags: Vec4,
  pub pool: Vec4,
  pub fall: Vec4,
}

#[derive(Clone, Copy, Debug)]
pub struct WaterPool {
  pub min: Vec2,
  pub max: Vec2,
  pub level: f32,
  pub fall_x: f32,
  pub fall_z: Vec2,
}

#[derive(Resource, Clone, Copy, Debug, ExtractResource)]
#[extract_app(bevy::render::RenderApp)]
pub struct WaterSettings {
  pub enabled: bool,
  pub level: f32,
  pub deep: Vec3,
  pub absorption: f32,
  pub wave_amp: f32,
  pub wave_scale: f32,
  pub sparkle: f32,
  pub shade_cell: f32,
  pub time: f32,
  pub pool: Option<WaterPool>,
}

impl Default for WaterSettings {
  fn default() -> Self {
    Self {
      enabled: false,
      level: 0.0,
      deep: Vec3::new(0.006, 0.022, 0.032),
      absorption: 0.02,
      wave_amp: 0.06,
      wave_scale: 0.18,
      sparkle: 2.0,
      shade_cell: 1.0,
      time: 0.0,
      pool: None,
    }
  }
}

impl WaterSettings {
  pub fn uniform(&self) -> WaterUniform {
    let (pool, fall, pool_level, pool_on) = if let Some(p) = self.pool {
      (
        Vec4::new(p.min.x, p.min.y, p.max.x, p.max.y),
        Vec4::new(p.fall_x, p.fall_z.x, p.fall_z.y, 0.0),
        p.level,
        1.0,
      )
    } else {
      (Vec4::ZERO, Vec4::ZERO, 0.0, 0.0)
    };
    WaterUniform {
      origin: Vec4::new(0.0, 0.0, 1.0, self.level),
      params: Vec4::new(self.wave_amp, self.wave_scale, self.sparkle, self.time),
      color: Vec4::new(self.deep.x, self.deep.y, self.deep.z, self.absorption),
      flags: Vec4::new(
        if self.enabled { 1.0 } else { 0.0 },
        self.shade_cell.max(1.0),
        pool_level,
        pool_on,
      ),
      pool,
      fall,
    }
  }
}
