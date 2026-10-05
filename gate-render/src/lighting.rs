use bevy::ecs::resource::Resource;
use bevy::render::render_resource::ShaderType;
use glam::{Vec3, Vec4};
use serde::Deserialize;

pub const MAX_LIGHTS: usize = 8;

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, PartialEq, ShaderType)]
pub struct LightDesc {
  pub kind_pos_dir: Vec4,
  pub color_intensity: Vec4,
  pub shape: Vec4,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, PartialEq, ShaderType)]
pub struct LightGlobals {
  pub count: u32,
  pub refl_tier: u32,
  pub refl_nest: u32,
  pub base_flags: u32,
  pub ambient: Vec4,
  pub exposure_pad: Vec4,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, Resource, ShaderType)]
pub struct LightPoolUniform {
  pub g: LightGlobals,
  pub lights: [LightDesc; MAX_LIGHTS],
  pub sky_color: Vec4,
}

#[derive(
  Resource, Clone, Copy, Debug, PartialEq, bevy::render::extract_resource::ExtractResource,
)]
#[extract_app(bevy::render::RenderApp)]
pub struct ReflectionSettings {
  pub tier: u32,
  pub nest: u32,
}

impl ReflectionSettings {
  pub const TIERS: u32 = 4;
  pub const NEST_CHOICES: [u32; 4] = [0, 1, 2, 4];

  pub fn tier(&self) -> u32 {
    self.tier.min(Self::TIERS - 1)
  }

  pub fn nest(&self) -> u32 {
    *Self::NEST_CHOICES.iter().min_by_key(|c| c.abs_diff(self.nest)).expect("NEST_CHOICES 非空")
  }
}

impl Default for ReflectionSettings {
  fn default() -> Self {
    Self { tier: 2, nest: 0 }
  }
}

#[derive(
  Resource, Clone, Copy, Debug, PartialEq, bevy::render::extract_resource::ExtractResource,
)]
#[extract_app(bevy::render::RenderApp)]
pub struct BaseSettings {
  pub shadow: bool,
  pub implicit_normal: bool,
  pub vol_tint: bool,
}

impl BaseSettings {
  pub const FLAG_SHADOW: u32 = 1 << 0;
  pub const FLAG_IMPLICIT_NORMAL: u32 = 1 << 1;
  pub const FLAG_VOL_TINT: u32 = 1 << 2;

  pub fn flags(&self) -> u32 {
    (if self.shadow { Self::FLAG_SHADOW } else { 0 })
      | (if self.implicit_normal { Self::FLAG_IMPLICIT_NORMAL } else { 0 })
      | (if self.vol_tint { Self::FLAG_VOL_TINT } else { 0 })
  }
}

impl Default for BaseSettings {
  fn default() -> Self {
    Self { shadow: true, implicit_normal: true, vol_tint: false }
  }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct DirLightCfg {
  pub dir: [f32; 3],
  #[serde(default = "default_angular_radius")]
  pub angular_radius_deg: f32,
  pub color: [f32; 3],
  pub intensity: f32,
  #[serde(default = "one")]
  pub disk_scale: f32,
}
fn default_angular_radius() -> f32 {
  0.0
}
fn one() -> f32 {
  1.0
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct SkyCfg {
  pub color: [f32; 3],
}

#[derive(
  Debug, Clone, PartialEq, Resource, Deserialize, bevy::render::extract_resource::ExtractResource,
)]
#[extract_app(bevy::render::RenderApp)]
pub struct LightingTheme {
  pub sun: Option<DirLightCfg>,
  pub ambient: [f32; 3],
  pub exposure: f32,
  pub sky: Option<SkyCfg>,
}

impl Default for LightingTheme {
  fn default() -> Self {
    Self {
      sun: Some(DirLightCfg {
        dir: Vec3::new(0.5, -0.8, 0.3).normalize().to_array(),
        angular_radius_deg: 0.0,
        color: [1.0, 0.96, 0.88],
        intensity: 0.8,
        disk_scale: 1.0,
      }),
      ambient: [0.08, 0.09, 0.12],
      exposure: 1.0,
      sky: Some(SkyCfg { color: crate::consts::MINECRAFT_SKY }),
    }
  }
}

pub fn parse_lighting_ron(src: &str) -> Result<LightingTheme, ron::error::SpannedError> {
  ron::de::from_str(src)
}

pub fn build_light_pool(theme: &LightingTheme) -> LightPoolUniform {
  let mut u = LightPoolUniform {
    g: LightGlobals {
      count: 0,
      refl_tier: 0,
      refl_nest: 0,
      base_flags: 0,
      ambient: Vec4::new(theme.ambient[0], theme.ambient[1], theme.ambient[2], 0.0),
      exposure_pad: Vec4::new(theme.exposure, 0.0, 0.0, 0.0),
    },
    lights: [const {
      LightDesc { kind_pos_dir: Vec4::ZERO, color_intensity: Vec4::ZERO, shape: Vec4::ZERO }
    }; MAX_LIGHTS],
    sky_color: Vec4::new(
      crate::consts::MINECRAFT_SKY[0],
      crate::consts::MINECRAFT_SKY[1],
      crate::consts::MINECRAFT_SKY[2],
      0.0,
    ),
  };
  if let Some(sky) = &theme.sky {
    u.sky_color = Vec4::new(sky.color[0], sky.color[1], sky.color[2], 0.0);
  }
  if let Some(sun) = &theme.sun {
    let l = -Vec3::from(sun.dir).normalize_or_zero();
    u.lights[0] = LightDesc {
      kind_pos_dir: Vec4::new(0.0, l.x, l.y, l.z),
      color_intensity: Vec4::new(sun.color[0], sun.color[1], sun.color[2], sun.intensity),
      shape: Vec4::new(sun.disk_scale.max(0.0), 0.0, 0.0, 0.0),
    };
    u.g.count = 1;
  }
  u
}
