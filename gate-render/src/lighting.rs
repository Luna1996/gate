//! 光源 wire 契约 + 数据驱动主题（方向光硬阴影 / sky 纯色环境光 / 发光体素直出）；CPU 侧只做数据打包。
//! Uniform 布局（WGSL `LightPool` 逐字段镜像）：LightGlobals(48B) + 8×LightDesc(384B) + sky_color(16B)。

use bevy::ecs::resource::Resource;
use bevy::render::render_resource::ShaderType;
use glam::{Vec3, Vec4};
use serde::Deserialize;

/// 光源上限（uniform 数组长度；当前只用 lights[0] = 方向光）
pub const MAX_LIGHTS: usize = 8;

/// 光源描述（shader 镜像，48B；uniform 数组 stride 16 的倍数 ✓）
#[repr(C)]
#[derive(Debug, Default, Clone, Copy, PartialEq, ShaderType)]
pub struct LightDesc {
  /// x = kind（0 = 方向光）；yzw = L 轴（指向光，已归一）
  pub kind_pos_dir: Vec4,
  /// rgb = 线性色，w = 强度
  pub color_intensity: Vec4,
  /// x = 方向光盘角半径（rad）；yzw reserved
  pub shape: Vec4,
}

/// 光池 header（48B）：count + 环境色 + 曝光
#[repr(C)]
#[derive(Debug, Default, Clone, Copy, PartialEq, ShaderType)]
pub struct LightGlobals {
  pub count: u32,
  pub _pad0: u32,
  pub _pad1: u32,
  pub _pad2: u32,
  /// rgb = 环境色（线性），w reserved
  pub ambient: Vec4,
  /// x = 曝光系数；yzw reserved
  pub exposure_pad: Vec4,
}

/// BG3 uniform 整体，WGSL `LightPool` 逐字段镜像。
#[repr(C)]
#[derive(Debug, Default, Clone, Copy, Resource, ShaderType)]
pub struct LightPoolUniform {
  pub g: LightGlobals,
  pub lights: [LightDesc; MAX_LIGHTS],
  /// 天空纯色（miss 背景与 sky 环境光共用）
  pub sky_color: Vec4,
}

/// 方向光配置（主题资产）
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct DirLightCfg {
  /// 光传播方向（指向场景）；打包时翻转为 L（指向光）
  pub dir: [f32; 3],
  /// 太阳盘角半径（rad）——硬阴影管线不使用此值
  #[serde(default = "default_angular_radius")]
  pub angular_radius_deg: f32,
  pub color: [f32; 3],
  pub intensity: f32,
}
fn default_angular_radius() -> f32 {
  0.0
}

/// 天空颜色配置（纯色）
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct SkyCfg {
  pub color: [f32; 3],
}

/// 光照主题（`assets/lighting/*.ron`）：方向光 + 环境 + 天空 + 曝光
/// ExtractResource：main world 资源自动提取进 render world（BG3 光池数据源）
#[derive(
  Debug, Clone, PartialEq, Resource, Deserialize, bevy::render::extract_resource::ExtractResource,
)]
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
      }),
      ambient: [0.08, 0.09, 0.12],
      exposure: 1.0,
      sky: Some(SkyCfg { color: crate::consts::MINECRAFT_SKY }),
    }
  }
}

/// RON 解析
pub fn parse_lighting_ron(src: &str) -> Result<LightingTheme, ron::error::SpannedError> {
  ron::de::from_str(src)
}

/// 构建光池 uniform：方向光（lights[0]）+ 天空 + 环境 + 曝光
pub fn build_light_pool(theme: &LightingTheme) -> LightPoolUniform {
  let mut u = LightPoolUniform {
    g: LightGlobals {
      count: 0,
      _pad0: 0,
      _pad1: 0,
      _pad2: 0,
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
      shape: Vec4::new(0.0, 0.0, 0.0, 0.0), // 硬阴影：角半径 = 0
    };
    u.g.count = 1;
  }
  u
}
