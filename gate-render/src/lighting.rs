//! 光源系统：光源 wire 契约 + 数据驱动主题（RON → LightPoolUniform）。
//!
//! Uniform 布局（WGSL `LightPool` 逐字段镜像）：
//!   LightGlobals(48B) + 8×LightDesc(384B) + sky_top(16B) + sky_horizon(16B) = 464B
//!   当前只填 lights[0] = 方向光；其余槽保留为 0 供未来扩展。

use bevy::ecs::resource::Resource;
use bevy::render::render_resource::ShaderType;
use glam::{Vec3, Vec4};
use serde::Deserialize;

/// 光源上限（uniform 数组长度；当前只用 lights[0] = 方向光）
pub const MAX_LIGHTS: usize = 8;
/// 阴影射线起点沿法线偏移（voxel），消除自遮挡 acne
pub const SHADOW_BIAS: f32 = 0.5;
/// 方向光阴影射线 t_max：场景 AABB 对角 ≈3118（[-256,-512,-256]~[1536,1536,1280]），
/// 表面点沿任意方向的遮挡必在其内；65536 的空气段让每条阴影射线多空走 8×（性能）。
/// 改世界尺度（GATE_TILES）时按对角线同步放大。
pub const SHADOW_DIR_T_MAX: f32 = 8192.0;
/// 发光体素 radiance 直出增益
pub const EMISSIVE_EMIT_GAIN: f32 = 4.0;

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
  /// 天空天顶色（线性，用于 sky() 渐变）
  pub sky_top: Vec4,
  /// 天空地平线色（线性）
  pub sky_horizon: Vec4,
}

// ============================================================================
// 数据驱动主题
// ============================================================================

/// 方向光配置（主题资产）
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct DirLightCfg {
  /// 光传播方向（指向场景）；打包时翻转为 L（指向光）
  pub dir: [f32; 3],
  /// 太阳盘角半径（rad）——保留但 Douglas 硬阴影方案不使用
  #[serde(default = "default_angular_radius")]
  pub angular_radius_deg: f32,
  pub color: [f32; 3],
  pub intensity: f32,
}
fn default_angular_radius() -> f32 {
  0.0
}

/// 天空颜色配置
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct SkyCfg {
  pub top: [f32; 3],
  pub horizon: [f32; 3],
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
      sky: Some(SkyCfg {
        top: [0.45, 0.55, 0.85],
        horizon: [0.95, 0.82, 0.65],
      }),
    }
  }
}

/// RON 解析
pub fn parse_lighting_ron(src: &str) -> Result<LightingTheme, ron::error::SpannedError> {
  ron::de::from_str(src)
}

/// 构建光池 uniform：方向光（lights[0]）+ 天空 + 环境 + 曝光
/// 当前只处理方向光，不做点光源/发光体素 NEE（Douglas 方案不支持）。
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
      LightDesc {
        kind_pos_dir: Vec4::ZERO,
        color_intensity: Vec4::ZERO,
        shape: Vec4::ZERO,
      }
    }; MAX_LIGHTS],
    sky_top: Vec4::new(theme.ambient[0], theme.ambient[1], theme.ambient[2], 0.0),
    sky_horizon: Vec4::new(theme.ambient[0], theme.ambient[1], theme.ambient[2], 0.0),
  };
  if let Some(sky) = &theme.sky {
    u.sky_top = Vec4::new(sky.top[0], sky.top[1], sky.top[2], 0.0);
    u.sky_horizon = Vec4::new(sky.horizon[0], sky.horizon[1], sky.horizon[2], 0.0);
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