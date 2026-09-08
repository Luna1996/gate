//! 光源 wire 契约 + 数据驱动主题。
//!
//! 严格对齐 Douglas Dwyer devlog #02 / #19 方案：
//! - 方向光硬阴影（命中点向太阳投 1 条射线，不通即阴影）
//! - sky 纯色环境光（Minecraft 白天天空蓝 #78A7FF）
//! - 发光体素：radiance 直出（albedo × emissive，无方向性、不受阴影）
//! - 无点光源、无 Phong 高光、无软阴影锥采样
//!
//! 光照数学全部在 GPU（dda.wgsl）；CPU 侧只做数据打包，不做任何光照计算。
//!
//! Uniform 布局（WGSL `LightPool` 逐字段镜像）：
//!   LightGlobals(48B) + 8×LightDesc(384B) + sky_color(16B) = 448B
//!   当前只填 lights[0] = 方向光；其余槽保留为 0 供未来扩展。

use bevy::ecs::resource::Resource;
use bevy::render::render_resource::ShaderType;
use glam::{Vec3, Vec4};
use serde::Deserialize;

/// 光源上限（uniform 数组长度；当前只用 lights[0] = 方向光）
pub const MAX_LIGHTS: usize = 8;
/// 阴影射线起点沿法线偏移（fine），消除自遮挡 acne
pub const SHADOW_BIAS: f32 = 0.5;
/// 方向光阴影射线 t_max：场景 AABB 对角 ≈3118（[-256,-512,-256]~[1536,1536,1280]），
/// 表面点沿任意方向的遮挡必在其内；65536 的空气段让每条阴影射线多空走 8×（性能）。
/// 改世界尺度（GATE_TILES）时按对角线同步放大。
pub const SHADOW_DIR_T_MAX: f32 = 8192.0;
/// 发光体素 radiance 直出增益
pub const EMISSIVE_EMIT_GAIN: f32 = 4.0;
/// 天空纯色（Minecraft 白天平原天空 #78A7FF；sRGB u8/255 直读——
/// sRGB→linear 转换在 WGSL sky_rgb() 内做，CPU 侧不碰光照数学）
pub const MINECRAFT_SKY: [f32; 3] = [120.0 / 255.0, 167.0 / 255.0, 1.0];

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
      sky: Some(SkyCfg {
        color: MINECRAFT_SKY,
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
    sky_color: Vec4::new(MINECRAFT_SKY[0], MINECRAFT_SKY[1], MINECRAFT_SKY[2], 0.0),
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

// ============================================================================
// 单测
// ============================================================================

#[cfg(test)]
mod tests {
  use super::*;

  /// 默认主题 = 1 个方向光 + Minecraft 纯色天空
  #[test]
  fn default_theme_pools_layout() {
    let pool = build_light_pool(&LightingTheme::default());
    assert_eq!(pool.g.count, 1);
    let sun = &pool.lights[0];
    assert_eq!(sun.kind_pos_dir.x, 0.0);
    let l = Vec3::new(sun.kind_pos_dir.y, sun.kind_pos_dir.z, sun.kind_pos_dir.w);
    assert!((l.length() - 1.0).abs() < 1e-5);
    assert!((sun.shape.x - 0.0).abs() < 1e-6, "硬阴影：角半径 = 0");
    assert_eq!(pool.g.ambient.x, LightingTheme::default().ambient[0]);
    assert_eq!(pool.g.exposure_pad.x, LightingTheme::default().exposure);
    assert_eq!(pool.sky_color.x, MINECRAFT_SKY[0]);
  }

  /// RON 解析：未知字段（points 等）忽略，旧单色天空格式不再兼容 top/horizon
  #[test]
  fn parse_ron_minimal_and_error() {
    let src = r#"(
            sun: Some((dir: (0.0, -1.0, 0.0), angular_radius_deg: 1.0,
                  color: (1.0, 1.0, 1.0), intensity: 2.0)),
            ambient: (0.01, 0.01, 0.01),
            exposure: 1.5,
        )"#;
    let t = parse_lighting_ron(src).expect("valid ron");
    assert!(t.sun.is_some());
    assert_eq!(t.exposure, 1.5);
    let pool = build_light_pool(&t);
    assert_eq!(pool.g.count, 1);
    assert_eq!(pool.g.exposure_pad.x, 1.5);
    // 无 sky 字段 → 回退 Minecraft 纯色
    assert_eq!(pool.sky_color.x, MINECRAFT_SKY[0]);
    assert!(parse_lighting_ron("( sun: (").is_err());
  }
}
