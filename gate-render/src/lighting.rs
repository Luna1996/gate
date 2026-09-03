//! P3.1 光源系统：光源 wire 契约 + 数据驱动主题 + CPU 参考着色。
//!
//! 严格对齐 Douglas Dwyer devlog #02 / #19 方案：
//! - 方向光硬阴影（命中点向太阳投 1 条射线，不通即阴影）
//! - sky 渐变环境光（按法线 y 混合天顶/地平线）
//! - 发光体素：radiance 直出（albedo × emissive，无方向性、不受阴影）
//! - 无点光源、无 Phong 高光、无软阴影锥采样
//!
//! Uniform 布局（WGSL `LightPool` 逐字段镜像）：
//!   LightGlobals(48B) + 8×LightDesc(384B) + sky_top(16B) + sky_horizon(16B) = 464B
//!   当前只填 lights[0] = 方向光；其余槽保留为 0 供未来扩展。

use bevy::ecs::resource::Resource;
use bevy::render::render_resource::ShaderType;
use glam::{Vec3, Vec4};
use serde::Deserialize;

use crate::brickmap::wire::BrickMapBuffers;
use crate::brickmap::{VolumeHit, cpu_reference_volumes_occluded};
use gate_voxel::VolumeTransform;

/// Vec4 的 yzw 分量
#[inline]
pub(crate) fn yzw(v: Vec4) -> Vec3 {
  Vec3::new(v.y, v.z, v.w)
}
/// Vec4 的 xyz 分量
#[inline]
pub(crate) fn xyz(v: Vec4) -> Vec3 {
  Vec3::new(v.x, v.y, v.z)
}

/// 光源上限（uniform 数组长度；当前只用 lights[0] = 方向光）
pub const MAX_LIGHTS: usize = 8;
/// 阴影射线起点沿法线偏移（fine），消除自遮挡 acne
pub const SHADOW_BIAS: f32 = 0.5;
/// 方向光阴影射线 t_max：覆盖整个可见场景
pub const SHADOW_DIR_T_MAX: f32 = 65536.0;
/// 发光体素 radiance 直出增益
pub const EMISSIVE_EMIT_GAIN: f32 = 4.0;

/// 光源描述（shader 镜像，48B；uniform 数组 stride 16 的倍数 ✓）
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, ShaderType)]
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
#[derive(Debug, Clone, Copy, PartialEq, ShaderType)]
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
#[derive(Debug, Clone, Copy, Resource, ShaderType)]
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
#[derive(Debug, Clone, PartialEq, Resource, Deserialize)]
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

// ============================================================================
// CPU 参考着色（WGSL 光照段逐字翻译）
// ============================================================================

/// 命中点材质（palette 两 words 解包）：
/// albedo（u8 → /255）+ roughness（w0>>24）+ emissive（w1 低 8bit）
///
/// Phase 3 统一：`vols[0]` = 主世界（obj_id=-1），`vols[1..N]` = 物体（obj_id=0..N-1）。
/// `hit.obj_id` 决定从哪个 volume 的 `b_palette` 取色。
fn hit_mat(vols: &[(&BrickMapBuffers, VolumeTransform)], hit: &VolumeHit) -> (Vec3, f32, f32) {
  let idx = if hit.obj_id == -1 { 0 } else { hit.obj_id as usize + 1 };
  let pal_buf = &vols[idx].0.b_palette;
  let w0 = pal_buf[hit.pal as usize * 2];
  let w1 = pal_buf[hit.pal as usize * 2 + 1];
  let albedo = Vec3::new(
    (w0 & 0xFF) as f32,
    ((w0 >> 8) & 0xFF) as f32,
    ((w0 >> 16) & 0xFF) as f32,
  ) / 255.0;
  let rough = ((w0 >> 24) & 0xFF) as f32 / 255.0;
  let emissive = (w1 & 0xFF) as f32 / 255.0;
  (albedo, rough, emissive)
}

// ============================================================================
// CPU 参考 sky() 函数（与 WGSL 逐字镜像）
// ============================================================================

/// WGSL sky() 的 CPU 镜像——渐变 + 太阳盘。
pub fn cpu_reference_sky(dir: Vec3, pool: &LightPoolUniform) -> Vec3 {
  let d = dir.normalize_or_zero();
  let h = d.y.clamp(0.0, 1.0);
  let t = {
    let x = (h / 0.35).clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
  };
  let col = xyz(pool.sky_horizon).lerp(xyz(pool.sky_top), t);
  if pool.g.count > 0 && pool.lights[0].kind_pos_dir.x < 0.5 {
    let sdir = yzw(pool.lights[0].kind_pos_dir);
    let cos_a = d.dot(sdir).max(0.0);
    let sun_c = xyz(pool.lights[0].color_intensity) * pool.lights[0].color_intensity.w;
    let glow = cos_a.max(0.0).powf(64.0) * 0.05 * if h > 0.0 { 1.0 } else { 0.0 };
    col + sun_c * glow
  } else {
    col
  }
}

// ============================================================================
// CPU 参考着色：逐字镜像 WGSL shade_hit
// ============================================================================

/// Douglas 基础光影：方向光硬阴影 + sky 渐变环境光 + 发光体素 radiance 直出。
/// 无点光源、无 Phong 高光、无软阴影锥采样。
///
/// Phase 3 统一：`vols[0]` = 主世界（identity transform），`vols[1..N]` = 物体。
/// 阴影射线 `cpu_reference_volumes_occluded` 遍历所有 volume。
pub fn cpu_reference_shade_hit(
  vols: &[(&BrickMapBuffers, VolumeTransform)],
  light_pool: &LightPoolUniform,
  origin: Vec3,
  dir: Vec3,
  hit: VolumeHit,
  _shadow_t_max: f32,
) -> Vec3 {
  let p = origin + dir * hit.t;
  let n = hit.normal;
  let (base, _rough, emissive) = hit_mat(vols, &hit);

  // sky 渐变环境光
  let h = n.y.clamp(0.0, 1.0);
  let t_sky = {
    let x = (h / 0.35).clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
  };
  let sky_grad = xyz(light_pool.sky_horizon).lerp(xyz(light_pool.sky_top), t_sky);
  let mut col = base * (xyz(light_pool.g.ambient) * 0.4 + sky_grad * 0.6);

  // 方向光硬阴影（Douglas devlog #02 方案：1 条射线，不通即阴影）
  if light_pool.g.count > 0 {
    let ld = &light_pool.lights[0];
    if ld.kind_pos_dir.x < 0.5 {
      let l_axis = yzw(ld.kind_pos_dir);
      let ndl = n.dot(l_axis).max(0.0);
      if ndl > 0.0 {
        let o = p + n * SHADOW_BIAS;
        let vis = if !cpu_reference_volumes_occluded(vols, o, l_axis, SHADOW_DIR_T_MAX) {
          1.0
        } else {
          0.0
        };
        let c = xyz(ld.color_intensity) * ld.color_intensity.w;
        col += base * c * (ndl * vis);
      }
    }
  }

  // 发光体素 radiance 直出（Devlog #19 radiance 分支 3 的直接光版本）
  col += base * (emissive * EMISSIVE_EMIT_GAIN);
  col * light_pool.g.exposure_pad.x
}

// ============================================================================
// 单测
// ============================================================================

#[cfg(test)]
mod tests {
  use super::*;
  use crate::brickmap::{BrickMapBuilder, cpu_reference_trace_volumes};
  use gate_voxel::{VolumeGrid, fill_bricks};
  use glam::{IVec3, Mat3};

  fn world_box(ext: i32, pal: u8) -> BrickMapBuffers {
    let mut g = VolumeGrid::new();
    g.palette_mut().get_mut(pal).color = [64, 64, 64];
    fill_bricks(&mut g, IVec3::ZERO, IVec3::splat(ext), 16, pal);
    BrickMapBuilder::build_full(&g).buffers().clone()
  }

  const ALBEDO: f32 = 64.0 / 255.0;

  /// 默认主题 = 1 个方向光
  #[test]
  fn default_theme_pools_layout() {
    let pool = build_light_pool(&LightingTheme::default());
    assert_eq!(pool.g.count, 1);
    let sun = &pool.lights[0];
    assert_eq!(sun.kind_pos_dir.x, 0.0);
    let l = yzw(sun.kind_pos_dir);
    assert!((l.length() - 1.0).abs() < 1e-5);
    assert!((sun.shape.x - 0.0).abs() < 1e-6, "硬阴影：角半径 = 0");
    assert_eq!(pool.g.ambient.x, LightingTheme::default().ambient[0]);
    assert_eq!(pool.g.exposure_pad.x, LightingTheme::default().exposure);
  }

  /// RON 解析 + 点光源主题不支持（silently ignored）
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
    assert!(parse_lighting_ron("( sun: (").is_err());
  }

  /// 垂直向下太阳硬阴影：顶面直射（vis=1），底面仅环境项
  #[test]
  fn shade_direct_and_occluded_extremes() {
    let world = world_box(512, 3);
    let theme = LightingTheme {
      sun: Some(DirLightCfg {
        dir: [0.0, -1.0, 0.0],
        angular_radius_deg: 0.0,
        color: [1.0, 1.0, 1.0],
        intensity: 2.0,
      }),
      ambient: [0.1, 0.1, 0.1],
      exposure: 1.0,
      sky: None,
    };
    let lp = build_light_pool(&theme);
    let vols: Vec<(&BrickMapBuffers, VolumeTransform)> = vec![(&world, VolumeTransform::IDENTITY)];

    // 顶面命中：N·L = 1，vis = 1（无遮挡）
    let hit = cpu_reference_trace_volumes(
      &vols,
      Vec3::new(256.0, 640.0, 256.0),
      -Vec3::Y,
      4096.0,
    )
    .expect("顶面必有命中");
    assert_eq!(hit.obj_id, -1);
    let rgb = cpu_reference_shade_hit(
      &vols,
      &lp,
      Vec3::new(256.0, 640.0, 256.0),
      -Vec3::Y,
      hit,
      4096.0,
    );
    // 硬阴影：ambient(0.1) + sky_none(0.1) → base*(0.04 + 0.06) = ALBEDO*0.1 + ALBEDO*2.0*1.0
    let expect = ALBEDO * 0.1 + ALBEDO * 2.0;
    assert!(
      (rgb.x - expect).abs() < 1e-4,
      "顶面 rgb={rgb:?} expect≈{expect}"
    );

    // 底面命中：N·L = -1 → 直射 0，仅环境项
    let hit = cpu_reference_trace_volumes(
      &vols,
      Vec3::new(256.0, -128.0, 256.0),
      Vec3::Y,
      4096.0,
    )
    .expect("底面必有命中");
    let rgb = cpu_reference_shade_hit(
      &vols,
      &lp,
      Vec3::new(256.0, -128.0, 256.0),
      Vec3::Y,
      hit,
      4096.0,
    );
    assert!((rgb.x - ALBEDO * 0.1).abs() < 1e-4, "底面 rgb={rgb:?}");
  }

  /// 物体遮挡太阳：地面命中点在物体正下方 → vis=0 → 仅环境项
  #[test]
  fn obj_object_casts_shadow_on_ground() {
    let world = world_box(512, 3);
    let theme = LightingTheme {
      sun: Some(DirLightCfg {
        dir: [0.0, -1.0, 0.0],
        angular_radius_deg: 0.0,
        color: [1.0, 1.0, 1.0],
        intensity: 2.0,
      }),
      ambient: [0.1, 0.1, 0.1],
      exposure: 1.0,
      sky: None,
    };
    let lp = build_light_pool(&theme);

    let mut og = VolumeGrid::new();
    og.palette_mut().get_mut(5).color = [64, 64, 64];
    fill_bricks(&mut og, IVec3::ZERO, IVec3::splat(64), 16, 5);
    let obj_bufs = BrickMapBuilder::build_full(&og).buffers().clone();
    let obj_tr = VolumeTransform::new(Vec3::new(240.0, 592.0, 240.0), Mat3::IDENTITY, 1.0);
    let vols_with: Vec<(&BrickMapBuffers, VolumeTransform)> = vec![
      (&world, VolumeTransform::IDENTITY),
      (&obj_bufs, obj_tr),
    ];
    let vols_empty: Vec<(&BrickMapBuffers, VolumeTransform)> = vec![(&world, VolumeTransform::IDENTITY)];

    let hit = VolumeHit {
      t: 384.0,
      pal: 3,
      obj_id: -1,
      normal: Vec3::Y,
    };
    let origin = Vec3::new(256.0, 896.0, 256.0);

    let lit = cpu_reference_shade_hit(&vols_empty, &lp, origin, -Vec3::Y, hit, 4096.0);
    let shadowed = cpu_reference_shade_hit(&vols_with, &lp, origin, -Vec3::Y, hit, 4096.0);
    assert!(
      lit.x > shadowed.x * 10.0,
      "有物体应显著更暗：lit={lit:?} shadowed={shadowed:?}"
    );
  }

  /// 发光体素 radiance 直出：无光源时 emissive 体素面也亮
  #[test]
  fn shade_emissive_direct_glow() {
    let mut g = VolumeGrid::new();
    {
      let pal = g.palette_mut();
      let mut e = gate_voxel::PaletteEntry::default();
      e.color = [255, 160, 40];
      e.emissive = 200;
      pal.set(3, e);
    }
    fill_bricks(&mut g, IVec3::ZERO, IVec3::splat(512), 16, 3);
    let world = BrickMapBuilder::build_full(&g).buffers().clone();
    let theme = LightingTheme {
      sun: None,
      ambient: [0.0, 0.0, 0.0],
      exposure: 1.0,
      sky: None,
    };
    let lp = build_light_pool(&theme);
    let vols: Vec<(&BrickMapBuffers, VolumeTransform)> = vec![(&world, VolumeTransform::IDENTITY)];

    let hit = cpu_reference_trace_volumes(
      &vols,
      Vec3::new(256.0, 640.0, 256.0),
      -Vec3::Y,
      4096.0,
    )
    .expect("顶面命中");
    let rgb = cpu_reference_shade_hit(
      &vols,
      &lp,
      Vec3::new(256.0, 640.0, 256.0),
      -Vec3::Y,
      hit,
      4096.0,
    );
    let expect = (255.0 / 255.0) * (200.0 / 255.0) * EMISSIVE_EMIT_GAIN;
    assert!(
      (rgb.x - expect).abs() < 1e-4,
      "直出 rgb={rgb:?} expect={expect}"
    );

    // 底面直出同值（radiance 无方向性）
    let hit = cpu_reference_trace_volumes(
      &vols,
      Vec3::new(256.0, -128.0, 256.0),
      Vec3::Y,
      4096.0,
    )
    .expect("底面命中");
    let rgb = cpu_reference_shade_hit(
      &vols,
      &lp,
      Vec3::new(256.0, -128.0, 256.0),
      Vec3::Y,
      hit,
      4096.0,
    );
    assert!((rgb.x - expect).abs() < 1e-4, "底面直出");
  }
}
