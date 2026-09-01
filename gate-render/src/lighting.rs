//! P3.1 光源系统：光源 wire 契约 + 数据驱动主题 + CPU 参考着色（NEE 直射光）。
//!
//! 结构对齐 brickmap 惯例：本模块是 CPU 侧与 dda.wgsl 光照段之间的字节/数学契约。
//! - `LightPoolUniform`（BG3 uniform，432B）：header（count/ambient/exposure）+ 8×光源描述
//! - `LightingTheme`：RON 数据资产（光源/环境/曝光），默认内置「暗色实验室」
//! - `cpu_reference_shade_hit` / `cpu_reference_scene_occluded`：WGSL 着色段逐字翻译的源，
//!   等价性由数值单测锁定（锥/球采样、遮挡两端值、直射合成公式）
//!
//! 软阴影采样（v3.2）：方向光 = 太阳盘角半径锥内固定 2 采样；点光 = 球面黄金螺旋
//! 固定 2 采样（立体角采样）。确定性采样 → 无帧间闪烁；P9 PT 为其完全体。

use bevy::ecs::resource::Resource;
use bevy::render::render_resource::ShaderType;
use glam::{IVec3, Vec3, Vec4};
use serde::Deserialize;

use crate::brickmap::cpu_reference_scene_occluded;
use crate::brickmap::mov::OBJ_WORLD;
use crate::brickmap::wire::BrickMapBuffers;
use crate::brickmap::{MovHit, MovPoolPacked};

/// Vec4 的 yzw 分量（glam 0.32 无 swizzle 方法，手写展开；WGSL 侧直接 .yzw）
#[inline]
fn yzw(v: Vec4) -> Vec3 {
  Vec3::new(v.y, v.z, v.w)
}
/// Vec4 的 xyz 分量
#[inline]
fn xyz(v: Vec4) -> Vec3 {
  Vec3::new(v.x, v.y, v.z)
}

/// 光源上限（uniform 预算：48B header + 16×48B = 816B）
/// P3.2：8→16——主题静态光源占前几槽，剩余槽给发光元件点光（按到眼睛距离截断）
pub const MAX_LIGHTS: usize = 16;
/// 阴影采样数（锥/球各 2 个固定偏移；确定性、无帧间闪烁）
pub const SHADOW_SAMPLES: usize = 2;
/// 阴影射线起点沿法线偏移（fine），消除自遮挡 acne
pub const SHADOW_BIAS: f32 = 0.5;
/// fine → 米（0.25cm/fine）；点光衰减用米制（intensity 语义 = 米制）
pub const FINES_PER_M: f32 = 400.0;
/// 发光体素 → NEE 点光强度映射：emissive u8/255 × GAIN（米制平方反比系数）。
/// 255 满档在 0.5m 处照度 ≈160（主题点光 12 同尺度）——暗室里明显可见
pub const EMISSIVE_GAIN: f32 = 40.0;
/// 发光体素 NEE 点光球半径（fine）：须大于发光结构半对角，球面采样点落在
/// 发光体素外，避免阴影射线起点即自遮挡（薄结构 ≤2 体素宽时成立；
/// 大面积发光体的自遮挡透光归 3.5d/P9 调优）
pub const EMISSIVE_RADIUS: f32 = 24.0;
/// 发光体素 → NEE 点光聚类粒度（fine）。同桶内发光体素合并为一个代表点光，
/// 物理合理性：距离 < 16 fine 的发光体素视觉上无法区分，合并后亮度守恒、位置近似质心。
/// 2 根 2×96×2 灯柱（768 体素）→ ~4-6 聚类点光（远 < 768）
pub const EMISSIVE_CLUSTER_SIZE: i32 = 32;
/// 发光体素 → NEE 点光最大距离（fine，超出此距离的发光体素不进光源池）
/// 20m ≈ 半室外场景宽度——再远的点光平方反比衰减已极弱
pub const EMISSIVE_MAX_DIST_FINES: f32 = 20.0 * FINES_PER_M;
/// 发光体素直出增益：albedo × (emissive/255) × GAIN，绕过 N·L/阴影（自发光无方向）
pub const EMISSIVE_EMIT_GAIN: f32 = 4.0;

/// 光源描述（shader 镜像，48B；uniform 数组 stride 16 的倍数 ✓）
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, ShaderType)]
pub struct LightDesc {
  /// x = kind（0 = 方向光 / 1 = 点光）；
  /// yzw = L 轴（指向光，已归一，方向光）或球心位置（fine，点光）
  pub kind_pos_dir: Vec4,
  /// rgb = 线性色，w = 强度（方向光无量纲；点光为米制衰减系数）
  pub color_intensity: Vec4,
  /// x = 方向光盘角半径（rad）/ 点光球半径（fine）；yzw reserved
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
  /// rgb = 环境色（线性，直接乘 albedo），w reserved
  pub ambient: Vec4,
  /// x = 曝光系数（合成后统一乘）；yzw reserved
  pub exposure_pad: Vec4,
}

/// BG3 uniform 整体，WGSL `LightPool` 逐字段镜像。
/// 同时作 render world 资源（extract_light_pool 每帧产出，prepare 写 uniform buffer）。
/// 布局：LightGlobals(48) + 16×LightDesc(768) + sky_top(16) + sky_horizon(16) = 848B
#[repr(C)]
#[derive(Debug, Clone, Copy, Resource, ShaderType)]
pub struct LightPoolUniform {
  pub g: LightGlobals,
  pub lights: [LightDesc; MAX_LIGHTS],
  /// 天空天顶色（线性，用于 sky() 渐变）；主题有太阳时有效
  pub sky_top: Vec4,
  /// 天空地平线色（线性）；主题有太阳时有效
  pub sky_horizon: Vec4,
}

// ============================================================================
// 数据驱动主题（光源/环境/曝光资产化）
// ============================================================================

/// 方向光配置（主题资产）
#[derive(Debug, Clone, Deserialize)]
pub struct DirLightCfg {
  /// 光传播方向（指向场景）；打包时翻转为 L（指向光）
  pub dir: [f32; 3],
  /// 太阳盘角半径（度）——软阴影锥采样角
  pub angular_radius_deg: f32,
  pub color: [f32; 3],
  pub intensity: f32,
}

/// 点光源配置（主题资产）
#[derive(Debug, Clone, Deserialize)]
pub struct PointLightCfg {
  /// 球心位置（fine）
  pub pos: [f32; 3],
  /// 球半径（fine）——立体角采样尺度
  pub radius: f32,
  pub color: [f32; 3],
  pub intensity: f32,
}

/// 天空颜色配置（3.5a 程序化天空）
#[derive(Debug, Clone, Deserialize)]
pub struct SkyCfg {
  /// 天顶色（线性）
  pub top: [f32; 3],
  /// 地平线色（线性）
  pub horizon: [f32; 3],
}

/// 光照主题（`assets/lighting/*.ron`）：光源/环境/曝光一体的数据驱动配置。
/// main world 资源（gate-app 加载 RON 后 insert；extract_light_pool 跨 world 提取）。
#[derive(Debug, Clone, Resource, Deserialize)]
pub struct LightingTheme {
  pub sun: Option<DirLightCfg>,
  #[serde(default)]
  pub points: Vec<PointLightCfg>,
  /// 环境光色（线性；P3.4 升级体素 AO 前为常数项）
  pub ambient: [f32; 3],
  pub exposure: f32,
  /// 天空渐变；None = 纯色背景（暗色实验室用 null 直通深蓝黑）
  #[serde(default)]
  pub sky: Option<SkyCfg>,
}

impl Default for LightingTheme {
  /// 首发主题「暗色实验室」：暖白低角度太阳（软阴影显著）+ 中央顶灯 + 深蓝黑环境
  fn default() -> Self {
    Self {
      sun: Some(DirLightCfg {
        dir: Vec3::new(0.45, -0.72, 0.35).normalize().to_array(),
        angular_radius_deg: 2.0,
        color: [1.0, 0.95, 0.85],
        intensity: 3.0,
      }),
      points: vec![PointLightCfg {
        pos: [260.0, 340.0, 260.0],
        radius: 12.0,
        color: [1.0, 0.85, 0.65],
        intensity: 12.0,
      }],
      ambient: [0.045, 0.05, 0.07],
      exposure: 1.0,
      sky: None,
    }
  }
}

/// RON 解析（与 gate-ui parse_theme_ron 同型；调用方负责读文件 + 失败回退默认）
pub fn parse_lighting_ron(src: &str) -> Result<LightingTheme, ron::error::SpannedError> {
  ron::de::from_str(src)
}

/// 发光体素 → 点光空间聚类：同桶内发光体素合并为一个代表点光（亮度守恒、
/// 位置 = 强度加权质心、颜色 = 强度加权平均色）。
/// 768 体素灯柱 → ~4-6 聚类，解决每体素一个点光导致的阴影射线爆炸。
pub fn cluster_emissive_lights(descs: &[LightDesc]) -> Vec<LightDesc> {
  // bucket → (total_intensity, weighted_pos_sum, weighted_color_sum)
  let mut buckets: std::collections::HashMap<IVec3, (f32, Vec3, Vec3)> = Default::default();
  for ld in descs {
    let pos = yzw(ld.kind_pos_dir);
    let bucket = IVec3::new(
      (pos.x as i32) / EMISSIVE_CLUSTER_SIZE,
      (pos.y as i32) / EMISSIVE_CLUSTER_SIZE,
      (pos.z as i32) / EMISSIVE_CLUSTER_SIZE,
    );
    let intensity = ld.color_intensity.w;
    let color = Vec3::new(
      ld.color_intensity.x,
      ld.color_intensity.y,
      ld.color_intensity.z,
    );
    let entry = buckets
      .entry(bucket)
      .or_insert((0.0, Vec3::ZERO, Vec3::ZERO));
    entry.0 += intensity;
    entry.1 += pos * intensity;
    entry.2 += color * intensity;
  }
  buckets
    .into_values()
    .map(|(total_intensity, weighted_pos, weighted_color)| {
      let center = weighted_pos / total_intensity.max(1e-6);
      let color = weighted_color / total_intensity.max(1e-6);
      LightDesc {
        kind_pos_dir: Vec4::new(1.0, center.x, center.y, center.z),
        color_intensity: Vec4::new(color.x, color.y, color.z, total_intensity),
        shape: Vec4::new(EMISSIVE_RADIUS, 0.0, 0.0, 0.0),
      }
    })
    .collect()
}

pub fn build_light_pool(
  theme: &LightingTheme,
  emissive: &[LightDesc],
  eye: Vec3,
) -> LightPoolUniform {
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
    // 天空参数：无 sky 时 sky_top/horizon = ambient 色（混合后等价于纯 ambient，保持向后兼容）
    sky_top: Vec4::new(theme.ambient[0], theme.ambient[1], theme.ambient[2], 0.0),
    sky_horizon: Vec4::new(theme.ambient[0], theme.ambient[1], theme.ambient[2], 0.0),
  };
  if let Some(sky) = &theme.sky {
    u.sky_top = Vec4::new(sky.top[0], sky.top[1], sky.top[2], 0.0);
    u.sky_horizon = Vec4::new(sky.horizon[0], sky.horizon[1], sky.horizon[2], 0.0);
  }
  if let Some(sun) = &theme.sun {
    let l = -Vec3::from(sun.dir).normalize_or_zero();
    u.lights[u.g.count as usize] = LightDesc {
      kind_pos_dir: Vec4::new(0.0, l.x, l.y, l.z),
      color_intensity: Vec4::new(sun.color[0], sun.color[1], sun.color[2], sun.intensity),
      shape: Vec4::new(sun.angular_radius_deg.to_radians(), 0.0, 0.0, 0.0),
    };
    u.g.count += 1;
  }
  for p in theme.points.iter().take(MAX_LIGHTS - u.g.count as usize) {
    u.lights[u.g.count as usize] = LightDesc {
      kind_pos_dir: Vec4::new(1.0, p.pos[0], p.pos[1], p.pos[2]),
      color_intensity: Vec4::new(p.color[0], p.color[1], p.color[2], p.intensity),
      shape: Vec4::new(p.radius, 0.0, 0.0, 0.0),
    };
    u.g.count += 1;
  }
  // 发光元件：空间聚类 → 按到 eye 距离升序截断到可用槽数
  // 不做硬距离过滤——远处点光靠平方反比自然衰减（d_m² → 0），无可见贡献
  let clustered = cluster_emissive_lights(emissive);
  let mut rest: Vec<(f32, &LightDesc)> = clustered
    .iter()
    .map(|ld| {
      let d = yzw(ld.kind_pos_dir) - eye;
      (d.length_squared(), ld)
    })
    .collect();
  rest.sort_by(|a, b| a.0.total_cmp(&b.0));
  for (_, ld) in rest.iter().take(MAX_LIGHTS - u.g.count as usize) {
    u.lights[u.g.count as usize] = **ld;
    u.g.count += 1;
  }
  u
}

// ============================================================================
// 发光体素注册表（P3.2 临时桥）
// ============================================================================

/// 发光体素 → NEE 点光注册表。数据源 = 编辑路径 O(1) 钩子（调用方回传
/// set_voxel 的位置与 palette 索引），非全量扫描（tile = 512³ fine 不可遍历）。
/// P5 ComponentTable 落地后由元件聚合替换此桥（观察 API 保持不变）。
#[derive(Debug, Default, Resource)]
pub struct EmissiveLights {
  by_pos: std::collections::HashMap<IVec3, LightDesc>,
}

impl EmissiveLights {
  /// 编辑观察：pos 放置 pal_idx（0 = 清除）。emissive=0 的 palette 不注册
  /// 且会移除同位置旧光源（覆盖编辑语义）。
  pub fn observe_edit(&mut self, palette: &gate_voxel::Palette, pos: IVec3, pal_idx: u8) {
    if pal_idx == 0 || palette.get(pal_idx).emissive == 0 {
      self.by_pos.remove(&pos);
      return;
    }
    let e = palette.get(pal_idx);
    let c = Vec3::new(e.color[0] as f32, e.color[1] as f32, e.color[2] as f32) / 255.0;
    let intensity = (e.emissive as f32 / 255.0) * EMISSIVE_GAIN;
    // 点光球心 = 体素中心（fine）
    let center = pos.as_vec3() + Vec3::splat(0.5);
    self.by_pos.insert(
      pos,
      LightDesc {
        kind_pos_dir: Vec4::new(1.0, center.x, center.y, center.z),
        color_intensity: Vec4::new(c.x, c.y, c.z, intensity),
        shape: Vec4::new(EMISSIVE_RADIUS, 0.0, 0.0, 0.0),
      },
    );
  }

  /// 当前全部发光光源（截断排序由 build_light_pool 负责）
  pub fn descs(&self) -> Vec<LightDesc> {
    self.by_pos.values().copied().collect()
  }

  pub fn len(&self) -> usize {
    self.by_pos.len()
  }

  pub fn is_empty(&self) -> bool {
    self.by_pos.is_empty()
  }
}

// ============================================================================
// CPU 参考着色（WGSL 光照段逐字翻译的源）
// ============================================================================

/// 锥内固定采样方向：轴 l、半角 theta，k ∈ [0, SHADOW_SAMPLES) 取 φ 对称两方向。
/// 与 WGSL `cone_sample_dir` 逐字一致。
pub fn cone_sample_dir(l: Vec3, theta: f32, k: usize) -> Vec3 {
  let t = if l.y.abs() < 0.999 {
    l.cross(Vec3::Y)
  } else {
    l.cross(Vec3::X)
  }
  .normalize_or_zero();
  let b = l.cross(t);
  let phi = k as f32 * std::f32::consts::PI;
  (l * theta.cos() + (t * phi.cos() + b * phi.sin()) * theta.sin()).normalize_or_zero()
}

/// 球面黄金螺旋固定偏移（单位向量）：z ∈ {±0.5}，方位角按黄金角错开。
/// 与 WGSL `sphere_sample_offset` 逐字一致。
pub fn sphere_sample_offset(k: usize) -> Vec3 {
  let z = 1.0 - 2.0 * (k as f32 + 0.5) / SHADOW_SAMPLES as f32;
  let r = (1.0 - z * z).max(0.0).sqrt();
  let golden = (1.0 + 5.0f32.sqrt()) * std::f32::consts::PI;
  let phi = k as f32 * golden;
  Vec3::new(r * phi.cos(), z, r * phi.sin())
}

/// 命中点材质（palette 两 words 解包）：
/// albedo（u8 → /255 线性近似）+ roughness（w0>>24）+ emissive（w1 低 8bit）
/// 与 WGSL `hit_mat` 逐字一致
fn hit_mat(world: &BrickMapBuffers, pool: &MovPoolPacked, hit: &MovHit) -> (Vec3, f32, f32) {
  let (w0, w1) = if hit.obj == OBJ_WORLD {
    (
      world.b_palette[hit.pal as usize * 2],
      world.b_palette[hit.pal as usize * 2 + 1],
    )
  } else {
    let base = pool.descs[hit.obj as usize].palette_base as usize;
    (
      pool.mov_palette[base + hit.pal as usize * 2],
      pool.mov_palette[base + hit.pal as usize * 2 + 1],
    )
  };
  let albedo = Vec3::new(
    (w0 & 0xFF) as f32,
    ((w0 >> 8) & 0xFF) as f32,
    ((w0 >> 16) & 0xFF) as f32,
  ) / 255.0;
  let rough = ((w0 >> 24) & 0xFF) as f32 / 255.0;
  let emissive = (w1 & 0xFF) as f32 / 255.0;
  (albedo, rough, emissive)
}

/// Phong 简化高光系数（视图相关项；3.5d 逐体素量化时随直光进体素均值）。
/// 用真反射向量（Blinn 半程向量在 l≈v 场景严重高估：太阳正照平面时
/// dot(h,n)≈1，而物理上反射光背向观察者）。roughness 0（镜面）→ 指数 128 /
/// 强度 0.35，1（全粗糙）→ 指数 4 / 强度 0。
/// 与 WGSL `phong_spec` 逐字一致。
pub fn phong_spec(n: Vec3, l: Vec3, v: Vec3, rough: f32) -> f32 {
  let r = (2.0 * n.dot(l)) * n - l; // = reflect(-l, n)
  let exp = (1.0 - rough) * 124.0 + 4.0;
  let strength = (1.0 - rough) * 0.35;
  r.dot(v).max(0.0).powf(exp) * strength
}

// ============================================================================
// 3.5a 程序化天空 CPU 参考实现（与 WGSL sky() 逐字镜像）
// ============================================================================

/// WGSL sky() 的 CPU 镜像——渐变 + 太阳盘。参数来自 LightPoolUniform。
pub fn cpu_reference_sky(dir: Vec3, pool: &LightPoolUniform) -> Vec3 {
  let d = dir.normalize_or_zero();
  let h = d.y.clamp(0.0, 1.0);
  // smoothstep(0, 0.35, h)
  let t = {
    let x = (h / 0.35).clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
  };
  let col = xyz(pool.sky_horizon).lerp(xyz(pool.sky_top), t);
  if pool.g.count > 0 && pool.lights[0].kind_pos_dir.x < 0.5 {
    let sdir = yzw(pool.lights[0].kind_pos_dir);
    let cos_a = d.dot(sdir).max(0.0);
    let radius = pool.lights[0].shape.x;
    let sun_c = xyz(pool.lights[0].color_intensity) * pool.lights[0].color_intensity.w;
    let disc = {
      let edge1 = radius.cos();
      let edge2 = (radius * 0.8).cos();
      let x = ((cos_a - edge1) / (edge2 - edge1)).clamp(0.0, 1.0);
      x * x * (3.0 - 2.0 * x)
    };
    let glow = cos_a.max(0.0).powf(64.0) * 0.05 * if h > 0.0 { 1.0 } else { 0.0 };
    let on_disk = if h > 0.0 { disc } else { 0.0 };
    col + sun_c * (on_disk * 0.3 + glow)
  } else {
    col
  }
}

// ============================================================================
// CPU 参考着色：逐字镜像 WGSL shade_hit
// ============================================================================

/// 含 P3.2 粗糙度高光（视图相关）与发光直出（绕过 N·L/阴影）。
/// 与 WGSL `shade_hit` 逐字一致；未命中（t ≥ 1e29）不适用（由调用方走天空色）。
pub fn cpu_reference_shade_hit(
  world: &BrickMapBuffers,
  pool: &MovPoolPacked,
  light_pool: &LightPoolUniform,
  origin: Vec3,
  dir: Vec3,
  hit: MovHit,
  _shadow_t_max: f32,
) -> Vec3 {
  let p = origin + dir * hit.t;
  let n = hit.normal;
  let v = (-dir).normalize_or_zero();
  let (base, rough, emissive) = hit_mat(world, pool, &hit);
  // 3.4 环境光：常数 ambient + sky 渐变混合（不含太阳盘——太阳已作为直射光单独计算）
  let h = n.y.clamp(0.0, 1.0);
  let t_sky = {
    let x = (h / 0.35).clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
  };
  let sky_grad = xyz(light_pool.sky_horizon).lerp(xyz(light_pool.sky_top), t_sky);
  let mut col = base * (light_pool.g.ambient.truncate() * 0.6 + sky_grad * 0.4);
  let vis_step = 1.0 / SHADOW_SAMPLES as f32;
  for i in 0..(light_pool.g.count as usize).min(MAX_LIGHTS) {
    let ld = &light_pool.lights[i];
    if ld.kind_pos_dir.x < 0.5 {
      // ---- 方向光：盘角半径锥采样软阴影 ----
      let l_axis = yzw(ld.kind_pos_dir);
      let ndl = n.dot(l_axis).max(0.0);
      if ndl <= 0.0 {
        continue;
      }
      let theta = ld.shape.x;
      let o = p + n * SHADOW_BIAS;
      let mut vis = 0.0f32;
      for k in 0..SHADOW_SAMPLES {
        let d = cone_sample_dir(l_axis, theta, k);
        if !cpu_reference_scene_occluded(world, pool, o, d, 65536.0) {
          vis += vis_step;
        }
      }
      let c = ld.color_intensity.truncate() * ld.color_intensity.w;
      col += base * c * (ndl * vis);
      // 简化高光：光源色（不乘 albedo），乘可见性
      col += c * (phong_spec(n, l_axis, v, rough) * vis);
    } else {
      // ---- 点光：仅米制平方反比（不采阴影——Douglas 方案，大幅省算力）----
      let center = yzw(ld.kind_pos_dir);
      let to_l = center - p;
      let dist = to_l.length();
      if dist < 1e-4 {
        continue;
      }
      let l_axis = to_l / dist;
      let ndl = n.dot(l_axis).max(0.0);
      if ndl <= 0.0 {
        continue;
      }
      let vis = 1.0; // 不采阴影
      let d_m = dist / FINES_PER_M;
      let atten = 1.0 / (d_m * d_m).max(1e-6);
      let c = ld.color_intensity.truncate() * ld.color_intensity.w;
      col += base * c * (ndl * vis * atten);
      col += c * (phong_spec(n, l_axis, v, rough) * vis * atten);
    }
  }
  // 发光直出（P3.2）：albedo × emissive × GAIN，无方向性、不受阴影影响
  col += base * (emissive * EMISSIVE_EMIT_GAIN);
  col * light_pool.g.exposure_pad.x
}

// ============================================================================
// 单测
// ============================================================================

#[cfg(test)]
mod tests {
  use super::*;
  use crate::brickmap::{BrickMapBuilder, MovObject, cpu_reference_trace_scene, pack_mov_pool};
  use gate_voxel::{TileGrid, fill_box};
  use glam::{IVec3, Mat3, Vec4Swizzles};

  /// [0,ext)³ L0 满铺盒世界（palette = 64 灰，即 albedo 64/255）
  fn world_box(ext: i32, pal: u8) -> BrickMapBuffers {
    let mut g = TileGrid::new();
    g.palette_mut().get_mut(pal).color = [64, 64, 64];
    fill_box(&mut g, IVec3::ZERO, IVec3::splat(ext), 0, pal);
    BrickMapBuilder::build_full(&g).buffers().clone()
  }

  /// u8 调色板 → albedo（与 hit_mat 同式）
  const ALBEDO: f32 = 64.0 / 255.0;

  #[test]
  fn default_theme_pools_layout() {
    let pool = build_light_pool(&LightingTheme::default(), &[], Vec3::ZERO);
    assert_eq!(pool.g.count, 2, "默认主题 = 太阳 + 1 点光");
    // 太阳：kind 0、L 归一、与 dir 反向、角半径 rad 化
    let sun = &pool.lights[0];
    assert_eq!(sun.kind_pos_dir.x, 0.0);
    let l = yzw(sun.kind_pos_dir);
    assert!((l.length() - 1.0).abs() < 1e-5);
    let cfg = LightingTheme::default().sun.unwrap();
    let expect = -Vec3::from(cfg.dir).normalize();
    assert!(l.dot(expect) > 0.9999);
    assert!((sun.shape.x - cfg.angular_radius_deg.to_radians()).abs() < 1e-6);
    // 点光：kind 1、位置/半径落位
    let pt = &pool.lights[1];
    assert_eq!(pt.kind_pos_dir.x, 1.0);
    assert_eq!(pt.shape.x, 12.0);
    // header：ambient / exposure
    assert_eq!(pool.g.ambient.x, LightingTheme::default().ambient[0]);
    assert_eq!(pool.g.exposure_pad.x, LightingTheme::default().exposure);
  }

  #[test]
  fn parse_ron_minimal_and_error() {
    let src = r#"(
            sun: Some((dir: (0.0, -1.0, 0.0), angular_radius_deg: 1.0,
                  color: (1.0, 1.0, 1.0), intensity: 2.0)),
            points: [],
            ambient: (0.01, 0.01, 0.01),
            exposure: 1.5,
        )"#;
    let t = parse_lighting_ron(src).expect("valid ron");
    assert!(t.sun.is_some());
    assert!(t.points.is_empty());
    assert_eq!(t.exposure, 1.5);
    let pool = build_light_pool(&t, &[], Vec3::ZERO);
    assert_eq!(pool.g.count, 1);
    assert_eq!(pool.g.exposure_pad.x, 1.5);
    assert!(parse_lighting_ron("( sun: (").is_err());
  }

  #[test]
  fn cone_sample_dir_unit_and_angle() {
    let l = Vec3::new(0.3, -0.8, 0.5).normalize();
    let theta = 2.0f32.to_radians();
    for k in 0..SHADOW_SAMPLES {
      let d = cone_sample_dir(l, theta, k);
      assert!((d.length() - 1.0).abs() < 1e-5);
      // 采样方向与轴的夹角恰为 theta（sin/cos 展开保持单位长）
      assert!((d.dot(l) - theta.cos()).abs() < 1e-5, "k={k}");
    }
  }

  #[test]
  fn sphere_offset_unit_and_spread() {
    for k in 0..SHADOW_SAMPLES {
      let o = sphere_sample_offset(k);
      assert!((o.length() - 1.0).abs() < 1e-6);
      let z = 1.0 - 2.0 * (k as f32 + 0.5) / SHADOW_SAMPLES as f32;
      assert!((o.y - z).abs() < 1e-6);
    }
  }

  /// 地面盒（哑光，排除高光项）+ 垂直向下太阳：顶面直射（全采样可见），底面仅环境项
  #[test]
  fn shade_direct_and_occluded_extremes() {
    let world = world_box_rough(512, 3, 255);
    let theme = LightingTheme {
      sun: Some(DirLightCfg {
        dir: [0.0, -1.0, 0.0],
        angular_radius_deg: 2.0,
        color: [1.0, 1.0, 1.0],
        intensity: 2.0,
      }),
      points: vec![],
      ambient: [0.1, 0.1, 0.1],
      exposure: 1.0,
      sky: None,
    };
    let lp = build_light_pool(&theme, &[], Vec3::ZERO);
    let pool = MovPoolPacked::default();

    // 顶面命中：从上方射向 (256, 512, 256)，法线 +Y，N·L = 1，锥采样全可见
    let hit = cpu_reference_trace_scene(
      &world,
      &pool,
      Vec3::new(256.0, 640.0, 256.0),
      -Vec3::Y,
      4096.0,
    )
    .expect("顶面必有命中");
    assert_eq!(hit.obj, OBJ_WORLD);
    let rgb = cpu_reference_shade_hit(
      &world,
      &pool,
      &lp,
      Vec3::new(256.0, 640.0, 256.0),
      -Vec3::Y,
      hit,
      4096.0,
    );
    // albedo(64/255) → ALBEDO * (0.1 + 1.0 * 2.0 * 1.0)
    assert!((rgb.x - ALBEDO * 2.1).abs() < 1e-4, "顶面 rgb={rgb:?}");
    assert!((rgb.y - ALBEDO * 2.1).abs() < 1e-4);

    // 底面命中：从下方射向 (256, 0, 256)，法线 -Y，N·L = -1 → 仅环境项
    let hit = cpu_reference_trace_scene(
      &world,
      &pool,
      Vec3::new(256.0, -128.0, 256.0),
      Vec3::Y,
      4096.0,
    )
    .expect("底面必有命中");
    let rgb = cpu_reference_shade_hit(
      &world,
      &pool,
      &lp,
      Vec3::new(256.0, -128.0, 256.0),
      Vec3::Y,
      hit,
      4096.0,
    );
    assert!((rgb.x - ALBEDO * 0.1).abs() < 1e-4, "底面 rgb={rgb:?}");
  }

  /// 点光米制平方反比：直接合成数值锁死（atten = 1/(d/400)²）
  #[test]
  fn point_light_inverse_square() {
    let world = world_box_rough(512, 3, 255);
    let theme = LightingTheme {
      sun: None,
      points: vec![PointLightCfg {
        pos: [256.0, 620.0, 256.0],
        radius: 4.0,
        color: [1.0, 1.0, 1.0],
        intensity: 10.0,
      }],
      ambient: [0.0, 0.0, 0.0],
      exposure: 1.0,
      sky: None,
    };
    let lp = build_light_pool(&theme, &[], Vec3::ZERO);
    let pool = MovPoolPacked::default();
    let o = Vec3::new(256.0, 760.0, 256.0);
    let hit = cpu_reference_trace_scene(&world, &pool, o, -Vec3::Y, 4096.0).expect("顶面命中");
    let rgb = cpu_reference_shade_hit(&world, &pool, &lp, o, -Vec3::Y, hit, 4096.0);
    // dist = 620 - 512 = 108 fine = 0.27 m → atten ≈ 13.717；ndl=1、vis=1（锥顶正照无遮挡）
    let d_m = 108.0 / 400.0;
    let expect = ALBEDO * 10.0 / (d_m * d_m);
    assert!((rgb.x - expect).abs() < 1e-3, "rgb={rgb:?} expect={expect}");
  }

  /// MOV 物体遮挡太阳：地面命中点在物体正下方 → 锥采样全被挡 → 仅环境项；
  /// 同一点无物体 → 全采样可见 → 直射满额（动态物体天然投影的 CPU 锁定）
  #[test]
  fn mov_object_casts_shadow_on_ground() {
    let world = world_box_rough(512, 3, 255);
    let theme = LightingTheme {
      sun: Some(DirLightCfg {
        dir: [0.0, -1.0, 0.0],
        angular_radius_deg: 2.0,
        color: [1.0, 1.0, 1.0],
        intensity: 2.0,
      }),
      points: vec![],
      ambient: [0.1, 0.1, 0.1],
      exposure: 1.0,
      sky: None,
    };
    let lp = build_light_pool(&theme, &[], Vec3::ZERO);

    // 物体：满铺 1-tile 盒悬在 (240, 592, 240)，罩住地面点 (256, 512, 256) 正上方
    let obj_bufs = world_box(512, 5);
    let pool_with = pack_mov_pool(&[MovObject {
      buffers: &obj_bufs,
      pos: Vec3::new(240.0, 592.0, 240.0),
      rot: Mat3::IDENTITY,
      scale: 1.0,
    }]);
    let pool_empty = MovPoolPacked::default();

    // 地面顶面命中点手工构造：origin=(256,896,256)、dir=-Y、t=384 → p=(256,512,256)
    let hit = MovHit {
      t: 384.0,
      pal: 3,
      obj: OBJ_WORLD,
      normal: Vec3::Y,
    };
    let origin = Vec3::new(256.0, 896.0, 256.0);

    // 无物体：直射满额 ALBEDO * (0.1 + 2.0)
    let lit = cpu_reference_shade_hit(&world, &pool_empty, &lp, origin, -Vec3::Y, hit, 4096.0);
    assert!((lit.x - ALBEDO * 2.1).abs() < 1e-4, "无物体 lit={lit:?}");
    // 有物体：2 个锥采样全被物体挡 → 仅环境项 ALBEDO * 0.1
    let shadowed = cpu_reference_shade_hit(&world, &pool_with, &lp, origin, -Vec3::Y, hit, 4096.0);
    assert!(
      (shadowed.x - ALBEDO * 0.1).abs() < 1e-4,
      "有物体 shadowed={shadowed:?}"
    );
    assert!(shadowed.x < lit.x * 0.1);
  }

  // ================= P3.2：palette 材质消费 + 发光元件 NEE =================

  /// emissive 测试 palette（彩色发光，strength 200/255）
  fn emissive_palette() -> gate_voxel::Palette {
    let mut p = gate_voxel::Palette::new();
    let mut e = gate_voxel::PaletteEntry::default();
    e.color = [255, 160, 40];
    e.emissive = 200;
    p.set(7, e);
    let mut e2 = gate_voxel::PaletteEntry::default();
    e2.color = [10, 10, 10];
    p.set(9, e2); // 非 emissive
    p
  }

  /// observe_edit：emissive 注册 / 非 emissive 忽略 / 覆盖移除 / clear 移除
  #[test]
  fn emissive_collector_observe_and_remove() {
    let pal = emissive_palette();
    let mut em = EmissiveLights::default();
    let pos = IVec3::new(100, 64, 100);
    em.observe_edit(&pal, pos, 7);
    assert_eq!(em.len(), 1);
    let ld = em.descs()[0];
    // 点光 = 体素中心；intensity = 200/255 × GAIN；半径 = EMISSIVE_RADIUS
    assert_eq!(ld.kind_pos_dir.x, 1.0);
    assert_eq!(yzw(ld.kind_pos_dir), Vec3::new(100.5, 64.5, 100.5));
    assert!((ld.color_intensity.w - 200.0 / 255.0 * EMISSIVE_GAIN).abs() < 1e-5);
    assert_eq!(ld.shape.x, EMISSIVE_RADIUS);
    // 非 emissive palette：忽略
    em.observe_edit(&pal, IVec3::new(1, 1, 1), 9);
    assert_eq!(em.len(), 1);
    // 同位置改放非 emissive：移除（覆盖编辑语义）
    em.observe_edit(&pal, pos, 9);
    assert!(em.is_empty());
    // clear（pal 0）：移除
    em.observe_edit(&pal, pos, 7);
    em.observe_edit(&pal, pos, 0);
    assert!(em.is_empty());
  }

  /// build_light_pool：发光体素空间聚类 → 距离剔除 → 距离升序截断到可用槽
  #[test]
  fn build_light_pool_emissive_clustering_and_truncation() {
    let theme = LightingTheme::default(); // 2 主题光源
    let pal = emissive_palette();
    let mut em = EmissiveLights::default();
    // 30 个发光体素沿 X 轴，间距 10 fine（< CLUSTER_SIZE=32 → 每 3-4 个聚成一桶）
    for i in 0..30 {
      let pos = IVec3::new(100 + i * 10, 0, 0);
      em.observe_edit(&pal, pos, 7);
    }
    let pool = build_light_pool(&theme, &em.descs(), Vec3::ZERO);
    // 30 个体素 → 聚类后 ~10 桶（远 < 30），全部在 20m 内 → 全部进光源池（2+10=12 < 16）
    assert!(
      pool.g.count < 30 + 2,
      "聚类后光源数应远小于原始体素数：{}",
      pool.g.count
    );
    assert!(pool.g.count > 2, "至少应有发光聚类进入光源池");
    // 验证聚类亮度守恒：所有发光聚类 intensity 之和 ≈ 原始 30 个体素 intensity 之和
    let total_clustered_intensity: f32 = pool.lights[2..pool.g.count as usize]
      .iter()
      .map(|ld| ld.color_intensity.w)
      .sum();
    let expected_total = 30.0 * (200.0 / 255.0) * EMISSIVE_GAIN;
    assert!(
      (total_clustered_intensity - expected_total).abs() < 1e-3,
      "聚类亮度守恒：total={total_clustered_intensity} expect={expected_total}"
    );
  }

  /// cluster_emissive_lights：同桶内合并、亮度守恒、质心正确
  #[test]
  fn cluster_emissive_two_voxels_one_cluster() {
    let pal = emissive_palette();
    let mut em = EmissiveLights::default();
    // 两个相邻体素（间距 2 fine < 32 → 同桶）
    em.observe_edit(&pal, IVec3::new(100, 64, 100), 7);
    em.observe_edit(&pal, IVec3::new(101, 64, 100), 7);
    let clustered = cluster_emissive_lights(&em.descs());
    assert_eq!(clustered.len(), 1, "2 相邻体素应聚成 1 桶");
    let ld = &clustered[0];
    // 位置在 kind_pos_dir.yzw（x = kind = 1.0）
    let center = yzw(ld.kind_pos_dir);
    // 质心 = (100.5 + 101.5) / 2 = 101.0
    assert!(
      (center.x - 101.0).abs() < 1e-3,
      "质心 x 应为 101.0，实际 {}",
      center.x
    );
    // 总 intensity = 2 × 单体素 intensity
    let single_intensity = (200.0 / 255.0) * EMISSIVE_GAIN;
    assert!(
      (ld.color_intensity.w - 2.0 * single_intensity).abs() < 1e-5,
      "亮度守恒"
    );
  }

  /// cluster_emissive_lights：间距 > CLUSTER_SIZE → 不同桶
  #[test]
  fn cluster_emissive_two_voxels_two_clusters() {
    let pal = emissive_palette();
    let mut em = EmissiveLights::default();
    // 两个体素间距 100 fine（> 32 → 不同桶）
    em.observe_edit(&pal, IVec3::new(100, 64, 100), 7);
    em.observe_edit(&pal, IVec3::new(200, 64, 100), 7);
    let clustered = cluster_emissive_lights(&em.descs());
    assert_eq!(clustered.len(), 2, "间距 > CLUSTER_SIZE 应聚成 2 桶");
  }

  /// 发光直出：emissive 体素面无任何光源（纯暗）也亮；背光面同样直出（无方向性）
  #[test]
  fn shade_emissive_direct_glow() {
    let mut g = TileGrid::new();
    {
      let pal = g.palette_mut();
      let mut e = gate_voxel::PaletteEntry::default();
      e.color = [255, 160, 40];
      e.emissive = 200;
      pal.set(3, e);
    }
    fill_box(&mut g, IVec3::ZERO, IVec3::splat(512), 0, 3);
    let world = BrickMapBuilder::build_full(&g).buffers().clone();
    let theme = LightingTheme {
      sun: None,
      points: vec![],
      ambient: [0.0, 0.0, 0.0],
      exposure: 1.0,
      sky: None,
    };
    let lp = build_light_pool(&theme, &[], Vec3::ZERO);
    let pool = MovPoolPacked::default();
    // 顶面命中（无光源：环境 0 + 直射 0 + 直出）
    let hit = cpu_reference_trace_scene(
      &world,
      &pool,
      Vec3::new(256.0, 640.0, 256.0),
      -Vec3::Y,
      4096.0,
    )
    .expect("顶面命中");
    let rgb = cpu_reference_shade_hit(
      &world,
      &pool,
      &lp,
      Vec3::new(256.0, 640.0, 256.0),
      -Vec3::Y,
      hit,
      4096.0,
    );
    // 直出 = albedo(255/255=1.0) × emissive(200/255) × GAIN
    let expect = (255.0 / 255.0) * (200.0 / 255.0) * EMISSIVE_EMIT_GAIN;
    assert!(
      (rgb.x - expect).abs() < 1e-4,
      "顶面直出 rgb={rgb:?} expect={expect}"
    );
    // 底面命中：直出无方向性 → 同值
    let hit = cpu_reference_trace_scene(
      &world,
      &pool,
      Vec3::new(256.0, -128.0, 256.0),
      Vec3::Y,
      4096.0,
    )
    .expect("底面命中");
    let rgb = cpu_reference_shade_hit(
      &world,
      &pool,
      &lp,
      Vec3::new(256.0, -128.0, 256.0),
      Vec3::Y,
      hit,
      4096.0,
    );
    assert!((rgb.x - expect).abs() < 1e-4, "底面直出 rgb={rgb:?}");
  }

  /// phong_spec：正反射（v = reflect(-l, n)）满额、全粗糙归零、背面归零、粗糙度单调
  #[test]
  fn phong_spec_unit() {
    let n = Vec3::Y;
    // l 45° 入射：r = 2(n·l)n - l = (-1/√2, 1/√2, 0)，v 取 r → dot(r,v)=1
    let s = std::f32::consts::FRAC_1_SQRT_2;
    let l = Vec3::new(s, s, 0.0);
    let v = Vec3::new(-s, s, 0.0);
    assert!(
      (phong_spec(n, l, v, 0.0) - 0.35).abs() < 1e-3,
      "镜面正反射 = 满强度"
    );
    assert!(phong_spec(n, l, v, 1.0) < 1e-6, "全粗糙无高光");
    // dot(r, v) < 0：max(0) 归零
    let v2 = Vec3::new(0.0, -1.0, 0.0);
    assert_eq!(phong_spec(n, l, v2, 0.0), 0.0);
    // 粗糙度↑ → 高光↓
    assert!(phong_spec(n, l, v, 0.2) < phong_spec(n, l, v, 0.0));
  }

  /// E2E：发光体素（点光）照亮邻近地面（暗室：无太阳/点光主题）
  #[test]
  fn emissive_nee_illuminates_neighbor() {
    // 地面（哑光）+ 上方悬空单个发光体素（pal 5 emissive 255）
    let mut g = TileGrid::new();
    {
      let pal = g.palette_mut();
      let mut e = gate_voxel::PaletteEntry::default();
      e.color = [64, 64, 64];
      e.roughness = 255; // 哑光：排除高光项干扰
      pal.set(3, e); // 地面
      let mut led = gate_voxel::PaletteEntry::default();
      led.color = [255, 200, 120];
      led.emissive = 255;
      pal.set(5, led);
    }
    fill_box(&mut g, IVec3::ZERO, IVec3::splat(512), 0, 3);
    fill_box(&mut g, IVec3::new(248, 592, 248), IVec3::splat(16), 4, 5);
    let world = BrickMapBuilder::build_full(&g).buffers().clone();
    let theme = LightingTheme {
      sun: None,
      points: vec![],
      ambient: [0.02, 0.02, 0.02],
      exposure: 1.0,
      sky: None,
    };
    // LED 体素注册为点光（模拟 EmissiveLights 产出；球心 = 体素中心 +0.5）
    let pal_ref = g.palette();
    let mut em = EmissiveLights::default();
    em.observe_edit(pal_ref, IVec3::new(248, 592, 248), 5);
    let lp = build_light_pool(&theme, &em.descs(), Vec3::ZERO);
    assert_eq!(lp.g.count, 1);
    let pool = MovPoolPacked::default();

    // 观察点在 LED 侧面外（x=288 ∉ [248,264)）→ 射线避开 LED 顶面命中地面
    let o = Vec3::new(288.0, 896.0, 256.0);
    let hit = cpu_reference_trace_scene(&world, &pool, o, -Vec3::Y, 4096.0).expect("地面命中");
    assert_eq!(hit.pal, 3, "必须命中地面而非 LED");
    let rgb = cpu_reference_shade_hit(&world, &pool, &lp, o, -Vec3::Y, hit, 4096.0);
    // 数值锁：光心 (248.5,592.5,248.5)、地面点 (288,512,256)；2 个球面采样点
    // 均在 LED 体外且射线不穿 LED → vis=1；哑光地面 → 无高光项
    let center = Vec3::new(248.5, 592.5, 248.5);
    let p = Vec3::new(288.0, 512.0, 256.0);
    let to_l = center - p;
    let dist = to_l.length();
    let d_m = dist / FINES_PER_M;
    let ndl = to_l.y / dist;
    let expect = ALBEDO * 0.02 + ALBEDO * EMISSIVE_GAIN * ndl / (d_m * d_m);
    assert!((rgb.x - expect).abs() < 0.5, "rgb={rgb:?} expect={expect}");
  }

  /// 正上方太阳照平面（Phong 真反射向量）：镜面反射方向 = 法线（朝上），
  /// 视线越接近反射方向高光越强；45° 视线 pow(0.707, 128) ≈ 0 → 无高光差。
  /// （Blinn 半程向量在该场景 dot(h,n)≈1 恒满额高估，已换 Phong）
  #[test]
  fn sun_overhead_flat_floor_specular_by_view_angle() {
    let theme = LightingTheme {
      sun: Some(DirLightCfg {
        dir: [0.0, -1.0, 0.0],
        angular_radius_deg: 2.0,
        color: [1.0, 1.0, 1.0],
        intensity: 2.0,
      }),
      points: vec![],
      ambient: [0.1, 0.1, 0.1],
      exposure: 1.0,
      sky: None,
    };
    let lp = build_light_pool(&theme, &[], Vec3::ZERO);
    let mirror = world_box_rough(512, 3, 0);
    let matte = world_box_rough(512, 3, 255);
    let pool = MovPoolPacked::default();
    let target = Vec3::new(256.0, 512.0, 256.0);
    for (oz, has_spec) in [(340.0f32, true), (640.0, false)] {
      let o = Vec3::new(256.0, 896.0, oz);
      let dir = (target - o).normalize();
      let h1 = cpu_reference_trace_scene(&mirror, &pool, o, dir, 4096.0).expect("顶面命中");
      let rm = cpu_reference_shade_hit(&mirror, &pool, &lp, o, dir, h1, 4096.0);
      let h2 = cpu_reference_trace_scene(&matte, &pool, o, dir, 4096.0).expect("顶面命中");
      let ra = cpu_reference_shade_hit(&matte, &pool, &lp, o, dir, h2, 4096.0);
      let (rmx, rax) = (rm.x, ra.x);
      let diff = rmx - rax;
      // 镜面-哑光差 = c × phong_spec：r = n = +Y → dot(r,v) = v.y
      let v = (o - target).normalize();
      let expect_diff = 2.0 * 0.35 * v.y.powf(128.0);
      if has_spec {
        assert!(
          diff > 0.01 && (diff - expect_diff).abs() < 1e-2,
          "oz={oz} diff={diff} expect≈{expect_diff} 应有高光差"
        );
      } else {
        assert!(diff.abs() < 1e-4, "oz={oz} 45°视线应无高光差 diff={diff}");
      }
    }
  }

  /// 点光斜照镜面地面：高光差分（同几何只差 roughness，视点在光源正上方）
  #[test]
  fn point_light_specular_diff() {
    let theme = LightingTheme {
      sun: None,
      points: vec![PointLightCfg {
        pos: [256.0, 620.0, 100.0],
        radius: 4.0,
        color: [1.0, 1.0, 1.0],
        intensity: 10.0,
      }],
      ambient: [0.0, 0.0, 0.0],
      exposure: 1.0,
      sky: None,
    };
    let lp = build_light_pool(&theme, &[], Vec3::ZERO);
    let mirror = world_box_rough(512, 3, 0);
    let matte = world_box_rough(512, 3, 255);
    let pool = MovPoolPacked::default();
    // 视点在光源正上方、观察点 = 光源正下方地面 → 正反射方向指向视点 → 高光满额
    let o = Vec3::new(256.0, 896.0, 100.0);
    let dir = -Vec3::Y;
    let h1 = cpu_reference_trace_scene(&mirror, &pool, o, dir, 4096.0).expect("顶面命中");
    let rm = cpu_reference_shade_hit(&mirror, &pool, &lp, o, dir, h1, 4096.0);
    let h2 = cpu_reference_trace_scene(&matte, &pool, o, dir, 4096.0).expect("顶面命中");
    let ra = cpu_reference_shade_hit(&matte, &pool, &lp, o, dir, h2, 4096.0);
    assert!(rm.x > ra.x, "镜面 rgb.x={} 应含高光高于哑光 {}", rm.x, ra.x);
  }

  /// 指定 roughness 的地面盒
  fn world_box_rough(ext: i32, pal: u8, rough: u8) -> BrickMapBuffers {
    let mut g = TileGrid::new();
    {
      let p = g.palette_mut();
      let mut e = gate_voxel::PaletteEntry::default();
      e.color = [64, 64, 64];
      e.roughness = rough;
      p.set(pal, e);
    }
    fill_box(&mut g, IVec3::ZERO, IVec3::splat(ext), 0, pal);
    BrickMapBuilder::build_full(&g).buffers().clone()
  }
}
