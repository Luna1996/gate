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
use glam::{Vec3, Vec4};
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

/// 光源上限（uniform 预算：48B header + 8×48B = 432B）
pub const MAX_LIGHTS: usize = 8;
/// 阴影采样数（锥/球各 2 个固定偏移；确定性、无帧间闪烁）
pub const SHADOW_SAMPLES: usize = 2;
/// 阴影射线起点沿法线偏移（fine），消除自遮挡 acne
pub const SHADOW_BIAS: f32 = 0.5;
/// fine → 米（0.25cm/fine）；点光衰减用米制（intensity 语义 = 米制）
pub const FINES_PER_M: f32 = 400.0;

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

/// BG3 uniform 整体（48 + 8×48 = 432B），WGSL `LightPool` 逐字段镜像。
/// 同时作 render world 资源（extract_light_pool 每帧产出，prepare 写 uniform buffer）。
#[repr(C)]
#[derive(Debug, Clone, Copy, Resource, ShaderType)]
pub struct LightPoolUniform {
    pub g: LightGlobals,
    pub lights: [LightDesc; MAX_LIGHTS],
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
        }
    }
}

/// RON 解析（与 gate-ui parse_theme_ron 同型；调用方负责读文件 + 失败回退默认）
pub fn parse_lighting_ron(src: &str) -> Result<LightingTheme, ron::error::SpannedError> {
    ron::de::from_str(src)
}

/// 主题 → GPU uniform 打包（方向归一、deg→rad、L 翻转都在此定式）
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
    };
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
    u
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

/// 命中点 albedo（palette u8 → 线性近似，P3.5 tonemap 时统一 sRGB 解码）
fn hit_rgb(world: &BrickMapBuffers, pool: &MovPoolPacked, hit: &MovHit) -> Vec3 {
    let w = if hit.obj == OBJ_WORLD {
        world.b_palette[hit.pal as usize * 2]
    } else {
        let base = pool.descs[hit.obj as usize].palette_base as usize;
        pool.mov_palette[base + hit.pal as usize * 2]
    };
    Vec3::new(
        (w & 0xFF) as f32,
        ((w >> 8) & 0xFF) as f32,
        ((w >> 16) & 0xFF) as f32,
    ) / 255.0
}

/// 直射光合成（NEE）：命中点 P、法线 N、albedo → 环境项 + 逐光源锥/球采样软阴影。
/// 与 WGSL `shade_hit` 逐字一致；未命中（t ≥ 1e29）不适用（由调用方走天空色）。
pub fn cpu_reference_shade_hit(
    world: &BrickMapBuffers,
    pool: &MovPoolPacked,
    light_pool: &LightPoolUniform,
    origin: Vec3,
    dir: Vec3,
    hit: MovHit,
    shadow_t_max: f32,
) -> Vec3 {
    let p = origin + dir * hit.t;
    let n = hit.normal;
    let base = hit_rgb(world, pool, &hit);
    let mut col = base * light_pool.g.ambient.truncate();
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
                if !cpu_reference_scene_occluded(world, pool, o, d, shadow_t_max) {
                    vis += vis_step;
                }
            }
            let c = ld.color_intensity.truncate() * ld.color_intensity.w;
            col += base * c * (ndl * vis);
        } else {
            // ---- 点光：球面立体角采样软阴影 + 米制平方反比 ----
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
            let radius = ld.shape.x;
            let o = p + n * SHADOW_BIAS;
            let mut vis = 0.0f32;
            for k in 0..SHADOW_SAMPLES {
                let sp = center + sphere_sample_offset(k) * radius;
                let seg = sp - o;
                let seg_len = seg.length();
                if seg_len < 1e-4 {
                    continue;
                }
                if !cpu_reference_scene_occluded(world, pool, o, seg / seg_len, seg_len) {
                    vis += vis_step;
                }
            }
            let d_m = dist / FINES_PER_M;
            let atten = 1.0 / (d_m * d_m).max(1e-6);
            let c = ld.color_intensity.truncate() * ld.color_intensity.w;
            col += base * c * (ndl * vis * atten);
        }
    }
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
    use glam::{IVec3, Mat3};

    /// [0,ext)³ L0 满铺盒世界（palette = 64 灰，即 albedo 64/255）
    fn world_box(ext: i32, pal: u8) -> BrickMapBuffers {
        let mut g = TileGrid::new();
        g.palette_mut().get_mut(pal).color = [64, 64, 64];
        fill_box(&mut g, IVec3::ZERO, IVec3::splat(ext), 0, pal);
        BrickMapBuilder::build_full(&g).buffers().clone()
    }

    /// u8 调色板 → albedo（与 hit_rgb 同式）
    const ALBEDO: f32 = 64.0 / 255.0;

    #[test]
    fn default_theme_pools_layout() {
        let pool = build_light_pool(&LightingTheme::default());
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
        let pool = build_light_pool(&t);
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

    /// 地面盒 + 垂直向下太阳：顶面直射（全采样可见），底面仅环境项
    #[test]
    fn shade_direct_and_occluded_extremes() {
        let world = world_box(512, 3);
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
        };
        let lp = build_light_pool(&theme);
        let pool = MovPoolPacked::default();

        // 顶面命中：从上方射向 (256, 512, 256)，法线 +Y，N·L = 1，锥采样全可见
        let hit = cpu_reference_trace_scene(&world, &pool, Vec3::new(256.0, 640.0, 256.0), -Vec3::Y, 4096.0)
            .expect("顶面必有命中");
        assert_eq!(hit.obj, OBJ_WORLD);
        let rgb = cpu_reference_shade_hit(&world, &pool, &lp, Vec3::new(256.0, 640.0, 256.0), -Vec3::Y, hit, 4096.0);
        // albedo(64/255) → ALBEDO * (0.1 + 1.0 * 2.0 * 1.0)
        assert!((rgb.x - ALBEDO * 2.1).abs() < 1e-4, "顶面 rgb={rgb:?}");
        assert!((rgb.y - ALBEDO * 2.1).abs() < 1e-4);

        // 底面命中：从下方射向 (256, 0, 256)，法线 -Y，N·L = -1 → 仅环境项
        let hit = cpu_reference_trace_scene(&world, &pool, Vec3::new(256.0, -128.0, 256.0), Vec3::Y, 4096.0)
            .expect("底面必有命中");
        let rgb = cpu_reference_shade_hit(&world, &pool, &lp, Vec3::new(256.0, -128.0, 256.0), Vec3::Y, hit, 4096.0);
        assert!((rgb.x - ALBEDO * 0.1).abs() < 1e-4, "底面 rgb={rgb:?}");
    }

    /// 点光米制平方反比：直接合成数值锁死（atten = 1/(d/400)²）
    #[test]
    fn point_light_inverse_square() {
        let world = world_box(512, 3);
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
        };
        let lp = build_light_pool(&theme);
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
        let world = world_box(512, 3);
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
        };
        let lp = build_light_pool(&theme);

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
        let hit = MovHit { t: 384.0, pal: 3, obj: OBJ_WORLD, normal: Vec3::Y };
        let origin = Vec3::new(256.0, 896.0, 256.0);

        // 无物体：直射满额 ALBEDO * (0.1 + 2.0)
        let lit = cpu_reference_shade_hit(&world, &pool_empty, &lp, origin, -Vec3::Y, hit, 4096.0);
        assert!((lit.x - ALBEDO * 2.1).abs() < 1e-4, "无物体 lit={lit:?}");
        // 有物体：2 个锥采样全被物体挡 → 仅环境项 ALBEDO * 0.1
        let shadowed = cpu_reference_shade_hit(&world, &pool_with, &lp, origin, -Vec3::Y, hit, 4096.0);
        assert!((shadowed.x - ALBEDO * 0.1).abs() < 1e-4, "有物体 shadowed={shadowed:?}");
        assert!(shadowed.x < lit.x * 0.1);
    }
}
