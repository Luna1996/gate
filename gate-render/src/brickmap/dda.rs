//! P2.4 DDA 主可见性 pass：WGSL compute + Core2d PostProcess blit
//!
//! 完整链路见 `.trae/specs/p24_dda_visibility/spec.md`。
//! - 静态视图（Startup 注入 Mat4，P2.6 再接动态相机）
//! - BG0：storage tex + 相机 uniform
//! - BG1：b_struct + b_leaves + palette + globals uniform（五 buffer + 两 uniform）
//! - DDA 着色器：`shaders/dda.wgsl`

use bevy::{
    image::Image,
    prelude::*,
    render::{extract_resource::ExtractResource, render_resource::*},
};
use std::ops::Mul;

use crate::gradient::{VIEW_SIZE, create_gradient_image};

// ============================================================================
// Task 1: 视图资源 + 图像资源
// ============================================================================

/// 主 world Startup 注入的静态视图配置（P2.4 不变）
///
/// 手算 perspective_rh(fovy=60°, aspect=1280/720) × look_at_rh：
/// - eye fine (700, 560, 700)，target fine (260, 120, 260)，距离 ~762
/// - 覆盖 tile(0,0,0) 全场景（fine 0..512）+ tile(1,0,0) 内增量热点（x 656..688）
/// - up = Vec3::Y，far 4000（DDA 用 inv_view_proj 反投影方向，far 只影响精度）
#[derive(Resource, Clone, Copy)]
pub struct DdaCameraConfig {
    pub view_proj: Mat4,
    pub inv_view_proj: Mat4,
    pub position_world: Vec3,
}

impl DdaCameraConfig {
    /// 静态视图构建（spec FR-11）
    pub fn build_static() -> Self {
        let eye = Vec3::new(700.0, 560.0, 700.0);
        let target = Vec3::new(260.0, 120.0, 260.0);
        let up = Vec3::Y;
        let aspect = VIEW_SIZE.x as f32 / VIEW_SIZE.y as f32;
        let fovy = 60.0_f32.to_radians();
        // near = 最细格点（0.25cm）粒度；near/far 比过大（如 0.01/4000）会让
        // inv_view_proj 条件数爆炸（求逆误差 5e-3），反投影射线方向失真
        let near = 1.0;
        let far = 4000.0;
        // Bevy Mat4：perspective_rh 右手系 +y 上 -z 前；look_at_rh 朝 -z
        let proj = Mat4::perspective_rh(fovy, aspect, near, far);
        let view = Mat4::look_at_rh(eye, target, up);
        let view_proj = proj.mul(view);
        let inv_view_proj = view_proj.inverse();
        Self {
            view_proj,
            inv_view_proj,
            position_world: eye,
        }
    }
}

/// 相机约束常量（spec FR-3 clamp；pub 供 gate-app 输入 system 与测试断言）
pub const PITCH_LIMIT: f32 = 89.0_f32.to_radians(); // ±89° 防万向节锁（up 与 view 共线）
pub const DIST_MIN: f32 = 32.0; // 最近 32 fine（8cm，不穿进体素内部失稳）
// DIST_MAX：原 8000（20m），用户要求取消"最远距离"设置——不再有上限 clamp。
// 为了避免滚轮异常操作产生 NaN（乘 0 或 exp overflow），仅保留下界 DIST_MIN。
// （透视 far 面由 CAM_FAR 负责，滚轮无限拉远时远处依然能可见，只要 CAM_FAR 足够大）

/// 轨道相机参数（main world 资源，gate-app 输入 system 操作；P2.6 spec FR-1）
///
/// 字段语义：target = 注视点（fine）、distance = 相机到 target 距离（fine）、
/// yaw = 绕 +Y 方位角（rad，atan2(x, z)）、pitch = 仰角（rad，+ 为上仰）。
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct OrbitCamera {
    pub target: Vec3,
    pub distance: f32,
    pub yaw: f32,
    pub pitch: f32,
}

impl OrbitCamera {
    /// 从眼位和目标点构造轨道参数（FR-1：offset → distance / pitch / yaw）
    pub fn from_eye(eye: Vec3, target: Vec3) -> Self {
        let offset = eye - target;
        let distance = offset.length();
        // atan2(y, |xz|) 等价 asin(y/len) 但 distance→0 时不出 NaN
        let pitch = offset.y.atan2(offset.xz().length());
        let yaw = offset.x.atan2(offset.z);
        Self {
            target,
            distance,
            yaw,
            pitch,
        }
    }

    /// 轨道参数重建眼位（FR-1 公式：eye = target + distance·(sin_yaw·cos_pitch, sin_pitch, cos_yaw·cos_pitch)）
    pub fn eye(&self) -> Vec3 {
        let (sin_yaw, cos_yaw) = self.yaw.sin_cos();
        let (sin_pitch, cos_pitch) = self.pitch.sin_cos();
        self.target + self.distance * Vec3::new(sin_yaw * cos_pitch, sin_pitch, cos_yaw * cos_pitch)
    }

    /// 应用约束（pitch ±89°、distance ≥ DIST_MIN；yaw 自由旋转不 clamp；
    ///  distance 不再有 DIST_MAX 上限——用户已取消"最远距离"设置）
    pub fn clamp(&mut self) {
        self.pitch = self.pitch.clamp(-PITCH_LIMIT, PITCH_LIMIT);
        self.distance = self.distance.max(DIST_MIN);
    }
}

impl DdaCameraConfig {
    /// 唯一矩阵构造点（spec FR-2）：orbit 参数 → perspective_rh × look_at_rh。
    ///
    /// fov/aspect/near/far 为显式参数（app 传 60° / 1280:720 / 1.0 / 4000），
    /// 不写死在 orbit 里——性能面板改 fov、多分辨率改 aspect 时复用同一入口。
    pub fn from_orbit(orbit: &OrbitCamera, fov_y: f32, aspect: f32, near: f32, far: f32) -> Self {
        let eye = orbit.eye();
        let view = Mat4::look_at_rh(eye, orbit.target, Vec3::Y);
        let proj = Mat4::perspective_rh(fov_y, aspect, near, far);
        let view_proj = proj.mul(view);
        Self {
            view_proj,
            inv_view_proj: view_proj.inverse(),
            position_world: eye,
        }
    }
}

/// Render-world 着色器绑定的 camera uniform（ShaderType derive = 144B）
#[derive(Resource, Clone, Copy, ShaderType)]
pub struct DdaViewUniform {
    pub inv_view_proj: Mat4,
    pub cam_pos_fine: Vec4, // w=1
    pub _pad: Vec4,
}

impl DdaViewUniform {
    pub fn from_cfg(cfg: &DdaCameraConfig) -> Self {
        Self {
            inv_view_proj: cfg.inv_view_proj,
            // 世界单位 = fine 单位（0.25cm）；DDA 着色器 dir 同样不缩放
            cam_pos_fine: cfg.position_world.extend(1.0),
            _pad: Vec4::ZERO,
        }
    }
}

/// DDA 着色器的输出纹理（main world 创建，提取进 render world）
#[derive(Resource, Clone, ExtractResource)]
pub struct DdaImages {
    pub target: Handle<Image>,
}

/// 工厂：与 GradientImage 同配置（rgba8unorm VIEW_SIZE，STORAGE|TEXTURE + RENDER_WORLD usage）
///
/// 复用 gradient.rs 的 create_gradient_image 逻辑（完全一致）
pub fn create_dda_image(images: &mut Assets<Image>) -> Handle<Image> {
    // 完全相同：Rgba8Unorm VIEW_SIZE + usage=STORAGE_BINDING | TEXTURE_BINDING
    // 直接复用 create_gradient_image 保证配置一致（避免重复参数后续 drift）
    create_gradient_image(images)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// TR-1.3: view_proj × inv_view_proj = IDENTITY（f32 求逆精度容差 1e-3）
    #[test]
    fn mat4_reversible_static_view() {
        let cfg = DdaCameraConfig::build_static();
        let r = cfg.view_proj * cfg.inv_view_proj;
        for i in 0..16 {
            let exp = if i / 4 == i % 4 { 1.0 } else { 0.0 };
            let got = r.to_cols_array()[i];
            // f32 mat4 求逆在 far/near=4000 下的固有精度 ~1.2e-4（f32 epsilon × 矩阵规模），
            // 1e-3 容差验证数值可逆；若条件数劣化（如 near 过小）会到 5e-3 量级，仍能被此断言抓出
            assert!((got - exp).abs() < 1e-3, "element {i}: {got} vs {exp}");
        }
    }

    /// AC-1①: from_eye → eye() 重建往返一致（<1e-4）
    #[test]
    fn orbit_roundtrip_from_eye_eye() {
        let eye = Vec3::new(700.0, 560.0, 700.0);
        let target = Vec3::new(260.0, 120.0, 260.0);
        let orbit = OrbitCamera::from_eye(eye, target);
        // 中间量健全性：offset=(440,440,440) → distance=440√3≈762.1024、pitch=atan(1/√2)、yaw=45°
        assert!(
            (orbit.distance - 762.1024).abs() < 0.01,
            "distance {}",
            orbit.distance
        );
        assert!(
            (orbit.pitch - 1.0_f32.atan2(2.0_f32.sqrt())).abs() < 1e-5,
            "pitch {}",
            orbit.pitch
        );
        assert!((orbit.yaw - std::f32::consts::FRAC_PI_4).abs() < 1e-5);
        let rebuilt = orbit.eye();
        assert!(
            (rebuilt - eye).length() < 1e-4,
            "eye rebuild drift {}",
            (rebuilt - eye).length()
        );
        // 任意参数化往返：负 pitch / 大 yaw
        let eye2 = Vec3::new(-123.0, 40.0, 900.0);
        let target2 = Vec3::new(512.0, 256.0, -30.0);
        let orbit2 = OrbitCamera::from_eye(eye2, target2);
        assert!((orbit2.eye() - eye2).length() < 1e-4);
    }

    /// AC-1②: clamp 生效（pitch 95°→89°、distance 1→32；上界一并验证）
    #[test]
    fn orbit_clamp_bounds() {
        // pitch clamp（上下界）+ distance 仅下界不再有 DIST_MAX 上限（用户已取消）
        let mut o = OrbitCamera {
            target: Vec3::ZERO,
            distance: 1.0,
            yaw: 0.0,
            pitch: 95.0_f32.to_radians(),
        };
        o.clamp();
        assert_eq!(o.pitch, PITCH_LIMIT);
        assert_eq!(o.distance, DIST_MIN);
        // 超大 distance（原 DIST_MAX=8000 的 100×）应保留原值（不上限 clamp）
        let big = 1.0e6;
        let mut o2 = OrbitCamera {
            target: Vec3::ZERO,
            distance: big,
            yaw: 0.0,
            pitch: -95.0_f32.to_radians(),
        };
        o2.clamp();
        assert_eq!(o2.pitch, -PITCH_LIMIT);
        assert_eq!(o2.distance, big);
    }

    /// AC-1③: from_orbit(from_eye(历史参数)) 与 build_static 三字段逐元素一致（<1e-5，回归保护）
    #[test]
    fn orbit_from_orbit_equals_build_static() {
        let orbit = OrbitCamera::from_eye(
            Vec3::new(700.0, 560.0, 700.0),
            Vec3::new(260.0, 120.0, 260.0),
        );
        let via_orbit = DdaCameraConfig::from_orbit(
            &orbit,
            60.0_f32.to_radians(),
            VIEW_SIZE.x as f32 / VIEW_SIZE.y as f32,
            1.0,
            4000.0,
        );
        let static_ = DdaCameraConfig::build_static();
        for i in 0..16 {
            let a = via_orbit.view_proj.to_cols_array()[i];
            let b = static_.view_proj.to_cols_array()[i];
            assert!((a - b).abs() < 1e-5, "view_proj[{i}]: {a} vs {b}");
            let a = via_orbit.inv_view_proj.to_cols_array()[i];
            let b = static_.inv_view_proj.to_cols_array()[i];
            assert!((a - b).abs() < 1e-5, "inv_view_proj[{i}]: {a} vs {b}");
        }
        assert!(via_orbit.position_world.distance(static_.position_world) < 1e-5);
    }

    /// FR-5: 窗口 resize 后按新 aspect 重算（fov/near/far 固定，矩阵与 perspective_rh 直算一致）
    #[test]
    fn config_rebuild_follows_window_aspect() {
        let orbit = OrbitCamera::from_eye(
            Vec3::new(700.0, 560.0, 700.0),
            Vec3::new(260.0, 120.0, 260.0),
        );
        let fovy = 60.0_f32.to_radians();
        for &(w, h) in &[(1280u32, 720u32), (1920, 1080), (1024, 769), (960, 540)] {
            let aspect = w as f32 / h as f32;
            let cfg = DdaCameraConfig::from_orbit(&orbit, fovy, aspect, 1.0, 4000.0);
            // 直算对照：perspective_rh × look_at_rh
            let view = Mat4::look_at_rh(orbit.eye(), orbit.target, Vec3::Y);
            let expect = Mat4::perspective_rh(fovy, aspect, 1.0, 4000.0).mul(view);
            for i in 0..16 {
                let a = cfg.view_proj.to_cols_array()[i];
                let b = expect.to_cols_array()[i];
                assert!((a - b).abs() < 1e-5, "{w}x{h} vp[{i}]: {a} vs {b}");
            }
            // 可逆性（resize 后反投影仍成立）
            let r = cfg.view_proj * cfg.inv_view_proj;
            for i in 0..16 {
                let exp = if i / 4 == i % 4 { 1.0 } else { 0.0 };
                assert!(
                    (r.to_cols_array()[i] - exp).abs() < 1e-3,
                    "{w}x{h} inv[{i}] drift"
                );
            }
        }
    }
}

// ============================================================================
// Task 2: WGSL 常量 Rust 镜像（AC-3 单测 + 槽位打包 roundtrip）
// ============================================================================

/// WGSL 着色器顶部 `const` 的 Rust 镜像副本（单测 assert_eq! 对 wire.rs 常量，防漂移）
pub mod wgsl_consts {
    // 与 WGSL 顶部 const 逐字对应；source = Rust wire.rs 对应项
    pub const TILE_INDEX_CAP: u32 = 128;
    pub const TILE_CAP: u32 = 1024;
    pub const BITMAP_BASE: u32 = 2_097_152; // 128³
    pub const TILE_BITMAP_WORDS: u32 = 1024;
    pub const DIR_BASE: u32 = 3_145_728; // BITMAP_BASE + TILE_CAP*TILE_BITMAP_WORDS = 2097152+1048576
    pub const CELL_DIR_WORDS: u32 = 32_768;
    pub const HDR_UNIFORM_MASK: u32 = 0xFF;
    pub const HDR_HAS_L1: u32 = 1 << 8;
    pub const HDR_HAS_L2: u32 = 1 << 9;
    pub const HDR_HAS_L3: u32 = 1 << 10;
    pub const HDR_HAS_BRICK: u32 = 1 << 11;
    pub const BRICK_SLAB_WORDS: u32 = 1024;
    // WGSL 无法 const array，单 const 分开
    pub const ST_L1_WORDS: u32 = 4;
    pub const ST_L2_WORDS: u32 = 32;
    pub const ST_L3_WORDS: u32 = 256;
    pub const ST_BRICK_PTR_WORDS: u32 = 2;
    pub const TILE_SUB: u32 = 512; // tile 边长 fine（32 cells × 16 sub/cell）
    pub const SUB_PER_CELL: u32 = 16; // 基元胞边长 fine
}

/// 与 WGSL `slot_unpack(word, half)` 等价：半 0 = lower 16bit；半 1 = upper 16bit
pub fn wgsl_slot_unpack(word: u32, half: u32) -> u16 {
    debug_assert!(half <= 1);
    ((word >> (16 * half)) & 0xFFFF) as u16
}

/// 与 WGSL `slot_pack(tag,pal) -> u16` 等价：tag<<8 | pal
pub fn wgsl_slot_pack(tag: u8, pal: u8) -> u16 {
    ((tag as u16) << 8) | (pal as u16)
}

#[cfg(test)]
mod const_tests {
    use super::wgsl_consts::*;
    use super::*;
    use crate::brickmap::wire as w;

    /// 确定性 xorshift64（同 dda_ref_tests）
    fn xorshift64(state: &mut u64) -> u32 {
        let mut x = *state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *state = x;
        (x & 0xFFFF_FFFF) as u32
    }

    /// TR-2.1: WGSL 常量 == wire.rs 对应常量（≥16 项对齐）
    #[test]
    fn wire_wgsl_constants_aligned() {
        // 直接按常量名对应
        // TILE_INDEX_CAP: i32, BITMAP_BASE/DIR_BASE/..._WORDS: usize → as u32 必要；
        // HDR_*: u32, TILE_CAP/TILE_SUB: u32 → 无需 as
        assert_eq!(TILE_INDEX_CAP, (w::TILE_INDEX_CAP as u32));
        assert_eq!(BITMAP_BASE, (w::BITMAP_BASE as u32));
        assert_eq!(TILE_BITMAP_WORDS, (w::TILE_BITMAP_WORDS as u32));
        assert_eq!(DIR_BASE, (w::DIR_BASE as u32));
        assert_eq!(CELL_DIR_WORDS, (w::CELL_DIR_WORDS as u32));
        assert_eq!(HDR_UNIFORM_MASK, w::HDR_UNIFORM_MASK);
        assert_eq!(HDR_HAS_L1, w::HDR_HAS_L1);
        assert_eq!(HDR_HAS_L2, w::HDR_HAS_L2);
        assert_eq!(HDR_HAS_L3, w::HDR_HAS_L3);
        assert_eq!(HDR_HAS_BRICK, w::HDR_HAS_BRICK);
        assert_eq!(BRICK_SLAB_WORDS, (w::BRICK_SLAB_WORDS as u32));
        assert_eq!(ST_L1_WORDS, (w::SLOT_TABLE_WORDS[1] as u32));
        assert_eq!(ST_L2_WORDS, (w::SLOT_TABLE_WORDS[2] as u32));
        assert_eq!(ST_L3_WORDS, (w::SLOT_TABLE_WORDS[3] as u32));
        // ST_BRICK_PTR_WORDS = 2：L4 BRICK = hdr 字 + brick_index 字（共 2 words）
        // wire.rs SLOT_TABLE_WORDS[4] = 0（brick 无槽表，存 b_leaves slab），分别对应，不等值
        assert_eq!(ST_BRICK_PTR_WORDS, 2u32);
        assert_eq!(TILE_CAP, 1024); // wire.rs TILE_CAP
        assert_eq!(TILE_SUB, 512); // tile 边长 fine = 32 cells × 16 sub/cell
        assert_eq!(SUB_PER_CELL, gate_voxel::SUB_PER_CELL as u32);
    }

    /// TR-2.2: 200 随机(tag, pal)，WGS L pack/unpack 与 wire.rs encode_slot/unpack_slot_word 等价
    #[test]
    fn slot_pack_roundtrip_wgsl_vs_rust_200_random() {
        let mut state: u64 = 0x517CC1B727220A95;
        for _ in 0..200 {
            let tag: u8 = (xorshift64(&mut state) % 3) as u8; // 0 empty,1 leaf,2 branch
            let pal: u8 = (xorshift64(&mut state) & 0xFF) as u8;
            // 先 WGSL 侧 pack → u16
            let s1 = wgsl_slot_pack(tag, pal);
            // wire.rs encode_slot = (tag << 8) | pal
            let s2 = w::encode_slot(tag as u16, pal);
            assert_eq!(s1, s2, "tag={tag} pal={pal}: {s1} vs {s2}");
            // 打包进两 slot word（word 里两个 half），half=0 解 s1 和 wire
            let word_half0 = (s1 as u32) | 0xDEADu32 << 16;
            assert_eq!(wgsl_slot_unpack(word_half0, 0), s1);
            assert_eq!(w::unpack_slot_word(word_half0, 0), s1);
            let word_half1 = 0xBEEFu32 | (s1 as u32) << 16;
            assert_eq!(wgsl_slot_unpack(word_half1, 1), s1);
            assert_eq!(w::unpack_slot_word(word_half1, 1), s1);
            // slot_tag + slot_palette （wgst s1 >>8 和 &0xFF）
            assert_eq!((s1 >> 8) as u8, tag);
            assert_eq!((s1 & 0xFF) as u8, pal);
            assert_eq!(w::slot_tag(s2), tag as u16);
            assert_eq!(w::slot_palette(s2), pal);
        }
    }
}

// ============================================================================
// Task 3: CPU DDA 参考实现（独立 A&W step ，仅复用 BrickMapView::get_voxel 读 palette）
// ============================================================================

use crate::brickmap::{BrickMapBuffers, BrickMapView};

/// A&W 细格步进 DDA 参考实现（CPU）。
///
/// 完全新写 step（不共享 view.rs 结构）。仅调用 `BrickMapView::get_voxel(fine: IVec3)`
/// 进行 palette 查询。返回 `Some((hit_t, palette))` 或 `None`（t >= t_max 前未命中 / 超步）
///
/// 关键正确性约定：cell 坐标用整数增量维护（floor(origin) 起步，每次穿越 +sign），
/// **不得**用 `floor(origin + dir*t)` 重算——t 恰好是边界穿越时刻时 pos 分量正好落在
/// 整数边界上，floor 会随机取到穿越前/后的胞，导致对角胞漏检（DDA vs brute 不等价）。
pub fn cpu_reference_dda_ray(
    buffers: &BrickMapBuffers,
    origin_fine: Vec3,
    dir_fine: Vec3, // fine units（归一化），magnitude 任意（delta 按 |dir| 缩放）
    t_max: f32,
    max_steps: u32,
) -> Option<(f32, u8)> {
    let view = BrickMapView::new(buffers);
    let mut t = 0.0f32;
    // init A&W 变量
    let sign = [
        if dir_fine.x >= 0.0 { 1 } else { -1 },
        if dir_fine.y >= 0.0 { 1 } else { -1 },
        if dir_fine.z >= 0.0 { 1 } else { -1 },
    ];
    let delta = [
        if dir_fine.x.abs() > 1e-30 {
            (1.0 / dir_fine.x).abs()
        } else {
            f32::INFINITY
        },
        if dir_fine.y.abs() > 1e-30 {
            (1.0 / dir_fine.y).abs()
        } else {
            f32::INFINITY
        },
        if dir_fine.z.abs() > 1e-30 {
            (1.0 / dir_fine.z).abs()
        } else {
            f32::INFINITY
        },
    ];
    let mut cell = [
        origin_fine.x.floor() as i32,
        origin_fine.y.floor() as i32,
        origin_fine.z.floor() as i32,
    ];
    let next_boundary = |c: i32, s: i32| -> f32 {
        // s=1 -> 下一个上界 (c+1).0; s=-1 -> 当前下界 c.0（负数 floor 刚好也是下一个朝向的边界）
        (if s >= 0 { c + 1 } else { c }) as f32
    };
    let tmax_x = if dir_fine.x.abs() <= 1e-30 {
        f32::INFINITY
    } else {
        (next_boundary(cell[0], sign[0]) - origin_fine.x) / dir_fine.x
    };
    let tmax_y = if dir_fine.y.abs() <= 1e-30 {
        f32::INFINITY
    } else {
        (next_boundary(cell[1], sign[1]) - origin_fine.y) / dir_fine.y
    };
    let tmax_z = if dir_fine.z.abs() <= 1e-30 {
        f32::INFINITY
    } else {
        (next_boundary(cell[2], sign[2]) - origin_fine.z) / dir_fine.z
    };
    let mut tmax = [tmax_x, tmax_y, tmax_z];

    // 初始胞采样（整数 cell，精确）
    if let Some(pal) = view.get_voxel(IVec3::from_array(cell)) {
        return Some((t, pal));
    }
    for _ in 0..max_steps {
        if t >= t_max {
            return None;
        }
        // 走最小分量；cell 整数增量步进（边界精确，无浮点 floor 漏检）
        if tmax[0] <= tmax[1] && tmax[0] <= tmax[2] {
            t = tmax[0];
            tmax[0] += delta[0];
            cell[0] += sign[0];
        } else if tmax[1] <= tmax[2] {
            t = tmax[1];
            tmax[1] += delta[1];
            cell[1] += sign[1];
        } else {
            t = tmax[2];
            tmax[2] += delta[2];
            cell[2] += sign[2];
        }
        if let Some(pal) = view.get_voxel(IVec3::from_array(cell)) {
            return Some((t, pal));
        }
    }
    None
}

/// 带 AABB 跳步的参考 DDA：先求射线 (origin + t·dir, t∈[0, t_max]) 与给定 AABB
/// [aabb_min, aabb_max] 的相交段，若不相交直接 None；否则把 DDA 起点推进到
/// AABB 入口再开始走（跳过 origin→AABB 之间的空胞）。
///
/// **设计要点（之前 WGSL 版踩过的坑，这里用单测锁死）**：
///   1. 用「相对标尺」：所有 t/tmax 都自"新起点 start = origin + dir·t_enter"量起。
///      不混用自 origin 的"全局 t"与自 start 的"相对 t"——这是之前两版的根因。
///   2. 新 t_max = min(t_exit, 全局上限) - t_enter（相对区间长度）。
///   3. start = origin + dir·t_enter（浮点），DDA 对它 floor 得起始胞；
///      tmax_* = (next_boundary(cell_*, sign_*) - start_*) / dir_*（仍相对 start）。
///   4. 返回的 hit_t 要加 t_enter 还原为"自 origin 量起"的全局距离（供比对使用）。
///
/// 与原版 cpu_reference_dda_ray 逐射线命中等价（单测 `aabb_skip_equivalence` 验证）。
pub fn cpu_reference_dda_ray_aabb_skip(
    buffers: &BrickMapBuffers,
    origin: Vec3,
    dir: Vec3,         // 归一化（AABB 求交不要求归一化，但这里统一用归一与原函数对齐）
    t_global_max: f32, // 自 origin 量起的全局上限（= frustum_length）
    max_steps: u32,
    aabb_min: Vec3,
    aabb_max: Vec3,
) -> Option<(f32, u8)> {
    // ---- 1) slab 法求射线与 AABB 的 t ∈ [t_enter, t_exit]（都自 origin 量起）----
    let mut t_enter = 0.0f32;
    let mut t_exit = t_global_max;
    let mut miss = false;
    for axis in 0..3 {
        let o = [origin.x, origin.y, origin.z][axis];
        let d = [dir.x, dir.y, dir.z][axis];
        let mn = [aabb_min.x, aabb_min.y, aabb_min.z][axis];
        let mx = [aabb_max.x, aabb_max.y, aabb_max.z][axis];
        if d.abs() < 1e-30 {
            // 轴平行：origin 分量必须 ∈ [mn, mx]（闭区间）才算通过
            if o < mn || o > mx {
                miss = true;
            }
        } else {
            let t1 = (mn - o) / d;
            let t2 = (mx - o) / d;
            let lo = t1.min(t2);
            let hi = t1.max(t2);
            t_enter = t_enter.max(lo);
            t_exit = t_exit.min(hi);
        }
    }
    if miss || t_exit < t_enter.max(0.0) {
        return None; // 与 AABB 完全不相交（含视锥段内不相交）
    }
    // 约束到视锥有效段
    let t_enter = t_enter.max(0.0);
    let t_exit = t_exit.min(t_global_max);
    if t_exit <= t_enter {
        return None; // 厚度为 0（擦边）或完全在视锥外
    }

    // ---- 2) 起点推进到 start = origin + dir·t_enter；后续全用「相对标尺」----
    let start = origin + dir * t_enter;
    // 相对 max t：从 start 到 t_exit（自 origin）的剩余长度
    let t_rel_max = t_exit - t_enter;

    let view = BrickMapView::new(buffers);
    let sign = [
        if dir.x >= 0.0 { 1 } else { -1 },
        if dir.y >= 0.0 { 1 } else { -1 },
        if dir.z >= 0.0 { 1 } else { -1 },
    ];
    let delta = [
        if dir.x.abs() > 1e-30 {
            (1.0 / dir.x).abs()
        } else {
            f32::INFINITY
        },
        if dir.y.abs() > 1e-30 {
            (1.0 / dir.y).abs()
        } else {
            f32::INFINITY
        },
        if dir.z.abs() > 1e-30 {
            (1.0 / dir.z).abs()
        } else {
            f32::INFINITY
        },
    ];
    let next_boundary = |c: i32, s: i32| -> f32 { (if s >= 0 { c + 1 } else { c }) as f32 };
    let mut cell = [
        start.x.floor() as i32,
        start.y.floor() as i32,
        start.z.floor() as i32,
    ];
    // tmax 分量：相对 start 的距离
    let tmax_x = if dir.x.abs() <= 1e-30 {
        f32::INFINITY
    } else {
        (next_boundary(cell[0], sign[0]) - start.x) / dir.x
    };
    let tmax_y = if dir.y.abs() <= 1e-30 {
        f32::INFINITY
    } else {
        (next_boundary(cell[1], sign[1]) - start.y) / dir.y
    };
    let tmax_z = if dir.z.abs() <= 1e-30 {
        f32::INFINITY
    } else {
        (next_boundary(cell[2], sign[2]) - start.z) / dir.z
    };
    let mut tmax = [tmax_x, tmax_y, tmax_z];

    // 初始胞采样（start 对应的胞；floor 过）
    if let Some(pal) = view.get_voxel(IVec3::from_array(cell)) {
        return Some((t_enter, pal)); // 全局 t = t_enter + 0（相对）
    }
    let mut t_rel = 0.0f32;
    for _ in 0..max_steps {
        if t_rel >= t_rel_max {
            return None;
        }
        if tmax[0] <= tmax[1] && tmax[0] <= tmax[2] {
            t_rel = tmax[0];
            tmax[0] += delta[0];
            cell[0] += sign[0];
        } else if tmax[1] <= tmax[2] {
            t_rel = tmax[1];
            tmax[1] += delta[1];
            cell[1] += sign[1];
        } else {
            t_rel = tmax[2];
            tmax[2] += delta[2];
            cell[2] += sign[2];
        }
        if let Some(pal) = view.get_voxel(IVec3::from_array(cell)) {
            return Some((t_enter + t_rel, pal));
        }
    }
    None
}

/// 细级：单个粗 cell（16³ fine）内的有界 fine DDA。
/// 射线段限制在 [t_lo, t_hi]（该 cell 的入出区间，自 origin 全局标尺）。
///
/// 入口 fine 胞用「解析 + clamp」确定：p = origin + dir·t_lo 落在 cell 入口面上时，
/// floor(p) 可能因浮点误差取到邻胞——clamp 到 [base, base+15] 保证起点一定在本 cell 内。
/// clamp 不会漏检：入口面外侧的最后一个 fine 胞属于前一个粗 cell，其细扫已覆盖。
/// 采样序列与 full DDA 在同区间的序列逐胞一致（fine 边界与粗边界 16 对齐）。
#[allow(clippy::too_many_arguments)] // cell 局部细扫的固有参数面（view+射线+cell 窗口）
fn dda_fine_scan_cell(
    view: &BrickMapView,
    origin: Vec3,
    dir: Vec3,
    sign: [i32; 3],
    delta: [f32; 3], // fine 步距（|dir| 分量倒数或 INF）
    cc: [i32; 3],    // 粗 cell 坐标（1 单位 = 16 fine）
    t_lo: f32,
    t_hi: f32,
) -> Option<(f32, u8)> {
    if t_hi <= t_lo {
        return None;
    }
    let p = origin + dir * t_lo;
    let pc = [p.x, p.y, p.z];
    let base = [cc[0] << 4, cc[1] << 4, cc[2] << 4];
    let mut fc = [0i32; 3];
    for i in 0..3 {
        fc[i] = (pc[i].floor() as i32).clamp(base[i], base[i] + 15);
    }
    // fine tmax：相对 t_lo 的距离（自 cell 入口重算，非累加——与 full 的 ulp 差异见主函数注释）
    let mut tmax_f = [f32::INFINITY; 3];
    for i in 0..3 {
        if dir[i].abs() > 1e-30 {
            let bnd = (fc[i] + if sign[i] >= 0 { 1 } else { 0 }) as f32;
            let t = (bnd - pc[i]) / dir[i];
            tmax_f[i] = if t < 0.0 { 0.0 } else { t };
        }
    }
    let span = t_hi - t_lo;
    let mut t_f = 0.0f32;
    // 初始 fine 胞采样
    if let Some(pal) = view.get_voxel(IVec3::from_array(fc)) {
        return Some((t_lo, pal));
    }
    // 48 = 3 轴 × 16：斜穿 16³ cell 的步数上界，几何上必在 span 内退出
    for _ in 0..48 {
        if t_f >= span {
            return None;
        }
        if tmax_f[0] <= tmax_f[1] && tmax_f[0] <= tmax_f[2] {
            t_f = tmax_f[0];
            tmax_f[0] += delta[0];
            fc[0] += sign[0];
        } else if tmax_f[1] <= tmax_f[2] {
            t_f = tmax_f[1];
            tmax_f[1] += delta[1];
            fc[1] += sign[1];
        } else {
            t_f = tmax_f[2];
            tmax_f[2] += delta[2];
            fc[2] += sign[2];
        }
        if let Some(pal) = view.get_voxel(IVec3::from_array(fc)) {
            return Some((t_lo + t_f, pal));
        }
    }
    None
}

/// 两级 DDA 参考实现（cell 粗步 + cell 内细步）。
///
/// 语义与 `cpu_reference_dda_ray` 完全一致：射线 (origin, dir, t∈[0,t_max]) 上
/// 首个非空 fine 体素，返回 (全局 t, palette) 或 None。
///
/// 结构（WGSL dda_main 逐字对应的源）：
///   1. 粗级 A&W：cell 粒度（16 fine）步进，delta_c = delta × 16（f32 乘 2 的幂，精确）；
///      每步先做 `BrickMapView::cell_occupied`（①+②，2 次 load），空 cell 整段跳过。
///   2. 细级：占用 cell 内 `dda_fine_scan_cell`，区间 [t_in, min(t_out, t_max)]。
///   3. 退出条件：t_out ≥ t_max（cell 出口越过上限）或粗步数耗尽。
///
/// 与 full 版的已知数值差异（两级等价性单测以 1.0 fine 容差覆盖，命中体素/palette 严格一致）：
///   - full 逐 fine 累加 tmax += delta；两级粗级按 delta_c = 16·delta 累加、细级自 cell
///     入口重算——长路径 ulp 漂移可达 ~0.1 fine，只影响返回 t 值，不影响命中胞序
///     （胞序由边界穿越的整数序决定，仅角点精确相切时才可能翻转，测度零）。
pub fn cpu_reference_dda_ray_two_level(
    buffers: &BrickMapBuffers,
    origin_fine: Vec3,
    dir_fine: Vec3, // fine units（归一化），magnitude 任意（delta 按 |dir| 缩放）
    t_max: f32,
    max_steps: u32,
) -> Option<(f32, u8)> {
    let view = BrickMapView::new(buffers);
    let o = [origin_fine.x, origin_fine.y, origin_fine.z];
    let d = [dir_fine.x, dir_fine.y, dir_fine.z];
    let sign = [
        if dir_fine.x >= 0.0 { 1 } else { -1 },
        if dir_fine.y >= 0.0 { 1 } else { -1 },
        if dir_fine.z >= 0.0 { 1 } else { -1 },
    ];
    let delta = [
        if d[0].abs() > 1e-30 {
            (1.0 / d[0]).abs()
        } else {
            f32::INFINITY
        },
        if d[1].abs() > 1e-30 {
            (1.0 / d[1]).abs()
        } else {
            f32::INFINITY
        },
        if d[2].abs() > 1e-30 {
            (1.0 / d[2]).abs()
        } else {
            f32::INFINITY
        },
    ];
    // 粗 delta = fine delta × 16（精确）
    let delta_c = [delta[0] * 16.0, delta[1] * 16.0, delta[2] * 16.0];
    // 粗 cell：floor(origin) >> 4（算术右移 = floor 除法，负坐标正确）
    let mut cc = [
        (o[0].floor() as i32) >> 4,
        (o[1].floor() as i32) >> 4,
        (o[2].floor() as i32) >> 4,
    ];
    // 粗 tmax：到下一个粗边界的距离（相对 origin，同 full 版公式，边界 ×16）
    let mut tmax_c = [f32::INFINITY; 3];
    for i in 0..3 {
        if d[i].abs() > 1e-30 {
            let bnd = ((cc[i] + if sign[i] >= 0 { 1 } else { 0 }) << 4) as f32;
            let t = (bnd - o[i]) / d[i];
            tmax_c[i] = if t < 0.0 { 0.0 } else { t };
        }
    }
    let mut t_in = 0.0f32; // 当前粗 cell 的入口 t（初始 cell = 0）
    for _ in 0..max_steps {
        let t_out = tmax_c[0].min(tmax_c[1]).min(tmax_c[2]);
        // 占用查询先行：空 cell 不做任何 fine 采样（性能核心）
        if view.cell_occupied(IVec3::from_array(cc))
            && let Some(r) = dda_fine_scan_cell(
                &view,
                origin_fine,
                dir_fine,
                sign,
                delta,
                cc,
                t_in,
                t_out.min(t_max),
            )
        {
            return Some(r);
        }
        if t_out >= t_max {
            return None;
        }
        // 粗级步进（整数增量，同 full 版约定）
        if tmax_c[0] <= tmax_c[1] && tmax_c[0] <= tmax_c[2] {
            t_in = tmax_c[0];
            tmax_c[0] += delta_c[0];
            cc[0] += sign[0];
        } else if tmax_c[1] <= tmax_c[2] {
            t_in = tmax_c[1];
            tmax_c[1] += delta_c[1];
            cc[1] += sign[1];
        } else {
            t_in = tmax_c[2];
            tmax_c[2] += delta_c[2];
            cc[2] += sign[2];
        }
    }
    None
}

/// 对固定场景 & 静态 DdaCameraConfig，按 32x32 网格渲染 ASCII 画（用于 AC-4 对照实机截图）
///
/// 返回 `Vec<char>`（32×32，row-major 32 字符换行）。
/// 字符规则：palette=1 → 'X', 2 → 'o', 3 → '#', 4 → '*', 其他非 0 → '+', 空 → '.'
pub fn cpu_dda_ascii_grid_32x32(cfg: &DdaCameraConfig, buffers: &BrickMapBuffers) -> Vec<char> {
    let w = 32usize;
    let h = 32usize;
    let mut out = Vec::with_capacity(w * h);
    let inv_vp = cfg.inv_view_proj;
    let cam_pos_fine = cfg.position_world; // 世界单位 = fine 单位
    for y in 0..h {
        for x in 0..w {
            // NDC：像素中心 (x+0.5, y+0.5)/size → 2u-1, 1-2v
            let u = ((x as f32 + 0.5) / w as f32) * 2.0 - 1.0;
            let v = 1.0 - ((y as f32 + 0.5) / h as f32) * 2.0;
            let near = inv_vp * Vec4::new(u, v, 0.0, 1.0);
            let far = inv_vp * Vec4::new(u, v, 1.0, 1.0);
            let near = near.truncate() / near.w;
            let far = far.truncate() / far.w;
            let dir_world = (far - near).normalize();
            let dir_fine = dir_world;
            let ch = match cpu_reference_dda_ray(buffers, cam_pos_fine, dir_fine, 2000.0, 2048) {
                None => '.',
                Some((_, 1)) => 'X',
                Some((_, 2)) => 'o',
                Some((_, 3)) => '#',
                Some((_, 4)) => '*',
                Some((_, _)) => '+',
            };
            out.push(ch);
        }
    }
    out
}

#[cfg(test)]
mod dda_ref_tests {
    use super::*;
    use crate::brickmap::BrickMapBuilder;
    use gate_voxel::{TileGrid, fill_box, fill_sphere};
    use glam::IVec3;

    /// 确定性 xorshift64（不用引入 rand crate，50 射线 + 200 打包足够）
    fn xorshift64(state: &mut u64) -> u32 {
        let mut x = *state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *state = x;
        (x & 0xFFFF_FFFF) as u32
    }

    /// 0..1 f32
    fn frand(state: &mut u64) -> f32 {
        (xorshift64(state) as f32) / (u32::MAX as f32)
    }

    /// 测试专用相机：小场景（fine 0..16）配近相机，与 demo 的 build_static 解耦
    fn test_cam() -> DdaCameraConfig {
        let eye = Vec3::new(24.0, 20.0, 24.0);
        let target = Vec3::new(8.0, 8.0, 8.0);
        let aspect = VIEW_SIZE.x as f32 / VIEW_SIZE.y as f32;
        let proj = Mat4::perspective_rh(60.0_f32.to_radians(), aspect, 1.0, 4000.0);
        let view = Mat4::look_at_rh(eye, target, Vec3::Y);
        let view_proj = proj.mul(view);
        DdaCameraConfig {
            view_proj,
            inv_view_proj: view_proj.inverse(),
            position_world: eye,
        }
    }

    fn build_box_sphere_scene() -> (BrickMapBuffers, DdaCameraConfig) {
        use gate_voxel::coords::MAX_LEVEL;
        let mut g = TileGrid::new();
        // box 16³ fine → 0..16 (最细 level=MAX_LEVEL) → pal=1
        fill_box(&mut g, IVec3::ZERO, IVec3::splat(16), MAX_LEVEL, 1);
        // sphere at (8,8,8) fine, r=4 fine (1cm), level=MAX_LEVEL pal=2
        fill_sphere(&mut g, IVec3::new(8, 8, 8), 4, MAX_LEVEL, 2);
        // hotspots pal 3+4: 1x1x1 at (12, 2, 12) MAX_LEVEL pal3 and (14,2,14) pal4
        g.set_voxel(IVec3::new(12, 2, 12), MAX_LEVEL, 3);
        g.set_voxel(IVec3::new(14, 2, 14), MAX_LEVEL, 4);
        let b = BrickMapBuilder::build_full(&g);
        (b.buffers().clone(), test_cam())
    }

    /// 50 条随机射线：cpu_reference_dda_ray 和 "独立枚举细格" 双方法结果一致
    /// （"独立枚举" = 不调用 BrickMapView.get_voxel 而是逐位置 floor 枚举 get_voxel，
    ///  实际上 DDA ref 实现已经用 view.get_voxel，第二验证方式是暴力从 t=0 到 t 按 0.5 fine 扫）
    #[test]
    fn dda_reference_equivalence_50_rays() {
        let (bufs, cfg) = build_box_sphere_scene();
        let mut state: u64 = 42;
        let origin = cfg.position_world;
        for _ in 0..50 {
            // 方向：在视锥内的随机像素（-1..1 NDC）
            let u = frand(&mut state) * 2.0 - 1.0;
            let v = frand(&mut state) * 2.0 - 1.0;
            let inv_vp = cfg.inv_view_proj;
            let near = inv_vp * Vec4::new(u, v, 0.0, 1.0);
            let far = inv_vp * Vec4::new(u, v, 1.0, 1.0);
            let near = near.truncate() / near.w;
            let far = far.truncate() / far.w;
            let dir_world = (far - near).normalize();
            let dir_fine = dir_world; // 世界单位 = fine 单位，dir 模长 1
            let a = cpu_reference_dda_ray(&bufs, origin, dir_fine, 2000.0, 2048);
            // Brute-force：以 0.5 fine 实际距离扫 2000 步；取首个非空
            // dir_fine.length() = 1，t = 实际 fine 距离
            let d_len = dir_fine.length().max(1e-20);
            let step_fine = 0.5f32; // 0.5 fine 步长保证 1-fine 胞至少 2 采样
            let mut brute: Option<(f32, u8)> = None;
            let view = BrickMapView::new(&bufs);
            for i in 0..2000u32 {
                let s = i as f32 * step_fine; // 实际 fine 距离
                if s / d_len >= 2000.0 {
                    break;
                }
                let t = s / d_len;
                let p = origin + dir_fine * t;
                let fine = IVec3::new(p.x.floor() as i32, p.y.floor() as i32, p.z.floor() as i32);
                if let Some(pal) = view.get_voxel(fine) {
                    brute = Some((t, pal));
                    break;
                }
            }
            match (a, brute) {
                (Some((ta, pa)), Some((tb, pb))) => {
                    assert_eq!(pa, pb, "palette mismatch");
                    // 0.5 fine coarse vs A&W 精确边界：ta 通常在 (tb-1, tb+0.5) 区间
                    assert!((ta - tb).abs() <= 1.0, "t mismatch {ta} vs {tb}");
                }
                (None, None) => {}
                (a, b) => {
                    panic!("hit pattern diff: a={a:?} b={b:?} origin={origin:?} dir={dir_fine:?}")
                }
            }
        }
    }

    /// 32x32 ASCII 网格：基线验证（首行/末行不应有全点，中心区域应有 'X' 盒体）
    #[test]
    fn cpu_dda_ascii_grid_32_has_expected_pattern() {
        let (bufs, cfg) = build_box_sphere_scene();
        // Debug: 直接查 BrickMapView 在代表性坐标的 palette
        let view = BrickMapView::new(&bufs);
        let probes = [
            (IVec3::new(0, 0, 0), "box-surface-x0-y0-z0 pal1"),
            (IVec3::new(8, 8, 8), "sphere-center pal2"),
            (IVec3::new(12, 2, 12), "hotspot pal3"),
            (IVec3::new(14, 2, 14), "hotspot pal4"),
            (IVec3::new(0, 8, 8), "box-x0-face pal1"),
        ];
        for (p, msg) in probes {
            println!(
                "probe get_voxel({:?}) {:?} = {:?}",
                p,
                msg,
                view.get_voxel(p)
            );
        }
        let grid = cpu_dda_ascii_grid_32x32(&cfg, &bufs);
        let grid_w = 32usize;
        // 先打印再断言（断言失败时也能看到输出）
        for row in 0..32 {
            let line: String = grid.iter().skip(row * grid_w).take(grid_w).collect();
            println!("{:02} [{}]", row, line);
        }
        // 统计字符分布
        use std::collections::HashMap;
        let mut counts: HashMap<char, usize> = HashMap::new();
        for c in &grid {
            *counts.entry(*c).or_default() += 1;
        }
        println!("COUNTS = {counts:?}");
        // 至少应有 32 个以上的非空（'.'）字符
        let non_empty: usize = grid.iter().filter(|c| **c != '.').count();
        assert!(
            non_empty >= 32,
            "non_empty pixels only {non_empty}（<32，画面全空可疑）"
        );
        // X 字符（盒体 pal1）≥ 8
        let xs: usize = grid.iter().filter(|c| **c == 'X').count();
        assert!(xs >= 8, "box pal=1 only {xs} 'X'");
    }

    /// 300 条随机射线（含远镜头 / 近镜头 / 相机在 AABB 内 / 轴平行退化情形）：
    /// cpu_reference_dda_ray（从 origin 直接走，足够多 steps）和
    /// cpu_reference_dda_ray_aabb_skip（先 AABB 跳步）两者结果逐像素一致。
    ///
    /// 此单测通过后，WGSL 翻译版可以按 `cpu_reference_dda_ray_aabb_skip` 的
    /// `相对标尺`算法逐字移植，保证正确性。
    #[test]
    fn aabb_skip_equivalence_300_rays() {
        let (bufs, _cfg) = build_box_sphere_scene();
        // 场景 AABB：fine 坐标 [0, 0, 0] .. [16, 16, 16]（brickmap 全在此区间）
        let aabb_min = Vec3::new(0.0, 0.0, 0.0);
        let aabb_max = Vec3::new(16.0, 16.0, 16.0);
        let mut state: u64 = 12345;
        let mut count = 0usize;
        // 5 种相机场景，每种 60 条随机视锥射线
        let cameras: Vec<(Vec3, Vec3)> = vec![
            // 近相机（在 AABB 外 24 fine 处，和 test_cam 一致）
            (Vec3::new(24.0, 20.0, 24.0), Vec3::new(8.0, 8.0, 8.0)),
            // 远相机（在 AABB 外 50k fine 处——模拟用户 zoom-out 很多下的情况）
            (
                Vec3::new(50_000.0, 40_000.0, 50_000.0),
                Vec3::new(8.0, 8.0, 8.0),
            ),
            // 相机在 AABB 内部（用户很近时贴脸模型）
            (Vec3::new(8.5, 8.5, 2.0), Vec3::new(8.5, 8.5, 16.0)),
            // 斜 + 轻微轴平行（xz 面视线，dir.y 很小）
            (Vec3::new(30.0, 8.0, 30.0), Vec3::new(8.0, 8.0, 8.0)),
            // 负坐标远端 + 指向 AABB（测试负 fine 坐标 floor 与 slab 求交）
            (
                Vec3::new(-10_000.0, 10_000.0, -10_000.0),
                Vec3::new(8.0, 8.0, 8.0),
            ),
        ];
        for (eye, target) in &cameras {
            for _ in 0..60 {
                let u = frand(&mut state) * 2.0 - 1.0;
                let v = frand(&mut state) * 2.0 - 1.0;
                // 手搓 look_at + perspective，不依赖 DdaCameraConfig
                let aspect = 16.0 / 9.0;
                let proj = Mat4::perspective_rh(60.0_f32.to_radians(), aspect, 1.0, 1_000_000.0);
                let view = Mat4::look_at_rh(*eye, *target, Vec3::Y);
                let inv_vp = proj.mul(view).inverse();
                let near = inv_vp * Vec4::new(u, v, 0.0, 1.0);
                let far = inv_vp * Vec4::new(u, v, 1.0, 1.0);
                let near = near.truncate() / near.w;
                let far = far.truncate() / far.w;
                let diff = far - near;
                let frustum_len = diff.length();
                let dir = diff.normalize();
                // 原版 DDA：给足够大的 steps = 2_000_000（覆盖远镜头 50k fine 的距离），
                // t_max = frustum_len（视锥外不算命中）。max_steps 足够大所以一定能走穿。
                let full = cpu_reference_dda_ray(&bufs, *eye, dir, frustum_len, 2_000_000);
                // AABB skip 版：max_steps 只给 2048（GPU 最终上限，比 AABB 对角 ~1732 略大），
                // 理论上只要 AABB 在视锥内、2048 一定能走穿整个 AABB 厚度内的片段。
                let skip = cpu_reference_dda_ray_aabb_skip(
                    &bufs,
                    *eye,
                    dir,
                    frustum_len,
                    2048,
                    aabb_min,
                    aabb_max,
                );
                match (full, skip) {
                    (Some((tf, pf)), Some((ts, ps))) => {
                        assert_eq!(
                            pf, ps,
                            "palette diff eye={eye:?} tgt={target:?} full_t={tf} skip_t={ts}"
                        );
                        // t 差 ≤ 1 fine（浮点计算的 floor 与首胞命中边界误差）
                        assert!(
                            (tf - ts).abs() <= 1.0,
                            "t diff eye={eye:?} full_t={tf} skip_t={ts} pal={pf}"
                        );
                    }
                    (None, None) => {}
                    (f, s) => panic!(
                        "hit mismatch eye={eye:?} tgt={target:?}\n  full = {f:?}\n  skip = {s:?}\n  frustum={frustum_len}"
                    ),
                }
                count += 1;
            }
        }
        assert_eq!(count, 300, "ray count mismatch {count}");
    }

    /// 两级 DDA 等价性：多物体场景（多 cell 体 + cell 间隙 + 负坐标）+ 固定退化方向 +
    /// 300 条随机射线，cpu_reference_dda_ray vs cpu_reference_dda_ray_two_level。
    /// 命中/未命中与 palette 严格一致；hit_t 容差 1.0 fine（同 aabb_skip 先例：
    /// full 累加 tmax vs 两级粗/细分别重算，长路径 ulp 漂移只影响 t 值不影响命中胞）。
    #[test]
    fn two_level_equivalence_300_rays() {
        // ---- 多物体场景（一个 tile 内）：外框盒 + 内浮球 + 独立柱，cell 间隙跨越 ----
        let mut g = TileGrid::new();
        use gate_voxel::coords::MAX_LEVEL;
        // 外框壳：32³ box（2×2×2 cells）
        fill_box(&mut g, IVec3::ZERO, IVec3::splat(32), MAX_LEVEL, 1);
        // 独立物体：偏移 (96, 32, 96) 的 16³ box（离开外框，中间隔空 cell）
        fill_box(
            &mut g,
            IVec3::new(96, 32, 96),
            IVec3::splat(16),
            MAX_LEVEL,
            2,
        );
        // 悬浮球：中心 (64, 64, 64)，r=8（跨 cell 边界）
        fill_sphere(&mut g, IVec3::new(64, 64, 64), 8, MAX_LEVEL, 3);
        let bufs = BrickMapBuilder::build_full(&g).buffers().clone();

        // ---- 固定退化方向：±轴平行穿物体中心 / 擦角 / 负坐标远端 ----
        let axis_rays: [(Vec3, Vec3); 8] = [
            (Vec3::new(-64.0, 16.0, 16.0), Vec3::X),   // -x 起步穿外框
            (Vec3::new(112.0, 40.0, 104.0), -Vec3::X), // +x 侧穿柱体
            (Vec3::new(16.0, -32.0, 16.0), Vec3::Y),   // -y 穿外框底
            (Vec3::new(16.0, 128.0, 16.0), -Vec3::Y),  // +y 上方穿外框顶
            (Vec3::new(16.0, 16.0, -16.0), Vec3::Z),   // -z 穿外框
            (Vec3::new(64.0, 64.0, -32.0), -Vec3::Z),  // +z 穿悬浮球心
            (Vec3::new(-256.0, -256.0, -256.0), Vec3::ONE), // 负坐标远端对角穿
            (Vec3::new(200.0, 200.0, 200.0), -Vec3::ONE), // 正远端对角（大概率 miss）
        ];
        for (i, (o, d)) in axis_rays.iter().enumerate() {
            let full = cpu_reference_dda_ray(&bufs, *o, *d, 4096.0, 2_000_000);
            let two = cpu_reference_dda_ray_two_level(&bufs, *o, *d, 4096.0, 16384);
            match (full, two) {
                (Some((tf, pf)), Some((tt, pt))) => {
                    assert_eq!(pf, pt, "axis[{i}] palette diff full_t={tf} two_t={tt}");
                    assert!((tf - tt).abs() <= 1.0, "axis[{i}] t diff {tf} vs {tt}");
                }
                (None, None) => {}
                (f, t) => {
                    panic!("axis[{i}] hit mismatch o={o:?} d={d:?}\n  full={f:?}\n  two={t:?}")
                }
            }
        }

        // ---- 300 条随机射线：球形壳内随机起点 + 随机方向（含大量纯空穿越） ----
        let mut state: u64 = 20260831;
        for ray in 0..300 {
            // 起点距场景中心 32~600 fine 的球壳
            let r = 32.0 + frand(&mut state) * 568.0;
            let theta = frand(&mut state) * std::f32::consts::TAU;
            let phi = (frand(&mut state) * 2.0 - 1.0).acos();
            let origin = Vec3::new(
                r * phi.sin() * theta.cos(),
                r * phi.cos(),
                r * phi.sin() * theta.sin(),
            ) + Vec3::new(64.0, 48.0, 64.0);
            // 随机单位方向（球面均匀）
            let dtheta = frand(&mut state) * std::f32::consts::TAU;
            let dphi = (frand(&mut state) * 2.0 - 1.0).acos();
            let dir = Vec3::new(
                dphi.sin() * dtheta.cos(),
                dphi.cos(),
                dphi.sin() * dtheta.sin(),
            );
            let full = cpu_reference_dda_ray(&bufs, origin, dir, 2048.0, 2_000_000);
            let two = cpu_reference_dda_ray_two_level(&bufs, origin, dir, 2048.0, 16384);
            match (full, two) {
                (Some((tf, pf)), Some((tt, pt))) => {
                    assert_eq!(
                        pf, pt,
                        "ray[{ray}] palette diff o={origin:?} d={dir:?} full_t={tf} two_t={tt}"
                    );
                    assert!(
                        (tf - tt).abs() <= 1.0,
                        "ray[{ray}] t diff {tf} vs {tt} o={origin:?} d={dir:?}"
                    );
                }
                (None, None) => {}
                (f, t) => panic!(
                    "ray[{ray}] hit mismatch o={origin:?} d={dir:?}\n  full={f:?}\n  two={t:?}"
                ),
            }
        }
    }
}

// ============================================================================
// Task 5: BrickMapDdaPlugin — 完整 pipeline/BG/dispatch/blit 装配
// 严格复用 gradient.rs scaffold（ExtractResourcePlugin → RenderStartup →
// PrepareBindGroups → RenderGraph dispatch 前置 → Core2d PostProcess blit），
// 仅不同：BG1 新增 brickmap 四 buffer（struct/leaves/palette + globals uniform），
// 以及 BG0 uniform 用 DdaViewUniform（DdaCameraConfig main-world Extract → 写）
// ============================================================================
use bevy::{
    core_pipeline::schedule::{Core2d, Core2dSystems, camera_driver},
    render::{
        Render, RenderApp, RenderStartup, RenderSystems,
        diagnostic::RecordDiagnostics,
        render_asset::RenderAssets,
        render_resource::{
            BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries,
            CachedComputePipelineId, CachedRenderPipelineId, ColorTargetState, ColorWrites,
            ComputePassDescriptor, ComputePipelineDescriptor, FragmentState, PipelineCache,
            RenderPassDescriptor, ShaderStages, StorageTextureAccess, TextureFormat,
            TextureSampleType, UniformBuffer, VertexState,
            binding_types::{
                storage_buffer_read_only_sized, texture_2d, texture_storage_2d, uniform_buffer,
            },
        },
        renderer::{RenderContext, RenderDevice, RenderQueue},
        texture::GpuImage,
        view::ViewTarget,
    },
};

use std::borrow::Cow;

use super::upload::GpuBrickMap;
use super::mov::GpuMovPool;
use crate::gradient::{BLIT_SHADER_ASSET_PATH, WORKGROUP_SIZE};

pub const DDA_SHADER_ASSET_PATH: &str = "shaders/dda.wgsl";

// --- DdaImages：main-world 创建的 handle，ExtractResource 自动传到 render world（main.rs setup 注入） ---
// （类型定义在 L110 附近，#[derive(Resource, Clone, ExtractResource)]）
// 这里再用 use 明确
#[derive(Resource)]
struct DdaBg0BindGroup(BindGroup);
#[derive(Resource)]
struct DdaBg1BindGroup(BindGroup);
#[derive(Resource)]
struct DdaBg2BindGroup(BindGroup);
#[derive(Resource)]
struct DdaBlitBindGroup(BindGroup);

#[derive(Resource)]
struct DdaPipelines {
    bg0_layout: BindGroupLayoutDescriptor,
    bg1_layout: BindGroupLayoutDescriptor,
    bg2_layout: BindGroupLayoutDescriptor,
    blit_layout: BindGroupLayoutDescriptor,
    compute_pipeline: CachedComputePipelineId,
    blit_pipeline: CachedRenderPipelineId,
}

pub struct BrickMapDdaPlugin;

impl Plugin for BrickMapDdaPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((bevy::render::extract_resource::ExtractResourcePlugin::<
            DdaImages,
        >::default(),));

        // main → render 的 ExtractSchedule：把 DdaCameraConfig 从 main world 读
        // （main.rs setup 注入的 Resource）→ 转成 DdaViewUniform（render world 资源，
        // 供 PrepareBindGroups 每帧写 uniform buffer）
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .add_systems(bevy::render::ExtractSchedule, extract_camera_config)
            .add_systems(RenderStartup, init_dda_pipelines)
            .add_systems(
                Render,
                prepare_dda_bind_groups
                    .in_set(RenderSystems::PrepareBindGroups)
                    // 同帧先用新 pool 重建 BG2 再绑 DDA（版本不匹配时差一帧也可接受，但同帧更稳）
                    .after(super::mov::prepare_mov_pool),
            )
            // 必须挂 RenderGraph::Render set（而非 Render schedule）：Render schedule 整体在
            // RenderGraph 之前 → begin_diagnostics_frame（Begin set）前执行，诊断 span 会被清空
            .add_systems(
                RenderGraph,
                dispatch_dda
                    .in_set(bevy::render::renderer::RenderGraphSystems::Render)
                    .before(camera_driver),
            )
            .add_systems(
                Core2d,
                // 显式 after gradient blit：同一 set 内系统默认无序（并行），
                // 不加 ordering 时 gradient 可能后执行覆盖 DDA 画面（实机截图已复现）
                blit_dda_view
                    .in_set(Core2dSystems::PostProcess)
                    .after(crate::gradient::GradientBlitSet),
            );
    }
}

fn extract_camera_config(
    mut commands: bevy::ecs::system::Commands,
    // ExtractSchedule 运行在 render world：Res 读不到 main world 资源，
    // 必须用 Extract<Res<T>> 系统参数跨 world 读取（与 ExtractResourcePlugin 同机制）
    cfg: Option<bevy::render::Extract<bevy::ecs::system::Res<crate::brickmap::DdaCameraConfig>>>,
) {
    // 无配置（某些 test 场景）不 panic
    let Some(cfg) = cfg else { return };
    let uniform = DdaViewUniform::from_cfg(&cfg);
    commands.insert_resource(uniform);
}

fn init_dda_pipelines(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
    _render_device: Res<RenderDevice>,
) {
    // ---- BG0：out tex storage + DdaViewUniform uniform（Compute）----
    let bg0 = BindGroupLayoutDescriptor::new(
        "DdaBg0",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                texture_storage_2d(TextureFormat::Rgba8Unorm, StorageTextureAccess::WriteOnly),
                uniform_buffer::<DdaViewUniform>(false),
            ),
        ),
    );

    // ---- BG1：struct/leaves/palette 三 storage + globals uniform（Compute，read-only）----
    let bg1 = BindGroupLayoutDescriptor::new(
        "DdaBg1",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                // 运行时 sized：min_binding_size=None
                storage_buffer_read_only_sized(false, None), // @binding(0) b_struct
                storage_buffer_read_only_sized(false, None), // @binding(1) b_leaves
                storage_buffer_read_only_sized(false, None), // @binding(2) b_palette
                uniform_buffer::<super::wire::BrickMapGlobals>(false), // @binding(3) globals
            ),
        ),
    );

    // ---- BG2：MOV object pool（mov_struct/mov_leaves/mov_palette/descs 四 storage + count uniform）----
    let bg2 = BindGroupLayoutDescriptor::new(
        "DdaBg2",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                storage_buffer_read_only_sized(false, None), // @binding(0) mov_struct
                storage_buffer_read_only_sized(false, None), // @binding(1) mov_leaves
                storage_buffer_read_only_sized(false, None), // @binding(2) mov_palette
                storage_buffer_read_only_sized(false, None), // @binding(3) mov_descs
                uniform_buffer::<super::mov::MovGlobals>(false), // @binding(4) count
            ),
        ),
    );

    // ---- blit BG layout：与 Gradient 完全一致（texture_2d f32）----
    let blit = BindGroupLayoutDescriptor::new(
        "DdaBlit",
        &BindGroupLayoutEntries::single(
            ShaderStages::FRAGMENT,
            texture_2d(TextureSampleType::Float { filterable: false }),
        ),
    );

    // ---- Compute pipeline：dda.wgsl entry dda_main ----
    let dda_shader = asset_server.load(DDA_SHADER_ASSET_PATH);
    let compute = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some(Cow::from("gate_dda_compute")),
        layout: vec![bg0.clone(), bg1.clone(), bg2.clone()],
        shader: dda_shader,
        entry_point: Some(Cow::from("dda_main")),
        ..default()
    });

    // ---- Blit render pipeline：复用 blit.wgsl（Gradient 用的同一份 WGSL，不同 BG layout handle）----
    let blit_shader = asset_server.load(BLIT_SHADER_ASSET_PATH);
    let blit_pipeline = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some(Cow::from("gate_dda_blit")),
        layout: vec![blit.clone()],
        vertex: VertexState {
            shader: blit_shader.clone(),
            entry_point: Some(Cow::from("vs_main")),
            ..default()
        },
        fragment: Some(FragmentState {
            shader: blit_shader,
            entry_point: Some(Cow::from("fs_main")),
            targets: vec![Some(ColorTargetState {
                format: TextureFormat::Rgba8UnormSrgb,
                blend: None,
                write_mask: ColorWrites::ALL,
            })],
            ..default()
        }),
        ..default()
    });

    commands.insert_resource(DdaPipelines {
        bg0_layout: bg0,
        bg1_layout: bg1,
        bg2_layout: bg2,
        blit_layout: blit,
        compute_pipeline: compute,
        blit_pipeline,
    });
}

#[allow(clippy::too_many_arguments)]
fn prepare_dda_bind_groups(
    mut commands: Commands,
    pipelines: Res<DdaPipelines>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    images: Option<Res<DdaImages>>,
    view_uniform: Option<Res<DdaViewUniform>>,
    gpu_brickmap: Option<Res<GpuBrickMap>>,
    gpu_mov: Option<Res<GpuMovPool>>,
    render_device: Res<RenderDevice>,
    pipeline_cache: Res<PipelineCache>,
    queue: Res<RenderQueue>,
) {
    // Prepare 与 Extract 的跨 world 同步在首帧可能还没完成，这些缺失是正常的，
    // 下一帧自动补齐；不应当 warn 级别噪音。
    let Some(images) = images else {
        bevy::log::debug_once!("DDA prepare: no DdaImages");
        return;
    };
    let Some(view_uniform) = view_uniform else {
        bevy::log::debug_once!("DDA prepare: no DdaViewUniform (extract_camera_config 未产出)");
        return;
    };
    let Some(gpu) = gpu_brickmap else {
        bevy::log::debug_once!("DDA prepare: no GpuBrickMap");
        return;
    };
    let Some(mov) = gpu_mov else {
        bevy::log::debug_once!("DDA prepare: no GpuMovPool");
        return;
    };
    let Some(tex_view) = gpu_images.get(&images.target) else {
        bevy::log::debug_once!("DDA prepare: GpuImage not ready");
        return;
    };

    // ---- 写 DdaViewUniform 到 UniformBuffer ----
    let mut u = UniformBuffer::from(view_uniform.into_inner());
    u.write_buffer(&render_device, &queue);

    // ---- 通过 PipelineCache 把 Descriptor 转成 BindGroupLayout handle ----
    let bg0_layout = pipeline_cache.get_bind_group_layout(&pipelines.bg0_layout);
    let bg1_layout = pipeline_cache.get_bind_group_layout(&pipelines.bg1_layout);
    let bg2_layout = pipeline_cache.get_bind_group_layout(&pipelines.bg2_layout);
    let blit_layout = pipeline_cache.get_bind_group_layout(&pipelines.blit_layout);

    // ---- BG0：out tex + view uniform ----
    let bg0 = render_device.create_bind_group(
        None,
        &bg0_layout,
        &BindGroupEntries::sequential((&tex_view.texture_view, &u)),
    );

    // ---- BG1：struct + leaves + palette + globals ----
    // globals：GpuBrickMap.globals 是 UniformBuffer，直接拿 binding
    let globals_bind = gpu.globals.binding().expect(
        "GpuBrickMap.globals uniform buffer 未初始化（RenderStartup init_empty_gpu 应默认构造）",
    );
    let bg1 = render_device.create_bind_group(
        None,
        &bg1_layout,
        &BindGroupEntries::sequential((
            gpu.struct_buf.as_entire_binding(),
            gpu.leaves.as_entire_binding(),
            gpu.palette.as_entire_binding(),
            globals_bind,
        )),
    );

    // ---- BG2：MOV pool（四 storage + count uniform）----
    let mov_globals_bind = mov.globals.binding().expect(
        "GpuMovPool.globals uniform buffer 未初始化（RenderStartup init_empty_mov_pool 应默认构造）",
    );
    let bg2 = render_device.create_bind_group(
        None,
        &bg2_layout,
        &BindGroupEntries::sequential((
            mov.struct_buf.as_entire_binding(),
            mov.leaves.as_entire_binding(),
            mov.palette.as_entire_binding(),
            mov.descs.as_entire_binding(),
            mov_globals_bind,
        )),
    );

    // ---- Blit BG：dda tex（和 gradient 相同的 texture_2d blit layout）----
    let blit_bg = render_device.create_bind_group(
        None,
        &blit_layout,
        &BindGroupEntries::single(&tex_view.texture_view),
    );

    commands.insert_resource(DdaBg0BindGroup(bg0));
    commands.insert_resource(DdaBg1BindGroup(bg1));
    commands.insert_resource(DdaBg2BindGroup(bg2));
    commands.insert_resource(DdaBlitBindGroup(blit_bg));
}

fn dispatch_dda(
    mut ctx: RenderContext,
    bg0: Option<Res<DdaBg0BindGroup>>,
    bg1: Option<Res<DdaBg1BindGroup>>,
    bg2: Option<Res<DdaBg2BindGroup>>,
    pipeline_cache: Res<PipelineCache>,
    pipelines: Res<DdaPipelines>,
    scale: Res<crate::gradient::RenderScale>,
) {
    let (Some(bg0), Some(bg1), Some(bg2)) = (bg0.as_ref(), bg1.as_ref(), bg2.as_ref()) else {
        bevy::log::debug_once!("DDA dispatch: bind groups missing");
        return;
    };
    // 首帧 PipelineCache 尚未完成着色器编译是正常时序（Render schedule 与 pipeline 编译异步），
    // 不应当 warn 刷屏：降级 debug 级别，仅在 DEBUG 级 filter 时可见。
    let Some(pipe) = pipeline_cache.get_compute_pipeline(pipelines.compute_pipeline) else {
        bevy::log::debug_once!("DDA dispatch: compute pipeline not ready");
        return;
    };
    // P2.7：诊断 span（recorder 缺失时 Option<&T> impl no-op，dispatch 绝不跳过——
    // RenderDiagnosticsPlugin 非默认装配，不能因无 recorder 丢渲染）
    let recorder = ctx.diagnostic_recorder();
    let recorder = recorder.as_deref();
    let span = recorder.time_span(ctx.command_encoder(), "gate_dda_compute");
    let mut pass = ctx
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor::default());
    pass.set_pipeline(pipe);
    pass.set_bind_group(0, &bg0.0, &[]);
    pass.set_bind_group(1, &bg1.0, &[]);
    pass.set_bind_group(2, &bg2.0, &[]);
    let gx = scale.size.x.div_ceil(WORKGROUP_SIZE);
    let gy = scale.size.y.div_ceil(WORKGROUP_SIZE);
    pass.dispatch_workgroups(gx, gy, 1);
    drop(pass);
    span.end(ctx.command_encoder());
}

// DDA blit 挂 Core2d PostProcess，Gradient blit 也挂在同一 set，
// 通过 build() 里显式 .after(gradient::blit_view) 保证 DDA 后执行覆盖渐变画面
// （Bevy 同 set 系统默认无序，靠 add_systems 注册顺序不可靠——实机截图已复现此 bug）
fn blit_dda_view(
    mut ctx: RenderContext,
    views: Query<&ViewTarget>,
    blit_bg: Option<Res<DdaBlitBindGroup>>,
    pipeline_cache: Res<PipelineCache>,
    pipelines: Res<DdaPipelines>,
) {
    let (Some(bg), Ok(target)) = (blit_bg.as_ref(), views.single()) else {
        bevy::log::debug_once!("DDA blit: bg or ViewTarget missing");
        return;
    };
    let Some(pipe) = pipeline_cache.get_render_pipeline(pipelines.blit_pipeline) else {
        bevy::log::debug_once!("DDA blit: blit pipeline not ready");
        return;
    };
    // 说明见 gradient.rs blit_view 同注释：forget_lifetime 与 pass_span 类型冲突 → time_span 覆盖整段
    // recorder 缺失（插件未装配）→ Option<&T> impl no-op，draw 绝不跳过
    let recorder = ctx.diagnostic_recorder();
    let recorder = recorder.as_deref();
    let span = recorder.time_span(ctx.command_encoder(), "gate_dda_blit");
    let pass = ctx
        .command_encoder()
        .begin_render_pass(&RenderPassDescriptor {
            label: Some("gate_dda_blit"),
            color_attachments: &[Some(target.get_color_attachment())],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            ..default()
        });
    let mut pass = pass.forget_lifetime();
    pass.set_pipeline(pipe);
    pass.set_bind_group(0, &bg.0, &[]);
    pass.draw(0..3, 0..1);
    drop(pass);
    span.end(ctx.command_encoder());
}
