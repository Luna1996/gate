//! DDA 主可见性 pass：WGSL compute + Core2d PostProcess blit
//!
//! - BG0：storage tex + 相机 uniform + beam depth + 眼睛适应状态（只读）
//! - BG1：b_struct + b_leaves + palette + globals uniform + 光照场 3D 纹理
//! - WGSL 源 = WESL 包 `shaders/voxel_raytrace/`（入口 `main.wesl`，见 `crate::shader`）

use bevy::{
  asset::RenderAssetUsages,
  image::Image,
  prelude::*,
  render::{extract_resource::ExtractResource, render_resource::*},
};
use std::ops::Mul;
use std::sync::LazyLock;

// ============================================================================
// 渲染目标共享基础设施
// ============================================================================

/// blit.wgsl 资产路径（全屏三角 blit）
pub const BLIT_SHADER_ASSET_PATH: &str = "shaders/blit.wgsl";
/// 初始渲染分辨率（窗口创建尺寸；resize 后由 RenderScale 资源接管，FR-5）
pub const VIEW_SIZE: UVec2 = UVec2::new(1280, 720);
/// beam pass 的 compute dispatch 工作组边长
pub const WORKGROUP_SIZE: u32 = 8;
/// 主 DDA pass 工作组边长：必须与 shaders/voxel_raytrace/ 中 dda_main 的 @workgroup_size
/// 严格一致，否则 dispatch 覆盖不足漏 trace 像素。
pub const DDA_WORKGROUP_SIZE: u32 = 8;

/// 当前渲染分辨率（main world `resize_render_targets` 更新，提取进 render world；
/// dispatch workgroup 数随它重算，shader 侧自行越界剔除）
#[derive(Resource, Clone, Copy, Debug, PartialEq, ExtractResource)]
pub struct RenderScale {
  pub size: UVec2,
}

impl Default for RenderScale {
  fn default() -> Self {
    Self { size: VIEW_SIZE }
  }
}

// ============================================================================
// 视图资源 + 图像资源
// ============================================================================

/// 主 world 注入的静态视图配置（矩阵来自 [`Self::build_static`] 的手算参数）。
///
/// 手算 perspective_rh(fovy=60°, aspect=1280/720) × look_at_rh：
/// - eye voxel (700, 560, 700)，target voxel (260, 120, 260)，距离 ~762
/// - up = Vec3::Y，far 4000（DDA 用 inv_view_proj 反投影方向，far 只影响精度）
#[derive(Resource, Clone, Copy)]
pub struct DdaCameraConfig {
  pub view_proj: Mat4,
  pub inv_view_proj: Mat4,
  pub position_world: Vec3,
}

impl DdaCameraConfig {
  /// 静态视图构建（eye/target 为手算常量）
  pub fn build_static() -> Self {
    let eye = Vec3::new(700.0, 560.0, 700.0);
    let target = Vec3::new(260.0, 120.0, 260.0);
    let up = Vec3::Y;
    let aspect = VIEW_SIZE.x as f32 / VIEW_SIZE.y as f32;
    let fovy = 60.0_f32.to_radians();
    // near 取 1.0：near/far 比过大会让 inv_view_proj 条件数变差、反投影方向失真
    let near = 1.0;
    let far = 4000.0;
    // Bevy Mat4：perspective_rh 右手系 +y 上 -z 前；look_at_rh 朝 -z
    let proj = Mat4::perspective_rh(fovy, aspect, near, far);
    let view = Mat4::look_at_rh(eye, target, up);
    let view_proj = proj.mul(view);
    let inv_view_proj = view_proj.inverse();
    Self { view_proj, inv_view_proj, position_world: eye }
  }
}

/// 调试视图模式（main world Resource，按 N 键循环 0→1→2→0）：
/// 0 = 正常画面；1 = 法向向量可视化；2 = G-buffer 状态图（sky=品红、face 6 色）
#[derive(Resource, Clone, Copy, Default, bevy::render::extract_resource::ExtractResource)]
pub struct DebugNormals(pub u32);

/// 相机约束常量（pub 供 gate-app 输入 system 与测试断言）
pub const PITCH_LIMIT: f32 = 89.0_f32.to_radians(); // ±89° 防万向节锁（up 与 view 共线）
pub const DIST_MIN: f32 = 32.0; // 最近 32 voxel（8cm，不穿进体素内部失稳）
// distance 只有下界、无上限 clamp；滚轮异常操作产生的 NaN 由下界兜底，远景可见性由透视 far 面负责。

/// 轨道相机参数（main world 资源，gate-app 输入 system 操作）
///
/// 字段语义：target = 注视点（voxel）、distance = 相机到 target 距离（voxel）、
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
    Self { target, distance, yaw, pitch }
  }

  /// 轨道参数重建眼位（FR-1 公式：eye = target + distance·(sin_yaw·cos_pitch, sin_pitch, cos_yaw·cos_pitch)）
  pub fn eye(&self) -> Vec3 {
    let (sin_yaw, cos_yaw) = self.yaw.sin_cos();
    let (sin_pitch, cos_pitch) = self.pitch.sin_cos();
    self.target + self.distance * Vec3::new(sin_yaw * cos_pitch, sin_pitch, cos_yaw * cos_pitch)
  }

  /// 应用约束（pitch ±89°、distance ≥ DIST_MIN；yaw 自由旋转不 clamp，distance 无上限）
  pub fn clamp(&mut self) {
    self.pitch = self.pitch.clamp(-PITCH_LIMIT, PITCH_LIMIT);
    self.distance = self.distance.max(DIST_MIN);
  }
}

impl DdaCameraConfig {
  /// 眼位 + 视线方向构造（幽灵/飞行相机用；轨道相机走 [`Self::from_orbit`]）。
  ///
  /// `forward` 必须是朝向场景的单位向量（调用方从 yaw/pitch 取，且 pitch 已 clamp 到 ±89°，
  /// 所以它与 +Y 不共线 → look_at_rh 的 up 基准不会退化）。与
  /// `look_at_rh(eye, eye + forward, +Y)` 逐字等价。
  pub fn from_eye_forward(
    eye: Vec3,
    forward: Vec3,
    fov_y: f32,
    aspect: f32,
    near: f32,
    far: f32,
  ) -> Self {
    let f = forward.normalize();
    let view = Mat4::look_at_rh(eye, eye + f, Vec3::Y);
    let proj = Mat4::perspective_rh(fov_y, aspect, near, far);
    let view_proj = proj.mul(view);
    Self { view_proj, inv_view_proj: view_proj.inverse(), position_world: eye }
  }

  /// 唯一矩阵构造点（spec FR-2）：orbit 参数 → perspective_rh × look_at_rh。
  ///
  /// fov/aspect/near/far 为显式参数（app 传 60° / 1280:720 / 1.0 / 4000），
  /// 不写死在 orbit 里——性能面板改 fov、多分辨率改 aspect 时复用同一入口。
  pub fn from_orbit(orbit: &OrbitCamera, fov_y: f32, aspect: f32, near: f32, far: f32) -> Self {
    let eye = orbit.eye();
    let view = Mat4::look_at_rh(eye, orbit.target, Vec3::Y);
    let proj = Mat4::perspective_rh(fov_y, aspect, near, far);
    let view_proj = proj.mul(view);
    Self { view_proj, inv_view_proj: view_proj.inverse(), position_world: eye }
  }
}

/// Render-world 着色器绑定的 camera uniform
/// WGSL `DdaViewUniform` 逐字对齐（2×mat4x4 + 4×vec4 = 128+64 = 192B）
#[derive(Resource, Clone, Copy, ShaderType)]
pub struct DdaViewUniform {
  pub view_proj: Mat4,
  pub inv_view_proj: Mat4,
  pub cam_pos_voxel: Vec4, // w=1
  /// x/y = debug 可视化（保留）；z = 2 跳过 chunk 步进；w = +2 skyout / +4 makegrid_only
  pub debug_mode: Vec4,
  /// x = 单像素角大小(rad) = 2·tan(FOV_Y/2)/render_h；y = LOD 早停开关（GATE_NO_LOD=1 关）
  pub lod: Vec4,
  pub probe_viz_params: Vec4,
}

/// 【诊断】GATE_SKIP_CHUNKWALK=1：trace_grid 在局部 slab 后直接 miss
static SKIP_CHUNKWALK: LazyLock<bool> =
  LazyLock::new(|| std::env::var("GATE_SKIP_CHUNKWALK").map(|v| v == "1").unwrap_or(false));
/// 【诊断】GATE_SKYOUT=1：dda_main 跳过全部 trace 直接输出天空色
static SKY_OUT: LazyLock<bool> =
  LazyLock::new(|| std::env::var("GATE_SKYOUT").map(|v| v == "1").unwrap_or(false));
/// 【诊断】GATE_MAKEGRID_ONLY=1：dda_main 只做 make_grid 不 trace
static MAKEGRID_ONLY: LazyLock<bool> =
  LazyLock::new(|| std::env::var("GATE_MAKEGRID_ONLY").map(|v| v == "1").unwrap_or(false));
/// 【诊断】GATE_NO_LOD=1：关闭八叉树远场早停
static LOD_DISABLED: LazyLock<bool> =
  LazyLock::new(|| std::env::var("GATE_NO_LOD").map(|v| v == "1").unwrap_or(false));
/// 【诊断】GATE_NO_BEAM=1：关闭 beam 预 pass，主 pass 从 t=0 起步。默认开启 beam。
static BEAM_DISABLED: LazyLock<bool> =
  LazyLock::new(|| std::env::var("GATE_NO_BEAM").map(|v| v == "1").unwrap_or(false));
/// 【诊断】GATE_NO_LUT=1：关闭方向可达掩码剔除（lod.w 传 1 → shader 端 eff = mask 旁路 LUT）。
static LUT_DISABLED: LazyLock<bool> =
  LazyLock::new(|| std::env::var("GATE_NO_LUT").map(|v| v == "1").unwrap_or(false));

impl DdaViewUniform {
  pub fn from_cfg(cfg: &DdaCameraConfig, debug_mode: u32, render_h: f32) -> Self {
    // 像素角大小：垂直 FOV 60°（gate-app FOV_Y 镜像）均分到 render_h 像素
    let px_ang = 2.0 * 30.0_f32.to_radians().tan() / render_h.max(1.0);
    Self {
      view_proj: cfg.view_proj,
      inv_view_proj: cfg.inv_view_proj,
      cam_pos_voxel: cfg.position_world.extend(1.0),
      debug_mode: Vec4::new(
        (debug_mode == 1) as u32 as f32,         // x = 法向可视化
        (debug_mode == 2) as u32 as f32,         // y = face 6 色诊断
        if *SKIP_CHUNKWALK { 2.0 } else { 0.0 }, // z = 2 跳过 chunk 步进
        if *SKY_OUT {
          2.0
        } else if *MAKEGRID_ONLY {
          4.0
        } else if debug_mode == 3 {
          1.0 // unlit 诊断：跳过全部光照 albedo 直出（测纯 trace 帧率）
        } else {
          0.0
        },
      ),
      lod: Vec4::new(
        px_ang,
        (!*LOD_DISABLED) as u32 as f32,
        *BEAM_DISABLED as u32 as f32,
        *LUT_DISABLED as u32 as f32, // w = 1 → shader 旁路方向掩码剔除
      ),
      probe_viz_params: Vec4::ZERO,
    }
  }
}

/// DDA 着色器的纹理（main world 创建，提取进 render world）
#[derive(Resource, Clone, ExtractResource)]
pub struct DdaImages {
  pub target: Handle<Image>,
}

/// 工厂：DDA 目标纹理（rgba8unorm VIEW_SIZE，STORAGE|TEXTURE + RENDER_WORLD usage）
/// COPY_DST：bevy resize 路径 copy_image_on_resize 会向新纹理拷贝旧内容，缺 COPY_DST 即验证崩溃
pub fn create_dda_image(images: &mut Assets<Image>) -> Handle<Image> {
  let mut image =
    Image::new_target_texture(VIEW_SIZE.x, VIEW_SIZE.y, TextureFormat::Rgba8Unorm, None);
  image.asset_usage = RenderAssetUsages::RENDER_WORLD;
  image.texture_descriptor.usage = TextureUsages::STORAGE_BINDING
    | TextureUsages::TEXTURE_BINDING
    | TextureUsages::COPY_SRC
    | TextureUsages::COPY_DST;
  images.add(image)
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
    assert!((orbit.distance - 762.1024).abs() < 0.01, "distance {}", orbit.distance);
    assert!((orbit.pitch - 1.0_f32.atan2(2.0_f32.sqrt())).abs() < 1e-5, "pitch {}", orbit.pitch);
    assert!((orbit.yaw - std::f32::consts::FRAC_PI_4).abs() < 1e-5);
    let rebuilt = orbit.eye();
    assert!((rebuilt - eye).length() < 1e-4, "eye rebuild drift {}", (rebuilt - eye).length());
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
    let mut o =
      OrbitCamera { target: Vec3::ZERO, distance: 1.0, yaw: 0.0, pitch: 95.0_f32.to_radians() };
    o.clamp();
    assert_eq!(o.pitch, PITCH_LIMIT);
    assert_eq!(o.distance, DIST_MIN);
    // 超大 distance（原 DIST_MAX=8000 的 100×）应保留原值（不上限 clamp）
    let big = 1.0e6;
    let mut o2 =
      OrbitCamera { target: Vec3::ZERO, distance: big, yaw: 0.0, pitch: -95.0_f32.to_radians() };
    o2.clamp();
    assert_eq!(o2.pitch, -PITCH_LIMIT);
    assert_eq!(o2.distance, big);
  }

  /// AC-1③: from_orbit(from_eye(build_static 的 eye/target)) 与 build_static 三字段逐元素一致（<1e-5，回归保护）
  #[test]
  fn orbit_from_orbit_equals_build_static() {
    let orbit =
      OrbitCamera::from_eye(Vec3::new(700.0, 560.0, 700.0), Vec3::new(260.0, 120.0, 260.0));
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
    let orbit =
      OrbitCamera::from_eye(Vec3::new(700.0, 560.0, 700.0), Vec3::new(260.0, 120.0, 260.0));
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
        assert!((r.to_cols_array()[i] - exp).abs() < 1e-3, "{w}x{h} inv[{i}] drift");
      }
    }
  }
}

// ============================================================================
// WGSL 常量 Rust 镜像（单测 assert_eq! 对 wire.rs 常量，防漂移）
// ============================================================================

/// WGSL 着色器顶部 `const` 的 Rust 镜像副本（单测 assert_eq! 对 wire.rs 常量，防漂移）
pub mod wgsl_consts {
  // 分裂树层级（Douglas Brick Tree：256 → 64 → 16 → 4 → 1）
  pub const CHUNK_SIZE: u32 = 256;
  pub const BRICK_FACTOR: u32 = 4;
  pub const MAX_LEVEL: u32 = 4;
  /// 每节点 fixed 字数（mask_lo + mask_hi + palette_u32）
  pub const NODE_FIXED_WORDS: u32 = 3;
  // b_struct Region ①：稠密 chunk 窗口
  pub const CHUNK_INDEX_CAP: u32 = 64;
  pub const CHUNK_INDEX_WORDS: u32 = 262_144; // 64³
  pub const TREE_BASE: u32 = 262_144;
  // palette / comp / state
  pub const PALETTE_WORDS: u32 = 512; // 256 条 × 2w
  pub const CHUNK_COMP_WORDS: u32 = 2048; // u16[4096] → 每 2 字打包 u32
  pub const STATE_ENTRY_COUNT: u32 = 256;
  pub const STATE_WORDS_PER_ENTRY: u32 = 4;
  pub const STATE_TOTAL_WORDS: u32 = 1024; // 256 × 4
  // 直光层（镜像 lighting.rs 常量；WGSL const 同步，单测防漂移）
  pub const SHADOW_BIAS: f32 = crate::lighting::SHADOW_BIAS;
  pub const SHADOW_DIR_T_MAX: f32 = crate::lighting::SHADOW_DIR_T_MAX;
  pub const EMISSIVE_EMIT_GAIN: f32 = crate::lighting::EMISSIVE_EMIT_GAIN;
  // 光照场（AO fill + 「体素即光源」的发光密度 ε 共用一张 3D 纹理）。cell = 16 voxel，
  // dims = 32³ cell → 世界覆盖 = 32×16 = 512 voxel = ±5.12m（相机中心）。
  // 寻址与 DDGI 同构：原点按 cell 向下对齐、槽位 = 世界 cell mod dims（世界锚定）。
  // 格式 Rgba16Unorm：.rgb = ε、.a = AO fill。
  // upload.rs 铺图依赖 LIGHT_FIELD_DIM×8 是 256 的整数倍（行对齐）。
  pub const LIGHT_FIELD_CELL: u32 = 16;
  pub const LIGHT_FIELD_DIM: u32 = 32;
  /// 起点从体素表面再外推的量（体素）。着色点锚在**体素中心**，`+ n×0.5` 恰好落在面平面
  /// 上；对 -X/-Y/-Z 面这个坐标是整数 → DDA 的 `floor` 落回**体素自己** → 自命中，而
  /// dda_main 的自命中防护把「命中自己」当**无遮挡**。外推 1/32 体素（≈0.6mm）把起点推过
  /// 边界，且足够小、不会漏掉紧贴表面的薄遮挡物。
  /// 必须与 WGSL `SHADOW_SURFACE_EPS` 一致（tests/wgsl_compile.rs 有对齐测试）。
  pub const SHADOW_SURFACE_EPS: f32 = 0.03125;
}

/// 与 WGSL `popcount(mask & (bit - 1u64))` 等价：mask bit=1 子块在 child offset
/// 表中的槽位（紧凑 child offset 只存 bit=1 的子块）
#[inline]
pub fn wgsl_child_slot_index(mask: u64, child_idx: u32) -> u32 {
  debug_assert!(child_idx < 64);
  (mask & ((1u64 << child_idx) - 1)).count_ones()
}

#[cfg(test)]
mod const_tests {
  use super::wgsl_child_slot_index;
  use super::wgsl_consts::*;
  use crate::brickmap::wire as w;

  /// 确定性 xorshift64（同 dda_ref_tests）
  fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
  }

  /// WGSL 常量 == wire.rs 对应常量（shader 源与 wire.rs 契约一致）
  #[test]
  fn wire_wgsl_constants_aligned() {
    assert_eq!(CHUNK_SIZE, w::CHUNK_SIZE as u32);
    assert_eq!(BRICK_FACTOR, w::BRICK_FACTOR as u32);
    assert_eq!(MAX_LEVEL, w::MAX_LEVEL);
    assert_eq!(NODE_FIXED_WORDS, w::NODE_FIXED_WORDS as u32);
    assert_eq!(CHUNK_INDEX_CAP, w::CHUNK_INDEX_CAP as u32);
    assert_eq!(CHUNK_INDEX_WORDS, w::CHUNK_INDEX_WORDS as u32);
    assert_eq!(TREE_BASE, w::TREE_BASE as u32);
    assert_eq!(PALETTE_WORDS, w::PALETTE_WORDS as u32);
    assert_eq!(CHUNK_COMP_WORDS, w::CHUNK_COMP_WORDS as u32);
    assert_eq!(STATE_ENTRY_COUNT, w::STATE_ENTRY_COUNT as u32);
    assert_eq!(STATE_WORDS_PER_ENTRY, w::STATE_WORDS_PER_ENTRY as u32);
    assert_eq!(STATE_TOTAL_WORDS, w::STATE_TOTAL_WORDS as u32);
  }

  /// child_slot_index 与「线性扫描 mask 低位」等价（200 随机 (mask, child_idx)）
  #[test]
  fn child_slot_index_matches_linear_scan() {
    let mut state: u64 = 0x517CC1B727220A95;
    for _ in 0..200 {
      let mask = xorshift64(&mut state) | (xorshift64(&mut state) << 32); // 全 64bit 随机
      let child_idx = (xorshift64(&mut state) % 64) as u32;
      // 线性扫描：slot = child_idx 位之前 bit=1 的个数
      let expect = (0..child_idx).filter(|&i| mask & (1u64 << i) != 0).count() as u32;
      assert_eq!(
        wgsl_child_slot_index(mask, child_idx),
        expect,
        "mask={mask:#x} child_idx={child_idx}"
      );
    }
  }
}

// ============================================================================
// CPU DDA 参考实现（独立 A&W step，仅复用 BrickMapView::get_voxel 读 palette）
// ============================================================================

use crate::brickmap::{BrickMapBuffers, BrickMapView};

/// A&W 细格步进 DDA 参考实现（CPU）。
///
/// 仅调用 `BrickMapView::get_voxel(voxel: IVec3)` 查询 palette。返回
/// `Some((hit_t, palette))` 或 `None`（t >= t_max 前未命中 / 超步）
///
/// 关键正确性约定：cell 坐标用整数增量维护（floor(origin) 起步，每次穿越 +sign），
/// **不得**用 `floor(origin + dir*t)` 重算——t 恰好是边界穿越时刻时 pos 分量正好落在
/// 整数边界上，floor 会随机取到穿越前/后的胞，导致对角胞漏检（DDA vs brute 不等价）。
pub fn cpu_reference_dda_ray(
  buffers: &BrickMapBuffers,
  origin_voxel: Vec3,
  dir_voxel: Vec3, // voxel units（归一化），magnitude 任意（delta 按 |dir| 缩放）
  t_max: f32,
  max_steps: u32,
) -> Option<(f32, u8)> {
  let view = BrickMapView::new(buffers);
  let mut t = 0.0f32;
  // init A&W 变量
  let sign = [
    if dir_voxel.x >= 0.0 { 1 } else { -1 },
    if dir_voxel.y >= 0.0 { 1 } else { -1 },
    if dir_voxel.z >= 0.0 { 1 } else { -1 },
  ];
  let delta = [
    if dir_voxel.x.abs() > 1e-30 { (1.0 / dir_voxel.x).abs() } else { f32::INFINITY },
    if dir_voxel.y.abs() > 1e-30 { (1.0 / dir_voxel.y).abs() } else { f32::INFINITY },
    if dir_voxel.z.abs() > 1e-30 { (1.0 / dir_voxel.z).abs() } else { f32::INFINITY },
  ];
  let mut cell =
    [origin_voxel.x.floor() as i32, origin_voxel.y.floor() as i32, origin_voxel.z.floor() as i32];
  let next_boundary = |c: i32, s: i32| -> f32 {
    // s=1 -> 下一个上界 (c+1).0; s=-1 -> 当前下界 c.0（负数 floor 刚好也是下一个朝向的边界）
    (if s >= 0 { c + 1 } else { c }) as f32
  };
  let tmax_x = if dir_voxel.x.abs() <= 1e-30 {
    f32::INFINITY
  } else {
    (next_boundary(cell[0], sign[0]) - origin_voxel.x) / dir_voxel.x
  };
  let tmax_y = if dir_voxel.y.abs() <= 1e-30 {
    f32::INFINITY
  } else {
    (next_boundary(cell[1], sign[1]) - origin_voxel.y) / dir_voxel.y
  };
  let tmax_z = if dir_voxel.z.abs() <= 1e-30 {
    f32::INFINITY
  } else {
    (next_boundary(cell[2], sign[2]) - origin_voxel.z) / dir_voxel.z
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
/// 标尺约定：所有 t/tmax 都自"新起点 start = origin + dir·t_enter"量起，不混用自 origin
/// 的全局 t；返回的 hit_t 已加回 t_enter，为自 origin 量起的全局距离。
///
/// 与 cpu_reference_dda_ray 逐射线命中等价（单测 `aabb_skip_equivalence_300_rays` 验证）。
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
    if dir.x.abs() > 1e-30 { (1.0 / dir.x).abs() } else { f32::INFINITY },
    if dir.y.abs() > 1e-30 { (1.0 / dir.y).abs() } else { f32::INFINITY },
    if dir.z.abs() > 1e-30 { (1.0 / dir.z).abs() } else { f32::INFINITY },
  ];
  let next_boundary = |c: i32, s: i32| -> f32 { (if s >= 0 { c + 1 } else { c }) as f32 };
  let mut cell = [start.x.floor() as i32, start.y.floor() as i32, start.z.floor() as i32];
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

/// 细级：单个粗 cell（16³ voxel）内的有界 voxel DDA。
/// 射线段限制在 [t_lo, t_hi]（该 cell 的入出区间，自 origin 全局标尺）。
///
/// 入口 voxel 胞用「解析 + clamp」确定：p = origin + dir·t_lo 落在 cell 入口面上时，
/// floor(p) 可能因浮点误差取到邻胞——clamp 到 [base, base+15] 保证起点一定在本 cell 内。
/// clamp 不会漏检：入口面外侧的最后一个 voxel 胞属于前一个粗 cell，其细扫已覆盖。
/// 采样序列与 full DDA 在同区间的序列逐胞一致（voxel 边界与粗边界 16 对齐）。
#[allow(clippy::too_many_arguments)] // cell 局部细扫的固有参数面（view+射线+cell 窗口）
fn dda_voxel_scan_cell(
  view: &BrickMapView,
  origin: Vec3,
  dir: Vec3,
  sign: [i32; 3],
  delta: [f32; 3], // voxel 步距（|dir| 分量倒数或 INF）
  cc: [i32; 3],    // 粗 cell 坐标（1 单位 = 16 voxel）
  t_lo: f32,
  t_hi: f32,
) -> Option<(f32, u8, u8)> {
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
  // voxel tmax：相对 t_lo 的距离（自 cell 入口重算，非累加——与 full 的 ulp 差异见主函数注释）
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
  // 初始 voxel 胞采样（起点在体内 → 无跨越面，axis=3 哨兵：调用方以 -dir 作法线）
  if let Some(pal) = view.get_voxel(IVec3::from_array(fc)) {
    return Some((t_lo, pal, 3));
  }
  // 48 = 3 轴 × 16：斜穿 16³ cell 的步数上界，几何上必在 span 内退出
  for _ in 0..48 {
    if t_f >= span {
      return None;
    }
    let axis = if tmax_f[0] <= tmax_f[1] && tmax_f[0] <= tmax_f[2] {
      t_f = tmax_f[0];
      tmax_f[0] += delta[0];
      fc[0] += sign[0];
      0u8
    } else if tmax_f[1] <= tmax_f[2] {
      t_f = tmax_f[1];
      tmax_f[1] += delta[1];
      fc[1] += sign[1];
      1u8
    } else {
      t_f = tmax_f[2];
      tmax_f[2] += delta[2];
      fc[2] += sign[2];
      2u8
    };
    if let Some(pal) = view.get_voxel(IVec3::from_array(fc)) {
      return Some((t_lo + t_f, pal, axis));
    }
  }
  None
}

/// 两级 DDA 命中记录：axis = 最后跨越的细格轴（0/1/2；3 = 起点即在体内，无跨越面）。
/// 面法线 = -sign[axis] 方向的单位轴向量（体素命中面恒与轴对齐）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DdaHit {
  pub t: f32,
  pub pal: u8,
  pub axis: u8,
}

/// 两级 DDA 参考实现（cell 粗步 + cell 内细步）。
///
/// 语义与 `cpu_reference_dda_ray` 完全一致：射线 (origin, dir, t∈[0,t_max]) 上
/// 首个非空 voxel 体素，返回命中记录或 None。
///
/// 结构（WGSL dda_main 逐字对应的源）：
///   1. 粗级 A&W：cell 粒度（16 voxel）步进，delta_c = delta × 16（f32 乘 2 的幂，精确）；
///      每步先做 `BrickMapView::cell_occupied`（①+②，2 次 load），空 cell 整段跳过。
///   2. 细级：占用 cell 内 `dda_voxel_scan_cell`，区间 [t_in, min(t_out, t_max)]。
///   3. 退出条件：t_out ≥ t_max（cell 出口越过上限）或粗步数耗尽。
///
/// 与 full 版的数值差异：粗级按 delta_c 累加、细级自 cell 入口重算，长路径 ulp 漂移可达
/// ~0.1 voxel，只影响返回 t 值，不影响命中胞序（等价性单测以 1.0 voxel 容差覆盖）。
pub fn cpu_reference_dda_ray_two_level(
  buffers: &BrickMapBuffers,
  origin_voxel: Vec3,
  dir_voxel: Vec3, // voxel units（归一化），magnitude 任意（delta 按 |dir| 缩放）
  t_max: f32,
  max_steps: u32,
) -> Option<DdaHit> {
  let view = BrickMapView::new(buffers);
  let o = [origin_voxel.x, origin_voxel.y, origin_voxel.z];
  let d = [dir_voxel.x, dir_voxel.y, dir_voxel.z];
  let sign = [
    if dir_voxel.x >= 0.0 { 1 } else { -1 },
    if dir_voxel.y >= 0.0 { 1 } else { -1 },
    if dir_voxel.z >= 0.0 { 1 } else { -1 },
  ];
  let delta = [
    if d[0].abs() > 1e-30 { (1.0 / d[0]).abs() } else { f32::INFINITY },
    if d[1].abs() > 1e-30 { (1.0 / d[1]).abs() } else { f32::INFINITY },
    if d[2].abs() > 1e-30 { (1.0 / d[2]).abs() } else { f32::INFINITY },
  ];
  // 粗 delta = voxel delta × 16（精确）
  let delta_c = [delta[0] * 16.0, delta[1] * 16.0, delta[2] * 16.0];
  // 粗 cell：floor(origin) >> 4（算术右移 = floor 除法，负坐标正确）
  let mut cc = [(o[0].floor() as i32) >> 4, (o[1].floor() as i32) >> 4, (o[2].floor() as i32) >> 4];
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
    // 占用查询先行：空 cell 不做任何 voxel 采样（性能核心）
    if view.cell_occupied(IVec3::from_array(cc))
      && let Some((t, pal, axis)) =
        dda_voxel_scan_cell(&view, origin_voxel, dir_voxel, sign, delta, cc, t_in, t_out.min(t_max))
    {
      return Some(DdaHit { t, pal, axis });
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

// ============================================================================
// 层次栈式 mask DDA（WGSL shaders/voxel_raytrace/ TraceFrame/init_tree_frame/trace_chunk/
// trace_grid chunk 间循环的 CPU 逐字镜像）
//
// 节点 mask 一次读进寄存器，该节点 4³=64 子块间步进只查 bit（零 load）；bit=1 分裂才压栈
// 下钻，bit=0 uniform 子块整格跳过/整格命中。
//
// 结构与 WGSL 严格一一对应：TreeFrameCpu ↔ TreeFrame，init_tree_frame_cpu ↔ init_tree_frame，
// trace_chunk_cpu ↔ trace_chunk（单 chunk 内 4 层栈帧），trace_volume_tree ↔ trace_grid 的
// 局部 slab + chunk 间 256³ A&W 段。
// ============================================================================

/// 层次遍历命中记录（镜像 WGSL VoxelHit/UnifiedHit）：face_id 0..5 = ±xyz 六面。
/// voxel = 命中固体体素 grid 局部 voxel 整数坐标——DDA 整数步进精确产出，
/// 着色（per-voxel normal/GI key）直接消费，禁用「命中点 ± 法线半步」启发式重建。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TreeHit {
  pub t: f32,
  pub pal: u8,
  pub face_id: u8,
  pub voxel: IVec3,
}

/// 镜像 WGSL face_index_from_normal：取最大分量轴，法向分量 ≥0 → 正面索引。
fn face_index_from_normal(n: Vec3) -> u8 {
  let ax = n.x.abs();
  let ay = n.y.abs();
  let az = n.z.abs();
  if ax >= ay.max(az) {
    if n.x >= 0.0 { 1 } else { 0 }
  } else if ay >= az {
    if n.y >= 0.0 { 3 } else { 2 }
  } else if n.z >= 0.0 {
    5
  } else {
    4
  }
}

/// 镜像 WGSL face_normal_from_index：face_id 0..5 → ±轴单位向量。
fn face_normal_from_index(f: u8) -> Vec3 {
  match f {
    0 => -Vec3::X,
    1 => Vec3::X,
    2 => -Vec3::Y,
    3 => Vec3::Y,
    4 => -Vec3::Z,
    5 => Vec3::Z,
    _ => Vec3::ZERO,
  }
}

/// brick 缓存（镜像 WGSL Brick）：一层分裂节点的掩码常驻，跨级跳时零加载复用。
/// level 编号与 Douglas octo_march_core 一致：3 = 根（256³，子块 64³）、
/// 2（64³，子块 16³）、1（16³，子块 4³）、0（4³，inline 1³ 叶）。
#[derive(Clone, Copy)]
struct BrickCpu {
  addr: usize, // 节点绝对字址（b_struct）
  mask: u64,   // 64bit 分裂掩码（bit=1 = 子块分裂；bit=0 = 统一子块，色=pal）
  pal: u8,     // 节点 palette（统一子块颜色，0=空气）
}

/// 镜像 WGSL trace_chunk：单 chunk 内 Douglas 式整数体素层级 DDA
/// （octo_march_core：integer voxel + brick mask 栈 + firstTrailingBit 跨级跳）。
///
/// 状态只有整数体素坐标 v + bricks[4]（下钻载入、跳层复用）；边界距离按整数对齐每次重算
/// （side_distance_for_ray）；跨 brick 后用 firstTrailingBit 一次跳到最粗可行层
/// （尾随零位 = 对齐 run 长度）。
///
/// chunk_base = 根节点绝对字址；chunk_min = chunk 原点（局部 voxel）；
/// 射线段 [t0, t1]（ro 系绝对 t）；entry_face = 进入本 chunk 的面。
/// 返回 (t, pal, face_id, chunk 局部命中体素 v) 或 None（走出 chunk 未命中 / budget 耗尽）。
#[allow(clippy::too_many_arguments)]
fn trace_chunk_cpu(
  b_struct: &[u32],
  chunk_base: usize,
  chunk_min: [f32; 3],
  ro: [f32; 3],
  rd: [f32; 3],
  sign: [i32; 3],
  t0: f32,
  t1: f32,
  entry_face: u8,
) -> Option<(f32, u8, u8, [i32; 3])> {
  // 擦边退化（t0>=t1：射线只蹭到 chunk 边界）→ 无体素内部可穿过，直接 miss
  if t0 >= t1 {
    return None;
  }
  // chunk 局部 voxel 坐标（chunk 原点 = 0）；t 仍是 ro 系绝对 t
  let ro_c = [ro[0] - chunk_min[0], ro[1] - chunk_min[1], ro[2] - chunk_min[2]];
  // 预算倒数：side 距离/步长增量改乘法（每外层省 3 个 fdiv；与 WGSL inv_rd 镜像）
  let inv_rd = [1.0 / rd[0], 1.0 / rd[1], 1.0 / rd[2]];
  let read_brick = |addr: usize| BrickCpu {
    addr,
    mask: ((b_struct[addr + 1] as u64) << 32) | b_struct[addr] as u64,
    pal: (b_struct[addr + 2] & 0xFF) as u8,
  };
  let mut bricks = [BrickCpu { addr: 0, mask: 0, pal: 0 }; 4];
  bricks[3] = read_brick(chunk_base);
  let mut level: u32 = 3;
  // 当前体素（chunk 局部 voxel 整数坐标，0..255；跨出 chunk 的步进瞬态可达 -1/256）
  let p0 = [ro_c[0] + rd[0] * t0, ro_c[1] + rd[1] * t0, ro_c[2] + rd[2] * t0];
  let mut v = [
    (p0[0].floor() as i32).clamp(0, 255),
    (p0[1].floor() as i32).clamp(0, 255),
    (p0[2].floor() as i32).clamp(0, 255),
  ];
  let mut cur_t = t0;
  let mut face = entry_face;
  // 防挂死安全网（与 WGSL trace_chunk 同值）
  let mut budget: u32 = 65536;
  loop {
    if budget == 0 {
      return None;
    }
    budget -= 1;
    if level > 3 {
      return None;
    }
    // ---- traverse：从当前 level 下钻到 v 处内容（Douglas traverse_bit_set）----
    loop {
      let b = bricks[level as usize];
      let log2 = level * 2;
      let cell = [(v[0] >> log2) & 3, (v[1] >> log2) & 3, (v[2] >> log2) & 3];
      let idx = (cell[2] * 16 + cell[1] * 4 + cell[0]) as usize;
      if level == 0 {
        // 叶节点 inline palette：bit=1（非空体素）才 load inline word 取色；
        // bit=0 空气体素零 load（mask 在手）。
        if (b.mask & (1u64 << idx)) != 0 {
          let w = b_struct[b.addr + 3 + (idx >> 2)];
          let leaf_pal = ((w >> ((idx & 3) * 8)) & 0xFF) as u8;
          if leaf_pal != 0 {
            return Some((cur_t, leaf_pal, face, v));
          }
        }
        break; // 空气 leaf
      }
      let bit = 1u64 << idx;
      if b.mask & bit == 0 {
        // 统一子块：颜色 = 节点 palette（0=空气）
        if b.pal != 0 {
          return Some((cur_t, b.pal, face, v));
        }
        break; // 空气统一子块
      }
      // 分裂子块 → popcount 定位 child
      let pop_below = (b.mask & (bit - 1)).count_ones() as usize;
      let child_addr = chunk_base + b_struct[b.addr + 3 + pop_below] as usize;
      let cb = read_brick(child_addr);
      // 统一子节点快路径：wire 任意层的分裂位都可能指向 3 字统一
      // 节点（mask=0，pal 直决；叶层统一节点无 inline 16 字，禁读 addr+3 之后）
      if cb.mask == 0 {
        if cb.pal != 0 {
          return Some((cur_t, cb.pal, face, v));
        }
        break;
      }
      level -= 1;
      bricks[level as usize] = cb;
    }
    // ---- v 处为空气：当前 level brick 内 DDA（Douglas dda）----
    let log2 = level * 2;
    let s = 1i32 << log2; // 子块边长 voxel：1/4/16/64
    let mut side = [1e30f32; 3];
    for i in 0..3 {
      if rd[i].abs() > 1e-30 {
        // side_distance_for_ray：v 对齐到 s 的基址；正向 → 基址+s，负向 → 基址
        let base = v[i] & !(s - 1);
        let boundary = if sign[i] >= 0 { base + s } else { base };
        side[i] = ((boundary as f32 - ro_c[i]) * inv_rd[i]).max(cur_t);
      }
    }
    // 每轴步长 t 增量（level 不变则不变）：inner 里 O(1) 加法
    let step_inc =
      [s as f32 * inv_rd[0].abs(), s as f32 * inv_rd[1].abs(), s as f32 * inv_rd[2].abs()];
    let mut step_axis: usize;
    let mut changed = false;
    loop {
      // 选最近边界轴
      let min = if side[0] <= side[1] && side[0] <= side[2] {
        0
      } else if side[1] <= side[2] {
        1
      } else {
        2
      };
      step_axis = min;
      cur_t = side[min];
      if cur_t >= t1 {
        return None; // 段内再无子块可入
      }
      let old_cell = (v[min] >> log2) & 3;
      // 沿 min 轴整子块跨越
      v[min] += sign[min] * s;
      side[min] += step_inc[min];
      face = (min * 2) as u8 + if sign[min] < 0 { 1 } else { 0 };
      // 跨出 brick（4 子块）？正向往 3→外、负向往 0→外
      let crossed = if sign[min] >= 0 { old_cell == 3 } else { old_cell == 0 };
      if crossed {
        changed = true;
        break;
      }
      // 新子块内容：level 0 查 inline palette（mask!=0 inline 叶才有）；
      // level 1..3 = 分裂位或节点统一实体色。
      // 命中直接返回（cur_t=进入距离、face=进入面）——省一整轮外层
      // （traverse 节点 load + side 重算）；仅「分裂子块」回 traverse 下钻。
      let b = bricks[level as usize];
      let cell = [(v[0] >> log2) & 3, (v[1] >> log2) & 3, (v[2] >> log2) & 3];
      let idx = (cell[2] * 16 + cell[1] * 4 + cell[0]) as usize;
      if level == 0 {
        // bit=1（非空体素）才 load inline word；bit=0 空气体素零 load
        if b.mask != 0 && (b.mask & (1u64 << idx)) != 0 {
          let w = b_struct[b.addr + 3 + (idx >> 2)];
          let dp = ((w >> ((idx & 3) * 8)) & 0xFF) as u8;
          if dp != 0 {
            return Some((cur_t, dp, face, v));
          }
        } else if b.mask == 0 && b.pal != 0 {
          // 防御：uniform 叶（正常下钻快路径已处理）
          return Some((cur_t, b.pal, face, v));
        }
      } else {
        let mb = (b.mask & (1u64 << idx)) != 0;
        if mb {
          break; // 分裂子块 → 回 traverse 下钻
        }
        if b.pal != 0 {
          return Some((cur_t, b.pal, face, v)); // 统一实体
        }
      }
      // 空气子块 → 回 loop 顶重选 min 轴继续
    }
    if changed {
      level += 1;
      if level > 3 {
        return None;
      }
    }
    // ---- firstTrailingBit 层级自适应跨级跳（Douglas march 尾部）----
    // 步进轴新坐标的尾随零位 = 对齐 run 长度：正向 comp=对齐基址（tz 直接读），
    // 负向 comp=区域尾址+1（基址|~mask 后 +1）。tz>>1 = 可跨步的最粗 level。
    let positive = sign[step_axis] >= 0;
    let cur_log2 = level * 2;
    let m: u32 = 0xFFFF_FFFFu32.wrapping_shl(cur_log2);
    let vmin_u = v[step_axis] as u32; // i32→u32 环绕（负值公式自然处理）
    let comp = if positive { vmin_u & m } else { (vmin_u & m) | !m };
    let tz = comp.wrapping_add(if positive { 0 } else { 1 }).trailing_zeros();
    let new_level = tz >> 1;
    level = level.max(new_level);
    if level > 3 {
      return None; // 跨出 chunk（tz≥8）
    }
    // 对齐快照：v 钳到 cur_t 射线点所在的当前 level 区域，步进轴取精确边界整数
    // （其余轴按射线实际位置吸附，消除只沿单轴步进的漂移）
    let mi = m as i32;
    let base = [v[0] & mi, v[1] & mi, v[2] & mi];
    let p = [ro_c[0] + rd[0] * cur_t, ro_c[1] + rd[1] * cur_t, ro_c[2] + rd[2] * cur_t];
    for i in 0..3 {
      let pf = p[i].floor() as i32;
      v[i] = pf.clamp(base[i], base[i] + !mi);
    }
    v[step_axis] = comp as i32;
  }
}

/// 镜像 WGSL trace_grid 的「局部 AABB slab + chunk 间 256³ A&W + trace_chunk」段。
///
/// ro/rd 为局部 voxel 坐标（rd 含 1/scale；t 为射线参数，主世界/物体同一标尺）；
/// 局部 AABB [l_min, l_max]（voxel）；view 携带 chunk 窗口与 b_struct。
/// chunk 步数上限与 WGSL make_grid 一致：(dims.x+dims.y+dims.z)*3 + 16。
fn trace_volume_tree(
  view: &BrickMapView,
  ro: Vec3,
  rd: Vec3,
  l_min: Vec3,
  l_max: Vec3,
  t_cap: f32,
) -> Option<TreeHit> {
  // ---- 局部 AABB slab ----
  let (tl_enter, tl_exit) = slab_box(ro, rd, l_min, l_max, 0.0, t_cap);
  if tl_exit < tl_enter.max(0.0) {
    return None;
  }
  let tl0 = tl_enter.max(0.0);
  let tl1 = tl_exit.min(t_cap);
  if tl1 <= tl0 {
    return None;
  }
  let ro_a = [ro.x, ro.y, ro.z];
  let rd_a = [rd.x, rd.y, rd.z];
  let sign = [
    if rd.x >= 0.0 { 1 } else { -1 },
    if rd.y >= 0.0 { 1 } else { -1 },
    if rd.z >= 0.0 { 1 } else { -1 },
  ];
  let delta = [
    if rd.x.abs() > 1e-30 { rd.x.abs().recip() } else { 1e30 },
    if rd.y.abs() > 1e-30 { rd.y.abs().recip() } else { 1e30 },
    if rd.z.abs() > 1e-30 { rd.z.abs().recip() } else { 1e30 },
  ];
  let delta_c = [delta[0] * 256.0, delta[1] * 256.0, delta[2] * 256.0];
  // ---- chunk 间 A&W（256³ 一格）+ chunk 内层次 mask DDA ----
  let start = ro + rd * tl0;
  let mut ci = [
    (start.x / 256.0).floor() as i32,
    (start.y / 256.0).floor() as i32,
    (start.z / 256.0).floor() as i32,
  ];
  let mut tmax_c = [1e30f32; 3];
  for i in 0..3 {
    if rd_a[i].abs() > 1e-30 {
      let side = if sign[i] >= 0 { 1.0 } else { 0.0 };
      let bnd = (ci[i] as f32 + side) * 256.0;
      tmax_c[i] = ((bnd - ro_a[i]) / rd_a[i]).max(tl0);
    }
  }
  let mut t_enter_c = tl0;
  // 首 chunk 进入面：normalize(-rd) 兜底（相机贴面/UB；按约定 UB 直接返回该 voxel）
  let mut entry_face = face_index_from_normal(-rd.normalize_or_zero());
  let dims = view.dims();
  let budget = (dims.x + dims.y + dims.z) as u32 * 3 + 16;
  for _ in 0..budget {
    let t_exit_c = tmax_c[0].min(tmax_c[1]).min(tmax_c[2]);
    let t1 = t_exit_c.min(tl1);
    // 窗口查找（窗口外 / entry=0 的空 chunk = 空气，直接步进）
    if let Some(chunk_base) = view.chunk_base(IVec3::new(ci[0], ci[1], ci[2])) {
      let chunk_min = [ci[0] as f32 * 256.0, ci[1] as f32 * 256.0, ci[2] as f32 * 256.0];
      if let Some((t, pal, face_id, v)) = trace_chunk_cpu(
        view.b_struct(),
        chunk_base,
        chunk_min,
        ro_a,
        rd_a,
        sign,
        t_enter_c,
        t1,
        entry_face,
      ) {
        // chunk 局部 v → grid 局部体素（镜像 WGSL trace_grid 的 ci*256 换算）
        let voxel = IVec3::new(v[0], v[1], v[2]) + IVec3::new(ci[0], ci[1], ci[2]) * 256;
        return Some(TreeHit { t, pal, face_id, voxel });
      }
    }
    if t_exit_c >= tl1 {
      break;
    }
    if tmax_c[0] <= tmax_c[1] && tmax_c[0] <= tmax_c[2] {
      t_enter_c = tmax_c[0];
      tmax_c[0] += delta_c[0];
      ci[0] += sign[0];
      entry_face = if sign[0] < 0 { 1 } else { 0 };
    } else if tmax_c[1] <= tmax_c[2] {
      t_enter_c = tmax_c[1];
      tmax_c[1] += delta_c[1];
      ci[1] += sign[1];
      entry_face = if sign[1] < 0 { 3 } else { 2 };
    } else {
      t_enter_c = tmax_c[2];
      tmax_c[2] += delta_c[2];
      ci[2] += sign[2];
      entry_face = if sign[2] < 0 { 5 } else { 4 };
    }
  }
  None
}

/// 主世界层次 DDA（identity 变换）：局部 AABB = chunk 窗口范围（与 WGSL
/// make_grid idx=0 的 is_world 分支一致）。
///
/// 语义同 `cpu_reference_dda_ray`：射线 (origin, dir, t∈[0,t_max]) 上首个非空体素。
pub fn cpu_reference_dda_ray_tree(
  buffers: &BrickMapBuffers,
  origin_voxel: Vec3,
  dir_voxel: Vec3,
  t_max: f32,
) -> Option<TreeHit> {
  let view = BrickMapView::new(buffers);
  let origin = view.origin();
  let dims = view.dims();
  let l_min = Vec3::new((origin.x * 256) as f32, (origin.y * 256) as f32, (origin.z * 256) as f32);
  let l_max = Vec3::new(
    ((origin.x + dims.x) * 256) as f32,
    ((origin.y + dims.y) * 256) as f32,
    ((origin.z + dims.z) * 256) as f32,
  );
  trace_volume_tree(&view, origin_voxel, dir_voxel, l_min, l_max, t_max)
}

/// 对固定场景 & 静态 DdaCameraConfig，按 32x32 网格渲染 ASCII 画。
///
/// 返回 `Vec<char>`（32×32，row-major 32 字符换行）。
/// 字符规则：palette=1 → 'X', 2 → 'o', 3 → '#', 4 → '*', 其他非 0 → '+', 空 → '.'
pub fn cpu_dda_ascii_grid_32x32(cfg: &DdaCameraConfig, buffers: &BrickMapBuffers) -> Vec<char> {
  let w = 32usize;
  let h = 32usize;
  let mut out = Vec::with_capacity(w * h);
  let inv_vp = cfg.inv_view_proj;
  let cam_pos_voxel = cfg.position_world; // 世界单位 = voxel 单位
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
      let dir_voxel = dir_world;
      let ch = match cpu_reference_dda_ray(buffers, cam_pos_voxel, dir_voxel, 2000.0, 2048) {
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

// ============================================================================
// 多 volume CPU 参考 trace
//
// 入口 = `cpu_reference_trace_volumes` / `cpu_reference_volumes_occluded`，
// 遍历 `vols: &[(&BrickMapBuffers, VolumeTransform)]`：
// - idx 0 = 主世界（identity transform、无界 chunk HashMap）→ 直接层次 DDA
// - idx 1..N = 物体（任意 transform、单 chunk）→ AABB 预剔除 + 局部变换 + 局部
//   tile 盒 slab + 层次 DDA → 局部法线经 rot → 世界法线
// ============================================================================

use gate_voxel::VolumeTransform;

/// 统一 volume 命中记录。
///
/// `obj_id` 约定与 `Volumes` 一致：-1 = 主世界，0..N-1 = 物体索引
/// （对应 `Volumes.list[1..]` 的 0-based 索引）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VolumeHit {
  pub t: f32,
  pub pal: u8,
  pub obj_id: i32,
  pub normal: Vec3,
}

/// slab 法射线-AABB 求交（与 WGSL `slab_box` / `cpu_reference_dda_ray_aabb_skip` 同型）。
/// 返回 (t_enter, t_exit)；t_exit < t_enter 表示不相交。
fn slab_box(ro: Vec3, rd: Vec3, mn: Vec3, mx: Vec3, t0: f32, t1: f32) -> (f32, f32) {
  let o = [ro.x, ro.y, ro.z];
  let d = [rd.x, rd.y, rd.z];
  let lo = [mn.x, mn.y, mn.z];
  let hi = [mx.x, mx.y, mx.z];
  let mut t_enter = t0;
  let mut t_exit = t1;
  for i in 0..3 {
    if d[i].abs() < 1e-30 {
      if o[i] < lo[i] || o[i] > hi[i] {
        return (1.0, 0.0); // 平行且在外 → miss 哨兵
      }
    } else {
      let ta = (lo[i] - o[i]) / d[i];
      let tb = (hi[i] - o[i]) / d[i];
      t_enter = t_enter.max(ta.min(tb));
      t_exit = t_exit.min(tb.max(ta));
    }
  }
  (t_enter, t_exit)
}

/// 单物体 ray：世界 AABB 预剔除 + 局部变换 + 局部 [0,256]³ slab + 层次栈式 DDA。
/// 返回 (全局 t, palette, **局部**面法线) 或 None。t 标尺全局（rd 含 1/scale，
/// 局部射线参数 = 世界射线参数，与 WGSL trace_grid 同一标尺）。
fn cpu_reference_object_ray_unified(
  bufs: &BrickMapBuffers,
  tr: VolumeTransform,
  origin: Vec3,
  dir: Vec3,
  t_cap: f32,
) -> Option<(f32, u8, Vec3)> {
  // ---- 世界 AABB 预剔除 ----
  let (w_mn, w_mx) = tr.world_aabb();
  let (t_enter, t_exit) = slab_box(origin, dir, w_mn, w_mx, 0.0, t_cap);
  if t_exit < t_enter.max(0.0) || t_enter >= t_cap {
    return None;
  }
  let t_hi_cap = t_exit.min(t_cap);
  if t_hi_cap <= t_enter.max(0.0) {
    return None;
  }
  // ---- 局部变换（rd 含 1/scale → t 标尺不变）----
  let wp = origin - tr.pos;
  let ro =
    Vec3::new(wp.dot(tr.rot.x_axis), wp.dot(tr.rot.y_axis), wp.dot(tr.rot.z_axis)) / tr.scale;
  let rd =
    Vec3::new(dir.dot(tr.rot.x_axis), dir.dot(tr.rot.y_axis), dir.dot(tr.rot.z_axis)) / tr.scale;
  // ---- 局部 [0,256]³ slab + chunk 间 + 层次 DDA（镜像 WGSL trace_grid 物体分支）----
  let view = BrickMapView::new(bufs);
  let hit = trace_volume_tree(&view, ro, rd, Vec3::ZERO, Vec3::splat(256.0), t_hi_cap)?;
  let n_local = face_normal_from_index(hit.face_id);
  Some((hit.t, hit.pal, n_local))
}

/// `trace_volumes()` CPU 参考：遍历所有 volume 取最近命中（层次栈式 mask DDA）。
///
/// - idx 0 = 主世界：identity transform → `cpu_reference_dda_ray_tree`
///   （局部 AABB = chunk 窗口范围，镜像 WGSL make_grid idx=0）
/// - idx 1..N = 物体：`cpu_reference_object_ray_unified`（世界 AABB + 局部 DDA）
///
/// 命中按 t 升序排序取最近；任一前置 volume 命中即压缩后续 volume 的 t_cap。
pub fn cpu_reference_trace_volumes(
  vols: &[(&BrickMapBuffers, VolumeTransform)],
  origin: Vec3,
  dir: Vec3,
  t_max: f32,
) -> Option<VolumeHit> {
  let mut best: Option<VolumeHit> = None;
  for (i, (bufs, tr)) in vols.iter().enumerate() {
    let obj_id = if i == 0 { -1 } else { (i - 1) as i32 };
    let cap = best.as_ref().map_or(t_max, |b| b.t.min(t_max));
    let hit = if obj_id == -1 {
      cpu_reference_dda_ray_tree(bufs, origin, dir, cap)
        .map(|h| (h.t, h.pal, face_normal_from_index(h.face_id)))
    } else {
      cpu_reference_object_ray_unified(bufs, *tr, origin, dir, cap)
        .map(|(t, pal, n_local)| (t, pal, (tr.rot * n_local).normalize_or_zero()))
    };
    if let Some((t, pal, normal)) = hit
      && best.as_ref().is_none_or(|b| t < b.t)
    {
      best = Some(VolumeHit { t, pal, obj_id, normal });
    }
  }
  best
}

/// `trace_volumes()` 遮挡快路径：t_max 内**任一**命中即 true。
/// 不做最近比较；阴影射线占比大时（每像素 × 光源 × 采样），此路径省去逐 volume t 排序。
pub fn cpu_reference_volumes_occluded(
  vols: &[(&BrickMapBuffers, VolumeTransform)],
  origin: Vec3,
  dir: Vec3,
  t_max: f32,
) -> bool {
  for (i, (bufs, tr)) in vols.iter().enumerate() {
    let hit = if i == 0 {
      cpu_reference_dda_ray_tree(bufs, origin, dir, t_max).is_some()
    } else {
      cpu_reference_object_ray_unified(bufs, *tr, origin, dir, t_max).is_some()
    };
    if hit {
      return true;
    }
  }
  false
}

#[cfg(test)]
mod dda_ref_tests {
  use super::*;
  use crate::brickmap::BrickMapBuilder;
  use gate_voxel::{VolumeGrid, fill_box, fill_sphere};
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

  /// 阴影射线起点锚在「命中体素中心 + n×(0.5 + SHADOW_SURFACE_EPS)」（dda_main）。
  /// 两个不变量一起锁：
  ///   · 不加 eps → 对 -X/-Y/-Z 面起点正好落在体素**低侧边界**，floor 落回自己 → t=0 自命中
  ///     （即「无遮挡」误判）；
  ///   · 加 eps 后 → 起点落在邻域空气格，背离墙面的射线必须无命中（真的遮挡测试）。
  ///   · +X 面（高侧边界）本来就正常，作为对照。
  #[test]
  fn shadow_ray_start_offset_clears_hit_voxel() {
    use super::wgsl_consts::SHADOW_SURFACE_EPS;
    // 2 体素厚的墙：x ∈ [10,12)，y/z ∈ [0,64)
    let mut g = VolumeGrid::new();
    fill_box(&mut g, IVec3::new(10, 0, 0), IVec3::new(2, 64, 64), 1);
    let buffers = BrickMapBuilder::build_full(&g).buffers().clone();
    let trace = |v: IVec3, n: glam::Vec3, eps: f32| {
      let center = v.as_vec3() + glam::Vec3::splat(0.5);
      cpu_reference_dda_ray_tree(&buffers, center + n * (0.5 + eps), n, 8192.0)
    };
    let lo = (IVec3::new(10, 32, 32), glam::Vec3::new(-1.0, 0.0, 0.0));
    let hi = (IVec3::new(11, 32, 32), glam::Vec3::new(1.0, 0.0, 0.0));
    // 无 eps 时低侧面自命中（返回的就是出发点体素）
    let bad = trace(lo.0, lo.1, 0.0).expect("旧写法应自命中（这就是漏光的根因）");
    assert_eq!(bad.voxel, lo.0, "自命中应发生在出发点体素");
    // eps 推过边界后，低侧面必须与高侧面一样「背离墙面 → 无命中」
    for (v, n) in [lo, hi] {
      assert!(
        trace(v, n, SHADOW_SURFACE_EPS).is_none(),
        "v={v:?} n={n:?}: 外推 {SHADOW_SURFACE_EPS} 后仍不该有命中（实得 {:?}）",
        trace(v, n, SHADOW_SURFACE_EPS)
      );
    }
  }

  /// 0..1 f32
  fn frand(state: &mut u64) -> f32 {
    (xorshift64(state) as f32) / (u32::MAX as f32)
  }

  /// 测试专用相机：小场景（voxel 0..16）配近相机，与 demo 的 build_static 解耦
  fn test_cam() -> DdaCameraConfig {
    let eye = Vec3::new(24.0, 20.0, 24.0);
    let target = Vec3::new(8.0, 8.0, 8.0);
    let aspect = VIEW_SIZE.x as f32 / VIEW_SIZE.y as f32;
    let proj = Mat4::perspective_rh(60.0_f32.to_radians(), aspect, 1.0, 4000.0);
    let view = Mat4::look_at_rh(eye, target, Vec3::Y);
    let view_proj = proj.mul(view);
    DdaCameraConfig { view_proj, inv_view_proj: view_proj.inverse(), position_world: eye }
  }

  fn build_box_sphere_scene() -> (BrickMapBuffers, DdaCameraConfig) {
    let mut g = VolumeGrid::new();
    // box 16³ voxel → 0..16，pal=1
    fill_box(&mut g, IVec3::ZERO, IVec3::splat(16), 1);
    // sphere at (8,8,8) voxel, r=4 voxel (1cm), pal=2
    fill_sphere(&mut g, IVec3::new(8, 8, 8), 4, 2);
    // hotspots pal 3+4: 1x1x1 at (12, 2, 12) pal3 and (14,2,14) pal4
    g.set_voxel_ivec3(IVec3::new(12, 2, 12), 3);
    g.set_voxel_ivec3(IVec3::new(14, 2, 14), 4);
    let b = BrickMapBuilder::build_full(&g);
    (b.buffers().clone(), test_cam())
  }

  /// 50 条随机射线：cpu_reference_dda_ray 和 "独立枚举细格" 双方法结果一致
  /// （"独立枚举" = 不调用 BrickMapView.get_voxel 而是逐位置 floor 枚举 get_voxel，
  ///  实际上 DDA ref 实现已经用 view.get_voxel，第二验证方式是暴力从 t=0 到 t 按 0.5 voxel 扫）
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
      let dir_voxel = dir_world; // 世界单位 = voxel 单位，dir 模长 1
      let a = cpu_reference_dda_ray(&bufs, origin, dir_voxel, 2000.0, 2048);
      // Brute-force：以 0.5 voxel 实际距离扫 2000 步；取首个非空
      // dir_voxel.length() = 1，t = 实际 voxel 距离
      let d_len = dir_voxel.length().max(1e-20);
      let step_voxel = 0.5f32; // 0.5 voxel 步长保证 1-voxel 胞至少 2 采样
      let mut brute: Option<(f32, u8)> = None;
      let view = BrickMapView::new(&bufs);
      for i in 0..2000u32 {
        let s = i as f32 * step_voxel; // 实际 voxel 距离
        if s / d_len >= 2000.0 {
          break;
        }
        let t = s / d_len;
        let p = origin + dir_voxel * t;
        let voxel = IVec3::new(p.x.floor() as i32, p.y.floor() as i32, p.z.floor() as i32);
        if let Some(pal) = view.get_voxel(voxel) {
          brute = Some((t, pal));
          break;
        }
      }
      match (a, brute) {
        (Some((ta, pa)), Some((tb, pb))) => {
          assert_eq!(pa, pb, "palette mismatch");
          // 0.5 voxel coarse vs A&W 精确边界：ta 通常在 (tb-1, tb+0.5) 区间
          assert!((ta - tb).abs() <= 1.0, "t mismatch {ta} vs {tb}");
        }
        (None, None) => {}
        (a, b) => {
          panic!("hit pattern diff: a={a:?} b={b:?} origin={origin:?} dir={dir_voxel:?}")
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
      println!("probe get_voxel({:?}) {:?} = {:?}", p, msg, view.get_voxel(p));
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
    assert!(non_empty >= 32, "non_empty pixels only {non_empty}（<32，画面全空可疑）");
    // X 字符（盒体 pal1）≥ 8
    let xs: usize = grid.iter().filter(|c| **c == 'X').count();
    assert!(xs >= 8, "box pal=1 only {xs} 'X'");
  }

  /// 300 条随机射线（含远镜头 / 近镜头 / 相机在 AABB 内 / 轴平行退化情形）：
  /// cpu_reference_dda_ray（从 origin 直接走，足够多 steps）和
  /// cpu_reference_dda_ray_aabb_skip（先 AABB 跳步）两者结果逐像素一致
  /// （palette 严格相等，t 差 ≤ 1 voxel）。
  #[test]
  fn aabb_skip_equivalence_300_rays() {
    let (bufs, _cfg) = build_box_sphere_scene();
    // 场景 AABB：voxel 坐标 [0, 0, 0] .. [16, 16, 16]（brickmap 全在此区间）
    let aabb_min = Vec3::new(0.0, 0.0, 0.0);
    let aabb_max = Vec3::new(16.0, 16.0, 16.0);
    let mut state: u64 = 12345;
    let mut count = 0usize;
    // 5 种相机场景，每种 60 条随机视锥射线
    let cameras: Vec<(Vec3, Vec3)> = vec![
      // 近相机（在 AABB 外 24 voxel 处，和 test_cam 一致）
      (Vec3::new(24.0, 20.0, 24.0), Vec3::new(8.0, 8.0, 8.0)),
      // 远相机（在 AABB 外 50k voxel 处——模拟用户 zoom-out 很多下的情况）
      (Vec3::new(50_000.0, 40_000.0, 50_000.0), Vec3::new(8.0, 8.0, 8.0)),
      // 相机在 AABB 内部（用户很近时贴脸模型）
      (Vec3::new(8.5, 8.5, 2.0), Vec3::new(8.5, 8.5, 16.0)),
      // 斜 + 轻微轴平行（xz 面视线，dir.y 很小）
      (Vec3::new(30.0, 8.0, 30.0), Vec3::new(8.0, 8.0, 8.0)),
      // 负坐标远端 + 指向 AABB（测试负 voxel 坐标 floor 与 slab 求交）
      (Vec3::new(-10_000.0, 10_000.0, -10_000.0), Vec3::new(8.0, 8.0, 8.0)),
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
        // full 版 DDA：给足够大的 steps = 2_000_000（覆盖远镜头 50k voxel 的距离），
        // t_max = frustum_len（视锥外不算命中）。max_steps 足够大所以一定能走穿。
        let full = cpu_reference_dda_ray(&bufs, *eye, dir, frustum_len, 2_000_000);
        // AABB skip 版：max_steps 只给 2048（GPU 最终上限，比 AABB 对角 ~1732 略大），
        // 理论上只要 AABB 在视锥内、2048 一定能走穿整个 AABB 厚度内的片段。
        let skip =
          cpu_reference_dda_ray_aabb_skip(&bufs, *eye, dir, frustum_len, 2048, aabb_min, aabb_max);
        match (full, skip) {
          (Some((tf, pf)), Some((ts, ps))) => {
            assert_eq!(pf, ps, "palette diff eye={eye:?} tgt={target:?} full_t={tf} skip_t={ts}");
            // t 差 ≤ 1 voxel（浮点计算的 floor 与首胞命中边界误差）
            assert!((tf - ts).abs() <= 1.0, "t diff eye={eye:?} full_t={tf} skip_t={ts} pal={pf}");
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

  /// 两级 DDA 等价性：多 cell 体 + cell 间隙 + 负坐标场景 + 固定退化方向 +
  /// 300 条随机射线，cpu_reference_dda_ray vs cpu_reference_dda_ray_two_level。
  /// 命中/未命中与 palette 严格一致；hit_t 容差 1.0 voxel（长路径 ulp 漂移只影响 t 值）。
  #[test]
  fn two_level_equivalence_300_rays() {
    // ---- 多物体场景（一个 chunk 内）：外框盒 + 内浮球 + 独立柱，间隙跨越 ----
    let mut g = VolumeGrid::new();
    // 外框壳：32³ box
    fill_box(&mut g, IVec3::ZERO, IVec3::splat(32), 1);
    // 独立物体：偏移 (96, 32, 96) 的 16³ box（离开外框，中间隔空隙）
    fill_box(&mut g, IVec3::new(96, 32, 96), IVec3::splat(16), 2);
    // 悬浮球：中心 (64, 64, 64)，r=8
    fill_sphere(&mut g, IVec3::new(64, 64, 64), 8, 3);
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
        (Some((tf, pf)), Some(h)) => {
          let (tt, pt) = (h.t, h.pal);
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
      // 起点距场景中心 32~600 voxel 的球壳
      let r = 32.0 + frand(&mut state) * 568.0;
      let theta = frand(&mut state) * std::f32::consts::TAU;
      let phi = (frand(&mut state) * 2.0 - 1.0).acos();
      let origin =
        Vec3::new(r * phi.sin() * theta.cos(), r * phi.cos(), r * phi.sin() * theta.sin())
          + Vec3::new(64.0, 48.0, 64.0);
      // 随机单位方向（球面均匀）
      let dtheta = frand(&mut state) * std::f32::consts::TAU;
      let dphi = (frand(&mut state) * 2.0 - 1.0).acos();
      let dir = Vec3::new(dphi.sin() * dtheta.cos(), dphi.cos(), dphi.sin() * dtheta.sin());
      let full = cpu_reference_dda_ray(&bufs, origin, dir, 2048.0, 2_000_000);
      let two = cpu_reference_dda_ray_two_level(&bufs, origin, dir, 2048.0, 16384);
      match (full, two) {
        (Some((tf, pf)), Some(h)) => {
          let (tt, pt) = (h.t, h.pal);
          assert_eq!(
            pf, pt,
            "ray[{ray}] palette diff o={origin:?} d={dir:?} full_t={tf} two_t={tt}"
          );
          assert!((tf - tt).abs() <= 1.0, "ray[{ray}] t diff {tf} vs {tt} o={origin:?} d={dir:?}");
        }
        (None, None) => {}
        (f, t) => {
          panic!("ray[{ray}] hit mismatch o={origin:?} d={dir:?}\n  full={f:?}\n  two={t:?}")
        }
      }
    }
  }

  /// 层次栈式 mask DDA 等价性：跨 chunk（含负坐标窗口）场景 + 8 条轴平行退化射线
  /// + 300 条随机射线，cpu_reference_dda_ray（full 逐体素参考）vs
  ///   cpu_reference_dda_ray_tree（WGSL trace_grid/trace_chunk 的 CPU 逐字镜像）。
  ///   hit/miss 与 palette 严格一致；hit_t 容差 1.0 voxel；命中面法线必须反向于射线
  ///   （n·dir < 0：face_id 是射线穿入面）。
  #[test]
  fn tree_traversal_equivalence_300_rays() {
    // ---- 跨 chunk / 负坐标场景（fill_box 第二参 = size，区间 [min, min+size)）----
    let mut g = VolumeGrid::new();
    // 负区大块：x/y/z -512..-448（chunk -2 一带，窗口 origin 含负）
    fill_box(&mut g, IVec3::new(-512, -64, -512), IVec3::new(64, 96, 64), 1);
    // 正区远块：512..608（chunk 2）
    fill_box(&mut g, IVec3::new(512, 0, 512), IVec3::new(96, 64, 96), 2);
    // 跨界球：心 (256,48,256) r=40，跨 x=256 / z=256 两条 chunk 边界
    fill_sphere(&mut g, IVec3::new(256, 48, 256), 40, 3);
    // 原点小盒：-16..16（chunk 0 与负 chunk 交界）
    fill_box(&mut g, IVec3::new(-16, -16, -16), IVec3::splat(32), 4);
    let bufs = BrickMapBuilder::build_full(&g).buffers().clone();
    let gl = &bufs.globals;
    // sanity：窗口确实跨负坐标且多 chunk（否则测试场景没有覆盖到目标路径）
    assert!(
      gl.index_origin_x < 0 || gl.index_origin_y < 0 || gl.index_origin_z < 0,
      "窗口 origin 应含负分量：({:?})",
      (gl.index_origin_x, gl.index_origin_y, gl.index_origin_z)
    );
    assert!(gl.tile_count >= 4, "应跨多个 chunk：tile_count={}", gl.tile_count);

    let cmp = |tag: &str, o: Vec3, d: Vec3, t_max: f32| {
      let full = cpu_reference_dda_ray(&bufs, o, d, t_max, 4_000_000);
      let tree = cpu_reference_dda_ray_tree(&bufs, o, d, t_max);
      match (full, tree) {
        (Some((tf, pf)), Some(h)) => {
          assert_eq!(pf, h.pal, "[{tag}] palette diff o={o:?} d={d:?} full_t={tf} tree_t={}", h.t);
          assert!((tf - h.t).abs() <= 1.0, "[{tag}] t diff {tf} vs {} o={o:?} d={d:?}", h.t);
          // 面法线反向于射线（穿入面）；相机贴面 UB 射线 entry_face 同样反向
          let n = face_normal_from_index(h.face_id);
          assert!(
            n.dot(d) < 0.001,
            "[{tag}] face normal {n:?} not against dir {d:?} (face_id={})",
            h.face_id
          );
        }
        (None, None) => {}
        (f, t) => panic!("[{tag}] hit mismatch o={o:?} d={d:?}\n  full={f:?}\n  tree={t:?}"),
      }
    };

    // ---- 8 条轴平行 / 退化射线（穿各特征 + 起点在体内 UB）----
    let axis_rays: [(Vec3, Vec3); 8] = [
      (Vec3::new(-700.0, 0.0, -480.0), Vec3::X),  // +x 穿负区盒
      (Vec3::new(700.0, 32.0, 560.0), -Vec3::X),  // -x 穿正区远块
      (Vec3::new(256.0, -200.0, 256.0), Vec3::Y), // +y 穿跨界球
      (Vec3::new(256.0, 300.0, 256.0), -Vec3::Y), // -y 穿球
      (Vec3::new(256.0, 48.0, 100.0), Vec3::Z),   // +z 穿球心
      (Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.3, 0.2)), // 起点在原点盒内（UB 贴面）
      (Vec3::new(-900.0, -900.0, -900.0), Vec3::ONE), // 负远端对角穿
      (Vec3::new(900.0, 900.0, 900.0), -Vec3::ONE), // 正远端对角
    ];
    for (i, (o, d)) in axis_rays.iter().enumerate() {
      cmp(&format!("axis{i}"), *o, d.normalize(), 8192.0);
    }

    // ---- 300 条随机射线：球壳随机起点 + 球面均匀随机方向 ----
    let mut state: u64 = 20260904;
    for ray in 0..300 {
      // 起点距场景中心 (128,32,128) 32~1500 voxel 的球壳（覆盖 ±512 远块）
      let r = 32.0 + frand(&mut state) * 1468.0;
      let theta = frand(&mut state) * std::f32::consts::TAU;
      let phi = (frand(&mut state) * 2.0 - 1.0).acos();
      let origin =
        Vec3::new(r * phi.sin() * theta.cos(), r * phi.cos(), r * phi.sin() * theta.sin())
          + Vec3::new(128.0, 32.0, 128.0);
      let dtheta = frand(&mut state) * std::f32::consts::TAU;
      let dphi = (frand(&mut state) * 2.0 - 1.0).acos();
      let dir = Vec3::new(dphi.sin() * dtheta.cos(), dphi.cos(), dphi.sin() * dtheta.sin());
      cmp(&format!("rand{ray}"), origin, dir, 4096.0);
    }
  }

  /// demo 场景特征复刻等价性：地形分层柱（非 brick 对齐 y）+ fill_bricks(e=16) 平台
  /// + 实心墙内 clear_voxel 单体孔洞（air-in-solid）+ 倒锥叠球（跨 chunk）
  /// + 交替 palette 增量编辑 + compact_all（GC 后节点流）。
  #[test]
  fn tree_equivalence_demo_like_edit_compact() {
    use gate_voxel::{VoxelCoord, fill_bricks};
    let mut g = VolumeGrid::new();
    // ---- 地形分层柱：16×8×16（y 非 16 对齐 → 4³ 边缘碎裂），跨 chunk -1..2 ----
    let mut x = -256;
    while x < 512 {
      let mut z = -256;
      while z < 256 {
        fill_box(&mut g, IVec3::new(x, 16, z), IVec3::new(16, 8, 16), 2);
        fill_box(&mut g, IVec3::new(x, 24, z), IVec3::new(16, 8, 16), 7);
        z += 16;
      }
      x += 16;
    }
    // ---- fill_bricks(e=16) 平台（level-2 uniform 节点模式）----
    fill_bricks(&mut g, IVec3::new(-128, 32, -128), IVec3::new(96, 16, 96), 16, 4);
    // ---- 实心墙 + clear_voxel 单体孔洞阵列（demo 正殿门模式：air-in-solid）----
    fill_box(&mut g, IVec3::new(0, 48, -72), IVec3::new(64, 64, 8), 4);
    let mut wy = 48;
    while wy < 112 {
      let mut wx = 0;
      while wx < 64 {
        g.clear_voxel(VoxelCoord::from_ivec3(IVec3::new(wx, wy, -68)));
        wx += 8;
      }
      wy += 8;
    }
    // ---- 倒锥叠球（浮空岛缩样，r 递增，跨 x=256/z=256 chunk 边界）----
    fill_sphere(&mut g, IVec3::new(256, 40, 256), 40, 13);
    fill_sphere(&mut g, IVec3::new(256, 56, 256), 64, 13);
    fill_sphere(&mut g, IVec3::new(256, 72, 256), 88, 13);
    // ---- 交替 palette 增量编辑（edit_tile 模式：4³ 粒度金/青交替）----
    let mut ez = 0;
    while ez < 32 {
      let mut ey = 0;
      while ey < 32 {
        let mut ex = 0;
        while ex < 32 {
          let pal = if (ex ^ ey ^ ez) & 4 == 0 { 11u8 } else { 8u8 };
          g.set_voxel_ivec3(IVec3::new(512 + ex, 16 + ey, 512 + ez), pal);
          ex += 4;
        }
        ey += 4;
      }
      ez += 4;
    }
    // ---- GC：编辑/分裂产生的废弃节点回收后序列化 ----
    g.compact_all();
    let bufs = BrickMapBuilder::build_full(&g).buffers().clone();

    let cmp = |tag: &str, o: Vec3, d: Vec3, t_max: f32| {
      let full = cpu_reference_dda_ray(&bufs, o, d, t_max, 4_000_000);
      let tree = cpu_reference_dda_ray_tree(&bufs, o, d, t_max);
      match (full, tree) {
        (Some((tf, pf)), Some(h)) => {
          assert_eq!(pf, h.pal, "[{tag}] palette diff o={o:?} d={d:?} full_t={tf} tree_t={}", h.t);
          assert!((tf - h.t).abs() <= 1.0, "[{tag}] t diff {tf} vs {} o={o:?} d={d:?}", h.t);
          let n = face_normal_from_index(h.face_id);
          assert!(
            n.dot(d) < 0.001,
            "[{tag}] face normal {n:?} not against dir {d:?} (face_id={})",
            h.face_id
          );
        }
        (None, None) => {}
        (f, t) => panic!("[{tag}] hit mismatch o={o:?} d={d:?}\n  full={f:?}\n  tree={t:?}"),
      }
    };

    // ---- 轴平行 / 退化射线（穿各特征 + 起点在体内 UB）----
    let axis_rays: [(Vec3, Vec3); 10] = [
      (Vec3::new(-400.0, 20.0, -64.0), Vec3::X), // +x 贴地形层穿墙
      (Vec3::new(600.0, 20.0, 0.0), -Vec3::X),   // -x 穿地形+编辑区
      (Vec3::new(32.0, 200.0, -68.0), -Vec3::Y), // -y 穿墙孔洞阵列
      (Vec3::new(256.0, 300.0, 256.0), -Vec3::Y), // -y 穿叠球锥顶
      (Vec3::new(256.0, -100.0, 256.0), Vec3::Y), // +y 从下穿叠球
      (Vec3::new(256.0, 48.0, -200.0), Vec3::Z), // +z 穿球心+地形
      (Vec3::new(0.0, 20.0, 0.0), Vec3::ONE),    // 起点在地形内（UB 贴面）
      (Vec3::new(-300.0, -100.0, -300.0), Vec3::ONE), // 负远端对角
      (Vec3::new(600.0, 400.0, 400.0), -Vec3::ONE), // 正远端对角
      (Vec3::new(-300.0, 33.0, 0.0), Vec3::X),   // +x 掠平台顶面（浅角）
    ];
    for (i, (o, d)) in axis_rays.iter().enumerate() {
      cmp(&format!("axis{i}"), *o, d.normalize(), 8192.0);
    }

    // ---- 300 条随机射线：球壳起点（覆盖 -256..512 场景）+ 球面均匀方向 ----
    let mut state: u64 = 20260905;
    for ray in 0..300 {
      let r = 32.0 + frand(&mut state) * 900.0;
      let theta = frand(&mut state) * std::f32::consts::TAU;
      let phi = (frand(&mut state) * 2.0 - 1.0).acos();
      let origin =
        Vec3::new(r * phi.sin() * theta.cos(), r * phi.cos(), r * phi.sin() * theta.sin())
          + Vec3::new(128.0, 48.0, 0.0);
      let dtheta = frand(&mut state) * std::f32::consts::TAU;
      let dphi = (frand(&mut state) * 2.0 - 1.0).acos();
      let dir = Vec3::new(dphi.sin() * dtheta.cos(), dphi.cos(), dphi.sin() * dtheta.sin());
      cmp(&format!("rand{ray}"), origin, dir, 4096.0);
    }
  }

  /// firstTrailingBit 层级自适应大步进 fuzz：跨 chunk 随机块（4³/16³/64³ 多尺度，
  /// 逼出统一叶节点、inline 叶、多层对齐跨跳）+ 2000 条球壳随机射线 + 16 条
  /// 轴平行/对角退化射线，暴力逐体素参考 vs 新整数体素层级遍历严格比对
  /// （palette 一致、t 容差 1.0、命中面法线反向）。
  #[test]
  fn tree_traversal_fuzz_2000_rays_multiscale() {
    let mut g = VolumeGrid::new();
    let mut state: u64 = 0xDEAD_BEEF_0001_0001;
    // ---- 随机块：尺度 4/16/64 三档，坐标跨 chunk（±768），少量单体素 ----
    for _ in 0..220 {
      let scale = [4i32, 16, 64][(frand(&mut state) * 3.0) as usize];
      let bx = ((frand(&mut state) * 384.0) as i32 - 192) * 4;
      let by = ((frand(&mut state) * 64.0) as i32) * 4;
      let bz = ((frand(&mut state) * 384.0) as i32 - 192) * 4;
      let pal = 1 + (frand(&mut state) * 12.0) as u8;
      fill_box(&mut g, IVec3::new(bx, by, bz), IVec3::splat(scale), pal);
    }
    for _ in 0..400 {
      let vx = ((frand(&mut state) * 1024.0) as i32) - 512;
      let vy = (frand(&mut state) * 160.0) as i32;
      let vz = ((frand(&mut state) * 1024.0) as i32) - 512;
      g.set_voxel_ivec3(IVec3::new(vx, vy, vz), 1 + (frand(&mut state) * 12.0) as u8);
    }
    // 几个跨 chunk 大球（跨边界对齐场景）
    fill_sphere(&mut g, IVec3::new(256, 40, 256), 56, 5);
    fill_sphere(&mut g, IVec3::new(-256, 24, -256), 40, 6);
    let bufs = BrickMapBuilder::build_full(&g).buffers().clone();

    let cmp = |tag: &str, o: Vec3, d: Vec3, t_max: f32| {
      let full = cpu_reference_dda_ray(&bufs, o, d, t_max, 8_000_000);
      let tree = cpu_reference_dda_ray_tree(&bufs, o, d, t_max);
      match (full, tree) {
        (Some((tf, pf)), Some(h)) => {
          assert_eq!(pf, h.pal, "[{tag}] palette diff o={o:?} d={d:?} full_t={tf} tree_t={}", h.t);
          assert!((tf - h.t).abs() <= 1.0, "[{tag}] t diff {tf} vs {} o={o:?} d={d:?}", h.t);
          // 命中体素固体性（voxel 显式携带的正确性门禁）：DDA 携带的 voxel 必须
          // 恰是 palette 一致的固体体素——着色链（per-voxel normal/GI key）以它为准
          let view = BrickMapView::new(&bufs);
          assert_eq!(
            view.get_voxel(h.voxel),
            Some(h.pal),
            "[{tag}] hit voxel {:?} not solid pal {pf} o={o:?} d={d:?}",
            h.voxel
          );
          let n = face_normal_from_index(h.face_id);
          assert!(
            n.dot(d) < 0.001,
            "[{tag}] face normal {n:?} not against dir {d:?} (face_id={})",
            h.face_id
          );
        }
        (None, None) => {}
        (f, t) => panic!("[{tag}] hit mismatch o={o:?} d={d:?}\n  full={f:?}\n  tree={t:?}"),
      }
    };

    // 轴平行/对角退化射线（贴边界坐标，压 firstTrailingBit 对齐位模式）
    let axis_rays: [(Vec3, Vec3); 16] = [
      (Vec3::new(-800.0, 0.0, 0.0), Vec3::X),
      (Vec3::new(800.0, 1.0, 0.0), -Vec3::X),
      (Vec3::new(0.0, -200.0, 0.0), Vec3::Y),
      (Vec3::new(0.0, 400.0, 0.0), -Vec3::Y),
      (Vec3::new(0.0, 8.0, -800.0), Vec3::Z),
      (Vec3::new(0.0, 8.0, 800.0), -Vec3::Z),
      (Vec3::new(-800.0, -800.0, -800.0), Vec3::ONE),
      (Vec3::new(800.0, 400.0, 800.0), -Vec3::ONE),
      (Vec3::new(-768.0, 16.0, -256.0), Vec3::X),
      (Vec3::new(-512.0, 16.0, 256.0), Vec3::X),
      (Vec3::new(256.0, 16.0, -512.0), Vec3::new(0.0, 0.0, 1.0)),
      (Vec3::new(300.0, -64.0, 300.0), Vec3::Y),
      (Vec3::new(64.0, 64.0, 64.0), Vec3::new(1.0, -0.3, 0.7)),
      (Vec3::new(-64.0, 200.0, -64.0), Vec3::new(-1.0, -1.0, -1.0)),
      (Vec3::new(512.0, 32.0, -512.0), Vec3::new(-1.0, 0.2, 1.0)),
      (Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.1, 0.05)),
    ];
    for (i, (o, d)) in axis_rays.iter().enumerate() {
      cmp(&format!("axis{i}"), *o, d.normalize(), 4096.0);
    }

    // 2000 条球壳随机射线
    for ray in 0..2000 {
      let r = 16.0 + frand(&mut state) * 1400.0;
      let theta = frand(&mut state) * std::f32::consts::TAU;
      let phi = (frand(&mut state) * 2.0 - 1.0).acos();
      let origin = Vec3::new(
        r * phi.sin() * theta.cos(),
        r * phi.cos().abs() * 0.5 + 16.0,
        r * phi.sin() * theta.sin(),
      );
      let dtheta = frand(&mut state) * std::f32::consts::TAU;
      let dphi = (frand(&mut state) * 2.0 - 1.0).acos();
      let dir = Vec3::new(dphi.sin() * dtheta.cos(), dphi.cos(), dphi.sin() * dtheta.sin());
      cmp(&format!("rand{ray}"), origin, dir, 4096.0);
    }
  }

  /// 世界查询统一接口等价性：薄墙（1 体素厚）/ 薄板 / 凹角（L 形交角）/ 掠射
  /// （|dir·面法线| ≈ 0）这几类几何是「epsilon / 起点偏置 / 层次遍历」最易分歧、
  /// 也最易漏光的形态 —— 专门锁 tree 遍历 vs 暴力逐体素参考的一致性。
  #[test]
  fn world_query_equivalence_thin_concave_grazing() {
    let mut g = VolumeGrid::new();
    // 薄墙：x=64 单层（1 体素厚）
    fill_box(&mut g, IVec3::new(64, 0, 0), IVec3::new(1, 128, 128), 1);
    // 薄板：y=32 单层
    fill_box(&mut g, IVec3::new(0, 32, 0), IVec3::new(128, 1, 128), 2);
    // 凹角：两片薄墙交于 (x=100, z=100) 棱
    fill_box(&mut g, IVec3::new(100, 0, 100), IVec3::new(1, 64, 64), 3);
    fill_box(&mut g, IVec3::new(100, 0, 100), IVec3::new(64, 64, 1), 4);
    // 悬空薄板（y 非 4³ 对齐，压 4³ 边缘碎裂）
    fill_box(&mut g, IVec3::new(-64, 17, -64), IVec3::new(96, 2, 96), 5);
    let bufs = BrickMapBuilder::build_full(&g).buffers().clone();

    let cmp = |tag: &str, o: Vec3, d: Vec3, t_max: f32| {
      let full = cpu_reference_dda_ray(&bufs, o, d, t_max, 4_000_000);
      let tree = cpu_reference_dda_ray_tree(&bufs, o, d, t_max);
      match (full, tree) {
        (Some((tf, pf)), Some(h)) => {
          assert_eq!(pf, h.pal, "[{tag}] palette diff o={o:?} d={d:?}");
          assert!((tf - h.t).abs() <= 1.0, "[{tag}] t diff {tf} vs {} o={o:?} d={d:?}", h.t);
          let n = face_normal_from_index(h.face_id);
          assert!(
            n.dot(d) < 0.001,
            "[{tag}] face normal {n:?} not against dir {d:?} (face_id={})",
            h.face_id
          );
        }
        (None, None) => {}
        (f, t) => panic!("[{tag}] hit mismatch o={o:?} d={d:?}\n  full={f:?}\n  tree={t:?}"),
      }
    };

    // 掠射 / 贴面 / 穿薄层：命中面法线与射线几乎垂直
    let grazing: [(Vec3, Vec3); 8] = [
      (Vec3::new(63.5, 40.0, 40.0), Vec3::new(-1.0, 0.0, 0.001)),
      (Vec3::new(40.0, 32.5, 40.0), Vec3::new(0.001, -1.0, 0.0)),
      (Vec3::new(150.0, 40.0, 100.5), Vec3::new(-1.0, 0.0, 0.0005)),
      (Vec3::new(64.5, 200.0, 64.0), Vec3::new(0.0, -1.0, 0.0)),
      (Vec3::new(40.0, 200.0, 40.0), Vec3::new(0.0, -1.0, 0.0)),
      (Vec3::new(130.0, 30.0, 130.0), Vec3::new(-1.0, 0.0, -1.0)),
      (Vec3::new(64.0, 64.0, -100.0), Vec3::new(0.0, 0.0, 1.0)),
      (Vec3::new(0.0, 33.0, 0.0), Vec3::new(1.0, 0.0, 0.0)),
    ];
    for (i, (o, d)) in grazing.iter().enumerate() {
      cmp(&format!("graze{i}"), *o, d.normalize(), 8192.0);
    }

    // 随机射线：球壳起点覆盖负区 / 薄板 / 凹角
    let mut state: u64 = 0xC0FF_EE00_0000_0007;
    for ray in 0..400 {
      let r = 8.0 + frand(&mut state) * 320.0;
      let theta = frand(&mut state) * std::f32::consts::TAU;
      let phi = (frand(&mut state) * 2.0 - 1.0).acos();
      let origin =
        Vec3::new(r * phi.sin() * theta.cos(), r * phi.cos() + 32.0, r * phi.sin() * theta.sin())
          + Vec3::new(32.0, 0.0, 32.0);
      let dtheta = frand(&mut state) * std::f32::consts::TAU;
      let dphi = (frand(&mut state) * 2.0 - 1.0).acos();
      let dir = Vec3::new(dphi.sin() * dtheta.cos(), dphi.cos(), dphi.sin() * dtheta.sin());
      cmp(&format!("rand{ray}"), origin, dir, 2048.0);
    }
  }
}

// ============================================================================
// BrickMapDdaPlugin — pipeline/BG/dispatch/blit 装配
// 链路：ExtractResourcePlugin → RenderStartup → PrepareBindGroups → RenderGraph dispatch →
// Core2d PostProcess blit。BG1 = brickmap 四 buffer（struct/leaves/palette + globals uniform），
// BG0 uniform 用 DdaViewUniform（DdaCameraConfig 从 main world Extract 后写入）。
// ============================================================================
use bevy::{
  core_pipeline::schedule::{Core2d, Core2dSystems, camera_driver},
  render::{
    Render, RenderApp, RenderStartup, RenderSystems,
    render_asset::RenderAssets,
    render_resource::{
      BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries,
      CachedComputePipelineId, CachedRenderPipelineId, ColorTargetState, ColorWrites,
      ComputePipelineDescriptor, Extent3d, FragmentState, PipelineCache, RenderPassDescriptor,
      SamplerBindingType, ShaderStages, StorageTextureAccess, TextureDescriptor, TextureDimension,
      TextureFormat, TextureSampleType, TextureUsages, TextureViewDescriptor, UniformBuffer,
      VertexState,
      binding_types::{
        sampler, storage_buffer_read_only_sized, storage_buffer_sized, texture_2d, texture_3d,
        texture_storage_2d, uniform_buffer,
      },
    },
    renderer::{RenderContext, RenderDevice, RenderQueue},
    texture::GpuImage,
    view::ViewTarget,
  },
};

use std::borrow::Cow;

use super::upload::GpuBrickMap;
use crate::lighting::{LightPoolUniform, LightingTheme, build_light_pool};

#[derive(Resource)]
pub(crate) struct DdaBg0BindGroup(pub(crate) BindGroup);
#[derive(Resource)]
pub(crate) struct DdaBg1BindGroup(pub(crate) BindGroup);
#[derive(Resource)]
pub(crate) struct DdaBg2BindGroup(pub(crate) BindGroup);
#[derive(Resource)]
pub(crate) struct DdaBg3BindGroup(pub(crate) BindGroup);
#[derive(Resource)]
struct DdaBlitBindGroup(BindGroup);

/// BG3 光池持久 GPU buffer（主题静态：prepare 覆写同 buffer，避免逐帧重分配）
#[derive(Resource)]
struct LightPoolGpu(UniformBuffer<LightPoolUniform>);

/// beam depth texture 缓存：低分辨率 r32float，resize 时重建
#[derive(Resource, Default)]
struct BeamDepthCache {
  texture: Option<Texture>,
  size: UVec2,
}

#[derive(Resource)]
#[allow(dead_code)]
pub(crate) struct DdaPipelines {
  pub(crate) bg0_layout: BindGroupLayoutDescriptor,
  pub(crate) bg1_layout: BindGroupLayoutDescriptor,
  pub(crate) bg2_layout: BindGroupLayoutDescriptor,
  pub(crate) bg3_layout: BindGroupLayoutDescriptor,
  blit_layout: BindGroupLayoutDescriptor,
  /// BG7：眼睛适应（out_tex 采样视图 + 状态/直方图 storage）
  eye_layout: BindGroupLayoutDescriptor,
  pub(crate) compute_pipeline: CachedComputePipelineId,
  pub(crate) beam_pipeline: CachedComputePipelineId,
  pub(crate) probe_viz_pipeline: CachedComputePipelineId,
  /// eye_adapt_histogram / eye_adapt_update（各 1 个 WG）
  eye_histogram_pipeline: CachedComputePipelineId,
  eye_update_pipeline: CachedComputePipelineId,
  blit_pipeline: CachedRenderPipelineId,
}

/// 眼睛适应的 GPU 状态（`EYE_WORDS` 字 storage buffer）+ 上一帧时间戳（算 dt）。
/// word 布局见 bindings.wesl 的 `eye_adapt`。
#[derive(bevy::ecs::resource::Resource)]
pub struct EyeAdaptGpu {
  pub buf: Option<Buffer>,
  /// 渲染侧墙钟（与 profiler 同源）：适应速度因此与帧率无关
  pub last: Option<std::time::Instant>,
  pub bg: Option<BindGroup>,
  /// 活参数（UI 可调）。挂在 GPU 资源上而不是单独一个 system param ——
  /// `prepare_dda_bind_groups` 已经是 Bevy 的 16 参数上限，再加一个会失去 SystemParamFunction。
  pub settings: EyeAdaptSettings,
  /// 参数区需要重传（`sync_eye_adapt_settings` 置位，prepare 消费；初值 true 保证首帧上传缺省）
  pub settings_dirty: bool,
}

impl Default for EyeAdaptGpu {
  fn default() -> Self {
    Self {
      buf: None,
      last: None,
      bg: None,
      settings: EyeAdaptSettings::default(),
      settings_dirty: true,
    }
  }
}

/// 把 main world 的活参数搬进 [`EyeAdaptGpu`]（**只做搬运**，真正上传在
/// `prepare_dda_bind_groups` 里做，那里才有 queue 和 buffer 句柄）。
/// `Res::is_changed` 由 `ExtractResourcePlugin` 在同步时标记 ⇒ 只有 UI 真改过才为真。
fn sync_eye_adapt_settings(eye_set: Option<Res<EyeAdaptSettings>>, mut eye: ResMut<EyeAdaptGpu>) {
  let Some(s) = eye_set else {
    return;
  };
  if !s.is_changed() || eye.settings == *s {
    return;
  }
  eye.settings = *s;
  eye.settings_dirty = true;
}

/// 眼睛适应（自动曝光）的**活参数**：由 debug overlay 的「Eye」页实时调，改完下一帧生效。
///
/// 传输链路：本资源（main world）→ `ExtractResourcePlugin`（只在变化时才同步进 render world，
/// 并标记 changed）→ `prepare_dda_bind_groups` 检测 `is_changed()` 后写 buffer 参数区 20B
/// → 下一帧 WESL 的 `eye_p(i)` 读到新值。**稳态零写入**。
///
/// 下标顺序即语义，与 WESL 侧 `eye_p(i)` 一一对应（改这里必须同步 `main.wesl`）。
#[derive(Resource, Clone, Copy, Debug, PartialEq, ExtractResource)]
pub struct EyeAdaptSettings {
  /// **总开关**：关掉 = 两个 eye pass 停发 + 曝光回落 1.0（= 关闭自动曝光）。
  /// 不占参数区槽位（host 侧开关，shader 不需要读到它）。初值受 `GATE_NO_EYE_ADAPT=1`
  /// 影响，之后由 debug overlay 的 Eye 页开关接管。
  pub enabled: bool,
  /// [0] 提亮上限（档，≥0）：适应暗处的最大增益 = 2^ev_max
  pub ev_max: f32,
  /// [1] 压暗上限（档，≤0）：适应亮处的最大衰减 = 2^ev_min
  pub ev_min: f32,
  /// [2] 变亮时间常数（秒）：往亮处适应多快（太小像"闪光"）
  pub tau_brighten: f32,
  /// [3] 变暗时间常数（秒）：往暗处适应多快（太小像"眨眼"）
  pub tau_darken: f32,
  /// [4] 目标中灰：百分位平均亮度被压到这个值
  pub key: f32,
}

impl Default for EyeAdaptSettings {
  /// 缺省取**保守**值（EV ±3）。上限越大，暗场里的 GI 残噪被同倍放大得越狠。
  fn default() -> Self {
    Self { enabled: true, ev_max: 3.0, ev_min: -3.0, tau_brighten: 2.0, tau_darken: 1.0, key: 0.18 }
  }
}

impl EyeAdaptSettings {
  /// 缺省 + 环境变量覆盖：`GATE_NO_EYE_ADAPT=1` 只决定**初值**，之后以面板开关为准
  /// （无 UI 的自动化运行也能一行命令切）。
  pub fn from_env() -> Self {
    let off = std::env::var("GATE_NO_EYE_ADAPT").map(|v| v == "1").unwrap_or(false);
    Self { enabled: !off, ..Self::default() }
  }
}

/// `eye_adapt` buffer 里**参数区**的起始字（= 状态/调试区 8 + 直方图 64）
const EYE_PARAM_WORD: u64 = 72;
/// 参数区字节偏移（同 `EYE_PARAM_WORD`）
const EYE_PARAM_OFFSET: u64 = EYE_PARAM_WORD * 4;

/// 参数区打包：5 个 f32 = 20B，顺序 = WESL `eye_p(i)` 的下标 = [`EyeAdaptSettings`] 字段顺序
fn eye_param_bytes(s: EyeAdaptSettings) -> [u8; 20] {
  let mut out = [0u8; 20];
  for (i, v) in [s.ev_max, s.ev_min, s.tau_brighten, s.tau_darken, s.key].iter().enumerate() {
    out[i * 4..i * 4 + 4].copy_from_slice(&v.to_bits().to_le_bytes());
  }
  out
}

pub struct BrickMapDdaPlugin;

impl Plugin for BrickMapDdaPlugin {
  fn build(&self, app: &mut App) {
    // 编译 dda WESL 包（shaders/voxel_raytrace/，入口 main.wesl）→ 插入 `Shader` 资产 + `DdaShaderHandle`。
    crate::shader::build_dda_shader(app);

    app.add_plugins((
      bevy::render::extract_resource::ExtractResourcePlugin::<DdaImages>::default(),
      // RenderScale 提取进 render world（dispatch workgroup 数随 resize 重算）
      bevy::render::extract_resource::ExtractResourcePlugin::<RenderScale>::default(),
      // LightingTheme 提取进 render world（BG3 光池数据源）
      bevy::render::extract_resource::ExtractResourcePlugin::<LightingTheme>::default(),
      // 眼睛适应的活参数（debug overlay 的 Eye 页可调；变化才同步 → 稳态零上传）
      bevy::render::extract_resource::ExtractResourcePlugin::<EyeAdaptSettings>::default(),
      crate::responsive::ResponsivePlugin,
    ));
    app.insert_resource(EyeAdaptSettings::from_env());

    // main → render 的 ExtractSchedule：把 DdaCameraConfig 从 main world 读
    // （main.rs setup 注入的 Resource）→ 转成 DdaViewUniform（render world 资源，
    // 供 PrepareBindGroups 每帧写 uniform buffer）
    let dda_shader = app.world().resource::<crate::shader::DdaShaderHandle>().clone();
    let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
      return;
    };
    render_app.insert_resource(dda_shader);
    render_app
      .add_systems(bevy::render::ExtractSchedule, extract_camera_config)
      .add_systems(RenderStartup, init_dda_pipelines)
      .add_systems(
        Render,
        // 活参数搬运（main world 的 EyeAdaptSettings → EyeAdaptGpu）：必须排在 prepare 之前
        sync_eye_adapt_settings.in_set(RenderSystems::PrepareResources),
      )
      .add_systems(
        Render,
        prepare_dda_bind_groups
          .in_set(RenderSystems::PrepareBindGroups)
          .after(sync_eye_adapt_settings)
          // prepare_dda_bind_groups 在 prepare (upload.rs) 之后运行：先 upload 写
          // grid_descs_buf 再绑 DDA BG2（同帧最稳，避免差一帧的旧 GridDesc 绑定）。
          .after(super::upload::prepare),
      )
      // 必须挂 RenderGraph::Render set（而非 Render schedule）：Render schedule 整体在
      // RenderGraph 之前 → begin_diagnostics_frame（Begin set）前执行，诊断 span 会被清空
      .add_systems(
        RenderGraph,
        dispatch_dda
          .in_set(bevy::render::renderer::RenderGraphSystems::Render)
          .before(camera_driver),
      )
      .add_systems(Core2d, blit_dda_view.in_set(Core2dSystems::PostProcess));
  }
}

fn extract_camera_config(
  mut commands: bevy::ecs::system::Commands,
  cfg: Option<bevy::render::Extract<bevy::ecs::system::Res<crate::brickmap::DdaCameraConfig>>>,
  debug: Option<bevy::render::Extract<bevy::ecs::system::Res<crate::brickmap::DebugNormals>>>,
  scale: Option<bevy::render::Extract<bevy::ecs::system::Res<RenderScale>>>,
) {
  let Some(cfg) = cfg else { return };
  let debug_mode = debug.map(|d| d.0).unwrap_or(0);
  let render_h = scale.map(|s| s.size.y as f32).unwrap_or(VIEW_SIZE.y as f32);
  let uniform = DdaViewUniform::from_cfg(&cfg, debug_mode, render_h);
  commands.insert_resource(uniform);
}

pub(crate) fn init_dda_pipelines(
  mut commands: Commands,
  asset_server: Res<AssetServer>,
  dda_shader: Res<crate::shader::DdaShaderHandle>,
  pipeline_cache: Res<PipelineCache>,
  _render_device: Res<RenderDevice>,
) {
  // ---- BG0：out tex write + DdaViewUniform uniform + beam depth rw ----
  let bg0 = BindGroupLayoutDescriptor::new(
    "DdaBg0",
    &BindGroupLayoutEntries::sequential(
      ShaderStages::COMPUTE,
      (
        texture_storage_2d(TextureFormat::Rgba8Unorm, StorageTextureAccess::WriteOnly),
        uniform_buffer::<DdaViewUniform>(false),
        // @binding(2) beam_depth：低分辨率 r32float，beam pass 写最近命中 t，主 pass 读
        texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::ReadWrite),
        // @binding(3) 眼睛适应的状态/直方图（**只读**视图；`dda_main` 只取曝光系数）。
        // 同一 buffer 在 BG7 以 read_write 被 eye_adapt_* 两个入口读写（不同 pass，屏障保证顺序）。
        storage_buffer_read_only_sized(false, None),
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
        storage_buffer_read_only_sized(false, None), // @binding(1) b_leaves（存放方向可达掩码 LUT）
        storage_buffer_read_only_sized(false, None), // @binding(2) b_palette
        uniform_buffer::<super::wire::BrickMapGlobals>(false), // @binding(3) globals
        // @binding(4)/(5)：光照场（AO fill + 发光密度 ε）——每 16³ 块
        // 实心占比的 3D 纹理 + 线性过滤采样器。
        texture_3d(TextureSampleType::Float { filterable: true }),
        sampler(SamplerBindingType::Filtering),
      ),
    ),
  );

  // ---- BG2：GridDesc 数组（主世界 + 物体同描述符）----
  // shader `trace_grid` 遍历 grid_descs[0..count]，无 kind 分支。
  // GridDesc 144B/entry：pos_scale/rot0/rot1/rot2 + aabb_min/max + tree_base/tree_depth/chunk_count/palette_base + index_origin/dims。
  let bg2 = BindGroupLayoutDescriptor::new(
    "DdaBg2",
    &BindGroupLayoutEntries::sequential(
      ShaderStages::COMPUTE,
      (
        storage_buffer_read_only_sized(false, None), // @binding(0) grid_descs: array<GridDesc>
      ),
    ),
  );

  // ---- BG3：光照光池 uniform（LightPoolUniform，与 WGSL 镜像）----
  let bg3 = BindGroupLayoutDescriptor::new(
    "DdaBg3",
    &BindGroupLayoutEntries::sequential(
      ShaderStages::COMPUTE,
      (uniform_buffer::<LightPoolUniform>(false),),
    ),
  );

  // ---- blit BG layout：storage texture（filterable 上采样采样）+ linear sampler ----
  let blit = BindGroupLayoutDescriptor::new(
    "DdaBlit",
    &BindGroupLayoutEntries::sequential(
      ShaderStages::FRAGMENT,
      (
        texture_2d(TextureSampleType::Float { filterable: true }),
        sampler(SamplerBindingType::Filtering),
      ),
    ),
  );

  // ---- BG7：眼睛适应（out_tex 采样视图 + 状态/直方图 storage 读写）----
  let eye = BindGroupLayoutDescriptor::new(
    "DdaBgEye",
    &BindGroupLayoutEntries::sequential(
      ShaderStages::COMPUTE,
      (
        texture_2d(TextureSampleType::Float { filterable: true }),
        storage_buffer_sized(false, None),
      ),
    ),
  );

  // ---- Compute pipeline：shaders/voxel_raytrace/ 两个入口（dda_main 主 trace+unlit 直出 / beam_main beam 预 pass）----
  let dda_shader = dda_shader.0.clone();
  let layouts =
    vec![bg0.clone(), bg1.clone(), bg2.clone(), bg3.clone(), crate::ddgi::ddgi_bg4_layout()];
  // 眼睛适应的两个入口自己的布局：**8 份相同的 eye layout**。
  // 原因：wgpu 要求 bind group 按索引**从 0 开始成前缀地**设置（跳过低索引直接设高索引会报
  // "expects a BindGroup to be set at index 0"）；而这两个入口的绑定在 @group(7)。
  // 于是把同一个 eye BG 依次设到 0~7 —— 每个索引的 layout 必须一致，故这里重复 8 份。
  let eye_layouts = vec![eye.clone(); 8];
  let compute = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
    label: Some(Cow::from("gate_dda_compute")),
    layout: layouts.clone(),
    shader: dda_shader.clone(),
    entry_point: Some(Cow::from("dda_main")),
    ..default()
  });
  // beam 预 pass：低分辨率输出最近命中 t，主 pass 取邻域 min t 跳过空空间
  let beam = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
    label: Some(Cow::from("gate_beam")),
    layout: layouts.clone(),
    shader: dda_shader.clone(),
    entry_point: Some(Cow::from("beam_main")),
    ..default()
  });
  // 眼睛适应（自动曝光）：直方图统计（1 个 WG）+ 适应更新（1 个线程）。
  // 结构对齐 UE EyeAdaptation：1/16 抽样 → 64 桶 log2 亮度直方图 → 5%~95% 百分位均值 →
  // 反馈 + 分方向时间平滑 → 曝光系数（下一帧 dda_main 读 BG0 binding(3)）。
  let eye_histogram = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
    label: Some(Cow::from("gate_eye_histogram")),
    layout: eye_layouts.clone(),
    shader: dda_shader.clone(),
    entry_point: Some(Cow::from("eye_adapt_histogram")),
    ..default()
  });
  let eye_update = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
    label: Some(Cow::from("gate_eye_update")),
    layout: eye_layouts.clone(),
    shader: dda_shader.clone(),
    entry_point: Some(Cow::from("eye_adapt_update")),
    ..default()
  });
  let probe_viz = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
    label: Some(Cow::from("gate_probe_viz")),
    layout: layouts,
    shader: dda_shader,
    entry_point: Some(Cow::from("probe_viz_main")),
    ..default()
  });

  // ---- Blit render pipeline：blit.wgsl（全屏三角）----
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
    bg3_layout: bg3,
    blit_layout: blit,
    eye_layout: eye,
    compute_pipeline: compute,
    beam_pipeline: beam,
    probe_viz_pipeline: probe_viz,
    eye_histogram_pipeline: eye_histogram,
    eye_update_pipeline: eye_update,
    blit_pipeline,
  });
  commands.insert_resource(LightPoolGpu(UniformBuffer::default()));
  commands.insert_resource(BeamDepthCache::default());
  commands.insert_resource(EyeAdaptGpu::default());
}

#[allow(clippy::too_many_arguments)]
fn prepare_dda_bind_groups(
  mut commands: Commands,
  pipelines: Res<DdaPipelines>,
  gpu_images: Res<RenderAssets<GpuImage>>,
  mut eye: ResMut<EyeAdaptGpu>,
  images: Option<Res<DdaImages>>,
  view_uniform: Option<Res<DdaViewUniform>>,
  gpu_brickmap: Option<Res<GpuBrickMap>>,
  ddgi_gpu: Option<Res<crate::ddgi::DdgiGpu>>,
  dbg: Option<Res<crate::ddgi::DdgiDebugSettings>>,
  lighting: Option<Res<LightingTheme>>,
  light_gpu: Option<ResMut<LightPoolGpu>>,
  render_device: Res<RenderDevice>,
  pipeline_cache: Res<PipelineCache>,
  queue: Res<RenderQueue>,
  scale: Res<RenderScale>,
  mut beam_cache: ResMut<BeamDepthCache>,
) {
  let Some(images) = images else {
    bevy::log::info_once!("DDA prepare: no DdaImages");
    return;
  };
  let Some(view_uniform) = view_uniform else {
    bevy::log::info_once!("DDA prepare: no DdaViewUniform");
    return;
  };
  let Some(gpu) = gpu_brickmap else {
    bevy::log::info_once!("DDA prepare: no GpuBrickMap");
    return;
  };
  let Some(tex_view) = gpu_images.get(&images.target) else {
    bevy::log::info_once!("DDA prepare: GpuImage not ready");
    return;
  };

  let mut view = *view_uniform; // Copy：解引用取出，便于覆写 probe_viz_params
  // probe 可视化参数：x = 世界空间探针总槽数（4 LOD 世界网格）；y = 方块边长 3px；
  // z = 层级选择（0=全部，1..=4=LOD0..3）；w 不再使用
  if let Some(g) = ddgi_gpu.as_ref() {
    let total = g.total_slots;
    let sel = dbg.map_or(0.0, |d| d.probe_viz_lod).clamp(0.0, 4.0);
    view.probe_viz_params = Vec4::new(total as f32, 3.0, sel, 0.0);
  }
  let mut u = UniformBuffer::from(view);
  u.write_buffer(&render_device, &queue);

  let bg0_layout = pipeline_cache.get_bind_group_layout(&pipelines.bg0_layout);
  let eye_layout = pipeline_cache.get_bind_group_layout(&pipelines.eye_layout);
  let bg1_layout = pipeline_cache.get_bind_group_layout(&pipelines.bg1_layout);
  let bg2_layout = pipeline_cache.get_bind_group_layout(&pipelines.bg2_layout);
  let bg3_layout = pipeline_cache.get_bind_group_layout(&pipelines.bg3_layout);
  let blit_layout = pipeline_cache.get_bind_group_layout(&pipelines.blit_layout);

  // ---- beam depth：低分辨率 r32float（全分辨率 / 4），resize 时重建 ----
  const BEAM_DIV: u32 = 4;
  let beam_size = UVec2::new(scale.size.x.div_ceil(BEAM_DIV), scale.size.y.div_ceil(BEAM_DIV));
  if beam_cache.texture.is_none() || beam_cache.size != beam_size {
    let tex = render_device.create_texture(&TextureDescriptor {
      label: Some("gate_beam_depth"),
      size: Extent3d { width: beam_size.x, height: beam_size.y, depth_or_array_layers: 1 },
      mip_level_count: 1,
      sample_count: 1,
      dimension: TextureDimension::D2,
      format: TextureFormat::R32Float,
      usage: TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC,
      view_formats: &[],
    });
    beam_cache.texture = Some(tex);
    beam_cache.size = beam_size;
  }
  let beam_tex = beam_cache.texture.as_ref().expect("beam texture not created");
  let beam_view = beam_tex.create_view(&TextureViewDescriptor::default());

  // ---- BG0：out tex write + view uniform + beam depth rw ----
  // ---- 眼睛适应的状态/直方图 buffer（word 布局见 bindings.wesl 的 `eye_adapt`）----
  // 同一 buffer 两处绑定：BG0 binding(3) 只读（`dda_main` 取曝光）+ BG7 binding(1) 读写
  // （`eye_adapt_*` 写状态/累加直方图）。首帧把曝光初始化为 1.0（否则第一帧全黑）。
  const EYE_WORDS: u64 = 80;
  if eye.buf.is_none() {
    let b = render_device.create_buffer(&BufferDescriptor {
      label: Some("dda_eye_adapt"),
      size: EYE_WORDS * 4,
      usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
      mapped_at_creation: false,
    });
    let mut init = [0u8; (EYE_WORDS * 4) as usize];
    init[..4].copy_from_slice(&1.0f32.to_bits().to_le_bytes());
    queue.write_buffer(&b, 0, &init);
    // 参数区初值（首帧 settings_dirty 也会再写一次，这里是保险）
    queue.write_buffer(&b, EYE_PARAM_OFFSET, &eye_param_bytes(eye.settings));
    eye.buf = Some(b);
  }
  // clone 一份句柄（Buffer 内部是 Arc）：后面还要改 eye.last/eye.bg，避免借用冲突
  let eye_buf = eye.buf.clone().expect("刚插入");
  // dt 只在开启眼睛适应时才上传（关闭时这条路径**完全不碰**任何每帧写入 ⇒ 零开销）。
  // 注意：CPU 写这张 buffer，GPU 的 eye_adapt_* 也写同一张（buffer 粒度的写-写冲突）。
  let now = std::time::Instant::now();
  if eye.settings.enabled {
    let dt = eye.last.map_or(1.0 / 60.0, |t| now.duration_since(t).as_secs_f32());
    queue.write_buffer(&eye_buf, 12, &dt.clamp(0.0, 0.25).to_bits().to_le_bytes());
  }
  eye.last = Some(now);
  // 活参数：只在设置变化时上传 20B（稳态零写入，也不去每帧碰这张 GPU 也在写的 buffer）
  if std::mem::take(&mut eye.settings_dirty) {
    let s = eye.settings;
    queue.write_buffer(&eye_buf, EYE_PARAM_OFFSET, &eye_param_bytes(s));
    // 关掉总开关的瞬间把曝光回落到 1.0：两个 eye pass 同时停发 ⇒ 之后没人再改 word[0]，
    // 否则画面会"冻结"在关掉那一刻的曝光上（看着像渲染卡住了）。
    if !s.enabled {
      queue.write_buffer(&eye_buf, 0, &1.0f32.to_bits().to_le_bytes());
    }
    // 每次真正推送都记一行，便于确认参数已到 GPU
    bevy::log::info!(
      target: "gate",
      "eye adapt 参数 → GPU：{} EV+ {:.2} / EV- {:.2} / tau+ {:.2}s / tau- {:.2}s / key {:.3}",
      if s.enabled { "on" } else { "off" },
      s.ev_max,
      s.ev_min,
      s.tau_brighten,
      s.tau_darken,
      s.key,
    );
  }

  let bg0 = render_device.create_bind_group(
    None,
    &bg0_layout,
    &BindGroupEntries::sequential((
      &tex_view.texture_view,
      &u,
      &beam_view,
      eye_buf.as_entire_binding(),
    )),
  );
  // BG7：眼睛适应（out_tex 采样视图 + 状态/直方图读写）
  let eye_bg = render_device.create_bind_group(
    None,
    &eye_layout,
    &BindGroupEntries::sequential((&tex_view.texture_view, eye_buf.as_entire_binding())),
  );
  eye.bg = Some(eye_bg);

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
      &gpu.light_view,
      &gpu.light_sampler,
    )),
  );

  // ---- BG2：GridDesc 数组（主世界 + 物体统一描述符）----
  // shader `dda_main` 遍历 grid_descs[0..arrayLength]，trace_grid 无 kind 分支。
  let bg2 = render_device.create_bind_group(
    None,
    &bg2_layout,
    &BindGroupEntries::sequential((gpu.grid_descs_buf.as_entire_binding(),)),
  );

  // ---- BG3：光照光池（主题静态，覆写同一持久 buffer）----
  let Some(lighting) = lighting else {
    bevy::log::info_once!("DDA prepare: no LightingTheme");
    return;
  };
  let Some(mut lp) = light_gpu else {
    bevy::log::info_once!("DDA prepare: no LightPoolGpu");
    return;
  };
  *lp.0.get_mut() = build_light_pool(&lighting);
  lp.0.write_buffer(&render_device, &queue);
  let bg3 = render_device.create_bind_group(None, &bg3_layout, &BindGroupEntries::single(&lp.0));

  // ---- Blit BG：dda tex（filterable）+ linear sampler（半分辨率上采样）----
  let blit_sampler = render_device.create_sampler(&SamplerDescriptor::default());
  let blit_bg = render_device.create_bind_group(
    None,
    &blit_layout,
    &BindGroupEntries::sequential((&tex_view.texture_view, &blit_sampler)),
  );

  commands.insert_resource(DdaBg0BindGroup(bg0));
  commands.insert_resource(DdaBg1BindGroup(bg1));
  commands.insert_resource(DdaBg2BindGroup(bg2));
  commands.insert_resource(DdaBg3BindGroup(bg3));
  commands.insert_resource(DdaBlitBindGroup(blit_bg));
}

#[allow(clippy::too_many_arguments)] // Bevy render system：各 bind group + 资源逐一注入
pub(crate) fn dispatch_dda(
  mut ctx: RenderContext,
  bg0: Option<Res<DdaBg0BindGroup>>,
  bg1: Option<Res<DdaBg1BindGroup>>,
  bg2: Option<Res<DdaBg2BindGroup>>,
  bg3: Option<Res<DdaBg3BindGroup>>,
  bg4: Option<Res<crate::ddgi::DdgiBg4>>,
  eye: Option<Res<EyeAdaptGpu>>,
  gpu: Option<Res<crate::ddgi::DdgiGpu>>,
  dbg: Option<Res<crate::ddgi::DdgiDebugSettings>>,
  pipeline_cache: Res<PipelineCache>,
  pipelines: Res<DdaPipelines>,
  scale: Res<RenderScale>,
  mut profiler: ResMut<crate::profiler::GpuProfilerRes>,
) {
  // 主 pass trace 命中后直接 unlit 着色直出 out_tex
  // （逐体素法线 + 天空渐变 + 太阳方向光项），无后续 direct/gi/denoise pass。
  let (Some(bg0), Some(bg1), Some(bg2), Some(bg3), Some(bg4)) =
    (bg0.as_ref(), bg1.as_ref(), bg2.as_ref(), bg3.as_ref(), bg4.as_ref())
  else {
    bevy::log::debug_once!("DDA dispatch: bind groups missing");
    return;
  };

  let dda_pipe = pipeline_cache.get_compute_pipeline(pipelines.compute_pipeline).or_else(|| {
    bevy::log::debug_once!("DDA dispatch: dda pipeline not ready");
    None
  });
  let beam_pipe = pipeline_cache.get_compute_pipeline(pipelines.beam_pipeline);

  let gx = scale.size.x.div_ceil(DDA_WORKGROUP_SIZE);
  let gy = scale.size.y.div_ceil(DDA_WORKGROUP_SIZE);
  // beam：低分辨率 dispatch = ceil(size / 4) / 8
  let bx = scale.size.x.div_ceil(4).div_ceil(WORKGROUP_SIZE);
  let by = scale.size.y.div_ceil(4).div_ceil(WORKGROUP_SIZE);

  // ---- beam 预 pass：低分辨率 trace 只输出最近命中 t（独立 compute pass，
  // beam 写 beam_depth，主 pass 读同 texture → pass 边界 barrier 保证可见性）----
  // GATE_NO_BEAM=1：跳过 beam pass，主 pass t_min=0（穿墙定位用）
  if !*BEAM_DISABLED && let Some(beam_pipe) = beam_pipe {
    crate::profiler::gpu_compute_pass(&mut profiler, ctx.command_encoder(), "gate_beam", |pass| {
      pass.set_pipeline(beam_pipe);
      pass.set_bind_group(0, &bg0.0, &[]);
      pass.set_bind_group(1, &bg1.0, &[]);
      pass.set_bind_group(2, &bg2.0, &[]);
      pass.set_bind_group(3, &bg3.0, &[]);
      pass.set_bind_group(4, &bg4.0, &[]);
      pass.dispatch_workgroups(bx, by, 1);
    });
  }

  // ---- 主 DDA pass：trace + unlit 着色直出 ----
  if let Some(dda_pipe) = dda_pipe {
    crate::profiler::gpu_compute_pass(
      &mut profiler,
      ctx.command_encoder(),
      "gate_dda_trace",
      |pass| {
        pass.set_pipeline(dda_pipe);
        pass.set_bind_group(0, &bg0.0, &[]);
        pass.set_bind_group(1, &bg1.0, &[]);
        pass.set_bind_group(2, &bg2.0, &[]);
        pass.set_bind_group(3, &bg3.0, &[]);
        pass.set_bind_group(4, &bg4.0, &[]);
        pass.dispatch_workgroups(gx, gy, 1);
      },
    );
  }

  // ---- 眼睛适应：直方图统计（1 个 WG）+ 适应更新（1 个线程）----
  // 必须排在主 pass 之后（统计"本帧已曝光"的画面：反馈环把百分位平均亮度压到目标中灰）；
  // 曝光系数由下一帧的 `dda_main` 通过 BG0 binding(3) 读到。
  if eye.as_ref().is_some_and(|e| e.settings.enabled)
    && let Some(eye_bg) = eye.as_ref().and_then(|e| e.bg.as_ref())
    && let Some(h) = pipeline_cache.get_compute_pipeline(pipelines.eye_histogram_pipeline)
    && let Some(u) = pipeline_cache.get_compute_pipeline(pipelines.eye_update_pipeline)
  {
    crate::profiler::gpu_compute_pass(
      &mut profiler,
      ctx.command_encoder(),
      "gate_eye_histogram",
      |pass| {
        pass.set_pipeline(h);
        // eye pipeline 的布局是 8 份相同 layout ⇒ 必须从 0 起逐个设（见 init_dda_pipelines）
        for i in 0..8u32 {
          pass.set_bind_group(i, eye_bg, &[]);
        }
        pass.dispatch_workgroups(1, 1, 1);
      },
    );
    crate::profiler::gpu_compute_pass(
      &mut profiler,
      ctx.command_encoder(),
      "gate_eye_update",
      |pass| {
        pass.set_pipeline(u);
        for i in 0..8u32 {
          pass.set_bind_group(i, eye_bg, &[]);
        }
        pass.dispatch_workgroups(1, 1, 1);
      },
    );
  }

  if dbg.is_some_and(|d| d.probe_viz)
    && let Some(ddgi) = gpu.as_ref()
    && let Some(pipe) = pipeline_cache.get_compute_pipeline(pipelines.probe_viz_pipeline)
  {
    let probe_count = ddgi.total_slots;
    crate::profiler::gpu_compute_pass(
      &mut profiler,
      ctx.command_encoder(),
      "gate_probe_viz",
      |pass| {
        pass.set_pipeline(pipe);
        pass.set_bind_group(0, &bg0.0, &[]);
        pass.set_bind_group(1, &bg1.0, &[]);
        pass.set_bind_group(2, &bg2.0, &[]);
        pass.set_bind_group(3, &bg3.0, &[]);
        pass.set_bind_group(4, &bg4.0, &[]);
        let wg = probe_count.div_ceil(64);
        pass.dispatch_workgroups(wg, 1, 1);
      },
    );
  }
}

#[cfg_attr(not(feature = "profile"), allow(unused_variables, unused_mut))]
fn blit_dda_view(
  mut ctx: RenderContext,
  views: Query<&ViewTarget>,
  blit_bg: Option<Res<DdaBlitBindGroup>>,
  pipeline_cache: Res<PipelineCache>,
  pipelines: Res<DdaPipelines>,
  mut profiler: ResMut<crate::profiler::GpuProfilerRes>,
) {
  let (Some(bg), Ok(target)) = (blit_bg.as_ref(), views.single()) else {
    bevy::log::debug_once!("DDA blit: bg or ViewTarget missing");
    return;
  };
  let Some(pipe) = pipeline_cache.get_render_pipeline(pipelines.blit_pipeline) else {
    bevy::log::debug_once!("DDA blit: blit pipeline not ready");
    return;
  };
  // profile 构建：scoped_render_pass（pass 时间戳 → wgpu-profiler → Tracy）；
  // profiler 未就绪/非 profile 构建：常规 begin_render_pass。
  #[cfg(feature = "profile")]
  if let Some(profiler) = crate::profiler::profiler_mut(&mut profiler) {
    let mut encoder_scope = profiler.scope("gate_dda_blit", ctx.command_encoder());
    let mut pass = encoder_scope.scoped_render_pass(
      "gate_dda_blit",
      RenderPassDescriptor {
        label: Some("gate_dda_blit"),
        color_attachments: &[Some(target.get_color_attachment())],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        ..default()
      },
    );
    pass.set_pipeline(pipe);
    pass.set_bind_group(0, &bg.0, &[]);
    pass.draw(0..3, 0..1);
    return;
  }
  let mut pass = ctx
    .command_encoder()
    .begin_render_pass(&RenderPassDescriptor {
      label: Some("gate_dda_blit"),
      color_attachments: &[Some(target.get_color_attachment())],
      depth_stencil_attachment: None,
      timestamp_writes: None,
      occlusion_query_set: None,
      ..default()
    })
    .forget_lifetime();
  pass.set_pipeline(pipe);
  pass.set_bind_group(0, &bg.0, &[]);
  pass.draw(0..3, 0..1);
}
