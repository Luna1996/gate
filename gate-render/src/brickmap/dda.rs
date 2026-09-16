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

// ============================================================================
// WGSL 常量 Rust 镜像（改 WGSL 顶部 const 时必须同步，无自动化校验）
// ============================================================================

/// WGSL 着色器顶部 `const` 的 Rust 镜像副本（改 WGSL 时必须一起改）
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
  /// 调色板字数（2^16 条 × 2w = 512KB/volume）；与 `wire.rs::PALETTE_WORDS` 同源
  pub const PALETTE_WORDS: u32 = crate::brickmap::wire::PALETTE_WORDS as u32;
  /// 叶父层 inline 字数与每字体素数（与 `wire.rs` 同源，供 WGSL 顶部 const 对齐）
  pub const LEAF_INLINE_WORDS: u32 = crate::brickmap::wire::LEAF_INLINE_WORDS as u32;
  pub const LEAF_VOXELS_PER_WORD: u32 = crate::brickmap::wire::LEAF_VOXELS_PER_WORD as u32;
  pub const CHUNK_COMP_WORDS: u32 = 2048; // u16[4096] → 每 2 字打包 u32
  pub const STATE_ENTRY_COUNT: u32 = 256;
  pub const STATE_WORDS_PER_ENTRY: u32 = 4;
  pub const STATE_TOTAL_WORDS: u32 = 1024; // 256 × 4
  // 直光层（镜像 lighting.rs 常量；WGSL const 同步，改时一起改）
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
  /// 必须与 WGSL `SHADOW_SURFACE_EPS` 一致。
  pub const SHADOW_SURFACE_EPS: f32 = 0.03125;
}

/// 与 WGSL `popcount(mask & (bit - 1u64))` 等价：mask bit=1 子块在 child offset
/// 表中的槽位（紧凑 child offset 只存 bit=1 的子块）
#[inline]
pub fn wgsl_child_slot_index(mask: u64, child_idx: u32) -> u32 {
  debug_assert!(child_idx < 64);
  (mask & ((1u64 << child_idx) - 1)).count_ones()
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
) -> Option<(f32, u16)> {
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
/// 与 cpu_reference_dda_ray 逐射线命中等价（AABB-skip 与 brute-force 两条路径同一套标尺）。
pub fn cpu_reference_dda_ray_aabb_skip(
  buffers: &BrickMapBuffers,
  origin: Vec3,
  dir: Vec3,         // 归一化（AABB 求交不要求归一化，但这里统一用归一与原函数对齐）
  t_global_max: f32, // 自 origin 量起的全局上限（= frustum_length）
  max_steps: u32,
  aabb_min: Vec3,
  aabb_max: Vec3,
) -> Option<(f32, u16)> {
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
) -> Option<(f32, u16, u8)> {
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
  pub pal: u16,
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
/// ~0.1 voxel，只影响返回 t 值，不影响命中胞序。
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
  pub pal: u16,
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
  pal: u16,    // 节点 palette（统一子块颜色，0=空气）
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
) -> Option<(f32, u16, u8, [i32; 3])> {
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
    pal: (b_struct[addr + 2] & 0xFFFF) as u16,
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
          let w = b_struct[b.addr + 3 + (idx >> 1)];
          let leaf_pal = ((w >> ((idx & 1) * 16)) & 0xFFFF) as u16;
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
          let w = b_struct[b.addr + 3 + (idx >> 1)];
          let dp = ((w >> ((idx & 1) * 16)) & 0xFFFF) as u16;
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
  pub pal: u16,
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
) -> Option<(f32, u16, Vec3)> {
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
      BindGroup, BindGroupEntries, BindGroupEntry, BindGroupLayoutDescriptor,
      BindGroupLayoutEntries, BindingResource, CachedComputePipelineId, CachedRenderPipelineId,
      ColorTargetState, ColorWrites, ComputePipelineDescriptor, Extent3d, FragmentState,
      PipelineCache, RenderPassDescriptor, SamplerBindingType, ShaderStages, StorageTextureAccess,
      TextureDescriptor, TextureDimension, TextureFormat, TextureSampleType, TextureUsages,
      TextureViewDescriptor, UniformBuffer, VertexState,
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
pub(crate) struct LightPoolGpu(UniformBuffer<LightPoolUniform>);

/// 辅助纹理缓存（屏幕尺寸相关，resize 时重建）：
///   · `texture`：beam depth（低分辨率 r32float）——beam 预 pass 写、主 pass 读；
///   · `gi_*`：半分辨率 GI 缓冲（菜单开关 `DdgiDebugSettings.gi_half_res`）——`gi_main` 写
///     （写入侧在 BG5），主 pass 双线性采样（BG0 binding 4/5）。存的是 **premultiplied valid**：
///     rgba16f 的 .rgb = gi·valid、.a = valid；rg32f 的 .r = cov·valid、.g = valid ⇒ 采样侧
///     按 valid 归一化，天空/介质像素不污染几何边缘。
///   · `gi_bg0`：GI pass 自己的 @group(0)（view uniform + beam depth，**不含** GI 采样视图 ——
///     同一 pass 内不能把同一张纹理既绑成采样又绑成存储，故 GI pass 用这份"瘦"版 BG0）。
#[derive(Resource, Default)]
pub(crate) struct AuxTexCache {
  texture: Option<Texture>,
  size: UVec2,
  gi_tex: Option<Texture>,
  gi_cov: Option<Texture>,
  gi_view: Option<TextureView>,
  gi_cov_view: Option<TextureView>,
  gi_size: UVec2,
  gi_bg0: Option<BindGroup>,
  /// group(5) 的 GI **采样侧** bind group（`dda_main` 用；layout = `DdaPipelines::gi_read_layout`）
  gi_read_bg: Option<BindGroup>,
}

impl AuxTexCache {
  /// 半分辨率 GI 的**写入侧**视图（BG5 的 binding 2/3 用）。
  /// `None` = 尚未创建（首帧，或本帧 `prepare_dda_bind_groups` 提前返回）⇒ 调用方须用占位纹理。
  pub(crate) fn gi_write_views(&self) -> Option<(&TextureView, &TextureView)> {
    Some((self.gi_view.as_ref()?, self.gi_cov_view.as_ref()?))
  }
}

#[derive(Resource)]
#[allow(dead_code)]
pub(crate) struct DdaPipelines {
  pub(crate) bg0_layout: BindGroupLayoutDescriptor,
  /// BG0 的"瘦"版：`view uniform + beam depth`（供 `gi_main` 用 —— 该 pass 要**写** GI 纹理，
  /// 故不能复用含 GI 采样视图的 `bg0_layout`）
  pub(crate) bg0_gi_layout: BindGroupLayoutDescriptor,
  /// group(5) 的"GI 采样侧"（`dda_main` 专用）：两张半分辨率 GI 纹理（绑定号 4/5）
  pub(crate) gi_read_layout: BindGroupLayoutDescriptor,
  pub(crate) bg1_layout: BindGroupLayoutDescriptor,
  pub(crate) bg2_layout: BindGroupLayoutDescriptor,
  pub(crate) bg3_layout: BindGroupLayoutDescriptor,
  blit_layout: BindGroupLayoutDescriptor,
  /// BG7：眼睛适应（out_tex 采样视图 + 状态/直方图 storage）
  eye_layout: BindGroupLayoutDescriptor,
  pub(crate) compute_pipeline: CachedComputePipelineId,
  pub(crate) beam_pipeline: CachedComputePipelineId,
  /// 半分辨率 GI（菜单开关）：`gi_main`
  pub(crate) gi_pipeline: CachedComputePipelineId,
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

  // ---- BG5（GI 采样侧，**只**给 `dda_main` 的 pipeline 用）：半分辨率 GI 的两张纹理 ----
  // 为什么单开一份、而且绑定号是 4/5：DDGI 各 pass 也绑 BG0，而 wgpu 把 bind group 里**所有**
  // 条目的资源都算进该 pass 的 usage scope ⇒ 采样视图若挂在 BG0，collect（BG5 写同一张纹理）
  // 就会在同一 pass 内撞 usage 冲突。放在 group(5) 的空闲绑定号（0..3 已被图集/GI 写入侧占）
  // 且只进 `dda_main` 的 layout，两个 pass 各自只见到一种用法。
  let gi_read = BindGroupLayoutDescriptor::new(
    "DdaBg5GiRead",
    &[
      BindGroupLayoutEntry {
        binding: 4,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Texture {
          sample_type: TextureSampleType::Float { filterable: true },
          view_dimension: TextureViewDimension::D2,
          multisampled: false,
        },
        count: None,
      },
      BindGroupLayoutEntry {
        binding: 5,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Texture {
          sample_type: TextureSampleType::Float { filterable: true },
          view_dimension: TextureViewDimension::D2,
          multisampled: false,
        },
        count: None,
      },
    ],
  );

  // ---- BG0（GI pass 专用瘦版）：view uniform + beam depth ----
  // 绑定号与完整版一致（1/2），只是**不含** out_tex / eye_adapt_ro / GI 采样视图：
  // `gi_main` 只做「反投影 + beam 起点 + 主 trace + ddgi_sample」，不需要 out_tex 与曝光。
  let bg0_gi = BindGroupLayoutDescriptor::new(
    "DdaBg0Gi",
    &[
      BindGroupLayoutEntry {
        binding: 1,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
          ty: BufferBindingType::Uniform,
          has_dynamic_offset: false,
          min_binding_size: Some(DdaViewUniform::min_size()),
        },
        count: None,
      },
      BindGroupLayoutEntry {
        binding: 2,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::StorageTexture {
          access: StorageTextureAccess::ReadWrite,
          format: TextureFormat::R32Float,
          view_dimension: TextureViewDimension::D2,
        },
        count: None,
      },
    ],
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
  // `dda_main` 比其它两个入口多一份 group(5)：半分辨率 GI 的采样侧（见 `gi_read`）。
  // 只加给它 —— beam / probe_viz 用不到，多一份 layout 会让它们也必须绑 group(5)。
  let dda_layouts = {
    let mut v = layouts.clone();
    v.push(gi_read.clone());
    v
  };
  // 眼睛适应的两个入口自己的布局：**8 份相同的 eye layout**。
  // 原因：wgpu 要求 bind group 按索引**从 0 开始成前缀地**设置（跳过低索引直接设高索引会报
  // "expects a BindGroup to be set at index 0"）；而这两个入口的绑定在 @group(7)。
  // 于是把同一个 eye BG 依次设到 0~7 —— 每个索引的 layout 必须一致，故这里重复 8 份。
  let eye_layouts = vec![eye.clone(); 8];
  let compute = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
    label: Some(Cow::from("gate_dda_compute")),
    layout: dda_layouts,
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
  // 半分辨率 GI（菜单开关 `DdgiDebugSettings.gi_half_res`）：只做「反投影 + beam 起点 + 主 trace
  // + ddgi_sample」，写两张 1/2 分辨率缓冲。group0 用瘦版（不含 GI 采样视图），并多一个 BG5
  // （图集写入侧 + GI 写入侧）—— layout 索引必须是 0..=5 的**前缀**（见上面 eye 的说明）。
  let gi_layouts = vec![
    bg0_gi.clone(),
    bg1.clone(),
    bg2.clone(),
    bg3.clone(),
    crate::ddgi::ddgi_bg4_layout(),
    crate::ddgi::ddgi_bg5_layout(),
  ];
  let gi = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
    label: Some(Cow::from("gate_gi")),
    layout: gi_layouts,
    shader: dda_shader.clone(),
    entry_point: Some(Cow::from("gi_main")),
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
    bg0_gi_layout: bg0_gi,
    gi_read_layout: gi_read,
    bg1_layout: bg1,
    bg2_layout: bg2,
    bg3_layout: bg3,
    blit_layout: blit,
    eye_layout: eye,
    compute_pipeline: compute,
    beam_pipeline: beam,
    gi_pipeline: gi,
    probe_viz_pipeline: probe_viz,
    eye_histogram_pipeline: eye_histogram,
    eye_update_pipeline: eye_update,
    blit_pipeline,
  });
  commands.insert_resource(LightPoolGpu(UniformBuffer::default()));
  commands.insert_resource(AuxTexCache::default());
  commands.insert_resource(EyeAdaptGpu::default());
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare_dda_bind_groups(
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
  mut beam_cache: ResMut<AuxTexCache>,
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

  // ---- 半分辨率 GI 缓冲（菜单开关 `gi_half_res`）：屏幕 1/2 分辨率，resize 时重建 ----
  // 存 premultiplied valid（见 AuxTexCache 的说明）：rgba16f = (gi·valid, valid)、
  // rg32f = (cov·valid, valid)。wgpu 新建纹理自动清零 ⇒ valid 初值 0 = "无数据"，
  // 采样侧据此退回 conf=0 的天光兜底（不会在第一帧把整屏 GI 拉黑或提亮）。
  let gi_size = UVec2::new((scale.size.x / 2).max(1), (scale.size.y / 2).max(1));
  if beam_cache.gi_tex.is_none() || beam_cache.gi_size != gi_size {
    let make = |label: &str, format: TextureFormat| {
      render_device.create_texture(&TextureDescriptor {
        label: Some(label),
        size: Extent3d { width: gi_size.x, height: gi_size.y, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format,
        // 既要被 gi_main 写（storage），又要被 dda_main 采样（texture binding）
        usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
      })
    };
    let gi_tex = make("gate_gi_half", TextureFormat::Rgba16Float);
    let gi_cov = make("gate_gi_cov_half", TextureFormat::Rg32Float);
    beam_cache.gi_view = Some(gi_tex.create_view(&TextureViewDescriptor::default()));
    beam_cache.gi_cov_view = Some(gi_cov.create_view(&TextureViewDescriptor::default()));
    beam_cache.gi_tex = Some(gi_tex);
    beam_cache.gi_cov = Some(gi_cov);
    beam_cache.gi_size = gi_size;
  }
  let gi_view = beam_cache.gi_view.as_ref().expect("gi view not created");
  let gi_cov_view = beam_cache.gi_cov_view.as_ref().expect("gi cov view not created");

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
  // group(5) 的 GI 采样侧（只进 `dda_main` 的 pipeline layout）：绑定号 4/5，给显式 entry 数组
  // —— `BindGroupEntries::sequential` 是按位置 = 绑定号，无法表达"从 4 开始"。
  let gi_read_layout = pipeline_cache.get_bind_group_layout(&pipelines.gi_read_layout);
  let gi_read_bg = render_device.create_bind_group(
    None,
    &gi_read_layout,
    &[
      BindGroupEntry { binding: 4, resource: BindingResource::TextureView(gi_view) },
      BindGroupEntry { binding: 5, resource: BindingResource::TextureView(gi_cov_view) },
    ],
  );
  beam_cache.gi_read_bg = Some(gi_read_bg);
  // GI pass 的 @group(0)（瘦版 layout）：绑定号是 1/2（与完整版对齐），故给显式 entry 数组
  // —— `BindGroupEntries::sequential` 是按位置 = 绑定号，无法表达"从 1 开始"。
  // 不绑 GI 采样视图是硬性要求：同一个 pass 里同一张纹理不能既作采样又作存储。
  let bg0_gi_layout = pipeline_cache.get_bind_group_layout(&pipelines.bg0_gi_layout);
  let gi_bg0 = render_device.create_bind_group(
    None,
    &bg0_gi_layout,
    &[
      BindGroupEntry { binding: 1, resource: u.binding().expect("view uniform 已写入") },
      BindGroupEntry { binding: 2, resource: BindingResource::TextureView(&beam_view) },
    ],
  );
  beam_cache.gi_bg0 = Some(gi_bg0);
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
  bg5: Option<Res<crate::ddgi::DdgiBg5>>,
  eye: Option<Res<EyeAdaptGpu>>,
  gpu: Option<Res<crate::ddgi::DdgiGpu>>,
  dbg: Option<Res<crate::ddgi::DdgiDebugSettings>>,
  aux: Option<Res<AuxTexCache>>,
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

  // ---- 半分辨率 GI（菜单开关）：排在主 pass 之前（主 pass 采样它的输出）----
  // 自门控：shader 里 `flags.y < 0.5 || misc.x < 0.5` 直接 return ⇒ 关掉时只付一次空 dispatch。
  // 诊断配色档（mode > 0.5）主 pass 强制走内联采样 ⇒ 这里也不必跑。
  if dbg.as_ref().is_some_and(|d| d.gi_half_res && d.mode < 0.5)
    && let Some(aux) = aux.as_ref()
    && let Some(gi_bg0) = aux.gi_bg0.as_ref()
    && let Some(bg5) = bg5.as_ref()
    && let Some(gi_pipe) = pipeline_cache.get_compute_pipeline(pipelines.gi_pipeline)
  {
    let gx = aux.gi_size.x.div_ceil(DDA_WORKGROUP_SIZE);
    let gy = aux.gi_size.y.div_ceil(DDA_WORKGROUP_SIZE);
    crate::profiler::gpu_compute_pass(&mut profiler, ctx.command_encoder(), "gate_gi", |pass| {
      pass.set_pipeline(gi_pipe);
      pass.set_bind_group(0, gi_bg0, &[]);
      pass.set_bind_group(1, &bg1.0, &[]);
      pass.set_bind_group(2, &bg2.0, &[]);
      pass.set_bind_group(3, &bg3.0, &[]);
      pass.set_bind_group(4, &bg4.0, &[]);
      pass.set_bind_group(5, &bg5.0, &[]);
      pass.dispatch_workgroups(gx, gy, 1);
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
        // 半分辨率 GI 的采样侧（layout 里的 group(5)）：没有它 `dda_main` 无法 dispatch。
        // 纹理未就绪（`prepare_dda_bind_groups` 本帧提前返回）时退化为不绑 —— 此时上面几个
        // bind group 也必然缺失，主 pass 根本不会走到这里。
        if let Some(gi_read) = aux.as_ref().and_then(|a| a.gi_read_bg.as_ref()) {
          pass.set_bind_group(5, gi_read, &[]);
        }
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
