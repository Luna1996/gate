//! DDA 主可见性 pass：WGSL compute + Core2d PostProcess blit。
//! BG0 = storage tex / 相机 uniform / beam depth / 眼睛适应状态（只读）；
//! BG1 = b_struct / b_leaves / palette / globals uniform / 光照场 3D 纹理。
//! WGSL 源 = WESL 包 `shaders/voxel_raytrace/`（入口 `main.wesl`）。

use bevy::{
  asset::RenderAssetUsages,
  image::Image,
  prelude::*,
  render::{extract_resource::ExtractResource, render_resource::*},
};
use std::ops::Mul;
use std::sync::LazyLock;

/// blit.wgsl 资产路径（全屏三角 blit）
pub const BLIT_SHADER_ASSET_PATH: &str = "shaders/blit.wgsl";
/// 初始渲染分辨率（窗口创建尺寸）
pub const VIEW_SIZE: UVec2 = UVec2::new(1280, 720);
/// beam pass 的 compute dispatch 工作组边长
pub const WORKGROUP_SIZE: u32 = 8;
/// 主 DDA pass 工作组边长：必须与 `shaders/voxel_raytrace/` 中 dda_main 的 `@workgroup_size` 一致。
pub const DDA_WORKGROUP_SIZE: u32 = 8;

/// 当前渲染分辨率（main world `resize_render_targets` 更新，提取进 render world）。
#[derive(Resource, Clone, Copy, Debug, PartialEq, ExtractResource)]
pub struct RenderScale {
  /// 渲染目标尺寸 = 窗口物理像素 ÷ `factor`。
  pub size: UVec2,
  /// 分辨率降采样倍数：1 = 全分辨率，2 = 半分辨率。
  pub factor: u32,
}

impl Default for RenderScale {
  fn default() -> Self {
    Self { size: VIEW_SIZE, factor: 1 }
  }
}

/// 后处理开关（main world 由菜单写，提取进 render world）。
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq, ExtractResource)]
pub struct PostFxSettings {
  /// FXAA：在最终 blit 里做边缘抗锯齿（`blit.wgsl::fs_fxaa`）。
  pub fxaa: bool,
}

/// 主 world 注入的静态视图配置（矩阵来自 [`Self::build_static`]）。
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
    let near = 1.0;
    let far = 4000.0;
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
pub const PITCH_LIMIT: f32 = 89.0_f32.to_radians();
pub const DIST_MIN: f32 = 32.0;

/// 轨道相机参数（main world 资源，gate-app 输入 system 操作）。
/// `target` = 注视点（voxel）、`distance` = 相机到 target 距离（voxel）、`yaw` = 绕 +Y 方位角（rad）、
/// `pitch` = 仰角（rad，+ 为上仰）。
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct OrbitCamera {
  pub target: Vec3,
  pub distance: f32,
  pub yaw: f32,
  pub pitch: f32,
}

impl OrbitCamera {
  /// 从眼位和目标点构造轨道参数。
  pub fn from_eye(eye: Vec3, target: Vec3) -> Self {
    let offset = eye - target;
    let distance = offset.length();
    let pitch = offset.y.atan2(offset.xz().length());
    let yaw = offset.x.atan2(offset.z);
    Self { target, distance, yaw, pitch }
  }

  /// 轨道参数重建眼位。
  pub fn eye(&self) -> Vec3 {
    let (sin_yaw, cos_yaw) = self.yaw.sin_cos();
    let (sin_pitch, cos_pitch) = self.pitch.sin_cos();
    self.target + self.distance * Vec3::new(sin_yaw * cos_pitch, sin_pitch, cos_yaw * cos_pitch)
  }

  /// 应用约束（pitch ±89°、distance ≥ DIST_MIN；yaw 无限制）。
  pub fn clamp(&mut self) {
    self.pitch = self.pitch.clamp(-PITCH_LIMIT, PITCH_LIMIT);
    self.distance = self.distance.max(DIST_MIN);
  }
}

impl DdaCameraConfig {
  /// 眼位 + 视线方向构造（幽灵/飞行相机用；轨道相机走 [`Self::from_orbit`]）。
  /// `forward` 必须是朝向场景的单位向量，且与 +Y 不共线。
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

  /// orbit 参数 → `perspective_rh` × `look_at_rh`（fov/aspect/near/far 为显式参数）。
  pub fn from_orbit(orbit: &OrbitCamera, fov_y: f32, aspect: f32, near: f32, far: f32) -> Self {
    let eye = orbit.eye();
    let view = Mat4::look_at_rh(eye, orbit.target, Vec3::Y);
    let proj = Mat4::perspective_rh(fov_y, aspect, near, far);
    let view_proj = proj.mul(view);
    Self { view_proj, inv_view_proj: view_proj.inverse(), position_world: eye }
  }
}

/// Render-world 着色器绑定的 camera uniform，与 WGSL `DdaViewUniform` 逐字对齐。
#[derive(Resource, Clone, Copy, ShaderType)]
pub struct DdaViewUniform {
  pub view_proj: Mat4,
  pub inv_view_proj: Mat4,
  pub cam_pos_voxel: Vec4, // w=1
  /// x/y = debug 可视化；z = 2 跳过 chunk 步进；w = +2 skyout / +4 makegrid_only
  pub debug_mode: Vec4,
  /// x = 单像素角大小(rad) = 2·tan(FOV_Y/2)/render_h；y = LOD 早停开关（GATE_NO_LOD=1 关）
  pub lod: Vec4,
  pub probe_viz_params: Vec4,
}

/// `GATE_SKIP_CHUNKWALK=1`：trace_grid 在局部 slab 后直接 miss。
static SKIP_CHUNKWALK: LazyLock<bool> =
  LazyLock::new(|| std::env::var("GATE_SKIP_CHUNKWALK").map(|v| v == "1").unwrap_or(false));
/// `GATE_SKYOUT=1`：dda_main 跳过全部 trace 直接输出天空色。
static SKY_OUT: LazyLock<bool> =
  LazyLock::new(|| std::env::var("GATE_SKYOUT").map(|v| v == "1").unwrap_or(false));
/// `GATE_MAKEGRID_ONLY=1`：dda_main 只做 make_grid 不 trace。
static MAKEGRID_ONLY: LazyLock<bool> =
  LazyLock::new(|| std::env::var("GATE_MAKEGRID_ONLY").map(|v| v == "1").unwrap_or(false));
/// `GATE_NO_LOD=1`：关闭八叉树远场早停。
static LOD_DISABLED: LazyLock<bool> =
  LazyLock::new(|| std::env::var("GATE_NO_LOD").map(|v| v == "1").unwrap_or(false));
/// `GATE_NO_BEAM=1`：关闭 beam 预 pass，主 pass 从 t=0 起步（默认开启 beam）。
static BEAM_DISABLED: LazyLock<bool> =
  LazyLock::new(|| std::env::var("GATE_NO_BEAM").map(|v| v == "1").unwrap_or(false));
/// `GATE_NO_LUT=1`：关闭方向可达掩码剔除（lod.w 传 1 → shader 端旁路 LUT）。
static LUT_DISABLED: LazyLock<bool> =
  LazyLock::new(|| std::env::var("GATE_NO_LUT").map(|v| v == "1").unwrap_or(false));

impl DdaViewUniform {
  pub fn from_cfg(cfg: &DdaCameraConfig, debug_mode: u32, render_h: f32) -> Self {
    // 垂直 FOV 60°（镜像 gate-app FOV_Y），均分到 render_h 像素
    let px_ang = 2.0 * 30.0_f32.to_radians().tan() / render_h.max(1.0);
    Self {
      view_proj: cfg.view_proj,
      inv_view_proj: cfg.inv_view_proj,
      cam_pos_voxel: cfg.position_world.extend(1.0),
      debug_mode: Vec4::new(
        (debug_mode == 1) as u32 as f32,
        (debug_mode == 2) as u32 as f32,
        if *SKIP_CHUNKWALK { 2.0 } else { 0.0 },
        if *SKY_OUT {
          2.0
        } else if *MAKEGRID_ONLY {
          4.0
        } else if debug_mode == 3 {
          1.0
        } else {
          0.0
        },
      ),
      lod: Vec4::new(
        px_ang,
        (!*LOD_DISABLED) as u32 as f32,
        *BEAM_DISABLED as u32 as f32,
        *LUT_DISABLED as u32 as f32,
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

/// 工厂：DDA 目标纹理（rgba8unorm VIEW_SIZE，STORAGE|TEXTURE + RENDER_WORLD usage）。
/// `COPY_DST` 必须保留：bevy resize 的 `copy_image_on_resize` 依赖它。
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

/// WGSL 着色器顶部 `const` 的 Rust 镜像副本（改 WGSL 时必须一起改）。
pub mod wgsl_consts {
  pub const CHUNK_SIZE: u32 = 256;
  pub const BRICK_FACTOR: u32 = 4;
  pub const MAX_LEVEL: u32 = 4;
  /// 每节点 fixed 字数（mask_lo + mask_hi + palette_u32）
  pub const NODE_FIXED_WORDS: u32 = 3;
  pub const CHUNK_INDEX_CAP: u32 = 64;
  pub const CHUNK_INDEX_WORDS: u32 = 262_144;
  pub const TREE_BASE: u32 = 262_144;
  /// 调色板字数（2^16 条 × 2w）；与 `wire.rs::PALETTE_WORDS` 同源
  pub const PALETTE_WORDS: u32 = crate::brickmap::wire::PALETTE_WORDS as u32;
  /// 叶父层 inline 字数与每字体素数，与 `wire.rs` 同源
  pub const LEAF_INLINE_WORDS: u32 = crate::brickmap::wire::LEAF_INLINE_WORDS as u32;
  pub const LEAF_VOXELS_PER_WORD: u32 = crate::brickmap::wire::LEAF_VOXELS_PER_WORD as u32;
  pub const CHUNK_COMP_WORDS: u32 = 2048; // u16[4096] → 每 2 字打包 u32
  pub const STATE_ENTRY_COUNT: u32 = 256;
  pub const STATE_WORDS_PER_ENTRY: u32 = 4;
  pub const STATE_TOTAL_WORDS: u32 = 1024;
  pub const SHADOW_BIAS: f32 = crate::lighting::SHADOW_BIAS;
  pub const SHADOW_DIR_T_MAX: f32 = crate::lighting::SHADOW_DIR_T_MAX;
  pub const EMISSIVE_EMIT_GAIN: f32 = crate::lighting::EMISSIVE_EMIT_GAIN;
  // 光照场（AO fill）：cell = 16 voxel，dims = 32³ cell → 世界覆盖 512 voxel = ±5.12m（相机中心）。
  // 世界锚定寻址：原点按 cell 向下对齐，槽位 = 世界 cell mod dims。格式 Rgba16Unorm：.a = AO fill，.rgb 恒 0。
  pub const LIGHT_FIELD_CELL: u32 = 16;
  pub const LIGHT_FIELD_DIM: u32 = 32;
  /// 射线起点沿法线自体素表面再外推的量（体素）；必须与 WGSL `SHADOW_SURFACE_EPS` 一致。
  pub const SHADOW_SURFACE_EPS: f32 = 0.03125;
}

/// 与 WGSL `popcount(mask & (bit - 1u64))` 等价：mask bit=1 子块在紧凑 child offset 表中的槽位。
#[inline]
pub fn wgsl_child_slot_index(mask: u64, child_idx: u32) -> u32 {
  debug_assert!(child_idx < 64);
  (mask & ((1u64 << child_idx) - 1)).count_ones()
}

use crate::brickmap::{BrickMapBuffers, BrickMapView};

/// A&W 细格步进 DDA 参考实现（CPU），仅用 `BrickMapView::get_voxel` 查 palette，
/// 返回 `Some((hit_t, palette))` 或 `None`。
/// cell 坐标必须用整数增量维护（`floor(origin)` 起步，每次穿越 `+sign`），不得用 `floor(origin + dir*t)` 重算。
pub fn cpu_reference_dda_ray(
  buffers: &BrickMapBuffers,
  origin_voxel: Vec3,
  dir_voxel: Vec3, // voxel 单位，magnitude 任意
  t_max: f32,
  max_steps: u32,
) -> Option<(f32, u16)> {
  let view = BrickMapView::new(buffers);
  let mut t = 0.0f32;
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
  let next_boundary = |c: i32, s: i32| -> f32 { (if s >= 0 { c + 1 } else { c }) as f32 };
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

  if let Some(pal) = view.get_voxel(IVec3::from_array(cell)) {
    return Some((t, pal));
  }
  for _ in 0..max_steps {
    if t >= t_max {
      return None;
    }
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

/// 带 AABB 跳步的参考 DDA：先求射线与 AABB 的相交段；不相交返回 `None`，相交则把起点推进到 AABB 入口。
/// 标尺：t 自 `start = origin + dir·t_enter` 量起，返回的 `hit_t` 已加回 `t_enter`（自 origin 的全局距离）。
pub fn cpu_reference_dda_ray_aabb_skip(
  buffers: &BrickMapBuffers,
  origin: Vec3,
  dir: Vec3,         // 归一化
  t_global_max: f32, // 自 origin 量起的全局上限
  max_steps: u32,
  aabb_min: Vec3,
  aabb_max: Vec3,
) -> Option<(f32, u16)> {
  let mut t_enter = 0.0f32;
  let mut t_exit = t_global_max;
  let mut miss = false;
  for axis in 0..3 {
    let o = [origin.x, origin.y, origin.z][axis];
    let d = [dir.x, dir.y, dir.z][axis];
    let mn = [aabb_min.x, aabb_min.y, aabb_min.z][axis];
    let mx = [aabb_max.x, aabb_max.y, aabb_max.z][axis];
    if d.abs() < 1e-30 {
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
    return None;
  }
  let t_enter = t_enter.max(0.0);
  let t_exit = t_exit.min(t_global_max);
  if t_exit <= t_enter {
    return None;
  }

  let start = origin + dir * t_enter;
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

  if let Some(pal) = view.get_voxel(IVec3::from_array(cell)) {
    return Some((t_enter, pal));
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

/// 细级：单个粗 cell（16³ voxel）内的有界 voxel DDA，射线段限制在 `[t_lo, t_hi]`（自 origin 全局标尺）。
/// 入口胞用解析 + clamp 到 `[base, base+15]` 确定，保证起点必在本 cell 内（入口面外侧胞由前一粗 cell 覆盖）。
#[allow(clippy::too_many_arguments)]
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
  // voxel tmax：相对 t_lo 的距离（自 cell 入口重算，非累加）
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
/// 语义同 `cpu_reference_dda_ray`：射线 (origin, dir, t∈[0,t_max]) 上首个非空 voxel 体素，返回命中记录或 None。
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
    // 占用查询先行：空 cell 不做任何 voxel 采样
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
// 层次栈式 mask DDA：WGSL shaders/voxel_raytrace/ 中 TraceFrame/init_tree_frame/trace_chunk/
// trace_grid 的 CPU 逐字镜像。节点 mask 一次读进寄存器，4³=64 子块间步进只查 bit（零 load）；
// bit=1 分裂才压栈下钻，bit=0 uniform 子块整格跳过/整格命中。
// ============================================================================

/// 层次遍历命中记录（镜像 WGSL VoxelHit/UnifiedHit）：face_id 0..5 = ±xyz 六面。
/// voxel = 命中固体体素 grid 局部 voxel 整数坐标（DDA 整数步进产出）。
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
/// level 编号：3 = 根（256³，子块 64³）、2（64³，子块 16³）、1（16³，子块 4³）、0（4³，inline 1³ 叶）。
#[derive(Clone, Copy)]
struct BrickCpu {
  addr: usize, // 节点绝对字址（b_struct）
  mask: u64,   // 64bit 分裂掩码（bit=1 = 子块分裂；bit=0 = 统一子块，色=pal）
  pal: u16,    // 节点 palette（统一子块颜色，0=空气）
}

/// 镜像 WGSL trace_chunk：单 chunk 内整数体素层级 DDA（integer voxel + brick mask 栈 + firstTrailingBit 跨级跳）。
/// chunk_base/chunk_min = chunk 根字址/原点（局部 voxel）；射线段 [t0, t1]（ro 系绝对 t）；返回 (t, pal, face_id, 局部 v)。
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
  // 擦边退化（t0>=t1：射线只蹭到 chunk 边界）→ 直接 miss
  if t0 >= t1 {
    return None;
  }
  // chunk 局部 voxel 坐标（chunk 原点 = 0）；t 仍是 ro 系绝对 t
  let ro_c = [ro[0] - chunk_min[0], ro[1] - chunk_min[1], ro[2] - chunk_min[2]];
  // 预算倒数：side 距离/步长增量改乘法；与 WGSL inv_rd 镜像
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
    // ---- traverse：从当前 level 下钻到 v 处内容 ----
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
    // ---- v 处为空气：当前 level brick 内 DDA ----
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
    // 每轴步长 t 增量（level 不变则不变）
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
      // 新子块内容：level 0 查 inline palette（mask!=0 inline 叶才有）；level 1..3 = 分裂位或节点统一实体色。
      // 命中直接返回（cur_t=进入距离、face=进入面）；仅「分裂子块」回 traverse 下钻。
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
    // ---- firstTrailingBit 层级自适应跨级跳 ----
    // 步进轴新坐标的尾随零位 = 对齐 run 长度：正向 comp=对齐基址，负向 comp=区域尾址+1；tz>>1 = 可跨步的最粗 level。
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
    // 对齐快照：v 钳到 cur_t 射线点所在的当前 level 区域，步进轴取精确边界整数（其余轴按射线实际位置吸附）。
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
/// ro/rd 为局部 voxel 坐标（rd 含 1/scale）；chunk 步数上限 = (dims.x+dims.y+dims.z)*3 + 16，与 WGSL make_grid 一致。
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

/// 主世界层次 DDA（identity 变换）：局部 AABB = chunk 窗口范围（与 WGSL make_grid idx=0 的 is_world 分支一致）。
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
// 多 volume CPU 参考 trace：入口 = `cpu_reference_trace_volumes` / `cpu_reference_volumes_occluded`。
// idx 0 = 主世界（identity transform、无界 chunk HashMap）；idx 1..N = 物体（任意 transform、单 chunk）。
// ============================================================================

use gate_voxel::VolumeTransform;

/// 统一 volume 命中记录。
/// `obj_id` 约定与 `Volumes` 一致：-1 = 主世界，0..N-1 = 物体索引（`Volumes.list[1..]` 的 0-based 索引）。
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
/// 返回 (全局 t, palette, 局部面法线) 或 None；t 标尺全局（rd 含 1/scale，与 WGSL trace_grid 同一标尺）。
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
/// idx 0 = 主世界（identity transform）；idx 1..N = 物体；命中后收紧其余 volume 的 t_cap。
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

/// `trace_volumes()` 遮挡快路径：t_max 内任一命中即 true（不做最近比较）。
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
      ColorTargetState, ColorWrites, ComputePipelineDescriptor, Extent3d, FilterMode,
      FragmentState, PipelineCache, RenderPassDescriptor, SamplerBindingType, ShaderStages,
      StorageTextureAccess, TextureDescriptor, TextureDimension, TextureFormat, TextureSampleType,
      TextureUsages, TextureViewDescriptor, UniformBuffer, VertexState,
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

/// BG3 光池持久 GPU buffer（prepare 每帧覆写同 buffer）。
#[derive(Resource)]
pub(crate) struct LightPoolGpu(UniformBuffer<LightPoolUniform>);

/// 辅助纹理缓存（屏幕尺寸相关，resize 时重建）：
///   · `texture`：beam depth（低分辨率 r32float），beam 预 pass 写、主 pass 读；
///   · `gi_*`：半分辨率 GI 缓冲（`DdgiDebugSettings.gi_half_res`），`gi_main` 写、主 pass 采样（BG0 4/5）；
///     存 premultiplied valid：rgba16f = (gi·valid, valid)、rg32f = (cov·valid, valid)，采样侧按 valid 归一化；
///   · `gi_bg0`：GI pass 自己的 @group(0)（view uniform + beam depth，不含 GI 采样视图）。
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
  /// group(5) 的 GI 采样侧 bind group（`dda_main` 用；layout = `DdaPipelines::gi_read_layout`）
  gi_read_bg: Option<BindGroup>,
}

impl AuxTexCache {
  /// 半分辨率 GI 的写入侧视图（BG5 的 binding 2/3 用）。
  /// `None` = 尚未创建（首帧，或本帧 `prepare_dda_bind_groups` 提前返回）⇒ 调用方须用占位纹理。
  pub(crate) fn gi_write_views(&self) -> Option<(&TextureView, &TextureView)> {
    Some((self.gi_view.as_ref()?, self.gi_cov_view.as_ref()?))
  }
}

#[derive(Resource)]
#[allow(dead_code)]
pub(crate) struct DdaPipelines {
  pub(crate) bg0_layout: BindGroupLayoutDescriptor,
  /// BG0 的"瘦"版：`view uniform + beam depth`（供 `gi_main` 用，该 pass 要写 GI 纹理）。
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
  /// 同 layout / 同 bind group，仅 fragment 入口换成 `fs_fxaa`（抗锯齿开关，见 [`PostFxSettings`]）
  blit_fxaa_pipeline: CachedRenderPipelineId,
}

/// 眼睛适应的 GPU 状态（`EYE_WORDS` 字 storage buffer）+ 上一帧时间戳（算 dt）。
/// word 布局见 bindings.wesl 的 `eye_adapt`。
#[derive(bevy::ecs::resource::Resource)]
pub struct EyeAdaptGpu {
  pub buf: Option<Buffer>,
  /// 渲染侧墙钟（与 profiler 同源）：适应速度与帧率无关
  pub last: Option<std::time::Instant>,
  pub bg: Option<BindGroup>,
  /// 活参数（UI 可调）。
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

/// 把 main world 的活参数搬进 [`EyeAdaptGpu`]（只做搬运，真正上传在 `prepare_dda_bind_groups`）。
/// `Res::is_changed` 由 `ExtractResourcePlugin` 在同步时标记。
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

/// 眼睛适应（自动曝光）的活参数：由 debug overlay 的「Eye」页实时调，改完下一帧生效。
/// 下标顺序即语义，与 WESL 侧 `eye_p(i)` 一一对应（改这里必须同步 `main.wesl`）。
#[derive(Resource, Clone, Copy, Debug, PartialEq, ExtractResource)]
pub struct EyeAdaptSettings {
  /// 总开关：关掉 = 两个 eye pass 停发 + 曝光回落 1.0。不占参数区槽位（host 侧开关）。
  /// 初值受 `GATE_NO_EYE_ADAPT=1` 影响，之后由 Eye 页开关接管。
  pub enabled: bool,
  /// [0] 提亮上限（档，≥0）：适应暗处的最大增益 = 2^ev_max
  pub ev_max: f32,
  /// [1] 压暗上限（档，≤0）：适应亮处的最大衰减 = 2^ev_min
  pub ev_min: f32,
  /// [2] 变亮时间常数（秒）：往亮处适应速度
  pub tau_brighten: f32,
  /// [3] 变暗时间常数（秒）：往暗处适应速度
  pub tau_darken: f32,
  /// [4] 目标中灰：百分位平均亮度被压到这个值
  pub key: f32,
}

impl Default for EyeAdaptSettings {
  /// 缺省值（EV ±3）。
  fn default() -> Self {
    Self { enabled: true, ev_max: 3.0, ev_min: -3.0, tau_brighten: 2.0, tau_darken: 1.0, key: 0.18 }
  }
}

impl EyeAdaptSettings {
  /// 缺省 + 环境变量覆盖：`GATE_NO_EYE_ADAPT=1` 只决定初值，之后以面板开关为准。
  pub fn from_env() -> Self {
    let off = std::env::var("GATE_NO_EYE_ADAPT").map(|v| v == "1").unwrap_or(false);
    Self { enabled: !off, ..Self::default() }
  }
}

/// `eye_adapt` buffer 里参数区的起始字（= 状态/调试区 8 + 直方图 64）
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
      // 后处理开关（抗锯齿）：只在变化时同步，blit 侧按它选 pipeline
      bevy::render::extract_resource::ExtractResourcePlugin::<PostFxSettings>::default(),
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
          // prepare_dda_bind_groups 在 prepare (upload.rs) 之后运行：先 upload 写 grid_descs_buf 再绑 BG2。
          .after(super::upload::prepare),
      )
      // 必须挂 RenderGraph::Render set（而非 Render schedule）。
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
        // @binding(3) 眼睛适应的状态/直方图（只读视图；`dda_main` 只取曝光系数）。
        // 同一 buffer 在 BG7 以 read_write 被 eye_adapt_* 两个入口读写（不同 pass）。
        storage_buffer_read_only_sized(false, None),
      ),
    ),
  );

  // ---- BG5（GI 采样侧，只给 `dda_main` 的 pipeline 用）：半分辨率 GI 的两张纹理 ----
  // 绑定号 4/5（group(5) 空闲号，0..3 已被图集/GI 写入侧占），只进 `dda_main` 的 layout。
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
  // 绑定号与完整版一致（1/2），不含 out_tex / eye_adapt_ro / GI 采样视图。
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
  // GridDesc 144B/entry：pos_scale/rot0/rot1/rot2 + aabb_min/max +
  // tree_base/tree_depth/chunk_count/palette_base + index_origin/dims。
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

  // ---- Compute pipeline：shaders/voxel_raytrace/ 两个入口
  // （dda_main 主 trace+unlit 直出 / beam_main beam 预 pass）----
  let dda_shader = dda_shader.0.clone();
  let layouts =
    vec![bg0.clone(), bg1.clone(), bg2.clone(), bg3.clone(), crate::ddgi::ddgi_bg4_layout()];
  // `dda_main` 比其它两个入口多一份 group(5)：半分辨率 GI 的采样侧（见 `gi_read`），只加给它。
  let dda_layouts = {
    let mut v = layouts.clone();
    v.push(gi_read.clone());
    v
  };
  // 眼睛适应的两个入口：8 份相同 eye layout（wgpu 要求 bind group 从 0 起按索引前缀设置，两入口绑定在 @group(7)）。
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
  // 半分辨率 GI（菜单开关 `DdgiDebugSettings.gi_half_res`）：
  // 反投影 + beam 起点 + 主 trace + ddgi_sample，写两张 1/2 分辨率缓冲。
  // group0 用瘦版（不含 GI 采样视图），并多一个 BG5；layout 索引必须是 0..=5 的前缀。
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
  // 1/16 抽样 → 64 桶 log2 亮度直方图 → 5%~95% 百分位均值 → 时间平滑 → 曝光系数（下一帧 dda_main 读 BG0 binding(3)）。
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
  // 两条：`fs_main`（纯 blit）/ `fs_fxaa`（FXAA 抗锯齿）；同 layout、同 bind group，运行时按 `PostFxSettings.fxaa` 选一条。
  let blit_shader = asset_server.load(BLIT_SHADER_ASSET_PATH);
  let blit_make = |label: &str, entry: &str| {
    pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
      label: Some(Cow::from(label.to_string())),
      layout: vec![blit.clone()],
      vertex: VertexState {
        shader: blit_shader.clone(),
        entry_point: Some(Cow::from("vs_main")),
        ..default()
      },
      fragment: Some(FragmentState {
        shader: blit_shader.clone(),
        entry_point: Some(Cow::from(entry.to_string())),
        targets: vec![Some(ColorTargetState {
          format: TextureFormat::Rgba8UnormSrgb,
          blend: None,
          write_mask: ColorWrites::ALL,
        })],
        ..default()
      }),
      ..default()
    })
  };
  let blit_pipeline = blit_make("gate_dda_blit", "fs_main");
  let blit_fxaa_pipeline = blit_make("gate_dda_blit_fxaa", "fs_fxaa");

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
    blit_fxaa_pipeline,
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
  // z = 层级选择（0=全部，1..=4=LOD0..3）；w = 0
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
  // rg32f = (cov·valid, valid)。
  // wgpu 新建纹理自动清零 ⇒ valid 初值 0 = "无数据"，采样侧退回 conf=0 的天光兜底。
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
  // 眼睛适应状态/直方图 buffer（word 布局见 bindings.wesl 的 `eye_adapt`）：同一 buffer 两处绑定 ——
  // BG0 binding(3) 只读（`dda_main` 取曝光）+ BG7 binding(1) 读写；首帧曝光初始化为 1.0。
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
    // 参数区初值。
    queue.write_buffer(&b, EYE_PARAM_OFFSET, &eye_param_bytes(eye.settings));
    eye.buf = Some(b);
  }
  // clone 一份句柄（Buffer 内部是 Arc）。
  let eye_buf = eye.buf.clone().expect("刚插入");
  // dt 只在开启眼睛适应时才上传（关闭时零每帧写入）。
  // CPU 与 GPU 的 eye_adapt_* 都写同一张 buffer（buffer 粒度写-写冲突）。
  let now = std::time::Instant::now();
  if eye.settings.enabled {
    let dt = eye.last.map_or(1.0 / 60.0, |t| now.duration_since(t).as_secs_f32());
    queue.write_buffer(&eye_buf, 12, &dt.clamp(0.0, 0.25).to_bits().to_le_bytes());
  }
  eye.last = Some(now);
  // 活参数：只在设置变化时上传 20B（稳态零写入）。
  if std::mem::take(&mut eye.settings_dirty) {
    let s = eye.settings;
    queue.write_buffer(&eye_buf, EYE_PARAM_OFFSET, &eye_param_bytes(s));
    // 关掉总开关时把曝光回落到 1.0。
    if !s.enabled {
      queue.write_buffer(&eye_buf, 0, &1.0f32.to_bits().to_le_bytes());
    }
    // 每次真正推送记一行日志。
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
  // GI pass 的 @group(0)（瘦版 layout）：绑定号是 1/2（与完整版对齐），给显式 entry 数组。
  // `BindGroupEntries::sequential` 是按位置 = 绑定号，无法表达"从 1 开始"。
  // 不绑 GI 采样视图是硬性要求：同一 pass 内同一张纹理不能既作采样又作存储。
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

  // ---- Blit BG：dda tex（filterable）+ linear sampler ----
  // 必须是 Linear：半分辨率档（factor=2）上采样与 FXAA 亚像素偏移都依赖线性采样；
  // factor=1 时线性与最近邻等价（采样点落在纹素中心）。
  let blit_sampler = render_device.create_sampler(&SamplerDescriptor {
    label: Some("gate_dda_blit_sampler"),
    mag_filter: FilterMode::Linear,
    min_filter: FilterMode::Linear,
    ..default()
  });
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
  // （逐体素法线 + 天空渐变 + 太阳方向光项），无其余 direct/gi/denoise pass。
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
        // 纹理未就绪（`prepare_dda_bind_groups` 本帧提前返回）时退化为不绑。
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
  post: Option<Res<PostFxSettings>>,
  pipeline_cache: Res<PipelineCache>,
  pipelines: Res<DdaPipelines>,
  mut profiler: ResMut<crate::profiler::GpuProfilerRes>,
) {
  let (Some(bg), Ok(target)) = (blit_bg.as_ref(), views.single()) else {
    bevy::log::debug_once!("DDA blit: bg or ViewTarget missing");
    return;
  };
  // 抗锯齿 = 换一条 fragment 入口（`fs_fxaa`），bind group 完全相同 ⇒ 开关只在这里分支
  let id = if post.as_ref().is_some_and(|p| p.fxaa) {
    pipelines.blit_fxaa_pipeline
  } else {
    pipelines.blit_pipeline
  };
  let Some(pipe) = pipeline_cache.get_render_pipeline(id) else {
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
