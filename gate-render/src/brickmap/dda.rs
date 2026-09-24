//! DDA 主可见性 pass：WGSL compute + Core2d PostProcess blit。
//! BG0 = storage tex / 相机 uniform / beam depth / 眼睛适应状态（只读）；
//! BG1 = b_struct / b_leaves / palette / globals uniform / 线性采样器 / 材质资产表 /
//! PBR 贴图数组 / **PBR 专用采样器**（MT2-3）。
//! WGSL 源 = WESL 包 `shaders/voxel_raytrace/`（入口 `main.wesl`）。

use bevy::{
  asset::RenderAssetUsages,
  image::Image,
  prelude::*,
  render::{extract_resource::ExtractResource, render_resource::*},
};
use glam::camera::{rh::proj, rh::view};
use std::ops::Mul;

/// blit.wgsl 资产路径（全屏三角 blit）
pub const BLIT_SHADER_ASSET_PATH: &str = "shaders/blit.wgsl";
/// 主 DDA pass 工作组边长：必须与 `shaders/voxel_raytrace/` 中 dda_main 的 `@workgroup_size` 一致。
pub const DDA_WORKGROUP_SIZE: u32 = 8;

/// 当前渲染分辨率（main world `resize_render_targets` 更新，提取进 render world）。
#[derive(Resource, Clone, Copy, Debug, PartialEq, ExtractResource)]
#[extract_app(bevy::render::RenderApp)]
pub struct RenderScale {
  /// 渲染目标尺寸 = 窗口物理像素 ÷ `factor`（向下取整；blit 用整数块复制映射回窗口）。
  pub size: UVec2,
  /// 分辨率降采样除数：1 = 全分辨率、2/3/4 = 降到 1/2、1/3、1/4。
  pub factor: u32,
}

impl RenderScale {
  /// 菜单「视频/渲染分辨率」的四个档位（**下标 = 选中序号**）；上采样恒为整数块复制（不插值）。
  pub const SCALE_CHOICES: [u32; 4] = [1, 2, 3, 4];
}

impl Default for RenderScale {
  fn default() -> Self {
    Self { size: crate::consts::VIEW_SIZE, factor: 1 }
  }
}

/// 后处理开关（main world 由菜单写，提取进 render world）。
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq, ExtractResource)]
#[extract_app(bevy::render::RenderApp)]
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
  /// 视线方向（单位向量，含俯仰）。流式加载用它排"视野优先"（M5：需求要按相机真在看的方向排，
  /// 否则半径启发式会把帧额平均撒到相机身后）。
  pub forward: Vec3,
}

impl DdaCameraConfig {
  /// 静态视图构建（eye/target 为手算常量）
  pub fn build_static() -> Self {
    let eye = Vec3::new(700.0, 560.0, 700.0);
    let target = Vec3::new(260.0, 120.0, 260.0);
    let up = Vec3::Y;
    let aspect = crate::consts::VIEW_SIZE.x as f32 / crate::consts::VIEW_SIZE.y as f32;
    let fovy = 60.0_f32.to_radians();
    let near = 1.0;
    let far = 4000.0;
    let proj = proj::directx::perspective(fovy, aspect, near, far);
    let view = view::look_at_mat4(eye, target, up);
    let view_proj = proj.mul(view);
    let inv_view_proj = view_proj.inverse();
    Self { view_proj, inv_view_proj, position_world: eye, forward: (target - eye).normalize() }
  }
}

/// 调试视图模式（main world Resource，按 N 键循环 0→1→2→0）：
/// 0 = 正常画面；1 = 法向向量可视化；2 = G-buffer 状态图（sky=品红、face 6 色）
#[derive(Resource, Clone, Copy, Default, bevy::render::extract_resource::ExtractResource)]
#[extract_app(bevy::render::RenderApp)]
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
    let view = view::look_at_mat4(eye, eye + f, Vec3::Y);
    let proj = proj::directx::perspective(fov_y, aspect, near, far);
    let view_proj = proj.mul(view);
    Self { view_proj, inv_view_proj: view_proj.inverse(), position_world: eye, forward: f }
  }

  /// orbit 参数 → `proj::directx::perspective` × `view::look_at_mat4`（fov/aspect/near/far 为显式参数）。
  pub fn from_orbit(orbit: &OrbitCamera, fov_y: f32, aspect: f32, near: f32, far: f32) -> Self {
    let eye = orbit.eye();
    let view = view::look_at_mat4(eye, orbit.target, Vec3::Y);
    let proj = proj::directx::perspective(fov_y, aspect, near, far);
    let view_proj = proj.mul(view);
    Self {
      view_proj,
      inv_view_proj: view_proj.inverse(),
      position_world: eye,
      forward: (orbit.target - eye).normalize_or_zero(),
    }
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
  /// x = 单像素角大小(rad) = 2·tan(FOV_Y/2)/render_h；y = 叶级 LOD 开关（`consts::DDA_LOD`：
  /// 远场 4³ 值块整块取一个色，阈值见 `trace.wesl::LEAF_LOD_FP`）
  pub lod: Vec4,
}

/// 本文件用到的开关（`consts.rs`）
use crate::brickmap::consts::{
  DDA_BEAM, DDA_CHUNKWALK, DDA_DIR_LUT, DDA_LOD, DDA_MAKEGRID_ONLY, DDA_SKY_ONLY, EYE_ADAPT,
};

/// 单像素角大小（rad/px）= `2·tan(FOV_Y/2) / render_h`。
/// **唯一的计算点**：`DdaViewUniform.lod.x`（shader 的 `fp = t·px_ang`）与常驻档位阶梯
/// （`brickmap::residency::want_level`）都用它 ⇒ 两侧的"像素预算"口径不会分叉。
pub fn px_ang(render_h: f32) -> f32 {
  2.0 * (crate::brickmap::consts::DDA_FOV_Y * 0.5).tan() / render_h.max(1.0)
}

impl DdaViewUniform {
  pub fn from_cfg(cfg: &DdaCameraConfig, debug_mode: u32, render_h: f32) -> Self {
    // 垂直 FOV 均分到 render_h 像素（FOV 见 `consts::DDA_FOV_Y`，镜像 gate-app consts）
    let px_ang = px_ang(render_h);
    Self {
      view_proj: cfg.view_proj,
      inv_view_proj: cfg.inv_view_proj,
      cam_pos_voxel: cfg.position_world.extend(1.0),
      debug_mode: Vec4::new(
        (debug_mode == 1) as u32 as f32,
        (debug_mode == 2) as u32 as f32,
        if DDA_CHUNKWALK { 0.0 } else { 2.0 },
        if DDA_SKY_ONLY {
          2.0
        } else if DDA_MAKEGRID_ONLY {
          4.0
        } else if debug_mode == 3 {
          1.0
        } else {
          0.0
        },
      ),
      lod: Vec4::new(
        px_ang,
        DDA_LOD as u32 as f32,
        !DDA_BEAM as u32 as f32,
        !DDA_DIR_LUT as u32 as f32,
      ),
    }
  }
}

/// DDA 着色器的纹理（main world 创建，提取进 render world）
#[derive(Resource, Clone, ExtractResource)]
#[extract_app(bevy::render::RenderApp)]
pub struct DdaImages {
  pub target: Handle<Image>,
}

/// 工厂：DDA 目标纹理（rgba8unorm `consts::VIEW_SIZE`，STORAGE|TEXTURE + RENDER_WORLD usage）。
/// `COPY_DST` 必须保留：bevy resize 的 `copy_image_on_resize` 依赖它。
pub fn create_dda_image(images: &mut Assets<Image>) -> Handle<Image> {
  let mut image = Image::new_target_texture(
    crate::consts::VIEW_SIZE.x,
    crate::consts::VIEW_SIZE.y,
    TextureFormat::Rgba8Unorm,
    None,
  );
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
  pub const SHADOW_BIAS: f32 = crate::consts::SHADOW_BIAS;
  pub const SHADOW_DIR_T_MAX: f32 = crate::consts::SHADOW_DIR_T_MAX;
  pub const EMISSIVE_EMIT_GAIN: f32 = crate::consts::EMISSIVE_EMIT_GAIN;
  /// 射线起点沿法线自体素表面再外推的量（体素）；必须与 WGSL `SHADOW_SURFACE_EPS` 一致。
  pub const SHADOW_SURFACE_EPS: f32 = 0.03125;
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
        sampler, storage_buffer_read_only_sized, storage_buffer_sized, texture_2d,
        texture_2d_array, texture_storage_2d, uniform_buffer,
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
///   · `gi_*`：GI 缓冲（网格尺寸 = 渲染分辨率 ÷ `GiSettings.gi_div`，1 / 2 / 4），`gi_main` 写、
///     主 pass 采样（BG0 binding 4）；
///     存 premultiplied valid：rgba16f = (gi·valid, valid)，采样侧按 valid 归一化；
///     覆盖度 cov 不再是独立张量 —— 它写的恒是「valid ? 1 : 0」，与 `gi_tex.a` 同一个场；
///   · `gi_res_a`/`gi_res_b`：**屏幕空间逐面 ReSTIR** 的 reservoir 双缓冲（BG4 binding 20/21，
///     每 GI 像素 `GI_RES_WORDS` 个 word；布局见 `gi/screen.wesl`）。两块随 GI 网格尺寸一起
///     重建（wgpu 新建 buffer 恒为零 ⇒ 新尺寸下 `M = 0` = 无历史）；`prepare_gi` 每帧换绑
///     （20 = 本帧写、21 = 上帧读）。
///   · `gi_bg0`：GI pass 自己的 @group(0)（view uniform + beam depth，不含 GI 采样视图）。
/// 换分辨率（`gi_div` 1↔2↔4）与换窗口尺寸走的是同一条重建路径 ⇒ 所有 GI 派生资源（reservoir、
/// 导引、历史、φ、中间靶、降噪输出）尺寸恒等于 `gi_size`，不存在「只重建一部分」的状态。
#[derive(Resource, Default)]
pub(crate) struct AuxTexCache {
  texture: Option<Texture>,
  size: UVec2,
  gi_tex: Option<Texture>,
  gi_view: Option<TextureView>,
  gi_size: UVec2,
  gi_res_a: Option<Buffer>,
  gi_res_b: Option<Buffer>,
  gi_bg0: Option<BindGroup>,
  /// group(5) 的 GI 采样侧 bind group（`dda_main` 用；layout = `DdaPipelines::gi_read_layout`）
  gi_read_bg: Option<BindGroup>,
  // ---- GI 降噪（`gi_denoise_temporal` + `gi_denoise_atrous1/2/4`，见 `gi/denoise.wesl`）----
  /// 导引 buffer（`gi_main` 写、两段降噪读）：每 GI 像素 `GI_DEN_GUIDE_WORDS` 个 u32。
  gi_guide: Option<Buffer>,
  /// **帧内逐面去重表**（`face_slots`，@group(5) @binding(8)）：每槽 `FACE_WORDS` 个 u32，
  /// 槽数 = GI 网格像素数 × 2（2 的幂）。每帧在 `gi_main` 之前整块清空（`clear_buffer`）。
  face_slots: Option<Buffer>,
  /// **二次顶点按面缓存**（`gi_sec_slots`，@group(5) @binding(9)）：每槽 `GI_SEC_WORDS` 个 u32，
  /// 槽数规则同 `face_slots`；由 GI 射线自己惰性填充 ⇒ 同样每帧整块清空。
  gi_sec_slots: Option<Buffer>,
  /// 时域历史双缓冲（每像素 `GI_DEN_HIST_WORDS` 个 u32）：`den_flip` 决定哪块是「上帧读」。
  gi_hist: [Option<Buffer>; 2],
  /// 每像素亮度 range 权重尺度 φ（f32，时域写 / atrous 读）。
  gi_phi: Option<Buffer>,
  /// atrous 链的 4 张 GI 网格尺寸 rgba16f：`[0]` = 时域输出、`[1]`/`[2]` = ping-pong、
  /// `[3]` = 最终结果（`dda_main` 的 group(5) binding 4 绑它）。
  gi_dn: [Option<Texture>; 4],
  /// `gi_dn` 的采样视图：`[0..3)` 给 atrous 的输入，`[3]` 给 `dda_main`。
  gi_dn_src: [Option<TextureView>; 4],
  /// `gi_dn` 的存储视图：时域写 `[0]`，atrous 写 `[1]`/`[2]`/`[3]`。
  gi_dn_dst: [Option<TextureView>; 4],
  /// 降噪 bind group：`[0]` = 时域；`[1..7]` = atrous 的 6 种 src→dst 组合（见 `DEN_ATROUS_CHAINS`）。
  den_bg: [Option<BindGroup>; 7],
  /// 降噪 pass 的小配置 buffer（`@group(0) @binding(20)`，1 个 word = atrous 核半径）。
  /// 降噪 pass 的 layout 只有 group(0) 一份（刻意），够不到 `@group(4)` 的 `gi_u` ⇒ 单独给一个。
  /// `den_cfg_r` = 已写入的值（只在档位变化时重写 + 打日志）。
  den_cfg: Option<Buffer>,
  den_cfg_r: u32,
  /// 历史双缓冲 + atrous 轮次的换绑状态（每次真正跑降噪时翻转一次）。
  den_flip: bool,
  /// ① **逐面合并**的落地 pass（`gi_face_flatten`）的 bind group：只有 group(0) 三个绑定
  /// （10 = 导引、18 = 逐面去重表、19 = GI 写入侧），布局见 `crate::gi::gi_flatten_layout`。
  /// 与 GI 分辨率同帧重建（三件资源都是）。
  gi_flatten_bg: Option<BindGroup>,
}
/// atrous 的 src→dst 组合表（下标 = `AuxTexCache::den_bg` 的 1..7）。
/// 链的选取只取决于 `GI_DEN_ATROUS_ITER`（见 `DEN_ATROUS_ROUNDS`）。
/// 5 轮时 b→a 与 a→b 交替出现（只有 4 张中间靶，靠 ping-pong 撑起任意轮数），
/// 所以这里按「用到的组合」去重列，而不是按轮次列。
pub(crate) const DEN_ATROUS_CHAINS: [[usize; 2]; 6] =
  [[0, 1], [1, 2], [2, 3], [0, 3], [1, 3], [2, 1]];
/// `den_bg` 里「第 `i` 轮（0 起）该用哪个 src→dst 组合」的查表（按 `GI_DEN_ATROUS_ITER` 取前 n 项）。
/// 步长依次 1/2/4/8/16（= NRD/RELAX 的 5 轮 a-trous），足迹半径 = 16 个 GI 像素。
///   1 轮：tmp→den                        （步长 1）
///   2 轮：tmp→a、a→den                   （1、2）
///   3 轮：tmp→a、a→b、b→den              （1、2、4）
///   4 轮：tmp→a、a→b、b→a、a→den         （1、2、4、8）
///   5 轮：tmp→a、a→b、b→a、a→b、b→den    （1、2、4、8、16）
pub(crate) const DEN_ATROUS_ROUNDS: [[usize; 5]; 5] =
  [[3, 0, 0, 0, 0], [0, 4, 0, 0, 0], [0, 1, 2, 0, 0], [0, 1, 5, 4, 0], [0, 1, 5, 1, 2]];

impl AuxTexCache {
  /// GI 写入侧视图（BG5 的 binding 2 用）。
  /// `None` = 尚未创建（首帧，或本帧 `prepare_dda_bind_groups` 提前返回）⇒ 调用方须用占位纹理。
  pub(crate) fn gi_write_view(&self) -> Option<&TextureView> {
    self.gi_view.as_ref()
  }

  /// 屏幕空间 reservoir 的双缓冲（BG4 binding 20/21 用；`prepare_gi` 决定哪块是「本帧写」）。
  /// `None` = 尚未创建（首帧 / prepare 提前返回）⇒ 调用方须用占位 buffer。
  pub(crate) fn gi_res_buffers(&self) -> Option<(&Buffer, &Buffer)> {
    Some((self.gi_res_a.as_ref()?, self.gi_res_b.as_ref()?))
  }

  /// 降噪导引 buffer（BG5 binding 6 用）。
  pub(crate) fn gi_guide_buffer(&self) -> Option<&Buffer> {
    self.gi_guide.as_ref()
  }

  /// 帧内逐面去重表（BG5 binding 8 用；`gi_main` 认领、`dda_face_main` 写着色、`dda_main` 查表）。
  pub(crate) fn face_slots_buffer(&self) -> Option<&Buffer> {
    self.face_slots.as_ref()
  }

  /// 二次顶点按面缓存（BG5 binding 9 用；只被 `gi_main` 里 GI 射线的命中着色使用）。
  pub(crate) fn gi_sec_slots_buffer(&self) -> Option<&Buffer> {
    self.gi_sec_slots.as_ref()
  }

  /// GI pass 的 @group(0)（view uniform + beam depth）。
  /// **光柱的掩码 pass 复用同一份**（两者要的绑定逐字相同，见 `volumetric::dispatch_fog`）——
  /// 这一份不由 GI 的开关控制（`prepare_dda_bind_groups` 无条件建），所以 GI 关掉时光柱照常能跑。
  pub(crate) fn gi_bg0(&self) -> Option<&BindGroup> {
    self.gi_bg0.as_ref()
  }
}

#[derive(Resource)]
#[allow(dead_code)]
pub(crate) struct DdaPipelines {
  pub(crate) bg0_layout: BindGroupLayoutDescriptor,
  /// BG0 的"瘦"版：`view uniform + beam depth`（供 `gi_main` 用，该 pass 要写 GI 纹理）。
  pub(crate) bg0_gi_layout: BindGroupLayoutDescriptor,
  /// group(5) 的"GI 采样侧"（`dda_main` 专用）：两张 GI 网格尺寸的纹理（绑定号 4/5）
  pub(crate) gi_read_layout: BindGroupLayoutDescriptor,
  pub(crate) bg1_layout: BindGroupLayoutDescriptor,
  pub(crate) bg2_layout: BindGroupLayoutDescriptor,
  pub(crate) bg3_layout: BindGroupLayoutDescriptor,
  blit_layout: BindGroupLayoutDescriptor,
  /// BG7：眼睛适应（out_tex 采样视图 + 状态/直方图 storage）
  eye_layout: BindGroupLayoutDescriptor,
  pub(crate) compute_pipeline: CachedComputePipelineId,
  /// 逐面着色 pass（`dda_face_main`）：layout / bind group 与主 pass **完全相同**，只是入口不同。
  pub(crate) face_pipeline: CachedComputePipelineId,
  /// 逐面 GI 累加 pass（`dda_face_accum`）：`dda_face_main` 的前一站，layout / bind group 同上。
  pub(crate) face_accum_pipeline: CachedComputePipelineId,
  pub(crate) beam_pipeline: CachedComputePipelineId,
  /// GI（菜单「渲染/GI」开关 + 分辨率档）：`gi_main`
  pub(crate) gi_pipeline: CachedComputePipelineId,
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
#[extract_app(bevy::render::RenderApp)]
pub struct EyeAdaptSettings {
  /// 总开关：关掉 = 两个 eye pass 停发 + 曝光回落 1.0。不占参数区槽位（host 侧开关）。
  /// 初值见 [`EYE_ADAPT`]（`consts.rs`），之后由 Eye 页开关接管。
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
  /// 启动值：初值来自 [`EYE_ADAPT`]（`consts.rs`），之后以面板开关为准。
  pub fn startup() -> Self {
    Self { enabled: EYE_ADAPT, ..Self::default() }
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
      // 镜面档位（菜单「渲染/反射」）：main world 是菜单的写入目标，render world 供
      // `prepare_dda_bind_groups` 写进 BG3 光池 uniform 的 `refl_tier` / `refl_nest`。
      bevy::render::extract_resource::ExtractResourcePlugin::<crate::lighting::ReflectionSettings>::default(),
      // 「基础」开关（菜单「渲染/基础」）：同上，写进 `base_flags`。
      bevy::render::extract_resource::ExtractResourcePlugin::<crate::lighting::BaseSettings>::default(),
      // 眼睛适应的活参数（debug overlay 的 Eye 页可调；变化才同步 → 稳态零上传）
      bevy::render::extract_resource::ExtractResourcePlugin::<EyeAdaptSettings>::default(),
      crate::responsive::ResponsivePlugin,
    ));
    app.insert_resource(EyeAdaptSettings::startup());
    // main world 侧先给出默认档位（菜单观察者按路径写它）；render world 那份由
    // `ExtractResourcePlugin::<ReflectionSettings>` 拷过去。
    app.init_resource::<crate::lighting::ReflectionSettings>();
    // 同上：「基础」开关（菜单「渲染/基础」）。
    app.init_resource::<crate::lighting::BaseSettings>();

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
  let render_h = scale.map(|s| s.size.y as f32).unwrap_or(crate::consts::VIEW_SIZE.y as f32);
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

  // ---- BG5（GI 采样侧，只给 `dda_main` 的 pipeline 用）：GI 与降噪导引 ----
  // 绑定号 4/6（group(5) 空闲号，0..3 已被图集/GI 写入侧占），只进 `dda_main` 的 layout。
  // 4 = GI（降噪后）；6 = **降噪导引**（`dda_main` 做几何感知上采样时按「同平面」筛 tap）。
  // 覆盖度 cov 不占绑定：它恒等于「valid ? 1 : 0」，`dda_main` 从 `gi_tex.a` 自取。
  // 6 的访问类型必须是 read_write：同一份 shader 模块里这个 var 在 `gi_main` 的 pipeline 里被写
  // （见 `gi::gi_bg5_layout`），WGSL 的声明类型是全局唯一的，两边 layout 不一致会被 wgpu 拒掉。
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
        binding: 6,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
          ty: BufferBindingType::Storage { read_only: false },
          has_dynamic_offset: false,
          min_binding_size: None,
        },
        count: None,
      },
      // 8 = 帧内逐面去重表（`dda_main` 只读、`dda_face_main` 写着色）。同 6：虽然这两个入口只读，
      // 但 WGSL 里 `face_slots` 是 `array<atomic<u32>>`（`gi_main` 要 CAS）⇒ 声明与 layout 都必须是
      // read-write，否则 wgpu 报绑定类型不匹配。
      BindGroupLayoutEntry {
        binding: 8,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
          ty: BufferBindingType::Storage { read_only: false },
          has_dynamic_offset: false,
          min_binding_size: None,
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
  // 6/7/8/9 = MT2-2 / MT2-3 的全局材质资产表 + PBR 贴图数组 + **PBR 专用采样器**
  // （**BG1 被 dda / beam / gi 三个 pass 共用**，这几条只在这一份 layout 里加；
  // `dda.rs` 是 BG1 layout 的唯一出处，`gi` 侧 no-op）。
  // **binding 号到 9 为止**：MT8-3 曾在 10/11 加一对反射缓存 buffer（`array<ReflEntry>`），
  // 已随反射缓存整体删除（实测负优化，见 `assets/shaders/voxel_raytrace/main.wesl` 文件头）。
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
        // @binding(4)：GI 缓冲（`gi_tex`）的采样器 —— ClampToEdge ×3、mipmap_filter = Nearest。
        // PBR 贴图走 @binding(8)（那一份要 Repeat + Linear mipmap，状态要求相反）。
        sampler(SamplerBindingType::Filtering),
        // @binding(5)：全局材质资产表（`array<MaterialAsset>`，32B/条 = 32KB）
        storage_buffer_read_only_sized(false, None),
        // @binding(6)/(7)：PBR 贴图数组（`texture_2d_array`，层 = 材质槽号）——
        // 视图维度必须是 `D2Array`（`texture_2d_array()` 已按此生成 layout entry）。
        texture_2d_array(TextureSampleType::Float { filterable: true }),
        texture_2d_array(TextureSampleType::Float { filterable: true }),
        // @binding(8)：PBR 贴图**专用采样器**（MT2-3）——三轴 Repeat（triplanar 平铺）+
        // Linear mag/min/**mipmap**（贴图集带完整 mip 链）。**不能与 4 合并**：GI 缓冲那个是
        // ClampToEdge 且无 mip，两者状态要求相反。desc 权威在
        // `pbr_texture::create_pbr_sampler`（占位与真身共用 `GpuBrickMap.pbr_sampler`）。
        sampler(SamplerBindingType::Filtering),
        // @binding(9)：叶级 LOD 诊断计数器（M0；`trace.wesl::LOD_DIAG` 打开时才有写入，其余时候恒 0）。
        // 放 BG1 而不是 BG0：BG1 是 dda / beam / gi / 光柱掩码四个**会调 trace** 的 pass 共用的那一份，
        // 而 BG0 有两份（完整版 + GI 瘦版）⇒ 挂这里只需改这一处 layout 与下面那一处 BG1 bind group。
        storage_buffer_sized(false, None),
        // @binding(10)：**M4 ray-guided 请求环缓冲**（read_write —— shader 侧原子追加，
        // 见 `trace.wesl::{REQ_ENABLE, req_push}`）。
        storage_buffer_sized(false, None),
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
    vec![bg0.clone(), bg1.clone(), bg2.clone(), bg3.clone(), crate::gi::gi_bg4_layout()];
  // `dda_main` 比其它两个入口多两份 group：group(5) = GI 的采样侧（见 `gi_read`）、
  // group(6) = 光柱（godray）的采样侧（见 `crate::volumetric::fog_read_layout`），只加给它。
  // 逐面两个 pass 与主 pass 共用同一份 layout（下面 `dda_layouts_face`）—— 它们不引用 group(6)，
  // 派发时也不需要设它（未用的绑定会被剪掉）。
  let dda_layouts = {
    let mut v = layouts.clone();
    v.push(gi_read.clone());
    v.push(crate::volumetric::fog_read_layout());
    v
  };
  // 逐面 pass 与主 pass 用同一份 layout（clone 一份，因为 `queue_compute_pipeline` 会取走）。
  let dda_layouts_face = dda_layouts.clone();
  let dda_layouts_accum = dda_layouts.clone();
  // 眼睛适应的两个入口：8 份相同 eye layout（wgpu 要求 bind group 从 0 起按索引前缀设置，两入口绑定在 @group(7)）。
  let eye_layouts = vec![eye.clone(); 8];
  let compute = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
    label: Some(Cow::from("gate_dda_compute")),
    layout: dda_layouts,
    shader: dda_shader.clone(),
    entry_point: Some(Cow::from("dda_main")),
    ..default()
  });
  // 逐面着色 pass（`dda_face_main`）：layout 与 `dda_main` **逐项相同**（group 0..5）⇒ 派发时
  // 直接复用主 pass 的 bind group，不额外建绑定；只是入口不同。
  let face = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
    label: Some(Cow::from("gate_dda_face")),
    layout: dda_layouts_face,
    shader: dda_shader.clone(),
    entry_point: Some(Cow::from("dda_face_main")),
    ..default()
  });
  // 逐面 GI 累加 pass（`dda_face_accum`）：同样复用主 pass 的 layout / bind group，只换入口。
  // 它是 `dda_face_main` 的**前一站**（见 `main.wesl`：同一个面的全部 texel 先累加进槽，再逐面着色）。
  let face_accum = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
    label: Some(Cow::from("gate_dda_face_accum")),
    layout: dda_layouts_accum,
    shader: dda_shader.clone(),
    entry_point: Some(Cow::from("dda_face_accum")),
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
  // GI（`GiSettings.gi_div` = 1 / 2 / 4；**每一档都跑这条链**）：
  // 反投影 + beam 起点 + 主 trace + 屏幕空间 ReSTIR，写两张 GI 网格尺寸的缓冲。
  // group0 用瘦版（不含 GI 采样视图），并多一个 BG5；layout 索引必须是 0..=5 的前缀。
  let gi_layouts = vec![
    bg0_gi.clone(),
    bg1.clone(),
    bg2.clone(),
    bg3.clone(),
    crate::gi::gi_bg4_layout(),
    crate::gi::gi_bg5_layout(),
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
    face_pipeline: face,
    face_accum_pipeline: face_accum,
    beam_pipeline: beam,
    gi_pipeline: gi,
    eye_histogram_pipeline: eye_histogram,
    eye_update_pipeline: eye_update,
    blit_pipeline,
    blit_fxaa_pipeline,
  });
  commands.insert_resource(LightPoolGpu(UniformBuffer::default()));
  commands.insert_resource(AuxTexCache::default());
  commands.insert_resource(EyeAdaptGpu::default());
}

/// `prepare_dda_bind_groups` 的**档位资源**打包成一个 `SystemParam`：
/// Bevy 的系统函数最多 16 个参数，本系统正好卡在边界上（见 `volumetric.rs::FogRes` 同一个先例）。
/// 各项都是只读、都只在"建 BG/纹理尺寸"时用几次。
#[derive(bevy::ecs::system::SystemParam)]
pub(crate) struct DdaTune<'w> {
  /// GI 档位（分辨率除数 / 降噪质量）—— 决定 GI 纹理尺寸与绑哪张降噪输出。
  pub gi: Option<Res<'w, crate::gi::GiSettings>>,
  /// 镜面档位（菜单「渲染/反射」）—— 写进 BG3 光池 uniform 的 `refl_tier` / `refl_nest`。
  pub refl: Option<Res<'w, crate::lighting::ReflectionSettings>>,
  /// 「基础」开关（菜单「渲染/基础」）—— 写进 BG3 光池 uniform 的 `base_flags`。
  pub base: Option<Res<'w, crate::lighting::BaseSettings>>,
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
  pbr_set: Option<Res<crate::pbr_texture::PbrTextureSet>>,
  lighting: Option<Res<LightingTheme>>,
  light_gpu: Option<ResMut<LightPoolGpu>>,
  render_device: Res<RenderDevice>,
  pipeline_cache: Res<PipelineCache>,
  queue: Res<RenderQueue>,
  scale: Res<RenderScale>,
  tune: DdaTune,
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

  let mut u = UniformBuffer::from(*view_uniform);
  u.write_buffer(&render_device, &queue);

  let bg0_layout = pipeline_cache.get_bind_group_layout(&pipelines.bg0_layout);
  let eye_layout = pipeline_cache.get_bind_group_layout(&pipelines.eye_layout);
  let bg1_layout = pipeline_cache.get_bind_group_layout(&pipelines.bg1_layout);
  let bg2_layout = pipeline_cache.get_bind_group_layout(&pipelines.bg2_layout);
  let bg3_layout = pipeline_cache.get_bind_group_layout(&pipelines.bg3_layout);
  let blit_layout = pipeline_cache.get_bind_group_layout(&pipelines.blit_layout);

  // ---- beam depth：低分辨率 r32float（全分辨率 ÷ BEAM_DIV），resize 时重建 ----
  let beam_div = crate::brickmap::consts::BEAM_DIV;
  let beam_size = UVec2::new(scale.size.x.div_ceil(beam_div), scale.size.y.div_ceil(beam_div));
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

  // ---- GI 缓冲（菜单「渲染/GI/分辨率」= 1/2/4）：网格 = 渲染分辨率 ÷ gi_div ----
  // 换分辨率（gi_div 档位切换）与换窗口尺寸都走这条重建路径（条件只看 gi_size 变没变）。
  // 存 premultiplied valid（见 AuxTexCache 的说明）：rgba16f = (gi·valid, valid)。
  // wgpu 新建纹理自动清零 ⇒ valid 初值 0 = "无数据"，采样侧退回 conf=0 的天光兜底。
  let gi_size = tune.gi.as_deref().copied().unwrap_or_default().gi_size(scale.size);
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
    let gi_tex = make("gate_gi", TextureFormat::Rgba16Float);
    beam_cache.gi_view = Some(gi_tex.create_view(&TextureViewDescriptor::default()));
    beam_cache.gi_tex = Some(gi_tex);
    beam_cache.gi_size = gi_size;
    // 屏幕空间路径的 reservoir 双缓冲（每像素 `GI_RES_WORDS` 个 u32）：随分辨率重建，
    // 新 buffer 由 wgpu 清零 ⇒ `M = 0`（无历史）⇒ 换分辨率后第一帧只走新鲜路径。
    let res_bytes =
      gi_size.x as u64 * gi_size.y as u64 * crate::wesl_consts::gi_consts().gi_res_words as u64 * 4;
    let make_res = |label: &str| {
      render_device.create_buffer(&BufferDescriptor {
        label: Some(label),
        size: res_bytes.max(4),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
      })
    };
    beam_cache.gi_res_a = Some(make_res("gate_gi_res_a"));
    beam_cache.gi_res_b = Some(make_res("gate_gi_res_b"));
    // ---- 降噪资源（随 GI 分辨率一起重建；新建 buffer/纹理由 wgpu 清零 ⇒ 历史 M = 0 = 无历史）----
    let c = crate::wesl_consts::gi_consts();
    let px = gi_size.x as u64 * gi_size.y as u64;
    let make_buf = |label: &str, bytes: u64| {
      render_device.create_buffer(&BufferDescriptor {
        label: Some(label),
        size: bytes.max(4),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
      })
    };
    beam_cache.gi_guide = Some(make_buf("gate_gi_guide", px * c.gi_den_guide_words as u64 * 4));
    // ---- 帧内逐面去重表（`face_slots`）：槽数 = GI 像素数 × 2（向上取 2 的幂，哈希才用得上位与），
    // 上限 2^21。参照：2K + 1/4 档 = 51.8 万 GI 像素 ⇒ 2^20 槽 = 32 MB。
    // 槽位不够只会抬高撞键率，而撞键 = 该面退回逐像素内联着色（正确性不变，只少赚）。
    let face_slots_n = (px * 2).next_power_of_two().clamp(256, 1u64 << 21);
    beam_cache.face_slots =
      Some(make_buf("gate_face_slots", face_slots_n * c.face_words as u64 * 4));
    // 二次顶点按面缓存：同一套槽数规则（被 GI 射线打到的面的数量与可见面同量级）。
    beam_cache.gi_sec_slots =
      Some(make_buf("gate_gi_sec_slots", face_slots_n * c.gi_sec_words as u64 * 4));
    beam_cache.gi_hist = [
      Some(make_buf("gate_gi_hist_a", px * c.gi_den_hist_words as u64 * 4)),
      Some(make_buf("gate_gi_hist_b", px * c.gi_den_hist_words as u64 * 4)),
    ];
    beam_cache.gi_phi = Some(make_buf("gate_gi_den_phi", px * 4));
    for (i, label) in
      ["gate_gi_dn_tmp", "gate_gi_dn_a", "gate_gi_dn_b", "gate_gi_den"].iter().enumerate()
    {
      let t = make(label, TextureFormat::Rgba16Float);
      beam_cache.gi_dn_src[i] = Some(t.create_view(&TextureViewDescriptor::default()));
      beam_cache.gi_dn_dst[i] = Some(t.create_view(&TextureViewDescriptor::default()));
      beam_cache.gi_dn[i] = Some(t);
    }
  }
  // clone 成句柄（`TextureView` 内部是 Arc）：之后还要可变借 `beam_cache` 写入 bind group 字段。
  let gi_view = beam_cache.gi_view.as_ref().expect("gi view not created").clone();

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
    // 每次真正推送记一行日志（settings 变化时才走这里）。
    bevy::log::debug!(
      target: "gate",
      "eye adapt → GPU: {} EV+ {:.2} / EV- {:.2} / tau+ {:.2}s / tau- {:.2}s / key {:.3}",
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
  // group(5) 的 GI 采样侧（只进 `dda_main` 的 pipeline layout）：绑定号 4/6，给显式 entry 数组
  // —— `BindGroupEntries::sequential` 是按位置 = 绑定号，无法表达"从 4 开始"。
  // binding 4 绑的是**降噪后**的那张（atrous 链的最终输出，`gi_dn[3]`），语义与原始 `gi_out` 完全
  // 一致（rgb = gi·valid、a = valid）；原始 `gi_out` 仍在（`gi_main` 写、时域读），保留作对照。
  // （覆盖度 cov 不占绑定：`dda_main` 从 `gi_tex.a` 自取，见 `bindings.wesl` 的说明。）
  // binding 6 = 降噪导引（与 gi_main 写的是同一块 buffer）⇒ `dda_main` 能做**几何感知上采样**：
  // 只接受与命中面同平面的 GI texel，棱边/墙角不渗色。
  let gi_read_layout = pipeline_cache.get_bind_group_layout(&pipelines.gi_read_layout);
  // binding 4 绑哪张，取决于「降噪质量」档（`DenoisePlan.on`）：
  //   · 档 0（完全不降噪）⇒ 直接绑**原始** `gi_out`（没有任何降噪 pass 会写 `gi_dn`，绑它会读到上一帧的陈旧值）；
  //   · 其余档 ⇒ 绑 atrous 链的最终输出 `gi_dn[3]`（语义与 `gi_out` 完全一致：rgb = gi·valid、a = valid）。
  let den_plan = match tune.gi.as_ref() {
    Some(g) => g.denoise_plan(),
    None => crate::gi::GiSettings::default().denoise_plan(),
  };
  let gi_den_view = if den_plan.on {
    beam_cache.gi_dn_src[3].as_ref().expect("降噪输出视图未创建").clone()
  } else {
    gi_view.clone()
  };
  let gi_guide = beam_cache.gi_guide.as_ref().expect("降噪导引 buffer 未创建").clone();
  // binding 8 = 帧内逐面去重表（`dda_main` 查表 / `dda_face_main` 写着色）；与本函数上方的 GI 资源同帧创建。
  let face_slots = beam_cache.face_slots.as_ref().expect("逐面去重表 buffer 未创建").clone();
  let gi_read_bg = render_device.create_bind_group(
    None,
    &gi_read_layout,
    &[
      BindGroupEntry { binding: 4, resource: BindingResource::TextureView(&gi_den_view) },
      BindGroupEntry { binding: 6, resource: gi_guide.as_entire_binding() },
      BindGroupEntry { binding: 8, resource: face_slots.as_entire_binding() },
    ],
  );
  beam_cache.gi_read_bg = Some(gi_read_bg);

  // ---- ① 逐面合并的 bind group（`gi_face_flatten`，layout 只有 group(0) 三个绑定）----
  // 10 = 导引（读键与法线）、18 = 逐面去重表（读 `gi_main` 累加进去的那对字段）、
  // 19 = GI 写入侧（把面均值写回）。**不绑 GI 的采样视图**：本 pass 不采样它
  // （同一 pass 内同一张纹理既采样又写入是 wgpu 的硬错）。
  let gi_flatten_layout = pipeline_cache.get_bind_group_layout(&crate::gi::gi_flatten_layout());
  let gi_flatten_bg = render_device.create_bind_group(
    None,
    &gi_flatten_layout,
    &[
      BindGroupEntry { binding: 10, resource: gi_guide.as_entire_binding() },
      BindGroupEntry { binding: 18, resource: face_slots.as_entire_binding() },
      BindGroupEntry { binding: 19, resource: BindingResource::TextureView(&gi_view) },
    ],
  );
  beam_cache.gi_flatten_bg = Some(gi_flatten_bg);

  // ---- 降噪的小配置（`@group(0) @binding(20)`，1 个 word = atrous 核半径）----
  // 降噪两个 pass 的 layout 只有 group(0)（刻意：不拉进 brickmap / 相机矩阵），而 bind group
  // 必须从索引 0 起成前缀设置 ⇒ 它们够不到 `@group(4)` 的 `gi_u`，只能用这个 4 字节的小 buffer。
  // 数值按菜单「渲染/RESTIR GI/降噪质量」的档位给（`DenoisePlan::radius`）：关/低 = 3×3（8 tap）、
  // 中/高 = 5×5（24 tap）；权威值在 WESL（`GI_DEN_ATROUS_R` / `_FAST`），Rust 只决定取哪一档。
  let den_r = den_plan.radius;
  if beam_cache.den_cfg.is_none() {
    beam_cache.den_cfg = Some(render_device.create_buffer(&BufferDescriptor {
      label: Some("gate_gi_den_cfg"),
      size: 16,
      usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
      mapped_at_creation: false,
    }));
  }
  if beam_cache.den_cfg_r != den_r {
    beam_cache.den_cfg_r = den_r;
    let cfg = beam_cache.den_cfg.as_ref().expect("刚创建");
    queue.write_buffer(cfg, 0, &den_r.to_le_bytes());
    bevy::log::debug!(
      target: "gate",
      "GI 降噪档位 → {}（atrous 核半径 {}，每轮 {} tap）",
      if den_plan.on {
        format!("{} 轮 atrous", den_plan.rounds)
      } else {
        "关（直接采样原始 GI）".to_string()
      },
      den_r,
      (2 * den_r + 1) * (2 * den_r + 1) - 1,
    );
  }
  let den_cfg = beam_cache.den_cfg.as_ref().expect("刚创建").clone();

  // ---- GI 降噪的 bind group：时域 1 个 + atrous 5 个（src→dst 组合，见 `DEN_ATROUS_CHAINS`）----
  // 每帧重建（纹理/buffer 都是持久句柄，只是换绑）；`den_flip` 决定历史哪块是「上帧读」。
  // 只在真正会跑降噪时翻转 `den_flip`（与 `res_flip` 同一套语义）。
  let den_runs = tune.gi.as_ref().is_some_and(|g| g.enabled);
  let (prev_i, cur_i) = if beam_cache.den_flip { (1usize, 0usize) } else { (0usize, 1usize) };
  {
    let guide = beam_cache.gi_guide.as_ref().expect("导引 buffer 未创建").clone();
    let phi = beam_cache.gi_phi.as_ref().expect("φ buffer 未创建").clone();
    let hist_prev = beam_cache.gi_hist[prev_i].as_ref().expect("历史 buffer 未创建").clone();
    let hist_cur = beam_cache.gi_hist[cur_i].as_ref().expect("历史 buffer 未创建").clone();
    let dn_src: Vec<TextureView> =
      beam_cache.gi_dn_src.iter().map(|v| v.as_ref().expect("降噪纹理未创建").clone()).collect();
    let dn_dst: Vec<TextureView> =
      beam_cache.gi_dn_dst.iter().map(|v| v.as_ref().expect("降噪纹理未创建").clone()).collect();
    let temporal_layout =
      pipeline_cache.get_bind_group_layout(&crate::gi::gi_den_temporal_layout());
    let atrous_layout = pipeline_cache.get_bind_group_layout(&crate::gi::gi_den_atrous_layout());
    beam_cache.den_bg[0] = Some(render_device.create_bind_group(
      None,
      &temporal_layout,
      &[
        BindGroupEntry { binding: 10, resource: guide.as_entire_binding() },
        BindGroupEntry { binding: 11, resource: BindingResource::TextureView(&gi_view) },
        BindGroupEntry { binding: 12, resource: hist_prev.as_entire_binding() },
        BindGroupEntry { binding: 13, resource: hist_cur.as_entire_binding() },
        BindGroupEntry { binding: 16, resource: BindingResource::TextureView(&dn_dst[0]) },
        BindGroupEntry { binding: 17, resource: phi.as_entire_binding() },
        BindGroupEntry { binding: 20, resource: den_cfg.as_entire_binding() },
      ],
    ));
    for (k, [s, d]) in DEN_ATROUS_CHAINS.iter().enumerate() {
      beam_cache.den_bg[k + 1] = Some(render_device.create_bind_group(
        None,
        &atrous_layout,
        &[
          BindGroupEntry { binding: 10, resource: guide.as_entire_binding() },
          // 13 = 本帧历史：atrous 只读其中的 M（短历史稳定化按 M 缩放步长）。必须挂**本帧**
          // 那份（时域 pass 刚写的），下一帧才轮到它变成 `hist_prev`。
          BindGroupEntry { binding: 13, resource: hist_cur.as_entire_binding() },
          BindGroupEntry { binding: 14, resource: BindingResource::TextureView(&dn_src[*s]) },
          BindGroupEntry { binding: 15, resource: BindingResource::TextureView(&dn_dst[*d]) },
          BindGroupEntry { binding: 17, resource: phi.as_entire_binding() },
          BindGroupEntry { binding: 20, resource: den_cfg.as_entire_binding() },
        ],
      ));
    }
  }
  if den_runs {
    beam_cache.den_flip = !beam_cache.den_flip;
  }
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
  // MT2-2：binding 6/7 = PBR 贴图数组。`PbrTextureSet` 是 main world 资源（`ExtractResourcePlugin`
  // 拷进 render world），它的两张图要等 `GpuImage` 就绪；**任一环节缺失都退化为占位视图**（1×1×1 的
  // `texture_2d_array`）⇒ 绑定永远成立、不 panic（贴图缺失时画面只是纯色回退）。
  let pbr_albedo = pbr_set.as_deref().and_then(|s| gpu_images.get(s.albedo_rough()));
  let pbr_metal = pbr_set.as_deref().and_then(|s| gpu_images.get(s.metal()));
  if pbr_albedo.is_none() || pbr_metal.is_none() {
    bevy::log::info_once!(
      target: "gate",
      "DDA prepare: PBR 贴图数组未就绪（贴图集或 GpuImage）⇒ BG1 binding 6/7 先绑 1×1×1 占位\
       （资产表 binding 5 若也未上传就是零初始化占位，长度已够）；不 panic"
    );
  }
  let pbr_albedo_view =
    pbr_albedo.map_or_else(|| gpu.pbr_albedo_rough_view.clone(), |i| i.texture_view.clone());
  let pbr_metal_view =
    pbr_metal.map_or_else(|| gpu.pbr_metal_view.clone(), |i| i.texture_view.clone());
  let bg1 = render_device.create_bind_group(
    None,
    &bg1_layout,
    &BindGroupEntries::sequential((
      gpu.struct_buf.as_entire_binding(),
      gpu.leaves.as_entire_binding(),
      gpu.palette.as_entire_binding(),
      globals_bind,
      &gpu.light_sampler,
      // @binding(5)：全局材质资产表（内容由 `prepare` 全量写一次）
      gpu.material_assets.as_entire_binding(),
      // @binding(6)/(7)：PBR 贴图数组（贴图集 / `GpuImage` 未就绪时是 1×1×1 占位视图）
      &pbr_albedo_view,
      &pbr_metal_view,
      // @binding(8)：PBR 专用采样器（MT2-3）——`init_empty_gpu` 建一次、占位与真身共用同一个实例，
      // 不存在"资源未就绪"的状态（采样器与它所采样的纹理无关）。
      &gpu.pbr_sampler,
      // @binding(9)：叶级 LOD 诊断计数器（M0）：固定 3 字的 buffer，`trace.wesl::LOD_DIAG` 关闭时
      // 无人写（shader 侧整段被折叠）⇒ 恒 0，读回侧也整个不注册。
      gpu.lod_diag.as_entire_binding(),
      // @binding(10)：M4 请求环缓冲：`trace.wesl::REQ_ENABLE` 关闭时无人写（shader 整段折叠）。
      gpu.lod_req.as_entire_binding(),
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
  // 镜面档位 + 嵌套层级（菜单「渲染/反射」）：`build_light_pool` 只认主题资产、把这两格留 0，
  // 真正的值在这里从菜单资源覆写（与 `LightGlobals` 那两格的文档一致）。
  lp.0.get_mut().g.refl_tier = tune.refl.as_ref().map_or(0, |r| r.tier());
  lp.0.get_mut().g.refl_nest = tune.refl.as_ref().map_or(0, |r| r.nest());
  // 「基础」开关位（菜单「渲染/基础」）：同上，`build_light_pool` 只负责留 0。
  // 资源缺失时给"两个都开"的缺省位（= 不改变观感），**不能给 0** —— 0 会静默关掉阴影与原色直出。
  lp.0.get_mut().g.base_flags =
    tune.base.as_ref().map_or(crate::lighting::BaseSettings::default().flags(), |b| b.flags());
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
  bg4: Option<Res<crate::gi::GiBg4>>,
  bg5: Option<Res<crate::gi::GiBg5>>,
  eye: Option<Res<EyeAdaptGpu>>,
  gi: Option<Res<crate::gi::GiSettings>>,
  gi_gpu: Option<Res<crate::gi::GiGpu>>,
  fog: crate::volumetric::FogRes,
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

  // 光柱（godray）的采样侧（group(6)）：**恒有** —— `prepare_fog` 每帧都建这一组
  // （关掉雾时绑的是 1×1 零纹理 ⇒ 主 pass 那一项恒 +0）。
  // 主 pass 的 layout 里有 group(6)，而逐面两个 pass 与它共用同一份 layout（`dda_layouts_face`）
  // ⇒ 三个 dispatch 都得设这一组（那两个入口不引用它，但 layout 里有 ⇒ 不设会被 wgpu 判成缺绑定）。
  let fog_read = fog.read_bg.as_ref().and_then(|b| b.0.as_ref());

  let gx = scale.size.x.div_ceil(DDA_WORKGROUP_SIZE);
  let gy = scale.size.y.div_ceil(DDA_WORKGROUP_SIZE);
  // beam：低分辨率 dispatch = ceil(size / 4) / 8
  let bx = scale.size.x.div_ceil(4).div_ceil(crate::brickmap::consts::WORKGROUP_SIZE);
  let by = scale.size.y.div_ceil(4).div_ceil(crate::brickmap::consts::WORKGROUP_SIZE);

  // ---- beam 预 pass：低分辨率 trace 只输出最近命中 t（独立 compute pass，
  // beam 写 beam_depth，主 pass 读同 texture → pass 边界 barrier 保证可见性）----
  // DDA_BEAM = false：跳过 beam pass，主 pass t_min=0（穿墙定位用）
  if DDA_BEAM && let Some(beam_pipe) = beam_pipe {
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

  // ---- GI（菜单「渲染/GI」开关；分辨率 = `GiSettings.gi_div`）：排在主 pass 之前（主 pass 采样输出）----
  // 派发规模 = `aux.gi_size`（= 渲染分辨率 ÷ gi_div）⇒ 分辨率档只差这里与资源尺寸。
  if gi.as_ref().is_some_and(|g| g.enabled)
    && let Some(aux) = aux.as_ref()
    && let Some(gi_bg0) = aux.gi_bg0.as_ref()
    && let Some(bg5) = bg5.as_ref()
    && let Some(gi_pipe) = pipeline_cache.get_compute_pipeline(pipelines.gi_pipeline)
  {
    let gx = aux.gi_size.x.div_ceil(DDA_WORKGROUP_SIZE);
    let gy = aux.gi_size.y.div_ceil(DDA_WORKGROUP_SIZE);
    // 逐面去重表**每帧整块清空**（`FACE_W_FLAG == 0` = 空槽）：清空必须排在 `gi_main` 之前，
    // 否则上一帧的认领会把本轮同槽的新键挡在门外（撞键只会少赚，但残留表会让收益归零）。
    // 二次顶点缓存（`gi_sec_slots`）**不清空**：它靠槽里键的 epoch 掩码自失效（uniform `gi_u.seq.y`，
    // 由 `prepare_gi` 逐项比对光照/几何的输入后自增）⇒ 省掉每帧那份 clear 带宽
    // （2K + 1/4 档那张表约 42MB/帧），见 `gi/common.wesl` 的「跨帧持久」段。
    if let Some(fs) = aux.face_slots_buffer() {
      ctx.command_encoder().clear_buffer(fs, 0, None);
    }
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

  // ---- ① 逐面合并的落地 pass（`gi_face_flatten`）----
  // 位置：`gi_main` 之后、降噪链**之前**（时域/atrous 的输入因此是"整个面一个值"）。
  // **与「降噪质量」档无关**：档 0 也跑（它改的是 GI 自身的输入，不属于降噪链）。
  // 它只读导引与逐面表、写 GI 纹理（不采样 GI）⇒ 单独一个只有 group(0) 的 layout。
  if gi.as_ref().is_some_and(|g| g.enabled)
    && let Some(aux) = aux.as_ref()
    && let Some(gi_gpu) = gi_gpu.as_ref()
    && let Some(bg) = aux.gi_flatten_bg.as_ref()
    && let Some(pipe) =
      gi_gpu.flatten_pipeline.and_then(|id| pipeline_cache.get_compute_pipeline(id))
  {
    let gx = aux.gi_size.x.div_ceil(DDA_WORKGROUP_SIZE);
    let gy = aux.gi_size.y.div_ceil(DDA_WORKGROUP_SIZE);
    crate::profiler::gpu_compute_pass(
      &mut profiler,
      ctx.command_encoder(),
      "gate_gi_face_flatten",
      |pass| {
        pass.set_pipeline(pipe);
        pass.set_bind_group(0, bg, &[]);
        pass.dispatch_workgroups(gx, gy, 1);
      },
    );
  }

  // ---- GI 降噪（时域 → 迭代 atrous）：必须紧跟 `gi_main`、排在主 pass 之前 ----
  // `dda_main` 的 group(5) binding 4 绑的就是这条链的最终输出（`gi_dn[3]`）。
  // 「降噪质量」档 0（关）⇒ 一轮都不派发：`dda_main` 那边绑的是原始 `gi_out`（见上面 `den_plan`）。
  if let Some(cur) = gi.as_ref()
    && cur.enabled
    && let Some(aux) = aux.as_ref()
    && let Some(gi_gpu) = gi_gpu.as_ref()
  {
    let plan = cur.denoise_plan();
    let n = plan.rounds.clamp(0, 5) as usize;
    if n > 0 {
      let gx = aux.gi_size.x.div_ceil(DDA_WORKGROUP_SIZE);
      let gy = aux.gi_size.y.div_ceil(DDA_WORKGROUP_SIZE);
      let rounds = &DEN_ATROUS_ROUNDS[n - 1];
      if let Some(bg) = aux.den_bg[0].as_ref()
        && let Some(pipe) =
          gi_gpu.den_pipelines[0].and_then(|id| pipeline_cache.get_compute_pipeline(id))
      {
        crate::profiler::gpu_compute_pass(
          &mut profiler,
          ctx.command_encoder(),
          "gate_gi_denoise_temporal",
          |pass| {
            pass.set_pipeline(pipe);
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(gx, gy, 1);
          },
        );
      }
      for i in 0..n {
        let Some(bg) = aux.den_bg[rounds[i] + 1].as_ref() else {
          continue;
        };
        let Some(pipe) =
          gi_gpu.den_pipelines[i + 1].and_then(|id| pipeline_cache.get_compute_pipeline(id))
        else {
          continue;
        };
        let label = [
          "gate_gi_denoise_atrous1",
          "gate_gi_denoise_atrous2",
          "gate_gi_denoise_atrous4",
          "gate_gi_denoise_atrous8",
          "gate_gi_denoise_atrous16",
        ][i];
        crate::profiler::gpu_compute_pass(&mut profiler, ctx.command_encoder(), label, |pass| {
          pass.set_pipeline(pipe);
          pass.set_bind_group(0, bg, &[]);
          pass.dispatch_workgroups(gx, gy, 1);
        });
      }
    }
  }

  // ---- 逐面 pass 两段：`dda_face_accum`（把每个面的全部 texel 的 GI 累加进槽）→
  //      `dda_face_main`（逐面算一次着色并写回槽）----
  // 位置：GI 之后（认领发生在 `gi_main`；两段都读降噪后的 GI）、主 pass 之前（主 pass 查表）。
  // 两段的 bind group 与主 pass 逐项相同（同一份 layout）⇒ 复用同一组对象，不额外建绑定。
  if gi.as_ref().is_some_and(|g| g.enabled)
    && let Some(aux) = aux.as_ref()
  {
    let gx = aux.gi_size.x.div_ceil(DDA_WORKGROUP_SIZE);
    let gy = aux.gi_size.y.div_ceil(DDA_WORKGROUP_SIZE);
    if let Some(accum_pipe) = pipeline_cache.get_compute_pipeline(pipelines.face_accum_pipeline) {
      crate::profiler::gpu_compute_pass(
        &mut profiler,
        ctx.command_encoder(),
        "gate_dda_face_accum",
        |pass| {
          pass.set_pipeline(accum_pipe);
          pass.set_bind_group(0, &bg0.0, &[]);
          pass.set_bind_group(1, &bg1.0, &[]);
          pass.set_bind_group(2, &bg2.0, &[]);
          pass.set_bind_group(3, &bg3.0, &[]);
          pass.set_bind_group(4, &bg4.0, &[]);
          if let Some(gi_read) = aux.gi_read_bg.as_ref() {
            pass.set_bind_group(5, gi_read, &[]);
          }
          if let Some(fog_read) = fog_read {
            pass.set_bind_group(6, fog_read, &[]);
          }
          pass.dispatch_workgroups(gx, gy, 1);
        },
      );
    }
    if let Some(face_pipe) = pipeline_cache.get_compute_pipeline(pipelines.face_pipeline) {
      crate::profiler::gpu_compute_pass(
        &mut profiler,
        ctx.command_encoder(),
        "gate_dda_face",
        |pass| {
          pass.set_pipeline(face_pipe);
          pass.set_bind_group(0, &bg0.0, &[]);
          pass.set_bind_group(1, &bg1.0, &[]);
          pass.set_bind_group(2, &bg2.0, &[]);
          pass.set_bind_group(3, &bg3.0, &[]);
          pass.set_bind_group(4, &bg4.0, &[]);
          if let Some(gi_read) = aux.gi_read_bg.as_ref() {
            pass.set_bind_group(5, gi_read, &[]);
          }
          if let Some(fog_read) = fog_read {
            pass.set_bind_group(6, fog_read, &[]);
          }
          pass.dispatch_workgroups(gx, gy, 1);
        },
      );
    }
  }

  // ---- 光柱（godray，`volumetric.rs`）：必须排在主 pass **之前**（主 pass 采样它的结果）----
  // 依赖只有 beam 预 pass（掩码那条主射线的 t_min 来源，已在最前跑过）；与 GI / 逐面两条链
  // **互不依赖**（各自的网格、资源、开关）。关掉时（`FogSettings.enabled = false`）一条 dispatch 都不发。
  // group(0) 复用 GI 那份 `gi_bg0`（view uniform + beam depth）、group(1..3) 复用主 pass 的绑定。
  crate::volumetric::dispatch_fog(
    &mut profiler,
    ctx.command_encoder(),
    fog.gpu.as_deref(),
    fog.settings.as_deref(),
    &pipeline_cache,
    aux.as_ref().and_then(|a| a.gi_bg0()),
    Some(&bg1.0),
    Some(&bg2.0),
    Some(&bg3.0),
  );

  // ---- 主 DDA pass：trace + unlit 着色直出 ----
  // group(6)（光柱的读侧）在上面就取好了；缺它只可能是首帧（`prepare_fog` 还没跑）⇒ 跳过主 pass。
  if fog_read.is_none() {
    bevy::log::debug_once!("DDA dispatch: 光柱 group(6) 未就绪（本帧跳过主 pass）");
    return;
  }
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
        // GI 的采样侧（layout 里的 group(5)）：没有它 `dda_main` 无法 dispatch。
        // 纹理未就绪（`prepare_dda_bind_groups` 本帧提前返回）时退化为不绑。
        if let Some(gi_read) = aux.as_ref().and_then(|a| a.gi_read_bg.as_ref()) {
          pass.set_bind_group(5, gi_read, &[]);
        }
        // 光柱的采样侧（group(6)）：`dda_main` 在曝光之前把它加到颜色上。
        if let Some(fog_read) = fog_read {
          pass.set_bind_group(6, fog_read, &[]);
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
}

#[cfg_attr(not(feature = "profile"), allow(unused_variables, unused_mut))]
fn blit_dda_view(
  mut ctx: RenderContext,
  views: Query<&ViewTarget>,
  blit_bg: Option<Res<DdaBlitBindGroup>>,
  post: Option<Res<PostFxSettings>>,
  scale: Option<Res<RenderScale>>,
  pipeline_cache: Res<PipelineCache>,
  pipelines: Res<DdaPipelines>,
  mut profiler: ResMut<crate::profiler::GpuProfilerRes>,
) {
  let (Some(bg), Ok(target)) = (blit_bg.as_ref(), views.single()) else {
    bevy::log::debug_once!("DDA blit: bg or ViewTarget missing");
    return;
  };
  // 抗锯齿 = 换一条 fragment 入口（`fs_fxaa`），bind group 完全相同 ⇒ 开关只在这里分支。
  // **只在像素大小 = 1 时生效**（`RenderScale.factor == 1`）：降采样档本身就是"大粒像素"的观感，
  // FXAA 会把整数块边界抹糊，与那些档位的意图相反 ⇒ 一律走纯 blit。
  // 菜单上「抗锯齿」那一行此时是**禁用态**（`debug_menu.rs::sync_video_menu`），但**开关值原样保留**
  // —— 它记的是"像素大小 = 1 时要不要抗锯齿"，切回 `1` 就立刻生效。
  // `RenderScale` 缺失（正常不该发生）⇒ 退回只看开关的旧行为，不静默关掉抗锯齿。
  let downscaled = scale.as_ref().is_some_and(|s| s.factor != 1);
  let id = if post.as_ref().is_some_and(|p| p.fxaa) && !downscaled {
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
