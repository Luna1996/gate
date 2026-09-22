//! 光柱（godray）：`volumetric.wesl` 的驱动侧。
//!
//! 算法、成本模型与取舍写在 WESL 文件头，这里只说**落点与接线**：
//!   · `godray_main`（**掩码**）：网格 = 渲染分辨率 ÷ `FogSettings.div`，每 texel **一条主射线**
//!     ⇒ 「这个方向看得见天空吗」× 天体方向权重 × 主光色 × 强度 ⇒ 写中间靶 **A**；
//!   · `godray_blur_a` / `godray_blur_b`（**径向模糊**两趟）：读 A 写 B、再读 B 写 A
//!     ⇒ A 里是最终结果（新 pass，但**复用同一份 layout 与同一个 bind group**）；
//!   · `dda_main` 在**曝光之前**把它**加到**颜色上（纯加法：不衰减背景、不加任何底色）。
//!
//! 三步都是**确定性**取值（无随机采样、不进时域）⇒ 没有噪声、没有滞后；
//! **天体**（白天 = 太阳、夜间 = 月亮，见 `sky.rs`）的屏幕坐标在这里算
//! （投影 `view_proj`，见 [`project_sun`]），当它转出画面/转到背后时 `朝向因子` 平滑淡出到 0，
//! 不会突然消失。
//!
//! 与 GI 的关系：**两条链完全独立**（各自的网格尺寸、资源、开关），唯一的共享是主 pass 的
//! `group(0..3)` 绑定（掩码 pass 要相机矩阵与 brickmap 做那条主射线）。

use bevy::render::render_resource::{BindGroupLayoutDescriptor, CachedComputePipelineId, ShaderType};
use glam::{Mat4, UVec2, Vec2, Vec4};

use crate::brickmap::dda::DDA_WORKGROUP_SIZE;

/// 光柱的参数 uniform（WESL `bindings.wesl` 的 `FogUniform` 逐字段镜像，字节一致）。
#[repr(C)]
#[derive(Debug, Default, Clone, Copy, ShaderType)]
pub struct FogUniform {
  /// x = **衰减**（模糊每步的权重乘它：`< 1` ⇒ 光柱离太阳越远越淡）、yzw 保留（恒 0）。
  pub params: Vec4,
  /// x = 保留（恒 0：网格分辨率除数已固定为 [`FOG_DIV`]，不再是档位）、
  /// y = **集中度**（太阳方向权重的指数）、z = **强度**、w = **朝向因子**。
  pub misc: Vec4,
  /// x/y 保留（恒 0）、z = **天体盘角径**（rad）、w = **光晕强度**（两项都只喂 `sky_primary`，
  /// 面板在「渲染/天空/天体盘」，见 [`FogSettings::sun_cone`] / [`FogSettings::halo`]）。
  pub misc2: Vec4,
  /// xy = **天体的屏幕坐标**（0..1；可能落在画面外 —— 模糊的取样方向就是它）、zw 保留（恒 0）。
  pub misc3: Vec4,
}

/// 径向模糊（光柱）的档位（菜单「渲染/天空/径向模糊」+「渲染/天空/天体盘」）。
///
/// **这五个量默认由时间曲线给**（`gate-render/src/sky.rs` 的关键帧表每帧写进来）；
/// 面板上每组各有一个「覆写」开关：打开后本资源的值不再被覆写、由面板滑杆直接决定。
/// 于是本结构既是"时间算出来的当前值"，也是"覆写值"（覆写开着时它才是真值）。
#[derive(bevy::ecs::resource::Resource, Clone, Copy, Debug, PartialEq)]
pub struct FogSettings {
  /// 总开关（面板上的「开启」）：关掉 ⇒ 三个 pass 一个都不派发，`dda_main` 的 group(6) 绑
  /// 1×1 零纹理（零成本）。**不受时间驱动**（它是"画不画光柱"，不是光柱的参数）。
  pub enabled: bool,
  /// **强度**：乘在光柱上。它是最主要的旋钮（"光柱有多亮"）。
  pub strength: f32,
  /// **衰减**：径向模糊每步的权重乘它（`< 1`）。越接近 1 ⇒ 光柱拖得越长（几乎不衰减、能到整屏）、
  /// 越小 ⇒ 光柱越贴紧太阳。
  pub decay: f32,
  /// **集中度**：太阳方向权重的指数 `pow(dot(视线, 太阳方向), 集中度)`。
  /// `1` ⇒ 整个天空都在发光（一层方向性眩光，不像光柱）；`20` ⇒ 太阳周围约十几度内
  /// （`10° ⇒ 0.72`、`20° ⇒ 0.33`）；再大 ⇒ 更尖锐（像激光束）。
  pub focus: f32,
  /// 天体的**角径**（rad）：天空里那个盘的半径（`sky_primary`）—— 它同时是盘的软边缘与
  /// **光晕尺度**的基准（晕铺到角径的 `HALO_T` 倍）。面板给度、这里存弧度。
  pub sun_cone: f32,
  /// **光晕强度**：天体盘外那圈解析拖尾的系数（`sky_primary` 的 halo），随角径一起缩放。
  pub halo: f32,
}

/// 光柱网格的分辨率除数（**必须与 WESL `volumetric.wesl::FOG_DIV` 一致**）。
/// 固定 1/4、不做成档位：掩码之后还有一趟横跨整屏的径向模糊，**最终画面的柔和程度由模糊长度
/// 决定、不由掩码分辨率决定** ⇒ 1/2 与 1/1 在观感上几乎无差别，而代价按网格像素数涨
/// （1/1 的主射线数量是 1/4 的 16 倍）。掩码 pass 是这条链唯一按像素计费的一步。
pub const FOG_DIV: u32 = 4;

impl FogSettings {
  /// 生效的强度（负值当成 0）。
  pub fn strength(&self) -> f32 {
    self.strength.max(0.0)
  }

  /// 生效的衰减（钳在 `0.05..=1`；1 = 不衰减）。
  pub fn decay(&self) -> f32 {
    self.decay.clamp(0.05, 1.0)
  }

  /// 生效的集中度（下限 1 = 不加权；上限 64 防止 `pow` 退化成阶跃）。
  pub fn focus(&self) -> f32 {
    self.focus.clamp(1.0, 64.0)
  }

  /// 光柱网格尺寸 = 渲染分辨率 ÷ [`FOG_DIV`]（逐轴向下取整，至少 1×1）。
  pub fn grid_size(&self, render_size: UVec2) -> UVec2 {
    UVec2::new(
      (render_size.x / FOG_DIV).max(1),
      (render_size.y / FOG_DIV).max(1),
    )
  }
}

impl Default for FogSettings {
  /// 缺省 = 菜单「渲染/天空/径向模糊」+「渲染/天空/天体盘」面板上那一组（**覆写值**；
  /// 覆写关着时这些字段每帧被 `sky.rs` 的时间曲线覆写，见结构体说明）。
  /// 面板初值（`assets/ui/debug_menu.toml`）必须与这里逐项一致。
  ///
  /// 强度 0.5 / 衰减 0.6 / 集中度 20 的观感是「**紧贴太阳、克制的一束**」：
  ///   · `集中度 20`：只有太阳周围约十几度内的天空参与（`10° ⇒ 0.72`、`20° ⇒ 0.33`、`45° ⇒ 0.006`）
  ///     ⇒ 光柱从太阳那一点射出来，画面其余部分完全不受影响；
  ///   · `衰减 0.6`：模糊每步权重掉得快 ⇒ 光柱短、贴紧太阳（像镜头眩光），不拖到整屏；
  ///   · `强度 0.5`：叠加量约为天空中亮度的三到五成 ⇒ 逆光时清楚、正常视角几乎不打扰。
  /// 想要"拉长、铺满画面"的光柱就把衰减调到 `0.92~0.98`；想要更含蓄就把强度压到 `0.2~0.3`。
  fn default() -> Self {
    Self {
      enabled: true,
      strength: 0.5,
      decay: 0.6,
      focus: 20.0,
      // 角径取"夸张尺寸"（真实太阳只有 0.27°）：`0.27°` 在 1080p 下约 5 像素，
      // 盘的软边缘与光晕几乎没有像素可用（见 `sky.rs` 文件头的取舍）。
      sun_cone: 1.2_f32.to_radians(),
      halo: 0.5,
    }
  }
}

// ============================================================================
// bind group layout
// ============================================================================

/// 光柱两步链的 layout：**三个 pass 各一份**。
/// 为什么不能共用一份：wgpu 的用法跟踪是**按 dispatch** 算的，而 `STORAGE_WRITE_ONLY` 与
/// `RESOURCE`（采样）互斥 —— 一份带全部号的 layout 会让「掩码写 A」同时"采样 A"（A 也出现在
/// 采样号上）而被判成冲突。所以每个 pass 只带它**真正用到**的那几个号。
/// 掩码 pass 把它当 **group(4)**（0..3 被 view/beam + brickmap 占满），两个模糊 pass 把它当
/// **group(0)**（它们只用这一组）⇒ 组号由 pipeline layout 数组的下标给。
/// 声明与访问模式必须与 `bindings.wesl` 逐字一致。
fn fog_layout(name: &'static str, slots: &[(u32, FogSlot)]) -> BindGroupLayoutDescriptor {
  use bevy::render::render_resource::*;
  const C: ShaderStages = ShaderStages::COMPUTE;
  let entries: Vec<BindGroupLayoutEntry> = slots
    .iter()
    .map(|(binding, slot)| BindGroupLayoutEntry {
      binding: *binding,
      visibility: C,
      ty: match slot {
        FogSlot::Uniform => BindingType::Buffer {
          ty: BufferBindingType::Uniform,
          has_dynamic_offset: false,
          min_binding_size: Some(FogUniform::min_size()),
        },
        // 采样侧取 `filterable: false`：这些图**只用 `textureLoad`**（手写双线性，见 WESL），
        // 而 `Rgba16Float` 在有的后端不一定可过滤 —— 声明成不可过滤最稳。
        FogSlot::Sample => BindingType::Texture {
          sample_type: TextureSampleType::Float { filterable: false },
          view_dimension: TextureViewDimension::D2,
          multisampled: false,
        },
        FogSlot::Write => BindingType::StorageTexture {
          access: StorageTextureAccess::WriteOnly,
          format: TextureFormat::Rgba16Float,
          view_dimension: TextureViewDimension::D2,
        },
      },
      count: None,
    })
    .collect();
  BindGroupLayoutDescriptor::new(name, &entries)
}

/// `fog_layout` 的槽位种类。
#[derive(Clone, Copy)]
enum FogSlot {
  Uniform,
  Sample,
  Write,
}

/// 掩码 pass 的 group(4)：uniform + **写 A**（它不读任何纹理）。
pub fn fog_layout_mask() -> BindGroupLayoutDescriptor {
  fog_layout("FogGodrayMask", &[(1, FogSlot::Uniform), (3, FogSlot::Write)])
}

/// 模糊第一趟的 group(0)：uniform + **采样 A** + **写 B**。
/// 号取 21/22/25：模糊 pass 的 layout 只有 group(0) 一组，而 0..20 已被主 pass 的 BG0 与 GI 降噪
/// 占用（见 `bindings.wesl` 的说明）。
pub fn fog_layout_blur_a() -> BindGroupLayoutDescriptor {
  fog_layout(
    "FogGodrayBlurA",
    &[(21, FogSlot::Uniform), (22, FogSlot::Sample), (25, FogSlot::Write)],
  )
}

/// 模糊第二趟的 group(0)：uniform + **采样 B** + **写 A**（A 里是最终结果）。
pub fn fog_layout_blur_b() -> BindGroupLayoutDescriptor {
  fog_layout(
    "FogGodrayBlurB",
    &[(21, FogSlot::Uniform), (24, FogSlot::Sample), (23, FogSlot::Write)],
  )
}

/// `dda_main` 的 group(6)：参数 uniform(0) + 光柱的采样视图(1)。
/// 参数 uniform 必须给：`dda_main` 的天空背景要走 `sky_primary`（天体盘 + 光晕），
/// 而那条路径只有 group(6)（与 `@group(4) @binding(1)` 是同一块 buffer，见 `bindings.wesl`）。
/// 关掉光柱时绑 1×1 的零纹理（uniform 照绑：天体盘与开关无关）。
/// 采样侧同样取 `filterable: false`（`dda_main` 用手写双线性取值）。
pub fn fog_read_layout() -> BindGroupLayoutDescriptor {
  use bevy::render::render_resource::*;
  const C: ShaderStages = ShaderStages::COMPUTE;
  BindGroupLayoutDescriptor::new(
    "FogGodrayRead",
    &[
      BindGroupLayoutEntry {
        binding: 0,
        visibility: C,
        ty: BindingType::Buffer {
          ty: BufferBindingType::Uniform,
          has_dynamic_offset: false,
          min_binding_size: Some(FogUniform::min_size()),
        },
        count: None,
      },
      BindGroupLayoutEntry {
        binding: 1,
        visibility: C,
        ty: BindingType::Texture {
          sample_type: TextureSampleType::Float { filterable: false },
          view_dimension: TextureViewDimension::D2,
          multisampled: false,
        },
        count: None,
      },
    ],
  )
}

// ============================================================================
// GPU 状态与资源
// ============================================================================

/// 光柱的全部 GPU 状态：uniform + 两张中间靶（随分辨率重建）+ bind group + pipeline。
/// 资源**不放进** `AuxTexCache`（GI 的缓存）：光柱的尺寸、生命周期、开关都与 GI 无关，
/// 混在一处会让两条链互相牵制；`dda_main` 侧只需要一个 group(6) 的 bind group（本模块给出）。
#[derive(bevy::ecs::resource::Resource, Default)]
pub struct FogGpu {
  pub uniform: bevy::render::render_resource::UniformBuffer<FogUniform>,
  /// 上一帧生效的网格尺寸（重建判定）。
  size: UVec2,
  /// 中间靶 **A**：掩码写入 → 模糊 A 读 → **模糊 B 写**（最终结果，`dda_main` 采样它的采样视图）。
  tex_a: Option<bevy::render::render_resource::Texture>,
  a_src: Option<bevy::render::render_resource::TextureView>,
  a_dst: Option<bevy::render::render_resource::TextureView>,
  /// 中间靶 **B**：模糊 A 写 → 模糊 B 读。
  tex_b: Option<bevy::render::render_resource::Texture>,
  b_src: Option<bevy::render::render_resource::TextureView>,
  b_dst: Option<bevy::render::render_resource::TextureView>,
  /// 三个 pass 各自的 bind group（**不能共用**：同一个 pass 里一块纹理不能既当采样源又当写入目标，
  /// 见 `fog_layout` 的说明）：
  /// 掩码（group(4) 位置）只写 A；模糊 A（group(0)）读 A 写 B；模糊 B（group(0)）读 B 写 A。
  mask_bg: Option<bevy::render::render_resource::BindGroup>,
  blur_a_bg: Option<bevy::render::render_resource::BindGroup>,
  blur_b_bg: Option<bevy::render::render_resource::BindGroup>,
  /// `dda_main` 的 group(6)（**恒有**：关掉光柱时是占位版）。
  read_bg: Option<bevy::render::render_resource::BindGroup>,
  /// 占位（关掉光柱时 read_bg 用）：1×1 零纹理。
  ph_tex: Option<bevy::render::render_resource::Texture>,
  ph_view: Option<bevy::render::render_resource::TextureView>,
  /// `[0]` = 掩码、`[1]`/`[2]` = 模糊 A/B。
  pipelines: [Option<CachedComputePipelineId>; 3],
}

/// 占位资源（关掉光柱 / 首帧）：1×1 rgba16f 零纹理。
fn placeholder(
  device: &bevy::render::renderer::RenderDevice,
  gpu: &mut FogGpu,
) -> bevy::render::render_resource::TextureView {
  use bevy::render::render_resource::*;
  if gpu.ph_tex.is_none() {
    let t = device.create_texture(&TextureDescriptor {
      label: Some("gate_godray_placeholder"),
      size: Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
      mip_level_count: 1,
      sample_count: 1,
      dimension: TextureDimension::D2,
      format: TextureFormat::Rgba16Float,
      usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
      view_formats: &[],
    });
    gpu.ph_view = Some(t.create_view(&TextureViewDescriptor::default()));
    gpu.ph_tex = Some(t);
  }
  gpu.ph_view.as_ref().expect("刚创建").clone()
}

/// 太阳的**屏幕坐标**（0..1，可能落在画面外）与**朝向因子**（太阳在相机背后 ⇒ 0）。
/// 做法：把「相机位置沿太阳方向外推 `SUN_PROJECT_DIST`」的世界点投影到裁剪空间。
///   · `w <= 0`（点在相机平面之后 = 太阳在背后）⇒ 没有屏幕坐标可言 ⇒ 朝向因子给 0（效果关闭）；
///   · 落在画面外 ⇒ **夹到画面边缘附近**（`±0.25` 屏外）而不是直接消失 —— 光仍然从太阳那一侧射进来
///     （这是这类实现的经典处理；把方向钉在边上会让模糊方向退化，所以留一点余量）；
///   · 朝向因子按"太阳方向与相机朝向的夹角余弦"平滑淡出（`w / 外推距离` 就是这个余弦）：
///     画面正侧方（90°）为 0、往前就满强度 ⇒ 太阳在画面边缘/刚出画面时光柱照样在。
fn project_sun(view_proj: Mat4, cam_pos: glam::Vec3, to_sun: glam::Vec3) -> (Vec2, f32) {
  /// 外推距离（voxel）：只要求"远到投影稳定"，与场景尺度无关。
  const SUN_PROJECT_DIST: f32 = 4096.0;
  if to_sun.length_squared() < 1e-6 {
    return (Vec2::new(0.5, 0.5), 0.0);
  }
  let p = cam_pos + to_sun * SUN_PROJECT_DIST;
  let clip = view_proj * p.extend(1.0);
  if clip.w <= 1e-3 {
    return (Vec2::new(0.5, 0.5), 0.0);
  }
  let ndc = clip.truncate() / clip.w;
  let uv = Vec2::new(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
  let uv = Vec2::new(uv.x.clamp(-0.25, 1.25), uv.y.clamp(-0.25, 1.25));
  let t = (clip.w / SUN_PROJECT_DIST / 0.25).clamp(0.0, 1.0);
  (uv, t * t * (3.0 - 2.0 * t))
}

pub struct FogPlugin;

impl bevy::app::Plugin for FogPlugin {
  fn build(&self, app: &mut bevy::app::App) {
    use bevy::ecs::schedule::IntoScheduleConfigs;
    app.init_resource::<FogSettings>();
    let Some(render_app) = app.get_sub_app_mut(bevy::render::RenderApp) else {
      return;
    };
    render_app
      .init_resource::<FogSettings>()
      .init_resource::<FogGpu>()
      .add_systems(bevy::render::RenderStartup, init_fog_gpu)
      .add_systems(
        bevy::render::RenderStartup,
        queue_fog_pipelines.after(crate::brickmap::dda::init_dda_pipelines),
      )
      .add_systems(bevy::render::ExtractSchedule, extract_fog_settings)
      .add_systems(
        bevy::render::Render,
        prepare_fog.in_set(bevy::render::RenderSystems::PrepareBindGroups),
      );
  }
}

fn init_fog_gpu(mut commands: bevy::ecs::system::Commands) {
  let d = FogSettings::default();
  bevy::log::info!(
    target: "gate",
    "光柱（godray）：默认 {} / 1÷{} 网格（固定）/ 强度 {} / 衰减 {} / 集中度 {} / 天体盘角径 {}° / 光晕 {}",
    if d.enabled { "开" } else { "关" },
    FOG_DIV,
    d.strength(),
    d.decay(),
    d.focus(),
    d.sun_cone.to_degrees(),
    d.halo,
  );
  commands.insert_resource(FogGpu::default());
}

fn queue_fog_pipelines(
  pipeline_cache: bevy::ecs::system::Res<bevy::render::render_resource::PipelineCache>,
  dda_shader: bevy::ecs::system::Res<crate::shader::DdaShaderHandle>,
  dda: bevy::ecs::system::Res<crate::brickmap::dda::DdaPipelines>,
  mut gpu: bevy::ecs::system::ResMut<FogGpu>,
) {
  use bevy::render::render_resource::ComputePipelineDescriptor;
  use std::borrow::Cow;
  if gpu.pipelines[0].is_some() {
    return;
  }
  // 掩码 pass：要相机矩阵 + beam depth（group(0) 复用 GI 的瘦版）+ brickmap（做那条主射线）
  // + 光柱自己那一组（group(4)）。与主 pass 的 group(1..3) 逐字相同 ⇒ 直接复用它的 layout。
  gpu.pipelines[0] = Some(pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
    label: Some(Cow::from("gate_godray")),
    layout: vec![
      dda.bg0_gi_layout.clone(),
      dda.bg1_layout.clone(),
      dda.bg2_layout.clone(),
      dda.bg3_layout.clone(),
      fog_layout_mask(),
    ],
    shader: dda_shader.0.clone(),
    entry_point: Some(Cow::from("godray_main")),
    ..Default::default()
  }));
  // 两趟模糊：只用光柱那一组（当 group(0) 用）⇒ 各一份只含所需号的 layout。
  for (i, (entry, layout)) in [
    ("godray_blur_a", fog_layout_blur_a()),
    ("godray_blur_b", fog_layout_blur_b()),
  ]
  .into_iter()
  .enumerate()
  {
    gpu.pipelines[i + 1] = Some(pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
      label: Some(Cow::from(match i {
        0 => "gate_godray_blur_a",
        _ => "gate_godray_blur_b",
      })),
      layout: vec![layout],
      shader: dda_shader.0.clone(),
      entry_point: Some(Cow::from(entry)),
      ..Default::default()
    }));
  }
}

fn extract_fog_settings(
  mut commands: bevy::ecs::system::Commands,
  settings: Option<bevy::render::Extract<bevy::ecs::system::Res<FogSettings>>>,
) {
  commands.insert_resource(settings.map_or_else(FogSettings::default, |s| **s));
}

/// 每帧：写 uniform（含太阳的屏幕坐标）、必要时重建资源、建 bind group。
#[allow(clippy::too_many_arguments)] // Bevy render system：各资源逐一注入（与 `prepare_gi` 同一形态）
fn prepare_fog(
  mut commands: bevy::ecs::system::Commands,
  device: bevy::ecs::system::Res<bevy::render::renderer::RenderDevice>,
  queue: bevy::ecs::system::Res<bevy::render::renderer::RenderQueue>,
  pipeline_cache: bevy::ecs::system::Res<bevy::render::render_resource::PipelineCache>,
  settings: bevy::ecs::system::Res<FogSettings>,
  view: Option<bevy::ecs::system::Res<crate::brickmap::dda::DdaViewUniform>>,
  scale: Option<bevy::ecs::system::Res<crate::brickmap::dda::RenderScale>>,
  lighting: Option<bevy::ecs::system::Res<crate::lighting::LightingTheme>>,
  mut gpu: bevy::ecs::system::ResMut<FogGpu>,
) {
  use bevy::render::render_resource::*;
  // 太阳的**屏幕坐标**：投影「相机沿太阳方向外推」的世界点（见 `project_sun`）。
  // `DirLightCfg.dir` 是光的**传播方向**（指向场景）⇒ 指向太阳要取反（与 `lighting.rs` 同口径）。
  let sun = lighting
    .as_ref()
    .and_then(|l| l.sun.as_ref())
    .map(|s| -glam::Vec3::from_array(s.dir).normalize_or_zero())
    .unwrap_or_default();
  let (view_proj, cam_pos) = view.as_deref().map_or((Mat4::IDENTITY, glam::Vec3::ZERO), |v| {
    (v.view_proj, v.cam_pos_voxel.truncate())
  });
  let (sun_uv, facing) = project_sun(view_proj, cam_pos, sun);

  // ---- uniform ----
  let size = scale
    .as_deref()
    .map_or(crate::consts::VIEW_SIZE, |s| s.size);
  *gpu.uniform.get_mut() = FogUniform {
    params: Vec4::new(settings.decay(), 0.0, 0.0, 0.0),
    misc: Vec4::new(
      0.0,
      settings.focus(),
      settings.strength(),
      facing,
    ),
    // z 与 WESL `SUN_CONE_MAX` 的关系：只在 `sky_primary` 里用于太阳盘半径（会硬夹）。
    misc2: Vec4::new(0.0, 0.0, settings.sun_cone.max(0.0), settings.halo.max(0.0)),
    misc3: Vec4::new(sun_uv.x, sun_uv.y, 0.0, 0.0),
  };
  gpu.uniform.write_buffer(&device, &queue);

  // ---- 资源（随 分辨率 ÷ div 重建）----
  let grid = settings.grid_size(size);
  if gpu.tex_a.is_none() || gpu.size != grid {
    let make_tex = |label: &str| {
      device.create_texture(&TextureDescriptor {
        label: Some(label),
        size: Extent3d { width: grid.x, height: grid.y, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba16Float,
        // 既被 pass 写（storage），又要被模糊/主 pass 采样（texture binding）
        usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
      })
    };
    let a = make_tex("gate_godray_a");
    gpu.a_src = Some(a.create_view(&TextureViewDescriptor::default()));
    gpu.a_dst = Some(a.create_view(&TextureViewDescriptor::default()));
    gpu.tex_a = Some(a);
    let b = make_tex("gate_godray_b");
    gpu.b_src = Some(b.create_view(&TextureViewDescriptor::default()));
    gpu.b_dst = Some(b.create_view(&TextureViewDescriptor::default()));
    gpu.tex_b = Some(b);
    gpu.size = grid;
    bevy::log::debug!(
      target: "gate",
      "光柱资源 → {}×{}（渲染 {}÷{}）；两张中间靶 rgba16f",
      grid.x,
      grid.y,
      size.x,
      FOG_DIV,
    );
  }

  // ---- bind group（每帧重建；句柄都是 Arc，重建只是换绑）----
  let mask_layout = pipeline_cache.get_bind_group_layout(&fog_layout_mask());
  let blur_a_layout = pipeline_cache.get_bind_group_layout(&fog_layout_blur_a());
  let blur_b_layout = pipeline_cache.get_bind_group_layout(&fog_layout_blur_b());
  let read_layout = pipeline_cache.get_bind_group_layout(&fog_read_layout());
  let ph_view = placeholder(&device, &mut gpu);
  if let (Some(a_src), Some(a_dst), Some(b_src), Some(b_dst)) =
    (gpu.a_src.clone(), gpu.a_dst.clone(), gpu.b_src.clone(), gpu.b_dst.clone())
  {
    let uniform = gpu.uniform.binding().expect("uniform 已写入");
    // 掩码：只写 A（它不读任何纹理 ⇒ 不带采样号，否则 A 会同时被判成"被采样"）。
    let mask_bg = device.create_bind_group(
      None,
      &mask_layout,
      &[
        BindGroupEntry { binding: 1, resource: uniform.clone() },
        BindGroupEntry { binding: 3, resource: BindingResource::TextureView(&a_dst) },
      ],
    );
    // 模糊 A：读 A、写 B。
    let blur_a_bg = device.create_bind_group(
      None,
      &blur_a_layout,
      &[
        BindGroupEntry { binding: 21, resource: uniform.clone() },
        BindGroupEntry { binding: 22, resource: BindingResource::TextureView(&a_src) },
        BindGroupEntry { binding: 25, resource: BindingResource::TextureView(&b_dst) },
      ],
    );
    // 模糊 B：读 B、写 A（A 里是最终结果）。
    let blur_b_bg = device.create_bind_group(
      None,
      &blur_b_layout,
      &[
        BindGroupEntry { binding: 21, resource: uniform.clone() },
        BindGroupEntry { binding: 24, resource: BindingResource::TextureView(&b_src) },
        BindGroupEntry { binding: 23, resource: BindingResource::TextureView(&a_dst) },
      ],
    );
    // group(6)：主 pass 采样**中间靶 A**（模糊 B 的输出 = 最终结果）；
    // 关掉光柱时绑占位零纹理 —— 否则那一项会一直留着上一帧的光柱（"关不掉的光柱"）。
    let final_view = if settings.enabled && settings.strength() > 0.0 {
      a_src.clone()
    } else {
      ph_view.clone()
    };
    let read_bg = device.create_bind_group(
      None,
      &read_layout,
      &[
        BindGroupEntry { binding: 0, resource: uniform },
        BindGroupEntry { binding: 1, resource: BindingResource::TextureView(&final_view) },
      ],
    );
    // `uniform` 这个借用到这里结束 ⇒ 之后才能可变借 `gpu` 写回句柄。
    gpu.mask_bg = Some(mask_bg);
    gpu.blur_a_bg = Some(blur_a_bg);
    gpu.blur_b_bg = Some(blur_b_bg);
    gpu.read_bg = Some(read_bg);
  } else {
    // 资源还没就绪（首帧）⇒ 光柱那一项用占位（恒 0）；**参数 uniform 照绑**（太阳盘要它）。
    gpu.mask_bg = None;
    gpu.blur_a_bg = None;
    gpu.blur_b_bg = None;
    gpu.read_bg = Some(device.create_bind_group(
      None,
      &read_layout,
      &[
        BindGroupEntry { binding: 0, resource: gpu.uniform.binding().expect("uniform 已写入") },
        BindGroupEntry { binding: 1, resource: BindingResource::TextureView(&ph_view) },
      ],
    ));
  }
  commands.insert_resource(FogReadBg(gpu.read_bg.clone()));
}

/// `dda_main` 的 group(6)（**恒有**：关掉光柱时是占位版）。
#[derive(bevy::ecs::resource::Resource)]
pub struct FogReadBg(pub Option<bevy::render::render_resource::BindGroup>);

/// 光柱在 `dispatch_dda` 里的三件套，**打包成一个 `SystemParam`**：
/// Bevy 的系统函数最多 16 个参数，而主 pass 的 dispatch 正好卡在边界上（拆成三个就超了）。
#[derive(bevy::ecs::system::SystemParam)]
pub(crate) struct FogRes<'w> {
  pub settings: Option<bevy::ecs::system::Res<'w, FogSettings>>,
  pub gpu: Option<bevy::ecs::system::Res<'w, FogGpu>>,
  pub read_bg: Option<bevy::ecs::system::Res<'w, FogReadBg>>,
}

/// 派发整条光柱链（掩码 → 模糊 A → 模糊 B）—— 由 `brickmap::dda::dispatch_dda`
/// 在**主 pass 之前**调用（主 pass 要采样它）。
/// 掩码 pass 的 group(0) 复用 GI pass 的 `gi_bg0`（view uniform + beam depth，两者要的东西逐字相同）、
/// group(1..3) 复用主 pass 的绑定、group(4) = 光柱那一组；两个模糊 pass 把它当 group(0)。
#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatch_fog(
  profiler: &mut crate::profiler::GpuProfilerRes,
  encoder: &mut bevy::render::render_resource::CommandEncoder,
  gpu: Option<&FogGpu>,
  settings: Option<&FogSettings>,
  pipeline_cache: &bevy::render::render_resource::PipelineCache,
  bg0: Option<&bevy::render::render_resource::BindGroup>,
  bg1: Option<&bevy::render::render_resource::BindGroup>,
  bg2: Option<&bevy::render::render_resource::BindGroup>,
  bg3: Option<&bevy::render::render_resource::BindGroup>,
) {
  let (Some(gpu), Some(s)) = (gpu, settings) else { return };
  if !s.enabled || s.strength() <= 0.0 {
    return;
  }
  let (Some(bg0), Some(bg1), Some(bg2), Some(bg3)) = (bg0, bg1, bg2, bg3) else {
    return;
  };
  let Some(mask_bg) = gpu.mask_bg.as_ref() else {
    return;
  };
  let gx = gpu.size.x.div_ceil(DDA_WORKGROUP_SIZE);
  let gy = gpu.size.y.div_ceil(DDA_WORKGROUP_SIZE);
  if gx == 0 || gy == 0 {
    return;
  }
  // ---- ① 掩码（1 条主射线/texel）----
  if let Some(pipe) = gpu.pipelines[0].and_then(|id| pipeline_cache.get_compute_pipeline(id)) {
    crate::profiler::gpu_compute_pass(profiler, encoder, "gate_godray", |pass| {
      pass.set_pipeline(pipe);
      pass.set_bind_group(0, bg0, &[]);
      pass.set_bind_group(1, bg1, &[]);
      pass.set_bind_group(2, bg2, &[]);
      pass.set_bind_group(3, bg3, &[]);
      pass.set_bind_group(4, mask_bg, &[]);
      pass.dispatch_workgroups(gx, gy, 1);
    });
  }
  // ---- ② 径向模糊两趟（各自的 layout/bind group，只换 pipeline）----
  for (i, (label, bg)) in [
    ("gate_godray_blur_a", gpu.blur_a_bg.as_ref()),
    ("gate_godray_blur_b", gpu.blur_b_bg.as_ref()),
  ]
  .into_iter()
  .enumerate()
  {
    let Some(bg) = bg else { continue };
    let Some(pipe) = gpu.pipelines[i + 1].and_then(|id| pipeline_cache.get_compute_pipeline(id))
    else {
      continue;
    };
    crate::profiler::gpu_compute_pass(profiler, encoder, label, |pass| {
      pass.set_pipeline(pipe);
      pass.set_bind_group(0, bg, &[]);
      pass.dispatch_workgroups(gx, gy, 1);
    });
  }
}
