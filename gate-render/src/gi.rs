//! GI：逐 (体素,面) 辐照度缓存（世界空间、相机无关的间接光存储）。
//! 缓存算法与布局见 `assets/shaders/voxel_raytrace/gi/cache.wesl`；Rust 侧只负责开 buffer
//! （条目表 / 哈希桶 / 分配游标 / 可见条目紧凑列表 + 计数器）、上传 uniform、建 bind group、并派发更新 pass。
//! 更新 pass（`gi_cache_update`）每帧在着色前跑一次，按「可见优先 + 收敛度优先」消费每帧预算
//! （见 `main.wesl` 的同名入口）。可见列表计数器由 Rust 每帧清零。

use bevy::render::render_resource::{
  BindGroupLayoutDescriptor, CachedComputePipelineId, ShaderType,
};
use glam::{Mat4, UVec2, UVec4, Vec4};

use crate::wesl_consts::gi_consts;

/// 缓存 uniform（WESL `bindings.wesl` 的 `GiUniform` 逐字段镜像，字节一致）。
#[repr(C)]
#[derive(Debug, Default, Clone, Copy, ShaderType)]
pub struct GiUniform {
  /// x = 保留（恒 0，帧计数器已迁到 `seq.x`）、y = 保留（恒 0）、z = GI 增益、w = 当前世代号
  /// （太阳/天光/世界全量变化 → 自增 ⇒ 旧世代条目在两侧都被当作无数据）
  pub params: Vec4,
  /// x = GI 开关（0/1）、yzw = 保留（恒 0）
  pub misc: Vec4,
  /// x = 保留（恒 0）、y = GI 分辨率除数（1 = 全分辨率、2 = 半分辨率；**整数值的 f32**，
  /// 只被 `gi_main` 用来把本 pass 的像素下标换成 beam 纹理下标）、zw = 保留（恒 0）
  pub flags: Vec4,
  /// xyz = 脏盒 0（主世界编辑）的世界 voxel AABB min、w = 本帧脏盒数（0 = 无脏区）。
  /// 盒 1..N-1 在 BG4 binding 18（`gi_dirty_boxes`）；盒 0 留在 uniform ⇒ 主世界单盒路径
  /// 与旧版逐字等价，且更新 pass 不发生 storage 读。
  pub dirty_min: Vec4,
  /// xyz = 盒 0 的 AABB max（开区间）、w = 盒 0 的失效余量（voxel，按 volume scale 放大）
  pub dirty_max: Vec4,
  /// x = 自增帧号（精确 u32；yzw 恒 0）。所有**整数**帧逻辑（轮转起点、条目 `[10]` 去重、
  /// 可见 claim、RNG 种子混入）都用它：`params.x` 曾是 f32 帧号，超过 2^24 后无法表示连续整数
  /// ⇒ 去重 / 分片 / claim 会偶发失效。`gpu.frame` 本就是 u32，这里逐字镜像、不再经 f32。
  /// 无符号回绕（约 4.29e9 帧）后 claim 的「frame == 已 claim」可能误判一次，属可忽略。
  pub seq: UVec4,
  /// 上一帧的相机矩阵（**屏幕空间路径的时域复用**：把本帧主命中点重投影到上帧 GI 网格）。
  /// `prepare_gi` 每帧把上帧实际用过的那一份写进 uniform，再把当前帧的存下来 ⇒ 与上帧逐位一致。
  /// 只被 `prev_view_proj` 消费（重投影三维点不需要逆矩阵），逆矩阵留作后续步骤备用。
  pub prev_view_proj: Mat4,
  pub prev_inv_view_proj: Mat4,
}

/// GI 档位（菜单「渲染/GI」）：`enabled` → uniform `misc.x`；`gi_div` → uniform `flags.y`。
#[derive(bevy::ecs::resource::Resource, Clone, Copy, Debug, PartialEq)]
pub struct GiSettings {
  /// GI 开关（关掉 = 整条 GI 链不派发，主 pass 只有太阳直射 + 天光兜底）。
  pub enabled: bool,
  /// GI **分辨率除数**：1 = 全分辨率、2 = 半分辨率（GI 网格边长 = 渲染分辨率 ÷ 本值）。
  /// 不是开关：1 与 2 **都跑 GI**，只是网格疏密与代价不同。
  pub gi_div: u32,
}

impl GiSettings {
  /// 生效的分辨率除数（只实现了 1 / 2 两档，越界值钳回来）。
  pub fn div(&self) -> u32 {
    self.gi_div.clamp(1, 2)
  }

  /// GI 网格尺寸 = 渲染分辨率 ÷ `div()`（逐轴向下取整，至少 1×1）。
  pub fn gi_size(&self, render_size: UVec2) -> UVec2 {
    let d = self.div();
    UVec2::new((render_size.x / d).max(1), (render_size.y / d).max(1))
  }
}

impl Default for GiSettings {
  fn default() -> Self {
    Self { enabled: true, gi_div: 2 }
  }
}

/// BG4 布局：uniform(0) + 缓存 buffer（13 条目表 / 14 哈希桶 / 15 分配游标 /
/// 16 可见条目紧凑列表 / 17 可见列表长度计数器 / 18 脏盒 1..N-1 / 19 单帧可见 claim /
/// 20·21 屏幕空间 reservoir 双缓冲：20 = 本帧写、21 = 上帧读）。
pub fn gi_bg4_layout() -> BindGroupLayoutDescriptor {
  use bevy::render::render_resource::*;
  const C: ShaderStages = ShaderStages::COMPUTE;
  let buf = |binding: u32| BindGroupLayoutEntry {
    binding,
    visibility: C,
    ty: BindingType::Buffer {
      ty: BufferBindingType::Storage { read_only: false },
      has_dynamic_offset: false,
      min_binding_size: None,
    },
    count: None,
  };
  BindGroupLayoutDescriptor::new(
    "GiBg4",
    &[
      BindGroupLayoutEntry {
        binding: 0,
        visibility: C,
        ty: BindingType::Buffer {
          ty: BufferBindingType::Uniform,
          has_dynamic_offset: false,
          min_binding_size: Some(GiUniform::min_size()),
        },
        count: None,
      },
      buf(13),
      buf(14),
      buf(15),
      buf(16),
      buf(17),
      buf(18),
      buf(19),
      // 20/21：reservoir 双缓冲（结构上是 `array<u32>`；21 只读但用同一种 binding 类型，
      // 与 13..19 保持一致的风格）。
      buf(20),
      buf(21),
    ],
  )
}

/// BG5 写入侧布局：GI 的两张输出纹理（2 = gi、3 = cov）+ 降噪导引 buffer（6）。
/// 纹理尺寸 = GI 网格（渲染分辨率 ÷ `GiSettings.gi_div`），与全分辨率无关。
/// 采样侧（binding 4/5）在 `brickmap::dda` 里单独一份 layout，只给 `dda_main`。
pub fn gi_bg5_layout() -> BindGroupLayoutDescriptor {
  use bevy::render::render_resource::*;
  const C: ShaderStages = ShaderStages::COMPUTE;
  let store = |binding: u32, format: TextureFormat| BindGroupLayoutEntry {
    binding,
    visibility: C,
    ty: BindingType::StorageTexture {
      access: StorageTextureAccess::WriteOnly,
      format,
      view_dimension: TextureViewDimension::D2,
    },
    count: None,
  };
  BindGroupLayoutDescriptor::new(
    "GiBg5",
    &[
      // GI 网格分辨率下的 GI：rgb = gi·valid、a = valid
      store(2, TextureFormat::Rgba16Float),
      // GI 网格分辨率下的覆盖度：r = cov·valid、g = valid
      store(3, TextureFormat::Rg32Float),
      // 降噪导引（`gi_main` 写、两段降噪读；布局见 WESL `gi/consts.wesl`）
      BindGroupLayoutEntry {
        binding: 6,
        visibility: C,
        ty: BindingType::Buffer {
          ty: BufferBindingType::Storage { read_only: false },
          has_dynamic_offset: false,
          min_binding_size: None,
        },
        count: None,
      },
    ],
  )
}

/// 降噪时域段的布局：**只有 group(0) 一份**（binding 号见 `bindings.wesl`）。
/// 单 group 是刻意的：两段降噪都不需要相机矩阵（上帧像素坐标由 `gi_main` 写进导引）、
/// 也不需要 `grid_descs`（平面检验容差同样写在导引里）⇒ pipeline layout 不必拉进 bg1..bg4。
pub fn gi_den_temporal_layout() -> BindGroupLayoutDescriptor {
  use bevy::render::render_resource::*;
  const C: ShaderStages = ShaderStages::COMPUTE;
  let ro = |binding: u32| BindGroupLayoutEntry {
    binding,
    visibility: C,
    ty: BindingType::Buffer {
      ty: BufferBindingType::Storage { read_only: true },
      has_dynamic_offset: false,
      min_binding_size: None,
    },
    count: None,
  };
  let rw = |binding: u32| BindGroupLayoutEntry {
    binding,
    visibility: C,
    ty: BindingType::Buffer {
      ty: BufferBindingType::Storage { read_only: false },
      has_dynamic_offset: false,
      min_binding_size: None,
    },
    count: None,
  };
  let tex = |binding: u32| BindGroupLayoutEntry {
    binding,
    visibility: C,
    ty: BindingType::Texture {
      sample_type: TextureSampleType::Float { filterable: true },
      view_dimension: TextureViewDimension::D2,
      multisampled: false,
    },
    count: None,
  };
  BindGroupLayoutDescriptor::new(
    "GiDenTemporal",
    &[
      ro(10),  // 导引
      tex(11), // 本帧原始 GI（gi_out 的采样视图）
      ro(12),  // 历史（上帧）
      rw(13),  // 历史（本帧）
      BindGroupLayoutEntry {
        binding: 16, // 时域输出
        visibility: C,
        ty: BindingType::StorageTexture {
          access: StorageTextureAccess::WriteOnly,
          format: TextureFormat::Rgba16Float,
          view_dimension: TextureViewDimension::D2,
        },
        count: None,
      },
      rw(17), // φ（每像素亮度 range 尺度）
    ],
  )
}

/// 降噪空间段（迭代 atrous）的布局：group(0)，binding 14 = 采样输入、15 = 存储输出、10 = 导引、17 = φ。
/// Rust 每轮换绑 14/15（同一份 layout、同一个入口，见 `AuxTexCache::den_bg`）。
pub fn gi_den_atrous_layout() -> BindGroupLayoutDescriptor {
  use bevy::render::render_resource::*;
  const C: ShaderStages = ShaderStages::COMPUTE;
  BindGroupLayoutDescriptor::new(
    "GiDenAtrous",
    &[
      BindGroupLayoutEntry {
        binding: 10,
        visibility: C,
        ty: BindingType::Buffer {
          ty: BufferBindingType::Storage { read_only: true },
          has_dynamic_offset: false,
          min_binding_size: None,
        },
        count: None,
      },
      BindGroupLayoutEntry {
        binding: 14,
        visibility: C,
        ty: BindingType::Texture {
          sample_type: TextureSampleType::Float { filterable: true },
          view_dimension: TextureViewDimension::D2,
          multisampled: false,
        },
        count: None,
      },
      BindGroupLayoutEntry {
        binding: 15,
        visibility: C,
        ty: BindingType::StorageTexture {
          access: StorageTextureAccess::WriteOnly,
          format: TextureFormat::Rgba16Float,
          view_dimension: TextureViewDimension::D2,
        },
        count: None,
      },
      BindGroupLayoutEntry {
        binding: 17,
        visibility: C,
        ty: BindingType::Buffer {
          ty: BufferBindingType::Storage { read_only: false },
          has_dynamic_offset: false,
          min_binding_size: None,
        },
        count: None,
      },
    ],
  )
}

#[derive(bevy::ecs::resource::Resource)]
pub struct GiGpu {
  pub uniform: bevy::render::render_resource::UniformBuffer<GiUniform>,
  /// 条目表 / 哈希桶（桶内存「条目下标 + 1」）/ `[0]` = 分配游标、`[1]` = 回收扫描游标。
  /// 三块全由 GPU 维护（着色侧建条目、更新 pass 写值），Rust 只开 buffer 并清零。
  pub gi_cache: bevy::render::render_resource::Buffer,
  pub gi_cache_hash: bevy::render::render_resource::Buffer,
  pub gi_cache_state: bevy::render::render_resource::Buffer,
  /// 可视条目调度：`gi_vis_list` = 着色侧 claim 后 append 的条目下标紧凑列表；
  /// `gi_vis_count` 的 `[0]` = 本帧列表长度（原子自增），由 Rust 每帧在更新 pass 之前清零。
  pub gi_vis_list: bevy::render::render_resource::Buffer,
  pub gi_vis_count: bevy::render::render_resource::Buffer,
  /// 脏盒 1..GI_CACHE_DIRTY_BOXES-1（`GI_CACHE_DIRTY_BOXES` × 2 个 vec4：`[2k]` = min.xyz + 余量、
  /// `[2k+1]` = max.xyz）。盒 0 走 uniform；每帧仅在盒数 ≥ 2 时写。
  pub gi_dirty_boxes: bevy::render::render_resource::Buffer,
  /// 单帧可见 claim（`GI_CACHE_SLOTS` 个 u32 的原子）：`atomicExchange(claim[idx], frame)`
  /// 精确保证「每条目每帧只 append 一次」（Rust 只管开 buffer 并清零，frame 从 1 起）。
  pub gi_seen_claim: bevy::render::render_resource::Buffer,
  pub frame: u32,
  /// 上一帧 `gi_main` 实际用过的相机矩阵（uniform `prev_view_proj` / `prev_inv_view_proj` 的来源）。
  /// 只在真正跑 GI 的帧更新 ⇒ 与上帧写入 reservoir 时用的矩阵逐位一致（时域重投影才准）。
  pub prev_view_proj: Mat4,
  pub prev_inv_view_proj: Mat4,
  /// reservoir 双缓冲的换绑状态：true ⇒ 本帧 `binding 20 = b`、`21 = a`（见 `prepare_gi`）。
  pub res_flip: bool,
  /// 当前世代号（uniform `params.w`）：太阳/天光变化、palette 变化、世界全量重建或脏盒超量时自增。
  pub generation: u32,
  /// 上一帧的太阳/天光状态哈希（与 `generation` 比较判断「变没变」）。
  pub light_hash: u64,
  /// 脏盒（世界 voxel AABB，`[min, max)` ＋逐盒余量）的跨帧保留：`BrickMapDirty` 只在当帧有效，
  /// 而更新 pass 要 `GI_CACHE_SLOTS / GI_CACHE_UPDATE_BUDGET` 帧才轮转完一遍 ⇒ 按住那么多帧上传并集。
  /// 与旧版单个 `dirty_lo/hi` 的差别只是「N 个盒而不是 1 个」，保留时长（`dirty_ttl`）与语义不变。
  pub dirty_boxes: Vec<crate::brickmap::upload::DirtyBox>,
  /// 脏区剩余有效帧数（0 = 本帧无脏区）。
  pub dirty_ttl: u32,
  /// 更新 pass 的 pipeline（`gi_cache_update`）。
  pub pipeline: Option<CachedComputePipelineId>,
  /// 降噪 pipeline：`[0]` = 时域、`[1..4]` = atrous 第 1/2/3 轮（步长 1/2/4）。
  /// layout 只有 group(0) 一份（见 [`gi_den_temporal_layout`] / [`gi_den_atrous_layout`]）。
  pub den_pipelines: [Option<CachedComputePipelineId>; 4],
}

#[derive(bevy::ecs::resource::Resource)]
pub struct GiBg4(pub bevy::render::render_resource::BindGroup);

/// GI 写入侧 bind group（`gi_main` 用）
#[derive(bevy::ecs::resource::Resource)]
pub struct GiBg5(pub bevy::render::render_resource::BindGroup);

/// GI 写入侧的占位纹理（1×1）：GI 缓冲未就绪时 BG5 仍须为 binding 2/3 提供视图。
/// 占位纹理不会被真正写入。
#[derive(bevy::ecs::resource::Resource, Default)]
struct GiPlaceholder {
  tex: Option<bevy::render::render_resource::Texture>,
  cov: Option<bevy::render::render_resource::Texture>,
  view: Option<bevy::render::render_resource::TextureView>,
  cov_view: Option<bevy::render::render_resource::TextureView>,
  /// BG4 binding 20/21 的占位（GI 缓冲未就绪时用；1 个 word，足够绑定，不会被读写）。
  res: Option<bevy::render::render_resource::Buffer>,
}

impl GiPlaceholder {
  fn views(
    &mut self,
    device: &bevy::render::renderer::RenderDevice,
  ) -> (&bevy::render::render_resource::TextureView, &bevy::render::render_resource::TextureView)
  {
    use bevy::render::render_resource::*;
    if self.tex.is_none() {
      let make = |label: &str, format: TextureFormat| {
        device.create_texture(&TextureDescriptor {
          label: Some(label),
          size: Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
          mip_level_count: 1,
          sample_count: 1,
          dimension: TextureDimension::D2,
          format,
          usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
          view_formats: &[],
        })
      };
      let t = make("gate_gi_placeholder", TextureFormat::Rgba16Float);
      let c = make("gate_gi_cov_placeholder", TextureFormat::Rg32Float);
      self.view = Some(t.create_view(&TextureViewDescriptor::default()));
      self.cov_view = Some(c.create_view(&TextureViewDescriptor::default()));
      self.tex = Some(t);
      self.cov = Some(c);
    }
    (self.view.as_ref().expect("刚创建"), self.cov_view.as_ref().expect("刚创建"))
  }

  /// BG4 binding 20/21 的占位 buffer（4 B）。
  fn res_buffer(
    &mut self,
    device: &bevy::render::renderer::RenderDevice,
  ) -> &bevy::render::render_resource::Buffer {
    use bevy::render::render_resource::{BufferDescriptor, BufferUsages};
    self.res.get_or_insert_with(|| {
      device.create_buffer(&BufferDescriptor {
        label: Some("gate_gi_res_placeholder"),
        size: 4,
        usage: BufferUsages::STORAGE,
        mapped_at_creation: false,
      })
    })
  }
}

pub struct GiPlugin;

impl bevy::app::Plugin for GiPlugin {
  fn build(&self, app: &mut bevy::app::App) {
    use bevy::ecs::schedule::IntoScheduleConfigs;
    use bevy::prelude::RenderGraph;
    app.init_resource::<GiSettings>();
    let Some(render_app) = app.get_sub_app_mut(bevy::render::RenderApp) else {
      return;
    };
    render_app
      .init_resource::<GiSettings>()
      .init_resource::<GiPlaceholder>()
      .add_systems(bevy::render::RenderStartup, init_gi_gpu)
      .add_systems(
        bevy::render::RenderStartup,
        queue_gi_pipelines.after(crate::brickmap::dda::init_dda_pipelines),
      )
      .add_systems(bevy::render::ExtractSchedule, extract_gi_settings)
      .add_systems(
        bevy::render::Render,
        prepare_gi
          .in_set(bevy::render::RenderSystems::PrepareBindGroups)
          .after(crate::brickmap::upload::prepare)
          .after(crate::brickmap::dda::prepare_dda_bind_groups),
      )
      .add_systems(
        RenderGraph,
        dispatch_gi
          .in_set(bevy::render::renderer::RenderGraphSystems::Render)
          .before(crate::brickmap::dda::dispatch_dda),
      );
  }
}

fn dummy_sized_buffer(
  device: &bevy::render::renderer::RenderDevice,
  label: &str,
  size: u64,
) -> bevy::render::render_resource::Buffer {
  use bevy::render::render_resource::{BufferDescriptor, BufferUsages};
  device.create_buffer(&BufferDescriptor {
    label: Some(label),
    size: size.max(4),
    usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
    mapped_at_creation: false,
  })
}

fn zero_storage_buffer(
  device: &bevy::render::renderer::RenderDevice,
  queue: &bevy::render::renderer::RenderQueue,
  label: &str,
  size: u64,
) -> bevy::render::render_resource::Buffer {
  let buf = dummy_sized_buffer(device, label, size);
  queue.write_buffer(&buf, 0, &vec![0u8; size.max(4) as usize]);
  buf
}

fn init_gi_gpu(
  mut commands: bevy::ecs::system::Commands,
  device: bevy::ecs::system::Res<bevy::render::renderer::RenderDevice>,
  queue: bevy::ecs::system::Res<bevy::render::renderer::RenderQueue>,
) {
  let c = gi_consts();
  bevy::log::info!(
    target: "gate",
    "GI 缓存：条目 {} × {} word = {:.1}MB、哈希桶 {} = {:.1}MB、可见列表 {} 项 = {:.1}MB、\
     可见 claim {} 项 = {:.1}MB（每帧更新预算 {} 条，可见优先 {}%；脏盒上限 {} 个，\
     条目去掉 word[18] 省下的 4MB 与 claim 互换 ⇒ 净显存不变）",
    c.gi_cache_slots,
    c.gi_cache_entry_words,
    c.gi_cache_bytes() as f64 / (1 << 20) as f64,
    c.gi_cache_buckets,
    c.gi_cache_bucket_bytes() as f64 / (1 << 20) as f64,
    c.gi_cache_visible_capacity,
    c.gi_cache_visible_bytes() as f64 / (1 << 20) as f64,
    c.gi_cache_slots,
    c.gi_cache_claim_bytes() as f64 / (1 << 20) as f64,
    c.gi_cache_update_budget,
    c.gi_cache_visible_share,
    c.gi_cache_dirty_boxes,
  );
  let gi_cache = zero_storage_buffer(&device, &queue, "gi_cache", c.gi_cache_bytes());
  let gi_cache_hash = zero_storage_buffer(&device, &queue, "gi_cache_hash", c.gi_cache_bucket_bytes());
  let gi_cache_state = zero_storage_buffer(&device, &queue, "gi_cache_state", 16);
  let gi_vis_list =
    zero_storage_buffer(&device, &queue, "gi_vis_list", c.gi_cache_visible_bytes());
  let gi_vis_count = zero_storage_buffer(&device, &queue, "gi_vis_count", 16);
  let gi_dirty_boxes =
    zero_storage_buffer(&device, &queue, "gi_dirty_boxes", c.gi_cache_dirty_box_bytes());
  // frame 从 1 起（prepare 每帧自增后才写 uniform）⇒ 0 = 未 claim，必须清零初始化。
  let gi_seen_claim =
    zero_storage_buffer(&device, &queue, "gi_seen_claim", c.gi_cache_claim_bytes());
  commands.insert_resource(GiGpu {
    uniform: bevy::render::render_resource::UniformBuffer::default(),
    gi_cache,
    gi_cache_hash,
    gi_cache_state,
    gi_vis_list,
    gi_vis_count,
    gi_dirty_boxes,
    gi_seen_claim,
    frame: 0,
    // 首帧没有「上一帧」⇒ 恒等矩阵；此时 reservoir 两块都是零（M = 0）⇒ 复用一律判无效。
    prev_view_proj: Mat4::IDENTITY,
    prev_inv_view_proj: Mat4::IDENTITY,
    res_flip: false,
    generation: 0,
    light_hash: 0,
    dirty_boxes: Vec::new(),
    dirty_ttl: 0,
    pipeline: None,
    den_pipelines: [None; 4],
  });
}

fn queue_gi_pipelines(
  dda: Option<bevy::ecs::system::Res<crate::brickmap::dda::DdaPipelines>>,
  pipeline_cache: bevy::ecs::system::Res<bevy::render::render_resource::PipelineCache>,
  dda_shader: bevy::ecs::system::Res<crate::shader::DdaShaderHandle>,
  mut gpu: bevy::ecs::system::ResMut<GiGpu>,
) {
  use bevy::render::render_resource::ComputePipelineDescriptor;
  use std::borrow::Cow;
  // ---- 降噪（两步四 pass）：不依赖 DdaPipelines —— layout 只有 group(0) 一份 ----
  if gpu.den_pipelines[0].is_none() {
    let den_temporal_layout = gi_den_temporal_layout();
    let den_atrous_layout = gi_den_atrous_layout();
    let label = [
      "gate_gi_denoise_temporal",
      "gate_gi_denoise_atrous1",
      "gate_gi_denoise_atrous2",
      "gate_gi_denoise_atrous4",
    ];
    let entry = [
      "gi_denoise_temporal",
      "gi_denoise_atrous1",
      "gi_denoise_atrous2",
      "gi_denoise_atrous4",
    ];
    for i in 0..4 {
      let layout = if i == 0 {
        vec![den_temporal_layout.clone()]
      } else {
        vec![den_atrous_layout.clone()]
      };
      gpu.den_pipelines[i] = Some(pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some(Cow::from(label[i])),
        layout,
        shader: dda_shader.0.clone(),
        entry_point: Some(Cow::from(entry[i])),
        ..Default::default()
      }));
    }
  }
  if gpu.pipeline.is_some() {
    return;
  }
  let Some(dda) = dda else {
    return;
  };
  let base = vec![
    dda.bg0_layout.clone(),
    dda.bg1_layout.clone(),
    dda.bg2_layout.clone(),
    dda.bg3_layout.clone(),
    gi_bg4_layout(),
  ];
  gpu.pipeline = Some(pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
    label: Some(Cow::from("gate_gi_cache_update")),
    layout: base,
    shader: dda_shader.0.clone(),
    entry_point: Some(Cow::from("gi_cache_update")),
    ..Default::default()
  }));
}

fn extract_gi_settings(
  mut commands: bevy::ecs::system::Commands,
  settings: Option<bevy::render::Extract<bevy::ecs::system::Res<GiSettings>>>,
) {
  commands.insert_resource(
    settings.map_or_else(GiSettings::default, |s| {
      GiSettings { enabled: s.enabled, gi_div: s.div() }
    }),
  );
}

/// 太阳 / 天光状态 → 64 位哈希（FNV-1a）：方向光方向、颜色、强度、天空色任一变化即变。
/// 只覆盖 GI 缓存真正读到的东西（更新 pass 的 miss 用 `sky_rgb()`，命中直射项用 `lights[0]`）。
fn light_state_hash(theme: Option<&crate::lighting::LightingTheme>) -> u64 {
  fn acc(h: u64, bits: u32) -> u64 {
    (h ^ bits as u64).wrapping_mul(0x100_0000_01b3)
  }
  let mut h = 0xcbf2_9ce4_8422_2325u64;
  let Some(t) = theme else { return h };
  h = acc(h, 1); // 主题已就绪
  match &t.sun {
    Some(s) => {
      h = acc(h, 2);
      for v in s.dir {
        h = acc(h, v.to_bits());
      }
      for v in s.color {
        h = acc(h, v.to_bits());
      }
      h = acc(h, s.intensity.to_bits());
    }
    None => h = acc(h, 3),
  }
  match &t.sky {
    Some(s) => {
      h = acc(h, 4);
      for v in s.color {
        h = acc(h, v.to_bits());
      }
    }
    None => h = acc(h, 5),
  }
  h
}

#[allow(clippy::too_many_arguments)]
fn prepare_gi(
  mut commands: bevy::ecs::system::Commands,
  device: bevy::ecs::system::Res<bevy::render::renderer::RenderDevice>,
  queue: bevy::ecs::system::Res<bevy::render::renderer::RenderQueue>,
  pipeline_cache: bevy::ecs::system::Res<bevy::render::render_resource::PipelineCache>,
  settings: bevy::ecs::system::Res<GiSettings>,
  view: Option<bevy::ecs::system::Res<crate::brickmap::dda::DdaViewUniform>>,
  aux: Option<bevy::ecs::system::Res<crate::brickmap::dda::AuxTexCache>>,
  dirty: Option<bevy::ecs::system::Res<crate::brickmap::upload::BrickMapDirty>>,
  lighting: Option<bevy::ecs::system::Res<crate::lighting::LightingTheme>>,
  mut gi_ph: bevy::ecs::system::ResMut<GiPlaceholder>,
  mut gpu: bevy::ecs::system::ResMut<GiGpu>,
) {
  gpu.frame = gpu.frame.wrapping_add(1);
  // 可见列表计数器清零：着色 pass 在上一帧末尾 append 完，本帧更新 pass 要读它。
  // 必须在更新 pass 之前（prepare 早于 RenderGraph），且 GPU 命令按队列顺序执行 ⇒ 不会与上一帧的写入打架。
  queue.write_buffer(&gpu.gi_vis_count, 0, &0u32.to_le_bytes());

  // ---- 变化失效：世代号 ----
  // 太阳/天光变了 ⇒ 缓存里所有条目都过期（全量作废，但不清缓冲，靠世代号判脏）。
  // `why` = 本帧世代自增的原因（可多个），末尾一条 info! 打印（A1/A2 的因果证据）。
  let mut why: Vec<&'static str> = Vec::new();
  let hash = light_state_hash(lighting.as_ref().map(|r| &**r));
  if hash != gpu.light_hash {
    gpu.light_hash = hash;
    gpu.generation = gpu.generation.wrapping_add(1);
    why.push("太阳/天光");
  }

  // ---- 变化失效：局部脏盒（编辑体素）----
  // `BrickMapDirty` 由 `brickmap::upload::extract` 每帧写入：逐 volume（主世界 + 物体）一个世界
  // voxel AABB 盒 ＋ 失效余量。全量上传（首帧 / force_full / 非增量）没有有意义的 AABB ⇒ 当作
  // 世界整体变化，走世代号；只改 palette（换色 / 改材质参数）同样无 AABB ⇒ 也走世代号
  // （palette 是共享的，改一个色号无法廉价定位受影响体素，全量失效是最省的**正确**做法；
  // 拖动调色期间世代会持续自增、松手后收敛）。
  // 脏盒只在当帧有效，但更新 pass 每帧只轮转到 1/sweeps 的条目 ⇒ 按住 `sweeps` 帧（并集）上传，
  // 保证脏区内的每个条目至少被扫到一次。
  // 轮转段预算因「可见优先」而变小（`GI_CACHE_VISIBLE_SHARE`）⇒ 整表扫一遍的帧数变长；
  // 脏区 TTL 必须跟着变长，否则脏区条目可能在脏区窗口关闭之后才被轮转到，留下上一世代的旧值。
  let sweeps = gi_consts().gi_cache_sweeps();
  let box_cap = gi_consts().gi_cache_dirty_boxes.max(1) as usize;
  // 本帧是否收到新的脏盒（决定是否打日志：盒在 TTL 内跨帧保留，逐帧打会刷屏）。
  let mut fresh_boxes = false;
  if let Some(d) = dirty.as_ref() {
    if d.full {
      gpu.generation = gpu.generation.wrapping_add(1);
      gpu.dirty_boxes.clear();
      gpu.dirty_ttl = 0;
      why.push("世界全量上传");
    } else {
      if d.palette_changed {
        gpu.generation = gpu.generation.wrapping_add(1);
        why.push("palette");
      }
      if !d.boxes.is_empty() {
        fresh_boxes = true;
        // 保留窗口过期 ⇒ 从这一帧的盒重新开始；否则与在留的盒取并集（重叠即并成一盒）。
        if gpu.dirty_ttl == 0 {
          gpu.dirty_boxes.clear();
        }
        for b in &d.boxes {
          match gpu.dirty_boxes.iter_mut().find(|e| e.overlaps(b)) {
            Some(e) => e.union_with(b),
            None => gpu.dirty_boxes.push(*b),
          }
        }
        gpu.dirty_ttl = sweeps;
        if gpu.dirty_boxes.len() > box_cap {
          // 溢出：宁可全量失效也不要漏失效 —— 自增世代并清空盒表（盒表已被世代覆盖）。
          gpu.dirty_boxes.clear();
          gpu.dirty_ttl = 0;
          gpu.generation = gpu.generation.wrapping_add(1);
          why.push("脏盒超量溢出");
        }
      }
    }
  }
  // 盒 0 走 uniform、盒 1..N-1 走 `gi_dirty_boxes`；`dirty_min.w` = 盒数（0 = 本帧无脏区）。
  let mut dirty_min = Vec4::ZERO;
  let mut dirty_max = Vec4::ZERO;
  if gpu.dirty_ttl > 0 {
    if let Some(b0) = gpu.dirty_boxes.first().copied() {
      dirty_min = Vec4::new(
        b0.lo.x as f32,
        b0.lo.y as f32,
        b0.lo.z as f32,
        gpu.dirty_boxes.len() as f32,
      );
      dirty_max = Vec4::new(b0.hi.x as f32, b0.hi.y as f32, b0.hi.z as f32, b0.margin);
    }
    gpu.dirty_ttl -= 1;
  } else {
    gpu.dirty_boxes.clear();
  }
  let box_n = dirty_min.w as usize;
  if !why.is_empty() || fresh_boxes {
    let why_s = why.join("+");
    bevy::log::info!(
      target: "gate",
      "GI 失效：世代 → {}（自增原因：{}）；本帧脏盒 {} 个{}（TTL {} 帧，上限 {}）",
      gpu.generation,
      if why_s.is_empty() { "-" } else { why_s.as_str() },
      box_n,
      gpu
        .dirty_boxes
        .first()
        .map(|b| format!("，盒 0 = [{},{},{}]..[{},{},{}] 余量 {:.1}", b.lo.x, b.lo.y, b.lo.z, b.hi.x, b.hi.y, b.hi.z, b.margin))
        .unwrap_or_default(),
      gpu.dirty_ttl,
      box_cap,
    );
  }
  // 盒 1.. 的字节：每盒 2 个 vec4（min.xyz + 余量 / max.xyz + 0）。盒数 < 2 时不必写（shader 不读）。
  if box_n >= 2 {
    let mut words = vec![0f32; box_cap * 8];
    for (k, b) in gpu.dirty_boxes.iter().enumerate().skip(1) {
      let o = k * 8;
      words[o] = b.lo.x as f32;
      words[o + 1] = b.lo.y as f32;
      words[o + 2] = b.lo.z as f32;
      words[o + 3] = b.margin;
      words[o + 4] = b.hi.x as f32;
      words[o + 5] = b.hi.y as f32;
      words[o + 6] = b.hi.z as f32;
      words[o + 7] = 0.0;
    }
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    queue.write_buffer(&gpu.gi_dirty_boxes, 0, &bytes);
  }

  // ---- uniform（字段与 WESL `GiUniform` 逐字段镜像）----
  let mut u = GiUniform::default();
  u.params = Vec4::new(0.0, 0.0, crate::consts::GI_GAIN, gpu.generation as f32);
  u.misc = Vec4::new(if settings.enabled { 1.0 } else { 0.0 }, 0.0, 0.0, 0.0);
  u.flags = Vec4::new(0.0, settings.div() as f32, 0.0, 0.0);
  u.dirty_min = dirty_min;
  u.dirty_max = dirty_max;
  // 整数帧号走 u32 通道（`seq.x`）：`gpu.frame` 本就是 u32，不再经 `params.x` 的 f32 截断。
  u.seq = UVec4::new(gpu.frame, 0, 0, 0);
  // 上一帧相机矩阵（屏幕空间路径的时域重投影）：写「上帧真正用过的那一份」，再把本帧存下来。
  u.prev_view_proj = gpu.prev_view_proj;
  u.prev_inv_view_proj = gpu.prev_inv_view_proj;
  *gpu.uniform.get_mut() = u;
  gpu.uniform.write_buffer(&device, &queue);
  // 只在真正会跑 `gi_main` 的帧更新「上一帧」⇒ 与上帧写 reservoir 时用的矩阵逐位一致
  // （GI 关掉一段时间再打开时，历史 reservoir 与 prev 矩阵都停留在最后一帧 GI，重投影仍自洽）。
  // 注意：**分辨率的任意取值都跑 GI** —— 这里只跟 `enabled` 走。
  let gi_runs = settings.enabled;
  if gi_runs && let Some(v) = view.as_ref() {
    gpu.prev_view_proj = v.view_proj;
    gpu.prev_inv_view_proj = v.inv_view_proj;
  }

  // ---- BG4：uniform + 缓存 buffer（绑定号 0/13…21，必须显式给 entry）----
  use bevy::render::render_resource::{BindGroupEntry, BindingResource};
  let bg4_layout = pipeline_cache.get_bind_group_layout(&gi_bg4_layout());
  // 屏幕空间 reservoir 双缓冲（binding 20 = 本帧写、21 = 上帧读）：两块由 AuxTexCache 随 GI
  // 分辨率一起创建；未就绪（首帧 / prepare 提前返回）时用 4 B 占位（该帧不会派发 `gi_main`）。
  let (res_cur, res_prev) = match aux.as_ref().and_then(|a| a.gi_res_buffers()) {
    Some((a, b)) if gpu.res_flip => (b, a),
    Some((a, b)) => (a, b),
    None => {
      let p = gi_ph.res_buffer(&device);
      (p, p)
    }
  };
  let bg4 = device.create_bind_group(
    None,
    &bg4_layout,
    &[
      BindGroupEntry { binding: 0, resource: gpu.uniform.binding().expect("uniform 已写入") },
      BindGroupEntry { binding: 13, resource: gpu.gi_cache.as_entire_binding() },
      BindGroupEntry { binding: 14, resource: gpu.gi_cache_hash.as_entire_binding() },
      BindGroupEntry { binding: 15, resource: gpu.gi_cache_state.as_entire_binding() },
      BindGroupEntry { binding: 16, resource: gpu.gi_vis_list.as_entire_binding() },
      BindGroupEntry { binding: 17, resource: gpu.gi_vis_count.as_entire_binding() },
      BindGroupEntry { binding: 18, resource: gpu.gi_dirty_boxes.as_entire_binding() },
      BindGroupEntry { binding: 19, resource: gpu.gi_seen_claim.as_entire_binding() },
      BindGroupEntry { binding: 20, resource: res_cur.as_entire_binding() },
      BindGroupEntry { binding: 21, resource: res_prev.as_entire_binding() },
    ],
  );
  commands.insert_resource(GiBg4(bg4));
  // 换绑：本帧的写入目标成为下一帧的读源。
  if gi_runs {
    gpu.res_flip = !gpu.res_flip;
  }

  // ---- BG5：GI 的写入侧（绑定号 2/3/6）----
  // GI 视图来自 `crate::brickmap::dda::AuxTexCache`；未就绪时用 1×1 占位纹理（bind group 必须给全条目）。
  // 视图/ buffer 都先 clone 成句柄（`TextureView`/`Buffer` 都是 Arc 包装）⇒ 之后还能再借一次 `gi_ph`。
  let bg5_layout = pipeline_cache.get_bind_group_layout(&gi_bg5_layout());
  let (gi_view, gi_cov_view) = match aux.as_ref().and_then(|a| a.gi_write_views()) {
    Some((a, b)) => (a.clone(), b.clone()),
    None => {
      let (a, b) = gi_ph.views(&device);
      (a.clone(), b.clone())
    }
  };
  // binding 6 = 降噪导引（`gi_main` 写）；未就绪时用 4 B 占位（该帧不会派发 `gi_main`）。
  let guide = aux
    .as_ref()
    .and_then(|a| a.gi_guide_buffer())
    .cloned()
    .unwrap_or_else(|| gi_ph.res_buffer(&device).clone());
  let bg5 = device.create_bind_group(
    None,
    &bg5_layout,
    &[
      BindGroupEntry { binding: 2, resource: BindingResource::TextureView(&gi_view) },
      BindGroupEntry { binding: 3, resource: BindingResource::TextureView(&gi_cov_view) },
      BindGroupEntry { binding: 6, resource: guide.as_entire_binding() },
    ],
  );
  commands.insert_resource(GiBg5(bg5));
}

fn dispatch_gi(
  mut ctx: bevy::render::renderer::RenderContext,
  bg0: Option<bevy::ecs::system::Res<crate::brickmap::dda::DdaBg0BindGroup>>,
  bg1: Option<bevy::ecs::system::Res<crate::brickmap::dda::DdaBg1BindGroup>>,
  bg2: Option<bevy::ecs::system::Res<crate::brickmap::dda::DdaBg2BindGroup>>,
  bg3: Option<bevy::ecs::system::Res<crate::brickmap::dda::DdaBg3BindGroup>>,
  bg4: Option<bevy::ecs::system::Res<GiBg4>>,
  gpu: Option<bevy::ecs::system::Res<GiGpu>>,
  pipeline_cache: bevy::ecs::system::Res<bevy::render::render_resource::PipelineCache>,
  mut profiler: bevy::ecs::system::ResMut<crate::profiler::GpuProfilerRes>,
) {
  let (Some(bg0), Some(bg1), Some(bg2), Some(bg3), Some(bg4), Some(gpu)) =
    (bg0.as_ref(), bg1.as_ref(), bg2.as_ref(), bg3.as_ref(), bg4.as_ref(), gpu.as_ref())
  else {
    return;
  };
  let Some(pipe_id) = gpu.pipeline else {
    return;
  };
  // 世界空间缓存被关掉（`GI_CACHE_ON = false`，主路径 = 屏幕空间逐面 ReSTIR）⇒ 整表轮转 pass
  // 没有任何消费者，直接不派发（省下它整帧的射线与访存开销）。
  if !gi_consts().gi_cache_on {
    return;
  }
  let Some(pipe) = pipeline_cache.get_compute_pipeline(pipe_id) else {
    return;
  };
  // 整表轮转：每个线程一个条目，WG=64。
  let wgs = gi_consts().gi_cache_update_wgs();
  crate::profiler::gpu_compute_pass(
    &mut profiler,
    ctx.command_encoder(),
    "gate_gi_cache_update",
    |pass| {
      pass.set_pipeline(pipe);
      pass.set_bind_group(0, &bg0.0, &[]);
      pass.set_bind_group(1, &bg1.0, &[]);
      pass.set_bind_group(2, &bg2.0, &[]);
      pass.set_bind_group(3, &bg3.0, &[]);
      pass.set_bind_group(4, &bg4.0, &[]);
      pass.dispatch_workgroups(wgs, 1, 1);
    },
  );
}

