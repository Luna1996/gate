//! DDGI（Dynamic Diffuse Global Illumination）：世界空间探针烘焙 + 活跃探针筛选 + cast/collect。
//! 4 级嵌套级联 LOD：LOD1~3 锚定世界 AABB 铺 cell 网格，LOD0 按 chunk 从探针池领固定 4096 槽的段；
//! 流程 `ddgi_bake0..3` → `ddgi_sort` → `seal` → `cast` → `collect`。

use bevy::render::render_resource::{CachedComputePipelineId, ShaderType};
use glam::{IVec3, IVec4, UVec3, UVec4, Vec4};

use crate::consts::{DDGI_GRID_MARGIN, DDGI_LOD_CELL_SIZES};
use crate::wesl_consts::ddgi_consts;

// 图集纹素数（irr / depth）与每帧射线预算（`DDGI_RAY_BUDGET`）的权威值都在 WESL
// （`ddgi/consts.wesl`），Rust 侧经 `wesl_consts::ddgi_consts()` 解析同一份源码得到。
pub const DDGI_LODS: u32 = 4;
// 槽位映射是世界锚定的：`slot = slot_base + (世界 cell 号 mod dims)`（见 WGSL `ddgi_slot`）。
// 前提：原点必须是 cell 整数倍。

// ---- LOD0 的 chunk 锚定（其它 LOD 保持世界 AABB 规则网格）----
// LOD0 按 chunk 拥有：需要探针的 chunk 从探针池领一段固定 4096 槽（= (256/16)³）的槽位，
// `slot = chunk_base[chunk] + 局部 cell 索引`；只有 LOD0 走这条路径，段基址一旦分配即固定不变。
/// chunk 边长（voxel）：世界体素卷按它切块（= 引擎自己的 brick chunk 粒度）。
pub const DDGI_CHUNK_VOXELS: i32 = 256;
/// 每 chunk 每轴含多少个 LOD0 cell：16。
pub const DDGI_CHUNK_LOD0_AXIS: i32 = DDGI_CHUNK_VOXELS / DDGI_LOD_CELL_SIZES[0];
/// 每 chunk 的 LOD0 段大小：4096 槽。
pub const DDGI_CHUNK_LOD0_SLOTS: u32 =
  (DDGI_CHUNK_LOD0_AXIS as u32) * (DDGI_CHUNK_LOD0_AXIS as u32) * (DDGI_CHUNK_LOD0_AXIS as u32);
/// 探针池里「该 chunk 未分配段」的哨兵基址。
pub const DDGI_CHUNK_NO_BASE: u32 = u32::MAX;

/// LOD0 的 chunk 网格几何（chunk 单位）：原点 = 世界 AABB 向下对齐到 256 再向外扩 1 chunk，
/// 维度 = 覆盖所需 chunk 数 +2；未分配的 chunk 由 `DDGI_CHUNK_NO_BASE` 标记。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DdgiChunkGeom {
  pub origin: IVec3,
  pub dims: UVec3,
}

impl Default for DdgiChunkGeom {
  fn default() -> Self {
    Self { origin: IVec3::ZERO, dims: UVec3::ONE }
  }
}

impl DdgiChunkGeom {
  pub fn from_world(aabb_min: IVec3, aabb_max: IVec3) -> Self {
    let c = DDGI_CHUNK_VOXELS;
    let aligned =
      IVec3::new(align_down(aabb_min.x, c), align_down(aabb_min.y, c), align_down(aabb_min.z, c))
        / c
        - IVec3::ONE;
    let span = (aabb_max - aligned * c).max(IVec3::ONE);
    let dims = UVec3::new(
      ((span.x + c - 1) / c).max(1) as u32 + 2,
      ((span.y + c - 1) / c).max(1) as u32 + 2,
      ((span.z + c - 1) / c).max(1) as u32 + 2,
    );
    Self { origin: aligned, dims }
  }

  #[inline]
  pub fn len(&self) -> u32 {
    self.dims.x * self.dims.y * self.dims.z
  }

  /// 空网格（dims 全零）：`from_world` 保证每轴 ≥2，实际不会出现。
  #[inline]
  pub fn is_empty(&self) -> bool {
    self.dims.x == 0 || self.dims.y == 0 || self.dims.z == 0
  }

  /// chunk 坐标 → 网格内线性下标（含边界检查）
  #[inline]
  pub fn linear(&self, cc: IVec3) -> Option<u32> {
    let r = cc - self.origin;
    if r.x < 0
      || r.y < 0
      || r.z < 0
      || r.x as u32 >= self.dims.x
      || r.y as u32 >= self.dims.y
      || r.z as u32 >= self.dims.z
    {
      return None;
    }
    Some((r.x as u32) + (r.y as u32) * self.dims.x + (r.z as u32) * self.dims.x * self.dims.y)
  }

  #[inline]
  pub fn coord(&self, idx: u32) -> IVec3 {
    let d = self.dims;
    self.origin
      + IVec3::new((idx % d.x) as i32, ((idx / d.x) % d.y) as i32, (idx / (d.x * d.y)) as i32)
  }
}

/// LOD0 探针池：chunk → 段基址（段大小恒为 `DDGI_CHUNK_LOD0_SLOTS`），配一个空闲链表。
/// 只有有几何（或紧邻几何）的 chunk 才领段；`sync` 只领段/归还，已有 chunk 的基址保持不变。
#[derive(Clone, Debug, Default)]
pub struct DdgiChunkPool {
  pub geom: DdgiChunkGeom,
  /// 每 chunk 的段基址（`DDGI_CHUNK_NO_BASE` = 未分配）
  pub bases: Vec<u32>,
  /// 空闲段基址（chunk 释放时归还，LIFO 复用）
  pub free: Vec<u32>,
  /// 池高水位（下一个新段基址）→ LOD0 槽位数 = 它
  pub next_base: u32,
  /// 任何分配/归并都自增：`prepare_ddgi` 据此判定「池变了 → 需要重烘」。
  pub serial: u64,
}

impl DdgiChunkPool {
  #[inline]
  pub fn lod0_slots(&self) -> u32 {
    self.next_base
  }

  /// 与期望的 chunk 集合同步。返回「是否发生变化」。
  pub fn sync(&mut self, geom: DdgiChunkGeom, wanted: &[IVec3]) -> bool {
    let mut changed = false;
    if self.geom != geom || self.bases.len() != geom.len() as usize {
      self.geom = geom;
      self.bases = vec![DDGI_CHUNK_NO_BASE; geom.len() as usize];
      self.free.clear();
      self.next_base = 0;
      changed = true;
    }
    let mut want: Vec<u32> = wanted.iter().filter_map(|cc| self.geom.linear(*cc)).collect();
    want.sort_unstable();
    want.dedup();
    for &l in want.iter() {
      if self.bases[l as usize] == DDGI_CHUNK_NO_BASE {
        let base = self.free.pop().unwrap_or_else(|| {
          let b = self.next_base;
          self.next_base += DDGI_CHUNK_LOD0_SLOTS;
          b
        });
        self.bases[l as usize] = base;
        changed = true;
      }
    }
    for l in 0..self.bases.len() {
      let b = self.bases[l];
      if b != DDGI_CHUNK_NO_BASE && want.binary_search(&(l as u32)).is_err() {
        self.bases[l] = DDGI_CHUNK_NO_BASE;
        self.free.push(b);
        changed = true;
      }
    }
    if changed {
      self.serial = self.serial.wrapping_add(1);
    }
    changed
  }
}

/// LOD0 段如何编址 —— uniform 里给 shader 的那两个 vec4（见 WGSL `DdgiUniform.chunk`）。
#[repr(C)]
#[derive(Debug, Default, Clone, Copy, ShaderType)]
pub struct DdgiChunkUniform {
  /// xyz = chunk 网格原点（chunk 坐标），w = 每 chunk 每轴 cell 数（16）
  pub origin: IVec4,
  /// xyz = chunk 网格维度，w = LOD0 已分配槽数（池高水位）
  pub dims: UVec4,
}

/// 主世界算出的「LOD0 需要探针段的 chunk」集合（chunk 坐标）。
/// 判定规则：chunk 自己有几何，或其邻域（cell 粒度）内有几何。
#[derive(bevy::ecs::resource::Resource, Clone, Debug, Default, PartialEq)]
pub struct DdgiLod0Chunks {
  /// 需要 LOD0 段的 chunk 坐标（世界体素坐标 / 256）。
  pub chunks: Vec<IVec3>,
}

// indirect args / 计数器合一 buffer 的 word 布局权威值在 WESL（`bindings.wesl` 的 `DDGI_INDIR_*_BASE`），
// Rust 只按解析出的 word 偏移算字节偏移与清零区间。

#[inline]
fn align_down(v: i32, a: i32) -> i32 {
  v.div_euclid(a) * a
}

/// 世界空间探针网格（4 级嵌套级联，各级独立原点）。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DdgiWorldGrid {
  /// 各 LOD 的世界原点（voxel，按该级 cell 边长对齐）。
  pub lod_origins: [IVec3; DDGI_LODS as usize],
  /// 各 LOD 的 cell 维度（cell 单位）。
  pub lod_dims: [UVec3; DDGI_LODS as usize],
  /// 各 LOD 在全局 slot 数组中的起始下标。
  pub lod_slot_base: [u32; DDGI_LODS as usize],
  pub total_slots: u32,
}

impl DdgiWorldGrid {
  #[inline]
  pub fn lod_count(&self, lod: usize) -> u32 {
    let d = self.lod_dims[lod];
    d.x * d.y * d.z
  }

  #[inline]
  pub fn lod_cell_size(&self, lod: usize) -> i32 {
    DDGI_LOD_CELL_SIZES[lod]
  }

  /// 由世界 AABB 推导 4 级嵌套级联：原点按该级 cell 边长向下对齐，dims = ceil(跨度 / cell)，
  /// 各级严格嵌套。原点必须是 cell 整数倍，世界锚定的槽位映射才成立。
  pub fn from_world(aabb_min: IVec3, aabb_max: IVec3) -> Self {
    let aabb_min = aabb_min - IVec3::splat(DDGI_GRID_MARGIN);
    let aabb_max = aabb_max + IVec3::splat(DDGI_GRID_MARGIN);
    let mut out = Self::default();
    let mut base = 0u32;
    for (lod, &cell) in DDGI_LOD_CELL_SIZES.iter().enumerate() {
      let origin = IVec3::new(
        align_down(aabb_min.x, cell),
        align_down(aabb_min.y, cell),
        align_down(aabb_min.z, cell),
      );
      let span = (aabb_max - origin).max(IVec3::ONE);
      let c = cell;
      let dims = UVec3::new(
        ((span.x + c - 1) / c).max(1) as u32,
        ((span.y + c - 1) / c).max(1) as u32,
        ((span.z + c - 1) / c).max(1) as u32,
      );
      out.lod_origins[lod] = origin;
      out.lod_dims[lod] = dims;
      out.lod_slot_base[lod] = base;
      base += dims.x * dims.y * dims.z;
    }
    out.total_slots = base;
    out
  }

  #[inline]
  pub fn is_empty(&self) -> bool {
    self.total_slots == 0
  }
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, ShaderType)]
pub struct DdgiLod {
  /// xyz = 世界原点（voxel），w = cell 边长
  pub origin: IVec4,
  /// xyz = cell 维度，w = 全局 slot 起始下标
  pub dims: UVec4,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, bevy::ecs::resource::Resource, ShaderType)]
pub struct DdgiUniform {
  pub lods: [DdgiLod; 4],
  /// x = frame, y = debug mode, z = gain, w = 保留（恒 0）
  pub params: Vec4,
  /// x = shade GI, y = total slots
  pub misc: Vec4,
  /// 脏区（世界 voxel AABB）：xyz = min，w = 1 表示有效（0 = 本帧无脏区）
  pub dirty_min: Vec4,
  /// xyz = max（不含）；与 dirty_min 一起决定哪些 cell 强制重烘（局部编辑增量）
  pub dirty_max: Vec4,
  /// LOD0 的 chunk 段编址（仅 lod==0 用；LOD1~3 走 `lods`）。见 `DdgiChunkUniform`。
  pub chunk: DdgiChunkUniform,
  /// 档位：x = 保留（恒 0）；y = 半分辨率 GI（1 = 走 1/2 分辨率缓冲）；zw 保留。
  pub flags: Vec4,
}

pub fn ddgi_bg4_layout() -> bevy::render::render_resource::BindGroupLayoutDescriptor {
  use bevy::render::render_resource::*;
  const C: ShaderStages = ShaderStages::COMPUTE;
  let tex = |binding: u32, sample_type: TextureSampleType| BindGroupLayoutEntry {
    binding,
    visibility: C,
    ty: BindingType::Texture {
      sample_type,
      view_dimension: TextureViewDimension::D2Array,
      multisampled: false,
    },
    count: None,
  };
  let buf = |binding: u32, read_only: bool| BindGroupLayoutEntry {
    binding,
    visibility: C,
    ty: BindingType::Buffer {
      ty: BufferBindingType::Storage { read_only },
      has_dynamic_offset: false,
      min_binding_size: None,
    },
    count: None,
  };
  BindGroupLayoutDescriptor::new(
    "DdgiBg4",
    &[
      BindGroupLayoutEntry {
        binding: 0,
        visibility: C,
        ty: BindingType::Buffer {
          ty: BufferBindingType::Uniform,
          has_dynamic_offset: false,
          min_binding_size: Some(DdgiUniform::min_size()),
        },
        count: None,
      },
      // 1/2：irradiance / depth 图集（采样侧）
      tex(1, TextureSampleType::Float { filterable: true }),
      tex(2, TextureSampleType::Float { filterable: false }),
      // 3：烘焙输出；4：age/flags；5：indirect/counter；6：worklist；7：slot_pos；8：cell_id；
      // 9：cast 射线样本；10：LOD0 chunk 表 + 使用计数前缀（读写）。
      buf(3, false),
      buf(4, false),
      buf(5, false),
      buf(6, false),
      buf(7, false),
      buf(8, false),
      buf(9, false),
      buf(10, false),
    ],
  )
}

/// BG5（collect + 半分辨率 GI 写入侧）：图集的写入侧 + GI 缓冲的写入侧；必须与 BG4 分离。
pub fn ddgi_bg5_layout() -> bevy::render::render_resource::BindGroupLayoutDescriptor {
  use bevy::render::render_resource::*;
  const C: ShaderStages = ShaderStages::COMPUTE;
  let store = |binding: u32, format: TextureFormat| BindGroupLayoutEntry {
    binding,
    visibility: C,
    ty: BindingType::StorageTexture {
      access: StorageTextureAccess::WriteOnly,
      format,
      view_dimension: TextureViewDimension::D2Array,
    },
    count: None,
  };
  let store2d = |binding: u32, format: TextureFormat| BindGroupLayoutEntry {
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
    "DdgiBg5",
    &[
      store(0, TextureFormat::Rgba16Float),
      // depth 图集用 Rgba16Float：.x = mean、.y = std（距离标准差），供采样侧做 Chebyshev 软遮挡。
      store(1, TextureFormat::Rgba16Float),
      // 半分辨率 GI：rgb = gi·valid、a = valid
      store2d(2, TextureFormat::Rgba16Float),
      // 半分辨率 GI 的覆盖度：r = cov·valid、g = valid
      store2d(3, TextureFormat::Rg32Float),
    ],
  )
}

/// BG6（仅 seal 用）：dispatch 参数 buffer 的写入侧；必须与 BG4 分离。
/// 该 buffer 在 cast/collect 作 indirect 参数源。
pub fn ddgi_bg6_layout() -> bevy::render::render_resource::BindGroupLayoutDescriptor {
  use bevy::render::render_resource::*;
  const C: ShaderStages = ShaderStages::COMPUTE;
  BindGroupLayoutDescriptor::new(
    "DdgiBg6",
    &[BindGroupLayoutEntry {
      binding: 0,
      visibility: C,
      ty: BindingType::Buffer {
        ty: BufferBindingType::Storage { read_only: false },
        has_dynamic_offset: false,
        min_binding_size: None,
      },
      count: None,
    }],
  )
}

#[derive(Debug, Clone, Copy)]
pub struct DdgiPipelines {
  /// per-LOD 烘焙入口（lod 0..DDGI_LODS，细→粗）；必须逐级拆成独立 pass（pass 边界即内存屏障）。
  pub bake: [CachedComputePipelineId; DDGI_LODS as usize],
  pub sort: CachedComputePipelineId,
  pub seal: CachedComputePipelineId,
  pub cast: CachedComputePipelineId,
  pub collect: CachedComputePipelineId,
}

/// 图集的一侧（纹理 + D2Array 视图）
pub struct DdgiAtlas {
  pub tex: bevy::render::render_resource::Texture,
  pub view: bevy::render::render_resource::TextureView,
}

#[derive(bevy::ecs::resource::Resource)]
pub struct DdgiGpu {
  pub uniform: bevy::render::render_resource::UniformBuffer<DdgiUniform>,
  pub indirect: bevy::render::render_resource::Buffer,
  /// dispatch 参数（`INDIRECT`）；必须与 `ddgi_indirect` 分开，只在 seal pass 以 storage 绑定（BG6）。
  pub args: bevy::render::render_resource::Buffer,
  pub worklist: bevy::render::render_resource::Buffer,
  pub slot_pos: bevy::render::render_resource::Buffer,
  /// LOD0 chunk 段的两张表，与使用计数前缀复用同一 u32 数组：
  /// [0, total_slots) 使用计数；[total_slots, +num_chunks) chunk_base；其后 slot_chunk。
  pub cell_slot: bevy::render::render_resource::Buffer,
  /// LOD0 探针池（chunk → 段基址，见 `DdgiChunkPool`）。
  pub pool: DdgiChunkPool,
  /// 上一次同步进来的 LOD0 chunk 集合（变化才重建池/反查表）。
  pub last_chunks: Vec<IVec3>,
  /// `cell_slot` 的构建键 `(total_slots, num_chunks, lod0_slots, pool.serial)`；变了才重建。
  pub cell_slot_key: Option<(u32, u32, u32, u64)>,
  /// 烘焙输出：每 slot 一条 (flags | off_b)
  pub cell: bevy::render::render_resource::Buffer,
  /// 每 slot 已烘焙的世界 cell 键 + 有效标志（滚动增量烘焙）
  pub cell_id: bevy::render::render_resource::Buffer,
  /// 每帧 age / enabled
  pub meta: bevy::render::render_resource::Buffer,
  /// 图集 ping-pong：BG4 绑「当前」（上一帧 collect 写入），BG5 绑「目标」（本帧 collect 写入）；
  /// 读写必须落在不同纹理 + 不同 bind group。
  pub irr: [DdgiAtlas; 2],
  pub depth: [DdgiAtlas; 2],
  /// 采样侧索引（0/1）；dispatch_ddgi 在 collect 跑完后翻转
  pub parity: usize,
  /// cast 输出的射线样本（方向 + 命中距离）/(辐亮度 + 1)
  pub samples: bevy::render::render_resource::Buffer,
  pub frame: u32,
  pub grid: DdgiWorldGrid,
  pub total_slots: u32,
  /// 已消费的世界修订号（变化 → 需要重烘焙）
  pub last_revision: u64,
  /// 有一次烘焙请求已发出但还没真正派发（pipeline/绑定组未就绪）；必须留到真正派发的那一帧。
  pub bake_pending: bool,
  pub pipelines: Option<DdgiPipelines>,
}

#[derive(bevy::ecs::resource::Resource)]
pub struct DdgiBg4(pub bevy::render::render_resource::BindGroup);

/// collect 专用 bind group（图集写入侧 + 样本/间接参数读取）
#[derive(bevy::ecs::resource::Resource)]
pub struct DdgiBg5(pub bevy::render::render_resource::BindGroup);

/// seal 专用 bind group（dispatch 参数 buffer 写入侧）
#[derive(bevy::ecs::resource::Resource)]
pub struct DdgiBg6(pub bevy::render::render_resource::BindGroup);

/// 半分辨率 GI 写入侧的占位纹理（1×1）：GI 缓冲未就绪时 BG5 仍须为 binding 2/3 提供视图。
/// 占位纹理不会被真正写入。
#[derive(bevy::ecs::resource::Resource, Default)]
struct GiPlaceholder {
  tex: Option<bevy::render::render_resource::Texture>,
  cov: Option<bevy::render::render_resource::Texture>,
  view: Option<bevy::render::render_resource::TextureView>,
  cov_view: Option<bevy::render::render_resource::TextureView>,
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
}

/// 本帧是否需要跑探针烘焙（由 prepare_ddgi 写入，dispatch_ddgi 读取）
#[derive(bevy::ecs::resource::Resource, Clone, Copy, Default)]
pub struct DdgiBakeThisFrame(pub bool);

#[derive(bevy::ecs::resource::Resource, Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct DdgiStage(pub u8);

impl DdgiStage {
  pub const OFF: u8 = 0;
  pub const ACTIVE: u8 = 1;
  pub const CAST: u8 = 2;
  pub const FULL: u8 = 3;

  pub fn new(v: u8) -> Self {
    Self(v.min(Self::FULL))
  }
  /// 启动阶段取自 [`crate::consts::DDGI_STAGE`]（0=Off 1=Active 2=Cast 3=Full）。
  pub fn startup() -> Self {
    Self::new(crate::consts::DDGI_STAGE)
  }
  pub fn run_active(&self) -> bool {
    self.0 >= Self::ACTIVE
  }
  /// 是否跑阶段二/三（cast + collect）
  pub fn run_cast(&self) -> bool {
    self.0 >= Self::CAST
  }
  pub fn shade_gi(&self) -> bool {
    self.0 >= Self::FULL
  }
}

/// 已加载世界的 AABB（voxel 坐标，闭区间）；由主世界算出并 extract 到 render world，
/// 未设置时取单位盒。DDGI 的各级网格锚定到它。
#[derive(bevy::ecs::resource::Resource, Clone, Copy, Debug, PartialEq, Eq)]
pub struct DdgiWorldAabb {
  pub min: IVec3,
  pub max: IVec3,
}

impl Default for DdgiWorldAabb {
  fn default() -> Self {
    Self { min: IVec3::ZERO, max: IVec3::ONE }
  }
}

#[derive(bevy::ecs::resource::Resource, Clone, Copy, Debug, PartialEq)]
pub struct DdgiDebugSettings {
  pub mode: f32,
  pub gain: f32,
  pub probe_viz: bool,
  pub probe_viz_lod: f32,
  /// Chebyshev 里 std 项的信任系数，对应 WGSL `misc.z`（0..1，默认 1 = 正常使用 std）。
  /// 拖到 0 = 完全忽略 std，`soft` 退回固定下限（保留作诊断用）。
  pub depth_soft_k: f32,
  /// 级联覆盖之外的天光兜底强度，对应 WGSL `misc.w`（0..1，默认 0.25）；独立于 `DDGI_SKY_AMBIENT`。
  pub far_ambient: f32,
  /// 半分辨率 GI（对应 WGSL `flags.y`）：`ddgi_sample` 搬到 1/2 分辨率的独立 pass，主 pass 只做
  /// 双线性取用。
  pub gi_half_res: bool,
}

impl Default for DdgiDebugSettings {
  fn default() -> Self {
    Self {
      mode: 0.0,
      gain: 1.0,
      probe_viz: false,
      probe_viz_lod: 0.0,
      depth_soft_k: 1.0,
      far_ambient: 0.25,
      gi_half_res: false,
    }
  }
}

pub struct DdgiPlugin;

impl bevy::app::Plugin for DdgiPlugin {
  fn build(&self, app: &mut bevy::app::App) {
    use bevy::ecs::schedule::IntoScheduleConfigs;
    use bevy::prelude::RenderGraph;
    let stage = DdgiStage::startup();
    bevy::log::info!(
      target: "gate",
      "DDGI stage = {} ({}) —— 见 consts.rs 的 DDGI_STAGE",
      stage.0,
      ["Off", "Active", "Cast", "Full"][stage.0.min(3) as usize],
    );
    app.insert_resource(stage);
    app.init_resource::<DdgiDebugSettings>();
    let Some(render_app) = app.get_sub_app_mut(bevy::render::RenderApp) else {
      return;
    };
    render_app
      .init_resource::<DdgiStage>()
      .init_resource::<DdgiWorldAabb>()
      .init_resource::<DdgiLod0Chunks>()
      .init_resource::<DdgiDebugSettings>()
      .init_resource::<DdgiBakeThisFrame>()
      .init_resource::<GiPlaceholder>()
      .add_systems(bevy::render::RenderStartup, init_ddgi_gpu)
      .add_systems(
        bevy::render::RenderStartup,
        queue_ddgi_pipelines.after(crate::brickmap::dda::init_dda_pipelines),
      )
      .add_systems(bevy::render::ExtractSchedule, extract_ddgi_settings)
      .add_systems(
        bevy::render::Render,
        prepare_ddgi
          .in_set(bevy::render::RenderSystems::PrepareBindGroups)
          .after(crate::brickmap::upload::prepare)
          .after(crate::brickmap::dda::prepare_dda_bind_groups),
      )
      .add_systems(
        RenderGraph,
        dispatch_ddgi
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

/// 容量不足时重建（并清零）。
fn ensure_storage_buffer(
  device: &bevy::render::renderer::RenderDevice,
  queue: &bevy::render::renderer::RenderQueue,
  buf: &mut bevy::render::render_resource::Buffer,
  label: &str,
  bytes: u64,
) {
  let bytes = bytes.max(4);
  if buf.size() >= bytes {
    return;
  }
  let new = dummy_sized_buffer(device, label, bytes);
  queue.write_buffer(&new, 0, &vec![0u8; bytes as usize]);
  *buf = new;
}

/// dispatch 参数 / 计数器 buffer：作为 storage 被 sort/seal 读写，args 还作 indirect 参数源。
/// 两者必须是不同 buffer（见 `DdgiGpu::args`）。
fn ddgi_indirect_buffer(
  device: &bevy::render::renderer::RenderDevice,
  queue: &bevy::render::renderer::RenderQueue,
  label: &str,
) -> bevy::render::render_resource::Buffer {
  use bevy::render::render_resource::{BufferDescriptor, BufferUsages};
  // 字节数由 WESL 的 word 布局推出（`DDGI_INDIR_*_BASE` / 计数器段 + 每 LOD 一个 word）。
  let bytes = ddgi_consts().indirect_bytes();
  let buf = device.create_buffer(&BufferDescriptor {
    label: Some(label),
    size: bytes,
    usage: BufferUsages::STORAGE
      | BufferUsages::INDIRECT
      | BufferUsages::COPY_DST
      | BufferUsages::COPY_SRC,
    mapped_at_creation: false,
  });
  queue.write_buffer(&buf, 0, &vec![0u8; bytes as usize]);
  buf
}

fn ddgi_array_view(
  tex: &bevy::render::render_resource::Texture,
) -> bevy::render::render_resource::TextureView {
  use bevy::render::render_resource::{TextureViewDescriptor, TextureViewDimension};
  tex.create_view(&TextureViewDescriptor {
    dimension: Some(TextureViewDimension::D2Array),
    ..Default::default()
  })
}

/// 图集容量 = `DDGI_ATLAS_LAYERS × DDGI_PROBES_PER_LAYER_AXIS²`（均为 WESL 权威值）；池与
/// `from_world` 都不做钳制，实际 `total_slots` 必须 ≤ 容量。

fn ddgi_array_tex(
  device: &bevy::render::renderer::RenderDevice,
  label: &str,
  format: bevy::render::render_resource::TextureFormat,
  size: (u32, u32),
) -> bevy::render::render_resource::Texture {
  use bevy::render::render_resource::{
    Extent3d, TextureDescriptor, TextureDimension, TextureUsages,
  };
  device.create_texture(&TextureDescriptor {
    label: Some(label),
    size: Extent3d {
      width: size.0,
      height: size.1,
      depth_or_array_layers: ddgi_consts().atlas_layers,
    },
    mip_level_count: 1,
    sample_count: 1,
    dimension: TextureDimension::D2,
    format,
    usage: TextureUsages::TEXTURE_BINDING
      | TextureUsages::STORAGE_BINDING
      | TextureUsages::COPY_DST
      | TextureUsages::COPY_SRC,
    view_formats: &[],
  })
}

/// 图集清零。Rgba16Float 的 0.0 是全 0 字节，直接写零块。
/// 必须清零（探针 enabled 后即可能被采样）。
fn zero_array_tex(
  queue: &bevy::render::renderer::RenderQueue,
  label: &str,
  tex: &bevy::render::render_resource::Texture,
  size: (u32, u32),
  bytes_per_texel: u32,
) {
  use bevy::render::render_resource::{
    Extent3d, Origin3d, TexelCopyBufferLayout, TexelCopyTextureInfo, TextureAspect,
  };
  let bytes = (size.0 * size.1 * ddgi_consts().atlas_layers * bytes_per_texel) as usize;
  bevy::log::info!(target: "gate", "DDGI 图集清零 {label}: {:.1}MB", bytes as f64 / (1 << 20) as f64);
  let data = vec![0u8; bytes];
  queue.write_texture(
    TexelCopyTextureInfo {
      texture: tex,
      mip_level: 0,
      origin: Origin3d::ZERO,
      aspect: TextureAspect::All,
    },
    &data,
    TexelCopyBufferLayout {
      offset: 0,
      bytes_per_row: Some(size.0 * bytes_per_texel),
      rows_per_image: Some(size.1),
    },
    Extent3d { width: size.0, height: size.1, depth_or_array_layers: ddgi_consts().atlas_layers },
  );
}

fn init_ddgi_gpu(
  mut commands: bevy::ecs::system::Commands,
  device: bevy::ecs::system::Res<bevy::render::renderer::RenderDevice>,
  queue: bevy::ecs::system::Res<bevy::render::renderer::RenderQueue>,
) {
  use bevy::render::render_resource::TextureFormat;
  let c = ddgi_consts();
  let irr_axis = c.probes_per_layer_axis * c.irr_texels;
  let dep_axis = c.probes_per_layer_axis * c.depth_texels;
  let mk_atlas = |label: &str, format: TextureFormat, size: (u32, u32), bpt: u32| {
    let tex = ddgi_array_tex(&device, label, format, size);
    // 两侧都要清零：首帧 BG4 采样 a、collect 写 b，翻转后 a 才被写。
    zero_array_tex(&queue, label, &tex, size, bpt);
    let view = ddgi_array_view(&tex);
    DdgiAtlas { tex, view }
  };
  let irr = [
    mk_atlas("ddgi_irr_a", TextureFormat::Rgba16Float, (irr_axis, irr_axis), 8),
    mk_atlas("ddgi_irr_b", TextureFormat::Rgba16Float, (irr_axis, irr_axis), 8),
  ];
  let depth = [
    mk_atlas("ddgi_depth_a", TextureFormat::Rgba16Float, (dep_axis, dep_axis), 8),
    mk_atlas("ddgi_depth_b", TextureFormat::Rgba16Float, (dep_axis, dep_axis), 8),
  ];

  let slot_pos = zero_storage_buffer(&device, &queue, "ddgi_slot_pos", 4096 * 16);
  let worklist = zero_storage_buffer(&device, &queue, "ddgi_worklist", 4096 * 16);
  let cell = zero_storage_buffer(&device, &queue, "ddgi_cell", 4096 * 4);
  let cell_id = zero_storage_buffer(&device, &queue, "ddgi_cell_id", 4096 * 16);
  let meta = zero_storage_buffer(&device, &queue, "ddgi_meta", 4096 * 4);
  // LOD0 chunk 段表 + 保留前缀（同一 buffer，见 `DdgiGpu::cell_slot`）；这里只放占位，
  // 内容尺寸随世界 AABB / LOD0 chunk 集变化，`prepare_ddgi` 按构建键重建。
  let cell_slot = zero_storage_buffer(&device, &queue, "ddgi_cell_slot", 4);
  let samples = zero_storage_buffer(&device, &queue, "ddgi_samples", c.sample_bytes(0));
  let indirect = ddgi_indirect_buffer(&device, &queue, "ddgi_indirect");
  let args = ddgi_indirect_buffer(&device, &queue, "ddgi_args");

  commands.insert_resource(DdgiGpu {
    uniform: bevy::render::render_resource::UniformBuffer::default(),
    indirect,
    args,
    worklist,
    slot_pos,
    cell_slot,
    pool: DdgiChunkPool::default(),
    last_chunks: Vec::new(),
    cell_slot_key: None,
    cell,
    cell_id,
    meta,
    irr,
    depth,
    parity: 0,
    samples,
    frame: 0,
    grid: DdgiWorldGrid::default(),
    total_slots: 0,
    last_revision: u64::MAX,
    bake_pending: false,
    pipelines: None,
  });
}

fn queue_ddgi_pipelines(
  dda: Option<bevy::ecs::system::Res<crate::brickmap::dda::DdaPipelines>>,
  pipeline_cache: bevy::ecs::system::Res<bevy::render::render_resource::PipelineCache>,
  dda_shader: bevy::ecs::system::Res<crate::shader::DdaShaderHandle>,
  mut gpu: bevy::ecs::system::ResMut<DdgiGpu>,
) {
  use bevy::render::render_resource::{
    BindGroupLayoutDescriptor, ComputePipelineDescriptor, PipelineCache,
  };
  use std::borrow::Cow;
  if gpu.pipelines.is_some() {
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
    ddgi_bg4_layout(),
  ];
  // collect 额外挂 BG5、seal 额外挂 BG6；layout 的位置就是 bind group 索引，seal 的 @group(5)
  // 已被 collect 占用，5 号位留空 layout。
  let mut collect_layout = base.clone();
  collect_layout.push(ddgi_bg5_layout());
  let mut seal_layout = base.clone();
  seal_layout.push(BindGroupLayoutDescriptor::new("DdgiBgEmpty5", &[]));
  seal_layout.push(ddgi_bg6_layout());
  let shader = dda_shader.0.clone();
  let mk = |pipeline_cache: &PipelineCache,
            label: &'static str,
            entry: &'static str,
            layout: Vec<BindGroupLayoutDescriptor>| {
    pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
      label: Some(Cow::from(label)),
      layout,
      shader: shader.clone(),
      entry_point: Some(Cow::from(entry)),
      ..Default::default()
    })
  };
  gpu.pipelines = Some(DdgiPipelines {
    bake: [
      mk(&pipeline_cache, "gate_ddgi_bake0", "ddgi_bake0", base.clone()),
      mk(&pipeline_cache, "gate_ddgi_bake1", "ddgi_bake1", base.clone()),
      mk(&pipeline_cache, "gate_ddgi_bake2", "ddgi_bake2", base.clone()),
      mk(&pipeline_cache, "gate_ddgi_bake3", "ddgi_bake3", base.clone()),
    ],
    sort: mk(&pipeline_cache, "gate_ddgi_sort", "ddgi_sort", base.clone()),
    seal: mk(&pipeline_cache, "gate_ddgi_seal", "ddgi_seal", seal_layout),
    cast: mk(&pipeline_cache, "gate_ddgi_cast", "ddgi_cast", base.clone()),
    collect: mk(&pipeline_cache, "gate_ddgi_collect", "ddgi_collect", collect_layout),
  });
}

#[allow(clippy::too_many_arguments)]
fn dispatch_ddgi(
  mut ctx: bevy::render::renderer::RenderContext,
  bg0: Option<bevy::ecs::system::Res<crate::brickmap::dda::DdaBg0BindGroup>>,
  bg1: Option<bevy::ecs::system::Res<crate::brickmap::dda::DdaBg1BindGroup>>,
  bg2: Option<bevy::ecs::system::Res<crate::brickmap::dda::DdaBg2BindGroup>>,
  bg3: Option<bevy::ecs::system::Res<crate::brickmap::dda::DdaBg3BindGroup>>,
  bg4: Option<bevy::ecs::system::Res<DdgiBg4>>,
  bg5: Option<bevy::ecs::system::Res<DdgiBg5>>,
  bg6: Option<bevy::ecs::system::Res<DdgiBg6>>,
  bake: Option<bevy::ecs::system::Res<DdgiBakeThisFrame>>,
  stage: bevy::ecs::system::Res<DdgiStage>,
  dbg: Option<bevy::ecs::system::Res<DdgiDebugSettings>>,
  mut gpu: bevy::ecs::system::ResMut<DdgiGpu>,
  pipeline_cache: bevy::ecs::system::Res<bevy::render::render_resource::PipelineCache>,
  mut profiler: bevy::ecs::system::ResMut<crate::profiler::GpuProfilerRes>,
) {
  if !stage.run_active() && !dbg.is_some_and(|d| d.probe_viz) {
    return;
  }
  let (Some(bg0), Some(bg1), Some(bg2), Some(bg3), Some(bg4)) =
    (bg0.as_ref(), bg1.as_ref(), bg2.as_ref(), bg3.as_ref(), bg4.as_ref())
  else {
    return;
  };
  let Some(pipes) = gpu.pipelines else {
    return;
  };
  let Some(p_sort) = pipeline_cache.get_compute_pipeline(pipes.sort) else {
    return;
  };
  let Some(p_seal) = pipeline_cache.get_compute_pipeline(pipes.seal) else {
    return;
  };
  if gpu.total_slots == 0 {
    return;
  }

  let set_bgs = |pass: &mut bevy::render::render_resource::ComputePass,
                 bg4: &bevy::render::render_resource::BindGroup| {
    pass.set_bind_group(0, &bg0.0, &[]);
    pass.set_bind_group(1, &bg1.0, &[]);
    pass.set_bind_group(2, &bg2.0, &[]);
    pass.set_bind_group(3, &bg3.0, &[]);
    pass.set_bind_group(4, bg4, &[]);
  };
  // 1D dispatch：WG=64，各 pass 覆盖 total_slots（bake 仅在修订号变化时跑）
  let wg_slots = gpu.total_slots.div_ceil(64).min(65535);

  // 烘焙：世界数据变化时重算探针位置。按 LOD 逐级派发 4 个独立 pass（细→粗），pass 边界即内存屏障。
  if bake.is_some_and(|b| b.0) {
    let n_lods = DDGI_LODS as usize;
    // 每级 dispatch 大小按该级实际槽数算（LOD0 = 池高水位，LOD1~3 = dims 乘积）。
    let lod0_slots = gpu.grid.lod_slot_base[1];
    let wg_per_lod: [u32; DDGI_LODS as usize] = std::array::from_fn(|lod| {
      let n = if lod == 0 {
        lod0_slots
      } else {
        let d = gpu.grid.lod_dims[lod];
        d.x * d.y * d.z
      };
      n.div_ceil(64).clamp(1, 65535)
    });
    const LABELS: [&str; DDGI_LODS as usize] =
      ["gate_ddgi_bake0", "gate_ddgi_bake1", "gate_ddgi_bake2", "gate_ddgi_bake3"];
    // 4 个入口必须全部就绪才开跑。
    let mut p_bake: [Option<&bevy::render::render_resource::ComputePipeline>; DDGI_LODS as usize] =
      [None; DDGI_LODS as usize];
    let mut ready = true;
    for (slot, &pipe_id) in p_bake.iter_mut().zip(pipes.bake.iter()) {
      *slot = pipeline_cache.get_compute_pipeline(pipe_id);
      if slot.is_none() {
        ready = false;
      }
    }
    if ready {
      for lod in 0..n_lods {
        let p = p_bake[lod].unwrap();
        crate::profiler::gpu_compute_pass(
          &mut profiler,
          ctx.command_encoder(),
          LABELS[lod],
          |pass| {
            pass.set_pipeline(p);
            set_bgs(pass, &bg4.0);
            pass.dispatch_workgroups(wg_per_lod[lod], 1, 1);
          },
        );
      }
      // 真正派发过了才撤销挂起标志。
      gpu.bake_pending = false;
    }
  }
  // sort：读烘焙结果做活跃判定 + worklist 压缩 + age/slot_pos/meta 刷新。
  crate::profiler::gpu_compute_pass(
    &mut profiler,
    ctx.command_encoder(),
    "gate_ddgi_sort",
    |pass| {
      pass.set_pipeline(p_sort);
      set_bgs(pass, &bg4.0);
      pass.dispatch_workgroups(wg_slots, 1, 1);
    },
  );
  // seal：为 cast/collect 准备 per-LOD indirect args；只在需要阶段二时跑。
  if stage.run_active() {
    crate::profiler::gpu_compute_pass(
      &mut profiler,
      ctx.command_encoder(),
      "gate_ddgi_seal",
      |pass| {
        pass.set_pipeline(p_seal);
        set_bgs(pass, &bg4.0);
        if let Some(bg6) = bg6.as_ref() {
          pass.set_bind_group(6, &bg6.0, &[]);
        }
        pass.dispatch_workgroups(1, 1, 1);
      },
    );
  }
  // 阶段二 cast / 阶段三 collect：seal 已备好 per-LOD indirect args 与 rpp。
  if stage.run_cast()
    && let (Some(p_cast), Some(p_coll), Some(bg5)) = (
      pipeline_cache.get_compute_pipeline(pipes.cast),
      pipeline_cache.get_compute_pipeline(pipes.collect),
      bg5.as_ref(),
    )
  {
    // cast / collect 各一次间接 dispatch（覆盖全部 LOD）；args 段字节偏移取自 WESL 的
    // `DDGI_INDIR_CAST_BASE` / `DDGI_INDIR_COLL_BASE`。
    let c = ddgi_consts();
    let (cast_off, coll_off) = (c.args_cast_offset(), c.args_coll_offset());
    crate::profiler::gpu_compute_pass(
      &mut profiler,
      ctx.command_encoder(),
      "gate_ddgi_cast",
      |pass| {
        pass.set_pipeline(p_cast);
        set_bgs(pass, &bg4.0);
        pass.dispatch_workgroups_indirect(&gpu.args, cast_off);
      },
    );
    crate::profiler::gpu_compute_pass(
      &mut profiler,
      ctx.command_encoder(),
      "gate_ddgi_collect",
      |pass| {
        pass.set_pipeline(p_coll);
        set_bgs(pass, &bg4.0);
        pass.set_bind_group(5, &bg5.0, &[]);
        pass.dispatch_workgroups_indirect(&gpu.args, coll_off);
      },
    );
    // 本帧写过图集 → 下一帧采样侧翻转。
    gpu.parity ^= 1;
  }
}

fn extract_ddgi_settings(
  mut commands: bevy::ecs::system::Commands,
  stage: Option<bevy::render::Extract<bevy::ecs::system::Res<DdgiStage>>>,
  debug: Option<bevy::render::Extract<bevy::ecs::system::Res<DdgiDebugSettings>>>,
  world_aabb: Option<bevy::render::Extract<bevy::ecs::system::Res<DdgiWorldAabb>>>,
  lod0_chunks: Option<bevy::render::Extract<bevy::ecs::system::Res<DdgiLod0Chunks>>>,
) {
  let s = stage.map_or(DdgiStage::OFF, |s| s.0.min(DdgiStage::FULL));
  commands.insert_resource(DdgiStage(s));
  let dbg = debug.map_or_else(DdgiDebugSettings::default, |d| DdgiDebugSettings {
    mode: d.mode,
    gain: d.gain,
    probe_viz: d.probe_viz,
    probe_viz_lod: d.probe_viz_lod,
    depth_soft_k: d.depth_soft_k,
    far_ambient: d.far_ambient,
    gi_half_res: d.gi_half_res,
  });
  commands.insert_resource(dbg);
  commands.insert_resource(
    world_aabb.map_or_else(DdgiWorldAabb::default, |a| DdgiWorldAabb { min: a.min, max: a.max }),
  );
  commands.insert_resource(lod0_chunks.map_or_else(DdgiLod0Chunks::default, |c| c.clone()));
}

#[allow(clippy::too_many_arguments)]
fn prepare_ddgi(
  mut commands: bevy::ecs::system::Commands,
  device: bevy::ecs::system::Res<bevy::render::renderer::RenderDevice>,
  queue: bevy::ecs::system::Res<bevy::render::renderer::RenderQueue>,
  pipeline_cache: bevy::ecs::system::Res<bevy::render::render_resource::PipelineCache>,
  revision: Option<bevy::ecs::system::Res<crate::brickmap::upload::BrickMapRevision>>,
  dirty: Option<bevy::ecs::system::Res<crate::brickmap::upload::BrickMapDirty>>,
  stage: bevy::ecs::system::Res<DdgiStage>,
  dbg: bevy::ecs::system::Res<DdgiDebugSettings>,
  world: bevy::ecs::system::Res<DdgiWorldAabb>,
  lod0_chunks: Option<bevy::ecs::system::Res<DdgiLod0Chunks>>,
  aux: Option<bevy::ecs::system::Res<crate::brickmap::dda::AuxTexCache>>,
  mut gi_ph: bevy::ecs::system::ResMut<GiPlaceholder>,
  mut gpu: bevy::ecs::system::ResMut<DdgiGpu>,
) {
  // ---- 网格推导 ----
  // LOD1~3 锚定世界 AABB，LOD0 按 chunk 由 `DdgiChunkPool` 领固定 4096 槽的段；
  // 槽位段顺序恒为 LOD0 → LOD1 → LOD2 → LOD3。
  let base_grid = DdgiWorldGrid::from_world(world.min, world.max);
  let chunk_geom = DdgiChunkGeom::from_world(world.min, world.max);
  let lod0_chunks = lod0_chunks.map_or_else(Vec::new, |c| c.chunks.clone());

  // 探针池同步：只在 chunk 集或网格几何变化时做事；已有 chunk 的段基址保持不变（见 `DdgiChunkPool::sync`）。
  let chunks_changed = gpu.last_chunks != lod0_chunks;
  let geom_changed = gpu.pool.geom != chunk_geom;
  let pool_changed = if chunks_changed || geom_changed {
    let changed = gpu.pool.sync(chunk_geom, &lod0_chunks);
    gpu.last_chunks = lod0_chunks;
    changed
  } else {
    false
  };
  let lod0_slots = gpu.pool.lod0_slots();

  let mut grid = base_grid;
  grid.lod_slot_base[0] = 0;
  let mut base = lod0_slots;
  for lod in 1..DDGI_LODS as usize {
    grid.lod_slot_base[lod] = base;
    base += grid.lod_count(lod);
  }
  grid.total_slots = base;
  let total = grid.total_slots;
  let grid_changed = grid != gpu.grid || pool_changed;
  let num_chunks = chunk_geom.len();

  // ---- 推进网格/修订号状态（仅在本帧会跑 pass 时）----
  let rev = revision.map_or(0, |r| r.0);
  let will_run = stage.run_active() || dbg.probe_viz;
  let rev_changed = rev != gpu.last_revision;
  if will_run {
    gpu.grid = grid;
    gpu.total_slots = total;
    if grid_changed {
      // 网格变了就打一次实际数值：LOD0 的 dims 字段仍是空间盒，槽位数是池高水位。
      for (lod, &cell) in DDGI_LOD_CELL_SIZES.iter().enumerate() {
        let o = grid.lod_origins[lod];
        let d = grid.lod_dims[lod];
        let count = if lod == 0 { lod0_slots } else { grid.lod_count(lod) };
        bevy::log::info!(
          "DDGI LOD{lod}: cell={} origin=({},{},{}) dims=({},{},{}) slot_base={} count={}",
          cell,
          o.x,
          o.y,
          o.z,
          d.x,
          d.y,
          d.z,
          grid.lod_slot_base[lod],
          count,
        );
      }
      bevy::log::info!(
        "DDGI LOD0 chunk 池: chunks={}/{} lod0_slots={}（旧 AABB 规则网格 LOD0={}，差 {}）",
        gpu.pool.bases.iter().filter(|&&b| b != DDGI_CHUNK_NO_BASE).count(),
        num_chunks,
        lod0_slots,
        grid.lod_count(0),
        grid.lod_count(0) as i64 - lod0_slots as i64,
      );
      bevy::log::info!("DDGI 总槽位 = {total}（图集容量 409600）");
    }
    gpu.last_revision = rev;
  }
  // 相机滚动（grid 变化）/ 池变化 → 按「世界 cell 键」增量补烘；世界编辑 → 只失效脏区内的 cell。
  // `gpu.bake_pending` 让请求黏住到真正派发的那一帧。
  let need_bake = (grid_changed || rev_changed || gpu.bake_pending) && total > 0 && will_run;
  gpu.bake_pending = need_bake;

  // ---- 脏区：本帧上传的世界 voxel AABB（max 不含）----
  let (dirty_min, dirty_max, dirty_valid) = match dirty.as_ref() {
    Some(d) if d.full => (IVec3::splat(-1_000_000_000), IVec3::splat(1_000_000_000), 1.0f32),
    Some(d) if d.min_voxel != d.max_voxel => (d.min_voxel, d.max_voxel, 1.0f32),
    _ => (IVec3::ZERO, IVec3::ZERO, 0.0f32),
  };

  // ---- 缓冲容量 ----
  let s = total as u64;
  ensure_storage_buffer(&device, &queue, &mut gpu.cell, "ddgi_cell", s * 4);
  ensure_storage_buffer(&device, &queue, &mut gpu.meta, "ddgi_meta", s * 4);
  ensure_storage_buffer(&device, &queue, &mut gpu.slot_pos, "ddgi_slot_pos", s * 16);
  ensure_storage_buffer(&device, &queue, &mut gpu.worklist, "ddgi_worklist", s * 16);
  ensure_storage_buffer(&device, &queue, &mut gpu.cell_id, "ddgi_cell_id", s * 16);
  // `cell_slot`：使用计数前缀 + LOD0 的两张 chunk 表（布局见 `DdgiGpu::cell_slot`）；
  // 构建键（total / chunk 数 / LOD0 槽数 / 池 serial）变了才重建。
  let key = (total, num_chunks, lod0_slots, gpu.pool.serial);
  if gpu.cell_slot_key != Some(key) {
    let words = (total + num_chunks + lod0_slots) as usize;
    let mut buf: Vec<u8> = Vec::with_capacity(words * 4);
    // 计数前缀必须归零。
    buf.resize(total as usize * 4, 0);
    for l in 0..num_chunks {
      let b = gpu.pool.bases[l as usize];
      buf.extend_from_slice(&b.to_le_bytes());
    }
    let mut slot_chunk = vec![0u32; lod0_slots as usize];
    for l in 0..num_chunks {
      let b = gpu.pool.bases[l as usize];
      if b != DDGI_CHUNK_NO_BASE {
        // 反向表：该段内 4096 个局部槽位都属于 chunk `l`
        for k in 0..DDGI_CHUNK_LOD0_SLOTS {
          slot_chunk[(b + k) as usize] = l;
        }
      }
    }
    for v in slot_chunk.iter() {
      buf.extend_from_slice(&v.to_le_bytes());
    }
    gpu.cell_slot = zero_storage_buffer(&device, &queue, "ddgi_cell_slot", (words as u64) * 4);
    queue.write_buffer(&gpu.cell_slot, 0, &buf);
    gpu.cell_slot_key = Some(key);
  }
  ensure_storage_buffer(
    &device,
    &queue,
    &mut gpu.samples,
    "ddgi_samples",
    ddgi_consts().sample_bytes(total),
  );

  gpu.frame = gpu.frame.wrapping_add(1);

  // ---- uniform ----
  let mut u = DdgiUniform::default();
  for (lod, &cell) in DDGI_LOD_CELL_SIZES.iter().enumerate() {
    let d = gpu.grid.lod_dims[lod];
    let o = gpu.grid.lod_origins[lod];
    u.lods[lod] = DdgiLod {
      origin: IVec4::new(o.x, o.y, o.z, cell),
      dims: UVec4::new(d.x, d.y, d.z, gpu.grid.lod_slot_base[lod]),
    };
  }

  u.params = Vec4::new(gpu.frame as f32, dbg.mode, dbg.gain, 0.0);
  u.misc = Vec4::new(
    if stage.shade_gi() { 1.0 } else { 0.0 },
    gpu.total_slots as f32,
    // z = Chebyshev std 信任系数（见 DdgiDebugSettings.depth_soft_k）
    dbg.depth_soft_k,
    // w = 级联覆盖外的天光兜底强度（见 DdgiDebugSettings.far_ambient）
    dbg.far_ambient,
  );
  u.dirty_min = Vec4::new(dirty_min.x as f32, dirty_min.y as f32, dirty_min.z as f32, dirty_valid);
  u.dirty_max = Vec4::new(dirty_max.x as f32, dirty_max.y as f32, dirty_max.z as f32, 0.0);
  // x = 1 时跳过逐角几何求交（`consts::DDGI_OCCLUSION` = false，量测 GPU 成本用，画面会漏光）；
  // y = 半分辨率 GI 仅在会跑 GI 着色时生效。
  u.flags = Vec4::new(
    if crate::consts::DDGI_OCCLUSION { 0.0 } else { 1.0 },
    if dbg.gi_half_res && stage.shade_gi() { 1.0 } else { 0.0 },
    0.0,
    0.0,
  );
  // LOD0 的 chunk 段编址（只有 lod==0 读它）；WGSL 用 `misc.y` + 这里的维度定位 cell_slot 尾部两张表。
  u.chunk = DdgiChunkUniform {
    origin: IVec4::new(
      chunk_geom.origin.x,
      chunk_geom.origin.y,
      chunk_geom.origin.z,
      DDGI_CHUNK_LOD0_AXIS,
    ),
    dims: UVec4::new(chunk_geom.dims.x, chunk_geom.dims.y, chunk_geom.dims.z, lod0_slots),
  };
  *gpu.uniform.get_mut() = u;
  gpu.uniform.write_buffer(&device, &queue);

  // 每帧清零活跃探针计数器段；indirect args 由 seal 当帧覆写。
  queue.write_buffer(
    &gpu.indirect,
    ddgi_consts().counter_clear_offset(),
    &vec![0u8; ddgi_consts().counter_clear_bytes() as usize],
  );

  // ---- BG4（图集采样侧 + 各 pass 共用缓冲）----
  // `p` = 采样侧（上一帧 collect 写入的那一半）；写入侧 `1-p` 只给 collect（BG5）。
  use bevy::render::render_resource::BindGroupEntries;
  let p = gpu.parity & 1;
  let bg4_layout = pipeline_cache.get_bind_group_layout(&ddgi_bg4_layout());
  let bg4 = device.create_bind_group(
    None,
    &bg4_layout,
    &BindGroupEntries::sequential((
      &gpu.uniform,
      &gpu.irr[p].view,
      &gpu.depth[p].view,
      gpu.cell.as_entire_binding(),
      gpu.meta.as_entire_binding(),
      gpu.indirect.as_entire_binding(),
      gpu.worklist.as_entire_binding(),
      gpu.slot_pos.as_entire_binding(),
      gpu.cell_id.as_entire_binding(),
      gpu.samples.as_entire_binding(),
      gpu.cell_slot.as_entire_binding(),
    )),
  );
  commands.insert_resource(DdgiBg4(bg4));

  // ---- BG5（collect 图集写入侧 + 半分辨率 GI 写入侧）----
  // GI 视图来自 `crate::brickmap::dda::AuxTexCache`；未就绪时用 1×1 占位纹理（bind group 必须给全条目）。
  let bg5_layout = pipeline_cache.get_bind_group_layout(&ddgi_bg5_layout());
  let (gi_view, gi_cov_view) = match aux.as_ref().and_then(|a| a.gi_write_views()) {
    Some(v) => v,
    None => gi_ph.views(&device),
  };
  let bg5 = device.create_bind_group(
    None,
    &bg5_layout,
    &BindGroupEntries::sequential((
      &gpu.irr[1 - p].view,
      &gpu.depth[1 - p].view,
      gi_view,
      gi_cov_view,
    )),
  );
  commands.insert_resource(DdgiBg5(bg5));

  // ---- BG6（seal 的 dispatch 参数写入侧）----
  let bg6_layout = pipeline_cache.get_bind_group_layout(&ddgi_bg6_layout());
  let bg6 = device.create_bind_group(
    None,
    &bg6_layout,
    &BindGroupEntries::single(gpu.args.as_entire_binding()),
  );
  commands.insert_resource(DdgiBg6(bg6));
  commands.insert_resource(DdgiBakeThisFrame(need_bake));
}
