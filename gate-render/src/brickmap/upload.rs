//! 砖块图上传通道：主世界与物体同一路径（dirty → VolumesBuilder → UploadSnapshot → GPU）。
//! 阶段：`RenderStartup` 建占位 buffers；`ExtractSchedule` 按预算 drain 脏 chunk 构建 snapshot；
//! `PrepareResources` 整块或按脏字节区间写 GPU；binding limit < 1GB 时退化为每帧全量上传。

use bevy::{
  log::{debug, info, warn},
  prelude::*,
  render::{
    Extract, ExtractSchedule, Render, RenderApp, RenderStartup, RenderSystems,
    render_resource::*,
    renderer::{RenderDevice, RenderQueue},
  },
};

use super::builder::{VolumesBuilder, VolumesSnapshot};
use super::wire::{
  BrickMapGlobals, CHUNK_COMP_WORDS, GridDesc, MARCH_MASK_WORDS, MaterialAsset, TREE_BASE,
  march_mask_lut_words,
};
use crate::pbr_texture::{METAL_DEMO_ID, PbrTextureSet, build_material_asset_table};
use glam::{IVec3, UVec3};

#[derive(Debug, Clone, Copy)]
pub struct BindingLimits {
  pub max_storage_buffer_binding_size: u64,
}
impl BindingLimits {
  pub fn probe(device: &RenderDevice) -> Self {
    Self { max_storage_buffer_binding_size: device.limits().max_storage_buffer_binding_size }
  }
  pub fn force_multi(&self) -> bool {
    self.max_storage_buffer_binding_size < crate::brickmap::consts::SINGLE_THRESHOLD_BYTES
  }
}

#[derive(Debug, Clone)]
pub enum BufferLayout {
  Single,
  Multi { node_slices: usize },
}
impl BufferLayout {
  pub fn from_limits(limits: &BindingLimits) -> Self {
    if limits.force_multi() {
      let per = crate::brickmap::consts::BUFFER_SLICE_TARGET;
      let left = limits
        .max_storage_buffer_binding_size
        .min(per)
        .max(crate::brickmap::consts::BUFFER_SLICE_MIN);
      let node_slices = ((TREE_BASE as u64).saturating_add(per) / left) as usize;
      Self::Multi { node_slices: node_slices.max(1) }
    } else {
      Self::Single
    }
  }
}

#[derive(Resource)]
pub struct VoxelScene {
  pub volumes: gate_voxel::Volumes,
  pub demo_force_full_rebuild: bool,
  /// 本帧待上传的改动是否**全部**来自"被实体完全包围"的笔触（⇒ 可见几何未变），且是**唯一**待上传改动。
  ///
  /// 由笔触在落笔那一刻判定并写入（`gate-app`：`stroke_hidden` 判"被包围"，并要求其余 dirty 队列为空），
  /// `extract` 消费：命中时只上传数据、不报 [`BrickMapDirty`] 盒 ⇒ GI 不必丢弃时域历史、`gi_sec_slots`
  /// 也不必整表失效（见 `gi::prepare_gi`）。
  ///
  /// CONSTRAINT: 每个写世界的路径都要**重写**它（笔触两条路径 + 自测），否则会残留上一次的值 ——
  /// 残留 true 会让一次真正可见的编辑被当作不可见。全量重建走 `full`，与它无关。
  pub interior_only_edit: bool,
  /// 是否有一笔普通笔触正在**跨帧**推进（`gate-app` 的 `ActiveStroke` 还有剩余块）。
  /// `extract` 用它判定"现在能不能压实树区"：压实要重传搬动过的树段，塞进笔触中途就是一次可见卡顿
  /// （见 `BrickMapBuilder::compact`）。
  pub edit_in_flight: bool,
}

#[derive(Resource, Clone, Debug)]
pub struct UploadBudget {
  pub max_bytes_per_frame: usize,
  pub incremental: bool,
}
impl Default for UploadBudget {
  fn default() -> Self {
    Self { max_bytes_per_frame: crate::brickmap::consts::UPLOAD_BYTES_PER_FRAME, incremental: true }
  }
}

/// 体素世界修订号（render world）：每完成一次真实上传（prepare 消费到 snapshot）自增。
/// 供依赖体素数据的下游 GPU pass 判定「世界是否变了」。
#[derive(Resource, Default, Clone, Copy, Debug, PartialEq, Eq)]
pub struct BrickMapRevision(pub u64);

/// 本帧上传改动产生的**世界 voxel 脏盒**（闭开 `[lo, hi)`）。
/// 唯一消费者是 `gi::prepare_gi` 的「世界几何修订号」——它只需要回答「本帧世界变没变」，
/// 因此盒用世界 AABB 就够（旧版还带一个「失效余量」供世界空间 GI 缓存条目失效判定，
/// 那条缓存已整条删除，余量与它一起移除）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DirtyBox {
  pub lo: IVec3,
  pub hi: IVec3,
}

impl DirtyBox {
  /// 两盒是否重叠 —— 重叠就并成一盒，避免盒数被同一片区域的多次编辑撑爆。
  pub fn overlaps(&self, other: &Self) -> bool {
    // 闭开区间 ⇒ 用闭区间判重叠（并集只会更大 ⇒ 只会多报变化，不会漏报）。
    self.lo.cmple(other.hi).all() && other.lo.cmple(self.hi).all()
  }

  /// 并入另一盒（取并集）。
  pub fn union_with(&mut self, other: &Self) {
    self.lo = self.lo.min(other.lo);
    self.hi = self.hi.max(other.hi);
  }
}

/// 本帧上传实际改动的世界 voxel 范围（render world）。
/// `full = true` = 全量上传；否则 `boxes` 为逐 volume（主世界 + 物体）的改动盒，可能为空。
#[derive(Resource, Default, Clone, Debug)]
pub struct BrickMapDirty {
  pub full: bool,
  /// 本帧调色板版本发生变化（只改材质的编辑：换色 / 改粗糙度等）。
  /// palette 是共享的，改一个色号无法廉价定位受影响体素 ⇒ 当作「世界整体变了」上报。
  pub palette_changed: bool,
  pub boxes: Vec<DirtyBox>,
}

/// volume **局部** voxel AABB（闭开 `[lo, hi)`）→ 世界 voxel AABB。
/// 世界变换与 shader 一致：`world = pos + rot · (local · scale)`；取局部 AABB 八个角点的世界外包
/// ⇒ 旋转 / 缩放都保守（只会多报变化，不会漏报）。主世界（identity、scale = 1）恒等于 `lo/hi`。
pub fn world_dirty_box(t: gate_voxel::VolumeTransform, lo: IVec3, hi: IVec3) -> DirtyBox {
  let s = if t.scale.is_finite() && t.scale > 0.0 { t.scale } else { 1.0 };
  let mut mn = glam::Vec3::splat(f32::MAX);
  let mut mx = glam::Vec3::splat(f32::MIN);
  for &x in &[lo.x, hi.x] {
    for &y in &[lo.y, hi.y] {
      for &z in &[lo.z, hi.z] {
        let w = t.pos + t.rot * (glam::Vec3::new(x as f32, y as f32, z as f32) * s);
        mn = mn.min(w);
        mx = mx.max(w);
      }
    }
  }
  DirtyBox {
    lo: IVec3::new(mn.x.floor() as i32, mn.y.floor() as i32, mn.z.floor() as i32),
    hi: IVec3::new(mx.x.ceil() as i32, mx.y.ceil() as i32, mx.z.ceil() as i32),
  }
}

/// 主世界 Pending 资源：在主 world `Last` schedule 按预算 drain dirty，供只读提取。
/// `data_chunks` / `comp_chunks` 元素 = `(volume_idx, coord)`：主世界 = 0，物体 = 1..N。
#[derive(Resource, Default)]
pub struct MainPending {
  pub force_full: bool,
  /// `(volume_idx, coord, 该 chunk 自上次上传以来的**节点级**改动)`
  pub data_chunks: Vec<(usize, gate_voxel::ChunkCoord, gate_voxel::TreeDirty)>,
  pub comp_chunks: Vec<(usize, gate_voxel::ChunkCoord)>,
  /// 与 `data_chunks` 并行的编辑 AABB：`(volume_idx, coord, lo, hi)`，闭开世界 voxel 区间。
  pub data_aabbs: Vec<(usize, gate_voxel::ChunkCoord, IVec3, IVec3)>,
}

/// 在主 world `Last` 阶段按预算 drain dirty → `MainPending`，遍历所有 volume 附带 `volume_idx`。
/// 进入时清空 `data_chunks` / `comp_chunks` / `data_aabbs`；`force_full` 处理一帧后复位。
///
/// 每个被 drain 的 chunk 顺手 `take_dirty()` 带走它的节点级改动 —— 必须在**同一处**取，
/// 否则 wire 层不知道"这个 chunk 哪些节点动过"，只能整棵重传。
pub fn poll_pending(
  scene: Option<ResMut<VoxelScene>>,
  budget: Option<Res<UploadBudget>>,
  mut pending: ResMut<MainPending>,
) {
  let (Some(mut scene), Some(budget)) = (scene, budget) else {
    return;
  };
  pending.data_chunks.clear();
  pending.comp_chunks.clear();
  pending.data_aabbs.clear();
  pending.force_full = false;
  if scene.demo_force_full_rebuild {
    pending.force_full = true;
    scene.demo_force_full_rebuild = false;
  }
  let budget_n =
    (budget.max_bytes_per_frame / crate::brickmap::consts::PER_CHUNK_BYTES).clamp(1, 64);

  let mut total_data_backlog = 0usize;
  let mut total_comp_backlog = 0usize;
  for grid in scene.volumes.list.iter() {
    total_data_backlog = total_data_backlog.saturating_add(grid.dirty.data_dirty_count());
    total_comp_backlog = total_comp_backlog.saturating_add(grid.dirty.comp_dirty_count());
  }

  let (data_n, comp_n) = if total_data_backlog > budget_n * 3 {
    (total_data_backlog, total_comp_backlog.max(budget_n))
  } else {
    (budget_n, budget_n.min(total_comp_backlog.max(1)))
  };

  for (vol_idx, grid) in scene.volumes.list.iter_mut().enumerate() {
    for c in grid.dirty.drain_data_budget(data_n) {
      let dirty = grid.chunk_mut(c).map(|t| t.take_dirty()).unwrap_or_default();
      pending.data_chunks.push((vol_idx, c, dirty));

      if let Some((lo, hi)) = grid.take_edit_aabb(c) {
        pending.data_aabbs.push((vol_idx, c, lo, hi));
      }
    }
    for c in grid.dirty.drain_comp_budget(comp_n) {
      pending.comp_chunks.push((vol_idx, c));
    }
  }
}

/// 主 world `Last`：递减转储请求。必须与 `poll_pending` 同阶段（都在同帧 ExtractSchedule 之前跑），
/// 这样"置位 → 自减 → 提取 → 转储"的先后在同帧内是确定的。
fn tick_voxel_dump_request(mut req: ResMut<VoxelDumpRequest>) {
  req.pending = req.pending.saturating_sub(1);
}

/// **数据转储请求**（菜单「游戏/世界/数据转储」）：主 world 置位 → 提取进 render world →
/// render 侧把 CPU 与 GPU 两份体素数据写进 `logs/`（见 [`dump_voxel_buffers`]）。
///
/// 为什么是**帧计数**而不是 bool：`ExtractResourcePlugin` 只在主 world 资源"有变化"时拷贝，
/// 而置位（菜单观察者，Update）与提取（帧尾 ExtractSchedule）之间还隔着 `Last` 的自减
/// ⇒ 用 [`Self::ARMED_FRAMES`] 帧的窗口保证 render 侧至少看见一次非零，随后自动归零
/// （点一次按钮 = 恰好转储一次）。
#[derive(
  Resource, Default, Clone, Copy, Debug, bevy::render::extract_resource::ExtractResource,
)]
#[extract_app(bevy::render::RenderApp)]
pub struct VoxelDumpRequest {
  /// 剩余待转储帧数（0 = 不转储）
  pub pending: u8,
}

impl VoxelDumpRequest {
  /// 置位后的存活帧数（≥ 2：跨一次 `Last` 自减后提取仍能看到非零）
  const ARMED_FRAMES: u8 = 2;

  /// 请求一次转储（菜单按钮的唯一入口）。
  pub fn arm(&mut self) {
    self.pending = Self::ARMED_FRAMES;
  }
}

/// ExtractSchedule 用的 CPU builder / pending 状态（render world resource）
/// `builder: Option<VolumesBuilder>` 持有 `Vec<BrickMapBuilder>`；pending chunks 带 volume_idx +
/// 该 chunk 的节点级改动（`TreeDirty`）。
#[derive(Resource, Default)]
pub struct BuilderMirror {
  pub builder: Option<VolumesBuilder>,
  pub pending_full: bool,
  /// `(volume_idx, coord, 节点级改动)`
  pub pending_data_chunks: Vec<(usize, gate_voxel::ChunkCoord, gate_voxel::TreeDirty)>,
  /// 与 `pending_data_chunks` 并行的编辑 AABB（`(volume_idx, coord, lo, hi)`）
  pub pending_data_aabbs: Vec<(usize, gate_voxel::ChunkCoord, IVec3, IVec3)>,
  /// 各 volume 调色板上次同步的写版本（判断「只改材质」是否需上传）。
  pub palette_versions: Vec<u64>,
  pub pending_comp_chunks: Vec<(usize, gate_voxel::ChunkCoord)>,
}

/// ExtractSchedule 产出 → PrepareResources 消费（render world resource）。
/// full 模式带整块 `b_struct`/`b_palette`；incremental 模式走 `struct_blobs`/`palette_blobs` 脏块。
#[derive(Resource, Clone)]
pub struct UploadSnapshot {
  pub volumes: VolumesSnapshot,
  pub state_bytes: Vec<u8>,
  pub comp_chunks: usize,
}

/// 上传 CPU 耗时样本（render world 资源，由 prepare 每帧覆盖）。
/// render→main 同步走 [`UploadCpuSampleChannel`]；GPU 拷贝发生在 submit 时，不计入此值。
#[derive(Resource, Clone, Copy, Debug, Default)]
pub struct UploadCpuSample {
  pub cpu_ms: f32,
  pub generation: u64,
}

/// render↔main 共享通道（`Arc<Mutex>`）。
/// prepare（render world, PrepareResources）写入；sync_gpu_timings（main world, Update）读出。
#[derive(Resource, Clone, Debug, Default)]
pub struct UploadCpuSampleChannel(pub std::sync::Arc<std::sync::Mutex<Option<UploadCpuSample>>>);

/// GPU 资源（render world）：统一 struct/leaves/palette/comp/state + grid_descs + globals。
/// `leaves` 存方向可达掩码 LUT（BG1 binding(1) 占位）；`grid_descs_count` 为有效条目数。
/// 另含 MT2-2 的两项**全局**（非 per-volume）资源：材质资产表 buffer 与 PBR 贴图数组的占位视图。
#[derive(Resource)]
pub struct GpuBrickMap {
  pub struct_buf: Buffer,
  pub leaves: Buffer,
  pub palette: Buffer,
  pub comp: Buffer,
  pub state: Buffer,
  pub grid_descs_buf: Buffer,
  pub grid_descs_count: u32,
  pub globals: UniformBuffer<BrickMapGlobals>,
  /// 主世界 chunk 窗口（chunk 单位）CPU 副本
  pub main_window_origin: IVec3,
  pub main_window_dims: UVec3,
  /// GI 缓冲（`gi_tex`）的**线性采样器**（BG1 binding 4，ClampToEdge ×3 + mipmap_filter = Nearest）。
  pub light_sampler: Sampler,
  /// **全局材质资产表**（BG1 binding 5）：storage buffer，`MATERIAL_ASSET_SLOTS × 32B`（当前 = 32KB）。
  /// **所有 volume 共用一张**（palette 的 PBR 变体里 `asset: u16` 是全局下标）。尺寸在
  /// [`init_empty_gpu`] 就按 WESL 常量定死（占位即最终尺寸，不需要扩容逻辑），
  /// 内容由 [`prepare`] 全量写一次（静态默认集，没有任何写入方 ⇒ 不需要增量路径）。
  pub material_assets: Buffer,
  /// 资产表内容是否已上传（一次性）：`false` = 仍是零初始化占位（贴图集还没就绪）。
  pub material_assets_uploaded: bool,
  /// PBR 贴图数组的**占位**（1×1×1 层，视图显式声明 `D2Array`）：贴图集 / `GpuImage` 未就绪时
  /// BG1 binding 6/7 绑它们 ⇒ **任何时刻都可绑定，绝不 panic**。
  /// 视图必须声明 `D2Array`：默认视图是单层 `D2`，拿去绑 `texture_2d_array` 会被 wgpu 拒（MT2-1 的坑）。
  pub pbr_albedo_rough_tex: Texture,
  pub pbr_albedo_rough_view: TextureView,
  pub pbr_metal_tex: Texture,
  pub pbr_metal_view: TextureView,
  /// **PBR 贴图专用采样器**（BG1 binding 8，MT2-3）：权威 desc 在
  /// [`crate::pbr_texture::create_pbr_sampler`]（Repeat×3 / Linear mag,min,mipmap / anisotropy ≤ 8 /
  /// lod_max 覆盖到 mip 链底）。**与 `light_sampler` 分开**：那一份是 ClampToEdge、mipmap_filter 为
  /// Nearest（GI 缓冲没有 mip 链），共用会连带改掉 GI 的采样行为。占位与真身共用这**一个**实例。
  pub pbr_sampler: Sampler,
}

fn init_empty_gpu(device: Res<RenderDevice>, mut commands: Commands) {
  // COPY_SRC：扩容时前缀拷贝（`copy_buffer_to_buffer`）必需。
  let make = |label: &str| -> Buffer {
    device.create_buffer(&BufferDescriptor {
      label: Some(label),
      size: 4,
      usage: BufferUsages::COPY_DST | BufferUsages::COPY_SRC | BufferUsages::STORAGE,
      mapped_at_creation: false,
    })
  };
  let globals = UniformBuffer::<BrickMapGlobals>::default();

  // GI 缓冲（`gi_tex`）的采样器：ClampToEdge ×3 + mipmap_filter = Nearest（那张纹理没有 mip 链）。
  let light_sampler = device.create_sampler(&SamplerDescriptor {
    label: Some("gate_light_sampler"),
    address_mode_u: AddressMode::ClampToEdge,
    address_mode_v: AddressMode::ClampToEdge,
    address_mode_w: AddressMode::ClampToEdge,
    mag_filter: FilterMode::Linear,
    min_filter: FilterMode::Linear,
    mipmap_filter: MipmapFilterMode::Nearest,
    ..Default::default()
  });

  // ---- MT2-2：全局材质资产表（占位即最终尺寸）+ PBR 贴图数组的占位视图 ----
  // 资产表：`MATERIAL_ASSET_SLOTS × 32B`（当前 = 1024 × 32B = 32KB）。表是**静态默认集**，
  // 尺寸由 WESL 权威常量定死 ⇒ 一开始就按满尺寸开，prepare 只需 `write_buffer` 写一次内容，
  // 期间（贴图集还没就绪）这份零初始化 buffer 就是合法占位（长度已够）。
  let asset_slots = crate::wesl_consts::material_consts().material_asset_slots;
  let material_assets = device.create_buffer(&BufferDescriptor {
    label: Some("gate_material_assets"),
    size: asset_slots as u64 * std::mem::size_of::<MaterialAsset>() as u64,
    usage: BufferUsages::COPY_DST | BufferUsages::COPY_SRC | BufferUsages::STORAGE,
    mapped_at_creation: false,
  });
  // 贴图数组占位（1×1×1 层）：`depth_or_array_layers = 1` + 视图 `D2Array` —— 两者缺一不可
  // （MT2-1/MT2-1c 的坑：默认视图是单层 `D2`，绑 `texture_2d_array` 会被 wgpu 拒）。
  let make_pbr_placeholder = |label: &str, format: TextureFormat| -> (Texture, TextureView) {
    let tex = device.create_texture(&TextureDescriptor {
      label: Some(label),
      size: Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
      mip_level_count: 1,
      sample_count: 1,
      dimension: TextureDimension::D2,
      format,
      usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
      view_formats: &[],
    });
    let view = tex.create_view(&TextureViewDescriptor {
      label: Some(label),
      dimension: Some(TextureViewDimension::D2Array),
      ..Default::default()
    });
    (tex, view)
  };
  // 格式与真身一致（`Rgba8Unorm` / `R8Unorm`）⇒ 占位与真身的采样类型都是可过滤 float，layout 通用。
  let (pbr_albedo_rough_tex, pbr_albedo_rough_view) =
    make_pbr_placeholder("gate_pbr_albedo_rough_placeholder", TextureFormat::Rgba8Unorm);
  let (pbr_metal_tex, pbr_metal_view) =
    make_pbr_placeholder("gate_pbr_metal_placeholder", TextureFormat::R8Unorm);

  // ---- MT2-3：PBR 贴图专用采样器（BG1 binding 8）----
  // desc 的权威在 `pbr_texture::create_pbr_sampler`（占位与真身共用这一个实例）。
  // 这里顺手把**采样策略 + mip 层数**打进启动日志：验收项"远处不闪 / 近处不糊"只能人工看，
  // 但"mip 链存在、采样器吃到 mip"这件事必须有据可查。
  // anisotropy = 8 不会 panic：wgpu 30 的 anisotropy 不再是 `Features`（已移到 DownlevelFlags，
  // 不支持时 wgpu-core 静默钳到 1），唯一的硬校验是 `>= 1`。
  let pbr_sampler = crate::pbr_texture::create_pbr_sampler(&device);
  info!(
    target: "gate",
    "PBR 采样器（BG1 binding 8）: Repeat×3 / Linear(mag,min,mipmap) / anisotropy_clamp = {} \
     / lod_max_clamp = {}；mip 链 = {} 层（GPU_TEX_SIZE = {} → 1×1，CPU 盒式平均）；\
     与 light_samp（ClampToEdge / mipmap Nearest / 无 mip）分属两个 sampler",
    crate::pbr_texture::PBR_ANISOTROPY_CLAMP,
    (crate::pbr_texture::PBR_MIP_LEVELS_MAX - 1) as f32,
    crate::pbr_texture::PBR_MIP_LEVELS_MAX,
    crate::pbr_texture::GPU_TEX_SIZE,
  );

  // ---- MT8-3：反射缓存的乒乓双缓冲（BG1 binding 10/11）已随反射缓存一起删除 ----
  // 那两块 buffer（`REFL_CACHE_SLOTS × 32B` = 合计 8 MiB）与 `refl_consts()` / `ReflEntry`
  // 都不再存在（用户实测判为负优化）。BG1 的 binding 号现在到 8 为止。

  commands.insert_resource(GpuBrickMap {
    struct_buf: make("gate_struct"),
    leaves: make("gate_leaves"),
    palette: make("gate_palette"),
    comp: make("gate_comp"),
    state: make("gate_state"),
    grid_descs_buf: make("gate_grid_descs"),
    grid_descs_count: 0,
    globals,
    main_window_origin: IVec3::ZERO,
    main_window_dims: UVec3::ZERO,
    light_sampler,
    material_assets,
    material_assets_uploaded: false,
    pbr_albedo_rough_tex,
    pbr_albedo_rough_view,
    pbr_metal_tex,
    pbr_metal_view,
    pbr_sampler,
  });
}

// ExtractSchedule（render sub-app）：只读访问主 world 资源，CPU 构建 snapshot。
// `Extract<T>` 的 T 必须 ReadOnlySystemParam，全部用 `Res<T>`。

#[allow(clippy::too_many_arguments)]
fn extract(
  mut commands: Commands,
  scene: Option<Extract<Res<VoxelScene>>>,
  budget: Option<Extract<Res<UploadBudget>>>,
  main_pending: Option<Extract<Res<MainPending>>>,
  mut mirror: ResMut<BuilderMirror>,
) {
  let (Some(scene), Some(budget), Some(main_pending)) = (scene, budget, main_pending) else {
    return;
  };
  if main_pending.force_full {
    mirror.pending_full = true;
  }
  mirror.pending_data_chunks.extend(main_pending.data_chunks.iter().cloned());
  mirror.pending_data_aabbs.extend(main_pending.data_aabbs.iter().copied());
  mirror.pending_comp_chunks.extend(main_pending.comp_chunks.iter().copied());

  let first = mirror.builder.is_none();
  let pending_full = std::mem::take(&mut mirror.pending_full);
  let mut pending_data: Vec<(usize, gate_voxel::ChunkCoord, gate_voxel::TreeDirty)> =
    std::mem::take(&mut mirror.pending_data_chunks);
  let pending_aabbs: Vec<(usize, gate_voxel::ChunkCoord, IVec3, IVec3)> =
    std::mem::take(&mut mirror.pending_data_aabbs);
  let _pending_comp: Vec<(usize, gate_voxel::ChunkCoord)> =
    std::mem::take(&mut mirror.pending_comp_chunks);
  let need_full = first || pending_full || !budget.incremental;

  let palette_dirty = scene
    .volumes
    .list
    .iter()
    .enumerate()
    .any(|(i, g)| mirror.palette_versions.get(i).copied() != Some(g.palette().version()));
  let dirty_any =
    need_full || !pending_data.is_empty() || !_pending_comp.is_empty() || palette_dirty;

  // 本帧上传改动范围（世界 voxel 脏盒）：全量上传 = full，增量 = 逐 volume（主世界 + 物体）一个盒。
  // 主世界与物体走同一路径：物体的局部 AABB 经 transform 转成世界 AABB，余量按 scale 放大（见 `world_dirty_box`）。
  // 只改材质（palette 版本变化）时没有有意义的 AABB，走 `palette_changed`。
  let mut dirty_aabb = BrickMapDirty { palette_changed: palette_dirty, ..Default::default() };
  if dirty_any {
    if need_full {
      dirty_aabb.full = true;
    } else if scene.interior_only_edit {
      // 本帧的脏数据全部来自"被实体完全包围"的笔触 ⇒ 可见几何（含 GI 二次命中的面）一个都没变：
      // 数据照常上传，但不报改动盒 ⇒ `prepare_gi` 认为世界没变（GI 复用上一帧历史、面缓存不失效）。
      // 只改材质的编辑仍由 `palette_changed` 上报（见 `VoxelScene::interior_only_edit`）。
    } else {
      // 逐 volume 先并集局部 AABB（同一 volume 的多个脏 chunk 合成一盒 ⇒ 主世界与旧版单盒等价）。
      let mut acc: Vec<(usize, IVec3, IVec3)> = Vec::new();
      for (vol_idx, c) in
        pending_data.iter().map(|(v, c, _)| (v, c)).chain(_pending_comp.iter().map(|(v, c)| (v, c)))
      {
        let (cl, ch) = match pending_aabbs.iter().find(|(v, cc, ..)| v == vol_idx && cc == c) {
          Some((_, _, alo, ahi)) => (*alo, *ahi),
          None => {
            let cl = c.0 * gate_voxel::CHUNK_SIZE;
            (cl, cl + IVec3::splat(gate_voxel::CHUNK_SIZE))
          }
        };
        match acc.iter_mut().find(|(v, ..)| *v == *vol_idx) {
          Some((_, lo, hi)) => {
            *lo = (*lo).min(cl);
            *hi = (*hi).max(ch);
          }
          None => acc.push((*vol_idx, cl, ch)),
        }
      }
      dirty_aabb.boxes = acc
        .into_iter()
        .filter_map(|(vol_idx, lo, hi)| {
          scene.volumes.list.get(vol_idx).map(|g| world_dirty_box(g.transform(), lo, hi))
        })
        .collect();
    }
  }
  commands.insert_resource(dirty_aabb);

  let volumes_ref = &scene.volumes;

  if !dirty_any {
    return;
  }
  let builder = mirror.builder.get_or_insert_with(|| VolumesBuilder::new_unbuilt(volumes_ref));
  if need_full {
    *builder = VolumesBuilder::build_full(volumes_ref);
    pending_data.clear();
  } else {
    // 压实的时机：必须"安静"（没有笔触在跑、且这一批就是全部待上传的改动），否则会把一次
    // 大块重传塞进笔触中途 —— 见 `BrickMapBuilder::compact`。
    let quiet = !scene.edit_in_flight
      && scene
        .volumes
        .list
        .iter()
        .all(|g| g.dirty.data_dirty_count() == 0 && g.dirty.comp_dirty_count() == 0);
    builder.sync(volumes_ref);
    for (vol_idx, c, dirty) in pending_data.drain(..) {
      builder.update_chunk(volumes_ref, vol_idx, c, &dirty, quiet);
    }
  }

  builder.sync_palettes(volumes_ref);
  let snapshot = builder.snapshot();

  mirror.palette_versions = scene.volumes.list.iter().map(|g| g.palette().version()).collect();

  let state_bytes = volumes_ref.main().state_table_bytes().to_vec();
  let comp_chunks = volumes_ref.main().comp_layer().len();
  commands.insert_resource(UploadSnapshot { volumes: snapshot, state_bytes, comp_chunks });
}

fn u8_of_u32(w: &[u32]) -> &[u8] {
  unsafe { std::slice::from_raw_parts(w.as_ptr() as *const u8, w.len() * 4) }
}

/// GridDesc 数组 → u8 字节视图（`#[repr(C)]` + 144B/entry，可直接 cast）
fn u8_of_grid_descs(descs: &[GridDesc]) -> &[u8] {
  unsafe { std::slice::from_raw_parts(descs.as_ptr() as *const u8, std::mem::size_of_val(descs)) }
}

/// MaterialAsset 数组 → u8 字节视图（`#[repr(C)]`、8 × u32 = 32B/条，可直接 cast）
fn u8_of_material_assets(assets: &[MaterialAsset]) -> &[u8] {
  unsafe { std::slice::from_raw_parts(assets.as_ptr() as *const u8, std::mem::size_of_val(assets)) }
}

/// GPU buffer 扩容尺寸策略（纯函数）：need ≥ 扩容阈值 → 32MB 对齐；否则 2× 增长（下限 64KB）。
fn grow_size(cap: u64, need: u64) -> u64 {
  let big = crate::brickmap::consts::BUFFER_GROW_BIG;
  let reserve = crate::brickmap::consts::BUFFER_GROW_RESERVE;
  if need >= big { need.div_ceil(reserve) * reserve } else { need.max(cap * 2).max(65536) }
}

/// 保证 buffer 能容纳 `bytes`，扩容时保留旧内容。
/// `prefix_valid=true` 时旧前缀走 GPU-GPU copy、仅 write 新增尾部；false 时整份一次 DMA。
fn ensure_with_copy(
  device: &RenderDevice,
  queue: &RenderQueue,
  cur: &mut Buffer,
  label: &str,
  bytes: &[u8],
  prefix_valid: bool,
) {
  let cap = cur.size();
  let need = bytes.len() as u64;
  if cap >= need {
    return;
  }
  let new_size = grow_size(cap, need);
  let new_buf = device.create_buffer(&BufferDescriptor {
    label: Some(label),
    size: new_size,
    // COPY_SRC：本 buffer 扩容时作为前缀拷贝的源（必需）。
    usage: BufferUsages::COPY_DST | BufferUsages::COPY_SRC | BufferUsages::STORAGE,
    mapped_at_creation: false,
  });
  if prefix_valid && cap > 0 {
    // COPY_BUFFER_ALIGNMENT=4；cap 恒为 words×4 或初始 4B，天然对齐。
    let mut enc = device
      .create_command_encoder(&CommandEncoderDescriptor { label: Some("gate_grow_prefix_copy") });
    enc.copy_buffer_to_buffer(cur, 0, &new_buf, 0, cap);
    queue.submit([enc.finish()]);
    queue.write_buffer(&new_buf, cap, &bytes[cap as usize..need as usize]);
  } else if !bytes.is_empty() {
    queue.write_buffer(&new_buf, 0, bytes);
  }
  *cur = new_buf;
}

/// 整块写（full 模式 / state 等小 buffer）：GPU 旧内容不可信 → prefix_valid=false
fn write(device: &RenderDevice, queue: &RenderQueue, cur: &mut Buffer, label: &str, bytes: &[u8]) {
  ensure_with_copy(device, queue, cur, label, bytes, false);
  if !bytes.is_empty() {
    queue.write_buffer(cur, 0, bytes);
  }
}

/// 增量路径：保证 buffer 容量 ≥ `need_bytes`，扩容时新 buffer 前缀走 GPU-GPU 拷贝。
/// 新增尾部 [cap..need) 由调用方随后逐块 `write_buffer` 脏块覆盖。
fn ensure_capacity(
  device: &RenderDevice,
  queue: &RenderQueue,
  cur: &mut Buffer,
  label: &str,
  need_bytes: u64,
) {
  let cap = cur.size();
  if cap >= need_bytes {
    return;
  }
  let new_size = grow_size(cap, need_bytes);
  let new_buf = device.create_buffer(&BufferDescriptor {
    label: Some(label),
    size: new_size,
    // COPY_SRC：本 buffer 扩容时作为前缀拷贝的源（必需）。
    usage: BufferUsages::COPY_DST | BufferUsages::COPY_SRC | BufferUsages::STORAGE,
    mapped_at_creation: false,
  });
  if cap > 0 {
    // COPY_BUFFER_ALIGNMENT=4；cap 恒为 words×4 或初始 4B，天然对齐。
    let mut enc = device
      .create_command_encoder(&CommandEncoderDescriptor { label: Some("gate_grow_prefix_copy") });
    enc.copy_buffer_to_buffer(cur, 0, &new_buf, 0, cap);
    queue.submit([enc.finish()]);
  }
  *cur = new_buf;
}

/// 全局材质资产表：**静态默认集**，一次全量上传（MT2-2）。
///
/// **为什么不需要增量路径**：本表当前**没有任何写入方** —— 没有 UI / 编辑能改它（palette 的
/// `IS_PBR` 变体也还没有写侧），内容只是「贴图集槽位 + 中性电介质默认值」的一次性快照。
/// 增量（只写被改的那几条）要等 **MT7** 有真实材质编辑时才存在"改了一条"这回事。
///
/// 贴图集还没提取进 render world 时不写：buffer 保持 [`init_empty_gpu`] 的零初始化占位
/// （长度已够 `MATERIAL_ASSET_SLOTS × 32B`），BG1 照样绑得上 ⇒ 不 panic。
fn upload_material_assets(queue: &RenderQueue, set: Option<&PbrTextureSet>, gpu: &mut GpuBrickMap) {
  if gpu.material_assets_uploaded {
    return;
  }
  let Some(set) = set else {
    return; // 贴图集未就绪：等它（绑定侧的占位回退由 `dda.rs` 提示一条 info_once）
  };
  let table = build_material_asset_table(set);
  queue.write_buffer(&gpu.material_assets, 0, u8_of_material_assets(&table));
  gpu.material_assets_uploaded = true;

  let layers = set.layers();
  info!(
    target: "gate",
    "材质资产表: 槽 0..{layers}: albedo_slot = roughmetal_slot = i、emissive/transmission/height \
     = MATERIAL_SLOT_NONE；其余槽位五个 *_slot 全 NONE（标量回退 = 中性灰 albedo(sRGB 128) + roughness 0.5 \
     + metallic 0 + specular 1.0(不调制电介质 F0) + emissive/transmission 0 + IOR 1.50）",
  );
  match set.slot_of(METAL_DEMO_ID) {
    Some(i) => info!(
      target: "gate",
      "材质资产表: 槽 {i}（{METAL_DEMO_ID}）metallic = 255（临时 demo 默认值）",
    ),
    None => warn!(
      target: "gate",
      "材质资产表: 贴图集里没有 `{METAL_DEMO_ID}` ⇒ 表里无金属槽；确认 assets/textures/pbr/{METAL_DEMO_ID}/ 存在",
    ),
  }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare(
  mut commands: Commands,
  snapshot: Option<ResMut<UploadSnapshot>>,
  pbr_set: Option<Res<PbrTextureSet>>,
  mut gpu: ResMut<GpuBrickMap>,
  device: Res<RenderDevice>,
  queue: Res<RenderQueue>,
  sample_channel: Option<Res<UploadCpuSampleChannel>>,
  mut revision: ResMut<BrickMapRevision>,
) {
  upload_material_assets(&queue, pbr_set.as_deref(), &mut gpu);
  let Some(snap) = snapshot else { return };
  let t0 = std::time::Instant::now();

  static PROBED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
  let limits = BindingLimits::probe(&device);
  let layout = BufferLayout::from_limits(&limits);
  if !PROBED.swap(true, std::sync::atomic::Ordering::Relaxed) {
    let gb = limits.max_storage_buffer_binding_size as f64 / (1 << 30) as f64;
    info!(target: "gate",
      "PROBE: max_storage_buffer_binding_size = {gb:.2}GB → {}",
      match layout { BufferLayout::Single => "单 buffer 路径", BufferLayout::Multi{..} => "多 buffer fallback 路径" },
    );
    if limits.force_multi() {
      warn!(target: "gate",
        "GPU storage binding < 1GB ⇒ 退化到全量上传（单 tile 170MB 跨帧）"
      );
    }
  }

  let is_full = matches!(snap.volumes.mode_tag, "full" | "fallback_full");
  let comp_bytes = snap.comp_chunks * CHUNK_COMP_WORDS * 4;

  // 方向可达掩码 LUT → b_leaves，与 volume 无关。
  {
    let lut_need = (MARCH_MASK_WORDS * 4) as u64;
    if gpu.leaves.size() < lut_need {
      let lut = march_mask_lut_words();
      write(&device, &queue, &mut gpu.leaves, "gate_leaves", u8_of_u32(&lut));
    }
  }

  let (struct_tx_bytes, palette_tx_bytes, grid_descs_tx_bytes);

  if is_full {
    let struct_bytes = u8_of_u32(&snap.volumes.b_struct);
    let palette_bytes = u8_of_u32(&snap.volumes.b_palette);
    let grid_descs_bytes = u8_of_grid_descs(&snap.volumes.grid_descs);
    struct_tx_bytes = struct_bytes.len();
    palette_tx_bytes = palette_bytes.len();
    grid_descs_tx_bytes = grid_descs_bytes.len();
    write(&device, &queue, &mut gpu.struct_buf, "gate_struct", struct_bytes);
    write(&device, &queue, &mut gpu.palette, "gate_palette", palette_bytes);
    write(&device, &queue, &mut gpu.grid_descs_buf, "gate_grid_descs", grid_descs_bytes);
  } else {
    ensure_capacity(
      &device,
      &queue,
      &mut gpu.struct_buf,
      "gate_struct",
      snap.volumes.struct_total_bytes as u64,
    );
    ensure_capacity(
      &device,
      &queue,
      &mut gpu.palette,
      "gate_palette",
      snap.volumes.palette_total_bytes as u64,
    );
    let grid_descs_bytes = u8_of_grid_descs(&snap.volumes.grid_descs);
    ensure_capacity(
      &device,
      &queue,
      &mut gpu.grid_descs_buf,
      "gate_grid_descs",
      grid_descs_bytes.len() as u64,
    );

    let mut s_tx = 0usize;
    for (off, payload) in &snap.volumes.struct_blobs {
      queue.write_buffer(&gpu.struct_buf, *off as u64, payload);
      s_tx += payload.len();
    }
    let mut p_tx = 0usize;
    for (off, payload) in &snap.volumes.palette_blobs {
      queue.write_buffer(&gpu.palette, *off as u64, payload);
      p_tx += payload.len();
    }
    struct_tx_bytes = s_tx;
    palette_tx_bytes = p_tx;
    grid_descs_tx_bytes = 0;
  }

  write(&device, &queue, &mut gpu.state, "gate_state", &snap.state_bytes);

  {
    let placeholder = vec![0u8; comp_bytes.max(4)];

    ensure_with_copy(&device, &queue, &mut gpu.comp, "gate_comp", &placeholder, true);
  }

  let main_desc = snap.volumes.grid_descs.first().copied().unwrap_or_default();
  let globals = BrickMapGlobals {
    index_origin_x: main_desc.index_origin_x,
    index_origin_y: main_desc.index_origin_y,
    index_origin_z: main_desc.index_origin_z,
    index_origin_w: 0,
    index_dims_x: main_desc.index_dims_x,
    index_dims_y: main_desc.index_dims_y,
    index_dims_z: main_desc.index_dims_z,
    index_dims_w: 0,
    tile_count: main_desc.chunk_count,

    node_words: 0,
    node_free_words: 0,
    brick_slabs: 0,
    brick_free: 0,
    rejected_tiles: 0,
    grid_count: snap.volumes.grid_descs.len() as u32,
    _pad1: 0,
    _pad2: 0,
    _pad3: 0,
    _pad4: 0,
  };
  gpu.globals.set(globals);
  gpu.globals.write_buffer(&device, &queue);

  gpu.grid_descs_count = snap.volumes.grid_descs.len() as u32;

  gpu.main_window_origin =
    IVec3::new(main_desc.index_origin_x, main_desc.index_origin_y, main_desc.index_origin_z);
  gpu.main_window_dims =
    UVec3::new(main_desc.index_dims_x, main_desc.index_dims_y, main_desc.index_dims_z);

  revision.0 = revision.0.wrapping_add(1);

  let elapsed = t0.elapsed();
  let cpu_ms = elapsed.as_secs_f32() * 1000.0;
  let tx_bytes_total =
    (struct_tx_bytes + palette_tx_bytes + grid_descs_tx_bytes + snap.state_bytes.len()) as f64;
  let mb = tx_bytes_total / (1 << 20) as f64;

  let chunks_show = if is_full {
    snap.volumes.grid_descs.iter().map(|g| g.chunk_count as usize).sum::<usize>()
  } else {
    snap.volumes.dirty_chunks
  };
  debug!(target: "gate",
    "UPLOAD[{}]: bytes={:.2}MB (s {}KB,pal {}KB,gd {}KB,state 4KB), chunks={}, comp={}KB, elapsed={:?}",
    snap.volumes.mode_tag, mb,
    struct_tx_bytes / 1024, palette_tx_bytes / 1024,
    grid_descs_tx_bytes / 1024, chunks_show, comp_bytes / 1024, elapsed,
  );

  static SAMPLE_GEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
  let generation = SAMPLE_GEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
  let sample = UploadCpuSample { cpu_ms, generation };
  commands.insert_resource(sample);
  if let Some(ch) = sample_channel
    && let Ok(mut g) = ch.0.lock()
  {
    *g = Some(sample);
  }

  let vram = gpu.struct_buf.size()
    + gpu.leaves.size()
    + gpu.palette.size()
    + gpu.comp.size()
    + gpu.state.size()
    + gpu.grid_descs_buf.size();
  if vram > crate::brickmap::consts::VRAM_WARN_BYTES {
    bevy::log::warn!("GPU VRAM {vram}B 超旧预算线（仅提示）");
  }
  if !limits.force_multi() {
    debug_assert!(
      snap.volumes.struct_total_bytes as u64 <= limits.max_storage_buffer_binding_size,
      "b_struct {} bytes > binding limit {}",
      snap.volumes.struct_total_bytes,
      limits.max_storage_buffer_binding_size,
    );
  }
  debug!(target: "gate",
    "GpuBrickMap: struct_buf={}B leaves={}B palette={}B comp={}B state={}B grid_descs={}B(count={}) bind_group_ready=pending(P2.4)",
    gpu.struct_buf.size(), gpu.leaves.size(), gpu.palette.size(),
    gpu.comp.size(), gpu.state.size(), gpu.grid_descs_buf.size(), gpu.grid_descs_count,
  );

  // 消费完本帧 snapshot 必须移除。
  commands.remove_resource::<UploadSnapshot>();
}

/// 转储文件魔数（小端写盘 ⇒ 文件头 4 字节读出来是 `VOXD`）
const DUMP_MAGIC: u32 = 0x4458_4F56;
/// 转储布局版本（布局一改就 +1；2 = 根节点固定 64 槽指针表 + 块内节点 arena）
const DUMP_LAYOUT_VERSION: u32 = 2;
/// 转储头字数（随后是 `volume_count × 16` 的逐 volume 表）
const DUMP_HEADER_WORDS: usize = 16;
/// 逐 volume 表每条字数
const DUMP_VOLUME_WORDS: usize = 16;

/// **数据转储**（菜单「游戏/世界/数据转储」）：把 **CPU 侧 wire 状态**与 **GPU 上实际字节**各写一份
/// `.bin` 到 `logs/`。两份文件的**布局逐字节相同** ⇒ 健康时应当逐字节相等，**首个不等的字/字节
/// 就是"从 CPU 数据到 GPU 视线"的变形点**（mask / inline leaf / 调色板都在这两份里）。
///
/// 文件布局（小端；两份文件唯一差别是"段内容取自哪一侧"）：
/// ```text
/// 偏移            长度                  内容
/// 0               64B                   头 16 字：magic / 版本 / 各段总字数 / 主世界窗口与全局计数
/// 64B             64B × N               逐 volume 表 16 字 × N：tree_base / palette_base / 各段字数 /
///                                       窗口 origin+dims / chunk_count / node_words / node_free_words / rejected
/// 64+64N          struct_words × 4B     struct 段：chunk 窗口 + 各 chunk 树块（mask_lo/hi + node palette + leaf inline）
/// …               palette_words × 4B    palette 段：8B 材质条目（2 字/条）
/// …               leaves_words × 4B     leaves 段：方向可达掩码 LUT（2 字/u64 对）
/// ```
/// **段内顺序 = [`VolumesBuilder::snapshot`] 的拼接序**（物体在前、主世界在后）⇒ 表里的
/// `tree_base` / `palette_base` 与 shader 的寻址（`tree_base + rel`）逐字一致，可直接用它们定位任一 chunk。
///
/// 定位与解码（与 `wire.rs` / `brickmap.wesl` 的契约一致）：
/// - chunk 窗口字址 = `tree_base + rel.x + rel.y×64 + rel.z×64²`，`rel = chunk − 窗口 origin`
///   （`CHUNK_INDEX_CAP = 64`）；该字是 `entry`，`entry != 0` ⇒ 树块首 = `tree_base + entry − 1`
///   （= shader 的 `chunk_base` = 根节点地址）；
/// - 根节点固定 `[ROOT_WIRE_WORDS]` = 3 + 64 字（掩码增减不改根的字数 ⇒ 根永不搬迁），
///   其余节点是块内 arena 里的块（地址任意，`node_words` 不再等于"紧排字数"）；
/// - 每个节点 3 字 fixed：`mask_lo` / `mask_hi` /（低 16 位 = 统一色，0 = AIR；高 16 位恒 0 ——
///   旧「LOD 代表色」字段，无消费方，见 `chunk_tree.rs::pack_node_palette`），
///   随后**按 mask 位序密集**放"该位置 1"的子块偏移（值 = 相对**根节点地址**的字偏移）；`mask = 0`
///   的节点到此为止（整个 4^level 子块同色）；叶父层换成 32 字 inline（每字 2 个体素 × 16 位索引，
///   0 = AIR）。
///
/// CPU 侧取 `BuilderMirror.builder`（与上传同源的那份状态）；GPU 侧走**真实 readback**
/// （copy → staging → map），因此能验到"上传有没有写坏/写漏"。
/// 刻意不含 `comp` / `state`：那是元件/状态表，没有 mask 语义。
fn dump_voxel_buffers(
  request: Option<Res<VoxelDumpRequest>>,
  mirror: Option<Res<BuilderMirror>>,
  gpu: Option<Res<GpuBrickMap>>,
  device: Res<RenderDevice>,
  queue: Res<RenderQueue>,
) {
  if request.is_none_or(|r| r.pending == 0) {
    return;
  }
  let (Some(mirror), Some(gpu)) = (mirror, gpu) else {
    warn!("数据转储：render 侧资源未就绪 → 忽略");
    return;
  };
  let Some(builder) = mirror.builder.as_ref() else {
    warn!("数据转储：CPU builder 尚未建立（还没有过一次上传）→ 忽略");
    return;
  };
  let buffers = builder.volume_buffers();
  let n = buffers.len();
  if n == 0 {
    warn!("数据转储：没有 volume → 忽略");
    return;
  }

  // ---- 布局：与 `VolumesBuilder::snapshot` 的拼接序一致（物体在前、主世界在后）----
  let layout_order: Vec<usize> = (1..n).chain(std::iter::once(0)).collect();
  let mut tree_base = vec![0u32; n];
  let mut palette_base = vec![0u32; n];
  let (mut struct_words, mut palette_words) = (0usize, 0usize);
  for &i in &layout_order {
    tree_base[i] = struct_words as u32;
    palette_base[i] = palette_words as u32;
    struct_words += buffers[i].b_struct.len();
    palette_words += buffers[i].b_palette.len();
  }
  let leaves_words = MARCH_MASK_WORDS;

  // ---- 头 + 逐 volume 表（两份文件逐字相同的部分，都从 CPU 侧取）----
  let mut head: Vec<u32> = Vec::with_capacity(DUMP_HEADER_WORDS + DUMP_VOLUME_WORDS * n);
  {
    let m = &buffers[0].globals;
    head.extend([
      DUMP_MAGIC,
      DUMP_LAYOUT_VERSION,
      n as u32,
      struct_words as u32,
      palette_words as u32,
      leaves_words as u32,
      m.index_origin_x as u32,
      m.index_origin_y as u32,
      m.index_origin_z as u32,
      m.index_dims_x,
      m.index_dims_y,
      m.index_dims_z,
      m.tile_count,
      m.node_words,
      m.node_free_words,
      m.rejected_tiles,
    ]);
  }
  for i in 0..n {
    let g = &buffers[i].globals;
    head.extend([
      tree_base[i],
      palette_base[i],
      buffers[i].b_struct.len() as u32,
      buffers[i].b_palette.len() as u32,
      g.index_origin_x as u32,
      g.index_origin_y as u32,
      g.index_origin_z as u32,
      g.index_dims_x,
      g.index_dims_y,
      g.index_dims_z,
      g.tile_count,
      g.node_words,
      g.node_free_words,
      g.rejected_tiles,
      0,
      0,
    ]);
  }

  // ---- CPU 侧字节：builder 的 wire 状态（与上传同源）+ 静态 LUT ----
  let lut = march_mask_lut_words();
  let head_bytes = head.len() * 4;
  let struct_bytes = struct_words * 4;
  let palette_bytes = palette_words * 4;
  let leaves_bytes = leaves_words * 4;
  let total = head_bytes + struct_bytes + palette_bytes + leaves_bytes;
  let mut cpu: Vec<u8> = Vec::with_capacity(total);
  cpu.extend_from_slice(u8_of_u32(&head));
  for &i in &layout_order {
    cpu.extend_from_slice(u8_of_u32(&buffers[i].b_struct));
  }
  for &i in &layout_order {
    cpu.extend_from_slice(u8_of_u32(&buffers[i].b_palette));
  }
  cpu.extend_from_slice(u8_of_u32(&lut));

  // ---- GPU 侧字节：三块 buffer 的实际内容（readback）----
  // 先尺寸守卫：小于 CPU 布局说明两侧不同步，此时绝不发 copy（wgpu 校验失败会毒化 device）。
  let gpu_sizes = [
    ("struct", gpu.struct_buf.size(), struct_bytes as u64),
    ("palette", gpu.palette.size(), palette_bytes as u64),
    ("leaves", gpu.leaves.size(), leaves_bytes as u64),
  ];
  if let Some((label, have, need)) = gpu_sizes.iter().find(|(_, have, need)| have < need) {
    warn!(
      "数据转储：GPU {label} buffer 只有 {have}B < 布局 {need}B → 本次放弃（先看上传是否掉队）"
    );
    return;
  }
  // staging 只装**三个段**（不含头区）：头区是纯元数据，直接取 CPU 那份即可 ⇒ 不必让 GPU 侧
  // 也写一遍（`MAP_READ` buffer 上的 `write_buffer` 也能省掉），最终文件 = CPU 头 + GPU 段。
  let payload_bytes = total - head_bytes;
  let staging = device.create_buffer(&BufferDescriptor {
    label: Some("gate_voxel_dump_staging"),
    size: payload_bytes as u64,
    usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
    mapped_at_creation: false,
  });
  let mut enc = device
    .create_command_encoder(&CommandEncoderDescriptor { label: Some("gate_voxel_dump_readback") });
  enc.copy_buffer_to_buffer(&gpu.struct_buf, 0, &staging, 0, struct_bytes as u64);
  enc.copy_buffer_to_buffer(&gpu.palette, 0, &staging, struct_bytes as u64, palette_bytes as u64);
  enc.copy_buffer_to_buffer(
    &gpu.leaves,
    0,
    &staging,
    (struct_bytes + palette_bytes) as u64,
    leaves_bytes as u64,
  );
  // 本次 submit 顺带把本帧 `prepare` 里那些 `write_buffer` 落进 buffer ⇒ 转储内容是"当前帧的 GPU 状态"。
  queue.submit([enc.finish()]);

  let slice = staging.slice(..);
  let (tx, rx) = std::sync::mpsc::channel();
  device.map_buffer(&slice, MapMode::Read, move |r| {
    let _ = tx.send(r);
  });
  // 阻塞等这一轮 copy 完成：转储是离散的用户动作，**同步拿结果**比跨帧状态机简单得多。
  // 带上限（设备挂起时不至于把 app 冻死）。
  if let Err(e) = device.poll(PollType::wait_indefinitely()) {
    warn!("数据转储：等待 readback 失败 {e} → 本次放弃");
    staging.unmap();
    return;
  }
  match rx.recv_timeout(std::time::Duration::from_secs(5)) {
    Ok(Ok(())) => {}
    Ok(Err(e)) => {
      warn!("数据转储：readback 映射失败 {e} → 本次放弃");
      staging.unmap();
      return;
    }
    Err(e) => {
      warn!("数据转储：readback 超时/断线 {e} → 本次放弃");
      staging.unmap();
      return;
    }
  }
  // GPU 侧文件 = CPU 头区 + 刚 readback 的三个段（头区两文件逐字节相同 ⇒ 整份文件可直接对比）
  let mut gpu_bytes: Vec<u8> = Vec::with_capacity(total);
  gpu_bytes.extend_from_slice(&cpu[..head_bytes]);
  match slice.get_mapped_range() {
    Ok(view) => gpu_bytes.extend_from_slice(&view),
    Err(e) => {
      warn!("数据转储：读取映射区间失败 {e:?} → 本次放弃");
      staging.unmap();
      return;
    }
  }
  staging.unmap();

  // ---- 落盘 + 首异点 ----
  let dir = crate::paths::logs_dir();
  if let Err(e) = std::fs::create_dir_all(&dir) {
    warn!("数据转储：建目录失败 {}：{e}", dir.display());
    return;
  }
  let cpu_path = dir.join("voxel_dump_cpu.bin");
  let gpu_path = dir.join("voxel_dump_gpu.bin");
  if let Err(e) = std::fs::write(&cpu_path, &cpu) {
    warn!("数据转储：写 {} 失败 {e}", cpu_path.display());
    return;
  }
  if let Err(e) = std::fs::write(&gpu_path, &gpu_bytes) {
    warn!("数据转储：写 {} 失败 {e}", gpu_path.display());
    return;
  }
  info!(
    "数据转储 → {} / {}（各 {}KB = 头 {}B + struct {}KB + palette {}KB + leaves {}KB；volume={n}）",
    cpu_path.display(),
    gpu_path.display(),
    total / 1024,
    head_bytes,
    struct_bytes / 1024,
    palette_bytes / 1024,
    leaves_bytes / 1024,
  );
  match cpu.iter().zip(gpu_bytes.iter()).position(|(a, b)| a != b) {
    None if cpu.len() == gpu_bytes.len() => info!("数据转储 CPU/GPU 逐字节一致"),
    Some(i) => {
      warn!("数据转储 CPU/GPU 首异字节 @{i}：cpu={:#04x} gpu={:#04x}", cpu[i], gpu_bytes[i])
    }
    // 两侧长度由同一份布局决定，长度不等只可能是写入被截断
    None => warn!("数据转储 CPU/GPU 长度不等：{}B vs {}B", cpu.len(), gpu_bytes.len()),
  }
}

/// 统一体素渲染上传插件：主世界与物体同一路径。
/// 物体是 `Volumes.list[1..N]` 的普通 `VolumeGrid`，走相同的 dirty → VolumesBuilder → UploadSnapshot 路径。
pub struct VolumePlugin;
impl Plugin for VolumePlugin {
  fn build(&self, app: &mut App) {
    let ch = UploadCpuSampleChannel::default();
    app
      .insert_resource(ch.clone())
      .init_resource::<MainPending>()
      .init_resource::<VoxelDumpRequest>()
      .add_plugins(
        // 「数据转储」请求（菜单「游戏/世界」）：主 world 置位 → render world 读取
        bevy::render::extract_resource::ExtractResourcePlugin::<VoxelDumpRequest>::default(),
      )
      .add_systems(Last, (poll_pending, tick_voxel_dump_request));
    let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
      return;
    };
    render_app
      .insert_resource(ch)
      .init_resource::<BrickMapRevision>()
      .insert_resource(BuilderMirror { pending_full: true, ..Default::default() })
      .add_systems(RenderStartup, init_empty_gpu)
      .add_systems(ExtractSchedule, extract)
      .add_systems(Render, prepare.in_set(RenderSystems::PrepareResources))
      // 转储必须在 prepare 之后：先写完本帧上传，再 readback（同一帧的 GPU 状态）
      .add_systems(Render, dump_voxel_buffers.after(prepare));
  }
}
