//! 砖块图上传通道（Phase 1，Douglas 1:1）
//!
//! 裁决（遗留决策点一）：探测 `RenderDevice.limits().max_storage_buffer_binding_size`
//! 并在**首帧 info 日志打印路径选择**。默认使用**单 buffer 路径**（NVIDIA 1660/3070
//! 都 ≥2GB，最坏 b_struct < 1GB 阈值）。多 buffer 代码保留为 fallback
//! 结构（`BufferLayout::Multi` + 单测覆盖），运行期若 limit<1GB 则 warn 并退化为
//! 每帧全量上传，不 crash。
//!
//! 阶段拆分：
//! - `RenderStartup`：device/queue 就绪后创建 0 大小的 `GpuBrickMap` buffers（占位）
//! - `ExtractSchedule`（render sub-app 调度）：用 `Extract<ResMut<T>>` 访问主世界
//!   `VoxelScene(Volumes)`，按预算 drain 脏 chunk，做全量/增量 CPU 构建，将字节
//!   snapshot 写入 render world resource `UploadSnapshot`
//! - `RenderSystems::PrepareResources`：读 `UploadSnapshot` →
//!   * full：整块 `queue.write_buffer` 写 GPU buffer
//!   * incremental：仅写 Builder 记录的脏字节区间（struct/palette），
//!     典型单 chunk 编辑 → 窗口条目字 + 新树 append（KB 级），
//!     不再整块 DMA，帧率不掉
//! - info 打印 PROBE 结论 + UPLOAD[full|incremental]

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
  march_mask_lut_words, BrickMapGlobals, CHUNK_COMP_WORDS, GridDesc, MARCH_MASK_WORDS, TREE_BASE,
};

// ----------------------------------------------------------------------------
// Limits + BufferLayout（CPU 单测覆盖单/多两模式）
// ----------------------------------------------------------------------------

pub const SINGLE_THRESHOLD_BYTES: u64 = (1 << 30) - 1;

#[derive(Debug, Clone, Copy)]
pub struct BindingLimits {
  pub max_storage_buffer_binding_size: u64,
}
impl BindingLimits {
  pub fn probe(device: &RenderDevice) -> Self {
    Self {
      max_storage_buffer_binding_size: device.limits().max_storage_buffer_binding_size,
    }
  }
  pub fn force_multi(&self) -> bool {
    self.max_storage_buffer_binding_size < SINGLE_THRESHOLD_BYTES
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
      let per = 64u64 << 20;
      let left = limits.max_storage_buffer_binding_size.min(per).max(4 << 20);
      let node_slices = ((TREE_BASE as u64).saturating_add(per) / left) as usize;
      Self::Multi {
        node_slices: node_slices.max(1),
      }
    } else {
      Self::Single
    }
  }
}

// ----------------------------------------------------------------------------
// 主世界 Resources
// ----------------------------------------------------------------------------

#[derive(Resource)]
pub struct VoxelScene {
  pub volumes: gate_voxel::Volumes,
  pub demo_force_full_rebuild: bool,
}

#[derive(Resource, Clone, Debug)]
pub struct UploadBudget {
  pub max_bytes_per_frame: usize,
  pub incremental: bool,
}
impl Default for UploadBudget {
  fn default() -> Self {
    Self {
      max_bytes_per_frame: 4 * 1024 * 1024,
      incremental: true,
    }
  }
}

/// 主世界 Pending 资源：在主 world `Last` schedule 按预算 drain dirty，供只读提取
///
/// data_chunks / comp_chunks 元素 = `(volume_idx, coord)`：主世界 = 0，物体 = 1..N。
#[derive(Resource, Default)]
pub struct MainPending {
  pub force_full: bool,
  pub data_chunks: Vec<(usize, gate_voxel::ChunkCoord)>,
  pub comp_chunks: Vec<(usize, gate_voxel::ChunkCoord)>,
}

/// 单 chunk 增量重建的预算折中（典型树几十 KB~数 MB；256KB 经验值）
const PER_CHUNK_BYTES: usize = 256 * 1024;

/// 在主 world `Last` 阶段（晚于用户 Update 编辑）：按预算 drain dirty → MainPending
///
/// **关键**：进入时先清空 data_chunks / comp_chunks（上一帧的坐标已经在 ExtractSchedule
/// 被 mirror 消费）；否则坐标会逐帧累积并重复 update_chunk，带来巨量假增量上传。
/// force_full 同样在处理完一帧后复位。
///
/// Phase 3 统一：遍历 `scene.volumes.list` 的所有 volume（主世界 + 物体），
/// 每个 volume 独立 drain dirty，附带 volume_idx。
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
  pending.force_full = false;
  if scene.demo_force_full_rebuild {
    pending.force_full = true;
    scene.demo_force_full_rebuild = false;
  }
  let budget_n = (budget.max_bytes_per_frame / PER_CHUNK_BYTES).clamp(1, 64);
  // 聚合所有 volume 的 backlog（主世界 + 物体）
  let mut total_data_backlog = 0usize;
  let mut total_comp_backlog = 0usize;
  for grid in scene.volumes.list.iter() {
    total_data_backlog = total_data_backlog.saturating_add(grid.dirty.data_dirty_count());
    total_comp_backlog = total_comp_backlog.saturating_add(grid.dirty.comp_dirty_count());
  }
  // 反推 backlog：Startup 首次构建有大量 chunk dirty，
  // 若按 budget_n 逐帧 drain，每帧 builder.update_chunk 会 CPU 阻塞冻结 Prepare
  // 全局调度 → BG1 绑定 / DDA dispatch 推迟 → 画面"只有 UI 全黑"。
  // 当 backlog > 3× 预算（即明显处于 Startup 批量构建积压，而不是零星增量编辑），
  // 一次性把 dirty 队列清空。这个判定不依赖任何外部 flag 时序，鲁棒。
  let (data_n, comp_n) = if total_data_backlog > budget_n * 3 {
    (total_data_backlog, total_comp_backlog.max(budget_n))
  } else {
    (budget_n, budget_n.min(total_comp_backlog.max(1)))
  };
  // 每个 volume 独立 drain（drain 内部按队列容量自截，data_n 是上限不是强制）
  for (vol_idx, grid) in scene.volumes.list.iter_mut().enumerate() {
    for c in grid.dirty.drain_data_budget(data_n) {
      pending.data_chunks.push((vol_idx, c));
    }
    for c in grid.dirty.drain_comp_budget(comp_n) {
      pending.comp_chunks.push((vol_idx, c));
    }
  }
}

// ----------------------------------------------------------------------------
// render world 资源
// ----------------------------------------------------------------------------

/// ExtractSchedule 用的 CPU builder / pending 状态（render world resource）
///
/// Phase 3 统一：`builder: Option<VolumesBuilder>` 持有 `Vec<BrickMapBuilder>`，
/// pending chunks 带 volume_idx。
#[derive(Resource, Default)]
pub struct BuilderMirror {
  pub builder: Option<VolumesBuilder>,
  pub pending_full: bool,
  pub pending_data_chunks: Vec<(usize, gate_voxel::ChunkCoord)>,
  pub pending_comp_chunks: Vec<(usize, gate_voxel::ChunkCoord)>,
}

/// ExtractSchedule 产出 → PrepareResources 消费（render world resource）
///
/// Phase 3 统一：`volumes: VolumesSnapshot`。full 模式带完整 `b_struct`/`b_palette`
/// 整块 DMA；incremental 模式不带整量字节（避免 100MB+ 级 memcpy 卡顿），
/// 改为 `struct_blobs`/`palette_blobs` 脏块（偏移+内容）逐块 write_buffer。
/// state/comp 仍是主世界 only（Phase 2 shader 重写后再扩展）。
#[derive(Resource, Clone)]
pub struct UploadSnapshot {
  pub volumes: VolumesSnapshot,
  pub state_bytes: Vec<u8>,
  pub comp_chunks: usize,
}

/// P2.7 上传 CPU 耗时样本（render world 资源，由 prepare 每帧 insert_resource 覆盖。
/// render→main 同步由 P2.7 Task 2 gate-app sync_gpu_timings 内**通过 Arc<Mutex> 共享**，
/// 详见下方 [`UploadCpuSampleChannel`]。上传段的 GPU 拷贝在 submit 时发生，测不到——OQ-2 选 A。
#[derive(Resource, Clone, Copy, Debug, Default)]
pub struct UploadCpuSample {
  pub cpu_ms: f32,
  pub generation: u64,
}

/// render↔main 共享通道（Arc<Mutex>），与 bevy RenderDiagnosticsMutex 同款模式。
/// prepare（render world, PrepareResources）写入；sync_gpu_timings（main world, Update）读出。
#[derive(Resource, Clone, Debug, Default)]
pub struct UploadCpuSampleChannel(pub std::sync::Arc<std::sync::Mutex<Option<UploadCpuSample>>>);

/// GPU 资源（render world）：统一 struct/leaves/palette/comp/state + grid_descs + globals。
///
/// `grid_descs_buf`：Phase 3 新增，GridDesc 数组（144B/entry）——主世界 + 物体统一描述符，
/// shader `trace_scene` 遍历无 kind 分支。`grid_descs_count` 跟踪有效条目数。
/// `globals`：保留旧 BrickMapGlobals uniform（Phase 1 shader 字节兼容，重写后移除）。
/// `leaves`：恒空占位（Douglas 格式 palette 直存节点；BG1 binding(1) 布局占位必需）。
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
}

// ----------------------------------------------------------------------------
// RenderStartup system：创建占位 buffers（render world）
// ----------------------------------------------------------------------------

fn init_empty_gpu(device: Res<RenderDevice>, mut commands: Commands) {
  // COPY_SRC：扩容前缀拷贝（copy_buffer_to_buffer 旧→新）必需
  let make = |label: &str| -> Buffer {
    device.create_buffer(&BufferDescriptor {
      label: Some(label),
      size: 4,
      usage: BufferUsages::COPY_DST | BufferUsages::COPY_SRC | BufferUsages::STORAGE,
      mapped_at_creation: false,
    })
  };
  let globals = UniformBuffer::<BrickMapGlobals>::default();
  commands.insert_resource(GpuBrickMap {
    struct_buf: make("gate_struct"),
    leaves: make("gate_leaves"),
    palette: make("gate_palette"),
    comp: make("gate_comp"),
    state: make("gate_state"),
    grid_descs_buf: make("gate_grid_descs"),
    grid_descs_count: 0,
    globals,
  });
}

// ----------------------------------------------------------------------------
// ExtractSchedule（render sub-app）：只读访问主 world 资源，CPU 构建 snapshot
//   Extract<T> 的 T 必须 ReadOnlySystemParam，所以全 Res<T>。
//   &VoxelScene.volumes 读视图 + MainPending.data_chunks(已 drained) → 就地 VolumesBuilder。
// ----------------------------------------------------------------------------

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
  mirror
    .pending_data_chunks
    .extend(main_pending.data_chunks.iter().copied());
  mirror
    .pending_comp_chunks
    .extend(main_pending.comp_chunks.iter().copied());

  let first = mirror.builder.is_none();
  let pending_full = std::mem::take(&mut mirror.pending_full);
  let mut pending_data: Vec<(usize, gate_voxel::ChunkCoord)> =
    std::mem::take(&mut mirror.pending_data_chunks);
  let _pending_comp: Vec<(usize, gate_voxel::ChunkCoord)> =
    std::mem::take(&mut mirror.pending_comp_chunks);
  let need_full = first || pending_full || !budget.incremental;
  let dirty_any = need_full || !pending_data.is_empty() || !_pending_comp.is_empty();
  // 如果非首帧 + 非强制全量 + 无脏 chunk → 跳过构建/上传（省 CPU 构建 + PCIe）
  if !dirty_any {
    return;
  }
  let volumes_ref = &scene.volumes;
  let builder = mirror
    .builder
    .get_or_insert_with(|| VolumesBuilder::new_unbuilt(volumes_ref));
  if need_full {
    *builder = VolumesBuilder::build_full(volumes_ref);
    pending_data.clear();
  } else {
    // 同步 volume 数量（新增物体追加 builder）+ 更新变换
    builder.sync(volumes_ref);
    for (vol_idx, c) in pending_data.drain(..) {
      builder.update_chunk(volumes_ref, vol_idx, c);
    }
  }
  let snapshot = builder.snapshot();
  // state/comp 暂仍主世界 only（Phase 2 shader 重写后再扩展到物体）
  let state_bytes = volumes_ref.main().state_table_bytes().to_vec();
  let comp_chunks = volumes_ref.main().comp_layer().len();
  commands.insert_resource(UploadSnapshot {
    volumes: snapshot,
    state_bytes,
    comp_chunks,
  });
}

// ----------------------------------------------------------------------------
// 辅助（u32 → u8 字节视图 + ensure/write buffer）
// ----------------------------------------------------------------------------

fn u8_of_u32(w: &[u32]) -> &[u8] {
  unsafe { std::slice::from_raw_parts(w.as_ptr() as *const u8, w.len() * 4) }
}

/// GridDesc 数组 → u8 字节视图（`#[repr(C)]` + 144B/entry，可直接 cast）
fn u8_of_grid_descs(descs: &[GridDesc]) -> &[u8] {
  unsafe {
    std::slice::from_raw_parts(
      descs.as_ptr() as *const u8,
      descs.len() * std::mem::size_of::<GridDesc>(),
    )
  }
}

/// GPU buffer 扩容尺寸策略（纯函数，单测覆盖）。
///
/// 旧策略 2× 翻倍的问题：b_struct 首次增长 ~10KB 就触发 163MB→327MB 翻倍 +
/// 整份 163MB PCIe 重写（首编辑 51ms 卡顿的 DMA 大头），且 VRAM 空耗近一倍。
/// 新策略：大 buffer（≥8MB，当前即 b_struct/b_leaves 大场景形态）按 32MB 水位
/// 对齐——扩容 DMA 量小、重建间隔 ≥32MB 增长；小 buffer 维持 2×（palette 等翻倍成本可忽略）。
fn grow_size(cap: u64, need: u64) -> u64 {
  const BIG: u64 = 8 << 20;
  const RESERVE: u64 = 32 << 20;
  if need >= BIG {
    need.div_ceil(RESERVE) * RESERVE
  } else {
    need.max(cap * 2).max(65536) // ≥64KB，2× amortize（原策略）
  }
}

/// 保证 buffer 能容纳 `bytes`，扩容时保留/写入新的整份内容（不丢旧字节）。
///
/// 之前的 bug（极限场景 GPU 端"全画面无体素"的根因）：
///   ensure() 只 `create_buffer(size = bytes)` 然后返回，新 buffer 全是 0，
///   之后 incremental 上传只 write_partial(dirty range)，164MB 里除了 dirty 的 6MB
///   其它区域（tile index 表 / leaves 指针 / l1/l2 表）都保持 0，tile entry = 0 被
///   WGSL sample_brickmap 当作空砖 → 全黑。
///
/// 修复：每次扩容（bytes > cur.size()）时创建更大 buffer（[`grow_size`] 水位策略）。
///
/// `prefix_valid`（增量镜像路径传 true）：bytes[0..cap) 与 GPU 现有内容一致
/// （CPU 镜像是权威拷贝、每帧脏区间全覆盖上传，故成立）→ 旧 buffer 前缀用
/// **GPU-GPU copy**（不占 PCIe，163MB <2ms）搬到新 buffer，只 write_buffer 新增
/// 尾部 [cap..need)——单次扩容 PCIe 从整份 163MB 降到增长量（10KB 级）。
/// full 重建等 GPU 旧内容不可信的场景传 false：整份 bytes 一次 DMA（原行为）。
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
    // COPY_SRC：本 buffer 下次扩容时要作为前缀拷贝的源（缺它第二次 grow 必炸）
    usage: BufferUsages::COPY_DST | BufferUsages::COPY_SRC | BufferUsages::STORAGE,
    mapped_at_creation: false,
  });
  if prefix_valid && cap > 0 {
    // 前缀经 GPU-GPU 拷贝（独立 encoder submit，不占 PCIe）；随后 write_buffer 的
    // 内部 submit 与之保持提交顺序，先拷贝后写尾部。
    // COPY_BUFFER_ALIGNMENT=4；cap 恒为 words×4 或初始 4B，天然对齐。
    let mut enc = device.create_command_encoder(&CommandEncoderDescriptor {
      label: Some("gate_grow_prefix_copy"),
    });
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

/// 增量路径专用：保证 buffer 容量 ≥ `need_bytes`；扩容时创建新 buffer 并把旧内容
/// **GPU-GPU 拷贝**为前缀（不占 PCIe）。新增尾部 [cap..need) 不在这里写——
/// 调用方随后逐块 `write_buffer` 脏块，追加的树尾部恰好被脏块全覆盖。
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
    // COPY_SRC：本 buffer 下次扩容时要作为前缀拷贝的源（缺它第二次 grow 必炸）
    usage: BufferUsages::COPY_DST | BufferUsages::COPY_SRC | BufferUsages::STORAGE,
    mapped_at_creation: false,
  });
  if cap > 0 {
    // COPY_BUFFER_ALIGNMENT=4；cap 恒为 words×4 或初始 4B，天然对齐。
    let mut enc = device.create_command_encoder(&CommandEncoderDescriptor {
      label: Some("gate_grow_prefix_copy"),
    });
    enc.copy_buffer_to_buffer(cur, 0, &new_buf, 0, cap);
    queue.submit([enc.finish()]);
  }
  *cur = new_buf;
}

// ----------------------------------------------------------------------------
// PrepareResources：snapshot → GPU 写；首帧探测 limits
// ----------------------------------------------------------------------------

pub(crate) fn prepare(
  mut commands: Commands,
  snapshot: Option<Res<UploadSnapshot>>,
  mut gpu: ResMut<GpuBrickMap>,
  device: Res<RenderDevice>,
  queue: Res<RenderQueue>,
  sample_channel: Option<Res<UploadCpuSampleChannel>>,
) {
  let Some(snap) = snapshot else { return };
  let t0 = std::time::Instant::now();

  // ---- 首帧探测（只打一次日志）----
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
        "GPU storage binding < 1GB，退化到全量上传（单 tile 170MB 会跨帧）——P2.3 多 buffer 切片代码保留，需要时接入"
      );
    }
  }

  let is_full = matches!(snap.volumes.mode_tag, "full" | "fallback_full");
  let comp_bytes = snap.comp_chunks * CHUNK_COMP_WORDS * 4;

  // P4：方向可达掩码 LUT（Douglas #18 Bitwise Masking）→ b_leaves。
  // 全局常量（8 octant × 64 入口格 × 2 u32 = 4KB），与 volume 无关；
  // 只在 buffer 尚未容纳时写一次，之后零 PCIe。
  {
    let lut_need = (MARCH_MASK_WORDS * 4) as u64;
    if gpu.leaves.size() < lut_need {
      let lut = march_mask_lut_words();
      write(&device, &queue, &mut gpu.leaves, "gate_leaves", u8_of_u32(&lut));
    }
  }

  // 传输字节统计（日志用）
  let (struct_tx_bytes, palette_tx_bytes, grid_descs_tx_bytes);

  if is_full {
    // 全量：整块 DMA + ensure 保证 GPU 容量够
    let struct_bytes = u8_of_u32(&snap.volumes.b_struct);
    let palette_bytes = u8_of_u32(&snap.volumes.b_palette);
    let grid_descs_bytes = u8_of_grid_descs(&snap.volumes.grid_descs);
    struct_tx_bytes = struct_bytes.len();
    palette_tx_bytes = palette_bytes.len();
    grid_descs_tx_bytes = grid_descs_bytes.len();
    write(
      &device,
      &queue,
      &mut gpu.struct_buf,
      "gate_struct",
      struct_bytes,
    );
    write(
      &device,
      &queue,
      &mut gpu.palette,
      "gate_palette",
      palette_bytes,
    );
    write(
      &device,
      &queue,
      &mut gpu.grid_descs_buf,
      "gate_grid_descs",
      grid_descs_bytes,
    );
  } else {
    // 增量路径：snapshot 不再全量拼接 b_struct（旧实现每个脏帧 memcpy 100MB+，
    // 是编辑 spike 的根因）。这里只按总字节 ensure 容量（扩容走 GPU-GPU 前缀
    // 拷贝，不占 PCIe），随后逐脏块 write_buffer——每块 = chunk 窗口条目字 +
    // 新 append 的树尾部（KB~MB 级）；追加尾部恰好覆盖扩容后的新区域。
    // GridDesc：volume 数变化已由 snapshot 判 need_full；增量路径内容不变，
    // 仅 ensure 维持容量。leaves 恒空，跳过。
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
  // state / comp 每次都整块写（state 4KB、comp 每 chunk 8KB，都很小）
  write(
    &device,
    &queue,
    &mut gpu.state,
    "gate_state",
    &snap.state_bytes,
  );
  // comp: 每 chunk 8KB 占位；build 后可能为 0 字节，ensure 至少 4B。
  // comp_bytes 只是预估上限；实际内容读 grid 时已经按真实 size 存。
  // comp 数据直接从 CPU 侧构建：UploadSnapshot 当前没带 comp 字节，
  // 这里用 gpu.comp size ≥ 预估的占位（历史行为：仅 buffer 大小对齐）。
  {
    let placeholder = vec![0u8; comp_bytes.max(4)];
    // comp 为占位通道（内容无意义，历史行为仅对齐 buffer 大小）；保留旧前缀即可
    ensure_with_copy(
      &device,
      &queue,
      &mut gpu.comp,
      "gate_comp",
      &placeholder,
      true,
    );
  }

  // globals：从主世界 GridDesc[0] 构造向后兼容 BrickMapGlobals（Phase 1 shader 字节兼容，
  // Phase 2 重写 dda.wgsl 后移除——届时 shader 走 grid_descs_buf，不再读 globals）。
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
    // node_words / node_free_words 等字段在 VolumesBuilder 内部，未暴露；
    // shader 重写后这些字段不再使用，此处置 0 不影响 Phase 2 之后的路径。
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

  let elapsed = t0.elapsed();
  let cpu_ms = elapsed.as_secs_f32() * 1000.0;
  let tx_bytes_total = (struct_tx_bytes
    + palette_tx_bytes
    + grid_descs_tx_bytes
    + snap.state_bytes.len()) as f64;
  let mb = tx_bytes_total / (1 << 20) as f64;
  // full: chunks = 所有 volume 的 chunk_count 之和；incremental: chunks = 本轮 dirty 数
  let chunks_show = if is_full {
    snap
      .volumes
      .grid_descs
      .iter()
      .map(|g| g.chunk_count as usize)
      .sum::<usize>()
  } else {
    snap.volumes.dirty_chunks
  };
  debug!(target: "gate",
    "UPLOAD[{}]: bytes={:.2}MB (s {}KB,pal {}KB,gd {}KB,state 4KB), chunks={}, comp={}KB, elapsed={:?}",
    snap.volumes.mode_tag, mb,
    struct_tx_bytes / 1024, palette_tx_bytes / 1024,
    grid_descs_tx_bytes / 1024, chunks_show, comp_bytes / 1024, elapsed,
  );
  // P2.7：写入共享通道（render↔main Arc<Mutex>，OQ-2 选 A 不提供 GPU 值）
  static SAMPLE_GEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
  let generation = SAMPLE_GEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
  let sample = UploadCpuSample { cpu_ms, generation };
  commands.insert_resource(sample);
  if let Some(ch) = sample_channel
    && let Ok(mut g) = ch.0.lock()
  {
    *g = Some(sample);
  }
  // VRAM 规模留档（v3.9.1 用户指令：2GB 内存预算断言取消，仅打印不拦截）
  let vram = gpu.struct_buf.size()
    + gpu.leaves.size()
    + gpu.palette.size()
    + gpu.comp.size()
    + gpu.state.size()
    + gpu.grid_descs_buf.size();
  if vram > 2u64 << 30 {
    bevy::log::warn!("GPU VRAM 超过旧预算线（仅提示）: {vram} bytes");
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
  // 消费完本帧 snapshot 必须移除；否则 prepare 每帧都读旧 snapshot → 140MB/帧假上传
  commands.remove_resource::<UploadSnapshot>();
}

// ----------------------------------------------------------------------------
// Plugin
// ----------------------------------------------------------------------------

/// 统一体素渲染上传插件（主世界 + 物体同一路径）
///
/// Phase 3 OBJ→Volume 统一后，OBJ 不再有独立 ObjScene/RenderObj/GpuObjPool 三段
/// 管道，而是作为 `Volumes.list[1..N]` 中的普通 `VolumeGrid`，走与主世界完全相同的
/// dirty → VolumesBuilder → UploadSnapshot 增量上传路径。
pub struct VolumePlugin;
impl Plugin for VolumePlugin {
  fn build(&self, app: &mut App) {
    // Render↔main 共享通道（UploadCpuSample）：插入同一个 Arc<Mutex> Resource 到两个世界
    let ch = UploadCpuSampleChannel::default();
    app
      .insert_resource(ch.clone())
      .init_resource::<MainPending>()
      .add_systems(Last, poll_pending);
    let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
      return;
    };
    render_app
      .insert_resource(ch)
      .insert_resource(BuilderMirror {
        pending_full: true,
        ..Default::default()
      })
      .add_systems(RenderStartup, init_empty_gpu)
      .add_systems(ExtractSchedule, extract)
      .add_systems(Render, prepare.in_set(RenderSystems::PrepareResources));
  }
}

// ----------------------------------------------------------------------------
// CPU 单测（不依赖 GPU）
// ----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
  use super::super::builder::BrickMapBuilder;
  use super::super::wire::{CHUNK_INDEX_WORDS, STATE_TOTAL_WORDS};
  use super::*;
  use gate_voxel::fill_box;

  #[test]
  fn limits_select_layout() {
    let multi = BindingLimits {
      max_storage_buffer_binding_size: 128 * (1 << 20),
    };
    assert!(multi.force_multi());
    let single = BindingLimits {
      max_storage_buffer_binding_size: 2 * (1 << 30),
    };
    assert!(!single.force_multi());
    let _ml = BufferLayout::from_limits(&multi);
    let _sl = BufferLayout::from_limits(&single);
  }

  #[test]
  fn comp_state_api_on_grid() {
    let mut g = gate_voxel::VolumeGrid::new();
    let chunk = gate_voxel::ChunkCoord::new(0, 0, 0);
    // set_comp 挂 level 2 brick (bx,by,bz)；get_comp 查 voxel 所在 brick
    g.set_comp(chunk, 1, 1, 1, 0xABCD);
    let voxel = gate_voxel::VoxelCoord::new(16 + 5, 16 + 3, 16 + 2);
    assert_eq!(g.get_comp(voxel), 0xABCD);
    assert_eq!(g.get_comp(gate_voxel::VoxelCoord::new(1, 1, 1)), 0);
    g.set_state(7, 2, 0x42);
    assert_eq!(g.get_state(7, 2), 0x42);
    assert_eq!(g.get_state(99, 0), 0);
    assert_eq!(g.state_table_bytes().len(), 256 * 16);
  }

  #[test]
  fn wire_constants() {
    assert_eq!(CHUNK_COMP_WORDS * 4, 4096 * 2); // u16[4096] → 8KB/chunk
    assert_eq!(STATE_TOTAL_WORDS * 4, 4096); // 256×4×4B
  }

  #[test]
  fn grow_size_watermark_policy() {
    // 小 buffer：2×（原策略），下限 64KB
    assert_eq!(grow_size(4, 2048), 65536); // palette 首扩
    assert_eq!(grow_size(32768, 40000), 65536);
    assert_eq!(grow_size(40000, 50000), 80000); // 恰好 2×
    // 大 buffer：32MiB 水位对齐——首增 ~10KB 不再翻倍到 2×cap
    let cap160m: u64 = 163_798_052; // 156.2MiB（实机 full 后 b_struct）
    let need = cap160m + 10 * 1024; // 首次编辑真实增长 ~10KB
    let grown = grow_size(cap160m, need);
    assert_eq!(grown, 160 * 1024 * 1024); // → 160MiB（下一 32MiB 边界），非 2×=312MiB
    assert!(grown >= need);
    // 水位内的增长由 ensure_with_copy 的 cap>=need 早退拦截，不会进 grow_size；
    // 刚跨过边界 → 立即扩到下一档（amortized）
    assert_eq!(grow_size(grown, grown + 1024), 192 * 1024 * 1024);
    // 连续跨档（160MiB+33MiB=193MiB → 224MiB）
    assert_eq!(
      grow_size(grown, grown + 33 * 1024 * 1024),
      224 * 1024 * 1024
    );
  }

  #[test]
  fn buffer_data_roundtrip() {
    let mut g = gate_voxel::VolumeGrid::new();
    fill_box(&mut g, glam::IVec3::ZERO, glam::IVec3::splat(8), 1);
    g.set_state(5, 3, 0xCAFEBABE);
    let state = g.state_table_bytes();
    let off = 5 * 16 + 3 * 4; // entry 5 + field 3
    assert_eq!(state[off..off + 4], 0xCAFEBABEu32.to_le_bytes());
    let buffers = BrickMapBuilder::build_full(&g).buffers().clone();
    let bytes = u8_of_u32(&buffers.b_struct);
    assert_eq!(bytes.len(), buffers.b_struct.len() * 4);
    assert!(buffers.globals.tile_count >= 1);
    // chunk 窗口条目非零（Region ① 至少 1 个 chunk）
    assert!(
      !buffers.b_struct[..CHUNK_INDEX_WORDS]
        .iter()
        .all(|&w| w == 0)
    );
  }
}
