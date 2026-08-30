//! 砖块图上传通道（P2.3）
//!
//! 裁决（遗留决策点一）：探测 `RenderDevice.limits().max_storage_buffer_binding_size`
//! 并在**首帧 info 日志打印路径选择**。默认使用**单 buffer 路径**（NVIDIA 1660/3070
//! 都 ≥2GB，最坏 750MB b_struct < 1GB 阈值）。多 buffer 代码保留为 fallback
//! 结构（`BufferLayout::Multi` + 单测覆盖），运行期若 limit<1GB 则 warn 并退化为
//! 每帧全量 8MB index + 4MB bitmap + 128MB dirs + N×64MB nodes 单独写，不 crash。
//!
//! 阶段拆分：
//! - `RenderStartup`：device/queue 就绪后创建 0 大小的 `GpuBrickMap` buffers（占位）
//! - `ExtractSchedule`（render sub-app 调度）：用 `Extract<ResMut<T>>` 访问主世界
//!   `VoxelScene(TileGrid)`，按预算 drain 脏 tile，做全量/增量 CPU 构建，将字节
//!   snapshot 写入 render world resource `UploadSnapshot`
//! - `RenderSystems::PrepareResources`：读 `UploadSnapshot` →
//!   * full：整块 `queue.write_buffer` 写 5 大类 GPU buffer
//!   * incremental：仅写 Builder 记录的脏字节区间（struct/leaves/palette），
//!     典型单 tile palette swap → ~132KB struct + 0 leaves + 2KB palette，
//!     不再整块 147MB DMA，帧率不掉
//! - info 打印 PROBE 结论 + UPLOAD[full|incremental]

use bevy::{
  log::{info, warn},
  prelude::*,
  render::{
    Extract, ExtractSchedule, Render, RenderApp, RenderStartup, RenderSystems,
    render_resource::*,
    renderer::{RenderDevice, RenderQueue},
  },
};

use super::builder::{BrickMapBuilder, DirtyRanges, TileUpdate};
use super::wire::{
  BrickMapBuffers, BrickMapGlobals, CELL_DIR_WORDS, NODE_STREAM_BASE, TILE_BITMAP_WORDS,
  TILE_COMP_WORDS,
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
      let node_slices = ((NODE_STREAM_BASE as u64).saturating_add(per) / left) as usize;
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
  pub grid: gate_voxel::TileGrid,
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
#[derive(Resource, Default)]
pub struct MainPending {
  pub force_full: bool,
  pub data_tiles: Vec<gate_voxel::TileCoord>,
  pub comp_tiles: Vec<gate_voxel::TileCoord>,
}

/// 在主 world `Last` 阶段（晚于用户 Update 编辑）：按预算 drain dirty → MainPending
///
/// **关键**：进入时先清空 data_tiles / comp_tiles（上一帧的坐标已经在 ExtractSchedule
/// 被 mirror 消费）；否则坐标会逐帧累积并重复 update_tile，带来 O(140MB/帧) 的假
/// 增量上传。force_full 同样在处理完一帧后复位。
pub fn poll_pending(
  scene: Option<ResMut<VoxelScene>>,
  budget: Option<Res<UploadBudget>>,
  mut pending: ResMut<MainPending>,
) {
  let (Some(mut scene), Some(budget)) = (scene, budget) else {
    return;
  };
  pending.data_tiles.clear();
  pending.comp_tiles.clear();
  pending.force_full = false;
  if scene.demo_force_full_rebuild {
    pending.force_full = true;
    scene.demo_force_full_rebuild = false;
  }
  let per_tile_floor = (TILE_BITMAP_WORDS + CELL_DIR_WORDS) * 4;
  let budget_n = (budget.max_bytes_per_frame / per_tile_floor.max(1)).clamp(1, 64);
  let data_backlog = scene.grid.dirty.data_dirty_count();
  let comp_backlog = scene.grid.dirty.comp_dirty_count();
  // 反推 backlog：Startup 首次构建有上百 tile dirty（极限场景 ~211），
  // 若按 budget_n (≈31) 逐帧 drain，每帧 builder.update_tile(31 tiles) 会 CPU 阻塞 2~3s
  // 冻结 Prepare 全局调度 → BG1 绑定 / DDA dispatch 推迟十几秒 → 画面"只有 UI+渐变全黑"。
  // 当 backlog > 3× 预算（即明显处于 Startup 批量构建积压，而不是 120 帧 1 tile 增量编辑），
  // 一次性把 dirty 队列清空。这个判定不依赖任何外部 flag 时序，鲁棒。
  let (data_n, comp_n) = if data_backlog > budget_n * 3 {
    (data_backlog, comp_backlog.max(budget_n))
  } else {
    (budget_n, budget_n.min(comp_backlog.max(1)))
  };
  pending
    .data_tiles
    .extend(scene.grid.dirty.drain_data_budget(data_n));
  pending
    .comp_tiles
    .extend(scene.grid.dirty.drain_comp_budget(comp_n));
}

// ----------------------------------------------------------------------------
// render world 资源
// ----------------------------------------------------------------------------

/// ExtractSchedule 用的 CPU builder / pending 状态（render world resource）
#[derive(Resource, Default)]
pub struct BuilderMirror {
  pub builder: Option<BrickMapBuilder>,
  pub pending_full: bool,
  pub pending_data_tiles: Vec<gate_voxel::TileCoord>,
  pub pending_comp_tiles: Vec<gate_voxel::TileCoord>,
}

/// ExtractSchedule 产出 → PrepareResources 消费（render world resource）
#[derive(Resource, Clone)]
pub struct UploadSnapshot {
  pub buffers: BrickMapBuffers,
  pub mode_tag: &'static str,
  pub state_bytes: Vec<u8>,
  pub comp_tiles: usize,
  /// 更新的 dirty tile 数（用于日志：不再误导写"总 tile_count=3"）
  pub dirty_tiles: usize,
  /// 增量脏字节区间；`mode_tag="full"` 时会被忽略（整块写）。
  pub dirty: DirtyRanges,
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

#[derive(Resource)]
pub struct GpuBrickMap {
  pub struct_buf: Buffer,
  pub leaves: Buffer,
  pub palette: Buffer,
  pub comp: Buffer,
  pub state: Buffer,
  pub globals: UniformBuffer<BrickMapGlobals>,
}

// ----------------------------------------------------------------------------
// RenderStartup system：创建占位 buffers（render world）
// ----------------------------------------------------------------------------

fn init_empty_gpu(device: Res<RenderDevice>, mut commands: Commands) {
  let make = |label: &str| -> Buffer {
    device.create_buffer(&BufferDescriptor {
      label: Some(label),
      size: 4,
      usage: BufferUsages::COPY_DST | BufferUsages::STORAGE,
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
    globals,
  });
}

// ----------------------------------------------------------------------------
// ExtractSchedule（render sub-app）：只读访问主 world 资源，CPU 构建 snapshot
//   Extract<T> 的 T 必须 ReadOnlySystemParam，所以全 Res<T>。
//   &VoxelScene.grid 读视图 + MainPending.data_tiles(已 drained) → 就地 BrickMapBuilder。
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
    .pending_data_tiles
    .extend(main_pending.data_tiles.clone());
  mirror
    .pending_comp_tiles
    .extend(main_pending.comp_tiles.clone());

  let first = mirror.builder.is_none();
  let pending_full = std::mem::take(&mut mirror.pending_full);
  let mut pending_data: Vec<gate_voxel::TileCoord> = std::mem::take(&mut mirror.pending_data_tiles);
  let pending_comp: Vec<_> = std::mem::take(&mut mirror.pending_comp_tiles);
  let need_full = first || pending_full || !budget.incremental;
  let dirty_any = need_full || !pending_data.is_empty() || !pending_comp.is_empty();
  // 如果非首帧 + 非强制全量 + 无脏 tile → 跳过构建/上传（节省 140MB CPU 构建 + PCIe）
  if !dirty_any {
    return;
  }
  let dirty_tiles = pending_data.len();
  let grid_ref = &scene.grid;
  let builder = mirror
    .builder
    .get_or_insert_with(|| BrickMapBuilder::new_unbuilt(grid_ref));
  if need_full {
    *builder = BrickMapBuilder::build_full(grid_ref);
    pending_data.clear();
  } else {
    for c in pending_data.drain(..) {
      builder.update_tile(grid_ref, c);
    }
  }
  let mode_tag = if first || need_full {
    "full"
  } else if budget.incremental {
    "incremental"
  } else {
    "fallback_full"
  };
  let buffers = builder.buffers().clone();
  // full 时 dirty ranges 无意义（prepare 走整块写）；incremental 取出累积的脏区间。
  let dirty = if need_full {
    DirtyRanges {
      struct_ranges: Vec::new(),
      leaves_ranges: Vec::new(),
      palette_changed: false,
    }
  } else {
    builder.take_dirty_ranges()
  };
  let state_bytes = grid_ref.state_table_bytes().to_vec();
  let comp_tiles = grid_ref.comp_layer().len();
  drop(pending_comp);
  let _ = TileUpdate::Rebuilt;

  commands.insert_resource(UploadSnapshot {
    buffers,
    mode_tag,
    state_bytes,
    comp_tiles,
    dirty_tiles,
    dirty,
  });
}

// ----------------------------------------------------------------------------
// 辅助（u32 → u8 字节视图 + ensure/write buffer）
// ----------------------------------------------------------------------------

fn u8_of_u32(w: &[u32]) -> &[u8] {
  unsafe { std::slice::from_raw_parts(w.as_ptr() as *const u8, w.len() * 4) }
}

/// 保证 buffer 能容纳 `bytes`，扩容时保留/写入新的整份内容（不丢旧字节）。
///
/// 之前的 bug（极限场景 GPU 端"全画面无体素"的根因）：
///   ensure() 只 `create_buffer(size = bytes)` 然后返回，新 buffer 全是 0，
///   之后 incremental 上传只 write_partial(dirty range)，164MB 里除了 dirty 的 6MB
///   其它区域（tile index 表 / leaves 指针 / l1/l2 表）都保持 0，tile entry = 0 被
///   WGSL sample_brickmap 当作空砖 → 全黑。
///
/// 修复：每次扩容（bytes > cur.size()）时，创建更大 buffer（×2 预留，避免逐帧重建），
///       然后把整份 `bytes` 一次性写进（164MB 扩容几帧一次）；不扩容时原样返回。
///       后续 write_partial 可以继续调用（冗余 dirty 区覆写但无害）。
fn ensure_with_copy(
  device: &RenderDevice,
  queue: &RenderQueue,
  cur: &mut Buffer,
  label: &str,
  bytes: &[u8],
) {
  let cap = cur.size();
  let need = bytes.len() as u64;
  if cap >= need {
    return;
  }
  let reserve = cap.saturating_mul(2).max(65536); // ≥64KB，2× amortize
  let new_size = need.max(reserve);
  *cur = device.create_buffer(&BufferDescriptor {
    label: Some(label),
    size: new_size,
    usage: BufferUsages::COPY_DST | BufferUsages::STORAGE,
    mapped_at_creation: false,
  });
  if !bytes.is_empty() {
    queue.write_buffer(cur, 0, bytes);
  }
}

/// 整块写（full 模式 / state 等小 buffer）
fn write(
  device: &RenderDevice,
  queue: &RenderQueue,
  cur: &mut Buffer,
  label: &str,
  bytes: &[u8],
) {
  ensure_with_copy(device, queue, cur, label, bytes);
  if !bytes.is_empty() {
    queue.write_buffer(cur, 0, bytes);
  }
}

/// 部分写：只写 [lo, hi)。GPU buffer 必须已经 ≥ hi（full 模式已 ensure 过一次）。
fn write_partial(queue: &RenderQueue, cur: &Buffer, bytes: &[u8], lo: usize, hi: usize) {
  let hi = hi.min(bytes.len());
  if lo >= hi {
    return;
  }
  debug_assert!(
    cur.size() >= hi as u64,
    "write_partial: buffer size {}B < hi {}B",
    cur.size(),
    hi
  );
  queue.write_buffer(cur, lo as u64, &bytes[lo..hi]);
}

// ----------------------------------------------------------------------------
// PrepareResources：snapshot → GPU 写；首帧探测 limits
// ----------------------------------------------------------------------------

fn prepare(
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

  let struct_bytes = u8_of_u32(&snap.buffers.b_struct);
  let leaves_bytes = u8_of_u32(&snap.buffers.b_leaves);
  let palette_bytes = u8_of_u32(&snap.buffers.b_palette);
  let comp_bytes = snap.comp_tiles * TILE_COMP_WORDS * 4;

  // bytes 统计（日志用）：full = 整块；incremental = dirty ranges 求和
  let is_full = matches!(snap.mode_tag, "full" | "fallback_full");
  let (struct_tx_bytes, leaves_tx_bytes, palette_tx_bytes) = if is_full {
    (struct_bytes.len(), leaves_bytes.len(), palette_bytes.len())
  } else {
    let s = snap
      .dirty
      .struct_ranges
      .iter()
      .map(|&(a, b)| (b.min(struct_bytes.len())).saturating_sub(a))
      .sum::<usize>();
    let l = snap
      .dirty
      .leaves_ranges
      .iter()
      .map(|&(a, b)| (b.min(leaves_bytes.len())).saturating_sub(a))
      .sum::<usize>();
    let p = if snap.dirty.palette_changed {
      palette_bytes.len()
    } else {
      0
    };
    (s, l, p)
  };

  if is_full {
    // 全量：整块 + ensure 保证 GPU 容量够
    write(&device, &queue, &mut gpu.struct_buf, "gate_struct", struct_bytes);
    write(&device, &queue, &mut gpu.leaves, "gate_leaves", leaves_bytes);
    write(&device, &queue, &mut gpu.palette, "gate_palette", palette_bytes);
  } else {
    // ensure_with_copy：如果 CPU 镜像扩容超过 GPU buffer，新 buffer 会立刻被整份写入，
    // 之后的 write_partial 只覆写 dirty 区（冗余但正确），不会出现"扩容后 buffer 归零只写 6MB"
    // 导致 tile index 表丢失全 0（全黑）。
    ensure_with_copy(&device, &queue, &mut gpu.struct_buf, "gate_struct", struct_bytes);
    ensure_with_copy(&device, &queue, &mut gpu.leaves, "gate_leaves", leaves_bytes);
    ensure_with_copy(&device, &queue, &mut gpu.palette, "gate_palette", palette_bytes);

    for (lo, hi) in snap.dirty.struct_ranges.iter().copied() {
      write_partial(&queue, &gpu.struct_buf, struct_bytes, lo, hi);
    }
    for (lo, hi) in snap.dirty.leaves_ranges.iter().copied() {
      write_partial(&queue, &gpu.leaves, leaves_bytes, lo, hi);
    }
    if snap.dirty.palette_changed {
      // palette 2048B 太小，整块写
      queue.write_buffer(&gpu.palette, 0, palette_bytes);
    }
  }
  // state / comp 每次都整块写（state 4KB、comp 在 MVP 3 tiles 下是 64KB，都很小）
  write(&device, &queue, &mut gpu.state, "gate_state", &snap.state_bytes);
  // comp: 每个 tile 1 字；build 后可能为 0 字节，ensure 至少 4B。
  // comp_bytes 只是预估上限；实际内容读 grid 时已经按真实 size 存。
  // MVP 下 comp 数据直接从 CPU 侧构建：UploadSnapshot 当前没带 comp 字节，
  // 这里用 gpu.comp size ≥ 预估的占位（历史行为：仅 buffer 大小对齐）。
  {
    let placeholder = vec![0u8; comp_bytes.max(4)];
    ensure_with_copy(&device, &queue, &mut gpu.comp, "gate_comp", &placeholder);
  }

  gpu.globals.set(snap.buffers.globals);
  gpu.globals.write_buffer(&device, &queue);

  let elapsed = t0.elapsed();
  let cpu_ms = elapsed.as_secs_f32() * 1000.0;
  let tx_bytes_total =
    (struct_tx_bytes + leaves_tx_bytes + palette_tx_bytes + snap.state_bytes.len()) as f64;
  let mb = tx_bytes_total / (1 << 20) as f64;
  // full: tiles = 总 tile_count；incremental: tiles = 本轮 dirty tile 数（不再误导）
  let tiles_show = if is_full {
    snap.buffers.globals.tile_count as usize
  } else {
    snap.dirty_tiles
  };
  info!(target: "gate",
    "UPLOAD[{}]: bytes={:.2}MB (s {}KB,l {}KB,pal {}KB,state 4KB), tiles={}, comp={}KB, elapsed={:?}",
    snap.mode_tag, mb,
    struct_tx_bytes / 1024, leaves_tx_bytes / 1024, palette_tx_bytes / 1024,
    tiles_show, comp_bytes / 1024, elapsed,
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
  // VRAM 预算断言（V3.1 ≤2GB）
  let vram = gpu.struct_buf.size()
    + gpu.leaves.size()
    + gpu.palette.size()
    + gpu.comp.size()
    + gpu.state.size();
  debug_assert!(vram <= 2u64 << 30, "GPU VRAM 超预算: {vram} bytes");
  if !limits.force_multi() {
    debug_assert!(
      struct_bytes.len() as u64 <= limits.max_storage_buffer_binding_size,
      "b_struct {} bytes > binding limit {}",
      struct_bytes.len(),
      limits.max_storage_buffer_binding_size,
    );
  }
  info!(target: "gate",
    "GpuBrickMap: struct_buf={}B leaves={}B palette={}B comp={}B state={}B bind_group_ready=pending(P2.4)",
    gpu.struct_buf.size(), gpu.leaves.size(), gpu.palette.size(),
    gpu.comp.size(), gpu.state.size(),
  );
  // 消费完本帧 snapshot 必须移除；否则 prepare 每帧都读旧 snapshot → 140MB/帧假上传
  commands.remove_resource::<UploadSnapshot>();
}

// ----------------------------------------------------------------------------
// Plugin
// ----------------------------------------------------------------------------

pub struct BrickMapUploadPlugin;
impl Plugin for BrickMapUploadPlugin {
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
  use super::super::wire::{BrickMapBuffers, INDEX_WORDS, STATE_TOTAL_WORDS};
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
    let mut g = gate_voxel::TileGrid::new();
    let t = gate_voxel::TileCoord::new(0, 0, 0);
    assert_eq!(g.get_comp(t, 17), 0);
    g.set_comp(t, 17, 0xABCD);
    assert_eq!(g.get_comp(t, 17), 0xABCD);
    g.set_state(7, 2, 0x42);
    assert_eq!(g.get_state(7, 2), 0x42);
    assert_eq!(g.get_state(99, 0), 0);
    assert_eq!(g.state_table_bytes().len(), 256 * 16);
  }

  #[test]
  fn wire_constants() {
    assert_eq!(TILE_COMP_WORDS * 4, 32768 * 2); // u16[32768] → 64KB
    assert_eq!(STATE_TOTAL_WORDS * 4, 4096); // 256×4×4B
  }

  #[test]
  fn buffer_data_roundtrip() {
    let mut g = gate_voxel::TileGrid::new();
    fill_box(&mut g, glam::IVec3::ZERO, glam::IVec3::splat(8), 4, 1);
    g.set_state(5, 3, 0xCAFEBABE);
    let state = g.state_table_bytes();
    let off = 5 * 16 + 3 * 4; // entry 5 + field 3
    assert_eq!(state[off..off + 4], 0xCAFEBABEu32.to_le_bytes());
    let buffers: BrickMapBuffers = BrickMapBuilder::build_full(&g).buffers().clone();
    let bytes = u8_of_u32(&buffers.b_struct);
    assert_eq!(bytes.len(), buffers.b_struct.len() * 4);
    assert!(buffers.globals.tile_count >= 1);
    assert!(!buffers.b_struct[0..INDEX_WORDS].iter().all(|&w| w == 0));
  }
}
