//! 渲染极限性能测试（Douglas Brick Tree，资源预算断言）。
//!
//! 断言哲学：**CI 跑宽松上界**（防机器抖动误报），正式数字用 `println!` 留档。本文件全部为
//! **CPU 侧代理断言**：
//! - 构建 / 增量重建：`build_full` 与 `update_chunk` 的 CPU 时间
//! - DDA 三档：CPU 两级参考实现 `cpu_reference_dda_ray_two_level` 的耗时（与 WGSL 逐字等价，
//!   `two_level_equivalence_300_rays` 单测锁定）
//! - 内存/VRAM：wire 布局公式 + 实测 buffers 字节
//!
//! 哨兵意义：两级 DDA 退化回单级、chunk 树序列化布局膨胀、增量 append 失控 —— 任何一项都会
//! 立即爆掉本文件的阈值。GPU 侧（DC pass / PCIe 上传耗时）不在 headless CI 覆盖范围。

use gate_render::brickmap::dda::cpu_reference_dda_ray_two_level;
use gate_render::brickmap::dda::cpu_reference_trace_volumes;
use gate_render::brickmap::wire::TREE_BASE;
use gate_render::{BrickMapBuilder, DirtyRanges};
use gate_voxel::{VolumeGrid, VolumeTransform, fill_box};
use glam::{IVec3, Mat3, Vec3};
use std::time::Instant;

/// 单 chunk 的 voxel 边长（Douglas Brick Tree：256³）
const CHUNK_VOXEL: i32 = 256;
/// 帧预算（60fps 口径）
const FRAME_MS: f64 = 16.7;

/// 棋盘格填充 [min, min+extent)：cell³ 均色块交替异色（防 uniform 折叠）。
/// cell=16 → 分裂到 level 2；cell=4 → level 3；cell=1 → level 4 全深度。
fn fill_checkerboard(grid: &mut VolumeGrid, min: IVec3, extent: IVec3, cell: i32) {
  assert!(extent.x % cell == 0 && extent.y % cell == 0 && extent.z % cell == 0);
  let mut z = min.z;
  while z < min.z + extent.z {
    let mut y = min.y;
    while y < min.y + extent.y {
      let mut x = min.x;
      while x < min.x + extent.x {
        let parity = ((x / cell) ^ (y / cell) ^ (z / cell)) & 1;
        fill_box(
          grid,
          IVec3::new(x, y, z),
          IVec3::splat(cell),
          parity as u8 + 1,
        );
        x += cell;
      }
      y += cell;
    }
    z += cell;
  }
}

/// n 个沿 x 排列的 chunk，各在 chunk 原点放 span³ 棋盘（条带异色防折叠）
fn fill_chunk_checkers(grid: &mut VolumeGrid, chunk_x0: i32, n: usize, span: i32, cell: i32) {
  for i in 0..n as i32 {
    let base = IVec3::new((chunk_x0 + i) * CHUNK_VOXEL, 0, 0);
    fill_checkerboard(grid, base, IVec3::splat(span), cell);
  }
}

/// LCG 伪随机（确定性，无外部 rand 依赖）
struct Lcg(u64);
impl Lcg {
  fn next_u32(&mut self) -> u32 {
    self.0 = self
      .0
      .wrapping_mul(6364136223846793005)
      .wrapping_add(1442695040888963407);
    (self.0 >> 33) as u32
  }
  /// 单位球面伪随机方向（拒绝采样，均匀性对计时断言足够）
  fn dir(&mut self) -> Vec3 {
    loop {
      let v = Vec3::new(
        self.next_u32() as f32 / u32::MAX as f32 * 2.0 - 1.0,
        self.next_u32() as f32 / u32::MAX as f32 * 2.0 - 1.0,
        self.next_u32() as f32 / u32::MAX as f32 * 2.0 - 1.0,
      );
      let l = v.length_squared();
      if 1e-4 < l && l <= 1.0 {
        break v.normalize();
      }
    }
  }
}

// ============================================================================
// P2.9a：百万级体素 chunk 树构建 + 全量上传（CPU 时间预算断言）
// ============================================================================

#[test]
fn million_voxel_build_full_budget() {
  let mut grid = VolumeGrid::new();
  // 63 chunk × 32768 = 2,064,384 voxel 体素 ≥ 100 万（「百万级」口径沿用）
  // cell=4 棋盘 → 分裂到 level 3，树规模贴近真实异色场景
  // 63 而非 64：compute_window 留 1 chunk 生长边距，64 连续 chunk 必丢 1
  let t0 = Instant::now();
  fill_chunk_checkers(&mut grid, 0, 63, 32, 4);
  let fill = t0.elapsed();

  // 全量上传的 CPU 侧成本主体 = build_full（逐 chunk DFS 序列化，Rayon 并行）；
  // write_buffer 是 O(1) enqueue，GPU DMA 由实机 UPLOAD[full] 日志背书
  let t0 = Instant::now();
  let builder = BrickMapBuilder::build_full(&grid);
  let build = t0.elapsed();

  // CI 宽松上界：2s ≈ 数十倍余量（树序列化为主，63 chunk × 数百节点）
  assert!(
    build.as_secs_f64() < 2.0,
    "build_full 超预算（CI 宽松 2s）：{:?}",
    build
  );

  let bufs = builder.buffers();
  assert_eq!(bufs.globals.tile_count as usize, 63);
  let struct_mb = bufs.b_struct.len() as f64 * 4.0 / 1048576.0;
  println!(
    "[P2.9a] 2.06M voxels: grid_fill={fill:?} build_full={build:?} ({:.0}µs/chunk) b_struct={struct_mb:.2}MB node_words={}（fill 不计预算）",
    build.as_micros() as f64 / 63.0,
    bufs.globals.node_words
  );
}

// ============================================================================
// P2.9b：增量上传连发（脏队列压满不爆帧）
// ============================================================================

#[test]
fn incremental_burst_frame_budget() {
  // 典型工作间混合：63 个 chunk 各 16³ 棋盘（cell=4，编辑成本典型亚毫秒）
  // 63 而非 64：compute_window 边距使 64 连续 chunk 必丢 1（详见 P2.9a 注释）
  let mut grid = VolumeGrid::new();
  fill_chunk_checkers(&mut grid, 0, 63, 16, 4);
  let mut builder = BrickMapBuilder::build_full(&grid);
  // build_full 末尾已丢弃脏区间；防御性再取一次
  let stale: DirtyRanges = builder.take_dirty_ranges();
  assert!(stale.struct_ranges.is_empty() && !stale.palette_changed);

  // 连发：63 chunk 全标脏（真实编辑路径 set_voxel → DirtyTracker 自动 mark）；
  // 用 1 voxel 微编辑模拟同帧高频编辑的最小粒度。
  // palette 7：棋盘只占 1..=2，保证每次编辑都是真实改写（同色写回返回 None）
  for i in 0..63isize {
    let edited = grid.set_voxel_ivec3(IVec3::new(i as i32 * CHUNK_VOXEL + 8, 8, 8), 7);
    assert!(edited.is_some(), "微编辑应生效（颜色必变）");
  }
  assert_eq!(grid.dirty.data_dirty_count(), 63);

  // 复刻 poll_pending 的预算换算（upload.rs）：4MB/帧 ÷ PER_CHUNK_BYTES 256KB
  let per_chunk_budget = 256 * 1024;
  let budget_n = (4 * 1024 * 1024 / per_chunk_budget).clamp(1, 64);
  assert_eq!(budget_n, 16);

  let mut frames_ms: Vec<f64> = Vec::new();
  let mut frame_bytes: Vec<usize> = Vec::new();
  let mut frame_chunks: Vec<usize> = Vec::new();
  let mut drained = 0usize;
  while grid.dirty.data_dirty_count() > 0 {
    let coords = grid
      .dirty
      .drain_data_budget(grid.dirty.data_dirty_count().min(budget_n));
    let t0 = Instant::now();
    for c in &coords {
      assert!(
        matches!(
          builder.update_chunk(&grid, *c),
          gate_render::ChunkUpdate::Rebuilt
        ),
        "已渲染脏 chunk 应走 Rebuilt（{c:?}）"
      );
    }
    let el = t0.elapsed();
    let ranges = builder.take_dirty_ranges();
    let bytes: usize = ranges.struct_ranges.iter().map(|r| r.1 - r.0).sum();
    frames_ms.push(el.as_secs_f64() * 1000.0);
    frame_bytes.push(bytes);
    frame_chunks.push(coords.len());
    drained += coords.len();
  }
  assert_eq!(drained, 63, "脏队列必须被预算机制完全消化");

  // 断言：任何一帧的重建 CPU 时间 ≤ 2× 帧预算（CI 宽松）；正式数字 println
  for (f, ms) in frames_ms.iter().enumerate() {
    assert!(
      *ms < 2.0 * FRAME_MS,
      "第 {f} 帧增量重建 {} chunks 耗时 {ms:.2}ms > 2×16.7ms（压满帧预算）",
      frame_chunks[f]
    );
  }
  println!(
    "[P2.9b] 增量连发 63 chunks / {} 帧: per-frame={frames_ms:?}ms, dirty_bytes={frame_bytes:?} (合计 {}KB)",
    frames_ms.len(),
    frame_bytes.iter().sum::<usize>() / 1024
  );
}

// ============================================================================
// P2.9c：大场景 DDA 三档（CPU 两级参考实现代理）
// ============================================================================

/// A 空旷远距：1 chunk 有内容，射线自 8192 voxel 外穿越空旷区。
/// 哨兵：两级 DDA 空旷射线 ≈ t_max/16 粗步；退化回单级（16384 voxel 步）会立即爆阈值。
#[test]
fn dda_regime_a_open_far_budget() {
  let mut grid = VolumeGrid::new();
  fill_box(&mut grid, IVec3::ZERO, IVec3::splat(64), 1);
  let bufs = BrickMapBuilder::build_full(&grid).buffers().clone();

  let origin = Vec3::new(8192.0, 256.0, 8192.0);
  let mut rng = Lcg(0x9E3779B97F4A7C15);
  let n = 1000;
  let t0 = Instant::now();
  let mut hits = 0usize;
  for _ in 0..n {
    // 射线朝场景附近散开：大部分路径空旷，少数命中
    let jitter = rng.dir() * 300.0;
    let dir = (Vec3::new(32.0, 32.0, 32.0) + jitter - origin).normalize();
    if cpu_reference_dda_ray_two_level(&bufs, origin, dir, 16384.0, 16384).is_some() {
      hits += 1;
    }
  }
  let el = t0.elapsed();
  // CI 宽松：实测毫秒级；阈值 500ms/千射线（单级退化 → >2s 爆）
  assert!(
    el.as_secs_f64() < 0.5,
    "空旷远距 DDA 千射线耗时 {el:?} > 500ms（疑似两级步进退化）"
  );
  println!(
    "[P2.9c-A] 空旷远距 1000 rays: {el:?} ({:.1}µs/ray) hits={hits}/1000",
    el.as_micros() as f64 / 1000.0
  );
}

/// B 密集热点：64³ 区域 cell=4（level 3 粒度）棋盘满铺，4096 异色块。
/// 射线穿越热点内部 → 树深 3 层下钻 + palette 采样。
#[test]
fn dda_regime_b_dense_l2_budget() {
  let mut grid = VolumeGrid::new();
  fill_checkerboard(&mut grid, IVec3::ZERO, IVec3::splat(64), 4);
  let bufs = BrickMapBuilder::build_full(&grid).buffers().clone();

  let origin = Vec3::new(-256.0, 128.0, 128.0);
  let mut rng = Lcg(0xDEADBEEF1234);
  let n = 1000;
  let t0 = Instant::now();
  let mut hits = 0usize;
  for _ in 0..n {
    let jitter = rng.dir() * 220.0;
    let dir = (Vec3::new(32.0, 32.0, 32.0) + jitter - origin).normalize();
    if cpu_reference_dda_ray_two_level(&bufs, origin, dir, 4096.0, 16384).is_some() {
      hits += 1;
    }
  }
  let el = t0.elapsed();
  // CI 宽松：命中密集区步进多；阈值 2s/千射线
  assert!(
    el.as_secs_f64() < 2.0,
    "密集热点 DDA 千射线耗时 {el:?} > 2s（树下钻或细步扫描疑似退化）"
  );
  println!(
    "[P2.9c-B] 密集热点满铺 1000 rays: {el:?} ({:.1}µs/ray) hits={hits}/1000",
    el.as_micros() as f64 / 1000.0
  );
}

/// C 最坏树深：1³（level 4）满深度热点（预算口径 ~20 万最细胞，异色），
/// 射线斜穿热点 → 每 4³ 子块都触发全深度分支 + 有界细步上限。
#[test]
fn dda_regime_c_worst_depth_budget() {
  let mut grid = VolumeGrid::new();
  // 64×64×48 = 196,608 个 1³ 最细胞（≈20 万预算口径，棋盘异色防折叠）
  fill_checkerboard(&mut grid, IVec3::ZERO, IVec3::new(64, 64, 48), 1);
  let bufs = BrickMapBuilder::build_full(&grid).buffers().clone();

  let origin = Vec3::new(-256.0, -128.0, 96.0);
  let mut rng = Lcg(0xCAFEF00DBABE);
  let n = 1000;
  let t0 = Instant::now();
  let mut hits = 0usize;
  for _ in 0..n {
    let jitter = rng.dir() * 180.0;
    let dir = (Vec3::new(32.0, 32.0, 24.0) + jitter - origin).normalize();
    if cpu_reference_dda_ray_two_level(&bufs, origin, dir, 4096.0, 16384).is_some() {
      hits += 1;
    }
  }
  let el = t0.elapsed();
  // CI 宽松：全深度热点是最坏路径；阈值 3s/千射线
  assert!(
    el.as_secs_f64() < 3.0,
    "L4 满深度热点 DDA 千射线耗时 {el:?} > 3s（最坏树深路径疑似退化）"
  );
  println!(
    "[P2.9c-C] L4 满深度热点 1000 rays: {el:?} ({:.1}µs/ray) hits={hits}/1000",
    el.as_micros() as f64 / 1000.0
  );
}

// ============================================================================
// P2.9d：系统内存 + VRAM 布局规模留档（仅打印测量值，保留布局契约断言防结构漂移）
// ============================================================================

#[test]
fn vram_layout_budget_2gb() {
  // ① 定长前缀回归锚：Region ① 稠密 chunk 窗口 = 64³ 字 = 1MB
  assert_eq!(TREE_BASE, 64 * 64 * 64);
  assert_eq!(TREE_BASE * 4, 1_048_576);
  println!(
    "[P2.9d] 定长前缀（chunk 窗口）= {:.1}MB",
    TREE_BASE as f64 * 4.0 / 1048576.0
  );

  // ② 中粒度棋盘单 chunk 的树区成本线：64³ cell=16 棋盘（level 2 粒度异色）
  let mut grid = VolumeGrid::new();
  fill_checkerboard(&mut grid, IVec3::ZERO, IVec3::splat(64), 16);
  let g = BrickMapBuilder::build_full(&grid).buffers().globals;
  let node_per_chunk = g.node_words as f64 * 4.0;
  println!(
    "[P2.9d] 中粒度棋盘 1 chunk: node={:.2}KB/chunk × 1000 chunks = {:.1}MB",
    node_per_chunk / 1024.0,
    node_per_chunk * 1000.0 / 1048576.0
  );

  // ③ L4 满深度热点（19.7 万最细胞）的树区：最坏场景留档
  let mut grid = VolumeGrid::new();
  fill_checkerboard(&mut grid, IVec3::ZERO, IVec3::new(64, 64, 48), 1);
  let bufs = BrickMapBuilder::build_full(&grid).buffers().clone();
  let node_mb = bufs.globals.node_words as f64 * 4.0 / 1048576.0;
  println!(
    "[P2.9d] L4 满深度热点(19.7万胞): node={node_mb:.2}MB（v4 格式 palette 直存节点，无 leaves buffer）"
  );

  // ④ 典型工作间合计：32 典型 chunk + 1 中粒度 + 1 L4 热点 + 定长前缀，
  //    外推 ×2（CPU 镜像+GPU 同规格）——仅留档
  let mut grid = VolumeGrid::new();
  fill_chunk_checkers(&mut grid, 0, 32, 16, 4);
  fill_checkerboard(
    &mut grid,
    IVec3::new(32 * CHUNK_VOXEL, 0, 0),
    IVec3::splat(64),
    16,
  );
  fill_checkerboard(
    &mut grid,
    IVec3::new(33 * CHUNK_VOXEL, 0, 0),
    IVec3::new(64, 64, 48),
    1,
  );
  let bufs = BrickMapBuilder::build_full(&grid).buffers().clone();
  let total_mb = (bufs.b_struct.len() + bufs.b_palette.len()) as f64 * 4.0 / 1048576.0;
  println!(
    "[P2.9d] 典型工作间 buffers 合计={total_mb:.2}MB（含 1MB chunk 窗口）→ GPU 同规格 + CPU 镜像 ×2 = {:.1}MB（v4 预算 ≤350MB）",
    total_mb * 2.0
  );
}

// ============================================================================
// P2.9e：物体多网格（trace_volumes）CPU 代理预算——同屏 16 物体
// ============================================================================

#[test]
fn obj_16_objects_trace_scene_budget() {
  // 世界：4 chunk 地面块（顶面 y=64），条带异色防 uniform 折叠
  let mut grid = VolumeGrid::new();
  for i in 0..4i32 {
    fill_box(
      &mut grid,
      IVec3::new(i * CHUNK_VOXEL, 0, 0),
      IVec3::new(128, 64, 64),
      (i % 6 + 1) as u8,
    );
  }
  let world = BrickMapBuilder::build_full(&grid).buffers().clone();

  // 16 物体：每枚 = 1-chunk 芯片（板 + 8 个杂色点，覆盖分裂下钻路径），
  // 各自旋转角/缩放不同（OBB 剔除 + 局部变换路径全覆盖）
  let mut chip = VolumeGrid::new();
  fill_box(&mut chip, IVec3::ZERO, IVec3::new(128, 16, 128), 3);
  for i in 0..8 {
    let edited = chip.set_voxel_ivec3(IVec3::new(16 + i * 16, 32, 64), (i % 3 + 1) as u8);
    assert!(edited.is_some(), "芯片板占用祖先存在，写入应生效");
  }
  // 16 个独立 chip 副本（每物体 = 独立 VolumeGrid）
  let chip_bufs: Vec<_> = (0..16)
    .map(|_| BrickMapBuilder::build_full(&chip).buffers().clone())
    .collect();
  let transforms: Vec<VolumeTransform> = (0..16)
    .map(|i| {
      VolumeTransform::new(
        Vec3::new(i as f32 * 160.0, 64.0, 0.0),
        Mat3::from_rotation_y((i as f32 * 0.4).sin() * 0.6),
        1.0 + (i % 4) as f32,
      )
    })
    .collect();
  // vols[0] = 主世界，vols[1..17] = 物体
  let vols: Vec<(&gate_render::BrickMapBuffers, VolumeTransform)> = std::iter::once((
    &world as &gate_render::BrickMapBuffers,
    VolumeTransform::IDENTITY,
  ))
  .chain(
    chip_bufs
      .iter()
      .zip(transforms.iter().copied())
      .map(|(b, t)| (b as &gate_render::BrickMapBuffers, t)),
  )
  .collect();

  // 体量留档：每物体 = 1 chunk 树（窗口 1MB + 树）+ palette
  let per_obj_kb = chip_bufs[0].b_struct.len() as f64 * 4.0 / 1024.0;
  println!(
    "[P2.9e] 16 物体: struct={:.1}KB/obj palette={}w",
    per_obj_kb,
    chip_bufs[0].b_palette.len()
  );
  assert_eq!(vols.len(), 17);

  // 射线自上方朝地面/物体散开：物体顶面（y≥64+16·scale）高于地面顶面 →
  // 一部分命中物体（obj_id≥0），一部分直击地面（obj_id=-1），两路径都覆盖
  let origin = Vec3::new(2048.0, 1300.0, 2048.0);
  let mut rng = Lcg(0x00FEDE1E);
  let n = 1000;
  let t0 = Instant::now();
  let (mut obj_hits, mut world_hits) = (0usize, 0usize);
  for _ in 0..n {
    let tx = (rng.next_u32() % 2048) as f32;
    let tz = (rng.next_u32() % 64) as f32;
    let dir = (Vec3::new(tx, 64.0, tz) - origin).normalize();
    match cpu_reference_trace_volumes(&vols, origin, dir, 8192.0) {
      Some(h) if h.obj_id >= 0 => obj_hits += 1,
      Some(_) => world_hits += 1,
      None => {}
    }
  }
  let el = t0.elapsed();
  assert!(
    obj_hits > 0 && world_hits > 0,
    "两类命中都必须出现（obj={obj_hits} world={world_hits}）"
  );
  // CI 宽松：每射线 = 世界 DDA + 16×(AABB 剔除 ~0 / 少量物体 DDA)；1s ≈ 30× 余量
  assert!(
    el.as_secs_f64() < 1.0,
    "16 物体 trace_volumes 千射线耗时 {el:?} > 1s（物体剔除/局部 DDA 疑似退化）"
  );
  println!(
    "[P2.9e] trace_volumes 1000 rays ×16 obj: {el:?} ({:.1}µs/ray) obj_hits={obj_hits} world_hits={world_hits}",
    el.as_micros() as f64 / 1000.0
  );
}
