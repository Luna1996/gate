//! P2.9 渲染极限性能测试（v3.1 资源预算断言）
//!
//! 断言哲学（v3.1 决策）：**CI 跑宽松上界**（防机器抖动误报），正式数字 `println!`
//! 留档；基准机实测数字记录于 TODO.md P2.9 条目。本文件全部为 **CPU 侧代理断言**：
//! - 构建 / 增量重建：Rayon `build_full` 与 `update_tile` 的 CPU 时间
//! - DDA 三档：CPU 两级参考实现 `cpu_reference_dda_ray_two_level`
//!   （与 WGSL 逐字等价，`two_level_equivalence_300_rays` 单测锁定）的耗时
//! - 内存/VRAM：wire 布局公式 + 实测 buffers 字节（docs/brickmap.md §7 预算表）
//!
//! 哨兵意义：两级 DDA 被改回单级（15× 退化）、寻址链引入意外分配、精细化
//! 序列化布局膨胀——任何一项都会立即爆掉本文件的阈值。
//!
//! GPU 侧（DC pass / PCIe 上传耗时）不在 headless CI 覆盖范围：由实机日志
//! 背书（RTX 4060 dev 构建：DC 2.75ms、UPLOAD[full] 157MB/120ms、
//! UPLOAD[incremental] 0.26MB/~250µs，见 TODO.md P2.4/P2.3 条目）；GPU 全量
//! 池 ≤2GB 的运行时护栏在 prepare 阶段 debug_assert（brickmap.md §10 决议 15）。

use gate_render::brickmap::dda::cpu_reference_dda_ray_two_level;
use gate_render::brickmap::wire::NODE_STREAM_BASE;
use gate_render::{BrickMapBuilder, DirtyRanges, MovObject, TileUpdate, cpu_reference_trace_scene};
use gate_voxel::{TileGrid, fill_box};
use glam::{IVec3, Mat3, Vec3};
use std::time::Instant;

/// 单 tile 的 fine 边长（32 基元胞 × 16 fine）
const TILE_FINE: i32 = 512;
/// 满铺 tile 的基元胞数（L0，4cm）
const CELLS_PER_TILE: usize = 32 * 32 * 32;
/// 帧预算（60fps 口径）
const FRAME_MS: f64 = 16.7;

/// n 个沿 x 排列的满 L0 tile（条带异色防 uniform 折叠，贴近真实场景）
fn fill_full_tiles(grid: &mut TileGrid, tile_x0: i32, n: usize) {
  for i in 0..n {
    let base = IVec3::new((tile_x0 + i as i32) * TILE_FINE, 0, 0);
    let pal = (i % 6 + 1) as u8;
    // 半 tile 条带两色：整体非 uniform，胞间有差异
    fill_box(grid, base, IVec3::new(TILE_FINE / 2, TILE_FINE, TILE_FINE), 0, pal);
    fill_box(
      grid,
      base + IVec3::new(TILE_FINE / 2, 0, 0),
      IVec3::new(TILE_FINE / 2, TILE_FINE, TILE_FINE),
      0,
      pal % 6 + 1,
    );
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

/// 异色棋盘填充 [min, min+extent) 区域（level 精细层），防 uniform 折叠
fn fill_checkerboard(grid: &mut TileGrid, min: IVec3, extent: IVec3, level: u8) {
  let e = match level {
    0 => 16,
    2 => 4,
    4 => 1,
    _ => panic!("测试只覆盖 L0/L2/L4"),
  };
  let mut z = min.z;
  while z < min.z + extent.z {
    let mut y = min.y;
    while y < min.y + extent.y {
      let mut x = min.x;
      while x < min.x + extent.x {
        let parity = ((x / e) ^ (y / e) ^ (z / e)) & 1;
        grid.set_voxel(IVec3::new(x, y, z), level, parity as u8 + 1);
        x += e;
      }
      y += e;
    }
    z += e;
  }
}

// ============================================================================
// P2.9a：百万级体素砖块图构建 + 全量上传（CPU 时间预算断言）
// ============================================================================

#[test]
fn million_voxel_build_full_budget() {
  let mut grid = TileGrid::new();
  // 64 tile × 32768 = 2,097,152 体素 ≥ 100 万（v3.1「百万级」口径）
  let t0 = Instant::now();
  fill_full_tiles(&mut grid, 0, 64);
  let fill = t0.elapsed();

  // 全量上传的 CPU 侧成本主体 = build_full（序列化+放置，Rayon 并行分批）；
  // write_buffer 是 O(1) enqueue，GPU DMA 由实机 UPLOAD[full] 日志背书
  let t0 = Instant::now();
  let builder = BrickMapBuilder::build_full(&grid);
  let build = t0.elapsed();

  // CI 宽松上界：P2.3 基线 211 tile（156MB blob）全量构建 219ms；
  // 64 满铺 tile 同量级，2s ≈ 10× 余量
  assert!(
    build.as_secs_f64() < 2.0,
    "build_full 超预算（CI 宽松 2s）：{:?}",
    build
  );

  let bufs = builder.buffers();
  assert_eq!(bufs.globals.tile_count as usize, 64);
  let struct_mb = bufs.b_struct.len() as f64 * 4.0 / 1048576.0;
  let leaves_mb = bufs.b_leaves.len() as f64 * 4.0 / 1048576.0;
  println!(
    "[P2.9a] 2.1M voxels: grid_fill={fill:?} build_full={build:?} ({:.0}µs/tile) b_struct={struct_mb:.1}MB b_leaves={leaves_mb:.1}MB node_words={}",
    build.as_micros() as f64 / 64.0,
    bufs.globals.node_words
  );
}

// ============================================================================
// P2.9b：增量上传连发（脏队列压满不爆帧）
// ============================================================================

#[test]
fn incremental_burst_frame_budget() {
  // 典型工作间混合：64 个 1/8 体量 tile（均匀为主，编辑成本典型亚毫秒）
  let mut grid = TileGrid::new();
  for i in 0..64isize {
    let base = IVec3::new(i as i32 * TILE_FINE, 0, 0);
    fill_box(&mut grid, base, IVec3::new(TILE_FINE, 64, 256), 0, (i % 6 + 1) as u8);
  }
  let mut builder = BrickMapBuilder::build_full(&grid);
  // build_full 末尾已丢弃脏区间（full_build_leaves_no_stale_dirty_marks）；防御性再取一次
  let stale: DirtyRanges = builder.take_dirty_ranges();
  assert!(stale.struct_ranges.is_empty() && stale.leaves_ranges.is_empty());

  // 连发：64 tile 全标脏（真实编辑路径 set_voxel → DirtyTracker 自动 mark）；
  // 用 L4 微编辑（1 fine 胞）模拟 P4.2 同帧高频编辑的最小粒度。
  // palette 7：填充只占 1..=6，保证每次编辑都是真实改写（同色写回返回 None）
  for i in 0..64isize {
    grid
      .set_voxel(IVec3::new(i as i32 * TILE_FINE + 8, 8, 8), 4, 7)
      .expect("区域内必有已占用祖先，编辑应生效");
  }
  assert_eq!(grid.dirty.data_dirty_count(), 64);

  // 复刻 poll_pending 的预算换算（upload.rs）：4MB/帧 ÷ 每 tile 下限 132KB → 31 tile/帧
  let per_tile_floor = (1024 + 32768) * 4; // TILE_BITMAP_WORDS + CELL_DIR_WORDS
  let budget_n = (4 * 1024 * 1024 / per_tile_floor).clamp(1, 64);
  assert_eq!(budget_n, 31);

  let mut frames_ms: Vec<f64> = Vec::new();
  let mut frame_bytes: Vec<usize> = Vec::new();
  let mut drained = 0usize;
  while grid.dirty.data_dirty_count() > 0 {
    let coords = grid
      .dirty
      .drain_data_budget(grid.dirty.data_dirty_count().min(budget_n));
    let t0 = Instant::now();
    for c in &coords {
      assert!(
        matches!(builder.update_tile(&grid, *c), TileUpdate::Rebuilt),
        "已放 slot 的脏 tile 应走 Rebuilt"
      );
    }
    let el = t0.elapsed();
    let ranges = builder.take_dirty_ranges();
    let bytes: usize = ranges
      .struct_ranges
      .iter()
      .map(|r| r.1 - r.0)
      .chain(ranges.leaves_ranges.iter().map(|r| r.1 - r.0))
      .sum();
    frames_ms.push(el.as_secs_f64() * 1000.0);
    frame_bytes.push(bytes);
    drained += coords.len();
  }
  assert_eq!(drained, 64, "脏队列必须被预算机制完全消化");

  // 断言：任何一帧的重建 CPU 时间 ≤ 2× 帧预算（CI 宽松）；正式数字 println
  for (f, ms) in frames_ms.iter().enumerate() {
    assert!(
      *ms < 2.0 * FRAME_MS,
      "第 {f} 帧增量重建 {} tiles 耗时 {ms:.2}ms > 2×16.7ms（压满帧预算）",
      if f == 0 { budget_n } else { budget_n }
    );
  }
  println!(
    "[P2.9b] 增量连发 64 tiles / {} 帧: per-frame={frames_ms:?}ms, dirty_bytes={frame_bytes:?} (合计 {}KB)",
    frames_ms.len(),
    frame_bytes.iter().sum::<usize>() / 1024
  );
}

// ============================================================================
// P2.9c：大场景 DDA 三档（CPU 两级参考实现代理）
// ============================================================================

/// A 空旷远距：1 tile 有内容，射线自 8192 fine 外穿越空旷区。
/// 哨兵：两级 DDA 空旷射线 ≈ t_max/16 粗步；若退化回单级（16384 fine 步），
/// 耗时 ×15 立即爆阈值（对应实机 DC 43ms→2.75ms 改造的回归锚）。
#[test]
fn dda_regime_a_open_far_budget() {
  let mut grid = TileGrid::new();
  fill_box(&mut grid, IVec3::ZERO, IVec3::splat(TILE_FINE), 0, 1);
  let bufs = BrickMapBuilder::build_full(&grid).buffers().clone();

  let origin = Vec3::new(8192.0, 256.0, 8192.0);
  let mut rng = Lcg(0x9E3779B97F4A7C15);
  let n = 1000;
  let t0 = Instant::now();
  let mut hits = 0usize;
  for _ in 0..n {
    // 射线朝场景附近散开：大部分路径空旷，少数命中
    let jitter = rng.dir() * 300.0;
    let dir = (Vec3::new(256.0, 256.0, 256.0) + jitter - origin).normalize();
    if cpu_reference_dda_ray_two_level(&bufs, origin, dir, 16384.0, 16384).is_some() {
      hits += 1;
    }
  }
  let el = t0.elapsed();
  // CI 宽松：实测毫秒级；阈值 500ms/千射线 ≈ 50× 余量（单级退化 → >2s 爆）
  assert!(
    el.as_secs_f64() < 0.5,
    "空旷远距 DDA 千射线耗时 {el:?} > 500ms（疑似两级步进退化）"
  );
  println!(
    "[P2.9c-A] 空旷远距 1000 rays: {el:?} ({:.1}µs/ray) hits={hits}/1000",
    el.as_micros() as f64 / 1000.0
  );
}

/// B 密集热点：tile 内 16³ 基元胞区域 L2（1cm）满铺异色，26.2 万精细胞。
/// 射线穿越热点内部 → 细步 + 全寻址链采样。
#[test]
fn dda_regime_b_dense_l2_budget() {
  let mut grid = TileGrid::new();
  // 16³ L0 胞 = 256³ fine，L2 胞 4³ fine → (256/4)³ = 262,144 胞
  fill_checkerboard(&mut grid, IVec3::ZERO, IVec3::splat(256), 2);
  let bufs = BrickMapBuilder::build_full(&grid).buffers().clone();

  let origin = Vec3::new(-256.0, 128.0, 128.0);
  let mut rng = Lcg(0xDEADBEEF1234);
  let n = 1000;
  let t0 = Instant::now();
  let mut hits = 0usize;
  for _ in 0..n {
    let jitter = rng.dir() * 220.0;
    let dir = (Vec3::new(128.0, 128.0, 128.0) + jitter - origin).normalize();
    if cpu_reference_dda_ray_two_level(&bufs, origin, dir, 4096.0, 16384).is_some() {
      hits += 1;
    }
  }
  let el = t0.elapsed();
  // CI 宽松：命中密集区细步多；阈值 2s/千射线
  assert!(
    el.as_secs_f64() < 2.0,
    "密集 L2 热点 DDA 千射线耗时 {el:?} > 2s（寻址链或细步扫描疑似退化）"
  );
  println!(
    "[P2.9c-B] 密集 L2 满铺 1000 rays: {el:?} ({:.1}µs/ray) hits={hits}/1000",
    el.as_micros() as f64 / 1000.0
  );
}

/// C 最坏树深：L4（0.25cm）满深度热点（预算表口径 ~20 万最细胞，异色），
/// 射线斜穿热点 → 每 L0 胞都触发全深度分支 + 有界细步上限。
#[test]
fn dda_regime_c_worst_depth_budget() {
  let mut grid = TileGrid::new();
  // 4×4×3 L0 胞 × 4096 L4 胞 = 196,608 L4 胞（≈20 万预算口径）
  fill_checkerboard(&mut grid, IVec3::ZERO, IVec3::new(256, 256, 192), 4);
  let bufs = BrickMapBuilder::build_full(&grid).buffers().clone();

  let origin = Vec3::new(-256.0, -128.0, 96.0);
  let mut rng = Lcg(0xCAFEF00DBABE);
  let n = 1000;
  let t0 = Instant::now();
  let mut hits = 0usize;
  for _ in 0..n {
    let jitter = rng.dir() * 180.0;
    let dir = (Vec3::new(128.0, 96.0, 96.0) + jitter - origin).normalize();
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
// P2.9d：系统内存 + VRAM 布局预算断言（≤2GB，工作间/关卡口径）
// ============================================================================

#[test]
fn vram_layout_budget_2gb() {
  // ① 定长前缀回归锚：index 8MB + bitmaps 4MB + dirs 128MB = 140MB
  //    （TILE_CAP=1024 的占位虚耗，brickmap.md §6「不做 rank 压缩」）
  assert_eq!(NODE_STREAM_BASE * 4, 36_700_160 * 4);
  let fixed_mb = NODE_STREAM_BASE as f64 * 4.0 / 1048576.0;
  assert!(fixed_mb < 150.0, "定长前缀 {fixed_mb:.1}MB 意外膨胀");

  // ② L2 精细化满铺单 tile 的 node stream 成本线：
  //    预算表 §7「L2 精细化满铺 4M L0 胞 × 152B = 608MB」→ 单 tile（32768 L0 胞）
  //    ≈ 4.98MB。异色棋盘防折叠，实测外推工作间 122 tiles（256m³）≤ 670MB。
  let mut grid = TileGrid::new();
  fill_checkerboard(&mut grid, IVec3::ZERO, IVec3::splat(TILE_FINE), 2);
  let g = BrickMapBuilder::build_full(&grid).buffers().globals;
  let node_per_tile = g.node_words as f64 * 4.0;
  println!(
    "[P2.9d] L2 满铺 1 tile: node={:.2}MB/tile（预算线 4.98MB×1.15）× 122 tiles = {:.0}MB（608MB 预算）",
    node_per_tile / 1048576.0,
    node_per_tile * 122.0 / 1048576.0
  );
  assert!(
    node_per_tile <= 152.0 * CELLS_PER_TILE as f64 * 1.15,
    "L2 精细化单 tile node {:.2}MB > 152B/胞 ×1.15（布局退化或折叠失效）",
    node_per_tile / 1048576.0
  );
  assert!(
    node_per_tile * 122.0 <= 670.0 * 1048576.0,
    "工作间 122 tile L2 满铺外推 {:.0}MB > 608MB×1.10",
    node_per_tile * 122.0 / 1048576.0
  );

  // ③ L4 满深度热点（19.7 万最细胞）的 node + brick 实测：预算表「L4 热点 ~1GB /
  //    BrickPool ≤1GB」口径。20 万胞只占总预算一小部分，实测留档 + 宽断言。
  let mut grid = TileGrid::new();
  fill_checkerboard(&mut grid, IVec3::ZERO, IVec3::new(256, 256, 192), 4);
  let bufs = BrickMapBuilder::build_full(&grid).buffers().clone();
  let node_mb = bufs.globals.node_words as f64 * 4.0 / 1048576.0;
  let leaves_mb = bufs.b_leaves.len() as f64 * 4.0 / 1048576.0;
  println!(
    "[P2.9d] L4 满深度热点(19.7万胞): node={node_mb:.1}MB leaves={leaves_mb:.1}MB bricks={}",
    bufs.globals.brick_slabs
  );
  // 20 万胞热点外推到预算口径（~1GB 上限）远有余；断言锁数量级防意外分配
  assert!(node_mb + leaves_mb < 256.0, "20 万胞热点 node+brick {node_mb:.1}+{leaves_mb:.1}MB 超 256MB 数量级锚");

  // ④ 典型工作间合计：64 典型 tile + 1 L2 满 tile + 1 L4 热点 + 定长前缀，
  //    外推 ×2（CPU 镜像+GPU 同规格）远低于 2GB（v3.1 系统内存/VRAM 决策）
  let mut grid = TileGrid::new();
  fill_full_tiles(&mut grid, 0, 62);
  fill_checkerboard(&mut grid, IVec3::new(62 * TILE_FINE, 0, 0), IVec3::splat(TILE_FINE), 2);
  fill_checkerboard(
    &mut grid,
    IVec3::new(63 * TILE_FINE, 0, 0),
    IVec3::new(256, 256, 192),
    4,
  );
  let bufs = BrickMapBuilder::build_full(&grid).buffers().clone();
  let total_mb = (bufs.b_struct.len() + bufs.b_leaves.len() + bufs.b_palette.len()) as f64 * 4.0
    / 1048576.0;
  println!(
    "[P2.9d] 典型工作间 buffers 合计={total_mb:.1}MB（含 140MB 定长前缀）→ GPU 同规格 + CPU 镜像 ×2 = {:.0}MB ≤ 2GB ✓",
    total_mb * 2.0
  );
  assert!(
    total_mb * 2.0 <= 2048.0,
    "工作间双份（GPU+CPU 镜像）{:.0}MB > 2GB",
    total_mb * 2.0
  );
}

// ============================================================================
// P2.9e：MOV 多网格（P2.10 trace_scene）CPU 代理预算——同屏 16 物体
// ============================================================================

#[test]
fn mov_16_objects_trace_scene_budget() {
  use gate_render::OBJ_WORLD;

  // 世界：8 满铺 tile 地面（顶面 y=512）
  let mut grid = TileGrid::new();
  fill_full_tiles(&mut grid, 0, 8);
  let world = BrickMapBuilder::build_full(&grid).buffers().clone();

  // 16 物体：每枚 = 1-tile 芯片（L0 板 + 8 个 L4 杂色点，覆盖 brick slab 路径），
  // 各自旋转角/缩放不同（OBB 剔除 + 局部变换路径全覆盖）
  let mut chip = TileGrid::new();
  fill_box(&mut chip, IVec3::ZERO, IVec3::new(128, 16, 128), 0, 3);
  for i in 0..8 {
    chip
      .set_voxel(IVec3::new(16 + i * 16, 32, 64), 4, (i % 3 + 1) as u8)
      .expect("L0 板占用祖先存在，写入应生效");
  }
  let chip_bufs = BrickMapBuilder::build_full(&chip).buffers().clone();
  let objs: Vec<MovObject> = (0..16)
    .map(|i| MovObject {
      buffers: &chip_bufs,
      pos: Vec3::new(i as f32 * 512.0, 512.0, 512.0),
      rot: Mat3::from_rotation_y((i as f32 * 0.4).sin() * 0.6),
      scale: 1.0 + (i % 4) as f32,
    })
    .collect();
  let pool = gate_render::pack_mov_pool(&objs);

  // pool 体量留档：v1 每物体 = bitmap 1024w + dirs 32768w + node stream
  let per_obj_kb = pool.mov_struct.len() as f64 * 4.0 / 1024.0 / objs.len() as f64;
  println!(
    "[P2.9e] 16 物体 pool: struct={:.1}KB/obj leaves={}w palette={}w descs={}×128B",
    per_obj_kb,
    pool.mov_leaves.len(),
    pool.mov_palette.len(),
    pool.descs.len()
  );
  assert_eq!(pool.descs.len(), 16);

  // 射线自上方朝地面/物体散开：物体顶面（y≥512+32·scale）近于世界顶面 →
  // 一部分命中物体（obj!=WORLD），一部分直击地面（WORLD），两路径都覆盖
  let origin = Vec3::new(4096.0, 2600.0, 4096.0);
  let mut rng = Lcg(0x00FEDE1E);
  let n = 1000;
  let t0 = Instant::now();
  let (mut obj_hits, mut world_hits) = (0usize, 0usize);
  for _ in 0..n {
    let tx = (rng.next_u32() % 8192) as f32;
    let tz = (rng.next_u32() % 4096) as f32;
    let dir = (Vec3::new(tx, 512.0, tz) - origin).normalize();
    match cpu_reference_trace_scene(&world, &pool, origin, dir, 16384.0) {
      Some(h) if h.obj != OBJ_WORLD => obj_hits += 1,
      Some(_) => world_hits += 1,
      None => {}
    }
  }
  let el = t0.elapsed();
  assert!(obj_hits > 0 && world_hits > 0, "两类命中都必须出现（obj={obj_hits} world={world_hits}）");
  // CI 宽松：每射线 = 世界 DDA + 16×(AABB 剔除 ~0 / 少量物体 DDA)；1s ≈ 30× 余量
  assert!(
    el.as_secs_f64() < 1.0,
    "16 物体 trace_scene 千射线耗时 {el:?} > 1s（物体剔除/局部 DDA 疑似退化）"
  );
  println!(
    "[P2.9e] trace_scene 1000 rays ×16 obj: {el:?} ({:.1}µs/ray) obj_hits={obj_hits} world_hits={world_hits}",
    el.as_micros() as f64 / 1000.0
  );
}
