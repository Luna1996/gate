//! P1.7 数据层极限性能测试
//!
//! 预算原则（v3.1 决策表）：
//! - **内存**：深尺寸断言（`TileGrid::memory_usage`），超限即 fail；工作间规模 ≤2GB（体素数据+镜像的 CPU 侧）
//! - **CPU 时间**：CI 跑宽松上界（本机实测 ×10 左右，防机器抖动误报），只防数量级回归；
//!   正式数字以基准机报告为准。所有实测数字经 `println!` 进日志（CI 用 `-- --nocapture`）
//! - 两个 >2GB RSS 的重测试用 `SERIAL_LOCK` 串行，避免并行峰值互相挤压、计时失真

#[cfg(test)]
use std::sync::Mutex;

#[cfg(test)]
use std::time::{Duration, Instant};

#[cfg(test)]
use glam::IVec3;

#[cfg(test)]
use crate::coords::TileCoord;
#[cfg(test)]
use crate::dirty::DirtyTracker;
#[cfg(test)]
use crate::grid::TileGrid;

#[cfg(test)]
const GB: usize = 1 << 30;
#[cfg(test)]
const MB: usize = 1 << 20;

#[cfg(test)]
static SERIAL_LOCK: Mutex<()> = Mutex::new(());

/// 极限一：单 Tile 最坏细分——32768 基元胞全部细化到 L4（0.25cm）
///
/// 每胞写 1 个 L4 体素：触发 uniform 打散（584 槽回填）+ brick 4096 槽分配 + 祖先掩码更新。
/// 验证：构建耗时、单次编辑/查询延迟仍是 O(树深)、内存上界。
#[test]
fn single_tile_worst_refinement() {
  let mut grid = TileGrid::new();
  let t0 = Instant::now();
  for cz in 0..32 {
    for cy in 0..32 {
      for cx in 0..32 {
        // 基元胞原点（每轴 ×16 最细格），palette 取非零确定值
        let fine = IVec3::new(cx, cy, cz) * 16;
        let p = ((cx + cy * 32 + cz * 1024) % 255 + 1) as u8;
        grid.set_voxel(fine, 4, p);
      }
    }
  }
  let build = t0.elapsed();

  assert_eq!(grid.tile_count(), 1);
  let usage = grid.memory_usage();
  assert_eq!(usage.cell_count, 32 * 32 * 32);
  // 读回抽查：3 个角 + 中心
  for f in [
    IVec3::new(0, 0, 0),
    IVec3::new(31, 0, 0) * 16,
    IVec3::new(0, 31, 31) * 16,
    IVec3::new(16, 16, 16),
  ] {
    assert!(grid.get_voxel(f).is_some(), "fine {f:?} 应有体素");
  }

  // 已细分 Tile 上的单次编辑 / 查询必须仍是 O(树深≤5)，不受规模影响
  let t1 = Instant::now();
  grid.set_voxel(IVec3::new(6, 5, 5), 4, 7); // cell 0 的另一个 L4 槽（真写入路径）
  let edit = t1.elapsed();
  let t2 = Instant::now();
  let _ = grid.get_voxel(IVec3::new(100, 200, 300));
  let query = t2.elapsed();

  println!(
    "P1.7 最坏单Tile: build={build:?} (32768 胞全 L4), 单次编辑={edit:?}, 单次查询={query:?}, 内存={} MB (cell_heap≈{} KB/胞)",
    usage.total_bytes() / MB,
    usage.tile_heap_bytes / 32 * 32 / (32 * 32 * 32) / 1024,
  );

  assert!(build < Duration::from_secs(10), "构建超预算: {build:?}");
  assert!(edit < Duration::from_millis(1), "单次编辑超预算: {edit:?}");
  assert!(
    query < Duration::from_millis(1),
    "单次查询超预算: {query:?}"
  );
  assert!(
    usage.total_bytes() <= 256 * MB,
    "最坏单 Tile 内存超上界: {} MB",
    usage.total_bytes() / MB
  );
}

/// 极限二：百万非空 Tile 稀疏扩张（沙盒稳健性场景）
///
/// 100×100×100 = 1,000,000 个 Tile 各含 1 个 L0 体素。
/// 断言：不崩溃（进程存活即通过）、读回正确、脏队列全量可 drain、时间上界。
/// 内存语义已裁决（v3.1）：≤2GB 限工作间/关卡场景；沙盒极值 = 不崩溃 + 数字上报
/// （occupancy 4KB/Tile 算术下限 ≈4GB，装箱后实测 4.2GB），8GB 为崩溃护栏。
#[test]
fn million_tile_sparse_expansion() {
  // poison 容忍：其他重测试 panic 不应连坐本测试
  let _serial = SERIAL_LOCK.lock().unwrap_or_else(|p| p.into_inner());
  const TILES: usize = 1_000_000;
  let mut grid = TileGrid::new();
  let t0 = Instant::now();
  for i in 0..TILES {
    let fine = IVec3::new(
      (i % 100) as i32,
      ((i / 100) % 100) as i32,
      (i / 10_000) as i32,
    ) * 512;
    grid.set_voxel(fine, 0, 1);
  }
  let build = t0.elapsed();

  assert_eq!(grid.tile_count(), TILES);
  // 读回抽查：首 / 中 / 尾 + 未写 tile
  assert_eq!(grid.get_voxel(IVec3::ZERO), Some(1));
  assert_eq!(grid.get_voxel(IVec3::splat(512 * 50)), Some(1));
  assert_eq!(
    grid.get_voxel(IVec3::new(512 * 99, 512 * 99, 512 * 99)),
    Some(1)
  );
  assert_eq!(grid.get_voxel(IVec3::new(512 * 100, 0, 0)), None);

  // 脏队列全量 drain（模拟 16 帧 × 65536 预算）
  let t1 = Instant::now();
  let mut drained = 0;
  while grid.dirty.data_dirty_count() > 0 {
    drained += grid.dirty.drain_data_budget(65_536).len();
  }
  let drain = t1.elapsed();
  assert_eq!(drained, TILES);

  let usage = grid.memory_usage();
  println!(
    "P1.7 百万Tile: build={build:?}, drain={drain:?}, tiles={}, 内存={} MB (inline={} MB, heap={} MB, dirty={} MB)",
    usage.tile_count,
    usage.total_bytes() / MB,
    usage.tile_inline_bytes / MB,
    usage.tile_heap_bytes / MB,
    usage.dirty_bytes / MB,
  );

  assert!(build < Duration::from_secs(30), "扩张超预算: {build:?}");
  assert!(drain < Duration::from_secs(10), "drain 超预算: {drain:?}");
  // 崩溃护栏（非正式预算）：当前实现 1M Tile 实测 ≈4.7GB，正式语义待裁决
  assert!(
    usage.total_bytes() < 8 * GB,
    "百万 Tile 内存异常: {} MB",
    usage.total_bytes() / MB
  );
}

/// 极限三：batch_edit 吞吐——1M 个 L4 体素单批写入（100³ 区域，343 基元胞）
///
/// 验证：吞吐上界、跨胞脏去重、同色重放全 no-op（O(1) 判定路径）。
#[test]
fn batch_edit_throughput_1m() {
  let mut grid = TileGrid::new();
  let ops: Vec<(IVec3, u8, u8)> = (0..1_000_000)
    .map(|i| {
      let p = (i % 254 + 1) as u8;
      (IVec3::new(i % 100, (i / 100) % 100, i / 10_000), 4u8, p)
    })
    .collect();

  let t0 = Instant::now();
  let applied = grid.batch_edit(ops.iter().copied());
  let build = t0.elapsed();
  assert_eq!(applied, 1_000_000);
  assert_eq!(
    grid.dirty.data_dirty_count(),
    1,
    "单 Tile 内 1M 编辑应去重为 1"
  );

  // 同色重放：全部 no-op，且不得再产生脏标记
  let t1 = Instant::now();
  assert_eq!(grid.batch_edit(ops.iter().copied()), 0);
  let noop = t1.elapsed();
  assert_eq!(grid.dirty.data_dirty_count(), 1);

  let usage = grid.memory_usage();
  println!(
    "P1.7 批量编辑: 1M L4 写入={build:?} (≈{:.0} 万/秒), 同色重放={noop:?}, 内存={} MB",
    1_000_000.0 / build.as_secs_f64() / 10_000.0,
    usage.total_bytes() / MB,
  );

  assert!(build < Duration::from_secs(10), "吞吐超预算: {build:?}");
  assert!(noop < build, "no-op 路径不应慢于首次写入");
  assert!(usage.total_bytes() <= 2 * GB);
}

/// 极限四：脏队列满载——1M mark（含 50% 重复）+ 按帧预算 drain
#[test]
fn dirty_tracker_flood_1m() {
  let mut d = DirtyTracker::new();
  let t0 = Instant::now();
  for i in 0..1_000_000u32 {
    let c = TileCoord::new((i % 1000) as i32, (i / 1000) as i32, 7);
    d.mark_data(c);
    d.mark_data(c); // 重复走去重路径
  }
  let mark = t0.elapsed();
  assert_eq!(d.data_dirty_count(), 1_000_000);

  // 16 帧 × 65536 预算 drain，FIFO 顺序校验（首元素 = 第一个标记的）
  let t1 = Instant::now();
  let mut drained = 0;
  let mut first: Option<TileCoord> = None;
  while d.data_dirty_count() > 0 {
    let batch = d.drain_data_budget(65_536);
    if first.is_none() {
      first = batch.first().copied();
    }
    drained += batch.len();
  }
  let drain = t1.elapsed();
  assert_eq!(drained, 1_000_000);
  assert_eq!(first, Some(TileCoord::new(0, 0, 7)));

  println!("P1.7 脏队列: 2M mark(1M 去重)={mark:?}, drain={drain:?}");

  assert!(mark < Duration::from_secs(10), "mark 超预算: {mark:?}");
  assert!(drain < Duration::from_secs(5), "drain 超预算: {drain:?}");
}

/// 极限五：工作间规模内存预算——8×8×4m @ L0 全铺 = 4M 基元胞
///
/// v3.1 决策：体素数据（CPU 侧权威）≤2GB。这是预算断言的正式场景，
/// 失败即说明数据结构布局需要优化（实测数字进日志）。
#[test]
fn workspace_scale_memory_budget() {
  let _serial = SERIAL_LOCK.lock().unwrap_or_else(|p| p.into_inner());
  let mut grid = TileGrid::new();
  let t0 = Instant::now();
  // 8m = 3200 最细格 → 200 基元胞/轴；4m → 100
  for z in 0..100 {
    for y in 0..200 {
      for x in 0..200 {
        grid.set_voxel(IVec3::new(x, y, z) * 16, 0, 1);
      }
    }
  }
  let build = t0.elapsed();

  let usage = grid.memory_usage();
  println!(
    "P1.7 工作间规模: 4M 基元胞 build={build:?}, tiles={}, 内存={} MB (inline={} MB, heap={} MB) / 预算 2048 MB",
    usage.tile_count,
    usage.total_bytes() / MB,
    usage.tile_inline_bytes / MB,
    usage.tile_heap_bytes / MB,
  );

  assert!(build < Duration::from_secs(30), "构建超预算: {build:?}");
  assert!(
    usage.total_bytes() <= 2 * GB,
    "工作间规模内存超预算: {} MB > 2048 MB（Cell 布局需优化，见裁决记录）",
    usage.total_bytes() / MB
  );
}
