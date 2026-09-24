//! M5 · 生产管线（`docs/editable-gigavoxel.md` §4 M5）：把"按坐标产出一棵 chunk 树"从主线程摘出去。
//!
//! M5 之前：`infinite_cubes::stream_chunks` 在主线程同步 `build_region`、每帧一个 chunk —— 生成是
//! ms 级 ⇒ 帧额只能给 1，收敛速度被这一条卡死。本模块提供 [`ChunkSource`]（实现方 = 程序化 /
//! `.vox` 区域 / 磁盘按层文件）与 [`ChunkProducer`]（N 个 worker 线程 + 去重派发 + 非阻塞取回）：
//! 主线程只剩"派发需求"和"挂载产出"两件廉价的事，重活并行在后台。
//!
//! **只依赖 std**（`gate-voxel` 是纯逻辑层，唯一外部依赖 `glam`）：`std::thread` +
//! `std::sync::mpsc`。挂载走既有 [`VolumeGrid::mount_chunk_tree`]（它自会标脏 ⇒ 走既有上传路径）。

use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender};

use crate::chunk_tree::ChunkTree;
use crate::coords::ChunkCoord;
use crate::volume::VolumeGrid;

/// 产出档位（"粗到细流式"的档）。**声明序即精细度序**（`Coarse < Full`，见 `#[derive(PartialOrd)]`）：
/// 消费端用 `>=` 判"已有的够不够细"（粗档要细化时重新产出即可，挂载会整体替换）。
/// - [`Detail::Full`]：逐体素（1 体素 = 2cm，现状口径）；
/// - [`Detail::Coarse`]：**16³ 量化（32cm 块）** —— 每个 16³ 格一个代表材质，节点数 / 树大小远小于
///   全分辨率。远景用它 ⇒ 同样的内存/帧额把视距推远（`docs/editable-gigavoxel.md` §10.3 第一条）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Detail {
  Coarse,
  Full,
}

/// 生产源：按坐标产出一棵 chunk 树。
///
/// CONSTRAINT: 实现方在 **worker 线程**上被调用 ⇒ 必须 `Send + Sync`，且**不得碰主线程的
/// `VolumeGrid`**；`scratch` 是 worker 线程独占、可复用的暂存（实现方拿它当中转，避免每次分配）。
pub trait ChunkSource: Send + Sync + 'static {
  /// 产出 `coord` 处的 chunk 树（`None` = 这个 chunk 没有内容，调用方不必挂载）。
  fn produce(&self, coord: ChunkCoord, detail: Detail, scratch: &mut VolumeGrid) -> Option<ChunkTree>;
}

/// worker 回传的一条产出。
struct Done {
  coord: ChunkCoord,
  detail: Detail,
  tree: Option<ChunkTree>,
}

/// 后台生产者：`workers` 个线程，每条消息恰好派给一个 worker（每 worker 一条队列，轮转派发）。
///
/// 用法（主线程，每帧）：先 [`Self::request`] 派需求（多派几条，`max_inflight` 封顶），再
/// [`Self::poll`] 取回成品去挂载。**去重**由 `inflight` 负责：同一个 chunk 不会同时产两份。
pub struct ChunkProducer {
  queues: Vec<Sender<(ChunkCoord, Detail)>>,
  next: usize,
  rx: Receiver<Done>,
  /// 已派发、还没取回的 chunk（去重 + 在飞上限）
  inflight: std::collections::HashSet<ChunkCoord>,
  /// 在飞上限：队列再长也只是排队（不加速），但太小会让 worker 空转 —— 粗档 chunk 只花几十 µs，
  /// 一帧就能吃掉几十条 ⇒ 按 worker 数的 32 倍留队列，真正的节流交给消费端的挂载预算。
  max_inflight: usize,
  workers: Vec<std::thread::JoinHandle<()>>,
}

impl ChunkProducer {
  /// 起 `workers` 个 worker（至少 1）。`workers` 取 CPU 核数的一半到全核都行；生成是纯 CPU、
  /// 与主线程争核，本仓默认给 2（见 `infinite_cubes`）。
  pub fn new(source: Arc<dyn ChunkSource>, workers: usize) -> Self {
    let workers = workers.max(1);
    let mut queues = Vec::with_capacity(workers);
    let mut handles = Vec::with_capacity(workers);
    let (done_tx, rx) = std::sync::mpsc::channel::<Done>();
    for i in 0..workers {
      let (tx, job_rx) = std::sync::mpsc::channel::<(ChunkCoord, Detail)>();
      let src = source.clone();
      let out = done_tx.clone();
      handles.push(
        std::thread::Builder::new()
          .name(format!("gate-chunk-{i}"))
          .spawn(move || {
            // worker 线程独占一份暂存：`VolumeGrid` 的调色板表是 512KB，建一次反复用
            let mut scratch = VolumeGrid::new();
            for (coord, detail) in job_rx.iter() {
              let tree = src.produce(coord, detail, &mut scratch);
              if out.send(Done { coord, detail, tree }).is_err() {
                return; // 主线程没了
              }
            }
            // 队列端断开（主线程 drop）⇒ 退出
          })
          .expect("spawn chunk worker"),
      );
      queues.push(tx);
    }
    drop(done_tx); // 只留 worker 手里的发送端
    Self {
      queues,
      next: 0,
      rx,
      inflight: Default::default(),
      max_inflight: workers * 32,
      workers: handles,
    }
  }

  /// 派发一条需求；`false` = 这次没派（已在飞 / 在飞已满 ⇒ 调用方下一帧再试）。
  pub fn request(&mut self, coord: ChunkCoord, detail: Detail) -> bool {
    if self.inflight.len() >= self.max_inflight || !self.inflight.insert(coord) {
      return false;
    }
    // 轮转派发；某条队列断了（那个 worker 退了）就换下一条 —— 只试一圈，全断才算失败。
    // （否则一个死掉的 worker 会让"表头那条"每帧都派不出去 ⇒ 表面上"加载停了"。）
    for _ in 0..self.queues.len() {
      let i = self.next % self.queues.len();
      self.next = self.next.wrapping_add(1);
      if self.queues[i].send((coord, detail)).is_ok() {
        return true;
      }
    }
    self.inflight.remove(&coord);
    false
  }

  /// 在飞的条数（需求派得出去、还没回来）
  pub fn inflight(&self) -> usize {
    self.inflight.len()
  }

  /// 这个坐标是否已在飞（消费端用来跳过"已派发但还没回来"的那些，免得重复入队或空转）
  pub fn in_flight(&self, coord: ChunkCoord) -> bool {
    self.inflight.contains(&coord)
  }

  /// 取回已完成的产出（非阻塞，本次最多 `max` 条）。返回的 `tree = None` 表示"这个 chunk 没有内容"，
  /// 调用方应把它记成"已生成、无内容"（免得每帧重复派发）。
  pub fn poll(&mut self, max: usize) -> Vec<(ChunkCoord, Detail, Option<ChunkTree>)> {
    let mut out = Vec::new();
    while out.len() < max {
      match self.rx.try_recv() {
        Ok(d) => {
          self.inflight.remove(&d.coord);
          out.push((d.coord, d.detail, d.tree));
        }
        Err(std::sync::mpsc::TryRecvError::Empty) => break,
        Err(std::sync::mpsc::TryRecvError::Disconnected) => break, // worker 全退：不再有产出
      }
    }
    out
  }
}

impl Drop for ChunkProducer {
  fn drop(&mut self) {
    // 丢掉派发端 ⇒ worker 的 `job_rx.iter()` 结束 ⇒ 线程退出；再 join 保证句柄不泄漏。
    self.queues.clear();
    for h in self.workers.drain(..) {
      let _ = h.join();
    }
  }
}
