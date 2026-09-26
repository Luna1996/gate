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

/// 产出档位（"粗到细流式"的梯级）。值 = **格粒度**：`grain() = 4^(4-值)` 体素 ——
/// `0` = 整 chunk 一格（5.12 m）、`1` = 64³（1.28 m）、`2` = 16³（32 cm）、`3` = 4³（8 cm）、
/// `4` = 逐体素（2 cm）。**声明序即精细度序**（`Detail(0) < Detail(4)`，见 `#[derive(PartialOrd)]`）：
/// 消费端用 `>=` 判"已有的够不够细"（不够细就重新产出，挂载会整体替换）。
///
/// CONSTRAINT：**档位由"这一级在当前屏幕上是否 ≤ 1 px"定，不由半径拍**（`docs/editable-gigavoxel.md`
/// §3.3）。粒度 `g` 体素的一档，在距离 `d` 处的像素尺寸是 `g / (d·px_ang)` ⇒ 只有当
/// `d ≥ g/px_ang` 时才允许用它；否则画面里就是 `g/(d·px_ang)` 像素的方块（"稍远就什么都看不清"）。
/// 允许档里取**最粗**的那个（内存最优）。切换判据见 `gate-app::infinite_cubes::detail_at`。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Detail(pub u8);

impl Detail {
  /// 逐体素（2 cm）—— 最细一档。
  pub const Full: Detail = Detail(4);
  /// 4³ 量化（8 cm）。
  pub const Fine: Detail = Detail(3);
  /// 16³ 量化（32 cm）。
  pub const Coarse: Detail = Detail(2);
  /// 64³ 量化（1.28 m）。
  pub const Wide: Detail = Detail(1);
  /// 整 chunk 一格（5.12 m）—— 最粗一档。
  pub const Chunk: Detail = Detail(0);

  /// 本档的**格粒度**（格边长，体素）：1、4、16、64、256。
  pub fn grain(self) -> i32 {
    1 << (2 * (4 - self.0.min(4)))
  }

  /// 比本档更粗一档（最粗档返回自己）。
  pub fn coarser(self) -> Detail {
    Detail(self.0.saturating_sub(1))
  }
}

/// 生产源：按坐标产出一棵 chunk 树。
///
/// CONSTRAINT: 实现方在 **worker 线程**上被调用 ⇒ 必须 `Send + Sync`，且**不得碰主线程的
/// `VolumeGrid`**；`scratch` 是 worker 线程独占、可复用的暂存（实现方拿它当中转，避免每次分配）。
///
/// `vol` = 该 chunk 属于哪个 volume（0 = 主世界，≥1 = 远场级，与 `Volumes.list` 下标同一口径）——
/// 多级世界的各 volume 共用**同一个** producer（一个线程池），卷号随任务一起走。
pub trait ChunkSource: Send + Sync + 'static {
  /// 产出 `vol` 的 `coord` 处的 chunk 树（`None` = 这个 chunk 没有内容，调用方不必挂载）。
  fn produce(
    &self,
    vol: usize,
    coord: ChunkCoord,
    detail: Detail,
    scratch: &mut VolumeGrid,
  ) -> Option<ChunkTree>;
}

/// 任务 / 产出的键：**卷号 + 坐标**。同一坐标在不同 volume 里是不同的东西（远场级的 chunk 坐标
/// 是它自己的级体素空间），只按坐标去重会让两个卷互相顶掉。
type JobKey = (u8, ChunkCoord);

/// worker 回传的一条产出。
struct Done {
  key: JobKey,
  detail: Detail,
  tree: Option<ChunkTree>,
}

/// 后台生产者：`workers` 个线程，每条消息恰好派给一个 worker（每 worker 一条队列，轮转派发）。
///
/// 用法（主线程，每帧）：先 [`Self::request`] 派需求（多派几条，`max_inflight` 封顶），再
/// [`Self::poll`] 取回成品去挂载。**去重**由 `inflight` 负责：同一个 chunk 不会同时产两份。
pub struct ChunkProducer {
  queues: Vec<Sender<(JobKey, Detail)>>,
  next: usize,
  rx: Receiver<Done>,
  /// 已派发、还没取回的 chunk（去重 + 在飞上限）
  inflight: std::collections::HashSet<JobKey>,
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
      let (tx, job_rx) = std::sync::mpsc::channel::<(JobKey, Detail)>();
      let src = source.clone();
      let out = done_tx.clone();
      handles.push(
        std::thread::Builder::new()
          .name(format!("gate-chunk-{i}"))
          .spawn(move || {
            // worker 线程独占一份暂存：`VolumeGrid` 的调色板表是 512KB，建一次反复用
            let mut scratch = VolumeGrid::new();
            for (key, detail) in job_rx.iter() {
              let (vol, coord) = (key.0 as usize, key.1);
              let tree = src.produce(vol, coord, detail, &mut scratch);
              if out.send(Done { key, detail, tree }).is_err() {
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
  pub fn request(&mut self, vol: usize, coord: ChunkCoord, detail: Detail) -> bool {
    let key: JobKey = (vol as u8, coord);
    if self.inflight.len() >= self.max_inflight || !self.inflight.insert(key) {
      return false;
    }
    // 轮转派发；某条队列断了（那个 worker 退了）就换下一条 —— 只试一圈，全断才算失败。
    // （否则一个死掉的 worker 会让"表头那条"每帧都派不出去 ⇒ 表面上"加载停了"。）
    for _ in 0..self.queues.len() {
      let i = self.next % self.queues.len();
      self.next = self.next.wrapping_add(1);
      if self.queues[i].send((key, detail)).is_ok() {
        return true;
      }
    }
    self.inflight.remove(&key);
    false
  }

  /// 在飞的条数（需求派得出去、还没回来）
  pub fn inflight(&self) -> usize {
    self.inflight.len()
  }

  /// 这个 volume 的这个坐标是否已在飞（消费端用来跳过"已派发但还没回来"的那些，免得重复入队或空转）
  pub fn in_flight(&self, vol: usize, coord: ChunkCoord) -> bool {
    self.inflight.contains(&(vol as u8, coord))
  }

  /// 取回已完成的产出（非阻塞，本次最多 `max` 条）。返回的 `tree = None` 表示"这个 chunk 没有内容"，
  /// 调用方应把它记成"已生成、无内容"（免得每帧重复派发）。
  pub fn poll(&mut self, max: usize) -> Vec<(usize, ChunkCoord, Detail, Option<ChunkTree>)> {
    let mut out = Vec::new();
    while out.len() < max {
      match self.rx.try_recv() {
        Ok(d) => {
          self.inflight.remove(&d.key);
          out.push((d.key.0 as usize, d.key.1, d.detail, d.tree));
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
