use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender};

use crate::chunk_tree::ChunkTree;
use crate::coords::ChunkCoord;
use crate::volume::VolumeGrid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Detail(pub u8);

impl Detail {
  pub const FULL: Detail = Detail(4);
  pub const FINE: Detail = Detail(3);
  pub const COARSE: Detail = Detail(2);
  pub const WIDE: Detail = Detail(1);
  pub const CHUNK: Detail = Detail(0);

  pub fn grain(self) -> i32 {
    1 << (2 * (4 - self.0.min(4)))
  }

  pub fn coarser(self) -> Detail {
    Detail(self.0.saturating_sub(1))
  }
}

pub trait ChunkSource: Send + Sync + 'static {
  fn produce(
    &self,
    vol: usize,
    coord: ChunkCoord,
    detail: Detail,
    scratch: &mut VolumeGrid,
  ) -> Option<ChunkTree>;

  fn content_y_range(&self) -> Option<(i32, i32)> {
    None
  }

  fn palette_log(
    &self,
    from: usize,
  ) -> Vec<(crate::palette::PaletteId, crate::palette::PaletteEntry)> {
    let _ = from;
    Vec::new()
  }

  fn clone_as_any(self: Arc<Self>) -> Option<Arc<dyn std::any::Any + Send + Sync>> {
    None
  }
}

type JobKey = (u8, ChunkCoord);

struct Done {
  key: JobKey,
  detail: Detail,
  tree: Option<ChunkTree>,
  words: usize,
}

pub struct ChunkProducer {
  queues: Vec<Sender<(JobKey, Detail)>>,
  next: usize,
  rx: Receiver<Done>,
  inflight: std::collections::HashSet<JobKey>,
  max_inflight: usize,
  workers: Vec<std::thread::JoinHandle<()>>,
}

impl ChunkProducer {
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
            let mut scratch = VolumeGrid::new();
            for (key, detail) in job_rx.iter() {
              let (vol, coord) = (key.0 as usize, key.1);
              let tree = src.produce(vol, coord, detail, &mut scratch);
              let words = tree.as_ref().map_or(0, |t| t.len_words());
              if out.send(Done { key, detail, tree, words }).is_err() {
                return;
              }
            }
          })
          .expect("spawn chunk worker"),
      );
      queues.push(tx);
    }
    drop(done_tx);
    Self {
      queues,
      next: 0,
      rx,
      inflight: Default::default(),
      max_inflight: workers * 32,
      workers: handles,
    }
  }

  pub fn request(&mut self, vol: usize, coord: ChunkCoord, detail: Detail) -> bool {
    let key: JobKey = (vol as u8, coord);
    if self.inflight.len() >= self.max_inflight || !self.inflight.insert(key) {
      return false;
    }
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

  pub fn inflight(&self) -> usize {
    self.inflight.len()
  }

  pub fn in_flight(&self, vol: usize, coord: ChunkCoord) -> bool {
    self.inflight.contains(&(vol as u8, coord))
  }

  pub fn poll(&mut self, max: usize) -> Vec<(usize, ChunkCoord, Detail, Option<ChunkTree>, usize)> {
    let mut out = Vec::new();
    while out.len() < max {
      match self.rx.try_recv() {
        Ok(d) => {
          self.inflight.remove(&d.key);
          out.push((d.key.0 as usize, d.key.1, d.detail, d.tree, d.words));
        }
        Err(std::sync::mpsc::TryRecvError::Empty) => break,
        Err(std::sync::mpsc::TryRecvError::Disconnected) => break,
      }
    }
    out
  }
}

impl Drop for ChunkProducer {
  fn drop(&mut self) {
    self.queues.clear();
    for h in self.workers.drain(..) {
      let _ = h.join();
    }
  }
}
