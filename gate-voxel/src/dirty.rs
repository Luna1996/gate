//! 脏标记与每帧上传预算队列。
//! 数据脏（`data_dirty`，触发 Brick Tree 重建上传）与元件脏（`comp_dirty`，只传 StateTable）分离。
//! 队列 FIFO + 去重，按帧预算 drain。

use std::collections::{HashSet, VecDeque};

use crate::coords::ChunkCoord;

#[derive(Debug, Default, Clone)]
pub struct DirtyTracker {
  data_dirty: HashSet<ChunkCoord>,
  comp_dirty: HashSet<ChunkCoord>,
  data_queue: VecDeque<ChunkCoord>,
  comp_queue: VecDeque<ChunkCoord>,
}

impl DirtyTracker {
  pub fn new() -> Self {
    Self::default()
  }

  pub fn mark_data(&mut self, coord: ChunkCoord) {
    if self.data_dirty.insert(coord) {
      self.data_queue.push_back(coord);
    }
  }

  pub fn mark_comp(&mut self, coord: ChunkCoord) {
    if self.comp_dirty.insert(coord) {
      self.comp_queue.push_back(coord);
    }
  }

  pub fn is_data_dirty(&self, coord: &ChunkCoord) -> bool {
    self.data_dirty.contains(coord)
  }

  pub fn data_dirty_count(&self) -> usize {
    self.data_dirty.len()
  }

  pub fn comp_dirty_count(&self) -> usize {
    self.comp_dirty.len()
  }

  pub fn drain_data_budget(&mut self, budget: usize) -> Vec<ChunkCoord> {
    let n = budget.min(self.data_queue.len());
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
      if let Some(c) = self.data_queue.pop_front() {
        self.data_dirty.remove(&c);
        out.push(c);
      }
    }
    out
  }

  pub fn drain_comp_budget(&mut self, budget: usize) -> Vec<ChunkCoord> {
    let n = budget.min(self.comp_queue.len());
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
      if let Some(c) = self.comp_queue.pop_front() {
        self.comp_dirty.remove(&c);
        out.push(c);
      }
    }
    out
  }

  pub fn heap_bytes(&self) -> usize {
    let entry = std::mem::size_of::<ChunkCoord>() + 1;
    let set = |cap: usize| cap * entry / 7 * 8;
    let q = |cap: usize| cap * std::mem::size_of::<ChunkCoord>();
    set(self.data_dirty.capacity())
      + set(self.comp_dirty.capacity())
      + q(self.data_queue.capacity())
      + q(self.comp_queue.capacity())
  }
}
