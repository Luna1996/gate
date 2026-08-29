//! 脏标记与每帧上传预算队列（P1.5）
//!
//! 数据脏 / 元件脏分离：体素编辑走 `data_dirty`（触发砖块图重建上传），
//! 元件状态变化走 `comp_dirty`（只传 StateTable，不重铺体素数据）。
//! 队列 FIFO + 去重，消费侧按帧预算 drain（P2.3 上传通道用）。

use std::collections::{HashSet, VecDeque};

use crate::coords::TileCoord;

#[derive(Debug, Default)]
pub struct DirtyTracker {
  data_dirty: HashSet<TileCoord>,
  comp_dirty: HashSet<TileCoord>,
  /// FIFO 待处理队列（与 set 联动去重）
  data_queue: VecDeque<TileCoord>,
  comp_queue: VecDeque<TileCoord>,
}

impl DirtyTracker {
  pub fn new() -> Self {
    Self::default()
  }

  pub fn mark_data(&mut self, coord: TileCoord) {
    if self.data_dirty.insert(coord) {
      self.data_queue.push_back(coord);
    }
  }

  pub fn mark_comp(&mut self, coord: TileCoord) {
    if self.comp_dirty.insert(coord) {
      self.comp_queue.push_back(coord);
    }
  }

  pub fn is_data_dirty(&self, coord: &TileCoord) -> bool {
    self.data_dirty.contains(coord)
  }

  /// 当前脏集合大小（调试/统计用）
  pub fn data_dirty_count(&self) -> usize {
    self.data_dirty.len()
  }

  pub fn comp_dirty_count(&self) -> usize {
    self.comp_dirty.len()
  }

  /// 按预算取出一批数据脏 Tile（每帧上传上限，防止帧尖峰）
  pub fn drain_data_budget(&mut self, budget: usize) -> Vec<TileCoord> {
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

  /// 元件脏同款预算 drain
  pub fn drain_comp_budget(&mut self, budget: usize) -> Vec<TileCoord> {
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
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn dedup_and_fifo() {
    let mut t = DirtyTracker::new();
    let a = TileCoord::new(0, 0, 0);
    let b = TileCoord::new(1, 0, 0);
    t.mark_data(a);
    t.mark_data(a); // 去重
    t.mark_data(b);
    assert_eq!(t.data_dirty_count(), 2);
    assert_eq!(t.drain_data_budget(1), vec![a]); // FIFO
    assert_eq!(t.data_dirty_count(), 1);
    assert_eq!(t.drain_data_budget(10), vec![b]); // 预算外不吐
    assert_eq!(t.drain_data_budget(10), vec![]);
  }

  #[test]
  fn data_and_comp_independent() {
    let mut t = DirtyTracker::new();
    let a = TileCoord::new(2, 3, 4);
    t.mark_data(a);
    assert!(t.is_data_dirty(&a));
    t.drain_data_budget(10);
    assert!(!t.is_data_dirty(&a));
    t.mark_comp(a);
    assert_eq!(t.comp_dirty_count(), 1);
    assert_eq!(t.data_dirty_count(), 0); // 通道互不影响
  }
}
