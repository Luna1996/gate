use std::collections::{HashMap, HashSet};

use gate_voxel::{BRICK_FACTOR, CHUNK_SIZE, ChunkCoord};
use glam::IVec3;

pub type Level = i32;

pub const LADDER: &[(Level, f32)] =
  &[(CHUNK_SIZE, 64.0), (64, 16.0), (16, 4.0), (BRICK_FACTOR, 0.0)];

const HYSTERESIS: f32 = 0.75;

pub const BOOTSTRAP_LEVEL: Level = 16;

const BOOTSTRAP_PER_FRAME: usize = 128;

pub fn want_level(dist_voxels: f32, px_ang: f32, cur_level: Level) -> Level {
  let fp = dist_voxels * px_ang;
  let raw = raw_level(fp);
  if raw >= cur_level {
    return raw;
  }
  if fp < threshold_of(cur_level) * HYSTERESIS { raw } else { cur_level }
}

pub(crate) fn raw_level(fp: f32) -> Level {
  for &(level, thr) in LADDER {
    if fp >= thr {
      return level;
    }
  }
  BRICK_FACTOR
}

fn threshold_of(level: Level) -> f32 {
  LADDER.iter().find(|&&(l, _)| l == level).map_or(0.0, |&(_, t)| t)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Entry {
  bytes: usize,
  level: Level,
  pinned_until: u64,
  since: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct ResidencyPolicy {
  pub budget_bytes: usize,
  pub min_resident_frames: u64,
  pub pin_frames: u64,
  pub max_install_per_frame: usize,
}

impl ResidencyPolicy {
  pub const DEFAULT: Self =
    Self { budget_bytes: 0, min_resident_frames: 30, pin_frames: 120, max_install_per_frame: 8 };

  pub fn budget_is_off(&self) -> bool {
    self.budget_bytes == 0
  }
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ResidencyPlan {
  pub install: Vec<(ChunkCoord, Level)>,
  pub evict: Vec<ChunkCoord>,
}

#[derive(Debug, Default, Clone)]
pub struct Residency {
  entries: HashMap<ChunkCoord, Entry>,
  bytes_total: usize,
  frame: u64,
}

impl Residency {
  pub fn new() -> Self {
    Self::default()
  }

  pub fn frame(&self) -> u64 {
    self.frame
  }

  pub fn tick(&mut self, frame: u64) {
    self.frame = frame;
  }

  pub fn resident_count(&self) -> usize {
    self.entries.len()
  }

  pub fn resident_bytes(&self) -> usize {
    self.bytes_total
  }

  pub fn is_resident(&self, c: ChunkCoord) -> bool {
    self.entries.contains_key(&c)
  }

  pub fn resident_level(&self, c: ChunkCoord) -> Option<Level> {
    self.entries.get(&c).map(|e| e.level)
  }

  pub fn note_resident(&mut self, c: ChunkCoord, bytes: usize, level: Level, frame: u64) {
    match self.entries.get_mut(&c) {
      Some(e) => {
        self.bytes_total = self.bytes_total.saturating_sub(e.bytes) + bytes;
        e.bytes = bytes;
        e.level = level;
      }
      None => {
        self.bytes_total += bytes;
        self.entries.insert(c, Entry { bytes, level, pinned_until: 0, since: frame });
      }
    }
  }

  pub fn note_gone(&mut self, c: ChunkCoord) {
    if let Some(e) = self.entries.remove(&c) {
      self.bytes_total = self.bytes_total.saturating_sub(e.bytes);
    }
  }

  pub fn note_bytes(&mut self, c: ChunkCoord, bytes: usize) {
    if let Some(e) = self.entries.get_mut(&c) {
      self.bytes_total = self.bytes_total.saturating_sub(e.bytes) + bytes;
      e.bytes = bytes;
    }
  }

  pub fn note_edit(&mut self, c: ChunkCoord, frame: u64, pin_frames: u64) {
    if let Some(e) = self.entries.get_mut(&c) {
      e.pinned_until = e.pinned_until.max(frame + pin_frames);
    }
  }

  pub fn nearest_top(&self, camera_chunk: IVec3, n: usize) -> Vec<ChunkCoord> {
    let mut v: Vec<ChunkCoord> = self.entries.keys().copied().collect();
    let key = |c: &ChunkCoord| (chunk_distance(camera_chunk, c.0), c.0.x, c.0.y, c.0.z);
    if n >= v.len() {
      return v;
    }
    v.select_nth_unstable_by(n, |a, b| key(a).cmp(&key(b)));
    v.truncate(n);
    v.sort_unstable_by_key(key);
    v
  }

  pub fn plan(
    &self,
    policy: &ResidencyPolicy,
    camera_chunk: IVec3,
    wants: impl Iterator<Item = (ChunkCoord, Level)>,
    must_keep: &HashSet<ChunkCoord>,
  ) -> ResidencyPlan {
    let mut install: Vec<(ChunkCoord, Level)> = wants
      .filter(|&(c, level)| {
        let cur = self.resident_level(c);
        if cur == Some(level) {
          return false;
        }
        let downgrade = cur.is_some_and(|l| l < level);
        let pinned = self.entries.get(&c).is_some_and(|e| e.pinned_until >= self.frame);
        !(downgrade && pinned)
      })
      .collect();
    install.sort_by_key(|&(c, level)| {
      (dist_of(camera_chunk, c), std::cmp::Reverse(level), c.0.x, c.0.y, c.0.z)
    });
    //
    let mut picked: Vec<(ChunkCoord, Level)> = Vec::with_capacity(install.len());
    let mut refine_budget = policy.max_install_per_frame;
    let mut boot_budget = BOOTSTRAP_PER_FRAME;
    for (c, level) in install {
      if self.resident_level(c).is_none() {
        if boot_budget == 0 {
          continue;
        }
        boot_budget -= 1;
        picked.push((c, level.max(BOOTSTRAP_LEVEL)));
      } else if refine_budget > 0 {
        refine_budget -= 1;
        picked.push((c, level));
      }
    }
    let install = picked;

    let evict = self.pick_evicts(policy, camera_chunk, must_keep);
    ResidencyPlan { install, evict }
  }

  fn pick_evicts(
    &self,
    policy: &ResidencyPolicy,
    camera_chunk: IVec3,
    must_keep: &HashSet<ChunkCoord>,
  ) -> Vec<ChunkCoord> {
    if policy.budget_is_off() {
      return Vec::new();
    }
    let mut over = self.resident_bytes().saturating_sub(policy.budget_bytes);
    if over == 0 {
      return Vec::new();
    }
    let mut order: Vec<(&ChunkCoord, &Entry)> = self.entries.iter().collect();
    order.sort_by_key(|(c, _)| {
      (std::cmp::Reverse(chunk_distance(camera_chunk, c.0)), c.0.x, c.0.y, c.0.z)
    });
    let mut out = Vec::new();
    for (c, e) in order {
      if over == 0 {
        break;
      }
      if must_keep.contains(c) || e.pinned_until >= self.frame {
        continue;
      }
      if self.frame.saturating_sub(e.since) < policy.min_resident_frames {
        continue;
      }
      over = over.saturating_sub(e.bytes);
      out.push(*c);
    }
    out
  }
}

pub fn chunk_distance(a: IVec3, b: IVec3) -> i32 {
  let d = (a - b).abs();
  d.x.max(d.y).max(d.z)
}

fn dist_of(a: IVec3, b: ChunkCoord) -> i32 {
  chunk_distance(a, b.0)
}