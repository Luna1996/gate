use std::collections::HashMap;

use gate_voxel::{CHUNK_SIZE, VolumeGrid};
use glam::IVec3;

use crate::classify::ContactVoxels;
use crate::field::{Field, NodeFill};

pub struct StaticChunk {
  pub vox: ContactVoxels,
  pub local: (IVec3, IVec3),
  pub stamp: u64,
}

#[derive(Default)]
pub struct StaticField {
  chunks: HashMap<IVec3, StaticChunk>,
  empty: HashMap<IVec3, (u64, bool)>,
}

impl StaticField {
  pub fn len(&self) -> usize {
    self.chunks.len()
  }

  pub fn is_empty(&self) -> bool {
    self.chunks.is_empty()
  }

  pub fn clear(&mut self) {
    self.chunks.clear();
    self.empty.clear();
  }

  pub fn get(&self, chunk: IVec3) -> Option<&StaticChunk> {
    self.chunks.get(&chunk)
  }

  pub fn corners(&self) -> usize {
    self.chunks.values().map(|c| c.vox.corners().len()).sum()
  }

  pub fn sync(&mut self, grid: &VolumeGrid, lo: IVec3, hi: IVec3) -> usize {
    let stamp = grid.edit_generation();
    let keep_lo = chunk_of(lo) - IVec3::ONE;
    let keep_hi = chunk_of(hi) + IVec3::ONE;
    let inside = |c: &IVec3| c.cmpge(keep_lo).all() && c.cmple(keep_hi).all();
    self.chunks.retain(|c, _| inside(c));
    self.empty.retain(|c, _| inside(c));

    let build_lo = chunk_of(lo);
    let build_hi = chunk_of(hi);
    let mut built = 0;
    for cz in build_lo.z..=build_hi.z {
      for cy in build_lo.y..=build_hi.y {
        for cx in build_lo.x..=build_hi.x {
          let cc = IVec3::new(cx, cy, cz);
          let resident = grid.chunk(gate_voxel::ChunkCoord(cc)).is_some();
          if self.chunks.get(&cc).is_some_and(|c| c.stamp == stamp && resident)
            || self.empty.get(&cc) == Some(&(stamp, resident))
          {
            continue;
          }
          match chunk_accel(grid, cc, stamp) {
            Some(c) => {
              self.chunks.insert(cc, c);
              self.empty.remove(&cc);
              built += 1;
            }
            None => {
              self.chunks.remove(&cc);
              self.empty.insert(cc, (stamp, resident));
            }
          }
        }
      }
    }
    built
  }
}

pub fn chunk_of(p: IVec3) -> IVec3 {
  let shift = CHUNK_SIZE.trailing_zeros();
  debug_assert_eq!(1 << shift, CHUNK_SIZE, "静态分块按 2 的幂切分");
  IVec3::new(p.x >> shift, p.y >> shift, p.z >> shift)
}

fn chunk_accel(grid: &VolumeGrid, cc: IVec3, stamp: u64) -> Option<StaticChunk> {
  let base = cc * CHUNK_SIZE;
  let top = base + IVec3::splat(CHUNK_SIZE - 1);
  let f = Field::new(grid);
  let mut lo = IVec3::splat(i32::MAX);
  let mut hi = IVec3::splat(i32::MIN);
  let mut any = false;
  f.for_each_node(base, top, 4, &mut |origin, fill| {
    if matches!(fill, NodeFill::Air) {
      return;
    }
    any = true;
    lo = lo.min(origin);
    hi = hi.max(origin + IVec3::splat(3));
  });
  if !any {
    return None;
  }
  let vox = ContactVoxels::build(&f, (lo, hi));
  Some(StaticChunk { vox, local: (lo, hi), stamp })
}
