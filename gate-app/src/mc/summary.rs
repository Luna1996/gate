use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use gate_voxel::PaletteId;
use glam::IVec3;

use super::voxel;
use super::world::SECTION_VOLUME;

pub const SEC: i32 = 16;
const FINE_PER_AXIS: i32 = 4;
const FINE: usize = (FINE_PER_AXIS * FINE_PER_AXIS * FINE_PER_AXIS) as usize;

pub(crate) const SOLID_MIN_PERMILLE: u32 = 16;

const SEC_CACHE_MAX: usize = 32_768;

pub struct Sec {
  pub fine: Box<[PaletteId; FINE]>,
  pub rep: PaletteId,
  pub solid: u16,
}

pub trait SectionReps: Send + Sync {
  fn section_reps(&self, sec: IVec3) -> Option<Box<[PaletteId; SECTION_VOLUME]>>;
}

pub struct Summary {
  secs: Mutex<Cache>,
}

#[derive(Default)]
struct Cache {
  map: HashMap<IVec3, Option<Arc<Sec>>>,
  order: VecDeque<IVec3>,
}

impl Default for Summary {
  fn default() -> Self {
    Self::new()
  }
}

impl Summary {
  pub fn new() -> Self {
    Self { secs: Mutex::new(Cache::default()) }
  }

  fn sec(&self, src: &dyn SectionReps, sec: IVec3) -> Option<Arc<Sec>> {
    {
      let c = self.secs.lock().unwrap_or_else(|e| e.into_inner());
      if let Some(hit) = c.map.get(&sec) {
        return hit.clone();
      }
    }
    let built = src.section_reps(sec).map(|reps| Arc::new(build(reps.as_ref())));
    let mut c = self.secs.lock().unwrap_or_else(|e| e.into_inner());
    c.map.insert(sec, built.clone());
    c.order.push_back(sec);
    while c.order.len() > SEC_CACHE_MAX {
      if let Some(old) = c.order.pop_front() {
        c.map.remove(&old);
      }
    }
    built
  }

  pub fn cell(&self, src: &dyn SectionReps, cell: IVec3, cell_blocks: i32) -> Option<PaletteId> {
    match cell_blocks {
      4 => {
        let sec = self.sec(src, cell.div_euclid(IVec3::splat(SEC)))?;
        let l = cell.rem_euclid(IVec3::splat(SEC)) / 4;
        let id =
          sec.fine[(l.x + FINE_PER_AXIS * l.z + FINE_PER_AXIS * FINE_PER_AXIS * l.y) as usize];
        (!id.is_air()).then_some(id)
      }
      16 => {
        let sec = self.sec(src, cell.div_euclid(IVec3::splat(SEC)))?;
        let occupied = sec.solid as u32 * 1000 >= SOLID_MIN_PERMILLE * SECTION_VOLUME as u32
          || sec.fine.iter().any(|id| !id.is_air());
        occupied.then_some(sec.rep)
      }
      _ => {
        let c = cell.div_euclid(IVec3::splat(SEC));
        let col_x = (cell.x + cell_blocks / 2).div_euclid(SEC);
        let col_z = (cell.z + cell_blocks / 2).div_euclid(SEC);
        let n = cell_blocks / SEC;
        for sy in (0..n).rev() {
          if let Some(sec) = self.sec(src, IVec3::new(col_x, c.y + sy, col_z)) {
            if !sec.rep.is_air() {
              return Some(sec.rep);
            }
          }
        }
        None
      }
    }
  }
}

fn build(reps: &[PaletteId]) -> Sec {
  let mut fine = Box::new([PaletteId::AIR; FINE]);
  let mut solid = 0u16;
  let mut groups = [PaletteId::AIR; 64];
  for gy in 0..FINE_PER_AXIS {
    for gz in 0..FINE_PER_AXIS {
      for gx in 0..FINE_PER_AXIS {
        let mut buf = [PaletteId::AIR; 64];
        let mut n = 0usize;
        for y in 0..4 {
          for z in 0..4 {
            for x in 0..4 {
              let (bx, by, bz) = (gx * 4 + x, gy * 4 + y, gz * 4 + z);
              let id = reps[(by * 256 + bz * 16 + bx) as usize];
              buf[n] = id;
              n += 1;
              if !id.is_air() {
                solid = solid.saturating_add(1);
              }
            }
          }
        }
        let cell_id = group(&buf, 64, 16);
        groups[(gx + FINE_PER_AXIS * gz + FINE_PER_AXIS * FINE_PER_AXIS * gy) as usize] = cell_id;
        fine[(gx + FINE_PER_AXIS * gz + FINE_PER_AXIS * FINE_PER_AXIS * gy) as usize] = cell_id;
      }
    }
  }
  Sec { fine, rep: voxel::rep_of_surface(&groups, 16).unwrap_or(PaletteId::AIR), solid }
}

fn group(buf: &[PaletteId], n: usize, per_layer: usize) -> PaletteId {
  let solid = buf[..n].iter().filter(|c| !c.is_air()).count();
  if (solid as u32) * 1000 < SOLID_MIN_PERMILLE * n as u32 {
    return PaletteId::AIR;
  }
  voxel::rep_of_surface(&buf[..n], per_layer).unwrap_or(PaletteId::AIR)
}