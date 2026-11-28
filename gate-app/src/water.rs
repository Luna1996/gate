use std::collections::HashSet;

use bevy::prelude::*;
use gate_render::VoxelScene;
use gate_voxel::{PaletteFlags, PaletteId, VolumeGrid, VoxelCoord};
use glam::IVec3;

pub const WATER_ID: PaletteId = PaletteId(3);
pub const SOURCE_ID: PaletteId = PaletteId(4);

const TICK_DT: f32 = 1.0 / 60.0;
const BUDGET: usize = 8192;
const COARSE: i32 = 4;
const RADIUS: i32 = 1024;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
  Open,
  Water,
  Solid,
}

#[derive(Resource)]
pub struct WaterSim {
  water: PaletteId,
  map_min: IVec3,
  map_max: IVec3,
  cam: IVec3,
  radius: i32,
  active: HashSet<IVec3>,
  wake: HashSet<IVec3>,
  last_edit: u64,
  acc: f32,
  enabled: bool,
}

impl Default for WaterSim {
  fn default() -> Self {
    Self {
      water: PaletteId::AIR,
      map_min: IVec3::ZERO,
      map_max: IVec3::ZERO,
      cam: IVec3::ZERO,
      radius: RADIUS,
      active: HashSet::new(),
      wake: HashSet::new(),
      last_edit: 0,
      acc: 0.0,
      enabled: false,
    }
  }
}

impl WaterSim {
  pub fn new(
    seed: &[IVec3],
    grid: &VolumeGrid,
    bounds: Option<(IVec3, IVec3)>,
    water: Option<PaletteId>,
  ) -> Self {
    let mut s = Self::default();
    s.reset(seed, grid, bounds, water);
    s
  }

  pub fn reset(
    &mut self,
    seed: &[IVec3],
    grid: &VolumeGrid,
    bounds: Option<(IVec3, IVec3)>,
    water: Option<PaletteId>,
  ) {
    self.water = water.unwrap_or(PaletteId::AIR);
    self.enabled = water.is_some();
    self.active.clear();
    self.wake.clear();
    self.acc = 0.0;
    let Some((mn, mx)) = bounds else {
      self.last_edit = grid.edit_log_seq();
      return;
    };
    self.map_min = mn;
    self.map_max = mx;
    self.cam = (mn + mx) / 2;
    if self.enabled {
      bevy::log::info!(
        "WATER 模拟：palette={:?} 体积=[{}]-[{}]",
        self.water,
        self.map_min,
        self.map_max
      );
    }
    for &p in seed {
      if !self.in_bounds(p) {
        continue;
      }
      if self.kind(grid, p) == Kind::Water && self.exposed_non_top(grid, p) {
        self.active.insert(p);
      }
    }
    self.last_edit = grid.edit_log_seq();
  }

  fn in_bounds(&self, p: IVec3) -> bool {
    p.cmpge(self.map_min).all()
      && p.cmplt(self.map_max).all()
      && (p - self.cam).abs().cmple(IVec3::splat(self.radius)).all()
  }

  fn kind(&self, grid: &VolumeGrid, p: IVec3) -> Kind {
    let Some(id) = grid.get_voxel(VoxelCoord::from_ivec3(p)) else { return Kind::Open };
    if id.is_air() {
      return Kind::Open;
    }
    let flags = grid.palette().get(id).flags;
    if flags.contains(PaletteFlags::FLUID) || flags.contains(PaletteFlags::FLUID_SOURCE) {
      Kind::Water
    } else {
      Kind::Solid
    }
  }

  fn exposed_non_top(&self, grid: &VolumeGrid, p: IVec3) -> bool {
    for d in [IVec3::NEG_Y, IVec3::X, IVec3::NEG_X, IVec3::Z, IVec3::NEG_Z] {
      if self.kind(grid, p + d) == Kind::Open {
        return true;
      }
    }
    false
  }

  fn fill(&self, grid: &mut VolumeGrid, p: IVec3) {
    grid.fill_brick_silent(p, 1, self.water);
  }

  fn coarse_block(&self, grid: &VolumeGrid, n: IVec3, level_top: i32) -> Option<IVec3> {
    let origin = n.div_euclid(IVec3::splat(COARSE)) * COARSE;
    if origin.y + COARSE > level_top {
      return None;
    }
    if !self.in_bounds(origin) || !self.in_bounds(origin + IVec3::splat(COARSE - 1)) {
      return None;
    }
    for y in 0..COARSE {
      for z in 0..COARSE {
        for x in 0..COARSE {
          if self.kind(grid, origin + IVec3::new(x, y, z)) != Kind::Open {
            return None;
          }
        }
      }
    }
    Some(origin)
  }

  fn wake_block(&self, grid: &VolumeGrid, origin: IVec3, next: &mut HashSet<IVec3>) {
    for y in 0..COARSE {
      for z in 0..COARSE {
        for x in 0..COARSE {
          let edge =
            x == 0 || y == 0 || z == 0 || x == COARSE - 1 || y == COARSE - 1 || z == COARSE - 1;
          if !edge {
            continue;
          }
          let p = origin + IVec3::new(x, y, z);
          if self.kind(grid, p) == Kind::Water && self.exposed_non_top(grid, p) {
            next.insert(p);
          }
        }
      }
    }
  }

  fn seed_edit(&self, grid: &VolumeGrid, lo: IVec3, hi: IVec3, wake: &mut HashSet<IVec3>) {
    let a = lo - IVec3::ONE;
    let b = hi + IVec3::ONE;
    for y in a.y..b.y {
      for z in a.z..b.z {
        for x in a.x..b.x {
          let p = IVec3::new(x, y, z);
          if !self.in_bounds(p) {
            continue;
          }
          match self.kind(grid, p) {
            Kind::Water => {
              if self.exposed_non_top(grid, p) {
                wake.insert(p);
              }
            }
            Kind::Open => {
              for d in [IVec3::Y, IVec3::NEG_Y, IVec3::X, IVec3::NEG_X, IVec3::Z, IVec3::NEG_Z] {
                let q = p + d;
                if self.kind(grid, q) == Kind::Water && self.exposed_non_top(grid, q) {
                  wake.insert(q);
                }
              }
            }
            Kind::Solid => {}
          }
        }
      }
    }
  }

  fn step(&mut self, grid: &mut VolumeGrid) {
    let mut queue: Vec<IVec3> = self.active.drain().collect();
    queue.extend(self.wake.drain());
    let mut seen = HashSet::new();
    let mut next = HashSet::new();
    let mut processed = 0usize;

    for p in queue {
      if !seen.insert(p) {
        continue;
      }
      if processed >= BUDGET {
        next.insert(p);
        continue;
      }
      processed += 1;
      if !self.in_bounds(p) || self.kind(grid, p) != Kind::Water {
        continue;
      }
      if !self.exposed_non_top(grid, p) {
        continue;
      }
      let level_top = p.y + 1;
      for d in [IVec3::NEG_Y, IVec3::X, IVec3::NEG_X, IVec3::Z, IVec3::NEG_Z] {
        let n = p + d;
        if !self.in_bounds(n) {
          continue;
        }
        match self.kind(grid, n) {
          Kind::Open => {
            if let Some(origin) = self.coarse_block(grid, n, level_top) {
              grid.fill_brick_silent(origin, COARSE, self.water);
              self.wake_block(grid, origin, &mut next);
            } else {
              self.fill(grid, n);
              next.insert(n);
              for dd in [IVec3::NEG_Y, IVec3::X, IVec3::NEG_X, IVec3::Z, IVec3::NEG_Z] {
                let q = n + dd;
                if self.kind(grid, q) == Kind::Water && self.exposed_non_top(grid, q) {
                  next.insert(q);
                }
              }
            }
          }
          Kind::Water => {
            if self.exposed_non_top(grid, n) {
              next.insert(n);
            }
          }
          Kind::Solid => {}
        }
      }
      if self.exposed_non_top(grid, p) {
        next.insert(p);
      }
    }
    self.active = next;
  }
}

pub fn tick_water(
  time: Res<Time>,
  mut scene: Option<ResMut<VoxelScene>>,
  mut sim: ResMut<WaterSim>,
  cam: Option<Res<gate_render::DdaCameraConfig>>,
) {
  if !sim.enabled {
    return;
  }
  sim.acc += time.delta_secs().min(0.1);
  if sim.acc < TICK_DT {
    return;
  }
  sim.acc -= TICK_DT;
  let Some(scene) = scene.as_mut() else { return };
  let grid = scene.volumes.main_mut();
  if let Some(cam) = cam.as_ref() {
    sim.cam = cam.position_world.floor().as_ivec3();
  }

  let seq = grid.edit_log_seq();
  if seq > sim.last_edit {
    let edits: Vec<(IVec3, IVec3)> = grid.edits_since(sim.last_edit).collect();
    let mut wake = HashSet::new();
    for &(lo, hi) in &edits {
      sim.seed_edit(grid, lo, hi, &mut wake);
    }
    if !wake.is_empty() {
      bevy::log::debug!("WATER wake {} 编辑", wake.len());
    }
    sim.wake.extend(wake);
    sim.last_edit = seq;
  }

  sim.step(grid);
}

pub fn advance_water_time(time: Res<Time>, mut settings: ResMut<gate_render::WaterSettings>) {
  settings.time += time.delta_secs();
}
