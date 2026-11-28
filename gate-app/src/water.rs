use std::collections::{HashMap, HashSet};

use bevy::prelude::*;
use gate_render::VoxelScene;
use gate_voxel::{PaletteFlags, PaletteId, VolumeGrid, VoxelCoord};
use glam::IVec3;

pub const WATER_ID: PaletteId = PaletteId(3);
pub const SOURCE_ID: PaletteId = PaletteId(4);

const TICK_DT: f32 = 1.0 / 60.0;
const BUDGET: usize = 4096;
const MASS_EPS: f32 = 0.005;
const RENDER_MIN: f32 = 0.02;
const FLOW_RATE: f32 = 30.0;
const FLOW_MAX_FRAC: f32 = 0.5;
const FLOW_EPS: f32 = 1.0e-4;
const WAKE_EPS: f32 = 1.0e-3;
const SOURCE_RATE: f32 = 90.0;
const MC_RADIUS: i32 = 64;
const TEST_RADIUS: i32 = 256;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
  Open,
  Water,
  Solid,
}

#[derive(Resource)]
pub struct WaterSim {
  cell: i32,
  water: PaletteId,
  radius: i32,
  mass: HashMap<IVec3, f32>,
  active: Vec<IVec3>,
  wake: Vec<IVec3>,
  sources: Vec<IVec3>,
  map_min: IVec3,
  map_max: IVec3,
  cam_cell: IVec3,
  last_edit: u64,
  acc: f32,
  enabled: bool,
}

impl Default for WaterSim {
  fn default() -> Self {
    Self {
      cell: 1,
      water: PaletteId::AIR,
      radius: TEST_RADIUS,
      mass: HashMap::new(),
      active: Vec::new(),
      wake: Vec::new(),
      sources: Vec::new(),
      map_min: IVec3::ZERO,
      map_max: IVec3::ZERO,
      cam_cell: IVec3::ZERO,
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
    cell: i32,
  ) -> Self {
    let mut s = Self::default();
    s.reset(seed, grid, bounds, water, cell);
    s
  }

  pub fn reset(
    &mut self,
    seed: &[IVec3],
    grid: &VolumeGrid,
    bounds: Option<(IVec3, IVec3)>,
    water: Option<PaletteId>,
    cell: i32,
  ) {
    self.cell = cell.max(1);
    self.water = water.unwrap_or(PaletteId::AIR);
    self.radius = if self.cell > 1 { MC_RADIUS } else { TEST_RADIUS };
    self.mass.clear();
    self.active.clear();
    self.wake.clear();
    self.sources.clear();
    self.acc = 0.0;
    self.enabled = water.is_some() && (self.cell > 1 || !seed.is_empty());
    let Some((mn, mx)) = bounds else {
      self.last_edit = grid.edit_log_seq();
      return;
    };
    let c = IVec3::splat(self.cell);
    self.map_min = mn.div_euclid(c);
    self.map_max = mx.div_euclid(c);
    self.cam_cell = (self.map_min + self.map_max) / 2;
    if self.enabled {
      bevy::log::info!(
        "WATER 模拟：cell={} palette={:?} cells=[{}]-[{}]",
        self.cell,
        self.water,
        self.map_min,
        self.map_max
      );
    }
    if seed.is_empty() {
      self.last_edit = grid.edit_log_seq();
      return;
    }
    let palette = grid.palette();
    for &p in seed {
      let cell = p.div_euclid(c);
      if !self.in_bounds(cell) {
        continue;
      }
      let Some(id) = grid.get_voxel(VoxelCoord::from_ivec3(p)) else { continue };
      let flags = palette.get(id).flags;
      if flags.contains(PaletteFlags::FLUID_SOURCE) {
        if !self.sources.contains(&cell) {
          self.sources.push(cell);
        }
      } else if flags.contains(PaletteFlags::FLUID) && self.mass.insert(cell, 1.0).is_none() {
        self.active.push(cell);
      }
    }
    self.last_edit = grid.edit_log_seq();
  }

  fn in_bounds(&self, p: IVec3) -> bool {
    p.cmpge(self.map_min).all()
      && p.cmplt(self.map_max).all()
      && (p - self.cam_cell).abs().cmple(IVec3::splat(self.radius)).all()
  }

  fn kind(&self, grid: &VolumeGrid, p: IVec3) -> Kind {
    let sample = p * self.cell + IVec3::splat(self.cell / 2);
    let Some(id) = grid.get_voxel(VoxelCoord::from_ivec3(sample)) else { return Kind::Open };
    if id.is_air() {
      return Kind::Open;
    }
    let flags = grid.palette().get(id).flags;
    if flags.contains(PaletteFlags::FLUID_SOURCE) {
      Kind::Solid
    } else if flags.contains(PaletteFlags::FLUID) {
      Kind::Water
    } else {
      Kind::Solid
    }
  }

  fn mass_at(&self, p: IVec3) -> f32 {
    self.mass.get(&p).copied().unwrap_or(0.0)
  }

  fn is_dynamic(&self, p: IVec3) -> bool {
    self.mass.contains_key(&p)
  }

  fn capacity(&self, grid: &VolumeGrid, p: IVec3) -> f32 {
    if self.kind(grid, p) == Kind::Solid { 0.0 } else { (1.0 - self.mass_at(p)).max(0.0) }
  }

  fn static_head(&self, grid: &VolumeGrid, p: IVec3) -> f32 {
    let mut top = p;
    for _ in 0..256 {
      let q = top + IVec3::Y;
      if self.in_bounds(q) && self.kind(grid, q) == Kind::Water && !self.is_dynamic(q) {
        top = q;
      } else {
        break;
      }
    }
    top.y as f32 + 1.0
  }

  fn write_cell(&self, grid: &mut VolumeGrid, p: IVec3, water: bool) {
    let palette = if water { self.water } else { PaletteId::AIR };
    grid.fill_brick(p * self.cell, self.cell, palette);
  }

  fn add_mass(
    &mut self,
    p: IVec3,
    amount: f32,
    grid: &mut VolumeGrid,
    next: &mut Vec<IVec3>,
    seen: &mut HashSet<IVec3>,
  ) {
    if amount <= FLOW_EPS || !self.in_bounds(p) {
      return;
    }
    let k = self.kind(grid, p);
    if k == Kind::Solid {
      return;
    }
    if !self.is_dynamic(p) {
      if k == Kind::Water {
        return;
      }
      self.mass.insert(p, 0.0);
    }
    let m = self.mass.get_mut(&p).expect("inserted above");
    *m = (*m + amount).min(1.0);
    if *m >= RENDER_MIN && k == Kind::Open {
      self.write_cell(grid, p, true);
    }
    if amount >= WAKE_EPS {
      self.wake_water(next, seen, p);
    }
  }

  fn remove_mass(
    &mut self,
    p: IVec3,
    amount: f32,
    grid: &mut VolumeGrid,
    next: &mut Vec<IVec3>,
    seen: &mut HashSet<IVec3>,
  ) {
    if amount <= 0.0 {
      return;
    }
    let Some(cur) = self.mass.get(&p).copied() else { return };
    let left = (cur - amount).max(0.0);
    let gone = left <= MASS_EPS;
    if gone {
      self.mass.remove(&p);
      if self.kind(grid, p) == Kind::Water {
        self.write_cell(grid, p, false);
      }
    } else {
      self.mass.insert(p, left);
    }
    if amount >= WAKE_EPS || gone {
      self.wake_water(next, seen, p);
    }
  }

  fn wake_water(&self, out: &mut Vec<IVec3>, seen: &mut HashSet<IVec3>, p: IVec3) {
    for d in [IVec3::ZERO, IVec3::Y, IVec3::NEG_Y, IVec3::X, IVec3::NEG_X, IVec3::Z, IVec3::NEG_Z] {
      let q = p + d;
      if self.mass.contains_key(&q) && seen.insert(q) {
        out.push(q);
      }
    }
  }

  fn seed_edit(&mut self, grid: &VolumeGrid, lo: IVec3, hi: IVec3, wake: &mut HashSet<IVec3>) {
    let c = IVec3::splat(self.cell);
    let a = lo.div_euclid(c) - IVec3::ONE;
    let b = (hi - IVec3::ONE).div_euclid(c) + IVec3::ONE;
    for y in a.y..=b.y {
      for z in a.z..=b.z {
        for x in a.x..=b.x {
          let p = IVec3::new(x, y, z);
          if !self.in_bounds(p) {
            continue;
          }
          match self.kind(grid, p) {
            Kind::Open => {
              self.mass.entry(p).or_insert(0.0);
              wake.insert(p);
              for d in [IVec3::Y, IVec3::NEG_Y, IVec3::X, IVec3::NEG_X, IVec3::Z, IVec3::NEG_Z] {
                let q = p + d;
                if self.is_dynamic(q) {
                  wake.insert(q);
                }
              }
            }
            Kind::Solid => {
              if self.is_dynamic(p) {
                wake.insert(p);
              }
            }
            Kind::Water => {
              if self.is_dynamic(p) {
                wake.insert(p);
                continue;
              }
              for d in [IVec3::Y, IVec3::NEG_Y, IVec3::X, IVec3::NEG_X, IVec3::Z, IVec3::NEG_Z] {
                let q = p + d;
                if !self.in_bounds(q) || self.kind(grid, q) != Kind::Open {
                  continue;
                }
                self.mass.entry(q).or_insert(0.0);
                wake.insert(q);
              }
            }
          }
        }
      }
    }
  }

  fn step(&mut self, grid: &mut VolumeGrid) {
    let mut next = Vec::new();
    let mut next_seen = HashSet::new();

    let mut active = std::mem::take(&mut self.active);
    active.append(&mut self.wake);
    let mut active_seen = HashSet::new();
    active.retain(|p| self.in_bounds(*p) && active_seen.insert(*p));

    let inject = SOURCE_RATE * TICK_DT;
    for i in 0..self.sources.len() {
      let t = self.sources[i] - IVec3::Y;
      if !self.in_bounds(t) {
        continue;
      }
      let cap = self.capacity(grid, t);
      if cap <= FLOW_EPS {
        continue;
      }
      let add = inject.min(cap);
      self.add_mass(t, add, grid, &mut next, &mut next_seen);
      if active_seen.insert(t) {
        active.push(t);
      }
    }

    active.sort_unstable_by_key(|p| p.y);
    let carry = if active.len() > BUDGET { active.split_off(BUDGET) } else { Vec::new() };

    let mut falling = HashSet::new();
    for &p in &active {
      let m = self.mass_at(p);
      if self.kind(grid, p) == Kind::Solid {
        self.mass.remove(&p);
        continue;
      }
      if falling.contains(&(p - IVec3::Y)) {
        falling.insert(p);
      }
      let below = p - IVec3::Y;
      if self.in_bounds(below) && self.kind(grid, below) == Kind::Open {
        let cap = self.capacity(grid, below);
        if cap > FLOW_EPS {
          let moved = m.min(cap);
          if moved > FLOW_EPS {
            self.remove_mass(p, moved, grid, &mut next, &mut next_seen);
            self.add_mass(below, moved, grid, &mut next, &mut next_seen);
            falling.insert(below);
          }
        }
      }
      let above = p + IVec3::Y;
      if self.in_bounds(above) && self.kind(grid, above) == Kind::Water && !self.is_dynamic(above) {
        let cap = 1.0 - self.mass_at(p);
        let add = (FLOW_RATE * TICK_DT).min(cap);
        if add > FLOW_EPS {
          self.add_mass(p, add, grid, &mut next, &mut next_seen);
        }
      }
    }

    let mut heads = HashMap::new();
    for &p in &active {
      let m = self.mass_at(p);
      if self.kind(grid, p) == Kind::Solid {
        continue;
      }
      let falls = falling.contains(&p);
      if falls && falling.contains(&(p - IVec3::Y)) {
        continue;
      }
      let hp = if falls { p.y as f32 + m } else { head_of(&self.mass, &falling, &mut heads, p) };
      let mut moves = [(IVec3::ZERO, 0.0_f32); 4];
      let mut total = 0.0;
      let mut count = 0;
      let mut source_in = 0.0;
      let mut sink_out = 0.0;
      for d in [IVec3::X, IVec3::NEG_X, IVec3::Z, IVec3::NEG_Z] {
        let n = p + d;
        if !self.in_bounds(n) {
          continue;
        }
        let nk = self.kind(grid, n);
        if nk == Kind::Solid {
          continue;
        }
        let dynamic = self.is_dynamic(n);
        let hn = if dynamic {
          head_of(&self.mass, &falling, &mut heads, n)
        } else if nk == Kind::Water {
          self.static_head(grid, n)
        } else {
          n.y as f32
        };
        let dh = hp - hn;
        if dh <= 0.0 {
          if dh < 0.0 && nk == Kind::Water && !dynamic {
            let cap = 1.0 - self.mass_at(p);
            if cap > FLOW_EPS {
              let want = (-dh * FLOW_RATE * TICK_DT).min(cap);
              if want > FLOW_EPS {
                source_in += want;
              }
            }
          }
          continue;
        }
        if nk == Kind::Water && !dynamic {
          sink_out += (dh * FLOW_RATE * TICK_DT).min(1.0);
          continue;
        }
        let cap = 1.0 - self.mass_at(n);
        if cap <= FLOW_EPS {
          continue;
        }
        let want = (dh * FLOW_RATE * TICK_DT).min(cap);
        if want <= FLOW_EPS {
          continue;
        }
        moves[count] = (n, want);
        count += 1;
        total += want;
      }
      if total > FLOW_EPS {
        let scale = (m * FLOW_MAX_FRAC / total).min(1.0);
        for &(n, want) in &moves[..count] {
          let cap = 1.0 - self.mass_at(n);
          let flux = (want * scale).min(cap);
          if flux <= FLOW_EPS {
            continue;
          }
          self.remove_mass(p, flux, grid, &mut next, &mut next_seen);
          self.add_mass(n, flux, grid, &mut next, &mut next_seen);
        }
      }
      if sink_out > FLOW_EPS {
        let out = sink_out.min(m * FLOW_MAX_FRAC);
        if out > FLOW_EPS {
          self.remove_mass(p, out, grid, &mut next, &mut next_seen);
        }
      }
      if source_in > FLOW_EPS {
        let cap = 1.0 - self.mass_at(p);
        let add = source_in.min(cap);
        if add > FLOW_EPS {
          self.add_mass(p, add, grid, &mut next, &mut next_seen);
        }
      }
    }

    next.extend(carry);
    self.active = next;
  }
}

fn head_of(
  mass: &HashMap<IVec3, f32>,
  falling: &HashSet<IVec3>,
  heads: &mut HashMap<IVec3, f32>,
  p: IVec3,
) -> f32 {
  if let Some(&h) = heads.get(&p) {
    return h;
  }
  let Some(&m) = mass.get(&p) else {
    return p.y as f32;
  };
  if falling.contains(&p) {
    let h = p.y as f32 + m;
    heads.insert(p, h);
    return h;
  }
  let mut top = p;
  while mass.contains_key(&(top + IVec3::Y)) && !falling.contains(&(top + IVec3::Y)) {
    top += IVec3::Y;
  }
  let h = top.y as f32 + mass.get(&top).copied().unwrap_or(0.0);
  let mut q = p;
  while q.y <= top.y {
    heads.insert(q, h);
    q += IVec3::Y;
  }
  h
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
    sim.cam_cell = (cam.position_world / sim.cell as f32).floor().as_ivec3();
  }

  let seq = grid.edit_log_seq();
  if seq > sim.last_edit {
    let edits: Vec<(IVec3, IVec3)> = grid.edits_since(sim.last_edit).collect();
    let mut wake = HashSet::new();
    if edits.len() > 64 {
      let (mut lo, mut hi) = (IVec3::splat(i32::MAX), IVec3::splat(i32::MIN));
      for &(a, b) in &edits {
        lo = lo.min(a);
        hi = hi.max(b);
      }
      sim.seed_edit(grid, lo, hi, &mut wake);
    } else {
      for &(lo, hi) in &edits {
        sim.seed_edit(grid, lo, hi, &mut wake);
      }
    }
    if !wake.is_empty() {
      bevy::log::debug!("WATER wake {} edit", wake.len());
    }
    sim.wake.extend(wake);
    sim.last_edit = seq;
  }

  sim.step(grid);
  sim.last_edit = grid.edit_log_seq();
}

pub fn advance_water_time(time: Res<Time>, mut settings: ResMut<gate_render::WaterSettings>) {
  settings.time += time.delta_secs();
}
