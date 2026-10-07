use std::cell::UnsafeCell;
use std::collections::HashMap;

use gate_voxel::{VolumeGrid, VolumeTransform, Volumes};
use glam::{IVec3, Vec3};
use micropool::iter::{
  ExactParallelSourceExt, IntoExactParallelRefMutSource, IntoExactParallelRefSource,
  ParallelIteratorExt,
};
use micropool::{ThreadPool, ThreadPoolBuilder};

use crate::body::{BodySet, MassProps};
use crate::contact::{ContactConfig, ContactPath, Probe, manifold_bounded};
use crate::dissolve::dissolve_into_main;
use crate::field::Field;
use crate::solver::{Constraint, ContactConstraint, ContactParams, color_by_body};

pub const VOXELS_PER_METER: f32 = 50.0;

pub const WORLD_BODY: usize = 0;

#[derive(Clone, Copy, Debug)]
pub struct StepConfig {
  pub gravity: Vec3,
  pub substeps: u32,
  pub iterations: u32,
  pub params: ContactParams,
  pub contact: ContactConfig,
  pub density: f32,
  pub sleep_lin_vel: f32,
  pub sleep_ang_vel: f32,
  pub sleep_time: f32,
  pub profile: bool,
}

impl Default for StepConfig {
  fn default() -> Self {
    Self {
      gravity: Vec3::new(0.0, -9.81 * VOXELS_PER_METER, 0.0),
      substeps: 32,
      iterations: 1,
      params: ContactParams::default(),
      contact: ContactConfig::default(),
      density: 1.0,
      sleep_lin_vel: 0.5,
      sleep_ang_vel: 0.05,
      sleep_time: 0.5,
      profile: false,
    }
  }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct StepProfile {
  pub broad_ms: f32,
  pub build_ms: f32,
  pub color_ms: f32,
  pub prepare_ms: f32,
  pub solve_ms: f32,
  pub integrate_ms: f32,
  pub sleep_ms: f32,
}

impl StepProfile {
  pub fn total_ms(&self) -> f32 {
    self.broad_ms
      + self.build_ms
      + self.color_ms
      + self.prepare_ms
      + self.solve_ms
      + self.integrate_ms
      + self.sleep_ms
  }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct StepStats {
  pub pairs: usize,
  pub contacts: usize,
  pub subdiv: u32,
  pub reused_contacts: usize,
  pub sleeping: usize,
  pub max_penetration: f32,
  pub corner_contacts: usize,
  pub edge_contacts: usize,
  pub lateral_contacts: usize,
  pub profile: StepProfile,
}

#[derive(Default)]
pub struct PhysicsWorld {
  pub bodies: BodySet,
  cache: HashMap<u64, [f32; 3]>,
  pool: Option<ThreadPool>,
}

impl PhysicsWorld {
  pub fn new() -> Self {
    Self::default()
  }

  pub fn with_threads(num_threads: usize) -> Self {
    Self {
      pool: Some(ThreadPoolBuilder::default().num_threads(num_threads).build()),
      ..Self::default()
    }
  }

  pub fn init_static_world(
    &mut self,
    grid: &VolumeGrid,
    bounds: (IVec3, IVec3),
    grid_index: usize,
  ) -> usize {
    let i = self.bodies.push_static(bounds, grid_index);
    self.bodies.build_accel(i, grid);
    debug_assert_eq!(i, WORLD_BODY, "静态世界必须是第 0 个体");
    i
  }

  pub fn rebuild_static(&mut self, grid: &VolumeGrid, bounds: (IVec3, IVec3)) -> usize {
    self.bodies.local_bounds[WORLD_BODY] = bounds;
    self.bodies.build_accel(WORLD_BODY, grid);
    self.cache.clear();
    self.bodies.vox[WORLD_BODY].len()
  }

  pub fn add_body(
    &mut self,
    grid: &VolumeGrid,
    props: MassProps,
    bounds: (IVec3, IVec3),
    grid_index: usize,
    tr: VolumeTransform,
  ) -> usize {
    let i = self.bodies.push_dynamic(props, bounds, grid_index, Vec3::ZERO, tr.rot, tr.scale);
    self.bodies.pos[i] = tr.pos + tr.rot * (self.bodies.com[i] * tr.scale);
    self.bodies.build_accel(i, grid);
    i
  }

  pub fn launch(&mut self, i: usize, lin_vel: Vec3) {
    if i == WORLD_BODY || i >= self.bodies.len() {
      return;
    }
    self.bodies.lin_vel[i] = lin_vel;
    self.bodies.wake(i);
  }

  pub fn remove_body(&mut self, i: usize) -> bool {
    if i == WORLD_BODY || i >= self.bodies.len() {
      return false;
    }
    self.bodies.remove(i);
    self.cache.clear();
    true
  }

  pub fn body_of_grid(&self, grid_index: usize) -> Option<usize> {
    (0..self.bodies.len())
      .find(|&i| !self.bodies.is_static(i) && self.bodies.grid_index[i] == grid_index)
  }

  pub fn clear_dynamic(&mut self) -> usize {
    let mut n = 0;
    while self.bodies.len() > WORLD_BODY + 1 {
      self.bodies.remove(WORLD_BODY + 1);
      n += 1;
    }
    self.cache.clear();
    n
  }

  pub fn step(&mut self, grids: &Volumes, dt: f32, cfg: &StepConfig) -> StepStats {
    let Self { bodies, cache, pool } = self;
    let pool = pool.get_or_insert_with(|| ThreadPoolBuilder::default().build());
    //
    let fast = (0..bodies.len())
      .filter(|&i| !bodies.sleeping[i] && bodies.lin_vel[i].length() * dt > MAX_STEP_TRAVEL)
      .count();
    let k = if fast == 0 || fast > MAX_SUBDIV_BODIES {
      1
    } else {
      let travel = (0..bodies.len())
        .filter(|&i| !bodies.sleeping[i])
        .map(|i| bodies.lin_vel[i].length())
        .fold(0.0f32, f32::max)
        * dt;
      (travel / MAX_STEP_TRAVEL).ceil().clamp(1.0, MAX_SUBDIV as f32) as u32
    };
    pool.install(|| {
      if k == 1 {
        let pairs = broad_phase(bodies);
        let mut s = step_inner(bodies, cache, grids, &pairs, &pairs, &[], dt, cfg);
        s.pairs = pairs.len();
        s.subdiv = 1;
        return s;
      }
      let pairs = broad_phase(bodies);
      let limit = MAX_STEP_TRAVEL / k as f32;
      let moving: Vec<bool> = (0..bodies.len())
        .map(|i| !bodies.sleeping[i] && bodies.lin_vel[i].length() * dt > limit)
        .collect();
      let (fresh, stale): (Vec<_>, Vec<_>) =
        pairs.iter().copied().partition(|&(a, b)| moving[a] || moving[b]);
      let stale_cons = build(grids, bodies, &stale, cfg);
      let mut stats = StepStats::default();
      for _ in 0..k {
        let s = step_inner(bodies, cache, grids, &fresh, &pairs, &stale_cons, dt / k as f32, cfg);
        merge_stats(&mut stats, &s);
      }
      stats.pairs = pairs.len();
      stats.subdiv = k;
      stats.reused_contacts = stale_cons.len() * k as usize;
      stats
    })
  }

  pub fn dissolve_body(&mut self, i: usize, volumes: &mut Volumes) -> usize {
    if i == WORLD_BODY || i >= self.bodies.len() {
      return 0;
    }
    let grid_index = self.bodies.grid_index[i];
    if grid_index == 0 || grid_index >= volumes.list.len() {
      return 0;
    }
    let tr = self.bodies.field_transform(i);
    volumes.list[grid_index].set_transform(tr.pos, tr.rot, tr.scale);
    let written = dissolve_into_main(volumes, grid_index, tr);
    self.remove_body(i);
    written
  }

  pub fn settled_for_dissolve(
    &self,
    volumes: &Volumes,
    eye: Vec3,
    view_dir: Vec3,
    min_sleep: f32,
    cos_threshold: f32,
  ) -> Vec<usize> {
    let mut out = Vec::new();
    for i in 0..self.bodies.len() {
      if i == WORLD_BODY || self.bodies.is_static(i) || !self.bodies.sleeping[i] {
        continue;
      }
      if self.bodies.sleep_timer[i] < min_sleep {
        continue;
      }
      match volumes.list.get(self.bodies.grid_index[i]) {
        Some(g) if g.chunk_count() > 0 => {}
        _ => continue,
      }
      let to_obj = self.bodies.pos[i] - eye;
      let d = to_obj.length();
      if d < 1e-6 || view_dir.dot(to_obj / d) >= cos_threshold {
        continue;
      }
      out.push(i);
    }
    out.sort_unstable_by(|a, b| b.cmp(a));
    out
  }
}

fn merge_stats(acc: &mut StepStats, s: &StepStats) {
  acc.contacts = acc.contacts.max(s.contacts);
  acc.max_penetration = acc.max_penetration.max(s.max_penetration);
  acc.corner_contacts += s.corner_contacts;
  acc.edge_contacts += s.edge_contacts;
  acc.lateral_contacts += s.lateral_contacts;
  acc.sleeping = s.sleeping;
  let (a, b) = (&mut acc.profile, &s.profile);
  a.broad_ms += b.broad_ms;
  a.build_ms += b.build_ms;
  a.color_ms += b.color_ms;
  a.prepare_ms += b.prepare_ms;
  a.solve_ms += b.solve_ms;
  a.integrate_ms += b.integrate_ms;
  a.sleep_ms += b.sleep_ms;
}

#[allow(clippy::too_many_arguments)]
fn step_inner(
  bodies: &mut BodySet,
  cache: &mut HashMap<u64, [f32; 3]>,
  grids: &Volumes,
  build_pairs: &[(usize, usize)],
  sleep_pairs: &[(usize, usize)],
  reused: &[ContactConstraint],
  dt: f32,
  cfg: &StepConfig,
) -> StepStats {
  let mut stats = StepStats::default();
  let substeps = cfg.substeps.max(1);
  let h = dt / substeps as f32;
  let mut prof = StepProfile::default();
  let t = cfg.profile.then(std::time::Instant::now);
  for &(a, b) in sleep_pairs {
    let (sa, sb) = (bodies.sleeping[a], bodies.sleeping[b]);
    if sa == sb {
      continue;
    }
    let (asleep, mover) = if sa { (a, b) } else { (b, a) };
    if !bodies.is_slow(mover, cfg.sleep_lin_vel, cfg.sleep_ang_vel) {
      bodies.wake(asleep);
    }
  }
  tick(&mut prof.broad_ms, t);
  let t = cfg.profile.then(std::time::Instant::now);
  let mut cons = build(grids, bodies, build_pairs, cfg);
  cons.extend_from_slice(reused);
  count_contacts(&cons, &mut stats);
  bodies.update_all_inertia();
  let mut carried = std::mem::take(cache);
  for c in &mut cons {
    if let Some(&imp) = carried.get(&c.key()) {
      c.set_impulse(imp);
    }
  }
  tick(&mut prof.build_ms, t);
  let t = cfg.profile.then(std::time::Instant::now);
  let ranges = color_by_body(&mut cons, bodies);
  tick(&mut prof.color_ms, t);
  let share = h / dt;
  for _ in 0..substeps {
    let t = cfg.profile.then(std::time::Instant::now);
    for i in 0..bodies.len() {
      bodies.integrate_velocity(i, cfg.gravity, h);
    }
    cons
      .par_iter_mut()
      .with_thread_pool(micropool::split_by_threads())
      .for_each(|c| c.prepare(bodies));
    for_each_color(&mut cons, bodies, &ranges, |c, b| c.warm_start(b, share));
    tick(&mut prof.prepare_ms, t);
    let t = cfg.profile.then(std::time::Instant::now);
    for _ in 0..cfg.iterations.max(1) {
      for_each_color(&mut cons, bodies, &ranges, |c, b| c.solve(b));
    }
    tick(&mut prof.solve_ms, t);
    let t = cfg.profile.then(std::time::Instant::now);
    for i in 0..bodies.len() {
      bodies.integrate_position(i, h);
    }
    tick(&mut prof.integrate_ms, t);
  }
  carried.clear();
  for c in &cons {
    carried.insert(c.key(), c.impulse());
  }
  let t = cfg.profile.then(std::time::Instant::now);
  stats.max_penetration = cons.iter().map(|c| c.penetration()).fold(0.0f32, f32::max);
  update_sleep(bodies, sleep_pairs, dt, cfg);
  tick(&mut prof.sleep_ms, t);
  stats.profile = prof;
  stats.sleeping = (0..bodies.len()).filter(|&i| bodies.sleeping[i]).count();
  *cache = carried;
  stats
}

const MAX_STEP_TRAVEL: f32 = 2.0;

const MAX_SUBDIV: u32 = 16;

const MAX_SUBDIV_BODIES: usize = 32;

struct BodyWriter<'a> {
  bodies: UnsafeCell<&'a mut BodySet>,
}

unsafe impl Sync for BodyWriter<'_> {}

impl<'a> BodyWriter<'a> {
  fn new(bodies: &'a mut BodySet) -> Self {
    Self { bodies: UnsafeCell::new(bodies) }
  }

  #[allow(clippy::mut_from_ref)]
  unsafe fn get(&self) -> &mut BodySet {
    unsafe { *self.bodies.get() }
  }
}

fn for_each_color(
  cons: &mut [ContactConstraint],
  bodies: &mut BodySet,
  ranges: &[(usize, usize)],
  f: impl Fn(&mut ContactConstraint, &mut BodySet) + Sync,
) {
  let writer = BodyWriter::new(bodies);
  for &(lo, hi) in ranges {
    cons[lo..hi].par_iter_mut().with_thread_pool(micropool::split_by_threads()).for_each(|c| {
      f(c, unsafe { writer.get() });
    });
  }
}

fn build(
  grids: &Volumes,
  bodies: &BodySet,
  pairs: &[(usize, usize)],
  cfg: &StepConfig,
) -> Vec<ContactConstraint> {
  let parts: Vec<Vec<ContactConstraint>> =
    pairs.par_iter().with_thread_pool(micropool::split_per(64)).fold_per_thread(
      Vec::new,
      |mut acc, &(ia, ib)| {
        build_pair_into(&mut acc, grids, bodies, ia, ib, cfg);
        acc
      },
      Vec::with_capacity,
      |mut parts, acc| {
        parts.push(acc);
        parts
      },
    );
  let total = parts.iter().map(Vec::len).sum();
  let mut cons = Vec::with_capacity(total);
  for part in parts {
    cons.extend(part);
  }
  cons
}

fn count_contacts(cons: &[ContactConstraint], stats: &mut StepStats) {
  stats.contacts = cons.len();
  stats.corner_contacts = 0;
  stats.edge_contacts = 0;
  stats.lateral_contacts = 0;
  for c in cons {
    match c.path() {
      ContactPath::Corner => stats.corner_contacts += 1,
      ContactPath::Edge => stats.edge_contacts += 1,
    }
    if c.normal().y.abs() < 0.9 && c.penetration() > 0.1 {
      stats.lateral_contacts += 1;
    }
  }
}

fn build_pair_into(
  out: &mut Vec<ContactConstraint>,
  grids: &Volumes,
  bodies: &BodySet,
  ia: usize,
  ib: usize,
  cfg: &StepConfig,
) {
  if bodies.sleeping[ia] && bodies.sleeping[ib] {
    return;
  }
  let (Some(ga), Some(gb)) =
    (grids.list.get(bodies.grid_index[ia]), grids.list.get(bodies.grid_index[ib]))
  else {
    return;
  };
  let probe = if bodies.vox[ia].len() <= bodies.vox[ib].len() { ia } else { ib };
  let other = if probe == ia { ib } else { ia };
  let (gp, go) = if probe == ia { (ga, gb) } else { (gb, ga) };
  let pa = Probe {
    field: Field::new_at(gp, bodies.field_transform(probe)),
    vox: &bodies.vox[probe],
    local: bodies.local_bounds[probe],
    world: bodies.world_aabb(probe),
  };
  let pb = Probe {
    field: Field::new_at(go, bodies.field_transform(other)),
    vox: &bodies.vox[other],
    local: bodies.local_bounds[other],
    world: bodies.world_aabb(other),
  };
  let m = manifold_bounded(&pa, &pb, &cfg.contact);
  out.reserve(m.points.len());
  for c in &m.points {
    out.push(ContactConstraint::new(c, probe, other, &cfg.params));
  }
}

#[derive(Clone, Copy)]
struct Sweep {
  body: usize,
  lo: Vec3,
  hi: Vec3,
}

fn broad_phase(bodies: &BodySet) -> Vec<(usize, usize)> {
  let mut out = Vec::new();
  if bodies.is_empty() {
    return out;
  }
  let dyn_start = if bodies.is_static(WORLD_BODY) { 1 } else { 0 };
  let mut sweep: Vec<Sweep> = (dyn_start..bodies.len())
    .map(|body| {
      let (lo, hi) = bodies.world_aabb(body);
      Sweep { body, lo, hi }
    })
    .collect();
  sweep.sort_by(|a, b| a.lo.x.total_cmp(&b.lo.x));
  for i in 0..sweep.len() {
    let a = sweep[i];
    for b in sweep.iter().skip(i + 1) {
      if b.lo.x > a.hi.x {
        break;
      }
      if a.lo.y <= b.hi.y && b.lo.y <= a.hi.y && a.lo.z <= b.hi.z && b.lo.z <= a.hi.z {
        out.push((a.body.min(b.body), a.body.max(b.body)));
      }
    }
    if dyn_start == 1 {
      out.push((WORLD_BODY, a.body));
    }
  }
  out
}

fn tick(acc: &mut f32, t: Option<std::time::Instant>) {
  if let Some(t) = t {
    *acc += t.elapsed().as_secs_f32() * 1000.0;
  }
}

fn update_sleep(bodies: &mut BodySet, pairs: &[(usize, usize)], dt: f32, cfg: &StepConfig) {
  let n = bodies.len();
  let mut disturbed = vec![false; n];
  for &(a, b) in pairs {
    if !bodies.is_static(b) && !bodies.is_slow(b, cfg.sleep_lin_vel, cfg.sleep_ang_vel) {
      disturbed[a] = true;
    }
    if !bodies.is_static(a) && !bodies.is_slow(a, cfg.sleep_lin_vel, cfg.sleep_ang_vel) {
      disturbed[b] = true;
    }
  }
  for (i, &disturb) in disturbed.iter().enumerate() {
    if bodies.is_static(i) {
      continue;
    }
    if disturb || !bodies.is_slow(i, cfg.sleep_lin_vel, cfg.sleep_ang_vel) {
      bodies.wake(i);
      continue;
    }
    let t = bodies.sleep_timer[i] + dt;
    bodies.sleep_timer[i] = t;
    if t >= cfg.sleep_time {
      bodies.put_to_sleep(i);
    }
  }
}
