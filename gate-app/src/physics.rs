use bevy::prelude::*;
use glam::{IVec3, Vec3};

use gate_physics::{
  Field, PhysicsWorld, Probe, StepConfig, manifold_bounded, mass_properties, tight_bounds,
};
use gate_render::{DdaCameraConfig, VoxelScene};
use gate_voxel::{VolumeGrid, Volumes};

use crate::edit::{BrushMaterial, BrushShape, Burst};

const REGION_SIDE: i32 = 32;
const REGION_DOWN: i32 = 192;
const REGION_UP: i32 = 16;
const FIXED_DT: f32 = 1.0 / 60.0;
const MAX_STEPS_PER_FRAME: u32 = 1;
pub const PHYS_SUBSTEPS: [u32; 4] = [4, 8, 16, 32];
pub const PHYS_RATE_HZ: [u32; 3] = [60, 30, 20];
const PHYS_SUBSTEPS_DEFAULT: usize = 1;

#[derive(Resource)]
pub struct PhysicsState {
  pub world: PhysicsWorld,
  pub cfg: StepConfig,
  static_ready: bool,
  region: Option<(IVec3, IVec3)>,
  accumulator: f32,
  last_penetration: f32,
  last_contacts: usize,
  last_pairs: usize,
  last_split: (u32, usize),
  last_mix: (usize, usize, usize),
  last_profile: gate_physics::StepProfile,
  pub tick_div: u32,
}

impl Default for PhysicsState {
  fn default() -> Self {
    Self {
      world: match crate::consts::phys_threads() {
        Some(n) => PhysicsWorld::with_threads(n),
        None => PhysicsWorld::new(),
      },
      cfg: StepConfig {
        substeps: PHYS_SUBSTEPS[PHYS_SUBSTEPS_DEFAULT],
        profile: crate::consts::phys_profile(),
        ..StepConfig::default()
      },
      static_ready: false,
      region: None,
      accumulator: 0.0,
      last_penetration: 0.0,
      last_contacts: 0,
      last_pairs: 0,
      last_split: (0, 0),
      last_mix: (0, 0, 0),
      last_profile: gate_physics::StepProfile::default(),
      tick_div: 1,
    }
  }
}

impl PhysicsState {
  pub fn ensure_region(&mut self, main: &VolumeGrid, lo: Vec3, hi: Vec3) -> bool {
    let aabb_lo = lo.floor().as_ivec3();
    let aabb_hi = hi.ceil().as_ivec3();
    if self.static_ready
      && let Some((clo, chi)) = self.region
      && clo.cmple(aabb_lo).all()
      && chi.cmpge(aabb_hi).all()
    {
      return false;
    }
    let region = region_covering(lo, hi);
    if self.static_ready {
      let n = self.world.rebuild_static(main, region);
      debug!(target: "gate", "PHYS[region] 重建 [{:?}]-[{:?}] 角棱 {n}", region.0, region.1);
    } else {
      self.world.init_static_world(main, region, 0);
      self.static_ready = true;
      info!(
        target: "gate",
        "PHYS[region] 建立 [{:?}]-[{:?}] 角棱 {}",
        region.0, region.1, self.world.bodies.vox[0].len()
      );
    }
    self.region = Some(region);
    true
  }

  pub fn ensure_static_world(&mut self, main: &VolumeGrid, center: Vec3) -> bool {
    if self.static_ready {
      return false;
    }
    self.ensure_region(main, center, center);
    true
  }

  pub fn add_object_body(
    &mut self,
    main: &VolumeGrid,
    grid: &VolumeGrid,
    grid_index: usize,
  ) -> Option<usize> {
    let props = mass_properties(grid, self.cfg.density)?;
    let bounds = tight_bounds(grid)?;
    let tr = grid.transform();
    self.ensure_static_world(main, tr.pos);
    Some(self.world.add_body(grid, props, bounds, grid_index, tr))
  }

  pub fn remove_object_body(&mut self, grid_index: usize) -> bool {
    self.world.body_of_grid(grid_index).is_some_and(|i| self.world.remove_body(i))
  }

  pub fn clear_object_bodies(&mut self) -> usize {
    self.world.clear_dynamic()
  }

  pub fn live_bodies(&self) -> usize {
    (0..self.world.bodies.len()).filter(|&i| !self.world.bodies.is_static(i)).count()
  }

  pub fn reset(&mut self) {
    self.world = PhysicsWorld::new();
    self.static_ready = false;
    self.region = None;
    self.accumulator = 0.0;
  }
}

fn region_covering(lo: Vec3, hi: Vec3) -> (IVec3, IVec3) {
  (
    lo.floor().as_ivec3() - IVec3::new(REGION_SIDE, REGION_DOWN, REGION_SIDE),
    hi.ceil().as_ivec3() + IVec3::new(REGION_SIDE, REGION_UP, REGION_SIDE),
  )
}

fn awake_bounds(world: &PhysicsWorld) -> Option<(Vec3, Vec3)> {
  let mut bounds: Option<(Vec3, Vec3)> = None;
  for i in 0..world.bodies.len() {
    if world.bodies.is_static(i) || world.bodies.sleeping[i] {
      continue;
    }
    let (blo, bhi) = world.bodies.world_aabb(i);
    bounds = Some(match bounds {
      Some((lo, hi)) => (lo.min(blo), hi.max(bhi)),
      None => (blo, bhi),
    });
  }
  bounds
}

pub(crate) fn follow_view(mut st: ResMut<PhysicsState>, scene: Option<Res<VoxelScene>>) {
  let Some(scene) = scene else { return };
  let Some((lo, hi)) = awake_bounds(&st.world) else { return };
  st.ensure_region(scene.volumes.main(), lo, hi);
}

pub(crate) fn step_physics(
  time: Res<Time>,
  mut st: ResMut<PhysicsState>,
  scene: Option<ResMut<VoxelScene>>,
  mut diag: Local<(f64, u32)>,
) {
  let _t = gate_render::profiler::SysTimer::new("PHYS 求解", &mut diag);
  let Some(mut scene) = scene else { return };
  if !st.static_ready || st.live_bodies() == 0 {
    st.accumulator = 0.0;
    return;
  }
  let div = st.tick_div.max(1);
  st.accumulator =
    (st.accumulator + time.delta_secs() / div as f32).min(FIXED_DT * MAX_STEPS_PER_FRAME as f32);
  let mut steps = 0u32;
  while st.accumulator >= FIXED_DT {
    let PhysicsState { world, cfg, .. } = &mut *st;
    let stats = world.step(&scene.volumes, FIXED_DT, cfg);
    st.accumulator -= FIXED_DT;
    st.last_penetration = stats.max_penetration;
    st.last_contacts = stats.contacts;
    st.last_pairs = stats.pairs;
    st.last_split = (stats.subdiv, stats.reused_contacts);
    st.last_mix = (stats.corner_contacts, stats.edge_contacts, stats.lateral_contacts);
    st.last_profile = stats.profile;
    steps += 1;
  }
  if steps == 0 {
    return;
  }
  if write_back(&st.world, &mut scene.volumes) {
    scene.transforms_dirty = true;
  }
}

pub(crate) fn register_callbacks(world: &mut World) {
  world.add_observer(|ev: On<gate_ui::MenuActionEvent>, mut st: ResMut<PhysicsState>| {
    match (ev.path.as_str(), &ev.action) {
      ("game/phys/substeps", gate_ui::MenuAction::Select(i)) => {
        st.cfg.substeps = PHYS_SUBSTEPS.get(*i).copied().unwrap_or(st.cfg.substeps);
        info!(target: "gate", "PHYS[cfg] 子步 → {}", st.cfg.substeps);
      }
      ("game/phys/rate", gate_ui::MenuAction::Select(i)) => {
        let hz = PHYS_RATE_HZ.get(*i).copied().unwrap_or(PHYS_RATE_HZ[0]);
        st.tick_div = (60 / hz).max(1);
        info!(target: "gate", "PHYS[cfg] 步频 → {hz} Hz");
      }
      _ => {}
    }
  });
}

pub(crate) fn selftest(
  mut done: Local<bool>,
  mut frames: Local<u32>,
  cam: Res<DdaCameraConfig>,
  mut st: ResMut<PhysicsState>,
  scene: Option<ResMut<VoxelScene>>,
) {
  if !crate::consts::phys_selftest() {
    return;
  }
  let Some(mut scene) = scene else { return };
  if !*done {
    *done = true;
    if crate::consts::phys_pile() {
      return;
    }
    let base = cam.position_world + cam.forward * 160.0;
    for k in 0..5 {
      let at = base + Vec3::new((k as f32 - 2.0) * 40.0, 40.0 + k as f32 * 48.0, 0.0);
      crate::objects::place(
        &mut scene,
        &mut st,
        at,
        Burst::fixed(BrushShape::Cube, 13, BrushMaterial::default()),
      );
    }
    info!(target: "gate", "PHYS[selftest] 投放 5 箱 @({:.0},{:.0},{:.0})", base.x, base.y, base.z);
    return;
  }
  *frames += 1;
  if !frames.is_multiple_of(60) {
    return;
  }
  let mut s = String::new();
  if !crate::consts::phys_pile() {
    for i in 0..st.world.bodies.len() {
      if st.world.bodies.is_static(i) {
        continue;
      }
      let p = st.world.bodies.field_transform(i).pos;
      let v = st.world.bodies.lin_vel[i].length();
      let w = st.world.bodies.ang_vel[i].length();
      let state = if st.world.bodies.sleeping[i] { "睡" } else { "醒" };
      s.push_str(&format!("[{i} {state} {:.0},{:.0},{:.0} v{v:.2} w{w:.3}]", p.x, p.y, p.z));
    }
  }
  info!(
    target: "gate",
    "PHYS[selftest] 体={} 内步{} 配{} 接触{} 复用{} 深度{:.2} 角{}棱{}横{} {s}",
    st.live_bodies(),
    st.last_split.0,
    st.last_pairs,
    st.last_contacts,
    st.last_split.1,
    st.last_penetration,
    st.last_mix.0,
    st.last_mix.1,
    st.last_mix.2
  );
  if st.cfg.profile {
    let p = st.last_profile;
    info!(
      target: "gate",
      "PHYS[prof] 宽{:.2} 窄{:.2} 色{:.2} 备{:.2} 解{:.2} 积{:.2} 睡{:.2} 共{:.2} ms/步",
      p.broad_ms, p.build_ms, p.color_ms, p.prepare_ms, p.solve_ms, p.integrate_ms, p.sleep_ms,
      p.total_ms()
    );
  }
  if st.live_bodies() >= 2 && !crate::consts::phys_pile() {
    let d1 = manifold_dump(&st, &scene.volumes, 1);
    let d2 = manifold_dump(&st, &scene.volumes, 2);
    info!(target: "gate", "PHYS[selftest]{d1}\nPHYS[selftest]{d2}");
  }
}

fn manifold_dump(st: &PhysicsState, volumes: &Volumes, body: usize) -> String {
  let b = &st.world.bodies;
  let Some(&gi) = b.grid_index.get(body) else { return String::new() };
  let a = Probe {
    field: Field::new_at(&volumes.list[gi], b.field_transform(body)),
    vox: &b.vox[body],
    local: b.local_bounds[body],
    world: b.world_aabb(body),
  };
  let w = Probe {
    field: Field::new_at(&volumes.list[b.grid_index[0]], b.field_transform(0)),
    vox: &b.vox[0],
    local: b.local_bounds[0],
    world: b.world_aabb(0),
  };
  let m = manifold_bounded(&a, &w, &st.cfg.contact);
  let lo = m.points.iter().map(|p| p.depth).fold(f32::MAX, f32::min);
  let hi = m.points.iter().map(|p| p.depth).fold(f32::MIN, f32::max);
  let mut s = format!(" 体{body} 接触{} 深{lo:.2}..{hi:.2}:", m.points.len());
  for p in m.points.iter().take(8) {
    s.push_str(&format!(
      " n({:+.1},{:+.1},{:+.1})d{:.2}@({:.1},{:.1},{:.1})",
      p.normal.x, p.normal.y, p.normal.z, p.depth, p.point.x, p.point.y, p.point.z
    ));
  }
  s
}

fn write_back(world: &PhysicsWorld, volumes: &mut Volumes) -> bool {
  let mut moved = false;
  for i in 0..world.bodies.len() {
    if world.bodies.is_static(i) {
      continue;
    }
    let Some(g) = volumes.list.get_mut(world.bodies.grid_index[i]) else { continue };
    let tr = world.bodies.field_transform(i);
    let cur = g.transform();
    if cur.rot != tr.rot || cur.scale != tr.scale || (cur.pos - tr.pos).length_squared() > 1e-6 {
      g.set_transform(tr.pos, tr.rot, tr.scale);
      moved = true;
    }
  }
  moved
}
