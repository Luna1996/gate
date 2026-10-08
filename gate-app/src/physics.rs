use bevy::prelude::*;
use glam::{IVec3, Vec3};

use gate_physics::{Field, PhysicsWorld, Probe, StepConfig, mass_properties, tight_bounds};
use gate_render::{DdaCameraConfig, VoxelScene};
use gate_voxel::{VolumeGrid, Volumes};

use crate::edit::{BrushMaterial, BrushShape, Burst};

const STATIC_MARGIN: f32 = 32.0;
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
  pub fn sync_statics(&mut self, main: &VolumeGrid, lo: Vec3, hi: Vec3) -> usize {
    let m = Vec3::splat(STATIC_MARGIN);
    let lo = (lo - m).floor().as_ivec3();
    let hi = (hi + m).ceil().as_ivec3();
    self.world.sync_static(main, lo, hi)
  }

  pub fn ensure_static_world(&mut self, main: &VolumeGrid, center: Vec3) -> bool {
    if self.static_ready {
      return false;
    }
    let b = center.floor().as_ivec3();
    self.world.init_static_world(main, (b, b), 0);
    self.static_ready = true;
    info!(target: "gate", "PHYS[static] 建立 角棱 {}", self.world.statics.corners());
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
    self.accumulator = 0.0;
  }
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

pub(crate) fn follow_view(
  mut st: ResMut<PhysicsState>,
  scene: Option<Res<VoxelScene>>,
  cam: Res<DdaCameraConfig>,
  mut diag: Local<(f64, u32)>,
) {
  let _t = gate_render::profiler::SysTimer::new("PHYS 静态", &mut diag);
  if !st.static_ready {
    return;
  }
  let Some(scene) = scene else { return };
  let (lo, hi) = match awake_bounds(&st.world) {
    Some(b) => b,
    None => (cam.position_world, cam.position_world),
  };
  st.sync_statics(scene.volumes.main(), lo, hi);
}

pub(crate) fn step_physics(
  time: Res<Time>,
  mut st: ResMut<PhysicsState>,
  scene: Option<ResMut<VoxelScene>>,
  mut diag: Local<(f64, u32)>,
  mut dbg_n: Local<u32>,
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
  *dbg_n += 1;
  if st.cfg.profile && *dbg_n % 30 == 0 {
    let p = st.last_profile;
    info!(
      target: "gate",
      "PHYS[profile] 体{} 对{} 接触{} 内步{} 宽{:.2} 配对{:.2} 窄{:.2} 装配{:.2} 色{:.2} 备{:.2} 解{:.2} 积{:.2} 位{:.2} 回写{:.2} 睡{:.2} | 合计{:.2}",
      st.live_bodies(),
      st.last_pairs,
      st.last_contacts,
      st.last_split.0,
      p.broad_ms,
      p.pair_ms,
      p.build_ms,
      p.setup_ms,
      p.color_ms,
      p.prepare_ms,
      p.solve_ms,
      p.integrate_ms,
      p.position_ms,
      p.carry_ms,
      p.sleep_ms,
      p.total_ms(),
    );
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
  let asleep = (1..st.world.bodies.len()).filter(|&i| st.world.bodies.sleeping[i]).count();
  info!(
    target: "gate",
    "PHYS[selftest] 体={} 睡{} 内步{} 配{} 接触{} 复用{} 深度{:.2} 角{}棱{}横{} {s}",
    st.live_bodies(),
    asleep,
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
  let gw = &volumes.list[b.grid_index[0]];
  let gw_tr = gw.transform();
  let (c_lo, c_hi) = (
    gate_physics::chunk_of(a.world.0.floor().as_ivec3()),
    gate_physics::chunk_of(a.world.1.ceil().as_ivec3()),
  );
  let mut cand: Vec<gate_physics::Contact> = Vec::new();
  let mut pts: Vec<gate_physics::Contact> = Vec::new();
  for cz in c_lo.z..=c_hi.z {
    for cy in c_lo.y..=c_hi.y {
      for cx in c_lo.x..=c_hi.x {
        let Some(sc) = st.world.statics.get(IVec3::new(cx, cy, cz)) else { continue };
        let w = Probe {
          field: Field::new_at(gw, gw_tr),
          vox: &sc.vox,
          local: sc.local,
          world: gate_physics::world_aabb_of(sc.local, gw_tr),
        };
        gate_physics::probe_pairs(&a, &w, &mut cand);
      }
    }
  }
  gate_physics::finish(&mut cand, &st.cfg.contact, a.field.transform().scale, &mut pts);
  let lo = pts.iter().map(|p| p.depth).fold(f32::MAX, f32::min);
  let hi = pts.iter().map(|p| p.depth).fold(f32::MIN, f32::max);
  let mut s = format!(" 体{body} 接触{} 深{lo:.2}..{hi:.2}:", pts.len());
  for p in pts.iter().take(8) {
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
    if world.bodies.is_static(i) || world.bodies.is_frozen(i) {
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
