use bevy::prelude::*;
use glam::{IVec3, Vec3};

use gate_render::{DdaCameraConfig, OrbitCamera, VoxelScene, raycast, raycast_objects};
use gate_ui::{MenuAction, MenuActionEvent};

use crate::{
  camera::{CameraMode, FlyCamera, MouseLock, cursor_ray},
  edit::{
    BrushShape, Burst, EditSettings, EditTarget, HoldRepeat, RANDOM_SIZE_MAX, Rng, apply_brush,
    brush_radius, burst, material_slot,
  },
  physics::PhysicsState,
};

const DELETE_REACH: f32 = 4096.0;
const FIRE_DISTANCE: f32 = 8.0;
const FIRE_SPEED: f32 = 800.0;
pub(crate) const PILE_COUNT: usize = 1000;
const PILE_ALTITUDE: f32 = 512.0;
const PILE_PITCH: f32 = 0.75;
const PILE_PROBE_UP: f32 = 256.0;
const PILE_PROBE_DOWN: f32 = 8192.0;
const PILE_REACH: f32 = 4096.0;
const PILE_LANDING: f32 = 160.0;
const PILE_AIM_MAX: i32 = 24;
const PILE_AIM_STEP: f32 = 0.06;
const PILE_PITCH_MAX: f32 = 1.45;

fn view_ray(windows: &Query<&Window>, cfg: &DdaCameraConfig, locked: bool) -> Option<(Vec3, Vec3)> {
  let window = windows.single().ok()?;
  cursor_ray(window, cfg, locked)
}

pub(crate) fn place(
  scene: &mut VoxelScene,
  phys: &mut PhysicsState,
  center: Vec3,
  b: Burst,
) -> Option<i32> {
  spawn_body(scene, phys, center, b, None)
}

pub(crate) fn fire(
  scene: &mut VoxelScene,
  phys: &mut PhysicsState,
  origin: Vec3,
  dir: Vec3,
  b: Burst,
) {
  let r = brush_radius(b.size);
  let center = origin + dir * (FIRE_DISTANCE + r as f32);
  spawn_body(scene, phys, center, b, Some(dir * FIRE_SPEED));
}

fn spawn_body(
  scene: &mut VoxelScene,
  phys: &mut PhysicsState,
  center: Vec3,
  b: Burst,
  launch: Option<Vec3>,
) -> Option<i32> {
  let r = brush_radius(b.size);
  let pos = center - Vec3::splat(r as f32);
  let obj_id = scene.volumes.spawn_object(pos, b.rot, 1.0);
  let written = {
    let grid = scene.volumes.volume_mut(obj_id)?;
    let slot = material_slot(grid, b.mat);
    apply_brush(grid, IVec3::splat(r), b.shape, b.size, slot)
  };
  match launch {
    Some(v) => debug!(
      target: "gate",
      "OBJECT[fire] obj={obj_id} {:?} size={} {written}vx v={:.0}vx/s",
      b.shape,
      b.size,
      v.length()
    ),
    None => info!(
      target: "gate",
      "OBJECT[place] obj={obj_id} @({:.1},{:.1},{:.1}) {:?} size={} {written}vx",
      pos.x,
      pos.y,
      pos.z,
      b.shape,
      b.size
    ),
  }
  let grid_index = obj_id as usize + 1;
  let main = scene.volumes.main();
  match scene.volumes.list.get(grid_index).and_then(|g| phys.add_object_body(main, g, grid_index)) {
    Some(i) => match launch {
      Some(v) => phys.world.launch(i, v),
      None => info!(target: "gate", "PHYS[body] obj={obj_id} 体={i}"),
    },
    None => warn!(target: "gate", "PHYS[body] obj={obj_id} 质量为 0 → 不参与物理"),
  }
  Some(obj_id)
}

fn delete_pointed(scene: &mut VoxelScene, phys: &mut PhysicsState, origin: Vec3, dir: Vec3) {
  let Some(hit) = raycast_objects(&scene.volumes, origin, dir, DELETE_REACH) else {
    debug!(target: "gate", "OBJECT[delete] 未命中物体");
    return;
  };
  if scene.volumes.despawn_object(hit.obj_id) {
    phys.remove_object_body(hit.obj_id as usize + 1);
    info!(target: "gate", "OBJECT[delete] obj={}", hit.obj_id);
  }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn object_input(
  mouse: Res<ButtonInput<MouseButton>>,
  time: Res<Time>,
  captured: Res<gate_ui::UiPointerCaptured>,
  intercepted: Res<gate_ui::MouseIntercepted>,
  windows: Query<&Window>,
  cfg: Res<DdaCameraConfig>,
  mode: Res<CameraMode>,
  lock: Res<MouseLock>,
  settings: Res<EditSettings>,
  scene: Option<ResMut<VoxelScene>>,
  phys: Option<ResMut<PhysicsState>>,
  mut hold: Local<HoldRepeat>,
  mut rng: Local<Rng>,
) {
  let (Some(mut scene), Some(mut phys)) = (scene, phys) else { return };
  if *mode != CameraMode::Fly
    || settings.target != EditTarget::Object
    || scene.demo_force_full_rebuild
  {
    return;
  }
  if captured.0 || intercepted.0 {
    return;
  }
  let dt = time.delta_secs();
  let placing = mouse.pressed(MouseButton::Left);
  let deleting =
    hold.erase_tick(mouse.just_pressed(MouseButton::Right), mouse.pressed(MouseButton::Right), dt);
  if !placing && !deleting {
    return;
  }
  let Some((origin, dir)) = view_ray(&windows, &cfg, lock.0) else { return };
  if deleting {
    delete_pointed(&mut scene, &mut phys, origin, dir);
    return;
  }
  fire(&mut scene, &mut phys, origin, dir, burst(&settings, &mut rng));
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn pile_fire(
  mut started: Local<bool>,
  mut fired: Local<usize>,
  mut aiming: Local<i32>,
  mut rng: Local<Rng>,
  cam: Res<DdaCameraConfig>,
  settings: Res<EditSettings>,
  mut fly: ResMut<FlyCamera>,
  mut orbit: ResMut<OrbitCamera>,
  scene: Option<ResMut<VoxelScene>>,
  phys: Option<ResMut<PhysicsState>>,
) {
  if !crate::consts::phys_pile() {
    return;
  }
  let (Some(mut scene), Some(mut phys)) = (scene, phys) else { return };
  if !*started {
    *started = true;
    let ground = ground_below(&scene, fly.pos);
    fly.pos.y = ground + PILE_ALTITUDE;
    orbit.pitch = PILE_PITCH;
    orbit.clamp();
    info!(
      target: "gate",
      "PHYS[pile] 升到地面以上 {PILE_ALTITUDE:.0} vx（地面 {ground:.0} → 机位 y {:.0}）、\
       俯角 {:.0}°，笔触按随机，连发 {} 件",
      fly.pos.y,
      PILE_PITCH.to_degrees(),
      crate::consts::phys_pile_count()
    );
    return;
  }
  if *fired == 0
    && *aiming <= PILE_AIM_MAX
    && !raycast(&scene.volumes, cam.position_world, cam.forward, PILE_REACH)
      .is_some_and(|h| h.normal.y > 0.5 && (h.t * cam.forward.x).hypot(h.t * cam.forward.z) <= PILE_LANDING)
  {
    if *aiming == PILE_AIM_MAX {
      warn!(target: "gate", "PHYS[pile] 视线始终未落到实地 → 按当前俯角开火");
    } else {
      orbit.pitch = (orbit.pitch + PILE_AIM_STEP).min(PILE_PITCH_MAX);
      orbit.clamp();
      if *aiming == 0 {
        info!(target: "gate", "PHYS[pile] 视线落空 → 逐步调陡俯角寻找实地");
      }
    }
    *aiming += 1;
    return;
  }
  if *fired >= crate::consts::phys_pile_count() {
    return;
  }
  let random = EditSettings { shape: BrushShape::Random, ..*settings };
  fire(&mut scene, &mut phys, cam.position_world, cam.forward, burst(&random, &mut rng));
  *fired += 1;
  if *fired == crate::consts::phys_pile_count() {
    info!(target: "gate", "PHYS[pile] 连发完成 {} 件", crate::consts::phys_pile_count());
  }
}

fn ground_below(scene: &VoxelScene, from: Vec3) -> f32 {
  let top = Vec3::new(from.x, from.y + PILE_PROBE_UP, from.z);
  raycast(&scene.volumes, top, Vec3::NEG_Y, PILE_PROBE_UP + PILE_PROBE_DOWN)
    .map_or(from.y, |h| h.voxel.y as f32 + 1.0)
}

pub(crate) fn register_callbacks(world: &mut World) {
  world.add_observer(
    |ev: On<MenuActionEvent>,
     mut settings: ResMut<EditSettings>,
     scene: Option<ResMut<VoxelScene>>,
     mut phys: Option<ResMut<PhysicsState>>| {
      match (ev.path.as_str(), &ev.action) {
        ("game/edit/place/target", MenuAction::Select(i)) => {
          settings.target = if *i == 0 { EditTarget::Object } else { EditTarget::World };
          info!(target: "gate", "交互对象 → {:?}", settings.target);
        }
        ("game/edit/place/clear", MenuAction::Button(0)) => {
          let Some(mut scene) = scene else { return };
          let n = scene.volumes.despawn_all_objects();
          let bodies = phys.as_mut().map_or(0, |p| p.clear_object_bodies());
          info!(target: "gate", "OBJECT[delete] 全部 n={n} 刚体={bodies}");
        }
        _ => {}
      }
    },
  );
}

pub(crate) fn sync_objects_menu(
  windows: Query<&Window>,
  cfg: Res<DdaCameraConfig>,
  lock: Res<MouseLock>,
  scene: Option<Res<VoxelScene>>,
  mut q_values: Query<(&gate_ui::MenuTextValue, &mut Text)>,
  mut diag: Local<(f64, u32)>,
) {
  let Some(scene) = scene else { return };
  if !q_values.iter().any(|(v, _)| v.path.starts_with("game/edit/place/")) {
    return;
  }
  let _t = gate_render::profiler::SysTimer::new("OBJECT 悬停", &mut diag);
  let count = scene.volumes.live_object_count();
  let pointed = view_ray(&windows, &cfg, lock.0)
    .and_then(|(origin, dir)| raycast_objects(&scene.volumes, origin, dir, DELETE_REACH))
    .map_or(-1, |h| h.obj_id);
  let count_text = format!("{count}");
  let pointed_text = if pointed < 0 { "—".to_string() } else { format!("obj {pointed}") };
  for (value, mut text) in &mut q_values {
    let s = match value.path.as_str() {
      "game/edit/place/count" => &count_text,
      "game/edit/place/hover" => &pointed_text,
      _ => continue,
    };
    if text.0 != *s {
      text.0 = s.clone();
    }
  }
}
