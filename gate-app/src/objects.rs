use bevy::prelude::*;
use glam::{IVec3, Mat3, Vec3};

use gate_render::{DdaCameraConfig, VoxelScene, raycast_objects};
use gate_ui::{MenuAction, MenuActionEvent};
use gate_voxel::fill_box;

use crate::{
  camera::{CameraMode, MouseLock, cursor_ray},
  edit::{BrushMaterial, EditSettings, EditTarget, material_slot},
};

const EDGE_DEFAULT: i32 = 32;
const EDGE_MIN: i32 = 4;
const DIST_DEFAULT: f32 = 80.0;
const DELETE_REACH: f32 = 4096.0;

#[derive(Resource, Clone, Copy, Debug)]
pub(crate) struct ObjectSettings {
  pub edge: i32,
  pub dist: f32,
}

impl Default for ObjectSettings {
  fn default() -> Self {
    Self { edge: EDGE_DEFAULT, dist: DIST_DEFAULT }
  }
}

fn view_ray(windows: &Query<&Window>, cfg: &DdaCameraConfig, locked: bool) -> Option<(Vec3, Vec3)> {
  let window = windows.single().ok()?;
  cursor_ray(window, cfg, locked)
}

fn place(scene: &mut VoxelScene, center: Vec3, edge: i32, mat: BrushMaterial) -> Option<i32> {
  let edge = edge.max(EDGE_MIN);
  let pos = center - Vec3::splat(edge as f32 * 0.5);
  let obj_id = scene.volumes.spawn_object(pos, Mat3::IDENTITY, 1.0);
  let grid = scene.volumes.volume_mut(obj_id)?;
  let slot = material_slot(grid, mat);
  let written = fill_box(grid, IVec3::ZERO, IVec3::splat(edge), slot);
  bevy::log::info!(
    "OBJECT[place] obj={obj_id} @({:.1},{:.1},{:.1}) 边长{edge} slot={slot} {written}vx",
    pos.x,
    pos.y,
    pos.z
  );
  Some(obj_id)
}

fn delete_pointed(scene: &mut VoxelScene, origin: Vec3, dir: Vec3) {
  let Some(hit) = raycast_objects(&scene.volumes, origin, dir, DELETE_REACH) else {
    bevy::log::debug!("OBJECT[delete] 未命中物体");
    return;
  };
  if scene.volumes.despawn_object(hit.obj_id) {
    bevy::log::info!("OBJECT[delete] obj={}", hit.obj_id);
  }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn object_input(
  mouse: Res<ButtonInput<MouseButton>>,
  captured: Res<gate_ui::UiPointerCaptured>,
  intercepted: Res<gate_ui::MouseIntercepted>,
  windows: Query<&Window>,
  cfg: Res<DdaCameraConfig>,
  mode: Res<CameraMode>,
  lock: Res<MouseLock>,
  settings: Res<EditSettings>,
  obj: Res<ObjectSettings>,
  scene: Option<ResMut<VoxelScene>>,
) {
  let Some(mut scene) = scene else { return };
  if *mode != CameraMode::Fly
    || settings.target != EditTarget::Object
    || scene.demo_force_full_rebuild
  {
    return;
  }
  if captured.0 || intercepted.0 {
    return;
  }
  let placing = mouse.just_pressed(MouseButton::Left);
  let deleting = mouse.just_pressed(MouseButton::Right);
  if !placing && !deleting {
    return;
  }
  let Some((origin, dir)) = view_ray(&windows, &cfg, lock.0) else { return };
  if deleting {
    delete_pointed(&mut scene, origin, dir);
    return;
  }
  place(&mut scene, origin + dir * obj.dist, obj.edge, settings.mat);
}

pub(crate) fn register_callbacks(world: &mut World) {
  world.add_observer(
    |ev: On<MenuActionEvent>,
     mut settings: ResMut<EditSettings>,
     mut obj: ResMut<ObjectSettings>,
     scene: Option<ResMut<VoxelScene>>| {
      match (ev.path.as_str(), &ev.action) {
        ("game/objects/target", MenuAction::Select(i)) => {
          settings.target = if *i == 0 { EditTarget::Object } else { EditTarget::World };
          bevy::log::info!("交互对象 → {:?}", settings.target);
        }
        ("game/objects/edge", MenuAction::Value(v)) => {
          obj.edge = (*v).round() as i32;
          bevy::log::info!("OBJECT 生成边长 → {}", obj.edge.max(EDGE_MIN));
        }
        ("game/objects/dist", MenuAction::Value(v)) => {
          obj.dist = *v;
          bevy::log::info!("OBJECT 生成距离 → {:.0}vx", obj.dist);
        }
        ("game/objects/clear", MenuAction::Button(0)) => {
          let Some(mut scene) = scene else { return };
          let n = scene.volumes.despawn_all_objects();
          bevy::log::info!("OBJECT[delete] 全部 n={n}");
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
) {
  let Some(scene) = scene else { return };
  if !q_values.iter().any(|(v, _)| v.path.starts_with("game/objects/")) {
    return;
  }
  let count = scene.volumes.live_object_count();
  let pointed = view_ray(&windows, &cfg, lock.0)
    .and_then(|(origin, dir)| raycast_objects(&scene.volumes, origin, dir, DELETE_REACH))
    .map_or(-1, |h| h.obj_id);
  let count_text = format!("{count}");
  let pointed_text = if pointed < 0 { "—".to_string() } else { format!("obj {pointed}") };
  for (value, mut text) in &mut q_values {
    let s = match value.path.as_str() {
      "game/objects/count" => &count_text,
      "game/objects/hover" => &pointed_text,
      _ => continue,
    };
    if text.0 != *s {
      text.0 = s.clone();
    }
  }
}
