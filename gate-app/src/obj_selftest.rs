use bevy::prelude::*;
use glam::{IVec3, Vec3};

use crate::edit::{BrushShape, EditSettings, apply_brush, material_slot};
use gate_render::VoxelScene;

pub(crate) fn object_selftest(
  scene: Option<ResMut<VoxelScene>>,
  settings: Res<EditSettings>,
  cam: Res<gate_render::DdaCameraConfig>,
  time: Res<Time>,
  mut done: Local<u8>,
) {
  if *done > 1 || crate::consts::phys_pile() {
    return;
  }
  let t = time.elapsed().as_secs_f32();
  let Some(mut scene) = scene else { return };
  if *done == 0 && t > 70.0 {
    *done = 1;
    let fwd = cam.forward.normalize_or_zero();
    let pos: Vec3 = cam.position_world + fwd * 250.0 - Vec3::Y * 60.0;
    let obj_id = scene.volumes.spawn_object(pos, glam::Mat3::IDENTITY, 1.0);
    let written = {
      let grid = scene.volumes.volume_mut(obj_id).expect("新对象");
      let slot = material_slot(grid, settings.mat);
      apply_brush(grid, IVec3::splat(2), BrushShape::Capsule, 4, slot)
    };
    info!(
      target: "gate",
      "OBJ SELFTEST 生成 obj={obj_id} @({:.0},{:.0},{:.0}) 自发光={} 体素={written} t={t:.1}s",
      pos.x,
      pos.y,
      pos.z,
      settings.mat.emissive
    );
  } else if *done == 1 && t > 90.0 {
    *done = 2;
    let ids: Vec<i32> = scene.volumes.live_object_ids().collect();
    for id in &ids {
      scene.volumes.despawn_object(*id);
    }
    info!(target: "gate", "OBJ SELFTEST 销毁 {ids:?} t={t:.1}s");
  }
}
