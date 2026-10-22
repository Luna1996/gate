use bevy::prelude::*;
use glam::IVec3;

use crate::edit::{BrushShape, EditSettings, apply_brush, material_slot};
use gate_render::VoxelScene;

pub(crate) fn object_selftest(
  mut scene: Option<ResMut<VoxelScene>>,
  settings: Res<EditSettings>,
  cam: Res<gate_render::DdaCameraConfig>,
  mut frame: Local<u32>,
) {
  *frame += 1;
  let n = *frame;
  let Some(mut scene) = scene else { return };
  if n == 3000 {
    let pos = cam.position_world + cam.forward * 24.0;
    let obj_id = scene.volumes.spawn_object(pos, glam::Mat3::IDENTITY, 1.0);
    let written = {
      let grid = scene.volumes.volume_mut(obj_id).expect("新对象");
      let slot = material_slot(grid, settings.mat);
      apply_brush(grid, IVec3::splat(2), BrushShape::Sphere, 5, slot)
    };
    info!(target: "gate", "OBJ SELFTEST 生成 obj={obj_id} 自发光={} 体素={written}", settings.mat.emissive);
  } else if n == 3600 {
    let ids: Vec<i32> = scene.volumes.live_object_ids().collect();
    for id in &ids {
      scene.volumes.despawn_object(*id);
    }
    info!(target: "gate", "OBJ SELFTEST 销毁 {ids:?}");
  }
}
