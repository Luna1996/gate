use std::f32::consts::FRAC_PI_2;

use gate_voxel::{PaletteEntry, PaletteId, VolumeGrid};
use glam::{IVec3, Mat3, Vec3};

fn cube(edge: i32) -> VolumeGrid {
  let mut g = VolumeGrid::new_object(0, Vec3::ZERO, Mat3::IDENTITY, 1.0);
  g.palette_mut().set(PaletteId(1), PaletteEntry { color: [200, 200, 200], ..Default::default() });
  gate_voxel::fill_box(&mut g, IVec3::ZERO, IVec3::splat(edge), PaletteId(1));
  g.compact_all();
  g
}

fn deepest(m: &gate_physics::Manifold) -> f32 {
  m.points.iter().map(|c| c.depth).fold(0.0f32, f32::max)
}