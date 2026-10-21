use gate_voxel::{PaletteEntry, PaletteId, VolumeGrid, Volumes};
use glam::{IVec3, Mat3, Vec3};

fn scene() -> VolumeGrid {
  let mut g = VolumeGrid::new_object(0, Vec3::ZERO, Mat3::IDENTITY, 1.0);
  g.palette_mut().set(PaletteId(1), PaletteEntry { color: [200, 180, 160], ..Default::default() });
  gate_voxel::fill_box(&mut g, IVec3::new(8, 8, 56), IVec3::new(48, 48, 8), PaletteId(1));
  gate_voxel::fill_sphere(&mut g, IVec3::new(96, 40, 40), 18, PaletteId(1));
  for i in 0..40i32 {
    g.set_voxel_ivec3(IVec3::new(8 + i * 2, 120, 8 + i), PaletteId(1));
    g.set_voxel_ivec3(IVec3::new(200 - i, 200, 130 + (i % 5)), PaletteId(1));
  }
  g.compact_all();
  g
}

fn lcg(s: &mut u64) -> f32 {
  *s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
  ((*s >> 33) as u32) as f32 / (u32::MAX as f32)
}
