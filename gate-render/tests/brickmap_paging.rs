use gate_render::brickmap::VolumesBuilder;
use gate_render::wesl_consts::trace_consts;
use gate_voxel::{PaletteEntry, PaletteId, VolumeGrid, Volumes};
use glam::{IVec3, Mat3, Vec3};

fn fill(grid: &mut VolumeGrid, seed: i32) {
  grid
    .palette_mut()
    .set(PaletteId(1), PaletteEntry { color: [200, 180, 160], ..Default::default() });
  let o = IVec3::splat(8);
  gate_voxel::fill_box(grid, o, IVec3::splat(48), PaletteId(1));
  gate_voxel::fill_sphere(grid, o + IVec3::new(40, 40, 40), 16 + seed, PaletteId(1));
}

fn object(seed: i32) -> VolumeGrid {
  let mut g =
    VolumeGrid::new_object(seed, Vec3::new(seed as f32 * 96.0, 0.0, 0.0), Mat3::IDENTITY, 1.0);
  fill(&mut g, seed);
  g.compact_all();
  g
}