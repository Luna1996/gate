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

#[test]
fn adding_volumes_keeps_existing_slots() {
  let budget = 256 * 1024 * 1024;
  let seg = trace_consts().seg_words;

  let mut main = VolumeGrid::new();
  fill(&mut main, 0);
  main.compact_all();

  let mut volumes = Volumes::new(main);
  for seed in 0..3 {
    volumes.list.push(object(seed));
  }

  let mut vb = VolumesBuilder::build_full(&volumes, budget);
  let first = vb.snapshot();
  assert_eq!(first.mode_tag, "full");
  let bases: Vec<u32> = first.grid_descs.iter().map(|d| d.tree_base).collect();
  let palettes: Vec<u32> = first.grid_descs.iter().map(|d| d.palette_base).collect();

  assert_eq!(bases[0], 0, "世界体应落在第 0 页起点");
  for (i, &b) in bases.iter().enumerate().skip(1) {
    assert!(b >= seg, "v{i} 应落在第 1 页（base={b} < seg={seg}）");
  }

  for seed in 3..6 {
    volumes.list.push(object(seed));
  }
  vb.sync(&volumes);
  let second = vb.snapshot();
  assert_eq!(second.mode_tag, "incremental", "新增物体不应触发全量快照");

  let bases2: Vec<u32> = second.grid_descs.iter().map(|d| d.tree_base).collect();
  let palettes2: Vec<u32> = second.grid_descs.iter().map(|d| d.palette_base).collect();
  assert_eq!(bases2.len(), bases.len() + 3, "新增体块数不对");
  assert_eq!(bases, bases2[..bases.len()], "新增物体推动了已有体块的树区 base");
  assert_eq!(palettes, palettes2[..palettes.len()], "新增物体推动了已有体块的调色板 base");

  let tx: usize =
    second.struct_blobs.iter().chain(second.struct_blobs_p1.iter()).map(|(_, b)| b.len()).sum();
  assert!(tx < 4 * 1024 * 1024, "增量上传 {tx} B 太大：新增物体仍在整块重发树区");

  let limit = 2_147_483_644usize;
  assert!(second.struct_total_bytes <= limit, "第 0 页越界");
  assert!(second.struct_total_bytes_p1 <= limit, "第 1 页越界");
}
