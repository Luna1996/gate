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

#[test]
fn tight_bounds_match_fill() {
  let g = cube(8);
  assert_eq!(gate_physics::tight_bounds(&g), Some((IVec3::ZERO, IVec3::splat(7))));
  assert_eq!(gate_physics::tight_bounds(&VolumeGrid::new()), None);
  let field = gate_physics::Field::new(&g);
  assert_eq!(field.local_aabb(), Some((Vec3::ZERO, Vec3::splat(8.0))));
}

#[test]
fn aligned_overlap_depth_is_axis_gap() {
  let a_grid = cube(8);
  let mut b_grid = cube(8);
  b_grid.set_transform(Vec3::new(6.3, 0.0, 0.0), Mat3::IDENTITY, 1.0);
  let m = gate_physics::manifold(
    &gate_physics::Field::new(&a_grid),
    &gate_physics::Field::new(&b_grid),
    &gate_physics::ContactConfig::default(),
  );
  assert!(!m.points.is_empty(), "应产生接触");
  let d = deepest(&m);
  assert!((d - 0.7).abs() < 0.02, "最深穿透 {d}（期望 0.7）");
  for c in &m.points {
    assert!((c.normal.length() - 1.0).abs() < 1e-4, "法线未归一 {:?}", c.normal);
    assert!(c.depth >= 0.0);
    assert!(c.point.x > 6.0 && c.point.x < 8.1, "接触点越界 {:?}", c.point);
    assert!(c.point.y > 0.0 && c.point.y < 8.0, "接触点越界 {:?}", c.point);
    assert!(c.point.z > 0.0 && c.point.z < 8.0, "接触点越界 {:?}", c.point);
    if c.depth > 0.3 {
      assert!(c.normal.x.abs() > 0.99, "深接触法线应为 ±x：{:?} depth={}", c.normal, c.depth);
    }
  }
}

#[test]
fn stacked_boxes_only_have_vertical_normals() {
  for gap in [0.0f32, 0.2, 0.5, 1.0] {
    let a_grid = cube(16);
    let mut b_grid = cube(16);
    b_grid.set_transform(Vec3::new(0.0, 16.0 - gap, 0.0), Mat3::IDENTITY, 1.0);
    let m = gate_physics::manifold(
      &gate_physics::Field::new(&a_grid),
      &gate_physics::Field::new(&b_grid),
      &gate_physics::ContactConfig::default(),
    );
    let deep_lateral = m.points.iter().filter(|c| c.normal.y.abs() < 0.99 && c.depth > 0.1).count();
    let vertical = m.points.iter().filter(|c| c.normal.y.abs() > 0.99).count();
    assert_eq!(deep_lateral, 0, "gap={gap} 出现有深度的横向法线");
    assert!(vertical > 0, "gap={gap} 必须有竖向接触");
  }
}

#[test]
fn separated_bodies_have_no_contact() {
  let a_grid = cube(8);
  let mut b_grid = cube(8);
  b_grid.set_transform(Vec3::new(9.0, 0.0, 0.0), Mat3::IDENTITY, 1.0);
  let m = gate_physics::manifold(
    &gate_physics::Field::new(&a_grid),
    &gate_physics::Field::new(&b_grid),
    &gate_physics::ContactConfig::default(),
  );
  assert!(m.points.is_empty(), "分离时不应有接触：{:?}", m.points);
}

#[test]
fn edges_touching_give_vertical_normal() {
  let a_grid = cube(8);
  let mut b_grid = cube(8);
  b_grid.set_transform(Vec3::new(7.0, 8.0, 0.0), Mat3::IDENTITY, 1.0);
  let m = gate_physics::manifold(
    &gate_physics::Field::new(&a_grid),
    &gate_physics::Field::new(&b_grid),
    &gate_physics::ContactConfig::default(),
  );
  assert!(!m.points.is_empty(), "贴棱应有接触");
  for c in &m.points {
    assert!(c.normal.y.abs() > 0.99, "贴棱法线应竖直：{:?}", c.normal);
    assert!(c.depth < 0.02, "贴棱深度应近 0：{}", c.depth);
  }
}

#[test]
fn skewed_edges_pick_minimum_translation_axis() {
  let a_grid = cube(8);
  let mut b_grid = cube(8);
  b_grid.set_transform(Vec3::new(7.3, 7.6, 0.0), Mat3::IDENTITY, 1.0);
  let m = gate_physics::manifold(
    &gate_physics::Field::new(&a_grid),
    &gate_physics::Field::new(&b_grid),
    &gate_physics::ContactConfig::default(),
  );
  let deep = m.points.iter().max_by(|x, y| x.depth.total_cmp(&y.depth)).expect("应有接触");
  assert!(deep.normal.y.abs() > 0.99, "最小平移轴应为 ±y：{:?}", deep.normal);
  assert!((deep.depth - 0.4).abs() < 0.02, "深度应为 0.4，实测 {}", deep.depth);
}

#[test]
fn long_edge_line_is_sampled_along_its_length() {
  let a_grid = cube(24);
  let mut b_grid = cube(24);
  b_grid.set_transform(Vec3::new(23.5, 24.0, 0.0), Mat3::IDENTITY, 1.0);
  let m = gate_physics::manifold(
    &gate_physics::Field::new(&a_grid),
    &gate_physics::Field::new(&b_grid),
    &gate_physics::ContactConfig::default(),
  );
  assert!(m.points.len() >= 4, "24 长的棱应采到多个点，实测 {}", m.points.len());
  let lo = m.points.iter().map(|c| c.point.z).fold(f32::MAX, f32::min);
  let hi = m.points.iter().map(|c| c.point.z).fold(f32::MIN, f32::max);
  assert!(hi - lo > 12.0, "接触点应铺满棱线，实测跨度 {}", hi - lo);
}

#[test]
fn box_face_on_thin_rail_is_supported() {
  for (lo_z, hi_z) in [(0i32, 24i32), (10, 13)] {
    let mut rail = VolumeGrid::new();
    rail
      .palette_mut()
      .set(PaletteId(1), PaletteEntry { color: [200, 200, 200], ..Default::default() });
    gate_voxel::fill_box(&mut rail, IVec3::new(0, 0, lo_z), IVec3::new(1, 1, hi_z), PaletteId(1));
    rail.compact_all();
    let mut box_grid = cube(8);
    box_grid.set_transform(Vec3::new(-3.5, 1.0, 8.0), Mat3::IDENTITY, 1.0);
    let m = gate_physics::manifold(
      &gate_physics::Field::new(&box_grid),
      &gate_physics::Field::new(&rail),
      &gate_physics::ContactConfig::default(),
    );
    assert!(!m.points.is_empty(), "刃 z∈[{lo_z},{hi_z}] 应托住箱子");
    assert!(
      m.points.iter().any(|c| c.normal.y < -0.99 && c.depth < 0.02),
      "刃 z∈[{lo_z},{hi_z}] 应有向上的支撑接触"
    );
  }
}

#[test]
fn quarter_turn_matches_aligned_depth() {
  let a_grid = cube(8);
  let mut b_grid = cube(8);
  b_grid.set_transform(Vec3::new(6.3, 0.0, 0.0), Mat3::from_rotation_z(FRAC_PI_2), 1.0);
  let m = gate_physics::manifold(
    &gate_physics::Field::new(&a_grid),
    &gate_physics::Field::new(&b_grid),
    &gate_physics::ContactConfig::default(),
  );
  assert!(!m.points.is_empty(), "应产生接触");
  let d = deepest(&m);
  assert!((d - 0.7).abs() < 0.02, "最深穿透 {d}（期望 0.7）");
  let (amn, amx) = gate_physics::Field::new(&a_grid).world_aabb().unwrap();
  let (bmn, bmx) = gate_physics::Field::new(&b_grid).world_aabb().unwrap();
  let lo = amn.max(bmn) - Vec3::splat(0.5);
  let hi = amx.min(bmx) + Vec3::splat(0.5);
  for c in &m.points {
    assert!((c.normal.length() - 1.0).abs() < 1e-4, "法线未归一 {:?}", c.normal);
    assert!(c.depth >= 0.0);
    assert!(
      c.point.cmpge(lo).all() && c.point.cmple(hi).all(),
      "接触点应在两体重叠区内：{:?}",
      c.point
    );
  }
}

#[test]
fn tilted_bodies_produce_sane_manifold() {
  let a_grid = cube(8);
  let mut b_grid = cube(8);
  b_grid.set_transform(Vec3::new(6.0, 0.0, 0.0), Mat3::from_rotation_z(0.35), 1.0);
  let m = gate_physics::manifold(
    &gate_physics::Field::new(&a_grid),
    &gate_physics::Field::new(&b_grid),
    &gate_physics::ContactConfig::default(),
  );
  assert!(!m.points.is_empty(), "应产生接触");
  assert!(m.points.len() <= 96, "接触点数应受 max_points 限制：{}", m.points.len());
  for c in &m.points {
    assert!((c.normal.length() - 1.0).abs() < 1e-4, "法线未归一 {:?}", c.normal);
    assert!(c.depth >= 0.0);
    assert!(c.point.x > -1.0 && c.point.x < 15.0, "接触点越界 {:?}", c.point);
    assert!(c.point.y > -1.0 && c.point.y < 9.0, "接触点越界 {:?}", c.point);
  }
}

#[test]
fn accel_bitmap_matches_per_voxel_classification() {
  use gate_physics::{Field, VoxelClass, classify};

  let mut g = VolumeGrid::new();
  g.palette_mut().set(PaletteId(1), PaletteEntry { color: [200, 200, 200], ..Default::default() });
  for x in 0..24 {
    let h = 6 + (x % 5);
    gate_voxel::fill_box(&mut g, IVec3::new(x * 2, 0, 0), IVec3::new(2, h, 20), PaletteId(1));
  }
  gate_voxel::fill_box(&mut g, IVec3::new(4, 12, 6), IVec3::new(6, 3, 5), PaletteId(1));
  gate_voxel::fill_box(&mut g, IVec3::new(30, 4, 4), IVec3::splat(3), PaletteId(1));
  g.compact_all();
  let f = Field::new(&g);

  let (lo, hi) = (IVec3::new(-3, -3, -3), IVec3::new(52, 22, 26));
  let accel = gate_physics::ContactVoxels::build(&f, (lo, hi));

  let mut want_c = Vec::new();
  let mut want_e = Vec::new();
  for z in lo.z..=hi.z {
    for y in lo.y..=hi.y {
      for x in lo.x..=hi.x {
        let p = IVec3::new(x, y, z);
        if !f.solid_local(p) {
          continue;
        }
        match classify(&f, p).class() {
          VoxelClass::Corner => want_c.push(p),
          VoxelClass::Edge => want_e.push(p),
          _ => {}
        }
      }
    }
  }
  want_c.sort_unstable_by_key(|v| (v.x, v.y, v.z));
  want_e.sort_unstable_by_key(|v| (v.x, v.y, v.z));
  assert_eq!(accel.corners(), want_c.as_slice(), "角集合不一致");
  assert_eq!(accel.edges(), want_e.as_slice(), "棱集合不一致");
}
