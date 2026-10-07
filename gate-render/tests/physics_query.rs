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

#[test]
fn physics_query_agrees_with_render_raycast() {
  let src = scene();
  let cases = [
    (Vec3::ZERO, Mat3::IDENTITY, 1.0f32),
    (Vec3::new(300.0, -120.0, 640.0), Mat3::IDENTITY, 1.0),
    (
      Vec3::new(-512.0, 256.0, 128.0),
      Mat3::from_rotation_y(0.7) * Mat3::from_rotation_x(-0.35),
      1.0,
    ),
    (Vec3::new(64.0, 64.0, 64.0), Mat3::from_rotation_z(1.2), 0.5),
  ];
  let mut rng = 0x1234_5678_9abc_def0u64;
  let (mut compared, mut hits, mut misses) = (0usize, 0usize, 0usize);
  for (pos, rot, scale) in cases {
    let mut obj = src.clone();
    obj.set_transform(pos, rot, scale);
    let mut volumes = Volumes::new(VolumeGrid::new());
    volumes.list.push(obj.clone());
    let field = gate_physics::Field::new(&obj);
    for _ in 0..500 {
      let aim = field.to_world(Vec3::new(
        lcg(&mut rng) * 128.0,
        lcg(&mut rng) * 128.0,
        lcg(&mut rng) * 128.0,
      ));
      let o = aim
        + Vec3::new(
          (lcg(&mut rng) - 0.5) * 1400.0,
          (lcg(&mut rng) - 0.5) * 1400.0,
          (lcg(&mut rng) - 0.5) * 1400.0,
        );
      let d = aim - o;
      if d.length_squared() <= 0.0 {
        continue;
      }
      let expect = gate_render::raycast(&volumes, o, d, 4096.0);
      let got =
        gate_physics::first_solid_local(&obj, field.to_local(o), field.dir_to_local(d), 4096.0);
      compared += 1;
      match (expect, got) {
        (None, None) => misses += 1,
        (Some(h), Some((v, t))) => {
          assert_eq!(
            h.voxel, v,
            "体素不一致 pos={pos:?} rot={rot:?} scale={scale} o={o:?} d={d:?}"
          );
          assert!(
            (h.t - t).abs() <= 1e-2 * (1.0 + h.t.abs()),
            "t 不一致 render={} physics={t}（pos={pos:?} scale={scale}）",
            h.t
          );
          hits += 1;
        }
        (a, b) => panic!(
          "命中结论不一致 pos={pos:?} scale={scale} o={o:?} d={d:?}：render={a:?} physics={b:?}"
        ),
      }
    }
  }
  assert!(compared >= 1900, "样本不足 compared={compared}");
  assert!(hits >= 300, "命中样本不足 hits={hits}");
  assert!(misses >= 100, "未命中样本不足 misses={misses}");
}
