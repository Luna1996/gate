use gate_physics::{PhysicsWorld, StepConfig, tight_bounds};
use gate_voxel::{PaletteEntry, PaletteId, VolumeGrid, VolumeTransform, Volumes, VoxelCoord};
use glam::{IVec3, Mat3, Vec3};

const EDGE: i32 = 16;
const DT: f32 = 1.0 / 60.0;

fn entry() -> PaletteEntry {
  PaletteEntry { color: [200, 200, 200], ..Default::default() }
}

fn ground_grid() -> VolumeGrid {
  let mut g = VolumeGrid::new();
  g.palette_mut().set(PaletteId(1), entry());
  gate_voxel::fill_box(&mut g, IVec3::ZERO, IVec3::new(64, 8, 64), PaletteId(1));
  g.compact_all();
  g
}

fn box_grid() -> VolumeGrid {
  let mut g = VolumeGrid::new_object(0, Vec3::ZERO, Mat3::IDENTITY, 1.0);
  g.palette_mut().set(PaletteId(1), entry());
  gate_voxel::fill_box(&mut g, IVec3::ZERO, IVec3::splat(EDGE), PaletteId(1));
  g.compact_all();
  g
}

fn spawn_box_dims(volumes: &mut Volumes, world: &mut PhysicsWorld, at: Vec3, dims: IVec3) -> usize {
  let mut g = VolumeGrid::new_object(0, Vec3::ZERO, Mat3::IDENTITY, 1.0);
  g.palette_mut().set(PaletteId(1), entry());
  gate_voxel::fill_box(&mut g, IVec3::ZERO, dims, PaletteId(1));
  g.compact_all();
  let props = gate_physics::mass_properties(&g, 1.0).unwrap();
  let bounds = tight_bounds(&g).unwrap();
  let index = volumes.list.len();
  volumes.list.push(g);
  world.add_body(
    &volumes.list[index],
    props,
    bounds,
    index,
    VolumeTransform::new(at, Mat3::IDENTITY, 1.0),
  )
}

fn spawn_box_at(volumes: &mut Volumes, world: &mut PhysicsWorld, at: Vec3, edge: i32) -> usize {
  spawn_box_dims(volumes, world, at, IVec3::splat(edge))
}

fn spawn_box(volumes: &mut Volumes, world: &mut PhysicsWorld, at: Vec3) -> usize {
  spawn_box_at(volumes, world, at, EDGE)
}

#[test]
fn box_inertia_matches_analytic_cube() {
  let p = gate_physics::mass_properties(&box_grid(), 1.0).unwrap();
  assert_eq!(p.mass, (EDGE * EDGE * EDGE) as f32);
  assert!((p.com - Vec3::splat(EDGE as f32 * 0.5)).length() < 1e-3, "质心 {:?}", p.com);
  let side = EDGE as f32;
  let expect = p.mass * (side * side * 2.0) / 12.0;
  let m = p.inertia.to_cols_array();
  for i in [0usize, 4, 8] {
    assert!((m[i] - expect).abs() < expect * 2e-3, "对角 {i} = {}，期望 {expect}", m[i]);
  }
  for i in [1usize, 2, 3, 5, 6, 7] {
    assert!(m[i].abs() < expect * 2e-3, "非对角应为 0：{i} = {}", m[i]);
  }
}

#[test]
fn box_dropped_on_ground_comes_to_rest() {
  let mut volumes = Volumes::new(ground_grid());
  let mut world = PhysicsWorld::new();
  world.init_static_world(&volumes.list[0], tight_bounds(&volumes.list[0]).unwrap(), 0);
  let b = spawn_box(&mut volumes, &mut world, Vec3::new(24.0, 32.0, 24.0));
  let cfg = StepConfig::default();
  for _ in 0..180 {
    world.step(&volumes, DT, &cfg);
  }
  let pose = world.bodies.field_transform(b);
  let v = world.bodies.lin_vel[b].length();
  assert!(v < 2.0, "未静止，速度 {v}");
  assert!(pose.pos.y > 7.0 && pose.pos.y < 8.3, "底部应停在地面顶面附近，实为 {:.3}", pose.pos.y);
  assert!(
    (pose.pos.x - 24.0).abs() < 1.0 && (pose.pos.z - 24.0).abs() < 1.0,
    "水平不应漂移：{:?}",
    pose.pos
  );
  assert!(world.bodies.ang_vel[b].length() < 0.5, "角速度 {:.3}", world.bodies.ang_vel[b].length());
}

#[test]
fn settled_box_dissolves_into_main_grid() {
  let mut volumes = Volumes::new(ground_grid());
  let mut world = PhysicsWorld::new();
  world.init_static_world(&volumes.list[0], tight_bounds(&volumes.list[0]).unwrap(), 0);
  let b = spawn_box(&mut volumes, &mut world, Vec3::new(24.0, 32.0, 24.0));
  let cfg = StepConfig::default();
  for _ in 0..600 {
    world.step(&volumes, DT, &cfg);
  }
  assert!(world.bodies.sleeping[b], "落定后应休眠");

  let grid_index = world.bodies.grid_index[b];
  let tr = world.bodies.field_transform(b);
  let bodies_before = world.bodies.len();
  let eye = tr.pos + Vec3::new(0.0, 40.0, 120.0);

  let facing = world.settled_for_dissolve(&volumes, eye, (tr.pos - eye).normalize(), 0.1, 0.5);
  assert!(facing.is_empty(), "相机正对时不该回收：{facing:?}");

  let away = world.settled_for_dissolve(&volumes, eye, (eye - tr.pos).normalize(), 0.1, 0.5);
  assert_eq!(away, vec![b], "背对且已睡够的体应入选");

  let written = world.dissolve_body(b, &mut volumes);
  assert!(written > 0, "融回应写入体素");
  assert_eq!(world.bodies.len(), bodies_before - 1, "体数应减一");
  assert_eq!(volumes.list[grid_index].chunk_count(), 0, "物体体积应清空");

  let probe = (tr.pos + Vec3::splat(EDGE as f32 * 0.5)).as_ivec3();
  assert!(
    volumes.main().get_voxel(VoxelCoord::from_ivec3(probe)).is_some(),
    "主网格应出现该物体的体素 @{probe:?}"
  );
}

#[test]
fn two_boxes_touching_are_quiet() {
  for overlap in [0.0f32, 0.3] {
    let mut volumes = Volumes::new(VolumeGrid::new());
    let mut world = PhysicsWorld::new();
    let a = spawn_box(&mut volumes, &mut world, Vec3::ZERO);
    let b = spawn_box(&mut volumes, &mut world, Vec3::new(EDGE as f32 - overlap, 0.0, 0.0));
    let cfg = StepConfig { gravity: Vec3::ZERO, ..StepConfig::default() };
    for _ in 0..60 {
      world.step(&volumes, DT, &cfg);
    }
    let pa = world.bodies.field_transform(a).pos;
    let pb = world.bodies.field_transform(b).pos;
    assert!(world.bodies.lin_vel[a].length() < 1.0, "a 被弹飞 {overlap}");
    assert!(world.bodies.lin_vel[b].length() < 1.0, "b 被弹飞 {overlap}");
    assert!((pa.y).abs() < 0.05 && (pa.z).abs() < 0.05, "a 侧向漂移 {pa:?}");
    assert!((pb.y).abs() < 0.05 && (pb.z).abs() < 0.05, "b 侧向漂移 {pb:?}");
    assert!((pb.x - (EDGE as f32 - overlap)).abs() < 0.3, "分离距离不对 {}", pb.x - pa.x);
  }
}

#[test]
fn deeply_overlapped_boxes_separate() {
  for overlap in [4.0f32, 8.0, 12.0] {
    let mut volumes = Volumes::new(VolumeGrid::new());
    let mut world = PhysicsWorld::new();
    let a = spawn_box(&mut volumes, &mut world, Vec3::ZERO);
    let b = spawn_box(&mut volumes, &mut world, Vec3::new(EDGE as f32 - overlap, 0.0, 0.0));
    let cfg = StepConfig { gravity: Vec3::ZERO, ..StepConfig::default() };
    for _ in 0..300 {
      world.step(&volumes, DT, &cfg);
    }
    let gap = world.bodies.field_transform(b).pos.x - world.bodies.field_transform(a).pos.x;
    assert!(
      gap > EDGE as f32 - 4.0,
      "初始重叠 {overlap} 的两箱必须自行分开，实测间距 {gap:.2}（初始 {:.1}）",
      EDGE as f32 - overlap
    );
  }
}

#[test]
fn dropped_box_bounces_before_resting() {
  let mut volumes = Volumes::new(ground_grid());
  let mut world = PhysicsWorld::new();
  world.init_static_world(&volumes.list[0], tight_bounds(&volumes.list[0]).unwrap(), 0);
  let b = spawn_box(&mut volumes, &mut world, Vec3::new(24.0, 40.0, 24.0));
  let cfg = StepConfig::default();
  let mut touched = false;
  let mut peak = 0.0f32;
  for _ in 0..240 {
    world.step(&volumes, DT, &cfg);
    let y = world.bodies.field_transform(b).pos.y;
    if y < 9.0 {
      touched = true;
    }
    if touched {
      peak = peak.max(y);
    }
  }
  let rest = world.bodies.field_transform(b).pos.y;
  assert!(peak > rest + 0.3, "落体应先回弹再静止：回弹峰值 {peak:.2}，静息 {rest:.2}");
}

fn stack_ids(volumes: &mut Volumes, world: &mut PhysicsWorld, layers: i32) -> Vec<usize> {
  (0..layers)
    .map(|k| {
      let y = 8.0 + k as f32 * (EDGE as f32 + 1.0);
      spawn_box(volumes, world, Vec3::new(24.0, y, 24.0))
    })
    .collect()
}

#[test]
fn box_settles_then_sleeps() {
  let mut volumes = Volumes::new(ground_grid());
  let mut world = PhysicsWorld::new();
  world.init_static_world(&volumes.list[0], tight_bounds(&volumes.list[0]).unwrap(), 0);
  let b = spawn_box(&mut volumes, &mut world, Vec3::new(24.0, 16.0, 24.0));
  let cfg = StepConfig::default();
  for _ in 0..180 {
    world.step(&volumes, DT, &cfg);
  }
  assert!(world.bodies.sleeping[b], "落地静置后应休眠，残余速度 {:?}", world.bodies.lin_vel[b]);
  assert_eq!(world.bodies.lin_vel[b], Vec3::ZERO);
  let before = world.bodies.field_transform(b).pos;
  for _ in 0..120 {
    world.step(&volumes, DT, &cfg);
  }
  assert_eq!(world.bodies.field_transform(b).pos, before, "休眠后位置不应漂移");
}

#[test]
fn dropped_box_wakes_a_sleeping_box() {
  let mut volumes = Volumes::new(ground_grid());
  let mut world = PhysicsWorld::new();
  world.init_static_world(&volumes.list[0], tight_bounds(&volumes.list[0]).unwrap(), 0);
  let b = spawn_box(&mut volumes, &mut world, Vec3::new(24.0, 16.0, 24.0));
  let cfg = StepConfig::default();
  for _ in 0..180 {
    world.step(&volumes, DT, &cfg);
  }
  assert!(world.bodies.sleeping[b], "前置条件：下方箱子已休眠");
  spawn_box(&mut volumes, &mut world, Vec3::new(24.0, 8.0 + EDGE as f32 + 4.0, 24.0));
  for _ in 0..20 {
    world.step(&volumes, DT, &cfg);
  }
  assert!(!world.bodies.sleeping[b], "落体接触时必须唤醒下方箱子");
}

#[test]
fn six_layer_stack_stays_stacked() {
  let mut volumes = Volumes::new(ground_grid());
  let mut world = PhysicsWorld::new();
  world.init_static_world(&volumes.list[0], tight_bounds(&volumes.list[0]).unwrap(), 0);
  let ids = stack_ids(&mut volumes, &mut world, 6);
  let cfg = StepConfig::default();
  for _ in 0..900 {
    world.step(&volumes, DT, &cfg);
  }
  for (k, &i) in ids.iter().enumerate() {
    let want = 8.0 + k as f32 * EDGE as f32;
    let y = world.bodies.field_transform(i).pos.y;
    assert!((y - want).abs() < 0.8 + k as f32 * 0.8, "第 {k} 层底部 {y:.3}，期望 ~{want}");
    assert!(world.bodies.sleeping[i], "第 {k} 层未休眠，残余 {:?}", world.bodies.lin_vel[i]);
  }
}

fn lever(
  volumes: &mut Volumes,
  world: &mut PhysicsWorld,
  weights: &[(f32, i32)],
) -> (usize, Vec<usize>) {
  let beam_len = 48.0;
  let center_x = 28.0;
  spawn_box_dims(volumes, world, Vec3::new(center_x - 2.0, 8.0, 24.0), IVec3::new(4, 8, 8));
  let beam = spawn_box_dims(
    volumes,
    world,
    Vec3::new(center_x - beam_len * 0.5, 16.0, 24.0),
    IVec3::new(48, 4, 8),
  );
  let ids = weights
    .iter()
    .map(|&(dx, edge)| {
      let e = edge as f32;
      spawn_box_dims(
        volumes,
        world,
        Vec3::new(center_x + dx - e * 0.5, 20.0, 28.0 - e * 0.5),
        IVec3::splat(edge),
      )
    })
    .collect();
  (beam, ids)
}

#[test]
fn balanced_beam_stays_level() {
  let mut volumes = Volumes::new(ground_grid());
  let mut world = PhysicsWorld::new();
  world.init_static_world(&volumes.list[0], tight_bounds(&volumes.list[0]).unwrap(), 0);
  let (beam, _) = lever(&mut volumes, &mut world, &[(-16.0, 12), (16.0, 12)]);
  let cfg = StepConfig::default();
  for _ in 0..600 {
    world.step(&volumes, DT, &cfg);
  }
  let tilt = world.bodies.rot[beam].x_axis.y;
  assert!(tilt.abs() < 0.06, "对称配重下横梁应保持水平，实测倾角分量 {tilt:.3}");
}

#[test]
fn lever_tips_toward_the_weight() {
  let mut volumes = Volumes::new(ground_grid());
  let mut world = PhysicsWorld::new();
  world.init_static_world(&volumes.list[0], tight_bounds(&volumes.list[0]).unwrap(), 0);
  let (beam, _) = lever(&mut volumes, &mut world, &[(16.0, 16)]);
  let cfg = StepConfig::default();
  let mut most = 0.0f32;
  for _ in 0..300 {
    world.step(&volumes, DT, &cfg);
    most = most.min(world.bodies.rot[beam].x_axis.y);
  }
  assert!(most < -0.2, "加配重那头（+x）应下沉，过程最大倾角分量只有 {most:.3}");
  let com = world.bodies.pos[beam];
  assert!((com.x - 28.0).abs() < 12.0 && (6.0..22.0).contains(&com.y), "横梁飞了：质心 {com:?}");
  let bw = &world.bodies;
  let mut low = f32::MAX;
  for i in 0..8 {
    let l = Vec3::new(
      if i & 1 == 0 { 0.0 } else { 48.0 },
      if i & 2 == 0 { 0.0 } else { 4.0 },
      if i & 4 == 0 { 0.0 } else { 8.0 },
    );
    low = low.min((bw.rot[beam] * l + bw.field_transform(beam).pos).y);
  }
  assert!(low > 7.0, "横梁扎进地面：最低角点 y={low:.2}");
}

#[test]
fn heavy_box_on_light_box_holds() {
  let mut volumes = Volumes::new(ground_grid());
  let mut world = PhysicsWorld::new();
  world.init_static_world(&volumes.list[0], tight_bounds(&volumes.list[0]).unwrap(), 0);
  let small = spawn_box_dims(&mut volumes, &mut world, Vec3::new(24.0, 8.0, 24.0), IVec3::splat(8));
  let big = spawn_box_dims(&mut volumes, &mut world, Vec3::new(20.0, 20.0, 20.0), IVec3::splat(16));
  let cfg = StepConfig::default();
  for _ in 0..600 {
    world.step(&volumes, DT, &cfg);
  }
  let y = world.bodies.field_transform(big).pos.y;
  assert!((15.0..=17.5).contains(&y), "大箱底部 {y:.3}，应压在小箱顶面 ~16");
  assert!(
    world.bodies.lin_vel[big].length() < 1.0,
    "大箱未落定，残余 {:?}",
    world.bodies.lin_vel[big]
  );
  assert!(world.bodies.sleeping[big], "大箱未休眠");
  assert!(world.bodies.sleeping[small], "小箱未休眠");
}

#[ignore = "参数扫描，按需运行"]
#[test]
fn sweep_substeps_and_iterations() {
  const WANT: [f32; 3] = [8.0, 24.0, 40.0];
  const CASES: [(u32, u32); 7] = [(4, 16), (8, 8), (16, 4), (32, 2), (64, 1), (16, 1), (4, 1)];
  eprintln!("子步 迭代 扫掠 |  层0压缩  层1压缩  层2压缩 | 顶层|v|  休眠  休眠帧  耗时ms");
  for (substeps, iterations) in CASES {
    let mut volumes = Volumes::new(ground_grid());
    let mut world = PhysicsWorld::new();
    world.init_static_world(&volumes.list[0], tight_bounds(&volumes.list[0]).unwrap(), 0);
    let ids = stack_ids(&mut volumes, &mut world, 3);
    let cfg = StepConfig { substeps, iterations, ..StepConfig::default() };
    let t0 = std::time::Instant::now();
    let mut sleep_frame = None;
    for f in 0..900 {
      world.step(&volumes, DT, &cfg);
      if sleep_frame.is_none() && ids.iter().all(|&i| world.bodies.sleeping[i]) {
        sleep_frame = Some(f);
      }
    }
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    let ys: Vec<f32> = ids.iter().map(|&i| world.bodies.field_transform(i).pos.y).collect();
    let comp: Vec<f32> = (0..3).map(|k| WANT[k] - ys[k]).collect();
    let v = world.bodies.lin_vel[ids[2]].length();
    let asleep = ids.iter().filter(|&&i| world.bodies.sleeping[i]).count();
    eprintln!(
      "{substeps:<4} {iterations:<4} {:>4} | {:>8.2} {:>8.2} {:>8.2} | {v:>7.2}  {asleep}/3  {:>6}  {ms:>7.0}",
      substeps * iterations,
      comp[0],
      comp[1],
      comp[2],
      sleep_frame.map_or("未睡".to_string(), |f| f.to_string())
    );
  }
}

#[test]
fn three_box_stack_stays_stacked() {
  let mut volumes = Volumes::new(ground_grid());
  let mut world = PhysicsWorld::new();
  world.init_static_world(&volumes.list[0], tight_bounds(&volumes.list[0]).unwrap(), 0);
  let ids = stack_ids(&mut volumes, &mut world, 3);
  let cfg = StepConfig::default();
  for _ in 0..900 {
    world.step(&volumes, DT, &cfg);
  }
  for (k, &i) in ids.iter().enumerate() {
    let want = 8.0 + k as f32 * EDGE as f32;
    let y = world.bodies.field_transform(i).pos.y;
    assert!((y - want).abs() < 1.0, "第 {k} 层底部 {y:.3}，期望 ~{want}（层间不应塌陷或飞起）");
    assert!(world.bodies.sleeping[i], "第 {k} 层未进入休眠，残余 {:?}", world.bodies.lin_vel[i]);
  }
}

struct Rng(u64);

impl Rng {
  fn next_u64(&mut self) -> u64 {
    self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = self.0;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
  }

  fn unit(&mut self) -> f32 {
    (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
  }

  fn range(&mut self, lo: f32, hi: f32) -> f32 {
    lo + (hi - lo) * self.unit()
  }

  fn int(&mut self, lo: i32, hi: i32) -> i32 {
    lo + (self.next_u64() % (hi - lo + 1) as u64) as i32
  }
}

fn blob_grid(rng: &mut Rng, size: i32) -> VolumeGrid {
  let mut g = VolumeGrid::new_object(0, Vec3::ZERO, Mat3::IDENTITY, 1.0);
  g.palette_mut().set(PaletteId(1), entry());
  let c = IVec3::splat(size / 2);
  for _ in 0..rng.int(2, 4) {
    let lo = IVec3::new(rng.int(0, c.x), rng.int(0, c.y), rng.int(0, c.z));
    let hi = IVec3::new(rng.int(c.x, size - 1), rng.int(c.y, size - 1), rng.int(c.z, size - 1));
    gate_voxel::fill_box(&mut g, lo, hi + IVec3::ONE, PaletteId(1));
  }
  g.compact_all();
  g
}

#[ignore = "基准测试，按需运行"]
#[test]
fn bench_pile() {
  const COUNT: usize = 1000;
  const COLS: i32 = 8;
  const SPACING: f32 = 16.0;
  const FRAMES: usize = 240;
  const PER_LAYER: usize = (COLS * COLS) as usize;
  let span = (COLS as f32 * SPACING) as i32 + 32;
  let cores = std::thread::available_parallelism().map_or(1, |n| n.get());

  let scene = |threads: usize| {
    let mut rng = Rng(0x5EED_5EED_5EED_5EED);
    let mut volumes = Volumes::new(VolumeGrid::new());
    {
      let g = &mut volumes.list[0];
      g.palette_mut().set(PaletteId(1), entry());
      gate_voxel::fill_box(g, IVec3::ZERO, IVec3::new(span, 16, span), PaletteId(1));
      g.compact_all();
    }
    let mut world = PhysicsWorld::with_threads(threads);
    world.init_static_world(&volumes.list[0], tight_bounds(&volumes.list[0]).unwrap(), 0);
    for i in 0..COUNT {
      let size = rng.int(8, 14);
      let g = blob_grid(&mut rng, size);
      let props = gate_physics::mass_properties(&g, 1.0).unwrap();
      let bounds = tight_bounds(&g).unwrap();
      let index = volumes.list.len();
      volumes.list.push(g);
      let at = Vec3::new(
        8.0 + (i % COLS as usize) as f32 * SPACING + rng.range(-3.0, 3.0),
        24.0 + (i / PER_LAYER) as f32 * SPACING,
        8.0 + ((i / COLS as usize) % COLS as usize) as f32 * SPACING + rng.range(-3.0, 3.0),
      );
      let axis = Vec3::new(rng.range(-1.0, 1.0), rng.range(-1.0, 1.0), rng.range(-1.0, 1.0));
      let rot =
        Mat3::from_axis_angle(axis.normalize_or_zero(), rng.range(0.0, std::f32::consts::TAU));
      world.add_body(
        &volumes.list[index],
        props,
        bounds,
        index,
        VolumeTransform::new(at, rot, 1.0),
      );
    }
    (volumes, world)
  };

  let cfg = StepConfig { profile: true, ..StepConfig::default() };
  let mut serial_ms = 0.0f64;
  for threads in [0usize, cores.saturating_sub(1)] {
    let (volumes, mut world) = scene(threads);
    let bodies = world.bodies.len() - 1;
    let mut peak_contacts = 0usize;
    let mut prof = gate_physics::StepProfile::default();
    let mut marks = Vec::new();
    let t0 = std::time::Instant::now();
    for f in 0..FRAMES {
      let s = world.step(&volumes, DT, &cfg);
      peak_contacts = peak_contacts.max(s.contacts);
      if (1..=3).contains(&(f * 4 / FRAMES)) && marks.len() < f * 4 / FRAMES {
        marks.push((0..world.bodies.len()).filter(|&i| world.bodies.sleeping[i]).count() - 1);
      }
      let p = s.profile;
      prof.broad_ms += p.broad_ms;
      prof.pair_ms += p.pair_ms;
      prof.build_ms += p.build_ms;
      prof.setup_ms += p.setup_ms;
      prof.color_ms += p.color_ms;
      prof.prepare_ms += p.prepare_ms;
      prof.solve_ms += p.solve_ms;
      prof.integrate_ms += p.integrate_ms;
      prof.position_ms += p.position_ms;
      prof.carry_ms += p.carry_ms;
      prof.sleep_ms += p.sleep_ms;
    }
    let ms = t0.elapsed().as_secs_f64() * 1000.0 / FRAMES as f64;
    if threads == 0 {
      serial_ms = ms;
    }
    let asleep = (0..world.bodies.len()).filter(|&i| world.bodies.sleeping[i]).count() - 1;
    let label =
      if threads == 0 { "串行".to_string() } else { format!("并行 {threads} 线程") };
    eprintln!(
      "{label}（核 {cores}）· {bodies} 体 {FRAMES} 帧 · {ms:.3} ms/帧 · 加速比 {:.1}× · 峰值接触 {peak_contacts} · 休眠 25%:{} 50%:{} 75%:{} 末期 {asleep}/{bodies}",
      if serial_ms > 0.0 { serial_ms / ms } else { 1.0 },
      marks.first().copied().unwrap_or(0),
      marks.get(1).copied().unwrap_or(0),
      marks.get(2).copied().unwrap_or(0),
    );
    eprintln!(
      "  分解 ms/帧：宽相 {:.3} · 配对 {:.3} · 窄相 {:.3} · 装配 {:.3} · 着色 {:.3} · 准备 {:.3} · 求解 {:.3} · 积分 {:.3} · 位修 {:.3} · 回写 {:.3} · 休眠 {:.3}  | 合计 {:.3}",
      prof.broad_ms / FRAMES as f32,
      prof.pair_ms / FRAMES as f32,
      prof.build_ms / FRAMES as f32,
      prof.setup_ms / FRAMES as f32,
      prof.color_ms / FRAMES as f32,
      prof.prepare_ms / FRAMES as f32,
      prof.solve_ms / FRAMES as f32,
      prof.integrate_ms / FRAMES as f32,
      prof.position_ms / FRAMES as f32,
      prof.carry_ms / FRAMES as f32,
      prof.sleep_ms / FRAMES as f32,
      prof.total_ms() / FRAMES as f32
    );
  }
}
