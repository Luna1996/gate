use gate_physics::{
  Field, PhysicsWorld, Probe, StepConfig, chunk_of, manifold_bounded, tight_bounds, world_aabb_of,
};
use gate_voxel::{PaletteEntry, PaletteId, VolumeGrid, VolumeTransform, Volumes};
use glam::{IVec3, Mat3, Vec3};

const DT: f32 = 1.0 / 60.0;
const EDGE: i32 = 16;

fn entry() -> PaletteEntry {
  PaletteEntry { color: [200, 200, 200], ..Default::default() }
}

fn flat_ground() -> VolumeGrid {
  let mut g = VolumeGrid::new();
  g.palette_mut().set(PaletteId(1), entry());
  gate_voxel::fill_box(&mut g, IVec3::ZERO, IVec3::new(64, 8, 64), PaletteId(1));
  g.compact_all();
  g
}

fn stepped_ground() -> VolumeGrid {
  let mut g = VolumeGrid::new();
  g.palette_mut().set(PaletteId(1), entry());
  for cx in 0..16 {
    let h = 8 + (cx % 3);
    gate_voxel::fill_box(&mut g, IVec3::new(cx * 4, 0, 0), IVec3::new(4, h, 64), PaletteId(1));
  }
  g.compact_all();
  g
}

fn spawn_dims(volumes: &mut Volumes, world: &mut PhysicsWorld, at: Vec3, dims: IVec3) -> usize {
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

fn run(name: &str, ground: VolumeGrid, at: Vec3, steps: usize) {
  let mut volumes = Volumes::new(ground);
  let mut world = PhysicsWorld::new();
  world.init_static_world(&volumes.list[0], tight_bounds(&volumes.list[0]).unwrap(), 0);
  let b = spawn_dims(&mut volumes, &mut world, at, IVec3::splat(EDGE));
  let cfg = StepConfig::default();
  let mut max_v = 0.0f32;
  let mut sleep_at = None;
  for f in 0..steps {
    world.step(&volumes, DT, &cfg);
    if f * 2 > steps {
      max_v = max_v.max(world.bodies.lin_vel[b].length());
    }
    if sleep_at.is_none() && world.bodies.sleeping[b] {
      sleep_at = Some(f);
    }
  }
  let p = world.bodies.field_transform(b).pos;
  eprintln!(
    "{name:<18} 末({:>7.1},{:>7.1},{:>7.1}) |v|max={max_v:>6.2} 睡={:?}",
    p.x, p.y, p.z, sleep_at
  );
}

fn dump_rest(name: &str, ground: VolumeGrid, at: Vec3) {
  let mut volumes = Volumes::new(ground);
  let mut world = PhysicsWorld::new();
  world.init_static_world(&volumes.list[0], tight_bounds(&volumes.list[0]).unwrap(), 0);
  let b = spawn_dims(&mut volumes, &mut world, at, IVec3::splat(EDGE));
  let bb = &world.bodies;
  let pa = Probe {
    field: Field::new_at(&volumes.list[bb.grid_index[b]], bb.field_transform(b)),
    vox: &bb.vox[b],
    local: bb.local_bounds[b],
    world: bb.world_aabb(b),
  };
  let cc = chunk_of(pa.world.0.floor().as_ivec3());
  let sc = world.statics.get(cc).expect("体所在静态分块");
  let gw = &volumes.list[0];
  let pb = Probe {
    field: Field::new_at(gw, gw.transform()),
    vox: &sc.vox,
    local: sc.local,
    world: world_aabb_of(sc.local, gw.transform()),
  };
  let m = manifold_bounded(&pa, &pb, &StepConfig::default().contact);
  eprintln!("{name}: 接触={}", m.points.len());
  for p in m.points.iter().take(4) {
    eprintln!(
      "   n({:+.2},{:+.2},{:+.2}) d{:+.3} @({:.2},{:.2},{:.2})",
      p.normal.x, p.normal.y, p.normal.z, p.depth, p.point.x, p.point.y, p.point.z
    );
  }
}

#[ignore = "复现探针，按需运行（见文件头注释）"]
#[test]
fn resting_jitter_probe() {
  let steps = 900;
  run("平地/整数对齐", flat_ground(), Vec3::new(24.0, 40.0, 24.0), steps);
  run("平地/偏移0.5", flat_ground(), Vec3::new(24.5, 40.0, 24.5), steps);
  run("平地/偏移0.3", flat_ground(), Vec3::new(24.3, 40.0, 24.3), steps);
  run("阶梯/整数对齐", stepped_ground(), Vec3::new(24.0, 40.0, 24.0), steps);
  dump_rest("静置/整数对齐", flat_ground(), Vec3::new(24.0, 8.0, 24.0));
  dump_rest("静置/偏移0.5", flat_ground(), Vec3::new(24.5, 8.0, 24.5));
  dump_rest("静置/偏移0.25", flat_ground(), Vec3::new(24.25, 8.0, 24.25));
}

fn lever(volumes: &mut Volumes, world: &mut PhysicsWorld, weights: &[(f32, i32)]) -> usize {
  spawn_dims(volumes, world, Vec3::new(26.0, 8.0, 24.0), IVec3::new(4, 8, 8));
  let beam = spawn_dims(volumes, world, Vec3::new(4.0, 16.0, 24.0), IVec3::new(48, 4, 8));
  for &(dx, edge) in weights {
    let e = edge as f32;
    spawn_dims(
      volumes,
      world,
      Vec3::new(28.0 + dx - e * 0.5, 20.0, 28.0 - e * 0.5),
      IVec3::splat(edge),
    );
  }
  beam
}

#[ignore = "参数扫描，按需运行（见文件头注释）"]
#[test]
fn retune_sweep() {
  let cases: [(u32, u32); 5] = [(32, 1), (64, 1), (32, 2), (16, 2), (8, 4)];
  eprintln!("子步 迭代 | 落地y | 重压轻:大y/睡 | 平衡梁:tilt | 杠杆:最负倾角 质心 最低角点y");
  for (substeps, iterations) in cases {
    let cfg = StepConfig { substeps, iterations, ..Default::default() };
    let mut volumes = Volumes::new(flat_ground());
    let mut world = PhysicsWorld::new();
    world.init_static_world(&volumes.list[0], tight_bounds(&volumes.list[0]).unwrap(), 0);
    let b = spawn_dims(&mut volumes, &mut world, Vec3::new(24.0, 32.0, 24.0), IVec3::splat(16));
    for _ in 0..180 {
      world.step(&volumes, DT, &cfg);
    }
    let drop_y = world.bodies.field_transform(b).pos.y;
    let mut volumes = Volumes::new(flat_ground());
    let mut world = PhysicsWorld::new();
    world.init_static_world(&volumes.list[0], tight_bounds(&volumes.list[0]).unwrap(), 0);
    let small = spawn_dims(&mut volumes, &mut world, Vec3::new(24.0, 8.0, 24.0), IVec3::splat(8));
    let big = spawn_dims(&mut volumes, &mut world, Vec3::new(20.0, 20.0, 20.0), IVec3::splat(16));
    for _ in 0..600 {
      world.step(&volumes, DT, &cfg);
    }
    let heavy = format!(
      "{:>6.2}/{}/{}",
      world.bodies.field_transform(big).pos.y,
      world.bodies.sleeping[big],
      world.bodies.sleeping[small]
    );

    let mut volumes = Volumes::new(flat_ground());
    let mut world = PhysicsWorld::new();
    world.init_static_world(&volumes.list[0], tight_bounds(&volumes.list[0]).unwrap(), 0);
    let beam = lever(&mut volumes, &mut world, &[(-16.0, 12), (16.0, 12)]);
    for _ in 0..600 {
      world.step(&volumes, DT, &cfg);
    }
    let level = world.bodies.rot[beam].x_axis.y;

    let mut volumes = Volumes::new(flat_ground());
    let mut world = PhysicsWorld::new();
    world.init_static_world(&volumes.list[0], tight_bounds(&volumes.list[0]).unwrap(), 0);
    let beam = lever(&mut volumes, &mut world, &[(16.0, 16)]);
    let mut most = 0.0f32;
    for _ in 0..300 {
      world.step(&volumes, DT, &cfg);
      most = most.min(world.bodies.rot[beam].x_axis.y);
    }
    let c = world.bodies.pos[beam];
    let mut low = f32::MAX;
    for i in 0..8 {
      let l = Vec3::new(
        if i & 1 == 0 { 0.0 } else { 48.0 },
        if i & 2 == 0 { 0.0 } else { 4.0 },
        if i & 4 == 0 { 0.0 } else { 8.0 },
      );
      low = low.min((world.bodies.rot[beam] * l + world.bodies.field_transform(beam).pos).y);
    }
    eprintln!(
      "{substeps:<4} {iterations:<4} | {drop_y:>6.2} | {heavy} | {level:>+6.3} | {most:>+6.3} ({:.1},{:.1}) {low:.2}",
      c.x, c.y
    );
  }
}
