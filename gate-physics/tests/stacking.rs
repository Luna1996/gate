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

fn stack_ids(volumes: &mut Volumes, world: &mut PhysicsWorld, layers: i32) -> Vec<usize> {
  (0..layers)
    .map(|k| {
      let y = 8.0 + k as f32 * (EDGE as f32 + 1.0);
      spawn_box(volumes, world, Vec3::new(24.0, y, 24.0))
    })
    .collect()
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