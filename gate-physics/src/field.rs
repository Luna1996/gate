use glam::{IVec3, Vec3};

use gate_voxel::{
  BrickState, CHUNK_SIZE, LEVEL_EXTENT, PaletteId, VolumeGrid, VolumeTransform, VoxelCoord,
};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum NodeFill {
  Air,
  Solid(PaletteId),
  Mixed,
}

impl NodeFill {
  pub fn is_solid(self) -> bool {
    matches!(self, NodeFill::Solid(_))
  }
}

#[derive(Clone, Copy)]
pub struct Field<'a> {
  grid: &'a VolumeGrid,
  tr: VolumeTransform,
}

impl<'a> Field<'a> {
  pub fn new(grid: &'a VolumeGrid) -> Self {
    Self { grid, tr: grid.transform() }
  }

  pub fn new_at(grid: &'a VolumeGrid, tr: VolumeTransform) -> Self {
    Self { grid, tr }
  }

  pub fn grid(&self) -> &'a VolumeGrid {
    self.grid
  }

  pub fn transform(&self) -> VolumeTransform {
    self.tr
  }

  pub fn to_local(&self, world: Vec3) -> Vec3 {
    let d = world - self.tr.pos;
    Vec3::new(d.dot(self.tr.rot.x_axis), d.dot(self.tr.rot.y_axis), d.dot(self.tr.rot.z_axis))
      / self.tr.scale
  }

  pub fn dir_to_local(&self, dir: Vec3) -> Vec3 {
    Vec3::new(dir.dot(self.tr.rot.x_axis), dir.dot(self.tr.rot.y_axis), dir.dot(self.tr.rot.z_axis))
      / self.tr.scale
  }

  pub fn to_world(&self, local: Vec3) -> Vec3 {
    self.tr.pos + self.tr.rot * (local * self.tr.scale)
  }

  pub fn voxel_local(&self, p: IVec3) -> Option<PaletteId> {
    self.grid.get_voxel(VoxelCoord::from_ivec3(p))
  }

  pub fn solid_local(&self, p: IVec3) -> bool {
    self.voxel_local(p).is_some()
  }

  pub fn solid_world(&self, world: Vec3) -> bool {
    self.solid_local(self.to_local(world).floor().as_ivec3())
  }

  pub fn node(&self, origin_local: IVec3, extent: i32) -> NodeFill {
    match self.grid.get_brick_state_extent(origin_local, extent) {
      BrickState::Air => NodeFill::Air,
      BrickState::Solid(p) => NodeFill::Solid(p),
      BrickState::Mixed => NodeFill::Mixed,
    }
  }

  pub fn for_each_node(
    &self,
    lo_local: IVec3,
    hi_local: IVec3,
    extent: i32,
    f: &mut impl FnMut(IVec3, NodeFill),
  ) {
    debug_assert!(
      LEVEL_EXTENT.contains(&extent),
      "extent 必须是 {LEVEL_EXTENT:?} 之一（got {extent}）"
    );
    let m = !(extent - 1);
    let lo = lo_local & IVec3::splat(m);
    let hi = hi_local & IVec3::splat(m);
    let mut z = lo.z;
    while z <= hi.z {
      let mut y = lo.y;
      while y <= hi.y {
        let mut x = lo.x;
        while x <= hi.x {
          let p = IVec3::new(x, y, z);
          f(p, self.node(p, extent));
          x += extent;
        }
        y += extent;
      }
      z += extent;
    }
  }
  pub fn local_bounds(&self) -> Option<(IVec3, IVec3)> {
    tight_bounds(self.grid)
  }

  pub fn local_aabb(&self) -> Option<(Vec3, Vec3)> {
    let (lo, hi) = self.local_bounds()?;
    Some((lo.as_vec3(), (hi + IVec3::ONE).as_vec3()))
  }

  pub fn world_aabb(&self) -> Option<(Vec3, Vec3)> {
    self.local_bounds().map(|b| world_aabb_of(b, self.tr))
  }
}

pub fn world_aabb_of(local: (IVec3, IVec3), tr: VolumeTransform) -> (Vec3, Vec3) {
  let lo = local.0.as_vec3();
  let hi = (local.1 + IVec3::ONE).as_vec3();
  let mut mn = Vec3::splat(f32::MAX);
  let mut mx = Vec3::splat(f32::MIN);
  for i in 0..8 {
    let c = Vec3::new(
      if i & 1 == 0 { lo.x } else { hi.x },
      if i & 2 == 0 { lo.y } else { hi.y },
      if i & 4 == 0 { lo.z } else { hi.z },
    );
    let w = tr.pos + tr.rot * (c * tr.scale);
    mn = mn.min(w);
    mx = mx.max(w);
  }
  (mn, mx)
}

pub fn chunk_bounds(grid: &VolumeGrid) -> Option<(Vec3, Vec3)> {
  let mut it = grid.chunk_coords();
  let first = it.next()?.0;
  let (mut lo, mut hi) = (first, first);
  for c in it {
    lo = lo.min(c.0);
    hi = hi.max(c.0);
  }
  let s = CHUNK_SIZE as f32;
  Some((lo.as_vec3() * s, (hi + IVec3::ONE).as_vec3() * s))
}

pub fn tight_bounds(grid: &VolumeGrid) -> Option<(IVec3, IVec3)> {
  let f = Field::new(grid);
  let mut lo = IVec3::splat(i32::MAX);
  let mut hi = IVec3::splat(i32::MIN);
  let mut any = false;
  for cc in grid.chunk_coords() {
    let base = cc.0 * CHUNK_SIZE;
    let top = base + IVec3::splat(CHUNK_SIZE - 1);
    f.for_each_node(base, top, 4, &mut |origin, fill| match fill {
      NodeFill::Air => {}
      NodeFill::Solid(_) => {
        any = true;
        lo = lo.min(origin);
        hi = hi.max(origin + IVec3::splat(3));
      }
      NodeFill::Mixed => {
        for dz in 0..4 {
          for dy in 0..4 {
            for dx in 0..4 {
              let p = origin + IVec3::new(dx, dy, dz);
              if grid.get_voxel(VoxelCoord::from_ivec3(p)).is_some() {
                any = true;
                lo = lo.min(p);
                hi = hi.max(p);
              }
            }
          }
        }
      }
    });
  }
  any.then_some((lo, hi))
}
