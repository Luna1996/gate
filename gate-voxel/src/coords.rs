use glam::IVec3;

pub const CHUNK_SIZE: i32 = 256;

pub const BRICK_FACTOR: i32 = 4;

pub const MAX_LEVEL: u8 = 4;

pub const LEVEL_EXTENT: [i32; 5] = [256, 64, 16, 4, 1];

#[inline]
pub fn child_linear_idx(local_x: i32, local_y: i32, local_z: i32) -> u32 {
  debug_assert!((0..BRICK_FACTOR).contains(&local_x));
  debug_assert!((0..BRICK_FACTOR).contains(&local_y));
  debug_assert!((0..BRICK_FACTOR).contains(&local_z));
  (local_z * BRICK_FACTOR * BRICK_FACTOR + local_y * BRICK_FACTOR + local_x) as u32
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ChunkCoord(pub IVec3);

impl ChunkCoord {
  pub fn new(x: i32, y: i32, z: i32) -> Self {
    Self(IVec3::new(x, y, z))
  }
}

impl Ord for ChunkCoord {
  fn cmp(&self, other: &Self) -> std::cmp::Ordering {
    (self.0.x, self.0.y, self.0.z).cmp(&(other.0.x, other.0.y, other.0.z))
  }
}

impl PartialOrd for ChunkCoord {
  fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
    Some(self.cmp(other))
  }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VoxelCoord {
  pub x: i32,
  pub y: i32,
  pub z: i32,
}

impl VoxelCoord {
  pub fn new(x: i32, y: i32, z: i32) -> Self {
    Self { x, y, z }
  }

  pub fn from_ivec3(v: IVec3) -> Self {
    Self { x: v.x, y: v.y, z: v.z }
  }

  pub fn to_ivec3(self) -> IVec3 {
    IVec3::new(self.x, self.y, self.z)
  }

  pub fn chunk(&self) -> ChunkCoord {
    let v = IVec3::new(self.x, self.y, self.z);
    ChunkCoord(v.div_euclid(IVec3::splat(CHUNK_SIZE)))
  }

  pub fn in_chunk(&self) -> IVec3 {
    let v = IVec3::new(self.x, self.y, self.z);
    v.rem_euclid(IVec3::splat(CHUNK_SIZE))
  }

  pub fn to_level_brick(&self, level: u8) -> (ChunkCoord, u32, u32, u32) {
    let chunk = self.chunk();
    let extent = LEVEL_EXTENT[level as usize];
    let local = self.in_chunk();
    let bx = (local.x / extent) as u32;
    let by = (local.y / extent) as u32;
    let bz = (local.z / extent) as u32;
    (chunk, bx, by, bz)
  }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BrickCoord {
  pub level: u8,
  pub chunk: ChunkCoord,
  pub x: u32,
  pub y: u32,
  pub z: u32,
}

impl BrickCoord {
  pub fn new(level: u8, chunk: ChunkCoord, x: u32, y: u32, z: u32) -> Self {
    Self { level, chunk, x, y, z }
  }

  pub fn size(&self) -> i32 {
    LEVEL_EXTENT[self.level as usize]
  }

  pub fn voxel_min(&self) -> VoxelCoord {
    let extent = self.size();
    let offset_x = self.x as i32 * extent;
    let offset_y = self.y as i32 * extent;
    let offset_z = self.z as i32 * extent;
    VoxelCoord::new(
      self.chunk.0.x * CHUNK_SIZE + offset_x,
      self.chunk.0.y * CHUNK_SIZE + offset_y,
      self.chunk.0.z * CHUNK_SIZE + offset_z,
    )
  }
}
