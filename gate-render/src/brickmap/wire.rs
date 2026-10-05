use gate_voxel::{PALETTE_BITS, PALETTE_ENTRY_COUNT, PaletteEntry, PaletteFlags};
use glam::{IVec3, Mat3, Vec3, Vec4};

pub use gate_voxel::PbrOverrides;

pub const CHUNK_SIZE: i32 = 256;
pub const BRICK_FACTOR: i32 = 4;
pub const MAX_LEVEL: u32 = 4;
pub const NODE_FIXED_WORDS: usize = 3;

pub const CHUNK_INDEX_CAP: usize = 64;
pub const CHUNK_INDEX_WORDS: usize = CHUNK_INDEX_CAP * CHUNK_INDEX_CAP * CHUNK_INDEX_CAP;
pub const TREE_BASE: usize = CHUNK_INDEX_WORDS;

pub const PALETTE_WORDS: usize = PALETTE_ENTRY_COUNT * 2;
pub const LEAF_VOXELS_PER_WORD: usize = gate_voxel::LEAF_VOXELS_PER_WORD;
pub const LEAF_INLINE_WORDS: usize = gate_voxel::LEAF_INLINE_WORDS;
pub const PALETTE_BYTES_PER_ENTRY: usize = 8;

const _: () = assert!(64 % LEAF_VOXELS_PER_WORD == 0);
const _: () = assert!(32 % PALETTE_BITS == 0);
pub const CHUNK_COMP_WORDS: usize = gate_voxel::COMP_BRICKS_PER_CHUNK / 2;
pub const STATE_WORDS_PER_ENTRY: usize = 4;
pub const STATE_ENTRY_COUNT: usize = 256;
pub const STATE_TOTAL_WORDS: usize = STATE_ENTRY_COUNT * STATE_WORDS_PER_ENTRY;

pub fn pack_palette_entry(e: &PaletteEntry) -> [u32; 2] {
  if e.flags.contains(PaletteFlags::IS_PBR) {
    return [
      e.color[0] as u32
        | (e.color[1] as u32) << 8
        | (e.color[2] as u32) << 16
        | (e.roughness as u32) << 24,
      e.emissive as u32
        | (e.transmission as u32) << 8
        | (e.flags.0 as u32) << 16
        | (e.metallic as u32) << 24,
    ];
  }
  let mut flags = e.flags.0;
  if e.transmission > 0 {
    flags |= PaletteFlags::TRANSMISSIVE.0;
  }
  [
    e.color[0] as u32
      | (e.color[1] as u32) << 8
      | (e.color[2] as u32) << 16
      | (e.roughness as u32) << 24,
    e.emissive as u32
      | (e.transmission as u32) << 8
      | (flags as u32) << 16
      | (e.metallic as u32) << 24,
  ]
}

pub fn pack_palette_entry_pbr(asset: u16, ov: PbrOverrides, flags: PaletteFlags) -> [u32; 2] {
  pack_palette_entry(&PaletteEntry::pbr(asset, ov, flags))
}

pub const MATERIAL_SLOT_NONE: u32 = 0xFFFF_FFFF;

pub struct TileMeta;

impl TileMeta {
  pub const TILE_FLAG: u32 = 0x8000_0000;
  pub const TILE_BITS: u32 = 0x3F;

  pub fn emissive_slot(uv_log2: u8) -> u32 {
    (uv_log2 as u32) << 16 | 0xFFFF
  }

  pub fn transmission_slot(tile: u8) -> u32 {
    Self::TILE_FLAG | (tile as u32 & Self::TILE_BITS)
  }

  pub fn is_tiled(transmission_slot: u32) -> bool {
    transmission_slot & Self::TILE_FLAG != 0
  }

  pub fn tile(transmission_slot: u32) -> u32 {
    transmission_slot & Self::TILE_BITS
  }

  pub fn uv_log2(emissive_slot: u32) -> u32 {
    (emissive_slot >> 16) & 0xFF
  }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, bevy::render::render_resource::ShaderType)]
pub struct MaterialAsset {
  pub albedo_slot: u32,
  pub roughmetal_slot: u32,
  pub emissive_slot: u32,
  pub transmission_slot: u32,
  pub height_slot: u32,
  pub albedo_rough: u32,
  pub emissive_metal: u32,
  pub transmission_ior: u32,
}

const _: () = assert!(std::mem::size_of::<MaterialAsset>() == 32);

pub const MATERIAL_DISPLACE_AMPLITUDE_SHIFT: u32 = 24;
pub const MATERIAL_DISPLACE_AMPLITUDE_MASK: u32 = 0xFF;

const _: () = assert!(MATERIAL_DISPLACE_AMPLITUDE_SHIFT + 8 == 32);
const _: () =
  assert!(MATERIAL_DISPLACE_AMPLITUDE_MASK << MATERIAL_DISPLACE_AMPLITUDE_SHIFT == 0xFF00_0000);

impl MaterialAsset {
  pub fn displacement_amplitude(&self) -> u8 {
    ((self.emissive_metal >> MATERIAL_DISPLACE_AMPLITUDE_SHIFT) & MATERIAL_DISPLACE_AMPLITUDE_MASK)
      as u8
  }

  pub fn with_displacement_amplitude(mut self, amplitude: u8) -> Self {
    self.emissive_metal = (self.emissive_metal
      & !(MATERIAL_DISPLACE_AMPLITUDE_MASK << MATERIAL_DISPLACE_AMPLITUDE_SHIFT))
      | ((amplitude as u32) << MATERIAL_DISPLACE_AMPLITUDE_SHIFT);
    self
  }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, bevy::render::render_resource::ShaderType)]
pub struct BrickMapGlobals {
  pub index_origin_x: i32,
  pub index_origin_y: i32,
  pub index_origin_z: i32,
  pub index_origin_w: i32,
  pub index_dims_x: u32,
  pub index_dims_y: u32,
  pub index_dims_z: u32,
  pub index_dims_w: u32,
  pub tile_count: u32,
  pub node_words: u32,
  pub node_free_words: u32,
  pub brick_slabs: u32,
  pub brick_free: u32,
  pub rejected_tiles: u32,
  pub grid_count: u32,
  pub _pad1: u32,
  pub _pad2: u32,
  pub _pad3: u32,
  pub _pad4: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Default, bevy::render::render_resource::ShaderType)]
pub struct GridDesc {
  pub pos_scale: Vec4,
  pub rot0: Vec4,
  pub rot1: Vec4,
  pub rot2: Vec4,

  pub aabb_min: Vec4,
  pub aabb_max: Vec4,

  pub tree_base: u32,
  pub tree_depth: u32,
  pub chunk_count: u32,
  pub palette_base: u32,

  pub index_origin_x: i32,
  pub index_origin_y: i32,
  pub index_origin_z: i32,
  pub grid_flags: u32,
  pub index_dims_x: u32,
  pub index_dims_y: u32,
  pub index_dims_z: u32,
  pub coverage_r: f32,
}

pub const GRID_FLAG_FAR: u32 = 1;

impl GridDesc {
  pub const IDENTITY: Self = Self {
    pos_scale: Vec4::new(0.0, 0.0, 0.0, 1.0),
    rot0: Vec4::new(1.0, 0.0, 0.0, 0.0),
    rot1: Vec4::new(0.0, 1.0, 0.0, 0.0),
    rot2: Vec4::new(0.0, 0.0, 1.0, 0.0),
    aabb_min: Vec4::ZERO,
    aabb_max: Vec4::ZERO,
    tree_base: 0,
    tree_depth: MAX_LEVEL,
    chunk_count: 0,
    palette_base: 0,
    index_origin_x: 0,
    index_origin_y: 0,
    index_origin_z: 0,
    grid_flags: 0,
    index_dims_x: 0,
    index_dims_y: 0,
    index_dims_z: 0,
    coverage_r: 0.0,
  };

  #[allow(clippy::too_many_arguments)]
  pub fn from_transform(
    pos: Vec3,
    rot: Mat3,
    scale: f32,
    tree_base: u32,
    palette_base: u32,
    chunk_count: u32,
    origin: IVec3,
    dims: IVec3,
  ) -> Self {
    let (mn, mx) = transform_aabb(pos, rot, scale);
    Self {
      pos_scale: pos.extend(scale),
      rot0: rot.x_axis.extend(0.0),
      rot1: rot.y_axis.extend(0.0),
      rot2: rot.z_axis.extend(0.0),
      aabb_min: mn.extend(0.0),
      aabb_max: mx.extend(0.0),
      tree_base,
      tree_depth: MAX_LEVEL,
      chunk_count,
      palette_base,
      index_origin_x: origin.x,
      index_origin_y: origin.y,
      index_origin_z: origin.z,
      grid_flags: 0,
      index_dims_x: dims.x as u32,
      index_dims_y: dims.y as u32,
      index_dims_z: dims.z as u32,
      coverage_r: 0.0,
    }
  }
}

pub fn window_world_aabb(
  t: gate_voxel::VolumeTransform,
  origin: IVec3,
  dims: IVec3,
) -> (Vec3, Vec3) {
  let s = if t.scale.is_finite() && t.scale > 0.0 { t.scale } else { 1.0 };
  let lo = origin * CHUNK_SIZE;
  let hi = (origin + dims) * CHUNK_SIZE;
  let mut mn = Vec3::splat(f32::MAX);
  let mut mx = Vec3::splat(f32::MIN);
  for &x in &[lo.x, hi.x] {
    for &y in &[lo.y, hi.y] {
      for &z in &[lo.z, hi.z] {
        let w = t.pos + t.rot * (Vec3::new(x as f32, y as f32, z as f32) * s);
        mn = mn.min(w);
        mx = mx.max(w);
      }
    }
  }
  (mn, mx)
}

pub const MARCH_MASK_OCTANTS: usize = 8;
pub const MARCH_MASK_ENTRIES: usize = 64;
pub const MARCH_MASK_WORDS_PER_ENTRY: usize = 2;
pub const MARCH_MASK_WORDS: usize =
  MARCH_MASK_OCTANTS * MARCH_MASK_ENTRIES * MARCH_MASK_WORDS_PER_ENTRY;

pub fn march_mask_lut_words() -> Vec<u32> {
  let mut out = vec![0u32; MARCH_MASK_WORDS];
  for oct in 0..MARCH_MASK_OCTANTS {
    let pos = [oct & 1 != 0, (oct >> 1) & 1 != 0, (oct >> 2) & 1 != 0];
    for entry in 0..MARCH_MASK_ENTRIES {
      let e = [(entry & 3) as i32, ((entry >> 2) & 3) as i32, ((entry >> 4) & 3) as i32];
      let mut mask = 0u64;
      for p in 0..MARCH_MASK_ENTRIES {
        let q = [(p & 3) as i32, ((p >> 2) & 3) as i32, ((p >> 4) & 3) as i32];
        let ok =
          [0, 1, 2].iter().all(|&i| if pos[i] { q[i] >= e[i] - 1 } else { q[i] <= e[i] + 1 });
        if ok {
          mask |= 1u64 << p;
        }
      }
      let base =
        entry * MARCH_MASK_WORDS_PER_ENTRY + oct * MARCH_MASK_ENTRIES * MARCH_MASK_WORDS_PER_ENTRY;
      out[base] = mask as u32;
      out[base + 1] = (mask >> 32) as u32;
    }
  }
  out
}

fn transform_aabb(pos: Vec3, rot: Mat3, scale: f32) -> (Vec3, Vec3) {
  let mut mn = Vec3::splat(f32::MAX);
  let mut mx = Vec3::splat(f32::MIN);
  for &x in &[0.0_f32, 256.0] {
    for &y in &[0.0, 256.0] {
      for &z in &[0.0, 256.0] {
        let local = Vec3::new(x, y, z) * scale;
        let w = pos + rot * local;
        mn = mn.min(w);
        mx = mx.max(w);
      }
    }
  }
  (mn, mx)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BrickMapBuffers {
  pub b_struct: Vec<u32>,
  pub b_palette: Vec<u32>,
  pub globals: BrickMapGlobals,
}