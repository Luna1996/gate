use std::collections::HashMap;

use bevy::{
  prelude::Resource,
  render::{
    render_resource::{
      AccelerationStructureFlags, AccelerationStructureGeometryFlags,
      AccelerationStructureUpdateMode, Blas, BlasBuildEntry, BlasGeometries,
      BlasGeometrySizeDescriptors, Buffer, BufferDescriptor, BufferUsages, CreateBlasDescriptor,
      CreateTlasDescriptor, Tlas, TlasInstance,
    },
    renderer::{RenderDevice, RenderQueue},
  },
};
use gate_voxel::{CHUNK_SIZE, ChunkCoord, ChunkTree, PaletteId};
use glam::Vec3;

pub const AABB_STRIDE: u64 = 24;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChunkAabbs {
  pub min: Vec3,
  pub max: Vec3,
  pub occupied: bool,
}

impl Default for ChunkAabbs {
  fn default() -> Self {
    Self { min: Vec3::ZERO, max: Vec3::ZERO, occupied: false }
  }
}

impl ChunkAabbs {
  pub fn to_flat(&self) -> [f32; 6] {
    [self.min.x, self.min.y, self.min.z, self.max.x, self.max.y, self.max.z]
  }

  pub fn len(&self) -> usize {
    1
  }

  pub fn is_empty(&self) -> bool {
    !self.occupied
  }
}

#[inline]
pub fn slab_axes(i: u32) -> (u32, u32, u32) {
  debug_assert!(i < 64);
  (i & 3, (i >> 2) & 3, (i >> 4) & 3)
}

#[inline]
pub fn slab_index(ix: u32, iy: u32, iz: u32) -> u32 {
  gate_voxel::child_linear_idx(ix as i32, iy as i32, iz as i32)
}

pub fn gather_chunk_aabbs(tree: &ChunkTree, chunk_min: Vec3, scale: f32) -> ChunkAabbs {
  let (mask, implicit) = match tree.node_view(0) {
    Some(v) => (v.mask, v.palette),
    None => (0, tree.root_palette()),
  };
  let implicit_solid = implicit != PaletteId::AIR;
  let s = step(scale);
  let mut lo = Vec3::splat(f32::INFINITY);
  let mut hi = Vec3::splat(f32::NEG_INFINITY);
  for i in 0..64u32 {
    if ((mask >> i) & 1 == 0) && !implicit_solid {
      continue;
    }
    let mn = block_min(chunk_min, i, scale);
    lo = lo.min(mn);
    hi = hi.max(mn + s);
  }
  if lo.x > hi.x {
    return ChunkAabbs::default();
  }
  ChunkAabbs { min: lo, max: hi, occupied: true }
}

#[inline]
fn step(scale: f32) -> f32 {
  (CHUNK_SIZE / 4) as f32 * scale
}

#[inline]
fn block_min(chunk_min: Vec3, i: u32, scale: f32) -> Vec3 {
  let (ix, iy, iz) = slab_axes(i);
  let s = step(scale);
  chunk_min + Vec3::new(ix as f32 * s, iy as f32 * s, iz as f32 * s)
}

pub const RT_MAX_INSTANCES: u32 = 65536;

const RT_FREE_DELAY_FRAMES: u64 = 8;

const IDENTITY_3X4: [f32; 12] = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];

const RT_INSTANCE_MASK: u8 = 0xFF;

#[inline]
pub fn rt_custom_index(c: ChunkCoord) -> u32 {
  ((c.0.x as u32) & 0xFF) | (((c.0.y as u32) & 0xFF) << 8) | (((c.0.z as u32) & 0xFF) << 16)
}

const _: () = assert!(
  super::wire::CHUNK_INDEX_CAP <= 64,
  "窗口每轴必须 ≤ 64，否则 custom_data 的低字节还原会撞"
);

pub fn rt_enabled(device: &RenderDevice) -> bool {
  crate::brickmap::consts::RT_RAY_QUERY
    && device
      .features()
      .contains(bevy::render::render_resource::WgpuFeatures::EXPERIMENTAL_RAY_QUERY)
}

struct ChunkBlas {
  blas: Blas,
  aabb: Buffer,
}

#[derive(Resource)]
pub struct RtScene {
  enabled: bool,
  tlas: Tlas,
  slots: Vec<Option<ChunkCoord>>,
  free_slots: Vec<u32>,
  by_chunk: HashMap<ChunkCoord, u32>,
  blas: HashMap<ChunkCoord, ChunkBlas>,
  pending_blas: Vec<ChunkCoord>,
  dirty_slots: Vec<u32>,
  graveyard: Vec<(u64, ChunkBlas)>,
  pub deferred: Vec<ChunkCoord>,
  frame: u64,
  never_built: bool,
}

impl RtScene {
  pub fn new(enabled: bool, device: &RenderDevice) -> Self {
    let tlas = device.wgpu_device().create_tlas(&CreateTlasDescriptor {
      label: Some("gate_rt_tlas"),
      max_instances: RT_MAX_INSTANCES,
      flags: AccelerationStructureFlags::PREFER_FAST_BUILD,
      update_mode: AccelerationStructureUpdateMode::Build,
    });
    Self {
      enabled,
      tlas,
      slots: vec![None; RT_MAX_INSTANCES as usize],
      free_slots: (0..RT_MAX_INSTANCES).rev().collect(),
      by_chunk: HashMap::new(),
      blas: HashMap::new(),
      pending_blas: Vec::new(),
      dirty_slots: Vec::new(),
      graveyard: Vec::new(),
      deferred: Vec::new(),
      frame: 0,
      never_built: true,
    }
  }

  pub fn tick(&mut self) {
    self.frame = self.frame.wrapping_add(1);
    let now = self.frame;
    self.graveyard.retain(|(due, _)| *due > now);
  }

  pub fn is_enabled(&self) -> bool {
    self.enabled
  }

  pub fn tlas(&self) -> &Tlas {
    &self.tlas
  }

  pub fn instance_count(&self) -> usize {
    self.by_chunk.len()
  }

  pub fn chunks(&self) -> impl Iterator<Item = ChunkCoord> + '_ {
    self.by_chunk.keys().copied()
  }

  pub fn contains(&self, c: ChunkCoord) -> bool {
    self.by_chunk.contains_key(&c)
  }

  pub fn slots_exhausted(&self) -> bool {
    self.free_slots.is_empty() && self.slots.len() == self.by_chunk.len()
  }

  pub fn insert_chunk(
    &mut self,
    device: &RenderDevice,
    queue: &RenderQueue,
    coord: ChunkCoord,
    aabbs: &ChunkAabbs,
  ) -> bool {
    if !self.enabled || self.by_chunk.contains_key(&coord) {
      return false;
    }
    if aabbs.is_empty() {
      return false;
    }
    let Some(slot) = self.free_slots.pop() else {
      return false;
    };
    let flat = aabbs.to_flat();
    let aabb = device.create_buffer(&BufferDescriptor {
      label: Some("gate_rt_aabb"),
      size: AABB_STRIDE,
      usage: BufferUsages::BLAS_INPUT | BufferUsages::COPY_DST,
      mapped_at_creation: false,
    });
    queue.write_buffer(&aabb, 0, f32_bytes(&flat));
    let blas = device.wgpu_device().create_blas(
      &CreateBlasDescriptor {
        label: Some("gate_rt_blas"),
        flags: AccelerationStructureFlags::PREFER_FAST_TRACE,
        update_mode: AccelerationStructureUpdateMode::Build,
      },
      BlasGeometrySizeDescriptors::AABBs { descriptors: vec![aabb_size_desc(1)] },
    );
    self.blas.insert(coord, ChunkBlas { blas, aabb });
    self.slots[slot as usize] = Some(coord);
    self.by_chunk.insert(coord, slot);
    self.pending_blas.push(coord);
    self.dirty_slots.push(slot);
    true
  }

  pub fn remove_chunk(&mut self, coord: ChunkCoord) -> bool {
    let Some(slot) = self.by_chunk.remove(&coord) else {
      return false;
    };
    self.slots[slot as usize] = None;
    self.free_slots.push(slot);
    self.pending_blas.retain(|c| c != &coord);
    if let Some(b) = self.blas.remove(&coord) {
      self.graveyard.push((self.frame + RT_FREE_DELAY_FRAMES, b));
    }
    self.dirty_slots.push(slot);
    true
  }

  pub fn clear(&mut self) {
    for (i, s) in self.slots.iter_mut().enumerate() {
      if s.take().is_some() {
        self.dirty_slots.push(i as u32);
      }
    }
    self.by_chunk.clear();
    for (_, b) in self.blas.drain() {
      self.graveyard.push((self.frame + RT_FREE_DELAY_FRAMES, b));
    }
    self.pending_blas.clear();
    self.free_slots = (0..RT_MAX_INSTANCES).rev().collect();
    self.never_built = true;
  }

  pub fn record(&mut self, encoder: &mut wgpu::CommandEncoder) -> bool {
    if !self.enabled
      || (!self.never_built && self.pending_blas.is_empty() && self.dirty_slots.is_empty())
    {
      return false;
    }
    for &slot in &self.dirty_slots {
      let inst = self.slots[slot as usize].and_then(|c| {
        self
          .blas
          .get(&c)
          .map(|b| TlasInstance::new(&b.blas, IDENTITY_3X4, rt_custom_index(c), RT_INSTANCE_MASK))
      });
      if let Some(s) = self.tlas.get_mut_single(slot as usize) {
        *s = inst;
      }
    }
    self.dirty_slots.clear();
    {
      let live: Vec<ChunkCoord> =
        self.pending_blas.iter().copied().filter(|c| self.blas.contains_key(c)).collect();
      let sizes: Vec<wgpu::BlasAABBGeometrySizeDescriptor> =
        live.iter().map(|_| aabb_size_desc(1)).collect();
      let entries: Vec<BlasBuildEntry> = live
        .iter()
        .zip(sizes.iter())
        .map(|(c, size)| BlasBuildEntry {
          blas: &self.blas[c].blas,
          geometry: BlasGeometries::AabbGeometries(vec![wgpu::BlasAabbGeometry {
            size,
            stride: AABB_STRIDE,
            aabb_buffer: &self.blas[c].aabb,
            primitive_offset: 0,
          }]),
        })
        .collect();
      encoder.build_acceleration_structures(entries.iter(), std::iter::once(&self.tlas));
    }
    self.pending_blas.clear();
    self.never_built = false;
    true
  }
}

fn aabb_size_desc(prims: u32) -> wgpu::BlasAABBGeometrySizeDescriptor {
  wgpu::BlasAABBGeometrySizeDescriptor {
    primitive_count: prims,
    flags: AccelerationStructureGeometryFlags::OPAQUE,
  }
}

fn f32_bytes(v: &[f32]) -> &[u8] {
  unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}
