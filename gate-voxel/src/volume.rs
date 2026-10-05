use std::collections::{HashMap, HashSet};

use glam::{IVec3, Mat3, Vec3};

use crate::chunk_tree::ChunkTree;
use crate::coords::{CHUNK_SIZE, ChunkCoord, VoxelCoord};
use crate::dirty::DirtyTracker;
use crate::palette::{Palette, PaletteId};

pub const COMP_BRICKS_PER_CHUNK: usize = 16 * 16 * 16;
pub const COMP_BRICK_EXTENT: i32 = 16;

const RESIDENT_LOG_MAX: usize = 16 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidentChange {
  pub c: ChunkCoord,
  pub mounted: bool,
}

#[derive(Debug, Clone)]
pub struct VolumeGrid {
  chunks: HashMap<ChunkCoord, ChunkTree>,
  palette: Palette,
  pub dirty: DirtyTracker,
  comp_layer: HashMap<ChunkCoord, Box<[u16; COMP_BRICKS_PER_CHUNK]>>,
  state_table: Vec<[u32; 4]>,
  pub state_dirty: bool,
  pub transform: VolumeTransform,
  pub obj_id: i32,
  edit_generation: u64,
  resident_seq: u64,
  resident_log: Vec<ResidentChange>,
  resident_log_epoch: u64,
  edit_aabbs: HashMap<ChunkCoord, (IVec3, IVec3)>,
  stream_window: Option<(IVec3, IVec3)>,
  empty_chunks: HashSet<ChunkCoord>,
  empty_log: Vec<ChunkCoord>,
  empty_seq: u64,
  far_level: bool,
  coverage_r: f32,
  attach_far: bool,
}

impl Default for VolumeGrid {
  fn default() -> Self {
    Self {
      chunks: HashMap::new(),
      palette: Palette::default(),
      dirty: DirtyTracker::new(),
      comp_layer: HashMap::new(),
      state_table: vec![[0u32; 4]; 256],
      state_dirty: true,
      transform: VolumeTransform::IDENTITY,
      obj_id: -1,
      edit_generation: 0,
      resident_seq: 0,
      resident_log: Vec::new(),
      resident_log_epoch: 0,
      edit_aabbs: HashMap::new(),
      stream_window: None,
      empty_chunks: HashSet::new(),
      empty_log: Vec::new(),
      empty_seq: 0,
      far_level: false,
      coverage_r: 0.0,
      attach_far: false,
    }
  }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VolumeTransform {
  pub pos: Vec3,
  pub rot: Mat3,
  pub scale: f32,
}

impl Default for VolumeTransform {
  fn default() -> Self {
    Self::IDENTITY
  }
}

impl VolumeTransform {
  pub const IDENTITY: Self = Self { pos: Vec3::ZERO, rot: Mat3::IDENTITY, scale: 1.0 };

  pub fn new(pos: Vec3, rot: Mat3, scale: f32) -> Self {
    Self { pos, rot, scale }
  }

  pub fn world_aabb(&self) -> (Vec3, Vec3) {
    let mut mn = Vec3::splat(f32::MAX);
    let mut mx = Vec3::splat(f32::MIN);
    for &x in &[0.0_f32, 256.0] {
      for &y in &[0.0, 256.0] {
        for &z in &[0.0, 256.0] {
          let local = Vec3::new(x, y, z) * self.scale;
          let w = self.pos + self.rot * local;
          mn = mn.min(w);
          mx = mx.max(w);
        }
      }
    }
    (mn, mx)
  }
}

#[derive(Debug, Default)]
pub struct Volumes {
  pub list: Vec<VolumeGrid>,
}

impl Volumes {
  pub fn new(main_world: VolumeGrid) -> Self {
    let mut list = Vec::with_capacity(8);
    list.push(main_world);
    Self { list }
  }

  pub fn main(&self) -> &VolumeGrid {
    &self.list[0]
  }

  pub fn main_mut(&mut self) -> &mut VolumeGrid {
    &mut self.list[0]
  }

  pub fn add_object(&mut self, pos: Vec3, rot: Mat3, scale: f32) -> usize {
    let obj_id = self.list.len() as i32 - 1;
    let grid = VolumeGrid::new_object(obj_id, pos, rot, scale);
    self.list.push(grid);
    obj_id as usize
  }

  pub fn object(&self, obj_id: usize) -> Option<&VolumeGrid> {
    self.list.get(obj_id + 1)
  }

  pub fn object_mut(&mut self, obj_id: usize) -> Option<&mut VolumeGrid> {
    self.list.get_mut(obj_id + 1)
  }

  pub fn add_far_level(&mut self, scale: f32) -> usize {
    let idx = self.list.len();
    let mut grid = VolumeGrid::new_object(idx as i32 - 1, Vec3::ZERO, Mat3::IDENTITY, scale);
    grid.far_level = true;
    self.list.push(grid);
    idx
  }

  pub fn len(&self) -> usize {
    self.list.len()
  }

  pub fn is_empty(&self) -> bool {
    self.list.is_empty()
  }

  pub fn all(&self) -> &[VolumeGrid] {
    &self.list
  }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirtyEdit {
  pub chunk: ChunkCoord,
}

impl VolumeGrid {
  pub fn new() -> Self {
    Self::default()
  }

  pub fn new_object(obj_id: i32, pos: Vec3, rot: Mat3, scale: f32) -> Self {
    Self { obj_id, transform: VolumeTransform::new(pos, rot, scale), ..Self::default() }
  }

  pub fn transform(&self) -> VolumeTransform {
    self.transform
  }

  pub fn set_transform(&mut self, pos: Vec3, rot: Mat3, scale: f32) {
    self.transform = VolumeTransform::new(pos, rot, scale);
  }

  pub fn obj_id(&self) -> i32 {
    self.obj_id
  }

  pub fn is_far_level(&self) -> bool {
    self.far_level
  }

  pub fn attach_far(&self) -> bool {
    self.attach_far
  }

  pub fn set_attach_far(&mut self, on: bool) {
    self.attach_far = on;
  }

  pub fn edit_generation(&self) -> u64 {
    self.edit_generation
  }

  pub fn take_edit_aabb(&mut self, chunk: ChunkCoord) -> Option<(IVec3, IVec3)> {
    self.edit_aabbs.remove(&chunk)
  }

  fn note_edit(&mut self, chunk: ChunkCoord, voxel: IVec3) {
    self.note_edit_box(chunk, voxel, voxel + IVec3::ONE);
  }

  fn note_edit_box(&mut self, chunk: ChunkCoord, lo: IVec3, hi: IVec3) {
    match self.edit_aabbs.entry(chunk) {
      std::collections::hash_map::Entry::Vacant(v) => {
        v.insert((lo, hi));
      }
      std::collections::hash_map::Entry::Occupied(mut o) => {
        let (l, h) = o.get_mut();
        *l = l.min(lo);
        *h = h.max(hi);
      }
    }
  }

  pub fn palette(&self) -> &Palette {
    &self.palette
  }

  pub fn palette_mut(&mut self) -> &mut Palette {
    &mut self.palette
  }

  pub fn chunk(&self, coord: ChunkCoord) -> Option<&ChunkTree> {
    self.chunks.get(&coord)
  }

  pub fn chunk_mut(&mut self, coord: ChunkCoord) -> Option<&mut ChunkTree> {
    self.chunks.get_mut(&coord)
  }

  pub fn chunk_coords(&self) -> impl Iterator<Item = ChunkCoord> + '_ {
    self.chunks.keys().copied()
  }

  pub fn chunk_count(&self) -> usize {
    self.chunks.len()
  }

  pub fn resident_seq(&self) -> u64 {
    self.resident_seq
  }

  pub fn resident_log_from(&self, from: usize) -> &[ResidentChange] {
    self.resident_log.get(from.min(self.resident_log.len())..).unwrap_or(&[])
  }

  pub fn resident_log_len(&self) -> usize {
    self.resident_log.len()
  }

  pub fn resident_log_epoch(&self) -> u64 {
    self.resident_log_epoch
  }

  fn note_resident_change(&mut self, c: ChunkCoord, mounted: bool) {
    self.resident_log.push(ResidentChange { c, mounted });
    if self.resident_log.len() > RESIDENT_LOG_MAX {
      self.resident_log.clear();
      self.resident_log_epoch = self.resident_log_epoch.wrapping_add(1);
    }
  }

  pub fn mark_empty_chunk(&mut self, cc: ChunkCoord) {
    if self.empty_chunks.insert(cc) {
      self.empty_log.push(cc);
      self.empty_seq = self.empty_seq.wrapping_add(1);
    }
  }

  pub fn empty_log_from(&self, from: usize) -> &[ChunkCoord] {
    self.empty_log.get(from.min(self.empty_log.len())..).unwrap_or(&[])
  }

  pub fn empty_count(&self) -> usize {
    self.empty_log.len()
  }

  pub fn empty_seq(&self) -> u64 {
    self.empty_seq
  }

  pub fn compact_all(&mut self) {
    use rayon::prelude::*;
    self.chunks.par_iter_mut().for_each(|(_, tree)| tree.compact());
  }

  pub fn mount_chunk_tree(&mut self, cc: ChunkCoord, mut tree: ChunkTree, applied_edits: u64) {
    debug_assert!(!tree.is_empty(), "mount_chunk_tree 不接受空树");
    tree.mark_replaced();
    self.chunks.insert(cc, tree);
    self.dirty.mark_data(cc);
    self.edit_generation = self.edit_generation.wrapping_add(applied_edits);
    self.resident_seq = self.resident_seq.wrapping_add(1);
    self.note_resident_change(cc, true);
  }

  pub fn unmount_chunk(&mut self, cc: ChunkCoord) -> bool {
    self.edit_aabbs.remove(&cc);
    self.comp_layer.remove(&cc);
    let had = self.chunks.remove(&cc).is_some();
    if had {
      self.resident_seq = self.resident_seq.wrapping_add(1);
      self.note_resident_change(cc, false);
    }
    had
  }

  pub fn take_chunk(&mut self, cc: ChunkCoord) -> Option<ChunkTree> {
    self.edit_aabbs.remove(&cc);
    self.comp_layer.remove(&cc);
    let t = self.chunks.remove(&cc);
    if t.is_some() {
      self.resident_seq = self.resident_seq.wrapping_add(1);
      self.note_resident_change(cc, false);
    }
    t
  }

  pub fn set_stream_window(&mut self, window: Option<(IVec3, IVec3)>) {
    self.stream_window = window;
  }

  pub fn stream_window(&self) -> Option<(IVec3, IVec3)> {
    self.stream_window
  }

  pub fn set_coverage_r(&mut self, r: f32) {
    self.coverage_r = r.max(0.0);
  }

  pub fn coverage_r(&self) -> f32 {
    self.coverage_r
  }

  pub fn get_voxel(&self, voxel: VoxelCoord) -> Option<PaletteId> {
    let chunk = voxel.chunk();
    let tree = self.chunks.get(&chunk)?;
    let local = voxel.in_chunk();
    tree.get_voxel(local.x, local.y, local.z)
  }

  pub fn get_uniform(&self, voxel: VoxelCoord, level: u8) -> Option<PaletteId> {
    let chunk = voxel.chunk();
    let tree = self.chunks.get(&chunk)?;
    let local = voxel.in_chunk();
    tree.get_uniform(local.x, local.y, local.z, level)
  }

  pub fn get_brick_state(&self, voxel: VoxelCoord, level: u8) -> crate::chunk_tree::BrickState {
    use crate::chunk_tree::BrickState;
    let chunk = voxel.chunk();
    let Some(tree) = self.chunks.get(&chunk) else {
      return BrickState::Air;
    };
    let local = voxel.in_chunk();
    tree.get_brick_state(local.x, local.y, local.z, level)
  }

  pub fn get_brick_state_extent(&self, voxel: IVec3, extent: i32) -> crate::chunk_tree::BrickState {
    self.get_brick_state(VoxelCoord::from_ivec3(voxel), crate::chunk_tree::level_of_extent(extent))
  }

  pub fn set_voxel(&mut self, voxel: VoxelCoord, palette: PaletteId) -> Option<DirtyEdit> {
    let chunk = voxel.chunk();
    let local = voxel.in_chunk();

    let tree = self.chunks.entry(chunk).or_insert_with(ChunkTree::empty);
    if tree.set_voxel(local.x, local.y, local.z, palette) {
      self.dirty.mark_data(chunk);
      self.note_edit(chunk, chunk.0 * CHUNK_SIZE + local);
      self.edit_generation = self.edit_generation.wrapping_add(1);
      Some(DirtyEdit { chunk })
    } else {
      None
    }
  }

  pub fn set_voxel_ivec3(&mut self, pos: IVec3, palette: PaletteId) -> Option<DirtyEdit> {
    self.set_voxel(VoxelCoord::from_ivec3(pos), palette)
  }

  pub fn set_brick_voxels(&mut self, voxel: IVec3, inside: u64, palette: PaletteId) -> u32 {
    const EXT: i32 = crate::coords::BRICK_FACTOR;
    let chunk = voxel.div_euclid(IVec3::splat(CHUNK_SIZE));
    let local = voxel.rem_euclid(IVec3::splat(CHUNK_SIZE));
    let cc = ChunkCoord(chunk);
    let changed = match self.chunks.get_mut(&cc) {
      Some(tree) => tree.set_brick_voxels([local.x, local.y, local.z], inside, palette),
      None => {
        if palette.is_air() {
          return 0;
        }
        let tree = self.chunks.entry(cc).or_insert_with(ChunkTree::empty);
        tree.set_brick_voxels([local.x, local.y, local.z], inside, palette)
      }
    };
    if changed > 0 {
      self.dirty.mark_data(cc);
      self.note_edit_box(cc, voxel, voxel + IVec3::splat(EXT));
      self.edit_generation = self.edit_generation.wrapping_add(1);
    }
    changed
  }

  pub fn fill_brick(&mut self, voxel: IVec3, extent: i32, palette: PaletteId) -> Option<DirtyEdit> {
    let chunk = voxel.div_euclid(IVec3::splat(CHUNK_SIZE));
    let local = voxel.rem_euclid(IVec3::splat(CHUNK_SIZE));
    let cc = ChunkCoord(chunk);
    let changed = match self.chunks.get_mut(&cc) {
      Some(tree) => tree.fill_brick([local.x, local.y, local.z], extent, palette),
      None => {
        if palette.is_air() {
          return None;
        }
        let tree = self.chunks.entry(cc).or_insert_with(ChunkTree::empty);
        tree.fill_brick([local.x, local.y, local.z], extent, palette)
      }
    };
    if changed {
      self.dirty.mark_data(cc);
      self.note_edit_box(cc, voxel, voxel + IVec3::splat(extent));
      self.edit_generation = self.edit_generation.wrapping_add(1);
      Some(DirtyEdit { chunk: cc })
    } else {
      None
    }
  }

  pub fn clear_voxel(&mut self, voxel: VoxelCoord) -> Option<DirtyEdit> {
    let chunk = voxel.chunk();
    let local = voxel.in_chunk();
    let tree = self.chunks.get_mut(&chunk)?;
    if tree.clear_voxel(local.x, local.y, local.z) {
      if tree.is_empty() {
        self.chunks.remove(&chunk);
      }
      self.dirty.mark_data(chunk);
      self.note_edit(chunk, chunk.0 * CHUNK_SIZE + local);
      self.edit_generation = self.edit_generation.wrapping_add(1);
      Some(DirtyEdit { chunk })
    } else {
      None
    }
  }

  pub fn batch_edit(&mut self, ops: impl IntoIterator<Item = (VoxelCoord, PaletteId)>) -> usize {
    let mut applied = 0;
    for (voxel, palette) in ops {
      if self.set_voxel(voxel, palette).is_some() {
        applied += 1;
      }
    }
    applied
  }

  pub fn set_comp(&mut self, chunk: ChunkCoord, bx: u32, by: u32, bz: u32, comp_id: u16) {
    let arr =
      self.comp_layer.entry(chunk).or_insert_with(|| Box::new([0u16; COMP_BRICKS_PER_CHUNK]));
    let idx = (bx as usize)
      + (by as usize) * COMP_BRICK_EXTENT as usize
      + (bz as usize) * COMP_BRICK_EXTENT as usize * COMP_BRICK_EXTENT as usize;
    if arr[idx] != comp_id {
      arr[idx] = comp_id;
      self.dirty.mark_comp(chunk);
    }
  }

  pub fn get_comp(&self, voxel: VoxelCoord) -> u16 {
    let chunk = voxel.chunk();
    let local = voxel.in_chunk();
    let bx = (local.x / COMP_BRICK_EXTENT) as usize;
    let by = (local.y / COMP_BRICK_EXTENT) as usize;
    let bz = (local.z / COMP_BRICK_EXTENT) as usize;
    let arr = match self.comp_layer.get(&chunk) {
      Some(a) => a,
      None => return 0,
    };
    arr[bx
      + by * COMP_BRICK_EXTENT as usize
      + bz * COMP_BRICK_EXTENT as usize * COMP_BRICK_EXTENT as usize]
  }

  pub fn comp_layer(&self) -> &HashMap<ChunkCoord, Box<[u16; COMP_BRICKS_PER_CHUNK]>> {
    &self.comp_layer
  }

  pub fn set_state(&mut self, id: u8, word: usize, value: u32) {
    debug_assert!(word < 4);
    if id as usize >= self.state_table.len() {
      self.state_table.resize(id as usize + 1, [0u32; 4]);
    }
    self.state_table[id as usize][word] = value;
    self.state_dirty = true;
  }

  pub fn get_state(&self, id: u8, word: usize) -> u32 {
    debug_assert!(word < 4);
    self.state_table.get(id as usize).map(|a| a[word]).unwrap_or(0)
  }

  pub fn state_table_bytes(&self) -> &[u8] {
    let slice = &self.state_table[..];
    unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, slice.len() * 16) }
  }
}