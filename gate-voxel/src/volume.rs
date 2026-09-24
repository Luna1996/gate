//! Volume + Volumes：主世界 / 独立物体的统一容器。
//! `VolumeGrid` = 单个体素 volume（主世界 = identity transform + 无界 chunk HashMap；物体 = 任意 transform）。
//! `Volumes` = `Vec<VolumeGrid>`，list[0] 恒为主世界（obj_id = -1），list[1..N] 为物体。

use std::collections::HashMap;

use glam::{IVec3, Mat3, Vec3};

use crate::chunk_tree::ChunkTree;
use crate::coords::{CHUNK_SIZE, ChunkCoord, VoxelCoord};
use crate::dirty::DirtyTracker;
use crate::palette::{Palette, PaletteId};

/// 组件层：level 2 brick = 16³ = 一个组件 cell；每 chunk = 16³ = 4096 个。
pub const COMP_BRICKS_PER_CHUNK: usize = 16 * 16 * 16;
pub const COMP_BRICK_EXTENT: i32 = 16;

/// 主世界 / 独立物体的统一容器。
/// 主世界 = identity transform + 无界 chunk（obj_id = -1）；物体 = 任意 transform（obj_id = 0..N-1）。
#[derive(Debug, Clone)]
pub struct VolumeGrid {
  chunks: HashMap<ChunkCoord, ChunkTree>,
  palette: Palette,
  pub dirty: DirtyTracker,
  comp_layer: HashMap<ChunkCoord, Box<[u16; COMP_BRICKS_PER_CHUNK]>>,
  state_table: Vec<[u32; 4]>,
  pub state_dirty: bool,
  /// 物体变换（主世界 = identity）；渲染侧据此推导 DDA 局部变换与世界 AABB。
  pub transform: VolumeTransform,
  /// 渲染器分配的 obj_id（主世界 = -1；物体 = 0..N-1，对应 `Volumes.list[1..]` 索引）。
  pub obj_id: i32,
  /// 体素编辑单调代数（每次实际改变体素 +1；noop 同色重复写不递增，palette 变化不计入）；渲染侧派生数据（如探针烘焙）据此判断是否需重烘。
  edit_generation: u64,
  /// 编辑产生的最小 voxel AABB（按 chunk 记录，闭开区间 `[lo, hi)`，世界 voxel 坐标）。
  /// 上传该 chunk 时由 `take_edit_aabb` 取走；未记录者消费方回退到 chunk 包围盒。
  edit_aabbs: HashMap<ChunkCoord, (IVec3, IVec3)>,
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
      edit_aabbs: HashMap::new(),
    }
  }
}

/// 物体变换：`world = pos + rot · (local · scale)`；主世界 = identity。
/// `rot` 列向量约定与 glam `Mat3` 一致（x/y/z_axis 即列）。
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

  /// 局部 [0,256]³·scale 经旋转平移后的世界 AABB 外包（剔除用）。
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

/// 全场景容器：`Vec<VolumeGrid>`，list[0] = 主世界（obj_id = -1），list[1..N] = 物体。
/// 渲染器遍历 `list` 生成 GridDesc 数组。
#[derive(Debug, Default)]
pub struct Volumes {
  pub list: Vec<VolumeGrid>,
}

impl Volumes {
  /// 构造：插入主世界 volume（obj_id=-1，identity transform）
  pub fn new(main_world: VolumeGrid) -> Self {
    let mut list = Vec::with_capacity(8);
    list.push(main_world);
    Self { list }
  }

  /// 主世界只读引用
  pub fn main(&self) -> &VolumeGrid {
    &self.list[0]
  }

  /// 主世界可变引用
  pub fn main_mut(&mut self) -> &mut VolumeGrid {
    &mut self.list[0]
  }

  /// 添加物体 volume，返回分配的 obj_id（= list 索引 - 1，即 0..N-1）
  pub fn add_object(&mut self, pos: Vec3, rot: Mat3, scale: f32) -> usize {
    let obj_id = self.list.len() as i32 - 1;
    let grid = VolumeGrid::new_object(obj_id, pos, rot, scale);
    self.list.push(grid);
    obj_id as usize
  }

  /// 按 obj_id 查物体只读引用（obj_id = 0..N-1 对应 list[1..N]）
  pub fn object(&self, obj_id: usize) -> Option<&VolumeGrid> {
    self.list.get(obj_id + 1)
  }

  /// 按 obj_id 查物体可变引用
  pub fn object_mut(&mut self, obj_id: usize) -> Option<&mut VolumeGrid> {
    self.list.get_mut(obj_id + 1)
  }

  /// volume 总数（含主世界）
  pub fn len(&self) -> usize {
    self.list.len()
  }

  pub fn is_empty(&self) -> bool {
    self.list.is_empty()
  }

  /// 所有 volume 只读切片
  pub fn all(&self) -> &[VolumeGrid] {
    &self.list
  }
}

/// 一次编辑产生的脏区域。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirtyEdit {
  pub chunk: ChunkCoord,
}

impl VolumeGrid {
  pub fn new() -> Self {
    Self::default()
  }

  /// 构造独立物体 volume：`obj_id` 由渲染器分配（>=0），`pos/rot/scale` 为世界变换。
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

  /// 体素编辑代数（派生数据重烘判据；palette 变化不计入）
  pub fn edit_generation(&self) -> u64 {
    self.edit_generation
  }

  /// 取走某 chunk 的编辑 AABB（闭开区间 `[lo, hi)`，世界 voxel 坐标）。
  /// None = 非体素编辑标脏（如批量导入）→ 消费方回退到 chunk 包围盒。
  pub fn take_edit_aabb(&mut self, chunk: ChunkCoord) -> Option<(IVec3, IVec3)> {
    self.edit_aabbs.remove(&chunk)
  }

  /// 记录单格编辑（同 chunk 取并集）
  fn note_edit(&mut self, chunk: ChunkCoord, voxel: IVec3) {
    self.note_edit_box(chunk, voxel, voxel + IVec3::ONE);
  }

  /// 记录一块编辑区域（闭开 `[lo, hi)`，世界 voxel 坐标；同 chunk 取并集）
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

  /// GC 所有 chunk，回收累积的废弃节点（见 `ChunkTree::compact`）；chunk 间零共享，rayon 并行。
  pub fn compact_all(&mut self) {
    use rayon::prelude::*;
    self.chunks.par_iter_mut().for_each(|(_, tree)| tree.compact());
  }

  /// 挂载外部预构建的 chunk 树（大体积批量导入用）。
  /// 调用契约：`tree` 整体替换 `cc` 上的树并标脏，`applied_edits` 累加到编辑代数；空树由调用方跳过。
  pub fn mount_chunk_tree(&mut self, cc: ChunkCoord, mut tree: ChunkTree, applied_edits: u64) {
    debug_assert!(!tree.is_empty(), "mount_chunk_tree 不接受空树");
    // 外部整体替换：节点身份空间与上一个树无关 ⇒ 让 wire 层丢掉旧映射全量重建
    tree.mark_replaced();
    self.chunks.insert(cc, tree);
    self.dirty.mark_data(cc);
    self.edit_generation = self.edit_generation.wrapping_add(applied_edits);
  }

  pub fn get_voxel(&self, voxel: VoxelCoord) -> Option<PaletteId> {
    let chunk = voxel.chunk();
    let tree = self.chunks.get(&chunk)?;
    let local = voxel.in_chunk();
    tree.get_voxel(local.x, local.y, local.z)
  }

  /// 查询指定 level 的 brick 是否 uniform 同色
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

  /// 按 brick 边长查询三态（`extent` ∈ `LEVEL_EXTENT`，与 `fill_brick` 同一套粒度）。
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

  /// 按 **4³ brick** 批量写体素：`voxel` = brick 最小角（世界 voxel 坐标，对齐 4），
  /// `inside` 的 bit `i = z*16 + y*4 + x` 标出"落在笔触形状内"的体素（见 [`ChunkTree::set_brick_voxels`]）。
  /// 一次调用只标一次脏、只记一个编辑 AABB —— 逐格 `set_voxel` 是每格各标一次。
  /// 返回实际改变的体素数（0 = 无改动，此时不标脏）。
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

  /// 填充对齐 brick（extent ∈ {256,64,16,4,1}）；`voxel` 为 brick 最小角的世界 voxel 坐标。
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

  /// 写指定 ChunkCoord 下 level 2 brick (bx, by, bz) 的组件 ID
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

  /// 读指定 world voxel 所在 level 2 brick 的组件 ID
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
