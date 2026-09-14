//! Volume + Volumes：主世界 / 独立物体的统一容器（Phase 0 + Phase 3）
//!
//! `VolumeGrid` = 单个体素 volume（主世界 = identity transform + 无界 chunk HashMap；
//! 独立物体 = 任意 transform + 通常 1~少数 chunk）。`Volumes` = `Vec<VolumeGrid>` 容器，
//! list[0] 永远是主世界（obj_id = -1），list[1..N] 是物体（obj_id = 0..N-1）。
//! Phase 3 OBJ→Volume 统一：obj.rs 三段独立管道被吸收为 Volumes 中的物体 Volume。

use std::collections::HashMap;

use glam::{IVec3, Mat3, Vec3};

use crate::chunk_tree::ChunkTree;
use crate::coords::{CHUNK_SIZE, ChunkCoord, VoxelCoord};
use crate::dirty::DirtyTracker;
use crate::palette::Palette;

/// 组件层：level 2 brick = 16³ = 4096 体素 = 一个组件 cell
/// 每 chunk = (256/16)³ = 16³ = 4096 个 level 2 brick = 4096 个 u16
pub const COMP_BRICKS_PER_CHUNK: usize = 16 * 16 * 16;
pub const COMP_BRICK_EXTENT: i32 = 16;

/// 主世界 / 独立物体的统一容器。
///
/// 主世界 = identity transform + 无界 chunk HashMap（obj_id = -1）。
/// 独立物体 = 任意 transform + 通常 1~少数 chunk（obj_id = 0..N-1）。
/// Phase 3 OBJ→Volume 统一后，OBJ 不再有独立 ObjScene/RenderObj/GpuObjPool 三段
/// 管道，而是作为 `Volumes.list[1..N]` 中的普通 `VolumeGrid`，走与主世界完全相同的
/// `DirtyTracker` → `BrickMapBuilder::update_chunk` → `UploadSnapshot` 增量上传路径。
#[derive(Debug, Clone)]
pub struct VolumeGrid {
  chunks: HashMap<ChunkCoord, ChunkTree>,
  palette: Palette,
  pub dirty: DirtyTracker,
  comp_layer: HashMap<ChunkCoord, Box<[u16; COMP_BRICKS_PER_CHUNK]>>,
  state_table: Vec<[u32; 4]>,
  pub state_dirty: bool,
  /// 物体变换（主世界 = identity：pos=0/rot=identity/scale=1）。
  /// 渲染时通过 GridDesc（§2.6）传到 shader，DDA 局部变换 + 世界 AABB 由它推导。
  pub transform: VolumeTransform,
  /// 渲染器分配的 obj_id（主世界 = -1；物体 = 0..N-1，对应 `Volumes.list[1..]` 索引）。
  pub obj_id: i32,
  /// 体素编辑单调代数（每次实际改变体素的 set/clear/fill +1；noop 同色重复写不递增）。
  /// 渲染侧探针烘焙等派生数据据此判断「自上次构建以来世界是否被编辑」→ 触发重烘。
  /// palette 变化不计入（探针位置不变，辐射度由射线更新 EMA 自然收敛）。
  edit_generation: u64,
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
    }
  }
}

/// 物体变换：`world = pos + rot · (local · scale)`。
/// 主世界 = identity（pos=0、rot=identity、scale=1），与 GridDesc 对齐。
/// rot 列向量约定与 glam Mat3 一致（x_axis/y_axis/z_axis 即列）。
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
  pub const IDENTITY: Self = Self {
    pos: Vec3::ZERO,
    rot: Mat3::IDENTITY,
    scale: 1.0,
  };

  pub fn new(pos: Vec3, rot: Mat3, scale: f32) -> Self {
    Self { pos, rot, scale }
  }

  /// 局部 [0,256]³·scale 经旋转平移后的世界 AABB 外包（剔除用，与 obj.rs 旧 world_aabb 同型）。
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

/// 全场景容器：`Vec<VolumeGrid>`，list[0] = 主世界（obj_id=-1），list[1..N] = 物体。
///
/// Phase 3 OBJ→Volume 统一入口：OBJ 不再有独立管道，而是作为 `Volumes.list[1..N]`
/// 中的普通 `VolumeGrid`，走与主世界完全相同的 dirty → builder → upload 路径。
/// 渲染器遍历 `list` 生成 GridDesc 数组，shader `trace_scene` 无 kind 分支。
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

/// 一次编辑产生的脏区域（Phase 1 上传管道用）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirtyEdit {
  pub chunk: ChunkCoord,
}

impl VolumeGrid {
  pub fn new() -> Self {
    Self::default()
  }

  /// 构造独立物体 volume（OBJ→Volume 统一入口）。
  /// `obj_id` 由渲染器分配（>=0）；`pos/rot/scale` 为世界变换。
  pub fn new_object(obj_id: i32, pos: Vec3, rot: Mat3, scale: f32) -> Self {
    Self {
      obj_id,
      transform: VolumeTransform::new(pos, rot, scale),
      ..Self::default()
    }
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

  /// GC 所有 chunk：回收编辑过程中累积的废弃节点（见 [`ChunkTree::compact`]）。
  ///
  /// chunk 间零共享 → rayon 并行；导入后全场景百万级节点，串行 ~1s → 并行 ~0.1s。
  pub fn compact_all(&mut self) {
    use rayon::prelude::*;
    self
      .chunks
      .par_iter_mut()
      .for_each(|(_, tree)| tree.compact());
  }

  /// 挂载外部预构建的 chunk 树（大体积批量导入专用）。
  ///
  /// 用 `tree` 整体替换 `cc` 上的树，标脏并按 `applied_edits` 累加编辑代数
  /// （保持「每次实际改变体素 +1」语义：导入方统计并行建树时真实生效的写入数）。
  ///
  /// # 调用契约
  /// `tree` 的体素集合即该 chunk 的最终状态——同 coord 已有非空树时会被
  /// 整体覆盖（导入方保证 fresh grid 或接受覆盖）。空树不挂载（避免污染
  /// chunk 表 + 空转脏标记），由调用方跳过。
  pub fn mount_chunk_tree(&mut self, cc: ChunkCoord, tree: ChunkTree, applied_edits: u64) {
    debug_assert!(!tree.is_empty(), "mount_chunk_tree 不接受空树");
    self.chunks.insert(cc, tree);
    self.dirty.mark_data(cc);
    self.edit_generation = self.edit_generation.wrapping_add(applied_edits);
  }

  // =========================================================================
  // 体素查询
  // =========================================================================

  pub fn get_voxel(&self, voxel: VoxelCoord) -> Option<u8> {
    let chunk = voxel.chunk();
    let tree = self.chunks.get(&chunk)?;
    let local = voxel.in_chunk();
    tree.get_voxel(local.x, local.y, local.z)
  }

  /// 查询指定 level 的 brick 是否 uniform 同色
  pub fn get_uniform(&self, voxel: VoxelCoord, level: u8) -> Option<u8> {
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

  // =========================================================================
  // 体素编辑
  // =========================================================================

  pub fn set_voxel(&mut self, voxel: VoxelCoord, palette: u8) -> Option<DirtyEdit> {
    let chunk = voxel.chunk();
    let local = voxel.in_chunk();

    let tree = self.chunks.entry(chunk).or_insert_with(ChunkTree::empty);
    if tree.set_voxel(local.x, local.y, local.z, palette) {
      self.dirty.mark_data(chunk);
      self.edit_generation = self.edit_generation.wrapping_add(1);
      Some(DirtyEdit { chunk })
    } else {
      None
    }
  }

  pub fn set_voxel_ivec3(&mut self, pos: IVec3, palette: u8) -> Option<DirtyEdit> {
    self.set_voxel(VoxelCoord::from_ivec3(pos), palette)
  }

  /// 填充对齐 brick（extent ∈ {256,64,16,4,1}，Douglas wire Uniform 节点同构）。
  ///
  /// 大体积均匀填充专用：树路径 O(depth)，不逐体素分裂（见 [`ChunkTree::fill_brick`]）。
  /// `voxel` 为 brick 最小角的世界 voxel 坐标。
  pub fn fill_brick(&mut self, voxel: IVec3, extent: i32, palette: u8) -> Option<DirtyEdit> {
    let chunk = voxel.div_euclid(IVec3::splat(CHUNK_SIZE));
    let local = voxel.rem_euclid(IVec3::splat(CHUNK_SIZE));
    let cc = ChunkCoord(chunk);
    let changed = match self.chunks.get_mut(&cc) {
      Some(tree) => tree.fill_brick([local.x, local.y, local.z], extent, palette),
      None => {
        if palette == 0 {
          return None; // 空 chunk 填空气 = noop，不建 chunk
        }
        let tree = self.chunks.entry(cc).or_insert_with(ChunkTree::empty);
        tree.fill_brick([local.x, local.y, local.z], extent, palette)
      }
    };
    if changed {
      self.dirty.mark_data(cc);
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
      // 如果 chunk 全空，可以删除
      if tree.is_empty() {
        self.chunks.remove(&chunk);
      }
      self.dirty.mark_data(chunk);
      self.edit_generation = self.edit_generation.wrapping_add(1);
      Some(DirtyEdit { chunk })
    } else {
      None
    }
  }

  pub fn batch_edit(&mut self, ops: impl IntoIterator<Item = (VoxelCoord, u8)>) -> usize {
    let mut applied = 0;
    for (voxel, palette) in ops {
      if self.set_voxel(voxel, palette).is_some() {
        applied += 1;
      }
    }
    applied
  }

  // =========================================================================
  // 组件层（level 2 brick = 16³ per cell）
  // =========================================================================

  /// 写指定 ChunkCoord 下 level 2 brick (bx, by, bz) 的组件 ID
  pub fn set_comp(&mut self, chunk: ChunkCoord, bx: u32, by: u32, bz: u32, comp_id: u16) {
    let arr = self
      .comp_layer
      .entry(chunk)
      .or_insert_with(|| Box::new([0u16; COMP_BRICKS_PER_CHUNK]));
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

  // =========================================================================
  // StateTable
  // =========================================================================

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
    self
      .state_table
      .get(id as usize)
      .map(|a| a[word])
      .unwrap_or(0)
  }

  pub fn state_table_bytes(&self) -> &[u8] {
    let slice = &self.state_table[..];
    unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, slice.len() * 16) }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn set_get_single_voxel() {
    let mut grid = VolumeGrid::new();
    assert_eq!(grid.get_voxel(VoxelCoord::new(5, 6, 7)), None);
    assert!(grid.set_voxel(VoxelCoord::new(5, 6, 7), 1).is_some());
    assert_eq!(grid.get_voxel(VoxelCoord::new(5, 6, 7)), Some(1));
  }

  #[test]
  fn edit_generation_tracks_real_changes() {
    let mut grid = VolumeGrid::new();
    assert_eq!(grid.edit_generation(), 0);
    // 实际写入 +1
    assert!(grid.set_voxel(VoxelCoord::new(1, 2, 3), 5).is_some());
    assert_eq!(grid.edit_generation(), 1);
    // 同色重复写 = noop，代数不变
    assert!(grid.set_voxel(VoxelCoord::new(1, 2, 3), 5).is_none());
    assert_eq!(grid.edit_generation(), 1);
    // fill_brick +1
    assert!(grid.fill_brick(IVec3::new(16, 0, 0), 4, 7).is_some());
    assert_eq!(grid.edit_generation(), 2);
    // clear +1；清空后 chunk 删除
    assert!(grid.clear_voxel(VoxelCoord::new(1, 2, 3)).is_some());
    assert_eq!(grid.edit_generation(), 3);
    // 清空气 = noop
    assert!(grid.clear_voxel(VoxelCoord::new(100, 100, 100)).is_none());
    assert_eq!(grid.edit_generation(), 3);
  }

  #[test]
  fn cross_chunk_editing() {
    let mut grid = VolumeGrid::new();
    // chunk 0: (255,255,255) 和 chunk 1: (256,0,0)
    assert!(grid.set_voxel(VoxelCoord::new(255, 255, 255), 1).is_some());
    assert!(grid.set_voxel(VoxelCoord::new(256, 0, 0), 2).is_some());
    assert_eq!(grid.chunk_count(), 2);
    assert_eq!(grid.get_voxel(VoxelCoord::new(255, 255, 255)), Some(1));
    assert_eq!(grid.get_voxel(VoxelCoord::new(256, 0, 0)), Some(2));
  }

  #[test]
  fn negative_coords_cross_chunk() {
    let mut grid = VolumeGrid::new();
    assert!(grid.set_voxel(VoxelCoord::new(-1, 0, 0), 3).is_some());
    assert_eq!(grid.get_voxel(VoxelCoord::new(-1, 0, 0)), Some(3));
    // 负 chunk
    let chunk = VoxelCoord::new(-1, 0, 0).chunk();
    assert_eq!(chunk, ChunkCoord::new(-1, 0, 0));
  }

  #[test]
  fn clear_removes_chunk_when_empty() {
    let mut grid = VolumeGrid::new();
    grid.set_voxel(VoxelCoord::new(10, 10, 10), 5);
    assert_eq!(grid.chunk_count(), 1);
    assert!(grid.clear_voxel(VoxelCoord::new(10, 10, 10)).is_some());
    assert_eq!(grid.chunk_count(), 0);
  }

  #[test]
  fn batch_edit_dedupes_dirty() {
    let mut grid = VolumeGrid::new();
    let ops: Vec<(VoxelCoord, u8)> = (0..100).map(|i| (VoxelCoord::new(i, 0, 0), 1u8)).collect();
    assert_eq!(grid.batch_edit(ops), 100);
    assert_eq!(grid.dirty.data_dirty_count(), 1); // 全在 chunk (0,0,0)
  }

  #[test]
  fn volumes_container_main_and_objects() {
    let main = VolumeGrid::new();
    assert_eq!(main.obj_id(), -1, "主世界 obj_id=-1");
    let mut vols = Volumes::new(main);
    assert_eq!(vols.len(), 1);
    assert_eq!(vols.main().obj_id(), -1);

    // 添加物体 0：identity + scale 2
    let id0 = vols.add_object(Vec3::new(100.0, 0.0, 0.0), Mat3::IDENTITY, 2.0);
    assert_eq!(id0, 0);
    assert_eq!(vols.len(), 2);
    let obj0 = vols.object(0).expect("物体 0 存在");
    assert_eq!(obj0.obj_id(), 0);
    assert_eq!(obj0.transform.pos, Vec3::new(100.0, 0.0, 0.0));
    assert_eq!(obj0.transform.scale, 2.0);

    // 添加物体 1
    let id1 = vols.add_object(Vec3::ZERO, Mat3::IDENTITY, 1.0);
    assert_eq!(id1, 1);
    assert_eq!(vols.len(), 3);

    // 可变访问
    vols
      .object_mut(0)
      .unwrap()
      .set_voxel_ivec3(IVec3::new(5, 5, 5), 3);
    assert_eq!(
      vols.object(0).unwrap().get_voxel(VoxelCoord::new(5, 5, 5)),
      Some(3)
    );

    // 主世界可变
    vols.main_mut().set_voxel_ivec3(IVec3::new(10, 10, 10), 7);
    assert_eq!(vols.main().get_voxel(VoxelCoord::new(10, 10, 10)), Some(7));

    // all() 切片
    assert_eq!(vols.all().len(), 3);
  }
}
