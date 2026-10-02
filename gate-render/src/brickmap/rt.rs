//! **硬件光追的粗结构**：把一棵 chunk 树折成**一个**紧致 AABB（`BlasAabbGeometry` 的输入）。
//!
//! # 为什么是 AABB 而不是三角网格
//!
//! 这里**不做贪心网格化**。原因：三角网格要把同材质共面的上千个体素面**合并成一个 quad**，
//! 于是"一个可见体素面"这个着色量子就没了 —— 而 GI 的整套复用机制建在它上面
//! （`gi_key0`/`gi_key1` = 体素 + 面号 + 物体、`face_slots` 逐面去重、`gi_sec_slots` 跨帧面缓存），
//! 介质续行（`MediumRay`）也建在它上面。
//!
//! AABB 表**不改任何语义**：几何仍是同一份体素、同一个键、同一套介质。它换掉的只是
//! **"下一个有几何的块在哪"那一步** —— 从软件逐 chunk 读索引区换成固定功能硬件走 BVH。
//!
//! # 粒度：**每 chunk 一个盒子**（不是每 64³ 子块一个）
//!
//! CONSTRAINT: 盒子必须与 chunk **一一对应**，不能细到子块。理由在 shader 侧：ray query 只给出
//! **入口 `t`**，调用方拿不到该盒的出口；于是"下钻完就前进"只能在**出口可算**的前提下成立 ——
//! chunk 盒的出口可以用 slab 公式算出来（`trace.wesl::chunk_exit_t`），子块盒的出口算不出来。
//! 若把子块盒放进 BLAS，射线在 chunk A 的子块盒里没找到几何时会**前进到 A 的出口**，
//! 从而**跳过** chunk B 里更近的几何（两个子块盒的 t 区间可以互相交叠，而两个 chunk 盒不会）。
//!
//! 代价是剔除比子块粒度粗：射线擦过 chunk 盒但错过实际几何时，白做一次下钻。
//! 收益是**不需要 `primitive_index`**（调用方只消费"最近的块 + 入口 t"），于是空子块可以
//! 直接不进表，也就不用造任何"替身盒"。
//!
//! # 占用判据（两路，`mask` 的语义是"偏离节点默认色"）
//!
//!   · bit 置位 ⇒ 显式子块 ⇒ **保守算占用**（里面可能全是空气，交给软件下钻去发现）；
//!   · bit 清零 ⇒ 取根的隐式色，非空气才算占用。
//!
//! 保守多收的代价只是"软件进去发现是空的"，**不会漏几何**；而少收会直接把几何挖掉。

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

/// AABB 表里一个盒子的字节数：两个 `vec3<f32>`（min/max）。
///
/// 必须 `>= wgpu::AABB_GEOMETRY_MIN_STRIDE` 且是 8 的倍数 —— 24 同时满足两者，
/// 且与 `BlasAabbGeometry::stride` 用的是同一个值。
pub const AABB_STRIDE: u64 = 24;

/// 一个 chunk 的粗结构：**被占用的 64³ 子块的并集盒**（格空间）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChunkAabbs {
  /// 格空间 AABB 的最小角。
  pub min: Vec3,
  /// 格空间 AABB 的最大角（`min < max` 当且仅当 `occupied`）。
  pub max: Vec3,
  /// 是否有任何被占用的子块。`false` ⇒ 整块空气 ⇒ 不必建 BLAS。
  pub occupied: bool,
}

impl Default for ChunkAabbs {
  fn default() -> Self {
    Self { min: Vec3::ZERO, max: Vec3::ZERO, occupied: false }
  }
}

impl ChunkAabbs {
  /// 盒子的扁平布局（与 GPU 缓冲逐字节同布局）：`[minx,miny,minz, maxx,maxy,maxz]`。
  pub fn to_flat(&self) -> [f32; 6] {
    [self.min.x, self.min.y, self.min.z, self.max.x, self.max.y, self.max.z]
  }

  /// 图元数（恒 1；见文件头"每 chunk 一个盒子"）。
  pub fn len(&self) -> usize {
    1
  }

  /// **没有一个子块被占用**（整块空气）⇒ 不必建 BLAS。
  pub fn is_empty(&self) -> bool {
    !self.occupied
  }
}

/// 子块下标 → 轴向格子 `(ix, iy, iz)`，各 0..4。
///
/// **必须**是 `gate_voxel::child_linear_idx` 的逆 —— 该函数是掩码位序的**唯一权威**
/// （`z*16 + y*4 + x`：x 在低位、z 在最高位）。
#[inline]
pub fn slab_axes(i: u32) -> (u32, u32, u32) {
  debug_assert!(i < 64);
  (i & 3, (i >> 2) & 3, (i >> 4) & 3)
}

/// 轴向格子 → 子块下标（[`slab_axes`] 的逆）。
///
/// 直接转发 [`gate_voxel::child_linear_idx`]：**不在这里重写式子** —— 两份实现会漂移，
/// 而漂移的后果是静默的（几何搬到错位置）。
#[inline]
pub fn slab_index(ix: u32, iy: u32, iz: u32) -> u32 {
  gate_voxel::child_linear_idx(ix as i32, iy as i32, iz as i32)
}

/// 把一棵 chunk 树折成**一个**并集盒。
///
/// `chunk_min` = 该 chunk 在**该 volume 自己的格空间**里的最小角（体素单位）；`scale` = 该 volume 的
/// `Grid::scale`（远场级 > 1）。返回的是**格空间**的 AABB —— 与 shader 里 `trace_grid` 的局部坐标
/// 同空间（局部变换在 `trace_grid` 入口做，这里不重复）。
pub fn gather_chunk_aabbs(tree: &ChunkTree, chunk_min: Vec3, scale: f32) -> ChunkAabbs {
  // 占用判据取**根节点**（`mask` + `palette`）：wire 里 root 那一层就是这两个字（`serialize_into`），
  // shader 也照这么读 ⇒ 这里必须同源。**不能用 `root_palette()`** —— 那个字段只承载"整 chunk 单色
  // （零节点）"的规范形，而有节点时 bit=0 子块的隐式色是 `nodes[0].palette`（两者可以不同）⇒ 用字段会
  // 把 bit=0 的子块判错；而且 `ChunkTree::proxy` **不保留**该字段 ⇒ 档位一变 AABB 表就跟着变，而它必须
  // 与档位无关（见 `BrickMapBuilder::relayout_resident_tree` 的 WHY）。
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
      continue; // 空子块：不参与并集
    }
    let mn = block_min(chunk_min, i, scale);
    lo = lo.min(mn);
    hi = hi.max(mn + s);
  }
  if lo.x > hi.x {
    return ChunkAabbs::default(); // 一个占用子块都没有
  }
  ChunkAabbs { min: lo, max: hi, occupied: true }
}

/// 一个 64³ 子块在该 volume 格空间里的边长。
#[inline]
fn step(scale: f32) -> f32 {
  (CHUNK_SIZE / 4) as f32 * scale
}

/// 第 `i` 个子块在该 volume 格空间里的最小角。
#[inline]
fn block_min(chunk_min: Vec3, i: u32, scale: f32) -> Vec3 {
  let (ix, iy, iz) = slab_axes(i);
  let s = step(scale);
  chunk_min + Vec3::new(ix as f32 * s, iy as f32 * s, iz as f32 * s)
}

// ============================================================================
// GPU 侧：每 chunk 一张 BLAS + 一棵 TLAS
// ============================================================================

/// TLAS 实例槽上限（= 主世界常驻块的编址空间）。
///
/// 一次开到位、之后不再重建：`Tlas` 被 BG1 绑着，换一个 `Tlas` 对象等于整张 bind group 失效。
/// 空槽是 `None` ⇒ `build_acceleration_structures` 会把它们压掉 ⇒ 不占实例数、也不参与求交。
pub const RT_MAX_INSTANCES: u32 = 65536;

/// 卸下的 BLAS 延迟多少帧才真正丢弃（墓地）。
///
/// CONSTRAINT: 不能就地 `drop`。`Blas` 的 `Drop` 会立即销毁加速结构，而**上一帧建出的 TLAS
/// 可能仍在 GPU 上被遍历**（绑定它的实例表里还握着对这张 BLAS 的引用）⇒ 驱动访问已释放的加速结构。
/// 等这么多帧再丢，任何可能引用它的提交都已经执行完。取 8 帧（≈0.13 s）远大于排队深度。
const RT_FREE_DELAY_FRAMES: u64 = 8;

/// 实例变换 = 恒等：AABB 本来就在该 volume 的格空间里（主世界 = 世界空间），
/// 局部变换由 `trace_grid` 在入出口做一次。
const IDENTITY_3X4: [f32; 12] = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];

/// 实例掩码：不筛（shader 侧不按 mask 过滤）。
const RT_INSTANCE_MASK: u8 = 0xFF;

/// chunk 坐标 → TLAS 实例的 `custom_data`（24 位）。
///
/// 取三个坐标的**低 8 位**：窗口 ≤ 64 宽（`CHUNK_INDEX_CAP`）⇒ shader 用
/// `(custom8 - (index_origin & 0xFF)) & 0xFF` 还原出的相对下标是**唯一**的（差值 < 128 时模 256 无歧义）。
/// 用绝对坐标的低位而不是窗口相对槽号，是为了**窗口平移时不必重写任何实例**（M6 每跨一个 chunk 就平移一次）。
#[inline]
pub fn rt_custom_index(c: ChunkCoord) -> u32 {
  ((c.0.x as u32) & 0xFF) | (((c.0.y as u32) & 0xFF) << 8) | (((c.0.z as u32) & 0xFF) << 16)
}

/// 还原式的前提：相对下标 < 128（模 256 才无歧义）⇒ 窗口每轴不超过 64。
const _: () =
  assert!(super::wire::CHUNK_INDEX_CAP <= 64, "窗口每轴必须 ≤ 64，否则 custom_data 的低字节还原会撞");

/// 是否走硬件光追：编译期开关打开 **且** 设备支持 ray query。
///
/// 本仓**不往 `WgpuSettings.features` 里加任何东西**（Bevy 默认 `Functionality` 档 ⇒ 支持 RT 的
/// adapter 上这一位已经自动打开）⇒ 这里只**查询**，不请求 —— 于是老卡照常建起设备、自动落到软件 DDA。
pub fn rt_enabled(device: &RenderDevice) -> bool {
  crate::brickmap::consts::RT_RAY_QUERY
    && device
      .features()
      .contains(bevy::render::render_resource::WgpuFeatures::EXPERIMENTAL_RAY_QUERY)
}

/// 一个常驻 chunk 的硬件光追部分。
struct ChunkBlas {
  blas: Blas,
  /// 盒子的扁平布局缓冲（6 个 f32 = 24 B）。
  /// CONSTRAINT: BLAS 只在 `Build` 时读它一次 ⇒ 建后不得改写内容（本仓从不改写，换块时整块丢弃）。
  aabb: Buffer,
}

/// 主世界的硬件光追粗结构（render world）：窗口内每个常驻 chunk 一个 TLAS 实例。
///
/// 只覆盖**主世界**（`Grid::vol == 0`）：物体 / 远场级的 chunk 坐标空间不同（有各自的 `rot/pos/scale`），
/// `trace_grid` 对它们仍走软件 DDA。
#[derive(Resource)]
pub struct RtScene {
  enabled: bool,
  tlas: Tlas,
  /// 槽 → chunk（`None` = 空槽）
  slots: Vec<Option<ChunkCoord>>,
  free_slots: Vec<u32>,
  by_chunk: HashMap<ChunkCoord, u32>,
  blas: HashMap<ChunkCoord, ChunkBlas>,
  /// 新建成、还没记进命令编码器的 BLAS
  pending_blas: Vec<ChunkCoord>,
  /// 实例表里改过的槽（本帧要把它们写回 `tlas`）
  dirty_slots: Vec<u32>,
  /// 已卸下的 BLAS 的**墓地**：`(丢弃帧, 资源)`。见 [`RT_FREE_DELAY_FRAMES`]。
  graveyard: Vec<(u64, ChunkBlas)>,
  /// **本帧没来得及建 BLAS 的常驻块**（调用方按每帧额度截断）。下一帧优先补。
  pub deferred: Vec<ChunkCoord>,
  /// 帧号（`tick` 推进）：墓地的到期判定用。
  frame: u64,
  /// 还没 build 过 ⇒ [`Self::record`] 必须无条件跑一次，哪怕实例表没变。
  ///
  /// CONSTRAINT: 不能省。`TLAS` 在 `PrepareBindGroups` 就被绑进 BG1，而 wgpu 在**提交时**会按
  /// 命令顺序校验"用到的 TLAS 已经建过"（`ValidateAsActionsError::UsedUnbuiltTlas`）。
  never_built: bool,
}

impl RtScene {
  pub fn new(enabled: bool, device: &RenderDevice) -> Self {
    let tlas = device.wgpu_device().create_tlas(&CreateTlasDescriptor {
      label: Some("gate_rt_tlas"),
      max_instances: RT_MAX_INSTANCES,
      // 实例表**每帧都可能变**（流式挂载 / 卸载）⇒ 构建速度优先于遍历速度。
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

  /// 推进一帧并释放墓地到期的 BLAS。调用方每帧（`prepare_rt_scene`）调一次。
  pub fn tick(&mut self) {
    self.frame = self.frame.wrapping_add(1);
    let now = self.frame;
    self.graveyard.retain(|(due, _)| *due > now);
  }

  pub fn is_enabled(&self) -> bool {
    self.enabled
  }

  /// TLAS 绑定资源（BG1 binding(11)）。
  pub fn tlas(&self) -> &Tlas {
    &self.tlas
  }

  /// 当前实例数（= 主世界常驻且被 RT 覆盖的 chunk 数）。
  pub fn instance_count(&self) -> usize {
    self.by_chunk.len()
  }

  /// 当前被 RT 覆盖的 chunk（全量对照用）。
  pub fn chunks(&self) -> impl Iterator<Item = ChunkCoord> + '_ {
    self.by_chunk.keys().copied()
  }

  /// 该 chunk 是否已有实例。
  pub fn contains(&self, c: ChunkCoord) -> bool {
    self.by_chunk.contains_key(&c)
  }

  /// 槽位耗尽（TLAS 开小了）：调用方据此报一条 warn，不静默丢几何。
  pub fn slots_exhausted(&self) -> bool {
    self.free_slots.is_empty() && self.slots.len() == self.by_chunk.len()
  }

  /// 装一个常驻 chunk（建 BLAS + 占槽）。已在 / 空盒 / 无槽 ⇒ 无操作并返回 false。
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
    // 整块空气：建了也永远打不中 ⇒ 省一张 BLAS。
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
      // BLAS_INPUT 是 wgpu 建加速结构时对几何输入缓冲的硬性要求（见 wgpu-core 的 `check_usage`）。
      usage: BufferUsages::BLAS_INPUT | BufferUsages::COPY_DST,
      mapped_at_creation: false,
    });
    queue.write_buffer(&aabb, 0, f32_bytes(&flat));
    let blas = device.wgpu_device().create_blas(
      &CreateBlasDescriptor {
        label: Some("gate_rt_blas"),
        // 几何长期不变（改一块 = 整块重建）⇒ 遍历速度优先。
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

  /// 卸一个 chunk：清槽、丢弃 BLAS。不在 ⇒ 无操作。
  pub fn remove_chunk(&mut self, coord: ChunkCoord) -> bool {
    let Some(slot) = self.by_chunk.remove(&coord) else {
      return false;
    };
    self.slots[slot as usize] = None;
    self.free_slots.push(slot);
    // 待建队列里可能还留着这个坐标（同帧内先挂载后换出）⇒ 一并摘掉，
    // 否则 `record` drain 时 `blas` 里已经没有它了。
    self.pending_blas.retain(|c| c != &coord);
    if let Some(b) = self.blas.remove(&coord) {
      // 进墓地而不是就地 drop（见 [`RT_FREE_DELAY_FRAMES`]）
      self.graveyard.push((self.frame + RT_FREE_DELAY_FRAMES, b));
    }
    self.dirty_slots.push(slot);
    true
  }

  /// 世界整体换掉（全量上传）：清空一切实例与 BLAS。
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
    // 清空后至少还要 build 一次（否则 TLAS 停在上一份实例表上）
    self.never_built = true;
  }

  /// 把本帧的实例表改动与新建 BLAS 记进命令编码器。返回是否记了什么东西。
  ///
  /// 顺序必须在**同一次调用**里：wgpu 要求被 TLAS 引用的 BLAS 在**同一批次或更早**建好
  /// （`build_acceleration_structures` 的两组参数就是为此）。
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
      // 尺寸描述符要**比 `BlasBuildEntry` 活得久**（后者持有 `&`）⇒ 先建满整张表，再建几何。
      // CONSTRAINT: 只认**还在 `blas` 里**的坐标。`pending_blas` 是"待建"队列，而同帧内
      // "先挂载后换出"的块会被 `remove_chunk` 从 `blas` 摘掉（见那里的 `retain`）。
      let live: Vec<ChunkCoord> = self
        .pending_blas
        .iter()
        .copied()
        .filter(|c| self.blas.contains_key(c))
        .collect();
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

/// AABB 几何的尺寸描述符（`OPAQUE` 是必须的：不带它 naga 侧拿不到任何 hit）。
fn aabb_size_desc(prims: u32) -> wgpu::BlasAABBGeometrySizeDescriptor {
  wgpu::BlasAABBGeometrySizeDescriptor {
    primitive_count: prims,
    flags: AccelerationStructureGeometryFlags::OPAQUE,
  }
}

/// `&[f32]` → `&[u8]`（`queue.write_buffer` 的入参）。
fn f32_bytes(v: &[f32]) -> &[u8] {
  unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

#[cfg(test)]
mod tests {
  use super::*;
  use glam::IVec3;

  fn tree_with(blocks: &[(u32, PaletteId)]) -> ChunkTree {
    // 用 `fill_brick` 逐块写：extent 64 正好是一个子块，且落在 LEVEL_EXTENT 上。
    let mut t = ChunkTree::empty();
    for &(i, pal) in blocks {
      let (ix, iy, iz) = slab_axes(i);
      let s = CHUNK_SIZE / 4;
      t.fill_brick([ix as i32 * s, iy as i32 * s, iz as i32 * s], s, pal);
    }
    t
  }

  /// 轴序必须自洽：`slab_index(slab_axes(i)) == i`，且与 `gate_voxel::child_linear_idx` 同形。
  #[test]
  fn slab_axes_roundtrip() {
    for i in 0..64u32 {
      let (ix, iy, iz) = slab_axes(i);
      assert_eq!(slab_index(ix, iy, iz), i, "下标 {i} 往返不一致");
      assert_eq!(i, iz * 16 + iy * 4 + ix, "下标 {i} 不是 z*16 + y*4 + x");
    }
    assert_eq!(slab_index(0, 1, 0), 4, "y 占 4 这一位");
    assert_eq!(slab_index(0, 0, 1), 16, "z 占 16 这一位");
  }

  /// 空树 ⇒ 未占用（不建 BLAS）。
  #[test]
  fn empty_tree_is_unoccupied() {
    let t = ChunkTree::empty();
    assert!(t.root_palette().is_air(), "前提：空树的根是空气");
    let a = gather_chunk_aabbs(&t, Vec3::ZERO, 1.0);
    assert!(a.is_empty(), "空树不该被标记为占用");
  }

  /// 盒子必须**合法**（`min < max`，三个轴都要）—— 反向盒 / 退化盒在 Vulkan 下是未定义行为。
  #[test]
  fn box_is_valid_and_inside_the_chunk() {
    let t = tree_with(&[(slab_index(0, 0, 0), PaletteId(5))]);
    let chunk_min = Vec3::new(512.0, -256.0, 0.0);
    let a = gather_chunk_aabbs(&t, chunk_min, 1.0);
    assert!(a.occupied);
    assert!(a.min.cmplt(a.max).all(), "三个轴都必须 min < max");
    let s = CHUNK_SIZE as f32;
    assert!(a.min.cmpge(chunk_min).all(), "盒子越出 chunk 下界");
    assert!(a.max.cmple(chunk_min + Vec3::splat(s)).all(), "盒子越出 chunk 上界");
  }

  /// 盒子必须**紧致**（= 被占用子块的并集），不是整个 chunk —— 否则剔除退化、每条射线都白下钻。
  #[test]
  fn box_is_tight_not_the_whole_chunk() {
    let idx = slab_index(1, 2, 3);
    let t = tree_with(&[(idx, PaletteId(5))]);
    let a = gather_chunk_aabbs(&t, Vec3::ZERO, 1.0);
    let s = (CHUNK_SIZE / 4) as f32;
    let (ix, iy, iz) = slab_axes(idx);
    assert_eq!(a.min, Vec3::new(ix as f32 * s, iy as f32 * s, iz as f32 * s), "起点应是该子块");
    assert_eq!(a.max, a.min + Vec3::splat(s), "恰好一个子块大小");
  }

  /// **保守性**：每个有实心体素的子块都必须落在并集盒内（覆盖不足 = 会挖掉几何）。
  #[test]
  fn box_covers_every_solid_slab() {
    let mut t = ChunkTree::empty();
    for ix in 0..4 {
      for iz in 0..4 {
        t.fill_brick([ix * 64, 0, iz * 64], 64, PaletteId(2));
      }
    }
    t.fill_brick([0, 64, 0], 64, PaletteId(7));
    let a = gather_chunk_aabbs(&t, Vec3::ZERO, 1.0);
    assert!(a.occupied);
    let s = CHUNK_SIZE / 4;
    for iy in 0..4 {
      for iz in 0..4 {
        for ix in 0..4 {
          let c = [ix * s + s / 2, iy * s + s / 2, iz * s + s / 2];
          if !t.get_voxel(c[0], c[1], c[2]).is_some_and(|p| !p.is_air()) {
            continue;
          }
          let v = Vec3::new(ix as f32, iy as f32, iz as f32) * s as f32;
          let vmax = v + Vec3::splat(s as f32);
          assert!(a.min.cmple(v).all() && a.max.cmpge(vmax).all(),
            "子块 ({ix},{iy},{iz}) 有实心体素，但没被并集盒覆盖");
        }
      }
    }
  }

  /// **AABB 与档位无关**：`ChunkTree::proxy` 在 `keep` 以上逐字保留根节点的 `mask` / `palette`，
  /// 而占用判据只读这两样 ⇒ proxy 树与全树的盒子必须**完全相同**。这条是
  /// [`crate::brickmap::builder::BrickMapBuilder::relayout_resident_tree`]（升级不重建 BLAS）的前提；
  /// 也是 `gather_chunk_aabbs` 必须读 `node_view(0).palette` 而不是 `root_palette()` 字段的原因。
  #[test]
  fn aabbs_are_invariant_across_proxy_levels() {
    let mut t = ChunkTree::uniform(PaletteId(2));
    t.fill_brick([0, 0, 0], 64, PaletteId(5));
    t.fill_brick([64, 64, 64], 64, PaletteId(7));
    let full = gather_chunk_aabbs(&t, Vec3::ZERO, 1.0);
    assert!(full.occupied);
    assert!(t.node_view(0).is_some_and(|v| !v.palette.is_air()), "前提：根的隐式色非空气");
    for keep in [16, 64] {
      let p = gather_chunk_aabbs(&t.proxy(keep), Vec3::ZERO, 1.0);
      assert_eq!(p, full, "proxy({keep}) 的盒子与全树不一致");
    }
  }

  /// 整块实心（无节点、root palette 非空气）⇒ 盒子 = 整个 chunk。
  #[test]
  fn uniform_solid_fills_the_chunk() {
    let t = ChunkTree::uniform(PaletteId(3));
    let a = gather_chunk_aabbs(&t, Vec3::ZERO, 1.0);
    assert!(a.occupied);
    assert_eq!(a.min, Vec3::ZERO);
    assert_eq!(a.max, Vec3::splat(CHUNK_SIZE as f32));
  }

  /// 远场级（`scale > 1`）：盒子的边长要乘 `scale`（一个 64³ 子块 = `64·scale` 个世界体素）。
  #[test]
  fn far_level_scales_box_size() {
    let t = tree_with(&[(slab_index(0, 0, 0), PaletteId(5))]);
    let a = gather_chunk_aabbs(&t, Vec3::new(1024.0, 0.0, 0.0), 16.0);
    assert!(a.occupied);
    assert_eq!(a.min, Vec3::new(1024.0, 0.0, 0.0), "chunk_min 要透传");
    assert_eq!(a.max, a.min + Vec3::splat(64.0 * 16.0), "边长 = 64·scale");
  }

  /// `custom_data` 的低 8 位要能在 shader 侧还原出**唯一**的窗口相对下标（差 < 128 ⇒ 模 256 无歧义）。
  #[test]
  fn custom_index_roundtrips_through_low_bytes() {
    let origin = IVec3::new(-3, 250, -130);
    for rel in [(0, 0, 0), (1, 2, 3), (63, 63, 63), (0, 63, 1)] {
      let c = ChunkCoord(origin + IVec3::new(rel.0, rel.1, rel.2));
      let w = rt_custom_index(c);
      assert!(w < (1 << 24), "custom_data 必须 < 2^24（wgpu 的硬校验）");
      let back = (
        ((w & 0xFF) as i32 - (origin.x & 0xFF)) & 0xFF,
        (((w >> 8) & 0xFF) as i32 - (origin.y & 0xFF)) & 0xFF,
        (((w >> 16) & 0xFF) as i32 - (origin.z & 0xFF)) & 0xFF,
      );
      assert_eq!(back, rel, "chunk {c:?} 的相对下标还原不一致");
    }
  }
}
