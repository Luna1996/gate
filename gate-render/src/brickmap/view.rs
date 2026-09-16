//! Brick Tree 软件遍历器：wire.rs §b_struct 寻址链的 CPU 独立实现（刻意不与 builder 共享
//! 代码，等价性测试的意义就在于两套实现互为对照）。
//!
//! 寻址链（3 步，全部 storage load）：① chunk 窗口：voxel → chunk（floor div 256）→ entry
//! （0 = 无 chunk）② chunk base → 根节点 fixed（mask_lo/mask_hi/palette，8B）③ mask bit 测试
//! → popcount 定位 child offset → 下钻（每层 1 次 load，uniform 子块零额外 load）。
//!
//! 语义与 `VolumeGrid::get_voxel` 严格一致：Some(palette 1..=255) / None（空）。

use glam::IVec3;

use super::wire::{BrickMapBuffers, CHUNK_INDEX_CAP, CHUNK_SIZE};

/// 稠密 chunk 窗口线性位置；窗口外返回 None
fn chunk_index_pos(origin: IVec3, dims: IVec3, chunk: IVec3) -> Option<usize> {
  let rel = chunk - origin;
  if rel.cmplt(IVec3::ZERO).any() || rel.cmpge(dims).any() {
    return None;
  }
  Some(
    (rel.x
      + rel.y * CHUNK_INDEX_CAP as i32
      + rel.z * CHUNK_INDEX_CAP as i32 * CHUNK_INDEX_CAP as i32) as usize,
  )
}

/// 读一个节点的 (mask, palette)。`node` 为 b_struct 内绝对字址。
/// palette word 低字节 = uniform 子块色（高字节 = LOD 子树多数色）
#[inline]
fn read_node(b_struct: &[u32], node: usize) -> (u64, u32) {
  let lo = b_struct[node] as u64;
  let hi = b_struct[node + 1] as u64;
  (hi << 32 | lo, b_struct[node + 2] & 0xFF)
}

/// 砖块图只读视图：持有与 GPU buffer 字节一致的缓冲区引用
pub struct BrickMapView<'a> {
  b_struct: &'a [u32],
  origin: IVec3,
  dims: IVec3,
}

impl<'a> BrickMapView<'a> {
  pub fn new(buffers: &'a BrickMapBuffers) -> Self {
    let g = &buffers.globals;
    Self {
      b_struct: &buffers.b_struct,
      origin: IVec3::new(g.index_origin_x, g.index_origin_y, g.index_origin_z),
      dims: IVec3::new(g.index_dims_x as i32, g.index_dims_y as i32, g.index_dims_z as i32),
    }
  }

  /// 从裸字切片构造（obj pool 拼接采样用：切片 = 某物体 b_struct 的连续窗口）
  pub fn from_parts(b_struct: &'a [u32], origin: IVec3, dims: IVec3) -> Self {
    Self { b_struct, origin, dims }
  }

  /// chunk 窗口查找 → DFS 树绝对字基址（0 = 无 chunk）
  /// 层次栈式 DDA（dda.rs trace_chunk_cpu）逐 chunk 调用，镜像 WGSL 窗口 entry 查找。
  pub(crate) fn chunk_base(&self, chunk: IVec3) -> Option<usize> {
    let ip = chunk_index_pos(self.origin, self.dims, chunk)?;
    let entry = self.b_struct[ip];
    if entry == 0 {
      return None;
    }
    Some(entry as usize - 1)
  }

  /// b_struct 字切片（层次遍历按绝对字址读节点 fixed 字 + child offset）
  pub(crate) fn b_struct(&self) -> &[u32] {
    self.b_struct
  }

  /// chunk 窗口 origin（chunk 单位；可为负）
  pub(crate) fn origin(&self) -> IVec3 {
    self.origin
  }

  /// chunk 窗口 dims（chunk 单位）
  pub(crate) fn dims(&self) -> IVec3 {
    self.dims
  }

  /// mask DDA 逐层下钻读单个最细格（1³）体素，O(分裂层数) = 最多 4 层
  /// 叶父层（level 3）inline 4 体素/word，故不存在 level 4 叶节点
  pub fn get_voxel(&self, voxel: IVec3) -> Option<u8> {
    let chunk = voxel.div_euclid(IVec3::splat(CHUNK_SIZE));
    let base = self.chunk_base(chunk)?;
    let mut local = voxel.rem_euclid(IVec3::splat(CHUNK_SIZE));
    let mut node = base;
    let mut extent = CHUNK_SIZE;
    loop {
      let (mask, pal) = read_node(self.b_struct, node);
      if mask == 0 {
        return (pal != 0).then_some(pal as u8);
      }
      let child_extent = extent >> 2;
      let ix = local.x / child_extent;
      let iy = local.y / child_extent;
      let iz = local.z / child_extent;
      let ci = (iz * 16 + iy * 4 + ix) as u64;
      let bit = 1u64 << ci;
      if mask & bit == 0 {
        return (pal != 0).then_some(pal as u8);
      }
      if child_extent == 1 {
        // 叶父层 inline 4 体素/word = b_struct[node + 3 + (ci >> 2)]，
        // palette = 该 word 的第 (ci & 3) 字节
        let w = self.b_struct[node + 3 + (ci >> 2) as usize];
        let leaf_pal = (w >> ((ci & 3) as u32 * 8)) & 0xFF;
        return (leaf_pal != 0).then_some(leaf_pal as u8);
      }
      // level 0-2：紧凑 popcount 定位 child offset
      let slot = (mask & (bit - 1)).count_ones() as usize;
      node = base + self.b_struct[node + 3 + slot] as usize;
      local = IVec3::new(
        local.x - ix * child_extent,
        local.y - iy * child_extent,
        local.z - iz * child_extent,
      );
      extent = child_extent;
    }
  }

  /// cell 灭占用查询（两级 DDA 粗步专用）：cc 为 cell 坐标（1 单位 = 16 voxel）。
  /// 语义：false ⇒ get_voxel 对该 cell 内全部 16³ voxel 位置都返回 None。
  ///
  /// 走树到 level 2（16³）粒度：uniform（mask=0 或 mask bit=0）→ occupied = palette != 0；
  /// 到达 extent=16 的 Split 节点 → occupied = true（ChunkTree merge 不变式：Split ⇒ 64 槽
  /// 颜色不全同 ⇒ 至少一槽非 AIR）。
  pub fn cell_occupied(&self, cc: IVec3) -> bool {
    // cell 的最小角 voxel → chunk + local（cell 恒不跨 chunk：16 | 256）
    let voxel_min = cc * 16;
    let chunk = voxel_min.div_euclid(IVec3::splat(CHUNK_SIZE));
    let Some(base) = self.chunk_base(chunk) else {
      return false;
    };
    let mut local = voxel_min.rem_euclid(IVec3::splat(CHUNK_SIZE));
    let mut node = base;
    let mut extent = CHUNK_SIZE;
    loop {
      let (mask, pal) = read_node(self.b_struct, node);
      if mask == 0 {
        return pal != 0;
      }
      if extent == 16 {
        // 已到目标 brick 粒度且是 Split → 内部必有非 AIR 体素
        return true;
      }
      let child_extent = extent >> 2;
      let ix = local.x / child_extent;
      let iy = local.y / child_extent;
      let iz = local.z / child_extent;
      let ci = (iz * 16 + iy * 4 + ix) as u64;
      let bit = 1u64 << ci;
      if mask & bit == 0 {
        return pal != 0;
      }
      // level 0-2：紧凑 popcount 定位 child offset
      let slot = (mask & (bit - 1)).count_ones() as usize;
      node = base + self.b_struct[node + 3 + slot] as usize;
      local = IVec3::new(
        local.x - ix * child_extent,
        local.y - iy * child_extent,
        local.z - iz * child_extent,
      );
      extent = child_extent;
    }
  }
}
