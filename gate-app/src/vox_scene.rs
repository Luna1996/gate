//! MagicaVoxel .vox 场景加载（vox-rs → VolumeGrid）
//!
//! nuke.vox 是「大场景分多个子网格」的导出：1193 组 SIZE+XYZI +
//! nTRN/nGRP/nSHP 节点树（1804 个 nTRN）。vox-rs 以默认 ReadOptions
//! 读取时把场景图烘焙成 flat instances（transform 已组合父级链），
//! 逐实例把稠密体素数组经 4×4 矩阵（90° 旋转 + 整数平移）变换后写入主世界。
//!
//! palette 映射：vox 色号 1..=255 → gate palette 同号（0 = AIR）；
//! MATL 材质的 rough/emit 线性映射到 PaletteEntry.roughness/emissive。

use std::collections::HashMap;
use std::io::BufReader;
use std::path::Path;

use glam::{IVec3};
use gate_voxel::{ChunkCoord, ChunkTree, PaletteEntry, VolumeGrid, CHUNK_SIZE};
use rayon::prelude::*;

/// 加载结果：场景 AABB（世界坐标）+ 统计
pub struct VoxSceneInfo {
  pub aabb_min: IVec3,
  pub aabb_max: IVec3,
  pub instances_used: usize,
  pub voxels_written: usize,
  pub voxels_dropped: usize,
}

/// 读 .vox → palette 映射 → 体素写入 grid。
///
/// `anchor` = 场景 AABB 底面中心的落点（x/z 居中，y = 底面高度）。
pub fn load_vox_scene(
  grid: &mut VolumeGrid,
  path: &Path,
  anchor: IVec3,
) -> Result<VoxSceneInfo, Box<dyn std::error::Error>> {
  let t0 = std::time::Instant::now();
  let file = std::fs::File::open(path)?;
  let mut reader = BufReader::new(file);
  let scene = vox_rs::Scene::read(&mut reader)?;
  bevy::log::info!(
    "VOX LOAD: {} ver={} models={} instances={} ({:?})",
    path.display(),
    scene.file_version,
    scene.models.len(),
    scene.instances.len(),
    t0.elapsed(),
  );

  paint_vox_palette(grid, &scene);

  // ---- 首遍：实例 AABB（gate 坐标，xform 已含 Z-up→Y-up）----
  let mut lo = IVec3::splat(i32::MAX);
  let mut hi = IVec3::splat(i32::MIN);
  let mut instances_used = 0usize;
  for inst in &scene.instances {
    if inst.hidden {
      continue;
    }
    let m = &scene.models[inst.model_index];
    let (mlo, mhi) = transformed_aabb(&inst.transform, m.size_x, m.size_y, m.size_z);
    lo = lo.min(mlo);
    hi = hi.max(mhi);
    instances_used += 1;
  }
  if instances_used == 0 {
    return Err("vox 场景没有可见实例".into());
  }
  // 重定基：AABB 底面中心 → anchor
  let offset = anchor - IVec3::new((lo.x + hi.x) / 2, lo.y, (lo.z + hi.z) / 2);

  // ---- 次遍：并行分桶 → 并行建树 → 挂载 ----
  // 逐体素 set_voxel_ivec3 是 31.7M 次「HashMap 哈希 + 两次树下降」的串行
  // 瓶颈；改为按 chunk 分桶后 rayon 并行建树（chunk 间零共享）。
  // 写入序与旧串行完全一致：collect 保序 → 文件序实例 → 实例内 x-major，
  // 同 voxel 异色的 last-writer-wins 结果确定，最终树逐位相同。
  let t1 = std::time::Instant::now();
  let per_instance: Vec<HashMap<ChunkCoord, Vec<u32>>> = scene
    .instances
    .par_iter()
    .filter_map(|inst| {
      if inst.hidden {
        return None;
      }
      Some(bucket_instance(inst, &scene.models, offset))
    })
    .collect();
  // 实例桶 → chunk 全局桶（文件序遍历，保持旧串行写入序）
  let mut buckets: HashMap<ChunkCoord, Vec<u32>> = HashMap::new();
  for map in per_instance {
    for (cc, mut v) in map {
      buckets.entry(cc).or_default().append(&mut v);
    }
  }
  // 每 chunk 并行建树；u32 打包 = local_x | local_y<<8 | local_z<<16 | palette<<24
  let mut written = 0usize;
  let mut dropped = 0usize;
  let mounts: Vec<(ChunkCoord, ChunkTree, u64, usize)> = buckets
    .into_par_iter()
    .map(|(cc, packed)| {
      let mut tree = ChunkTree::empty();
      let mut applied = 0u64;
      for &p in &packed {
        if tree.set_voxel(
          (p & 0xFF) as i32,
          ((p >> 8) & 0xFF) as i32,
          ((p >> 16) & 0xFF) as i32,
          (p >> 24) as u8,
        ) {
          applied += 1;
        }
      }
      (cc, tree, applied, packed.len())
    })
    .collect();
  for (cc, tree, applied, total) in mounts {
    written += applied as usize;
    dropped += total - applied as usize;
    grid.mount_chunk_tree(cc, tree, applied);
  }
  bevy::log::info!(
    "VOX BUILD: written={written} dropped={dropped} aabb=[{}]-[{}] ({:?})",
    lo + offset,
    hi + offset,
    t1.elapsed(),
  );
  Ok(VoxSceneInfo {
    aabb_min: lo + offset,
    aabb_max: hi + offset,
    instances_used,
    voxels_written: written,
    voxels_dropped: dropped,
  })
}

/// vox RGBA + MATL → gate palette（色号 1..=255，0 = AIR 不映射）
fn paint_vox_palette(grid: &mut VolumeGrid, scene: &vox_rs::Scene) {
  let pal = grid.palette_mut();
  for i in 1..=255usize {
    let rgba = scene.palette.colors[i];
    let mat = &scene.materials[i];
    let mut e = PaletteEntry::default();
    e.color = [rgba.r, rgba.g, rgba.b];
    e.roughness = mat
      .rough
      .map(|v| (v.clamp(0.0, 1.0) * 255.0) as u8)
      .unwrap_or(200);
    e.emissive = mat
      .emit
      .map(|v| (v.clamp(0.0, 1.0) * 255.0) as u8)
      .unwrap_or(0);
    pal.set(i as u8, e);
  }
}

/// vox-rs Transform（移植自 ogt_vox，二者矩阵字段逐一同构）：平移在
/// m30..m32，点变换按行向量约定 p' = p·M：
///   wx = m00·x + m10·y + m20·z + m30（wy/wz 同理）。
///
/// **pivot 约定**（ogt_vox 头文件 "EXPLANATION OF MODEL PIVOTS"）：
/// 模型中心 pivot = floor(size/2)（整数格点，**不是** size/2.0）；
/// nTRN 平移 t 是 pivot 的世界坐标，变换绕 pivot 进行：
///   world = R·(local − pivot) + t
/// MagicaVoxel 编辑器内旋转物体时自动把中心补偿烘进 _t，加载方只需
/// 减去 pivot 再乘矩阵——**切勿把 pivot 加回去**（旧实现 R·(p−c)+c+t
/// 会给每个模型附加与尺寸相关的 c 偏移，rebase 后不同尺寸模块相互错位）。
/// 父级 group 链已由 vox-rs 默认 ReadOptions flatten 烘焙进 transform。
///
/// 坐标系适配：vox Z-up → gate Y-up，交换输出 (X, Z, Y)。
/// R 元素 ∈ {0, ±1}、t 为整数、pivot 为整数 → 纯整数运算无舍入误差。
#[inline]
fn xform(
  t: &vox_rs::Transform,
  sx: u32,
  sy: u32,
  sz: u32,
  x: u32,
  y: u32,
  z: u32,
) -> (i32, i32, i32) {
  let lx = x as i32 - (sx as i32 / 2);
  let ly = y as i32 - (sy as i32 / 2);
  let lz = z as i32 - (sz as i32 / 2);
  let wx = t.m00 as i32 * lx + t.m10 as i32 * ly + t.m20 as i32 * lz + t.m30 as i32;
  let wy = t.m01 as i32 * lx + t.m11 as i32 * ly + t.m21 as i32 * lz + t.m31 as i32;
  let wz = t.m02 as i32 * lx + t.m12 as i32 * ly + t.m22 as i32 * lz + t.m32 as i32;
  // vox (X 右, Y 前, Z 上) → gate (X 右, Y 上, Z 前)
  (wx, wz, wy)
}

/// 模型盒 8 角（格点 0 与 size）经变换后的整数 AABB。
/// pivot = floor(size/2) 时模型体素恰好覆盖 [−pivot, size−pivot]，
/// 故角点取 0/size 即模型几何边界（90° 旋转 + 整数平移，无误差）。
fn transformed_aabb(t: &vox_rs::Transform, sx: u32, sy: u32, sz: u32) -> (IVec3, IVec3) {
  let mut lo = IVec3::splat(i32::MAX);
  let mut hi = IVec3::splat(i32::MIN);
  for &cz in &[0u32, sz] {
    for &cy in &[0u32, sy] {
      for &cx in &[0u32, sx] {
        let p = IVec3::from(xform(t, sx, sy, sz, cx, cy, cz));
        lo = lo.min(p);
        hi = hi.max(p);
      }
    }
  }
  (lo, hi)
}

/// 单实例体素分桶：chunk → u32 打包（chunk 内 local 0..255 各 8 bit + palette）。
///
/// 并行建树的数据面：每 chunk 的体素收集为一个 `Vec<u32>`，chunk 间零共享。
/// x-major 行切片迭代（无逐索引边界检查）+ last-chunk 缓存——连续体素几乎
/// 都落同一 chunk，HashMap 只在跨 chunk 时触碰。
fn bucket_instance(
  inst: &vox_rs::Instance,
  models: &[vox_rs::Model],
  offset: IVec3,
) -> HashMap<ChunkCoord, Vec<u32>> {
  let m = &models[inst.model_index];
  let (sx, sy, sz) = (m.size_x as usize, m.size_y as usize, m.size_z as usize);
  let t = &inst.transform;
  let mut map: HashMap<ChunkCoord, Vec<u32>> = HashMap::new();
  let mut cur_key: Option<ChunkCoord> = None;
  let mut cur_buf: Vec<u32> = Vec::new();
  for z in 0..sz {
    for y in 0..sy {
      let row = (z * sy + y) * sx;
      for (x, &c) in m.voxels[row..row + sx].iter().enumerate() {
        if c == 0 {
          continue;
        }
        let world =
          IVec3::from(xform(t, m.size_x, m.size_y, m.size_z, x as u32, y as u32, z as u32))
            + offset;
        let cc = ChunkCoord(world.div_euclid(IVec3::splat(CHUNK_SIZE)));
        if cur_key != Some(cc) {
          if let Some(k) = cur_key.take() {
            map.entry(k).or_default().append(&mut cur_buf);
          }
          cur_key = Some(cc);
        }
        let local = world.rem_euclid(IVec3::splat(CHUNK_SIZE));
        cur_buf.push(
          local.x as u32 | (local.y as u32) << 8 | (local.z as u32) << 16 | (c as u32) << 24,
        );
      }
    }
  }
  if let Some(k) = cur_key.take() {
    map.entry(k).or_default().append(&mut cur_buf);
  }
  map
}

#[cfg(test)]
mod tests {
  use super::*;

  /// 真实资产冒烟：文件缺失自动跳过（CI 无 110MB 资产）
  #[test]
  fn nuke_vox_loads_and_writes() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/vox/nuke.vox");
    if !path.exists() {
      return;
    }
    let mut grid = VolumeGrid::new();
    let info = load_vox_scene(&mut grid, &path, IVec3::ZERO).expect("load nuke.vox");
    println!(
      "instances_used={} written={} dropped={} aabb=[{}]-[{}]",
      info.instances_used, info.voxels_written, info.voxels_dropped, info.aabb_min, info.aabb_max
    );
    assert!(info.instances_used > 0);
    assert!(info.voxels_written > 0);
    assert!(
      info.voxels_dropped < info.voxels_written,
      "重写占比过大，疑似实例重复"
    );
    assert!(info.aabb_max.cmpgt(info.aabb_min).all(), "AABB 非退化");
    // 有实体 chunk 落盘（中心点可能是空气，不做点查断言）
    assert!(grid.chunk_count() > 0);
  }

  /// 复刻 vox-rs/ogt_vox 的 _r 字节解码（codec.rs parse_transform），
  /// 供合成测试构造旋转矩阵：bits0-1/2-3 选 row0/row1 的基轴，
  /// bits4/5/6 取反；w.x = dot(row0, l)，w.y/z 同理。
  fn decode_packed_rotation(bits: u32) -> vox_rs::Transform {
    const AXES: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let i0 = (bits & 3) as usize;
    let i1 = ((bits >> 2) & 3) as usize;
    let i2 = (0..3).find(|&i| i != i0 && i != i1).unwrap();
    let mut row = [AXES[i0], AXES[i1], AXES[i2]];
    for (axis, bit) in [(0usize, 1u32 << 4), (1, 1 << 5), (2, 1 << 6)] {
      if bits & bit != 0 {
        row[axis] = [-row[axis][0], -row[axis][1], -row[axis][2]];
      }
    }
    vox_rs::Transform {
      m00: row[0][0],
      m01: row[1][0],
      m02: row[2][0],
      m10: row[0][1],
      m11: row[1][1],
      m12: row[2][1],
      m20: row[0][2],
      m21: row[1][2],
      m22: row[2][2],
      ..vox_rs::Transform::identity()
    }
  }

  /// 合成用例（不依赖资产）：锁定 ogt pivot 约定 world = R·(p − floor(size/2)) + t。
  #[test]
  fn xform_pivot_convention() {
    // identity 旋转 + t=(100,200,50)，4³ 模型 pivot=(2,2,2)
    let t = vox_rs::Transform {
      m30: 100.0,
      m31: 200.0,
      m32: 50.0,
      ..vox_rs::Transform::identity()
    };
    // pivot 体素 → 平移 t（gate 轴交换后 y/z 互换）
    assert_eq!(xform(&t, 4, 4, 4, 2, 2, 2), (100, 50, 200));
    // 角体素 (0,0,0) → t − (2,2,2) vox → gate (98, 48, 198)
    assert_eq!(xform(&t, 4, 4, 4, 0, 0, 0), (98, 48, 198));

    // 奇数尺寸：pivot = floor(size/2)，ogt 文档的 3×4×1 例子 pivot=(1,2,0)
    assert_eq!(xform(&vox_rs::Transform::identity(), 3, 4, 1, 1, 2, 0), (0, 0, 0));
    assert_eq!(xform(&vox_rs::Transform::identity(), 3, 4, 1, 0, 0, 0), (-1, 0, -2));

    // 绕 vox Z 轴 90° 旋转（packed byte 33：row0=+Y, row1=−X, row2=+Z），
    // vox 空间 (wx,wy) = (ly, −lx)
    let r = {
      let mut m = decode_packed_rotation(33);
      m.m30 = 100.0;
      m.m31 = 200.0;
      m.m32 = 50.0;
      m
    };
    // pivot 恒落到 t
    assert_eq!(xform(&r, 4, 4, 4, 2, 2, 2), (100, 50, 200));
    // (0,0,0)：l=(−2,−2,−2) → vox (−2, 2, −2)+t = (98, 202, 48) → gate (98, 48, 202)
    assert_eq!(xform(&r, 4, 4, 4, 0, 0, 0), (98, 48, 202));
    // (4,0,0) 角：l=(2,−2,−2) → vox (−2, −2, −2)+t = (98, 198, 48) → gate (98, 48, 198)
    assert_eq!(xform(&r, 4, 4, 4, 4, 0, 0), (98, 48, 198));
  }

  /// 真实资产全实例不变量：每个可见实例的 pivot 体素（floor(size/2)）
  /// 必须映射到 nTRN 平移 t——ogt「translation = pivot 世界位置」约定。
  #[test]
  fn nuke_pivot_maps_to_translation() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/vox/nuke.vox");
    if !path.exists() {
      return;
    }
    let file = std::fs::File::open(&path).unwrap();
    let mut r = std::io::BufReader::new(file);
    let scene = vox_rs::Scene::read(&mut r).unwrap();
    let mut checked = 0usize;
    for inst in &scene.instances {
      if inst.hidden {
        continue;
      }
      let m = &scene.models[inst.model_index];
      let (gx, gy, gz) = xform(
        &inst.transform,
        m.size_x,
        m.size_y,
        m.size_z,
        m.size_x / 2,
        m.size_y / 2,
        m.size_z / 2,
      );
      assert_eq!(
        (gx, gy, gz),
        (
          inst.transform.m30 as i32,
          inst.transform.m32 as i32,
          inst.transform.m31 as i32
        ),
        "instance {} pivot 未落到 nTRN 平移 t（size={}×{}×{}）",
        checked,
        m.size_x,
        m.size_y,
        m.size_z
      );
      checked += 1;
    }
    assert!(checked > 0);
    println!("pivot invariant OK over {checked} visible instances");
  }
}
