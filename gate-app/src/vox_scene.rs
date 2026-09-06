//! MagicaVoxel .vox 场景加载（vox-rs → VolumeGrid）
//!
//! nuke.vox 是「大场景分多个子网格」的导出：1193 组 SIZE+XYZI +
//! nTRN/nGRP/nSHP 节点树（1804 个 nTRN）。vox-rs 以默认 ReadOptions
//! 读取时把场景图烘焙成 flat instances（transform 已组合父级链），
//! 逐实例把稠密体素数组经 4×4 矩阵（90° 旋转 + 整数平移）变换后写入主世界。
//!
//! **翻转轴补偿**（1-voxel 偏移 + 接缝串色根因）：voxel p 的立方体是
//! [p, p+1)，点变换 world = R·(p−pivot) + t 只给出 max 角；旋转带负号
//! 的轴（如 Y→−X）上真实立方体区间是 [q−1, q]，必须注册 min 角
//! q−1——否则该实例沿翻转轴整体偏移 +1 voxel，与邻接模块重叠、
//! last-writer-wins 串色。补偿 = 每实例常量 flip_j = min(0, m_0j, m_1j, m_2j)
//! （vox 轴系，见 `instance_flip`）；90°/180° 旋转必有 1~2 个翻转轴，
//! identity 实例不受影响。
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
  // 材质统计（诊断白像素：emissive 体素在直接光 + GI 射线端点都会高频贡献亮度）
  let em: Vec<u8> = (1..=255u8)
    .filter(|&i| pal.get(i).emissive > 0)
    .collect();
  bevy::log::info!("VOX MATERIAL: {} emissive palette indices = {:?}", em.len(), em);
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
/// 坐标系适配：vox (X 右, Y 远, Z 上) 右手系 → gate (X 右, Y 上, Z 近) 右手系。
/// vox Y 远离观察者、gate Z 朝向观察者，方向相反，须取反 gate Z = −vox Y
/// 以保持手性（旧映射 (X, Z, Y) 行列式 = −1 = 反射 → 场景左右镜像）。
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
  // gate Z = −vox Y（手性修正）
  (wx, wz, -wy)
}

/// 翻转轴补偿（vox 轴系）：世界 j 轴的映射系数 m_ij ∈ {0, ±1} 且恰有一个
/// 非零。该系数为 −1 时，voxel 立方体 [p, p+1) 经变换后落在 [q−1, q]，
/// 注册格点须取 min 角 q−1 = 点变换结果 −1；系数非负则取 max 角即点变换
/// 本身。逐实例常量，identity 旋转全 0。
#[inline]
fn instance_flip(t: &vox_rs::Transform) -> IVec3 {
  IVec3::new(
    t.m00.min(t.m10).min(t.m20).min(0.0) as i32,
    t.m01.min(t.m11).min(t.m21).min(0.0) as i32,
    t.m02.min(t.m12).min(t.m22).min(0.0) as i32,
  )
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
  bucket_model(
    &models[inst.model_index],
    &inst.transform,
    instance_flip(&inst.transform),
    offset,
  )
}

/// 单模型体素分桶核心（诊断测试复用）：`flip` 为 vox 轴系翻转补偿
/// （见 `instance_flip`），非零轴注册 min 角；诊断时传 ZERO 复现旧行为。
fn bucket_model(
  m: &vox_rs::Model,
  t: &vox_rs::Transform,
  flip: IVec3,
  offset: IVec3,
) -> HashMap<ChunkCoord, Vec<u32>> {
  let (sx, sy, sz) = (m.size_x as usize, m.size_y as usize, m.size_z as usize);
  // flip 是 vox 轴系常量 → 预换到 gate 轴系（gate Z = −vox Y 故 z 分量取反）
  let flip_gate = IVec3::new(flip.x, flip.z, -flip.y);
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
        let world = IVec3::from(xform(t, m.size_x, m.size_y, m.size_z, x as u32, y as u32, z as u32))
          + flip_gate
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
    // pivot 体素 → 平移 t（gate 轴交换 + Z 取反后 y/z 互换且 z 取负）
    assert_eq!(xform(&t, 4, 4, 4, 2, 2, 2), (100, 50, -200));
    // 角体素 (0,0,0) → t − (2,2,2) vox → gate (98, 48, -198)
    assert_eq!(xform(&t, 4, 4, 4, 0, 0, 0), (98, 48, -198));

    // 奇数尺寸：pivot = floor(size/2)，ogt 文档的 3×4×1 例子 pivot=(1,2,0)
    assert_eq!(xform(&vox_rs::Transform::identity(), 3, 4, 1, 1, 2, 0), (0, 0, 0));
    assert_eq!(xform(&vox_rs::Transform::identity(), 3, 4, 1, 0, 0, 0), (-1, 0, 2));

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
    assert_eq!(xform(&r, 4, 4, 4, 2, 2, 2), (100, 50, -200));
    // (0,0,0)：l=(−2,−2,−2) → vox (−2, 2, −2)+t = (98, 202, 48) → gate (98, 48, -202)
    assert_eq!(xform(&r, 4, 4, 4, 0, 0, 0), (98, 48, -202));
    // (4,0,0) 角：l=(2,−2,−2) → vox (−2, −2, −2)+t = (98, 198, 48) → gate (98, 48, -198)
    assert_eq!(xform(&r, 4, 4, 4, 4, 0, 0), (98, 48, -198));
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
          -(inst.transform.m31 as i32),
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

  /// 翻转轴补偿单元：2×1×1 模型旋转 X→−Y、Y→X（vox 系），立方体注册
  /// min 角后几何对称展开于 t；无补偿（旧行为）整体沿 gate y +1。
  #[test]
  fn bucket_model_flip_snaps_min_corner() {
    // vox：world_x = +local_y (m10=1)、world_y = −local_x (m01=−1)、
    // world_z = +local_z；t = (100, 50, 0)。m00/m11 显式清零（置换矩阵）。
    let t = vox_rs::Transform {
      m00: 0.0,
      m01: -1.0,
      m10: 1.0,
      m11: 0.0,
      m30: 100.0,
      m31: 50.0,
      ..vox_rs::Transform::identity()
    };
    assert_eq!(instance_flip(&t), IVec3::new(0, -1, 0));
    let model = vox_rs::Model {
      size_x: 2,
      size_y: 1,
      size_z: 1,
      voxels: vec![1, 2],
    };
    let cells = |buf: &[u32]| -> Vec<(IVec3, u8)> {
      buf.iter()
        .map(|&p| {
          (
            IVec3::new(
              (p & 0xFF) as i32,
              ((p >> 8) & 0xFF) as i32,
              ((p >> 16) & 0xFF) as i32,
            ),
            (p >> 24) as u8,
          )
        })
        .collect()
    };
    // 补偿后：p=0 → gate(100,0,250) 色1；p=1 → gate(100,0,251) 色2
    // （offset.z=300 推到正值区间；几何 z 对称于 300−t_y=250）
    let off = IVec3::new(0, 0, 300);
    let fixed = bucket_model(&model, &t, instance_flip(&t), off);
    let v = cells(&fixed[&ChunkCoord(IVec3::ZERO)]);
    assert!(v.contains(&(IVec3::new(100, 0, 250), 1)), "{v:?}");
    assert!(v.contains(&(IVec3::new(100, 0, 251), 2)), "{v:?}");
    // 旧行为（flip=0）：注册 max 角 → 整体 z −1（1-voxel 偏移根因）
    let old = bucket_model(&model, &t, IVec3::ZERO, off);
    let v = cells(&old[&ChunkCoord(IVec3::ZERO)]);
    assert!(v.contains(&(IVec3::new(100, 0, 249), 1)), "{v:?}");
    assert!(v.contains(&(IVec3::new(100, 0, 250), 2)), "{v:?}");
  }

  /// 诊断（全量资产 + 排序扫描，较慢，单跑）：
  /// `cargo test -p gate-app --release nuke_flip_conflict -- --ignored --nocapture`
  ///
  /// 统计「同一格被写成 ≥2 种颜色」的跨实例冲突体素数，对比翻转补偿
  /// 前/后。正确拼装的场景模块间应无空隙无重叠 → 补偿后冲突数应骤降；
  /// 剩余冲突按「写入方是否全为 identity 实例」分类——identity 对相撞
  /// 与旋转约定无关，只能是作者有意放置的相交几何（管道穿墙等）。
  #[test]
  #[ignore = "全量资产诊断：--release -- --ignored 单跑"]
  fn nuke_flip_conflict_diagnostic() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/vox/nuke.vox");
    if !path.exists() {
      return;
    }
    let file = std::fs::File::open(&path).unwrap();
    let mut r = std::io::BufReader::new(file);
    let scene = vox_rs::Scene::read(&mut r).unwrap();

    let run = |use_flip: bool| -> (usize, usize, usize, usize) {
      let per_instance: Vec<HashMap<ChunkCoord, Vec<u64>>> = scene
        .instances
        .par_iter()
        .filter_map(|inst| {
          if inst.hidden {
            return None;
          }
          let flip = if use_flip {
            instance_flip(&inst.transform)
          } else {
            IVec3::ZERO
          };
          let t = &inst.transform;
          let identity_rot = t.m00 == 1.0
            && t.m11 == 1.0
            && t.m22 == 1.0
            && t.m01 == 0.0
            && t.m02 == 0.0
            && t.m10 == 0.0
            && t.m12 == 0.0
            && t.m20 == 0.0
            && t.m21 == 0.0;
          Some(bucket_model(
            &scene.models[inst.model_index],
            t,
            flip,
            IVec3::ZERO,
          ))
          .map(|map| {
            map
              .into_iter()
              .map(|(cc, v)| {
                (
                  cc,
                  v.into_iter()
                    .map(|p| ((p as u64) << 1) | identity_rot as u64)
                    .collect::<Vec<u64>>(),
                )
              })
              .collect::<HashMap<_, _>>()
          })
        })
        .collect();
      let mut buckets: HashMap<ChunkCoord, Vec<u64>> = HashMap::new();
      for map in per_instance {
        for (cc, v) in map {
          buckets.entry(cc).or_default().extend(v);
        }
      }
      // 逐 chunk 排序扫描：key = packed32<<1 | identity_rot（packed =
      // local24 | palette<<24 → L 在 key bit 1..24、P 在 bit 25..32、rot
      // 在 bit 0）。按 L 排序后同格连续；统计写多次格数、异色冲突格数、
      // 其中全 identity 写入方的冲突格数。
      buckets
        .into_par_iter()
        .map(|(_, mut v)| {
          v.sort_unstable_by_key(|&k| (k >> 1) & 0x00FF_FFFF);
          let (mut multi, mut conflict, mut conflict_id) = (0usize, 0usize, 0usize);
          let mut i = 0;
          while i < v.len() {
            let coord = (v[i] >> 1) & 0x00FF_FFFF;
            let mut seen = [0u64; 4]; // palette < 256 的 distinct 位图
            let mut npals = 0usize;
            let (mut any_rot, mut any_id) = (false, false);
            let mut j = i;
            while j < v.len() && ((v[j] >> 1) & 0x00FF_FFFF) == coord {
              let k = v[j];
              let pal = ((k >> 25) & 0xFF) as usize;
              let bit = 1u64 << (pal & 63);
              if seen[pal >> 6] & bit == 0 {
                seen[pal >> 6] |= bit;
                npals += 1;
              }
              if k & 1 == 1 {
                any_id = true;
              } else {
                any_rot = true;
              }
              j += 1;
            }
            if j - i > 1 {
              multi += 1;
            }
            if npals > 1 {
              conflict += 1;
              if any_id && !any_rot {
                conflict_id += 1;
              }
            }
            i = j;
          }
          (v.len(), multi, conflict, conflict_id)
        })
        .reduce(
          || (0, 0, 0, 0),
          |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2, a.3 + b.3),
        )
    };

    let (w0, m0, c0, _) = run(false);
    let (w1, m1, c1, c1_id) = run(true);
    println!("before(flip=0): written={w0} multi_cells={m0} conflict_cells={c0}");
    println!("after (flip) : written={w1} multi_cells={m1} conflict_cells={c1}");
    println!("after: 冲突中全 identity 写入方对={c1_id}（含旋转方={}", c1 - c1_id);
    assert_eq!(w0, w1, "两约定写入体素数应一致");
    assert!(c1 < c0 / 2, "翻转补偿后异色冲突应减半以上（before={c0} after={c1}）");
  }

  /// 一次性诊断（#[ignore]，手动 cargo test -- --ignored 跑）：
  /// 按用户截图机位（eye/target 取自 FPS overlay）离屏渲染 480x270，
  /// 复刻 dda_main unlit（debug_mode==3）路径的逐体素法线着色，
  /// 找出「孤立高亮点」：ndl≈1 但 8 邻域暗/未命中，打印其体素/6邻域/法线，
  /// 判定是几何真相（细小特征剪影）还是遍历把切向邻体素带出。
  #[test]
  #[ignore = "诊断：需 nuke.vox 资产，CPU 离屏渲染 ~ 分钟级"]
  fn nuke_white_pixel_diagnostic() {
    use gate_render::brickmap::cpu_reference_dda_ray_tree;
    use gate_render::{BrickMapBuilder, BrickMapView, DdaCameraConfig, OrbitCamera};
    use glam::{IVec3, Vec3, Vec4};

    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/vox/nuke.vox");
    if !path.exists() {
      return;
    }
    // 默认 GATE_TILES=2 → anchor = EXT_FINE_HALF = (512,16,512)，与 scene.rs 启动一致
    let mut grid = VolumeGrid::new();
    let info = load_vox_scene(&mut grid, &path, IVec3::new(512, 16, 512)).expect("load nuke.vox");
    println!("voxels written = {} aabb=[{}]-[{}]", info.voxels_written, info.aabb_min, info.aabb_max);
    let bufs = BrickMapBuilder::build_full(&grid).buffers().clone();
    let view = BrickMapView::new(&bufs);

    // 截图机位（FPS overlay 文本）
    let eye = Vec3::new(835.0, 251.8, 438.4);
    let target = Vec3::new(658.3, 213.2, 472.7);
    let orbit = OrbitCamera::from_eye(eye, target);
    let cfg = DdaCameraConfig::from_orbit(&orbit, 60.0_f32.to_radians(), 16.0 / 9.0, 1.0, 65536.0);

    // day_outdoor.ron：sun dir（光传播）(0.5,-0.8,0.3) → L 指向太阳 = 反向
    let l = Vec3::new(-0.5, 0.8, -0.3).normalize();

    const W: usize = 480;
    const H: usize = 270;

    // face_id 0..5 → ±轴（镜像 dda.rs face_normal_from_index）
    let face_n = |f: u8| match f {
      0 => Vec3::new(-1.0, 0.0, 0.0),
      1 => Vec3::new(1.0, 0.0, 0.0),
      2 => Vec3::new(0.0, -1.0, 0.0),
      3 => Vec3::new(0.0, 1.0, 0.0),
      4 => Vec3::new(0.0, 0.0, -1.0),
      _ => Vec3::new(0.0, 0.0, 1.0),
    };

    #[derive(Clone, Copy)]
    struct Px {
      hit: bool,
      ndl: f32,
      fdl: f32,
      pal: u8,
    }
    let mut img = vec![Px { hit: false, ndl: 0.0, fdl: 0.0, pal: 0 }; W * H];

    let mut n_hits = 0u32;
    for py in 0..H {
      for px in 0..W {
        let u = (px as f32 + 0.5) / W as f32 * 2.0 - 1.0;
        let vv = 1.0 - (py as f32 + 0.5) / H as f32 * 2.0;
        let n4 = cfg.inv_view_proj * Vec4::new(u, vv, 0.0, 1.0);
        let f4 = cfg.inv_view_proj * Vec4::new(u, vv, 1.0, 1.0);
        let near = n4.truncate() / n4.w;
        let far = f4.truncate() / f4.w;
        let dir = (far - near).normalize();

        if let Some(h) = cpu_reference_dda_ray_tree(&bufs, near, dir, 20000.0) {
          n_hits += 1;
          // 逐体素隐式法线（镜像 implicit_normal_local）：d 指向实体反侧 = 朝外
          let solid = |dv: IVec3| view.get_voxel(h.voxel + dv).is_some() as i32;
          let d = Vec3::new(
            (solid(IVec3::new(-1, 0, 0)) - solid(IVec3::new(1, 0, 0))) as f32,
            (solid(IVec3::new(0, -1, 0)) - solid(IVec3::new(0, 1, 0))) as f32,
            (solid(IVec3::new(0, 0, -1)) - solid(IVec3::new(0, 0, 1))) as f32,
          );
          let fn_ = face_n(h.face_id);
          let n = if d.length_squared() > 0.25 { d.normalize() } else { fn_ };
          img[py * W + px] = Px {
            hit: true,
            ndl: n.dot(l).max(0.0),
            fdl: fn_.dot(l).max(0.0),
            pal: h.pal,
          };
        }
      }
    }

    // 孤立高亮点：ndl≥0.9 且 8 邻域多数（≥6/8）暗（未命中或 ndl<0.2）
    let mut sparks = 0u32;
    let mut sparks_face_dark = 0u32; // 命中面本身背光（fdl<0.2）却被逐体素法线点亮
    let mut printed = 0u32;
    for py in 1..H - 1 {
      for px in 1..W - 1 {
        let c = img[py * W + px];
        if !c.hit || c.ndl < 0.9 {
          continue;
        }
        let mut dark = 0;
        for dy in -1i32..=1 {
          for dx in -1i32..=1 {
            if dx == 0 && dy == 0 {
              continue;
            }
            let q = img[(py as i32 + dy) as usize * W + (px as i32 + dx) as usize];
            if !q.hit || q.ndl < 0.2 {
              dark += 1;
            }
          }
        }
        if dark < 6 {
          continue;
        }
        sparks += 1;
        if c.fdl < 0.2 {
          sparks_face_dark += 1;
        }
        if printed >= 40 {
          continue;
        }
        // 取该像素命中体素详情重放一遍（打印用）
        let u = (px as f32 + 0.5) / W as f32 * 2.0 - 1.0;
        let vv = 1.0 - (py as f32 + 0.5) / H as f32 * 2.0;
        let n4 = cfg.inv_view_proj * Vec4::new(u, vv, 0.0, 1.0);
        let f4 = cfg.inv_view_proj * Vec4::new(u, vv, 1.0, 1.0);
        let near = n4.truncate() / n4.w;
        let far = f4.truncate() / f4.w;
        let dir = (far - near).normalize();
        let h = cpu_reference_dda_ray_tree(&bufs, near, dir, 20000.0).unwrap();
        let occ = |dv: IVec3| view.get_voxel(h.voxel + dv).map(|p| p as i32).unwrap_or(-1);
        println!(
          "SPARK px=({px},{py}) pal={} voxel={} face={} ndl={:.2} fdl={:.2} t={:.1} occ[+x={} -x={} +y={} -y={} +z={} -z={}]",
          h.pal, h.voxel, h.face_id, c.ndl, c.fdl, h.t,
          occ(IVec3::new(1, 0, 0)), occ(IVec3::new(-1, 0, 0)),
          occ(IVec3::new(0, 1, 0)), occ(IVec3::new(0, -1, 0)),
          occ(IVec3::new(0, 0, 1)), occ(IVec3::new(0, 0, -1)),
        );
        printed += 1;
      }
    }
    let total = W * H;
    println!("hits={n_hits}/{total} sparks={sparks} of which face_backlit_but_voxel_lit={sparks_face_dark}");
  }
}
