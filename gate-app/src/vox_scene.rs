//! MagicaVoxel .vox 场景加载（vox-rs → VolumeGrid）：vox-rs 默认把场景图烘焙成 flat instances，
//! 逐实例经 4×4 矩阵（90° 旋转 + 整数平移）写入主世界。翻转轴须注册 min 角。
//! palette：vox 色号 1..=255 → gate palette 同号（0 = AIR），MATL 的 rough/emit 映射到 roughness/emissive。

use std::collections::HashMap;
use std::io::BufReader;
use std::path::Path;

use gate_voxel::{
  CHUNK_SIZE, ChunkCoord, ChunkTree, PALETTE_INDEX_MAX, PaletteEntry, PaletteId, VolumeGrid,
};
use glam::IVec3;
use rayon::prelude::*;

/// 加载结果：场景 AABB（世界坐标）+ 统计
pub struct VoxSceneInfo {
  pub aabb_min: IVec3,
  pub aabb_max: IVec3,
  pub instances_used: usize,
  pub voxels_written: usize,
  pub voxels_dropped: usize,
}

/// `assets/vox` 下可选的世界模型名（`.vox` 去扩展名，字典序；目录缺失/为空 → `vec!["nuke"]`）。
/// 供 DebugMenu 模型下拉取选项；选项是文件名字面量，必须与 `load_vox_scene` 的路径拼法一致。
pub fn scan_vox_models() -> Vec<String> {
  let dir = gate_render::assets_dir().join("vox");
  let Ok(entries) = std::fs::read_dir(&dir) else {
    bevy::log::warn!("vox 目录不可读（{}），模型下拉回退到 nuke", dir.display());
    return vec!["nuke".to_string()];
  };
  let mut names: Vec<String> = entries
    .filter_map(|e| e.ok())
    .filter_map(|e| {
      let path = e.path();
      let is_vox = path.extension().is_some_and(|x| x.eq_ignore_ascii_case("vox"));
      if !is_vox {
        return None;
      }
      path.file_stem().map(|s| s.to_string_lossy().into_owned())
    })
    .collect();
  names.sort();
  names.dedup();
  if names.is_empty() {
    // 空目录 → 保留默认项，让 UI 有名字可选
    bevy::log::warn!("vox 目录下没有 .vox 文件（{}），模型下拉回退到 nuke", dir.display());
    return vec!["nuke".to_string()];
  }
  names
}

/// 读 .vox → palette 映射 → 体素写入 grid。
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

  // 首遍：实例 AABB（gate 坐标，xform 已含 Z-up→Y-up）
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

  // 按 chunk 分桶后 rayon 并行建树（chunk 间零共享）。写入序确定：collect 保序 →
  // 文件序实例 → 实例内 x-major，故同 voxel 异色的 last-writer-wins 结果确定。
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
  // 几何实际引用的色号集（u32 打包：palette 在高 8 位）；未引用的槽保持全零，
  // 留给编辑材质按"条目全零"认领（见 gate-app/src/edit.rs）。
  let used_pal: [bool; 256] = per_instance
    .par_iter()
    .fold(
      || [false; 256],
      |mut acc, map| {
        for words in map.values() {
          for &w in words {
            acc[((w >> 24) & 0xFF) as usize] = true;
          }
        }
        acc
      },
    )
    .reduce(
      || [false; 256],
      |mut a, b| {
        for i in 0..256 {
          a[i] |= b[i];
        }
        a
      },
    );
  paint_vox_palette(grid, &scene, &used_pal);
  // 实例桶 → chunk 全局桶（文件序遍历，保证写入序）
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
        // 打包字高 8 位是 .vox 自身色号（256 色调色板），转成材质索引
        if tree.set_voxel(
          (p & 0xFF) as i32,
          ((p >> 8) & 0xFF) as i32,
          ((p >> 16) & 0xFF) as i32,
          PaletteId((p >> 24) as u16),
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

/// vox RGBA + MATL → gate palette（色号 1..=255，0 = AIR 不映射）。
/// 只铺 `used_pal` 引用的色号，其余槽保持全零（编辑材质按"条目全零"认领）。
fn paint_vox_palette(grid: &mut VolumeGrid, scene: &vox_rs::Scene, used_pal: &[bool; 256]) {
  let pal = grid.palette_mut();
  let mut painted = 0usize;
  for (i, &used) in used_pal.iter().enumerate().take(256).skip(1) {
    if !used {
      continue;
    }
    let rgba = scene.palette.colors[i];
    let mat = &scene.materials[i];
    let e = PaletteEntry {
      color: [rgba.r, rgba.g, rgba.b],
      roughness: mat.rough.map(|v| (v.clamp(0.0, 1.0) * 255.0) as u8).unwrap_or(200),
      emissive: mat.emit.map(|v| (v.clamp(0.0, 1.0) * 255.0) as u8).unwrap_or(0),
      ..Default::default()
    };
    pal.set(PaletteId(i as u16), e);
    painted += 1;
  }
  // 材质统计
  let em: Vec<u16> = (1..=255u16).filter(|&i| pal.get(PaletteId(i)).emissive > 0).collect();
  bevy::log::info!(
    "VOX MATERIAL: {} emissive palette indices = {:?}（引用 {} 个色号，本 volume 余 {} 个空槽留给编辑材质）",
    em.len(),
    em,
    painted,
    PALETTE_INDEX_MAX as usize - painted
  );
}

/// vox-rs Transform（与 ogt_vox 矩阵逐一同构）：行向量约定 p' = p·M，平移在 m30..m32。
/// pivot = floor(size/2)，world = R·(local − pivot) + t（勿加回 pivot）；坐标映射取 gate Z = −vox Y。
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

/// 翻转轴补偿（vox 轴系）：映射系数为 −1 时 voxel [p, p+1) 变换后落在 [q−1, q]，须注册 min 角；
/// 系数非负则取 max 角。逐实例常量，identity 全 0。
#[inline]
fn instance_flip(t: &vox_rs::Transform) -> IVec3 {
  IVec3::new(
    t.m00.min(t.m10).min(t.m20).min(0.0) as i32,
    t.m01.min(t.m11).min(t.m21).min(0.0) as i32,
    t.m02.min(t.m12).min(t.m22).min(0.0) as i32,
  )
}

/// 模型盒 8 角（格点 0 与 size）经变换后的整数 AABB。
/// pivot = floor(size/2) 时体素覆盖 [−pivot, size−pivot]，角点取 0/size 即几何边界。
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

/// 单实例体素分桶：chunk → u32 打包（local 各 8 bit + 高 8 位 .vox 色号）；高 8 位是 .vox
/// 自身色号（256 色调色板），建树时按 `PaletteId(色号)` 落同名槽。chunk 间零共享。
fn bucket_instance(
  inst: &vox_rs::Instance,
  models: &[vox_rs::Model],
  offset: IVec3,
) -> HashMap<ChunkCoord, Vec<u32>> {
  bucket_model(&models[inst.model_index], &inst.transform, instance_flip(&inst.transform), offset)
}

/// 单模型体素分桶核心（测试复用）：`flip` = vox 轴系翻转补偿（见 `instance_flip`），
/// 非零轴注册 min 角；传 ZERO 即不补偿。
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
        let world =
          IVec3::from(xform(t, m.size_x, m.size_y, m.size_z, x as u32, y as u32, z as u32))
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
        cur_buf
          .push(local.x as u32 | (local.y as u32) << 8 | (local.z as u32) << 16 | (c as u32) << 24);
      }
    }
  }
  if let Some(k) = cur_key.take() {
    map.entry(k).or_default().append(&mut cur_buf);
  }
  map
}
