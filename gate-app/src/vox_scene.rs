//! MagicaVoxel .vox 场景加载（vox-rs → VolumeGrid）
//!
//! nuke.vox 是「大场景分多个子网格」的导出：1193 组 SIZE+XYZI +
//! nTRN/nGRP/nSHP 节点树（1804 个 nTRN）。vox-rs 以默认 ReadOptions
//! 读取时把场景图烘焙成 flat instances（transform 已组合父级链），
//! 逐实例把稠密体素数组经 4×4 矩阵（90° 旋转 + 整数平移）变换后写入主世界。
//!
//! palette 映射：vox 色号 1..=255 → gate palette 同号（0 = AIR）；
//! MATL 材质的 rough/emit 线性映射到 PaletteEntry.roughness/emissive。

use std::io::BufReader;
use std::path::Path;

use glam::{IVec3, Vec3};
use gate_voxel::{PaletteEntry, VolumeGrid};

/// 加载结果：场景 AABB（世界坐标）+ 统计
pub struct VoxSceneInfo {
  pub aabb_min: IVec3,
  pub aabb_max: IVec3,
  pub instances_used: usize,
  pub voxels_written: usize,
  pub voxels_dropped: usize,
}

impl VoxSceneInfo {
  /// AABB 中心（相机轨道 target）
  pub fn center(&self) -> Vec3 {
    let c = (self.aabb_min + self.aabb_max) / 2;
    Vec3::new(c.x as f32, c.y as f32, c.z as f32)
  }

  /// AABB 对角线长度（相机初始距离依据）
  pub fn diagonal(&self) -> f32 {
    let d = self.aabb_max - self.aabb_min;
    Vec3::new(d.x as f32, d.y as f32, d.z as f32).length()
  }
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

  // ---- 次遍：逐实例写体素 ----
  let t1 = std::time::Instant::now();
  let mut written = 0usize;
  let mut dropped = 0usize;
  for inst in &scene.instances {
    if inst.hidden {
      continue;
    }
    let m = &scene.models[inst.model_index];
    let (sx, sy, sz) = (m.size_x as usize, m.size_y as usize, m.size_z as usize);
    let t = &inst.transform;
    for z in 0..sz {
      for y in 0..sy {
        let row = (z * sy + y) * sx;
        for x in 0..sx {
          let c = m.voxels[row + x];
          if c == 0 {
            continue;
          }
          let p = IVec3::from(xform(t, m.size_x, m.size_y, m.size_z, x as f32, y as f32, z as f32))
          + offset;
          if grid.set_voxel_ivec3(p, c).is_some() {
            written += 1;
          } else {
            dropped += 1;
          }
        }
      }
    }
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

/// vox-rs Transform = 列主 mat4（ogt_vox 语义）：平移在 m30..m32。
/// 旋转 **绕模型包围盒中心**（目验裁决 2026-09-05：四片 1/4 球壳绕原点转
/// 会甩成风车状，切图工具的 nTRN pivot 在模型中心，非原点）：
///   p' = R·(p - c) + c + t，c = size/2
/// 坐标系适配：Z-up→Y-up 交换 (X, Z, Y)，不做任何整体旋转。
#[inline]
fn xform(
  t: &vox_rs::Transform,
  sx: u32,
  sy: u32,
  sz: u32,
  x: f32,
  y: f32,
  z: f32,
) -> (i32, i32, i32) {
  let cx = sx as f32 * 0.5;
  let cy = sy as f32 * 0.5;
  let cz = sz as f32 * 0.5;
  let (lx, ly, lz) = (x - cx, y - cy, z - cz);
  let wx = t.m00 * lx + t.m10 * ly + t.m20 * lz + t.m30 + cx;
  let wy = t.m01 * lx + t.m11 * ly + t.m21 * lz + t.m31 + cy;
  let wz = t.m02 * lx + t.m12 * ly + t.m22 * lz + t.m32 + cz;
  (wx.round() as i32, wz.round() as i32, wy.round() as i32)
}

/// 模型盒 8 角经变换后的整数 AABB（90° 旋转 + 整数平移 → round 无误差）
fn transformed_aabb(t: &vox_rs::Transform, sx: u32, sy: u32, sz: u32) -> (IVec3, IVec3) {
  let mut lo = IVec3::splat(i32::MAX);
  let mut hi = IVec3::splat(i32::MIN);
  for cz in [0.0f32, sz as f32] {
    for cy in [0.0, sy as f32] {
      for cx in [0.0, sx as f32] {
        let p = IVec3::from(xform(t, sx, sy, sz, cx, cy, cz));
        lo = lo.min(p);
        hi = hi.max(p);
      }
    }
  }
  (lo, hi)
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

  /// 数据反推 pivot：找「同模型 + R_j = Rot90·R_i」的实例对，
  /// 解 (I − Rot90)·p = t_j − Rot90·t_i（xy 二元线性），统计 p 分布。
  #[test]
  fn diag_solve_pivot() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/vox/nuke.vox");
    if !path.exists() {
      return;
    }
    let file = std::fs::File::open(&path).unwrap();
    let mut r = std::io::BufReader::new(file);
    let scene = vox_rs::Scene::read(&mut r).unwrap();
    // Rot90 = 绕 vox Z 轴 +90°（竖直轴）
    let rot90 = |v: [f32; 3]| -> [f32; 3] { [-v[1], v[0], v[2]] };
    // model_index → [(R, t)]
    let mut by_model: std::collections::HashMap<usize, Vec<([f32; 9], [f32; 3])>> =
      std::collections::HashMap::new();
    for inst in &scene.instances {
      if inst.hidden {
        continue;
      }
      let t = &inst.transform;
      let r = [t.m00, t.m10, t.m20, t.m01, t.m11, t.m21, t.m02, t.m12, t.m22];
      by_model.entry(inst.model_index).or_default().push((
        r,
        [t.m30, t.m31, t.m32],
      ));
    }
    let mut solved = 0;
    for (model, insts) in &by_model {
      if insts.len() < 2 {
        continue;
      }
      for i in 0..insts.len() {
        for j in 0..insts.len() {
          if i == j {
            continue;
          }
          let (ri, ti) = &insts[i];
          let (rj, tj) = &insts[j];
          // R_j == Rot90·R_i？逐列：col(Rot90·R_i, k) = Rot90·col(R_i, k)
          let col = |r: &[f32; 9], k: usize| -> [f32; 3] {
            [r[k * 3], r[k * 3 + 1], r[k * 3 + 2]]
          };
          let rot90_of = |c: [f32; 3]| -> [f32; 3] { [-c[1], c[0], c[2]] };
          let match_rot = (0..3).all(|k| {
            let a = col(rj, k);
            let b = rot90_of(col(ri, k));
            a.iter().zip(&b).all(|(x, y)| (x - y).abs() < 1e-5)
          });
          if !match_rot {
            continue;
          }
          // (I − Rot90) p = t_j − Rot90·t_i，xy 2×2：
          // [ 1  1][-1  1] p = d → p = ((d.x − d.y)/2, (d.x + d.y)/2)
          let d = [tj[0] - rot90_of(*ti)[0], tj[1] - rot90_of(*ti)[1]];
          let px = (d[0] - d[1]) / 2.0;
          let py = (d[0] + d[1]) / 2.0;
          println!(
            "PIVOT model={model} i={i} j={j}: p=({px:.1},{py:.1}) size={:?}",
            {
              let m = &scene.models[*model];
              (m.size_x, m.size_y, m.size_z)
            }
          );
          solved += 1;
          if solved >= 20 {
            return;
          }
        }
      }
    }
    println!("PIVOT solved pairs: {solved}");
  }
}
