use std::collections::HashMap;
use std::io::BufReader;
use std::path::Path;

use gate_voxel::{
  CHUNK_SIZE, ChunkCoord, ChunkTree, PALETTE_INDEX_MAX, PaletteEntry, PaletteFlags, PaletteId,
  PbrOverrides, VolumeGrid, override_byte,
};
use glam::IVec3;
use rayon::prelude::*;

pub struct VoxSceneInfo {
  pub aabb_min: IVec3,
  pub aabb_max: IVec3,
  pub instances_used: usize,
  pub voxels_written: usize,
  pub voxels_dropped: usize,
}

pub fn scan_vox_models() -> Vec<String> {
  let dir = gate_render::assets_dir().join("vox");
  let Ok(entries) = std::fs::read_dir(&dir) else {
    bevy::log::warn!("vox 目录不可读（{}）→ 模型下拉回退 nuke", dir.display());
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
    bevy::log::warn!("vox 目录无 .vox 文件（{}）→ 模型下拉回退 nuke", dir.display());
    return vec!["nuke".to_string()];
  }
  names
}

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
    "VOX LOAD {} ver={} models={} instances={} {:?}",
    path.display(),
    scene.file_version,
    scene.models.len(),
    scene.instances.len(),
    t0.elapsed(),
  );

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
  let offset = anchor - IVec3::new((lo.x + hi.x) / 2, lo.y, (lo.z + hi.z) / 2);

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
  let mut buckets: HashMap<ChunkCoord, Vec<u32>> = HashMap::new();
  for map in per_instance {
    for (cc, mut v) in map {
      buckets.entry(cc).or_default().append(&mut v);
    }
  }
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
    "VOX BUILD written={written} dropped={dropped} aabb=[{}]-[{}] {:?}",
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

fn paint_vox_palette(grid: &mut VolumeGrid, scene: &vox_rs::Scene, used_pal: &[bool; 256]) {
  let pal = grid.palette_mut();
  let mut painted = 0usize;
  for (i, &used) in used_pal.iter().enumerate().take(256).skip(1) {
    if !used {
      continue;
    }
    let rgba = scene.palette.colors[i];
    let e = matl_to_entry([rgba.r, rgba.g, rgba.b], &scene.materials[i]);
    pal.set(PaletteId(i as u16), e);
    painted += 1;
  }
  let em: Vec<u16> = (1..=255u16).filter(|&i| pal.get(PaletteId(i)).emissive > 0).collect();
  bevy::log::info!(
    "VOX MATERIAL emissive={} {:?} 引用色号={} 空槽={}",
    em.len(),
    em,
    painted,
    PALETTE_INDEX_MAX as usize - painted
  );
  bevy::log::info!(
    "VOX MATL 映射 _rough→roughness _emit→emissive _metal→metallic PBR_ASSET={VOX_PBR_ASSET:?}"
  );
}

pub const VOX_PBR_ASSET: Option<u16> = None;

pub fn matl_to_entry(color: [u8; 3], mat: &vox_rs::Material) -> PaletteEntry {
  let byte =
    |v: Option<f32>, default: u8| v.map(|x| (x.clamp(0.0, 1.0) * 255.0) as u8).unwrap_or(default);
  match VOX_PBR_ASSET {
    None => PaletteEntry {
      color,
      roughness: byte(mat.rough, 200),
      emissive: byte(mat.emit, 0),
      metallic: byte(mat.metal, 0),
      ..Default::default()
    },
    Some(asset) => PaletteEntry::pbr(
      asset,
      PbrOverrides {
        roughness: mat.rough.map(|v| override_byte(v.clamp(0.0, 1.0))).unwrap_or(0),
        metallic: mat.metal.map(|v| override_byte(v.clamp(0.0, 1.0))).unwrap_or(0),
        emissive: mat.emit.map(|v| override_byte(v.clamp(0.0, 1.0))).unwrap_or(0),
        transmission: 0,
        specular: 0,
      },
      PaletteFlags::default(),
    ),
  }
}

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
  (wx, wz, -wy)
}

#[inline]
fn instance_flip(t: &vox_rs::Transform) -> IVec3 {
  IVec3::new(
    t.m00.min(t.m10).min(t.m20).min(0.0) as i32,
    t.m01.min(t.m11).min(t.m21).min(0.0) as i32,
    t.m02.min(t.m12).min(t.m22).min(0.0) as i32,
  )
}

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

fn bucket_instance(
  inst: &vox_rs::Instance,
  models: &[vox_rs::Model],
  offset: IVec3,
) -> HashMap<ChunkCoord, Vec<u32>> {
  bucket_model(&models[inst.model_index], &inst.transform, instance_flip(&inst.transform), offset)
}

fn bucket_model(
  m: &vox_rs::Model,
  t: &vox_rs::Transform,
  flip: IVec3,
  offset: IVec3,
) -> HashMap<ChunkCoord, Vec<u32>> {
  let (sx, sy, sz) = (m.size_x as usize, m.size_y as usize, m.size_z as usize);
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