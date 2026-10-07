use std::collections::HashMap;

use gate_voxel::{
  PALETTE_INDEX_MAX, Palette, PaletteEntry, PaletteId, VolumeTransform, Volumes, VoxelCoord,
};
use glam::{IVec3, Vec3};

use crate::field::{Field, tight_bounds, world_aabb_of};

pub fn dissolve_into_main(volumes: &mut Volumes, grid_index: usize, tr: VolumeTransform) -> usize {
  if grid_index == 0 || grid_index >= volumes.list.len() {
    return 0;
  }
  let Some((lo, hi)) = tight_bounds(&volumes.list[grid_index]) else { return 0 };
  let remap = merge_palette(volumes, grid_index);
  if remap.is_empty() {
    return 0;
  }
  let (mn, mx) = world_aabb_of((lo, hi), tr);
  let w_lo = mn.floor().as_ivec3();
  let w_hi = mx.ceil().as_ivec3();
  let (main_part, objects) = volumes.list.split_at_mut(1);
  let main = &mut main_part[0];
  let src = &objects[grid_index - 1];
  let field = Field::new_at(src, tr);
  let mut written = 0usize;
  for wz in w_lo.z..w_hi.z {
    for wy in w_lo.y..w_hi.y {
      for wx in w_lo.x..w_hi.x {
        let center = Vec3::new(wx as f32, wy as f32, wz as f32) + Vec3::splat(0.5);
        let v = field.to_local(center).floor().as_ivec3();
        if (v.cmplt(lo) | v.cmpgt(hi)).any() {
          continue;
        }
        let Some(pid) = src.get_voxel(VoxelCoord::from_ivec3(v)) else { continue };
        if pid.is_air() {
          continue;
        }
        let Some(&dst) = remap.get(&pid.get()) else { continue };
        if main.set_voxel_ivec3(IVec3::new(wx, wy, wz), PaletteId(dst)).is_some() {
          written += 1;
        }
      }
    }
  }
  volumes.despawn_object(grid_index as i32 - 1);
  written
}

fn merge_palette(volumes: &mut Volumes, grid_index: usize) -> HashMap<u16, u16> {
  let mut used: Vec<(u16, PaletteEntry)> = Vec::new();
  {
    let pal = volumes.list[grid_index].palette();
    for i in 1u16..=PALETTE_INDEX_MAX {
      let id = PaletteId(i);
      if pal.occupied(id) {
        used.push((i, *pal.get(id)));
      }
    }
  }
  if used.is_empty() {
    return HashMap::new();
  }
  let (main_part, _) = volumes.list.split_at_mut(1);
  let main = &mut main_part[0];
  let mut remap = HashMap::with_capacity(used.len());
  for (sid, entry) in used {
    let slot = match find_same(main.palette(), entry) {
      Some(d) => d,
      None => {
        let d = first_free(main.palette()).unwrap_or(PALETTE_INDEX_MAX);
        main.palette_mut().set(PaletteId(d), entry);
        d
      }
    };
    remap.insert(sid, slot);
  }
  remap
}

fn find_same(pal: &Palette, entry: PaletteEntry) -> Option<u16> {
  (1u16..=PALETTE_INDEX_MAX).find(|&i| {
    let id = PaletteId(i);
    pal.occupied(id) && *pal.get(id) == entry
  })
}

fn first_free(pal: &Palette) -> Option<u16> {
  (1u16..=PALETTE_INDEX_MAX).find(|&i| pal.is_empty_slot(PaletteId(i)))
}
