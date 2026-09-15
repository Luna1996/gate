//! 测试场景构造器：方块 / 球 / 文字的体素化工具，供单元测试与基准场景库使用。
//!
//! 统一用 1³ 体素写，Brick Tree 自适应合并 uniform leaf。

use glam::IVec3;

use crate::coords::LEVEL_EXTENT;
use crate::volume::VolumeGrid;

/// 填充与轴对齐包围盒相交的所有体素（1³，确定性）。
///
/// 按 4³ 网格分块：完全含于 box 的整块走 `fill_brick`（O(depth) 树路径写入），边缘块逐体素。
pub fn fill_box(grid: &mut VolumeGrid, min: IVec3, extent: IVec3, palette: u8) -> usize {
  assert!(extent.cmpgt(IVec3::ZERO).all());
  let max = min + extent;
  let mut count = 0;

  // 4³ 网格遍历：块起点 = min 向下对齐 4，块终点 = max 向上对齐 4
  let blk_lo = IVec3::new(min.x & !3, min.y & !3, min.z & !3);
  let blk_hi = IVec3::new((max.x + 3) & !3, (max.y + 3) & !3, (max.z + 3) & !3);
  let mut bz = blk_lo.z;
  while bz < blk_hi.z {
    let mut by = blk_lo.y;
    while by < blk_hi.y {
      let mut bx = blk_lo.x;
      while bx < blk_hi.x {
        let b_lo = IVec3::new(bx, by, bz);
        let b_hi = b_lo + IVec3::splat(4);
        if b_lo.cmpge(min).all() && b_hi.cmple(max).all() {
          // 整块在 box 内 → fill_brick 树路径（64 体素 1 次调用）
          if grid.fill_brick(b_lo, 4, palette).is_some() {
            count += 64;
          }
        } else {
          // 边缘块：逐体素
          let mut z = b_lo.z.max(min.z);
          while z < b_hi.z.min(max.z) {
            let mut y = b_lo.y.max(min.y);
            while y < b_hi.y.min(max.y) {
              let mut x = b_lo.x.max(min.x);
              while x < b_hi.x.min(max.x) {
                if grid.set_voxel_ivec3(IVec3::new(x, y, z), palette).is_some() {
                  count += 1;
                }
                x += 1;
              }
              y += 1;
            }
            z += 1;
          }
        }
        bx += 4;
      }
      by += 4;
    }
    bz += 4;
  }
  count
}

/// 用 e³ 对齐 brick 填充 [min, min+extent) 盒（大体积均匀填充专用）。
///
/// extent 各轴必须是 e 的倍数且 min 各轴对齐 e；brick 级树路径写入
/// （[`VolumeGrid::fill_brick`]），零逐体素分裂开销。返回实际写入的 brick 数。
pub fn fill_bricks(grid: &mut VolumeGrid, min: IVec3, extent: IVec3, e: i32, palette: u8) -> usize {
  assert!(
    LEVEL_EXTENT.contains(&e),
    "e 必须是 brick 粒度 {LEVEL_EXTENT:?} 之一（got {e}）"
  );
  assert!(
    extent.x % e == 0 && extent.y % e == 0 && extent.z % e == 0,
    "extent 必须是 e 的倍数（extent={extent} e={e}）"
  );
  assert!(
    min.x % e == 0 && min.y % e == 0 && min.z % e == 0,
    "min 必须对齐 e（min={min} e={e}）"
  );
  let hi = min + extent;
  let mut count = 0;
  let mut z = min.z;
  while z < hi.z {
    let mut y = min.y;
    while y < hi.y {
      let mut x = min.x;
      while x < hi.x {
        if grid.fill_brick(IVec3::new(x, y, z), e, palette).is_some() {
          count += 1;
        }
        x += e;
      }
      y += e;
    }
    z += e;
  }
  count
}

/// 填充球体（体素中心到球心距离 ≤ 半径）。
///
/// 按 4³ 分块：8 角全在球内的整块走 [`VolumeGrid::fill_brick`]（O(depth) 树路径
/// 写入），边缘壳逐体素。
pub fn fill_sphere(grid: &mut VolumeGrid, center: IVec3, radius: i32, palette: u8) -> usize {
  assert!(radius > 0);
  let r2 = radius * radius;
  let lo = center - IVec3::splat(radius);
  let hi = center + IVec3::splat(radius);
  let mut count = 0usize;
  // 4³ 网格分块遍历
  let blk_lo = IVec3::new(lo.x & !3, lo.y & !3, lo.z & !3);
  let blk_hi = IVec3::new((hi.x + 3) & !3, (hi.y + 3) & !3, (hi.z + 3) & !3);
  let mut bz = blk_lo.z;
  while bz <= blk_hi.z {
    let mut by = blk_lo.y;
    while by <= blk_hi.y {
      let mut bx = blk_lo.x;
      while bx <= blk_hi.x {
        // 8 角最大距离平方 ≤ r² → 整块在球内
        let mut max_d2 = 0i64;
        for cz in [0i32, 4] {
          for cy in [0, 4] {
            for cx in [0, 4] {
              let corner = IVec3::new(bx + cx, by + cy, bz + cz);
              let d = (corner - center).as_i64vec3();
              max_d2 = max_d2.max(d.dot(d));
            }
          }
        }
        if max_d2 <= r2 as i64 {
          if grid
            .fill_brick(IVec3::new(bx, by, bz), 4, palette)
            .is_some()
          {
            count += 64;
          }
        } else {
          // 边缘壳：逐体素
          let x_end = (bx + 4).min(hi.x + 1);
          let y_end = (by + 4).min(hi.y + 1);
          let z_end = (bz + 4).min(hi.z + 1);
          let mut z = bz.max(lo.z);
          while z < z_end {
            let mut y = by.max(lo.y);
            while y < y_end {
              let mut x = bx.max(lo.x);
              while x < x_end {
                let d = (IVec3::new(x, y, z) - center).as_i64vec3();
                if d.dot(d) <= r2 as i64
                  && grid.set_voxel_ivec3(IVec3::new(x, y, z), palette).is_some()
                {
                  count += 1;
                }
                x += 1;
              }
              y += 1;
            }
            z += 1;
          }
        }
        bx += 4;
      }
      by += 4;
    }
    bz += 4;
  }
  count
}

/// 5×7 位图字体
const FONT: &[(u8, [u8; 7])] = &[
  (b'0', [0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E]),
  (b'1', [0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E]),
  (b'2', [0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F]),
  (b'3', [0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E]),
  (b'4', [0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02]),
  (b'5', [0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E]),
  (b'6', [0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E]),
  (b'7', [0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08]),
  (b'8', [0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E]),
  (b'9', [0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C]),
  (b'A', [0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11]),
  (b'B', [0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E]),
  (b'C', [0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E]),
  (b'D', [0x1C, 0x12, 0x11, 0x11, 0x11, 0x12, 0x1C]),
  (b'E', [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F]),
  (b'F', [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10]),
  (b'G', [0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0F]),
  (b'H', [0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11]),
  (b'I', [0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E]),
  (b'J', [0x07, 0x02, 0x02, 0x02, 0x02, 0x12, 0x0C]),
  (b'K', [0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11]),
  (b'L', [0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F]),
  (b'M', [0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11]),
  (b'N', [0x11, 0x11, 0x19, 0x15, 0x13, 0x11, 0x11]),
  (b'O', [0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E]),
  (b'P', [0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10]),
  (b'Q', [0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D]),
  (b'R', [0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11]),
  (b'S', [0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E]),
  (b'T', [0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04]),
  (b'U', [0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E]),
  (b'V', [0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04]),
  (b'W', [0x11, 0x11, 0x11, 0x15, 0x15, 0x1B, 0x11]),
  (b'X', [0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11]),
  (b'Y', [0x11, 0x11, 0x11, 0x0A, 0x04, 0x04, 0x04]),
  (b'Z', [0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F]),
];

fn glyph(ch: u8) -> &'static [u8; 7] {
  const BLANK: [u8; 7] = [0; 7];
  FONT
    .iter()
    .find(|(c, _)| *c == ch.to_ascii_uppercase())
    .map(|(_, rows)| rows)
    .unwrap_or(&BLANK)
}

pub fn text_size(text: &str) -> IVec3 {
  let cols = if text.is_empty() {
    0
  } else {
    text.len() as i32 * 6 - 1
  };
  IVec3::new(cols, 7, 1)
}

pub fn draw_text(grid: &mut VolumeGrid, origin: IVec3, text: &str, palette: u8) -> usize {
  let mut count = 0;
  for (gi, ch) in text.bytes().enumerate() {
    let rows = glyph(ch);
    for (r, bits) in rows.iter().enumerate() {
      for col in 0..5 {
        if bits & (1 << (4 - col)) != 0 {
          let p = origin + IVec3::new(gi as i32 * 6 + col, r as i32, 0);
          if grid.set_voxel_ivec3(p, palette).is_some() {
            count += 1;
          }
        }
      }
    }
  }
  count
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::VoxelCoord;

  #[test]
  fn box_readback() {
    let mut grid = VolumeGrid::new();
    let n = fill_box(&mut grid, IVec3::new(0, 0, 0), IVec3::splat(5), 1);
    assert_eq!(n, 125);
    assert_eq!(grid.get_voxel(VoxelCoord::new(2, 2, 2)), Some(1));
    assert_eq!(grid.get_voxel(VoxelCoord::new(5, 2, 2)), None);
  }

  #[test]
  fn box_cross_chunk() {
    let mut grid = VolumeGrid::new();
    // 600³ box 跨多个 chunk
    let n = fill_box(&mut grid, IVec3::new(-1, 0, 0), IVec3::new(5, 1, 1), 2);
    assert_eq!(n, 5);
    assert!(grid.chunk_count() >= 2);
  }

  #[test]
  fn sphere_center_in_solid() {
    let mut grid = VolumeGrid::new();
    fill_sphere(&mut grid, IVec3::new(100, 100, 100), 10, 4);
    assert_eq!(grid.get_voxel(VoxelCoord::new(100, 100, 100)), Some(4));
    assert_eq!(grid.get_voxel(VoxelCoord::new(110, 110, 110)), None);
  }

  #[test]
  fn text_glyph_count() {
    let mut grid = VolumeGrid::new();
    let want = "GATE"
      .bytes()
      .map(|c| {
        let rows = glyph(c);
        rows.iter().map(|r| r.count_ones() as usize).sum::<usize>()
      })
      .sum();
    assert_eq!(draw_text(&mut grid, IVec3::ZERO, "GATE", 5), want);
    assert_eq!(text_size("GATE"), IVec3::new(23, 7, 1));
  }
}
