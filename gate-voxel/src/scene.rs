//! 测试场景构造器（Phase 0，Brick Tree 适配）
//!
//! 方块 / 球 / 文字的体素化工具，供单元测试、基准场景库与 Phase 1 渲染验证使用。
//! 统一用 1³ 体素写，Brick Tree 自适应合并 uniform leaf。

use glam::IVec3;

use crate::coords::VoxelCoord;
use crate::volume::VolumeGrid;

/// 填充与轴对齐包围盒相交的所有体素（1³，确定性）
pub fn fill_box(grid: &mut VolumeGrid, min: IVec3, extent: IVec3, palette: u8) -> usize {
  assert!(extent.cmpgt(IVec3::ZERO).all());
  let max = min + extent;
  let mut count = 0;
  let mut z = min.z;
  while z < max.z {
    let mut y = min.y;
    while y < max.y {
      let mut x = min.x;
      while x < max.x {
        if grid.set_voxel_ivec3(IVec3::new(x, y, z), palette).is_some() {
          count += 1;
        }
        x += 1;
      }
      y += 1;
    }
    z += 1;
  }
  count
}

/// 填充球体（立方中心到球心距离 ≤ 半径）
pub fn fill_sphere(grid: &mut VolumeGrid, center: IVec3, radius: i32, palette: u8) -> usize {
  assert!(radius > 0);
  let r2 = radius * radius;
  let lo = center - IVec3::splat(radius);
  let hi = center + IVec3::splat(radius);
  let mut count = 0;
  let mut z = lo.z;
  while z <= hi.z {
    let mut y = lo.y;
    while y <= hi.y {
      let mut x = lo.x;
      while x <= hi.x {
        let d = IVec3::new(x, y, z) - center;
        if d.dot(d) <= r2 {
          if grid.set_voxel_ivec3(IVec3::new(x, y, z), palette).is_some() {
            count += 1;
          }
        }
        x += 1;
      }
      y += 1;
    }
    z += 1;
  }
  count
}

/// 5×7 位图字体（和旧版相同）
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
  let cols = if text.is_empty() { 0 } else { text.len() as i32 * 6 - 1 };
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
