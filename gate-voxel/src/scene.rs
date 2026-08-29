//! 测试场景构造器（P1.6）
//!
//! 方块 / 球 / 文字的体素化工具，供单元测试、基准场景库（确定性回放）与
//! P2 渲染验证使用。支持任意层级，可多分辨率混合叠加。
//!
//! 体素化语义（确定性，无随机）：
//! - 方块：所有与包围盒相交的层级立方
//! - 球：立方中心落在半径内
//! - 文字：5×7 位图字体，每个像素一个层级立方，XY 平面 + 单层厚度
//!
//! 返回值统一为「形状包含的层级立方数」（含同色重写，不含实际生效判断），
//! 只依赖几何形状，可作断言基线。

use glam::IVec3;

use crate::coords::{LEVEL_SUB_EXTENT, Level};
use crate::grid::TileGrid;

/// 层级立方的边长（最细格数）
fn extent_of(level: Level) -> i32 {
  LEVEL_SUB_EXTENT[level as usize]
}

/// 欧氏下取整到对齐网格（负坐标正确落到邻接网格）
fn floor_aligned(v: i32, e: i32) -> i32 {
  v.div_euclid(e) * e
}

/// 填充与轴对齐包围盒相交的所有层级立方
///
/// `min` 为包围盒最小角（最细格坐标，可负），`extent` 为各轴尺寸（> 0）。
pub fn fill_box(
  grid: &mut TileGrid,
  min: IVec3,
  extent: IVec3,
  level: Level,
  palette: u8,
) -> usize {
  assert!(extent.cmpgt(IVec3::ZERO).all(), "extent must be positive");
  let e = extent_of(level);
  let max = min + extent;
  let mut count = 0;
  let mut z = floor_aligned(min.z, e);
  while z < max.z {
    let mut y = floor_aligned(min.y, e);
    while y < max.y {
      let mut x = floor_aligned(min.x, e);
      while x < max.x {
        grid.set_voxel(IVec3::new(x, y, z), level, palette);
        count += 1;
        x += e;
      }
      y += e;
    }
    z += e;
  }
  count
}

/// 填充球体（立方中心到球心距离 ≤ 半径）
pub fn fill_sphere(
  grid: &mut TileGrid,
  center: IVec3,
  radius: i32,
  level: Level,
  palette: u8,
) -> usize {
  assert!(radius > 0, "radius must be positive");
  let e = extent_of(level);
  let r2 = radius * radius;
  let mut count = 0;
  // 遍历球包围盒内的对齐立方，判定中心
  let lo = center - IVec3::splat(radius + e);
  let hi = center + IVec3::splat(radius + e);
  let mut z = floor_aligned(lo.z, e);
  while z <= hi.z {
    let mut y = floor_aligned(lo.y, e);
    while y <= hi.y {
      let mut x = floor_aligned(lo.x, e);
      while x <= hi.x {
        let c = IVec3::new(x + e / 2, y + e / 2, z + e / 2);
        let d = c - center;
        if d.dot(d) <= r2 {
          grid.set_voxel(IVec3::new(x, y, z), level, palette);
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

/// 5×7 位图字体：bit4 = 最左列，行 0 = 顶部。未知字符按空格处理
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

/// 文字占据的包围盒尺寸（最细格单位）
pub fn text_size(text: &str, level: Level) -> IVec3 {
  let e = extent_of(level);
  let cols = if text.is_empty() {
    0
  } else {
    text.len() as i32 * 6 - 1
  };
  IVec3::new(cols * e, 7 * e, e)
}

/// 在 XY 平面绘制文字（行 0 顶部，字距 1 列，z 单层厚度）
pub fn draw_text(
  grid: &mut TileGrid,
  origin: IVec3,
  text: &str,
  level: Level,
  palette: u8,
) -> usize {
  let e = extent_of(level);
  let mut count = 0;
  for (gi, ch) in text.bytes().enumerate() {
    let rows = glyph(ch);
    for (r, bits) in rows.iter().enumerate() {
      for col in 0..5 {
        if bits & (1 << (4 - col)) != 0 {
          let p = origin + IVec3::new((gi as i32 * 6 + col) * e, r as i32 * e, 0);
          grid.set_voxel(p, level, palette);
          count += 1;
        }
      }
    }
  }
  count
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::coords::TileCoord;

  #[test]
  fn box_aligned_count_and_readback() {
    let mut grid = TileGrid::new();
    // L4（1 最细格）5×5×5 = 125
    let n = fill_box(&mut grid, IVec3::new(0, 0, 0), IVec3::splat(5), 4, 1);
    assert_eq!(n, 125);
    assert_eq!(grid.get_voxel(IVec3::new(2, 2, 2)), Some(1));
    assert_eq!(grid.get_voxel(IVec3::new(5, 2, 2)), None);
  }

  #[test]
  fn box_coarse_covers_full_cells() {
    let mut grid = TileGrid::new();
    // L0 立方边长 16：8×8×8 = 512 个基元胞
    let n = fill_box(&mut grid, IVec3::new(0, 0, 0), IVec3::splat(128), 0, 2);
    assert_eq!(n, 512);
    // 基元胞内任意点读回同色
    assert_eq!(grid.get_voxel(IVec3::new(15, 15, 15)), Some(2));
    assert_eq!(grid.get_voxel(IVec3::new(0, 0, 0)), Some(2));
    assert_eq!(grid.get_voxel(IVec3::new(128, 0, 0)), None);
  }

  #[test]
  fn box_spans_negative_and_tiles() {
    let mut grid = TileGrid::new();
    // 跨 tile 边界（TILE_SUB=512）且含负坐标，L4 逐格
    let n = fill_box(&mut grid, IVec3::new(-2, 0, 0), IVec3::new(5, 1, 1), 4, 3);
    assert_eq!(n, 5);
    assert_eq!(grid.get_voxel(IVec3::new(-1, 0, 0)), Some(3)); // tile (-1,0,0)
    assert_eq!(grid.get_voxel(IVec3::new(0, 0, 0)), Some(3)); // tile (0,0,0)
    assert_eq!(grid.get_voxel(IVec3::new(2, 0, 0)), Some(3));
    assert_eq!(grid.get_voxel(IVec3::new(3, 0, 0)), None);
    assert!(grid.tile(TileCoord::new(-1, 0, 0)).is_some());
    assert!(grid.tile(TileCoord::new(0, 0, 0)).is_some());
  }

  #[test]
  fn sphere_center_in_solid() {
    let mut grid = TileGrid::new();
    let n = fill_sphere(&mut grid, IVec3::new(100, 100, 100), 10, 4, 4);
    // 中心与近轴点必在球内
    assert_eq!(grid.get_voxel(IVec3::new(100, 100, 100)), Some(4));
    assert_eq!(grid.get_voxel(IVec3::new(105, 100, 100)), Some(4));
    // 角落远点不在
    assert_eq!(grid.get_voxel(IVec3::new(110, 110, 110)), None);
    // 计数与遍历一致（重跑不增）
    assert_eq!(
      fill_sphere(&mut grid, IVec3::new(100, 100, 100), 10, 4, 4),
      n
    );
  }

  #[test]
  fn text_glyph_count_and_readback() {
    let mut grid = TileGrid::new();
    // 字形逐个核对像素数
    assert_eq!(draw_text(&mut grid, IVec3::ZERO, "GATE", 4, 5), {
      let want: usize = "GATE"
        .bytes()
        .map(|c| {
          glyph(c)
            .iter()
            .map(|r| r.count_ones() as usize)
            .sum::<usize>()
        })
        .sum();
      want
    });
    // 'G' 第 1 行左列有体素（首行 0x0E 左上角为空）
    assert_eq!(grid.get_voxel(IVec3::new(0, 1, 0)), Some(5));
    // 字间空隙（'G' 宽 5 + 1 间隔 → x=5 列为空）
    assert_eq!(grid.get_voxel(IVec3::new(5, 0, 0)), None);
    // text_size
    assert_eq!(text_size("GATE", 4), IVec3::new(23, 7, 1));
  }

  #[test]
  fn mixed_resolution_coexist() {
    let mut grid = TileGrid::new();
    // L0 大方块 + L4 小球 + L2 文字，三者读回互不干扰
    fill_box(&mut grid, IVec3::new(0, 0, 0), IVec3::splat(64), 0, 1);
    fill_sphere(&mut grid, IVec3::new(200, 200, 200), 8, 4, 2);
    draw_text(&mut grid, IVec3::new(0, 100, 0), "OK", 2, 3);
    assert_eq!(grid.get_voxel(IVec3::new(1, 1, 1)), Some(1));
    assert_eq!(grid.get_voxel(IVec3::new(200, 200, 200)), Some(2));
    // 'O' 第 1 行左列（L2 每像素 4 最细格）
    assert_eq!(grid.get_voxel(IVec3::new(0, 104, 0)), Some(3));
    // 同基元胞先写粗后写细：细写覆盖粗区域，L4 读到新色
    grid.set_voxel(IVec3::new(1, 1, 1), 4, 6);
    assert_eq!(grid.get_voxel(IVec3::new(1, 1, 1)), Some(6));
    // 同 L4 邻点仍是粗层颜色（细分只影响子区域）
    assert_eq!(grid.get_voxel(IVec3::new(2, 2, 2)), Some(1));
  }
}
