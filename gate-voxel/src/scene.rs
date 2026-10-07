use glam::{IVec3, Vec3};

use crate::coords::LEVEL_EXTENT;
use crate::palette::PaletteId;
use crate::volume::VolumeGrid;

pub fn fill_box(
  grid: &mut VolumeGrid,
  min: IVec3,
  extent: IVec3,
  palette: impl Into<PaletteId>,
) -> usize {
  let palette = palette.into();
  assert!(extent.cmpgt(IVec3::ZERO).all());
  let max = min + extent;
  let mut count = 0;

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
          if grid.fill_brick(b_lo, 4, palette).is_some() {
            count += 64;
          }
        } else {
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

pub fn fill_bricks(
  grid: &mut VolumeGrid,
  min: IVec3,
  extent: IVec3,
  e: i32,
  palette: impl Into<PaletteId>,
) -> usize {
  let palette = palette.into();
  assert!(LEVEL_EXTENT.contains(&e), "e 必须是 brick 粒度 {LEVEL_EXTENT:?} 之一（got {e}）");
  assert!(
    extent.x % e == 0 && extent.y % e == 0 && extent.z % e == 0,
    "extent 必须是 e 的倍数（extent={extent} e={e}）"
  );
  assert!(min.x % e == 0 && min.y % e == 0 && min.z % e == 0, "min 必须对齐 e（min={min} e={e}）");
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

pub fn fill_sphere(
  grid: &mut VolumeGrid,
  center: IVec3,
  radius: i32,
  palette: impl Into<PaletteId>,
) -> usize {
  let palette = palette.into();
  assert!(radius > 0);
  let r2 = radius * radius;
  let lo = center - IVec3::splat(radius);
  let hi = center + IVec3::splat(radius);
  let mut count = 0usize;

  let blk_lo = IVec3::new(lo.x & !3, lo.y & !3, lo.z & !3);
  let blk_hi = IVec3::new((hi.x + 3) & !3, (hi.y + 3) & !3, (hi.z + 3) & !3);
  let mut bz = blk_lo.z;
  while bz <= blk_hi.z {
    let mut by = blk_lo.y;
    while by <= blk_hi.y {
      let mut bx = blk_lo.x;
      while bx <= blk_hi.x {
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
          if grid.fill_brick(IVec3::new(bx, by, bz), 4, palette).is_some() {
            count += 64;
          }
        } else {
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
  FONT.iter().find(|(c, _)| *c == ch.to_ascii_uppercase()).map(|(_, rows)| rows).unwrap_or(&BLANK)
}

pub fn text_size(text: &str) -> IVec3 {
  let cols = if text.is_empty() { 0 } else { text.len() as i32 * 6 - 1 };
  IVec3::new(cols, 7, 1)
}

pub fn draw_text(
  grid: &mut VolumeGrid,
  origin: IVec3,
  text: &str,
  palette: impl Into<PaletteId>,
) -> usize {
  let palette = palette.into();
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

pub type DisplaceFn<'a> = dyn Fn(Vec3, Vec3) -> f32 + 'a;

#[derive(Clone, Copy)]
pub struct Displace<'a> {
  pub f: &'a DisplaceFn<'a>,
  pub bound: f32,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FillStats {
  pub voxels: usize,
  pub whole_bricks: usize,
  pub shell_voxels: usize,
}

#[derive(Clone, Copy)]
enum Shape {
  Box { min: IVec3, max: IVec3 },
  Sphere { center: IVec3, radius: i32 },
}

impl Shape {
  fn sdf(&self, p: Vec3) -> f32 {
    match *self {
      Shape::Box { min, max } => {
        let (lo, hi) = (min.as_vec3(), max.as_vec3());
        (lo - p).max(p - hi).max_element()
      }
      Shape::Sphere { center, radius } => (p - center.as_vec3()).length() - radius as f32,
    }
  }

  fn normal(&self, p: Vec3) -> Vec3 {
    match *self {
      Shape::Box { min, max } => {
        let (lo, hi) = (min.as_vec3(), max.as_vec3());
        let q = (lo - p).max(p - hi);
        let axis = if q.x >= q.y && q.x >= q.z {
          0
        } else if q.y >= q.z {
          1
        } else {
          2
        };
        let center = (lo + hi) * 0.5;
        let sign = if p[axis] >= center[axis] { 1.0 } else { -1.0 };
        let mut n = Vec3::ZERO;
        n[axis] = sign;
        n
      }
      Shape::Sphere { center, .. } => (p - center.as_vec3()).normalize_or_zero(),
    }
  }

  fn aabb(&self) -> (IVec3, IVec3) {
    match *self {
      Shape::Box { min, max } => (min, max),
      Shape::Sphere { center, radius } => {
        (center - IVec3::splat(radius), center + IVec3::splat(radius))
      }
    }
  }
}

fn fill_shape(
  grid: &mut VolumeGrid,
  shape: Shape,
  grain: i32,
  palette: PaletteId,
  disp: Option<Displace<'_>>,
) -> FillStats {
  debug_assert!(LEVEL_EXTENT.contains(&grain), "grain 必须是 brick 粒度 {LEVEL_EXTENT:?} 之一");
  let bound = disp.as_ref().map_or(0.0, |d| d.bound.max(0.0));
  let grow = bound.ceil() as i32;
  let (s_lo, s_hi) = shape.aabb();
  let g = IVec3::splat(grain);
  let lo = (s_lo - IVec3::splat(grow)).div_euclid(g) * g;
  let hi = (s_hi + IVec3::splat(grow + 1) + g - IVec3::ONE).div_euclid(g) * g;
  let lip = (grain - 1) as f32 * 0.5 * 3.0_f32.sqrt();

  let mut stats = FillStats::default();
  let mut bz = lo.z;
  while bz < hi.z {
    let mut by = lo.y;
    while by < hi.y {
      let mut bx = lo.x;
      while bx < hi.x {
        let b_lo = IVec3::new(bx, by, bz);
        let mut max_sdf = f32::MIN;
        for dz in [0, grain - 1] {
          for dy in [0, grain - 1] {
            for dx in [0, grain - 1] {
              max_sdf = max_sdf.max(shape.sdf((b_lo + IVec3::new(dx, dy, dz)).as_vec3()));
            }
          }
        }
        if max_sdf <= -bound {
          grid.fill_brick(b_lo, grain, palette);
          stats.whole_bricks += 1;
          stats.voxels += (grain as usize) * (grain as usize) * (grain as usize);
          bx += grain;
          continue;
        }
        let center = (b_lo * 2 + IVec3::splat(grain - 1)).as_vec3() * 0.5;
        if shape.sdf(center) - lip >= bound {
          bx += grain;
          continue;
        }
        let mut z = b_lo.z;
        while z < b_lo.z + grain {
          let mut y = b_lo.y;
          while y < b_lo.y + grain {
            let mut x = b_lo.x;
            while x < b_lo.x + grain {
              let p = Vec3::new(x as f32, y as f32, z as f32);
              let keep = match &disp {
                Some(d) => shape.sdf(p) <= (d.f)(p, shape.normal(p)),
                None => shape.sdf(p) <= 0.0,
              };
              stats.shell_voxels += 1;
              if keep {
                grid.set_voxel_ivec3(IVec3::new(x, y, z), palette);
                stats.voxels += 1;
              }
              x += 1;
            }
            y += 1;
          }
          z += 1;
        }
        bx += grain;
      }
      by += grain;
    }
    bz += grain;
  }
  stats
}

pub fn fill_box_displaced(
  grid: &mut VolumeGrid,
  min: IVec3,
  extent: IVec3,
  palette: impl Into<PaletteId>,
  disp: Option<Displace<'_>>,
) -> FillStats {
  assert!(extent.cmpgt(IVec3::ZERO).all());
  let shape = Shape::Box { min, max: min + extent - IVec3::ONE };
  fill_shape(grid, shape, 4, palette.into(), disp)
}

pub fn fill_sphere_displaced(
  grid: &mut VolumeGrid,
  center: IVec3,
  radius: i32,
  palette: impl Into<PaletteId>,
  disp: Option<Displace<'_>>,
) -> FillStats {
  assert!(radius > 0);
  fill_shape(grid, Shape::Sphere { center, radius }, 4, palette.into(), disp)
}

pub fn fill_bricks_displaced(
  grid: &mut VolumeGrid,
  min: IVec3,
  extent: IVec3,
  e: i32,
  palette: impl Into<PaletteId>,
  disp: Option<Displace<'_>>,
) -> FillStats {
  assert!(LEVEL_EXTENT.contains(&e), "e 必须是 brick 粒度 {LEVEL_EXTENT:?} 之一（got {e}）");
  assert!(
    extent.x % e == 0 && extent.y % e == 0 && extent.z % e == 0,
    "extent 必须是 e 的倍数（extent={extent} e={e}）"
  );
  assert!(min.x % e == 0 && min.y % e == 0 && min.z % e == 0, "min 必须对齐 e（min={min} e={e}）");
  let shape = Shape::Box { min, max: min + extent - IVec3::ONE };
  fill_shape(grid, shape, e, palette.into(), disp)
}
