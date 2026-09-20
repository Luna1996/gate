//! 测试场景构造器：方块 / 球 / 文字的体素化工具，供单元测试与基准场景库使用。
//! 统一用 1³ 体素写，Brick Tree 自适应合并 uniform leaf。
//! 另有 MT6 的 **CSG 表面位移**（材质高度图 → 真实体素几何，见 `docs/PLAN.md` §3 D2 与文末「MT6」段）。

use glam::{IVec3, Vec3};

use crate::coords::LEVEL_EXTENT;
use crate::palette::PaletteId;
use crate::volume::VolumeGrid;

/// 填充与轴对齐包围盒相交的所有体素（1³）；`palette` 用 `impl Into<PaletteId>`，便于槽号字面量（`2` / `7`）。
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

/// 用 e³ 对齐 brick 填充 [min, min+extent) 盒，返回写入的 brick 数。
/// extent 各轴须为 e 的倍数，min 各轴须对齐 e。
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

/// 填充球体（体素中心到球心距离 ≤ 半径），返回写入体素数。
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

// ============================================================================
// MT6 · CSG 表面位移（材质高度图 → 真实体素几何）
// ============================================================================
//
// **为什么在这里**：`docs/PLAN.md` §3 D2 定稿「凹凸的来源 = 真实体素几何」—— 不 bake 法线、
// 不用法线贴图造假凹凸，而是在**体素化那一刻**（Douglas #22 说的 "CSG side"）按材质高度图
// 位移表面，产物就是普通体素：凹凸处真的一格一格错开 ⇒ DDA 能命中、能投影阴影、GI 能正确遮蔽
// （凹槽的暗部由真实几何的遮蔽自然给出，不需要 AO 贴图，见 §3 D1「AO」一行）。
//
// **零渲染依赖（硬约束 8）**：位移的输入只能是**普通 CPU 闭包**（[`DisplaceFn`]）。
// 高度图（`assets/textures/pbr/<id>/<id>_height.png`）在 `gate-app` 侧解码成普通数组
// （`gate-app/src/height_field.rs`），本 crate 不认识任何贴图 / GPU 类型。
//
// **只动表面一层（为什么能实时）**：位移幅度上界 `bound` 决定哪些块能走整块快路径
// （[`VolumeGrid::fill_brick`]，O(深度)）—— 块内所有格点在位移后仍实心 ⇒ 整块写；
// 只有表面壳层退化为逐体素。`bound ≤ e`（块粒度）时壳层就是最外一层 e³ 块
// （§5 R7「位移让体素数膨胀」的缓解措施：幅度设上限 + 只让表面退化）。
//
// **一次性产物（MT6-5）**：本段产出的就是普通体素 —— 位移后**不存在**"位移参数 / 位移重算"状态
// （本 crate 不存任何"材质 → 高度图"的引用），因此编辑笔触与位移结果互不干扰：
// 笔触覆盖即普通体素，擦除即普通空格。

/// 位移的**带符号高度源**（MT6-1 / MT6-3）：`(体素点, 单位外法线) → 沿法线的偏移体素数`。
///
/// | 约定 | 取值 |
/// |---|---|
/// | `p` | **整数格点坐标**（与 [`fill_sphere`] 的"体素中心"口径一致，不是体素最小角） |
/// | `n` | **形状的**单位外法线（盒 = 主导轴朝外、球 = 径向），由本 crate 解析给出 |
/// | 返回 `> 0` | 沿法线**外推**（体素长出来） |
/// | 返回 `< 0` | **内缩**（该格不写 ⇒ 被挖掉） |
/// | 返回 `0` | 保持原表面 |
///
/// **必须是纯 CPU 闭包**（硬约束 8）：不许把 `bevy_image` / GPU 纹理类型带进本 crate。
/// 典型实现见 `gate-app/src/height_field.rs`：切空间 UV → `fract` 平铺 → 双线性采样 → ×幅度 − 偏置。
pub type DisplaceFn<'a> = dyn Fn(Vec3, Vec3) -> f32 + 'a;

/// 位移参数：偏移闭包 + **幅度上界**（体素，`>= 0`）。见 [`DisplaceFn`] 的约定。
///
/// `bound` 必须是 `|f(..)|` 的**上界**（不是平均幅度）：它只用于"哪些 e³ 块能整块写"的判定 ——
/// 块内所有格点的形状有符号距离都 ≤ `-bound` ⇒ 位移后仍实心 ⇒ `fill_brick` 整块写。
/// 写小了会漏掉本该整块写的块（只是慢，不会错）；写大了会漏写（错）—— 所以宁可写大一点。
#[derive(Clone, Copy)]
pub struct Displace<'a> {
  pub f: &'a DisplaceFn<'a>,
  pub bound: f32,
}

/// 位移填充的统计：MT6-3 验收的"可打印写入体素数" + MT6-6 的体量代价。
/// （本 crate 零日志依赖 ⇒ 这里只返回数字，**由调用方打印**。）
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FillStats {
  /// 判定为实心并被写入（或尝试写入）的体素数。
  /// ⚠️ 与基础函数的计数口径略有不同：基础函数只在 `set_voxel` 真的改变了格子时才 +1
  /// （重复写同色不算），这里算的是"位移后落在实心区内"的格数 —— 在空网格上两者相等。
  pub voxels: usize,
  /// 走整块快路径（`VolumeGrid::fill_brick`）的块数 —— "位移只动表面一层"的量化证据。
  pub whole_bricks: usize,
  /// 壳层里逐体素判定的格数 —— 位移的真正代价（内部块仍是整块写）。
  pub shell_voxels: usize,
}

/// 位移填充要用的形状：有符号距离（`< 0` = 实心）+ 解析外法线。
#[derive(Clone, Copy)]
enum Shape {
  /// 体素闭区间 `[min, max]`（`max = min + extent - 1`，与 [`fill_box`] 的写入集合逐格一致）
  Box {
    min: IVec3,
    max: IVec3,
  },
  Sphere {
    center: IVec3,
    radius: i32,
  },
}

impl Shape {
  /// 有符号距离（单位 = 体素，采样点 = 整数格点）：
  /// 盒取 **L∞（切比雪夫）**距离、球取欧氏距离 —— 两者梯度模都 ≤ 1
  /// ⇒ 可直接与"偏移体素数"比较，也给逐块剪枝一个 Lipschitz 上界。
  /// 与基础函数的判据等价：盒的 `sdf <= 0` ⟺ 格点在 `[min, min+extent)`；球的 `sdf <= 0` ⟺ 格点距离 ≤ 半径。
  fn sdf(&self, p: Vec3) -> f32 {
    match *self {
      Shape::Box { min, max } => {
        let (lo, hi) = (min.as_vec3(), max.as_vec3());
        (lo - p).max(p - hi).max_element()
      }
      Shape::Sphere { center, radius } => (p - center.as_vec3()).length() - radius as f32,
    }
  }

  /// 单位外法线：盒 = 离得最近的那一面朝外、球 = 径向。
  /// 只对壳层（`|sdf| <= bound`）的点有几何意义 —— 深内部 / 深外部的点不会调用位移闭包。
  fn normal(&self, p: Vec3) -> Vec3 {
    match *self {
      Shape::Box { min, max } => {
        let (lo, hi) = (min.as_vec3(), max.as_vec3());
        // 逐轴的 `q_i` = 该轴到区间的有符号距离（内部为负）⇒ 取最大者的那一轴 = 最近的那一面
        let q = (lo - p).max(p - hi);
        let axis = if q.x >= q.y && q.x >= q.z {
          0
        } else if q.y >= q.z {
          1
        } else {
          2
        };
        // 朝外 = 该轴偏离盒中心的方向（`q_axis > 0` 时 p 必在区间外，符号一致）
        let center = (lo + hi) * 0.5;
        let sign = if p[axis] >= center[axis] { 1.0 } else { -1.0 };
        let mut n = Vec3::ZERO;
        n[axis] = sign;
        n
      }
      Shape::Sphere { center, .. } => (p - center.as_vec3()).normalize_or_zero(),
    }
  }

  /// 形状本身的体素 AABB（闭区间）
  fn aabb(&self) -> (IVec3, IVec3) {
    match *self {
      Shape::Box { min, max } => (min, max),
      Shape::Sphere { center, radius } => (
        center - IVec3::splat(radius),
        center + IVec3::splat(radius), // 球判据含边界（`d² <= r²`），闭区间与之一致
      ),
    }
  }
}

/// 位移填充的**唯一实现**（[`fill_box_displaced`] / [`fill_sphere_displaced`] / [`fill_bricks_displaced`] 都走这里）。
///
/// 按块粒度 `grain` 逐块分三类（`disp = None` 时 `bound = 0`，退化为"只判形状内外"）：
/// 1. **整块实心**（块内体素范围的 8 个角的有符号距离都 ≤ `-bound`）⇒ [`VolumeGrid::fill_brick`] 整块写，O(深度)；
/// 2. **整块在外**（用"块中心 sdf − 半对角线"的 Lipschitz 下界判 ≥ `bound`）⇒ 整块跳过；
/// 3. 其余 = 壳层 ⇒ 逐体素判 `sdf(p) <= disp(p, n)`（满足即写；不满足**不写**，
///    即内缩是"该格不写"而非"清空既有格子" ⇒ 位移不会误删网格里原有的几何）。
fn fill_shape(
  grid: &mut VolumeGrid,
  shape: Shape,
  grain: i32,
  palette: PaletteId,
  disp: Option<Displace<'_>>,
) -> FillStats {
  debug_assert!(LEVEL_EXTENT.contains(&grain), "grain 必须是 brick 粒度 {LEVEL_EXTENT:?} 之一");
  let bound = disp.as_ref().map_or(0.0, |d| d.bound.max(0.0));
  // 外推最多 ceil(bound) 格 ⇒ 扫描范围要相应外扩（内缩不需要：不写即可）；再对齐到 grain。
  let grow = bound.ceil() as i32;
  let (s_lo, s_hi) = shape.aabb();
  let g = IVec3::splat(grain);
  let lo = (s_lo - IVec3::splat(grow)).div_euclid(g) * g;
  let hi = (s_hi + IVec3::splat(grow + 1) + g - IVec3::ONE).div_euclid(g) * g;
  // 壳层剪枝用的 Lipschitz 半径（块中心到块内最远格点的距离；`|∇sdf| <= 1`）
  let lip = (grain - 1) as f32 * 0.5 * 3.0_f32.sqrt();

  let mut stats = FillStats::default();
  let mut bz = lo.z;
  while bz < hi.z {
    let mut by = lo.y;
    while by < hi.y {
      let mut bx = lo.x;
      while bx < hi.x {
        let b_lo = IVec3::new(bx, by, bz);
        // ① 整块实心？块内体素是整数格点 `[b_lo, b_lo+grain-1]`，而 sdf 对盒（L∞）沿各轴单调、
        // 对球是凸函数 ⇒ 取**体素范围**的 8 个角，其最大值就是块内格点的上界（球：格点都在角点凸包内）。
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
        // ② 整块在外？块中心（= 块内体素范围的几何中心）到块内最远格点不超过 `lip`
        // ⇒ 块内最小 sdf ≥ `sdf(中心) - lip`。注意中心是 `b_lo + (grain-1)/2`（不是 `b_lo/2`）。
        let center = (b_lo * 2 + IVec3::splat(grain - 1)).as_vec3() * 0.5;
        if shape.sdf(center) - lip >= bound {
          bx += grain;
          continue;
        }
        // ③ 壳层：逐体素判定（`disp = None` 时就是"格点在形状内"）
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

/// [`fill_box`] 的**位移版**（MT6-3）：按 [`DisplaceFn`] 沿表面法线外推 / 内缩体素。
///
/// `disp = None` 与 [`fill_box`] 的写入集合**逐格相同**（见 `tests::zero_displace_matches_fill_box`），
/// 只有统计口径不同（[`FillStats::voxels`] 记"落在实心区内的格数"，`fill_box` 记"真的改变了格子的次数"）。
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

/// [`fill_sphere`] 的**位移版**（MT6-3）。`disp = None` 与 [`fill_sphere`] 逐格相同。
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

/// [`fill_bricks`] 的**位移版**（MT6-3）：`e` 对齐的整块填充 + 壳层逐体素。
///
/// ⚠️ 返回值与 [`fill_bricks`] **不同**：那个返回写入的 **brick 数**，这里返回 [`FillStats`]
/// （`whole_bricks` 即那个 brick 数、`voxels` 是写入体素数）。`e` 越大壳层越厚
/// （壳层 = 2·ceil(bound) 层 e³ 块）⇒ 逐体素代价按 `e²` 上涨，位移场景建议 `e = 4`。
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

#[cfg(test)]
mod tests {
  use super::*;

  /// 把 `[lo, hi)` 区域内的体素导出成位串（用来逐格比对两次填充的写入集合）。
  fn dump(grid: &VolumeGrid, lo: IVec3, hi: IVec3) -> Vec<bool> {
    let mut out = Vec::new();
    let mut z = lo.z;
    while z < hi.z {
      let mut y = lo.y;
      while y < hi.y {
        let mut x = lo.x;
        while x < hi.x {
          out.push(
            !grid
              .get_voxel(crate::VoxelCoord::from_ivec3(IVec3::new(x, y, z)))
              .unwrap_or(PaletteId::AIR)
              .is_air(),
          );
          x += 1;
        }
        y += 1;
      }
      z += 1;
    }
    out
  }

  /// 位移关闭（恒定 0 偏移）时，位移版与基础版（`fill_box` / `fill_sphere`）的写入集合**逐格相同**。
  /// 这是"新增 API 不改变既有语义"的机器证据（含负坐标与非 4 对齐的盒子）。
  #[test]
  fn zero_displace_matches_base_fill() {
    let zero = |_p: Vec3, _n: Vec3| 0.0f32;
    for (min, extent) in [
      (IVec3::new(0, 0, 0), IVec3::new(4, 4, 4)),
      (IVec3::new(-5, 3, -9), IVec3::new(9, 6, 13)),
      (IVec3::new(64, 240, 512), IVec3::new(32, 16, 48)),
    ] {
      let mut a = VolumeGrid::new();
      let n_a = fill_box(&mut a, min, extent, 3);
      let mut b = VolumeGrid::new();
      let st = fill_box_displaced(&mut b, min, extent, 3, Some(Displace { f: &zero, bound: 0.0 }));
      assert_eq!(n_a, st.voxels, "盒 min={min} extent={extent}");
      let (lo, hi) = (min - IVec3::splat(2), min + extent + IVec3::splat(2));
      assert_eq!(dump(&a, lo, hi), dump(&b, lo, hi), "盒 min={min} extent={extent}");
    }
    for (center, r) in [(IVec3::ZERO, 5), (IVec3::new(7, -3, 2), 11)] {
      let mut a = VolumeGrid::new();
      let n_a = fill_sphere(&mut a, center, r, 3);
      let mut b = VolumeGrid::new();
      let st = fill_sphere_displaced(&mut b, center, r, 3, Some(Displace { f: &zero, bound: 0.0 }));
      assert_eq!(n_a, st.voxels, "球 center={center} r={r}");
      let (lo, hi) = (center - IVec3::splat(r + 2), center + IVec3::splat(r + 3));
      assert_eq!(dump(&a, lo, hi), dump(&b, lo, hi), "球 center={center} r={r}");
    }
  }

  /// 恒定偏移 = 沿法线整体平移表面：`-1` 内缩一格、`+1` 外推一格（体积按各轴 ±2 变化）。
  /// 同时验证"内缩 = 不写"（内缩后表面那层格子在网格里是空的）。
  #[test]
  fn constant_offset_shrinks_and_grows() {
    let extent = IVec3::new(9, 7, 5);
    let shrink = |_p: Vec3, _n: Vec3| -1.0f32;
    let grow = |_p: Vec3, _n: Vec3| 1.0f32;

    let mut a = VolumeGrid::new();
    let st =
      fill_box_displaced(&mut a, IVec3::ZERO, extent, 7, Some(Displace { f: &shrink, bound: 1.0 }));
    // 每轴两端各丢一格 ⇒ (9-2)·(7-2)·(5-2)
    assert_eq!(st.voxels, 7 * 5 * 3);
    assert_eq!(a.get_voxel(crate::VoxelCoord::from_ivec3(IVec3::ZERO)), None, "内缩后表面格为空");

    let mut b = VolumeGrid::new();
    let st =
      fill_box_displaced(&mut b, IVec3::ZERO, extent, 7, Some(Displace { f: &grow, bound: 1.0 }));
    // 每轴两端各长一格 ⇒ 11·9·7
    assert_eq!(st.voxels, 11 * 9 * 7);
    assert!(
      b.get_voxel(crate::VoxelCoord::from_ivec3(IVec3::new(-1, -1, -1))).is_some(),
      "外推后该格应被写入"
    );
  }

  /// 幅度上界 `bound ≤ grain` 时**只有表面壳层逐体素**、内部整块写 ——
  /// 这是"位移后仍能实时"的结构性证据（`docs/PLAN.md` §5 R7）。
  ///
  /// 24³ 盒 / `grain = 4` / `bound = 4`：扫描范围 = 盒外扩 `ceil(bound)` 后的对齐盒 = 32³ = 512 块，
  /// 其中"三轴都离表面 ≥ 4"的内部块 = 4³ = 64 块（整块写 ⇒ `fill_brick`），其余 448 块是壳层。
  #[test]
  fn only_surface_shell_is_per_voxel() {
    let zero = |_p: Vec3, _n: Vec3| 0.0f32;
    let mut g = VolumeGrid::new();
    let st = fill_box_displaced(
      &mut g,
      IVec3::ZERO,
      IVec3::splat(24),
      4,
      Some(Displace { f: &zero, bound: 4.0 }),
    );
    assert_eq!(st.whole_bricks, 64);
    assert_eq!(st.shell_voxels, (512 - 64) * 64);
    assert_eq!(st.voxels, 24 * 24 * 24);
  }
}
