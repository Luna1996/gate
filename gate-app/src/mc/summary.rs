//! **远场摘要金字塔**（`docs/mc_map.md` §8.2）：给远场级（M8）用的"粗粒度块色"。
//!
//! # 为什么需要它
//!
//! 远场级的一个 chunk 覆盖 `16·scale` 个方块每轴。要按"每个 `scale³` 方块一格"填满它，最直接的做法是
//! 逐块采样 —— 但 L3（`scale = 64`）的**一个** chunk 就有 `64³` 个 MC chunk 列、`4096 × 64` 个
//! section，逐块读完整张图要几千万次 region 解压，不可行（`docs/mc_map.md` §8.2 的量化）。
//!
//! 这里的做法是**按级降采样**：每一级只读它需要的**一个** section 列，把该 section 的 4096 个方块
//! 折成一个代表色，再用代表色填格。三级的采样口径（`FAR_GRAIN = 16` 级体素 = 一格的边长）：
//!
//! | 级 | `scale` | 一格（方块） | 采样 |
//! |---|---|---|---|
//! | L1 | 4 | 4 | 读所在 section，取格内 `4³` 方块的多数色 |
//! | L2 | 16 | 16 | 格正好是一个 section ⇒ 取该 section 的多数色 |
//! | L3 | 64 | 64 | 只读格**中心**那一个 chunk 列（1/16 的采样面），取该列在本格高度内的 section 多数色 |
//!
//! **为什么 L3 敢只采中心**：远场的格边长是按"这一级在屏幕上 ≈ 0.46 px"选的（`infinite_cubes::FAR_GRAIN`）
//! ⇒ 格内的一切都是亚像素的，"格中心有没有东西"就是"这一格显不显示"的全部信息。代价是 L3 会把
//! 偏心的街区采样成空 —— 这与"远场只求轮廓"的取舍一致。
//!
//! # 缓存
//!
//! 摘要按 **section** 缓存（[`Sec`]，约 200 B/节）：一个 section 的 4096 个方块要读一次 region，
//! 而它同时供给 L1（`fine` 的 64 格）与 L2（`rep`）⇒ 缓存它把"每块一次解压"摊成"每节一次"。

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use gate_voxel::PaletteId;
use glam::IVec3;

use super::world::SECTION_VOLUME;
use super::voxel;

/// 一个 section 的方块边长（= 一个方块 = 16 体素）
pub const SEC: i32 = 16;
/// 每节里 `4³` 方的格数（每轴 4 个）—— L1 格的粒度
const FINE_PER_AXIS: i32 = 4;
const FINE: usize = (FINE_PER_AXIS * FINE_PER_AXIS * FINE_PER_AXIS) as usize;

/// "这一格算实体"的最小实体占比（千分比）。远场要的是**轮廓**：格内实体太少就不写，免得把街道、
/// 空地也糊成实心；格内实体够多就整格填成多数色。
const SOLID_MIN_PERMILLE: u32 = 125;

/// 摘要缓存的节数上限（一节约 200 B ⇒ 32768 节约 6.4 MB）。远场按 chunk 列采样，相邻 chunk 会复用
/// 同一列里的多个 section，这个量级够覆盖几十个远场 chunk 的重叠面。
const SEC_CACHE_MAX: usize = 32_768;

/// 一节的摘要
pub struct Sec {
  /// 每 `4³` 方块一个代表色（下标 `x + 4·z + 16·y`）；实体不足的格是 [`PaletteId::AIR`]
  pub fine: Box<[PaletteId; FINE]>,
  /// 整节的多数色（全空 → [`PaletteId::AIR`]）
  pub rep: PaletteId,
  /// 非空气方块数（0..=4096）
  pub solid: u16,
}

/// 摘要的数据来源：一个 section 的 4096 个方块代表色（下标 `y·256 + z·16 + x`，空气 = 0）。
///
/// 实现方必须线程安全（远场产出在 worker 线程上跑，见 `gate_voxel::ChunkSource`），且应缓存调色板认领
/// 的结果 —— 返回的就是"方块状态 → 代表色"的翻译，与 [`super::source`] 的产出一致。
pub trait SectionReps: Send + Sync {
  /// `None` = 该 section 不存在 / 全空 / 越界（都当空气）
  fn section_reps(&self, sec: IVec3) -> Option<Box<[PaletteId; SECTION_VOLUME]>>;
}

/// 摘要金字塔（每源一份；内部缓存按 section 键）
pub struct Summary {
  secs: Mutex<Cache>,
}

#[derive(Default)]
struct Cache {
  map: HashMap<IVec3, Option<Arc<Sec>>>,
  order: VecDeque<IVec3>,
}

impl Default for Summary {
  fn default() -> Self {
    Self::new()
  }
}

impl Summary {
  pub fn new() -> Self {
    Self { secs: Mutex::new(Cache::default()) }
  }

  /// 取一节的摘要（`None` = 那一节是空的）
  fn sec(&self, src: &dyn SectionReps, sec: IVec3) -> Option<Arc<Sec>> {
    {
      let c = self.secs.lock().unwrap_or_else(|e| e.into_inner());
      if let Some(hit) = c.map.get(&sec) {
        return hit.clone();
      }
    }
    let built = src.section_reps(sec).map(|reps| Arc::new(build(reps.as_ref())));
    let mut c = self.secs.lock().unwrap_or_else(|e| e.into_inner());
    c.map.insert(sec, built.clone());
    c.order.push_back(sec);
    while c.order.len() > SEC_CACHE_MAX {
      if let Some(old) = c.order.pop_front() {
        c.map.remove(&old);
      }
    }
    built
  }

  /// `scale` 级、格原点 `cell`（**方块**坐标，须是 `scale` 的倍数）处的代表色；`None` = 这一格算空。
  ///
  /// 三级的采样口径见模块头。格原点恒对齐 section（`scale` 是 4 的倍数），所以 L1/L2 只需一次除法
  /// 定位，L3 用格中心所在的那一个 chunk 列。
  pub fn cell(&self, src: &dyn SectionReps, cell: IVec3, scale: i32) -> Option<PaletteId> {
    match scale {
      4 => {
        let sec = self.sec(src, cell.div_euclid(IVec3::splat(SEC)))?;
        let l = cell.rem_euclid(IVec3::splat(SEC)) / 4;
        let id = sec.fine[(l.x + FINE_PER_AXIS * l.z + FINE_PER_AXIS * FINE_PER_AXIS * l.y) as usize];
        (!id.is_air()).then_some(id)
      }
      16 => {
        let sec = self.sec(src, cell.div_euclid(IVec3::splat(SEC)))?;
        (sec.solid as u32 * 1000 >= SOLID_MIN_PERMILLE * SECTION_VOLUME as u32).then_some(sec.rep)
      }
      _ => {
        // L3：只读格**中心**那一个 chunk 列 —— xz 取格中心所在的 chunk，y 仍是本格自己的高度范围
        // （`scale/16` 个 section）。xz 与 y 的取法不同是必须的：本格高 `scale` 个方块，若 y 也按
        // "中心"取，读到的就是本格上面半格的那一节。
        let c = cell.div_euclid(IVec3::splat(SEC));
        let col_x = (cell.x + scale / 2).div_euclid(SEC);
        let col_z = (cell.z + scale / 2).div_euclid(SEC);
        let n = scale / SEC;
        let mut reps = Vec::with_capacity(n as usize);
        for sy in 0..n {
          if let Some(sec) = self.sec(src, IVec3::new(col_x, c.y + sy, col_z)) {
            if !sec.rep.is_air() {
              reps.push(sec.rep);
            }
          }
        }
        voxel::rep_of(&reps)
      }
    }
  }
}

/// 一份 section 的 4096 个代表色 → 摘要（`fine` 的 64 格 + 整节多数色 + 实体数）
fn build(reps: &[PaletteId]) -> Sec {
  let mut fine = Box::new([PaletteId::AIR; FINE]);
  let mut solid = 0u16;
  let mut groups = [PaletteId::AIR; 64];
  for gy in 0..FINE_PER_AXIS {
    for gz in 0..FINE_PER_AXIS {
      for gx in 0..FINE_PER_AXIS {
        let mut buf = [PaletteId::AIR; 64];
        let mut n = 0usize;
        for y in 0..4 {
          for z in 0..4 {
            for x in 0..4 {
              let (bx, by, bz) = (gx * 4 + x, gy * 4 + y, gz * 4 + z);
              let id = reps[(by * 256 + bz * 16 + bx) as usize];
              buf[n] = id;
              n += 1;
              if !id.is_air() {
                solid = solid.saturating_add(1);
              }
            }
          }
        }
        // 实体不足的格当空 —— 与 L2/L3 的判据同源（免得 L1 把街道也画出来）
        let cell_id = group(&buf, 64);
        groups[(gx + FINE_PER_AXIS * gz + FINE_PER_AXIS * FINE_PER_AXIS * gy) as usize] = cell_id;
        fine[(gx + FINE_PER_AXIS * gz + FINE_PER_AXIS * FINE_PER_AXIS * gy) as usize] = cell_id;
      }
    }
  }
  Sec { fine, rep: voxel::rep_of(&groups).unwrap_or(PaletteId::AIR), solid }
}

/// 一组方块的代表色：实体占比不足 `SOLID_MIN_PERMILLE` → 空
fn group(buf: &[PaletteId], n: usize) -> PaletteId {
  let solid = buf[..n].iter().filter(|c| !c.is_air()).count();
  if (solid as u32) * 1000 < SOLID_MIN_PERMILLE * n as u32 {
    return PaletteId::AIR;
  }
  voxel::rep_of(buf).unwrap_or(PaletteId::AIR)
}

#[cfg(test)]
mod tests {
  use super::*;

  /// 合成世界：`f(block) -> PaletteId`。用来在无外部存档的情况下验证三级采样口径。
  struct Fake {
    f: Box<dyn Fn(IVec3) -> PaletteId + Send + Sync>,
  }

  impl SectionReps for Fake {
    fn section_reps(&self, sec: IVec3) -> Option<Box<[PaletteId; SECTION_VOLUME]>> {
      let mut out = Box::new([PaletteId::AIR; SECTION_VOLUME]);
      let mut any = false;
      for y in 0..16 {
        for z in 0..16 {
          for x in 0..16 {
            let b = sec * SEC + IVec3::new(x, y, z);
            let id = (self.f)(b);
            out[(y * 256 + z * 16 + x) as usize] = id;
            any |= !id.is_air();
          }
        }
      }
      any.then_some(out)
    }
  }

  /// **L1（`scale = 4`）**：格内 `4³` 方块多数色；格内实体太少 → 空。
  #[test]
  fn l1_cell_is_the_majority_of_its_own_section() {
    let stone = PaletteId(7);
    let s = Fake {
      f: Box::new(move |b: IVec3| {
        // 第一格（`x,z,y ∈ [0,4)`）铺满 64 块 ⇒ 实体；第二格只铺 4 块（< 1/8）⇒ 应算空
        let in_first = |v: i32| (0..4).contains(&v);
        if in_first(b.x) && in_first(b.y) && in_first(b.z) {
          return stone;
        }
        ((4..8).contains(&b.x) && b.y == 0 && b.z == 0).then_some(stone).unwrap_or(PaletteId::AIR)
      }),
    };
    let sum = Summary::new();
    assert_eq!(sum.cell(&s, IVec3::new(0, 0, 0), 4), Some(stone), "第一格整格 stone");
    assert_eq!(sum.cell(&s, IVec3::new(4, 0, 0), 4), None, "实体不足的格算空");
    assert_eq!(sum.cell(&s, IVec3::new(0, 4, 0), 4), None, "没东西的格算空");
    // 缓存：第二次取同一节应命中（`sec` 的 map 里有这条）
    assert_eq!(sum.cell(&s, IVec3::new(0, 0, 0), 4), Some(stone));
  }

  /// **L2（`scale = 16`）**：格正好是一个 section ⇒ 整节多数色；整节空 → 空。
  #[test]
  fn l2_cell_is_the_section_majority() {
    let stone = PaletteId(3);
    let wood = PaletteId(9);
    let s = Fake {
      f: Box::new(move |b: IVec3| {
        if b.y < 0 || b.y >= 8 {
          return PaletteId::AIR;
        }
        if b.x < 8 { stone } else { wood }
      }),
    };
    let sum = Summary::new();
    // 一半 stone 一半 wood，各自 2048/4096 = 50% ⇒ 多数色按 (n, Reverse(id)) 取 id 小的 stone
    assert_eq!(sum.cell(&s, IVec3::ZERO, 16), Some(stone));
    assert_eq!(sum.cell(&s, IVec3::new(0, 16, 0), 16), None, "第二节全空");
  }

  /// **L3（`scale = 64`）**：只读格中心那一个 chunk 列；中心列有东西才算这一格。
  #[test]
  fn l3_cell_samples_the_center_column_only() {
    let stone = PaletteId(5);
    // 只在 (x,z) ∈ [32,48) 的立方体里有东西 —— 正好落在 `cell = 0` 的中心 chunk 列
    let s = Fake {
      f: Box::new(move |b: IVec3| {
        if (0..64).contains(&b.y) && (32..48).contains(&b.x) && (32..48).contains(&b.z) {
          stone
        } else {
          PaletteId::AIR
        }
      }),
    };
    let sum = Summary::new();
    assert_eq!(sum.cell(&s, IVec3::ZERO, 64), Some(stone), "中心列有东西");
    // 挪一格：中心列落到 (96,96)，那里是空的 ⇒ 该格算空（这是"只采中心"的已知代价）
    assert_eq!(sum.cell(&s, IVec3::new(64, 0, 0), 64), None);
  }

  /// 三级里的**同一件事**：空 section 一律算空、不写格（免得远场把虚空也糊上）
  #[test]
  fn empty_sections_never_produce_a_cell() {
    let s = Fake { f: Box::new(|_| PaletteId::AIR) };
    let sum = Summary::new();
    for scale in [4, 16, 64] {
      assert_eq!(sum.cell(&s, IVec3::ZERO, scale), None, "scale {scale}");
    }
  }
}
