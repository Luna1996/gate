//! **`ChunkSource` 实现**：把"我们的一块 = MC 的一个 chunk-section"翻成体素。
//!
//! 坐标（`docs/mc_map.md` §1）：我们的 chunk 坐标 = MC 的 `(chunkX, sectionY, chunkZ)`，
//! 1 方块 = 16³ 体素 ⇒ 一个 section（16³ 方块）正好是 256³ 体素 = 我们的一块，**无缩放无偏移**。
//!
//! 产出走 [`ChunkTree`] 的 brick 级写接口（`fill_brick` / `set_brick_cells`）：一个方块 = 64 个
//! `4³` 砖，每砖一次写，**与本砖内有几种颜色无关**（逐格写要付 64 倍的下钻）。这决定了本节制的成本：
//! 单色整块（石/陶土/混凝土……地图上大多数）只写 1 次；多色块（草、原木端面、台阶的显式 uv）最多 64 次。
//!
//! 档位（`Detail` 的粒度直接落在这套对齐上）：
//!
//! | `Detail` | 粒度 | 每方块写几次 |
//! |---|---|---|
//! | `Full` | 逐 texel | 1..64 |
//! | `Fine` | `4³` 体素 | 1..64（每砖等值） |
//! | `Coarse` | 整块（16） | 1 |
//! | `Wide` / `Chunk` | `4³` / 整 section 个方块 | 1 / 每 64 个方块 |
//!
//! **远场级（`vol ≥ 1`）**走 [`super::summary`] 的摘要金字塔：远场 chunk 的坐标是它自己的级体素空间
//! （级体素 = `FAR_SCALES[vol-1]` 个世界体素），一个 chunk = `16·scale` 个方块每轴 ⇒ 每格 `scale³` 方块
//! 取一个摘要色、按 `FAR_GRAIN` 级体素填格（见该模块的采样口径）。

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use gate_voxel::{ChunkCoord, ChunkSource, ChunkTree, Detail, PaletteEntry, PaletteId};
use glam::IVec3;

use super::assets::Assets;
use super::material::Pool;
use super::summary::{self, Summary};
use super::voxel::{self, BlockPlan, Fill};
use super::world::{self, BlockState, World};
use crate::infinite_cubes::{FAR_GRAIN, FAR_SCALES};

/// 一个 MC chunk-section 的方块边长（= 我们一块的体素边长 / 16）
const SEC: i32 = 16;

/// 生产源（worker 线程共享 `&self`）
pub struct McCity {
  world: Arc<World>,
  assets: Arc<Assets>,
  pool: Arc<Pool>,
  /// 方块状态 → 计划（按规范键去重；`None` 也缓存：缺资产的方块不必每次重试）
  plans: Mutex<HashMap<String, Option<Arc<BlockPlan>>>>,
  /// 远场级的摘要金字塔（按 section 缓存，见 [`super::summary`]）
  summary: Summary,
  /// 已产出的 section 数（诊断）
  produced: Mutex<usize>,
}

impl McCity {
  pub fn new(world: Arc<World>, assets: Arc<Assets>, pool: Arc<Pool>) -> Self {
    Self {
      world,
      assets,
      pool,
      plans: Mutex::new(HashMap::new()),
      summary: Summary::new(),
      produced: Mutex::new(0),
    }
  }

  /// 这个方块状态的写入计划（`None` = 画不出东西：空气 / 缺资产）
  pub fn plan_for(&self, st: &BlockState) -> Option<Arc<BlockPlan>> {
    let key = st.key();
    if let Some(hit) = self.plans.lock().unwrap_or_else(|e| e.into_inner()).get(&key) {
      return hit.clone();
    }
    let built = voxel::plan(&self.assets, &self.pool, st);
    self.plans.lock().unwrap_or_else(|e| e.into_inner()).insert(key, built.clone());
    built
  }

  /// 调色板日志的 `[from..]` 段（主线程按游标补装进各 volume 的调色板）
  pub fn palette_from(&self, from: usize) -> Vec<(PaletteId, PaletteEntry)> {
    self.pool.log_from(from)
  }

  /// 诊断：累计去过 region 的次数（见 [`World::reads`]）—— 远场采样成本的核对点
  #[cfg(test)]
  pub fn world_reads(&self) -> usize {
    self.world.reads()
  }

  /// **远场级产出**（`docs/mc_map.md` §8.2）：一个远场 chunk = `16·scale` 个方块每轴，逐格取摘要色。
  ///
  /// 格的**级体素**原点 = `k · FAR_GRAIN`、块的方块原点 = `origin + k · scale` —— 与
  /// `infinite_cubes::build_region_far` 的"格中心采样"是同一套坐标（`scale` 级体素 = 一个格）。
  fn produce_far(&self, coord: ChunkCoord, scale: i32) -> Option<ChunkTree> {
    let blocks = 16 * scale; // 远场 chunk = 256 级体素 = 16·scale 方块（每轴）
    let origin = coord.0 * blocks;
    let mut tree = ChunkTree::empty();
    for kz in 0..16 {
      for ky in 0..16 {
        for kx in 0..16 {
          let cell = origin + IVec3::new(kx, ky, kz) * scale;
          if let Some(id) = self.summary.cell(self, cell, scale) {
            tree.fill_brick([kx * FAR_GRAIN, ky * FAR_GRAIN, kz * FAR_GRAIN], FAR_GRAIN, id);
          }
        }
      }
    }
    (!tree.is_empty()).then_some(tree)
  }
}

impl summary::SectionReps for McCity {
  /// 一个 section 的 4096 个方块代表色（走 [`Self::plan_for`] 的缓存；空气 = 0）。
  fn section_reps(&self, sec: IVec3) -> Option<Box<[PaletteId; world::SECTION_VOLUME]>> {
    if !(0..world::SECTIONS_PER_CHUNK).contains(&sec.y) {
      return None;
    }
    let chunk = self.world.chunk(sec.x, sec.z)?;
    let layer = chunk.section(sec.y)?;
    if layer.is_empty_layer() {
      return None;
    }
    let plans: Vec<Option<Arc<BlockPlan>>> =
      layer.palette.iter().map(|s| self.plan_for(s)).collect();
    if plans.iter().all(Option::is_none) {
      return None;
    }
    let mut states = Vec::new();
    layer.unpack_into(&mut states);
    let mut out = Box::new([PaletteId::AIR; world::SECTION_VOLUME]);
    for (i, &pi) in states.iter().enumerate() {
      out[i] = plans.get(pi as usize).and_then(|p| p.as_ref()).map_or(PaletteId::AIR, |p| p.rep);
    }
    Some(out)
  }
}

impl ChunkSource for McCity {
  fn produce(
    &self,
    vol: usize,
    coord: ChunkCoord,
    detail: Detail,
    _scratch: &mut gate_voxel::VolumeGrid,
  ) -> Option<ChunkTree> {
    if vol > 0 {
      return FAR_SCALES.get(vol - 1).and_then(|&scale| self.produce_far(coord, scale));
    }
    let (cx, sy, cz) = (coord.0.x, coord.0.y, coord.0.z);
    if !(0..world::SECTIONS_PER_CHUNK).contains(&sy) {
      return None; // 1.17 世界高 256 ⇒ 只有 16 层
    }
    let chunk = self.world.chunk(cx, cz)?;
    let sec = chunk.section(sy)?;
    if sec.is_empty_layer() {
      return None; // 城市里大多数层是空气：直接不产出（不占池、不占 GPU）
    }
    // 本节每个调色板项的计划：属性 → 形状的解析按状态缓存，这里只是一次查表
    let plans: Vec<Option<Arc<BlockPlan>>> =
      sec.palette.iter().map(|s| self.plan_for(s)).collect();
    if plans.iter().all(Option::is_none) {
      return None;
    }
    let mut states = Vec::new();
    sec.unpack_into(&mut states);

    let grain = detail.grain();
    let mut tree = ChunkTree::empty();
    match grain {
      // 逐 texel 那两档都走**每 `4³` 砖一个代表色**（8 cm）。
      //
      // WHY: 逐 texel 实测不可用 —— 一层满实体 section 的 CPU 树 **20 MB**、产出 **78 ms**
      // （`real_map_produce`，出生点 chunk 的 y 0..6）。原因是结构性的、不是颜色量化能救的：
      // 一个方块的表面层有 56 个 `4³` 砖，每个砖横跨 4×4 个 texel ⇒ 砖内必然多色 ⇒ 每个砖都要一张
      // 4³ 值表（CPU 128 B / wire 24 B），4096 个方块就是 2600 万个砖。8 cm 档让每砖**整砖同色**
      // （值表消失、可并进父层），观感上仍保留贴图的色块变化。见 `docs/mc_map.md` §6。
      1 | 4 => {
        for (i, &pi) in states.iter().enumerate() {
          let Some(plan) = plans.get(pi as usize).and_then(|p| p.clone()) else { continue };
          let org = block_origin(i);
          for f in &plan.fills {
            apply(&mut tree, org, f);
          }
        }
      }
      16 => {
        for (i, &pi) in states.iter().enumerate() {
          let Some(plan) = plans.get(pi as usize).and_then(|p| p.clone()) else { continue };
          if plan.rep.is_air() {
            continue;
          }
          tree.fill_brick(block_origin(i).to_array(), SEC, plan.rep);
        }
      }
      _ => coarse(&mut tree, &states, &plans, grain),
    }
    let n = {
      let mut g = self.produced.lock().unwrap_or_else(|e| e.into_inner());
      *g += 1;
      *g
    };
    // 首个 section 是一条一次性里程碑：把"资产缺了多少 / 调色板认领了多少槽 / 区块缓存读数"一起落盘
    // —— 这些是"画面对不对"之外唯一可核的证据。`GATE_LOG=gate_app=debug` 还能看到每个 section。
    if n == 1 {
      bevy::log::info!(
        "MC 首个 section 产出（chunk {cx},{sy},{cz}）：调色板 {} 槽（溢出回退 {} 次）、区块缓存 {:?}、缺失资产 {} 个",
        self.pool.len(),
        self.pool.overflow_count(),
        self.world.stats(),
        self.assets.missing_names().len(),
      );
    } else if n % 256 == 0 {
      bevy::log::debug!("MC section 累计产出 {n}（chunk {cx},{sy},{cz}）");
    }
    (!tree.is_empty()).then_some(tree)
  }

  fn palette_log(&self, from: usize) -> Vec<(PaletteId, PaletteEntry)> {
    self.palette_from(from)
  }
}

/// 第 `i` 个方块（`i = y*256 + z*16 + x`）在 chunk 内的体素原点
fn block_origin(i: usize) -> IVec3 {
  IVec3::new((i % 16) as i32, (i / 256) as i32, ((i / 16) % 16) as i32) * SEC
}

/// 一次 brick 级写
fn apply(tree: &mut ChunkTree, org: IVec3, f: &Fill) {
  let Fill::Brick { off, extent, id } = f;
  tree.fill_brick((org + IVec3::from(*off)).to_array(), *extent, *id);
}

/// 更粗的两档：先在"每方块一个代表色"上聚合，再整段写（`Wide` = 4³ 个方块、`Chunk` = 整层）。
///
/// 粗档的判据是**过半非空气**才给色：4³ 个方块里只塞了一个实体块就把整格填满，远处会糊成实心块
/// （这跟近处 `Coarse` 那档"宁可膨胀也别让结构消失"的取舍相反 —— 那里一格就是一个方块，
/// 膨胀与消失是同一件事的两面；这里一格是 64 个方块，膨胀的代价大得多）。
fn coarse(tree: &mut ChunkTree, states: &[u16], plans: &[Option<Arc<BlockPlan>>], grain: i32) {
  let mut reps = [PaletteId::AIR; 4096];
  for (i, &pi) in states.iter().enumerate() {
    reps[i] = plans.get(pi as usize).and_then(|p| p.as_ref()).map_or(PaletteId::AIR, |p| p.rep);
  }
  if grain >= 256 {
    if let Some(id) = group_rep(&reps) {
      tree.fill_brick([0, 0, 0], 256, id);
    }
    return;
  }
  let n = 16 / 4; // 每轴 4 组
  let mut buf = [PaletteId::AIR; 64];
  for by in 0..n {
    for bz in 0..n {
      for bx in 0..n {
        for z in 0..4 {
          for y in 0..4 {
            for x in 0..4 {
              let src = ((by * 4 + y) * 256 + (bz * 4 + z) * 16 + (bx * 4 + x)) as usize;
              buf[(z * 16 + y * 4 + x) as usize] = reps[src];
            }
          }
        }
        if let Some(id) = group_rep(&buf) {
          tree.fill_brick([bx * 64, by * 64, bz * 64], 64, id);
        }
      }
    }
  }
}

/// 一组方块的代表色（**过半非空气** + 组内出现最多的那个色；否则整组算空）
fn group_rep(buf: &[PaletteId]) -> Option<PaletteId> {
  let solid = buf.iter().filter(|c| !c.is_air()).count();
  if solid * 2 < buf.len() {
    return None;
  }
  voxel::rep_of(buf)
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn block_origin_matches_the_nbt_index_order() {
    // i = y*256 + z*16 + x（x 最密，与 `BlockStates` 同序）
    assert_eq!(block_origin(0), IVec3::ZERO);
    assert_eq!(block_origin(1), IVec3::new(16, 0, 0), "x 走一格");
    assert_eq!(block_origin(16), IVec3::new(0, 0, 16), "z 走一格");
    assert_eq!(block_origin(256), IVec3::new(0, 16, 0), "y 走一格");
    assert_eq!(block_origin(4095), IVec3::new(15, 15, 15) * 16, "最后一格");
  }

  /// `coarse` 的聚合：4³ 个方块里取出现最多的代表色；**过半是空气的组不写**（免得远处糊成实心）
  #[test]
  fn coarse_aggregates_blocks_by_majority() {
    let stone = PaletteId(3);
    let plans: Vec<Option<Arc<BlockPlan>>> = vec![
      None, // 调色板 0 = 空气
      Some(Arc::new(BlockPlan { rep: stone, fills: Vec::new() })),
    ];
    // 第 0 组（x,y,z 各 0..4）里 60 个 stone + 4 个空气；其余方块全空气
    let mut states = vec![0u16; 4096];
    let mut n = 0;
    for y in 0..4 {
      for z in 0..4 {
        for x in 0..4 {
          if n < 60 {
            states[(y * 256 + z * 16 + x) as usize] = 1;
          }
          n += 1;
        }
      }
    }
    let mut tree = ChunkTree::empty();
    coarse(&mut tree, &states, &plans, 64);
    assert_eq!(tree.get_voxel(8, 8, 8), Some(stone), "第 0 组应整段填成 stone");
    assert!(tree.get_voxel(80, 8, 8).is_none(), "空组不写");
    // 整层档：该层 4096 个方块里只有 60 个实体（远不过半）⇒ 整层算空，什么都不写
    let mut t2 = ChunkTree::empty();
    coarse(&mut t2, &states, &plans, 256);
    assert!(t2.is_empty(), "绝大多数是空气的层不该被填实");
    // 反过来：过半实体时整层给代表色
    for s in states.iter_mut() {
      *s = 1;
    }
    let mut t3 = ChunkTree::empty();
    coarse(&mut t3, &states, &plans, 256);
    assert_eq!(t3.get_voxel(200, 200, 200), Some(stone));
  }
}
