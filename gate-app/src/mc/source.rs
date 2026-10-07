use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

use gate_voxel::{ChunkCoord, ChunkSource, ChunkTree, Detail, PaletteEntry, PaletteId};
use glam::IVec3;

use super::assets::Assets;
use super::lod;
use super::material::Pool;
use super::summary::{self, Summary};
use super::voxel::{self, BlockPlan, Fill};
use super::world::{self, BlockState, World};
use crate::infinite_cubes::{FAR_GRAIN, FAR_SCALES};

const SEC: i32 = 16;

fn cell_blocks_of(scale: i32) -> i32 {
  FAR_GRAIN * scale / super::VOXELS_PER_BLOCK
}

pub struct McCity {
  world: Arc<World>,
  assets: Arc<Assets>,
  pool: Arc<Pool>,
  plans: Mutex<HashMap<String, Option<Arc<BlockPlan>>>>,
  summary: Summary,
  lod: lod::Cell,
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
      lod: Arc::new(std::sync::RwLock::new(None)),
      produced: Mutex::new(0),
    }
  }

  pub fn plan_for(&self, st: &BlockState) -> Option<Arc<BlockPlan>> {
    let key = st.key();
    if let Some(hit) = self.plans.lock().unwrap_or_else(|e| e.into_inner()).get(&key) {
      return hit.clone();
    }
    let built = voxel::plan(&self.assets, &self.pool, st);
    self.plans.lock().unwrap_or_else(|e| e.into_inner()).insert(key, built.clone());
    built
  }

  pub fn palette_from(&self, from: usize) -> Vec<(PaletteId, PaletteEntry)> {
    self.pool.log_from(from)
  }

  fn produce_far(&self, coord: ChunkCoord, scale: i32) -> Option<ChunkTree> {
    let cell_blocks = cell_blocks_of(scale);
    if let Some(view) = self.lod().filter(|v| v.supports(cell_blocks)) {
      return self.far_tree(coord, cell_blocks, |cell| view.cell(cell, cell_blocks));
    }
    let summary = &self.summary;
    self.far_tree(coord, cell_blocks, |cell| summary.cell(self, cell, cell_blocks))
  }

  fn far_tree(
    &self,
    coord: ChunkCoord,
    cell_blocks: i32,
    cell: impl Fn(IVec3) -> Option<PaletteId>,
  ) -> Option<ChunkTree> {
    let cells = gate_voxel::CHUNK_SIZE / FAR_GRAIN;
    let origin = coord.0 * cells * cell_blocks;
    let mut tree = ChunkTree::empty();
    for kz in 0..cells {
      for ky in 0..cells {
        for kx in 0..cells {
          if let Some(id) = cell(origin + IVec3::new(kx, ky, kz) * cell_blocks) {
            tree.fill_brick([kx * FAR_GRAIN, ky * FAR_GRAIN, kz * FAR_GRAIN], FAR_GRAIN, id);
          }
        }
      }
    }
    (!tree.is_empty()).then_some(tree)
  }

  pub fn lod(&self) -> Option<Arc<lod::View>> {
    self.lod.read().unwrap_or_else(|e| e.into_inner()).clone()
  }

  pub fn install_lod(&self, f: lod::File) {
    let view = lod::View::new(f, |key| self.plan_for(&BlockState::from_key(key)).map(|p| p.rep));
    bevy::log::info!(
      "LOD 装载：{} 列 {} 节、{} KB、名字 {} 个（{} 个没认领到槽 ⇒ 那些节算空）；远场 scale ≥ {} 改走它",
      view.cols(),
      view.sections(),
      view.heap_bytes() / 1024,
      view.names(),
      view.missing(),
      lod::CELL,
    );
    *self.lod.write().unwrap_or_else(|e| e.into_inner()) = Some(Arc::new(view));
  }

  pub fn build_lod(
    &self,
    path: &Path,
    cancel: &AtomicBool,
    progress: impl Fn(usize, usize) + Sync,
  ) -> Result<lod::BuildStats, String> {
    let (stats, f) = lod::build(&self.world, path, cancel, progress)?;
    self.install_lod(f);
    Ok(stats)
  }
}

impl summary::SectionReps for McCity {
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
  fn content_y_range(&self) -> Option<(i32, i32)> {
    Some((0, world::SECTIONS_PER_CHUNK * SEC * super::VOXELS_PER_BLOCK))
  }

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
      return None;
    }
    let chunk = self.world.chunk(cx, cz)?;
    let sec = chunk.section(sy)?;
    if sec.is_empty_layer() {
      return None;
    }
    let plans: Vec<Option<Arc<BlockPlan>>> = sec.palette.iter().map(|s| self.plan_for(s)).collect();
    if plans.iter().all(Option::is_none) {
      return None;
    }
    let mut states = Vec::new();
    sec.unpack_into(&mut states);

    let grain = detail.grain();
    let mut tree = ChunkTree::empty();
    match grain {
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

  fn clone_as_any(self: Arc<Self>) -> Option<Arc<dyn std::any::Any + Send + Sync>> {
    let any: Arc<dyn std::any::Any + Send + Sync> = self;
    Some(any)
  }
}

fn block_origin(i: usize) -> IVec3 {
  IVec3::new((i % 16) as i32, (i / 256) as i32, ((i / 16) % 16) as i32) * SEC
}

fn apply(tree: &mut ChunkTree, org: IVec3, f: &Fill) {
  let Fill::Brick { off, extent, id } = f;
  tree.fill_brick((org + IVec3::from(*off)).to_array(), *extent, *id);
}

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
  let n = 16 / 4;
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

fn group_rep(buf: &[PaletteId]) -> Option<PaletteId> {
  let solid = buf.iter().filter(|c| !c.is_air()).count();
  if solid * 2 < buf.len() {
    return None;
  }
  voxel::rep_of(buf)
}
