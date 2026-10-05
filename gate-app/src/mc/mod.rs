pub mod assets;
pub mod lod;
pub mod material;
pub mod model;
pub mod source;
pub mod summary;
pub mod voxel;
pub mod world;

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use gate_voxel::VolumeGrid;
use glam::IVec3;

use crate::infinite_cubes::SEAM_MARGIN;
use crate::vox_scene::VoxSceneInfo;

pub const MC_MAP: &str = "mc_map";

pub const VOXELS_PER_BLOCK: i32 = 16;

pub const NEAR_COVER_CHUNKS: i32 = WINDOW_CHUNKS - SEAM_MARGIN;

const WINDOW_CHUNKS: i32 = 32;

pub fn map_dir() -> PathBuf {
  std::env::var_os("GATE_MC_MAP")
    .map(PathBuf::from)
    .unwrap_or_else(|| PathBuf::from(r"C:\game\Greenfield v0.5.4\Greenfield v0.5.4"))
}

pub fn assets_root() -> PathBuf {
  std::env::var_os("GATE_MC_ASSETS")
    .map(PathBuf::from)
    .unwrap_or_else(|| PathBuf::from(r"C:\game\Greenfield v0.5.4\assets_mc"))
}

pub fn spawn_eye() -> Option<IVec3> {
  let b = world::spawn(&map_dir())?;
  Some(IVec3::new(b[0], b[1], b[2]) * VOXELS_PER_BLOCK)
}

pub fn build(
  grid: &mut VolumeGrid,
  cam_eye: IVec3,
) -> Result<(VoxSceneInfo, Arc<source::McCity>), String> {
  let dir = map_dir();
  if !world::looks_like_world(&dir) {
    return Err(format!("{} 不是 Anvil 世界（缺 region/ 或 level.dat）", dir.display()));
  }
  let root = assets_root();
  if !root.is_dir() {
    return Err(format!("{} 不是资产目录（见 docs/mc_map.md §3 的解包步骤）", root.display()));
  }
  let world = Arc::new(world::World::new(&dir));
  let assets = Arc::new(assets::Assets::new(&root));
  let pool = Arc::new(material::Pool::new());

  let chunk = gate_voxel::CHUNK_SIZE;
  let c = cam_eye.div_euclid(IVec3::splat(chunk));
  let win = WINDOW_CHUNKS;
  grid.set_stream_window(Some((c - IVec3::splat(win), IVec3::splat(win * 2))));
      let cover = NEAR_COVER_CHUNKS;
  grid.set_coverage_r(cover as f32 * chunk as f32);
  grid.set_attach_far(true);
  let half = IVec3::splat(WINDOW_CHUNKS * chunk);
  bevy::log::info!(
    "MC 地图 {}{}；资产 {}；窗口 ±{} chunk（±{:.0} m）",
    world.dir().display(),
    match spawn_eye() {
      Some(e) => format!("，出生点体素 {}", e),
      None => "（level.dat 读不到出生点）".to_string(),
    },
    assets.root().display(),
    WINDOW_CHUNKS,
    WINDOW_CHUNKS as f32 * chunk as f32 * 0.02,
  );
  let stamp = lod::stamp(world.dir());
  let city = Arc::new(source::McCity::new(world, assets, pool));
  let lod_path = lod::path_for(MC_MAP);
  match lod::load(&lod_path, stamp) {
    Ok(f) => city.install_lod(f),
    Err(why) => {
      bevy::log::warn!(
        "LOD {} → 远场暂按需读 Anvil（L1 ≈ 350–520 ms/chunk，比 LOD 慢 293–6305×）；已起后台重建",
        why.reason(&lod_path)
      );
      start_lod_build(city.clone());
    }
  }
  Ok((
    VoxSceneInfo {
      aabb_min: c * chunk - half,
      aabb_max: c * chunk + half,
      instances_used: 1,
      voxels_written: 0,
      voxels_dropped: 0,
    },
    city,
  ))
}

static LOD_RUNNING: AtomicBool = AtomicBool::new(false);
static LOD_CANCEL: AtomicBool = AtomicBool::new(false);
static LOD_PERCENT: AtomicU32 = AtomicU32::new(0);

pub fn lod_build_state() -> (bool, u32) {
  (LOD_RUNNING.load(Ordering::Relaxed), LOD_PERCENT.load(Ordering::Relaxed))
}

pub fn start_lod_build(city: Arc<source::McCity>) {
  if LOD_RUNNING.swap(true, Ordering::SeqCst) {
    bevy::log::info!("LOD 构建已在进行中 → 忽略");
    return;
  }
  LOD_CANCEL.store(false, Ordering::SeqCst);
  LOD_PERCENT.store(0, Ordering::Relaxed);
  let spawned = std::thread::Builder::new().name("mc-lod-build".into()).spawn(move || {
    let t0 = std::time::Instant::now();
    let last = AtomicU32::new(0);
    let r = city.build_lod(&lod::path_for(MC_MAP), &LOD_CANCEL, |done, total| {
      let pct = (done * 100 / total.max(1)) as u32;
      LOD_PERCENT.store(pct, Ordering::Relaxed);
      if pct >= last.load(Ordering::Relaxed) + 5 {
        last.store(pct, Ordering::Relaxed);
        bevy::log::info!(
          "LOD 构建 {pct}%（{done}/{total} 列，已用 {:.0}s）",
          t0.elapsed().as_secs_f64()
        );
      }
    });
    LOD_RUNNING.store(false, Ordering::SeqCst);
    match r {
      Ok(s) => bevy::log::info!(
        "LOD 构建完成：{} 列 {} 节、{:.1} MB、{:.1}s → 已启用（远场 scale ≥ {} 走内存采样）",
        s.cols,
        s.sections,
        s.bytes as f64 / 1048576.0,
        s.secs,
        lod::CELL,
      ),
      Err(e) => bevy::log::warn!("LOD 构建中止 / 失败：{e}"),
    }
  });
  if spawned.is_err() {
    LOD_RUNNING.store(false, Ordering::SeqCst);
    bevy::log::warn!("LOD 构建：起线程失败");
  }
}

pub fn cancel_lod_build() {
  if LOD_RUNNING.load(Ordering::SeqCst) {
    LOD_CANCEL.store(true, Ordering::SeqCst);
    bevy::log::info!("LOD 构建 → 中止请求");
  }
}