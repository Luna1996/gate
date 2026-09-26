//! **Minecraft Java（Anvil）地图的流式加载**（`docs/mc_map.md`）。
//!
//! # 对齐：1 方块 = 16³ 体素
//!
//! 体素单位取 MC 的**材质分辨率**（16×16）⇒ 一个方块 = 16³ 体素。由此得到一个精确对齐：
//!
//! ```text
//! 我们的 chunk 边长 = 256 体素 = 256/16 = 16 方块 = MC 的**一个 chunk-section**
//! 我们的 chunk 坐标 = (chunkX, sectionY, chunkZ)      // floor(方块/16) == floor(体素/256)
//! ```
//!
//! 于是 M5/M6/M8 那套流式环（窗口跟相机、射线请求驱动装载、LRU 卸载、多 volume）**原样可用**：
//! [`source::McCity::produce`] 只要把"我们的一块 = MC 的一个 section"翻译成方块 → 体素即可，
//! 不需要任何缩放或偏移。（1 体素 = 2 cm ⇒ 一个 section = 5.12 m，与 `infinite_cubes` 的 chunk 同尺寸。）
//!
//! # 数据来源
//!
//! | 数据 | 位置 | 用途 |
//! |---|---|---|
//! | 世界存档 | `<map>/region/r.X.Z.mca`（Anvil） | 方块状态（[`world`]，经 `fastanvil` + `fastnbt`） |
//! | blockstate / model | `<assets>/blockstates/*.json`、`<assets>/models/block/*.json` | 形状（谁占哪几个 1/16 格） |
//! | 贴图 | `<assets>/textures/block/*.png`（16×16） | 逐 texel 颜色 |
//!
//! 两个目录各有一个环境变量覆盖：`GATE_MC_MAP`（存档根）与 `GATE_MC_ASSETS`（资产根），缺省是本机
//! Greenfield 的那两份（见 `docs/mc_map.md` §2/§3）。资产 = **资源包优先、客户端 jar 兜底**地解出来的
//! 一份（解包步骤也在 §3）。
//!
//! # 分层
//!
//! | 模块 | 职责 |
//! |---|---|
//! | [`world`] | 存档读取（region 索引 / 解压 / NBT → 区块结构，全部走现成 crate） |
//! | [`assets`] | blockstate / model JSON + 贴图 PNG 的读取与缓存 |
//! | [`model`] | blockstate → 元素盒（`parent` 继承、贴图变量、`variants` / `multipart` 条件） |
//! | [`voxel`] | 元素盒 → 16³ 体素色（逐 texel）＋ 各档粒度的写入计划 |
//! | [`material`] | 逐 texel 颜色 → 调色板槽（worker 认领，主线程装表） |
//! | [`source`] | `ChunkSource`：我们的"一块 = 一个 section"翻译 + 挂进流式环 |

pub mod assets;
pub mod material;
pub mod model;
pub mod source;
pub mod summary;
pub mod voxel;
pub mod world;

use std::path::PathBuf;
use std::sync::Arc;

use glam::IVec3;
use gate_voxel::VolumeGrid;

use crate::vox_scene::VoxSceneInfo;

/// 世界名（「游戏/世界」页模型下拉里就选它，见 `crate::debug_menu`）
pub const MC_MAP: &str = "mc_map";

/// 流式窗口每轴 chunk 数（与 `infinite_cubes::attach_far_levels` 的 `WINDOW_CHUNKS` 同值 ——
/// 它是 `b_struct` 索引区的上限）。64³ chunk × 5.12 m = 相机周围 ±164 m。
const WINDOW_CHUNKS: i32 = 32;

/// 存档根目录（`GATE_MC_MAP` 覆盖；缺省 = 本机的 Greenfield 1.17.1）
pub fn map_dir() -> PathBuf {
  std::env::var_os("GATE_MC_MAP")
    .map(PathBuf::from)
    .unwrap_or_else(|| PathBuf::from(r"C:\game\Greenfield v0.5.4\Greenfield v0.5.4"))
}

/// 资产根目录（`GATE_MC_ASSETS` 覆盖；缺省 = 本机解出来的那一份）
pub fn assets_root() -> PathBuf {
  std::env::var_os("GATE_MC_ASSETS")
    .map(PathBuf::from)
    .unwrap_or_else(|| PathBuf::from(r"C:\game\Greenfield v0.5.4\assets_mc"))
}

/// 出生点（**体素**坐标）：`level.dat` 的 `SpawnX/Y/Z` × 16。读不到 → `None`（调用方回退默认机位）。
pub fn spawn_eye() -> Option<IVec3> {
  let b = world::spawn(&map_dir())?;
  Some(IVec3::new(b[0], b[1], b[2]) * 16)
}

/// 建 MC 世界：起生产源、把流式窗口钉在相机周围，并**不预铺任何 chunk** ——
/// 一个 section 是 16³ 个方块（最多 4096 次模型体素化），同步铺一圈会卡住启动；内容交给流式环
/// 按"相机半径 + 射线请求"逐帧产出（与 `infinite_cubes` 同一套节流）。
///
/// 返回 `(场景信息, 生产源)`：源要交给 [`crate::infinite_cubes::Streaming::source`]，
/// 否则流式环起来的还是 `infinite_cubes` 的生产源。
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
  grid.set_stream_window(Some((c - IVec3::splat(WINDOW_CHUNKS), IVec3::splat(WINDOW_CHUNKS * 2))));
  // **M8**：挂三级远场（`scene.rs` 按这个标记调 `attach_far_levels_mc`）—— 视距从 ±164 m
  // （`WINDOW_CHUNKS`）推到 ±655 m / ±2.6 km / ±10.5 km。摘要金字塔见 `mc::summary`。
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
  let city = Arc::new(source::McCity::new(world, assets, pool));
  Ok((
    VoxSceneInfo {
      aabb_min: c * chunk - half,
      aabb_max: c * chunk + half,
      instances_used: 1,
      voxels_written: 0, // 由流式环逐帧铺
      voxels_dropped: 0,
    },
    city,
  ))
}

#[cfg(test)]
mod tests {
  use super::*;
  use gate_voxel::{ChunkCoord, ChunkSource, Detail, PaletteId, VolumeGrid};

  /// **真地图 + 真资产端到端**（默认 `#[ignore]`）：从出生点区块的某一层真产出一棵树，
  /// 打印"实体体素数 / 树叶字节数 / 耗时 / 槽位数"。这是"模型 + 贴图 + 流式翻译"三件事真能跑通的
  /// 直接证据，也是选 [`material::Pool`] 颜色量化位宽的依据（逐 texel 原样上色时每个 `4³` 砖都
  /// 不同色 ⇒ 树 30 MB、一层 120 ms）。
  /// 跑法：`cargo test --release -p gate-app --bin gate-app -- --ignored --nocapture real_map_produce`
  #[test]
  #[ignore = "需要外部存档与资产：设 GATE_MC_MAP / GATE_MC_ASSETS，或放默认路径"]
  fn real_map_produce() {
    let world = Arc::new(world::World::new(map_dir()));
    let assets = Arc::new(assets::Assets::new(assets_root()));
    let sp = world::spawn(&map_dir()).expect("level.dat 应有出生点");
    let (cx, cz) = (sp[0].div_euclid(16), sp[2].div_euclid(16));
    let mut scratch = VolumeGrid::new();
    // 量化位宽扫描：8 = 原样（逐 texel）。同一个 section 连产两次：第一次含"建计划"的冷开销
    // （每个方块状态的 16³ 栅格化只做一次、之后全局复用），第二次才是稳态。
    for bits in [8u8, 5, 4] {
      let pool = Arc::new(material::Pool::quantized(bits));
      let city = source::McCity::new(world.clone(), assets.clone(), pool.clone());
      let mut line = format!("颜色 {bits} 位/通道：");
      let (mut words, mut n) = (0usize, 0usize);
      for (round, sy) in [(0, 4), (1, 4), (1, 3), (1, 5)] {
        let t = std::time::Instant::now();
        let tree = city.produce(0, ChunkCoord(IVec3::new(cx, sy, cz)), Detail::Full, &mut scratch);
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        if round == 1 {
          words += tree.as_ref().map_or(0, |t| t.len_words());
          n += 1;
        } else {
          line.push_str(&format!("y{sy} 冷 {ms:.1} ms；"));
        }
        if round == 1 && sy == 4 {
          line.push_str(&format!("y{sy} 热 {ms:.1} ms；"));
        }
      }
      println!(
        "{line}{n} 层 {:.0} KB/层（共 {:.1} MB）、调色板 {} 槽",
        words as f64 * 4.0 / 1024.0 / n.max(1) as f64,
        words as f64 * 4.0 / 1048576.0,
        pool.len(),
      );
    }
    // 粗档（16³ = 一个方块一格）也量一下：这决定远景能铺多快
    let pool = Arc::new(material::Pool::new());
    let city = source::McCity::new(world.clone(), assets.clone(), pool.clone());
    let t = std::time::Instant::now();
    let n = 8usize;
    let mut words = 0usize;
    for i in 0..n as i32 {
      if let Some(tree) = city.produce(0, ChunkCoord(IVec3::new(cx + i, 4, cz)), Detail::Coarse, &mut scratch) {
        words += tree.len_words();
      }
    }
    println!(
      "Coarse 档：{:.2} ms/section（{} 字 = {:.1} KB）",
      t.elapsed().as_secs_f64() * 1000.0 / n as f64,
      words / n,
      words as f64 / n as f64 * 4.0 / 1024.0
    );
    println!("缺失资产 {} 个：{:?}", assets.missing_names().len(), assets.missing_names());
    assert!(pool.len() > 1, "应认领到多个调色板槽");
    // 颜色抽样：产出的树里逐 4 格取一次，按槽号统计出现次数，打印前 8 个的**实际颜色**
    // —— 这是"贴图真被用上了"的直接证据（白色陶土 ≈ 222、橡木木板 ≈ 154,125,88、石 ≈ 127）。
    let pool = Arc::new(material::Pool::new());
    let city = source::McCity::new(world.clone(), assets.clone(), pool.clone());
    if let Some(tree) = city.produce(0, ChunkCoord(IVec3::new(cx, 4, cz)), Detail::Full, &mut scratch) {
      let mut tally: std::collections::HashMap<PaletteId, usize> = Default::default();
      for y in (0..256).step_by(4) {
        for z in (0..256).step_by(4) {
          for x in (0..256).step_by(4) {
            if let Some(id) = tree.get_voxel(x, y, z) {
              *tally.entry(id).or_default() += 1;
            }
          }
        }
      }
      let entries: std::collections::HashMap<PaletteId, _> = pool.log_from(0).into_iter().collect();
      let mut top: Vec<_> = tally.into_iter().collect();
      top.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
      println!("出生点 chunk y=4 的颜色抽样（每 4 格一次，共 262144 个采样）：");
      for (id, n) in top.iter().take(8) {
        println!("  槽 {id:>5} × {n:<6} 颜色 {:?}", entries.get(id).map(|e| e.color));
      }
    }
  }

  /// **远场级（M8）真地图端到端**（默认 `#[ignore]`）：在出生点产三级远场 chunk，打印每级的耗时 /
  /// 填格数 / 树字长。这是"摘要金字塔真能吃下 Greenfield"的直接证据，也是选采样口径的依据
  /// （见 [`summary`] 的模块头：L2/L3 的耗时随"读多少个 chunk 列"线性涨）。
  /// 跑法：`cargo test --release -p gate-app --bin gate-app -- --ignored --nocapture real_map_far_produce`
  #[test]
  #[ignore = "需要外部存档与资产：设 GATE_MC_MAP / GATE_MC_ASSETS，或放默认路径"]
  fn real_map_far_produce() {
    use crate::infinite_cubes::FAR_SCALES;

    let world = Arc::new(world::World::new(map_dir()));
    let assets = Arc::new(assets::Assets::new(assets_root()));
    let pool = Arc::new(material::Pool::new());
    let city = source::McCity::new(world, assets, pool.clone());
    let eye = spawn_eye().expect("level.dat 应有出生点");
    let mut scratch = VolumeGrid::new();
    let mut total_reads = 0usize;
    for (i, &scale) in FAR_SCALES.iter().enumerate() {
      let vol = i + 1;
      let c = (eye / scale).div_euclid(IVec3::splat(gate_voxel::CHUNK_SIZE));
      let before = city.world_reads();
      let t = std::time::Instant::now();
      let tree = city.produce(vol, ChunkCoord(c), Detail::Full, &mut scratch);
      let ms = t.elapsed().as_secs_f64() * 1000.0;
      // 填格数：每格一个 `FAR_GRAIN³` 等值砖 ⇒ 在格原点采一次即可
      let grain = crate::infinite_cubes::FAR_GRAIN;
      let mut filled = 0usize;
      if let Some(t) = tree.as_ref() {
        for ky in 0..16 {
          for kz in 0..16 {
            for kx in 0..16 {
              if t.get_voxel(kx * grain, ky * grain, kz * grain).is_some() {
                filled += 1;
              }
            }
          }
        }
      }
      let reads = city.world_reads() - before;
      total_reads += reads;
      println!(
        "L{vol} scale {scale}：chunk {c:?} ⇒ {filled}/4096 格非空、树 {} 字 = {:.1} KB、{ms:.1} ms\
         （本块新增缓存区块 {reads}）",
        tree.as_ref().map_or(0, |t| t.len_words()),
        tree.as_ref().map_or(0, |t| t.len_words()) as f64 * 4.0 / 1024.0,
      );
    }
    println!("三级合计新增读入区块 {total_reads}（缓存命中不再计数）");
    assert!(pool.len() > 1, "远场也应认领到调色板槽");
  }
}
