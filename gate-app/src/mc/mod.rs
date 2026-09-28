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
//! | [`lod`] | **离线粗粒度世界**：整张图的 section 级摘要存盘 → 远场 `scale ≥ 16` 直接内存采样 |

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

use glam::IVec3;
use gate_voxel::VolumeGrid;

use crate::vox_scene::VoxSceneInfo;
use crate::infinite_cubes::SEAM_MARGIN;

/// 世界名（「游戏/世界」页模型下拉里就选它，见 `crate::debug_menu`）
pub const MC_MAP: &str = "mc_map";

/// 一个 MC 方块跨多少**世界体素**：1 方块 = 32 cm、1 体素 = 2 cm ⇒ 16。
///
/// 这是"方块 ↔ 体素"唯一的换算口径（`spawn_eye` 的出生点、远场格的方块边长、`far_tree` 的块原点
/// 都按它算）—— 别再拿 `FAR_GRAIN` 当它用：`FAR_GRAIN` 是**格**的级体素边长，两者只是在
/// `FAR_GRAIN = 16` 时数值巧合相同（见 `mc_map.md` §8.9）。
pub const VOXELS_PER_BLOCK: i32 = 16;

/// **主世界的覆盖半径**（chunk）：分壳裁剪的接力半径。L1 从它起接手。
///
/// = **近场预载盘半径（菜单"粗档半径" = `WINDOW_CHUNKS` = 32）− `SEAM_MARGIN` chunk = 30** —— 预载
/// **多铺**两圈、裁剪**少切**两圈，多出来的那两圈是**交接余量**（2 chunk = 10.24 m ≥ √2 chunk）。
///
/// WHY 必须有余量：内容锚在 **chunk 角**（`infinite_cubes::stream_chunks`：`floor(cam / chunk)`），
/// 而裁剪半径按**相机**量 ⇒ 预载盘沿某个方向最多比"相机量到的半径"短 `√2` chunk（≈1.42）。
/// 若裁剪半径 = 预载半径（旧口径），交接处就会出现一圈**谁都没有内容**的环（宽 ≤ √2 chunk）：
/// 射线穿过去 ⇒ **楼被切掉一条**（对着相机的那面整片消失，透出后面那一级），而且相机在 chunk 内
/// 挪十几米就换个方向出现 ⇒ "稍微移动就正常了"。留 2 chunk 余量 ⇒ 裁剪面恒落在内容**之内**。
///
/// CONSTRAINT: **不是**"与粗档半径同值" —— 两者相差 `SEAM_MARGIN` 才是对的（`mc::build` 会核对并
/// 警告）。与 `FAR_PRELOAD_INNER` 同源：后者 = 本值 / 4 − 1（见那边的推导）。
///
/// REF: 这个值一度只能取 8 —— 取 32 时预载盘体积 ×16 ⇒ ①规划器每帧枚举 8 万格（`HashSet` 去重
/// 8 ms/帧）②需求超过池上限 ⇒ 卡 + 抖；再把池调大又会撞 `grow_region` 的一次顶到位预分配（单次
/// 44.5 GB 崩）。两处修掉后才敢用 32：去重改位图（同一次枚举 ~1 ms）、`grow_region` 改有界倍增。
/// 实测 18094 块常驻 / 835 MB / CPU 3.9 ms/帧（见 `mc_map.md` §8.16.1）。
pub const NEAR_COVER_CHUNKS: i32 = WINDOW_CHUNKS - SEAM_MARGIN;

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

/// 出生点（**体素**坐标）：`level.dat` 的 `SpawnX/Y/Z` × [`VOXELS_PER_BLOCK`]。读不到 → `None`
/// （调用方回退默认机位）。
pub fn spawn_eye() -> Option<IVec3> {
  let b = world::spawn(&map_dir())?;
  Some(IVec3::new(b[0], b[1], b[2]) * VOXELS_PER_BLOCK)
}

/// 建 MC 世界：起生产源、把流式窗口钉在相机周围，并**不预铺任何 chunk** ——
/// 一个 section 是 16³ 个方块（最多 4096 次模型体素化），同步铺一圈会卡住启动；内容交给流式环
/// 按"相机半径 + 射线请求"逐帧产出（与 `infinite_cubes` 同一套节流）。
///
/// 返回 `(场景信息, 生产源)`：源要交给 [`crate::infinite_cubes::Streaming::source`]，
/// 否则流式环起来的还是 `infinite_cubes` 的生产源。
pub fn build(grid: &mut VolumeGrid, cam_eye: IVec3) -> Result<(VoxSceneInfo, Arc<source::McCity>), String> {
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
  // 窗口半宽用**常量** `WINDOW_CHUNKS`（= 索引区上限），不读菜单的"粗档半径"：那个值在**菜单加载
  // 之后**才写进 `Streaming`，而窗口在启动建场时就要定下来 ⇒ 读到 0 的话窗口会退化成 ±1 chunk
  // （近场只剩脚下 5 m）⇒ 主世界一块都装不上（实测 `chunks 0`）。
  let win = WINDOW_CHUNKS;
  grid.set_stream_window(Some((c - IVec3::splat(win), IVec3::splat(win * 2))));
  // **覆盖半径**（分壳裁剪的接力半径，见 `VolumeGrid::set_coverage_r`）= **近场预载盘半径 − 余量**。
  //
  // 互斥与密铺要**分开**满足：**画**要互斥（裁剪面把主世界与 L1 切开，重叠带里两级的膨胀格会逐像素
  // 抢近 ⇒ 花斑），**内容**要重叠（预载多铺两圈，见 `NEAR_COVER_CHUNKS` 的 WHY —— 内容锚 chunk 角、
  // 裁剪按相机，不给余量就会在交接处留一圈**谁都没有内容**的环 ⇒ 楼被切掉一条）。两者不矛盾：
  // 裁剪只按距离切，内容重叠的部分不会被画两次。
  //
  // WARNING: **别把它放大到窗口半宽（32）去追求"彻底互斥"** —— 那会让预载盘体积 ×16：规划器每帧要
  // 枚举 8 万个格（`STREAM/MAIN` 从 ~1 ms 涨到 10+ ms），需求（1 万+ 块）又超过池上限（10922）⇒
  // 卡死 + 抖动（实测 `STREAM 10.67 / MAIN 11.38 ms/帧`、`chunks 10922 cap 10922 unload 6 ready 288`）。
  // 要放大，得先把 `region_chunks_of` 的预留口径改掉（按该级实际目标预留，而不是池块数 × 上界字数）
  // + 把规划器改成增量枚举。
  let cover = NEAR_COVER_CHUNKS;
  grid.set_coverage_r(cover as f32 * chunk as f32);
  // CONSTRAINT: 菜单"粗档半径"（**预载盘**半径）必须 ≥ 覆盖半径 + `SEAM_MARGIN`。默认两侧都是
  // `WINDOW_CHUNKS` 与 `WINDOW_CHUNKS − SEAM_MARGIN` ⇒ 天然成立；只有把滑块调到 < 30 chunk 才会
  // 破。**不在这里核对** —— `mc::build` 跑在菜单加载**之前**，读到的 `coarse_radius` 是占位默认值
  // （实测 3）⇒ 只会误报（同 `win` 那一处的 CONSTRAINT）。
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
  // **离线粗粒度世界**（[`lod`]）：算过一次、且与当前存档对得上（指纹）就装上 —— 远场 `scale ≥ 16`
  // 从此走内存查表；拿不到就**自动起一次后台重建**（回落到按需读 Anvil 只作为"重建还没完成"的临时
  // 状态，不是终局）。
  let stamp = lod::stamp(world.dir());
  let city = Arc::new(source::McCity::new(world, assets, pool));
  let lod_path = lod::path_for(MC_MAP);
  match lod::load(&lod_path, stamp) {
    Ok(f) => city.install_lod(f),
    // WARNING: 回落路径慢 **293–6305×**（L1 349–520 ms/chunk、L2 917 ms vs LOD 0.1–1.3 ms/chunk），
    // 表现为"远处一直空着 + 帧率长期偏低"⇒ 这条归因日志必须带代价，且必须触发重建。
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
      voxels_written: 0, // 由流式环逐帧铺
      voxels_dropped: 0,
    },
    city,
  ))
}

/// LOD 构建任务的进程级状态：`RUNNING` 防重复启动；`CANCEL` 供「中止」按钮置位（`mc::lod::build` 每列查一次）
static LOD_RUNNING: AtomicBool = AtomicBool::new(false);
static LOD_CANCEL: AtomicBool = AtomicBool::new(false);
/// 构建进度（%）：进度回调写，DebugMenu 的「LOD 状态」行读（见 `debug_menu::lod_state_tick`）
static LOD_PERCENT: AtomicU32 = AtomicU32::new(0);

/// `(在跑?, 进度%)`：给「LOD 状态」行用。没在跑时百分比无意义。
pub fn lod_build_state() -> (bool, u32) {
  (LOD_RUNNING.load(Ordering::Relaxed), LOD_PERCENT.load(Ordering::Relaxed))
}

/// 起一次 LOD 构建（后台线程；已在跑则忽略）。结果由 [`source::McCity::build_lod`] **热装**，不必重启：
/// 此后新的远场块直接走内存采样，已经在 GPU 上的那些自然按 LRU 换出。
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
      // 每跨 5% 一行（构建全程十几行；`build` 自己每 4096 列叫一次）
      if pct >= last.load(Ordering::Relaxed) + 5 {
        last.store(pct, Ordering::Relaxed);
        bevy::log::info!("LOD 构建 {pct}%（{done}/{total} 列，已用 {:.0}s）", t0.elapsed().as_secs_f64());
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

/// 中止正在进行的构建（没在跑就什么都不做）
pub fn cancel_lod_build() {
  if LOD_RUNNING.load(Ordering::SeqCst) {
    LOD_CANCEL.store(true, Ordering::SeqCst);
    bevy::log::info!("LOD 构建 → 中止请求");
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use super::summary::SOLID_MIN_PERMILLE;
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
      let cells = gate_voxel::CHUNK_SIZE / grain;
      let mut filled = 0usize;
      if let Some(t) = tree.as_ref() {
        for ky in 0..cells {
          for kz in 0..cells {
            for kx in 0..cells {
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
        "L{vol} scale {scale}：chunk {c:?} ⇒ {filled}/{} 格非空、树 {} 字 = {:.1} KB、{ms:.1} ms\
         （本块新增缓存区块 {reads}）",
        cells * cells * cells,
        tree.as_ref().map_or(0, |t| t.len_words()),
        tree.as_ref().map_or(0, |t| t.len_words()) as f64 * 4.0 / 1024.0,
      );
    }
    println!("三级合计新增读入区块 {total_reads}（缓存命中不再计数）");
    assert!(pool.len() > 1, "远场也应认领到调色板槽");
  }

  /// **远场表示是否"删几何"**（默认 `#[ignore]`）：粗档写侧的判据是"**整节实体占比 ≥ `SOLID_MIN_PERMILLE`
  /// 或任一 `4³` 细格 ≥ 同一门槛**才保留整节，其余丢掉"。若一个 section 明明有几何却因为两条都不过被丢掉
  /// ⇒ 那一节在粗档里**消失**（画面上就是"楼变中空 / 能看穿"），而细档是有的 ⇒ **LOD 不单调**（粗档删掉了
  /// 细档有的东西），任何 LOD 边界都必然突变，加档位只能把突变挪个距离、治不了根。
  ///
  /// 这一条**不需要知道相机距离**：直接拿存档按上面的**真实判据**统计"有几何但会被丢掉"的节与暴露面。
  /// 跑法：`cargo test --release -p gate-app --bin gate-app -- --ignored --nocapture real_map_rep`
  #[test]
  #[ignore = "需要外部存档：设 GATE_MC_MAP，或放默认路径"]
  fn real_map_far_rep_loses_thin_geometry() {
    let world = world::World::new(map_dir());
    let eye = spawn_eye().expect("level.dat 应有出生点");
    let (cx0, cz0) = (eye.x.div_euclid(256), eye.z.div_euclid(256));
    let mut buf = Vec::new();
    // 占比分桶（千分比）
    let mut buckets = [0usize; 9];
    let (mut nonempty, mut dropped_with_geom, mut kept) = (0usize, 0usize, 0usize);
    // 候选门槛（千分比）扫描：每档统计"整节占比与最厚细格占比**都不足**该门槛"的节数（即会被删的）+ 被删暴露面
    let mut sweep: Vec<(u32, usize, u64, u64)> =
      [125u32, 62, 32, 16, 8, 4, 2, 1].iter().map(|p| (*p, 0usize, 0u64, 0u64)).collect();
    // 可见表面（暴露面）：**"中空"= 可见表面被删**，比"体素占比"更贴近画面
    let (mut faces_kept, mut faces_dropped) = (0u64, 0u64);
    let mut examples: Vec<(i32, i32, i32, u32, u32)> = Vec::new();
    for dx in 0..16 {
      for dz in 0..16 {
        let Some(chunk) = world.chunk(cx0 + dx, cz0 + dz) else { continue };
        for sy in 0..world::SECTIONS_PER_CHUNK {
          let Some(sec) = chunk.section(sy).filter(|s| !s.is_empty_layer()) else { continue };
          nonempty += 1;
          sec.unpack_into(&mut buf);
          let solid = buf
            .iter()
            .filter(|&&i| sec.palette.get(i as usize).is_some_and(|s| !world::is_air(&s.name)))
            .count() as u32;
          let solid_at = |x: i32, y: i32, z: i32| -> bool {
            if !(0..16).contains(&x) || !(0..16).contains(&y) || !(0..16).contains(&z) {
              return false; // 节外按空气：边界面的多计在两个桶里同等发生，不影响比较
            }
            let i = (y * 256 + z * 16 + x) as usize;
            sec
              .palette
              .get(buf[i] as usize)
              .is_some_and(|s| !world::is_air(&s.name))
          };
          // 暴露面 = 实心格朝向空气（或节外）的那一面
          let mut faces = 0u64;
          for y in 0..16 {
            for z in 0..16 {
              for x in 0..16 {
                if solid_at(x, y, z) {
                  faces += u64::from(!solid_at(x - 1, y, z));
                  faces += u64::from(!solid_at(x + 1, y, z));
                  faces += u64::from(!solid_at(x, y - 1, z));
                  faces += u64::from(!solid_at(x, y + 1, z));
                  faces += u64::from(!solid_at(x, y, z - 1));
                  faces += u64::from(!solid_at(x, y, z + 1));
                }
              }
            }
          }
          let permille = solid * 1000 / world::SECTION_VOLUME as u32;
          let b = match permille {
            0..=62 => 0,     // < 1/16
            63..=124 => 1,   // < 1/8
            125..=249 => 2,
            250..=499 => 3,
            500..=749 => 4,
            750..=999 => 5,
            _ => 6,
          };
          buckets[b] += 1;
          if solid == 0 {
            continue;
          }
          // 一个 `4³` 细格里的实体数：整节是否"有厚结构"的判据（`summary` 的细格档、`lod` 的细格用的
          // 就是"某一格够实体"）。写侧真实判据 = 整节占比够 **或** 最厚细格占比够。
          let mut cells = [0u32; 64];
          let mut max_cell = 0u32;
          for gy in 0..4 {
            for gz in 0..4 {
              for gx in 0..4 {
                let mut n = 0u32;
                for y in 0..4 {
                  for z in 0..4 {
                    for x in 0..4 {
                      n += u32::from(solid_at(gx * 4 + x, gy * 4 + y, gz * 4 + z));
                    }
                  }
                }
                cells[(gx + 4 * gz + 16 * gy) as usize] = n;
                max_cell = max_cell.max(n);
              }
            }
          }
          let kept_by =
            |p: u32| max_cell * 1000 >= p * 64 || solid * 1000 >= p * world::SECTION_VOLUME as u32;
          for (p, n, f, occ) in sweep.iter_mut() {
            if !kept_by(*p) {
              *n += 1;
              *f += faces;
            }
            // 该门槛下会被画出的细格数（L1 树密度的代理量）
            *occ += cells.iter().filter(|c| **c * 1000 >= *p * 64).count() as u64;
          }
          if kept_by(SOLID_MIN_PERMILLE) {
            kept += 1;
            faces_kept += faces;
          } else {
            dropped_with_geom += 1;
            faces_dropped += faces;
            if examples.len() < 6 {
              examples.push((cx0 + dx, sy, cz0 + dz, solid, max_cell));
            }
          }
        }
      }
    }
    let total_faces = faces_kept + faces_dropped;
    println!("出生点 16×16 列共 {nonempty} 个非空节（× 64 细格 = {} 格）；门槛 P（‰）扫描：", nonempty * 64);
    for (p, n, f, occ) in &sweep {
      println!(
        "  P = {p:>3}‰ ⇒ 会被删 {n:>4} 节 / {f} 面（{:.4}%）；画出细格 {occ}（{:.2}%）",
        *f as f64 * 100.0 / total_faces.max(1) as f64,
        *occ as f64 * 100.0 / (nonempty * 64).max(1) as f64
      );
    }
    println!(
      "  当前门槛 {SOLID_MIN_PERMILLE}‰：保留 {kept}、**丢掉但有几何** {dropped_with_geom}、被删暴露面 \
       {faces_dropped}（{:.4}%）",
      faces_dropped as f64 * 100.0 / total_faces.max(1) as f64
    );
    println!(
      "  实体占比分桶（<1/16 / <1/8 / 1/8+ / 1/4+ / 1/2+ / 3/4+ / 满）：{:?}",
      &buckets[..7]
    );
    println!("  被丢掉的样例 (chunkX, Y, chunkZ, 实体数, 最厚细格实体数)：{examples:?}");
    assert!(nonempty > 0);
  }

  /// **离线粗粒度世界（`mc::lod`）端到端**（默认 `#[ignore]`，构建整张图要几分钟）：
  /// 建一份文件 → 同一个远场 chunk 分别用 LOD 与 Anvil 两条路产出 → 打印耗时/树长/逐格色差。
  /// 这是"LOD 真能替代按需读 Anvil"的直接证据，也是文件大小的依据。
  /// 跑法：`cargo test --release -p gate-app --bin gate-app -- --ignored --nocapture real_map_build_lod`
  #[test]
  #[ignore = "需要外部存档与资产：设 GATE_MC_MAP / GATE_MC_ASSETS，或放默认路径"]
  fn real_map_build_lod() {
    use crate::infinite_cubes::{FAR_GRAIN, FAR_SCALES};
    use std::sync::atomic::{AtomicBool, AtomicUsize};

    let world = std::sync::Arc::new(world::World::new(map_dir()));
    let assets = std::sync::Arc::new(assets::Assets::new(assets_root()));
    let pool = std::sync::Arc::new(material::Pool::new());
    // 测试写到**独立文件名**（不碰真 `mc_map.lod`）；`GATE_LOD_REAL=1` 时写真实文件名并留下它 ——
    // 应用级实跑（`GATE_BENCH=orbit`）需要一份与当前格式对得上的文件。
    let keep = std::env::var("GATE_LOD_REAL").is_ok_and(|v| v != "0");
    let path = if keep { lod::path_for(MC_MAP) } else { lod::path_for("mc_map_test") };
    let cancel = AtomicBool::new(false);
    let step = AtomicUsize::new(0);
    let (stats, f) = lod::build(&world, &path, &cancel, |d, t| {
      if step.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % 8 == 0 {
        println!("  构建 {:.1}%（{d}/{t} 列）", d as f64 * 100.0 / t as f64);
      }
    })
    .expect("构建应成功");
    println!(
      "LOD 文件：{} 列、{} 节、名字 {} 个、{:.1} MB、{:.1} s（{:.0} 列/s）",
      stats.cols,
      stats.sections,
      stats.names,
      stats.bytes as f64 / 1048576.0,
      stats.secs,
      stats.cols as f64 / stats.secs,
    );

    // 两条路各建一个源（`McCity` 自持 LOD 句柄，互不干扰）
    let with_lod = std::sync::Arc::new(source::McCity::new(world.clone(), assets.clone(), pool.clone()));
    with_lod.install_lod(f);
    let anvil = std::sync::Arc::new(source::McCity::new(world, assets, pool));
    let eye = spawn_eye().expect("level.dat 应有出生点");
    let mut scratch = VolumeGrid::new();
    // 每轴格数与格的方块边长都由级参数推出（见 `source::cell_blocks_of` / `far_tree`）
    let cells = gate_voxel::CHUNK_SIZE / FAR_GRAIN;
    for (i, &scale) in FAR_SCALES.iter().enumerate() {
      let vol = i + 1;
      let cell_blocks = FAR_GRAIN * scale / VOXELS_PER_BLOCK;
      if !with_lod.lod().is_some_and(|v| v.supports(cell_blocks)) {
        println!("L{vol}（scale {scale}，格 {cell_blocks} 方块）不在 LOD 能表达的档里，跳过");
        continue;
      }
      let c = (eye / scale).div_euclid(IVec3::splat(gate_voxel::CHUNK_SIZE));
      let t = std::time::Instant::now();
      let tl = with_lod.produce(vol, ChunkCoord(c), Detail::Full, &mut scratch);
      let ms_lod = t.elapsed().as_secs_f64() * 1000.0;
      let t = std::time::Instant::now();
      let ta = anvil.produce(vol, ChunkCoord(c), Detail::Full, &mut scratch);
      let ms_anvil = t.elapsed().as_secs_f64() * 1000.0;
      let words = |t: &Option<gate_voxel::ChunkTree>| t.as_ref().map_or(0, |t| t.len_words());
      // 逐格比色（格原点处的体素就是那一格填的代表色）
      let (mut same, mut diff, mut both_empty, mut one_side) = (0usize, 0usize, 0usize, 0usize);
      for ky in 0..cells {
        for kz in 0..cells {
          for kx in 0..cells {
            let p = IVec3::new(kx, ky, kz) * FAR_GRAIN;
            let a = ta.as_ref().and_then(|t| t.get_voxel(p.x, p.y, p.z));
            let b = tl.as_ref().and_then(|t| t.get_voxel(p.x, p.y, p.z));
            match (a, b) {
              (None, None) => both_empty += 1,
              (Some(x), Some(y)) if x == y => same += 1,
              (None, Some(_)) | (Some(_), None) => one_side += 1,
              _ => diff += 1,
            }
          }
        }
      }
      println!(
        "L{vol} scale {scale}：LOD {ms_lod:.1} ms / {} 字 vs Anvil {ms_anvil:.1} ms / {} 字 ⇒ {:.0}×；\
         逐格：同色 {same}、异色 {diff}、单边有料 {one_side}、都空 {both_empty}",
        words(&tl),
        words(&ta),
        ms_anvil / ms_lod.max(0.001),
      );
      assert!(ms_lod < ms_anvil, "LOD 应该更快：{ms_lod:.1} vs {ms_anvil:.1} ms");
      // 口径差（见 `lod` 模块头）：格 = 一个节时两条路几乎同口径 ⇒ 允许一成；格 = 细格（4 方块）时
      // 平手取色的规则不同（LOD 按节内调色板序、summary 按全局槽号）；格 > 一个节时 LOD 采满格内
      // `n³` 个节、而 `summary` 只采格中心那一列 ⇒ 差异本来就更大（那是**更准**，不是错）。
      //
      // ⚠️ 表面色口径（`voxel::rep_of_surface`）之后，L3 由 <25% 升到 **34%**：色取"最上面那一层
      // 够实的层"，而"哪一层够实"**对采样范围敏感** —— LOD 采满格内 64 个节、`summary` 只看中心那
      // 4 个，偏心内容会让两边挑中不同的层。这是**范围差**（不是规则的错），所以放宽到 40% 而不是
      // 改规则；L1/L2 两条路范围几乎相同，仍钉在 15% / 10%（实测 11.7% / 7.5%）。
      let (tol_pct, why) = if cell_blocks == lod::CELL {
        (10, "同口径")
      } else if cell_blocks == lod::FINE_CELL {
        (15, "细格：平手取色的规则不同")
      } else {
        (40, "LOD 采满格内各节 vs summary 只采中心列：表面色口径下「哪层够实」对范围敏感，实测 34%")
      };
      // "单边有料"= 两条路对"这一格空不空"不一致。格 = 细格 / 一个节时口径几乎相同 ⇒ 钉 5%；
      // 格 > 一个节时 LOD **采满格内 `n³` 个节**、`summary` 只采格中心那一列 ⇒ LOD 必然多看见一些
      // 偏心内容（那是更准，不是错）⇒ 放宽到 10%。（实测 L3 由 5% 升到 6%：口径门槛降到 16‰ 后
      // 保留的节变多，中心列的漏采随之变多。）
      let one_side_pct = if cell_blocks == lod::FINE_CELL || cell_blocks == lod::CELL { 5 } else { 10 };
      assert!(one_side * 100 < one_side_pct * (same + diff + one_side), "单边有料应 < {one_side_pct}%：{one_side}");
      assert!(
        (diff + one_side) * 100 < tol_pct * (same + diff + one_side),
        "异色应 < {tol_pct}%（{why}）：{diff} + {one_side} / {}",
        same + diff + one_side
      );
    }
    if !keep {
      let _ = std::fs::remove_file(&path);
    }
  }
}
