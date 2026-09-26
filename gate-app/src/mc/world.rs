//! **MC 存档读取**（Anvil，1.13–1.17）：格式里最难的部分交给现成 crate，本模块只做"翻译 + 缓存 + 并发"。
//!
//! | 环节 | 谁做 |
//! |---|---|
//! | region 4 KB 索引、扇区载荷、压缩（zlib / gzip / lz4 / 裸） | [`fastanvil::Region::read_chunk`] |
//! | NBT → Rust 结构 | [`fastnbt::from_bytes`]（serde） |
//! | `BlockStates` 的位打包展开（1.15 跨长 / 1.16 留白两种） | [`fastanvil::expand_blockstates`] |
//! | 本节里的领域结构（`Name` + `Properties`） | 本模块 |
//!
//! **为什么不用 `fastanvil::pre18::JavaChunk`**：那个结构把方块压成 `Block`，而 `Block` 只公开
//! `name()` / `encoded_description()`，其中 `encoded_description` **有意丢掉了 `waterlogged` 与
//! `powered` 两个属性**（它服务于 fastanvil 自己的彩色渲染）。而 blockstate 的 `variants` 条件匹配
//! 要按属性找形状（台阶朝向、楼梯形状、门的开合、红石灯亮灭都在属性里）⇒ 属性必须原样保留。
//! fastanvil 的文档也是这个用法："You can create your own chunk structures to (de)serialize using fastnbt"。
//!
//! **并发**：`ChunkSource::produce` 在 worker 线程上以 `&self` 调用，而读一个区块是
//! `seek + 解压 + NBT 解析`（0.5–3 ms），且 `Region::read_chunk` 要 `&mut self` ⇒ region 句柄表与
//! 已解析的区块都进 `Mutex`。已解析区块再挂一层**有界 LRU**：同一个区块的 16 个 section 会被先后
//! 产出、相邻 chunk 也会复用（3 个 worker 各自的区块集合在空间上连续）。

use std::collections::{HashMap, HashSet, VecDeque};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use fastanvil::Region;
use serde::Deserialize;

/// 一个 region 覆盖 32×32 个 chunk
pub const REGION_CHUNKS: i32 = 32;
/// 一个 section 的方块数（16³）
pub const SECTION_VOLUME: usize = 4096;
/// 一个区块的 section 数（1.17 世界高 256）
pub const SECTIONS_PER_CHUNK: i32 = 16;

/// 已解析区块的缓存条数上限（一条 = 16 个 section 的方块数组 ≈ 150 KB ⇒ 64 条约 10 MB）。
const CHUNK_CACHE: usize = 64;

/// 一个方块状态：名字 + 属性（都来自区块调色板）
#[derive(Debug, Clone, Default, Deserialize)]
pub struct BlockState {
  #[serde(rename = "Name", default)]
  pub name: String,
  #[serde(rename = "Properties", default)]
  pub props: HashMap<String, String>,
}

impl BlockState {
  /// 去掉 `minecraft:` 前缀
  pub fn short_name(&self) -> &str {
    self.name.strip_prefix("minecraft:").unwrap_or(&self.name)
  }

  pub fn prop(&self, key: &str) -> Option<&str> {
    self.props.get(key).map(String::as_str)
  }

  /// 规范键（属性按名排序）：blockstate 的 `variants` / `multipart` 条件只把属性当**集合**看，
  /// 键的顺序无关 ⇒ 生成计划时用它去重（`floor(...)` 那些带浮点的键另说，见 `model`）。
  pub fn key(&self) -> String {
    let mut props: Vec<(&str, &str)> = self.props.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
    props.sort_unstable();
    let mut s = String::with_capacity(self.name.len() + 16 * props.len());
    s.push_str(&self.name);
    for (k, v) in props {
      s.push(',');
      s.push_str(k);
      s.push('=');
      s.push_str(v);
    }
    s
  }
}

/// 一个 section 的方块：调色板 + 4096 个下标（`y*256 + z*16 + x`，x 最密）
#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct Section {
  #[serde(default)]
  pub y: i8,
  #[serde(default)]
  pub palette: Vec<BlockState>,
  /// 位打包的调色板下标（1.17 对全空气层不写这个字段）
  #[serde(rename = "BlockStates", default)]
  states: Option<fastnbt::LongArray>,
}

impl Section {
  /// 该层有没有实体方块。两种情况：调色板为空（1.17 对全空层不写 `Palette` / `BlockStates`，只留一个
  /// `Y` —— 真地图实测该层的 tags 就是 `["Y"]`）；或调色板只有一项且那一项是空气。
  pub fn is_empty_layer(&self) -> bool {
    self.palette.is_empty() || (self.palette.len() == 1 && is_air(&self.palette[0].name))
  }

  /// 展开本层 4096 格的**调色板下标**（下标序 `y*256 + z*16 + x`，x 最密；与 MC 一致）。
  ///
  /// CONSTRAINT: 展开是 4096 格 × 复制的批量操作，**一节只许调一次**（逐格调 = 每格一次 8 KB 分配）。
  /// 拿到 `buf` 之后按格查 `palette[idx]` 才是热路径。
  pub fn unpack_into(&self, buf: &mut Vec<u16>) {
    buf.clear();
    match &self.states {
      Some(s) => {
        let mut v = fastanvil::expand_blockstates(s, self.palette.len().max(1));
        v.resize(SECTION_VOLUME, 0);
        *buf = v;
      }
      // 没有 `BlockStates` + 调色板非空 = 整层同一种方块（下标恒 0）
      None => buf.resize(SECTION_VOLUME, 0),
    }
  }
}

/// 区块 NBT 的根：只要 `Level`（其余字段与体素化无关）
#[derive(Deserialize)]
struct ChunkRoot {
  #[serde(rename = "Level")]
  level: LevelNbt,
}

#[derive(Deserialize)]
struct LevelNbt {
  #[serde(rename = "Sections", default)]
  sections: Vec<Section>,
}

/// 一个已解析的区块（1.17：`Level.Sections[]`，每项带自己的 `Y`）
#[derive(Debug, Default, Deserialize)]
pub struct Chunk {
  #[serde(rename = "Sections", default)]
  pub sections: Vec<Section>,
}

impl Chunk {
  /// 取 `section_y` 那一层
  pub fn section(&self, section_y: i32) -> Option<&Section> {
    self.sections.iter().find(|s| s.y as i32 == section_y)
  }
}

#[derive(Default)]
struct State {
  /// region 句柄（`None` = 这个 region 文件不存在）
  regions: HashMap<(i32, i32), Option<Region<File>>>,
  chunks: HashMap<(i32, i32), Arc<Chunk>>,
  order: VecDeque<(i32, i32)>,
  /// 读 / 解析失败的区块（记住它，免得每帧重试同一个坏文件）
  bad: HashSet<(i32, i32)>,
  /// **真的去过 region 的**次数（诊断：远场成本模型按"读了多少个祖先 chunk 列"核对，见 `mc::summary`）
  reads: usize,
}

/// 一个 Anvil 世界目录（`region/` + `level.dat`）
pub struct World {
  dir: PathBuf,
  state: Mutex<State>,
}

impl World {
  pub fn new(dir: impl Into<PathBuf>) -> Self {
    Self { dir: dir.into(), state: Mutex::new(State::default()) }
  }

  pub fn dir(&self) -> &Path {
    &self.dir
  }

  pub fn region_path(&self, rx: i32, rz: i32) -> PathBuf {
    self.dir.join("region").join(format!("r.{rx}.{rz}.mca"))
  }

  /// 打开一个 region。**文件不存在是稀疏存档的正常情况**（356/1156 个 region 有料）⇒ 静默 `None`；
  /// 文件在却打不开 / 头不合法属真异常 ⇒ `warn!` 带原因。
  fn region(&self, rx: i32, rz: i32) -> Option<Region<File>> {
    let path = self.region_path(rx, rz);
    match File::open(&path) {
      Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
      Err(e) => {
        bevy::log::warn!("MC region {} 打开失败：{e} → 当空", path.display());
        None
      }
      Ok(f) => match Region::from_stream(f) {
        Ok(r) => Some(r),
        Err(e) => {
          // 稀疏存档里存在**截断 / 空**的 region 文件（地图边缘、被裁剪过的旧文件），远场扫描会大量
          // 触及 ⇒ 这是正常数据形态，不是本程序的失败。真读坏区块时下面还有逐区块的 `warn!`。
          bevy::log::debug!("MC region {} 头部不合法：{e} → 当空", path.display());
          None
        }
      },
    }
  }

  /// chunk 坐标 → 已解析的区块（`None` = 该位置没有区块，或读取 / 解析失败）
  pub fn chunk(&self, cx: i32, cz: i32) -> Option<Arc<Chunk>> {
    let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(c) = st.chunks.get(&(cx, cz)) {
      return Some(c.clone());
    }
    if st.bad.contains(&(cx, cz)) {
      return None;
    }
    let key = (cx.div_euclid(REGION_CHUNKS), cz.div_euclid(REGION_CHUNKS));
    let (lx, lz) = (cx.rem_euclid(REGION_CHUNKS) as usize, cz.rem_euclid(REGION_CHUNKS) as usize);
    // 开过的 region 记成 `Some/None` 两个状态：`None` = 文件不存在（稀疏存档的正常情况），别每次重开
    let region =
      st.regions.entry(key).or_insert_with(|| self.region(key.0, key.1)).as_mut();
    // `read_chunk` 给的是**解压后**的 NBT（压缩类型它自己认；`x`/`z` 是 region 内的 0..32）
    let raw = region.map(|r| r.read_chunk(lx, lz)).unwrap_or(Ok(None));
    st.reads += 1;
    let parsed = match raw {
      Ok(Some(bytes)) => match fastnbt::from_bytes::<ChunkRoot>(&bytes) {
        Ok(root) => Some(Arc::new(Chunk { sections: root.level.sections })),
        Err(e) => {
          bevy::log::warn!("MC 区块 ({cx},{cz}) NBT 解析失败：{e} → 当空");
          None
        }
      },
      Ok(None) => None, // 该位置没有区块（稀疏存档的正常情况）
      Err(e) => {
        bevy::log::warn!("MC 区块 ({cx},{cz}) 读取失败：{e} → 当空");
        None
      }
    };
    match parsed {
      Some(chunk) => {
        st.chunks.insert((cx, cz), chunk.clone());
        st.order.push_back((cx, cz));
        while st.order.len() > CHUNK_CACHE {
          if let Some(old) = st.order.pop_front() {
            st.chunks.remove(&old);
          }
        }
        Some(chunk)
      }
      None => {
        st.bad.insert((cx, cz));
        None
      }
    }
  }

  /// 诊断读数：`(已缓存区块数, 读取失败区块数)`
  pub fn stats(&self) -> (usize, usize) {
    let st = self.state.lock().unwrap_or_else(|e| e.into_inner());
    (st.chunks.len(), st.bad.len())
  }

  /// 诊断读数：累计**真去过 region** 的次数（命中缓存不计数）—— 远场成本模型的核对点。
  /// 只有 `real_map_far_produce` 这条取证测试在非测试构建里不用它，故随测试编译。
  #[cfg(test)]
  pub fn reads(&self) -> usize {
    self.state.lock().unwrap_or_else(|e| e.into_inner()).reads
  }
}

/// 空气（三种）：`air` / `cave_air` / `void_air`
pub fn is_air(name: &str) -> bool {
  matches!(name, "minecraft:air" | "minecraft:cave_air" | "minecraft:void_air")
}

/// `level.dat` 的出生点（**方块**坐标）；缺失 / 解析失败 → `None`（调用方回退默认机位）。
/// `level.dat` 是 gzip 包着的 NBT：解压用 `flate2`，解析仍走 `fastnbt`。
pub fn spawn(dir: &Path) -> Option<[i32; 3]> {
  #[derive(Deserialize)]
  struct Root {
    #[serde(rename = "Data")]
    data: Data,
  }
  #[derive(Deserialize)]
  struct Data {
    #[serde(rename = "SpawnX", default)]
    x: i32,
    #[serde(rename = "SpawnY", default)]
    y: i32,
    #[serde(rename = "SpawnZ", default)]
    z: i32,
  }
  let f = File::open(dir.join("level.dat")).ok()?;
  let mut buf = Vec::new();
  std::io::Read::read_to_end(&mut flate2::read::GzDecoder::new(f), &mut buf).ok()?;
  let root: Root = fastnbt::from_bytes(&buf).ok()?;
  Some([root.data.x, root.data.y, root.data.z])
}

/// `Dir` 是否为 Anvil 世界（有 `region/` 或 `level.dat`）
pub fn looks_like_world(dir: &Path) -> bool {
  dir.join("region").is_dir() || dir.join("level.dat").is_file()
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn air_names_are_the_three_vanilla_ones() {
    assert!(is_air("minecraft:air") && is_air("minecraft:cave_air") && is_air("minecraft:void_air"));
    assert!(!is_air("minecraft:stone") && !is_air("air"));
  }

  /// 规范键与属性顺序无关（`variants` 的条件也当集合看）
  #[test]
  fn state_key_is_order_independent() {
    let mk = |pairs: &[(&str, &str)]| BlockState {
      name: "minecraft:oak_stairs".into(),
      props: pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect(),
    };
    let a = mk(&[("facing", "east"), ("half", "bottom")]);
    let b = mk(&[("half", "bottom"), ("facing", "east")]);
    assert_eq!(a.key(), b.key());
    assert_eq!(a.key(), "minecraft:oak_stairs,facing=east,half=bottom");
  }

  /// **真地图冒烟**（默认 `#[ignore]`：依赖外部存档）。这一条是"现成 crate 真能吃下 Greenfield 1.17.1"
  /// 的直接证据 —— 打印出生点区块各层的调色板规模、方块名统计与出生点那一格。
  /// 跑法：`cargo test --release -p gate-app --bin gate-app -- --ignored --nocapture real_map`
  #[test]
  #[ignore = "需要外部存档：设 GATE_MC_MAP，或放默认路径"]
  fn real_map_spawn_chunk_reads() {
    let dir = std::env::var("GATE_MC_MAP")
      .unwrap_or_else(|_| r"C:\game\Greenfield v0.5.4\Greenfield v0.5.4".to_string());
    let world = World::new(&dir);
    let sp = spawn(Path::new(&dir)).expect("level.dat 应有出生点");
    println!("出生点方块 {sp:?} ⇒ chunk ({}, {})", sp[0].div_euclid(16), sp[2].div_euclid(16));
    let (cx, cz) = (sp[0].div_euclid(16), sp[2].div_euclid(16));
    let chunk = world.chunk(cx, cz).expect("出生点区块应存在");
    let mut names: std::collections::BTreeMap<String, usize> = Default::default();
    let mut layers = 0;
    let mut states = Vec::new();
    for sy in 0..SECTIONS_PER_CHUNK {
      let Some(sec) = chunk.section(sy).filter(|s| !s.is_empty_layer()) else { continue };
      layers += 1;
      sec.unpack_into(&mut states);
      let (mut solid, mut air) = (0usize, 0usize);
      for idx in states.iter() {
        let b = &sec.palette[*idx as usize];
        if is_air(&b.name) {
          air += 1;
        } else {
          solid += 1;
          *names.entry(b.short_name().to_string()).or_default() += 1;
        }
      }
      println!(
        "  section y={sy:<2} 调色板 {:>4} 项  实体 {solid:>5}/4096（空气 {air}）",
        sec.palette.len()
      );
    }
    assert!(layers > 0, "至少要有一层有数据");
    let mut top: Vec<_> = names.into_iter().collect();
    top.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
    println!("非空气方块（按格数，前 15）：");
    for (n, c) in top.iter().take(15) {
      println!("  {n:<34} {c}");
    }
    let sec = chunk.section(sp[1].div_euclid(16)).expect("该层存在");
    sec.unpack_into(&mut states);
    let i = (sp[1].rem_euclid(16) * 256 + sp[2].rem_euclid(16) * 16 + sp[0].rem_euclid(16)) as usize;
    let b = &sec.palette[states[i] as usize];
    println!("出生点那一格 = {} {:?}", b.name, b.props);
    assert!(!is_air(&b.name), "出生点正下方不该是空气");
    println!("缓存读数 (区块数, 读取失败数) = {:?}", world.stats());
  }
}
