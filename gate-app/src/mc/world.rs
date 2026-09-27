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
//! `seek + 解压 + NBT 解析`（0.5–3 ms），且 `Region::read_chunk` 要 `&mut self` ⇒ 句柄表与已解析的
//! 区块都要加锁。旧版是**一把全局锁罩住整张地图** ⇒ 3 个 worker 事实上串成 1 个读取器（实测量级
//! 远场 ≈ 2 块/s，与 worker 数无关）。现在拆两层：
//!
//! - **按 region 分片**（[`SHARDS`] 把锁拆开）：不同 region 的读互不等待；
//! - **每 region 一池独立句柄**（[`READERS_PER_REGION`]）：`Region` 自带 seek 位置，句柄各自一把锁
//!   ⇒ **同一个 region 内的多个 worker 也能并行读**（远场一块的 256 列大多落在同一个 region）。
//!
//! I/O 一律在分片锁外做，分片锁只保护查表与记账。已解析的列再挂一层**全局字节预算的 LRU**
//! （[`COL_CACHE_BYTES`]）：同一个列的 16 个 section 会被先后产出、相邻列也会复用。

use std::collections::{HashMap, HashSet, VecDeque};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use fastanvil::Region;
use serde::Deserialize;

/// 一个 region 覆盖 32×32 个 chunk
pub const REGION_CHUNKS: i32 = 32;
/// 一个 section 的方块数（16³）
pub const SECTION_VOLUME: usize = 4096;
/// 一个区块的 section 数（1.17 世界高 256）
pub const SECTIONS_PER_CHUNK: i32 = 16;

/// 分片数：锁按 region 拆成这么多把。worker 只有 3 个，分片只为"不同 region 别互相等锁"。
const SHARDS: usize = 32;
/// 一个 region 文件上并行读的独立句柄数。
///
/// CONSTRAINT: 必须 ≥ 生产 worker 数（`infinite_cubes::PRODUCER_WORKERS`）；两者一起改。
const READERS_PER_REGION: usize = 4;
/// 每个分片保留的 region 句柄池数（LRU）。全图 356 个 region 有料，这个量级基本不发生换出。
const REGION_FILES_PER_SHARD: usize = 64;

/// **列缓存的全局字节预算**（跨分片共用一个计数）。
///
/// 解析后的列不轻：一个城市列 ≈ 8 个非空节，每节的调色板（`String` + `HashMap` 属性表）就有几十 KB
/// （口径见 [`Chunk::heap_bytes`]）⇒ 按**条数**封顶会随分片数翻几十倍，只能全局按字节记。
/// 取值 ≈ 1400 列（实测 `heap_bytes` 36 KB/列）＝ 近场那一圈（相机周围 ±8 chunk × 13 层 ≈ 290 列）
/// 的 5 倍余量：远场那些**只读一次**的列被 LRU 先换出去，近场复用的列留得下。
const COL_CACHE_BYTES: usize = 48 * 1024 * 1024;

/// 一个调色板项的堆占用估值（[`Chunk::heap_bytes`] 用）：`String` 的名字 + `HashMap` 的属性表
/// 与两侧字符串。
const PALETTE_ENTRY_BYTES: usize = 96;

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

  /// [`Self::key`] 的逆：`mc::lod` 的文件里存的是规范键（纯数据、与调色板无关），装载时用它还原成
  /// 状态再去查计划（`mc::source` 的 `plan_for`）。
  pub fn from_key(key: &str) -> Self {
    let mut it = key.split(',');
    let name = it.next().unwrap_or_default().to_string();
    let mut props = HashMap::new();
    for kv in it {
      if let Some((k, v)) = kv.split_once('=') {
        props.insert(k.to_string(), v.to_string());
      }
    }
    Self { name, props }
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

  /// 堆占用的**估算**（列缓存的字节预算用，不是精确测量）：每个调色板项按
  /// [`PALETTE_ENTRY_BYTES`]、位数组按 8 B/字。
  pub fn heap_bytes(&self) -> usize {
    self
      .sections
      .iter()
      .map(|s| s.palette.len() * PALETTE_ENTRY_BYTES + s.states.as_ref().map_or(0, |a| a.len() * 8))
      .sum()
  }
}

/// 一个 region 文件的**读句柄池**：`Region` 自带 seek 位置 ⇒ 每个句柄各自一把锁，同一个文件也能并行读。
struct RegionFile {
  path: PathBuf,
  /// `None` = 还没开（按需开：远场扫描会碰到上百个 region，一上来全开就是上百个 fd）
  readers: Box<[Mutex<Option<Region<File>>>]>,
  /// 轮转取用哪个句柄
  next: AtomicUsize,
  /// 句柄 0 开不出来（不存在 / 头不合法）⇒ 整条当空（稀疏存档的正常形态，见 [`open_region`]）
  dead: bool,
}

impl RegionFile {
  fn new(path: PathBuf) -> Self {
    let first = open_region(&path, true);
    let dead = first.is_none();
    let mut readers: Vec<Mutex<Option<Region<File>>>> =
      (0..READERS_PER_REGION).map(|_| Mutex::new(None)).collect();
    readers[0] = Mutex::new(first);
    Self { path, readers: readers.into_boxed_slice(), next: AtomicUsize::new(0), dead }
  }

  /// 读一列（`lx` / `lz` = region 内 0..32）：`Ok(None)` = 该位置没有区块；`Err` = 读不了（带原因）。
  fn read(&self, lx: usize, lz: usize) -> Result<Option<Vec<u8>>, String> {
    if self.dead {
      return Ok(None);
    }
    let i = self.next.fetch_add(1, Ordering::Relaxed) % self.readers.len();
    let mut slot = lock(&self.readers[i]);
    if slot.is_none() {
      match open_region(&self.path, false) {
        Some(r) => *slot = Some(r),
        // 句柄 0 已经报过原因，这里再报就是重复噪音
        None => return Ok(None),
      }
    }
    let r = slot.as_mut().expect("刚判过");
    r.read_chunk(lx, lz).map_err(|e| e.to_string())
  }
}

/// 开一个 region 句柄。**文件不存在是稀疏存档的正常情况**（356/1156 个 region 有料）⇒ 静默 `None`；
/// 文件在却打不开属真异常 ⇒ `loud` 时 `warn!` 带原因。
fn open_region(path: &Path, loud: bool) -> Option<Region<File>> {
  match File::open(path) {
    Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
    Err(e) => {
      if loud {
        bevy::log::warn!("MC region {} 打开失败：{e} → 当空", path.display());
      } else {
        bevy::log::debug!("MC region {} 打开失败：{e} → 当空", path.display());
      }
      None
    }
    Ok(f) => match Region::from_stream(f) {
      Ok(r) => Some(r),
      Err(e) => {
        // 稀疏存档里存在**截断 / 空**的 region 文件（地图边缘、被裁剪过的旧文件），远场扫描会大量
        // 触及 ⇒ 这是正常数据形态，不是本程序的失败。真读坏区块时 `World::chunk` 还有逐区块的 `warn!`。
        bevy::log::debug!("MC region {} 头部不合法：{e} → 当空", path.display());
        None
      }
    },
  }
}

/// 一个分片的账目：region 句柄池 + 已解析的列。**分片锁只保护这里**，I/O 与 NBT 解析都在锁外。
#[derive(Default)]
struct Shard {
  /// region 句柄池（LRU：`file_order` 最旧的先丢；句柄被 `Arc` 借走时不会被提前关）
  files: HashMap<(i32, i32), Arc<RegionFile>>,
  file_order: VecDeque<(i32, i32)>,
  /// 列 → `(已解析的区块, 它的 [`Chunk::heap_bytes`])`
  cols: HashMap<(i32, i32), (Arc<Chunk>, usize)>,
  col_order: VecDeque<(i32, i32)>,
  /// 读 / 解析失败的区块（记住它，免得每帧重试同一个坏文件）
  bad: HashSet<(i32, i32)>,
  /// **真的去过 region 的**次数（诊断：远场成本模型按"读了多少个祖先 chunk 列"核对，见 `mc::summary`）
  reads: usize,
}

/// 一个 Anvil 世界目录（`region/` + `level.dat`）
pub struct World {
  dir: PathBuf,
  shards: Box<[Mutex<Shard>]>,
  /// 列缓存的全局字节计数（见 [`COL_CACHE_BYTES`]）
  col_bytes: AtomicUsize,
}

impl World {
  pub fn new(dir: impl Into<PathBuf>) -> Self {
    Self {
      dir: dir.into(),
      shards: (0..SHARDS).map(|_| Mutex::new(Shard::default())).collect(),
      col_bytes: AtomicUsize::new(0),
    }
  }

  pub fn dir(&self) -> &Path {
    &self.dir
  }

  pub fn region_path(&self, rx: i32, rz: i32) -> PathBuf {
    self.dir.join("region").join(format!("r.{rx}.{rz}.mca"))
  }

  /// chunk 坐标 → 已解析的区块（`None` = 该位置没有区块，或读取 / 解析失败）。
  ///
  /// 同一列被两个 worker 同时要会读两次（白读一遍）—— 不值得为它加在飞表：需求表本就把同一块去重了，
  /// 撞上的概率远低于锁的收益。
  pub fn chunk(&self, cx: i32, cz: i32) -> Option<Arc<Chunk>> {
    let key = (cx, cz);
    let shard = self.shard(cx, cz);
    {
      let mut st = lock(shard);
      if let Some((c, _)) = st.cols.get(&key) {
        let c = c.clone();
        st.col_order.retain(|k| *k != key);
        st.col_order.push_back(key);
        return Some(c);
      }
      if st.bad.contains(&key) {
        return None;
      }
    }
    let (rx, rz) = region_key(cx, cz);
    let file = self.region_file(shard, rx, rz);
    let (lx, lz) = (cx.rem_euclid(REGION_CHUNKS) as usize, cz.rem_euclid(REGION_CHUNKS) as usize);
    // `read_chunk` 给的是**解压后**的 NBT（压缩类型它自己认；`x`/`z` 是 region 内的 0..32）
    let parsed = match file.read(lx, lz) {
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
    let mut st = lock(shard);
    st.reads += 1;
    match parsed {
      Some(chunk) => {
        let size = chunk.heap_bytes();
        if let Some((_, old)) = st.cols.insert(key, (chunk.clone(), size)) {
          self.col_bytes.fetch_sub(old, Ordering::Relaxed);
        }
        st.col_order.retain(|k| *k != key);
        st.col_order.push_back(key);
        // 超全局预算就从**本分片**的 LRU 头换出（远场那些只读一次的列都在头部；精确到全局的最旧不在
        // 这把锁里，故此处的换出是"本片优先"的近似，超出的部分由下一个分片自己收敛）
        let mut total = self.col_bytes.fetch_add(size, Ordering::Relaxed) + size;
        while total > COL_CACHE_BYTES {
          let Some(old_key) = st.col_order.pop_front() else { break };
          if let Some((_, freed)) = st.cols.remove(&old_key) {
            total = self.col_bytes.fetch_sub(freed, Ordering::Relaxed).saturating_sub(freed);
          }
        }
        Some(chunk)
      }
      None => {
        st.bad.insert(key);
        None
      }
    }
  }

  /// 取（或建）某个 region 的句柄池
  fn region_file(&self, shard: &Mutex<Shard>, rx: i32, rz: i32) -> Arc<RegionFile> {
    let key = (rx, rz);
    let mut st = lock(shard);
    if let Some(f) = st.files.get(&key).cloned() {
      st.file_order.retain(|k| *k != key);
      st.file_order.push_back(key);
      return f;
    }
    let f = Arc::new(RegionFile::new(self.region_path(rx, rz)));
    st.files.insert(key, f.clone());
    st.file_order.push_back(key);
    while st.files.len() > REGION_FILES_PER_SHARD {
      let Some(old) = st.file_order.pop_front() else { break };
      st.files.remove(&old);
    }
    f
  }

  /// 这一列落在哪个分片：按 region 定（同一个 region 恒同片 ⇒ 句柄池与缓存不会分叉）
  fn shard(&self, cx: i32, cz: i32) -> &Mutex<Shard> {
    let (rx, rz) = region_key(cx, cz);
    &self.shards[shard_index(rx, rz)]
  }

  /// 诊断读数：`(已缓存区块数, 读取失败区块数)`
  pub fn stats(&self) -> (usize, usize) {
    self.shards.iter().fold((0, 0), |(n, bad), s| {
      let st = lock(s);
      (n + st.cols.len(), bad + st.bad.len())
    })
  }

  /// 诊断读数：累计**真去过 region** 的次数（命中缓存不计数）—— 远场成本模型的核对点。
  /// 只有 `real_map_far_produce` 这条取证测试在非测试构建里不用它，故随测试编译。
  #[cfg(test)]
  pub fn reads(&self) -> usize {
    self.shards.iter().map(|s| lock(s).reads).sum()
  }
}

/// 区块坐标 → 所属 region 坐标（`div_euclid`：负坐标也正确）
fn region_key(cx: i32, cz: i32) -> (i32, i32) {
  (cx.div_euclid(REGION_CHUNKS), cz.div_euclid(REGION_CHUNKS))
}

/// region 坐标 → 分片序号
fn shard_index(rx: i32, rz: i32) -> usize {
  let h = (rx as u32).wrapping_mul(0x9E37_79B1) ^ (rz as u32).wrapping_mul(0x85EB_CA77);
  ((h ^ (h >> 15)) as usize) % SHARDS
}

/// 拿锁：**中毒了也当没毒**（一个 worker panic 不该让整张地图再也读不出来）
fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
  m.lock().unwrap_or_else(|e| e.into_inner())
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
    // `from_key` 是 `key` 的逆（`mc::lod` 的文件靠它还原状态）
    assert_eq!(BlockState::from_key(&a.key()).key(), a.key());
    let plain = BlockState::from_key("minecraft:stone");
    assert_eq!((plain.name.as_str(), plain.props.len()), ("minecraft:stone", 0));
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
    println!(
      "出生点那一列 heap_bytes = {} KB（列缓存预算 {{}} 的取值依据）；缓存读数 (区块数, 读取失败数) = {:?}",
      chunk.heap_bytes() / 1024,
      world.stats()
    );
  }
}
