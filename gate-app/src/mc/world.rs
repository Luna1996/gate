use std::collections::{HashMap, HashSet, VecDeque};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use fastanvil::Region;
use serde::Deserialize;

pub const REGION_CHUNKS: i32 = 32;
pub const SECTION_VOLUME: usize = 4096;
pub const SECTIONS_PER_CHUNK: i32 = 16;

const SHARDS: usize = 32;
const READERS_PER_REGION: usize = 4;
const REGION_FILES_PER_SHARD: usize = 64;

const COL_CACHE_BYTES: usize = 48 * 1024 * 1024;

const PALETTE_ENTRY_BYTES: usize = 96;

#[derive(Debug, Clone, Default, Deserialize)]
pub struct BlockState {
  #[serde(rename = "Name", default)]
  pub name: String,
  #[serde(rename = "Properties", default)]
  pub props: HashMap<String, String>,
}

impl BlockState {
  pub fn short_name(&self) -> &str {
    self.name.strip_prefix("minecraft:").unwrap_or(&self.name)
  }

  pub fn prop(&self, key: &str) -> Option<&str> {
    self.props.get(key).map(String::as_str)
  }

  pub fn key(&self) -> String {
    let mut props: Vec<(&str, &str)> =
      self.props.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
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

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct Section {
  #[serde(default)]
  pub y: i8,
  #[serde(default)]
  pub palette: Vec<BlockState>,
  #[serde(rename = "BlockStates", default)]
  states: Option<fastnbt::LongArray>,
}

impl Section {
  pub fn is_empty_layer(&self) -> bool {
    self.palette.is_empty() || (self.palette.len() == 1 && is_air(&self.palette[0].name))
  }

  pub fn unpack_into(&self, buf: &mut Vec<u16>) {
    buf.clear();
    match &self.states {
      Some(s) => {
        let mut v = fastanvil::expand_blockstates(s, self.palette.len().max(1));
        v.resize(SECTION_VOLUME, 0);
        *buf = v;
      }
      None => buf.resize(SECTION_VOLUME, 0),
    }
  }
}

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

#[derive(Debug, Default, Deserialize)]
pub struct Chunk {
  #[serde(rename = "Sections", default)]
  pub sections: Vec<Section>,
}

impl Chunk {
  pub fn section(&self, section_y: i32) -> Option<&Section> {
    self.sections.iter().find(|s| s.y as i32 == section_y)
  }

  pub fn heap_bytes(&self) -> usize {
    self
      .sections
      .iter()
      .map(|s| s.palette.len() * PALETTE_ENTRY_BYTES + s.states.as_ref().map_or(0, |a| a.len() * 8))
      .sum()
  }
}

struct RegionFile {
  path: PathBuf,
  readers: Box<[Mutex<Option<Region<File>>>]>,
  next: AtomicUsize,
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

  fn read(&self, lx: usize, lz: usize) -> Result<Option<Vec<u8>>, String> {
    if self.dead {
      return Ok(None);
    }
    let i = self.next.fetch_add(1, Ordering::Relaxed) % self.readers.len();
    let mut slot = lock(&self.readers[i]);
    if slot.is_none() {
      match open_region(&self.path, false) {
        Some(r) => *slot = Some(r),
        None => return Ok(None),
      }
    }
    let r = slot.as_mut().expect("刚判过");
    r.read_chunk(lx, lz).map_err(|e| e.to_string())
  }
}

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
        bevy::log::debug!("MC region {} 头部不合法：{e} → 当空", path.display());
        None
      }
    },
  }
}

#[derive(Default)]
struct Shard {
  files: HashMap<(i32, i32), Arc<RegionFile>>,
  file_order: VecDeque<(i32, i32)>,
  cols: HashMap<(i32, i32), (Arc<Chunk>, usize)>,
  col_order: VecDeque<(i32, i32)>,
  bad: HashSet<(i32, i32)>,
  reads: usize,
}

pub struct World {
  dir: PathBuf,
  shards: Box<[Mutex<Shard>]>,
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
    let parsed = match file.read(lx, lz) {
      Ok(Some(bytes)) => match fastnbt::from_bytes::<ChunkRoot>(&bytes) {
        Ok(root) => Some(Arc::new(Chunk { sections: root.level.sections })),
        Err(e) => {
          bevy::log::warn!("MC 区块 ({cx},{cz}) NBT 解析失败：{e} → 当空");
          None
        }
      },
      Ok(None) => None,
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

  fn shard(&self, cx: i32, cz: i32) -> &Mutex<Shard> {
    let (rx, rz) = region_key(cx, cz);
    &self.shards[shard_index(rx, rz)]
  }

  pub fn stats(&self) -> (usize, usize) {
    self.shards.iter().fold((0, 0), |(n, bad), s| {
      let st = lock(s);
      (n + st.cols.len(), bad + st.bad.len())
    })
  }
}

fn region_key(cx: i32, cz: i32) -> (i32, i32) {
  (cx.div_euclid(REGION_CHUNKS), cz.div_euclid(REGION_CHUNKS))
}

fn shard_index(rx: i32, rz: i32) -> usize {
  let h = (rx as u32).wrapping_mul(0x9E37_79B1) ^ (rz as u32).wrapping_mul(0x85EB_CA77);
  ((h ^ (h >> 15)) as usize) % SHARDS
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
  m.lock().unwrap_or_else(|e| e.into_inner())
}

pub fn is_air(name: &str) -> bool {
  matches!(name, "minecraft:air" | "minecraft:cave_air" | "minecraft:void_air")
}

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

pub fn looks_like_world(dir: &Path) -> bool {
  dir.join("region").is_dir() || dir.join("level.dat").is_file()
}
