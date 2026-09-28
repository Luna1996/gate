//! **离线粗粒度世界**（`docs/mc_map.md` §8.4）：把整张图预计算成一份 section 级摘要，远端直接内存采样。
//!
//! # 为什么
//!
//! 远场按"格中心那一列"读 256 个 chunk 列（[`super::summary`]），实测 L2 **295 ms/块** —— 3 个 worker
//! 也只有 ~10 块/s ⇒ 铺满 ±2.6 km 要几分钟。而那 256 次列读换来的东西很小：每节一个代表色 + 实体数。
//! 把这份东西**离线算一次**存盘（本模块），远场一块就退化成 4096 次内存查表（≈1 ms/块）。
//!
//! # 口径（与 `summary` 同源，才谈得上替代）
//!
//! 一格 = 一个 section（16 方块）：`solid` = 该节非空气方块数（0..4096）、`rep` = **最上面那一层
//! 有料的方块**里的多数非空气状态（表面色，见 [`voxel::rep_of_surface`]；整节多数色会把"一层水/草
//! 压厚基座"染成基座 ⇒ 与近场成片色差）。实体占比不足 [`SOLID_MIN_PERMILLE`] 的节**不写**（与
//! `summary` 一致：远场只求轮廓，不然街道 / 空地会被糊成实心）。
//!
//! **v2 起**每节还存一份 `4³` **细格**（[`FINE_CELL`] 方块一格，RLE）—— 它就是 `summary::Sec::fine`
//! 的落盘形式：细格 = 格内 64 个方块的多数非空气状态，占比不足 [`SOLID_MIN_PERMILLE`] 的细格算空。
//! 有了它，**细档（4 方块格）也能走这份文件**，L1 不再按需读 Anvil（实测 ~174 ms/块 ⇒ 0.2 ms/块）。
//! 细格还让"节该不该留"这个门槛变细：整节占比不足但仍有个别细格够实体的节**要留**
//! （薄墙就是这个形状），否则 L1 会凭空少一块几何。
//!
//! 两处口径差（都是**替代**关系、不是叠加关系，同一帧里不会混用；实测见 `mc::tests::real_map_build_lod`）：
//! - `rep`（**表面**色：最上面那一层有料的方块里的多数）按**方块**取（`summary` 先按 `4³` 格折一层
//!   再折一层），平手取节内调色板下标小者（`summary` 按全局槽号）⇒ 约 7% 的格会挑到不同的色（多为邻近色）；
//! - 一格的边长 > 一个节时，本文件**采满格内 `n³` 个节**（再取最上面那一层有料的节的色），
//!   而 `summary` 只采格中心那一列（当初为省列读）⇒ 越粗的级差异越大，但那是**更准**：格子不该因为
//!   街区偏心就被采成空。
//!
//! 存的是**方块状态的规范键**（[`super::world::BlockState::key`]），不是调色板槽号 —— 槽号是每次运行
//! worker 重新认领的（[`super::material::Pool`]），而键是纯数据；装载时按键重查 `plan_for` 得到代表色。
//! 于是构建器**不需要资产与调色板**，只要存档。
//!
//! # 文件
//!
//! `data/lod/<世界名>.lod`：头（版本 / 地图指纹 / 列范围）+ 名字表 + 逐列 RLE 的节摘要。
//! 全图 1088² 列里只有约 1/3 有料 ⇒ RLE 把体积压到十几 MB（平铺的 18.9 M 个节要 75 MB）。

use std::collections::HashMap;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Mutex, RwLock};

use glam::IVec3;
use rayon::prelude::*;

use super::summary::SOLID_MIN_PERMILLE;
use super::voxel;
use super::world::{self, World, SECTIONS_PER_CHUNK, SECTION_VOLUME};
use gate_voxel::PaletteId;

/// 文件魔数（前 8 字节）
const MAGIC: &[u8; 8] = b"GATELOD1";
/// 格式版本（布局变了就 +1：装载器按它判"文件作废"）。v2 = 每节多一份 `4³` 细格。
const VERSION: u32 = 2;
/// 一格的方块边长：**正好一个 section** —— 格的边长是它的整数倍时，每一格只用到本文件里的整数个节。
pub const CELL: i32 = 16;
/// **细格**的方块边长（每轴）：一个 section 里 `4³` 个细格 —— 细档（L1 的 1.28 m 格）用这一档。
pub const FINE_CELL: i32 = 4;
/// 每轴细格数与每节的细格总数
const FINE_PER_AXIS: i32 = CELL / FINE_CELL;
const FINE_TOTAL: usize = (FINE_PER_AXIS * FINE_PER_AXIS * FINE_PER_AXIS) as usize;
/// 细格 RLE 里"空气"的名字下标（`names` 不会到这个量级：实测 300 上下）
const FINE_AIR: u16 = u16::MAX;
/// 每处理多少列叫一次进度（构建时日志的粒度）
const PROGRESS_COLS: usize = 4096;

/// **采样口径 + 文件布局的版本号**：只要改变"一格怎么算有料 / 细格粒度 / 文件布局"就必须 +1。
///
/// WHY 必须进 [`stamp`]：`stamp` 只认地图指纹（路径 + `level.dat`），**改动代码不会让它变** ⇒ 旧文件
/// 会被静默复用，画面继续按旧口径渲染 —— "重建 LOD 看起来毫无变化"的根源（L1 的格边长 = `FINE_CELL`
/// = 4，`View::supports(4)` 为真 ⇒ L1 一律读文件，不走 `summary`）。带上版本号后，口径一变旧文件立即
/// 作废（回落到 `summary`，等重建）。
///
/// v2 = 每节带 `4³` 细格（本文件头）；v3 = `SOLID_MIN_PERMILLE` 125‰ → 16‰；
/// v4 = 代表色改取**表面**（最上面那一层有料的方块/细格），不再是整节多数色（见 [`voxel::rep_of_surface`]）。
const FORMAT_TAG: u64 = 4;

/// 地图指纹：路径 + `level.dat` 的大小与 mtime + 采样口径版本（[`FORMAT_TAG`]）。
/// 地图一变（换存档 / 改过）**或口径变** ⇒ 旧文件作废。
pub fn stamp(map_dir: &Path) -> u64 {
  let mut h = std::collections::hash_map::DefaultHasher::new();
  FORMAT_TAG.hash(&mut h);
  map_dir.hash(&mut h);
  if let Ok(m) = fs::metadata(map_dir.join("level.dat")) {
    m.len().hash(&mut h);
    if let Ok(t) = m.modified() {
      t.hash(&mut h);
    }
  }
  h.finish()
}

/// LOD 文件路径：`<data>/lod/<世界名>.lod`（`data_dir` 与 `config.toml` 同一口径）。
pub fn path_for(world_name: &str) -> PathBuf {
  gate_render::data_dir().join("lod").join(format!("{world_name}.lod"))
}

// ---------------------------------------------------------------------------
// 文件读写
// ---------------------------------------------------------------------------

/// 一份 LOD 文件的内容（名字表 + 逐列 RLE）。**名字还没解析成调色板槽号**，见 [`View`]。
pub struct File {
  /// 地图指纹（[`stamp`]）
  pub stamp: u64,
  /// 覆盖范围的起点（chunk 坐标）
  pub col_min: (i32, i32),
  /// 覆盖范围的尺寸（chunk 数）
  pub dims: (i32, i32),
  /// 竖向上的节数（1.17 世界高 256 ⇒ 16）
  pub sec_y: i32,
  /// 节摘要里出现的方块状态键（`rep` 与细格的名字下标都指向它）
  pub names: Vec<String>,
  /// 每列的非空节位掩码（`bit y`）
  mask: Vec<u16>,
  /// 每列条目区在 [`Self::body`] 里的字节偏移
  offs: Vec<u32>,
  /// 条目区：每个非空节 4 B = `u16 名字下标 + u16 实体数`
  body: Vec<u8>,
  /// 每个非空节的细格 RLE 在 [`Self::fine_body`] 里的字节偏移（顺序 = 掩码里置位的顺序）
  fine_offs: Vec<u32>,
  /// 细格区：每节一串 `(u16 名字下标, u8 连续个数)`，长度和恰为 [`FINE_TOTAL`]
  fine_body: Vec<u8>,
}

impl File {
  /// 这一列的条目（按 `y` 升序）在 body 里的起点
  fn col_off(&self, i: usize) -> usize {
    self.offs[i] as usize
  }

  /// 堆占用（日志口径）
  pub fn heap_bytes(&self) -> usize {
    self.body.len()
      + self.fine_body.len()
      + self.mask.len() * 2
      + self.offs.len() * 4
      + self.fine_offs.len() * 4
      + self.names.iter().map(|n| n.len() + 24).sum::<usize>()
  }

  /// 读一个 LOD 文件（`Err` 带原因：缺文件 / 魔数不符 / 版本不符 / 截断）
  pub fn read(path: &Path) -> Result<Self, String> {
    let raw = fs::read(path).map_err(|e| format!("读 {} 失败：{e}", path.display()))?;
    let mut c = Cursor { b: &raw, i: 0 };
    if c.take(8)? != MAGIC.as_slice() {
      return Err("魔数不符".into());
    }
    let version = c.u32()?;
    if version != VERSION {
      return Err(format!("版本 {version} ≠ {VERSION}（布局变了，重新构建）"));
    }
    let stamp = c.u64()?;
    let col_min = (c.i32()?, c.i32()?);
    let dims = (c.i32()?, c.i32()?);
    let sec_y = c.i32()?;
    let name_count = c.u32()? as usize;
    let mut names = Vec::with_capacity(name_count);
    for _ in 0..name_count {
      let n = c.u32()? as usize;
      names.push(String::from_utf8(c.take(n)?.to_vec()).map_err(|e| format!("名字不是 UTF-8：{e}"))?);
    }
    if dims.0 <= 0 || dims.1 <= 0 || !(1..=SECTIONS_PER_CHUNK).contains(&sec_y) {
      return Err(format!("范围不合法：{}×{} 列、{sec_y} 节", dims.0, dims.1));
    }
    let cols = dims.0 as usize * dims.1 as usize;
    let mut mask = Vec::with_capacity(cols);
    for _ in 0..cols {
      mask.push(c.u16()?);
    }
    let mut offs = Vec::with_capacity(cols);
    for _ in 0..cols {
      offs.push(c.u32()?);
    }
    let body_len = c.u32()? as usize;
    let body = c.take(body_len)?.to_vec();
    // 条目数必须与掩码逐位吻合（文件被截断 / 手改过时当场现形，而不是采样时越界）
    let want: u32 = mask.iter().map(|m| m.count_ones()).sum();
    if body.len() != want as usize * 4 {
      return Err(format!("条目区 {} B ≠ 掩码推出来的 {} B", body.len(), want * 4));
    }
    let mut fine_offs = Vec::with_capacity(want as usize);
    for _ in 0..want {
      fine_offs.push(c.u32()?);
    }
    let fine_len = c.u32()? as usize;
    let fine_body = c.take(fine_len)?.to_vec();
    // 细格区的自洽性：从 0 起逐节走一遍，每节的 RLE 必须正好铺满 `FINE_TOTAL` 格、且与下一条偏移吻合
    let mut at = 0usize;
    for k in 0..want as usize {
      if fine_offs[k] as usize != at {
        return Err(format!("细格偏移 [{k}] = {} ≠ 走到 {}（区间不连续）", fine_offs[k], at));
      }
      let mut cells = 0usize;
      while cells < FINE_TOTAL {
        if at + 3 > fine_body.len() {
          return Err("细格区截断".into());
        }
        let run = fine_body[at + 2] as usize;
        if run == 0 || cells + run > FINE_TOTAL {
          return Err(format!("细格 RLE 长度非法：{run}（已 {cells}/{FINE_TOTAL}）"));
        }
        cells += run;
        at += 3;
      }
    }
    if at != fine_body.len() {
      return Err(format!("细格区多出 {} B（掩码只推到 {at}）", fine_body.len() - at));
    }
    Ok(Self { stamp, col_min, dims, sec_y, names, mask, offs, body, fine_offs, fine_body })
  }

  /// 写一个 LOD 文件（目录不存在则建）；返回写出的字节数
  pub fn write(&self, path: &Path) -> Result<usize, String> {
    let mut out: Vec<u8> = Vec::with_capacity(self.heap_bytes() + 64);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&self.stamp.to_le_bytes());
    out.extend_from_slice(&self.col_min.0.to_le_bytes());
    out.extend_from_slice(&self.col_min.1.to_le_bytes());
    out.extend_from_slice(&self.dims.0.to_le_bytes());
    out.extend_from_slice(&self.dims.1.to_le_bytes());
    out.extend_from_slice(&self.sec_y.to_le_bytes());
    out.extend_from_slice(&(self.names.len() as u32).to_le_bytes());
    for n in &self.names {
      out.extend_from_slice(&(n.len() as u32).to_le_bytes());
      out.extend_from_slice(n.as_bytes());
    }
    for m in &self.mask {
      out.extend_from_slice(&m.to_le_bytes());
    }
    for o in &self.offs {
      out.extend_from_slice(&o.to_le_bytes());
    }
    out.extend_from_slice(&(self.body.len() as u32).to_le_bytes());
    out.extend_from_slice(&self.body);
    for o in &self.fine_offs {
      out.extend_from_slice(&o.to_le_bytes());
    }
    out.extend_from_slice(&(self.fine_body.len() as u32).to_le_bytes());
    out.extend_from_slice(&self.fine_body);
    if let Some(dir) = path.parent()
      && let Err(e) = fs::create_dir_all(dir)
    {
      return Err(format!("建目录 {} 失败：{e}", dir.display()));
    }
    fs::write(path, &out).map_err(|e| format!("写 {} 失败：{e}", path.display()))?;
    Ok(out.len())
  }
}

/// 读游标（越界当场报错，不做静默补零）
struct Cursor<'a> {
  b: &'a [u8],
  i: usize,
}

impl<'a> Cursor<'a> {
  fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
    let end = self.i.checked_add(n).ok_or("长度溢出")?;
    if end > self.b.len() {
      return Err(format!("文件截断：要 {} B，只剩 {} B", n, self.b.len() - self.i));
    }
    let s = &self.b[self.i..end];
    self.i = end;
    Ok(s)
  }
  fn u16(&mut self) -> Result<u16, String> {
    Ok(u16::from_le_bytes(self.take(2)?.try_into().expect("刚好 2 B")))
  }
  fn u32(&mut self) -> Result<u32, String> {
    Ok(u32::from_le_bytes(self.take(4)?.try_into().expect("刚好 4 B")))
  }
  fn u64(&mut self) -> Result<u64, String> {
    Ok(u64::from_le_bytes(self.take(8)?.try_into().expect("刚好 8 B")))
  }
  fn i32(&mut self) -> Result<i32, String> {
    Ok(i32::from_le_bytes(self.take(4)?.try_into().expect("刚好 4 B")))
  }
}

// ---------------------------------------------------------------------------
// 采样视图
// ---------------------------------------------------------------------------

/// 把 [`File`] 的名字解析成**本次运行**的调色板代表色之后的采样视图。
pub struct View {
  f: File,
  /// 名字表 → 代表色（`None` = 这个名字没认领到槽：缺资产 / 解析不出来）
  rep: Vec<Option<PaletteId>>,
}

impl View {
  /// `resolve` = 方块状态键 → 代表色（`mc::source::McCITY` 用 `plan_for` 提供）
  pub fn new(f: File, resolve: impl Fn(&str) -> Option<PaletteId>) -> Self {
    let rep = f.names.iter().map(|n| resolve(n)).collect();
    Self { f, rep }
  }

  pub fn heap_bytes(&self) -> usize {
    self.f.heap_bytes() + self.rep.len() * std::mem::size_of::<Option<PaletteId>>()
  }

  /// 名字表里没认领到调色板槽的个数（缺资产 / 解析不出来 ⇒ 那些节算空）
  pub fn missing(&self) -> usize {
    self.rep.iter().filter(|r| r.is_none()).count()
  }

  /// 名字表条数
  pub fn names(&self) -> usize {
    self.f.names.len()
  }

  /// 文件覆盖的列数
  pub fn cols(&self) -> usize {
    self.f.dims.0 as usize * self.f.dims.1 as usize
  }

  /// 写进去的非空节数
  pub fn sections(&self) -> u32 {
    self.f.mask.iter().map(|m| m.count_ones()).sum()
  }

  /// 这份文件能不能服务"格 = `cell_blocks` 方块"的采样口径。
  ///
  /// 文件里有两档数据：每节的**整体**摘要（`rep`/`solid`，任何 ≥ 一节的格都能用）与每节的 `4³`
  /// **细格** ⇒ 能服务的格边长是 `FINE_CELL`（4）与 `CELL`（16）的整数倍。别的档（例如 8）没有数据，
  /// 调用方（[`super::source`]）应当回退 Anvil。
  pub fn supports(&self, cell_blocks: i32) -> bool {
    cell_blocks == FINE_CELL || (cell_blocks >= CELL && cell_blocks % CELL == 0)
  }

  /// 一节（chunk 坐标 + 节号）→ `(列下标, 本列内的序号, 整份文件内的序号)`；`None` = 越界 / 空节。
  ///
  /// 全序号 O(1)：条目区每条恒 4 B ⇒ `offs[col] / 4` 就是该列之前有多少条。
  fn entry(&self, cx: i32, y: i32, cz: i32) -> Option<(usize, usize, usize)> {
    if !(0..self.f.sec_y).contains(&y) {
      return None;
    }
    let (ix, iz) = (cx - self.f.col_min.0, cz - self.f.col_min.1);
    if ix < 0 || iz < 0 || ix >= self.f.dims.0 || iz >= self.f.dims.1 {
      return None;
    }
    let col = (iz * self.f.dims.0 + ix) as usize;
    let mask = self.f.mask[col];
    if mask & (1 << y) == 0 {
      return None;
    }
    let k = (mask & ((1u16 << y) - 1)).count_ones() as usize;
    Some((col, k, self.f.col_off(col) / 4 + k))
  }

  /// 一节（chunk 坐标 + 节号）的 `(代表色, 实体数)`；`None` = 越界 / 空节 / 名字没解析出来
  fn section(&self, cx: i32, y: i32, cz: i32) -> Option<(PaletteId, u32)> {
    let (col, k, _) = self.entry(cx, y, cz)?;
    let off = self.f.col_off(col) + k * 4;
    let name = u16::from_le_bytes([self.f.body[off], self.f.body[off + 1]]) as usize;
    let solid = u16::from_le_bytes([self.f.body[off + 2], self.f.body[off + 3]]) as u32;
    self.rep.get(name).copied().flatten().map(|r| (r, solid))
  }

  /// 一节的第 `l` 个细格（`l` = `x + 4·z + 16·y`，见 [`FINE_PER_AXIS`]）的代表色；
  /// `None` = 该格算空 / 名字没解析出来。
  fn fine(&self, k_global: usize, l: usize) -> Option<PaletteId> {
    let mut at = *self.f.fine_offs.get(k_global)? as usize;
    let mut seen = 0usize;
    loop {
      let name = u16::from_le_bytes([self.f.fine_body[at], self.f.fine_body[at + 1]]);
      let run = self.f.fine_body[at + 2] as usize;
      if l < seen + run {
        if name == FINE_AIR {
          return None;
        }
        return self.rep.get(name as usize).copied().flatten();
      }
      seen += run;
      at += 3;
    }
  }

  /// `cell_blocks` 方块边长的一格、格原点 `cell`（**方块**坐标）处的代表色；`None` = 这一格算空。
  ///
  /// 口径与 `summary::cell` 对齐（同样的实体占比门槛），但一格含多个节时**采满格内的 `n³` 个节**
  /// 再取多数色 —— `summary` 那里只采格中心那一列（当初是为了省列读），这里查表几乎免费，没有理由
  /// 再丢信息。
  pub fn cell(&self, cell: IVec3, cell_blocks: i32) -> Option<PaletteId> {
    if cell_blocks == FINE_CELL {
      // 细档：取所在节里的那一个细格（`cell` 对齐 `FINE_CELL` ⇒ 一次除法定位）
      let sec = cell.div_euclid(IVec3::splat(CELL));
      let (_, _, k) = self.entry(sec.x, sec.y, sec.z)?;
      let l = cell.rem_euclid(IVec3::splat(CELL)) / FINE_CELL;
      let idx = (l.x + FINE_PER_AXIS * l.z + FINE_PER_AXIS * FINE_PER_AXIS * l.y) as usize;
      return self.fine(k, idx);
    }
    let n = cell_blocks / CELL;
    if n <= 0 {
      return None; // 比一个节还细、又不是细格的档（例如 8）本文件表达不了 ⇒ 调用方应该走 Anvil
    }
    let base = cell.div_euclid(IVec3::splat(CELL));
    // **节只要存在就算有料** —— 写侧（`build_lod`）已经按"整节实体够 **或** 任一细格够"筛过一遍
    // （见文件里 `fine_any` 那段），所以"存在"本身就是"有料"。这里**不能再**拿整节实体占比判一次：
    // 那会把"只有薄墙/一层楼板、整节实体 < `SOLID_MIN_PERMILLE`"的节（典型的**高楼**：16 方块见方
    // 的一节里仅一层楼板 ≈ 6%）重新丢掉 ⇒ 楼在 5.12 m 档（L2，格 = 1 节）变空，且随距离换档来回翻
    // ⇒ 用户看到的"高楼中空 + 距离微变就突变"。
    if n == 1 {
      let (rep, _) = self.section(base.x, base.y, base.z)?;
      return Some(rep);
    }
    // 取**最上面那一层有料的节**（不是这几节的多数色）：与细档/整节档同一条"表面"口径
    // （`voxel::rep_of_surface`，「每层」= 本格一层的节数 `n²`）—— 否则最粗的格又会退回"把水面染成河床"。
    // 缺的节填空气占位，保持 y-major 的排布（`rep_of_surface` 按层切分）。
    let per_layer = (n * n) as usize;
    let mut reps: Vec<PaletteId> = Vec::with_capacity(per_layer * n as usize);
    for dy in 0..n {
      for dz in 0..n {
        for dx in 0..n {
          reps.push(self.section(base.x + dx, base.y + dy, base.z + dz).map_or(PaletteId::AIR, |(r, _)| r));
        }
      }
    }
    voxel::rep_of_surface(&reps, per_layer)
  }
}

// ---------------------------------------------------------------------------
// 构建
// ---------------------------------------------------------------------------

/// 构建统计（日志与验收口径）
#[derive(Debug, Clone, Copy)]
pub struct BuildStats {
  /// 扫过的列数
  pub cols: usize,
  /// 写进去的非空节数
  pub sections: usize,
  /// 名字表条数
  pub names: usize,
  /// 文件字节数
  pub bytes: usize,
  /// 墙钟秒数
  pub secs: f64,
}

/// 把整张图预计算成一份 LOD 文件。**只读存档**（不碰资产 / 调色板：存的是方块状态键）。
///
/// 覆盖范围由 `region/` 里**实际存在的 region 文件**决定（不扫整张图的包围盒：Greenfield 的包围盒
/// 里有 2/3 是空 region）。`cancel` 置位即停（返回 `Err`）；`progress` 每 [`PROGRESS_COLS`] 列叫一次。
/// 返回 `(统计, 刚写出的内容)` —— 调用方拿后者直接热装，不必再读一遍文件。
pub fn build(
  world: &World,
  out: &Path,
  cancel: &AtomicBool,
  progress: impl Fn(usize, usize) + Sync,
) -> Result<(BuildStats, File), String> {
  let t0 = std::time::Instant::now();
  let (col_min, dims, regions) = region_bounds(world.dir())?;
  let cols = (dims.0 * dims.1) as usize;
  bevy::log::info!(
    "LOD 构建：{} 个 region、{}×{} 列（chunk {}..{}）→ {}",
    regions,
    dims.0,
    dims.1,
    col_min.0,
    col_min.0 + dims.0,
    out.display()
  );
  let names = Mutex::new(Interner::default());
  let done = AtomicUsize::new(0);
  // 逐列并行：`world.chunk` 自己带分片锁与句柄池（见 `mc::world` 的并发说明）
  let per_col: Vec<Option<Vec<SecCol>>> = (0..cols)
    .into_par_iter()
    .map(|i| {
      if cancel.load(Ordering::Relaxed) {
        return None;
      }
      let cx = col_min.0 + (i as i32) % dims.0;
      let cz = col_min.1 + (i as i32) / dims.0;
      let out = column(world, &names, cx, cz);
      let n = done.fetch_add(1, Ordering::Relaxed) + 1;
      if n % PROGRESS_COLS == 0 {
        progress(n, cols);
      }
      out
    })
    .collect();
  if cancel.load(Ordering::Relaxed) {
    return Err("已取消".into());
  }
  let names = names.into_inner().unwrap_or_else(|e| e.into_inner());
  let mut mask = Vec::with_capacity(cols);
  let mut offs = Vec::with_capacity(cols);
  let mut body: Vec<u8> = Vec::new();
  let mut fine_offs: Vec<u32> = Vec::new();
  let mut fine_body: Vec<u8> = Vec::new();
  let mut sections = 0usize;
  for col in &per_col {
    let mut m = 0u16;
    if let Some(items) = col {
      for it in items {
        m |= 1 << it.y;
      }
    }
    mask.push(m);
    offs.push(body.len() as u32);
    if let Some(items) = col {
      for it in items {
        body.extend_from_slice(&it.rep.to_le_bytes());
        body.extend_from_slice(&it.solid.to_le_bytes());
        fine_offs.push(fine_body.len() as u32);
        for &(name, len) in &it.fine {
          fine_body.extend_from_slice(&name.to_le_bytes());
          fine_body.push(len);
        }
        sections += 1;
      }
    }
  }
  let f = File {
    stamp: stamp(world.dir()),
    col_min,
    dims,
    sec_y: SECTIONS_PER_CHUNK,
    names: names.list,
    mask,
    offs,
    body,
    fine_offs,
    fine_body,
  };
  let bytes = f.write(out)?;
  let stats = BuildStats { cols, sections, names: f.names.len(), bytes, secs: t0.elapsed().as_secs_f64() };
  Ok((stats, f))
}

/// `region/` 里实际存在的文件 → `(列起点, 列尺寸, region 数)`
fn region_bounds(dir: &Path) -> Result<((i32, i32), (i32, i32), usize), String> {
  let rdir = dir.join("region");
  let rd = fs::read_dir(&rdir).map_err(|e| format!("读 {} 失败：{e}", rdir.display()))?;
  let (mut lo, mut hi, mut n) = ((i32::MAX, i32::MAX), (i32::MIN, i32::MIN), 0usize);
  for e in rd.flatten() {
    let name = e.file_name().to_string_lossy().into_owned();
    let Some(t) = name.strip_prefix("r.").and_then(|s| s.strip_suffix(".mca")) else { continue };
    let mut it = t.split('.');
    let (Some(x), Some(z)) = (it.next().and_then(|s| s.parse::<i32>().ok()), it.next().and_then(|s| s.parse::<i32>().ok()))
    else {
      continue;
    };
    lo = (lo.0.min(x), lo.1.min(z));
    hi = (hi.0.max(x), hi.1.max(z));
    n += 1;
  }
  if n == 0 {
    return Err(format!("{} 里没有 r.X.Z.mca", rdir.display()));
  }
  let dims = ((hi.0 - lo.0 + 1) * world::REGION_CHUNKS, (hi.1 - lo.1 + 1) * world::REGION_CHUNKS);
  Ok(((lo.0 * world::REGION_CHUNKS, lo.1 * world::REGION_CHUNKS), dims, n))
}

/// 一列里一个节的中间形态（构建期）
struct SecCol {
  /// 节号（0..16）
  y: u8,
  /// **最上面那一层有料的方块**里的多数非空气状态的名字下标（表面色，见 [`voxel::rep_of_surface`]）
  rep: u16,
  /// 整节非空气方块数
  solid: u16,
  /// 细格 RLE：`(名字下标, 连续个数)`，[`FINE_AIR`] = 空；长度和恰为 [`FINE_TOTAL`]
  fine: Vec<(u16, u8)>,
}

/// 一列的节摘要（`y` 升序）；没有内容 → `None`
fn column(world: &World, names: &Mutex<Interner>, cx: i32, cz: i32) -> Option<Vec<SecCol>> {
  let chunk = world.chunk(cx, cz)?;
  let mut buf: Vec<u16> = Vec::new();
  let mut counts: Vec<u32> = Vec::new();
  let mut out: Vec<SecCol> = Vec::new();
  for y in 0..SECTIONS_PER_CHUNK {
    let Some(sec) = chunk.section(y).filter(|s| !s.is_empty_layer()) else { continue };
    sec.unpack_into(&mut buf);
    // 本节调色板一次认领到底（细格只查表，不再逐个加锁）：名字表 ~300 条，多认几个状态没关系
    let ids: Vec<u16> = {
      let mut g = names.lock().unwrap_or_else(|e| e.into_inner());
      sec.palette
        .iter()
        .map(|st| if world::is_air(&st.name) { FINE_AIR } else { g.intern(&st.key()) })
        .collect()
    };
    let fine = fine_cells(&ids, &buf, &mut counts);
    // 节的代表色 = **细格粒度**的表面色，与 `summary::build` 的 `rep_of_surface(&groups, 16)`
    // **逐字同一条规则**：一层 = `4×4` 个细格（RLE 序 `x + 4·z + 16·y` ⇒ 第 `ly` 层 = `[ly·16, ..)`），
    // 候选层 = 从高到低第一个"够实"（实体 ≥ 半层 = 16 个细格里 8 个非空）的层；色取该层**细格代表色**
    // 的多数（每个细格一票，不按方块数加权）。没有这样的层 ⇒ 退回 64 个细格的多数色。
    // 整格多数色会把"一层水 / 草皮压厚基座"染成基座（水面 → 沙/石、草地 → 土）。
    //
    // CONSTRAINT: 必须与 `summary`（回落 Anvil 时走的那条路）**完全一致** —— 同一格的**替代**实现，
    // 不一致 = 远场在"有 LOD"与"回落 Anvil"两种状态下颜色不同。⚠️ 门判在**细格**粒度上：改成
    // "方块粒度"或"按方块数加权"实测把 L2 的异色率从 10% 顶到 17%（`real_map_build_lod` 逐格比色）。
    let mut cells = [FINE_AIR; FINE_TOTAL];
    {
      let mut i = 0usize;
      for &(name, len) in &fine {
        for _ in 0..len {
          cells[i] = name;
          i += 1;
        }
      }
      debug_assert_eq!(i, FINE_TOTAL, "细格 RLE 的长度和必须恰好是 64");
    }
    let layer_len = (FINE_PER_AXIS * FINE_PER_AXIS) as usize;
    let mut rep = FINE_AIR;
    for ly in (0..FINE_PER_AXIS as usize).rev() {
      let layer = &cells[ly * layer_len..(ly + 1) * layer_len];
      let occupied = layer.iter().filter(|&&n| n != FINE_AIR).count();
      if occupied * 2 < layer_len {
        continue;
      }
      rep = majority_name(layer);
      break;
    }
    if rep == FINE_AIR {
      rep = majority_name(&cells);
    }
    // 整节的实体数：写不写这一节看它（与 `summary` 的节门槛同源）
    let solid = buf
      .iter()
      .filter(|&&pi| sec.palette.get(pi as usize).is_some_and(|st| !world::is_air(&st.name)))
      .count() as u16;
    // 留不留这一节：整节够实体**或**有任何一个细格够实体（薄墙就属于后者 —— 整节 1/8 的门槛会把它
    // 判空，而它的细格是实的；细档靠的就是这些细格）
    let fine_any = fine.iter().any(|&(n, _)| n != FINE_AIR);
    if !fine_any && solid as u32 * 1000 < SOLID_MIN_PERMILLE * SECTION_VOLUME as u32 {
      continue;
    }
    out.push(SecCol { y: y as u8, rep, solid, fine });
  }
  (!out.is_empty()).then_some(out)
}

/// 计数数组里计数最大的那一项（平手取下标小者）；全零 → `None`
fn majority(counts: &[u32]) -> Option<(u32, usize)> {
  let mut best: Option<(u32, usize)> = None;
  for (i, &c) in counts.iter().enumerate() {
    if c == 0 {
      continue;
    }
    if best.is_none_or(|(bc, bi)| c > bc || (c == bc && i < bi)) {
      best = Some((c, i));
    }
  }
  best
}

/// `cells`（细格代表色，[`FINE_AIR`] = 空）里出现最多的那个名字；平手取名字下标小者；全空 → [`FINE_AIR`]。
/// 与 [`voxel::rep_of`] 的"平手取下标小者"同一条规则（那边的下标是全局槽号、这边是名字下标）。
fn majority_name(cells: &[u16]) -> u16 {
  let (mut best, mut best_n) = (FINE_AIR, 0usize);
  for &n in cells.iter().filter(|&&n| n != FINE_AIR) {
    let c = cells.iter().filter(|&&m| m == n).count();
    if c > best_n || (c == best_n && n < best) {
      best_n = c;
      best = n;
    }
  }
  best
}

/// 一节的 `4³` 细格（RLE，顺序 = `x + 4·z + 16·y`，与 `View::fine` 的索引同一套）。
///
/// 每格 = 该格的**表面层**色：**最上面那一层实体 ≥ 半层**（4×4 = 16 个方块里 ≥ 8 个非空气）
/// 的层里的多数非空气状态（`ids` = 本节调色板 → 名字下标，空气项 = [`FINE_AIR`]）；没有这样的层
/// （薄墙 / 竖直面）⇒ 退回**整格多数色**。实体占比不足 [`SOLID_MIN_PERMILLE`] ⇒ [`FINE_AIR`]。
///
/// CONSTRAINT: 与 `summary::build` 的 `group` + [`voxel::rep_of_surface`] **逐字同一条规则** ——
/// 同一格的**替代**实现，不一致 = 远场在"有 LOD"与"回落 Anvil"两种状态下颜色不同。
/// 回归在 `mc::tests::real_map_build_lod`（逐格比色）。`tally` 是复用的临时缓冲。
fn fine_cells(ids: &[u16], buf: &[u16], tally: &mut Vec<u32>) -> Vec<(u16, u8)> {
  // 本节块下标：`i = y·256 + z·16 + x`
  let pi_at = |bx: i32, by: i32, bz: i32| -> usize { buf[(by * 256 + bz * 16 + bx) as usize] as usize };
  let solid_at = |pi: usize| ids.get(pi).is_some_and(|&id| id != FINE_AIR);
  let mut runs: Vec<(u16, u8)> = Vec::with_capacity(8);
  for gy in 0..FINE_PER_AXIS {
    for gz in 0..FINE_PER_AXIS {
      for gx in 0..FINE_PER_AXIS {
        // 一遍扫完 `4³` 个方块：整格的实体数（判空口径不变 —— 薄墙 / 一层楼板仍能过门槛）
        let mut solid = 0u32;
        for y in 0..4 {
          for z in 0..4 {
            for x in 0..4 {
              if solid_at(pi_at(gx * 4 + x, gy * 4 + y, gz * 4 + z)) {
                solid += 1;
              }
            }
          }
        }
        let mut rep = FINE_AIR;
        if solid * 1000 >= SOLID_MIN_PERMILLE * 64 {
          // 找**最上面那一层"够实"**的局部 y（≥ 半层 = 16 个方块里 8 个）
          let mut top: Option<i32> = None;
          for y in (0..4).rev() {
            let mut n = 0u32;
            for z in 0..4 {
              for x in 0..4 {
                if solid_at(pi_at(gx * 4 + x, gy * 4 + y, gz * 4 + z)) {
                  n += 1;
                }
              }
            }
            if n * 2 >= 16 {
              top = Some(y);
              break;
            }
          }
          tally.clear();
          tally.resize(ids.len(), 0);
          match top {
            Some(y) => {
              for z in 0..4 {
                for x in 0..4 {
                  let pi = pi_at(gx * 4 + x, gy * 4 + y, gz * 4 + z);
                  if solid_at(pi) {
                    tally[pi] += 1;
                  }
                }
              }
            }
            None => {
              for y in 0..4 {
                for z in 0..4 {
                  for x in 0..4 {
                    let pi = pi_at(gx * 4 + x, gy * 4 + y, gz * 4 + z);
                    if solid_at(pi) {
                      tally[pi] += 1;
                    }
                  }
                }
              }
            }
          }
          rep = majority(tally).map_or(FINE_AIR, |(_, bi)| ids[bi]);
        }
        match runs.last_mut() {
          Some((n, len)) if *n == rep && *len < u8::MAX => *len += 1,
          _ => runs.push((rep, 1)),
        }
      }
    }
  }
  debug_assert_eq!(runs.iter().map(|&(_, l)| l as usize).sum::<usize>(), FINE_TOTAL);
  runs
}

/// 名字表（构建期把方块状态键收敛成下标）
#[derive(Default)]
struct Interner {
  map: HashMap<String, u16>,
  list: Vec<String>,
}

impl Interner {
  fn intern(&mut self, key: &str) -> u16 {
    if let Some(&i) = self.map.get(key) {
      return i;
    }
    let i = self.list.len() as u16;
    self.list.push(key.to_string());
    self.map.insert(key.to_string(), i);
    i
  }
}

/// [`load`] 拿不到 LOD 的三种结局。**三种的处置相同**（远场回落 `summary` + 自动重建，见
/// `mc::build`），分开只为了让归因日志说清"为什么"。
pub enum Unavailable {
  /// 文件不在（没建过，或 `data/` 被清过 —— 它不在版本控制里）
  Missing,
  /// 读失败 / 损坏 / **版本不符**（`VERSION` 或 [`FORMAT_TAG`] 变过 ⇒ 文件是按旧口径算的）
  Bad(String),
  /// 指纹不符：换了地图，或存档被改过 —— LOD 是**从存档算出来的**，存档变了它就不代表任何东西了
  Stale,
}

impl Unavailable {
  /// 一行"为什么拿不到"（含路径）：调用点的归因日志用
  pub fn reason(&self, path: &Path) -> String {
    let p = path.display();
    match self {
      Self::Missing => format!("无 {p}"),
      Self::Bad(e) => format!("{p} 不可用：{e}"),
      Self::Stale => format!("{p} 是别的地图 / 存档改过了 → 作废"),
    }
  }
}

/// 读一个文件并校验（魔数 / 版本 / 指纹）；`Err` = 不可用（原因由调用点落日志 —— 那里才知道代价与
/// 后续动作，见 [`Unavailable`]）。
pub fn load(path: &Path, want_stamp: u64) -> Result<File, Unavailable> {
  let f = match File::read(path) {
    Ok(f) => f,
    Err(_) if !path.exists() => return Err(Unavailable::Missing),
    Err(e) => return Err(Unavailable::Bad(e)),
  };
  if f.stamp != want_stamp {
    return Err(Unavailable::Stale);
  }
  Ok(f)
}

/// 已装载的 LOD（进程级的共享句柄）：`McCITY` 每次远场采样读它，构建任务完成后热装新的一份。
pub type Cell = std::sync::Arc<RwLock<Option<std::sync::Arc<View>>>>;

#[cfg(test)]
mod tests {
  use super::*;

  /// 合成一份文件：往返读取后**掩码 / 偏移 / 条目 / 名字表 / 细格**逐字节一致，且采样口径对得上。
  #[test]
  fn file_round_trip_and_sampling() {
    // 2×2 列、3 节；(1,0) 列放两节 (y0 solid 4096 → 实体、y2 solid 100 → 空)
    let dims = (2, 2);
    let mut mask = vec![0u16; 4];
    let mut offs = vec![0u32; 4];
    let mut body = Vec::new();
    let names = vec!["minecraft:stone".to_string(), "minecraft:air".to_string()];
    // 列 1（ix=1, iz=0）
    mask[1] = (1 << 0) | (1 << 2);
    offs[1] = 0;
    body.extend_from_slice(&0u16.to_le_bytes());
    body.extend_from_slice(&4096u16.to_le_bytes());
    body.extend_from_slice(&0u16.to_le_bytes());
    body.extend_from_slice(&100u16.to_le_bytes());
    // 细格：y0 一节 64 格全是 stone（一个 run）；y2 一节 64 格全空（一个 run）
    let mut fine_offs = vec![0u32; 2];
    let mut fine_body = Vec::new();
    fine_body.extend_from_slice(&0u16.to_le_bytes());
    fine_body.push(64);
    fine_offs[1] = fine_body.len() as u32;
    fine_body.extend_from_slice(&FINE_AIR.to_le_bytes());
    fine_body.push(64);
    let f = File {
      stamp: 7,
      col_min: (-1, -1),
      dims,
      sec_y: 16,
      names,
      mask,
      offs,
      body,
      fine_offs,
      fine_body,
    };
    let dir = std::env::temp_dir().join("gate_lod_test");
    let path = dir.join("t.lod");
    let n = f.write(&path).expect("应能写出");
    let back = File::read(&path).expect("应能读回");
    assert_eq!(back.stamp, 7);
    assert_eq!(back.col_min, (-1, -1));
    assert_eq!(back.dims, dims);
    assert_eq!(back.mask, f.mask);
    assert_eq!(back.body, f.body);
    assert_eq!(back.fine_offs, f.fine_offs);
    assert_eq!(back.fine_body, f.fine_body);
    assert_eq!(back.names, f.names);
    assert_eq!(n, fs::metadata(&path).expect("刚写过").len() as usize);
    // 采样：col_min = (-1,-1) ⇒ chunk (-1,-1) 就是 ix=iz=0（空列）；chunk (0,-1) 是 ix=1 那一列
    let stone = PaletteId(3);
    let v = View::new(
      back,
      |k| (k == "minecraft:stone").then_some(stone),
    );
    assert!(v.section(0, 0, -1).is_some_and(|(r, s)| r == stone && s == 4096));
    // 实体不足的节：**写侧**不会落盘（只有"整节实体够 **或** 任一细格有料"才写），所以**读侧以
    // "节存在"为有料**、不再按 125‰ 复判 —— 复判会把只有薄墙/一层楼板的节（高楼）丢掉 ⇒ 中空 +
    // 随距离换档翻。下面这个手写的"实体少 + 细格全空"的节，读出来照样算有料。
    assert!(v.section(0, 2, -1).is_some_and(|(_, s)| s == 100));
    assert_eq!(v.cell(IVec3::new(0, 0, -16), 16), Some(stone), "格 = 一节、实体过半");
    assert_eq!(v.cell(IVec3::new(0, 32, -16), 16), Some(stone), "节存在 ⇒ 算有料");
    assert_eq!(v.cell(IVec3::new(-16, 0, -16), 16), None, "空列");
    // 细档（4 方块格）：格内那一格是 stone ⇒ 有料；y2 那一节细格全空 ⇒ 空（**不看整节的 solid**）
    assert_eq!(v.cell(IVec3::new(0, 0, -16), FINE_CELL), Some(stone), "细格 stone");
    assert_eq!(v.cell(IVec3::new(12, 12, -16), FINE_CELL), Some(stone), "同一节第 64 个细格");
    assert_eq!(v.cell(IVec3::new(0, 32, -16), FINE_CELL), None, "细格全空");
    // n > 1：格内 4³ 节取**最上面那一节有料的**代表色（这里只有 y=0 那一节过门槛）
    assert_eq!(v.cell(IVec3::new(0, 0, -64), 64), Some(stone));
    // 能服务的格边长只有"一节"与"细格"两档
    assert!(v.supports(FINE_CELL) && v.supports(16) && v.supports(64));
    assert!(!v.supports(8), "8 方块既不是细格也不是整数节 ⇒ 该回退 Anvil");
    let _ = fs::remove_file(&path);
    let _ = fs::remove_dir(&dir);
  }

  /// [`load`] 三种"拿不到"的结局各归各的变体（调用点 `mc::build` 靠它写归因日志 + 触发自动重建）。
  #[test]
  fn load_distinguishes_missing_stale_bad() {
    let dir = std::env::temp_dir().join("gate_lod_load_test");
    let path = dir.join("t.lod");
    let _ = fs::remove_file(&path);
    assert!(matches!(load(&path, 1), Err(Unavailable::Missing)));
    // 最小合法文件（1×1 列、空列）：指纹对得上 ⇒ Ok，对不上 ⇒ Stale
    let f = File {
      stamp: 7,
      col_min: (0, 0),
      dims: (1, 1),
      sec_y: 16,
      names: vec!["minecraft:air".to_string()],
      mask: vec![0],
      offs: vec![0],
      body: Vec::new(),
      fine_offs: Vec::new(),
      fine_body: Vec::new(),
    };
    f.write(&path).expect("应能写出");
    assert!(load(&path, 7).is_ok());
    assert!(matches!(load(&path, 8), Err(Unavailable::Stale)));
    // 版本字节被改 ⇒ Bad（而不是静默当"没有文件"）
    let mut raw = fs::read(&path).expect("刚写过");
    raw[8] = 99;
    fs::write(&path, &raw).expect("应能改写");
    assert!(matches!(load(&path, 7), Err(Unavailable::Bad(_))));
    let _ = fs::remove_file(&path);
    let _ = fs::remove_dir(&dir);
  }

  /// 构建期的细格 RLE：门槛按**细格自己**的 64 块算（不是整节），并把连续的同一状态并成一条 run。
  #[test]
  fn fine_cells_rle_and_threshold() {
    // 本节调色板 → 名字下标：0 = 某实心状态、1 = 空气
    let ids = [0u16, FINE_AIR];
    let mut tally = Vec::new();
    // 整节实心 ⇒ 一个 run 铺满 64 格
    let buf = vec![0u16; SECTION_VOLUME];
    assert_eq!(fine_cells(&ids, &buf, &mut tally), vec![(0u16, 64u8)]);
    // 门槛 = `SOLID_MIN_PERMILLE`‰ × 64 块：16‰ ⇒ 1.024 ⇒ 需 ≥ 2 块。第一个细格给 2 块（恰好过）
    let mut buf = vec![1u16; SECTION_VOLUME];
    for i in [0usize, 1] {
      buf[i] = 0;
    }
    assert_eq!(
      fine_cells(&ids, &buf, &mut tally),
      vec![(0u16, 1u8), (FINE_AIR, 63u8)],
      "细格 0 有 2 块（= 门槛上界）⇒ 实体"
    );
    // 少一块 ⇒ 不足门槛 ⇒ 整节一个 run（全空）
    buf[1] = 1;
    assert_eq!(fine_cells(&ids, &buf, &mut tally), vec![(FINE_AIR, 64u8)]);
  }

  /// 细格也取**表面**：一节里 y=3 一层水、y=0..2 三层沙 ⇒ 细格 = **水**（整格多数色会取沙）。
  /// 远场水面就靠这一条才不会被染成河床。
  #[test]
  fn fine_cells_take_the_surface() {
    // 0 = 水（实体）、1 = 沙（实体）、2 = `FINE_AIR`
    let ids = [0u16, 1u16, FINE_AIR];
    let mut buf = vec![2u16; SECTION_VOLUME];
    for y in 0..3 {
      for z in 0..16 {
        for x in 0..16 {
          buf[(y * 256 + z * 16 + x) as usize] = 1;
        }
      }
    }
    for z in 0..16 {
      for x in 0..16 {
        buf[(3 * 256 + z * 16 + x) as usize] = 0;
      }
    }
    let mut tally = Vec::new();
    // `gy = 0` 的 16 个细格（y 0..3）最上面那一层是水 ⇒ 水；`gy ≥ 1` 全空
    assert_eq!(fine_cells(&ids, &buf, &mut tally), vec![(0u16, 16u8), (FINE_AIR, 48u8)]);
  }
}