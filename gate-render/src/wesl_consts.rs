//! WESL 跨端常量的单一来源：权威值只写在 `.wesl` 里，Rust 启动时解析同一份源码。
//! 解析失败 / 常量缺失 → `error!` + `panic!`；改 `.wesl` 重启 app 即生效。

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use bevy::log::{error, info};

use crate::paths::dda_wesl_dir;

/// GI 两侧共用的常量（权威值在 WESL `gi/`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GiConsts {
  /// 条目容量（`GI_CACHE_SLOTS`）
  pub gi_cache_slots: u32,
  /// 每条目 word 数（`GI_CACHE_ENTRY_WORDS`）
  pub gi_cache_entry_words: u32,
  /// 哈希桶数（`GI_CACHE_BUCKETS`，2 的幂）
  pub gi_cache_buckets: u32,
  /// 每帧更新的条目数上限（`GI_CACHE_UPDATE_BUDGET`）
  pub gi_cache_update_budget: u32,
  /// 可见条目紧凑列表容量（`GI_CACHE_VISIBLE_CAPACITY`）
  pub gi_cache_visible_capacity: u32,
  /// 可见优先占更新预算的百分比（`GI_CACHE_VISIBLE_SHARE`，50 = 1/2）
  pub gi_cache_visible_share: u32,
  /// 脏盒上限（`GI_CACHE_DIRTY_BOXES`）：超出即退化为自增世代
  pub gi_cache_dirty_boxes: u32,
  /// 脏盒失效余量的基准值（voxel，`GI_CACHE_DIRTY_MARGIN_VOXELS`）；物体 volume 按
  /// `max(本值, 本值 × scale)` 放大（见 `brickmap::upload::world_dirty_box`）
  pub gi_cache_dirty_margin_voxels: u32,
  /// 世界空间缓存总开关（`GI_CACHE_ON`）：false ⇒ 主路径是屏幕空间逐面 ReSTIR
  /// （`gi/screen.wesl`），更新 pass **不派发**（省下整表轮转的开销）。两条路径在 `gi_main` 二选一。
  pub gi_cache_on: bool,
  /// 屏幕空间 reservoir 每像素 word 数（`GI_RES_WORDS`）：Rust 按它开 ping-pong 两块 buffer。
  pub gi_res_words: u32,
  /// 降噪导引每像素 word 数（`GI_DEN_GUIDE_WORDS`）：Rust 按它开导引 buffer。
  pub gi_den_guide_words: u32,
  /// 降噪历史每像素 word 数（`GI_DEN_HIST_WORDS`）：Rust 按它开历史 ping-pong 两块 buffer。
  pub gi_den_hist_words: u32,
  /// atrous 迭代次数（`GI_DEN_ATROUS_ITER`，1..=3）：Rust 按它决定派发几轮 / 每轮的 src→dst。
  pub gi_den_atrous_iter: u32,
}

impl GiConsts {
  /// 条目表字节数。
  #[inline]
  pub fn gi_cache_bytes(&self) -> u64 {
    self.gi_cache_slots as u64 * self.gi_cache_entry_words as u64 * 4
  }
  /// 哈希桶字节数。
  #[inline]
  pub fn gi_cache_bucket_bytes(&self) -> u64 {
    self.gi_cache_buckets as u64 * 4
  }
  /// 可见条目紧凑列表字节数。
  #[inline]
  pub fn gi_cache_visible_bytes(&self) -> u64 {
    self.gi_cache_visible_capacity as u64 * 4
  }
  /// 单帧可见 claim 字节数（每槽 4 B）。
  #[inline]
  pub fn gi_cache_claim_bytes(&self) -> u64 {
    self.gi_cache_slots as u64 * 4
  }
  /// 脏盒 buffer 字节数（每盒 2 个 vec4 = 32 B）。
  #[inline]
  pub fn gi_cache_dirty_box_bytes(&self) -> u64 {
    self.gi_cache_dirty_boxes.max(1) as u64 * 32
  }
  /// 更新 pass 的 workgroup 数（每 WG 64 线程）。
  #[inline]
  pub fn gi_cache_update_wgs(&self) -> u32 {
    self.gi_cache_update_budget.div_ceil(64)
  }
  /// 更新 pass 的**轮转段**线程数 = 预算 − 可见优先段（`GI_CACHE_VISIBLE_SHARE` 百分比）。
  /// 必须与 WESL `gi_cache_update` 的同一算式一致。
  #[inline]
  pub fn gi_cache_rotate_threads(&self) -> u32 {
    let visible = self.gi_cache_update_budget.saturating_mul(self.gi_cache_visible_share) / 100;
    self.gi_cache_update_budget.saturating_sub(visible).max(1)
  }
  /// 整表轮转一遍的帧数（脏区 TTL 用它：脏区条目必须在这段时间内至少被轮转到一次）。
  #[inline]
  pub fn gi_cache_sweeps(&self) -> u32 {
    self.gi_cache_slots.div_ceil(self.gi_cache_rotate_threads()).max(1)
  }
}

/// 需要的全部常量名（缺一即 fail fast）。
const REQUIRED: &[&str] = &[
  "GI_CACHE_SLOTS",
  "GI_CACHE_ENTRY_WORDS",
  "GI_CACHE_BUCKETS",
  "GI_CACHE_UPDATE_BUDGET",
  "GI_CACHE_VISIBLE_CAPACITY",
  "GI_CACHE_VISIBLE_SHARE",
  "GI_CACHE_DIRTY_BOXES",
  "GI_CACHE_DIRTY_MARGIN_VOXELS",
  "GI_RES_WORDS",
  "GI_DEN_GUIDE_WORDS",
  "GI_DEN_HIST_WORDS",
  "GI_DEN_ATROUS_ITER",
];

/// 需要的全部 `bool` 常量名（缺一即 fail fast）。
const REQUIRED_BOOL: &[&str] = &["GI_CACHE_ON"];

/// 解析 WESL 包里的跨端常量（首次读盘，之后走 `OnceLock`）。
/// 失败（文件读不到 / 常量缺失 / 不是字面量）→ `error!` + `panic!`。
pub fn gi_consts() -> &'static GiConsts {
  static CONSTS: OnceLock<GiConsts> = OnceLock::new();
  CONSTS.get_or_init(GiConsts::load)
}

impl GiConsts {
  fn load() -> Self {
    let dir = dda_wesl_dir();
    let values = parse_package_u32_consts(&dir);
    let missing: Vec<&str> =
      REQUIRED.iter().copied().filter(|name| !values.contains_key(*name)).collect();
    if !missing.is_empty() {
      let msg = format!(
        "WESL 跨端常量缺失（{}）：{:?}\n\
         —— 权威值只写在 .wesl 里，Rust 从源码解析；请检查 gi/cache.wesl，\
         并确保它们写成 `const NAME: u32 = <字面量>u;` 形式。",
        dir.display(),
        missing
      );
      error!("{msg}");
      panic!("{msg}");
    }
    let bools = parse_package_bool_consts(&dir);
    let missing_bool: Vec<&str> =
      REQUIRED_BOOL.iter().copied().filter(|name| !bools.contains_key(*name)).collect();
    if !missing_bool.is_empty() {
      let msg = format!(
        "WESL 跨端 bool 常量缺失（{}）：{:?}\n\
         —— 请确保它们写成 `const NAME: bool = true|false;` 形式（如 gi/cache.wesl 的 GI_CACHE_ON）。",
        dir.display(),
        missing_bool
      );
      error!("{msg}");
      panic!("{msg}");
    }
    let get = |name: &str| -> u32 { values[name] };
    let out = GiConsts {
      gi_cache_slots: get("GI_CACHE_SLOTS"),
      gi_cache_entry_words: get("GI_CACHE_ENTRY_WORDS"),
      gi_cache_buckets: get("GI_CACHE_BUCKETS"),
      gi_cache_update_budget: get("GI_CACHE_UPDATE_BUDGET"),
      gi_cache_visible_capacity: get("GI_CACHE_VISIBLE_CAPACITY"),
      gi_cache_visible_share: get("GI_CACHE_VISIBLE_SHARE"),
      gi_cache_dirty_boxes: get("GI_CACHE_DIRTY_BOXES"),
      gi_cache_dirty_margin_voxels: get("GI_CACHE_DIRTY_MARGIN_VOXELS"),
      gi_cache_on: bools["GI_CACHE_ON"],
      gi_res_words: get("GI_RES_WORDS"),
      gi_den_guide_words: get("GI_DEN_GUIDE_WORDS"),
      gi_den_hist_words: get("GI_DEN_HIST_WORDS"),
      gi_den_atrous_iter: get("GI_DEN_ATROUS_ITER"),
    };

    if out.gi_cache_slots == 0 || out.gi_cache_entry_words == 0 || out.gi_cache_buckets == 0 {
      let msg = format!("WESL GI 缓存常量存在 0 值（缓存会退化成空 buffer）：{out:?}");
      error!("{msg}");
      panic!("{msg}");
    }
    if out.gi_res_words < 2 {
      let msg = format!("GI_RES_WORDS = {} 太小（reservoir 至少要有键位与累计量）：{out:?}", out.gi_res_words);
      error!("{msg}");
      panic!("{msg}");
    }
    if out.gi_den_guide_words < 4 || out.gi_den_hist_words < 4 {
      let msg = format!("降噪 buffer 布局常量太小（导引 ≥ 4 字、历史 ≥ 4 字）：{out:?}");
      error!("{msg}");
      panic!("{msg}");
    }
    if !(1..=3).contains(&out.gi_den_atrous_iter) {
      let msg = format!(
        "GI_DEN_ATROUS_ITER = {} 越界：只实现了步长 1/2/4 三轮（Rust 侧的派发链表就 3 项）：{out:?}",
        out.gi_den_atrous_iter
      );
      error!("{msg}");
      panic!("{msg}");
    }
    if out.gi_cache_visible_share > 100 {
      let msg = format!(
        "GI_CACHE_VISIBLE_SHARE = {} 超过 100%（可见优先的份额不能超过整个更新预算）：{out:?}",
        out.gi_cache_visible_share
      );
      error!("{msg}");
      panic!("{msg}");
    }
    info!(
      target: "gate",
      "WESL 跨端常量（源 {}）：GI 条目 {} × {} word = {:.1}MB、哈希桶 {} = {:.1}MB、\
       每帧更新预算 {} 条（可见优先 {}%）、可见列表 {} 项 = {:.1}MB、claim {} 项 = {:.1}MB、\
       脏盒上限 {} 个（余量基准 {} voxel）、整表轮转 {} 帧；\
       世界空间缓存开关 GI_CACHE_ON = {}（false = 屏幕空间逐面 ReSTIR 主路径）、\
       屏幕空间 reservoir {} word/像素；\
       降噪：导引 {} word/像素、历史 {} word/像素（双缓冲）、atrous 迭代 {} 轮",
      dir.display(),
      out.gi_cache_slots,
      out.gi_cache_entry_words,
      out.gi_cache_bytes() as f64 / (1 << 20) as f64,
      out.gi_cache_buckets,
      out.gi_cache_bucket_bytes() as f64 / (1 << 20) as f64,
      out.gi_cache_update_budget,
      out.gi_cache_visible_share,
      out.gi_cache_visible_capacity,
      out.gi_cache_visible_bytes() as f64 / (1 << 20) as f64,
      out.gi_cache_slots,
      out.gi_cache_claim_bytes() as f64 / (1 << 20) as f64,
      out.gi_cache_dirty_boxes,
      out.gi_cache_dirty_margin_voxels,
      out.gi_cache_sweeps(),
      out.gi_cache_on,
      out.gi_res_words,
      out.gi_den_guide_words,
      out.gi_den_hist_words,
      out.gi_den_atrous_iter,
    );
    out
  }
}

/// 递归收集 `.wesl` 文件（包结构是"根 + 子目录模块"，模块路径就是目录结构）。
fn collect_wesl_files(dir: &Path, out: &mut Vec<PathBuf>) {
  let Ok(entries) = std::fs::read_dir(dir) else {
    return;
  };
  for entry in entries.flatten() {
    let path = entry.path();
    if path.is_dir() {
      collect_wesl_files(&path, out);
    } else if path.extension().is_some_and(|e| e == "wesl") {
      out.push(path);
    }
  }
}

/// 解析整个包的全部 `const NAME: u32 = <字面量>;`（同名以先遇到的为准，包内不应重名）。
fn parse_package_u32_consts(dir: &Path) -> HashMap<String, u32> {
  let mut files = Vec::new();
  collect_wesl_files(dir, &mut files);
  files.sort();
  let mut out = HashMap::new();
  for path in files {
    let Ok(src) = std::fs::read_to_string(&path) else {
      error!(target: "gate", "读不到 WESL 源文件 {}", path.display());
      continue;
    };
    for (name, value) in parse_u32_consts_in_source(&src) {
      out.entry(name).or_insert(value);
    }
  }
  out
}

/// 解析整个包的全部 `const NAME: bool = true|false;`（同名以先遇到的为准）。
fn parse_package_bool_consts(dir: &Path) -> HashMap<String, bool> {
  let mut files = Vec::new();
  collect_wesl_files(dir, &mut files);
  files.sort();
  let mut out = HashMap::new();
  for path in files {
    let Ok(src) = std::fs::read_to_string(&path) else {
      error!(target: "gate", "读不到 WESL 源文件 {}", path.display());
      continue;
    };
    for line in src.lines() {
      if let Some((name, value)) = parse_bool_const_line(line) {
        out.entry(name).or_insert(value);
      }
    }
  }
  out
}

/// 抽一行 `const NAME: bool = true|false;`（行尾 `//` 注释先剥掉）。
fn parse_bool_const_line(raw: &str) -> Option<(String, bool)> {
  let line = raw.trim().trim_start_matches('\u{feff}').split("//").next()?.trim();
  let rest = line.strip_prefix("const ")?.trim_start();
  let (name, rest) = rest.split_once(':')?;
  let name = name.trim();
  if name.is_empty() || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
    return None;
  }
  let rest = rest.trim_start().strip_prefix("bool")?.trim_start();
  let rest = rest.strip_prefix('=')?.trim().strip_suffix(';')?.trim();
  match rest {
    "true" => Some((name.to_string(), true)),
    "false" => Some((name.to_string(), false)),
    _ => None,
  }
}

/// 从一段 WGSL/WESL 源码里抽全部 `const NAME: u32 = <字面量>;`。
pub fn parse_u32_consts_in_source(src: &str) -> HashMap<String, u32> {
  let mut out = HashMap::new();
  for line in src.lines() {
    if let Some((name, value)) = parse_u32_const_line(line) {
      out.entry(name).or_insert(value);
    }
  }
  out
}

/// 抽一行 `const NAME: u32 = <字面量>;`。
/// 只认字面量（十进制 / `0x` 十六进制，可带 `u` 后缀）；派生式返回 `None`，行尾 `//` 注释先剥掉。
fn parse_u32_const_line(raw: &str) -> Option<(String, u32)> {
  let line = raw.trim().trim_start_matches('\u{feff}').split("//").next()?.trim();
  let rest = line.strip_prefix("const ")?.trim_start();
  let (name, rest) = rest.split_once(':')?;
  let name = name.trim();
  if name.is_empty() || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
    return None;
  }
  let rest = rest.trim_start().strip_prefix("u32")?.trim_start();
  let rest = rest.strip_prefix('=')?.trim();
  let rest = rest.strip_suffix(';')?.trim();
  let rest = rest.strip_suffix('u').unwrap_or(rest).trim();
  let value = match rest.strip_prefix("0x").or_else(|| rest.strip_prefix("0X")) {
    Some(hex) => u32::from_str_radix(hex, 16).ok()?,
    None => rest.parse::<u32>().ok()?,
  };
  Some((name.to_string(), value))
}
