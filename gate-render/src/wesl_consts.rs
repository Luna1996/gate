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
  /// 屏幕空间 reservoir 每像素 word 数（`GI_RES_WORDS`）：Rust 按它开 ping-pong 两块 buffer。
  pub gi_res_words: u32,
  /// 降噪导引每像素 word 数（`GI_DEN_GUIDE_WORDS`）：Rust 按它开导引 buffer。
  pub gi_den_guide_words: u32,
  /// 降噪历史每像素 word 数（`GI_DEN_HIST_WORDS`）：Rust 按它开历史 ping-pong 两块 buffer。
  pub gi_den_hist_words: u32,
  /// atrous 迭代次数（`GI_DEN_ATROUS_ITER`，1..=5）：Rust 按它决定派发几轮 / 每轮的 src→dst。
  pub gi_den_atrous_iter: u32,
}

/// 需要的全部常量名（缺一即 fail fast）。
const REQUIRED: &[&str] =
  &["GI_RES_WORDS", "GI_DEN_GUIDE_WORDS", "GI_DEN_HIST_WORDS", "GI_DEN_ATROUS_ITER"];

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
         —— 权威值只写在 .wesl 里，Rust 从源码解析；请检查 gi/screen.wesl 与 gi/consts.wesl，\
         并确保它们写成 `const NAME: u32 = <字面量>u;` 形式。",
        dir.display(),
        missing
      );
      error!("{msg}");
      panic!("{msg}");
    }
    let get = |name: &str| -> u32 { values[name] };
    let out = GiConsts {
      gi_res_words: get("GI_RES_WORDS"),
      gi_den_guide_words: get("GI_DEN_GUIDE_WORDS"),
      gi_den_hist_words: get("GI_DEN_HIST_WORDS"),
      gi_den_atrous_iter: get("GI_DEN_ATROUS_ITER"),
    };

    if out.gi_res_words < 2 {
      let msg = format!(
        "GI_RES_WORDS = {} 太小（reservoir 至少要有方向与累计量）：{out:?}",
        out.gi_res_words
      );
      error!("{msg}");
      panic!("{msg}");
    }
    if out.gi_den_guide_words < 4 || out.gi_den_hist_words < 4 {
      let msg = format!("降噪 buffer 布局常量太小（导引 ≥ 4 字、历史 ≥ 4 字）：{out:?}");
      error!("{msg}");
      panic!("{msg}");
    }
    if !(1..=5).contains(&out.gi_den_atrous_iter) {
      let msg = format!(
        "GI_DEN_ATROUS_ITER = {} 越界：只实现了步长 1/2/4/8/16 五轮（Rust 侧的派发链表就 5 项）：{out:?}",
        out.gi_den_atrous_iter
      );
      error!("{msg}");
      panic!("{msg}");
    }
    info!(
      target: "gate",
      "WESL 跨端常量（源 {}）：屏幕空间 reservoir {} word/像素；\
       降噪：导引 {} word/像素、历史 {} word/像素（双缓冲）、atrous 迭代 {} 轮",
      dir.display(),
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
