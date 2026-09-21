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
  /// atrous 单轮核半径 · **中/高 档**（`GI_DEN_ATROUS_R`）：Rust 按菜单写进降噪的小配置 buffer。
  pub gi_den_atrous_r: u32,
  /// atrous 单轮核半径 · **关/低 档**（`GI_DEN_ATROUS_R_FAST`）。
  pub gi_den_atrous_r_fast: u32,
  /// 每 texel 新鲜候选数 · **关/低/中 档**（`GI_SS_CAND_N`，所有分辨率档共用；菜单日志要用真实值，
  /// 所以这几个也跟着解析（避免 Rust 侧另抄一份、与 `.wesl` 漂移）。
  pub gi_ss_cand_n: u32,
  /// 每 texel 新鲜候选数 · **高 档**（`GI_SS_CAND_N_HQ`，所有分辨率档共用）。
  pub gi_ss_cand_n_hq: u32,
  /// reservoir 记忆窗（帧）· **关/低/中 档**（`GI_SS_M_CAP_K`）。
  pub gi_ss_m_cap_k: u32,
  /// reservoir 记忆窗（帧）· **高 档**（`GI_SS_M_CAP_K_HQ`）。
  pub gi_ss_m_cap_k_hq: u32,
  /// **帧内逐面去重表**每槽 word 数（`FACE_WORDS`，权威值在 `gi/common.wesl`）：
  /// Rust 按它开表 buffer（槽数 = GI 网格像素数 × 2，向上取 2 的幂）。
  pub face_words: u32,
}

/// 需要的全部常量名（缺一即 fail fast）。
const REQUIRED: &[&str] = &[
  "GI_RES_WORDS",
  "GI_DEN_GUIDE_WORDS",
  "GI_DEN_HIST_WORDS",
  "GI_DEN_ATROUS_ITER",
  "GI_DEN_ATROUS_R",
  "GI_DEN_ATROUS_R_FAST",
  "GI_SS_CAND_N",
  "GI_SS_CAND_N_HQ",
  "GI_SS_M_CAP_K",
  "GI_SS_M_CAP_K_HQ",
  "FACE_WORDS",
];

/// 材质资产两侧共用的常量（权威值在 WESL `common.wesl`）。
/// **独立于 [`GiConsts`]**：GI 常量与材质常量各自的解析互不牵连（一个缺失只 fail 自己那一侧）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaterialConsts {
  /// 材质资产槽数上限（`MATERIAL_ASSET_SLOTS`）：Rust 按它开资产表 buffer 并钳制 `asset` 索引。
  /// **全局一张表**（所有 volume 共用，`asset: u16` 是全局下标），不是 per-volume。
  pub material_asset_slots: u32,
  /// 贴图槽位（`texture_2d_array` 层）数目上限（`MATERIAL_TEX_SLOTS`）：Rust 按它校验/分配层数。
  pub material_tex_slots: u32,
}

/// [`MaterialConsts`] 需要的常量名（缺一即 fail fast）。
const MATERIAL_REQUIRED: &[&str] = &["MATERIAL_ASSET_SLOTS", "MATERIAL_TEX_SLOTS"];

// MT8-3 的 `ReflConsts` / `refl_consts()` / `REFL_ENTRY_BYTES` 已随反射缓存一起删除（实测负优化）。
// 这里只保留**通用**解析器（`parse_package_u32_consts` / `parse_u32_consts_in_source`），
// GI 与材质两组常量仍在用；`pbr_texture.rs` 也直接用后者抽 `.wesl` 里的字面量。

/// 解析 WESL 包里的跨端常量（首次读盘，之后走 `OnceLock`）。
/// 失败（文件读不到 / 常量缺失 / 不是字面量）→ `error!` + `panic!`。
pub fn gi_consts() -> &'static GiConsts {
  static CONSTS: OnceLock<GiConsts> = OnceLock::new();
  CONSTS.get_or_init(GiConsts::load)
}

/// 解析 WESL 包里的材质资产常量（首次读盘，之后走 `OnceLock`）。
/// 失败（文件读不到 / 常量缺失 / 不是字面量）→ `error!` + `panic!`；与 [`gi_consts`] 相互独立。
pub fn material_consts() -> &'static MaterialConsts {
  static CONSTS: OnceLock<MaterialConsts> = OnceLock::new();
  CONSTS.get_or_init(MaterialConsts::load)
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
      gi_den_atrous_r: get("GI_DEN_ATROUS_R"),
      gi_den_atrous_r_fast: get("GI_DEN_ATROUS_R_FAST"),
      gi_ss_cand_n: get("GI_SS_CAND_N"),
      gi_ss_cand_n_hq: get("GI_SS_CAND_N_HQ"),
      gi_ss_m_cap_k: get("GI_SS_M_CAP_K"),
      gi_ss_m_cap_k_hq: get("GI_SS_M_CAP_K_HQ"),
      face_words: get("FACE_WORDS"),
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
    if out.face_words < 8 {
      let msg = format!(
        "FACE_WORDS = {} 太小：逐面去重表每槽至少要放「标志 + 键 2 + 认领者 texel + palette + 颜色 3」= 8 字：{out:?}",
        out.face_words
      );
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
    if !(1..=4).contains(&out.gi_den_atrous_r) || !(1..=4).contains(&out.gi_den_atrous_r_fast) {
      let msg =
        format!("atrous 核半径越界（1..=4；tap 数按 (2R+1)²-1 涨，越界会造成性能悬崖）：{out:?}");
      error!("{msg}");
      panic!("{msg}");
    }
    if out.gi_ss_cand_n < 1
      || out.gi_ss_cand_n_hq < out.gi_ss_cand_n
      || out.gi_ss_m_cap_k < 1
      || out.gi_ss_m_cap_k_hq < out.gi_ss_m_cap_k
    {
      let msg =
        format!("「降噪质量」档的采样/记忆窗常量不自洽（高档应 ≥ 基础档、且都 ≥ 1）：{out:?}");
      error!("{msg}");
      panic!("{msg}");
    }
    info!(
      target: "gate",
      "WESL 跨端常量（源 {}）：屏幕空间 reservoir {} word/像素；\
       降噪：导引 {} word/像素、历史 {} word/像素（双缓冲）、atrous 迭代 {} 轮（核半径：质量档 {} / 快速档 {}）",
      dir.display(),
      out.gi_res_words,
      out.gi_den_guide_words,
      out.gi_den_hist_words,
      out.gi_den_atrous_iter,
      out.gi_den_atrous_r,
      out.gi_den_atrous_r_fast,
    );
    out
  }
}

impl MaterialConsts {
  fn load() -> Self {
    let dir = dda_wesl_dir();
    let values = parse_package_u32_consts(&dir);
    let missing: Vec<&str> =
      MATERIAL_REQUIRED.iter().copied().filter(|name| !values.contains_key(*name)).collect();
    if !missing.is_empty() {
      let msg = format!(
        "WESL 材质跨端常量缺失（{}）：{:?}\n\
         —— 权威值只写在 .wesl 里，Rust 从源码解析；请检查 common.wesl 的材质资产常量段，\
         并确保它们写成 `const NAME: u32 = <字面量>u;` 形式。",
        dir.display(),
        missing
      );
      error!("{msg}");
      panic!("{msg}");
    }
    let get = |name: &str| -> u32 { values[name] };
    let out = MaterialConsts {
      material_asset_slots: get("MATERIAL_ASSET_SLOTS"),
      material_tex_slots: get("MATERIAL_TEX_SLOTS"),
    };

    // asset 索引是 u16（PBR 变体 word1 低 16 位）⇒ 表最多 2^16 项；0 则任何资产都索引不到。
    if out.material_asset_slots < 1 || out.material_asset_slots > 65_536 {
      let msg = format!("材质资产槽数越界（1..=65536，asset 索引是 u16）：{out:?}");
      error!("{msg}");
      panic!("{msg}");
    }
    if out.material_tex_slots < 1 {
      let msg = format!("贴图槽位上限为 0 ⇒ 任何贴图都放不下（至少 1）：{out:?}");
      error!("{msg}");
      panic!("{msg}");
    }
    let asset_bytes = std::mem::size_of::<crate::brickmap::wire::MaterialAsset>() as u32;
    info!(
      target: "gate",
      "WESL 跨端常量（源 {}）：材质资产表 {} 槽（全局一张表，所有 volume 共用；{} B/槽 = {} KB）；\
       贴图槽位上限 {} 层",
      dir.display(),
      out.material_asset_slots,
      asset_bytes,
      (out.material_asset_slots as u64 * asset_bytes as u64) / 1024,
      out.material_tex_slots,
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
