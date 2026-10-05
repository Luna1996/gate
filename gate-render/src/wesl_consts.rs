
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use bevy::log::{error, info};

use crate::paths::dda_wesl_dir;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GiConsts {
    pub gi_res_words: u32,
    pub gi_den_guide_words: u32,
    pub gi_den_hist_words: u32,
    pub gi_den_atrous_iter: u32,
    pub gi_den_atrous_r: u32,
    pub gi_den_atrous_r_fast: u32,
      pub gi_ss_cand_n: u32,
    pub gi_ss_cand_n_hq: u32,
    pub gi_ss_m_cap_k: u32,
    pub gi_ss_m_cap_k_hq: u32,
      pub face_words: u32,
      pub gi_sec_words: u32,
}

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
  "GI_SEC_WORDS",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaterialConsts {
      pub material_asset_slots: u32,
    pub material_tex_slots: u32,
}

const MATERIAL_REQUIRED: &[&str] = &["MATERIAL_ASSET_SLOTS", "MATERIAL_TEX_SLOTS"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TraceConsts {
    pub lod_diag: u32,
    pub req_enable: u32,
    pub req_sample: u32,
    pub req_per_ray_max: u32,
      pub volumes: u32,
        pub index_entry_empty: u32,
}

const TRACE_REQUIRED: &[&str] =
  &["LOD_DIAG", "REQ_ENABLE", "REQ_SAMPLE", "REQ_PER_RAY_MAX", "GRID_VOLUMES", "INDEX_ENTRY_EMPTY"];


pub fn gi_consts() -> &'static GiConsts {
  static CONSTS: OnceLock<GiConsts> = OnceLock::new();
  CONSTS.get_or_init(GiConsts::load)
}

pub fn material_consts() -> &'static MaterialConsts {
  static CONSTS: OnceLock<MaterialConsts> = OnceLock::new();
  CONSTS.get_or_init(MaterialConsts::load)
}

pub fn trace_consts() -> &'static TraceConsts {
  static CONSTS: OnceLock<TraceConsts> = OnceLock::new();
  CONSTS.get_or_init(TraceConsts::load)
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
      gi_sec_words: get("GI_SEC_WORDS"),
    };

    if out.gi_res_words < 8 {
      let msg = format!(
        "GI_RES_WORDS = {} 太小（reservoir = 状态位 + 面键 2 + 着色法线 + 累计量 rgb + M = 8）：{out:?}",
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
    if out.gi_sec_words < 5 {
      let msg = format!(
        "GI_SEC_WORDS = {} 太小：二次顶点缓存每槽至少要放「键 2 + 辐亮度 3」= 5 字：{out:?}",
        out.gi_sec_words
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
      "WESL 跨端常量（源 {}）：屏幕空间 reservoir {} word/像素；降噪：导引 {} / 历史 {} word/像素（双缓冲）、\
       atrous {} 轮（核半径 质量档 {} / 快速档 {}）",
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
      "WESL 跨端常量（源 {}）：材质资产表 {} 槽 × {asset_bytes}B = {} KB（全局一张表，所有 volume 共用）；\
       贴图槽位上限 {} 层",
      dir.display(),
      out.material_asset_slots,
      (out.material_asset_slots as u64 * asset_bytes as u64) / 1024,
      out.material_tex_slots,
    );
    out
  }
}

impl TraceConsts {
  fn load() -> Self {
    let dir = dda_wesl_dir();
    let values = parse_package_u32_consts(&dir);
    let missing: Vec<&str> =
      TRACE_REQUIRED.iter().copied().filter(|name| !values.contains_key(*name)).collect();
    if !missing.is_empty() {
      let msg = format!(
        "WESL trace 跨端常量缺失（{}）：{:?}\n\
         —— 权威值只写在 .wesl 里，Rust 从源码解析；请检查 voxel_raytrace/trace.wesl 与 common.wesl，\
         并确保它们写成 `const NAME: u32 = <字面量>u;` 形式。",
        dir.display(),
        missing
      );
      error!("{msg}");
      panic!("{msg}");
    }
    let get = |name: &str| -> u32 { values[name] };
    let out = TraceConsts {
      lod_diag: get("LOD_DIAG"),
      req_enable: get("REQ_ENABLE"),
      req_sample: get("REQ_SAMPLE"),
      req_per_ray_max: get("REQ_PER_RAY_MAX"),
      volumes: get("GRID_VOLUMES"),
      index_entry_empty: get("INDEX_ENTRY_EMPTY"),
    };

        if out.req_sample < 1 || out.req_per_ray_max < 1 {
      let msg =
        format!("请求通道节流常量非法（`%REQ_SAMPLE` 与 `<REQ_PER_RAY_MAX` 都要 ≥ 1）：{out:?}");
      error!("{msg}");
      panic!("{msg}");
    }
            if out.volumes != crate::brickmap::consts::VOLUMES as u32 {
      let msg = format!(
        "grid volume 数不一致：trace.wesl::GRID_VOLUMES = {}，\
         brickmap::consts::VOLUMES = {}（两侧必须相等，它决定 `lod_req` 的分段与长度）",
        out.volumes,
        crate::brickmap::consts::VOLUMES
      );
      error!("{msg}");
      panic!("{msg}");
    }
        if out.index_entry_empty != crate::brickmap::consts::INDEX_ENTRY_EMPTY {
      let msg = format!(
        "已知空块哨兵不一致：common.wesl::INDEX_ENTRY_EMPTY = {:#x}，\
         brickmap::consts::INDEX_ENTRY_EMPTY = {:#x}（两侧必须相等）",
        out.index_entry_empty,
        crate::brickmap::consts::INDEX_ENTRY_EMPTY
      );
      error!("{msg}");
      panic!("{msg}");
    }
    info!(
      target: "gate",
      "WESL 跨端常量（源 {}）：LOD 诊断 {}；ray-guided 请求 {}（采样 1/{}/条射线、每条 ≤ {} 条）；\
       grid volume 数 {}",
      dir.display(),
      if out.lod_diag != 0 { "开" } else { "关" },
      if out.req_enable != 0 { "开" } else { "关" },
      out.req_sample,
      out.req_per_ray_max,
      out.volumes,
    );
    out
  }
}

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

pub fn parse_u32_consts_in_source(src: &str) -> HashMap<String, u32> {
  let mut out = HashMap::new();
  for line in src.lines() {
    if let Some((name, value)) = parse_u32_const_line(line) {
      out.entry(name).or_insert(value);
    }
  }
  out
}

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
