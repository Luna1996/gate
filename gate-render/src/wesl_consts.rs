use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use bevy::log::{error, info};

use crate::paths::dda_wesl_dir;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GiConsts {
  pub irc_words: u32,
  pub lt_tile_words: u32,
  pub lt_node_words: u32,
  pub irc_bucket: u32,
  pub irc_chain: u32,
  pub lt_bucket: u32,
  pub lt_top: u32,
  pub gi_stat_words: u32,
}

const REQUIRED: &[&str] = &[
  "IRC_WORDS",
  "LT_TILE_WORDS",
  "LT_NODE_WORDS",
  "IRC_BUCKET",
  "IRC_CHAIN",
  "LT_BUCKET",
  "LT_TOP",
  "GI_STAT_WORDS",
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
  pub occ_volumes: u32,
  pub index_entry_empty: u32,
  pub seg_words: u32,
}

const TRACE_REQUIRED: &[&str] = &[
  "LOD_DIAG",
  "REQ_ENABLE",
  "REQ_SAMPLE",
  "REQ_PER_RAY_MAX",
  "GRID_VOLUMES",
  "OCC_VOLUMES",
  "INDEX_ENTRY_EMPTY",
  "SEG_WORDS",
];

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
      irc_words: get("IRC_WORDS"),
      lt_tile_words: get("LT_TILE_WORDS"),
      lt_node_words: get("LT_NODE_WORDS"),
      irc_bucket: get("IRC_BUCKET"),
      irc_chain: get("IRC_CHAIN"),
      lt_bucket: get("LT_BUCKET"),
      lt_top: get("LT_TOP"),
      gi_stat_words: get("GI_STAT_WORDS"),
    };

    if out.irc_words < 16 {
      let msg = format!(
        "IRC_WORDS = {} 太小（键 2 + 值 + 均值 3 + n + var + stamp + 累加 5）：{out:?}",
        out.irc_words
      );
      error!("{msg}");
      panic!("{msg}");
    }
    if out.lt_tile_words < 12 || out.lt_node_words < 12 {
      let msg = format!("光源瓦片/节点表字宽太小（≥ 12）：{out:?}");
      error!("{msg}");
      panic!("{msg}");
    }
    if !out.irc_bucket.is_power_of_two() || !out.lt_bucket.is_power_of_two() {
      let msg = format!("桶宽必须是 2 的幂：{out:?}");
      error!("{msg}");
      panic!("{msg}");
    }
    info!(
      target: "gate",
      "WESL 跨端常量（源 {}）：世界空间辐照度缓存 {} word/槽 × {} 槽/桶、链 {} 级；\
       光源瓦片 {} word / 节点 {} word、{}/桶、顶层 {}",
      dir.display(),
      out.irc_words,
      out.irc_bucket,
      out.irc_chain,
      out.lt_tile_words,
      out.lt_node_words,
      out.lt_bucket,
      out.lt_top,
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
      occ_volumes: get("OCC_VOLUMES"),
      index_entry_empty: get("INDEX_ENTRY_EMPTY"),
      seg_words: get("SEG_WORDS"),
    };

    if out.seg_words == 0 {
      let msg = format!("SEG_WORDS = 0：树区分页边界非法（{out:?}）");
      error!("{msg}");
      panic!("{msg}");
    }

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
    if out.occ_volumes != crate::brickmap::consts::OCC_VOLUMES as u32 {
      let msg = format!(
        "occ volume 容量不一致：trace.wesl::OCC_VOLUMES = {}，\
         brickmap::consts::OCC_VOLUMES = {}（两侧必须相等，它决定 `occ` 缓冲的分槽）",
        out.occ_volumes,
        crate::brickmap::consts::OCC_VOLUMES
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
