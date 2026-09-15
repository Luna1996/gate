//! WESL **跨端常量**的单一来源：权威值只写在 `.wesl` 里，Rust 启动时解析同一份源码。
//!
//! 图集尺寸 / LOD 级数 / 射线预算 / `ddgi_indirect` 的 word 布局两侧都要知道（WESL 按它们
//! 寻址、分配线程，Rust 按它们 `create_texture` / `create_buffer` / 算字节偏移），各写一份
//! 就会静默漂移且无任何编译或运行时报错。解析失败 / 常量缺失 → `error!` + `panic!`。
//! 与 [`crate::shader::compile_dda_wesl`] 共用 `DDA_WESL_DIR`，改 `.wesl` 重启 app 即生效。
//! 例外：`DDGI_LODS` 要定 `[T; N]` 数组与 `ShaderType` 布局长度，仍在 Rust 声明并断言一致。

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use bevy::log::{error, info};

use crate::shader::DDA_WESL_DIR;

/// DDGI 两侧共用的常量（值来自 WESL 源码，见模块注释）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DdgiConsts {
  /// 每探针 irradiance 图边长的纹素数（`DDGI_IRR_TEXELS`）
  pub irr_texels: u32,
  /// 每探针 depth 图边长的纹素数（`DDGI_DEPTH_TEXELS`）
  pub depth_texels: u32,
  /// 图集每层每轴的探针数（`DDGI_PROBES_PER_LAYER_AXIS`）
  pub probes_per_layer_axis: u32,
  /// 图集层数（`DDGI_ATLAS_LAYERS`）
  pub atlas_layers: u32,
  /// LOD 级数（`DDGI_LOD_COUNT`）；必须等于 `crate::ddgi::DDGI_LODS`
  pub lod_count: u32,
  /// 每帧射线总预算（`DDGI_RAY_BUDGET`），seal 用它算 rpp
  pub ray_budget: u32,
  /// worklist 打包的 cell 下标位宽（`DDGI_WL_IDX_MASK`）
  pub wl_idx_mask: u32,
  /// worklist 打包的 lod 位偏移（`DDGI_WL_LOD_SHIFT`）
  pub wl_lod_shift: u32,
  /// `ddgi_args` 里 cast / collect indirect args 段的起始 word
  pub indir_cast_base: u32,
  pub indir_coll_base: u32,
  /// `ddgi_indirect` 里 rpp / 活跃探针计数器段的起始 word
  pub indir_rpp_base: u32,
  pub indir_count_base: u32,
}

impl DdgiConsts {
  /// 图集容量（可寻址的探针槽位上限）= 层数 × 层内每轴探针数的平方。
  #[inline]
  pub fn atlas_capacity(&self) -> u32 {
    self.atlas_layers * self.probes_per_layer_axis * self.probes_per_layer_axis
  }
  /// `ddgi_args`（独立 buffer，BG6）里 cast / collect indirect args 的**字节**偏移。
  #[inline]
  pub fn args_cast_offset(&self) -> u64 {
    self.indir_cast_base as u64 * 4
  }
  #[inline]
  pub fn args_coll_offset(&self) -> u64 {
    self.indir_coll_base as u64 * 4
  }
  /// `ddgi_indirect` 的字节数：覆盖到最后要写的 word（计数器段 + 每 LOD 一个）。
  #[inline]
  pub fn indirect_bytes(&self) -> u64 {
    (self.indir_count_base + self.lod_count) as u64 * 4
  }
  /// 每帧清零「活跃探针计数器」的字节偏移 / 字节数（seal 之前的 prepare 做）。
  #[inline]
  pub fn counter_clear_offset(&self) -> u64 {
    self.indir_count_base as u64 * 4
  }
  #[inline]
  pub fn counter_clear_bytes(&self) -> u64 {
    self.lod_count as u64 * 4
  }
  /// 射线样本缓冲的字节数。**不是** `ray_budget × 32`：seal 的 `rpp` 有下限 1，
  /// 活跃探针数超过预算时 `total_ray = total_active`，上界是总槽位数（见 WESL seal 注释）。
  #[inline]
  pub fn sample_bytes(&self, total_slots: u32) -> u64 {
    self.ray_budget.max(total_slots) as u64 * 32
  }
}

/// 需要的全部常量名（缺一个就 fail fast；一次报全，便于排查）。
const REQUIRED: &[&str] = &[
  "DDGI_IRR_TEXELS",
  "DDGI_DEPTH_TEXELS",
  "DDGI_PROBES_PER_LAYER_AXIS",
  "DDGI_ATLAS_LAYERS",
  "DDGI_LOD_COUNT",
  "DDGI_RAY_BUDGET",
  "DDGI_WL_IDX_MASK",
  "DDGI_WL_LOD_SHIFT",
  "DDGI_INDIR_CAST_BASE",
  "DDGI_INDIR_COLL_BASE",
  "DDGI_INDIR_RPP_BASE",
  "DDGI_INDIR_COUNT_BASE",
];

/// 解析 WESL 包里的跨端常量。首次调用读盘，之后走 `OnceLock`（每帧只多一次原子读）。
///
/// 失败（文件读不到 / 常量缺失 / 不是字面量）→ `error!` + `panic!`。
pub fn ddgi_consts() -> &'static DdgiConsts {
  static CONSTS: OnceLock<DdgiConsts> = OnceLock::new();
  CONSTS.get_or_init(DdgiConsts::load)
}

impl DdgiConsts {
  fn load() -> Self {
    let dir = Path::new(DDA_WESL_DIR);
    let values = parse_package_u32_consts(dir);
    let missing: Vec<&str> = REQUIRED
      .iter()
      .copied()
      .filter(|name| !values.contains_key(*name))
      .collect();
    if !missing.is_empty() {
      let msg = format!(
        "WESL 跨端常量缺失（{}）：{:?}\n\
         —— 权威值只写在 .wesl 里，Rust 从源码解析；请检查 ddgi/consts.wesl 与 bindings.wesl，\
         并确保它们写成 `const NAME: u32 = <字面量>u;` 形式。",
        DDA_WESL_DIR,
        missing
      );
      error!("{msg}");
      panic!("{msg}");
    }
    let get = |name: &str| -> u32 { values[name] };
    let out = DdgiConsts {
      irr_texels: get("DDGI_IRR_TEXELS"),
      depth_texels: get("DDGI_DEPTH_TEXELS"),
      probes_per_layer_axis: get("DDGI_PROBES_PER_LAYER_AXIS"),
      atlas_layers: get("DDGI_ATLAS_LAYERS"),
      lod_count: get("DDGI_LOD_COUNT"),
      ray_budget: get("DDGI_RAY_BUDGET"),
      wl_idx_mask: get("DDGI_WL_IDX_MASK"),
      wl_lod_shift: get("DDGI_WL_LOD_SHIFT"),
      indir_cast_base: get("DDGI_INDIR_CAST_BASE"),
      indir_coll_base: get("DDGI_INDIR_COLL_BASE"),
      indir_rpp_base: get("DDGI_INDIR_RPP_BASE"),
      indir_count_base: get("DDGI_INDIR_COUNT_BASE"),
    };
    // LOD 级数在 Rust 侧要定 `[T; DDGI_LODS]` 数组与 `ShaderType` 布局的长度（编译期常量，
    // 拿不到运行期解析结果）→ 这里做一次 fail-fast 断言，保证两侧一致。
    if out.lod_count != crate::ddgi::DDGI_LODS {
      let msg = format!(
        "WESL DDGI_LOD_COUNT = {} 与 Rust ddgi::DDGI_LODS = {} 不一致：\
         Rust 用它定数组长度与 uniform 布局（编译期常量），无法跟随运行期解析；\
         改级数必须同时改这两处。",
        out.lod_count,
        crate::ddgi::DDGI_LODS
      );
      error!("{msg}");
      panic!("{msg}");
    }
    if out.atlas_layers == 0 || out.probes_per_layer_axis == 0 || out.irr_texels == 0
      || out.depth_texels == 0
    {
      let msg = format!("WESL 跨端常量存在 0 值（图集尺寸会退化成空纹理）：{out:?}");
      error!("{msg}");
      panic!("{msg}");
    }
    info!(
      target: "gate",
      "WESL 跨端常量（源 {DDA_WESL_DIR}）：irr={} depth={} 层内轴={} 层数={} 容量={} \
       LOD={} 射线预算={} WL 掩码=0x{:X}/shift={} indirect word={}/{}/{}/{}",
      out.irr_texels, out.depth_texels, out.probes_per_layer_axis, out.atlas_layers,
      out.atlas_capacity(), out.lod_count, out.ray_budget, out.wl_idx_mask, out.wl_lod_shift,
      out.indir_cast_base, out.indir_coll_base, out.indir_rpp_base, out.indir_count_base,
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

/// 解析整个包的全部 `const NAME: u32 = <字面量>;`（同名以**先遇到的**为准，包内不应重名）。
fn parse_package_u32_consts(dir: &Path) -> HashMap<String, u32> {
  let mut files = Vec::new();
  collect_wesl_files(dir, &mut files);
  files.sort(); // 目录遍历顺序不保证稳定 → 排序后同名冲突的取用顺序也是确定的
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

/// 从一段 WGSL/WESL 源码里抽全部 `const NAME: u32 = <字面量>;`（见 [`parse_u32_const_line`]）。
///
/// 公开出来是为了让 `tests/wgsl_compile.rs` 能用**同一个解析器**去扫 WESL 编译产物
/// （展平后的 WGSL），从而端到端断言"Rust 解析到的值 == shader 实际用的值"。
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
///
/// 只认**字面量**（十进制 / `0x` 十六进制，可带 `u` 后缀）——派生式（如
/// `const DDGI_PROBES_PER_LAYER: u32 = A * A;`）返回 `None`：Rust 不需要它们（能从已解析的
/// 基础量自己算），而误当字面量解析会得到错值。行尾 `//` 注释会被先剥掉。
fn parse_u32_const_line(raw: &str) -> Option<(String, u32)> {
  let line = raw
    .trim()
    .trim_start_matches('\u{feff}')
    .split("//")
    .next()?
    .trim();
  let rest = line.strip_prefix("const ")?.trim_start();
  let (name, rest) = rest.split_once(':')?;
  let name = name.trim();
  if name.is_empty() || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
    return None;
  }
  let rest = rest.trim_start().strip_prefix("u32")?.trim_start();
  let rest = rest.strip_prefix('=')?.trim();
  let rest = rest.strip_suffix(';')?.trim(); // 必须以分号结尾：杜绝"跨行声明"被误读
  let rest = rest.strip_suffix('u').unwrap_or(rest).trim();
  let value = match rest.strip_prefix("0x").or_else(|| rest.strip_prefix("0X")) {
    Some(hex) => u32::from_str_radix(hex, 16).ok()?,
    None => rest.parse::<u32>().ok()?,
  };
  Some((name.to_string(), value))
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn parses_literals_and_skips_derived() {
    assert_eq!(
      parse_u32_const_line("const DDGI_IRR_TEXELS: u32 = 4u;"),
      Some(("DDGI_IRR_TEXELS".to_string(), 4))
    );
    // 十六进制 + 行尾注释
    assert_eq!(
      parse_u32_const_line("const DDGI_WL_IDX_MASK: u32 = 0x7FFFFu; // 19 位"),
      Some(("DDGI_WL_IDX_MASK".to_string(), 0x7FFFF))
    );
    // 派生式不是字面量 → 必须跳过（否则会解析出错值）
    assert_eq!(
      parse_u32_const_line("const DDGI_PROBES_PER_LAYER: u32 = DDGI_PROBES_PER_LAYER_AXIS * DDGI_PROBES_PER_LAYER_AXIS;"),
      None
    );
    // 注释里的示例、其它类型、非 const 行都不能被当成常量
    assert_eq!(parse_u32_const_line("// const FAKE: u32 = 1u;"), None);
    assert_eq!(parse_u32_const_line("const DDGI_ALPHA: f32 = 0.015;"), None);
    assert_eq!(parse_u32_const_line("let x = 1u;"), None);
  }

  /// 跑一次完整加载（等价于 CI 里检查 WESL 侧没被改坏）：常量都在、级数与 Rust 一致、
  /// 图集容量装得进 worklist 的下标位宽、indirect word 布局自洽。
  #[test]
  fn load_cross_boundary_consts_succeeds() {
    let c = ddgi_consts();
    assert_eq!(c.lod_count, crate::ddgi::DDGI_LODS);
    assert!(c.atlas_capacity() > 0);
    // worklist 的 cell 下标位宽必须装得下图集容量 —— 否则 LOD0 的下标被截断、成片全黑。
    assert!(
      c.atlas_capacity() <= c.wl_idx_mask + 1,
      "图集容量 {} 超出 worklist 下标位宽 0x{:X}",
      c.atlas_capacity(),
      c.wl_idx_mask
    );
    // indirect 缓冲的 word 布局必须自洽（rpp / 计数器段在 args 段之后）
    assert!(c.indir_count_base >= c.indir_rpp_base);
    assert!(c.indir_rpp_base >= c.indir_coll_base);
  }
}
