//! WGSL 编译期校验：用 naga（bevy_render 同版本）parse + validate shader。
//! shader 语法/类型错误在运行时才暴露（黑屏 + 日志），此测试把失败提前到 CI。
//!
//! dda shader 已拆成 WESL 包（`gate-app/assets/shaders/voxel_raytrace/`），由 `wesl-rs`
//! 运行时编译回单份 WGSL（见 `gate_render::shader::compile_dda_wesl`）；本测试
//! 编译同一份包并校验产物，等价于运行时会交给 Bevy 的 WGSL。
//!
//! 注意：wgpu 实际编译链 = naga → SPIR-V（DXC/FXC 再翻译），此处 validate 覆盖
//! naga 前端与验证器；平台后端差异仍由实机 F5 验收兜底。

use std::path::PathBuf;

fn manifest_dir() -> PathBuf {
  PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn asset_path(rel: &str) -> PathBuf {
  manifest_dir().join("../gate-app/assets").join(rel)
}

/// `dda` WESL 包的常量声明文件（Rust 侧契约常量都在这里逐字声明）。
const DDA_CONSTS_WESL: &str = "shaders/voxel_raytrace/ddgi/consts.wesl";

fn validate_wgsl(label: &str, src: &str) -> naga::Module {
  let module =
    naga::front::wgsl::parse_str(src).unwrap_or_else(|e| panic!("{label} naga parse 失败: {e:?}"));
  let mut validator = naga::valid::Validator::new(
    naga::valid::ValidationFlags::all(),
    naga::valid::Capabilities::all(),
  );
  validator
    .validate(&module)
    .unwrap_or_else(|e| panic!("{label} naga validate 失败: {e:?}"));
  module
}

#[test]
fn wgsl_shaders_parse_and_validate() {
  let dda = gate_render::shader::compile_dda_wesl().expect("dda WESL 包编译失败");
  validate_wgsl("dda (WESL)", &dda);

  let blit_path = asset_path("shaders/blit.wgsl");
  let blit = std::fs::read_to_string(&blit_path).expect("读 blit.wgsl");
  validate_wgsl("blit.wgsl", &blit);
}

/// pipeline 以**字符串**引用 entry point 名（`dda_main` 等）；WESL 的名字改写
/// （mangler）或拆分失误都会让 pipeline 找不到入口，且只在运行时以黑屏/日志体现。
/// 此测试直接对编译产物的 entry point 列表做精确断言。
#[test]
fn dda_entry_points_are_preserved() {
  let dda = gate_render::shader::compile_dda_wesl().expect("dda WESL 包编译失败");
  let module = validate_wgsl("dda (WESL)", &dda);
  let mut names: Vec<&str> = module
    .entry_points
    .iter()
    .map(|e| e.name.as_str())
    .collect();
  names.sort_unstable();
  assert_eq!(
    names,
    [
      "beam_main",
      "dda_main",
      "ddgi_bake0",
      "ddgi_bake1",
      "ddgi_bake2",
      "ddgi_bake3",
      "ddgi_cast",
      "ddgi_collect",
      "ddgi_seal",
      "ddgi_sort",
      "eye_adapt_histogram",
      "eye_adapt_update",
      "probe_viz_main",
    ],
    "dda entry point 集合与 pipeline 期望不符"
  );
}

/// `SHADOW_SURFACE_EPS` 是「阴影/GI 起点推过体素边界」的量化契约，必须与 Rust 侧常量
/// 逐字一致。值 ≤0 会让 -X/-Y/-Z 面的阴影射线在 t=0 自命中出发点体素，而 dda_main 把
/// 自命中当「无遮挡」→ 封闭空间漏直射太阳光（室内明暗完全跟着面朝向走）。
/// CPU 侧的行为由 `brickmap::dda::dda_ref_tests::shadow_ray_start_offset_clears_hit_voxel` 锁定。
#[test]
fn shadow_surface_eps_matches_rust_const() {
  let src = std::fs::read_to_string(asset_path(DDA_CONSTS_WESL)).expect("读 ddgi/consts.wesl");
  let expect = format!(
    "const SHADOW_SURFACE_EPS: f32 = {};",
    gate_render::brickmap::dda::wgsl_consts::SHADOW_SURFACE_EPS
  );
  assert!(
    src.contains(&expect),
    "ddgi/consts.wesl 中未找到 `{expect}`"
  );
}

/// DDGI 的跨端常量（图集纹素数 / 层内轴 / 层数 / 级数 / 射线预算 / indirect word 布局）**以
/// WESL 为单一来源**：Rust 侧不再声明副本，而是启动时解析 `.wesl` 源码（`wesl_consts`）。
///
/// 这里做的是**端到端**校验：拿编译产物（展平后的 WGSL，即实际交给 GPU 的那份）再解析一遍，
/// 与 `ddgi_consts()` 逐项对齐 —— 覆盖"解析器和 WESL 编译器看到的不是同一份/同一批常量"
/// 这类漂移（例如常量写成了派生式、或搬到了解析器扫不到的目录）。
/// 历史症状：`DDGI_PROBES_PER_LAYER_AXIS` 一边 40 一边 16 → 大部分探针写到图集外（大面积无
/// GI）；`DDGI_RAY_BUDGET` 一边 131072 一边 65536 → rpp 静默减半（室内噪声变大）。
#[test]
fn ddgi_cross_boundary_consts_match_compiled_wgsl() {
  let dda = gate_render::shader::compile_dda_wesl().expect("dda WESL 包编译失败");
  let used =
    gate_render::wesl_consts::parse_u32_consts_in_source(&dda);
  let c = gate_render::wesl_consts::ddgi_consts();
  // DDGI_ATLAS_LAYERS 不参与 WESL 的寻址（只有 Rust 用它定纹理层数）→ 展平后可能被剔除，
  // 不在此断言；其余常量都是 shader 真正要用的，必须逐字一致。
  for (name, want) in [
    ("DDGI_IRR_TEXELS", c.irr_texels),
    ("DDGI_DEPTH_TEXELS", c.depth_texels),
    ("DDGI_PROBES_PER_LAYER_AXIS", c.probes_per_layer_axis),
    ("DDGI_LOD_COUNT", c.lod_count),
    ("DDGI_RAY_BUDGET", c.ray_budget),
    ("DDGI_WL_IDX_MASK", c.wl_idx_mask),
    ("DDGI_WL_LOD_SHIFT", c.wl_lod_shift),
    ("DDGI_INDIR_CAST_BASE", c.indir_cast_base),
    ("DDGI_INDIR_COLL_BASE", c.indir_coll_base),
    ("DDGI_INDIR_RPP_BASE", c.indir_rpp_base),
    ("DDGI_INDIR_COUNT_BASE", c.indir_count_base),
  ] {
    assert_eq!(
      used.get(name).copied(),
      Some(want),
      "编译产物里的 `{name}` 与 Rust 解析值不一致：Rust 侧按它分配显存/算偏移，\
       不一致会静默错位（症状：大面积无 GI、或 rpp 异常）"
    );
  }
}

/// 每层探针数必须**派生**自 AXIS（写死的 256=16² 正是那次"探针读到空白"错位的另一半原因），
/// 且 `DDGI_LOD_COUNT` 必须等于 Rust 的 `DDGI_LODS`（后者要定 `[T; N]` 数组与 uniform 布局的
/// 长度，编译期常量无法来自运行期解析，由 `wesl_consts::load` 断言）。
#[test]
fn ddgi_derived_consts_and_lod_count_are_consistent() {
  let src = std::fs::read_to_string(asset_path(DDA_CONSTS_WESL)).expect("读 ddgi/consts.wesl");
  let derived =
    "const DDGI_PROBES_PER_LAYER: u32 = DDGI_PROBES_PER_LAYER_AXIS * DDGI_PROBES_PER_LAYER_AXIS;";
  assert!(
    src.contains(derived),
    "ddgi/consts.wesl 的 DDGI_PROBES_PER_LAYER 应是派生式，未找到 `{derived}`"
  );
  let c = gate_render::wesl_consts::ddgi_consts();
  assert_eq!(c.lod_count, gate_render::ddgi::DDGI_LODS);
}

/// worklist `.w` 里 cell 下标的位宽必须装得下**图集容量**，且 lod 位段紧接在下标之后。
///
/// 该下标是「本级内的线性 cell 下标」，上限 = 该级 cell 数 ≤ 总槽位 ≤ 图集容量
/// （`DDGI_ATLAS_LAYERS × 层内轴²`）。曾经只用 16 位：dims 改成按世界 AABB 算之后 LOD0 有
/// 342576 个 cell，下标在 collect 侧被 `& 0xFFFF` 截断 → `atlas_slot` 折回低 65536 个槽位
/// → 一部分探针的图集被远处探针反复覆写、其余恒空 → **部分区域正常、部分区域全黑**，
/// 而**没有任何编译/运行时报错**。此测试把它挡在 CI。
#[test]
fn ddgi_worklist_pack_covers_atlas_capacity() {
  let c = gate_render::wesl_consts::ddgi_consts();
  let (mask, shift) = (c.wl_idx_mask, c.wl_lod_shift);
  let capacity = c.atlas_capacity();
  assert!(
    capacity <= mask + 1,
    "图集容量 {capacity} 超出 worklist cell 下标位宽 0x{mask:X} —— \
     LOD0 的 cell 下标会被截断，症状是部分区域全黑。调大 DDGI_WL_IDX_MASK / DDGI_WL_LOD_SHIFT"
  );
  // lod 位段必须紧跟在 age(8 位) + 下标之后
  assert_eq!(
    shift,
    8 + mask.count_ones(),
    "DDGI_WL_LOD_SHIFT 与 DDGI_WL_IDX_MASK 的位宽不匹配（应为 8 + 下标位宽）"
  );
}
