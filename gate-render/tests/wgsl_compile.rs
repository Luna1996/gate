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

/// DDGI 图集纹素数在 Rust 与 WGSL **两处独立声明**：Rust 侧决定纹理尺寸 / 显存 / collect
/// 线程数，WGSL 侧决定八面体方向分辨率与纹素坐标。任一侧漏改都会让两边错位 —— 典型症状是
/// 深度纹素写进**相邻探针**、或方向查到错纹理，表现为跟着探针投影走的成片锯齿/亮暗块，
/// 而且**不会**有编译错误。此测试把这类静默错位挡在 CI。
#[test]
fn ddgi_texels_match_rust_consts() {
  let src = std::fs::read_to_string(asset_path(DDA_CONSTS_WESL)).expect("读 ddgi/consts.wesl");
  for (name, val) in [
    ("DDGI_IRR_TEXELS", gate_render::ddgi::IRRADIANCE_TEXELS),
    ("DDGI_DEPTH_TEXELS", gate_render::ddgi::DEPTH_TEXELS),
  ] {
    let expect = format!("const {name}: u32 = {val}u;");
    assert!(
      src.contains(&expect),
      "ddgi/consts.wesl 中未找到 `{expect}`"
    );
  }
}

/// DDGI **图集布局**也在两处独立声明：Rust 决定纹理尺寸与「槽位 → 层」的映射，WGSL 决定
/// 采样时的 layer 内寻址（`ddgi_probe_in_layer`）。两者不一致 → 写进图集的探针**读不回来**，
/// 症状是**按网格边界切开的大面积无 GI**，且没有任何编译/运行时报错。
/// 已踩过一次：Rust 改成 40 而 WGSL 还是 16 → 大部分探针读到空白。此测试把它挡在 CI。
#[test]
fn ddgi_atlas_layout_matches_rust_consts() {
  let src = std::fs::read_to_string(asset_path(DDA_CONSTS_WESL)).expect("读 ddgi/consts.wesl");
  let axis = gate_render::ddgi::DDGI_ATLAS_PROBES_PER_LAYER_AXIS;
  let expect = format!("const DDGI_PROBES_PER_LAYER_AXIS: u32 = {axis}u;");
  assert!(
    src.contains(&expect),
    "ddgi/consts.wesl 中未找到 `{expect}`"
  );
  // 每层探针数必须**派生**自 AXIS（写死的 256=16² 正是那次错位的另一半原因）
  let derived =
    "const DDGI_PROBES_PER_LAYER: u32 = DDGI_PROBES_PER_LAYER_AXIS * DDGI_PROBES_PER_LAYER_AXIS;";
  assert!(
    src.contains(derived),
    "ddgi/consts.wesl 的 DDGI_PROBES_PER_LAYER 应是派生式，未找到 `{derived}`"
  );
}

/// worklist `.w` 里 cell 下标的位宽必须装得下**图集容量**。
///
/// 该下标是「本级内的线性 cell 下标」，上限 = 该级 cell 数 ≤ 总槽位 ≤ 图集容量
/// （`DDGI_ATLAS_LAYERS × AXIS²`）。曾经只用 16 位：dims 改成按世界 AABB 算之后 LOD0 有
/// 342576 个 cell，下标在 collect 侧被 `& 0xFFFF` 截断 → `atlas_slot` 折回低 65536 个槽位
/// → 一部分探针的图集被远处探针反复覆写、其余恒空 → **部分区域正常、部分区域全黑**，
/// 而**没有任何编译/运行时报错**。此测试把它挡在 CI。
#[test]
fn ddgi_worklist_pack_covers_atlas_capacity() {
  let src = std::fs::read_to_string(asset_path(DDA_CONSTS_WESL)).expect("读 ddgi/consts.wesl");
  const MASK: u32 = 0x7FFFF; // 19 位；必须与 ddgi/consts.wesl 的 DDGI_WL_IDX_MASK 一致
  let expect = format!("const DDGI_WL_IDX_MASK: u32 = 0x{MASK:X}u;");
  assert!(
    src.contains(&expect),
    "ddgi/consts.wesl 中未找到 `{expect}`"
  );
  let bits = MASK.count_ones();
  // lod 位段必须紧跟在下标之后（age 8 位 + 下标 bits 位）
  let lod_shift = format!("const DDGI_WL_LOD_SHIFT: u32 = {}u;", 8 + bits);
  assert!(
    src.contains(&lod_shift),
    "ddgi/consts.wesl 中未找到 `{lod_shift}`"
  );
  let capacity = gate_render::ddgi::DDGI_ATLAS_LAYERS
    * gate_render::ddgi::DDGI_ATLAS_PROBES_PER_LAYER_AXIS
    * gate_render::ddgi::DDGI_ATLAS_PROBES_PER_LAYER_AXIS;
  assert!(
    capacity <= MASK + 1,
    "图集容量 {capacity} 超出 worklist cell 下标位宽（{bits} 位）—— \
     LOD0 的 cell 下标会被截断，症状是部分区域全黑。调大 DDGI_WL_IDX_MASK / DDGI_WL_LOD_SHIFT"
  );
}
