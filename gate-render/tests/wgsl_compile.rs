//! P3.1 WGSL 编译期校验：用 naga（bevy_render 同版本）parse + validate 全部 shader 资产。
//! shader 语法/类型错误在运行时才暴露（黑屏 + 日志），此测试把失败提前到 CI。
//!
//! 注意：wgpu 实际编译链 = naga → SPIR-V（DXC/FXC 再翻译），此处 validate 覆盖
//! naga 前端与验证器；平台后端差异仍由实机 F5 验收兜底。

use std::path::PathBuf;

fn manifest_dir() -> PathBuf {
  PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn compile_wgsl(rel_path: &str) {
  let path = manifest_dir().join("../gate-app/assets").join(rel_path);
  let src = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{rel_path} 读取失败: {e}"));
  let module = naga::front::wgsl::parse_str(&src)
    .unwrap_or_else(|e| panic!("{rel_path} naga parse 失败: {e:?}"));
  let mut validator = naga::valid::Validator::new(
    naga::valid::ValidationFlags::all(),
    naga::valid::Capabilities::all(),
  );
  let info = validator
    .validate(&module)
    .unwrap_or_else(|e| panic!("{rel_path} naga validate 失败: {e:?}"));
  let _ = info;
}

#[test]
fn wgsl_shaders_parse_and_validate() {
  compile_wgsl("shaders/dda.wgsl");
  compile_wgsl("shaders/blit.wgsl");
}

/// `SHADOW_SURFACE_EPS` 是「阴影/GI 起点推过体素边界」的量化契约，必须与 Rust 侧常量
/// 逐字一致。值 ≤0 会让 -X/-Y/-Z 面的阴影射线在 t=0 自命中出发点体素，而 dda_main 把
/// 自命中当「无遮挡」→ 封闭空间漏直射太阳光（室内明暗完全跟着面朝向走）。
/// CPU 侧的行为由 `brickmap::dda::dda_ref_tests::shadow_ray_start_offset_clears_hit_voxel` 锁定。
#[test]
fn shadow_surface_eps_matches_rust_const() {
  let path = manifest_dir().join("../gate-app/assets/shaders/dda.wgsl");
  let src = std::fs::read_to_string(&path).expect("读 dda.wgsl");
  let expect = format!(
    "const SHADOW_SURFACE_EPS: f32 = {};",
    gate_render::brickmap::dda::wgsl_consts::SHADOW_SURFACE_EPS
  );
  assert!(src.contains(&expect), "dda.wgsl 中未找到 `{expect}`");
}
