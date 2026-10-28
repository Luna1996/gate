use std::path::{Path, PathBuf};

const ENTRY_POINTS: &[&str] = &[
  "dda_main",
  "dda_preview",
  "dda_visible",
  "dda_shadow",
  "dda_shade",
  "dda_glass",
  "dda_refl",
  "dda_refl_compose",
  "dda_face_accum",
  "dda_face_main",
  "beam_main",
  "gi_main",
  "gi_wal_update",
  "gi_face_flatten",
  "gi_denoise_temporal",
  "gi_denoise_atrous1",
  "gi_denoise_atrous2",
  "gi_denoise_atrous4",
  "gi_denoise_atrous8",
  "gi_denoise_atrous16",
  "godray_main",
  "godray_blur_a",
  "godray_blur_b",
  "eye_adapt_histogram",
  "eye_adapt_update",
];

fn main() {
  forbid_debug_build();

  let root = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"))
    .join("../assets/shaders/voxel_raytrace");
  println!("cargo:rerun-if-changed={}", root.display());
  rerun_if_changed(&root);

  let src = compile(&root).unwrap_or_else(|e| panic!("WESL 编译失败（{}）:\n{e}", root.display()));
  validate(&src);
}

fn rerun_if_changed(dir: &Path) {
  let entries = std::fs::read_dir(dir)
    .unwrap_or_else(|e| panic!("读取 WESL 目录失败（{}）: {e}", dir.display()));
  for entry in entries {
    let path = entry.expect("读取 WESL 目录项").path();
    if path.is_dir() {
      rerun_if_changed(&path);
    } else if path.extension().is_some_and(|e| e == "wesl") {
      println!("cargo:rerun-if-changed={}", path.display());
    }
  }
}

fn compile(dir: &Path) -> Result<String, wesl::Error> {
  let options =
    wesl::CompileOptions { mangler: wesl::ManglerKind::None, validate: true, ..Default::default() };
  wesl::Compiler::new(options)
    .compile_module(dir, &"package::main".parse().expect("WESL 模块路径合法"))
    .map(|result| result.to_string())
}

fn validate(src: &str) {
  let module = naga::front::wgsl::parse_str(src)
    .unwrap_or_else(|e| panic!("WESL→WGSL 解析失败:\n{}", e.emit_to_string(src)));
  naga::valid::Validator::new(
    naga::valid::ValidationFlags::all(),
    naga::valid::Capabilities::all(),
  )
  .validate(&module)
  .unwrap_or_else(|e| panic!("WESL→WGSL 校验失败: {e}"));

  for name in ENTRY_POINTS {
    assert!(module.entry_points.iter().any(|e| e.name == *name), "WESL 产物缺少入口 {name}");
  }
}

fn forbid_debug_build() {
  if std::env::var("PROFILE").as_deref() == Ok("debug") {
    panic!(
      "禁止 debug 构建：本仓库只允许 release 系 profile，请用 `--profile build-fast` / `balance` / `run-fast`：\n\
       cargo build / run / check / test --profile build-fast\n\
       cargo clippy --profile build-fast --workspace --all-targets -- -D warnings"
    );
  }
}
