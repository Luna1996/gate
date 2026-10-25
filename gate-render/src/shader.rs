use std::path::{Path, PathBuf};

use bevy::prelude::*;
use bevy::shader::Shader;

use crate::paths::dda_wesl_dir;

pub const DDA_SHADER_PATH: &str = "shaders/voxel_raytrace/main.wesl";

#[derive(Resource, Clone)]
pub struct DdaShaderHandle(pub Handle<Shader>);

pub fn compile_dda_wesl() -> Result<String, wesl::Error> {
  compile_wesl_entry(dda_wesl_dir().join("main.wesl"))
}

pub fn compile_wesl_entry(entry: impl AsRef<Path>) -> Result<String, wesl::Error> {
  let entry = entry.as_ref();
  let dir: PathBuf = entry.parent().map(Path::to_path_buf).unwrap_or_else(|| PathBuf::from("."));
  let stem = entry.file_stem().and_then(|s| s.to_str()).unwrap_or("package");

  let module_path =
    if stem == "package" { "package".to_string() } else { format!("package::{stem}") };

  let options =
    wesl::CompileOptions { mangler: wesl::ManglerKind::None, validate: true, ..Default::default() };
  wesl::Compiler::new(options)
    .compile_module(&dir, &module_path.parse().expect("WESL 模块路径字面量合法"))
    .map(|result| result.to_string())
}

pub fn build_dda_shader(app: &mut App) {
  let t0 = std::time::Instant::now();
  let source = match compile_dda_wesl() {
    Ok(source) => source,
    Err(e) => {
      let msg = format!("DDA WESL 编译失败（{}）:\n{e}", dda_wesl_dir().display());
      error!("{msg}");
      panic!("{msg}");
    }
  };
  info!(
    target: "gate",
    "WESL→WGSL：{} ms（{} KB）",
    t0.elapsed().as_millis(),
    source.len() / 1024
  );

  crate::wesl_consts::gi_consts();
  crate::wesl_consts::trace_consts();
  let handle = app
    .world_mut()
    .resource_mut::<Assets<Shader>>()
    .add(Shader::from_wgsl(source, DDA_SHADER_PATH));
  app.insert_resource(DdaShaderHandle(handle));
}
