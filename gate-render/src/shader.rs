//! WESL（WGSL 超集，带 import）编译：启动时把 `assets/shaders/voxel_raytrace/` 下的 WESL 包
//! 编译成单份 WGSL 插入 `Assets<Shader>`，之后走 Bevy 常规 WGSL/naga 路径。
//! 改 `.wesl` 后重启 app 即生效。

use std::path::{Path, PathBuf};

use bevy::prelude::*;
use bevy::shader::Shader;

use crate::paths::dda_wesl_dir;

/// dda shader 的逻辑资产路径（`Shader::path`，仅用于日志/诊断）。
pub const DDA_SHADER_PATH: &str = "shaders/voxel_raytrace/main.wesl";

/// 已编译 dda shader 的资产 handle。main world 侧保活、render world 侧供 pipeline 创建读取。
#[derive(Resource, Clone)]
pub struct DdaShaderHandle(pub Handle<Shader>);

/// 编译 DDA 的 WESL 包，返回展平后的 WGSL。
pub fn compile_dda_wesl() -> Result<String, wesl::Error> {
  compile_wesl_entry(dda_wesl_dir().join("main.wesl"))
}

/// 编译 `entry` 所在目录的 WESL 包，返回展平后的 WGSL。
/// 入口文件 `<stem>.wesl` 对应模块 `package::<stem>`（走 `compile_module(包目录, 模块路径)`）。
pub fn compile_wesl_entry(entry: impl AsRef<Path>) -> Result<String, wesl::Error> {
  let entry = entry.as_ref();
  let dir: PathBuf = entry.parent().map(Path::to_path_buf).unwrap_or_else(|| PathBuf::from("."));
  let stem = entry.file_stem().and_then(|s| s.to_str()).unwrap_or("package");

  let module_path =
    if stem == "package" { "package".to_string() } else { format!("package::{stem}") };

  let options = wesl::CompileOptions {
    // 禁用名字改写（mangler）：Rust 侧按源码里的常量名 / 入口点名引用，不能被 mangle。
    mangler: wesl::ManglerKind::None,
    ..Default::default()
  };
  wesl::Compiler::new(options)
    .compile_module(&dir, &module_path.parse().expect("WESL 模块路径字面量合法"))
    .map(|result| result.to_string())
}

/// 编译 dda WESL 包并作为 `Shader` 资产插入 `Assets<Shader>`，注册 [`DdaShaderHandle`]。
/// 编译失败 → `error!` 打印 pretty 诊断后 `panic!`（fail fast）。
pub fn build_dda_shader(app: &mut App) {
  let source = match compile_dda_wesl() {
    Ok(source) => source,
    Err(e) => {
      let msg = format!("DDA WESL 编译失败（{}）:\n{e}", dda_wesl_dir().display());
      error!("{msg}");
      panic!("{msg}");
    }
  };

  crate::wesl_consts::gi_consts();
  let handle = app
    .world_mut()
    .resource_mut::<Assets<Shader>>()
    .add(Shader::from_wgsl(source, DDA_SHADER_PATH));
  app.insert_resource(DdaShaderHandle(handle));
}
