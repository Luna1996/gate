//! WESL（WGSL 超集，带 import）编译。
//!
//! voxel raytrace shader 按子系统拆成 `gate-app/assets/shaders/voxel_raytrace/` 下的
//! WESL 包，由 `wesl` crate 在**启动时**编译回单份 WGSL 字符串，再作为普通 `Shader`
//! 资产插入 `Assets<Shader>`（见 [`build_dda_shader`]），交给 Bevy 常规 WGSL/naga 路径。
//!
//! 为什么不用 Bevy 自带的 `shader_format_wesl`：其 shader loader 不会自动加载 WESL
//! import 依赖（Bevy 自己的 `load_shader_library` 宏文档已承认该限制），需手工预加载
//! 全部子模块，且缺失时会在 `ShaderCache` 里 `unwrap()` panic。
//!
//! 为什么不用自定义 `AssetLoader`（`.wesl` 扩展）：Bevy 的 `ShaderLoader::extensions()`
//! **无条件**包含 `"wesl"`，再注册一个同扩展名的 loader 会触发 "Duplicate AssetLoader"
//! 告警，并要求 `.meta` 才能消歧（虽然后注册者胜出，但依赖该顺序很脆）。直接插入资产
//! 则完全绕开 loader 注册表。
//!
//! 迭代方式：改任一 `.wesl` 后**重启 app** 即生效（无需 cargo 重编译；每次启动都重新
//! 读盘编译）。编译失败会在 `Plugin::build` 里 error + panic（含 pretty 诊断），不会静默黑屏。

use std::path::{Path, PathBuf};

use bevy::prelude::*;
use bevy::shader::Shader;

/// voxel raytrace WESL 包根目录（内含 `main.wesl` 与各子模块）。
pub const DDA_WESL_DIR: &str = concat!(
  env!("CARGO_MANIFEST_DIR"),
  "/../gate-app/assets/shaders/voxel_raytrace"
);

/// dda shader 的逻辑资产路径（`Shader::path`，仅用于日志/诊断）。
pub const DDA_SHADER_PATH: &str = "shaders/voxel_raytrace/main.wesl";

/// 已编译 dda shader 的资产 handle。main world 侧保活、render world 侧供 pipeline 创建读取。
#[derive(Resource, Clone)]
pub struct DdaShaderHandle(pub Handle<Shader>);

/// 编译 DDA 的 WESL 包，返回展平后的 WGSL。
pub fn compile_dda_wesl() -> Result<String, wesl::Error> {
  compile_wesl_entry(Path::new(DDA_WESL_DIR).join("main.wesl"))
}

/// 编译某个 `.wesl` 入口文件所属的包，返回展平后的 WGSL。
///
/// `entry` 所在目录即 WESL 包根目录；入口文件 `main.wesl` 对应模块 `package::main`。
/// 显式走 `compile_module(包目录, 模块路径)` 而不是 `compile(文件路径)`：后者的
/// 「文件路径 → 模块路径」推断在 0.5 里对根模块有歧义（`ModulePath::new_root()`），
/// 而具名入口 `package::main` 与我们的包布局一一对应，语义确定。
pub fn compile_wesl_entry(entry: impl AsRef<Path>) -> Result<String, wesl::Error> {
  let entry = entry.as_ref();
  let dir: PathBuf = entry
    .parent()
    .map(Path::to_path_buf)
    .unwrap_or_else(|| PathBuf::from("."));
  let stem = entry
    .file_stem()
    .and_then(|s| s.to_str())
    .unwrap_or("package");
  // 入口模块：包根目录下 `<stem>.wesl`（约定为 `main.wesl` → `package::main`）。
  let module_path = if stem == "package" {
    "package".to_string()
  } else {
    format!("package::{stem}")
  };

  let options = wesl::CompileOptions {
    // 拆分是纯重构：禁用名字改写，保证 Rust 侧约定常量/入口点名与从前逐字一致。
    mangler: wesl::ManglerKind::None,
    ..Default::default()
  };
  wesl::Compiler::new(options)
    .compile_module(&dir, &module_path.parse().expect("WESL 模块路径字面量合法"))
    .map(|result| result.to_string())
}

/// 编译 dda WESL 包并作为 `Shader` 资产插入 `Assets<Shader>`，注册 [`DdaShaderHandle`]。
///
/// 编译失败：`error!` 打印 pretty 诊断后 `panic!`（fail fast —— 静默黑屏更难查；
/// CI 侧另有 `tests/wgsl_compile.rs` 提前拦截）。
pub fn build_dda_shader(app: &mut App) {
  let source = match compile_dda_wesl() {
    Ok(source) => source,
    Err(e) => {
      let msg = format!("DDA WESL 编译失败（{DDA_WESL_DIR}）:\n{e}");
      error!("{msg}");
      panic!("{msg}");
    }
  };
  let handle = app
    .world_mut()
    .resource_mut::<Assets<Shader>>()
    .add(Shader::from_wgsl(source, DDA_SHADER_PATH));
  app.insert_resource(DdaShaderHandle(handle));
}
