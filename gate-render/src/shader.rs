//! WESL（WGSL 超集，带 import）编译：启动时把 `assets/shaders/voxel_raytrace/` 下的 WESL 包
//! 编译成单份 WGSL 插入 `Assets<Shader>`，之后走 Bevy 常规 WGSL/naga 路径。
//! 改 `.wesl` 后重启 app 即生效。
//!
//! 同一份源码出**两版**（[`DdaShaderHandle`] / [`DdaShaderRtHandle`]），靠 WESL 的条件编译
//! （`CompileOptions::features` 里的 `ray_query` 开关，见 `trace.wesl` 的 `@if(ray_query)`）。
//! 为什么必须是两版而不是运行期 `if`：`enable wgpu_ray_query;` 会让**没有该特性的设备连 shader module
//! 都建不出来**（不是运行期失败，是 `create_shader_module` 直接报错）⇒ 软件版里不能出现它。

use std::path::{Path, PathBuf};

use bevy::prelude::*;
use bevy::shader::Shader;

use crate::paths::dda_wesl_dir;

/// dda shader 的逻辑资产路径（`Shader::path`，仅用于日志/诊断）。
pub const DDA_SHADER_PATH: &str = "shaders/voxel_raytrace/main.wesl";
/// RT 版（同一份源码、`ray_query` 开）的逻辑路径。
pub const DDA_SHADER_RT_PATH: &str = "shaders/voxel_raytrace/main.wesl#ray_query";

/// 条件编译的 feature flag 名：与 `trace.wesl` / `bindings.wesl` 里的 `@if(ray_query)` 逐字一致。
const FEATURE_RAY_QUERY: &str = "ray_query";

/// 已编译 dda shader 的资产 handle（**软件 DDA 版**）。main world 侧保活、render world 侧供 pipeline 创建读取。
#[derive(Resource, Clone)]
pub struct DdaShaderHandle(pub Handle<Shader>);

/// **RT 版**（`@if(ray_query)` 打开、含 `enable wgpu_ray_query;`）的资产 handle。
///
/// 非 RT 设备上它**永远不被 pipeline 引用** ⇒ Bevy 不会为它建 `ShaderModule` ⇒ 不会因缺特性而失败
/// （Bevy 是"被 `queue_*_pipeline` 引用时才建 module"）。
#[derive(Resource, Clone)]
pub struct DdaShaderRtHandle(pub Handle<Shader>);

/// 编译 DDA 的 WESL 包，返回展平后的 WGSL。`ray_query = true` ⇒ 打开硬件光追分支。
pub fn compile_dda_wesl(ray_query: bool) -> Result<String, wesl::Error> {
  compile_wesl_entry(dda_wesl_dir().join("main.wesl"), ray_query)
}

/// 编译 `entry` 所在目录的 WESL 包，返回展平后的 WGSL。
/// 入口文件 `<stem>.wesl` 对应模块 `package::<stem>`（走 `compile_module(包目录, 模块路径)`）。
pub fn compile_wesl_entry(entry: impl AsRef<Path>, ray_query: bool) -> Result<String, wesl::Error> {
  let entry = entry.as_ref();
  let dir: PathBuf = entry.parent().map(Path::to_path_buf).unwrap_or_else(|| PathBuf::from("."));
  let stem = entry.file_stem().and_then(|s| s.to_str()).unwrap_or("package");

  let module_path =
    if stem == "package" { "package".to_string() } else { format!("package::{stem}") };

  // 未指定的 flag 按 `Feature::Disable` 处理 ⇒ 软件版什么都不用设。
  let mut features = wesl::pass::Features::default();
  features.set(FEATURE_RAY_QUERY, ray_query);

  let options = wesl::CompileOptions {
    // 禁用名字改写（mangler）：Rust 侧按源码里的常量名 / 入口点名引用，不能被 mangle。
    mangler: wesl::ManglerKind::None,
    // WESL 自带的符号校验表**不含 `rayQuery*` 这类内建**（只报 `UndefinedSymbol`）⇒ RT 版关掉它。
    // 这一版的安全性由下游兜住：`tests/wesl_package.rs` 用 naga 校验产物（`Capabilities::all()`），
    // 运行期 wgpu 建 module 时还会再校验一次 —— 那是权威口径。
    validate: !ray_query,
    features,
    ..Default::default()
  };
  wesl::Compiler::new(options)
    .compile_module(&dir, &module_path.parse().expect("WESL 模块路径字面量合法"))
    .map(|result| result.to_string())
}

/// 编译 dda WESL 包的**两版**并作为 `Shader` 资产插入 `Assets<Shader>`。
/// 编译失败 → `error!` 打印 pretty 诊断后 `panic!`（fail fast）。
pub fn build_dda_shader(app: &mut App) {
  let source = match compile_dda_wesl(false) {
    Ok(source) => source,
    Err(e) => {
      let msg = format!("DDA WESL 编译失败（{}）:\n{e}", dda_wesl_dir().display());
      error!("{msg}");
      panic!("{msg}");
    }
  };
  let source_rt = match compile_dda_wesl(true) {
    Ok(source) => source,
    Err(e) => {
      let msg = format!("DDA WESL 编译失败（ray_query 版，{}）:\n{e}", dda_wesl_dir().display());
      error!("{msg}");
      panic!("{msg}");
    }
  };

  // 跨端常量 fail fast：编译通过但解析不到常量（写成派生式 / 改名）在启动时就该炸，而不是运行期静默失效
  crate::wesl_consts::gi_consts();
  crate::wesl_consts::trace_consts();
  let handle = app
    .world_mut()
    .resource_mut::<Assets<Shader>>()
    .add(Shader::from_wgsl(source, DDA_SHADER_PATH));
  let handle_rt = app
    .world_mut()
    .resource_mut::<Assets<Shader>>()
    .add(Shader::from_wgsl(source_rt, DDA_SHADER_RT_PATH));
  app.insert_resource(DdaShaderHandle(handle));
  app.insert_resource(DdaShaderRtHandle(handle_rt));
}
