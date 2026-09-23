//! WESL 包的**编译期**守门：把 `assets/shaders/voxel_raytrace/` 编译成 WGSL 并用 naga 校验。
//!
//! 为什么需要它：着色器只在**运行期**（app 启动读盘编译 + wgpu 建 pipeline）才暴露问题，
//! 而两类错误在 Rust 侧完全看不见：
//!   ① WESL 只把「从根模块 `main.wesl` 可达」的东西写进产物 ⇒ 子模块里没有任何调用点的
//!      `@compute` 会被整条剪掉，pipeline 按名字找入口时才失败；
//!   ② 子模块里引用了**没 import** 的绑定 ⇒ 产物里是个未声明标识符，naga 解析时才报错。
//! 这两类都在开发体积散射（godray）时实际撞到过 ⇒ 固化成这个测试。
//!
//! 注意它**验证不到**的：pipeline layout 与 bind group 的匹配（那是 wgpu 运行期的事）。

/// WESL 编译通过 + 全部 `@compute` 入口都在产物里 + 产物是合法 WGSL（naga 校验）。
#[test]
fn wesl_package_compiles_and_validates() {
  let src = gate_render::shader::compile_dda_wesl().expect("WESL 编译失败");
  // 全部入口（Rust 侧按名字找它们建 pipeline，见 `brickmap::dda` / `volumetric` / `gi`）。
  for name in [
    "dda_main",
    "dda_face_main",
    "dda_face_accum",
    "beam_main",
    "gi_main",
    "gi_denoise_temporal",
    "gi_denoise_atrous1",
    "gi_denoise_atrous16",
    "godray_main",
    "godray_blur_a",
    "godray_blur_b",
    "eye_adapt_histogram",
    "eye_adapt_update",
  ] {
    assert!(src.contains(name), "WESL 产物里没有 {name}（入口被剪掉了？）");
  }
  let module = naga::front::wgsl::parse_str(&src).expect("WESL 产物不是合法 WGSL");
  naga::valid::Validator::new(
    // 能力给满：校验口径不窄于运行期设备（产物无 64 位整数，不额外要求特性）。
    naga::valid::ValidationFlags::all(),
    naga::valid::Capabilities::all(),
  )
  .validate(&module)
  .expect("WGSL 校验失败");
}
