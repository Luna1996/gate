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

/// WESL 编译通过（**两版**）+ 全部 `@compute` 入口都在产物里 + 产物是合法 WGSL（naga 校验）。
///
/// 两版都要过：软件版是**回退路径**（`ray_query` 关），RT 版是默认路径；任一侧的 `@if` 写错都会在
/// 这里炸（例如把 `enable wgpu_ray_query;` 漏在软件版里、或 RT 版少声明 `tlas`）。
#[test]
fn wesl_package_compiles_and_validates() {
  for ray_query in [false, true] {
    let src = gate_render::shader::compile_dda_wesl(ray_query).expect("WESL 编译失败");
    assert_eq!(src.contains("enable wgpu_ray_query"), ray_query, "enable 指令与 flag 不对应");
    assert_eq!(
      src.contains("var tlas: acceleration_structure"),
      ray_query,
      "tlas 声明与 flag 不对应"
    );
    validate_all_entries(&src, ray_query);
  }
}

fn validate_all_entries(src: &str, ray_query: bool) {
  // 全部入口（Rust 侧按名字找它们建 pipeline，见 `brickmap::dda` / `volumetric` / `gi`）。
  for name in [
    "dda_main",
    "dda_face_main",
    "dda_face_accum",
    "beam_main",
    "gi_main",
    "gi_face_flatten",
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
  if ray_query {
    assert!(src.contains("trace_grid_rt"), "RT 版产物里没有光追遍历函数");
  } else {
    assert!(!src.contains("trace_grid_rt"), "软件版产物里不该有光追遍历函数");
  }
  let module = naga::front::wgsl::parse_str(src).expect("WESL 产物不是合法 WGSL");
  naga::valid::Validator::new(
    // 能力给满：校验口径不窄于运行期设备（产物无 64 位整数，不额外要求特性）。
    naga::valid::ValidationFlags::all(),
    naga::valid::Capabilities::all(),
  )
  .validate(&module)
  .expect("WGSL 校验失败");
}

/// 跨端常量（`wesl_consts`）能从**同一份源码**解析出来：Rust 侧不留副本，解析不到即启动 panic
/// ⇒ 在测试里先炸，别等到运行期。覆盖 `gi/` 与 `trace.wesl` 两组（缺一个即 panic）。
#[test]
fn cross_language_consts_parse_from_wesl() {
  gate_render::wesl_consts::gi_consts();
  gate_render::wesl_consts::trace_consts();
}

/// 面键的**量化编码原点步长**两侧必须逐字相等（`gi::KEY_ORG_Q` ↔ `gi/common.wesl::GI_KEY_ORG_Q`）：
/// 不等 = Rust 判"原点变了"的时机与 shader 实际换编码基准的时机错位 ⇒ 复用链要么被无谓地每帧
/// 作废、要么跨基准误配（同一体素拿到另一个坐标系下的历史）。
#[test]
fn key_org_q_matches_wesl() {
  let path = gate_render::paths::dda_wesl_dir().join("gi").join("common.wesl");
  let src = std::fs::read_to_string(&path).expect("读不到 gi/common.wesl");
  let v = gate_render::wesl_consts::parse_u32_consts_in_source(&src)["GI_KEY_ORG_Q"];
  assert_eq!(v, gate_render::gi::KEY_ORG_Q as u32, "gi::KEY_ORG_Q 与 gi/common.wesl 不一致");
}

/// **区域修订表**的常量两侧必须逐字相等（`gi::REGION_CHUNKS` / `REGION_TABLE` ↔
/// `gi/common.wesl::GI_REGION_CHUNKS` / `GI_REGION_TABLE`）：不等 = Rust 标记的区域与 shader
/// 查询的区域错位 ⇒ 二次顶点缓存的局部失效既不覆盖该失效的、又误伤别的（画面滞后或闪烁）。
#[test]
fn region_consts_match_wesl() {
  let path = gate_render::paths::dda_wesl_dir().join("gi").join("common.wesl");
  let src = std::fs::read_to_string(&path).expect("读不到 gi/common.wesl");
  let v = gate_render::wesl_consts::parse_u32_consts_in_source(&src);
  assert_eq!(
    v["GI_REGION_CHUNKS"],
    gate_render::gi::REGION_CHUNKS as u32,
    "gi::REGION_CHUNKS 与 gi/common.wesl 不一致"
  );
  assert_eq!(
    v["GI_REGION_TABLE"],
    gate_render::gi::REGION_TABLE as u32,
    "gi::REGION_TABLE 与 gi/common.wesl 不一致"
  );
}
