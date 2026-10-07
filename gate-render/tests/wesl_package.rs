fn validate_all_entries(src: &str, ray_query: bool) {
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
    naga::valid::ValidationFlags::all(),
    naga::valid::Capabilities::all(),
  )
  .validate(&module)
  .expect("WGSL 校验失败");
}

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
