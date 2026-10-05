














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
fn cross_language_consts_parse_from_wesl() {
  gate_render::wesl_consts::gi_consts();
  gate_render::wesl_consts::trace_consts();
}




#[test]
fn key_org_q_matches_wesl() {
  let path = gate_render::paths::dda_wesl_dir().join("gi").join("common.wesl");
  let src = std::fs::read_to_string(&path).expect("读不到 gi/common.wesl");
  let v = gate_render::wesl_consts::parse_u32_consts_in_source(&src)["GI_KEY_ORG_Q"];
  assert_eq!(v, gate_render::gi::KEY_ORG_Q as u32, "gi::KEY_ORG_Q 与 gi/common.wesl 不一致");
}




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
