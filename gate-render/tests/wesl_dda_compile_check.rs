use gate_render::shader::compile_dda_wesl;

fn naga_check(src: &str, tag: &str) {
  let module = match naga::front::wgsl::parse_str(src) {
    Ok(m) => m,
    Err(e) => panic!("naga 解析失败({tag}):\n{}", e.emit_to_string(src)),
  };
  match naga::valid::Validator::new(
    naga::valid::ValidationFlags::all(),
    naga::valid::Capabilities::all(),
  )
  .validate(&module)
  {
    Ok(_) => {}
    Err(e) => panic!("naga 校验失败({tag}): {e}"),
  }
}

#[test]
fn dda_wesl_compiles_main() {
  match compile_dda_wesl(false) {
    Ok(src) => {
      assert!(src.len() > 1024, "WESL 输出异常短: {}", src.len());
      naga_check(&src, "ray_query=off");
    }
    Err(e) => panic!("WESL 编译失败(ray_query=off): {e}"),
  }
}

#[test]
fn dda_wesl_compiles_ray_query() {
  match compile_dda_wesl(true) {
    Ok(src) => {
      assert!(src.len() > 1024, "WESL 输出异常短: {}", src.len());
      naga_check(&src, "ray_query=on");
    }
    Err(e) => panic!("WESL 编译失败(ray_query=on): {e}"),
  }
}
