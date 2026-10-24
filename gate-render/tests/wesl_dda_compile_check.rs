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