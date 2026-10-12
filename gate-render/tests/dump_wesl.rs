#[test]
fn dump_sun_transmittance() {
  let src = gate_render::shader::compile_dda_wesl(false).expect("WESL 编译失败");
  let lines: Vec<&str> = src.lines().collect();
  let mut shown = 0usize;
  for (i, l) in lines.iter().enumerate() {
    if l.contains("fn sun_transmittance") {
      let lo = i.saturating_sub(2);
      let hi = (i + 60).min(lines.len());
      eprintln!("---- sun_transmittance @ {i} ----");
      for (j, l) in lines[lo..hi].iter().enumerate() {
        eprintln!("{:5}: {}", lo + j, l);
      }
      shown += 1;
    }
  }
  eprintln!("total lines = {}, occ = {shown}", lines.len());
  assert!(shown > 0);
}
