fn main() {
  forbid_debug_build();

  println!("cargo:rerun-if-changed=../assets/locales");
  if let Ok(entries) = std::fs::read_dir("../assets/locales") {
    for e in entries.flatten() {
      let p = e.path();
      match p.extension().and_then(|x| x.to_str()) {
        Some("toml") | Some("yml") | Some("yaml") => {
          println!("cargo:rerun-if-changed={}", p.display())
        }
        _ => {}
      }
    }
  }
}

fn forbid_debug_build() {
  if std::env::var("PROFILE").as_deref() == Ok("debug") {
    panic!(
      "禁止 debug 构建：本仓库只允许 release，请加 `--release`：\n\
       cargo build / run / check / test --release\n\
       cargo clippy --release --workspace --all-targets -- -D warnings"
    );
  }
}
