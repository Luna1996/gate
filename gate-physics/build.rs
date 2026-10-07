fn main() {
  if std::env::var("PROFILE").as_deref() == Ok("debug") {
    panic!(
      "禁止 debug 构建：本仓库只允许 release，请加 `--release`：\n\
       cargo build / run / check / test --release\n\
       cargo clippy --release --workspace --all-targets -- -D warnings"
    );
  }
}
