//! 禁止 debug 构建：本仓库只允许 release。
//!
//! 为什么每个 crate 各自内联这份判据、而不是共享一个文件：rust-analyzer 无法解析
//! `include!("../build_guard.rs")`，也无法解析越出 crate 目录的 `#[path]` 模块，
//! 两种共享写法都会在 IDE 里报错（rustc 本身都能编过）。判据只有几行，重复代价小于
//! 一个跨 crate 的构建依赖。
//!
//! 判据取 build script 的 `PROFILE` 环境变量：dev → `debug`，release 与继承 release 的
//! 自定义 profile（如 `--profile profiling`）→ `release`。**不要改用 `DEBUG`**：那个跟随
//! debug info 设置，`--profile profiling`（`debug = 1`）下同样是 `true`，会误伤。
fn main() {
  if std::env::var("PROFILE").as_deref() == Ok("debug") {
    panic!(
      "禁止 debug 构建：本仓库只允许 release，请加 `--release`：\n\
       cargo build / run / check / test --release\n\
       cargo clippy --release --workspace --all-targets -- -D warnings"
    );
  }
}
