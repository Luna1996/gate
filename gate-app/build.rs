//! 文案表（`assets/locales/*.yml`）是 `rust_i18n::i18n!(...)` 这个 **proc macro 在编译期**
//! 读进去、把每条文案内联进二进制的。但宏读文件不走 `include_str!`，cargo 无从得知文件变过
//! ⇒ 改完 yml 后 `cargo run` 不会重编，看到的还是旧文案（表现为"改了没反应"）。
//!
//! 这里把它显式登记成构建依赖：改任一文案表（或增删语言文件）都会让本 crate 重编，进而重跑宏。
//! 目录项只覆盖增删文件；**内容变更必须逐个文件声明**，因为 cargo 对目录只比 mtime。
//!
//! 另外本 crate **禁止 debug 构建**（见 `forbid_debug_build`）：`build` / `run` / `check` /
//! `clippy` / `test` 一律加 `--release`，与 .vscode/tasks.json 的构建任务一致。
//! 同一份判据在 gate-voxel / gate-render / gate-ui 的 build.rs 里各内联一份
//! （不共享文件的原因见那里）。

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

/// dev profile 直接编译期失败，杜绝产出 debug 二进制（无逃生开关）。
///
/// 判据取 build script 的 `PROFILE` 环境变量：dev → `debug`，release 与继承 release 的自定义
/// profile（如 `--profile profiling`）→ `release`。**不要改用 `DEBUG`**：那个跟随 debug info
/// 设置，`--profile profiling`（`debug = 1`）下同样是 `true`，会误伤。
fn forbid_debug_build() {
  if std::env::var("PROFILE").as_deref() == Ok("debug") {
    panic!(
      "禁止 debug 构建：本仓库只允许 release，请加 `--release`：\n\
       cargo build / run / check / test --release\n\
       cargo clippy --release --workspace --all-targets -- -D warnings"
    );
  }
}
