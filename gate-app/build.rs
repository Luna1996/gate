//! 文案表（`assets/locales/*.yml`）是 `rust_i18n::i18n!(...)` 这个 **proc macro 在编译期**
//! 读进去、把每条文案内联进二进制的。但宏读文件不走 `include_str!`，cargo 无从得知文件变过
//! ⇒ 改完 yml 后 `cargo run` 不会重编，看到的还是旧文案（表现为"改了没反应"）。
//!
//! 这里把它显式登记成构建依赖：改任一文案表（或增删语言文件）都会让本 crate 重编，进而重跑宏。
//! 目录项只覆盖增删文件；**内容变更必须逐个文件声明**，因为 cargo 对目录只比 mtime。
fn main() {
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
