---
description:
globs:
alwaysApply: true
---

# UI 文案表（`assets/locales/*.toml`）的约定

中文文案表在 [assets/locales/zh-CN.toml](../../assets/locales/zh-CN.toml)（rust-i18n 在**编译期** codegen 进二进制）。
每语言一个文件（`assets/locales/<locale>.toml`）；加语言 = 加文件 + 调 `rust_i18n::set_locale`。
表头 `_version = 1` 就是这个布局的版本号。

写 / 改文案时按下面几条来：

- 键 = 「模块.控件」语义路径，调用点写 `t!("模块.控件")`
- **键必须带引号**：TOML 里裸点号会被解释成嵌套表，与扁平 key 语义不同
- `label` 空间只容得下 **4 个汉字或 8 个字母**，超限就换词
- 占位符 `%{name}` 原样替换；定长对齐的数字由调用点先 `format!` 好再传入
- 只收**可见 UI 文案**；日志、技术标识符（LOD / DDA）、单位（v/s、vx）、材质预设名保持英文
- `*.tip` 的值是 Markdown（经 gate-ui tooltip 渲染）：多行 / 列表 / 引用用 `"""…"""` 写，
  续行顶格（缩进 4 空格会被当成代码块）；行内 `code` 标标识符与数字，**粗体** 标关键结论
- `*.tip` **只写"让人明白"的最低限度**：不抄实测数字、不写历史对比、不复述调用点注释
- DebugMenu 的文案 key 在本表 `menu.*` 段：菜单树（[assets/ui/debug_menu.toml](../../assets/ui/debug_menu.toml)）
  存 key，由 gate-ui 的 `UiTranslator` 解析，切语言后整棵树重解析（gate-app 的 `sync_ui_locale`）
- 改本表需**重新编译**（proc macro 编译期内联）；`gate-app/build.rs` 已登记 rerun-if-changed
