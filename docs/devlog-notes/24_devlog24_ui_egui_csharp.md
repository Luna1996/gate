# Devlog #24 - Adding UI to my hybrid C# game engine was surprisingly tricky

- 视频: https://youtu.be/LLKCnN5a_FY （15:12，2024）
- 类别: 其他 | ⭐ UI 库选型教训：为何弃 ImGui 选 egui + C# 绑定自动生成

## 一句话
引擎改为 Rust 后端 + C# 前端 API 后找不到满意的 C# UI 库：ImGui 不安全 API + 全局状态 + begin/end 易错；最终把 egui 绑定到 C#——serde_generate 出类型 + rustdoc JSON 出函数签名，自动生成 80% 绑定。

## 内容要点
- **背景**：模组 API 面向 C#（sandboxed mods），UI 必须随之用 C# 可用的库。
- **弃 ImGui 三理由**：
  1. unsafe API：误用即 UB → 对要跑不可信第三方模组的引擎是 security non-starter；
  2. API 表达力弱、组合性差；
  3. 一个全局静态状态对象 → 多线程（他有 server/client 双线程）下的全局状态脚枪。
- **egui 的 API 设计优点**（他认为是 Rust 生态标杆）：作用域操作用闭包（window(|ui| ...)），begin/end 配对错误在编译期即不可能；「设计好 API 的原则 = 最小化犯错路径 + 用类型系统静态强制约束」。
- **自动绑定双难题**：
  1. C#↔Rust 类型互通 → egui 多为按值传递的 plain data + serde Serialize → **serde_generate** 自动生成 C# 侧类型定义，靠序列化跨 FFI 传对象；
  2. Rust 无反射 → 拿 **rustdoc JSON** 遍历 egui 全部函数/类型，与 serde_generate 的类型对上 → 自动出 80% 绑定，手写剩 400~500 个（比手写 2000+ 方法省 2.5 个月）。
- 成果开源（egui 的 C# 绑定）；另提及现成替代 egui.net。

## 对 gate 的启示
- gate 已裁决 bevy_ui 原生 + 自研 gate-ui，本篇作为选型对照确认了同一判断：**unsafe + 全局状态 + begin/end 三宗罪**正是要避开的坑；闭包作用域式 API（gate-ui 的 Panel/Label 组合器设计）应效仿 egui。
- 「用编译期静态约束消灭运行时错误」与 gate 的 newtype ID / 类型驱动连接语义一脉相承。
- rustdoc JSON + serde_generate 的自动绑定套路，若将来 gate 要暴露 C#/脚本 API 可复用，避免手写 FFI 地狱。
