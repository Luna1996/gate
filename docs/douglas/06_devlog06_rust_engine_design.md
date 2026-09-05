# Devlog #6 - Designing a FLEXIBLE game engine with Rust

- 视频: https://youtu.be/TRVrjSD6zgA （13:26，2023）
- 类别: 其他 | ⭐⭐ 架构设计：事件系统 + ECS 混合模型的直接参照

## 一句话
抛弃 Unity 式 OOP 对象图（Rust 借用检查器不允许互引用），改用「ECS 世界状态 + 事件系统」混合架构：数据走 ECS 管道、反应式行为走事件，系统保留私有内部状态（GPU 句柄等）。

## 内容要点
- **OOP 在 Rust 不可行**：Unity 式组件互引用构成依赖图，Rust 强制对象图成树；借用检查器逼出解耦 → 数据导向设计。
- **函数式视角**：游戏 = 当前状态 → 变换 → 下一状态的管道，无需引用只传裸数据 → 自然引出 ECS。
- **纯 ECS 不够**：系统需要私有状态（图形系统存 GPU 句柄）且要对事件做反应式响应。
- **最终架构**：World state（entities/components + 共享资源如系统时间）← systems（可存私有状态）= 事件处理器集合，可收可发事件（输入系统广播点击 → 网络系统响应发包）。事件注册用泛型实现，新事件只需定义函数并注册 handler。事件库开源（即后来的 geese）。
- **SVO 修改移植 Rust**：八叉树各类操作（数组转树/树合并/地形生成）本质同构 → 抽象出 `OctreeBuilder` trait，用泛型静态分发 → **零成本抽象**，对比 OOP 虚调用无运行时开销。
- **Web 多线程**：浏览器不能开原生线程 → SharedArrayBuffer + wasm-bindgen 系 crate（wasm-thread）实现 Web Worker 间共享 wasm 模块，真正的线程。

## 对 gate 的启示
- gate 的 CPU 事件驱动模拟队列与他的事件系统同型：系统私有状态（gate 的 StateTable / 组件元数据）+ 事件广播（gate 的 dirty component queue）验证了该架构可扩展到多人/模组。
- `OctreeBuilder` trait + 泛型静态分发可直接套用到 gate 的八叉树构建/合并/细化三套逻辑合并。
- 网络同步走「服务器持有八叉树，客户端都是内部服务器的客户端」模型——gate 未来多人化可参考。
