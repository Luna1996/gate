# 番外 - How to code PONG w/ Rust and Geese（事件系统教程）

- 视频: https://youtu.be/zqNTbttpmaY （20:53，2024）
- 类别: 其他 | ⭐ geese 事件系统官方用法教程（非 devlog 正片）

## 一句话
用 geese + macroquad 20 分钟写 PONG 的手把手教程：系统=事件处理器集合（DAG 依赖），数据放 Store（哑系统单例），图形/输入/物理三系统通过事件协作，游戏结束由物理系统 raise event 通知图形系统。

## 内容要点
- **geese 速览**：类似 MVC/Bevy ECS query 的架构模式；系统声明依赖成 DAG，可热插拔、易增删监听器。
- **核心习语**：
  - 系统 struct 持 `GeeseContextHandle`，实现 `GeeseSystem` trait（构造器 + 注册 handlers）；
  - 事件类型随意（任何 Send+Sync 类型），习惯集中放 `on::` 模块（`on::NewFrame(delta)`、`on::GameOver(winner)`）；
  - `context.flush(AddSystem/GameGraphics)` 添加系统、`flush(on::NewFrame)` 每帧广播；
  - **Store**：把任意 struct 变成哑系统存可变单例数据（GameWorld），初始值为 Default；系统间用依赖 + `get()` / `get_mut()` 读写，读写锁由 geese 管；
  - 系统 raise event（physics 检出球出界 → `raise(on::GameOver)` → graphics 监听并持久化胜者状态）；
  - 根系统 `PongGame` 以依赖列表拉起全部系统，消除 main 里逐个注册的样板。
- 完整示例在 geese 仓库。

## 对 gate 的启示
- 这是 gate 事件驱动模拟层的 API 设计速查：Store（纯数据）vs System（行为+私有状态）二分，与 gate 的「StateTable 数据 / 模拟系统行为」切分完全对应。
- 「物理系统产出事件、渲染系统消费并自己存跨帧状态」= gate 的「模拟 tick 只改 StateTable、渲染侧自持 GPU 缓存」模式的直接参照。
- get/get_mut 的借用仲裁由框架做 → gate 的 dirty component queue 消费侧可借鉴同样接口形态。
