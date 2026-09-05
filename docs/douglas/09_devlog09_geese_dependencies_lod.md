# Devlog #9 - Codebase OVERHAUL, new EVENT SYSTEM, LODs, and MORE

- 视频: https://youtu.be/YQ83XfZQHHA （10:11，2023）
- 类别: 其他 | ⭐ 事件系统 2.0：依赖 DAG + 全引擎重写

## 一句话
重写 8000 行：事件库 geese 2.0 加入「系统间依赖」——事件系统构成 DAG，可互相查询（图形系统暴露分辨率，体素/光照渲染器各自依赖它），打破 1.0 时代巨型单体系统的必然；另加 LOD、greedy meshing 试验、客户端预测。

## 内容要点
- **geese 1.0 的问题**：系统完全隔离只能靠事件通信 → 被迫造巨型单体系统（一个 graphics system 管帧缓冲+网格化+渲染所有事）。
- **geese 2.0**：系统可声明依赖其他系统，handler 构成 DAG；依赖系统可被查询而非发事件；已上 crates.io。例：低层 graphic system（分辨率/上下文）← voxel renderer / lighting renderer 各自依赖并查询。
- **其余改进**：全代码文档注释；材质 u16 → 专用 MaterialId 类型（类型系统编码信息）；图形系统运行时热重载（如开关 vsync 不重启，涉及 GL 上下文重建）；改用 ureq/web 框架发布 wasm；网络抽象出 geese-pool。
- **体素同步自动化**：服务器自动追踪脏体素区域并隐式推送，服务端编辑零网络代码。
- **LOD**：远处 chunk 用低精度版本，省 vertex shader 调用。
- **greedy meshing 试验**：与 parallax ray marching 性能相当（本视频画面全是纯光栅化）；后续另开视频。
- **客户端预测**：编辑体素先在本地显示，服务器确认前可见 → 单机也变顺滑。

## 对 gate 的启示
- 「依赖 DAG + 查询而非事件」值得 gate 借鉴：编辑器/模拟/渲染三域之间，读侧查询（分辨率、调色板、tile 元数据）走直接依赖，写侧通知走事件，避免都堆成事件总线大泥球。
- 服务器脏区域隐式同步 = gate 的 dirty tile 队列，将来多人化直接复用。
- 类型化 ID（MaterialId）与 gate 的 palette u8/组件 u16 一致，坚持 newtype 封装。
- 他从 ray marching 回头试 greedy meshing 作为性能基线——gate 可在 M3 后做同样的对照基准，验证纯 compute DDA 是否真占优。
