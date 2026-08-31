# Devlog #1 - Implementing sparse voxel octrees and the ray caster

- 视频: https://youtu.be/zbmYzugnEC0 （8:41，2022）
- 类别: 基础渲染 | 阶段: 项目起点

## 一句话
项目立项：C#/Vulkan + 稀疏体素八叉树（SVO）+ compute shader 光线步进，跑通首个体素画面。

## 内容要点
- **动机**：想要多动态光源，shadow mapping 不合适 → 直接选光线追踪路线；每像素射线，命中求色，阴影=再投一条射线，反射=射线弹射。
- **技术选型**：C++；OpenGL（macOS 无 compute）→ OpenCL（维护差、需 GL interop）→ 最终 Vulkan。900 行才画一个三角形，搭框架花两周。
- **数据结构**：稀疏体素八叉树；256³ 体素从 16MB（1B/voxel）压到 <0.5MB，作为 storage buffer 传 GPU。
- **渲染算法**：compute shader 光线步进 → 写 image → 上屏。先 AABB 求交跳到最近物体，在八叉树内按空分支整体步进（大 box 同质可一次跨过），空了就跳出当前 octree 换下一个 object。
- **调试**：用 RenderDoc 单步 GPU shader、查看变量值（作者也不知道这可行）。

## 踩坑（很有参考价值）
1. 浮点不精确导致射线在相邻两体素间**乒乓死循环** → 修复：保证每次至少向前推进一步。
2. 射线偶尔**丢失当前体素位置** → 修复：把射线位置吸附（snap）到当前体素内。
3. 未解之谜：命中列表数组（shared）从 4 扩到 8/16 就变慢，且去掉 shared（private memory）直接慢一倍——疑似 cache/GPU 内存行为，作者留作后续课题。
- 性能基线：GTX 1070，7-10 ms/帧。

## 对 gate 的启示
- 「AABB → 空分支整体跳过」与我们 DDA 的 slab 相对标尺 + bitmask 跳空是同一家族；他踩的乒乓/丢位两个浮点坑值得对照自查 WGSL 边界步进逻辑。
- 他的动机链（多光源→RT）与我们 StateTable 发光/多主题光照的演进方向一致。
