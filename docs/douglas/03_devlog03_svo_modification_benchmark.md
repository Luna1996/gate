# Devlog #3 - Sparse voxel octree modification and benchmarking

- 视频: https://youtu.be/yjOLx4O634I （7:36，2022）
- 类别: 基础渲染

## 一句话
解决「编辑时不重建整棵 SVO」的增量修改算法，并用 SIMD + C++ 重写做基准，C++ 比 C# 快一倍。

## 内容要点
- **为什么难**：从平面体素数组生成 256³ SVO 要 80ms+；编辑若走"全量重建"完全不可行。
- **增量修改算法**（本质是 target/source 两棵 SVO 的合并，像两张透明图叠放）：
  - source 为空处保留 target 原样；source 同质（单层单材质）且完全覆盖 target 时直接整体覆写；
  - 否则 target 八分，对重叠子八分体递归——利用同质区域短路，避免逐体素合并；
  - 递归必然终止于单体素对单体素同质叶。
  - 3 天实现，作者自嘲第一版用了 goto，待重构。
- **八分体坐标技巧**：体素在第 n 层的 3D 八分体坐标 = 其整数坐标二进制第 n 位的三个 bit；用 XOR/位运算 + x86 SIMD intrinsics 批量算。
- **基准数字**：C# 放一棵树 ≈11ms/次；C++ 重写后 ≈5ms；全量生成 80ms(C#) → 40ms(C++)。同为 unsafe 指针风格代码，.NET JIT 仍差一倍——结论：运行时本身存在固有差距。

## 对 gate 的启示
- 他的「同质区域短路合并」与我们的 per-Tile 增量重建/自由度更高的可变叶八叉树思路同源；其八分体坐标=坐标位提取的技巧在 gate 的 cell/octant 寻址里可直接复用。
- 他编辑后要 GPU 重新上传整棵树；gate 的 dirty-tile 增量上传路径更细，这一集印证了"编辑局部性"是正确方向。
- 性能结论对我们意义不大（Rust release 已接近 C++），但 SIMD 批量坐标计算的思路在 CPU 侧 flood-fill/寻址可借鉴。
