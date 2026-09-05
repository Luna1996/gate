# Shorts 合集 - Voxel Engine Weekly Update（5 则周更短视频）

- 类别: Shorts | ⭐ 历史演进碎片（早期光栅化时代，约 2023）
- 合并索引：
  - Collisions and linear physics: https://youtube.com/shorts/ozGtkNelKQ4
  - Rendering optimizations, realtime lighting, Perlin noise: https://youtube.com/shorts/-6oKAJM6fuE
  - Memory allocator for GPU buffers and textures: https://youtube.com/shorts/6RORASnbwT8
  - Efficient chunk meshing and VBO generation with SIMD: https://youtube.com/shorts/Pcb1SJDuYFI
  - Chunk loading and infinite worlds: https://youtube.com/shorts/XYBD3hvDPek

## 一句话
5 则 30 秒周更：线性碰撞 → 渲染优化全家桶 → GPU 显存分配器 → SIMD 网格生成 → 无限世界 chunk 流式加载；记录他从 mesh 光栅化路线转向 ray marching/DDA 路线前的技术积累。

## 内容要点
1. **线性物理**：重力 + 线性碰撞弹跳可用，旋转待做（后续 #11 SAT 展开）。
2. **渲染优化包**：共享 VBO + glMultiDraw + 视锥剔除 + 两级 CPU 面剔除 → 集成显卡 60fps 且带动态阴影 + 镜面高光；Perlin noise 地形实验。
3. **GPU 显存分配器**：抽象层把全部体素数据塞进 1 个大 buffer + 几张 3D 纹理；可视化显示分配/释放随编辑移动（bottom 显示区垂直线）。
4. **SIMD 网格生成**：从八叉树生成 parallax ray marching 用的包围盒 VBO，用 SIMD intrinsics 从 20ms → **6ms/次单线程**，计划再并行化。
5. **无限世界**：chunk 按距离加载/卸载出内存，跨 chunk 边界可编辑。

## 对 gate 的启示
- 显存分配器（大 buffer + 3D 纹理统一管理）正是 gate 的 brick map 内存布局思路的早期同款；gate 的 `ensure_with_copy` 全量回写策略可参考他"分配/释放随编辑移动"的 arena 化设计降低回写成本。
- SIMD 八叉树遍历生成（20ms→6ms）佐证 gate CPU 侧 DDA 参考实现/烘焙路径的 SIMD 优化空间。
- chunk 流式加载与无限世界的边界编辑问题，gate 沙盒模式的 100 万 tile 目标未来同样要过这道坎。
- 历史脉络：他早期是 mesh+光栅化 → 中期 parallax ray marching → 后期 SVO+DDA+光追；路线切换成本很高，gate 直接从 SVO+DDA 起步是正确选择。
