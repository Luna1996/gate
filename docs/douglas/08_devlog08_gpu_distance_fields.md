# Devlog #8 - GPU-generated DISTANCE FIELDS

- 视频: https://youtu.be/REKcTBgkrsE （12:53，2023）
- 类别: 基础渲染 | 实验未并入主线，但技术本身有独立价值

## 一句话
用 GPU 光栅化 + min-blending 实时生成体素距离场（DF）加速纯光线步进；最终实测仍不敌 parallax 方案而未并入引擎，但思路是空区域跳跃的正统解法。

## 内容要点
- **DF 加速原理**：DF 存「空间每点到最近表面的距离」，ray marching 时按 DF 值大步跨空（sphere tracing）；对体素/DDA 用**无穷范数（切比雪夫距离）**而非欧氏——「能画的最大立方体」，正好对应 DDA 步进语义。
- **采样速度**：DF 是平坦数组/纹理，采样远快于八叉树遍历 → 内存读少、GPU 快（他反复强调 ray marching 是内存带宽瓶颈）。
- **生成难点**：naive O(n³·k³)；DP 的 O(n³) 理论最优但**不可并行、只能 CPU 跑**，不适合实时。
- **他的 GPU 算法**（核心创新，把问题反过来做）：
  1. 只遍历**表面体素**（八叉树快速标记）——DF 对实心内部本来就无意义（射线不进入几何）；
  2. 对每个表面体素，向 3D DF 缓冲**光栅化一圈边长 2K-1 的 quad 立方体**（K=8/16）；
  3. fragment 里算距离，重叠 quad 用 **min blending** 取最小——完全吃满 GPU 光栅化+混合硬件，集显都能跑。
- **实测**：新 chunk DF 生成 15-30ms CPU + 10-20ms GPU（可再缓存/多线程）；渲染快但**多项场景下仍输 parallax ray marching** → 不并入主线。
- 顺带：Rust 冷编译 >30s → 决定拆 cargo workspace + 重写事件系统（即 #9）。

## 对 gate 的启示
- **无穷范数 DF + DDA** 的组合与 gate 的 bitmask 跳空是可叠加的两级加速：DDA 进 tile 后先查 3D DF 纹理大步跨空，命中候选区再走 bitmask。这是 1660 降档的候选手段（代价：DF 纹理 VRAM + 脏区重算）。
- 「min-blend 光栅化生成 DF」不依赖 compute，对低端卡友好；生成成本数据（25-50ms/chunk）可作预算参考。
- 他的结论也提醒：DF 不是银弹，parallax（光栅化+盒内步进）在他的负载下更强——gate 坚持纯 compute DDA 时，空区跳步的最优解要在实测里选。
