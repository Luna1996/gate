# Devlog #23 - Adding global illumination to my game engine w/ DDGI

- 视频: https://youtu.be/L1vhle74AEU （13:39，2024）
- 类别: 光影 | ⭐⭐ gate P9 GI 的直接施工图

## 一句话
实现 DDGI（动态漫反射 GI）：表面邻近放探针存全方向辐照 + 深度图防漏光，每帧固定射线预算分摊到活跃探针，端点用上一帧结果 → 无限次反弹。

## 内容要点
- **要解决的问题**：缝隙阴影与树荫一样亮——缺间接光；路径追踪（#19）噪、贵，DDGI 是低噪高效替代。
- **原理**：探针放在待照表面附近，存储各方向入射光；着色时取邻近探针重度混合。**漏光问题**：探针额外存深度图（各方向最近表面距离），采样时「探针到墙的距离 < 探针到着色点的距离」→ 剔除该探针。
- **更新**：探针投随机射线，端点命中体素用**上一帧 DDGI 输出**着色 → 无限反弹；命中天空取天空色。
- **两部分结构**：
  - **烘焙（每模型上传一次）**：网格 cell = 16³ 体素，每 cell 至多 1 探针带偏移；沿 con tree **BFS 向下找靠中心的最大空叶**放探针（空 cell → 居中；半满 → 推向空边；全满 → 无探针）；再下采样出 4 级 LOD 探针。启发式目标：探针间距尽量大、离表面有距离（贴墙的探针一半纹素浪费）。
  - **每帧 shader 组**：①活跃探针识别（每 cell 一个 compute 调用，共享内存载邻接 cell 信息；附近无表面（本 cell+6 邻接 cell 无体素）或与非网格对齐对象 bbox 无重叠 → 不活跃；幸存者 GPU atomics 进 **worklist**）；②投射线（**每帧总射线数固定**，摊给活跃探针 → 性能不随屏上探针数波动；方向用 Fibonacci 球）；③把射线样本转成存储：irradiance 用**八面体映射**压进 2D 纹理数组，每纹素同时存平均深度。
- **着色采样**：找最近 8 个 cell → 三线性权重 × 视线可见性权重 × 前后位置权重 → 8 探针辐照加权平均。
- 依据 3 篇论文（原论文 + 生产改进）。另：chunk/编辑/模型导入回归；渲染器迁回 **Vulkan**；材质支持电介质/金属/哑光/镜面/发光。

## 对 gate 的启示
- **P9 的「DDGI」落地参数直接抄他**：16³ cell 粒度恰好等于 gate 的 cell 层级——探针放置算法（cell 内 BFS 找最大空叶、偏移、全满不放）可以逐条映射到 gate 的可变叶八叉树；4 级 LOD 探针对应 gate 的 tile/brick 层级。
- **每帧固定射线预算 + worklist** 与 gate 模拟的 per-tick work budget 是同一哲学，DDGI 侧照搬即可满足帧时硬指标。
- 活跃探针剔除（6 邻接 cell 无体素即休眠）在 gate 里可由 bitmap 一条 AND 指令判出，几乎免费。
- 八面体映射 + 纹理数组存 irradiance/depth 是标准存储方案；gate 用 wgpu 实现时同一套布局可用。
- 时间线注意：他是 #19 路径追踪（噪）→ #23 DDGI（稳）的演进；gate 亦可先上 #19 式廉价间接光、P9 再上 DDGI，两者共享 radiance/存储代码。

# 参考文献（DDGI）
"Dynamic Diffuse Global Illumination with Ray-Traced Irradiance Fields" by Majercik et. al
"Scaling Probe-Based Real-Time Dynamic Global Illumination for Production" by Majercik et. al
"Improving Probes in Dynamic Diffuse Global Illumination" by Rohacek