# R3 光影效果

> 施工优先级：同类内保持原编号顺序；新编号为文件内全局顺序号，旧 `Px.y` 编号保留作历史锚点。

- [ ] **~R3-1（3.1a）~** ~~逐面着色风格与「细分致光滑」~~ — **v5 废弃**。架构回退到逐体素 flat shading（per-voxel，对齐 Douglas #22/#23 最新架构），见 README 决策表「光照量化粒度（v5）」。**细分策略本身仍值得保留**（曲面热点 0.25cm 小体素 = 更细腻几何），但不再服务于逐面着色的明暗台阶效果
- [ ] **R3-2（3.4a）** **密度场 AO（对标 #15，3.4 的「待后续」子任务）**：每 cell（或 16³ 粒度）统计体素占据数 → 一张小 3D 纹理（R8/R16）；DDA 命中后按世界坐标采样（硬件 trilinear 单次纹理读），>0.5 部分映射为暗度乘进直光/环境光；**采样点 per-voxel**——采样位置 = 命中点沿命中面法线偏移半体素，作为该体素的 AO 暗度值；密度计数在 tile builder 构建时顺手产出、编辑时局部重算、增量上传管线复用——成本几乎为零（比 SSAO 便宜一个量级）；#15 两条踩坑不必重踩（Minecraft 式逐体素 AO：微体素粒度太细失败；SSAO：随机采样噪声被纯色风格放大成闪烁）；环境光 sky 渐变部分已于 2026-09-01 落地（存档 3.4）
- [ ] **R3-3（3.5）** 后处理：ACES tonemap + sRGB（评估复用 Bevy 后处理链）+ **轻量 bloom**（阈值 + 小半径模糊，发光体/太阳光晕，可开关）
- [ ] **R3-4（3.5b）** **距离雾/大气透视（对标截图 3/5 远山氛围）**：shade_hit 出口按命中 t 指数衰减混向 `sky(视线方向)`；密度/高度衰减参数主题化；近零成本（compute 内数行）
- [ ] **R3-5（3.5c）** **God rays 屏幕空间廉价版（对标截图 5 第一步）**：后处理径向模糊（太阳屏幕位置 CPU 传入；DDA 命中 t/天空遮罩防穿透穿帮；强度/衰减长度可调，可开关）；9.8 介质散射版落地后可退役或保底
- [ ] **~R3-6（3.5d+）~** ~~Phase 2 hashmap 管线~~ — **v5 全部废弃**。逐面着色 + per-voxel face_light hashmap 架构彻底移除（代码已删：face_light.rs + dda.wgsl 四段式 + Rust 端 fl_clear/fl_light/fl_composite）。
- [ ] **R3-7（3.6）** 锁定体素全息视觉变体
- [ ] **R3-8（3.7）** 性能验收：1080p@60fps（GTX 1660）、每帧 1 万体素编辑不掉帧（基准以 1cm 工作区分辨率计）；**极限压测**：满屏 0.25cm 热点 + 最坏阴影射线路径的帧时间上界断言，VRAM ≤2GB 断言；**外部锚点（#17）**：Douglas GTX 1660 Ti 主射线+阴影射线 7ms/帧（未做 marching 循环优化、无 depth prepass，自评 compute bound 仍有余量）——M2 阴影射线预算可行性的同卡证据，3.5d 软阴影采样有余量
- [ ] **~R3-9（9.1a）~** ~~半分辨率路径追踪 pass~~ — **已取消**。被 R3-10 DDGI 取代；radiance emissive 分支保留在 R3-10 probe endpoint
- [ ] **R3-10（9.1b） ⚠️ 优先级提升：** **DDGI 全局光照（1:1 复刻 Douglas #23）** — probe worklist + 固定预算射线 + 上一帧 DDGI 输出自闭环。probe endpoint emissive 分支保留（gate 核心画面：通电 LED 珠照亮周围墙）。
  ```wgsl
  fn probe_endpoint_shade(hit) -> vec3<f32> {
      if hit.sky { return sky(rd); }
      if hit.emissive { return hit.albedo * hit.emissive * GAIN; }  // gate 扩展
      return sample_prev_ddgi(hit.world_pos, hit.normal);
  }
  ```
  composite（v5）：`final = triplanar_texture × max(sky_grad(直光, per-pixel 1 条太阳射线), ddgi_probe_indirect(间接光)) + emissive_radiance` — **无任何 per-voxel 光照缓存**。
  - ✅ 共用底座：DDA/trace_grid()（两级+OBJ）、sky()/shade_hit、StateTable emissive 占位
  - ❌ DDGI 新增：①探针烘焙（cell 16³ BFS 空叶 + 偏移启发式 + 4 级 LOD）②每帧固定预算射线分摊（Fibonacci 球 + worklist atomics + 上一帧 DDGI 着色 → 无限次反弹）③活跃探针剔除（6 邻接 cell bitmap AND）④irradiance + 深度图 → 八面体映射 2D 纹理数组 ⑤漏光治理（深度图，采样时 probe-to-wall < probe-to-point 剔除）⑥着色采样（最近 8 cell 三线性 × 视线可见性 × 前后权重）
  - **参考文献（必读，按序）**：
    1. [Majercik 2019 原论文](https://jcgt.org/published/0008/02/01/paper.pdf) — §3 八面体映射存储（irradiance 8×8 + depth 16×16）、§4 射线更新、§5 视线感知查询、附录 GLSL
    2. [Majercik 2021 生产版](https://jcgt.org/published/0010/02/01/paper-lowres.pdf) — §3 探针状态机（活跃/休眠）、§4 调参、§5 级联体积
    3. [Rohacek 2022 漏光治理](https://old.cescg.org/CESCG-2025/wp-content/uploads/2022/04/Rohacek-Improving-Probes-in-Dynamic-Diffuse-Global-Illumination.pdf) — §3.1 BFS 探针放置、§3.2 锐利背面剔除、§3.3 漏光治理
  - **实现 spike 路径**（论文章节即 checklist）：
    0. 读三篇论文 + gate 实现笔记（**前置**）→ 每篇对应一个模块规格
    1. 探针烘焙（Rohacek §3.1）：cell 16³ BFS 最大空叶 + gate 偏移启发式 → 验证层级对应
    2. 单探针射线更新（Majercik §4）：Fibonacci + worklist atomics → 简单方向平均存辐射度验证间接光传播
    3. 八面体映射 + 深度图（Majercik §3）：irradiance 8×8 + depth 16×16 纹理图集
    4. 探针状态机 + 活跃剪枝（Production §3）
    5. 漏光治理 + 锐利背面剔除（Rohacek §3.2/3.3）
    6. 着色采样（Majercik §5）：最近 8 cell 三线性 × 视线可见性 × 前后权重
- [ ] **R3-11（9.2）** 时域累积 + 相机运动重投影
- [ ] **~R3-12（9.3）~** ~~À-trous 去噪~~ — **已取消**。DDGI 探针多帧采样天然时间平均；可选重开本条目如 DDGI 远距离闪烁实测仍需
- [ ] **R3-13（9.4）** 上采样 + 与 M2 直光合成
- [ ] **R3-14（9.5）** 蓝噪声序列管理
- [ ] **R3-15（9.6）** 实机调优迭代回路（用户在环，仅「暗色实验室」主题）
- [ ] **R3-16（9.7）** 验收：与 bevy_vox_scene 参考图并排对比超越（暗色实验室场景，RTX 3070 1080p60）；**观感对标锚点 = Douglas Dwyer devlog 截图**（图1 发光房间 = PT 自发光照明 + DDGI 颜色渗透同级；图5 黄金时刻 = god rays + 天空 + 雾综合观感；DDGI 探针放置对照 #23 cell 16³ 落地）；**着色粒度 1:1 对齐 Douglas #22/#23**（逐体素 flat shading），验收标准 = 整体观感并排不劣化
- [ ] **R3-17（9.8）** **God rays 介质散射升级（两步走第二步）**：DDGI 探针射线端点内（或单独介质 pass）参与介质积分（太阳方向体积 NEE 采样），物理正确光柱取代/增强 R3-5 屏幕空间版；1660 降档路径 = 关 GI 时回退 R3-5 屏幕空间版
