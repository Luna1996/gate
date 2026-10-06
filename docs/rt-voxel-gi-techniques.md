# 实时微体素渲染 / 全局光照：公开技术检索结论

本文是一次广泛公开检索（论文 + 生产 talk + 官方 SDK 文档）的结论沉淀，用于**避免重复检索**，并记录哪些方向已被验证否决。

范围：① 微体素几何的存储与光线遍历加速；② 实时 GI 算法；③ 采样与降噪；④ 与本项目架构的迁移性判断。

## 0. 本项目的架构坐标

现有实现 = **GigaVoxels/ESVO 血统 + 两级辐射缓存 + 时间累积**：

- 几何：256³ 区块 + 4³ 细分体素分裂树 + 8×8×8 occ 位图 + 块级 DDA（含 ESVO 式 beam：4×4 tile 中心射线 → 3×3 邻域取 min → 当 `t_min` 跳过前段空档）
- 流式：射线驱动 LOD 请求（= GigaVoxels 的 ray-guided streaming）
- GI：屏幕空间 GI texel（`gi_size`）→ 半球射线 → 降噪（temporal + 5×atrous）→ **世界空间哈希"面槽"**（`face_slots`，键屏蔽面号 → 逐体素）+ 二级缓存 `gi_sec_slots`（含 epoch/局部版本键）+ WAL 世界累积
- 主 pass：按"面槽"取 GI（保证逐体素同色），另有一条 per-pixel 太阳阴影射线

**这正是生产界的共识形态**：idTech8 的 GI 就是「world radiance cache（空间哈希）+ 级联/局部辐照度体积 + 交错更新 + final gather（零着色只查缓存）+ 去噪」。所以大方向不必怀疑；缺的是几个已在生产验证过的具体机制。

检索时间：2026-10；下次要再查，先看本文的"已否掉"一节。

## 1. 几何 / 加速结构

| 技术 | 来源 | 与本项目的关系 |
|---|---|---|
| SVDAG（合并同构子树） | Kämpe 2013；Aokana 2025 | 静态内容内存压缩的主力。对应本项目 b_struct 顶满 2GB 单 binding 的问题 |
| HashDAG（可编辑 + 版本化根 + 哈希节点查询 + 虚拟地址） | Careil 2020 | 面向"可修改的大规模体素场景"，与本项目可编辑近场同场景 |
| 颜色解耦（DFS 序颜色数组 / 属性块 c0,c1+权重插值） | Dolonius 2017 | 几何与颜色分开压缩；遍历时二分查找属性块。本项目 palette 可参考 |
| Hybrid Voxel Formats（分层格式组合 + 元编程生成遍历） | Arbore 2024 | 给出「内存 vs 求交性能」的 Pareto 前沿与可做变换清单；brickmap 本身就是 hybrid format |
| Aokana（多浅 SVDAG + LOD + 流式 + GPU-driven + Hi-Z + visibility buffer） | 2025 | 内存最多降 9×、渲染快 4.8×，只驻留 ~5%。visibility buffer 路线不适用本项目（纯光追），但压缩/流式值得对照 |
| GigaVoxels DP（8³ brick pool + 按像素射线状态触发 brick 生产） | Richermoz & Neyret 2024 | 本项目已有 brick pool + 射线驱动加载；差异在它假设全透明体素 |
| ESVO beam optimization | Laine & Karras 2010 | 本项目 beam 的出处；也说明**beam 只服务主射线**（见 §5 已否掉） |
| out-of-core SVO 构建（Morton 序 + 外部排序） | Baert 2013 | 若要把构建搬到离线/多线程可参考 |

## 2. GI 算法

| 技术 | 来源 | 与本项目的关系 |
|---|---|---|
| **世界辐照度缓存 / SHaRC**（空间哈希 1D 数组 + N 维哈希 + LOD 量化；**用缓存命中提前结束路径**） | NVIDIA RTXGI 2.0；DOOM The Dark Ages 采用 | 与 `face_slots`/`gi_sec_slots` 同构。关键差异见 §4 |
| **On-Surface Radiance Caching**（两层都在表面；primary 存入射、secondary 存**出射**且可互采样 → 随时间无限弹射；纹理空间刻意低于屏幕） | Tatzgern 2024 (HPG) | 本项目只有 primary 层。这是"补 secondary 层"的正规设计 |
| **Surfel GI (GIBS)**（可见几何实时 surfel 化 + 辐照度累积；**出屏后缓存仍持久**） | Halen 2021, SIGGRAPH Advances | 面槽本质就是 surfel 缓存；最接近的邻居 |
| **DDGI**（探针级辐照度 + 可见性加权 moment 插值防漏光）+ **DDGI Resampling**（探针当光源进 ReSTIR） | Majercik 2019 (JCGT) / 2021 | 探针可承担远景/多次弹射 |
| **idTech8 GI**：world radiance cache（空间哈希，25cm³ 量化 + LOD，仅 14MB）+ 级联/局部辐照度体积（16³×6；**每帧只交错更新 1 级联 + 1 体积**；每探针 64/32/16 条**可见性**射线）+ final gather（零着色）+ 去噪 + 上采样；另加 transparencies froxel 体积 | Sousa, SIGGRAPH 2025 | ①"只追可见性、命中触发缓存更新、final gather 只查缓存"与本项目结构同构；②交错更新是把成本摊到多帧的正规做法；③哈希设计（1D + 量化 + LOD）很省内存 |
| **GI-1.0 两级辐射缓存**（screen cache：自适应采样 / ray guiding / radiance blending / probe 屏蔽；world cache：哈希单元 + fingerprint + 线性探测，**只缓存直接光**）+ SH 辐照度 + 去噪 | AMD 2023 | 面槽≈它的 world cache；"world 只缓存直接光"对应本项目 `sun_bounce=false, depth=off`；**自适应采样/ray guiding 是本次要做的第 1 项** |
| **ReSTIR GI / PT / PT Enhanced**（时空蓄水池重采样；1spp 下 MSE 提升 9–166×；PT Enhanced 再快 2–3×） | Ouyang 2021 / Lin 2022 / 2026 | 可把"每 texel N 条射线"压到"每帧 1 条 + 复用" |
| **Radiance Cascades / Holographic RC**（无噪声、无随机、不依赖时间缓存；成本对场景尺寸常数；用 penumbra hypothesis 复用短区间替代长射线） | Sannikov 2023 / Freeman+ 2025 | **成熟形态是 2D**（Path of Exile 2 用于屏幕空间照明）。3D 未成熟；可考虑只用于屏幕空间那一层 |
| VCT + 3D clipmap（在体素金字塔上锥体追踪；环形寻址 + 增量更新） | Crassin 2011 / Panteleev (GTC) | **本项目 brickmap 本身就是多分辨率金字塔，锥体追踪可直接建在上面**（无需体素化）。近场射线 + 远场锥体是可行混合 |
| LPV / Irradiance Volumes（Greger 1998）/ 2-band SH | idTech7/8、OGRE-Next、Godot 文档 | 远景 GI 的低成本层 |
| Voxel GI 方法总览（VCT / IFD / CIVCT / 探针 的 pros-cons） | OGRE-Next 文档；Godot 文档 | 选型清单 |
| Scaleable 大场景 GI（辐照度 clipmap + **用屏幕 gbuffer 反馈 voxelize**，把可见辐射补进体素场） | Gaijin, GDC 2019 | 0.7–1.6ms 量级；"屏幕反馈补世界体素"可参考 |
| Dynamic Voxel-Based GI（静态/动态分离，只重算动态部分） | CGF 2025 | 与本项目"编辑块重算"思路一致 |
| Instant Radiosity / VPL、Photon mapping(hash grid)、Neural(NRC/TransGI) | 检索中一并覆盖 | VPL 与光照数量挂钩、不适用单太阳；Neural 需训练管线，与"不损正确性"约束冲突，暂不考虑 |

## 3. 采样与降噪

- **方差引导的自适应采样分配 / ray guiding**（GI-1.0 §2.1.2；ReSTIR 的目标 PDF 估计）：本项目现在是"均匀 × 分帧 + realloc"，**没有按方差分配**。纯算法、不动数据结构 —— 本次第 1 项。
- **SVGF / A-SVGF / BMFR / ReBLUR**（ReBLUR 是 AC Shadows 用的层次化循环降噪器）：本项目为 temporal + 5×atrous；换 SVGF 类通常同噪声下减 30–50% 采样。
- **SER（Shader Execution Reordering）**：DOOM 用它吃发散；**wgpu 不暴露**，只能软件近似（按材质/方向分桶）。
- 低差异序列 / 蓝噪声去相关：改善同样本数下的降噪效果。

## 4. 与文献的关键差异（易踩）

**缓存的键挂在"射线命中点"上，不是"像素所属的体素"。** SHaRC / on-surface cache / GIBS 的键是世界空间的**命中点**（跨帧持久，含法线/LOD/时间量化），射线命中就去查/填 —— 覆盖率天然高。

本项目曾试过"按像素所属体素"做同帧缓存（生产者放 GI texel、消费者放主 pass 像素），覆盖率≈0：远景体素小于一像素，两者命中的体素几乎不同。**要再做这条，键必须挂在命中点 + 跨帧持久 + 正确失效键。**

**逐体素同色是本项目独有约束**：所有 surfel/探针/哈希方法都在**面上**定义值。任何选型末端都要再接一层"按体素键聚合"（即 `face_slots` 屏蔽面号那一层）。ReSTIR 类还要额外确认其空间相关性不破坏该量子化。

## 5. 已验证并否掉的方向（不要重走）

| 方向 | 否掉理由（实测） |
|---|---|
| 太阳阴影图 / CSM | 太阳每帧微动，逐帧重建=负优化 |
| 太阳空间 min/max beam（按太阳轴分列存占据距离区间） | 需要每帧随太阳重建的太阳空间结构；用户判定为负优化 |
| 把主射线的 `level_cap` / 网格调度（`t_lo`）补到阴影/介质射线 | 静态确定性 A/B：基线 trace 6.43ms → +level_cap 6.00–6.51 → +t_lo 6.31–6.44，**全在噪声内**。原因：阴影射线本就有 `GI_REACH_T`(3072 体素) 这层界，`level_cap` 永不 bind；i≥1 的网格是**远场独立区域**（`grid_is_far` + 各自 AABB），不是 LOD 副本，追它们是必要开销，不是冗余 |
| 按体素同帧缓存太阳可见度 | 射线计数 162.4 → 198.8/texel（更差）；覆盖率≈0（见 §4） |
| "realloc 的『已填满⇒少发』是死代码" | 假设被实测推翻：该分支命中率 77–80%，工作正常 |
| GI texel 级 setup 早退（候选=0 时跳过 setup） | `cand_n = max(1, …)`，本项目配置下每 texel 至少 1 条候选，不存在 0 候选 texel，收益为 0（只对 `分帧 ≥ 4` 的 stride 情形有意义） |
| 单纯的 pass 删除类"清理" | render graph 逐趟审计：**每一趟输出都有消费者**，无死 pass（唯一停派的 `dda_face_main` 本就不派发，pipeline 仅启动期成本） |
| **方差引导的自适应采样分配**（加装在**二元** `b_err` 之上） | 第一次尝试：射线 5.8–6.1 → **10.4–10.7/texel**、`gate_gi` 7.7–8.2 → 14.7–15.3ms——总量守不住。**根因不是归一化写法，而是权重与 `realloc` 的逐 texel「填满判定」互相拉扯**（压低射线数 → `hist.m` 涨得慢 → 判定没填满 → `b_err` 跳到 4× 补回），且当时把**已带权**的每 texel 预算存进 guide → 归一化自引用、越权越偏。**已修并重做成功，见下。** |

## 5.5 已实现并验证：连续 `b_err` + 方差引导分配

**判据（不需要目验）**：等总射线数下比"平均创新量"（本帧新样本均值与历史均值的相对偏差，按亮度归一）。

fly bench、同配置（GI 1/4 + 分帧 8 + `realloc=强`）、同构建、关 vsync、profiler 关 debug groups：

| 变体 | 候选射线/texel | 平均创新量 | `gate_gi` | 帧周期 |
|---|---|---|---|---|
| 二元 `b_err`（原状） | 2.0–2.6 | 1.45 | 0.98–1.07ms | 15.9–17.1ms |
| ① 连续 `b_err` | 1.5–2.2 | 1.53 | 0.85–1.11ms | 12.3–15.0ms |
| ①+② 连续 + 方差引导 | 1.6–2.2 | **1.34** | 1.00–1.22ms | 12.3–14.0ms |

- **②在①同一预算下把创新量降 12%**（1.53→1.34）⇒ 重分配有效。
- 相对原状：射线 **−15%**、创新量 **−7.4%** ⇒ 更少射线且更收敛。
- 代价：`gate_gi` +0.1ms（两次 barrier + 两次归约 + 打包）；配置的 GI 分辨率/总射线越少，这部分占比越大。

**成立的两个必要条件（缺一即失败）**：
1. **`b_err` 必须连续**（按窗的空缺比例插值，而非"填满/没填满"二元跳变）。二元的判定是硬反馈：压低某 texel 的射线数 → `hist.m` 涨得慢 → 判未填满 → ×4 补回，任何重分配都会被抵消。
2. **guide 里必须存"未带权"的每 texel 射线基数**（`cand_n0·b_err/share`，不含 `w`）。存已带权的实际速率会让归一化 `Σ(base·w)/Σbase` 自引用 → 越权越偏 → 预算爆掉（实测 +75%）。
   附带：`m_cap_k` 也要用未带权的 `cand_base`，否则"上限压小 → r.m 达不到 warm_m → 判未填满"形成次级正反馈。

**当前设计的两点固有性质**：
- 归一化在 **workgroup 内**（8×8）做 ⇒ 每块预算密度相同、块间不会出现差异；但分配是**块内相对**的——整块都噪时块内无人可"让出"，该块退化为 `w≈1`（安全，不放大）。
- 权重上下限 [0.5, 2.0]，噪声估计是创新量的 EMA（滞后一帧）。


## 6. 落地顺序建议（收益 / 风险）

1. ~~**方差引导的自适应采样分配**~~ → **已完成**（见 §5.5：连续 `b_err` + 方差引导，射线 −15%、创新量 −7.4%）。
2. **给面槽加 secondary 层**（on-surface/GIBS 两层结构，secondary 存出射辐射并允许互采样）→ 远景/多弹射 GI 不再发射线（= idTech8 的 final gather 结构）。
3. **远景 GI 换成级联辐照度体积 + 2-band SH**，把 3072 体素 reach 的射线预算让给近场（顺带绕开"阴影射线没法用 beam"的死结）。
4. **内存侧 SVDAG / 颜色属性块**（若要真正推开 2GB 天花板）。
5. **高风险高回报**：屏幕空间 Radiance Cascades（无噪声 GI），3D 部分仍需自有世界缓存。

## 7. 度量纪律（本仓库已踩过的坑）

- 按 pass 的 GPU 时间要用**修好后的 profiler**（`GpuProfilerSettings.enable_debug_groups = false`）。开着时 query 配对错位，会出现"比整帧还长"的 ms 与整趟缺失。
- A/B 必须**同构建特征**：profiler 插桩本身约 1.5ms。
- `RENDER 帧周期` 行里带渲染尺寸；全屏切换时机会改它（960×540 vs 640×360 = 2.25× 像素），跨 run 对比必须核对。
- 静态相机 + 关 vsync = 确定性基准；fly 路径的工作量 run-to-run 波动可达 ±25%。

## 来源

- [Fast as Hell: idTech8 Global Illumination (Sousa, SIGGRAPH 2025)](https://advances.realtimerendering.com/s2025/content/SOUSA_SIGGRAPH_2025_Final.pdf)
- [Path Tracing in DOOM: The Dark Ages](https://static.graphicsprogrammingconference.com/public/2025/talks/a-beautiful-hell-path-tracing-in-doom-the-dark-ages/Khan-Stack-a-beautiful-hell-path-tracing-in-doom-the-dark-ages.pdf)
- [Ray tracing the world of Assassin's Creed Shadows (SIGGRAPH 2025)](https://advances.realtimerendering.com/s2025/content/Advances%202025%20-%20Raytracing%20the%20world%20of%20Assassin's%20Creed%20Shadows.pdf)
- [AC Shadows: inside the revamped Anvil Engine (Digital Foundry)](https://www.digitalfoundry.net/articles/digitalfoundry-2025-assassins-creed-shadows-tech-qa-rtgi-shader-compilation-taa-and-more)
- [Aokana: A GPU-Driven Voxel Rendering Framework for Open World Games](https://arxiv.org/html/2505.02017)
- [Hybrid Voxel Formats for Efficient Ray Tracing](https://arxiv.org/html/2410.14128)
- [Out-of-Core Construction of Sparse Voxel Octrees](https://dl.acm.org/doi/pdf/10.1145/2492045.2492048)
- [Holographic Radiance Cascades for 2D Global Illumination](https://arxiv.org/pdf/2505.02041v1)
- [Radiance Cascades — interactive walkthrough](https://jason.today/rc)
- [Radiance Cascades (Osborne & Sannikov 2024)](https://arxiv.org/pdf/2408.14425v1)
- [ReSTIR GI: Path Resampling for Real-Time Path Tracing](https://research.nvidia.com/publication/2021-06_restir-gi-path-resampling-real-time-path-tracing)
- [ReSTIR PT Enhanced](https://research.nvidia.com/labs/rtr/publication/lin2026restirptenhanced/lin2026restirptenhanced.pdf)
- [Dynamic Diffuse Global Illumination with Ray-Traced Irradiance Fields (JCGT)](https://jcgt.org/published/0008/02/01/paper.pdf)
- [Dynamic Diffuse Global Illumination Resampling](https://arxiv.org/pdf/2108.05263v1)
- [GI-1.0: A Fast Scalable Two-Level Radiance Caching Scheme](https://arxiv.org/pdf/2310.19855v1)
- [Radiance Caching with On-Surface Caches for Real-Time GI](https://dl.acm.org/doi/pdf/10.1145/3675382)
- [Global Illumination based on Surfels (GIBS)](https://www.advances.realtimerendering.com/s2021/SIGGRAPH%20Advances%202021%20-%20Surfel%20GI.pdf)
- [NVIDIA RTX Global Illumination (NRC / SHaRC / DDGI)](https://developer.nvidia.com/rtx/ray-tracing/rtxgi)
- [NVIDIA RTX Mega Geometry (SIGGRAPH 2025)](https://dl.acm.org/doi/pdf/10.1145/3721243.3735983)
- [OGRE-Next: Global Illumination Methods](https://ogrecave.github.io/ogre-next/api/2.3/_gi_methods.html)
- [Practical Real-Time Voxel-Based GI for Current GPUs (3D clipmap)](https://cgvr.informatik.uni-bremen.de/theses/finishedtheses/VoxelConeTracing/S4552-rt-voxel-based-global-illumination-gpus.pdf)
- [Scalable Real-time GI for Large Scenes (GDC 2019)](https://media.gdcvault.com/gdc2019/presentations/Yudintsev_Anton_Scalable_Realtime_Global.pdf)
- [Dynamic Voxel-Based Global Illumination (CGF 2025)](https://onlinelibrary.wiley.com/doi/10.1111/cgf.15262)
- [Godot: Introduction to global illumination](https://docs.godotengine.org/zh_TW/4.x/tutorials/3d/global_illumination/introduction_to_global_illumination.html)
- [Advances in Real-Time Rendering in Games Part II (SIGGRAPH 2025)](https://dl.acm.org/doi/pdf/10.1145/3721241.3744991)
