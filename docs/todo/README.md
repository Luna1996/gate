# gate 项目施工总纲（Voxel 电路解谜游戏）v3

> 单兵开发 · Bevy 引擎 + 自研光追微体素管线 · 多分辨率可变叶体素（4cm/1cm/0.25cm）· 无 Mesh 化
> 渲染先行，游戏逻辑后置；决策记录只保留最终结果，修改/推翻/撤销前的旧结论与过程不落入本文档

**文档组织（2026-09-02 拆分）**：原 TODO.md 过长，按 v3.10 施工队列拆分为多级文件。新编号规则 = 队列前缀 + 文件内顺序号（如 R1-1、R3-9），旧 Px.y 编号保留作历史锚点（`.trae/specs/` 等交叉引用仍可用）。见文末「旧→新 ID 映射表」。

---

## 已定关键决策（勿反复）

| 决策点 | 结论 | 理由摘要 |
|---|---|---|
| 引擎 | **Bevy**（窗口/ECS/资产/UI/相机） | 用户拍板；自研管线以 **render system** 挂入（0.19 无 render node：compute 挂 `RenderGraph` schedule，上屏 pass 挂 `Core2d` 的 `PostProcess` set，ADR-0002/0003） |
| 渲染路线 | 自研 wgpu compute 光追管线（Teardown/VoxTrace 式），**无 Mesh 化** | 微体素高级视觉正统路径；编辑 O(树深)；Bevy 的 mesh/PBR 路线已被否决 |
| 开发顺序 | **基础渲染 → 交互桥接 → 游戏逻辑 → 高级渲染** | 用户拍板；渲染先行可尽早暴露最大技术风险 |
| **数据结构（v4）** | **Douglas Brick Tree 全量换轨，严格 1:1**（1:1 复刻 #17/#18/#22/#23）：**chunk 256³**（Douglas 早期 chunk 大小）+ **分裂树**（root=chunk256³ → down=1³，分裂因子 4³=64，层级 256→64→16→4→1 共 4 层）+ **u64 occupancy mask per non-leaf**；mask 进寄存器后 bitwise AND + popcount 定位子节点，每层零额外 load；uniform leaf **自适应**（无固定最小叶子，大平面停在 level 1/2，小结构分裂到 level 3/4）；palette 直接存 BrickTreeNode fixed；组件 ID 挂 **BrickCoord{level:2}=16³**（天然对齐 gate cell + DDGI probe cell）。**编辑精度 = 1³**（分裂到 level 4），和 gate 当前相同 | 从 v3 的可变叶八叉树（128³ Tile + 2³ 分裂 + 5 level）彻底换轨；VRAM 从 ~2GB → ~350MB（省 ~1.85GB）；每步 storage load 从 2~5 → 1；五步链（TileIndex→Bitmap→Dirs→Node→Slot）被 mask+palette 单次 load 取代；TileBitmaps/CellDirs/b_leaves/TileIndex 全部删除；R2-3 规划全部 SUPERSEDED |
| 世界结构 | **Volume + 保留 Chunk 分层**：全场景 `Vec<Volume>`，每个 volume = chunk 分层（256³，原 Tile 概念扩展）+ ChunkTree（Brick Tree），主世界 = Volume[0]（chunks: HashMap<ChunkCoord, ChunkTree> 无界），独立物体 = Volume[1..N]（通常 1~少数几个 chunk，任意 transform） | Douglas #23 "readding the chunk system" 架构对齐 |
| **编辑成本** | O(Brick Tree 深度 ≤4) + 向下分裂 uniform leaf + 自底向上 merge。chunk 256³ / 4³ 分裂因子 = 4 层（256→64→16→4→1）。体素级编辑流程：定位 BrickCoord → 从 root DFA 找到最下统一 leaf → 分裂该 leaf → 插入新 palette → 向上检查 64 子块是否可 merge | Brick Tree 分裂 + merge 比 TileGrid 直接 cell 编辑更复杂；Phase 0 必须先过 spike 验证（递归 update 正确性 + merge 条件判断 + 紧凑 child offset 更新） |
| 电路语义 | 洪泛连通同材质 = 元件 | 产品核心差异化；连通性同时服务将来物理碎块 |
| 模拟位置 | CPU 事件驱动，**不上 GPU** | 依赖链结构 + 步进/回放刚需；成本∝翻转数而非总数 |
| 状态可视化 | 每体素存元件 ID；状态表每元件 1 字节每 tick 上传 | 状态变化零体素写入；通道在 P3 就用占位数据搭好 |
| 特殊体素 | 调色板视觉变体 / CPU 校验标志 / 侧表数据 三级分流 | 不摊平进体素词 |
| 场景 | **游戏内容范围** = 有界工作间；**引擎能力目标** = 流式大世界（P14 能力线） | CPU 世界本就无界稀疏；tile 粒度上传/重建/脏跟踪天然是流式单位；2GB GPU 绑定上限下大世界**必须**流式（P1.7：百万 tile≈4GB）——流式凡技术上可能即做，是引擎能力而非仅内容取舍 |
| **画质基线（v3）** | **M2 @ GTX 1660 1080p60；M3 @ RTX 3070 1080p60** | 验收写死硬件+分辨率+帧率，杜绝玄学；1660 上 M3 提供降档（关 GI） |
| **资源预算（v3.1）** | 所有极限性能测试必须**断言资源上限**：系统内存 ≤2GB（体素数据+镜像）、VRAM ≤2GB（砖块图+G-Buffer）——**限工作间/关卡场景**；沙盒极值仅要求不崩溃 + 实测数字上报（P1.7 裁决：100 万非空 tile 受 occupancy 4KB/Tile 下限约束 ≈4GB）。关键操作（编辑/洪泛/模拟 tick/砖块构建与上传/DDA）带 CPU 时间预算断言。CI 跑宽松上界（防机器抖动误报），基准机跑正式数字 | 防止功能正确但资源失控；预算不写死则性能目标不可验收、不可回归 |
| **内部观察（v3）** | **仅剖面切割**（DDA 裁剪平面，近零成本），不做 X-ray 半透明 | 编辑/调试封闭电路刚需；管线早期预留 |
| **光照主题（v3）** | 数据驱动配置（光源/环境/曝光资产化），首发仅深调**暗色实验室**；另含**「自然日光」户外主题**（天空/雾/太阳参数资产化，对标 Douglas Dwyer devlog 观感，demo 场景验收用） | M3 调优矩阵不随主题数翻倍；其余主题后置扩展 |
| **动态体素对象 OBJ**（已被 v4 吸收，独立决策项取消） | **已被数据结构 v4 吸收**：Volume 统一架构下（见数据结构 v4 行 + 世界结构行），主世界和独立物体共享完全相同的 ChunkTree 格式（256³ chunk + 4³ 分裂 + u64 mask + compact child offset）、完全相同的 Douglas mask DDA 路径（每层单次 load → bitwise AND → popcount）、完全相同的单一 struct buffer 池。差异只通过 GridDesc（变换矩阵 + 世界 AABB + tree_base/tree_depth/palette_base）吸收。原 BrickMapGlobals vs ObjDesc 分裂、BG0 vs BG2 分裂、kind 分支、obj 五步链、1-tile 限制——**全部删除**。细节见 `docs/unified-grid-plan.md`（v4 chunk256³ 最终版）+ ADR-0007（v4 最终版） | 原 OBJ v6/v7 独立决策项不再需要。统一后 shader 删掉 obj_cell_occupied / obj_sample_voxel 整套实现（~200 行），obj.rs 文件删除，VRAM 共享池化。未来多 chunk 物体、动态 Volume 拆分（碎裂/切割）无障碍。Phase 0 gate-voxel Tile 内重写（chunk 128³→256³ + 4³ 分裂 + 编辑 API）是前置条件 |
| **光源表示（v3.2）** | 方向光 = 带角半径太阳盘（**软阴影**锥采样）；点光 = 球形光（立体角采样）；**发光元件进 NEE 光源列表**（ComponentTable 聚合驱动，数量有界） | Douglas octo 观感对标（PT 光照/软阴影/自发光照明是其标志性画面）；「通电电路照亮暗室」= 电路游戏核心视觉语言，M2 即具备而非等 P9 |
| **光照量化粒度（v5）** | **逐体素 flat shading，1:1 复刻 Douglas #22/#23**：无 per-voxel 光照缓存；直光硬阴影 per-pixel 直接算；间接光从 DDGI probe irradiance 采样；composite 主 pass 一次出 final color | 废弃逐面着色 + face_light hashmap（控制流 bug 反复 + 碰撞率 39%+ 无根治方案）；hashmap 是 #19 PT indirect 时代遗物，Douglas #23 已移除 |
| **体素 normal 存储策略（v4）** | **CPU 侧不存 normal（天生对齐 Douglas #22），GPU 侧 upload 时 bake normal → 渲染缓存**：normal 不是体素固有属性，是可丢弃的派生数据；dirty tile upload 时一次性派生生成，edit 操作不触发 CPU 侧 normal 维护 | gate 从数据结构设计上就不存在 Douglas #22 的 "edit 后同步维护邻居 normal → huge headache" 问题 |
| **纹理路线（v5）** | **Triplanar PBR 纹理系统**（对标 Douglas #22）：upload bake normal + triplanar 纹理坐标 + albedo/roughness/metallic 进 GPU 渲染缓存；shader triplanar 采样 × per-voxel 光照值 = 最终颜色 | 视觉基础升级（纯色块 → 纹理化材质）+ 架构一致性（与 bake normal 同构） |
| **光照管线最终形态（v5）** | **1:1 复刻 Douglas #23，无 per-voxel hashmap**。composite：`final = triplanar_texture × max(sky_grad(直光), ddgi_probe_indirect(间接光)) + emissive_radiance(发光直出)`。直光 = DDA trace 命中后 1 条太阳射线（per-pixel）；间接光 = 最近 8 cell probe irradiance 三线性 × 视线可见性 × 前后权重；emissive = 材质自发光直接加 | DDGI probe worklist（GPU atomic）替代可见体素注册，探针 radiance 存储（八面体映射 2D texture array）替代 radiance 累加；probe endpoint emissive 分支保留（gate 核心画面：通电 LED 珠照亮周围墙） |
| **Douglas #22/#23 架构对齐（v5）** | 天生对齐 #22（隐式 normal + 稀疏树）；本次对齐：废弃逐面着色 + hashmap → 逐体素 flat shading + 无 per-voxel 光照缓存；光照管线对齐 #23（probe worklist + 固定预算射线 + 上一帧 DDGI 自闭环）。需补：①triplanar PBR + bake normal ②region fill/copy 编辑原语 | 省 Douglas 数月重构成本（gate 天生没存过 normal） |
| 渲染里程碑 | M2 = 可发布保底线；M3 = 视觉飞跃线；M4 = 冲刺线 | 每级独立成立 |
| **操作设备（v3）** | 键鼠首发，手柄后置 | 三维体素编辑手柄适配成本极高，P12.4 移出关键路径 |
| **节奏（v3）** | 无硬期限，质量优先；Steam 页面暂不开架，**P12 整体后置** | 里程碑表是唯一范围纪律；Platform trait 抽象保留在 P0 防后补成本 |
| **测试（v3）** | 自测为主（确定性回放 + 基准场景）；真人试玩降级为未来渠道 | 现无社群；P7.5/P8.6 按自测口径编写 |
| Steam SDK | 薄集成 + Platform trait（Null 后端） | 构建管线早铺，成就/工坊晚做 |

---

**UNRESOLVED（推迟裁决，带触发点）**

| 问题 | 触发裁决点 |
|---|---|
| 白名单连接具体语义（接触=建图边 vs 白名单对合并元件） | **P5.1 动工前** |
| 编辑刷子粒度切换机制（跟随命中 / 手动档位 / 按材质自动） | **P4.1 动工前** |
| **物体（OBJ）网格是否进洪泛/模拟**——修复判定 = 图案比对（便宜）vs 物体真实通电（需 DSU/模拟器作用域扩展到物体网格，贵一个量级） | **P10.5 动工前** |
| **细结构着色轮廓不清**——导线等 1 体素厚结构无立体感 | **已解决**：triplanar PBR 纹理化材质自然提供轮廓差异（砖缝/金属反光）；若实测仍需增强，备选：边缘描边后处理（depth/normal Sobel 勾边）/ 玩法侧状态色补偿（通电发光 + hover 轮廓）。触发点 = M2 验收后搭电路导线场景实测 |
| 统一 UI 框架选型 | **已裁决**：bevy_ui 原生 + 自研组件库（gate-ui），风格现代化参考 Minecraft Modern UI mod——组件库规模见 2.7 前置条目 |

**工程既定**：测试 UI 与游戏内 UI 使用同一套框架——**bevy_ui 原生 + 自研组件库 gate-ui**（egui/bevy_lunex/混合方案否决：egui 数据面板生产力虽强但依赖第三方+观感调试工具化，lunex 版本滞后仅布局引擎）；Modern UI 风格 = 主题令牌（颜色/圆角/间距/字体数据资产化，对齐光照主题哲学）+ 自研 widget；光照主题为纯数据资产（不含硬编码）。

**Bevy 集成要点**（新增风险项，P0 就要打通）：
- 自研管线 = Bevy `RenderApp` 里的 render system（0.19 无 render node：compute 挂 `RenderGraph` schedule，上屏 pass 挂 `Core2d` 的 `PostProcess` set，ADR-0002/0003）
- 从 Bevy 相机取视图矩阵喂给 DDA；输出写回 Bevy view target（后处理/tonemap 可复用其链路）
- buffer/texture 用 `RenderDevice` 创建，与 Bevy 资源生命周期对齐
- Bevy 版本锁定：render graph API 随版本破坏性变更频繁，开工即锁版本，周期内不追新

---

**参考仓库**

| 仓库 | 用途 | 阶段 |
|---|---|---|
| [dust](https://github.com/dust-engine/dust) | Rust 光追体素先例；`pumicite` = Rust 版 VDB 式稀疏索引参考实现 | 全程 |
| [bonsai](https://github.com/scallyw4g/bonsai) | 八叉树流式世界 + 自研线程池的 C 参考实现（想法密度高，风格奇特） | P2/P9 |
| [roxlap](https://github.com/NCrashed/roxlap) | 体素编辑/流式/拾取 API 设计参考 | P1/P4 |
| [bevy_vox_scene](https://github.com/oliver-dew/bevy_vox_scene) | 视觉基准（M3 验收对照物）+ 预制件资产管线思路 | P3/P10 |
| [voxel-engine-v0](https://github.com/aleksanderhan/voxel-engine-v0) | wgpu 多pass帧图 + SVO/DDA 的 Rust 代码参考 | P0/P2 |
| [VoxTrace](https://github.com/MarcinJablonowski/VoxTrace) | 可运行效果参照（仅阅读源码，无许可证不得复制） | P3/P9 |

**核心文献（用户指定，已定位）**

| 文献 | 用途 | 阶段 |
|---|---|---|
| [GigaVoxels (Crassin et al., I3D 2009)](https://maverick.inria.fr/Publications/2009/CNLE09/CNLE09.pdf) | **管线奠基**：SVO/砖块射线投射 + 流式，P2 理论总纲 | P2 |
| [High-Resolution Sparse Voxel DAGs (Kämpe et al., 2013)](https://www.cse.chalmers.se/~uffe/HighResolutionSparseVoxelDAGs.pdf) | DAG 压缩结构——注意 DAG 写入弱项与海量编辑冲突，用于明确"为何不选" | P2 |
| [Efficient Stream Compaction (Billeter et al., HPG 2009)](https://www.cse.chalmers.se/~uffe/streamcompaction.pdf) | GPU 并行压缩原语：砖块图构建/稀疏结构维护 | P2 |
| [A Ray-Box Intersection Algorithm and Efficient Dynamic Voxel Rendering (Majercik et al., JCGT 2018)](https://jcgt.org/published/0007/03/04/paper.pdf) | 快速射线-盒求交（拾取/碰撞复用）；动态体素吞吐对照基线 | P4/P11 |

**补充文献（自选）**

| 文献 | 用途 |
|---|---|
| [Amanatides & Woo, A Fast Voxel Traversal Algorithm (1987)](http://www.cse.yorku.ca/~amana/research/grid.pdf) | 3D DDA 起源，一切遍历代码的原型 |
| [Laine & Karras, Efficient Sparse Voxel Octrees (I3D 2010)](https://research.nvidia.com/sites/default/files/pubs/2010-02_Efficient-Sparse-Voxel/octrees.pdf) | ropes 邻域步进（voxel-engine-v0 即用此法），DDA 加速关键 |
| [Museth, NanoVDB (2021)](https://research.nvidia.com/labs/prl/nanovdb/nanovdb2021.pdf) | GPU 友好 VDB 线性内存布局最佳教材——**抄结构不引依赖** |
| [Majercik et al., **Dynamic Diffuse Global Illumination with Ray-Traced Irradiance Fields (JCGT 2019)**](https://jcgt.org/published/0008/02/01/paper.pdf) | **R3-10 必读 1/3**：原论文，八面体映射 irradiance 存储布局 + 探针射线更新 + 视线感知查询；附录有 GLSL 参考代码 |
| [Majercik et al., **Scaling Probe-Based Real-Time Dynamic Global Illumination for Production (JCGT 2021)**](https://jcgt.org/published/0010/02/01/paper-lowres.pdf) | **R3-10 必读 2/3**：生产改进，探针状态机剪枝 + 级联多分辨率体积 + self-shadow bias 调参；R1-10/R1-13 流式大世界直接复用 |
| [Roháček & Iser, **Improving Probes in Dynamic Diffuse Global Illumination (CESCG 2022)**](https://old.cescg.org/CESCG-2025/wp-content/uploads/2022/04/Rohacek-Improving-Probes-in-Dynamic-Diffuse-Global-Illumination.pdf) | **R3-10 必读 3/3**：漏光治理，改进探针放置（BFS 找 cell 内最大空叶）+ 锐利背面剔除 |
| [SVGF (Heitz et al., HPG 2019)](https://research.nvidia.com/labs/publications/spatiotemporal-variance-guided-filtering-real-time-reconstruction-path-traced-global-illumination) | 降噪器候选：A-Trous 的时域方差引导升级版（DDGI 路线下若有远距离闪烁可重开本链） |
| [ReSTIR GI (Bitterli et al., 2020)](https://cs.dartmouth.edu/~wjarosz/publications/bitterli20restir.pdf) | M4 冲刺线的重采样 GI |
| [Raytracing Voxels in Teardown (GPUC 2025)](https://static.graphicsprogrammingconference.com/public/2025/slides/raytracing-voxels-in-teardown/Rundlett-Gustafsson-raytracing-voxels-in-teardown-and-beyond.pdf) | 施工总纲：Sparse Shapes / Indexed Palette / 半分辨率GI |
| [Teardown Frame Teardown (acko.net)](https://acko.net/blog/teardown-frame-teardown/) | 管线细节与优化动机的逐帧实证 |

**工程参考**：Bevy 自定义 render node 官方示例（`custom_render_phase` 等，P0 关键）· `bevy_vox_scene` 源码（.vox→材质映射）· OpenVDB/NanoVDB 决策：**运行时不引入**，仅在离线工具链（CSG/导出）场景可作构建期依赖

---

## 建议执行顺序（关键路径）

```text
P0 Bevy基建 → P1 最小数据层 → P2 渲染M1（体素上屏）
→ P3 渲染M2（光影保底线）→ P4 交互桥接（可编辑闭环）
→ P5 洪泛元件 → P6 事件驱动模拟（游戏内核成型）
→ P7 完整编辑UX → P8 谜题框架 → P9 渲染M3（GI飞跃）
→ P10 场景资产 → P11 冲刺随时插空（P12 平台发布整体后置）
```

OBJ 动态体素对象线（随主线穿插推进）：**P2.10 多网格渲染 → P4.7 平滑移动/交互 → P10.5 物体编辑=修复 → P11.6 烘焙优化（可推迟）**

流式大世界引擎线（与内容线解耦的引擎能力线）：**P14 全部为引擎能力、零游戏内容耦合**——14.1 tile 驻留管理器可在 P3 后任意点插入（机制全是 P2.3/P2.9/P2.10 雏形扩用），14.2 程序化分页 → 14.3 LOD → 14.4 重定基+剔除（依赖 M4 冲刺位），与 OBJ 线并行推进

**v3.10 施工队列重排（2026-09-02，用户指令）**：未完成任务已按「数据结构 > 基础渲染 > 光影效果 > 玩法外壳 > 其他」重排进下方 **R1-R5 施工队列**；原 P3-P14 阶段结构改为「P0-P2 + P3 已完成项存档 + R1-R5 队列」，阶段号保留作 ID（全文交叉引用不变），同类内保持原编号顺序。上图为里程碑依赖视图（两个生死里程碑不变），R1-R5 为施工优先级视图；M2 验收门（3.7）仍由 R2+R3 完成后触发。

两个生死里程碑：
1. **P2.7 首个画面上屏**——砖块图 + DDA 是否真的能在 Bevy 里跑起来
2. **P4.2 编辑闭环**——点击到画面同帧更新，此后一切迭代都有载体

---

## 里程碑验收标准汇总

| 里程碑 | 验收标准 | 状态 |
|---|---|---|
| **M1（P2）** | 测试体素场景经砖块图+DDA 在 Bevy 内上屏，相机自由飞行，1080p @ GTX 1660 无异常 | ☑（P2.8 2026-09-01 验收） |
| **M2（P3）** | 暗色实验室主题光影实时渲染（**per-pixel 硬阴影 + DDGI 间接光 + Triplanar PBR 纹理化材质 + 发光元件直接照明**）；**「自然日光」户外主题（程序化天空/距离雾/天空环境光）对照 Douglas Dwyer 截图观感验收**；1080p60 @ GTX 1660；万体素编辑不掉帧；理论可上架 | ☐ |
| **可编辑（P4）** | 点击放/删体素画面同帧更新，连续操作流畅，剖面切割可用 | ☐ |
| **OBJ 底座（P2.10+P4.7+P10.5）** | 静态/平滑移动的多网格体素对象渲染正确；NPC 寄件电子产品可被逐体素修复 | ☐ |
| **内核（P5+P6）** | 洪泛成元件 → 事件驱动模拟 → 通电变色全链路真数据跑通 | ☐ |
| **可玩（P7+P8）** | 教学关自测通过：无说明能完成并想优化解法 | ☐ |
| **M3（P9）** | PT GI + DDGI + 降噪稳定 @ RTX 3070 1080p60，观感超越 bevy_vox_scene 参考图；**god rays 介质散射 + Douglas Dwyer 图1/图5 对照达标** | ☐ |
| **M4（P11）** | VoxTrace demo 同级表现 | ☐ |
| **引擎能力·流式（P14）** | ≥10 万 tile 程序化世界连续飞行：GPU 驻留有界、无卡顿尖峰、LOD 无闪烁、>4km 精度正确 | ☐ |

---

## 旧→新 ID 映射表

> 新编号格式：**R{N}-{m}**（队列前缀 + 文件内顺序号）。旧 Px.y 编号保留作历史锚点，`.trae/specs/` 等交叉引用仍可用。

### R1 数据结构（→ [r1-data-structure.md](r1-data-structure.md)）

| 旧 ID | 新 ID | 任务 |
|---|---|---|
| 5.1 | R1-1 | tile 内两遍连通标记 + CCL 加速 |
| 5.2 | R1-2 | 跨 tile DSU |
| 5.3 | R1-3 | 增量维护 |
| 5.4 | R1-4 | comp_layer 写回 + comp_dirty |
| 5.5 | R1-5 | ComponentTable |
| 5.6 | R1-6 | 端口系统 |
| 5.7 | R1-7 | 元件层镜像上传通道接真数据 |
| 5.8 | R1-8 | 压测 |
| 5.9 | R1-9 | 确定性测试 |
| 14.1 | R1-10 | S1 tile 驻留管理器 |
| 14.2 | R1-11 | S2 程序化世界生成分页源 |
| 14.3 | R1-12 | S3 LOD |
| 14.4 | R1-13 | S4 精度与剔除 |
| 14.5 | R1-14 | P14 验收 |

### R2 基础渲染（→ [r2-base-rendering.md](r2-base-rendering.md)）

| 旧 ID | 新 ID | 任务 |
|---|---|---|
| 3.3 | R2-1 | 状态调制通道 |
| 3.5e | R2-2 | 方向位掩码 LUT 预过滤 |
| 3.5f | R2-3 | DDA 访存压缩三件套 |
| 3.5g | R2-4 | Beam 两级渲染 + 阴影降档 |

### R3 光影效果（→ [r3-lighting.md](r3-lighting.md)）

| 旧 ID | 新 ID | 任务 |
|---|---|---|
| 3.1a | R3-1 | ~~逐面着色~~ — **v5 废弃**（对齐 Douglas #22/#23 逐体素 flat shading） |
| 3.4a | R3-2 | 密度场 AO |
| 3.5 | R3-3 | 后处理：ACES + sRGB + 轻量 bloom |
| 3.5b | R3-4 | 距离雾/大气透视 |
| 3.5c | R3-5 | God rays 屏幕空间廉价版 |
| 3.5d+ | R3-6 | ~~Phase 2 hashmap~~ — **v5 废弃**（1:1 复刻 Douglas #23 移除 hashmap） |
| 3.6 | R3-7 | 锁定体素全息视觉变体 |
| 3.7 | R3-8 | 性能验收 M2 |
| 9.1a | R3-9 | ~~半分辨率路径追踪 pass~~ — **已取消**（被 DDGI hybrid 取代） |
| 9.1b | R3-10 ⚠️ | **DDGI 全局光照（1:1 复刻 Douglas #23）** — probe worklist + 固定预算射线 + 上一帧 DDGI 自闭环；probe endpoint emissive 分支保留 |
| 9.2 | R3-11 | 时域累积 + 相机运动重投影 |
| 9.3 | R3-12 | ~~À-trous 边缘感知小波去噪~~ — **已取消**（DDGI 探针天然时间平均） |
| 9.4 | R3-13 | 上采样 + 与 M2 直光合成 |
| 9.5 | R3-14 | 蓝噪声序列管理 |
| 9.6 | R3-15 | 实机调优迭代回路 |
| 9.7 | R3-16 | 验收 M3 |
| 9.8 | R3-17 | God rays 介质散射升级（DDGI 探针射线内） |

### R4 玩法外壳（→ [r4-gameplay.md](r4-gameplay.md)）

| 旧 ID | 新 ID | 任务 |
|---|---|---|
| 4.1 | R4-1 | CPU 拾取 |
| 4.2 | R4-2 | 编辑闭环 |
| 4.3 | R4-3 | 视觉反馈 |
| 4.4 | R4-4 | 基础工具 |
| 4.5 | R4-5 | 剖面切割 |
| 4.6 | R4-6 | P4 验收 |
| 4.7 | R4-7 | OBJ-2 平滑移动与交互 |
| 6.1 | R4-8 | 元件行为语义表 |
| 6.2 | R4-9 | 事件驱动核心 |
| 6.3 | R4-10 | 固定步长模拟时钟 |
| 6.4 | R4-11 | StateTable 更新 + 每 tick 上传 |
| 6.5 | R4-12 | 每 tick 工作预算 |
| 6.6 | R4-13 | 调试接口 |
| 6.7 | R4-14 | 确定性回放 |
| 6.8 | R4-15 | 差量存档 |
| 6.9 | R4-16 | 压测 |
| 6.10 | R4-17 | 线程池纪律 |
| 7.1 | R4-18 | 批量操作 |
| 7.2 | R4-19 | 撤销重做 |
| 7.3 | R4-20 | 观测工具 UI |
| 7.4 | R4-21 | 相机打磨 |
| 7.5 | R4-22 | 音频占位 |
| 7.6 | R4-23 | 手感迭代 |
| 8.1 | R4-24 | 关卡格式 |
| 8.2 | R4-25 | 编辑校验层 |
| 8.3 | R4-26 | 目标判定 |
| 8.4 | R4-27 | 评分 |
| 8.5 | R4-28 | 解法字符串编解码 |
| 8.6 | R4-29 | 谜题内容 |
| 8.7 | R4-30 | 沙盒模式 |
| 10.1 | R4-31 | .vox 导入器 |
| 10.2 | R4-32 | 工作间雕刻 + 打光 |
| 10.3 | R4-33 | 元件预制件库 |
| 10.4 | R4-34 | 背景与工作区隔离 |
| 10.5 | R4-35 | OBJ-3 物体编辑 = 修复玩法 |

### R5 其他（→ [r5-other.md](r5-other.md)）

| 旧 ID | 新 ID | 任务 |
|---|---|---|
| 3.8 | R5-1 | 达成即理论可上架 |
| 11.1 | R5-2 | ReSTIR 精修直光 |
| 11.2 | R5-3 | 玻璃透射 + 半透明路线 |
| 11.3 | R5-4 | 逐体素动画 |
| 11.6 | R5-5 | OBJ-4 烘焙/解烘焙优化 |
| 11.7 | R5-6 | 体素物理避坑清单 |
| 12.1 | R5-7 | steamworks-rs 薄集成 + Platform trait |
| 12.2 | R5-8 | steamcmd + VDF 脚本化构建上传 |
| 12.3 | R5-9 | Steam Cloud |
| 12.4 | R5-10 | Steam Input 手柄映射 |
| 12.5 | R5-11 | 设置系统 |
| 12.6 | R5-12 | 本地化框架 |
| 12.7 | R5-13 | 冲刺期商店素材 |
| 13.1 | R5-14 | 确定性测试 CI 化 |
| 13.2 | R5-15 | 性能基准场景库 |
| 13.3 | R5-16 | 跨 GPU 验证 |
| 13.4 | R5-17 | 存档版本迁移测试 |
| 13.5 | R5-18 | panic 隔离策略 |

### R6 纹理与材质（→ [r6-textures.md](r6-textures.md)）

| 旧 ID | 新 ID | 任务 |
|---|---|---|
| N/A | R6-1 | Bake normal spike（CPU Rust 验证邻域分析算法） |
| N/A | R6-2 | Baked attributes buffer 格式设计 |
| N/A | R6-3 | Dirty upload + bake pipeline 集成 |
| N/A | R6-4 | Shader triplanar 采样 + composite pass 集成 |
| N/A | R6-5 | Texture atlas + PBR 材质资产化 |
| N/A | R6-6 | Region fill 编辑原语 |
| N/A | R6-7 | Region copy 编辑原语 |
| N/A | R6-8 | CSG 基础（sphere/cylinder/torus 栅格化） |
| N/A | R6-9 | Fill/copy 上传集成验证 |
| N/A | R6-10 | 性能预算断言 |
| N/A | R6-11 | 视觉验收（triplanar + bake normal + DDGI hybrid 叠加） |
| N/A | R6-12 | Fallback 验证 |

---

## 文件索引

| 文件 | 内容 |
|---|---|
| [README.md](README.md) | 本文件：决策表 + 里程碑 + ID 映射 + 文件索引 + changelog |
| [completed.md](completed.md) | P0-P3 已完成项存档（只读参考） |
| [r1-data-structure.md](r1-data-structure.md) | R1 数据结构（P5 洪泛元件 + P14 流式大世界） |
| [r2-base-rendering.md](r2-base-rendering.md) | R2 基础渲染 |
| [r3-lighting.md](r3-lighting.md) | R3 光影效果（直光 + DDGI + 后处理） |
| [r4-gameplay.md](r4-gameplay.md) | R4 玩法外壳（P4+P6+P7+P8+P10） |
| [r5-other.md](r5-other.md) | R5 其他（P11 冲刺线 + P12 平台后置 + P13 测试） |
| [r6-textures.md](r6-textures.md) | R6 纹理与材质（triplanar PBR + bake normal + CSG/fill/copy — 对标 Douglas #22） |
| [texture-pipeline.md](../texture-pipeline.md) | **架构规格文档**：triplanar + bake normal 完整设计 + 实现 checklist |

---

## Changelog

### v5（2026-09-02）— 1:1 复刻 Douglas #22/#23

**决策变更**：
- 废弃逐面着色 + face_light hashmap 管线 → 逐体素 flat shading + 无 per-voxel 光照缓存
- 光照管线最终形态改为 1:1 复刻 Douglas #23：probe worklist + 固定预算射线 + 上一帧 DDGI 自闭环；composite 公式移除 hashmap
- Triplanar PBR 纹理系统（对标 Douglas #22）新增决策条目：upload bake normal + triplanar 纹理坐标进 GPU 渲染缓存
- 架构对齐 Douglas #22/#23：gate 天生对齐隐式 normal + 稀疏树；需补 triplanar + region fill/copy

**原因**：hashmap 控制流 bug 反复（2 天内两次修）+ 碰撞率 39%+ 无根治方案；Douglas #23 已移除 hashmap 是架构级组件而非 gate 必须；逐体素 flat shading 是 Douglas 最终方案且更简洁

**清理**：
- 删除 `docs/phase2-temporal-reuse-pinned-slot.md`（hashmap 技术方案文档）
- R3-1/R3-6 标记废弃；ID 映射表更新
- UNRESOLVED「细结构着色轮廓不清」更新为 triplanar PBR 纹理化自然解决
