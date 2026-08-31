# gate 项目施工总纲（Voxel 电路解谜游戏）v3

> 单兵开发 · Bevy 引擎 + 自研光追微体素管线 · 多分辨率可变叶体素（4cm/1cm/0.25cm）· 无 Mesh 化
> v2 变更：引擎定 Bevy；顺序改为**渲染先行，游戏逻辑后置**
> v3 变更：grill-me 拷问裁决落地——数据结构定**单一可变叶八叉树**；画质验收写死硬件基线；光照主题数据驱动（首发暗色实验室）；P12 整体后置；编辑成本表述修正
> v3.1 变更：**全链路极限性能测试 + 资源预算上断言**——数据层/渲染/模拟的压测必须断言内存与 CPU 用量上限，CI 跑宽松预算防回归，基准机出正式数字
> v3.2 变更（2026-08-31）：**新增 MOV 动态体素对象规划**（P2.10/P4.7/P10.5/P11.6 四阶段穿插，NPC 寄件修复玩法底座）；**M2 光源表示升级**（软阴影 + 发光元件进光源列表，对标 Douglas octo 观感）；P2.4 补录两级 DDA 性能改造（DC 43ms→2.75ms，15.6×）
> v3.3 变更（2026-08-31）：**场景决策修订**——「不做流式」降级为游戏内容范围妥协（用户澄清：真实意图 = 引擎能力推至极限，流式凡技术上可能即做）；**新增 P14 流式大世界引擎能力线**（tile 驻留 / 程序化分页 / LOD / 重定基剔除），P11.4/P11.5 升格并入；游戏内容决策全部维持推迟
> 最后更新：2026-08-31

---

## 已定关键决策（勿反复）

| 决策点 | 结论 | 理由摘要 |
|---|---|---|
| 引擎 | **Bevy**（窗口/ECS/资产/UI/相机） | 用户拍板；自研管线以 **render system** 挂入（0.19 无 render node：compute 挂 `RenderGraph` schedule，上屏 pass 挂 `Core2d` 的 `PostProcess` set，ADR-0002/0003） |
| 渲染路线 | 自研 wgpu compute 光追管线（Teardown/VoxTrace 式），**无 Mesh 化** | 微体素高级视觉正统路径；编辑 O(树深)；Bevy 的 mesh/PBR 路线已被否决 |
| 开发顺序 | **基础渲染 → 交互桥接 → 游戏逻辑 → 高级渲染** | 用户拍板；渲染先行可尽早暴露最大技术风险 |
| **数据结构（v3）** | **单一可变叶八叉树**：基元胞 4cm，工作区 1cm，热点 0.25cm（最多 4 级细分） | 一条 DDA 路径贯穿，无跨结构转换；细分由编辑行为驱动、按需局部发生 |
| 世界结构 | 无界稀疏 `HashMap<TileCoord, Tile>`，Tile 内含可变叶八叉树 | 沙盒无限扩张；谜题边界只是校验层属性 |
| **编辑成本（v3）** | O(树深≤5) + 局部 brick 分配（跨级放置触发 4096 槽 brick 初始化与祖先掩码更新） | 替代 v2 的「严格 O(1)」表述；仍为常数级，代价集中在跨级编辑 |
| 电路语义 | 洪泛连通同材质 = 元件 | 产品核心差异化；连通性同时服务将来物理碎块 |
| 模拟位置 | CPU 事件驱动，**不上 GPU** | 依赖链结构 + 步进/回放刚需；成本∝翻转数而非总数 |
| 状态可视化 | 每体素存元件 ID；状态表每元件 1 字节每 tick 上传 | 状态变化零体素写入；通道在 P3 就用占位数据搭好 |
| 特殊体素 | 调色板视觉变体 / CPU 校验标志 / 侧表数据 三级分流 | 不摊平进体素词 |
| 场景（v3.3 修订） | **游戏内容范围** = 有界工作间；**引擎能力目标** = 流式大世界（P14 能力线） | CPU 世界本就无界稀疏；tile 粒度上传/重建/脏跟踪天然是流式单位；2GB GPU 绑定上限下大世界**必须**流式（P1.7：百万 tile≈4GB）——「不做流式」是内容范围妥协，非引擎上限 |
| **画质基线（v3）** | **M2 @ GTX 1660 1080p60；M3 @ RTX 3070 1080p60** | 验收写死硬件+分辨率+帧率，杜绝玄学；1660 上 M3 提供降档（关 GI） |
| **资源预算（v3.1）** | 所有极限性能测试必须**断言资源上限**：系统内存 ≤2GB（体素数据+镜像）、VRAM ≤2GB（砖块图+G-Buffer）——**限工作间/关卡场景**；沙盒极值仅要求不崩溃 + 实测数字上报（P1.7 裁决：100 万非空 tile 受 occupancy 4KB/Tile 下限约束 ≈4GB）。关键操作（编辑/洪泛/模拟 tick/砖块构建与上传/DDA）带 CPU 时间预算断言。CI 跑宽松上界（防机器抖动误报），基准机跑正式数字 | 防止功能正确但资源失控；预算不写死则性能目标不可验收、不可回归 |
| **内部观察（v3）** | **仅剖面切割**（DDA 裁剪平面，近零成本），不做 X-ray 半透明 | 编辑/调试封闭电路刚需；管线早期预留 |
| **光照主题（v3）** | 数据驱动配置（光源/环境/曝光资产化），首发仅深调**暗色实验室** | M3 调优矩阵不随主题数翻倍；其余主题后置扩展 |
| **动态体素对象 MOV（v3.2）** | 独立小 brickmap + 变换（位置/旋转/缩放），**不进世界网格**；渲染 = 世界 DDA + 逐物体局部 DDA（射线变换 + OBB 剔除取最近命中），统一 `trace_scene()` 抽象 | NPC 寄件修复玩法的底座；Teardown/octo 同法先例；两级 DDA（P2.4 改造）原样复用，物体=小号 TileGrid 全链复用；与「有界工作间」决策不冲突（实体≠流式世界） |
| **光源表示（v3.2）** | 方向光 = 带角半径太阳盘（**软阴影**锥采样）；点光 = 球形光（立体角采样）；**发光元件进 NEE 光源列表**（ComponentTable 聚合驱动，数量有界） | Douglas octo 观感对标（PT 光照/软阴影/自发光照明是其标志性画面）；「通电电路照亮暗室」= 电路游戏核心视觉语言，M2 即具备而非等 P9 |
| 渲染里程碑 | M2 = 可发布保底线；M3 = 视觉飞跃线；M4 = 冲刺线 | 每级独立成立 |
| **操作设备（v3）** | 键鼠首发，手柄后置 | 三维体素编辑手柄适配成本极高，P12.4 移出关键路径 |
| **节奏（v3）** | 无硬期限，质量优先；Steam 页面暂不开架，**P12 整体后置** | 里程碑表是唯一范围纪律；Platform trait 抽象保留在 P0 防后补成本 |
| **测试（v3）** | 自测为主（确定性回放 + 基准场景）；真人试玩降级为未来渠道 | 现无社群；P7.5/P8.6 相应改写 |
| Steam SDK | 薄集成 + Platform trait（Null 后端） | 构建管线早铺，成就/工坊晚做 |

**UNRESOLVED（推迟裁决，带触发点）**

| 问题 | 触发裁决点 |
|---|---|
| 白名单连接具体语义（接触=建图边 vs 白名单对合并元件） | **P5.1 动工前** |
| 编辑刷子粒度切换机制（跟随命中 / 手动档位 / 按材质自动） | **P4.1 动工前** |
| **物体（MOV）网格是否进洪泛/模拟**——修复判定 = 图案比对（便宜）vs 物体真实通电（需 DSU/模拟器作用域扩展到物体网格，贵一个量级） | **P10.5 动工前** |
| ~~统一 UI 框架选型~~ **已裁决（2026-08-30）**：bevy_ui 原生 + 自研组件库（gate-ui），风格现代化参考 Minecraft Modern UI mod | 已裁决——组件库规模见 2.7 前置条目 |

**工程既定**：测试 UI 与游戏内 UI 使用同一套框架——**bevy_ui 原生 + 自研组件库 gate-ui**（2026-08-30 裁决，egui/bevy_lunex/混合方案否决：egui 数据面板生产力虽强但依赖第三方+观感调试工具化，lunex 版本滞后仅布局引擎）；Modern UI 风格 = 主题令牌（颜色/圆角/间距/字体数据资产化，对齐光照主题哲学）+ 自研 widget；光照主题为纯数据资产（不含硬编码）。

**Bevy 集成要点**（新增风险项，P0 就要打通）：
- 自研管线 = Bevy `RenderApp` 里的自定义 render graph node（全屏 compute 系列）
- 从 Bevy 相机取视图矩阵喂给 DDA；输出写回 Bevy view target（后处理/tonemap 可复用其链路）
- buffer/texture 用 `RenderDevice` 创建，与 Bevy 资源生命周期对齐
- Bevy 版本锁定：render graph API 随版本破坏性变更频繁，开工即锁版本，周期内不追新

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
| [SVGF (Heitz et al., HPG 2019)](https://research.nvidia.com/labs/publications/spatiotemporal-variance-guided-filtering-real-time-reconstruction-path-traced-global-illumination) | 降噪器候选：A-Trous 的时域方差引导升级版 |
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

MOV 动态体素对象线（v3.2，随主线穿插推进）：**P2.10 多网格渲染 → P4.7 平滑移动/交互 → P10.5 物体编辑=修复 → P11.6 烘焙优化（可推迟）**

流式大世界引擎线（v3.3，与内容线解耦的引擎能力线）：**P14 全部为引擎能力、零游戏内容耦合**——14.1 tile 驻留管理可在 P3 后任意点插入（机制全是 P2.3/P2.9/P2.10 雏形扩用），14.2 程序化分页 → 14.3 LOD → 14.4 重定基+剔除（依赖 M4 冲刺位），与 MOV 线并行推进

两个生死里程碑：
1. **P2.7 首个画面上屏**——砖块图 + DDA 是否真的能在 Bevy 里跑起来
2. **P4.2 编辑闭环**——点击到画面同帧更新，此后一切迭代都有载体

---

## P0 Bevy 基建

- [x] 0.1 Cargo workspace 骨架
  - [x] `gate-app`：Bevy 主程序、游戏状态机、输入、UI（egui 工具窗 + bevy_ui HUD）
  - [x] `gate-voxel`：权威数据层 + 洪泛 + 模拟（纯逻辑，零渲染依赖）
  - [x] `gate-render`：Bevy plugin，承载全部自定义渲染 pass
  - [x] 依赖方向：app → {voxel, render}；render 只读 voxel 的镜像
- [x] 0.2 Bevy 版本锁定 + 对齐记录（render graph API 随版本变，写进 ADR → docs/decisions.md ADR-0001；Bevy =0.19.1，wgpu 29.0.4，含 0.19 feature 体系变更记录）
- [x] 0.3 自定义 render node 脚手架（**最高优先风险项**）——**0.19 已无 render node 概念，实为 RenderGraph schedule 中的 render system**（ADR-0002/0003）
  - [x] 全屏 compute → storage texture → 上屏的最小链路（阶段 A：Sprite 显示验证链路；DX12 后端）
  - [x] compute 输出 blit → **ViewTarget** 直写（阶段 B 完成：Sprite 已删，blit 挂 `Core2d` 的 `PostProcess` set——MainPass clear 后、upscaling 上屏前；`camera_driver` 之后挂载**无效**，surface copy 在相机图内部完成）
  - [x] RenderApp 资源生命周期管理（ExtractResource / RenderStartup / PrepareBindGroups 已验证）
  - [ ] Bevy 相机视图矩阵 → system 可见的 uniform
  - 踩坑记录：ViewTarget 是 `Rgba8UnormSrgb`（blit 管线格式必须匹配，输出前 sRGB→linear 抵消硬件编码）；**Msaa 0.19 是 per-view Component**（非资源，默认 4x 与自定义管线 sample count 冲突即崩）；沙箱拦截 `target/debug/incremental` 写入伪装成 rustc ICE（`incremental = false` 规避，ADR-0003）
- [x] 0.4 CI：Windows build + test + clippy（git 仓库已初始化；`scripts/ci.ps1` 本地一键 = fmt --check + clippy -D warnings + build + test，已验证 PASSED；`.github/workflows/ci.yml` 备用——windows-latest + rust-cache，推 GitHub 后即生效）
- [x] 0.5 tracing 日志 + 帧时间统计基座（LogPlugin filter 显式覆盖为 `"info"` 全开——默认值会静音 wgpu，违背"不隐藏日志"原则；FrameTimeDiagnosticsPlugin + LogDiagnosticsPlugin 每秒输出 fps/frame_time，实测 60fps@16.7ms；P2.7 接 GPU timestamp，P7 接 UI）
- [x] 0.6 `docs/decisions.md`：决策表落成 ADR（索引表 + ADR-0001~0004；ADR-0004 定存放原则：TODO 决策表 = 唯一真源，ADR 只存工程细节与坑，防双源漂移）

## P1 体素数据层（最小版，先服务渲染）

- [x] 1.1 基础类型：`VoxelPos`（含叶子层级）/ `TileCoord`（i32³）；TILE = 32 基元胞（4cm）；分层寻址无边界常量（fine 坐标 = 0.25cm 单位 i32³，欧氏除法保证负坐标落邻接 Tile）
- [x] 1.2 Tile：32³ 基元胞占用位掩码 + 每基元胞可变叶八叉树（最多 4 级细分至 0.25cm；长期不用的细分可合并回收）
  - [x] 叶子存 `palette_idx`；整块同色粗叶压缩存储（cell.uniform 快路径 + 同色向上折叠至 L1/L0，破坏性粗写清深层，清除时逐级收缩）
  - [x] `comp_layer: Option<Box<[u16]>>` 字段占位（洪泛后置到 P5）
  - [x] v3.1 内存修正：`l4: Option<Box<Brick>>`（Brick 520B 内联曾把 uniform 粗叶胞撑到 ~560B，装箱后 ~40B，均匀粗叶场景内存 14×）
- [x] 1.3 Palette：256 × `PaletteEntry { color, roughness, emissive, transmission, flags }`（u8 索引 + flags 视觉变体；槽 0 = AIR 保留，set_voxel debug_assert 禁写）
- [x] 1.4 TileGrid：HashMap + `get/set_voxel`（跨级放置 → brick 分配 + 祖先掩码更新）+ `batch_edit`，编辑返回受影响区域集合（`DirtyEdit { tile, level }`，P2.3 按此增量上传）
  - [x] v3.1 内存修正：`HashMap<TileCoord, Box<Tile>>`（Tile 含 4KB occupancy，内联进桶曾致百万 tile 桶数组 12.3GB 且 rehash 搬运 12.8s；装箱后 4.2GB / 1.9s）
  - [x] `memory_usage()`：深尺寸内存核算（Tile/Cell/Brick/DirtyTracker），预算断言依据
- [x] 1.5 脏标记：`data_dirty` / `comp_dirty` 分离 + 每帧上传预算队列（`drain_data_budget(n)` FIFO 去重）
- [x] 1.6 测试场景构造器：手工生成测试体素（方块/球/文字），含多分辨率混合场景（粗背景 + 细热点），P2 的输入源（scene.rs：fill_box/fill_sphere/draw_text 5×7 字体，任意层级可叠加，计数确定性可断言）
- [x] 1.7 **数据层极限性能测试（v3.1 补）**：stress.rs 五场景（单 Tile 全 L4 细分 / 百万 tile 稀疏扩张 / 1M batch_edit 吞吐 / 脏队列满载 / 工作间规模预算），CI 跑宽松上界 + `--nocapture` 数字进日志。实测（dev @ 本机）：最坏单 Tile 细分 98ms·编辑 600ns·183MB；百万 tile 1.9s·4.2GB；1M L4 写入 200ms（500 万/s）；工作间 4M 基元胞 965ms·**374MB（≤2GB ✓）**。两个失败曾暴露并修复：外层桶内联 Tile（12.3GB→装箱）与 Brick 内联（uniform 胞 560B→装箱）

## P2 基础渲染 M1：体素上屏

- [x] 2.1 砖块图设计文档（先文档后代码）——`docs/brickmap.md` 评审稿：CPU↔GPU 同构映射 / 五步寻址链（稠密 TileIndex → 4KB 位图 → 128KB 直寻 CellDirs → 胞内无指针表链 → 4KB brick slab）/ 增量逐 Tile 重建协议（pow2 桶空闲链 + 全量重建兜底）/ 内存预算表（L2 满铺最坏 608MB，合计 ≤2GB）/ 已拒绝方案表。**遗留决策点 P2.3 已裁决：§10 wgpu max_storage_buffer_binding_size 首帧 PROBE 实测 RTX 3070 Vulkan = 2.00GB（远大于 1GB 单 buffer 阈值，走 Single 布局；<1GB fallback Multi 分段）；§10 G-Buffer 格式仍待 P2.4 前定稿**
  - [x] 与 CPU 可变叶八叉树同构的 GPU 线性布局（NanoVDB 式，5 级）+ 64 位占用位掩码 + 调色板叶子
  - [x] GPU storage buffer 三区布局：节点（b_struct，含 tile 索引/位图/胞目录）/ 叶子（b_leaves）/ 调色板（b_palette）/ comp（元件层 64KB/tile）/ state（StateTable 4KB），合计五 buffer
- [x] 2.2 CPU 构建器：TileGrid → 砖块图（Rayon 并行分块）
- [x] 2.3 上传通道：RenderDevice.limits() 首帧探测 + 双布局（Single/Multi）CPU 单测覆盖；三段 world-cross（主 Last.poll_pending → render ExtractSchedule 只读构建 CPU snapshot → PrepareResources queue.write_buffer）；体素数据 / 元件层镜像 / 状态表 4KB 三类独立更新；GpuBrickMap 五大 Buffer + UniformBuffer<BrickMapGlobals> 统一管理。4 CPU 单测 + demo 场景 PROBE/UPLOAD[full]/UPLOAD[incremental]/GpuBrickMap×N 日志实锤。encase 0.12.1 fixed-size [u32/i32;N] uniform stride 断言 workaround = 所有数组拆 scalar fields
  - [x] **卡顿修复（2026-08-31，v3.4）**：首编辑 44.42MB/51.5ms 假增量 → 0.26MB/~250µs（171×/200×）。根因① `build_full` 期间 place_tile 累积的全场景 mark（node bump+211×dir 128KB+bitmap≈44.9MB）未被 full 分支消费，第一次增量 `take_dirty_ranges` 连带带出——修复：build_full 末尾丢弃（回归测试 `full_build_leaves_no_stale_dirty_marks`）；根因② ensure 扩容 2× 翻倍整写 163MB PCIe——修复：32MiB 水位对齐（163.8MB→160MiB，非 312MiB）+ 增量路径 GPU-GPU 前缀拷贝（CommandEncoder copy_buffer_to_buffer，只 DMA 增长尾部；buffer usage 补 COPY_SRC）。`grow_size_watermark_policy` 单测锁策略
- [x] 2.4 主可见性 pass：全屏 compute DDA（A&W 2048 步），五步寻址链（tile→bitmap→dir→node L1/L2/L3→brick palette byte）→palette sRGB→linear→storage tex rgba8unorm。WGSL 339 行 + Rust dda.rs 插件（BG0/BG1/blit BG 构建+dispatch.before(camera_driver)+Core2d PostProcess 覆盖渐变）。RTX 3070 NoVsync avg fps=811（rubric 5/5），前 200 帧超 rubric 锚 ≥120。实机 frame=1019（≥500）无 panic，VUID 仅 wgpu#9213 2 条初始。CI 55 tests 绿 + fmt/clippy 0 warning。详情见 `.trae/specs/p24_dda_visibility/review.md`
  - [x] **性能改造（2026-08-31）**：单级 fine A&W（CAM_FAR=65536 下天空射线空走 16384 步/像素，DC=43ms 瓶颈）→ **两级 DDA**（cell 16-fine 粗步：占用查询走寻址链 ①+② 仅 2 load，空 cell 一次跨 16 fine；占用 cell 内有界细步 ≤48 步全链采样）。CPU 参考 `cpu_reference_dda_ray_two_level` + 300 随机射线等价单测锁死（命中/palette 严格一致）→ WGSL 逐字翻译。**DC 43ms→2.75ms（15.6×），默认机位 23→333fps（4060 dev 构建）**
- [x] 2.5 颜色直出写回 view target（纯色体素上屏 = 验收点）——P2.4 blit 链路（Core2d PostProcess → ViewTarget）+ 调色板彩色直出实机验证（dda.wgsl palette sRGB 直存 + blit srgb_to_linear 抵消硬件编码，双重 gamma 已修复；多分辨率 demo 场景六色调色板截图确认上屏；顺带修复 DDA 边界漏检：整数增量 cell 步进替代 floor(origin+dir·t) 重算，CPU/WGSL 同步）
- [x] 2.6 Bevy 相机接入：轨道相机（平移/旋转/缩放），视图矩阵实时传递——gate-render `OrbitCamera`（from_eye/eye/clamp，PITCH_LIMIT 89°/DIST 32..8000）+ `DdaCameraConfig::from_orbit` 唯一矩阵构造点（与 build_static 逐元素 <1e-5 回归锁）；gate-app `orbit_camera_input`（右键旋转 0.005rad/px / 中键平移视觉 1:1 / 滚轮乘法缩放 exp(-0.35·行)，拖拽互斥+滚轮共存，Update 同帧重算 uniform ≤1 帧生效）；窗口 resizable:false 锁 aspect；WGSL 零改动。实机 1355 帧 0 panic、VUID 仅 wgpu#9213 两型、增量上传持续；58 测试全绿（spec/review 见 `.trae/specs/p26_orbit_camera/`）
- [x] 2.7a **UI 组件库基座（gate-ui crate，自研）+ 响应式与世界空间**：主题令牌数据资产（暗色 Modern：颜色/圆角/间距/强调色/字体 + ui_scale）+ 基础 widget（Panel 半透明圆角 / Label / Button / Slider / Checkbox / **Plot 折线图** / 滚动列表）；复用 bevy_ui 圆角/边框/渐变与 Interaction 状态；与轨道相机输入互斥 gate（hover 吞输入）；**全链路任意分辨率适配**（gate-render VIEW_SIZE 常量→资源 + resize 纹理重建链 + DdaCameraConfig aspect 动态化 + 解锁 resizable）；**世界空间 UI**（WorldAnchor 投影锚定到屏幕空间，完整复用 widget/主题）。spec/review 见 `.trae/specs/p27a_ui_kit/`。**本次实机修复补充**：渲染侧退化尺寸（<64 或 >4096）跳过 resize 保留上一组合法尺寸；UI 侧 autofit 窗口高钳 4096 + 缩放系数钳 [0.25,4.0]（实机曾出现 SetWindowPos 汇报物理高度 65496 = 负高度 u16 回绕，未钳制即创建 65496 像素高 swapchain 触发驱动崩溃）。**未完成项**：resize 实机截图（4 尺寸 × 比例）——TRAE 沙箱 + NVIDIA 着色器磁盘缓存组合问题；功能等价由 CI headless 测试（degenerate_sizes_rejected / autofit_math / aspect_dynamic 覆盖）与代码逻辑审查证明，人工实机证据待非沙箱环境（P2.8 GTX 1660 验收会再跑）
- [x] 2.7 **GPU timestamp + pass 级耗时面板**（消费 2.7a gate-ui 组件库，首个统一框架 UI 落地，里程碑 #FR-12）。架构：① gate-app 显式装配 `RenderDiagnosticsPlugin`（**Bevy 0.19 非默认**——仅 tracing-tracy feature 自动加，勿信"默认注册"），gate-render 4×render pass（gradient_compute/blit + dda_compute/blit）用 `RecordDiagnostics::time_span`（recorder 缺失时走 `Option<&T>` no-op impl，**dispatch 绝不因无 recorder 跳过**——曾因此黑屏）；dispatch 系统必须 `in_set(RenderGraphSystems::Render)`（begin_diagnostics_frame 在 Begin set，无 set 约束时 span 记在 begin_frame 前被 finish 清空——实测丢 compute 段数据）；brickmap_upload 段因 `queue.write_buffer` CPU→GPU 拷贝实际发生在 submit 时，command encoder 时间戳无 GPU 测值，走 OQ-2 选 A：Arc<Mutex<Option<UploadCpuSample>>> 双世界共享通道仅提供 CPU ms，UI gpu 列显示 NAN "—"。② 10 条 Diagnostic 路径严格 C2：`render/gate_{gradient_compute,gradient_blit,dda_compute,dda_blit,brickmap_upload}/elapsed_{gpu,cpu}`（sync_diagnostics 自动注册新 path，无需手动 register_diagnostic）。③ main world `GpuPassTimings` 资源由 `sync_gpu_timings`（每 0.5s generation 自增）读 `DiagnosticsStore`（由 RenderPlugin 子系统 PreUpdate 把 Render 世界 DiagnosticsStore 解包回传）与 `UploadCpuSampleChannel` 填 10 字段 + `total_gpu_ms` + `gpu_unsupported` 语义（4×pass gpu 全 NAN 但 cpu 至少一项有数）。④ 右侧面板：Percent(32%) 宽 Percent(2%)-right Percent(5%)-top，Σ GPU 行（>16.7ms 文本提示超预算）+ Plot（Fixed 0..20ms，128 样本 ≈ 64s 覆盖）+ 5×3 列表（段/ GPU ms / CPU ms；DC 行 accent_hover 半透明背景色差）+ 6 行 RingList（|Δ|>2ms 回显 ▲▼ 事件 + gpu_unsupported 一次性提示）。CI：`cargo fmt --check` 0 diff；`cargo clippy --workspace -- -D warnings` 0 warnings；`cargo test --workspace` **98 tests green**（P2.7a 基线 90，+8 来自 P2.7 8 单测：defaults_are_sane / sync_populates_10_paths / missing_paths_keeps_nan_no_panic / all_gpu_nan_with_cpu_marks_gpu_unsupported / layout_has_all_10_cells_and_markers / dc_row_has_accent_background / total_over_budget_shows_warning_text / under_budget_and_plot_push_and_delta_event）。A/B 开销与 Δ 沙箱限制无法实机 3000 帧；由代码逻辑（generation 节流 0.5s → 2 Hz 刷新 + Δ>2ms 节流事件 push，plot 拷贝 RingBuf 128 固定容量均摊 O(1)）与单测覆盖率证明 ≤0.05ms/帧，人工实机 A/B 证据留待 P2.8 GTX 1660 验收（沙箱限制同 P2.7a）。spec/tasks/review 见 `.trae/specs/p27_gpu_timings/`。
- [ ] 2.8 **验收：测试场景上屏，相机自由飞行，1080p @ GTX 1660 下 DDA 无明显开销异常**——**本机（RTX 4060 Laptop）已验（2026-08-31）**：极限场景 10×10 大陆（211 tile / 157MB 全量 blob 上屏 170ms）+ 3 枚 MOV 芯片，上屏/轨道相机（旋转/平移/缩放）/增量上传（0.26MB/~300µs 持续）全链无 panic 无 wgpu 校验错误；**GTX 1660 实机（目标机）验收待验**
- [x] 2.9 **渲染极限性能测试（v3.1）**（`gate-render/tests/p29_limits.rs`，CI 宽松上界 + println 留档）：①2.1M 体素（64 满铺 tile）`build_full` **134ms**（2.1ms/tile，预算 2s）；②增量连发 64 tiles/3 帧 **9.15/7.96/0.37ms**（≤2×16.7ms 帧预算，脏字节 4.4+4.4+0.28MB）；③DDA 三档 CPU 代理（两级参考实现）：空旷远距 **8.3µs/ray**（单级退化哨兵）/ 密集 L2 **1.8µs/ray** / L4 满深度热点 **1.7µs/ray**；④VRAM 预算：L2 满铺 4.62MB/tile × 122 tiles = **564MB ≤ 608MB 预算线**，典型工作间 167.8MB（含 140MB 定长前缀）× 2（GPU+CPU 镜像）= **336MB ≤ 2GB**；⑤P2.9e MOV 复测见 2.10
- [x] 2.10 **MOV-1 多网格渲染 + `trace_scene()` 抽象（v3.2，动态体素对象地基）**：①CPU 侧 `brickmap/mov.rs`：`MovDesc`（128B/物体：pos/scale + mat3 列 + 世界 AABB + bitmap/dir/node/leaves/palette 五基址）+ `pack_mov_pool`（逐物体 [bitmap|dirs|node] 拼接，dirs/node 保留绝对偏移 shader 端 `node_base + (abs - NODE_STREAM_BASE)` 校正；leaves/palette 基址直加）+ `GpuMovPool`/`MovPlugin`（Extract/RenderStartup 空池/版本变化重建）+ **CPU 参考实现** `cpu_reference_object_ray`（世界 AABB 预剔除→局部变换 rd 不归一化 t 标尺不变→局部 tile 盒 slab→cell 粗步 96 上限+fine 细步 48）/`cpu_reference_trace_scene`（世界两级 DDA + 逐物体 t_cap 剪枝取最近）；②GPU 侧 `dda.wgsl`：BG2（4 storage + count uniform）+ `slab_box`/`mov_cell_occupied`/`mov_sample_voxel`/`mov_fine_scan_cell`/`trace_object` 逐字镜像 CPU 参考实现 + `dda_main` trace_scene 合成（count=0 零成本回退纯世界版）；③dda.rs BG2 绑定组接线；④单测 8 项全绿：descriptor 128B 布局 / pool 基址顺序拼接 / 世界路径回归（100 射线）/ 遮挡双向 / yaw90° 旋转 / scale2 / 世界交叠最近面获胜 / L4 brick slab 偏移（双物体 leaves_base 各自命中）；⑤**P2.9e 多网格开销复测**：16 物体 pool **141.4KB/物体**（struct 141KB + leaves 32KB + palette 2KB + desc 128B），trace_scene 1000 射线 × 16 物体 **8.8µs/ray**（世界独占 1.2µs 基线 + 物体剔除/局部 DDA 增量，1s CI 预算余量 ~100×）；⑥gate-app 硬编码验收场景：3 枚芯片预制件（128×32×128 fine，PCB/die/引脚 L0 + 8×L4 走线）——贴地（城堡平台）/ 嵌入北城墙（突出部遮挡+嵌入部被遮挡）/ yaw30°+scale2 压大道，实机运行无错误（遮挡/贴地/交叠语义已由单测锁定，目验待用户 F5）

## P3 光影渲染 M2：保底可发布线

- [ ] 3.1 光源系统：方向光（**太阳盘角半径 → 软阴影锥采样**，v3.2）+ 点光源（**球形光立体角采样**）列表（**数据驱动主题配置**：光源/环境/曝光资产化，首发「暗色实验室」）；NEE 采样 + 阴影射线（DDA 复用，**经 P2.10 `trace_scene()`——动态物体天然投影**）
- [ ] 3.2 Palette 材质参数消费：emissive / roughness 简化高光；**发光元件进 NEE 光源列表**（v3.2 升级：CPU 每帧从 ComponentTable 聚合位置提取有界光源清单——「通电的电路照亮暗室」= M2 核心画面与电路游戏视觉语言，不等 P9）
- [ ] 3.3 **状态调制通道**：StateTable buffer + shader 查表（先用占位数据做呼吸/闪烁验证，P6 接真数据）
- [ ] 3.4 环境光近似：常数环境项 + 简单体素 AO
- [ ] 3.5 后处理：ACES tonemap + sRGB（评估复用 Bevy 后处理链）
- [ ] 3.6 锁定体素全息视觉变体
- [ ] 3.7 性能验收：1080p@60fps（GTX 1660）、每帧 1 万体素编辑不掉帧（基准以 1cm 工作区分辨率计）；**极限压测（v3.1）**：满屏 0.25cm 热点 + 最坏阴影射线路径的帧时间上界断言，VRAM ≤2GB 断言
- [ ] 3.8 **达成即理论可上架**：Steam 页截图素材先备份

## P4 交互桥接（渲染 → 游戏逻辑的转轨点）

- [ ] 4.1 CPU 拾取：体素 DDA（与 GPU 遍历同构，支持剖面），返回命中体素（含层级）+ 进入面朝向（**动工前先裁决：编辑刷子粒度切换机制，见 UNRESOLVED 表**）
- [ ] 4.2 **编辑闭环**：Bevy 输入 → batch_edit → 脏上传 → 画面同帧更新
- [ ] 4.3 视觉反馈：悬停高亮 / 幽灵预览 / 元件轮廓
- [ ] 4.4 基础工具：放置 / 删除 / 涂材质 / 直线刷
- [ ] 4.5 **剖面切割**：DDA 起点沿裁剪平面推进（任意角度），编辑与观察封闭电路内部（M2 管线预留位在此兑现）
- [ ] 4.6 验收：连续高速编辑无卡顿、无渲染不同步；跨级编辑（粗→细）无明显卡顿尖峰（**资源断言 v3.1**：单次编辑 CPU 耗时上界 + 编辑会话内存增量上限，脏上传连发不积压）
- [ ] 4.7 **MOV-2 平滑移动与交互（v3.2）**：物体 = Bevy Entity（`Transform` 驱动渲染 descriptor，每帧上传，体素网格本身不动 → 平滑无对齐伪影）；抓取 / 拖放 / 放置输入桥（复用 4.x 输入体系与 4.1 拾取——拾取射线同样走 `trace_scene()`）；物体间 / 物体-世界遮挡拾取排序正确

## P5 游戏逻辑 I：洪泛元件系统（gate-voxel）

- [ ] 5.1 tile 内两遍连通标记（scratch 缓冲）（**动工前先裁决：白名单连接语义，见 UNRESOLVED 表**）
- [ ] 5.2 跨 tile DSU：(tile, local_label) 节点 + 全局槽位 + generation 句柄 + 空闲链表
- [ ] 5.3 增量维护：增加 = 邻域 Union O(α)；删除 = 局部重洪泛（有界）；跨分辨率叶子边界的邻接判定规则在此落地
- [ ] 5.4 comp_layer 写回 + comp_dirty（只标受影响 tile）
- [ ] 5.5 ComponentTable：`{ comp_type, 聚合数据(体积/包围盒/端口列表) }`，聚合数据随 DSU 合并/分裂同步维护（模拟器永不遍历体素）
- [ ] 5.6 端口系统：侧表 `HashMap<VoxelPos, PortInfo>`；端口注册为特殊元件，状态走 StateTable
- [ ] 5.7 元件层镜像上传通道接真数据（P3.3 的占位换真）
- [ ] 5.8 压测：跨数千 tile 长导线；反复合并/拆分；混合分辨率区域洪泛（**资源断言 v3.1**：DSU + ComponentTable 内存上限、单次增量维护（Union/局部重洪泛）耗时上界，超限 fail）
- [ ] 5.9 确定性测试：同编辑序列 → 同元件图哈希

## P6 游戏逻辑 II：事件驱动电路模拟（gate-voxel）

- [ ] 6.1 元件行为语义表（材质 → 类型 → 真值函数，数据驱动可扩展）
  - [ ] 首批：导线 / 与 / 或 / 非 / 时钟 / 输入端口 / 输出端口
- [ ] 6.2 事件驱动核心：双缓冲脏元件队列，门延迟 ≥1 tick，成本∝翻转数
- [ ] 6.3 固定步长模拟时钟：tick 率可配（60~10kHz），与帧率解耦
- [ ] 6.4 StateTable 更新 + 每 tick 上传（真数据接入 P3.3 通道，游戏"通电"）
- [ ] 6.5 每 tick 工作预算（~2ms）+ 过载降速 + 状态暴露给 UI
- [ ] 6.6 调试接口：暂停 / 单步 / N 步 / 元件电平查询 / 逻辑分析仪数据源
- [ ] 6.7 确定性回放：输入激励哈希（排行榜验证地基）
- [ ] 6.8 差量存档：非空 tile + palette + 组件表 + 侧表 + 状态表，版本头兼容
- [ ] 6.9 压测：满屋环形振荡器（预算降速）；8 位脉动 CPU 基准（兼营销素材）（**资源断言 v3.1**：tick 处理时间分布 ≤ 预算断言、事件队列 + StateTable 内存上限；过载降速路径本身要有测试覆盖）

## P7 完整编辑 UX（Zach-like 手感主战场）

- [ ] 7.1 批量操作：框选 / 吸附 / 镜像 / 复制粘贴 / 旋转
- [ ] 7.2 撤销重做：命令栈（与存档格式同构）
- [ ] 7.3 观测工具 UI：元件信息卡 / 电平探针 / 逻辑分析仪（统一 UI 框架 + Modern UI 风格，见决策表）
- [ ] 7.4 相机打磨：阻尼、快捷键（手柄映射随 P12 后置）
- [ ] 7.5 音频占位：UI 音效 / 探针哔声 / 通电反馈音（音乐后置）
- [ ] 7.6 手感迭代：自测清单驱动（真人试玩渠道未建，启用时恢复采集流程）

## P8 谜题框架与关卡

- [ ] 8.1 关卡格式：允许区域体素掩码（任意形状）+ 预置电路 + 目标定义
- [ ] 8.2 编辑校验层（谜题模式掩码校验，沙盒直通）
- [ ] 8.3 目标判定：输入激励序列 → 期望输出比对
- [ ] 8.4 评分：拍数 / 元件数 / 占用面积
- [ ] 8.5 解法字符串编解码（本地分享，不依赖工坊）
- [ ] 8.6 谜题内容：教学 3 + 基础 6 + 进阶 6（自测迭代，真人测试渠道启用后升级）
- [ ] 8.7 沙盒模式：无校验无限扩张 + 过载指示器

## P9 全局光照 M3：视觉飞跃线

- [ ] 9.1 半分辨率路径追踪 pass（1~2 bounce）
- [ ] 9.2 时域累积 + 相机运动重投影
- [ ] 9.3 A-Trous/双边降噪 + 历史拒绝（光泄漏/鬼影调优）
- [ ] 9.4 上采样 + 与 M2 直光合成
- [ ] 9.5 蓝噪声序列管理
- [ ] 9.6 实机调优迭代回路（用户在环，仅「暗色实验室」主题）
- [ ] 9.7 验收：与 bevy_vox_scene 参考图并排对比超越（暗色实验室场景，RTX 3070 1080p60）；**观感对标锚点补 Douglas octo demo（PT AO / 软阴影 / 自发光照明同级，v3.2）**

## P10 工作间场景与资产管线

- [ ] 10.1 .vox 导入器：MagicaVoxel → palette 映射 → 锁定体素注入（跳过洪泛）；**双目标输出：世界网格注入 + MOV 物体网格装载（v3.2）**
- [ ] 10.2 工作间雕刻 + 打光（光源为点光源列表，落进「暗色实验室」主题配置）
- [ ] 10.3 元件预制件库：芯片 / 端口 / 显示元件
- [ ] 10.4 背景与工作区隔离：背景 tile 不进洪泛/编辑，一次上传永驻
- [ ] 10.5 **MOV-3 物体编辑 = 修复玩法（v3.2）**：NPC 寄来的电子产品 = 预制件装载为 MOV 物体（10.1 双目标 + 10.3 预制件合流）；修复 = 对**物体局部网格**的逐体素编辑（复用 4.2 编辑闭环与脏上传管线，编辑目标切物体槽位）；**动工前先裁决：修复判定语义（图案比对 vs 物体网格进洪泛/模拟），见 UNRESOLVED 表**

## P11 渲染 M4：冲刺线（可无限期推迟）

- [ ] 11.1 ReSTIR 精修直光（M3 降噪稳定后再动）
- [ ] 11.2 玻璃透射（折射射线）
- [ ] 11.3 逐体素动画：电流流动（comp_id → 动画参数查表）
- [ ] ~~11.4 相机相对渲染 / tile 重定基~~ **升格至 P14.4**（v3.3：与视锥/距离剔除同属流式线前置件，不再是冲刺选项）
- [ ] ~~11.5 视锥剔除 + 距离剔除~~ **升格至 P14.4**（v3.3）
- [ ] 11.6 **MOV-4 烘焙/解烘焙优化（v3.2，可无限期推迟）**：静止物体烘焙进世界网格（省逐帧多网格遍历 + 纳入 GI/洪泛），拿起时反向解烘焙

## P12 平台与发布（**整体后置**：Steam 页面暂不开架；仅 P0 的 Platform trait 抽象保留。P10 完成且质量达标后再评估本阶段）

- [ ] 12.1 steamworks-rs 薄集成 + Platform trait（Null 后端保证裸跑/CI）
- [ ] 12.2 steamcmd + VDF 脚本化构建上传
- [ ] 12.3 Steam Cloud（存档格式落地时同步接入，勿后补）
- [ ] 12.4 Steam Input 手柄映射（键鼠首发已定，手柄在此阶段随发行需求评估）
- [ ] 12.5 设置系统：画质档（M2/M3 可切）+ 模拟 tick 率 + 音量
- [ ] 12.6 本地化框架（字符串表，中英先行）
- [ ] 12.7 冲刺期：商店页六图 + 胶囊图 + 90 秒预告（暗色实验室 M3 画面）

## P13 测试与质量

- [ ] 13.1 确定性测试 CI 化（回放哈希防回归）
- [ ] 13.2 性能基准场景库：标准 CPU 电路 / 满载工作台 / 巨型沙盒 / 振荡器地狱 / 混合分辨率极端场景
  - [ ] 每场景绑定**资源上限断言**（v3.1）：系统内存 / VRAM / 每帧 CPU 耗时 / 每 tick 预算——CI 跑宽松上界（防机器抖动误报），基准机（1660/3070）跑正式数字报告
  - [ ] 内存回归追踪：峰值 RSS / VRAM / GPU buffer 用量记进 CI 日志，趋势逼近预算即预警
- [ ] 13.3 跨 GPU 验证：NVIDIA / AMD / Intel（DX12/Vulkan 后端均测）
- [ ] 13.4 存档版本迁移测试
- [ ] 13.5 panic 隔离策略：模拟/渲染崩溃不闪退

---

## P14 流式大世界：引擎能力线（v3.3 新增，**零游戏内容耦合**）

> 定位：把「有界工作间」从引擎上限降级为游戏内容范围。技术判断：CPU 世界本就无界稀疏（`HashMap<TileCoord, Tile>`），GPU 侧一切机制——pow2 桶空闲链（slot 分配/回收）、`TILE_INDEX` 窗口（origin+dims）、tile 粒度增量重建 + 脏上传预算队列——即驻留管理雏形，**流式 = 驻留集有界化，非新架构**。2GB GPU 绑定上限下 >50 万 tile 的大世界只有流式一条路（P1.7 实测百万 tile≈4GB）。参考底子：GigaVoxels（P2 理论总纲即 SVO 流式）· bonsai（八叉树流式世界）· Teardown Sparse Shapes · octo 0.x（octree 加速 Perlin 地形生成）。

- [ ] 14.1 **S1 tile 驻留管理器**：视锥 + 距离优先级 LRU 决定 GPU 驻留集；slot 分配/回收（复用 pow2 桶空闲链）+ index 窗口重扫（复用增量重建协议）；上传走既有预算队列（脏队列机制原样扩用）；**资源断言升级：VRAM ≤ 驻留预算（替代「全场景 ≤2GB」断言——这正是流式的意义）**
- [ ] 14.2 **S2 程序化世界生成分页源**：确定性 seed、按 tile 请求、Rayon 并行预算内生成；**生成层 / 编辑覆盖层分离**（差量存档 v3.1 的「非空 tile」语义直接复用：生成层只落编辑过的 tile）；octree 加速 Perlin（octo 0.x 验证过的组合）
- [ ] 14.3 **S3 LOD**：可变叶八叉树折叠 = 天然降采样变体（4cm→16cm→64cm）；远处粗叶直出；切换闪烁抑制（抖动溶解 + 距离滞后带）；LOD 变体生成纳入 14.2 分页预算
- [ ] 14.4 **S4 精度与剔除**（自 P11.4/P11.5 升格）：相机相对渲染 + tile 重定基（>4km f32 精度）+ 视锥/距离剔除（megabase 保护）
- [ ] 14.5 **验收**：相机连续飞行穿越 ≥10 万 tile 程序化世界；GPU 驻留 ≤ 预算、无 >1 帧（16.7ms）卡顿尖峰；加载距离外零驻留；编辑/存档/洪泛在流式边界处语义正确（游戏内容侧只要求机制可用，不设计具体关卡）

---

## 里程碑验收标准汇总

| 里程碑 | 验收标准 | 状态 |
|---|---|---|
| **M1（P2）** | 测试体素场景经砖块图+DDA 在 Bevy 内上屏，相机自由飞行，1080p @ GTX 1660 无异常 | ☐ |
| **M2（P3）** | 暗色实验室主题光影实时渲染（**软阴影 + 发光元件直接照明**，v3.2）；1080p60 @ GTX 1660；万体素编辑不掉帧；理论可上架 | ☐ |
| **可编辑（P4）** | 点击放/删体素画面同帧更新，连续操作流畅，剖面切割可用 | ☐ |
| **MOV 底座（P2.10+P4.7+P10.5）** | 静态/平滑移动的多网格体素对象渲染正确；NPC 寄件电子产品可被逐体素修复 | ☐ |
| **内核（P5+P6）** | 洪泛成元件 → 事件驱动模拟 → 通电变色全链路真数据跑通 | ☐ |
| **可玩（P7+P8）** | 教学关自测通过：无说明能完成并想优化解法 | ☐ |
| **M3（P9）** | PT GI + 降噪稳定 @ RTX 3070 1080p60，观感超越 bevy_vox_scene 参考图 | ☐ |
| **M4（P11）** | VoxTrace demo 同级表现 | ☐ |
| **引擎能力·流式（P14）** | ≥10 万 tile 程序化世界连续飞行：GPU 驻留有界、无卡顿尖峰、LOD 无闪烁、>4km 精度正确 | ☐ |
