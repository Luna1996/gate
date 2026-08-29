# gate 项目施工总纲（Voxel 电路解谜游戏）v3

> 单兵开发 · Bevy 引擎 + 自研光追微体素管线 · 多分辨率可变叶体素（4cm/1cm/0.25cm）· 无 Mesh 化
> v2 变更：引擎定 Bevy；顺序改为**渲染先行，游戏逻辑后置**
> v3 变更：grill-me 拷问裁决落地——数据结构定**单一可变叶八叉树**；画质验收写死硬件基线；光照主题数据驱动（首发暗色实验室）；P12 整体后置；编辑成本表述修正
> 最后更新：2026-08-29

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
| 场景 | 有界工作间（静态锁定体素），不做流式大世界 | 护栏已埋，将来可扩展 |
| **画质基线（v3）** | **M2 @ GTX 1660 1080p60；M3 @ RTX 3070 1080p60** | 验收写死硬件+分辨率+帧率，杜绝玄学；1660 上 M3 提供降档（关 GI） |
| **内部观察（v3）** | **仅剖面切割**（DDA 裁剪平面，近零成本），不做 X-ray 半透明 | 编辑/调试封闭电路刚需；管线早期预留 |
| **光照主题（v3）** | 数据驱动配置（光源/环境/曝光资产化），首发仅深调**暗色实验室** | M3 调优矩阵不随主题数翻倍；其余主题后置扩展 |
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

**工程既定**：编辑器/工具窗口 UI 用 egui，游戏 HUD 用 bevy_ui；光照主题为纯数据资产（不含硬编码）。

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
- [x] 1.3 Palette：256 × `PaletteEntry { color, roughness, emissive, transmission, flags }`（u8 索引 + flags 视觉变体；槽 0 = AIR 保留，set_voxel debug_assert 禁写）
- [x] 1.4 TileGrid：HashMap + `get/set_voxel`（跨级放置 → brick 分配 + 祖先掩码更新）+ `batch_edit`，编辑返回受影响区域集合（`DirtyEdit { tile, level }`，P2.3 按此增量上传）
- [x] 1.5 脏标记：`data_dirty` / `comp_dirty` 分离 + 每帧上传预算队列（`drain_data_budget(n)` FIFO 去重）
- [x] 1.6 测试场景构造器：手工生成测试体素（方块/球/文字），含多分辨率混合场景（粗背景 + 细热点），P2 的输入源（scene.rs：fill_box/fill_sphere/draw_text 5×7 字体，任意层级可叠加，计数确定性可断言）

## P2 基础渲染 M1：体素上屏

- [ ] 2.1 砖块图设计文档（先文档后代码）
  - [ ] 与 CPU 可变叶八叉树同构的 GPU 线性布局（NanoVDB 式，5 级）+ 64 位占用位掩码 + 调色板叶子
  - [ ] GPU storage buffer 三区布局：节点 / 叶子 / 调色板
- [ ] 2.2 CPU 构建器：TileGrid → 砖块图（Rayon 并行分块）
- [ ] 2.3 上传通道：体素数据 / 元件层镜像 / 调色板三类独立更新
- [ ] 2.4 主可见性 pass：全屏 compute，逐像素 DDA → G-Buffer（pos/normal/palette_idx）
- [ ] 2.5 颜色直出写回 view target（纯色体素上屏 = 验收点）
- [ ] 2.6 Bevy 相机接入：轨道相机（平移/旋转/缩放），视图矩阵实时传递
- [ ] 2.7 GPU timestamp + pass 级耗时面板
- [ ] 2.8 **验收：测试场景上屏，相机自由飞行，1080p @ GTX 1660 下 DDA 无明显开销异常**

## P3 光影渲染 M2：保底可发布线

- [ ] 3.1 光源系统：方向光 + 点光源列表（**数据驱动主题配置**：光源/环境/曝光资产化，首发「暗色实验室」）；NEE 采样 + 阴影射线（DDA 复用）
- [ ] 3.2 Palette 材质参数消费：emissive / roughness 简化高光
- [ ] 3.3 **状态调制通道**：StateTable buffer + shader 查表（先用占位数据做呼吸/闪烁验证，P6 接真数据）
- [ ] 3.4 环境光近似：常数环境项 + 简单体素 AO
- [ ] 3.5 后处理：ACES tonemap + sRGB（评估复用 Bevy 后处理链）
- [ ] 3.6 锁定体素全息视觉变体
- [ ] 3.7 性能验收：1080p@60fps（GTX 1660）、每帧 1 万体素编辑不掉帧（基准以 1cm 工作区分辨率计）
- [ ] 3.8 **达成即理论可上架**：Steam 页截图素材先备份

## P4 交互桥接（渲染 → 游戏逻辑的转轨点）

- [ ] 4.1 CPU 拾取：体素 DDA（与 GPU 遍历同构，支持剖面），返回命中体素（含层级）+ 进入面朝向（**动工前先裁决：编辑刷子粒度切换机制，见 UNRESOLVED 表**）
- [ ] 4.2 **编辑闭环**：Bevy 输入 → batch_edit → 脏上传 → 画面同帧更新
- [ ] 4.3 视觉反馈：悬停高亮 / 幽灵预览 / 元件轮廓
- [ ] 4.4 基础工具：放置 / 删除 / 涂材质 / 直线刷
- [ ] 4.5 **剖面切割**：DDA 起点沿裁剪平面推进（任意角度），编辑与观察封闭电路内部（M2 管线预留位在此兑现）
- [ ] 4.6 验收：连续高速编辑无卡顿、无渲染不同步；跨级编辑（粗→细）无明显卡顿尖峰

## P5 游戏逻辑 I：洪泛元件系统（gate-voxel）

- [ ] 5.1 tile 内两遍连通标记（scratch 缓冲）（**动工前先裁决：白名单连接语义，见 UNRESOLVED 表**）
- [ ] 5.2 跨 tile DSU：(tile, local_label) 节点 + 全局槽位 + generation 句柄 + 空闲链表
- [ ] 5.3 增量维护：增加 = 邻域 Union O(α)；删除 = 局部重洪泛（有界）；跨分辨率叶子边界的邻接判定规则在此落地
- [ ] 5.4 comp_layer 写回 + comp_dirty（只标受影响 tile）
- [ ] 5.5 ComponentTable：`{ comp_type, 聚合数据(体积/包围盒/端口列表) }`，聚合数据随 DSU 合并/分裂同步维护（模拟器永不遍历体素）
- [ ] 5.6 端口系统：侧表 `HashMap<VoxelPos, PortInfo>`；端口注册为特殊元件，状态走 StateTable
- [ ] 5.7 元件层镜像上传通道接真数据（P3.3 的占位换真）
- [ ] 5.8 压测：跨数千 tile 长导线；反复合并/拆分；混合分辨率区域洪泛
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
- [ ] 6.9 压测：满屋环形振荡器（预算降速）；8 位脉动 CPU 基准（兼营销素材）

## P7 完整编辑 UX（Zach-like 手感主战场）

- [ ] 7.1 批量操作：框选 / 吸附 / 镜像 / 复制粘贴 / 旋转
- [ ] 7.2 撤销重做：命令栈（与存档格式同构）
- [ ] 7.3 观测工具 UI：元件信息卡 / 电平探针 / 逻辑分析仪
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
- [ ] 9.7 验收：与 bevy_vox_scene 参考图并排对比超越（暗色实验室场景，RTX 3070 1080p60）

## P10 工作间场景与资产管线

- [ ] 10.1 .vox 导入器：MagicaVoxel → palette 映射 → 锁定体素注入（跳过洪泛）
- [ ] 10.2 工作间雕刻 + 打光（光源为点光源列表，落进「暗色实验室」主题配置）
- [ ] 10.3 元件预制件库：芯片 / 端口 / 显示元件
- [ ] 10.4 背景与工作区隔离：背景 tile 不进洪泛/编辑，一次上传永驻

## P11 渲染 M4：冲刺线（可无限期推迟）

- [ ] 11.1 ReSTIR 精修直光（M3 降噪稳定后再动）
- [ ] 11.2 玻璃透射（折射射线）
- [ ] 11.3 逐体素动画：电流流动（comp_id → 动画参数查表）
- [ ] 11.4 相机相对渲染 / tile 重定基（沙盒 >4km 精度）
- [ ] 11.5 视锥剔除 + 距离剔除（megabase 保护）

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
- [ ] 13.3 跨 GPU 验证：NVIDIA / AMD / Intel（DX12/Vulkan 后端均测）
- [ ] 13.4 存档版本迁移测试
- [ ] 13.5 panic 隔离策略：模拟/渲染崩溃不闪退

---

## 里程碑验收标准汇总

| 里程碑 | 验收标准 | 状态 |
|---|---|---|
| **M1（P2）** | 测试体素场景经砖块图+DDA 在 Bevy 内上屏，相机自由飞行，1080p @ GTX 1660 无异常 | ☐ |
| **M2（P3）** | 暗色实验室主题光影实时渲染；1080p60 @ GTX 1660；万体素编辑不掉帧；理论可上架 | ☐ |
| **可编辑（P4）** | 点击放/删体素画面同帧更新，连续操作流畅，剖面切割可用 | ☐ |
| **内核（P5+P6）** | 洪泛成元件 → 事件驱动模拟 → 通电变色全链路真数据跑通 | ☐ |
| **可玩（P7+P8）** | 教学关自测通过：无说明能完成并想优化解法 | ☐ |
| **M3（P9）** | PT GI + 降噪稳定 @ RTX 3070 1080p60，观感超越 bevy_vox_scene 参考图 | ☐ |
| **M4（P11）** | VoxTrace demo 同级表现 | ☐ |
