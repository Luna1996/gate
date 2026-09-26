# Editable GigaVoxels：Ray-Guided 大世界 + 高速编辑

状态：**M0 + M1 + M3（主体）已落地并跑通**（见 §9、§10），M2 起未实现；另有一个 M4/M5/M6 的最小可用
试验世界 `infinite_cubes`（见 §10）。目标是把 GigaVoxels（Crassin & Neyret 2009）的 **ray-guided 流式**
引入本仓，同时**不牺牲**已有的节点级增量编辑。

- 参考：`docs/douglas/` 的 devlog 索引（#16 渲染距离、#33 GPU 显存分配器 + 无限世界 chunk 流式加载）。
- 相关既有实现：`gate-voxel/src/chunk_tree.rs`（Brick Tree）、`gate-render/src/brickmap/`（wire / builder / upload）、
  `assets/shaders/voxel_raytrace/`（trace / world / main / gi）。

## 0. 目标与非目标

**目标**

1. **大世界（Ray-Guided 流式）**：常驻集有界；细节按射线真实需求分配；视距从当前"全量常驻"推到公里级。
2. **高速编辑不退化**：resident 全分辨率内的编辑仍是节点级（实测 356 B / 约 20 µs），
   且该代价与世界大小、视距、LOD 档位**解耦**。

**非目标（明确不抄）**

- 预滤波金字塔 + cone tracing 抗锯齿：本仓走硬表面方块世界，抗锯齿交给 TAA + 材质解析 mip（`common.wesl` 的 `pbr_mip_lod`）。
- brick 用定长 3D 纹理 + 三线性插值：保留精确体素查询与 16 位调色板索引。
- 纯 CPU 侧生产、GPU 只当缓存：保持 **CPU 树为权威** + 节点级增量上传（这正是编辑不牺牲的来源）。

## 1. 架构前提（两个目标为何不冲突）

> **CPU 树 = 权威（编辑改它、推节点级增量）；GPU = 派生缓存（流式只决定"哪些 chunk、以什么分辨率常驻"）。**

推论：

- 编辑延迟与服务解耦：resident 内的编辑仍是 `TreeDirty` → 节点级重写，实测 356 B / 约 20 µs；
- 两者的**唯一耦合点**是四条规则：
  1. **提升先于写入**：精确编辑要求该 chunk 处于逐体素档，先提升再落笔；
  2. **钉住**：正在编辑 / 刚编辑过的 chunk 不降级、不换出；
  3. **换出前落盘**：有本地编辑的 chunk 先写覆盖层，重载 = 生成基线 + 回放覆盖层；
  4. **驻留切换不等于世界几何变化**：proxy↔full 只失效脏盒范围的 GI 历史，不做 `world_rev++`。

## 2. 现状盘点：GigaVoxels 组件对照

| GigaVoxels 组件 | 本仓现状 | 计划 |
|---|---|---|
| N³-tree + brick（定长体素块） | 已有：4³ 值块（wire 定长 35 字）+ 上层拓扑节点（掩码 + tile 色 + 紧凑 child 表） | 不动 |
| 节点 mipmap 值（node value） | 缺：只有"mask-0 子块的色"，混合节点没有代表色 | M2 |
| node pool / brick pool（按层定长池） | 已有等价物：每 chunk 树块 + 全局空闲段（块级分配 / 释放 / 压实） | 只需 proxy 短树能走现有安装路径 |
| GPU 缓存 + LRU（时间相干） | 缺：全量常驻 | M3 |
| ray-guided 请求（渲染 pass 发射请求） | 缺 | M4 |
| producer（按层生产、粗到细流式） | 缺（现只有 `mount_chunk_tree` 整块挂载） | M5 |
| brick marching + cone tracing | 不适用（走层次 mask-DDA + TAA + 材质 mip） | 不做 |

## 3. 量化依据

### 3.1 实测锚点（castle.vox，59 chunk）

| 项 | 数值 |
|---|---|
| 全量序列化 59 chunk | 合计约 56 ms（均值约 0.95 ms / 最大约 2.0 ms） |
| 树规模 | 节点合计 77 万；CPU 树与 GPU `b_struct` 均约 111 MB（每 chunk 约 1.9 MB） |
| 节点级增量上传 | 单格编辑 356 B / 约 20 µs；`size=17` 一笔 39 KB / 约 38 µs |
| 笔触（每笔） | `size=17` 约 0.15 ms、`size=33` 约 0.62 ms、`size=61` 约 3.0 ms、`size=121` 约 15.5 ms |
| chunk 窗口 | 64³ 个 chunk 条目 = 16384³ 体素（`CHUNK_INDEX_CAP = 64`） |
| 上传预算 | `UPLOAD_BYTES_PER_FRAME = 4 MB/帧` |
| 布局约束 | `layout_order` = 物体在前、主世界在后；前置 volume 变长会移动主世界 `tree_base` 并触发全量重传（`bases_shifted`） |
| **proxy 常驻代价**（castle 59 chunk，实测） | 全树 **22.58 M 字 = 90.3 MB**（1.53 MB/chunk）；4³ 塌缩 2.84 M 字（12.6%）；**16³ 120 K 字（0.53% = 8.1 KB/chunk）**；64³ 7.7 K 字（0.034%）；整 chunk 4.0 K 字（0.018%）。⇒ 远场按 16³ 常驻比全分辨率省 **约 190 倍** |

### 3.2 估算项（未实测，量级参考）

- 地形类 chunk 树约 0.6 MB；4³ 叶改为"掩码 + popcount 打包值表"后约 0.15 MB。
- proxy 逐层截断（**估算，已被 §3.1 末尾的实测取代**）：16³ 约 70 KB/chunk、64³ 约 1 KB/chunk、整 chunk 单色约 0.3 KB/chunk。
- 每射线平均占用段 20–60 步；每步有效带宽约 16 B。
- 生长中的世界装载吞吐：步行 10 m/s 约 6.5 chunk/s（约 12 MB/s）；载具 100 m/s 约 65 chunk/s（约 124 MB/s）。

### 3.3 旁轴精度（px_ang）与 LOD 判据

```
px_ang = 2·tan(FOV_Y/2) / render_h        # gate-render/src/brickmap/dda.rs（1080p/60° ≈ 1.069e-3 rad/px）
fp     = t · px_ang                       # 体素/像素：一个像素在该距离覆盖多少体素
```

**停止下钻的条件按"被丢弃的细节尺度"（= 子块边长）给**，而不是节点自身边长：

| 停在哪一档 | 节点边长 | 丢弃尺度 | 停的条件 | 1080p 安全距离（**本仓 2 cm/体素**） | 每 chunk 内存 |
|---|---|---|---|---|---|
| 逐体素（现状） | 1 | — | `fp < 1` | 小于 935 体素 = **18.7 m** | 1.9 MB（实测） |
| 4³ 块单色 | 4 | 1 | `fp ≥ 1` | 935 体素 = **18.7 m** | 约 0.5 MB（估） |
| 16³ 块单色 | 16 | 4 | `fp ≥ 4` | 3 740 体素 = **74.8 m** | 约 70 KB（估） |
| 64³ 块单色 | 64 | 16 | `fp ≥ 16` | 14 960 体素 = **299 m** | 约 1 KB（估） |
| 整 chunk 单色 | 256 | 64 | `fp ≥ 64` | 59 840 体素 = **1.20 km** | 约 0.3 KB（估） |

⚠️ **本表早先按"1 体素 = 25 cm"折算过，那是错的**：本仓 1 体素 = **2 cm**（`VOXEL_PER_METER = 50`，
`infinite_cubes` 的 room = 148 体素 ≈ 2.96 m 可对账）⇒ 同样的判据下**距离只有那张表的 1/12.5**。上面
已按 2 cm 重算。屏幕分辨率翻倍则这些距离翻倍；上采样则缩小。

### 3.4 视距配档

| 档 | 视距（**原表按 25 cm/体素**） | ≈ 本仓 2 cm/体素 | 全分辨率半径 | 显存（地形 / 密实） | 相对现状帧成本 | 卡点 |
|---|---|---|---|---|---|---|
| A | 1.5 km | **120 m** | 4 chunk | 0.16 / 0.51 GB | 约 1.0 倍 | 内存 |
| B | 3–8 km | **240–640 m** | 4–6 chunk | 0.05 / 0.2 GB | 1.2–1.5 倍 | 内存转带宽 |
| C | 16–30 km | **1.3–2.4 km** | 8 chunk + 多级 proxy | 0.1 / 0.35 GB | 1.5–2.5 倍 | 带宽 / 生产吞吐 |
| D | 100 km 以上 | **8 km 以上** | 同 B/C | 同 B/C | 同 C | 只剩带宽与生产吞吐 |

⚠️ **两个必须一起看的前提**（原文只写了第一个，第二个是硬约束）：
1. 上表的视距按 **25 cm/体素**算的 ⇒ 本仓 2 cm/体素下要除以 12.5（已补一列）。**"30 km" 在 2 cm/体素下
   对应的是 1.3–2.4 km**，不是 30 km。
2. **索引区只有 64³ = 262 144 个 chunk 槽 ⇒ 水平方向硬上限 ±32 chunk = ±164 m**（`b_struct` 的定长
   索引区 + `CHUNK_INDEX_CAP = 64`）。远场的 coarse/proxy 表示**也要占一个槽** ⇒ 无论把 proxy 做得多粗，
   **超出 ±164 m 的东西根本没有槽位可寻址**。所以"公里级视距"不是调参，是**换一层寻址**：
   多级 clipmap / 分层 brick map（每级一个更粗的窗口 + 独立的索引区），或者把索引区改成按需扩容的稀疏表。
   在这之前，请求圈能覆盖到的上限就是那个窗口（±164 m）。

遍历成本是次线性的：步数约 `R_detail·256/4 + (R_far − R_detail)·256/64`，视距扩大到 8 倍时步数约增至 1.44 倍。

### 3.5 编辑侧代价

| 操作 | 代价 |
|---|---|
| resident 全分辨率内一次编辑 | 356 B 上传 / 约 20 µs（现状不变） |
| 首次唤醒一个 chunk（提升到逐体素档） | 一次性约 0.9–2.4 ms（采用预取掩盖） |
| 代表色维护（M2 后） | 沿路径 4 个节点、父层聚合 O(64)/层，约 200–300 ns |

## 4. 里程碑

| # | 目标 | 交付 | 对"大世界"的贡献 | 对"编辑"的影响 | 改动面 / 风险 |
|---|---|---|---|---|---|
| M0 | 可观测性先行 **〔已落地，见 §9〕** | 颜色契约断言（`illegal` 计数恒 0）+ 采样计数器与 GPU 读回（`LOD_DIAG`）；编辑延迟沿用既有 `EDIT[...]` | — | — | 全部在诊断开关后；低 |
| M1 | 叶级 LOD（零新字段）+ 编辑规则桩 **〔已落地，见 §9〕** | `fp = t·px_ang` 派生阈值；level-0 分支：`fp ≥ 1` 且 `mask != 0` 时返回首个非空体素色；`lod.y` 接线 | 叶内步进降约 4 倍；远景不闪 | 只读决策，不写数据 | 1 个 shader 文件；低 |
| M2 | 叶代表色 **〔已落地，见 §9〕** | 叶块的代表值 = **块内按体素数加权的众数槽号**（复用节点字高 16 位，§9）；替掉旧口径"首个非空体素色"，压掉 `fp ≥ 1` 下 4 像素的块内替换误差 | 远景颜色正确性 | 编辑额外一次 64 格扫描（百纳秒级） | 不碰 wire 尺寸；中，独立提交 |
| M2b | 上层三档 LOD（**推迟到 M3 之后**） | `fp` 派生 `depth_cap` 的 16³/64³/整 chunk 三档；前提是分裂节点的代表色（颜色契约） | 视距的真正杠杆 | — | 需先有 M3 的公里级视距才能验证 |
| M3 | 驻留状态机 + LRU + proxy + 编辑三规则 **〔主体已落地，见 §9〕** | `ResidencyPolicy`/`Residency`（档位阶梯 + 迟滞 + 最小驻留 + 编辑钉住 + 每帧上限）、`ChunkTree::proxy`、builder 的 `ensure_resident`/`ensure_resident_tree`/`evict`/`resident_bytes_of`、逐帧调度 `plan_residency` | 有界常驻，大世界成立 | 编辑优先于流式（钉住 + 结构性唤醒） | 剩：CPU 侧卸载与覆盖层落盘（要等 M5 的 `ChunkSource`） |
| M4 | ray-guided 请求通道 **〔切片 1+2 已落地，见 §9〕** | shader 有界 request buffer（chunk key + 所需档位 + 溢出计数）；Rust 异步回读（1–2 帧延迟）+ 合并排序 + 主射线优先于阴影 / GI；关闭时回退启发式半径 | 细节按真实需求分配 | 编辑走优先通道 | shader 与 Rust 各一处；中高 |
| M5 | 生产管线（ChunkSource）+ 粗粒度层 **〔已落地，见 §9〕** | `gate_voxel::{ChunkSource, ChunkProducer, Detail}`（worker 池 + 去重派发 + 非阻塞取回）；`infinite_cubes` 的确定性槽号方案 + 16³ 量化产出；消费端两级半径（全分辨率 / 粗档）与按档卸载 | 粗到细流式、后台产出；视距由粗档那圈撑开 | — | 新模块 + 生成器改造；中 |
| M6 | 环形窗口（窗口跟相机 + 平移索引区） **〔已落地，见 §9〕** | 窗口每帧钉在相机中心；`origin` 变了由 builder **平移索引区**（条目存块相对地址 ⇒ 只搬家）：不丢 CPU chunk、不重传树块；`compute_window` 对钉住的窗口逐字采信 | 跨过 16384³ 上限、消除"接近边界整块重定"的卡顿 | 无影响 | builder + 窗口同步；中 |
| M7 | 试验世界 `infinite_cubes`（M4/M5/M6 的最小可用形态）**〔已落地，见 §10〕** | 程序化生成器（`docs/infinite_cubes.md` 规则）+ 逐帧加载 / **真卸载** + 相机跟随窗口（M6 起靠平移索引区，不再整块重定）；规矩：只在"读系统文件"那一步换成生成、卸载走真实流程、末端不落盘 | 用真实流水线验证流式闭环 | 卸载即丢本地修改（既定语义） | `gate-app/src/infinite_cubes.rs` + 两处 `VolumeGrid` API + `plan_residency` 反向同步；中 |
| **M8** | **多级远场（clipmap 式）〔未开工，验收 = 视距 ≥ 5 km @ 1080p〕** | 近场（L0，现状 ±164 m）之外按 **粒度 ×4 / 覆盖 ×4** 加 3 级：每级 = 一个**带 `scale` 的额外 volume** + 自己的 64³ 窗口 + "每格一次采样"的远场产出；装载**只由射线请求驱动**（远场不铺满）；DDA 按"上一级覆盖半径"给每级**分壳裁剪**（`t_min`） | 梯级（1080p，1 px = d/935）：`g 2cm → ±164m`、`g 0.35m → ±717m`、`g 1.4m → ±2.9km`、`g 5.6m → ±11.5km` ⇒ **L3 覆盖 5 km**，每级内沿粒度恒 ≈ 2 px（接缝不可见）；远场每 chunk ≈ 66 KB / 9 ms | — | `scene`/`infinite_cubes` 多级化 + `snapshot` 的窗口 AABB + shader 分壳；中高 |

关键路径：M0 → M1 → M2 → M3 → M4 → M5 → M6；M7（试验世界）已提前落地为最小可用形态，它会随 M4/M5/M6 一起收敛。

## 5. 量化验收口径

| 维度 | 门槛 | 观测手段 |
|---|---|---|
| 编辑延迟 | resident 内单笔不超过 20 µs（超标不超过 10%）；M2 后额外不超过 300 ns | 既有 `EDIT[...]` 日志 + M0 埋点 |
| 首次唤醒 | 单块不超过 3 ms，且能被预取掩盖 | 唤醒日志 + 帧时间尖峰 |
| 常驻有界 | 巡航 1 km 后显存 / 内存不单调增长；装载不超过 2 chunk/帧（4 MB 预算内） | 驻留日志 + `UPLOAD[...]` 的 bytes 与 mode |
| 颜色正确性 | 非法早停计数恒 0；各档停止时丢弃尺度不超过 1 像素 | M0 的断言与计数器 |
| 确定性 | 同路径回放载入 / 换出序列一致；同 key 生成逐位一致 | 回放测试 |
| GI 稳定 | 驻留切换只影响脏盒；持续流式时远景噪点不长期偏高 | 定点截图 + GI 历史命中率 |
| 视距 | 1080p：全分辨率半径 ≥ 4 chunk（20 m）；三级 proxy 在本仓 **2 cm/体素**下约 75 m / 300 m / 1.2 km；**水平硬上限 = 窗口 ±164 m**（见 §3.4 注 2：再远要换寻址层） | §3.4 的算术（已按 2 cm 修正），运行时以 `b_struct` 大小与 `STREAM`/`RESID` 日志核对 |

## 6. 跨里程碑硬不变量

1. 编辑路径不变：resident 全分辨率内仍是节点级 356 B / 约 20 µs。
2. wire 尺寸不动：M2 的代表色复用节点字高 16 位空位（§9 的事实修正）；其余阶段不改节点读取语义。
3. 主世界保持布局最后（物体不变长），否则 `bases_shifted` 触发全量重传。
4. `mask != 0 && level > 0` 的节点不得当作颜色使用（M0 的断言持续有效）。
5. 驻留切换不等于世界几何变化（GI 只失效脏盒）。
6. 每帧 `UPLOAD[mode=incremental]` 的比例可观测（回归指标）。

## 7. 风险登记

| 风险 | 缓解 | 处理时机 |
|---|---|---|
| M2 代表色的字段语义（16 位里"色 + 权重"怎么分） | **已定**：16 位整段存**槽号**（块内加权众数），不存颜色、不存权重（权重只在写入时用来选众数）；wire 尺寸与 `DUMP_LAYOUT_VERSION` 都不动（M2 已落地，见 §9） | 已解决 |
| 物体变长导致 `bases_shifted` 全量重传 | 物体后置，或改用定长槽位（布局层决定，硬前提） | 必须在 M3 之前 |
| 回读延迟与线程池纪律 | 有界 buffer + 溢出计数 + 主射线优先 + 关闭时回退启发式 | M4 |
| 编辑与流式互抖 | 钉住 + 迟滞 + 最小驻留 + 预取 | M3 |
| GI 每帧失效 | 脏盒粒度失效，不做 `world_rev++` | M3 |
| 坐标精度（i32-chunk 尺度） | M3–M5 把世界限制在 16384³ 内；M6 才做相机相对渲染 | M6 |
| 缺代表色时 proxy 颜色偏 | M2 先于 M3；M3 之前 proxy 只允许 uniform / 截断 | M3 |

## 8. 开工顺序

- 可立即开工：**M0 + M1**（零 wire 变更、全部在开关后、可用"关闭时逐像素与现状一致"做回归验证）。
- M2 单独提交（wire 变更）。
- **已定**（M3 的硬前提）：物体**不设数量上限** ⇒ 保持现有 `layout_order`（物体在前、主世界在后 ——
  主世界增长不漂移前置的 `tree_base`），不做"定长槽位"那种会限制物体数/浪费内存的方案。物体变长的
  代价改为用**增长余量**摊薄（与块内 arena 同一套 `grow_block` 口径），使"物体变长 ⇒ `bases_shifted`
  全量重传"成为罕见事件；真发生时按一次性全量重传处理。

## 9. 实现进度

### M1：叶级 LOD —— 已落地

- `trace.wesl`：新增 `leaf_lod_pal`（4³ 值块的代表色 = 块内**首个非空体素**的色）与两个 trace 变体
  （不透明 / 介质）在"叶层入口体素是空气"时的早停；`view_u.lod.y`（`consts::DDA_LOD`）第一次被真正读取。
  介质变体写 `hit_pal` 而不直接 return ⇒ 代表色若是可穿透材质，仍走原来的介质段语义。
- 回归口径：`DDA_LOD = false` ⇒ `lod.y = 0` ⇒ 逐像素与现状一致。
- 阈值 = **`fp >= 1`**（3.3 表的原值）：一个像素已覆盖 ≥ 1 体素 ⇒ 4³ 块约 4 像素宽。代表色按用户
  决定取**最便宜的规则**：`leaf_lod_pal` 先取块内 `pal`（非空时，O(1)），否则取第一个**置位且非空**
  的体素色 —— 不做平均、不维护新字段、不动 wire。
- 已知代价（明确接受）：块内结构被整体替换 ⇒ 轮廓误差可达整块（4 像素）。视觉不合适就把
  `LEAF_LOD_FP` 调回 4（只影响远场），根治要等代表色。
- 未做（**推迟到 M3 之后**）：`fp` 派生 `depth_cap` 的上层三档。除了颜色契约，更硬的理由是可验证性：
  那三档的门槛是 `fp >= 16/64` ⇒ t ≥ 9 975 体素（1080p ≈ 2.5 km），本仓当前场景（最远 ~1 km）
  **既测不出性能、也看不出画面**，只有 M3 把视距推到公里级之后才能验收。

### M0：可观测性 —— 已落地

- 颜色契约**文档化 + 结构上不触发**（对着色射线保持 `depth_cap = 3`，上层闸门永不进入）；约束写在
  `world.wesl::WorldRayQuery` 与 `trace.wesl` 两处早停点的 `CONSTRAINT` 注释里。
- **诊断计数器 + 读回**（开关 = `trace.wesl::LOD_DIAG`，**单一来源**：Rust 经
  `wesl_consts::trace_consts` 解析同一份源码，不再各抄一份 `bool`）：
  `lod_diag`（BG1 binding 9）= `[采样叶入口数, 叶级早停数, 非法早停数]`，**只增不清**（CPU 读差值，
  无清零竞态）；计数按"叶节点地址 `& 63`"采样 1/64（全量是 ~10M/帧 的单地址原子加，会把收益本身吃掉）；
  读回每 `REPORT_PERIOD_SECS` 一行 `DIAG[leaf_in … lod_stop … % illegal …]`
  （`profiler::report_lod_diag`，与 `upload::dump_voxel_buffers` 同一套同步 readback）。
- 首次覆盖数（castle + 静态机位 + `fp ≥ 1`）：**叶入口的 18.5% 被 LOD 拦下，非法早停 0** ——
  这正是 5 轮人工对照都拿不到的那个数（分子 / 分母）。
- 未做：步数直方图（等 M2b 上层三档落地才有解释对象）。
- 无需新增：编辑延迟埋点 —— 既有 `EDIT[...]` 日志已含耗时（见 3.1）。

### M2：叶代表色 —— 已落地

- **字段**：节点 palette word（wire 第 3 字）的 **bit16..31** 存**叶块代表值** = 块内**按体素数加权的
  众数**槽号（`chunk_tree::leaf_rep_palette`；平手取槽号小者；全空气 = 0）。非叶节点恒 0。
  wire 尺寸与既有读点都不变（读侧一律 `& 0xFFFF`）⇒ `DUMP_LAYOUT_VERSION` 不动（§9 的事实修正兑现）。
- **为什么存槽号而不是平均色**：槽号让命中直接进 `fetch_material`（材质属性照旧）、调色板改色自动
  跟随，也不必另立"命中 = 颜色"的第二条表示；众数 = "这块看起来最像的那个材质"（旧口径取"首个非空
  体素"，可能只是一粒别的色 ⇒ 整块 4 像素被染成它）。
- **维护点只有一处口径**（`pack_palette_word`）：全量序列化 `serialize_with_layout` 与增量节点重写
  `builder::rewrite_node` 共用它 ⇒ 编辑后不需要额外同步，成本 = 一次 64 格扫描（常见块 2~4 种材质）
  百纳秒量级，落在既有"节点级重写"那一笔里。块内材质种类 > 8 时退回"首个非空"（不为噪声块付二次扫描）。
- **读侧**：`trace.wesl::leaf_lod_pal` 先取 `Brick::pal_rep`，为 0（旧数据 / 全空气）才退回旧口径的逐格
  扫描 ⇒ 老 dump 照常可跑；顺带常见路径从"最多 8 次 inline 读"降为 1 次字段读。
  `ChunkTree::rep_of`（proxy 塌缩色）同步改成同一口径，两条路径不再分叉。
- 单测：`leaf_rep_is_weighted_mode`（众数 / 平手 / 全空气 / 未置位格的 tile 色计入 / 爆表退回）、
  `wire_palette_word_carries_leaf_rep`（wire 落位 + 非叶恒 0）；`gate-render` 的 `assert_wire_matches_grid`
  现在逐节点比对**整个** palette word ⇒ 增量与全量的一致性由既有 `incremental_layout_survives_random_edits`
  （200 步随机编辑）持续取证。
- 未做：非叶节点的代表值（M2b 的三档上层 LOD 才要用），仍按"本节点 `palette` / 首个非空子树"。

### M4：ray-guided 请求通道 —— 已落地（切片 1 + 2）

- **产生**（`trace.wesl`）：`REQ_ENABLE`（默认 0，关掉时整段折叠）+ `req_push`。发射点 = **chunk 级 DDA
  里"窗口有这个槽位、GPU 上没有树块"**（`entry == 0`）那一步；只报"这个距离还需要细节"的
  （`req_level(fp)` 给出整 chunk 档 ⇒ 连一个平色都够 ⇒ 不发）。请求字 = chunk 相对窗口下标（3×6 位）
  + 所需档位（2 位，与 `residency::want_level` 同一阶梯，两边不分叉）+ 射线类型（1 位，留给消费端排序）。
- **通道**：`bindings::lod_req`（BG1 **@binding(10)**，read_write 环缓冲；10/11 曾是已删的反射缓存
  乒乓，那条"别再往 10 起加东西"的告诫已随之更新）+ `brickmap::consts::{REQ_CAP, LOD_REQ_WORDS}`：
  `[0]` 累计条数、`[1]` 保留、`[2]` 用途戳计数器 + 用途戳表（`USE_WORDS` 格稠密表）、其后 `REQ_CAP`
  条（满了覆盖最旧的）。开关与节流常量
  （`REQ_ENABLE` / `REQ_SAMPLE` / `REQ_PER_RAY_MAX`）的**权威值只在 `trace.wesl`**，Rust 经
  `wesl_consts::trace_consts` 解析同一份源码（单一来源；先前那种"两侧须同时改"的约定已删除）。
  ⚠️ **`REQ_CAP` 就是"请求通道的信息量"**：环是最近性采样，而"每条射线只报最前面的两个缺失
  chunk"⇒ 最近的缺失块被反复报、把远处的挤掉。1024 只覆盖"刚执行的那一小片屏幕"（实测合并后
  只剩 8–27 个 chunk，全是眼前那一对）⇒ 消费端据此装载时，加载区域退化成半径立方体（实跑：暂停
  流式飞出去看到的正是"规整的立方体"）。现在 **1 M 字（4 MB）= 整屏采样射线一帧的量**，合并后
  给出上千个 chunk 的需求分布；回读侧用**稠密 64³ 计数器**（键就是窗口相对下标，18 位）而非
  HashMap —— 几十万条事件的哈希要 20–30 ms（每 2 s 抖一下），稠密数组是 1 M 次顺序写（~2 ms）。
- **主射线优先**：请求闸门由 `WorldRayQuery::req` 给 —— 主 pass 用 `world_cfg_primary`，阴影 / 反射 /
  GI / 光柱 / 光束一律 `world_cfg_full`（`req = 0`）⇒ 次级射线整段被折叠（不是靠"少发"）。
- **节流（首轮实跑后补上，必须有）**：首版没有闸门 ⇒ 实测 `REQ[本窗口 4.8 亿条、溢出 4.8 亿]`：GPU 全
  耗在 `lod_req[0]` 这一个字的原子写上，帧率崩到个位数 —— 而流式是逐帧的 ⇒ 表现为"**加载停了**"。
  现在三重闸门（都在 `trace_grid` 的发射点）：① 只要"需要 16³ 或更细"的（`fp < 16` ⇒ 100m 内，
  更远的连平色都够，见 `REQ_LEVEL_KEEP`）；② 按**射线哈希**采样 `1/REQ_SAMPLE = 1/16`（同一条射线
  稳定命中、不闪烁）；③ 每条射线最多 `REQ_PER_RAY_MAX = 2` 条。
  **容量不写死在 shader**：`req_push` 用 `arrayLength(&lod_req)` 取绑定实际长度 ⇒ 与 Rust 的
  `LOD_REQ_WORDS` 同源。（首轮还撞过一次"两侧常量不一致"：WESL `REQ_CAP` 4096 / Rust 1024 ⇒
  越界写 + 回读解包全是垃圾 ⇒ 这条同源设计就是为了让这类不一致不可能再发生。）
- **回读 + 消费**（`profiler::report_lod_requests` → `LodRequestFeed` → `infinite_cubes::stream_chunks`）：
  每 `REPORT_PERIOD_SECS` 同步读回、**合并**（同一 chunk 的条数 = 有多少条主射线要它、档位取最细）、
  还原成**绝对** chunk 坐标后写进跨世界的 `LodRequestFeed`（`Arc<Mutex<..>>`，与
  `UploadCpuSampleChannel` 同一套手法）；`stream_chunks` 第 ① 步把请求**插在半径补块之前**（票数多的
  先）⇒ 同样的帧额先补"人在看的"。本窗口没有请求时把表清空 ⇒ 逐字退回纯半径启发式。
  日志：`REQ[去重 M chunk（最热 …）；本窗口 N 条、超容丢失 K；用途戳 U chunk]`。
  策略抽成纯函数 `plan_generation`（单测 `requests_outrank_radius_fill`：请求优先 / `fill = 0` 时
  只放请求 / 无请求逐字等价 / 三种丢弃条件）。
- **装载跟请求走（未命中 = 换入）**：请求是"射线撞上 `entry == 0`（GPU 上没有）"⇒ 这是缓存**未命中**，
  是"该装哪一块"的权威来源。两条路径都必须接上它，缺一就退化：
  - `plan_residency`（GPU 侧）把请求的 chunk 补进 `wants`（档位直接取 `LodRequest::level`，两边
    同一把尺）—— **不能只按用途戳建需求**：射线没进过那个 chunk ⇒ 它没有戳 ⇒ 永远进不了需求集
    ⇒ 缺口再也补不上；
  - `stream_chunks`（CPU 侧）按请求产出（若 CPU 也还没有那块）。
- **档位跟请求走**（论文的 "refinement 由渲染结果给"）：`LodRequest::level`（0/1/2/3 = 全分辨率
  /16³/64³/整 chunk）直接映射成 `Detail`（`detail_of_req`）与 `Level`（`LEVEL_OF_REQ`）。按距离
  重算会把"射线只要 64³ 的远处大块"升到全分辨率（同一份内存少装 20 倍），且档位随相机微动跳变
  ⇒ **细化 → 几何内容变 → 已在收敛的 GI 时域永远接不上**。距离判据只剩**半径补块**那一路在用。
- **用量戳（usage stamp）当 LRU 的唯一信号**：`[2]` 是单调计数器、`[3 .. 3+USE_WORDS)` 是窗口内每
  chunk 一格的**稠密**表（`atomicMax` 写，按构造不可能溢出）。取代原先"每射线一条环记录"的可见性
  投票 —— 后者实测把环打爆、消费端读到的是垃圾。回读侧按"(上次戳, 本次戳]"区间筛出一批
  `LodUse` 发给 `ChunkUseFeed`。
- **池容量（定长 pool）**：`Streaming::request_bytes` 是 **GPU 常驻池预算**，经
  `gate_render::pool_capacity_chunks` 换算成格数；**CPU 与 GPU 两侧用同一个换算**（各算各的会出现
  "CPU 还留着 / GPU 已换出"的错配）。超预算 ⇒ `pick_evicts` 按 `last_use` 换出最旧。
  换出的只是 **GPU 侧**；CPU 树的去留由 `stream_chunks` ② 按**容量 + LRU** 独立裁（同一个信号、
  同一个容量、同样的保护半径 `unload_radius`）。已删除 `REQUEST_TTL_SECS` 那套"最后被请求时刻"口径。
- **`fill`：池满就不许再按半径产出**。`plan_generation` 的 `fill` = 池容量 − 窗口内常驻；请求
  **不占** `fill`（未命中换进一个槽位、同时挤出一个 LRU 槽位），半径补块只吃 `fill`。不做这件事的
  直接后果是空转：实测 `ready` 队列恒顶在 `READY_MAX`（≈384 棵已产出待挂载的树）、每帧白挂 3.7 个
  chunk 又被 LRU 立刻换出（`resident` 不动、`gen`/`unload` 每帧都在动）。
  **装载即使用**：CPU 侧挂载时也记一笔 `last_used`（与 GPU 侧 `note_resident` 同口径）—— 请求带进来
  的块在射线眼里还是"缺的"，不记它就会"一挂上就被判最旧 ⇒ 换出 ⇒ 下窗口又请求"。
- **窗口是请求唯一硬边界**：±32 chunk ≈ ±164 m（`b_struct` 索引区的定义域）。要再远就是 M8 的事。
- **已验证**（`GATE_BENCH=static/orbit` + `GATE_LOG=info,gate_render=debug,gate_app=debug`）：
  `REQ` 从"每 2 s 窗口三千多万条"降到 **700–1300 条**（去重后 6–71 chunk，`超容丢失 0`），
  `RESID` 收敛（resident 恒定、`evict` 与 `install` 同量级），`STREAM` 的 `chunks` 恒等于 `cap`
  且 `ready 0`。
- 单测/验证：`cargo test --release -p gate-render wesl`（WESL 编译 + 校验）、`gate-app` 的单测、clippy
  无新增告警。

### M5：生产管线 + 粗粒度层 —— 已落地

- **管线**（`gate_voxel/produce.rs`，**只用 std**：`thread` + `mpsc`）：`trait ChunkSource`
  （`produce(coord, detail, scratch) -> Option<ChunkTree>`，在 worker 线程上被调用 ⇒ `Send + Sync`、
  不碰主线程的 grid）+ `ChunkProducer`（N 个 worker、每 worker 一条队列轮流派发、`inflight` 去重 +
  在飞上限、`poll` 非阻塞取回、`Drop` 时 join）。主线程只剩"派发需求"与"挂载产出"。
- **挂载**仍走既有路径：`VolumeGrid::mount_chunk_tree`（整体替换、自会标脏）—— 新增的
  `VolumeGrid::take_chunk` 让 worker 的暂存把树**搬**出来（不 clone 整棵树）。
- **确定性槽号方案**（后台生产的硬前提，`infinite_cubes::{slot_of, entry_of, material_slots}`）：
  槽号 = `2 + n_pbr + 类块 × 16 + 色相 × 2 + 档位`（PBR 走 `2 + asset`），与生成顺序 / 线程调度无关；
  建世界时用 `material_slots` 把**整表**装进调色板 ⇒ worker 从头到尾不写调色板。
  取代了原来的"在调色板里找同内容 / 第一个空槽"（那依赖顺序，并行会分叉且与主线程争调色板）。
- **粗粒度层**（`Detail::Coarse`）：`build_region` 的粗档 = 每 16³ 格用 `voxel_at` 采样 2×2×2 个点
  取众数（平手取槽号小者），整格填成该色（32cm 块）⇒ 树小一两个数量级、产出也快。
  `Detail` 的声明序 = 精细度序（`Coarse < Full`），消费端用 `>=` 判"够不够细"。
- **消费端两级半径**（`infinite_cubes::plan_generation` + `Streaming`）：`load_radius` 内全分辨率、
  `coarse_radius`（xz 圆柱 8 chunk）+ `coarse_height`（±6 chunk，y 薄板）内粗档；卸载**按档**
  （同一形状，否则会在边上反复装卸）；粗档 chunk 进到 `load_radius` 内会被重新全分辨率产出顶掉
  （`mount_chunk_tree` 整体替换）。
- **调参口径（首轮实跑后定）** —— 首版还有一个"极慢"的问题，两处原因与修法：
  1. **挂载预算**：`per_frame = 1` 是同步生成时代的旋钮（那时瓶颈是生成），生成搬到 worker 之后
     它就成了硬瓶颈（粗档环 3757 个 chunk ⇒ 一分钟才铺完）。现在按**序列化字数**给预算
     （`Streaming::mount_words` = 256 K 字 = 1 MB/帧 ≈ 0.5 ms/帧，与 `UPLOAD_BYTES_PER_FRAME` 同量级）
     + 条数上限 48：**每帧约 3.7 个全分辨率或 87 个粗档** ⇒ 粗档环约 43 帧、全分辨率圈约 27 帧铺满。
     产出侧：派发 128/帧、in-flight `workers × 32 = 96`（派发只是哈希插入 + channel send ⇒ 便宜；
     让**挂载预算**成为唯一节流阀），待挂载队列在 `Pipeline::ready`。
  2. **需求表缓存**：需求侧原来每帧扫整个环 + 排序（数千项）≈ 1 ms 帧时间，且算完就丢。现在
     `Pipeline::demand` 存**排好序的候选表**，只在「相机换 chunk」或每 `DEMAND_REBUILD_FRAMES = 30`
     帧重建（顺带吸收 ray-guided 请求的更新）；每帧只从表头派发（已满足 / 已在飞的当场划过）。
  3. **排序**：原来只按距离 ⇒ 帧额被平均撒到相机身后（"完全没有 ray-guided 的感觉"的主因）。
     现在排序键 = **近处圈绝对优先 → 视野锥内（±60°）→ 距离 → 坐标**（`DdaCameraConfig::forward`）。
- **尺度实测**（`coarse_detail_is_quantized_and_small` 同一批数据，27 chunk 区域）：
  全分辨率 **69.4 K 字/chunk（278 KB）**、粗档 **2.9 K 字/chunk（11.7 KB）** ⇒ 粗档便宜 **24 倍**；
  两个环合计常驻 ≈ **80 MB**（35 MB 全分辨率 + 44 MB 粗档）。运行期取证：`STREAM[gen N(req M)
  unload K chunks C ready R]` —— `ready` 常年在涨 = 挂载预算不够，`req M > 0` = 请求真在驱动加载。
- 单测：`producer_output_matches_sync_build`（**管线产出 ≡ 同步产出，逐位相同** —— 搬线程 + 改槽号
  都不该动一个体素）、`material_slots_match_room_lookup`（方案表 ↔ 按 room 查询一致、槽号单射、
  `n_pbr = 0` 不造 PBR 变体）、`coarse_detail_is_quantized_and_small`（粗档与点查询同源 + 字数 ≤ 1/4）、
  `view_cone_outranks_distance`（同距离时视野内的先于身后的）。
- **未做**：① 磁盘按层文件 / `.vox` 区域那两种源（`ChunkSource` 的接口留好了，实现只用程序化）；
  ② 更粗的层级（64³ / 整 chunk）与"按预算限常驻"—— 那要等 M6 之后按视距实测决定。

### M6：窗口跟相机 + 平移索引区 —— 已落地

- **窗口每帧钉在相机中心**（`infinite_cubes::stream_chunks` 第 ⓪ 步：`origin = 相机 chunk − dims/2`）。
  旧版是"离边界 < 6 chunk 就整块重定"（把全部 CPU chunk 卸掉 + `demo_force_full_rebuild` ⇒ 全量重传，
  一次性卡顿、且丢本地修改）；现在这条路整段删掉了。
- **平移索引区**（`BrickMapBuilder::set_window` / `VolumesBuilder::sync_windows`）：窗口条目存的是
  **块相对地址**（`base + 1`，与相位无关）⇒ 相位一变只需把条目**搬家**：**不重传树块、不重新序列化、
  不动 CPU 侧内容**。代价 = 索引区（定长 `TREE_BASE` = 64³ 槽 = 1 MB）一次重写。
  CONSTRAINT：**必须两阶段**（先全读出、再整体清零、最后按新槽位写回）—— 平移是**循环位移**，某个
  chunk 的新槽位可能正是另一个 chunk 的旧槽位；边读边搬会互相覆盖。首轮就是边搬 ⇒ 画面上出现
  **"别处 chunk 的几何"那种错位巨块**（截图里的黄色巨平面），且随每次跨 chunk 越坏越多。
  单测用"两个相邻 chunk + **反向**平移"钉死这条（naive 版必挂）。
  `extract` 里把"窗口动过"并进提前返回条件（它是唯一"没有脏 chunk 也要跑 builder"的理由）。
- **`compute_window` 对钉窗逐字采信**（`stream_window` 有值时不再按内容扩张）：否则 `origin` 会随
  内容漂移，平移就无从谈起。
- CONSTRAINT：常驻环半径 ≪ 半窗宽（现在 8 ≪ 32）—— 否则相机会走到"窗内但已过界"的 chunk 旁边，
  那些 chunk 会被卸载规则带走。相机**跳变**（debug 位移）会让内容掉出窗口，此时条目被清（GPU 视作
  空气）、CPU 内容保留（与"卸载即丢"的语义一致）。
- **⚠️ 修：`grid_descs` 必须在增量路径上传**（实跑暴露；M6 引入的"影子相位"）。shader 的 `make_grid`
  只读 `grid_descs`（BG2 binding 0；**不读** `globals`），而 `upload::prepare` 的**增量**分支原先只
  `ensure_capacity` 了 `grid_descs_buf`、**没写内容**（只有全量分支写）⇒ 首次全量之后，shader 一直用
  **开局那个窗口**的 `index_origin` / `index_dims` / `aabb` 寻址，而索引区早已按新相位搬过家：
  · 整片空间随相机**每跨一个 chunk 平移一个 chunk**（"相机微动、整个世界位移"）；
  · 跑出开局那个 AABB 之后整屏 miss（"跑出范围就不加载了"）；
  · 对**已加载区域**反复发请求（寻址错位 ⇒ 看到 `entry == 0`），`REQ` 常年几万条/窗。
  单卷世界每帧只多写 64 B。`GridDesc` 里 `index_*` / `aabb_*` / `tree_base` / `chunk_count` 都是运行期量，
  这条**不能**当静态描述符省掉。
- **未做**：**相机相对坐标**。2cm/体素下 f32 世界坐标到 ~10⁵ 体素（数十公里）才吃紧 ⇒ 推迟到
  §3.4 的档 D（100 km 级）真需要时再做；在那之前 i32 chunk 坐标 + f32 世界坐标够用。
- 单测：`set_window_translates_index_entries`（条目搬家 / 树块一字不动 / 掉窗只清条目）。
- 取证：`WINDOW 平移 … → … dims …`（`debug!`）。旧的 `STREAM[窗口重定 …]` 不再出现 = 不再整块重定。

### 颜色契约（M1 起生效，M2 要延续）

`gd >= depth_cap` 这条早停返回的是**本节点**的 `pal`，而当前子块是**分裂**的 ⇒ 它不保证是该子块的
颜色（通常 0 = 空气）。因此：**只有"只消费 t"的调用者可用**（beam 预 pass，且给出的是保守下界）。
上层三档（`depth_cap <= 2`）要对着色射线启用，前提是给**分裂节点**一个代表色（M2 只给了叶块，那三档属 M2b）。

### M2 的一处事实修正（降低风险）

wire 的**节点字高 16 位**（`pack_palette_word` 的 bit16..31）原本就是「LOD 子树多数色」字段，曾经整段
移除且恒 0（因为旧实现逐节点递归计票占序列化 88%），shader 侧所有节点读取都 `& 0xFFFF`。
⇒ M2 的代表色复用了这 16 位：**没有 +1 word、没有 `DUMP_LAYOUT_VERSION` +1**。字段语义已定 =
**叶块代表值（槽号）**，见 §9「M2：叶代表色」。

### M3a-1：常驻决策核心 —— 已落地

- `brickmap/residency.rs`（新模块）：`ResidencyPolicy`（预算 / 需求半径 / 迟滞 / 最小驻留 / 编辑钉住 /
  每帧唤醒上限）+ `Residency`（账目与决策），**纯逻辑**（不碰 GPU、不碰树）。7 个单测覆盖：预算关闭
  空转、LRU 换出先后、钉住与"刚驻留"必须跳过、迟滞边界、`must_keep`、唤醒的排序/截断/半径过滤、
  切比雪夫距离。
- `builder.rs` 新增四个 API：`is_resident` / `resident_words`（预算按"块占用字数×4"算）/
  `ensure_resident`（唤醒 = 从 CPU 树整块重装）/ `evict`（只归还 GPU 块，CPU 树不动）。2 个单测证明
  **换出 → 唤醒无损**：唤醒重建的 wire 与 grid **逐节点相同**（含掩码 / uniform 色 / inline / 指针）。
- 默认 `budget_bytes = 0` = **不限 ⇒ 策略空转**，行为与"没有常驻管理"逐字节一致（回归口径）。
- 未落地（M3a-2 / M3b）——三处约束已探明，写在这里免得下一刀再撞：
  1. **逐帧调度不能直接塞进 `upload::extract`**：它在"本帧无脏改动"时提前返回（安静帧根本不跑），
     而常驻决策必须每帧跑；且唤醒需要 CPU 树（只在 `ExtractSchedule` 拿得到）⇒ 要一个独立的
     ExtractSchedule 系统，并与 `UploadSnapshot` 协作（唤醒本身会产生新的上传内容）。
  2. **没有 proxy 短树时"换出 = 那个 chunk 变成空气"** ⇒ 有限预算会直接在世界里打洞。所以
     "启用有限预算"必须与 M3b（proxy）同一刀落地，否则只能拿它做一次性诊断（看日志，不看画面）。
  3. **编辑路径不需要改**：`update_chunk` 对"没有块的 chunk"本来就是整棵重装 ⇒ 规则 1（提升先于写入）
     结构性成立，代价就是 §3.5 那笔一次性 0.9–2.4 ms 唤醒。

### M3a-2：逐帧常驻调度 —— 已落地（已跑通验证）

- `upload.rs`：`ResidencyState`（账目 + 策略 + 本帧被编辑的 chunk）+ `plan_residency`（`ExtractSchedule`，
  排在 `extract` 之后 `.chain()`）。分工：决策在 `residency.rs`（纯逻辑 + 单测），调度只做三件事 ——
  ① 逐 chunk 算距离 → 档位（`want_level`）② 落实安装 / 换出（档位变化的旧块先 `evict` 再
  `ensure_resident_tree`）③ 把这次改动**标脏**（上传由 `extract` 的**唯一**快照带走 —— 早先这里是
  "自己再 insert 一份 `UploadSnapshot`"，那是错的，见 §9 M6 行的注）。
- `dda.rs` 抽出 `px_ang(render_h)`：shader 的 `fp = t·px_ang` 与档位阶梯**共用同一入口**，口径不分叉。
- **实测**（临时把阶梯阈值调低，让当前机位 956 体素也能触发；验完已还原）：

  ```
  RESID[resident 60 107424KB install 2 evict 0]   ← 起始：60 chunk 全树 = 107 MB
  RESID[resident 60  90117KB install 2 evict 0]   ← 每帧恰好 2 个（每帧安装上限生效）
  RESID[resident 60    417KB install 2 evict 0]   ← 30 帧后全代理化：417 KB ≈ 7 KB/chunk
  ```

  走通的东西：需求/档位计算、proxy 生成与安装、每帧上限、**重出 snapshot 被 `prepare` 正常消费**
  （零校验错误）、账目与实际块一致（7 KB/chunk ↔ 单测实测 16³ = 8.1 KB/chunk）。
  30 帧内世界常驻 107 MB → 417 KB。
  ⚠️ 上面"重出 snapshot"这一句已作废（就是 §9 M6 行记的那个 bug）：既然快照必须唯一，本节的实测
  口径要理解成"改动被同一帧/次帧的唯一快照带走"。
- 默认阶梯 = §3.3 表（16³ ⇔ `fp ≥ 4`、64³ ⇔ `fp ≥ 16`、整 chunk ⇔ `fp ≥ 64`）⇒ 720p 下 623 m 起才允许
  16³ 档（轮廓外扩 ≤ 4 像素，与叶级档同量级）。**要更保守就把它 ×4**（16³ ⇔ `fp ≥ 16`，2.5 km 起），
  改一个常量。
- 预算默认仍**不限**（`budget_bytes = 0`）⇒ 只按距离降级、不整块丢弃；有限预算留给 M5 的流式策略。
- 记在案的代价：每帧 O(chunk 数) 的记账（10k chunk ≈ 100 µs/帧）—— M5 规模上来时要换空间索引。
- 验证口径：`RESID[...]` 是 **`debug!`**（按日志规则：周期性运行轨迹）。`RUST_LOG` 会被 bevy
  `LogPlugin` 的 filter 覆盖（实测无效）⇒ 改用 **`GATE_LOG`** 环境变量覆盖整条过滤串
  （`GATE_LOG=gate_render=debug,gate_app=debug`，见 `consts::DEFAULT_LOG_FILTER` 与 `main.rs`）。

### M3b-1：proxy 树（"保住视觉"的那一半）—— 已落地

- `ChunkTree::proxy(keep_extent)`（`chunk_tree.rs`）：把 `keep_extent` **及以下**塌成"子树代表色"，
  保留以上的几何与拓扑；代表色口径 = `rep_of`（先本节点 `palette`，否则按子块序找第一个非空子树；
  叶层取第一个置位且非空的 inline），**与 shader `leaf_lod_pal` 同一口径**。
- CONSTRAINT（这条就是"不许牺牲视觉"的落点）：**不挖洞** —— 子树里有任何实体 ⇒ 塌出的必是**实体色**
  （只有全空气子树才塌成空气）⇒ proxy 的实心集合是原树的**超集**，外扩最多一个 `keep_extent`。
  2 个单测咬死这条：`proxy_has_no_holes`（逐体素"原实体不许丢"+ 全空气区不许长实体 + 节点数必降）、
  `proxy_collapses_only_at_or_below_keep_extent`（`keep = BRICK_FACTOR` 必须与不截断逐点等价；
  塌缩只影响本块，相邻块不受影响；`keep = 256` 整块同色且落在 `root_palette` 上）。
- 代价（实测，城堡 59 chunk，见 §3.1）：全树 90.3 MB → **16³ 只有 479 KB（0.53%）** ⇒ 同样预算能常驻
  **约 190 倍**的 chunk 数。这就是"远场几乎免费"的量化结论。
- 未落地（M3a-2）：逐帧调度（每帧按相机距离分级 → 唤醒/换出 → 重出 `UploadSnapshot`）；**预算默认仍是不限**，
  所以今天行为零变化。分级标准就用 M2b 的阶梯（`fp ≥ 4/16/64` ⇒ 16³/64³/整 chunk）——proxy 档位由**距离**
  定、不由预算定，否则近处被压粗就会直接看出来。

### 实测（profile + 动态机位，已完成）

方法：`--features profile` 的逐 pass GPU 均值日志（每 2 s 一行，`REPORT_PERIOD_SECS`）+ `BENCH_UNFOCUSED` +
`AUTO_ORBIT`（相机绕城堡环绕，横距 ~800–870 m、yaw 周期 18 s、目标圆周期 42 s）；场景 = `castle.vox`
（aabb 1764×518×1512 体素），1280×720、vsync 60 fps。5 轮各 ~87 个采样，**逐样本相位对齐**
（不同轮次的 `gate_dda_trace` 逐样本吻合），比较"逐样本求和"（与相位无关）。原始日志 `logs/run{1..5}_*.log`。

| pass | off | never（闸门永不触发） | fp4（两次复测） | fp1 |
|---|---|---|---|---|
| `gate_dda_trace` | 48.34 | 46.06 (−4.7%) | 43.8 / 44.1（**−9.4 / −8.7%**） | 42.77（**−11.5%**） |
| `gate_gi` | 103.94 | 103.58 (−0.3%) | 101.1 / 100.9（−2.8%） | 94.24（**−9.3%**） |
| `gate_godray` | 7.83 | 7.78 | 6.8 / 6.3（−13.5 / −19.8%） | 6.39（−18.4%） |
| `gate_beam` | 1.04 | 4.96（**+377%**） | 4.9 / 5.0（+374 / +383%） | 1.34（+29%） |
| **total** | 205.58 | 209.57（**+1.9%**） | 203.5 / 204.4（−1.0 / −0.6%） | 189.03（**−8.1%**） |

（单位 = 前 87 个采样的逐样本求和，ms/帧 × 87。）

1. **无异常**：5 轮零 `ERROR` / `WARN` / `panic` / VUID；`gate_dda_trace` 两次同配置复测相差 0.7%。
2. **fp1 是有效档**：总 GPU −8.1%；尾部（p90）`gate_dda_trace` −37%、`gate_gi` −30% —— 掉的正是重帧。
3. **fp4 在当前机位几乎持平**（−1%）：720p 下 `fp ≥ 4` ⇔ t ≥ 2494 体素（623 m），本场景机位在
   793–874 m，只有远端命中；收益被"闸门本身的固定开销"（`never` 档 +1.9%）抵掉大半。
4. `gate_beam` 的 +377% 是**编译产物差异，不是负载变化**：`t_cap_beam = min(t_cap, 0.35/(BEAM_DIV·px_ang))`
   = 54.6 体素 ⇒ `fp ≤ 0.09`，任何阈值都不可能在 beam 内触发；实测也证明它与阈值无关（fp1 仅 +29%）。
   绝对量 0.046 ms/帧 = 单帧 0.3%，不构成问题。
5. **余量分布不均（这条决定 M2 优先级）**：`total` 中位数只有 0.71–0.85 ms，但 p90 = 4.4–6.1 ms、
   p99/max = 11.3–12.8 ms —— 重帧已用掉 16.6 ms 预算的 68–77%。即**轻帧余量 20 倍、重帧只剩 1.3 倍**。
   LOD 的价值正落在重帧：fp1 的 p90 −28%、max −11%（`gate_dda_trace` p90 −37%）。

由此的三处计划修正：

- **M1 阈值已落到 `fp ≥ 1`**（3.3 表原值）：实测比关闭省总 GPU 8.1%、重帧 p90 −28%。代表色按决定取
  最便宜的规则（首个非空体素色），"块内结构整体替换（≤ 4 像素）"的代价明确接受。
- **M2 收窄为"叶代表色"；上层三档改为 M2b、推迟到 M3 之后**：三档门槛 `fp ≥ 16/64` ⇒ t ≥ 9 975 体素
  （1080p ≈ 2.5 km），而当前场景最远 ~1 km ⇒ 现在做既测不出性能也看不出画面。M2 只剩"给叶一个更好的
  代表色"，用来压掉 `fp ≥ 1` 的块内替换误差。
- **M0 的计数与读回下一步就做（不再并入 M4）**：这次归因全靠人工跑 5 轮对照（`never` / `fp4` / `fp1` /
  `off`）才做到；没有计数就分不开"闸门本身的固定开销"（`never` 档 total +1.9%）与"激活收益"。先落
  "采样射线数 + 叶级早停数"两项计数与读回，下一次 A/B 不必再靠 5 轮人工对照。

### 待验收（视觉，需人在场）

1. `DDA_LOD = true` 时远景不应出现整块跳色；开关前后差异只应在远场。
2. 若把 `LEAF_LOD_FP` 调到 1.0：确认 4 像素块的压平在可接受范围内（这是 M2 要解决的问题）。

## 10. 交接记录（截至 2026-09-24，本轮收尾）

### 10.1 已落地

| 项 | 位置 | 验证状态 |
|---|---|---|
| M0 颜色契约 + 采样计数 + GPU 读回 | `LOD_DIAG`（shader 与 `consts` **两侧须同开**）、`profiler::report_lod_diag` | 已跑通：叶入口 18.5% 被拦、`illegal = 0` |
| M1 叶级 LOD（阈值 `fp ≥ 1`） | `trace.wesl::leaf_lod_pal` + `view_u.lod.y` | 已跑通：总 GPU −8.1%、重帧 p90 −28%、max −11% |
| M2 叶代表色（节点字 bit16..31 = 块内加权众数槽号） | `chunk_tree::{leaf_rep_palette, pack_palette_word}`、`builder.rs::rewrite_node`、`trace.wesl::leaf_lod_pal` | 单测 2 项 + 既有 wire↔grid **逐节点整字**比对（含 200 步随机编辑）；**视觉验收待做**（见 10.2 第 3 条） |
| M3a-1 常驻决策核心 | `gate-render/src/brickmap/residency.rs` | 单测 7 项（档位阶梯 + 迟滞 + 钉住 + LRU + 每帧上限） |
| M3a-2 逐帧调度 | `upload.rs::plan_residency`（`ExtractSchedule`，`.chain()` 在 `extract` 之后） | 已跑通：`RESID` 107 MB → 417 KB，每帧安装上限生效 |
| M3b-1 proxy 树 | `ChunkTree::proxy` / `rep_of` | 单测 2 项咬死"不挖洞" |
| M7 试验世界 `infinite_cubes` | `gate-app/src/infinite_cubes.rs`、`gate_voxel::VolumeGrid::{unmount_chunk, set_stream_window, stream_window}`、`plan_residency` 第 ⑤ 步反向同步、`builder::compute_window` 认窗口提示 | 加载/卸载已跑通（`STREAM[gen 1 unload 15 …]`、零错误）；**最后一轮窗口修复未复验**（见 10.2） |
| M4 切片 1+2（请求通道 + 并入需求） | `trace.wesl::{REQ_ENABLE, REQ_SAMPLE, REQ_PER_RAY_MAX, req_push, req_level}`、`world.wesl::world_cfg_primary`、`bindings::lod_req`（BG1 binding 10）、`brickmap::consts::{REQ_CAP, LOD_REQ_WORDS}`、`wesl_consts::trace_consts`、`profiler::{report_lod_requests, LodRequestFeed}`、`infinite_cubes::plan_generation` | WESL 编译 + 校验通过、`requests_outrank_radius_fill` 单测、clippy 干净；**运行期未取证**（两侧开关默认关） |
| M5 生产管线 + 粗粒度层 | `gate_voxel/src/produce.rs`（`ChunkSource` / `ChunkProducer` / `Detail`）、`VolumeGrid::take_chunk`、`infinite_cubes::{slot_of, material_slots, build_region(Coarse), plan_generation}` | 单测 4 项（管线产出≡同步产出 **逐位相同**、槽号表↔查询一致且单射、粗档字数 ≤ 1/4、两级半径与档位规则）；**运行期未取证** |
| 断口取证：「CPU 有 / GPU 无」告警 | `upload.rs::plan_residency` 第 ⑦ 段（`RESID[!!CPU 有 / GPU 无 …]`） | 待 10.2 第 1 条的实跑 |
| M6 窗口跟相机 + 平移索引区 | `infinite_cubes` 第 ⓪ 步、`BrickMapBuilder::set_window` / `VolumesBuilder::sync_windows`、`compute_window` 钉窗、`extract` 的 `window_moved` 触发 | 单测 `set_window_translates_index_entries`（条目搬家 / 树块不动 / 掉窗清条目）；**运行期未取证** |
| 流式调试控件（「世界」页） | `assets/ui/debug_menu.toml` 的 `game/world/stream_pause` + 六个参数滑杆、`debug_menu.rs` 的 world 观察者、`infinite_cubes::Streaming::{paused, load_radius, …}` | 改滑杆即改运行期参数；**暂停** = 冻结整个流式环（不跟窗 / 不派发 / 不挂载 / 不卸载）⇒ 相机可飞出加载边界直接看"世界到此为止" |
| **修：`UploadSnapshot` 被覆盖**（实跑暴露） | `upload.rs::plan_residency`（删掉它的 `insert_resource`）、`upload.rs::extract`（提前返回条件加 `has_dirty`）、`builder::{BrickMapBuilder::has_dirty, VolumesBuilder::has_dirty}` | 已定位：`snapshot()` 会**取走**增量脏区间；`plan_residency` 在本帧 `extract` 之后**再出一份**并覆盖它 ⇒ 挂载的树块与**平移后的索引区**都没上传，而 GPU 端的窗口相位照旧跟着 `globals` 走 ⇒ 画面出现别处 chunk 的几何 + "缺块永远补不上"。实跑已见好转（`REQ` 从 1400 万/窗掉到 3–10 K/窗） |
| **修：`grid_descs` 漏写**（实跑暴露；"整个世界随相机平移"的根因） | `upload.rs::prepare` 的**增量**分支补 `write_buffer(grid_descs_buf)`（原先只有全量分支写，而 shader 的 `make_grid` 只读它、不读 `globals`） | 单卷每帧 +64 B；**待实跑取证**：跨 chunk 时世界不再整体跳一个 chunk、飞出开局 AABB 后仍继续渲染 |
| **ray-guided 装载落地**（请求圈 + 通道加宽 + 档位归距离） | `infinite_cubes::{Streaming::{requests_load, request_bytes}, plan_generation, 卸载三分法}`、`brickmap::consts::{REQ_CAP = 1M, REQ_FEED_MAX}`、`profiler::report_lod_requests`（稠密 64³ 合并）、菜单 `game/world/{requests_load, request_mb}` | 单测 `requests_outrank_radius_fill`、`every_chunk_has_content`；**待实跑取证**：`REQ[...]` 的"去重 N chunk"应从个位数涨到几百~上千、加载边界应顺着视线、粗档装载率 ~2.9 K chunk/s（原先 Full 只 222/s） |

### 10.2 待验证（下轮第一件事）

1. **`grid_descs` 漏写的取证**（§9 倒数第二行；这条的优先级最高 —— 它一次性解释了三轮实跑的所有怪象）：
   在 `infinite_cubes` 里飞过几个 chunk，预期：① 世界**不再整体跳一个 chunk**（相机微动只看到视差）；
   ② 飞出开局那个 AABB（±32 chunk）之后**仍然有画面**；③ `REQ[...]` 的"本窗口 N 条"掉到几十~几百
   （此前常年 3 K–1400 万/窗，因为寻址错位把已加载区域当成"缺失"）。取证日志：
   `GATE_LOG=gate_render=debug` 下 `WINDOW 平移` 与 `STREAM[...]` 应稳定，不再有"同一对相邻 chunk
   长期占满环"。
2. **`infinite_cubes` 的窗口修复**：菜单「游戏/世界」→ 选 `infinite_cubes` → 「重载世界」。预期 = 近处
   ±10m 结构完整、无空洞、无轴对齐断口；超出后整片干净结束。**取证手段已就位**：`plan_residency` 第 ⑦
   段会在"本帧没有待安装动作、相机近旁仍有有内容的 chunk 不在 GPU 上"时报
   `RESID[!!CPU 有 / GPU 无 N chunk（3 chunk 内最近 …）→ 画面上是空洞 / 齐平断口]`（只报数量变化）
   ⇒ 有断口必有这行，不必靠肉眼找。看边界时先用「世界」页的**暂停流式**把流式环冻住，再飞出边界
   （冻结后不跟窗、不卸载 ⇒ 已加载的那一圈原地不动），顺便用同页滑杆量不同半径下的观感。
3. **`UploadSnapshot` 覆盖的修复取证**（§9 最后一行）—— **已过**：相机**静置** ≥ 10 s（等流式环铺满）
   后 `REQ` 的"本窗口 N 条"落到 **700–1300 条**、`超容丢失 0`；`STREAM` 的 `chunks` 恒等于 `cap`、
   `ready 0`；`RESID` 的 resident 恒定、`evict` 与 `install` 同量级。飞行时请求略多（新进视野的
   chunk 本来就要加载），但不再出现"同一对相邻 chunk 长期占满环"。
   *（"环满溢出"这个指标已废弃：那个计数是**累计**量，累计条数一过环容量它每窗口就等于窗口条数，
   看着像 100% 溢出、其实一条都没丢 —— 见 `trace.wesl::REQ_RESERVED`。）*
4. **ray-guided 装载的取证**（§9 最后一行）—— **已过**：`requests_load` 开、`request_mb` 1024。
   飞行一段后**暂停**、退回飞出去看边界 —— 是**顺着视线的一整块**，不是以相机为中心的对称立方体；
   把 `requests_load` 关掉再比一次退回立方体。合并后的 chunk 数在飞行中为 **6–71 个/窗**（不再是
   个位数：请求通道现在只承载"缺了"，"看见了"走用途戳表，不再挤占环）。
5. **M1 的视觉验收**（一直没做）：`consts::DDA_LOD` 开关前后各看一次远景，确认不开时逐像素与旧版一致、
   开时只在 935 体素外交替（`LEAF_LOD_FP` 现为 1.0，块内替换误差 ≤ 4 像素）。
6. **M2 的视觉验收**：远景里 `fp ≥ 1` 的块应呈现**块内多数材质**的色（不再是"首个非空体素"那种偶发跳色）；
   重点看混合材质区（草地+土、砖+灰浆）的 LOD 边界是否比 M2 前更稳。`LOD_DIAG` 打开时可顺带确认
   叶级早停数不变、`illegal` 仍为 0。

7. **尚未定论的观感问题**（缺证据，别急着改代码）：
   - *room 中间的 cube "随移动变化"*：**先排除上面第 1 条**（寻址错位会让整片空间乱跳，看成"材质在变"）。
     若第 1 条修好后仍在，再按 LOD 切换查：`material_of(room, n_pbr)` 是纯函数、`build_region` 的产出与
     生成顺序无关（单测 `producer_output_matches_sync_build` 逐位比对已咬死）⇒ 变化只能来自
     `Detail::Coarse`（16³ = `32cm` 量化）↔ `Detail::Full`（跨 `load_radius` 时**整体替换**）。
   - *静态 GI 不收敛*：需要"`world_rev` 是否每帧自增（`flags.z` = 能否复用历史）"的证据。相关日志
     （`GI 二次顶点缓存 epoch` / `RESID` / `STREAM`）都是 `debug!` ⇒ 跑一次
     `GATE_LOG=gate_render=debug,gate_app=debug`（见 M3a-2 节末的注）就能取证。

8. **帧率**：上面三处修复后应显著回升 —— 世界有"洞"/远景缺失时，**每个像素的射线要走满整个窗口**
   （最多 ~192 个 chunk 步）才 miss，加上持续的装载 + 上传 churn，FPS 自然掉到十几。先量一次
   **静置（世界已铺满）**时的帧率；若那时仍是个位数，用 `--features profile` 拿逐 pass 的 GPU 时间
   （每 `REPORT_PERIOD_SECS` 一行一个 pass）—— 那才是性能问题的下一步取证，不要凭猜改渲染。

### 10.3 已知限制（当前架构的硬边界，不是待修 bug）

- **视距**：全分辨率那一圈仍是 `load_radius` = 2 chunk ≈ 10m。实测（infinite_cubes）全分辨率
  278 KB/chunk、粗档（16³）11.7 KB/chunk —— 后者便宜 24 倍 ⇒ 粗档环给到 xz 8 chunk（≈41m）+ y ±6，
  两个环合计常驻 ≈ 80 MB。（城堡那种密实树是 1.5 MB/chunk ⇒ ±20m 就要 700 MB，那才需要更粗的层级；
  再往外到百米级要 64³ / 整 chunk 档与"按预算限常驻"，未做。）
  ⇒ M4 的"请求驱动需求"现在能用到粗档环边，但**扩视距仍靠档位**、不靠请求。
- **窗口是 `b_struct` 索引区的定义域**（M6 已解决）：相机走远不再"整块重定"，而是**平移索引区**
  （只搬 1 MB 条目、不重传树块、不丢 CPU chunk）。剩下的边界是**坐标精度**：i32 chunk 坐标 + f32 世界
  坐标在 2cm/体素下到数十公里才吃紧 ⇒ 相机相对坐标推迟到真需要（§3.4 档 D）时做。
- `Streaming` 现值：全分辨率 `load_radius / unload_radius = 2 / 3`、粗档 `coarse_radius 8`（xz 圆柱）
  × `coarse_height 6`（y 薄板）、挂载预算 `mount_words 256 K`（1 MB/帧）+ `mount_count 48`；
  卸载半径只比加载大 1（迟滞够用且把全分辨率常驻集真正限住；差 2 会留下 9³ 的尾迹 ≈ 700 MB）。
  这几项都能在菜单「世界」页**实时改**（滑杆写的就是这个资源）；`load_radius` 是内存的硬杠杆，
  `unload_radius` 须大于它（差值 = 迟滞带）。
- `infinite_cubes` 的 PBR 档位已接真资产槽：运行期用 `PbrTextureSet::ids()`、`Startup` 那一块用
  `gate_render::material_ids()`（贴图集是 `Update` 里才插入的资源，见该函数的时序说明）。
- **卸载即丢本地修改**（无覆盖层落盘）—— `infinite_cubes` 的既定语义；要保留就得做 M3 规则 3 的覆盖层。

### 附：GI 代价取证（实测记录，2026-09-25）

复跑口径：`GATE_BENCH=static`（相机与世界都静止的**对照**）/ `GATE_BENCH=orbit`（相机持续绕行），
配 `--features profile` 读 `GPU 逐 pass`（每 `REPORT_PERIOD_SECS` 一行）。两者都不改渲染代码 ——
失焦跑帧与自动绕行是两个**运行期**开关（`consts::bench` / `consts::bench_orbit`）。

**基线**（720p、GI 分帧 ÷8 / 1/2 分辨率 / 降噪高，infinite_cubes）：

| 场景 | 帧 | `gate_gi` | `gate_dda_trace` |
|---|---|---|---|
| static（世界静止） | 24.3–25.0 ms | 19.0–19.6 ms | 4.2–4.3 ms |
| orbit（动态） | 18.4–19.0 ms | 15.2–15.6 ms | 2.25–2.4 ms |
| static + 次级行程夹取 12 chunk | 21.8 ms | **19.37 ms（未变）** | **1.52 ms** ✔ |
| static + `GI 分帧 ÷1` | 110.2 ms | **105.26 ms** | 3.94 ms |
| static + GI 候选改面法线 | 22.2 ms | **16.88 ms** ✔ | 4.23 ms |
| static + 面法线 + GI 分辨率 **1/4** | **9.00 ms** ✔✔ | **4.56 ms** ✔✔ | 3.67 ms |
| **orbit（动态/流式中）+ 上述全部** | **7.41–7.58 ms** ✔✔ | **4.34 ms** | 2.34–2.45 ms |
| （对照）orbit，改造前 | 18.4–19.0 ms | 15.2–15.6 ms | 2.25–2.4 ms |
| + 可见性投票（churn 修复） | 21.5–22.5 ms | 16.3–17.0 ms | 4.1–4.3 ms |
| + 候选按面共享（非认领者不发） | 21.45 ms | 16.35 ms（−4%） | 3.97 ms |
| + 二次顶点缓存 2^19 → 2^24 槽（10 MB → 336 MB） | 22.59 ms | **17.39 ms（无收益，已回退）** | 4.14 ms |

**⚠️ 上面这张表的后半段（churn 修复之后的全部行）作废**：它们全部测于"三处失效架构 bug 还在"的状态
（见下一节），**结论不能再用**。前四行（改造前的基线）仍然有效，可以当"改造前"的参照。

**被实测否掉的四类"看似可省"的东西**（别再往这些方向花时间）：
1. **失效策略**：世界静止反而更贵（19.6 vs 15.6）。—— *结论仍然成立，但当时的解释（"每帧作废历史
   不成立"）是错的*：两种场景其实**都**在每帧作废历史（静态世界也在流式加载），所以两者一样贵。
2. **空区行走**：把 `req == 0` 的次级射线行程夹到 ±61 m，`gate_gi` 一动不动。—— *这一条只在当时那个
   状态下成立*：后来复测（键范围修好之后）12 → 1 chunk 能让 `gate_gi` 16.2 → 6.2 ms，
   即**行程确实是一个真杠杆**。当时测不出来是因为代价被别的东西盖住了。
3. **每 texel 的候选数**：兑现分数预算后 `gate_gi` 16.9 → 17.3，不动。—— *同一类误判*：那条路径当时
   根本没走到（候选数走的是"无历史"的 8 条），所以改它当然没反应。
4. **缓存命中率**：二次顶点缓存放大 33 倍（10 MB → 336 MB），`gate_gi` 16.35 → 17.39，**反而略差**。
   —— **这条是彻底错的**：当时槽里键的 epoch 每帧 +1 ⇒ 整张表每帧自失效、命中率恒为 0，
   放大槽数自然无收益。修好 epoch 之后这张表才开始真的有命中。

### 附二：三处失效架构 bug（2026-09-25 下半场，实测）

三处都是**同一个形状**：判"变了没变"的口径比实际需要粗 / 键的范围比世界小，
于是"加速结构静默退化成关"，代价体现为"每条候选付全价"。

| # | 位置 | 症状 | 修法 |
|---|---|---|---|
| ① | `gi.rs` 二次顶点缓存的 epoch | key 里含"任何上传都自增"的 `world_rev` ⇒ 流式世界每帧 +1 ⇒ `gi_sec_slots` 整表每帧自失效 | 拆成两个口径：`wide_rev`（只全量重建 / 调色板）给 epoch 与 `flags.z`；`world_rev` 只在「太阳反弹」打开时才用 |
| ② | `gi.rs` 的 `flags.z`（允许复用历史） | 同上一条的口径 ⇒ `flags.z` 恒 0 ⇒ `hist_ok` 恒 false ⇒ 候选数走 8 条那条路（`b_err = hi = 4`）× 时域永不累积 | 同上：默认用 `wide_rev` |
| ③ | `gi/common.wesl` 的面键范围 | `gi_key_ok` 只有 **13 bit（±4096 体素 = ±82 m）**，而流式世界的绝对坐标是 ±10000 ⇒ `pk_ok` 对**每个像素**都是 false ⇒ 时域复用、空间复用、逐面认领（`face_slots`）**全部静默关掉** | 键放宽到 **16 bit/轴（±32768 体素 = ±655 m）**：`x|y` 一个 word、`z|face(3)|obj(13)` 另一个 |

**实测（720p、GI 1/2、分帧 ÷8、降噪高、infinite_cubes）**：

| 场景 | 帧 | `gate_gi` | `gate_dda_trace` | `gate_dda_face` |
|---|---|---|---|---|
| static，修 ①②③ 之前 | 18.0–22.8 ms | 13.7–17.7 ms | 3.3–4.2 ms | 0.02 ms |
| static，修 ①②③ 之后 | **5.5–5.7 ms** | **1.13–1.26 ms** | 2.3–2.6 ms | 0.89 ms |
| orbit，修之前 | 18.4–19.0 ms | 15.2–15.6 ms | 2.25–2.4 ms | 0.02 ms |
| orbit，修之后 | **5.25–5.62 ms** | **1.66–1.72 ms** | 1.80–2.07 ms | 0.67–0.73 ms |

**修复过程中的取证（沿用同一套 A/B，都在光栅化状态可比的前提下）**：

| 判据（static，1/2 档） | `gate_gi` | 说明 |
|---|---|---|
| `gi_ray_radiance` 提前 return（stub 掉整条候选射线） | **0.36 ms** | ⇒ 代价**全在候选射线里**，每 texel 的固定开销（prologue / beam / reservoir / flatten）只值 0.36 ms |
| 保留 raycast、去掉二次顶点着色 | 10.5–13.3 ms | ⇒ 其中 ~85% 是 `world_trace_medium`（那条带介质的 GI 射线） |
| `cand_n` 强制 ≤ 1 | **1.08–1.99 ms** | ⇒ 代价线性于候选数，且当时每条候选数是 8（"无历史"路径）而非 1 |
| 次级行程夹取 12 → 1 chunk | 16.2 → 6.2 ms | ⇒ 行程是真杠杆（见上表第 2 条的更正） |

**新的代价模型**：`gate_gi` ≈ `0.3 ms（每 texel 固定）+ 候选数 × ~200 ns`；
候选数 ≈ 被认领的面数 × 每帧每面候选（有历史时再乘 1/4 的轮转抽样）。

**下一步的两个直接后果**（都在同一次修复里解锁）：

1. **GI 分辨率可以拨回 1/1**（画质）：`gate_gi` 只剩 1.7 ms ⇒ 1/1 约 4× ⇒ ~7 ms，整帧仍在 16.7 ms 内。
   之前"1/4 是唯一出路"的结论是那个 bug 状态下的产物，**不再适用**。
2. **分帧 ÷8 可以拨回 ÷1/÷2**（光照响应）：记忆窗 τ = `GI_SS_M_CAP_K × cand_n0 / cand_n` 帧，
   现在时域累积真的成立了 ⇒ ÷8 会把光照响应拖到 ~4 s。射线便宜了，就不必再摊。
3. **主 pass 的 `gate_dda_trace`（1.8–2.6 ms）现在是最大单项**，且 `gate_dda_face` 已经真的在工作
   （逐面着色摊薄）⇒ 下一步取证应该转向它，而不是 GI。

### 10.4 下一步（建议顺序）

**验收目标（本轮用户定死）：视距 ≥ 5 km、帧率稳定 ≥ 60。**

0. **档位梯级（已落地：`Detail` 五档 + `infinite_cubes::detail_at` + 解析量化产出）** ——
   **判据（用户口径，已实现）**：粒度 `g` 体素的一档，只有在该 chunk 的距离处 `g/(d·px_ang) ≤ 1`
   时才允许使用；取允许档里**最粗**的（内存最优）。1080p 基准（`1/px_ang = 935 体素`）下门槛
   `d ≥ g·18.7 m`，换 chunk（5.12 m）就是 `3.65·g`：

   | 档 | 粒度 | 1 px 距离 | **被使用于**（1080p） | 每 chunk 树 |
   |---|---|---|---|---|
   | `Detail::Full` | 2 cm | 18.7 m | **0 – 75 m** | 286 KB |
   | `Detail::Fine` | 8 cm | 75 m | **75 m – 300 m** | ~140 KB |
   | `Detail::Coarse` | 32 cm | 300 m | 300 m – 1.2 km | 10 KB |
   | `Detail::Wide` | 1.28 m | 1.2 km | 1.2 km – 4.8 km | ~3 KB |
   | `Detail::Chunk` | 5.12 m | 4.8 km | ≥ 4.8 km | ~0.3 KB |

   关键点：**"允许用"与"用得多细"是两件事** —— 8 cm 档自己在 19 m 处就够了 1 px，但下一档（32 cm）
   要到 300 m 才够 1 px ⇒ 19–300 m 之间只能用 8 cm 档（取允许档里最粗的），于是**逐体素一路到 75 m**
   （不是 10 m，也不再是"切到 32 cm 就满屏方块"）。**代价**：判据让每一档在自己的外壳处"过细"
   （可达 `ratio` 倍）⇒ 内存/产出都按 `ratio` 涨 —— 这是"处处 ≤ 1 px"必须付的钱。
3. **窗口是硬边界**：请求圈再大也只能到 ±32 chunk（±164 m）⇒ 上表里 32 cm 及更粗的档**今天用不上**
   （它们的门槛 300 m+ 在窗口外）。要让它们上场，得先有 M8 的寻址（见下）。
2. **M8 · 多级远场**（§4 表最后一行）—— 5 km 的唯一路径。四条硬事实：
   - **粒度判据**（1080p）：1 px 在距离 d 处 = `d/935` 米。⇒ 接缝要 ≤ ~2 px，则该级在它的**内沿**
     `d_in` 处必须 `g ≤ d_in/467`。
   - **每级的"覆盖/粒度"比是常数**：覆盖 = `32 × 256 × 级体素`（窗口 64³ × chunk 256 级体素），
     而粒度 `g = cell × 级体素`（`cell` = 树的一个格有几个级体素）⇒ `覆盖/g = 8192/cell`。
     接缝处的像素数 `= g/(d_out/935) = 935·cell/8192` —— **只由 `cell` 决定，与级体素无关**：
     `cell = 16` ⇒ 1.8 px、`cell = 4` ⇒ 0.46 px。所以**每级的壳要窄**（下一级从本级的 `1/4` 距离处接上），
     而 `cell = 4` 时每级能覆盖 **4×** 距离（`cell = 16` 只能 1.1×，等于没有）。
   - ⇒ 梯级（`cell = 4`，粒度/g 每级 ×4）：`g 2cm → ±164m`（L0，现状）、`g 0.35m → ±717m`、
     `g 1.4m → ±2.9km`、`g 5.6m → ±11.5km` ⇒ **3 个远场级覆盖 5 km**，接缝处恒 ≈ 2 px。
     每级远场 chunk 的成本是**均匀**的：64³ 个格 × 1 次采样 ≈ **9 ms**、树 ≈ **66 KB**
     （`infinite_cubes` 这种格子状世界）；靠 view-driven 装载，5 km 视野内的远场约 2500 个 chunk
     ⇒ ~160 MB 常驻、~7 s（3 worker）铺完。
   - **必须只按视线加载**：铺满 ±11.5 km 的远场要 262144 chunk × 66 KB = 17 GB ✗ ⇒ 远场的装载器只能
     是"射线请求"（即已经做好的请求圈 + `request_bytes`），半径环只负责近场保底。
   - **复用现成机制**（这是 M8 便宜的原因）：每级 = 一个 **带 `scale` 的额外 volume** —— `Grid` 的
     `pos/scale`、`grid_descs` 数组、多 volume 的 snapshot/上传/常驻、`set_window` 平移**都已经支持**；
     `infinite_cubes` 的产出只要加"级体素 = scale 世界体素 + 每格一次采样"就能产出远场。
   - 唯一的新机制：**分壳裁剪** —— 每级的 DDA 从"上一级覆盖半径"起（`trace_grid` 的 `t_min` 已有，
     与 beam 的邻居下界取 `max`），否则远场的膨胀块会盖到近场前面。
   - 切片建议：**① 生成器加 scale + 单测**（不动渲染路径，零回归）→ **② L1 接进 `Volumes` +
     `snapshot` 窗口 AABB + 分壳**（验一次 717 m）→ **③ 逐级加 L2/L3，并用实测的 `b_struct` 大小核对
     每级内存**（没有实测之前不要一次把梯级推满）。
1. **帧率（60 fps）**：先取证再动刀 —— `cargo run --release --features profile`，飞行 20 s，把逐 pass 的
   `PROF[...]` 行拿回来（每 `REPORT_PERIOD_SECS` 一行）。已知的结构性开销（按嫌疑排序）：
   - ~~**GI 历史每帧作废**：流式期间 `dirty.boxes` 每帧非空 ⇒ GI 走"不复用历史"的贵路径（每像素重新寻
     光照）。这是"一动就掉帧"最可能的大头，且**与本仓的流式策略直接冲突**（越努力加载，GI 越贵）。~~
     **〔已修，见「附二」：`flags.z` 与二次顶点缓存的 epoch 都改成"世界整体"口径；`gate_gi` 15.4 → 1.7 ms〕**
   - 每帧 O(常驻 chunk 数) 的扫描：`plan_residency` ①+②、`stream_chunks` 卸载三分法、`resident_chunks`
     的 Vec 分配 —— 常驻 1–2 万 chunk 时约 3–5 ms/帧，可用"仅在相机跨 chunk / 常驻集变化时重算"削掉。
   - `b_struct` 扩容：32 MB 一跳、整块 GPU-GPU 前缀拷贝 ⇒ 爬坡期有几十 ms 的卡顿（可改增长策略）。
2. 复验 10.2 第 1 条，并定死 `Streaming` 的三个半径 / 帧额。
3. **M2b / M4 切片 3**：M8 落地后才有意义（那时才有 16³/64³/整 chunk 三档的用武之地）。
4. **相机相对坐标**：M8 的 L2 到 ±10.5 km 后，f32 世界坐标（2 cm/体素 ⇒ 5 km = 25 万体素）仍在
   安全区（~10⁵ 体素吃紧是数十公里）；要再放到 30 km+ 就得与"相机相对坐标"一起做。

### 10.6 常驻内存：节点布局 + 一处 churn（2026-09-25 下半场）

**A. `ChunkTree` 节点改平坦布局（`gate-voxel/src/chunk_tree.rs`）**

实测（`cargo test -p gate-app mem_per_chunk -- --nocapture`，`infinite_cubes` 全分辨率，直读
`ChunkTree::heap_bytes`）：一个 chunk 有 **25,336 个节点**，其中 **97.3% 只是 `Node::Uniform`**（载荷
2 B）、`Leaf` **0 个**。而 `Node` 是枚举 ⇒ 每个节点都按最大变体（`Leaf` 的 32 字值表）占地
**152 B**，且 `Split` 还要各自一个 `Vec<u32>` 头。

改法：`Node` → **定长 16 B** 的平坦结构（`mask` / `off` / `palette` / `kind`），变长载荷移到两个池 ——
`children`（紧凑子块表，`u32` 顺序池 + `(off,len)` 段表首次适配，块容量按 2 的幂取档）与
`leaves`（值块值表，`mask == 0` 的共用一张全零表）。`NodeView.children` 仍是紧凑 `&[u32]`，
wire 序列化输出与 `dirty` 的"节点 id"语义逐位不变（既有测试即闸门）。

| | 树（每 chunk） | 序列化 |
|---|---|---|
| 改前 | **3.79 MB**（25,336 × 152 B） | 286 KB |
| 改后 | **0.54 MB**（25,336 × 16 B + 池） | 286 KB |

整进程实测：池容量 585 → 私用 **7.9 GB**；改后 3,843 → **12.4 GB**（每 chunk 由 13.5 MB 降到 3.2 MB）。

**B. 派发没挡"已产出待挂载"⇒ 同一块被反复重产重挂（`infinite_cubes::stream_chunks`）**

`producer.in_flight()` 只覆盖 worker 手上的；产出被 `poll` 取走后它就不为真，于是 `plan_generation`
把同一块重新派发。`ready` 队列（`READY_MAX = 384`）因此恒满、真正的**新**块被堵在队尾：

    STREAM[gen 3(req 0) unload 0 chunks 1941 cap 4096 ready 378]   ← chunks 不动、gen 每帧都在涨

修法：`Pipeline` 加 `ready_set`，派发时把"已在待挂载队列"当作"已经在飞"；`poll` 时同块只留先到的那份。
修后同一机位**收敛**（尾部无任何 `STREAM` 行、`REQ[本窗口 0 条请求]`），GPU 5.58 ms/frame。

**C. 池预算的每块字节数改成直读值**

`CPU_CHUNK_BYTES_EST` 原为 `7 MB`（用"两次运行的进程私用内存差"推的，把 GPU `struct_buf` 一起算了进去）
⇒ 改为 **1 MiB**（直读 `heap_bytes` 的 0.54 MB + 一倍余量给编辑后出现值表的 chunk）。默认预算随之为
**2 GiB ≈ 2,048 块**（本机 16 GB：私用 8.8 GB）。

**D. 剩下的每块开销**（下一站）：差分实测**池容量每 +1 块，进程私用 +2.0 MB**，而 CPU 树只占 0.54
⇒ 约 **1.5 MB/chunk 在 GPU 侧的 builder/wgpu**（`struct_buf` 高水位 + `BrickMapBuilder` 的逐 chunk
槽表与空闲段表）。要再往 5 km 推，这一项比 CPU 树更值得动。

**E. 编辑优先通道：现状已满足** —— `pending_data` → `builder.update_chunk` 是**无条件**执行的，
编辑的装载直接进 builder，绕过池预算与请求队列；`note_edit` 再钉住 120 帧并禁止降级。请求字里的
`kind` 位（bit 20）目前无人读，**没有活可干**。编辑不落盘（流式世界的既定语义）。

### 10.7 两处"永不收敛"（2026-09-25 深夜，实测）—— 视野内加载慢 + GI 抖动/彩色线条横扫

**A. GPU 字节预算按"均值"给 ⇒ 池被卡在预算线上，`install` 恒为 0**

`GPU_CHUNK_BYTES_EST` 曾写成 `304 KB`（实测每块 **367 KB**：`RESID[resident 952 349851KB]`）。
`pick_evicts` 只在"实际字节 > 预算"时换出 ⇒ 低估 20% 让预算线**先于块数上限**到达：

    RESID[resident 1897 622573KB install 0 evict 1]
    RESID[resident 1896 622181KB install 0 evict 4]   ← 每帧重复，install 永远 0

后果正是两个报告现象：新挂上的块立刻被换掉 ⇒ **视线内的缺口再也补不齐**（"加载的很慢"）；同时
每 2 s 有 4 块被换出+重装 ⇒ 每次重装都改该 chunk 的 AABB ⇒ **GI 在那个区域被反复作废**，作废带按
LRU 顺序扫过视野（"彩色的线条一直在平移"）。

改法：常数取**每块字节上界**（`384 KB`，全分辨率为主时的实测最大值 + 5%），并写明"必须取上界"。
修后同一机位**完全静止**：`REQ[本窗口 0 条请求]`、无任何 `STREAM` / `RESID` 行（`gen/unload/install/evict`
全 0）。**教训：任何"字节预算 vs 块数上限"并存的池，预算必须按上界给；按均值给就是把上限提前。**

**B. 每帧安装上限 2 ⇒ 冷启动 17 s**

`ResidencyPolicy::max_install_per_frame = 2`（本意是把 0.9–2.4 ms/块的尖峰摊到多帧）。池空时没有尖峰
可摊，摊的代价是 `2048 / 2 / 60 ≈ 17 s` 才填满；而全部安装的 CPU 总账只有 `2048 × 1.65 ms ≈ 3.4 s`。
改为 **8/帧** 并把挂载字数预算从 256 K 提到 1024 K（8 × 73 K）：实测填满 **3.4 s**（正好到 CPU 底线）。

**C. GI 记忆窗按"帧"给 ⇒ 在 `interval = 8` 下只剩 2.7 s / 80 样本**

`GI_SS_M_CAP_K = 20` 的字面含义是"20 **帧**"，但 GI 每 `interval` 帧才一个 tick（默认 8）⇒ 窗的墙钟
长度 = `20 × 8 = 160 帧 ≈ 2.7 s`，窗内样本 ≈ `20 × 4 候选 = 80` ⇒ 噪声下限 `1/√80 ≈ 11%`。

**试过 96 / HQ 160，画面更差（偏移/拖影加重），已撤回。** 结论：这个窗**不是**"水彩晕开 + 彩色线条
一直在动"的根因 —— 加长它只是把陈旧历史的权重放大。撤回的依据是用户实测反馈。

**D. 复用 / 认领这条链在本轮之前**从未真正运行过**（这才是 GI 现状的关键前提）**

`gi_key_ok` 原先只有 ±4096 体素（±82 m），相机远在 ±6000 体素 ⇒ **每一个像素**都判 `pk_ok = false`
⇒ 时域合并（`gi_res_adopt`）、空间复用（③）、逐面认领三件事**全部静默退化成"关"**。本轮把键放宽到
±32768 之后它们才第一次真正生效 ⇒ **现在看到的形态是这条链的首次真实表现，不是"从好的状态退化"**。

所以"水彩晕开 / 彩色线条"应当按"这条链没被调过"来排查，而不是继续调记忆窗。定位顺序（都在
「渲染 / RESTIR GI」菜单里，各 60 秒、无需改代码）：

| 步骤 | 操作 | 若是它的问题会看到 |
|---|---|---|
| 1 | **降噪 = 关** | 变成纯噪点、**波纹消失** ⇒ 降噪链（atrous 判据 `GI_DEN_N_DOT` / `gi_den_same_plane`）放行了跨边界样本 |
| 2 | **分辨率 = 全** | 波纹的**尺度/形态明显变化** ⇒ 半分辨率的上采样/采样栅格是元凶 |
| 3 | **启用 = 关** | 画面明显变干净 ⇒ 确认是 GI 而非主 pass（否则要去查主 pass 的粗档代理） |
| 4 | 移动 vs 静止各一张截图 | 区分"屏幕锚定"（不随相机走 ⇒ 栅格/上采样）与"世界锚定"（跟着面走 ⇒ 复用/认领） |

### 10.8 夺回「逐体素」：两处 8fc4a63 退化（2026-09-26）

**现象**：斜视的墙面 / 地面出现**沿格行的长条**（条纹与台阶行对齐、沿长度方向拉长），逐体素风味
消失。`da18260` / `75f08a6` / `d2e0d77` 都没有这个形态。

**定位**：castle 默认机位下做**同机位实拍 A/B**（改 `data/config.toml` 的 `[menu]` 值或着色器常量 ⇒
只重启、不重编），逐条否证：

| 改动 | 长条 |
|---|---|
| 降噪质量 = 关（整条降噪链不跑） | 在 |
| `GI_KEY_LIMIT = 4096`（时域复用 / 面认领 / 逐面均值 / `dda_main` 查表全关） | 在 |
| `SECONDARY_REACH_CHUNKS = 0` | 在 —— **它在本场景是空操作**：窗口只 ±32 chunk 而相机在窗口内，±3072 体素盖满全窗口 ⇒ `w_mn/w_mx` 一字未变 |
| `voxel_normal` 换回 GI 候选的着色法线（叶级 LOD 仍开） | 在（长条换了个来源） |
| `LEAF_LOD_FP` 关（= `3.4e38`）（`h.n` 仍在） | 在 |
| **上面两条同时改回** | **消失，与 `da18260` 一致** |

**根因：两处独立退化，都出自 `8fc4a63`**

1. **GI 候选的着色法线**从 `voxel_normal`（体素中心的梯度法线）换成 DDA 的入射**面**法线 `h.n`
   （为省 6 次邻域探测）。面法线是**逐面**的常数方向（台阶顶面 / 侧面各一个方向）⇒ 起伏表面上整个
   半球的取向整片偏转；而主 pass 的 `shade_face` 一直用 `voxel_normal` ⇒ GI 与**可见着色**的法线
   不再一致，间接光看起来"照着另一个面"。
   改法：`gi_ss_main` 恢复 `let n = voxel_normal(gg, h.voxel, h.face_id, h.n);`。
2. **叶级 LOD 第一次真正生效**：`LEAF_LOD_FP = 1` 在此之前是**死代码** —— `leaf_lod_pal` 依赖 M2 写进
   节点字高 16 位的「块内众数槽号」，而 M2 正是 `8fc4a63` 才落地的 ⇒ 这条早停从"几乎恒返回 0"变成
   "近中景整片命中" ⇒ 「一个 4³ 值块一个色」。
   改法：`LEAF_LOD_FP = 3.4e38`（关）。代价 = 总 GPU +8.1%、重帧 p90 +28%；要保留收益就把阈值推到
   远场（`fp >= 4` ⇔ ≈935 m）。

**同时撤回的误判**：曾据「复用链在 `8fc4a63` 首次生效」推断长条来自时域跨体素借历史，并实施了
"命中体素签名"判据（`GI_RES_W_POS`，reservoir 8 → 9 字）。上表两条 A/B（降噪关、复用链关）证明
长条与复用链无关；且该判据会把远场「一个体素面小于一个 texel」的历史接受全部拒掉，重新引入
"静止干净、一运动远处大面起雪花"（见 `gi/screen.wesl` ① 段的说明）⇒ 已完整撤回，`GI_RES_WORDS`
回到 8。

### 10.9 GI 收敛慢：重分配的代理量用了二值 `hist_ok`（2026-09-26）

**现象**：静态机位下 GI 要十几秒才"停"（画面持续在慢慢变干净）。

**算术**：当前默认档（降噪质量 高 cand=8、分帧 ÷8、重分配 强）下，**已收敛像素每帧只发 0.25 条
候选**，而记忆窗被 `m_cap_k × (cand_n0 / cand_n)` 放大到 `32 × 8 = 256` 个样本 ⇒
`256 / 0.25 = 1024` 帧 ≈ **17 s** 才填满。

**根因**：`gi/screen.wesl` 的 `b_err`（重分配）用 `hist_ok` —— **二值**判据（本帧有没有接到历史）——
作代理量：它在第 2 帧就翻真，于是**整段填充期**都按 `lo` 发（强档 = 1/4 速率）。而这个档的**本意**是
"这个像素的时域累积还在不在工作"：对**刚开始累积**（`M` 很小）的像素恰恰该按 `hi` 发。

**改法**：代理量换成 `hist.m`（**窗内已累积样本数**），判据 = 窗填满了没有
（`hist.m >= m_cap_base × cand_n0`，即 `gi_res_cap` 的 `cap`）：没填满 ⇒ `hi`（尽快填满）、填满 ⇒
`lo`（稳态）。

**为什么不花性能**：**一个样本就是一条射线** ⇒ "填满一个 256 样本的窗"无论快慢都要 256 条射线；
本改动只把同一批射线**提前**花掉 ⇒ 稳态每帧射线数、稳态噪声、窗长（帧数）全都不变，代价只是填充期
（≈1 s）的瞬时负载。镜头平移时 `hist.m` 一直被重置（历史被拒）⇒ 照旧走 `hi`，那段行为逐位不变。

### 10.5 本轮用过的临时开关（都已还原，勿留）

`consts::{BENCH_UNFOCUSED, AUTO_ORBIT}`、`scene::setup` 里强制世界名与强制 `CameraMode::Orbit`（验流式
世界）、`residency::LADDER` 阈值（验常驻调度）、`RESID` / `STREAM` 日志临时提到 `info!`。
注意：**`RUST_LOG` 会被 bevy `LogPlugin` 的 filter 覆盖**（实测无效），要取证只能临时提级或改 filter 字段。



## 附：参考

- GigaVoxels, Real-time Voxel-based Library（Guehl & Neyret, GTC 2013）：https://inria.hal.science/hal-00808121/file/S3335_PascalGuehl.pdf
- Building with Bricks: CUDA-based Out-of-Core GigaVoxel Rendering（Crassin, Neyret, Eisemann）：https://icare3d.org/research/publications/CNE09/IntelConf_Final.pdf
- VDB: High-Resolution Sparse Volumes with Dynamic Topology（Museth, TOG 2013）：https://www.museth.org/Ken/OpenVDB_files/Museth_TOG13.pdf
- Compression and Interactive Visualization of Terabyte Scale Volumetric RGBA Data（Derin et al. 2022，每节点 min/avg/max 色）：https://dl.acm.org/doi/fullHtml/10.1145/3532719.3543256
- Parallax Voxel Ray Marcher（Lund University EDAN35 课程报告，2023；本地 PDF，未入库）
