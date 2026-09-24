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

| 停在哪一档 | 节点边长 | 丢弃尺度 | 停的条件 | 1080p 安全距离 | 每 chunk 内存 |
|---|---|---|---|---|---|
| 逐体素（现状） | 1 | — | `fp < 1` | 小于 935 体素 | 1.9 MB（实测） |
| 4³ 块单色 | 4 | 1 | `fp ≥ 1` | 935 体素 | 约 0.5 MB（估） |
| 16³ 块单色 | 16 | 4 | `fp ≥ 4` | 3 740 体素 / 935 m | 约 70 KB（估） |
| 64³ 块单色 | 64 | 16 | `fp ≥ 16` | 14 960 体素 / 3.7 km | 约 1 KB（估） |
| 整 chunk 单色 | 256 | 64 | `fp ≥ 64` | 59 840 体素 / 15 km | 约 0.3 KB（估） |

（距离按 1 体素 = 25 cm 折算。屏幕分辨率翻倍则这些距离翻倍；上采样则缩小。）

### 3.4 视距配档

| 档 | 视距 | 全分辨率半径 | 显存（地形 / 密实） | 相对现状帧成本 | 卡点 |
|---|---|---|---|---|---|
| A | 1.5 km | 4 chunk | 0.16 / 0.51 GB | 约 1.0 倍 | 内存 |
| B | 3–8 km | 4–6 chunk | 0.05 / 0.2 GB | 1.2–1.5 倍 | 内存转带宽 |
| C | 16–30 km | 8 chunk + 多级 proxy | 0.1 / 0.35 GB | 1.5–2.5 倍 | 带宽 / 生产吞吐 |
| D | 100 km 以上 | 同 B/C | 同 B/C | 同 C | 只剩带宽与生产吞吐 |

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
| M4 | ray-guided 请求通道 **〔切片 1 已落地，见 §9〕** | shader 有界 request buffer（chunk key + 所需档位 + 溢出计数）；Rust 异步回读（1–2 帧延迟）+ 合并排序 + 主射线优先于阴影 / GI；关闭时回退启发式半径 | 细节按真实需求分配 | 编辑走优先通道 | shader 与 Rust 各一处；中高 |
| M5 | 生产管线（ChunkSource） | `trait ChunkSource { fn produce(coord, level_range) -> ChunkTree }`：程序化 / `.vox` 区域 / 磁盘按层文件；线程池 + 预算；产出走 `mount_chunk_tree` | 粗到细流式、后台加载 | 编辑以覆盖层叠加，不改生成基线 | 新模块；中 |
| M6 | 真无限 | 环形窗口（`compute_window` 改玩家为中心）+ shader 窗口寻址模运算 + 平移帧条目重写 + 相机相对坐标 | 跨过 16384³ 上限 | 无影响 | shader 寻址 + builder + 坐标；高 |
| M7 | 试验世界 `infinite_cubes`（M4/M5/M6 的最小可用形态）**〔已落地，见 §10〕** | 程序化生成器（`docs/infinite_cubes.md` 规则）+ 逐帧加载 / **真卸载** + 相机跟随窗口（接近边界整块重定）；规矩：只在"读系统文件"那一步换成生成、卸载走真实流程、末端不落盘 | 用真实流水线验证流式闭环 | 卸载即丢本地修改（既定语义） | `gate-app/src/infinite_cubes.rs` + 两处 `VolumeGrid` API + `plan_residency` 反向同步；中 |

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
| 视距 | 1080p 全分辨率半径约 4 chunk；三级 proxy 把视距推到 8–30 km；总显存 50–550 MB | 3.4 节的算术，运行时以 `b_struct` 大小与驻留日志核对 |

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
- **诊断计数器 + 读回**（`trace.wesl::LOD_DIAG` 与 `consts::LOD_DIAG`，**两侧须同时开**）：
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

### M4（切片 1）：ray-guided 请求通道 —— 已落地

- **产生**（`trace.wesl`）：`REQ_ENABLE`（默认 0，关掉时整段折叠）+ `req_push`。发射点 = **chunk 级 DDA
  里"窗口有这个槽位、GPU 上没有树块"**（`entry == 0`）那一步；只报"这个距离还需要细节"的
  （`req_level(fp)` 给出整 chunk 档 ⇒ 连一个平色都够 ⇒ 不发）。请求字 = chunk 相对窗口下标（3×6 位）
  + 所需档位（2 位，与 `residency::want_level` 同一阶梯，两边不分叉）+ 射线类型（1 位，留给消费端排序）。
- **通道**：`bindings::lod_req`（BG1 **@binding(10)**，read_write 环缓冲；10/11 曾是已删的反射缓存
  乒乓，那条"别再往 10 起加东西"的告诫已随之更新）+ `brickmap::consts::{RAY_GUIDED_REQUESTS, REQ_CAP,
  LOD_REQ_WORDS}`：`[0]` 累计条数、`[1]` 溢出计数、其后 `REQ_CAP = 1024` 条（满了覆盖最旧的）。
- **主射线优先**：请求闸门由 `WorldRayQuery::req` 给 —— 主 pass 用 `world_cfg_primary`，阴影 / 反射 /
  GI / 光柱 / 光束一律 `world_cfg_full`（`req = 0`）⇒ 次级射线整段被折叠（不是靠"少发"）。
- **回读**（`profiler::report_lod_requests`）：每 `REPORT_PERIOD_SECS` 同步读回（与 `report_lod_diag`
  同一取舍、同一注册条件），**合并**（同一 chunk 的条数 = 有多少条射线要它、档位取最细）后落一行
  `REQ[新 N 条 → M chunk；最热 (x,y,z)l<档>×<票> …；溢出 K]`（坐标 = 相对下标 + `main_window_origin`）。
- **未做（切片 2）**：① 把请求**并入需求**驱动加载 —— 现在仍走 `infinite_cubes` 的距离半径；
  ② "常驻档位比射线要求粗"这条请求（shader 现在只看得见"没有树块"，看不见"档位不够"—— 那需要在
  窗口条目里编码常驻档位）。两条开关默认关 ⇒ 行为零变化。
- 单测/验证：`cargo test --release -p gate-render wesl`（WESL 编译 + 校验通过）、clippy 无新增告警；
  运行期取证 = 两侧开关同时打开，看 `REQ[...]`。

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
  `ensure_resident_tree`）③ **有变化时重出 `UploadSnapshot`**（`prepare` 只认它）。
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
- 默认阶梯 = §3.3 表（16³ ⇔ `fp ≥ 4`、64³ ⇔ `fp ≥ 16`、整 chunk ⇔ `fp ≥ 64`）⇒ 720p 下 623 m 起才允许
  16³ 档（轮廓外扩 ≤ 4 像素，与叶级档同量级）。**要更保守就把它 ×4**（16³ ⇔ `fp ≥ 16`，2.5 km 起），
  改一个常量。
- 预算默认仍**不限**（`budget_bytes = 0`）⇒ 只按距离降级、不整块丢弃；有限预算留给 M5 的流式策略。
- 记在案的代价：每帧 O(chunk 数) 的记账（10k chunk ≈ 100 µs/帧）—— M5 规模上来时要换空间索引。
- 验证口径：`RESID[...]` 是 **`debug!`**（按日志规则：周期性运行轨迹）。注意 `RUST_LOG` 会被 bevy
  `LogPlugin` 的 filter 覆盖（实测无效）⇒ 要取证就临时把那一行提到 `info!`。

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
| M4 切片 1（请求通道：产生 + 合并读回；消费端未接） | `trace.wesl::{REQ_ENABLE, req_push, req_level}`、`world.wesl::world_cfg_primary`、`bindings::lod_req`（BG1 binding 10）、`brickmap::consts::{RAY_GUIDED_REQUESTS, REQ_CAP}`、`profiler::report_lod_requests` | WESL 编译 + 校验通过、clippy 干净；**运行期未取证**（两侧开关默认关） |
| 断口取证：「CPU 有 / GPU 无」告警 | `upload.rs::plan_residency` 第 ⑦ 段（`RESID[!!CPU 有 / GPU 无 …]`） | 待 10.2 第 1 条的实跑 |

### 10.2 待验证（下轮第一件事）

1. **`infinite_cubes` 的窗口修复**：菜单「游戏/世界」→ 选 `infinite_cubes` → 「重载世界」。预期 = 近处
   ±10m 结构完整、无空洞、无轴对齐断口；超出后整片干净结束。**取证手段已就位**：`plan_residency` 第 ⑦
   段会在"本帧没有待安装动作、相机近旁仍有有内容的 chunk 不在 GPU 上"时报
   `RESID[!!CPU 有 / GPU 无 N chunk（3 chunk 内最近 …）→ 画面上是空洞 / 齐平断口]`（只报数量变化）
   ⇒ 有断口必有这行，不必靠肉眼找。
2. **M1 的视觉验收**（一直没做）：`consts::DDA_LOD` 开关前后各看一次远景，确认不开时逐像素与旧版一致、
   开时只在 935 体素外交替（`LEAF_LOD_FP` 现为 1.0，块内替换误差 ≤ 4 像素）。
3. **M2 的视觉验收**：远景里 `fp ≥ 1` 的块应呈现**块内多数材质**的色（不再是"首个非空体素"那种偶发跳色）；
   重点看混合材质区（草地+土、砖+灰浆）的 LOD 边界是否比 M2 前更稳。`LOD_DIAG` 打开时可顺带确认
   叶级早停数不变、`illegal` 仍为 0。

### 10.3 已知限制（当前架构的硬边界，不是待修 bug）

- **视距 = 加载半径 × 5.12 m**：一个 chunk 是 256³ 体素、树约 1.5 MB，却只有 5.12 m 宽 ⇒ 全分辨率下
  ±10m ≈ 190 MB、±20m ≈ 700 MB。**远场要粗粒度层**（按 16³ 格直接生成 ≈ 20 KB/chunk、量化 32cm），
  否则视距上不去 —— 这是 M5 的一部分，不在本轮范围。
- **窗口是 `b_struct` 索引区的定义域**：相机接近窗口边（< 6 chunk）就整块重定窗口（丢 CPU chunk + 全量
  重传，一次性卡顿）。根治 = M6 环形窗口。
- `Streaming::{load_radius, unload_radius, per_frame}` 现值 **2 / 3 / 1**；卸载半径只比加载大 1（迟滞够用
  且把常驻集真正限住；差 2 会留下 9³ 的尾迹 ≈ 700 MB）。
- `infinite_cubes` 的 PBR 档位已接真资产槽：运行期用 `PbrTextureSet::ids()`、`Startup` 那一块用
  `gate_render::material_ids()`（贴图集是 `Update` 里才插入的资源，见该函数的时序说明）。
- **卸载即丢本地修改**（无覆盖层落盘）—— `infinite_cubes` 的既定语义；要保留就得做 M3 规则 3 的覆盖层。

### 10.4 下一步（建议顺序）

1. 复验 10.2 第 1 条，并定死 `Streaming` 的三个半径 / 帧额。
2. **M2 叶代表色已落地**（§9），只欠 10.2 第 3 条的视觉验收。M2b（上层三档 LOD）仍等公里级视距。
3. **M4**：切片 1（请求通道：产生 + 合并读回）已落地（§9）。切片 2 = 把请求并入需求（替换
   `infinite_cubes` 的距离半径）+ "档位不够"那条请求（要先把常驻档位编进窗口条目）；同时也是 M0
   步数直方图的读回通道（同一套回读骨架）。
4. **M5 生产管线 + 粗粒度层**（`ChunkSource`）：远景按粗格生成，视距才可能从 10m 走到百米级。
5. **M6 环形窗口 + 相机相对坐标**：消掉"接近边界整块重定"的卡顿。

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
