# R3 光影效果

> 施工优先级：同类内保持原编号顺序；新编号为文件内全局顺序号，旧 `Px.y` 编号保留作历史锚点。

- [ ] **~R3-1（3.1a）~** ~~逐面着色风格与「细分致光滑」~~ — **v5 废弃**。架构回退到逐体素 flat shading（per-voxel，对齐 Douglas #22/#23 最新架构），见 README 决策表「光照量化粒度（v5）」。**细分策略本身仍值得保留**（曲面热点 0.25cm 小体素 = 更细腻几何），但不再服务于逐面着色的明暗台阶效果
- [ ] **R3-2（3.4a） ⚠️ 待 R3-10 DDGI 落地后裁决** **密度场 AO（对标 #15）**：Douglas 最终架构无独立 AO pass——AO 由 DDGI 间接光 + 天空环境光承担（docs/douglas-final.md §5）；若 DDGI 落地后缝隙暗部不足再启用本条。原方案：每 cell（或 16³ 粒度）统计体素占据数 → 一张小 3D 纹理（R8/R16）；DDA 命中后按世界坐标采样（硬件 trilinear 单次纹理读），>0.5 部分映射为暗度乘进直光/环境光；**采样点 per-voxel**——采样位置 = 命中点沿命中面法线偏移半体素，作为该体素的 AO 暗度值；密度计数在 tile builder 构建时顺手产出、编辑时局部重算、增量上传管线复用——成本几乎为零（比 SSAO 便宜一个量级）；#15 两条踩坑不必重踩（Minecraft 式逐体素 AO：微体素粒度太细失败；SSAO：随机采样噪声被纯色风格放大成闪烁）；环境光 sky 渐变部分已于 2026-09-01 落地（存档 3.4）
- [ ] **R3-3（3.5）** 后处理：ACES tonemap + sRGB（评估复用 Bevy 后处理链）+ **轻量 bloom**（阈值 + 小半径模糊，发光体/太阳光晕，可开关）
- [ ] **R3-4（3.5b）** **距离雾/大气透视（对标截图 3/5 远山氛围）**：shade_hit 出口按命中 t 指数衰减混向 `sky(视线方向)`；密度/高度衰减参数主题化；近零成本（compute 内数行）
- [ ] **R3-5（3.5c）** **God rays 屏幕空间廉价版（对标截图 5 第一步）**：后处理径向模糊（太阳屏幕位置 CPU 传入；DDA 命中 t/天空遮罩防穿透穿帮；强度/衰减长度可调，可开关）；9.8 介质散射版落地后可退役或保底
- [ ] **~R3-6（3.5d+）~** ~~Phase 2 hashmap 管线~~ — **v5 全部废弃**。逐面着色 + per-voxel face_light hashmap 架构彻底移除（代码已删：face_light.rs + dda.wgsl 四段式 + Rust 端 fl_clear/fl_light/fl_composite）。
- [ ] **R3-7（3.6）** 锁定体素全息视觉变体
- [ ] **R3-8（3.7）** 性能验收：1080p@60fps（GTX 1660）、每帧 1 万体素编辑不掉帧（基准以 1cm 工作区分辨率计）；**极限压测**：满屏 0.25cm 热点 + 最坏阴影射线路径的帧时间上界断言，VRAM ≤2GB 断言；**外部锚点（#17）**：Douglas GTX 1660 Ti 主射线+阴影射线 7ms/帧（未做 marching 循环优化、无 depth prepass，自评 compute bound 仍有余量）——M2 阴影射线预算可行性的同卡证据
- [ ] **~R3-9（9.1a）~** ~~半分辨率路径追踪 pass~~ — **已取消**。被 R3-10 DDGI 取代；radiance emissive 分支保留在 R3-10 probe endpoint
- [ ] **R3-10（9.1b） ⚠️ 优先级提升：** **DDGI 全局光照（1:1 复刻 Douglas #23）** — probe worklist + 固定预算射线 + 上一帧 DDGI 输出自闭环。probe endpoint emissive 分支保留（gate 核心画面：通电 LED 珠照亮周围墙）。
  ```wgsl
  fn probe_endpoint_shade(hit) -> vec3<f32> {
      if hit.sky { return sky(rd); }
      if hit.emissive { return hit.albedo * hit.emissive * GAIN; }  // gate 扩展
      return sample_prev_ddgi(hit.world_pos, hit.normal);
  }
  ```
  composite（v5）：`final = albedo × (direct + ddgi_indirect) + emissive_radiance`（直光与间接光**求和**——间接光含≥1次弹射能量，max 会吞掉颜色渗透；triplanar 纹理属 R6，当前 base = palette albedo）— **无任何 per-voxel 光照缓存**。
  - ✅ 共用底座：DDA/trace_grid()（两级+OBJ）、sky()/shade_hit、StateTable emissive 占位
  - ❌ DDGI 新增：①探针烘焙（cell 16³ BFS 空叶 + 偏移启发式 + 4 级 LOD）②每帧固定预算射线分摊（Fibonacci 球 + worklist atomics + 上一帧 DDGI 着色 → 无限次反弹）③活跃探针剔除（6 邻接 cell bitmap AND）④irradiance + 深度图 → 八面体映射 2D 纹理数组 ⑤漏光治理（深度图，采样时 probe-to-wall < probe-to-point 剔除）⑥着色采样（最近 8 cell 三线性 × 视线可见性 × 前后权重）
  - **参考文献（必读，按序）**：
    1. [Majercik 2019 原论文](https://jcgt.org/published/0008/02/01/paper.pdf) — §3 八面体映射存储（irradiance 8×8 + depth 16×16）、§4 射线更新、§5 视线感知查询、附录 GLSL
    2. [Majercik 2021 生产版](https://jcgt.org/published/0010/02/01/paper-lowres.pdf) — §3 探针状态机（活跃/休眠）、§4 调参、§5 级联体积
    3. [Rohacek 2022 漏光治理](https://old.cescg.org/CESCG-2025/wp-content/uploads/2022/04/Rohacek-Improving-Probes-in-Dynamic-Diffuse-Global-Illumination.pdf) — §3.1 BFS 探针放置、§3.2 锐利背面剔除、§3.3 漏光治理
  - **实现 spike 路径**（论文章节即 checklist）：
    - ✅ 0. 读三篇论文（2026-09-04 完成，PDF 已抓取精读）→ 模块规格并入本条目与 [ddgi.rs](../../gate-render/src/ddgi.rs) 模块头注释
    - ✅ 1. 探针烘焙（Rohacek §3.1）：`gate-voxel` 新增 `BrickState` 三态查询（`ChunkTree::get_brick_state`，cell 16³ = level 2 brick 层级对应锁单测）+ [ddgi.rs](../../gate-render/src/ddgi.rs) `bake_probe_grid`（Air 居中 / Mixed BFS 4³ 空子砖最近中心 → 1³ ±4 盒兜底 / Solid 无探针）；八面体编解码 + Fibonacci 球 + wire 打包 + 帧计划全部落地带单测
    - ✅ 2. 单探针射线更新（Majercik §4）：[dda.wgsl](../../gate-app/assets/shaders/dda.wgsl) `ddgi_update` 入口（workgroup_size=64 = 1 workgroup/探针；射线方向 = 8×8 irradiance oct texel 中心，1 ray ↔ 1 texel；trace_scene 端点着色 sky/emissive 直出/直光 1-bounce → EMA α=0.1 写 irradiance + 同方向 16×16 depth 稀疏写；probe 存 pre-exposure 辐射度，`shade_hit_linear`/`shade_hit` 拆分曝光只在最终合成施加）
    - ✅ 3. 渲染侧接线：[ddgi.rs](../../gate-render/src/ddgi.rs) `DdgiPlugin`（VoxelScene → Extract 一次性烘焙 `ProbeBake` → prepare 建 4 storage buffer + DdgiMeta uniform + BG4；帧号推进 + frame_plan 环形轮转写 meta）+ [dda.rs](../../gate-render/src/brickmap/dda.rs) 双 compute pipeline 共用 5 组布局，ddgi_update pass 先于主 trace dispatch；烘焙稀疏化 = 每 chunk 迭代 18³ cell 区（16³ 本块 + ring ±1），border cell ≤2× 幂等去重，不扫 bbox 全空间
    - ✅ 4. 着色采样（Majercik §5）+ composite 接线：[dda.wgsl](../../gate-app/assets/shaders/dda.wgsl) `sample_ddgi`（最近 8 cell 三线性 × 锐利背面权重 wn=clamp(N·d/0.2,0,1) Rohacek §3.2 × 漏光 chevron wd=clamp((dtex−dist)/4+0.5,0,1) Rohacek §3.3；irradiance 八面体双线性 8×8、depth 最近邻 16×16；加权和按权重和归一化）接进 `shade_hit_linear` 的 `base × sample_ddgi(p,n)` 项——probe 射线端点走同一函数即上一帧 DDGI 自闭环（无限反弹），emissive 端点保持直出不采样。[ddgi.rs](../../gate-render/src/ddgi.rs) CPU 参考 `cpu_sample_ddgi`（WGSL 逐行镜像 + DdgiProbeArrays 同构布局）+ 6 端到端单测（全同色归一化/背面全剔除/chevron 全遮挡/NO_PROBE 与空集早退/oct 上下半球方向色/64³ 封闭房间烘焙分布）锁死数学
    - ✅ 5a. 编辑后探针重烘：`VolumeGrid.edit_generation`（u64 单调计数，实际体素变更 +1、同色 noop/palette 变更不计）→ extract 版本比对触发重烘（ProbeBake 携带代数 + inflight 去重），prepare 重建 4 buffer + irradiance/depth 清零重新 EMA 收敛、帧计数归零
    - [ ] 5b. 目验后调参/精修 + 探针状态机（Production §3）+ LOD 级联（N=10 显存告警再启）+ oct 折缝 warp（若目见解缝明显）
  - **已定决策（2026-09-04，spike 0-4 副产物）**：
    - **gate 探针分配 = 活跃壳**：探针仅存在于「cell 或 6 邻接 cell 含实体」的 cell。远场纯空气探针在活跃门控（Douglas：cell+6 邻接无体素 → 不活跃）下永不更新、数据恒零，分配纯属显存浪费；表面点三线性 8 邻域恰好全部落在壳内，采样完备性不受影响。大开放空间 GI 梯度提升路径 = 放宽活跃半径或 LOD 级联（P14 一并裁决）
    - **GPU 存储 = storage buffer 非纹理**：irradiance rgba32f 8×8（128 words/探针）+ depth r32 16×16（256 words/探针），避开 storage-texture 格式/读写特性坑；八面体采样本就手动双线性，随机访问 buffer 更直接。f16 打包（显存减半）后置
    - **射线预算**：4096 射线/帧 = 64 探针/帧 × 64 射线（1 ray ↔ 1 oct texel），环形轮转 `probe_idx = (cycle_base + wg) % probe_count`，全量刷新一轮 = ceil(probe_count/64) 帧；EMA α=0.1
    - **射线方向 = irradiance 8×8 texel 中心**（1 ray ↔ 1 texel，无 scatter/atomics）；Fibonacci 球函数保留给帧间相位旋转/蓝噪声（R3-14）。depth 16×16 同方向 oct 映射稀疏写，未写 texel 保持 tmax 远距初值（0 会被漏光剔除误读成「探针贴墙」）
    - **曝光只在最终合成施加一次**：probe irradiance 存 pre-exposure 辐射度（端点着色用 `shade_hit_linear`），EMA 跨帧累积不会把曝光平方化
    - **DDGI 已上屏（spike 4 起）**：`shade_hit_linear` 末项 `base × sample_ddgi(p,n)`；probe 端点着色走同一函数 = 上一帧 DDGI 自闭环无限反弹。**ddgi_update 与主 trace 必须分两个 compute pass**（[dda.rs](../../gate-render/src/brickmap/dda.rs) `dispatch_dda`）：ddgi_update 写 ddgi_irr/depth storage、dda_main 读同 buffer，wgpu 仅在 pass 边界自动插 memory barrier，同 pass 内 dispatch 间读写同 storage 是 race（UB）
    - **采样权重 = Rohacek 锐利形态**（非 Majercik 原版软背面）：背面权重 wn=clamp(N·d/0.2,0,1)——N·d<0 严格 0，杜绝薄墙穿透漏光；深度 chevron wd=clamp((dtex−dist)/4+0.5,0,1)（半宽 = cell×0.25）；加权和按 wsum 归一化（全剔除 → 间接光 0）
    - **八面体折缝**：irradiance 双线性边界硬钳（不做折缝 warp 修正），轻微接缝待 spike 5 目验后决定是否精修；depth 取最近邻（保守剔除，不跨 texel 混深度）
    - **烘焙遍历稀疏化**：每 chunk 迭代 18³ cell（本块 + ring ±1），border cell ≤2× 访问幂等去重；不扫 chunk bbox 全空间（N=10 稀疏世界稠密 cell 表 ≈130MB）
    - **三态查询**：`get_uniform` 的 None 同时表示「uniform 空气」与「含空气混合」（语义歧义，勿再借用）——探针类需求一律用 `get_brick_state`
    - **性能基线（2026-09-04 A/B 实测，RTX 4060 Laptop / 1600x900 / N=10 demo 场景 / GATE_TILES 默认）**：全开 ~63fps（冷机，含 vis 缓存 +7% 与 implicit normal）= 主 trace chunk 遍历 12.8ms（65%，全分辨率下的结构成本，非 bug）+ 着色 5.3ms（阴影射线 4.7 + sample_ddgi 1.4）+ 框架/blit/DDGI pass 1.5ms。定位手段 = dda.wgsl 逐层禁用（跳着色 → 跳阴影 → 跳 trace → `GATE_SKIP_CHUNKWALK=1`）+ 同轮运行时开关（V 键）。CPU 仅 0.26 核（非瓶颈）。剩余候选：Beam 保守起步（#18 最大单项）、半分辨率阴影缓冲、f16 打包 irradiance（≈1ms 内，优先级低）
    - **方向位掩码 LUT 已实验并回滚（2026-09-04，Douglas #18 优化 2）**：完整实现过（27 方向符号类 × 64 cell × u64 保守表，30 万随机射线 property test 锁保守性，复用 BG1 binding(1)），实测 51→45fps **负优化**。根因：Douglas 的 LUT 配套「brick 内 4³ DDA 全步进」（一次剪 64 步），gate 的层次 DFS 只步进射线实际穿过的 cell（uniform 跳过已零成本），表面附近下探候选大多真命中 → 剪枝率低而每候选 +5 load 纯亏。**不要重试**，除非将来大空腔稀疏场景实测剪枝率 >50%。另踩坑：剪枝弹栈后不可 continue 外层循环——父帧停在分裂 cell 会再压同一 child → 每像素耗尽 65536 budget → GPU TDR（DeviceLost）。剩余候选优化按收益排序：帧率驱动降分辨率 > 逐体素直光 > Beam 保守起步
    - **逐体素直光已落地（2026-09-04，Douglas #19，用户裁决 1:1 抛弃 (voxel,face) 变体）**：[vis_cache.rs](../../gate-render/src/brickmap/vis_cache.rs) + dda.wgsl BG5。per-voxel 语义 = 每体素从**受光面**（朝太阳面）中心投唯一一条阴影射线，vis 全面共享；实现 = 4M slots × u32 哈希表（slot = tag30<<2|state，0空/1遮挡/2可见；CAS 插入线性探测 ≤4），key = obj_id4bit + 体素局部坐标 3×14bit，**从命中点 floor(p) 取体素坐标（trace 零改动）**；跨帧持久（方向光 vis 视角无关）+ `edit_generation` 整表清零（16MB 一次性 DMA）。`GATE_NO_VIS_CACHE=1` 旁路；**V 键运行时切换**（同轮 A/B 防热节流污染）。**同轮 A/B 实测（15s 自动翻转 3 对段）：OFF 47-49fps vs ON 51-53fps ≈ +7%（~0.8ms）**；收益小于 Douglas 的 1-2ms——他场景近景大面占比高 + PT 时代 hashmap 兼做降噪
    - **per-voxel implicit normal 落地（2026-09-04，Douglas #22 用户目验裁决：原实现是逐面着色非逐体素）**：`shade_hit_linear` 的法线从 DDA face normal（`face_normal_from_index(face_id)`，同体素跨面变明暗 = 逐面）改为 **per-voxel implicit normal**——6 邻域 occupancy 差分（`implicit_normal_local`，sample_brickmap 点查），一体素一法线 → sky 渐变/ndl/DDGI 采样全部体素粒度 → **一体素一色**。face normal 保留作差分零向量退化 + debug 可视化。性能：首算 6 次树点查/voxel → **BG5 第二张表 vis_norm 缓存**（64bit 槽 = tag32 + oct16×2；先 lo 后 hi 写入防撕裂），跨帧持久 + 编辑清零连带；冷机实测 60-62fps（缓存稳态）。已知差异：lighting.rs CPU 镜像 `cpu_reference_shade_hit` 未同步 implicit normal（仍 face normal）——等价测试只锁直光公式骨架，不一致为已知，后续同步
    - **分辨率策略（2026-09-04 用户裁决纠正）**：**默认全分辨率**——Douglas 最终画面 sharp = 独显全速 + FXAA（1660 Ti 7ms 是全速数字）；半分辨率只是他的集显降档路径（#17），勿当主线。实现上 [responsive.rs](../../gate-render/src/responsive.rs) 默认 factor=1，`GATE_RES_SCALE=2` 开降档；[blit.wgsl](../../gate-app/assets/shaders/blit.wgsl) 改 uv 双线性上采样（1:1 时无损，降档时自动生效）。曾短暂默认半分辨率（110-170fps）被用户裁决回滚
    - **unlit 诊断模式（2026-09-04）**：N 键循环加第 4 态（`debug_mode==3` → `debug_mode.w=1`）——跳过全部光照（直光/vis/DDGI/曝光）albedo 直出，测纯 trace+框架帧率上限。注意 w 通道编码：2=sky 4=makegrid 1=unlit，0=正常
    - **Beam 保守起步已实验并回滚（2026-09-04，Douglas #18 优化 3）**：完整实现过（dda.wgsl `beam_main` 半分辨率预 trace 写 `beam_t_max` storage 纹理 → 独立 compute pass 最先 dispatch（BG6 write/read 跨 pass barrier）→ `dda_main` 读 3x3 邻域 min 作主射线 `t_start`；`beam_t_max = render_h/tan(30°)` 保守距离判据），实测 full-lit **63→52fps 负优化**、unlit+beam 56fps，已全部回滚。根因与方向 LUT 同构：beam 省的是 ray-box 高单价步进（Douglas 当年跨树层级/跨 brick 用 ray-box 求交），而 gate 已实现全程层次栈式 DDA（连跨层级都是 tmax 单分量加法），空气跳过 = mask 寄存器 + uniform 整格跳过，零成本；beam pass 自身反而新增全分辨率半量 trace + 纹理读写。**不要重试 LUT/beam 类「减步进单价」优化**——gate 底座上步进已不是瓶颈。教训存档：①wgpu bind group binding 号必须显式声明（`BindGroupEntries::single` 默认 binding 0）；②复用 pipeline 布局的 pass 必须设齐所有 bind group，否则访问违例崩溃（-1073741819）；③storage 资源不能「不写就读」（跳过 dispatch = 读未初始化 = 崩溃）
    - **LOD 远场父节点提前终止已落地（2026-09-04，Douglas devlog #2「八叉树层级遍历早停 = 天然 LOD」）**：远场子节点投影 <1px 时不下钻，整块按子树多数色于子块入口命中。**代表色编码** = 节点 palette word 高字节（低字节 uniform 色语义不变，wire 格式不变）：[chunk_tree.rs](../../gate-voxel/src/chunk_tree.rs) `node_lod()` 序列化时递归计 64 子块**实体多数票（排除空气，子树纯空气 → 0）**`pack_pal_lod(pal, lod)`。排除空气是关键：地形薄表面区域空气体积占多数，含空气计票则 lod 恒 0 永不早停（首版教训，+18%→+73% 的差距全在这）。**GPU 判据**（dda.wgsl `trace_chunk` 下钻点）：子节点边长 `sub = 64>>(level*2)`（64/16/4 fine）投影角 `< 像素角大小 px_ang = 2·tan(FOV_Y/2)/render_h` ⇔ `t_enter > sub·scale/px_ang`（lod.x 由 [dda.rs](../../gate-render/src/brickmap/dda.rs) `DdaViewUniform.lod` 按 RenderScale 每帧算，1600x900 下阈值 ≈ 49.9k/12.5k/3.1k）；先按最小边长 4 预筛（近场射线零额外 load），`lod==0`（子树纯空气）继续下钻。**语义取舍：区域含实体即早停出图 → 剪影/颜色误差 ≤ 子块边长**（触发条件保证亚像素），不再「不漏实体」保守——掠地射线的天空像素可能被近端区域提前截住（地平线带剪影微胖）。`GATE_NO_LOD=1` 关闭（spike A/B 开关）。**实测（冷机前 30s，RTX 4060 / 1600x900 / full-lit GPU span）**：N=2 demo 场景 15.47 vs 15.55ms 中性（远场占比小）；**GATE_TILES=10：16.2ms/60fps vs 25.6ms/34fps = trace -37%、帧率 +73%**——LOD 把大场景 trace 成本压到接近小场景水平。坑存档：①trace_chunk 内 mask_lo/mask_hi/pal_word 三 load 必须保持并发在飞（MLP 重叠延迟），改成「按需条件读」实测 15.5→22.7ms 严重退化，勿再试；②`sample_brickmap` 点查的 palette word 必须 `& 0xFF`（lod 高字节非零会把空气节点判成占用，污染 implicit normal 6 邻域差分）。剩余候选：leaf 压缩 wire v2（1³ 叶 3-word → 1-word，地形树 ~2× 小）、帧率驱动动态分辨率、R3-11 时域重投影（时域升采样形态，买清晰度不买帧率）

- [ ] **R3-11（9.2）** 时域累积 + 相机运动重投影
- [ ] **~R3-12（9.3）~** ~~À-trous 去噪~~ — **已取消**。DDGI 探针多帧采样天然时间平均；可选重开本条目如 DDGI 远距离闪烁实测仍需
- [ ] **R3-13（9.4）** 上采样 + 与 M2 直光合成
- [ ] **R3-14（9.5）** 蓝噪声序列管理
- [ ] **R3-15（9.6）** 实机调优迭代回路（用户在环，仅「暗色实验室」主题）
- [ ] **R3-16（9.7）** 验收：与 bevy_vox_scene 参考图并排对比超越（暗色实验室场景，RTX 3070 1080p60）；**观感对标锚点 = Douglas Dwyer devlog 截图**（图1 发光房间 = PT 自发光照明 + DDGI 颜色渗透同级；图5 黄金时刻 = god rays + 天空 + 雾综合观感；DDGI 探针放置对照 #23 cell 16³ 落地）；**着色粒度 1:1 对齐 Douglas #22/#23**（逐体素 flat shading），验收标准 = 整体观感并排不劣化
- [ ] **R3-17（9.8）** **God rays 介质散射升级（两步走第二步）**：DDGI 探针射线端点内（或单独介质 pass）参与介质积分（太阳方向体积 NEE 采样），物理正确光柱取代/增强 R3-5 屏幕空间版；1660 降档路径 = 关 GI 时回退 R3-5 屏幕空间版
- [ ] **R3-18（3.4 复刻）⚠️ 与 R3-10 合并为「最终光照管线」战役一次到位（2026-09-04 用户指令：不做无 GI 过渡态，直接上 Douglas 最终方案）** **直光层（#02/#17/#23 形态）**：v5 composite 的直光组件——①太阳硬阴影（DDA 命中后 1 条向太阳射线，Douglas 原案）②sky 渐变环境光（按法线 y 混合天顶/地平线，#02）③发光体素 emissive radiance 直出（无方向性、不受阴影）；④太阳方向/sky 参数走数据驱动光照主题；⑤CPU 参考同步（lighting.rs）+ 等价单测。**着色粒度 = 逐体素 flat（v5 决策），不做逐像素着色；阴影射线为 Douglas 原案形态**。实现顺序上先于 DDGI（composite 间接光项从 0 起步由 R3-10 填上），验收与 R3-10 合并进行——合成完整最终管线后才整体验收
