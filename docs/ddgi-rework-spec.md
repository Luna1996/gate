# DDGI 1:1 重做规格（R3-10 战役 v2）

> 生成于 2026-09-06 grill 会话。基准优先级：**Douglas 当前代码（截图实锤）> Devlog #23 视频 > 三篇论文**。
> 本文档 SUPERSEDED r3-lighting.md 中 R3-10 的「已定决策（2026-09-04，spike 0-4 副产物）」全部 6 条。
> 信息来源：#23 字幕逐字 + 8 张截图（4 运行时/图解 + 4 代码：sort.glsl 210-262 行、
> get_probe_flags_intersection/probe_near_surface、collect_radiance）。

---

## 1. 目标与验收线

把 gate 现有 DDGI 实现（R3-10 spike 0-5a）按 Douglas 当前代码严格 1:1 重做。

**验收（三条全过）**：
1. 单测全绿（`cargo test --release`；CPU 镜像固定种子锁新数学 + fuzz 等价门禁）
2. GATE_BENCH 时序不劣于重做前基线（重做前先采基线数据）
3. 用户在现有 demo 场景目验四项：暗部不死黑 / LED 颜色渗透 / 无漏光 / 无明显折缝

**交付方式**：一次到位（核心循环 + 滚动级联同批交付），one-step 迁移。

---

## 2. 已锁定决策（12 条）

### 架构映射
| # | 决策 | 证据 |
|---|------|------|
| D1 | 探针网格 = 世界级全 cell 覆盖（16³ cell = level 2 brick，含纯空气 cell），**废除活跃壳** | 截图1：全空间黄点铺满；字节码 per-model 的世界级推广 |
| D2 | 活跃判定每帧 GPU，三条件 OR：本 cell 有表面 ∨ 6 邻接 flags 交集有表面 ∨ OBJ bbox 桶重叠；shared memory 3D halo 数组（local 域外扩 1 圈）+ memoryBarrierShared + barrier | 截图3：`probe_near_surface` 逐字 |
| D3 | 烘焙算法零改动：Air 居中 / Mixed BFS 4³ 空子砖靠中心 / Solid 无探针 | 截图3（漫画图解）与现有 `bake_probe_grid` 逐条吻合 |

### 每帧三段式管线
| # | 决策 | 证据 |
|---|------|------|
| D4 | ①活跃判定 pass：逐 LOD dispatch（`outside_lower_grid()` 空间划分——本 LOD 只管更细网格覆盖外）；产出 worklist（subgroupExclusiveAdd + subgroupBroadcastFirst 分配）、indirect dispatch 参数、探针元数据 next 写入 | 截图1/2：sort.glsl；RenderDoc `vkCmdDispatchIndirect` |
| D5 | ②射线投射 pass：`dispatch_workgroups_indirect` 消费 worklist；**4096 射线/帧固定总预算**分摊活跃探针（起步值，bench 后调）；球面均匀随机方向（PCG hash + 帧号种子，Fibonacci 球保留）；端点 = sky / emissive 直出 / 直光 1-bounce + 上一帧 DDGI（自闭环无限反弹） | 字幕 + RenderDoc；预算值无实锤取现值 |
| D6 | ③irradiance 投影：`irr = π·Σ max(0, dot(d, dir_i))·L_i / Σ max(0, dot(d, dir_i))`（对 texel 方向遍历本帧全部射线样本），**全程 f16**（`SHADER_F16` feature + rgba16f）；射线数与 texel 数解耦，**废除 1ray↔1texel** | 截图4：`collect_radiance` 逐字 |

### 存储
| # | 决策 | 证据 |
|---|------|------|
| D7 | 载体 = 2D 纹理数组（**废除 storage buffer**）：irradiance rgba16f 8×8 oct texel/探针/层；depth 独立 image 资源（16×16 oct）；探针元数据（packed offset+age）= **双缓冲纹理 ping-pong**（previous 读 / next 写），废除 positions/cell_index buffer | RenderDoc：`2D Array Image 252` + `CS RW 0 → depth_target` + `previous/next_ddgi_probes` texelFetch/imageStore |
| D8 | depth 语义保留：未写 texel = tmax 远距初值（0 会被 chevron 误判贴墙）；depth 稀疏写 + chevron 剔除权重不变 | 现有已验证资产 |

### LOD 级联
| # | 决策 | 证据 |
|---|------|------|
| D9 | 4 级相机滚动级联（Majercik 2021 §5）：级数/每级 cell 数/覆盖半径/级间混合带宽全部抄论文默认；base 世界级 16³ cell 为最细级；采样按相机距离选级 + 过渡带混合（base 全覆盖 → 粗级服务远场） | 截图1 远场稀疏实证；用户拍板（推翻「固定下采样」推荐） |
| D10 | 滚动数据继承 = **age + reuse bounds**：age∈[0,255] 随更新 +1；`reusable = reuse_min_bound ≤ cell < reuse_max_bound && offset 未变` → 继承 age，否则 age=0 重新收敛（EMA 历史随纹理 ping-pong 继承，无条带重烘） | 截图2：250-256 行逐字 |
| D11 | 探针状态机（Production §3 生命周期）不纳入——age/can_skip 已覆盖其职能 | 用户裁决 |

### 范围外
| # | 排除项 | 去向 |
|---|--------|------|
| D12 | OBJ 自身探针、电介质/金属/镜面材质、god rays 介质散射 | P14 / R6 / R3-17 |

### 推断实现（非 1:1 实锤，代码中标注）
- `can_skip_update(cell, cell_center, age)` 具体逻辑：截图/示意图均未覆盖 → 按 Majercik 2021 探针更新分摊逻辑保守实现（age 越大跳过概率越高，随机 hash 驱动）
- oct 分辨率：论文默认 8×8 起步；Douglas 2025 版疑似 16×16（RenderDoc 目测）→ 目验 irradiance 模糊再升
- `irradiance = π·result/result.w` 之后的 EMA/tonemap/blowup/迟滞行未截到 → 按 Majercik 2019 §4 全套补齐（已锁定）

---

## 3. 被推翻的旧决策（删除清单）

| 旧决策（2026-09-04） | 替代 |
|----------------------|------|
| 烘焙时活跃壳（只给壳 cell 分配探针） | D1 全 cell + D2 每帧判定 |
| 4096/帧环形轮转 64 探针/帧 | D5 固定预算分摊活跃探针（worklist） |
| 1 ray ↔ 1 texel 确定性写 | D6 随机射线 + 投影累积 |
| storage buffer 载体（rgba32f/r32） | D7 纹理数组（rgba16f + depth 独立） |
| LOD 级联推迟 P14 | D9 本战役一次到位 |
| 「条带重烘」滚动迁移设想（grill 中间方案） | D10 age+reuse bounds 继承 |

**连带删除**：`RAYS_PER_PROBE=64` 常量语义、活跃壳逻辑、环形轮转 frame_plan、positions/cell_index buffer、
对应旧单测；`probe_idx = (cycle_base + wg) % probe_count` 全链路。

**保留资产（已验证 1:1，零改动）**：BFS 放置算法、oct 编解码、chevron/锐利背面采样权重、
端点自闭环、pre-exposure 约定、edit_generation 重烘触发机制（语义改为 base 重烘 + age 归零）。

---

## 4. Milestone + Sub-task

> 执行纪律：改 dda.wgsl 必须走 `gate-wgsl-shader-optimization` 闭环
> （CPU 参考先行 → cargo test fuzz 等价门禁 → WGSL 逐字镜像 → GATE_BENCH 同会话交替 A/B）。
> 所有测试 `cargo test --release`；wgsl 一律用 Edit 工具改（禁 PowerShell，BOM 坑）；
> 同文件编辑串行；`@workgroup_size` 与 Rust dispatch 严格一致。

### M1：规格与 CPU 参考底座
| ID | sub-task | 验收标准 | 优先级 | 依赖 |
|----|---------|---------|--------|------|
| M1-1 | 采重做前 GATE_BENCH 基线数据（当前 spike 0-5a 实现） | logs/ 留存基线帧时（同场景同相机路径），M5-3 对比用 | P0 | — |
| M1-2 | ddgi.rs 常量与 wire 契约重定义：纹理数组布局（每层探针数、层序）、packed offset+age 位域（bitfieldInsert 镜像 DDGI_LOG2_PROBE_SPACING）、age/reuse 常量、ray budget、级联常量占位 | `wire_constants` 单测更新锁死；与 WGSL 侧逐字对齐表完成 | P0 | — |
| M1-3 | CPU 参考实现（skill 闭环第一步）：PCG hash（固定种子）、球面均匀方向生成、collect_radiance 投影公式、EMA+tonemap/blowup/迟滞（Majercik 2019 §4）、can_skip_update（推断实现，标注）、reuse bounds 判定、age 传递 | 每个函数有独立单测；已知输入→期望输出锁死；推断实现带 `// INFERENCE:` 注释 | P0 | M1-2 |

### M1 基线数据（M1-1，2026-09-06 已采）
- 环境：RTX 3070（Vulkan，driver 610.88）/ nuke.vox（instances=1802，written=31,720,800）/ GATE_BENCH=1 隐藏窗口 + Fifo vsync / 相机静止在初始位姿（轨道相机无输入，天然可复现）
- 稳态（t≥30s，3312 帧）：wall_ms p50=16.663 / p95=16.875 / p99=17.240（vsync 60Hz ±1%）
- GPU：frame_gpu_ms p50=0.662 / p95=0.681；trace_ms p50=0.649 / p95=0.668
- **注意**：frame_gpu = trace+blit 之和；旧 DDGI/direct/GI pass 已在 spike 0-5a 全拆（dda.rs:2349），日志 ddgi/direct/gi 列 -1 属预期而非采集故障
- 留存：`gate-app/logs/baseline-pre-ddgi-rework-20260906/`（gpu_frame.log / frame_time.log / fps.log / latest.log）；`*.log` 被 gitignore，跨机协作需 `git add -f` 该目录
- 构建状态：db6b52e（工作区含 ddgi.rs M1-2 常量改动，仅 CPU 侧，不影响渲染时序）

### M2：探针烘焙重做（CPU 侧）
| ID | sub-task | 验收标准 | 优先级 | 依赖 |
|----|---------|---------|--------|------|
| M2-1 | `bake_probe_grid` 扩展：全 cell 覆盖（纯空气 cell 也放探针居中），删活跃壳筛选 | 64³ 封闭房间单测：探针数 = cell 总数 − Solid cell 数；空气 cell 居中 | P0 | M1-2 |
| M2-2 | 探针元数据双缓冲纹理上传：packed offset+age(u8) 布局、previous/next 两份、初烘 age=0 | 上传/回读 roundtrip 单测 | P0 | M2-1 |
| M2-3 | 4 级级联烘焙：每级 cell 尺寸 BFS 推广（32³/64³/128³/256³ cell，抄论文默认）、`outside_lower_grid` 空间划分掩码 | 各级 64³ 房间分布单测；级间嵌套关系锁单测 | P0 | M2-1 |

### M3：每帧 GPU 管线（WGSL 逐字镜像）
| ID | sub-task | 验收标准 | 优先级 | 依赖 |
|----|---------|---------|--------|------|
| M3-1 | `ddgi_active` 判定 shader：shared halo 数组 + 三条件 OR（含 object_buckets）+ subgroup worklist 分配 + indirect 参数写入 + next 元数据 imageStore + outside_lower_grid | fuzz 等价门禁 vs CPU 镜像（固定种子）通过；subgroup feature 缺失时 fallback atomicAdd（标注偏差） | P0 | M1-3, M2-3 |
| M3-2 | `ddgi_cast` 射线投射 shader：dispatch_indirect 消费 worklist、PCG 随机方向、端点着色（sky/emissive/直光+prev DDGI 自闭环）、directions/radiances 样本缓冲写入 | CPU 镜像同输入射线序列 → 样本缓冲逐位一致；4096 预算分摊逻辑单测 | P0 | M3-1 |
| M3-3 | `ddgi_update` 投影 shader：collect_radiance f16 逐字镜像、EMA+tonemap/blowup/迟滞、can_skip_update、depth 稀疏写（tmax 初值语义） | fuzz 等价门禁 vs M1-3；`π·Σ/Σw` 公式逐字比对测试 | P0 | M3-2 |
| M3-4 | `sample_ddgi` 采样重写：纹理数组访问、双缓冲读 previous、距离选级 + 过渡带混合、age 无关性验证 | 6 组既有端到端单测迁移全绿（全同色归一化/背面剔除/chevron 遮挡/NO_PROBE 早退/oct 方向色/64³ 房间） | P0 | M3-3, M2-2 |

### M4：渲染编排接线（gate-render）
| ID | sub-task | 验收标准 | 优先级 | 依赖 |
|----|---------|---------|--------|------|
| M4-1 | DdgiPlugin 重构：rgba16f/depth 纹理数组创建、双缓冲 ping-pong、BG4 布局重排、`SHADER_F16`+subgroup features 启用与探测 | 空跑管线创建成功；feature 不支持时的报错信息明确 | P0 | M2-2, M3-4 |
| M4-2 | 三 pass 编排：active → cast → update 全部先于主 trace dispatch；indirect dispatch workgroup_size 与 shader 一致 | RenderDoc/日志确认 pass 序；间接 dispatch 数与活跃探针数一致（日志抽验） | P0 | M3-1..3 |
| M4-3 | 级联滚动 CPU 侧：相机 → 每 LOD volume 原点（cell 对齐步进）、reuse bounds 计算、per-LOD dispatch 参数 | bounds 移动单测：相机平移后旧区域 age 继承、新区域归零 | P0 | M2-3, M3-1 |
| M4-4 | 编辑响应重接：edit_generation → base 重烘 + 全级 age 归零（重新收敛） | 编辑后探针分布更新单测；收敛行为日志验证 | P1 | M4-3 |

### M5：清理与验收
| ID | sub-task | 验收标准 | 优先级 | 依赖 |
|----|---------|---------|--------|------|
| M5-1 | 旧代码/旧单测 one-step 删除（§3 清单全项） | `rg -i "活跃壳\|cycle_base\|RAYS_PER_PROBE"` 无残留引用；cargo check 通过 | P0 | M4 全部 |
| M5-2 | 全量测试 | `cargo test --release` 全绿 | P0 | M5-1 |
| M5-3 | GATE_BENCH A/B 对比 | 帧时不劣于 M1-1 基线；4096 预算调参（记录最终值）；劣化则定位回修 | P0 | M5-2 |
| M5-4 | demo 场景目验引导 | 给用户四项检查清单（暗部/颜色渗透/漏光/折缝）+ 观感锚点（Douglas 图4：暗部有间接细节、草地反弹墙脚） | P0 | M5-3 |
| M5-5 | 文档收尾：r3-lighting.md R3-10 更新（本 spec 链接、旧决策标 SUPERSEDED）、ddgi.rs 模块头注释同步、项目记忆更新 | 文档与实现一致；无过期描述 | P1 | M5-4 通过 |

### 并行分支
- M2 与 M1-3 可并行（M2 只依赖 M1-2）
- M3-1/M3-2/M3-3 串行（样本缓冲依赖链），M3-4 依赖 M3-3
- M4-3 可与 M3-3 并行启动（CPU 侧纯计算）

---

## 5. 风险预判与 fallback

| 风险 | 影响 | fallback | 定位阶段 |
|------|------|----------|---------|
| wgpu 29 subgroup 原语后端支持差异 | M3-1 worklist 分配 | per-invocation `atomicAdd`（记录为已知偏差，性能差异小） | M3-1 |
| rgba16f storage write / depth 格式精度（f16 在 t_max=8192 处间隔 >1 世界单位，chevron 半宽=4） | M4-1 | depth 改 r32f + `textureLoad`（最近邻本就无需 filter）或相对深度归一化 f16——**开放点 D-Open1，实现时 A/B 定** | M4-1 |
| 三 pass + 判定开销 > 环形轮转（每帧新增活跃判定全 cell 扫描） | M5-3 bench | 判定 pass 每 cell 1 线程（非 1 workgroup）降开销；预算/级联参数调优 | M5-3 |
| indirect dispatch 与 `@workgroup_size` 不一致 → 部分屏幕 trace | M4-2 | dispatch 参数单测 + 日志抽验（既有坑：仅左上 1/4 屏） | M4-2 |
| 级联滚动边界闪烁（reuse bounds 算错） | M5-4 目验 | bounds 单测先行（M4-3）；目验异常时 dump age 图排查 | M5-4 |
| SHADER_F16 精度不足（irradiance 累积） | M3-3 | 累积步骤升 f32（result 用 f32 vec4，仅存储 f16）——偏离截图但保数值安全，标注 | M3-3/M5-4 |
| base 世界级全 cell 探针数失控（nuke 体素 AABB ≈ 121×39×71 cell ≈ 335k 探针，irr+depth ≈ 385MB） | M2-3 显存 | base 烘焙范围钳到体素 AABB（与 D1 不冲突：D1 反对的是活跃壳，不是世界包围盒裁剪；表面 cell 均在 AABB 内，采样完备性不受影响）；仍超则 base cell 升 32 | M2-3 |

---

## 6. Checklist（执行时勾选）

- [x] M1-1 基线数据采集（数据见 §4「M1 基线数据」）
- [x] M1-2 常量/wire 契约 + 单测（级联占位 + WGSL 对齐表在 ddgi.rs 模块头；顺修 probe_in_layer 层内回绕 bug）
- [x] M1-3 CPU 参考 + 单测（PCG hash/球面均匀方向/collect_radiance/更新链/age+reuse/can_skip 推断；更新链对照 RTXGI ProbeBlendingCS.hlsl L508-550 逐字修正：亮度钳制不受 prev 全黑豁免、暗化保底用 sign(lerp_delta)；81 测试全绿）
- [x] M2-1 全 cell 烘焙 + 单测（域 = chunk bbox cell 域去 ±1 ring；Air/Mixed 全覆盖、Solid 跳过、删 neighbor_has_solid；挖空 chunk 不回收 → 全空气覆盖语义，运行时 ddgi_active 剔除；81 测试全绿）
- [x] M2-2 元数据双缓冲纹理 + roundtrip（build_meta_texture_data 层主序 16×16/层、1 texel=1 packed meta、初烘 age=0；顺修 unpack_probe_meta 保留位掩码；82 测试全绿）
- [x] M2-3 级联烘焙 + outside_lower_grid（cell_state_at 三态分类（32/128 走 2×2×2 子 cell 递归合成）+ probe_position_sized 推广 BFS + bake_cascade_grid + outside_lower_grid 半开区间划分；顺修 probe_leaf_sized 真实空叶尺寸（Air=cell_size / 4³ 空砖=4 / 1³ 兜底=1）使「空叶大者优先」真正生效——原 DDGI_CELL.min(half) 压平会让贴墙 4³ 叶探针凭距离压过 16³ 空叶探针；84 测试全绿）
- [x] M3-1 ddgi_active（CPU 镜像 + 10 单测 done，commit `2551233`；WGSL 待 M3 三 pass 同批镜像）
- [x] M3-2 ddgi_cast（CPU 镜像 + 6 单测 done，commit `ed1fbeb`；含 RTXGI 实锤的采样方向修正）
- [x] M3-3 ddgi_update（CPU 镜像 + f16 门禁 + 6 单测 done，commit `5fc032f`；WGSL 待同批镜像）
- [ ] M3-4 sample_ddgi + 6 组单测迁移
- [ ] M4-1 资源/features/BG4
- [ ] M4-2 三 pass 编排 + indirect 一致性
- [ ] M4-3 级联滚动 + reuse bounds
- [ ] M4-4 编辑响应重接
- [ ] M5-1 旧代码清除
- [ ] M5-2 全量测试绿
- [ ] M5-3 bench A/B（不劣化）
- [ ] M5-4 目验四项通过
- [ ] M5-5 文档/记忆收尾

---

## 7. 目验观感锚点（Douglas #23 图4）

石墙房间内侧暗部有间接光细节；草地绿色反弹到墙脚（颜色渗透）；木地板人字纹清晰
（镜面反射属 R6 不验收）；无墙面漏光、无明显八面体折缝。

---

## 8. 交接进度（handoff，2026-09-07）

> 跨机器接续用。本机 IDE 记忆/偏好不随仓库走，本章自包含。

**进度快照**：M1、M2 全部 done（`abf1caa`）；M3-1/M3-2/M3-3 **CPU 镜像 + 单测全 done**
（`2551233` active / `ed1fbeb` cast+采样修正 / `5fc032f` update+f16 门禁，master 本地）；
WGSL 三 pass 均未写。测试基线：`cargo test -p gate-render --release` = 106+7+1 全绿，
其中 ddgi 单测 52 个。

**M3-1/2/3 CPU 镜像已落地**（ddgi.rs「M3-x」章节，可作 WGSL 镜像源）：
- M3-1：`DdgiProbeFlags`（ENABLED/NO_SURFACES 位标志）+ `brickstate_to_flags` +
  `probe_flags_intersection`（6 邻接 AND，边界外视为自身 flags）+ `probe_near_surface_flags`
  （三条件 OR）+ `cpu_ddgi_active`（worklist 带 age、next_meta 从 prev_meta 拷贝、
  age 生命周期 active 内闭环：scroll→can_skip→increment）+ `compute_cell_flags`。
  逐字对照 sort.glsl L218-262（截图1-3）。
- M3-2：`cast_rays_per_probe`（4096 预算均摊，保底 1）+ `cast_ray_dir`
  （Fibonacci 球 × PCG 随机旋转——旋转种子 = ray_rand(probe,frame,0)，帧内保
  Fibonacci 蓝噪声结构）+ `cast_endpoint_radiance`（miss→sky/PROBE_T_MAX；
  emissive→albedo×emissive×gain 直出；常规→直光 1-bounce + prev DDGI 自闭环，
  间接采样点偏移 DDGI_NORMAL_BIAS，pre-exposure）+ `cpu_ddgi_cast`
  （样本缓冲 slot×rays+i 布局，逐位确定性）。
- M3-3：`f32_to_f16`/`f16_to_f32`/`f16_round`（rgba16f 存储语义 CPU 镜像，
  round-to-nearest-even 全分支正确，位级黄金值锁死）+ `collect_radiance_ex`
  （带 Σw，区分「零覆盖→保留 prev」与「投影为 0→正常暗化」）+ `update_depth_texel`
  （余弦加权均值 + EMA，`DDGI_DEPTH_ALPHA`=0.2 INFERENCE）+ `cpu_ddgi_update`
  （投影→M1-3 更新链→f16_round 写回；读侧 f16 舍入模拟纹理值；零覆盖 texel 稀疏
  保留 prev——`DDGI_TEXEL_MIN_WEIGHT`=1e-4 INFERENCE；非 worklist 探针 prev 拷贝
  = ping-pong 语义；@workgroup_size(64)=8×8 irr texel，depth 256=线程内 4 轮）。
- **采样方向修正（RTXGI Irradiance.hlsl 实锤，旧代码两处反向）**：
  ① 背面权重 = clamp(n·(接收点→探针)/DDGI_NORMAL_BIAS)——探针在法线前侧（空气侧）
  通过；② irradiance oct 采样方向 = **表面法线 n**（oct 图按「接收法线」索引，与 D6
  collect_radiance 的 (d·dir_i)+ 余弦权重同源）；③ depth 方向 = 探针→接收点（正确未动）。
  旧 GI pass 从未目验（spike 0-5a 已拆），M3-2 真实几何端点测试首次暴露。

**下一步 = 三 pass WGSL 同批逐字镜像**（ddgi_active / ddgi_cast / ddgi_update 三个
新 shader 文件或 dda.wgsl 内三 entry，共享 D7 纹理数组绑定 + BG4 重排）：
1. 每个函数组在 WGSL 侧对照 ddgi.rs 对应 CPU 镜像逐字翻译；常量对齐模块头
   「WGSL 对齐表」（含新增 DDGI_DEPTH_ALPHA / DDGI_TEXEL_MIN_WEIGHT），
   `wire_constants`/`v2_wire_constants` 单测防漂移；wgsl_compile.rs 注册新 shader
   做 naga parse+validate 门禁。
2. M3-1 subgroup worklist 分配（subgroupExclusiveAdd + subgroupBroadcastFirst），
   不可用时 fallback `atomicAdd`（风险表已记，标注偏差）；M3-2 需要 trace_grid +
   sky/emissive/直光着色 + prev DDGI 采样（dda.wgsl 现有资产复用）；M3-3 需要
   SHADER_F16 feature + rgba16f storage write（M4-1 资源就绪后才能真跑，可先写好
   过 naga 门禁）。
3. **WGSL 现状**：无独立 ddgi shader，仅 `gate-app/assets/shaders/dda.wgsl`（trace+unlit
   直出，旧 vis_table/direct/gi/denoise 光照链已注释拆除）与 `blit.wgsl`。
4. **插件壳现状**：旧 `DdgiPlugin`（ddgi.rs）仍是 **storage buffer 载体**
   （DdgiGpu：positions/cell_index/irradiance/depth buffer + meta uniform + BG4），
   只做烘焙上传、无渲染 pass，保持可编译。M4-1 重构为 rgba16f/r32 纹理数组 + 元数据
   双缓冲 ping-pong + BG4 重排 + SHADER_F16/subgroup features；M5-1 才 one-step 删旧链路。
5. M3-4（sample_ddgi 迁移）依赖 WGSL 侧纹理数组就绪，与 M4-1 同批做。

**环境与命令（Windows / PowerShell）**
- 跑测试（一律 release）：`cargo test -p gate-render --release ddgi`
- 跑 bench：`$env:GATE_BENCH='1'` 后启动 gate-app（后台静默、vsync、写日志）；
  日志在 `gate-app/logs/`（`frame_time.log` wall 帧时、`gpu_frame.log` 逐帧 GPU pass、
  `latest.log` 应用日志）。M1-1 基线留存 `gate-app/logs/baseline-pre-ddgi-rework-20260906/`
  （RTX 3070 / nuke.vox：wall p50=16.663ms、frame_gpu p50=0.662ms），M5-3 对比用。
- **禁止自行截图验证渲染**；允许并推荐跑 gate-app 后读 `gate-app/logs/` 验证行为。

**跨机器必知的坑（本机项目记忆，仓库外）**
- 改 `.wgsl` 一律用编辑器/Edit 工具，**禁用 PowerShell `Set-Content`/`-replace`**：
  会加 UTF-8 BOM → naga `expected global item found "\u{feff}"`，shader 整个加载失败、
  画面全黑且无弹窗（仅 latest.log 有 ERROR）。
- shader `@workgroup_size` 与 Rust dispatch（gx/gy/workgroup 数）必须严格一致：
  只改 shader 不改 dispatch → 仅左上 1/4 屏被 trace 且 GPU 时长虚假降低 4×。
- shader 优化后必须先验证渲染正确再采信 bench 时序：射线集体假 miss 提前终止会让 trace
  时序「大幅优化」而画面全黑。
- PowerShell 不支持 `&&` 和 bash heredoc；git 用 `git -C <path>`；多段 commit message
  用多个 `-m`；`2>&1` 的 CLIXML 包装会吞 Bevy 日志尾部，重定向用
  `| Out-File run.log -Encoding utf8`。
- 沙箱拦截 `target/debug/incremental`（已设 `[profile.dev] incremental=false` 规避）；
  `Start-Process -WindowStyle Hidden` 会使 winit 窗口句柄失效。
- wgpu 29.0.4 已知未修 Vulkan bug：初始帧 VUID validation 报错、偶发 DeviceLost
  （GATE_BENCH 后台节流低复现），非 gate 代码问题。

**参考资料**
- `docs/douglas/23_devlog23_ddgi.md`：Devlog #23 大纲（transcripts 字幕文件为 0 字节空文件）。
- RTXGI 源码对照：`ProbeBlendingCS.hlsl` L508-550（更新链 EMA/tonemap/迟滞，M1-3 已逐字对齐）。
- `docs/todo/r3-lighting.md`：旧 R3 光照规划，部分决策已被本 spec §3 推翻（SUPERSEDED）。
- 开放点 **D-Open1**：depth 纹理 f16 在 t_max=8192 处精度不足，M4-1 做 r32f vs f16 A/B 定夺。
