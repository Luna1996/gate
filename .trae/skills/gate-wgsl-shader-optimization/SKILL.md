---
name: "gate-wgsl-shader-optimization"
description: "gate 体素光追引擎（Rust+Bevy+wgpu）WGSL compute shader 优化闭环：CPU 参考实现先行→cargo test fuzz 等价门禁→WGSL 逐字镜像→GATE_BENCH 同会话交替 A/B。修改/优化 dda.wgsl 或 brickmap DDA 遍历算法时必须遵循。"
---

# gate WGSL Shader 优化工作流

适用于 `gate-app/assets/shaders/*.wgsl`（核心是 `dda.wgsl` 的 chunk 内层次遍历 `trace_chunk`）
与 `gate-render/src/brickmap/dda.rs`（CPU 参考实现）。用户偏好：**CPU 参考实现先行 +
cargo test 等价性门禁，禁止自行截图验证画面正确性**（允许读 `gate-app/logs/` 日志；画面
由用户目测确认）。

## 铁律：CPU 先行，WGSL 逐字镜像

1. 任何遍历/相交算法改动，先改 `gate-render/src/brickmap/dda.rs` 的 `trace_chunk_cpu`
   （以及它的暴力逐体素 oracle 对照）。
2. 测试门禁（必须全过才算改完）：
   ```
   cargo test -p gate-render --lib brickmap::dda::
   ```
   关键测试：`tree_traversal_equivalence_300_rays`、`tree_equivalence_demo_like_edit_compact`、
   `tree_traversal_fuzz_2000_rays_multiscale`（随机块尺度 4/16/64、跨 chunk、轴/对角退化射线；
   palette 严格一致、t 容差 1.0、命中面法线反向）。
3. WGSL `trace_chunk` 与 `trace_chunk_cpu` **控制流逐字对应**。例外（WGSL-only，需在注释里
   写明 "WGSL-only 近似，CPU 镜像不含"）：方向可达掩码 LUT（b_leaves）、beam depth_cap
   ——这些是 GPU 近似开关，CPU 测试作为它们关闭时的精确 oracle。
   **禁止恢复 split 子节点远场多数色 LOD 早停**（曾两次导致拉远穿墙+逐面着色，
   2026-09-06 二次移除）：子树多数色≠表面色 → 内部色渗出穿墙；命中点落子块入口
   空气体素 → 隐式法线退化回退面法线。LOD 早停只允许 uniform 节点（c_mask==0
   快路径即精确 palette）。正确 LOD 走 R1-12 子树折叠（预计算粗体素），不做多数色近似。

## WGSL 编辑方式（naga 坑，已踩过）

- **只能用 Edit 工具改 .wgsl**。PowerShell/重定向写文件会加 BOM → naga 编译失败。
- dda.wgsl 是**混合行尾**（CRLF+LF），大块 old_string 常匹配失败。对策：小范围唯一锚点
  分段 Edit；确需整段替换时，按行号用 .NET 拼接：
  ```powershell
  $enc = New-Object System.Text.UTF8Encoding($false)  # 无 BOM
  [System.IO.File]::WriteAllText($p, [string]::Join("`n", $lines) + "`n", $enc)
  ```
- 移位 RHS 必须 u32：`v >> vec3<u32>(level*2u)`、标量 `v[mn] >> log2`（log2 声明 u32）；
  i32 移位 RHS 报 "automatic conversions cannot convert vec3<i32> to u32"。
- `countOneBits` 不要作用于 u64：拆 `countOneBits(u32(x)) + countOneBits(u32(x >> 32u))`。
- **没有 `u32(bool)` 转换**：用 `select(0u, 1u, cond)`。
- i32↔u32 位环绕用 `bitcast<u32>` / `bitcast<i32>`（firstTrailingBit 公式需要）。
- `firstTrailingBit` 返回 i32，0 → -1（全零），特判跨出 chunk。
- `cargo build` **不会**发现 shader 错误：naga 在运行时编译，错误只在
  `gate-app/logs/latest.log`（搜 `ERROR` / `pipeline_cache`）。改完 shader 必须跑一次看日志。
- **wgsl-analyzer 误报 u64**：`u64`/`i64` 对应可选的 shader-int64 能力，naga 在支持的
  GPU（Vulkan shaderInt64）上正常编译，但 wgsl-analyzer 0.12.224 及更早版本的类型检查器
  不认识 u64（2026-03 PR #456 才加，配置项 `extensions.shaderInt64`），会报一堆
  "u64 不存在" 的假错误——不影响构建，以 latest.log 为准。解决：升级扩展，或
  `"wgsl-analyzer.diagnostics.typeErrors": "off"`（新版键名 `semanticErrors`）。

## wire 格式速查（语义改错=穿墙/漏命中）

- gate 4³ 分裂树 level：3=根(256³,子块64³)、2(64³,子块16³)、1(16³,子块4³)、0=叶(4³,inline 1³)。
  子块边长 `s = 1 << (level*2)`。
- mask bit=1 = **子块分裂**（≠含实体）；bit=0 = 统一子块，色=节点 palette 字段
  （`b_struct[addr+2] & 0xFF`，0=空气）。child 定位 `popcount(mask & (bit-1))`，
  `child_addr = chunk_base + b_struct[addr+3+pop]`。
- **分裂位可能指向 3 字统一节点（mask=0），任意层都可能**；下钻读到 child 后必须
  `c_mask==0` 快路径 pal 直决/break；叶层统一节点**无 inline 16 字**，禁读 addr+3+。
- 仅 mask!=0 的 level-0 节点有 inline palette：
  `(b_struct[addr+3+(idx>>2)] >> ((idx&3)*8)) & 0xFF`。inline 叶节点的 mask bit=1 = 该
  体素非空——空气体素（bit=0）先查 mask 再决定是否 load inline word（热路径省 load）。
- palette word 打包：低字节=uniform 子块色，高字节=(w>>8)&0xFF = `node_lod` 子树多数色
  （wire 格式保留、builder 仍写入，但 **shader 禁用于远场早停**——色渗出穿墙，见铁律 3）。
- LUT（b_leaves）布局：`lut_base = min(oct*128 + entry_i*2, 1022)`，oct bit0=x正/bit1=y正/bit2=z正
  （零分量按正），entry `z*16+y*4+x`；仅 `pal==0` 空气节点可用 `mask & reach` 剔除。
  `lut_disable = view_u.lod.w > 0.5`（GATE_NO_LUT）。

## GATE_BENCH 实测流程

**必须带 `profile` feature 构建**（否则 `GateProfilerPlugin` 只注册空资源，不产生任何数据）：

```powershell
cargo build -p gate-app --release --features profile
$env:GATE_BENCH="1"
Start-Process -FilePath <target\release\gate-app.exe> -ArgumentList "--nuke" `
  -WorkingDirectory "c:\repo\repo.rust\gate\gate-app" -WindowStyle Minimized
Start-Sleep -Seconds 30
Select-String -Path gate-app\logs\latest.log -Pattern "逐 pass" | Select-Object -Last 2
```

数据源是 **`latest.log` 里的 `GPU 逐 pass 均值（共 X ms/frame）：…` 行**，它**每 2 秒**输出
一个聚合窗口（`GPU[2.0s x120]` = 该窗口 120 帧的均值）；列名 = compute pass 的 label
（`gate_dda_trace` / `gate_ddgi_cast` / `gate_ddgi_bake0..3` / …），合计即当帧全部 GPU 工作。

- **必须取稳态窗口**：启动后头几秒会夹带一次性成本（首帧 `ddgi_dirty.full` → 全量烘焙，
  `bake1` 在该窗口可达 1.4ms）。用 `Select-Object -Last 1` 只取最后一个窗口，或确认相邻
  窗口数值一致后再读数 —— 曾因取到启动窗口，把一次性烘焙误判成"每帧 1.4ms 的瓶颈"。
- **`gpu_frame.log` / `frame_time.log` / `fps.log` 已废弃**（写入代码在重构中消失，文件
  最后更新停在 9/7、9/9）。旧版本节教的 `wall_ms` / `frame_gpu_ms` 列与现在的 pass 均值
  **不同源、不可比**，不要再拿它当基线。
- 画面错误会表现为 trace "假优化"（射线集体 miss → 黑屏但时序变好）。日志里搜
  `DeviceLost`/`ERROR`；最终画面由用户目测确认，不要自己截图。
- **稳态基线**（RTX 3070 / nuke.vox / 1280×720，2026-09-12，各取 2 个稳态窗口）：

  | 配置 | 合计 | 明细（ms） |
  |---|---|---|
  | `GATE_DDGI_STAGE=0`（DDGI 关） | 1.32 | trace 1.27 |
  | 默认（DDGI Full） | 2.86 | trace 1.60 + cast 0.94 + collect 0.22 + sort 0.05 + seal 0.01 |

  ⇒ DDGI 净开销 ≈ 1.55ms，`cast` 占 61%（131072 条射线的场景 trace + 命中点采样）。
  **注意**：trace 在 DDGI 开时由 1.27 涨到 1.60，那 0.33ms 是**着色侧 `ddgi_sample`** 的
  成本（含级联混合带），不要误当成 ray 遍历变慢。

## A/B 方法论（关键教训）

- **单次启动对比不可信**：进程间 GPU 时钟/热漂移 ±0.07ms，曾把"零开销的 LOD 冷路径"
  误判为慢 0.07ms。
- 正确做法：**同一会话内交替 on/off 多轮**（on,off,on,off 各取 median），median 差异
  小于 p25–p75 带宽即视为无差异。
- 诊断开关（环境变量，进程启动前设置）：`GATE_NO_LUT=1`（关 b_leaves 掩码剔除）、
  `GATE_NO_BEAM=1`。`GATE_NO_LOD=1` 已失效（远场早停已移除，见铁律 3）。无 env 开关的
  改动，临时把 WGSL 条件改成 `false && ...` 做 A/B，测完恢复。
- **DDGI 的 A/B**：整条链路用 `GATE_DDGI_STAGE=0`(Off) / `3`(Full)；局部单点用运行时滑杆
  （Debug 面板：Borrow 借针半径、Cheb std 深度信任系数，见 `DdgiDebugSettings`）——
  滑杆可同会话内来回切，比改常量重编译可靠得多。
- 热路径优化（除法→预计算 inv_rd 乘法、命中直返省一整轮外层、bit-first 空气零 load）
  要同时改 CPU 镜像并过测试；近似开关（LUT）只加 WGSL，注释标明。

## 常用命令

```
cargo test -p gate-render --lib brickmap::dda::         # CPU 等价性门禁（trace_chunk 遍历）
cargo test -p gate-render --test wgsl_compile --release # shader naga 校验 + Rust/WGSL 常量对齐
cargo build -p gate-app --release                      # 发布构建（shader 运行时加载）
cargo build -p gate-app --release --features profile   # 带 GPU pass 计时（GATE_BENCH 用）
cargo run -p gate-app --release -- --nuke              # 正常运行（GATE_BENCH=1 加基准）
```
参考资料：Douglas octo-release 算法原文在 `gate-app/logs/octo_march_core.txt`
（march_intersection_buffer / traverse_bit_set / dda / firstTrailingBit）。
