# P2.7 GPU timestamp + pass 级耗时面板 —— 需求规格

## 一、问题陈述

P2.4~P2.6 完成了 compute shader DDA 主可见性渲染（compute dispatch → storage texture → blit → ViewTarget），P2.7a 交付了统一 bevy_ui 组件库（面板/标签/按钮/滑杆/复选/折线图/滚动列表 + 世界空间投影 + 输入门控 + 响应式）。当前缺少**细粒度 GPU 时间戳**：P2.8 GTX 1660 验收（1080p60，直光+阴影）必须知道**具体 pass 级 GPU 耗时**——是 compute dispatch 占瓶颈，还是 DDA blit / gradient blit，以及 P2.3 砖块图上传的 CPU→GPU write_buffer 开销——才能定向优化。现有 `LogDiagnosticsPlugin` 输出纯文字到控制台，无法在运行时 UI 内实时查看、对比、录制趋势。

**目标**：在主世界 UI 新增**实机 GPU 耗时面板**（复用 2.7a gate-ui widget），每 0.5s 刷新 5 项 pass 级 GPU/CPU 耗时 + 1 条上传耗时，提供 128 样本滑动折线。同时接入 bevy_render `DiagnosticsRecorder` API 给自定义 5 个 pass 打时间戳，不重复造轮子。

---

## 二、范围与边界

### 用户
- **自己（开发者）**：跑 P2.8 GTX 1660、P2.9 v3.1 极限测试时，看面板数值与折线对比优化前后。
- **未来实机测试员**：非沙箱真机验收时截面板截图做证据。

### 目标（本次必须交付）

1. **接入 bevy_render 内置 GPU 时间戳**：4 个自定义 pass + 1 个 CPU→GPU 上传段，共 5 段 GPU/CPU 时间戳，写入 Bevy `DiagnosticsStore`（路径形如 `render/gate_dda_compute/elapsed_gpu`）。
2. **诊断桥接**：Render 世界的 `DiagnosticsStore` 值（PreUpdate 已同步到 main world）提取到 main world 专用资源，消弭 1 帧延迟 + 缺失值。
3. **UI 面板**：复用 gate-ui Panel/Label/Plot/List，新增**固定布局**（不做响应式复杂适配，直接 Px + Percent 组合满足 1080p 基线），显示 6 行（5 段 + 上传）× 3 列（段名 / GPU ms / CPU ms），顶部一行显示 Σ GPU。
4. **折线图**：Plot 组件上叠 `Σ GPU（ms）` 数据曲线，`PlotDomain::Auto(0.0, 20.0)` 固定域（GTX 1660 16.7ms 预算）。
5. **环形列表**：显示 Σ GPU 近 10 次跳变（>2ms 或异常值才追加，不刷屏）。
6. **CI 护栏**：`cargo fmt/clippy/test -D warnings` 全绿，workspace test ≥ 90 绿（P2.7a 是 90，不要求新增 widget 数量单测 + 诊断桥接 headless 单测 ≥ 3）。

### 非目标（明确不做）

1. ❌ Tracy/Optick/Pix 外部剖析接入（交给 P9 性能专项）
2. ❌ Pipeline statistics（shader 调用数/图元数）：先只拿 GPU/CPU 时长；未来扩展由 `RecordDiagnostics::pass_span` 自带，不需要改代码结构
3. ❌ 自定义 wgpu QuerySet / resolve_buffer（直接复用 `RenderDiagnosticsPlugin`，已包含 Features::TIMESTAMP_QUERY 缺失兜底）
4. ❌ GTX 1660 真机验收（P2.8 专项）
5. ❌ 面板可配置开关/保存/导出 CSV

---

## 三、功能需求（FR）

### FR-1 诊断接入：5 段 time span（rule）
在 gate-render 的 4 个自定义 pass + 1 个上传段写入时间戳：

| 段名（Diagnostic path 后缀） | 来源文件 | 类型 | 写入位置 |
|---|---|---|---|
| `gate_gradient_compute` | gradient.rs `dispatch_gradient` | compute，`time_span(encoder, name)` + guard `.end(encoder)` | dispatch 前 → dispatch_workgroups 后 |
| `gate_gradient_blit` | gradient.rs `blit_view` | render pass，`pass_span(pass, name)` + guard `.end(pass)` | begin_render_pass → draw 之后 |
| `gate_dda_compute` | dda.rs `dispatch_dda` | compute，`time_span` / guard | begin_compute_pass → dispatch_workgroups 后 |
| `gate_dda_blit` | dda.rs `blit_dda_view` | render pass，`pass_span` / guard | begin_render_pass → draw 之后 |
| `gate_brickmap_upload` | upload.rs 某 Prepare 系统（写 queue.write_buffer） | CPU-side，`time_span(command_encoder)`（虽无 GPU 操作但能测 CPU write_buffer + ensure 两个函数合计，另也有 elapsed_cpu）| ensure 前 → 所有 write_buffer 后 |

**API 用法（设计参考，不写进 spec 实现细节）**：
```rust
let rec = ctx.diagnostic_recorder();
let span = rec.time_span(ctx.command_encoder(), "gate_dda_compute");
// ... dispatch_workgroups ...
span.end(ctx.command_encoder());
```
`RenderDiagnosticsPlugin` 已由 bevy_render 默认添加，DiagnosticsRecorder 每帧初始化，`Rec<T> for Option<Arc<T>>` 已实现——没装插件就 no-op，零降级。

### FR-2 诊断桥接：main world 资源（rule）
main world 插入 `GpuPassTimings { items: [PassTiming; 5], upload_ms_cpu: f32, total_gpu_ms: f32, generation: u64 }`，每 0.5s 由系统从 `DiagnosticsStore` 提取：

- Diagnostic path 匹配规则：`render/<段名>/elapsed_gpu`、`render/<段名>/elapsed_cpu`
- `DiagnosticsStore::get(&path).and_then(|d| d.value_smoothed())` 取最近 N 均值（若 N=0 回退 latest）
- 缺失值：保持 `f32::NAN`，UI 显示 "—"
- 1 帧延迟是 Bevy sync_diagnostics（PreUpdate）的机制性延迟：UI 接受无补偿
- 不要求 timestamp_query feature：`elapsed_cpu` 总可用（Feature 缺失时 elapsed_gpu 统一 NAN，面板显示 "—" 且列表追加 "GPU timestamps unsupported" 一次）

### FR-3 UI 面板：6 行 × 3 列表格 + 汇总 + 折线（rule）
复用 gate-ui 已交付的 Panel / Label / Plot，结构：
- 容器 position_type=Absolute，`right: Percent(2.0) / top: Percent(5.0)`，与 2.7a 性能面板左右不重叠
- 面板宽 `Percent(32.0)`，min `px(320)`，max `px(540)`（响应式由 2.7a 系统自动驱动）
- 标题 "GPU 耗时 · 2.7" + 汇总 "Σ GPU  xx.xx ms" 红色 accent 色（`> 16.7` 即过 60fps 预算才红，否则正常 text 色）
- 行：段名（缩写避免换行）/ GPU ms（两位小数）/ CPU ms（两位小数）
  - 行 1：Gradient compute（GC）
  - 行 2：Gradient blit（GB）
  - 行 3：DDA compute（DC）★ 最重要瓶颈行，段名 label 加 accent background 高亮
  - 行 4：DDA blit（DB）
  - 行 5：BrickMap upload（UP）——只有 CPU ms 时 GPU 列显示 "—"
- Plot：128×高度，折线 = total_gpu_ms，域 `PlotDomain::Fixed(0.0, 20.0)`（GTX 1660 60fps 16.7ms 是硬预算）；容量 128 样本
- RingList：容量 6，仅当 `|Δ total| > 2ms` 追加一条 `"+x.xxms / -y.yyms"`；且 GPU unsupported 提示只写 1 次（Local<bool>）

### FR-4 输入门控复用（rule）
新 UI 容器必须触发 `Interaction` → `UiPointerCaptured` 保持 true，轨道相机输入系统的 gate 2.7a 已写 `if captured.0 { return }`，需保持效果（hover 面板时相机不响应）。

### FR-5 2.7a 性能面板并存（rule）
两面板（2.7a 左 / 2.7 右）不得在视觉上重叠（1080p 基线 26% + 32% = 58% < 100%），各自独立生成实体，widget 树无父子关系。

---

## 四、非功能需求（NFR）

### NFR-1 性能开销（rule）
**pass 级打戳本身的开销（写入 query + resolve）对总 GPU 时间影响 ≤ 0.05ms**。验证：同一帧开/关面板时，`gate_dda_compute.elapsed_gpu` 差值 5 帧均值 < 0.05ms。

**面板 UI 绘制 CPU 开销 ≤ 0.2ms / 帧**（5 行 Label、1 张 128 宽度的 Plot 重绘、6 行 list 回写，全是 CPU memcpy + bevy_ui 标准 layout 计算）。
- 基线：无面板时 demo_fps_feed 报告 fps N；加面板后 N / N' ≥ 0.98（2% 可接受）

### NFR-2 质量预算可视化（rubric）
**Scale：0~2**
- `2`：汇总标签超过 16.7ms 立即变红 accent_pressed（视觉醒目），Plot 域上限 20ms 刚好覆盖 50fps 下限，折线接触域上限时画面有视觉信号
- `1`：有红警示色但不区分 16.7ms 界，域有但不贴合 GTX 预算
- `0`：无阈值视觉信号

**pass threshold ≥ 2。**

### NFR-3 CI 护栏（rule）
`cargo fmt --all --check` 0 diff，`cargo clippy --workspace --all-targets -- -D warnings` 0 warning/error，`cargo test --workspace` 全绿。workspace test ≥ 90。

### NFR-4 实机稳定性（rule）
运行 ≥ 3000 帧无 panic、无 crash、无新增非 wgpu 已知 VUID 验证错误。帧率稳定基线 ±5% 内。

---

## 五、约束与依赖

- **约束 C1**：必须使用 bevy_render 内置 `RecordDiagnostics` / `DiagnosticsRecorder`（在 `bevy_render::diagnostic` 导出），禁止重新实现 `wgpu::QuerySet`、`create_buffer(MapAsync)`、copy_buffer_to_buffer 等（RenderDiagnosticsPlugin 已经做了 3 帧环形 + 缺失 Feature 兜底 + map_async 回调）
- **约束 C2**：Diagnostic path 必须是 render/ 前缀的 5 段名（与 P2.8 验收脚本 grep 统一，不得拼别的前缀）
- **约束 C3**：UI 只写在 gate-app（demo 代码），gate-ui crate 不接受任何渲染专属依赖；gate-render 内的诊断戳通过 `ctx.diagnostic_recorder()`（RenderContext 方法）零依赖 gate-ui
- **依赖 D1**：`RenderDiagnosticsPlugin` 已随 `bevy_render::RenderPlugin` 默认注册（0.19 代码 `lib.rs line 382`），无需额外 enable；`Feature::TIMESTAMP_QUERY` 由 `DiagnosticsRecorder::new` 自动探测
- **依赖 D2**：2.7a gate-ui 8 种 widget 已交付（FR-3 使用其中 4 种 Panel/Label/Plot/RingList，Plot 支持 Domain::Fixed 和 capacity；RingList 支持 push 自动顶替）

---

## 六、假设

1. 桌面 Vulkan/DX12 后端都有 TIMESTAMP_QUERY（RTX 3070、GTX 1660 均支持）。Metal/WebGPU 下 GPU ms 显示 "—"，不视为本次 bug
2. `DiagnosticsStore` 中 `value_smoothed()` 的窗口大小是合适的（无需重写平均逻辑）
3. P2.8 GTX 1660 实机验收前不会拆主通道面板结构；布局调整只需调 Val 数字

---

## 七、开放问题（OQ，评审时回答）

1. **OQ-1 折线域：`PlotDomain::Auto` 还是 `Fixed(0,20)`？** —— Fixed 对性能回归有参考线（16.7 就是 100% 预算），但万一 P3 以后直光把单帧推到 20+ms 会截顶。
2. **OQ-2 上传段 GPU 时间是否需要：`ensure/write_buffer` 在 CPU 端调用 `queue.write_buffer` 真正 GPU 拷贝是 submit 时完成，time_span(command_encoder) 测不到 GPU 拷贝耗时**。选择 A：去掉 gate_brickmap_upload 的 elapsed_gpu 行；B：用 `recorder.record_f32` + 独立 COPY_SRC buffer，但 `write_buffer` 之后没位置写 query。选 A。
3. **OQ-3 汇总行颜色：>16.7ms 变红（GTX 1660 60fps 预算）还是 >8.3ms（120fps）？** 选 16.7（P2.8 基线：60fps）。将来 UI 改 token 即可。

---

## 八、验收标准（AC）

AC 类型仅 `rule` 或 `rubric`。

| ID | 类型 | 内容 | 观察方式 |
|---|---|---|---|
| AC-1 | rule | 4 个 pass（GC/GB/DC/DB）+ 1 个 upload 段均在 `DiagnosticsStore` 有 `render/<段>/elapsed_cpu`（支持平台下还有 elapsed_gpu），至少在 PreUpdate 之后的 Update 首帧可通过 path 查到 | headless test：mock `DiagnosticsStore` + 断言 10 个路径存在（5 × elapsed_cpu + 5 × elapsed_gpu，缺失 gpu 可 NAN 但路径要创建）|
| AC-2 | rule | `GpuPassTimings` 资源在 Update 期间每帧 refresh，缺失值为 NAN；generation 单调递增 | 单测：mock 2 帧 DiagnosticsStore，断言读取值正确；缺失时不崩溃 |
| AC-3 | rule | 右侧 GPU 面板 UI 布局与 2.7a 左面板视觉不重叠（1080p 基线） | 实机截图两张对比：左 26% + 右 32% 之和 58%，边框不相交；容器 absolute 坐标 right/top 与 left/top 无冲突 |
| AC-4 | rule | 关键行 DDA compute 行有高亮背景色（accent 色），一眼可读 | 实机截图：第 3 行背景色与 header accent 色差 ΔE≥60（目测判定 pass） |
| AC-5 | rule | Σ GPU > 16.7ms 时汇总标签变红（≤ 16.7ms 用默认 text 色） | headless：构造 2 组 total 值，断言 label.color 变化；实机目测：在 1080p baseline 有一次触发红字（需要构造高负载，可选；若当前负载 < 16.7ms 则用 mock 值验证） |
| AC-6 | rule | RingList 追加规则：\|Δ total\| > 2ms；"GPU timestamps unsupported" 仅 1 次 | 单测：构造跳变值序列，断言 RingList.len() 期望；unsupported 提示用 mock 无 gpu 路径，断言只出现 1 次 |
| AC-7 | rubric | 开销（NFR-1 + NFR-2） | 实机 A/B：面板开 vs 关，DDA GPU 差值 < 0.05ms；FPS 退化 ≤ 2%（或满足 NFR-2 scale ≥ 2）|
| AC-8 | rule | CI 护栏（NFR-3）：fmt 0 diff / clippy 0 warning / workspace tests ≥ 90 | `cargo` 三次命令输出 |
| AC-9 | rule | 实机 ≥ 3000 帧 0 panic 0 crash 0 新增 VUID（wgpu #9213 除外）| 跑一次应用 60s，run.err 统计 |
| AC-10 | rule | OQ 结论实现一致：Fixed(0,20) 域、去掉 upload 的 GPU 列、阈值 16.7ms | 代码审查 3 处：plot 创建、label 显示、阈值比较 |
