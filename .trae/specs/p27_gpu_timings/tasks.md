# P2.7 GPU timestamp + pass 级耗时面板 —— 任务分解

每任务包含：依赖、范围、TR（rule / rubric）、完成证据。

---

## Task 1：4 个 pass + 1 上传段诊断戳接入（gate-render）

**依赖**：无。**优先级**：高。**Status**: pending

**范围**：
1. [gradient.rs](file:///c:/repo/repo.rust/gate/gate-render/src/gradient.rs) `dispatch_gradient`：`let span = ctx.diagnostic_recorder().time_span(ctx.command_encoder(), "gate_gradient_compute"); ... span.end(...)` guard
2. [gradient.rs](file:///c:/repo/repo.rust/gate/gate-render/src/gradient.rs) `blit_view`：`let span = ctx.diagnostic_recorder().pass_span(pass, "gate_gradient_blit"); ... span.end(pass)`
3. [dda.rs](file:///c:/repo/repo.rust/gate/gate-render/src/brickmap/dda.rs) `dispatch_dda`：同上 compute span
4. [dda.rs](file:///c:/repo/repo.rust/gate/gate-render/src/brickmap/dda.rs) `blit_dda_view`：同上 render pass span
5. [upload.rs](file:///c:/repo/repo.rust/gate/gate-render/src/brickmap/upload.rs) `upload_gpu_buffers`（PrepareResources 系统）：在 ensure/write_buffer 外包裹 `ctx.diagnostic_recorder().time_span(encoder, "gate_brickmap_upload")`——无独立 RenderContext 时从主 App 提供的 `RenderContext` 取；若该系统无 RenderContext 参数，先加 `mut ctx: RenderContext` 参数再写 span。

**注意**：`time_span/pass_span` 的 guard 必须显式 `.end(...)`，否则 drop 触发 error/panic。

**TR（全部 rule）**：
- TR-1.1：修改后 5 处均有 span 创建 + end（代码审查，无裸 drop 路径）
- TR-1.2：不添加非 render/ 前缀的 Diagnostic path（grep 审查，`"gate_"` 只出现在 span name 字符串里）
- TR-1.3：编译通过（cargo check gate-render），不引入新依赖

**完成证据**：`git diff` 输出 5 个 span 块；`cargo check` 通过日志；`cargo test -p gate-render` 全绿（不新增单元测试，因为 DiagnosticRecorder 在 main-world headless App 里没 RenderDevice；由 Task3 用 mock DiagnosticsStore 验证读取链）。

---

## Task 2：main world 诊断桥接（GpuPassTimings 资源 + 系统）

**依赖**：无（与 Task 1 可并行）。**优先级**：高。**Status**: pending

**范围**（放在 gate-app main.rs 新 module 块 `// ================= 2.7 GPU timings =================`）：
1. 定义：
```rust
#[derive(Resource, Default)]
pub struct GpuPassTimings {
  pub gc_gpu_ms: f32, pub gc_cpu_ms: f32,
  pub gb_gpu_ms: f32, pub gb_cpu_ms: f32,
  pub dc_gpu_ms: f32, pub dc_cpu_ms: f32,
  pub db_gpu_ms: f32, pub db_cpu_ms: f32,
  pub up_gpu_ms: f32, pub up_cpu_ms: f32,
  pub total_gpu_ms: f32,
  pub gpu_unsupported: bool,
  pub generation: u64,
}
```
2. 系统 `sync_gpu_timings(mut timings, diagnostics, time, mut every_0_5s: Local<bool>)`：
   - 每帧都跑，但仅每 0.5s `time.delta_acc` 累计 ≥ 0.5 时写入 generation +1；中间帧只累积时间不写（降低 UI Plot 重绘节奏）
   - 5 条路径匹配：`DiagnosticPath::new("render/gate_dda_compute/elapsed_gpu")` 等 10 条
   - `value_smoothed()` 为 None 或 store.get() 为 None → 保持 NAN
   - `gpu_unsupported = 全部 5×elapsed_gpu 都是 NAN（且 elapsed_cpu 都有值）`；一旦 true，后续不翻回
   - `total_gpu_ms = gc_gpu + gb_gpu + dc_gpu + db_gpu`；任一 NAN → total 为可用部分之和（并标注可能非完整）
3. 在 `App::build`（main.rs）`add_systems(Update, sync_gpu_timings)`。若 RenderDiagnosticsPlugin 未同步（第 1-2 帧无值）→ 全 NAN，显示 "—"，UI 不 panic。

**TR（rule）**：
- TR-2.1：GpuPassTimings::default() 后单字段全部 0.0 / 0 / false 初始合理（单测）
- TR-2.2：mock DiagnosticsStore 的 headless 单测：注入 10 个路径各 1.0，检查 timings 字段值一致；路径缺失时 NAN 保持不崩溃；gpu_unsupported 标志语义正确（3 条）

**完成证据**：gate-app 3 条单元测试通过。渲染路径 DiagnosticsStore 构造方法：`app.init_resource::<DiagnosticsStore>()` 然后 `store.add(Diagnostic::new(path).with_suffix("ms")); store.get_mut(path).unwrap().add_measurement(...)`—— 纯 main-world 可构造，不依赖 RenderApp。

---

## Task 3：右侧 GPU 面板 UI 构建 + Plot/RingList/阈值高亮

**依赖**：Task 2（用 GpuPassTimings 资源）。**优先级**：高。**Status**: pending

**范围**（main.rs 新系统块）：
1. `demo_gpu_panel_setup`：与 2.7a 性能面板相同的 `Local<bool> done` + `Option<Res<UiTheme>>` 主题就绪后 spawn 一次。结构：
   - 容器 Absolute: `right: Percent(2.0), top: Percent(5.0), width: Percent(32.0), min_width: px(320), max_width: px(540)`
   - Panel
     - Label "GPU 耗时 · 2.7"
     - Label "Σ GPU xx.xx ms" 颜色：`total_gpu_ms > 16.7 → accent_pressed 红（已在 token）否则 text`。加 marker `DemoTotalLabel`
     - Plot（折线域 PlotDomain::Fixed(0.0, 20.0)，capacity 128，128 宽，用 blank_plot_image 创建 handle）。加 marker `DemoGpuPlot`
     - Row 1~5（段名缩写 / GPU ms / CPU ms），DC 行背景色 = accent_hover（半透明）突出
       - 段名字符串缩写一致："GC" "GB" "DC" "DB" "UP"
       - NAN 显示 "—"，否则 `"{:>6.2}"`（2 位小数 + 对齐，保持列齐）
     - RingList 6 行（跳变追加）marker `DemoGpuList`
2. `demo_gpu_refresh(timings: Res<GpuPassTimings>, labels..., plots..., lists...)` 每帧跑：
   - 每帧写 10 个字段 Label（如有 generation 变化时？每帧直接写，bevy_ui 内部 change detection，Text 未变则零 layout 重算）
   - Plot：生成 generation 变化时 push(total_gpu_ms)；域 0..20
   - RingList：`|Δ total| > 2.0ms` 追加；首次发现 `gpu_unsupported` 写一次 "GPU timestamps unsupported"（`Local<Option<bool>>` 守门）
3. 阈值检查：`total_gpu_ms > 16.7` 时把 `DemoTotalLabel` 的 BackgroundColor（或文本颜色）改成 accent_pressed；否则还原为 text 色

**TR**：
- TR-3.1（rule）：headless 构建 App，spawn 面板后断言容器节点的 right/top/width 三个 Val 字段（数值+类型）匹配 Percent(2.0)/Percent(5.0)/Percent(32.0)
- TR-3.2（rule）：DC 行的背景色与其他 4 行色差 ΔE ≥ 60（目测在实机截图即通过；代码审查：DC 行单独有 BackgroundColor(token accent) 设置而其他行没有）
- TR-3.3（rule）：阈值 16.7ms 变色 headless 验证：注入 total=16.699 vs total=16.701 两帧，断言 DemoTotalLabel 的颜色翻转（2 单测）
- TR-3.4（rubric，scale 0-2，threshold ≥ 1）：UI 布局合理。`2` = 行对齐整齐，列宽均匀，Plot 和 list 上下间距与 token spacing.md 一致；`1` = 可用但行有微小错位；`0` = 溢出或文本换行错。由实机截图评审。

**完成证据**：headless 3+2 单测；实机截图 GPU 面板与 2.7a 面板互不遮挡；Plot 有数据（折线可见）；List 跳变条目正常出现。

---

## Task 4：入口装配 + A/B 开销测试

**依赖**：Task 1+2+3 完成。**优先级**：中。**Status**: pending

**范围**：
1. 主 main.rs 顶部 `mod 2.7` 两个系统（`sync_gpu_timings` + `demo_gpu_panel_setup` + `demo_gpu_refresh`）按正确顺序注册：
   - Update：`demo_gpu_panel_setup`（主题就绪 1 次）、`sync_gpu_timings`（在 `demo_gpu_refresh` 之前，显式 `.before(demo_gpu_refresh)`）、`demo_gpu_refresh`
2. A/B 开销测试脚本（实机运行）：
   - A：基准（构建时把右侧面板整个 spawn 代码注释掉，或直接本地 patch）
   - B：面板正常
   - 跑 60s 读 `DiagnosticsStore` 的 `gate_dda_compute.elapsed_gpu`，比较均值差 ≤ 0.05ms（NFR-1）
   - FPS 退化（demo_fps_feed 报告） ≤ 2%
3. 回滚注释 patch（基准只是测试步骤，不进仓库）

**TR**：
- TR-4.1（rule）：实际运行输出 A/B 两组日志，差值 ≤ 0.05ms 与 ≤ 2%
- TR-4.2（rule）：系统注册顺序正确——在同 1 帧内，`timings.generation` 变化发生在 refresh 读取之前（自增后下一帧再消费 or 同帧 before）。两种都 OK；验证法：trace 日志打印顺序或单测

**完成证据**：A/B 测试运行日志写入 review.md；系统顺序审查；实际跑 3000 帧稳定性。

---

## Task 5：CI 护栏 + 3000 帧实机稳定

**依赖**：Task 1-4 完成。**优先级**：高。**Status**: pending

**范围**：
1. `cargo fmt --all --check` 0 diff
2. `cargo clippy --workspace --all-targets -- -D warnings` 0 warning
3. `cargo test --workspace` 全绿，计数 ≥ 90
4. 实机启动一次（非沙箱），运行 ≥ 60s（约 3600 帧 @ 60fps），无 panic 无 crash；日志 VUID 错误仅为 wgpu #9213 两个
5. TODO.md 勾选 2.7；spec.md 中 OQ 回答写进 review.md §Decision

**TR**：
- TR-5.1（rule）：3 条 cargo 命令输出达标
- TR-5.2（rule）：3000 帧稳定（run.err 统计：无 panic/crash/STATUS 行）
- TR-5.3（rule）：OQ 三个问题在 review.md 里有明确回答

**完成证据**：3 条命令输出；实机运行日志尾部；TODO.md diff。

---

## Task 6：独立审查 + review.md + TODO 勾选

**依赖**：Task 1~5 全部 completed。**优先级**：高。**Status**: pending

审查要点：
1. AC-1~AC-10 逐项附证据
2. Task 6 用独立上下文（或至少一整套自审查 checklist）重新检查：
   - 5 段 span 是否都 end() 过（无 drop panic）
   - Diagnostics path 命名规范（与 spec 枚举完全一致，C2）
   - UI 与 2.7a 面板不重叠（截图比对像素坐标）
   - NFR 开销 A/B 证据
3. Review result = `pass` / `fail` / `blocked`；fail 时回填 Task 1~5 为 pending 条目
4. TODO.md 勾选 `2.7`，写简短交付摘要
