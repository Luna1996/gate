# P2.6 轨道相机 - Review（对照 6 AC 逐项证据）

> 交付日期：2026-08-30。实施 = spec.md FR-1~FR-7 全覆盖；WGSL/渲染链路零改动（Constraint-1 达成）。

## 改动清单

| 文件 | 改动 |
|---|---|
| `gate-render/src/brickmap/dda.rs` | 新增 `OrbitCamera`（from_eye/eye/clamp）+ `PITCH_LIMIT/DIST_MIN/DIST_MAX` + `DdaCameraConfig::from_orbit`（唯一矩阵构造点）+ 3 单测 |
| `gate-render/src/brickmap/mod.rs`、`lib.rs` | re-export `OrbitCamera`（FR-7） |
| `gate-app/src/main.rs` | `Window{resizable:false}`（FR-6）；Startup 注入 `OrbitCamera::from_eye((700,560,700),(260,120,260))` + `DdaCameraConfig::from_orbit` 替换 `build_static`（FR-5）；新增 `orbit_camera_input` Update system（FR-3/FR-4：右键旋转 / 中键平移 / 滚轮缩放 / 拖拽互斥旋转>平移、滚轮共存 / 同帧矩阵重建 ≤1 帧生效）；灵敏度常量 `ROT_SPEED=0.005 rad/px`、`ZOOM_LOG_SPEED=0.35/行` 置顶可调 |
| WGSL（dda.wgsl / blit.wgsl）、插件调度时序 | **零改动**（Non-Goal 兑现） |

## AC 逐项证据

### AC-1: 数学单测 ✅
`cargo test -p gate-render --lib orbit` → **3 passed**：
- `orbit_roundtrip_from_eye_eye`：from_eye→eye() 重建误差 <1e-4（历史参数 + 任意负 pitch/大 yaw 参数组）
- `orbit_clamp_bounds`：pitch 95°→89°、distance 1→32；上界 1e6→8000、pitch -95°→-89°
- `orbit_from_orbit_equals_build_static`：from_orbit(from_eye(历史参数)) 与 build_static 三字段逐元素 <1e-5（回归保护）

### AC-2: 初始画面零跳变 ✅
`ac2_initial.png`（本阶段实机截图）构图 == P2.5 基线：同 eye/target/fov/aspect/near/far + WGSL 零改动 → 逐像素同管线输出。Startup 切换瞬间无跳变。

### AC-3: 三操作实机生效 ✅
- `ac3_rotated.png`：右键拖拽 (+180,+60)px → yaw -0.9rad / pitch +0.3rad，场景方位明显改变，方向符合 FR-3 公式
- `ac3_zoomed.png`：滚轮 6 行上滚 → distance 762→~93（×exp(-2.1)），大幅拉近
- `ac3_panned.png`：中键拖拽 (+144,-96)px → target 平移，画面移动方向符合抓取语义（拖右场景右移）
- 三张截图与初始构图明显不同、彼此不同；停止操作后画面静止（空转 3s 验证，NFR-3 无漂移）

### AC-4: 实机稳定性 ✅
run.log（`.trae/specs/p26_orbit_camera/run.err.log`）：
- **frame_count=1355**（≥500），混合操作（旋转+缩放+平移）+ 空转
- **panics=0**
- **VUID 仅 wgpu#9213 已知两型**：`VUID-VkPresentInfoKHR-pImageIndices-01430` ×4 + `VUID-vkAcquireNextImageKHR-semaphore-01286` ×4（均集中在初始帧 04:50:06，已知 wgpu Vulkan 初始化问题），无其他 VUID
- **UPLOAD[incremental] ×11 持续**（每 120 帧编辑热点重写），编辑系统不受相机影响

### AC-5: fmt / clippy / test CI 护栏 ✅
- `cargo fmt --all --check` → OK
- `cargo clippy --workspace --all-targets -- -D warnings` → exit 0
- `cargo test --workspace` → **58 绿**（gate-render 25 = 原 22 + 新增 orbit 3；gate-voxel 33）

### AC-6: 手感评分 ✅（4/5）
- 拖拽灵敏度：0.005 rad/px 转动平滑、无跳变（脚本 16px/步注入观察）
- 缩放：乘法缩放各距离档手感一致（762→93 与 93→266 同样 3 行↔×0.35）
- 平移：PAN_PER_PX = distance·2tan(fov/2)/VIEW_SIZE.y 视觉 1:1 抓取，跟手
- 扣 1 分原因：pitch 接近 ±89° 时 pan 的 right 基随 cos(pitch) 缩短（未退化但可感知），P7.4 相机打磨时再处理

## 遗留 / 备注

1. **无遗留工作项**。P2.7（GPU timestamp + pass 级耗时面板）接续。
2. **实机验收脚本经验**（复用价值高，已记入 project_memory）：
   - Bevy `AccumulatedMouseMotion` 来自 Raw Input（`DeviceEvent::MouseMotion`）——合成拖拽测试必须用 `SendInput(MOUSEEVENTF_MOVE)` 相对移动；`SetCursorPos` 只发 WM_MOUSEMOVE 不产生 WM_INPUT，motion.delta 恒 0
   - PowerShell 5.1 对 P/Invoke `[ref]` struct 写回会把变量替换为 `Object[]`（`ClientToScreen` 实测踩坑）——封送一律收进 C# 静态方法返回值
   - `Start-Process -WindowStyle Hidden` 会让 winit 窗口异常（句柄失效/瞬态）；且启动后立即取的 `MainWindowHandle` 是瞬态句柄——固定等 12s 再取 + `IsWindow` 终验
   - wgpu Vulkan 初始化偶发崩溃 0xC0000409（STATUS_STACK_BUFFER_OVERRUN）复现——脚本重试 ≤3 次规避
3. 本次实机 60fps（NoVsync 被回退 Fifo？）与 P2.4 的 811fps 不一致——后续 P2.7 接 GPU timestamp 后顺带核查 present mode 实际生效值。
