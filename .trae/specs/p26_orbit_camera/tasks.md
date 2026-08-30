# P2.6 轨道相机 - 任务清单（Tasks）

> 对应 `spec.md` FR/NFR 编号；每 Task 附验证方式。执行顺序 = 编号顺序。

## Task 1: OrbitCamera 数学层（gate-render，FR-1/FR-2/FR-7/NFR-1）
- [x] `dda.rs` 新增 `OrbitCamera { target, distance, yaw, pitch }`（Resource derive）：
  - `from_eye(eye, target)`：offset → distance/asin/atan2
  - 眼位重建：`eye = target + distance * (sin_yaw*cos_pitch, sin_pitch, cos_yaw*cos_pitch)`
- [x] `DdaCameraConfig::from_orbit(orbit: &OrbitCamera, fov_y: f32, aspect: f32, near: f32, far: f32)`：唯一矩阵构造点（perspective_rh × look_at_rh）
- [x] clamp 常量：`PITCH_LIMIT = 89°`、`DIST_MIN = 32`、`DIST_MAX = 8000`（pub，供 app 引用与测试断言）
- [x] 导出：`brickmap/mod.rs` + `lib.rs` re-export `OrbitCamera`
- [x] 单测 ×3：往返一致 / clamp 生效 / from_orbit == build_static（历史参数回归）
- **验证**：`cargo test -p gate-render --lib orbit`（新增 ≥3 绿）→ 3 passed ✅

## Task 2: 输入 + 每帧管线（gate-app，FR-3/FR-4/FR-5/FR-6）
- [x] `main.rs`：`Window { resizable: false, .. }`
- [x] Startup：注入 `OrbitCamera::from_eye(Vec3::new(700,560,700), Vec3::new(260,120,260))`；`DdaCameraConfig` 改由 from_orbit 生成（替换 build_static 调用）
- [x] Update system `orbit_camera_input`：
  - 右键拖拽旋转（0.005 rad/px，pitch clamp）
  - 中键拖拽平移（PAN_PER_PX = distance * 2*tan(fov/2) / VIEW_SIZE.y，right/up 正交基）
  - 滚轮乘法缩放（exp(-line * 0.35)，clamp [32,8000]）
  - 同帧多操作互斥：旋转 > 平移 > 缩放（拖拽类不叠加）
  - 末尾 `*cfg = DdaCameraConfig::from_orbit(...)`（ResMut 同 system 写，保证 ≤1 帧生效）
- [x] 输入用 `ButtonInput<MouseButton>` + `AccumulatedMouseMotion` + `AccumulatedMouseScroll`（0.19 API：`AccumulatedMouseScroll { unit: MouseScrollUnit, delta: Vec2 }`，prelude 不含需显式 use）
- **验证**：`cargo build -p gate-app` 零警告 ✅

## Task 3: CI 护栏（NFR-2）
- [x] `cargo fmt --all`
- [x] `cargo clippy --workspace --all-targets -- -D warnings`
- [x] `cargo test --workspace`（58 绿 = render 25 + voxel 33，含新增 3 orbit 测试）
- **验证**：三条 exit 0 ✅

## Task 4: 实机验收（AC-2/AC-3/AC-4/NFR-3/AC-6）
- [x] 启动截图（初始构图 == 2.5 基线）→ `ac2_initial.png`
- [x] 右键旋转后截图 / 滚轮缩放后截图 / 中键平移后截图 → `ac3_rotated/zoomed/panned.png`（×3 语义核对通过）
- [x] 持续操作 500 帧日志核查：frame_count=1355、无 panic、VUID 仅 wgpu#9213 两型（各 ×4 初始帧）、UPLOAD[incremental] ×11 持续
- [x] 手感自测评分：4/5（rubric；灵敏度常量未调，保持 spec 值 0.005/0.35）
- **验证**：截图 ×4 + run log 留档（spec 目录）✅（合成输入必须 SendInput 相对移动——SetCursorPos 不产生 Raw Input，见 review.md 备注）

## Task 5: 收尾
- [x] `review.md`（对照 6 AC 逐项给证据）
- [x] TODO.md 勾选 2.6（附实现要点）
- **验证**：用户确认通过
