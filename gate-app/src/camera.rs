//! 相机与输入：两种相机模式（轨道 / 幽灵飞行）、左键拾取 recenter、调试开关（V 可见性缓存）。
//!
//! 模式互斥由 [`CameraMode`] 单点决定：两套输入系统各自在「不是自己的模式」时直接返回，
//! 由 [`build_camera_config`] 按当前模式统一构造 `DdaCameraConfig`（**唯一矩阵构造点**）。
//! 朝向（yaw/pitch）两种模式**共享** —— 右键拖拽 = 转头，切换模式时视线方向连续。

use bevy::{
  input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll, MouseScrollUnit},
  prelude::*,
};

use gate_render::{
  BrickMapBuffers, BrickMapBuilder, DdaCameraConfig, OrbitCamera, VIEW_SIZE, VoxelScene,
  cpu_reference_trace_volumes,
};
use gate_voxel::VolumeTransform;

// ---- 相机参数（P2.6 from_orbit 使用；用户已取消"最远距离"限制）----
// CAM_FAR = 透视投影 far 面；shaders/voxel_raytrace/ 内 DDA 射线 t_max 同步到此量级。
// 原 4000（80m）→ 现 65536（1310.72m）足够 zoom-out 到整个 tile 场景（~1000 voxel）
// 缩成屏幕 1% 像素仍可见。DIST_MAX 已删除，滚轮 zoom-out 距离本身无上限。
pub(crate) const FOV_Y: f32 = 60.0_f32.to_radians();
pub(crate) const CAM_NEAR: f32 = 1.0;
pub(crate) const CAM_FAR: f32 = 65536.0;
// ---- 输入灵敏度（spec FR-3；手感调整只改这里）----
const ROT_SPEED: f32 = 0.005; // rad/px（右键拖拽旋转）
pub(crate) const ZOOM_LOG_SPEED: f32 = 0.35; // /行（滚轮乘法缩放，各距离档手感一致）

// ---- 幽灵模式（Minecraft spectator 风格：无碰撞自由飞行）----
// 速度单位 = voxel/s；1 voxel = 2cm（512 voxel = 10.24m）→ 128 v/s ≈ 2.6 m/s。
/// 默认飞行速度（voxel/s；低速档基础速度）
pub(crate) const FLY_SPEED_DEFAULT: f32 = 128.0;
/// 高速档倍率：Shift 切到高速时实际速度 = 基础速度 × 此值（低速的 2 倍）
pub(crate) const FLY_SPEED_FAST_MUL: f32 = 2.0;
/// 飞行速度滑杆的范围 / 步进（Camera tab 与 clamp 共用；16 v/s ≈ 0.32 m/s，2048 v/s ≈ 41 m/s）
pub(crate) const FLY_SPEED_MIN: f32 = 16.0;
pub(crate) const FLY_SPEED_MAX: f32 = 2048.0;
pub(crate) const FLY_SPEED_STEP: f32 = 16.0;

/// 相机模式（main world Resource）。切换的唯一入口是 DebugView 的 Camera tab 互斥按钮组。
#[derive(Resource, Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum CameraMode {
  /// 轨道相机（P2.6）：右键旋转 / 中键平移 / 滚轮缩放
  Orbit,
  /// 幽灵飞行：WASD 沿视线平移、Space 升 / Ctrl 降，**不做碰撞检测**（默认模式）
  #[default]
  Fly,
}

/// 幽灵相机状态。**朝向不在这里** —— 复用 [`OrbitCamera`] 的 yaw/pitch（两模式共享同一套
/// 朝向语义），所以切换模式时视线方向连续，只有位置需要对一次（见 [`sync_camera_mode_switch`]）。
#[derive(Resource, Clone, Copy, Debug)]
pub struct FlyCamera {
  /// 眼位（voxel）
  pub pos: Vec3,
  /// 基础飞行速度（voxel/s；**低速档**值，滑杆改的就是它）
  pub speed: f32,
  /// Shift 切换：true = 高速档（实际速度 = `speed` × [`FLY_SPEED_FAST_MUL`]）
  pub fast: bool,
}

impl Default for FlyCamera {
  fn default() -> Self {
    Self {
      pos: Vec3::new(700.0, 560.0, 700.0),
      speed: FLY_SPEED_DEFAULT,
      fast: false,
    }
  }
}

impl FlyCamera {
  /// 本帧实际飞行速度（低速档 = 基础速度，高速档 = 基础速度 × 倍率）
  pub fn effective_speed(&self) -> f32 {
    if self.fast {
      self.speed * FLY_SPEED_FAST_MUL
    } else {
      self.speed
    }
  }
}

/// yaw/pitch → 视线单位向量（`OrbitCamera::eye()` 的偏移方向取反，即 eye→target 方向）。
/// 唯一出处：中键平移基、飞行前进方向、矩阵构造都从这里取，避免三处各写一遍符号。
pub(crate) fn look_forward(yaw: f32, pitch: f32) -> Vec3 {
  let (sin_yaw, cos_yaw) = yaw.sin_cos();
  let (sin_pitch, cos_pitch) = pitch.sin_cos();
  Vec3::new(-sin_yaw * cos_pitch, -sin_pitch, -cos_yaw * cos_pitch)
}

/// 【诊断】GATE_ORBIT=1：相机每帧**边转边沿圆轨迹平移**（0.35 rad/s + 50 体素/s ≈ 2 m/s）。
///
/// 用途：复现"相机移动中"才出现的帧时/画质问题。**必须带平移** —— 纯旋转不改变相机所在的
/// world cell / chunk，滚动网格、chunk 池同步、"最大空子块"重烘这些路径一次都不会触发，
/// 结果看起来和静止一模一样（第一次就是这么白测的）。半径 ≈ 333 体素（13m），会跨过
/// LOD0 cell（0.64m）、LOD1~3 cell 与 256 体素的 chunk 边界。
/// 配 `GATE_BENCH=1` 启动，日志里每 2 秒那行"GPU 逐 pass 均值"就是移动中的统计。
pub(crate) fn auto_orbit_system(time: Res<Time>, mut orbit: ResMut<OrbitCamera>) {
  let dt = time.delta_secs();
  orbit.yaw -= 0.35 * dt;
  let a = time.elapsed_secs() * 0.15;
  let speed = 500.0 * dt; // 体素/帧（≈20 m/s @60fps：每 ~13 帧跨一个 256 体素 chunk）
  orbit.target += Vec3::new(-a.sin(), 0.0, a.cos()) * speed;
  orbit.clamp();
}

/// 共享转头：右键拖拽旋转 yaw/pitch。两种模式都生效 —— 轨道是「绕目标转」，幽灵是「原地转头」，
/// 只有矩阵构造按模式取目标/眼位。`pitch` 已 clamp 到 ±89°，保证视线与 +Y 不共线。
pub(crate) fn camera_look_input(
  mouse: Res<ButtonInput<MouseButton>>,
  motion: Res<AccumulatedMouseMotion>,
  captured: Res<gate_ui::UiPointerCaptured>,
  mut orbit: ResMut<OrbitCamera>,
) {
  if captured.0 || !mouse.pressed(MouseButton::Right) {
    return;
  }
  let delta = motion.delta;
  orbit.yaw -= delta.x * ROT_SPEED;
  orbit.pitch += delta.y * ROT_SPEED;
  orbit.clamp();
}

/// 轨道相机输入（P2.6 spec FR-3/FR-4；仅 Orbit 模式）：
/// - 中键拖拽 = 平移 target（按当前距离缩放 pan 速度，1:1 跟手）
/// - 滚轮 = 对数缩放（exp(±ZOOM_LOG_SPEED·lines)，各距离档手感一致）+ Shift 细调 1/10
/// - 旋转（右键）在 [`camera_look_input`]，两模式共享
#[allow(clippy::too_many_arguments)] // Bevy system：输入/资源逐一注入
pub(crate) fn orbit_camera_input(
  mouse: Res<ButtonInput<MouseButton>>,
  keys: Res<ButtonInput<KeyCode>>,
  motion: Res<AccumulatedMouseMotion>,
  scroll: Res<AccumulatedMouseScroll>,
  captured: Res<gate_ui::UiPointerCaptured>,
  windows: Query<&Window>,
  mode: Res<CameraMode>,
  mut orbit: ResMut<OrbitCamera>,
) {
  if *mode != CameraMode::Orbit {
    return;
  }
  // UI 指针捕获优先（2.7a FR-6）：hover/按下控件时吞掉拖拽/滚轮
  if !captured.0 {
    let delta = motion.delta;
    if mouse.pressed(MouseButton::Middle) {
      // 正交基：forward = eye→target；right = forward × Y；up = right × forward
      // （与 look_at_rh 的 xaxis/yaxis 同构；pitch ±89° clamp 保证 forward 不与 Y 共线）
      let forward = look_forward(orbit.yaw, orbit.pitch);
      let right = forward.cross(Vec3::Y).normalize();
      let up = right.cross(forward).normalize();
      let pan_per_px = orbit.distance * 2.0 * (FOV_Y / 2.0).tan() / window_height(&windows);
      orbit.target += (right * (-delta.x) + up * delta.y) * pan_per_px;
    }

    // 滚轮缩放独立于拖拽（spec：滚轮可与拖拽共存）；Line 单位，Pixel 按典型行高 16px 折算
    let lines = match scroll.unit {
      MouseScrollUnit::Line => scroll.delta.y,
      MouseScrollUnit::Pixel => scroll.delta.y / 16.0,
    };
    if lines != 0.0 {
      // Shift 细调：步进缩为 1/10（同距离变化所需滚轮行 ×10）
      let zoom_speed = if keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight) {
        ZOOM_LOG_SPEED * 0.1
      } else {
        ZOOM_LOG_SPEED
      };
      orbit.distance *= (-lines * zoom_speed).exp();
      orbit.clamp();
    }
  }
}

/// 幽灵模式飞行输入（仅 Fly 模式，**无碰撞**）：WASD 沿视线平移，Space 上升 / Ctrl 下降（世界 +Y）。
/// 前进/右向取自当前 yaw/pitch，所以「往哪看就往哪飞」；斜向移动归一化，速度与单键一致。
/// Shift **按下切换**低速/高速档（高速 = 低速 × [`FLY_SPEED_FAST_MUL`]），不是按住加速。
///
/// **不按 UI 指针捕获闸门**：`UiPointerCaptured` 只对鼠标有意义（拖滑杆/点击控件），
/// 用它挡键盘会导致"鼠标恰好停在 DebugView 面板上时飞不动"。
pub(crate) fn fly_camera_input(
  keys: Res<ButtonInput<KeyCode>>,
  time: Res<Time>,
  mode: Res<CameraMode>,
  orbit: Res<OrbitCamera>,
  mut fly: ResMut<FlyCamera>,
) {
  if *mode != CameraMode::Fly {
    return;
  }
  // Control 切换低速/高速档（just_pressed = 按一下切一次，不是按住）
  if keys.just_pressed(KeyCode::ControlLeft) || keys.just_pressed(KeyCode::ControlRight) {
    fly.fast = !fly.fast;
    bevy::log::info!(
      "FLY SPEED MODE → {} ({:.0} v/s)",
      if fly.fast { "fast" } else { "slow" },
      fly.effective_speed(),
    );
  }
  // dt 上限 0.1s：断点/长卡顿后不会一帧瞬移出去（代价是那种帧里"飞得慢一点"）
  let dt = time.delta_secs().min(0.1);
  let forward = look_forward(orbit.yaw, orbit.pitch);
  let right = forward.cross(Vec3::Y).normalize_or_zero();
  let mut dir = Vec3::ZERO;
  if keys.pressed(KeyCode::KeyW) {
    dir += forward;
  }
  if keys.pressed(KeyCode::KeyS) {
    dir -= forward;
  }
  if keys.pressed(KeyCode::KeyD) {
    dir += right;
  }
  if keys.pressed(KeyCode::KeyA) {
    dir -= right;
  }
  if keys.pressed(KeyCode::Space) {
    dir += Vec3::Y;
  }
  if keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight) {
    dir -= Vec3::Y;
  }
  if let Some(d) = dir.try_normalize() {
    // 先把速度读出来：`fly.pos += ... fly.effective_speed() ...` 会同时要求可变与不可变借用
    let speed = fly.effective_speed();
    fly.pos += d * speed * dt;
  }
}

/// 切换模式时对一次位置，避免视野跳变（朝向 yaw/pitch 本来就共享）：
/// - Orbit → Fly：飞行起点 = 轨道眼位
/// - Fly → Orbit：把轨道 target 放到「沿当前朝向 `distance` 处」，使 `orbit.eye() == fly.pos`
///
/// 首帧资源刚插入时 `is_changed()` 也为真：只要 `FlyCamera` 初始化成轨道眼位（scene.rs 如此），
/// 这一步是幂等的，不会动初始机位。
pub(crate) fn sync_camera_mode_switch(
  mode: Res<CameraMode>,
  mut orbit: ResMut<OrbitCamera>,
  mut fly: ResMut<FlyCamera>,
) {
  if !mode.is_changed() {
    return;
  }
  match *mode {
    CameraMode::Fly => fly.pos = orbit.eye(),
    CameraMode::Orbit => {
      orbit.target = fly.pos + look_forward(orbit.yaw, orbit.pitch) * orbit.distance;
    }
  }
  bevy::log::info!(
    "CAMERA MODE → {:?} (eye=({:.1},{:.1},{:.1}) yaw={:.2} pitch={:.2})",
    *mode,
    fly.pos.x,
    fly.pos.y,
    fly.pos.z,
    orbit.yaw,
    orbit.pitch,
  );
}

/// 按当前模式重建 `DdaCameraConfig`（**唯一矩阵构造点**；幂等，成本 = 一次 4×4 求逆）。
/// 必须排在所有相机输入之后：同帧的位移/旋转当帧生效。aspect 读当前窗口物理尺寸
/// （FR-5：resize 后 ≤1 帧生效）。
pub(crate) fn build_camera_config(
  mode: Res<CameraMode>,
  orbit: Res<OrbitCamera>,
  fly: Res<FlyCamera>,
  windows: Query<&Window>,
  mut cfg: ResMut<DdaCameraConfig>,
) {
  let Ok(window) = windows.single() else { return };
  let aspect = window.physical_width().max(1) as f32 / window.physical_height().max(1) as f32;
  *cfg = match *mode {
    CameraMode::Orbit => DdaCameraConfig::from_orbit(&orbit, FOV_Y, aspect, CAM_NEAR, CAM_FAR),
    CameraMode::Fly => DdaCameraConfig::from_eye_forward(
      fly.pos,
      look_forward(orbit.yaw, orbit.pitch),
      FOV_Y,
      aspect,
      CAM_NEAR,
      CAM_FAR,
    ),
  };
}

/// 主窗口物理高度（pan_per_px 1:1 基准；无窗口时回退 VIEW_SIZE.y）
fn window_height(windows: &Query<&Window>) -> f32 {
  windows
    .single()
    .map(|w| w.physical_height().max(1) as f32)
    .unwrap_or(VIEW_SIZE.y as f32)
}

/// 屏幕光标 → 世界射线 `(origin, dir)`（voxel 空间；origin = 相机眼位）。
///
/// 唯一出处：轨道模式的左键 recenter 与幽灵模式的体素编辑共用同一套反投影，
/// 避免 NDC / y 翻转 / 退化保护在两处各写一遍。指针不在窗口内 / 矩阵退化 → None。
pub(crate) fn cursor_ray(window: &Window, cfg: &DdaCameraConfig) -> Option<(Vec3, Vec3)> {
  let cursor = window.cursor_position()?;
  let sf = window.scale_factor();
  let phys = cursor * sf; // 物理像素（左上原点，y 向下）
  let pw = window.physical_width().max(1) as f32;
  let ph = window.physical_height().max(1) as f32;
  let u = (phys.x / pw) * 2.0 - 1.0; // [-1, 1]
  let v = 1.0 - (phys.y / ph) * 2.0; // [-1, 1]，翻转 y（NDC +y 朝上）
  let near = cfg.inv_view_proj * Vec4::new(u, v, 0.0, 1.0);
  let far = cfg.inv_view_proj * Vec4::new(u, v, 1.0, 1.0);
  let near = near.truncate() / near.w;
  let far = far.truncate() / far.w;
  let dir = (far - near).normalize_or_zero();
  if dir.length_squared() < 1e-20 {
    return None;
  }
  Some((cfg.position_world, dir))
}

/// 左键拾取 recenter（**仅轨道模式**）：射线命中体素表面 → 轨道目标移到命中点。
/// - 未命中任何体素 / 物体 → 不做操作。
/// - 命中点用射线入点 voxel 坐标（命中面外侧向内偏半个 voxel，避免 target 贴着面导致
///   距离过近时 pitch clamp 抖动）。
/// - UI 捕获指针（UI 控件上点击）时跳过，避免误触发。
/// - 幽灵模式下不生效：那里没有"轨道目标"可言（左键留给体素编辑）。
/// - CPU picking：用 `cpu_reference_trace_volumes` 同步跑主世界+物体两级 DDA，
///   射线由 [`cursor_ray`] 从 `DdaCameraConfig.inv_view_proj` 反投影。
#[allow(clippy::too_many_arguments)] // 多资源 = 点击成本可接受
pub(crate) fn left_click_pick_recenter(
  mouse: Res<ButtonInput<MouseButton>>,
  captured: Res<gate_ui::UiPointerCaptured>,
  windows: Query<&Window>,
  cfg: Res<DdaCameraConfig>,
  mode: Res<CameraMode>,
  scene: Option<Res<VoxelScene>>,
  mut orbit: ResMut<OrbitCamera>,
) {
  if *mode != CameraMode::Orbit {
    return;
  }
  if !mouse.just_pressed(MouseButton::Left) {
    return;
  }
  if captured.0 {
    return; // UI 控件点击 → 吞掉
  }
  let Some(scene) = scene else {
    return;
  };
  let Ok(window) = windows.single() else {
    return;
  };
  let Some((origin, dir)) = cursor_ray(window, &cfg) else {
    return;
  };
  let t_max = CAM_FAR - CAM_NEAR;
  // ---- 2) CPU picking：从 VoxelScene.volumes 同步构建各 volume brickmap + trace
  // 点击低频（用户输入），且极限场景 ~300 chunk 单次 build_full <150ms；
  // 故意不做跨帧缓存——编辑（每 120 帧 chunk 改写）会让缓存与实际渲染画面
  // 不匹配，造成"点到空气也 recenter"的错觉。宁可点击时重建也不提供假命中。
  let per_vol_bufs: Vec<BrickMapBuffers> = scene
    .volumes
    .list
    .iter()
    .map(|v| BrickMapBuilder::build_full(v).buffers().clone())
    .collect();
  let vols_with_tr: Vec<(&BrickMapBuffers, VolumeTransform)> = per_vol_bufs
    .iter()
    .zip(scene.volumes.list.iter().map(|v| v.transform))
    .collect();
  // ---- 3) trace_volumes：主世界 + 物体统一求最近 ----
  if let Some(hit) = cpu_reference_trace_volumes(&vols_with_tr, origin, dir, t_max) {
    // 命中点 = origin + t·dir；再朝命中法线方向推半个 voxel（让 target 落在体素内部）。
    let mut p = origin + dir * hit.t;
    let half = 0.5;
    p += hit.normal * half; // 法线朝射线来向 → *+half 把点推进命中体素内 0.5 voxel
    orbit.target = p;
    bevy::log::info!(
      "PICK → target=({:.1},{:.1},{:.1})  t={:.1}  pal={}  obj_id={}",
      p.x,
      p.y,
      p.z,
      hit.t,
      hit.pal,
      hit.obj_id,
    );
    // 注：DdaCameraConfig 由 orbit_camera_input 同帧末尾重建（本系统在其之后），
    // 因此新 target 下帧生效，避免 Update 中段重复 cfg 构造。
  }
}

#[cfg(test)]
mod camera_math_tests {
  use super::*;

  /// `look_forward` 必须 == 归一化视线方向（target - eye）。它是三处共用的唯一出处
  /// （中键平移基、幽灵飞行前进方向、矩阵构造），符号写反会让三处同时错。
  #[test]
  fn look_forward_is_view_direction() {
    for (yaw, pitch) in [(0.0, 0.0), (0.7, 0.3), (-1.2, -0.9), (3.0, 1.5), (0.0, -1.55)] {
      let o = OrbitCamera {
        target: Vec3::ZERO,
        distance: 100.0,
        yaw,
        pitch,
      };
      let expect = (-o.eye()).normalize();
      let got = look_forward(yaw, pitch);
      assert!(
        (got - expect).length() < 1e-6,
        "yaw={yaw} pitch={pitch}: {got:?} vs {expect:?}"
      );
    }
  }

  /// 幽灵相机启动位置 = 轨道眼位（场景初始化如此）→ 切换模式那一帧是幂等的，
  /// 不会把初始机位搬走。这里直接锁住 Orbit 分支的式子。
  #[test]
  fn orbit_target_from_fly_eye_roundtrips() {
    let o = OrbitCamera {
      target: Vec3::new(260.0, 120.0, 260.0),
      distance: 866.0,
      yaw: 0.4,
      pitch: -0.3,
    };
    let eye = o.eye();
    // Fly → Orbit：把 target 放到「沿当前朝向 distance 处」→ 重建出的 eye 必须回到原处
    let target2 = eye + look_forward(o.yaw, o.pitch) * o.distance;
    let o2 = OrbitCamera {
      target: target2,
      ..o
    };
    assert!((o2.eye() - eye).length() < 1e-3, "eye={eye:?} → {:?}", o2.eye());
  }
}

#[cfg(test)]
mod aabb_zoom_tests {
  use super::*;
  use gate_render::{BrickMapGlobals, cpu_reference_dda_ray, cpu_reference_dda_ray_aabb_skip};
  use gate_voxel::PaletteEntry;
  use glam::{Mat4, Vec3, Vec4};

  // ========================= AABB 滚远消失问题 headless 复现 =========================
  //
  // 用自建最小 demo 场景 + 轨道相机 zoom-out（0 / 1 / 2 / 5 行滚轮）
  // 在 32x32 视锥网格上跑 CPU DDA：full 2M 步 vs AABB-skip 2048 步。
  // 若任一场景 diff_hit+diff_pal > 0 → WGSL shader 必然也有完全一样的 bug，
  // 因为 WGSL AABB 代码就是 `cpu_reference_dda_ray_aabb_skip` 的逐字翻译。
  //
  // 运行：cargo test -p gate-app demo_scene_aabb_zoom_out -- --nocapture

  fn scene_for_aabb_zoom_headless() -> (gate_voxel::VolumeGrid, BrickMapGlobals) {
    let mut grid = gate_voxel::VolumeGrid::new();
    paint(&mut grid);
    build_scene(&mut grid);
    let builder = BrickMapBuilder::build_full(&grid);
    (grid, builder.buffers().globals)
  }

  // 自包含最小场景（调色板 + 几何），不依赖 scene.rs 的 demo 极限场景
  fn paint(grid: &mut gate_voxel::VolumeGrid) {
    let palette: &[(u8, [u8; 3], u8)] = &[
      (1, [118, 118, 126], 220),
      (2, [214, 64, 64], 180),
      (3, [72, 196, 96], 200),
      (4, [72, 120, 224], 180),
      (5, [240, 204, 64], 160),
      (6, [64, 216, 216], 160),
    ];
    let pal = grid.palette_mut();
    for &(idx, color, rough) in palette {
      let mut e = PaletteEntry::default();
      e.color = color;
      e.roughness = rough;
      pal.set(idx, e);
    }
  }
  fn build_scene(grid: &mut gate_voxel::VolumeGrid) {
    use gate_voxel::{draw_text, fill_box, fill_bricks, fill_sphere};
    fill_box(grid, glam::IVec3::ZERO, glam::IVec3::new(512, 16, 512), 1);
    for i in (0..512).step_by(128) {
      fill_box(
        grid,
        glam::IVec3::new(i, 16, 0),
        glam::IVec3::new(4, 4, 512),
        6,
      );
      fill_box(
        grid,
        glam::IVec3::new(0, 16, i),
        glam::IVec3::new(512, 4, 4),
        6,
      );
    }
    fill_bricks(
      grid,
      glam::IVec3::new(128, 16, 128),
      glam::IVec3::new(128, 256, 128),
      16,
      2,
    );
    fill_sphere(grid, glam::IVec3::new(192, 320, 192), 64, 5);
    fill_box(
      grid,
      glam::IVec3::new(256, 192, 144),
      glam::IVec3::new(96, 4, 32),
      2,
    );
    fill_bricks(
      grid,
      glam::IVec3::new(352, 16, 128),
      glam::IVec3::new(64, 192, 64),
      16,
      6,
    );
    fill_sphere(grid, glam::IVec3::new(384, 80, 320), 48, 2);
    fill_sphere(grid, glam::IVec3::new(192, 48, 352), 24, 4);
    draw_text(grid, glam::IVec3::new(64, 16, 448), "GATE", 5);
    fill_box(
      grid,
      glam::IVec3::new(656, 64, 64),
      glam::IVec3::new(32, 32, 32),
      5,
    );
    fill_box(
      grid,
      glam::IVec3::new(1264, 240, 240),
      glam::IVec3::new(32, 32, 32),
      4,
    );
  }

  #[test]
  fn demo_scene_aabb_zoom_out_headless() {
    let (grid, g) = scene_for_aabb_zoom_headless();
    let builder = BrickMapBuilder::build_full(&grid);
    let buffers = builder.buffers().clone();
    // 与实际 setup 中 orbit eye/target 一致：eye(700,560,700) target(260,120,260)
    let base_eye = Vec3::new(700.0, 560.0, 700.0);
    let target = Vec3::new(260.0, 120.0, 260.0);
    let aabb_min = Vec3::new(
      g.index_origin_x as f32 * 512.0,
      g.index_origin_y as f32 * 512.0,
      g.index_origin_z as f32 * 512.0,
    );
    let aabb_max = Vec3::new(
      (g.index_origin_x + g.index_dims_x as i32) as f32 * 512.0,
      (g.index_origin_y + g.index_dims_y as i32) as f32 * 512.0,
      (g.index_origin_z + g.index_dims_z as i32) as f32 * 512.0,
    );
    // 初始 distance (700-260,560-120,700-260).len() ≈ 866 voxel
    let base_offset = base_eye - target;
    let base_dist = base_offset.length();
    let base_dir = base_offset / base_dist;
    // 每个 "滚轮 -1 行"（往外滚）: distance *= exp(0.35) ≈ 1.419
    let zoom_factor_for_lines = |lines: i32| (lines as f32 * ZOOM_LOG_SPEED).exp();
    // 测试：0（初始）/ +2（两下滚出 distance ×≈2）/ +5（多滚几下 distance ×≈5.8）/ +10（×28，超远）/ +20（×815）
    for lines in [0i32, 2, 5, 10, 20] {
      let factor = zoom_factor_for_lines(lines);
      let distance = base_dist * factor;
      let eye = target + base_dir * distance;
      let aspect = 16.0 / 9.0;
      let proj = Mat4::perspective_rh(FOV_Y, aspect, CAM_NEAR, CAM_FAR);
      let view = Mat4::look_at_rh(eye, target, Vec3::Y);
      let vp = proj * view;
      let inv_vp = vp.inverse();
      let grid_w = 32usize;
      let mut full_hits = 0usize;
      let mut skip_hits = 0usize;
      let mut diff_hit = 0usize;
      let mut diff_pal = 0usize;
      for iy in 0..grid_w {
        for ix in 0..grid_w {
          let u = (ix as f32 + 0.5) / (grid_w as f32) * 2.0 - 1.0;
          let v = 1.0 - ((iy as f32 + 0.5) / (grid_w as f32)) * 2.0;
          let near = inv_vp * Vec4::new(u, v, 0.0, 1.0);
          let far = inv_vp * Vec4::new(u, v, 1.0, 1.0);
          let near = near.truncate() / near.w;
          let far = far.truncate() / far.w;
          let diff = far - near;
          let frustum_len = diff.length();
          let dir = diff.normalize();
          // 极限场景 AABB 尺寸：12×5×12 tile = (5632, 2560, 5632) voxel，
          // 对角穿越 worst-case ~9000 voxel；给 16384 留 2× 余量避免截断漏画。
          let aabb_max_steps: u32 = 16384;
          let r_full = cpu_reference_dda_ray(&buffers, eye, dir, frustum_len, 2_000_000);
          let r_skip = cpu_reference_dda_ray_aabb_skip(
            &buffers,
            eye,
            dir,
            frustum_len,
            aabb_max_steps,
            aabb_min,
            aabb_max,
          );
          if r_full.is_some() {
            full_hits += 1;
          }
          if r_skip.is_some() {
            skip_hits += 1;
          }
          match (r_full, r_skip) {
            (Some((_, p1)), Some((_, p2))) if p1 != p2 => diff_pal += 1,
            (Some(_), None) | (None, Some(_)) => diff_hit += 1,
            _ => {}
          }
        }
      }
      println!(
        "ZOOM lines={lines:>3} distance={distance:>8.0} AABB voxel min=({:.0},{:.0},{:.0}) max=({:.0},{:.0},{:.0})\n  \
         32x32 full_hits={full_hits}/{} skip_hits={skip_hits}/{} diff_hit={diff_hit} diff_pal={diff_pal}",
        aabb_min.x,
        aabb_min.y,
        aabb_min.z,
        aabb_max.x,
        aabb_max.y,
        aabb_max.z,
        grid_w * grid_w,
        grid_w * grid_w,
      );
      // 硬断言：Rust AABB skip 与 full DDA 在真实 demo scene + 真实 zoom-out 操作下必须逐像素一致
      assert!(
        diff_hit + diff_pal == 0,
        "zoom lines={lines} distance={distance}: full/skip MISMATCH {}/{} (hit+pal)",
        diff_hit + diff_pal,
        grid_w * grid_w
      );
    }
  }
}
