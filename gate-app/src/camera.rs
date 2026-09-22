//! 相机与输入：两种相机模式（轨道 / 幽灵飞行）、左键拾取 recenter。
//! 模式互斥由 `CameraMode` 单点决定；`build_camera_config` 是唯一矩阵构造点。
//! 朝向 yaw/pitch 两模式共享，切换模式时视线方向连续。

use bevy::{
  input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll, MouseScrollUnit},
  prelude::*,
};
use serde::{Deserialize, Serialize};

use gate_render::{
  BrickMapBuffers, BrickMapBuilder, DdaCameraConfig, OrbitCamera, VIEW_SIZE, VoxelScene,
  cpu_reference_trace_volumes,
};
use gate_voxel::VolumeTransform;

use crate::consts::{
  CAM_FAR, CAM_NEAR, FLY_SPEED_DEFAULT, FLY_SPEED_FAST_MUL, FOV_Y, ROT_SPEED, ZOOM_LOG_SPEED,
};

/// 相机模式（main world Resource）。切换的唯一入口是 DebugMenu 的「玩家/相机/相机模式」切换组。
#[derive(Resource, Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
pub enum CameraMode {
  /// 轨道相机：右键旋转 / 中键平移 / 滚轮缩放
  Orbit,
  /// 幽灵飞行：WASD 沿视线平移、Space 升 / Shift 降，不做碰撞检测（默认模式）
  #[default]
  Fly,
}

/// 幽灵相机状态；朝向复用 `OrbitCamera` 的 yaw/pitch（两模式共享），位置在模式切换时同步
/// （见 `sync_camera_mode_switch`）。
#[derive(Resource, Clone, Copy, Debug)]
pub struct FlyCamera {
  /// 眼位（voxel）
  pub pos: Vec3,
  /// 基础飞行速度（voxel/s；低速档值，滑杆改的就是它）
  pub speed: f32,
  /// true = 高速档（实际速度 = `speed` × `FLY_SPEED_FAST_MUL`）
  pub fast: bool,
}

impl Default for FlyCamera {
  fn default() -> Self {
    Self { pos: Vec3::new(700.0, 560.0, 700.0), speed: FLY_SPEED_DEFAULT, fast: false }
  }
}

impl FlyCamera {
  /// 本帧实际飞行速度（含高速档倍率）
  pub fn effective_speed(&self) -> f32 {
    if self.fast { self.speed * FLY_SPEED_FAST_MUL } else { self.speed }
  }
}

/// 跨启动保留的相机姿态（持久化在 `<安装根>/data/config.toml` 的 `[camera]` 节）。
/// 只存两模式共有的量：眼位 + yaw/pitch（朝向共享）。
/// 轨道参数由 eye/yaw/pitch/distance 反推 —— 保证 `orbit.eye() == eye`，
/// 首帧 `sync_camera_mode_switch` 才幂等（否则它会用 `orbit.eye()` 覆盖眼位）。
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct CameraPose {
  pub mode: CameraMode,
  pub eye: [f32; 3],
  pub yaw: f32,
  pub pitch: f32,
  /// 轨道半径（仅 Orbit 模式有意义）
  pub distance: f32,
}

impl CameraPose {
  /// 记录本帧相机状态（退出时调用）。
  pub fn capture(mode: CameraMode, orbit: &OrbitCamera, fly: &FlyCamera) -> Self {
    // 眼位取当前模式自己的那个：Orbit 用 `orbit.eye()`，Fly 用 `fly.pos`。
    let eye = match mode {
      CameraMode::Orbit => orbit.eye(),
      CameraMode::Fly => fly.pos,
    };
    Self { mode, eye: eye.to_array(), yaw: orbit.yaw, pitch: orbit.pitch, distance: orbit.distance }
  }

  /// 眼位的世界坐标（voxel）。
  fn eye_vec(&self) -> Vec3 {
    Vec3::from_array(self.eye)
  }

  /// 还原轨道参数：`target = eye − distance·dir`，与 `OrbitCamera::eye()` 互为逆运算。
  pub fn to_orbit(self) -> OrbitCamera {
    let dir = look_forward(self.yaw, self.pitch) * -1.0;
    let mut o = OrbitCamera {
      target: self.eye_vec() - dir * self.distance,
      distance: self.distance,
      yaw: self.yaw,
      pitch: self.pitch,
    };
    o.clamp();
    o
  }
}

/// yaw/pitch → 视线单位向量（eye→target 方向，即 `OrbitCamera::eye()` 偏移方向取反）；
/// 中键平移基 / 飞行前进 / 矩阵构造共用。
pub(crate) fn look_forward(yaw: f32, pitch: f32) -> Vec3 {
  let (sin_yaw, cos_yaw) = yaw.sin_cos();
  let (sin_pitch, cos_pitch) = pitch.sin_cos();
  Vec3::new(-sin_yaw * cos_pitch, -sin_pitch, -cos_yaw * cos_pitch)
}

/// 相机每帧边转（yaw 0.35 rad/s）边沿圆轨迹平移（500 voxel/s）；由 `consts::AUTO_ORBIT` 决定是否注册。
/// 必须带平移：纯旋转不改变所在 world cell / chunk，"移动中"才触发的路径不会跑。
pub(crate) fn auto_orbit_system(time: Res<Time>, mut orbit: ResMut<OrbitCamera>) {
  let dt = time.delta_secs();
  orbit.yaw -= 0.35 * dt;
  let a = time.elapsed_secs() * 0.15;
  let speed = 500.0 * dt; // 500 voxel/s
  orbit.target += Vec3::new(-a.sin(), 0.0, a.cos()) * speed;
  orbit.clamp();
}

/// 共享转头：右键拖拽旋转 yaw/pitch，两种模式都生效（轨道 = 绕目标转，幽灵 = 原地转头）。
/// `pitch` 已 clamp 到 ±89°，保证视线与 +Y 不共线。
pub(crate) fn camera_look_input(
  mouse: Res<ButtonInput<MouseButton>>,
  motion: Res<AccumulatedMouseMotion>,
  captured: Res<gate_ui::UiPointerCaptured>,
  intercepted: Res<gate_ui::MouseIntercepted>,
  mut orbit: ResMut<OrbitCamera>,
) {
  if captured.0 || intercepted.0 || !mouse.pressed(MouseButton::Right) {
    return;
  }
  let delta = motion.delta;
  orbit.yaw -= delta.x * ROT_SPEED;
  orbit.pitch += delta.y * ROT_SPEED;
  orbit.clamp();
}

/// 轨道相机输入（仅 Orbit 模式）：中键拖拽平移 target（按距离缩放 pan，1:1 跟手）、
/// 滚轮对数缩放（exp(±ZOOM_LOG_SPEED·lines)）+ Shift 细调 1/10；右键旋转在 `camera_look_input`。
#[allow(clippy::too_many_arguments)]
pub(crate) fn orbit_camera_input(
  mouse: Res<ButtonInput<MouseButton>>,
  keys: Res<ButtonInput<KeyCode>>,
  motion: Res<AccumulatedMouseMotion>,
  scroll: Res<AccumulatedMouseScroll>,
  captured: Res<gate_ui::UiPointerCaptured>,
  intercepted: Res<gate_ui::MouseIntercepted>,
  windows: Query<&Window>,
  mode: Res<CameraMode>,
  mut orbit: ResMut<OrbitCamera>,
) {
  if *mode != CameraMode::Orbit {
    return;
  }
  // UI 指针捕获/鼠标拦截优先：hover/按下控件时吞掉拖拽/滚轮
  if !captured.0 && !intercepted.0 {
    let delta = motion.delta;
    if mouse.pressed(MouseButton::Middle) {
      // 正交基：forward = eye→target；right = forward × Y；up = right × forward
      // （pitch ±89° clamp 保证 forward 不与 Y 共线）
      let forward = look_forward(orbit.yaw, orbit.pitch);
      let right = forward.cross(Vec3::Y).normalize();
      let up = right.cross(forward).normalize();
      let pan_per_px = orbit.distance * 2.0 * (FOV_Y / 2.0).tan() / window_height(&windows);
      orbit.target += (right * (-delta.x) + up * delta.y) * pan_per_px;
    }

    // 滚轮缩放独立于拖拽；Line 单位，Pixel 按 16px 行高折算
    let lines = match scroll.unit {
      MouseScrollUnit::Line => scroll.delta.y,
      MouseScrollUnit::Pixel => scroll.delta.y / 16.0,
    };
    if lines != 0.0 {
      // Shift 细调：步进缩为 1/10
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

/// 幽灵模式飞行输入（仅 Fly 模式，无碰撞）：WASD 沿视线平移，Space 升 / Shift 降（世界 +Y），
/// 斜向归一化；不受 `UiPointerCaptured` 拦截，但文本输入焦点（`gate_ui::TextInputFocus`）会挡键盘，
/// 滑杆精细拖动占用的 Shift（`gate_ui::UiShiftCaptured`）也不算下降。
pub(crate) fn fly_camera_input(
  keys: Res<ButtonInput<KeyCode>>,
  time: Res<Time>,
  focus: Res<gate_ui::TextInputFocus>,
  shift_captured: Res<gate_ui::UiShiftCaptured>,
  mode: Res<CameraMode>,
  orbit: Res<OrbitCamera>,
  mut fly: ResMut<FlyCamera>,
) {
  if *mode != CameraMode::Fly || focus.0.is_some() {
    return;
  }
  // Control 切换低速/高速档（just_pressed = 按一下切一次）
  if keys.just_pressed(KeyCode::ControlLeft) || keys.just_pressed(KeyCode::ControlRight) {
    fly.fast = !fly.fast;
    bevy::log::debug!(
      "FLY 速度档 → {} {:.0} v/s",
      if fly.fast { "fast" } else { "slow" },
      fly.effective_speed(),
    );
  }
  // dt 上限 0.1s
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
  // Shift 被滑杆精细拖动占用时不下降（同一个按键按一次只能有一个语义）
  if !shift_captured.0
    && (keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight))
  {
    dir -= Vec3::Y;
  }
  if let Some(d) = dir.try_normalize() {
    let speed = fly.effective_speed();
    fly.pos += d * speed * dt;
  }
}

/// 切换模式时对一次位置：Orbit→Fly 取轨道眼位；Fly→Orbit 把 target 放到「沿当前朝向 `distance`
/// 处」使 `orbit.eye() == fly.pos`；`is_changed()` 首帧为真但幂等。
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
    "相机模式 → {:?} eye=({:.1},{:.1},{:.1}) yaw={:.2} pitch={:.2}",
    *mode,
    fly.pos.x,
    fly.pos.y,
    fly.pos.z,
    orbit.yaw,
    orbit.pitch,
  );
}

/// 按当前模式重建 `DdaCameraConfig`（唯一矩阵构造点；幂等，成本 = 一次 4×4 求逆）。
/// 必须排在所有相机输入之后（同帧位移/旋转当帧生效）；aspect 读当前窗口物理尺寸。
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
  windows.single().map(|w| w.physical_height().max(1) as f32).unwrap_or(VIEW_SIZE.y as f32)
}

/// NDC `(u, v)` → 世界射线 `(origin, dir)`（voxel 空间；origin = 相机眼位）；矩阵退化 → None。
fn ndc_ray(cfg: &DdaCameraConfig, u: f32, v: f32) -> Option<(Vec3, Vec3)> {
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

/// 屏幕光标 → 世界射线 `(origin, dir)`。
/// 轨道 recenter / 幽灵编辑共用；指针不在窗口内 / 矩阵退化 → None。
pub(crate) fn cursor_ray(window: &Window, cfg: &DdaCameraConfig) -> Option<(Vec3, Vec3)> {
  let cursor = window.cursor_position()?;
  let sf = window.scale_factor();
  let phys = cursor * sf; // 物理像素（左上原点，y 向下）
  let pw = window.physical_width().max(1) as f32;
  let ph = window.physical_height().max(1) as f32;
  // 指针在别的显示器上时 winit 仍可能给出窗口外的坐标 ⇒ 直接当"没有指针"，
  // 否则会拿一条越界的 NDC 射线去拾取（诊断视图会莫名其妙什么都不画）。
  if !(0.0..pw).contains(&phys.x) || !(0.0..ph).contains(&phys.y) {
    return None;
  }
  let u = (phys.x / pw) * 2.0 - 1.0; // [-1, 1]
  let v = 1.0 - (phys.y / ph) * 2.0; // [-1, 1]，翻转 y（NDC +y 朝上）
  ndc_ray(cfg, u, v)
}

/// 左键拾取 recenter（仅 Orbit 模式）：射线命中体素表面 → 轨道 target 移到命中点（沿入面
/// 法线推进半个 voxel）；未命中 / UI 捕获 / Fly 模式不做操作。CPU picking 走 `cpu_reference_trace_volumes`。
#[allow(clippy::too_many_arguments)]
pub(crate) fn left_click_pick_recenter(
  mouse: Res<ButtonInput<MouseButton>>,
  captured: Res<gate_ui::UiPointerCaptured>,
  intercepted: Res<gate_ui::MouseIntercepted>,
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
  if captured.0 || intercepted.0 {
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
  // CPU picking：点击时同步 build_full 各 volume 的 brickmap 再 trace（不做跨帧缓存）。
  let per_vol_bufs: Vec<BrickMapBuffers> =
    scene.volumes.list.iter().map(|v| BrickMapBuilder::build_full(v).buffers().clone()).collect();
  let vols_with_tr: Vec<(&BrickMapBuffers, VolumeTransform)> =
    per_vol_bufs.iter().zip(scene.volumes.list.iter().map(|v| v.transform)).collect();
  // 主世界 + 物体统一求最近
  if let Some(hit) = cpu_reference_trace_volumes(&vols_with_tr, origin, dir, t_max) {
    let mut p = origin + dir * hit.t;
    let half = 0.5;
    p += hit.normal * half; // 沿入面法线推进命中体素内 0.5 voxel
    orbit.target = p;
    bevy::log::debug!(
      "PICK → target=({:.1},{:.1},{:.1}) t={:.1} pal={} obj_id={}",
      p.x,
      p.y,
      p.z,
      hit.t,
      hit.pal,
      hit.obj_id,
    );
    // 新 target 由 build_camera_config 在本系统之后同帧重建并当帧生效
  }
}
