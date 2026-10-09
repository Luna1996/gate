use bevy::{
  input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll, MouseScrollUnit},
  prelude::*,
  window::{CursorGrabMode, CursorOptions, PrimaryWindow},
};
use serde::{Deserialize, Serialize};

use gate_render::{DdaCameraConfig, OrbitCamera, VIEW_SIZE, VoxelScene, raycast};
use gate_ui::widgets::px;

use crate::consts::{
  CAM_FAR, CAM_NEAR, CROSSHAIR_ARM, CROSSHAIR_GAP, CROSSHAIR_THICK, FLY_SPEED_DEFAULT,
  FLY_SPEED_FAST_MUL, FOV_Y, ROT_SPEED, ZOOM_LOG_SPEED,
};

#[derive(Resource, Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
pub enum CameraMode {
  Orbit,
  #[default]
  Fly,
}

#[derive(Resource, Clone, Copy, Debug)]
pub struct FlyCamera {
  pub pos: Vec3,
  pub speed: f32,
  pub fast: bool,
}

impl Default for FlyCamera {
  fn default() -> Self {
    Self { pos: Vec3::new(700.0, 560.0, 700.0), speed: FLY_SPEED_DEFAULT, fast: false }
  }
}

impl FlyCamera {
  pub fn effective_speed(&self) -> f32 {
    if self.fast { self.speed * FLY_SPEED_FAST_MUL } else { self.speed }
  }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct CameraPose {
  pub mode: CameraMode,
  pub eye: [f32; 3],
  pub yaw: f32,
  pub pitch: f32,
  pub distance: f32,
}

impl CameraPose {
  pub fn capture(mode: CameraMode, orbit: &OrbitCamera, fly: &FlyCamera) -> Self {
    let eye = match mode {
      CameraMode::Orbit => orbit.eye(),
      CameraMode::Fly => fly.pos,
    };
    Self { mode, eye: eye.to_array(), yaw: orbit.yaw, pitch: orbit.pitch, distance: orbit.distance }
  }

  fn eye_vec(&self) -> Vec3 {
    Vec3::from_array(self.eye)
  }

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

pub(crate) fn look_forward(yaw: f32, pitch: f32) -> Vec3 {
  let (sin_yaw, cos_yaw) = yaw.sin_cos();
  let (sin_pitch, cos_pitch) = pitch.sin_cos();
  Vec3::new(-sin_yaw * cos_pitch, -sin_pitch, -cos_yaw * cos_pitch)
}

pub(crate) fn auto_orbit_system(time: Res<Time>, mut orbit: ResMut<OrbitCamera>) {
  let dt = time.delta_secs();
  orbit.yaw -= 0.35 * dt;
  let a = time.elapsed_secs() * 0.15;
  let speed = 500.0 * dt;
  orbit.target += Vec3::new(-a.sin(), 0.0, a.cos()) * speed;
  orbit.clamp();
}

pub(crate) fn camera_look_input(
  mouse: Res<ButtonInput<MouseButton>>,
  motion: Res<AccumulatedMouseMotion>,
  captured: Res<gate_ui::UiPointerCaptured>,
  intercepted: Res<gate_ui::MouseIntercepted>,
  mode: Res<CameraMode>,
  mut orbit: ResMut<OrbitCamera>,
) {
  if *mode != CameraMode::Orbit || captured.0 || intercepted.0 || !mouse.pressed(MouseButton::Right)
  {
    return;
  }
  let delta = motion.delta;
  orbit.yaw -= delta.x * ROT_SPEED;
  orbit.pitch += delta.y * ROT_SPEED;
  orbit.clamp();
}

#[derive(Resource, Clone, Copy, Debug)]
pub struct MouseLock(pub bool);

impl Default for MouseLock {
  fn default() -> Self {
    Self(true)
  }
}

pub(crate) fn toggle_mouse_lock(
  keys: Res<ButtonInput<KeyCode>>,
  focus: Res<gate_ui::TextInputFocus>,
  mode: Res<CameraMode>,
  mut lock: ResMut<MouseLock>,
) {
  if *mode != CameraMode::Fly || focus.0.is_some() || !keys.just_pressed(KeyCode::KeyQ) {
    return;
  }
  lock.0 = !lock.0;
  bevy::log::info!("鼠标锁定 → {}", if lock.0 { "on" } else { "off" });
}

pub(crate) fn apply_mouse_lock(
  lock: Res<MouseLock>,
  mode: Res<CameraMode>,
  mut cursor: Query<(&Window, &mut CursorOptions), With<PrimaryWindow>>,
) {
  let Ok((window, mut cursor)) = cursor.single_mut() else { return };
  let want = lock.0 && *mode == CameraMode::Fly && window.focused;
  let (visible, grab) =
    if want { (false, CursorGrabMode::Confined) } else { (true, CursorGrabMode::None) };
  if cursor.visible != visible || cursor.grab_mode != grab {
    cursor.visible = visible;
    cursor.grab_mode = grab;
  }
}

pub(crate) fn free_look_input(
  motion: Res<AccumulatedMouseMotion>,
  mode: Res<CameraMode>,
  lock: Res<MouseLock>,
  mut orbit: ResMut<OrbitCamera>,
) {
  if *mode != CameraMode::Fly || !lock.0 {
    return;
  }
  let delta = motion.delta;
  orbit.yaw -= delta.x * ROT_SPEED;
  orbit.pitch += delta.y * ROT_SPEED;
  orbit.clamp();
}

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
  if !captured.0 && !intercepted.0 {
    let delta = motion.delta;
    if mouse.pressed(MouseButton::Middle) {
      let forward = look_forward(orbit.yaw, orbit.pitch);
      let right = forward.cross(Vec3::Y).normalize();
      let up = right.cross(forward).normalize();
      let pan_per_px = orbit.distance * 2.0 * (FOV_Y / 2.0).tan() / window_height(&windows);
      orbit.target += (right * (-delta.x) + up * delta.y) * pan_per_px;
    }

    let lines = match scroll.unit {
      MouseScrollUnit::Line => scroll.delta.y,
      MouseScrollUnit::Pixel => scroll.delta.y / 16.0,
    };
    if lines != 0.0 {
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
  if keys.just_pressed(KeyCode::ControlLeft) || keys.just_pressed(KeyCode::ControlRight) {
    fly.fast = !fly.fast;
    bevy::log::debug!(
      "FLY 速度档 → {} {:.0} v/s",
      if fly.fast { "fast" } else { "slow" },
      fly.effective_speed(),
    );
  }
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
  if !shift_captured.0 && (keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight)) {
    dir -= Vec3::Y;
  }
  if let Some(d) = dir.try_normalize() {
    let speed = fly.effective_speed();
    fly.pos += d * speed * dt;
  }
}

pub(crate) fn auto_fly_system(
  time: Res<Time>,
  mut orbit: ResMut<OrbitCamera>,
  mode: Res<CameraMode>,
  mut fly: ResMut<FlyCamera>,
) {
  if *mode != CameraMode::Fly {
    return;
  }
  let dt = time.delta_secs().min(0.1);
  orbit.yaw -= crate::consts::BENCH_FLY_YAW * dt;
  let a = time.elapsed_secs() * crate::consts::BENCH_FLY_TURN;
  let dir = Vec3::new(-a.sin(), 0.0, a.cos());
  fly.pos += dir * crate::consts::BENCH_FLY_SPEED * dt;
}

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

fn window_height(windows: &Query<&Window>) -> f32 {
  windows.single().map(|w| w.physical_height().max(1) as f32).unwrap_or(VIEW_SIZE.y as f32)
}

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

pub(crate) fn cursor_ray(
  window: &Window,
  cfg: &DdaCameraConfig,
  locked: bool,
) -> Option<(Vec3, Vec3)> {
  if locked {
    return ndc_ray(cfg, 0.0, 0.0);
  }
  let cursor = window.cursor_position()?;
  let sf = window.scale_factor();
  let phys = cursor * sf;
  let pw = window.physical_width().max(1) as f32;
  let ph = window.physical_height().max(1) as f32;
  if !(0.0..pw).contains(&phys.x) || !(0.0..ph).contains(&phys.y) {
    return None;
  }
  let u = (phys.x / pw) * 2.0 - 1.0;
  let v = 1.0 - (phys.y / ph) * 2.0;
  ndc_ray(cfg, u, v)
}

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
    return;
  }
  let Some(scene) = scene else {
    return;
  };
  let Ok(window) = windows.single() else {
    return;
  };
  let Some((origin, dir)) = cursor_ray(window, &cfg, false) else {
    return;
  };
  if let Some(hit) = raycast(&scene.volumes, origin, dir, CAM_FAR - CAM_NEAR) {
    let mut p = origin + dir * hit.t;
    let half = 0.5;
    p += hit.normal * half;
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
  }
}

#[derive(Component)]
pub(crate) struct Crosshair;

const CROSSHAIR_COLOR: Color = Color::WHITE;

pub(crate) fn spawn_crosshair(mut commands: Commands) {
  commands
    .spawn((
      Name::new("crosshair"),
      Crosshair,
      Node {
        position_type: PositionType::Absolute,
        left: Val::Percent(50.0),
        top: Val::Percent(50.0),
        ..default()
      },
      Visibility::Hidden,
    ))
    .with_children(|p| {
      for (dx, dy) in [(-1.0_f32, 0.0_f32), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0)] {
        let (w, h) = if dx == 0.0 {
          (CROSSHAIR_THICK, CROSSHAIR_ARM)
        } else {
          (CROSSHAIR_ARM, CROSSHAIR_THICK)
        };
        let off = CROSSHAIR_GAP + CROSSHAIR_ARM / 2.0;
        p.spawn((
          Node {
            position_type: PositionType::Absolute,
            width: px(w),
            height: px(h),
            left: px(dx * off - w / 2.0),
            top: px(dy * off - h / 2.0),
            ..default()
          },
          BackgroundColor(CROSSHAIR_COLOR),
        ));
      }
    });
}

pub(crate) fn sync_crosshair(
  mode: Res<CameraMode>,
  lock: Res<MouseLock>,
  mut crosshairs: Query<&mut Visibility, With<Crosshair>>,
) {
  let want =
    if *mode == CameraMode::Fly && lock.0 { Visibility::Visible } else { Visibility::Hidden };
  for mut vis in &mut crosshairs {
    if *vis != want {
      *vis = want;
    }
  }
}
