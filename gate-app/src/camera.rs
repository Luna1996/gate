//! 相机与输入：轨道相机拖拽/滚轮、左键拾取 recenter、调试开关（V 可见性缓存）。

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
// CAM_FAR = 透视投影 far 面；dda.wgsl 内 DDA 射线 t_max 同步到此量级。
// 原 4000（10m）→ 现 65536（163.84m）足够 zoom-out 到整个 tile 场景（~1000 fine）
// 缩成屏幕 1% 像素仍可见。DIST_MAX 已删除，滚轮 zoom-out 距离本身无上限。
pub(crate) const FOV_Y: f32 = 60.0_f32.to_radians();
pub(crate) const CAM_NEAR: f32 = 1.0;
pub(crate) const CAM_FAR: f32 = 65536.0;
// ---- 输入灵敏度（spec FR-3；手感调整只改这里）----
const ROT_SPEED: f32 = 0.005; // rad/px（右键拖拽旋转）
pub(crate) const ZOOM_LOG_SPEED: f32 = 0.35; // /行（滚轮乘法缩放，各距离档手感一致）

/// 轨道相机输入（P2.6 spec FR-3/FR-4）：
/// - 右键拖拽 = 旋转（yaw -= dx·ROT_SPEED, pitch += dy·ROT_SPEED）
/// - 中键拖拽 = 平移 target（按当前距离缩放 pan 速度，1:1 跟手）
/// - 滚轮 = 对数缩放（exp(±ZOOM_LOG_SPEED·lines)，各距离档手感一致）+ Shift 细调 1/10
/// - 拖拽类互斥（旋转 > 平移），滚轮可与拖拽共存
/// - 末尾同帧重建 DdaCameraConfig（from_orbit 唯一矩阵构造点）→ ≤1 帧生效
pub(crate) fn orbit_camera_input(
  mouse: Res<ButtonInput<MouseButton>>,
  keys: Res<ButtonInput<KeyCode>>,
  motion: Res<AccumulatedMouseMotion>,
  scroll: Res<AccumulatedMouseScroll>,
  captured: Res<gate_ui::UiPointerCaptured>,
  windows: Query<&Window>,
  mut orbit: ResMut<OrbitCamera>,
  mut cfg: ResMut<DdaCameraConfig>,
) {
  // UI 指针捕获优先（2.7a FR-6）：hover/按下控件时吞掉拖拽/滚轮；
  // 但 cfg 重建在 gate 之外——resize 期间即便指针在 UI 上，aspect 也要跟上
  if !captured.0 {
    let delta = motion.delta;
    let rotating = mouse.pressed(MouseButton::Right);
    let panning = mouse.pressed(MouseButton::Middle);

    if rotating {
      orbit.yaw -= delta.x * ROT_SPEED;
      orbit.pitch += delta.y * ROT_SPEED;
      orbit.clamp();
    } else if panning {
      // 正交基：forward = eye→target；right = forward × Y；up = right × forward
      // （与 look_at_rh 的 xaxis/yaxis 同构；pitch ±89° clamp 保证 forward 不与 Y 共线）
      let (sin_yaw, cos_yaw) = orbit.yaw.sin_cos();
      let (sin_pitch, cos_pitch) = orbit.pitch.sin_cos();
      let forward = -Vec3::new(sin_yaw * cos_pitch, sin_pitch, cos_yaw * cos_pitch);
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

  // 每帧无条件重建矩阵（幂等；成本 = 一次 4×4 求逆，可忽略），省 dirty 标记。
  // aspect 读当前窗口物理尺寸（FR-5：resize 后 ≤1 帧生效，2.6 的恒定假设解除）
  let Ok(window) = windows.single() else { return };
  let size = UVec2::new(
    window.physical_width().max(1),
    window.physical_height().max(1),
  );
  *cfg = DdaCameraConfig::from_orbit(
    &orbit,
    FOV_Y,
    size.x as f32 / size.y as f32,
    CAM_NEAR,
    CAM_FAR,
  );
}

/// 主窗口物理高度（pan_per_px 1:1 基准；无窗口时回退 VIEW_SIZE.y）
fn window_height(windows: &Query<&Window>) -> f32 {
  windows
    .single()
    .map(|w| w.physical_height().max(1) as f32)
    .unwrap_or(VIEW_SIZE.y as f32)
}

/// 左键点击：把旋转中心（OrbitCamera.target）搬到点击像素命中的体素位置。
/// - 未命中任何体素 / 物体 → 不做操作。
/// - 命中点用射线入点 fine 坐标（命中面外侧向内偏半个 fine，避免 target 贴着面导致
///   距离过近时 pitch clamp 抖动）。
/// - UI 捕获指针（UI 控件上点击）时跳过，避免误触发。
/// - CPU picking：用 `cpu_reference_trace_volumes` 同步跑主世界+物体两级 DDA，
///   复用 DdaCameraConfig.inv_view_proj 反投影构造射线（origin=相机、dir=命中像素 far）。
#[allow(clippy::too_many_arguments)] // 多资源 = 点击成本可接受
pub(crate) fn left_click_pick_recenter(
  mouse: Res<ButtonInput<MouseButton>>,
  captured: Res<gate_ui::UiPointerCaptured>,
  windows: Query<&Window>,
  cfg: Res<DdaCameraConfig>,
  scene: Option<Res<VoxelScene>>,
  mut orbit: ResMut<OrbitCamera>,
) {
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
  // ---- 1) 构造射线：cursor 逻辑像素 → 物理像素 → NDC → 反投影 ----
  let Some(cursor) = window.cursor_position() else {
    return; // 指针不在窗口
  };
  let sf = window.scale_factor() as f32;
  let phys = cursor * sf; // 物理像素（左上原点，y 向下）
  let pw = window.physical_width().max(1) as f32;
  let ph = window.physical_height().max(1) as f32;
  let u = (phys.x / pw) * 2.0 - 1.0; // [-1, 1]
  let v = 1.0 - (phys.y / ph) * 2.0; // [-1, 1]，翻转 y（NDC +y 朝上）
  let near = cfg.inv_view_proj * Vec4::new(u, v, 0.0, 1.0);
  let far = cfg.inv_view_proj * Vec4::new(u, v, 1.0, 1.0);
  let near = near.truncate() / near.w;
  let far = far.truncate() / far.w;
  let delta = far - near;
  let dir = delta.normalize_or_zero();
  if dir.length_squared() < 1e-20 {
    return;
  }
  let t_max = (CAM_FAR - CAM_NEAR).max(delta.length());
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
    .map(|(b, t)| (b, t))
    .collect();
  // ---- 3) trace_volumes：主世界 + 物体统一求最近 ----
  if let Some(hit) = cpu_reference_trace_volumes(&vols_with_tr, cfg.position_world, dir, t_max) {
    // 命中点 = origin + t·dir；再朝命中法线方向推半个 fine（让 target 落在体素内部）。
    let mut p = cfg.position_world + dir * hit.t;
    let half = 0.5;
    p += hit.normal * half; // 法线朝射线来向 → *+half 把点推进命中体素内 0.5 fine
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
    // 初始 distance (700-260,560-120,700-260).len() ≈ 866 fine
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
          // 极限场景 AABB 尺寸：12×5×12 tile = (5632, 2560, 5632) fine，
          // 对角穿越 worst-case ~9000 fine；给 16384 留 2× 余量避免截断漏画。
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
        "ZOOM lines={lines:>3} distance={distance:>8.0} AABB fine min=({:.0},{:.0},{:.0}) max=({:.0},{:.0},{:.0})\n  \
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
