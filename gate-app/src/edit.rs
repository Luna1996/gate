//! 体素编辑：幽灵模式下左键放置 / 右键擦除，笔触 = 形状（球/立方）× 大小（voxel）× 材质。
//! 目标选取：光标 → 世界射线（[`crate::camera::cursor_ray`]）→ 主世界体素 DDA（[`raycast_main`]）；
//! 放置落点取命中面外侧一格（Minecraft 惯例），擦除取命中格自身，笔触以该格为中心展开。
//! `set_voxel` 内部 `mark_data(chunk)` 驱动增量上传（`poll_pending` → `update_chunk` → GPU），
//! 同一 dirty AABB 同时驱动 DDGI 重烘与光照场重算；非幽灵模式不生效（左键仍是 recenter）。

use bevy::input::mouse::AccumulatedMouseMotion;
use bevy::prelude::*;
use glam::{IVec3, Vec3};

use gate_render::{DdaCameraConfig, VoxelScene};
use gate_voxel::{PaletteEntry, VolumeGrid, VoxelCoord};

use crate::camera::{CameraMode, cursor_ray};

// ============================================================================
// 笔触
// ============================================================================

/// 笔触形状
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum BrushShape {
  /// 球（size = 1 时退化为单格）
  #[default]
  Sphere,
  /// 正方体
  Cube,
}

/// 笔触大小范围（voxel）：1 = 单格，N = (2N-1)³ 的跨度
pub const EDIT_SIZE_MIN: u32 = 1;
pub const EDIT_SIZE_MAX: u32 = 16;

/// 编辑"手长"（voxel）：射线超过这个距离不算命中（1 voxel = 2cm → 256 ≈ 5.1m）
pub const EDIT_REACH: f32 = 256.0;

/// 右键「点击 vs 拖拽转头」的累计位移阈值（物理像素）
const DRAG_PX: f32 = 4.0;

// ============================================================================
// 材质预设
// ============================================================================

/// 编辑材质预设（颜色 sRGB + 自发光 0..255）；字段经 [`EditMaterial::entry`] 落进 `PaletteEntry`。
#[derive(Clone, Copy, Debug)]
pub struct EditMaterial {
  pub name: &'static str,
  pub color: [u8; 3],
  pub emissive: u8,
}

/// 材质表（Edit tab 的滑杆顺序 = 本表顺序）
pub const EDIT_MATERIALS: [EditMaterial; 6] = [
  EditMaterial { name: "Grey", color: [150, 152, 158], emissive: 0 },
  EditMaterial { name: "Red", color: [196, 62, 58], emissive: 0 },
  EditMaterial { name: "Green", color: [74, 178, 92], emissive: 0 },
  EditMaterial { name: "Blue", color: [70, 118, 214], emissive: 0 },
  EditMaterial { name: "Yellow", color: [232, 198, 72], emissive: 0 },
  // 自发光：发光体素着色直出并经 GI 传播（shaders/voxel_raytrace/common.wesl palette_emissive）
  EditMaterial { name: "Lamp", color: [255, 238, 196], emissive: 200 },
];

impl EditMaterial {
  /// 落进调色板的条目（粗糙度给吃光的 200；目前着色只消费 color / emissive）
  pub fn entry(&self) -> PaletteEntry {
    let mut e = PaletteEntry::default();
    e.color = self.color;
    e.roughness = 200;
    e.emissive = self.emissive;
    e
  }
}

// ============================================================================
// 设置资源（UI 是它的视图）
// ============================================================================

/// 编辑设置（main world Resource）
#[derive(Resource, Clone, Copy, Debug)]
pub struct EditSettings {
  pub shape: BrushShape,
  /// 笔触大小（voxel）
  pub size: u32,
  /// 选中材质（[`EDIT_MATERIALS`] 下标）
  pub material: usize,
  /// 材质 → 调色板槽（首次使用时分配，见 [`ensure_material`]）
  pub slots: [Option<u8>; EDIT_MATERIALS.len()],
}

impl Default for EditSettings {
  fn default() -> Self {
    Self { shape: BrushShape::Sphere, size: 3, material: 0, slots: [None; EDIT_MATERIALS.len()] }
  }
}

/// 给指定材质取调色板槽（懒分配，结果缓存在 `settings.slots`）。
/// 空槽判据 = 条目全零（场景 palette 从索引 1 起占用）；一个空槽都没有 → 回退 255 并 warn。
fn material_slot(grid: &mut VolumeGrid, settings: &mut EditSettings, idx: usize) -> u8 {
  if let Some(s) = settings.slots[idx] {
    return s;
  }
  let empty = PaletteEntry::default();
  let free = (1u16..255).find(|&i| *grid.palette().get(i as u8) == empty).map(|i| i as u8);
  let slot = match free {
    Some(s) => s,
    None => {
      bevy::log::warn!("编辑材质找不到空调色板槽，复用 255（会覆盖该槽原有材质）");
      255
    }
  };
  settings.slots[idx] = Some(slot);
  slot
}

/// 确保材质已写进它的槽（幂等：内容相同就不写，避免无谓的 palette 重铺）
fn ensure_material(grid: &mut VolumeGrid, settings: &mut EditSettings, idx: usize) -> u8 {
  let slot = material_slot(grid, settings, idx);
  let want = EDIT_MATERIALS[idx].entry();
  if *grid.palette().get(slot) != want {
    grid.palette_mut().set(slot, want);
  }
  slot
}

// ============================================================================
// 射线 × 体素 DDA
// ============================================================================

/// 世界空间射线 × 体素的 Amanatides-Woo 步进。返回 `(命中体素, 入面法线, 命中 t)`。
///
/// 入面法线指向射线来向（朝外），故 `命中体素 + 法线` 即前方那格空气 —— 放置落点。
/// `dir` 分量可以为 0（`1.0/0.0 = inf`，该轴不会被选中）；仅主世界（identity 变换）。
pub fn raycast_main(
  grid: &VolumeGrid,
  origin: Vec3,
  dir: Vec3,
  max_dist: f32,
) -> Option<(IVec3, IVec3, f32)> {
  let mut v = IVec3::new(origin.x.floor() as i32, origin.y.floor() as i32, origin.z.floor() as i32);
  let step = IVec3::new(
    if dir.x >= 0.0 { 1 } else { -1 },
    if dir.y >= 0.0 { 1 } else { -1 },
    if dir.z >= 0.0 { 1 } else { -1 },
  );
  let inv = Vec3::new(1.0 / dir.x, 1.0 / dir.y, 1.0 / dir.z);
  // 到下一个边界轴的参数距离，以及每跨一格的增量
  let mut t_max = Vec3::new(
    ((v.x + if step.x > 0 { 1 } else { 0 }) as f32 - origin.x) * inv.x,
    ((v.y + if step.y > 0 { 1 } else { 0 }) as f32 - origin.y) * inv.y,
    ((v.z + if step.z > 0 { 1 } else { 0 }) as f32 - origin.z) * inv.z,
  );
  let t_delta = Vec3::new(inv.x.abs(), inv.y.abs(), inv.z.abs());
  let mut face = IVec3::ZERO; // 起点格的入面无定义（相机嵌在固体里）→ 零向量
  let mut t = 0.0f32;
  // 步数上限：三轴各走 max_dist 格的上界，防退化射线空转
  let max_steps = (3.0 * max_dist) as u32 + 3;
  for _ in 0..max_steps {
    if grid.get_voxel(VoxelCoord::from_ivec3(v)).unwrap_or(0) != 0 {
      return Some((v, face, t));
    }
    let axis = if t_max.x <= t_max.y && t_max.x <= t_max.z {
      0
    } else if t_max.y <= t_max.z {
      1
    } else {
      2
    };
    t = t_max[axis];
    if t > max_dist {
      return None;
    }
    v[axis] += step[axis];
    t_max[axis] += t_delta[axis];
    face = IVec3::ZERO;
    face[axis] = -step[axis]; // 入面朝来向：+x 步进 → 入面法线 -x
  }
  None
}

// ============================================================================
// 笔触施加
// ============================================================================

/// 以 `center` 为中心施加一次笔触，返回实际改变的体素数。
/// `palette == 0` → 擦除（挖空）；否则只填充空气格（Minecraft 惯例：不啃掉已有几何）。
/// 球判据 `d² ≤ r² + r`（r = size-1）：r=0 → 仅中心格；r=1 → 3³ 去掉 8 个角。
pub fn apply_brush(
  grid: &mut VolumeGrid,
  center: IVec3,
  shape: BrushShape,
  size: u32,
  palette: u8,
) -> usize {
  let r = size.saturating_sub(1) as i32;
  let mut changed = 0usize;
  for dz in -r..=r {
    for dy in -r..=r {
      for dx in -r..=r {
        if shape == BrushShape::Sphere && dx * dx + dy * dy + dz * dz > r * r + r {
          continue;
        }
        let p = center + IVec3::new(dx, dy, dz);
        let cur = grid.get_voxel(VoxelCoord::from_ivec3(p)).unwrap_or(0);
        if palette == 0 {
          if cur == 0 {
            continue; // 擦除：本来就是空气
          }
        } else if cur != 0 {
          continue; // 放置：不覆盖已有几何
        }
        if grid.set_voxel_ivec3(p, palette).is_some() {
          changed += 1;
        }
      }
    }
  }
  changed
}

// ============================================================================
// 输入系统
// ============================================================================

/// 体素编辑输入（仅幽灵模式；轨道模式左键仍是 recenter）。
/// - 左键 = 放置（当前形状/大小/材质）
/// - 右键 = 擦除；按下到释放累计位移 > [`DRAG_PX`] 视为「拖拽转头」，不编辑
#[allow(clippy::too_many_arguments)] // Bevy system：输入/资源逐一注入
pub(crate) fn voxel_edit_input(
  mouse: Res<ButtonInput<MouseButton>>,
  motion: Res<AccumulatedMouseMotion>,
  captured: Res<gate_ui::UiPointerCaptured>,
  windows: Query<&Window>,
  cfg: Res<DdaCameraConfig>,
  mode: Res<CameraMode>,
  mut settings: ResMut<EditSettings>,
  scene: Option<ResMut<VoxelScene>>,
  mut right_drag_px: Local<f32>,
) {
  // 右键位移累计（区分点击/拖拽）；两种模式都累计，避免切模式后残留旧值
  if mouse.just_pressed(MouseButton::Right) {
    *right_drag_px = 0.0;
  }
  if mouse.pressed(MouseButton::Right) {
    *right_drag_px += motion.delta.length();
  }
  if *mode != CameraMode::Fly || captured.0 {
    return;
  }
  let place = mouse.just_pressed(MouseButton::Left);
  let erase = mouse.just_released(MouseButton::Right) && *right_drag_px < DRAG_PX;
  if !place && !erase {
    return;
  }
  let Some(mut scene) = scene else { return };
  let Ok(window) = windows.single() else { return };
  let Some((origin, dir)) = cursor_ray(window, &cfg) else {
    return;
  };
  let mat_idx = settings.material.min(EDIT_MATERIALS.len() - 1);
  let (shape, size) = (settings.shape, settings.size);
  let grid = scene.volumes.main_mut();
  let Some((hit, face, _t)) = raycast_main(grid, origin, dir, EDIT_REACH) else {
    return;
  };
  let (center, pal) = if erase {
    (hit, 0u8)
  } else {
    // 放置落点 = 命中面外侧一格
    let slot = ensure_material(grid, &mut settings, mat_idx);
    (hit + face, slot)
  };
  let changed = apply_brush(grid, center, shape, size, pal);
  if changed > 0 {
    bevy::log::info!(
      "EDIT[{}]: {} voxel(s) @ ({},{},{}) shape={:?} size={} material={}",
      if erase { "erase" } else { "place" },
      changed,
      center.x,
      center.y,
      center.z,
      shape,
      size,
      if erase { "-" } else { EDIT_MATERIALS[mat_idx].name },
    );
  }
}

/// `GATE_EDIT_SELFTEST=1`：第 60 帧朝初始注视点刷一次笔触，在没有鼠标输入的情况下走通整条
/// 编辑链路（`set_voxel` → `mark_data` → 增量上传 → DDGI 重烘 / 光照场重算）。
/// 只在设了该变量时才注册（见 main.rs），正常运行零开销。
pub(crate) fn edit_selftest(
  scene: Option<ResMut<VoxelScene>>,
  orbit: Res<gate_render::OrbitCamera>,
  mut settings: ResMut<EditSettings>,
  mut frame: Local<u32>,
) {
  *frame += 1;
  if *frame != 60 {
    return;
  }
  let Some(mut scene) = scene else { return };
  let dir = (orbit.target - orbit.eye()).normalize_or_zero();
  if dir.length_squared() < 1e-12 {
    bevy::log::warn!("EDIT SELFTEST: 相机朝向退化，跳过");
    return;
  }
  let mat = settings.material.min(EDIT_MATERIALS.len() - 1);
  let (shape, size) = (settings.shape, settings.size);
  let grid = scene.volumes.main_mut();
  let Some((hit, face, t)) = raycast_main(grid, orbit.eye(), dir, EDIT_REACH) else {
    bevy::log::warn!("EDIT SELFTEST: 射线未命中（{EDIT_REACH} 内无体素），跳过");
    return;
  };
  let slot = ensure_material(grid, &mut settings, mat);
  let center = hit + face;
  let n = apply_brush(grid, center, shape, size, slot);
  bevy::log::info!(
    "EDIT SELFTEST: hit=({},{},{}) t={t:.1} → placed {n} voxel(s) @ ({},{},{}) shape={:?} size={size} slot={slot}",
    hit.x,
    hit.y,
    hit.z,
    center.x,
    center.y,
    center.z,
    shape,
  );
}

#[cfg(test)]
mod tests {
  use super::*;
  use gate_voxel::fill_box;

  /// 1 体素厚的地板：y ∈ [0,1)，x/z ∈ [-64,64)
  fn grid_with_floor() -> VolumeGrid {
    let mut g = VolumeGrid::new();
    fill_box(&mut g, IVec3::new(-64, 0, -64), IVec3::new(128, 1, 128), 1);
    g
  }

  #[test]
  fn raycast_hits_floor_top_face() {
    let g = grid_with_floor();
    let (v, n, t) = raycast_main(&g, Vec3::new(0.5, 10.5, 0.5), Vec3::new(0.0, -1.0, 0.0), 64.0)
      .expect("应命中地板");
    assert_eq!(v, IVec3::new(0, 0, 0), "命中格");
    assert_eq!(n, IVec3::Y, "入面法线应朝射线来向（+Y）");
    assert!((t - 9.5).abs() < 1e-3, "命中距离 t={t}（应为 10.5-1.0）");
    // 放置落点 = 命中格 + 法线 = 地板上方那格空气
    assert_eq!(v + n, IVec3::new(0, 1, 0));
  }

  #[test]
  fn raycast_respects_reach_and_feature_absence() {
    let g = grid_with_floor();
    // 手长不够 → 未命中
    assert!(
      raycast_main(&g, Vec3::new(0.5, 1000.5, 0.5), Vec3::new(0.0, -1.0, 0.0), 64.0).is_none(),
      "超过 hand reach 不该命中"
    );
    // 朝天空打 → 未命中
    assert!(raycast_main(&g, Vec3::new(0.5, 10.5, 0.5), Vec3::Y, 64.0).is_none(), "朝天空不该命中");
  }

  #[test]
  fn brush_shape_size_and_air_only_placement() {
    // 立方体：size=3 → 5³ 全中
    let mut cube = VolumeGrid::new();
    assert_eq!(apply_brush(&mut cube, IVec3::ZERO, BrushShape::Cube, 3, 7), 125);
    // 球：同样跨度但少（去掉 8 个角）
    let mut sphere = VolumeGrid::new();
    let n = apply_brush(&mut sphere, IVec3::ZERO, BrushShape::Sphere, 3, 7);
    assert!(n > 0 && n < 125, "球体素应少于等跨度立方体：{n}");
    // size=1 → 单格（两种形状一致）
    let mut one = VolumeGrid::new();
    assert_eq!(apply_brush(&mut one, IVec3::ZERO, BrushShape::Sphere, 1, 7), 1);
    assert_eq!(apply_brush(&mut one, IVec3::ZERO, BrushShape::Cube, 1, 7), 0, "同格重复放置不改变");
    // 放置只填空气：往已有几何上刷不再改
    let mut on_solid = VolumeGrid::new();
    apply_brush(&mut on_solid, IVec3::ZERO, BrushShape::Cube, 2, 3);
    assert_eq!(
      apply_brush(&mut on_solid, IVec3::ZERO, BrushShape::Cube, 2, 9),
      0,
      "刷子不该啃掉已有几何"
    );
    // 擦除：把它清空
    assert_eq!(apply_brush(&mut on_solid, IVec3::ZERO, BrushShape::Cube, 2, 0), 27);
    assert_eq!(on_solid.get_voxel(VoxelCoord::from_ivec3(IVec3::ZERO)), None);
  }

  #[test]
  fn material_slot_allocates_distinct_free_slots() {
    // 场景占用索引 1..=3（其余为空）
    let mut g = VolumeGrid::new();
    for i in 1..=3u8 {
      let mut e = PaletteEntry::default();
      e.color = [i, i, i];
      g.palette_mut().set(i, e);
    }
    let mut s = EditSettings::default();
    let a = ensure_material(&mut g, &mut s, 0);
    let b = ensure_material(&mut g, &mut s, 1);
    assert_ne!(a, b, "不同材质拿到不同槽");
    assert!(a > 3 && b > 3, "应避开场景占用的 1..=3：{a} {b}");
    assert_eq!(ensure_material(&mut g, &mut s, 0), a, "同一材质槽稳定");
    // 写入的条目与预设一致（颜色 + 自发光）
    assert_eq!(g.palette().get(a).color, EDIT_MATERIALS[0].color);
    assert_eq!(g.palette().get(b).emissive, EDIT_MATERIALS[1].emissive);
  }
}
