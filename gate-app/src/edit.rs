//! 体素编辑：幽灵模式下左键放置 / 右键擦除，笔触 = 形状（球/立方）× 大小（voxel）× 材质。
//! 目标选取：光标 → 世界射线（[`crate::camera::cursor_ray`]）→ 主世界体素 DDA（[`raycast_main`]）；
//! 放置落点取命中面外侧一格（Minecraft 惯例），擦除取命中格自身，笔触以该格为中心展开。
//! `set_voxel` 内部 `mark_data(chunk)` 驱动增量上传（`poll_pending` → `update_chunk` → GPU），
//! 同一 dirty AABB 同时驱动 DDGI 重烘与光照场重算；非幽灵模式不生效（左键仍是 recenter）。

use bevy::input::mouse::AccumulatedMouseMotion;
use bevy::prelude::*;
use glam::{IVec3, Vec3};

use gate_render::{DdaCameraConfig, VoxelScene};
use gate_voxel::{
  BRICK_FACTOR, BrickState, LEVEL_EXTENT, PALETTE_INDEX_MAX, PaletteEntry, PaletteId, VolumeGrid,
  VoxelCoord,
};

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

/// 笔触大小下界（voxel）：1 = 单格，N = (2N-1) 的跨度。
/// **无上界**：写入按 brick 整块进行（见 [`apply_brush`]），实际可用规模由写入体量与内存决定。
pub const EDIT_SIZE_MIN: u32 = 1;

/// 编辑"手长"（voxel）：射线超过这个距离不算命中（1 voxel = 2cm → 256 ≈ 5.1m）
pub const EDIT_REACH: f32 = 256.0;

/// 右键「点击 vs 拖拽转头」的累计位移阈值（物理像素）
const DRAG_PX: f32 = 4.0;

// ============================================================================
// 笔触材质
// ============================================================================

/// 笔触材质参数：菜单「游戏/编辑」的四个控件（颜色 / 自发光 / 透明度 / 光滑度）直接写这里，
/// 由 [`ensure_material`] 落进调色板槽。字段与 `PaletteEntry` 一一对应，数值域按 UI 友好度收窄。
///
/// 说明：之前这里是 6 个写死的预设（`EDIT_MATERIALS`）+ 一个「选中材质」索引，但那个索引
/// **从未被 UI 写过**（永远是 0 = Grey），所以预设实际只有 Grey 可达；颜色也改不了。
/// 现在改为直接由菜单驱动一份材质参数。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BrushMaterial {
  /// sRGB [r, g, b]
  pub color: [u8; 3],
  /// 自发光强度 0..255（0 = 不发光；发光体素着色直出并经 GI 传播）
  pub emissive: u8,
  /// 透射率 0..255（0 = 不透明，255 = 全透）
  pub transmission: u8,
  /// 粗糙度 0..255（0 = 镜面，255 = 完全粗糙）
  pub roughness: u8,
}

impl Default for BrushMaterial {
  /// 与 `assets/ui/debug_menu.toml` 的初值保持一致（中灰 / 不发光 / 不透明 / 半粗糙），
  /// 这样 TOML 缺失时观感不变。
  fn default() -> Self {
    Self { color: [0x96, 0x98, 0x9E], emissive: 0, transmission: 0, roughness: 128 }
  }
}

impl BrushMaterial {
  /// 落进调色板的条目（`flags` 留给后续语义位，暂不消费）
  pub fn entry(&self) -> PaletteEntry {
    let mut e = PaletteEntry::default();
    e.color = self.color;
    e.roughness = self.roughness;
    e.emissive = self.emissive;
    e.transmission = self.transmission;
    e
  }

  /// 日志用的 `#RRGGBB`
  pub fn hex(&self) -> String {
    format!("#{:02X}{:02X}{:02X}", self.color[0], self.color[1], self.color[2])
  }
}

/// 菜单「透明度」滑杆 0..100 → `PaletteEntry.transmission`。
///
/// 约定 **100 = 完全不透明**、0 = 全透。取这个方向是因为菜单初值就是 100，
/// 若把 100 当"全透"则默认笔触一放下去就是看不见的。
pub fn opacity_pct_to_transmission(pct: f32) -> u8 {
  (((100.0 - pct.clamp(0.0, 100.0)) / 100.0) * 255.0).round() as u8
}

/// 菜单「光滑度」滑杆 0..100 → `PaletteEntry.roughness`（0 = 镜面、255 = 完全粗糙）。
pub fn smooth_pct_to_roughness(pct: f32) -> u8 {
  (((100.0 - pct.clamp(0.0, 100.0)) / 100.0) * 255.0).round() as u8
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
  /// 当前笔触材质参数（由菜单「游戏/编辑」的四个控件驱动）
  pub mat: BrushMaterial,
}

impl Default for EditSettings {
  fn default() -> Self {
    Self { shape: BrushShape::Sphere, size: 3, mat: BrushMaterial::default() }
  }
}

/// 取当前笔触材质的调色板槽：**按内容去重** —— 已有同内容的槽就复用，否则认领一个新空槽。
///
/// 这就是"每次落笔时判断需不需要建新的 palette 值"的语义。反过来（把唯一一个槽的内容
/// 反复改写）会让**此前用同一槽放下的体素跟着变色**：改一次颜色，整片旧体素一起变，
/// 那不是编辑器的行为。
///
/// 顺序：先在**已占用**槽里找内容一致的（去重），再挑一个空槽写入；一个空槽都没有 →
/// 复用 `PALETTE_INDEX_MAX` 并 warn（会覆盖该槽原有材质）。空槽判据见 `Palette::is_empty_slot`
/// （占用位图，与内容无关）。
///
/// 代价：每次落笔线性扫一遍 65536 槽（同一趟里顺便找空槽，两者都找到即提前退出）。
/// 落笔是手动点击级频率，且"材质 → 槽"的缓存会随场景重建失效、需要额外失效逻辑，故不做缓存。
fn material_slot(grid: &mut VolumeGrid, mat: BrushMaterial) -> PaletteId {
  let want = mat.entry();
  let mut existing = None;
  let mut free = None;
  {
    let pal = grid.palette();
    for i in 1u16..=PALETTE_INDEX_MAX {
      let id = PaletteId(i);
      if pal.occupied(id) {
        if existing.is_none() && *pal.get(id) == want {
          existing = Some(id);
        }
      } else if free.is_none() {
        free = Some(id);
      }
      if existing.is_some() && free.is_some() {
        break;
      }
    }
  }
  if let Some(slot) = existing {
    return slot;
  }
  let slot = match free {
    Some(s) => s,
    None => {
      bevy::log::warn!("编辑材质找不到空调色板槽，复用 {PALETTE_INDEX_MAX}（会覆盖该槽原有材质）");
      PaletteId(PALETTE_INDEX_MAX)
    }
  };
  grid.palette_mut().set(slot, want);
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
    if !grid.get_voxel(VoxelCoord::from_ivec3(v)).unwrap_or(PaletteId::AIR).is_air() {
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

/// 单格是否落在笔触区域内（`d` = 该格相对中心的有符号偏移）。
fn brush_contains(shape: BrushShape, d: IVec3, r: i32) -> bool {
  match shape {
    BrushShape::Cube => d.x.abs() <= r && d.y.abs() <= r && d.z.abs() <= r,
    // 球判据 d² ≤ r² + r（r = size-1）：r=0 → 仅中心格；r=1 → 3³ 去掉 8 个角
    BrushShape::Sphere => d.length_squared() <= r.saturating_mul(r).saturating_add(r),
  }
}

/// 对齐块 `[lo, lo+extent)` 是否**完全不在**笔触区域内（可整块剪枝）。
///
/// 笔触区域是凸的（立方体 / 球），且"盒到中心的最小距离点"都是把中心 clamp 进盒
/// （切比雪夫度量 / 欧氏度量各自成立）→ 最近点已在区域外 ⇒ 整块在区域外。
fn brush_box_disjoint(shape: BrushShape, center: IVec3, r: i32, lo: IVec3, extent: i32) -> bool {
  let hi = lo + IVec3::splat(extent - 1);
  !brush_contains(shape, center.clamp(lo, hi) - center, r)
}

/// 对齐块是否**完全在**笔触区域内（凸区域包含一个盒 ⟺ 包含它的 8 个角）
fn brush_box_inside(shape: BrushShape, center: IVec3, r: i32, lo: IVec3, extent: i32) -> bool {
  let e = extent - 1;
  for i in 0..8 {
    let c = IVec3::new(
      if i & 1 == 0 { 0 } else { e },
      if i & 2 == 0 { 0 } else { e },
      if i & 4 == 0 { 0 } else { e },
    );
    if !brush_contains(shape, lo + c - center, r) {
      return false;
    }
  }
  true
}

/// 笔触的层级填充（`lo` = 对齐到 `extent` 的块最小角）。
///
/// **整块落在笔触内**的块一次写掉（[`VolumeGrid::fill_brick`]：O(深度) 树路径，且直接产出
/// uniform 的上级节点）；只有边界上「部分覆盖」的块才下钻一级，最小到 1³ 才逐体素。
/// 对照逐体素 `set_voxel`：每一格都要走一次完整树下降 + 沿途 `try_merge`（每次扫 64 个子块、
/// 紧凑表 memmove），31³ 笔触近 3 万次；而整块写一次顶 4³/16³/64³ 格。
///
/// `palette` 为 AIR 即擦除；语义与逐体素版一致（放置只填空气、擦除只挖实体）。
fn fill_brush_level(
  grid: &mut VolumeGrid,
  shape: BrushShape,
  center: IVec3,
  r: i32,
  lo: IVec3,
  extent: i32,
  palette: PaletteId,
  changed: &mut usize,
) {
  if brush_box_disjoint(shape, center, r, lo, extent) {
    return;
  }
  let erase = palette.is_air();
  if extent > 1 && brush_box_inside(shape, center, r, lo, extent) {
    // 整块都在笔触内 → 按 brick 三态决定能否一次写完：
    //   空气 + 放置 / 同色实体 + 擦除 → 整块写（一次 O(深度)，落成 uniform 上级节点）
    //   空气 + 擦除、实体 + 放置 → 本块无需改动（不啃掉已有几何）
    //   内容不一致 → 下钻
    match grid.get_brick_state_extent(lo, extent) {
      BrickState::Air => {
        if !erase {
          grid.fill_brick(lo, extent, palette);
          *changed += (extent as usize).pow(3);
        }
        return;
      }
      BrickState::Solid(_) => {
        if erase {
          grid.fill_brick(lo, extent, palette);
          *changed += (extent as usize).pow(3);
        }
        return;
      }
      BrickState::Mixed => {}
    }
  }
  if extent == 1 {
    // 收尾单格：extent==1 的「部分覆盖」就是完全覆盖，故这里不再查 brick 三态
    if brush_contains(shape, lo - center, r) {
      let cur = grid.get_voxel(VoxelCoord::from_ivec3(lo)).unwrap_or(PaletteId::AIR);
      if cur.is_air() != erase && grid.set_voxel_ivec3(lo, palette).is_some() {
        *changed += 1;
      }
    }
    return;
  }
  let sub = extent / BRICK_FACTOR;
  for i in 0..BRICK_FACTOR * BRICK_FACTOR * BRICK_FACTOR {
    let d = IVec3::new(
      i % BRICK_FACTOR,
      (i / BRICK_FACTOR) % BRICK_FACTOR,
      i / (BRICK_FACTOR * BRICK_FACTOR),
    );
    fill_brush_level(grid, shape, center, r, lo + d * sub, sub, palette, changed);
  }
}

/// 以 `center` 为中心施加一次笔触，返回实际改变的体素数。
/// `palette == 0` → 擦除（挖空）；否则只填充空气格（Minecraft 惯例：不啃掉已有几何）。
/// 球判据 `d² ≤ r² + r`（r = size-1）：r=0 → 仅中心格；r=1 → 3³ 去掉 8 个角。
///
/// 写入自顶向下按 brick 粒度进行（见 [`fill_brush_level`]）：能整块写的绝不逐体素，
/// 且整块写直接落成 uniform 上级节点，不依赖事后合并。
pub fn apply_brush(
  grid: &mut VolumeGrid,
  center: IVec3,
  shape: BrushShape,
  size: u32,
  palette: PaletteId,
) -> usize {
  // size 无上限（菜单可输入任意值），这里只做「不溢出 i32」的类型收敛
  let r = size.saturating_sub(1).min(i32::MAX as u32) as i32;
  let mut changed = 0usize;
  // 起始层级：整块要能装进笔触（extent ≤ 跨度 2r+1），取满足的最大 brick 粒度。
  // 小笔触因此直接落到 4³/1³，不会做无谓的粗层下钻。
  let span = r.saturating_mul(2).saturating_add(1);
  let start = LEVEL_EXTENT.iter().copied().find(|&e| e <= span).unwrap_or(1);
  // 覆盖笔触 AABB 的全部对齐块。世界对齐 ⇒ 块必然整个落在单个 chunk 内
  // （chunk 边长 256 是各 brick 粒度的整数倍），满足 fill_brick 的对齐与边界约束。
  let s = IVec3::splat(start);
  let b_lo = center.saturating_sub(IVec3::splat(r)).div_euclid(s) * s;
  let b_hi = center.saturating_add(IVec3::splat(r)).div_euclid(s) * s;
  let mut x = b_lo.x;
  while x <= b_hi.x {
    let mut y = b_lo.y;
    while y <= b_hi.y {
      let mut z = b_lo.z;
      while z <= b_hi.z {
        fill_brush_level(grid, shape, center, r, IVec3::new(x, y, z), start, palette, &mut changed);
        z += start;
      }
      y += start;
    }
    x += start;
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
  intercepted: Res<gate_ui::MouseIntercepted>,
  windows: Query<&Window>,
  cfg: Res<DdaCameraConfig>,
  mode: Res<CameraMode>,
  settings: Res<EditSettings>,
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
  if *mode != CameraMode::Fly || captured.0 || intercepted.0 {
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
  let (shape, size) = (settings.shape, settings.size);
  let grid = scene.volumes.main_mut();
  let Some((hit, face, _t)) = raycast_main(grid, origin, dir, EDIT_REACH) else {
    return;
  };
  let (center, pal) = if erase {
    (hit, PaletteId::AIR)
  } else {
    // 放置落点 = 命中面外侧一格；槽位按当前材质**内容**取/建（参数变了就是新材质，旧体素不受影响）
    let slot = material_slot(grid, settings.mat);
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
      if erase { "-".to_string() } else { settings.mat.hex() },
    );
  }
}

/// `GATE_EDIT_SELFTEST=1`：第 60 帧朝初始注视点刷一次笔触，在没有鼠标输入的情况下走通整条
/// 编辑链路（`set_voxel` → `mark_data` → 增量上传 → DDGI 重烘 / 光照场重算）。
/// 只在设了该变量时才注册（见 main.rs），正常运行零开销。
pub(crate) fn edit_selftest(
  scene: Option<ResMut<VoxelScene>>,
  orbit: Res<gate_render::OrbitCamera>,
  settings: Res<EditSettings>,
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
  let (shape, size) = (settings.shape, settings.size);
  let grid = scene.volumes.main_mut();
  let Some((hit, face, t)) = raycast_main(grid, orbit.eye(), dir, EDIT_REACH) else {
    bevy::log::warn!("EDIT SELFTEST: 射线未命中（{EDIT_REACH} 内无体素），跳过");
    return;
  };
  let slot = material_slot(grid, settings.mat);
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
