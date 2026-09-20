//! 体素编辑：幽灵模式下左键放置 / 右键擦除，笔触 = 形状（球/立方）× 大小（voxel）× 材质。
//! 目标选取：光标 → 世界射线（`crate::camera::cursor_ray`）→ 主世界体素 DDA（`raycast_main`）；
//! 放置落点 = 命中面外侧一格（擦除取命中格自身），写经 `set_voxel` → `mark_data(chunk)` 驱动增量上传。

use bevy::input::mouse::AccumulatedMouseMotion;
use bevy::prelude::*;
use glam::{IVec3, Vec3};

use gate_render::brickmap::wire::pack_palette_entry;
use gate_render::{DdaCameraConfig, VoxelScene};
use gate_voxel::{
  BRICK_FACTOR, BrickState, LEVEL_EXTENT, PALETTE_INDEX_MAX, PaletteEntry, PaletteFlags, PaletteId,
  PbrOverrides, VolumeGrid, VoxelCoord, override_value,
};

use crate::{
  camera::{CameraMode, cursor_ray},
  consts::{DRAG_PX, EDIT_REACH},
};

/// 笔触形状
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum BrushShape {
  /// 球（size = 1 时退化为单格）
  #[default]
  Sphere,
  Cube,
}

/// 笔触材质参数：菜单「游戏/编辑」的控件直接写这里，由 `material_slot` 落进调色板槽。
///
/// **两个变体（PLAN D1 的 8B 变体复用）**，与 `PaletteEntry` 一一对应：
/// - `pbr = false`（默认）：`color` / `emissive` / `transmission` / `roughness` 就是平凡 payload
///   （字节布局与改动前逐位相同）；
/// - `pbr = true`：8B 被解释为 `asset_slot` + 5 个槽级覆盖 ⇒ 那四个滑杆改驱动 `*_ov` 字段
///   （`color` 不参与：D1 明确 PBR 变体不做逐实例染色，albedo 来自资产贴图）。
///
/// **`ior_x100` 是例外**：D1 规定 IOR 是**资产级**物理基值（`MaterialAsset::transmission_ior`
/// 的高 16 位），槽级只有 `specular` 覆盖 ⇒ 这 8B 里**没有** IOR 的位置，本字段只是菜单值的记录
/// （见 `summary()` 的说明），不参与 `entry()`。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BrushMaterial {
  /// sRGB [r, g, b]（**仅平凡变体**）
  pub color: [u8; 3],
  /// 自发光强度 0..255（平凡变体直接就是这个字节；发光体素着色直出并经 GI 传播）
  pub emissive: u8,
  /// 透射率 0..255（0 = 不透明，255 = 全透；平凡变体直接就是这个字节）
  pub transmission: u8,
  /// 粗糙度 0..255（0 = 镜面，255 = 完全粗糙；平凡变体直接就是这个字节）
  pub roughness: u8,
  /// `IS_PBR` 变体开关（菜单「PBR 变体」）
  pub pbr: bool,
  /// PBR 变体引用的材质资产槽号（= `PbrTextureSet` 的层号 = 全局资产表下标）
  pub asset_slot: u32,
  /// PBR 变体：roughness 覆盖（0 = 不覆盖；由「光滑度」滑杆驱动）
  pub roughness_ov: u8,
  /// PBR 变体：metallic 覆盖（0 = 不覆盖；由「金属度」滑杆驱动）
  pub metallic_ov: u8,
  /// PBR 变体：emissive 覆盖（0 = 不覆盖；由「自发光」滑杆驱动）
  pub emissive_ov: u8,
  /// PBR 变体：transmission 覆盖（0 = 不覆盖；由「透明度」滑杆驱动）
  pub transmission_ov: u8,
  /// PBR 变体：specular 覆盖（0 = 不覆盖；语义同 glTF `KHR_materials_specular`，只调制电介质 F0）
  pub specular_ov: u8,
  /// 「折射率」滑杆值（IOR ×100）：100..=300 对应 1.00..3.00，默认 150 = 1.50。
  /// **属资产级参数**（见类型注释），当前只记录/回显。
  pub ior_x100: u16,
}

impl Default for BrushMaterial {
  /// 与 `assets/ui/debug_menu.toml` 初值一致
  /// （中灰 / 不发光 / 不透明 / 半粗糙 / 平凡变体 / 槽 0 / 全部覆盖"不覆盖" / IOR 1.50）。
  fn default() -> Self {
    Self {
      color: [0x96, 0x98, 0x9E],
      emissive: 0,
      transmission: 0,
      roughness: 128,
      pbr: false,
      asset_slot: 0,
      roughness_ov: 0,
      metallic_ov: 0,
      emissive_ov: 0,
      transmission_ov: 0,
      specular_ov: 0,
      ior_x100: 150,
    }
  }
}

impl BrushMaterial {
  /// 落进调色板的条目：按 `pbr` 开关产出**两种变体之一**（D1）。平凡路径与改动前完全一致
  /// （不设 `flags`）；PBR 路径走 `PaletteEntry::pbr`（构造），打包由 `wire.rs::pack_palette_entry`
  /// 按 `IS_PBR` 分派 —— 本函数**不打包**。
  pub fn entry(&self) -> PaletteEntry {
    if self.pbr { self.pbr_entry() } else { self.plain_entry() }
  }

  /// 平凡变体条目（现状语义，逐位不变）
  fn plain_entry(&self) -> PaletteEntry {
    PaletteEntry {
      color: self.color,
      roughness: self.roughness,
      emissive: self.emissive,
      transmission: self.transmission,
      ..Default::default()
    }
  }

  /// PBR 变体条目：`asset` + 5 个槽级覆盖。
  ///
  /// `TRANSMISSIVE` 位（D1：**由调用方决定**，打包函数不推断）按槽级透射覆盖判定：
  /// 覆盖字节 `>= 2` ⇔ 覆盖值 `> 0`（字节 `1` 是"覆盖为精确 0"，不是介质）——与平凡变体那条
  /// "`transmission > 0` ⇒ 介质"同义。**只看覆盖、不看资产**：当前默认资产集的标量透射率全 0
  /// （`gate-render/src/pbr_texture.rs::build_material_asset_table`），且这里读不到资产表；
  /// 将来资产真的带透射时，这里要一并看资产（登记为后续项）。
  fn pbr_entry(&self) -> PaletteEntry {
    let flags =
      if self.transmission_ov >= 2 { PaletteFlags::TRANSMISSIVE } else { PaletteFlags::default() };
    PaletteEntry::pbr(
      self.asset_slot.min(u16::MAX as u32) as u16,
      PbrOverrides {
        roughness: self.roughness_ov,
        metallic: self.metallic_ov,
        emissive: self.emissive_ov,
        transmission: self.transmission_ov,
        specular: self.specular_ov,
      },
      flags,
    )
  }

  /// 日志用的 `#RRGGBB`（只对平凡变体有意义 —— PBR 变体不做逐实例染色）
  pub fn hex(&self) -> String {
    format!("#{:02X}{:02X}{:02X}", self.color[0], self.color[1], self.color[2])
  }

  /// 日志用的一行摘要（菜单每次改动都打 `材质 → {summary}`，MT7-1 的验收之一）。
  /// 覆盖字节按 D1 的编码解回参数值（`0` = 不覆盖），避免把编码字节直接当数值给人看。
  pub fn summary(&self) -> String {
    if !self.pbr {
      return format!(
        "平凡 {} 自发光={} 透射率={} 粗糙度={}",
        self.hex(),
        self.emissive,
        self.transmission,
        self.roughness
      );
    }
    format!(
      "PBR 变体 asset={} 覆盖[粗糙度={} 金属度={} 自发光={} 透射率={} 高光度={}] IOR={:.2}\
       （IOR 是资产级参数，本槽不携带；起作用的槽级旋钮是「高光度」）",
      self.asset_slot,
      ov_text(self.roughness_ov),
      ov_text(self.metallic_ov),
      ov_text(self.emissive_ov),
      ov_text(self.transmission_ov),
      ov_text(self.specular_ov),
      self.ior_x100 as f32 / 100.0,
    )
  }
}

/// 覆盖字节 → 日志文本：`0` = 不覆盖，其余显示参数值（`(v−1)/254`）。
fn ov_text(byte: u8) -> String {
  match override_value(byte) {
    None => "不覆盖".to_string(),
    Some(v) => format!("{v:.3}"),
  }
}

/// 菜单「透明度」滑杆 0..100 → `PaletteEntry.transmission`（100 = 完全不透明，0 = 全透）。
pub fn opacity_pct_to_transmission(pct: f32) -> u8 {
  (((100.0 - pct.clamp(0.0, 100.0)) / 100.0) * 255.0).round() as u8
}

/// 菜单「光滑度」滑杆 0..100 → `PaletteEntry.roughness`（0 = 镜面、255 = 完全粗糙）。
pub fn smooth_pct_to_roughness(pct: f32) -> u8 {
  (((100.0 - pct.clamp(0.0, 100.0)) / 100.0) * 255.0).round() as u8
}

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

/// 取当前笔触材质的调色板槽：按内容去重 —— 已有同内容的槽复用，否则认领一个新空槽
/// （空槽判据见 `Palette::is_empty_slot`）；一个空槽都没有则复用 `PALETTE_INDEX_MAX` 并 warn
/// （会覆盖该槽原有材质）。单次调用线性扫全部 65536 槽。
///
/// **去重判据 = 8B payload**（`pack_palette_entry` 的输出，也就是真正上传到 GPU 的东西），
/// 不是"字段看起来一样"：它在**两个变体上都是单射**（每个字节落在固定 bit 段、无重叠、无丢弃字段，
/// 见 `PaletteEntry` 的文档）⇒ 判据等价，但把"内容"定义在字节层更硬。**PBR 变体走的是同一条
/// 去重路径**（内容 = `asset` + 5 个覆盖 + flags），不另开一条（MT7-3）。
fn material_slot(grid: &mut VolumeGrid, mat: BrushMaterial) -> PaletteId {
  let want = pack_palette_entry(&mat.entry());
  let mut existing = None;
  let mut free = None;
  {
    let pal = grid.palette();
    for i in 1u16..=PALETTE_INDEX_MAX {
      let id = PaletteId(i);
      if pal.occupied(id) {
        if existing.is_none() && pack_palette_entry(pal.get(id)) == want {
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
  grid.palette_mut().set(slot, mat.entry());
  slot
}

/// 世界空间射线 × 体素的 Amanatides-Woo 步进。返回 `(命中体素, 入面法线, 命中 t)`；
/// 法线指向射线来向（朝外），故 `命中体素 + 法线` 即前方那格空气。`dir` 分量可为 0（该轴不被选中）；仅主世界。
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

/// 单格是否落在笔触区域内（`d` = 该格相对中心的有符号偏移）。
fn brush_contains(shape: BrushShape, d: IVec3, r: i32) -> bool {
  match shape {
    BrushShape::Cube => d.x.abs() <= r && d.y.abs() <= r && d.z.abs() <= r,
    // 球判据 d² ≤ r² + r（r = size-1）：r=0 → 仅中心格；r=1 → 3³ 去掉 8 个角
    BrushShape::Sphere => d.length_squared() <= r.saturating_mul(r).saturating_add(r),
  }
}

/// 对齐块 `[lo, lo+extent)` 是否完全不在笔触区域内（可整块剪枝）。
/// 依据：笔触区域凸，且盒到中心的最小距离点 = 把中心 clamp 进盒 → 最近点在区域外即整块在区域外。
fn brush_box_disjoint(shape: BrushShape, center: IVec3, r: i32, lo: IVec3, extent: i32) -> bool {
  let hi = lo + IVec3::splat(extent - 1);
  !brush_contains(shape, center.clamp(lo, hi) - center, r)
}

/// 对齐块是否完全在笔触区域内（凸区域包含盒 ⟺ 包含它的 8 个角）
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

/// 笔触的层级填充（`lo` = 对齐到 `extent` 的块最小角）：整块落在笔触内的走一次
/// `VolumeGrid::fill_brick`（O(深度)，直接产出 uniform 上级节点），部分覆盖的才下钻，最小到 1³。
/// `palette` 为 AIR 即擦除；放置只填空气、擦除只挖实体。
#[allow(clippy::too_many_arguments)] // 递归下钻：参数即递归状态（笔触 + 当前块 + 输出），打包成结构体反而每层重建
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
    // 整块都在笔触内 → 按 brick 三态：空气+放置 / 实体+擦除 → 整块写；空气+擦除 / 实体+放置
    // → 不改动（不啃掉已有几何）；Mixed → 下钻。
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
    // 收尾单格：extent==1 时「部分覆盖」即完全覆盖，无需查 brick 三态
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
/// `palette == 0` → 擦除；否则只填充空气格（不啃掉已有几何）。写入自顶向下按 brick 粒度进行，
/// 整块写直接落成 uniform 上级节点（见 `fill_brush_level`）。
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
  // 起始层级：取 extent ≤ 跨度 2r+1 的最大 brick 粒度（小笔触直接落到 4³/1³）。
  let span = r.saturating_mul(2).saturating_add(1);
  let start = LEVEL_EXTENT.iter().copied().find(|&e| e <= span).unwrap_or(1);
  // 覆盖笔触 AABB 的全部对齐块；世界对齐 ⇒ 块必落在单个 chunk 内（256 是各 brick 粒度的整数倍）。
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

/// 体素编辑输入（仅幽灵模式；轨道模式左键仍是 recenter）：左键 = 放置，右键 = 擦除；
/// 按下到释放累计位移 > `DRAG_PX` 视为「拖拽转头」，不编辑。
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
    // 放置落点 = 命中面外侧一格；槽位按材质内容取/建（参数变了即新材质，旧体素不受影响）
    let slot = material_slot(grid, settings.mat);
    (hit + face, slot)
  };
  let changed = apply_brush(grid, center, shape, size, pal);
  if changed > 0 {
    bevy::log::info!(
      "EDIT[{}]: {} voxel(s) @ ({},{},{}) shape={:?} size={} slot={} material={}",
      if erase { "erase" } else { "place" },
      changed,
      center.x,
      center.y,
      center.z,
      shape,
      size,
      pal,
      if erase { "-".to_string() } else { settings.mat.summary() },
    );
  }
}

/// 第 60 帧朝初始注视点刷一次笔触，走通编辑 → 增量上传链路；由 `consts::EDIT_SELFTEST` 决定是否注册。
/// 只在设了该变量时注册（见 main.rs）。
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
    "EDIT SELFTEST: hit=({},{},{}) t={t:.1} → placed {n} voxel(s) @ ({},{},{}) shape={:?} size={size} slot={slot} material={}",
    hit.x,
    hit.y,
    hit.z,
    center.x,
    center.y,
    center.z,
    shape,
    settings.mat.summary(),
  );
}

#[cfg(test)]
mod tests {
  use super::*;

  /// 平凡变体的落盘路径与改动前完全一致（`flags` 全 0 + 两个字的期望值）。
  #[test]
  fn plain_entry_packing_unchanged() {
    let plain = BrushMaterial {
      color: [10, 20, 30],
      roughness: 200,
      emissive: 7,
      transmission: 9,
      ..Default::default()
    };
    assert_eq!(plain.entry().flags.0, 0, "平凡变体不设任何 flags");
    // word0 = 0x0A | 0x14<<8 | 0x1E<<16 | 0xC8<<24；word1 = 0x07 | 0x09<<8 | TRANSMISSIVE<<16
    assert_eq!(pack_palette_entry(&plain.entry()), [0xC81E140A, 0x00200907]);
  }

  /// MT7-1：PBR 变体的 `entry()` 走 `PaletteEntry::pbr`（构造）+ 变体分派打包，
  /// 且 `TRANSMISSIVE` 由**槽级透射覆盖**决定（字节 1 = 覆盖为精确 0 ⇒ 不是介质）。
  #[test]
  fn pbr_entry_packing_and_medium_flag() {
    let pbr = BrushMaterial {
      pbr: true,
      asset_slot: 0x0102,
      roughness_ov: 1,
      metallic_ov: 255,
      specular_ov: 255,
      ..Default::default()
    };
    // word0 = roughness覆盖(1) | metallic覆盖(255)<<8；word1 = asset(0x0102) | IS_PBR(0x10)<<16 | specular(255)<<24
    assert_eq!(pack_palette_entry(&pbr.entry()), [0x0000_FF01, 0xFF10_0102]);
    assert_eq!(pbr.entry().pbr_asset(), 0x0102);

    let medium = BrushMaterial { transmission_ov: 2, ..pbr };
    assert_ne!(
      pack_palette_entry(&medium.entry())[1] & 0x0020_0000,
      0,
      "透射覆盖值 > 0 ⇒ 标 TRANSMISSIVE（D1：PBR 变体的介质位由调用方决定）"
    );
    let zero = BrushMaterial { transmission_ov: 1, ..pbr };
    assert_eq!(
      pack_palette_entry(&zero.entry())[1] & 0x0020_0000,
      0,
      "字节 1 是「覆盖为精确 0」⇒ 不是介质（与平凡变体的 transmission > 0 同义）"
    );
  }

  /// MT7-3：PBR 变体也按**内容（8B payload）**去重 —— 同内容复用同一槽，内容变了才认领新槽，
  /// 且落进槽的确实是 PBR 变体的打包结果。
  #[test]
  fn pbr_material_dedups_by_payload() {
    let mut grid = VolumeGrid::new();
    let a = BrushMaterial { pbr: true, asset_slot: 10, metallic_ov: 255, ..Default::default() };
    let s1 = material_slot(&mut grid, a);
    assert_eq!(pack_palette_entry(grid.palette().get(s1)), [0x0000_FF00, 0x0010_000A]);
    assert_eq!(material_slot(&mut grid, a), s1, "同 8B payload ⇒ 复用同一槽");
    // 覆盖值不同、资产槽不同 ⇒ 各自认领新槽
    let b = BrushMaterial { metallic_ov: 128, ..a };
    let c = BrushMaterial { asset_slot: 11, ..a };
    assert_ne!(material_slot(&mut grid, b), s1);
    assert_ne!(material_slot(&mut grid, c), s1);
    assert_eq!(*grid.palette().get(s1), a.entry(), "旧槽内容不得被改写");
  }

  /// MT7-3 的验收本体：**改材质不影响旧体素**（旧体素指着旧槽，旧槽内容不变）。
  #[test]
  fn material_change_does_not_touch_old_voxels() {
    let mut grid = VolumeGrid::new();
    let a = BrushMaterial { pbr: true, asset_slot: 3, ..Default::default() };
    let s1 = material_slot(&mut grid, a);
    let coord = IVec3::new(4, 5, 6);
    grid.set_voxel_ivec3(coord, s1);
    let s2 = material_slot(&mut grid, BrushMaterial { asset_slot: 4, ..a });
    assert_ne!(s1, s2);
    let v = grid.get_voxel(VoxelCoord::from_ivec3(coord)).unwrap_or(PaletteId::AIR);
    assert_eq!(v, s1, "旧体素仍指向旧槽");
    assert_eq!(*grid.palette().get(v), a.entry(), "旧槽内容 = 最初的材质");
  }

  /// 平凡变体的去重行为与改动前一致（同内容复用 / 不同内容新槽）。
  #[test]
  fn plain_material_dedup_unchanged() {
    let mut grid = VolumeGrid::new();
    let a = BrushMaterial::default();
    let s1 = material_slot(&mut grid, a);
    assert_eq!(material_slot(&mut grid, a), s1);
    let b = BrushMaterial { color: [1, 2, 3], ..a };
    assert_ne!(material_slot(&mut grid, b), s1);
  }
}
