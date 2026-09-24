//! 体素编辑：幽灵模式下左键放置 / 右键擦除，笔触 = 形状（球/立方）× 大小（voxel）× 材质。
//! 目标选取：准星/光标 → 世界射线（`crate::camera::cursor_ray`）→ 统一 CPU 射线入口
//! `gate_render::raycast`（层次栈式 mask DDA，见 `brickmap::raytrace`）；
//! 放置落点 = 命中面外侧一格（擦除取命中格自身），写经 `set_voxel` → `mark_data(chunk)` 驱动增量上传。
//!
//! **MT8-5 的笔触接线（本文件）**：落笔时按**笔触材质的资产**判断要不要位移 ——
//! PBR 变体槽 + 该资产 `displacement_amplitude > 0` ⇒ 走 `gate_voxel::fill_*_displaced`
//! （高度图在 `height_field::MaterialDisplaceCache` 里按材质 id 缓存，解码只付一次），
//! 否则（平凡变体 / 幅度 0 / 无高度图 / 超尺寸上限）走**原有** `apply_brush`，行为逐位不变。
//! 位移语义与 MT6 的样例（`scene.rs::build_displace_sample`）**完全同一套**：
//! 它是"同一张高度图 + 同一套切空间 / Repeat / 双线性 / 偏置 0.5"的采样（`docs/PLAN.md` §3 D2）。
//! 每笔的 `EDIT[...]` 日志都带"按材质 X 位移，幅度 N 体素 / 未位移（原因）"⇒ 有没有位移有据可查。

use std::time::Instant;

use bevy::prelude::*;
use glam::IVec3;

use gate_render::brickmap::wire::pack_palette_entry;
use gate_render::{DdaCameraConfig, PbrTextureSet, VoxelScene, raycast};
use gate_voxel::{
  BRICK_FACTOR, BrickState, Displace, FillStats, LEVEL_EXTENT, PALETTE_INDEX_MAX, PaletteEntry,
  PaletteFlags, PaletteId, PbrOverrides, VolumeGrid, VoxelCoord, fill_box_displaced,
  fill_sphere_displaced,
};

use crate::{
  camera::{CameraMode, MouseLock, cursor_ray},
  consts::{
    DEMO_DISPLACE_TEX_SCALE, EDIT_BUDGET_MS, EDIT_DISPLACE_SIZE_MAX, EDIT_REACH, EDIT_REPEAT_SECS,
  },
  height_field::{MaterialDisplace, MaterialDisplaceCache},
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
/// - `pbr = false`（默认）：`color` / `emissive` / `transmission` / `roughness` / `metallic`
///   就是平凡 payload（`metallic` 原本是废弃 `_pad`，默认 `0` ⇒ 打包结果与改动前逐位相同）；
/// - `pbr = true`：8B 被解释为 `asset_slot` + `flags`，**参数全部来自材质资产与它的贴图**
///   （albedo / roughness / metallic / emissive / specular / IOR 都是资产级的）⇒ 菜单上那七个
///   材质控件在 PBR 模式下**整行置灰、不生效**（见 `debug_menu.rs::sync_edit_menu`），这里也就
///   没有 `*_ov` 这种字段。
///
/// 槽级覆盖（`PbrOverrides`）这条能力**只留给场景资产**：`.vox` 的 MATL 元数据
/// （`_rough` / `_metal` / `_emit`）由 `vox_scene.rs::matl_to_entry` 落进那 5 个覆盖字节。
/// 编辑器画笔**不写**覆盖 —— 一刀切：用 PBR 就用资产那一份。
///
/// **IOR 不在本结构里**：它是**资产级**物理基值（`MaterialAsset::transmission_ior` 的高 16 位，同时服务
/// 玻璃折射与电介质 F0），这 8B 里没有它的位置 ⇒ 菜单上也没有这个控件（曾经有过一个只记录、
/// 不改画面的滑杆，已删除）。
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
  /// 金属度（**二值**：`0` = 电介质、`255` = 完全金属；平凡变体直接就是这个字节，菜单上是开关）。
  ///
  /// **为什么是二值**：metallic 取连续值的唯一理由是**让贴图表达过渡**（锈蚀 / 磨损 / 掉漆的边缘），
  /// 物理上"是不是金属"没有中间态。平凡变体没有贴图可承载过渡 ⇒ 开关就是它的全部语义。
  /// （PBR 那边的连续量来自 roughmetal 贴图的 B 通道，与本字段无关。）
  ///
  /// **镜面的关键旋钮**：金属的 `F0 = albedo`（D1「F0 的唯一来源规则」，见 `common.wesl::f0_of`）
  /// ⇒ 「颜色 = 白 + 光滑度 100 + 金属度开」就是一个 `F0 = 1` 的镜面；
  /// 电介质的 `F0` 只有 `((IOR−1)/(IOR+1))² · specular = 0.04` ⇒ 正视下反射项只有 4%、看不出镜面。
  /// 开关同时给出 `kD = 1 − metallic = 0`（漫反射清零）—— 少了它，那 96% 的漫反射会把镜面盖住。
  /// **PBR 变体下本字段不参与**（参数来自资产的 roughmetal 贴图）—— 那份 8B 里也没有槽级旋钮。
  pub metallic: u8,
  /// `IS_PBR` 变体开关（菜单「PBR 变体」）
  pub pbr: bool,
  /// PBR 变体引用的材质资产槽号（= `PbrTextureSet` 的层号 = 全局资产表下标）
  pub asset_slot: u32,
}

impl Default for BrushMaterial {
  /// 与 `assets/ui/debug_menu.toml` 初值一致
  /// （中灰 / 不发光 / 不透明 / 半粗糙 / 平凡变体 / 槽 0）。
  fn default() -> Self {
    Self {
      color: [0x96, 0x98, 0x9E],
      emissive: 0,
      transmission: 0,
      roughness: 128,
      metallic: 0,
      pbr: false,
      asset_slot: 0,
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

  /// 平凡变体条目：`color` / `roughness` / `metallic` / `emissive` / `transmission` 逐字节直落
  /// （默认 `metallic = 0` ⇒ 与引入该字段前**逐位相同**）
  fn plain_entry(&self) -> PaletteEntry {
    PaletteEntry {
      color: self.color,
      roughness: self.roughness,
      metallic: self.metallic,
      emissive: self.emissive,
      transmission: self.transmission,
      ..Default::default()
    }
  }

  /// PBR 变体条目：**只有 `asset`**（槽级覆盖全"不覆盖"）—— 参数由资产与它的贴图决定。
  ///
  /// `TRANSMISSIVE` 位（D1：**由调用方决定**，打包函数不推断）在编辑器这条路上**恒不置**：
  /// 默认资产集的标量透射率全 0（`gate-render/src/pbr_texture.rs::build_material_asset_table`），
  /// 且槽级覆盖已不再由菜单写 ⇒ 画不出 PBR 玻璃（要玻璃用平凡变体的「透明度」）。
  /// 将来资产真的带透射时，这里要改成看资产（登记为后续项）。
  fn pbr_entry(&self) -> PaletteEntry {
    PaletteEntry::pbr(
      self.asset_slot.min(u16::MAX as u32) as u16,
      PbrOverrides::default(),
      PaletteFlags::default(),
    )
  }

  /// 日志用的 `#RRGGBB`（只对平凡变体有意义 —— PBR 变体不做逐实例染色）
  pub fn hex(&self) -> String {
    format!("#{:02X}{:02X}{:02X}", self.color[0], self.color[1], self.color[2])
  }

  /// 日志用的一行摘要（菜单每次改动都打 `材质 → {summary}`，MT7-1 的验收之一）。
  pub fn summary(&self) -> String {
    if !self.pbr {
      return format!(
        "平凡 {} 自发光={} 透射率={} 粗糙度={} 金属度={}",
        self.hex(),
        self.emissive,
        self.transmission,
        self.roughness,
        self.metallic
      );
    }
    format!("PBR 变体 asset={}", self.asset_slot)
  }
}

/// 菜单「透明度」滑杆 0..100 → `PaletteEntry.transmission`（**100 = 全透**，`0` = 完全不透明）。
///
/// 映射与标签**同向**：滑杆值 = 透射率百分比（「光滑度」仍是 `inverted_*`：光滑度越大 ⇒ 粗糙度越小，
/// 两者各用各的映射）。输出 `> 0` 会被写入侧（`wire.rs::pack_palette_entry`）标成可穿透介质；
/// `debug_menu.toml` 的出厂初值 = `0`（新笔触默认**不透明**）。
pub fn transparency_pct_to_transmission(pct: f32) -> u8 {
  ((pct.clamp(0.0, 100.0) / 100.0) * 255.0).round() as u8
}

/// 菜单「光滑度」滑杆 0..100 → `PaletteEntry.roughness`（0 = 镜面、255 = 完全粗糙）。
pub fn smooth_pct_to_roughness(pct: f32) -> u8 {
  (((100.0 - pct.clamp(0.0, 100.0)) / 100.0) * 255.0).round() as u8
}

/// 菜单「金属度」开关 → `PaletteEntry.metallic` 字节（`0` = 电介质、`255` = 完全金属）。
/// 二值 ⇒ 只给两个端点：着色侧读 `f32(byte)/255.0` ⇒ 精确 `0.0` / `1.0`，无中间态。
pub fn metal_toggle_to_metallic(on: bool) -> u8 {
  if on { 255 } else { 0 }
}

/// 编辑设置（main world Resource）
#[derive(Resource, Clone, Copy, Debug)]
pub struct EditSettings {
  pub shape: BrushShape,
  /// 笔触大小（voxel）
  pub size: u32,
  /// **偏移距离**（voxel，菜单「游戏/编辑/笔触/偏移距离」）：笔触几何中心相对**点击选中的体素**
  /// 沿命中面法线方向的偏移。
  ///
  /// - **放置**：中心 = 命中体素 `+ face · round(offset)`（往**外**推）；
  /// - **摧毁**：中心 = 命中体素 `− face · round(offset)`（往**内**挖）—— 同一个值取反方向。
  ///
  /// `face` 是 DDA 给的入面法线（指向射线来向）⇒ 正偏移对放置是"离开表面"、对摧毁是"深入表面"。
  /// 笔触中心必须是**整数体素格** ⇒ 落笔时四舍五入到格（`1.5 ⇒ 2`；`offset = 1` 恰好等于旧的
  /// "贴面外一格"）。菜单里可自由输入；**改「笔触大小」时自动置为 `size / 2`**（`default` 与之一致）。
  pub offset: f32,
  /// 当前笔触材质参数（由菜单「游戏/编辑/材质」的控件驱动）
  pub mat: BrushMaterial,
}

impl Default for EditSettings {
  fn default() -> Self {
    // offset = size / 2 = 1.5：与"改大小就自动跟着走"那条规则一致（size 默认 3，与菜单初值同）
    Self { shape: BrushShape::Sphere, size: 3, offset: 1.5, mat: BrushMaterial::default() }
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
      bevy::log::warn!("编辑材质无空调色板槽 → 复用 {PALETTE_INDEX_MAX}（覆盖该槽原材质）");
      PaletteId(PALETTE_INDEX_MAX)
    }
  };
  grid.palette_mut().set(slot, mat.entry());
  slot
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

/// 笔触半径（格）= `size - 1`；只做「不溢出 i32」的类型收敛（菜单 size 无上限）
fn brush_radius(size: u32) -> i32 {
  size.saturating_sub(1).min(i32::MAX as u32) as i32
}

/// 4³ 块 `[blo, blo+extent)` 到 `center` 的度量区间：球 = `d²`（各轴求和），立方 = `L∞`（最大轴距）。
fn block_metric_range(shape: BrushShape, center: IVec3, blo: IVec3, extent: i32) -> (f32, f32) {
  let (mut dmin, mut dmax) = (0.0f32, 0.0f32);
  for a in 0..3 {
    let lo = (blo[a] - center[a]) as f32;
    let hi = lo + (extent - 1) as f32;
    // 该轴上块到 0 的距离区间（块含 0 ⇒ near = 0）
    let near = if lo <= 0.0 && hi >= 0.0 { 0.0 } else { lo.abs().min(hi.abs()) };
    let far = lo.abs().max(hi.abs());
    match shape {
      BrushShape::Cube => {
        dmin = dmin.max(near);
        dmax = dmax.max(far);
      }
      BrushShape::Sphere => {
        dmin += near * near;
        dmax += far * far;
      }
    }
  }
  (dmin, dmax)
}

/// 笔触是否**只动了被实体完全包围的区域**（⇒ 可见几何不变，GI 历史可继续复用）。
///
/// 判据：笔触区域外一层全是实体 ⇒ 任何从外部进入的射线都会先命中外壳，而外壳不被笔触修改
/// ⇒ 新出现的实体/空气界面看不见（山体内部挖洞、封在内部的空腔都属于这种）。
/// 按 4³ 块粒度求外壳、**要求整块实心**（比逐体素更严）⇒ 只会多报"可见"，不会漏报。
/// 位移笔触不适用（位移会写到区域外 ≤ 幅度格，外壳要再厚一档才安全），调用方按"可见"处理。
fn stroke_hidden(grid: &VolumeGrid, shape: BrushShape, center: IVec3, r: i32) -> bool {
  // 外壳的度量区间：立方 `L∞ ∈ (r, r+1]`；球 `d² ∈ (r²+r, (r+1)²+r]`
  let (d_lo, d_hi) = match shape {
    BrushShape::Cube => (r as f32, (r + 1) as f32),
    BrushShape::Sphere => {
      let rf = r as f32;
      (rf * rf + rf, (rf + 1.0) * (rf + 1.0) + rf)
    }
  };
  const EXT: i32 = 4;
  let reach = r.saturating_add(8); // 含"块最小角可能在区域外一格再对齐下取"的余量
  let lo0 = center.saturating_sub(IVec3::splat(reach)).div_euclid(IVec3::splat(EXT)) * EXT;
  let hi0 = center.saturating_add(IVec3::splat(reach));
  let mut x = lo0.x;
  while x <= hi0.x {
    let mut y = lo0.y;
    while y <= hi0.y {
      let mut z = lo0.z;
      while z <= hi0.z {
        let blo = IVec3::new(x, y, z);
        let (dmin, dmax) = block_metric_range(shape, center, blo, EXT);
        if dmin <= d_hi
          && dmax > d_lo
          && !matches!(grid.get_brick_state_extent(blo, EXT), BrickState::Solid(_))
        {
          return false;
        }
        z += EXT;
      }
      y += EXT;
    }
    x += EXT;
  }
  true
}

/// 普通笔触的待处理块（显式 DFS 栈的一项）：`lo` = 对齐到 `extent` 的块最小角。
struct BrushJob {
  lo: IVec3,
  extent: i32,
}

/// 普通笔触的**分帧执行状态**：显式栈 + 累计统计，可跨帧续做。
///
/// 分帧原因：size=61 的球一笔 ≈33ms（castle.vox 实测），整帧做完整帧必掉。这里把"哪些块还没
/// 处理"存下来，[`Self::step`] 在给定预算内尽量推进，其余留到下一帧（不丢笔、不阻塞帧）。
/// 结果与"一次做完"逐位相同：块之间互不相交，且每块的写入只依赖该块自身当前内容。
pub(crate) struct PlainBrush {
  shape: BrushShape,
  center: IVec3,
  r: i32,
  palette: PaletteId,
  erase: bool,
  stack: Vec<BrushJob>,
  /// 累计实际改变的体素数（整块写按 `extent³` 计；与 [`apply_brush`] 同口径）
  pub(crate) changed: usize,
  /// 累计 CPU 时间（只含 `step` 内的推进）
  pub(crate) cpu: std::time::Duration,
  /// 已推进的帧数（>1 即真的跨帧了）
  pub(crate) frames: u32,
}

impl PlainBrush {
  pub(crate) fn new(center: IVec3, shape: BrushShape, size: u32, palette: PaletteId) -> Self {
    let r = brush_radius(size);
    // 起始层级：取 extent ≤ 跨度 2r+1 的最大 brick 粒度（小笔触直接落到 4³/1³）。
    let span = r.saturating_mul(2).saturating_add(1);
    let start = LEVEL_EXTENT.iter().copied().find(|&e| e <= span).unwrap_or(1);
    // 覆盖笔触 AABB 的全部对齐块；世界对齐 ⇒ 块必落在单个 chunk 内（256 是各 brick 粒度的整数倍）。
    let s = IVec3::splat(start);
    let b_lo = center.saturating_sub(IVec3::splat(r)).div_euclid(s) * s;
    let b_hi = center.saturating_add(IVec3::splat(r)).div_euclid(s) * s;
    let mut stack = Vec::new();
    let mut x = b_lo.x;
    while x <= b_hi.x {
      let mut y = b_lo.y;
      while y <= b_hi.y {
        let mut z = b_lo.z;
        while z <= b_hi.z {
          stack.push(BrushJob { lo: IVec3::new(x, y, z), extent: start });
          z += start;
        }
        y += start;
      }
      x += start;
    }
    // LIFO 弹出 ⇒ 反序入栈，处理顺序与旧递归的 x→y→z 升序一致（几何结果与顺序无关，此处只为可复现）
    stack.reverse();
    Self {
      shape,
      center,
      r,
      palette,
      erase: palette.is_air(),
      stack,
      changed: 0,
      cpu: std::time::Duration::ZERO,
      frames: 0,
    }
  }

  /// 在 `budget` 内推进；`true` = 已完成（栈空）。预算按"每处理一个块检查一次"计 ⇒ 超支 ≤ 单块代价。
  pub(crate) fn step(&mut self, grid: &mut VolumeGrid, budget: std::time::Duration) -> bool {
    let t0 = Instant::now();
    self.frames += 1;
    while let Some(job) = self.stack.pop() {
      self.one(grid, job);
      if t0.elapsed() >= budget {
        break;
      }
    }
    self.cpu += t0.elapsed();
    self.stack.is_empty()
  }

  /// 是否擦除（日志用）
  pub(crate) fn is_erase(&self) -> bool {
    self.erase
  }

  /// 处理一个块：能剪枝/整块写就不下钻，否则展开 4³ 子块入栈（等价于旧的递归下钻）。
  fn one(&mut self, grid: &mut VolumeGrid, job: BrushJob) {
    let (lo, extent) = (job.lo, job.extent);
    if brush_box_disjoint(self.shape, self.center, self.r, lo, extent) {
      return;
    }
    if extent > 1 && brush_box_inside(self.shape, self.center, self.r, lo, extent) {
      // 整块都在笔触内 → 按 brick 三态：空气+放置 / 实体+擦除 → 整块写；空气+擦除 / 实体+放置
      // → 不改动（不啃掉已有几何）；Mixed → 下钻。
      match grid.get_brick_state_extent(lo, extent) {
        BrickState::Air => {
          if !self.erase {
            grid.fill_brick(lo, extent, self.palette);
            self.changed += (extent as usize).pow(3);
          }
          return;
        }
        BrickState::Solid(_) => {
          if self.erase {
            grid.fill_brick(lo, extent, self.palette);
            self.changed += (extent as usize).pow(3);
          }
          return;
        }
        BrickState::Mixed => {}
      }
    }
    if extent == 1 {
      // 收尾单格（只在 `start == 1` 的极小笔触出现）：extent==1 时「部分覆盖」即完全覆盖，无需查三态
      if brush_contains(self.shape, lo - self.center, self.r) {
        let cur = grid.get_voxel(VoxelCoord::from_ivec3(lo)).unwrap_or(PaletteId::AIR);
        if cur.is_air() != self.erase && grid.set_voxel_ivec3(lo, self.palette).is_some() {
          self.changed += 1;
        }
      }
      return;
    }
    if extent == BRICK_FACTOR {
      // 4³ 砖：**一次写完**（一次下钻 + 一次向上合并 + 一次脏标记）—— 逐格写要付 64 次。
      // 壳层（球面边界）就是这一档：它是笔触的主要代价来源。
      let mut inside = 0u64;
      for i in 0..(BRICK_FACTOR * BRICK_FACTOR * BRICK_FACTOR) {
        let d = IVec3::new(
          i % BRICK_FACTOR,
          (i / BRICK_FACTOR) % BRICK_FACTOR,
          i / (BRICK_FACTOR * BRICK_FACTOR),
        );
        if brush_contains(self.shape, lo + d - self.center, self.r) {
          inside |= 1u64 << i;
        }
      }
      self.changed += grid.set_brick_voxels(lo, inside, self.palette) as usize;
      return;
    }
    let sub = extent / BRICK_FACTOR;
    for i in (0..BRICK_FACTOR * BRICK_FACTOR * BRICK_FACTOR).rev() {
      let d = IVec3::new(
        i % BRICK_FACTOR,
        (i / BRICK_FACTOR) % BRICK_FACTOR,
        i / (BRICK_FACTOR * BRICK_FACTOR),
      );
      self.stack.push(BrushJob { lo: lo + d * sub, extent: sub });
    }
  }
}

/// 以 `center` 为中心施加一次笔触（**一次做完**），返回实际改变的体素数。
/// 写入自顶向下按 brick 粒度进行，整块写直接落成 uniform 上级节点（见 [`PlainBrush::one`]）。
/// 分帧版本见 [`PlainBrush`]（`voxel_edit_input` 用），两者结果逐位相同。
pub fn apply_brush(
  grid: &mut VolumeGrid,
  center: IVec3,
  shape: BrushShape,
  size: u32,
  palette: PaletteId,
) -> usize {
  let mut b = PlainBrush::new(center, shape, size, palette);
  let _ = b.step(grid, std::time::Duration::MAX);
  b.changed
}

// ============================================================================
// MT8-5 · 笔触 → 材质位移（本任务接的最后一环）
// ============================================================================

/// 一次落笔的结果（`EDIT[...]` 日志要的东西；耗时由调用方测，因为它还含槽位分配等）。
struct BrushRun {
  /// 普通路径 = 真正改变的体素数（`apply_brush` 的口径）；
  /// 位移路径 = 位移后落在实心区内的体素数（`FillStats::voxels` 的口径，见它的文档）
  voxels: usize,
  /// 位移路径的整块写 / 壳层逐体素统计（普通路径 = `None`）
  stats: Option<FillStats>,
  /// 位移说明（直接进 `EDIT[...]` 日志）：位移了 = 按哪个材质、幅度多少；没位移 = **原因**
  displace: String,
}

/// **落笔的唯一入口**：按笔触材质决定走位移填充还是原有填充。
///
/// 位移可走（PBR 变体 + 资产幅度 > 0 + 尺寸在限内 + 贴图集就绪）⇒ `fill_*_displaced`；
/// 否则 ⇒ 原 `apply_brush`（**逐位不变**：平凡笔触、`amplitude = 0` 的材质、擦除、超尺寸全都落在这条路）。
///
/// 材质 id 的解析走**槽 → `asset` → id**（`PbrTextureSet` 是"层号 ↔ id"的唯一权威，见 `pbr_texture.rs`）：
/// 放置槽由 [`material_slot`] 按内容去重产出 ⇒ 它与 `settings.mat.entry()` 的 8B payload 逐字节相同，
/// 但"读槽"更贴着**真正写下去的东西**（将来别的调用方只拿得到槽也能复用）。
///
/// ⚠️ **位移路径与普通路径的一处语义差异（如实记录，本次不改 `gate-voxel`）**：`fill_shape` 对
/// "落在形状内的格子"是**无条件写入**，而普通笔触只填空气、整块实心处整块跳过（[`PlainBrush::one`]
/// 的 brick 三态判定）。落点在命中面外侧 ⇒ 球/立方必然压住一层既有体素 ⇒ 位移笔触会把压在里面的
/// 那些体素改成**本笔触材质**（只是着色变了：实心/空气格局只会更实，**不会挖掉**既有几何）。
/// 这与"不做'对既有体素表面做位移'"（用户决策 B）不冲突：既有表面的凹凸不会被重算。
fn run_brush(
  grid: &mut VolumeGrid,
  center: IVec3,
  shape: BrushShape,
  size: u32,
  palette: PaletteId,
  pbr_set: Option<&PbrTextureSet>,
  cache: Option<&mut MaterialDisplaceCache>,
) -> BrushRun {
  let plain = |grid: &mut VolumeGrid, displace: String| BrushRun {
    voxels: apply_brush(grid, center, shape, size, palette),
    stats: None,
    displace,
  };
  // 擦除（AIR 槽）不位移：位移只产出"实心体素"，与"挖空"无关（D2：位移发生在 CSG 体素化那一刻）
  if palette.is_air() {
    return plain(grid, "未位移（擦除路径）".to_string());
  }
  let entry = *grid.palette().get(palette);
  let (source, reason) = brush_displace_source(entry, size, pbr_set, cache);
  let Some((md, id)) = source else {
    return plain(grid, reason);
  };
  // 组装 `Displace`（与 `scene.rs::build_displace_sample` 同一写法：闭包借用 `md`，
  // 必须在**同一作用域**里取闭包再组装 —— `Displace` 里装的是闭包的引用）
  let f = md.displace_fn();
  let bound = md.bound();
  let Some(st) =
    fill_brush_displaced(grid, center, shape, size, palette, Displace { f: &f, bound })
  else {
    // 球 size=1（半径 0）：`fill_sphere_displaced` 要求 radius > 0 ⇒ 退回普通填充（单格无凹凸可言）
    return plain(grid, "未位移（球 size=1：半径为 0，位移球要求 radius > 0）".to_string());
  };
  BrushRun {
    voxels: st.voxels,
    stats: Some(st),
    // 幅度/来源都来自**材质资产**（MT8-5：改资产即改凹凸，`consts` 不再是入口）
    displace: format!(
      "已按材质 `{id}` 位移，幅度 {} 体素（峰-峰，偏置双向 ±{bound}，一张高度图铺 {DEMO_DISPLACE_TEX_SCALE} 体素）",
      md.amplitude()
    ),
  }
}

/// 笔触的**位移源解析**（`docs/PLAN.md` §4b MT8-5 / §3 D2）：返回 `((位移源, 材质 id), 未位移的原因)`，
/// 两个分支互斥。四个前置条件（缺一即"不位移"，且**原因进日志**，不只说"没位移"）：
/// 1. **PBR 变体**（`IS_PBR`）：只有这种槽里才有 `asset: u16`（D1 的 8B 变体复用）；
///    平凡变体没有资产 ⇒ 没有高度图 ⇒ 不可能位移；
/// 2. **尺寸 ≤ [`EDIT_DISPLACE_SIZE_MAX`]**：位移壳层是逐体素的，代价 ≈ O(size²)（见该常量的实测说明）；
/// 3. **贴图集就绪**：槽号 → id 必须过 `PbrTextureSet::ids()`（它按 id 字典序定层号、缺素材会顺延）；
/// 4. **资产幅度 > 0 且高度图可用**：由 [`MaterialDisplaceCache`] 按 id 解析（首次 ≈22ms，之后解码耗时 0）。
fn brush_displace_source<'a>(
  entry: PaletteEntry,
  size: u32,
  pbr_set: Option<&'a PbrTextureSet>,
  cache: Option<&'a mut MaterialDisplaceCache>,
) -> (Option<(&'a MaterialDisplace, &'a str)>, String) {
  if !entry.flags.contains(PaletteFlags::IS_PBR) {
    return (None, "未位移（平凡变体：槽里没有资产 ⇒ 没有高度图）".to_string());
  }
  if size > EDIT_DISPLACE_SIZE_MAX {
    bevy::log::warn!(
      "笔触位移跳过：size={size} > 上限 {EDIT_DISPLACE_SIZE_MAX} vx（位移壳层逐体素 ≈ O(size²)）\
       ⇒ 普通填充，无凹凸"
    );
    return (None, format!("未位移（size {size} > 上限 {EDIT_DISPLACE_SIZE_MAX}）"));
  }
  let Some(set) = pbr_set else {
    // 贴图集是异步就绪的资源（`finish_pbr_textures` 插入）：启动后头几帧落笔会落在这里
    return (None, format!("未位移（PBR 贴图集未就绪：读不到资产槽 {} → id）", entry.pbr_asset()));
  };
  let Some(id) = entry_asset_id(&entry, set) else {
    return (None, format!("未位移（资产槽 {} 不在贴图集里）", entry.pbr_asset()));
  };
  let Some(cache) = cache else {
    return (None, "未位移（位移缓存资源缺失）".to_string());
  };
  match cache.get_or_load(id, DEMO_DISPLACE_TEX_SCALE) {
    Some(md) => (Some((md, id)), String::new()),
    // 幅度 = 0（资产值）/ id 不在材质目录集 / 高度图缺文件 —— 三种都由缓存那行日志说清了原因
    None => {
      (None, format!("未位移（材质 `{id}` 的资产幅度 = 0 或高度图不可用，见上面的缓存日志）"))
    }
  }
}

/// 槽里的 `asset: u16` → 材质 id（`assets/textures/pbr/<id>/` 的目录名）。
/// **非 PBR 变体不查**（那两个字节在平凡变体里是 emissive / transmission，不是资产槽）。
fn entry_asset_id<'a>(entry: &PaletteEntry, set: &'a PbrTextureSet) -> Option<&'a str> {
  (entry.flags.contains(PaletteFlags::IS_PBR))
    .then(|| set.ids().get(entry.pbr_asset() as usize).map(String::as_str))
    .flatten()
}

/// 位移笔触的填充：把笔触形状映射到 `gate-voxel` 的位移填充器（MT6-3 的现成 API，本次一字不改）。
///
/// | 笔触形状 | 位移填充器 | 写入集合是否与普通笔触一致 |
/// |---|---|---|
/// | `Cube` | [`fill_box_displaced`]（`min = center − r`、`extent = (2r+1)³`） | **逐格一致**（球盒判据同为 `L∞`/`\|d\| ≤ r`）|
/// | `Sphere` | [`fill_sphere_displaced`]（`radius = r = size − 1`） | 略小：位移球判据是**欧氏距离 ≤ r**，而球笔触是 `d² ≤ r² + r` |
///
/// 球那一行是**有意**的差异（本次不改 `gate-voxel`）：`d² ≤ r² + r` 是为了让 `r = 1` 正好等于
/// "3³ 去 8 角"，而在欧氏距离上多出半个格子的半径（`√(r²+r) − r`，r=15 时 ≈0.49 格）无法用
/// "整数半径"表达 ⇒ 位移球比普通球的外壳小不到一格（外表面整体内缩 <1 格）。
/// 半径 0（`size = 1`）无位移可言 ⇒ 返回 `None`（调用方退回普通填充）。
fn fill_brush_displaced(
  grid: &mut VolumeGrid,
  center: IVec3,
  shape: BrushShape,
  size: u32,
  palette: PaletteId,
  disp: Displace<'_>,
) -> Option<FillStats> {
  // 与 `apply_brush` 同口径：size 无上限（菜单可输入任意值），只做"不溢出 i32"的类型收敛
  let r = size.saturating_sub(1).min(i32::MAX as u32) as i32;
  match shape {
    BrushShape::Cube => {
      let extent = IVec3::splat(r.saturating_mul(2).saturating_add(1));
      Some(fill_box_displaced(grid, center - IVec3::splat(r), extent, palette, Some(disp)))
    }
    BrushShape::Sphere => {
      (r > 0).then(|| fill_sphere_displaced(grid, center, r, palette, Some(disp)))
    }
  }
}

/// `FillStats` → 日志尾巴（位移路径专用；普通路径没有这些数字）。
/// `whole_bricks` 是"走 `fill_brick` 整块写"的块数，位移填充器（`fill_box_displaced` /
/// `fill_sphere_displaced`）的块粒度恒为 4 ⇒ 一块 = 64 体素。
fn stats_suffix(stats: Option<&FillStats>) -> String {
  match stats {
    Some(st) => format!(
      "（整块写 {} 块 = {} 体素 + 壳层逐体素 {} 格）",
      st.whole_bricks,
      st.whole_bricks * 64,
      st.shell_voxels
    ),
    None => String::new(),
  }
}

/// 按住键连发的计时器（左键 = 放置、右键 = 擦除各一个，互不干扰）
#[derive(Default)]
pub(crate) struct HoldRepeat {
  place: f32,
  erase: f32,
}

/// 推进一个按钮的连发计时，返回本帧是否落一笔。按下当帧**立即**触发，之后按住每
/// `EDIT_REPEAT_SECS` 触发一次；松开清零（下次按下重新立即触发）。
/// 扣的是整间隔、余数留到下一帧 ⇒ 触发频率与帧率无关（高帧率下不漂移）。
fn hold_repeat(just_pressed: bool, held: bool, acc: &mut f32, dt: f32) -> bool {
  if !held {
    *acc = 0.0;
    return false;
  }
  if just_pressed {
    *acc = 0.0;
    return true;
  }
  *acc += dt;
  if *acc < EDIT_REPEAT_SECS {
    return false;
  }
  *acc -= EDIT_REPEAT_SECS;
  true
}

/// 一条 `EDIT[...]` 日志的全部字段（普通路径分帧完成后由 [`ActiveStroke`] 填，位移路径当帧填）
struct StrokeLog {
  erase: bool,
  voxels: usize,
  center: IVec3,
  shape: BrushShape,
  size: u32,
  off: i32,
  pal: PaletteId,
  /// 材质摘要（放置）/ `-`（擦除）
  material: String,
  /// 位移说明或未位移的**原因**
  reason: String,
  stats: Option<FillStats>,
  /// 该笔的实际 CPU 时间（普通路径 = 各帧 `step` 累计）
  cpu: std::time::Duration,
  /// 跨了几帧（>1 才在日志里体现）
  frames: u32,
  /// 本笔是"按下"（true）还是"连发"（false）
  fresh: bool,
}

impl StrokeLog {
  fn emit(&self) {
    if self.voxels == 0 {
      return;
    }
    let cost = if self.frames > 1 {
      format!("{:?} 跨{}帧", self.cpu, self.frames)
    } else {
      format!("{:?}", self.cpu)
    };
    let line = format!(
      "EDIT[{}] {}vx @({},{},{}) shape={:?} size={} offset={} slot={} material={} | {}{} | {cost}",
      if self.erase { "erase" } else { "place" },
      self.voxels,
      self.center.x,
      self.center.y,
      self.center.z,
      self.shape,
      self.size,
      self.off,
      self.pal,
      self.material,
      self.reason,
      stats_suffix(self.stats.as_ref()),
    );
    // 连发只记 debug：按住不放按 `EDIT_REPEAT_SECS` 的频率刷 info 会淹掉真正离散的动作
    if self.fresh {
      bevy::log::info!("{line}");
    } else {
      bevy::log::debug!("{line}");
    }
  }
}

/// 进行中的普通笔触 = 分帧状态 + **落笔那一帧**的日志上下文（分帧会跨帧，日志在完成那帧才打）
pub(crate) struct ActiveStroke {
  brush: PlainBrush,
  size: u32,
  off: i32,
  fresh: bool,
  material: String,
  reason: String,
}

impl ActiveStroke {
  fn to_log(&self) -> StrokeLog {
    StrokeLog {
      erase: self.brush.is_erase(),
      voxels: self.brush.changed,
      center: self.brush.center,
      shape: self.brush.shape,
      size: self.size,
      off: self.off,
      pal: self.brush.palette,
      material: self.material.clone(),
      reason: self.reason.clone(),
      stats: None,
      cpu: self.brush.cpu,
      frames: self.brush.frames,
      fresh: self.fresh,
    }
  }
}

/// 普通笔触的单帧预算（见 `consts::EDIT_BUDGET_MS`）
fn brush_budget() -> std::time::Duration {
  std::time::Duration::from_secs_f32(EDIT_BUDGET_MS * 0.001)
}

/// 这一笔会不会走位移（只判定、不执行）：分帧调度要知道哪条路是原子的（位移路径原子）。
fn brush_displaced(
  entry: PaletteEntry,
  size: u32,
  pbr_set: Option<&PbrTextureSet>,
  cache: Option<&mut MaterialDisplaceCache>,
) -> (bool, String) {
  let (source, reason) = brush_displace_source(entry, size, pbr_set, cache);
  (source.is_some(), reason)
}

/// 体素编辑输入（仅幽灵模式；轨道模式左键仍是 recenter）：左键 = 放置，右键 = 擦除；
/// 按住不放持续落笔（间隔见 `consts::EDIT_REPEAT_SECS`）；锁定鼠标时射线取屏幕中心（准星），否则取光标。
///
/// 普通笔触按 `consts::EDIT_BUDGET_MS` **分帧推进**：同一时刻只跑一笔，做完了才接新的按下
/// （不排队、不丢笔）；跨帧期间新的按下被忽略 —— 定位与材质在按下那一帧就固定了。
#[allow(clippy::too_many_arguments)] // Bevy system：输入/资源逐一注入
pub(crate) fn voxel_edit_input(
  mouse: Res<ButtonInput<MouseButton>>,
  time: Res<Time>,
  captured: Res<gate_ui::UiPointerCaptured>,
  intercepted: Res<gate_ui::MouseIntercepted>,
  windows: Query<&Window>,
  cfg: Res<DdaCameraConfig>,
  mode: Res<CameraMode>,
  lock: Res<MouseLock>,
  settings: Res<EditSettings>,
  // MT8-5：位移源（槽 → 资产 id）要过贴图集 + 按 id 缓存的高度场（首次解码 ≈22ms，之后 0）
  pbr_set: Option<Res<PbrTextureSet>>,
  mut displace_cache: ResMut<MaterialDisplaceCache>,
  scene: Option<ResMut<VoxelScene>>,
  mut active: Local<Option<ActiveStroke>>,
  mut hold: Local<HoldRepeat>,
) {
  let Some(mut scene) = scene else { return };
  // 换模式 / 世界重载 ⇒ 丢弃未做完的笔触：它记的块坐标属于那一刻的世界
  if *mode != CameraMode::Fly || scene.demo_force_full_rebuild {
    *active = None;
    return;
  }
  // 有笔触还在跨帧推进 ⇒ 本帧不压实树区（压实会重传搬动过的树段，塞进笔触中途 = 一次卡顿）
  scene.edit_in_flight = active.is_some();
  // 先把上一帧没做完的推进一个预算；没做完就本帧不再接新输入
  if active.is_some() {
    let done = {
      let st = active.as_mut().expect("is_some 已判定");
      st.brush.step(scene.volumes.main_mut(), brush_budget())
    };
    if !done {
      return;
    }
    if let Some(st) = active.take() {
      st.to_log().emit();
    }
  }
  // UI 指针捕获期间不落笔，也不推进计时 ⇒ 指针一离开 UI 立即续上按住的那一笔
  if captured.0 || intercepted.0 {
    return;
  }
  let dt = time.delta_secs();
  let place = hold_repeat(
    mouse.just_pressed(MouseButton::Left),
    mouse.pressed(MouseButton::Left),
    &mut hold.place,
    dt,
  );
  let erase = hold_repeat(
    mouse.just_pressed(MouseButton::Right),
    mouse.pressed(MouseButton::Right),
    &mut hold.erase,
    dt,
  );
  if !place && !erase {
    return;
  }
  // 本帧这一笔是"按下"还是"连发"：日志分级用（连发只记 debug）
  let fresh = (place && mouse.just_pressed(MouseButton::Left))
    || (erase && mouse.just_pressed(MouseButton::Right));
  let Ok(window) = windows.single() else { return };
  let Some((origin, dir)) = cursor_ray(window, &cfg, lock.0) else {
    return;
  };
  let (shape, size) = (settings.shape, settings.size);
  // 笔触只写主世界 ⇒ 物体命中不落笔（物体的命中 voxel 是它自己的局部系）
  let Some(hit) = raycast(&scene.volumes, origin, dir, EDIT_REACH) else {
    return;
  };
  if hit.obj_id != -1 {
    return;
  }
  let (hit, face) = (hit.voxel, hit.face);
  // 笔触几何中心 = 命中体素沿**入面法线**偏移 `round(offset)` 格（见 `EditSettings::offset`）：
  // 放置往**外**推、摧毁往**内**挖 —— 同一个值取反方向。中心必须是整数体素格 ⇒ 四舍五入。
  let off = settings.offset.max(0.0).round() as i32;
  let pal = if erase {
    PaletteId::AIR
  } else {
    // 放置落点 = 命中体素沿法线外推 `off` 格（`off = 1` 即旧的"贴面外一格"）；
    // 槽位按材质内容取/建（参数变了即新材质，旧体素不受影响）
    material_slot(scene.volumes.main_mut(), settings.mat)
  };
  let center = if erase { hit - face * off } else { hit + face * off };
  let material = if erase { "-".to_string() } else { settings.mat.summary() };
  // MT8-5：走位移还是普通填充由**笔触材质的资产**决定（`run_brush` 里解析）
  let entry = *scene.volumes.main().palette().get(pal);
  let (displaced, reason) =
    brush_displaced(entry, size, pbr_set.as_deref(), Some(&mut displace_cache));
  // 这一笔会不会改动**可见**几何（判定必须在写入**前**做：外壳块可能与笔触区域重叠，前态才代表"外面那层"）。
  // 位移笔触会写到区域外 ≤ 幅度格 ⇒ 一律按"可见"，见 `stroke_hidden`。
  //
  // 再加一道前提：只有"这一笔就是本帧全部待上传改动"时才敢声明不可见 —— 否则上一帧可见编辑的积压会被
  // 连同这一笔一起按"不可见"上报（GI 该丢的历史没丢）。积压时退回"可见"（保守但正确）。
  let hidden = !displaced && stroke_hidden(scene.volumes.main(), shape, center, brush_radius(size));
  let no_backlog = scene
    .volumes
    .list
    .iter()
    .all(|g| g.dirty.data_dirty_count() == 0 && g.dirty.comp_dirty_count() == 0);
  scene.interior_only_edit = hidden && no_backlog;
  let grid = scene.volumes.main_mut();
  if displaced {
    // 位移路径：原子执行。位移壳层逐体素（≈O(size²)）但尺寸被 `EDIT_DISPLACE_SIZE_MAX` 限住 ⇒ 代价有界
    let t0 = Instant::now();
    let run =
      run_brush(grid, center, shape, size, pal, pbr_set.as_deref(), Some(&mut displace_cache));
    StrokeLog {
      erase,
      voxels: run.voxels,
      center,
      shape,
      size,
      off,
      pal,
      material,
      reason: run.displace,
      stats: run.stats,
      cpu: t0.elapsed(),
      frames: 1,
      fresh,
    }
    .emit();
    return;
  }
  // 普通路径：分帧推进（本帧先跑一个预算；小笔触当帧就完，不额外增延迟）
  let mut st = ActiveStroke {
    brush: PlainBrush::new(center, shape, size, pal),
    size,
    off,
    fresh,
    material,
    reason,
  };
  if st.brush.step(grid, brush_budget()) {
    st.to_log().emit();
  } else {
    *active = Some(st);
  }
}

/// 第 60 / 70 帧各朝初始注视点刷一次笔触（**同参数连落两笔**：第 1 笔付高度图解码、第 2 笔
/// 命中 `MaterialDisplaceCache` ⇒ 日志里能看到"解码耗时 0"与命中计数），走通编辑 → 增量上传链路；
/// 由 `consts::EDIT_SELFTEST` 决定是否注册。只在设了该变量时注册（见 main.rs）。
pub(crate) fn edit_selftest(
  scene: Option<ResMut<VoxelScene>>,
  orbit: Res<gate_render::OrbitCamera>,
  settings: Res<EditSettings>,
  pbr_set: Option<Res<PbrTextureSet>>,
  mut displace_cache: ResMut<MaterialDisplaceCache>,
  mut frame: Local<u32>,
) {
  *frame += 1;
  let nth = match *frame {
    60 => 1,
    70 => 2,
    _ => return,
  };
  let Some(mut scene) = scene else { return };
  let dir = (orbit.target - orbit.eye()).normalize_or_zero();
  if dir.length_squared() < 1e-12 {
    bevy::log::warn!("EDIT SELFTEST: 相机朝向退化 → 跳过");
    return;
  }
  let (shape, size) = (settings.shape, settings.size);
  let Some(hit) = raycast(&scene.volumes, orbit.eye(), dir, EDIT_REACH) else {
    bevy::log::warn!("EDIT SELFTEST: 射线未命中体素 → 跳过");
    return;
  };
  if hit.obj_id != -1 {
    return;
  }
  let (hit, face, t) = (hit.voxel, hit.face, hit.t);
  // 自测笔触可能带位移（写到区域外）⇒ 一律按"可见"上报，避免残留值压制 GI 失效
  scene.interior_only_edit = false;
  scene.edit_in_flight = false;
  let grid = scene.volumes.main_mut();
  let slot = material_slot(grid, settings.mat);
  // 与 `voxel_edit_input` 的**放置路径同一条中心公式**（偏移四舍五入到格，见 `EditSettings::offset`）
  let center = hit + face * settings.offset.max(0.0).round() as i32;
  let t0 = Instant::now();
  let run =
    run_brush(grid, center, shape, size, slot, pbr_set.as_deref(), Some(&mut displace_cache));
  let elapsed = t0.elapsed();
  bevy::log::info!(
    "EDIT SELFTEST 第{nth}笔 hit=({},{},{}) t={t:.1} → {}vx @({},{},{}) shape={:?} size={size} slot={slot} material={} | {}{} | {:?}",
    hit.x,
    hit.y,
    hit.z,
    run.voxels,
    center.x,
    center.y,
    center.z,
    shape,
    settings.mat.summary(),
    run.displace,
    stats_suffix(run.stats.as_ref()),
    elapsed,
  );
}

#[cfg(test)]
mod tests {
  use super::*;

  /// 逐格导出 `[lo, hi)` 的"实心/空气"位串（比对两次填充的写入集合用，与 `gate-voxel` 的单测同一手法）
  fn dump(grid: &VolumeGrid, lo: IVec3, hi: IVec3) -> Vec<bool> {
    let mut out = Vec::new();
    for z in lo.z..hi.z {
      for y in lo.y..hi.y {
        for x in lo.x..hi.x {
          out.push(
            !grid
              .get_voxel(VoxelCoord::from_ivec3(IVec3::new(x, y, z)))
              .unwrap_or(PaletteId::AIR)
              .is_air(),
          );
        }
      }
    }
    out
  }

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

  /// 零回归护栏（本任务 ①）：**没位移就必有一条原因**，且四种"不位移"能各自区分 ——
  /// 平凡变体（槽里没有资产）/ 超尺寸上限 / 贴图集未就绪 / 资产槽越界。
  /// 前三种不需要 `PbrTextureSet`（判定顺序在它之前）⇒ 单测可覆盖。
  #[test]
  fn no_displace_always_reports_a_reason() {
    let plain = BrushMaterial::default().entry();
    let (src, why) = brush_displace_source(plain, 4, None, None);
    assert!(src.is_none());
    assert!(why.contains("平凡变体"), "{why}");

    let pbr = BrushMaterial { pbr: true, asset_slot: 7, ..Default::default() }.entry();
    assert!(pbr.flags.contains(PaletteFlags::IS_PBR));

    let (src, why) = brush_displace_source(pbr, EDIT_DISPLACE_SIZE_MAX, None, None);
    assert!(src.is_none());
    assert!(why.contains("贴图集未就绪"), "{why}");

    let (src, why) = brush_displace_source(pbr, EDIT_DISPLACE_SIZE_MAX + 1, None, None);
    assert!(src.is_none());
    assert!(why.contains("上限"), "{why}");
  }

  /// 位移笔触的形状映射（本任务 ① 的"逐格可复算"）：**Cube 笔触就是
  /// `fill_box_displaced(center − r, (2r+1)³)`** —— 位移关闭（恒 0 偏移、bound 0）时与
  /// `apply_brush` 的 Cube **逐格相同**。含 size=1（r=0，单格）与负坐标。
  #[test]
  fn displaced_cube_matches_plain_brush() {
    let zero = |_p: Vec3, _n: Vec3| 0.0f32;
    for (size, center) in
      [(1u32, IVec3::new(0, 0, 0)), (5, IVec3::new(7, -3, 11)), (16, IVec3::new(-9, 40, 3))]
    {
      let r = size as i32 - 1;
      let mut plain = VolumeGrid::new();
      apply_brush(&mut plain, center, BrushShape::Cube, size, PaletteId(3));
      let mut displaced = VolumeGrid::new();
      let st = fill_brush_displaced(
        &mut displaced,
        center,
        BrushShape::Cube,
        size,
        PaletteId(3),
        Displace { f: &zero, bound: 0.0 },
      )
      .expect("Cube 恒有位移实现");
      let span = (2 * r + 1) as usize;
      assert_eq!(st.voxels, span * span * span, "位移关闭时整盒都被写入");
      let lo = center - IVec3::splat(r + 2);
      let hi = center + IVec3::splat(r + 3);
      assert_eq!(dump(&plain, lo, hi), dump(&displaced, lo, hi), "size={size} center={center}");
    }
  }

  /// **交互性能量化**（`#[ignore]`：size 512 单笔数百毫秒，不进常规门禁）：
  /// 位移笔触 vs 同尺寸普通笔触的"一笔"耗时 —— 这是 `consts::EDIT_DISPLACE_SIZE_MAX` 的取值依据
  /// （数字写在那个常量的文档里）。跑法：
  /// `cargo test --release -p gate-app -- --ignored --nocapture displace_brush_cost_by_size`
  #[test]
  #[ignore = "性能量化：size 512 单笔数百毫秒，常规门禁不跑"]
  fn displace_brush_cost_by_size() {
    let mut cache = MaterialDisplaceCache::default();
    let id = crate::consts::DEMO_DISPLACE_HEIGHT_MAP;
    let Some(md) = cache.get_or_load(id, DEMO_DISPLACE_TEX_SCALE) else {
      panic!("`{id}` 的高度图不可用（本量化需要该素材）");
    };
    let f = md.displace_fn();
    let bound = md.bound();
    println!(
      "位移源: 材质 `{id}` 幅度 {} 体素（bound {bound}）—— 空网格上**一笔**的耗时：",
      md.amplitude()
    );
    for size in [16u32, 32, 64, 128, 512] {
      for shape in [BrushShape::Cube, BrushShape::Sphere] {
        let center = IVec3::new(4096, 4096, 4096);
        let mut g = VolumeGrid::new();
        let t0 = Instant::now();
        let n = apply_brush(&mut g, center, shape, size, PaletteId(4));
        let plain = t0.elapsed();
        let mut g2 = VolumeGrid::new();
        let t0 = Instant::now();
        let st = fill_brush_displaced(
          &mut g2,
          center,
          shape,
          size,
          PaletteId(4),
          Displace { f: &f, bound },
        )
        .expect("Cube/Sphere(size>1) 恒有位移实现");
        let displaced = t0.elapsed();
        println!(
          "size={size:>3} {shape:?}: 不位移 {plain:?}（{n} 体素）| 位移 {displaced:?}\
           （{} 体素 = 整块 {} 块 + 壳层 {} 格）= {:.0}×",
          st.voxels,
          st.whole_bricks,
          st.shell_voxels,
          displaced.as_secs_f64() / plain.as_secs_f64().max(1e-9),
        );
      }
    }
  }

  /// MT7-1 + **一刀切**：PBR 变体的 `entry()` 只带 `asset`（5 个覆盖字节全 0 = 不覆盖），
  /// 且编辑器**不再产介质**（`TRANSMISSIVE` 恒不置 —— 要玻璃用平凡变体的「透明度」）。
  #[test]
  fn pbr_entry_carries_asset_only() {
    let pbr = BrushMaterial { pbr: true, asset_slot: 0x0102, ..Default::default() };
    // word0 = 0（5 个覆盖字节全 0）；word1 = asset(0x0102) | IS_PBR(0x10)<<16
    assert_eq!(pack_palette_entry(&pbr.entry()), [0x0000_0000, 0x0010_0102]);
    assert_eq!(pbr.entry().pbr_asset(), 0x0102);
    assert_eq!(
      pack_palette_entry(&pbr.entry())[1] & 0x0020_0000,
      0,
      "编辑器不产介质位（PBR 玻璃只能靠资产，当前资产集不透射）"
    );
    // 平凡参数不参与 PBR payload ⇒ 在 PBR 模式下改它们（控件已置灰）不改变落盘结果
    let noisy = BrushMaterial { color: [9, 9, 9], roughness: 7, metallic: 200, ..pbr };
    assert_eq!(pack_palette_entry(&noisy.entry()), pack_palette_entry(&pbr.entry()));
  }

  /// MT7-3：PBR 变体也按**内容（8B payload）**去重 —— 同内容复用同一槽，内容变了才认领新槽。
  /// 一刀切后 PBR 的内容 = **资产槽号**（+ flags）⇒ 只有换资产才会认领新槽。
  #[test]
  fn pbr_material_dedups_by_payload() {
    let mut grid = VolumeGrid::new();
    let a = BrushMaterial { pbr: true, asset_slot: 10, ..Default::default() };
    let s1 = material_slot(&mut grid, a);
    assert_eq!(pack_palette_entry(grid.palette().get(s1)), [0x0000_0000, 0x0010_000A]);
    assert_eq!(material_slot(&mut grid, a), s1, "同 8B payload ⇒ 复用同一槽");
    // 平凡参数变化**不进** PBR payload ⇒ 仍是同一槽
    assert_eq!(material_slot(&mut grid, BrushMaterial { color: [1, 2, 3], roughness: 42, ..a }), s1);
    // 换资产 ⇒ 认领新槽
    let c = BrushMaterial { asset_slot: 11, ..a };
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

  /// 分帧推进与"一次做完"等价：预算 0（每帧只处理一个块）逐体素结果 + 计数完全一致。
  #[test]
  fn plain_brush_slicing_equals_single_pass() {
    let center = IVec3::new(20, 20, 20);
    for shape in [BrushShape::Sphere, BrushShape::Cube] {
      for size in [1u32, 4, 7, 17] {
        // 参照：一次做完
        let mut once = VolumeGrid::new();
        let n_once = apply_brush(&mut once, center, shape, size, PaletteId(3));
        // 分帧：预算 0 ⇒ 每帧只处理一个块
        let mut sliced = VolumeGrid::new();
        let mut brush = PlainBrush::new(center, shape, size, PaletteId(3));
        let mut frames = 0;
        while !brush.step(&mut sliced, std::time::Duration::ZERO) {
          frames += 1;
          assert!(frames < 100_000, "预算 0 也应在有限帧内做完");
        }
        assert_eq!(brush.changed, n_once, "{shape:?} size={size}：改动计数应一致");
        if size > 1 {
          assert!(frames > 1, "{shape:?} size={size}：预算 0 应跨多帧（实际 {frames}）");
        }
        assert_brush_region_eq(&once, &sliced, center, size, &format!("{shape:?} size={size}"));
        // 无限预算 ⇒ 一帧做完
        let mut one_shot = PlainBrush::new(center, shape, size, PaletteId(3));
        assert!(one_shot.step(&mut VolumeGrid::new(), std::time::Duration::MAX));
        assert_eq!(one_shot.frames, 1);
      }
    }
    // 擦除路径同样等价（挖既有实体：走 Solid 整块写 + Mixed 下钻）
    let mut filled = VolumeGrid::new();
    apply_brush(&mut filled, center, BrushShape::Cube, 31, PaletteId(5));
    let mut once = filled.clone();
    let n_once = apply_brush(&mut once, center, BrushShape::Sphere, 9, PaletteId::AIR);
    let mut sliced = filled;
    let mut brush = PlainBrush::new(center, BrushShape::Sphere, 9, PaletteId::AIR);
    let mut frames = 0;
    while !brush.step(&mut sliced, std::time::Duration::ZERO) {
      frames += 1;
      assert!(frames < 100_000);
    }
    assert_eq!(brush.changed, n_once, "擦除计数应一致");
    assert!(frames > 1, "擦除预算 0 应跨多帧（实际 {frames}）");
    assert_brush_region_eq(&once, &sliced, center, 9, "erase sphere size=9");
  }

  /// 笔触区域 ±1 格内逐体素比对（±1 是为了同时验证"没有多写区域外一格"）
  fn assert_brush_region_eq(a: &VolumeGrid, b: &VolumeGrid, center: IVec3, size: u32, what: &str) {
    let r = size as i32;
    for x in -r - 1..=r + 1 {
      for y in -r - 1..=r + 1 {
        for z in -r - 1..=r + 1 {
          let p = center + IVec3::new(x, y, z);
          let c = VoxelCoord::from_ivec3(p);
          assert_eq!(a.get_voxel(c), b.get_voxel(c), "{what}：体素 {p:?} 不一致");
        }
      }
    }
  }

  /// 可见性判定：外壳全为实体 ⇒ "不可见"（GI 可复用历史）；外壳碰到空气 ⇒ "可见"；空世界恒"可见"。
  #[test]
  fn stroke_hidden_only_when_fully_enclosed() {
    let mut grid = VolumeGrid::new();
    // 32³ 实心块（其余为空气）
    for x in 0..32 {
      for y in 0..32 {
        for z in 0..32 {
          grid.set_voxel_ivec3(IVec3::new(x, y, z), PaletteId(1));
        }
      }
    }
    let inner = IVec3::new(16, 16, 16);
    assert!(stroke_hidden(&grid, BrushShape::Sphere, inner, 8), "块内部 ⇒ 不可见");
    assert!(stroke_hidden(&grid, BrushShape::Cube, inner, 8), "块内部 ⇒ 不可见");
    assert!(
      stroke_hidden(&grid, BrushShape::Cube, inner, 14),
      "外壳最远到 ±15 格，仍在 [0,32) 实体块内 ⇒ 不可见"
    );
    // 越出块外：外壳碰到空气 ⇒ 可见（块面在 ±16 格处）
    assert!(!stroke_hidden(&grid, BrushShape::Cube, inner, 15), "立方外壳压到块面 z=0/32 ⇒ 可见");
    assert!(!stroke_hidden(&grid, BrushShape::Cube, inner, 16), "立方越出块面 ⇒ 可见");
    let near_face = IVec3::new(16, 16, 28);
    assert!(!stroke_hidden(&grid, BrushShape::Sphere, near_face, 8), "球越出块面 ⇒ 可见");
    assert!(!stroke_hidden(&grid, BrushShape::Cube, near_face, 8), "立方越出块面 ⇒ 可见");
    // 空世界：外壳全是空气 ⇒ 可见
    assert!(!stroke_hidden(&VolumeGrid::new(), BrushShape::Sphere, inner, 8), "空世界 ⇒ 可见");
    assert!(!stroke_hidden(&VolumeGrid::new(), BrushShape::Cube, inner, 0), "单格落点也⇒ 可见");
  }

  /// 旧口径的逐格参照笔触（本测试的对照物）：只填空气 / 只挖实体，逐格写。
  fn naive_brush(
    grid: &mut VolumeGrid,
    center: IVec3,
    shape: BrushShape,
    size: u32,
    palette: PaletteId,
  ) -> usize {
    let r = brush_radius(size);
    let erase = palette.is_air();
    let mut changed = 0usize;
    for x in -r..=r {
      for y in -r..=r {
        for z in -r..=r {
          let d = IVec3::new(x, y, z);
          if !brush_contains(shape, d, r) {
            continue;
          }
          let p = center + d;
          let cur = grid.get_voxel(VoxelCoord::from_ivec3(p)).unwrap_or(PaletteId::AIR);
          if cur.is_air() != erase && grid.set_voxel_ivec3(p, palette).is_some() {
            changed += 1;
          }
        }
      }
    }
    changed
  }

  /// 4³ 批量写路径与"逐格写"的旧口径等价：空世界放置 / 实心块擦除 / 贴面擦除，逐体素 + 计数一致。
  #[test]
  fn plain_brush_batched_writes_match_naive_reference() {
    let center = IVec3::new(20, 20, 20);
    for shape in [BrushShape::Sphere, BrushShape::Cube] {
      // 放置：空世界
      for size in [1u32, 2, 4, 5, 8, 9, 17, 33] {
        let mut fast = VolumeGrid::new();
        let n_fast = apply_brush(&mut fast, center, shape, size, PaletteId(3));
        let mut slow = VolumeGrid::new();
        let n_slow = naive_brush(&mut slow, center, shape, size, PaletteId(3));
        assert_eq!(n_fast, n_slow, "place {shape:?} size={size}：改动计数不一致");
        assert_brush_region_eq(&fast, &slow, center, size, &format!("place {shape:?} {size}"));
      }
      // 擦除：先在笔触外扩一圈铺实体（让外壳/内部都非空），再挖
      for size in [4u32, 9, 17] {
        let mut base = VolumeGrid::new();
        naive_brush(&mut base, center, shape, size + 8, PaletteId(7));
        let mut fast = base.clone();
        let n_fast = apply_brush(&mut fast, center, shape, size, PaletteId::AIR);
        let mut slow = base;
        let n_slow = naive_brush(&mut slow, center, shape, size, PaletteId::AIR);
        assert_eq!(n_fast, n_slow, "erase {shape:?} size={size}：改动计数不一致");
        assert_brush_region_eq(&fast, &slow, center, size + 8, &format!("erase {shape:?} {size}"));
      }
    }
  }

  /// 性能记录（忽略型，常规门禁不跑）：castle.vox 上"一笔"的固定成本。
  ///
  /// - 笔触本身：4³ 批量写之前（逐格写）`size=61 place 33.17ms / erase 20.34ms`，现在 ≈5ms；
  /// - 序列化（**只在全量安装 / 整棵重建时付**）：59 chunk 合计 173–208ms（均值 ≈2.9ms / 最大 ≈8.3ms）；
  /// - 增量上传（节点级重写，**每笔都付**）：单格 356B / 13.7µs，size=4 → 1.6KB / 4.4µs，
  ///   size=9 → 10KB / 17.9µs，size=17 → 39KB / 51.7µs（旧实现：每笔整棵 chunk 树 ≈2MB + 2.9ms）。
  ///
  /// 跑法：`cargo test --release -p gate-app brush_cost_on_castle -- --ignored --nocapture`
  #[test]
  #[ignore = "性能记录：需读 assets/vox/castle.vox"]
  fn brush_cost_on_castle() {
    let path = gate_render::assets_dir().join("vox").join("castle.vox");
    let mut volumes = gate_voxel::Volumes::new(VolumeGrid::new());
    let info = crate::vox_scene::load_vox_scene(volumes.main_mut(), &path, IVec3::ZERO)
      .expect("载入 castle");
    let (mut lo, mut hi) = (IVec3::splat(i32::MAX), IVec3::splat(i32::MIN));
    for c in volumes.main().chunk_coords() {
      lo = lo.min(c.0);
      hi = hi.max(c.0);
    }
    let (wlo, whi) = (lo * 256, (hi + IVec3::ONE) * 256);
    let origin = glam::Vec3::new(
      (whi.x + 96) as f32,
      ((wlo.y + whi.y) / 2) as f32,
      ((wlo.z + whi.z) / 2) as f32,
    );
    let hit =
      gate_render::raycast(&volumes, origin, glam::Vec3::NEG_X, EDIT_REACH).expect("射线应命中");
    println!(
      "hit voxel={:?} face={:?} | instances={} voxels={}",
      hit.voxel, hit.face, info.instances_used, info.voxels_written
    );
    for size in [17u32, 33, 61, 121] {
      let c = hit.voxel + hit.face; // 与落笔同一公式（off = 1）
      let t0 = Instant::now();
      let n = apply_brush(volumes.main_mut(), c, BrushShape::Sphere, size, PaletteId(2));
      let place = t0.elapsed();
      let t1 = Instant::now();
      let m = apply_brush(volumes.main_mut(), c, BrushShape::Sphere, size, PaletteId::AIR);
      let erase = t1.elapsed();
      println!("size={size:4} place {place:>10.2?} ({n}vx) / erase {erase:>10.2?} ({m}vx)");
    }
    // 序列化成本（增量路径每笔对"碰到的那个 chunk"做一次）
    let mut times: Vec<(usize, std::time::Duration)> = volumes
      .main()
      .chunk_coords()
      .filter_map(|cc| volumes.main().chunk(cc))
      .map(|tree| {
        let t = Instant::now();
        let words = tree.serialize().len();
        (words, t.elapsed())
      })
      .collect();
    times.sort_by_key(|(w, _)| std::cmp::Reverse(*w));
    let total: std::time::Duration = times.iter().map(|(_, d)| *d).sum();
    let max = times.first().copied().unwrap_or_default();
    println!(
      "序列化 chunks={} 总计={:?} 均值={:?} 最大={:?}（{}字）",
      times.len(),
      total,
      total / times.len().max(1) as u32,
      max.1,
      max.0
    );

    // 增量上传：节点级重写后**实际写进 GPU 的 struct 字节**（旧实现=整棵 chunk 树 ≈1.4MB/笔）。
    // 先消化掉建场景与上面那些大笔触留下的标记（含 reset），只看"一笔小编辑"的量。
    let mut vbuild = gate_render::brickmap::VolumesBuilder::build_full(&volumes);
    let first = vbuild.snapshot();
    println!("全量 mode={} struct={}KB", first.mode_tag, first.struct_total_bytes / 1024);
    let all: Vec<_> = volumes.main().chunk_coords().collect();
    for cc in all {
      let dirty = volumes.main_mut().chunk_mut(cc).map(|t| t.take_dirty()).unwrap_or_default();
      vbuild.update_chunk(&volumes, 0, cc, &dirty, true);
    }
    let _ = volumes.main_mut().dirty.drain_data_budget(usize::MAX);
    let _ = vbuild.snapshot();

    for (label, size) in [("单格", 0u32), ("size=4", 4), ("size=9", 9), ("size=17", 17)] {
      let c = hit.voxel + hit.face + IVec3::new(size as i32, 0, 0);
      let n = if size == 0 {
        usize::from(volumes.main_mut().set_voxel_ivec3(c, PaletteId(2)).is_some())
      } else {
        apply_brush(volumes.main_mut(), c, BrushShape::Sphere, size, PaletteId(2))
      };
      let coords = volumes.main_mut().dirty.drain_data_budget(4096);
      let (mut marks, mut chunks, mut reset) = (0usize, 0usize, 0usize);
      let t0 = Instant::now();
      for cc in coords {
        let dirty = volumes.main_mut().chunk_mut(cc).map(|t| t.take_dirty()).unwrap_or_default();
        marks += dirty.nodes.len();
        reset += usize::from(dirty.reset);
        chunks += 1;
        vbuild.update_chunk(&volumes, 0, cc, &dirty, true);
      }
      let cpu = t0.elapsed();
      let snap = vbuild.snapshot();
      let bytes: usize = snap.struct_blobs.iter().map(|(_, p)| p.len()).sum();
      println!(
        "{label:>7} ({n}vx) → 脏 chunk {chunks} / 节点标记 {marks} / reset {reset} / \
         上传 {bytes}B / mode={} / builder {cpu:>10.2?}",
        snap.mode_tag,
      );
    }
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

  /// 按住连发：按下当帧立即落一笔；之后每 `EDIT_REPEAT_SECS` 一笔；松开清零、再按下又立即触发。
  /// 触发次数**与帧率无关**（余数结转 ⇒ 60fps 与 240fps 同样时长内触发次数相同）。
  #[test]
  fn hold_repeat_fires_immediately_then_on_interval() {
    let dt = 0.02f32;
    let mut acc = 0.0f32;
    assert!(hold_repeat(true, true, &mut acc, dt), "按下当帧立即触发");
    // 头 4 帧（0.08s）不足一个间隔 ⇒ 一笔都不该落
    for _ in 0..4 {
      assert!(!hold_repeat(false, true, &mut acc, dt), "不足 {EDIT_REPEAT_SECS} 不该触发");
    }
    // 按住 1s：触发次数 ≈ 1s / 间隔（按下那笔不算），只准 ±1（浮点余数）
    let over_1s = 1.0f32 / dt;
    let expect = (1.0 / EDIT_REPEAT_SECS).floor() as usize;
    let mut got = 0usize;
    for _ in 4..over_1s as usize {
      got += hold_repeat(false, true, &mut acc, dt) as usize;
    }
    let lo = expect.saturating_sub(1);
    assert!((lo..=expect + 1).contains(&got), "1s 内触发 {got} 次，预期 {expect}±1");
    // 松开清零 → 再按下又立即触发
    assert!(!hold_repeat(false, false, &mut acc, dt));
    assert!(hold_repeat(true, true, &mut acc, dt));

    // 帧率无关：同样 1s（含按下当帧），60fps 与 240fps 的触发次数一致
    let (mut slow, mut fast) = (0.0f32, 0.0f32);
    let (mut n_slow, mut n_fast) = (0usize, 0usize);
    hold_repeat(true, true, &mut slow, 1.0 / 60.0);
    hold_repeat(true, true, &mut fast, 1.0 / 240.0);
    for _ in 0..59 {
      n_slow += hold_repeat(false, true, &mut slow, 1.0 / 60.0) as usize;
    }
    for _ in 0..239 {
      n_fast += hold_repeat(false, true, &mut fast, 1.0 / 240.0) as usize;
    }
    assert_eq!(n_slow, n_fast, "60fps {n_slow} 次 vs 240fps {n_fast} 次");
    assert!(n_slow > 5, "1s 内至少该触发若干次（实际 {n_slow}）");
  }
}
