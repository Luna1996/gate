//! `infinite_cubes` 世界（规则见 `docs/infinite_cubes.md`）：向六个方向无限生长的 3m room 网格 ——
//! 每条棱是 50cm 截面的纯白柱体、每个 room 中心一个 90cm cube，cube 材质由 room 坐标当种子随机。
//!
//! 本模块只负责**按坐标生成体素**（[`build_region`]）：这是 M5 `ChunkSource` 的雏形 ——
//! 真流式版本把"读系统文件"那一步换成调用它，其余（挂载 / 常驻 / 卸载）都走真实流程。
//!
//! **尺度**：1 体素 = 2cm（`README` §4）。为对齐 4³ brick（批量填充的前提；不对齐的话生成成本会从
//! µs 级掉到 ms 级）取 **4 的倍数**：
//! - room = **152 体素 = 3.04m**（规格 3m）
//! - 棱柱截面 = **24 体素 = 48cm**（规格 50cm）
//! - cube = **44 体素 = 88cm**（规格 90cm；落格到 brick 对齐 ⇒ 相对 room 中心最多偏 3 体素 = 6cm）
//!
//! **材质**：等概率六选一（PBR / 光源 / 玻璃 / 镜面 / 金属 / 普通）；**除 PBR 外各类都生成颜色**，
//! 再各自生成该类描述的那一项值，其余保持默认。每类的具体值由种子量化成**有限档位**（每类 ≤ 8 档）
//! ⇒ 调色板槽数有界：room 数量无限，不能一 room 一槽。

use bevy::prelude::{Res, ResMut, Resource};
use gate_voxel::{
  PALETTE_INDEX_MAX, PaletteEntry, PaletteFlags, PaletteId, PbrOverrides, VolumeGrid,
  fill_bricks,
};
use glam::IVec3;

/// room 边长（体素）。取 4 的倍数以便 brick 对齐。
pub const ROOM: i32 = 152;
/// 棱柱截面边长（体素，纯白）。
pub const COLUMN: i32 = 24;
/// room 中心 cube 的边长（体素）。
pub const CUBE: i32 = 44;
/// 柱体的固定材质槽（1 = 纯白；0 = 空气）。
pub const WHITE_SLOT: PaletteId = PaletteId(1);

/// 起始视野的半径（chunk 数）：静态首建时只铺这么多，其余由 [`stream_chunks`] 按需生成。
pub const START_CHUNKS: i32 = 1;

/// 流式驱动参数（主世界资源）。**是否启用由 `grid.stream_window()` 判定** —— 只有
/// [`crate::scene::build_infinite_cubes`] 会设置它，换世界时新 `VolumeGrid` 自然是 `None`。
#[derive(Resource)]
pub struct Streaming {
  /// 加载半径（chunk，切比雪夫）
  pub load_radius: i32,
  /// 卸载半径（chunk）：必须 > 加载半径（迟滞，免边界反复装卸）
  pub unload_radius: i32,
  /// 每帧最多生成几个 chunk（生成是 ms 级一次性成本，必须摊帧）
  pub per_frame: usize,
}

impl Default for Streaming {
  fn default() -> Self {
    // 加载半径 2 chunk ≈ 10m：一个 chunk 是 256³ 体素（5.12m）、树约 1.5MB ⇒ 半径再大就白付内存。
    // 卸载半径只比加载半径大 1：迟滞够用，且把**常驻集真正限住**（相机一路飞过去的"尾迹"会一直
    // 堆到卸载半径外才回收；差 2 就会留下 9³ 的尾迹 ≈ 700MB）。
    Self { load_radius: 2, unload_radius: 3, per_frame: 1 }
  }
}

/// **流式加载 / 卸载**（`infinite_cubes` 世界）：按相机位置生成 chunk、超出半径就**真卸载**
/// （CPU 树一起丢 ⇒ 卸载区里的修改永久丢失，与"不落盘"的语义一致）。
///
/// 与真 ray-guided 流程的唯一区别：这里的**需求**来自相机距离，而不是渲染 pass 发回请求
/// （请求通道见 `docs/editable-gigavoxel.md` §9 M4）。生成 / 挂载 / 常驻 / 换出 / 卸载全走既有流水线：
/// 生成直接写主世界对应 chunk 盒（`fill_bricks` 自会标脏 ⇒ 走既有上传路径），
/// 卸载只拿掉 CPU 树，显存由常驻调度的反向同步（`plan_residency` 第 ⑤ 步）跟着归还。
pub fn stream_chunks(
  stream: ResMut<Streaming>,
  mut scene: ResMut<gate_render::VoxelScene>,
  cam: Option<Res<gate_render::DdaCameraConfig>>,
) {
  let Some(cam) = cam else { return };
  let Some((mut w_origin, w_dims)) = scene.volumes.main().stream_window() else {
    return; // 非流式世界
  };
  let chunk = gate_voxel::CHUNK_SIZE;
  let center = (cam.position_world / chunk as f32).floor().as_ivec3();

  // ⓪ 窗口跟着相机。窗口是 `b_struct` 索引区的**定义域**：跑出窗口的 chunk 生成得出、却装不上 GPU
  //    （索引区之外），于是画面出现成片的空洞与"齐平断口"。最小可用版本 = 相机离窗口边 < MARGIN
  //    就整块重定窗口（丢掉全部 CPU chunk + 全量重传，一次性卡顿）；M6 的环形窗口才是根治。
  const MARGIN: i32 = 6;
  let local = center - w_origin;
  if local.min_element() < MARGIN || (w_dims - IVec3::ONE - local).min_element() < MARGIN {
    w_origin = center - w_dims / 2;
    let grid = scene.volumes.main_mut();
    for c in grid.chunk_coords().collect::<Vec<_>>() {
      grid.unmount_chunk(c); // 旧窗口的坐标系作废 ⇒ 已加载内容全部丢弃（有本地修改也一样，符合不落盘语义）
    }
    grid.set_stream_window(Some((w_origin, w_dims)));
    scene.demo_force_full_rebuild = true;
    bevy::log::info!("STREAM[窗口重定 origin {} 相机 {}]", w_origin, center);
    return; // 本帧只做重定，生成从下一帧开始
  }
  let grid = scene.volumes.main_mut();

  // ① 生成：半径内**且在窗口内**尚未挂载的 chunk，近的优先，每帧 ≤ per_frame
  let r = stream.load_radius;
  let mut todo: Vec<(i32, IVec3)> = Vec::new();
  let mut generated = 0usize;
  for dx in -r..=r {
    for dy in -r..=r {
      for dz in -r..=r {
        let c = center + IVec3::new(dx, dy, dz);
        let t = c - w_origin;
        if t.cmplt(IVec3::ZERO).any() || t.cmpge(w_dims).any() {
          continue; // 窗口外：生成了也装不上
        }
        if grid.chunk(gate_voxel::ChunkCoord(c)).is_none() {
          todo.push((dx.abs().max(dy.abs()).max(dz.abs()), c));
        }
      }
    }
  }
  todo.sort_unstable_by_key(|(d, c)| (*d, c.x, c.y, c.z));
  for (_, c) in todo.into_iter().take(stream.per_frame) {
    let lo = c * chunk;
    // TODO(streaming): 传 `PbrTextureSet::ids()`（贴图集就绪后 PBR 那一档才不会退化成普通色）
    build_region(grid, lo, lo + IVec3::splat(chunk), &[]);
    generated += 1;
  }

  // ② 卸载：超出卸载半径、或已落到窗口外的，都真卸载
  let far: Vec<gate_voxel::ChunkCoord> = grid
    .chunk_coords()
    .filter(|c| {
      let t = c.0 - w_origin;
      t.cmplt(IVec3::ZERO).any()
        || t.cmpge(w_dims).any()
        || (c.0 - center).abs().max_element() > stream.unload_radius
    })
    .collect();
  let unloaded = far.len();
  for cc in far {
    grid.unmount_chunk(cc);
  }
  if generated > 0 || unloaded > 0 {
    bevy::log::debug!("STREAM[gen {generated} unload {unloaded} chunks {}]", grid.chunk_count());
  }
}

/// room 坐标 → 确定性随机流。**同一个 room 在任何机器、任何时候都得到同一材质**。
fn room_hash(room: IVec3) -> u64 {
  let mut h = 0x9E37_79B9_7F4A_7C15u64;
  let keys = [0x9E37_79B1_85EB_CA87u64, 0xC2B2_AE3D_27D4_EB4Fu64, 0x1656_67B1_9E37_79F9u64];
  for (k, v) in keys.into_iter().zip([room.x, room.y, room.z]) {
    h ^= (v as i64 as u64).wrapping_mul(k);
    h = h.rotate_left(31).wrapping_mul(0x9E37_79B9_7F4A_7C15);
  }
  h ^= h >> 30;
  h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
  h ^= h >> 27;
  h.wrapping_mul(0x94D0_49BB_1331_11EB) ^ (h >> 31)
}

/// 六个材质类（等概率）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Kind {
  Pbr,
  Light,
  Glass,
  Mirror,
  Metal,
  Plain,
}

impl Kind {
  fn of(h: u64) -> Self {
    match h % 6 {
      0 => Kind::Pbr,
      1 => Kind::Light,
      2 => Kind::Glass,
      3 => Kind::Mirror,
      4 => Kind::Metal,
      _ => Kind::Plain,
    }
  }
}

/// 8 档量化色相（非 PBR 各类共用）⇒ 每类最多 8 个调色板槽。
fn hue(h: u64, shift: u32) -> [u8; 3] {
  const HUES: [[u8; 3]; 8] = [
    [0xE8, 0x4C, 0x3C],
    [0xF2, 0x9E, 0x2C],
    [0xF2, 0xE2, 0x3C],
    [0x5C, 0xD6, 0x4C],
    [0x3C, 0xC8, 0xC8],
    [0x4C, 0x7C, 0xF2],
    [0x9C, 0x5C, 0xE8],
    [0xE8, 0x5C, 0xB4],
  ];
  HUES[((h >> shift) % 8) as usize]
}

/// 8 档量化档位（值 = `base + 档 × step`）——用来把"生成的连续值"压成**有限**槽数。
fn tier(h: u64, shift: u32, base: u8, step: u8) -> u8 {
  base.saturating_add(((h >> shift) % 8) as u8 * step)
}

/// 该 room 的 cube 材质。**严格按 `docs/infinite_cubes.md` 规则 5**：等概率六选一，
/// **除 PBR 外各类都生成颜色**，再各自生成描述的那一项值，其余保持 `PaletteEntry::default()`
/// （不发光 / 不透明 / 半粗糙 / 非金属）。
///
/// `pbr_ids` = 可用的 PBR 材质列表（`PbrTextureSet::ids()`）；为空 ⇒ PBR 那一档退化成"普通"
/// （贴图集还没扫出来时不该凭空造 asset 槽号）。
pub fn material_of(room: IVec3, pbr_ids: &[String]) -> PaletteEntry {
  let h = room_hash(room);
  match Kind::of(h) {
    // 唯一不生成颜色的一类：颜色由 PBR 贴图决定。
    Kind::Pbr if !pbr_ids.is_empty() => {
      let asset = ((h >> 4) % pbr_ids.len() as u64).min(u16::MAX as u64) as u16;
      PaletteEntry::pbr(asset, PbrOverrides::default(), PaletteFlags::default())
    }
    Kind::Light => PaletteEntry { color: hue(h, 16), emissive: tier(h, 8, 96, 20), ..Default::default() },
    Kind::Glass => PaletteEntry { color: hue(h, 16), transmission: tier(h, 8, 152, 12), ..Default::default() },
    Kind::Mirror => PaletteEntry { color: hue(h, 16), roughness: tier(h, 8, 0, 3), ..Default::default() },
    Kind::Metal => PaletteEntry { color: hue(h, 16), metallic: 255, roughness: tier(h, 8, 0, 8), ..Default::default() },
    Kind::Pbr | Kind::Plain => PaletteEntry { color: hue(h, 16), ..Default::default() },
  }
}

/// room 中心 cube 的轴对齐体素范围（brick 对齐后）。
fn cube_range(room: IVec3) -> (IVec3, IVec3) {
  let min = ((room * ROOM + IVec3::splat(ROOM / 2) - IVec3::splat(CUBE / 2)) / 4) * 4;
  (min, min + IVec3::splat(CUBE))
}

/// 内容去重的调色板槽：找内容相同的已有槽，否则占用第一个空槽（0 = 空气、1 = 柱体白，从 2 起找）。
fn slot_for(grid: &mut VolumeGrid, entry: PaletteEntry) -> PaletteId {
  let pal = grid.palette_mut();
  for i in 2..=PALETTE_INDEX_MAX as u16 {
    let id = PaletteId(i);
    if pal.is_empty_slot(id) {
      pal.set(id, entry);
      return id;
    }
    if *pal.get(id) == entry {
      return id;
    }
  }
  panic!("调色板槽用尽（infinite_cubes 的材质档位应 ≪ {PALETTE_INDEX_MAX}）");
}

/// 把 `[lo, hi)` 体素范围内的 `infinite_cubes` 结构填进 `grid`（要求 lo/hi 各轴是 4 的倍数）。
///
/// 三段：① 三条轴向的棱柱（room 棱上的无限柱体在本范围内的那一段）② 每个 room 中心的 cube
/// ③ 其余留空。槽 1 由调用方先设成纯白（见 `scene::build_infinite_cubes`）。
/// 返回**覆盖的体素数**（各填充盒体积之和；柱体与 cube 重叠处会重复计一次）。
pub fn build_region(grid: &mut VolumeGrid, lo: IVec3, hi: IVec3, pbr_ids: &[String]) -> u64 {
  debug_assert!(lo % 4 == IVec3::ZERO && hi % 4 == IVec3::ZERO, "区域须 brick 对齐 4");
  let (span, half) = (hi - lo, COLUMN / 2);
  let mut covered = 0u64;
  let first = |v: i32| (v - 1).div_euclid(ROOM);
  let last = |v: i32| (v + 1).div_euclid(ROOM);
  // ① 棱柱。三条轴各一组：柱轴 = 长边，截面 = 另两轴的 24×24。
  let edges = |a0: i32, a1: i32| (first(a0)..=last(a1)).collect::<Vec<_>>();
  let col_area = (COLUMN * COLUMN) as u64;
  for &a in &edges(lo.x, hi.x) {
    for &b in &edges(lo.y, hi.y) {
      // 沿 Z
      fill_bricks(grid, IVec3::new(a * ROOM - half, b * ROOM - half, lo.z), IVec3::new(COLUMN, COLUMN, span.z), 4, WHITE_SLOT);
      covered += col_area * span.z as u64;
    }
    for &b in &edges(lo.z, hi.z) {
      // 沿 Y
      fill_bricks(grid, IVec3::new(a * ROOM - half, lo.y, b * ROOM - half), IVec3::new(COLUMN, span.y, COLUMN), 4, WHITE_SLOT);
      covered += col_area * span.y as u64;
    }
  }
  for &a in &edges(lo.y, hi.y) {
    for &b in &edges(lo.z, hi.z) {
      // 沿 X
      fill_bricks(grid, IVec3::new(lo.x, a * ROOM - half, b * ROOM - half), IVec3::new(span.x, COLUMN, COLUMN), 4, WHITE_SLOT);
      covered += col_area * span.x as u64;
    }
  }
  // ② cube（材质由 room 坐标定）。
  let cube_vol = (CUBE * CUBE * CUBE) as u64;
  for rx in (lo.x - ROOM).div_euclid(ROOM)..=(hi.x + ROOM).div_euclid(ROOM) {
    for ry in (lo.y - ROOM).div_euclid(ROOM)..=(hi.y + ROOM).div_euclid(ROOM) {
      for rz in (lo.z - ROOM).div_euclid(ROOM)..=(hi.z + ROOM).div_euclid(ROOM) {
        let room = IVec3::new(rx, ry, rz);
        let (cmin, cmax) = cube_range(room);
        if cmax.cmple(lo).any() || cmin.cmpge(hi).any() {
          continue;
        }
        let id = slot_for(grid, material_of(room, pbr_ids));
        let fmin = cmin.max(lo);
        fill_bricks(grid, fmin, cmax.min(hi) - fmin, 4, id);
        covered += cube_vol;
      }
    }
  }
  covered
}

#[cfg(test)]
mod tests {
  use super::*;
  use gate_voxel::VoxelCoord;
  use gate_voxel::PaletteEntry;

  fn build(lo: IVec3, hi: IVec3) -> VolumeGrid {
    let mut grid = VolumeGrid::new();
    grid.palette_mut().set(WHITE_SLOT, PaletteEntry { color: [0xFF, 0xFF, 0xFF], ..Default::default() });
    build_region(&mut grid, lo, hi, &[]);
    grid
  }

  fn solid(g: &VolumeGrid, p: IVec3) -> bool {
    g.get_voxel(VoxelCoord::new(p.x, p.y, p.z)).is_some()
  }

  /// 规则 1/2：棱上有柱体，离开棱（且不在 cube 内）留空。
  #[test]
  fn columns_sit_on_room_edges() {
    let g = build(IVec3::splat(-304), IVec3::splat(304));
    for p in [IVec3::new(0, 0, 37), IVec3::new(0, 0, -37), IVec3::new(ROOM, 0, 5), IVec3::new(0, ROOM, 5), IVec3::new(5, ROOM, ROOM)] {
      assert!(solid(&g, p), "棱上 {p:?} 应是柱体");
    }
    // room 内部、离棱与 cube 都远 ⇒ 空
    let empty = IVec3::new(40, 40, 40);
    assert!(!solid(&g, empty), "{empty:?} 应留空");
    assert_eq!(g.palette().get(WHITE_SLOT).color, [0xFF, 0xFF, 0xFF], "柱体是纯白");
  }

  /// 规则 3/4：room 中心有 cube，材质由 room 坐标决定且**可重复**。
  #[test]
  fn cube_is_at_room_center_and_deterministic() {
    let g = build(IVec3::splat(-304), IVec3::splat(304));
    let (cmin, cmax) = cube_range(IVec3::ZERO);
    let c = (cmin + cmax) / 2;
    assert!(solid(&g, c), "room 中心 {c:?} 应是 cube");
    assert!(!solid(&g, cmin - IVec3::X), "cube 之外应留空");
    assert_eq!(cmax - cmin, IVec3::splat(CUBE));
    assert_eq!(material_of(IVec3::new(3, -7, 11), &[]), material_of(IVec3::new(3, -7, 11), &[]), "同 room 同材质");
    let kinds: std::collections::HashSet<_> = (0..96).map(|i| Kind::of(room_hash(IVec3::new(i, 0, 0)))).collect();
    assert!(kinds.len() >= 5, "六类应都出现过（实际 {} 类）", kinds.len());
  }

  /// 规则 5：**除 PBR 外都生成颜色**；此外只动自己那一项（普通类只动颜色）。
  #[test]
  fn material_kinds_only_touch_their_own_fields() {
    let d = PaletteEntry::default();
    for i in 0..512 {
      let e = material_of(IVec3::new(i, i * 7, i * 13), &[]);
      if e.flags.contains(PaletteFlags::IS_PBR) {
        continue; // PBR 变体：不生成颜色，emissive/transmission 是 asset 的两个字节
      }
      assert_ne!(e.color, d.color, "非 PBR 必须生成颜色：{e:?}");
      let extra = [
        e.emissive != d.emissive,
        e.transmission != d.transmission,
        e.roughness != d.roughness,
        e.metallic != d.metallic,
      ];
      assert!(extra.iter().filter(|x| **x).count() <= 2, "第 {i} 个 room 动了太多属性：{e:?}");
    }
  }
}
