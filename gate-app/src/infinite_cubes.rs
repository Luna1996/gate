//! `infinite_cubes` 世界（规则见 `docs/infinite_cubes.md`）：向六个方向无限生长的 3m room 网格 ——
//! 每条棱是 50cm 截面的纯白柱体、每个 room 中心一个 90cm cube，cube 材质由 room 坐标当种子随机。
//!
//! 本模块只负责**按坐标生成体素**（[`build_region`]）与**流式驱动**（[`stream_chunks`]）：
//! 生成走 M5 的生产管线（[`InfiniteCubes`] 实现 `gate_voxel::ChunkSource`，worker 线程并行产出），
//! 挂载 / 常驻 / 卸载全走真实流程（真 world 只是把"读系统文件"那一步换成了本模块的生成函数）。
//!
//! **尺度**：1 体素 = 2cm（`README` §4）。为对齐 4³ brick（批量填充的前提；不对齐的话生成成本会从
//! µs 级掉到 ms 级）取 **4 的倍数**：
//! - room = **148 体素 = 2.96m**（规格 3m）—— 4 的倍数里离 3m 最近、又满足 `(room − cube) % 8 == 0`
//!   （cube 偏移既 4 对齐又精确居中）的那个取值
//! - 棱柱截面 = **24 体素 = 48cm**（规格 50cm）
//! - cube = **44 体素 = 88cm**（规格 90cm），落在 room 几何中心（负坐标 room 同样）
//!
//! **生成盒契约**：[`build_region`] 只写 `[lo, hi)` 内的体素（越界那份属于邻块、由邻块自己写），
//! 且要求 lo/hi **对齐 chunk** —— 半生成的 chunk 会被 `stream_chunks` 当成"已生成"跳过，
//! 而它里面缺的那部分永远补不回来（画面上的缺口与齐平断口）。起始铺块据此用 [`initial_box`]。
//!
//! **材质**：等概率六选一（PBR / 光源 / 玻璃 / 镜面 / 金属 / 普通）；**除 PBR 外各类都生成颜色**
//! （8 档色相），再各自生成该类描述的那一项值，其余保持默认。那项值**每类只两档**：
//! 光源自发光 `127 | 255`、玻璃透明度 `127 | 191`、镜面与金属的粗糙度 `0 | 127`（金属度恒 255）。
//! 槽号由 [`slot_of`] 的**固定方案**给出（与生成顺序、线程调度无关 —— 后台生产的前提），
//! 整表由 [`material_slots`] 在建世界时一次装进调色板。

use bevy::prelude::{Res, ResMut, Resource};
use gate_voxel::{
  ChunkCoord, ChunkProducer, ChunkSource, ChunkTree, Detail, PaletteEntry, PaletteFlags, PaletteId,
  PbrOverrides, VolumeGrid, fill_bricks,
};
use glam::{IVec3, Vec3};

/// room 边长（体素）：4 的倍数，且 `(ROOM − CUBE) % 8 == 0`（见下面的断言）。
pub const ROOM: i32 = 148;
/// 棱柱截面边长（体素，纯白）。
pub const COLUMN: i32 = 24;
/// room 中心 cube 的边长（体素）。
pub const CUBE: i32 = 44;
/// cube 偏移 `(ROOM − CUBE) / 2` 必须同时"是 4 的倍数"（brick 对齐）与"整数"（精确居中）
/// ⇒ `ROOM − CUBE` 是 8 的倍数。
const _: () = assert!(ROOM % 4 == 0 && CUBE % 4 == 0 && (ROOM - CUBE) % 8 == 0);
/// 柱体的固定材质槽（1 = 纯白；0 = 空气）。
pub const WHITE_SLOT: PaletteId = PaletteId(1);

/// 起始视野的半径（chunk 数）：静态首建时只铺这么多，其余由 [`stream_chunks`] 按需生成。
pub const START_CHUNKS: i32 = 1;

/// 起始铺块的体素盒（[`crate::scene::build_infinite_cubes`] 用）：以 `center` 所在 chunk 为中心、
/// 半径 [`START_CHUNKS`] 个 chunk，各轴**整 chunk 对齐** —— 半块会被 [`stream_chunks`] 当成
/// "已生成"跳过（见模块头的生成盒契约）。
pub(crate) fn initial_box(center: IVec3) -> (IVec3, IVec3) {
  let chunk = IVec3::splat(gate_voxel::CHUNK_SIZE);
  let c = center.div_euclid(chunk);
  ((c - IVec3::splat(START_CHUNKS)) * chunk, (c + IVec3::splat(START_CHUNKS + 1)) * chunk)
}

/// 生产管线的 worker 线程数（M5）：生成是纯 CPU、与主线程争核 ⇒ 给 3 个够把"每帧 1 个"的瓶颈
/// 打开，又不会在 4/8 核机器上把渲染挤掉。
const PRODUCER_WORKERS: usize = 3;

/// 每帧最多**派发**几条需求（进 worker 队列）：派发本身极便宜（一次哈希插入 + 一次 channel send），
/// 所以给得比挂载预算大 —— 让挂载预算（主线程真正的成本）成为唯一的节流阀。
const DISPATCH_PER_FRAME: usize = 128;

/// 需求表（已排序的候选清单）一次最多装几条：比粗档环的 chunk 数大即可（装满整个环）。
const DEMAND_TABLE_MAX: usize = 8192;

/// 需求表**重建周期**（帧）：相机不换 chunk 时也要定期重建，好吸收 ray-guided 请求的更新
/// （请求每 `REPORT_PERIOD_SECS` 换一批）与卸载腾出的空位。重建要扫环 + 排序（数千项），
/// 每帧做就是白花 ~1ms 帧时间，所以缓存起来。
const DEMAND_REBUILD_FRAMES: u32 = 30;

/// 每帧最多取回几条产出（取回本身极便宜，挂载才是要摊帧的那一步）
const POLL_MAX: usize = 256;

/// 生产管线的活状态（M5）：worker 池 + 需求表 + 待挂载队列 + 各 chunk 的**现有档位** + 空产出表。
struct Pipeline {
  producer: ChunkProducer,
  /// **需求表**（按 [`plan_generation`] 的键排好序）：只在相机换 chunk / 每
  /// [`DEMAND_REBUILD_FRAMES`] 帧重建；每帧只从表头派发（已满足 / 已在飞的当场划过）
  demand: std::collections::VecDeque<(IVec3, Detail)>,
  /// 建表时的相机 chunk（None = 还没建过）
  demand_center: Option<IVec3>,
  /// 距上次建表过了几帧
  demand_age: u32,
  /// 已取回、还没挂载的产出（挂载按**字数预算**逐帧消化，见 [`Streaming::mount_words`]）
  ready: std::collections::VecDeque<(ChunkCoord, Detail, ChunkTree)>,
  /// 已挂载 chunk 的档位（粗 / 细）：判"够不够细"（要细化就重新产出，挂载会整体替换）。
  /// 建世界时同步铺的起始块不在表里 —— 缺省按 [`Detail::Full`] 算（它们确实是全分辨率）。
  detail: std::collections::HashMap<ChunkCoord, Detail>,
  /// 产出过、但**没有内容**的 chunk（免得每帧重复派发；infinite_cubes 不会出现，留作通用性）
  empty: std::collections::HashSet<ChunkCoord>,
}

/// 流式驱动参数（主世界资源）。**是否启用由 `grid.stream_window()` 判定** —— 只有
/// [`crate::scene::build_infinite_cubes`] 会设置它，换世界时新 `VolumeGrid` 自然是 `None`。
///
/// 尺度实测（infinite_cubes，`coarse_detail_is_quantized_and_small` 同一批数据）：全分辨率
/// ≈ **69.4 K 字/chunk（278 KB）**、粗档（16³ 量化）≈ **2.9 K 字/chunk（11.7 KB）** ⇒ 粗档便宜 24 倍。
#[derive(Resource)]
pub struct Streaming {
  /// **暂停流式**（调试开关）：为 `true` 时 [`stream_chunks`] 整段不跑 —— 不跟窗、不派发、不挂载、
  /// 不卸载 ⇒ 已加载的那一圈冻结在原地，相机可以直接飞出去看"世界到此为止"的加载边界。
  pub paused: bool,
  /// **全分辨率**加载半径（chunk，切比雪夫）
  pub load_radius: i32,
  /// 全分辨率卸载半径（chunk）：必须 > 加载半径（迟滞，免边界反复装卸）
  pub unload_radius: i32,
  /// **粗档**（16³ 量化）加载半径（chunk，xz 上的切比雪夫）
  pub coarse_radius: i32,
  /// 粗档的**竖向**半高（chunk）：`xz` 上是圆柱、`y` 上是薄板 —— 三维立方环里绝大多数 chunk
  /// 在头顶/脚下几十米处（谁也不看），同样的内存不如换成 xz 上的半径。
  pub coarse_height: i32,
  /// 每帧挂载的**序列化字数**预算（挂载要主线程做：装树 + 标脏 + 后面排队序列化上传）。
  /// 序列化约 0.5 ms/MB ⇒ 256 K 字（1 MB）/帧 ≈ 0.5 ms/帧，与 `UPLOAD_BYTES_PER_FRAME` 同量级；
  /// 折算成 chunk = 每帧约 3.7 个全分辨率 或 87 个粗档。
  pub mount_words: usize,
  /// 每帧挂载的条数上限（防"一堆极小树"把单帧的记账开销顶爆）
  pub mount_count: usize,
  /// M5 生产管线：首次进入流式世界时惰性起（要 `n_pbr` 才能定槽号方案）
  pipeline: Option<std::sync::Mutex<Pipeline>>,
}

impl Default for Streaming {
  fn default() -> Self {
    // 加载半径 2 chunk ≈ 10m（125 个 chunk ≈ 35 MB）；卸载半径只比它大 1：迟滞够用，且把
    // **全分辨率常驻集真正限住**（相机飞过去的尾迹要到卸载半径外才回收；差 2 会留下 9³ 尾迹）。
    // 粗档半径 8 chunk（≈41m，圆柱 ±6 层 ≈ 3757 个 chunk ≈ 44 MB）：16³ 量化便宜 24 倍 ⇒
    // 这一圈是大世界视距的杠杆。两者合计常驻约 80 MB。
    Self {
      paused: false,
      load_radius: 2,
      unload_radius: 3,
      coarse_radius: 8,
      coarse_height: 6,
      mount_words: 256 * 1024,
      mount_count: 48,
      pipeline: None,
    }
  }
}

/// M5 的生产源（程序化）：在 worker 线程上跑，**只引用确定性槽号**（[`slot_of`]），
/// 不碰主线程的 `VolumeGrid`（`scratch` 是 worker 独占的）。
struct InfiniteCubes {
  n_pbr: usize,
}

impl ChunkSource for InfiniteCubes {
  fn produce(&self, coord: ChunkCoord, detail: Detail, scratch: &mut VolumeGrid) -> Option<ChunkTree> {
    let lo = coord.0 * gate_voxel::CHUNK_SIZE;
    build_region(scratch, lo, lo + IVec3::splat(gate_voxel::CHUNK_SIZE), self.n_pbr, detail);
    scratch.take_chunk(coord) // 搬所有权（不 clone 整棵树）；取走即清空暂存
  }
}

/// **流式加载 / 卸载**（`infinite_cubes` 世界）：按相机位置产出 chunk、超出半径就**真卸载**
/// （CPU 树一起丢 ⇒ 卸载区里的修改永久丢失，与"不落盘"的语义一致）。
///
/// 需求两路（M4 + M5）：**主射线发回的 ray-guided 请求**（见 `docs/editable-gigavoxel.md` §9 M4）
/// 与**相机距离半径**（本模块的 `Streaming`）。产出走 M5 的 worker 池（[`ChunkProducer`]），
/// 主线程只做"派发 + 挂载"；挂载 / 常驻 / 换出全走既有流水线（`mount_chunk_tree` 自会标脏 ⇒
/// 走既有上传路径，显存由 `plan_residency` 的反向同步跟着归还）。
///
/// 两档半径（M5 粗粒度层）：`load_radius` 内是**全分辨率**、`coarse_radius` 内是**16³ 粗档**
/// （树小得多 ⇒ 视距的杠杆）；粗档 chunk 进到 `load_radius` 内会被重新全分辨率产出顶掉
/// （`mount_chunk_tree` 是整体替换）。
pub fn stream_chunks(
  mut stream: ResMut<Streaming>,
  mut scene: ResMut<gate_render::VoxelScene>,
  cam: Option<Res<gate_render::DdaCameraConfig>>,
  pbr: Option<Res<gate_render::PbrTextureSet>>,
  feed: Option<Res<gate_render::LodRequestFeed>>,
) {
  // WHY: 暂停 = 冻结整个流式环（连窗口都不跟）—— 让相机能飞出加载边界，看"世界到此为止"的那一圈。
  if stream.paused {
    return;
  }
  let Some(cam) = cam else { return };
  let Some((mut w_origin, w_dims)) = scene.volumes.main().stream_window() else {
    return; // 非流式世界
  };
  let chunk = gate_voxel::CHUNK_SIZE;
  let center = (cam.position_world / chunk as f32).floor().as_ivec3();
  let forward = cam.forward;

  // ⓪ 窗口跟着相机（**M6**）。窗口是 `b_struct` 索引区的定义域：跑出窗口的 chunk 生成得出、
  //    却装不上 GPU（索引区之外）⇒ 画面成片空洞。旧版是"接近边界就整块重定"（丢全部 CPU chunk +
  //    全量重传，一次性卡顿）；现在每帧把窗口钉在**相机为中心的 `w_dims`** 上，移动交给 builder 的
  //    **平移索引区**（只搬 1 MB 条目、不重传树块、不丢 CPU 内容）⇒ 世界随你走，不卡。
  //    CONSTRAINT：常驻环必须远小于半窗宽，否则相机走到窗边会卸载掉"还在窗内但已过界"的 chunk。
  let want_origin = center - w_dims / 2;
  if want_origin != w_origin {
    scene.volumes.main_mut().set_stream_window(Some((want_origin, w_dims)));
    w_origin = want_origin;
  }
  let grid = scene.volumes.main_mut();
  let (load_r, unload_r, coarse_r, coarse_h, mount_words, mount_count) = (
    stream.load_radius,
    stream.unload_radius,
    stream.coarse_radius,
    stream.coarse_height,
    stream.mount_words,
    stream.mount_count,
  );

  // ① 生产管线（M5）：**派发**需求给 worker 池（请求优先，再半径补块），本帧只**挂载**已完成的。
  //    管线惰性起：要 `n_pbr` 才能定槽号方案 —— 与建世界时装进调色板的那份同源（`pbr_asset_ids`）。
  if stream.pipeline.is_none() {
    let n_pbr = crate::scene::pbr_asset_ids(pbr.as_deref()).len();
    let source = std::sync::Arc::new(InfiniteCubes { n_pbr });
    stream.pipeline = Some(std::sync::Mutex::new(Pipeline {
      producer: ChunkProducer::new(source, PRODUCER_WORKERS),
      demand: Default::default(),
      demand_center: None,
      demand_age: 0,
      ready: Default::default(),
      detail: Default::default(),
      empty: Default::default(),
    }));
  }
  let requests: Vec<(u32, IVec3)> = feed
    .as_ref()
    .map(|f| {
      f.0.lock().unwrap_or_else(|e| e.into_inner()).iter().map(|q| (q.votes, q.chunk)).collect()
    })
    .unwrap_or_default();
  let mut generated = 0usize;
  {
    let mut pipe = stream
      .pipeline
      .as_ref()
      .expect("上面刚装上")
      .lock()
      .unwrap_or_else(|e| e.into_inner());
    // 需求表：相机换 chunk、或每 `DEMAND_REBUILD_FRAMES` 帧才重建（扫环 + 排序是数千项，
    // 每帧做白花约 1 ms）；其余帧只从表头派发。
    let mut gen_req = 0usize;
    pipe.demand_age += 1;
    if pipe.demand_center != Some(center) || pipe.demand_age >= DEMAND_REBUILD_FRAMES {
      let have = |c: IVec3, want: Detail| {
        let held = pipe.detail.get(&ChunkCoord(c)).copied().unwrap_or(Detail::Full);
        (grid.chunk(ChunkCoord(c)).is_some() && held >= want) || pipe.empty.contains(&ChunkCoord(c))
      };
      let (batch, from_req) = plan_generation(
        GenScope {
          center,
          w_origin,
          w_dims,
          load_radius: load_r,
          coarse_radius: coarse_r,
          coarse_height: coarse_h,
          forward,
        },
        DEMAND_TABLE_MAX,
        &requests,
        have,
      );
      pipe.demand = batch.into();
      pipe.demand_center = Some(center);
      pipe.demand_age = 0;
      gen_req = from_req;
    }
    // 派发：从表头取（已满足 / 已在飞的当场划过），到在飞上限为止
    let mut dispatched = 0usize;
    while let Some((c, detail)) = pipe.demand.front().copied() {
      if pipe.producer.in_flight(ChunkCoord(c)) {
        pipe.demand.pop_front();
        continue;
      }
      if !pipe.producer.request(ChunkCoord(c), detail) {
        break; // 在飞满（或已在飞）⇒ 下帧继续
      }
      pipe.demand.pop_front();
      dispatched += 1;
      if dispatched >= DISPATCH_PER_FRAME {
        break;
      }
    }
    // 取回产出 → 待挂载队列（`None` = 这个 chunk 没有内容，记下免得每帧反复派发）
    for (cc, detail, tree) in pipe.producer.poll(POLL_MAX) {
      match tree {
        Some(tree) => pipe.ready.push_back((cc, detail, tree)),
        None => {
          pipe.empty.insert(cc);
        }
      }
    }
    // 挂载：按**字数预算**逐帧消化（装树 + 标脏 + 后面的序列化上传都吃这条预算）；粗档细化也走
    // 这条路（`mount_chunk_tree` 是**整体替换** ⇒ 粗树被全分辨率树顶掉）。
    let mut words = 0usize;
    let mut words_mounted = 0usize;
    while !pipe.ready.is_empty() && words_mounted < mount_count {
      let w = pipe.ready.front().expect("刚看过 len").2.len_words();
      // 预算用尽就停，但**至少挂一个**：否则一个超大 chunk 会永远排不上
      if words_mounted > 0 && words + w > mount_words {
        break;
      }
      let (cc, detail, tree) = pipe.ready.pop_front().expect("刚看过 front");
      grid.mount_chunk_tree(cc, tree, 0);
      pipe.detail.insert(cc, detail);
      words += w;
      words_mounted += 1;
      generated += 1;
    }

    // ② 卸载：窗口外一律卸；全分辨率档超出 `unload_radius` 卸；粗档超出 `coarse_radius` 卸。
    //    （粗档树小 ⇒ 它的半径能比全分辨率大得多，这正是视距的杠杆。）
    let far: Vec<ChunkCoord> = grid
      .chunk_coords()
      .filter(|c| {
        let t = c.0 - w_origin;
        if t.cmplt(IVec3::ZERO).any() || t.cmpge(w_dims).any() {
          return true;
        }
        let d = c.0 - center;
        match pipe.detail.get(c).copied().unwrap_or(Detail::Full) {
          Detail::Full => d.abs().max_element() > unload_r,
          // 粗档环与需求侧**同一形状**（xz 圆柱 + y 薄板），否则会在边上反复装卸
          Detail::Coarse => d.x.abs().max(d.z.abs()) > coarse_r || d.y.abs() > coarse_h,
        }
      })
      .collect();
    let unloaded = far.len();
    for cc in far {
      grid.unmount_chunk(cc);
      pipe.detail.remove(&cc);
    }
    if generated > 0 || unloaded > 0 || !pipe.ready.is_empty() {
      bevy::log::debug!(
        "STREAM[gen {generated}(req {gen_req}) unload {unloaded} chunks {} ready {}]",
        grid.chunk_count(),
        pipe.ready.len()
      );
    }
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

/// 色相档数（非 PBR 各类共用）
const HUE_COUNT: usize = 8;
/// 每类的槽位数：8 色相 × 2 档（PBR 例外 —— 每资产一个槽，见 [`slot_of`]）
const SLOTS_PER_KIND: u16 = (HUE_COUNT * 2) as u16;
/// PBR 资产的槽号起点
const PBR_BASE: u16 = 2;
/// 8 档量化色相（非 PBR 各类共用）
const HUE_TABLE: [[u8; 3]; HUE_COUNT] = [
  [0xE8, 0x4C, 0x3C],
  [0xF2, 0x9E, 0x2C],
  [0xF2, 0xE2, 0x3C],
  [0x5C, 0xD6, 0x4C],
  [0x3C, 0xC8, 0xC8],
  [0x4C, 0x7C, 0xF2],
  [0x9C, 0x5C, 0xE8],
  [0xE8, 0x5C, 0xB4],
];

/// 二选一档位：`h >> shift` 的最低位决定取 `a` 还是 `b`（值域由调用方逐类列出）。
fn pick(h: u64, shift: u32, a: u8, b: u8) -> u8 {
  if (h >> shift) & 1 == 0 { a } else { b }
}

/// 非 PBR 各 `Kind` 的槽号块起点（`2 + n_pbr` 之后按 `Kind` 声明序排）
fn kind_base(kind: Kind, n_pbr: usize) -> u16 {
  let block = match kind {
    Kind::Pbr => 0,
    Kind::Light => 1,
    Kind::Glass => 2,
    Kind::Mirror => 3,
    Kind::Metal => 4,
    Kind::Plain => 5,
  };
  PBR_BASE + n_pbr as u16 + block * SLOTS_PER_KIND
}

/// **(类, 色相, 档位, 资产) → 槽号**：`2 + n_pbr + 类块 × 16 + 色相 × 2 + 档位`；PBR 走 `2 + asset`。
///
/// 这就是槽号方案本身 —— [`material_of`]（按 room 取槽）与 [`material_slots`]（枚举整表）都调它
/// ⇒ 两边不可能分叉。`Plain` 的两档内容相同 ⇒ 槽号里档位恒按 0 算（省一半槽）。
///
/// **为什么必须确定性**（M5）：后台并行生产时"在调色板里找同内容 / 第一个空槽"依赖生成顺序，
/// 线程调度一变槽号就变（还会与主线程争调色板）；现在槽号是纯函数，worker 只引用它。
pub fn slot_of(kind: Kind, hue: usize, notch: u8, asset: u16, n_pbr: usize) -> PaletteId {
  if matches!(kind, Kind::Pbr) && n_pbr > 0 {
    return PaletteId(PBR_BASE + asset.min(n_pbr as u16 - 1));
  }
  let notch = if matches!(kind, Kind::Plain | Kind::Pbr) { 0 } else { notch & 1 };
  PaletteId(kind_base(kind, n_pbr) + slot_in_kind(hue, notch))
}

/// 类块内的偏移：`色相 × 2 + 档位`
#[inline]
fn slot_in_kind(hue: usize, notch: u8) -> u16 {
  (hue.min(HUE_COUNT - 1) as u16) * 2 + notch as u16
}

/// 由**显式参数**造条目（`material_of` 与 `material_slots` 共用 ⇒ 不可能分叉）。
/// 各档取值见 `docs/infinite_cubes.md` 规则 5（光源 127/255、玻璃 127/191、镜面与金属粗糙度 0/127）。
fn entry_of(kind: Kind, color: [u8; 3], notch: u8, asset: u16) -> PaletteEntry {
  let hi = notch != 0; // 0/1 两档；调用方保证取值，`hi` 之外都算 0 档
  match kind {
    // 唯一不生成颜色的一类：颜色由 PBR 贴图决定。
    Kind::Pbr => PaletteEntry::pbr(asset, PbrOverrides::default(), PaletteFlags::default()),
    Kind::Light => {
      PaletteEntry { color, emissive: if hi { 255 } else { 127 }, ..Default::default() }
    }
    Kind::Glass => {
      PaletteEntry { color, transmission: if hi { 191 } else { 127 }, ..Default::default() }
    }
    Kind::Mirror => {
      PaletteEntry { color, roughness: if hi { 127 } else { 0 }, ..Default::default() }
    }
    Kind::Metal => PaletteEntry {
      color,
      metallic: 255,
      roughness: if hi { 127 } else { 0 },
      ..Default::default()
    },
    Kind::Plain => PaletteEntry { color, ..Default::default() },
  }
}

/// 固定方案产出的**全部** `(槽号, 条目)` —— 建世界时装一次；后台线程只引用槽号（不再写调色板）。
pub fn material_slots(n_pbr: usize) -> Vec<(PaletteId, PaletteEntry)> {
  let mut out = vec![(WHITE_SLOT, PaletteEntry { color: [255, 255, 255], ..Default::default() })];
  for asset in 0..n_pbr.min(u16::MAX as usize) as u16 {
    out.push((slot_of(Kind::Pbr, 0, 0, asset, n_pbr), entry_of(Kind::Pbr, [0; 3], 0, asset)));
  }
  for kind in [Kind::Light, Kind::Glass, Kind::Mirror, Kind::Metal, Kind::Plain] {
    // `Plain` 两档内容相同（槽号也归一到 0 档）⇒ 只枚举 0 档
    let notches: &[u8] = if matches!(kind, Kind::Plain) { &[0] } else { &[0, 1] };
    for (hue, &color) in HUE_TABLE.iter().enumerate() {
      for &notch in notches {
        out.push((slot_of(kind, hue, notch, 0, n_pbr), entry_of(kind, color, notch, 0)));
      }
    }
  }
  out
}

/// 该 room 的 cube 材质 + **确定性槽号**（`docs/infinite_cubes.md` 规则 5）：等概率六选一，
/// **除 PBR 外各类都生成颜色**，再各自生成描述的那一项值，其余保持 `PaletteEntry::default()`。
///
/// `n_pbr` = 可用 PBR 资产数；为 0 ⇒ PBR 那一档退化成"普通"（贴图集还没扫出来时不该凭空造 asset 槽号）。
pub fn material_of(room: IVec3, n_pbr: usize) -> (PaletteId, PaletteEntry) {
  let h = room_hash(room);
  let kind = match Kind::of(h) {
    Kind::Pbr if n_pbr == 0 => Kind::Plain,
    k => k,
  };
  let hue = ((h >> 16) % HUE_COUNT as u64) as usize;
  let notch = pick(h, 8, 0, 1);
  let asset = ((h >> 4) % n_pbr.max(1) as u64).min(u16::MAX as u64) as u16;
  (slot_of(kind, hue, notch, asset, n_pbr), entry_of(kind, HUE_TABLE[hue], notch, asset))
}

/// 体素点查询（世界坐标 → 槽号）：柱体棱 → 白；room 中心 cube → 该 room 的材质；其余 → 空气。
///
/// **只给粗粒度层用**（16³ 量化：每格采样几个点定代表材质）；全分辨率走 `build_region` 的盒填充。
/// 两边的判据同源：柱体截面 = 两轴都落在 `[k·ROOM − COLUMN/2, k·ROOM + COLUMN/2)`，cube 用
/// [`cube_range`]。
fn voxel_at(p: IVec3, n_pbr: usize) -> PaletteId {
  let half = COLUMN / 2;
  let near = |v: i32| {
    let m = v.rem_euclid(ROOM);
    m < half || m >= ROOM - half
  };
  if (near(p.x) && near(p.y)) || (near(p.y) && near(p.z)) || (near(p.x) && near(p.z)) {
    return WHITE_SLOT;
  }
  let room = p.div_euclid(IVec3::splat(ROOM));
  let (cmin, cmax) = cube_range(room);
  if p.cmpge(cmin).all() && p.cmplt(cmax).all() {
    return material_of(room, n_pbr).0;
  }
  PaletteId::AIR
}

/// room 中心 cube 的轴对齐体素范围。偏移 `(ROOM − CUBE) / 2` 是 4 的倍数（上面的常量断言）⇒
/// 对**所有** room（含负坐标）都是精确居中、且落格到 brick。
fn cube_range(room: IVec3) -> (IVec3, IVec3) {
  let min = room * ROOM + IVec3::splat((ROOM - CUBE) / 2);
  (min, min + IVec3::splat(CUBE))
}

/// 按 `grain³` brick 写 `[min, min+ext) ∩ [lo, hi)`（空交集不写），返回写入的体素数。
///
/// **CONSTRAINT**：生成器只写自己那一盒 —— 越界那半属于邻块（邻块会写它）。越界写会把邻块
/// 变成"半个 chunk"，而 [`stream_chunks`] 只按"chunk 在不在"判断要不要生成 ⇒ 那些 chunk
/// 永远补不齐，画面上就是缺口。两个盒的端点都是 `grain` 的倍数（见 [`build_region`] 的前置条件）
/// ⇒ 交出来的盒仍然 brick 对齐。
fn fill_clipped(
  grid: &mut VolumeGrid,
  min: IVec3,
  ext: IVec3,
  grain: i32,
  lo: IVec3,
  hi: IVec3,
  palette: PaletteId,
) -> u64 {
  let a = min.max(lo);
  let b = (min + ext).min(hi);
  if b.cmple(a).any() {
    return 0;
  }
  let voxels = (grain * grain * grain) as usize;
  (fill_bricks(grid, a, b - a, grain, palette) * voxels) as u64
}

/// 把 `[lo, hi)` 体素范围内的 `infinite_cubes` 结构填进 `grid`。
///
/// **前置条件**：lo/hi 各轴既对齐 4（brick 批量写）**又对齐 chunk**（否则生成出半块，见模块头
/// 的生成盒契约）。
///
/// - [`Detail::Full`]：三段 —— ① 三条轴向的棱柱 ② 每个 room 中心的 cube ③ 其余留空（4³ grain）；
/// - [`Detail::Coarse`]：**16³ 量化（32cm 块）** —— 每格用 [`voxel_at`] 采样 8 点定一个代表材质，
///   整格填成该色。远景用它：树小、生成快、同样内存能把视距推远（`docs/editable-gigavoxel.md` §10.3）。
///
/// `n_pbr` = 可用 PBR 资产数（[`material_of`] / [`slot_of`] 的槽号方案要吃它）。
/// 返回**写入的体素数**（柱体与 cube 重叠处会重复计一次）。
pub fn build_region(
  grid: &mut VolumeGrid,
  lo: IVec3,
  hi: IVec3,
  n_pbr: usize,
  detail: Detail,
) -> u64 {
  let chunk = gate_voxel::CHUNK_SIZE;
  debug_assert!(
    lo % chunk == IVec3::ZERO && hi % chunk == IVec3::ZERO,
    "区域须对齐 {chunk} 的 chunk（半块会被 stream_chunks 当已生成而跳过）：lo={lo} hi={hi}"
  );
  match detail {
    Detail::Full => build_region_full(grid, lo, hi, n_pbr),
    Detail::Coarse => build_region_coarse(grid, lo, hi, n_pbr),
  }
}

/// 全分辨率：棱柱 + cube（4³ grain）
fn build_region_full(grid: &mut VolumeGrid, lo: IVec3, hi: IVec3, n_pbr: usize) -> u64 {
  let (span, half) = (hi - lo, COLUMN / 2);
  let mut covered = 0u64;
  let first = |v: i32| (v - 1).div_euclid(ROOM);
  let last = |v: i32| (v + 1).div_euclid(ROOM);
  // ① 棱柱。三条轴各一组：柱轴 = 长边，截面 = 另两轴的 24×24。
  let edges = |a0: i32, a1: i32| (first(a0)..=last(a1)).collect::<Vec<_>>();
  for &a in &edges(lo.x, hi.x) {
    for &b in &edges(lo.y, hi.y) {
      // 沿 Z
      let min = IVec3::new(a * ROOM - half, b * ROOM - half, lo.z);
      covered += fill_clipped(grid, min, IVec3::new(COLUMN, COLUMN, span.z), 4, lo, hi, WHITE_SLOT);
    }
    for &b in &edges(lo.z, hi.z) {
      // 沿 Y
      let min = IVec3::new(a * ROOM - half, lo.y, b * ROOM - half);
      covered += fill_clipped(grid, min, IVec3::new(COLUMN, span.y, COLUMN), 4, lo, hi, WHITE_SLOT);
    }
  }
  for &a in &edges(lo.y, hi.y) {
    for &b in &edges(lo.z, hi.z) {
      // 沿 X
      let min = IVec3::new(lo.x, a * ROOM - half, b * ROOM - half);
      covered += fill_clipped(grid, min, IVec3::new(span.x, COLUMN, COLUMN), 4, lo, hi, WHITE_SLOT);
    }
  }
  // ② cube（材质与槽号都由 room 坐标定）。
  for rx in (lo.x - ROOM).div_euclid(ROOM)..=(hi.x + ROOM).div_euclid(ROOM) {
    for ry in (lo.y - ROOM).div_euclid(ROOM)..=(hi.y + ROOM).div_euclid(ROOM) {
      for rz in (lo.z - ROOM).div_euclid(ROOM)..=(hi.z + ROOM).div_euclid(ROOM) {
        let room = IVec3::new(rx, ry, rz);
        let (cmin, cmax) = cube_range(room);
        if cmax.cmple(lo).any() || cmin.cmpge(hi).any() {
          continue;
        }
        let id = material_of(room, n_pbr).0;
        covered += fill_clipped(grid, cmin, IVec3::splat(CUBE), 4, lo, hi, id);
      }
    }
  }
  covered
}

/// 粗粒度层：16³ 一格，格内 8 点采样取众数（平手取槽号小者）⇒ 细柱 / 薄结构在粗档也留得下痕迹。
fn build_region_coarse(grid: &mut VolumeGrid, lo: IVec3, hi: IVec3, n_pbr: usize) -> u64 {
  const CELL: i32 = 16;
  /// 格内采样点相对格角的偏移（2×2×2，取格内 1/4 与 3/4 处）
  const S: [i32; 2] = [CELL / 4, CELL - CELL / 4];
  let mut covered = 0u64;
  let mut z = lo.z;
  while z < hi.z {
    let mut y = lo.y;
    while y < hi.y {
      let mut x = lo.x;
      while x < hi.x {
        let cell = IVec3::new(x, y, z);
        let mut tally: [(u16, u8); 8] = [(0, 0); 8];
        let mut used = 0usize;
        for &dz in &S {
          for &dy in &S {
            for &dx in &S {
              let id = voxel_at(cell + IVec3::new(dx, dy, dz), n_pbr);
              if id.is_air() {
                continue;
              }
              match tally[..used].iter_mut().find(|(v, _)| *v == id.get()) {
                Some((_, n)) => *n += 1,
                None => {
                  tally[used] = (id.get(), 1);
                  used += 1;
                }
              }
            }
          }
        }
        if used > 0 {
          // 众数；平手取槽号小者（确定性）
          let best =
            tally[..used].iter().max_by_key(|(id, n)| (*n, std::cmp::Reverse(*id))).unwrap().0;
          covered += fill_clipped(grid, cell, IVec3::splat(CELL), CELL, lo, hi, PaletteId(best));
        }
        x += CELL;
      }
      y += CELL;
    }
    z += CELL;
  }
  covered
}

/// [`plan_generation`] 的几何 / 策略输入（打包成一项，免去一长串位置参数）
#[derive(Clone, Copy)]
struct GenScope {
  /// 相机所在 chunk
  center: IVec3,
  /// 流式窗口（chunk 原点 + 各轴跨度）：窗口外产出也装不上 GPU
  w_origin: IVec3,
  w_dims: IVec3,
  /// 全分辨率半径（chunk，切比雪夫）
  load_radius: i32,
  /// 粗档（16³ 量化）半径：xz 上的切比雪夫
  coarse_radius: i32,
  /// 粗档的竖向半高（chunk）
  coarse_height: i32,
  /// 视线方向（单位向量）。`ZERO` = 不做视野偏置（按纯距离排）
  forward: Vec3,
}

/// "在视野内"的点积门槛：`±60°` 锥。只用它分两档 —— 连续值参与排序反而会被远处的巧合压过距离。
const VIEW_COS: f32 = 0.5;

/// 本帧该产出哪些 chunk（[`stream_chunks`] 第 ① 步的策略，抽成纯函数以便单测）：
/// **请求优先**（票数多的先，M4 切片 2），再用半径补块（**视野优先**，见下），合计 ≤ `want`。
///
/// **两级半径**（M5 粗粒度层）：`load_radius` 内要 [`Detail::Full`]，`coarse_radius` ×
/// `coarse_height` 的圆柱内用 [`Detail::Coarse`]（16³ 量化）—— 后者便宜二十几倍，所以能推得远。
///
/// **排序**（半径启发式最容易搞砸的一点）：`load_radius` 内永远最优先（脚下与眼前，缺了就掉进
/// 虚空），其余先给**视野锥内**的、再按距离。只按距离排的话，帧额会被平均撒到相机身后 ——
/// 那就完全没有"ray-guided"的感觉。
///
/// 过滤（都要过）：
/// - **窗口内**：窗口是 `b_struct` 索引区的定义域，窗口外产出也装不上 GPU；
/// - **够不够细**由 `have(c, want)` 回答（已挂载且档位 ≥ 要的档 ⇒ 不算需求；粗档在近处会因此被细化）；
/// - 请求那一侧还额外要求**在粗档环内**：更远的会被卸载规则立刻回收（白付一次产出 + 上传）。
///
/// 返回 `(本帧的 (chunk, 档位) 列表, 其中来自请求的条数)`。
fn plan_generation(
  scope: GenScope,
  want: usize,
  requests: &[(u32, IVec3)],
  have: impl Fn(IVec3, Detail) -> bool,
) -> (Vec<(IVec3, Detail)>, usize) {
  let GenScope { center, w_origin, w_dims, load_radius, coarse_radius, coarse_height, forward } =
    scope;
  let in_window = |c: IVec3| {
    let t = c - w_origin;
    t.cmpge(IVec3::ZERO).all() && t.cmplt(w_dims).all()
  };
  let in_ring = |c: IVec3| {
    let d = c - center;
    d.x.abs().max(d.z.abs()) <= coarse_radius && d.y.abs() <= coarse_height
  };
  // 该 chunk 该用哪一档：近处全分辨率，远处粗档
  let detail_of = |c: IVec3| {
    if (c - center).abs().max_element() <= load_radius { Detail::Full } else { Detail::Coarse }
  };
  // 排序键（越小越先）：近处圈 → 视野锥内 → 距离 → 坐标（确定性）
  let key = |c: IVec3| {
    let d = c - center;
    let dist = d.abs().max_element();
    let far = (dist > load_radius) as i32;
    let off_view = if far == 0 || forward == Vec3::ZERO {
      0
    } else {
      (d.as_vec3().normalize_or_zero().dot(forward) <= VIEW_COS) as i32
    };
    (far, off_view, dist, c.x, c.y, c.z)
  };
  let mut picked: Vec<(IVec3, Detail)> = Vec::with_capacity(want);
  let mut requested: Vec<(u32, IVec3)> =
    requests.iter().copied().filter(|(_, c)| in_window(*c) && in_ring(*c)).collect();
  requested.sort_unstable_by_key(|(votes, c)| (std::cmp::Reverse(*votes), c.x, c.y, c.z));
  picked.extend(
    requested
      .into_iter()
      .filter(|(_, c)| !have(*c, detail_of(*c)))
      .take(want)
      .map(|(_, c)| (c, detail_of(c))),
  );
  let from_req = picked.len();
  if picked.len() < want {
    let mut todo: Vec<(IVec3, Detail)> = Vec::new();
    for dx in -coarse_radius..=coarse_radius {
      for dz in -coarse_radius..=coarse_radius {
        for dy in -coarse_height..=coarse_height {
          let c = center + IVec3::new(dx, dy, dz);
          let d = detail_of(c);
          if in_window(c) && !have(c, d) && !picked.iter().any(|(p, _)| *p == c) {
            todo.push((c, d));
          }
        }
      }
    }
    todo.sort_unstable_by_key(|(c, _)| key(*c));
    picked.extend(todo.into_iter().take(want - picked.len()));
  }
  (picked, from_req)
}

#[cfg(test)]
mod tests {
  use super::*;
  use gate_voxel::VoxelCoord;
  use gate_voxel::PaletteEntry;

  fn build(lo: IVec3, hi: IVec3) -> VolumeGrid {
    let mut grid = VolumeGrid::new();
    grid.palette_mut().set(WHITE_SLOT, PaletteEntry { color: [0xFF, 0xFF, 0xFF], ..Default::default() });
    build_region(&mut grid, lo, hi, 0, Detail::Full);
    grid
  }

  /// M5：**生产管线产出 ≡ 同步产出**（逐位相同）。管线把生成搬到 worker 线程、把槽号改成确定性方案，
  /// 这两件事都不该改变任何一个体素 —— 所以拿 `serialize()` 的字节直接对。
  #[test]
  fn producer_output_matches_sync_build() {
    let cc = ChunkCoord(IVec3::new(-1, 0, 2));
    let lo = cc.0 * gate_voxel::CHUNK_SIZE;
    let mut sync_grid = VolumeGrid::new();
    build_region(&mut sync_grid, lo, lo + IVec3::splat(gate_voxel::CHUNK_SIZE), 0, Detail::Full);
    let sync_tree = sync_grid.chunk(cc).expect("同步产出应有内容").serialize();

    let src = std::sync::Arc::new(InfiniteCubes { n_pbr: 0 });
    let mut scratch = VolumeGrid::new();
    let async_tree = src.produce(cc, Detail::Full, &mut scratch).expect("管线产出应有内容").serialize();
    assert_eq!(async_tree, sync_tree, "后台产出与同步产出必须逐位相同");
    assert!(scratch.chunk(cc).is_none(), "暂存被取空（同 chunk 不会下次再用到旧内容）");
  }

  /// M5：槽号方案是**纯函数 + 单射** —— `material_slots` 装出来的表必须与"按 room 查到的槽"一一对应，
  /// 且不同材质不共槽（后台并行生产的前提：worker 只引用槽号，不再与主线程争调色板）。
  #[test]
  fn material_slots_match_room_lookup() {
    for n_pbr in [0usize, 3] {
      let table: std::collections::HashMap<PaletteId, PaletteEntry> =
        material_slots(n_pbr).into_iter().collect();
      assert_eq!(table.len(), material_slots(n_pbr).len(), "槽号不得重复");
      for i in 0..2048i64 {
        let room = IVec3::new(i as i32, (i * 7) as i32, (i * 13) as i32);
        let (slot, entry) = material_of(room, n_pbr);
        assert_eq!(table.get(&slot).copied(), Some(entry), "room {room} 的槽号不在方案表里");
      }
    }
    // 没有 PBR 资产时 PBR 那一档退化成"普通"：不该出现 Pbr 变体
    let no_pbr: Vec<PaletteEntry> = (0..512)
      .map(|i| material_of(IVec3::new(i, 0, 0), 0).1)
      .collect();
    assert!(
      no_pbr.iter().all(|e| !e.flags.contains(PaletteFlags::IS_PBR)),
      "n_pbr = 0 时不该造 PBR 变体"
    );
  }

  /// M5 粗粒度层：16³ 量化（a）与全分辨率点查询同源 ⇒ 采样点落在实体上就一定是那个材质；
  /// （b）树显著更小（这正是"远景用粗档"的理由）。
  #[test]
  fn coarse_detail_is_quantized_and_small() {
    let (lo, hi) = region();
    // (a) 棱上的 16³ 格在粗档里也是白的（8 点采样一定采到柱体）
    let mut coarse = VolumeGrid::new();
    build_region(&mut coarse, lo, hi, 0, Detail::Coarse);
    let g = &coarse;
    assert_eq!(g.get_voxel(VoxelCoord::new(0, 0, 8)), Some(WHITE_SLOT), "棱上的格该是柱体的白");
    // 点查询与盒填充同源（柱体判据 / cube 判据共用）
    assert_eq!(voxel_at(IVec3::new(0, 0, 37), 0), WHITE_SLOT);
    // (b) 同范围：粗档序列化字数远小于全分辨率
    let mut full = VolumeGrid::new();
    build_region(&mut full, lo, hi, 0, Detail::Full);
    let words = |g: &VolumeGrid| -> usize {
      g.chunk_coords().map(|c| g.chunk(c).map_or(0, |t| t.serialize().len())).sum()
    };
    let (wc, wf) = (words(&coarse), words(&full));
    assert!(wc * 4 < wf, "粗档字数应远小于全分辨率（粗 {wc} vs 全 {wf}）");
  }

  /// 测试用的整 chunk 盒（chunk -1..=1，覆盖原点两侧的 room）。
  fn region() -> (IVec3, IVec3) {
    let chunk = gate_voxel::CHUNK_SIZE;
    (-IVec3::splat(chunk), IVec3::splat(chunk * 2))
  }

  fn solid(g: &VolumeGrid, p: IVec3) -> bool {
    g.get_voxel(VoxelCoord::new(p.x, p.y, p.z)).is_some()
  }

  /// 测试用 scope：以原点为中心的窗口（半宽 = `window`），不做视野偏置（`forward = ZERO`）
  fn gen_scope(window: i32, load_radius: i32, coarse_radius: i32, coarse_height: i32) -> GenScope {
    GenScope {
      center: IVec3::ZERO,
      w_origin: IVec3::splat(-window),
      w_dims: IVec3::splat(window * 2),
      load_radius,
      coarse_radius,
      coarse_height,
      forward: Vec3::ZERO,
    }
  }

  /// M4 切片 2 + M5：**请求优先于半径补块**（票数多的先），而窗口外 / 粗档半径外 / 已够细的请求一律丢；
  /// 两级半径：`load_radius` 内要**全分辨率**、`coarse_radius` 内给**粗档**；没有请求时逐字退回
  /// "半径 + 近的先"（= 开关关掉、或没有需求时的行为）。
  #[test]
  fn requests_outrank_radius_fill() {
    let scope = gen_scope(8, 1, 3, 3);
    let mounted: std::collections::HashSet<IVec3> = [IVec3::new(1, 0, 0)].into_iter().collect();
    // "已挂载且够细"：那一块已挂载（全分辨率，满足任何档位要求）
    let have = |c: IVec3, _want: Detail| mounted.contains(&c);
    // 票数最高的三条分别撞在"粗档半径外 / 窗口外 / 已够细"上 —— 三种都必须丢
    let reqs = [
      (99u32, IVec3::new(5, 0, 0)),
      (98, IVec3::new(9, 0, 0)),
      (97, IVec3::new(1, 0, 0)),
      (9, IVec3::new(2, 0, 0)),
      (5, IVec3::new(3, 0, 0)),
    ];
    // 帧额 2 全给请求：票数多的在前；两条都在 load_radius(1) 外 ⇒ 都是粗档
    let (batch, from_req) = plan_generation(scope, 2, &reqs, have);
    assert_eq!(
      batch,
      vec![(IVec3::new(2, 0, 0), Detail::Coarse), (IVec3::new(3, 0, 0), Detail::Coarse)]
    );
    assert_eq!(from_req, 2);
    // 帧额 4：请求先占 2，剩下按半径近的先 —— 先是中心（全分辨率），再是 (-1,-1,-1)
    let (batch, from_req) = plan_generation(scope, 4, &reqs, have);
    assert_eq!(from_req, 2);
    assert_eq!(
      batch,
      vec![
        (IVec3::new(2, 0, 0), Detail::Coarse),
        (IVec3::new(3, 0, 0), Detail::Coarse),
        (IVec3::ZERO, Detail::Full),
        (IVec3::new(-1, -1, -1), Detail::Full),
      ]
    );
    // 无请求（开关关掉）⇒ 逐字退回半径启发式
    let (batch, from_req) = plan_generation(scope, 3, &[], have);
    assert_eq!(from_req, 0);
    assert_eq!(
      batch,
      vec![
        (IVec3::ZERO, Detail::Full),
        (IVec3::new(-1, -1, -1), Detail::Full),
        (IVec3::new(-1, -1, 0), Detail::Full),
      ]
    );
  }

  /// M5：需求排序**视野优先** —— 同样距离时，相机看着的 chunk 必须先于身后的。
  /// （半径启发式若只按距离排，帧额会被平均撒在身后，那就完全没有"ray-guided"的感觉。）
  #[test]
  fn view_cone_outranks_distance() {
    let mut scope = gen_scope(20, 0, 3, 3);
    scope.forward = Vec3::new(0.0, 0.0, 1.0); // 朝 +Z 看
    let ahead = IVec3::new(0, 0, 3);
    let behind = IVec3::new(0, 0, -3);
    // 环内共 7×7×7 = 343 个候选 ⇒ 要够 400 才能保证"身后那个也被选中"（它排得很后，这正是论点）
    let (batch, _) = plan_generation(scope, 400, &[], |_, _| false);
    let at = |c: IVec3| batch.iter().position(|(p, _)| *p == c).expect("环内必然入选");
    assert_eq!(batch[0].0, IVec3::ZERO, "近处圈永远第一");
    assert!(at(ahead) < at(behind), "同距离：视野内 {} < 身后 {}", at(ahead), at(behind));
  }

  /// M5 粗粒度层：**档位只由距离定** —— `load_radius` 内全分辨率、外到 `coarse_radius` 用粗档，
  /// 半径外一个都不要（这正是"视距的杠杆"：粗档那一圈便宜得多）。
  #[test]
  fn far_ring_loads_coarse() {
    let (batch, _) = plan_generation(gen_scope(20, 1, 3, 3), 200, &[], |_, _| false);
    assert!(
      batch
        .iter()
        .all(|(c, d)| if c.abs().max_element() <= 1 { *d == Detail::Full } else { *d == Detail::Coarse }),
      "档位只由距离定"
    );
    assert!(
      batch.iter().any(|(c, d)| c.abs().max_element() == 3 && *d == Detail::Coarse),
      "粗档那一圈得真的在表里"
    );
    assert!(batch.iter().all(|(c, _)| c.abs().max_element() <= 3), "半径外不该要");
  }

  /// 规则 1/2：棱上有柱体，离开棱（且不在 cube 内）留空。
  #[test]
  fn columns_sit_on_room_edges() {
    let (lo, hi) = region();
    let g = build(lo, hi);
    for p in [IVec3::new(0, 0, 37), IVec3::new(0, 0, -37), IVec3::new(ROOM, 0, 5), IVec3::new(0, ROOM, 5), IVec3::new(5, ROOM, ROOM)] {
      assert!(solid(&g, p), "棱上 {p:?} 应是柱体");
    }
    // room 内部、离棱与 cube 都远 ⇒ 空
    let empty = IVec3::new(40, 40, 40);
    assert!(!solid(&g, empty), "{empty:?} 应留空");
    assert_eq!(g.palette().get(WHITE_SLOT).color, [0xFF, 0xFF, 0xFF], "柱体是纯白");
  }

  /// 生成盒契约：`build_region` **只写自己那一盒**。越界（本模块最容易踩的那一处）会把邻块
  /// 变成"半个 chunk"，而 `stream_chunks` 只按"chunk 在不在"决定要不要生成 ⇒ 那些 chunk
  /// 永远补不齐，画面上就是缺口与齐平断口。
  #[test]
  fn region_never_touches_chunks_outside_the_box() {
    let (lo, hi) = (IVec3::new(-256, 0, 256), IVec3::new(256, 256, 512));
    let g = build(lo, hi);
    let chunk = gate_voxel::CHUNK_SIZE;
    let (c0, c1) = (lo / chunk, hi / chunk);
    for c in g.chunk_coords() {
      assert!(
        c.0.cmpge(c0).all() && c.0.cmplt(c1).all(),
        "chunk {:?} 落在生成盒 [{c0:?}, {c1:?}) 之外（越界写）",
        c.0
      );
    }
    let want = (c1 - c0).x * (c1 - c0).y * (c1 - c0).z;
    assert_eq!(g.chunk_count() as i32, want, "盒内每个 chunk 都该被写到");
  }

  /// 规则 3/4：room 中心有 cube，材质由 room 坐标决定且**可重复**。
  #[test]
  fn cube_is_at_room_center_and_deterministic() {
    let (lo, hi) = region();
    let g = build(lo, hi);
    let (cmin, cmax) = cube_range(IVec3::ZERO);
    let c = (cmin + cmax) / 2;
    assert!(solid(&g, c), "room 中心 {c:?} 应是 cube");
    assert!(!solid(&g, cmin - IVec3::X), "cube 之外应留空");
    assert_eq!(cmax - cmin, IVec3::splat(CUBE));
    assert_eq!(material_of(IVec3::new(3, -7, 11), 0), material_of(IVec3::new(3, -7, 11), 0), "同 room 同材质");
    let kinds: std::collections::HashSet<_> = (0..96).map(|i| Kind::of(room_hash(IVec3::new(i, 0, 0)))).collect();
    assert!(kinds.len() >= 5, "六类应都出现过（实际 {} 类）", kinds.len());
  }

  /// 规则 3 的"**中心**"：cube 精确居中（不是"最多偏几个体素"），且偏移落格 4
  /// （`fill_bricks` 的对齐断言要求）——负坐标 room 同样。
  #[test]
  fn cube_is_exactly_centered_in_every_room() {
    let off = (ROOM - CUBE) / 2;
    assert_eq!(off % 4, 0, "偏移必须落格 4；不满足就改 ROOM（约束见文件头的常量断言）");
    for (rx, ry, rz) in [(0, 0, 0), (-1, 3, -7), (5, -2, 4), (-13, 8, -1)] {
      let room = IVec3::new(rx, ry, rz);
      let (cmin, cmax) = cube_range(room);
      assert_eq!(cmin, room * ROOM + IVec3::splat(off));
      assert_eq!(
        (cmin + cmax) / 2,
        room * ROOM + IVec3::splat(ROOM / 2),
        "cube 中心 = room 中心"
      );
      assert_eq!(cmin % 4, IVec3::ZERO, "cube 最小角须落格 4：{cmin:?}");
    }
  }

  /// 起始铺块必须**整 chunk 对齐**：半块会被 `stream_chunks` 当成"已生成"跳过 ⇒ 相机周围
  /// 永远缺一块（见模块头的生成盒契约）。
  #[test]
  fn initial_box_is_chunk_aligned() {
    let chunk = IVec3::splat(gate_voxel::CHUNK_SIZE);
    for center in [IVec3::new(512, 16, 512), IVec3::new(60, 60, 60), IVec3::new(-300, 0, 5)] {
      let (lo, hi) = initial_box(center);
      assert_eq!(lo % chunk, IVec3::ZERO, "lo 未对齐 chunk：{lo}");
      assert_eq!(hi % chunk, IVec3::ZERO, "hi 未对齐 chunk：{hi}");
      assert_eq!(hi - lo, chunk * (START_CHUNKS * 2 + 1), "边长 = 半径×2+1 个 chunk");
      let t = center - lo;
      assert!(t.cmpge(IVec3::ZERO).all() && t.cmplt(hi - lo).all(), "相机点须落在起始块内");
    }
  }

  /// 规则 5 的**档位口径**：光源 / 玻璃 / 镜面 / 金属各只有两档，且两档都得出得现。
  #[test]
  fn kind_values_are_exactly_two_notches() {
    use std::collections::HashSet;
    let (mut light, mut glass, mut mirror, mut metal) =
      (HashSet::new(), HashSet::new(), HashSet::new(), HashSet::new());
    for i in 0..4096 {
      let room = IVec3::new(i, i * 7, i * 13);
      let (_, e) = material_of(room, 0);
      match Kind::of(room_hash(room)) {
        Kind::Light => {
          light.insert(e.emissive);
        }
        Kind::Glass => {
          glass.insert(e.transmission);
        }
        Kind::Mirror => {
          assert_eq!(e.metallic, 0, "镜面是电介质");
          mirror.insert(e.roughness);
        }
        Kind::Metal => {
          assert_eq!(e.metallic, 255, "金属度恒 1");
          metal.insert(e.roughness);
        }
        Kind::Pbr | Kind::Plain => {}
      }
    }
    assert_eq!(light, HashSet::from([127, 255]), "光源自发光两档");
    assert_eq!(glass, HashSet::from([127, 191]), "玻璃透明度两档");
    assert_eq!(mirror, HashSet::from([0, 127]), "镜面粗糙度两档（0 = 全镜面）");
    assert_eq!(metal, HashSet::from([0, 127]), "金属粗糙度两档");
  }

  /// 规则 5：**除 PBR 外都生成颜色**；此外只动自己那一项（普通类只动颜色）。
  #[test]
  fn material_kinds_only_touch_their_own_fields() {
    let d = PaletteEntry::default();
    for i in 0..512 {
      let (_, e) = material_of(IVec3::new(i, i * 7, i * 13), 0);
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
