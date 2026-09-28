//! `infinite_cubes` 世界（规则见 `docs/infinite_cubes.md`）：向六个方向无限生长的 3m room 网格 ——
//! 每条棱是 50cm 截面的纯白柱体、每个 room 中心一个 90cm cube，cube 材质由 room 坐标当种子随机。
//!
//! 本模块只负责**按坐标生成体素**（[`build_region`]）与**流式驱动**（[`stream_chunks`]）：
//! 生成走 M5 的生产管线（[`InfiniteCubes`] 实现 `gate_voxel::ChunkSource`，worker 线程并行产出），
//! 挂载 / 常驻 / 卸载全走真实流程（真 world 只是把"读系统文件"那一步换成了本模块的生成函数）。
//!
//! **尺度**：体素没有固定物理边长（`README` §4「坐标系约定」）—— 本世界按显示换算
//! `VOXEL_PER_METER = 50`（2 cm/体素）折算下面这些"规格米数"，体素数本身与世界无关。
//! 为对齐 4³ brick（批量填充的前提；不对齐的话生成成本会从
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

use bevy::prelude::{Local, Res, ResMut, Resource};
use gate_voxel::{
  ChunkCoord, ChunkProducer, ChunkSource, ChunkTree, Detail, PaletteEntry, PaletteFlags, PaletteId,
  PbrOverrides, VolumeGrid, Volumes, fill_bricks,
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

// ---------------------------------------------------------------------------
// M8 · 多级远场（`docs/editable-gigavoxel.md` §4 M8）
// ---------------------------------------------------------------------------

/// **远场梯级**：每级的「级体素」= 多少个主世界体素（= 该 volume 的 `transform.scale`）。
///
/// 梯级由两条约束一起定（§10.4 第 2 条）：
/// - **接缝 ≤ 1 px**：本级在内沿 `d_in` 处要 `g ≤ d_in/935`（1080p，`1/px_ang = 935` 体素）；
/// - **每级的「覆盖/粒度」比恒定**：覆盖 = 32 chunk × 256 级体素 = `8192·scale` 体素，
///   粒度 `g = FAR_GRAIN · scale` ⇒ `覆盖/g = 8192/FAR_GRAIN = 2048`，接缝处恒
///   `935·FAR_GRAIN/8192 = 0.46 px`（**与 scale 无关** ⇒ 接缝宽度不随级数变）。
///
/// ⇒ 覆盖半径（世界体素 / 米，米数按本世界显示换算 50 体素/米折算）：`4` → 32768 / **655 m**、
/// `16` → 131072 / **2.6 km**、`64` → 524288 / **10.5 km**。**5 km 视距由 L3 覆盖**（§10.4 的验收目标）。
///
/// CONSTRAINT: 级体素必须是 `FAR_GRAIN` 的倍数（格点才能落在世界坐标的整数格上）。
pub const FAR_SCALES: [i32; 3] = [4, 16, 64];

/// **默认的流式世界内存预算**（= [`Streaming::request_bytes`] 的默认值，2 GB）。
///
/// 单独提出来是因为**远场窗口的水平半宽**要在建卷时按它算（见 `attach_far_levels_with` 的
/// CONSTRAINT：窗口 dims 不能在运行时改），而菜单里的「请求内存」只改 `Streaming` 的字段。
pub const DEFAULT_REQUEST_BYTES: usize = 2 * 1024 * 1024 * 1024;
/// **远场每格的边长（级体素）**：每 chunk `(256/16)³ = 16³ = 4096` 格。
///
/// 取值是"**接缝宽度** vs **内存 / 产出**"的取舍，实测（`far_detail_is_much_smaller_than_full`
/// 与 `produce_cost` 同一批数据，`infinite_cubes`）：
///
/// | `FAR_GRAIN` | 格（世界体素，L1） | 树（wire） | 接缝像素（本级内沿） |
/// |---|---|---|---|
/// | 4 | 16 | 418 KB/chunk —— **比全分辨率（288 KB）还大** | 2 px |
/// | 16 | 64 | **4–6 KB/chunk**（≈ 50–60× 便宜） | 7 px |
///
/// `grain = 4` 不可用：格（16 世界体素）比白柱截面（24）还细 ⇒ 远场把柱体的**截面**也画出来，
/// 采样结果是"孤立的一格格"，父层无从合并（树比近场还大）；而 §10.4 的"覆盖/g = 8192/cell"推出来的
/// `cell = 4` 正是按"接缝 ≤ 0.5 px"给的 —— 那条判据买不到这个内存。
/// `grain = 16`（格 = 64 世界体素 ≈ 655 m 处的一像素）让**相邻格多半同色** ⇒ 父层大量合并，
/// 树掉到个位数 KB，一帧能产出上百个 chunk。代价 = 接缝处 7 px 的块（本级内沿；外沿恒 1.8 px）——
/// **这是本轮明确接受的偏差**（见 `docs/editable-gigavoxel.md` §10.4 的"处处 ≤ 1 px"是那条判据的
/// 理论值，不是可承受的内存）。
///
/// CONSTRAINT：必须是 4 的幂且在 `LEVEL_EXTENT`（`fill_brick` 的合法粒度）内。
pub const FAR_GRAIN: i32 = 16;

/// 第 `vol` 级（`vol ≥ 1`，与 `Volumes.list` 下标同一口径）的**裁剪半径**（**世界体素**）=
/// **该级内容的远边往回收 [`SEAM_MARGIN`] 个本级 chunk**：每 chunk 256 级体素 ⇒
/// `(r_out + 1 − SEAM_MARGIN)·256·scale`（窗口 dims 仍是 `2·r_out + 2`，见
/// [`attach_far_levels_with`]；`r_out` 来自 [`far_radius_ladder`]）。
///
/// WHY 不是内容远边 `(r_out + 1)`：内容锚在 chunk 角、裁剪按相机 ⇒ 内容远边沿某方向最多短
/// `√2` chunk ⇒ 拿它当裁剪面会在**外圈**留一圈没内容的环（见 [`SEAM_MARGIN`]）。收 2 chunk 是
/// "内容恒越过裁剪面"的必要条件。
pub fn far_coverage_voxels(vol: usize) -> i32 {
  let ladder = far_radius_ladder(gate_render::pool_capacity_chunks_far(DEFAULT_REQUEST_BYTES));
  let (_, r_out) = ladder[(vol - 1).min(ladder.len() - 1)];
  (r_out + 1 - SEAM_MARGIN).max(1) * gate_voxel::CHUNK_SIZE * FAR_SCALES[vol - 1]
}

/// **给主世界挂上三级远场 volume（M8）**：逐级 `Volumes::add_far_level` + 装调色板 + 钉窗口。
///
/// 三个口径：
/// - **窗口的水平半宽 = 该级的半径阶梯外圈**（竖向仍取 `CHUNK_INDEX_CAP` 的 64，要盖住世界全高）
///   ⇒ 覆盖 = [`far_coverage_voxels`]；逐帧由 [`stream_chunks`] 钉在相机中心（平移索引区，不重传树块）。
/// - **调色板逐份复制**：每个 volume 有自己的 `b_palette`（512 KB），而远场产出的槽号来自**同一套
///   方案**（[`material_slots`]）⇒ 必须把同一张表装进每一份（worker 从头到尾不写调色板）。
/// - **与主世界共享原点**（`add_far_level` 的 `pos = 0`）⇒ 格点锚在世界坐标上，窗口滑动不改相位。
///
/// `center` = 相机眼位（世界体素）：起始窗口就钉在它周围，省掉开局那几帧的跨级平移。
///
/// **MC 地图**（`docs/mc_map.md` §8.2）走 [`attach_far_levels_mc`]：那一边的槽号不是固定方案。
pub fn attach_far_levels(volumes: &mut Volumes, pbr_ids: &[String], center: IVec3) {
  let slots = material_slots(pbr_ids.len());
  attach_far_levels_with(volumes, center, |g| {
    for (id, entry) in slots.iter() {
      g.palette_mut().set(*id, *entry);
    }
  });
}

/// **MC 地图的远场级**：只挂 volume，**不装调色板** —— MC 的槽号是 worker 按方块状态认领的
/// （`mc::material::Pool` 的日志），由 `stream_chunks` ⓪''' 的 `palette_log` 重放填进**所有** volume
/// （含远场级）。这里若也按 `material_slots` 装一份，两套槽号会互相覆盖。
pub fn attach_far_levels_mc(volumes: &mut Volumes, center: IVec3) {
  attach_far_levels_with(volumes, center, |_| {});
}

/// [`attach_far_levels`] / [`attach_far_levels_mc`] 的共同实现：逐级 `add_far_level` + 钉窗口。
fn attach_far_levels_with(
  volumes: &mut Volumes,
  center: IVec3,
  fill_palette: impl Fn(&mut gate_voxel::VolumeGrid),
) {
  /// 窗口每轴 chunk 数（= `brickmap::wire::CHUNK_INDEX_CAP`，与 `scene::build_infinite_cubes` 同值）
  const WINDOW_CHUNKS: i32 = 32;
  // 远场窗口的**水平半宽 = 该级半径阶梯的外圈**（见 `far_radius_ladder`）：descriptor 的世界 AABB
  // 就是按窗口算的（`brickmap::builder` 的 `window_world_aabb`），而 shader 的分壳裁剪读到的正是那个
  // AABB ⇒ 把窗口缩到"这一级填得满的半径"，级间接力（下一级从本级 AABB 出口起）才成立：
  // L1 只被判到 225 m、L2 从那里接手，而不是让 L1 名义上管到 655 m 却只有 6% 的块。
  // CONSTRAINT: dims 定下后不能在运行时改（它定义 `b_struct` 索引区的大小，一改各卷 base 就漂移 ⇒
  // 降级全量快照）⇒ 这里按**默认** `request_bytes` 算阶梯。菜单把「请求内存」调小只会让实际容量变小，
  // 而 `plan_generation_far` 把预载夹在窗口内，不会溢出。
  let ladder = far_radius_ladder(gate_render::pool_capacity_chunks_far(DEFAULT_REQUEST_BYTES));
  let mut dims = IVec3::splat(WINDOW_CHUNKS * 2);
  for (k, &scale) in FAR_SCALES.iter().enumerate() {
    let (_, r_out) = ladder[k];
    // 竖向保持 64：要盖住世界全高；**水平半宽才是"这一级的视距"**
    dims.x = 2 * r_out + 2;
    dims.z = 2 * r_out + 2;
    let dims = dims;
    let vol = volumes.add_far_level(scale as f32);
    let g = &mut volumes.list[vol];
    fill_palette(g);
    // 相机在该级的 chunk 空间：`floor(cam_world / scale / 256)` ⇒ 窗口以它为中心
    let c = (center / scale).div_euclid(IVec3::splat(gate_voxel::CHUNK_SIZE));
    g.set_stream_window(Some((c - dims / 2, dims)));
    // **覆盖半径 = 分壳裁剪的接力半径**（下一级只从它起参与求交）：本级内容真正铺到的半径 =
    // `(r_out+1)·256·scale` 世界体素（最外那一圈 chunk 的外缘）—— 与窗口 AABB 分开，见
    // `VolumeGrid::set_coverage_r` 的 WHY。
    g.set_coverage_r(far_coverage_voxels(vol) as f32);
    // 启动里程碑（一次性，3 行）：覆盖半径是验收"视距 ≥ 5 km"的直接依据（体素数；米数按本世界换算）
    bevy::log::info!(
      "FAR L{vol} scale {scale}（1 级体素 = {scale} 世界体素）覆盖 ±{:.2}km；窗口 {}×{}×{} chunk × 256 级体素、\
       格 {FAR_GRAIN} 级体素 = {:.2}m；半径阶梯 r{r_in}..{r_out}",
      far_coverage_voxels(vol) as f32 * 0.02 / 1000.0,
      dims.x,
      dims.y,
      dims.z,
      FAR_GRAIN as f32 * scale as f32 * 0.02,
      r_in = ladder[k].0,
      r_out = r_out,
    );
  }
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

/// **在飞 + 待挂载的总量上限**（背压闸门）：worker 的产出速度远高于"挂载 → 标脏 → 上传"的消化速度
/// （实测粗档产出 **0.30 ms/chunk** ⇒ 3 个 worker 约 10 K chunk/s，而挂载预算只有 ~48 条/帧 ≈ 2.9 K/s）
/// ⇒ 不封顶的话 `ready` 会**无界增长**（每秒上万棵树、上百 MB 内存，worker 全在往一个只会变长的队列里
/// 灌 —— 表现为"一动就掉帧、越飞越卡"）。满仓就不再派发，把产能留给"预算调大"的时刻。
const READY_MAX: usize = 384;

/// 每 volume 卸载扫描（`stream_chunks` ②）的**兜底全扫周期**（帧）：平时靠"窗口平移 / 块数变化 /
/// 超容量"三个信号跳过那趟 O(常驻块数) 的遍历，这个周期只是防"同帧进一块、出一块"这种块数不变的
/// 换手漏过（约 4 s 一次，成本摊到几百分之一）。
const SWEEP_FRAMES: u32 = 240;

/// **每个远场级**的在飞条数上限（见 [`VolState::inflight`] 的 WHY）。
///
/// REF: 这个值原先按"产出贵"定（远场一块要按需读 Anvil，几十 ~ 几百 ms）⇒ 6 条就把 worker 喂满，
/// 调大只会让"每帧完成量"成簇 ⇒ 填充期掉帧。装上离线 LOD 后**产出降到 0.2 ms/块**（469×），闸门
/// 换成了它 —— 实测（`GATE_BENCH=orbit`，`video/vsync` 关）：6 → 24 时 L1 同时刻常驻 **199 → 637**
/// （t = 9 s，**3.2×**），而填充期帧周期中位 6 ms、`> 10 ms` 2/25（最大 12.7）。取 **12**（6 与 24
/// 的中值；24 那一档填充期有 8% 的帧越过 10 ms 底线）—— 底数是"每帧完成几条"不再由产出决定。
///
/// CONSTRAINT: 别只看填充速度调大：填充期的帧稳由**每帧完成量**决定，而那正是这条闸门在管的。
const FAR_INFLIGHT_MAX: usize = PRODUCER_WORKERS * 4;

/// **单 volume 的流式状态**（M8：主世界 + 每个远场级各一份）。主世界与远场级走**同一套**驱动逻辑，
/// 差别只有"需求怎么算"（见 [`plan_generation`] 与 [`plan_generation_far`]）。
#[derive(Default)]
struct VolState {
  /// **需求表**（按 [`plan_generation`] 的键排好序）：只在相机换 chunk / 每
  /// [`DEMAND_REBUILD_FRAMES`] 帧重建；每帧只从表头派发（已满足 / 已在飞的当场划过）
  demand: std::collections::VecDeque<(IVec3, Detail)>,
  /// 建表时的相机 chunk（该 volume 自己的 chunk 空间；None = 还没建过）
  demand_center: Option<IVec3>,
  /// 距上次建表过了几帧
  demand_age: u32,
  /// 已取回、还没挂载的产出（挂载按**字数预算**逐帧消化，见 [`Streaming::mount_words`]）。
  ///
  /// 末位 = **序列化字数**，由 worker 线程算好带过来（`produce::ChunkProducer::poll` 的末位）：
  /// `ChunkTree::len_words()` = `serialize().len()`，在主线程现算就是"每挂一块整树序列化一遍"
  /// （实测 8–31 ms/帧，见 `mount_words` 的 CONSTRAINT）。
  ready: std::collections::VecDeque<(ChunkCoord, Detail, ChunkTree, usize)>,
  /// [`Self::ready`] 里已有的 chunk（派发去重用：`producer.in_flight` 只覆盖 worker 手上的）
  ready_set: std::collections::HashSet<ChunkCoord>,
  /// 已挂载 chunk 的档位（粗 / 细）：判"够不够细"（要细化就重新产出，挂载会整体替换）。
  /// 建世界时同步铺的起始块不在表里 —— 缺省按 [`Detail::Full`] 算（它们确实是全分辨率）。
  /// **远场级不用它**（只有一档），留空即可。
  detail: std::collections::HashMap<ChunkCoord, Detail>,
  /// 上一轮卸载扫描时的常驻集序号（M8 闸门）：序号没变、窗口没动、也没超容量时整段跳过那趟
  /// O(常驻块数) 的遍历（见 [`stream_chunks`] ② 的说明）。
  last_seq: u64,
  /// **LRU 的"最近使用"**（论文 §III.A 的 usage stamp）：chunk → 最近一次"被主射线看到"的批次序号
  /// （`ChunkUseFeed` 每 `REPORT_PERIOD_SECS` 换一批，`use_seq` 每批 +1）。
  ///
  /// 它取代了原来的"最后被请求时刻 + TTL"：那个键**只在"块缺了"时更新**，块一装好就冻结 ⇒ 看起来
  /// 永远最旧 ⇒ 被回收 ⇒ 再被请求 ⇒ 反复产出（且相机静止时世界也在变 ⇒ GI 时域永远接不上）。
  /// 用途戳反过来：**只要射线还在看它，键就一直涨** ⇒ 正在看的块永远排最后被回收。
  last_used: std::collections::HashMap<ChunkCoord, u64>,
  /// 产出过、但**没有内容**的 chunk（免得每帧重复派发；infinite_cubes 不会出现，留作通用性）
  empty: std::collections::HashSet<ChunkCoord>,
  /// [`Self::demand`] 的集合视图（卸载扫描的**保护集**要按成员判，逐条 `contains`）。
  /// 与需求表同生共死（只在重建需求表时重填）⇒ 别每帧从 `demand` 现折一遍（8192 条插入/帧）。
  demand_set: std::collections::HashSet<ChunkCoord>,
  /// **换出空闲**（M8 的卸载闸门用它收口）：上一次"超容量"扫描**一个可换出的块都没找到** ⇒ 在
  /// 需求表重建 / 窗口移动之前，再扫也是同样结果。
  ///
  /// WHY 需要它：`cap` 是按**全分辨率 1 MiB/块**估的（可编辑树的最坏情形），而粗档块只有几十 KB
  /// ⇒ 菜单「粗档半径」调大后**需求集本身就能超过 `cap`**，于是"常驻 > cap"是**长期**状态，
  /// 而保护集（需求 ∪ 在飞 ∪ `unload_radius` 圈）把每一块都护住 ⇒ 每次扫描都空手而归，
  /// 却每帧重扫一遍 O(常驻块数)（实测 MC 地图 = 8.7 ms/帧里的主要部分）。
  trim_idle: bool,
  /// 已派发、还没取回的条数（**本卷**的在飞数）。
  ///
  /// WHY 要按卷数：worker 池的去重 / 在飞上限是**全局**的（`ChunkProducer::max_inflight`），而各卷的
  /// 单块成本差三个数量级 —— MC 远场一块要读几百个 chunk 列（`mc::summary`），近场一块只读 1~16 个。
  /// 不按卷封顶的话，远场一开场就把全局在飞槽位占满（`READY_MAX` 也跟着顶住）⇒ **近场加载被饿死**。
  inflight: usize,
}

/// 生产管线的活状态（M5）：worker 池 + **逐 volume 的**流式状态。
///
/// M8：三级远场与主世界共用**一个** worker 池（一个线程池，任务带卷号）与一套节流常量 ——
/// 各自起池会让线程数按级数翻倍，而生成是纯 CPU。用一个池还有个好处：远场与近场**按同一份帧额**
/// 竞争，不会出现"远处刷得凶、脚下反而饿着"。
struct Pipeline {
  producer: ChunkProducer,
  /// 这个池在给哪个源干活（换世界时比对它决定是否重建）
  source: std::sync::Arc<dyn ChunkSource>,
  /// 每个 volume 一份（下标 = `Volumes.list` 下标 = shader 的 `grid_descs` 下标）
  vols: Vec<VolState>,
  /// 用途批次的序号（每**换一批** `ChunkUseFeed` +1），就是各 [`VolState::last_used`] 的值域。
  use_seq: u64,
  /// 最近处理过的用途戳（判"有没有换新的一批"；同批反复处理会把"没有新信息"当成"又一轮最近"）。
  use_stamp: u32,
}

/// 单 volume 的几何 scope（该 volume **自己的 chunk 空间**）：窗口 + 相机在其空间里的 chunk。
#[derive(Clone, Copy)]
struct VolScope {
  /// 该 volume chunk 空间里的**流式中心**：横向 = 相机、竖向 = 相机（竖向无限的世界）或**内容带中点**
  /// （源声明了 `content_y_range` 的远场级，见 ⓪）
  center: IVec3,
  /// 该 volume 当前的窗口（chunk 单位）
  w_origin: IVec3,
  w_dims: IVec3,
  /// 本帧窗口是否刚平移过（卸载扫描的闸门之一：平移才可能有块"出门"）
  moved: bool,
  /// 远场预载枚举的**竖向半高**（chunk）：内容带已知时 = `内容行数/2 + 1`，否则 [`FAR_PRELOAD_HY`]
  v_hy: i32,
}

impl VolScope {
  fn in_window(&self, c: IVec3) -> bool {
    let t = c - self.w_origin;
    t.cmpge(IVec3::ZERO).all() && t.cmplt(self.w_dims).all()
  }
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
  /// 相机近旁的**保护半径**（chunk）：这一圈内一律不参与 LRU 裁切（也必须 > `load_radius`，迟滞，
  /// 免边界反复装卸）。用途戳要等 `REPORT_PERIOD_SECS`（2 s）才来第一批 ⇒ 头两秒没有任何"最近使用"
  /// 信号，此时若只按 `last_used` 裁，会先把相机脚下的块（`last_used = 0`，与所有块并列最旧）卸掉。
  pub unload_radius: i32,
  /// **粗档**（16³ 量化）加载半径（chunk，xz 上的切比雪夫）
  pub coarse_radius: i32,
  /// 粗档的**竖向**半高（chunk）：`xz` 上是圆柱、`y` 上是薄板 —— 三维立方环里绝大多数 chunk
  /// 在头顶/脚下几十米处（谁也不看），同样的内存不如换成 xz 上的半径。
  pub coarse_height: i32,
  /// 每帧挂载的**序列化字数**预算（挂载要主线程做：装树 + 标脏 + 后面排队序列化上传）。
  /// 序列化约 0.5 ms/MB ⇒ 256 K 字（1 MB）/帧 ≈ 0.5 ms/帧，与 `UPLOAD_BYTES_PER_FRAME` 同量级；
  /// 折算成 chunk = 每帧约 3.7 个全分辨率 或 87 个粗档。
  ///
  /// CONSTRAINT: 这个预算的**分子（每块字数）必须来自 worker**（`VolState::ready` 的末位），
  /// 别在挂载循环里现算 `ChunkTree::len_words()` —— 它是 `serialize().len()`（整树走一遍 + 等长分配），
  /// 放在主线程等于"每挂一块序列化一遍"：实测（`GATE_BENCH=fly`，MC 城市移动中）**8–31 ms/帧**，
  /// 是当时帧时掉到十几 fps 的唯一大头。
  pub mount_words: usize,
  /// 每帧挂载的条数上限（防"一堆极小树"把单帧的记账开销顶爆）
  pub mount_count: usize,
  /// **常驻池的内存预算**（字节）：交给渲染侧当 `ResidencyPolicy::budget_bytes`（超了就按 LRU 换出），
  /// 本模块的 CPU 侧保留量也按**同一个块数**封顶（[`gate_render::pool_capacity_chunks`]）。
  ///
  /// 口径是**块数**，且按 **CPU 侧**的每块字节折算（[`gate_render::pool_capacity_chunks`]）——
  /// 实测 CPU 侧约 **0.54 MB/chunk**（GPU 侧的 wire 只要约 0.30 MB），见 `CPU_CHUNK_BYTES_EST`。
  pub request_bytes: usize,
  /// **诊断：远场级上限**（0..=3）。`vol > 本值` 的远场级**整卷停用**（不产出、不挂载、已常驻的全部卸载）
  /// —— 相机上依次关 L3 → L2 → L1，看画面上某一圈的异常在哪一级消失，就能钉死它是哪一级造成的。
  ///
  /// 不改任何渲染语义：静音的卷只是"没内容"，射线穿过去看到的是它后面的东西。
  pub far_levels_max: usize,
  /// **是否让请求圈真的装载**（关闭 ⇒ 回到"请求只在半径环内重排"的旧行为，便于 A/B）
  pub requests_load: bool,
  /// **生产源**：`None` = 用内置的 `infinite_cubes` 生成器；MC 地图在建世界时塞自己的源（见
  /// `crate::mc::build`）。换世界时两边不一样 ⇒ 池会重建（见 [`stream_chunks`] ①）。
  source: Option<std::sync::Arc<dyn ChunkSource>>,
  /// 内置生成器的源 + 建它时的 `n_pbr`（**惰性建一次，缓存住**）。
  /// WHY 必须缓存：判"要不要重建池"用的是 `Arc::ptr_eq`，每帧 `Arc::new` 会让它**恒为假** ⇒
  /// 池每帧重建 ⇒ 在飞的产出全被丢掉、一块也挂不上（实测常驻集卡在初始的 27 块不动）。
  /// `n_pbr` 变了（用户换 PBR 资产）⇒ 槽号方案变了 ⇒ 那时才该换一个新的源。
  builtin: Option<(usize, std::sync::Arc<dyn ChunkSource>)>,
  /// [`ChunkSource::palette_log`] 的游标：已经装进各 volume 调色板的条数（换源 / 换世界清零重放）
  palette_applied: usize,
  /// M5 生产管线：首次进入流式世界时惰性起（要 `n_pbr` 才能定槽号方案）
  pipeline: Option<std::sync::Mutex<Pipeline>>,
}

impl Streaming {
  /// **换世界时同步生产源**（`None` = 回到内置的 `infinite_cubes` 生成器）。
  ///
  /// 只动这一项 —— 半径 / 预算那些量**同时被 DebugMenu「世界」页拥有**：`apply_initial_state` 会把
  /// 菜单的最终控件值重放成 `MenuActionEvent`（在 `setup` 之后一帧执行）⇒ 世界代码写它们会被当场
  /// 覆盖（实测：`tune_for_mc` 设的 `coarse_radius=4` 被菜单的 8 顶掉）。世界的调参只能走菜单，
  /// 见 `docs/mc_map.md` §7。
  pub fn set_source(&mut self, source: Option<std::sync::Arc<dyn ChunkSource>>) {
    self.source = source;
  }

  /// 是否挂了**自定义生产源**（⇒ 槽号由源自己认领，走 `palette_log` 重放；见
  /// [`attach_far_levels_mc`]）。`false` = 内置生成器的固定槽号方案。
  pub fn has_custom_source(&self) -> bool {
    self.source.is_some()
  }

  /// 当前的生产源（`None` = 内置生成器）。给"要拿到这个源本身"的调用方用 —— 见
  /// [`ChunkSource::clone_as_any`]（DebugMenu 的「构建 LOD 缓存」）。
  pub fn source(&self) -> Option<std::sync::Arc<dyn ChunkSource>> {
    self.source.clone()
  }
}

impl Default for Streaming {
  fn default() -> Self {
    // **档位不在这里定**：请求那一路由射线报回的 `level` 定（[`detail_of_req`]）、半径补块那一路由
    // "这一级在屏幕上是否 ≤ 1 px"定（[`detail_at`]）—— 1080p 下逐体素一路到 **~75 m**（14.6 chunk）、
    // 8 cm 档从 75 m 起、32 cm 档要 ~300 m（58 chunk）才允许。所以这几个半径只管**要哪些 / 保哪些**：
    // - `load_radius` 2 chunk（≈10 m）：脚下与眼前永远先装（排序第一优先）；
    // - `unload_radius` 3 chunk：这一圈内**一律不裁**（保底常驻，相机回头不该看到洞）；
    // - 更外面的可见部分交给**请求**（ray-guided），总量由 `request_bytes`（常驻池预算）封顶。
    Self {
      paused: false,
      load_radius: 2,
      unload_radius: 3,
      coarse_radius: 3,
      coarse_height: 3,
      // 挂载字数预算：**8 个全分辨率 chunk/帧**（每个 73 K 字）。原值 256 K 只够 3.5 个，与 GPU 侧
      // `max_install_per_frame` 一起把冷启动拖到十几秒（见 `ResidencyPolicy::max_install_per_frame`）。
      mount_words: 1024 * 1024,
      mount_count: 48,
      // 池预算 **2 GB**（≈2048 块；换算与实测见 [`Streaming::request_bytes`]）。
      // 依据是**本机 16 GB 内存**：实测每块的**整进程**开销约 `3.2 MB`（CPU 树 `0.54` + GPU wire
      // `0.33` + builder/wgpu 约 `2.3`）⇒ 2048 块 ≈ `6.5 GB` + 约 `1.8 GB` 进程基座。
      request_bytes: DEFAULT_REQUEST_BYTES,
      requests_load: true,
      far_levels_max: 3,
      source: None,
      builtin: None,
      palette_applied: 0,
      pipeline: None,
    }
  }
}

/// M5 的生产源（程序化）：在 worker 线程上跑，**只引用确定性槽号**（[`slot_of`]），
/// 不碰主线程的 `VolumeGrid`（`scratch` 是 worker 独占的）。
///
/// **M8**：同一个源同时服务主世界（`vol == 0`）与远场级（`vol ≥ 1`）—— 远场走
/// [`build_region_far`]（级体素 = `FAR_SCALES[vol-1]`，每格一次采样）。
struct InfiniteCubes {
  n_pbr: usize,
}

impl ChunkSource for InfiniteCubes {
  fn produce(
    &self,
    vol: usize,
    coord: ChunkCoord,
    detail: Detail,
    scratch: &mut VolumeGrid,
  ) -> Option<ChunkTree> {
    let lo = coord.0 * gate_voxel::CHUNK_SIZE;
    let hi = lo + IVec3::splat(gate_voxel::CHUNK_SIZE);
    if vol == 0 {
      build_region(scratch, lo, hi, self.n_pbr, detail);
    } else {
      // 远场级：格粒度固定 = FAR_GRAIN 级体素（`Detail` 不参与 —— 远场只有一档）
      build_region_far(scratch, lo, hi, self.n_pbr, FAR_SCALES[vol - 1], FAR_GRAIN);
    }
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
/// 主世界的档位（M5 粗粒度层）：`load_radius` 内按"这一级在屏幕上是否 ≤ 1 px"给档（[`detail_at`]）、
/// 粗档 chunk 进到细档门槛内会被重新产出顶掉（`mount_chunk_tree` 是整体替换）。
///
/// **M8 · 三级远场**：主世界（`vol == 0`）之外，每个远场级（`vol ≥ 1`）是一个独立 volume ——
/// 级体素 = `FAR_SCALES[vol-1]` 个主世界体素、窗口同样 64³ chunk（覆盖 `±655 m / ±2.6 km / ±10.5 km`）。
/// 三处口径与主世界**刻意一致**，因为整套"窗口平移 + 请求 + LRU"机制由此原样复用：
/// - **窗口钉在相机上**（与 M6 同一手法）：换到该级的 chunk 空间（`相机世界坐标 / scale / 256`）
///   ⇒ `builder.set_window` 的平移索引区一字不改；
/// - **只由请求驱动装载**（[`plan_generation_far`]）：远场**不做半径补块** —— 铺满一级
///   64³ = 262144 个 chunk ≈ 17 GB（§10.4），而射线真正在看的（视锥内）只有约 1.5 万个/级；
/// - **卸载按容量 + LRU**：与主世界同一把尺（用途戳 + `unload_radius` 保护圈），
///   容量按**远场自己的每块字节**换算（[`gate_render::pool_capacity_chunks_far`]），**每级各一份**。
pub fn stream_chunks(
  mut stream: ResMut<Streaming>,
  mut scene: ResMut<gate_render::VoxelScene>,
  cam: Option<Res<gate_render::DdaCameraConfig>>,
  pbr: Option<Res<gate_render::PbrTextureSet>>,
  feed: Option<Res<gate_render::LodRequestFeed>>,
  use_feed: Option<Res<gate_render::ChunkUseFeed>>,
  // 卸载扫描的兜底周期计数（见 ② 的闸门说明）
  mut frames: Local<u32>,
  // 诊断：本系统的耗时（每 60 帧一行，见 `gate_render::profiler::SysTimer`）
  mut diag: Local<(f64, u32)>,
) {
  let _t = gate_render::profiler::SysTimer::new("STREAM 流式装载", &mut diag);
  // WHY: 暂停 = 冻结整个流式环（连窗口都不跟）—— 让相机能飞出加载边界，看"世界到此为止"的那一圈。
  if stream.paused {
    return;
  }
  *frames = frames.wrapping_add(1);
  let Some(cam) = cam else { return };
  if scene.volumes.main().stream_window().is_none() {
    return; // 非流式世界
  }
  let chunk = gate_voxel::CHUNK_SIZE;
  let forward = cam.forward;
  let cam_world = cam.position_world;

  // ⓪' **GPU 常驻池预算**（论文 §III.A 的定长 pool）交给渲染侧：`plan_residency` 用它当
  //     `ResidencyPolicy::budget_bytes`。暂停时这条路不走 ⇒ 渲染侧沿用上一次的值（暂停的语义就是冻结）。
  scene.residency_budget_bytes = stream.request_bytes;

  // ⓪ 窗口跟着相机（**M6**）。窗口是 `b_struct` 索引区的定义域：跑出窗口的 chunk 生成得出、
  //    却装不上 GPU（索引区之外）⇒ 画面成片空洞。旧版是"接近边界就整块重定"（丢全部 CPU chunk +
  //    全量重传，一次性卡顿）；现在每帧把窗口钉在**相机为中心的 `dims`** 上，移动交给 builder 的
  //    **平移索引区**（只搬 1 MB 条目、不重传树块、不丢 CPU 内容）⇒ 世界随你走，不卡。
  //    CONSTRAINT：常驻环必须远小于半窗宽，否则相机走到窗边会卸载掉"还在窗内但已过界"的 chunk。
  //    **远场级同一手法**，只是"相机位置"要换到该级的 chunk 空间（`cam_world / scale`）。
  let n_vol = scene.volumes.len();
  // 远场级的**竖向内容范围**（世界体素，`None` = 竖向无限，如程序化世界）：声明了就把窗口的竖向
  // 中点钉在内容带中心 —— 否则飞过内容高度后预载够不到地面（见 `ChunkSource::content_y_range`）。
  let content_y = stream.source().and_then(|s| s.content_y_range());
  let mut scopes: Vec<VolScope> = Vec::with_capacity(n_vol);
  for vol in 0..n_vol {
    let scale = scene.volumes.list[vol].transform().scale;
    let Some((o, d)) = scene.volumes.list[vol].stream_window() else {
      // 既不是主世界也不是远场级（未使用物体路径）：占位保持卷号对齐，后面按 `is_far_level` 跳过
      scopes.push(VolScope {
        center: IVec3::ZERO,
        w_origin: IVec3::ZERO,
        w_dims: IVec3::ZERO,
        moved: false,
        v_hy: FAR_PRELOAD_HY,
      });
      continue;
    };
    let cam_c = (cam_world / scale / chunk as f32).floor().as_ivec3();
    // 源声明了内容带：**竖向换成内容带的中点**（横向仍跟相机）—— 主世界与远场级都要。
    // 主世界尤其重要：它的预载盒竖向若跟着相机，飞到盒子上方后盒里全是空气（`STREAM[v0 …] y a..b`
    // 会看到 `a` 跟着相机走高而不含内容带）⇒ 地面只剩射线请求补 ⇒ 画面正中空洞。
    // 锚到内容带之后，预载盒、LRU 保护圈、卸载判定全在内容空间里算，与相机高度无关。
    let (c, v_hy) = match content_y {
      Some((lo, hi)) => {
        let span = (chunk as f32 * scale).round() as i32; // 本级 chunk 的世界体素边长
        let lo_c = lo.div_euclid(span);
        // `hi` 是开区间端 ⇒ 向上取整才是"要用到第几个 chunk"；`max(1)` 兜住比一个 chunk 还矮的世界
        let hi_c = (hi + span - 1).div_euclid(span);
        let rows = (hi_c - lo_c).max(1);
        (IVec3::new(cam_c.x, lo_c + (rows - 1) / 2, cam_c.z), rows / 2 + 1)
      }
      None => (cam_c, FAR_PRELOAD_HY),
    };
    let want = c - d / 2;
    let moved = want != o;
    if moved {
      scene.volumes.list[vol].set_stream_window(Some((want, d)));
    }
    scopes.push(VolScope { center: c, w_origin: want, w_dims: d, moved, v_hy });
  }
  let (load_r, unload_r, coarse_r, coarse_h, mount_words, mount_count) = (
    stream.load_radius,
    stream.unload_radius,
    stream.coarse_radius,
    stream.coarse_height,
    stream.mount_words,
    stream.mount_count,
  );
  let (request_bytes, requests_load) = (stream.request_bytes, stream.requests_load);
  // 诊断：远场级上限（见 `Streaming::far_levels_max`）
  let far_max = stream.far_levels_max;
  // **池容量**（chunk）：与渲染侧的 GPU 常驻池**同源换算**（[`gate_render::pool_capacity_chunks`]）
  // —— 两侧各按各的公式算就会出现"CPU 还留着 / GPU 已换出"（每帧重传）或反之（CPU 卸了 / GPU 还占着）。
  // **主世界与远场级分开换算**（M8）：主世界按"可编辑的全分辨率树"（1 MiB/块），远场按几 KB/块
  // —— 同一个口径会把远场卡在 2048 块，而视锥内一级就要 ≈ 1.5 万块（见 `pool_capacity_chunks_far`）。
  let cap = gate_render::pool_capacity_chunks(request_bytes);
  let cap_far = gate_render::pool_capacity_chunks_far(request_bytes);
  // 远场半径阶梯（见 `far_radius_ladder`）：每级"填得满的一整圈" + 级间接力（下一级内圈 = 本级外圈）
  let ladder = far_radius_ladder(cap_far);
  // `cap_main` = 主世界**目标常驻块数**（= 池容量），**请求与预载共用**。它不再随"池里还剩几个空位"
  // 变：预载的语义是"用会被看到的块换掉不会被看到的"（新装的块带最新用途戳，被换掉的是 LRU 最旧
  // 那端），而不是"只在池里还有空位时才补" —— 后者会让池一满预载就整段停掉，相机一动只剩请求在补。
  let cap_main = DEMAND_TABLE_MAX.min(cap);

  // ① 生产管线（M5）：**派发**需求给 worker 池（请求优先，再半径补块），本帧只**挂载**已完成的。
  //    管线惰性起（缺省源要 `n_pbr` 才能定槽号方案 —— 与建世界时装进调色板的那份同源）；
  //    **换源（换世界）⇒ 重建池**：旧池的 worker 还在产旧世界的东西，且新 volume 的调色板是空的
  //    ⇒ 调色板游标清零、从头重放（见 `ChunkSource::palette_log` 的契约）。
  let n_pbr = crate::scene::pbr_asset_ids(pbr.as_deref()).len();
  let source: std::sync::Arc<dyn ChunkSource> = if let Some(s) = stream.source.clone() {
    s
  } else {
    // 内置生成器：**同一个 `n_pbr` 只建一次并缓存**（判"换源"靠 `Arc::ptr_eq`，每帧新建会让它恒为假）
    match stream.builtin.as_ref().filter(|(n, _)| *n == n_pbr).map(|(_, s)| s.clone()) {
      Some(s) => s,
      None => {
        let s: std::sync::Arc<dyn ChunkSource> = std::sync::Arc::new(InfiniteCubes { n_pbr });
        stream.builtin = Some((n_pbr, s.clone()));
        s
      }
    }
  };
  let stale = match &stream.pipeline {
    None => true,
    Some(p) => !std::sync::Arc::ptr_eq(&p.lock().unwrap_or_else(|e| e.into_inner()).source, &source),
  };
  let mut palette_applied = stream.palette_applied;
  if stale {
    stream.pipeline = Some(std::sync::Mutex::new(Pipeline {
      producer: ChunkProducer::new(source.clone(), PRODUCER_WORKERS),
      source: source.clone(),
      vols: (0..n_vol).map(|_| VolState::default()).collect(),
      use_seq: 0,
      use_stamp: 0,
    }));
    palette_applied = 0;
  }
  // 请求与用途戳各取一次（都是每 `REPORT_PERIOD_SECS` 一批的**快照**，不是队列 ⇒ 每帧看到同一份）。
  let requests: Vec<gate_render::LodRequest> =
    if requests_load { feed.as_ref().map(|f| f.peek()).unwrap_or_default() } else { Vec::new() };
  let uses: Vec<gate_render::LodUse> =
    use_feed.as_ref().map(|f| f.peek()).unwrap_or_default();

  let mut generated = 0usize;
  {
    let mut guard = stream
      .pipeline
      .as_ref()
      .expect("上面刚装上")
      .lock()
      .unwrap_or_else(|e| e.into_inner());
    // 拆借：`producer` 与逐卷状态是两个字段，必须分开借（一个池服务所有卷）
    let Pipeline { producer, vols, use_seq, use_stamp, source: _ } = &mut *guard;
    while vols.len() < n_vol {
      vols.push(VolState::default()); // 换世界后卷数可能变（长度对齐）
    }

    // ⓪'' **用途戳 → LRU**（论文 §III.A）：每**换一批**（回读侧每 `REPORT_PERIOD_SECS` 换一批）就把
    //      `use_seq` +1，批内每个 chunk 记成"本批次被看到过"。`last_used` 的值域就是这个批次序号 ⇒
    //      **单调、可比大小**，正是卸载需要的 LRU 键。批号**跨卷共用**（同一批里各卷的戳一起到）。
    //      `peek` 每帧都返回**同一份快照**（GPU 侧也要读它）⇒ 用批内最大戳判"是不是新的一批"，
    //      否则每帧推进一次批次号 = 把"没有新信息"当成"又一轮最近"。
    let stamp = uses.iter().map(|u| u.stamp).max().unwrap_or(0);
    if stamp != 0 && stamp != *use_stamp {
      *use_stamp = stamp;
      *use_seq += 1;
      let seq = *use_seq;
      for u in &uses {
        if let Some(st) = vols.get_mut(u.vol as usize) {
          st.last_used.insert(ChunkCoord(u.chunk), seq);
        }
      }
    }

    // 取回产出 → **各自 volume 的**待挂载队列（一次 poll；`None` = 这个 chunk 没有内容，记下免得反复派发）
    for (v, cc, detail, tree, words) in producer.poll(POLL_MAX) {
      // 诊断：静音的远场级 —— 产出直接丢弃，否则它会堆在 `ready` 里白占背压额度
      if v > far_max {
        if let Some(st) = vols.get_mut(v) {
          st.inflight = st.inflight.saturating_sub(1);
        }
        continue;
      }
      let Some(st) = vols.get_mut(v) else { continue };
      st.inflight = st.inflight.saturating_sub(1);
      match tree {
        // 同一块的在飞重复（派发修好前遗留的）⇒ 留先到的那个，别再堆一份
        Some(_) if !st.ready_set.insert(cc) => {}
        Some(tree) => st.ready.push_back((cc, detail, tree, words)),
        None => {
          st.empty.insert(cc);
          // **同时也告诉渲染侧**：这是"已知空块"，不是"还没加载" —— GPU 索引条目会写成哨兵，
          // shader 就不再对它发 ray-guided 请求（见 `gate_voxel::VolumeGrid::mark_empty_chunk`
          // 与 `gate_render::brickmap::consts::INDEX_ENTRY_EMPTY`）。
          if let Some(g) = scene.volumes.list.get_mut(v) {
            g.mark_empty_chunk(cc);
          }
        }
      }
    }

    // ⓪''' **调色板日志 → 各 volume 的调色板**（见 `ChunkSource::palette_log` 的契约）：必须在
    //       **挂载之前**装好 —— 树里的槽号是 worker 认领的，主线程不装表就会引用到全 0 的槽。
    //       每帧都取一次（不只是有取回产出的帧）：worker 认领的槽可能属于还在飞的树。
    let updates = source.palette_log(palette_applied);
    if !updates.is_empty() {
      for (id, e) in &updates {
        for g in scene.volumes.list.iter_mut() {
          g.palette_mut().set(*id, *e);
        }
      }
      bevy::log::debug!("MC 调色板 +{} 槽（累计 {}）", updates.len(), palette_applied + updates.len());
      palette_applied += updates.len();
    }

    // 挂载的**字数预算跨卷共用**（一个池、一份帧额 ⇒ 远场不会把近场的帧额吃掉）
    let mut words = 0usize;
    let mut words_mounted = 0usize;

    // WHY 轮转起点：以下各段的预算（`DISPATCH_PER_FRAME` / `READY_MAX` 背压 / `mount_count` 挂载额）
    // 都是**跨卷共用**的，而循环按卷号顺序走 ⇒ 末尾那个卷永远只能捡前面剩下的、在背压下常常一条都派不
    // 出去（实测：MC 的 L3 在相机持续移动时恒为 0 个常驻，L1/L2 各涨到两三百）。每帧把起点转一格，
    // 让每一卷轮流当"先到先得"的那个。
    let first = (*frames as usize) % n_vol.max(1);
    for i in 0..n_vol {
      let vol = (first + i) % n_vol;
      // 本卷需求表里来自请求的条数（只有非远场那条路会写；远场整表都是请求，见下面的日志）
      let mut gen_req = 0usize;
      let scope = scopes[vol];
      let st = &mut vols[vol];
      let far_level = scene.volumes.list[vol].is_far_level();
      // 非流式、非远场的 volume（未使用物体路径）：跳过
      if !far_level && scene.volumes.list[vol].stream_window().is_none() {
        continue;
      }
      // 该 volume 的池容量：主世界与远场级**各自的口径**（见上面 `cap` / `cap_far` 的说明）
      let cap = if far_level { cap_far } else { cap };
      let grid = &mut scene.volumes.list[vol];
      // 诊断：静音的远场级 —— 已常驻的整卷卸载后跳过（不产出、不挂载、不等 LRU）
      if far_level && vol > far_max {
        for cc in grid.chunk_coords().collect::<Vec<_>>() {
          grid.unmount_chunk(cc);
          st.detail.remove(&cc);
          st.last_used.remove(&cc);
        }
        st.demand.clear();
        st.ready.clear();
        st.ready_set.clear();
        continue;
      }

      // ①a **重建需求表**（相机换 chunk / 每 [`DEMAND_REBUILD_FRAMES`] 帧，顺带吸收新的请求）
      st.demand_age += 1;
      if st.demand_center != Some(scope.center) || st.demand_age >= DEMAND_REBUILD_FRAMES {
        let have = |c: IVec3, want: Detail| {
          let held = st.detail.get(&ChunkCoord(c)).copied().unwrap_or(Detail::Full);
          (grid.chunk(ChunkCoord(c)).is_some() && held >= want) || st.empty.contains(&ChunkCoord(c))
        };
        let batch: Vec<(IVec3, Detail)> = if far_level {
          let (r_in, r_out) = ladder[(vol - 1).min(ladder.len() - 1)];
          plan_generation_far(vol, scope, far_cap(vol, cap), r_in, r_out, &requests, have)
        } else {
          let (batch, from_req) = plan_generation(
            GenScope {
              center: scope.center,
              w_origin: scope.w_origin,
              w_dims: scope.w_dims,
              load_radius: load_r,
              coarse_radius: coarse_r,
              // 竖向半高：源声明了内容带就用**内容带的行数**（[`VolScope::v_hy`]）——
              // 预载盒的竖向于是**锚在地面上、与相机高度无关**。用菜单的 `coarse_height` 时，
              // 相机飞到它之上（MC 世界只有 16 行）预载盒里全是空气 ⇒ 地面只剩射线请求补 ⇒
              // "升空俯瞰正中是空洞"（判据：`STREAM[v0 …] y a..b` 必须恒包含内容带）。
              coarse_height: if content_y.is_some() { scope.v_hy } else { coarse_h },
              forward,
            },
            cap_main,
            &requests,
            have,
          );
          gen_req = from_req;
          batch
        };
        // 需求表的规模是流式世界的**第一诊断读数**（每 `DEMAND_REBUILD_FRAMES` 一行）：预载
        // 与请求各占多少、池里已经有多少。排查"常驻集涨得比预载盘大"这类问题就靠它。
        if far_level {
          bevy::log::debug!(
            "DEMAND[v{vol} far] batch {}（上限 {}；常驻 {}）",
            batch.len(),
            far_cap(vol, cap),
            grid.chunk_count(),
          );
        } else {
          bevy::log::debug!(
            "DEMAND[v{vol} far{far_level}] batch {}（请求 {gen_req}；预载盘 {coarse_r}×{coarse_h}、上限 {cap_main}；常驻 {}）",
            batch.len(),
            grid.chunk_count(),
          );
        }
        st.demand = batch.into();
        st.demand_set = st.demand.iter().map(|(c, _)| ChunkCoord(*c)).collect();
        st.demand_center = Some(scope.center);
        st.demand_age = 0;
        // 需求集换了 ⇒ 保护集换了 ⇒ 允许再试一次"超容量换出"（见 `VolState::trim_idle`）
        st.trim_idle = false;
      }

      // ①b 派发：从表头取（已满足 / 已在飞的当场划过），到在飞上限为止
      let mut dispatched = 0usize;
      while !(far_level && st.inflight >= FAR_INFLIGHT_MAX) {
        // 背压（见 `READY_MAX`）：待挂载 + 在飞已经够多就停 —— 否则 worker 会把产出灌进一个无界队列
        if st.ready.len() + producer.inflight() >= READY_MAX {
          break;
        }
        let Some((c, detail)) = st.demand.front().copied() else { break };
        if producer.in_flight(vol, ChunkCoord(c)) {
          st.demand.pop_front();
          continue;
        }
        // **已在待挂载队列里的也算"已经在飞"**：`in_flight` 只覆盖 worker 手上的，产出取回后它就不再为真
        // ⇒ 不挡的话同一块会被反复重产、反复重挂（`chunk_count` 不动、`gen` 每帧都在涨、
        // 而 `ready` 顶在 `READY_MAX` 把真正的新块堵在后面）。
        if st.ready_set.contains(&ChunkCoord(c)) {
          st.demand.pop_front();
          continue;
        }
        if !producer.request(vol, ChunkCoord(c), detail) {
          break; // 在飞满（或已在飞）⇒ 下帧继续
        }
        st.inflight += 1;
        st.demand.pop_front();
        dispatched += 1;
        if dispatched >= DISPATCH_PER_FRAME {
          break;
        }
      }

      // ①c 挂载：按**字数预算**逐帧消化（装树 + 标脏 + 后面的序列化上传都吃这条预算）；主世界的
      //     粗档细化也走这条路（`mount_chunk_tree` 是**整体替换** ⇒ 粗树被细树顶掉）。
      while !st.ready.is_empty() && words_mounted < mount_count {
        // CONSTRAINT: 字数**必须取 worker 算好的那份**，别在这里现算 `tree.len_words()`
        // —— 那是整树序列化一遍（实测 8–31 ms/帧，见 [`Streaming::mount_words`]）。
        let w = st.ready.front().expect("刚看过 len").3;
        // 预算用尽就停，但**至少挂一个**：否则一个超大 chunk 会永远排不上
        if words_mounted > 0 && words + w > mount_words {
          break;
        }
        let (cc, detail, tree, _words) = st.ready.pop_front().expect("刚看过 front");
        st.ready_set.remove(&cc);
        grid.mount_chunk_tree(cc, tree, 0);
        st.detail.insert(cc, detail);
        // **装载即使用**（与 GPU 侧的 `note_resident` 同一个口径）：请求带进来的块在射线眼里
        // 还是"缺的"（射线没进过它 ⇒ 没有用途戳），若不在这里记一笔，它一挂上就被 LRU 判成最旧
        // ⇒ 立刻换出 ⇒ 下个窗口又请求同一块（装卸死循环）。
        *use_seq += 1;
        st.last_used.insert(cc, *use_seq);
        words += w;
        words_mounted += 1;
        generated += 1;
      }

      // ② 卸载：**容量 + LRU**（论文 §III.A），不再有"半径 / TTL 保护"。
      //
      // 论文的缓存替换只有一条规则：**最近使用**。旧的三分法（窗口外卸 / **半径环内一律保留** /
      // 请求圈按"最后被请求时刻 + TTL"保留）有两个结构性后果：
      //   · 半径环是"永久保留"的 ⇒ 相机不动时**环形永远不环**（常驻只增不减，日志里恒 `unload 0`）；
      //   · 请求圈的键只在"块缺了"时更新（装好即冻结）⇒ 看着最旧 ⇒ 被卸 ⇒ 又被请求 ⇒ **反复产出**。
      // 现在：窗口外一律卸（装不上 GPU，确定性）；其余按**容量**裁，超了卸"最久没被看到"的。
      // 安全网：相机近旁 `unload_radius` 一圈 + 在飞 / 待挂载的一律不动 —— 用途戳每
      // `REPORT_PERIOD_SECS` 才来一批，头两秒没有任何信号。远场级用的是**该级自己的 chunk 空间**里的
      // 距离（`unload_r` 在 L3 上 = 3 × 256 × 64 体素 ≈ 983 m —— 与它的格尺度成比例，正是想要的）。
      //
      // CONSTRAINT：这里只卸 **CPU 侧**（`VolumeGrid`）。GPU 侧的换出由渲染侧的 `plan_residency`
      // 独立决定（主世界同一个容量、同一个 `last_used` 信号；远场级按"CPU 没了就归还"）。
      //
      // **闸门**（M8）：这趟是 O(常驻块数) 的收集 + 判定，只在三种情形下有事可做：
      //   ① 窗口动了（有块出了窗口）② **超容量**且上次扫描确实换出过东西 ③ 每 `SWEEP_FRAMES` 帧兜底。
      // 三者都不成立时整段跳过（相机静置且已收敛 —— 也就是绝大多数帧）。
      //
      // CONSTRAINT: 这里**不能**用"常驻集变了就扫"（`seq != last_seq`）当主判据：`cap` 按**全分辨率
      // 1 MiB/块**估，而粗档块只有几十 KB ⇒ 菜单「粗档半径」调大后**需求集本身就超过 `cap`**，
      // 而保护集（需求 ∪ 待挂载 ∪ `unload_radius` 圈）把每一块都护住 ⇒ "常驻 > cap"是**长期**状态、
      // 每次扫描都空手而归，可每帧重扫一遍 O(常驻) 就成了纯浪费（实测 MC 地图 8.7 ms/帧的主要来源）。
      // 空手一次就置 `trim_idle`（需求表重建 / 窗口移动时清）⇒ 之后只在兜底帧再试。
      let seq = grid.resident_seq();
      let over_cap = grid.chunk_count() > cap;
      let need_scan = scope.moved
        || (*frames % SWEEP_FRAMES == 0)
        || (over_cap && !st.trim_idle && seq != st.last_seq);
      st.last_seq = seq;
      if !need_scan {
        continue;
      }
      let resident: Vec<ChunkCoord> = grid.chunk_coords().collect();
      // **① 出窗的先摘掉**：这一趟只需要 `in_window` 一个判据（O(常驻) 的算术），
      // 保护集（需求 / 待挂载 / 在飞）对"已经出窗"的块没有意义 —— 出窗就装不上 GPU 了。
      let mut out: Vec<ChunkCoord> = Vec::new();
      for c in &resident {
        if !scope.in_window(c.0) {
          out.push(*c);
        }
      }
      // **② 只有超容量才需要"最久没被看到"的那一份**（下面的 LRU 候选）。
      //
      // WHY 分成两段：上面那 4 次哈希查询（`demand_set` / `ready_set` / `in_flight` / `last_used`）
      // 乘上常驻块数就是**每秒几百 ms** 的量 —— 实测（`GATE_BENCH=fly`，MC 城市移动中）这趟扫描
      // 占主世界调度 **24–38 ms/次**（≈2–4 次/s ⇒ 帧率在 20–70 之间摆）。而容量**没超**时
      // （`cap` 4 GB ⇒ 65536 块，实测常驻 1.9 万）候选一份都用不上 ⇒ 白付。
      let over = resident.len().saturating_sub(out.len()).saturating_sub(cap);
      st.trim_idle = false; // 没超容量时不敢自称"无事可做"（保守：下次超容量仍允许扫）
      if over > 0 {
        // 保护集直接用缓存好的两份集合（`ready_set` 就是"已取回、待挂载"那份），别每帧现折：
        // 现折一次是 8192（需求）+ ≤384（待挂载）条插入，每帧白花。
        let mut cand: Vec<(u64, ChunkCoord)> = Vec::new();
        for c in &resident {
          if !scope.in_window(c.0) {
            continue; // ① 已经收走了
          }
          if (c.0 - scope.center).abs().max_element() <= unload_r
            || st.demand_set.contains(c)
            || st.ready_set.contains(c)
            || producer.in_flight(vol, *c)
          {
            continue;
          }
          cand.push((st.last_used.get(c).copied().unwrap_or(0), *c));
        }
        st.trim_idle = cand.is_empty();
        // 超容量 ⇒ 卸掉"最久没被看到"的那些（`last_used` 升序 ⇒ LRU 在前）
        cand.sort_unstable_by_key(|(t, c)| (*t, c.0.x, c.0.y, c.0.z));
        out.extend(cand.into_iter().take(over).map(|(_, c)| c));
      }
      let n_unload = out.len();
      for cc in out {
        grid.unmount_chunk(cc);
        st.detail.remove(&cc);
        st.last_used.remove(&cc);
      }

      if generated > 0 || n_unload > 0 || !st.ready.is_empty() {
        if far_level {
          // 诊断：常驻块的**切比雪夫半径上限** + **竖向范围**。预载按半径阶梯从 `r_in` 往外填，
          // `rmax` 是"填到哪"的直接判据：它应当落在 `far_radius_ladder` 的 `r_out` 附近
          // （`unload 0` 同时成立才算"这一圈填满且不再抖动"）；远小于 `r_out` 说明枚举被容量或
          // "已知空"截断 ⇒ 那一圈之外只剩请求驱动（远场 1/64 采样）⇒ 几何稀疏、射线穿过去（= 中空）。
          // `y a..b` 是本级常驻块的竖向范围，必须**包含内容带在本级的行** —— 不包含 ⇒ 那一段高度
          // 本级一个块都没有 ⇒ 射线穿过去（用户判据：中空处着色 = sky color ⇒ 是 miss，不是着色问题）。
          let rmax = grid
            .chunk_coords()
            .map(|c| (c.0 - scope.center).abs().max_element())
            .max()
            .unwrap_or(0);
          let (y0, y1) = grid
            .chunk_coords()
            .map(|c| c.0.y)
            .fold((i32::MAX, i32::MIN), |(lo, hi), y| (lo.min(y), hi.max(y)));
          bevy::log::debug!(
            "STREAM[v{vol} far gen{generated} unload {n_unload} chunks {} cap {cap} rmax {rmax} \
             y {}..{} 中心 {} ready {}]",
            grid.chunk_count(),
            if y0 > y1 { 0 } else { y0 },
            if y0 > y1 { 0 } else { y1 },
            scope.center.y,
            st.ready.len(),
          );
        } else {
          // 诊断：常驻块的**竖向分布**（chunk）。主世界的预载盒是"以相机为中心"的
          // `coarse_height` ⇒ 相机飞到盒子之上时**地面根本不在预载范围内**，脚下那一圈只剩
          // 射线请求补 ⇒ "升空俯瞰正中是空洞"。`y0..y1` 是判据：它应当恒**包含内容带**
          // （MC 内容 = chunk 0..16），而不是跟着相机走高。
          let (y0, y1) = grid
            .chunk_coords()
            .map(|c| c.0.y)
            .fold((i32::MAX, i32::MIN), |(lo, hi), y| (lo.min(y), hi.max(y)));
          bevy::log::debug!(
            "STREAM[v0 gen{generated}(req {gen_req}) unload {n_unload} chunks {} cap {cap} y {}..{} 相机 {} ready {}]",
            grid.chunk_count(),
            if y0 > y1 { 0 } else { y0 },
            if y0 > y1 { 0 } else { y1 },
            scope.center.y,
            st.ready.len(),
          );
        }
      }
    }
  }
  // 调色板游标回写（`stream.pipeline` 那段借用已结束）
  stream.palette_applied = palette_applied;
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
/// - 其余档位：**量化** —— 每 `grain³` 格一个代表材质（解析求格内占比，见 [`build_region_quantized`]）。
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
  if detail == Detail::Full {
    build_region_full(grid, lo, hi, n_pbr)
  } else {
    build_region_quantized(grid, lo, hi, n_pbr, detail.grain())
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

/// 柱板（三条轴向 `COLUMN` 宽的矩形板，周期 `ROOM`）在区间 `[a, b)` 上覆盖的**长度比例**。
///
/// `near(v)` = `v mod ROOM` 落在 room 边界 ±`COLUMN/2` 内 ⇒ 覆盖集正是一族等距板
/// `∪_k [k·ROOM − half, k·ROOM + half)`（`m ≥ ROOM − half` 那一半恰好是 `k+1` 的 `k·ROOM − half`）。
fn near_frac(a: i32, b: i32) -> f32 {
  let half = COLUMN / 2;
  let mut acc = 0i32;
  for k in (a - half).div_euclid(ROOM)..=(b + half).div_euclid(ROOM) {
    let lo = (k * ROOM - half).max(a);
    let hi = (k * ROOM + half).min(b);
    if hi > lo {
      acc += hi - lo;
    }
  }
  acc as f32 / (b - a) as f32
}

/// 量化档：每 `grain³` 格取**格内体积占比最大**的那个材质，整格填成该色。
///
/// **解析求占比**（三条轴向的柱板比例相乘 + cube 的盒交），不是"格内采样碰运气"：格内有没有细柱是
/// **算出来的** ⇒ 不会随视角/位置忽隐忽现（旧版 16³ 格内 8 点采样，柱子只有约 1/3 的概率被戳到）。
/// 成本 = O(格数)，且每轴只需一张比例表（每轴格数 = 256/grain，最多 256）。
///
/// CONSTRAINT：**cube 与柱体互不相交** —— cube 在 room 内占 `[(ROOM−CUBE)/2, …+CUBE)`（52..96），
/// 而柱板只占 room 边界的 ±12 ⇒ 任一轴上 cube 都不落在 `near()` 区间里 ⇒ 两个占比可直接比大小、
/// 不必消重。
///
/// 空（`air`）只有在格内**完全没有实体**时才胜出 —— 与旧口径一致（细结构宁可稍微"膨胀"也别消失）。
fn build_region_quantized(
  grid: &mut VolumeGrid,
  lo: IVec3,
  hi: IVec3,
  n_pbr: usize,
  grain: i32,
) -> u64 {
  let span = hi - lo;
  let n = (span / grain).to_array().map(|v| v as usize);
  // 逐轴比例表：格下标 → 柱板占比
  let axis = |base: i32, count: usize| {
    (0..count)
      .map(|i| near_frac(base + i as i32 * grain, base + (i as i32 + 1) * grain))
      .collect::<Vec<_>>()
  };
  let (fx, fy, fz) = (axis(lo.x, n[0]), axis(lo.y, n[1]), axis(lo.z, n[2]));
  let vol = (grain * grain * grain) as f32;
  let off = (ROOM - CUBE) / 2;
  let mut covered = 0u64;
  let mut memo: Option<(IVec3, PaletteId)> = None;
  for kz in 0..n[2] {
    for ky in 0..n[1] {
      for kx in 0..n[0] {
        let cell = IVec3::new(
          lo.x + kx as i32 * grain,
          lo.y + ky as i32 * grain,
          lo.z + kz as i32 * grain,
        );
        let (a, b, c) = (fx[kx], fy[ky], fz[kz]);
        // 三对轴"任两轴同时落在板上"的并集：容斥（三轴独立 ⇒ 交集的比例相乘）
        let white = (a * b + b * c + a * c - 2.0 * a * b * c) * vol;
        let mut best: Option<(f32, PaletteId)> =
          if white > 0.5 { Some((white, WHITE_SLOT)) } else { None };
        // cube：与格相交的 room（其 cube 的区间 [r·ROOM+off, +CUBE) 与格有交）
        let end = cell + IVec3::splat(grain);
        for rx in (end.x - off - CUBE).div_euclid(ROOM)..=(cell.x - off).div_euclid(ROOM) {
          for ry in (end.y - off - CUBE).div_euclid(ROOM)..=(cell.y - off).div_euclid(ROOM) {
            for rz in (end.z - off - CUBE).div_euclid(ROOM)..=(cell.z - off).div_euclid(ROOM) {
              let room = IVec3::new(rx, ry, rz);
              let (cmin, cmax) = cube_range(room);
              let a0 = cmin.max(cell);
              let b0 = cmax.min(end);
              if b0.cmple(a0).any() {
                continue;
              }
              let d = b0 - a0;
              let v = (d.x * d.y * d.z) as f32;
              let id = match memo {
                Some((r, id)) if r == room => id,
                _ => {
                  let id = material_of(room, n_pbr).0;
                  memo = Some((room, id));
                  id
                }
              };
              // 平手取槽号小者（确定性，与旧口径一致）
              let better = match best {
                Some((bv, bid)) => v > bv || (v == bv && id < bid),
                None => true,
              };
              if better {
                best = Some((v, id));
              }
            }
          }
        }
        if let Some((_, id)) = best {
          covered += fill_clipped(grid, cell, IVec3::splat(grain), grain, lo, hi, id);
        }
      }
    }
  }
  covered
}

/// **远场级产出（M8）**：把 `[lo, hi)`（**级体素**坐标）按 `FAR_GRAIN³` 的格填进 `grid`，
/// **每格一次点采样**（§10.4 的字面口径）。
///
/// 采样点 = 格的**世界中心** `(p + FAR_GRAIN/2) · scale`（`scale` = 级体素有几个主世界体素）——
/// 于是格点锚在**世界坐标**上：窗口滑动只换"哪些 chunk 装着"，不移动格点相位 ⇒ 远处不闪。
///
/// 为什么是点采样而不是 M5 粗档那套解析求占比：
/// - **成本**：每 chunk `16³ = 4096` 格、每格一次 [`voxel_at`]（几次整数运算 + 一次 room 哈希）
///   ≈ **0.3 ms/chunk**（与 M5 的 `Detail::Coarse`（同样 16³ 个格、同样解析求占比）实测同量级）；
///   解析版在同样的格数上要十几倍（它要逐轴算占比 + 扫相交 room），而远场要"几秒铺出上万个 chunk"。
/// - **观感**：解析求占比在格内**不允许空气胜出**（M5 有意为之："宁可膨胀也别消失"）⇒ 粗档格很容易
///   整片判成白（这个 lattice 世界里白柱的格内占比高于 cube）⇒ 远处是一片**白墙**。点采样给出
///   **稀疏**的白格点阵：不发明几何、颜色都是真材质。⇒ **宁可稀疏的真相，不要成片的假白**。
///   要换成解析口径只改这一个函数。
///
/// CONSTRAINT（与 [`build_region`] 同一份生成盒契约）：lo/hi 各轴对齐 chunk（因而也对齐 `FAR_GRAIN`）。
pub fn build_region_far(
  grid: &mut VolumeGrid,
  lo: IVec3,
  hi: IVec3,
  n_pbr: usize,
  scale: i32,
  grain: i32,
) -> u64 {
  let chunk = gate_voxel::CHUNK_SIZE;
  debug_assert!(
    lo % chunk == IVec3::ZERO && hi % chunk == IVec3::ZERO,
    "远场区域须对齐 {chunk} 的 chunk：lo={lo} hi={hi}"
  );
  debug_assert!(grain as i64 * scale as i64 > 0, "级体素与格宽必须为正");
  let half = grain / 2;
  let voxels = (grain * grain * grain) as u64;
  let mut covered = 0u64;
  let mut z = lo.z;
  while z < hi.z {
    let mut y = lo.y;
    while y < hi.y {
      let mut x = lo.x;
      while x < hi.x {
        // 格中心（级体素）→ 世界体素：`world = p · scale`（远场级是 scale-only 变换、与主世界同原点）
        let c = IVec3::new(x + half, y + half, z + half) * scale;
        let id = voxel_at(c, n_pbr);
        if !id.is_air() {
          covered += fill_bricks(grid, IVec3::new(x, y, z), IVec3::splat(grain), grain, id) as u64
            * voxels;
        }
        x += grain;
      }
      y += grain;
    }
    z += grain;
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

/// "在视野内"的点积门槛：`±60°` 锥。**只当排序的次序键用**（同距离时先给看得见的那一侧）；
/// 连续值参与主排序反而会被远处的巧合压过距离。
const VIEW_COS: f32 = 0.5;

/// **交接余量**（chunk）：每一级的内容都要**越过**自己（以及上一级）的裁剪面这么多。
///
/// WHY 需要它：内容锚在 **chunk 角**（`scope.center = floor(cam / chunk)`），而分壳裁剪的半径按
/// **相机**量 ⇒ 任何一块内容边缘，沿某个方向最多比"相机量到的半径"短 `√2` chunk（≈1.42，两个水平轴
/// 同时偏一格）。于是（`chunk` 索引 = 该 chunk 的**近角**，所以内容范围是 `[r_in·chunk, (r_out+1)·chunk]`）：
///
/// - **外缘**：本级内容的远边 `(r_out + 1)·chunk` 最不利时只有 `(r_out + 1 − √2)·chunk` ⇒ 拿
///   `(r_out + 1)·chunk` 当裁剪面会在**外圈**留环。
/// - **内缘**：下一级内容的近边 `r_in·chunk` 最不利时推到 `(r_in + √2)·chunk` ⇒ 只按"半径相等"接力
///   会在**内圈**留环。
///
/// 环里**谁都没有内容** ⇒ 射线直接穿过去 ⇒ 画面上"楼被切掉一条、透出后面那一级"，而且相机在 chunk
/// 内挪十几米就换个方向出现（用户口径："稍微移动近一点就正常了"）。取 **2** ≥ √2 就盖住它。
///
/// CONSTRAINT: 这一条与"画要互斥"不冲突 —— 裁剪只按距离切，内容重叠的部分不会被画两次。别用"两级
/// 半径相等"来理解密铺（那是**内缘**各差 `SEAM_MARGIN` 的错位写法）。
pub(crate) const SEAM_MARGIN: i32 = 2;

/// **远场预载的内圈半径**（该级自己的 chunk 单位）：比它近的东西由更细的一级负责，不必在这一级重复装。
///
/// = **主世界覆盖半径（`mc::NEAR_COVER_CHUNKS` = 30 chunk）折成本级 chunk，再留 [`SEAM_MARGIN`]**：
/// `30 / 4 − 2 = 5`。与主世界覆盖半径**同源**只是必要条件（两者锚在不同的 chunk 角上）—— 真正的判据
/// 见 [`SEAM_MARGIN`]：本级内容的**近边**最不利时（`(r_in + √2)·4` chunk）仍要落在主世界裁剪半径
/// （30）之内 ⇒ `r_in ≤ 30/4 − √2 = 6.09`，取 5 是留足余量（6 只剩 1.8 m 富余）。
///
/// 取 5 **不花代价**：阶梯的预算摊到 `r_in..r_out` 上，本级 `r_out` 仍由预算定（实测 L1 仍是 16），
/// 多出来的只是最内那一圈块。
const FAR_PRELOAD_INNER: i32 = crate::mc::NEAR_COVER_CHUNKS / 4 - 2;

/// **预载底子档的封顶**（`detail_at` 之上再取 min）：盘内实际档位 = `min(detail_at(距离), 本值)`。
///
/// `Detail::Fine` = **不要 `Detail::Coarse`**（当前取值）：盘内 = 内圈逐体素、41–164 m 走 `4³` 体素档。
/// （米数是程序化世界的显示换算；档位本身按**格粒度**读 —— 见 [`Detail`]。）
///
/// WHY 不能要 `Detail::Coarse`：只有它会把方块**逐块 `fill_brick` 填满 `16³`**（`mc/source.rs` 的
/// `grain == 16` 分支只写 `plan.rep`）⇒ **子方块形状丢掉**（台阶 / 栅栏 / 火把都塌成整格）；而
/// `Detail::Fine`（`4³` 体素）走 `plan.fills` ⇒ 形状完整保留。
///
/// WHY 止步于 `Detail::Fine` 而不上 `Detail::Full`：**MC 侧的产出只有 `4³` 体素一档**
/// （`mc/voxel.rs::BlockPlan::fills` 就是 `4³` 砖；`mc/source.rs` 里 `grain == 1` 与 `grain == 4`
/// 走**同一分支**）⇒ 对 MC 地图 Full 与 Fine 的产出**逐字相同**，"最准"就是 Fine；取 Full 只会
/// 让程序化世界在 41–75 m 白花逐体素的内存（那里 `detail_at` 要的是 Full）。
///
/// 代价：MC 的**满方块**（城市里绝大多数）在 Coarse 与 Fine 下都是**一条整块写** ⇒ 逐字相同，
/// 增量只来自子方块模型 ⇒ 常驻字节基本不变（`upload.rs` 的 64 KB/块 估计仍成立）。
/// 装不下 / 想换视距时按档调：往下退一档就是 `Detail::Coarse`（形状会丢，见上）；或调小「粗档半径」。
const PRELOAD_DETAIL_CAP: Detail = Detail::Fine;

/// **全档位预载的内圈半径**（该级自己的 chunk 单位）：它的外面按 [`PRELOAD_DETAIL_CAP`] 封顶。
///
/// 内圈按裸 `detail_at` 给档（8 chunk = 41 m 处要逐体素），外面封顶到 [`PRELOAD_DETAIL_CAP`]
/// ⇒ 41 m 之外一律 `4³` 体素档。判据不变：**近处画面上仍是最细那两档** —— 那是射线按"≤ 1 px"
/// 报 level 细化出来的（`detail_of_req`）。
const PRELOAD_FULL_CHUNKS: i32 = 8;

/// 远场预载枚举的**外圈上限**（该级自己的 chunk 单位）：与窗口半宽同量级（窗口 64³）。
const FAR_PRELOAD_OUTER: i32 = 30;

/// 远场预载的**竖向半高**（该级自己的 chunk 单位）—— **只在源没声明内容范围时用它**
/// （见 [`ChunkSource::content_y_range`]：MC 那边由内容带推出 `内容行数/2 + 1`）。
///
/// 远场级的 chunk 竖向要盖住"世界有多高"这一层内容：L1 的 chunk 只有 20.48 m 高（16³ 节），
/// 城市立起来一百多米 ⇒ 太薄的板会把高楼挡在预载之外。取 8：L1 = ±164 m（够）、L2/L3 更大只会
/// 枚举到空块（产出便宜、一次 `None` 就登记成"已知空"，见 [`VolState::empty`]）。
const FAR_PRELOAD_HY: i32 = 8;

/// **单个远场级**的目标常驻块数（= 该级的需求表上限），由该级的池容量 `cap` 定。
///
/// 与 [`plan_generation`] 的 `cap` 同义：**请求与预载共用**这一个预算。
///
/// CONSTRAINT: 这里的 `cap` 已经是**常驻目标**（`gate_render::pool_capacity_chunks_far`，比预留区小
/// 1/4），且 `FAR_POOL_CHUNKS` 的口径是**块/级** ⇒ 三个远场级各有一份，互不挤占 ⇒ 直接用满。
/// 别再乘系数：那只把阶梯的 `r_out` 压小（覆盖变近），省下的块不会给别的级用。
///
/// TODO(mc): L1 没有离线粗粒度世界兜底（`mc::lod` 只覆盖 `scale ≥ 16`）⇒ 它那 683 块至今按需读
/// 存档（实测冷读 ~174 ms/块）。把 `mc::lod` 扩到 `scale = 4`（每节加 4³ 细格）之后才该放开它的量。
fn far_cap(vol: usize, cap: usize) -> usize {
  let _ = vol;
  cap
}

/// 远场级的**内容层数**（竖向）：环带成本按 `每环 = 8·层数·r` 块算，这个"层数"是**实测标定值**，
/// 不是"世界高 / chunk 高"那个理论上界 —— 上界假设"每个水平格、每一层都有内容"，而内容只集中在
/// 少数几层 ⇒ 那会把成本高估 3–6 倍，把整级压在很短的半径里（L1 因此只到 246 m）。
///
/// REF: 实测每环块数（`GATE_BENCH=orbit` 3.5 min，用 `常驻数 / Σ_{r_in}^{r_out} r` 摊平）：
/// **L1 = 13.7·r**（r8..12 共 683 块）、**L2 = 5.1·r**（r2..19 共 969 块）、**L3 = 1.4·r**
/// （r4..7 共 31 块；城市 2.29 km 到此为止 ⇒ 只是下界）。
///
/// 下表 = 实测值 × **约 1.6–1.8 倍余量**：阶梯必须仍是一个**上界**，否则外圈填不满、洞就回来了；
/// 余量就是替"更密的图"买的。取值后各级利用率 ~55%（旧上界下是 22%），帧时实测见
/// `docs/mc_map.md` §8.8。
const FAR_CONTENT_LAYERS: [i32; 3] = [3, 1, 1];

/// 远场级的**半径阶梯**（各级自己的 chunk 单位）：`[(r_in, r_out); 3]`。
///
/// 构造性不变量（这条就是"不再有空洞"的定义）：
///
/// 1. **每级的环带是完整的一圈**（`r_in..=r_out` 一次填满，不留半圈）；
/// 2. **画互斥**：本级的裁剪半径（[`far_coverage_voxels`]）= 内容远边 − [`SEAM_MARGIN`] 个本级
///    chunk，下一级的裁剪半径接着它 ⇒ 交接面上每像素只由一级画；
/// 3. **内容密铺**：下一级的内圈要落进本级的裁剪半径**之内**（再留一个下级 chunk 的错位余量，
///    见 [`SEAM_MARGIN`]：`r_in' = (r_out − 1)/4 − 1`）⇒ 交接面上两级都有内容，射线穿不过去。
///
/// 反面（旧的"半径相等"写法）：`r_in' = r_out/4` 且裁剪半径 = 内容远边 ⇒ ①内圈 ②外圈各留一圈
/// **谁都没有内容**的环，射线从环里穿过去、打到楼的内表面 ⇒ 画面上的"骨架 / 中空 / 楼被切一条"。
///
/// `r_out` 由该级容量反算（环带块数 `8·层数·Σ_{r_in}^{r_out} r ≤ budget`）；容量连一整环都放不下时
/// `r_out < r_in` ⇒ 该级不预载，几何交给上一级，绝不放半圈进去。
fn far_radius_ladder(cap: usize) -> [(i32, i32); 3] {
  let mut out = [(0, 0); 3];
  let mut r_in = FAR_PRELOAD_INNER;
  for k in 0..FAR_SCALES.len() {
    let budget = far_cap(k + 1, cap) as u64;
    let layers = FAR_CONTENT_LAYERS[k].max(1) as u64;
    let per_ring = |r: i32| 8 * layers * r.max(0) as u64;
    let (mut r_out, mut sum) = (r_in - 1, 0u64);
    while r_out + 1 <= FAR_PRELOAD_OUTER && sum + per_ring(r_out + 1) <= budget {
      r_out += 1;
      sum += per_ring(r_out);
    }
    out[k] = (r_in, r_out);
    // 下一级的内圈（下级 chunk 数）：先折到本级的裁剪半径（`r_out - 1` 个本级 chunk，= 1 个下级
    // chunk 的 4 倍边长），再退 1 个**下级** chunk 作错位余量（≥ √2 ⇒ 见 `SEAM_MARGIN`）。
    r_in = ((r_out - SEAM_MARGIN + 1) / 4 - 1).max(1);
  }
  out
}

/// **档位判据**（[`Detail`] 的 CONSTRAINT）：粒度 `g` 体素的一档，只有在该 chunk 的距离处
/// `g / (d·px_ang) ≤ 1` 时才允许使用；取允许档里**最粗**的那个（内存最优）。
///
/// 基准 1080p（`px_ang = 2·tan30°/1080 = 1.069e-3` ⇒ `1/px_ang = 935 体素`）⇒ 门槛距离
/// `d ≥ g·935 体素 = g·18.7 m`；换 chunk（5.12 m）为单位就是 `dist ≥ 3.65·g` ⇒ 各档门槛（chunk）：
/// 逐体素 **3.7**、4³（8 cm）**14.6**、16³（32 cm）**58**、64³（1.28 m）**234**、整 chunk **934**。
/// 渲染分辨率更低（像素角更大）时门槛更近；用 1080p 作基准 ⇒ 低分辨率下偏保守（更细、费一点内存）。
fn detail_at(dist_chunks: i32) -> Detail {
  /// 1080p 下"粒度 g 体素"的门槛距离，按 chunk 计（`g·18.7 m / 5.12 m`）
  const CHUNKS_PER_GRAIN: f32 = 3.65;
  let allowed = dist_chunks as f32 / CHUNKS_PER_GRAIN; // 允许的最粗粒度（体素）
  if allowed >= 256.0 {
    Detail::Chunk
  } else if allowed >= 64.0 {
    Detail::Wide
  } else if allowed >= 16.0 {
    Detail::Coarse
  } else if allowed >= 4.0 {
    Detail::Fine
  } else {
    Detail::Full
  }
}

/// **请求档位 → [`Detail`]**：`LodRequest::level` 是射线按"屏幕上是不是 ≤ 1 px"算出来的
/// （`trace.wesl::req_level`：0 = 全分辨率 4³、1 = 16³、2 = 64³、3 = 整 chunk）。
///
/// 只有四档：射线分不出 `Full` 与 `Fine` 的差别（2 cm 与 8 cm 落在同一像素内），所以不造第五档。
///
/// level 1 落到 [`Detail::Fine`] 而**不是** `Detail::Coarse`：Coarse(grain 16) 会把 MC 的方块
/// 整块填满 ⇒ 子方块形状丢掉（见 [`PRELOAD_DETAIL_CAP`]）。改成 Fine 之后，请求与 [`detail_at`]
/// 在 75–164 m 上取值一致 ⇒ 两者不打架（否则请求先产出 Coarse、预载下一帧再把它升到 Fine，
/// 同一块产出两遍）。
fn detail_of_req(level: u8) -> Detail {
  match level {
    0 => Detail::Full,
    1 => Detail::Fine,
    2 => Detail::Wide,
    _ => Detail::Chunk,
  }
}

/// 本帧该产出哪些 chunk（[`stream_chunks`] 第 ① 步的策略，抽成纯函数以便单测）：
/// **请求优先**（票数多的先，M4），再用**预载圆盘**补齐（见下），请求那一路不受 `budget` 约束。
///
/// **请求那一侧不受半径环约束**（这是"ray-guided"的落点）：只要落在**窗口内**就能装 —— 窗口是
/// `b_struct` 索引区的定义域，也是请求圈唯一的硬边界（±32 chunk ≈ ±164 m）。
///
/// **档位跟请求走**（论文的口径："refinement 由渲染结果给"）：请求自带的 `level` 就是射线算出来的
/// "这个 chunk 需要多细"，直接用它（[`detail_of_req`]）。旧版这里一律按距离给档（`detail_at`）⇒
/// 只要 64³ 的远处大块也被升到全分辨率（278 KB/chunk，同一份内存少装 20 倍），且档位随相机微小
/// 移动跳变 ⇒ **细化 → 几何内容变 → 已在收敛的 GI 时域永远接不上**。距离判据（`detail_of`）只剩
/// **预载**那一侧在用 —— 那一路没有射线信息，只能按距离。
///
/// # 预载为什么是**圆盘**而不是视锥
///
/// 请求那一路的语义是"**我已经看过、发现它缺**"（`trace.wesl` 撞到没有树块的 chunk 才记一条）⇒
/// 它永远落后于视线：转身之后新露出来的那一整片，在"看之前"必然是空的 —— 这就是"一转就看到区块
/// 在加载"。所以预载必须**朝向无关**：以相机为中心的一个 XZ 圆盘（`coarse_radius` ×
/// `coarse_height`），把"转身可能看到的"整圈先列进需求表。
///
/// 旧版这里是"前向锥 + 身后一条"：锥只覆盖当前朝向，侧向 40–160 m 完全落在锥外 ⇒ 转身 90° 露出的
/// 那一片只能靠请求一点点补。
///
/// 排序键 = **欧氏距离**（近的先）→ 视野内 → 坐标（确定性）。距离为主 ⇒ 排序**朝向无关**：
/// 需求表在转身前后是同一份，池里的块不会被重排着一遍遍换掉（churn）。
///
/// **`cap` = 目标常驻块数**（池容量）：**请求与预载合起来**不许超过它。
///
/// 旧口径的 `fill` = "池里还剩几个空位"，池满就整段停掉 ⇒ 相机一动只剩请求在补。现在的语义是
/// "用会被看到的块换掉不会被看到的"：新装的块带最新用途戳，被换掉的是 LRU 里最久没被看到的那一端。
///
/// CONSTRAINT: 两者必须**共用**这一个预算。各自独立取 `cap` 的话（请求 + 预载 = 2×cap），
/// 超出池容量的那一半会在每次重建需求表时被互相挤出去再装回来 —— 实测 `RESID[install 8]`
/// 连续 1937/2000 帧（`max_install_per_frame` 顶死），帧率从 60 掉到 36。预载只拿剩下的
/// `cap − 请求数`，稳态下"需求集 = 常驻集"就不再抖。
///
/// 过滤（都要过）：
/// - **窗口内**：窗口是 `b_struct` 索引区的定义域，窗口外产出也装不上 GPU；
/// - **够不够细**由 `have(c, want)` 回答（已挂载且档位 ≥ 要的档 ⇒ 不算需求；粗档在近处会因此被细化）。
///
/// 返回 `(本帧的 (chunk, 档位) 列表, 其中来自请求的条数)`。
fn plan_generation(
  scope: GenScope,
  cap: usize,
  requests: &[gate_render::LodRequest],
  have: impl Fn(IVec3, Detail) -> bool,
) -> (Vec<(IVec3, Detail)>, usize) {
  let GenScope { center, w_origin, w_dims, load_radius, coarse_radius, coarse_height, forward } =
    scope;
  let in_window = |c: IVec3| {
    let t = c - w_origin;
    t.cmpge(IVec3::ZERO).all() && t.cmplt(w_dims).all()
  };
  // 该 chunk 该用哪一档：由"这一级在屏幕上是否 ≤ 1 px"定（[`detail_at`]），不由半径拍
  let detail_of = |c: IVec3| detail_at((c - center).abs().max_element());
  let mut picked: Vec<(IVec3, Detail)> = Vec::new();
  // M8：**只收主世界的请求**（`vol == 0`）—— 远场级的 chunk 坐标是它自己的级体素空间，
  // 数值上会落在主世界的窗口内 ⇒ 不过滤就会让主世界去装"远场请求里那些坐标"的 chunk（白装）。
  let mut requested: Vec<gate_render::LodRequest> = requests
    .iter()
    .filter(|r| r.vol == 0 && in_window(r.chunk))
    .copied()
    .collect();
  requested.sort_unstable_by_key(|r| (std::cmp::Reverse(r.votes), r.chunk.x, r.chunk.y, r.chunk.z));
  picked.extend(
    requested
      .into_iter()
      .filter(|r| !have(r.chunk, detail_of_req(r.level)))
      .map(|r| (r.chunk, detail_of_req(r.level))),
  );
  let from_req = picked.len();
  // 预载只拿"请求之外剩下的"槽位（见上面的 CONSTRAINT）
  let budget = cap.saturating_sub(from_req);
  if budget > 0 {
    // **去重位图**（替代 `HashSet`）：候选是**整个盘** —— 半径 32 chunk 时是 65×19×65 = 8 万个格，
    // `HashSet` 每次 `insert` ≈ 100 ns ⇒ 8 ms/帧（实测 `MAIN 11.38 ms/帧` 就是这么来的）。位图对
    // 这个立方体域是 10 KB/帧、每次测试 ~2 ns ⇒ 同一次枚举 ~1 ms。
    //
    // 只有**盘内**的坐标需要去重：预载循环被 `r`/`h` 界住，盘外的请求坐标不会与预载候选撞。
    let r = coarse_radius.max(load_radius);
    let h = coarse_height;
    let (nx, ny, nz) = ((2 * r + 1) as usize, (2 * h + 1) as usize, (2 * r + 1) as usize);
    let mut seen = vec![0u64; (nx * ny * nz + 63) / 64];
    let bit_of = |c: IVec3| -> Option<(usize, u64)> {
      let d = c - center;
      if d.x.abs() > r || d.y.abs() > h || d.z.abs() > r {
        return None;
      }
      let i = ((d.x + r) as usize * ny + (d.y + h) as usize) * nz + (d.z + r) as usize;
      Some((i >> 6, 1u64 << (i & 63)))
    };
    for (c, _) in &picked {
      if let Some((w, m)) = bit_of(*c) {
        seen[w] |= m;
      }
    }
    // 预载目标档（下面的判据与 `picked.extend` **共用同一份**，避免两处口径分叉）：
    // 内圈（`PRELOAD_FULL_CHUNKS`）按 `detail_at`，更远按 `PRELOAD_DETAIL_CAP` 封顶。
    let detail_for = |c: IVec3| {
      let d = detail_of(c);
      if (c - center).abs().max_element() <= PRELOAD_FULL_CHUNKS {
        d
      } else {
        Detail(d.0.min(PRELOAD_DETAIL_CAP.0))
      }
    };
    let mut todo: Vec<IVec3> = Vec::new();
    // 判据是"**够不够细**"（`detail_for`），不是"有没有块"。
    //
    // WHY：常驻块可能是**远处加载时**按当时距离产出的更粗的档；用"有没有块"当判据的话，它飞近后
    // 永远升不上来（`have(c, Detail::Chunk)` 恒真）。也别指望"细化交给射线"那条路 —— 请求只在
    // 射线发现"这块没有树"时才发，已经有树（哪怕很粗）的块不会被再要。
    //
    // 只升不降（降级交给 LRU 换出）⇒ 一个块每次"变细"只重装一次，且**不会来回抖**：档位随距离
    // 单调，升级后回到远处时 `have(细档, 粗档)` 恒真 ⇒ 不再产生需求。
    // 与那次 60 → 33 fps 的旧尝试的区别：这里用的是**封顶后**的 `detail_for`（内圈 41 m、外面
    // 8 cm），不是裸的 `detail_of`（≤75 m 要 Full）⇒ 需求量级与"原地加载"完全一致。
    let mut push = |c: IVec3, todo: &mut Vec<IVec3>| {
      if !in_window(c) || have(c, detail_for(c)) {
        return;
      }
      if let Some((w, m)) = bit_of(c) {
        if seen[w] & m != 0 {
          return;
        }
        seen[w] |= m;
      }
      todo.push(c);
    };
    // 近处立方核：脚下与眼前，**永远第一优先**（距离排序自然把它排在最前）
    for dx in -load_radius..=load_radius {
      for dz in -load_radius..=load_radius {
        for dy in -coarse_height..=coarse_height {
          push(center + IVec3::new(dx, dy, dz), &mut todo);
        }
      }
    }
    // 外圈：XZ 上的**圆盘**（欧氏半径 ≤ `coarse_radius`）。
    //
    // WHy 是圆而不是切比雪夫方块：分壳的接力半径是**球面**（`world.wesl::trace_scene` 用
    // `grid_coverage_r`），而窗口 AABB 只是承载它的盒子 ⇒ 内容必须与接力同形，否则方块的四个角
    // 会填到 1.41×半径、那里却是下一级的接力范围 ⇒ 角上一条斜缝。
    let r2 = coarse_radius * coarse_radius;
    for dx in -coarse_radius..=coarse_radius {
      for dz in -coarse_radius..=coarse_radius {
        if dx * dx + dz * dz > r2 {
          continue;
        }
        if dx.abs().max(dz.abs()) <= load_radius {
          continue; // 近处立方核已经 push 过
        }
        for dy in -coarse_height..=coarse_height {
          push(center + IVec3::new(dx, dy, dz), &mut todo);
        }
      }
    }
    drop(push);
    // 欧氏距离为主（朝向无关、转身不重排）⇒ 同距离再看视野、最后看坐标（确定性）
    todo.sort_unstable_by_key(|c| {
      let d = *c - center;
      let off = if forward == Vec3::ZERO || d == IVec3::ZERO {
        0
      } else {
        (d.as_vec3().normalize_or_zero().dot(forward) <= VIEW_COS) as i32
      };
      (d.length_squared(), off, c.x, c.y, c.z)
    });
    // 档位一律取 `detail_for`（内圈按 `detail_at`、外面封顶到 [`PRELOAD_DETAIL_CAP`]）——
    // 封顶后的档位一旦被靠近，由上面的判据按 `detail_for` 重装顶细。
    picked.extend(todo.into_iter().take(budget).map(|c| (c, detail_for(c))));
  }
  (picked, from_req)
}

/// **远场级的需求**（M8 第 ① 步的策略，抽成纯函数以便单测）：**请求优先 + 预载球壳**。
///
/// 与 [`plan_generation`] 的差别：
/// - **没有半径阶梯**：远场的格已经是 `FAR_GRAIN` 级体素 ⇒ 档位恒 [`Detail::Full`]
///   （它在这里只表示"已经有了，别再产出"），不再走 `detail_at`。
/// - **预载是"球壳"**：内圈 `FAR_PRELOAD_INNER` 之内的东西由更细的一级负责，本级不必重复装；
///   外圈到窗口边（装不上 GPU 的坐标没有意义）。竖向 `±FAR_PRELOAD_HY`。
///
/// 铺满一级 `64³ = 262144` 个 chunk 要十几 GB，所以**外圈上限**是对"装得下"的让步；但"只由请求
/// 驱动"是不对的 —— 同 [`plan_generation`]：请求永远落后于视线。装了 `mc::lod` 之后本级一块只要
/// 0.2 ms（L3 一格一查表），几千块的预载完全跑得动。
///
/// 过滤：**本卷**的请求（`r.vol == vol`；远场级的 chunk 坐标与主世界数值上会撞）+ **窗口内**
/// （窗口是 `b_struct` 索引区的定义域）+ 还没有（`have`）。
/// 排序：请求按**票数降序**；预载按**欧氏距离升序**（朝向无关）；两组各自内部有序、请求整体在前。
/// `cap` 与 [`plan_generation`] 同义：**请求与预载合起来**不许超过它（共用预算，否则互相挤）。
fn plan_generation_far(
  vol: usize,
  scope: VolScope,
  cap: usize,
  r_in: i32,
  r_out: i32,
  requests: &[gate_render::LodRequest],
  have: impl Fn(IVec3, Detail) -> bool,
) -> Vec<(IVec3, Detail)> {
  let mut req: Vec<gate_render::LodRequest> = requests
    .iter()
    .filter(|r| r.vol as usize == vol && scope.in_window(r.chunk))
    .copied()
    .collect();
  req.sort_unstable_by_key(|r| (std::cmp::Reverse(r.votes), r.chunk.x, r.chunk.y, r.chunk.z));
  let mut out: Vec<(IVec3, Detail)> = req
    .into_iter()
    .filter(|r| !have(r.chunk, Detail::Full))
    .map(|r| (r.chunk, Detail::Full))
    .collect();
  // 预载只拿"请求之外剩下的"槽位（与 `plan_generation` 同一口径）
  let budget = cap.saturating_sub(out.len());
  if budget == 0 {
    return out;
  }
  let mut seen: std::collections::HashSet<IVec3> = out.iter().map(|(c, _)| *c).collect();
  let mut todo: Vec<IVec3> = Vec::new();
  // 半径阶梯（见 `far_radius_ladder`）：`r_in..=r_out` 是**该级填得满的一整圈**。
  // `r_out < r_in`（容量连一环都放不下）⇒ 不预载，交给上一级 —— 绝不放半圈进去。
  //
  // **形状 = 圆**（x/z 上的欧氏半径），不是切比雪夫方块：分壳的接力半径是**球面**（见
  // `world.wesl::trace_scene`），内容必须同形 —— 方块会在四个角上填到 1.41×半径、而接力点只到
  // 1.0×半径 ⇒ 角上多出来的那块被下一级盖住/留下一条斜缝。
  let outer = r_out.min(scope.w_dims.min_element() / 2);
  let (lo2, hi2) = {
    let r = r_in.min(outer);
    (r * r, outer * outer)
  };
  'shell: for dx in -outer..=outer {
    for dz in -outer..=outer {
      let d2 = dx * dx + dz * dz;
      if d2 < lo2 || d2 > hi2 {
        continue;
      }
      for dy in -scope.v_hy..=scope.v_hy {
        let c = scope.center + IVec3::new(dx, dy, dz);
        if scope.in_window(c) && !have(c, Detail::Full) && seen.insert(c) {
          todo.push(c);
        }
      }
      // 攒够候选（留 4× 余量给"距离排序后取前 budget 条"）就停 —— 按半径递增扫，所以停在哪一圈
      // 就是"这一级当前填到哪"
      if todo.len() >= budget.saturating_mul(4) {
        break 'shell;
      }
    }
  }
  todo.sort_unstable_by_key(|c| {
    let d = *c - scope.center;
    (d.length_squared(), c.x, c.y, c.z)
  });
  out.extend(todo.into_iter().take(budget).map(|c| (c, Detail::Full)));
  out
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

  /// **产出成本实测**（`cargo test -p gate-app produce_cost -- --nocapture`）：流式世界的加载速率
  /// 完全由它决定（worker 数 × 1/成本）。粗档与全分辨率都量一下 —— 前者是"看得远"的主力。
  #[test]
  fn produce_cost() {
    let src = InfiniteCubes { n_pbr: 0 };
    let mut scratch = VolumeGrid::new();
    for (detail, n) in [(Detail::Coarse, 64usize), (Detail::Full, 8)] {
      let t = std::time::Instant::now();
      let mut words = 0usize;
      for i in 0..n {
        let cc = ChunkCoord(IVec3::new(i as i32, 0, 0));
        if let Some(tree) = src.produce(0, cc, detail, &mut scratch) {
          words += tree.len_words();
        }
      }
      let ms = t.elapsed().as_secs_f64() * 1000.0 / n as f64;
      println!("[{detail:?}] {ms:.2} ms/chunk, {:.1} K 字/chunk", words as f64 / n as f64 / 1024.0);
    }
  }

  /// **常驻内存实测**（`cargo test -p gate-app mem_per_chunk -- --nocapture`）：流式世界的常驻内存
  /// 几乎全在 CPU 侧的树里。量三个数：树的**堆字节** / 序列化字节（≈ GPU 侧 `struct_buf`）/ 节点数 ——
  /// 池预算按哪个数算、以及"CPU 侧只留序列化形式"能省多少，都看这一行。
  #[test]
  fn mem_per_chunk() {
    let src = InfiniteCubes { n_pbr: 0 };
    let mut scratch = VolumeGrid::new();
    for detail in [Detail::Full, Detail::Coarse] {
      let n = 32usize;
      let (mut heap, mut wire, mut nodes) = (0usize, 0usize, 0usize);
      // 节点种类分布（用 `node_view` 反推）：决定子块池该"定长 64 槽"还是"紧凑池"
      let (mut split, mut fanout, mut uni, mut leaf) = (0usize, 0usize, 0usize, 0usize);
      for i in 0..n {
        let cc = ChunkCoord(IVec3::new(i as i32, 0, 0));
        if let Some(tree) = src.produce(0, cc, detail, &mut scratch) {
          heap += tree.heap_bytes();
          wire += tree.len_words() * 4;
          nodes += tree.node_capacity();
          for id in 0..tree.node_capacity() as u32 {
            let Some(v) = tree.node_view(id) else { continue };
            if !v.children.is_empty() {
              split += 1;
              fanout += v.children.len();
            } else if v.mask != 0 {
              leaf += 1;
            } else {
              uni += 1;
            }
          }
        }
      }
      println!(
        "[{detail:?}] 32 chunk 合计：树 {} B（节点 {} × {} B）、序列化 {} B ⇒ 比例 {:.1}×；\
         可达节点分类：Uniform {} / Split {}（子块 {}）/ Leaf {}",
        heap,
        nodes,
        ChunkTree::NODE_BYTES,
        wire,
        heap as f64 / wire as f64,
        uni,
        split,
        fanout,
        leaf,
      );
    }
  }

  /// **每个 chunk 都必须有内容**（这是 `pipe.empty` 与"请求永不满足"的前提）：柱体沿三条轴每隔
  /// `ROOM`（148）就有一条，而 chunk 边长 256 > 148 ⇒ 任何 chunk 的 x/y 跨度里都至少有一条柱线穿过。
  /// 若某个 chunk 产出 `None`，`stream_chunks` 会把它记进 `pipe.empty` 而**永不挂载** —— 那块的
  /// `entry` 永远是 0 ⇒ shader 每帧为它发请求（表现为"有一个洞一直补不上"）。
  #[test]
  fn every_chunk_has_content() {
    let src = InfiniteCubes { n_pbr: 0 };
    let mut scratch = VolumeGrid::new();
    let mut empty = Vec::new();
    for x in -6..=6 {
      for y in -6..=6 {
        for z in -6..=6 {
          let cc = ChunkCoord(IVec3::new(x, y, z));
          if src.produce(0, cc, Detail::Coarse, &mut scratch).is_none() {
            empty.push(cc.0);
          }
        }
      }
    }
    assert!(empty.is_empty(), "这些 chunk 产出为空（会被请求但永不挂载）：{empty:?}");
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
    let async_tree = src.produce(0, cc, Detail::Full, &mut scratch).expect("管线产出应有内容").serialize();
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

  /// **M8**：远场级产出 = **每格一次点采样**（格中心 · `scale` 的世界坐标）—— 与 [`voxel_at`] 逐格
  /// 同源。顺带钉住两件事：
  /// ① **格点锚在世界坐标上**（窗口滑动只换"装着哪些 chunk"、不移动格点相位 ⇒ 远处不闪）——
  ///    用例就是"格心的世界坐标点采样"，与"这个 chunk 被摆在哪"无关；
  /// ② **稀疏**：这个 lattice 世界里多数格是空气，远场是白格点阵而不是整片实体
  ///    （这是"点采样"相对"解析求占比"的观感取舍，见 [`build_region_far`] 的说明）。
  #[test]
  fn far_region_is_one_point_sample_per_cell() {
    for (i, &scale) in FAR_SCALES.iter().enumerate() {
      let vol = i + 1;
      let cc = ChunkCoord(IVec3::new(-1, 0, 2));
      let lo = cc.0 * gate_voxel::CHUNK_SIZE;
      let hi = lo + IVec3::splat(gate_voxel::CHUNK_SIZE);
      let mut grid = VolumeGrid::new();
      let covered = build_region_far(&mut grid, lo, hi, 0, scale, FAR_GRAIN);
      assert!(covered > 0, "L{vol}（scale {scale}）远场 chunk 不该全空");
      let tree = grid.chunk(cc).expect("远场 chunk 有内容");
      let n = gate_voxel::CHUNK_SIZE / FAR_GRAIN;
      let (mut solid, mut cells) = (0usize, 0usize);
      for kz in 0..n {
        for ky in 0..n {
          for kx in 0..n {
            let p = IVec3::new(kx, ky, kz) * FAR_GRAIN;
            let want = voxel_at((lo + p + IVec3::splat(FAR_GRAIN / 2)) * scale, 0);
            let got = tree.get_voxel(p.x, p.y, p.z);
            assert_eq!(
              got,
              if want.is_air() { None } else { Some(want) },
              "L{vol} 格 {p:?} 的槽号必须 = 格心世界坐标的点采样"
            );
            solid += usize::from(got.is_some());
            cells += 1;
          }
        }
      }
      assert!(solid < cells / 2, "L{vol} 远场应是稀疏的（实体 {solid}/{cells} 格）");
    }
  }

  /// **M8**：梯级的**裁剪半径**逐级递增、最后一级盖过 §10.4 的 5 km 目标，且每一级的**内容环带**
  /// 都**越过**裁剪面（画互斥 + 内容密铺，见 [`far_radius_ladder`] 的不变量）。
  ///
  /// CONSTRAINT：这里断言的是**关系**（内圈落进上级裁剪半径内 / 覆盖递增 / 末级 > 5 km），不是具体
  /// 数字 —— 数字随 `FAR_POOL_CHUNKS` 与 `CPU_FAR_CHUNK_BYTES_EST` 变，抄进测试只会变成"改容量就红"。
  #[test]
  fn far_ladder_covers_five_km() {
    assert_eq!(FAR_SCALES, [4, 16, 64]);
    assert_eq!(FAR_GRAIN, 16, "每 chunk 16³ = 4096 格（取值依据见 `FAR_GRAIN` 的实测表）");
    let ladder = far_radius_ladder(gate_render::pool_capacity_chunks_far(DEFAULT_REQUEST_BYTES));
    let mut prev_out = 0;
    for (k, &(r_in, r_out)) in ladder.iter().enumerate() {
      assert!(r_in >= 1, "L{} 的内圈必须为正（0 与主世界重叠）", k + 1);
      assert!(r_out >= r_in, "L{} 的环带为空（r{r_in}..{r_out}）⇒ 该级不该存在", k + 1);
      if k > 0 {
        // 下一级的内圈 = 上一级的裁剪半径折成下级 chunk 再退 1（错位余量，见 SEAM_MARGIN）
        assert_eq!(r_in, ((prev_out - SEAM_MARGIN + 1) / 4 - 1).max(1), "L{} 的内圈要接上一级", k + 1);
        // 本级内容的**近边**（最不利 `(r_in + √2)·4` 个上级 chunk）要落进上一级的裁剪半径之内
        let near_worst = (r_in as f32 + std::f32::consts::SQRT_2) * 4.0;
        let prev_cov = (prev_out + 1 - SEAM_MARGIN) as f32;
        assert!(near_worst <= prev_cov, "L{} 内容近边 {near_worst} 超出上级裁剪半径 {prev_cov}", k + 1);
      }
      // 内容必须越过**本级**的裁剪半径：内容范围（chunk 角为原点）= `[r_in·c, (r_out+1)·c]`，
      // 最不利方向（相机与窗口角错开 ≤ √2 chunk）短掉 √2 后仍要 ≥ 裁剪半径 (r_out+1−SEAM_MARGIN)·c
      let content_far_worst = r_out as f32 + 1.0 - std::f32::consts::SQRT_2;
      let cov = (r_out + 1 - SEAM_MARGIN) as f32;
      assert!(cov <= content_far_worst, "L{} 的裁剪半径 {cov} 超出了内容远边下界 {content_far_worst}", k + 1);
      prev_out = r_out;
    }
    // 主世界 → L1 的交接：L1 的内容**近边**最不利时（`(r_in + √2)·4`，主世界 chunk 单位）仍要落在
    // 主世界的裁剪半径之内，且留 ≥ 1 chunk 富余（富余太小等于没修）
    let (r_in1, _) = ladder[0];
    let l1_near_worst = (r_in1 as f32 + std::f32::consts::SQRT_2) * 4.0;
    assert!(
      l1_near_worst + 1.0 <= crate::mc::NEAR_COVER_CHUNKS as f32,
      "L1 内容近边下界 {l1_near_worst} chunk 要 ≤ 主世界裁剪半径 {} − 1",
      crate::mc::NEAR_COVER_CHUNKS
    );
    let m = |v: i32| v as f32 * 0.02;
    let cov: Vec<f32> = (1..=FAR_SCALES.len()).map(|v| m(far_coverage_voxels(v))).collect();
    assert!(cov.windows(2).all(|w| w[1] > w[0]), "逐级覆盖必须递增：{cov:?}");
    assert!(cov[cov.len() - 1] > 5000.0, "5 km 视距必须落在梯级之内：{cov:?}");
  }

  /// **M8**：远场 chunk 的树远小于同 chunk 的全分辨率树（"远场几乎免费"的来源）。
  ///
  /// 读数用来核对 [`FAR_GRAIN`] 的取值是否划算：`grain = 4`（级体素）时每格只有 16 世界体素
  /// —— 比白柱（24）还细 ⇒ 远场反而把柱体的截面画出来，**树比全分辨率还大**（实测 418 KB vs 288 KB）。
  /// `grain = 16` 时每格 64 世界体素（≈ 一像素在 655 m 处），树掉到个位数 KB，且"相邻格多半同色"
  /// ⇒ 父层能合并。**这是"接缝略宽"与"内存/产出可承受"之间的取舍**（见 [`FAR_GRAIN`] 的说明）。
  #[test]
  fn far_detail_is_much_smaller_than_full() {
    let src = InfiniteCubes { n_pbr: 0 };
    let mut scratch = VolumeGrid::new();
    let cc = ChunkCoord(IVec3::new(0, 0, 0));
    let full = src.produce(0, cc, Detail::Full, &mut scratch).expect("全分辨率有内容").len_words();
    for vol in 1..=FAR_SCALES.len() {
      let w = src.produce(vol, cc, Detail::Full, &mut scratch).expect("远场有内容").len_words();
      println!(
        "[L{vol} scale {:>2}] {w:>6} 字/chunk（{:>3} KB）；全分辨率 {full} 字（{} KB）= {:.0}× 便宜",
        FAR_SCALES[vol - 1],
        w * 4 / 1024,
        full * 4 / 1024,
        full as f64 / w as f64,
      );
      assert!(w * 8 < full, "L{vol} 远场 {w} 字应远小于全分辨率 {full} 字");
    }
  }

  /// 测试用的整 chunk 盒（chunk -1..=1，覆盖原点两侧的 room）。
  fn region() -> (IVec3, IVec3) {
    let chunk = gate_voxel::CHUNK_SIZE;
    (-IVec3::splat(chunk), IVec3::splat(chunk * 2))
  }

  fn solid(g: &VolumeGrid, p: IVec3) -> bool {
    g.get_voxel(VoxelCoord::new(p.x, p.y, p.z)).is_some()
  }

  /// **预载必须"认升级"，且只升不降**。
  ///
  /// 锁的回归：从**远处**加载过的块（那时按距离只能拿更粗的档）飞近后必须被**按 `detail_for`
  /// 重新产出**顶细，否则它永久停在粗档 —— 请求那一路救不了它（请求只在"这块没有树"时才发）。
  /// 反向也要锁：常驻块已够细时**不许**再产生需求，否则帧帧重装（曾经 60 → 33 fps）。
  #[test]
  fn preload_upgrades_coarse_chunks_and_never_downgrades() {
    // 半径 12 ⇒ 盘里同时有"内圈（≤ [`PRELOAD_FULL_CHUNKS`]，目标按 `detail_at`）"与
    // "外圈（更远，目标封顶到 [`PRELOAD_DETAIL_CAP`]）"
    let scope = gen_scope(1024, 1, 12, 3);
    let near = IVec3::new(2, 0, 0); // 内圈 ⇒ 目标 Full
    let far = IVec3::new(10, 0, 0); // 外圈 ⇒ 目标封顶到 PRELOAD_DETAIL_CAP
    let plan_with = |held: Detail| {
      let have = |c: IVec3, want: Detail| (c == near || c == far) && held >= want;
      plan_generation(scope, 4096, &[], have).0
    };
    // ① 常驻是 Coarse ⇒ 内圈那块必须被升级到 Full
    let batch = plan_with(Detail::Coarse);
    assert!(
      batch.contains(&(near, Detail::Full)),
      "内圈的粗档块必须被重新要（升级到 Full）：{batch:?}"
    );
    // ② 已经够细（Full）⇒ 不再要它（只升不降，否则帧帧重装）
    let batch = plan_with(Detail::Full);
    assert!(!batch.iter().any(|(c, _)| *c == near), "已够细就不该再要：{batch:?}");
    // ③ 外圈已经到封顶档 ⇒ 不该再要（否则盘一铺满就帧帧重装）
    let batch = plan_with(PRELOAD_DETAIL_CAP);
    assert!(!batch.iter().any(|(c, _)| *c == far), "封顶档不该被反复重发：{batch:?}");
    // ④ 外圈的**更粗**块必须被升到封顶档 —— 这就是"从远处重载再飞过来一直是整块"那一类
    let batch = plan_with(PRELOAD_DETAIL_CAP.coarser());
    assert!(
      batch.contains(&(far, PRELOAD_DETAIL_CAP)),
      "外圈的粗档块必须被升到封顶档：{batch:?}"
    );
  }

  /// 测试用 scope：以原点为中心的窗口（半宽 = `window`）；`forward` 由用例按需覆盖
  /// （它只当排序的次序键，见 [`plan_generation`]）。
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

  /// M4 + M8：**请求优先**（票数多的先），预载只能填在请求之后；**请求不受预载预算约束**
  /// （只要在窗口内就装，这是"ray-guided"的落点），但窗口外 / 已够细的请求仍要丢。
  /// **档位由请求自带**（`LodRequest::level`，论文口径 "refinement 由渲染结果给"）：请求不只决定
  /// **装哪里**，多细也由它一起报回来（[`detail_of_req`]）。
  #[test]
  fn requests_outrank_preload() {
    let req = |votes: u32, c: IVec3, level: u8| gate_render::LodRequest {
      vol: 0,
      chunk: c,
      votes,
      level,
    };
    // 窗口开大：让梯级的每一档都落在窗口内（现实里 64³ 索引区只到 8 cm 档，见 `detail_at`）
    let scope = gen_scope(1024, 1, 3, 3);
    let mounted: std::collections::HashSet<IVec3> = [IVec3::new(1, 0, 0)].into_iter().collect();
    let have = |c: IVec3, _want: Detail| mounted.contains(&c);
    // 票数降序；档位横跨射线能报的四档
    let reqs = [
      req(99, IVec3::new(5, 0, 0), 0),
      req(98, IVec3::new(20, 0, 0), 1),
      req(97, IVec3::new(60, 0, 0), 2),
      req(96, IVec3::new(300, 0, 0), 3),
      req(95, IVec3::new(1000, 0, 0), 3),
      req(94, IVec3::new(1, 0, 0), 0),
    ];
    // `cap = 0`：**只放请求**、一条预载都不要（这是"远场预载关掉"的对照口径）
    let (batch, from_req) = plan_generation(scope, 0, &reqs, have);
    assert_eq!(
      batch,
      vec![
        (IVec3::new(5, 0, 0), Detail::Full),
        (IVec3::new(20, 0, 0), Detail::Fine),
        (IVec3::new(60, 0, 0), Detail::Wide),
        (IVec3::new(300, 0, 0), Detail::Chunk),
        (IVec3::new(1000, 0, 0), Detail::Chunk),
      ],
      "票数序、档位逐字取请求报回的；已挂载的 (1,0,0) 不算需求"
    );
    assert_eq!(from_req, 5);
    // `cap = 7`：请求已经占掉 5 条 ⇒ 预载只拿剩下的 2 条（按**距离**近的补：原点与最近的一格）。
    // 预算由两者**共用**（见 [`plan_generation`] 的 CONSTRAINT：各自独立取会把池挤爆 → 帧帧重装）
    let (batch, from_req) = plan_generation(scope, 7, &reqs, have);
    assert_eq!(from_req, 5);
    assert_eq!(batch.len(), 7);
    assert_eq!(batch[5], (IVec3::ZERO, Detail::Full));
    assert_eq!(batch[6], (IVec3::new(-1, 0, 0), Detail::Full), "同距离时按坐标定序（确定性）");
    // 请求圈**只在窗口内**：窗口半宽 8 ⇒ (9,0,0) 在窗外，无论多少票都不装（空位转给预载）
    let narrow = gen_scope(8, 1, 3, 3);
    let (batch, from_req) = plan_generation(narrow, 1, &[req(999, IVec3::new(9, 0, 0), 0)], have);
    assert_eq!(from_req, 0, "窗口外的请求不该装：{batch:?}");
    assert_eq!(batch, vec![(IVec3::ZERO, Detail::Full)]);
    // 无请求（请求装载关掉）⇒ 逐字退回预载盘
    let (batch, from_req) = plan_generation(narrow, 3, &[], have);
    assert_eq!(from_req, 0);
    assert_eq!(
      batch,
      vec![
        (IVec3::ZERO, Detail::Full),
        (IVec3::new(-1, 0, 0), Detail::Full),
        (IVec3::new(0, -1, 0), Detail::Full),
      ],
      "按距离升序（同距离按坐标）"
    );
    // M8：**别的卷的请求不算主世界的需求** —— 远场级的 chunk 坐标会落在主世界窗口内，
    // 不过滤就会让主世界去装那些坐标上的 chunk（白装，且需求表被远场的坐标污染）。
    let far_req =
      gate_render::LodRequest { vol: 1, chunk: IVec3::new(2, 2, 2), votes: 999, level: 0 };
    let (batch, from_req) = plan_generation(narrow, 1, &[far_req], have);
    assert_eq!(from_req, 0, "v1 的请求不该进主世界的需求表：{batch:?}");
    assert_eq!(batch, vec![(IVec3::ZERO, Detail::Full)], "这条空位转给预载");
  }

  /// M5/M6：预载**按距离升序、朝向无关** —— 同距离时视野内先于身后（次序键），但**身后那侧也在
  /// 候选集里**（这是"转身不该看到加载"的落点：旧版是前向锥，转身 90° 露出的那片整片在锥外）。
  #[test]
  fn preload_is_distance_ordered_and_view_independent() {
    let mut scope = gen_scope(20, 0, 3, 3);
    scope.forward = Vec3::new(0.0, 0.0, 1.0); // 朝 +Z 看
    let ahead = IVec3::new(0, 0, 3);
    let behind = IVec3::new(0, 0, -3);
    let (batch, _) = plan_generation(scope, 4096, &[], |_, _| false);
    let at = |c: IVec3| batch.iter().position(|(p, _)| *p == c).expect("候选集里必然入选");
    assert_eq!(batch[0].0, IVec3::ZERO, "距离最近的永远第一");
    assert!(at(behind) > 0, "身后那侧同样在候选集里（预载不看向）");
    assert!(at(ahead) < at(behind), "同距离：视野内 {} < 身后 {}", at(ahead), at(behind));
    // 距离单调不减（预载的主序）
    let mut last = 0;
    for (c, _) in &batch {
      let d = (*c - scope.center).length_squared();
      assert!(d >= last, "距离必须单调不减：{c:?}（{d}）在 {last} 之后");
      last = d;
    }
  }

  /// M8：**档位梯级只由"这一级在屏幕上是否 ≤ 1 px"定**（[`detail_at`]）—— 半径环只决定"要哪些"、
  /// 不决定"多细"。1080p 下台阶 = `3.65 × 粒度` 个 chunk：1 体素(2 cm) 3.7、4³(8 cm) 14.6、
  /// 16³(32 cm) 58、64³(1.28 m) 234、整 chunk(5.12 m) 934。
  #[test]
  fn detail_ladder_follows_screen_size() {
    assert_eq!(detail_at(0), Detail::Full);
    assert_eq!(detail_at(14), Detail::Full);
    assert_eq!(detail_at(15), Detail::Fine);
    assert_eq!(detail_at(58), Detail::Fine);
    assert_eq!(detail_at(59), Detail::Coarse);
    assert_eq!(detail_at(233), Detail::Coarse);
    assert_eq!(detail_at(234), Detail::Wide);
    assert_eq!(detail_at(934), Detail::Wide);
    assert_eq!(detail_at(935), Detail::Chunk);
    // 请求档位是**四档**（射线分不出 Full 与 Fine —— 2 cm 与 8 cm 落在同一像素内）；
    // level 1 落 `Fine`（不是 `Coarse`：Coarse 丢 MC 的子方块形状，见 `detail_of_req`）
    for (level, want) in
      [(0u8, Detail::Full), (1, Detail::Fine), (2, Detail::Wide), (3, Detail::Chunk)]
    {
      assert_eq!(detail_of_req(level), want, "level {level} 的档位映射");
    }
    // 半径内同样按判据给档（`load_radius` 只管优先级），半径外一个都不要
    let (batch, _) = plan_generation(gen_scope(20, 1, 6, 6), 400, &[], |_, _| false);
    assert!(batch.iter().all(|(c, d)| *d == Detail::Full), "6 chunk = 31 m 内都还没到 8 cm 档");
    assert!(batch.iter().all(|(c, _)| c.abs().max_element() <= 6), "半径外不该要");
  }

  /// **M8**：远场级的需求 = **请求 + 预载球壳** —— 请求部分只收**本卷**的（远场级的 chunk 坐标与
  /// 主世界数值上会撞）、按票数降序；预载部分在**内圈之外**（更近的东西由更细的一级负责）、按距离升序。
  #[test]
  fn far_generation_is_requests_then_preload() {
    let scope =
      VolScope {
        center: IVec3::ZERO,
        w_origin: IVec3::splat(-32),
        w_dims: IVec3::splat(64),
        moved: false,
        v_hy: FAR_PRELOAD_HY,
      };
    let req = |vol: u8, votes: u32, c: IVec3| gate_render::LodRequest {
      vol,
      chunk: c,
      votes,
      level: 0,
    };
    let reqs = [
      req(2, 3, IVec3::new(0, 0, 0)),
      req(2, 9, IVec3::new(1, 0, 0)),  // 已挂载 ⇒ 不算需求
      req(0, 99, IVec3::new(2, 0, 0)), // 主世界的请求（坐标会撞）⇒ 必须丢
      req(2, 5, IVec3::new(40, 0, 0)), // 窗口外（半宽 32）⇒ 装不上 GPU，必须丢
      req(1, 99, IVec3::new(3, 0, 0)), // 别的远场级 ⇒ 必须丢
    ];
    let mounted: std::collections::HashSet<IVec3> = [IVec3::new(1, 0, 0)].into_iter().collect();
    let have = |c: IVec3, _want: Detail| mounted.contains(&c);
    // `budget = 0`：只留请求（本卷 + 窗口内 + 还没挂载的；档位恒 Full）
    assert_eq!(
      plan_generation_far(2, scope, 0, FAR_PRELOAD_INNER, 8, &reqs, have),
      vec![(IVec3::new(0, 0, 0), Detail::Full)],
      "只留本卷 + 窗口内 + 还没挂载的；档位恒 Full"
    );
    assert!(
      plan_generation_far(2, scope, 0, FAR_PRELOAD_INNER, 8, &[], |_, _| false).is_empty(),
      "无请求且不预载 ⇒ 空表"
    );
    // 有预载预算：请求整体在前，预载**从内圈之外**起（更近的由更细的一级负责）
    let batch = plan_generation_far(2, scope, 64, FAR_PRELOAD_INNER, 8, &reqs, have);
    assert_eq!(batch[0], (IVec3::new(0, 0, 0), Detail::Full), "请求永远第一");
    let pre = &batch[1..];
    assert!(!pre.is_empty(), "预载应当有内容：{}", batch.len());
    assert!(
      pre.iter().all(|(c, _)| (*c - scope.center).abs().max_element() >= FAR_PRELOAD_INNER),
      "预载必须落在内圈之外（内圈里的由更细的一级装）"
    );
    assert!(
      pre.iter().all(|(c, _)| scope.in_window(*c)),
      "窗口外的坐标装不上 GPU"
    );
    let mut last = 0;
    for (c, _) in pre {
      let d = (*c - scope.center).length_squared();
      assert!(d >= last, "预载按距离升序：{c:?}（{d}）在 {last} 之后");
      last = d;
    }
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
