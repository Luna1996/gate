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

/// **在飞 + 待挂载的总量上限**（背压闸门）：worker 的产出速度远高于"挂载 → 标脏 → 上传"的消化速度
/// （实测粗档产出 **0.30 ms/chunk** ⇒ 3 个 worker 约 10 K chunk/s，而挂载预算只有 ~48 条/帧 ≈ 2.9 K/s）
/// ⇒ 不封顶的话 `ready` 会**无界增长**（每秒上万棵树、上百 MB 内存，worker 全在往一个只会变长的队列里
/// 灌 —— 表现为"一动就掉帧、越飞越卡"）。满仓就不再派发，把产能留给"预算调大"的时刻。
const READY_MAX: usize = 384;

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
  /// [`Self::ready`] 里已有的 chunk（派发去重用：`producer.in_flight` 只覆盖 worker 手上的）
  ready_set: std::collections::HashSet<ChunkCoord>,
  /// 已挂载 chunk 的档位（粗 / 细）：判"够不够细"（要细化就重新产出，挂载会整体替换）。
  /// 建世界时同步铺的起始块不在表里 —— 缺省按 [`Detail::Full`] 算（它们确实是全分辨率）。
  detail: std::collections::HashMap<ChunkCoord, Detail>,
  /// **LRU 的"最近使用"**（论文 §III.A 的 usage stamp）：chunk → 最近一次"被主射线看到"的批次序号
  /// （`ChunkUseFeed` 每 `REPORT_PERIOD_SECS` 换一批，`use_seq` 每批 +1）。
  ///
  /// 它取代了原来的"最后被请求时刻 + TTL"：那个键**只在"块缺了"时更新**，块一装好就冻结 ⇒ 看起来
  /// 永远最旧 ⇒ 被回收 ⇒ 再被请求 ⇒ 反复产出（且相机静止时世界也在变 ⇒ GI 时域永远接不上）。
  /// 用途戳反过来：**只要射线还在看它，键就一直涨** ⇒ 正在看的块永远排最后被回收。
  last_used: std::collections::HashMap<ChunkCoord, u64>,
  /// 用途批次的序号（每**换一批** `ChunkUseFeed` +1），就是 [`Self::last_used`] 的值域。
  use_seq: u64,
  /// 最近处理过的用途戳（判"有没有换新的一批"；同批反复处理会把"没有新信息"当成"又一轮最近"）。
  use_stamp: u32,
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
  pub mount_words: usize,
  /// 每帧挂载的条数上限（防"一堆极小树"把单帧的记账开销顶爆）
  pub mount_count: usize,
  /// **常驻池的内存预算**（字节）：交给渲染侧当 `ResidencyPolicy::budget_bytes`（超了就按 LRU 换出），
  /// 本模块的 CPU 侧保留量也按**同一个块数**封顶（[`gate_render::pool_capacity_chunks`]）。
  ///
  /// 口径是**块数**，且按 **CPU 侧**的每块字节折算（[`gate_render::pool_capacity_chunks`]）——
  /// 实测 CPU 侧约 **0.54 MB/chunk**（GPU 侧的 wire 只要约 0.30 MB），见 `CPU_CHUNK_BYTES_EST`。
  pub request_bytes: usize,
  /// **是否让请求圈真的装载**（关闭 ⇒ 回到"请求只在半径环内重排"的旧行为，便于 A/B）
  pub requests_load: bool,
  /// M5 生产管线：首次进入流式世界时惰性起（要 `n_pbr` 才能定槽号方案）
  pipeline: Option<std::sync::Mutex<Pipeline>>,
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
      request_bytes: 2 * 1024 * 1024 * 1024,
      requests_load: true,
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
  use_feed: Option<Res<gate_render::ChunkUseFeed>>,
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

  // ⓪' **GPU 常驻池预算**（论文 §III.A 的定长 pool）交给渲染侧：`plan_residency` 用它当
  //     `ResidencyPolicy::budget_bytes`。暂停时这条路不走 ⇒ 渲染侧沿用上一次的值（暂停的语义就是冻结）。
  scene.residency_budget_bytes = stream.request_bytes;

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
  let (request_bytes, requests_load) = (stream.request_bytes, stream.requests_load);
  let in_window = |c: IVec3| {
    let t = c - w_origin;
    t.cmpge(IVec3::ZERO).all() && t.cmplt(w_dims).all()
  };
  // **池容量**（chunk）：与渲染侧的 GPU 常驻池**同源换算**（[`gate_render::pool_capacity_chunks`]）
  // —— 两侧各按各的公式算就会出现"CPU 还留着 / GPU 已换出"（每帧重传）或反之（CPU 卸了 / GPU 还占着）。
  // `fill` = **还剩几个空位**（池容量 − 窗口内的 CPU 常驻）。它是半径补块的预算：池满 ⇒ `fill = 0`
  // ⇒ 不再按半径产出（否则就是"装一块、被 LRU 换掉、下帧再装"的空转）。**请求不受它限制** ——
  // 未命中换进一个槽位、同时挤出一个 LRU 槽位，池满时正是最该放行的。
  let cap = gate_render::pool_capacity_chunks(request_bytes);
  let fill = cap.saturating_sub(grid.chunk_coords().filter(|c| in_window(c.0)).count());
  let fill = DEMAND_TABLE_MAX.min(fill);

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
      ready_set: Default::default(),
      detail: Default::default(),
      last_used: Default::default(),
      use_seq: 0,
      use_stamp: 0,
      empty: Default::default(),
    }));
  }
  let mut generated = 0usize;
  {
    let mut pipe = stream
      .pipeline
      .as_ref()
      .expect("上面刚装上")
      .lock()
      .unwrap_or_else(|e| e.into_inner());
    let mut gen_req = 0usize;
    // ⓪'' **用途戳 → LRU**（论文 §III.A）：每**换一批**（回读侧每 `REPORT_PERIOD_SECS` 换一批）就把
    //      `use_seq` +1，批内每个 chunk 记成"本批次被看到过"。`last_used` 的值域就是这个批次序号 ⇒
    //      **单调、可比大小**，正是 `②卸载` 需要的 LRU 键。
    //      `peek` 每帧都返回**同一份快照**（GPU 侧也要读它）⇒ 用批内最大戳判"是不是新的一批"，
    //      否则每帧推进一次批次号 = 把"没有新信息"当成"又一轮最近"。
    if let Some(f) = use_feed.as_ref() {
      let batch = f.peek();
      let stamp = batch.iter().map(|u| u.stamp).max().unwrap_or(0);
      if stamp != 0 && stamp != pipe.use_stamp {
        pipe.use_stamp = stamp;
        pipe.use_seq += 1;
        let seq = pipe.use_seq;
        for u in &batch {
          pipe.last_used.insert(ChunkCoord(u.chunk), seq);
        }
      }
    }
    pipe.demand_age += 1;
    if pipe.demand_center != Some(center) || pipe.demand_age >= DEMAND_REBUILD_FRAMES {
      // 请求：**票数决定装哪里**（`plan_generation` 用它排序），**档位由射线报回**（`LodRequest::level`
      // ⇒ 论文的 "refinement 由渲染结果给"）。请求圈**不受半径环约束** —— 这是 ray-guided 的落点。
      let requests: Vec<gate_render::LodRequest> = if requests_load {
        feed.as_ref().map(|f| f.peek()).unwrap_or_default()
      } else {
        Vec::new()
      };
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
        fill,
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
      // 背压（见 `READY_MAX`）：待挂载 + 在飞已经够多就停 —— 否则 worker 会把产出灌进一个无界队列
      if pipe.ready.len() + pipe.producer.inflight() >= READY_MAX {
        break;
      }
      if pipe.producer.in_flight(ChunkCoord(c)) {
        pipe.demand.pop_front();
        continue;
      }
      // **已在待挂载队列里的也算"已经在飞"**：`in_flight` 只覆盖 worker 手上的，产出取回后它就不再为真
      // ⇒ 不挡的话同一块会被反复重产、反复重挂（`chunk_count` 不动、`gen` 每帧都在涨、
      // 而 `ready` 顶在 `READY_MAX` 把真正的新块堵在后面）。
      if pipe.ready_set.contains(&ChunkCoord(c)) {
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
        // 同一块的在飞重复（派发修好前遗留的）⇒ 留先到的那个，别再堆一份
        Some(_) if !pipe.ready_set.insert(cc) => {}
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
      pipe.ready_set.remove(&cc);
      grid.mount_chunk_tree(cc, tree, 0);
      pipe.detail.insert(cc, detail);
      // **装载即使用**（与 GPU 侧的 `note_resident` 同一个口径）：请求带进来的块在射线眼里
      // 还是"缺的"（射线没进过它 ⇒ 没有用途戳），若不在这里记一笔，它一挂上就被 LRU 判成最旧
      // ⇒ 立刻换出 ⇒ 下个窗口又请求同一块（装卸死循环）。
      pipe.use_seq += 1;
      let seq = pipe.use_seq;
      pipe.last_used.insert(cc, seq);
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
    // `REPORT_PERIOD_SECS` 才来一批，头两秒没有任何信号。
    //
    // CONSTRAINT：这里只卸 **CPU 侧**（`VolumeGrid`）。GPU 侧的换出由渲染侧的 `plan_residency`
    // 独立决定（同一个容量、同一个 `last_used` 信号）；两侧都卸同一批是正常的收敛结果，不是重复劳动。
    let resident: Vec<ChunkCoord> = grid.chunk_coords().collect();
    let pending: std::collections::HashSet<ChunkCoord> =
      pipe.ready.iter().map(|(cc, ..)| *cc).collect();
    let mut far: Vec<ChunkCoord> = Vec::new();
    let mut cand: Vec<(u64, ChunkCoord)> = Vec::new();
    for c in &resident {
      if !in_window(c.0) {
        far.push(*c);
        continue;
      }
      if (c.0 - center).abs().max_element() <= unload_r
        || pending.contains(c)
        || pipe.producer.in_flight(*c)
      {
        continue;
      }
      cand.push((pipe.last_used.get(c).copied().unwrap_or(0), *c));
    }
    // 超容量 ⇒ 卸掉"最久没被看到"的那些（`last_used` 升序 ⇒ LRU 在前）
    let over = resident.len().saturating_sub(far.len()).saturating_sub(cap);
    if over > 0 {
      cand.sort_unstable_by_key(|(t, c)| (*t, c.0.x, c.0.y, c.0.z));
      far.extend(cand.into_iter().take(over).map(|(_, c)| c));
    }
    let unloaded = far.len();
    for cc in far {
      grid.unmount_chunk(cc);
      pipe.detail.remove(&cc);
      pipe.last_used.remove(&cc);
    }
    if generated > 0 || unloaded > 0 || !pipe.ready.is_empty() {
      bevy::log::debug!(
        "STREAM[gen {generated}(req {gen_req}) unload {unloaded} chunks {} cap {cap} ready {}]",
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
fn detail_of_req(level: u8) -> Detail {
  match level {
    0 => Detail::Full,
    1 => Detail::Coarse,
    2 => Detail::Wide,
    _ => Detail::Chunk,
  }
}

/// 本帧该产出哪些 chunk（[`stream_chunks`] 第 ① 步的策略，抽成纯函数以便单测）：
/// **请求优先**（票数多的先，M4），再用半径补块（**视野优先**，见下），合计 ≤ `want`。
///
/// **请求那一侧不受半径环约束**（这是"ray-guided"的落点）：只要落在**窗口内**就能装 —— 窗口是
/// `b_struct` 索引区的定义域，也是请求圈唯一的硬边界（±32 chunk ≈ ±164 m）。
///
/// **档位跟请求走**（论文的口径："refinement 由渲染结果给"）：请求自带的 `level` 就是射线算出来的
/// "这个 chunk 需要多细"，直接用它（[`detail_of_req`]）。旧版这里一律按距离给档（`detail_at`）⇒
/// 只要 64³ 的远处大块也被升到全分辨率（278 KB/chunk，同一份内存少装 20 倍），且档位随相机微小
/// 移动跳变 ⇒ **细化 → 几何内容变 → 已在收敛的 GI 时域永远接不上**。距离判据（`detail_of`）只剩
/// **半径补块**那一侧在用 —— 那一路没有射线信息，只能按距离。
///
/// **排序**（半径启发式最容易搞砸的一点）：`load_radius` 内永远最优先（脚下与眼前，缺了就掉进
/// 虚空），请求那一路按票数，其余先给**视野锥内**的、再按距离。只按距离排的话，帧额会被平均撒到
/// 相机身后 —— 那就完全没有"ray-guided"的感觉。
///
/// **请求不占 `fill`**（这是"缓存恒满"的前提）：`fill` 是**池里还剩几个空位**（口径 = 池容量 −
/// 窗口内常驻），只用来限**半径补块**。未命中必须无条件放行 —— 它换进的是一个槽位、同时挤出一个
/// LRU 槽位；若也按 `fill` 截，池一满就再也补不上缺口。反过来，池满时半径补块必须**停**：继续
/// 按半径产出就是"装一块、马上被 LRU 换掉、下帧再装"的空转（实测 `ready` 队列顶在 `READY_MAX`
/// 不降、每帧白挂 3.7 个 chunk）。
///
/// 过滤（都要过）：
/// - **窗口内**：窗口是 `b_struct` 索引区的定义域，窗口外产出也装不上 GPU；
/// - **够不够细**由 `have(c, want)` 回答（已挂载且档位 ≥ 要的档 ⇒ 不算需求；粗档在近处会因此被细化）。
///
/// 返回 `(本帧的 (chunk, 档位) 列表, 其中来自请求的条数)`。
fn plan_generation(
  scope: GenScope,
  fill: usize,
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
  let mut picked: Vec<(IVec3, Detail)> = Vec::new();
  let mut requested: Vec<gate_render::LodRequest> =
    requests.iter().filter(|r| in_window(r.chunk)).copied().collect();
  requested.sort_unstable_by_key(|r| (std::cmp::Reverse(r.votes), r.chunk.x, r.chunk.y, r.chunk.z));
  picked.extend(
    requested
      .into_iter()
      .filter(|r| !have(r.chunk, detail_of_req(r.level)))
      .map(|r| (r.chunk, detail_of_req(r.level))),
  );
  let from_req = picked.len();
  if fill > 0 {
    // 请求那一批可能上千条（`REQ_FEED_MAX`）⇒ 去重走集合，别在候选循环里线性扫
    let taken: std::collections::HashSet<IVec3> = picked.iter().map(|(c, _)| *c).collect();
    let mut todo: Vec<(IVec3, Detail)> = Vec::new();
    for dx in -coarse_radius..=coarse_radius {
      for dz in -coarse_radius..=coarse_radius {
        for dy in -coarse_height..=coarse_height {
          let c = center + IVec3::new(dx, dy, dz);
          let d = detail_of(c);
          if in_window(c) && !have(c, d) && !taken.contains(&c) {
            todo.push((c, d));
          }
        }
      }
    }
    todo.sort_unstable_by_key(|(c, _)| key(*c));
    picked.extend(todo.into_iter().take(fill));
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
        if let Some(tree) = src.produce(cc, detail, &mut scratch) {
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
        if let Some(tree) = src.produce(cc, detail, &mut scratch) {
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
          if src.produce(cc, Detail::Coarse, &mut scratch).is_none() {
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

  /// M4 + M8：**请求优先于半径补块**（票数多的先）；**请求不受半径环约束**（只要在窗口内就装，
  /// 这是"ray-guided"的落点），但窗口外 / 已够细的请求仍要丢。
  /// **档位由请求自带**（`LodRequest::level`，论文口径 "refinement 由渲染结果给"）：请求不只决定
  /// **装哪里**，多细也由它一起报回来（[`detail_of_req`]）。没有请求时逐字退回"半径 + 近的先 + 距离定档"。
  #[test]
  fn requests_outrank_radius_fill() {
    let req = |votes: u32, c: IVec3, level: u8| gate_render::LodRequest { chunk: c, votes, level };
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
    // `fill = 0`（池满）：**只放请求**、一条半径补块都不要 —— 未命中换进一个槽位、同时挤出一个
    // LRU 槽位，池满时正是最该放行的；而半径补块此时产出就是空转。
    let (batch, from_req) = plan_generation(scope, 0, &reqs, have);
    assert_eq!(
      batch,
      vec![
        (IVec3::new(5, 0, 0), Detail::Full),
        (IVec3::new(20, 0, 0), Detail::Coarse),
        (IVec3::new(60, 0, 0), Detail::Wide),
        (IVec3::new(300, 0, 0), Detail::Chunk),
        (IVec3::new(1000, 0, 0), Detail::Chunk),
      ],
      "票数序、档位逐字取请求报回的；已挂载的 (1,0,0) 不算需求"
    );
    assert_eq!(from_req, 5);
    // 有空位（`fill = 2`）：请求之后按半径近的先补 2 个
    let (batch, from_req) = plan_generation(scope, 2, &reqs, have);
    assert_eq!(from_req, 5);
    assert_eq!(batch.len(), 7);
    assert_eq!(batch[5], (IVec3::ZERO, Detail::Full));
    assert_eq!(batch[6], (IVec3::new(-1, -1, -1), Detail::Full));
    // 请求圈**只在窗口内**：窗口半宽 8 ⇒ (9,0,0) 在窗外，无论多少票都不装（空位转给半径补块）
    let narrow = gen_scope(8, 1, 3, 3);
    let (batch, from_req) = plan_generation(narrow, 1, &[req(999, IVec3::new(9, 0, 0), 0)], have);
    assert_eq!(from_req, 0, "窗口外的请求不该装：{batch:?}");
    assert_eq!(batch, vec![(IVec3::ZERO, Detail::Full)]);
    // 无请求（请求装载关掉）⇒ 逐字退回半径启发式
    let (batch, from_req) = plan_generation(narrow, 3, &[], have);
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
    // 请求档位是**四档**（射线分不出 Full 与 Fine —— 2 cm 与 8 cm 落在同一像素内）
    for (level, want) in
      [(0u8, Detail::Full), (1, Detail::Coarse), (2, Detail::Wide), (3, Detail::Chunk)]
    {
      assert_eq!(detail_of_req(level), want, "level {level} 的档位映射");
    }
    // 半径内同样按判据给档（`load_radius` 只管优先级），半径外一个都不要
    let (batch, _) = plan_generation(gen_scope(20, 1, 6, 6), 400, &[], |_, _| false);
    assert!(batch.iter().all(|(c, d)| *d == Detail::Full), "6 chunk = 31 m 内都还没到 8 cm 档");
    assert!(batch.iter().all(|(c, _)| c.abs().max_element() <= 6), "半径外不该要");
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
