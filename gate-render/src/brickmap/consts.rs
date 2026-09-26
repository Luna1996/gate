//! brickmap 旋钮（DDA + 上传预算）：改这里即改默认值。
//!
//! 全是编译期常量（改值后重新编译）。要给某项加 debug_menu 控件时，把它挪进 Bevy 资源
//! 再按菜单连线（做法见 gate-render 的 `RenderScale` / `EyeAdaptSettings`）。

/// 渲染目标（= 窗口）初始宽度，像素
pub const VIEW_W: u32 = 1280;
/// 渲染目标（= 窗口）初始高度，像素
pub const VIEW_H: u32 = 720;
/// beam 预 pass 的 workgroup 边长（设备上限内越大越快）
pub const WORKGROUP_SIZE: u32 = 8;
/// 轨道相机俯仰角上限（弧度，= 89°）
pub const PITCH_LIMIT: f32 = 89.0_f32.to_radians();
/// 轨道相机最近距离（体素）
pub const DIST_MIN: f32 = 32.0;
/// beam 深度纹理边长 = 全分辨率 ÷ 本值
pub const BEAM_DIV: u32 = 4;
/// 垂直视场角（弧度）；须与 gate-app `consts::FOV_Y` 一致
pub const DDA_FOV_Y: f32 = 60.0_f32.to_radians();
/// CPU 参考遍历器的步数上限（与 WGSL `trace_chunk` 同值）
pub const CPU_TRACE_BUDGET: u32 = 65536;

/// 诊断：chunk 步进（关掉 → trace_grid 在局部 slab 后直接 miss）
pub const DDA_CHUNKWALK: bool = true;
/// 诊断：只出天空（跳过全部 trace）
pub const DDA_SKY_ONLY: bool = false;
/// 诊断：只建 grid 不 trace
pub const DDA_MAKEGRID_ONLY: bool = false;
/// 八叉树远场早停（LOD）
pub const DDA_LOD: bool = true;
/// 叶级 LOD 诊断计数器的字数（`lod_diag`：叶入口 / 叶级早停 / 非法早停）。须与
/// `trace.wesl::DIAG_*` 的槽位对齐。
///
/// 开关不在这里：权威值是 `trace.wesl::LOD_DIAG`，Rust 经 [`crate::wesl_consts::trace_consts`]
/// 解析同一份源码（单一来源，见该函数的说明）。
pub const LOD_DIAG_WORDS: usize = 3;
/// **M4 ray-guided 请求通道**（`docs/editable-gigavoxel.md` §4 M4）：shader 把"射线想要细节、而那个
/// chunk 在 GPU 上没有树块"记成一条请求（chunk 相对窗口下标 + 所需档位 + 射线类型），Rust 侧按
/// `REPORT_PERIOD_SECS` 回读 + 合并排序后驱动流式加载。
///
/// 请求环缓冲槽数：新的覆盖旧的 ⇒ 回读只看到最近这么多条（**这不是"丢失"**：消费端也只读最新
/// `REQ_CAP` 条；真的丢要等"单窗口新增条数 > `REQ_CAP`"）。
/// **shader 侧不写死容量**：`trace.wesl::req_push` 用 `arrayLength(&lod_req)` 取绑定的实际长度 ⇒
/// 这里就是唯一来源。开关同样不在这里（权威值是 `trace.wesl::REQ_ENABLE`，经
/// [`crate::wesl_consts::trace_consts`] 解析）。
///
/// CONSTRAINT（**这个值就是"ray-guided 的信息量"**）：环是**最近性**采样，而"每条射线只报最前面的
/// 两个缺失 chunk"（`REQ_PER_RAY_MAX`）⇒ 最近的缺失块被反复报、把远处的挤掉。环太小就只剩"刚执行的
/// 那一小片屏幕"（实测 1024：合并后 8–27 个 chunk，全是眼前那一对）⇒ 消费端每窗只学到几十个"要装
/// 哪里"，加载既慢又不看视线。
/// 1 M 字（4 MB）= **整屏采样射线一帧的量**（约 `像素/16 × 2` 条事件）：环里装得下"整个画面这一帧
/// 想看什么" ⇒ 合并后给出上千个 chunk 的需求分布（消费端再按票数取前 [`REQ_FEED_MAX`] 条）。
/// 回读侧用**稠密 64³ 计数器**做合并（`profiler::report_lod_requests`：一次 1 M 遍历，无哈希）。
pub const REQ_CAP: usize = 1024 * 1024;
/// 回读**合并后**喂给消费端的条数上限（票数最高的前 N 条）。
/// 需求表只需要"最想要的那一批"：票数排序天然把近处/正对着的排在前面；不封顶会让
/// `LodRequestFeed` 带着上万个 chunk 进主线程（每帧拷一份 + 排序 = 几十 ms 的尖峰）。
///
/// 环里最多同时存几条**请求**。环只承载"缺了"这一类（`trace.wesl::req_push`）；"看见了"这类
/// 走下面的**用途戳表**（稠密，按构造不可能溢出）⇒ 这里给一个够宽的量就够（实测每窗口几十万条 →
/// 去重后 44–150 个 chunk，消费端按票数取前 [`REQ_FEED_MAX`] 条）。
pub const REQ_FEED_MAX: usize = 16 * 1024;
/// `lod_req` 头部字数：`[0]` 累计条数、`[1]` 保留、`[2]` 用途戳计数器，其后是用途戳表。
pub const USE_BASE: usize = 3;
/// **grid volume 数**（主世界 + 远场级）：权威在 `trace.wesl::GRID_VOLUMES`，Rust 经
/// [`crate::wesl_consts::trace_consts`] 解析同一份源码并**核对相等**（不等就 panic）。
/// 请求环与用途戳表按它定长 ⇒ 两侧不一致会越界写。
pub const VOLUMES: usize = 4;
/// 请求缓冲字数：`[0]` = 累计条数（只增，CPU 读差值配对 `REQ_CAP` 取模）、`[1]` = 保留、
/// `[2]` = 用途戳计数器、`[3 .. 3+USE_WORDS×VOLUMES)` = **逐 volume** 的用途戳表、
/// 其后 [`REQ_CAP`] 个请求字。
pub const LOD_REQ_WORDS: usize = 3 + USE_WORDS * VOLUMES + REQ_CAP;
/// **用途戳表**的格数 = 每 volume 一个窗口 chunk 数（64³）。
///
/// 论文 §III.A 的 usage stamp：`trace.wesl::req_use` 在遍历中对**已加载**的 chunk 记一笔
/// "这条主射线看到了它"（值 = 单调计数器发的戳，`atomicMax` 合并）。它是**缓存替换（LRU）唯一的
/// "最近使用"来源** —— 取代原先"每射线一条环记录"的可见性投票：后者会被约 1000 万条/秒的投票打爆
/// （实测环 100% 溢出 ⇒ 消费端读到的需求列表退化成 71 条垃圾），而稠密表**按构造不可能溢出**。
///
/// M8：表按 **volume** 分段（`[USE_BASE + vol*USE_WORDS, … + USE_WORDS)`）—— 远场级的 chunk 坐标
/// 是它自己的级体素空间，与主世界**数值上会撞**，不分段就互相顶掉。
pub const USE_WORDS: usize = 64 * 64 * 64;
/// **远场级池容量**（块/级，M8）：CPU 侧 LRU 的容量，**同时**决定远场 volume 的树区预留区大小
/// （见 [`FAR_RESERVE_WORDS_PER_CHUNK`]）。
///
/// 实测（用户机 `logs/latest.log`）：远场每次只在视锥的"走廊方向"上装几块~几十块（v1/v2/v3 各 10/35/30），
/// 取 2048 留了两个数量级的余量，而预留区仍是可接受的 16 MB/级（三级 48 MB）。
/// 这个数**不必**随「请求内存」滑杆涨：远场的数量由"射线真看到多少"定，不由预算定。
pub const FAR_POOL_CHUNKS: usize = 2048;
/// 远场级树区**每块的上界字数**：实测远场 chunk 的 wire = 1219–1463 字（4.9–5.9 KB，见
/// `far_detail_is_much_smaller_than_full`），加 `install_blob` 的 25% 余量与根预留 ⇒ 取
/// **2048 字（8 KB）/块**。
pub const FAR_RESERVE_WORDS_PER_CHUNK: usize = 2048;
/// 远场级 volume 的**树区预留字数**（每个远场 volume 固定这么多 ⇒ 它的 `b_struct` 长度**永不变**）。
///
/// WHY 必须预留（M8 首轮实跑暴露）：`VolumesBuilder::snapshot` 的 `bases_shifted` 以"各 volume 的
/// `b_struct` 长度"为判据，而**远场级排在主世界之前**（布局序 = 物体/远场在前、主世界最后）⇒
/// 远场一变长，主世界的 `tree_base` 就漂移 ⇒ 降级**全量快照**：把全部 volume 拼一遍。
/// 实测（`--features profile` + `GATE_LOG=gate=debug`）：
/// `UPLOAD[full]: bytes=577MB … elapsed=81–200ms`，每装几块远场就来一次（`extract` = 107–227 ms/帧）
/// ⇒ 帧率掉到个位数。这就是 §8 那条"物体变长的代价用**增长余量**摊薄"在远场级上的落地：
/// 预留一段够用满池的固定区 ⇒ 长度恒定 ⇒ 布局不漂移。用完（罕见）才真长一次。
pub const FAR_TREE_RESERVE_WORDS: usize = FAR_POOL_CHUNKS * FAR_RESERVE_WORDS_PER_CHUNK;

/// **索引条目的哨兵：这块已知没有内容**（`docs/mc_map.md` §8）。
///
/// `b_struct` 的 chunk 窗口条目编码是"本 volume 内树块首字址 + 1，0 = 无此 chunk"（`wire.rs` 头注释）。
/// `0` 同时表示"还没加载"与"是空的" —— 对 shader 是同一件事，所以射线在**永远装不上东西的空 chunk**
/// 上会一直发请求（`trace.wesl` 的 `entry == 0` 请求闸门）。
///
/// 无限世界不会遇到（每个 chunk 都有内容），而 MC 地图里**大多数 section 是空气**：实跑最热一块
/// **103 万票/窗**（`REQ[本窗口 6.8M 条、超容丢失 5.8M]`），请求环被这一块打满 ⇒ 真正的请求全被挤掉，
/// 且每秒几百万次原子写在同一个字上。
///
/// 于是让"已知空"成为索引里的一等状态：CPU 把**流式源产出过 `None`** 的 chunk 写成这个哨兵，
/// shader 见到它既不请求（`trace.wesl` 的请求闸门）也不遍历（按空气算）。
/// 取值 `u32::MAX`：合法条目是"树块首 + 1"，树区不可能有 40 亿字。
pub const INDEX_ENTRY_EMPTY: u32 = u32::MAX;

/// 请求环在 `lod_req` 里的起始字下标（`[0]` 累计条数 / `[1]` 保留 / `[2]` 用途戳计数器 /
/// `[3 .. 3+USE_WORDS×VOLUMES)` 逐 volume 用途戳表）。**必须与 `trace.wesl::REQ_BASE` 一致**。
pub const REQ_BASE: usize = USE_BASE + USE_WORDS * VOLUMES;
/// beam 预 pass（关掉则主 pass 从 t=0 起步）
pub const DDA_BEAM: bool = true;
/// 方向可达掩码剔除（LUT）
pub const DDA_DIR_LUT: bool = true;
/// 自动曝光初值（运行期以 Eye 页开关为准）
pub const EYE_ADAPT: bool = true;

/// 单次绑定字节上限（超过走多缓冲路径）
pub const SINGLE_THRESHOLD_BYTES: u64 = (1 << 30) - 1;
/// 每 chunk 上传字节预算（`max_bytes_per_frame` 按它折算 chunk 数）
pub const PER_CHUNK_BYTES: usize = 256 * 1024;
/// `UploadBudget` 缺省的每帧上传字节数
pub const UPLOAD_BYTES_PER_FRAME: usize = 4 * 1024 * 1024;
/// 多缓冲切片的目标字节数
pub const BUFFER_SLICE_TARGET: u64 = 64 << 20;
/// 多缓冲切片的下限字节数
pub const BUFFER_SLICE_MIN: u64 = 4 << 20;
/// 缓冲扩容阈值（字节）
pub const BUFFER_GROW_BIG: u64 = 8 << 20;
/// 缓冲扩容预留量（字节）
pub const BUFFER_GROW_RESERVE: u64 = 32 << 20;
/// brickmap 显存占用超过它就 warn（仅提示）
pub const VRAM_WARN_BYTES: u64 = 2 << 30;
