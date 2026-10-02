import package::bindings::{b_leaves, b_palette, b_struct, grid_descs, lod_diag, lod_req, occ, view_u};
// TLAS 只在 RT 版存在（`bindings.wesl` 里那条声明也带 `@if(ray_query)`）⇒ import 必须同步条件化。
@if(ray_query)
import package::bindings::{tlas};
import package::brickmap::{Grid};
import package::common::{CHUNK_INDEX_CAP, CHUNK_SIZE, DIR_LUT, FLAGS_IS_PBR, FLAGS_TRANSMISSIVE, INDEX_ENTRY_EMPTY, NODE_FIXED_WORDS, face_normal_from_index};

// 层级命中记录：t 为 ro 系绝对 t；face_id 0..5 = ±xyz 六面（命中面法线索引）。
// voxel = 命中固体体素 chunk 局部整数坐标，着色阶段直接消费。
struct VoxelHit {
  hit: bool,
  t: f32,
  pal: u32,
  face_id: u32,
  voxel: vec3<i32>,
}

// 介质穿透（透明/玻璃）：透射材质按可穿过介质处理，在 DDA 内部一路走完。
// `stop_on_air = true`：遇空气停在出射面；false：空气也穿过（阴影射线）。
//
// ---- **吸收口径：逐"介质段"一次** ----
// 一整段**连续的同一材质**介质 = 一个半透明片：进入该段时 `tint *= 该材质的透过率`（**一次**），
// 之后这段里跨多少格都不再衰减 ⇒ `tint` 只由"穿过了几段什么材质"决定，**与厚度/格数/距离无关**。
// 落到用户手里就是「透明度」= **混合比**（100% = 全透、0% = 不透明），滑杆整段可用、且没有
// "厚度图"式的假投影（代价：厚玻璃不会更暗）。
// `last_pal` = 当前介质段材质（0 = 不在介质内）：既是"是否在介质段内"的判据（`medium_exit`），
// 也是"**这一段是否已经染过**"的记忆键 —— 同材质仍在继续 ⇒ 不重复衰减（见 `pass_through_medium`）；
// 离开介质（撞空气）时调用方清 0 ⇒ 之后再遇到同材质算**新的一段**（两块玻璃各染一次）。
// 字段一改需同步 `world.wesl` 的构造点。
struct MediumRay {
  stop_on_air: bool,
  palette_base: u32,
  last_pal: u32,
  tint: vec3<f32>,
}

// brick 缓存槽：一层分裂节点常驻（镜像 CPU `BrickCpu`）。`mask` = 64 位分裂掩码（wire 的两个
// word 拼成），语义同 `sample_brickmap`。
struct Brick {
  addr: u32,  // 节点绝对字址（b_struct）
  mask: u64,  // 分裂掩码（bit=1 = 子块分裂；bit=0 = 统一子块，色 = pal）
  pal: u32,   // 节点 palette（统一子块颜色，0 = 空气）
  // 节点 palette word 的高 16 位 = **叶块代表值**（M2，写侧 `chunk_tree::leaf_rep_palette` 的
  // 加权众数；非叶节点恒 0）。只有 `level == 0`（4³ 值块）的 brick 会用它，见 `leaf_lod_pal`。
  pal_rep: u32,
}

/// 子块 `idx`（0..63）是否分裂。
/// 与下一函数一起**保持函数形式**：内联进调用点会算错（槽下标错 ⇒ 下钻到错的子节点）。
fn mask_bit(mask: u64, idx: u32) -> bool {
  return (mask & (u64(1) << idx)) != u64(0);
}

/// `idx` 之前（不含 idx）置位的分裂位个数 = 该子块的槽下标（与 `sample_brickmap` 同式）。
fn mask_pop_below(mask: u64, idx: u32) -> u32 {
  let below = mask & ((u64(1) << idx) - u64(1));
  return u32(countOneBits(below));
}

/// 叶级 LOD 的像素阈值，取 `fp = t·px_ang`：`1/fp` = 一个体素在屏幕上占多少像素。
/// `fp >= 1` ⇔ 一个像素已经覆盖 ≥ 1 个体素 ⇒ 一个 4³ 值块约 4 像素宽，块内 1 体素尺度的细节
/// 已不可分辨 ⇒ 整块取一个颜色（代表色 = `leaf_lod_pal`：M2 存好的**块内加权众数**槽号，
/// 写侧 `chunk_tree::leaf_rep_palette`）。
///
/// CONSTRAINT: 块内结构被**整体替换**，轮廓误差可达整块（4 像素）—— 这是"一个块一个色"的固有
/// 代价。M2 已把"代表色取哪一粒体素"换成众数（块内多数材质），残余误差 = 少数派的那部分体素
/// 被染成多数派的色。
///
/// WARNING: **M2 之后这条早停才真正开始生效**（此前 `leaf_lod_pal` 几乎恒返回 0，`fp = 1` 等于没
/// 打开），于是近中景整片变成「一个 4³ 值块一个色」⇒ 斜视的墙面 / 地面表现为**沿格行的长条**，
/// 与逐体素着色的既定口径冲突。见 docs/editable-gigavoxel.md §10.8 的同机位实拍 A/B（关掉它
/// 画面即与 da18260 基线一致）。故**取 `3.4e38`（f32 上限）= 关闭**：
///   · 关闭的代价 = 总 GPU +8.1%、重帧 p90 +28%（`fp >= 1` 相对关闭的实测收益，见 §9）；
///   · 要保留这份收益就把阈值推到远场（`fp >= 4` ⇔ 3 740 体素≈935 m），**不要**调回 1。
const LEAF_LOD_FP: f32 = 3.4e38;

/// 代表色扫描的置位上限（**成本护栏**，不是语义）：扫不到就返回 0 = 放弃本块的 LOD，
/// 调用点退回逐格 march（只是慢，不会漏命中）。
const LEAF_LOD_SCAN_CAP: u32 = 8u;

/// **诊断开关（M0）**：置 `1u` ⇒ 统计叶级 LOD 的覆盖与颜色契约违规，写进 `bindings::lod_diag`。
/// 编译期常量 ⇒ 置 0 时整段被折叠掉，正常路径零开销（与 `DIAG_DISABLE_MEDIUM_TINT` 同款）。
/// 这里是**唯一来源**：Rust 侧经 `wesl_consts::trace_consts` 解析同一份源码，按它决定要不要注册回读
/// （不再各抄一份 `bool`，也就没有"两侧须同时改"的约定）。
/// 采样闸门 = 叶节点地址 `& 63`：只统计 1/64 的叶入口（全量计数会有 ~10M/帧 的单地址原子加，
/// 把收益本身吃掉）；分子分母用**同一个**闸门 ⇒ 比值无偏。
const LOD_DIAG: u32 = 0u;
/// `lod_diag` 槽位：0 = 采样到的叶入口数（本来要逐格 march 的那些），1 = 其中被叶级 LOD 拦下的，
/// 2 = 非法早停（`gd >= depth_cap` 且本节点 `pal == 0` —— 那不是该子块的颜色）。
/// 3..8 = **主 pass 逐面查表的覆盖计数**（`main.wesl::dda_main` 的仪表，槽位常量在那边）：
/// 精确命中 / 覆盖 texel 回退命中 / 回退不可用（无键或异面或异法线或异材质） / 回退槽失配 / 内联着色。
const DIAG_LEAF_ENTRIES: u32 = 0u;
const DIAG_LEAF_STOPS: u32 = 1u;
const DIAG_ILLEGAL_STOPS: u32 = 2u;
/// 诊断采样闸门：叶节点地址低位为零的叶才计数（≈1/64）。`LOD_DIAG = 0` 时恒 false ⇒ 整个 `if` 被折叠。
fn diag_sampled(addr: u32) -> bool {
  return LOD_DIAG == 1u && (addr & 63u) == 0u;
}

// ============================================================================
// M4 · ray-guided 请求通道（`docs/editable-gigavoxel.md` §4 M4）
// ============================================================================

/// 请求通道开关（**已关闭**）：置 `1u` ⇒ 射线撞上"**窗口里有槽位、GPU 上还没有树块**"的 chunk 时记
/// 一条请求（`bindings::lod_req`），Rust 侧回读 + 合并排序后驱动流式加载。
///
/// REF: ray-guided 流式已整条取消 —— 常驻的需求 / 换出改由**相机距离**给（`residency::nearest_top`），
/// 见 `upload::plan_residency` 的函数头。这里恒 0：`req_push` / `req_use` 整段被折叠，正常路径零开销，
/// `lod_req` 那张表也不再被写（`wesl_consts::trace_consts` 读到 0 ⇒ 不注册回读）。
///
/// CONSTRAINT：置回 `1u` 之前先把 `plan_residency` 的需求集换回"由渲染结果给"，否则常驻会同时被
/// 两套信号驱动。
const REQ_ENABLE: u32 = 0u;
/// 请求**采样率倒数**：约每 `REQ_SAMPLE` 条射线只有 1 条发请求。
/// 请求是"需求的采样"，不是逐像素民意调查 —— 全量会变成每秒数亿次原子写（实测 4.8 亿/窗口），
/// GPU 全耗在 `lod_req[0]` 这一个字上，帧率崩到个位数 ⇒ 表现为"加载停了"。
const REQ_SAMPLE: u32 = 16u;
/// 每条射线最多记几条（一条射线连续穿过很多缺失 chunk 时只记最前面的几条）
const REQ_PER_RAY_MAX: u32 = 2u;
/// 缓冲字下标（`[0]` 累计条数、`[1]` 保留、`[2]` 用途戳计数器、`[3 .. 3+USE_WORDS)` 用途戳表、
/// 其后是环）。**容量不在这里写死**：`req_push` 用 `arrayLength(&lod_req)` 取绑定的实际长度 ⇒ 与 Rust 的
/// `consts::LOD_REQ_WORDS` **同源**，不会出现两边常量不一致（那会导致越界写 + 解包全垃圾）。
const REQ_COUNT: u32 = 0u;
/// `[1]` **保留**：原先这里是"溢出计数"（`n >= cap` 就 +1），但 `n` 是**累计**的 ⇒ 累计条数一旦超过
/// 环容量，它每窗口都恰好等于窗口条数，看着像"100% 溢出"其实一条都没丢（消费端只读最新 `cap` 条）。
/// "这一窗口有没有丢"改由消费端按 `REQ_CAP` 判（见 `profiler::report_lod_requests`）。
const REQ_RESERVED: u32 = 1u;
/// 请求字里 chunk 相对窗口下标的每轴位数（窗口 = 64³ chunk ⇒ 6 位）
const REQ_REL_BITS: u32 = 6u;
/// 只报"需要 ≤ 这个档位"的请求（0 = 全分辨率 4³、1 = 16³）：更远的（64³ / 整 chunk 档）连一个平色
/// 都够 ⇒ 不值得为它加载。
const REQ_LEVEL_KEEP: u32 = 1u;

/// **grid volume 数**（主世界 + 远场级；M8）：`grid_descs` 的条目数上限，也是请求字里"卷号"的位宽
/// （2 bit ⇒ 至多 4 个）。请求环与用途戳表按它分段 ⇒ **Rust 侧 [`brickmap::consts::VOLUMES`] 必须与它
/// 逐字相等**（`wesl_consts::trace_consts` 会核对，不等就 panic）。
/// `brickmap::consts::{LOD_REQ_WORDS, REQ_BASE}` 由它算出长度与分段起点。
const GRID_VOLUMES: u32 = 4u;

// ---------------------------------------------------------------------------
// 用途戳（usage stamp，论文 §III.A）：**"哪些 chunk 现在真的被看"的唯一来源**。
//
// 为什么要有它：论文的缓存替换靠"最近使用"（LRU），而本仓原先把"最后被**请求**时刻"当 LRU 键 ——
// 那个信号只在"块缺了"时才更新，块一装好就冻结 ⇒ 看起来永远是最旧的 ⇒ 被换出 ⇒ 再被请求 ⇒
// 反复产出。用途戳反过来：**只要射线还在看它，戳就一直被刷新**，所以正在看的东西永远排最后。
//
// 与请求环共用同一块绑定（前缀区）⇒ 不加 binding、不改任何 layout；表是**稠密**的（窗口内每 chunk
// 一个 u32）⇒ **按构造不可能溢出**（这正是把"看见了"从环记录改成戳的原因：环记录会被 1000 万条/秒
// 的可见性投票打爆，而 100% 溢出时消费端读到的就是垃圾）。
//
// 戳值由 `USE_TICK` 计数器发（`atomicAdd`）⇒ 单调、与 CPU 帧号无关、**不需要清零**：CPU 侧只认
// "(上次读数, 本次读数] 区间内的戳"，陈旧条目天然落在区间外。
//
// M8：表按 **volume 分段**（每段 `USE_WORDS` 格，段号 = `Grid::vol`）—— 远场级的 chunk 坐标是它自己的
// 级体素空间（与主世界**数值上会撞**），不分段就会互相顶掉。
const USE_WORDS: u32 = 64u * 64u * 64u;
const USE_TICK: u32 = 2u;
const USE_BASE: u32 = 3u;
const REQ_BASE: u32 = 3u + GRID_VOLUMES * USE_WORDS;


/// 该距离"够用的档位"下标：0 = 全分辨率（4³）、1 = 16³、2 = 64³、3 = 整 chunk（256³）。
/// 阶梯与 Rust `residency::want_level` **同一口径**（`fp ≥ 4/16/64` ⇒ 16³/64³/整 chunk）——
/// 消费端按同一个阶梯决定装到哪一档，两边不分叉。
fn req_level(fp: f32) -> u32 {
  if (fp >= 64.0) { return 3u; }
  if (fp >= 16.0) { return 2u; }
  if (fp >= 4.0) { return 1u; }
  return 0u;
}

/// 采样判定：拿射线本身做哈希（同一条射线稳定命中同一结论，不闪烁）⇒ 约 `1/REQ_SAMPLE` 的射线发请求。
/// 用射线而不是像素下标：`trace_grid` 没有像素 id（GI / 光柱 / 玻璃状态机都走它）。
fn req_hash(origin: vec3<f32>, dir: vec3<f32>) -> u32 {
  return u32(bitcast<i32>(origin.x * 64.0)) ^ (u32(bitcast<i32>(origin.y * 64.0)) << 11u)
    ^ (u32(bitcast<i32>(origin.z * 64.0)) << 21u) ^ u32(bitcast<i32>(dir.z * 4096.0));
}

/// 远场级的请求**采样率倒数**：远场只需要"哪一带要看"的粗信号（它的窗口里几乎全是"缺了"），
/// 而请求字的追加是**单地址原子加**（`lod_req[0]`）⇒ 用主世界那个 1/16 会把 GPU 全烧在它上面。
/// 实测（`GATE_LOG=info` 的首窗 `REQ[...]`，相机刚进世界、三级远场全空）：远场用 1/16 时首窗
/// **1200 万条**（≈ 100k 次/帧的单地址原子加，与 `docs/editable-gigavoxel.md` §9 M4 记的
/// "4.8 亿条 ⇒ 帧率崩到个位数"同一形态）；1/64 把这一波按 4× 压小，而合并后的 chunk 数不变
/// （样本更少、分布不变 ⇒ 该装哪里照旧看得出来）。
const REQ_SAMPLE_FAR: u32 = 64u;

/// 按卷的采样判定：主世界 `REQ_SAMPLE`、远场级 `REQ_SAMPLE_FAR`（见 [`REQ_SAMPLE_FAR`]）。
/// 采样率与用途戳共用本判据（两者都是每射线一次的原子写，远场同样是"粗信号就够"）。
fn req_sampled(origin: vec3<f32>, dir: vec3<f32>, vol: u32) -> bool {
  let m = select(REQ_SAMPLE, REQ_SAMPLE_FAR, vol > 0u);
  return (req_hash(origin, dir) % m) == 0u;
}

/// 记一条请求。`rel` = chunk 相对窗口下标，`level` = [`req_level`] 的下标，`kind` = 0 主射线 / 1 次级
/// （位留给消费端排序；当前只有主射线会走到这里），`vol` = 哪个 volume（0 = 主世界，≥1 = 远场级）。
///
/// 环缓冲追加：`n = atomicAdd([0])` ⇒ 写 `[REQ_BASE + n % cap]`。环满即**覆盖最旧的**：消费端只读最新
/// 的 `cap` 条（见 `REQ_RESERVED` 的说明）。**容量取自绑定的实际长度**（见 `REQ_COUNT` 的说明）。
/// **不去重、不排序**：那是消费端的事。
///
/// CONSTRAINT（M8）：`rel` 是**各 volume 自己的窗口**里的相对下标 —— 卷号必须随字一起走，否则远场级
/// 的 chunk 会被消费端当成主世界的坐标（两个空间数值上会撞）。
fn req_push(rel: vec3<u32>, level: u32, kind: u32, vol: u32) {
  if (REQ_ENABLE == 0u) { return; }
  let cap = arrayLength(&lod_req) - REQ_BASE;
  let n = atomicAdd(&lod_req[REQ_COUNT], 1u);
  let w = (rel.x & 63u)
    | ((rel.y & 63u) << REQ_REL_BITS)
    | ((rel.z & 63u) << (REQ_REL_BITS * 2u))
    | ((level & 3u) << 18u)
    | ((kind & 1u) << 20u)
    | ((vol & 3u) << 21u);
  atomicStore(&lod_req[REQ_BASE + (n % cap)], w);
}

/// 记一条**用途戳**："这条主射线看到了 `vol` 的 `rel` 那个 chunk"。只喂缓存替换（LRU），**不进需求表**。
///
/// 戳 = `USE_TICK` 计数器的当前值（`atomicAdd` 取）⇒ 单调递增；表项用 `atomicMax` ⇒ 同一窗口内
/// 任意多条射线并发写只留最新的一笔。窗口相对下标 3×6 位 ⇒ 偏移落在
/// `[USE_BASE + vol·USE_WORDS, …+USE_WORDS)` 之内（每 volume `USE_WORDS` 恰好是窗口 chunk 数 64³）
/// ⇒ 与请求环互不重叠。
fn req_use(rel: vec3<u32>, vol: u32) {
  if (REQ_ENABLE == 0u) { return; }
  let i = (rel.x & 63u)
    | ((rel.y & 63u) << REQ_REL_BITS)
    | ((rel.z & 63u) << (REQ_REL_BITS * 2u));
  let t = atomicAdd(&lod_req[USE_TICK], 1u);
  atomicMax(&lod_req[USE_BASE + (vol & 3u) * USE_WORDS + i], t);
}

/// 叶级 LOD 代表色（M2）：节点 palette word 高 16 位存的就是它 —— 写侧算的是**块内按体素数加权的
/// 众数**槽号（`chunk_tree::leaf_rep_palette`），即"这块看起来最像的那个材质"。
///
/// 只在"块内混合"（`mask != 0`）且**入口体素是空气**时被调用（入口是实体时上游已精确返回）。
/// 字段为 0（旧数据 / 全空气）时退回旧口径：块内**首个非空体素**色 —— 掩码位置位只表示"该格不取
/// `pal`"，它的 inline 半字仍可能是 0（空气）⇒ 不能只取最低置位，必须逐位置位扫到第一个非零值；
/// `pal` 只在**存在未置位格**（`mask != ~u64(0)`）时才是块内真实存在的颜色（掩码满时它是"无未置位格"
/// 的残留值，见 `chunk_tree.rs::Leaf::tile_color`）。
/// 成本：常见路径 1 次字段读；退回扫描时最坏 `LEAF_LOD_SCAN_CAP` 次（一次扫描替掉的是块内最多
/// 约 12 次逐格查）。
fn leaf_lod_pal(b: Brick) -> u32 {
  if (b.pal_rep != 0u) { return b.pal_rep; }
  if (b.mask != ~u64(0) && b.pal != 0u) { return b.pal; }
  var m = b.mask;
  for (var k: u32 = 0u; k < LEAF_LOD_SCAN_CAP; k = k + 1u) {
    if (m == u64(0)) { return 0u; }
    // 最低置位下标：`low - 1` 的置位数 = 它下面有多少位（u64 无 firstTrailingBit 重载）
    let i = u32(countOneBits(m & (u64(0) - m)) - u64(1));
    let w = b_struct[b.addr + NODE_FIXED_WORDS + (i >> 1u)];
    let v = (w >> ((i & 1u) * 16u)) & 0xFFFFu;
    if (v != 0u) { return v; }
    m = m & (m - u64(1));
  }
  return 0u;
}

/// 读 palette 槽的介质属性：`rgb` = **该材质的"一段"透过率**（乘性；一段 = 连续的同一材质，
/// 由 `pass_through_medium` 在**进入该段时乘一次** —— 见 `MediumRay` 的头注释），
/// `a` = 不透明度（`a >= 1` = 不透明、**不是介质**；调用方只用它做这一个判定）。
/// 布局与 `wire.rs::pack_palette_entry` 一致（一次读两个 word）。
/// **介质判定读 `flags` 的 `TRANSMISSIVE` 位**（bit5）而不是 `transmission` 字节：本函数在 DDA 内
/// **逐格**调用，且 PBR 变体里 bit8..15 已是 `asset` 的一部分（读它得不到透射率）。
/// flags 与 transmission 同在 word1 ⇒ 零额外读取；该位由写入侧维护（见 `wire.rs::pack_palette_entry`）。
///
/// **为什么这里直接读 palette、不走 `fetch_material`**（硬约束 1 的唯一入口规则）：本函数在 DDA 热路径上
/// 调用，而 `fetch_material` 的 PBR 分支要 triplanar 采样（6 次纹理采样）= 逐格做不起；
/// 与 `common.wesl::material_ior`（"只读 palette + 资产表、不采样贴图"，同一个成本理由）同一条例外。
/// 这里取的是**介质透过率**（DDA 的物理量），不是着色 —— 着色仍然只有 `brdf_reflected` 一条路径。
fn medium_of(palette_base: u32, pal: u32) -> vec4<f32> {
  let w1 = b_palette[palette_base + pal * 2u + 1u];
  // `FLAGS_*` 是**字节级掩码**（配 `common.wesl::palette_flags` 的 `>> 16`）：flags 落在 word1 的
  // bit16..23 ⇒ 先取出字节再比。
  let flags = (w1 >> 16u) & 0xFFu;
  // 非介质：调用方只消费 `a >= 1.0` 这一判据 ⇒ 直接给 a = 1.0。
  if ((flags & FLAGS_TRANSMISSIVE) == 0u) { return vec4<f32>(1.0, 1.0, 1.0, 1.0); }
  if ((flags & FLAGS_IS_PBR) != 0u) {
    // PBR 变体：w1 低 16 位是 `asset`，读不出透射率 ⇒ 只按 TRANSMISSIVE 位判"是介质"，
    // 单格透过率取中性（白 = 不吸收）。吸收色待补：走材质资产要多一次资产表 load，逐格不合算。
    return vec4<f32>(1.0, 1.0, 1.0, 0.0);
  }
  let w0 = b_palette[palette_base + pal * 2u];
  let srgb = vec3<f32>(f32(w0 & 0xFFu), f32((w0 >> 8u) & 0xFFu), f32((w0 >> 16u) & 0xFFu)) / 255.0;
  let a = 1.0 - f32((w1 >> 8u) & 0xFFu) / 255.0;
  // 一段透过率 = **消色差密度衰减 × 颜色吸收**：
  //   · `1 − a` = 密度衰减（"不透明度"的本体）：白/浅色材质（`srgb ≈ 白`）靠它变不透；
  //   · `mix(白, srgb², a)` = 颜色吸收（彩色玻璃随不透明度加深而更出色），乘上 `1 − a`
  //     ⇒ 暗色材质在"不透明度 = 0"时不发黑；
  //   · `srgb²` = palette 色是 sRGB 编码字节，平方是廉价 γ≈2 线性化（`material_ior` 那条同源近似）。
  return vec4<f32>((1.0 - a) * mix(vec3<f32>(1.0), srgb * srgb, a), a);
}

/// **诊断开关（临时 A/B 用，默认必须 0u）**：`1u` ⇒ 介质吸收**整体关闭**（每段不再染 `tint`，
/// 只剩界面反射的乘性衰减）。人工验证配方：
///   · 置 `1u` 后画面上的暗区/条纹**消失** ⇒ 那些来自**吸收**（`medium_of` 的透过率太低）；
///   · 仍**在** ⇒ 与吸收无关，去查几何侧：`main.wesl::GLASS_MAX_BOUNCE`（状态机的多界面/全反射路径）、
///     `main.wesl::PBR_REFLECTION_ENABLED`（MT4 反射射线）、以及 `dda_main` 的 in-scatter 项。
/// 编译期常量 ⇒ `0u` 时整个分支被折叠掉，正常路径零开销（WESL 运行期读盘编译：改这一行 + 重启即生效）。
const DIAG_DISABLE_MEDIUM_TINT: u32 = 0u;

/// 可穿透介质 → 返回 true（调用方按介质继续步进）；不可穿透（`a >= 1`，含 `transmission == 0`）返回 false。
/// 只在 `penetrate = true` 时被调用 —— 而 `penetrate = true` 只有一条入口链
/// （`trace_glass`（主 pass 的玻璃状态机）与 `gi/ray.wesl` 的 `sun_transmittance` / 二次射线
/// 都经 `world_trace_medium` → `trace_grid` → 本函数）⇒ 硬约束 3「口径不分叉」由**只有这一份实现**保证。
///
/// **吸收在"材质变化"这一刻结算一次**（逐段 alpha，见 `MediumRay` 的头注释）：
///   · 同材质仍在继续（`pal == last_pal`）⇒ **立即返回**：一次比较、连 palette 读都省掉 ——
///     厚块不会因为"跨了很多格"而多付衰减；
///   · 材质变了（含从空气进入）⇒ 取一次材质参数，把该材质的透过率乘进 `tint`（**整段只这一次**）。
/// 离开介质（撞空气）由调用方清 `last_pal = 0` ⇒ 之后再遇到同材质算**新的一段**。
fn pass_through_medium(mc: ptr<function, MediumRay>, pal: u32) -> bool {
  if (pal == (*mc).last_pal) { return true; }
  let m = medium_of((*mc).palette_base, pal);
  if (m.a >= 1.0) { return false; }
  if (DIAG_DISABLE_MEDIUM_TINT == 0u) { (*mc).tint = (*mc).tint * m.rgb; }
  (*mc).last_pal = pal;
  return true;
}

/// 介质段内撞到空气 = 出射面：返回 true（调用方带回 `last_pal`），只在 `penetrate = true` 时被调用。
fn medium_exit(mc: ptr<function, MediumRay>) -> bool {
  return (*mc).stop_on_air && ((*mc).last_pal != 0u);
}

/// **射线级不变量**（只由本 volume 局部系的 `rd` / `sign_v` 决定）：同一条射线**每次进入 chunk**
/// （`trace_chunk_opaque` / `trace_chunk`）都要用到的量。
///
/// WHY 必须传参而不能就地重算：这两个函数是**每条射线每个 chunk 进入一次**，而其中 `1.0/rd` 是
/// **三次除法**、`oct` / `dir_side` / `face_base` 是三组 `select` —— 同一条射线进入十几次就重算十几次。
/// 而 `trace_grid` 本来就已经算出**同一批值**（`grid_inv_rd` / `grid_axis_on` / `DIR_LUT` 的 `side` 与
/// 三轴 face id / `dir_mask`）⇒ 传参是**纯等式变换**：同样的输入、同样的运算、同样的顺序，
/// 只是从"每次进入算一遍"变成"每条射线每个 volume 算一遍"（位精确：`1.0/rd` 的两次求值逐位相同，
/// `DIR_LUT` 的 side/face 与 `select(...)` 逐位等价，见 `common.wesl::DIR_LUT`）。
struct RayBasis {
  /// `1.0 / rd`（chunk 局部系；与 `trace_grid::grid_inv_rd` 是同一个值）
  inv_rd: vec3<f32>,
  /// `abs(inv_rd)`
  abs_inv_rd: vec3<f32>,
  /// `abs(rd) > 1e-30`（= `trace_grid::grid_axis_on`）
  axis_on: vec3<bool>,
  /// 每轴步进所取的那一侧边界（0 = 取基址、1 = 取基址 + 边长）= `DIR_LUT` 的 side
  dir_side: vec3<f32>,
  /// 每轴两个面的 face id 基（0/1、2/3、4/5 的**低位**）= `DIR_LUT` 的 face_x/y/z
  face_base: vec3<u32>,
  /// 方向位掩码（`DIR_LUT` 的下标；= 处处那个 `oct`）
  oct: u32,
}

// 单 chunk 内层级遍历（不透明版）：chunk_base = 根节点绝对字址，chunk_min = chunk 原点（局部 voxel）。
// [t0,t1] 与返回 t 为 ro 系绝对 t；entry_face = 进入面；depth_cap 下 gate 深度 d=3-level 达上限时返回子块入口 t（保守下界）。
// `basis` = 本射线的**不变量**（见 `RayBasis` 的说明：调用方算一次，这里只消费）。
fn trace_chunk_opaque(chunk_base: u32, chunk_min: vec3<f32>,
                      ro: vec3<f32>, rd: vec3<f32>, sign_v: vec3<i32>, basis: RayBasis,
                      t0: f32, t1: f32, entry_face: u32, depth_cap: u32) -> VoxelHit {
  if (t0 >= t1) { return VoxelHit(false, 0.0, 0u, 0u, vec3<i32>(0)); }
  // chunk 局部 voxel 坐标（chunk 原点 = 0）；t 仍是 ro 系绝对 t
  let ro_c = ro - chunk_min;
  let inv_rd = basis.inv_rd;
  let abs_inv_rd = basis.abs_inv_rd;
  let axis_on = basis.axis_on;
  let dir_side = basis.dir_side;
  let face_base = basis.face_base;
  let oct = basis.oct;
  var bricks: array<Brick, 4u>;
  let r_ml = b_struct[chunk_base];
  let r_mh = b_struct[chunk_base + 1u];
  let r_pw = b_struct[chunk_base + 2u];
  let r_pal = r_pw & 0xFFFFu;  // 节点字低 16 位 = uniform 色（16 位材质索引）
  let r_rep = r_pw >> 16u;     // 高 16 位 = 叶代表值（M2；根是层 3 节点 ⇒ 恒 0，带上只为口径完整）
  bricks[3u] = Brick(chunk_base, (u64(r_mh) << 32u) | u64(r_ml), r_pal, r_rep);
  var level: u32 = 3u;
  // 当前体素（chunk 局部 voxel 整数坐标 0..255；跨出 chunk 的步进瞬态可达 -1/256）
  let p0 = ro_c + rd * t0;
  var v = clamp(vec3<i32>(floor(p0)), vec3<i32>(0), vec3<i32>(255));
  var cur_t = t0;
  var face = entry_face;
  // 防挂死安全网：迭代上限。
  var budget: u32 = 65536u;
  let lut_disable = view_u.lod.w > 0.5;
  // 叶级 LOD：`lod.y` = `consts::DDA_LOD`；`px_ang` = 单像素角大小（rad，见 `lod` 字段说明）。
  let leaf_lod = view_u.lod.y > 0.5;
  let px_ang = view_u.lod.x;
  loop {
    if (budget == 0u) { break; }
    budget = budget - 1u;
    if (level > 3u) { return VoxelHit(false, 0.0, 0u, 0u, vec3<i32>(0)); }
    // ---- traverse：从当前 level 下钻到 v 处内容 ----
    loop {
      let b = bricks[level];
      // WGSL 移位 RHS 必须 u32
      let sh = vec3<u32>(level * 2u);
      let cell = vec3<i32>(v >> sh) & vec3<i32>(3);
      let idx = u32(cell.z * 16 + cell.y * 4 + cell.x);
      if (level == 0u) {
        // 叶节点：bit=1 的体素取 inline 半字；bit=0 的体素 = 本节点 uniform 色（0 = 空气）。
        // 口径同 `sample_brickmap` / `get_voxel`。
        if (mask_bit(b.mask, idx)) {
          let w = b_struct[b.addr + NODE_FIXED_WORDS + (idx >> 1u)];
          let leaf_pal = (w >> ((idx & 1u) * 16u)) & 0xFFFFu;
          if (leaf_pal != 0u) { return VoxelHit(true, cur_t, leaf_pal, face, v); }
        } else if (b.pal != 0u) {
          return VoxelHit(true, cur_t, b.pal, face, v);
        }
        // 入口体素是空气 ⇒ 下面是"叶级 LOD 早停 / 退回逐格 march"的分界：诊断计数放这里
        // （分母 = 本来要逐格 march 的叶入口数，分子 = 其中被 LOD 拦下的）。
        let diag = diag_sampled(b.addr);
        if (diag) { atomicAdd(&lod_diag[DIAG_LEAF_ENTRIES], 1u); }
        // 叶级 LOD：整块不足一个像素时不再逐格走，直接取块内首个非空体素色
        // （t 恒取块入口 = 保守下界，见 `LEAF_LOD_FP`）。
        if (leaf_lod && b.mask != u64(0) && cur_t * px_ang >= LEAF_LOD_FP) {
          let p = leaf_lod_pal(b);
          if (p != 0u) {
            if (diag) { atomicAdd(&lod_diag[DIAG_LEAF_STOPS], 1u); }
            return VoxelHit(true, cur_t, p, face, v);
          }
        }
        break;
      }
      if (!mask_bit(b.mask, idx)) {
        // 统一子块：色 = 节点 palette（0 = 空气）
        if (b.pal != 0u) { return VoxelHit(true, cur_t, b.pal, face, v); }
        break;
      }
      // depth_cap（beam 保守）：gate 深度 d=3-level 的分裂子块到达 cap → 子块入口 t
      // CONSTRAINT: 这里返回的 `pal` 是**本节点**的 uniform 色，而当前子块是**分裂**的 ⇒ 它不保证是
      // 该子块的颜色（通常 0 = 空气）。因此这条早停只对"只消费 t"的调用者合法（beam 预 pass）；
      // 着色射线必须走 `depth_cap >= 3`（不早停）。上层三档 LOD 要启用须先有**非叶**节点的代表值
      // （M2 只给了叶块，M2b 尚未落地；docs/editable-gigavoxel.md 硬不变量 4）。
      let gd = 3u - level;
      if (gd >= depth_cap) {
        // 颜色契约的机器可校验形式（M0）：`pal == 0` = "返回了不属于该子块的颜色"最露骨的情形。
        if (LOD_DIAG == 1u && b.pal == 0u && (b.addr & 63u) == 0u) {
          atomicAdd(&lod_diag[DIAG_ILLEGAL_STOPS], 1u);
        }
        return VoxelHit(true, cur_t, b.pal, face, v);
      }
      let pop = mask_pop_below(b.mask, idx);
      let child_addr = chunk_base + b_struct[b.addr + NODE_FIXED_WORDS + pop];
      let c_ml = b_struct[child_addr];
      let c_mh = b_struct[child_addr + 1u];
      let c_pw = b_struct[child_addr + 2u];
      let c_pal = c_pw & 0xFFFFu;  // 节点字低 16 位 = uniform 色
      let c_rep = c_pw >> 16u;     // 高 16 位 = 叶代表值（M2；只有层 0 的值块非 0）
      let c_mask = (u64(c_mh) << 32u) | u64(c_ml);
      // 统一子节点快路径：wire 任意层的分裂位都可能指向 3 字统一节点（mask=0，禁读 addr+3 之后）
      if (c_mask == u64(0)) {
        if (c_pal != 0u) { return VoxelHit(true, cur_t, c_pal, face, v); }
        break;
      }
      level = level - 1u;
      bricks[level] = Brick(child_addr, c_mask, c_pal, c_rep);
    }
    // ---- 整砖 LUT 跳过（仅 palette==0 节点安全）----
    {
      let b = bricks[level];
      if (b.pal == 0u) {
        let sh = vec3<u32>(level * 2u);
        let ecell = vec3<i32>(v >> sh) & vec3<i32>(3);
        let entry_i = u32(ecell.z * 16 + ecell.y * 4 + ecell.x);
        let lut_base = min(oct * 64u + entry_i, 511u);
        // `b_leaves` 每项 1 个 u64（Rust 侧按 2 个 u32 写入，低字在前）
        var reach = b_leaves[lut_base];
        reach = select(reach, ~u64(0), lut_disable);
        if ((b.mask & reach) == u64(0)) {
          level = level + 1u;
          if (level > 3u) { return VoxelHit(false, 0.0, 0u, 0u, vec3<i32>(0)); }
        }
      }
    }
    // ---- v 处为空气：当前 level brick 内 DDA ----
    let log2 = level * 2u;
    let s = 1i << (level * 2u); // 子块边长 voxel：1/4/16/64
    let s_mask = ~(s - 1i);
    // side_distance_for_ray：v 对齐到 s 的基址；正向 → 基址+s，负向 → 基址
    let base_v = v & vec3<i32>(s_mask);
    let boundary = vec3<f32>(base_v) + dir_side * f32(s);
    var side = vec3<f32>(1e+30);
    side = select(side, (boundary - ro_c) * inv_rd, axis_on);
    side = max(side, vec3<f32>(cur_t));
    let step_inc = f32(s) * abs_inv_rd;
    var step_axis: u32 = 0u;
    var changed = false;
    // 内层步进的循环不变量（数组读 + 三个向量化转换）提到循环外
    let b = bricks[level];
    let sh2 = vec3<u32>(log2);
    let sm_mask = vec3<i32>(s_mask);
    let sm1 = vec3<i32>(s - 1);
    loop {
      var mn: u32;
      if (side.x <= side.y && side.x <= side.z) { mn = 0u; }
      else if (side.y <= side.z) { mn = 1u; }
      else { mn = 2u; }
      step_axis = mn;
      cur_t = side[mn];
      if (cur_t >= t1) { return VoxelHit(false, 0.0, 0u, 0u, vec3<i32>(0)); }
      let old_cell = (v[mn] >> log2) & 3;
      // 步进即对齐：v 恒 = `cur_t` 处的真实体素（本循环内的命中直接上报它）。格基址每步从 v 重算；
      // 步进轴取新格边界，其余轴按射线位置钳在本格内。
      let cell_min = v & sm_mask;
      v = clamp(vec3<i32>(floor(ro_c + rd * cur_t)), cell_min, cell_min + sm1);
      v[mn] = select(cell_min[mn] + s, cell_min[mn] - 1i, sign_v[mn] < 0);
      side[mn] = side[mn] + step_inc[mn];
      face = mn * 2u + face_base[mn];
      let crossed = select(old_cell == 0, old_cell == 3, sign_v[mn] >= 0);
      if (crossed) { changed = true; break; }
      let cell = vec3<i32>(v >> sh2) & vec3<i32>(3);
      let idx = u32(cell.z * 16 + cell.y * 4 + cell.x);
      if (level == 0u) {
        // 叶层同上：bit=1 取 inline；否则（含 mask==0 的整节点 uniform）= 本节点色。
        if (mask_bit(b.mask, idx)) {
          let w = b_struct[b.addr + NODE_FIXED_WORDS + (idx >> 1u)];
          let dp = (w >> ((idx & 1u) * 16u)) & 0xFFFFu;
          if (dp != 0u) { return VoxelHit(true, cur_t, dp, face, v); }
        } else if (b.pal != 0u) {
          return VoxelHit(true, cur_t, b.pal, face, v);
        }
      } else {
        let mb = mask_bit(b.mask, idx);
        if (mb) { break; }
        if (b.pal != 0u) { return VoxelHit(true, cur_t, b.pal, face, v); }
      }
    }
    if (changed) {
      level = level + 1u;
      if (level > 3u) { return VoxelHit(false, 0.0, 0u, 0u, vec3<i32>(0)); }
    }
    // ---- firstTrailingBit 层级自适应跨级跳 ----
    // 步进轴新坐标的尾随零位 = 对齐 run 长度，tz >> 1 = 可跨步的最粗 level。
    let positive = sign_v[step_axis] >= 0;
    let cur_log2 = level * 2u;
    let m: u32 = 0xFFFFFFFFu << cur_log2;
    let not_m = ~m;
    let vmin_u = u32(v[step_axis]);
    var comp: u32;
    if (positive) { comp = vmin_u & m; } else { comp = (vmin_u & m) | not_m; }
    let new_level = firstTrailingBit(comp + select(1u, 0u, positive)) >> 1u;
    level = max(level, new_level);
    if (level > 3u) { return VoxelHit(false, 0.0, 0u, 0u, vec3<i32>(0)); }
    // 对齐快照：v 钳到 cur_t 处当前 level 区域，步进轴取精确边界整数。
    let mi = i32(m);
    let base = v & vec3<i32>(mi);
    let p = ro_c + rd * cur_t;
    let region_max = base + vec3<i32>(i32(not_m));
    v = clamp(vec3<i32>(floor(p)), base, region_max);
    v[step_axis] = i32(comp);
  }
  return VoxelHit(false, 0.0, 0u, 0u, vec3<i32>(0));
}

// 单 chunk 内层级遍历（介质穿透版）：调用点恒传穿透语义，无 `penetrate` 开关；mc = 介质穿透上下文。
// 命中 / 穿透 / 出射面判定只在两处唯一出口做。
// **介质吸收**（逐段 alpha，见 `MediumRay` 的头注释）：`tint` 只在**进入一个新材质段**时乘一次
// （`pass_through_medium`），本函数**不做任何距离/入口 t 的记账**，也没有 chunk 边界收尾。
// `basis` = 本射线的不变量（同 `trace_chunk_opaque`）。
fn trace_chunk(chunk_base: u32, chunk_min: vec3<f32>,
               ro: vec3<f32>, rd: vec3<f32>, sign_v: vec3<i32>, basis: RayBasis,
               t0: f32, t1: f32, entry_face: u32, depth_cap: u32,
               mc: ptr<function, MediumRay>) -> VoxelHit {
  if (t0 >= t1) { return VoxelHit(false, 0.0, 0u, 0u, vec3<i32>(0)); }
  // chunk 局部 voxel 坐标（chunk 原点 = 0）；t 仍是 ro 系绝对 t
  let ro_c = ro - chunk_min;
  let inv_rd = basis.inv_rd;
  let abs_inv_rd = basis.abs_inv_rd;
  let axis_on = basis.axis_on;
  let dir_side = basis.dir_side;
  let face_base = basis.face_base;
  let oct = basis.oct;
  var bricks: array<Brick, 4u>;
  let r_ml = b_struct[chunk_base];
  let r_mh = b_struct[chunk_base + 1u];
  let r_pw = b_struct[chunk_base + 2u];
  let r_pal = r_pw & 0xFFFFu;  // 节点字低 16 位 = uniform 色（16 位材质索引）
  let r_rep = r_pw >> 16u;     // 高 16 位 = 叶代表值（M2；根是层 3 节点 ⇒ 恒 0，带上只为口径完整）
  bricks[3u] = Brick(chunk_base, (u64(r_mh) << 32u) | u64(r_ml), r_pal, r_rep);
  var level: u32 = 3u;
  // 当前体素（chunk 局部 voxel 整数坐标 0..255；跨出 chunk 的步进瞬态可达 -1/256）
  let p0 = ro_c + rd * t0;
  var v = clamp(vec3<i32>(floor(p0)), vec3<i32>(0), vec3<i32>(255));
  var cur_t = t0;
  var face = entry_face;
  // 防挂死安全网：迭代上限。
  var budget: u32 = 65536u;
  let lut_disable = view_u.lod.w > 0.5;
  // 叶级 LOD：与 `trace_chunk_opaque` 同一口径（`lod.y` 开关 + `LEAF_LOD_FP` 阈值）。
  let leaf_lod = view_u.lod.y > 0.5;
  let px_ang = view_u.lod.x;
  loop {
    if (budget == 0u) { break; }
    budget = budget - 1u;
    if (level > 3u) { return VoxelHit(false, 0.0, 0u, 0u, vec3<i32>(0)); }
    // ---- traverse：从当前 level 下钻到 v 处内容 ----
    var hit_pal = 0u;
    loop {
      let b = bricks[level];
      // WGSL 移位 RHS 必须 u32
      let sh = vec3<u32>(level * 2u);
      let cell = vec3<i32>(v >> sh) & vec3<i32>(3);
      let idx = u32(cell.z * 16 + cell.y * 4 + cell.x);
      if (level == 0u) {
        // 叶节点：bit=1 取 inline 半字；bit=0 的体素 = 本节点色（口径见 `trace_chunk_opaque` 的叶层注释）。
        if (mask_bit(b.mask, idx)) {
          let w = b_struct[b.addr + NODE_FIXED_WORDS + (idx >> 1u)];
          hit_pal = (w >> ((idx & 1u) * 16u)) & 0xFFFFu;
        } else {
          hit_pal = b.pal;
        }
        // 入口体素是空气 ⇒ 叶级 LOD（同 `trace_chunk_opaque`，含同一套诊断计数）。走 `hit_pal`
        // 而不是直接 return：代表色若是可穿透介质，下面"唯一出口"仍要按介质段语义处理（穿透 / 出射面）。
        let diag = diag_sampled(b.addr);
        if (diag) { atomicAdd(&lod_diag[DIAG_LEAF_ENTRIES], 1u); }
        if (hit_pal == 0u && leaf_lod && b.mask != u64(0) && cur_t * px_ang >= LEAF_LOD_FP) {
          let p = leaf_lod_pal(b);
          if (p != 0u) {
            if (diag) { atomicAdd(&lod_diag[DIAG_LEAF_STOPS], 1u); }
            hit_pal = p;
          }
        }
        break;
      }
      if (!mask_bit(b.mask, idx)) {
        // 统一子块：色 = 节点 palette（0 = 空气）
        hit_pal = b.pal;
        break;
      }
      // depth_cap（beam 保守）：gate 深度 d=3-level 的分裂子块到达 cap → 子块入口 t
      // CONSTRAINT: 同 `trace_chunk_opaque` 的同名分支 —— 返回的 `pal` 不带当前分裂子块的颜色语义，
      // 只有"只消费 t"的调用者可用（docs/editable-gigavoxel.md 硬不变量 4）。
      let gd = 3u - level;
      if (gd >= depth_cap) {
        // 颜色契约的机器可校验形式（M0）：`pal == 0` = "返回了不属于该子块的颜色"最露骨的情形。
        if (LOD_DIAG == 1u && b.pal == 0u && (b.addr & 63u) == 0u) {
          atomicAdd(&lod_diag[DIAG_ILLEGAL_STOPS], 1u);
        }
        return VoxelHit(true, cur_t, b.pal, face, v);
      }
      let pop = mask_pop_below(b.mask, idx);
      let child_addr = chunk_base + b_struct[b.addr + NODE_FIXED_WORDS + pop];
      let c_ml = b_struct[child_addr];
      let c_mh = b_struct[child_addr + 1u];
      let c_pw = b_struct[child_addr + 2u];
      let c_pal = c_pw & 0xFFFFu;  // 节点字低 16 位 = uniform 色
      let c_rep = c_pw >> 16u;     // 高 16 位 = 叶代表值（M2；只有层 0 的值块非 0）
      let c_mask = (u64(c_mh) << 32u) | u64(c_ml);
      // 统一子节点快路径：wire 任意层的分裂位都可能指向 3 字统一节点（mask=0，禁读 addr+3 之后）
      if (c_mask == u64(0)) {
        hit_pal = c_pal;
        break;
      }
      level = level - 1u;
      bricks[level] = Brick(child_addr, c_mask, c_pal, c_rep);
    }
    // ---- traverse 唯一出口：命中 / 穿透 / 出射面 ----
    if (hit_pal != 0u) {
      if (!pass_through_medium(mc, hit_pal)) {
        return VoxelHit(true, cur_t, hit_pal, face, v);
      }
    } else {
      // 空气格 = 本段介质的终点：判"是否到了出射面"，并把 `last_pal` 清 0
      // （离开介质 ⇒ 之后再遇到同材质算**新的一段**，各染一次）。
      if (medium_exit(mc)) { return VoxelHit(true, cur_t, (*mc).last_pal, face, v); }
      (*mc).last_pal = 0u;
    }
    // ---- 整砖 LUT 跳过（仅 palette==0 节点安全）----
    {
      let b = bricks[level];
      if (b.pal == 0u) {
        let sh = vec3<u32>(level * 2u);
        let ecell = vec3<i32>(v >> sh) & vec3<i32>(3);
        let entry_i = u32(ecell.z * 16 + ecell.y * 4 + ecell.x);
        let lut_base = min(oct * 64u + entry_i, 511u);
        // LUT 读法同 `trace_chunk_opaque`
        var reach = b_leaves[lut_base];
        reach = select(reach, ~u64(0), lut_disable);
        if ((b.mask & reach) == u64(0)) {
          level = level + 1u;
          if (level > 3u) { return VoxelHit(false, 0.0, 0u, 0u, vec3<i32>(0)); }
        }
      }
    }
    // ---- v 处为空气：当前 level brick 内 DDA ----
    let log2 = level * 2u;
    let s = 1i << (level * 2u); // 子块边长 voxel：1/4/16/64
    let s_mask = ~(s - 1i);
    // side_distance_for_ray：v 对齐到 s 的基址；正向 → 基址+s，负向 → 基址
    let base_v = v & vec3<i32>(s_mask);
    let boundary = vec3<f32>(base_v) + dir_side * f32(s);
    var side = vec3<f32>(1e+30);
    side = select(side, (boundary - ro_c) * inv_rd, axis_on);
    side = max(side, vec3<f32>(cur_t));
    let step_inc = f32(s) * abs_inv_rd;
    var step_axis: u32 = 0u;
    var changed = false;
    // 内层步进的循环不变量（数组读 + 三个向量化转换）提到循环外
    let b = bricks[level];
    let sh2 = vec3<u32>(log2);
    let sm_mask = vec3<i32>(s_mask);
    let sm1 = vec3<i32>(s - 1);
    loop {
      var mn: u32;
      if (side.x <= side.y && side.x <= side.z) { mn = 0u; }
      else if (side.y <= side.z) { mn = 1u; }
      else { mn = 2u; }
      step_axis = mn;
      cur_t = side[mn];
      if (cur_t >= t1) {
        // 越出本 chunk ⇒ miss（吸收与距离无关，无需边界收尾）。`last_pal` 留着 ⇒
        // 下一个 chunk 接着走时，同材质跨 chunk 仍是同一段（不重复衰减）。
        return VoxelHit(false, 0.0, 0u, 0u, vec3<i32>(0));
      }
      let old_cell = (v[mn] >> log2) & 3;
      // 步进即对齐（同 `trace_chunk_opaque`）。
      let cell_min = v & sm_mask;
      v = clamp(vec3<i32>(floor(ro_c + rd * cur_t)), cell_min, cell_min + sm1);
      v[mn] = select(cell_min[mn] + s, cell_min[mn] - 1i, sign_v[mn] < 0);
      side[mn] = side[mn] + step_inc[mn];
      face = mn * 2u + face_base[mn];
      let crossed = select(old_cell == 0, old_cell == 3, sign_v[mn] >= 0);
      if (crossed) { changed = true; break; }
      let cell = vec3<i32>(v >> sh2) & vec3<i32>(3);
      let idx = u32(cell.z * 16 + cell.y * 4 + cell.x);
      hit_pal = 0u;
      if (level == 0u) {
        if (mask_bit(b.mask, idx)) {
          let w = b_struct[b.addr + NODE_FIXED_WORDS + (idx >> 1u)];
          hit_pal = (w >> ((idx & 1u) * 16u)) & 0xFFFFu;
        } else {
          hit_pal = b.pal;  // 未分裂体素（含整节点 uniform）= 本节点色
        }
      } else {
        if (mask_bit(b.mask, idx)) { break; }
        hit_pal = b.pal;
      }
      if (hit_pal != 0u) {
        if (pass_through_medium(mc, hit_pal)) { continue; }
        return VoxelHit(true, cur_t, hit_pal, face, v);
      }
      // 空气格：与 traverse 出口同一处理 —— 判出射面，并把 `last_pal` 清 0（本段介质到此结束）。
      if (medium_exit(mc)) { return VoxelHit(true, cur_t, (*mc).last_pal, face, v); }
      (*mc).last_pal = 0u;
    }
    if (changed) {
      level = level + 1u;
      if (level > 3u) { return VoxelHit(false, 0.0, 0u, 0u, vec3<i32>(0)); }
    }
    // ---- firstTrailingBit 层级自适应跨级跳 ----
    // 步进轴新坐标的尾随零位 = 对齐 run 长度，tz>>1 = 可跨步的最粗 level。
    let positive = sign_v[step_axis] >= 0;
    let cur_log2 = level * 2u;
    let m: u32 = 0xFFFFFFFFu << cur_log2;
    let not_m = ~m;
    let vmin_u = u32(v[step_axis]);
    var comp: u32;
    if (positive) { comp = vmin_u & m; } else { comp = (vmin_u & m) | not_m; }
    let new_level = firstTrailingBit(comp + select(1u, 0u, positive)) >> 1u;
    level = max(level, new_level);
    if (level > 3u) { return VoxelHit(false, 0.0, 0u, 0u, vec3<i32>(0)); }
    // 对齐快照：v 钳到 cur_t 处当前 level 区域，步进轴取精确边界整数。
    let mi = i32(m);
    let base = v & vec3<i32>(mi);
    let p = ro_c + rd * cur_t;
    let region_max = base + vec3<i32>(i32(not_m));
    v = clamp(vec3<i32>(floor(p)), base, region_max);
    v[step_axis] = i32(comp);
  }
  return VoxelHit(false, 0.0, 0u, 0u, vec3<i32>(0));
}

// 统一 DDA：主网格与逐物体共用一条管线（严格镜像 Rust `brickmap/dda.rs`）：slab→chunk 间 A&W→
// trace_chunk，差异由 `Grid` 参数化。

// 统一命中结构：hit / t / pal / n（世界空间法线）/ face_id（0..5）/ obj_id（-1 = 主世界，>=0 = 物体）。
// voxel 为命中固体体素 grid 局部整数坐标（主世界 = 世界坐标），由 DDA 整数步进精确产出。
struct UnifiedHit {
  hit: bool,
  t: f32,
  pal: u32,
  n: vec3<f32>,        // 世界空间法线（光影用）
  face_id: u32,        // 命中面 0..5（编码见 `face_normal_from_index`）
  voxel: vec3<i32>,    // grid 局部命中体素坐标
  obj_id: i32,         // -1 = 主世界, >=0 = 物体索引
}

/// chunk 级命中 → grid 级命中：面法线换到世界系，体素坐标补 chunk 偏移（`ci` = 命中 chunk）。
fn grid_hit(h: VoxelHit, ci: vec3<i32>,
            col0: vec3<f32>, col1: vec3<f32>, col2: vec3<f32>, obj_id: i32) -> UnifiedHit {
  let n_local = face_normal_from_index(h.face_id);
  let n_world = normalize(n_local.x * col0 + n_local.y * col1 + n_local.z * col2);
  let voxel = h.voxel + ci * i32(CHUNK_SIZE);
  return UnifiedHit(true, h.t, h.pal, n_world, h.face_id, voxel, obj_id);
}

/// **GI 候选射线的行程上限**（chunk）：GI 只需要**局部**信息，而窗口是 ±32 chunk ⇒ 它在"窗口内但
/// 没加载"的空间里一路走空。出界按"没几何"处理即可（调用方落回天空）。
///
/// CONSTRAINT: 这个夹取**只属于 GI 候选射线**，由调用点把它折进 `t_cap`（`gi/ray.wesl`）——
/// **不能**再放在 `trace_grid` 里按 `req` 一刀切：**阴影**与**反射**要在场景尺度上成立（本仓 MC
/// 城市的水面倒影在百米级、楼影同样），用 GI 的 12 chunk 去夹会把它们截断在 61 m；而从前那种夹法
/// 还是**轴对齐的盒子**（`w_mn/w_mx`）⇒ 在平整的反射面 / 受影面上切出一条**轴对齐、带直角**的边界
/// （用户截图"水面上一个明显的矩形的一个角、色差很严重"就是它：倒影只在每个水面点周围 61 m 的方盒里）。
/// 现在 `trace_grid` 只受调用方给的 `t_cap` 与**窗口 AABB**约束 ⇒ 次级射线的可见范围**与主射线一致**，
/// 代价也以主射线为界（阴影射线现在要走到窗口边，实测代价是每帧 1–3 ms 量级；要收回就把它自己的
/// `t_cap` 夹小 —— 那是"阴影能看多远"的旋钮，不再是全局一刀切）。
///
/// 实测（2026-09-25）：GI 行程 12 → 1 让 `gate_gi` 16.2 → 6.2 ms —— 代价**随行程线性**，这是 GI
/// 的两个真杠杆之一（另一个是候选数）。见 `docs/editable-gigavoxel.md` 的「附二」。
const SECONDARY_REACH_CHUNKS: f32 = 12.0;

// ============================================================================
// 空空间跳过（粗占用位图，`bindings::occ`）
// ============================================================================

/// 开关（诊断）：置 `0u` ⇒ 退回原路径（**两者语义逐位等价**，见 `trace_grid` 里的 CONSTRAINT）。
const OCC_SKIP: u32 = 1u;
/// 每个 volume 的位图字数：与 Rust `brickmap::builder::OCC_WORDS` **必须同值**。
/// 写成推导式（64³ 个 chunk 槽 ÷ 32 位/字）而不是字面量：两边漂移是静默的（寻址错位）。
const OCC_WORDS: u32 = (64u * 64u * 64u) / 32u;

// ============================================================================
// 硬件光追（`@if(ray_query)`）：TLAS 取代"逐 chunk 读索引区找下一个有几何的块"那一段
// ============================================================================

/// 每条射线最多发起几次 ray query（每次 = 一个被占用的 chunk）。到顶就置 `exhausted`
/// ⇒ 调用点落回软件步进（**不能当 miss** —— 那会把后面的几何吞掉）。
///
/// 正常情形只有 **1–3 次**：查询给的是"最近的**块**"，随后软件下钻**整块**并在未命中时越过它；
/// 只有"擦到盒但错过几何"才会多查一次。取 64 只是防挂死的安全网。
const RT_QUERY_BUDGET: u32 = 64u;
/// RT 说 miss 时，是否**直接按 miss 收**（不再落回软件步进）。
///
/// WARNING: 置 `1u` 会让"RT 的覆盖不完整"或"查询没命中"直接变成**画面上少几何** —— 实测（近场
/// 依赖 RT）就是**主世界整片不渲染**，而远场（软件 DDA）照常 ⇒ 症状是"近场没渲染"。
/// 它省下的是"miss 射线不必再走一遍软件步进"，但那正是软件路径作为**语义真值**的价值所在；
/// 在"查询确实会命中"被证实之前，本值必须保持 `0u`。
///
/// `exhausted` 只能挡住"预算耗尽 / 两侧不一致"这几种可判定的不完整，挡不住"TLAS 少了几千个实例"
/// 或"驱动侧 AABB ray query 就是不报候选"—— 那两种只能靠"落回软件"兜住。
const RT_TRUST_MISS: u32 = 0u;
/// 实例 cull mask：不筛（Rust 侧建实例时一律给 `0xFF`）。
const RT_CULL_MASK: u32 = 0xFFu;
/// ray flags：空。几何带 `OPAQUE` 只免掉 any-hit —— **procedural（AABB）命中不会因此自动提交**，
/// 仍然要在 `rt_query` 里显式 `rayQueryConfirmIntersection`。
const RT_FLAGS: u32 = 0u;

/// 提交一次查询并返回**最近命中**（`kind == RAY_QUERY_INTERSECTION_NONE` ⇒ 没打中）。
///
/// CONSTRAINT：这里用的是 **naga 的 ray query 接口**（`rayQueryInitialize(&q, as, RayDesc(..))` +
/// `rayQueryProceed(&q)` + `rayQueryGetCommittedIntersection(&q)`），**不是** WGSL 草案里的
/// `rayQueryInitializeEXT` 那一族 —— 本仓着色器最终由 naga 解析（编译期测试 + 运行期建 module），
/// 而 naga 30 只认前者（后者报 `no definition in scope`）。
@if(ray_query)
fn rt_query(ro: vec3<f32>, rd: vec3<f32>, tmin: f32, tmax: f32) -> RayIntersection {
  var q: ray_query;
  rayQueryInitialize(&q, tlas, RayDesc(RT_FLAGS, RT_CULL_MASK, tmin, tmax, ro, rd));
  // CONSTRAINT: 本仓的图元是 **AABB（procedural）几何** —— 它的命中在 Vulkan/SPIR-V 里是**候选交点**，
  // 必须显式 `rayQueryConfirmIntersection` 才会提交（只 `Proceed` 不确认 ⇒ `getCommittedIntersection`
  // 永远是 NONE = 整片 miss）。候选按 t 升序给出 ⇒ 确认第一个就是最近的那个块。
  if (rayQueryProceed(&q)) {
    if (rayQueryGetCandidateIntersection(&q).kind != RAY_QUERY_INTERSECTION_NONE) {
      rayQueryConfirmIntersection(&q);
    }
  }
  return rayQueryGetCommittedIntersection(&q);
}

/// chunk 盒 `[mn, mn+CHUNK_SIZE]` 在射线上的**出口** t（没有穿过 ⇒ 返回 `t_hi`）。
///
/// 这是"每 chunk 一个盒子"这个粒度换来的能力：出口可算 ⇒ 下钻完就能安全越过整块。子块粒度的盒
/// 算不出出口，因此不能用（见 `rt.rs` 文件头"粒度"的 CONSTRAINT）。
@if(ray_query)
fn chunk_exit_t(mn: vec3<f32>, ro: vec3<f32>, inv_rd: vec3<f32>, axis_on: vec3<bool>, t_hi: f32) -> f32 {
  var t1 = t_hi;
  for (var a: u32 = 0u; a < 3u; a = a + 1u) {
    // 轴平行 ⇒ 该轴对出口无贡献（`0 × ∞` 会出 NaN）
    if (!axis_on[a]) { continue; }
    let ta = (mn[a] - ro[a]) * inv_rd[a];
    let tb = (mn[a] + f32(CHUNK_SIZE) - ro[a]) * inv_rd[a];
    t1 = min(t1, max(ta, tb));
  }
  return t1;
}

/// 硬件光追版的**主世界**遍历：ray query 直接给出"下一个有几何的 chunk"，块内仍走软件
/// [`trace_chunk_opaque`]。换掉的只是**"逐个 chunk 读 1 MB 索引区"那一段**。
///
/// WHY 一次查询够用：查询返回的是**最近的块**（不是子块）⇒ 拿到入口 `t` 之后**下钻整块**即可
/// —— AABB 是保守包围盒，从最近盒的入口 `t` 起下钻不会漏掉任何几何。这正是"一次查询回答整条
/// 射线"的形态；若改成"每个 64³ 子块查一次"，就退化成"每次都要从 BVH 根下行"的迭代查询（实测净亏）。
///
/// CONSTRAINT: 只用于 `obj_id < 0`（主世界）且 `penetrate == false`。物体的格空间带旋转 / 缩放，
/// 其 AABB 不在 TLAS 里；介质穿透版的出射面语义建在 **chunk 边界**上，而这里会整块跳过空气 chunk。
///
/// `exhausted`（出参）：`1` = 这次遍历**不能证明**是 miss（预算耗尽 / 实例集与 `b_struct` 不一致）
/// ⇒ 调用点必须落回软件步进。`0` + 返回 miss = 候选全抽干且两侧一致 ⇒ 这就是真 miss。
@if(ray_query)
fn trace_grid_rt(ro: vec3<f32>, rd: vec3<f32>, tl0: f32, tl1: f32, depth_cap: u32,
                 sign_v: vec3<i32>, basis: RayBasis, g: Grid,
                 col0: vec3<f32>, col1: vec3<f32>, col2: vec3<f32>,
                 exhausted: ptr<function, u32>) -> UnifiedHit {
  let miss = UnifiedHit(false, 0.0, 0u, vec3<f32>(0.0), 0u, vec3<i32>(0), g.obj_id);
  let index_origin = g.index_origin;
  let tree_base = g.tree_base;
  // 首个 chunk 的入射面（同一个兜底口径：取方向的主轴面对应的面号，见 `trace_grid` 里同名的段）。
  let ad = abs(rd);
  var entry_face: u32 = 4u;
  if (ad.x >= max(ad.y, ad.z)) {
    entry_face = basis.face_base.x;
  } else if (ad.y >= ad.z) {
    entry_face = basis.face_base.y;
  } else {
    entry_face = basis.face_base.z;
  }
  var t_lo = tl0;
  var guard: u32 = 0u;
  loop {
    if (guard >= RT_QUERY_BUDGET) { *exhausted = 1u; break; }
    guard = guard + 1u;
    let q = rt_query(ro, rd, t_lo, tl1);
    if (q.kind == RAY_QUERY_INTERSECTION_NONE) { break; }
    let c8 = q.instance_custom_data;
    // `custom_data` = chunk 坐标三轴低 8 位（`rt.rs::rt_custom_index`）⇒ 模 256 的差就是窗口相对下标
    // （窗口每轴 ≤ 64 ⇒ 无歧义）。
    let rel = vec3<i32>(
      (i32(c8 & 0xFFu) - (index_origin.x & 0xFF)) & 0xFF,
      (i32((c8 >> 8u) & 0xFFu) - (index_origin.y & 0xFF)) & 0xFF,
      (i32((c8 >> 16u) & 0xFFu) - (index_origin.z & 0xFF)) & 0xFF,
    );
    // 窗口范围检查：解码错（或实例表里有窗口外的块）时**必须早退**，否则会按错地址读 `b_struct`
    // —— 读出来的"树基址"是垃圾，软件下钻会去访问别的 chunk 的内容。
    // 这是**不一致**而不是 miss ⇒ 置 `exhausted`，让调用点落回软件（那里是语义真值）。
    if (any(rel < vec3<i32>(0)) || any(vec3<u32>(rel) >= g.index_dims)) { *exhausted = 1u; break; }
    let ci = index_origin + rel;
    let rel_u = vec3<u32>(rel);
    let index_addr = tree_base + rel_u.x + rel_u.y * CHUNK_INDEX_CAP
      + rel_u.z * (CHUNK_INDEX_CAP * CHUNK_INDEX_CAP);
    let entry = b_struct[index_addr];
    // 常驻块必有非零条目（`install_blob` 与窗口条目在同一个调用里写）⇒ 走到这里说明两侧不一致。
    if (entry == 0u || entry == INDEX_ENTRY_EMPTY) { *exhausted = 1u; break; }
    let chunk_base = tree_base + entry - 1u;
    let chunk_min = vec3<f32>(vec3<i32>(i32(CHUNK_SIZE)) * ci);
    let t0 = max(q.t, t_lo);
    let t1 = chunk_exit_t(chunk_min, ro, basis.inv_rd, basis.axis_on, tl1);
    if (t1 <= t0) {
      // 数值边界：本块在入口之后已经走完 ⇒ 前进一点点再查，避免原地打转。
      t_lo = t0 + 1.0e-3;
      if (t_lo >= tl1) { break; }
      continue;
    }
    let h = trace_chunk_opaque(chunk_base, chunk_min, ro, rd, sign_v, basis,
                               t0, t1, entry_face, depth_cap);
    if (h.hit) { return grid_hit(h, ci, col0, col1, col2, g.obj_id); }
    // 本块这条射线上确实没有几何（AABB 是**保守**的）⇒ 从块尾之外继续找下一个块。
    // CONSTRAINT: 必须**严格前进**（`t1 > t0 >= t_lo` 已保证），否则会反复命中同一块、烧完预算。
    t_lo = max(t1 + max(1.0e-3, t1 * 1.0e-6), t_lo + 1.0e-3);
    if (t_lo >= tl1) { break; }
  }
  return miss;
}

// 统一层次 DDA：slab→chunk 间 A&W→trace_chunk，`penetrate = false` 时一次都不碰 mc。
/// `req` = 本 pass 是否允许发 ray-guided 请求（0 = 不发；见 `REQ_ENABLE` 的 CONSTRAINT）。
///
/// `gap`（出参/入参）：**第一次走到"没加载"的 chunk 时的 `t`**（只增不减地取 min；初值 `1e30` = 暂无）。
/// 调用方（`world.wesl::trace_scene`）用它给远场级的内边界兜底：近场没命中而路上有"没加载"的格子 ⇒
/// 那一段用**粗级**顶上，而不是留洞（渐进加载的"先给糊的"）。`INDEX_ENTRY_EMPTY`（CPU 确认"这块是
/// 空的"）**不算** gap —— 那里本来就没有内容，粗级也一样空。
fn trace_grid(g: Grid, origin: vec3<f32>, dir: vec3<f32>, t_cap: f32, t_min: f32, depth_cap: u32,
              penetrate: bool, req: u32, mc: ptr<function, MediumRay>,
              gap: ptr<function, f32>) -> UnifiedHit {
  let w_mn = g.w_mn;
  let w_mx = g.w_mx;
  let l_min = g.l_min;
  let l_max = g.l_max;
  let index_origin = g.index_origin;
  let index_dims = g.index_dims;
  let tree_base = g.tree_base;
  let max_chunk_steps = g.max_chunk_steps;
  let col0 = g.col0;
  let col1 = g.col1;
  let col2 = g.col2;
  let obj_id = g.obj_id;
  // M8：请求字与用途戳表都要卷号（远场级的 chunk 坐标与主世界**数值上会撞**）
  let vol = g.vol;
  let miss = UnifiedHit(false, 0.0, 0u, vec3<f32>(0.0), 0u, vec3<i32>(0), obj_id);
  // M4：本条射线已发出的请求数（见 `REQ_PER_RAY_MAX`）
  var req_sent: u32 = 0u;
  // ---- 世界 AABB 预剔除（slab 内联）----
  var bx_enter = 0.0;
  var bx_exit = t_cap;
  if (abs(dir.x) < 1e-30) {
    if (origin.x < w_mn.x || origin.x > w_mx.x) { return miss; }
  } else {
    let ta = (w_mn.x - origin.x) / dir.x;
    let tb = (w_mx.x - origin.x) / dir.x;
    bx_enter = max(bx_enter, min(ta, tb));
    bx_exit = min(bx_exit, max(ta, tb));
  }
  if (abs(dir.y) < 1e-30) {
    if (origin.y < w_mn.y || origin.y > w_mx.y) { return miss; }
  } else {
    let ta = (w_mn.y - origin.y) / dir.y;
    let tb = (w_mx.y - origin.y) / dir.y;
    bx_enter = max(bx_enter, min(ta, tb));
    bx_exit = min(bx_exit, max(ta, tb));
  }
  if (abs(dir.z) < 1e-30) {
    if (origin.z < w_mn.z || origin.z > w_mx.z) { return miss; }
  } else {
    let ta = (w_mn.z - origin.z) / dir.z;
    let tb = (w_mx.z - origin.z) / dir.z;
    bx_enter = max(bx_enter, min(ta, tb));
    bx_exit = min(bx_exit, max(ta, tb));
  }
  let bx0 = max(bx_enter, 0.0);
  if (bx_exit < bx0 || bx_enter >= t_cap) { return miss; }
  let t_hi_cap = min(bx_exit, t_cap);
  let scale = g.scale;
  // 主世界（obj_id < 0）= identity 变换（make_grid 保证 pos=0/rot=I/scale=1），局部坐标直取世界坐标。
  var ro: vec3<f32>;
  var rd: vec3<f32>;
  var tl0: f32;
  var tl1: f32;
  if (obj_id < 0) {
    ro = origin;
    rd = dir;
    tl0 = bx0;
    tl1 = t_hi_cap;
  } else {
    // ---- 局部变换 ----
    let wp = origin - g.pos;
    ro = vec3<f32>(dot(wp, col0), dot(wp, col1), dot(wp, col2)) / scale;
    rd = vec3<f32>(dot(dir, col0), dot(dir, col1), dot(dir, col2)) / scale;
    // ---- 局部 AABB slab（内联）----
    var tl_enter = 0.0;
    var tl_exit = t_hi_cap;
    if (abs(rd.x) < 1e-30) {
      if (ro.x < l_min.x || ro.x > l_max.x) { return miss; }
    } else {
      let ta = (l_min.x - ro.x) / rd.x;
      let tb = (l_max.x - ro.x) / rd.x;
      tl_enter = max(tl_enter, min(ta, tb));
      tl_exit = min(tl_exit, max(ta, tb));
    }
    if (abs(rd.y) < 1e-30) {
      if (ro.y < l_min.y || ro.y > l_max.y) { return miss; }
    } else {
      let ta = (l_min.y - ro.y) / rd.y;
      let tb = (l_max.y - ro.y) / rd.y;
      tl_enter = max(tl_enter, min(ta, tb));
      tl_exit = min(tl_exit, max(ta, tb));
    }
    if (abs(rd.z) < 1e-30) {
      if (ro.z < l_min.z || ro.z > l_max.z) { return miss; }
    } else {
      let ta = (l_min.z - ro.z) / rd.z;
      let tb = (l_max.z - ro.z) / rd.z;
      tl_enter = max(tl_enter, min(ta, tb));
      tl_exit = min(tl_exit, max(ta, tb));
    }
    if (tl_exit < max(tl_enter, 0.0)) { return miss; }
    tl0 = max(tl_enter, 0.0);
    tl1 = min(tl_exit, t_hi_cap);
  }
  if (tl1 <= tl0) { return miss; }
  // t_min = 3×3 邻域低分辨率射线的最小命中距离，跳过 [tl0, t_min) 的空空间 traversal（保守下界）。
  tl0 = max(tl0, t_min);
  if (tl1 <= tl0) { return miss; }
  // debug_mode.z == 2（诊断）：跳过 chunk 步进层。
  if (view_u.debug_mode.z > 1.5) { return miss; }
  // ---- chunk 间 A&W（256³ 一格）+ chunk 内层次 mask DDA ----
  var sign_v = vec3<i32>(1i);
  sign_v = select(sign_v, vec3<i32>(-1i), rd < vec3<f32>(0.0));
  // 方向位掩码 LUT：一次查表拿到 side（boundary 基址）与三轴步进 face id。
  let dir_mask = u32(sign_v.x >= 0) | (u32(sign_v.y >= 0) << 1u) | (u32(sign_v.z >= 0) << 2u);
  let dir_word = DIR_LUT[dir_mask];
  let side = vec3<f32>(
    f32(dir_word & 1u),
    f32((dir_word >> 1u) & 1u),
    f32((dir_word >> 2u) & 1u)
  );
  let face_x = (dir_word >> 8u) & 0xFFu;
  let face_y = (dir_word >> 16u) & 0xFFu;
  let face_z = (dir_word >> 24u) & 0xFFu;
  let grid_inv_rd = 1.0 / rd;
  let grid_axis_on = abs(rd) > vec3<f32>(1e-30);
  let delta = select(vec3<f32>(1e+30), abs(grid_inv_rd), grid_axis_on);
  let delta_c = delta * f32(CHUNK_SIZE);
  // ---- 射线级不变量：本 volume 内**每次 chunk 进入**都复用这一份（见 `RayBasis`）----
  // 与 `trace_chunk*` 里原来就地重算的表达式**逐项等价**：`side` / `face_x/y/z` / `dir_mask` 全出自
  // 上面那次 `DIR_LUT` 查表，而 `DIR_LUT` 与那三组 `select(...)` 逐位等价（见 `common.wesl` 的表注释）。
  let basis = RayBasis(
    grid_inv_rd,
    abs(grid_inv_rd),
    grid_axis_on,
    side,
    vec3<u32>(face_x & 1u, face_y & 1u, face_z & 1u),
    dir_mask);
  // 硬件光追路径（仅 RT 版 shader）：主世界 + 不透明射线**先试 TLAS** —— 它一次给出"下一个有几何的
  // chunk"，正是软件循环里"逐个 chunk 读 1 MB 索引区"那一段的替代。其余（物体 / 远场级 / 介质穿透）
  // 全程走软件步进：物体的 AABB 不在 TLAS 里，介质穿透的出射面语义建在 chunk 边界上。
  //
  // CONSTRAINT: RT 命中就返回；**覆盖完整的 miss 也直接返回**（见 `RT_TRUST_MISS`）；只有
  // `exhausted`（预算耗尽 / 两侧不一致）才落回下面的软件步进 —— 软件是语义真值，RT 只是加速。
  @if(ray_query)
  if (obj_id < 0 && !penetrate) {
    var ex: u32 = 0u;
    let h = trace_grid_rt(ro, rd, tl0, tl1, depth_cap, sign_v, basis, g, col0, col1, col2, &ex);
    if (h.hit) { return h; }
    if (RT_TRUST_MISS == 1u && ex == 0u) { return miss; }
  }
  let start = ro + rd * tl0;
  var ci = vec3<i32>(floor(start / f32(CHUNK_SIZE)));
  // tmax_c：到下一 chunk 边界的 t（ro 系绝对）
  let bnd_c = (vec3<f32>(ci) + side) * f32(CHUNK_SIZE);
  var tmax_c = select(vec3<f32>(1e+30), (bnd_c - ro) * grid_inv_rd, grid_axis_on);
  tmax_c = max(tmax_c, vec3<f32>(tl0));
  var t_enter_c = tl0;
  // entry_face = 跨入当前 chunk 的面（逆射线方向的主轴面）；首 chunk 用它兜底（相机贴面/UB）。
  var entry_face: u32 = 4u;
  let ad = abs(rd);
  if (ad.x >= max(ad.y, ad.z)) {
    entry_face = face_x;
  } else if (ad.y >= ad.z) {
    entry_face = face_y;
  } else {
    entry_face = face_z;
  }
  for (var step_c: u32 = 0u; step_c < max_chunk_steps; step_c = step_c + 1u) {
    let t_exit_c = min(tmax_c.x, min(tmax_c.y, tmax_c.z));
    let t1 = min(t_exit_c, tl1);
    // ---- 空空间跳过：当前 chunk 所在的 4³ 组里一个 GPU 树块都没有 ⇒ **索引条目必为 0** ----
    //
    // 只省掉那一次读（1 MB 索引区，L2 常驻 / 竞争激烈），换成 32 KB 位图里的一次 L1 读。
    //
    // CONSTRAINT（**这是本优化唯一的正确性依据，别改成"跳转"**）：
    //   · 只**删掉一次必然为 0 的读**，遍历路径与 `ci` / `tmax_c` / `t_enter_c` / `entry_face`
    //     与不跳过时**逐位相同** —— 落点、下钻起点、命中体素因此不可能变。
    //   · 曾经写过"直接跳到组出口"的版本（省掉 A&W 的算术），它必须**重新推导** `tmax_c`；
    //     而原路径的 `tmax_c` 是逐 chunk `+= delta_c` **累加**出来的，一次重算只差 ULP，
    //     但在体素边界上 `floor` 会翻转 ⇒ 命中体素换了（表现为"本来不发光的体素变发光"）。
    //     要省那份算术，必须让重算与累加**逐位一致**，否则不许动。
    //
    // 只在**请求通道关闭**时生效：`REQ_ENABLE` 打开时那条 `entry == 0` 分支要发请求，
    // 而省掉读就看不见那件事。
    //
    // 覆盖**每个 volume**（位图按 `vol * OCC_WORDS` 分段）。收益最大的是**远场级**：它一窗口只有
    // 几百块常驻（占用率 ≪ 1%），而主射线每条都要穿过去。
    var group_empty = false;
    if (OCC_SKIP != 0u && vol < GRID_VOLUMES && REQ_ENABLE == 0u) {
      let rg = ci - index_origin;
      if (all(rg >= vec3<i32>(0)) && all(vec3<u32>(rg) < index_dims)) {
        let rgg = vec3<u32>(rg);
        let gg = (rgg.x >> 2u) + (rgg.y >> 2u) * 16u + (rgg.z >> 2u) * 256u;
        let base = vol * OCC_WORDS + gg * 2u;
        group_empty = occ[base] == 0u && occ[base + 1u] == 0u;
      }
    }
    // ---- chunk 窗口定位（窗口外 / entry==0 = 空气）----
    var air_chunk = true;
    let rel = ci - index_origin;
    if (all(rel >= vec3<i32>(0)) && all(vec3<u32>(rel) < index_dims)) {
      let rel_u = vec3<u32>(rel);
      let index_addr = tree_base + rel_u.x + rel_u.y * CHUNK_INDEX_CAP
        + rel_u.z * (CHUNK_INDEX_CAP * CHUNK_INDEX_CAP);
      // `group_empty` ⇒ 该组的每个 chunk 都没有树块 ⇒ 索引条目必为 0（与读出来的值相同）⇒ 整趟读可以省。
      // CONSTRAINT: 必须用 `if` 而不是 `select` —— `select` 两个分支都会求值，读照样发生。
      var entry: u32 = 0u;
      if (!group_empty) { entry = b_struct[index_addr]; }
      // **"空"与"没加载"是两件事**（见 `common.wesl` 的条目编码）：
      // - `entry == 0` = 还没加载 ⇒ 射线本来要的细节缺了，该请求（M4）；
      // - `entry == INDEX_ENTRY_EMPTY` = CPU 已确认这块没有内容 ⇒ **不该请求**（发了也永远装不上）。
      //   无限世界不需要这条（每个 chunk 都有内容），MC 地图必需：城里大多数 section 是空气，实测最热
      //   一块 **103 万票/窗**（`REQ[6.8M 条、超容丢失 5.8M]`）—— 请求环被这一块整圈覆盖，真正的请求
      //   全被挤掉，且每秒数百万次原子写砸在同一个字上。
      // 两者对遍历都是空气。
      let has_tree = entry != 0u && entry != INDEX_ENTRY_EMPTY;
      // "没加载"（`entry == 0`）= 本该有的细节缺了 ⇒ 记下最早的那个 `t`，供远场级兜底（见参数 `gap`）。
      // CONSTRAINT: `group_empty` 时我们**没读条目**（省了那次 1 MB 读）⇒ 分不出"没加载"与"已知空"，
      // 因此按保守方向记一笔（当作可能没加载）。代价是真空区也会去问一次远场；那里远场同样为空，
      // 只是多走一趟遍历 —— 而**漏记**会让洞填不上，两害相权取轻。
      if (group_empty || entry == 0u) { *gap = min(*gap, t_enter_c); }
      // M4：三道闸门（不加节流会变成每秒数亿次原子写，GPU 全耗在 `lod_req[0]` 上）：
      // ① 只要 16³ 或更细的（`REQ_LEVEL_KEEP`）② 采样 `1/REQ_SAMPLE` ③ 每条射线 ≤ `REQ_PER_RAY_MAX`。
      // M8：**远场级跳过闸门 ①** —— 那个闸门问的是"这个距离值不值得加载细节"，而远场级的
      // chunk 尺度本身就是为那个距离设计的（`scale` 就是按"一格 ≈ 一像素"选的）⇒ 一律报 0 档。
      // 不跳的话：L2 起 `req_level` 恒 ≥ 2 ⇒ 永远发不出请求 ⇒ 远场永远空着。
      if (entry == 0u && req != 0u && req_sent < REQ_PER_RAY_MAX && req_sampled(origin, dir, vol)) {
        let lv = select(req_level(t_enter_c * view_u.lod.x), 0u, vol > 0u);
        if (lv <= REQ_LEVEL_KEEP) {
          req_push(rel_u, lv, 0u, vol);
          req_sent += 1u;
        }
      }
      if (has_tree) {
        air_chunk = false;
        // **用途戳**（论文 §III.A 的 usage stamp）：已加载的 chunk 要留下"射线在看它"的证据 ——
        // 缓存替换（LRU）唯一的"最近使用"来源。**不占 `req_sent` 的名额**（那条留给"缺了"的请求：
        // 命中射线多，名额要留给真正需要加载的地方），只走 `REQ_SAMPLE` 采样闸门。
        // 与请求环的关键差别：戳表是**稠密**的（窗口内每 chunk 一格 + `atomicMax`）⇒ 并发再多也不
        // 会溢出、也不会把"缺了"的请求挤出环；而"每射线一条环记录"的写法会被可见性投票打爆。
        if (req != 0u && req_sampled(origin, dir, vol)) {
          req_use(rel_u, vol);
        }
        let chunk_base = tree_base + entry - 1u;
        let chunk_min = vec3<f32>(ci) * f32(CHUNK_SIZE);
        // chunk 内遍历从 max(chunk 入口, t_min) 开始，跳过 t_min 前的空空间
        let t0c = max(t_enter_c, t_min);
        // ---- 整 chunk 同材质介质：**整块跳过** ----
        // 判据 = 本 chunk 的根节点**统一**（mask 两字全 0）+ 它的 uniform 色**是可穿透介质**：
        // 此时"射线在本 chunk 内的整段"必然都是这一种介质 ⇒ 里面既没有命中、也没有出射面（无空气）
        // ⇒ **完全不必开 chunk**（不调 `trace_chunk`）。
        // 吸收按"逐段一次"（见 `MediumRay` 头注释）：只在**这一段的材质与上一段不同**时乘一次
        // （同材质跨 chunk 继续 ⇒ 不再乘第二次），这里不需要任何距离/跨度。
        // 收益：穿大块玻璃的射线从"每 4/16/64 体素一步（每步一次比较）"降到"**每 256³ chunk 一次**" ⇒
        // GI/阴影射线穿玻璃的代价与厚度**解耦**（每增厚 256 体素才多一个 chunk 步）。
        // `depth_cap` 不必判（它只约束**分裂子块**，统一 chunk 没有分裂子块）。
        // `last_pal` 写回该材质：供下一个 chunk 的"同段判定"与 `medium_exit`（出射面判定：
        // 介质在 chunk 边界遇空气 ⇒ 出射面落在边界）使用。`air_chunk` 保持 false（本 chunk 内没有空气，
        // 不该走"空气 chunk 出射"分支）。`t1 <= t0c`（beam 的 `t_min` 已把本 chunk 整段跳掉）时
        // **不改状态**，交回 `trace_chunk`（它自己会因 `t0 >= t1` 早返回）。
        var chunk_skipped = false;
        if (penetrate && t1 > t0c) {
          let r_ml = b_struct[chunk_base];
          let r_mh = b_struct[chunk_base + 1u];
          if (r_ml == 0u && r_mh == 0u) {
            let r_pal = b_struct[chunk_base + 2u] & 0xFFFFu;
            if (r_pal != 0u && r_pal != (*mc).last_pal) {
              let m_uni = medium_of((*mc).palette_base, r_pal);
              if (m_uni.a < 1.0) {
                if (DIAG_DISABLE_MEDIUM_TINT == 0u) {
                  (*mc).tint = (*mc).tint * m_uni.rgb;
                }
                (*mc).last_pal = r_pal;
                chunk_skipped = true;
              }
            }
          }
        }
        if (penetrate && !chunk_skipped) {
          let h = trace_chunk(chunk_base, chunk_min, ro, rd, sign_v, basis,
                              t0c, t1, entry_face, depth_cap, mc);
          if (h.hit) { return grid_hit(h, ci, col0, col1, col2, obj_id); }
        } else if (!penetrate) {
          let h = trace_chunk_opaque(chunk_base, chunk_min, ro, rd, sign_v, basis,
                                     t0c, t1, entry_face, depth_cap);
          if (h.hit) { return grid_hit(h, ci, col0, col1, col2, obj_id); }
        }
      }
    }
    // 空气 chunk：介质段在此终止 ⇒ 出射面落在 chunk 边界（法线 = 跨入本 chunk 的面）。
    if (penetrate && air_chunk && medium_exit(mc)) {
      let nl = face_normal_from_index(entry_face);
      let nw = normalize(nl.x * col0 + nl.y * col1 + nl.z * col2);
      return UnifiedHit(true, t_enter_c, (*mc).last_pal, nw, entry_face, vec3<i32>(0), obj_id);
    }
    if (t_exit_c >= tl1) { break; }
    if (tmax_c.x <= tmax_c.y && tmax_c.x <= tmax_c.z) {
      t_enter_c = tmax_c.x; tmax_c.x = tmax_c.x + delta_c.x; ci.x = ci.x + sign_v.x;
      entry_face = face_x;
    } else if (tmax_c.y <= tmax_c.z) {
      t_enter_c = tmax_c.y; tmax_c.y = tmax_c.y + delta_c.y; ci.y = ci.y + sign_v.y;
      entry_face = face_y;
    } else {
      t_enter_c = tmax_c.z; tmax_c.z = tmax_c.z + delta_c.z; ci.z = ci.z + sign_v.z;
      entry_face = face_z;
    }
  }
  // 走完本 grid 时仍在介质段内 ⇒ 出射面落在 grid 出口（法线 = 最后一次跨入的面）。
  if (penetrate && medium_exit(mc)) {
    let nl = face_normal_from_index(entry_face);
    let nw = normalize(nl.x * col0 + nl.y * col1 + nl.z * col2);
    return UnifiedHit(true, tl1, (*mc).last_pal, nw, entry_face, vec3<i32>(0), obj_id);
  }
  return miss;
}

// 从 `grid_descs[idx]` 解码统一 `Grid`（主世界 idx=0 = identity 变换，远场级 / 物体 idx≥1 = 带变换）。
// chunk 步数上限由窗口 dims 估算（曼哈顿对角 ×3 + 余量）。
fn make_grid(idx: u32) -> Grid {
  let d = grid_descs[idx];
  let origin = vec3<i32>(d.index_origin_x, d.index_origin_y, d.index_origin_z);
  let dims = vec3<i32>(i32(d.index_dims_x), i32(d.index_dims_y), i32(d.index_dims_z));
  let chunk_budget = u32(dims.x + dims.y + dims.z) * 3u + 16u;
  // 主世界（idx=0，identity）：跳过变换段（`obj_id < 0` 分支随常量折叠）；远场级 / 物体读 `rot/pos/scale`。
  let is_world = idx == 0u;
  // 局部 AABB = **窗口**（chunk 单位 → 局部体素）：所有 grid 都按窗口给。
  // WHY 不能写死 [0,256]（远场级会因此只渲染一个 chunk）：物体路径原先用 `vec3(0)..vec3(256)` 是因为
  // 物体的窗口恰是"一个 chunk 的内容窗口"；远场级的窗口是 64³ chunk ⇒ 必须取真窗口。
  let l_mn = vec3<f32>(vec3<i32>(i32(CHUNK_SIZE)) * origin);
  let l_mx = l_mn + vec3<f32>(vec3<i32>(i32(CHUNK_SIZE)) * dims);
  // 变换段：主世界是 identity（Rust `VolumeTransform::IDENTITY`）⇒ 不读 `rot0..2` / `pos_scale`。
  var col0 = vec3<f32>(1.0, 0.0, 0.0);
  var col1 = vec3<f32>(0.0, 1.0, 0.0);
  var col2 = vec3<f32>(0.0, 0.0, 1.0);
  var pos = vec3<f32>(0.0);
  var scale = 1.0;
  if (!is_world) {
    col0 = d.rot0.xyz;      // 旋转矩阵列
    col1 = d.rot1.xyz;
    col2 = d.rot2.xyz;
    pos = d.pos_scale.xyz;  // 位置
    scale = d.pos_scale.w;
  }
  return Grid(
    d.aabb_min.xyz, d.aabb_max.xyz,       // 世界 AABB
    l_mn, l_mx,                            // 局部 AABB（= 窗口）
    chunk_budget,
    col0, col1, col2,                      // 旋转矩阵列
    pos, scale,                            // pos + scale
    d.tree_base, d.palette_base,           // 数据源基址
    origin, vec3<u32>(d.index_dims_x, d.index_dims_y, d.index_dims_z),
    // obj_id 约定（镜像 Rust）：idx=0 主世界 → -1；idx≥1 物体 / 远场级 → idx-1。
    select(-1i, i32(idx) - 1i, idx > 0u),
    idx,                                   // vol：请求字 / 用途戳表的分段号（M8）
    d.coverage_r,                          // 覆盖半径：分壳裁剪的接力半径（M8）
  );
}

// ============================================================================
// M8 · 分壳裁剪（`docs/editable-gigavoxel.md` §4 M8）
// ============================================================================

/// `GridDesc._pad0` 的位编码：bit0 = 该 grid 是**远场级**。
/// 写侧 = Rust `brickmap::wire::GRID_FLAG_FAR`（`VolumesBuilder::snapshot` 按 `VolumeGrid::is_far_level` 置位）；
/// 读侧只有 [`grid_is_far`] 一个调用点（`world.wesl::trace_scene` 的分壳裁剪）。
const GRID_FLAG_FAR: u32 = 1u;

/// 该 grid 是不是远场级。
fn grid_is_far(idx: u32) -> bool {
  return (grid_descs[idx]._pad0 & GRID_FLAG_FAR) != 0u;
}

/// 世界 AABB 的**出口**距离（`ro` 系绝对 t，起点在盒内时为 > 0）：分壳裁剪用它算"上一级的覆盖半径"。
///
/// 为什么用 AABB 出口而不是"窗口半宽"（常数）：远场级的窗口是**相机居中**的盒 ⇒ 同一级沿不同方向
/// 的覆盖距离在 `R .. √3·R` 之间（对角更长）。用常数 R 会在对角方向把"上一级其实还覆盖着"的那段
/// 也交给下一级 ⇒ 两级在同一段空间里各画一份（近场细节 + 远场膨胀格）。
fn aabb_exit(mn: vec3<f32>, mx: vec3<f32>, origin: vec3<f32>, dir: vec3<f32>) -> f32 {
  var e = 1e+30;
  let inv = 1.0 / dir;
  for (var a: u32 = 0u; a < 3u; a = a + 1u) {
    // 轴平行时该轴对出口无贡献（`0 × ∞` 会出 NaN ⇒ 必须跳过）
    if (abs(dir[a]) > 1e-30) {
      e = min(e, max((mn[a] - origin[a]) * inv[a], (mx[a] - origin[a]) * inv[a]));
    }
  }
  return e;
}
