import package::bindings::{gi_cache, gi_cache_hash, gi_cache_state, gi_dirty_boxes, gi_seen_claim, gi_u, gi_vis_count, gi_vis_list};
import package::gi::consts::{GI_CACHE_EVICT_AGE, GI_CACHE_EVICT_AT, GI_CACHE_EVICT_STARVE_AGE};

// ============================================================================
// 逐 (体素,面) 辐照度缓存 —— 世界空间、相机无关的 GI 存储。
//
// 全部设计围绕「键里带面」这一点展开：
//   * 键 = (体素坐标, 面, 物体)：**面**在键里 ⇒ 薄板两侧天然是两个条目，不跨面插值 ⇒ 不漏光；
//   * 取值 = 命中体素**自己**的那个条目（不插值）⇒ 没有级联接缝、没有网格分辨率问题；
//   * 存储 = 哈希网格 + 开放寻址，按需创建，没有 atlas / 段池 / 认领位图 / LOD；
//   * 更新 = 从该面出发发真实光路（含二次顶点 NEE），运行时平均累积 ⇒ 收敛后确定、无屏幕噪声；
//   * 有界工作集 = 分配游标只增不减会在飞行十几屏新地形后耗尽 `GI_CACHE_SLOTS`（此后新露出的面
//     永远拿不到条目 ⇒ 着色退化成天光兜底、开/关 GI 画面一致且不可恢复）。所以「久未露面」的条目
//     会被回收复用：轮转段顺手回收，分配游标耗尽时着色侧也能现场夺取一个槽位（见 `gi_reclaim_slot`）；
//   * 复用 = 每条目自带一个 RIS reservoir：除本条目自己的余弦样本外，还「借用」同平面邻条目的
//     reservoir 方向并从本条目重发（重连接射线）⇒ 相邻条目的估计互相关联，大平面上的逐体素斑块被抹平。
//
// 约定（与着色侧一致）：条目存的是 `E_store`，取值侧再乘增益并除 π（`gi_cache_read(...) * params.z / π`）。
//
// 每帧预算：更新 pass 的 `GI_CACHE_UPDATE_BUDGET` 个线程切成两段 ——
//   · **可见段**（`GI_CACHE_VISIBLE_SHARE` %）消费「上一帧被着色侧看到」的条目紧凑列表
//     （着色侧 `gi_cache_read` 命中时顺手 claim + append，见 `gi_mark_visible`）⇒ 预算花在真正要紧的条目上；
//     列表比线程数长时取一个**轮换窗口**（起点随帧号推进）⇒ 多帧内覆盖全部可见条目；
//   · **轮转段**（余下预算）按帧号分片扫整表，覆盖远处 / 不可见条目（不饿死），
//     已收敛条目降频（`GI_CACHE_CONVERGED_N`，**两段都降频**），世代过期 / 脏区条目永远优先。
// 轮转起点仍由帧号推出 ⇒ 无跨帧游标、无竞态、丢帧不漏更新。
//
// 「变化失效」：太阳/天光/palette/世界全量变化 → 世代号（`gi_u.params.w`）整体作废旧数据（不清缓冲）；
// 编辑体素 → `gi_u.dirty_min.w` 给出的**脏盒数**（0 = 无）配合逐盒世界 voxel AABB 局部失效
// （盒 0 在 uniform，其余在 `gi_dirty_boxes`，见 `gi_dirty_hit`）。两者都只就地重置条目，
// 随后按正常调度重新收敛（整表扫一遍需 GI_CACHE_SLOTS / 轮转段线程数 帧）。
// ============================================================================

/// 本缓存总开关（着色侧还会与 uniform 的 GI 开关一起判）。
/// 条目还没建出来时（`count = 0`）返回 0，由着色侧的 `amb · ao` 天光兜底，这里不做第二套兜底。
///
/// **第一步（屏幕空间逐面 ReSTIR）已把它置 false**：主路径改为「每帧完整估计 + 屏幕空间复用」
/// （见 `gi/screen.wesl`），本世界空间缓存退居 **A/B 对照**（置回 true 即恢复旧路径，两条路径在
/// `gi_main` 里二选一）。退役（删除本文件与 `gi_cache_update` 派发）放在第③步。
/// Rust 侧从本文件解析同一个字面量（`wesl_consts.rs` 的 `gi_cache_on`）⇒ false 时连更新 pass
/// 都不派发（省下整表轮转的 ~0.4 ms）。
const GI_CACHE_ON: bool = false;
/// 条目容量（GI_CACHE_ENTRY_WORDS × 4 B = 72 B/条 ⇒ 72 MB）。游标过 `GI_CACHE_EVICT_AT`% 后靠回收
/// 复用老条目（见 `gi_cache_evict`）⇒ 这是**有界工作集**，不是「满了就永远建不出条目」。
const GI_CACHE_SLOTS: u32 = 1048576u;
/// 哈希桶数（2 的幂；× 4 B = 32 MB）。桶内存「条目下标 + 1」，0 = 空。
const GI_CACHE_BUCKETS: u32 = 8388608u;
// 单帧「已可见」claim（`gi_seen_claim`，GI_CACHE_SLOTS × 4 B = 4 MB）：着色侧用 `atomicExchange`
// 精确保证「每条目每帧只 append 一次」。与条目表去掉 word[18] 省下的 4 MB 刚好互换 ⇒ 净显存不变。
/// 脏盒上限：本帧生效的世界 voxel AABB 盒数（主世界 + 各物体 volume 的编辑各占一盒）。
/// 超出即退化为「自增世代」全量失效（宁可全量失效也不漏失效），见 `gi_dirty_hit`。
const GI_CACHE_DIRTY_BOXES: u32 = 8u;
/// 开放寻址最大探测步数。
const GI_CACHE_PROBE: u32 = 8u;
/// 每条目每帧发射的半球射线数（新鲜余弦样本）。
const GI_CACHE_RAYS: u32 = 2u;
/// 每条目每帧额外发射的**借用**射线数（RIS 复用）：从同平面邻体素条目的 reservoir 借方向，
/// 再从本条目出发重发一条射线。总有射线数 = GI_CACHE_RAYS + GI_CACHE_REUSE。
const GI_CACHE_REUSE: u32 = 2u;
/// RIS 的候选数上限 M：M 超过本值就把 w_sum 与 M 同比例缩回（w_sum / M 不变）
/// ⇒ 抽样概率不会被历史永久支配（每条目每帧只进 GI_CACHE_RAYS + GI_CACHE_REUSE 个候选）。
const GI_CACHE_M_CAP: u32 = 32u;
/// 每帧更新条目数上限（更新 pass 的线程数 = 可见段 + 轮转段的预算总和）。
const GI_CACHE_UPDATE_BUDGET: u32 = 65536u;
/// 可见条目紧凑列表容量（× 4 B = 4 MB）。着色侧 claim 后 append 条目下标、更新 pass 消费；
/// 溢出即丢弃本帧多余的可见条目（不越界；claim 早已置位，本帧不会再重复 append）。
const GI_CACHE_VISIBLE_CAPACITY: u32 = 1048576u;
/// 回收扫描游标所在的 state 下标（`gi_cache_state[1]`；`[0]` 是分配游标）。
/// 分配游标耗尽时从它开始找一个可复用的槽位（见 `gi_reclaim_slot`）。
const GI_CACHE_STATE_RECLAIM: u32 = 1u;
/// 一次分配里最多尝试多少个回收候选（游标耗尽时才走这条路）。32 次足以在「已回收槽位稀疏分布」
/// 的表里找到目标，同时把最坏开销钉死在 32 次表读 + 最多 32 次桶清理。
const GI_CACHE_RECLAIM_TRIES: u32 = 32u;
/// 条目 word 数。
const GI_CACHE_ENTRY_WORDS: u32 = 18u;
/// 条目内 word 偏移：`[0,1]` 键、`[2,4]` 世界位置、`[5]` 世界法线（八面体 f16x2）、
/// `[6,8]` E、`[9]` 已累积样本数、`[10]` 最近**处理**（更新或轮转，同帧去重用）的帧、
/// `[11]` 世代号（`gi_u.params.w` 的快照）、
/// `[12]` reservoir 样本方向（八面体 f16x2）、`[13,15]` reservoir 样本的 L、
/// `[16]` reservoir 跨帧权重和 w_sum、`[17]` reservoir 候选数 M。
/// 「最近被着色读取的帧」不落在条目里，改由 `gi_seen_claim` 承担（见 `gi_mark_visible`）。
const GI_CACHE_W_POS: u32 = 2u;
const GI_CACHE_W_NRM: u32 = 5u;
const GI_CACHE_W_E: u32 = 6u;
const GI_CACHE_W_COUNT: u32 = 9u;
const GI_CACHE_W_FRAME: u32 = 10u;
const GI_CACHE_W_GEN: u32 = 11u;
const GI_CACHE_W_RDIR: u32 = 12u;
const GI_CACHE_W_RL: u32 = 13u;
const GI_CACHE_W_WSUM: u32 = 16u;
const GI_CACHE_W_M: u32 = 17u;
/// 射线起点沿法线的外推（voxel 单位）：与着色侧同一套「半个体素 + eps」，避免第一步撞自己。
const GI_CACHE_RAY_BIAS: f32 = 0.5;

/// 键合法范围：13 bit 有符号偏移（voxel 坐标绝对值 < 4096）。
fn gi_key_ok(v: vec3<i32>) -> bool {
  return all(v >= vec3<i32>(-4096)) && all(v < vec3<i32>(4096));
}

/// 键 word 0：x、y 各 13 bit（+4096 偏移）。
fn gi_key0(v: vec3<i32>) -> u32 {
  return u32(v.x + 4096) | (u32(v.y + 4096) << 13u);
}

/// 键 word 1：z 13 bit、面 3 bit（13..15）、物体 id 16 bit（16..31，+1 存，0 = 主世界）。
fn gi_key1(v: vec3<i32>, face: u32, obj: i32) -> u32 {
  return u32(v.z + 4096) | ((face & 7u) << 13u) | (u32(obj + 1) << 16u);
}

/// 32 位混合哈希：键 → 桶下标。
/// 只用 `+ / * / >>`：本工程的 WESL 解析器不接受 `^`（异或）运算符，故按「乘-加-移位」混合。
fn gi_bucket_of(k0: u32, k1: u32) -> u32 {
  var h = k0 * 0x9E3779B1u + k1 * 0x85EBCA77u;
  h = (h + (h >> 16u)) * 0x7FEB352Du;
  h = (h + (h >> 15u)) * 0x846CA68Bu;
  return (h + (h >> 16u)) & (GI_CACHE_BUCKETS - 1u);
}

/// 32 位整数混合（RNG 用）。
fn gi_mix(x: u32) -> u32 {
  var h = x * 747796405u + 2891336453u;
  h = ((h >> ((h >> 28u) + 4u)) + h) * 277803737u;
  return (h >> 22u) + h;
}

/// [0,1)²。同一 (idx, frame, ray) 恒定 ⇒ 每条目的样本序列可复现，收敛后不抖。
fn gi_rand2(seed: u32) -> vec2<f32> {
  let a = gi_mix(seed);
  let b = gi_mix(a + 0x9E3779B9u);
  return vec2<f32>(f32(a), f32(b)) * (1.0 / 4294967296.0);
}

/// 世界法线 → 一个 word（八面体编码 + f16x2）。法线是单位向量，解码误差 < 1°。
fn gi_pack_dir(n: vec3<f32>) -> u32 {
  let l = abs(n.x) + abs(n.y) + abs(n.z);
  var o = n.xy / max(l, 1e-6);
  if (n.z < 0.0) {
    o = (vec2<f32>(1.0) - abs(o.yx)) * select(vec2<f32>(-1.0), vec2<f32>(1.0), o >= vec2<f32>(0.0));
  }
  return pack2x16float(o);
}

fn gi_unpack_dir(p: u32) -> vec3<f32> {
  let o = unpack2x16float(p);
  var n = vec3<f32>(o.x, o.y, 1.0 - abs(o.x) - abs(o.y));
  if (n.z < 0.0) {
    n = vec3<f32>(
      (1.0 - abs(n.y)) * select(-1.0, 1.0, n.x >= 0.0),
      (1.0 - abs(n.x)) * select(-1.0, 1.0, n.y >= 0.0),
      n.z,
    );
  }
  return normalize(n);
}

/// 面法线的余弦加权半球方向：`u` = [0,1)²，`dir` 恒在 `n` 半球内。
fn gi_cos_hemisphere(n: vec3<f32>, u: vec2<f32>) -> vec3<f32> {
  let r = sqrt(u.x);
  let phi = 6.2831853 * u.y;
  let axis = select(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 0.0), abs(n.z) > 0.9);
  let t = normalize(cross(axis, n));
  let b = cross(n, t);
  return normalize(t * (r * cos(phi)) + b * (r * sin(phi)) + n * sqrt(max(1.0 - u.x, 0.0)));
}

/// 条目键 word 0；无效条目（键为 0）返回 0。
fn gi_entry_key0(idx: u32) -> u32 {
  return gi_cache[idx * GI_CACHE_ENTRY_WORDS];
}

/// 条目键 word 1（面 3 bit + 物体 id 16 bit）。
fn gi_entry_key1(idx: u32) -> u32 {
  return gi_cache[idx * GI_CACHE_ENTRY_WORDS + 1u];
}

/// 条目键 → 体素坐标（`gi_key0/gi_key1` 的逆；找同平面邻体素要用）。
fn gi_entry_voxel(idx: u32) -> vec3<i32> {
  let k0 = gi_entry_key0(idx);
  let k1 = gi_entry_key1(idx);
  return vec3<i32>(
    i32(k0 & 0x1FFFu) - 4096,
    i32((k0 >> 13u) & 0x1FFFu) - 4096,
    i32(k1 & 0x1FFFu) - 4096,
  );
}

/// 条目键 → 面（0..5）。
fn gi_entry_face(idx: u32) -> u32 {
  return (gi_entry_key1(idx) >> 13u) & 7u;
}

/// 条目键 → 物体 id（-1 = 主世界）。
fn gi_entry_obj(idx: u32) -> i32 {
  return i32(gi_entry_key1(idx) >> 16u) - 1i;
}

/// 条目是否已建（键非 0）。
fn gi_entry_valid(idx: u32) -> bool {
  let e = idx * GI_CACHE_ENTRY_WORDS;
  return (gi_cache[e] | gi_cache[e + 1u]) != 0u;
}

fn gi_entry_pos(idx: u32) -> vec3<f32> {
  let e = idx * GI_CACHE_ENTRY_WORDS + GI_CACHE_W_POS;
  return vec3<f32>(
    bitcast<f32>(gi_cache[e]),
    bitcast<f32>(gi_cache[e + 1u]),
    bitcast<f32>(gi_cache[e + 2u]),
  );
}

fn gi_entry_nrm(idx: u32) -> vec3<f32> {
  return gi_unpack_dir(gi_cache[idx * GI_CACHE_ENTRY_WORDS + GI_CACHE_W_NRM]);
}

/// 已累积样本数（0 = 还没更新过）。
fn gi_entry_count(idx: u32) -> f32 {
  return bitcast<f32>(gi_cache[idx * GI_CACHE_ENTRY_WORDS + GI_CACHE_W_COUNT]);
}

/// 条目最近一次被**处理**（更新或轮转）的帧号（0 = 还没处理过）。
/// 词里就是 `gi_u.seq.x` 原值（精确 u32），这里无需换算。
/// 更新 pass 用它挡掉「同一帧里被两个调度分支同时选中」的条目，避免同帧累积两次。
fn gi_entry_frame(idx: u32) -> u32 {
  return gi_cache[idx * GI_CACHE_ENTRY_WORDS + GI_CACHE_W_FRAME];
}

// ---- RIS reservoir（样本级复用）----
// 每条目自带一个加权蓄水池：本帧的候选（新鲜余弦 + 借邻居方向）与**历史**候选一起竞争，
// 只保留一个样本 `(dir, L)`、以及跨帧累计的标量权重和 `w_sum` 与候选数 `M`。
// 邻居借用时读 `dir`（再从自己出发重发一条射线），并用 `w_sum / M`（≈ 该条目的目标函数积分 Z）
// 与样本自己的 L 把权重算成 `p̂ / p`（见 `gi_cache_update` 的借用分支）。

/// reservoir 样本方向（八面体 f16x2；M = 0 时无意义）。
fn gi_entry_reservoir_dir(idx: u32) -> vec3<f32> {
  return gi_unpack_dir(gi_cache[idx * GI_CACHE_ENTRY_WORDS + GI_CACHE_W_RDIR]);
}

/// reservoir 样本的入射辐亮度 L（选中时从本条目射线得到的值）。
fn gi_entry_reservoir_l(idx: u32) -> vec3<f32> {
  let e = idx * GI_CACHE_ENTRY_WORDS + GI_CACHE_W_RL;
  return vec3<f32>(
    bitcast<f32>(gi_cache[e]),
    bitcast<f32>(gi_cache[e + 1u]),
    bitcast<f32>(gi_cache[e + 2u]),
  );
}

/// reservoir 跨帧累计的权重和 `w_sum`（0 = 还没样本）。
fn gi_entry_w_sum(idx: u32) -> f32 {
  return bitcast<f32>(gi_cache[idx * GI_CACHE_ENTRY_WORDS + GI_CACHE_W_WSUM]);
}

/// reservoir 候选数 `M`（0 = 还没样本；上限 `GI_CACHE_M_CAP`）。
fn gi_entry_m(idx: u32) -> f32 {
  return bitcast<f32>(gi_cache[idx * GI_CACHE_ENTRY_WORDS + GI_CACHE_W_M]);
}

/// 写回 reservoir（候选竞争后的样本与累计量）。
fn gi_entry_store_reservoir(idx: u32, dir: vec3<f32>, l: vec3<f32>, w_sum: f32, m: f32) {
  let e = idx * GI_CACHE_ENTRY_WORDS + GI_CACHE_W_RL;
  gi_cache[idx * GI_CACHE_ENTRY_WORDS + GI_CACHE_W_RDIR] = gi_pack_dir(dir);
  gi_cache[e] = bitcast<u32>(l.x);
  gi_cache[e + 1u] = bitcast<u32>(l.y);
  gi_cache[e + 2u] = bitcast<u32>(l.z);
  gi_cache[idx * GI_CACHE_ENTRY_WORDS + GI_CACHE_W_WSUM] = bitcast<u32>(w_sum);
  gi_cache[idx * GI_CACHE_ENTRY_WORDS + GI_CACHE_W_M] = bitcast<u32>(m);
}

// ---- 失效（变化失效）：世代号 + 局部脏盒 ----
// 两条路径都只做「就地重置一个条目」，不批量清零 68MB 缓冲；重置后的条目按正常轮转重新收敛。
//   · 世代：太阳方向/颜色、天光、palette 变化、世界全量变化 → Rust 自增 `gi_u.params.w`，旧世代条目视作无数据；
//   · 脏盒：编辑体素 → `gi_u.dirty_min.w` 给盒数，逐盒世界 voxel AABB（盒 0 在 uniform、其余在
//     `gi_dirty_boxes`）＋逐盒余量，位置落在任一盒内的条目失效（见 `gi_dirty_hit`）。

/// 当前世代（Rust 每帧写 `gi_u.params.w`）。
fn gi_current_generation() -> u32 {
  return u32(gi_u.params.w);
}

/// 条目建立/最近一次更新时的世代。
fn gi_entry_generation(idx: u32) -> u32 {
  return gi_cache[idx * GI_CACHE_ENTRY_WORDS + GI_CACHE_W_GEN];
}

/// 条目属于旧世代（⇒ 当作无数据，不显示上一世代的值）。
fn gi_entry_stale(idx: u32) -> bool {
  return gi_entry_generation(idx) != gi_current_generation();
}

/// 位置是否落在本帧任一脏盒内（盒 = 世界 voxel AABB，闭开 `[min, max)`，再外扩该盒**自带**的余量）。
/// 盒 0 走 uniform：`gi_u.dirty_min.xyz` / `gi_u.dirty_max.xyz`，余量 = `gi_u.dirty_max.w` ——
/// 主世界（scale = 1）的单盒路径与旧版逐字等价，且不发生 storage 读。
/// 盒 1..count-1 走 `gi_dirty_boxes`（每盒 2 个 vec4：`[2k]` = min.xyz + 余量、`[2k+1]` = max.xyz），
/// 用于物体 volume（`vol_idx != 0`）的编辑：那些盒的世界 AABB 与余量（按 volume scale 放大）在 Rust 侧算好。
/// `count` = `gi_u.dirty_min.w`（0 = 本帧无脏区）；盒数溢出 `GI_CACHE_DIRTY_BOXES` 在 Rust 侧退化为世代自增。
fn gi_dirty_hit(pos: vec3<f32>) -> bool {
  let count = u32(gi_u.dirty_min.w);
  if (count == 0u) { return false; }
  let m0 = gi_u.dirty_max.w;
  if (all(pos >= gi_u.dirty_min.xyz - vec3<f32>(m0))
      && all(pos < gi_u.dirty_max.xyz + vec3<f32>(m0))) {
    return true;
  }
  for (var b = 1u; b < count; b = b + 1u) {
    let lo = gi_dirty_boxes[b * 2u];
    let hi = gi_dirty_boxes[b * 2u + 1u];
    if (all(pos >= lo.xyz - vec3<f32>(lo.w)) && all(pos < hi.xyz + vec3<f32>(lo.w))) {
      return true;
    }
  }
  return false;
}

/// 就地失效：`E = 0`、`n = 0`、reservoir 清空、世代 = 当前代（此后按正常轮转重新累积，等价于刚建出来）。
fn gi_entry_invalidate(idx: u32, gen: u32) {
  let e = idx * GI_CACHE_ENTRY_WORDS + GI_CACHE_W_E;
  gi_cache[e] = bitcast<u32>(0.0);
  gi_cache[e + 1u] = bitcast<u32>(0.0);
  gi_cache[e + 2u] = bitcast<u32>(0.0);
  gi_cache[idx * GI_CACHE_ENTRY_WORDS + GI_CACHE_W_COUNT] = bitcast<u32>(0.0);
  gi_cache[idx * GI_CACHE_ENTRY_WORDS + GI_CACHE_W_GEN] = gen;
  gi_entry_store_reservoir(idx, vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0), 0.0, 0.0);
}

/// 查条目下标；未建返回 `0xFFFFFFFF`。
/// 桶最后写、键先写 ⇒ 读到桶时键与几何都已就绪；抢输的重复条目只会短暂命中不到，不会读错。
fn gi_lookup(k0: u32, k1: u32) -> u32 {
  let start = gi_bucket_of(k0, k1);
  for (var i = 0u; i < GI_CACHE_PROBE; i = i + 1u) {
    let b = (start + i) & (GI_CACHE_BUCKETS - 1u);
    let slot = atomicLoad(&gi_cache_hash[b]);
    if (slot == 0u) { return 0xFFFFFFFFu; }
    let e = (slot - 1u) * GI_CACHE_ENTRY_WORDS;
    if (gi_cache[e] == k0 && gi_cache[e + 1u] == k1) { return slot - 1u; }
  }
  return 0xFFFFFFFFu;
}

/// 取键对应的条目，不存在就地创建（`pos` / `n` 由着色侧当场给出，更新 pass 直接拿来发射线）。
/// 返回条目下标；容量满 / 探测失败返回 `0xFFFFFFFF`。
fn gi_alloc(k0: u32, k1: u32, pos: vec3<f32>, n: vec3<f32>) -> u32 {
  let start = gi_bucket_of(k0, k1);
  for (var i = 0u; i < GI_CACHE_PROBE; i = i + 1u) {
    let b = (start + i) & (GI_CACHE_BUCKETS - 1u);
    let cur = atomicLoad(&gi_cache_hash[b]);
    if (cur != 0u) {
      let e = (cur - 1u) * GI_CACHE_ENTRY_WORDS;
      if (gi_cache[e] == k0 && gi_cache[e + 1u] == k1) { return cur - 1u; }
      continue;
    }
    // 游标用完就现场回收一个槽位（见 `gi_reclaim_slot`）—— 游标只增不减 ⇒ 只靠 bump 会在飞行
    // 十几屏新地形后耗尽容量（此后新面永远建不出条目，着色侧退化成天光兜底）。
    var idx = atomicAdd(&gi_cache_state[0u], 1u);
    if (idx >= GI_CACHE_SLOTS) {
      idx = gi_reclaim_slot(gi_u.seq.x);
      if (idx == 0xFFFFFFFFu) { return 0xFFFFFFFFu; }
    }
    let e = idx * GI_CACHE_ENTRY_WORDS;
    gi_cache[e] = k0;
    gi_cache[e + 1u] = k1;
    gi_cache[e + GI_CACHE_W_POS] = bitcast<u32>(pos.x);
    gi_cache[e + GI_CACHE_W_POS + 1u] = bitcast<u32>(pos.y);
    gi_cache[e + GI_CACHE_W_POS + 2u] = bitcast<u32>(pos.z);
    gi_cache[e + GI_CACHE_W_NRM] = gi_pack_dir(n);
    gi_cache[e + GI_CACHE_W_E] = bitcast<u32>(0.0);
    gi_cache[e + GI_CACHE_W_E + 1u] = bitcast<u32>(0.0);
    gi_cache[e + GI_CACHE_W_E + 2u] = bitcast<u32>(0.0);
    gi_cache[e + GI_CACHE_W_COUNT] = bitcast<u32>(0.0);
    gi_cache[e + GI_CACHE_W_FRAME] = bitcast<u32>(0.0);
    gi_cache[e + GI_CACHE_W_GEN] = gi_current_generation();
    gi_entry_store_reservoir(idx, vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0), 0.0, 0.0);
    if (atomicCompareExchangeWeak(&gi_cache_hash[b], 0u, idx + 1u).exchanged) { return idx; }
    // 抢输了：本条目在本帧是孤儿（不链进桶），交给赢家；下一帧重新命中即可。
    // 孤儿不在任何桶里 ⇒ 不会被 claim、`frame − claim` 只增不减 ⇒ 迟早被回收路径按「久未露面」收回。
    return 0xFFFFFFFFu;
  }
  return 0xFFFFFFFFu;
}

/// 是否应该回收 `idx`：游标过水位（`GI_CACHE_EVICT_AT`%）且该条目「最近被 claim 的帧」比当前帧
/// 早超过 `GI_CACHE_EVICT_AGE`；饥饿态（游标越界）放宽到 `GI_CACHE_EVICT_STARVE_AGE`。
/// 世代过期 / 脏区条目由调用方先处理（它们正要重收敛，不能回收）。
/// 「从未被 claim」（claim = 0）直接可回收：这类条目是 `gi_alloc` 抢桶失败的孤儿 —— 不在任何桶里，
/// `gi_lookup` 永远查不到，留着只占槽位。它同时消掉了「启动后前 `GI_CACHE_EVICT_AGE` 帧内
/// 没有任何条目够老 ⇒ 回收完全失效」的空窗（`frame` 从 1 起，否则要等 300 帧才第一次回收，
/// 而这段时间正是飞行最容易把容量打满的时候）。
fn gi_should_evict(idx: u32, frame: u32) -> bool {
  let cur = atomicLoad(&gi_cache_state[0u]);
  if (cur < GI_CACHE_SLOTS * GI_CACHE_EVICT_AT / 100u) { return false; }
  let claim = atomicLoad(&gi_seen_claim[idx]);
  if (claim == 0u) { return true; }
  let age = frame - claim;
  if (age > GI_CACHE_EVICT_AGE) { return true; }
  // 饥饿态（游标已越过容量 ⇒ 新条目正在失败）：放宽到 `GI_CACHE_EVICT_STARVE_AGE`。
  // 只用宽松门槛时回收常常找不到候选（表里全是年轻条目）⇒ 新面依旧拿不到槽位。
  return cur >= GI_CACHE_SLOTS && age > GI_CACHE_EVICT_STARVE_AGE;
}

/// 回收条目：清哈希桶里指向它的那一格 → 清键与世代（此后 `gi_entry_valid` 为假）。
/// 桶清理只按同样的探测顺序找「值为 idx + 1」的那一格写 0。
/// **已知代价**：线性探测下清空一格会形成"洞"，让「本该探测过该格之后」的键查不到
/// （偶发漏命中）。这里接受这一代价，不写墓碑哨兵：
///   · 漏命中的代价是下一帧重新分配一个同键条目（自愈）；
///   · 原条目变成孤儿（不在任何桶里），不会再被 claim ⇒ `frame − claim` 只会变大，
///     几百帧后被同一路径回收，槽位回到可用池；
///   · 实测探测开销不成问题：8M 桶 / 最多 1M 条目（负载 ≤ 12.5%）下 `GI_CACHE_PROBE = 8`
///     的探测预算从未被打满（回读统计「探测耗尽」恒为 0）。
/// 写墓碑需要一个哨兵值，且要在 `gi_lookup`（继续探测）与 `gi_alloc`（可复用）两处都特殊处理，
/// 而墓碑只增不减 —— 在探测预算本就充裕的场景里得不偿失。
fn gi_cache_evict(idx: u32) {
  let k0 = gi_entry_key0(idx);
  let k1 = gi_entry_key1(idx);
  let start = gi_bucket_of(k0, k1);
  for (var i = 0u; i < GI_CACHE_PROBE; i = i + 1u) {
    let b = (start + i) & (GI_CACHE_BUCKETS - 1u);
    if (atomicLoad(&gi_cache_hash[b]) == idx + 1u) {
      atomicStore(&gi_cache_hash[b], 0u);
      break;
    }
  }
  let e = idx * GI_CACHE_ENTRY_WORDS;
  gi_cache[e] = 0u;
  gi_cache[e + 1u] = 0u;
  gi_cache[e + GI_CACHE_W_GEN] = 0u;
}

/// 找一个**立即可复用**的槽位（只在分配游标耗尽时调用）：
/// `gi_cache_state[1]` 是回收扫描游标，`atomicAdd` 推进 ⇒ 同一 dispatch 内各线程拿到的候选互不重复
/// （不会两个线程抢同一槽位）。候选先看是不是已被轮转段回收成空槽（键全 0），否则「久未露面」
/// （`gi_should_evict`）就**就地**回收它 —— 全程只在本 dispatch（着色 pass）内读写，没有跨 pass 交接。
///
/// 这里刻意**不用空闲链表**：链表要求「更新 pass 推入、着色 pass 弹出」，两者只隔一个 pass 边界，
/// 弹出的线程读到的链指针可能还是上一帧的键（实测：标记为空闲的槽位 43 万个，而沿表头只能走到
/// 3 千个 —— 链被读旧值截断）。改成「就地夺取」后，读到旧值只会让本次候选被跳过（再试下一个），
/// 不会破坏任何结构；轮转段的回收只负责「提前腾地方」，不负责交接。
fn gi_reclaim_slot(frame: u32) -> u32 {
  for (var t = 0u; t < GI_CACHE_RECLAIM_TRIES; t = t + 1u) {
    let cand = atomicAdd(&gi_cache_state[GI_CACHE_STATE_RECLAIM], 1u) % GI_CACHE_SLOTS;
    if (!gi_entry_valid(cand)) { return cand; }
    if (gi_should_evict(cand, frame)) {
      gi_cache_evict(cand);
      return cand;
    }
  }
  return 0xFFFFFFFFu;
}

/// 运行时平均：`E ← E + (contrib - E) / (n + 1)`。三通道展开写，避免向量的动态下标。
fn gi_accumulate(idx: u32, contrib: vec3<f32>) {
  let e = idx * GI_CACHE_ENTRY_WORDS + GI_CACHE_W_E;
  let n = bitcast<f32>(gi_cache[idx * GI_CACHE_ENTRY_WORDS + GI_CACHE_W_COUNT]);
  let a = 1.0 / (n + 1.0);
  gi_cache[e] = bitcast<u32>(mix(bitcast<f32>(gi_cache[e]), contrib.x, a));
  gi_cache[e + 1u] = bitcast<u32>(mix(bitcast<f32>(gi_cache[e + 1u]), contrib.y, a));
  gi_cache[e + 2u] = bitcast<u32>(mix(bitcast<f32>(gi_cache[e + 2u]), contrib.z, a));
  gi_cache[idx * GI_CACHE_ENTRY_WORDS + GI_CACHE_W_COUNT] = bitcast<u32>(n + 1.0);
  // [10] = 最近**任一处理**（更新 / 轮转）的帧：更新 pass 的调度去重读的就是它。
  // 帧号取自 `gi_u.seq.x`（精确 u32；f32 在 >2^24 后不能表示连续整数 ⇒ 去重会偶发失效）。
  gi_cache[idx * GI_CACHE_ENTRY_WORDS + GI_CACHE_W_FRAME] = gi_u.seq.x;
  gi_cache[idx * GI_CACHE_ENTRY_WORDS + GI_CACHE_W_GEN] = gi_current_generation();
}

/// 同面、同一平面内第 `s`（0..3：沿面内两轴 a1/a2 各 ±1）个邻体素**已就绪**条目的下标。
/// 越界 / 未建 / 还没数据 / 旧世代 → `0xFFFFFFFF`。
/// 「值级播种」（`gi_seed_from_neighbors`）与更新 pass 的「借用候选」共用这一套取邻居逻辑。
fn gi_neighbor_entry(v: vec3<i32>, face: u32, obj: i32, s: u32) -> u32 {
  let a = face >> 1u;
  let a1 = (a + 1u) % 3u;
  let a2 = (a + 2u) % 3u;
  // 轴分量就地构造，避免向量的动态下标（naga 不允许）。
  let sign = 1 - 2 * i32(s % 2u);
  let other = select(a1, a2, s >= 2u);
  var d = vec3<i32>(0, 0, 0);
  if (other == 0u) { d = vec3<i32>(sign, 0, 0); }
  else if (other == 1u) { d = vec3<i32>(0, sign, 0); }
  else { d = vec3<i32>(0, 0, sign); }
  let nv = v + d;
  if (!gi_key_ok(nv)) { return 0xFFFFFFFFu; }
  let idx = gi_lookup(gi_key0(nv), gi_key1(nv, face, obj));
  if (idx == 0xFFFFFFFFu) { return 0xFFFFFFFFu; }
  if (gi_entry_count(idx) <= 0.0) { return 0xFFFFFFFFu; }
  // 旧世代 / 已失效的邻居不能用，否则会把上一世代的值（或方向）带进当前代。
  if (gi_entry_stale(idx)) { return 0xFFFFFFFFu; }
  return idx;
}

/// 新建条目时的**邻面播种**：取同面、同一平面内 4 个邻体素条目的 E 平均值。
/// 两个作用：① 新露出的面第一帧就有平滑可信的值（世界空间缓存不必等自己收敛才"上线"）；
/// ② 相邻体素的值互相关联 —— 没有它，大平面上会看到逐体素斑块。
/// 只取**同一个面**（同法线、同平面）的邻居 ⇒ 不跨薄板、不跨缝隙。
fn gi_seed_from_neighbors(v: vec3<i32>, face: u32, obj: i32) -> vec4<f32> {
  var acc = vec3<f32>(0.0);
  var cnt = 0.0;
  for (var s = 0u; s < 4u; s = s + 1u) {
    let idx = gi_neighbor_entry(v, face, obj, s);
    if (idx == 0xFFFFFFFFu) { continue; }
    let e = idx * GI_CACHE_ENTRY_WORDS + GI_CACHE_W_E;
    acc = acc + vec3<f32>(
      bitcast<f32>(gi_cache[e]),
      bitcast<f32>(gi_cache[e + 1u]),
      bitcast<f32>(gi_cache[e + 2u]),
    );
    cnt = cnt + 1.0;
  }
  if (cnt <= 0.0) { return vec4<f32>(0.0); }
  return vec4<f32>(acc / cnt, 1.0);
}

/// 更新调度的「可见性」记录：用 `gi_seen_claim[idx]` 原子 claim 本帧的写入权，claim 到就 append
/// 条目下标进紧凑列表（`gi_vis_list[atomicAdd(&gi_vis_count[0])]`），供更新 pass 的可见段处理。
/// **每条目每帧恰好 append 一次**：`atomicExchange` 是精确去重（旧版「读 word[18] → 写」在两线程同时
/// 命中同一条目时会都通过 ⇒ 列表出现重复项、浪费预算）。
/// 列表满（`GI_CACHE_VISIBLE_CAPACITY`）即丢弃该 append（claim 已置位，本帧不再重复尝试），
/// 不越界；被丢弃的条目本帧只能靠轮转段覆盖。
fn gi_mark_visible(idx: u32) {
  let frame = gi_u.seq.x;
  if (atomicExchange(&gi_seen_claim[idx], frame) == frame) { return; }
  let slot = atomicAdd(&gi_vis_count[0u], 1u);
  if (slot < GI_CACHE_VISIBLE_CAPACITY) { gi_vis_list[slot] = idx; }
}

/// 着色侧读取：命中返回 `(E, 1)`；未建则**就地建条目**并返回邻面播种值（没有邻面则 `(0, 0)`）。
/// 返回 0 不会造成黑洞：着色侧本就有 `amb·ao` 兜底；也不会漏光，因为播种只取同面同平面的邻面。
/// 条目已存在（无论有没有数据）就记一次可见性：这是「哪些条目正在被看见」的唯一信息源，
/// 更新 pass 的可见段据此把预算花在真正要紧的条目上。
/// **新建条目也必须当帧 claim**：否则它只能等轮转段（`GI_CACHE_SLOTS` / 轮转段线程数 = 32 帧才轮到
/// 一次）才能拿到第一份数据；相机移动时这些新面往往下一帧就不再被读到 ⇒ 永远不被 claim、
/// 永远进不了可见优先的调度，新进入视角的区域收敛极慢（且长期停在无数据状态）。
fn gi_cache_read(v: vec3<i32>, face: u32, obj: i32, pos: vec3<f32>, n: vec3<f32>) -> vec4<f32> {
  if (!gi_key_ok(v)) { return vec4<f32>(0.0); }
  let k0 = gi_key0(v);
  let k1 = gi_key1(v, face, obj);
  let idx = gi_lookup(k0, k1);
  if (idx == 0xFFFFFFFFu) {
    let seed = gi_seed_from_neighbors(v, face, obj);
    let nidx = gi_alloc(k0, k1, pos, n);
    // 新条目立刻进可见列表：`gi_mark_visible` 只依赖 `gi_seen_claim[idx]`，对刚分配的条目成立
    // —— 该槽位本帧还没被 claim 过（claim 至多是上一帧的值），`atomicExchange` 必然 append。
    if (nidx != 0xFFFFFFFFu) { gi_mark_visible(nidx); }
    return seed;
  }
  gi_mark_visible(idx);
  if (gi_entry_count(idx) <= 0.0) { return vec4<f32>(0.0); }
  // 旧世代条目 = 无数据（不显示上一世代的值，由着色侧 amb·ao 兜底）。
  if (gi_entry_stale(idx)) { return vec4<f32>(0.0); }
  let e = idx * GI_CACHE_ENTRY_WORDS + GI_CACHE_W_E;
  return vec4<f32>(
    bitcast<f32>(gi_cache[e]),
    bitcast<f32>(gi_cache[e + 1u]),
    bitcast<f32>(gi_cache[e + 2u]),
    1.0,
  );
}

/// 更新 pass 的二次顶点回读：只查不建（射线命中的面如果没条目就只吃天光，
/// 条目创建权留给屏幕 —— 屏幕驱动缓存边界，覆盖范围自然跟着可见性走）。
fn gi_cache_peek(v: vec3<i32>, face: u32, obj: i32) -> vec3<f32> {
  if (!gi_key_ok(v)) { return vec3<f32>(0.0); }
  let idx = gi_lookup(gi_key0(v), gi_key1(v, face, obj));
  if (idx == 0xFFFFFFFFu) { return vec3<f32>(0.0); }
  if (gi_entry_count(idx) <= 0.0) { return vec3<f32>(0.0); }
  // 旧世代条目 = 无数据（否则会把上一世代的值当二次顶点的入射光注进新世代）。
  if (gi_entry_stale(idx)) { return vec3<f32>(0.0); }
  let e = idx * GI_CACHE_ENTRY_WORDS + GI_CACHE_W_E;
  return vec3<f32>(
    bitcast<f32>(gi_cache[e]),
    bitcast<f32>(gi_cache[e + 1u]),
    bitcast<f32>(gi_cache[e + 2u]),
  );
}
