import package::bindings::{face_slots, gi_out, gi_res_cur, gi_res_prev, gi_u, light_u};
import package::gi::common::{FACE_WORDS, FACE_W_FLAG, FACE_W_K0, FACE_W_K1, FACE_W_PAL, FACE_W_RAW, FACE_W_RAW_N, FACE_W_TEXEL, GI_ACC_CLAMP, GI_ACC_MAX_N, GI_ACC_SCALE, face_slot_of, gi_cos_hemisphere, gi_hist_search_offset, gi_key0, gi_key1, gi_key_ok, gi_mix, gi_pack_dir, gi_rand2, gi_ring_offset, gi_unpack_dir};
import package::gi::consts::{GI_DEN_N_DOT, GI_HIST_SEARCH_TAPS, GI_PI, GI_SS_CAND_N, GI_SS_CAND_N_HQ, GI_SS_M_CAP_K, GI_SS_M_CAP_K_HQ, GI_SS_RAY_BATCH, GI_SS_REUSE_CLAMP_HI, GI_SS_REUSE_CLAMP_LO, GI_SS_REUSE_M_K, GI_SS_REUSE_TAPS, SHADOW_SURFACE_EPS};
import package::gi::denoise::{gi_den_guide_store, gi_den_same_plane, gi_same_surface_n};
import package::gi::ray::{gi_ray_radiance};
import package::trace::{make_grid};
import package::world::{WorldHit, voxel_normal};

// ============================================================================
// 屏幕空间 ReSTIR GI —— **GI 的唯一路径**（逐像素 reservoir，每帧完整估计）。
//
// 业界形态（Ouyang 2021 / UE5.2+ Hardware Lumen）：逐像素一个 reservoir，
// 「新鲜候选 → 时域复用 → 空间复用 → 着色 → 降噪」。本文件的四条关键取舍：
//
//  ① **样本等价 = `gi_den_same_plane`**（面号 + 物体 + 沿面法线轴的体素坐标），**不是**位置 /
//     法线启发式：主命中面键逐位相等 ⇒ 目标函数逐字相同 ⇒ 合并累计量是精确的 RIS 合并；
//     键不同但同平面（切向偏移一格）⇒ 目标函数差一阶小量 ⇒ 合并近似成立，是「相邻面的台阶被
//     抹平」的来源。判据只看键 ⇒ **不受 `voxel_normal` 的梯度法线影响**（凸棱上那是斜向的，
//     任何「法线相似 + 容差」的判据都会在那里静默变味）。
//  ② **空间复用零射线**：跨体素的同平面邻居只**并累计量**，不借方向、不重发射线。
//     旧版让邻居「借代表性方向 + 从本像素重发」并把这个候选并入 RIS 竞争，有两个后果：
//     · 它只在「tap 键不同」时触发 ⇒ 只发生在体素面交界处 ⇒ 交界处被塞进偏亮的候选成一个亮斑；
//     · 代表性样本是按亮度加权抽出来的（不是余弦样本）⇒ 源 pdf 与权重 `π·L` 不匹配 ⇒
//       系统性偏亮，而且这个偏亮的方向会被邻像素逐帧重新选中、在屏幕上漂移
//       （表现为「沿体素边缘的网格亮斑 + 水下光斑式浮动」）。
//     现在**根本不存代表样本**（见 `GiRes`）：空间复用只并累计量，方向恒由本像素当帧的新鲜余弦采样给出。
//  ③ **空间复用的结果绝不回写时域历史**。
//     空间复用读的是**上一帧**邻居的 reservoir；把合并结果存回本像素，下一帧的历史里就混进了
//     邻居的历史 ⇒ 时域链被横向污染（相关性、拖影、移动物体「前沿带」）。
//     所以 `gi_res_store_cur` 只写**新鲜 + 时域**的 `r`，空间合并只作用在本帧的着色估计上。
//  ④ **逐体素平坦不被破坏**：同一个面的所有像素并的是同一组邻居 ⇒ 面内仍是一个值（不产生面内
//     梯度）；被平滑掉的是**相邻面之间**的台阶，这正是「网格亮斑」的来源。
//  ⑤ **时域接受 = 判据 + 搜索**：判据（①）**一个字不改**；落点不通过时在它的 3×3 邻域里继续找
//     （`GI_HIST_SEARCH_TAPS`）—— 修的是"重投影落在哪个 texel"这个**定位**问题，不是放宽几何
//     条件。见 `gi_ss_main` 的 ① 段。
//
//  ⑥ **累加量的前提被破坏 ⇒ 本帧不复用任何历史**（门 = `gi_u.flags.z`，含义「上帧那条累加量在本帧
//     仍然成立」）：时域（① 段）与空间（③ 段）两条复用都要求它为真。两个触发源（都在 Rust 侧判定）：
//     · **遮挡关系变了** —— 判据是"间接光真的依赖的那些几何"变没变（`gi.rs::GiGpu::wide_rev`）：
//       默认（「太阳反弹」关）二次顶点与**任何别的**几何无关 ⇒ 流式挂载 / 卸载、逐体素编辑**都不算**
//       "变了"（它们只改别的体素，本像素的着色照旧）；只有全量重建 / 调色板变化才算。
//       打开「太阳反弹」后二次顶点会发阴影射线 ⇒ 那时回落到"任何上传都算"（`world_rev`）。
//     · **光照阶跃**（主光 / 天光一帧内跳过阈值，或主光方向一帧内转过 60° 以上）。
//     为什么粗门仍然必须存在：reservoir 里存的是 `w_sum` / `M` 这条**累加量**，而它的验证射线只重发
//     **被选中的那一个代表样本**的光路 —— 代表样本没变**不代表**累加量里那些别的样本还有效。
//     典型症状：擦掉一块自发光体素后，接收面上「打到它」的候选**全部**失效，但只要代表样本
//     （例如一条逃向天空的方向）仍然成立，整条累加量就照收 ⇒ 那束光只能靠记忆窗淡出
//     （τ = `GI_SS_M_CAP_K` 帧；相机不动时看不出任何即时变化）。
//     CONSTRAINT: **别把门改成"任何上传都算"**（曾经如此）：流式世界每帧都在挂载 / 卸载 chunk
//     ⇒ `flags.z` 恒 0 ⇒ ① 每个像素走"没有历史"的贵路径（候选数 ×4）；② 时域累积恒不成立
//     （`M` 每帧重来）⇒ 表现就是"噪声严重、静态也不收敛"。实测 720p：`gate_gi` 里约 10 ms 是这一条。
//     光照阶跃是同一回事的另一半：二次顶点缓存被 epoch 作废（新鲜候选算的已经是新光照），而
//     累加量还是旧光照的 ⇒ 两侧时间基准劈开，靠记忆窗淡出会整屏滞后、各像素不同步，且相机一动
//     就借到旧值成大鬼影。逐帧的小变化（自动流逝 / 拖动「时刻」）不触发这条门：那种变化每个窗口帧
//     只有一小步，历史跟着走正是想要的行为。
//     **两个触发源都没触发的帧上这条门恒开** ⇒ 热路径零代价。
//
// 每像素每帧的射线预算（`GI_SS_*`，权威值在 `gi/consts.wesl`）：
//   `GI_SS_CAND_N` 条新鲜余弦候选（最高档改用 `GI_SS_CAND_N_HQ`；去遮挡像素再翻倍；开「重分配」
//   档时再按 `lo`/`hi` 缩放）
//   + 0 条验证射线：世界几何变过就整帧不复用历史（⑥），没变则上帧那条光路与累计量都成立
//     ⇒ 逐样本校验连同它的样本字段与函数一起删掉了（见 `GiRes`）。
//   其中命中几何的射线各带 ≤1 条太阳 NEE 阴影射线。**没有任何空间复用射线**。
//
// 薄板 / 墙缝安全：判据是唯一的 `gi_den_same_plane`（面号 + 物体 + 沿面法线轴的体素坐标），
// 时域与空间都走它；1 格薄板的另一侧面号相反 / 法线轴坐标差一整格 ⇒ 精确拒掉。
// **不用着色法线**：`voxel_normal` 在凸棱上给的是斜向梯度法线（见 `gi/denoise.wesl` 的说明）。
// **MT5-1 放行介质像素后这条判据一字未改**：放行改变的只是「哪些像素写 valid」，键的定义、
// 键的写入与键的比较都没有任何变化 ⇒ 玻璃薄板两侧同样被精确拒掉（见 `gi_ss_main` 的说明）。
// ============================================================================

/// reservoir 字数（Rust 侧从本文件解析同名字面量开 buffer）。
const GI_RES_WORDS: u32 = 8u;
/// word 布局。**判据要用的字排在最前**：`gi_tap_head` 只读前 3 个、`gi_res_head_prev` 读前 4 个
/// 就能判掉绝大多数候选（每少读一个字，直接就是带宽）。
const GI_RES_W_FLAGS: u32 = 0u; // 状态位（见下）
const GI_RES_W_PK: u32 = 1u; // 主命中面键（gi_key0 / gi_key1）
/// 主命中面的**着色法线**（八面体 f16x2）：空间复用与上帧时域接受要判「是不是同一个面」，
/// 相似度那一条用的就是它。`gi_main` 算了**一次**存进来 ⇒ 后续一律读、不重算（那个梯度要探
/// 6 个邻域体素）。
const GI_RES_W_NRM: u32 = 3u;
const GI_RES_W_WSUM: u32 = 4u; // 累计权重 Σ w（rgb = π·L 的和）
const GI_RES_W_M: u32 = 7u; // 候选数 M

/// flags 位：有样本（M > 0）。M = 0 的 reservoir 一律视为无效。
const GI_RES_VALID: u32 = 1u;
/// 主键可编码（体素在 `gi_key_ok` 范围内）⇒ 允许参与复用。
const GI_RES_PK_OK: u32 = 2u;

/// 单个 GI 像素的 reservoir：**累计量 + 判据**，就这些。
/// `w_sum` / `M` 是跨帧累计量（`M` 上限 = `GI_SS_M_CAP_K` × 本帧新鲜候选数），估计就是 `w_sum / M`。
/// 判「同一个面」用的键与着色法线也在里面：下一帧读它做时域接受、邻居读它做空间合并
/// （法线是 `voxel_normal` 的结果，只在 `gi_main` 算一次，下游一律读）。
/// **不存「代表样本」**（方向 / 辐亮度 / 二次顶点键）：那三样只为逐样本校验而存在，而世界一变就
/// 整帧不复用历史（文件头 ⑥）、世界没变时上帧光路又逐位成立 ⇒ 没有消费者（见 `gi/ray.wesl`）。
struct GiRes {
  w_sum: vec3<f32>,
  m: f32,
  pk0: u32,
  pk1: u32,
  flags: u32,
  nrm: vec3<f32>,
}

/// 空 reservoir（天空 / 未命中像素）：`M = 0` ⇒ 下一帧任何时域/空间复用都判为无效，不会读到脏值。
/// 法线给 `+z` 只为避开零向量（它对 `dot` 的结果不构成任何"看起来像同一个面"的假象）。
fn gi_res_none() -> GiRes {
  return GiRes(vec3<f32>(0.0), 0.0, 0u, 0u, 0u, vec3<f32>(0.0, 0.0, 1.0));
}

/// 读**上一帧**的 reservoir（`gi_res_prev`）：累计量 + 判据，**整条 8 字**。
/// 只给**时域复用**用（每像素一次）；空间 tap 走 `gi_tap_head` / `gi_tap_merge`。
fn gi_res_load_prev(idx: u32) -> GiRes {
  let b = idx * GI_RES_WORDS;
  return GiRes(
    vec3<f32>(
      bitcast<f32>(gi_res_prev[b + GI_RES_W_WSUM]),
      bitcast<f32>(gi_res_prev[b + GI_RES_W_WSUM + 1u]),
      bitcast<f32>(gi_res_prev[b + GI_RES_W_WSUM + 2u]),
    ),
    bitcast<f32>(gi_res_prev[b + GI_RES_W_M]),
    gi_res_prev[b + GI_RES_W_PK],
    gi_res_prev[b + GI_RES_W_PK + 1u],
    gi_res_prev[b + GI_RES_W_FLAGS],
    gi_unpack_dir(gi_res_prev[b + GI_RES_W_NRM]),
  );
}

/// 空间复用读 tap 的第一级：`(flags, pk0, pk1)` 三个字（布局里排在最前 ⇒ 连续）。绝大多数 tap
/// 在这里就被判掉（空 / 天空像素 / 键不可编码），不必读累计量那 4 个字 —— 每像素
/// `GI_SS_REUSE_TAPS` 个 tap，少读的字数直接是带宽。
fn gi_tap_head(idx: u32) -> vec3<u32> {
  let b = idx * GI_RES_WORDS;
  return vec3<u32>(gi_res_prev[b + GI_RES_W_FLAGS], gi_res_prev[b + GI_RES_W_PK],
                   gi_res_prev[b + GI_RES_W_PK + 1u]);
}

/// 空间复用读 tap 的第二级（只在并入时）：累计量（rgb = `w_sum`、a = `M`）。4 个字，同样连续。
fn gi_tap_merge(idx: u32) -> vec4<f32> {
  let b = idx * GI_RES_WORDS;
  return vec4<f32>(
    bitcast<f32>(gi_res_prev[b + GI_RES_W_WSUM]),
    bitcast<f32>(gi_res_prev[b + GI_RES_W_WSUM + 1u]),
    bitcast<f32>(gi_res_prev[b + GI_RES_W_WSUM + 2u]),
    bitcast<f32>(gi_res_prev[b + GI_RES_W_M]),
  );
}

/// **历史搜索**读候选的第一级：`(flags, pk0, pk1, nrm)` 四个字 —— 接受判据只需要这四个
/// （`gi_same_surface_n` 的平面归属 + 法线相似），通过之后才读累计量那 4 个字。
/// 与 `gi_tap_head` 同一手法：搜索最多试 9 个候选，省下的字数直接是带宽。
fn gi_res_head_prev(idx: u32) -> vec4<u32> {
  let b = idx * GI_RES_WORDS;
  return vec4<u32>(gi_res_prev[b + GI_RES_W_FLAGS], gi_res_prev[b + GI_RES_W_PK],
                   gi_res_prev[b + GI_RES_W_PK + 1u], gi_res_prev[b + GI_RES_W_NRM]);
}

/// 写**本帧**的 reservoir（`gi_res_cur`；两块 buffer 每帧在 Rust 侧 ping-pong 换绑）。
/// **只允许写「新鲜 + 时域」的结果**（见文件头 ③）：空间复用的合并量绝不经过这里。
fn gi_res_store_cur(idx: u32, r: GiRes) {
  let b = idx * GI_RES_WORDS;
  gi_res_cur[b + GI_RES_W_FLAGS] = r.flags;
  gi_res_cur[b + GI_RES_W_PK] = r.pk0;
  gi_res_cur[b + GI_RES_W_PK + 1u] = r.pk1;
  gi_res_cur[b + GI_RES_W_NRM] = gi_pack_dir(r.nrm);
  gi_res_cur[b + GI_RES_W_WSUM] = bitcast<u32>(r.w_sum.x);
  gi_res_cur[b + GI_RES_W_WSUM + 1u] = bitcast<u32>(r.w_sum.y);
  gi_res_cur[b + GI_RES_W_WSUM + 2u] = bitcast<u32>(r.w_sum.z);
  gi_res_cur[b + GI_RES_W_M] = bitcast<u32>(r.m);
}

/// 并入另一个 reservoir 的**累计量**（`w_sum += w_sum`、`M += M`）。只给时域复用用
/// （调用方先判「同一个面」）⇒ 目标函数近似相同 ⇒ 合并的偏差只落在起点差一格上。
fn gi_res_adopt(r: GiRes, h: GiRes) -> GiRes {
  var out = r;
  out.w_sum = r.w_sum + h.w_sum;
  out.m = r.m + h.m;
  return out;
}

/// M 上限（业界：M ≤ 20 × M_new）：超过就同比例缩回 —— `w_sum / M` 不变（估计不变），
/// 只把跨越的帧数封顶（`m_cap_k` 就是记忆窗的帧数 ⇒ 决定光照 / 几何渐变跟得多快）。
/// 以**本帧新鲜候选数**为基准是必须的：去遮挡帧的候选数翻倍 ⇒ 上限跟着翻倍，否则 boost 打进去的
/// 候选会被立刻压掉、boost 等于白做。
/// `m_cap_k` 由调用方按「降噪质量」档给（`GI_SS_M_CAP_K` / `GI_SS_M_CAP_K_HQ`）——它同时就是记忆帧数。
fn gi_res_cap(r: GiRes, cand_n: u32, m_cap_k: u32) -> GiRes {
  // CONSTRAINT: `cand_n == 0`（本 texel 这一帧**没轮到**发候选：分数预算的轮转、或不是面的认领者）
  // 时窗上限必须**保持不变**，不能按 0 缩放 —— 否则每次轮到"没发候选"就把整条历史清零，
  // 于是"没轮到"的那些帧上报 `has_evidence = false`，逐面均值 / 空间复用 / `dda_main` 的 cov
  // 全部塌成"无数据"。`cand_n >= 1` 时 `m_cap_k * cand_n` 与 `cand_n` 无关（见 `m_cap_k` 的算法），
  // 所以取 `max(..., 1)` 对原来的路径逐位不变。
  let cap = f32(m_cap_k * max(cand_n, 1u));
  if (r.m <= cap) { return r; }
  let shrink = cap / r.m;
  var out = r;
  out.w_sum = r.w_sum * shrink;
  out.m = cap;
  return out;
}

/// 每像素每帧的可复现种子（帧号 + 像素坐标哈希）—— 不用屏幕无关的全局随机。
fn gi_ss_seed(coord: vec2<i32>, frame: u32) -> u32 {
  let ph = gi_mix(u32(coord.x) * 0x9E3779B1u + gi_mix(u32(coord.y) * 0x85EBCA77u + 0x165667B1u));
  return gi_mix(ph + gi_mix(frame * 0x9E3779B9u + 0x27D4EB2Fu));
}

/// 世界点 → **上一帧**的 GI 像素坐标（用 `gi_u.prev_view_proj`；落在视锥 / 屏幕外则 `ok = false`）。
/// 只重投影三维点（不重投影深度）⇒ 不需要上一帧的逆矩阵。
fn gi_ss_reproject(p_world: vec3<f32>, size: vec2<u32>, ok: ptr<function, bool>) -> vec2<i32> {
  let clip = gi_u.prev_view_proj * vec4<f32>(p_world, 1.0);
  if (clip.w <= 1e-6) {
    *ok = false;
    return vec2<i32>(0);
  }
  let ndc = clip.xyz / clip.w;
  if (any(ndc < vec3<f32>(-1.0)) || any(ndc > vec3<f32>(1.0))) {
    *ok = false;
    return vec2<i32>(0);
  }
  let c = vec2<i32>(i32(floor((ndc.x * 0.5 + 0.5) * f32(size.x))),
                    i32(floor((0.5 - ndc.y * 0.5) * f32(size.y))));
  if (c.x < 0 || c.y < 0 || c.x >= i32(size.x) || c.y >= i32(size.y)) {
    *ok = false;
    return vec2<i32>(0);
  }
  *ok = true;
  return c;
}

/// 空间复用的候选 tap 偏移在 `gi/common.wesl::gi_ring_offset`（与历史搜索共用一张 3×3 环表）。

/// 屏幕空间 ReSTIR 的本像素入口（`gi_main` 调用）。
/// 只处理「命中实体体素」的像素；**天空（miss）像素写 valid = 0**（与 `dda_main` 的 miss 分支一致）。
///
/// **MT5-1 · transmissive（玻璃）表面照常产出 GI**：介质像素不再整格判废，而是与不透明表面走
/// **同一条**路径 —— 同一个 `fetch_material` + 同一个 `brdf_reflected`（经 `gi_ray_radiance`）、
/// 同一套导引键（`gi_key0/gi_key1`）。此前判废是为了"介质像素的 GI 无处消费"（`dda_main` 的玻璃
/// 像素着色的是玻璃**后面**的那张不透明面，键对不上），代价是 GI 场在玻璃处留一个洞：那里既没有
/// 自己的值、也没有可借的同面邻居（`gi_cov` 也是 0 ⇒ 玻璃后面的东西按"无 GI 数据"兜天光）。
/// 现在玻璃面自己就是一个合法着色点：它的 `valid`/键/值会被 `dda_main` 的预乘归一化与上采样兜底
/// （`gi_upsample_joint` 的"最近 texel"档）以及降噪的空间 tap 读到，而不是一个洞。
///
/// ⚠️ **显式写明的近似（有意为之，不是"物理正确"）**：玻璃表面的 GI 目前按 **albedo 漫反射** 算
/// —— `brdf_reflected` 里**没有透射项**（没有 BTDF、没有按 IOR 偏折的透射方向、没有介质内的
/// 多次散射）。物理上玻璃的 GI 主要应由**透射**贡献（`main.wesl` 的玻璃像素走的正是
/// `trace_glass` 折射状态机）。本版的取舍：把"透射方向 + 折射后的二次顶点"接进 RIS 需要一整套
/// 新的样本口径（会动到硬约束 2 的键定义），**留待后续**；因此玻璃的 GI 会偏亮（它按
/// `kD·albedo` 漫反射出射），这是本版**有意保留的简化**。
///
/// **复用判据未动**（硬约束 2）：本函数只改「哪些像素有效」，键仍是精确几何面键
/// （命中体素 + 面号 + 物体，且不含任何位置/法线启发式）⇒ 薄板/墙缝不漏光的根本不变。
///
/// **已知的边界行为（既有判据 × MT5-1 的组合，不是漏光）**：`gi_same_surface` 只要求
/// 「**同一平面** + 着色法线相似」⇒ **同一平面内相邻的**玻璃体素与不透明体素会被当作同一个面
/// 互相借样本（与"相邻同平面的不透明体素"是**同一条既有设计**）。它不跨面、不跨格、不跨薄板，
/// 所以不是漏光；代价是"玻璃与不透明墙并排"处会有轻微串色。
fn gi_ss_main(coord: vec2<i32>, size: vec2<u32>, h: WorldHit) {
  let idx = u32(coord.y) * size.x + u32(coord.x);
  let sun_dir = light_u.lights[0].kind_pos_dir.yzw;
  let sun_c = light_u.lights[0].color_intensity.xyz * light_u.lights[0].color_intensity.w;
  // 二次顶点太阳反弹（菜单「渲染/RESTIR GI/太阳反弹」）：0 = 连 NEE 阴影射线都不发。
  let sun_bounce = gi_u.params.x;
  // ② 二次顶点缓存的 epoch（`gi_u.seq.y`）：`gi_face_shade` 的输入变了才自增 ⇒ 槽里的键掩码跟着变、
  // 整张表自失效（跨帧持久，见 `gi/common.wesl`）。**它不在流式世界里每帧变**（见 `gi.rs::ShadeKey`）。
  let epoch = gi_u.seq.y;
  // ④ 第二条弹射的**倍数**（菜单「渲染/光照/二次弹射」，`gi_u.misc.w`）：0 = 关；
  // 1 = 每个候选都带；> 1 = 弹射分裂（每个候选以 `1/本值` 的概率带上、带上时乘本值 ⇒ 期望无偏）。
  let b2_mult = gi_u.misc.w;

  // 只有天空（miss）无效：MT5-1 前这里还要求"首命中非介质"，那条判废已删除（见函数头）。
  if (!h.hit) {
    textureStore(gi_out, coord, vec4<f32>(0.0));
    gi_res_store_cur(idx, gi_res_none());
    // 降噪导引同步清零：整格清零 ⇒ 键为零 ⇒ 降噪与上采样都当作无效跳过。
    gi_den_guide_store(coord, size, false, 0u, 0u, vec3<f32>(0.0, 0.0, 1.0), false, vec2<i32>(0));
    return;
  }

  let gg = make_grid(u32(h.obj_id) + 1u);
  let vc = vec3<f32>(h.voxel) + vec3<f32>(0.5);
  let p_voxel = gg.pos + vec3<f32>(dot(vc, gg.col0), dot(vc, gg.col1), dot(vc, gg.col2)) * gg.scale;
  // CONSTRAINT: GI 候选必须用**与主 pass 同一个着色法线** `voxel_normal`（体素中心的梯度法线），
  // 不能用 DDA 的入射面法线 `h.n`。两者在有起伏的表面上不是一回事：面法线是**逐面**的（台阶的
  // 顶面/侧面各自一个常数方向）⇒ 间接光的半球整片偏转 ⇒ 沿墙面出现与台阶行对齐的**长条**；
  // `voxel_normal` 逐体素连续 ⇒ 与可见着色同向、逐体素。代价：6 次邻域探测。
  let n = voxel_normal(gg, h.voxel, h.face_id, h.n);
  // 沿 n 外推 0.5/max|n| + SHADOW_SURFACE_EPS：与 `dda_main` 同一套外推（起点离开命中体素）。
  let n_off = 0.5 / max(max(abs(n.x), abs(n.y)), max(abs(n.z), 1e-3)) + SHADOW_SURFACE_EPS;
  let org = p_voxel + n * n_off;

  // 主命中面键（唯一的复用判据）+ 是否可编码。
  let pk0 = gi_key0(h.voxel);
  let pk1 = gi_key1(h.voxel, h.face_id, h.obj_id);
  let pk_ok = gi_key_ok(h.voxel);

  let frame = gi_u.seq.x;
  let seed = gi_ss_seed(coord, frame);

  // ---- 降噪导引：面键 + **着色法线** + 上帧重投影像素 ----
  // 重投影用**真实命中点** `h.point`（不是体素中心）：静止相机下它正好映回本像素
  // ⇒ 上帧历史就是本像素自己的（M 才能一路累加）。用体素中心会偏 1~2 像素，导致约 1/3 的像素
  // 每帧都因为「上帧那个像素不是同一个面」而丢掉历史（实测接受率只有 68%）。
  // 法线在这里**只算一次**（`voxel_normal` 要探 6 个邻域体素）：导引、reservoir、历史三处都存它，
  // 下游（降噪、atrous、上采样、`dda_main` 在 gi_div = 1 时）一律读，不再重算。
  var prev_ok = false;
  let prev_pc = gi_ss_reproject(h.point, size, &prev_ok);
  gi_den_guide_store(coord, size, true, pk0, pk1, n, prev_ok, prev_pc);

  // ---- 帧内逐面去重：认领本面（`face_slots`，见 `gi/common.wesl` 的表头注释）----
  // 每个 GI texel 用**自己命中的那个面键**去抢一个槽：抢到的成为该面的"认领者"，负责在
  // `dda_face_main` 里把这一面的着色算一次（结果写回同一槽）；抢不到（槽被别的键占）⇒ 该面
  // 退回逐像素内联着色 —— 撞键只损失收益、不会出错。
  // 判据用 `pk_ok`（键可编码），与查表侧 `dda_main` 的条件一致：越界键会与远处的面撞键。
  // 位置必须在 `gi_den_guide_store` 之后（认领者要被 `dda_face_main` 用 texel 下标找回）。
  if (pk_ok) {
    var face_ok = false;
    let fslot = face_slot_of(pk0, pk1, &face_ok);
    if (face_ok) {
      let fe = fslot * FACE_WORDS;
      if (atomicCompareExchangeWeak(&face_slots[fe + FACE_W_FLAG], 0u, 1u).exchanged) {
        atomicStore(&face_slots[fe + FACE_W_K0], pk0);
        atomicStore(&face_slots[fe + FACE_W_K1], pk1);
        atomicStore(&face_slots[fe + FACE_W_TEXEL], idx);
        atomicStore(&face_slots[fe + FACE_W_PAL], h.pal);
      }
    }
  }

  // ---- ① 时域复用：先判「有没有可用历史」，它决定本帧的新鲜候选数（去遮挡 boost）----
  // 接受判据 = **同一个面**（`gi_den_same_plane` 平面归属 + 着色法线相似，见 `gi/denoise.wesl`）。
  // 为什么不能只要求「键逐位相等」：一个体素面**比一个 texel 还小**时（远处大面、低分辨率档），
  // 相机一动命中的体素逐帧都在换 ⇒ 键几乎帧帧不等 ⇒ 历史被无谓丢掉、M 永远是 1
  // （现象：静止干净、一运动远处大面就起雪花）。而平面归属与距离/分辨率无关；相似度那一条
  // 还顺带把凸棱上「着色平滑、面键却不同」的那一段接上（那正是棱边一圈噪的来源）。
  // 判据放宽后仍要挡"光路已经变了、键却还相等"的情况 —— 那靠 **⑥ 的粗门**（世界几何一变就不复用），
  // 而不是逐样本重发射线校验：reservoir 里只有累计量，没有能代表它的单个样本（见 `GiRes`）。
  // 于是这条链上**一条射线都没有** —— 静态 / 纯相机运动下省掉的验证射线，连同它的样本字段一起删掉了。
  //
  // ---- 历史搜索（`GI_HIST_SEARCH_TAPS`，`gi/common.wesl::gi_hist_search_offset`）----
  // 判据一个字不改，改的是**去哪个像素找**：落点不通过时，在它的 3×3 邻域里按「十字优先、对角
  // 随后」再试几个候选（正中恒排第一 ⇒ 静止相机下行为与「只认落点」逐位一致）。
  // 修的是一个**定位**错误，不是放宽判据：重投影投的是本 texel 的真实命中点，而"包含这个落点的
  // 那个上帧 texel"的射线可能打到**相邻台阶**上（GI 一个 texel 覆盖 4×4 屏幕像素，斜面上一个体素
  // 面常常只占一个 texel）—— 但本面在上一帧确实有人采过，就在落点旁边一格。
  // 收益来自"接上历史"本身：接不上时 `M` 从 1 重来 ⇒ **零时域累积**，本帧 2~4 条射线的原始估计
  // 直接就是画面（降噪侧的历史权重在 M = 1 时为 0）⇒ 这是 σ ÷√M 量级的差别，比加候选数大一个量级。
  var hist = gi_res_none();
  var hist_ok = false;
  // **累加量的前提被破坏 ⇒ 本帧不复用时域历史**（文件头 ⑥）：`flags.z` = 「上帧那条累加量在本帧
  // 仍然成立」（几何没变且不是光照阶跃）。
  // 判据一个字不改 —— 只是累加量的前提不成立时连拿都不拿（代表样本验证不了整条累加量）。
  if (pk_ok && prev_ok && gi_u.flags.z > 0.5) {
    for (var s = 0u; s <= GI_HIST_SEARCH_TAPS; s = s + 1u) {
      let pc = prev_pc + gi_hist_search_offset(s);
      if (pc.x < 0 || pc.y < 0 || pc.x >= i32(size.x) || pc.y >= i32(size.y)) { continue; }
      let nidx = u32(pc.y) * size.x + u32(pc.x);
      // 第一级只读四个字（flags / 键 2 / 法线）：大多数候选在这里退出，不必读累计量那 4 个字。
      let head = gi_res_head_prev(nidx);
      if ((head.x & (GI_RES_VALID | GI_RES_PK_OK)) != (GI_RES_VALID | GI_RES_PK_OK)) { continue; }
      if (!gi_same_surface_n(head.y, head.z, gi_unpack_dir(head.w), pk0, pk1, n)) { continue; }
      // 走到这里必然「世界没变」（见上面的门）⇒ 上帧那条光路仍逐位成立；而 `gi_res_head_prev`
      // 已经判过「键可编码」⇒ 整条收下即可（reservoir 只存累计量与判据，没有逐样本校验的输入）。
      hist = gi_res_load_prev(nidx);
      hist_ok = true;
      break;
    }
  }

  // ---- ② 新鲜候选：`cand_n` 条余弦加权半球方向（无历史 = 去遮挡 ⇒ 候选数翻倍）----
  // 每条走 `gi_ray_radiance`（1 条 GI 射线，命中时含 1 条太阳 NEE 阴影射线）。
  // **候选数按「降噪质量」档给，与分辨率档无关**（见 `gi/consts.wesl` 的算术）：
  // 关/低/中 档 = `GI_SS_CAND_N`，高 档 = `GI_SS_CAND_N_HQ`（射线翻倍，是全链最贵的一项）。
  // 去遮挡帧再翻倍（没有历史可借）。
  // RIS：累计量 `w_sum = Σ π·L`、`M = cand_n`（估计 = w_sum / M = π·⟨L⟩）。
  // 候选方向恒是本像素当帧的新鲜余弦采样 ⇒ 源 pdf 恒为余弦、权重 `π·L` 自洽（见文件头 ②）。
  // 降噪质量档（0 = 关 / 1 = 低 / 2 = 中 / 3 = 高，菜单「渲染/RESTIR GI/降噪质量」）：本 pass 只关心
  // 「是不是最高档」—— 只有最高档把候选数翻倍、记忆窗拉长；时域/atrous 的开关与核半径
  // 都是渲染 pass 的事，由 Rust 按档位决定（本 pass 也照常跑：reservoir 与导引始终要写，
  // 否则「关」档连原始 GI 都没有）。
  let hq = u32(gi_u.flags.w) >= 3u;
  let base_n = select(GI_SS_CAND_N, GI_SS_CAND_N_HQ, hq);
  let cand_n0 = select(base_n, base_n * 2u, !hist_ok);
  // ---- 每帧采样预算分摊（菜单「渲染/RESTIR GI/分帧」，`gi_u.misc.y`）----
  // `share = N`：把 N 帧的采样预算摊成每帧一份 ⇒ 帧时间恒定（不开分帧 = 1，就是原行为）。
  // **窗内样本总数不变**：候选数 ÷N 与记忆窗 ×N 相乘抵消（`gi_res_cap` 的 `cap = m_cap_k × cand_n`）
  // ⇒ 稳态噪声不变，只有响应时间 ×N。每帧至少 1 条（整数条数）⇒ 基准 4 条（低/中档）时 N=8 实际
  // 退化成 N=4；真正 8 倍摊薄需要「降噪质量 = 高」（基准 8 条 ⇒ 1 条/帧）。
  // 方向不重复：种子含逐帧自增的帧号（`gi_ss_seed(coord, gi_u.seq.x)`）⇒ 每帧的候选都是新的独立样本。
  let share = max(u32(gi_u.misc.y), 1u);
  // ---- 预算重分配（菜单「渲染/RESTIR GI/重分配」，`gi_u.misc.z`）----
  // 代理量 = **本帧是否拿到了可用历史**（`hist_ok`，上面刚判出来）：它正是"这个像素的时域累积还在
  // 不在工作"的直接度量。接上了 ⇒ 按 `lo` 倍发（它已经在累积，多发的边际收益低）；没接上 ⇒ 按
  // `hi` 倍发（这种像素每帧都是全新估计，只能靠候选数压方差）。
  //
  // **不要换回"上一帧 reservoir 的 M"**（曾经的写法）：那个代理量在**永远接不上历史**的像素上恒
  // 等于本帧候选数 ⇒ `cand_n ← 4·√(32 / cand_n)` 把自己钉在不动点（≈ 关档的 2 倍，档位表里 4.0
  // 那一档永远够不到）⇒ 从「关」拨到「强」只差 1.41× 的 σ，肉眼看不出。改用 `hist_ok` 之后代理量
  // 与候选数**脱钩**，`lo`/`hi` 才真正生效，也就不再需要靠 `m_cap_k` 去压反馈振荡。
  //
  // 抖动 = `floor(want + dither)`：整数条数会把预算差变成同心环，用逐帧抖动的分数部分摊成噪声。
  let err_tier = u32(gi_u.misc.z);
  var b_err = 1.0;
  if (err_tier > 0u) {
    let lo = select(select(0.70, 0.50, err_tier >= 2u), 0.25, err_tier >= 3u);
    let hi = select(select(1.40, 2.00, err_tier >= 2u), 4.00, err_tier >= 3u);
    b_err = select(hi, lo, hist_ok);
  }
  let dither = gi_rand2(gi_mix(seed + 0x2545F491u)).x;
  var cand_n = max(1u, u32(floor(f32(cand_n0) * b_err / f32(share) + dither)));
  // ---- 把**分数预算**兑现成"少发射线"（否则整数钳制会白烧 N 倍）----
  // WHY：上面的 `max(1u, …)` 把预算直接舍成"每 texel 每帧 1 条"。而「降噪质量 = 高」（基准 8 条）
  // 配「分帧 ÷8」的本意是 **1 条/texel/8 帧**，实际却按 1 条/帧 发 ⇒ 白烧 8 倍射线（文件上面
  // 「整数钳制」那段算术自己就记着这个退化）。这里按**逐帧轮转**抽样兑现分数：预算是 `f` 条/texel/帧
  // 时，每帧只让约 `1/stride` 的 texel 发候选，其余 texel 照常走**时域累积**（零射线）与**空间复用**
  // （读邻居、零射线）⇒ 帧时间 ÷stride，而 8 帧窗口内的总样本数不变（与「分帧」的语义一致），
  // 只有光照响应随 N 变慢。抽样用 `idx` 与帧号混合 ⇒ 相邻 texel 不同步（不会出现整块屏幕同时变钝的环）。
  if (cand_n == 1u) {
    let frac = f32(cand_n0) * b_err / f32(share);
    if (frac < 0.999) {
      let stride = max(u32(ceil(1.0 / max(frac, 1.0 / 64.0))), 1u);
      if (gi_mix(idx ^ (gi_u.seq.x * 0x9E3779B9u)) % stride != 0u) {
        cand_n = 0u;
      }
    }
  }
  // ---- 候选是否只在"面的认领者"上发射 ----
  //
  // `0` = 每个 texel 按自己的分数预算发候选（**逐面平均才有东西可平均**）；`1` = 只有认领者发。
  //
  // WHY 默认 0：`gi_face_flatten` 的降噪前提是"同一个面的 N 个 texel 各自是**同一个积分的独立估计**
  // ⇒ 面内平均把方差 ÷N"。开 1 时一个体素面只剩认领者那 1 条样本，`÷N` 永远不发生；而且非认领者
  // 的时域状态留空 ⇒ 它们只能靠 flatten 摊值，一旦 flatten 因撞键（槽被别的键占着）跳过本面，
  // 面上就是"1/4 的 texel 有数据、3/4 没有"的棋盘 ⇒ 逐体素斑驳死点（`a` / cov 在面内不一致）。
  //
  // CONSTRAINT: 开 1 的代价与收益都要重新量 —— 它把每条面的射线数降到 1/N（N = 面覆盖的 texel 数，
  // 1/1 分辨率下约 16、1/2 下约 4），换来 N 倍的候选开销。GI 现在只值 ~1 ms（GPU 逐 pass 共 ~5 ms、
  // 整帧预算 16.7 ms），所以先按"质量优先"取 0；要省射线再改回 1 并把噪声一并量出来。
  const GI_SS_CLAIM_ONLY: u32 = 0u;
  if (GI_SS_CLAIM_ONLY != 0u && pk_ok && cand_n > 0u) {
    var fok2 = false;
    let fs2 = face_slot_of(pk0, pk1, &fok2);
    if (fok2) {
      let fe2 = fs2 * FACE_WORDS;
      if (atomicLoad(&face_slots[fe2 + FACE_W_FLAG]) != 0u
          && atomicLoad(&face_slots[fe2 + FACE_W_K0]) == pk0
          && atomicLoad(&face_slots[fe2 + FACE_W_K1]) == pk1
          && atomicLoad(&face_slots[fe2 + FACE_W_TEXEL]) != idx) {
        cand_n = 0u;
      }
    }
  }
  // 记忆窗上限：按"每帧实际候选数相对基准缩小了几倍"放大。减发一侧 ⇒ 窗内样本数 = 基准值
  // （稳态噪声不变、只有响应变慢）；增发一侧按本式取 max(...,1) ⇒ 窗内样本数随之变大。
  // （本项零性能成本，但窗越长光照响应越钝 ⇒ 这就是分摊/重分配的代价。）
  let m_cap_k =
    select(GI_SS_M_CAP_K, GI_SS_M_CAP_K_HQ, hq) * max(cand_n0 / max(cand_n, 1u), 1u);
  var r = gi_res_none();
  // **ILP 批量**：批内 `GI_SS_RAY_BATCH` 条射线只差方向、互相独立 ⇒ 先把它们全发出去（多条层次
  // DDA 的依赖 load 同时在飞），再按原本的 `k` 顺序做累计与蓄水池更新。方向序列、样本与比较顺序
  // 都与逐条写法完全一致 ⇒ 结果**逐位相同**（`gi_ray_radiance` 无副作用：它对 `gi_sec_slots` 的
  // 填充是写同值的幂等操作，只影响性能不影响值）。
  // 批大小取 1 即退化成逐条；分帧把 `cand_n` 降到 1~2 时最后一批不满（`nb`），按实际条数跑。
  var k = 0u;
  loop {
    if (k >= cand_n) { break; }
    let nb = min(GI_SS_RAY_BATCH, cand_n - k);
    var b_s: array<vec3<f32>, GI_SS_RAY_BATCH>;
    for (var i = 0u; i < nb; i = i + 1u) {
      let kk = k + i;
      let cseed = gi_mix(seed + kk * 0x9E3779B9u + 1u);
      let dir = gi_cos_hemisphere(n, gi_rand2(cseed));
      // ④ **弹射分裂的抽签**：每个候选独立地以 `1/b2_mult` 的概率带上第二条弹射，带上时把它乘
      // `b2_mult` ⇒ 期望值不变（无偏），只有方差变大。**必须逐候选随机**（不能用 `kk % mult` 这种
      // 确定性规则）：分帧档下 `cand_n` 可能只有 1 条，那时确定性规则会让每一帧的唯一候选都命中
      // ⇒ 时域累积到的是一个恒为 `mult ×` 的值，直接变成系统性偏差。
      // 抽签用同一批哈希（`gi_rand2` 的第二个分量），不另造随机源；种子含帧号 ⇒ 逐帧独立。
      let b2 = select(0.0, b2_mult, gi_rand2(gi_mix(cseed + 0x27D4EB2Fu)).y < 1.0 / max(b2_mult, 1.0));
      b_s[i] = gi_ray_radiance(org, dir, sun_dir, sun_c, sun_bounce, epoch, b2, cseed);
    }
    for (var i = 0u; i < nb; i = i + 1u) {
      // 累计量就是全部：辐亮度估计 = `w_sum / M`。
      r.w_sum = r.w_sum + b_s[i] * GI_PI;
      r.m = r.m + 1.0;
    }
    k = k + nb;
  }

  // ---- ④ 时域合并 + 写**时域** reservoir（空间复用的合并量绝不写回，见文件头 ③）----
  // CONSTRAINT: 必须排在 ③ 之前 —— ③ 的钳制锚是「本像素当帧的估计」，含时域历史才干净；
  // 只含本帧那几条新鲜候选时锚自身就是噪声，拿它去钳邻居的长历史会把锚的偏倚带进画面。
  r.pk0 = pk0;
  r.pk1 = pk1;
  r.flags = GI_RES_VALID | select(0u, GI_RES_PK_OK, pk_ok);
  r.nrm = n;
  if (hist_ok) { r = gi_res_adopt(r, hist); }
  r = gi_res_cap(r, cand_n, m_cap_k);
  gi_res_store_cur(idx, r);

  // ---- ③ 空间复用：读**上一帧**邻居的 reservoir 只并累计量（零射线），只影响本帧估计（见 ④）----
  // tap 的中心 = 本帧命中点上帧的重投影位置（`prev_pc`），不是本像素：读上帧 reservoir 时
  // 像素 `coord + offset` 上帧看到的是「上帧恰好落在那里的那个面」，相机一平移就完全不是同一个面
  // ⇒ 命中率掉到 0。把中心放到「上帧本面所在的屏幕位置」，读到的才是**本面上帧的邻居**。
  // 静止相机下 `prev_pc` 恒等于本像素 ⇒ 与朴素做法逐位一致。
  // 两种 tap 都并累计量（`w_sum` / `M`），判据是唯一的 `gi_same_surface`：
  //  a) **同键**：主命中面键逐位相等 ⇒ 目标函数逐字相同 ⇒ 合并是纯降噪、零偏差；
  //  b) **同一个面**（面号/物体/法线轴坐标相等 + 着色法线相似）：cosθ 几乎相同、平面相同，
  //     只有射线起点差一格 ⇒ 目标函数差一个一阶小量；这是「相邻面的台阶」「凸棱上倾斜的那一段」
  //     被抹平、网格亮斑与棱边噪声消失的原因。薄板另一侧 / 隔一格的平行面仍被精确拒掉。
  //
  // **重投影失败（`prev_ok = false`）⇒ 本帧不做空间复用**：这时手里没有"本面上帧在屏幕上的位置"
  // 这个锚（`prev_pc` 无意义），退回**本像素坐标**读邻居等于借一个**完全无关的世界点**的上帧累加量。
  // 而"重投影失败"恰恰是相机前进/后退时大批出现的一类像素（命中点被推到视锥外 / 落进相机背后，
  // 新入屏的轮廓像素也是这一类）⇒ 那条回退是"靠近 / 远离时画面被上帧的屏幕内容拖住"的一个直接来源。
  // 关掉它只损失这一小撮像素的空间平滑（它们本来也在由时域与 atrous 兜），**没有射线代价**
  // （`hist_ok` 不受影响 ⇒ 不会触发去遮挡候选翻倍）。
  let tb = prev_pc;
  var sk_sum = vec3<f32>(0.0);
  var sk_m = 0.0;
  // **累加量的前提不成立 ⇒ 本帧不复用空间邻域**（文件头 ⑥）：tap 读的是**上一帧**邻居的 reservoir，
  // 那些累加量同样可能已经失效（同一个理由：代表样本验证不了它）。
  // `r.m < 0.5`：本帧既没发候选（分数预算把本 texel 排到了别的帧）**又**没接上历史 ⇒ 手里没有
  // "本位置"的当帧证据，`est_self` 会是 0，拿它当钳制锚会把邻居的累加量全拉向 0 ⇒ 这一帧不做空间复用
  // （零射线，只是少一次平滑；下一帧轮到本 texel 发候选时就恢复）。
  let reuse_taps =
    select(GI_SS_REUSE_TAPS, 0u, !prev_ok || gi_u.flags.z <= 0.5 || r.m < 0.5);
  // 借来的值要锚在「本像素当帧自己的估计」附近（`GI_SS_REUSE_CLAMP_*`）：`w_sum / M`，
  // **含**时域合并（排在上面的 ④）、**不含**借来的量 ⇒ 它就是"我这个位置"的当帧证据，也是唯一的锚。
  let est_self = r.w_sum / max(r.m, 1e-6);
  for (var t = 0u; t < reuse_taps; t = t + 1u) {
    let nc = tb + gi_ring_offset(t);
    if (nc.x < 0 || nc.y < 0 || nc.x >= i32(size.x) || nc.y >= i32(size.y)) { continue; }
    let nidx = u32(nc.y) * size.x + u32(nc.x);
    let head = gi_tap_head(nidx);
    if ((head.x & (GI_RES_VALID | GI_RES_PK_OK)) != (GI_RES_VALID | GI_RES_PK_OK)) { continue; }
    // 平面归属先判（只读键）⇒ 大多数 tap 在这里就退出，不必再读法线那一个字。
    if (!gi_den_same_plane(head.y, head.z, pk0, pk1)) { continue; }
    if (dot(gi_unpack_dir(gi_res_prev[nidx * GI_RES_WORDS + GI_RES_W_NRM]), n) < GI_DEN_N_DOT) {
      continue;
    }
    let wm = gi_tap_merge(nidx);
    // `M` 是有效位的正数（上面已判 VALID）⇒ 直接并；`w_sum` 与 `M` 同步缩放过的累计量，
    // 比值恒是「该面 π·⟨L⟩ 的滑动平均」⇒ 并进来只是加权平均，不改变估计的量纲。
    // **值先钳到本像素估计的 `[LO, HI]` 倍内**（`GI_SS_REUSE_CLAMP_*` 的说明）：判据只证明"同一个面"、
    // 不证明"同一个位置"，钳制把"借来的位置差异"切掉，只保留"借来的证据量"。
    let v = wm.xyz / max(wm.w, 1e-6);
    let vc = clamp(v, est_self * GI_SS_REUSE_CLAMP_LO, est_self * GI_SS_REUSE_CLAMP_HI);
    sk_sum = sk_sum + vc * wm.w;
    sk_m = sk_m + wm.w;
  }

  // ---- ⑤ 着色：估计 = w_sum / M = π·⟨L⟩ ⇒ 乘增益、除 π；空间复用只加在这一步 ----
  // 与 ③ 的钳制锚 `est_self` 是同一个量（`w_sum / M`）：钳制就是把借来的值拉回它附近。
  var est = est_self;
  if (sk_m > 0.0) {
    // **借来的权重不超过我们自己的**（`GI_SS_REUSE_M_K`，默认 `1.0` ⇒ 最多各占一半）。
    // 空间复用的目的是"抹平相邻面之间的台阶"，**不是**用邻居的长历史替换本像素的估计：
    // 邻居的 `M` 上限与我们相同，而 8 个 tap 合计可达它的数倍 ⇒ 不设限时本帧几条新鲜样本被完全
    // 淹没，画面被"上一帧 + 屏幕偏移一格"的值支配（诊断实测：只接受逐位相同的键时拖影基本消失）。
    // `sk_sum` / `sk_m` 同比例缩 ⇒ 邻居的估计值不变，只压它的权重。
    let keep = min(1.0, GI_SS_REUSE_M_K * r.m / max(sk_m, 1e-6));
    let denom = max(r.m + sk_m * keep, 1e-6);
    est = (r.w_sum + sk_sum * keep) / denom;
  }
  let gi = est * (gi_u.params.z / GI_PI);
  // 本 texel 的**当帧证据**：发了候选（`r.m`）或从时域 / 空间邻居借到了量（`sk_m`）。
  // 「候选只在面的认领者上发射」（见上方那条闸门）⇒ 一个面里**只有认领者**有证据，其余 texel 在
  // 接上历史之前 `est` 恒为 0。
  let has_evidence = (r.m + sk_m) > 0.0;
  // CONSTRAINT: 没有证据的 texel **必须写 `a = 0`**，不能写"值是 0 却报有效"：
  //   · `dda_main` 把 `a` 当 cov —— `cov = 1` 会把天光乘 `GI_AMBIENT_FLOOR` 压暗 ⇒ 无数据处被渲染成
  //     "天光被压暗 **且** GI = 0" = 全黑（症状：自发光完全照不亮任何体素）；
  //   · `gi_face_flatten` / `dda_face_accum` 的逐面均值按 `a > 0` 取参与集合 ⇒ 把 0 混进去会把
  //     面均值稀释成 0、整片 GI 归零。
  // 覆盖度 cov 与 valid 是同一个场（旧实现单开一张 `gi_cov_out` 写的恒是「valid ? 1 : 0」），
  // 所以这里只写这一张、`dda_main` 从 `a` 自取。
  textureStore(gi_out, coord, vec4<f32>(gi, select(0.0, 1.0, has_evidence)));

  // ---- ① 逐面合并：把本 texel 的估计定点累加到该面的槽（`gi_face_flatten` 事后取面均值）----
  // 同一个面的所有 texel 的估计是**同一个积分**的独立估计 ⇒ 面内平均是纯降噪（方差 ÷ texel 数）。
  // 只有"槽已被本面认领、且两个键 word 逐位相等"才并：撞键 / 抢在认领之前到达的 texel 一律跳过
  // （那条路退化成逐 texel，只损失收益、不会污染别的面）。
  // 定点 + CAS 预留计数的手法与 `dda_face_accum` 逐字一致（`GI_ACC_*` 的溢出账也共用）。
  // `has_evidence`：无证据的 texel **不参与**逐面均值（它的 `est` 是 0，"没有数据"不等于"值是 0"，
  // 混进去会把面均值拉向 0）—— 与 `gi_face_flatten` / `dda_face_accum` 的 `a > 0` 参与集合同一条口径。
  if (pk_ok && has_evidence) {
    var fok = false;
    let fslot = face_slot_of(pk0, pk1, &fok);
    if (fok) {
      let fe = fslot * FACE_WORDS;
      if (atomicLoad(&face_slots[fe + FACE_W_FLAG]) != 0u
          && atomicLoad(&face_slots[fe + FACE_W_K0]) == pk0
          && atomicLoad(&face_slots[fe + FACE_W_K1]) == pk1) {
        // 归一化 + 定点取整（进位到最近，避免系统性偏暗）—— 与 `dda_face_accum` 同一句。
        let q = clamp(gi, vec3<f32>(0.0), vec3<f32>(GI_ACC_CLAMP));
        let qv = vec3<u32>(round(q * GI_ACC_SCALE));
        var reserved = false;
        loop {
          let c = atomicLoad(&face_slots[fe + FACE_W_RAW_N]);
          if (c + 1u > GI_ACC_MAX_N) { break; }
          if (atomicCompareExchangeWeak(&face_slots[fe + FACE_W_RAW_N], c, c + 1u).exchanged) {
            reserved = true;
            break;
          }
        }
        if (reserved) {
          atomicAdd(&face_slots[fe + FACE_W_RAW], qv.x);
          atomicAdd(&face_slots[fe + FACE_W_RAW + 1u], qv.y);
          atomicAdd(&face_slots[fe + FACE_W_RAW + 2u], qv.z);
        }
      }
    }
  }
}