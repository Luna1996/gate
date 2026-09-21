import package::bindings::{face_slots, gi_out, gi_res_cur, gi_res_prev, gi_u, light_u};
import package::gi::common::{FACE_WORDS, FACE_W_FLAG, FACE_W_K0, FACE_W_K1, FACE_W_PAL, FACE_W_TEXEL, face_slot_of, gi_cos_hemisphere, gi_key0, gi_key1, gi_key_ok, gi_mix, gi_pack_dir, gi_rand2, gi_unpack_dir};
import package::gi::consts::{GI_DEN_N_DOT, GI_LUMA, GI_PI, GI_SS_CAND_N, GI_SS_CAND_N_HQ, GI_SS_ERR_M_REF, GI_SS_M_CAP_K, GI_SS_M_CAP_K_HQ, GI_SS_RAY_BATCH, GI_SS_REUSE_TAPS, SHADOW_SURFACE_EPS};
import package::gi::denoise::{gi_den_guide_store, gi_den_same_plane, gi_same_surface_n};
import package::gi::ray::{gi_ray_radiance, gi_ray_secondary_key};
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
//     现在代表样本恒来自**本像素的新鲜余弦采样**（源 pdf 恒为余弦，`π·L` 才是正确权重）。
//  ③ **空间复用的结果绝不回写时域历史**。
//     空间复用读的是**上一帧**邻居的 reservoir；把合并结果存回本像素，下一帧的历史里就混进了
//     邻居的历史 ⇒ 时域链被横向污染（相关性、拖影、移动物体「前沿带」）。
//     所以 `gi_res_store_cur` 只写**新鲜 + 时域**的 `r`，空间合并只作用在本帧的着色估计上。
//  ④ **逐体素平坦不被破坏**：同一个面的所有像素并的是同一组邻居 ⇒ 面内仍是一个值（不产生面内
//     梯度）；被平滑掉的是**相邻面之间**的台阶，这正是「网格亮斑」的来源。
//
// 每像素每帧的射线预算（`GI_SS_*`，权威值在 `gi/consts.wesl`）：
//   `GI_SS_CAND_N` 条新鲜余弦候选（最高档改用 `GI_SS_CAND_N_HQ`；去遮挡像素再翻倍）
//   + ≤1 条时域二次顶点验证射线（世界几何逐位未变时整条省掉）
//   其中命中几何的射线各带 ≤1 条太阳 NEE 阴影射线。**没有任何空间复用射线**。
//
// 薄板 / 墙缝安全：判据是唯一的 `gi_den_same_plane`（面号 + 物体 + 沿面法线轴的体素坐标），
// 时域与空间都走它；1 格薄板的另一侧面号相反 / 法线轴坐标差一整格 ⇒ 精确拒掉。
// **不用着色法线**：`voxel_normal` 在凸棱上给的是斜向梯度法线（见 `gi/denoise.wesl` 的说明）。
// **MT5-1 放行介质像素后这条判据一字未改**：放行改变的只是「哪些像素写 valid」，键的定义、
// 键的写入与键的比较都没有任何变化 ⇒ 玻璃薄板两侧同样被精确拒掉（见 `gi_ss_main` 的说明）。
// ============================================================================

/// reservoir 字数（Rust 侧从本文件解析同名字面量开 buffer）。
const GI_RES_WORDS: u32 = 14u;
/// word 布局。
const GI_RES_W_DIR: u32 = 0u; // 选中样本方向（八面体 f16x2）
const GI_RES_W_RAD: u32 = 1u; // 选中样本的入射辐亮度 L（rgb）
const GI_RES_W_WSUM: u32 = 4u; // 累计权重 Σ w（rgb = π·L 的和）
const GI_RES_W_M: u32 = 7u; // 候选数 M
const GI_RES_W_PK: u32 = 8u; // 主命中面键（gi_key0 / gi_key1）
const GI_RES_W_SK: u32 = 10u; // 选中样本的二次顶点键（gi_key0 / gi_key1）
const GI_RES_W_FLAGS: u32 = 12u; // 状态位（见下）
/// 主命中面的**着色法线**（八面体 f16x2）：空间复用与上帧时域接受要判「是不是同一个面」，
/// 相似度那一条用的就是它。`gi_main` 算了**一次**存进来 ⇒ 后续一律读、不重算（那个梯度要探
/// 6 个邻域体素）。
const GI_RES_W_NRM: u32 = 13u;
/// flags 位：有样本（M > 0）。M = 0 的 reservoir 一律视为无效。
const GI_RES_VALID: u32 = 1u;
/// 主键可编码（体素在 `gi_key_ok` 范围内）⇒ 允许参与复用。
const GI_RES_PK_OK: u32 = 2u;
/// 二次顶点命中（false = 逃逸到天空，键取 (0,0)）。
const GI_RES_SK_HIT: u32 = 4u;
/// 二次顶点键可比较（`gi_key_ok` 或天空）⇒ 只有它置位才允许做二次顶点键的相等判定。
const GI_RES_SK_OK: u32 = 8u;

/// 单个 GI 像素的 reservoir。
/// `w_sum` / `M` 是**跨帧累计量**（`M` 上限 = `GI_SS_M_CAP_K` × 本帧新鲜候选数）；
/// `dir`/`rad`/`sk*` 是被选中的那个样本（本像素最近一次的新鲜余弦采样，供时域二次顶点验证用；
/// 估计只用 `w_sum / M`，所以代表样本抽得偏亮**不会**让估计偏亮）。
/// 判「同一个面」用的键与**着色法线**都在这里：下一帧读它做时域接受、邻居读它做空间合并
/// （法线是 `voxel_normal` 的结果，只在 `gi_main` 算一次，下游一律读）。
struct GiRes {
  dir: vec3<f32>,
  rad: vec3<f32>,
  w_sum: vec3<f32>,
  m: f32,
  pk0: u32,
  pk1: u32,
  sk0: u32,
  sk1: u32,
  flags: u32,
  nrm: vec3<f32>,
}

/// 空 reservoir（天空 / 未命中像素）：`M = 0` ⇒ 下一帧任何时域/空间复用都判为无效，不会读到脏值。
fn gi_res_none() -> GiRes {
  return GiRes(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0), 0.0, 0u, 0u, 0u, 0u, 0u,
               vec3<f32>(0.0, 0.0, 1.0));
}

/// 读**上一帧**的 reservoir（`gi_res_prev`）。只给**时域复用**用（每像素一次，需要 dir/sk）。
fn gi_res_load_prev(idx: u32) -> GiRes {
  let b = idx * GI_RES_WORDS;
  return GiRes(
    gi_unpack_dir(gi_res_prev[b + GI_RES_W_DIR]),
    vec3<f32>(
      bitcast<f32>(gi_res_prev[b + GI_RES_W_RAD]),
      bitcast<f32>(gi_res_prev[b + GI_RES_W_RAD + 1u]),
      bitcast<f32>(gi_res_prev[b + GI_RES_W_RAD + 2u]),
    ),
    vec3<f32>(
      bitcast<f32>(gi_res_prev[b + GI_RES_W_WSUM]),
      bitcast<f32>(gi_res_prev[b + GI_RES_W_WSUM + 1u]),
      bitcast<f32>(gi_res_prev[b + GI_RES_W_WSUM + 2u]),
    ),
    bitcast<f32>(gi_res_prev[b + GI_RES_W_M]),
    gi_res_prev[b + GI_RES_W_PK],
    gi_res_prev[b + GI_RES_W_PK + 1u],
    gi_res_prev[b + GI_RES_W_SK],
    gi_res_prev[b + GI_RES_W_SK + 1u],
    gi_res_prev[b + GI_RES_W_FLAGS],
    gi_unpack_dir(gi_res_prev[b + GI_RES_W_NRM]),
  );
}

/// 空间复用读 tap 的第一级：`(flags, pk0, pk1)` 三个字。绝大多数 tap 在这里就被判掉
/// （空 / 天空像素 / 键不可编码），不必读整条 14 字 —— 每像素 `GI_SS_REUSE_TAPS` 个 tap，
/// 少读的字数直接是带宽。
fn gi_tap_head(idx: u32) -> vec3<u32> {
  let b = idx * GI_RES_WORDS;
  return vec3<u32>(gi_res_prev[b + GI_RES_W_FLAGS], gi_res_prev[b + GI_RES_W_PK],
                   gi_res_prev[b + GI_RES_W_PK + 1u]);
}

/// 空间复用读 tap 的第二级（只在并入时）：累计量（rgb = `w_sum`、a = `M`）。4 个字。
fn gi_tap_merge(idx: u32) -> vec4<f32> {
  let b = idx * GI_RES_WORDS;
  return vec4<f32>(
    bitcast<f32>(gi_res_prev[b + GI_RES_W_WSUM]),
    bitcast<f32>(gi_res_prev[b + GI_RES_W_WSUM + 1u]),
    bitcast<f32>(gi_res_prev[b + GI_RES_W_WSUM + 2u]),
    bitcast<f32>(gi_res_prev[b + GI_RES_W_M]),
  );
}

/// 写**本帧**的 reservoir（`gi_res_cur`；两块 buffer 每帧在 Rust 侧 ping-pong 换绑）。
/// **只允许写「新鲜 + 时域」的结果**（见文件头 ③）：空间复用的合并量绝不经过这里。
fn gi_res_store_cur(idx: u32, r: GiRes) {
  let b = idx * GI_RES_WORDS;
  gi_res_cur[b + GI_RES_W_DIR] = gi_pack_dir(r.dir);
  gi_res_cur[b + GI_RES_W_RAD] = bitcast<u32>(r.rad.x);
  gi_res_cur[b + GI_RES_W_RAD + 1u] = bitcast<u32>(r.rad.y);
  gi_res_cur[b + GI_RES_W_RAD + 2u] = bitcast<u32>(r.rad.z);
  gi_res_cur[b + GI_RES_W_WSUM] = bitcast<u32>(r.w_sum.x);
  gi_res_cur[b + GI_RES_W_WSUM + 1u] = bitcast<u32>(r.w_sum.y);
  gi_res_cur[b + GI_RES_W_WSUM + 2u] = bitcast<u32>(r.w_sum.z);
  gi_res_cur[b + GI_RES_W_M] = bitcast<u32>(r.m);
  gi_res_cur[b + GI_RES_W_PK] = r.pk0;
  gi_res_cur[b + GI_RES_W_PK + 1u] = r.pk1;
  gi_res_cur[b + GI_RES_W_SK] = r.sk0;
  gi_res_cur[b + GI_RES_W_SK + 1u] = r.sk1;
  gi_res_cur[b + GI_RES_W_FLAGS] = r.flags;
  gi_res_cur[b + GI_RES_W_NRM] = gi_pack_dir(r.nrm);
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
/// 只抑制历史对代表样本抽样的支配。以**本帧新鲜候选数**为基准是必须的：
/// 去遮挡帧的候选数翻倍 ⇒ 上限跟着翻倍，否则 boost 打进去的候选会被立刻压掉、boost 等于白做。
/// `m_cap_k` 由调用方按「降噪质量」档给（`GI_SS_M_CAP_K` / `GI_SS_M_CAP_K_HQ`）——它同时就是记忆帧数。
fn gi_res_cap(r: GiRes, cand_n: u32, m_cap_k: u32) -> GiRes {
  let cap = f32(m_cap_k * cand_n);
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

/// 空间复用的候选 tap 偏移（上帧本面所在位置的 3×3 环）。
fn gi_ss_tap_offset(s: u32) -> vec2<i32> {
  if (s == 0u) { return vec2<i32>(1, 0); }
  if (s == 1u) { return vec2<i32>(-1, 0); }
  if (s == 2u) { return vec2<i32>(0, 1); }
  if (s == 3u) { return vec2<i32>(0, -1); }
  if (s == 4u) { return vec2<i32>(1, 1); }
  if (s == 5u) { return vec2<i32>(1, -1); }
  if (s == 6u) { return vec2<i32>(-1, 1); }
  return vec2<i32>(-1, -1);
}

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
  // 判据放宽后仍接一条重连接射线（≤1）做二次顶点键验证：它校验的是「这个样本描述的光路还在不在」。
  // **世界几何自写入历史那一帧起逐位未变时（uniform `flags.z`）这条验证射线整条省掉** ——
  // 输入完全相同 ⇒ 结果必然逐位相同，这是静态 / 纯相机运动下最大的一笔省。
  var hist = gi_res_none();
  var hist_ok = false;
  if (pk_ok && prev_ok) {
    let cand = gi_res_load_prev(u32(prev_pc.y) * size.x + u32(prev_pc.x));
    if ((cand.flags & (GI_RES_VALID | GI_RES_PK_OK)) == (GI_RES_VALID | GI_RES_PK_OK)
        && gi_same_surface_n(cand.pk0, cand.pk1, cand.nrm, pk0, pk1, n)) {
      var same_sk = false;
      if (gi_u.flags.z > 0.5) {
        same_sk = (cand.flags & GI_RES_SK_OK) != 0u;
      } else {
        let s_r = gi_ray_secondary_key(org, cand.dir);
        same_sk = s_r.sk_hit == ((cand.flags & GI_RES_SK_HIT) != 0u)
          && s_r.sk_ok && s_r.sk0 == cand.sk0 && s_r.sk1 == cand.sk1;
      }
      if (same_sk) {
        hist = cand;
        hist_ok = true;
      }
    }
  }

  // ---- ② 新鲜候选：`cand_n` 条余弦加权半球方向（无历史 = 去遮挡 ⇒ 候选数翻倍）----
  // 每条走 `gi_ray_radiance`（1 条 GI 射线，命中时含 1 条太阳 NEE 阴影射线）。
  // **候选数按「降噪质量」档给，与分辨率档无关**（见 `gi/consts.wesl` 的算术）：
  // 关/低/中 档 = `GI_SS_CAND_N`，高 档 = `GI_SS_CAND_N_HQ`（射线翻倍，是全链最贵的一项）。
  // 去遮挡帧再翻倍（没有历史可借）。
  // RIS：累计量 `w_sum = Σ π·L`、`M = cand_n`（估计 = w_sum / M = π·⟨L⟩）。
  // 代表样本 = **只在这一组余弦候选里**做加权蓄水池抽样（源 pdf 恒是余弦 ⇒ 权重 `π·L` 才自洽；
  // 见文件头 ②：绝不让空间复用进来的方向参与这个竞争）。
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
  // ---- 误差驱动重分配（菜单「渲染/RESTIR GI/采样重分配」，`gi_u.misc.z`）----
  // 代理量 = **上一帧 reservoir 的样本数 M**（`gi_res_prev`，已绑定，零新通道）：它是"这个像素
  // 已经有多干净"的直接度量 —— 与场景尺度无关（计数不是辐亮度）、天然时间平滑（它就是累积量）、
  // 且去遮挡/键变化时会被 `gi_ss_reproject` 的判据重置 ⇒ 自动"哪里脏补哪里"。
  // 收敛的像素（M 大）少发新射线、刚变脏的（M 小）多发；`m_cap_k` 仍按 `cand_n0 / cand_n` 同步放大
  // ⇒ **窗内样本数不变**：噪声不变，只有响应时间随"干不干净"变（收敛的变慢、刚脏的变快），
  // 因此不会出现"少采样→更脏→要更多采样"的分配振荡。
  // 抖动 = `floor(want + dither)`：整数条数会把预算差变成同心环，用逐帧抖动的分数部分摊成噪声。
  let err_tier = u32(gi_u.misc.z);
  var b_err = 1.0;
  if (err_tier > 0u) {
    let m_prev = bitcast<f32>(gi_res_prev[idx * GI_RES_WORDS + GI_RES_W_M]);
    let lo = select(select(0.70, 0.50, err_tier >= 2u), 0.25, err_tier >= 3u);
    let hi = select(select(1.40, 2.00, err_tier >= 2u), 4.00, err_tier >= 3u);
    b_err = clamp(sqrt(GI_SS_ERR_M_REF / max(m_prev, 1.0)), lo, hi);
  }
  let dither = gi_rand2(gi_mix(seed + 0x2545F491u)).x;
  let cand_n = max(1u, u32(floor(f32(cand_n0) * b_err / f32(share) + dither)));
  // 记忆窗上限：按"每帧实际候选数相对基准缩小了几倍"放大，保持窗内样本数 = 基准值。
  // （本项零性能成本，但窗越长光照响应越钝 ⇒ 这就是分摊/重分配的代价。）
  let m_cap_k = select(GI_SS_M_CAP_K, GI_SS_M_CAP_K_HQ, hq) * max(cand_n0 / cand_n, 1u);
  var sel_cum = 0.0;
  var sel_have = false;
  var sel_dir = vec3<f32>(1.0, 0.0, 0.0);
  var sel_l = vec3<f32>(0.0);
  var sel_sk0 = 0u;
  var sel_sk1 = 0u;
  var sel_hit = false;
  var sel_ok = false;
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
    var b_dir: array<vec3<f32>, GI_SS_RAY_BATCH>;
    var b_s: array<GiRaySample, GI_SS_RAY_BATCH>;
    for (var i = 0u; i < nb; i = i + 1u) {
      let kk = k + i;
      let dir = gi_cos_hemisphere(n, gi_rand2(gi_mix(seed + kk * 0x9E3779B9u + 1u)));
      b_dir[i] = dir;
      b_s[i] = gi_ray_radiance(org, dir, sun_dir, sun_c, sun_bounce);
    }
    for (var i = 0u; i < nb; i = i + 1u) {
      let kk = k + i;
      let dir = b_dir[i];
      let s = b_s[i];
      let w = s.l * GI_PI;
      let lum = max(dot(w, GI_LUMA), 0.0);
      r.w_sum = r.w_sum + w;
      r.m = r.m + 1.0;
      sel_cum = sel_cum + lum;
      let u = gi_rand2(gi_mix(seed + kk * 0x85EBCA77u + 7u)).x;
      if (!sel_have || u * sel_cum < lum) {
        sel_have = true;
        sel_dir = dir;
        sel_l = s.l;
        sel_sk0 = s.sk0;
        sel_sk1 = s.sk1;
        sel_hit = s.sk_hit;
        sel_ok = s.sk_ok;
      }
    }
    k = k + nb;
  }

  // ---- ③ 空间复用：读**上一帧**邻居的 reservoir 只并累计量（零射线），只影响本帧估计（见 ④）----
  // tap 的中心 = 本帧命中点上帧的重投影位置（`prev_pc`），不是本像素：读上帧 reservoir 时
  // 像素 `coord + offset` 上帧看到的是「上帧恰好落在那里的那个面」，相机一平移就完全不是同一个面
  // ⇒ 命中率掉到 0。把中心放到「上帧本面所在的屏幕位置」，读到的才是**本面上帧的邻居**。
  // 静止相机下 `prev_pc` 恒等于本像素 ⇒ 与朴素做法逐位一致；重投影失败时退回本像素。
  // 两种 tap 都并累计量（`w_sum` / `M`），判据是唯一的 `gi_same_surface`：
  //  a) **同键**：主命中面键逐位相等 ⇒ 目标函数逐字相同 ⇒ 合并是纯降噪、零偏差；
  //  b) **同一个面**（面号/物体/法线轴坐标相等 + 着色法线相似）：cosθ 几乎相同、平面相同，
  //     只有射线起点差一格 ⇒ 目标函数差一个一阶小量；这是「相邻面的台阶」「凸棱上倾斜的那一段」
  //     被抹平、网格亮斑与棱边噪声消失的原因。薄板另一侧 / 隔一格的平行面仍被精确拒掉。
  let tb = select(coord, prev_pc, prev_ok);
  var sk_sum = vec3<f32>(0.0);
  var sk_m = 0.0;
  for (var t = 0u; t < GI_SS_REUSE_TAPS; t = t + 1u) {
    let nc = tb + gi_ss_tap_offset(t);
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
    sk_sum = sk_sum + wm.xyz;
    sk_m = sk_m + wm.w;
  }
  r.dir = sel_dir;
  r.rad = sel_l;
  r.pk0 = pk0;
  r.pk1 = pk1;
  r.sk0 = sel_sk0;
  r.sk1 = sel_sk1;
  r.flags = GI_RES_VALID | select(0u, GI_RES_PK_OK, pk_ok)
          | select(0u, GI_RES_SK_HIT, sel_hit) | select(0u, GI_RES_SK_OK, sel_ok);
  r.nrm = n;

  // ---- ④ 时域合并 + 写**时域** reservoir（空间复用的合并量绝不写回，见文件头 ③）----
  if (hist_ok) { r = gi_res_adopt(r, hist); }
  r = gi_res_cap(r, cand_n, m_cap_k);
  gi_res_store_cur(idx, r);

  // ---- ⑤ 着色：估计 = w_sum / M = π·⟨L⟩ ⇒ 乘增益、除 π；空间复用只加在这一步 ----
  var est = r.w_sum / max(r.m, 1e-6);
  if (sk_m > 0.0) {
    est = (r.w_sum + sk_sum) / max(r.m + sk_m, 1e-6);
  }
  let gi = est * (gi_u.params.z / GI_PI);
  // valid = 1（供 `dda_main` 的天光兜底权重使用）：GI 有数据 ⇒ 天光只占 `GI_AMBIENT_FLOOR` 倍。
  // 覆盖度 cov 与 valid 是同一个场（旧实现单开一张 `gi_cov_out` 写的恒是「valid ? 1 : 0」），
  // 所以这里只写这一张、`dda_main` 从 `a` 自取。
  textureStore(gi_out, coord, vec4<f32>(gi, 1.0));
}