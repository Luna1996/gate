import package::bindings::{gi_cov_out, gi_out, gi_res_cur, gi_res_prev, gi_u, light_u};
import package::common::{palette_transmission};
import package::gi::cache::{gi_cos_hemisphere, gi_key0, gi_key1, gi_key_ok, gi_mix, gi_pack_dir, gi_rand2, gi_unpack_dir};
import package::gi::consts::{GI_DEN_N_DOT, GI_DEN_PLANE_K, GI_PI, GI_SS_M_CAP, GI_SS_REUSE_RAYS, GI_SS_REUSE_TAPS, SHADOW_SURFACE_EPS};
import package::gi::denoise::{gi_den_guide_store};
import package::gi::ray::{gi_ray_radiance, gi_ray_secondary_key};
import package::trace::{make_grid};
import package::world::{WorldHit, voxel_normal};

// ============================================================================
// 屏幕空间逐面 ReSTIR 路径 —— 每帧完整估计（`GI_CACHE_ON = false` 时接管 gi_main）。
//
// 目标：**同帧收敛**。新进入视野的像素当帧就拿到估计，不允许「等 N 帧才有值」。
// 因此这里没有世界空间条目、没有运行时平均，只有「本帧候选 + 上帧/邻居的 reservoir」。
// 代价是**画面必然有噪声**（降噪是下一步）。
//
// 每个 GI 像素每帧：
//   ① 新鲜：1 条余弦加权半球方向 ⇒ `gi_ray_radiance`（1 条 GI 射线，命中时含 1 条太阳 NEE 阴影射线）；
//   ② 时域：用**上一帧**的相机矩阵把本像素的主命中点重投影到上帧的 GI 网格，读上帧 reservoir；
//      判据是**精确键**（不是启发式）：主命中面键 (体素,面,物体) 必须完全一致；
//      再接一条重连接射线（本帧唯一的那条，≤1）沿历史方向重发，用它的二次顶点键与历史存的那个做
//      **逐位相等**比较 —— 不一致就丢掉历史累计量（世界变了 / 不是同一个面）；一致就把历史累计量
//      `w_sum`/`M` 接过来。重连接射线只求键（裸 `world_raycast`，不带 NEE）、**不进 RIS**
//      （见 ② 处注释：会把估计「锁定」到某个方向而系统性偏亮）。
//      **世界几何自写入历史那一帧起逐位未变时（uniform `flags.z`）这条验证射线整条省掉** ——
//      输入完全相同 ⇒ 结果必然逐位相同（见 ② 处注释），这是静态/纯相机运动下最大的一笔省。
//      ⇒ 每像素射线数：新鲜 1 条（含 ≤1 条 NEE 阴影）+ 重连接 ≤1 条（裸射线，稳态通常 0）= 1~3 条。
//   ③ 空间：上帧 **本面所在位置**（= 本帧命中点上帧的重投影，静止相机下就是这个像素自身）周围
//      3×3 环上的 8 个 tap（起点按像素哈希轮转；`GI_SS_REUSE_RAYS = 0` 时只用
//      4 个轴邻 = 改造前的行为）。两档：
//      a) **主命中面键完全一致**（同一个体素同一个面 ⇒ 目标函数逐字相同）⇒ 直接并累计量
//         （零射线、零偏差，与改造前同一条快路）；
//      b) **主键不同但过相似性判定**（法线 `dot ≥ GI_DEN_N_DOT` + 平面检验
//         `|(p_nbr − p_c)·n_c| < GI_DEN_PLANE_K × 体素边长`，与降噪侧**同一套**判据；位置取命中
//         体素中心）⇒ **借它的方向、从本像素重发一条射线**（≤`GI_SS_REUSE_RAYS` 条/像素），
//         把得到的那条 L 作为本帧候选。
//         借的只是「方向」：reservoir 存的方向恒是该像素**最近一次新鲜余弦采样**（`gi_res_adopt`
//         只并累计量、不换样本方向）⇒ 源 pdf 就是（邻居法线下的）余弦分布，权重与新鲜候选同口径
//         `π·L`，**偏差只落在估计权重上**（邻居法线与自己的法线差 ≤ 25° + f16 八面体量化 < 1°，
//         一阶小量），几何/可见性**零偏差**：射线从本像素重发、遮挡由真实几何决定
//         ⇒ 不会把邻居的光带过来、不跨薄板、不跨墙缝（这正是 ReSTIR 空间复用「只借方向」的取舍）。
//      ⇒ 每像素主射线：新鲜 1 + 时域键验证 ≤1 + 借用 ≤`GI_SS_REUSE_RAYS` = 2 ~ 4 条。
//   ④ RIS 合并：余弦候选 w = π·L（余弦 pdf 下 f/p = L·cosθ·π/cosθ）；**本帧的「新鲜 + 借用」
//      先做帧内等权平均，再作为一个候选计入 M**（M 的单位 = 帧候选 = 记忆帧数，见下），
//      `w_sum += Σw`、`M += 候选数`，`M` 触及 `GI_SS_M_CAP` 时把 w_sum 与 M 同比例缩回。
//      估计 = `w_sum / M` = π·⟨L⟩（与旧缓存 `E_store` 同语义）⇒ 写进 `gi_out` 前再乘增益、除 π。
//      为什么借用射线**不各自占一个 M 增量**：M 封顶时每帧多插 k 个候选等价于把记忆帧数从 M 缩到
//      M/k ⇒ 逐帧闪烁 ∝ √k 变差（k = 3 时实测/推导都是 +1.7×），静态空间 σ 基本不变；改成帧内
//      等权平均后，k 条射线共享一个 M 增量 ⇒ 记忆帧数不变，而逐帧闪烁与静态 σ 同时按 1/√k 下降。
//
// 薄板/墙缝安全：主键里的**面**（`gi_key1` 的 3 bit）使薄板两侧是两个不同的键；时域与空间两路都
// 要求精确键相等或（借用档）法线 + 平面检验通过（另一侧法线相反 / 平面距离 = 1 个体素边长）；
// 且**所有辐亮度都由本像素自己的射线求得** ⇒ 任何一档都不可能把邻居的光搬过来。
// ============================================================================

/// reservoir 字数（Rust 侧从本文件解析同名字面量开 buffer）。
const GI_RES_WORDS: u32 = 17u;
/// word 布局。
const GI_RES_W_DIR: u32 = 0u; // 选中样本方向（八面体 f16x2）
const GI_RES_W_RAD: u32 = 1u; // 选中样本的入射辐亮度 L（rgb）
const GI_RES_W_WSUM: u32 = 4u; // 累计权重 Σ w（rgb = π·L 的和）
const GI_RES_W_M: u32 = 7u; // 候选数 M
const GI_RES_W_PK: u32 = 8u; // 主命中面键（gi_key0 / gi_key1）
const GI_RES_W_SK: u32 = 10u; // 选中样本的二次顶点键（gi_key0 / gi_key1）
const GI_RES_W_FLAGS: u32 = 12u; // 状态位（见下）
/// 主命中面的**逐体素法线**（八面体 f16x2；与降噪导引同一个 `gi_pack_dir` 口径）。
/// 空间借用档的相似性判定要用它：它描述的是「上一个方向是从哪个半球里采出来的」。
const GI_RES_W_NRM: u32 = 13u;
/// 主命中**体素中心**的世界位置（3 word；与降噪导引同一口径 ⇒ 平面检验同义）。
const GI_RES_W_POS: u32 = 14u;
/// flags 位：有样本（M > 0）。M = 0 的 reservoir 一律视为无效。
const GI_RES_VALID: u32 = 1u;
/// 主键可编码（体素在 `gi_key_ok` 范围内）⇒ 允许参与复用。
const GI_RES_PK_OK: u32 = 2u;
/// 二次顶点命中（false = 逃逸到天空，键取 (0,0)）。
const GI_RES_SK_HIT: u32 = 4u;
/// 二次顶点键可比较（`gi_key_ok` 或天空）⇒ 只有它置位才允许做二次顶点键的相等判定。
const GI_RES_SK_OK: u32 = 8u;

/// 单个 GI 像素（网格 = 渲染分辨率 / gi_div）的 reservoir。
/// `w_sum` / `M` 是**跨帧累计量**（RIS 的 M 上限抑制历史支配）；`dir`/`rad`/`sk*` 是被选中
/// 的那个样本（供下游借用方重连接用；本像素自己的估计只用 `w_sum / M`）。
/// `nrm`/`pos` = 主命中面的逐体素法线与命中体素中心世界位置（**空间借用档的相似性判定用**：
/// 两者都是本像素自己的几何，和 `pk0/pk1` 一样随 reservoir 一起跨帧存 —— 这样邻居读到的
/// 几何与它读到的方向/键**必然来自同一帧**，不需要再去读导引（那份是单缓冲、同一 pass 内
/// 会被本帧其他线程改写 ⇒ 读到的是本帧/上帧的混合，非确定）。
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
  pos: vec3<f32>,
}

/// 空 reservoir（天空 / 介质像素）：`M = 0` ⇒ 下一帧任何时序/空间复用都判为无效，不会读到脏值。
fn gi_res_none() -> GiRes {
  return GiRes(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0), 0.0, 0u, 0u, 0u, 0u, 0u,
               vec3<f32>(0.0), vec3<f32>(0.0));
}

/// 首个候选（新鲜射线）建 reservoir。`nrm`/`pos` = 主命中面的逐体素法线与命中体素中心世界位置
/// （空间借用档的相似性判定的两端之一；存进 reservoir ⇒ 与 `pk0/pk1`、`dir` 同帧同源）。
fn gi_res_new(pk0: u32, pk1: u32, pk_ok: bool, dir: vec3<f32>, rad: vec3<f32>, sk0: u32,
               sk1: u32, sk_hit: bool, sk_ok: bool, nrm: vec3<f32>, pos: vec3<f32>) -> GiRes {
  var flags = GI_RES_VALID;
  if (pk_ok) { flags = flags | GI_RES_PK_OK; }
  if (sk_hit) { flags = flags | GI_RES_SK_HIT; }
  if (sk_ok) { flags = flags | GI_RES_SK_OK; }
  return GiRes(dir, rad, rad * GI_PI, 1.0, pk0, pk1, sk0, sk1, flags, nrm, pos);
}

/// 读**上一帧**的 reservoir（`gi_res_prev`）。
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
    vec3<f32>(
      bitcast<f32>(gi_res_prev[b + GI_RES_W_POS]),
      bitcast<f32>(gi_res_prev[b + GI_RES_W_POS + 1u]),
      bitcast<f32>(gi_res_prev[b + GI_RES_W_POS + 2u]),
    ),
  );
}

/// 写**本帧**的 reservoir（`gi_res_cur`；两块 buffer 每帧在 Rust 侧 ping-pong 换绑）。
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
  gi_res_cur[b + GI_RES_W_POS] = bitcast<u32>(r.pos.x);
  gi_res_cur[b + GI_RES_W_POS + 1u] = bitcast<u32>(r.pos.y);
  gi_res_cur[b + GI_RES_W_POS + 2u] = bitcast<u32>(r.pos.z);
}

/// 并入另一个 reservoir 的**累计量**（`w_sum += w_sum`、`M += M`）。
/// 只允许在「目标函数逐字相同」时调用 ⇒ 主命中面键完全一致（调用方先判）。
/// 这样做无偏：两者都是同一个积分的估计量，取加权平均仍是该积分的无偏估计。
/// 选中样本保持**本帧新鲜候选**（它只作为「代表样本」被下游借用方重连接，不参与估计）。
fn gi_res_adopt(r: GiRes, h: GiRes) -> GiRes {
  var out = r;
  out.w_sum = r.w_sum + h.w_sum;
  out.m = r.m + h.m;
  return out;
}

/// M 上限：超过就同比例缩回（`w_sum / M` 不变，只抑制历史对抽样的支配）。
fn gi_res_cap(r: GiRes) -> GiRes {
  if (r.m <= f32(GI_SS_M_CAP)) { return r; }
  let shrink = f32(GI_SS_M_CAP) / r.m;
  var out = r;
  out.w_sum = r.w_sum * shrink;
  out.m = f32(GI_SS_M_CAP);
  return out;
}

/// 每像素每帧的可复现种子（帧号 + 像素坐标哈希）—— 不用屏幕无关的全局随机。
fn gi_ss_seed(coord: vec2<i32>, frame: u32) -> u32 {
  let ph = gi_mix(u32(coord.x) * 0x9E3779B1u + gi_mix(u32(coord.y) * 0x85EBCA77u + 0x165667B1u));
  return gi_mix(ph + gi_mix(frame * 0x9E3779B9u + 0x27D4EB2Fu));
}

/// 世界点 → **上一帧**的 GI 像素坐标（用 `gi_u.prev_view_proj`；落在视锥/屏幕外则 `ok = false`）。
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

/// 空间复用的候选 tap 偏移（上帧同屏的 3×3 环）。
/// **前 4 个恒是 4 个轴邻、顺序与改造前的空间复用逐字一致** —— `GI_SS_REUSE_RAYS = 0`
/// （A/B 的对照档）时只用这 4 个 ⇒ 那条路径与改造前完全相同。
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

/// 空间借用档的相似性判定（决定「能不能借邻居的方向」）—— 与降噪侧**同一套**判据：
///  ① 法线相似 `dot(n_nbr, n_c) ≥ GI_DEN_N_DOT`：邻居存的法线是**逐体素法线**（不是面法线）⇒
///     1 体素薄板的另一侧法线相反、拐角处法线斜着 ⇒ 都直接拒；
///  ② 平面检验 `|(p_nbr − p_c)·n_c| < tol`：两端都取命中**体素中心** ⇒ 同一平面上相邻体素
///     该内积**恰好为 0**，而薄板另一侧 = 1 个体素世界边长 > tol（= 0.5 × 边长）⇒ 精确拒掉，
///     且与距离无关（远景薄板也不会因为像素足迹变大而漏）。
/// 只用几何（法线 / 位置），没有任何「颜色相近」的启发式。
/// `tol` 由调用方给（= `GI_DEN_PLANE_K` × 本像素命中体素的世界边长）。
fn gi_ss_similar(p_c: vec3<f32>, n_c: vec3<f32>, tol: f32, nb: GiRes) -> bool {
  return dot(nb.nrm, n_c) >= GI_DEN_N_DOT && abs(dot(nb.pos - p_c, n_c)) < tol;
}

/// 屏幕空间逐面 ReSTIR 的本像素入口（由 `gi_main` 在 `GI_CACHE_ON = false` 时调用）。
/// 只处理「首命中即不透明」；天空 / 介质像素写 valid = 0（与旧路径、`dda_main` 的玻璃分支一致）。
fn gi_ss_main(coord: vec2<i32>, size: vec2<u32>, h: WorldHit) {
  let idx = u32(coord.y) * size.x + u32(coord.x);
  let sun_dir = light_u.lights[0].kind_pos_dir.yzw;
  let sun_c = light_u.lights[0].color_intensity.xyz * light_u.lights[0].color_intensity.w;

  if (!(h.hit && palette_transmission(h.palette_base, h.pal) <= 0.0)) {
    textureStore(gi_out, coord, vec4<f32>(0.0));
    textureStore(gi_cov_out, coord, vec4<f32>(0.0));
    gi_res_store_cur(idx, gi_res_none());
    // 降噪导引同步清零：valid = 0 ⇒ 两段降噪都跳过（与 `gi_out` 的 valid = 0 一致）。
    gi_den_guide_store(coord, size, false, 0u, 0u, vec3<f32>(0.0), vec3<f32>(0.0), 0.0, false,
                       vec2<i32>(0));
    return;
  }

  let gg = make_grid(u32(h.obj_id) + 1u);
  let vc = vec3<f32>(h.voxel) + vec3<f32>(0.5);
  let p_voxel = gg.pos + vec3<f32>(dot(vc, gg.col0), dot(vc, gg.col1), dot(vc, gg.col2)) * gg.scale;
  let n = voxel_normal(gg, h.voxel, h.n);
  // 沿 n 外推 0.5/max|n| + SHADOW_SURFACE_EPS：与 `dda_main` / 旧缓存射线起点同一套外推。
  let n_off = 0.5 / max(max(abs(n.x), abs(n.y)), max(abs(n.z), 1e-3)) + SHADOW_SURFACE_EPS;
  let org = p_voxel + n * n_off;

  // 主命中面键（复用判据）+ 是否可编码。
  let pk0 = gi_key0(h.voxel);
  let pk1 = gi_key1(h.voxel, h.face_id, h.obj_id);
  let pk_ok = gi_key_ok(h.voxel);

  let frame = gi_u.seq.x;
  let seed = gi_ss_seed(coord, frame);

  // ---- 降噪导引：命中体素中心 + 面键 + 法线 + 平面检验容差 + 上帧重投影像素 ----
  // 重投影用**真实命中点** `h.point`（不是体素中心）：静止相机下它正好映回本像素
  // ⇒ 上帧历史就是本像素自己的（M 才能一路累加）。用体素中心会偏 1~2 像素，导致约 1/3 的像素
  // 每帧都因为「上帧那个像素不是同一个面」而丢掉历史（实测接受率只有 68%）。
  var den_prev_ok = false;
  let den_pc = gi_ss_reproject(h.point, size, &den_prev_ok);
  gi_den_guide_store(coord, size, true, pk0, pk1, p_voxel, n, GI_DEN_PLANE_K * gg.scale, den_prev_ok,
                     den_pc);

  // ---- ① 新鲜路径：1 条余弦加权半球方向（GI 射线；命中时 helper 内含 1 条太阳 NEE 阴影射线）----
  let dir_f = gi_cos_hemisphere(n, gi_rand2(gi_mix(seed + 1u)));
  let s_f = gi_ray_radiance(org, dir_f, sun_dir, sun_c);
  var r = gi_res_new(pk0, pk1, pk_ok, dir_f, s_f.l, s_f.sk0, s_f.sk1, s_f.sk_hit, s_f.sk_ok,
                     n, p_voxel);

  if (pk_ok) {
    // ---- ② 时域复用：重投影 + 精确键筛（主键 + 二次顶点键）----
    // 重投影用**真实命中点**（`den_pc`，与降噪导引、空间 tap 共用同一份）：静止相机下它正好映回
    // 本像素 ⇒ 上帧历史就是本像素自己的。改用体素中心会偏 1~2 像素，约 1/3 的像素每帧都因为
    // 「上帧那个像素不是同一个面」丢掉历史（实测接受率只有 68%）。
    // **重投影只负责提名候选，接不接受由下面两个精确键判据决定** ⇒ 提名变准只会提高接受率，
    // 不可能放进一个错的候选（跨薄板/墙缝的判定完全不依赖重投影）。
    // TEMP(A/B)：flags.x 的 bit1 ⇒ 退回体素中心重投影（= 改造前行为）
    var prev_ok = den_prev_ok;
    var pc = den_pc;
    if ((u32(gi_u.flags.x) & 2u) != 0u) {
      pc = gi_ss_reproject(p_voxel, size, &prev_ok);
    }
    if (prev_ok) {
      let hist = gi_res_load_prev(u32(pc.y) * size.x + u32(pc.x));
      // 主键必须完全一致：不同体素 / 不同面 / 不同物体一律不接（薄板两侧是两个键）。
      if ((hist.flags & GI_RES_VALID) != 0u && (hist.flags & GI_RES_PK_OK) != 0u
          && hist.pk0 == pk0 && hist.pk1 == pk1) {
        // 重连接射线（本帧唯一的 ≤1 预算）：沿历史方向重发，用它的二次顶点键验证历史样本。
        // 它**只用于验证**、不并进 RIS：同一个方向每帧被重加，会让 `w_sum / M` 逐渐「锁定」到
        // 那个方向的 L —— 抽样是按亮度加权的，被锁定的必然偏亮 ⇒ 估计系统性偏亮（实测 ~+25%）。
        // 只需要键 ⇒ 走轻量版（不做 NEE / 不读缓存），这条射线的开销就是一条裸 `world_raycast`。
        //
        // **整条省掉的条件**（uniform `flags.z`，Rust `prepare_gi` 的 `skip_verify`）：世界几何自
        // 「写入 `hist` 的那一帧」起逐位未变 ⇒ 这条射线与当初写入 `hist.sk` 的那条输入完全相同
        // （方向取自 `hist.dir`，原点由已校验逐位相等的主键唯一决定）⇒ 结果必然逐位相同 ⇒
        // 判定退化成「历史那个键当时可比较吗」，一个位测试就够。世界变过就照旧发射线。
        var same_sk = false;
        if (gi_u.flags.z > 0.5) {
          same_sk = (hist.flags & GI_RES_SK_OK) != 0u;
        } else {
          let s_r = gi_ray_secondary_key(org, hist.dir);
          same_sk = s_r.sk_hit == ((hist.flags & GI_RES_SK_HIT) != 0u)
            && s_r.sk_ok && s_r.sk0 == hist.sk0 && s_r.sk1 == hist.sk1;
        }
        if (same_sk) {
          // 二次顶点键逐位一致 ⇒ 历史样本就是「本面看到的同一个方向的同一段光路」⇒ 累计量有效。
          // 累计量是**独立余弦样本权重的运行平均**（每帧只进 1 个新鲜候选）⇒ 无偏，且随时间降噪。
          r = gi_res_adopt(r, hist);
        }
      }
    }

    // ---- ③ 空间复用：上帧 3×3 环上的 8 个 tap（起点按像素哈希轮转 ⇒ 各像素借的邻居不同）----
    // `GI_SS_REUSE_RAYS = 0` 时退化成「只用 4 个轴邻 + 只做同键直并」＝ 改造前的空间复用（A/B 对照档）。
    //
    // **tap 的中心 = 本帧命中点上帧的重投影位置（`den_pc`），不是本像素**。这一条是空间复用能不能
    // 扛住相机运动的分水岭：像素 `coord + offset` 上帧看到的是「上帧恰好落在那里的那个面」，它只在
    // 相机静止（或纯旋转且表面正面朝向相机）时才与本面共面 —— 相机一平移，上帧那个像素看到的是
    // 完全不同的面 ⇒ 同键率掉到 0（实测平移段 0.000/像素）、相似性通过率掉到 ~5%，8 个 tap 全白读。
    // 把中心放到「上帧本面所在的屏幕位置」上，读到的才是**本面上帧的邻居**（同键 ⇒ 零射线直并）
    // 与它们的世界共面邻面（借用档）。
    // 静止相机下 `den_pc` 恒等于本像素（命中点沿同一条相机射线投回同一像素）⇒ **与改造前逐位一致**；
    // 重投影失败（上帧本面不在视锥/屏幕内）时退回本像素 = 改造前行为（此时没有任何可复用的世界对应）。
    // TEMP(A/B)：flags.x > 0.5 ⇒ tap 中心退回本像素（= 改造前行为）
    let tb = select(select(coord, den_pc, den_prev_ok), coord, gi_u.flags.x > 0.5);
    let taps = select(GI_SS_REUSE_TAPS, 4u, GI_SS_REUSE_RAYS == 0u);
    let rot = select(gi_ss_seed(coord, frame + 0x51u) % taps, 0u, GI_SS_REUSE_RAYS == 0u);
    let tol = GI_DEN_PLANE_K * gg.scale;
    var borrow_n = 0.0; // 本帧借到的候选数（= 真正发出去的借用射线数）
    var borrow_l = vec3<f32>(0.0);
    for (var t = 0u; t < taps; t = t + 1u) {
      let nc = tb + gi_ss_tap_offset((t + rot) % taps);
      if (nc.x < 0 || nc.y < 0 || nc.x >= i32(size.x) || nc.y >= i32(size.y)) { continue; }
      let nb = gi_res_load_prev(u32(nc.y) * size.x + u32(nc.x));
      if ((nb.flags & (GI_RES_VALID | GI_RES_PK_OK)) != (GI_RES_VALID | GI_RES_PK_OK)) { continue; }
      // a) 同键快路：同一个体素同一个面 ⇒ 目标函数逐字相同 ⇒ 直接并累计量（零射线、零偏差）。
      if (nb.pk0 == pk0 && nb.pk1 == pk1) {
        r = gi_res_adopt(r, nb);
        continue;
      }
      // b) 借用档：主键不同 ⇒ 过相似性判定（法线 + 平面检验，见 `gi_ss_similar`）才借它的方向，
      //    从**本像素**重发一条射线。借的只是方向、辐亮度由本像素自己求 ⇒ 不会把邻居的光带过来、
      //    不跨薄板/墙缝（遮挡由真实几何决定，与邻居是谁无关）。
      if (borrow_n >= f32(GI_SS_REUSE_RAYS)) { continue; }
      if (!gi_ss_similar(p_voxel, n, tol, nb)) { continue; }
      let d_b = nb.dir;
      // 方向必须落在本像素自己的半球内（邻居法线与自己最多差 25°，余弦半球的边缘上会有少数方向
      // 落到本面之下）：那条射线只会立刻自交、白费预算 ⇒ 直接跳过。丢弃只取决于几何、与亮度无关
      // ⇒ 不引入偏差。
      if (dot(d_b, n) <= 1e-3) { continue; }
      // 与新鲜候选**同一条求值口径**（`gi_ray_radiance`：命中时含二次顶点的太阳 NEE）。
      // 不能用只求键的轻量版：那会让借用候选丢掉二次顶点的太阳项 ⇒ 借用比例一高就系统性偏暗。
      let s_b = gi_ray_radiance(org, d_b, sun_dir, sun_c);
      borrow_l = borrow_l + s_b.l;
      borrow_n = borrow_n + 1.0;
    }
    // 借用射线与新鲜射线做**帧内等权平均**，再作为一个候选计入 M（见文件头 ④ 的推导：每条借用
    // 射线各占一个 M 增量 = 把记忆帧数从 M 缩到 M/k ⇒ 闪烁 ∝ √k 变差；共享一个 M 增量 ⇒ 反而 ↓√k）。
    // 做法：从 w_sum 里扣掉「未平均的新鲜项」（= `gi_res_new` 放进去的 `rad·π`），再加回
    // 「新鲜 + 借用的等权平均」—— 只碰本帧的新鲜候选，不动时域历史 / 同键直并的累计量。
    if (borrow_n > 0.0) {
      let inv = GI_PI / (1.0 + borrow_n);
      r.w_sum = r.w_sum - r.rad * GI_PI + (r.rad + borrow_l) * inv;
    }
  }

  // ---- ④ RIS 合并 + 写输出 ----
  r = gi_res_cap(r);
  // 估计 = w_sum / M = π·⟨L⟩（= 旧缓存 `E_store`）⇒ 乘增益、除 π 后与 `gi_supply` 逐字同口径。
  let est = r.w_sum / max(r.m, 1e-6);
  // 选中样本的方向/亮度也留在 reservoir 里供下一帧借用方重连接（估计只用累计量）。
  gi_res_store_cur(idx, r);
  let gi = est * (gi_u.params.z / GI_PI);
  // 有数据 = 1（与旧路径 `gi_supply` 的 w 语义一致）：供 `dda_main` 的天光兜底权重使用。
  textureStore(gi_out, coord, vec4<f32>(gi, 1.0));
  textureStore(gi_cov_out, coord, vec4<f32>(1.0, 1.0, 0.0, 0.0));
}
