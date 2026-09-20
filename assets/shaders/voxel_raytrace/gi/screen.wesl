import package::bindings::{gi_cov_out, gi_out, gi_res_cur, gi_res_prev, gi_u, light_u};
import package::common::{palette_transmission};
import package::gi::common::{gi_cos_hemisphere, gi_key0, gi_key1, gi_key_ok, gi_mix, gi_pack_dir, gi_rand2, gi_unpack_dir};
import package::gi::consts::{GI_DEN_N_DOT, GI_DEN_PLANE_K, GI_LUMA, GI_PI, GI_SS_CAND_BOOST, GI_SS_CAND_N, GI_SS_M_CAP_K, GI_SS_REUSE_TAPS, SHADOW_SURFACE_EPS};
import package::gi::denoise::{gi_den_guide_store};
import package::gi::ray::{gi_ray_radiance, gi_ray_secondary_key};
import package::trace::{make_grid};
import package::world::{WorldHit, voxel_normal};

// ============================================================================
// 屏幕空间 ReSTIR GI —— **GI 的唯一路径**（逐像素 reservoir，每帧完整估计）。
//
// 业界形态（Ouyang 2021 / UE5.2+ Hardware Lumen）：逐像素一个 reservoir，
// 「新鲜候选 → 时域复用 → 空间复用 → 着色 → 降噪」。本文件的四条关键取舍：
//
//  ① **样本等价 = 精确几何面键**，不是位置/法线启发式。
//     主命中面键（体素 + 面 + 物体）逐位相等 ⇒ 目标函数逐字相同 ⇒ 合并累计量是精确的 RIS 合并，
//     零偏差、零额外射线。面在键里 ⇒ 薄板两侧天然是两个键 ⇒ 不漏光。
//  ② **空间复用零射线**：跨体素的同平面邻居只**并累计量**，不借方向、不重发射线。
//     旧版让邻居「借代表性方向 + 从本像素重发」并把这个候选并入 RIS 竞争，有两个后果：
//     · 它只在「tap 键不同」时触发 ⇒ 只发生在体素面交界处 ⇒ 交界处被塞进偏亮的候选成一个亮斑；
//     · 代表性样本是按亮度加权抽出来的（不是余弦样本）⇒ 源 pdf 与权重 `π·L` 不匹配 ⇒
//       系统性偏亮，而且这个偏亮的方向会被邻像素逐帧重新选中、在屏幕上漂移
//       （表现为「沿体素边缘的网格亮斑 + 水下光斑式浮动」）。
//     现在代表样本恒来自**本像素的新鲜余弦采样**（源 pdf 恒为余弦，`π·L` 才是正确权重），
//     跨体素平滑完全交给「并累计量」——同键（目标函数逐字相同）与同平面（法线相似 + 平面检验
//     通过 ⇒ 法线相同、平面相同 ⇒ cosθ 相同，只有射线起点差一格）两种都并。
//  ③ **空间复用的结果绝不回写时域历史**。
//     空间复用读的是**上一帧**邻居的 reservoir；把合并结果存回本像素，下一帧的历史里就混进了
//     邻居的历史 ⇒ 时域链被横向污染（相关性、拖影、移动物体「前沿带」）。
//     所以 `gi_res_store_cur` 只写**新鲜 + 时域**的 `r`，空间合并只作用在本帧的着色估计上。
//  ④ **逐体素平坦不被破坏**：同一个面的所有像素并的是同一组邻居 ⇒ 面内仍是一个值（不产生面内
//     梯度）；被平滑掉的是**相邻面之间**的台阶，这正是「网格亮斑」的来源。
//
// 每像素每帧的射线预算（`GI_SS_*`，权威值在 `gi/consts.wesl`）：
//   `GI_SS_CAND_N` 条新鲜余弦候选（去遮挡像素改用 `GI_SS_CAND_BOOST`）
//   + ≤1 条时域二次顶点验证射线（世界几何逐位未变时整条省掉）
//   其中命中几何的射线各带 ≤1 条太阳 NEE 阴影射线。**没有任何空间复用射线**。
//
// 薄板 / 墙缝安全：主键与二次顶点键都带「面」；时域要求键逐位相等，空间要求键相等或
// 「法线相似 + 平面检验」通过（1 格薄板的另一侧法线相反 / 平面距离一整格 ⇒ 精确拒掉）。
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
/// 空间复用的同平面判定用它：它描述「样本方向是从哪个半球里采出来的」。
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

/// 单个 GI 像素的 reservoir。
/// `w_sum` / `M` 是**跨帧累计量**（`M` 上限 = `GI_SS_M_CAP_K` × 本帧新鲜候选数）；
/// `dir`/`rad`/`sk*` 是被选中的那个样本（本像素最近一次的新鲜余弦采样，供降噪/诊断用；
/// 估计只用 `w_sum / M`，所以代表样本抽得偏亮**不会**让估计偏亮）。
/// `nrm`/`pos` = 主命中面的逐体素法线与命中体素中心世界位置（空间复用的同平面判定用）：
/// 两者都是本像素自己的几何，和 `pk0/pk1` 一样随 reservoir 一起跨帧存 —— 这样邻居读到的
/// 几何与它读到的键**必然来自同一帧**（导引是单缓冲、同一 pass 内会被其他线程改写）。
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

/// 上一帧 reservoir 里**空间复用真正要读**的那几个字（几何 + 法线）。
struct GiTapGeom {
  n: vec3<f32>,
  p: vec3<f32>,
}

/// 空 reservoir（天空 / 介质像素）：`M = 0` ⇒ 下一帧任何时域/空间复用都判为无效，不会读到脏值。
fn gi_res_none() -> GiRes {
  return GiRes(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0), 0.0, 0u, 0u, 0u, 0u, 0u,
               vec3<f32>(0.0), vec3<f32>(0.0));
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
    vec3<f32>(
      bitcast<f32>(gi_res_prev[b + GI_RES_W_POS]),
      bitcast<f32>(gi_res_prev[b + GI_RES_W_POS + 1u]),
      bitcast<f32>(gi_res_prev[b + GI_RES_W_POS + 2u]),
    ),
  );
}

/// 空间复用读 tap 的第一级：`(flags, pk0, pk1)` 三个字。绝大多数 tap 在这里就被判掉
/// （空 / 天空像素 / 键不可编码），不必读整条 17 字 —— 每像素 8 个 tap，少读的字数直接是带宽。
fn gi_tap_head(idx: u32) -> vec3<u32> {
  let b = idx * GI_RES_WORDS;
  return vec3<u32>(gi_res_prev[b + GI_RES_W_FLAGS], gi_res_prev[b + GI_RES_W_PK],
                   gi_res_prev[b + GI_RES_W_PK + 1u]);
}

/// 空间复用读 tap 的第二级（只在通过第一级、且**键不同**时才需要）：逐体素法线 + 命中体素中心。
fn gi_tap_geom(idx: u32) -> GiTapGeom {
  let b = idx * GI_RES_WORDS;
  return GiTapGeom(gi_unpack_dir(gi_res_prev[b + GI_RES_W_NRM]), vec3<f32>(
    bitcast<f32>(gi_res_prev[b + GI_RES_W_POS]),
    bitcast<f32>(gi_res_prev[b + GI_RES_W_POS + 1u]),
    bitcast<f32>(gi_res_prev[b + GI_RES_W_POS + 2u]),
  ));
}

/// 空间复用读 tap 的第三级（只在并入时）：累计量（rgb = `w_sum`、a = `M`）。4 个字。
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
  gi_res_cur[b + GI_RES_W_POS] = bitcast<u32>(r.pos.x);
  gi_res_cur[b + GI_RES_W_POS + 1u] = bitcast<u32>(r.pos.y);
  gi_res_cur[b + GI_RES_W_POS + 2u] = bitcast<u32>(r.pos.z);
}

/// 并入另一个 reservoir 的**累计量**（`w_sum += w_sum`、`M += M`）。只给时域复用用
/// （调用方先判「主命中面键逐位相等」）⇒ 目标函数逐字相同 ⇒ 合并无偏。
fn gi_res_adopt(r: GiRes, h: GiRes) -> GiRes {
  var out = r;
  out.w_sum = r.w_sum + h.w_sum;
  out.m = r.m + h.m;
  return out;
}

/// M 上限（业界：M ≤ 20 × M_new）：超过就同比例缩回 —— `w_sum / M` 不变（估计不变），
/// 只抑制历史对代表样本抽样的支配。以**本帧新鲜候选数**为基准是必须的：
/// 去遮挡帧的候选数翻倍 ⇒ 上限跟着翻倍，否则 boost 打进去的候选会被立刻压掉、boost 等于白做。
fn gi_res_cap(r: GiRes, cand_n: u32) -> GiRes {
  let cap = f32(GI_SS_M_CAP_K * cand_n);
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

/// 同平面判定（决定「能不能并邻居的累计量」）—— 与降噪侧**同一套**判据：
///  ① 法线相似 `dot(n_nbr, n_c) ≥ GI_DEN_N_DOT`：两端都是**逐体素法线**（不是面法线）⇒
///     1 体素薄板的另一侧法线相反、拐角处法线斜着 ⇒ 都直接拒；
///  ② 平面检验 `|(p_nbr − p_c)·n_c| < tol`：两端都取命中**体素中心** ⇒ 同一平面上相邻体素
///     该内积**恰好为 0**，而薄板另一侧 = 1 个体素世界边长 > tol（= 0.5 × 边长）⇒ 精确拒掉，
///     且与距离无关（远景薄板也不会因为像素足迹变大而漏）。
/// 只用几何（法线 / 位置），没有任何「颜色相近」的启发式。
/// `tol` 由调用方给（= `GI_DEN_PLANE_K` × 本像素命中体素的世界边长）。
fn gi_ss_coplanar(p_c: vec3<f32>, n_c: vec3<f32>, tol: f32, nb: GiTapGeom) -> bool {
  return dot(nb.n, n_c) >= GI_DEN_N_DOT && abs(dot(nb.p - p_c, n_c)) < tol;
}

/// 屏幕空间 ReSTIR 的本像素入口（`gi_main` 调用）。
/// 只处理「首命中即不透明」；天空 / 介质像素写 valid = 0（与 `dda_main` 的玻璃分支一致）。
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
  // 沿 n 外推 0.5/max|n| + SHADOW_SURFACE_EPS：与 `dda_main` 同一套外推（起点离开命中体素）。
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
  var prev_ok = false;
  let prev_pc = gi_ss_reproject(h.point, size, &prev_ok);
  gi_den_guide_store(coord, size, true, pk0, pk1, p_voxel, n, GI_DEN_PLANE_K * gg.scale, prev_ok,
                     prev_pc);

  // ---- ① 时域复用：先判「有没有可用历史」，它决定本帧的新鲜候选数（去遮挡 boost）----
  // 判据是**精确键**（不是启发式）：主命中面键（体素,面,物体）必须完全一致；
  // 再接一条重连接射线（本帧唯一的那条，≤1）沿历史方向重发，用它的二次顶点键与历史存的那个做
  // **逐位相等**比较 —— 不一致就丢掉历史累计量（世界变了 / 不是同一个面）；一致就接过来。
  // 重连接射线只求键（裸 `world_raycast`，不带 NEE）、**不进 RIS**（见下方 ② 处注释）。
  // **世界几何自写入历史那一帧起逐位未变时（uniform `flags.z`）这条验证射线整条省掉** ——
  // 输入完全相同 ⇒ 结果必然逐位相同，这是静态 / 纯相机运动下最大的一笔省。
  var hist = gi_res_none();
  var hist_ok = false;
  if (pk_ok && prev_ok) {
    let cand = gi_res_load_prev(u32(prev_pc.y) * size.x + u32(prev_pc.x));
    if ((cand.flags & (GI_RES_VALID | GI_RES_PK_OK)) == (GI_RES_VALID | GI_RES_PK_OK)
        && cand.pk0 == pk0 && cand.pk1 == pk1) {
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

  // ---- ② 新鲜候选：`cand_n` 条余弦加权半球方向（无历史 = 去遮挡 ⇒ 用 boost 档）----
  // 每条走 `gi_ray_radiance`（1 条 GI 射线，命中时含 1 条太阳 NEE 阴影射线）。
  // RIS：累计量 `w_sum = Σ π·L`、`M = cand_n`（估计 = w_sum / M = π·⟨L⟩）。
  // 代表样本 = **只在这一组余弦候选里**做加权蓄水池抽样（源 pdf 恒是余弦 ⇒ 权重 `π·L` 才自洽；
  // 见文件头 ②：绝不让空间复用进来的方向参与这个竞争）。
  let cand_n = select(GI_SS_CAND_N, GI_SS_CAND_BOOST, !hist_ok);
  var sel_cum = 0.0;
  var sel_have = false;
  var sel_dir = vec3<f32>(1.0, 0.0, 0.0);
  var sel_l = vec3<f32>(0.0);
  var sel_sk0 = 0u;
  var sel_sk1 = 0u;
  var sel_hit = false;
  var sel_ok = false;
  var r = gi_res_none();
  for (var k = 0u; k < cand_n; k = k + 1u) {
    let dir = gi_cos_hemisphere(n, gi_rand2(gi_mix(seed + k * 0x9E3779B9u + 1u)));
    let s = gi_ray_radiance(org, dir, sun_dir, sun_c);
    let w = s.l * GI_PI;
    let lum = max(dot(w, GI_LUMA), 0.0);
    r.w_sum = r.w_sum + w;
    r.m = r.m + 1.0;
    sel_cum = sel_cum + lum;
    let u = gi_rand2(gi_mix(seed + k * 0x85EBCA77u + 7u)).x;
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

  // ---- ③ 空间复用：读**上一帧**邻居的 reservoir 只并累计量（零射线），只影响本帧估计（见 ④）----
  // tap 的中心 = 本帧命中点上帧的重投影位置（`prev_pc`），不是本像素：读上帧 reservoir 时
  // 像素 `coord + offset` 上帧看到的是「上帧恰好落在那里的那个面」，相机一平移就完全不是同一个面
  // ⇒ 命中率掉到 0。把中心放到「上帧本面所在的屏幕位置」，读到的才是**本面上帧的邻居**。
  // 静止相机下 `prev_pc` 恒等于本像素 ⇒ 与朴素做法逐位一致；重投影失败时退回本像素。
  // 两种 tap 都并累计量（`w_sum` / `M`）：
  //  a) **同键**：主命中面键逐位相等 ⇒ 目标函数逐字相同 ⇒ 合并是纯降噪、零偏差；
  //  b) **同平面**（键不同但过 `gi_ss_coplanar`）：法线相同、平面相同 ⇒ cosθ 相同，只有射线
  //     起点差一格 ⇒ 目标函数差一个一阶小量；这是「相邻面的台阶」被抹平、网格亮斑消失的原因。
  //     1 体素薄板的另一侧过不了平面检验 ⇒ 不跨薄板、不漏光。
  let tb = select(coord, prev_pc, prev_ok);
  let tol = GI_DEN_PLANE_K * gg.scale;
  var sk_sum = vec3<f32>(0.0);
  var sk_m = 0.0;
  for (var t = 0u; t < GI_SS_REUSE_TAPS; t = t + 1u) {
    let nc = tb + gi_ss_tap_offset(t);
    if (nc.x < 0 || nc.y < 0 || nc.x >= i32(size.x) || nc.y >= i32(size.y)) { continue; }
    let nidx = u32(nc.y) * size.x + u32(nc.x);
    let head = gi_tap_head(nidx);
    if ((head.x & (GI_RES_VALID | GI_RES_PK_OK)) != (GI_RES_VALID | GI_RES_PK_OK)) { continue; }
    // a) 同键：直接并。
    var ok = head.y == pk0 && head.z == pk1;
    if (!ok) {
      // b) 同平面：要读几何（法线 + 体素中心）做平面检验。
      ok = gi_ss_coplanar(p_voxel, n, tol, gi_tap_geom(nidx));
    }
    if (!ok) { continue; }
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
  r.pos = p_voxel;

  // ---- ④ 时域合并 + 写**时域** reservoir（空间复用的合并量绝不写回，见文件头 ③）----
  if (hist_ok) { r = gi_res_adopt(r, hist); }
  r = gi_res_cap(r, cand_n);
  gi_res_store_cur(idx, r);

  // ---- ⑤ 着色：估计 = w_sum / M = π·⟨L⟩ ⇒ 乘增益、除 π；空间复用只加在这一步 ----
  var est = r.w_sum / max(r.m, 1e-6);
  if (sk_m > 0.0) {
    est = (r.w_sum + sk_sum) / max(r.m + sk_m, 1e-6);
  }
  let gi = est * (gi_u.params.z / GI_PI);
  // valid = 1（供 `dda_main` 的天光兜底权重使用）：GI 有数据 ⇒ 天光只占 `GI_AMBIENT_FLOOR` 倍。
  textureStore(gi_out, coord, vec4<f32>(gi, 1.0));
  // storage texture 的 store 值恒为 vec4（rg32f 也不例外，多出的通道被丢弃）：r = cov、g = valid。
  textureStore(gi_cov_out, coord, vec4<f32>(1.0, 1.0, 0.0, 0.0));
}