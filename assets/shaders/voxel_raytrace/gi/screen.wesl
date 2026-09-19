import package::bindings::{gi_cov_out, gi_out, gi_res_cur, gi_res_prev, gi_u, light_u};
import package::common::{palette_transmission};
import package::gi::cache::{gi_cos_hemisphere, gi_key0, gi_key1, gi_key_ok, gi_mix, gi_pack_dir, gi_rand2, gi_unpack_dir};
import package::gi::consts::{GI_DEN_PLANE_K, GI_PI, GI_SS_M_CAP, SHADOW_SURFACE_EPS};
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
//      ⇒ 每像素射线数：新鲜 1 条（含 ≤1 条 NEE 阴影）+ 重连接 ≤1 条 = 1~3 条。
//   ③ 空间：3×3 的 4 个轴邻（读上帧 reservoir 的同屏坐标）。**仅主命中面键完全一致时直接并**
//      （同一个体素同一个面 ⇒ 目标函数逐字相同 ⇒ 无偏、零射线开销）；键不一致的跨面复用本步不做
//      （那需要重连接射线，而射线预算留给时域验证）。
//   ④ RIS 合并：余弦候选 w = π·L（余弦 pdf 下 f/p = L·cosθ·π/cosθ）；`w_sum += Σw`、`M += 候选数`，
//      `M` 触及 `GI_SS_M_CAP` 时把 w_sum 与 M 同比例缩回（抑制历史支配）；估计 = `w_sum / M`
//      = π·⟨L⟩（与旧缓存 `E_store` 同语义）⇒ 写进 `gi_out` 前再乘增益、除 π（见 `gi_ss_main`）。
//      每帧只插 1 个**独立**的余弦候选，其余全部来自「已通过精确键筛的累计量」⇒ 估计是同一个积分
//      的运行平均（无偏），M 上限决定记忆长度（≈ M_CAP / 每帧候选数 帧）。
//
// 薄板/墙缝安全：主键里的**面**（`gi_key1` 的 3 bit）使薄板两侧是两个不同的键，键不等就一律不接；
// 二次顶点键同理（射线穿过板子后打到的面与历史不同 ⇒ 丢弃历史样本）。
// ============================================================================

/// reservoir 字数（Rust 侧从本文件解析同名字面量开 buffer）。
const GI_RES_WORDS: u32 = 13u;
/// word 布局。
const GI_RES_W_DIR: u32 = 0u; // 选中样本方向（八面体 f16x2）
const GI_RES_W_RAD: u32 = 1u; // 选中样本的入射辐亮度 L（rgb）
const GI_RES_W_WSUM: u32 = 4u; // 累计权重 Σ w（rgb = π·L 的和）
const GI_RES_W_M: u32 = 7u; // 候选数 M
const GI_RES_W_PK: u32 = 8u; // 主命中面键（gi_key0 / gi_key1）
const GI_RES_W_SK: u32 = 10u; // 选中样本的二次顶点键（gi_key0 / gi_key1）
const GI_RES_W_FLAGS: u32 = 12u; // 状态位（见下）
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
}

/// 空 reservoir（天空 / 介质像素）：`M = 0` ⇒ 下一帧任何时序/空间复用都判为无效，不会读到脏值。
fn gi_res_none() -> GiRes {
  return GiRes(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0), 0.0, 0u, 0u, 0u, 0u, 0u);
}

/// 首个候选（新鲜射线）建 reservoir。
fn gi_res_new(pk0: u32, pk1: u32, pk_ok: bool, dir: vec3<f32>, rad: vec3<f32>, sk0: u32,
               sk1: u32, sk_hit: bool, sk_ok: bool) -> GiRes {
  var flags = GI_RES_VALID;
  if (pk_ok) { flags = flags | GI_RES_PK_OK; }
  if (sk_hit) { flags = flags | GI_RES_SK_HIT; }
  if (sk_ok) { flags = flags | GI_RES_SK_OK; }
  return GiRes(dir, rad, rad * GI_PI, 1.0, pk0, pk1, sk0, sk1, flags);
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
  var r = gi_res_new(pk0, pk1, pk_ok, dir_f, s_f.l, s_f.sk0, s_f.sk1, s_f.sk_hit, s_f.sk_ok);

  if (pk_ok) {
    // ---- ② 时域复用：重投影 + 精确键筛（主键 + 二次顶点键）----
    // （reservoir 沿用体素中心重投影 —— 那是上一步定下的语义，不动它。）
    var prev_ok = false;
    let pc = gi_ss_reproject(p_voxel, size, &prev_ok);
    if (prev_ok) {
      let hist = gi_res_load_prev(u32(pc.y) * size.x + u32(pc.x));
      // 主键必须完全一致：不同体素 / 不同面 / 不同物体一律不接（薄板两侧是两个键）。
      if ((hist.flags & GI_RES_VALID) != 0u && (hist.flags & GI_RES_PK_OK) != 0u
          && hist.pk0 == pk0 && hist.pk1 == pk1) {
        // 重连接射线（本帧唯一的 ≤1 预算）：沿历史方向重发，用它的二次顶点键验证历史样本。
        // 它**只用于验证**、不并进 RIS：同一个方向每帧被重加，会让 `w_sum / M` 逐渐「锁定」到
        // 那个方向的 L —— 抽样是按亮度加权的，被锁定的必然偏亮 ⇒ 估计系统性偏亮（实测 ~+25%）。
        // 只需要键 ⇒ 走轻量版（不做 NEE / 不读缓存），这条射线的开销就是一条裸 `world_raycast`。
        let s_r = gi_ray_secondary_key(org, hist.dir);
        let same_sk = s_r.sk_hit == ((hist.flags & GI_RES_SK_HIT) != 0u)
          && s_r.sk_ok && s_r.sk0 == hist.sk0 && s_r.sk1 == hist.sk1;
        if (same_sk) {
          // 二次顶点键逐位一致 ⇒ 历史样本就是「本面看到的同一个方向的同一段光路」⇒ 累计量有效。
          // 累计量是**独立余弦样本权重的运行平均**（每帧只进 1 个新鲜候选）⇒ 无偏，且随时间降噪。
          r = gi_res_adopt(r, hist);
        }
      }
    }

    // ---- ③ 空间复用：3×3 的 4 个轴邻，仅「主命中面键完全一致」直接并 ----
    for (var s = 0u; s < 4u; s = s + 1u) {
      var off = vec2<i32>(0, 0);
      if (s == 0u) { off = vec2<i32>(1, 0); }
      else if (s == 1u) { off = vec2<i32>(-1, 0); }
      else if (s == 2u) { off = vec2<i32>(0, 1); }
      else { off = vec2<i32>(0, -1); }
      let nc = coord + off;
      if (nc.x < 0 || nc.y < 0 || nc.x >= i32(size.x) || nc.y >= i32(size.y)) { continue; }
      let nb = gi_res_load_prev(u32(nc.y) * size.x + u32(nc.x));
      if ((nb.flags & (GI_RES_VALID | GI_RES_PK_OK)) != (GI_RES_VALID | GI_RES_PK_OK)) { continue; }
      // 精确键筛：只有同一个体素同一个面才直接并（同一目标函数 ⇒ 无偏、零射线）。
      if (nb.pk0 == pk0 && nb.pk1 == pk1) { r = gi_res_adopt(r, nb); }
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
