import package::bindings::{gi_sec_slots, light_u};
import package::common::{MATERIAL_DEBUG_NONE, SurfaceLight, brdf_reflected, emissive_radiance, face_view_dir, fetch_material, sky_rgb};
import package::gi::common::{GI_SEC_WORDS, GI_SEC_W_K0, GI_SEC_W_K1, GI_SEC_W_RAD, gi_cos_hemisphere, gi_key0, gi_key1, gi_key_ok, gi_rand2, gi_sec_key_masked, gi_sec_slot_of};
import package::gi::consts::{GI_EMIT_GAIN, GI_PI, GI_SKY_AMBIENT, GI_SKY_RADIANCE_SCALE, GI_T_MAX, SHADOW_SURFACE_EPS};
import package::lightfield::{light_field_ao};
import package::trace::{make_grid};
import package::world::{WorldHit, voxel_center_world, voxel_normal, world_trace_medium};

// ============================================================================
// 单条 GI 射线的辐射求值（**唯一口径**：新鲜候选、借用候选、时域/空间复用全部走这里）。
//
// 一次求值 = 1 条真实光路（`world_trace_medium`，**穿过透射介质续行**）＋ 命中时最多 1 条太阳
// NEE 阴影射线。`sun_bounce = 0`（菜单「渲染/RESTIR GI/太阳反弹」关）时连那条阴影射线都不发，
// 二次顶点只剩天光项。
//
// **辐亮度只算一次弹射**：命中面按「直射太阳 + 天光 × AO」着色后除以 π 得到它的出射辐亮度，
// 没有更深的弹射。多重弹射曾经靠世界空间缓存的回读实现（`gi_cache_peek`），那条路是个
// **跨帧反馈环**（缓存值进射线、射线值又进缓存，且缓存用 `1/n` 运行平均收敛）——
// 表现就是「像拿刷子一层层刷、稳态要等二十秒、有无 GI 的异常区」。删除后 GI 的输入
// 只依赖几何与光照，是**逐帧确定**的：没有可累积的中间量，也就没有「收敛」这件事。
// 代价是丢掉二次以上的弹射（多面互反射），这是刻意的取舍。
//
// 出射辐亮度约定（**与 `dda_main` 的不透明分支逐项对齐**，硬约束 3 / `docs/PLAN.md` MT3-4）：
//   `L_o = BRDF(材质, 二次顶点, 光源项) / π + albedo · emissive`
// 其中 `BRDF(...)` = `common.wesl::brdf_reflected` —— **与主 pass 同一个求值函数**
// （MT3-7 的"变体统一"落到 GI 侧就是这件事），材质也走同一个 `fetch_material` 入口。
// `/π` 只作用在**反射项**上（把"辐照度量纲"的出射亮度折成 radiance）；自发光本来就是辐亮度、
// **不除**（`emissive_radiance` 单独成函数就是为了这条）。天光项在二次顶点乘 `light_field_ao`
// （与主着色点同一套 AO，口径不分叉）。
//
// 逃逸到天空 ⇒ `sky_rgb()`（唯一的 miss 辐亮度口径）。
// ============================================================================

/// 太阳方向的直光透过率（1 = 无遮挡，0 = 被不透明体素全挡），**一次 `world_trace_medium` 走完**。
/// `stop_on_air = false` ⇒ 空气与透射体素都不停、只在不透明体素上停 ⇒ 命中即「被挡住」，
/// 沿途穿过的透射介质累积进 `tint`（旧写法是「先探一次不透明首命中；若它是透射材质，
/// 再把已经走过的那一段整段丢掉、从同一 origin 重走一遍介质」—— 两条路等价，
/// 但玻璃场景下旧写法要打**两段**遍历，本写法只有一段）。
/// `self_voxel`/`self_obj`：起点贴命中面外侧，豁免 t≈0 自命中。
/// 与旧写法的唯一差别在「起点落在**透射**的命中体素内部」这种 UB 情形：旧写法按自命中返回 1.0，
/// 本写法按它后面的不透明遮挡给值 —— 起点已沿法线外推半个体素 + eps，正常路径不会走到那里。
fn sun_transmittance(origin: vec3<f32>, sun_dir: vec3<f32>, self_voxel: vec3<i32>, self_obj: i32) -> f32 {
  var tint = vec3<f32>(1.0);
  let h = world_trace_medium(make_grid(0u), origin, sun_dir, GI_T_MAX, false, &tint);
  let through = (tint.r + tint.g + tint.b) / 3.0;
  // 段内没有不透明遮挡 ⇒ 只有介质的乘性透过（无介质时恒 1.0）。
  if (!h.hit) { return through; }
  if (all(h.voxel == self_voxel) && h.obj_id == self_obj) { return 1.0; }
  return 0.0;
}

/// 一条 GI 射线的求值结果。
/// * `l` = 入射辐亮度（**不含** cos 因子与 π；复用候选的系数由调用方按各自 pdf 给）。
/// * `sk0`/`sk1` = 二次顶点的**精确键**（`gi_key0`/`gi_key1` 口径：命中体素 + 面 + 物体）。
///   `sk_hit = false` = 射线逃逸到天空（键取 (0,0)，同向天空两次求值必然相等）。
/// * `sk_ok` = 键可编码且可比较。复用**只在 `sk_ok` 为真时接受历史样本**
///   （键编码范围外无法做精确比较 ⇒ 宁可不复用，也不用启发式）。
struct GiRaySample {
  l: vec3<f32>,
  sk0: u32,
  sk1: u32,
  sk_hit: bool,
  sk_ok: bool,
}

/// **二次顶点着色（唯一实现）**：命中面的出射辐亮度（**不含**射线自身的介质透过率 `tint`，
/// `tint` 由调用方逐射线乘）。
///
/// 全部输入都只由"命中的那个可见体素面"决定 ⇒ 可以按面缓存：
///   · 采样点一律取**体素中心**（`voxel_center_world`），与主 pass 的 `p_voxel` 同口径。它同时是
///     材质取值的点（`fetch_material` 内部本就量化到体素中心，传体素中心让 mip LOD 的距离也逐面
///     确定）、太阳 NEE 起点的基准（沿面法线外推半体素 + eps）、AO 的采样点；
///   · 视向量锚点 = 同一个体素中心（`face_view_dir` 那份契约）；
///   · 着色法线是 (voxel, face) 的纯函数。
///
/// 出射辐亮度约定与 `dda_main` 的不透明分支逐项对齐（硬约束 3）：`BRDF(...) / π + albedo·emissive`
/// —— 同一个 `brdf_reflected`、同一个 `fetch_material`；`/π` 只作用于反射项（自发光本就是 radiance）。
/// 与主 pass 的差异只有两条、都是物理上该不同的：① 太阳项乘 `sun_bounce`（菜单「太阳反弹」关掉时
/// 二次顶点不做 NEE）；② 没有 GI 项（二次顶点只算一次弹射）⇒ `SurfaceLight.gi` 传 0。
///
/// **不含 bounce-2**：这里只出"一次弹射"的辐亮度 —— 它必须是与帧无关的纯函数（② 的跨帧持久缓存
/// 就吃这一点）。第二条弹射由 `gi_bounce2` 在**缓存之外**逐候选叠加（见那个函数的说明）。
fn gi_face_shade(sh: WorldHit, org: vec3<f32>, sun_dir: vec3<f32>, sun_c: vec3<f32>,
                 sun_bounce: f32) -> vec3<f32> {
  let hg = make_grid(u32(sh.obj_id) + 1u);
  let hp = voxel_center_world(hg, sh.voxel);
  let hn = voxel_normal(hg, sh.voxel, sh.face_id, sh.n);
  let ndl = max(dot(hn, sun_dir), 0.0);
  var sun = 0.0;
  // `sun_bounce` 来自 uniform（一次派发内恒定 ⇒ 分支不发散）：关掉时整条阴影射线都不发。
  if (sun_bounce > 0.0 && ndl > 0.0) {
    let h_off = 0.5 / max(max(abs(hn.x), abs(hn.y)), max(abs(hn.z), 1e-3)) + SHADOW_SURFACE_EPS;
    sun = sun_transmittance(hp + hn * h_off, sun_dir, sh.voxel, sh.obj_id);
  }
  // 天光项：与主着色点同一套系数 × 同一套 AO（口径不分叉）。
  let amb = light_u.sky_color.xyz * GI_SKY_AMBIENT * light_field_ao(hp, sh.obj_id);
  // 视向量：**逐面**（与 `dda_main` 共用 `face_view_dir`）。`org` = 这条 GI 射线自己的起点（不是
  // 相机眼位：二次顶点"回到观察者"的方向就是回到上一个顶点）；锚点按面量化 ⇒ 它不再精确等于 -dir。
  let v = face_view_dir(org, hp);
  let m = fetch_material(sh.palette_base, sh.pal, hp, hn, MATERIAL_DEBUG_NONE);
  return brdf_reflected(m, hn, v, SurfaceLight(sun_dir, sun_c, sun * sun_bounce, amb, vec3<f32>(0.0)))
         / GI_PI
       + emissive_radiance(m, GI_EMIT_GAIN);
}

/// `gi_face_shade` 的**带缓存**版本（`gi_sec_slots`：epoch 掩码键、跨帧持久，见 `gi/common.wesl`）。
/// 同一个面在同一 epoch 内被成千上万条射线打到时只算一次（含它那条太阳 NEE 阴影射线）。
/// 读侧两次键校验、写侧「辐亮度 → 键」的顺序与撕裂窗口的说明都在 `gi/common.wesl`。
fn gi_face_shade_cached(sh: WorldHit, org: vec3<f32>, sun_dir: vec3<f32>, sun_c: vec3<f32>,
                        sun_bounce: f32, epoch: u32, k0: u32, k1: u32, key_ok: bool) -> vec3<f32> {
  var l = vec3<f32>(0.0);
  var slot_ok = false;
  var sb = 0u;
  var mine = vec2<u32>(0u, 0u);
  if (key_ok) {
    let slot = gi_sec_slot_of(k0, k1, &slot_ok);
    if (slot_ok) {
      sb = slot * GI_SEC_WORDS;
      mine = gi_sec_key_masked(k0, k1, epoch);
      if (atomicLoad(&gi_sec_slots[sb + GI_SEC_W_K0]) == mine.x
          && atomicLoad(&gi_sec_slots[sb + GI_SEC_W_K1]) == mine.y) {
        let r = vec3<f32>(
          bitcast<f32>(atomicLoad(&gi_sec_slots[sb + GI_SEC_W_RAD])),
          bitcast<f32>(atomicLoad(&gi_sec_slots[sb + GI_SEC_W_RAD + 1u])),
          bitcast<f32>(atomicLoad(&gi_sec_slots[sb + GI_SEC_W_RAD + 2u])));
        // **两次键校验**：写入顺序是「辐亮度 → 键」，所以"两次读到的键都是本键"⇒ 中间那次辐亮度
        // 必定来自本面本 epoch 的写入者（见 `gi/common.wesl` 的撕裂窗口说明）。
        if (atomicLoad(&gi_sec_slots[sb + GI_SEC_W_K0]) == mine.x
            && atomicLoad(&gi_sec_slots[sb + GI_SEC_W_K1]) == mine.y) {
          return r;
        }
      }
    }
  }
  l = gi_face_shade(sh, org, sun_dir, sun_c, sun_bounce);
  // 只在"槽里不是本键"时写（已经是本键就说明值已落地，重复写只是白费带宽）。
  // 允许覆写**别的键**（含上一个 epoch 留下的陈旧项）⇒ 不需要任何清表。
  if (slot_ok) {
    let c0 = atomicLoad(&gi_sec_slots[sb + GI_SEC_W_K0]);
    let c1 = atomicLoad(&gi_sec_slots[sb + GI_SEC_W_K1]);
    if (c0 != mine.x || c1 != mine.y) {
      atomicStore(&gi_sec_slots[sb + GI_SEC_W_RAD], bitcast<u32>(l.x));
      atomicStore(&gi_sec_slots[sb + GI_SEC_W_RAD + 1u], bitcast<u32>(l.y));
      atomicStore(&gi_sec_slots[sb + GI_SEC_W_RAD + 2u], bitcast<u32>(l.z));
      atomicStore(&gi_sec_slots[sb + GI_SEC_W_K0], mine.x);
      atomicStore(&gi_sec_slots[sb + GI_SEC_W_K1], mine.y);
    }
  }
  return l;
}

/// **第二条弹射**（菜单「渲染/光照/二次弹射」）：在二次顶点 `sh` 上再发**一条**余弦加权射线，
/// 返回它的入射辐亮度 `L2`（`gi_face_shade` 口径；逃逸取天光）。**不含** `kd·albedo` 系数 ——
/// 系数由调用方 `gi_bounce2` 乘（它手上有命中面的材质）。
///
/// 为什么是"一条余弦射线"就够了（无偏）：`brdf_reflected` 的环境项本就按
/// `(kD·albedo + F0)·(amb + gi)` 消费**余弦平均后的入射辐亮度**（`amb` / `gi` 都是那个量纲）
/// ⇒ 单条余弦射线的 `L2` 就是该均值的一个无偏估计，直接当一次弹射的环境项用。
///
/// 为什么**不进 `gi_sec_slots`**：它的方向要逐候选变（见 `gi_bounce2` 的"为什么方向按候选/帧变"），
/// 而那张表要求"同一面同一 epoch 内逐位相同" ⇒ 进表就会把逐帧新样本冻成常数。
/// 命中面的着色**走缓存**（`gi_face_shade_cached`）⇒ 第三条顶点那条太阳 NEE 仍然是"每面每 epoch
/// 一次"，不是每条射线一次（这是本项成本可控的关键）。
fn gi_bounce2_probe(org: vec3<f32>, dir: vec3<f32>, sun_dir: vec3<f32>, sun_c: vec3<f32>,
                    sun_bounce: f32, epoch: u32) -> vec3<f32> {
  var tint = vec3<f32>(1.0);
  let h = world_trace_medium(make_grid(0u), org, dir, GI_T_MAX, false, &tint);
  if (!h.hit) {
    return sky_rgb() * GI_SKY_RADIANCE_SCALE * tint;
  }
  let k0 = gi_key0(h.voxel);
  let k1 = gi_key1(h.voxel, h.face_id, h.obj_id);
  return gi_face_shade_cached(h, org, sun_dir, sun_c, sun_bounce, epoch, k0, k1, gi_key_ok(h.voxel))
         * tint;
}

/// 命中面 `sh` 的**二次弹射项**（辐亮度量纲，已含 `kD·albedo` 与调用方给的倍数 `mult`）。
/// `mult` = 0 → 关（整支被折掉，连那条射线都不发）；`mult` = 1 → 每个候选都追；
/// `mult` > 1 → **弹射分裂**：只有 `1/mult` 的候选带上这一项、并把它乘 `mult` 补齐期望值
/// （RIS 的权重就是 `π·L`，而 `L` 本身是"含这一项"的无偏估计 ⇒ RIS 依旧自洽、估计无偏，只是方差变大）。
///
/// **方向按候选（含帧号）变**：这一项不在任何缓存里，方向随机化才能让降噪链的时域累积把它平均掉。
/// 若按面取恒定方向（MT8-6 对反射的做法）它就成了一个**冻住的**逐面常数：噪声降不成"逐面小块"，
/// 而是永远偏在那个方向上 —— 对"一次弹射已完全确定"的 GI 来说那是一条更差的取舍。
fn gi_bounce2(sh: WorldHit, org: vec3<f32>, sun_dir: vec3<f32>, sun_c: vec3<f32>,
              sun_bounce: f32, epoch: u32, seed: u32, mult: f32) -> vec3<f32> {
  let hg = make_grid(u32(sh.obj_id) + 1u);
  let hp = voxel_center_world(hg, sh.voxel);
  let hn = voxel_normal(hg, sh.voxel, sh.face_id, sh.n);
  let m = fetch_material(sh.palette_base, sh.pal, hp, hn, MATERIAL_DEBUG_NONE);
  let off = 0.5 / max(max(abs(hn.x), abs(hn.y)), max(abs(hn.z), 1e-3)) + SHADOW_SURFACE_EPS;
  let d2 = gi_cos_hemisphere(hn, gi_rand2(seed));
  let l2 = gi_bounce2_probe(hp + hn * off, d2, sun_dir, sun_c, sun_bounce, epoch);
  // `kD·albedo·L2`：与 `brdf_reflected` 的环境项同一个量纲与相位（见 `gi_bounce2_probe`）。
  return (1.0 - m.metallic) * m.albedo * l2 * mult;
}

/// 一条 GI 射线的入射辐亮度：真实光路 + 二次顶点「太阳 NEE + 天光 × AO」，逃逸取天光。
/// 不含 cos 因子与 π —— RIS 的权重是 `p̂ / p`，系数在调用方给（余弦候选给 π）。
///
/// 光路用 `world_trace_medium`（`stop_on_air = false`）：撞到**透射**体素不停、沿途按格累积乘性吸收
/// `tint` ⇒ 从玻璃前面出发的射线会**穿过玻璃**打到它后面的真实表面。`tint` 乘在**整个** `l` 上
/// （BRDF 项与自发光项都乘）—— 与主 pass 的 `col = col * through + refl` 是同一个"乘性吸收"结构
/// （硬约束 3：口径不分叉）。只有不透明体素才停 ⇒ 停点与 `sk0/sk1` 的口径不变（硬约束 2）。
/// 与 `sun_transmittance` 同一套写法；只遍历**主世界网格**（物体网格没有介质续行，且
/// `Volumes::add_object` 在应用里没有调用点 ⇒ 无实际行为差异）。
/// 命中分支的着色走 `gi_face_shade_cached`（按面缓存：同一个面被多条射线打到时只算一次），
/// 命中面的**二次弹射**由 `gi_bounce2` 在缓存之外叠加（`bounce2` = 倍数，0 = 关；`seed` 是候选种子）。
/// `epoch`（uniform `gi_u.seq.y`）= 光照/几何/光照场窗口的修订号，参与槽里键的掩码 ⇒ 它一变整张表
/// 自失效（跨帧持久，见 `gi/common.wesl` 的「跨帧持久」段）。
fn gi_ray_radiance(org: vec3<f32>, dir: vec3<f32>, sun_dir: vec3<f32>, sun_c: vec3<f32>,
                    sun_bounce: f32, epoch: u32, bounce2: f32, seed: u32) -> GiRaySample {
  var tint = vec3<f32>(1.0);
  let sh = world_trace_medium(make_grid(0u), org, dir, GI_T_MAX, false, &tint);
  // 逃逸 → 天光辐亮度（穿过介质看到的天空同样按 `tint` 吸收）；天空没有面 ⇒ 不进缓存。
  if (!sh.hit) {
    return GiRaySample(sky_rgb() * GI_SKY_RADIANCE_SCALE * tint, 0u, 0u, false, true);
  }
  let k0 = gi_key0(sh.voxel);
  let k1 = gi_key1(sh.voxel, sh.face_id, sh.obj_id);
  let key_ok = gi_key_ok(sh.voxel);
  var l = gi_face_shade_cached(sh, org, sun_dir, sun_c, sun_bounce, epoch, k0, k1, key_ok);
  if (bounce2 > 0.0) {
    l = l + gi_bounce2(sh, org, sun_dir, sun_c, sun_bounce, epoch, seed, bounce2);
  }
  return GiRaySample(l * tint, k0, k1, true, key_ok);
}

/// 只求**二次顶点键**的轻量射线（时域复用的二次顶点验证专用）：键的口径与 `gi_ray_radiance`
/// **逐字一致**（同一次 `world_trace_medium` + `stop_on_air = false`、同一套 `gi_key0/gi_key1`
/// —— MT5-2 后两侧都必须穿介质续行，否则"玻璃后面的那个面"这条键在两次求值间会不一致，
/// 时域验证会恒判不等、白白丢掉历史），但**不做**二次顶点 NEE、不取 albedo ⇒ 少发一条阴影射线。
/// `l` 恒为 0（调用方只用键），`tint` 也只为满足接口、无消费者。
fn gi_ray_secondary_key(org: vec3<f32>, dir: vec3<f32>) -> GiRaySample {
  var tint = vec3<f32>(1.0);
  let sh = world_trace_medium(make_grid(0u), org, dir, GI_T_MAX, false, &tint);
  return GiRaySample(
    vec3<f32>(0.0),
    gi_key0(sh.voxel),
    gi_key1(sh.voxel, sh.face_id, sh.obj_id),
    sh.hit,
    !sh.hit || gi_key_ok(sh.voxel),
  );
}