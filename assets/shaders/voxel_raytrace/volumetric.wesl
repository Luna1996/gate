import package::bindings::{BEAM_BACKOFF, BEAM_DIV, beam_depth, fog_guide_r, fog_guide_t, fog_guide_w, fog_hist_in, fog_hist_out, fog_out, fog_phi_w, fog_raw_in, fog_tmp_out, fog_u, fog_u_read, light_u, view_u};
import package::common::{sky_rgb};
import package::gi::common::{gi_key0, gi_key1, gi_mix, gi_pack_dir, gi_rand2};
import package::gi::consts::{GI_DEN_GUIDE_WORDS, GI_DEN_G_KEY0, GI_DEN_G_KEY1, GI_DEN_G_NRM, GI_DEN_G_PREV, GI_LUMA};
import package::gi::denoise::{gi_den_same_plane};
import package::trace::{make_grid};
import package::world::{face_normal_world, world_cfg_full, world_raycast, world_trace_medium};

// ============================================================================
// 体积散射（godray / 光轴）—— **唯一路径 = 视图射线上的单次散射 + 逐样本真阴影射线**
//
// 为什么是这条路（而不是屏幕空间径向模糊 / froxel / 世界空间预计算）：
//   · 本引擎**没有 shadow map**：阴影本来就是一条 DDA 射线（`gi/ray.wesl::sun_transmittance`），
//     逐样本精确、无 bias、无级联接缝 ⇒ 光轴的形状直接由真实几何决定，不需要为体积光再引入一套
//     阴影贴图（那会引入两个阴影口径）。这是本引擎做体积光最大的红利。
//   · 太阳会动（昼夜循环）⇒ 任何世界空间预计算（逐体素太阳透过率 / VLM）都不成立：
//     太阳每变一次就得整块重建。**不缓存太阳可见性**，每帧现算。
//
// 成本模型（决定了整个算法形态）：一条雾样本的阴影射线 ≈ 一条全屏射线的量级 ⇒
// 「每像素每帧打 N 条采样射线」是不可接受的（N = 8 就是全屏 8 条射线）。所以：
//   · **每像素每帧只采 S 条**（S = 1..4，档位给），沿视线的位置用**自由飞行分布**抽（见 `fog_free_flight`）
//     —— 它一次采样就把「沿整段视线的单次散射积分」无偏地估出来，不需要步进；
//   · **环境项（天光散射）用闭式解**，一条射线都不发；
//   · 剩下的方差交给**时域累积**（`fog_den_temporal`，本文件）+ **复用 GI 的迭代 atrous**（空间段）。
// 于是「S = 1 + 时域累积」就能收敛出连续的光柱，而帧成本是「每像素 1 条阴影射线 + 1 条主射线」。
//
// 单次散射的积分（**含地面雾**：密度沿视线 `ρ(t) = exp(−h·dir.y·t)`，相机高度处 ρ = 1）：
//   L(x) = ∫₀^{t_end} σ_s·ρ·e^{−τ(t)}·( sun_c·phase·vis(t) + amb ) dt
//        = (σ_s/σ_t)·(1 − e^{−τ_max})·E[ sun_c·phase·vis(t) + amb ]，τ 服从截断指数分布
// ⇒ 太阳项 = 自由飞行采样（`vis` 由**真阴影射线**给出），环境项 = 闭式解。
// `h` = 0 时 `τ(t) = σ_t·t`，整套退化成最朴素的形式（`fog_optical_depth` / `fog_tau_to_t` 都有 `a → 0` 支路）。
//
// ---- 「地面雾」（高度衰减）不是可选的润色，它是"利落 vs 雾霾"的开关 ----
// 均匀介质在**室外长视线**上的 `1 − e^{−σt·d}` 会积到 1：一条几乎不随方向变化的加性薄纱盖在所有像素上
// ⇒ 整幅画面像雾霾天（这是本档最早版本的实测症状）。而参考观感是「抬头看天干净、只有近地那层空气有光柱」——
// 真实的光柱也基本只出现在**近地/室内那层**（室内空气静止、有尘；室外高空干净）。
// 密度以**相机高度**为基准（ρ = 1 在眼睛高度）有两个好处：
//   ① 无论场景建在什么高度、相机飞多高，"眼前那层雾"的厚度恒定（不用跟着世界原点调参数）；
//   ② 往上看指数变薄（天空迅速干净），往下看变浓（地面雾的自然形态）。
// 代价：**零**（只是把 `σt·t` 换成闭式积分，见上面两个助手），并且**不引入任何近似**（抽的还是光学深度）。
//
// 与 `dda_main` 的合成点：**线性 HDR、曝光之前**（`write_output` 里才乘曝光）⇒ 雾与场景同一条曝光曲线。
// 合成只做**加法**（`col + 雾`）：本档是「光柱叠加」而不是「空气透视」——
// **不衰减背景**。这是刻意的取舍：参考观感是「背景保持自己的颜色与对比，只有阳光照到的那段雾发亮」；
// 一旦乘上 `e^{-τ}`，几百 voxel 深的场景会被整体压暗、变成"起雾"。
// 想要空气透视/乳白雾感就把「环境散射」调大（那会给整幅画面叠一层各向同性的散射）。
//
// 有意不做的（知道缺什么才不算到顶）：
//   · 多重散射：只有单次 + 各向同性环境近似（「光柱增益」把丢掉的那部分能量补回来）；
//   · 背景衰减（见上）；
//   · 雾不遮蔽 GI 射线、也不给 GI 贡献能量（GI 链一字未动）；
//   · 玻璃后面的雾（主命中是介质时不续行，停在界面）。
// ============================================================================

/// 每帧每个雾像素的**采样数上限**（uniform 里的值按它硬夹）。档位表在 Rust `FogSettings`：
/// 低 = 1、中 = 1、高 = 2、极高 = 4（**改这里要同步那张表**：本值必须 ≥ 表内最大档）。
const FOG_SAMPLE_MAX: u32 = 4u;
/// 单次散射反照率 σ_s/σ_t（1.0 = 不吸收）：0.9 让厚雾略微自吸收、不糊成一片白。
const FOG_ALBEDO: f32 = 0.9;
/// 天空像素（主射线 miss）的**哨兵面键** = `(0, 1)`。
/// 为什么可用：真实命中的键恒满足 `k0 ≥ 1`（`voxel.x ≥ −4095` ⇒ `+4096 ≥ 1`）与
/// `k1 ≥ 0x10000`（`obj+1 ≥ 1` 在 bit16 以上）⇒ 哨兵与任何真实面**不可能撞键**；
/// 而所有天空像素共用它 ⇒ 它们按 `gi_den_same_plane` 互为「同一个平面」⇒ 天空那部分照常时域累积
/// （光柱打在天空上正是最想看的画面，不能因为「没有面」就被降噪链判成无效）。
/// 前提：命中体素落在键的可编码范围内（|voxel| < 4096），与 `gi_key_ok` 的假设同源。
const FOG_SKY_K0: u32 = 0u;
const FOG_SKY_K1: u32 = 1u;
/// 太阳锥的**半角上限**（rad）：uniform 里的值按它硬夹（防止滑杆滑出太阳盘的量级）。
const FOG_CONE_MAX: f32 = 0.06;
/// 太阳盘的亮度倍数（乘 `sun_c`）：盘要**过曝**才是"太阳"（8 位输出里直接打到白）。
/// 与 `日晕` 是两个不同的量：盘是"那个圆"，晕是它外面那圈拖尾（见 `sky_primary`）。
const SUN_DISK_GAIN: f32 = 40.0;

/// **主射线看到的天空**（`dda_main` 的 miss 背景）：常量天光 + **太阳盘** + **日晕**。
/// 与 `sky_rgb()` 的分工（硬约束：口径不分叉）：`sky_rgb()` 是**间接光的天空项**
/// （GI 的 miss、玻璃与镜面反射的逃逸），太阳作为光源已经由各处 NEE / `sun_c` 负责
/// ⇒ 那些地方**不能再见到太阳盘**（会把同一份能量算两遍）。本函数只服务「看得见的那颗太阳」。
/// 因为只在**主射线 miss** 时求值，太阳被树/墙挡住时它自动消失（不需要额外的可见性判定）。
/// 盘半径与阴影射线的锥角是**同一个量**（`misc2.z`）：两者都是"太阳的角径"。
fn sky_primary(dir: vec3<f32>, sun_dir: vec3<f32>, sun_c: vec3<f32>) -> vec3<f32> {
  let base = sky_rgb();
  let ang = clamp(fog_u_read.misc2.z, 1e-4, FOG_CONE_MAX);
  let cos_t = dot(dir, sun_dir);
  // 盘：角半径内是硬核（外沿 15% 走 `smoothstep` 抗锯齿 —— 盘只有几个像素宽，硬边会闪）。
  let disk = smoothstep(cos(ang), cos(ang * 0.85), cos_t);
  // 晕：盘外的能量拖尾（解析的，不占任何 pass）。指数取 96：5° 处 ≈0.7、15° 处 ≈0.04。
  let halo = pow(max(cos_t, 0.0), 96.0);
  return base + sun_c * (disk * SUN_DISK_GAIN + halo * max(fog_u_read.misc2.w, 0.0));
}

/// Henyey-Greenstein 相位函数（单次散射的**方向**因子）。
/// 约定（PBRT 口径，**写在这里以免再次搞反**）：`cos_t = dot(ω_o, ω_i)`，两个方向都**从散射点指出** ——
/// `ω_o` = 指向观察者（= −视线方向 `dir`）、`ω_i` = 指向太阳（= `light_u` 的 `L`，它"已归一、指向光"）。
///   `p = (1 − g²) / (4π · (1 + g² + 2g·cos_t)^{3/2})` —— 注意是 **+2g**（不是 −2g）。
/// `g > 0`（前向散射）的峰值落在 `cos_t = −1`，即 **视线正对太阳**：光沿自己的传播方向继续走
/// ⇒「朝着光源看最亮」（车灯前那团光晕）。**物理自检**：`g` 取正数时朝着太阳看应当最亮；
/// 若反过来（背对太阳最亮），就是把 `+2g` 写成了 `−2g`（或 cos 少了一个负号）——两者只差这一个符号。
fn fog_phase(g: f32, cos_t: f32) -> f32 {
  let g2 = g * g;
  let d = max(1.0 + g2 + 2.0 * g * cos_t, 1e-4);
  return (1.0 - g2) / (12.5663706 * d * sqrt(d));
}

/// 段内的**自由飞行**位置：t 服从「第一次散射发生在 t」的条件分布（截断到 `[0, t_end]`）。
/// **抽的是光学深度 τ 而不是距离** —— 这一点让「地面雾」（密度沿视线变化）**不需要任何近似**：
/// 散射事件的光学深度分布恒是截断指数 `p(τ) = e^{−τ}/(1−e^{−τ_max})`（与介质沿视线是否均匀无关），
/// 于是 `τ = −ln(1 − u·(1−e^{−τ_max}))`，再反解回 t（见 `fog_tau_to_t`）。
fn fog_free_flight(u: f32, tau_max: f32) -> f32 {
  return -log(max(1.0 - u * (1.0 - exp(-tau_max)), 1e-6));
}

/// **地面雾**的光学深度：密度沿视线 `ρ(t) = exp(−a·t)`（`a = h·dir.y`，相机高度处 `ρ = 1`，见文件头）
/// ⇒ `τ(t_end) = ∫σt·ρ dt = σt·(1 − e^{−a·t_end})/a`。
/// `a → 0`（水平看 / 没开地面雾）退化成 `σt·t_end`；这里用一阶判据避开 0/0（两条支路在交界处连续）。
fn fog_optical_depth(sigma_t: f32, a: f32, t_end: f32) -> f32 {
  if (abs(a * t_end) < 1e-3) { return sigma_t * t_end; }
  return sigma_t * (1.0 - exp(-a * t_end)) / a;
}

/// `fog_optical_depth` 的**逆**：光学深度 → 距离（自由飞行采样抽到 τ 之后要落回 t）。
fn fog_tau_to_t(tau: f32, sigma_t: f32, a: f32) -> f32 {
  if (abs(a) < 1e-6) { return tau / max(sigma_t, 1e-9); }
  return -log(max(1.0 - a * tau / max(sigma_t, 1e-9), 1e-6)) / a;
}

/// 以 `r` 为轴、半角 `cone` 的锥内取一个方向（锥内**面积均匀**；与 `main.wesl::refl_cone_dir`
/// 同一套写法）。它代表**太阳的角半径**：一条阴影射线 = 太阳盘上的一个样本 ⇒
/// 多帧累积后光柱边缘自然变软（免费拿到半影，见 `fog_cone_max` 的由来）。
fn fog_cone_dir(r: vec3<f32>, u: vec2<f32>, cone: f32) -> vec3<f32> {
  let c = clamp(cone, 0.0, FOG_CONE_MAX);
  if (c <= 1e-6) { return r; }
  let axis = select(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 0.0), abs(r.z) > 0.9);
  let t = normalize(cross(axis, r));
  let b = cross(r, t);
  let rad = c * sqrt(u.x);
  let phi = 6.2831853 * u.y;
  return normalize(r + t * (rad * cos(phi)) + b * (rad * sin(phi)));
}

/// 太阳方向的可见性（0/1）× 沿途介质的乘性透过（穿过玻璃时被染色）。
/// 与 `gi/ray.wesl::sun_transmittance` **同一套写法**（同一条 `world_trace_medium`、同一个
/// 「撞到不透明体素即判被挡」的判据、同一个 `tint` 取平均的口径），唯一差别是：
/// t 上限来自 uniform（菜单「阴影距离」）而不是 8192 —— 远处的遮挡对这一段雾本来就没多少能量，
/// 把阴影距离砍短是这个效果**唯一**的成本闸门（实测档位 64/128/256/512 voxel）。
/// 不需要自命中豁免：采样点恒在空气里（雾只积到主命中之前）。
fn fog_sun_vis(p: vec3<f32>, sun_dir: vec3<f32>, max_dist: f32) -> f32 {
  var tint = vec3<f32>(1.0);
  let h = world_trace_medium(make_grid(0u), p, sun_dir, max(1.0, max_dist), false, &tint);
  let through = (tint.r + tint.g + tint.b) / 3.0;
  if (!h.hit) { return through; }
  return 0.0;
}

/// 写雾的导引（word 布局 = `gi/consts.wesl` 的 `GI_DEN_G_*`，与 GI 的导引**逐字同构**）。
/// 为什么复用 GI 的布局：空间段直接跑现成的 `gi_den_atrous*` 入口（不复制那份几百行的滤波），
/// 它按这套布局读键 / 法线 / 上帧像素；雾只需要保证写进去的内容满足它的判据。
/// `n` = **可见面法线**（不是着色法线）：雾是低频场，跨凸棱少滤一点无害，而算 `voxel_normal`
/// 那个梯度要额外探 4~6 个体素 —— 不值得。天空像素给 `+Y`（所有天空像素同值）。
fn fog_guide_store(coord: vec2<i32>, size: vec2<u32>, k0: u32, k1: u32, n: vec3<f32>,
                   prev_px: vec2<i32>, prev_ok: bool) {
  let b = (u32(coord.y) * size.x + u32(coord.x)) * GI_DEN_GUIDE_WORDS;
  fog_guide_w[b + GI_DEN_G_KEY0] = k0;
  fog_guide_w[b + GI_DEN_G_KEY1] = k1;
  fog_guide_w[b + GI_DEN_G_NRM] = gi_pack_dir(n);
  var code = 0xFFFFFFFFu;
  if (prev_ok) { code = u32(prev_px.x) | (u32(prev_px.y) << 16u); }
  fog_guide_w[b + GI_DEN_G_PREV] = code;
}

/// **雾的唯一生产 pass**（`main.wesl::fog_main` 的薄入口调它；入口必须留在主模块里 ——
/// WESL 只把「从根模块可达」的东西写进产物，子模块里没有任何调用点的 `@compute` 会被剪掉）。
/// 网格 = 渲染分辨率 ÷ `FogSettings.div`。
/// ① 主射线（= 本像素的雾需要的那点 G-buffer：**只为了 t_end**）+ beam 起点加速（与 `gi_main` 同口径）；
/// ② 写导引（面键 + 面法线 + 上帧重投影像素）；
/// ③ S 条自由飞行采样 × 每条一条阴影射线 ⇒ 单次散射；环境项闭式解。
/// 输出：`fog_out` = (线性 HDR 散射辐亮度, 1.0)。a 恒 1（**每个像素都有雾**，没有「无效像素」这回事），
/// 与 GI 的 premultiplied-valid 不同 —— 所以 `dda_main` 那侧不做 valid 归一化。
fn fog_pass_scatter(gid: vec3<u32>) {
  let size = textureDimensions(fog_out);
  if (gid.x >= size.x || gid.y >= size.y) { return; }
  let coord = vec2<i32>(i32(gid.x), i32(gid.y));
  // 本 pass 像素中心 → NDC（与 `gi_main` / `dda_main` 的反投影逐字同口径，y 已翻）。
  let px = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5))
         / vec2<f32>(f32(size.x), f32(size.y));
  let ndc = vec2<f32>(px.x * 2.0 - 1.0, 1.0 - px.y * 2.0);
  let near_world_h = view_u.inv_view_proj * vec4<f32>(ndc.x, ndc.y, 0.0, 1.0);
  let far_world_h = view_u.inv_view_proj * vec4<f32>(ndc.x, ndc.y, 1.0, 1.0);
  let near_world = near_world_h.xyz / near_world_h.w;
  let far_world = far_world_h.xyz / far_world_h.w;
  let diff_world = far_world - near_world;
  let frustum_length = length(diff_world);
  let dir_voxel = normalize(diff_world);
  let origin_voxel = view_u.cam_pos_voxel.xyz;

  // beam 预 pass 的保守起点（`view_u.lod.z > 0.5` = 关闭 beam ⇒ t_min = 0）。
  // 与 `main.wesl::gi_main` 同一套下标换算：beam 纹素 = 本 pass 像素 / (BEAM_DIV / fog_div)。
  let fog_div = max(u32(fog_u.misc.x), 1u);
  var t_min = 0.0;
  if (view_u.lod.z < 0.5) {
    let beam_size = textureDimensions(beam_depth);
    let bx = gid.x / max(BEAM_DIV / fog_div, 1u);
    let by = gid.y / max(BEAM_DIV / fog_div, 1u);
    t_min = 1e+30;
    for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
      for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
        let sx = clamp(i32(bx) + dx, 0, i32(beam_size.x) - 1);
        let sy = clamp(i32(by) + dy, 0, i32(beam_size.y) - 1);
        t_min = min(t_min, textureLoad(beam_depth, vec2<i32>(sx, sy)).x);
      }
    }
    t_min = max(t_min - BEAM_BACKOFF, 0.0);
  }
  let h = world_raycast(make_grid(0u), origin_voxel, dir_voxel, frustum_length,
                        world_cfg_full(t_min, 3u));

  // ---- ① 导引（时域重投影 + 空间段的平面判据都吃它）----
  // 重投影锚点：命中 ⇒ **真实命中点**（不是体素中心 —— 静止相机下它正好映回本像素，与 GI 同一条理由）；
  // 天空 ⇒ **沿视线到远平面的点**。为什么天空不能用其它锚点：`WorldHit.t` 在 miss 时是 0
  // （`point = 相机位置`），投出去 w ≈ 0 ⇒ `prev_ok` 恒假 ⇒ 天空那半边**永远拿不到历史**，
  // 而那正是最想看的光柱（打在天空上的那部分）。
  let prev_anchor = select(origin_voxel + dir_voxel * frustum_length, h.point, h.hit);
  let prev_h = fog_u.prev_view_proj * vec4<f32>(prev_anchor, 1.0);
  var prev_ok = false;
  var prev_px = vec2<i32>(0);
  if (prev_h.w > 1e-6) {
    let ndc = prev_h.xyz / prev_h.w;
    let p = (ndc.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5)) * vec2<f32>(f32(size.x), f32(size.y));
    let ip = vec2<i32>(floor(p));
    if (ip.x >= 0 && ip.y >= 0 && ip.x < i32(size.x) && ip.y < i32(size.y)) {
      prev_px = ip;
      prev_ok = true;
    }
  }
  if (h.hit) {
    let gg = make_grid(u32(h.obj_id) + 1u);
    fog_guide_store(coord, size, gi_key0(h.voxel), gi_key1(h.voxel, h.face_id, h.obj_id),
                    face_normal_world(gg, h.face_id), prev_px, prev_ok);
  } else {
    fog_guide_store(coord, size, FOG_SKY_K0, FOG_SKY_K1, vec3<f32>(0.0, 1.0, 0.0), prev_px, prev_ok);
  }

  // ---- ② 单次散射 ----
  let sigma_t = max(fog_u.params.x, 0.0);
  if (sigma_t <= 0.0) {
    textureStore(fog_out, coord, vec4<f32>(0.0));
    return;
  }
  // 积分区间 = [0, t_end]：命中 ⇒ 到命中点为止（雾只存在于表面之前）；miss ⇒ 到远平面。
  // **再按「雾距」截断**（`misc2.y`）——这一条是"不要变成起雾"的关键（见 `FogUniform.misc2` 的说明）：
  // 不截断时，天空那条射线的透射率会被积成 `e^{-σt·∞} → 0`，天空与远景被雾色 100% 替换。
  // `h.t` 在 miss 时是 0（`WorldHit` 的约定）⇒ 用 `select` 取远平面，与 prev 锚点同一套写法。
  let t_end = min(select(frustum_length, h.t, h.hit), max(fog_u.misc2.y, 1.0));
  // ---- 介质沿视线的光学深度（含**地面雾**）----
  // `a = h·dir.y`：往上看 ⇒ ρ 沿视线指数变薄（天空迅速变干净）、往下看 ⇒ 变浓（地面雾）。
  // `h = 0` 时 `a = 0`，两条公式都退化成均匀介质的 `σt·t_end` ⇒ 老口径逐位不变。
  let a = max(fog_u.params.y, 0.0) * dir_voxel.y;
  let tau_max = fog_optical_depth(sigma_t, a, t_end);
  let k = 1.0 - exp(-tau_max);

  let sun_dir = light_u.lights[0].kind_pos_dir.yzw;
  let sun_c = light_u.lights[0].color_intensity.xyz * light_u.lights[0].color_intensity.w;
  let sky = light_u.sky_color.xyz;
  // 相位：`cos_t = dot(ω_o, ω_i)` —— `ω_o` = 指向观察者 = **−视线方向**，`ω_i` = 指向太阳 = `sun_dir`
  // ⇒ `cos_t = -dot(dir_voxel, sun_dir)`（正对太阳时 = −1 = 前向峰值，见 `fog_phase` 的自检）。
  let phase = fog_phase(clamp(fog_u.params.z, -0.95, 0.95), -dot(dir_voxel, sun_dir));
  // 环境（天光）散射：闭式解已含 (σ_s/σ_t)·k，见文件头。
  let amb = sky * max(fog_u.params.w, 0.0);

  // 本像素本帧的抖动种子（帧号混入 ⇒ 每帧都是新的独立样本，时域累积才能收敛出光柱）。
  let seed = gi_mix(
    u32(coord.x) * 0x9E3779B1u
    + gi_mix(u32(coord.y) * 0x85EBCA77u
    + gi_mix(fog_u.seq.x * 0x27D4EB2Fu + 0x165667B1u)));
  let r_cone = gi_rand2(seed);
  let r_t = gi_rand2(seed + 0x9E3779B9u);
  // 太阳盘上的一个样本：整条射线（本像素本帧的**所有**采样点）共用同一个方向 ——
  // 逐点换方向只会把「一个方向上的可见性」打散成噪声，而共用 + 多帧累积 = 半影。
  let sdir = fog_cone_dir(sun_dir, r_cone, fog_u.misc.y);
  let s_n = clamp(u32(fog_u.misc.w), 1u, FOG_SAMPLE_MAX);
  let shadow_dist = max(fog_u.misc.z, 1.0);
  var vis_sum = 0.0;
  for (var i = 0u; i < s_n; i = i + 1u) {
    // 分层 + 抖动：第 i 层的 u ∈ [i, i+1)/S（S = 1 时退化成单个随机样本）。
    // 光学深度 → 距离：地面雾让这一步不是简单的 τ/σt（见 `fog_tau_to_t`）。
    let tau = fog_free_flight((f32(i) + r_t.y) / f32(s_n), tau_max);
    let t = fog_tau_to_t(tau, sigma_t, a);
    vis_sum = vis_sum + fog_sun_vis(origin_voxel + dir_voxel * t, sdir, shadow_dist);
  }
  // src = 单次散射的点辐射源（太阳项按采样均值估计 + 环境项）；再乘 (σ_s/σ_t)·k 得到辐亮度。
  // 太阳项上的**光柱增益**（`misc2.x`）不是给观感凑数：本实现只有**单次**散射 + 各向同性环境近似，
  // 而真实的光柱里多重散射占的份额可观（把同一份阳光反复散射回视线）⇒ 丢掉的那部分能量在这里补。
  let src = sun_c * phase * (vis_sum / f32(s_n)) * max(fog_u.misc2.x, 0.0) + amb;
  textureStore(fog_out, coord, vec4<f32>(src * FOG_ALBEDO * k, 1.0));
}

// ---- 雾的降噪：时域累积（本文件）+ 迭代 atrous（**复用** `gi/denoise.wesl::gi_den_atrous`）----
// 复用空间段的前提是三件事逐字对齐，缺一不可（改任一处都要同时看这三条）：
//   ① 导引 word 布局 = `GI_DEN_G_*`（`fog_guide_store` 写的就是这套，见上）；
//   ② φ 的语义 = 「亮度 range 权重尺度」，写在 `fog_phi_w`（atrous 从它的 group(0) binding 17 读）；
//   ③ 历史的 word 布局 = `FOG_DEN_H_*`（只有本文件与 Rust 的 buffer 尺寸知道，不跨模块）。
/// 历史缓冲每像素字数：累积色 rgb（3）+ 历史长度 M（1）+ 面键（2）。
/// 与 GI 的 9 字不同（它多存两个矩与着色法线，雾的历史权重不吃矩、也不比法线 —— 见文件头）。
const FOG_DEN_HIST_WORDS: u32 = 6u;
const FOG_DEN_H_COL: u32 = 0u;
const FOG_DEN_H_M: u32 = 3u;
const FOG_DEN_H_KEY0: u32 = 4u;
const FOG_DEN_H_KEY1: u32 = 5u;
/// 时域记忆窗上限（帧）：`w = (M−1)/M` ⇒ 稳态残余噪声 ≈ `1/√M`。
/// 取 128（≈ 2.1s@60fps）：雾**每帧只有 S 条采样**，而它的主要噪声是太阳可见性的 0/1 抖动
/// （对比度 ≈ 整个雾的亮度）⇒ 收敛比 GI（每帧 4~8 条候选）慢得多，窗口必须比 GI 的 96 更长。
/// 代价是响应慢（太阳缓慢移动时会有一点点滞后，由历史钳制兜住）；太阳**突变**（菜单改方向）
/// 由 Rust 侧整块清历史处理，不走这个窗口。
const FOG_DEN_M_MAX: u32 = 128u;
/// φ = `FOG_DEN_PHI_K` × max(帧间差 σ, 相对下限 × 参考亮度)：`exp(-|Δluma|/φ)` 是 atrous 跨 tap 的
/// 亮度门。K = 2 是**有意放大**的（对光柱边缘的抖动很宽容：那里相邻像素的差可以接近整个对比度），
/// 下限 8% 保证「σ 估成 0 的干净区域」也能跨体素平滑（否则大半径那几轮空转）。
const FOG_DEN_PHI_K: f32 = 2.0;
const FOG_DEN_PHI_FLOOR_REL: f32 = 0.08;

/// 雾的时域累积（入口 = `main.wesl::fog_den_temporal`；同上的「入口在主模块」理由）。
fn fog_pass_temporal(gid: vec3<u32>) {
  let size = textureDimensions(fog_tmp_out);
  if (gid.x >= size.x || gid.y >= size.y) { return; }
  let coord = vec2<i32>(i32(gid.x), i32(gid.y));
  let idx = u32(coord.y) * size.x + u32(coord.x);
  let gb = idx * GI_DEN_GUIDE_WORDS;
  let hb = idx * FOG_DEN_HIST_WORDS;
  let k0 = fog_guide_t[gb + GI_DEN_G_KEY0];
  let k1 = fog_guide_t[gb + GI_DEN_G_KEY1];
  let c_cur = textureLoad(fog_raw_in, coord, 0).rgb;

  // ---- ① 3×3 **同平面**邻域的 min/max + 参考亮度 ----
  // 判据用 `gi_den_same_plane`（与 GI 降噪、雾的上采样**同一份实现**：键 + 面号 + 物体，
  // 逐位比较 ⇒ 薄板另一侧 / 隔一格的平行面精确拒掉）。雾不需要法线相似那一条（低频场）。
  var mn = c_cur;
  var mx = c_cur;
  var lsum = dot(c_cur, GI_LUMA);
  var cnt = 1.0;
  for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
    for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
      if (dx == 0 && dy == 0) { continue; }
      let tx = coord.x + dx;
      let ty = coord.y + dy;
      if (tx < 0 || ty < 0 || tx >= i32(size.x) || ty >= i32(size.y)) { continue; }
      let tb = (u32(ty) * size.x + u32(tx)) * GI_DEN_GUIDE_WORDS;
      if (!gi_den_same_plane(fog_guide_t[tb + GI_DEN_G_KEY0], fog_guide_t[tb + GI_DEN_G_KEY1],
                             k0, k1)) { continue; }
      let c = textureLoad(fog_raw_in, vec2<i32>(tx, ty), 0).rgb;
      mn = min(mn, c);
      mx = max(mx, c);
      lsum = lsum + dot(c, GI_LUMA);
      cnt = cnt + 1.0;
    }
  }
  let l_ref = lsum / cnt;

  // ---- ② 历史（重投影 + 判据）+ ③ 钳制 ----
  // 判据只有两条：**同一个平面** + `M > 0`。不通过 ⇒ `M = 1`、色 = 本帧原始值（**本帧重建**：
  // 不拖影、不糊，也不回退到 0 / 旧值 ⇒ 第 1 帧就是稳态，没有 ramp）。
  var color = c_cur;
  var m = 1.0;
  var sigma = 0.0;
  let code = fog_guide_t[gb + GI_DEN_G_PREV];
  if (code != 0xFFFFFFFFu) {
    let ppx = i32(code & 0xFFFFu);
    let ppy = i32((code >> 16u) & 0xFFFFu);
    if (ppx < i32(size.x) && ppy < i32(size.y)) {
      let pidx = u32(ppy) * size.x + u32(ppx);
      let hpb = pidx * FOG_DEN_HIST_WORDS;
      let h_m = bitcast<f32>(fog_hist_in[hpb + FOG_DEN_H_M]);
      if (h_m > 0.0
          && gi_den_same_plane(fog_hist_in[hpb + FOG_DEN_H_KEY0],
                               fog_hist_in[hpb + FOG_DEN_H_KEY1], k0, k1)) {
        m = min(h_m + 1.0, f32(FOG_DEN_M_MAX));
        let hcol = vec3<f32>(
          bitcast<f32>(fog_hist_in[hpb + FOG_DEN_H_COL]),
          bitcast<f32>(fog_hist_in[hpb + FOG_DEN_H_COL + 1u]),
          bitcast<f32>(fog_hist_in[hpb + FOG_DEN_H_COL + 2u]));
        // σ = **帧间差**（钳制之前的原始历史）—— 雾真正的噪声量。
        // ⚠️ 这里与 GI 那份的 σ 口径**刻意不同**：GI 用「同一个面内 3×3 邻域的亮度标准差」，
        // 那个假设（面内差异只可能是噪声）对 GI 成立、对雾**不成立**（雾沿视线连续变化，
        // 面内相邻像素的差异是真实信号）—— 照抄会让 `σ_α/σ` 塌到下限、时域累积等于关掉。
        sigma = dot(abs(c_cur - hcol), GI_LUMA);
        // 历史钳到本帧同平面邻域的 min/max：太阳转动 / 去遮挡时旧值被拉回本帧的范围（anti-ghosting）。
        let hc = clamp(hcol, mn, mx);
        let w = (m - 1.0) / m;
        color = mix(c_cur, hc, w);
      }
    }
  }
  // φ：空间段（复用的 `gi_den_atrous`）从 group(0) binding 17 读这个 word。
  fog_phi_w[idx] = FOG_DEN_PHI_K * max(max(sigma, FOG_DEN_PHI_FLOOR_REL * l_ref), 1e-5);

  // ---- ④ 写历史 + 输出 ----
  fog_hist_out[hb + FOG_DEN_H_COL] = bitcast<u32>(color.x);
  fog_hist_out[hb + FOG_DEN_H_COL + 1u] = bitcast<u32>(color.y);
  fog_hist_out[hb + FOG_DEN_H_COL + 2u] = bitcast<u32>(color.z);
  fog_hist_out[hb + FOG_DEN_H_M] = bitcast<u32>(m);
  fog_hist_out[hb + FOG_DEN_H_KEY0] = k0;
  fog_hist_out[hb + FOG_DEN_H_KEY1] = k1;
  textureStore(fog_tmp_out, coord, vec4<f32>(color, 1.0));
}

/// **雾的几何感知上采样**（`dda_main` 取雾的唯一入口）—— 做法与 `main.wesl::gi_upsample_joint`
/// 相同（取最近 2×2 texel，只接受与本像素**同一个平面**的，按双线性权重平均；一个都没接受 ⇒
/// 退回最近的 texel），差别只有一条：**不比法线**。
/// GI 那条法线相似度是为「一个体素面内必须同值」（逐体素风味）服务的；雾是低频场（沿视线连续变化），
/// 同一平面上相邻体素接到的是同一片雾 ⇒ 只要平面归属（精确位域）对就不会渗色。
/// `k0/k1` = 本像素主命中的面键（天空像素传 `FOG_SKY_K0/FOG_SKY_K1`）。
fn fog_upsample_joint(uv: vec2<f32>, gsize: vec2<u32>, k0: u32, k1: u32) -> vec3<f32> {
  let g = uv * vec2<f32>(f32(gsize.x), f32(gsize.y));
  let base = floor(g - vec2<f32>(0.5));
  let f = g - vec2<f32>(0.5) - base;
  var acc = vec3<f32>(0.0);
  var wsum = 0.0;
  for (var j: i32 = 0; j < 2; j = j + 1) {
    for (var i: i32 = 0; i < 2; i = i + 1) {
      let c = vec2<i32>(base) + vec2<i32>(i, j);
      if (c.x < 0 || c.y < 0 || c.x >= i32(gsize.x) || c.y >= i32(gsize.y)) { continue; }
      let bw = select(1.0 - f.x, f.x, i == 1) * select(1.0 - f.y, f.y, j == 1);
      if (bw <= 1e-6) { continue; }
      let tb = (u32(c.y) * gsize.x + u32(c.x)) * GI_DEN_GUIDE_WORDS;
      if (!gi_den_same_plane(fog_guide_r[tb + GI_DEN_G_KEY0], fog_guide_r[tb + GI_DEN_G_KEY1],
                             k0, k1)) { continue; }
      acc = acc + textureLoad(fog_tex, c, 0).rgb * bw;
      wsum = wsum + bw;
    }
  }
  if (wsum > 0.0) { return acc / wsum; }
  // 一个同平面的 texel 都没有（轮廓 / 凸角）：退回最近的 texel。雾**每个 texel 都有效**（a 恒 1），
  // 所以这里恒有值，不需要 fallback 实参（与 GI 那版按 `a` 判有效性不同）。
  let nc = clamp(base + vec2<f32>(select(0.0, 1.0, f.x >= 0.5), select(0.0, 1.0, f.y >= 0.5)),
                 vec2<f32>(0.0), vec2<f32>(f32(gsize.x) - 1.0, f32(gsize.y) - 1.0));
  return textureLoad(fog_tex, vec2<i32>(nc), 0).rgb;
}
