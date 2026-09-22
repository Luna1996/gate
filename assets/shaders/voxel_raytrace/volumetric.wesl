import package::bindings::{BEAM_BACKOFF, BEAM_DIV, beam_depth, fog_a_r_blur, fog_a_w, fog_a_w_blur, fog_b_r_blur, fog_b_w_blur, fog_u, fog_u_blur, fog_u_read, light_u, view_u};
import package::common::{sky_rgb};
import package::trace::{make_grid};
import package::world::{world_cfg_full, world_raycast};

// ============================================================================
// 光柱（godray）—— **屏幕空间的径向模糊**：掩码 → 两趟模糊 → 主 pass 相加
//
// 为什么不做体积散射：体积散射要在每条视线上估「空气里有多少段被阳光照到」，那是 **0/1 的
// 可见性**，单帧单样本的噪声等于信号本身 ⇒ 要么每条视线打很多条阴影射线（贵），
// 要么靠长时域累积（拖影）。本档要的是「看得见的光柱」，不要求它在物理上完全正确，
// 于是走业界最经典的那条路（GPU Gems 3 «Volumetric Light Scattering as a Post-Process»
// 的形态，只是在光栅化引擎里换成「真射线求交」）：
//
//   ① **掩码**（`godray_pass_mask`，网格 = 渲染分辨率 ÷ `div`）：每 texel 一条**主射线** ⇒
//      「这个方向看得见天空吗」= 阳光有没有被几何挡住；再乘一个**太阳方向权重**
//      `pow(dot(视线, 太阳方向), 集中度)` —— 只有朝着太阳那片天空有值（其余方向几乎为零）；
//      再乘太阳色与强度 ⇒ 掩码里装的已经是"光"。
//   ② **径向模糊**（`godray_pass_blur`，两趟）：沿「本像素 → 太阳的屏幕位置」这条线段
//      取样累加（每步衰减）⇒ 光从太阳的位置沿屏幕铺开，被几何的**屏幕轮廓**挡住 ⇒
//      一束一束的光柱。两趟分别走 `GODRAY_STEP_A` / `GODRAY_STEP_B` 比例，叠起来正好到太阳。
//   ③ `dda_main` 把它**加到**颜色上（曝光之前、线性 HDR）。
//
// 两张中间靶的**固定角色**（Rust 侧每帧绑同一组视图，不需要 ping-pong 换绑）：
//   `fog_a`：掩码 pass 写 → 模糊 A 读 → **模糊 B 写**（最终结果，`dda_main` 采样它）；
//   `fog_b`：模糊 A 写 → 模糊 B 读。
// ⇒ 三个入口共用**同一个 bind group**（group(4) 的 1..5），只是各自引用的 var 不同。
//
// 观感与代价（**有意写明的取舍**）：
//   · ✅ **噪声 = 0、滞后 = 0**：三步都是确定性取值，不进时域、不含任何随机采样；
//   · ✅ 代价恒定且很小：掩码 1 条主射线/texel（1/4 分辨率）+ 两趟 `GODRAY_TAPS` 次取值；
//   · ❌ **纯屏幕空间**：遮挡只按屏幕轮廓算（不是真实的三维体积遮挡）——
//     树冠/窗口/建筑边缘投下来的光柱对，但"光柱被远处物体挡住"这类体积关系算不出来；
//   · ❌ 太阳转到画面**外/背后**时没有光柱（掩码里没有太阳那片天空）；
//     `朝向因子` 让它在太阳接近画面边缘时**平滑淡出**，不会突然消失。
// ============================================================================

/// 天体盘的亮度倍数（乘 `sun_c`）：盘的**中心**要略高于白顶（8 位输出里发白）才像"光源本体"，
/// 但**不能**像早期那样靠 `40×` 过曝 —— 那样盘的边缘会是一条生硬的截断线（整盘都是白的，
/// 只在外沿几个像素里从白掉到黑）。现在取一个"刚好压过白顶"的量级，让外侧约三成半径是**可见的渐变**，
/// 再交给下面的光晕接续（见 `sky_primary`）。月亮靠 `lights[0].shape.x` 再乘一个更小的倍数。
const SUN_DISK_GAIN: f32 = 3.0;
/// **盘的软边缘形状**：先算归一化半径的平方 `t² = (θ/角径)²`，盘面 = `(1 - t²)^LIMB_POW`。
/// 这是**球面**的轮廓（`LIMB_POW = 0.5` 时正是 `√(1-t²)`，即球的正视投影轮廓 + 天然的边缘变暗）：
/// 中心平坦、向外渐暗、在 `t = 1`（盘的边缘）处恰好到 0 ⇒ 与光晕无缝接上，没有硬边也没有台阶。
const LIMB_POW: f32 = 0.5;
/// **光晕的尺度**（单位 = 盘的角径）：光晕在 `t = HALO_T`（= 角径的 10 倍）处衰减到 0。
/// 为什么要**跟着角径缩放**：晕的物理来源是大气散射，肉眼看到的那圈亮边就贴着盘子
/// ⇒ 角径越大，晕也该越铺开。固定角尺度（早期做法）在小盘上会变成"针尖 + 一大团雾"，
/// 大盘上又会变成"白饼外面贴一圈" —— 这正是"过渡生硬"的来源。
const HALO_T: f32 = 10.0;
/// 光晕剖面的幂（`(1 - t²/HALO_T²)^HALO_POW`）：2 时中心平缓、外沿收敛，接近真实的大气晕。
const HALO_POW: f32 = 2.0;
/// 天体盘角径的**上限**（rad）：uniform 里的值按它硬夹（防止滑杆滑出天体的量级）。
const SUN_CONE_MAX: f32 = 0.06;
/// 网格分辨率除数（**权威值**，Rust 侧 `volumetric::FOG_DIV` 必须与它一致）：
/// 光柱网格边长 = 渲染分辨率 ÷ 本值。
/// 固定 4（1/4 分辨率）而不是做成档位：掩码之后还有一趟横跨最长整屏的径向模糊，
/// **最终画面的柔和程度由模糊长度决定、不由掩码分辨率决定** ⇒ 1/2 与 1/1 在观感上几乎无差别，
/// 而代价按网格像素数涨（1/1 的主射线数量是 1/4 的 16 倍）。
/// 唯一代价：细于 4 像素的遮挡物（细栏杆、细枝）可能被掩码漏掉、让光从那里漏进来 ——
/// 对本效果的观感无影响（光柱本身就是软的）。
const FOG_DIV: u32 = 4u;
/// 每趟模糊的 tap 数（两趟共 2×本值次取值）。12 在 1/4 分辨率下已经足够平滑
/// （模糊的"台阶"尺度 = 线段长度 ÷ 本值，而线段最长跨整屏）。
const GODRAY_TAPS: u32 = 12u;
/// 两趟模糊各自走"到太阳"的多少比例：两趟叠起来 ≈ 1（正好到太阳的位置）。
/// 分两趟是为了在同样的 tap 数下把模糊足迹拉长一倍（第一步平滑近处、第二步平滑远处）。
const GODRAY_STEP_A: f32 = 0.35;
const GODRAY_STEP_B: f32 = 0.65;

/// **主射线看到的天空**（`dda_main` 的 miss 背景）：常量天光 + **日月外观**（盘 + 光晕）。
/// 「天体」= 那个方向光：白天是太阳、夜间是**反日点的月亮**（日月共用一个槽位，见 `sky.rs`）——
/// 盘与晕都在**归一化半径** `t` 上定义（`t = 角距 ÷ 角径`，用 `t² = (1-cosθ)/(1-cos 角径)` 算，
/// 省一次 `acos`）：
///   · 盘 = `(1 - t²)^LIMB_POW` × `SUN_DISK_GAIN` × `shape.x`（`shape.x` = 盘亮度倍数：太阳 `1`、
///     月亮 `0.6`）—— 球面轮廓，中心平、边缘渐暗、在 `t = 1` 处归零；
///   · 晕 = `(1 - t²/HALO_T²)^HALO_POW` × `光晕`（`misc2.w`）—— 同一根半径上的解析拖尾，
///     铺到 `HALO_T` 倍角径处 ⇒ 日月、大小盘都自洽，且与盘的边缘无缝相接（不再有"硬边 + 亮环"）。
/// **`misc2.x` = 日月外观开关**（面板「渲染/天空/日月/开启」）：关掉 ⇒ 盘与晕**一起**消失，
/// 那一片天区只剩基色（两者是同一个天体的外观，不是两件事）。
///
/// 与 `sky_rgb()` 的分工（硬约束：口径不分叉）：`sky_rgb()` 是**间接光的天空项**
/// （GI 的 miss、玻璃与镜面反射的逃逸），太阳作为光源已经由各处 NEE / `sun_c` 负责
/// ⇒ 那些地方**不能再见到天体**（会把同一份能量算两遍）。本函数只服务「看得见的那颗天体」。
/// 因为只在**主射线 miss** 时求值，天体被树/墙挡住时它自动消失（不需要额外的可见性判定）。
fn sky_primary(dir: vec3<f32>, sun_dir: vec3<f32>, sun_c: vec3<f32>) -> vec3<f32> {
  let base = sky_rgb();
  // 外观开关关掉 ⇒ 只剩基色。**用真分支**（不是乘 0）：省掉下面两次 `pow`；条件来自 uniform
  // ⇒ 整个 dispatch 一致，分支开销可忽略（与 `dda_main` 的「直光阴影」同一个取舍）。
  if (fog_u_read.misc2.x <= 0.5) { return base; }
  let ang = clamp(fog_u_read.misc2.z, 1e-4, SUN_CONE_MAX);
  let cos_t = dot(dir, sun_dir);
  // 归一化半径的平方：(θ/角径)² ≈ (1-cosθ)/(1-cos 角径)（小角下二阶等价）。
  let t2 = (1.0 - cos_t) / max(1.0 - cos(ang), 1e-12);
  let limb = clamp(1.0 - t2, 0.0, 1.0);
  let disk = pow(limb, LIMB_POW);
  let halo = pow(max(1.0 - t2 / (HALO_T * HALO_T), 0.0), HALO_POW);
  let disk_gain = max(light_u.lights[0].shape.x, 0.0);
  return base
       + sun_c * (disk * SUN_DISK_GAIN * disk_gain
                  + halo * max(fog_u_read.misc2.w, 0.0));
}

/// **掩码 pass**（入口 = `main.wesl::godray_main`）：网格 = 渲染分辨率 ÷ `FOG_DIV`。
/// 每个 texel 一条**主射线**（与 `dda_main` 同口径的反投影 + beam 起点加速）：
///   天空（未命中）⇒ 这个方向有阳光可以散射进眼睛；命中几何 ⇒ 被挡住。
/// 再乘太阳方向权重 × 太阳色 × 强度 × 朝向因子 ⇒ 写进 `fog_a`（rgba16f，rgb = 光、a = 1）。
fn godray_pass_mask(gid: vec3<u32>) {
  let size = textureDimensions(fog_a_w);
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
  // 与 `main.wesl::gi_main` 同一套下标换算：beam 纹素 = 本 pass 像素 / (BEAM_DIV / FOG_DIV)。
  // 两个除数默认相等（都是 4）⇒ 这条除法在默认值下是 1:1；写成除法是为了万一两边改了不同档。
  var t_min = 0.0;
  if (view_u.lod.z < 0.5) {
    let beam_size = textureDimensions(beam_depth);
    let bx = gid.x / max(BEAM_DIV / FOG_DIV, 1u);
    let by = gid.y / max(BEAM_DIV / FOG_DIV, 1u);
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

  let sun_dir = light_u.lights[0].kind_pos_dir.yzw;
  let sun_c = light_u.lights[0].color_intensity.xyz * light_u.lights[0].color_intensity.w;
  // 天空 = 这条视线上没有几何 ⇒ 阳光可以散射进来（这是"光柱"的唯一来源）。
  let sky = select(0.0, 1.0, !h.hit);
  // 太阳方向权重：只有朝着太阳那一片天空有值。
  // `集中度`（`misc.y`）= 指数：越大越集中在太阳附近（1 ⇒ 整个天空都发光 = 一层方向性眩光；
  // 8~16 ⇒ 只在太阳周围二三十度内 ⇒ 就是"太阳射出来的光柱"）。
  let w = pow(clamp(dot(dir_voxel, sun_dir), 0.0, 1.0), max(fog_u.misc.y, 1.0));
  // `misc.z` = 强度、`misc.w` = 朝向因子（太阳在画面背后 ⇒ 0，见 Rust `prepare_fog`）。
  let lit = sun_c * (sky * w * max(fog_u.misc.z, 0.0) * clamp(fog_u.misc.w, 0.0, 1.0));
  textureStore(fog_a_w, coord, vec4<f32>(lit, 1.0));
}

/// 一张 `texture_2d<f32>` 的**双线性**取值：与 `main.wesl::fog_upsample_joint` 同一套手写
/// 2×2 权重（模糊那两个 pass 的 layout 里没有采样器可绑）。
fn godray_sample_bilinear(tex: texture_2d<f32>, uv: vec2<f32>) -> vec3<f32> {
  let size = textureDimensions(tex);
  let g = uv * vec2<f32>(f32(size.x), f32(size.y)) - vec2<f32>(0.5);
  let base = floor(g);
  let f = g - base;
  var acc = vec3<f32>(0.0);
  for (var j: i32 = 0; j < 2; j = j + 1) {
    for (var i: i32 = 0; i < 2; i = i + 1) {
      let c = clamp(vec2<i32>(base) + vec2<i32>(i, j), vec2<i32>(0),
                    vec2<i32>(i32(size.x) - 1, i32(size.y) - 1));
      let bw = select(1.0 - f.x, f.x, i == 1) * select(1.0 - f.y, f.y, j == 1);
      acc = acc + textureLoad(tex, c, 0).rgb * bw;
    }
  }
  return acc;
}

/// **径向模糊的共享累加**：从 `uv` 沿直线朝**太阳的屏幕位置**（`misc3.xy`，可能落在屏幕外）
/// 取样 `GODRAY_TAPS` 次，每步权重乘 `衰减`（`params.x`）⇒ 光从太阳那侧铺过来、离太阳越远越淡。
/// 归一化用权重和（结果落在掩码的取值范围内，强度由掩码那一步给）。
/// `step` = 本趟走"到太阳"的多少比例：两趟叠起来把足迹拉到整条线段（见常量说明）。
/// `u` / `src` 都由调用方传：掩码与模糊两个 pass 的绑定在**不同的组号**上（见 `bindings.wesl`），
/// 而只读的纹理可以当函数参数（`textureStore` 的目标必须是模块级变量，所以两个入口各自写）。
fn godray_blur_accum(
  u: vec4<f32>,
  src: texture_2d<f32>,
  uv: vec2<f32>,
  step: f32,
) -> vec3<f32> {
  let to_sun = (u.xy - uv) * step;
  let decay = clamp(u.z, 0.05, 1.0);
  var acc = vec3<f32>(0.0);
  var wsum = 0.0;
  var w = 1.0;
  for (var i = 1u; i <= GODRAY_TAPS; i = i + 1u) {
    // 从本像素朝太阳的方向取样（i 越大越靠近太阳）：累加是确定性的（同序、同权重）。
    let p = uv + to_sun * (f32(i) / f32(GODRAY_TAPS));
    acc = acc + godray_sample_bilinear(src, p) * w;
    wsum = wsum + w;
    w = w * decay;
  }
  return acc / max(wsum, 1e-5);
}

/// 模糊 pass 的参数（`xyz` = 太阳屏幕坐标 + 衰减；`u` 那个 vec4 的打包就够用）。
/// 两个入口共用（贴在同一处，改一处即可）。
fn godray_blur_params() -> vec4<f32> {
  return vec4<f32>(fog_u_blur.misc3.x, fog_u_blur.misc3.y, fog_u_blur.params.x, 0.0);
}

/// **径向模糊 · 第一趟**（入口 = `main.wesl::godray_blur_a`）：读 A、写 B。
fn godray_pass_blur_a(gid: vec3<u32>) {
  let size = textureDimensions(fog_b_w_blur);
  if (gid.x >= size.x || gid.y >= size.y) { return; }
  let uv = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5))
         / vec2<f32>(f32(size.x), f32(size.y));
  let c = godray_blur_accum(godray_blur_params(), fog_a_r_blur, uv, GODRAY_STEP_A);
  textureStore(fog_b_w_blur, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(c, 1.0));
}

/// **径向模糊 · 第二趟**（入口 = `main.wesl::godray_blur_b`）：读 B、写 **A**（A 里是最终结果）。
fn godray_pass_blur_b(gid: vec3<u32>) {
  let size = textureDimensions(fog_a_w_blur);
  if (gid.x >= size.x || gid.y >= size.y) { return; }
  let uv = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5))
         / vec2<f32>(f32(size.x), f32(size.y));
  let c = godray_blur_accum(godray_blur_params(), fog_b_r_blur, uv, GODRAY_STEP_B);
  textureStore(fog_a_w_blur, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(c, 1.0));
}
