import package::bindings::{BASE_FLAG_IMPLICIT_NORMAL, BASE_FLAG_SHADOW, BASE_FLAG_VOL_TINT, BEAM_BACKOFF, BEAM_DIV, beam_depth, eye_adapt, eye_adapt_ro, face_slots, face_slots_flat, fog_tex, gi_dn_dst, gi_dn_tmp, gi_guide, gi_guide_out, gi_out, gi_out_flat, gi_tex, gi_u, light_samp, light_u, lod_diag, out_tex, out_tex_src, view_u};
import package::common::{FLAGS_TRANSMISSIVE, MaterialParams, SurfaceLight, brdf_reflected, dielectric_f0, emissive_radiance, f0_of, face_view_dir, fetch_material, fresnel_schlick, linear_to_srgb, material_ior, palette_flags, palette_rgb_linear, sky_rgb, srgb_to_linear};
import package::gi::common::{FACE_WORDS, FACE_W_COL, FACE_W_FLAG, FACE_W_GI, FACE_W_GI_N, FACE_W_K0, FACE_W_K1, FACE_W_PAL, FACE_W_RAW, FACE_W_RAW_N, FACE_W_SUN, FACE_W_TEXEL, face_slot_color, face_slot_find, face_slot_find_flat, gi_key0, gi_key1, gi_key_ok, gi_unkey_voxel, gi_mix, gi_pack_dir, gi_rand2, gi_unpack_dir, GI_ACC_CLAMP, GI_ACC_MAX_N, GI_ACC_SCALE};
import package::gi::consts::{GI_AMBIENT_FLOOR, GI_DEN_G_KEY0, GI_DEN_G_KEY1, GI_DEN_G_NRM, GI_DEN_GUIDE_WORDS, GI_EMIT_GAIN, GI_LUMA, GI_PI, GI_SKY_AMBIENT, SHADOW_SURFACE_EPS};
import package::gi::denoise::{gi_den_atrous, gi_den_temporal, gi_same_surface};
import package::gi::ray::{sun_transmittance};
import package::gi::screen::{gi_ss_main};
import package::gi::wal::{gi_wal_pool, GI_WAL_MATURE_N};
import package::trace::{diag_sampled, make_grid};
import package::volumetric::{godray_pass_blur_a, godray_pass_blur_b, godray_pass_mask, godray_sample_bilinear, sky_primary};
import package::world::{WorldHit, camera_solid_pal, face_normal_world, surface_offset, vol_tint_color, voxel_center_world, voxel_normal, world_cfg_full, world_cfg_primary, world_miss, world_raycast, world_trace_medium_scene};

// RT 版（`ray_query` 开）才声明这条扩展。**必须放在入口模块、且在所有 import 之后**
// （WESL 的解析器不接受 `import` 出现在其它全局声明 / 指令之后）。
// 软件版里出现它会让 `create_shader_module` 在无该特性的设备上直接失败（不是运行期退化）。
@if(ray_query)
enable wgpu_ray_query;

// 全屏逐像素层次栈式 mask DDA（64-tree）：chunk 间 256³ A&W，chunk 内 4 层栈帧（256→64→16→4→1）。
// 常量/寻址/slot 打包须与 Rust `brickmap::dda::wgsl_consts`、`brickmap::raytrace`、`wire.rs::encode_slot` 一致。

// beam 预 pass：全分辨率 / BEAM_DIV 的低分辨率保守 trace，输出每像素最近命中 t。
// 只行进到 min(t_cap, 0.35/(BEAM_DIV·px_ang))；miss 存该上限（非 t_cap）。
@compute @workgroup_size(8, 8, 1)
fn beam_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let beam_size = textureDimensions(beam_depth);
  if (gid.x >= beam_size.x || gid.y >= beam_size.y) { return; }
  let px = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5))
         / vec2<f32>(f32(beam_size.x), f32(beam_size.y));
  let uv = vec2<f32>(px.x * 2.0 - 1.0, 1.0 - px.y * 2.0);
  let near_ndc = vec4<f32>(uv.x, uv.y, 0.0, 1.0);
  let far_ndc  = vec4<f32>(uv.x, uv.y, 1.0, 1.0);
  let near_world_h = view_u.inv_view_proj * near_ndc;
  let far_world_h  = view_u.inv_view_proj * far_ndc;
  let near_world = near_world_h.xyz / near_world_h.w;
  let far_world  = far_world_h.xyz  / far_world_h.w;
  let dir = normalize(far_world - near_world);
  let origin = view_u.cam_pos_voxel.xyz;
  let t_cap = length(far_world - near_world);
  let d_beam = 0.35 / (f32(BEAM_DIV) * max(view_u.lod.x, 1e-6));
  let t_cap_beam = min(t_cap, d_beam);

  let h = world_raycast(make_grid(0u), origin, dir, t_cap_beam, world_cfg_full(0.0, 3u));
  let t = select(t_cap_beam, h.t, h.hit);
  textureStore(beam_depth, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(t, 0.0, 0.0, 0.0));
}

/// 折射/反射后新射线起点沿新方向外推的距离（voxel），须大于浮点噪声且远小于 1 格。
const GLASS_EPS: f32 = 1e-3;
/// 一条主射线最多处理的玻璃界面数（入射/出射各算一次），不限介质段厚度。
const GLASS_MAX_BOUNCE: u32 = 8u;
/// Fresnel 反射率门槛：低于此值不追反射射线。
const GLASS_REFL_MIN: f32 = 0.02;
/// 玻璃反射命中点的**常量天光**系数（`sky · 本值`）：玻璃反射不采样 GI、也不追阴影。
/// MT4-1 起这条反射命中的**着色**已接进统一入口（`fetch_material` + `brdf_reflected`），
/// 本常量只剩"天光取多大"这一个含义（比不透明表面的 `GI_SKY_AMBIENT` 大得多 —— 玻璃反射看到的
/// 是"整片环境"，与旧式 `sky · GLASS_REFL_AMBIENT` 同源，观感不变）。
const GLASS_REFL_AMBIENT: f32 = 0.15;

// ============================================================================
// MT4 · 反光度（镜面反射）—— `docs/PLAN.md` §4 MT4-1 ~ MT4-4
//
// 本里程碑只加**一条**射线：主 pass 的不透明表面在"值得追"时追加一条**反射射线**，把反射点的
// **出射亮度**当作镜面项**替换**掉 `F0·(amb+gi)` 那个环境近似（`dda_main` 的反射段，MT4-3）。
// GI 链一字未动 —— 反射样本不进 ReSTIR，理由写在 `dda_main` 里那段注释（MT4-2）。
//
// ⚠️ **镜面是逐体素面的块状镜面**（视向量 / 种子 / 法线全按面取，见 MT8-6 与
// `common.wesl::face_view_dir`）：一个体素面 = 一个反射值。**单个镜面方块只有 1 个面 ⇒ 只有 1 个
// 颜色**；要看到"成像"必须涂一整片镜面（面数 = 镜像的分辨率）。
//
// ---- MT4-4（运行期档位）----
// "反射内容着色到哪一档"由 uniform `light_u.g.refl_tier` 给（菜单「渲染/反射」，0 = 关 ..= 3），
// 四档的定义与成本见 `gate-render/src/lighting.rs::ReflectionSettings`；档位只改**着色口径**，
// 不改上面那条逐面量化的粒度。
//
// ---- MT4-5（反射链：镜子里的镜子）----
// `light_u.g.refl_nest`（菜单「渲染/反射/嵌套层级」，0 / 1 / 2 / 4）= 最多再嵌几级。实现是
// `refl_env_chain` 里那个 `for (lvl = 0; lvl <= nest; ++lvl)` 循环，成本闸门与画风约束写在循环上方那段注释里。
// **`0` 就是本文件此前唯一的行为**（反射命中点只做完整着色、不再反射）。
//
// ---- MT8-3（反射缓存）已按实测**整体删除**（历史记录，别再捡回来）----
// MT8-3（`§4b`）曾把这条射线的粒度从"每像素"改成"每个可见体素面"：在 BG1 加一对乒乓 buffer
// （binding 10/11）+ 键/槽哈希 + 每帧旋转刷新，`refl_key_of`/`refl_slot_of`/`refl_seed` 都在本文件。
// **用户实测判定它是负优化**（"反射缓存和所有 GPU per voxel hashmap 都是负优化，删掉"）：
//   · 它换来的是"省掉一部分射线"，而 MT8-1/MT8-4 已经把逐像素射线本身的成本压下来了
//     （MT4 时代 3.37ms → 缓存版 1.42ms → **关缓存后 ≈1.5~1.6ms**，见下面 `PBR_REFLECTION_ENABLED`
//     的实测账）⇒ 换来的收益对不上它引入的复杂度（两个大 buffer、键/哈希、刷新滞后与显存）；
//   · 它依赖的前提（MT8-2 的"逐面恒定视向量"）本身已被实测推翻（白边 + 镜面消失）⇒ 前提没了。
// 删除后本段**回到 MT4 的形态**：每个触发像素各发一条射线（成本与 MT4 逐像素发线一致，这是当前接受的现状）。
// ⚠️ 这是**实测推翻**，不是"漏了/没写完"—— 不要再按旧注释把缓存加回来（`@group(1)` 保持 0..9）。
//
// ---- MT8-6（抖动种子：按屏幕像素 → 按体素面）----
// 缓存删除后种子一度是**按像素**取的，用户实测判定它带来两个症状（**同一个根因**）：
//   · 「表面反射不是逐体素着色」：同面内逐像素各采锥内一个**不同**的样本，而锥角
//     （`PBR_REFL_CONE_K × roughness` ≈ 5.6°，见下面那个常量）把反射方向"抹"开 ⇒ 反射出一片
//     **连续模糊**的树影 / 屋顶 / 天空，不像材质色那样一格一格；
//   · 「呈现颗粒感」：于是单个像素退化成一次"随机采样"，而 MT4 **有意不做时域累积** ⇒ 噪声永不收敛，
//     在光滑面上就是一层椒盐颗粒。
// ⇒ MT8-6 把抖动种子改成**按体素面**取（键 = grid 局部体素坐标 + 面号 + obj_id）：同一个体素面上的
// 所有像素拿到**同一个**抖动量 ⇒ **一个体素面 = 一条反射射线方向 = 一个反射值**（反射逐体素），
// 且同面内不再有逐像素随机（颗粒感消失）。
// ⚠️ **这不是缓存**：射线仍然**每个触发像素各发一条**（成本不变），唯一效果是"同一个面的抖动样本一致"。
// ============================================================================

/// **总开关（A/B 自检用）**：`1u` = 开（仓库默认 = 功能本体）、`0u` = 关。
/// 与 MT2-4 的 `PBR_DEBUG_ASSET` 同一个先例：**WESL 编译期常量**（改一行、重启即生效；WESL 是
/// 运行期读盘编译，不用 rebuild）。**运行期档位**是另一件事：`light_u.g.refl_tier`（菜单「渲染/反射」，
/// 见 MT4-4 段）只在总开关为 `1u` 时起作用。
/// `0u` 时 `dda_main` 里整段反射代码被常量折叠掉：
///   · 画面**逐位回到 MT3 状态**（不发射线、不加反射项）；
///   · 这同时就是 MT4-2 要的"关掉反射后 GI 画面与 MT3 一致"的**自检手段**（GI 侧从未读过反射量，
///     所以开关对 GI 的唯一影响只能来自主 pass 的着色，绝不可能来自 GI 自身）。
/// A/B 口径（MT4-4）：`cargo run --release -p gate-app --features profile`，读 `logs/latest.log`
/// 里逐 pass GPU 均值那行的 `gate_dda_trace=`（主 pass = 本文件 `dda_main`），改 `0u` 重启再跑一次。
/// **实测（RTX 3070 / 1280×720 / castle.vox / GI 1/2 分辨率 + 低档降噪；各跑 3 次取稳定段）**：
/// `0u` = **1.26ms**（29 个采样点：1.26~1.28）、`1u` = **3.35~3.38ms**（3 次：3.34~3.83）
/// ⇒ **+2.1ms（+167%）**（最差一次 +2.6ms / +203%）。
/// 其中"代码在、判据恒不成立"（临时把 `PBR_REFL_MIN_F0` 设 2.0）实测 **1.51ms** ⇒ 拆账：
/// +0.25ms 是代码体积/寄存器压力（±20%），**+1.85ms 是真的把反射射线发出去了**。
const PBR_REFLECTION_ENABLED: u32 = 1u;

/// 触发判据 ①：`F0` 的**亮度投影**低于本值不发线 —— 镜面反射率本身太低时，那条射线的视觉贡献
/// 还不如它的噪声与成本，不值得追。
/// 取 0.03：默认平凡材质是电介质（`specular = 1.0`）⇒ `F0 = ((1.5−1)/(1.5+1))² = 0.04` 恰在门槛之上
/// （**要保留这条**：否则"光滑度拉满 ⇒ 镜面"整个消失）；把 PBR 变体的 `specular` 槽级覆盖调到
/// ≤ 0.5（`F0 ≤ 0.02 < 0.03`）即整类不追 —— 这就是"用 F0 判"的可操作旋钮（MT7 的 UI 接上即可）。
const PBR_REFL_MIN_F0: f32 = 0.03;
/// 触发判据 ②：`roughness` 高于本值不发线（**成本控制的主闸门**，也是「光滑度」滑杆的落点）。
/// 理由：一条方向射线只代表得了"足够窄"的那一瓣镜面（MT4-1 的成本红线就是"只有一条"）；
/// `roughness` 大时 GGX 瓣宽远超一个方向能代表的尺度 ⇒ 那条射线既代表不了那一瓣（**方差巨大**），
/// 又与 MT3 已经加过的 `F0·(amb+gi)` 环境项重复 ⇒ 不如不追。
///
/// **取 0.05**：着色按面量化（MT8-6 + `face_view_dir`）⇒ 一个面只有**一个**样本，瓣宽一旦大于面，
/// 一个面就被好几个不同方向的内容"糊"成一个值 —— 那既不是镜面也不是逐面平色。
/// **瓣半角 ≈ atan(roughness) ⇒ roughness ≤ 0.05（瓣 ≤ 2.9°）**才值得一条射线；
/// 0.05 以上由 `fade`（反射段的替换式权重）连续收敛回 `F0·(amb+gi)` 近似口径。
/// 本值同时是「光滑度」滑杆的门限：**≥ 96% 才是镜面**（见下）。
///
/// **落到「光滑度」滑杆**（`smooth_pct_to_roughness`，线性）：roughness ≤ 0.05 ⇔ **光滑度 ≥ 96%**
/// （100% ⇒ roughness 0 = 完全镜面）；96% 以下按 `fade`（`dda_main`）连续淡出到 0 ⇒ 扫滑杆时
/// 亮度单调、门限处不出现台阶。⇒ **要镜面就把某块涂成光滑度拉满**。
/// 城堡场景 256 个 MATL 的 `_rough` 全是 `"0.1"` ⇒ roughness ≈ 0.098 > 0.05 ⇒ **默认场景整片不发线**
/// （`gate_dda_trace` ≈ 1.4ms 量级）。数据上偏光的材质被挡在门外是**有意**的：按面单样本下宽瓣只会带来蜡感。
const PBR_REFL_MAX_ROUGH: f32 = 0.05;
/// 抖动锥的张角系数：锥半角 ≈ `atan(本值 × roughness)`（rad）。粗糙度越大瓣越宽 ⇒ 取样方向越散，
/// 同时权重按 `dda_main` 里的 `fade` 连续淡出到 0（⇒ 粗糙度扫描下总亮度单调、无跳变）。
const PBR_REFL_CONE_K: f32 = 1.0;

// ---- 「基础」开关位（`light_u.g.base_flags`，菜单「渲染/基础」；**置位 = 算**）----
// `BASE_FLAG_*` 两个常量定义在 `bindings.wesl`（紧挨 uniform 声明）。位编码与 Rust
// `BaseSettings::FLAG_*` 同步。

// ---- MT8-3（反射缓存）代码已删除（历史记录）----
// 这里曾经是 `REFL_CACHE_SLOTS` / `REFL_REFRESH_K` / `refl_key_of` / `refl_slot_of` / `refl_seed`
// 五个符号：反射缓存的槽数、旋转刷新周期、面键、键→槽哈希、按面取的抖动种子。
// **用户实测判为负优化，整体删除**（删掉后逐像素发线的成本 ≈1.5~1.6ms，收益对不上复杂度）
// ⇒ 不要再按旧注释把它们加回来（`@group(1)` 保持 0..9，无乒乓、无 hashmap）。
// **当时**删除后种子回到了"按像素"（MT8-3 改按面只是为了"同面复用逐位一致"）。MT8-6 起种子**又回到按面**
// —— 但这次的理由是**画面正确性**（反射逐体素 + 去颗粒感，见文件头的 MT8-6 段与 `dda_main` 的反射段），
// 与"复用"无关：这里仍然**每个触发像素各发一条射线**，没有槽、没有刷新、没有写回。

/// 在**以 `r` 为轴、半角 ≈ `atan(spread)` 的锥内**取一个方向（`u` = [0,1)² 抖动，来自 `gi_rand2`）。
/// 切线基的构造与 `gi/common.wesl::gi_cos_hemisphere` **同一个写法**（选一个与 `r` 不平行的轴），
/// 只是把"余弦半球"换成"面积均匀的锥"：这一条射线代表的是**整个镜面瓣**，不是对 BRDF 做重要性采样
/// ⇒ 用锥内**面积均匀**的径向分布（`sqrt(u.x)`）。
/// 本函数只做几何，**不是 RNG**（随机数在调用点由现成的 `gi_rand2` 给出，不另造）。
fn refl_cone_dir(r: vec3<f32>, u: vec2<f32>, spread: f32) -> vec3<f32> {
  let axis = select(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 0.0), abs(r.z) > 0.9);
  let t = normalize(cross(axis, r));
  let b = cross(r, t);
  let rad = spread * sqrt(u.x);
  let phi = 6.2831853 * u.y;
  return normalize(r + t * (rad * cos(phi)) + b * (rad * sin(phi)));
}

// ============================================================================
// MT2-4 · 调试通道「强制所有表面用资产 N」（`docs/PLAN.md` §4 MT2-4 / MT3-2）
//
// ⚠️ MT3 起**这里只剩一个编译期常量**：取值与实现都搬进了 `common.wesl::fetch_material`
// （调试通道与正式路径**共享同一条代码** —— MT2-4 那条"单独的 albedo 直出路径"已删除）。
// 关掉 / 打开、以及"看什么"的说明都在 `fetch_material` 的文档注释里；本文件只把它作为实参传下去。
// ============================================================================

/// `0xFFFFFFFF` = **关闭**（每个表面走自己条目的变体：`IS_PBR` 决定参数从哪来）；
/// 设为**槽号 N**（= `assets/textures/pbr/` 字典序第 N 个材质，与资产表的 `*_slot` 同一个号）
/// ⇒ **所有不透明表面**都强制用资产 N 的**整份材质**（贴图 + 标量 + IOR）。
/// 编译期常量 ⇒ 关闭时 `fetch_material` 里那条调试分支被折掉，正常路径零开销。
const PBR_DEBUG_ASSET: u32 = 0xFFFFFFFFu;

/// 状态机产物：最终命中的（不透明）表面 / 沿途累积的透射色 / 累积的反射项 / 当前射线方向。
struct GlassOut {
  ray: WorldHit,
  tint: vec3<f32>,
  refl: vec3<f32>,
  dir: vec3<f32>,
}

/// 玻璃状态机：每轮处理一个界面事件（入射面 Fresnel 分反射/折射，出射面折射回空气或全内反射，
/// 命中不透明实体则返回由调用方着色）；`t_min` 是 beam 预 pass 的保守起点，只对第一段有效。
fn trace_glass(origin_voxel: vec3<f32>, dir_voxel: vec3<f32>, t_min: f32, max_t: f32,
               sun_dir: vec3<f32>, sky: vec3<f32>, sun_c: vec3<f32>) -> GlassOut {
  var org = origin_voxel;
  var dir = dir_voxel;
  var t_start = t_min;
  var in_medium = false;
  var tint = vec3<f32>(1.0);
  var refl = vec3<f32>(0.0);
  for (var bounce = 0u; bounce < GLASS_MAX_BOUNCE; bounce = bounce + 1u) {
    var seg = vec3<f32>(1.0);
    var best: WorldHit;
    if (in_medium) {
      // 介质段走**场景**查询（近场起点 ≡ 原来的单卷主世界路径；远场起点落在哪一级就查那一级）——
      // 远场玻璃的出射面在远场级里，单卷主世界查不到（见 `world_trace_medium_scene`）。
      best = world_trace_medium_scene(org, dir, max_t, true, &seg);
    } else {
      best = world_raycast(make_grid(0u), org, dir, max_t, world_cfg_primary(t_start, 3u));
    }
    tint = tint * seg;
    if (!best.hit) { return GlassOut(best, tint, refl, dir); }
    // 界面事件的判据是「这个材质是不是介质」，只当布尔用（不参与颜色计算，透射色来自 `world_trace_medium`
    // 的 `seg`）⇒ 一律读 `FLAGS_TRANSMISSIVE` 位。PBR 变体里 transmission 字节属于 `asset`
    // （`docs/PLAN.md` D1），按字节读会得到随机值把不透明材质误判成玻璃。
    if ((palette_flags(best.palette_base, best.pal) & FLAGS_TRANSMISSIVE) == 0u) {
      return GlassOut(best, tint, refl, dir);
    }
    let p_hit = org + dir * best.t;
    // 界面法线取**逐体素法线**（`voxel_normal`：可见面 + 邻域梯度倾斜，与不透明路径 / GI 二次顶点
    // 同一个函数、同一口径）—— 玻璃的几何量（起点外推与这里的折射 / 全反射）都按它算，
    // 描述的才是体素场那个平滑面。它与起点外推用的是同一个法线 ⇒ 入射角与真实界面自洽。
    // 掠射（法线倾斜 + 视线很斜）时 `dot(-dir, nn)` 可能为负 ⇒ 被下面的 clamp 成 0 = 掠射入射。
    let nn = voxel_normal(make_grid(0u), best.voxel, best.face_id, best.n);
    let ci = clamp(dot(-dir, nn), 0.0, 1.0);
    // MT3-8：折射率与 F0 都从**材质**取，不再有本文件里的硬编码常量（硬约束 9：F0 只有一个来源）。
    // 每轮界面各取一次：进出的是**同一个介质**，而 `best.pal` 就是该介质的材质（`world_trace_medium`
    // 在段末端返回的是介质材质本身）⇒ 不引入"当前介质 IOR"这种额外状态。
    // 平凡变体没有 IOR 字段 ⇒ `material_ior` 给默认 1.5、F0 = ((1.5−1)/(1.5+1))² = 0.04
    // ⇒ 折射与 Fresnel 与改动前玻璃那两份硬编码（IOR 1.5 / F0 0.04）一致（本条验收）。
    let ior = material_ior(best.palette_base, best.pal);
    // ---- 视向量（**这条界面事件的着色量**）：**逐面、且以相机为锚** ----
    // `v` = "由本界面的体素中心指向**相机**" ⇒ 同一玻璃面内所有像素同值（逐面），且随视角变化。
    // 它喂两件事，都是反光：① 界面 Fresnel（`fres`，含掠射增益）；② 反射方向 `reflect(-v, nn)`。
    // ⚠️ **锚点必须是相机（观察者），不能是本段射线的起点 `org`**：本函数的入射点就是命中面中心
    // （`center + nn·(0.5+ε)`）⇒ 拿 `org` 当锚会让 `v` 恒等于 `nn` ⇒ `fres` 恒为 `F0`（无掠射增益）、
    // `reflect(-v, nn) ≡ +nn`（反射线垂直射出该面、与视角无关）。
    // 对**首个**界面本式恰好等于 `dda_main` 传进来的 `fv`（同一把尺子 `face_view_dir`），
    // 对内部界面则是"相机看那个体素"的方向 ⇒ 仍然逐面恒定。
    // `ci`（几何量：折射/全内反射的判据）用**本段射线的精确方向** —— 那是光路几何、不是着色。
    let v = face_view_dir(view_u.cam_pos_voxel.xyz, voxel_center_world(make_grid(0u), best.voxel));
    if (!in_medium) {
      // 玻璃走的是折射状态机（不是 BRDF）：Fresnel 用**物理 IOR 导出的 F0**、不加槽级 specular 调制
      // （那条调制是给不透明 BRDF 的观感旋钮；`specular = 1.0` = 不调制 = 物理基值）。
      let f0 = dielectric_f0(ior, 1.0);
      let fres = f0 + (1.0 - f0) * pow(1.0 - clamp(dot(v, nn), 0.0, 1.0), 5.0);
      if (fres > GLASS_REFL_MIN) {
        // 反射方向由 `v` 给出（`reflect` 要的是入射方向 ⇒ 传 `-v`）⇒ 逐面（一个玻璃面一条反射方向）。
        let rdir = reflect(-v, nn);
        let r_org = p_hit + nn * GLASS_EPS;
        let rhit = world_raycast(make_grid(0u), r_org, rdir,
                                 max_t, world_cfg_full(0.0, 3u));
        var rc = sky_rgb();
        if (rhit.hit) {
          // ---- MT4-1 · 这条第三处着色表达式已接进统一入口 ----
          // 与主 pass、GI 二次顶点走**同一组函数**：`fetch_material` 取材质 + `brdf_reflected` 求值；
          // 着色法线也同样是 `voxel_normal` 的隐式邻域梯度（与主 pass 一致）。
          // `SurfaceLight` 刻意与 `gi/ray.wesl` 的二次顶点**逐字对齐**（硬约束 3 不分叉），差异只有两条：
          //   · `sun_vis = 1.0`（**不做 NEE**：不给玻璃的反射再加一条阴影射线，沿用本条既有的"无遮挡"
          //     近似 —— 有意为之，不是漏了）；
          //   · 天光 = `sky · GLASS_REFL_AMBIENT`（玻璃反射沿用自己那个更暗的常量天光，与旧式同源）；
          //     `gi = 0`（反射点没有 GI 缓冲，与二次顶点同一条约定）。
          // （统一入口的直接后果）反射命中自发光表面时会看到它的自发光（`emissive_radiance`
          // 与主 pass / 二次顶点同一句）。
          let rgg = make_grid(u32(rhit.obj_id) + 1u);
          let rn = voxel_normal(rgg, rhit.voxel, rhit.face_id, rhit.n);
          let mr = fetch_material(rhit.palette_base, rhit.pal, rhit.point, rn, PBR_DEBUG_ASSET);
          // 反射命中点的视向量：用它**自己那条射线**（起点 `r_org` → 命中体素中心）⇒ **逐面**
          // （硬约束 3：所有着色点共用 `face_view_dir` 同一份契约）。
          rc = brdf_reflected(mr, rn, face_view_dir(r_org, voxel_center_world(rgg, rhit.voxel)),
                              SurfaceLight(sun_dir, sun_c, 1.0, sky * GLASS_REFL_AMBIENT, vec3<f32>(0.0)))
            + emissive_radiance(mr, GI_EMIT_GAIN);
        }
        refl = refl + tint * fres * rc;
      }
      let eta = 1.0 / ior;
      let k = 1.0 - eta * eta * (1.0 - ci * ci);
      dir = normalize(eta * dir + (eta * ci - sqrt(max(k, 0.0))) * nn);
      in_medium = true;
    } else {
      let eta = ior;
      let k = 1.0 - eta * eta * (1.0 - ci * ci);
      if (k < 0.0) {
        dir = reflect(dir, nn);
      } else {
        dir = normalize(eta * dir + (eta * ci - sqrt(k)) * nn);
        in_medium = false;
      }
    }
    org = p_hit + dir * GLASS_EPS;
    t_start = 0.0;
  }
  return GlassOut(WorldHit(false, 0.0, vec3<f32>(0.0), 0u, vec3<f32>(0.0), 0u, vec3<i32>(0), 0, 0u),
                  tint, refl, dir);
}

/// 自动曝光 + sRGB 编码输出（天空 / 不透明 / 玻璃三条路径共用）。
/// 曝光系数必须在 sRGB 编码之前乘。
fn write_output(coord: vec2<i32>, col: vec3<f32>) {
  let exposure = bitcast<f32>(eye_adapt_ro[0u]);
  let exposed = col * select(1.0, exposure, exposure > 0.0);
  textureStore(out_tex, coord, vec4<f32>(linear_to_srgb(exposed), 1.0));
}

/// **覆盖空档的邻域搜索半径**（GI texel）：`gi_upsample_joint` 的 2×2 与"最近 texel"两档都失败时，
/// 在最近 texel 周围 `(2R+1)²` 里找数据（先同面、再任何有效 texel）。见该函数里的 WHY。
const GI_UPSAMPLE_HOLE_R: i32 = 3;

/// **小面的"跨面联合池化"半径**（`dda_face_main`，GI texel）—— 2026-10-03 第九轮，
/// 针对用户实拍的"自发光体素周围网格状光纹"。
///
/// **症状与隔离**（用户实测，各切一个菜单开关）：暗场里一个自发光方块周围，落光呈**固定格子**
/// （单元格 = 一个可见体素面）。**GI 分辨率 1/1 ⇒ 完全消失**（不是变小）；**降噪质量关 ⇒ 完全消失**；
/// **WAL 关 ⇒ 更明显**（面表池化在压它，但只压 ~1.6×）。三条合起来指向：**每个面的估计噪声**
/// 被降噪链稳定成一张固定格子 —— 1/1 时一个可见面跨 ~8 个 texel，等于自动多平均 8 倍 ⇒ 看不见了。
///
/// **量级对得上**：1/4 档下一个可见面通常只覆盖 **1 个 GI texel** ⇒ `dda_face_accum` 的"面均值"
/// 只有**一个样本**；而小而亮的发光体在半球采样下命中率 p ≈ 1%~8% ⇒ 单 texel 的相对 σ =
/// √((1−p)/(p·M))，M ≈ 256 个窗口样本时是 **5%~7%** —— 暗场里就是绕着发光体的那张格子。
///
/// **修法**：在本面的**代表 texel**（= 本 pass 的认领者）周围 `(2R+1)²` 里，用**全管线同一个判据**
/// `gi_same_surface` 把**共面**的 texel 值平均进来 —— 与 `gi_wal_*` 的池化是同一条道理
/// （"共面邻面物理上几乎是同一个积分"，且同样**不跨台阶/棱边**），只是作用在**降噪后的场**上。
/// R=3（49 tap，实测多数面上可接受 ~40 个）⇒ 这类残差 ÷≈6。
/// **成本**：只在 `gi_n ≤ GI_FACE_POOL_MIN_N` 的面上跑一次（每面 49 次导引读 + 纹理取值），
/// 按 1/4 档 ~4.5 万个面算 ≈ +0.1~0.3 ms/帧，**不加任何射线、不加 pass**。
const GI_FACE_POOL_R: i32 = 3;
/// 触发池化的**面均值样本数**上限：`FACE_W_GI_N` 数的是"这个面在 GI 网格上占几个 texel"。
/// 取 4：1/4 档下多数可见面是 1 个 texel（值 = 单样本）、2 档下 2~4 个 —— 这些"面均值"没有
/// 统计意义，池化是纯收益；更大的面**自己就在平均**（跨 texel 的无偏均值，正是设计要的
/// 「面内恒定」），不该再被池化糊掉。
const GI_FACE_POOL_MIN_N: u32 = 4u;

/// **几何感知上采样**（joint bilateral）—— `dda_main` 取 GI 的唯一入口。
/// GI 跑在低于全分辨率的网格上（`GiSettings.gi_div` = 1 / 2 / 4），纯双线性会把**几何边界
/// 两侧**的值混在一起（棱边/墙角一侧的亮 GI 渗到另一侧），而这正是「GI 低于全分辨率」唯一
/// 真正丢掉的信息。档位越低（1/4）这一层越要紧：一个 GI texel 覆盖 4×4 个全分辨率像素。
/// 做法：取 GI 网格上最近的 2×2 texel，**只接受与本像素同一个面**的 texel
/// （`gi_same_surface`：平面归属靠键 ⇒ 薄板另一侧 / 隔一格的平行面不可能被接受；
/// 相似度靠**存的着色法线** ⇒ 凸棱上那一段倾斜的法线与本像素比仍在容差内 ⇒ 与着色一致地连起来），
/// 按双线性权重加权平均。退化顺序：一个都没接受 ⇒ **最近的 texel**（它至少在几何上覆盖本像素，
/// 比双线性更不容易渗色）⇒ 最近 texel 无有效数据 ⇒ **`GI_UPSAMPLE_HOLE_R` 邻域搜索**
/// （先找同一个面的最近 texel，再退到任何有效 texel）⇒ 都没有 ⇒ 传进来的纯双线性结果。
/// 有效性判据 = 导引的「键非零」（与降噪侧一致）；最近 texel 那一档直接用纹理的 `a`（那里不插值，
/// `a` 恒是 0/1）。
/// `n` = 本像素的着色法线（`dda_main` 自己那条）—— 相似度那一条用它。
/// 返回值 `a` = **"本函数额外找到了数据"**（0/1）：调用方用 `cov = max(双线性 a, 本值)` ——
/// 只有邻域搜索那一档会给出 1（前面的档都由调用方的双线性 `a` 覆盖）。
/// 代价：每全分辨率像素 4 次导引读（每次 3 word）；**只有前两档都失败**的像素才付邻域搜索
/// （最多 `(2R+1)²` 次导引读），**不加任何射线、不加 pass**。
fn gi_upsample_joint(
  guv: vec2<f32>,
  gsize: vec2<u32>,
  k0: u32,
  k1: u32,
  n: vec3<f32>,
  fallback: vec3<f32>,
) -> vec4<f32> {
  let g = guv * vec2<f32>(f32(gsize.x), f32(gsize.y));
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
      let gk0 = gi_guide_out[tb + GI_DEN_G_KEY0];
      let gk1 = gi_guide_out[tb + GI_DEN_G_KEY1];
      if ((gk0 | gk1) == 0u) { continue; }
      if (!gi_same_surface(gk0, gk1, gi_guide_out[tb + GI_DEN_G_NRM], k0, k1, n)) { continue; }
      // 有效 texel 的 a 恒为 1（`gi_out` 与降噪链都写 `vec4(v, 1.0)`）⇒ `rgb` 就是 gi 本身。
      acc = acc + textureLoad(gi_tex, c, 0).rgb * bw;
      wsum = wsum + bw;
    }
  }
  if (wsum > 0.0) { return vec4<f32>(acc / wsum, 0.0); }
  // 一个同面的 texel 都没有（凸角 / 贴天空的轮廓）：退回最近的 texel。
  let nc = clamp(base + vec2<f32>(select(0.0, 1.0, f.x >= 0.5), select(0.0, 1.0, f.y >= 0.5)),
                 vec2<f32>(0.0), vec2<f32>(f32(gsize.x) - 1.0, f32(gsize.y) - 1.0));
  let nci = vec2<i32>(nc);
  let sn = textureLoad(gi_tex, nci, 0);
  if (sn.a > 0.5) { return vec4<f32>(sn.rgb, 0.0); }
  // ---- 邻域搜索（`GI_UPSAMPLE_HOLE_R`）----
  // 走到这里 = **本像素的 4 个覆盖 texel 一个有效的都没有**（含"最近 texel 是天空"）。
  // 直接返回 `fallback` 会让这一格 **GI = 0 且 cov = 0**（只剩全额天光兜底）⇒ 背光面上是一块
  // **黑洞**：1/4 档下一个 texel = 4×4 像素，而**远景细几何 / 薄板 / 掠射面**在自己的 4 个 texel 上
  // 常常一个都没命中（覆盖它的是天空或别的面）。实测症状（2026-10-03 第八轮用户配图）：
  // "远处出现 GI 黑洞"（远景天际线的背光面上一块块暗斑）。
  // 半径 3（±3 texel = ±12 屏幕像素）足以接上同一个面上的邻域证据；判据仍是**同一个**
  // `gi_same_surface` ⇒ 不跨几何。先找同面（最近者），找不到才退到"任何有效 texel"（宁可借错面
  // 的一点光，也不要黑洞 —— 借来的值仍受天光地板与后续降噪约束）。
  var best_ok = false;
  var best_d2 = 0;
  var best = vec3<f32>(0.0);
  var any_ok = false;
  var any_d2 = 0;
  var any = vec3<f32>(0.0);
  for (var dy: i32 = -GI_UPSAMPLE_HOLE_R; dy <= GI_UPSAMPLE_HOLE_R; dy = dy + 1) {
    for (var dx: i32 = -GI_UPSAMPLE_HOLE_R; dx <= GI_UPSAMPLE_HOLE_R; dx = dx + 1) {
      let c = nci + vec2<i32>(dx, dy);
      if (c.x < 0 || c.y < 0 || c.x >= i32(gsize.x) || c.y >= i32(gsize.y)) { continue; }
      let tb = (u32(c.y) * gsize.x + u32(c.x)) * GI_DEN_GUIDE_WORDS;
      let gk0 = gi_guide_out[tb + GI_DEN_G_KEY0];
      let gk1 = gi_guide_out[tb + GI_DEN_G_KEY1];
      if ((gk0 | gk1) == 0u) { continue; }
      let d2 = dx * dx + dy * dy;
      if (!any_ok || d2 < any_d2) {
        any_ok = true;
        any_d2 = d2;
        any = textureLoad(gi_tex, c, 0).rgb;
      }
      if (gi_same_surface(gk0, gk1, gi_guide_out[tb + GI_DEN_G_NRM], k0, k1, n)
          && (!best_ok || d2 < best_d2)) {
        best_ok = true;
        best_d2 = d2;
        best = textureLoad(gi_tex, c, 0).rgb;
      }
    }
  }
  if (best_ok) { return vec4<f32>(best, 1.0); }
  if (any_ok) { return vec4<f32>(any, 1.0); }
  return vec4<f32>(fallback, 0.0);
}

// ---- WAL（世界空间累积层）的注入点在**写侧**，不在显示侧（2026-10-03 第八轮）----
// 这里曾经是 `gi_wal_blend`：在显示端把降噪结果**整个替换**成面表的池化值。那等于把降噪链的
// **空间滤波**（`gi_face_flatten` 的逐面平均 + 5 轮 atrous）白算 —— WAL 自己的池化只有 7×7 个面
// （≈38 屏幕像素，且约一半 tap 是空的：没被认领过的面没有条目），残余噪声在暗场里就是**固定的
// 斑纹**（世界锚定 ⇒ 等它"收敛"不会消失）。现在改成：`gi/screen.wesl::gi_ss_main` 在写 `gi_out`
// 时按成熟度把面表值混进本 texel 的估计（连同 `FACE_W_RAW*` 合并量一起），随后照常走整条降噪链。
// ⚠️ **不要把它加回显示端**（重复注入 + 又绕开 atrous）。成熟度的分子/分母（`Σn`、`GI_WAL_MATURE_N`）
// 与失效语义仍全在 `gi/wal.wesl`。

/// 本面的 GI 输入。
/// `use_face = true` ⇒ 走**逐面均值**（认领路径：`dda_face_accum` 已把该面所有 GI texel 的值累加进
/// 槽里）。它同时做两件事：面内 GI 恒定（= 1/4 档的边缘锐利、无 4px 台阶），以及**无偏降噪** ——
/// 同一个面的所有 texel 的 GI 射线起点都是该体素中心（`gi_ss_main` 的 `org`）⇒ 它们是对**同一个
/// 积分**的独立估计 ⇒ 取均值的方差是单个 texel 的 1/K。只取某一个 texel（认领者）时，它的降噪
/// 残差会被整面一起放大成**相干闪烁**（原来逐像素各自独立、被眼睛平均掉），面越大越明显。
/// `use_face = false` ⇒ 按 `uv` 逐像素采样（未认领 / 撞键的面，与去重前的口径一致）。
struct GiInput {
  use_face: bool,
  face_avg: vec3<f32>,   // 已含增益，与 `gi_tex.rgb` 同一量纲
  face_cov: f32,
  uv: vec2<f32>,         // 逐像素路径的采样位置（全分辨率归一化）
}

/// 反射抖动种子：**按面**取（MT8-6 —— 一个可见体素面 = 一条反射方向 = 一个反射值，与材质色同格）。
/// 键 = grid 局部体素坐标 + 面号 + obj_id：`voxel` 是 **grid 局部**坐标 ⇒ 同一世界坐标上主世界与
/// 物体可能各有一个体素，带上 obj_id 才不会让两者共用种子。
/// `gi_mix` 是 GI 侧现成的哈希（`gi/common.wesl`；本工程 WESL 解析器不接受 `^`，故按乘-加-移位写）。
fn refl_seed(voxel: vec3<i32>, face_id: u32, obj_id: i32) -> u32 {
  let rkey = face_id | ((u32(obj_id) + 1u) << 3u);
  return gi_mix(
    u32(voxel.x) * 0x9E3779B1u
    + gi_mix(u32(voxel.y) * 0x85EBCA77u
    + gi_mix(u32(voxel.z) * 0xC2B2AE3Du
    + gi_mix(rkey * 0x27D4EB2Fu + 0x165667B1u))));
}

/// **镜面反射的内容（唯一实现）**：从一个可见面出发追反射链，返回累积的反射辐亮度 `refl_env`。
/// **不含**主表面那一次 Fresnel 与粗糙度淡出、也不含"替换式"那一扣 —— 那三项在消费侧按同一个
/// `fade` 一起算（见 `shade_face` 的反射段）。
/// 光源项从 uniform 自取（与 `shade_face` 的调用方传进来的那三份**逐字同源**：同一个
/// `light_u.lights[0]` / `sky_color`）⇒ 两边不会因为"谁传了什么"而分叉。
fn refl_env_chain(p_voxel: vec3<f32>, n: vec3<f32>, v: vec3<f32>, rseed: u32, m: MaterialParams,
                  frustum_length: f32, n_off: f32) -> vec3<f32> {
  let sun_dir = light_u.lights[0].kind_pos_dir.yzw;
  let sun_c = light_u.lights[0].color_intensity.xyz * light_u.lights[0].color_intensity.w;
  let sky = light_u.sky_color.xyz;
  // 起点 = 与阴影射线**同一个构造**（命中体素中心沿 n 推 `world::surface_offset` ⇒ 恰在表面外侧），
  // 口径一致、不另立第二套外推 —— `n_off` 由调用方（`shade_face`）算好传入，两份公式只留一份。
  // 它同时是下面那次着色里"本条射线的起点"。
  // 镜面方向由**逐面**视向量 `v` 给出（`reflect` 要入射方向 ⇒ 传 `-v`）⇒ 一个体素面一条方向、
  // 一个反射值 = "反射逐体素"。代价：反射是**块状镜面**（按面采样），不再是逐像素的清晰镜面 ——
  // 本档的预期，见 `common.wesl::face_view_dir` 里"已知代价"那一段与两次实测的历史。
  let mirror = reflect(-v, n);
  var rdir = refl_cone_dir(mirror, gi_rand2(rseed), PBR_REFL_CONE_K * m.roughness);
  // 抖动后的方向若掉到面下方（掠射 + 大锥角）会立刻自命中所在体素 ⇒ 退回完美镜面方向。
  if (dot(rdir, n) <= 0.0) { rdir = mirror; }
  let r_org = p_voxel + n * n_off;
  // ============================================================================
  // MT4-5 · **反射链**（`nest` = 最多再嵌几级，见 `LightGlobals::refl_nest`）
  //
  // 一次反射的内容 = 命中点自己的出射亮度 **+** 它自己的反射（前提是它**也**是镜面）。串起来：
  //   `refl_env = local(0) + w(0)·[ local(1) + w(1)·[ … ] ]`
  // `w(k)` = 第 k 级命中面的 `fresnel_schlick(f0, v·n) × 粗糙度淡出`，`chain_w` 是累积的 Π w。
  // `refl_env` 累积的是**不含主表面那一次 Fresnel** 的反射内容 —— 那一次在循环外乘（见
  // `shade_face` 的「替换式」那段），与嵌套层级无关。
  //
  // **成本闸门**：只有"上一级命中的面**本身也是镜面**"（与主表面同一个 `PBR_REFL_*` 判据）
  // 才继续 —— 墙 / 地形 / 普通方块在**第一级就断链** ⇒ 日常场景下 `nest` 拨到几都一样，
  // 多出来的只是一次判据。钱只花在真的把镜子对着镜子摆的地方：最坏 `nest + 1` 条反射射线
  // （档 3 时每条再 +1 条阴影射线）。
  //
  // **画风不变**：每一级的方向都取自**逐面**视向量与按面种子（MT8-6）⇒ 嵌套的镜像仍是
  // 块状镜面，一个体素面一个反射值。
  // ============================================================================
  let tier = light_u.g.refl_tier;
  let nest = light_u.g.refl_nest;
  var refl_env = vec3<f32>(0.0);
  var chain_w = vec3<f32>(1.0);
  var chain_org = r_org;
  var chain_dir = rdir;
  for (var lvl = 0u; lvl <= nest; lvl = lvl + 1u) {
    // WHY: 反射射线必须**穿透介质**（`stop_on_air = false`）—— 用不透明 DDA 时玻璃在镜中是一块
    // 平色不透明面（既不透明也不染色），而"隔着一层玻璃看镜子"时镜面自己的反射射线会被前方玻璃
    // 当成墙体挡住 ⇒ 镜子照出一片玻璃的平色。两种摆法都因此看不到效果。
    // 逐级累积的 `r_tint` = 这条反射射线沿途介质段的乘性吸收（与主 pass `col * through + refl`
    // 同一个结构）；每级独立起算，不跨级复用。
    // 已知简化：玻璃**表面自身**的界面反射与 in-scatter 不进镜像（这一级只取"透过它看到的东西"）。
    var r_tint = vec3<f32>(1.0);
    let h = world_trace_medium_scene(chain_org, chain_dir, frustum_length, false, &r_tint);
    if (!h.hit) {
      // 逃逸到天空 ⇒ 反射内容就是天空，链条结束。**档 ≥2** 走 `sky_primary`（方向性天空 +
      // 太阳盘/光晕：镜子里看得见太阳，与主 pass 的 miss 同一个口径）；**档 1** 走 `sky_rgb()`
      // （纯色天光，全工程唯一的 miss 辐亮度）。两侧都只算 ALU、不发线。
      refl_env = refl_env
               + chain_w * r_tint
                 * select(sky_rgb(), sky_primary(chain_dir, sun_dir, sun_c), tier >= 2u);
      break;
    }
    // 命中点**必须**经 `fetch_material` + `brdf_reflected`（硬约束 1：不许手写第二套近似
    // 着色 —— 那正是 MT4-1 刚从 `trace_glass` 里清掉的那类第四处表达式）。
    let hgg = make_grid(u32(h.obj_id) + 1u);
    let hn = voxel_normal(hgg, h.voxel, h.face_id, h.n);
    let hm = fetch_material(h.palette_base, h.pal, h.point, hn, PBR_DEBUG_ASSET);
    // 命中点 = 一个**完整的着色点**：材质与着色都走全工程同一组函数（`fetch_material` +
    // `brdf_reflected` + `emissive_radiance`，硬约束 1/3 不分叉），差异只有两条，都是写明的：
    //   · `gi = 0`：那是本像素**屏空间**的缓冲，反射点常常不在屏幕上，借过来就是错的；
    //   · `sun_vis`：**档 3** 走真实 NEE；**档 1/2** 取 1.0（无遮挡近似 —— 与 `trace_glass` 的
    //     玻璃反射同一条有意约定：不给反射再加一条阴影射线）。天光 `amb` 取
    //     `sky · GI_SKY_AMBIENT`（与 GI 二次顶点逐字相同，反射点没有 GI 缓冲 ⇒ 没有 `cov` 地板）。
    // `emissive_radiance` 与玻璃反射同一句 ⇒ 发光体素在镜子里发光。
    let hamb = sky * GI_SKY_AMBIENT;
    // 命中点的视向量：用它**自己那条射线**（起点 `chain_org` → 命中体素中心）⇒ **逐面**
    // （硬约束 3：所有着色点共用 `face_view_dir` 同一份契约）。
    let hv = face_view_dir(chain_org, voxel_center_world(hgg, h.voxel));
    // 命中点沿自己的着色法线外推半个体素 + eps —— 与主路径的阴影射线**同一个构造**。
    // 它同时是阴影射线的起点与（若继续）下一级反射射线的起点，口径一致、只有一份公式。
    let h_off = 0.5 / max(max(abs(hn.x), abs(hn.y)), max(abs(hn.z), 1e-3))
              + SHADOW_SURFACE_EPS;
    let h_start = h.point + hn * h_off;
    var h_vis = 1.0;
    // 反射链的太阳 NEE 与不透明路径**同一个「直光阴影」开关**（关掉就是全场景不做遮挡）
    if (tier >= 3u && (light_u.g.base_flags & BASE_FLAG_SHADOW) != 0u) {
      h_vis = sun_transmittance(h_start, sun_dir, h.voxel, h.obj_id);
    }
    // 「本级也值得追，**并且**还允许再追一级」——前半句与主表面**同一个判据**
    // （F0 亮度门 × 粗糙度淡出 > 0）；后半句是 `lvl < nest`。
    // 两件事必须分开判：**扣掉本级那条近似，前提是下一级真的会去取代它** ——
    // 末级（`lvl == nest`）命中镜面时没有下一级，扣了就是白丢一份能量（回归到"旧口径"而非更好）。
    let hf0 = f0_of(hm);
    let h_fade = clamp(1.0 - hm.roughness / PBR_REFL_MAX_ROUGH, 0.0, 1.0);
    let h_continue =
      lvl < nest && h_fade > 0.0
      && dot(hf0, GI_LUMA) >= PBR_REFL_MIN_F0;
    var h_local = brdf_reflected(hm, hn, hv,
                                 SurfaceLight(sun_dir, sun_c, h_vis, hamb, vec3<f32>(0.0)))
                + emissive_radiance(hm, GI_EMIT_GAIN);
    if (h_continue) {
      // 「替换式」（与主表面同一条口径）：本级既然是镜面，它**自身**那条 `F0·(amb+gi)` 近似
      // 由下一级的真实反射取代 ⇒ 先扣掉，否则镜面环境项又被算一遍。
      h_local = h_local - hf0 * hamb;
    }
    refl_env = refl_env + chain_w * r_tint * h_local;
    if (!h_continue) { break; }
    // ---- 沿本级继续反射：方向 / 种子 / 起点 ----
    // 方向与主表面**同一个构造**（`reflect(-v, n)`，`v` 逐面）。抖动种子把**层级**混进按面种子里
    // （`gi_mix` 是 GI 侧现成的哈希；本工程 WESL 解析器不接受 `^`，故按加-混写）。
    let h_mirror_dir = reflect(-hv, hn);
    var h_dir = refl_cone_dir(h_mirror_dir, gi_rand2(gi_mix(rseed + lvl * 0x9E3779B1u)),
                              PBR_REFL_CONE_K * hm.roughness);
    // 抖动后掉到面下方（掠射 + 大锥角）会立刻自命中所在体素 ⇒ 退回完美镜面方向。
    if (dot(h_dir, hn) <= 0.0) { h_dir = h_mirror_dir; }
    chain_org = h_start;
    chain_dir = h_dir;
    // 下一级的贡献要乘上本级的 Fresnel × 粗糙度淡出（`h_fade > 0` 已由 `h_mirror` 保证）。
    chain_w = chain_w * fresnel_schlick(hf0, max(dot(hv, hn), 0.0)) * h_fade;
  }
  return refl_env;
}

/// **逐面着色（唯一实现）**：`dda_main`（未认领 / 撞键的面，逐像素）与 `dda_face_main`
/// （被认领的面，逐面算一次）共用同一份代码 ⇒ 去重不改变任何像素的结果。
///
/// 输入 = 一个**可见体素面**（`first`）+ 本面射线的远平面距离 + GI 输入（见 `GiInput`）。
/// 除 GI 外的全部输入都只由这个面决定（材质量化到体素中心 / 着色法线 / 视向量锚点 /
/// 阴影与反射射线的起点与种子 / 玻璃状态机的入射点与方向），这就是它可以按面缓存的前提。
/// ⚠️ `ray_dir` 是唯一一个**逐像素**的实参，且**只被天空背景那一行消费**（`sky_primary` 要用它定太阳盘）。
/// 命中面（含逐面缓存那条路径）根本不会走到那一行 ⇒ 那里的实参不被消费，但必须是合法值
/// （不能是零向量：`sky_primary` 里对方向做了点积）。
///
/// `n_hint` / `n_hint_ok`：调用方**已经算好**的着色法线（`dda_face_main` 从降噪导引里拿 —— 那里存的
/// 就是同一个键的 `voxel_normal`，逐位相同）⇒ 省掉 `voxel_normal` 的 4 次邻域点查询。
/// `dda_main` 传 `false`（它没有可直接复用的同键法线）。
///
/// `sun_io`：太阳可见性的**借用口**（只有「覆盖 texel 回退」路径会传借用值，见 `GI_FACE_FALLBACK`）。
/// 入参 ≥ 0 ⇒ **直接采用、不发射阴影射线**（该值由 `dda_face_main` 逐面烘焙进 `FACE_W_SUN`）；
/// 入参 < 0 ⇒ 按原路自算（`ndl > 0` 且「直光阴影」开 ⇒ 一条 `sun_transmittance`）。
/// 出口恒是**本次着色实际用到**的值（早退分支写 0：原色直出 / 天空没有太阳量可报）——
/// `dda_face_main` 把它烘焙进槽，回退路径据此跳过逐像素的阴影射线。
/// 借用是**有意为之的近似**：回退的准入已保证被借的面与本像素同平面 / 同逐位着色法线 / 同材质、
/// 且屏距 ≤ 一个 GI texel（细节在该尺度下本来就是亚像素的）——与"整条着色都借用"的旧口径相比，
/// 借用面从"整色"缩到"只有射线量"，材质 / 法线 / 视角 / GI / 自发光全部逐像素。
fn shade_face(first: WorldHit, origin_voxel: vec3<f32>, ray_dir: vec3<f32>, frustum_length: f32,
              gi_in: GiInput, sun_dir: vec3<f32>, sky: vec3<f32>, sun_c: vec3<f32>,
              n_hint: vec3<f32>, n_hint_ok: bool, sun_io: ptr<function, f32>) -> vec3<f32> {
  // 先把入参读成局部量（`sun_io` 同时是出口，见函数头），再把出口清零：早退分支没有太阳量可报。
  let sun_hint = *sun_io;
  *sun_io = 0.0;
  var best = first;
  var through = vec3<f32>(1.0);
  var refl = vec3<f32>(0.0);
  // 首命中是不是玻璃（走 `trace_glass` 状态机）看 `FLAGS_TRANSMISSIVE` 位，不读 transmission 字节：
  // PBR 变体里那个字节属于 `asset`（`docs/PLAN.md` D1），按字节读会把不透明材质误判成玻璃。
  // 平凡变体下该位恒等于 `transmission > 0`（由写入侧 `wire.rs::pack_palette_entry` 维护）⇒ 行为不变。
  //
  // ---- MT8-7 · 玻璃的**整条光路按体素量化** ----
  // 喂给状态机的起点与方向都**只由命中体素决定** ⇒ 折射/全反射/介质吸收/背景采样**全部**只依赖
  // （命中体素 + 它被看见的那个面）⇒ 跑遍整个状态机，一个面上所有像素得到**完全相同**的结果：
  //   · 入射点 = 该**命中面的中心**沿**逐体素法线**外推半格（`fp + fnn·(0.5/|fnn| + eps)`）——
  //     与阴影/反射射线的起点**同一个构造**（`n_off`），不另立第二套外推；`fnn` 也是 `trace_glass`
  //     内部做折射 / 全反射的同一个法线 ⇒ 界面点与界面朝向自洽（描述的是那个平滑面，不是单张台阶面）。
  //   · 入射方向 = 该**体素中心**的视向量的反向（`-face_view_dir(相机, 体素中心)`）—— 与反光/高光
  //     同一把尺子（`face_view_dir` 的锚点就是体素中心）。
  // ⇒ **一个玻璃面 = 一个平色**（含透过它看到的那一格背景），与材质色、高光、反射**同一粒度**。
  // 成本**不变**（同样的循环、同样的射线数）。`t_min` 传 0 —— beam 预 pass 的 `t_min` 是沿
  // **相机那条射线**量的，换到这个起点后没有意义（而从这里到玻璃另一侧的距离本来就很短）。
  // 玻璃自身的不透明外观（in-scatter 的"另一端"）：非玻璃像素恒 0 ⇒ 下方整支被折掉、零开销。
  var glass_col = vec3<f32>(0.0);
  if (first.hit && (palette_flags(first.palette_base, first.pal) & FLAGS_TRANSMISSIVE) != 0u) {
    let fg = make_grid(u32(first.obj_id) + 1u);
    let fp = voxel_center_world(fg, first.voxel);
    let fnn = voxel_normal(fg, first.voxel, first.face_id, first.n);
    let f_off = surface_offset(fg, fnn, SHADOW_SURFACE_EPS);
    // 玻璃**界面点**：光路起点与 in-scatter 的阴影射线起点共用它，不另立第二套外推。
    let f_face = fp + fnn * f_off;
    let fv = face_view_dir(origin_voxel, fp);
    let g = trace_glass(f_face, -fv, 0.0, frustum_length, sun_dir, sky, sun_c);
    best = g.ray;
    through = g.tint;
    refl = g.refl;
    // ---- MT8-9 · in-scatter（与不透明路径**逐项同一套**光源项）----
    // `col` 里只有 `背景 × through` ⇒ 必须有 `玻璃自身的不透明外观 × (1 − through)` 补上来，
    // 否则中间档发黑；且 `透明度 = 0%`（`transmission = 0` ⇒ **不是介质** ⇒ 走真正的不透明路径）
    // 与中间档之间会出现模型跳变。物理上 `T → 0` **必须连续收敛**到不透明外观 ⇒ 这里用与不透明路径
    // **同一套**着色（硬约束 1：`fetch_material` + `brdf_reflected`，多一个**调用点**、不是新公式），
    // `SurfaceLight` 与不透明路径**逐字段同构**（太阳 + 阴影 + 环境 + GI）：
    //   · 太阳项必须经 `sun_transmittance` 那条**真阴影射线** —— `sun_vis` 恒 1 会让玻璃无条件吃到
    //     满太阳（深室内白得发光）；
    //   · GI 项必须接（与不透明路径同一个 `gi_upsample_joint` + 同一套键 + 同一个 `cov` 地板）——
    //     背光/阴影面那一侧 `ndl = 0` 且 `amb` 被 `cov` 地板压到最低，不透明面在同一处靠 GI 兜底。
    // 成本（只对"非全透"的玻璃像素付，且有下面的早退）：**+1 条阴影射线 + 4 次导引读 + 1 次 GI 采样**
    // （≈ 每玻璃像素一条射线；MT4 实测全屏一条射线 = 1.85 ms ⇒ 覆盖率 5% ≈ 0.09 ms）。
    // **早退**：`through ≈ 1`（透明度 100% = 全透）时整段跳过。
    let f_alpha = vec3<f32>(1.0) - through;
    if (max(max(f_alpha.x, f_alpha.y), f_alpha.z) > 1.0 / 255.0) {
      let fm = fetch_material(first.palette_base, first.pal, first.point, fnn, PBR_DEBUG_ASSET);
      // ---- 阴影（与不透明路径同一个构造：同一条 `sun_transmittance`、同一个界面点 `f_face`）----
      let f_ndl = max(dot(fnn, sun_dir), 0.0);
      var f_sun = 0.0;
      if (f_ndl > 0.0) {
        f_sun = sun_transmittance(f_face, sun_dir, first.voxel, first.obj_id);
      }
      // ---- 环境 + GI（与不透明路径逐字段同构：同一个 `gi_upsample_joint`、同一套键、同一个 cov 地板）
      // 键必须取**玻璃那个体素/面**的（不是 `best` 的）：GI 缓冲在本像素上存的就是"主射线命中的那个面"
      // 的 GI，而主射线命中的正是这个玻璃面（`gi/screen.wesl` MT5-1：介质像素照常产出 GI）⇒ 键对得上。
      let fk0 = gi_key0(first.voxel, first.obj_id);
      let fk1 = gi_key1(first.voxel, first.face_id, first.obj_id);
      var f_gi = vec3<f32>(0.0);
      var f_cov = 0.0;
      if (gi_in.use_face) {
        f_gi = gi_in.face_avg;
        f_cov = gi_in.face_cov;
      } else if (gi_u.misc.x > 0.5) {
        let fs = textureSampleLevel(gi_tex, light_samp, gi_in.uv, 0.0);
        let fgw = max(fs.a, 1e-4);
        let fup = gi_upsample_joint(gi_in.uv, vec2<u32>(textureDimensions(gi_tex)), fk0, fk1, fnn,
                                    fs.rgb / fgw);
        f_gi = fup.rgb;
        f_cov = max(fs.a / fgw, fup.a);
      }
      // WAL：不在这里混 —— 面表的值已在**写侧**注入 `gi_out`（见上面的说明段），这里读到的
      // `gi_tex` 已经含它（且已过 atrous）。
      let f_amb = sky * GI_SKY_AMBIENT * mix(1.0, GI_AMBIENT_FLOOR, f_cov);
      glass_col = brdf_reflected(fm, fnn, fv,
                                 SurfaceLight(sun_dir, sun_c, f_sun, f_amb, f_gi))
                + emissive_radiance(fm, GI_EMIT_GAIN);
    }
  }
  // ---- 天空背景（**只在这里用 `sky_primary`**）：常量天光 + 天体盘 + 光晕 ----
  // `through` = 沿途介质的乘性吸收（玻璃染色），`refl` = 玻璃那条状态机累积的反射项。
  // ⚠️ 除了这一行，全工程的天空口径仍是 `sky_rgb()`（GI 的 miss / 玻璃与镜面反射的逃逸）——
  // 太阳作为**光源**已经由各处的 NEE 负责，那些地方再画一次太阳盘就是重复计能量（见 `sky_primary`）。
  var col = vec3<f32>(0.0);
  if (best.hit) {
    // 材质参数（albedo / roughness / metallic / emissive / transmission / ior / specular）由下面那次
    // `fetch_material` 一次拿全 —— 这里不再单独取 albedo（避免出现"第二处取值口径"）。
    let gg = make_grid(u32(best.obj_id) + 1u);
    // 命中体素的世界中心：阴影与反射射线的起点 / **视向量的锚点**（`face_view_dir`）
    // 都取它（同一个助手给出 ⇒ 不会出现"两处各自算一遍中心"的口径分叉）。
    let p_voxel = voxel_center_world(gg, best.voxel);
    // ---- 基础开关「隐式法相」清零 ⇒ **原色直出**（菜单「渲染/基础」）----
    // 不算邻域梯度法线（省 `voxel_normal` 的 4 次点查询）；材质取值的 triplanar 权重需要**一个**法线
    // ⇒ 用 DDA 权威的可见面法线 `best.n`（不查邻域）。取到材质就直接返回 `albedo`：
    // 后面的 GI / 太阳 / 阴影射线 / 反射链 / 玻璃 in-scatter 全部不算 —— 这是"看材质本色"的
    // 调试视图，也是全链最省的一档。**天空（未命中）不走这里**：那上面没有材质可直出。
    if ((light_u.g.base_flags & BASE_FLAG_IMPLICIT_NORMAL) == 0u) {
      return fetch_material(best.palette_base, best.pal, best.point, best.n, PBR_DEBUG_ASSET).albedo;
    }
    let k0 = gi_key0(best.voxel, best.obj_id);
    let k1 = gi_key1(best.voxel, best.face_id, best.obj_id);
    // 着色法线（`voxel_normal`，主世界要探 4 个 / 物体 6 个邻域体素）：**先试导引复用，键相等就用它、不再自己算**。
    // 复用合法性由键给出：键逐位编码了精确体素 + 面号 + 物体，而 `voxel_normal` 是
    // `(命中体素, 可见面)` 的纯函数 ⇒ 键逐位相等 ⇒ 法线必然逐位相等。
    // 只在 gi_div = 1 时尝试：此时 GI 射线与本像素**逐位相同**，导引那格必然命中同一体素面，
    // 于是这里的复用替代掉整条法线。gi_div > 1 时导引那格来自另一条射线（更粗的网格、射向格中心），
    // 它命中的**体素**在多数像素上都与本像素不同（1/4 档实测复用率 ≈ 0，主 pass 时间无变化）
    // ⇒ 不做尝试，连那次必然失败的键读取都省掉。
    var n = best.n;
    var n_reused = false;
    if (n_hint_ok) {
      // 调用方给的**同键**着色法线（`dda_face_main` 从降噪导引里取，见函数头）⇒ 直接用，不重算。
      n = n_hint;
      n_reused = true;
    } else if (u32(gi_u.flags.y) == 1u) {
      let gi_size = vec2<u32>(textureDimensions(gi_tex));
      // texel 下标由采样 uv 反推（全分辨率档下 `gi_in.uv` 恰是本像素中心 ⇒ 逐位等于 `gid`）。
      let tp = vec2<i32>(floor(gi_in.uv * vec2<f32>(f32(gi_size.x), f32(gi_size.y))));
      if (tp.x >= 0 && tp.y >= 0 && tp.x < i32(gi_size.x) && tp.y < i32(gi_size.y)) {
        let gb = (u32(tp.y) * gi_size.x + u32(tp.x)) * GI_DEN_GUIDE_WORDS;
        if (gi_guide_out[gb + GI_DEN_G_KEY0] == k0 && gi_guide_out[gb + GI_DEN_G_KEY1] == k1) {
          n = gi_unpack_dir(gi_guide_out[gb + GI_DEN_G_NRM]);
          n_reused = true;
        }
      }
    }
    if (!n_reused) { n = voxel_normal(gg, best.voxel, best.face_id, best.n); }
    // ---- MT3-2 · 材质取值（**唯一入口**）----
    // `PBR_DEBUG_ASSET` 是**编译期常量** ⇒ 关闭（`0xFFFFFFFF`）时 `fetch_material` 里那条调试分支
    // 被折掉，正常路径零开销（所以不需要运行时开关 / uniform 字段）。位置必须在**着色法线 `n`
    // 解析之后**：triplanar 的三轴权重用 `n`、UV 用**命中点的世界坐标** `best.point`（连续点）
    // —— MT8-1 起在 `triplanar_sample` 内部量化到体素中心（`MATERIAL_FLAT_SHADING`），
    // 这里传连续点即可（`fetch_material` 的入参契约不变，仍是"命中点的世界位置"）。
    // 平凡变体（`IS_PBR = 0`）与 PBR 变体在这里产出**同一个** `MaterialParams`
    // ⇒ 下面的着色代码对两者完全一致（D1 约束 1 / MT3-7）。
    let m = fetch_material(best.palette_base, best.pal, best.point, n, PBR_DEBUG_ASSET);
    // ---- 视向量（**唯一构造** `face_view_dir`）：**按体素面** ----
    // 起点 = 相机眼位 `origin_voxel`，锚点 = 命中体素中心 `p_voxel` ⇒ 同一个体素面内所有像素同值。
    // 为什么按面、代价是什么、历史两次实测结论：见 `common.wesl::face_view_dir`（**改这里前必读**）。
    // 它只喂反光/高光三项（GGX 高光、`mirror`、反射权重）—— 本工程没有别的地方读 `v`。
    let v = face_view_dir(origin_voxel, p_voxel);
    // 沿 n 外推半个体素 + SHADOW_SURFACE_EPS，使起点离开命中体素（半个体素按该 grid 的体素算）。
    let n_off = surface_offset(gg, n, SHADOW_SURFACE_EPS);
    let ndl = max(dot(n, sun_dir), 0.0);
    var sun = 0.0;
    if (sun_hint >= 0.0) {
      // **借用**（覆盖 texel 回退路径）：采用被借面已算好的太阳可见性，**不发射线** ——
      // 阴影射线是逐像素内联的最大单项（实测 5 ms 量级），而回退存在的价值就是省掉它。
      // 与本像素自洽：准入保证同平面 / 同逐位着色法线（⇒ 同一个 `ndl`）/ 同材质。
      sun = sun_hint;
    } else if (ndl > 0.0) {
      // 「基础/直光阴影」清零 ⇒ **不发射阴影射线**（省每像素 1 条 DDA），`sun = 1` = 太阳照旧直射、
      // 只是没有明暗遮挡。**必须用真分支**：`select` 会把两侧都算了（射线照发，白花钱）。
      sun = 1.0;
      if ((light_u.g.base_flags & BASE_FLAG_SHADOW) != 0u) {
        sun = sun_transmittance(p_voxel + n * n_off, sun_dir, best.voxel, best.obj_id);
      }
    }
    // 出口 = 本次实际用到的值（`dda_face_main` 烘焙它、回退路径借用它；见函数头）。
    *sun_io = sun;
    let gi_on = gi_u.misc.x > 0.5;
    var gi = vec3<f32>(0.0);
    var cov = 0.0;
    // GI 一律走缓冲通道（`gi_main` + 降噪写在 1/gi_div 分辨率的 GI 网格上，见 `gi/`）。两条路径
    // （见 `GiInput`）：认领面用**逐面均值**（无偏降噪 + 面内平色）、未认领面按 `gi_in.uv` 逐像素
    // 做一次几何感知上采样（内部 2×2 texel）。`gi_on = false` 时两者都留 0 ⇒ 只有太阳直射 + 天光兜底。
    if (gi_in.use_face) {
      gi = gi_in.face_avg;
      cov = gi_in.face_cov;
    } else if (gi_on) {
      let s = textureSampleLevel(gi_tex, light_samp, gi_in.uv, 0.0);
      // premultiplied-valid 归一化：valid = 0（天空）像素不参与插值。
      let gw = max(s.a, 1e-4);
      // 覆盖度 cov 直接取自 `a`：`gi_out.a = valid`，而独立的 cov 通道写的恒是「valid ? 1 : 0」
      // ⇒ `cov = valid / max(valid, 1e-4)` 与旧的 `sc.r / gw` 是同一个表达式（不再单开一张纹理）。
      // GI 值走**几何感知上采样**（纯双线性作 fallback）：不接受与采样点不同「面」的 texel
      // ⇒ 棱边 / 墙角不渗色、不糊（相似度那一条用采样点的着色法线 `n`）。cov 不参与
      // （它是 0/1 覆盖度，双线性归一化即可）。
      let up = gi_upsample_joint(gi_in.uv, vec2<u32>(textureDimensions(gi_tex)), k0, k1, n,
                                 s.rgb / gw);
      gi = up.rgb;
      // `up.a` = 邻域搜索补到的数据（见 `gi_upsample_joint` 的退化顺序）⇒ 那些像素不再是"无数据"。
      cov = max(s.a / gw, up.a);
    }
    // WAL：不在这里混 —— 面表的值已在**写侧**注入 `gi_out`（见上面的说明段），这里读到的
    // `gi_tex` 已经含它且已过 `gi_face_flatten` / 时域 / atrous。
    // cov 由缓存给出（命中 = 1、没有数据 = 0）：有数据处天光只占少量、无数据处天光兜底。
    // mix 实参顺序必须是 `mix(1.0, GI_AMBIENT_FLOOR, cov)`（cov = 1 → FLOOR）。
    let amb = sky * GI_SKY_AMBIENT * mix(1.0, GI_AMBIENT_FLOOR, cov);
    // ---- MT3-3 / MT3-7 · 着色（**唯一 BRDF 求值函数**，平凡与 PBR 共用）----
    // 与 `gi/ray.wesl` 的二次顶点逐项对齐（硬约束 3 / `docs/PLAN.md` MT3-4）：两侧都只是把各自的
    // 光源项交给同一个 `brdf_reflected`。差异只有三处，且都是物理上该不同的：
    //   ① 本处有 GI 项（`gi`）；二次顶点只算一次弹射 ⇒ `SurfaceLight.gi` 恒 0；
    //   ② 本处的天光含 `cov` 地板；二次顶点没有 GI 缓冲 ⇒ 天光全额。
    //   ③ 本处不除 π（二次顶点要除 π 转辐亮度）—— 能量约定见 `brdf_reflected` 的文档注释。
    // 视向量 `v`（上方已构造，**逐面**）在这里被消费：与 `gi/ray.wesl` 的二次顶点**同一个助手、
    // 同一口径**（硬约束 3：两边都走 `face_view_dir`，不分叉）。
    col = brdf_reflected(m, n, v, SurfaceLight(sun_dir, sun_c, sun, amb, gi))
        + emissive_radiance(m, GI_EMIT_GAIN);
    // ---- MT4-1 · 环境镜面反射（**本里程碑唯一的新增射线**）----
    // 设计（`docs/PLAN.md` §4 MT4-1）：该表面的镜面反射"值得追"时**追加一条**反射射线，
    // 把反射点的环境光照当作**环境/间接镜面项**加到 `col` 上。触发判据与成本控制见顶部
    // `PBR_REFL_*` 三个常量（**不是无条件每像素发射线**）。
    //
    // ---- MT4-2 · 边界：反射样本**不进 ReSTIR**（只在本函数里加到 `col`）----
    // ReSTIR 的样本等价判据是「**精确几何面键 + 着色法线相似**」（硬约束 2 / `gi/consts.wesl`
    // 文件头），这是薄板/墙缝不漏光的根本。反射方向是**按 F0 与 roughness 加权抽出来的**一条方向、
    // 且抖动锥与 MT3 的余弦候选完全不是同一族分布 ⇒ 它既不描述"同一条光路"，也不满足"同一个面上、
    // 同一族源 pdf 的样本"。
    // 一旦把它并进 reservoir / 导引 / 历史，判据就会把"方向不同的两个样本"判成等价 —— 那正是
    // §5 R1 点名的失效形态：薄板漏光、屏幕漂移的光斑。
    // 所以本射线**不写任何 GI 资源**：不进 `gi_out`、不进 `gi_res_cur/prev`、不进导引与历史；
    // 结果只出现在 `col` 上，且 `col * through + refl` 那一步之后也就离开本 pass（GI 链读的是它
    // 自己的缓冲，读不到这里）。
    // 玻璃路径下这一项与其余着色一样被 `through`（沿途介质吸收）乘一次 —— 反射光同样要"穿过"
    // 那段玻璃回到观察者，与 `col` 其余部分同一条口径（不额外单开一条乘性吸收）。
    // 自检：`PBR_REFLECTION_ENABLED = 0u` 重启 ⇒ 本段被整段折掉，GI 画面与 MT3 逐位一致。
    // ---- MT4 形态：**每个触发像素各发一条反射射线**（MT8-3 的缓存已整体删除，历史记录见文件头）----
    // 这里曾经是"读上一帧缓存 → 命中且不刷新就复用 → 否则发线并着色 → 写回"的乒乓缓存三段式
    // （面键/槽哈希 + 每帧旋转刷新 K=16）。**用户实测判为负优化**（删掉后逐像素发线的成本
    // ≈1.5~1.6ms，收益对不上复杂度）⇒ 键/槽/刷新/写回全部删除，本段回到 MT4 的逐像素形态。
    // ---- MT4-4 · **运行期档位**（`light_u.g.refl_tier`，菜单「渲染/反射」，0 = 关 ..= 3）----
    // 档位只改**反射内容的着色口径**（命中点要不要太阳直射/自发光、逃逸天空走哪个口径、要不要补
    // 阴影射线），不改逐面量化的粒度（见下面 MT8-6 段）。
    // 与编译期总开关 `PBR_REFLECTION_ENABLED` 的分工：那条是 A/B 自检用的（关掉整段被折掉、
    // 零寄存器压力），本条是运行期旋钮 ⇒ 只要总开关打开，**档位为 0 时代码仍然在**（这是菜单开关的
    // 固有代价，登记在 `PBR_REFLECTION_ENABLED` 的成本拆账里：代码在、判据不成立 ≈ +0.25ms）。
    if (PBR_REFLECTION_ENABLED != 0u && light_u.g.refl_tier > 0u) {
      let tier = light_u.g.refl_tier;
      let f0r = f0_of(m);
      let f0_luma = dot(f0r, GI_LUMA);
      if (f0_luma >= PBR_REFL_MIN_F0 && m.roughness <= PBR_REFL_MAX_ROUGH) {
        // `refl_env`（反射内容）在下面射线落地处初始化：它 = 反射命中点的出射亮度，或"逃逸到天空"
        // 时取的那份天空。
        // ---- MT8-6 · 抖动种子**按体素面**取（键 = grid 局部体素坐标 + 面号 + obj_id）----
        // 一个体素面 = 一条反射方向 = 一个反射值（与材质色同格）；按像素取时同面内每个像素各采锥内
        // 一个不同样本、锥角（≈5.6°）把方向抹开 ⇒ 树影/屋顶边缘变成连续模糊，且**没有时域累积**时
        // 噪声永不收敛（颗粒感）。详细的三条实测结论在文件头的 MT8-6 段，种子的构造在 `refl_seed`。
        // ⚠️ **这不是缓存**：下面照样**每个触发像素各发一条射线**（成本与改前完全一致）。
        // MT8-3 的反射缓存已按实测删除、**不要复活**。
        // 种子**不含帧号**：这条路径不做时域累积 ⇒ 混入帧号只会让每个面逐帧换方向 = 静止画面上的闪烁。
        // 反射内容（含 MT4-5 的嵌套链）由 `refl_env_chain` 算，种子由 `refl_seed` 按面给出。
        let refl_env = refl_env_chain(p_voxel, n, v,
                                      refl_seed(best.voxel, best.face_id, best.obj_id),
                                      m, frustum_length, n_off);
        // 权重 = Schlick Fresnel（同一个 `fresnel_schlick`：完美镜面下 h ≈ n ⇒ 余弦取 `v·n`）
        //        × 粗糙度淡出（`roughness → PBR_REFL_MAX_ROUGH` 时权重**连续**归零 ⇒ 粗糙度扫描下
        //        总亮度单调、阈值处不出现亮度台阶；这也是 MT4-3 的能量自检要看的量）。
        // 人工自检方式（MT4-3）：用「游戏/编辑」的笔刷在同一面墙上涂出几块不同粗糙度的色块
        // （「光滑度」滑杆 100% / 90% / 75% / 50% / 40% ⇒ roughness 0 / 0.098 / 0.255 / 0.5 / 0.6），
        // 亮度应随 roughness **单调不增**、且在 50%（= 阈值）处**连续**（不出现台阶），开/关反射
        // 各看一次即可确认没有"高光算两遍"的能量翻倍。
        // ---- MT4-3 · **替换式**能量口径（不是叠加）----
        // `brdf_reflected` 的环境项是 `(kD·albedo + F0)·(amb + gi)`，其中 `F0·(amb+gi)` 就是
        // "环境为常数"时**同一个镜面量**的粗糙近似。真追到的 `refl_env` 是反射点的**出射辐亮度**
        // ⇒ 两者量纲口径不同，必须**替换**：按 `fade` 扣掉那项近似、换成真值。
        // 保留加法会让镜面环境项算两遍（电介质 F0 = 0.04 ⇒ 天光/GI 约 +4%；金属最坏 2× ⇒ 镜面发白）。
        // `fade` 同时乘在"扣"与"加"两侧 ⇒ 粗糙度扫到阈值处**连续**收敛回近似口径，总亮度单调、
        // 不出现台阶（电介质那 4% 的差也在同一个 `fade` 下一起归零）。
        // 权重**不进任何缓存**（MT8-3 的缓存已删除）：它只由本像素的 `m`（材质）与 `v·n` 决定。
        let fade = clamp(1.0 - m.roughness / PBR_REFL_MAX_ROUGH, 0.0, 1.0);
        // 镜面反射的菲涅耳余弦取 `v·n`（**逐面**视向量）—— 与 `brdf_reflected` 里
        // "完美镜面 h ≈ n ⇒ 余弦取 v·n" 那条口径**逐字相同**（硬约束 3）。
        // 逐面 ⇒ 掠射角的权重从"跨体素边界连续渐亮"变成一个面一个值（成块变亮是本档的预期代价）。
        col = col - fade * f0r * (amb + gi)
            + fresnel_schlick(f0r, max(dot(v, n), 0.0)) * fade * refl_env;
      }
    }
    col = col * through + refl;
  } else {
    // 天空像素：常量天光 + 天体盘 + 光晕。`glass_col` 此时恒 0（无命中即无玻璃首命中）
    // ⇒ 省掉末尾那次 in-scatter 混入（那一项是 `+ 0`）。
    return sky_primary(ray_dir, sun_dir, sun_c) * through + refl;
  }
  // ---- MT8-9 · 透明度的 in-scatter 混入（`col` 此刻 = 背景×through + refl）----
  // 「透明度」= **混合比**（标准 alpha 合成，`through` 是逐段 alpha，与厚度无关见 `trace.wesl::MediumRay`）：
  //   透明度 100%（`through = 1`）⇒ 纯背景（上方已早退、不花额外成本）；
  //   中间档 ⇒ 背景与"这块玻璃**按不透明材质着色**的样子"按 `(1 − through)` 线性混合 ⇒ **单调、不发黑**；
  //   透明度 0%（`transmission = 0`）⇒ 不是介质 ⇒ 走上面的不透明分支；因为 `glass_col` 与那条分支
  //     用的是**同一套着色** ⇒ 那一档**连续收敛**。
  // `glass_col` 只有首命中是介质时才非 0 ⇒ 非玻璃像素这一行是 `+ 0`（零成本）。
  col = col + glass_col * (vec3<f32>(1.0) - through);
  return col;
}

/// **「覆盖 texel 回退」总开关（A/B 自检用）**：`1u` = 开（仓库默认）、`0u` = 关（回到逐像素内联口径）。
///
/// WHY 需要它：这条回退改变的是**远场小面**（一个可见体素面 < 一个 GI texel）的颜色来源 ——
/// `0u` 时那些像素各自内联整条 `shade_face`（含阴影射线 + 材质 + 法线 + GI 采样，实测这是主 pass
/// 的最大单项）；`1u` 时用「该像素所在 GI texel 命中的那个面」的**逐面量**把这条重复计算接回去。
/// 近中景（面比 texel 大的那些像素）走的是精确键，**一个像素都不受影响**。
///
/// ⚠️ **`1u` 的借用面只保留"逐面射线量"**：太阳可见性借被借面的（`FACE_W_SUN`，见 `shade_face` 的
/// `sun_io`），材质 / 着色法线 / 视角 / GI / 自发光 / 反射全部**逐像素**重算。
/// **为什么不能再借整条颜色**：借整色 = 这一格（1/4 档 = 4×4 屏幕像素）的像素整块显示"那个面"的
/// 材质色，而那是**另一个体素**的面（逐体素纹理量化 ⇒ 相邻面本来就不同色）⇒ 远场整片成为
/// **屏幕对齐的小方块**、与体素格无关（2026-10-03 用户报的"GI 没有逐体素着色"）。
/// ⚠️ **玻璃（介质）也没有例外**：曾经给它留过"整色借用"（理由是 `trace_glass` 的逐面光路状态机
/// 逐像素重算更贵），结果是**水面**在掠射角下整片变成屏幕对齐的方块、倒影/桥影被抹平（同日第二次
/// 用户实测）。现在一律逐像素重算 —— 代价与"GI 关掉"时同一口径（见 `dda_main` 的回退段）。
/// 关掉它（`0u`）即回到"只有精确键命中才用槽色、其余全部内联"的行为（逐位一致）。
const GI_FACE_FALLBACK: u32 = 1u;

/// **逐面查表覆盖计数的槽位**（`lod_diag` 的 3..9；`trace.wesl::DIAG_*` 的 0..2 是叶级 LOD 的）。
/// 六个槽互斥且穷尽（GI 开 + 回退开时）⇒ 一次读数就给出"每个像素从哪条路拿到颜色"。
/// 只在 `LOD_DIAG = 1`（`trace.wesl`）时被写：置 0 时调用点整段被折掉，正常构建零开销。
/// 实测（2026-09-28，MC 城市 / 1/2 档 / 静止机位）：天空 32% / 精确 0% / 回退 40% /
/// 回退无键 9.5% / 回退槽失配 4.6% / 内联 14% —— 就是这组数字定下了"认领粒度实际是 GI texel"与
/// "内联的唯一来源是回退失败"两条结论（见 `docs/editable-gigavoxel.md` §10 第五轮）。
const DIAG_FACE_EXACT: u32 = 3u;    // 精确键命中
const DIAG_FACE_FB_HIT: u32 = 4u;   // 覆盖 texel 回退命中（逐像素重着色 + 借太阳可见性）
const DIAG_FACE_FB_NOKEY: u32 = 5u; // 回退不可用：无主命中 / 异平面 / 异着色法线 / 异材质
const DIAG_FACE_FB_MISS: u32 = 6u;  // 回退失败：键有、槽没认领 / 键不符（撞键）
const DIAG_FACE_INLINE: u32 = 7u;   // 命中像素最终走内联着色
const DIAG_FACE_SKY: u32 = 8u;      // 天空像素（`first.hit == false`）
// 主射线的**像素分类**（槽位 9..13，见 `consts::LOD_DIAG_WORDS` 的说明）：同一个采样闸门
// （1/64 像素）⇒ 四项互斥穷尽、和 = 采样总数。用 `obj_id` 分近场/远场（`make_grid` 约定：
// 主世界 -1、物体/远场 `idx-1 >= 0`）。
const DIAG_MAIN_RAYS: u32 = 9u;     // 采样到的主射线总数（分母）
const DIAG_MAIN_NEAR: u32 = 10u;    // 命中主世界（近场）⇒ 代价在 chunk 内下钻
const DIAG_MAIN_FAR: u32 = 11u;     // 命中远场级 / 物体 ⇒ 代价在远场遍历
const DIAG_MAIN_SKY: u32 = 12u;     // 未命中（天空）⇒ 代价在空区步进
// 世界累积层（WAL）的**接管比例**（槽位 23..26）：分母 = 槽位 25（本帧**被逐面着色**的面数，
// `dda_face_main` 每个面记一次、1/64 票）—— 不能用屏幕像素做分母：显示链的像素要么取某个面的
// **烘焙色**（精确键：GI 已按那个面取 WAL），要么按**本像素自己的面**逐像素着色（覆盖回退 / 内联：
// 读到的是该面位置上的 GI 场，WAL 已在写侧按面注入）⇒ 两种情况都是"面级"的量 ⇒ 按面统计才是
// 真实接管比例。
// 判定 = **同面邻面池化后**的权重（成熟 = 池子样本数 Σn ≥ `GI_WAL_MATURE_N`、未熟 = 按比例渐入）。
// 槽位 26 = 采样面的 `Σn` 之和 ⇒ 平均 Σn 是"池子有多热"的量具：正常应接近
// `GI_WAL_CAP_N × 池子面数`（几千）；若长期只有几百 ⇒ 条目被区域失效反复打回短窗。
const DIAG_GI_WAL_MATURE: u32 = 23u;
const DIAG_GI_WAL_FRESH: u32 = 24u;
const DIAG_WAL_FACES: u32 = 25u;
const DIAG_WAL_NSUM: u32 = 26u;

/// **本像素所在那个 GI texel 的主命中面键 + 该 texel 的导引字址**（返回值 = 该 texel 有主命中）。
/// 下标口径与 `shade_face` 里 `gi_div = 1` 的法线复用**逐字相同**（`floor(uv · gi_size)`，
/// `gi_guide_out` 的布局见 `gi/consts.wesl`：键非零 = 有效，**没有**独立有效位）。
/// 它只回答"那个 texel 有主命中"，**不回答"命中的是同一个面"** —— 调用方必须再判
/// 同平面 / 同着色法线 / 同材质（见 `dda_main` 的覆盖回退段）。
/// `guide_off_out` = 该 texel 在 `gi_guide_out` 里的字址（取 `GI_DEN_G_NRM` 那格用它）。
fn gi_cover_key(uv: vec2<f32>, key_out: ptr<function, vec2<u32>>,
                guide_off_out: ptr<function, u32>) -> bool {
  let gi_size = vec2<u32>(textureDimensions(gi_tex));
  let tp = vec2<i32>(floor(uv * vec2<f32>(f32(gi_size.x), f32(gi_size.y))));
  *guide_off_out = 0u;
  if (tp.x < 0 || tp.y < 0 || tp.x >= i32(gi_size.x) || tp.y >= i32(gi_size.y)) { return false; }
  let gb = (u32(tp.y) * gi_size.x + u32(tp.x)) * GI_DEN_GUIDE_WORDS;
  *guide_off_out = gb;
  *key_out = vec2<u32>(gi_guide_out[gb + GI_DEN_G_KEY0], gi_guide_out[gb + GI_DEN_G_KEY1]);
  return (*key_out).x != 0u || (*key_out).y != 0u;
}

// DDA 主入口：命中 → **逐面去重查表**（命中即直出）；未认领 / 撞键 / miss → 内联 `shade_face`。
@compute @workgroup_size(8, 8, 1)
fn dda_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let size = textureDimensions(out_tex);
  if (gid.x >= size.x || gid.y >= size.y) { return; }
  let coord0 = vec2<i32>(i32(gid.x), i32(gid.y));

  // WHY: 相机陷在实心体素内部时（未做相机碰撞检测）主射线在 t≈0 命中相机自己所在的体素，
  // 整屏像素共享同一个面键 ⇒ 画面是几块平色，而且整条 GI / 玻璃状态机 / 阴影仍在按这个假面运行。
  // 该状态没有"正确的着色"可言 ⇒ 整屏直接输出该体素的本色并早退，其余产物（含光柱）一律不消费。
  // 输出经 `write_output`（曝光 + sRGB 编码）⇒ 与其余路径同一收尾。
  let cam_g = make_grid(0u);
  let cam_pal = camera_solid_pal(view_u.cam_pos_voxel.xyz);
  if (cam_pal != 0u) {
    write_output(coord0, palette_rgb_linear(cam_g.palette_base, cam_pal));
    return;
  }

  let px = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5)) / vec2<f32>(f32(size.x), f32(size.y));
  let uv = vec2<f32>(px.x * 2.0 - 1.0, 1.0 - px.y * 2.0);
  let near_ndc = vec4<f32>(uv.x, uv.y, 0.0, 1.0);
  let far_ndc  = vec4<f32>(uv.x, uv.y, 1.0, 1.0);
  let near_world_h = view_u.inv_view_proj * near_ndc;
  let far_world_h  = view_u.inv_view_proj * far_ndc;
  let near_world = near_world_h.xyz / near_world_h.w;
  let far_world  = far_world_h.xyz  / far_world_h.w;
  let diff_world = far_world - near_world;
  let frustum_length = length(diff_world);
  let dir_voxel = normalize(diff_world);
  let origin_voxel = view_u.cam_pos_voxel.xyz;

  // lod.z > 0.5（`GATE_NO_BEAM=1`）关闭 beam，t_min = 0。
  var t_min = 0.0;
  if (view_u.lod.z < 0.5) {
    let beam_size = textureDimensions(beam_depth);
    let bx = gid.x / BEAM_DIV;
    let by = gid.y / BEAM_DIV;
    t_min = 1e+30;
    for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
      for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
        let sx = clamp(i32(bx) + dx, 0, i32(beam_size.x) - 1);
        let sy = clamp(i32(by) + dy, 0, i32(beam_size.y) - 1);
        let bt = textureLoad(beam_depth, vec2<i32>(sx, sy)).x;
        t_min = min(t_min, bt);
      }
    }
    t_min = max(t_min - BEAM_BACKOFF, 0.0);
  }
  let sun_dir = light_u.lights[0].kind_pos_dir.yzw;
  let sky = light_u.sky_color.xyz;
  let sun_c = light_u.lights[0].color_intensity.xyz * light_u.lights[0].color_intensity.w;
  let first = world_raycast(make_grid(0u), origin_voxel, dir_voxel, frustum_length,
                            world_cfg_primary(t_min, 3u));

  // ---- 逐面去重查表（`face_slots`：认领在 `gi/screen.wesl`、计算在 `dda_face_main`）----
  // 两级，准入都是「**同一个面**」，但两级拿到的**东西**不同：
  //   ① **精确键** = 本像素自己的可见面（体素 + 面号 + 物体）**逐位相等**（认领者算的就是它）⇒
  //      直接取它的**整条烘焙色**（材质 / 阴影 / GI / 自发光全是这个面的）。
  //   ② **覆盖 texel 回退**（`GI_FACE_FALLBACK`）= 本像素**所在那个 GI texel** 命中的面，
  //      准入 = **同平面 + 同（逐位）着色法线 + 同材质**（见下面回退段）。它**只借该面的太阳可见性**
  //      （`FACE_W_SUN`）、**其余全部逐像素重算** —— 见 `GI_FACE_FALLBACK` 的 ⚠️ 段。
  // 两级都查不到（未认领 / 撞键 / GI 关 / 键越界 / 覆盖 texel 异面或异法线或异材质）⇒ 内联着色。
  //
  // WHY ②必须存在：认领发生在 **GI 分辨率**（`gi_main` 每个 texel 抢自己命中的那个面），而消费者是
  // **全分辨率**像素。一个面上屏幕 ≤ 一个 texel（远场小面 / 细杆 / 薄板边缘）时，它只可能被"包含它的
  // 那个 texel"认领，而它自己的**体素**在 1/2 档下与相邻 2×2 像素里的邻居不是同一个键 ⇒ 精确键对不上
  // ⇒ 从前那些像素**每个都内联整条 `shade_face`**（含阴影射线 + 材质 + 法线 + GI 采样），实测这是
  // 主 pass 的最大单项。回退借"覆盖 texel 的面"的**逐面射线量**（太阳可见性，逐面只算一次）接住
  // 它们，其余仍逐像素 ⇒ 省掉的是**每条候选都发一条阴影射线**，而不是把整条着色也借走。
  //
  // ⚠️ 回退借的是**另一个体素**的太阳可见性（回退唯一的借用量）⇒ "借得对"必须显式判足三条 ——
  // 三条说的是同一件事：**借来的阴影必须与本像素自洽**（同平面 ⇒ 同一个朝向；同着色法线 ⇒
  // 同一个 `ndl`、albedo 的三轴权重也一致；同材质 ⇒ 同一条 BRDF）。少一条就出
  // **零散的单体素面大小的明暗点**（实测根因：`隐式法相` 清零后斑点消失 ⇒ 载体是**着色法线**）：
  //   · **同平面**（`gi_den_same_plane`）：挡住跨朝向（台阶顶面 / 薄板背面 / 玻璃后面的实心墙）；
  //   · **同着色法线**（导引里的 `GI_DEN_G_NRM` 与本像素 `gi_pack_dir(voxel_normal)` **逐位相等**）：
  //     `voxel_normal` 是**邻域梯度**法线，在同一平面内部也会因凸点 / 台阶而倾斜（本工程一个方块是
  //     16³ 微体素 ⇒ 这类凸点零散分布在**体素尺度**上）；它经 `triplanar_sample` 的三轴混合权重
  //     直接改变 albedo，又决定 `ndl` 与反光 ⇒ 缓存色里烘的那一份必须与本像素自己的一致。
  //     判"逐位相等"而不是 `gi_same_surface` 的 `dot ≥ 0.7`：后者对 GI 积分够用（45° 凸棱法线算
  //     同一个面，见 `gi/denoise.wesl`），但**对颜色不够** —— 45° 的三轴权重差足以让 albedo 明显变。
  //   · **同材质**（`pal` 相等）：挡住同一平面上相邻的**另一种方块**。
  //
  // ---- 本像素自己的**着色法线**：查表**之前**算一次（准入门槛 + 内联色的 `n_hint` 都要它）----
  // `隐式法相` 清零时不看它（`shade_face` 提前返回 albedo 且用 DDA 面法线取材质）⇒ 不算、也不判。
  let need_n = (light_u.g.base_flags & BASE_FLAG_IMPLICIT_NORMAL) != 0u;
  var n_own = vec3<f32>(0.0);
  var n_own_word = 0u;
  if (first.hit && need_n) {
    n_own = voxel_normal(make_grid(u32(first.obj_id) + 1u), first.voxel, first.face_id, first.n);
    n_own_word = gi_pack_dir(n_own);
  }
  // ---- 帧内逐面查表的**覆盖计数**（归因仪表；`LOD_DIAG = 0` 时整段被折掉）----
  // 采样闸门 = 像素下标 & 63（1/64），与 `trace.wesl::diag_sampled` 同一个判据（比值无偏）。
  // 六个槽位互斥且穷尽（前提：GI 开、回退开）⇒ 一次读数就给出"每个像素从哪条路拿到颜色"。
  let diag_pix = gid.x + gid.y * size.x;
  let diag_on = diag_sampled(diag_pix);
  // ---- 主射线的**像素分类**（槽位 9..13）----
  // 同一个闸门（`diag_sampled`，1/64 像素）⇒ 四项互斥且穷尽，占比按四项和归一。
  //
  // WHY 要它：四类像素的**代价完全不同**，而"该砍哪里"取决于这个混合比 ——
  //   · **近场命中**（`obj_id < 0`）= 主世界命中 ⇒ 代价在 **chunk 内下钻**（依赖 load 链）；
  //   · **远场/物体命中**（`obj_id >= 0`）= 代价在**远场级的遍历**（窗口大、占用稀疏）；
  //   · **未命中**（天空） = 代价在**空区步进**（逐 chunk 走完窗口）；
  //   · 采样总数 = 分母。
  // 不先量这个比，"优化"就只能在几类之间盲猜（前几轮的实际处境）。
  if (diag_on) {
    atomicAdd(&lod_diag[DIAG_MAIN_RAYS], 1u);
    if (!first.hit) {
      atomicAdd(&lod_diag[DIAG_MAIN_SKY], 1u);
    } else if (first.obj_id < 0) {
      atomicAdd(&lod_diag[DIAG_MAIN_NEAR], 1u);
    } else {
      atomicAdd(&lod_diag[DIAG_MAIN_FAR], 1u);
    }
  }
  var col = vec3<f32>(0.0);
  var cached = false;
  if (first.hit && gi_u.misc.x > 0.5) {
    if (gi_key_ok(first.voxel, first.obj_id)) {
      var ok = false;
      var pal_exact = 0u;
      var sun_exact = 0u;
      let c = face_slot_color(
        gi_key0(first.voxel, first.obj_id),
        gi_key1(first.voxel, first.face_id, first.obj_id),
        &pal_exact,
        &sun_exact,
        &ok);
      if (ok) {
        col = c;
        cached = true;
        if (diag_on) { atomicAdd(&lod_diag[DIAG_FACE_EXACT], 1u); }
      }
    }
    var fb_tried = false;
    var fb_unusable = false;
    if (!cached && GI_FACE_FALLBACK == 1u) {
      fb_tried = true;
      var gk = vec2<u32>(0u, 0u);
      var gb_cov = 0u;
      var ok2 = false;
      var pal2 = 0u;
      var sun2 = 0u;
      var c2 = vec3<f32>(0.0);
      // 三条准入见文件头部那段 ⚠️（同平面 / 同逐位着色法线 / 同材质）。
      if (gi_cover_key(px, &gk, &gb_cov)
          && gi_den_same_plane(gk.x, gk.y, gi_key0(first.voxel, first.obj_id),
                               gi_key1(first.voxel, first.face_id, first.obj_id))
          && (!need_n || gi_guide_out[gb_cov + GI_DEN_G_NRM] == n_own_word)) {
        c2 = face_slot_color(gk.x, gk.y, &pal2, &sun2, &ok2);
      }
      if (ok2 && pal2 == first.pal) {
        // **逐像素重着色，只借"太阳可见性"**（`FACE_W_SUN`，`dda_face_main` 烘焙）。
        // 这是"逐体素着色"的前提：材质（逐体素纹理量化）/ 着色法线 / 视角 / GI（逐面 WAL 或
        // 逐像素上采样）/ 自发光 / 反射全部按**本像素自己的面**算 —— 旧口径借整条颜色时，
        // 这一格的像素整块显示"那个面"的颜色，而那张网格是**屏幕对齐**的 ⇒ 远场 / 掠射面
        // 看起来就是一片"平行于屏幕的大正方形"、与体素格无关。
        // ⚠️ **玻璃（介质）没有例外**（2026-10-03 第八轮撤掉的例外）：水面这类大面积介质在掠射
        // 角下几乎每个像素都过不了精确键 ⇒ 旧例外把它们整片变成"邻居水面面的颜色"（同平面 /
        // 同法线 / 同材质对一张水平面恒成立）⇒ 倒影与桥影被抹成屏幕对齐的方块。
        // 逐像素跑 `trace_glass`（最多 `GLASS_MAX_BOUNCE` 次界面事件）确实更贵，但**与"GI 关掉"
        // 时同一口径** —— 那条路径本来就每个像素跑一遍 ⇒ 本项的代价上限 = 那些像素在 GI 关掉时
        // 的代价，不会比关掉 GI 更贵；不透明面仍只借阴影（逐面一次，见 `shade_face` 的 `sun_io`）。
        // （`c2` / `face_slot_color` 的色只在**精确键**那条路消费；这里只需要 `pal` 与 `sun`。）
        var sun_io = bitcast<f32>(sun2);
        col = shade_face(first, origin_voxel, dir_voxel, frustum_length,
                         GiInput(false, vec3<f32>(0.0), 0.0, px), sun_dir, sky, sun_c,
                         n_own, need_n, &sun_io);
        cached = true;
        if (diag_on) { atomicAdd(&lod_diag[DIAG_FACE_FB_HIT], 1u); }
      } else {
        fb_unusable = true;
      }
    }
    if (diag_on && !cached) {
      if (fb_tried && fb_unusable) {
        atomicAdd(&lod_diag[DIAG_FACE_FB_NOKEY], 1u);
      } else if (fb_tried) {
        atomicAdd(&lod_diag[DIAG_FACE_FB_MISS], 1u);
      }
      atomicAdd(&lod_diag[DIAG_FACE_INLINE], 1u);
    }
  } else if (diag_on) {
    atomicAdd(&lod_diag[DIAG_FACE_SKY], 1u);
  }
  if (!cached) {
    // `n_hint` = 上面已经算好的本像素自己的着色法线（`need_n` = false 时留空 ⇒ 函数内按原路取）。
    // `sun_io` 传 -1 = **不借用**：逐像素自算（发射阴影射线）——这是"完全逐像素"的那条路。
    var sun_io = -1.0;
    col = shade_face(first, origin_voxel, dir_voxel, frustum_length,
                     GiInput(false, vec3<f32>(0.0), 0.0, px), sun_dir, sky, sun_c,
                     n_own, need_n, &sun_io);
  }
  // ---- 光柱（godray，`volumetric.wesl`）：**线性 HDR 相加，必须在 `write_output` 的曝光之前** ----
  // 三步都是确定性取值（掩码 1 条主射线 → 两趟径向模糊）⇒ **没有噪声、没有滞后**；
  // 这里是纯**加法**：`col = col + 光柱`，**不衰减背景**、也不给画面加任何底色。
  // 为什么加在**查表之后**而不是 `shade_face` 里面：光柱是**逐像素**的量（沿屏幕朝太阳铺开），
  // 而 `shade_face` 的结果会被逐面去重表按**面**缓存（`dda_face_main` 写进 `FACE_W_COL` 的那份不含光柱）。
  // 加在里面就会把「一个面一个光柱值」缓存下来；加在这里则两条路径（查表命中 / 内联着色）各加一次，
  // 逐面去重的缓存内容**逐位不变**。
  // 采样用**手写双线性**（`godray_sample_bilinear`）：光柱网格是渲染分辨率 ÷ `div`，
  // 而它的内容是低频的（模糊过）⇒ 双线性足够，且不需要给本 pipeline 拉一个采样器进来。
  // 关掉光柱时 Rust 在 group(6) 绑 1×1 零纹理 ⇒ 这一行恒 `+0`。
  col = col + godray_sample_bilinear(fog_tex, vec2<f32>(textureDimensions(fog_tex)), px);
  // ---- 诊断：按卷着色（`BASE_FLAG_VOL_TINT`）----
  // 放在最后、覆盖 `col` 全部内容（含光柱）：诊断要的是"这一像素的几何来自哪一级"，
  // 掺进光照/雾反而难分辨。`first` 是主命中，带 `palette_base`（无需给 `WorldHit` 加字段）。
  if ((light_u.g.base_flags & BASE_FLAG_VOL_TINT) != 0u) {
    col = vol_tint_color(first);
  }
  write_output(coord0, col);
}

/// 工作组数组：8×8 = 64 lane 的**归并缓冲**（槽下标 + 定点值）。
const FACE_ACC_WG: u32 = 64u;
/// "本 lane 不参与"的哨兵（槽下标最大远小于它）
const FACE_ACC_NONE: u32 = 0xFFFFFFFFu;

var<workgroup> fa_slot: array<u32, FACE_ACC_WG>;
var<workgroup> fa_val: array<vec3<u32>, FACE_ACC_WG>;

/// **逐面 GI 累加 pass**（`GiInput.use_face` 的数据源）：派发规模 = GI 网格；每个有面键的 texel
/// 把自己的 GI 值（**降噪后**）定点累加到该面的槽里，并预留一个计数位。
/// 为什么不放在 `gi_ss_main` 里累加（那样连 pass 都不用加）：同一个 dispatch 内读不到别的线程刚写的
/// 键（认领是 CAS），参与集合会随线程调度变 —— 那正是要去掉的"每帧不一样"。放在这里
/// （GI 降噪之后、`dda_face_main` 之前）⇒ 参与集合 = 该面的全部 texel，逐帧稳定。
///
/// **工作组内先归并**：一个面的 texel 在屏幕上连续 ⇒ 8×8 的工作组通常只落在少数几个面上。
/// 先在**共享内存**里按槽下标分组求和，再由每组**一个** lane 发全局原子 ⇒ 全局原子次数从
/// 「每 texel 7 次」降到「每 (工作组, 面) 7 次」：整屏一个面时 `230k×7 → 3.6k×7`（≈64×），
/// 也就是**面越大收益越大** —— 逐面缓存本来要的就是这个方向。许多小面散布时退化成接近原来的
/// 次数，代价只是共享内存里一次 `O(64²)` 扫描（微秒量级）。
/// 全局侧仍用"CAS 预留计数"保证和不超过 `GI_ACC_MAX_N`，只是预留单位从 texel 变成**整组**：
/// 一组塞不进就整组不进（参与集合的细节因此与逐 texel 版略有差异，上限本身不变）。
@compute @workgroup_size(8, 8, 1)
fn dda_face_accum(@builtin(global_invocation_id) gid: vec3<u32>,
                  @builtin(local_invocation_index) lid: u32) {
  // 本 lane 的 (槽, 定点值)。**不能提前 return**：下面有 workgroupBarrier，要求统一控制流。
  var slot = FACE_ACC_NONE;
  var val = vec3<u32>(0u);
  let gsize = textureDimensions(gi_tex);
  if (gid.x < gsize.x && gid.y < gsize.y) {
    let texel = gid.y * gsize.x + gid.x;
    let gb = texel * GI_DEN_GUIDE_WORDS;
    let k0 = gi_guide_out[gb + GI_DEN_G_KEY0];
    let k1 = gi_guide_out[gb + GI_DEN_G_KEY1];
    if ((k0 | k1) != 0u) {
      var ok = false;
      // 双探针（`face_slot_find`）：认领可能落在第二格 —— 只探基槽会**静默漏掉那些面**
      // （它们就永远没有逐面均值 ⇒ `dda_face_main` 取到 `gi_n = 0` ⇒ 只剩天光兜底）。见该函数的 ⚠️。
      let s = face_slot_find(k0, k1, &ok);
      if (ok) {
        let fe = s * FACE_WORDS;
        let guv = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5))
                / vec2<f32>(f32(gsize.x), f32(gsize.y));
        let smp = textureSampleLevel(gi_tex, light_samp, guv, 0.0);
        if (smp.a > 0.0) {   // 无效（天空）texel 不参与
          // 归一化（premultiplied：rgb = gi·valid、a = valid），再定点取整（进位到最近，避免系统性偏暗）。
          let q = clamp(smp.rgb / max(smp.a, 1e-4), vec3<f32>(0.0), vec3<f32>(GI_ACC_CLAMP));
          val = vec3<u32>(round(q * GI_ACC_SCALE));
          slot = s;
        }
      }
    }
  }
  fa_slot[lid] = slot;
  fa_val[lid] = val;
  workgroupBarrier();
  if (slot == FACE_ACC_NONE) { return; }
  // **组代表** = 同槽 lane 中下标最小者：只有它发全局原子。
  for (var j = 0u; j < lid; j = j + 1u) {
    if (fa_slot[j] == slot) { return; }
  }
  // 代表把本组同槽的贡献先求和（含自己）⇒ 一次预留、三次原子加。
  var sum = vec3<u32>(0u);
  var cnt = 0u;
  for (var j = 0u; j < FACE_ACC_WG; j = j + 1u) {
    if (fa_slot[j] == slot) {
      sum = sum + fa_val[j];
      cnt = cnt + 1u;
    }
  }
  let fe = slot * FACE_WORDS;
  // 计数用 CAS **预留**：保证"加和"与"加计数"一一对应，且总量不超过 `GI_ACC_MAX_N`
  // （直接 `atomicAdd(count)` 后再判断会让并发冲过上限 ⇒ 和溢出）。
  loop {
    let c = atomicLoad(&face_slots[fe + FACE_W_GI_N]);
    if (c + cnt > GI_ACC_MAX_N) { return; }
    if (atomicCompareExchangeWeak(&face_slots[fe + FACE_W_GI_N], c, c + cnt).exchanged) { break; }
  }
  atomicAdd(&face_slots[fe + FACE_W_GI], sum.x);
  atomicAdd(&face_slots[fe + FACE_W_GI + 1u], sum.y);
  atomicAdd(&face_slots[fe + FACE_W_GI + 2u], sum.z);
}

/// **逐面着色 pass**：派发规模 = GI 网格，只有**认领者**（`FACE_W_TEXEL` == 自己的 texel 下标）干活
/// ⇒ 活跃线程数 = 被认领的面数，其余线程几条指令退出。位置：GI 降噪之后（要读降噪后的 GI）、
/// `dda_main` 之前（它要查表）。
/// 认领者 texel 的中心同时是两个东西：GI 采样位置（逐面一个采样点）与"本面在屏幕上的代表点"
/// （算射线长度，替代逐像素的 `frustum_length`）。
@compute @workgroup_size(8, 8, 1)
fn dda_face_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let gsize = textureDimensions(gi_tex);
  if (gid.x >= gsize.x || gid.y >= gsize.y) { return; }
  let texel = gid.y * gsize.x + gid.x;
  // 本 texel 的主命中面键 = 降噪导引里的键（`gi_ss_main` 已写；天空像素整格清零）。
  let gb = texel * GI_DEN_GUIDE_WORDS;
  let k0 = gi_guide_out[gb + GI_DEN_G_KEY0];
  let k1 = gi_guide_out[gb + GI_DEN_G_KEY1];
  if ((k0 | k1) == 0u) { return; }
  // 同一格里的**着色法线**（`gi_main` 用本 texel 的命中面写的 `voxel_normal`）：本 pass 要着色的面
  // 与写它的那个面是**同一个键**（体素 + 面 + 物体），而 `voxel_normal` 是那个键的纯函数
  // ⇒ 逐位相同，直接当 `shade_face` 的着色法线用，省掉每面那 4 次邻域点查询（`world_solid`）。
  let g_nrm = gi_unpack_dir(gi_guide_out[gb + GI_DEN_G_NRM]);
  // 逐面两个 pass 的 group(5) 里就有 `face_slots`（`dda_layouts`）⇒ 这里用 `face_slot_find`（读的是
  // `@group(5) @binding(8)` 那份声明）。**新入口**（`gi_face_flatten`）的 layout 里没有 group(5)，
  // 它必须用 `face_slot_find_flat`（见那个函数的说明）。
  // 双探针（与认领侧同一套判据）：基槽 → 第二格（撞键的面）。
  var ok = false;
  let slot = face_slot_find(k0, k1, &ok);
  if (!ok) { return; }
  let fe = slot * FACE_WORDS;
  if (atomicLoad(&face_slots[fe + FACE_W_TEXEL]) != texel) { return; }

  // ---- 从面键重建命中（位布局与"加回编码原点"见 `gi/common.wesl` 的 `gi_unkey_voxel`）----
  let face_id = (k1 >> 16u) & 7u;
  let obj_id = i32((k1 >> 19u) & 0x1FFFu) - 1;
  let voxel = gi_unkey_voxel(k0, k1, obj_id);
  let gg = make_grid(u32(obj_id) + 1u);
  let p_voxel = voxel_center_world(gg, voxel);
  // `point` 传**体素中心**：`fetch_material` 内部本来就量化到体素中心（UV 逐位相同）；唯一差别是
  // mip LOD 的距离输入从"连续命中点"换成"体素中心"（量级 ≤ 0.87 体素，而逐像素版本的那个距离
  // 本来就因 `floor` 在法线轴的 ±1 抖动而不定）。
  let fnw = face_normal_world(gg, face_id);
  let first = WorldHit(true, 0.0, p_voxel, atomicLoad(&face_slots[fe + FACE_W_PAL]),
                       fnw, face_id, voxel, obj_id, gg.palette_base);
  // ---- WAL 接管计数（槽位 23..26，1/64 个**面**）：分母 = 本帧被逐面着色的面数。
  // 判据与 `gi/screen.wesl` 的注入点（`gi_wal_sample` 的成熟度权重）**同一次池化**、同一个公式
  // ⇒ 读数就是"多少面从面表拿到了值、池子有多热"；开关打开却恒 0 ⇒ 掩码 / 表 / 绑定没接上。
  if (diag_sampled(texel)) {
    atomicAdd(&lod_diag[DIAG_WAL_FACES], 1u);
    if (gi_u.params.y > 0.5) {
      let wp = gi_wal_pool(voxel, obj_id, first.pal, k1);
      if (wp.w > 0.0) {
        atomicAdd(&lod_diag[DIAG_WAL_NSUM], u32(min(wp.w, 1.0e7)));
        if (wp.w >= GI_WAL_MATURE_N) {
          atomicAdd(&lod_diag[DIAG_GI_WAL_MATURE], 1u);
        } else {
          atomicAdd(&lod_diag[DIAG_GI_WAL_FRESH], 1u);
        }
      }
    }
  }
  let gi_uv = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5))
            / vec2<f32>(f32(gsize.x), f32(gsize.y));
  let ndc = vec2<f32>(gi_uv.x * 2.0 - 1.0, 1.0 - gi_uv.y * 2.0);
  let nw_h = view_u.inv_view_proj * vec4<f32>(ndc.x, ndc.y, 0.0, 1.0);
  let fw_h = view_u.inv_view_proj * vec4<f32>(ndc.x, ndc.y, 1.0, 1.0);
  let frustum_length = length(fw_h.xyz / fw_h.w - nw_h.xyz / nw_h.w);
  let sun_dir = light_u.lights[0].kind_pos_dir.yzw;
  let sky = light_u.sky_color.xyz;
  let sun_c = light_u.lights[0].color_intensity.xyz * light_u.lights[0].color_intensity.w;
  // ---- 逐面 GI 均值（`dda_face_accum` 已把该面所有 texel 的值定点累加进来）----
  // `GI_ACC_MAX_N` 之外的 texel 不参与 ⇒ 大面也稳定；`gi_n == 0`（无非天空 texel）时 cov = 0，
  // 走"无 GI 数据"的天光兜底（与逐像素路径同口径）。
  let gi_n = atomicLoad(&face_slots[fe + FACE_W_GI_N]);
  var gi_avg = vec3<f32>(0.0);
  if (gi_n > 0u) {
    let inv = 1.0 / (f32(gi_n) * GI_ACC_SCALE);
    gi_avg = vec3<f32>(
      f32(atomicLoad(&face_slots[fe + FACE_W_GI])),
      f32(atomicLoad(&face_slots[fe + FACE_W_GI + 1u])),
      f32(atomicLoad(&face_slots[fe + FACE_W_GI + 2u]))) * inv;
  }
  // **折叠**（`GI_ACC_MAX_N` 只封顶、不遗忘 ⇒ 必须在这里补上遗忘）：
  // `FACE_W_GI` / `FACE_W_GI_N` 由 `dda_face_accum` 只增不减地累加，而这个 pass 的认领者恰好是
  // 每面每帧唯一写者 ⇒ 到顶时把**和与计数同比例减半**：比值 `sum/n` 逐位不变（画面连续无跳变），
  // 而窗口变成"τ ≈ `GI_ACC_MAX_N` 个样本的指数窗"，从此**继续跟着画面走**。
  // 不做这一步的后果（实测症状）：面的值 = 它**头 1024 个样本**的平均，然后永久冻结。小面每帧只进
  // 1~2 个样本 ⇒ 要 500~1000 帧才填满，那整段时间里显示的是"被新样本缓慢稀释的累计平均"
  // ⇒ **静止时多个小片体素低频明暗涨落**，填满后再也不更新。与业界对齐的写法见 NRD 的
  // `maxAccumulatedFrameNum` / SVGF 的 `α = max(σ_c/σ_m, 1/(m+1))`：上限必须**同时改变权重**。
  if (gi_n >= GI_ACC_MAX_N) {
    for (var i = 0u; i < 3u; i = i + 1u) {
      let w = fe + FACE_W_GI + i;
      atomicStore(&face_slots[w], atomicLoad(&face_slots[w]) >> 1u);
    }
    atomicStore(&face_slots[fe + FACE_W_GI_N], gi_n >> 1u);
  }
  // ---- 小面的**跨面联合池化**（见 `GI_FACE_POOL_R` 的 WHY）----
  // `dda_face_accum` 的"面均值"在面只占 1~2 个 texel 时其实是**单样本** ⇒ 那是单 texel 的噪声，
  // 被降噪链稳定成固定格子 ⇒ 暗场里绕发光体的网格纹。这里用**全管线同一个判据**在本面代表 texel
  // 周围取共面 texel 的平均（= `gi_wal_*` 池化同一条道理，只是作用在**降噪后的场**上）。
  // CONSTRAINT: 只对小面生效 —— 大面的 `gi_n` 本身就是跨 texel 的**无偏均值**（设计要的「面内恒定」），
  // 再池化只会把它糊掉。
  if (gi_n <= GI_FACE_POOL_MIN_N) {
    var acc = vec3<f32>(0.0);
    var wsum = 0.0;
    for (var dy: i32 = -GI_FACE_POOL_R; dy <= GI_FACE_POOL_R; dy = dy + 1) {
      for (var dx: i32 = -GI_FACE_POOL_R; dx <= GI_FACE_POOL_R; dx = dx + 1) {
        let c = vec2<i32>(i32(gid.x) + dx, i32(gid.y) + dy);
        if (c.x < 0 || c.y < 0 || c.x >= i32(gsize.x) || c.y >= i32(gsize.y)) { continue; }
        let tb = (u32(c.y) * gsize.x + u32(c.x)) * GI_DEN_GUIDE_WORDS;
        let tk0 = gi_guide_out[tb + GI_DEN_G_KEY0];
        let tk1 = gi_guide_out[tb + GI_DEN_G_KEY1];
        if ((tk0 | tk1) == 0u) { continue; }
        if (!gi_same_surface(tk0, tk1, gi_guide_out[tb + GI_DEN_G_NRM], k0, k1, g_nrm)) { continue; }
        let v = textureLoad(gi_tex, c, 0);
        if (v.a <= 0.5) { continue; }
        acc = acc + v.rgb;
        wsum = wsum + 1.0;
      }
    }
    if (wsum > 0.5) { gi_avg = acc / wsum; }
  }
  // `ray_dir` 传该面的世界法线：逐面路径**只处理命中面**（天空在那里不渲染）⇒ 这个实参不会被消费，
  // 但必须是合法单位向量（`shade_face` 的注释里写了理由）。
  // `sun_io` 传 -1 = 本面自算（发射阴影射线）；出口值**烘焙进 `FACE_W_SUN`** —— `dda_main` 的
  // 覆盖 texel 回退借它跳过逐像素阴影射线（见 `GI_FACE_FALLBACK` 段）。
  var sun_io = -1.0;
  let col = shade_face(first, view_u.cam_pos_voxel.xyz, fnw, frustum_length,
                       GiInput(true, gi_avg, select(0.0, 1.0, gi_n > 0u), gi_uv),
                       sun_dir, sky, sun_c, g_nrm, true, &sun_io);
  atomicStore(&face_slots[fe + FACE_W_SUN], bitcast<u32>(sun_io));
  atomicStore(&face_slots[fe + FACE_W_COL], bitcast<u32>(col.x));
  atomicStore(&face_slots[fe + FACE_W_COL + 1u], bitcast<u32>(col.y));
  atomicStore(&face_slots[fe + FACE_W_COL + 2u], bitcast<u32>(col.z));
}

// GI（**唯一路径 = 屏幕空间逐面 ReSTIR**，见 `gi/screen.wesl`）。
// 本 pass 只负责：反投影 + beam 起点 + 主 trace（= G-Buffer），随后整条 GI 都交给 `gi_ss_main`
// （新鲜候选 → 时域复用 → 空间复用 → 着色）并写降噪导引。
// 输出：gi_out.rgb = gi·valid、gi_out.a = valid（覆盖度 cov 从 `a` 导出，见 `dda_main`）；
// 主 pass 按 valid 归一化，天空像素（valid = 0）不参与插值。
@compute @workgroup_size(8, 8, 1)
fn gi_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let size = textureDimensions(gi_out);
  if (gid.x >= size.x || gid.y >= size.y) { return; }
  let coord = vec2<i32>(i32(gid.x), i32(gid.y));

  let px = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5))
         / vec2<f32>(f32(size.x), f32(size.y));
  let uv = vec2<f32>(px.x * 2.0 - 1.0, 1.0 - px.y * 2.0);
  let near_ndc = vec4<f32>(uv.x, uv.y, 0.0, 1.0);
  let far_ndc  = vec4<f32>(uv.x, uv.y, 1.0, 1.0);
  let near_world_h = view_u.inv_view_proj * near_ndc;
  let far_world_h  = view_u.inv_view_proj * far_ndc;
  let near_world = near_world_h.xyz / near_world_h.w;
  let far_world  = far_world_h.xyz  / far_world_h.w;
  let diff_world = far_world - near_world;
  let frustum_length = length(diff_world);
  let dir_voxel = normalize(diff_world);
  let origin_voxel = view_u.cam_pos_voxel.xyz;

  // WHY: 相机陷在实心体素内部（同 `dda_main`）⇒ 整屏按"无 GI 数据"处理：走 `gi_ss_main` 的 miss
  // 分支（写 valid = 0、清 reservoir、清降噪导引），**不发射任何 GI 射线**。复用这条既有分支而不是
  // 自建早退的收益：离开该体素的那一帧从零重建（M = 1、无历史），异常态的累积不会带进正常画面。
  if (camera_solid_pal(origin_voxel) != 0u) {
    gi_ss_main(coord, size, world_miss());
    return;
  }

  // beam 纹理 = 全分辨率 / BEAM_DIV；本 pass 网格 = 全分辨率 / gi_div ⇒ 下标除 (BEAM_DIV / gi_div)。
  // gi_div = 1 → 除 BEAM_DIV；gi_div = 2 → 除 BEAM_DIV/2；gi_div = 4 → 1:1（一个 GI 像素正好一个 beam 纹素）。
  let gi_div = max(u32(gi_u.flags.y), 1u);
  var t_min = 0.0;
  if (view_u.lod.z < 0.5) {
    let beam_size = textureDimensions(beam_depth);
    let bx = gid.x / max(BEAM_DIV / gi_div, 1u);
    let by = gid.y / max(BEAM_DIV / gi_div, 1u);
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
  // 天空（miss）像素由 `gi_ss_main` 写 valid = 0 后早返回；介质（玻璃）像素**自 MT5-1 起与不透明
  // 像素同样产出 GI**（同一个 `fetch_material` + `brdf_reflected`，只是按 albedo 漫反射近似，
  // 见 `gi/screen.wesl::gi_ss_main` 的说明）。
  gi_ss_main(coord, size, h);
}

// ============================================================================
// GI 降噪（`gi/denoise.wesl`）：**时域累积** + **迭代 atrous** 两段，都跑在 GI 网格
// （= 渲染分辨率 / gi_div）上。
// 顺序（Rust `dispatch_dda`）：`gi_main` → `gi_face_flatten` → `gi_denoise_temporal`
// → `gi_denoise_atrous1/2/4/8/16`（步长 1/2/4/8/16 = NRD/RELAX 的 5 轮 a-trous；实际跑几轮
// = `GI_DEN_ATROUS_ITER`，中间靶只有 4 张 ⇒ 后面几轮靠 a↔b ping-pong）。最终结果落在 `gi_den`
// （`dda_main` 采样它，原始 `gi_out` 保留作对照）。
// ============================================================================

/// ① **逐面合并的落地 pass**：把每个 GI texel 的估计换成"它所在那个面"的均值。
/// 位置：紧跟 `gi_main`（定点累加在那一步做完了）、排在降噪链**之前** —— 于是时域累积与 atrous
/// 拿到的是"整个面一个值"的输入（面内不再有逐 texel 的独立噪声）。
/// 派发规模 = GI 网格；只有"槽已被本面认领、且两个键 word 逐位相等"的 texel 会被改写，
/// 撞键 / 未认领的面保持 `gi_main` 写下的逐 texel 值（只损失收益，不损失正确性）。
///
/// 与 `dda_face_accum` 是两件事：那个求的是**降噪之后**的逐面均值（供 `dda_face_main` 逐面着色，
/// 因为时域/atrous 之后面内仍可能有细微差异），本 pass 求的是**降噪之前**的逐面均值（为了降噪）。
/// 定点参数与 CAS 预留计数的溢出账共用 `GI_ACC_*`（见 `gi/common.wesl`）。
@compute @workgroup_size(8, 8, 1)
fn gi_face_flatten(@builtin(global_invocation_id) gid: vec3<u32>) {
  let size = textureDimensions(gi_out_flat);
  if (gid.x >= size.x || gid.y >= size.y) { return; }
  let idx = gid.y * size.x + gid.x;
  let gb = idx * GI_DEN_GUIDE_WORDS;
  let k0 = gi_guide[gb + GI_DEN_G_KEY0];
  let k1 = gi_guide[gb + GI_DEN_G_KEY1];
  // 天空 texel：`gi_out` 已是 0（`gi_ss_main` 的 miss 分支），不动。
  if ((k0 | k1) == 0u) { return; }
  var ok = false;
  // 查槽走 `face_slot_find_flat`（读 `@group(0) @binding(18)` 那份声明）：本 pass 的 layout 只有
  // group(0)，调 `face_slot_find` 会把 group(5) 拖进来 ⇒ pipeline 创建直接失败（见那个函数的说明）。
  // 双探针：认领可能落在第二格 —— 只探基槽会让那些面**没有面内平均**（降噪输入留着逐 texel 噪声）。
  let slot = face_slot_find_flat(k0, k1, &ok);
  if (!ok) { return; }
  let fe = slot * FACE_WORDS;
  let n = atomicLoad(&face_slots_flat[fe + FACE_W_RAW_N]);
  if (n == 0u) { return; }
  let inv = 1.0 / (f32(n) * GI_ACC_SCALE);
  let flat = vec3<f32>(
    f32(atomicLoad(&face_slots_flat[fe + FACE_W_RAW])),
    f32(atomicLoad(&face_slots_flat[fe + FACE_W_RAW + 1u])),
    f32(atomicLoad(&face_slots_flat[fe + FACE_W_RAW + 2u]))) * inv;
  textureStore(gi_out_flat, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(flat, 1.0));
}

/// 时域累积：读本帧原始 GI + 导引 + 上帧历史 → 写累积色/矩/历史长度（双缓冲）+ φ。
@compute @workgroup_size(8, 8, 1)
fn gi_denoise_temporal(@builtin(global_invocation_id) gid: vec3<u32>) {
  let size = textureDimensions(gi_dn_tmp);
  if (gid.x >= size.x || gid.y >= size.y) { return; }
  gi_den_temporal(size, vec2<i32>(i32(gid.x), i32(gid.y)));
}

/// atrous 第 1 轮（步长 1）。
@compute @workgroup_size(8, 8, 1)
fn gi_denoise_atrous1(@builtin(global_invocation_id) gid: vec3<u32>) {
  let size = textureDimensions(gi_dn_dst);
  if (gid.x >= size.x || gid.y >= size.y) { return; }
  gi_den_atrous(size, vec2<i32>(i32(gid.x), i32(gid.y)), 1);
}

/// atrous 第 2 轮（步长 2）。
@compute @workgroup_size(8, 8, 1)
fn gi_denoise_atrous2(@builtin(global_invocation_id) gid: vec3<u32>) {
  let size = textureDimensions(gi_dn_dst);
  if (gid.x >= size.x || gid.y >= size.y) { return; }
  gi_den_atrous(size, vec2<i32>(i32(gid.x), i32(gid.y)), 2);
}

/// atrous 第 3 轮（步长 4）。
@compute @workgroup_size(8, 8, 1)
fn gi_denoise_atrous4(@builtin(global_invocation_id) gid: vec3<u32>) {
  let size = textureDimensions(gi_dn_dst);
  if (gid.x >= size.x || gid.y >= size.y) { return; }
  gi_den_atrous(size, vec2<i32>(i32(gid.x), i32(gid.y)), 4);
}

/// atrous 第 4 轮（步长 8）。GI 是低频信号，大半径才不会伤到细节；
/// 硬拒判据（不是同一个面就 cut）与半径无关 ⇒ 大核也不会跨几何糊。
@compute @workgroup_size(8, 8, 1)
fn gi_denoise_atrous8(@builtin(global_invocation_id) gid: vec3<u32>) {
  let size = textureDimensions(gi_dn_dst);
  if (gid.x >= size.x || gid.y >= size.y) { return; }
  gi_den_atrous(size, vec2<i32>(i32(gid.x), i32(gid.y)), 8);
}

/// atrous 第 5 轮（步长 16）：足迹半径 = `GI_DEN_ATROUS_R` × 16 个 GI 像素。
@compute @workgroup_size(8, 8, 1)
fn gi_denoise_atrous16(@builtin(global_invocation_id) gid: vec3<u32>) {
  let size = textureDimensions(gi_dn_dst);
  if (gid.x >= size.x || gid.y >= size.y) { return; }
  gi_den_atrous(size, vec2<i32>(i32(gid.x), i32(gid.y)), 16);
}

// ============================================================================
// 光柱（godray）：**三个薄入口**，实现全在 `volumetric.wesl`
// ============================================================================
// 为什么入口留在本文件：WESL 只把「从根模块（`main.wesl`）可达」的东西写进编译产物 ——
// 子模块里没有任何调用点的 `@compute` 会被整条剪掉（pipeline 按名字找入口时会直接失败）。
// 算法、成本模型、取舍都在 `volumetric.wesl` 的文件头。
/// 掩码 pass：每 texel 一条主射线 ⇒ 「这个方向看得见天空吗」× 太阳方向权重 × 太阳色 × 强度。
@compute @workgroup_size(8, 8, 1)
fn godray_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  godray_pass_mask(gid);
}

/// 径向模糊 · 第一趟：读中间靶 A、写 B（走 `GODRAY_STEP_A` 比例）。
@compute @workgroup_size(8, 8, 1)
fn godray_blur_a(@builtin(global_invocation_id) gid: vec3<u32>) {
  godray_pass_blur_a(gid);
}

/// 径向模糊 · 第二趟：读 B、写 **A**（A 里就是最终结果，`dda_main` 采样它；走 `GODRAY_STEP_B`）。
@compute @workgroup_size(8, 8, 1)
fn godray_blur_b(@builtin(global_invocation_id) gid: vec3<u32>) {
  godray_pass_blur_b(gid);
}

// 自动曝光：pass1 `eye_adapt_histogram` 抽样 → log2 亮度 → 64 桶直方图；
// pass2 `eye_adapt_update` 取 5%~95% 百分位平均 + 时间平滑 → 写曝光系数（下一帧 `dda_main` 用）。
/// log2 亮度桶覆盖 [-12, +12]：24 档 / 64 桶 = 0.375 档每桶。
const EYE_LUM_MIN_LOG2: f32 = -12.0;
const EYE_LUM_RANGE_LOG2: f32 = 24.0;
/// 状态/调试区字数（曝光系数、已适应 EV、手动偏移、帧间隔、百分位均值、目标 EV…）—— 直方图区紧随其后。
const EYE_HIST_BASE: u32 = 8u;
const EYE_HIST_BINS: u32 = 64u;
/// 活参数区起始字 = 状态/调试区 + 直方图区，由 Rust `EyeAdaptSettings` 上传。
/// 下标 = `eye_p(i)` 的实参，顺序即语义（改 Rust 结构体必须同步）：0 ev_max 1 ev_min 2 tau_brighten 3 tau_darken 4 key。
const EYE_PARAM_BASE: u32 = EYE_HIST_BASE + EYE_HIST_BINS;

fn eye_p(i: u32) -> f32 {
  return bitcast<f32>(atomicLoad(&eye_adapt[EYE_PARAM_BASE + i]));
}

/// WG 内共享直方图（原子类型只能位于 workgroup 地址空间）。
var<workgroup> eye_hist: array<atomic<u32>, EYE_HIST_BINS>;

@compute @workgroup_size(256)
fn eye_adapt_histogram(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dims = vec2<i32>(textureDimensions(out_tex_src));
  for (var i = 0u; i < EYE_HIST_BINS; i = i + 1u) { atomicStore(&eye_hist[i], 0u); }
  workgroupBarrier();
  let stride = 16u;
  let axis = u32(dims.x / 16);
  let step = axis * u32(dims.y / 16);
  let bin_w = EYE_LUM_RANGE_LOG2 / f32(EYE_HIST_BINS);
  for (var k = gid.x; k < step; k = k + 256u) {
    let sx = i32((k % axis) * stride + stride / 2u);
    let sy = i32((k / axis) * stride + stride / 2u);
    if (sx >= dims.x || sy >= dims.y) { continue; }
    let c = textureLoad(out_tex_src, vec2<i32>(sx, sy), 0).rgb;
    let luma = dot(srgb_to_linear(c), GI_LUMA);
    let b = u32(clamp((log2(max(luma, 1e-5)) - EYE_LUM_MIN_LOG2) / bin_w, 0.0, f32(EYE_HIST_BINS - 1u)));
    atomicAdd(&eye_hist[b], 1u);
  }
  workgroupBarrier();
  for (var i = gid.x; i < EYE_HIST_BINS; i = i + 256u) {
    let v = atomicLoad(&eye_hist[i]);
    if (v > 0u) { atomicAdd(&eye_adapt[EYE_HIST_BASE + i], v); }
  }
}

@compute @workgroup_size(1)
fn eye_adapt_update() {
  var total = 0.0;
  for (var i = 0u; i < EYE_HIST_BINS; i = i + 1u) { total = total + f32(atomicLoad(&eye_adapt[EYE_HIST_BASE + i])); }
  if (total > 0.0) {
    let lo = total * 0.05;
    let hi = total * 0.95;
    let bin_w = EYE_LUM_RANGE_LOG2 / f32(EYE_HIST_BINS);
    var cum = 0.0;
    var acc = 0.0;
    var wsum = 0.0;
    for (var i = 0u; i < EYE_HIST_BINS; i = i + 1u) {
      let n = f32(atomicLoad(&eye_adapt[EYE_HIST_BASE + i]));
      let c0 = cum;
      cum = cum + n;
      let w = max(0.0, min(cum, hi) - max(c0, lo));
      acc = acc + (EYE_LUM_MIN_LOG2 + (f32(i) + 0.5) * bin_w) * w;
      wsum = wsum + w;
    }
    if (wsum > 0.0) {
      let avg_log2 = acc / wsum;
      atomicStore(&eye_adapt[4u], bitcast<u32>(avg_log2));
      let ev_cur = bitcast<f32>(atomicLoad(&eye_adapt[1u]));
      let ev_want = clamp(ev_cur + (log2(eye_p(4u)) - avg_log2), eye_p(1u), eye_p(0u));
      atomicStore(&eye_adapt[5u], bitcast<u32>(ev_want));
      let dt = clamp(bitcast<f32>(atomicLoad(&eye_adapt[3u])), 0.0, 0.25);
      let tau = select(eye_p(3u), eye_p(2u), ev_want > ev_cur);
      // 与帧率无关的指数平滑
      let k = select(0.0, 1.0 - exp(-dt / max(tau, 1e-3)), dt > 0.0);
      let ev_new = mix(ev_cur, ev_want, k);
      atomicStore(&eye_adapt[1u], bitcast<u32>(ev_new));
      let manual = bitcast<f32>(atomicLoad(&eye_adapt[2u]));
      let exp_val = exp2(clamp(ev_new + manual, eye_p(1u), eye_p(0u)));
      atomicStore(&eye_adapt[0u], bitcast<u32>(exp_val));
    }
  }
  for (var i = 0u; i < EYE_HIST_BINS; i = i + 1u) { atomicStore(&eye_adapt[EYE_HIST_BASE + i], 0u); }
}
