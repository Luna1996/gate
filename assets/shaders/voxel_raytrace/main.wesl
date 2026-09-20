import package::bindings::{BEAM_BACKOFF, BEAM_DIV, beam_depth, eye_adapt, eye_adapt_ro, gi_dn_dst, gi_dn_tmp, gi_guide_out, gi_out, gi_tex, gi_u, light_samp, light_u, out_tex, out_tex_src, view_u};
import package::common::{FLAGS_TRANSMISSIVE, SurfaceLight, brdf_reflected, dielectric_f0, emissive_radiance, f0_of, fetch_material, fresnel_schlick, linear_to_srgb, material_ior, palette_flags, sky_rgb, srgb_to_linear};
import package::gi::common::{gi_key0, gi_key1, gi_mix, gi_rand2, gi_unpack_dir};
import package::gi::consts::{GI_AMBIENT_FLOOR, GI_DEN_G_KEY0, GI_DEN_G_KEY1, GI_DEN_G_NRM, GI_DEN_GUIDE_WORDS, GI_EMIT_GAIN, GI_SKY_AMBIENT, SHADOW_SURFACE_EPS};
import package::gi::denoise::{gi_den_atrous, gi_den_temporal, gi_same_surface};
import package::gi::ray::{sun_transmittance};
import package::gi::screen::{gi_ss_main};
import package::lightfield::{light_field_ao};
import package::trace::{make_grid};
import package::world::{WorldHit, voxel_normal, world_cfg_full, world_raycast, world_trace_medium};

// 全屏逐像素层次栈式 mask DDA（64-tree）：chunk 间 256³ A&W，chunk 内 4 层栈帧（256→64→16→4→1）。
// 常量/寻址/slot 打包须与 Rust `brickmap::dda::wgsl_consts`、`view.rs::get_voxel`、`wire.rs::encode_slot` 一致。

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
// 环境光照当作**环境/间接镜面项**加到着色上（`dda_main`）。GI 链一字未动 —— 反射样本不进 ReSTIR，
// 理由写在 `dda_main` 里那段注释（MT4-2）。
// ============================================================================

/// **总开关（MT4-4 的 A/B 用）**：`1u` = 开（仓库默认 = 功能本体）、`0u` = 关。
/// 与 MT2-4 的 `PBR_DEBUG_ASSET` 同一个先例：**WESL 编译期常量**（改一行、重启即生效；WESL 是
/// 运行期读盘编译，不用 rebuild），不做运行时 uniform —— 菜单开关属 MT7，本次不做。
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
/// 还不如它的噪声与成本，不值得追。取 0.03：默认平凡材质是电介质（`specular = 1.0`）
/// ⇒ `F0 = ((1.5−1)/(1.5+1))² = 0.04` 恰在门槛之上（追）；把 PBR 变体的 `specular` 槽级覆盖调到
/// ≤ 0.5（`F0 ≤ 0.02 < 0.03`）即整类不追 —— 这就是"用 F0 判"的可操作旋钮（MT7 的 UI 接上即可）。
const PBR_REFL_MIN_F0: f32 = 0.03;
/// 触发判据 ②：`roughness` 高于本值不发线（**成本控制的主闸门**）。
/// 理由：一条方向射线只代表得了"足够窄"的那一瓣镜面（MT4-1 的成本红线就是"只有一条"）；
/// `roughness` 大时 GGX 瓣宽远超一个方向能代表的尺度 ⇒ 那条射线既是坏估计（噪声大），
/// 又与 MT3 已经加过的 `F0·(amb+gi)` 环境项重复（见 `brdf_reflected` 的环境项）⇒ 不如不追。
/// 取 0.5：与业界"单射线反射只用于中低粗糙度"的做法一致。
/// **本场景（castle.vox）的实际影响面很大**：该 .vox 的 256 个 MATL 条目 `_rough` 全是 `"0.1"`
/// ⇒ 每个材质的 `roughness ≈ 0.098`、`fade ≈ 0.8` ⇒ **几乎每个不透明像素都会发一条反射射线**
/// （MT4-4 实测：`gate_dda_trace` 1.26ms → ≈3.4ms）。这是场景材质决定的，不是判据失效；
/// 把某块涂成 `roughness > 0.5`（「游戏/编辑」光滑度滑杆 < 50%）即可看到它不再发线。
const PBR_REFL_MAX_ROUGH: f32 = 0.5;
/// 抖动锥的张角系数：锥半角 ≈ `atan(本值 × roughness)`（rad）。粗糙度越大瓣越宽 ⇒ 取样方向越散，
/// 同时权重按 `dda_main` 里的 `fade` 连续淡出到 0（⇒ 粗糙度扫描下总亮度单调、无跳变）。
const PBR_REFL_CONE_K: f32 = 1.0;

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
      best = world_trace_medium(make_grid(0u), org, dir, max_t, true, &seg);
    } else {
      best = world_raycast(make_grid(0u), org, dir, max_t, world_cfg_full(t_start, 3u));
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
    let nn = best.n;
    let ci = clamp(dot(-dir, nn), 0.0, 1.0);
    // MT3-8：折射率与 F0 都从**材质**取，不再有本文件里的硬编码常量（硬约束 9：F0 只有一个来源）。
    // 每轮界面各取一次：进出的是**同一个介质**，而 `best.pal` 就是该介质的材质（`world_trace_medium`
    // 在段末端返回的是介质材质本身）⇒ 不引入"当前介质 IOR"这种额外状态。
    // 平凡变体没有 IOR 字段 ⇒ `material_ior` 给默认 1.5、F0 = ((1.5−1)/(1.5+1))² = 0.04
    // ⇒ 折射与 Fresnel 与改动前玻璃那两份硬编码（IOR 1.5 / F0 0.04）一致（本条验收）。
    let ior = material_ior(best.palette_base, best.pal);
    if (!in_medium) {
      // 玻璃走的是折射状态机（不是 BRDF）：Fresnel 用**物理 IOR 导出的 F0**、不加槽级 specular 调制
      // （那条调制是给不透明 BRDF 的观感旋钮；`specular = 1.0` = 不调制 = 物理基值）。
      let f0 = dielectric_f0(ior, 1.0);
      let fres = f0 + (1.0 - f0) * pow(1.0 - ci, 5.0);
      if (fres > GLASS_REFL_MIN) {
        let rdir = reflect(dir, nn);
        let rhit = world_raycast(make_grid(0u), p_hit + nn * GLASS_EPS, rdir,
                                 max_t, world_cfg_full(0.0, 3u));
        var rc = sky_rgb();
        if (rhit.hit) {
          // ---- MT4-1 · **这条第三处着色表达式已接进统一入口** ----
          // 改动前这里是 `palette_albedo(rhit.pal) * (sun_c·N·L + sky·GLASS_REFL_AMBIENT)`，两个毛病：
          //   ① 它对 **PBR 变体条目按字节误读 word0**（那里按 D1 是 roughness/metallic/emissive/
          //      transmission 四个覆盖字节，不是 rgb）⇒ 把覆盖字节当颜色用，颜色是垃圾；
          //   ② 它是全工程**第四处着色代码路径**（硬约束 1 / 3 都不许）。
          // 现在与主 pass、GI 二次顶点走**同一组函数**：`fetch_material` 取材质 + `brdf_reflected` 求值；
          // 着色法线也同样是 `voxel_normal` 的隐式邻域梯度（与主 pass 一致）。
          // `SurfaceLight` 刻意与 `gi/ray.wesl` 的二次顶点**逐字对齐**（硬约束 3 不分叉），差异只有两条：
          //   · `sun_vis = 1.0`（**不做 NEE**：不给玻璃的反射再加一条阴影射线，沿用本条既有的"无遮挡"
          //     近似 —— 有意为之，不是漏了）；
          //   · 天光 = `sky · GLASS_REFL_AMBIENT`（玻璃反射沿用自己那个更暗的常量天光，与旧式同源）；
          //     `gi = 0`（反射点没有 GI 缓冲，与二次顶点同一条约定）。
          // 有意的一处观感变化（统一入口的直接后果）：反射命中自发光表面时**会**看到它的自发光
          // （`emissive_radiance` 与主 pass / 二次顶点同一句），旧式那条近似看不到。
          let rgg = make_grid(u32(rhit.obj_id) + 1u);
          let rn = voxel_normal(rgg, rhit.voxel, rhit.face_id, rhit.n);
          let mr = fetch_material(rhit.palette_base, rhit.pal, rhit.point, rn, PBR_DEBUG_ASSET);
          rc = brdf_reflected(mr, rn, -rdir,
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

/// **几何感知上采样**（joint bilateral）—— `dda_main` 取 GI 的唯一入口。
/// GI 跑在低于全分辨率的网格上（`GiSettings.gi_div` = 1 / 2 / 4），纯双线性会把**几何边界
/// 两侧**的值混在一起（棱边/墙角一侧的亮 GI 渗到另一侧），而这正是「GI 低于全分辨率」唯一
/// 真正丢掉的信息。档位越低（1/4）这一层越要紧：一个 GI texel 覆盖 4×4 个全分辨率像素。
/// 做法：取 GI 网格上最近的 2×2 texel，**只接受与本像素同一个面**的 texel
/// （`gi_same_surface`：平面归属靠键 ⇒ 薄板另一侧 / 隔一格的平行面不可能被接受；
/// 相似度靠**存的着色法线** ⇒ 凸棱上那一段倾斜的法线与本像素比仍在容差内 ⇒ 与着色一致地连起来），
/// 按双线性权重加权平均。退化顺序：一个都没接受 ⇒ **最近的 texel**（它至少在几何上覆盖本像素，
/// 比双线性更不容易渗色）⇒ 最近 texel 无有效数据 ⇒ 传进来的纯双线性结果。
/// 有效性判据 = 导引的「键非零」（与降噪侧一致）；最近 texel 那一档直接用纹理的 `a`（那里不插值，
/// `a` 恒是 0/1）。
/// `n` = 本像素的着色法线（`dda_main` 自己那条）—— 相似度那一条用它。
/// 代价：每全分辨率像素 4 次导引读（每次 3 word），**不加任何射线、不加 pass**。
fn gi_upsample_joint(
  guv: vec2<f32>,
  gsize: vec2<u32>,
  k0: u32,
  k1: u32,
  n: vec3<f32>,
  fallback: vec3<f32>,
) -> vec3<f32> {
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
  if (wsum > 0.0) { return acc / wsum; }
  // 一个同面的 texel 都没有（凸角 / 贴天空的轮廓）：退回最近的 texel。
  let nc = clamp(base + vec2<f32>(select(0.0, 1.0, f.x >= 0.5), select(0.0, 1.0, f.y >= 0.5)),
                 vec2<f32>(0.0), vec2<f32>(f32(gsize.x) - 1.0, f32(gsize.y) - 1.0));
  let sn = textureLoad(gi_tex, vec2<i32>(nc), 0);
  if (sn.a > 0.5) { return sn.rgb; }
  return fallback;
}

// DDA 主入口：命中 → 直光着色（硬阴影 + sky 环境 + emissive），miss → 纯色天空
@compute @workgroup_size(8, 8, 1)
fn dda_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let size = textureDimensions(out_tex);
  if (gid.x >= size.x || gid.y >= size.y) { return; }
  let coord0 = vec2<i32>(i32(gid.x), i32(gid.y));

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
                            world_cfg_full(t_min, 3u));
  var best = first;
  var dir = dir_voxel;
  var through = vec3<f32>(1.0);
  var refl = vec3<f32>(0.0);
  // 首命中是不是玻璃（走 `trace_glass` 状态机）看 `FLAGS_TRANSMISSIVE` 位，不读 transmission 字节：
  // PBR 变体里那个字节属于 `asset`（`docs/PLAN.md` D1），按字节读会把不透明材质误判成玻璃。
  // 平凡变体下该位恒等于 `transmission > 0`（由写入侧 `wire.rs::pack_palette_entry` 维护）⇒ 行为不变。
  if (first.hit && (palette_flags(first.palette_base, first.pal) & FLAGS_TRANSMISSIVE) != 0u) {
    let g = trace_glass(origin_voxel, dir_voxel, t_min, frustum_length, sun_dir, sky, sun_c);
    best = g.ray;
    dir = g.dir;
    through = g.tint;
    refl = g.refl;
  }
  var col = sky_rgb() * through + refl;
  if (best.hit) {
    col = vec3<f32>(0.0);
    // 材质参数（albedo / roughness / metallic / emissive / transmission / ior / specular）由下面那次
    // `fetch_material` 一次拿全 —— 这里不再单独取 albedo（避免出现"第二处取值口径"）。
    let gg = make_grid(u32(best.obj_id) + 1u);
    let vc = vec3<f32>(best.voxel) + vec3<f32>(0.5);
    let p_voxel = gg.pos + vec3<f32>(dot(vc, gg.col0), dot(vc, gg.col1), dot(vc, gg.col2)) * gg.scale;
    let k0 = gi_key0(best.voxel);
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
    if (u32(gi_u.flags.y) == 1u) {
      let gi_size = vec2<u32>(textureDimensions(gi_tex));
      if (gid.x < gi_size.x && gid.y < gi_size.y) {
        let gb = (gid.y * gi_size.x + gid.x) * GI_DEN_GUIDE_WORDS;
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
    // 解析之后**：triplanar 的三轴权重用 `n`、UV 用**命中点的世界坐标** `best.point`（连续点，
    // 不是体素中心）。平凡变体（`IS_PBR = 0`）与 PBR 变体在这里产出**同一个** `MaterialParams`
    // ⇒ 下面的着色代码对两者完全一致（D1 约束 1 / MT3-7）。
    let m = fetch_material(best.palette_base, best.pal, best.point, n, PBR_DEBUG_ASSET);
    // 沿 n 外推 0.5/max|n| + SHADOW_SURFACE_EPS，使起点离开命中体素。
    let n_off = 0.5 / max(max(abs(n.x), abs(n.y)), max(abs(n.z), 1e-3)) + SHADOW_SURFACE_EPS;
    let ndl = max(dot(n, sun_dir), 0.0);
    var sun = 0.0;
    if (ndl > 0.0) {
      sun = sun_transmittance(p_voxel + n * n_off, sun_dir, best.voxel, best.obj_id);
    }
    let gi_on = gi_u.misc.x > 0.5;
    var gi = vec3<f32>(0.0);
    var cov = 0.0;
    // GI 一律走缓冲通道（`gi_main` + 降噪写在 1/gi_div 分辨率的 GI 网格上，见 `gi/`）：
    // 这里只做一次几何感知上采样（内部用 2×2 texel）。uv 必须按**全分辨率尺寸**（`size`）
    // 归一化，不能除以 GI 纹理自身尺寸 —— 1:1 / 2:1 / 4:1 上采样都靠这条。
    // `gi_on = false` 时 gi/cov 恒 0 ⇒ 只有太阳直射 + 天光兜底。
    if (gi_on) {
      let guv = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5))
              / vec2<f32>(f32(size.x), f32(size.y));
      let s = textureSampleLevel(gi_tex, light_samp, guv, 0.0);
      // premultiplied-valid 归一化：valid = 0（天空）像素不参与插值。
      let gw = max(s.a, 1e-4);
      // 覆盖度 cov 直接取自 `a`：`gi_out.a = valid`，而独立的 cov 通道写的恒是「valid ? 1 : 0」
      // ⇒ `cov = valid / max(valid, 1e-4)` 与旧的 `sc.r / gw` 是同一个表达式（不再单开一张纹理）。
      // GI 值走**几何感知上采样**（纯双线性作 fallback）：不接受与本像素不同「面」的 texel
      // ⇒ 棱边 / 墙角不渗色、不糊（相似度那一条用本像素的着色法线 `n`）。cov 不参与
      // （它是 0/1 覆盖度，双线性归一化即可）。
      gi = gi_upsample_joint(guv, vec2<u32>(textureDimensions(gi_tex)), k0, k1, n, s.rgb / gw);
      cov = s.a / gw;
    }
    // AO 只乘到常量天光项（gi 已自带逐方向遮挡）。
    let ao = light_field_ao(p_voxel, best.obj_id);
    // cov 由缓存给出（命中 = 1、没有数据 = 0）：有数据处天光只占少量、无数据处天光兜底。
    // mix 实参顺序必须是 `mix(1.0, GI_AMBIENT_FLOOR, cov)`（cov = 1 → FLOOR）。
    let amb = sky * GI_SKY_AMBIENT * mix(1.0, GI_AMBIENT_FLOOR, cov);
    // ---- MT3-3 / MT3-7 · 着色（**唯一 BRDF 求值函数**，平凡与 PBR 共用）----
    // 与 `gi/ray.wesl` 的二次顶点逐项对齐（硬约束 3 / `docs/PLAN.md` MT3-4）：两侧都只是把各自的
    // 光源项交给同一个 `brdf_reflected`。差异只有三处，且都是物理上该不同的：
    //   ① 本处有 GI 项（`gi`）；二次顶点只算一次弹射 ⇒ `SurfaceLight.gi` 恒 0；
    //   ② 本处的天光含 `cov` 地板；二次顶点没有 GI 缓冲 ⇒ 只有 AO；
    //   ③ 本处不除 π（二次顶点要除 π 转辐亮度）—— 能量约定见 `brdf_reflected` 的文档注释。
    // `v = -dir` = 由着色点指向观察者（与二次顶点那边同一个构造方式）。
    col = brdf_reflected(m, n, -dir, SurfaceLight(sun_dir, sun_c, sun, amb * ao, gi))
        + emissive_radiance(m, GI_EMIT_GAIN);
    // ---- MT4-1 · 环境镜面反射（**本里程碑唯一的新增射线**）----
    // 设计（`docs/PLAN.md` §4 MT4-1）：该表面的镜面反射"值得追"时**追加一条**反射射线，
    // 把反射点的环境光照当作**环境/间接镜面项**加到 `col` 上。触发判据与成本控制见顶部
    // `PBR_REFL_*` 三个常量（**不是无条件每像素发射线**）。
    //
    // ---- MT4-2 · 边界：反射样本**不进 ReSTIR**（只在本函数里加到 `col`）----
    // ReSTIR 的样本等价判据是「**精确几何面键 + 二次顶点键逐位相等**」（硬约束 2 / `gi/consts.wesl`
    // 文件头），这是薄板/墙缝不漏光的根本。反射方向是**按 F0 与 roughness 加权抽出来的**一条方向、
    // 且抖动锥与 MT3 的余弦候选完全不是同一族分布 ⇒ 它既不描述"同一条光路"，也不满足"逐位相等"。
    // 一旦把它并进 reservoir / 导引 / 历史（或让 `gi_ray_radiance` 返回它），判据就会把"方向不同的
    // 两个样本"判成等价 —— 那正是 §5 R1 点名的失效形态：薄板漏光、屏幕漂移的光斑。
    // 所以本射线**不写任何 GI 资源**：不进 `gi_out`、不进 `gi_res_cur/prev`、不进导引与历史，
    // 也不参与 `gi_ray_radiance`；结果只出现在 `col` 上，且 `col * through + refl` 那一步之后
    // 也就离开本 pass（GI 链读的是它自己的缓冲，读不到这里）。
    // 玻璃路径下这一项与其余着色一样被 `through`（沿途介质吸收）乘一次 —— 反射光同样要"穿过"
    // 那段玻璃回到观察者，与 `col` 其余部分同一条口径（不额外单开一条乘性吸收）。
    // 自检：`PBR_REFLECTION_ENABLED = 0u` 重启 ⇒ 本段被整段折掉，GI 画面与 MT3 逐位一致。
    if (PBR_REFLECTION_ENABLED != 0u) {
      let f0r = f0_of(m);
      let f0_luma = dot(f0r, vec3<f32>(0.2126, 0.7152, 0.0722));
      if (f0_luma >= PBR_REFL_MIN_F0 && m.roughness <= PBR_REFL_MAX_ROUGH) {
        // 抖动种子：**只用像素坐标、不含帧号**。反射结果不参与任何时域累积（见上面的 MT4-2 边界）
        // ⇒ 逐帧变化的抖动只会变成闪烁；固定种子让粗糙面上的散斑是"静止图案"而不是"逐帧噪点"。
        // `gi_mix` / `gi_rand2` 是 GI 侧现成的哈希与 RNG（`gi/common.wesl`，与 `gi_ss_seed` 的
        // 像素哈希同一个写法）—— **不另造 RNG**。
        let rseed =
          gi_mix(u32(coord0.x) * 0x9E3779B1u + gi_mix(u32(coord0.y) * 0x85EBCA77u + 0x165667B1u));
        let mirror = reflect(dir, n);
        var rdir = refl_cone_dir(mirror, gi_rand2(rseed), PBR_REFL_CONE_K * m.roughness);
        // 抖动后的方向若掉到面下方（掠射 + 大锥角）会立刻自命中所在体素 ⇒ 退回完美镜面方向。
        if (dot(rdir, n) <= 0.0) { rdir = mirror; }
        // 起点 = 与阴影射线**同一个构造**（`p_voxel + n * n_off`：命中体素中心沿 n 推 `0.5/|n| + eps`
        // ⇒ 恰在表面外侧），口径一致、不另立第二套外推。
        var refl_env = sky_rgb();  // 逃逸到天空 ⇒ 与 GI 的 miss 口径一致（全工程唯一的 miss 辐亮度）
        let rh = world_raycast(make_grid(0u), p_voxel + n * n_off, rdir,
                               frustum_length, world_cfg_full(0.0, 3u));
        if (rh.hit) {
          // 反射命中点**必须**经 `fetch_material` + `brdf_reflected`（硬约束 1：不许手写第二套近似
          // 着色 —— 那正是 MT4-1 刚从 `trace_glass` 里清掉的那类第四处表达式）。
          let rgg = make_grid(u32(rh.obj_id) + 1u);
          let rn = voxel_normal(rgg, rh.voxel, rh.face_id, rh.n);
          let mr = fetch_material(rh.palette_base, rh.pal, rh.point, rn, PBR_DEBUG_ASSET);
          // 只取 `brdf_reflected` 的**环境项**：`sun_c = 0` 且 `sun_vis = 0` ⇒ 函数里那段太阳直射
          // 被整个跳过（判据 `ndl > 0 && sun_vis > 0` 为假），返回的恰是 `(kD·albedo + F0)·(amb + gi)`
          // —— 也就是"反射点自己的环境光照"，不含它的直射高光。
          //   · **这就是 MT4-3 的去重**（§4 MT4-3 / §5 R1）：直射高光（GGX 那一项）**只由 MT3 的 BRDF
          //     在本着色点算一次**；反射射线不做 NEE ⇒ 高光不可能被算两遍（若这里带上 `sun_c`，
          //     本像素的直射高光就会被"环境反射"再叠一遍 ⇒ 能量翻倍）。
          //   · 天光口径与 GI 二次顶点**逐字相同**（`gi/ray.wesl`：`sky · GI_SKY_AMBIENT · ao`，
          //     没有 `cov` 地板 —— 反射点没有 GI 缓冲，与二次顶点是同一条约定，硬约束 3 不分叉）；
          //     `gi` 传 0：那是本像素**屏空间**的缓冲，反射点常常不在屏幕上，借过来就是错的。
          //   · **有意写明的代价**：反射点自己的太阳直照（它可能是个被阳光照亮的镜面）在这一次反射里
          //     看不到 —— 因为不做 NEE 就拿不到它的阴影可见性，"宁可不追"比"漏一片光"好。
          let ramb = sky * GI_SKY_AMBIENT * light_field_ao(rh.point, rh.obj_id);
          refl_env = brdf_reflected(mr, rn, -rdir,
                                    SurfaceLight(sun_dir, vec3<f32>(0.0), 0.0, ramb, vec3<f32>(0.0)));
        }
        // 权重 = Schlick Fresnel（同一个 `fresnel_schlick`：完美镜面下 h ≈ n ⇒ 余弦取 `v·n`）
        //        × 粗糙度淡出（`roughness → PBR_REFL_MAX_ROUGH` 时权重**连续**归零 ⇒ 粗糙度扫描下
        //        总亮度单调、阈值处不出现亮度台阶；这也是 MT4-3 的能量自检要看的量）。
        // 人工自检方式（MT4-3）：用「游戏/编辑」的笔刷在同一面墙上涂出几块不同粗糙度的色块
        // （「光滑度」滑杆 100% / 90% / 75% / 50% / 40% ⇒ roughness 0 / 0.098 / 0.255 / 0.5 / 0.6），
        // 亮度应随 roughness **单调不增**、且在 50%（= 阈值）处**连续**（不出现台阶），开/关反射
        // 各看一次即可确认没有"高光算两遍"的能量翻倍。
        // 注意这里**只有加法**：MT3 环境项里的 `F0·(amb+gi)` 是"环境为常数"时同一个量的近似，
        // 严格做法是"用方向性估计替换它"，但两者量纲口径不同（一个是本点的辐照度猜测、一个是反射
        // 点的出射辐亮度）⇒ 直接相减会在 GI 强的面上把镜面反射**减成负值**（比轻微偏亮更难接受）。
        // 本版**有意保留加法**，代价上限是"镜面环境项最多算两遍"（电介质 F0 = 0.04 ⇒ 天光/GI 那部分
        // 最多 +4%；金属最坏 2×），登记在 MT4-3 的报告里，替换式写法留待后续。
        let fade = clamp(1.0 - m.roughness / PBR_REFL_MAX_ROUGH, 0.0, 1.0);
        col = col + fresnel_schlick(f0r, max(dot(-dir, n), 0.0)) * fade * refl_env;
      }
    }
    col = col * through + refl;
  }
  write_output(coord0, col);
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
// 顺序（Rust `dispatch_dda`）：`gi_main` → `gi_denoise_temporal` → `gi_denoise_atrous1/2/4/8/16`
// （步长 1/2/4/8/16 = NRD/RELAX 的 5 轮 a-trous；实际跑几轮 = `GI_DEN_ATROUS_ITER`，
// 中间靶只有 4 张 ⇒ 后面几轮靠 a↔b ping-pong）。最终结果落在 `gi_den`（`dda_main` 采样它，
// 原始 `gi_out` 保留作对照）。
// ============================================================================

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

// 自动曝光：pass1 `eye_adapt_histogram` 抽样 → log2 亮度 → 64 桶直方图；
// pass2 `eye_adapt_update` 取 5%~95% 百分位平均 + 时间平滑 → 写曝光系数（下一帧 `dda_main` 用）。
/// log2 亮度桶覆盖 [-12, +12]：24 档 / 64 桶 = 0.375 档每桶。
const EYE_LUM_MIN_LOG2: f32 = -12.0;
const EYE_LUM_RANGE_LOG2: f32 = 24.0;
const EYE_HIST_BINS: u32 = 64u;
/// 活参数区起始字（= 状态/调试区 8 + 直方图 64），由 Rust `EyeAdaptSettings` 上传。
/// 下标 = `eye_p(i)` 的实参，顺序即语义（改 Rust 结构体必须同步）：0 ev_max 1 ev_min 2 tau_brighten 3 tau_darken 4 key。
const EYE_PARAM_BASE: u32 = 8u + EYE_HIST_BINS;

fn eye_p(i: u32) -> f32 {
  return bitcast<f32>(atomicLoad(&eye_adapt[EYE_PARAM_BASE + i]));
}

/// WG 内共享直方图（原子类型只能位于 workgroup 地址空间）。
var<workgroup> eye_hist: array<atomic<u32>, 64>;

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
    let luma = dot(srgb_to_linear(c), vec3<f32>(0.2126, 0.7152, 0.0722));
    let b = u32(clamp((log2(max(luma, 1e-5)) - EYE_LUM_MIN_LOG2) / bin_w, 0.0, f32(EYE_HIST_BINS - 1u)));
    atomicAdd(&eye_hist[b], 1u);
  }
  workgroupBarrier();
  for (var i = gid.x; i < EYE_HIST_BINS; i = i + 256u) {
    let v = atomicLoad(&eye_hist[i]);
    if (v > 0u) { atomicAdd(&eye_adapt[8u + i], v); }
  }
}

@compute @workgroup_size(1)
fn eye_adapt_update() {
  var total = 0.0;
  for (var i = 0u; i < EYE_HIST_BINS; i = i + 1u) { total = total + f32(atomicLoad(&eye_adapt[8u + i])); }
  if (total > 0.0) {
    let lo = total * 0.05;
    let hi = total * 0.95;
    let bin_w = EYE_LUM_RANGE_LOG2 / f32(EYE_HIST_BINS);
    var cum = 0.0;
    var acc = 0.0;
    var wsum = 0.0;
    for (var i = 0u; i < EYE_HIST_BINS; i = i + 1u) {
      let n = f32(atomicLoad(&eye_adapt[8u + i]));
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
  for (var i = 0u; i < EYE_HIST_BINS; i = i + 1u) { atomicStore(&eye_adapt[8u + i], 0u); }
}
