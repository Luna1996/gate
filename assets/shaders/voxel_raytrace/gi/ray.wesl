import package::bindings::{light_u};
import package::common::{MATERIAL_DEBUG_NONE, SurfaceLight, brdf_reflected, emissive_radiance, fetch_material, sky_rgb};
import package::gi::common::{gi_key0, gi_key1, gi_key_ok};
import package::gi::consts::{GI_EMIT_GAIN, GI_PI, GI_SKY_AMBIENT, GI_SKY_RADIANCE_SCALE, GI_T_MAX, SHADOW_SURFACE_EPS};
import package::lightfield::{light_field_ao};
import package::trace::{make_grid};
import package::world::{world_trace_medium, voxel_normal};

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
  let h = world_trace_medium(make_grid(0u), origin, sun_dir, 8192.0, false, &tint);
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

/// 一条 GI 射线的入射辐亮度：真实光路 + 二次顶点「太阳 NEE + 天光 × AO」，逃逸取天光。
/// 不含 cos 因子与 π —— RIS 的权重是 `p̂ / p`，系数在调用方给（余弦候选给 π）。
///
/// **MT5-2 · 二次射线穿介质续行**：光路用 `world_trace_medium`（`stop_on_air = false`）而不是
/// `world_raycast` —— 撞到**透射**体素不停、沿途按格累积乘性吸收 `tint`，于是一条从玻璃前面出发的
/// GI 射线会**穿过玻璃**打到它后面的真实表面。旧写法用 `world_raycast`（`penetrate = false`，
/// 撞到任何实体体素都停）⇒ 二次顶点落在**玻璃面本身**上：玻璃后面的表面拿不到 GI，
/// 而且玻璃面被当成"不透明表面"着色。
/// 与 `sun_transmittance`（同样 `world_trace_medium` + `stop_on_air = false`）是**同一套写法与口径**：
///   · **停点/键不变**（硬约束 2）：只有不透明体素才停 ⇒ `sk0/sk1` 仍取最终命中的那个不透明面，
///     `gi_key0/gi_key1` 的口径**一字未改** ⇒ 复用判据（键逐位相等）不受影响；
///   · **吸收口径**：`tint` 乘在**整个** `l` 上（BRDF 项与自发光项都乘）—— 与主 pass 的
///     `col = col * through + refl` 是同一个"乘性吸收"结构（硬约束 3：口径不分叉）。
/// 与旧写法的唯一范围差别：只遍历**主世界网格**（与 `sun_transmittance` 一致 —— 物体网格没有
/// 介质续行；而当前 `Volumes::add_object` 在应用里也没有调用点 ⇒ 无实际行为差异）。
fn gi_ray_radiance(org: vec3<f32>, dir: vec3<f32>, sun_dir: vec3<f32>, sun_c: vec3<f32>,
                    sun_bounce: f32) -> GiRaySample {
  var tint = vec3<f32>(1.0);
  let sh = world_trace_medium(make_grid(0u), org, dir, GI_T_MAX, false, &tint);
  // 逃逸 → 天光辐亮度（穿过介质看到的天空同样按 `tint` 吸收：`tint` 无介质时恒为 1.0）。
  var l = sky_rgb() * GI_SKY_RADIANCE_SCALE * tint;
  if (sh.hit) {
    let hit_p = org + dir * sh.t;
    let hn = voxel_normal(make_grid(u32(sh.obj_id) + 1u), sh.voxel, sh.face_id, sh.n);
    let ndl = max(dot(hn, sun_dir), 0.0);
    var sun = 0.0;
    // `sun_bounce` 来自 uniform（一次派发内恒定 ⇒ 分支不发散）：关掉时整条阴影射线都不发。
    if (sun_bounce > 0.0 && ndl > 0.0) {
      let h_off = 0.5 / max(max(abs(hn.x), abs(hn.y)), max(abs(hn.z), 1e-3)) + SHADOW_SURFACE_EPS;
      sun = sun_transmittance(hit_p + hn * h_off, sun_dir, sh.voxel, sh.obj_id);
    }
    // 二次顶点的天光项：与主着色点同一套系数 × 同一套 AO（口径不分叉）。
    let amb = light_u.sky_color.xyz * GI_SKY_AMBIENT * light_field_ao(hit_p, sh.obj_id);
    // ---- MT3-4 · 与 `dda_main` **同一个** BRDF 与同一个材质入口（硬约束 3）----
    // 只有三处不同（`dda_main` 那边的同一张表在 `main.wesl` 的注释里），都是物理上该不同的：
    //   ① 太阳项乘 `sun_bounce`（菜单「渲染/RESTIR GI/太阳反弹」：关掉时二次顶点不做太阳 NEE）；
    //   ② **没有 GI 项**（二次顶点只算一次弹射）⇒ `SurfaceLight.gi` 传 0；
    //   ③ 反射项 `/π` 转辐亮度（自发光不除，见文件头）。
    // `v = -dir`：出射方向就是"回到上一个顶点"的方向（与主 pass 的 `-dir` 同一个构造方式）。
    // 材质取值走同一入口 `fetch_material`；`MATERIAL_DEBUG_NONE` = 调试通道**不作用于 GI**
    // （与 MT2-4 一致：那条通道只是主 pass 的调试视图）。
    let m = fetch_material(sh.palette_base, sh.pal, hit_p, hn, MATERIAL_DEBUG_NONE);
    // MT5-2：二次顶点的辐亮度整体乘 `tint`（BRDF 项与自发光项都乘）—— 它就是"这个二次顶点发出的
    // 光，穿过玻璃到达本着色点"的乘性吸收；无介质时 `tint == 1.0` ⇒ 逐位等于改动前。
    l = (brdf_reflected(m, hn, -dir, SurfaceLight(sun_dir, sun_c, sun * sun_bounce, amb, vec3<f32>(0.0))) / GI_PI
      + emissive_radiance(m, GI_EMIT_GAIN)) * tint;
  }
  return GiRaySample(
    l,
    gi_key0(sh.voxel),
    gi_key1(sh.voxel, sh.face_id, sh.obj_id),
    sh.hit,
    !sh.hit || gi_key_ok(sh.voxel),
  );
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