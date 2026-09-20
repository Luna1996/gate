import package::bindings::{light_u};
import package::common::{palette_albedo, palette_emissive, palette_transmission, sky_rgb};
import package::gi::common::{gi_key0, gi_key1, gi_key_ok};
import package::gi::consts::{GI_EMIT_GAIN, GI_PI, GI_SKY_AMBIENT, GI_SKY_RADIANCE_SCALE, GI_T_MAX, SHADOW_SURFACE_EPS};
import package::lightfield::{light_field_ao};
import package::trace::{make_grid};
import package::world::{world_cfg_full, world_raycast, world_trace_medium, voxel_normal};

// ============================================================================
// 单条 GI 射线的辐射求值（**唯一口径**：新鲜候选、借用候选、时域/空间复用全部走这里）。
//
// 一次求值 = 1 条真实光路（`world_raycast`）＋ 命中时最多 1 条太阳 NEE 阴影射线。
// `sun_bounce = 0`（菜单「渲染/RESTIR GI/太阳反弹」关）时连那条阴影射线都不发，二次顶点只剩天光项。
//
// **辐亮度只算一次弹射**：命中面按「直射太阳 + 天光 × AO」着色后除以 π 得到它的出射辐亮度，
// 没有更深的弹射。多重弹射曾经靠世界空间缓存的回读实现（`gi_cache_peek`），那条路是个
// **跨帧反馈环**（缓存值进射线、射线值又进缓存，且缓存用 `1/n` 运行平均收敛）——
// 表现就是「像拿刷子一层层刷、稳态要等二十秒、有无 GI 的异常区」。删除后 GI 的输入
// 只依赖几何与光照，是**逐帧确定**的：没有可累积的中间量，也就没有「收敛」这件事。
// 代价是丢掉二次以上的弹射（多面互反射），这是刻意的取舍。
//
// 出射辐亮度约定（与 `dda_main` 的不透明分支逐项对齐，只是除以 π 转成 radiance）：
//   `L_o = albedo · (太阳直射项 + 天光项) / π + albedo · emissive`
// 其中天光项在二次顶点乘 `light_field_ao`（与主着色点同一套 AO，口径不分叉）。
//
// 逃逸到天空 ⇒ `sky_rgb()`（唯一的 miss 辐亮度口径）。
// ============================================================================

/// 太阳方向的直光透过率（1 = 无遮挡，0 = 被不透明体素全挡），一次穿透 trace 走完整段介质。
/// `self_voxel`/`self_obj`：起点贴命中面外侧，豁免 t≈0 自命中。
fn sun_transmittance(origin: vec3<f32>, sun_dir: vec3<f32>, self_voxel: vec3<i32>, self_obj: i32) -> f32 {
  let sh = world_raycast(make_grid(0u), origin, sun_dir, 8192.0, world_cfg_full(0.0, 3u));
  if (!sh.hit) { return 1.0; }
  if (all(sh.voxel == self_voxel) && sh.obj_id == self_obj) { return 1.0; }
  if (palette_transmission(sh.palette_base, sh.pal) <= 0.0) { return 0.0; }
  var tint = vec3<f32>(1.0);
  let h = world_trace_medium(make_grid(0u), origin, sun_dir, 8192.0, false, &tint);
  let through = (tint.r + tint.g + tint.b) / 3.0;
  if (!h.hit) { return through; }
  if (all(h.voxel == self_voxel) && h.obj_id == self_obj) { return 1.0; }
  if (palette_transmission(h.palette_base, h.pal) <= 0.0) { return 0.0; }
  return through;
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
fn gi_ray_radiance(org: vec3<f32>, dir: vec3<f32>, sun_dir: vec3<f32>, sun_c: vec3<f32>,
                    sun_bounce: f32) -> GiRaySample {
  let sh = world_raycast(make_grid(0u), org, dir, GI_T_MAX, world_cfg_full(0.0, 3u));
  // 逃逸 → 天光辐亮度。
  var l = sky_rgb() * GI_SKY_RADIANCE_SCALE;
  if (sh.hit) {
    let alb = palette_albedo(sh.palette_base, sh.pal);
    let hit_p = org + dir * sh.t;
    let hn = voxel_normal(make_grid(u32(sh.obj_id) + 1u), sh.voxel, sh.n);
    let ndl = max(dot(hn, sun_dir), 0.0);
    var sun = 0.0;
    // `sun_bounce` 来自 uniform（一次派发内恒定 ⇒ 分支不发散）：关掉时整条阴影射线都不发。
    if (sun_bounce > 0.0 && ndl > 0.0) {
      let h_off = 0.5 / max(max(abs(hn.x), abs(hn.y)), max(abs(hn.z), 1e-3)) + SHADOW_SURFACE_EPS;
      sun = sun_transmittance(hit_p + hn * h_off, sun_dir, sh.voxel, sh.obj_id);
    }
    // 二次顶点的天光项：与主着色点同一套系数 × 同一套 AO（口径不分叉）。
    let amb = light_u.sky_color.xyz * GI_SKY_AMBIENT * light_field_ao(hit_p, sh.obj_id);
    l = alb * (sun_c * ndl * sun * sun_bounce + amb) / GI_PI
      + alb * palette_emissive(sh.palette_base, sh.pal) * GI_EMIT_GAIN;
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
/// 逐字一致（同一次 `world_raycast`、同一套 `gi_key0/gi_key1`），但**不做**二次顶点 NEE、
/// 不取 albedo ⇒ 少发一条阴影射线。`l` 恒为 0（调用方只用键）。
fn gi_ray_secondary_key(org: vec3<f32>, dir: vec3<f32>) -> GiRaySample {
  let sh = world_raycast(make_grid(0u), org, dir, GI_T_MAX, world_cfg_full(0.0, 3u));
  return GiRaySample(
    vec3<f32>(0.0),
    gi_key0(sh.voxel),
    gi_key1(sh.voxel, sh.face_id, sh.obj_id),
    sh.hit,
    !sh.hit || gi_key_ok(sh.voxel),
  );
}