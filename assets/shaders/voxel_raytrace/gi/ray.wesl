import package::common::{palette_albedo, palette_emissive, palette_transmission, sky_rgb};
import package::gi::cache::{gi_cache_peek, gi_key0, gi_key1, gi_key_ok};
import package::gi::consts::{GI_EMIT_GAIN, GI_PI, GI_SKY_RADIANCE_SCALE, GI_SUN_BOUNCE, GI_T_MAX, SHADOW_SURFACE_EPS};
import package::trace::{make_grid};
import package::world::{world_cfg_full, world_raycast, world_trace_medium, voxel_normal};

// ============================================================================
// 单条 GI 射线的辐射求值（**唯一口径**，两条路径共用）。
//
// 旧路径（世界空间缓存 `gi_cache_update`）与本步新增的屏幕空间路径（`gi/screen.wesl`）
// 都必须走这里，否则两侧的亮度口径（`E_store` 语义、二次顶点回读、除 π）会分叉、画面会跳变。
//
// 一次求值 = 1 条真实光路（`world_raycast`）＋ 命中时最多 1 条太阳 NEE 阴影射线；
// 命中面读「同体素同面」的旧缓存条目（只查不建 ⇒ 缓存规模跟着可见性走），
// 没有条目就只吃天光；逃逸取 `sky_rgb()`。
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
/// * `l` = 入射辐亮度（**不含** cos 因子与 π；两种复用候选的系数由调用方按各自 pdf 给）。
/// * `sk0`/`sk1` = 二次顶点的**精确键**（`gi_key0`/`gi_key1` 口径：命中体素 + 面 + 物体）。
///   `sk_hit = false` = 射线逃逸到天空（键取 (0,0)，同向天空两次求值必然相等）。
/// * `sk_ok` = 键可编码且可比较。屏幕空间复用**只在 `sk_ok` 为真时接受历史样本**
///   （键编码范围外无法做精确比较 ⇒ 宁可不复用，也不用启发式）。
struct GiRaySample {
  l: vec3<f32>,
  sk0: u32,
  sk1: u32,
  sk_hit: bool,
  sk_ok: bool,
}

/// 一条 GI 射线的入射辐亮度（被积函数的 L 部分）：真实光路 + 二次顶点 NEE，逃逸取天光。
/// 不含 cos 因子与 π —— RIS 的权重是 `p̂ / p`，两种候选的系数在调用方给。
fn gi_ray_radiance(org: vec3<f32>, dir: vec3<f32>, sun_dir: vec3<f32>, sun_c: vec3<f32>) -> GiRaySample {
  let sh = world_raycast(make_grid(0u), org, dir, GI_T_MAX, world_cfg_full(0.0, 3u));
  // 逃逸 → 天光辐亮度。
  var l = sky_rgb() * GI_SKY_RADIANCE_SCALE;
  if (sh.hit) {
    let alb = palette_albedo(sh.palette_base, sh.pal);
    let hit_p = org + dir * sh.t;
    let hn = voxel_normal(make_grid(u32(sh.obj_id) + 1u), sh.voxel, sh.n);
    // 二次顶点回读**同体素同面**的条目（不插值 ⇒ 跨薄板/墙缝不漏光）；没有条目就只吃天光。
    // 条目创建权留给屏幕（屏幕驱动缓存边界），这里只查不建 ⇒ 缓存规模跟着可见性走。
    let e_hit = gi_cache_peek(sh.voxel, sh.face_id, sh.obj_id);
    let ndl = max(dot(hn, sun_dir), 0.0);
    var sun = 0.0;
    if (ndl > 0.0) {
      let h_off = 0.5 / max(max(abs(hn.x), abs(hn.y)), max(abs(hn.z), 1e-3)) + SHADOW_SURFACE_EPS;
      sun = sun_transmittance(hit_p + hn * h_off, sun_dir, sh.voxel, sh.obj_id);
    }
    // 出射辐亮度约定：入射项带 1/π，直射项同样带 1/π（不含 GI 增益 —— 增益是给屏幕看的，
    // 进光输运会变成 gain² 反馈）。
    l = alb * (e_hit / GI_PI + sun_c * ndl * sun * GI_SUN_BOUNCE / GI_PI)
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

/// 只求**二次顶点键**的轻量射线（屏幕空间复用的时域验证专用）：键的口径与 `gi_ray_radiance`
/// 逐字一致（同一次 `world_raycast`、同一套 `gi_key0/gi_key1`），但**不做**二次顶点 NEE、
/// 不读缓存、不取 albedo ⇒ 少发一条阴影射线。`l` 恒为 0（调用方只用键）。
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
