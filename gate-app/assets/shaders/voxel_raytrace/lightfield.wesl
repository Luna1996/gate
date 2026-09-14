import package::bindings::{light_samp, light_tex, view_u};

// ============================================================================
// 光照场（Douglas #15 的 AO fill + 「体素即光源」的发光密度 ε，共用一张 3D 纹理）。
//
// 形态与 DDGI 级联同构，因此是可流式的：
//   · cell = LIGHT_FIELD_CELL 体素（= Douglas #15 的 16³ 填充率网格），dims = 32³
//   · 原点 = align_down(相机 − dims/2·cell, cell)：相机恒在盒中心（偏差 < 1 cell）
//   · 槽位 = 世界 cell mod dims（**世界锚定**）：相机滚动只换「新进窗口那条带」的纹素，
//     其余纹素保持自己的世界身份 → 纹素不会因相机移动而整幅失效
// 与 Rust `brickmap/upload.rs::light_field_origin_cell` 是同一个式子，必须逐字一致。
// ============================================================================
const LIGHT_FIELD_CELL: f32 = 16.0;
const LIGHT_FIELD_DIM: i32 = 32;
/// 场窗口原点（世界 voxel，按 cell 对齐；相机恒在盒中心）。
fn light_field_origin_voxel() -> vec3<f32> {
  let half = f32(LIGHT_FIELD_DIM / 2) * LIGHT_FIELD_CELL;
  return floor((view_u.cam_pos_voxel.xyz - vec3<f32>(half)) / LIGHT_FIELD_CELL) * LIGHT_FIELD_CELL;
}
/// 世界 voxel 位置 → 光照场纹理 uv。
/// 连续 cell 坐标卷绕到 [0, dim) 后归一化。纹素 c 存的是**整个 cell** [16c, 16c+16) 的
/// 均值（CPU tally 如此），故采样位置取 `r` 本身：cell 中心（r = c+0.5）恰好落在纹素 c 的
/// 中心（硬件三线性在此取到纹素原值），cell 边界处才与相邻纹素各半 —— 与 tally 语义一致。
fn light_field_uv(p_voxel: vec3<f32>) -> vec3<f32> {
  let d = vec3<f32>(f32(LIGHT_FIELD_DIM));
  let cc = p_voxel / LIGHT_FIELD_CELL;
  let r = cc - floor(cc / d) * d;
  return r / d;
}
/// 点是否落在场窗口内。窗口外不采样：环面卷绕会把远处几何映到错误纹素。
fn light_field_contains(p_voxel: vec3<f32>) -> bool {
  let rel = p_voxel - light_field_origin_voxel();
  let s = vec3<f32>(f32(LIGHT_FIELD_DIM) * LIGHT_FIELD_CELL);
  return all(rel >= vec3<f32>(0.0)) && all(rel < s);
}
/// AO（Douglas #15）：只作用于**间接**部分（常量天光 + GI），直射太阳由阴影负责。
/// 模型：平面上以该点为中心的球（这里是 16³ 块）恰好一半实体 → 不压暗；
/// 地板与墙的夹角处 ≈ 3/4 实体 → 按超出 50% 的部分压暗。
const AO_GAIN: f32 = 2.0;
fn light_field_ao(p_voxel: vec3<f32>, obj_id: i32) -> f32 {
  if (obj_id >= 0) { return 1.0; }
  if (!light_field_contains(p_voxel)) { return 1.0; }
  let fill = textureSampleLevel(light_tex, light_samp, light_field_uv(p_voxel), 0.0).a;
  return clamp(1.0 - (fill - 0.5) * AO_GAIN, 0.0, 1.0);
}
/// 发光密度 ε（0..1，= cell 内 Σ 发光强度 / cell 体积）。
fn light_field_emit(p_voxel: vec3<f32>) -> vec3<f32> {
  return textureSampleLevel(light_tex, light_samp, light_field_uv(p_voxel), 0.0).rgb;
}
/// cast 射线**沿程累加**发光（「体素即光源」在 GI 传输侧的接法）。
///
/// 为什么必须按体积聚合：单个发光体素 2cm，射线在 32cm 探针间距下命中它的概率 ~3e-5
/// （rpp=10 时约每 3000 帧一次）—— 逐体素求交在采样上不可行。这里把 cell 的发光摊成
/// **密度** ε 并沿程积分 radiance += ε·ΔL（功率守恒：摊开面积不改变该点收到的期待辐照度，
/// 但方差降 256 倍）。只累加到几何首命中 t_max 为止 → 两侧遮挡依然正确，不引入漏光。
fn light_field_gather(origin: vec3<f32>, dir: vec3<f32>, t_max: f32) -> vec3<f32> {
  // 射线 × 包围球求交（球保守包含盒；用二次式而非 slab，避免 dir 分量为 0 的除零）
  let size = f32(LIGHT_FIELD_DIM) * LIGHT_FIELD_CELL;
  let center = light_field_origin_voxel() + vec3<f32>(size * 0.5);
  let radius = size * 0.8660254; // 半对角线
  let oc = origin - center;
  let b = dot(oc, dir);
  let disc = b * b - (dot(oc, oc) - radius * radius);
  if (disc <= 0.0) { return vec3<f32>(0.0); }
  let sq = sqrt(disc);
  let t_lo = max(-b - sq, 0.0);
  let t_hi = min(-b + sq, t_max);
  if (t_hi <= t_lo) { return vec3<f32>(0.0); }
  // 粗步进：步长 = 1 cell，头半步居中（2·半径/步长 ≈ 55 步上限）
  let step = LIGHT_FIELD_CELL;
  var t = t_lo + step * 0.5;
  var acc = vec3<f32>(0.0);
  var i = 0;
  loop {
    if (t >= t_hi || i >= 64) { break; }
    let p = origin + dir * t;
    if (light_field_contains(p)) {
      acc = acc + light_field_emit(p) * step;
    }
    t = t + step;
    i = i + 1;
  }
  return acc;
}
