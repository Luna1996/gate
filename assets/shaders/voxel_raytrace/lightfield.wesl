import package::bindings::{light_samp, light_tex, view_u};
import package::ddgi::consts::{DDGI_EMIT_GAIN};

// ============================================================================
// 光照场：AO fill 与发光密度 ε 共用一张 3D 纹理（16³ 填充率网格，Douglas #15）。
//   · cell = LIGHT_FIELD_CELL 体素，dims = 32³；原点 = align_down(相机 − dims/2·cell, cell)
//   · 纹素下标 = 世界 cell mod dims（世界锚定）：相机滚动只换「新进窗口那条带」的纹素
// 原点式与 Rust `brickmap/upload.rs::light_field_origin_cell` 是同一个式子，必须逐字一致。
// ============================================================================
const LIGHT_FIELD_CELL: f32 = 16.0;
const LIGHT_FIELD_DIM: i32 = 32;
/// 场窗口原点（世界 voxel，按 cell 对齐；相机恒在盒中心）。
fn light_field_origin_voxel() -> vec3<f32> {
  let half = f32(LIGHT_FIELD_DIM / 2) * LIGHT_FIELD_CELL;
  return floor((view_u.cam_pos_voxel.xyz - vec3<f32>(half)) / LIGHT_FIELD_CELL) * LIGHT_FIELD_CELL;
}
/// 世界 voxel 位置 → 光照场纹理 uv：连续 cell 坐标卷绕到 [0, dim) 后归一化。
/// 纹素 c 存的是整个 cell [16c, 16c+16) 的均值（CPU tally 同语义），故采样位置取 `r` 本身：
/// cell 中心（r = c+0.5）落在纹素中心（三线性取到原值），cell 边界处与相邻纹素各半。
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
/// 发光密度 ε（**线性 RGB**，0..1，= cell 内 Σ albedo·emissive / cell 体积）。
/// 颜色在 CPU tally 侧就带上了（见 `upload.rs::build_light_chunk`），这里是纯取用 ——
/// 只存强度的话，红发光会投出白光。
fn light_field_emit(p_voxel: vec3<f32>) -> vec3<f32> {
  return textureSampleLevel(light_tex, light_samp, light_field_uv(p_voxel), 0.0).rgb;
}
/// cast 射线沿程累加发光：把 cell 的发光摊成密度 ε 并积分 radiance += ε·Δs
/// （单体素 2cm，32cm 探针间距下命中概率 ~3e-5，逐体素求交在采样上不可行）。
/// 只累加到几何首命中 t_max 为止 → 两侧遮挡依然正确，不引入漏光。
///
/// **增益标定**：`Δs / LIGHT_FIELD_CELL` 使「穿过一整格全发光体」的积分恰好等于表面路径的
/// `alb·emissive·DDGI_EMIT_GAIN` —— 两条路径共用同一个增益、量纲可比；比例与格边长无关。
/// 不归一（原式是裸的 `ε·Δs`）会让体积项比表面路径亮一个格边长（16×）⇒ 发光周围糊成一团
/// 过曝的白块，且它的形状（整格立方）会盖过发光体本身。
///
/// `jitter ∈ [0, 1)`：起点相位抖动。步长 == 格边长时每格恰好采到一个样点，若起点相位固定，
/// 每个探针的积分值会带一个由它自己亚格相位决定的**静态偏置** ⇒ 三线性插值出来就是一圈一圈
/// 的亮斑。抖动把它打散成零均值噪声（同探针的方向集逐帧轮换，时域上还会再平滑一次）。
fn light_field_gather(origin: vec3<f32>, dir: vec3<f32>, t_max: f32, jitter: f32) -> vec3<f32> {
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
  // 粗步进：步长 = 1 cell，起点按 jitter 抖动（2·半径/步长 ≈ 55 步上限）
  let step = LIGHT_FIELD_CELL;
  let gain = DDGI_EMIT_GAIN * (step / LIGHT_FIELD_CELL);
  var t = t_lo + step * clamp(jitter, 0.0, 1.0);
  var acc = vec3<f32>(0.0);
  var i = 0;
  loop {
    if (t >= t_hi || i >= 64) { break; }
    let p = origin + dir * t;
    if (light_field_contains(p)) {
      acc = acc + light_field_emit(p) * gain;
    }
    t = t + step;
    i = i + 1;
  }
  return acc;
}
