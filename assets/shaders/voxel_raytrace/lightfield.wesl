import package::bindings::{light_samp, light_tex, view_u};

// ============================================================================
// 光照场：**AO fill**（16³ 填充率网格，Douglas #15）。
//   · cell = LIGHT_FIELD_CELL 体素，dims = 32³；原点 = align_down(相机 − dims/2·cell, cell)
//   · 纹素下标 = 世界 cell mod dims（世界锚定）：相机滚动只换「新进窗口那条带」的纹素
// 原点式与 Rust `brickmap/upload.rs::light_field_origin_cell` 是同一个式子，必须逐字一致。
//
// 纹理的 rgb 三个通道已不再承载"发光密度 ε"（见下方 AO 说明）：自发光只走"射线命中即返回
// 它的颜色"这一条（Douglas #19 的辐射函数第三分支，即 `main.wesl` 的 cast/dda_main）。
// CPU 侧对应地不再累加发光，rgb 恒写 0 ⇒ 3D 纹理里只有 `.a` 有意义。
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
