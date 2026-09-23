import package::bindings::{BASE_FLAG_AO, light_samp, light_tex, light_u, view_u};

// 光照场：16³ cell 填充率网格（dims = 32³），世界锚定（纹素下标 = 世界 cell mod dims）；纹理只 `.a`（AO fill）有意义，rgb 恒 0。
// 窗口原点 = 相机中心对齐 cell，与 Rust `light_field_origin_cell` 必须逐字一致。
// ⚠️ 这个 fill 是"**实心占比**"，不是"天空遮挡"：cell 越实心 ⇒ ao 越小（全实心 ⇒ ao = 0），
// 与"上方天空是否被挡"无关（来源见 `gate-render/src/brickmap/upload.rs` 的 `build_light_chunk`）。
const LIGHT_FIELD_CELL: f32 = 16.0;
const LIGHT_FIELD_DIM: i32 = 32;
/// 场窗口原点（世界 voxel，按 cell 对齐；相机恒在盒中心）。
fn light_field_origin_voxel() -> vec3<f32> {
  let half = f32(LIGHT_FIELD_DIM / 2) * LIGHT_FIELD_CELL;
  return floor((view_u.cam_pos_voxel.xyz - vec3<f32>(half)) / LIGHT_FIELD_CELL) * LIGHT_FIELD_CELL;
}
/// 世界 voxel → 光照场 uv：连续 cell 坐标卷绕到 [0, dim) 后归一化；纹素 c 存整个 cell [16c, 16c+16) 的均值。
fn light_field_uv(p_voxel: vec3<f32>) -> vec3<f32> {
  let d = vec3<f32>(f32(LIGHT_FIELD_DIM));
  let cc = p_voxel / LIGHT_FIELD_CELL;
  let r = cc - floor(cc / d) * d;
  return r / d;
}
/// 点是否落在场窗口内（窗口外不采样）。
fn light_field_contains(p_voxel: vec3<f32>) -> bool {
  let rel = p_voxel - light_field_origin_voxel();
  let s = vec3<f32>(f32(LIGHT_FIELD_DIM) * LIGHT_FIELD_CELL);
  return all(rel >= vec3<f32>(0.0)) && all(rel < s);
}
/// AO 增益：只作用于间接部分（常量天光 + GI），直射太阳由阴影负责。
const AO_GAIN: f32 = 2.0;
/// 光照场 AO 的**唯一出口**（菜单「渲染/基础/AO」把 `BASE_FLAG_AO` 清零即整条关掉）：
/// 四个调用点 —— `main.wesl` 的间接项 / 反射命中点的天光项 / 玻璃 in-scatter 的天光项，
/// 以及 `gi/ray.wesl` 的二次顶点天光项 —— 都从这里取，所以开关只写在这一个地方。
/// 关掉时 CPU 侧也**不再重铺与上传**这张纹理（它的唯一消费者就是本函数），见 upload.rs 的 `extract`。
fn light_field_ao(p_voxel: vec3<f32>, obj_id: i32) -> f32 {
  if ((light_u.g.base_flags & BASE_FLAG_AO) == 0u) { return 1.0; }
  if (obj_id >= 0) { return 1.0; }
  if (!light_field_contains(p_voxel)) { return 1.0; }
  let fill = textureSampleLevel(light_tex, light_samp, light_field_uv(p_voxel), 0.0).a;
  return clamp(1.0 - (fill - 0.5) * AO_GAIN, 0.0, 1.0);
}
