import package::bindings::{g};
import package::brickmap::{Grid, sample_brickmap};
import package::trace::{UnifiedHit, make_grid, trace_grid};

// 着色法线 = **逐体素（隐式）法线**：命中体素 6 邻域占据差分的负梯度；退化时退回面法线。
//
// 向外 = 占据度的负梯度，占据度用 `world_solid`（palette 0 = 非占据）。
// 效果：球/柱/斜面这类体素化曲面按 26 邻域方向的平滑法线着色（不再是一格一格的平面），
// 而薄板仍是面法线 —— 两边的极端都各得其所。
//
// 【退化回退为什么必须是面法线】1 体素厚的薄板（地板/墙 —— 体素世界的主力几何）的 ±x/±z
// 邻接全实心、±y 全空气，二值差分三项全部抵消 → 零向量；任何**对称**模板（±2、±4…）同样
// 退化，因为薄板两侧往外都是空气 —— 几何上本来就不存在唯一方向，面法线是唯一有定义的答案。
//
// 旧实现正是在这一步硬编码 (1,0,0)：薄墙只有 +X 那侧碰巧正确，-X 侧 8 个角探针全落背面、
// 被 DDGI 的 wn 闸门剔光（Probe 品红、整面全黑），地板则一半探针被误用（漏光/发暗），太阳
// 直光也按 +X 计算。**那次失败的是"退化回退"，不是隐式法线本身** —— 所以这次按面法线退。
//
// 代价：每像素 6 次 world_solid（一次 chunk 定位 + 最多 4 层树下钻）。
fn voxel_normal(g: Grid, voxel: vec3<i32>, face_n: vec3<f32>) -> vec3<f32> {
  let x = vec3<i32>(1, 0, 0);
  let y = vec3<i32>(0, 1, 0);
  let z = vec3<i32>(0, 0, 1);
  // 占据度差分（- 侧减 + 侧）：实心一侧计数更大 → 差分**直接指向空气侧 = 向外法线**
  // （例：+X 面朝空气 → -X 是实心、+X 是空气 → d.x = 1-0 = +1 ✓）
  let d = vec3<f32>(
    f32(world_solid(g, voxel - x)) - f32(world_solid(g, voxel + x)),
    f32(world_solid(g, voxel - y)) - f32(world_solid(g, voxel + y)),
    f32(world_solid(g, voxel - z)) - f32(world_solid(g, voxel + z)),
  );
  let len = length(d);
  // 退化（薄板/实心内部）→ 退回面法线；否则归一到 26 邻域方向之一
  return select(face_n, d / len, len > 1e-3);
}

// 场景级命中：UnifiedHit + 命中 volume 的 palette 基址（dda_main 取 albedo 用）
struct SceneHit {
  uh: UnifiedHit,
  palette_base: u32,
}

// 主世界先跑 + 逐物体收缩 t_cap 取最近命中（beam 预 pass 与 dda_main 主射线共用）。
// 物体循环不被世界 miss 短路：天空背景前的物体必须可见。
// g0 = 主世界网格（由 world_raycast 统一入口提供，避免重复读 grid_descs）。
fn trace_scene(g0: Grid, origin: vec3<f32>, dir: vec3<f32>, t_cap: f32, t_min: f32, depth_cap: u32) -> SceneHit {
  var best_t = 1e+30;
  var best = SceneHit(trace_grid(g0, origin, dir, t_cap, t_min, depth_cap), 0u);
  if (best.uh.hit) {
    best_t = best.uh.t;
    best.palette_base = g0.palette_base;
  }
  let n = g.grid_count;
  for (var i: u32 = 1u; i < n; i = i + 1u) {
    let gi = make_grid(i);
    let mhi = trace_grid(gi, origin, dir, min(best_t, t_cap), t_min, depth_cap);
    if (mhi.hit && mhi.t < best_t) {
      best_t = mhi.t;
      best.uh = mhi;
      best.palette_base = gi.palette_base;
    }
  }
  return best;
}

// ============================================================================
// 世界查询统一接口（唯一入口）—— 2026-09-13 抽取，纯结构重构（渲染结果逐像素不变）
//
// 【为什么要有它】本文件曾并存多套「世界里有没有实体 / 射线是否穿过世界」的实现：
//   ① 主可见性 trace_scene→trace_grid→trace_chunk 的体素/brickmap DDA；
//   ② DDGI 探针投线 ddgi_cast（原先直接调 trace_scene）；
//   ③ 逐角几何遮挡（原先自带一套粗层预算 DDA + 硬距离/步数上限，现统一到完整层次 DDA，
//      与主 trace 同一套遍历 —— 见「逐角几何遮挡参数」常量处的说明）；
//   ④ 探针放置的占用查询（ddgi_brick_state / ddgi_cell_state_sized）。
// 它们读同一份 brickmap 数据，但起点/终点偏置、epsilon、步进规则、「实体」判定各自为政，
// 是漏光类问题反复出现的工程根因。现在**所有世界求值只走下面两个入口**；
// 以后新增调用者必须走这里，禁止再写私有遍历。
//
//   world_solid(g, voxel) -> bool                      单点占据（1³ 体素）：palette != 0 即实体
//   world_raycast(g0, origin, dir, max_t, cfg) -> WorldHit   世界求交（射线 / 线段）
//
// 【块级占用】ddgi_brick_state / ddgi_cell_state_sized 是 4³/16³/64³ 的**聚合**块查询
// （0=全空 / 1=全实 / 2=混合），其 1³ 语义与 world_solid 同源（都是 palette != 0）。
// 探针放置的「子块是否全空」正是它们的应用；它们已收口于同一份树，无需第二实现。
// ============================================================================

/// 单点占据：grid 局部体素坐标 voxel 处是否实体（palette != 0）。
/// 与探针放置的「全空/全满」判定**同一语义**；是所有点查询的唯一入口
/// （voxel_normal 6 邻域、ddgi_place_probe 的 1³ 空叶兜底）。
fn world_solid(g: Grid, voxel: vec3<i32>) -> bool {
  return sample_brickmap(g, voxel) != 0u;
}

/// 世界求交配置：用**显式字段**覆盖各调用者合理的差异，绝不让每处各写一份遍历。
/// 现在只有**完整层次 DDA**（trace_scene：主世界 + 逐物体；返回精确 t/normal/voxel/palette）：
///   · t_min：起点沿 dir 跳过的距离（beam 预 pass / 逐角遮挡起点偏置用）；
///   · depth_cap：树深度上限（3 = 全深度，即不限制）。
struct WorldRayQuery {
  t_min: f32,
  depth_cap: u32,
}

/// 完整层次 DDA 的配置构造。
fn world_cfg_full(t_min: f32, depth_cap: u32) -> WorldRayQuery {
  return WorldRayQuery(t_min, depth_cap);
}

/// 世界求交结果（完整层次 DDA，字段皆有定义）。
struct WorldHit {
  hit: bool,
  t: f32,               // 命中距离（ro 系绝对 t）；未命中 = 0
  point: vec3<f32>,     // 命中点 = origin + dir*t
  pal: u32,
  n: vec3<f32>,         // 世界空间法线
  face_id: u32,         // 命中面 0..5
  voxel: vec3<i32>,     // grid 局部命中体素
  obj_id: i32,          // -1 = 主世界，>=0 = 物体
  palette_base: u32,    // 命中 volume 的 palette 基址
}

/// 世界求交**唯一入口**。g0 = 主世界网格（make_grid(0u)），trace_scene 在其基础上追加逐物体网格。
/// trace_scene→trace_grid→trace_chunk 自带 65536 步防挂死安全网；超限返回 miss（= 保守判可见），
/// 正常射线（含逐角遮挡的 cell 对角线线段）远不会触发。
fn world_raycast(g0: Grid, origin: vec3<f32>, dir: vec3<f32>, max_t: f32, cfg: WorldRayQuery) -> WorldHit {
  let s = trace_scene(g0, origin, dir, max_t, cfg.t_min, cfg.depth_cap);
  return WorldHit(
    s.uh.hit, s.uh.t, origin + dir * s.uh.t, s.uh.pal, s.uh.n,
    s.uh.face_id, s.uh.voxel, s.uh.obj_id, s.palette_base);
}
