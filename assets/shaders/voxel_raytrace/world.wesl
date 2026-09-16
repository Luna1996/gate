import package::bindings::{g};
import package::brickmap::{Grid, sample_brickmap};
import package::trace::{UnifiedHit, make_grid, trace_grid};

// 着色法线 = 逐体素（隐式）法线：命中体素 6 邻域占据差分的负梯度，退化时退回面法线。
//
// 向外 = 占据度（`world_solid`，palette 0 = 非占据）的负梯度，故球/柱/斜面按 26 邻域方向平滑着色。
//
// 【退化回退必须是面法线】1 体素厚的薄板（地板/墙）的 ±x/±z 邻接全实心、±y 全空气，
// 二值差分三项全部抵消 → 零向量；任何对称模板（±2、±4…）同样退化 —— 几何上本就不存在
// 唯一方向，面法线是唯一有定义的答案。
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
// g0 = 主世界网格（由 world_raycast 入口提供，避免重复读 grid_descs）。
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
// 世界查询统一接口（唯一入口）：所有世界求值只走下面两个函数，新增调用者不得再写私有遍历。
//
//   world_solid(g, voxel) -> bool                    单点占据（1³ 体素）：palette != 0 即实体
//   world_raycast(g0, origin, dir, max_t, cfg) -> WorldHit   世界求交（射线 / 线段）
// ============================================================================

/// 单点占据：grid 局部体素坐标 voxel 处是否实体（palette != 0），是所有点查询的唯一入口
/// （voxel_normal 6 邻域、探针放置的空叶判定等），与块级查询的 1³ 语义同源。
fn world_solid(g: Grid, voxel: vec3<i32>) -> bool {
  return sample_brickmap(g, voxel) != 0u;
}

/// 世界求交配置。目前只有完整层次 DDA（trace_scene：主世界 + 逐物体，
/// 返回精确 t/normal/voxel/palette）：
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

/// 世界求交唯一入口。g0 = 主世界网格（make_grid(0u)），trace_scene 在其基础上追加逐物体网格。
/// trace_scene→trace_grid→trace_chunk 自带 65536 步防挂死安全网；超限返回 miss（= 保守判可见），
/// 正常射线远不会触发。
fn world_raycast(g0: Grid, origin: vec3<f32>, dir: vec3<f32>, max_t: f32, cfg: WorldRayQuery) -> WorldHit {
  let s = trace_scene(g0, origin, dir, max_t, cfg.t_min, cfg.depth_cap);
  return WorldHit(
    s.uh.hit, s.uh.t, origin + dir * s.uh.t, s.uh.pal, s.uh.n,
    s.uh.face_id, s.uh.voxel, s.uh.obj_id, s.palette_base);
}
