import package::bindings::{g};
import package::brickmap::{Grid, sample_brickmap};
import package::trace::{MediumRay, UnifiedHit, make_grid, trace_grid};

// 着色法线 = 逐体素（隐式）法线 = 可见面法线 + 该面**切平面内**的邻域倾斜。
// 倾斜量取命中体素 6 邻域占据度的负梯度（palette 0 = 非占据），但只保留与 `face_n` 垂直的那部分
// ⇒ 法线恒落在以可见面为轴的锥内（最大 54.7°），**永远不会与所见的面垂直**。
// 为什么必须这样：可见面是整数 DDA 精确给出的权威，梯度只配在它内部做倾斜。旧的「梯度直出 +
// 零梯度时 select 退回 face_n」在 1 体素薄板的切向边缘给出纯切向法线（与面内点积 = 0）⇒ 被
// GI 的 `GI_DEN_N_DOT` 判成不同面（`gi/denoise.wesl`）⇒ 丢历史 / atrous 硬切 / 上采样拒绝。
// 退化情形全部自然收敛且不需要分支：薄板内部（梯度 = 0）与介质出射面（梯度 = -face_n）都得到
// `face_n`；厚块凸棱与薄板边缘得到同一个 45° 斜法线（取值集与 `gi/consts.wesl` 的容差推导一致）。
// 成本与旧实现同阶：仍是 6 次 world_solid，省掉 length()，多 3 条 mat-vec。
fn voxel_normal(g: Grid, voxel: vec3<i32>, face_n: vec3<f32>) -> vec3<f32> {
  let x = vec3<i32>(1, 0, 0);
  let y = vec3<i32>(0, 1, 0);
  let z = vec3<i32>(0, 0, 1);
  let d = vec3<f32>(
    f32(world_solid(g, voxel - x)) - f32(world_solid(g, voxel + x)),
    f32(world_solid(g, voxel - y)) - f32(world_solid(g, voxel + y)),
    f32(world_solid(g, voxel - z)) - f32(world_solid(g, voxel + z)),
  );
  // 梯度在 grid 局部系；`face_n` 已是世界系（`grid_hit` 已乘 col0..col2）—— 先统一坐标系。
  let dw = d.x * g.col0 + d.y * g.col1 + d.z * g.col2;
  // 投影掉沿可见面的分量，只留切向倾斜。|face_n| = 1 且 t ⊥ face_n ⇒ 长度恒 >= 1，无退化点。
  let t = dw - face_n * dot(dw, face_n);
  return normalize(face_n + t);
}

// 场景级命中：`UnifiedHit` + 命中 volume 的 palette 基址（取 albedo 用）。
struct SceneHit {
  uh: UnifiedHit,
  palette_base: u32,
}

// 主世界先跑 + 逐物体收缩 t_cap 取最近命中；g0 = 主世界网格（由 world_raycast 提供）。
// 物体循环不被世界 miss 短路：天空背景前的物体必须可见。
fn trace_scene(g0: Grid, origin: vec3<f32>, dir: vec3<f32>, t_cap: f32, t_min: f32, depth_cap: u32) -> SceneHit {
  var mc_off = MediumRay(false, g0.palette_base, 0u, vec3<f32>(1.0));
  var best_t = 1e+30;
  var best = SceneHit(trace_grid(g0, origin, dir, t_cap, t_min, depth_cap, false, &mc_off), 0u);
  if (best.uh.hit) {
    best_t = best.uh.t;
    best.palette_base = g0.palette_base;
  }
  let n = g.grid_count;
  for (var i: u32 = 1u; i < n; i = i + 1u) {
    let gi = make_grid(i);
    let mhi = trace_grid(gi, origin, dir, min(best_t, t_cap), t_min, depth_cap, false, &mc_off);
    if (mhi.hit && mhi.t < best_t) {
      best_t = mhi.t;
      best.uh = mhi;
      best.palette_base = gi.palette_base;
    }
  }
  return best;
}

// 世界查询统一接口（唯一入口）：所有世界求值只走 world_solid / world_raycast，不得另写私有遍历。

/// 单点占据：grid 局部体素坐标 `voxel` 处是否实体（palette != 0），所有点查询的唯一入口。
fn world_solid(g: Grid, voxel: vec3<i32>) -> bool {
  return sample_brickmap(g, voxel) != 0u;
}

/// 世界求交配置：`t_min` = 起点沿 dir 跳过的距离；`depth_cap` = 树深度上限（3 = 全深度，不限）。
struct WorldRayQuery {
  t_min: f32,
  depth_cap: u32,
}

/// 完整层次 DDA 的配置构造。
fn world_cfg_full(t_min: f32, depth_cap: u32) -> WorldRayQuery {
  return WorldRayQuery(t_min, depth_cap);
}

/// 世界求交结果。
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

/// 世界求交唯一入口；g0 = 主世界网格（`make_grid(0u)`），在其上追加逐物体网格。
/// 自带 65536 步防挂死安全网，超限返回 miss（= 保守判可见）。
fn world_raycast(g0: Grid, origin: vec3<f32>, dir: vec3<f32>, max_t: f32, cfg: WorldRayQuery) -> WorldHit {
  let s = trace_scene(g0, origin, dir, max_t, cfg.t_min, cfg.depth_cap);
  return WorldHit(
    s.uh.hit, s.uh.t, origin + dir * s.uh.t, s.uh.pal, s.uh.n,
    s.uh.face_id, s.uh.voxel, s.uh.obj_id, s.palette_base);
}

/// 介质穿透入口（模式 1/2）：在指定 grid 内把透射材质当作可穿过，一次走完整段介质。
/// 结局：撞实体 → hit 实体；到段末端 → hit 介质材质（t/n = 出射面）；走到底 → miss；`tint`（乘性）累积透射色。
fn world_trace_medium(g: Grid, origin: vec3<f32>, dir: vec3<f32>, max_t: f32,
                      stop_on_air: bool, tint: ptr<function, vec3<f32>>) -> WorldHit {
  var mc = MediumRay(stop_on_air, g.palette_base, 0u, vec3<f32>(1.0));
  let h = trace_grid(g, origin, dir, max_t, 0.0, 3u, true, &mc);
  *tint = *tint * mc.tint;
  return WorldHit(
    h.hit, h.t, origin + dir * h.t, h.pal, h.n,
    h.face_id, h.voxel, h.obj_id, g.palette_base);
}
