import package::bindings::{g};
import package::brickmap::{Grid, sample_brickmap};
import package::common::{face_normal_from_index};
import package::trace::{MediumRay, UnifiedHit, aabb_exit, grid_is_far, make_grid, trace_grid};

// 着色法线 = 逐体素（隐式）法线 = 可见面法线 + 该面**切平面内**的邻域倾斜。
// 倾斜量取命中体素邻域占据度（主世界 4 个面邻居 / 物体 6 个，见下面的成本说明）的负梯度，
// 但只保留与 `face_n` 垂直的那部分
// ⇒ 法线恒落在以可见面为轴的锥内（最大 54.7°），**永远不会与所见的面垂直**。
// 为什么必须这样：可见面是整数 DDA 精确给出的权威，梯度只配在它内部做倾斜。旧的「梯度直出 +
// 零梯度时 select 退回 face_n」在 1 体素薄板的切向边缘给出纯切向法线（与面内点积 = 0）⇒ 被
// GI 的 `GI_DEN_N_DOT` 判成不同面（`gi/denoise.wesl`）⇒ 丢历史 / atrous 硬切 / 上采样拒绝。
// 退化情形全部自然收敛且不需要分支：薄板内部（梯度 = 0）与介质出射面（梯度 = -face_n）都得到
// `face_n`；厚块凸棱与薄板边缘得到同一个 45° 斜法线（取值集与 `gi/consts.wesl` 的容差推导一致）。
// 成本：主世界（identity 变换）**4 次** world_solid，物体（任意旋转）6 次。
//   · 主世界（`obj_id < 0`，`make_grid` 保证 rot=I / scale=1 / pos=0）⇒ `face_n` 恰是 ±e_a
//     （a = face_id>>1）、`dw` 逐位等于 `d`；而下面的投影 `t = dw - face_n·dot(dw,face_n)`
//     把 d 沿 a 的那个分量**整项抵消成 0** ⇒ 沿 a 的那一对点查询对结果零贡献，整对省掉。
//     `face_id` 必须来自 DDA 的面（本工程全部调用点都如此）—— 换非轴对齐的法线会改变结果。
//   · 物体（`obj_id ≥ 0`）三列旋转任意、`face_n` 不是轴对齐 ⇒ 分量一个都不能省，照旧 6 次。
fn voxel_normal(g: Grid, voxel: vec3<i32>, face_id: u32, face_n: vec3<f32>) -> vec3<f32> {
  let x = vec3<i32>(1, 0, 0);
  let y = vec3<i32>(0, 1, 0);
  let z = vec3<i32>(0, 0, 1);
  // skip = 3 ⇒ 三个分量都要（物体）；0/1/2 ⇒ 该轴那一对点查询整对跳过（主世界）
  let skip = select(3u, face_id >> 1u, g.obj_id < 0);
  var d = vec3<f32>(0.0);
  if (skip != 0u) { d.x = f32(world_solid(g, voxel - x)) - f32(world_solid(g, voxel + x)); }
  if (skip != 1u) { d.y = f32(world_solid(g, voxel - y)) - f32(world_solid(g, voxel + y)); }
  if (skip != 2u) { d.z = f32(world_solid(g, voxel - z)) - f32(world_solid(g, voxel + z)); }
  // 梯度在 grid 局部系；`face_n` 已是世界系（`grid_hit` 已乘 col0..col2）—— 先统一坐标系。
  // 主世界三列 = 单位阵 ⇒ 局部系就是世界系，mat-vec 也一并省掉（逐位等价：d 分量恒为 -1/0/1）。
  var dw = d;
  if (skip == 3u) { dw = d.x * g.col0 + d.y * g.col1 + d.z * g.col2; }
  // 投影掉沿可见面的分量，只留切向倾斜。|face_n| = 1 且 t ⊥ face_n ⇒ 长度恒 >= 1，无退化点。
  let t = dw - face_n * dot(dw, face_n);
  return normalize(face_n + t);
}

/// 命中体素的**世界空间中心**（`voxel + 0.5` 经该 grid 的变换）—— 它是"这个体素在世界里的锚点"，
/// 主 pass 用它当阴影与反射射线的起点（`main.wesl::dda_main` 的 `p_voxel`）。
///
/// 表达式与 `main.wesl::dda_main` 里原有那行 `p_voxel` **逐位相同**（同一个 `dot` 形式 ⇒ 取值口径不变；
/// 主世界 identity 下就是 `voxel + 0.5`，物体下随变换走 —— 物体路径当前没有调用点，
/// `Volumes::add_object` 在应用里未被调用）。
///
/// **它就是视向量的锚点**（2026-09-21 起）：全部着色点走 `common.wesl::face_view_dir`（本函数的
/// 结果当第二个实参）⇒ 同一个体素面内所有像素拿到同一个 `v` ⇒ 反光/高光逐体素成块
/// （为什么按面、代价与两次实测结论在 `face_view_dir` 的注释里）。
///
/// **为什么求中心要用 DDA 给的整数 `voxel`、而不是"命中点 + `floor`"**：DDA 给的 `voxel` 是**整数**
/// （权威），而命中点的法线轴分量恰好落在整数面平面上 ⇒ 浮点误差会让 `floor(命中点)` 取到相邻
/// （通常是空气）体素，同一面内逐像素翻转。
fn voxel_center_world(g: Grid, voxel: vec3<i32>) -> vec3<f32> {
  let vc = vec3<f32>(voxel) + vec3<f32>(0.5);
  return g.pos + vec3<f32>(dot(vc, g.col0), dot(vc, g.col1), dot(vc, g.col2)) * g.scale;
}

/// 面号 → **世界空间面法线**（= `trace.wesl::grid_hit` 里那两行的提取，表达式逐位相同）。
/// 它是 `voxel_normal` 的第一个实参，也是**逐面着色量子**里唯一带面号的那一项：
/// 材质/视向量都只锚在体素中心，只有它区分同一个体素的不同可见面（同一个面的像素才同色）。
/// 逐面 pass（`dda_face_main`）从面键重建命中时用它 —— 主世界 identity ⇒ 就是轴单位向量。
fn face_normal_world(g: Grid, face_id: u32) -> vec3<f32> {
  let nl = face_normal_from_index(face_id);
  return normalize(nl.x * g.col0 + nl.y * g.col1 + nl.z * g.col2);
}

// 场景级命中：`UnifiedHit` + 命中 volume 的 palette 基址（取 albedo 用）。
struct SceneHit {
  uh: UnifiedHit,
  palette_base: u32,
}

// 主世界先跑 + 逐物体/远场级收缩 t_cap 取最近命中；g0 = 主世界网格（由 world_raycast 提供）。
// 物体循环不被世界 miss 短路：天空背景前的物体必须可见。
//
// **M8 · 分壳裁剪**：远场级（`grid_is_far`）只从**上一级的覆盖半径**起参与求交 —— `shell` 随每级推进，
// 每跑完一级就把它推到该级世界 AABB 的出口（`aabb_exit`）。不做这件事的直接后果：远场级的格是
// `4·scale` 体素的膨胀块，而它的窗口**同样包含相机**（与主世界一样钉在相机上）⇒ 不裁的话
// 近处的膨胀格会盖在近场细节前面（画面变成"远处的大方块糊住脚下"）。
// **次级射线**（`req == 0`）不受影响也不参与：`trace_grid` 自己的 `SECONDARY_REACH_CHUNKS` 把它们的
// 行程夹在 ±61 m，而第一级远场从 164 m 起 ⇒ 远场对 GI / 阴影 / 镜面天然不可见（近场已覆盖那段）。
fn trace_scene(g0: Grid, origin: vec3<f32>, dir: vec3<f32>, t_cap: f32, t_min: f32, depth_cap: u32,
               req: u32) -> SceneHit {
  var mc_off = MediumRay(false, g0.palette_base, 0u, vec3<f32>(1.0));
  var best_t = 1e+30;
  var best = SceneHit(trace_grid(g0, origin, dir, t_cap, t_min, depth_cap, false, req, &mc_off), 0u);
  if (best.uh.hit) {
    best_t = best.uh.t;
    best.palette_base = g0.palette_base;
  }
  let n = g.grid_count;
  // **M8 分壳**：`shell` = "上一级覆盖到哪"，每跑完一级就推进到该级世界 AABB 的出口；远场级从它起画。
  // CONSTRAINT: **必须从"第 0 级（主世界）的出口"起步、且每一级都推进** —— 只推进远场级的话，
  // L1 拿到的 `shell` 还是 `t_min`（≈ 0）⇒ 它的 1.28 m 膨胀格从相机脸上就开始画：近场细节被一片
  // "材质随机、颜色花"的大方块糊住（实测现象：`docs/editable-gigavoxel.md` §9 M8 的"模型都不对"）。
  var shell = aabb_exit(g0.w_mn, g0.w_mx, origin, dir);
  for (var i: u32 = 1u; i < n; i = i + 1u) {
    let gi = make_grid(i);
    let far = grid_is_far(i);
    let t_lo = select(t_min, max(t_min, shell), far);
    let mhi = trace_grid(gi, origin, dir, min(best_t, t_cap), t_lo, depth_cap, false, req, &mc_off);
    if (mhi.hit && mhi.t < best_t) {
      best_t = mhi.t;
      best.uh = mhi;
      best.palette_base = gi.palette_base;
    }
    // 推进放在 `if (far)` 之外：命中与否都要推进（下一级从**这一级**的覆盖半径起）
    shell = max(shell, aabb_exit(gi.w_mn, gi.w_mx, origin, dir));
  }
  return best;
}

// 世界查询统一接口（唯一入口）：所有世界求值只走 world_solid / world_raycast，不得另写私有遍历。

/// 单点占据：grid 局部体素坐标 `voxel` 处是否实体（palette != 0），所有点查询的唯一入口。
fn world_solid(g: Grid, voxel: vec3<i32>) -> bool {
  return sample_brickmap(g, voxel) != 0u;
}

/// 相机所在体素（主世界，整数格 = `floor` 相机位置）的 palette 条目；0 = 相机在空气里。
/// WHY: 未做相机碰撞检测 ⇒ 相机可以陷在实心体素内部。那时主射线在 t≈0 命中相机自己所在的体素，
/// 全屏像素共享同一个面键（画面退化成几块平色），而 GI / 玻璃状态机仍在按这个假面运行。
/// 该状态没有正确的着色可言 ⇒ `dda_main` 与 `gi_main` 都用本判据整屏早退（唯一判据，不分叉）。
fn camera_solid_pal(origin_voxel: vec3<f32>) -> u32 {
  return sample_brickmap(make_grid(0u), vec3<i32>(floor(origin_voxel)));
}

/// 世界求交配置：`t_min` = 起点沿 dir 跳过的距离；`depth_cap` = 早停闸门深度（`trace_chunk_opaque`
/// 的 `gd >= depth_cap` 分支，`gd = 3 - level`）。
///
/// CONSTRAINT: 对**着色射线**只有 `depth_cap = 3`（不早停）合法 —— `depth_cap <= 2` 的早停会把本节点的
/// `pal` 当成分裂子块的颜色返回，而那一位不是该子块的颜色（docs/editable-gigavoxel.md 硬不变量 4）。
/// 目前唯一使用者是 beam 预 pass：它只消费 `t`，且该早停给出的是保守下界（`t` 只会更近）。
/// 上层三档 LOD 要复用这条闸门，须先有 M2 的代表色字段（每节点一个可增量维护的平均色）。
struct WorldRayQuery {
  t_min: f32,
  depth_cap: u32,
  /// 本 pass 是否允许发 ray-guided 请求（M4；见 `trace.wesl::REQ_ENABLE` 与 `world_cfg_primary`）
  req: u32,
}

/// 完整层次 DDA 的配置构造。`req = 0`：**次级射线**（阴影 / 反射 / GI / 光柱）不发请求 ——
/// 主射线优先，正因为它们的像素才代表"用户真的在看这里"。
fn world_cfg_full(t_min: f32, depth_cap: u32) -> WorldRayQuery {
  return WorldRayQuery(t_min, depth_cap, 0u);
}

/// 同 [`world_cfg_full`]，但**允许发 ray-guided 请求**：只给主着色 pass 用（`req = 1`）。
fn world_cfg_primary(t_min: f32, depth_cap: u32) -> WorldRayQuery {
  return WorldRayQuery(t_min, depth_cap, 1u);
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

/// 未命中（天空 / 早退）的 `WorldHit`：唯一构造点，字段顺序只在这里与结构体同步。
fn world_miss() -> WorldHit {
  return WorldHit(false, 0.0, vec3<f32>(0.0), 0u, vec3<f32>(0.0), 0u, vec3<i32>(0), -1, 0u);
}

/// 世界求交唯一入口；g0 = 主世界网格（`make_grid(0u)`），在其上追加逐物体网格。
/// 自带 65536 步防挂死安全网，超限返回 miss（= 保守判可见）。
fn world_raycast(g0: Grid, origin: vec3<f32>, dir: vec3<f32>, max_t: f32, cfg: WorldRayQuery) -> WorldHit {
  let s = trace_scene(g0, origin, dir, max_t, cfg.t_min, cfg.depth_cap, cfg.req);
  return WorldHit(
    s.uh.hit, s.uh.t, origin + dir * s.uh.t, s.uh.pal, s.uh.n,
    s.uh.face_id, s.uh.voxel, s.uh.obj_id, s.palette_base);
}

/// 介质穿透入口（模式 1/2）：在指定 grid 内把透射材质当作可穿过，一次走完整段介质。
/// 结局：撞实体 → hit 实体；到段末端 → hit 介质材质（t/n = 出射面）；走到底 → miss；
/// `tint`（乘性）累积透射色 —— **按"介质段"一次**：每进入**一段连续的同一材质**乘一次该材质的
/// 透过率，**与厚度/格数/距离无关**（见 `trace.wesl::pass_through_medium` 与 `MediumRay` 的头注释）。
fn world_trace_medium(g: Grid, origin: vec3<f32>, dir: vec3<f32>, max_t: f32,
                      stop_on_air: bool, tint: ptr<function, vec3<f32>>) -> WorldHit {
  var mc = MediumRay(stop_on_air, g.palette_base, 0u, vec3<f32>(1.0));
  let h = trace_grid(g, origin, dir, max_t, 0.0, 3u, true, 0u, &mc);
  *tint = *tint * mc.tint;
  return WorldHit(
    h.hit, h.t, origin + dir * h.t, h.pal, h.n,
    h.face_id, h.voxel, h.obj_id, g.palette_base);
}
