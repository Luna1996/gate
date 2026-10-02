import package::bindings::{g, grid_descs};
import package::brickmap::{Grid, sample_brickmap};
import package::common::{face_normal_from_index};
import package::trace::{MediumRay, UnifiedHit, grid_is_far, make_grid, trace_grid};

/// **诊断：按卷着色**（`bindings.wesl::BASE_FLAG_VOL_TINT`）：把主射线命中来自哪一级画成固定色。
///
/// 用途：画面上出现"某一圈的楼变骨架 / 透空"时，一眼看出那圈属于哪一级（以及洞是谁留下的）。
/// 卷号从命中的 `palette_base` 反查 —— **每个 volume 的 palette 段各有自己的基址**（Rust 侧
/// `VolumesBuilder::snapshot` 按 `palette_bases` 累加），所以 `palette_base` 唯一标识一次命中来自哪个
/// volume，不需要给 `WorldHit` 加字段。
fn vol_tint_color(h: WorldHit) -> vec3<f32> {
  if (!h.hit) {
    return vec3<f32>(0.02, 0.02, 0.03); // 天空：近黑（有内容的像素才显色）
  }
  var idx = 255u;
  let n = g.grid_count;
  for (var i: u32 = 0u; i < n; i = i + 1u) {
    if (grid_descs[i].palette_base == h.palette_base) {
      idx = i;
      break;
    }
  }
  // 主世界 = 绿、L1 = 红、L2 = 蓝、L3 = 黄、其它（物体）= 洋红。
  // 取值偏亮：`write_output` 会再过一次曝光 + sRGB 编码，暗色会被压得难分辨。
  switch (idx) {
    case 0u: { return vec3<f32>(0.05, 0.80, 0.10); }
    case 1u: { return vec3<f32>(0.90, 0.06, 0.06); }
    case 2u: { return vec3<f32>(0.10, 0.16, 0.90); }
    case 3u: { return vec3<f32>(0.90, 0.85, 0.05); }
    default: { return vec3<f32>(0.20, 0.90, 0.90); }
  }
}

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

/// 命中体素**外侧**的射线起点偏移量（世界体素）= `(0.5/|n|∞ + eps)` 个**该 grid 的体素**
/// （即 `0.5/|n|∞ + eps` 乘以 `g.scale`）。
///
/// WHY 必须乘 `g.scale`：远场级的一个体素是 `scale` 个世界体素（L1/L2/L3 = 4/16/64）⇒ 只按 0.5 **世界**
/// 体素外推时，起点仍落在**命中体素内部** ⇒ 射线在 `t≈0` 自命中那个粗块：GI 的二次顶点变成它自己
/// （远场"自己照亮自己"，与近场分层）、太阳阴影靠自命中豁免恒亮、镜面照出自己。主世界 `scale = 1`
/// ⇒ 逐位不变。
///
/// 调用点一律 `voxel_center_world(g, voxel) + n * surface_offset(g, n, eps)`（**同一个构造**，
/// 别再另写一份）。基点是**体素中心**（与 `face_view_dir` / 逐面缓存同一个锚），不是命中点 ——
/// `WorldHit::point` 在逐面缓存路径上被重建成体素中心，用它会两条路径不一致。
fn surface_offset(g: Grid, n: vec3<f32>, eps: f32) -> f32 {
  return (0.5 / max(max(abs(n.x), abs(n.y)), max(abs(n.z), 1e-3)) + eps) * g.scale;
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

/// 该 grid 的**覆盖半径**（世界体素）= 它的内容真正铺到的半径（`Grid::coverage_r`，写侧
/// `VolumeGrid::set_coverage_r`）。**别用世界 AABB 的出口当接力半径** —— AABB 是盒子（索引区 /
/// 请求区，可以远大于内容）⇒ 接力点随方向变（正面对 368 m、对角 521 m）⇒ 分界线在画面上是**方**的、
/// 四个角还会留缝。半径 ⇒ 每一级是球壳、分界线是圆的。
fn coverage_r_of(g: Grid) -> f32 {
  return g.coverage_r;
}

/// 本级的**上限距离**（沿射线的 `t`）：本级内容铺在**圆柱**里 —— x/z 上是半径 `cov` 的圆盘、竖向铺满
/// 内容带（见 `VolumeGrid::set_coverage_r` 的 WHY：扁世界的地面永远在带内）。所以交接面必须是**同形
/// 的柱面**，而不是球面。
///
/// WHY 不能用球面（`t = cov`）：射线距离与水平半径的关系是 `水平半径 = t · |dir.xz|`。球面交接会让
/// **俯视的陡射线**提前失去近场（t 超过 cov），而远场的内圈是柱面（水平 163.84 m 才起）⇒ 中间一条
/// **无人区**（没人画）：实测相机越高这条带越宽，且正好落在两级的圆弧交界上。
/// 竖直方向（`|dir.xz| → 0`）柱面永不到头 ⇒ 返回 ∞，由 `t_cap` / `best_t` 兜住。
///
/// **对所有射线都用**（含次级）：把每条射线的可见范围都收在"那一级的柱面"内 —— 这既是**最省的正确
/// 边界**（内容就在柱面内，柱面外只有空步），也让次级射线（尤其**倒影**）的可见范围**与直接视野
/// 一致**：从前只裁主射线、次级射线退到"窗口 AABB"，在平整的反射面上倒影的边界会落在**窗口的方形**
/// 上（轴对齐、带直角 —— 水面上"一个矩形的一个角"就是这么来的）；现在落在与直接视野同一个柱面上
/// （曲边界，且落在"直接看得见的内容"的边界上）。
///
/// CONSTRAINT: 公式的前提是**起点在相机处**（水平半径 = `t·|dir.xz|`）—— 起点在柱面内时它给出的是
/// **略宽松**的上界（不会提前剪掉内容，只多一点步数）；[`trace_scene`] 的 `t_lo` 另按"起点是否已在
/// `shell` 之外"处理交接面。
fn level_cap(cov: f32, dir: vec3<f32>) -> f32 {
  return cov / max(length(dir.xz), 1e-4);
}

// 主世界先跑 + 逐物体/远场级收缩 t_cap 取最近命中；g0 = 主世界网格（由 world_raycast 提供）。
// 物体循环不被世界 miss 短路：天空背景前的物体必须可见。
//
// **M8 · 分壳裁剪（球面）**：远场级（`grid_is_far`）只从**上一级的覆盖半径**起参与求交 ——
// `shell` 随每级推进，每跑完一级就把它推到该级的覆盖半径（``Grid::coverage_r``）。不做这件事的直接
// 后果：远场级的格是 `4·scale` 体素的膨胀块，而它的窗口**同样包含相机**（与主世界一样钉在相机上）
// ⇒ 不裁的话近处的膨胀格会盖在近场细节前面（画面变成"远处的大方块糊住脚下"）。
// **次级射线**（`req == 0`）不受影响也不参与：`trace_grid` 自己的 `SECONDARY_REACH_CHUNKS` 把它们的
// 行程夹在 ±61 m，而第一级远场从 164 m 起 ⇒ 远场对 GI / 阴影 / 镜面天然不可见（近场已覆盖那段）。
fn trace_scene(g0: Grid, origin: vec3<f32>, dir: vec3<f32>, t_cap: f32, t_min: f32, depth_cap: u32,
               req: u32) -> SceneHit {
  var mc_off = MediumRay(false, g0.palette_base, 0u, vec3<f32>(1.0));
  // 近场这一趟里**第一次遇到"没加载"的 chunk** 的 `t`（`1e30` = 没有）。见下面的远场兜底。
  var gap: f32 = 1.0e+30;
  let n = g.grid_count;
  // `shell` = **上一级覆盖到的水平半径**（世界体素，柱面半径）—— 与内容同形，所以在每级用
  // [`level_cap`] 折算成射线距离。
  var shell = coverage_r_of(g0);
  // 二级射线的**起点水平半径**（相对相机）：见循环里那条远场门。相机位置用主世界窗口 AABB 的中心估
  // （误差 ≤ 半个 chunk，对"起点在 163 m 内 / 外"这种判断足够）。
  let cam_est = (g0.w_mn + g0.w_mx) * 0.5;
  let origin_xz = length((origin - cam_est).xz);
  // **每条射线**都裁到主世界自己的覆盖半径（见 `level_cap`）：内容之外不能再画，否则会与下一级交叠
  // （膨胀格逐像素抢近 = dither）；对次级射线同时是"与直接视野一致"的边界。
  var best_t = 1e+30;
  var g0_cap = t_cap;
  if (g0.coverage_r > 0.0) {
    g0_cap = min(t_cap, level_cap(g0.coverage_r, dir));
  }
  var best = SceneHit(
    trace_grid(g0, origin, dir, g0_cap, t_min, depth_cap, false, req, &mc_off, &gap), 0u);
  if (best.uh.hit) {
    best_t = best.uh.t;
    best.palette_base = g0.palette_base;
  }
  // **远场兜底（渐进加载的"先给糊的"）**：近场这一趟遇到了"没加载"的 chunk（`gap < 1e30`）**且近场没
  // 命中** ⇒ 从那个 `t` 起就让远场级参与，用粗级把洞填上，而不是留一片空洞。
  //
  // CONSTRAINT①：**只在近场没命中时**兜底。近场命中时把远场提前，会让远场的**膨胀格**盖在近处的细几何
  // 前面（画面变成"大方块糊住脚下"）—— 那正是分壳裁剪原本要防的。
  // CONSTRAINT②：提前量**以壳半径为界**（下面 `min(shell_cap, fb)`）⇒ 多走的行程 ≤ 近场覆盖半径，
  // 成本有界；无论洞有多大都不会让远场从 0 开始扫全窗口。
  var fb = select(1.0e+30, gap, !best.uh.hit);
  for (var i: u32 = 1u; i < n; i = i + 1u) {
    let gi = make_grid(i);
    let far = grid_is_far(i);
    // **次级射线（GI / 阴影 / 镜面，`req == 0`）只有起点在本级之内时才吃本级**：
    // 远场的格是 `4·scale` 体素的膨胀块（法线/材质都是块状的），让**近处**表面的次级射线打到它会把
    // 噪声直接打进近场 —— 实测：主世界的接力半径从 164 m 降到 46 m 后，水面的 GI 噪点明显变多，
    // 就因为 L1 落进了次级射线的 ±61 m 行程内。而"一律跳过远场"又会把**远场自己**的 GI 抽掉
    // （它的次级射线只看到近场 ⇒ 远景仍然平）。所以按**起点距离**判：起点比本级更近 ⇒ 本级不是它的
    // 环境（跳过）；起点就在本级里 ⇒ 本级正是它的环境（参与）⇒ 近场干净 + 远场有完整 GI。
    // CONSTRAINT: 不能靠"远场起点比 `SECONDARY_REACH_CHUNKS` 远"来保证 —— 接力半径改成球面后
    // 它可以是任意值（`shell` 就是本级的起始半径，故这里与它比）。
    if (far && req == 0u && origin_xz < shell) {
      continue;
    }
    // 交接面 = **柱面**（与内容同形）：`shell` 是水平半径 ⇒ 用 [`level_cap`] 折成射线距离。
    // 两级于是**互斥且密铺**：这一级画到自己的柱面为止，下一级从同一个柱面开始。
    // WHY: [`level_cap`] 的前提是**起点在相机处**（水平半径 = `t·|dir.xz|`）。玻璃状态机与反射链的起点
    // 是**命中面**，可以在远场里（水平半径已超过 `shell`）⇒ 用相机口径折算会得到一个正的距离，把
    // **起点所在的那一段整段剪掉**（远场玻璃的入射面被跳过 ⇒ 状态机落回天空）。起点已在 `shell` 之外
    // ⇒ 本级就是"起点所在的级"，从 `t_min` 起参与。主射线起点是相机（`origin_xz ≈ 0 < shell`）
    // ⇒ 恒走原路径、逐位不变。
    let shell_cap = level_cap(shell, dir);
    // 兜底时用 `min(shell_cap, fb)`：`fb == 1e30`（无兜底）⇒ 与改动前逐位相同。
    let t_lo = select(t_min, select(max(t_min, min(shell_cap, fb)), t_min, origin_xz >= shell), far);
    // CONSTRAINT: 兜底**只给第一级远场**。再往外那一级的"洞"归它上一级负责（远场各级的膨胀格互相
    // 叠加会让洞里出现"糊上加糊"，而它补的是同一片区域）。用完即失效 ⇒ 后面的级回到常规壳裁剪。
    if (far) {
      fb = 1.0e+30;
    }
    // 上限同理（物体 `coverage_r == 0` 不裁）。对起点在本级内的次级射线，这条是**略宽松的上界**
    // （见 `level_cap` 的 CONSTRAINT）⇒ 不裁内容、只省空步，并且让倒影的边界与直接视野重合。
    var t_hi = min(best_t, t_cap);
    if (gi.coverage_r > 0.0) {
      t_hi = min(t_hi, level_cap(gi.coverage_r, dir));
    }
    // **空区间早退**：本级在这条射线上的参与区间为空（近场已命中在更近处 / 超出 `t_cap`）⇒ 直接进下一级。
    // `trace_grid` 自己会在 slab 之后判 `tl1 <= tl0` 返回 miss，但那要先付世界 AABB + 局部 AABB 的
    // ~12 次除法与一次变换；这里只用两个已经算好的标量判**同一个结论**（纯等式变换：区间空 ⇒ 那次
    // 调用除了返回 miss 什么都不会做 —— 连 `mc` 的 tint 都不动，因为它在那之前就 return 了）。
    // `shell` 照常推进：与"调用了 trace_grid、它返回 miss"后的状态逐位相同。
    if (t_lo >= t_hi) {
      if (far) {
        shell = max(shell, coverage_r_of(gi));
      }
      continue;
    }
    let mhi = trace_grid(gi, origin, dir, t_hi, t_lo, depth_cap, false, req, &mc_off, &gap);
    if (mhi.hit && mhi.t < best_t) {
      best_t = mhi.t;
      best.uh = mhi;
      best.palette_base = gi.palette_base;
    }
    // 只让**远场级**推进 `shell`：物体（非流式卷）的覆盖半径没有"这一级管到多远"的语义，
    // 推进它会把后面远场级的起点抬到物体尺度上去。
    if (far) {
      shell = max(shell, coverage_r_of(gi));
    }
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
  // 介质穿透路径不需要远场兜底的 `gap`（它的调用方不消费）⇒ 传一个局部占位。
  var gap: f32 = 1.0e+30;
  let h = trace_grid(g, origin, dir, max_t, 0.0, 3u, true, 0u, &mc, &gap);
  *tint = *tint * mc.tint;
  return WorldHit(
    h.hit, h.t, origin + dir * h.t, h.pal, h.n,
    h.face_id, h.voxel, h.obj_id, g.palette_base);
}

/// **次级射线的世界查询唯一入口**（GI / 阴影 / 玻璃状态机 / 反射链）：把 [`world_trace_medium`] 的
/// "只查主世界"换成"查主射线看到的**同一个场景**"（主世界 + 全部远场级）。
///
/// WHY 必须这样：次级射线的起点是**命中面**，而远场表面（远场级里的水面 / 楼面 / 玻璃）的起点在
/// **主世界窗口之外** ⇒ 单卷版本每次求交都 miss ⇒ 那些像素的间接光退化成"只有天光"，与近场
/// （真实城市 GI）之间在级别边界上出现一条**分层**的色差。主射线一直看整个场景（[`trace_scene`]）。
///
/// 近场起点（`origin_xz < coverage_r(g0)`）走单卷路径：那时 [`trace_scene`] 的远场门
/// （`origin_xz < shell`）对每一级都成立 ⇒ 只有主世界参与 ⇒ 与逐卷循环**语义逐位相同**
/// （这里只是把那个循环省掉，近场热路径零额外开销）。
///
/// CONSTRAINT: `tint` 是**乘性**吸收 ⇒ 逐卷各乘一次、与卷序无关；每个卷用**独立**的 `MediumRay`
/// （与 [`world_trace_medium`] 同款）⇒ `last_pal` 不跨卷延续（介质段不跨卷：一块玻璃 / 一片水面
/// 落在一个卷内）。`stop_on_air` 语义与 [`world_trace_medium`] 逐字相同。
fn world_trace_medium_scene(origin: vec3<f32>, dir: vec3<f32>, max_t: f32, stop_on_air: bool,
                            tint: ptr<function, vec3<f32>>) -> WorldHit {
  let g0 = make_grid(0u);
  let cam_est = (g0.w_mn + g0.w_mx) * 0.5;
  let origin_xz = length((origin - cam_est).xz);
  if (g0.coverage_r <= 0.0 || origin_xz < g0.coverage_r) {
    return world_trace_medium(g0, origin, dir, max_t, stop_on_air, tint);
  }
  let n = g.grid_count;
  var best = world_miss();
  var best_t = 1e+30;
  // 与 [`trace_scene`] 同一套：`shell` = 上一级覆盖到的水平半径；远场级按起点距离决定是否参与。
  // 起点已在远场（上面的早退保证）⇒ 参与的都是"起点所在的那一级"及其后继；交接面在起点之前 ⇒ 从 0 起。
  // CONSTRAINT: `tint` 的逐卷累加按**卷号序**，而次级射线的行程被 `SECONDARY_REACH_CHUNKS`（±61 m）
  // 夹住、远场各级的环带又宽达数百米 ⇒ 射线实际只触及起点所在那一级的邻域，卷号序即射线序。
  var shell = coverage_r_of(g0);
  for (var i: u32 = 0u; i < n; i = i + 1u) {
    let gi = make_grid(i);
    let far = grid_is_far(i);
    if (far && origin_xz < shell) {
      // `shell` 只增 ⇒ 这一级之后的远场级必然也被挡掉（`break` 省掉它们的解码）。
      break;
    }
    var mc = MediumRay(stop_on_air, gi.palette_base, 0u, vec3<f32>(1.0));
    var gap: f32 = 1.0e+30;
    let h = trace_grid(gi, origin, dir, min(best_t, max_t), 0.0, 3u, true, 0u, &mc, &gap);
    *tint = *tint * mc.tint;
    if (h.hit && h.t < best_t) {
      best_t = h.t;
      best = WorldHit(
        h.hit, h.t, origin + dir * h.t, h.pal, h.n,
        h.face_id, h.voxel, h.obj_id, gi.palette_base);
    }
    if (far) {
      shell = max(shell, coverage_r_of(gi));
    }
  }
  return best;
}
