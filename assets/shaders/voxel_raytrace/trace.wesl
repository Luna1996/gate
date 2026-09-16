import package::bindings::{b_leaves, b_palette, b_struct, g, grid_descs, view_u};
import package::brickmap::{Grid};
import package::common::{CHUNK_INDEX_CAP, CHUNK_SIZE, DIR_LUT, NODE_FIXED_WORDS, face_normal_from_index, palette_transmission};

// 层级命中记录：t 为 ro 系绝对 t；face_id 0..5 = ±xyz 六面（命中面法线索引）。
//   pre-check 命中（射线起点在固体 leaf 内，相机在体内 UB）时 face_id 由调用方用
//   normalize(-rd) 反推（首 chunk entry_face）。
// voxel：命中固体体素 chunk 局部 voxel 整数坐标 —— DDA 步进为整数加法，无浮点噪声；
//   着色阶段直接消费，不要用「命中点 ± 法线半步」启发式重建（棱边/UB face 下会选错邻体素）。
struct VoxelHit {
  hit: bool,
  t: f32,
  pal: u32,
  face_id: u32,
  voxel: vec3<i32>,
}

// ============================================================================
// 介质穿透（透明/玻璃）：把"透射材质"当作可穿过的介质，**在 DDA 内部**一路走完
// ============================================================================
//
// 为什么必须做进 DDA：透射体素在树里是**同材质的 uniform 节点**（4³/16³/64³），
// 穿透就是让 DDA 用"整块跳过"的既有快路径跨过去 —— 一块 31³ 玻璃只需十几次步进，
// 而"每命中一格就重跑一次完整 trace"要几十次 trace（每格都要重做 slab + chunk 间
// A&W + 树下降）。这是性能差一个数量级的根源。
//
// `penetrate` **不在本结构里**，它是按值参数（见 `trace_chunk`）：不透明路径必须能**完全绕开**
// 本结构 —— 只要在最热循环里读一次指针目标，取地址就会把这个状态挤进 local memory，
// 每步退化成一次 local load（实测帧率腰斩就是这条路）。调用点统一写成 `if (penetrate) { ... }`，
// `penetrate = false` 时**一次都不碰** mc。
//
// 模式由 `penetrate` / `stop_on_air` 组合决定（不传这两个开关时行为与改造前逐位一致）：
//   · 正常（penetrate=false）            透射材质 = 实体命中（找**入射面**用这个）
//   · 介质内（penetrate=true, stop=true）  透射材质穿过；遇到空气 = **出射面**，停下返回
//   · 穿透（penetrate=true, stop=false）   透射材质穿过；空气也穿过（阴影射线：一路吃到遮挡）
//
// `tint` 是**沿途累积的透射色**（乘性，出射后由调用方乘进 throughput）；
// `last_pal` 是"当前所在介质段的材质"（0 = 不在介质内），用于**同一段只染色一次**
// —— 一个连续的、同材质的透明区域算一次事件，与厚度无关（Douglas 的 absorptive 语义）。
struct MediumRay {
  stop_on_air: bool,
  palette_base: u32,
  last_pal: u32,
  tint: vec3<f32>,
}

// brick 缓存槽：一层分裂节点常驻（对应 CPU BrickCpu / Douglas BrickMaskEntry）
struct Brick {
  addr: u32,  // 节点绝对字址（b_struct）
  mask: u64,  // 64bit 分裂掩码（bit=1 = 子块分裂；bit=0 = 统一子块，色=pal）
  pal: u32,   // 节点 palette（统一子块颜色，0=空气）
}

/// 读一个 palette 槽的介质属性：`rgb` = 透射色（乘性），`a` = 不透明度。
///   a < 1 → 可穿透介质（透射率 > 0）；a == 1 → 完全不透，**不是介质**，按不透明表面处理。
/// 透射色 = mix(白, albedo, a)：全透 → 无色（纯玻璃）；越不透明 → 越显自身颜色（有色玻璃）。
/// 一次读两个 word（与 `wire.rs::pack_palette_entry` 同布局；见 common.wesl 的 palette_*）。
fn medium_of(palette_base: u32, pal: u32) -> vec4<f32> {
  let w0 = b_palette[palette_base + pal * 2u];
  let srgb = vec3<f32>(f32(w0 & 0xFFu), f32((w0 >> 8u) & 0xFFu), f32((w0 >> 16u) & 0xFFu)) / 255.0;
  let a = 1.0 - f32((b_palette[palette_base + pal * 2u + 1u] >> 8u) & 0xFFu) / 255.0;
  // gamma 2.0 近似（x²）代替精确 sRGB→linear：它只是**乘性染色**，而本函数会被内联进
  // `trace_chunk` 的热路径多处 —— 精确转换（3×pow + 3×select）在这里不值得。
  return vec4<f32>(mix(vec3<f32>(1.0), srgb * srgb, a), a);
}

/// 介质穿透：`pal` 是可穿透介质 → 累积透射色（**同一段只染一次**）并返回 true
/// （调用方按"空气"继续步进）；完全不透 → false（调用方按实体命中返回）。
/// **只在 `penetrate = true` 时才会被调用**（调用点用它做短路）。
fn pass_through_medium(mc: ptr<function, MediumRay>, pal: u32) -> bool {
  // 同一介质段（材质没变）→ 直接放行。穿越大块时绝大多数步进走这条，
  // 省掉每次两步 palette load（`medium_of` 要读 word0 + word1）。
  if ((*mc).last_pal == pal) { return true; }
  let m = medium_of((*mc).palette_base, pal);
  if (m.a >= 1.0) { return false; }
  (*mc).tint = (*mc).tint * m.rgb;
  (*mc).last_pal = pal;
  return true;
}

/// 介质段内撞到空气 ⇒ 这里就是**出射面**：返回 true，调用方带回 (cur_t, face)。
/// **只在 `penetrate = true` 时才会被调用**（调用点用它做短路）。
fn medium_exit(mc: ptr<function, MediumRay>) -> bool {
  return (*mc).stop_on_air && ((*mc).last_pal != 0u);
}

// 单 chunk 内层级遍历。chunk_base = 根节点绝对字址；chunk_min = chunk 原点（局部 voxel）。
// 射线段 [t0, t1] 与返回的 t 均为 ro 系绝对 t；entry_face = 进入本 chunk 的面（首 chunk 由
// 调用方用 normalize(-rd) 兜底，后续 chunk 为跨 chunk 面）；走出 chunk 未命中 → hit=false。
// depth_cap（beam 保守模式）：gate 深度 d=(3-level) 的分裂子块且 d>=depth_cap 时返回子块入口 t
//   （保守下界）；主/阴影 pass depth_cap=3 恒不触发（d<=2）。
// split 子节点一律下钻；uniform 子节点由 c_mask==0 快路径以精确 palette 早停。
// ============================================================================
// trace_chunk 的两份特化（shader 界常见的"用代码量换性能"）
// ============================================================================
//
// **为什么不合并成一个带 `penetrate: bool` 参数的函数**：实测过 —— 只要介质代码存在于这个
// 函数体里，即便 `penetrate` 全程为 `false`、一次都不执行，编译器也无法常量折叠它，介质分支
// 的寄存器与代码体积照付。实测把 `penetrate` 换成字面量 `false`（介质代码仍在、逻辑上全短路）
// 帧率就从 85~120 回到 130~150。这就是"最热函数里多 2 个栈变量、多 7 条表达式"的代价。
//
// WESL 没有可用的编译期特化：`generics` 对标量参数是 `todo!("literal generics")`（只能用于
// 类型参数）；`@if` 只能按模块级常量裁剪，做不到"同一个 pass 里两条特化并存"。故手写两份 ——
// 两者仅差 **2 处出口判定**，其余 200 行逐字相同。
//
//   · [`trace_chunk_opaque`]：不透明路径（主射线 / 阴影 / DDGI cast / beam）—— 与引入介质
//     之前逐位一致，参数最少、无任何介质痕迹；
//   · [`trace_chunk`]：穿透路径（[`world_trace_medium`]）—— 把可穿透介质当空气越过并累积透射色。

// 单 chunk 内层级遍历（**不透明版**）。chunk_base = 根节点绝对字址；chunk_min = chunk 原点（局部 voxel）。
// 射线段 [t0, t1] 与返回的 t 均为 ro 系绝对 t；entry_face = 进入本 chunk 的面（首 chunk 由
// 调用方用 normalize(-rd) 兜底，后续 chunk 为跨 chunk 面）；走出 chunk 未命中 → hit=false。
// depth_cap（beam 保守模式）：gate 深度 d=(3-level) 的分裂子块且 d>=depth_cap 时返回子块入口 t
//   （保守下界）；主/阴影 pass depth_cap=3 恒不触发（d<=2）。
// split 子节点一律下钻；uniform 子节点由 c_mask==0 快路径以精确 palette 早停。
fn trace_chunk_opaque(chunk_base: u32, chunk_min: vec3<f32>,
                      ro: vec3<f32>, rd: vec3<f32>, sign_v: vec3<i32>,
                      t0: f32, t1: f32, entry_face: u32, depth_cap: u32) -> VoxelHit {
  // 擦边退化（t0>=t1：射线只蹭到 chunk 边界）→ 无体素内部可穿过，直接 miss
  if (t0 >= t1) { return VoxelHit(false, 0.0, 0u, 0u, vec3<i32>(0)); }
  // chunk 局部 voxel 坐标（chunk 原点 = 0）；t 仍是 ro 系绝对 t
  let ro_c = ro - chunk_min;
  // side 距离/步长增量改用乘法（每外层省 3 个 fdiv）
  let inv_rd = 1.0 / rd;
  // 循环不变量预计算（sign_v / rd 在整个 trace_chunk 内不变）
  let abs_inv_rd = abs(inv_rd);                  // 每轮复用，省 abs()
  let axis_on = abs(rd) > vec3<f32>(1e-30);      // 该轴是否可步进（rd 分量为 0 时禁用）
  let oct = select(0u, 1u, sign_v.x >= 0)        // 方向八分体索引（b_leaves 查表用）
    | (select(0u, 1u, sign_v.y >= 0) << 1u)
    | (select(0u, 1u, sign_v.z >= 0) << 2u);
  let dir_side = select(vec3<f32>(0.0), vec3<f32>(1.0), sign_v >= vec3<i32>(0)); // 边界基址偏移
  let face_base = vec3<u32>(                     // 每轴两个面的 face id 基（0/1、2/3、4/5）
    select(1u, 0u, sign_v.x >= 0),
    select(1u, 0u, sign_v.y >= 0),
    select(1u, 0u, sign_v.z >= 0));
  var bricks: array<Brick, 4u>;
  let r_ml = b_struct[chunk_base];
  let r_mh = b_struct[chunk_base + 1u];
  let r_pal = b_struct[chunk_base + 2u] & 0xFFFFu;  // 节点字低 16 位 = uniform 色（16 位材质索引）
  bricks[3u] = Brick(chunk_base, (u64(r_mh) << 32u) | u64(r_ml), r_pal);
  var level: u32 = 3u;
  // 当前体素（chunk 局部 voxel 整数坐标，0..255；跨出 chunk 的步进瞬态可达 -1/256）
  let p0 = ro_c + rd * t0;
  var v = clamp(vec3<i32>(floor(p0)), vec3<i32>(0), vec3<i32>(255));
  var cur_t = t0;
  var face = entry_face;
  // 防挂死安全网：迭代上限（真实场景每 chunk 仅几十~几百次）
  var budget: u32 = 65536u;
  let lut_disable = view_u.lod.w > 0.5;
  loop {
    if (budget == 0u) { break; }
    budget = budget - 1u;
    if (level > 3u) { return VoxelHit(false, 0.0, 0u, 0u, vec3<i32>(0)); }
    // ---- traverse：从当前 level 下钻到 v 处内容（Douglas traverse_bit_set）----
    loop {
      let b = bricks[level];
      // WGSL 移位 RHS 必须 u32
      let sh = vec3<u32>(level * 2u);
      let cell = vec3<i32>(v >> sh) & vec3<i32>(3); // &3 恒为 0..3
      let idx = u32(cell.z * 16 + cell.y * 4 + cell.x);
      if (level == 0u) {
        // 叶节点 inline palette：bit=1（非空体素）才 load inline word 取色；
        // bit=0 空气体素零 load（mask 在手）。
        // 每字 2 个体素（16 位索引）：字序号 = idx >> 1u，半字 = (idx & 1u) * 16u。
        if ((b.mask & (u64(1) << idx)) != u64(0)) {
          let w = b_struct[b.addr + NODE_FIXED_WORDS + (idx >> 1u)];
          let leaf_pal = (w >> ((idx & 1u) * 16u)) & 0xFFFFu;
          if (leaf_pal != 0u) { return VoxelHit(true, cur_t, leaf_pal, face, v); }
        }
        break; // 空气 leaf
      }
      let bit = u64(1) << idx;
      if ((b.mask & bit) == u64(0)) {
        // 统一子块：颜色 = 节点 palette（0=空气）
        if (b.pal != 0u) { return VoxelHit(true, cur_t, b.pal, face, v); }
        break; // 空气统一子块
      }
      // depth_cap（beam 保守）：gate 深度 d=3-level 的分裂子块到达 cap → 子块入口 t
      let gd = 3u - level;
      if (gd >= depth_cap) { return VoxelHit(true, cur_t, b.pal, face, v); }
      // 分裂子块 → popcount 定位 child offset
      let below = b.mask & (bit - u64(1));
      let pop = countOneBits(u32(below)) + countOneBits(u32(below >> 32u));
      let child_addr = chunk_base + b_struct[b.addr + NODE_FIXED_WORDS + pop];
      let c_ml = b_struct[child_addr];
      let c_mh = b_struct[child_addr + 1u];
      let c_pw = b_struct[child_addr + 2u];
      let c_pal = c_pw & 0xFFFFu;  // 节点字低 16 位 = uniform 色
      let c_mask = (u64(c_mh) << 32u) | u64(c_ml);
      // 统一子节点快路径：wire 任意层的分裂位都可能指向 3 字统一节点（mask=0，pal 直决；
      // 叶层统一节点无 inline 16 字，禁读 addr+3 之后）
      if (c_mask == u64(0)) {
        if (c_pal != 0u) { return VoxelHit(true, cur_t, c_pal, face, v); }
        break;
      }
      level = level - 1u;
      bricks[level] = Brick(child_addr, c_mask, c_pal);
    }
    // ---- 整砖 LUT 跳过（Douglas march mask skip；仅 palette==0 节点安全）----
    // 当前 brick 从当前格/象限无可达分裂子块 → 直接升一层跨步（不重新 traverse：
    // 父帧当前格就是本砖，重 traverse 会原地再下钻；父层 dda 先跨步出本砖再判位）
    {
      let b = bricks[level];
      if (b.pal == 0u) {
        let sh = vec3<u32>(level * 2u);
        let ecell = vec3<i32>(v >> sh) & vec3<i32>(3);
        let entry_i = u32(ecell.z * 16 + ecell.y * 4 + ecell.x);
        let lut_base = min(oct * 64u + entry_i, 511u);
        var reach = b_leaves[lut_base];
        reach = select(reach, ~u64(0), lut_disable);
        if ((b.mask & reach) == u64(0)) {
          level = level + 1u;
          if (level > 3u) { return VoxelHit(false, 0.0, 0u, 0u, vec3<i32>(0)); }
        }
      }
    }
    // ---- v 处为空气：当前 level brick 内 DDA（Douglas dda）----
    let log2 = level * 2u;
    let s = 1i << (level * 2u); // 子块边长 voxel：1/4/16/64
    let s_mask = ~(s - 1i);    // 对齐掩码（two's complement: ~(s-1) = -s）
    // side_distance_for_ray：v 对齐到 s 的基址；正向 → 基址+s，负向 → 基址
    let base_v = v & vec3<i32>(s_mask);
    let boundary = vec3<f32>(base_v) + dir_side * f32(s);
    var side = vec3<f32>(1e+30);
    side = select(side, (boundary - ro_c) * inv_rd, axis_on);
    side = max(side, vec3<f32>(cur_t));
    // 每轴步长 t 增量（level 不变则不变）：inner 里 O(1) 加法
    let step_inc = f32(s) * abs_inv_rd;
    var step_axis: u32 = 0u;
    var changed = false;
    loop {
      // 选最近边界轴
      var mn: u32;
      if (side.x <= side.y && side.x <= side.z) { mn = 0u; }
      else if (side.y <= side.z) { mn = 1u; }
      else { mn = 2u; }
      step_axis = mn;
      cur_t = side[mn];
      if (cur_t >= t1) { return VoxelHit(false, 0.0, 0u, 0u, vec3<i32>(0)); } // 段内再无子块可入
      let old_cell = (v[mn] >> log2) & 3;
      // 沿 mn 轴整子块跨越
      v[mn] = v[mn] + sign_v[mn] * s;
      side[mn] = side[mn] + step_inc[mn];
      face = mn * 2u + face_base[mn];
      // 跨出 brick（4 子块）？正向往 3→外、负向往 0→外
      let crossed = select(old_cell == 0, old_cell == 3, sign_v[mn] >= 0);
      if (crossed) { changed = true; break; }
      // 新子块内容：level 0 查 inline palette（仅 mask!=0 的 inline 叶）；
      // level 1..3 = 分裂位或节点统一实体色。
      // 命中直接返回（cur_t=进入距离、face=进入面）；仅「分裂子块」回 traverse 下钻。
      let b = bricks[level];
      let sh2 = vec3<u32>(log2);
      let cell = vec3<i32>(v >> sh2) & vec3<i32>(3);
      let idx = u32(cell.z * 16 + cell.y * 4 + cell.x);
      if (level == 0u) {
        // bit=1（非空体素）才 load inline word；bit=0 空气体素零 load
        if (b.mask != u64(0) && (b.mask & (u64(1) << idx)) != u64(0)) {
          let w = b_struct[b.addr + NODE_FIXED_WORDS + (idx >> 1u)];
          let dp = (w >> ((idx & 1u) * 16u)) & 0xFFFFu;
          if (dp != 0u) { return VoxelHit(true, cur_t, dp, face, v); }
        } else if (b.mask == u64(0) && b.pal != 0u) {
          // 防御：uniform 叶（正常下钻快路径已处理）
          return VoxelHit(true, cur_t, b.pal, face, v);
        }
      } else {
        let mb = (b.mask & (u64(1) << idx)) != u64(0);
        if (mb) { break; } // 分裂子块 → 回 traverse 下钻
        if (b.pal != 0u) { return VoxelHit(true, cur_t, b.pal, face, v); } // 统一实体
      }
      // 空气子块 → 回 loop 顶重选 mn 轴继续
    }
    if (changed) {
      level = level + 1u;
      if (level > 3u) { return VoxelHit(false, 0.0, 0u, 0u, vec3<i32>(0)); }
    }
    // ---- firstTrailingBit 层级自适应跨级跳（Douglas march 尾部）----
    // 步进轴新坐标的尾随零位 = 对齐 run 长度：正向 comp=对齐基址（tz 直接读），
    // 负向 comp=区域尾址（基址|~mask，+1 后读 tz）。tz>>1 = 可跨步的最粗 level。
    let positive = sign_v[step_axis] >= 0;
    let cur_log2 = level * 2u;
    let m: u32 = 0xFFFFFFFFu << cur_log2;
    let not_m = ~m;                              // = (1<<cur_log2)-1，复用于 region_max
    let vmin_u = u32(v[step_axis]); // i32→u32 位环绕（负值公式自然处理）
    var comp: u32;
    if (positive) { comp = vmin_u & m; } else { comp = (vmin_u & m) | not_m; }
    let tz_i = firstTrailingBit(comp + select(1u, 0u, positive)); // 0 → -1
    var new_level: u32 = 0xFFFFFFFFu;
    if (tz_i >= 0) { new_level = u32(tz_i) >> 1u; }
    level = max(level, new_level);
    if (level > 3u) { return VoxelHit(false, 0.0, 0u, 0u, vec3<i32>(0)); } // 跨出 chunk（tz≥8）
    // 对齐快照：v 钳到 cur_t 射线点所在的当前 level 区域，步进轴取精确边界整数
    // （其余轴按射线实际位置吸附，消除只沿单轴步进的漂移）
    let mi = i32(m);
    let base = v & vec3<i32>(mi);
    let p = ro_c + rd * cur_t;
    let region_max = base + vec3<i32>(i32(not_m));
    v = clamp(vec3<i32>(floor(p)), base, region_max);
    v[step_axis] = i32(comp);
  }
  return VoxelHit(false, 0.0, 0u, 0u, vec3<i32>(0));
}

// 单 chunk 内层级遍历（**介质穿透版**）。调用点恒传字面量穿透语义，故不再有 `penetrate` 开关。
// 命中 / 穿透 / 出射面判定只在两处 **唯一出口** 做（理由见 [`MediumRay`]：分支里重复判定会被
// naga 提升成函数级变量，即 GPU 上的 local memory / 栈）。
// mc：介质穿透上下文（见 [`MediumRay`]）。
fn trace_chunk(chunk_base: u32, chunk_min: vec3<f32>,
               ro: vec3<f32>, rd: vec3<f32>, sign_v: vec3<i32>,
               t0: f32, t1: f32, entry_face: u32, depth_cap: u32,
               mc: ptr<function, MediumRay>) -> VoxelHit {
  // 擦边退化（t0>=t1：射线只蹭到 chunk 边界）→ 无体素内部可穿过，直接 miss
  if (t0 >= t1) { return VoxelHit(false, 0.0, 0u, 0u, vec3<i32>(0)); }
  // chunk 局部 voxel 坐标（chunk 原点 = 0）；t 仍是 ro 系绝对 t
  let ro_c = ro - chunk_min;
  // side 距离/步长增量改用乘法（每外层省 3 个 fdiv）
  let inv_rd = 1.0 / rd;
  // 循环不变量预计算（sign_v / rd 在整个 trace_chunk 内不变）
  let abs_inv_rd = abs(inv_rd);                  // 每轮复用，省 abs()
  let axis_on = abs(rd) > vec3<f32>(1e-30);      // 该轴是否可步进（rd 分量为 0 时禁用）
  let oct = select(0u, 1u, sign_v.x >= 0)        // 方向八分体索引（b_leaves 查表用）
    | (select(0u, 1u, sign_v.y >= 0) << 1u)
    | (select(0u, 1u, sign_v.z >= 0) << 2u);
  let dir_side = select(vec3<f32>(0.0), vec3<f32>(1.0), sign_v >= vec3<i32>(0)); // 边界基址偏移
  let face_base = vec3<u32>(                     // 每轴两个面的 face id 基（0/1、2/3、4/5）
    select(1u, 0u, sign_v.x >= 0),
    select(1u, 0u, sign_v.y >= 0),
    select(1u, 0u, sign_v.z >= 0));
  var bricks: array<Brick, 4u>;
  let r_ml = b_struct[chunk_base];
  let r_mh = b_struct[chunk_base + 1u];
  let r_pal = b_struct[chunk_base + 2u] & 0xFFFFu;  // 节点字低 16 位 = uniform 色（16 位材质索引）
  bricks[3u] = Brick(chunk_base, (u64(r_mh) << 32u) | u64(r_ml), r_pal);
  var level: u32 = 3u;
  // 当前体素（chunk 局部 voxel 整数坐标，0..255；跨出 chunk 的步进瞬态可达 -1/256）
  let p0 = ro_c + rd * t0;
  var v = clamp(vec3<i32>(floor(p0)), vec3<i32>(0), vec3<i32>(255));
  var cur_t = t0;
  var face = entry_face;
  // 防挂死安全网：迭代上限（真实场景每 chunk 仅几十~几百次）
  var budget: u32 = 65536u;
  let lut_disable = view_u.lod.w > 0.5;
  loop {
    if (budget == 0u) { break; }
    budget = budget - 1u;
    if (level > 3u) { return VoxelHit(false, 0.0, 0u, 0u, vec3<i32>(0)); }
    // ---- traverse：从当前 level 下钻到 v 处内容（Douglas traverse_bit_set）----
    // 命中只**记录** pal，不在分支里 return：介质穿透与出射面判定统一放在 traverse 之后的
    // **唯一出口**做一次。原因见 [`MediumRay`]：判定若在各分支里重复，naga 无法把中间值留在
    // SSA 里，会把它们提升成函数级变量（GPU 上是 local memory / 栈）—— 实测最热函数
    // `trace_chunk` 的 local 变量会从 16 涨到 22，帧率掉三分之一。
    // `hit_pal` 同时服务于下方步进循环（那里只重置、不重复声明），两处共用一个栈槽。
    var hit_pal = 0u;
    loop {
      let b = bricks[level];
      // WGSL 移位 RHS 必须 u32
      let sh = vec3<u32>(level * 2u);
      let cell = vec3<i32>(v >> sh) & vec3<i32>(3); // &3 恒为 0..3
      let idx = u32(cell.z * 16 + cell.y * 4 + cell.x);
      if (level == 0u) {
        // 叶节点 inline palette：bit=1（非空体素）才 load inline word 取色；
        // bit=0 空气体素零 load（mask 在手）。
        // 每字 2 个体素（16 位索引）：字序号 = idx >> 1u，半字 = (idx & 1u) * 16u。
        if ((b.mask & (u64(1) << idx)) != u64(0)) {
          let w = b_struct[b.addr + NODE_FIXED_WORDS + (idx >> 1u)];
          hit_pal = (w >> ((idx & 1u) * 16u)) & 0xFFFFu;
        }
        break;
      }
      let bit = u64(1) << idx;
      if ((b.mask & bit) == u64(0)) {
        // 统一子块：颜色 = 节点 palette（0=空气）
        hit_pal = b.pal;
        break;
      }
      // depth_cap（beam 保守）：gate 深度 d=3-level 的分裂子块到达 cap → 子块入口 t
      let gd = 3u - level;
      if (gd >= depth_cap) { return VoxelHit(true, cur_t, b.pal, face, v); }
      // 分裂子块 → popcount 定位 child offset
      let below = b.mask & (bit - u64(1));
      let pop = countOneBits(u32(below)) + countOneBits(u32(below >> 32u));
      let child_addr = chunk_base + b_struct[b.addr + NODE_FIXED_WORDS + pop];
      let c_ml = b_struct[child_addr];
      let c_mh = b_struct[child_addr + 1u];
      let c_pw = b_struct[child_addr + 2u];
      let c_pal = c_pw & 0xFFFFu;  // 节点字低 16 位 = uniform 色
      let c_mask = (u64(c_mh) << 32u) | u64(c_ml);
      // 统一子节点快路径：wire 任意层的分裂位都可能指向 3 字统一节点（mask=0，pal 直决；
      // 叶层统一节点无 inline 16 字，禁读 addr+3 之后）
      if (c_mask == u64(0)) {
        hit_pal = c_pal;
        break;
      }
      level = level - 1u;
      bricks[level] = Brick(child_addr, c_mask, c_pal);
    }
    // ---- traverse 的唯一出口：命中 / 穿透 / 出射面 ----
    if (hit_pal != 0u) {
      // 介质 → 累积透射色后按空气继续（**不做**出射判定）；不透明 → 实体命中
      if (!pass_through_medium(mc, hit_pal)) {
        return VoxelHit(true, cur_t, hit_pal, face, v);
      }
    } else {
      // 真空气：介质段内撞到空气 ⇒ 这里就是出射面
      if (medium_exit(mc)) { return VoxelHit(true, cur_t, (*mc).last_pal, face, v); }
      (*mc).last_pal = 0u;
    }
    // ---- 整砖 LUT 跳过（Douglas march mask skip；仅 palette==0 节点安全）----
    // 当前 brick 从当前格/象限无可达分裂子块 → 直接升一层跨步（不重新 traverse：
    // 父帧当前格就是本砖，重 traverse 会原地再下钻；父层 dda 先跨步出本砖再判位）
    {
      let b = bricks[level];
      if (b.pal == 0u) {
        let sh = vec3<u32>(level * 2u);
        let ecell = vec3<i32>(v >> sh) & vec3<i32>(3);
        let entry_i = u32(ecell.z * 16 + ecell.y * 4 + ecell.x);
        let lut_base = min(oct * 64u + entry_i, 511u);
        var reach = b_leaves[lut_base];
        reach = select(reach, ~u64(0), lut_disable);
        if ((b.mask & reach) == u64(0)) {
          level = level + 1u;
          if (level > 3u) { return VoxelHit(false, 0.0, 0u, 0u, vec3<i32>(0)); }
        }
      }
    }
    // ---- v 处为空气：当前 level brick 内 DDA（Douglas dda）----
    let log2 = level * 2u;
    let s = 1i << (level * 2u); // 子块边长 voxel：1/4/16/64
    let s_mask = ~(s - 1i);    // 对齐掩码（two's complement: ~(s-1) = -s）
    // side_distance_for_ray：v 对齐到 s 的基址；正向 → 基址+s，负向 → 基址
    let base_v = v & vec3<i32>(s_mask);
    let boundary = vec3<f32>(base_v) + dir_side * f32(s);
    var side = vec3<f32>(1e+30);
    side = select(side, (boundary - ro_c) * inv_rd, axis_on);
    side = max(side, vec3<f32>(cur_t));
    // 每轴步长 t 增量（level 不变则不变）：inner 里 O(1) 加法
    let step_inc = f32(s) * abs_inv_rd;
    var step_axis: u32 = 0u;
    var changed = false;
    loop {
      // 选最近边界轴
      var mn: u32;
      if (side.x <= side.y && side.x <= side.z) { mn = 0u; }
      else if (side.y <= side.z) { mn = 1u; }
      else { mn = 2u; }
      step_axis = mn;
      cur_t = side[mn];
      if (cur_t >= t1) { return VoxelHit(false, 0.0, 0u, 0u, vec3<i32>(0)); } // 段内再无子块可入
      let old_cell = (v[mn] >> log2) & 3;
      // 沿 mn 轴整子块跨越
      v[mn] = v[mn] + sign_v[mn] * s;
      side[mn] = side[mn] + step_inc[mn];
      face = mn * 2u + face_base[mn];
      // 跨出 brick（4 子块）？正向往 3→外、负向往 0→外
      let crossed = select(old_cell == 0, old_cell == 3, sign_v[mn] >= 0);
      if (crossed) { changed = true; break; }
      // 新子块内容：level 0 查 inline palette（仅 mask!=0 的 inline 叶）；
      // level 1..3 = 分裂位或节点统一实体色。
      // 与 traverse 一样，命中 / 穿透 / 出射面判定只在**唯一出口**做一次（说明见那里）。
      let b = bricks[level];
      let sh2 = vec3<u32>(log2);
      let cell = vec3<i32>(v >> sh2) & vec3<i32>(3);
      let idx = u32(cell.z * 16 + cell.y * 4 + cell.x);
      hit_pal = 0u;   // 复用 traverse 的声明（每轮步进重置；少一个函数级变量 = 少一个栈槽）
      if (level == 0u) {
        // bit=1（非空体素）才 load inline word；bit=0 空气体素零 load
        if (b.mask != u64(0) && (b.mask & (u64(1) << idx)) != u64(0)) {
          let w = b_struct[b.addr + NODE_FIXED_WORDS + (idx >> 1u)];
          hit_pal = (w >> ((idx & 1u) * 16u)) & 0xFFFFu;
        } else if (b.mask == u64(0)) {
          // 防御：uniform 叶（正常下钻快路径已处理）
          hit_pal = b.pal;
        }
      } else {
        if ((b.mask & (u64(1) << idx)) != u64(0)) { break; } // 分裂子块 → 回 traverse 下钻
        hit_pal = b.pal;                                      // 统一实体
      }
      if (hit_pal != 0u) {
        if (pass_through_medium(mc, hit_pal)) { continue; } // 介质：继续步进
        return VoxelHit(true, cur_t, hit_pal, face, v);
      }
      // 空气子块 → 回 loop 顶重选 mn 轴继续；若刚离开介质段，这里就是出射面
      if (medium_exit(mc)) { return VoxelHit(true, cur_t, (*mc).last_pal, face, v); }
      (*mc).last_pal = 0u;
    }
    if (changed) {
      level = level + 1u;
      if (level > 3u) { return VoxelHit(false, 0.0, 0u, 0u, vec3<i32>(0)); }
    }
    // ---- firstTrailingBit 层级自适应跨级跳（Douglas march 尾部）----
    // 步进轴新坐标的尾随零位 = 对齐 run 长度：正向 comp=对齐基址（tz 直接读），
    // 负向 comp=区域尾址（基址|~mask，+1 后读 tz）。tz>>1 = 可跨步的最粗 level。
    let positive = sign_v[step_axis] >= 0;
    let cur_log2 = level * 2u;
    let m: u32 = 0xFFFFFFFFu << cur_log2;
    let not_m = ~m;                              // = (1<<cur_log2)-1，复用于 region_max
    let vmin_u = u32(v[step_axis]); // i32→u32 位环绕（负值公式自然处理）
    var comp: u32;
    if (positive) { comp = vmin_u & m; } else { comp = (vmin_u & m) | not_m; }
    let tz_i = firstTrailingBit(comp + select(1u, 0u, positive)); // 0 → -1
    var new_level: u32 = 0xFFFFFFFFu;
    if (tz_i >= 0) { new_level = u32(tz_i) >> 1u; }
    level = max(level, new_level);
    if (level > 3u) { return VoxelHit(false, 0.0, 0u, 0u, vec3<i32>(0)); } // 跨出 chunk（tz≥8）
    // 对齐快照：v 钳到 cur_t 射线点所在的当前 level 区域，步进轴取精确边界整数
    // （其余轴按射线实际位置吸附，消除只沿单轴步进的漂移）
    let mi = i32(m);
    let base = v & vec3<i32>(mi);
    let p = ro_c + rd * cur_t;
    let region_max = base + vec3<i32>(i32(not_m));
    v = clamp(vec3<i32>(floor(p)), base, region_max);
    v[step_axis] = i32(comp);
  }
  return VoxelHit(false, 0.0, 0u, 0u, vec3<i32>(0));
}

// ============================================================================
// 统一 DDA：trace_grid 一套管线同时服务主网格和逐物体 OBJ。
// 以下函数严格对应 Rust `brickmap/dda.rs` CPU 参考实现（逐字镜像）。
// ============================================================================

// slab 法射线-AABB 求交，返回 (t_enter, t_exit)；平行且在外 → (1.0, 0.0) miss 哨兵
// 对应 dda.rs::slab_box
fn slab_box(ro: vec3<f32>, rd: vec3<f32>, mn: vec3<f32>, mx: vec3<f32>, t0: f32, t1: f32) -> vec2<f32> {
  var t_enter = t0;
  var t_exit = t1;
  // X
  if (abs(rd.x) < 1e-30) {
    if (ro.x < mn.x || ro.x > mx.x) { return vec2<f32>(1.0, 0.0); }
  } else {
    let ta = (mn.x - ro.x) / rd.x;
    let tb = (mx.x - ro.x) / rd.x;
    t_enter = max(t_enter, min(ta, tb));
    t_exit = min(t_exit, max(ta, tb));
  }
  // Y
  if (abs(rd.y) < 1e-30) {
    if (ro.y < mn.y || ro.y > mx.y) { return vec2<f32>(1.0, 0.0); }
  } else {
    let ta = (mn.y - ro.y) / rd.y;
    let tb = (mx.y - ro.y) / rd.y;
    t_enter = max(t_enter, min(ta, tb));
    t_exit = min(t_exit, max(ta, tb));
  }
  // Z
  if (abs(rd.z) < 1e-30) {
    if (ro.z < mn.z || ro.z > mx.z) { return vec2<f32>(1.0, 0.0); }
  } else {
    let ta = (mn.z - ro.z) / rd.z;
    let tb = (mx.z - ro.z) / rd.z;
    t_enter = max(t_enter, min(ta, tb));
    t_exit = min(t_exit, max(ta, tb));
  }
  return vec2<f32>(t_enter, t_exit);
}

// ============================================================================
// 统一 DDA 核心：主世界和物体都是体素网格 DDA，走同一条 slab→chunk 间 A&W→trace_chunk
// 流程，差异由 Grid 参数化（tree_base + 变换矩阵 + chunk 窗口），零 kind 分支。
// ============================================================================

// 统一命中结构：hit/t/pal/n(世界空间法线)/face_id(0..5)/obj_id(-1=主世界,>=0=物体)。
// voxel：命中固体体素 grid 局部 voxel 整数坐标（主世界 = 世界坐标），由 DDA 整数步进精确产出。
struct UnifiedHit {
  hit: bool,
  t: f32,
  pal: u32,
  n: vec3<f32>,        // 世界空间法线（光影用）
  face_id: u32,        // 命中面 0..5（与 face_index_from_normal 对齐）
  voxel: vec3<i32>,    // grid 局部 voxel 命中体素（trace_chunk 的 v + ci*256）
  obj_id: i32,         // -1 = 主世界, >=0 = 物体索引
}

// 统一层次 DDA：slab→chunk 间 A&W→trace_chunk，所有网格（主世界 / 物体 / 任何体素 grid）
// 走同一条路径、零 kind 分支。g 由 make_grid 构造（grid_descs 只读一次，参数不按值传大结构体）。
/// chunk 级命中 → grid 级命中：面法线换到世界系、体素坐标补上 chunk 偏移。
/// 抽成函数是为了让 [`trace_grid`] 里那个二选一分流的两侧都能写成 `let h = ...`——
/// `var h: VoxelHit` 会变成一个 57 字节的栈变量，而每次 trace 都要跨若干 chunk。
fn grid_hit(h: VoxelHit, ci: vec3<i32>,
            col0: vec3<f32>, col1: vec3<f32>, col2: vec3<f32>, obj_id: i32) -> UnifiedHit {
  // 统一法线计算：trace_chunk 内部已推好 face_id，调用方零分支
  let n_local = face_normal_from_index(h.face_id);
  let n_world = normalize(n_local.x * col0 + n_local.y * col1 + n_local.z * col2);
  // chunk 局部 v → grid 局部体素（ci 为 trace_chunk 命中时所在 chunk，未步进）
  let voxel = h.voxel + ci * i32(CHUNK_SIZE);
  return UnifiedHit(true, h.t, h.pal, n_world, h.face_id, voxel, obj_id);
}

// penetrate：是否启用介质穿透（按值参数；false 时一次都不碰 mc，行为与改造前逐位一致）。
fn trace_grid(g: Grid, origin: vec3<f32>, dir: vec3<f32>, t_cap: f32, t_min: f32, depth_cap: u32,
              penetrate: bool, mc: ptr<function, MediumRay>) -> UnifiedHit {
  let w_mn = g.w_mn;
  let w_mx = g.w_mx;
  let l_min = g.l_min;
  let l_max = g.l_max;
  let index_origin = g.index_origin;
  let index_dims = g.index_dims;
  let tree_base = g.tree_base;
  let max_chunk_steps = g.max_chunk_steps;
  let col0 = g.col0;
  let col1 = g.col1;
  let col2 = g.col2;
  let obj_id = g.obj_id;
  let miss = UnifiedHit(false, 0.0, 0u, vec3<f32>(0.0), 0u, vec3<i32>(0), obj_id);
  // ---- 世界 AABB 预剔除（slab 内联）----
  var bx_enter = 0.0;
  var bx_exit = t_cap;
  if (abs(dir.x) < 1e-30) {
    if (origin.x < w_mn.x || origin.x > w_mx.x) { return miss; }
  } else {
    let ta = (w_mn.x - origin.x) / dir.x;
    let tb = (w_mx.x - origin.x) / dir.x;
    bx_enter = max(bx_enter, min(ta, tb));
    bx_exit = min(bx_exit, max(ta, tb));
  }
  if (abs(dir.y) < 1e-30) {
    if (origin.y < w_mn.y || origin.y > w_mx.y) { return miss; }
  } else {
    let ta = (w_mn.y - origin.y) / dir.y;
    let tb = (w_mx.y - origin.y) / dir.y;
    bx_enter = max(bx_enter, min(ta, tb));
    bx_exit = min(bx_exit, max(ta, tb));
  }
  if (abs(dir.z) < 1e-30) {
    if (origin.z < w_mn.z || origin.z > w_mx.z) { return miss; }
  } else {
    let ta = (w_mn.z - origin.z) / dir.z;
    let tb = (w_mx.z - origin.z) / dir.z;
    bx_enter = max(bx_enter, min(ta, tb));
    bx_exit = min(bx_exit, max(ta, tb));
  }
  let bx0 = max(bx_enter, 0.0);
  if (bx_exit < bx0 || bx_enter >= t_cap) { return miss; }
  let t_hi_cap = min(bx_exit, t_cap);
  let scale = g.scale;
  // 主世界（obj_id < 0）= identity 变换（make_grid 保证 pos=0/rot=I/scale=1、
  // 局部 AABB == 世界 AABB）→ 局部坐标直取世界坐标，跳过局部变换与等价的局部 slab。
  var ro: vec3<f32>;
  var rd: vec3<f32>;
  var tl0: f32;
  var tl1: f32;
  if (obj_id < 0) {
    ro = origin;
    rd = dir;
    tl0 = bx0;
    tl1 = t_hi_cap;
  } else {
    // ---- 局部变换 ----
    let wp = origin - g.pos;
    ro = vec3<f32>(dot(wp, col0), dot(wp, col1), dot(wp, col2)) / scale;
    rd = vec3<f32>(dot(dir, col0), dot(dir, col1), dot(dir, col2)) / scale;
    // ---- 局部 AABB slab（内联）----
    var tl_enter = 0.0;
    var tl_exit = t_hi_cap;
    if (abs(rd.x) < 1e-30) {
      if (ro.x < l_min.x || ro.x > l_max.x) { return miss; }
    } else {
      let ta = (l_min.x - ro.x) / rd.x;
      let tb = (l_max.x - ro.x) / rd.x;
      tl_enter = max(tl_enter, min(ta, tb));
      tl_exit = min(tl_exit, max(ta, tb));
    }
    if (abs(rd.y) < 1e-30) {
      if (ro.y < l_min.y || ro.y > l_max.y) { return miss; }
    } else {
      let ta = (l_min.y - ro.y) / rd.y;
      let tb = (l_max.y - ro.y) / rd.y;
      tl_enter = max(tl_enter, min(ta, tb));
      tl_exit = min(tl_exit, max(ta, tb));
    }
    if (abs(rd.z) < 1e-30) {
      if (ro.z < l_min.z || ro.z > l_max.z) { return miss; }
    } else {
      let ta = (l_min.z - ro.z) / rd.z;
      let tb = (l_max.z - ro.z) / rd.z;
      tl_enter = max(tl_enter, min(ta, tb));
      tl_exit = min(tl_exit, max(ta, tb));
    }
    if (tl_exit < max(tl_enter, 0.0)) { return miss; }
    tl0 = max(tl_enter, 0.0);
    tl1 = min(tl_exit, t_hi_cap);
  }
  if (tl1 <= tl0) { return miss; }
  // t_min = 3×3 邻域低分辨率射线的最小命中距离，跳过 [tl0, t_min) 的空空间 traversal。
  // 保守性见 beam_main 的 d_beam 与 BEAM_BACKOFF（邻域 min t 可能比真实首命中晚 ~2.3）。
  tl0 = max(tl0, t_min);
  if (tl1 <= tl0) { return miss; }
  // debug_mode.z == 2（诊断）：跳过 chunk 步进层（隔离 make_grid+slab 与遍历的成本）
  if (view_u.debug_mode.z > 1.5) { return miss; }
  // ---- chunk 间 A&W（256³ 一格）+ chunk 内层次 mask DDA（trace_chunk）----
  var sign_v = vec3<i32>(1i);
  sign_v = select(sign_v, vec3<i32>(-1i), rd < vec3<f32>(0.0));
  // 方向位掩码 LUT：一次查表拿到 side（boundary 基址）与三轴步进 face id（见 DIR_LUT）
  let dir_mask = u32(sign_v.x >= 0) | (u32(sign_v.y >= 0) << 1u) | (u32(sign_v.z >= 0) << 2u);
  let dir_word = DIR_LUT[dir_mask];
  let side = vec3<f32>(
    f32(dir_word & 1u),
    f32((dir_word >> 1u) & 1u),
    f32((dir_word >> 2u) & 1u)
  );
  let face_x = (dir_word >> 8u) & 0xFFu;
  let face_y = (dir_word >> 16u) & 0xFFu;
  let face_z = (dir_word >> 24u) & 0xFFu;
  // 预计算 inv_rd，循环内复用
  let grid_inv_rd = 1.0 / rd;
  let grid_axis_on = abs(rd) > vec3<f32>(1e-30);
  let delta = select(vec3<f32>(1e+30), abs(grid_inv_rd), grid_axis_on);
  let delta_c = delta * f32(CHUNK_SIZE);
  let start = ro + rd * tl0;
  var ci = vec3<i32>(floor(start / f32(CHUNK_SIZE)));
  // tmax_c：到下一 chunk 边界的 t（ro 系绝对）
  let bnd_c = (vec3<f32>(ci) + side) * f32(CHUNK_SIZE);
  var tmax_c = select(vec3<f32>(1e+30), (bnd_c - ro) * grid_inv_rd, grid_axis_on);
  tmax_c = max(tmax_c, vec3<f32>(tl0));
  var t_enter_c = tl0;
  // entry_face：跨入当前 chunk 的面。首 chunk 用 normalize(-rd) 兜底（相机贴面/UB；
  // 按约定 UB 直接返回该 voxel 颜色，face_id 反推）；后续 chunk 由跨轴 + sign_v 更新。
  // face_index_from_normal 在此内联（省函数调用）
  let en = normalize(-rd);
  let eax = abs(en.x);
  let eay = abs(en.y);
  let eaz = abs(en.z);
  var entry_face: u32 = 4u;
  if (eax >= max(eay, eaz)) {
    entry_face = select(0u, 1u, en.x >= 0.0);
  } else if (eay >= eaz) {
    entry_face = select(2u, 3u, en.y >= 0.0);
  } else {
    entry_face = select(4u, 5u, en.z >= 0.0);
  }
  for (var step_c: u32 = 0u; step_c < max_chunk_steps; step_c = step_c + 1u) {
    let t_exit_c = min(tmax_c.x, min(tmax_c.y, tmax_c.z));
    let t1 = min(t_exit_c, tl1);
    // ---- chunk 窗口定位（窗口外 / entry==0 = 空气）----
    var air_chunk = true;
    let rel = ci - index_origin;
    if (all(rel >= vec3<i32>(0)) && all(vec3<u32>(rel) < index_dims)) {
      let rel_u = vec3<u32>(rel);
      let index_addr = tree_base + rel_u.x + rel_u.y * CHUNK_INDEX_CAP
        + rel_u.z * (CHUNK_INDEX_CAP * CHUNK_INDEX_CAP);
      let entry = b_struct[index_addr];
      if (entry != 0u) {
        air_chunk = false;
        let chunk_base = tree_base + entry - 1u;
        let chunk_min = vec3<f32>(ci) * f32(CHUNK_SIZE);
        // chunk 内遍历从 max(chunk 入口, t_min) 开始，跳过 t_min 前的空空间
        let t0c = max(t_enter_c, t_min);
        // 二选一分流：不透明走无介质的特化版（理由见文件头 [`trace_chunk_opaque`] 的说明）
        if (penetrate) {
          let h = trace_chunk(chunk_base, chunk_min, ro, rd, sign_v,
                              t0c, t1, entry_face, depth_cap, mc);
          if (h.hit) { return grid_hit(h, ci, col0, col1, col2, obj_id); }
        } else {
          let h = trace_chunk_opaque(chunk_base, chunk_min, ro, rd, sign_v,
                                     t0c, t1, entry_face, depth_cap);
          if (h.hit) { return grid_hit(h, ci, col0, col1, col2, obj_id); }
        }
      }
    }
    // 空气 chunk：介质段在此终止 ⇒ 出射面就落在 chunk 边界（法线 = 跨入本 chunk 的面）
    if (penetrate && air_chunk && medium_exit(mc)) {
      let nl = face_normal_from_index(entry_face);
      let nw = normalize(nl.x * col0 + nl.y * col1 + nl.z * col2);
      return UnifiedHit(true, t_enter_c, (*mc).last_pal, nw, entry_face, vec3<i32>(0), obj_id);
    }
    if (t_exit_c >= tl1) { break; }
    if (tmax_c.x <= tmax_c.y && tmax_c.x <= tmax_c.z) {
      t_enter_c = tmax_c.x; tmax_c.x = tmax_c.x + delta_c.x; ci.x = ci.x + sign_v.x;
      entry_face = face_x;
    } else if (tmax_c.y <= tmax_c.z) {
      t_enter_c = tmax_c.y; tmax_c.y = tmax_c.y + delta_c.y; ci.y = ci.y + sign_v.y;
      entry_face = face_y;
    } else {
      t_enter_c = tmax_c.z; tmax_c.z = tmax_c.z + delta_c.z; ci.z = ci.z + sign_v.z;
      entry_face = face_z;
    }
  }
  // 走完本 grid 时仍在介质段内 ⇒ 出射面落在 grid 出口（法线 = 最后一次跨入的面）
  if (penetrate && medium_exit(mc)) {
    let nl = face_normal_from_index(entry_face);
    let nw = normalize(nl.x * col0 + nl.y * col1 + nl.z * col2);
    return UnifiedHit(true, tl1, (*mc).last_pal, nw, entry_face, vec3<i32>(0), obj_id);
  }
  return miss;
}

// 从 grid_descs[idx] 解码统一 Grid（主世界 idx=0 = identity 变换，物体 idx≥1 = 任意变换）。
// chunk 步数上限由窗口 dims 估算（曼哈顿对角 ×3 + 余量）；物体 dims=(1,1,1)。
fn make_grid(idx: u32) -> Grid {
  let d = grid_descs[idx];
  let origin = vec3<i32>(d.index_origin_x, d.index_origin_y, d.index_origin_z);
  let dims = vec3<i32>(i32(d.index_dims_x), i32(d.index_dims_y), i32(d.index_dims_z));
  let chunk_budget = u32(dims.x + dims.y + dims.z) * 3u + 16u;
  // 主世界（idx=0，identity：局部=世界）：局部 AABB = 窗口 AABB（origin 可为负，
  // [0,256]³ 默认盒会把窗口绝大部分 slab 剔除）；物体 = 单 chunk 局部 [0,256]³。
  let is_world = idx == 0u;
  let l_mn = select(vec3<f32>(0.0), d.aabb_min.xyz, is_world);
  let l_mx = select(vec3<f32>(f32(CHUNK_SIZE)), d.aabb_max.xyz, is_world);
  return Grid(
    d.aabb_min.xyz, d.aabb_max.xyz,       // 世界 AABB
    l_mn, l_mx,                            // 局部 AABB
    chunk_budget,
    d.rot0.xyz, d.rot1.xyz, d.rot2.xyz,    // 旋转矩阵列
    d.pos_scale.xyz, d.pos_scale.w,        // pos + scale
    d.tree_base, d.palette_base,           // 数据源基址
    origin, vec3<u32>(d.index_dims_x, d.index_dims_y, d.index_dims_z),
    // obj_id 约定（镜像 cpu_reference_trace_volumes）：idx=0 主世界→-1；
    // idx≥1 物体→idx-1（volume.rs list[0]=主世界, list[1..]=物体 obj_id 0..N-1）。
    // 下游按 obj_id+1 反查 grid_descs 取变换（如 p_voxel 归一化）。
    select(-1i, i32(idx) - 1i, idx > 0u),
  );
}
