import package::bindings::{b_leaves, b_palette, b_struct, g, grid_descs, view_u};
import package::brickmap::{Grid};
import package::common::{CHUNK_INDEX_CAP, CHUNK_SIZE, DIR_LUT, FLAGS_IS_PBR, FLAGS_TRANSMISSIVE, NODE_FIXED_WORDS, face_normal_from_index};

// 层级命中记录：t 为 ro 系绝对 t；face_id 0..5 = ±xyz 六面（命中面法线索引）。
// voxel = 命中固体体素 chunk 局部整数坐标，着色阶段直接消费，不要用「命中点 ± 法线半步」重建。
struct VoxelHit {
  hit: bool,
  t: f32,
  pal: u32,
  face_id: u32,
  voxel: vec3<i32>,
}

// 介质穿透（透明/玻璃）：透射材质按可穿过介质处理，在 DDA 内部一路走完。
// `stop_on_air = true`：遇空气停在出射面；false：空气也穿过（阴影射线）；`tint` 沿途累积透射色（乘性）。
// `last_pal` = 当前介质段材质（0 = 不在介质内），同段只染一次；`penetrate` 是按值参数、不在本结构里。
struct MediumRay {
  stop_on_air: bool,
  palette_base: u32,
  last_pal: u32,
  tint: vec3<f32>,
}

// brick 缓存槽：一层分裂节点常驻（镜像 CPU `BrickCpu`）。
struct Brick {
  addr: u32,  // 节点绝对字址（b_struct）
  mask: u64,  // 64bit 分裂掩码（bit=1 = 子块分裂；bit=0 = 统一子块，色 = pal）
  pal: u32,   // 节点 palette（统一子块颜色，0 = 空气）
}

/// 读 palette 槽的介质属性：`rgb` = 透射色（乘性），`a` = 不透明度（a == 1 = 不透明，非介质）。
/// 布局与 `wire.rs::pack_palette_entry` 一致（一次读两个 word）。
/// **介质判定读 `flags` 的 `TRANSMISSIVE` 位**（bit5）而不是 `transmission` 字节：本函数在 DDA 内
/// **逐体素**调用，且 PBR 变体里 bit8..15 已是 `asset` 的一部分（读它得不到透射率）。
/// flags 与 transmission 同在 word1 ⇒ 零额外读取；该位由写入侧维护（见 `wire.rs::pack_palette_entry`）。
fn medium_of(palette_base: u32, pal: u32) -> vec4<f32> {
  let w1 = b_palette[palette_base + pal * 2u + 1u];
  // 非介质：改动前的 `a = 1 − transmission/255` 在 transmission == 0 时恒等于 1.0，且调用方只用
  // `a >= 1.0` 判"不透明"（rgb 那条路径不消费）⇒ 这里直接给 a = 1.0，行为与改动前逐行等价。
  if ((w1 & FLAGS_TRANSMISSIVE) == 0u) { return vec4<f32>(1.0, 1.0, 1.0, 1.0); }
  if ((w1 & FLAGS_IS_PBR) != 0u) {
    // PBR 变体：w1 低 16 位是 `asset`，读不出透射率 ⇒ 只按 TRANSMISSIVE 位判"是介质"，
    // 吸收色取中性（白 = 不染色，非乘性暗化）。**PBR 变体的介质吸收色走材质资产，属 MT3/MT5**。
    return vec4<f32>(1.0, 1.0, 1.0, 0.0);
  }
  let w0 = b_palette[palette_base + pal * 2u];
  let srgb = vec3<f32>(f32(w0 & 0xFFu), f32((w0 >> 8u) & 0xFFu), f32((w0 >> 16u) & 0xFFu)) / 255.0;
  let a = 1.0 - f32((w1 >> 8u) & 0xFFu) / 255.0;
  return vec4<f32>(mix(vec3<f32>(1.0), srgb * srgb, a), a);
}

/// 可穿透介质 → 累积透射色（同段只染一次）并返回 true（调用方按空气继续步进）；不可穿透返回 false。
/// 只在 `penetrate = true` 时被调用。
fn pass_through_medium(mc: ptr<function, MediumRay>, pal: u32) -> bool {
  if ((*mc).last_pal == pal) { return true; }
  let m = medium_of((*mc).palette_base, pal);
  if (m.a >= 1.0) { return false; }
  (*mc).tint = (*mc).tint * m.rgb;
  (*mc).last_pal = pal;
  return true;
}

/// 介质段内撞到空气 = 出射面：返回 true（调用方带回 `last_pal`），只在 `penetrate = true` 时被调用。
fn medium_exit(mc: ptr<function, MediumRay>) -> bool {
  return (*mc).stop_on_air && ((*mc).last_pal != 0u);
}

// 单 chunk 内层级遍历（不透明版）：chunk_base = 根节点绝对字址，chunk_min = chunk 原点（局部 voxel）。
// [t0,t1] 与返回 t 为 ro 系绝对 t；entry_face = 进入面；depth_cap 下 gate 深度 d=3-level 达上限时返回子块入口 t（保守下界）。
fn trace_chunk_opaque(chunk_base: u32, chunk_min: vec3<f32>,
                      ro: vec3<f32>, rd: vec3<f32>, sign_v: vec3<i32>,
                      t0: f32, t1: f32, entry_face: u32, depth_cap: u32) -> VoxelHit {
  if (t0 >= t1) { return VoxelHit(false, 0.0, 0u, 0u, vec3<i32>(0)); }
  // chunk 局部 voxel 坐标（chunk 原点 = 0）；t 仍是 ro 系绝对 t
  let ro_c = ro - chunk_min;
  let inv_rd = 1.0 / rd;
  let abs_inv_rd = abs(inv_rd);
  let axis_on = abs(rd) > vec3<f32>(1e-30);
  let oct = select(0u, 1u, sign_v.x >= 0)
    | (select(0u, 1u, sign_v.y >= 0) << 1u)
    | (select(0u, 1u, sign_v.z >= 0) << 2u);
  let dir_side = select(vec3<f32>(0.0), vec3<f32>(1.0), sign_v >= vec3<i32>(0));
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
  // 当前体素（chunk 局部 voxel 整数坐标 0..255；跨出 chunk 的步进瞬态可达 -1/256）
  let p0 = ro_c + rd * t0;
  var v = clamp(vec3<i32>(floor(p0)), vec3<i32>(0), vec3<i32>(255));
  var cur_t = t0;
  var face = entry_face;
  // 防挂死安全网：迭代上限。
  var budget: u32 = 65536u;
  let lut_disable = view_u.lod.w > 0.5;
  loop {
    if (budget == 0u) { break; }
    budget = budget - 1u;
    if (level > 3u) { return VoxelHit(false, 0.0, 0u, 0u, vec3<i32>(0)); }
    // ---- traverse：从当前 level 下钻到 v 处内容 ----
    loop {
      let b = bricks[level];
      // WGSL 移位 RHS 必须 u32
      let sh = vec3<u32>(level * 2u);
      let cell = vec3<i32>(v >> sh) & vec3<i32>(3);
      let idx = u32(cell.z * 16 + cell.y * 4 + cell.x);
      if (level == 0u) {
        // 叶节点 inline palette：bit=1（非空体素）才 load inline word 取色；每字 2 个体素（16 位索引）。
        if ((b.mask & (u64(1) << idx)) != u64(0)) {
          let w = b_struct[b.addr + NODE_FIXED_WORDS + (idx >> 1u)];
          let leaf_pal = (w >> ((idx & 1u) * 16u)) & 0xFFFFu;
          if (leaf_pal != 0u) { return VoxelHit(true, cur_t, leaf_pal, face, v); }
        }
        break;
      }
      let bit = u64(1) << idx;
      if ((b.mask & bit) == u64(0)) {
        // 统一子块：色 = 节点 palette（0 = 空气）
        if (b.pal != 0u) { return VoxelHit(true, cur_t, b.pal, face, v); }
        break;
      }
      // depth_cap（beam 保守）：gate 深度 d=3-level 的分裂子块到达 cap → 子块入口 t
      let gd = 3u - level;
      if (gd >= depth_cap) { return VoxelHit(true, cur_t, b.pal, face, v); }
      let below = b.mask & (bit - u64(1));
      let pop = countOneBits(u32(below)) + countOneBits(u32(below >> 32u));
      let child_addr = chunk_base + b_struct[b.addr + NODE_FIXED_WORDS + pop];
      let c_ml = b_struct[child_addr];
      let c_mh = b_struct[child_addr + 1u];
      let c_pw = b_struct[child_addr + 2u];
      let c_pal = c_pw & 0xFFFFu;  // 节点字低 16 位 = uniform 色
      let c_mask = (u64(c_mh) << 32u) | u64(c_ml);
      // 统一子节点快路径：wire 任意层的分裂位都可能指向 3 字统一节点（mask=0，禁读 addr+3 之后）
      if (c_mask == u64(0)) {
        if (c_pal != 0u) { return VoxelHit(true, cur_t, c_pal, face, v); }
        break;
      }
      level = level - 1u;
      bricks[level] = Brick(child_addr, c_mask, c_pal);
    }
    // ---- 整砖 LUT 跳过（仅 palette==0 节点安全）----
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
    // ---- v 处为空气：当前 level brick 内 DDA ----
    let log2 = level * 2u;
    let s = 1i << (level * 2u); // 子块边长 voxel：1/4/16/64
    let s_mask = ~(s - 1i);
    // side_distance_for_ray：v 对齐到 s 的基址；正向 → 基址+s，负向 → 基址
    let base_v = v & vec3<i32>(s_mask);
    let boundary = vec3<f32>(base_v) + dir_side * f32(s);
    var side = vec3<f32>(1e+30);
    side = select(side, (boundary - ro_c) * inv_rd, axis_on);
    side = max(side, vec3<f32>(cur_t));
    let step_inc = f32(s) * abs_inv_rd;
    var step_axis: u32 = 0u;
    var changed = false;
    loop {
      var mn: u32;
      if (side.x <= side.y && side.x <= side.z) { mn = 0u; }
      else if (side.y <= side.z) { mn = 1u; }
      else { mn = 2u; }
      step_axis = mn;
      cur_t = side[mn];
      if (cur_t >= t1) { return VoxelHit(false, 0.0, 0u, 0u, vec3<i32>(0)); }
      let old_cell = (v[mn] >> log2) & 3;
      v[mn] = v[mn] + sign_v[mn] * s;
      side[mn] = side[mn] + step_inc[mn];
      face = mn * 2u + face_base[mn];
      let crossed = select(old_cell == 0, old_cell == 3, sign_v[mn] >= 0);
      if (crossed) { changed = true; break; }
      let b = bricks[level];
      let sh2 = vec3<u32>(log2);
      let cell = vec3<i32>(v >> sh2) & vec3<i32>(3);
      let idx = u32(cell.z * 16 + cell.y * 4 + cell.x);
      if (level == 0u) {
        if (b.mask != u64(0) && (b.mask & (u64(1) << idx)) != u64(0)) {
          let w = b_struct[b.addr + NODE_FIXED_WORDS + (idx >> 1u)];
          let dp = (w >> ((idx & 1u) * 16u)) & 0xFFFFu;
          if (dp != 0u) { return VoxelHit(true, cur_t, dp, face, v); }
        } else if (b.mask == u64(0) && b.pal != 0u) {
          return VoxelHit(true, cur_t, b.pal, face, v);
        }
      } else {
        let mb = (b.mask & (u64(1) << idx)) != u64(0);
        if (mb) { break; }
        if (b.pal != 0u) { return VoxelHit(true, cur_t, b.pal, face, v); }
      }
    }
    if (changed) {
      level = level + 1u;
      if (level > 3u) { return VoxelHit(false, 0.0, 0u, 0u, vec3<i32>(0)); }
    }
    // ---- firstTrailingBit 层级自适应跨级跳 ----
    // 步进轴新坐标的尾随零位 = 对齐 run 长度，tz>>1 = 可跨步的最粗 level。
    let positive = sign_v[step_axis] >= 0;
    let cur_log2 = level * 2u;
    let m: u32 = 0xFFFFFFFFu << cur_log2;
    let not_m = ~m;
    let vmin_u = u32(v[step_axis]);
    var comp: u32;
    if (positive) { comp = vmin_u & m; } else { comp = (vmin_u & m) | not_m; }
    let tz_i = firstTrailingBit(comp + select(1u, 0u, positive));
    var new_level: u32 = 0xFFFFFFFFu;
    if (tz_i >= 0) { new_level = u32(tz_i) >> 1u; }
    level = max(level, new_level);
    if (level > 3u) { return VoxelHit(false, 0.0, 0u, 0u, vec3<i32>(0)); }
    // 对齐快照：v 钳到 cur_t 处当前 level 区域，步进轴取精确边界整数。
    let mi = i32(m);
    let base = v & vec3<i32>(mi);
    let p = ro_c + rd * cur_t;
    let region_max = base + vec3<i32>(i32(not_m));
    v = clamp(vec3<i32>(floor(p)), base, region_max);
    v[step_axis] = i32(comp);
  }
  return VoxelHit(false, 0.0, 0u, 0u, vec3<i32>(0));
}

// 单 chunk 内层级遍历（介质穿透版）：调用点恒传穿透语义，无 `penetrate` 开关；mc = 介质穿透上下文。
// 命中 / 穿透 / 出射面判定只在两处唯一出口做。
fn trace_chunk(chunk_base: u32, chunk_min: vec3<f32>,
               ro: vec3<f32>, rd: vec3<f32>, sign_v: vec3<i32>,
               t0: f32, t1: f32, entry_face: u32, depth_cap: u32,
               mc: ptr<function, MediumRay>) -> VoxelHit {
  if (t0 >= t1) { return VoxelHit(false, 0.0, 0u, 0u, vec3<i32>(0)); }
  // chunk 局部 voxel 坐标（chunk 原点 = 0）；t 仍是 ro 系绝对 t
  let ro_c = ro - chunk_min;
  let inv_rd = 1.0 / rd;
  let abs_inv_rd = abs(inv_rd);
  let axis_on = abs(rd) > vec3<f32>(1e-30);
  let oct = select(0u, 1u, sign_v.x >= 0)
    | (select(0u, 1u, sign_v.y >= 0) << 1u)
    | (select(0u, 1u, sign_v.z >= 0) << 2u);
  let dir_side = select(vec3<f32>(0.0), vec3<f32>(1.0), sign_v >= vec3<i32>(0));
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
  // 当前体素（chunk 局部 voxel 整数坐标 0..255；跨出 chunk 的步进瞬态可达 -1/256）
  let p0 = ro_c + rd * t0;
  var v = clamp(vec3<i32>(floor(p0)), vec3<i32>(0), vec3<i32>(255));
  var cur_t = t0;
  var face = entry_face;
  // 防挂死安全网：迭代上限。
  var budget: u32 = 65536u;
  let lut_disable = view_u.lod.w > 0.5;
  loop {
    if (budget == 0u) { break; }
    budget = budget - 1u;
    if (level > 3u) { return VoxelHit(false, 0.0, 0u, 0u, vec3<i32>(0)); }
    // ---- traverse：从当前 level 下钻到 v 处内容 ----
    var hit_pal = 0u;
    loop {
      let b = bricks[level];
      // WGSL 移位 RHS 必须 u32
      let sh = vec3<u32>(level * 2u);
      let cell = vec3<i32>(v >> sh) & vec3<i32>(3);
      let idx = u32(cell.z * 16 + cell.y * 4 + cell.x);
      if (level == 0u) {
        // 叶节点 inline palette：bit=1（非空体素）才 load inline word 取色；每字 2 个体素（16 位索引）。
        if ((b.mask & (u64(1) << idx)) != u64(0)) {
          let w = b_struct[b.addr + NODE_FIXED_WORDS + (idx >> 1u)];
          hit_pal = (w >> ((idx & 1u) * 16u)) & 0xFFFFu;
        }
        break;
      }
      let bit = u64(1) << idx;
      if ((b.mask & bit) == u64(0)) {
        // 统一子块：色 = 节点 palette（0 = 空气）
        hit_pal = b.pal;
        break;
      }
      // depth_cap（beam 保守）：gate 深度 d=3-level 的分裂子块到达 cap → 子块入口 t
      let gd = 3u - level;
      if (gd >= depth_cap) { return VoxelHit(true, cur_t, b.pal, face, v); }
      let below = b.mask & (bit - u64(1));
      let pop = countOneBits(u32(below)) + countOneBits(u32(below >> 32u));
      let child_addr = chunk_base + b_struct[b.addr + NODE_FIXED_WORDS + pop];
      let c_ml = b_struct[child_addr];
      let c_mh = b_struct[child_addr + 1u];
      let c_pw = b_struct[child_addr + 2u];
      let c_pal = c_pw & 0xFFFFu;  // 节点字低 16 位 = uniform 色
      let c_mask = (u64(c_mh) << 32u) | u64(c_ml);
      // 统一子节点快路径：wire 任意层的分裂位都可能指向 3 字统一节点（mask=0，禁读 addr+3 之后）
      if (c_mask == u64(0)) {
        hit_pal = c_pal;
        break;
      }
      level = level - 1u;
      bricks[level] = Brick(child_addr, c_mask, c_pal);
    }
    // ---- traverse 唯一出口：命中 / 穿透 / 出射面 ----
    if (hit_pal != 0u) {
      if (!pass_through_medium(mc, hit_pal)) {
        return VoxelHit(true, cur_t, hit_pal, face, v);
      }
    } else {
      if (medium_exit(mc)) { return VoxelHit(true, cur_t, (*mc).last_pal, face, v); }
      (*mc).last_pal = 0u;
    }
    // ---- 整砖 LUT 跳过（仅 palette==0 节点安全）----
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
    // ---- v 处为空气：当前 level brick 内 DDA ----
    let log2 = level * 2u;
    let s = 1i << (level * 2u); // 子块边长 voxel：1/4/16/64
    let s_mask = ~(s - 1i);
    // side_distance_for_ray：v 对齐到 s 的基址；正向 → 基址+s，负向 → 基址
    let base_v = v & vec3<i32>(s_mask);
    let boundary = vec3<f32>(base_v) + dir_side * f32(s);
    var side = vec3<f32>(1e+30);
    side = select(side, (boundary - ro_c) * inv_rd, axis_on);
    side = max(side, vec3<f32>(cur_t));
    let step_inc = f32(s) * abs_inv_rd;
    var step_axis: u32 = 0u;
    var changed = false;
    loop {
      var mn: u32;
      if (side.x <= side.y && side.x <= side.z) { mn = 0u; }
      else if (side.y <= side.z) { mn = 1u; }
      else { mn = 2u; }
      step_axis = mn;
      cur_t = side[mn];
      if (cur_t >= t1) { return VoxelHit(false, 0.0, 0u, 0u, vec3<i32>(0)); }
      let old_cell = (v[mn] >> log2) & 3;
      v[mn] = v[mn] + sign_v[mn] * s;
      side[mn] = side[mn] + step_inc[mn];
      face = mn * 2u + face_base[mn];
      let crossed = select(old_cell == 0, old_cell == 3, sign_v[mn] >= 0);
      if (crossed) { changed = true; break; }
      let b = bricks[level];
      let sh2 = vec3<u32>(log2);
      let cell = vec3<i32>(v >> sh2) & vec3<i32>(3);
      let idx = u32(cell.z * 16 + cell.y * 4 + cell.x);
      hit_pal = 0u;
      if (level == 0u) {
        if (b.mask != u64(0) && (b.mask & (u64(1) << idx)) != u64(0)) {
          let w = b_struct[b.addr + NODE_FIXED_WORDS + (idx >> 1u)];
          hit_pal = (w >> ((idx & 1u) * 16u)) & 0xFFFFu;
        } else if (b.mask == u64(0)) {
          hit_pal = b.pal;
        }
      } else {
        if ((b.mask & (u64(1) << idx)) != u64(0)) { break; }
        hit_pal = b.pal;
      }
      if (hit_pal != 0u) {
        if (pass_through_medium(mc, hit_pal)) { continue; }
        return VoxelHit(true, cur_t, hit_pal, face, v);
      }
      if (medium_exit(mc)) { return VoxelHit(true, cur_t, (*mc).last_pal, face, v); }
      (*mc).last_pal = 0u;
    }
    if (changed) {
      level = level + 1u;
      if (level > 3u) { return VoxelHit(false, 0.0, 0u, 0u, vec3<i32>(0)); }
    }
    // ---- firstTrailingBit 层级自适应跨级跳 ----
    // 步进轴新坐标的尾随零位 = 对齐 run 长度，tz>>1 = 可跨步的最粗 level。
    let positive = sign_v[step_axis] >= 0;
    let cur_log2 = level * 2u;
    let m: u32 = 0xFFFFFFFFu << cur_log2;
    let not_m = ~m;
    let vmin_u = u32(v[step_axis]);
    var comp: u32;
    if (positive) { comp = vmin_u & m; } else { comp = (vmin_u & m) | not_m; }
    let tz_i = firstTrailingBit(comp + select(1u, 0u, positive));
    var new_level: u32 = 0xFFFFFFFFu;
    if (tz_i >= 0) { new_level = u32(tz_i) >> 1u; }
    level = max(level, new_level);
    if (level > 3u) { return VoxelHit(false, 0.0, 0u, 0u, vec3<i32>(0)); }
    // 对齐快照：v 钳到 cur_t 处当前 level 区域，步进轴取精确边界整数。
    let mi = i32(m);
    let base = v & vec3<i32>(mi);
    let p = ro_c + rd * cur_t;
    let region_max = base + vec3<i32>(i32(not_m));
    v = clamp(vec3<i32>(floor(p)), base, region_max);
    v[step_axis] = i32(comp);
  }
  return VoxelHit(false, 0.0, 0u, 0u, vec3<i32>(0));
}

// 统一 DDA：主网格与逐物体 OBJ 共用一条管线，严格镜像 Rust `brickmap/dda.rs` CPU 参考实现。

// slab 法射线-AABB 求交，返回 (t_enter, t_exit)；平行且在外 → (1.0, 0.0) miss 哨兵。
fn slab_box(ro: vec3<f32>, rd: vec3<f32>, mn: vec3<f32>, mx: vec3<f32>, t0: f32, t1: f32) -> vec2<f32> {
  var t_enter = t0;
  var t_exit = t1;
  if (abs(rd.x) < 1e-30) {
    if (ro.x < mn.x || ro.x > mx.x) { return vec2<f32>(1.0, 0.0); }
  } else {
    let ta = (mn.x - ro.x) / rd.x;
    let tb = (mx.x - ro.x) / rd.x;
    t_enter = max(t_enter, min(ta, tb));
    t_exit = min(t_exit, max(ta, tb));
  }
  if (abs(rd.y) < 1e-30) {
    if (ro.y < mn.y || ro.y > mx.y) { return vec2<f32>(1.0, 0.0); }
  } else {
    let ta = (mn.y - ro.y) / rd.y;
    let tb = (mx.y - ro.y) / rd.y;
    t_enter = max(t_enter, min(ta, tb));
    t_exit = min(t_exit, max(ta, tb));
  }
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

// 统一 DDA 核心：主世界与物体走同一条 slab→chunk 间 A&W→trace_chunk 流程，差异由 `Grid` 参数化。

// 统一命中结构：hit / t / pal / n（世界空间法线）/ face_id（0..5）/ obj_id（-1 = 主世界，>=0 = 物体）。
// voxel 为命中固体体素 grid 局部整数坐标（主世界 = 世界坐标），由 DDA 整数步进精确产出。
struct UnifiedHit {
  hit: bool,
  t: f32,
  pal: u32,
  n: vec3<f32>,        // 世界空间法线（光影用）
  face_id: u32,        // 命中面 0..5（与 face_index_from_normal 对齐）
  voxel: vec3<i32>,    // grid 局部命中体素坐标
  obj_id: i32,         // -1 = 主世界, >=0 = 物体索引
}

/// chunk 级命中 → grid 级命中：面法线换到世界系，体素坐标补 chunk 偏移（`ci` = 命中 chunk）。
fn grid_hit(h: VoxelHit, ci: vec3<i32>,
            col0: vec3<f32>, col1: vec3<f32>, col2: vec3<f32>, obj_id: i32) -> UnifiedHit {
  let n_local = face_normal_from_index(h.face_id);
  let n_world = normalize(n_local.x * col0 + n_local.y * col1 + n_local.z * col2);
  let voxel = h.voxel + ci * i32(CHUNK_SIZE);
  return UnifiedHit(true, h.t, h.pal, n_world, h.face_id, voxel, obj_id);
}

// 统一层次 DDA：slab→chunk 间 A&W→trace_chunk，`penetrate = false` 时一次都不碰 mc。
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
  // 主世界（obj_id < 0）= identity 变换（make_grid 保证 pos=0/rot=I/scale=1），局部坐标直取世界坐标。
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
  // t_min = 3×3 邻域低分辨率射线的最小命中距离，跳过 [tl0, t_min) 的空空间 traversal（保守下界）。
  tl0 = max(tl0, t_min);
  if (tl1 <= tl0) { return miss; }
  // debug_mode.z == 2（诊断）：跳过 chunk 步进层。
  if (view_u.debug_mode.z > 1.5) { return miss; }
  // ---- chunk 间 A&W（256³ 一格）+ chunk 内层次 mask DDA ----
  var sign_v = vec3<i32>(1i);
  sign_v = select(sign_v, vec3<i32>(-1i), rd < vec3<f32>(0.0));
  // 方向位掩码 LUT：一次查表拿到 side（boundary 基址）与三轴步进 face id。
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
  // entry_face = 跨入当前 chunk 的面（逆射线方向的主轴面）；首 chunk 用它兜底（相机贴面/UB）。
  var entry_face: u32 = 4u;
  let ad = abs(rd);
  if (ad.x >= max(ad.y, ad.z)) {
    entry_face = face_x;
  } else if (ad.y >= ad.z) {
    entry_face = face_y;
  } else {
    entry_face = face_z;
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
    // 空气 chunk：介质段在此终止 ⇒ 出射面落在 chunk 边界（法线 = 跨入本 chunk 的面）。
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
  // 走完本 grid 时仍在介质段内 ⇒ 出射面落在 grid 出口（法线 = 最后一次跨入的面）。
  if (penetrate && medium_exit(mc)) {
    let nl = face_normal_from_index(entry_face);
    let nw = normalize(nl.x * col0 + nl.y * col1 + nl.z * col2);
    return UnifiedHit(true, tl1, (*mc).last_pal, nw, entry_face, vec3<i32>(0), obj_id);
  }
  return miss;
}

// 从 `grid_descs[idx]` 解码统一 `Grid`（主世界 idx=0 = identity 变换，物体 idx≥1 = 任意变换）。
// chunk 步数上限由窗口 dims 估算（曼哈顿对角 ×3 + 余量）；物体 dims=(1,1,1)。
fn make_grid(idx: u32) -> Grid {
  let d = grid_descs[idx];
  let origin = vec3<i32>(d.index_origin_x, d.index_origin_y, d.index_origin_z);
  let dims = vec3<i32>(i32(d.index_dims_x), i32(d.index_dims_y), i32(d.index_dims_z));
  let chunk_budget = u32(dims.x + dims.y + dims.z) * 3u + 16u;
  // 主世界（idx=0，identity）：局部 AABB = 窗口 AABB（origin 可为负）；物体 = 单 chunk 局部 [0,256]³。
  let is_world = idx == 0u;
  let l_mn = select(vec3<f32>(0.0), d.aabb_min.xyz, is_world);
  let l_mx = select(vec3<f32>(f32(CHUNK_SIZE)), d.aabb_max.xyz, is_world);
  // 变换段：主世界是 identity（Rust `VolumeTransform::IDENTITY`；`trace_grid` 的 `obj_id < 0` 分支
  // 早已依赖这条 —— 它把局部坐标直取世界坐标）⇒ 不读 `rot0..2` / `pos_scale` 这 4 个 vec4，
  // 直接给常量。这样下游的 `obj_id < 0` 分支（ro/rd 直取、`voxel_normal` 的 mat-vec 短路）
  // 也能整段常量折叠掉。
  var col0 = vec3<f32>(1.0, 0.0, 0.0);
  var col1 = vec3<f32>(0.0, 1.0, 0.0);
  var col2 = vec3<f32>(0.0, 0.0, 1.0);
  var pos = vec3<f32>(0.0);
  var scale = 1.0;
  if (!is_world) {
    col0 = d.rot0.xyz;      // 旋转矩阵列
    col1 = d.rot1.xyz;
    col2 = d.rot2.xyz;
    pos = d.pos_scale.xyz;  // 物体位置
    scale = d.pos_scale.w;
  }
  return Grid(
    d.aabb_min.xyz, d.aabb_max.xyz,       // 世界 AABB
    l_mn, l_mx,                            // 局部 AABB
    chunk_budget,
    col0, col1, col2,                      // 旋转矩阵列
    pos, scale,                            // pos + scale
    d.tree_base, d.palette_base,           // 数据源基址
    origin, vec3<u32>(d.index_dims_x, d.index_dims_y, d.index_dims_z),
    // obj_id 约定（镜像 Rust）：idx=0 主世界 → -1；idx≥1 物体 → idx-1。
    select(-1i, i32(idx) - 1i, idx > 0u),
  );
}
