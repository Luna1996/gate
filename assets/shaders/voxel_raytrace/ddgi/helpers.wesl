import package::bindings::{b_struct, ddgi_cell_slot, ddgi_lod0, ddgi_lod0_state, ddgi_u, g};
import package::brickmap::{Grid};
import package::common::{CHUNK_INDEX_CAP, NODE_FIXED_WORDS};
import package::ddgi::consts::{DDGI_BIG_CHANGE, DDGI_BRIGHTNESS, DDGI_CHANGE_DROP, DDGI_CHUNK_LOD, DDGI_CLAIM_CAVITY, DDGI_DELTA_CLAMP, DDGI_DEPTH_TEXELS, DDGI_HYSTERESIS, DDGI_IRRAD_GAMMA, DDGI_IRR_TEXELS, DDGI_LOD0_BRICKS_PER_AXIS, DDGI_LOD0_CELLS_PER_BRICK, DDGI_LOD0_CLAIM_BIAS, DDGI_LOD0_SLOTS_PER_BRICK, DDGI_LOD0_STATE_ALLOC, DDGI_LOD0_STATE_CLAIM, DDGI_LOD_COUNT, DDGI_META_ACTIVE, DDGI_META_ENABLED, DDGI_META_NBR_MASK, DDGI_META_NBR_SHIFT, DDGI_META_NBR_VALID, DDGI_MIN_STEP, DDGI_NO_PROBE, DDGI_PROBE_MIN_BLOCK, DDGI_PROBES_PER_LAYER, DDGI_PROBES_PER_LAYER_AXIS, DDGI_USAGE_STICKY};
import package::trace::{make_grid};
import package::world::{world_solid};

fn ddgi_merge_state(a: u32, b: u32) -> u32 {
  if (a == 3u) { return b; }
  if (a == b) { return a; }
  return 2u;
}

fn ddgi_pop_below(cidx: u32, mask_lo: u32, mask_hi: u32) -> u32 {
  if (cidx < 32u) {
    return countOneBits(mask_lo & ((1u << cidx) - 1u));
  }
  return countOneBits(mask_lo) + countOneBits(mask_hi & ((1u << (cidx - 32u)) - 1u));
}

fn ddgi_chunk_base(g: Grid, voxel: vec3<i32>) -> u32 {
  let m = ((voxel % vec3<i32>(256)) + vec3<i32>(256)) % vec3<i32>(256);
  let chunk_i = (voxel - m) / vec3<i32>(256);
  let rel = chunk_i - g.index_origin;
  if (any(rel < vec3<i32>(0))) { return 0u; }
  let rel_u = vec3<u32>(rel);
  if (any(rel_u >= g.index_dims)) { return 0u; }
  let index_addr = g.tree_base + rel_u.x + rel_u.y * CHUNK_INDEX_CAP
    + rel_u.z * (CHUNK_INDEX_CAP * CHUNK_INDEX_CAP);
  let entry = b_struct[index_addr];
  if (entry == 0u) { return 0u; }
  return g.tree_base + entry - 1u;
}

fn ddgi_agg4(chunk_base: u32, node: u32) -> u32 {
  let mask_lo = b_struct[node];
  let mask_hi = b_struct[node + 1u];
  let pal = b_struct[node + 2u] & 0xFFFFu;
  var seen: u32 = 3u;
  for (var idx = 0u; idx < 64u; idx = idx + 1u) {
    let in_hi = idx >= 32u;
    let word = select(mask_lo, mask_hi, in_hi);
    let bit_in = select(idx, idx - 32u, in_hi);
    var vp = pal;
    if (((word >> bit_in) & 1u) != 0u) {
      // 叶父层 inline：每字 2 体素（16 位索引）
      let w = b_struct[node + NODE_FIXED_WORDS + (idx >> 1u)];
      vp = (w >> ((idx & 1u) * 16u)) & 0xFFFFu;
    }
    let st = select(1u, 0u, vp == 0u);
    seen = ddgi_merge_state(seen, st);
    if (seen == 2u) { return 2u; }
  }
  return select(0u, seen, seen != 3u);
}

fn ddgi_agg16(chunk_base: u32, node: u32) -> u32 {
  let mask_lo = b_struct[node];
  let mask_hi = b_struct[node + 1u];
  let pal = b_struct[node + 2u] & 0xFFFFu;
  var seen: u32 = 3u;
  for (var idx = 0u; idx < 64u; idx = idx + 1u) {
    let in_hi = idx >= 32u;
    let word = select(mask_lo, mask_hi, in_hi);
    let bit_in = select(idx, idx - 32u, in_hi);
    var st: u32;
    if (((word >> bit_in) & 1u) == 0u) {
      st = select(1u, 0u, pal == 0u);
    } else {
      let child = chunk_base + b_struct[node + NODE_FIXED_WORDS + ddgi_pop_below(idx, mask_lo, mask_hi)];
      st = ddgi_agg4(chunk_base, child);
    }
    seen = ddgi_merge_state(seen, st);
    if (seen == 2u) { return 2u; }
  }
  return select(0u, seen, seen != 3u);
}

fn ddgi_agg64(chunk_base: u32, node: u32) -> u32 {
  let mask_lo = b_struct[node];
  let mask_hi = b_struct[node + 1u];
  let pal = b_struct[node + 2u] & 0xFFFFu;
  var seen: u32 = 3u;
  for (var idx = 0u; idx < 64u; idx = idx + 1u) {
    let in_hi = idx >= 32u;
    let word = select(mask_lo, mask_hi, in_hi);
    let bit_in = select(idx, idx - 32u, in_hi);
    var st: u32;
    if (((word >> bit_in) & 1u) == 0u) {
      st = select(1u, 0u, pal == 0u);
    } else {
      let child = chunk_base + b_struct[node + NODE_FIXED_WORDS + ddgi_pop_below(idx, mask_lo, mask_hi)];
      st = ddgi_agg16(chunk_base, child);
    }
    seen = ddgi_merge_state(seen, st);
    if (seen == 2u) { return 2u; }
  }
  return select(0u, seen, seen != 3u);
}

fn ddgi_brick_state(g: Grid, origin: vec3<i32>, level: u32) -> u32 {
  let cb = ddgi_chunk_base(g, origin);
  if (cb == 0u) { return 0u; }
  let m = ((origin % vec3<i32>(256)) + vec3<i32>(256)) % vec3<i32>(256);
  let local = vec3<u32>(m);
  var node = cb;
  for (var cur = 0u; cur < level; cur = cur + 1u) {
    let mask_lo = b_struct[node];
    let mask_hi = b_struct[node + 1u];
    let pal = b_struct[node + 2u] & 0xFFFFu;
    let shift = 8u - (cur + 1u) * 2u;
    let cx = (local.x >> shift) & 3u;
    let cy = (local.y >> shift) & 3u;
    let cz = (local.z >> shift) & 3u;
    let cidx = cz * 16u + cy * 4u + cx;
    let in_hi = cidx >= 32u;
    let word = select(mask_lo, mask_hi, in_hi);
    let bit_in = select(cidx, cidx - 32u, in_hi);
    if (((word >> bit_in) & 1u) == 0u) {
      return select(1u, 0u, pal == 0u);
    }
    let child = cb + b_struct[node + NODE_FIXED_WORDS + ddgi_pop_below(cidx, mask_lo, mask_hi)];
    if (cur + 1u == level) {
      // 均匀节点快速路径：mask 全 0 → palette 直决
      if (b_struct[child] == 0u && b_struct[child + 1u] == 0u) {
        return select(1u, 0u, (b_struct[child + 2u] & 0xFFFFu) == 0u);
      }
      if (level == 1u) { return ddgi_agg64(cb, child); }
      if (level == 2u) { return ddgi_agg16(cb, child); }
      return ddgi_agg4(cb, child);
    }
    node = child;
  }
  return 0u;
}

fn ddgi_cell_state_sized(g: Grid, cmin: vec3<i32>, size: i32) -> u32 {
  // 4³ 与 16³ 都是分裂树原生级（LEVEL_EXTENT 的 level 3 / 2）。
  if (size == 4) { return ddgi_brick_state(g, cmin, 3u); }
  if (size == 16) { return ddgi_brick_state(g, cmin, 2u); }
  if (size == 64) { return ddgi_brick_state(g, cmin, 1u); }
  // 按 16³ / 64³ 子块聚合（size ≥64 走 64³）。
  let use64 = size >= 64;
  let sub_size = select(16i, 64i, use64);
  let hlevel = select(2u, 1u, use64);
  let n = max(size / sub_size, 1);
  var seen: u32 = 3u;
  for (var k = 0; k < n; k = k + 1) {
    for (var j = 0; j < n; j = j + 1) {
      for (var i = 0; i < n; i = i + 1) {
        let sub = cmin + vec3<i32>(i, j, k) * sub_size;
        seen = ddgi_merge_state(seen, ddgi_brick_state(g, sub, hlevel));
        if (seen == 2u) { return 2u; }
      }
    }
  }
  return seen;
}

fn ddgi_pcg(v: u32) -> u32 {
  let state = v * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}
fn ddgi_ray_rand(probe_id: u32, frame: u32, ray_index: u32) -> u32 {
  return ddgi_pcg(probe_id ^ ddgi_pcg(frame ^ ddgi_pcg(ray_index)));
}
fn ddgi_rand2(r: u32) -> vec2<f32> {
  return vec2<f32>(f32(r & 0xFFFFu), f32(r >> 16u)) / 65536.0;
}

fn ddgi_snz(v: f32) -> f32 {
  return select(-1.0, 1.0, v >= 0.0);
}
fn ddgi_oct_encode(n: vec3<f32>) -> vec2<f32> {
  let d = n / (abs(n.x) + abs(n.y) + abs(n.z));
  let p = d.xy;
  if (d.z < 0.0) {
    return vec2<f32>((1.0 - abs(p.y)) * ddgi_snz(p.x), (1.0 - abs(p.x)) * ddgi_snz(p.y));
  }
  return p;
}
fn ddgi_oct_decode(e: vec2<f32>) -> vec3<f32> {
  var n = vec3<f32>(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
  if (n.z < 0.0) {
    let t = vec2<f32>((1.0 - abs(n.y)) * ddgi_snz(n.x), (1.0 - abs(n.x)) * ddgi_snz(n.y));
    n.x = t.x;
    n.y = t.y;
  }
  return normalize(n);
}
fn ddgi_oct_texel_dir(tx: u32, ty: u32, s: u32) -> vec3<f32> {
  let e = (vec2<f32>(f32(tx), f32(ty)) + vec2<f32>(0.5)) / f32(s) * 2.0 - vec2<f32>(1.0);
  return ddgi_oct_decode(e);
}

// ---- 世界网格 slot 索引 ----
fn ddgi_lod_cell_size(lod: u32) -> i32 {
  // cell 边长的权威来源是 uniform（Rust 侧 DDGI_LOD_CELL_SIZES）。
  return ddgi_u.lods[lod].origin.w;
}
fn ddgi_lod_count(lod: u32) -> u32 {
  // 槽数 = 下一级的 slot_base − 本级 slot_base（末级取 total_slots）。段池级（LOD0 / chunk 级）的
  // 槽数是池高水位，这条差值公式自动成立，不需要按级特判。
  let base = ddgi_lod_slot_base(lod);
  if (lod + 1u < DDGI_LOD_COUNT) { return ddgi_lod_slot_base(lod + 1u) - base; }
  return u32(ddgi_u.misc.y) - base;
}
fn ddgi_lod_slot_base(lod: u32) -> u32 {
  return ddgi_u.lods[lod].dims.w;
}
// ---- chunk 级（`DDGI_CHUNK_LOD`）的段编址 ----
// cell_slot buffer 尾部打包两张表：
//   [total_slots, +num_chunks)               chunk_base：chunk 线性下标 → 段基址
//   [total_slots+num_chunks, +chunk_slots)   slot_chunk：chunk 级局部槽 → chunk 线性下标
fn ddgi_chunk_axis() -> i32 { return ddgi_u.chunk.origin.w; }
fn ddgi_chunk_count() -> u32 {
  let d = ddgi_u.chunk.dims.xyz;
  return d.x * d.y * d.z;
}
fn ddgi_chunk_base_off() -> u32 { return u32(ddgi_u.misc.y); }
fn ddgi_slot_chunk_off() -> u32 { return u32(ddgi_u.misc.y) + ddgi_chunk_count(); }
/// 向下取整除法（WGSL 的 i32 `/` 向零截断）；chunk 坐标必须用它。
fn ddgi_floor_div(a: i32, b: i32) -> i32 {
  return select(a / b, (a - b + 1) / b, a < 0);
}
fn ddgi_rem_pos(v: i32, a: i32) -> i32 { return v - ddgi_floor_div(v, a) * a; }
/// 「世界某粒度坐标」→ chunk 网格坐标（可为越界值，调用方须过 ddgi_chunk_in_grid）。
/// `per_axis` = 每 chunk 每轴该粒度的单位数（chunk 级 = cell 数、LOD0 = 砖数）。
fn ddgi_chunk_of_unit(u: vec3<i32>, per_axis: i32) -> vec3<i32> {
  let o = ddgi_u.chunk.origin.xyz;
  return vec3<i32>(
    ddgi_floor_div(u.x, per_axis) - o.x,
    ddgi_floor_div(u.y, per_axis) - o.y,
    ddgi_floor_div(u.z, per_axis) - o.z,
  );
}
/// 世界 cell（chunk 级粒度）→ chunk 网格坐标（可为越界值，调用方须过 ddgi_chunk_in_grid）
fn ddgi_wc_to_chunk(wc: vec3<i32>) -> vec3<i32> {
  return ddgi_chunk_of_unit(wc, ddgi_chunk_axis());
}
fn ddgi_chunk_linear(cc: vec3<i32>) -> u32 {
  let d = ddgi_u.chunk.dims.xyz;
  return u32(cc.x) + u32(cc.y) * d.x + u32(cc.z) * d.x * d.y;
}
fn ddgi_chunk_in_grid(cc: vec3<i32>) -> bool {
  let d = vec3<i32>(ddgi_u.chunk.dims.xyz);
  return all(cc >= vec3<i32>(0)) && all(cc < d);
}
/// chunk 级的段基址（`DDGI_NO_PROBE` = 该 chunk 没领段）
fn ddgi_chunk_base_of(cc: vec3<i32>) -> u32 {
  return ddgi_cell_slot[ddgi_chunk_base_off() + ddgi_chunk_linear(cc)];
}
/// chunk 级局部 cell 线性下标（chunk 内，x + y*axis + z*axis²）
fn ddgi_chunk_local_linear(wc: vec3<i32>) -> u32 {
  let a = ddgi_chunk_axis();
  let lx = u32(ddgi_rem_pos(wc.x, a));
  let ly = u32(ddgi_rem_pos(wc.y, a));
  let lz = u32(ddgi_rem_pos(wc.z, a));
  let ua = u32(a);
  return lx + ly * ua + lz * ua * ua;
}

// ---- LOD0（最细级）的砖段表 ----
/// 每 chunk 的砖表项数（每轴砖数³）。
fn ddgi_lod0_bricks_per_chunk() -> u32 {
  let a = u32(DDGI_LOD0_BRICKS_PER_AXIS);
  return a * a * a;
}
/// `ddgi_lod0` 里「槽 → 全局砖线性号」区的起始 word。
fn ddgi_lod0_slot_brick_off() -> u32 { return ddgi_chunk_count(); }
/// 世界 4-voxel cell 所属 16³ 砖的段基址；`DDGI_NO_PROBE` = 该砖没领段（其 cell 不可编址）。
fn ddgi_lod0_seg_of(wc4: vec3<i32>) -> u32 {
  let a = DDGI_LOD0_BRICKS_PER_AXIS;
  let c = DDGI_LOD0_CELLS_PER_BRICK;
  let br = vec3<i32>(
    ddgi_floor_div(wc4.x, c),
    ddgi_floor_div(wc4.y, c),
    ddgi_floor_div(wc4.z, c),
  );
  let cc = ddgi_chunk_of_unit(br, a);
  if (!ddgi_chunk_in_grid(cc)) { return DDGI_NO_PROBE; }
  let off = ddgi_lod0[ddgi_chunk_linear(cc)];
  if (off == DDGI_NO_PROBE) { return DDGI_NO_PROBE; }
  let ua = u32(a);
  let lin = u32(ddgi_rem_pos(br.x, a)) + u32(ddgi_rem_pos(br.y, a)) * ua
    + u32(ddgi_rem_pos(br.z, a)) * ua * ua;
  return ddgi_lod0[off + lin];
}
/// 4-voxel cell 在它所属砖内的局部线性下标（x + y*C + z*C²）。
fn ddgi_lod0_local_linear(wc4: vec3<i32>) -> u32 {
  let c = DDGI_LOD0_CELLS_PER_BRICK;
  let uc = u32(c);
  return u32(ddgi_rem_pos(wc4.x, c)) + u32(ddgi_rem_pos(wc4.y, c)) * uc
    + u32(ddgi_rem_pos(wc4.z, c)) * uc * uc;
}
/// LOD0 槽（级内下标）→ 该槽覆盖的世界 4-voxel cell。
fn ddgi_lod0_slot_world_cell(idx: u32) -> vec3<i32> {
  let g = ddgi_lod0[ddgi_lod0_slot_brick_off() + idx];
  let npc = ddgi_lod0_bricks_per_chunk();
  let d = ddgi_u.chunk.dims.xyz;
  let ci = g / npc;
  let lin = g % npc;
  let cc = vec3<i32>(i32(ci % d.x), i32((ci / d.x) % d.y), i32(ci / (d.x * d.y)));
  let a = DDGI_LOD0_BRICKS_PER_AXIS;
  let ua = u32(a);
  let bl = vec3<i32>(i32(lin % ua), i32((lin / ua) % ua), i32(lin / (ua * ua)));
  let br = (ddgi_u.chunk.origin.xyz + cc) * a + bl;
  let c = DDGI_LOD0_CELLS_PER_BRICK;
  let uc = u32(c);
  let l = idx % DDGI_LOD0_SLOTS_PER_BRICK;
  let lc = vec3<i32>(i32(l % uc), i32((l / uc) % uc), i32(l / (uc * uc)));
  return br * c + lc;
}

/// `p` 处是否「三格厚」：自身是空气，且沿三个轴都各有一个相邻空气（= 一个 7 体素的十字）。
/// 这是细级（cell 边长 4）唯一放得下探针的形状：比 `ddgi_block_empty(p, 3)`（27 体素的全空立方）弱 ——
/// 3 voxel 宽的过道里，全空立方会被 4 voxel 网格的相位卡住（截面刚好跨格时 cell 内找不到 3³），
/// 十字判定则总能落在过道里。仍然排除 1~2 voxel 的墙缝（薄板/裂缝）：那里沿某一轴凑不出三格空气。
fn ddgi_air_cross(g: Grid, p: vec3<i32>) -> bool {
  if (world_solid(g, p)) { return false; }
  return !world_solid(g, p + vec3<i32>(1, 0, 0))
    && !world_solid(g, p + vec3<i32>(-1, 0, 0))
    && !world_solid(g, p + vec3<i32>(0, 1, 0))
    && !world_solid(g, p + vec3<i32>(0, -1, 0))
    && !world_solid(g, p + vec3<i32>(0, 0, 1))
    && !world_solid(g, p + vec3<i32>(0, 0, -1));
}

/// 从 `v` 起沿 `d` 方向连续空气的体素数（含 `v` 自身；撞到实体或走满 `cap` 停）。
fn ddgi_air_run(g: Grid, v: vec3<i32>, d: vec3<i32>, cap: i32) -> i32 {
  var n = 0;
  var p = v;
  for (var i = 0; i < cap; i = i + 1) {
    if (world_solid(g, p)) { break; }
    n = n + 1;
    p = p + d;
  }
  return n;
}

/// 沿 ±`d` 量空腔宽度：**任一侧超出量程（`cap`）⇒ 该轴开阔**，不算窄；否则宽度落在
/// `[DDGI_PROBE_MIN_BLOCK, DDGI_CLAIM_CAVITY]` 才算窄。
/// 量程判断不能省：只退一侧为 0（朝表面那侧）时，另一侧被 cap 截断会算出恰好 = CAVITY，
/// 把"开阔"误判成"窄"（开阔地面就是这样）。
fn ddgi_axis_narrow(g: Grid, v: vec3<i32>, d: vec3<i32>, cap: i32) -> bool {
  let ru = ddgi_air_run(g, v, d, cap);
  if (ru >= cap) { return false; }
  let rd = ddgi_air_run(g, v, -d, cap);
  if (rd >= cap) { return false; }
  let w = ru + rd - 1;
  return w >= DDGI_PROBE_MIN_BLOCK && w <= DDGI_CLAIM_CAVITY;
}

/// 「这一处需要细级」的**几何**判据：**三个轴里至少有一个**方向的空腔宽度落在
/// `[DDGI_PROBE_MIN_BLOCK, DDGI_CLAIM_CAVITY]` 之间 —— 窄过道 / 低矮夹层 / 立缝 / 拱下。
/// 三个轴全测，包括"朝向表面那一轴"：那一侧行程为 0，宽度就等于另一侧的开放空间，
/// 正好是"头顶高度 / 身前空间"这种量（早期的版本把它排除了，于是"宽 >16 但矮 ≤16"的过道
/// 地板/低处永远不认领）。
/// 开阔地形、大厅、悬崖：三轴全开阔 ⇒ 不认领；1~2 voxel 的缝 ⇒ 不认领（十字档也放不下）。
/// 只用几何、不看亮度：与启动/编辑造成的瞬态无关（那些"到处都黑"的阶段不是"分辨率不够"）。
fn ddgi_cavity_narrow(g: Grid, p: vec3<f32>) -> bool {
  let v = vec3<i32>(floor(p));
  let cap = DDGI_CLAIM_CAVITY + 1;
  return ddgi_axis_narrow(g, v, vec3<i32>(1, 0, 0), cap)
    || ddgi_axis_narrow(g, v, vec3<i32>(0, 1, 0), cap)
    || ddgi_axis_narrow(g, v, vec3<i32>(0, 0, 1), cap);
}

/// LOD0 段分配游标 = 已领的槽数（= 下一个空闲段基址）。游标之外的槽位没人领过。
fn ddgi_lod0_next() -> u32 {
  return atomicLoad(&ddgi_lod0_state[DDGI_LOD0_STATE_ALLOC]);
}
/// 认领位图的一位（chunk 网格线性下标 + chunk 内砖线性号 → bit）。返回是否是本线程抢到的。
/// 弱 CAS 允许偶发失败：失败即本轮放弃，下一帧由别的着色点重试（认领没有副作用）。
fn ddgi_lod0_claim_bit(cit: u32, lin: u32) -> bool {
  let words = ddgi_lod0_bricks_per_chunk() / 32u;
  let w = DDGI_LOD0_STATE_CLAIM + cit * words + (lin >> 5u);
  let bit = 1u << (lin & 31u);
  let old = atomicLoad(&ddgi_lod0_state[w]);
  if ((old & bit) != 0u) { return false; }
  return atomicCompareExchangeWeak(&ddgi_lod0_state[w], old, old | bit).exchanged;
}

/// 4³ cell → 它所属瓦片的定位键 `(chunk 网格线性下标, chunk 内瓦片线性号, 1)`；第 3 位 =
/// 该 chunk 有没有瓦片表（0 = 网格外 / 没表 ⇒ 前两位无意义）。`ddgi_lod0_claim` 与调试视图共用。
fn ddgi_lod0_tile_key(cell: vec3<i32>) -> vec3<u32> {
  let a = DDGI_LOD0_BRICKS_PER_AXIS;
  let t = DDGI_LOD0_CELLS_PER_BRICK;
  let br = vec3<i32>(
    ddgi_floor_div(cell.x, t),
    ddgi_floor_div(cell.y, t),
    ddgi_floor_div(cell.z, t),
  );
  let cc = ddgi_chunk_of_unit(br, a);
  if (!ddgi_chunk_in_grid(cc)) { return vec3<u32>(0u, 0u, 0u); }
  let cit = ddgi_chunk_linear(cc);
  if (ddgi_lod0[cit] == DDGI_NO_PROBE) { return vec3<u32>(0u, 0u, 0u); }
  let ua = u32(a);
  let lin = u32(ddgi_rem_pos(br.x, a)) + u32(ddgi_rem_pos(br.y, a)) * ua
    + u32(ddgi_rem_pos(br.z, a)) * ua * ua;
  return vec3<u32>(cit, lin, 1u);
}

/// 认领位是否已置（含「已放弃」：领段后判定失败、或容量满）。只读，供热路径外诊断用。
fn ddgi_lod0_claim_seen(cit: u32, lin: u32) -> bool {
  let words = ddgi_lod0_bricks_per_chunk() / 32u;
  let w = DDGI_LOD0_STATE_CLAIM + cit * words + (lin >> 5u);
  return (atomicLoad(&ddgi_lod0_state[w]) & (1u << (lin & 31u))) != 0u;
}

/// 认领 `p` 所在的 8³ 瓦片，并当场给它领一段（写反向表，最后写瓦片表基址）。
/// `n` = 该点所在表面的法线（轴对齐）：用来沿法线往空气里退，检查附近能否放下细级探针。
/// 只在 `ddgi_sample` 各级全落空（拿不到任何有效样点）时调用；段一旦领到就不再变，探针位置因此不随相机移动。
/// 已有段 / 已被别人认领 / 该 chunk 没瓦片表 / 附近放不下探针 / 预留容量满 → 直接返回。
fn ddgi_lod0_claim(p: vec3<f32>, n: vec3<f32>) {
  // 两个尺度必须分清：cell 边长 = 本级的 `origin.w`（4），瓦片边长 = 每瓦片 cell 数 × cell 边长（8）。
  // `ddgi_lod0_seg_of` / `ddgi_lod0_tile_key` 收的都是 **4 voxel cell 坐标**。
  let cs = ddgi_u.lods[0u].origin.w;
  let vf = vec3<i32>(floor(p));
  let cell = vec3<i32>(
    ddgi_floor_div(vf.x, cs),
    ddgi_floor_div(vf.y, cs),
    ddgi_floor_div(vf.z, cs),
  );
  if (ddgi_lod0_seg_of(cell) != DDGI_NO_PROBE) { return; }
  let key = ddgi_lod0_tile_key(cell);
  if (key.z == 0u) { return; }
  let cit = key.x;
  let lin = key.y;
  if (!ddgi_lod0_claim_bit(cit, lin)) { return; }
  // 门槛：附近得真能放下一个细级探针（十字档），否则领段也是白领 —— 见 `DDGI_LOD0_CLAIM_BIAS`。
  // 沿法线往空气里退 1..BIAS voxel，任一位置能摆下十字就放行。判定失败也**不清认领位**：
  // 这块瓦片以后不再重试（树叶/栅栏这类多孔表面一个探针也放不下，试也是浪费时间）。
  let gw = make_grid(0u);
  let step = vec3<i32>(n);
  if (any(step != vec3<i32>(0))) {
    let here = vec3<i32>(floor(p));
    var placeable = false;
    for (var k = 1; k <= DDGI_LOD0_CLAIM_BIAS; k = k + 1) {
      if (ddgi_air_cross(gw, here + step * k)) {
        placeable = true;
        break;
      }
    }
    if (!placeable) { return; }
  }
  let seg = atomicAdd(&ddgi_lod0_state[DDGI_LOD0_STATE_ALLOC], DDGI_LOD0_SLOTS_PER_BRICK);
  // 容量满：认领位已置 ⇒ 这块瓦片以后不再重试（预留区间见 `ddgi_lod_count(0u)`）。
  if (seg + DDGI_LOD0_SLOTS_PER_BRICK > ddgi_lod_count(0u)) { return; }
  let g = cit * ddgi_lod0_bricks_per_chunk() + lin;
  for (var k = 0u; k < DDGI_LOD0_SLOTS_PER_BRICK; k = k + 1u) {
    ddgi_lod0[ddgi_lod0_slot_brick_off() + seg + k] = g;
  }
  // 瓦片表基址最后写：并发读者在它落地前一律把该瓦片当「未领段」。
  ddgi_lod0[ddgi_lod0[cit] + lin] = seg;
}

// ---- 槽位映射（按 lod 分派：LOD0 → 砖段表；chunk 级 → chunk 局部；其余 → 世界锚定环面）----
/// 世界 cell → 槽位的纯算术映射；读某个世界 cell 自己的烘焙记录/探针位置时必须用它。
/// LOD0 / chunk 级在该 cell 不可编址时返回 0，调用方必须先过 `ddgi_cell_in_window`。
fn ddgi_slot_own(lod: u32, wc: vec3<i32>) -> u32 {
  // 段池级（LOD0 / chunk 级）的段基址是「池内」下标，必须再加本级的 `slot_base` 才是全局槽位
  // ——槽位编号在图集/数组里是全局的，`ddgi_slot_world_cell` 收到的也是「全局槽位 − slot_base」。
  if (lod == 0u) {
    let seg = ddgi_lod0_seg_of(wc);
    if (seg == DDGI_NO_PROBE) { return 0u; }
    // 预留区间被压缩（chunk 池长大）时，超出当前区间的老段一律当未领段。
    if (seg + DDGI_LOD0_SLOTS_PER_BRICK > ddgi_lod_count(0u)) { return 0u; }
    return ddgi_lod_slot_base(lod) + seg + ddgi_lod0_local_linear(wc);
  }
  if (lod == DDGI_CHUNK_LOD) {
    let cc = ddgi_wc_to_chunk(wc);
    if (!ddgi_chunk_in_grid(cc)) { return 0u; }
    let base = ddgi_chunk_base_of(cc);
    if (base == DDGI_NO_PROBE) { return 0u; }
    return ddgi_lod_slot_base(lod) + base + ddgi_chunk_local_linear(wc);
  }
  let d = ddgi_u.lods[lod].dims.xyz;
  let di = vec3<i32>(d);
  let r = ((wc % di) + di) % di;
  return ddgi_lod_slot_base(lod) + u32(r.x) + u32(r.y) * d.x + u32(r.z) * d.x * d.y;
}
fn ddgi_slot(lod: u32, wc: vec3<i32>) -> u32 {
  return ddgi_slot_own(lod, wc);
}
/// 「屏幕使用」粘滞标记：把该槽位置为满值（`DDGI_USAGE_STICKY`，`ddgi_sort` 每帧 -1）。
/// 必须只在着色路径调用。
fn ddgi_usage_mark(slot: u32) {
  if (ddgi_cell_slot[slot] != DDGI_USAGE_STICKY) {
    ddgi_cell_slot[slot] = DDGI_USAGE_STICKY;
  }
}
/// 窗口原点（cell 单位）：origin.xyz 按 cs 对齐，整除精确；只服务按空间盒判定窗口的级。
fn ddgi_origin_cell(lod: u32) -> vec3<i32> {
  return ddgi_u.lods[lod].origin.xyz / ddgi_u.lods[lod].origin.w;
}
/// 槽位在本级内的下标 → 该槽位当前覆盖的世界 cell
fn ddgi_slot_world_cell(lod: u32, idx: u32) -> vec3<i32> {
  if (lod == 0u) { return ddgi_lod0_slot_world_cell(idx); }
  if (lod == DDGI_CHUNK_LOD) {
    let cl = ddgi_cell_slot[ddgi_slot_chunk_off() + idx];
    let base = ddgi_cell_slot[ddgi_chunk_base_off() + cl];
    let l = idx - base;
    let a = u32(ddgi_chunk_axis());
    let lc = vec3<i32>(i32(l % a), i32((l / a) % a), i32(l / (a * a)));
    let d = ddgi_u.chunk.dims.xyz;
    let cc = vec3<i32>(i32(cl % d.x), i32((cl / d.x) % d.y), i32(cl / (d.x * d.y)));
    return (ddgi_u.chunk.origin.xyz + cc) * ddgi_chunk_axis() + lc;
  }
  let di = vec3<i32>(ddgi_u.lods[lod].dims.xyz);
  let o = ddgi_origin_cell(lod);
  let r = vec3<i32>(ddgi_slot_cell(lod, idx));
  return o + ((r - o) % di + di) % di;
}
/// 世界 cell 是否在本级可编址窗口内；LOD0 = 所属砖领了段、chunk 级 = 空间盒 ∩ 该 chunk 领了段。
fn ddgi_cell_in_window(lod: u32, wc: vec3<i32>) -> bool {
  if (lod == 0u) { return ddgi_lod0_seg_of(wc) != DDGI_NO_PROBE; }
  if (lod == DDGI_CHUNK_LOD) {
    // 世界 AABB 对齐的空间盒 ∩ 该 cell 所属 chunk 已领段（两层都要判）。
    let c = wc - ddgi_origin_cell(DDGI_CHUNK_LOD);
    if (!(all(c >= vec3<i32>(0)) && all(c < vec3<i32>(ddgi_u.lods[DDGI_CHUNK_LOD].dims.xyz)))) {
      return false;
    }
    let cc = ddgi_wc_to_chunk(wc);
    if (!ddgi_chunk_in_grid(cc)) { return false; }
    return ddgi_chunk_base_of(cc) != DDGI_NO_PROBE;
  }
  let c = wc - ddgi_origin_cell(lod);
  return all(c >= vec3<i32>(0)) && all(c < vec3<i32>(ddgi_u.lods[lod].dims.xyz));
}
/// 槽位在本级内的下标 → 局部 cell 坐标（LOD0 = 砖内 0..4 / chunk 内 0..axis / 网格内 0..dims）
fn ddgi_slot_cell(lod: u32, idx: u32) -> vec3<u32> {
  if (lod == 0u) {
    let uc = u32(DDGI_LOD0_CELLS_PER_BRICK);
    let l = idx % DDGI_LOD0_SLOTS_PER_BRICK;
    return vec3<u32>(l % uc, (l / uc) % uc, l / (uc * uc));
  }
  if (lod == DDGI_CHUNK_LOD) {
    let cl = ddgi_cell_slot[ddgi_slot_chunk_off() + idx];
    let base = ddgi_cell_slot[ddgi_chunk_base_off() + cl];
    let l = idx - base;
    let a = u32(ddgi_chunk_axis());
    return vec3<u32>(l % a, (l / a) % a, l / (a * a));
  }
  let d = ddgi_u.lods[lod].dims;
  return vec3<u32>(idx % d.x, (idx / d.x) % d.y, idx / (d.x * d.y));
}
fn ddgi_slot_lod_of(slot: u32) -> u32 {
  for (var l = DDGI_LOD_COUNT; l > 1u; l = l - 1u) {
    if (slot >= ddgi_lod_slot_base(l - 1u)) { return l - 1u; }
  }
  return 0u;
}
/// 探针在图集层内的 (x, y) 坐标（层号 = id / DDGI_PROBES_PER_LAYER）
fn ddgi_probe_in_layer(id: u32) -> vec2<u32> {
  let in_layer = id % DDGI_PROBES_PER_LAYER;
  return vec2<u32>(in_layer % DDGI_PROBES_PER_LAYER_AXIS, in_layer / DDGI_PROBES_PER_LAYER_AXIS);
}
fn ddgi_irr_coord(id: u32, tx: u32, ty: u32) -> vec3<u32> {
  let p = ddgi_probe_in_layer(id);
  return vec3<u32>(id / DDGI_PROBES_PER_LAYER, p.x * DDGI_IRR_TEXELS + tx, p.y * DDGI_IRR_TEXELS + ty);
}
fn ddgi_depth_coord(id: u32, tx: u32, ty: u32) -> vec3<u32> {
  let p = ddgi_probe_in_layer(id);
  return vec3<u32>(id / DDGI_PROBES_PER_LAYER, p.x * DDGI_DEPTH_TEXELS + tx, p.y * DDGI_DEPTH_TEXELS + ty);
}

// ---- 烘焙记录 / meta 打包 ----
// cell record: bit0 = 探针存在（cell 非全满）、bit1 = cell 有体素（非全空）、bit8.. = cell 内归一化偏移
const DDGI_REC_ENABLED: u32 = 1u;
const DDGI_REC_OCCUPIED: u32 = 2u;
fn ddgi_rec_pack(enabled: u32, occupied: u32, off_b: vec3<u32>) -> u32 {
  return (enabled & 1u) | ((occupied & 1u) << 1u)
    | ((off_b.x & 0xFFu) << 8u)
    | ((off_b.y & 0xFFu) << 16u)
    | ((off_b.z & 0xFFu) << 24u);
}
fn ddgi_rec_off(rec: u32) -> vec3<u32> {
  return vec3<u32>((rec >> 8u) & 0xFFu, (rec >> 16u) & 0xFFu, (rec >> 24u) & 0xFFu);
}
// meta word = age(低 8bit) | ENABLED | ACTIVE | 邻接可见性掩码(bit10-15) | NBR_VALID(bit16)
//   ENABLED：烘焙放了探针；ACTIVE：本 cell 或 6 邻接有体素/物体且未被更细 LOD 覆盖。
//   NBR/NBR_VALID：6 邻接可见性掩码及其「已算过」位（未置位时消费侧按被挡处理）。
fn ddgi_meta_pack(age: u32, enabled: bool, is_active: bool) -> u32 {
  return (age & 0xFFu)
    | select(0u, DDGI_META_ENABLED, enabled)
    | select(0u, DDGI_META_ACTIVE, is_active);
}
/// 把本帧算出的邻接可见性掩码写进 meta（同时置 NBR_VALID）。
fn ddgi_meta_with_nbr(p: u32, nbr_mask: u32) -> u32 {
  return (p & ~(DDGI_META_NBR_MASK << DDGI_META_NBR_SHIFT) & ~DDGI_META_NBR_VALID)
    | ((nbr_mask & DDGI_META_NBR_MASK) << DDGI_META_NBR_SHIFT)
    | DDGI_META_NBR_VALID;
}
fn ddgi_meta_age(p: u32) -> u32 {
  return p & 0xFFu;
}
/// 读回邻接可见性掩码（空间混合门控用）。
fn ddgi_meta_nbr(p: u32) -> u32 {
  return (p >> DDGI_META_NBR_SHIFT) & DDGI_META_NBR_MASK;
}
/// 6 邻接方向单位向量，位序：0=-X,1=+X,2=-Y,3=+Y,4=-Z,5=+Z。
fn ddgi_nbr_dir(a: u32) -> vec3<f32> {
  if (a == 0u) { return vec3<f32>(-1.0, 0.0, 0.0); }
  if (a == 1u) { return vec3<f32>(1.0, 0.0, 0.0); }
  if (a == 2u) { return vec3<f32>(0.0, -1.0, 0.0); }
  if (a == 3u) { return vec3<f32>(0.0, 1.0, 0.0); }
  if (a == 4u) { return vec3<f32>(0.0, 0.0, -1.0); }
  return vec3<f32>(0.0, 0.0, 1.0);
}
/// chunk 级分支邻居循环下标（+X,-X,+Y,-Y,+Z,-Z）→ 掩码位：+X→1, -X→0, +Y→3, -Y→2, +Z→5, -Z→4。
fn ddgi_nbr_bit_chunk(a: u32) -> u32 {
  return (a >> 1u) * 2u + (1u - (a & 1u));
}

fn ddgi_lum(c: vec3<f32>) -> f32 {
  return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}
fn ddgi_maxc(c: vec3<f32>) -> f32 {
  return max(c.x, max(c.y, c.z));
}
/// 时域混合系数：分收敛斜坡（`1/age`）与稳态窗口（`base`）两段，以 `age` 到达上限分界。
fn ddgi_age_alpha(age: u32, base: f32) -> f32 {
  return max(base, 1.0 / f32(max(age, 1u)));
}

fn ddgi_update_texel(prev: vec3<f32>, proj: vec3<f32>) -> vec3<f32> {
  let new_t = pow(proj, vec3<f32>(1.0 / DDGI_IRRAD_GAMMA));
  var h = select(DDGI_HYSTERESIS, 0.0, all(prev == vec3<f32>(0.0)));
  if (ddgi_maxc(prev - new_t) > DDGI_BIG_CHANGE) {
    h = max(h - DDGI_CHANGE_DROP, 0.0);
  }
  var delta = new_t - prev;
  if (ddgi_lum(delta) > DDGI_BRIGHTNESS) {
    delta = delta * DDGI_DELTA_CLAMP;
  }
  var lerp_delta = (1.0 - h) * delta;
  if (ddgi_maxc(new_t) < ddgi_maxc(prev)) {
    let s = vec3<f32>(sign(lerp_delta.x), sign(lerp_delta.y), sign(lerp_delta.z));
    lerp_delta = vec3<f32>(
      clamp(max(abs(lerp_delta.x), DDGI_MIN_STEP), 0.0, abs(delta.x)),
      clamp(max(abs(lerp_delta.y), DDGI_MIN_STEP), 0.0, abs(delta.y)),
      clamp(max(abs(lerp_delta.z), DDGI_MIN_STEP), 0.0, abs(delta.z)),
    ) * s;
  }
  return prev + lerp_delta;
}
