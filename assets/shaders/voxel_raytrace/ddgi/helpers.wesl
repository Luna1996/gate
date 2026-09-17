import package::bindings::{b_struct, ddgi_cell_slot, ddgi_u, g};
import package::brickmap::{Grid};
import package::common::{CHUNK_INDEX_CAP, NODE_FIXED_WORDS};
import package::ddgi::consts::{DDGI_BIG_CHANGE, DDGI_BRIGHTNESS, DDGI_CHANGE_DROP, DDGI_DELTA_CLAMP, DDGI_DEPTH_TEXELS, DDGI_HYSTERESIS, DDGI_IRRAD_GAMMA, DDGI_IRR_TEXELS, DDGI_LOD_COUNT, DDGI_META_ACTIVE, DDGI_META_ENABLED, DDGI_META_NBR_MASK, DDGI_META_NBR_SHIFT, DDGI_META_NBR_VALID, DDGI_MIN_STEP, DDGI_NO_PROBE, DDGI_PROBES_PER_LAYER, DDGI_PROBES_PER_LAYER_AXIS, DDGI_USAGE_STICKY};

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
      // 均匀节点快速路径：mask 全 0 → palette 直决（省 64 次 agg 迭代；空气 16³ 最常见）
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
  if (size == 16) { return ddgi_brick_state(g, cmin, 2u); }
  if (size == 64) { return ddgi_brick_state(g, cmin, 1u); }
  // 按 16³（size 32）或 64³（size 128/256）子块聚合：
  // size 32 → 2³ 个 16³（hlevel 2）；128 → 2³ 个 64³；256 → 4³ 个 64³。
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


// ============================================================================

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
  // LOD0 的槽数是**探针池高水位**（chunk 段之和），不等于空间盒 dims 乘积 —— 后者仍是
  // 级联包含/混合带用的空间盒。LOD1~3 仍是规则网格 → 等于 dims 乘积。
  if (lod == 0u) { return ddgi_u.chunk.dims.w; }
  let d = ddgi_u.lods[lod].dims;
  return d.x * d.y * d.z;
}
fn ddgi_lod_slot_base(lod: u32) -> u32 {
  return ddgi_u.lods[lod].dims.w;
}
// ---- LOD0 的 chunk 段编址 ----
// 【与 LOD1~3 的边界】LOD0 **只**用下面这套 chunk 公式；lod>=1 一律走「世界锚定的环面映射」
// （见 ddgi_slot_own 的后半段）。两套编址互不干扰，唯一交汇点是槽位段顺序：
//   [0, lod0_slots) 属 LOD0（池分配），之后依次是 LOD1/2/3 的规则网格段。
//
// cell_slot buffer 尾部打包了两张 LOD0 表（见 Rust `DdgiGpu::cell_slot`）：
//   [total_slots, +num_chunks)               chunk_base：chunk 线性下标 → 段基址
//   [total_slots+num_chunks, +lod0_slots)    slot_chunk：LOD0 局部槽 → chunk 线性下标
fn ddgi_chunk_axis() -> i32 { return ddgi_u.chunk.origin.w; }
fn ddgi_chunk_count() -> u32 {
  let d = ddgi_u.chunk.dims.xyz;
  return d.x * d.y * d.z;
}
fn ddgi_chunk_base_off() -> u32 { return u32(ddgi_u.misc.y); }
fn ddgi_slot_chunk_off() -> u32 { return u32(ddgi_u.misc.y) + ddgi_chunk_count(); }
/// 向下取整除法（WGSL 的 `/` 对 i32 是**截断**，负数会算错 → chunk 坐标必须用它）
fn ddgi_floor_div(a: i32, b: i32) -> i32 {
  return select(a / b, (a - b + 1) / b, a < 0);
}
fn ddgi_rem_pos(v: i32, a: i32) -> i32 { return v - ddgi_floor_div(v, a) * a; }
/// 世界 cell → chunk 在 chunk 网格内的坐标（可为越界值，调用方须过 ddgi_chunk_in_grid）
fn ddgi_wc_to_chunk(wc: vec3<i32>) -> vec3<i32> {
  let a = ddgi_chunk_axis();
  let o = ddgi_u.chunk.origin.xyz;
  return vec3<i32>(
    ddgi_floor_div(wc.x, a) - o.x,
    ddgi_floor_div(wc.y, a) - o.y,
    ddgi_floor_div(wc.z, a) - o.z,
  );
}
fn ddgi_chunk_linear(cc: vec3<i32>) -> u32 {
  let d = ddgi_u.chunk.dims.xyz;
  return u32(cc.x) + u32(cc.y) * d.x + u32(cc.z) * d.x * d.y;
}
fn ddgi_chunk_in_grid(cc: vec3<i32>) -> bool {
  let d = vec3<i32>(ddgi_u.chunk.dims.xyz);
  return all(cc >= vec3<i32>(0)) && all(cc < d);
}
/// chunk 的 LOD0 段基址（`DDGI_NO_PROBE` = 该 chunk 没领段）
fn ddgi_chunk_base_of(cc: vec3<i32>) -> u32 {
  return ddgi_cell_slot[ddgi_chunk_base_off() + ddgi_chunk_linear(cc)];
}
/// LOD0 局部 cell 线性下标（chunk 内，x + y*axis + z*axis²）
fn ddgi_lod0_local_linear(wc: vec3<i32>) -> u32 {
  let a = ddgi_chunk_axis();
  let lx = u32(ddgi_rem_pos(wc.x, a));
  let ly = u32(ddgi_rem_pos(wc.y, a));
  let lz = u32(ddgi_rem_pos(wc.z, a));
  let ua = u32(a);
  return lx + ly * ua + lz * ua * ua;
}

// ---- 槽位映射（按 lod 分派：lod==0 → chunk 局部；lod>=1 → 世界锚定环面）----
// 槽位与世界 cell 的身份绑定与相机无关：相机滚动只让新进入窗口的那条带换掉世界 cell
// （旧数据本就该丢），其余槽位保持自己的身份 ⇒ 图集不因相机移动而整体失效。
/// 世界 cell → 槽位的**纯算术**映射。读"某个世界 cell **自己**的烘焙记录/探针位置"的地方
/// （判定邻接占用、粗级继承细级探针位置）必须用它。
/// ⚠️ lod==0 且该 chunk 没领段 / 越界时返回 0（不是一个有意义的槽位，只保证不越界读）：
/// 调用方**必须**先用 `ddgi_cell_in_window` 过滤。
fn ddgi_slot_own(lod: u32, wc: vec3<i32>) -> u32 {
  if (lod == 0u) {
    let cc = ddgi_wc_to_chunk(wc);
    if (!ddgi_chunk_in_grid(cc)) { return 0u; }
    let base = ddgi_chunk_base_of(cc);
    if (base == DDGI_NO_PROBE) { return 0u; }
    return base + ddgi_lod0_local_linear(wc);
  }
  let d = ddgi_u.lods[lod].dims.xyz;
  let di = vec3<i32>(d);
  let r = ((wc % di) + di) % di;
  return ddgi_lod_slot_base(lod) + u32(r.x) + u32(r.y) * d.x + u32(r.z) * d.x * d.y;
}
fn ddgi_slot(lod: u32, wc: vec3<i32>) -> u32 {
  return ddgi_slot_own(lod, wc);
}
/// 「屏幕使用」粘滞标记：把该槽位置为**满值**（`DDGI_USAGE_STICKY`，`ddgi_sort` 每帧 -1）。
///
/// 由**着色**路径在每个"候选角"上调用（见 `ddgi_sample_lod` 的闸门之后 / 图集取样之前），
/// 代表"本帧有像素会用这个探针的数据"。工作集选择（sort）就靠它 —— 见 DDGI_USAGE_STICKY：
/// 射线的去处跟着像素走，而不是按距离猜。
/// 置满值（而非 +1）才能让记账的密度不决定窗口长度：footprint 很小的探针每 ~4 帧才被点到一次，
/// 按 +1 计会被逐帧抹平 ⇒ 它进进出出工作集、`age` 长期停在 12~32 ⇒ 大增益 ⇒ 逐帧跳变。
/// 满值写入是幂等的，故不必用原子（原子反而把记账变成独占访问，相邻像素会撞同一个 cache line）。
/// 必须只在**着色**路径调用：cast 回读 GI 也走 `ddgi_sample`，那里记账会形成"射线多的探针拿到
/// 更多射线"的正反馈。所有可能给出显示值的 `ddgi_sample_lod` 调用都要记（否则只被兜底路径采样
/// 到的探针永远拿不到射线）。
fn ddgi_usage_mark(slot: u32) {
  // 先读一次只是为了省重复写：满值时不再写（记账点每帧几十万次，读比写便宜）。
  if (ddgi_cell_slot[slot] != DDGI_USAGE_STICKY) {
    ddgi_cell_slot[slot] = DDGI_USAGE_STICKY;
  }
}
/// 窗口原点（cell 单位）。origin.xyz 已按 cs 对齐，故整除精确。（LOD0 的编址原点在
/// `ddgi_u.chunk`，不在这里；本函数只服务 lod>=1 与 LOD0 的**空间盒**包含判定。）
fn ddgi_origin_cell(lod: u32) -> vec3<i32> {
  return ddgi_u.lods[lod].origin.xyz / ddgi_u.lods[lod].origin.w;
}
/// 槽位在本级内的下标 → 该槽位**当前**覆盖的世界 cell
fn ddgi_slot_world_cell(lod: u32, idx: u32) -> vec3<i32> {
  if (lod == 0u) {
    // LOD0：反查表给出该槽所属 chunk，再解 chunk 内局部 cell。
    // （槽位池没有空洞时 idx∈[0,lod0_slots) 全部有主；有空洞时这里是"指向 chunk 0"的
    //   无效解 —— 但空洞槽位不会被任何 cell 的 ddgi_slot_own 引用，bake 对其多写一次无害。）
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
/// 世界 cell 是否在本级**可编址**窗口内。LOD0 = 该 cell 所属 chunk **领了段**（内容驱动：
/// 空 chunk 的 cell 没有探针，采样侧必须把它们当"窗口外"剔除，否则会读到槽位 0 的数据）。
fn ddgi_cell_in_window(lod: u32, wc: vec3<i32>) -> bool {
  if (lod == 0u) {
    // LOD0 = 世界 AABB 对齐的空间盒 ∩ 该 cell 所属 chunk 已领段（两层都要判）。
    // 只判"chunk 已领段"不够：chunk 锚定后编址域按 256 对齐放大，AABB 之外那一圈会落在已领段
    // chunk 的 padding 里而被当成窗口内 ⇒ 叠上空间盒后结果只会更少、不会更多。
    let c = wc - ddgi_origin_cell(0u);
    if (!(all(c >= vec3<i32>(0)) && all(c < vec3<i32>(ddgi_u.lods[0].dims.xyz)))) { return false; }
    let cc = ddgi_wc_to_chunk(wc);
    if (!ddgi_chunk_in_grid(cc)) { return false; }
    return ddgi_chunk_base_of(cc) != DDGI_NO_PROBE;
  }
  let c = wc - ddgi_origin_cell(lod);
  return all(c >= vec3<i32>(0)) && all(c < vec3<i32>(ddgi_u.lods[lod].dims.xyz));
}
/// 槽位在本级内的下标 → 局部 cell 坐标（chunk 内 0..axis / 网格内 0..dims）
fn ddgi_slot_cell(lod: u32, idx: u32) -> vec3<u32> {
  if (lod == 0u) {
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
//   ENABLED：烘焙阶段放了探针（有 irr/depth 数据可采样）
//   ACTIVE ：本 cell 或 6 邻接有体素 / 物体，且未被更细 LOD 覆盖 —— 本帧需要投线
//   NBR    ：6 邻接可见性掩码（见 DDGI_META_NBR_SHIFT），由 sort 每帧为活跃探针重算
//   NBR_VALID：掩码是否已算过；bake 与非活跃路径写 0 ⇒ 消费侧按**被挡**处理（保守不混）。
//            未算 ≠ 可见：这类探针本帧未投线、图集值是陈旧的，混进来会把隔墙（含看到天空的）
//            辐照度抹进墙内。
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
/// 读回邻接可见性掩码（`ddgi_probe_blend_irr` 的空间混合门控用）。
fn ddgi_meta_nbr(p: u32) -> u32 {
  return (p >> DDGI_META_NBR_SHIFT) & DDGI_META_NBR_MASK;
}
/// 6 邻接方向单位向量，位序与 `DDGI_META_NBR_SHIFT` 一致：0=-X,1=+X,2=-Y,3=+Y,4=-Z,5=+Z。
/// 同轴两位相邻 ⇒ 反向 = `bit ^ 1u`。
fn ddgi_nbr_dir(a: u32) -> vec3<f32> {
  if (a == 0u) { return vec3<f32>(-1.0, 0.0, 0.0); }
  if (a == 1u) { return vec3<f32>(1.0, 0.0, 0.0); }
  if (a == 2u) { return vec3<f32>(0.0, -1.0, 0.0); }
  if (a == 3u) { return vec3<f32>(0.0, 1.0, 0.0); }
  if (a == 4u) { return vec3<f32>(0.0, 0.0, -1.0); }
  return vec3<f32>(0.0, 0.0, 1.0);
}
/// `ddgi_probe_blend_irr` 的 **LOD0 分支**邻居循环下标（+X,-X,+Y,-Y,+Z,-Z）→ 掩码位。
/// LOD1~3 分支用 `a * 2 + (plus ? 1 : 0)`（两分支的循环结构不同，故各有一个映射）。
/// 注意分号写法：`(a >> 1u) * 2u + (1u - (a & 1u))` —— +X→1, -X→0, +Y→3, -Y→2, +Z→5, -Z→4。
fn ddgi_nbr_bit_lod0(a: u32) -> u32 {
  return (a >> 1u) * 2u + (1u - (a & 1u));
}

fn ddgi_lum(c: vec3<f32>) -> f32 {
  return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}
fn ddgi_maxc(c: vec3<f32>) -> f32 {
  return max(c.x, max(c.y, c.z));
}
/// 时域混合系数：分**收敛斜坡**与**稳态窗口**两段，以 `age` 到达上限（`DDGI_AGE_MAX`）分界。
///
///   · 斜坡段（`age < 1/base`）：`1/age` 的等权滑动平均 —— `age` 就是该探针已积累的 collect
///     次数，故误差按 1/√n 下降。`age ≤ 1` 时为 1.0（精确覆写），换主/编辑时 age 被压到
///     1~8 ⇒ 这几帧快速跟上新位置。
///   · 稳态段（`age ≥ 1/base`）：恒定 `base`（长窗口，负责压稳态逐帧脉动）。
///
/// 写成 `max(base, 1/age)` 而不是两段 if：两者在 `base ≥ 1/DDGI_AGE_MAX` 时等价，但两段 if 在
/// `1/base < DDGI_AGE_MAX`（辐照度 base = 1/64）时**不单调** —— age 从 1/base 涨到上限期间
/// alpha 一路掉到 1/255，到 age == DDGI_AGE_MAX 又跳回 base（此处是 4× 跳变）⇒ 每个探针到龄
/// 那一刻都会突然重新跟随单帧噪声，表现为周期性的闪烁。`max` 形式没有这个跳变。
fn ddgi_age_alpha(age: u32, base: f32) -> f32 {
  return max(base, 1.0 / f32(max(age, 1u)));
}

// ddgi_update_texel：未被调用（collect 直接用 mix(prev, new, a)）。
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
