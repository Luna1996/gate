import package::bindings::{gi_stats, gi_u, lt_event, lt_node, lt_node_key, lt_tile, lt_tile_key, view_u};
import package::brickmap::{Grid, sample_brickmap};
import package::common::{FLAGS_TRANSMISSIVE, MATERIAL_DEBUG_NONE, emissive_radiance, face_normal_from_index, fetch_material, material_emissive, palette_flags};
import package::gi::common::{gi_mix, gi_rand2};
import package::gi::consts::{GI_PI, GI_STAT_LT_DROP, GI_STAT_LT_EVENT, GI_STAT_LT_LINKED, GI_STAT_LT_TILES, GI_STAT_NEE, LT_BUCKET, LT_EVENT_CAP, LT_EVENT_R_MAX, LT_EVENT_R_MIN, LT_EVICT_FRAMES, LT_MAX_DIST, LT_NEE_MAX_D, LT_NODE_WORDS, LT_POWER_MAX, LT_POWER_SCALE, LT_SHADOW_EPS, LT_TILE_WORDS, LT_TOP};
import package::trace::{grid_is_far, make_grid};
import package::world::{world_trace_medium_scene};

const LT_NONE: u32 = 0xFFFFFFFFu;
const LT_LOCK: u32 = 1u;
const LT_K1_VALID: u32 = 0x80000000u;
const LT_K1_OBJ: u32 = 0x20000000u;

const LT_T_MASK: u32 = 0u;
const LT_T_STAMP: u32 = 2u;
const LT_T_UNIT: u32 = 3u;
const LT_T_POWER: u32 = 4u;
const LT_T_CELL: u32 = 5u;
const LT_T_REF: u32 = 8u;

const LT_N_TOTAL: u32 = 0u;
const LT_N_CHILD: u32 = 1u;

const LT_LEAF_LEVEL: u32 = 3u;

struct LtTile {
  face: u32,
  obj: u32,
  nc: u32,
  t1: u32,
  t2: u32,
}

fn lt_make_tile_key(voxel: vec3<i32>, face: u32, obj: u32) -> vec2<u32> {
  let a = face >> 1u;
  let u = vec3<u32>(voxel);
  let nc = u[a];
  let c1 = u[(a + 1u) % 3u];
  let c2 = u[(a + 2u) % 3u];
  if (obj == 0u) {
    let t1 = (c1 & 0x3FFFFu) >> 3u;
    let t2 = (c2 & 0x3FFFFu) >> 3u;
    return vec2<u32>((nc & 0x3FFFFu) | ((t1 & 0x3FFFu) << 18u),
                     (t1 >> 14u) | (t2 << 1u) | (face << 16u) | LT_K1_VALID);
  }
  let t1 = (c1 & 0x3FFFu) >> 3u;
  let t2 = (c2 & 0x3FFFu) >> 3u;
  return vec2<u32>((nc & 0x3FFFu) | (t1 << 14u) | ((t2 & 0x7Fu) << 25u),
                   (t2 >> 7u) | ((obj & 0x1FFFu) << 4u) | (face << 17u) | LT_K1_OBJ | LT_K1_VALID);
}

fn lt_tile_bit(voxel: vec3<i32>, face: u32) -> u32 {
  let a = face >> 1u;
  let u = vec3<u32>(voxel);
  return (u[(a + 1u) % 3u] & 7u) | ((u[(a + 2u) % 3u] & 7u) << 3u);
}

fn lt_decode(k: vec2<u32>) -> LtTile {
  if ((k.y & LT_K1_OBJ) == 0u) {
    return LtTile((k.y >> 16u) & 7u, 0u, k.x & 0x3FFFFu,
                  (k.x >> 18u) | ((k.y & 1u) << 14u), (k.y >> 1u) & 0x7FFFu);
  }
  return LtTile((k.y >> 17u) & 7u, (k.y >> 4u) & 0x1FFFu, k.x & 0x3FFFu,
                (k.x >> 14u) & 0x7FFu, (k.x >> 25u) | ((k.y & 0xFu) << 7u));
}

fn lt_tile_load(tile: u32) -> LtTile {
  return lt_decode(vec2<u32>(atomicLoad(&lt_tile_key[tile * 2u]),
                             atomicLoad(&lt_tile_key[tile * 2u + 1u])));
}

fn lt_voxel(t: LtTile, bit: u32) -> vec3<i32> {
  let a = t.face >> 1u;
  var c = vec3<u32>(0u);
  c[a] = t.nc;
  c[(a + 1u) % 3u] = (t.t1 << 3u) | (bit & 7u);
  c[(a + 2u) % 3u] = (t.t2 << 3u) | (bit >> 3u);
  if (t.obj == 0u) {
    let cam = vec3<i32>(floor(view_u.cam_pos_voxel.xyz));
    let d = (c - vec3<u32>(cam)) & vec3<u32>(0x3FFFFu);
    return cam + vec3<i32>(d ^ vec3<u32>(0x20000u)) - vec3<i32>(0x20000);
  }
  return vec3<i32>(c ^ vec3<u32>(0x2000u)) - vec3<i32>(0x2000);
}

fn lt_to_world(g: Grid, l: vec3<f32>) -> vec3<f32> {
  let s = l * g.scale;
  return g.pos + s.x * g.col0 + s.y * g.col1 + s.z * g.col2;
}

fn lt_normal_world(g: Grid, face: u32) -> vec3<f32> {
  let nl = face_normal_from_index(face);
  return normalize(nl.x * g.col0 + nl.y * g.col1 + nl.z * g.col2);
}

fn lt_face_local(voxel: vec3<i32>, face: u32, uv: vec2<f32>) -> vec3<f32> {
  let a = face >> 1u;
  var l = vec3<f32>(voxel);
  l[a] = l[a] + select(0.0, 1.0, (face & 1u) == 1u);
  l[(a + 1u) % 3u] = l[(a + 1u) % 3u] + uv.x;
  l[(a + 2u) % 3u] = l[(a + 2u) % 3u] + uv.y;
  return l;
}

fn lt_center_local(t: LtTile) -> vec3<f32> {
  return lt_face_local(lt_voxel(t, 0u), t.face, vec2<f32>(4.0));
}

fn lt_make_node_key(level: u32, cell: vec3<i32>) -> vec2<u32> {
  let c = vec3<u32>(cell) & vec3<u32>((1u << (20u - level)) - 1u);
  return vec2<u32>(c.x | ((c.y & 0x7FFFu) << 17u),
                   (c.y >> 15u) | (c.z << 2u) | (level << 19u) | LT_K1_VALID);
}

fn lt_hash(k: vec2<u32>) -> u32 {
  return gi_mix(k.x ^ gi_mix(k.y + 0x85EBCA77u));
}

fn lt_tile_base(k: vec2<u32>) -> u32 {
  let buckets = (arrayLength(&lt_tile) / LT_TILE_WORDS) / LT_BUCKET;
  return (lt_hash(k) & (buckets - 1u)) * LT_BUCKET;
}

fn lt_node_base(k: vec2<u32>) -> u32 {
  let buckets = (arrayLength(&lt_node) / LT_NODE_WORDS) / LT_BUCKET;
  return (lt_hash(k) & (buckets - 1u)) * LT_BUCKET;
}

fn lt_tile_find(k: vec2<u32>) -> u32 {
  let base = lt_tile_base(k);
  for (var j = 0u; j < LT_BUCKET; j = j + 1u) {
    let b = (base + j) * 2u;
    if (atomicLoad(&lt_tile_key[b + 1u]) == k.y && atomicLoad(&lt_tile_key[b]) == k.x) {
      return base + j;
    }
  }
  return LT_NONE;
}

fn lt_tile_claim(k: vec2<u32>) -> u32 {
  let base = lt_tile_base(k);
  var free = LT_BUCKET;
  for (var j = 0u; j < LT_BUCKET; j = j + 1u) {
    let b = (base + j) * 2u;
    let k1 = atomicLoad(&lt_tile_key[b + 1u]);
    if (k1 == k.y && atomicLoad(&lt_tile_key[b]) == k.x) { return base + j; }
    if (k1 <= LT_LOCK && free == LT_BUCKET) { free = j; }
  }
  for (var j = free; j < LT_BUCKET; j = j + 1u) {
    let b = (base + j) * 2u;
    var k1 = atomicLoad(&lt_tile_key[b + 1u]);
    for (var t = 0u; t < 4u && k1 == 0u; t = t + 1u) {
      let r = atomicCompareExchangeWeak(&lt_tile_key[b + 1u], 0u, LT_LOCK);
      if (r.exchanged) {
        atomicStore(&lt_tile_key[b], k.x);
        atomicStore(&lt_tile_key[b + 1u], k.y);
        return base + j;
      }
      k1 = r.old_value;
    }
    for (var t = 0u; t < 64u && k1 == LT_LOCK; t = t + 1u) {
      k1 = atomicLoad(&lt_tile_key[b + 1u]);
    }
    if (k1 == k.y && atomicLoad(&lt_tile_key[b]) == k.x) { return base + j; }
  }
  return LT_NONE;
}

fn lt_node_find(k: vec2<u32>) -> u32 {
  let base = lt_node_base(k);
  for (var j = 0u; j < LT_BUCKET; j = j + 1u) {
    let b = (base + j) * 2u;
    if (atomicLoad(&lt_node_key[b + 1u]) == k.y && atomicLoad(&lt_node_key[b]) == k.x) {
      return base + j;
    }
  }
  return LT_NONE;
}

fn lt_node_claim(k: vec2<u32>) -> u32 {
  let base = lt_node_base(k);
  var free = LT_BUCKET;
  for (var j = 0u; j < LT_BUCKET; j = j + 1u) {
    let b = (base + j) * 2u;
    let k1 = atomicLoad(&lt_node_key[b + 1u]);
    if (k1 == k.y && atomicLoad(&lt_node_key[b]) == k.x) { return base + j; }
    if (k1 <= LT_LOCK && free == LT_BUCKET) { free = j; }
  }
  for (var j = free; j < LT_BUCKET; j = j + 1u) {
    let b = (base + j) * 2u;
    var k1 = atomicLoad(&lt_node_key[b + 1u]);
    for (var t = 0u; t < 4u && k1 == 0u; t = t + 1u) {
      let r = atomicCompareExchangeWeak(&lt_node_key[b + 1u], 0u, LT_LOCK);
      if (r.exchanged) {
        atomicStore(&lt_node_key[b], k.x);
        atomicStore(&lt_node_key[b + 1u], k.y);
        return base + j;
      }
      k1 = r.old_value;
    }
    for (var t = 0u; t < 64u && k1 == LT_LOCK; t = t + 1u) {
      k1 = atomicLoad(&lt_node_key[b + 1u]);
    }
    if (k1 == k.y && atomicLoad(&lt_node_key[b]) == k.x) { return base + j; }
  }
  return LT_NONE;
}

fn lt_register(voxel: vec3<i32>, face: u32, obj: u32, le_luma: f32) -> u32 {
  let s = lt_tile_claim(lt_make_tile_key(voxel, face, obj));
  if (s == LT_NONE) {
    atomicAdd(&gi_stats[GI_STAT_LT_DROP], 1u);
    return LT_NONE;
  }
  let b = s * LT_TILE_WORDS;
  let bit = lt_tile_bit(voxel, face);
  let w = b + LT_T_MASK + (bit >> 5u);
  let m = 1u << (bit & 31u);
  if ((atomicLoad(&lt_tile[w]) & m) == 0u) { atomicOr(&lt_tile[w], m); }
  let frame = gi_u.seq.x;
  if (atomicLoad(&lt_tile[b + LT_T_STAMP]) != frame) { atomicStore(&lt_tile[b + LT_T_STAMP], frame); }
  if (atomicLoad(&lt_tile[b + LT_T_UNIT]) == 0u) {
    atomicStore(&lt_tile[b + LT_T_UNIT], bitcast<u32>(max(le_luma, 1e-4)));
  }
  return s;
}

fn lt_nth_bit(lo: u32, hi: u32, n: u32) -> u32 {
  var m = lo;
  var base = 0u;
  var k = n;
  let pl = countOneBits(lo);
  if (k >= pl) {
    k = k - pl;
    m = hi;
    base = 32u;
  }
  for (var i = 0u; i < k; i = i + 1u) { m = m & (m - 1u); }
  return base + firstTrailingBit(m);
}

fn lt_imp(power: u32, cell_min: vec3<f32>, size: f32, x: vec3<f32>, nx: vec3<f32>) -> f32 {
  if (power == 0u) { return 0.0; }
  let d = cell_min + vec3<f32>(size * 0.5) - x;
  if (dot(nx, d) + 0.5 * size * (abs(nx.x) + abs(nx.y) + abs(nx.z)) <= 0.0) { return 0.0; }
  return f32(power) / max(dot(d, d), size * size * 0.5);
}

fn lt_tile_imp(tile: u32, x: vec3<f32>) -> f32 {
  let p = atomicLoad(&lt_tile[tile * LT_TILE_WORDS + LT_T_POWER]);
  if (p == 0u) { return 0.0; }
  let t = lt_tile_load(tile);
  var c = lt_center_local(t);
  var n = face_normal_from_index(t.face);
  if (t.obj != 0u) {
    let g = make_grid(t.obj);
    c = lt_to_world(g, c);
    n = lt_normal_world(g, t.face);
  }
  let d = x - c;
  let h = dot(n, d);
  if (h <= 0.0) { return 0.0; }
  return f32(p) * (0.1 + 0.9 * h / max(length(d), 1e-3));
}

fn lt_oct(j: u32) -> vec3<i32> {
  return vec3<i32>(i32(j & 1u), i32((j >> 1u) & 1u), i32((j >> 2u) & 1u));
}

fn lt_choose(imp: ptr<function, array<f32, 8>>, sum: f32, u: ptr<function, f32>) -> u32 {
  let t = (*u) * sum;
  var acc = 0.0;
  var last = 0u;
  for (var j = 0u; j < 8u; j = j + 1u) {
    let w = (*imp)[j];
    if (w <= 0.0) { continue; }
    last = j;
    if (t < acc + w) {
      *u = clamp((t - acc) / w, 0.0, 0.999999);
      return j;
    }
    acc = acc + w;
  }
  *u = 0.5;
  return last;
}

struct LtPick {
  tile: u32,
  pdf: f32,
}

fn lt_walk(x: vec3<f32>, nx: vec3<f32>, u_in: f32) -> LtPick {
  let none = LtPick(LT_NONE, 0.0);
  var u = u_in;
  let st = f32(1u << LT_TOP);
  let base = vec3<i32>(floor(x / st - vec3<f32>(0.5)));
  var imp: array<f32, 8>;
  var slots: array<u32, 8>;
  var sum = 0.0;
  for (var j = 0u; j < 8u; j = j + 1u) {
    let c = base + lt_oct(j);
    let s = lt_node_find(lt_make_node_key(LT_TOP, c));
    slots[j] = s;
    var w = 0.0;
    if (s != LT_NONE) {
      w = lt_imp(atomicLoad(&lt_node[s * LT_NODE_WORDS + LT_N_TOTAL]), vec3<f32>(c) * st, st, x, nx);
    }
    imp[j] = w;
    sum = sum + w;
  }
  if (!(sum > 0.0)) { return none; }
  var pick = lt_choose(&imp, sum, &u);
  var pdf = imp[pick] / sum;
  var cell = base + lt_oct(pick);
  var slot = slots[pick];
  for (var k = LT_TOP; k > LT_LEAF_LEVEL; k = k - 1u) {
    let cs = f32(1u << (k - 1u));
    let nb = slot * LT_NODE_WORDS;
    sum = 0.0;
    for (var j = 0u; j < 8u; j = j + 1u) {
      let w = lt_imp(atomicLoad(&lt_node[nb + LT_N_CHILD + j]),
                     vec3<f32>(cell * 2 + lt_oct(j)) * cs, cs, x, nx);
      imp[j] = w;
      sum = sum + w;
    }
    if (!(sum > 0.0)) { return none; }
    pick = lt_choose(&imp, sum, &u);
    pdf = pdf * imp[pick] / sum;
    cell = cell * 2 + lt_oct(pick);
    slot = lt_node_find(lt_make_node_key(k - 1u, cell));
    if (slot == LT_NONE) { return none; }
  }
  let nb = slot * LT_NODE_WORDS;
  sum = 0.0;
  for (var j = 0u; j < 8u; j = j + 1u) {
    let r = atomicLoad(&lt_node[nb + LT_N_CHILD + j]);
    var w = 0.0;
    if (r != 0u) { w = lt_tile_imp(r - 1u, x); }
    imp[j] = w;
    sum = sum + w;
  }
  if (!(sum > 0.0)) { return none; }
  pick = lt_choose(&imp, sum, &u);
  return LtPick(atomicLoad(&lt_node[nb + LT_N_CHILD + pick]) - 1u, pdf * imp[pick] / sum);
}

fn lt_pdf(tile: u32, x: vec3<f32>, nx: vec3<f32>) -> f32 {
  let tb = tile * LT_TILE_WORDS;
  if (atomicLoad(&lt_tile[tb + LT_T_POWER]) == 0u) { return 0.0; }
  let leaf = vec3<i32>(bitcast<i32>(atomicLoad(&lt_tile[tb + LT_T_CELL])),
                       bitcast<i32>(atomicLoad(&lt_tile[tb + LT_T_CELL + 1u])),
                       bitcast<i32>(atomicLoad(&lt_tile[tb + LT_T_CELL + 2u])));
  let st = f32(1u << LT_TOP);
  let base = vec3<i32>(floor(x / st - vec3<f32>(0.5)));
  var cell = leaf >> vec3<u32>(LT_TOP - LT_LEAF_LEVEL);
  let rel = cell - base;
  if (any(rel < vec3<i32>(0)) || any(rel > vec3<i32>(1))) { return 0.0; }
  var sum = 0.0;
  var mine = 0.0;
  var slot = LT_NONE;
  for (var j = 0u; j < 8u; j = j + 1u) {
    let c = base + lt_oct(j);
    let s = lt_node_find(lt_make_node_key(LT_TOP, c));
    if (s == LT_NONE) { continue; }
    let w = lt_imp(atomicLoad(&lt_node[s * LT_NODE_WORDS + LT_N_TOTAL]), vec3<f32>(c) * st, st, x, nx);
    sum = sum + w;
    if (all(c == cell)) {
      mine = w;
      slot = s;
    }
  }
  if (!(mine > 0.0)) { return 0.0; }
  var pdf = mine / sum;
  for (var k = LT_TOP; k > LT_LEAF_LEVEL; k = k - 1u) {
    let child = leaf >> vec3<u32>(k - 1u - LT_LEAF_LEVEL);
    let cs = f32(1u << (k - 1u));
    let nb = slot * LT_NODE_WORDS;
    sum = 0.0;
    mine = 0.0;
    for (var j = 0u; j < 8u; j = j + 1u) {
      let cc = cell * 2 + lt_oct(j);
      let w = lt_imp(atomicLoad(&lt_node[nb + LT_N_CHILD + j]), vec3<f32>(cc) * cs, cs, x, nx);
      sum = sum + w;
      if (all(cc == child)) { mine = w; }
    }
    if (!(mine > 0.0)) { return 0.0; }
    pdf = pdf * mine / sum;
    cell = child;
    slot = lt_node_find(lt_make_node_key(k - 1u, cell));
    if (slot == LT_NONE) { return 0.0; }
  }
  let nb = slot * LT_NODE_WORDS;
  sum = 0.0;
  mine = 0.0;
  for (var j = 0u; j < 8u; j = j + 1u) {
    let r = atomicLoad(&lt_node[nb + LT_N_CHILD + j]);
    if (r == 0u) { continue; }
    let w = lt_tile_imp(r - 1u, x);
    sum = sum + w;
    if (r - 1u == tile) { mine = w; }
  }
  if (!(mine > 0.0)) { return 0.0; }
  return pdf * mine / sum;
}

fn lt_active() -> bool {
  return atomicLoad(&gi_stats[GI_STAT_LT_LINKED]) != 0u;
}

fn lt_nee_on() -> bool {
  return gi_u.flags.x > 0.5 && lt_active();
}

struct LtSample {
  y: vec3<f32>,
  le: vec3<f32>,
  nl: vec3<f32>,
  pa: f32,
  ok: bool,
}

fn lt_pick(x: vec3<f32>, nx: vec3<f32>, seed: u32) -> LtSample {
  var out = LtSample(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0), 0.0, false);
  if (!lt_active()) { return out; }
  let r0 = gi_rand2(seed);
  let pick = lt_walk(x, nx, r0.x);
  if (pick.tile == LT_NONE) { return out; }
  let tb = pick.tile * LT_TILE_WORDS;
  let mlo = atomicLoad(&lt_tile[tb + LT_T_MASK]);
  let mhi = atomicLoad(&lt_tile[tb + LT_T_MASK + 1u]);
  let pop = countOneBits(mlo) + countOneBits(mhi);
  if (pop == 0u) { return out; }
  let bit = lt_nth_bit(mlo, mhi, min(u32(r0.y * f32(pop)), pop - 1u));
  let t = lt_tile_load(pick.tile);
  let voxel = lt_voxel(t, bit);
  let g = make_grid(t.obj);
  let ny = lt_normal_world(g, t.face);
  let pal = sample_brickmap(g, voxel);
  if (pal == 0u || (palette_flags(g.palette_base, pal) & FLAGS_TRANSMISSIVE) != 0u) { return out; }
  let pc = lt_to_world(g, vec3<f32>(voxel) + vec3<f32>(0.5));
  let le = emissive_radiance(fetch_material(
    g.palette_base, pal, pc, ny, g.voxel_per_meter, MATERIAL_DEBUG_NONE),
                             gi_u.params.x);
  if (!(max(le.x, max(le.y, le.z)) > 0.0)) { return out; }
  let y = lt_to_world(g, lt_face_local(voxel, t.face, gi_rand2(gi_mix(seed + 0x68E31DA4u))));
  let d = y - x;
  let dist2 = max(dot(d, d), 1e-6);
  let dist = sqrt(dist2);
  if (dist > LT_NEE_MAX_D) { return out; }
  let dir = d / dist;
  let cx = dot(nx, dir);
  let cy = -dot(ny, dir);
  if (cx <= 0.0 || cy <= 0.0) { return out; }
  out.y = y;
  out.le = le;
  out.nl = ny;
  out.pa = pick.pdf / (f32(pop) * g.scale * g.scale);
  out.ok = true;
  return out;
}

fn lt_nee(x: vec3<f32>, nx: vec3<f32>, seed: u32) -> vec3<f32> {
  if (!lt_active()) { return vec3<f32>(0.0); }
  let r0 = gi_rand2(seed);
  let pick = lt_walk(x, nx, r0.x);
  if (pick.tile == LT_NONE) { return vec3<f32>(0.0); }
  let tb = pick.tile * LT_TILE_WORDS;
  let mlo = atomicLoad(&lt_tile[tb + LT_T_MASK]);
  let mhi = atomicLoad(&lt_tile[tb + LT_T_MASK + 1u]);
  let pop = countOneBits(mlo) + countOneBits(mhi);
  if (pop == 0u) { return vec3<f32>(0.0); }
  let bit = lt_nth_bit(mlo, mhi, min(u32(r0.y * f32(pop)), pop - 1u));
  let t = lt_tile_load(pick.tile);
  let voxel = lt_voxel(t, bit);
  let g = make_grid(t.obj);
  let ny = lt_normal_world(g, t.face);
  let pal = sample_brickmap(g, voxel);
  var le = vec3<f32>(0.0);
  if (pal != 0u && (palette_flags(g.palette_base, pal) & FLAGS_TRANSMISSIVE) == 0u) {
    let pc = lt_to_world(g, vec3<f32>(voxel) + vec3<f32>(0.5));
    le = emissive_radiance(fetch_material(
      g.palette_base, pal, pc, ny, g.voxel_per_meter, MATERIAL_DEBUG_NONE), gi_u.params.x);
  }
  if (!(max(le.x, max(le.y, le.z)) > 0.0)) {
    atomicAnd(&lt_tile[tb + LT_T_MASK + (bit >> 5u)], ~(1u << (bit & 31u)));
    return vec3<f32>(0.0);
  }
  let y = lt_to_world(g, lt_face_local(voxel, t.face, gi_rand2(gi_mix(seed + 0x68E31DA4u))));
  let d = y - x;
  let dist2 = max(dot(d, d), 1e-6);
  let dist = sqrt(dist2);
  if (dist > LT_NEE_MAX_D) { return vec3<f32>(0.0); }
  let dir = d / dist;
  let cx = dot(nx, dir);
  let cy = -dot(ny, dir);
  if (cx <= 0.0 || cy <= 0.0) { return vec3<f32>(0.0); }
  atomicAdd(&gi_stats[GI_STAT_NEE], 1u);
  var tint = vec3<f32>(1.0);
  let h = world_trace_medium_scene(x, dir, dist - LT_SHADOW_EPS * g.scale, false, false, &tint);
  if (h.hit) { return vec3<f32>(0.0); }
  let frame = gi_u.seq.x;
  if (atomicLoad(&lt_tile[tb + LT_T_STAMP]) != frame) { atomicStore(&lt_tile[tb + LT_T_STAMP], frame); }
  let pdf_w = pick.pdf * dist2 / (f32(pop) * g.scale * g.scale * cy);
  return le * tint * (cx / (cx + GI_PI * pdf_w));
}

fn lt_hit_weight(voxel: vec3<i32>, face: u32, obj: u32, le_luma: f32, x: vec3<f32>,
                 nx: vec3<f32>, dir: vec3<f32>, dist: f32, ny: vec3<f32>, scale: f32) -> f32 {
  let s = lt_register(voxel, face, obj, le_luma);
  if (s == LT_NONE) { return 1.0; }
  let cx = dot(nx, dir);
  let cy = -dot(ny, dir);
  if (cx <= 0.0 || cy <= 1e-4) { return 1.0; }
  let psel = lt_pdf(s, x, nx);
  if (!(psel > 0.0)) { return 1.0; }
  let tb = s * LT_TILE_WORDS;
  let pop = max(countOneBits(atomicLoad(&lt_tile[tb + LT_T_MASK]))
              + countOneBits(atomicLoad(&lt_tile[tb + LT_T_MASK + 1u])), 1u);
  let pdf_w = psel * dist * dist / (f32(pop) * scale * scale * cy);
  return cx / (cx + GI_PI * pdf_w);
}

fn lt_chain_add(leaf: vec3<i32>, delta: u32, create: bool) {
  for (var k = LT_LEAF_LEVEL + 1u; k <= LT_TOP; k = k + 1u) {
    let key = lt_make_node_key(k, leaf >> vec3<u32>(k - LT_LEAF_LEVEL));
    var s = LT_NONE;
    if (create) { s = lt_node_claim(key); } else { s = lt_node_find(key); }
    if (s == LT_NONE) { continue; }
    let c = (leaf >> vec3<u32>(k - 1u - LT_LEAF_LEVEL)) & vec3<i32>(1);
    let nb = s * LT_NODE_WORDS;
    atomicAdd(&lt_node[nb + LT_N_TOTAL], delta);
    atomicAdd(&lt_node[nb + LT_N_CHILD + u32(c.x | (c.y << 1u) | (c.z << 2u))], delta);
  }
}

fn lt_tile_free(s: u32) {
  let b = s * LT_TILE_WORDS;
  for (var i = 0u; i < LT_TILE_WORDS; i = i + 1u) { atomicStore(&lt_tile[b + i], 0u); }
  atomicStore(&lt_tile_key[s * 2u], 0u);
  atomicStore(&lt_tile_key[s * 2u + 1u], 0u);
}

fn lt_event_push(center: vec3<f32>, radius: f32) {
  let i = atomicAdd(&lt_event[0], 1u);
  atomicAdd(&gi_stats[GI_STAT_LT_EVENT], 1u);
  if (i >= LT_EVENT_CAP) { return; }
  let o = 1u + i * 4u;
  atomicStore(&lt_event[o], bitcast<u32>(i32(center.x)));
  atomicStore(&lt_event[o + 1u], bitcast<u32>(i32(center.y)));
  atomicStore(&lt_event[o + 2u], bitcast<u32>(i32(center.z)));
  atomicStore(&lt_event[o + 3u], bitcast<u32>(radius));
}

fn lt_event_radius(power: u32) -> f32 {
  return clamp(LT_EVENT_R_MIN + sqrt(f32(power)) * 2.0, LT_EVENT_R_MIN, LT_EVENT_R_MAX);
}

fn lt_tile_center_world(s: u32) -> vec3<f32> {
  let t = lt_tile_load(s);
  var c = lt_center_local(t);
  if (t.obj != 0u) {
    c = lt_to_world(make_grid(t.obj), c);
  }
  return c;
}

fn lt_sweep_tile(s: u32) {
  let k1 = atomicLoad(&lt_tile_key[s * 2u + 1u]);
  if (k1 == 0u) { return; }
  if (k1 == LT_LOCK) {
    atomicStore(&lt_tile_key[s * 2u + 1u], 0u);
    return;
  }
  let t = lt_decode(vec2<u32>(atomicLoad(&lt_tile_key[s * 2u]), k1));
  let b = s * LT_TILE_WORDS;
  let frame = gi_u.seq.x;
  var mlo = atomicLoad(&lt_tile[b + LT_T_MASK]);
  var mhi = atomicLoad(&lt_tile[b + LT_T_MASK + 1u]);
  var pop = countOneBits(mlo) + countOneBits(mhi);
  var evict = pop == 0u || frame - atomicLoad(&lt_tile[b + LT_T_STAMP]) > LT_EVICT_FRAMES;
  let g = make_grid(t.obj);
  if (!evict) {
    let age = frame - atomicLoad(&lt_tile[b + LT_T_STAMP]);
    let checks = 1u + min(63u, age * age / 64u);
    let seed = gi_mix(s * 0x9E3779B1u + frame * 0x85EBCA77u);
    for (var c = 0u; c < checks; c = c + 1u) {
      let bit = lt_nth_bit(mlo, mhi, (seed + c * 0x27D4EB2Fu) % pop);
      let voxel = lt_voxel(t, bit);
      let pal = sample_brickmap(g, voxel);
      var ok = pal != 0u && (palette_flags(g.palette_base, pal) & FLAGS_TRANSMISSIVE) == 0u
        && material_emissive(g.palette_base, pal) > 0.0;
      if (ok) {
        let front = sample_brickmap(g, voxel + vec3<i32>(face_normal_from_index(t.face)));
        ok = front == 0u || (palette_flags(g.palette_base, front) & FLAGS_TRANSMISSIVE) != 0u;
      }
      if (!ok) {
        atomicAnd(&lt_tile[b + LT_T_MASK + (bit >> 5u)], ~(1u << (bit & 31u)));
        if (bit < 32u) {
          mlo = mlo & ~(1u << bit);
        } else {
          mhi = mhi & ~(1u << (bit & 31u));
        }
        pop = pop - 1u;
        evict = pop == 0u;
        if (evict) { break; }
      } else {
        break;
      }
    }
  }
  let center = lt_to_world(g, lt_center_local(t));
  if (!evict) {
    evict = distance(center, view_u.cam_pos_voxel.xyz) > LT_MAX_DIST;
    if (t.obj != 0u && grid_is_far(t.obj)) {
      let g0 = make_grid(0u);
      let mid = (g0.w_mn + g0.w_mx) * 0.5;
      evict = evict || (g0.coverage_r > 0.0 && length((center - mid).xz) < g0.coverage_r);
    }
  }
  let leaf = vec3<i32>(floor(center)) >> vec3<u32>(LT_LEAF_LEVEL);
  var p_old = atomicLoad(&lt_tile[b + LT_T_POWER]);
  let p_death = p_old;
  let old_leaf = vec3<i32>(bitcast<i32>(atomicLoad(&lt_tile[b + LT_T_CELL])),
                           bitcast<i32>(atomicLoad(&lt_tile[b + LT_T_CELL + 1u])),
                           bitcast<i32>(atomicLoad(&lt_tile[b + LT_T_CELL + 2u])));
  if (p_old != 0u && (evict || any(leaf != old_leaf))) {
    let ls = lt_node_find(lt_make_node_key(LT_LEAF_LEVEL, old_leaf));
    if (ls != LT_NONE) {
      let nb = ls * LT_NODE_WORDS;
      let r = nb + LT_N_CHILD + (atomicLoad(&lt_tile[b + LT_T_REF]) & 7u);
      if (atomicLoad(&lt_node[r]) == s + 1u) { atomicStore(&lt_node[r], 0u); }
      atomicAdd(&lt_node[nb + LT_N_TOTAL], 0u - p_old);
    }
    lt_chain_add(old_leaf, 0u - p_old, false);
    atomicStore(&lt_tile[b + LT_T_POWER], 0u);
    p_old = 0u;
  }
  if (evict) {
    if (p_death > 0u && pop == 0u && frame - atomicLoad(&lt_tile[b + LT_T_STAMP]) > 16u) {
      lt_event_push(center, lt_event_radius(p_death));
    }
    lt_tile_free(s);
    return;
  }
  atomicAdd(&gi_stats[GI_STAT_LT_TILES], 1u);
  let unit = bitcast<f32>(atomicLoad(&lt_tile[b + LT_T_UNIT]));
  let p_now = clamp(u32(unit * f32(pop) * LT_POWER_SCALE + 0.5), 1u, LT_POWER_MAX);
  if (p_old == 0u) {
    let ls = lt_node_claim(lt_make_node_key(LT_LEAF_LEVEL, leaf));
    if (ls == LT_NONE) { return; }
    let nb = ls * LT_NODE_WORDS;
    var idx = 8u;
    for (var j = 0u; j < 8u && idx == 8u; j = j + 1u) {
      if (atomicLoad(&lt_node[nb + LT_N_CHILD + j]) == 0u
          && atomicCompareExchangeWeak(&lt_node[nb + LT_N_CHILD + j], 0u, s + 1u).exchanged) {
        idx = j;
      }
    }
    if (idx == 8u) { return; }
    atomicAdd(&lt_node[nb + LT_N_TOTAL], p_now);
    lt_chain_add(leaf, p_now, true);
    atomicStore(&lt_tile[b + LT_T_CELL], bitcast<u32>(leaf.x));
    atomicStore(&lt_tile[b + LT_T_CELL + 1u], bitcast<u32>(leaf.y));
    atomicStore(&lt_tile[b + LT_T_CELL + 2u], bitcast<u32>(leaf.z));
    atomicStore(&lt_tile[b + LT_T_REF], idx);
    atomicStore(&lt_tile[b + LT_T_POWER], p_now);
  } else if (p_now != p_old) {
    let ls = lt_node_find(lt_make_node_key(LT_LEAF_LEVEL, leaf));
    if (ls != LT_NONE) { atomicAdd(&lt_node[ls * LT_NODE_WORDS + LT_N_TOTAL], p_now - p_old); }
    lt_chain_add(leaf, p_now - p_old, false);
    atomicStore(&lt_tile[b + LT_T_POWER], p_now);
  }
  atomicAdd(&gi_stats[GI_STAT_LT_LINKED], 1u);
}

fn lt_sweep_node(s: u32) {
  let k1 = atomicLoad(&lt_node_key[s * 2u + 1u]);
  if (k1 == 0u) { return; }
  let b = s * LT_NODE_WORDS;
  let total = atomicLoad(&lt_node[b + LT_N_TOTAL]);
  if (k1 != LT_LOCK && total != 0u && total < 0x80000000u) { return; }
  for (var i = 0u; i < LT_NODE_WORDS; i = i + 1u) { atomicStore(&lt_node[b + i], 0u); }
  atomicStore(&lt_node_key[s * 2u], 0u);
  atomicStore(&lt_node_key[s * 2u + 1u], 0u);
}

fn lt_sweep_slot(s: u32) {
  if ((gi_u.seq.x & 255u) == 0u) {
    if (s < arrayLength(&lt_node) / LT_NODE_WORDS) { lt_sweep_node(s); }
    return;
  }
  if (s < arrayLength(&lt_tile) / LT_TILE_WORDS) { lt_sweep_tile(s); }
}
