import package::bindings::{gi_stats, gi_u, irc, irc_key, light_u, lt_event, view_u};
import package::brickmap::{Grid};
import package::gi::common::{gi_div, gi_mix};
import package::gi::consts::{GI_LUMA, GI_STAT_IRC_DROP, GI_STAT_IRC_EVICT, GI_STAT_IRC_LIVE, GI_STAT_IRC_LOCKED, GI_STAT_IRC_ORPHAN, IRC_ACC_SCALE, IRC_BUCKET, IRC_CHAIN, IRC_EVENT_N, IRC_EVICT_GUARD, IRC_EVICT_PRESSURE, IRC_FALLBACK_LEVELS, IRC_HET_ALPHA, IRC_HET_K, IRC_HET_MIN, IRC_HET_SE_MAX, IRC_MAX_LEVEL, IRC_N_HALF, IRC_PARENT_CONF, IRC_RV_K, IRC_RV_MAX, IRC_RV_MIN, IRC_RV_PRIOR, IRC_SIG_HI, IRC_SIG_LO, IRC_V_MAX, IRC_VAR_ALPHA, IRC_W_MIN, IRC_WORDS, IRC_Z2, LT_EVENT_CAP, SUN_SLOTS, SUN_SPP};
import package::trace::{grid_is_far, make_grid};
import package::world::{voxel_center_world};

const IRC_W_VAL: u32 = 0u;
const IRC_W_META: u32 = 1u;
const IRC_W_MEAN: u32 = 2u;
const IRC_W_N: u32 = 5u;
const IRC_W_VAR: u32 = 6u;
const IRC_W_STAMP: u32 = 7u;
const IRC_W_SAMPLED: u32 = 8u;
const IRC_W_RV: u32 = 9u;
const IRC_W_ACC: u32 = 10u;
const IRC_W_ACC_W: u32 = 13u;
// 16 槽 × 2 bit：0 未采样，1/2/3 = 可见度 0/0.5/1。
const IRC_W_SUN: u32 = 14u;
const IRC_W_HET: u32 = 15u;
const IRC_W_SUN_TINT: u32 = 16u;

const IRC_META_SUN_SHIFT: u32 = 8u;
const IRC_META_SUN_MASK: u32 = 0xFFFu;

// 距离 LOD 切换前用父级预混最后一段，避免全局光变化瞬态在层级边界留下弧线。
const IRC_LOD_BLEND: f32 = 0.2;

const IRC_NONE: u32 = 0xFFFFFFFFu;
const IRC_LOCK: u32 = 1u;
const IRC_K1_VALID: u32 = 0x80000000u;
const IRC_K1_OBJ: u32 = 0x20000000u;

const IRC_FRAME_SAMPLES_MAX: f32 = 256.0;
const IRC_N_MAX: f32 = 32768.0;

fn rgb9e5_pack(c: vec3<f32>) -> u32 {
  let v = clamp(c, vec3<f32>(0.0), vec3<f32>(65408.0));
  let m = max(v.x, max(v.y, v.z));
  if (!(m > 0.0)) { return 0u; }
  var e = clamp(floor(log2(m)) + 16.0, 0.0, 31.0);
  var d = exp2(e - 24.0);
  if (floor(m / d + 0.5) >= 512.0) {
    e = e + 1.0;
    d = d * 2.0;
  }
  let q = min(vec3<u32>(floor(v / d + vec3<f32>(0.5))), vec3<u32>(511u));
  return q.x | (q.y << 9u) | (q.z << 18u) | (u32(e) << 27u);
}

fn rgb9e5_unpack(p: u32) -> vec3<f32> {
  return vec3<f32>(f32(p & 0x1FFu), f32((p >> 9u) & 0x1FFu), f32((p >> 18u) & 0x1FFu))
    * exp2(f32(p >> 27u) - 24.0);
}

struct IrcStat {
  rv_now: f32,
  rv_mean: f32,
  het: f32,
}

fn irc_rv_code(rv: f32) -> u32 {
  return 1u + u32((log2(clamp(rv, IRC_RV_MIN, IRC_RV_MAX)) + 4.0) * 100.0 + 0.5);
}

fn irc_rv_decode(q: u32) -> f32 {
  return select(0.0, exp2(f32(q) * 0.01 - 4.01), q != 0u);
}

fn irc_stat_pack(st: IrcStat) -> u32 {
  return irc_rv_code(st.rv_now) | (irc_rv_code(st.rv_mean) << 10u)
    | (u32(sqrt(clamp(st.het, 0.0, 1.0)) * 4095.0 + 0.5) << 20u);
}

fn irc_stat_unpack(w: u32) -> IrcStat {
  let t = f32(w >> 20u) * (1.0 / 4095.0);
  return IrcStat(irc_rv_decode(w & 0x3FFu), irc_rv_decode((w >> 10u) & 0x3FFu), t * t);
}

struct IrcCell {
  c: vec3<u32>,
  level: u32,
  face: u32,
  obj: u32,
}

fn irc_coord_bits(obj: u32) -> vec3<u32> {
  return select(vec3<u32>(14u, 13u, 14u), vec3<u32>(18u), obj == 0u);
}

fn irc_cell_of(v: vec3<i32>, level: u32, face: u32, obj: u32) -> IrcCell {
  let bits = irc_coord_bits(obj);
  let m = vec3<u32>(v) & ((vec3<u32>(1u) << bits) - vec3<u32>(1u));
  return IrcCell(m >> vec3<u32>(level), level, face, obj);
}

fn irc_coord_mask(obj: u32, level: u32) -> vec3<u32> {
  let bits = irc_coord_bits(obj);
  return (vec3<u32>(1u) << (max(bits, vec3<u32>(level)) - vec3<u32>(level))) - vec3<u32>(1u);
}

fn irc_cell_key(cl: IrcCell) -> vec2<u32> {
  let c = cl.c;
  let tail = (cl.face << 22u) | (cl.level << 25u) | IRC_K1_VALID;
  if (cl.obj == 0u) {
    return vec2<u32>(c.x | ((c.y & 0x3FFFu) << 18u), (c.y >> 14u) | (c.z << 4u) | tail);
  }
  return vec2<u32>(c.x | (c.z << 14u) | ((c.y & 0xFu) << 28u),
                   (c.y >> 4u) | ((cl.obj & 0x1FFFu) << 9u) | tail | IRC_K1_OBJ);
}

fn irc_decode(k: vec2<u32>) -> IrcCell {
  let face = (k.y >> 22u) & 7u;
  let level = (k.y >> 25u) & 15u;
  if ((k.y & IRC_K1_OBJ) == 0u) {
    return IrcCell(vec3<u32>(k.x & 0x3FFFFu, (k.x >> 18u) | ((k.y & 0xFu) << 14u),
                             (k.y >> 4u) & 0x3FFFFu), level, face, 0u);
  }
  return IrcCell(vec3<u32>(k.x & 0x3FFFu, (k.x >> 28u) | ((k.y & 0x1FFu) << 4u),
                           (k.x >> 14u) & 0x3FFFu), level, face, (k.y >> 9u) & 0x1FFFu);
}

fn irc_slot_count() -> u32 {
  return arrayLength(&irc) / IRC_WORDS;
}

fn irc_sun_mean(mask: u32) -> vec2<f32> {
  let lo = mask & 0x55555555u;
  let hi = (mask >> 1u) & 0x55555555u;
  let seen = countOneBits(lo | hi);
  let sum = 2u * countOneBits(hi) + countOneBits(lo) - seen;
  return vec2<f32>(f32(sum) / (2.0 * f32(max(seen, 1u))), f32(seen));
}

fn irc_bucket_base(k: vec2<u32>) -> u32 {
  let buckets = irc_slot_count() / IRC_BUCKET;
  return (gi_mix(k.x ^ gi_mix(k.y + 0x9E3779B9u)) & (buckets - 1u)) * IRC_BUCKET;
}

fn irc_find(k: vec2<u32>) -> u32 {
  let base = irc_bucket_base(k);
  for (var j = 0u; j < IRC_BUCKET; j = j + 1u) {
    let b = (base + j) * 2u;
    if (atomicLoad(&irc_key[b + 1u]) == k.y && atomicLoad(&irc_key[b]) == k.x) { return base + j; }
  }
  return IRC_NONE;
}

fn irc_claim(k: vec2<u32>) -> u32 {
  let base = irc_bucket_base(k);
  let frame = gi_u.seq.x;
  var free = IRC_BUCKET;
  for (var j = 0u; j < IRC_BUCKET; j = j + 1u) {
    let b = (base + j) * 2u;
    let k1 = atomicLoad(&irc_key[b + 1u]);
    if (k1 == k.y && atomicLoad(&irc_key[b]) == k.x) { return base + j; }
    if (k1 <= IRC_LOCK && free == IRC_BUCKET) { free = j; }
  }
  for (var j = free; j < IRC_BUCKET; j = j + 1u) {
    let b = (base + j) * 2u;
    var k1 = atomicLoad(&irc_key[b + 1u]);
    for (var t = 0u; t < 4u && k1 == 0u; t = t + 1u) {
      let r = atomicCompareExchangeWeak(&irc_key[b + 1u], 0u, IRC_LOCK);
      if (r.exchanged) {
        atomicStore(&irc_key[b], k.x);
        atomicStore(&irc_key[b + 1u], k.y);
        atomicStore(&irc[(base + j) * IRC_WORDS + IRC_W_STAMP], frame);
        return base + j;
      }
      k1 = r.old_value;
    }
    for (var t = 0u; t < 64u && k1 == IRC_LOCK; t = t + 1u) {
      k1 = atomicLoad(&irc_key[b + 1u]);
    }
    if (k1 == k.y && atomicLoad(&irc_key[b]) == k.x) { return base + j; }
  }
  for (var attempt = 0u; attempt < 2u; attempt = attempt + 1u) {
    var victim = IRC_NONE;
    var oldest = 0xFFFFFFFFu;
    for (var j = 0u; j < IRC_BUCKET; j = j + 1u) {
      let s = base + j;
      let k1 = atomicLoad(&irc_key[s * 2u + 1u]);
      if (k1 == 0u || k1 == IRC_LOCK) { continue; }
      let st = atomicLoad(&irc[s * IRC_WORDS + IRC_W_STAMP]);
      if (frame - st < IRC_EVICT_GUARD) { continue; }
      if (st < oldest) {
        oldest = st;
        victim = s;
      }
    }
    if (victim == IRC_NONE) { return IRC_NONE; }
    let kb = victim * 2u;
    let old_k1 = atomicLoad(&irc_key[kb + 1u]);
    let r = atomicCompareExchangeWeak(&irc_key[kb + 1u], old_k1, IRC_LOCK);
    if (!r.exchanged) { continue; }
    for (var i = 0u; i < IRC_WORDS; i = i + 1u) {
      atomicStore(&irc[victim * IRC_WORDS + i], 0u);
    }
    atomicStore(&irc_key[kb], k.x);
    atomicStore(&irc_key[kb + 1u], k.y);
    atomicStore(&irc[victim * IRC_WORDS + IRC_W_STAMP], frame);
    atomicAdd(&gi_stats[GI_STAT_IRC_EVICT], 1u);
    return victim;
  }
  return IRC_NONE;
}

struct IrcAddr {
  v: vec3<i32>,
  obj: u32,
  level: u32,
  spf: f32,
  parent: f32,
}

fn irc_addr(g: Grid, idx: u32, voxel: vec3<i32>, p: vec3<f32>) -> IrcAddr {
  let dist = length(p - view_u.cam_pos_voxel.xyz);
  let px = max(dist * view_u.lod.x, 1e-6);
  let cell_tgt = max(gi_u.params.y, f32(gi_div()));
  var foot = px * cell_tgt;
  var v = voxel;
  var obj = 0u;
  var lmin = 0u;
  if (idx != 0u) {
    if (grid_is_far(idx)) {
      v = vec3<i32>(floor(p));
      lmin = u32(log2(max(g.scale, 1.0)) + 0.5);
    } else {
      obj = idx;
      foot = foot / max(g.scale, 1e-6);
    }
  }
  let lf = log2(max(foot, 1e-6));
  let level = clamp(u32(ceil(max(lf, 0.0))), lmin, IRC_MAX_LEVEL);
  var parent = 0.0;
  if (level < IRC_MAX_LEVEL) {
    parent = 1.0 - smoothstep(0.0, IRC_LOD_BLEND, f32(level) - lf);
  }
  let cell_px = f32(1u << level) * gi_u.params.y / foot;
  let per = cell_px / max(gi_u.flags.y, 1.0);
  return IrcAddr(v, obj, level, per * per, parent);
}

// 直光太阳阴影固定 level0，逐体素，不跟随 GI 的距离 LOD 变粗。
fn sun_addr(g: Grid, idx: u32, voxel: vec3<i32>, p: vec3<f32>) -> IrcAddr {
  var v = voxel;
  var obj = 0u;
  if (idx != 0u) {
    if (grid_is_far(idx)) {
      v = vec3<i32>(floor(p));
    } else {
      obj = idx;
    }
  }
  return IrcAddr(v, obj, 0u, 0.0, 0.0);
}

struct SunVis {
  vis: f32,
  tint: vec3<f32>,
  ok: bool,
}

fn irc_sun_tint(raw: u32) -> vec3<f32> {
  return select(vec3<f32>(1.0), rgb9e5_unpack(raw), raw != 0u);
}

fn irc_sun_code() -> u32 {
  let l = light_u.lights[0].kind_pos_dir.yzw;
  let r = light_u.lights[0].shape.y;
  let h = gi_mix(bitcast<u32>(l.x)
    ^ gi_mix(bitcast<u32>(l.y)
    ^ gi_mix(bitcast<u32>(l.z)
    ^ gi_mix(bitcast<u32>(r) ^ gi_u.seq.z))));
  return h & IRC_META_SUN_MASK;
}

// 同一太阳 code 下 16 槽全满后可见度已确定；移动相机只改变新露出的格子，
// 已满的格子重复采样只会得到相同结果。返回 0=需采样，1=本帧首次命中，2=本帧已命中。
// 命中时刷新 stamp，避免可见条目按年龄回收。
fn irc_sun_cached(a: IrcAddr, face: u32) -> u32 {
  let s = irc_find(irc_cell_key(irc_cell_of(a.v, a.level, face, a.obj)));
  if (s == IRC_NONE) { return 0u; }
  let b = s * IRC_WORDS;
  let meta_w = atomicLoad(&irc[b + IRC_W_META]);
  if (((meta_w >> IRC_META_SUN_SHIFT) & IRC_META_SUN_MASK) != irc_sun_code()) {
    return 0u;
  }
  if (atomicLoad(&irc[b + IRC_W_SUN]) != 0xFFFFFFFFu) { return 0u; }
  if (atomicLoad(&irc[b + IRC_W_STAMP]) != gi_u.seq.x) {
    atomicStore(&irc[b + IRC_W_STAMP], gi_u.seq.x);
    return 1u;
  }
  return 2u;
}

// code 变化时清空旧方向槽位；之后只写当前 code 的样本。
// 读取端 code 不匹配直接判无效，回退单条当前太阳方向的硬阴影，避免残影。
fn irc_sun_begin(a: IrcAddr, face: u32) -> bool {
  let s = irc_claim(irc_cell_key(irc_cell_of(a.v, a.level, face, a.obj)));
  if (s == IRC_NONE) { return false; }
  let b = s * IRC_WORDS;
  let code = irc_sun_code();
  let meta_w = atomicLoad(&irc[b + IRC_W_META]);
  if (((meta_w >> IRC_META_SUN_SHIFT) & IRC_META_SUN_MASK) == code) { return true; }
  atomicStore(&irc[b + IRC_W_SUN], 0u);
  atomicStore(&irc[b + IRC_W_SUN_TINT], 0u);
  atomicStore(&irc[b + IRC_W_META],
              (meta_w & ~(IRC_META_SUN_MASK << IRC_META_SUN_SHIFT))
              | (code << IRC_META_SUN_SHIFT));
  atomicStore(&irc[b + IRC_W_STAMP], gi_u.seq.x);
  return false;
}

fn irc_sun_lookup(a: IrcAddr, face: u32) -> SunVis {
  let code = irc_sun_code();
  for (var i = 0u; i < IRC_FALLBACK_LEVELS; i = i + 1u) {
    let s = irc_find(irc_cell_key(irc_cell_of(a.v, a.level + i, face, a.obj)));
    if (s == IRC_NONE) { continue; }
    let b = s * IRC_WORDS;
    let meta_w = atomicLoad(&irc[b + IRC_W_META]);
    if (((meta_w >> IRC_META_SUN_SHIFT) & IRC_META_SUN_MASK) != code) { continue; }
    let m = irc_sun_mean(atomicLoad(&irc[b + IRC_W_SUN]));
    if (m.y < 0.5) { continue; }
    return SunVis(m.x, irc_sun_tint(atomicLoad(&irc[b + IRC_W_SUN_TINT])), true);
  }
  return SunVis(0.0, vec3<f32>(1.0), false);
}

fn irc_sun_next(a: IrcAddr, face: u32) -> u32 {
  let s = irc_find(irc_cell_key(irc_cell_of(a.v, a.level, face, a.obj)));
  if (s == IRC_NONE) { return 0u; }
  let mask = atomicLoad(&irc[s * IRC_WORDS + IRC_W_SUN]);
  for (var i = 0u; i < SUN_SLOTS; i = i + 1u) {
    if (((mask >> (i * 2u)) & 3u) == 0u) { return i; }
  }
  return (gi_u.seq.x * SUN_SPP) & (SUN_SLOTS - 1u);
}

fn irc_sun_sample(a: IrcAddr, face: u32, slot: u32, vis: f32, tint: vec3<f32>) {
  let s = irc_claim(irc_cell_key(irc_cell_of(a.v, a.level, face, a.obj)));
  if (s == IRC_NONE) { return; }
  let b = s * IRC_WORDS;
  let sh = slot * 2u;
  let state = 1u + u32(clamp(vis, 0.0, 1.0) * 2.0 + 0.5);
  atomicAnd(&irc[b + IRC_W_SUN], ~(3u << sh));
  atomicOr(&irc[b + IRC_W_SUN], state << sh);
  if (vis > 0.0 && (slot == 0u || atomicLoad(&irc[b + IRC_W_SUN_TINT]) == 0u)) {
    atomicStore(&irc[b + IRC_W_SUN_TINT], rgb9e5_pack(max(tint, vec3<f32>(0.0005))));
  }
  atomicStore(&irc[b + IRC_W_STAMP], gi_u.seq.x);
}

fn irc_tap(k: vec2<u32>) -> vec4<f32> {
  let s = irc_find(k);
  if (s == IRC_NONE) { return vec4<f32>(0.0); }
  let b = s * IRC_WORDS;
  let conf = f32(atomicLoad(&irc[b + IRC_W_META]) & 0xFFu) * (1.0 / 255.0);
  if (conf <= 0.0) { return vec4<f32>(0.0); }
  return vec4<f32>(rgb9e5_unpack(atomicLoad(&irc[b + IRC_W_VAL])) * conf, conf);
}

fn irc_plane(v: vec3<i32>, level: u32, face: u32, obj: u32) -> vec4<f32> {
  let own = irc_tap(irc_cell_key(irc_cell_of(v, level, face, obj)));
  if (level == 0u) { return own; }
  let axis = face >> 1u;
  let a1 = (axis + 1u) % 3u;
  let a2 = (axis + 2u) % 3u;
  let s = i32(1u << level);
  let inv = 1.0 / f32(s);
  let f1 = (f32(v[a1] & (s - 1)) + 0.5) * inv - 0.5;
  let f2 = (f32(v[a2] & (s - 1)) + 0.5) * inv - 0.5;
  let w1 = abs(f1);
  let w2 = abs(f2);
  var d1 = vec3<i32>(0);
  var d2 = vec3<i32>(0);
  d1[a1] = select(-s, s, f1 > 0.0);
  d2[a2] = select(-s, s, f2 > 0.0);
  var acc = own * ((1.0 - w1) * (1.0 - w2));
  if (w1 > 0.02) {
    acc = acc + irc_tap(irc_cell_key(irc_cell_of(v + d1, level, face, obj))) * (w1 * (1.0 - w2));
  }
  if (w2 > 0.02) {
    acc = acc + irc_tap(irc_cell_key(irc_cell_of(v + d2, level, face, obj))) * ((1.0 - w1) * w2);
  }
  if (w1 > 0.02 && w2 > 0.02) {
    acc = acc + irc_tap(irc_cell_key(irc_cell_of(v + d1 + d2, level, face, obj))) * (w1 * w2);
  }
  return acc;
}

fn irc_display(a: IrcAddr, n_in: vec3<f32>) -> vec4<f32> {
  let n = select(n_in, vec3<f32>(0.0, 1.0, 0.0), dot(n_in, n_in) < 1e-6);
  let w = n * n;
  var acc = vec4<f32>(0.0);
  var wsum = 0.0;
  for (var ax = 0u; ax < 3u; ax = ax + 1u) {
    if (w[ax] < IRC_W_MIN) { continue; }
    let face = ax * 2u + select(0u, 1u, n[ax] > 0.0);
    var r = irc_plane(a.v, a.level, face, a.obj);
    var l = a.level + 1u;
    if (a.parent > 0.0 && r.a > 0.0) {
      r = r * (1.0 - a.parent) + irc_plane(a.v, l, face, a.obj) * a.parent;
      l = l + 1u;
    }
    for (var i = 0u; i < IRC_FALLBACK_LEVELS && r.a < 0.5; i = i + 1u) {
      r = r + irc_plane(a.v, l, face, a.obj) * (1.0 - r.a);
      l = l + 1u;
    }
    acc = acc + r * w[ax];
    wsum = wsum + w[ax];
  }
  if (!(acc.a > 0.0)) { return vec4<f32>(0.0); }
  return vec4<f32>(acc.rgb / acc.a, acc.a / wsum);
}

struct IrcHit {
  e: vec3<f32>,
  conf: f32,
  sun: f32,
  tint: vec3<f32>,
  sun_ok: bool,
  n: f32,
  fresh: bool,
}

fn irc_lookup(a: IrcAddr, face: u32) -> IrcHit {
  var out = IrcHit(vec3<f32>(0.0), 0.0, 0.0, vec3<f32>(1.0), false, 0.0, false);
  let frame = gi_u.seq.x;
  let code = irc_sun_code();
  for (var i = 0u; i < IRC_FALLBACK_LEVELS; i = i + 1u) {
    let s = irc_find(irc_cell_key(irc_cell_of(a.v, a.level + i, face, a.obj)));
    if (s == IRC_NONE) { continue; }
    let b = s * IRC_WORDS;
    if (atomicLoad(&irc[b + IRC_W_STAMP]) != frame) { atomicStore(&irc[b + IRC_W_STAMP], frame); }
    let meta_w = atomicLoad(&irc[b + IRC_W_META]);
    if (i == 0u) {
      out.n = bitcast<f32>(atomicLoad(&irc[b + IRC_W_N]));
      out.fresh = out.n >= 1.0 && (frame - atomicLoad(&irc[b + IRC_W_SAMPLED])) <= 1u;
    }
    if (!out.sun_ok && ((meta_w >> IRC_META_SUN_SHIFT) & IRC_META_SUN_MASK) == code) {
      let sm = irc_sun_mean(atomicLoad(&irc[b + IRC_W_SUN]));
      if (sm.y > 0.5) {
        out.sun = sm.x;
        out.tint = irc_sun_tint(atomicLoad(&irc[b + IRC_W_SUN_TINT]));
        out.sun_ok = true;
      }
    }
    let conf = f32(meta_w & 0xFFu) * (1.0 / 255.0);
    if (out.conf <= 0.0 && conf > 0.0) {
      out.e = rgb9e5_unpack(atomicLoad(&irc[b + IRC_W_VAL]));
      out.conf = conf;
    }
    if (out.conf > 0.0 && out.sun_ok) { break; }
  }
  return out;
}

fn irc_accumulate(a: IrcAddr, n_in: vec3<f32>, e: vec3<f32>, has_e: bool, rnd: vec2<f32>) {
  let n = select(n_in, vec3<f32>(0.0, 1.0, 0.0), dot(n_in, n_in) < 1e-6);
  let w = n * n;
  var ev = min(e, vec3<f32>(IRC_V_MAX));
  ev = select(vec3<f32>(0.0), ev, ev >= vec3<f32>(0.0));
  for (var ax = 0u; ax < 3u; ax = ax + 1u) {
    if (w[ax] < IRC_W_MIN) { continue; }
    let face = ax * 2u + select(0u, 1u, n[ax] > 0.0);
    var wq = 0u;
    var q = vec3<u32>(0u);
    if (has_e) {
      wq = max(u32(w[ax] * 256.0 + 0.5), 1u);
      q = vec3<u32>(ev * (f32(wq) * (IRC_ACC_SCALE / 256.0)) + vec3<f32>(rnd.x));
    }
    var spf = a.spf;
    for (var i = 0u; i < IRC_CHAIN; i = i + 1u) {
      let keep = spf <= IRC_FRAME_SAMPLES_MAX || rnd.y * spf < IRC_FRAME_SAMPLES_MAX;
      spf = spf * 4.0;
      if (!keep) { continue; }
      let s = irc_claim(irc_cell_key(irc_cell_of(a.v, a.level + i, face, a.obj)));
      if (s == IRC_NONE) {
        atomicAdd(&gi_stats[GI_STAT_IRC_DROP], 1u);
        continue;
      }
      let b = s * IRC_WORDS;
      if (wq != 0u) {
        atomicAdd(&irc[b + IRC_W_ACC], q.x);
        atomicAdd(&irc[b + IRC_W_ACC + 1u], q.y);
        atomicAdd(&irc[b + IRC_W_ACC + 2u], q.z);
        atomicAdd(&irc[b + IRC_W_ACC_W], wq);
      }
    }
  }
}

struct IrcPar {
  e: vec3<f32>,
  conf: f32,
  rv: f32,
  het: f32,
  slot: u32,
}

fn irc_parent(cl: IrcCell) -> IrcPar {
  let axis = cl.face >> 1u;
  let a1 = (axis + 1u) % 3u;
  let a2 = (axis + 2u) % 3u;
  let level = cl.level + 1u;
  let pm = irc_coord_mask(cl.obj, level);
  let pc = (cl.c >> vec3<u32>(1u)) & pm;
  var c1 = pc;
  var c2 = pc;
  c1[a1] = (pc[a1] + select(pm[a1], 1u, (cl.c[a1] & 1u) == 1u)) & pm[a1];
  c2[a2] = (pc[a2] + select(pm[a2], 1u, (cl.c[a2] & 1u) == 1u)) & pm[a2];
  var c3 = c1;
  c3[a2] = c2[a2];
  var acc = vec4<f32>(0.0);
  var stat = IrcStat(0.0, 0.0, 0.0);
  let s0 = irc_find(irc_cell_key(IrcCell(pc, level, cl.face, cl.obj)));
  if (s0 != IRC_NONE) {
    let pb = s0 * IRC_WORDS;
    let conf = f32(atomicLoad(&irc[pb + IRC_W_META]) & 0xFFu) * (1.0 / 255.0);
    acc = vec4<f32>(rgb9e5_unpack(atomicLoad(&irc[pb + IRC_W_VAL])) * conf, conf) * 0.5625;
    stat = irc_stat_unpack(atomicLoad(&irc[pb + IRC_W_RV]));
  }
  acc = acc + irc_tap(irc_cell_key(IrcCell(c1, level, cl.face, cl.obj))) * 0.1875;
  acc = acc + irc_tap(irc_cell_key(IrcCell(c2, level, cl.face, cl.obj))) * 0.1875;
  acc = acc + irc_tap(irc_cell_key(IrcCell(c3, level, cl.face, cl.obj))) * 0.0625;
  if (!(acc.a > 0.0)) { return IrcPar(vec3<f32>(0.0), 0.0, stat.rv_now, stat.het, s0); }
  return IrcPar(acc.rgb / acc.a, acc.a, stat.rv_now, stat.het, s0);
}

fn irc_resolve_slot(s: u32) {
  let kb = s * 2u;
  let b = s * IRC_WORDS;
  let k1 = atomicLoad(&irc_key[kb + 1u]);
  if (k1 == 0u) {
    if (atomicLoad(&irc[b + IRC_W_N]) != 0u || atomicLoad(&irc[b + IRC_W_MEAN]) != 0u) {
      atomicAdd(&gi_stats[GI_STAT_IRC_ORPHAN], 1u);
    }
    return;
  }
  if (k1 == IRC_LOCK) { atomicAdd(&gi_stats[GI_STAT_IRC_LOCKED], 1u); }
  let frame = gi_u.seq.x;
  let wq = atomicLoad(&irc[b + IRC_W_ACC_W]);
  var stamp = atomicLoad(&irc[b + IRC_W_STAMP]);
  if (wq != 0u) { stamp = frame; }
  let age = frame - stamp;
  var evict = k1 == IRC_LOCK || age > gi_u.seq.y;
  if (!evict && age > IRC_EVICT_PRESSURE) {
    let base = s - (s % IRC_BUCKET);
    var free = 0u;
    for (var j = 0u; j < IRC_BUCKET && free < 2u; j = j + 1u) {
      if (atomicLoad(&irc_key[(base + j) * 2u + 1u]) == 0u) { free = free + 1u; }
    }
    evict = free < 2u;
  }
  if (evict) {
    for (var i = 0u; i < IRC_WORDS; i = i + 1u) { atomicStore(&irc[b + i], 0u); }
    atomicStore(&irc_key[kb], 0u);
    atomicStore(&irc_key[kb + 1u], 0u);
    atomicAdd(&gi_stats[GI_STAT_IRC_EVICT], 1u);
    return;
  }
  atomicAdd(&gi_stats[GI_STAT_IRC_LIVE], 1u);

  // 灯事件（本帧 sweep/register 广播的灯生死）：条目世界中心落在事件半径内
  // ⇒ 一次性降信任（n 钳到 IRC_EVENT_N），新样本立刻占主导。灯死亡的反弹
  // 重建靠 streak 级联要几十秒（间接面靠邻域逐层传播），事件跳过级联。
  let ev_count = min(atomicLoad(&lt_event[0]), LT_EVENT_CAP);
  var n = min(bitcast<f32>(atomicLoad(&irc[b + IRC_W_N])) * gi_u.misc.z, gi_u.params.w);
  if (ev_count > 0u) {
    let cl_ev = irc_decode(vec2<u32>(atomicLoad(&irc_key[kb]), k1));
    let cell = i32(1u << cl_ev.level);
    let vc = vec3<i32>(cl_ev.c) * cell + vec3<i32>(cell / 2);
    let pw = voxel_center_world(make_grid(cl_ev.obj), vc);
    for (var e = 0u; e < ev_count; e = e + 1u) {
      let o = 1u + e * 4u;
      let ec = vec3<f32>(f32(bitcast<i32>(atomicLoad(&lt_event[o]))),
                    f32(bitcast<i32>(atomicLoad(&lt_event[o + 1u]))),
                    f32(bitcast<i32>(atomicLoad(&lt_event[o + 2u]))));
      let er = bitcast<f32>(atomicLoad(&lt_event[o + 3u]));
      if (distance(pw, ec) < er) {
        n = min(n, IRC_EVENT_N);
        break;
      }
    }
  }

  var meta_w = atomicLoad(&irc[b + IRC_W_META]);
  var mean = vec3<f32>(bitcast<f32>(atomicLoad(&irc[b + IRC_W_MEAN])),
                       bitcast<f32>(atomicLoad(&irc[b + IRC_W_MEAN + 1u])),
                       bitcast<f32>(atomicLoad(&irc[b + IRC_W_MEAN + 2u])));
  var varr = bitcast<f32>(atomicLoad(&irc[b + IRC_W_VAR]));
  var streak = (meta_w >> 20u) & 3u;
  var share = 0.0;

  if (wq != 0u) {
    let wsum = f32(wq) * (1.0 / 256.0);
    let fm = vec3<f32>(f32(atomicLoad(&irc[b + IRC_W_ACC])),
                       f32(atomicLoad(&irc[b + IRC_W_ACC + 1u])),
                       f32(atomicLoad(&irc[b + IRC_W_ACC + 2u]))) / (wsum * IRC_ACC_SCALE);
    let lf = dot(fm, GI_LUMA);
    if (n >= 1.0) {
      let d = lf - dot(mean, GI_LUMA);
      let s2 = d * d / (1.0 / wsum + 1.0 / n);
      let lim = IRC_Z2 * max(varr, 1e-10);
      if (s2 > lim) {
        streak = min(streak + 1u, 3u);
        if (streak >= 2u) { n = max(n * max(lim / s2, 0.125), min(n, 2.0)); }
      } else {
        streak = 0u;
      }
      varr = mix(varr, s2, clamp(wsum / (n + wsum), IRC_VAR_ALPHA, 0.5));
    } else {
      varr = max(lf * lf, 1e-8);
      streak = 0u;
    }
    let n_new = min(n + wsum, IRC_N_MAX);
    share = clamp(wsum / n_new, 0.0, 1.0);
    mean = mix(mean, fm, share);
    n = n_new;
    atomicStore(&irc[b + IRC_W_SAMPLED], frame);
    atomicStore(&irc[b + IRC_W_ACC], 0u);
    atomicStore(&irc[b + IRC_W_ACC + 1u], 0u);
    atomicStore(&irc[b + IRC_W_ACC + 2u], 0u);
    atomicStore(&irc[b + IRC_W_ACC_W], 0u);
    atomicStore(&irc[b + IRC_W_MEAN], bitcast<u32>(mean.x));
    atomicStore(&irc[b + IRC_W_MEAN + 1u], bitcast<u32>(mean.y));
    atomicStore(&irc[b + IRC_W_MEAN + 2u], bitcast<u32>(mean.z));
    atomicStore(&irc[b + IRC_W_VAR], bitcast<u32>(varr));
  }
  atomicStore(&irc[b + IRC_W_N], bitcast<u32>(n));
  atomicStore(&irc[b + IRC_W_STAMP], stamp);

  let stat = irc_stat_unpack(atomicLoad(&irc[b + IRC_W_RV]));
  var het = stat.het;
  let hq = atomicExchange(&irc[b + IRC_W_HET], 0u);
  let hn = hq >> 28u;
  if (hn >= IRC_HET_MIN) {
    let md = f32(hq & 0x3FFFu) / (1023.0 * f32(hn));
    let ms = f32((hq >> 14u) & 0x3FFFu) / (4092.0 * f32(hn));
    het = mix(het, max(md - IRC_HET_K * ms, 0.0), IRC_HET_ALPHA);
  } else {
    het = het * (1.0 - IRC_HET_ALPHA);
  }

  var out_e = mean;
  var out_c = n / (n + IRC_N_HALF);
  var rv = IRC_RV_PRIOR;
  var rv_mean = IRC_RV_PRIOR;
  let cl = irc_decode(vec2<u32>(atomicLoad(&irc_key[kb]), k1));
  if (cl.level < 15u) {
    let par = irc_parent(cl);
    let prior = select(IRC_RV_PRIOR, par.rv, par.rv > 0.0);
    let l = dot(mean, GI_LUMA);
    var rv_own = prior;
    if (n >= 1.0 && l > 1e-6) { rv_own = varr / (l * l); }
    rv = clamp((n * rv_own + IRC_RV_K * prior) / (n + IRC_RV_K), IRC_RV_MIN, IRC_RV_MAX);
    rv_mean = select(rv, mix(stat.rv_mean, rv, share), stat.rv_mean > 0.0);
    if (par.conf > 0.0) {
      if (n < 0.5) {
        out_e = par.e;
      } else {
        let dv = mean - par.e;
        let d2 = dot(dv, dv) / max(max(dot(mean, mean), dot(par.e, par.e)), 1e-12);
        let se2 = rv_mean / n;
        var w = smoothstep(IRC_SIG_LO, IRC_SIG_HI, d2 / se2);
        if (par.slot != IRC_NONE && se2 < IRC_HET_SE_MAX) {
          atomicAdd(&irc[par.slot * IRC_WORDS + IRC_W_HET],
                    u32(min(d2, 1.0) * 1023.0 + 0.5) | (u32(se2 * 4092.0 + 0.5) << 14u)
                    | (1u << 28u));
          w = max(w, par.het / (par.het + se2));
        }
        out_e = mix(par.e, mean, min(w, out_c));
      }
      out_c = max(out_c, par.conf * IRC_PARENT_CONF);
    }
  }
  atomicStore(&irc[b + IRC_W_RV], irc_stat_pack(IrcStat(rv, rv_mean, het)));
  atomicStore(&irc[b + IRC_W_VAL], rgb9e5_pack(out_e));
  atomicStore(&irc[b + IRC_W_META],
              (meta_w & 0xFF000000u)
              | (meta_w & (IRC_META_SUN_MASK << IRC_META_SUN_SHIFT))
              | u32(clamp(out_c, 0.0, 1.0) * 255.0 + 0.5)
              | (streak << 20u));
}
