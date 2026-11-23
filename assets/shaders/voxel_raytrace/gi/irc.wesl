import package::bindings::{gi_stats, gi_u, irc, irc_key, view_u};
import package::brickmap::{Grid};
import package::common::{face_normal_from_index};
import package::gi::common::{gi_mix};
import package::gi::consts::{GI_LUMA, GI_STAT_IRC_DROP, GI_STAT_IRC_EVICT, GI_STAT_IRC_LIVE, IRC_ACC_SCALE, IRC_BUCKET, IRC_CHAIN, IRC_FALLBACK_LEVELS, IRC_MAX_LEVEL, IRC_N_HALF, IRC_PARENT_CONF, IRC_SHRINK_C, IRC_SUN_ALPHA, IRC_V_MAX, IRC_VAR_ALPHA, IRC_W_MIN, IRC_WORDS, IRC_Z2};
import package::trace::{grid_is_far};

const IRC_W_VAL: u32 = 0u;
const IRC_W_META: u32 = 1u;
const IRC_W_MEAN: u32 = 2u;
const IRC_W_N: u32 = 5u;
const IRC_W_VAR: u32 = 6u;
const IRC_W_STAMP: u32 = 7u;
const IRC_W_SAMPLED: u32 = 8u;
const IRC_W_ACC: u32 = 10u;
const IRC_W_ACC_W: u32 = 13u;
const IRC_W_ACC_SUN: u32 = 14u;

const IRC_NONE: u32 = 0xFFFFFFFFu;
const IRC_LOCK: u32 = 1u;
const IRC_K1_VALID: u32 = 0x80000000u;
const IRC_K1_OBJ: u32 = 0x20000000u;

const IRC_FRAME_SAMPLES_MAX: f32 = 256.0;
const IRC_N_MAX: f32 = 4096.0;

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
        return base + j;
      }
      k1 = r.old_value;
    }
    for (var t = 0u; t < 64u && k1 == IRC_LOCK; t = t + 1u) {
      k1 = atomicLoad(&irc_key[b + 1u]);
    }
    if (k1 == k.y && atomicLoad(&irc_key[b]) == k.x) { return base + j; }
  }
  return IRC_NONE;
}

struct IrcAddr {
  v: vec3<i32>,
  obj: u32,
  level: u32,
  spf: f32,
}

fn irc_addr(g: Grid, idx: u32, voxel: vec3<i32>, p: vec3<f32>) -> IrcAddr {
  let dist = length(p - view_u.cam_pos_voxel.xyz);
  let px = max(dist * view_u.lod.x, 1e-6);
  var foot = px * gi_u.params.y;
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
  let level = clamp(u32(log2(max(foot, 1.0))), lmin, IRC_MAX_LEVEL);
  let cell_px = f32(1u << level) * gi_u.params.y / foot;
  let per = cell_px / max(gi_u.flags.y, 1.0);
  return IrcAddr(v, obj, level, per * per);
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

fn irc_display(a: IrcAddr, n: vec3<f32>) -> vec4<f32> {
  let w = n * n;
  var acc = vec4<f32>(0.0);
  var wsum = 0.0;
  for (var ax = 0u; ax < 3u; ax = ax + 1u) {
    if (w[ax] < IRC_W_MIN) { continue; }
    let face = ax * 2u + select(0u, 1u, n[ax] > 0.0);
    var r = irc_plane(a.v, a.level, face, a.obj);
    var l = a.level + 1u;
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
  sun_ok: bool,
  n: f32,
  fresh: bool,
}

fn irc_lookup(a: IrcAddr, face: u32) -> IrcHit {
  var out = IrcHit(vec3<f32>(0.0), 0.0, 0.0, false, 0.0, false);
  let frame = gi_u.seq.x;
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
    if (!out.sun_ok && ((meta_w >> 16u) & 0xFu) != 0u) {
      out.sun = f32((meta_w >> 8u) & 0xFFu) * (1.0 / 255.0);
      out.sun_ok = true;
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

fn irc_accumulate(a: IrcAddr, n: vec3<f32>, e: vec3<f32>, has_e: bool, sun: f32, rnd: vec2<f32>) {
  let w = n * n;
  var ev = min(e, vec3<f32>(IRC_V_MAX));
  ev = select(vec3<f32>(0.0), ev, ev >= vec3<f32>(0.0));
  var sunq = 0u;
  if (sun >= 0.0) { sunq = (1u << 20u) | u32(clamp(sun, 0.0, 1.0) * 255.0 + 0.5); }
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
      if (sunq != 0u) { atomicAdd(&irc[b + IRC_W_ACC_SUN], sunq); }
    }
  }
}

fn irc_accumulate_sun(a: IrcAddr, face: u32, sun: f32, rnd: vec2<f32>) {
  irc_accumulate(a, face_normal_from_index(face), vec3<f32>(0.0), false, sun, rnd);
}

fn irc_parent(cl: IrcCell) -> vec4<f32> {
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
  var acc = irc_tap(irc_cell_key(IrcCell(pc, level, cl.face, cl.obj))) * 0.5625;
  acc = acc + irc_tap(irc_cell_key(IrcCell(c1, level, cl.face, cl.obj))) * 0.1875;
  acc = acc + irc_tap(irc_cell_key(IrcCell(c2, level, cl.face, cl.obj))) * 0.1875;
  acc = acc + irc_tap(irc_cell_key(IrcCell(c3, level, cl.face, cl.obj))) * 0.0625;
  if (!(acc.a > 0.0)) { return vec4<f32>(0.0); }
  return vec4<f32>(acc.rgb / acc.a, acc.a);
}

fn irc_resolve_slot(s: u32) {
  let kb = s * 2u;
  let k1 = atomicLoad(&irc_key[kb + 1u]);
  if (k1 == 0u) { return; }
  let b = s * IRC_WORDS;
  let frame = gi_u.seq.x;
  let wq = atomicLoad(&irc[b + IRC_W_ACC_W]);
  let sunq = atomicLoad(&irc[b + IRC_W_ACC_SUN]);
  var stamp = atomicLoad(&irc[b + IRC_W_STAMP]);
  if (wq != 0u || sunq != 0u) { stamp = frame; }
  if (k1 == IRC_LOCK || frame - stamp > gi_u.seq.y) {
    for (var i = 0u; i < IRC_WORDS; i = i + 1u) { atomicStore(&irc[b + i], 0u); }
    atomicStore(&irc_key[kb], 0u);
    atomicStore(&irc_key[kb + 1u], 0u);
    atomicAdd(&gi_stats[GI_STAT_IRC_EVICT], 1u);
    return;
  }
  atomicAdd(&gi_stats[GI_STAT_IRC_LIVE], 1u);

  var meta_w = atomicLoad(&irc[b + IRC_W_META]);
  var mean = vec3<f32>(bitcast<f32>(atomicLoad(&irc[b + IRC_W_MEAN])),
                       bitcast<f32>(atomicLoad(&irc[b + IRC_W_MEAN + 1u])),
                       bitcast<f32>(atomicLoad(&irc[b + IRC_W_MEAN + 2u])));
  var n = min(bitcast<f32>(atomicLoad(&irc[b + IRC_W_N])) * gi_u.misc.z, gi_u.params.w);
  var varr = bitcast<f32>(atomicLoad(&irc[b + IRC_W_VAR]));
  var streak = (meta_w >> 20u) & 3u;
  var sun8 = (meta_w >> 8u) & 0xFFu;
  var sun_n = (meta_w >> 16u) & 0xFu;
  if (gi_u.flags.z > 0.5) { sun_n = 0u; }

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
      varr = mix(varr, s2, IRC_VAR_ALPHA);
    } else {
      varr = max(lf * lf, 1e-8);
      streak = 0u;
    }
    let n_new = min(n + wsum, IRC_N_MAX);
    mean = mix(mean, fm, clamp(wsum / n_new, 0.0, 1.0));
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
  if (sunq != 0u) {
    let v = f32(sunq & 0xFFFFFu) / (255.0 * f32(max(sunq >> 20u, 1u)));
    let a = max(1.0 / f32(sun_n + 1u), IRC_SUN_ALPHA);
    sun8 = u32(clamp(mix(f32(sun8) * (1.0 / 255.0), v, a), 0.0, 1.0) * 255.0 + 0.5);
    sun_n = min(sun_n + 1u, 15u);
    atomicStore(&irc[b + IRC_W_ACC_SUN], 0u);
  }
  atomicStore(&irc[b + IRC_W_N], bitcast<u32>(n));
  atomicStore(&irc[b + IRC_W_STAMP], stamp);

  var out_e = mean;
  var out_c = n / (n + IRC_N_HALF);
  let cl = irc_decode(vec2<u32>(atomicLoad(&irc_key[kb]), k1));
  if (cl.level < 15u) {
    let par = irc_parent(cl);
    if (par.a > 0.0) {
      if (n < 0.5) {
        out_e = par.rgb;
      } else {
        let l = max(dot(mean, GI_LUMA), 1e-6);
        let dv = mean - par.rgb;
        let d2 = dot(dv, dv) / max(dot(mean, mean), 1e-12);
        let se2 = varr / (n * l * l);
        out_e = mix(par.rgb, mean, d2 / (d2 + IRC_SHRINK_C * se2 + 1e-9));
      }
      out_c = max(out_c, par.a * IRC_PARENT_CONF);
    }
  }
  atomicStore(&irc[b + IRC_W_VAL], rgb9e5_pack(out_e));
  atomicStore(&irc[b + IRC_W_META],
              (meta_w & 0xFF000000u) | u32(clamp(out_c, 0.0, 1.0) * 255.0 + 0.5)
              | (sun8 << 8u) | (sun_n << 16u) | (streak << 20u));
}
