import package::bindings::{gi_u, res_cur, res_prev, res_tmp};
import package::gi::consts::{GI_LUMA, GI_PI, RES_WORDS};
import package::gi::irc::{rgb9e5_pack, rgb9e5_unpack};
import package::gi::light::{LtSample, lt_pick};

const RES_W_Y: u32 = 0u;
const RES_W_LE: u32 = 3u;
const RES_W_NL: u32 = 4u;
const RES_W_PA: u32 = 6u;
const RES_W_SUM: u32 = 7u;
const RES_W_M: u32 = 10u;
const RES_W_VIS: u32 = 11u;
const RES_W_REV: u32 = 12u;
const RES_W_K0: u32 = 13u;
const RES_W_K1: u32 = 14u;
const RES_W_X: u32 = 15u;
const RES_W_N: u32 = 18u;
const RES_W_FLAGS: u32 = 21u;

const RES_VALID: u32 = 1u;
const RES_M_CAP: f32 = 1000000.0;

fn res_same_surface(p: Res, r: Res) -> bool {
  if (!p.valid) { return false; }
  return p.k0 == r.k0 && p.k1 == r.k1;
}

struct Res {
  y: vec3<f32>,
  le: vec3<f32>,
  nl: vec3<f32>,
  pa: f32,
  sum: vec3<f32>,
  m: f32,
  vis: f32,
  rev: u32,
  k0: u32,
  k1: u32,
  x: vec3<f32>,
  n: vec3<f32>,
  valid: bool,
}

fn res_none() -> Res {
  return Res(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0), 0.0, vec3<f32>(0.0), 0.0, 0.0,
             0u, 0u, 0u, vec3<f32>(0.0), vec3<f32>(0.0), false);
}

fn res_begin(x: vec3<f32>, n: vec3<f32>, k0: u32, k1: u32, rev: u32) -> Res {
  var r = res_none();
  r.x = x;
  r.n = n;
  r.k0 = k0;
  r.k1 = k1;
  r.rev = rev;
  return r;
}

fn res_pack_n(n: vec3<f32>) -> vec2<u32> {
  return vec2<u32>(pack2x16float(n.xy), pack2x16float(vec2<f32>(n.z, 0.0)));
}

fn res_unpack_n(a: u32, b: u32) -> vec3<f32> {
  let lo = unpack2x16float(a);
  let n = vec3<f32>(lo.x, lo.y, unpack2x16float(b).x);
  let l = length(n);
  return select(vec3<f32>(0.0, 1.0, 0.0), n / l, l > 1e-6);
}

fn res_load_prev(idx: u32) -> Res {
  let b = idx * RES_WORDS;
  let flags = atomicLoad(&res_prev[b + RES_W_FLAGS]);
  var r = res_none();
  r.valid = (flags & RES_VALID) != 0u;
  r.y = vec3<f32>(bitcast<f32>(atomicLoad(&res_prev[b + RES_W_Y])),
                  bitcast<f32>(atomicLoad(&res_prev[b + RES_W_Y + 1u])),
                  bitcast<f32>(atomicLoad(&res_prev[b + RES_W_Y + 2u])));
  r.le = rgb9e5_unpack(atomicLoad(&res_prev[b + RES_W_LE]));
  r.nl = res_unpack_n(atomicLoad(&res_prev[b + RES_W_NL]),
                      atomicLoad(&res_prev[b + RES_W_NL + 1u]));
  r.pa = bitcast<f32>(atomicLoad(&res_prev[b + RES_W_PA]));
  r.sum = vec3<f32>(bitcast<f32>(atomicLoad(&res_prev[b + RES_W_SUM])),
                    bitcast<f32>(atomicLoad(&res_prev[b + RES_W_SUM + 1u])),
                    bitcast<f32>(atomicLoad(&res_prev[b + RES_W_SUM + 2u])));
  r.m = bitcast<f32>(atomicLoad(&res_prev[b + RES_W_M]));
  r.vis = f32(atomicLoad(&res_prev[b + RES_W_VIS]));
  r.rev = atomicLoad(&res_prev[b + RES_W_REV]);
  r.k0 = atomicLoad(&res_prev[b + RES_W_K0]);
  r.k1 = atomicLoad(&res_prev[b + RES_W_K1]);
  r.x = vec3<f32>(bitcast<f32>(atomicLoad(&res_prev[b + RES_W_X])),
                  bitcast<f32>(atomicLoad(&res_prev[b + RES_W_X + 1u])),
                  bitcast<f32>(atomicLoad(&res_prev[b + RES_W_X + 2u])));
  r.n = res_unpack_n(atomicLoad(&res_prev[b + RES_W_N]),
                     atomicLoad(&res_prev[b + RES_W_N + 1u]));
  return r;
}

fn res_load_tmp(idx: u32) -> Res {
  let b = idx * RES_WORDS;
  let flags = atomicLoad(&res_tmp[b + RES_W_FLAGS]);
  var r = res_none();
  r.valid = (flags & RES_VALID) != 0u;
  r.y = vec3<f32>(bitcast<f32>(atomicLoad(&res_tmp[b + RES_W_Y])),
                  bitcast<f32>(atomicLoad(&res_tmp[b + RES_W_Y + 1u])),
                  bitcast<f32>(atomicLoad(&res_tmp[b + RES_W_Y + 2u])));
  r.le = rgb9e5_unpack(atomicLoad(&res_tmp[b + RES_W_LE]));
  r.nl = res_unpack_n(atomicLoad(&res_tmp[b + RES_W_NL]),
                      atomicLoad(&res_tmp[b + RES_W_NL + 1u]));
  r.pa = bitcast<f32>(atomicLoad(&res_tmp[b + RES_W_PA]));
  r.sum = vec3<f32>(bitcast<f32>(atomicLoad(&res_tmp[b + RES_W_SUM])),
                    bitcast<f32>(atomicLoad(&res_tmp[b + RES_W_SUM + 1u])),
                    bitcast<f32>(atomicLoad(&res_tmp[b + RES_W_SUM + 2u])));
  r.m = bitcast<f32>(atomicLoad(&res_tmp[b + RES_W_M]));
  r.vis = f32(atomicLoad(&res_tmp[b + RES_W_VIS]));
  r.rev = atomicLoad(&res_tmp[b + RES_W_REV]);
  r.k0 = atomicLoad(&res_tmp[b + RES_W_K0]);
  r.k1 = atomicLoad(&res_tmp[b + RES_W_K1]);
  r.x = vec3<f32>(bitcast<f32>(atomicLoad(&res_tmp[b + RES_W_X])),
                  bitcast<f32>(atomicLoad(&res_tmp[b + RES_W_X + 1u])),
                  bitcast<f32>(atomicLoad(&res_tmp[b + RES_W_X + 2u])));
  r.n = res_unpack_n(atomicLoad(&res_tmp[b + RES_W_N]),
                     atomicLoad(&res_tmp[b + RES_W_N + 1u]));
  return r;
}

fn res_store_tmp(idx: u32, r: Res) {
  let b = idx * RES_WORDS;
  let n0 = res_pack_n(r.n);
  let nl = res_pack_n(r.nl);
  var flags = 0u;
  if (r.valid) { flags = flags | RES_VALID; }
  atomicStore(&res_tmp[b + RES_W_Y], bitcast<u32>(r.y.x));
  atomicStore(&res_tmp[b + RES_W_Y + 1u], bitcast<u32>(r.y.y));
  atomicStore(&res_tmp[b + RES_W_Y + 2u], bitcast<u32>(r.y.z));
  atomicStore(&res_tmp[b + RES_W_LE], rgb9e5_pack(r.le));
  atomicStore(&res_tmp[b + RES_W_NL], nl.x);
  atomicStore(&res_tmp[b + RES_W_NL + 1u], nl.y);
  atomicStore(&res_tmp[b + RES_W_PA], bitcast<u32>(r.pa));
  atomicStore(&res_tmp[b + RES_W_SUM], bitcast<u32>(r.sum.x));
  atomicStore(&res_tmp[b + RES_W_SUM + 1u], bitcast<u32>(r.sum.y));
  atomicStore(&res_tmp[b + RES_W_SUM + 2u], bitcast<u32>(r.sum.z));
  atomicStore(&res_tmp[b + RES_W_M], bitcast<u32>(r.m));
  atomicStore(&res_tmp[b + RES_W_VIS], u32(r.vis));
  atomicStore(&res_tmp[b + RES_W_REV], r.rev);
  atomicStore(&res_tmp[b + RES_W_K0], r.k0);
  atomicStore(&res_tmp[b + RES_W_K1], r.k1);
  atomicStore(&res_tmp[b + RES_W_X], bitcast<u32>(r.x.x));
  atomicStore(&res_tmp[b + RES_W_X + 1u], bitcast<u32>(r.x.y));
  atomicStore(&res_tmp[b + RES_W_X + 2u], bitcast<u32>(r.x.z));
  atomicStore(&res_tmp[b + RES_W_N], n0.x);
  atomicStore(&res_tmp[b + RES_W_N + 1u], n0.y);
  atomicStore(&res_tmp[b + RES_W_FLAGS], flags);
  atomicStore(&res_tmp[b + RES_W_FLAGS + 1u], 0u);
  atomicStore(&res_tmp[b + RES_W_FLAGS + 2u], 0u);
}

fn res_load_cur(idx: u32) -> Res {
  let b = idx * RES_WORDS;
  let flags = atomicLoad(&res_cur[b + RES_W_FLAGS]);
  var r = res_none();
  r.valid = (flags & RES_VALID) != 0u;
  r.y = vec3<f32>(bitcast<f32>(atomicLoad(&res_cur[b + RES_W_Y])),
                  bitcast<f32>(atomicLoad(&res_cur[b + RES_W_Y + 1u])),
                  bitcast<f32>(atomicLoad(&res_cur[b + RES_W_Y + 2u])));
  r.le = rgb9e5_unpack(atomicLoad(&res_cur[b + RES_W_LE]));
  r.nl = res_unpack_n(atomicLoad(&res_cur[b + RES_W_NL]),
                      atomicLoad(&res_cur[b + RES_W_NL + 1u]));
  r.pa = bitcast<f32>(atomicLoad(&res_cur[b + RES_W_PA]));
  r.sum = vec3<f32>(bitcast<f32>(atomicLoad(&res_cur[b + RES_W_SUM])),
                    bitcast<f32>(atomicLoad(&res_cur[b + RES_W_SUM + 1u])),
                    bitcast<f32>(atomicLoad(&res_cur[b + RES_W_SUM + 2u])));
  r.m = bitcast<f32>(atomicLoad(&res_cur[b + RES_W_M]));
  r.vis = f32(atomicLoad(&res_cur[b + RES_W_VIS]));
  r.rev = atomicLoad(&res_cur[b + RES_W_REV]);
  r.k0 = atomicLoad(&res_cur[b + RES_W_K0]);
  r.k1 = atomicLoad(&res_cur[b + RES_W_K1]);
  r.x = vec3<f32>(bitcast<f32>(atomicLoad(&res_cur[b + RES_W_X])),
                  bitcast<f32>(atomicLoad(&res_cur[b + RES_W_X + 1u])),
                  bitcast<f32>(atomicLoad(&res_cur[b + RES_W_X + 2u])));
  r.n = res_unpack_n(atomicLoad(&res_cur[b + RES_W_N]),
                     atomicLoad(&res_cur[b + RES_W_N + 1u]));
  return r;
}

fn res_direct(r: Res) -> vec3<f32> {
  if (!r.valid || r.m <= 0.0) { return vec3<f32>(0.0); }
  return r.sum / r.m;
}

fn res_store_cur(idx: u32, r: Res) {  let b = idx * RES_WORDS;
  let n0 = res_pack_n(r.n);
  let nl = res_pack_n(r.nl);
  var flags = 0u;
  if (r.valid) { flags = flags | RES_VALID; }
  atomicStore(&res_cur[b + RES_W_Y], bitcast<u32>(r.y.x));
  atomicStore(&res_cur[b + RES_W_Y + 1u], bitcast<u32>(r.y.y));
  atomicStore(&res_cur[b + RES_W_Y + 2u], bitcast<u32>(r.y.z));
  atomicStore(&res_cur[b + RES_W_LE], rgb9e5_pack(r.le));
  atomicStore(&res_cur[b + RES_W_NL], nl.x);
  atomicStore(&res_cur[b + RES_W_NL + 1u], nl.y);
  atomicStore(&res_cur[b + RES_W_PA], bitcast<u32>(r.pa));
  atomicStore(&res_cur[b + RES_W_SUM], bitcast<u32>(r.sum.x));
  atomicStore(&res_cur[b + RES_W_SUM + 1u], bitcast<u32>(r.sum.y));
  atomicStore(&res_cur[b + RES_W_SUM + 2u], bitcast<u32>(r.sum.z));
  atomicStore(&res_cur[b + RES_W_M], bitcast<u32>(r.m));
  atomicStore(&res_cur[b + RES_W_VIS], u32(r.vis));
  atomicStore(&res_cur[b + RES_W_REV], r.rev);
  atomicStore(&res_cur[b + RES_W_K0], r.k0);
  atomicStore(&res_cur[b + RES_W_K1], r.k1);
  atomicStore(&res_cur[b + RES_W_X], bitcast<u32>(r.x.x));
  atomicStore(&res_cur[b + RES_W_X + 1u], bitcast<u32>(r.x.y));
  atomicStore(&res_cur[b + RES_W_X + 2u], bitcast<u32>(r.x.z));
  atomicStore(&res_cur[b + RES_W_N], n0.x);
  atomicStore(&res_cur[b + RES_W_N + 1u], n0.y);
  atomicStore(&res_cur[b + RES_W_FLAGS], flags);
  atomicStore(&res_cur[b + RES_W_FLAGS + 1u], 0u);
  atomicStore(&res_cur[b + RES_W_FLAGS + 2u], 0u);
}

fn res_chat(y: vec3<f32>, le: vec3<f32>, nl: vec3<f32>, pa: f32,
            x: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
  if (!(pa > 0.0)) { return vec3<f32>(0.0); }
  let d = y - x;
  let dist2 = max(dot(d, d), 1e-6);
  let dir = d * inverseSqrt(dist2);
  let cr = dot(n, dir);
  let cl = -dot(nl, dir);
  if (cr <= 0.0 || cl <= 1e-4) { return vec3<f32>(0.0); }
  let pdf_w = pa * dist2 / cl;
  let w_light = pdf_w / (pdf_w + cr / GI_PI);
  return le * (cr / (GI_PI * pdf_w)) * w_light;
}

fn res_pick_chat(p: LtSample, x: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
  if (!p.ok) { return vec3<f32>(0.0); }
  return res_chat(p.y, p.le, p.nl, p.pa, x, n);
}

fn res_add_pick(r: Res, p: LtSample, vis: f32, rnd: f32) -> Res {
  var out = r;
  if (!p.ok) {
    out.m = out.m + 1.0;
    return out;
  }
  let c = res_pick_chat(p, out.x, out.n);
  let w = dot(c, GI_LUMA);
  out.sum = out.sum + c * vis;
  out.m = out.m + 1.0;
  if (rnd * max(dot(out.sum, GI_LUMA), 1e-9) < w) {
    out.y = p.y;
    out.le = p.le;
    out.nl = p.nl;
    out.pa = p.pa;
    out.vis = vis;
  }
  return out;
}

fn res_adopt(r: Res, p: Res, rnd: f32) -> Res {
  if (p.m <= 0.0 || p.rev != r.rev || !res_same_surface(p, r)) { return r; }
  var out = r;
  let ps = dot(p.sum, GI_LUMA);
  out.sum = out.sum + p.sum;
  out.m = out.m + p.m;
  if (rnd * max(dot(out.sum, GI_LUMA), 1e-9) < ps) {
    out.y = p.y;
    out.le = p.le;
    out.nl = p.nl;
    out.pa = p.pa;
    out.vis = p.vis;
  }
  return out;
}

fn res_add_nb(r: Res, p: Res, rnd: f32) -> Res {
  if (p.m <= 0.0 || !res_same_surface(p, r)) { return r; }
  let c = res_chat(p.y, p.le, p.nl, p.pa, r.x, r.n) * p.vis;
  let w = dot(c, GI_LUMA);
  var out = r;
  out.sum = out.sum + c;
  out.m = out.m + 1.0;
  if (rnd * max(dot(out.sum, GI_LUMA), 1e-9) < w) {
    out.y = p.y;
    out.le = p.le;
    out.nl = p.nl;
    out.pa = p.pa;
    out.vis = p.vis;
  }
  return out;
}

fn res_finish(r: Res) -> Res {
  var out = r;
  if (out.m > RES_M_CAP) {
    out.sum = out.sum * (RES_M_CAP / out.m);
    out.m = RES_M_CAP;
  }
  out.valid = out.m > 0.0;
  return out;
}
