import package::bindings::{b_leaves, b_palette, grid_descs, lod_diag, lod_req, occ, view_u};
@if(ray_query)
import package::bindings::{tlas};
import package::brickmap::{Grid, ts};
import package::common::{CHUNK_INDEX_CAP, CHUNK_SIZE, DIR_LUT, FLAGS_IS_PBR, FLAGS_TRANSMISSIVE, INDEX_ENTRY_EMPTY, NODE_FIXED_WORDS, face_normal_from_index};

struct VoxelHit {
  hit: bool,
  t: f32,
  pal: u32,
  face_id: u32,
  voxel: vec3<i32>,
}

struct MediumRay {
  stop_on_air: bool,
  palette_base: u32,
  last_pal: u32,
  tint: vec3<f32>,
}

struct Brick {
  addr: u32,    mask: u64,    pal: u32,         pal_rep: u32,
}

fn mask_bit(mask: u64, idx: u32) -> bool {
  return (mask & (u64(1) << idx)) != u64(0);
}

fn mask_pop_below(mask: u64, idx: u32) -> u32 {
  let below = mask & ((u64(1) << idx) - u64(1));
  return u32(countOneBits(below));
}

const LEAF_LOD_FP: f32 = 3.4e38;

const LEAF_LOD_SCAN_CAP: u32 = 8u;

const LOD_DIAG: u32 = 1u;
const DIAG_LEAF_ENTRIES: u32 = 0u;
const DIAG_LEAF_STOPS: u32 = 1u;
const DIAG_ILLEGAL_STOPS: u32 = 2u;
const DIAG_LEAF_ITER: u32 = 32u;
const DIAG_LEAF_MARCH: u32 = 33u;
const DIAG_NEAR_CALLS: u32 = 34u;
const DIAG_BVH_NODES: u32 = 35u;
const DIAG_FAR_ITERS: u32 = 36u;
const DIAG_FAR_TRACES: u32 = 37u;
fn diag_ray(origin: vec3<f32>, dir: vec3<f32>) -> bool {
  return LOD_DIAG == 1u && ((bitcast<u32>(origin.x) ^ bitcast<u32>(dir.z)) & 63u) == 0u;
}
fn diag_sampled(addr: u32) -> bool {
  return LOD_DIAG == 1u && (addr & 63u) == 0u;
}

const REQ_ENABLE: u32 = 0u;
const REQ_SAMPLE: u32 = 16u;
const REQ_PER_RAY_MAX: u32 = 2u;
const REQ_COUNT: u32 = 0u;
const REQ_RESERVED: u32 = 1u;
const REQ_REL_BITS: u32 = 6u;
const REQ_LEVEL_KEEP: u32 = 1u;

const GRID_VOLUMES: u32 = 4u;

const USE_WORDS: u32 = 64u * 64u * 64u;
const USE_TICK: u32 = 2u;
const USE_BASE: u32 = 3u;
const REQ_BASE: u32 = 3u + GRID_VOLUMES * USE_WORDS;

fn req_level(fp: f32) -> u32 {
  if (fp >= 64.0) { return 3u; }
  if (fp >= 16.0) { return 2u; }
  if (fp >= 4.0) { return 1u; }
  return 0u;
}

fn req_hash(origin: vec3<f32>, dir: vec3<f32>) -> u32 {
  return u32(bitcast<i32>(origin.x * 64.0)) ^ (u32(bitcast<i32>(origin.y * 64.0)) << 11u)
    ^ (u32(bitcast<i32>(origin.z * 64.0)) << 21u) ^ u32(bitcast<i32>(dir.z * 4096.0));
}

const REQ_SAMPLE_FAR: u32 = 64u;

fn req_sampled(origin: vec3<f32>, dir: vec3<f32>, vol: u32) -> bool {
  let m = select(REQ_SAMPLE, REQ_SAMPLE_FAR, vol > 0u);
  return (req_hash(origin, dir) % m) == 0u;
}

fn req_push(rel: vec3<u32>, level: u32, kind: u32, vol: u32) {
  if (REQ_ENABLE == 0u) { return; }
  let cap = arrayLength(&lod_req) - REQ_BASE;
  let n = atomicAdd(&lod_req[REQ_COUNT], 1u);
  let w = (rel.x & 63u)
    | ((rel.y & 63u) << REQ_REL_BITS)
    | ((rel.z & 63u) << (REQ_REL_BITS * 2u))
    | ((level & 3u) << 18u)
    | ((kind & 1u) << 20u)
    | ((vol & 3u) << 21u);
  atomicStore(&lod_req[REQ_BASE + (n % cap)], w);
}

fn req_use(rel: vec3<u32>, vol: u32) {
  if (REQ_ENABLE == 0u) { return; }
  let i = (rel.x & 63u)
    | ((rel.y & 63u) << REQ_REL_BITS)
    | ((rel.z & 63u) << (REQ_REL_BITS * 2u));
  let t = atomicAdd(&lod_req[USE_TICK], 1u);
  atomicMax(&lod_req[USE_BASE + (vol & 3u) * USE_WORDS + i], t);
}

fn leaf_lod_pal(b: Brick) -> u32 {
  if (b.pal_rep != 0u) { return b.pal_rep; }
  if (b.mask != ~u64(0) && b.pal != 0u) { return b.pal; }
  var m = b.mask;
  for (var k: u32 = 0u; k < LEAF_LOD_SCAN_CAP; k = k + 1u) {
    if (m == u64(0)) { return 0u; }
        let i = u32(countOneBits(m & (u64(0) - m)) - u64(1));
    let w = ts(b.addr + NODE_FIXED_WORDS + (i >> 1u));
    let v = (w >> ((i & 1u) * 16u)) & 0xFFFFu;
    if (v != 0u) { return v; }
    m = m & (m - u64(1));
  }
  return 0u;
}

fn medium_of(palette_base: u32, pal: u32) -> vec4<f32> {
  let w1 = b_palette[palette_base + pal * 2u + 1u];
      let flags = (w1 >> 16u) & 0xFFu;
    if ((flags & FLAGS_TRANSMISSIVE) == 0u) { return vec4<f32>(1.0, 1.0, 1.0, 1.0); }
  if ((flags & FLAGS_IS_PBR) != 0u) {
            return vec4<f32>(1.0, 1.0, 1.0, 0.0);
  }
  let w0 = b_palette[palette_base + pal * 2u];
  let srgb = vec3<f32>(f32(w0 & 0xFFu), f32((w0 >> 8u) & 0xFFu), f32((w0 >> 16u) & 0xFFu)) / 255.0;
  let a = 1.0 - f32((w1 >> 8u) & 0xFFu) / 255.0;
            return vec4<f32>((1.0 - a) * mix(vec3<f32>(1.0), srgb * srgb, a), a);
}

const DIAG_DISABLE_MEDIUM_TINT: u32 = 0u;

fn pass_through_medium(mc: ptr<function, MediumRay>, pal: u32) -> bool {
  if (pal == (*mc).last_pal) { return true; }
  let m = medium_of((*mc).palette_base, pal);
  if (m.a >= 1.0) { return false; }
  if (DIAG_DISABLE_MEDIUM_TINT == 0u) { (*mc).tint = (*mc).tint * m.rgb; }
  (*mc).last_pal = pal;
  return true;
}

fn medium_exit(mc: ptr<function, MediumRay>) -> bool {
  return (*mc).stop_on_air && ((*mc).last_pal != 0u);
}

struct RayBasis {
    inv_rd: vec3<f32>,
    abs_inv_rd: vec3<f32>,
    axis_on: vec3<bool>,
    dir_side: vec3<f32>,
    face_base: vec3<u32>,
    oct: u32,
}

fn trace_chunk_opaque(chunk_base: u32, chunk_min: vec3<f32>,
                      ro: vec3<f32>, rd: vec3<f32>, sign_v: vec3<i32>, basis: RayBasis,
                      t0: f32, t1: f32, entry_face: u32, depth_cap: u32) -> VoxelHit {
  if (t0 >= t1) { return VoxelHit(false, 0.0, 0u, 0u, vec3<i32>(0)); }
    let ro_c = ro - chunk_min;
  let inv_rd = basis.inv_rd;
  let abs_inv_rd = basis.abs_inv_rd;
  let axis_on = basis.axis_on;
  let dir_side = basis.dir_side;
  let face_base = basis.face_base;
  let oct = basis.oct;
  var bricks: array<Brick, 4u>;
  let r_ml = ts(chunk_base);
  let r_mh = ts(chunk_base + 1u);
  let r_pw = ts(chunk_base + 2u);
  let r_pal = r_pw & 0xFFFFu;    let r_rep = r_pw >> 16u;       bricks[3u] = Brick(chunk_base, (u64(r_mh) << 32u) | u64(r_ml), r_pal, r_rep);
  var level: u32 = 3u;
    let p0 = ro_c + rd * t0;
  var v = clamp(vec3<i32>(floor(p0)), vec3<i32>(0), vec3<i32>(255));
  var cur_t = t0;
  var face = entry_face;
    var budget: u32 = 65536u;
  let lut_disable = view_u.lod.w > 0.5;
    let leaf_lod = view_u.lod.y > 0.5;
  let px_ang = view_u.lod.x;
  loop {
    if (budget == 0u) { break; }
    budget = budget - 1u;
    if (level > 3u) { return VoxelHit(false, 0.0, 0u, 0u, vec3<i32>(0)); }
        loop {
      let b = bricks[level];
            let sh = vec3<u32>(level * 2u);
      let cell = vec3<i32>(v >> sh) & vec3<i32>(3);
      let idx = u32(cell.z * 16 + cell.y * 4 + cell.x);
      if (level == 0u) {
                        if (mask_bit(b.mask, idx)) {
          let w = ts(b.addr + NODE_FIXED_WORDS + (idx >> 1u));
          let leaf_pal = (w >> ((idx & 1u) * 16u)) & 0xFFFFu;
          if (leaf_pal != 0u) { return VoxelHit(true, cur_t, leaf_pal, face, v); }
        } else if (b.pal != 0u) {
          return VoxelHit(true, cur_t, b.pal, face, v);
        }
                        let diag = diag_sampled(b.addr);
        if (diag) { atomicAdd(&lod_diag[DIAG_LEAF_ENTRIES], 1u); }
                        if (leaf_lod && b.mask != u64(0) && cur_t * px_ang >= LEAF_LOD_FP) {
          let p = leaf_lod_pal(b);
          if (p != 0u) {
            if (diag) { atomicAdd(&lod_diag[DIAG_LEAF_STOPS], 1u); }
            return VoxelHit(true, cur_t, p, face, v);
          }
        }
        break;
      }
      if (!mask_bit(b.mask, idx)) {
                if (b.pal != 0u) { return VoxelHit(true, cur_t, b.pal, face, v); }
        break;
      }
                                    let gd = 3u - level;
      if (gd >= depth_cap) {
                if (LOD_DIAG == 1u && b.pal == 0u && (b.addr & 63u) == 0u) {
          atomicAdd(&lod_diag[DIAG_ILLEGAL_STOPS], 1u);
        }
        return VoxelHit(true, cur_t, b.pal, face, v);
      }
      let pop = mask_pop_below(b.mask, idx);
      let child_addr = chunk_base + ts(b.addr + NODE_FIXED_WORDS + pop);
      let c_ml = ts(child_addr);
      let c_mh = ts(child_addr + 1u);
      let c_pw = ts(child_addr + 2u);
      let c_pal = c_pw & 0xFFFFu;        let c_rep = c_pw >> 16u;           let c_mask = (u64(c_mh) << 32u) | u64(c_ml);
            if (c_mask == u64(0)) {
        if (c_pal != 0u) { return VoxelHit(true, cur_t, c_pal, face, v); }
        break;
      }
      level = level - 1u;
      bricks[level] = Brick(child_addr, c_mask, c_pal, c_rep);
    }
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
        let log2 = level * 2u;
    let s = 1i << (level * 2u);     let s_mask = ~(s - 1i);
        let base_v = v & vec3<i32>(s_mask);
    let boundary = vec3<f32>(base_v) + dir_side * f32(s);
    var side = vec3<f32>(1e+30);
    side = select(side, (boundary - ro_c) * inv_rd, axis_on);
    side = max(side, vec3<f32>(cur_t));
    let step_inc = f32(s) * abs_inv_rd;
    var step_axis: u32 = 0u;
    var changed = false;
        let b = bricks[level];
    let sh2 = vec3<u32>(log2);
    let sm_mask = vec3<i32>(s_mask);
    let sm1 = vec3<i32>(s - 1);
    loop {
      var mn: u32;
      if (side.x <= side.y && side.x <= side.z) { mn = 0u; }
      else if (side.y <= side.z) { mn = 1u; }
      else { mn = 2u; }
      step_axis = mn;
      cur_t = side[mn];
      if (cur_t >= t1) { return VoxelHit(false, 0.0, 0u, 0u, vec3<i32>(0)); }
      let old_cell = (v[mn] >> log2) & 3;
                  let cell_min = v & sm_mask;
      v = clamp(vec3<i32>(floor(ro_c + rd * cur_t)), cell_min, cell_min + sm1);
      v[mn] = select(cell_min[mn] + s, cell_min[mn] - 1i, sign_v[mn] < 0);
      side[mn] = side[mn] + step_inc[mn];
      face = mn * 2u + face_base[mn];
      let crossed = select(old_cell == 0, old_cell == 3, sign_v[mn] >= 0);
      if (crossed) { changed = true; break; }
      let cell = vec3<i32>(v >> sh2) & vec3<i32>(3);
      let idx = u32(cell.z * 16 + cell.y * 4 + cell.x);
      if (level == 0u) {
                if (mask_bit(b.mask, idx)) {
          let w = ts(b.addr + NODE_FIXED_WORDS + (idx >> 1u));
          let dp = (w >> ((idx & 1u) * 16u)) & 0xFFFFu;
          if (dp != 0u) { return VoxelHit(true, cur_t, dp, face, v); }
        } else if (b.pal != 0u) {
          return VoxelHit(true, cur_t, b.pal, face, v);
        }
      } else {
        let mb = mask_bit(b.mask, idx);
        if (mb) { break; }
        if (b.pal != 0u) { return VoxelHit(true, cur_t, b.pal, face, v); }
      }
    }
    if (changed) {
      level = level + 1u;
      if (level > 3u) { return VoxelHit(false, 0.0, 0u, 0u, vec3<i32>(0)); }
    }
            let positive = sign_v[step_axis] >= 0;
    let cur_log2 = level * 2u;
    let m: u32 = 0xFFFFFFFFu << cur_log2;
    let not_m = ~m;
    let vmin_u = u32(v[step_axis]);
    var comp: u32;
    if (positive) { comp = vmin_u & m; } else { comp = (vmin_u & m) | not_m; }
    let new_level = firstTrailingBit(comp + select(1u, 0u, positive)) >> 1u;
    level = max(level, new_level);
    if (level > 3u) { return VoxelHit(false, 0.0, 0u, 0u, vec3<i32>(0)); }
        let mi = i32(m);
    let base = v & vec3<i32>(mi);
    let p = ro_c + rd * cur_t;
    let region_max = base + vec3<i32>(i32(not_m));
    v = clamp(vec3<i32>(floor(p)), base, region_max);
    v[step_axis] = i32(comp);
  }
  return VoxelHit(false, 0.0, 0u, 0u, vec3<i32>(0));
}

fn trace_chunk(chunk_base: u32, chunk_min: vec3<f32>,
               ro: vec3<f32>, rd: vec3<f32>, sign_v: vec3<i32>, basis: RayBasis,
               t0: f32, t1: f32, entry_face: u32, depth_cap: u32,
               mc: ptr<function, MediumRay>) -> VoxelHit {
  if (t0 >= t1) { return VoxelHit(false, 0.0, 0u, 0u, vec3<i32>(0)); }
    let ro_c = ro - chunk_min;
  let inv_rd = basis.inv_rd;
  let abs_inv_rd = basis.abs_inv_rd;
  let axis_on = basis.axis_on;
  let dir_side = basis.dir_side;
  let face_base = basis.face_base;
  let oct = basis.oct;
  var bricks: array<Brick, 4u>;
  let r_ml = ts(chunk_base);
  let r_mh = ts(chunk_base + 1u);
  let r_pw = ts(chunk_base + 2u);
  let r_pal = r_pw & 0xFFFFu;    let r_rep = r_pw >> 16u;       bricks[3u] = Brick(chunk_base, (u64(r_mh) << 32u) | u64(r_ml), r_pal, r_rep);
  var level: u32 = 3u;
    let p0 = ro_c + rd * t0;
  var v = clamp(vec3<i32>(floor(p0)), vec3<i32>(0), vec3<i32>(255));
  var cur_t = t0;
  var face = entry_face;
    var budget: u32 = 65536u;
  let lut_disable = view_u.lod.w > 0.5;
    let leaf_lod = view_u.lod.y > 0.5;
  let px_ang = view_u.lod.x;
  loop {
    if (budget == 0u) { break; }
    budget = budget - 1u;
    if (level > 3u) { return VoxelHit(false, 0.0, 0u, 0u, vec3<i32>(0)); }
        var hit_pal = 0u;
    loop {
      let b = bricks[level];
            let sh = vec3<u32>(level * 2u);
      let cell = vec3<i32>(v >> sh) & vec3<i32>(3);
      let idx = u32(cell.z * 16 + cell.y * 4 + cell.x);
      if (level == 0u) {
                if (mask_bit(b.mask, idx)) {
          let w = ts(b.addr + NODE_FIXED_WORDS + (idx >> 1u));
          hit_pal = (w >> ((idx & 1u) * 16u)) & 0xFFFFu;
        } else {
          hit_pal = b.pal;
        }
                        let diag = diag_sampled(b.addr);
        if (diag) { atomicAdd(&lod_diag[DIAG_LEAF_ENTRIES], 1u); }
        if (hit_pal == 0u && leaf_lod && b.mask != u64(0) && cur_t * px_ang >= LEAF_LOD_FP) {
          let p = leaf_lod_pal(b);
          if (p != 0u) {
            if (diag) { atomicAdd(&lod_diag[DIAG_LEAF_STOPS], 1u); }
            hit_pal = p;
          }
        }
        break;
      }
      if (!mask_bit(b.mask, idx)) {
                hit_pal = b.pal;
        break;
      }
                        let gd = 3u - level;
      if (gd >= depth_cap) {
                if (LOD_DIAG == 1u && b.pal == 0u && (b.addr & 63u) == 0u) {
          atomicAdd(&lod_diag[DIAG_ILLEGAL_STOPS], 1u);
        }
        return VoxelHit(true, cur_t, b.pal, face, v);
      }
      let pop = mask_pop_below(b.mask, idx);
      let child_addr = chunk_base + ts(b.addr + NODE_FIXED_WORDS + pop);
      let c_ml = ts(child_addr);
      let c_mh = ts(child_addr + 1u);
      let c_pw = ts(child_addr + 2u);
      let c_pal = c_pw & 0xFFFFu;        let c_rep = c_pw >> 16u;           let c_mask = (u64(c_mh) << 32u) | u64(c_ml);
            if (c_mask == u64(0)) {
        hit_pal = c_pal;
        break;
      }
      level = level - 1u;
      bricks[level] = Brick(child_addr, c_mask, c_pal, c_rep);
    }
        if (hit_pal != 0u) {
      if (!pass_through_medium(mc, hit_pal)) {
        return VoxelHit(true, cur_t, hit_pal, face, v);
      }
    } else {
                  if (medium_exit(mc)) { return VoxelHit(true, cur_t, (*mc).last_pal, face, v); }
      (*mc).last_pal = 0u;
    }
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
        let log2 = level * 2u;
    let s = 1i << (level * 2u);     let s_mask = ~(s - 1i);
        let base_v = v & vec3<i32>(s_mask);
    let boundary = vec3<f32>(base_v) + dir_side * f32(s);
    var side = vec3<f32>(1e+30);
    side = select(side, (boundary - ro_c) * inv_rd, axis_on);
    side = max(side, vec3<f32>(cur_t));
    let step_inc = f32(s) * abs_inv_rd;
    var step_axis: u32 = 0u;
    var changed = false;
        let b = bricks[level];
    let sh2 = vec3<u32>(log2);
    let sm_mask = vec3<i32>(s_mask);
    let sm1 = vec3<i32>(s - 1);
    loop {
      var mn: u32;
      if (side.x <= side.y && side.x <= side.z) { mn = 0u; }
      else if (side.y <= side.z) { mn = 1u; }
      else { mn = 2u; }
      step_axis = mn;
      cur_t = side[mn];
      if (cur_t >= t1) {
                        return VoxelHit(false, 0.0, 0u, 0u, vec3<i32>(0));
      }
      let old_cell = (v[mn] >> log2) & 3;
            let cell_min = v & sm_mask;
      v = clamp(vec3<i32>(floor(ro_c + rd * cur_t)), cell_min, cell_min + sm1);
      v[mn] = select(cell_min[mn] + s, cell_min[mn] - 1i, sign_v[mn] < 0);
      side[mn] = side[mn] + step_inc[mn];
      face = mn * 2u + face_base[mn];
      let crossed = select(old_cell == 0, old_cell == 3, sign_v[mn] >= 0);
      if (crossed) { changed = true; break; }
      let cell = vec3<i32>(v >> sh2) & vec3<i32>(3);
      let idx = u32(cell.z * 16 + cell.y * 4 + cell.x);
      hit_pal = 0u;
      if (level == 0u) {
        if (mask_bit(b.mask, idx)) {
          let w = ts(b.addr + NODE_FIXED_WORDS + (idx >> 1u));
          hit_pal = (w >> ((idx & 1u) * 16u)) & 0xFFFFu;
        } else {
          hit_pal = b.pal;          }
      } else {
        if (mask_bit(b.mask, idx)) { break; }
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
            let positive = sign_v[step_axis] >= 0;
    let cur_log2 = level * 2u;
    let m: u32 = 0xFFFFFFFFu << cur_log2;
    let not_m = ~m;
    let vmin_u = u32(v[step_axis]);
    var comp: u32;
    if (positive) { comp = vmin_u & m; } else { comp = (vmin_u & m) | not_m; }
    let new_level = firstTrailingBit(comp + select(1u, 0u, positive)) >> 1u;
    level = max(level, new_level);
    if (level > 3u) { return VoxelHit(false, 0.0, 0u, 0u, vec3<i32>(0)); }
        let mi = i32(m);
    let base = v & vec3<i32>(mi);
    let p = ro_c + rd * cur_t;
    let region_max = base + vec3<i32>(i32(not_m));
    v = clamp(vec3<i32>(floor(p)), base, region_max);
    v[step_axis] = i32(comp);
  }
  return VoxelHit(false, 0.0, 0u, 0u, vec3<i32>(0));
}

struct UnifiedHit {
  hit: bool,
  t: f32,
  pal: u32,
  n: vec3<f32>,          face_id: u32,          voxel: vec3<i32>,      obj_id: i32,         }

fn grid_hit(h: VoxelHit, ci: vec3<i32>,
            col0: vec3<f32>, col1: vec3<f32>, col2: vec3<f32>, obj_id: i32) -> UnifiedHit {
  let n_local = face_normal_from_index(h.face_id);
  let n_world = normalize(n_local.x * col0 + n_local.y * col1 + n_local.z * col2);
  let voxel = h.voxel + ci * i32(CHUNK_SIZE);
  return UnifiedHit(true, h.t, h.pal, n_world, h.face_id, voxel, obj_id);
}

const SECONDARY_REACH_CHUNKS: f32 = 12.0;

const OCC_SKIP: u32 = 1u;
const OCC_WORDS: u32 = (64u * 64u * 64u) / 32u;
const OCC_VOLUMES: u32 = 1024u;

const RT_QUERY_BUDGET: u32 = 64u;
const RT_TRUST_MISS: u32 = 0u;
const RT_CULL_MASK: u32 = 0xFFu;
const RT_FLAGS: u32 = 0u;

@if(ray_query)
fn rt_query(ro: vec3<f32>, rd: vec3<f32>, tmin: f32, tmax: f32) -> RayIntersection {
  var q: ray_query;
  rayQueryInitialize(&q, tlas, RayDesc(RT_FLAGS, RT_CULL_MASK, tmin, tmax, ro, rd));
        if (rayQueryProceed(&q)) {
    if (rayQueryGetCandidateIntersection(&q).kind != RAY_QUERY_INTERSECTION_NONE) {
      rayQueryConfirmIntersection(&q);
    }
  }
  return rayQueryGetCommittedIntersection(&q);
}

@if(ray_query)
fn chunk_exit_t(mn: vec3<f32>, ro: vec3<f32>, inv_rd: vec3<f32>, axis_on: vec3<bool>, t_hi: f32) -> f32 {
  var t1 = t_hi;
  for (var a: u32 = 0u; a < 3u; a = a + 1u) {
        if (!axis_on[a]) { continue; }
    let ta = (mn[a] - ro[a]) * inv_rd[a];
    let tb = (mn[a] + f32(CHUNK_SIZE) - ro[a]) * inv_rd[a];
    t1 = min(t1, max(ta, tb));
  }
  return t1;
}

@if(ray_query)
fn trace_grid_rt(ro: vec3<f32>, rd: vec3<f32>, tl0: f32, tl1: f32, depth_cap: u32,
                 sign_v: vec3<i32>, basis: RayBasis, g: Grid,
                 col0: vec3<f32>, col1: vec3<f32>, col2: vec3<f32>,
                 exhausted: ptr<function, u32>) -> UnifiedHit {
  let miss = UnifiedHit(false, 0.0, 0u, vec3<f32>(0.0), 0u, vec3<i32>(0), g.obj_id);
  let index_origin = g.index_origin;
  let tree_base = g.tree_base;
    let ad = abs(rd);
  var entry_face: u32 = 4u;
  if (ad.x >= max(ad.y, ad.z)) {
    entry_face = basis.face_base.x;
  } else if (ad.y >= ad.z) {
    entry_face = basis.face_base.y;
  } else {
    entry_face = basis.face_base.z;
  }
  var t_lo = tl0;
  var guard: u32 = 0u;
  loop {
    if (guard >= RT_QUERY_BUDGET) { *exhausted = 1u; break; }
    guard = guard + 1u;
    let q = rt_query(ro, rd, t_lo, tl1);
    if (q.kind == RAY_QUERY_INTERSECTION_NONE) { break; }
    let c8 = q.instance_custom_data;
            let rel = vec3<i32>(
      (i32(c8 & 0xFFu) - (index_origin.x & 0xFF)) & 0xFF,
      (i32((c8 >> 8u) & 0xFFu) - (index_origin.y & 0xFF)) & 0xFF,
      (i32((c8 >> 16u) & 0xFFu) - (index_origin.z & 0xFF)) & 0xFF,
    );
                if (any(rel < vec3<i32>(0)) || any(vec3<u32>(rel) >= g.index_dims)) { *exhausted = 1u; break; }
    let ci = index_origin + rel;
    let rel_u = vec3<u32>(rel);
    let index_addr = tree_base + rel_u.x + rel_u.y * CHUNK_INDEX_CAP
      + rel_u.z * (CHUNK_INDEX_CAP * CHUNK_INDEX_CAP);
    let entry = ts(index_addr);
        if (entry == 0u || entry == INDEX_ENTRY_EMPTY) { *exhausted = 1u; break; }
    let chunk_base = tree_base + entry - 1u;
    let chunk_min = vec3<f32>(vec3<i32>(i32(CHUNK_SIZE)) * ci);
    let t0 = max(q.t, t_lo);
    let t1 = chunk_exit_t(chunk_min, ro, basis.inv_rd, basis.axis_on, tl1);
    if (t1 <= t0) {
            t_lo = t0 + 1.0e-3;
      if (t_lo >= tl1) { break; }
      continue;
    }
    let h = trace_chunk_opaque(chunk_base, chunk_min, ro, rd, sign_v, basis,
                               t0, t1, entry_face, depth_cap);
    if (h.hit) { return grid_hit(h, ci, col0, col1, col2, g.obj_id); }
            t_lo = max(t1 + max(1.0e-3, t1 * 1.0e-6), t_lo + 1.0e-3);
    if (t_lo >= tl1) { break; }
  }
  return miss;
}

fn trace_grid(g: Grid, origin: vec3<f32>, dir: vec3<f32>, t_cap: f32, t_min: f32, depth_cap: u32,
              penetrate: bool, req: u32, mc: ptr<function, MediumRay>,
              gap: ptr<function, f32>) -> UnifiedHit {
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
    let vol = g.vol;
  let miss = UnifiedHit(false, 0.0, 0u, vec3<f32>(0.0), 0u, vec3<i32>(0), obj_id);
    var req_sent: u32 = 0u;
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
        let wp = origin - g.pos;
    ro = vec3<f32>(dot(wp, col0), dot(wp, col1), dot(wp, col2)) / scale;
    rd = vec3<f32>(dot(dir, col0), dot(dir, col1), dot(dir, col2)) / scale;
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
    tl0 = max(tl0, t_min);
  if (tl1 <= tl0) { return miss; }
    if (view_u.debug_mode.z > 1.5) { return miss; }
    var sign_v = vec3<i32>(1i);
  sign_v = select(sign_v, vec3<i32>(-1i), rd < vec3<f32>(0.0));
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
        let basis = RayBasis(
    grid_inv_rd,
    abs(grid_inv_rd),
    grid_axis_on,
    side,
    vec3<u32>(face_x & 1u, face_y & 1u, face_z & 1u),
    dir_mask);
      @if(ray_query)
  if (obj_id < 0 && !penetrate) {
    var ex: u32 = 0u;
    let h = trace_grid_rt(ro, rd, tl0, tl1, depth_cap, sign_v, basis, g, col0, col1, col2, &ex);
    if (h.hit) { return h; }
    if (RT_TRUST_MISS == 1u && ex == 0u) { return miss; }
  }
  let start = ro + rd * tl0;
  var ci = vec3<i32>(floor(start / f32(CHUNK_SIZE)));
    let bnd_c = (vec3<f32>(ci) + side) * f32(CHUNK_SIZE);
  var tmax_c = select(vec3<f32>(1e+30), (bnd_c - ro) * grid_inv_rd, grid_axis_on);
  tmax_c = max(tmax_c, vec3<f32>(tl0));
  var t_enter_c = tl0;
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
    var group_empty = false;
    if (OCC_SKIP != 0u && vol < OCC_VOLUMES && REQ_ENABLE == 0u) {
      let rg = ci - index_origin;
      if (all(rg >= vec3<i32>(0)) && all(vec3<u32>(rg) < index_dims)) {
        let rgg = vec3<u32>(rg);
        let gg = (rgg.x >> 2u) + (rgg.y >> 2u) * 16u + (rgg.z >> 2u) * 256u;
        let base = vol * OCC_WORDS + gg * 2u;
        group_empty = occ[base] == 0u && occ[base + 1u] == 0u;
      }
    }
        var air_chunk = true;
    let rel = ci - index_origin;
    if (all(rel >= vec3<i32>(0)) && all(vec3<u32>(rel) < index_dims)) {
      let rel_u = vec3<u32>(rel);
      let index_addr = tree_base + rel_u.x + rel_u.y * CHUNK_INDEX_CAP
        + rel_u.z * (CHUNK_INDEX_CAP * CHUNK_INDEX_CAP);
                  var entry: u32 = 0u;
      if (!group_empty) { entry = ts(index_addr); }
                                                let has_tree = entry != 0u && entry != INDEX_ENTRY_EMPTY;
                              if (group_empty || entry == 0u) { *gap = min(*gap, t_enter_c); }
                                    if (entry == 0u && req != 0u && req_sent < REQ_PER_RAY_MAX && req_sampled(origin, dir, vol)) {
        let lv = select(req_level(t_enter_c * view_u.lod.x), 0u, vol > 0u);
        if (lv <= REQ_LEVEL_KEEP) {
          req_push(rel_u, lv, 0u, vol);
          req_sent += 1u;
        }
      }
      if (has_tree) {
        air_chunk = false;
                                                if (req != 0u && req_sampled(origin, dir, vol)) {
          req_use(rel_u, vol);
        }
        let chunk_base = tree_base + entry - 1u;
        let chunk_min = vec3<f32>(ci) * f32(CHUNK_SIZE);
                let t0c = max(t_enter_c, t_min);
                                                                                                                var chunk_skipped = false;
        if (penetrate && t1 > t0c) {
          let r_ml = ts(chunk_base);
          let r_mh = ts(chunk_base + 1u);
          if (r_ml == 0u && r_mh == 0u) {
            let r_pal = ts(chunk_base + 2u) & 0xFFFFu;
            if (r_pal != 0u && r_pal != (*mc).last_pal) {
              let m_uni = medium_of((*mc).palette_base, r_pal);
              if (m_uni.a < 1.0) {
                if (DIAG_DISABLE_MEDIUM_TINT == 0u) {
                  (*mc).tint = (*mc).tint * m_uni.rgb;
                }
                (*mc).last_pal = r_pal;
                chunk_skipped = true;
              }
            }
          }
        }
        if (penetrate && !chunk_skipped) {
          let h = trace_chunk(chunk_base, chunk_min, ro, rd, sign_v, basis,
                              t0c, t1, entry_face, depth_cap, mc);
          if (h.hit) { return grid_hit(h, ci, col0, col1, col2, obj_id); }
        } else if (!penetrate) {
          let h = trace_chunk_opaque(chunk_base, chunk_min, ro, rd, sign_v, basis,
                                     t0c, t1, entry_face, depth_cap);
          if (h.hit) { return grid_hit(h, ci, col0, col1, col2, obj_id); }
        }
      }
    }
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
    if (penetrate && medium_exit(mc)) {
    let nl = face_normal_from_index(entry_face);
    let nw = normalize(nl.x * col0 + nl.y * col1 + nl.z * col2);
    return UnifiedHit(true, tl1, (*mc).last_pal, nw, entry_face, vec3<i32>(0), obj_id);
  }
  return miss;
}

fn make_grid(idx: u32) -> Grid {
  let d = grid_descs[idx];
  let origin = vec3<i32>(d.index_origin_x, d.index_origin_y, d.index_origin_z);
  let dims = vec3<i32>(i32(d.index_dims_x), i32(d.index_dims_y), i32(d.index_dims_z));
  let chunk_budget = u32(dims.x + dims.y + dims.z) * 3u + 16u;
    let is_world = idx == 0u;
        let l_mn = vec3<f32>(vec3<i32>(i32(CHUNK_SIZE)) * origin);
  let l_mx = l_mn + vec3<f32>(vec3<i32>(i32(CHUNK_SIZE)) * dims);
    var col0 = vec3<f32>(1.0, 0.0, 0.0);
  var col1 = vec3<f32>(0.0, 1.0, 0.0);
  var col2 = vec3<f32>(0.0, 0.0, 1.0);
  var pos = vec3<f32>(0.0);
  var scale = 1.0;
  if (!is_world) {
    col0 = d.rot0.xyz;          col1 = d.rot1.xyz;
    col2 = d.rot2.xyz;
    pos = d.pos_scale.xyz;      scale = d.pos_scale.w;
  }
  return Grid(
    d.aabb_min.xyz, d.aabb_max.xyz,           l_mn, l_mx,                                chunk_budget,
    col0, col1, col2,                          pos, scale,                                d.tree_base, d.palette_base,               origin, vec3<u32>(d.index_dims_x, d.index_dims_y, d.index_dims_z),
        select(-1i, i32(idx) - 1i, idx > 0u),
    idx,                                       d.coverage_r,                            );
}

const GRID_FLAG_FAR: u32 = 1u;

fn grid_is_far(idx: u32) -> bool {
  return (grid_descs[idx]._pad0 & GRID_FLAG_FAR) != 0u;
}

fn aabb_exit(mn: vec3<f32>, mx: vec3<f32>, origin: vec3<f32>, dir: vec3<f32>) -> f32 {
  var e = 1e+30;
  let inv = 1.0 / dir;
  for (var a: u32 = 0u; a < 3u; a = a + 1u) {
        if (abs(dir[a]) > 1e-30) {
      e = min(e, max((mn[a] - origin[a]) * inv[a], (mx[a] - origin[a]) * inv[a]));
    }
  }
  return e;
}
