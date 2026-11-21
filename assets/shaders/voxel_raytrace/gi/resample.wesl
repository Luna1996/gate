// NAADF-style GI resampler: per-pixel bands, 8x8 buckets, lit/unlit sample rings.
// Layouts and kernel order follow the Caelix GI reference; strides are mirrored by
// gate-render/src/wesl_consts.rs and the buffers in brickmap/dda.rs.
import package::bindings::{gi_out, gi_u, light_u, view_u};
import package::brickmap::{Grid};
import package::common::{sky_rgb};
import package::gi::common::{gi_cos_hemisphere, gi_key0, gi_key1, gi_key_ok, gi_local_rev, gi_pack_dir, gi_pose_salt, gi_unpack_dir};
import package::gi::consts::{GI_PI, GI_SKY_RADIANCE_SCALE, GI_T_MAX, SHADOW_SURFACE_EPS};
import package::gi::ray::{GI_REACH_T, gi_bounce2, gi_face_shade_cached};
import package::trace::make_grid;
import package::world::{WorldHit, voxel_normal_fast, world_trace_medium_scene};

const GI_BUCKET_SIZE: u32 = 8u;
const GI_REFINED_PER_BUCKET: u32 = 32u;
const GI_COMPACT_PER_BUCKET: u32 = 8u;
const GI_HISTORY_SLOTS: u32 = 128u;
const GI_MAX_GROUPS_X: u32 = 32768u;
const GI_LIT_RING_FRAMES: u32 = 2u;
const GI_UNLIT_RING_FRAMES: u32 = 4u;
const GI_MAX_HISTORY_FRAMES: u32 = 64u;

const GI_LIT_WORDS: u32 = 14u;
const GI_UNLIT_WORDS: u32 = 9u;
const GI_BUCKET_SAMPLE_WORDS: u32 = 8u;
const GI_BUCKET_INFO_WORDS: u32 = 8u;
const GI_PIXEL_INFO_WORDS: u32 = 2u;

const GI_SAMPLE_SKY_FLAG: u32 = 0x100u;
const GI_BUCKET_SKY_FLAG: u32 = 0x8u;

const GI_KEY_FLAG_HIT: u32 = 1u;
const GI_KEY_FLAG_DIFFUSE: u32 = 2u;
const GI_KEY_FLAG_OPAQUE: u32 = 4u;

const GI_COUNT_LIT_HEAD: u32 = 0u;
const GI_COUNT_UNLIT_HEAD: u32 = 1u;
const GI_COUNT_LIT_VALID: u32 = 2u;
const GI_COUNT_UNLIT_VALID: u32 = 3u;
const GI_COUNT_LIT_START: u32 = 4u;
const GI_COUNT_UNLIT_START: u32 = 5u;
const GI_COUNT_LIT_COPRIME: u32 = 6u;
const GI_COUNT_UNLIT_COPRIME: u32 = 7u;
const GI_COUNT_FRAME_BASE: u32 = 8u;

const GI_RAND_REPROJECT: u32 = 5u;
const GI_RAND_REFINE: u32 = 6u;
const GI_RAND_SPATIAL: u32 = 7u;

struct GiLitSample {
  key: u32,
  token: u32,
  ox: f32,
  oy: f32,
  oz: f32,
  vx: f32,
  vy: f32,
  vz: f32,
  primary_normal: u32,
  sx: f32,
  sy: f32,
  sz: f32,
  secondary_normal: u32,
  radiance: u32,
}

struct GiUnlitSample {
  key: u32,
  token: u32,
  ox: f32,
  oy: f32,
  oz: f32,
  vx: f32,
  vy: f32,
  vz: f32,
  primary_normal: u32,
}

struct GiBucketSample {
  px: f32,
  py: f32,
  pz: f32,
  normals: u32,
  sx: f32,
  sy: f32,
  sz: f32,
  radiance: u32,
}

struct GiBucketInfo {
  lit_count: atomic<u32>,
  unlit_count: atomic<u32>,
  dist_range: u32,
  packed: u32,
  lit_ratio: f32,
  max_luminance: f32,
  reserved0: u32,
  reserved1: u32,
}

struct GiPixelInfo {
  dist_range: u32,
  hashes: u32,
}

struct GiResUniform {
  frame: vec4<u32>,
  limits: vec4<u32>,
  settings: vec4<f32>,
  settings2: vec4<f32>,
  settings3: vec4<f32>,
  settings4: vec4<f32>,
}

@group(0) @binding(0) var<uniform> gia_u: GiResUniform;
@group(0) @binding(40) var gia_surface_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(41) var gia_key_out: texture_storage_2d<rgba32uint, write>;
@group(0) @binding(42) var gia_surface: texture_2d<f32>;
@group(0) @binding(43) var gia_key: texture_2d<u32>;
@group(0) @binding(44) var gia_own_out: texture_storage_2d<r32uint, write>;
@group(0) @binding(45) var<storage, read_write> gia_pixels: array<GiPixelInfo>;
@group(0) @binding(46) var<storage, read_write> gia_buckets: array<GiBucketInfo>;
@group(0) @binding(47) var<storage, read_write> gia_counts: array<atomic<u32>>;
@group(0) @binding(48) var<storage, read_write> gia_args: array<u32>;
@group(0) @binding(49) var<storage, read_write> gia_lit: array<GiLitSample>;
@group(0) @binding(50) var<storage, read_write> gia_unlit: array<GiUnlitSample>;
@group(0) @binding(51) var<storage, read_write> gia_refined: array<GiBucketSample>;
@group(0) @binding(52) var<storage, read_write> gia_compact: array<GiBucketSample>;

fn gi_res_frame_size() -> vec2<u32> {
  return vec2<u32>(gia_u.frame.x, gia_u.frame.y);
}

fn gi_res_frame_slot() -> u32 {
  return gia_u.frame.z % GI_HISTORY_SLOTS;
}

fn gi_res_buckets_x() -> u32 {
  return (gia_u.frame.x + GI_BUCKET_SIZE - 1u) / GI_BUCKET_SIZE;
}

fn gi_res_bucket_index(texel: vec2<u32>) -> u32 {
  return (texel.x / GI_BUCKET_SIZE) + (texel.y / GI_BUCKET_SIZE) * gi_res_buckets_x();
}

fn gi_res_texel_in_range(p: vec2<u32>) -> bool {
  let s = gi_res_frame_size();
  return p.x < s.x && p.y < s.y;
}

fn gi_res_store_primary(
  coord: vec2<i32>,
  p: vec3<f32>,
  distance: f32,
  key: u32,
  token: u32,
  flags: u32,
  n: vec3<f32>,
) {
  textureStore(gia_surface_out, coord, vec4<f32>(p, distance));
  textureStore(gia_key_out, coord, vec4<u32>(key, token, flags, gi_pack_dir(n)));
}

fn gi_res_store_miss(coord: vec2<i32>) {
  textureStore(gia_surface_out, coord, vec4<f32>(0.0, 0.0, 0.0, GI_T_MAX));
  textureStore(gia_key_out, coord, vec4<u32>(0u));
}

fn gi_res_load_surface(p: vec2<u32>) -> vec4<f32> {
  return textureLoad(gia_surface, vec2<i32>(p), 0);
}

fn gi_res_load_key(p: vec2<u32>) -> vec4<u32> {
  return textureLoad(gia_key, vec2<i32>(p), 0);
}

// 0..5 as +X, -X, +Y, -Y, +Z, -Z (the objectNormals order of the Caelix reference).
fn gi_axis_index(n: vec3<f32>) -> u32 {
  let a = abs(n);
  if (a.x >= a.y && a.x >= a.z) {
    return select(1u, 0u, n.x >= 0.0);
  }
  if (a.y >= a.z) {
    return select(3u, 2u, n.y >= 0.0);
  }
  return select(5u, 4u, n.z >= 0.0);
}

fn gi_axis_normal(axis: u32) -> vec3<f32> {
  if (axis == 0u) { return vec3<f32>(1.0, 0.0, 0.0); }
  if (axis == 1u) { return vec3<f32>(-1.0, 0.0, 0.0); }
  if (axis == 2u) { return vec3<f32>(0.0, 1.0, 0.0); }
  if (axis == 3u) { return vec3<f32>(0.0, -1.0, 0.0); }
  if (axis == 4u) { return vec3<f32>(0.0, 0.0, 1.0); }
  return vec3<f32>(0.0, 0.0, -1.0);
}

fn gi_pack_radiance(rgb: vec3<f32>) -> u32 {
  let c = vec3<f32>(
    select(0.0, min(max(rgb.r, 0.0), 65408.0), rgb.r > 0.0),
    select(0.0, min(max(rgb.g, 0.0), 65408.0), rgb.g > 0.0),
    select(0.0, min(max(rgb.b, 0.0), 65408.0), rgb.b > 0.0),
  );
  let maxc = max(c.r, max(c.g, c.b));
  if (maxc < 1.0e-9) {
    return 0u;
  }
  var exp_shared = max(-16, i32(floor(log2(maxc)))) + 1 + 15;
  var denom = exp2(f32(exp_shared - 15 - 9));
  if (u32(floor(maxc / denom + 0.5)) >= 512u) {
    denom = denom * 2.0;
    exp_shared = exp_shared + 1;
  }
  exp_shared = clamp(exp_shared, 0, 31);
  let m = min(vec3<u32>(floor(c / denom + vec3<f32>(0.5))), vec3<u32>(511u));
  return m.r | (m.g << 9u) | (m.b << 18u) | (u32(exp_shared) << 27u);
}

fn gi_unpack_radiance(packed: u32) -> vec3<f32> {
  let scale = exp2(f32(i32(packed >> 27u) - 15 - 9));
  return vec3<f32>(f32(packed & 511u), f32((packed >> 9u) & 511u), f32((packed >> 18u) & 511u))
    * scale;
}

fn gi_luminance(c: vec3<f32>) -> f32 {
  return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn gi_oct_encode(n: vec3<f32>) -> vec2<f32> {
  let l = abs(n.x) + abs(n.y) + abs(n.z);
  var o = n.xy / max(l, 1e-9);
  if (n.z < 0.0) {
    o = (vec2<f32>(1.0, 1.0) - abs(o.yx))
      * select(vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), o >= vec2<f32>(0.0, 0.0));
  }
  return o;
}

fn gi_oct_decode(o: vec2<f32>) -> vec3<f32> {
  var n = vec3<f32>(o.x, o.y, 1.0 - abs(o.x) - abs(o.y));
  let t = max(-n.z, 0.0);
  n = vec3<f32>(
    n.x + select(t, -t, n.x >= 0.0),
    n.y + select(t, -t, n.y >= 0.0),
    n.z,
  );
  return normalize(n);
}

fn gi_pack_oct16(n: vec3<f32>) -> u32 {
  let half = vec2<f32>(0.5, 0.5);
  let o = clamp(gi_oct_encode(normalize(n)) * half + half, vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0));
  let q = vec2<u32>(o * 65535.0 + half);
  return q.x | (q.y << 16u);
}

fn gi_oct16_to_12(oct16: u32) -> u32 {
  return (((oct16 & 0xFFFFu) >> 4u) << 8u) | (((oct16 >> 16u) >> 4u) << 20u);
}

fn gi_unpack_oct12(normals: u32) -> vec3<f32> {
  let o = vec2<f32>(f32((normals >> 8u) & 0xFFFu), f32((normals >> 20u) & 0xFFFu)) / 4095.0;
  return gi_oct_decode(o * 2.0 - vec2<f32>(1.0, 1.0));
}

fn gi_pack_f16_pair(lo: f32, hi: f32) -> u32 {
  return pack2x16float(vec2<f32>(lo, hi));
}

fn gi_unpack_f16_pair(v: u32) -> vec2<f32> {
  return unpack2x16float(v);
}

fn gi_pcg(v: u32) -> u32 {
  let state = v * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

fn gi_rand_seed(pixel: vec2<u32>, frame: u32, kernel_id: u32) -> u32 {
  return gi_pcg(pixel.x + gi_pcg(pixel.y + gi_pcg(frame ^ (kernel_id * 0x9E3779B9u))));
}

fn gi_rand(state: ptr<function, u32>) -> f32 {
  *state = gi_pcg(*state);
  return f32(*state >> 8u) * (1.0 / 16777216.0);
}

fn gi_res_rand2(state: ptr<function, u32>) -> vec2<f32> {
  let x = gi_rand(state);
  let y = gi_rand(state);
  return vec2<f32>(x, y);
}

fn gi_next_pow2(x: u32) -> u32 {
  if (x <= 1u) { return 1u; }
  return 1u << (firstLeadingBit(x - 1u) + 1u);
}

// Object space of this engine: the grid's local voxel space. The current transform lives in
// grid_descs, so a stored local position follows a moving instance.
fn gi_local_pos(g: Grid, p_world: vec3<f32>) -> vec3<f32> {
  let d = p_world - g.pos;
  return vec3<f32>(dot(d, g.col0), dot(d, g.col1), dot(d, g.col2)) / max(g.scale, 1e-6);
}

fn gi_world_pos(g: Grid, local: vec3<f32>) -> vec3<f32> {
  return g.pos + (local.x * g.col0 + local.y * g.col1 + local.z * g.col2) * g.scale;
}

fn gi_res_camera_pos() -> vec3<f32> {
  return view_u.cam_pos_voxel.xyz;
}

// World direction of the primary ray through a GI texel, the same mapping gi_main uses.
fn gi_camera_ray_direction(pixel: vec2<u32>) -> vec3<f32> {
  let size = vec2<f32>(f32(gia_u.frame.x), f32(gia_u.frame.y));
  let px = (vec2<f32>(pixel) + vec2<f32>(0.5)) / size;
  let n = vec2<f32>(px.x * 2.0 - 1.0, 1.0 - px.y * 2.0);
  let nh = view_u.inv_view_proj * vec4<f32>(n, 0.0, 1.0);
  let fh = view_u.inv_view_proj * vec4<f32>(n, 1.0, 1.0);
  return normalize(fh.xyz / fh.w - nh.xyz / nh.w);
}

// Projects a world point to the GI texel whose primary ray passes closest to it this frame.
fn gi_project_to_texel(world_position: vec3<f32>, texel: ptr<function, vec2<i32>>) -> bool {
  *texel = vec2<i32>(0);
  let clip = view_u.view_proj * vec4<f32>(world_position, 1.0);
  if (clip.w <= 1e-6) {
    return false;
  }
  let ndc = clip.xyz / clip.w;
  if (any(ndc.xy < vec2<f32>(-1.0)) || any(ndc.xy > vec2<f32>(1.0))) {
    return false;
  }
  let size = vec2<f32>(f32(gia_u.frame.x), f32(gia_u.frame.y));
  let q = vec2<i32>(
    i32(floor((ndc.x * 0.5 + 0.5) * size.x)),
    i32(floor((0.5 - ndc.y * 0.5) * size.y)),
  );
  *texel = q;
  return q.x >= 0 && q.y >= 0 && q.x < i32(gia_u.frame.x) && q.y < i32(gia_u.frame.y);
}

fn gi_surface_world_normal(p: vec2<u32>) -> vec3<f32> {
  return gi_unpack_dir(gi_res_load_key(p).w);
}

// ---------------------------------------------------------------------------------------------
// GiPreparePixels: 3x3 virtual-distance range and hash set per pixel.
// ---------------------------------------------------------------------------------------------
fn gi_r_prepare_pixels(gid: vec3<u32>) {
  let p = gid.xy;
  if (!gi_res_texel_in_range(p)) {
    return;
  }

  let centre_distance = gi_res_load_surface(p).w;
  var min_distance = centre_distance;
  var max_distance = centre_distance;
  let centre_hash = gi_res_load_key(p).x & 0xFFFFu;
  var bloom = 0u;

  for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
    for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
      if (dx == 0 && dy == 0) {
        continue;
      }
      let q = vec2<i32>(p) + vec2<i32>(dx, dy);
      if (q.x < 0 || q.y < 0 || q.x >= i32(gia_u.frame.x) || q.y >= i32(gia_u.frame.y)) {
        continue;
      }
      let d = gi_res_load_surface(vec2<u32>(q)).w;
      min_distance = min(min_distance, d);
      max_distance = max(max_distance, d);
      let h = gi_res_load_key(vec2<u32>(q)).x & 0xFFFFu;
      bloom = bloom | (1u << (h & 15u));
    }
  }

  gia_pixels[p.x + p.y * gia_u.frame.x] =
    GiPixelInfo(gi_pack_f16_pair(min_distance, max_distance), centre_hash | (bloom << 16u));
}

// ---------------------------------------------------------------------------------------------
// GiPrepareBuckets: clear the bucket counters and the per-frame count slot, then build the
// normal mask and distance range.
// ---------------------------------------------------------------------------------------------
fn gi_r_prepare_buckets(gid: vec3<u32>) {
  if (gid.x == 0u) {
    let slot = gi_res_frame_slot();
    atomicStore(&gia_counts[GI_COUNT_FRAME_BASE + 2u * slot], 0u);
    atomicStore(&gia_counts[GI_COUNT_FRAME_BASE + 2u * slot + 1u], 0u);
  }

  let b = gid.x;
  if (b >= gia_u.limits.z) {
    return;
  }

  let buckets_x = gi_res_buckets_x();
  let origin = vec2<u32>(b % buckets_x, b / buckets_x) * GI_BUCKET_SIZE;

  var normal_mask = 0u;
  var min_distance = GI_T_MAX;
  var max_distance = 0.0;

  for (var y: u32 = 0u; y < GI_BUCKET_SIZE; y = y + 1u) {
    for (var x: u32 = 0u; x < GI_BUCKET_SIZE; x = x + 1u) {
      let p = origin + vec2<u32>(x, y);
      if (!gi_res_texel_in_range(p)) {
        continue;
      }
      let key = gi_res_load_key(p);
      if ((key.z & GI_KEY_FLAG_HIT) == 0u) {
        continue;
      }
      let n = gi_unpack_dir(key.w);
      normal_mask = normal_mask | (1u << gi_axis_index(n));
      let d = gi_res_load_surface(p).w;
      min_distance = min(min_distance, d);
      max_distance = max(max_distance, d);
    }
  }

  atomicStore(&gia_buckets[b].lit_count, 0u);
  atomicStore(&gia_buckets[b].unlit_count, 0u);
  gia_buckets[b].dist_range = gi_pack_f16_pair(min_distance, max_distance);
  gia_buckets[b].packed = normal_mask << 4u;
  gia_buckets[b].lit_ratio = 0.0;
  gia_buckets[b].max_luminance = 0.0;
  gia_buckets[b].reserved0 = 0u;
  gia_buckets[b].reserved1 = 0u;
}

// ---------------------------------------------------------------------------------------------
// GiGenerateSamples: one new cosine-distributed path sample per eligible pixel.
// ---------------------------------------------------------------------------------------------

// The engine shades in units of outgoing radiance divided by pi (brdf_reflected has no 1/pi);
// the resampler stores incident radiance, so the traced value is multiplied by pi once here.
fn gi_trace_sample(
  p: vec2<u32>,
  key: vec4<u32>,
  seed: u32,
  sample: ptr<function, GiLitSample>,
  is_lit: ptr<function, bool>,
  is_unlit: ptr<function, bool>,
  own: ptr<function, u32>,
) {
  *is_lit = false;
  *is_unlit = false;
  *own = 0u;
  *sample = GiLitSample(0u, 0u, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0u, 0.0, 0.0, 0.0, 0u, 0u);

  let surface = gi_res_load_surface(p);
  let pw = surface.xyz;
  let n = gi_unpack_dir(key.w);
  let slot = key.x >> 16u;
  let g = make_grid(slot);
  let local = gi_local_pos(g, pw);
  (*sample).key = key.x;
  (*sample).token = key.y;
  (*sample).ox = local.x;
  (*sample).oy = local.y;
  (*sample).oz = local.z;

  var rng = seed;
  let dir = gi_cos_hemisphere(n, gi_res_rand2(&rng));
  let org = pw + n * (SHADOW_SURFACE_EPS * g.scale * 2.0);

  let sun_dir = light_u.lights[0].kind_pos_dir.yzw;
  let sun_c = light_u.lights[0].color_intensity.xyz * light_u.lights[0].color_intensity.w;
  let sun_bounce = gi_u.params.x;
  let epoch = gi_u.seq.y;
  let b2_mult = gi_u.misc.w;
  let sky_p = clamp(gia_u.settings.x, 1e-4, 1.0);

  var c = vec3<f32>(0.0);
  var secondary = dir;
  var secondary_n = vec3<f32>(0.0, 0.0, 1.0);
  var is_sky = true;

  var tint = vec3<f32>(1.0);
  let h = world_trace_medium_scene(org, dir, min(GI_T_MAX, GI_REACH_T), false, false, &tint);
  if (!h.hit) {
    if (gi_rand(&rng) < sky_p) {
      c = sky_rgb() * GI_SKY_RADIANCE_SCALE * tint / sky_p;
    }
  } else {
    is_sky = false;
    secondary = h.point;
    let hg = make_grid(u32(h.obj_id) + 1u);
    let hn = voxel_normal_fast(hg, h.voxel, h.face_id, h.n);
    secondary_n = select(hn, -hn, dot(hn, dir) > 0.0);
    let k0 = gi_key0(h.voxel, h.obj_id);
    let k1 = gi_key1(h.voxel, h.face_id, h.obj_id);
    let key_ok = gi_key_ok(h.voxel, h.obj_id);
    var l = gi_face_shade_cached(h, org, sun_dir, sun_c, sun_bounce, epoch, k0, k1, key_ok);
    if (b2_mult > 0.0 && gi_rand(&rng) < 1.0 / max(b2_mult, 1.0)) {
      l = l + gi_bounce2(h, org, sun_dir, sun_c, sun_bounce, epoch, seed, b2_mult);
    }
    c = l * tint * GI_PI;
  }

  *own = gi_pack_radiance(c);

  let threshold = max(gia_u.settings.y, 1e-6);
  let s = gi_luminance(c);
  if (s < threshold) {
    let keep = max(s, 0.05 * threshold) / threshold;
    c = c * (1.0 / keep);
    if (gi_rand(&rng) > keep) {
      c = vec3<f32>(0.0);
    }
  }

  (*sample).primary_normal = gi_axis_index(n) | select(0u, GI_SAMPLE_SKY_FLAG, is_sky);
  (*sample).sx = secondary.x;
  (*sample).sy = secondary.y;
  (*sample).sz = secondary.z;
  (*sample).secondary_normal = select(0u, gi_pack_oct16(secondary_n), !is_sky);
  (*sample).radiance = gi_pack_radiance(c);

  if ((*sample).radiance != 0u) {
    *is_lit = true;
  } else if (gi_rand(&rng) < 1.0 / f32(max(gia_u.settings.z, 1.0))) {
    *is_unlit = true;
  }
}

fn gi_r_generate_samples(gid: vec3<u32>) {
  let p = gid.xy;
  if (!gi_res_texel_in_range(p)) {
    return;
  }

  let key = gi_res_load_key(p);
  let eligible = (key.z & (GI_KEY_FLAG_HIT | GI_KEY_FLAG_DIFFUSE))
    == (GI_KEY_FLAG_HIT | GI_KEY_FLAG_DIFFUSE);
  let slot = gi_res_frame_slot();
  let frame = gia_u.frame.w;
  let seed = gi_rand_seed(p, frame, 2u);

  var sample = GiLitSample(0u, 0u, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0u, 0.0, 0.0, 0.0, 0u, 0u);
  var is_lit = false;
  var is_unlit = false;
  var own = 0u;
  if (eligible) {
    gi_trace_sample(p, key, seed, &sample, &is_lit, &is_unlit, &own);
  }

  textureStore(gia_own_out, vec2<i32>(p), vec4<u32>(own));

  if (u32(gia_u.settings4.x) == 1u) {
    textureStore(gi_out, vec2<i32>(p), vec4<f32>(gi_unpack_radiance(own), 1.0));
  }

  if (is_lit) {
    let k = atomicAdd(&gia_counts[GI_COUNT_FRAME_BASE + 2u * slot], 1u);
    let index = (atomicLoad(&gia_counts[GI_COUNT_LIT_HEAD]) + k) % max(gia_u.limits.x, 1u);
    gia_lit[index] = sample;
  } else if (is_unlit) {
    let k = atomicAdd(&gia_counts[GI_COUNT_FRAME_BASE + 2u * slot + 1u], 1u);
    let index = (atomicLoad(&gia_counts[GI_COUNT_UNLIT_HEAD]) + k) % max(gia_u.limits.y, 1u);
    gia_unlit[index] = GiUnlitSample(
      sample.key,
      sample.token,
      sample.ox,
      sample.oy,
      sample.oz,
      sample.vx,
      sample.vy,
      sample.vz,
      sample.primary_normal,
    );
  }
}

// ---------------------------------------------------------------------------------------------
// GiPrepareHistory: valid sample counts, ring starts and indirect dispatch arguments.
// ---------------------------------------------------------------------------------------------

fn gi_prepare_ring(ring: u32, capacity: u32, slot: u32) {
  let head = (atomicLoad(&gia_counts[GI_COUNT_LIT_HEAD + ring])
    + atomicLoad(&gia_counts[GI_COUNT_FRAME_BASE + 2u * slot + ring])) % max(capacity, 1u);
  atomicStore(&gia_counts[GI_COUNT_LIT_HEAD + ring], head);

  var total = 0u;
  let max_frames = u32(max(gia_u.settings.w, 0.0));
  for (var i: u32 = 0u; i < max_frames; i = i + 1u) {
    let frame_slot = (slot + GI_HISTORY_SLOTS - i) % GI_HISTORY_SLOTS;
    let n = atomicLoad(&gia_counts[GI_COUNT_FRAME_BASE + 2u * frame_slot + ring]);
    if (total + n > capacity) {
      break;
    }
    total = total + n;
  }

  atomicStore(&gia_counts[GI_COUNT_LIT_VALID + ring], total);
  atomicStore(&gia_counts[GI_COUNT_LIT_START + ring], (head + capacity - total) % max(capacity, 1u));
  atomicStore(&gia_counts[GI_COUNT_LIT_COPRIME + ring], gi_pcg(gia_u.frame.w * 2u + ring) | 1u);

  let groups = select(0u, gi_next_pow2((total + 63u) / 64u), total > 0u);
  gia_args[3u * ring] = min(groups, GI_MAX_GROUPS_X);
  gia_args[3u * ring + 1u] = max(1u, groups / GI_MAX_GROUPS_X);
  gia_args[3u * ring + 2u] = 1u;
}

fn gi_r_prepare_history() {
  let slot = gi_res_frame_slot();
  gi_prepare_ring(0u, max(gia_u.limits.x, 1u), slot);
  gi_prepare_ring(1u, max(gia_u.limits.y, 1u), slot);
}

// ---------------------------------------------------------------------------------------------
// GiReprojectLit / GiReprojectUnlit / GiRefineBuckets
// ---------------------------------------------------------------------------------------------

fn gi_shuffled_sample_index(
  ring: u32,
  group: u32,
  lane: u32,
  index: ptr<function, u32>,
) -> bool {
  *index = 0u;
  let valid = atomicLoad(&gia_counts[GI_COUNT_LIT_VALID + ring]);
  if (valid == 0u) {
    return false;
  }
  let groups = gi_next_pow2((valid + 63u) / 64u);
  if (group >= groups) {
    return false;
  }
  let coprime = atomicLoad(&gia_counts[GI_COUNT_LIT_COPRIME + ring]);
  let shuffled = ((coprime * group + gia_u.frame.w) % groups) * 64u + lane;
  if (shuffled >= valid) {
    return false;
  }
  *index = shuffled;
  return true;
}

// Moves a stored sample's primary surface to this frame and validates it at its landing pixel.
fn gi_reproject_to_bucket(
  key: u32,
  token: u32,
  object_position: vec3<f32>,
  virtual_offset: vec3<f32>,
  world: ptr<function, vec3<f32>>,
  bucket: ptr<function, u32>,
) -> bool {
  *world = vec3<f32>(0.0);
  *bucket = 0u;

  if (token == 0u) {
    return false;
  }

  let slot = key >> 16u;
  let obj = i32(slot) - 1;
  let voxel = vec3<i32>(floor(object_position));
  if ((gi_pose_salt(obj) ^ gi_local_rev(voxel, obj)) != token) {
    return false;
  }

  let g = make_grid(slot);
  let primary = gi_world_pos(g, object_position);
  let virtual_point = primary + virtual_offset;

  var texel = vec2<i32>(0);
  if (!gi_project_to_texel(virtual_point, &texel)) {
    return false;
  }

  let info = gia_pixels[u32(texel.y) * gia_u.frame.x + u32(texel.x)];
  let range = gi_unpack_f16_pair(info.dist_range);
  let distance = length(virtual_point - gi_res_camera_pos());
  if (distance < range.x * (1022.0 / 1024.0) || distance > range.y * (1026.0 / 1024.0)) {
    return false;
  }

  let hash16 = key & 0xFFFFu;
  let centre_hash = info.hashes & 0xFFFFu;
  let bloom = info.hashes >> 16u;
  if (hash16 != centre_hash && (bloom & (1u << (hash16 & 15u))) == 0u) {
    return false;
  }

  *world = primary;
  *bucket = gi_res_bucket_index(vec2<u32>(texel));
  return true;
}

fn gi_r_reproject_lit(group: u32, lane: u32) {
  var i = 0u;
  if (!gi_shuffled_sample_index(0u, group, lane, &i)) {
    return;
  }

  let start = atomicLoad(&gia_counts[GI_COUNT_LIT_START]);
  let sample = gia_lit[(start + i) % max(gia_u.limits.x, 1u)];

  var world = vec3<f32>(0.0);
  var bucket = 0u;
  if (!gi_reproject_to_bucket(
    sample.key,
    sample.token,
    vec3<f32>(sample.ox, sample.oy, sample.oz),
    vec3<f32>(sample.vx, sample.vy, sample.vz),
    &world,
    &bucket,
  )) {
    return;
  }

  let k = atomicAdd(&gia_buckets[bucket].lit_count, 1u);
  if (k < GI_REFINED_PER_BUCKET) {
    let is_sky = (sample.primary_normal & GI_SAMPLE_SKY_FLAG) != 0u;
    gia_refined[bucket * GI_REFINED_PER_BUCKET + k] = GiBucketSample(
      world.x,
      world.y,
      world.z,
      (sample.primary_normal & 7u)
        | select(0u, GI_BUCKET_SKY_FLAG, is_sky)
        | gi_oct16_to_12(sample.secondary_normal),
      sample.sx,
      sample.sy,
      sample.sz,
      sample.radiance,
    );
  }
}

fn gi_r_reproject_unlit(group: u32, lane: u32) {
  var i = 0u;
  if (!gi_shuffled_sample_index(1u, group, lane, &i)) {
    return;
  }

  let start = atomicLoad(&gia_counts[GI_COUNT_UNLIT_START]);
  let sample = gia_unlit[(start + i) % max(gia_u.limits.y, 1u)];

  var world = vec3<f32>(0.0);
  var bucket = 0u;
  if (!gi_reproject_to_bucket(
    sample.key,
    sample.token,
    vec3<f32>(sample.ox, sample.oy, sample.oz),
    vec3<f32>(sample.vx, sample.vy, sample.vz),
    &world,
    &bucket,
  )) {
    return;
  }

  atomicAdd(&gia_buckets[bucket].unlit_count, u32(max(gia_u.settings.z, 1.0)));
}

fn gi_r_refine_buckets(gid: vec3<u32>) {
  let b = gid.x;
  if (b >= gia_u.limits.z) {
    return;
  }

  let lit_count = atomicLoad(&gia_buckets[b].lit_count);
  let unlit_count = atomicLoad(&gia_buckets[b].unlit_count);
  let lit = min(lit_count, GI_REFINED_PER_BUCKET);
  let unlit_effective = f32(unlit_count) * (f32(lit) / f32(max(lit_count, 1u)));

  var max_luminance = 0.0;
  for (var i: u32 = 0u; i < lit; i = i + 1u) {
    max_luminance = max(
      max_luminance,
      gi_luminance(gi_unpack_radiance(gia_refined[b * GI_REFINED_PER_BUCKET + i].radiance)),
    );
  }

  var rng = gi_rand_seed(vec2<u32>(b, 0u), gia_u.frame.w, GI_RAND_REFINE);
  let leveling = u32(gia_u.settings3.x) != 0u;
  var extra_unlit = 0u;
  var stored = 0u;
  for (var j: u32 = 0u; j < lit; j = j + 1u) {
    var s = gia_refined[b * GI_REFINED_PER_BUCKET + j];
    let radiance = gi_unpack_radiance(s.radiance);
    let r = select(1.0, max_luminance / max(gi_luminance(radiance), 1e-6), leveling);
    if (gi_rand(&rng) * r > 1.0) {
      extra_unlit = extra_unlit + 1u;
      continue;
    }
    if (stored < GI_COMPACT_PER_BUCKET) {
      s.radiance = gi_pack_radiance(radiance * r);
      gia_compact[b * GI_COMPACT_PER_BUCKET + stored] = s;
      stored = stored + 1u;
    }
  }

  let kept_lit = f32(lit - extra_unlit);
  let lit_ratio = kept_lit / max(1.0, kept_lit + unlit_effective + f32(extra_unlit));
  if (f32(lit) + unlit_effective < gia_u.settings3.y) {
    stored = 0u;
  }

  let normal_mask = (gia_buckets[b].packed >> 4u) & 63u;
  let total_comp = min(7u, (lit_count + unlit_count) / 64u);
  gia_buckets[b].packed = stored | (normal_mask << 4u) | (total_comp << 10u);
  gia_buckets[b].lit_ratio = lit_ratio;
  gia_buckets[b].max_luminance = max_luminance;
}

fn gi_r_reproject_lit_entry(gid: vec3<u32>, lane: u32) {
  gi_r_reproject_lit(gid.x + gid.y * GI_MAX_GROUPS_X, lane);
}

fn gi_r_reproject_unlit_entry(gid: vec3<u32>, lane: u32) {
  gi_r_reproject_unlit(gid.x + gid.y * GI_MAX_GROUPS_X, lane);
}

fn gi_r_debug_buckets(gid: vec3<u32>) {
  let p = gid.xy;
  if (!gi_res_texel_in_range(p)) {
    return;
  }
  let b = gi_res_bucket_index(p);
  let stored = f32(gia_buckets[b].packed & 15u) / f32(GI_COMPACT_PER_BUCKET);
  textureStore(
    gi_out,
    vec2<i32>(p),
    vec4<f32>(stored, gia_buckets[b].lit_ratio, clamp(gia_buckets[b].max_luminance, 0.0, 4.0) * 0.25, 1.0),
  );
}
