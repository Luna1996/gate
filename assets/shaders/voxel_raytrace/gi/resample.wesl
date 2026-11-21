// NAADF-style GI resampler: per-pixel bands, 8x8 buckets, lit/unlit sample rings.
// Layouts and kernel order follow the Caelix GI reference; strides are mirrored by
// gate-render/src/wesl_consts.rs and the buffers in brickmap/dda.rs.
import package::bindings::{view_u};
import package::brickmap::{Grid};
import package::gi::common::{gi_pack_dir, gi_unpack_dir};
import package::gi::consts::{GI_T_MAX};

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
  lit_count: u32,
  unlit_count: u32,
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

fn gi_pack_oct16(n: vec3<f32>) -> u32 {
  let o = clamp(gi_pack_dir(normalize(n)) * 0.5 + vec2<f32>(0.5), vec2<f32>(0.0), vec2<f32>(1.0));
  let q = vec2<u32>(o * 65535.0 + vec2<f32>(0.5));
  return q.x | (q.y << 16u);
}

fn gi_oct16_to_12(oct16: u32) -> u32 {
  return (((oct16 & 0xFFFFu) >> 4u) << 8u) | (((oct16 >> 16u) >> 4u) << 20u);
}

fn gi_unpack_oct12(normals: u32) -> vec3<f32> {
  let o = vec2<f32>(f32((normals >> 8u) & 0xFFFu), f32((normals >> 20u) & 0xFFFu)) / 4095.0;
  return normalize(gi_unpack_dir(o * 2.0 - vec2<f32>(1.0)));
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

fn gi_rand2(state: ptr<function, u32>) -> vec2<f32> {
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

  gia_buckets[b] = GiBucketInfo(
    0u,
    0u,
    gi_pack_f16_pair(min_distance, max_distance),
    normal_mask << 4u,
    0.0,
    0.0,
    0u,
    0u,
  );
}
