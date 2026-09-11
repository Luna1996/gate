// ============================================================================
// DDA Compute Shader：全屏逐像素层次栈式 mask DDA（Douglas devlog #17 / 64-tree）
//
// 遍历结构（Rust 参考 cpu_reference_dda_ray_tree，dda.rs 逐字镜像）：
//   chunk 间：256³ 一格 A&W，窗口 entry=0 的空 chunk 只 1 load 直接跨越；
//   chunk 内：4 层栈帧（256→64→16→4→1），节点 mask 一次 load 进寄存器，
//             子块步进只查 bit（零 load）；bit=1 分裂才压栈下钻，
//             bit=0 uniform 子块整格跳过/整格命中。
// sample_brickmap（树点查询）仅保留给 implicit normals 6 邻域 occupancy。
//
// BG0（与 Rust DdaViewUniform 144B + out tex rgba8unorm 对应）：
//   @group(0) @binding(0) = out storage tex（DDA 写入 rgba8unorm，linear RGB）
//   @group(0) @binding(1) = uniform DdaViewUniform（144B 对齐）
//
// BG1（与 Rust GpuBrickMap 资源 1:1 对应）：
//   @group(1) @binding(0) = b_struct: array<u32>（TileIndex + TileBitmaps + CellDirs + NodeStream 定长前缀 + 可变 node 区）
//   @group(1) @binding(1) = b_leaves: array<u32>（P4 重定向：方向可达掩码 LUT，
//                          8 octant × 64 入口格 × 2 u32 = 1024 words = 4KB，
//                          CPU 端 wire::march_mask_lut_words 同布局）
//   @group(1) @binding(2) = b_palette: array<u32>（palette 256 entries × 2 words = 512 words）
//   @group(1) @binding(3) = uniform BrickMapGlobals（scalar 字段 20×u32/i32 = 80B）
//
// BG2（Phase 3 OBJ→Volume 统一：GridDesc 数组，与 Rust GpuBrickMap.grid_descs_buf 1:1）：
//   @group(2) @binding(0) = grid_descs: array<GridDesc>（144B/entry，主世界 + 物体统一描述符）
//   shader `dda_main` 用 `arrayLength(&grid_descs)` 取 volume 数，遍历 trace_grid 无 kind 分支。
//
// BG3（R3-18 直光层：光照光池，与 Rust LightPoolUniform 448B 1:1）：
//   @group(3) @binding(0) = uniform LightPool（LightGlobals 48B + 8×LightDesc 384B + sky_color 16B）
//   数据源 = assets/lighting/*.ron 主题（main world LightingTheme → ExtractResource → build_light_pool）
//
// 顶部常量与 Rust `brickmap::dda::wgsl_consts` 完全一致（单测 TR-2.1 assert_eq 防漂移）。
// BrickMap 五步寻址链严格对应 Rust `gate-render/src/brickmap/view.rs::get_voxel`（逐段注释 L 号）。
// Slot 打包规则对应 Rust `gate-render/src/brickmap/wire.rs::encode_slot/unpack_slot_word/pack_palette_entry`。
// trace_grid 统一 DDA（主网格 + 逐物体 GridDesc），OBJ 等价性单测锁死 CPU 侧。
// ============================================================================

// --- 常量区（与 Rust wgsl_consts mod 字节对齐）---
// Douglas Brick Tree：256³ chunk，4³ 分裂因子，4 层（256→64→16→4→1）
const CHUNK_SIZE: u32 = 256u;
const BRICK_FACTOR: u32 = 4u;
const MAX_LEVEL: u32 = 4u;
const NODE_FIXED_WORDS: u32 = 3u;    // mask_lo + mask_hi + palette
// b_struct Region ①：稠密 chunk 窗口（64³ = 262144 字 = 1MB）
const CHUNK_INDEX_CAP: u32 = 64u;
const CHUNK_INDEX_WORDS: u32 = 262144u;  // 64³
const TREE_BASE: u32 = 262144u;          // Region ② 起始


// P2：方向位掩码 LUT。射线方向符号编码为 3-bit 掩码（bit0=x>=0, bit1=y>=0,
// bit2=z>=0，共 8 种），查表一次性得到 side（boundary 计算用）和三轴步进 face id，
// 消除 trace_chunk/trace_grid 热循环内的 select(vec3(0),vec3(1),sign>=0) 与
// select(0u,1u,sign<0) 等运行时方向分支。打包：bits0-2=side, bits8-15=face_x,
// bits16-23=face_y, bits24-31=face_z。face 规则：sign<0 穿对面（x→1, y→3, z→5）。
const DIR_LUT: array<u32, 8u> = array<u32, 8u>(
  0x05030100u, // dir=0 (-,-,-): side=(0,0,0), face=(1,3,5)
  0x05030001u, // dir=1 (+,-,-): side=(1,0,0), face=(0,3,5)
  0x05020102u, // dir=2 (-,+,-): side=(0,1,0), face=(1,2,5)
  0x05020003u, // dir=3 (+,+,-): side=(1,1,0), face=(0,2,5)
  0x04030104u, // dir=4 (-,-,+): side=(0,0,1), face=(1,3,4)
  0x04030005u, // dir=5 (+,-,+): side=(1,0,1), face=(0,3,4)
  0x04020106u, // dir=6 (-,+,+): side=(0,1,1), face=(1,2,4)
  0x04020007u, // dir=7 (+,+,+): side=(1,1,1), face=(0,2,4)
);

// ---- Pure face math helpers (no hashmap dependency) ----
fn face_index_from_normal(n: vec3<f32>) -> u32 {
  let ax = abs(n.x); let ay = abs(n.y); let az = abs(n.z);
  if (ax >= max(ay, az)) { return select(0u, 1u, n.x >= 0.0); }
  if (ay >= az)          { return select(2u, 3u, n.y >= 0.0); }
  return select(4u, 5u, n.z >= 0.0);
}
fn face_normal_from_index(f: u32) -> vec3<f32> {
  if (f == 0u) { return vec3<f32>(-1.0, 0.0, 0.0); }
  if (f == 1u) { return vec3<f32>( 1.0, 0.0, 0.0); }
  if (f == 2u) { return vec3<f32>(0.0, -1.0, 0.0); }
  if (f == 3u) { return vec3<f32>(0.0,  1.0, 0.0); }
  if (f == 4u) { return vec3<f32>(0.0, 0.0, -1.0); }
  if (f == 5u) { return vec3<f32>(0.0, 0.0,  1.0); }
  return vec3<f32>(0.0);
}
fn face_color_from_index(f: u32) -> vec3<f32> {
  if (f == 0u) { return vec3<f32>(1.0, 0.2, 0.2); }
  if (f == 1u) { return vec3<f32>(0.2, 1.0, 0.2); }
  if (f == 2u) { return vec3<f32>(0.2, 0.2, 1.0); }
  if (f == 3u) { return vec3<f32>(1.0, 1.0, 0.2); }
  if (f == 4u) { return vec3<f32>(0.2, 1.0, 1.0); }
  if (f == 5u) { return vec3<f32>(1.0, 0.2, 1.0); }
  return vec3<f32>(1.0);
}

// --- BG0：输出 + 视图 uniform ---
// @binding(0) = out storage write（rgba8unorm，linear RGB；ACES → sRGB 后输出）
// @binding(1) = uniform DdaViewUniform
// @binding(2) = beam depth storage（r32float，低分辨率最近命中 t；beam pass 写，主 pass 读）
@group(0) @binding(0) var out_tex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var beam_depth: texture_storage_2d<r32float, read_write>;

// P3 beam 比例：低分辨率 = 全分辨率 / BEAM_DIV（4 = 1/4 分辨率，像素数 1/16）
const BEAM_DIV: u32 = 4u;

// P5：beam 起点保守回退（世界单位）。邻域 beam 射线命中主射线首命中体素时，
// 允许经侧面/远面入射——两射线穿过同一单位体素，入口 t 差 ≤ 体素视向对角 √3，
// 加 3×3 半宽横向偏移（d_beam 内 t·px_ang ≤ 0.35/BEAM_DIV → 6px ≤ 0.53 体素）
// ≈ 2.3；再留浮点裕量取 4。只多走 4 单位空空间，对比可跳过的 ~d_beam
// （百单位级）近似零成本。旧实现直接以 min t 起步 → 薄壁/剪影处穿墙（用户实测）。
const BEAM_BACKOFF: f32 = 4.0;

struct DdaViewUniform {
  view_proj: mat4x4<f32>,       // 64B：世界→裁剪空间（probe 可视化投影用）
  inv_view_proj: mat4x4<f32>,   // 64B
  cam_pos_voxel: vec4<f32>,      // 16B，w=1
  debug_mode: vec4<f32>,        // 16B：x = 法向可视化，y = face 6 色诊断
  lod: vec4<f32>,               // 16B：x = 像素角大小(rad)，y = LOD 早停开关
  probe_viz_params: vec4<f32>,  // 16B：x = 总探针数，y = 方块边长(px)
}
@group(0) @binding(1) var<uniform> view_u: DdaViewUniform;

// --- BG1：brick map 全部数据 + globals ---
@group(1) @binding(0) var<storage, read> b_struct: array<u32>;
@group(1) @binding(1) var<storage, read> b_leaves: array<u64>;
@group(1) @binding(2) var<storage, read> b_palette: array<u32>;

// BrickMapGlobals scalar mirror（wire.rs 110-130）
struct Globals {
  index_origin_x: i32,
  index_origin_y: i32,
  index_origin_z: i32,
  index_origin_w: i32,
  index_dims_x: u32,
  index_dims_y: u32,
  index_dims_z: u32,
  index_dims_w: u32,
  tile_count: u32,
  node_words: u32,
  node_free_words: u32,
  brick_slabs: u32,
  brick_free: u32,
  rejected_tiles: u32,
  grid_count: u32,
  _pad1: u32,
  _pad2: u32,
  _pad3: u32,
  _pad4: u32,
}
@group(1) @binding(3) var<uniform> g: Globals;

// --- BG2：GridDesc 数组（Phase 3 OBJ→Volume 统一；144B/entry，与 Rust GridDesc 字节一致）---
//   主世界 = grid_descs[0]（identity 变换），物体 = grid_descs[1..N]
//   每个 GridDesc 携带变换、世界 AABB、tree_base（在 b_struct 内绝对字基址）、
//   palette_base（在 b_palette 内绝对字基址）、chunk 窗口 origin/dims。
struct GridDesc {
  pos_scale: vec4<f32>,   // xyz = 位置，w = scale
  rot0: vec4<f32>,        // xyz = 旋转矩阵列 0，w = 0
  rot1: vec4<f32>,        // xyz = 旋转矩阵列 1，w = 0
  rot2: vec4<f32>,        // xyz = 旋转矩阵列 2，w = 0
  aabb_min: vec4<f32>,    // xyz = 世界 AABB min，w = 0
  aabb_max: vec4<f32>,    // xyz = 世界 AABB max，w = 0
  tree_base: u32,         // b_struct 内本 volume 树基址（含 chunk 窗口段）
  tree_depth: u32,        // 4（Douglas Brick Tree 最大分裂深度）
  chunk_count: u32,       // 本 volume 的 chunk 数
  palette_base: u32,      // b_palette 内本 volume palette 基址
  index_origin_x: i32,    // chunk 窗口 origin（chunk 单位）
  index_origin_y: i32,
  index_origin_z: i32,
  _pad0: u32,
  index_dims_x: u32,      // chunk 窗口 dims（chunk 单位）
  index_dims_y: u32,
  index_dims_z: u32,
  _pad1: u32,
}
@group(2) @binding(0) var<storage, read> grid_descs: array<GridDesc>;

// --- BG3：光照光池（R3-18 直光层；与 Rust LightPoolUniform 逐字段镜像，448B）---
// 只用 lights[0] = 方向光（kind=0）；硬阴影方案无点光源/无软阴影（Douglas #02/#17/#23）。
struct LightDesc {
  kind_pos_dir: vec4<f32>,    // x = kind（0=方向光）；yzw = L 轴（指向光，已归一）
  color_intensity: vec4<f32>, // rgb = 线性色，w = 强度
  shape: vec4<f32>,           // reserved（硬阴影：角半径 = 0）
}
struct LightGlobals {
  count: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
  ambient: vec4<f32>,     // rgb = 环境色（线性）
  exposure_pad: vec4<f32>, // x = 曝光系数
}
struct LightPool {
  g: LightGlobals,
  lights: array<LightDesc, 8u>,
  sky_color: vec4<f32>,   // 天空纯色（miss 背景 + sky 环境光共用；Minecraft #78A7FF）
}
@group(3) @binding(0) var<uniform> light_u: LightPool;

// 统一网格上下文——一套 DDA 跑所有网格（主世界 + 物体）
// 由 `make_grid(idx)` 从 `grid_descs[idx]` 构造；携带 tree_base/palette_base/
// index_origin/dims 作为数据源基址，trace_grid → trace_chunk 统一从
// b_struct[tree_base + ...] / b_palette[palette_base + ...] 取数据。
struct Grid {
  w_mn: vec3<f32>,              // 世界 AABB min
  w_mx: vec3<f32>,              // 世界 AABB max
  l_min: vec3<f32>,             // 局部 AABB min（voxel 坐标）= vec3(0.0)
  l_max: vec3<f32>,             // 局部 AABB max（voxel 坐标）= vec3(256.0)
  max_chunk_steps: u32,         // chunk 间 DDA 步数上限
  col0: vec3<f32>,              // 变换矩阵列（旋转）
  col1: vec3<f32>,
  col2: vec3<f32>,
  pos: vec3<f32>,               // 物体位置
  scale: f32,
  tree_base: u32,               // b_struct 内本 volume 树基址（含 chunk 窗口段）
  palette_base: u32,            // b_palette 内本 volume palette 基址
  index_origin: vec3<i32>,      // chunk 窗口 origin（chunk 单位）
  index_dims: vec3<u32>,        // chunk 窗口 dims（chunk 单位）
  obj_id: i32,                  // -1 = 主世界，>=0 = 物体索引（debug/implicit normal 用）
}

// Douglas Brick Tree mask DDA 采样（1:1 复刻 devlog #17/#18）
// voxel → chunk 窗口定位 → DFS 树 4 层 mask 遍历 → palette
// Phase 3 统一：从 Grid 参数读取 tree_base / index_origin / index_dims，
// 而非 BG1 globals（主世界 + 物体走同一路径，b_struct[tree_base + ...] 取树）。
// 仅 implicit normals 6 邻域 occupancy 点查询用；主遍历走层次栈式 trace_chunk。
fn sample_brickmap(g: Grid, voxel: vec3<i32>) -> u32 {
  // ---- chunk 窗口定位 ----
  let m = ((voxel % vec3<i32>(i32(CHUNK_SIZE))) + vec3<i32>(i32(CHUNK_SIZE))) % vec3<i32>(i32(CHUNK_SIZE));
  let chunk_i = (voxel - m) / vec3<i32>(i32(CHUNK_SIZE));
  let origin = g.index_origin;
  let dims = g.index_dims;
  let rel = chunk_i - origin;
  if (any(rel < vec3<i32>(0))) { return 0u; }
  let rel_u = vec3<u32>(rel);
  if (any(rel_u >= dims)) { return 0u; }
  // Region ① chunk 窗口：entry = 本 volume 内树相对字基址 + 1（0 = 无此 chunk）
  let index_addr = g.tree_base + rel_u.x + rel_u.y * CHUNK_INDEX_CAP + rel_u.z * (CHUNK_INDEX_CAP * CHUNK_INDEX_CAP);
  let entry = b_struct[index_addr];
  if (entry == 0u) { return 0u; }
  // 统一 buffer 内绝对字基址 = volume 基址 + entry - 1（entry 编码的是本 volume 内相对地址）
  let chunk_base = g.tree_base + entry - 1u;

  // ---- DFS 4 层 mask 遍历（wire v3：level 3 inline 4 体素/word，level 0-2 紧凑）----
  let local = vec3<u32>(m);  // chunk 内 voxel 坐标 0..255
  var node_addr = chunk_base;
  for (var level = 0u; level < MAX_LEVEL; level = level + 1u) {
    let mask_lo = b_struct[node_addr];
    let mask_hi = b_struct[node_addr + 1u];
    let palette = b_struct[node_addr + 2u] & 0xFFu;

    // 该层 4³ 子块索引：shift = 8 - (level+1)*2（level 0:>>6, 1:>>4, 2:>>2, 3:>>0）
    let shift = 8u - (level + 1u) * 2u;
    let cx = (local.x >> shift) & 3u;
    let cy = (local.y >> shift) & 3u;
    let cz = (local.z >> shift) & 3u;
    let child_idx = cz * 16u + cy * 4u + cx;

    // mask 64-bit 拆两个 u32
    var mask_word: u32 = mask_lo;
    var bit_in_word: u32 = child_idx;
    if (child_idx >= 32u) {
      mask_word = mask_hi;
      bit_in_word = child_idx - 32u;
    }
    let bit = 1u << bit_in_word;
    if ((mask_word & bit) == 0u) {
      return palette;  // uniform 子块（palette=0 = AIR）
    }

    if (level == 3u) {
      // wire v3：level 3 inline palette，4 体素/word = b_struct[node + 3 + (child_idx >> 2)]
      // palette = 该 word 的第 (child_idx & 3) 字节
      let w = b_struct[node_addr + NODE_FIXED_WORDS + (child_idx >> 2u)];
      return (w >> ((child_idx & 3u) * 8u)) & 0xFFu;
    }

    // level 0-2：紧凑 popcount 定位 child offset
    var popcount_below: u32;
    if (child_idx < 32u) {
      popcount_below = countOneBits(mask_lo & ((1u << bit_in_word) - 1u));
    } else {
      popcount_below = countOneBits(mask_lo) + countOneBits(mask_hi & ((1u << bit_in_word) - 1u));
    }
    node_addr = chunk_base + b_struct[node_addr + NODE_FIXED_WORDS + popcount_below];
  }
  return b_struct[node_addr + 2u] & 0xFFu;  // fallback（wire v2 下不可达）
}


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
  let pal = b_struct[node + 2u] & 0xFFu;
  var seen: u32 = 3u;
  for (var idx = 0u; idx < 64u; idx = idx + 1u) {
    let in_hi = idx >= 32u;
    let word = select(mask_lo, mask_hi, in_hi);
    let bit_in = select(idx, idx - 32u, in_hi);
    var vp = pal;
    if (((word >> bit_in) & 1u) != 0u) {
      let w = b_struct[node + NODE_FIXED_WORDS + (idx >> 2u)];
      vp = (w >> ((idx & 3u) * 8u)) & 0xFFu;
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
  let pal = b_struct[node + 2u] & 0xFFu;
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
  let pal = b_struct[node + 2u] & 0xFFu;
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
    let pal = b_struct[node + 2u] & 0xFFu;
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
        return select(1u, 0u, (b_struct[child + 2u] & 0xFFu) == 0u);
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

fn ddgi_leaf16(g: Grid, cmin: vec3<i32>, center: vec3<f32>) -> vec4<f32> {
  var best_d2 = 1e30;
  var best_p = vec3<f32>(0.0);
  var found = 0.0;
  var mixed: array<vec3<i32>, 64>;
  var nmix = 0u;
  for (var k = 0; k < 4; k = k + 1) {
    for (var j = 0; j < 4; j = j + 1) {
      for (var i = 0; i < 4; i = i + 1) {
        let sub = cmin + vec3<i32>(i, j, k) * 4;
        let st = ddgi_brick_state(g, sub, 3u);
        if (st == 0u) {
          let p = vec3<f32>(sub) + 2.0;
          let d2 = dot(p - center, p - center);
          if (d2 < best_d2) { best_d2 = d2; best_p = p; found = 1.0; }
        } else if (st == 2u && nmix < 64u) {
          mixed[nmix] = sub;
          nmix = nmix + 1u;
        }
      }
    }
  }
  if (found > 0.5) { return vec4<f32>(best_p, 1.0); }
  for (var m = 0u; m < nmix; m = m + 1u) {
    let par = mixed[m];
    for (var k = 0; k < 4; k = k + 1) {
      for (var j = 0; j < 4; j = j + 1) {
        for (var i = 0; i < 4; i = i + 1) {
          let v = par + vec3<i32>(i, j, k);
          if (sample_brickmap(g, v) == 0u) {
            let p = vec3<f32>(v) + 0.5;
            let d2 = dot(p - center, p - center);
            if (d2 < best_d2) { best_d2 = d2; best_p = p; found = 1.0; }
          }
        }
      }
    }
  }
  return vec4<f32>(best_p, found);
}

// ============================================================================

// 层级命中记录：t 为 ro 系绝对 t；face_id 0..5 = ±xyz 六面（命中面法线索引）。
//   pre-check 命中（射线起点在固体 leaf 内，相机在体内 UB）：face_id 由调用方
//   用 normalize(-rd) 反推（首 chunk entry_face）。
// voxel：命中固体体素 chunk 局部 voxel 整数坐标——DDA 步进本身精确（整数加法），
//   无浮点噪声；着色阶段直接消费，禁用任何「命中点 ± 法线半步」启发式重建
//   （启发式在体素棱边/UB fallback face 下会选错邻体素 → 6 邻域差分串色）。
struct VoxelHit {
  hit: bool,
  t: f32,
  pal: u32,
  face_id: u32,
  voxel: vec3<i32>,
}

// brick 缓存槽：一层分裂节点常驻（对应 CPU BrickCpu / Douglas BrickMaskEntry）
struct Brick {
  addr: u32,  // 节点绝对字址（b_struct）
  mask: u64,  // 64bit 分裂掩码（bit=1 = 子块分裂；bit=0 = 统一子块，色=pal）
  pal: u32,   // 节点 palette（统一子块颜色，0=空气）
}

// 单 chunk 内层级遍历。chunk_base = 根节点绝对字址；chunk_min = chunk 原点（局部 voxel）。
// 射线段 [t0, t1]（ro 系绝对 t）；entry_face = 进入本 chunk 的面（首 chunk 由调用方
// 用 normalize(-rd) 兜底，后续 chunk 为跨 chunk 面）。
// 返回 VoxelHit（t 为 ro 系绝对 t）；走出 chunk 未命中 → hit=false。
//
// depth_cap（beam 保守模式）：gate 深度 d=(3-level) 的分裂子块且 d>=depth_cap 时
// 返回子块入口 t（保守下界）；主/阴影 pass depth_cap=3 恒不触发（d<=2）。
// LOD：split 子节点一律下钻（#2 勘误：远场多数色早停色渗出→穿墙 + 逐面着色，
// 见 docs/douglas-final.md；uniform 子节点精确 palette 早停 = c_mask==0 快路径）。
fn trace_chunk(chunk_base: u32, chunk_min: vec3<f32>,
               ro: vec3<f32>, rd: vec3<f32>, sign_v: vec3<i32>,
               t0: f32, t1: f32, entry_face: u32, depth_cap: u32) -> VoxelHit {
  // 擦边退化（t0>=t1：射线只蹭到 chunk 边界）→ 无体素内部可穿过，直接 miss
  if (t0 >= t1) { return VoxelHit(false, 0.0, 0u, 0u, vec3<i32>(0)); }
  // chunk 局部 voxel 坐标（chunk 原点 = 0）；t 仍是 ro 系绝对 t
  let ro_c = ro - chunk_min;
  // 预算倒数：side 距离/步长增量改乘法（每外层省 3 个 fdiv）
  let inv_rd = 1.0 / rd;
  // 不变量预计算（sign_v / rd 在整个 trace_chunk 内不变）
  let abs_inv_rd = abs(inv_rd);                  // 优化1：省每轮 abs(inv_rd)
  let axis_on = abs(rd) > vec3<f32>(1e-30);      // 优化1：省每轮 abs(rd)>eps
  let oct = select(0u, 1u, sign_v.x >= 0)        // 优化2：省每次 LUT 检查 3 个 select
    | (select(0u, 1u, sign_v.y >= 0) << 1u)
    | (select(0u, 1u, sign_v.z >= 0) << 2u);
  let dir_side = select(vec3<f32>(0.0), vec3<f32>(1.0), sign_v >= vec3<i32>(0)); // 优化3
  let face_base = vec3<u32>(                     // 优化4：省 inner 每步 select+比较
    select(1u, 0u, sign_v.x >= 0),
    select(1u, 0u, sign_v.y >= 0),
    select(1u, 0u, sign_v.z >= 0));
  var bricks: array<Brick, 4u>;
  let r_ml = b_struct[chunk_base];
  let r_mh = b_struct[chunk_base + 1u];
  let r_pal = b_struct[chunk_base + 2u] & 0xFFu;
  bricks[3u] = Brick(chunk_base, (u64(r_mh) << 32u) | u64(r_ml), r_pal);
  var level: u32 = 3u;
  // 当前体素（chunk 局部 voxel 整数坐标，0..255；跨出 chunk 的步进瞬态可达 -1/256）
  let p0 = ro_c + rd * t0;
  var v = clamp(vec3<i32>(floor(p0)), vec3<i32>(0), vec3<i32>(255));
  var cur_t = t0;
  var face = entry_face;
  // 防挂死安全网：几何上界 ≈ 25k 量级；真实场景每 chunk 仅几十~几百次迭代
  var budget: u32 = 65536u;
  let lut_disable = view_u.lod.w > 0.5;
  loop {
    if (budget == 0u) { break; }
    budget = budget - 1u;
    if (level > 3u) { return VoxelHit(false, 0.0, 0u, 0u, vec3<i32>(0)); }
    // ---- traverse：从当前 level 下钻到 v 处内容（Douglas traverse_bit_set）----
    loop {
      let b = bricks[level];
      // WGSL 移位 RHS 必须 u32
      let sh = vec3<u32>(level * 2u);
      let cell = vec3<i32>(v >> sh) & vec3<i32>(3); // &3 恒 0..3，clamp 死代码已删
      let idx = u32(cell.z * 16 + cell.y * 4 + cell.x);
      if (level == 0u) {
        // 叶节点 inline palette：bit=1（非空体素）才 load inline word 取色；
        // bit=0 空气体素零 load（mask 在手）。
        if ((b.mask & (u64(1) << idx)) != u64(0)) {
          let w = b_struct[b.addr + NODE_FIXED_WORDS + (idx >> 2u)];
          let leaf_pal = (w >> ((idx & 3u) * 8u)) & 0xFFu;
          if (leaf_pal != 0u) { return VoxelHit(true, cur_t, leaf_pal, face, v); }
        }
        break; // 空气 leaf
      }
      let bit = u64(1) << idx;
      if ((b.mask & bit) == u64(0)) {
        // 统一子块：颜色 = 节点 palette（0=空气）
        if (b.pal != 0u) { return VoxelHit(true, cur_t, b.pal, face, v); }
        break; // 空气统一子块
      }
      // depth_cap（beam 保守）：gate 深度 d=3-level 的分裂子块到达 cap → 子块入口 t
      let gd = 3u - level;
      if (gd >= depth_cap) { return VoxelHit(true, cur_t, b.pal, face, v); }
      // 分裂子块 → popcount 定位 child（用原始 mask，非 LUT eff）
      let below = b.mask & (bit - u64(1));
      let pop = countOneBits(u32(below)) + countOneBits(u32(below >> 32u));
      let child_addr = chunk_base + b_struct[b.addr + NODE_FIXED_WORDS + pop];
      let c_ml = b_struct[child_addr];
      let c_mh = b_struct[child_addr + 1u];
      let c_pw = b_struct[child_addr + 2u];
      let c_pal = c_pw & 0xFFu;
      let c_mask = (u64(c_mh) << 32u) | u64(c_ml);
      // 统一子节点快路径（旧 c_mask==0）：wire 任意层的分裂位都可能指向 3 字统一
      // 节点（mask=0，pal 直决；叶层统一节点无 inline 16 字，禁读 addr+3 之后）
      if (c_mask == u64(0)) {
        if (c_pal != 0u) { return VoxelHit(true, cur_t, c_pal, face, v); }
        break;
      }
      // 勘误（#2，用户实测）：split 子节点远场多数色早停（palette 高字节 node_lod）
      // 色渗出→穿墙，且命中点落子块入口空气体素 → 6 邻域差分退化回退面法线 →
      // 逐面着色。禁恢复：split 一律下钻；uniform 子节点早停已由上方 c_mask==0
      // 快路径以精确 palette 覆盖（docs/douglas-final.md #2 勘误）。
      level = level - 1u;
      bricks[level] = Brick(child_addr, c_mask, c_pal);
    }
    // ---- 整砖 LUT 跳过（Douglas march mask skip；仅 palette==0 节点安全）----
    // 当前 brick 从当前格/象限无可达分裂子块 → 直接升一层跨步（不重新 traverse：
    // 父帧当前格就是本砖，重 traverse 会原地再下钻；父层 dda 先跨步出本砖再判位）
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
    // ---- v 处为空气：当前 level brick 内 DDA（Douglas dda）----
    let log2 = level * 2u;
    let s = 1i << (level * 2u); // 子块边长 voxel：1/4/16/64
    let s_mask = ~(s - 1i);    // 对齐掩码（two's complement: ~(s-1) = -s）
    // side_distance_for_ray：v 对齐到 s 的基址；正向 → 基址+s，负向 → 基址
    let base_v = v & vec3<i32>(s_mask);
    let boundary = vec3<f32>(base_v) + dir_side * f32(s);
    var side = vec3<f32>(1e+30);
    side = select(side, (boundary - ro_c) * inv_rd, axis_on);
    side = max(side, vec3<f32>(cur_t));
    // 每轴步长 t 增量（level 不变则不变）：inner 里 O(1) 加法
    let step_inc = f32(s) * abs_inv_rd;
    var step_axis: u32 = 0u;
    var changed = false;
    loop {
      // 选最近边界轴
      var mn: u32;
      if (side.x <= side.y && side.x <= side.z) { mn = 0u; }
      else if (side.y <= side.z) { mn = 1u; }
      else { mn = 2u; }
      step_axis = mn;
      cur_t = side[mn];
      if (cur_t >= t1) { return VoxelHit(false, 0.0, 0u, 0u, vec3<i32>(0)); } // 段内再无子块可入
      let old_cell = (v[mn] >> log2) & 3;
      // 沿 mn 轴整子块跨越
      v[mn] = v[mn] + sign_v[mn] * s;
      side[mn] = side[mn] + step_inc[mn];
      face = mn * 2u + face_base[mn];
      // 跨出 brick（4 子块）？正向往 3→外、负向往 0→外
      let crossed = select(old_cell == 0, old_cell == 3, sign_v[mn] >= 0);
      if (crossed) { changed = true; break; }
      // 新子块内容：level 0 查 inline palette（mask!=0 inline 叶才有）；
      // level 1..3 = 分裂位或节点统一实体色。
      // 命中直接返回（cur_t=进入距离、face=进入面）——省一整轮外层
      // （traverse 节点 load + LUT + side 重算）；仅「分裂子块」回 traverse 下钻。
      let b = bricks[level];
      let sh2 = vec3<u32>(log2);
      let cell = vec3<i32>(v >> sh2) & vec3<i32>(3);
      let idx = u32(cell.z * 16 + cell.y * 4 + cell.x);
      if (level == 0u) {
        // bit=1（非空体素）才 load inline word；bit=0 空气体素零 load
        if (b.mask != u64(0) && (b.mask & (u64(1) << idx)) != u64(0)) {
          let w = b_struct[b.addr + NODE_FIXED_WORDS + (idx >> 2u)];
          let dp = (w >> ((idx & 3u) * 8u)) & 0xFFu;
          if (dp != 0u) { return VoxelHit(true, cur_t, dp, face, v); }
        } else if (b.mask == u64(0) && b.pal != 0u) {
          // 防御：uniform 叶（正常下钻快路径已处理）
          return VoxelHit(true, cur_t, b.pal, face, v);
        }
      } else {
        let mb = (b.mask & (u64(1) << idx)) != u64(0);
        if (mb) { break; } // 分裂子块 → 回 traverse 下钻
        if (b.pal != 0u) { return VoxelHit(true, cur_t, b.pal, face, v); } // 统一实体
      }
      // 空气子块 → 回 loop 顶重选 mn 轴继续
    }
    if (changed) {
      level = level + 1u;
      if (level > 3u) { return VoxelHit(false, 0.0, 0u, 0u, vec3<i32>(0)); }
    }
    // ---- firstTrailingBit 层级自适应跨级跳（Douglas march 尾部）----
    // 步进轴新坐标的尾随零位 = 对齐 run 长度：正向 comp=对齐基址（tz 直接读），
    // 负向 comp=区域尾址（基址|~mask，+1 后读 tz）。tz>>1 = 可跨步的最粗 level。
    let positive = sign_v[step_axis] >= 0;
    let cur_log2 = level * 2u;
    let m: u32 = 0xFFFFFFFFu << cur_log2;
    let not_m = ~m;                              // = (1<<cur_log2)-1，复用于 region_max
    let vmin_u = u32(v[step_axis]); // i32→u32 位环绕（负值公式自然处理）
    var comp: u32;
    if (positive) { comp = vmin_u & m; } else { comp = (vmin_u & m) | not_m; }
    let tz_i = firstTrailingBit(comp + select(1u, 0u, positive)); // 0 → -1
    var new_level: u32 = 0xFFFFFFFFu;
    if (tz_i >= 0) { new_level = u32(tz_i) >> 1u; }
    level = max(level, new_level);
    if (level > 3u) { return VoxelHit(false, 0.0, 0u, 0u, vec3<i32>(0)); } // 跨出 chunk（tz≥8）
    // 对齐快照：v 钳到 cur_t 射线点所在的当前 level 区域，步进轴取精确边界整数
    // （其余轴按射线实际位置吸附，消除只沿单轴步进的漂移）
    let mi = i32(m);
    let base = v & vec3<i32>(mi);
    let p = ro_c + rd * cur_t;
    let region_max = base + vec3<i32>(i32(not_m));
    v = clamp(vec3<i32>(floor(p)), base, region_max);
    v[step_axis] = i32(comp);
  }
  return VoxelHit(false, 0.0, 0u, 0u, vec3<i32>(0));
}

// ============================================================================
// 统一 DDA：trace_grid 一套管线同时服务主网格和逐物体 OBJ
// 以下函数严格对应 Rust `brickmap/dda.rs` CPU 参考实现（逐字镜像）。
// ============================================================================

// slab 法射线-AABB 求交，返回 (t_enter, t_exit)；平行且在外 → (1.0, 0.0) miss 哨兵
// 对应 dda.rs::slab_box
fn slab_box(ro: vec3<f32>, rd: vec3<f32>, mn: vec3<f32>, mx: vec3<f32>, t0: f32, t1: f32) -> vec2<f32> {
  var t_enter = t0;
  var t_exit = t1;
  // X
  if (abs(rd.x) < 1e-30) {
    if (ro.x < mn.x || ro.x > mx.x) { return vec2<f32>(1.0, 0.0); }
  } else {
    let ta = (mn.x - ro.x) / rd.x;
    let tb = (mx.x - ro.x) / rd.x;
    t_enter = max(t_enter, min(ta, tb));
    t_exit = min(t_exit, max(ta, tb));
  }
  // Y
  if (abs(rd.y) < 1e-30) {
    if (ro.y < mn.y || ro.y > mx.y) { return vec2<f32>(1.0, 0.0); }
  } else {
    let ta = (mn.y - ro.y) / rd.y;
    let tb = (mx.y - ro.y) / rd.y;
    t_enter = max(t_enter, min(ta, tb));
    t_exit = min(t_exit, max(ta, tb));
  }
  // Z
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

// ============================================================================
// 统一 DDA 核心（Phase 3 OBJ→Volume 统一）
// 主世界和物体本质都是体素网格 DDA——用相同的 slab→chunk 间 A&W→trace_chunk 流程，
// 差异通过 Grid 参数化（tree_base + 变换矩阵 + chunk 窗口）。
// 零 kind 分支：所有 grid 从统一 b_struct[tree_base..] 读树、b_palette[palette_base..] 取色。
// ============================================================================

// 统一命中结构：hit/t/pal/n(世界空间法线)/face_id(0..5)/obj_id(-1=主世界,>=0=物体)
// voxel：命中固体体素 grid 局部 voxel 整数坐标（主世界 = 世界坐标）——DDA 整数步进
//   精确产出，dda_main 着色直接消费（p_voxel 由 voxel 中心经 volume 变换得到）。
struct UnifiedHit {
  hit: bool,
  t: f32,
  pal: u32,
  n: vec3<f32>,        // 世界空间法线（光影用）
  face_id: u32,        // 命中面 0..5（与 face_index_from_normal 对齐）
  voxel: vec3<i32>,    // grid 局部 voxel 命中体素（trace_chunk 的 v + ci*256）
  obj_id: i32,         // -1 = 主世界, >=0 = 物体索引
}

// 统一层次 DDA：slab→chunk 间 A&W→trace_chunk（完全通用，零 kind 分支）
// 所有网格（brickmap / obj / 未来任何体素 grid）走同一条路径
// 【性能】参数直接传 volume 索引（不传 15 字段 Grid 结构体）——WGSL 函数大结构体
// 按值传参在 naga/驱动下实测有 ~µs 级开销（slab_box 内联 17ms→2.75ms 实证）；
// slab 也直接内联（原 slab_box 调用版实测 17ms 纯调用开销）
// trace_grid 版本：接收 Grid 参数（make_grid 已读过 grid_descs，一次读完），
// 消除旧 trace_grid_idx 内部重复读 grid_descs[idx] 的 storage buffer 双读。
// 原 trace_grid_idx(idx,...) 改名并改签名。
fn trace_grid(g: Grid, origin: vec3<f32>, dir: vec3<f32>, t_cap: f32, t_min: f32, depth_cap: u32) -> UnifiedHit {
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
  // P1：主世界（obj_id < 0）= identity 变换（make_grid 保证 pos=0/rot=I/scale=1、
  // 局部 AABB == 世界 AABB）→ 局部坐标直取世界坐标，跳过整段局部变换
  // （9 dot + 6 除法）与和世界 slab 数学等价的局部 slab。
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
  // P3 beam：t_min = 3×3 邻域低分辨率射线的最小命中距离，跳过 [tl0, t_min) 的
  // 空空间 traversal。保守性（beam_main d_beam 注释）：d_beam 内体素投影双向宽
  // ≥ 2·BEAM_DIV px → 必有邻域 beam 网格点命中它 → min t ≤ 真实首命中 t +
  // ~2.3（同体素侧入差，见 BEAM_BACKOFF）；dda_main 处已回退 4 单位补偿。
  // d_beam 之外由主射线完整 trace，不漏任何几何。
  tl0 = max(tl0, t_min);
  if (tl1 <= tl0) { return miss; }
  // debug_mode.z == 2（诊断）：跳过 chunk 步进层（二分 make_grid+slab vs 遍历成本）
  if (view_u.debug_mode.z > 1.5) { return miss; }
  // ---- chunk 间 A&W（256³ 一格）+ chunk 内层次 mask DDA（trace_chunk）----
  var sign_v = vec3<i32>(1i);
  sign_v = select(sign_v, vec3<i32>(-1i), rd < vec3<f32>(0.0));
  // P2：方向位掩码 LUT——一次查表拿到 side（boundary）+ 三轴步进 face id，
  // 消除 trace_chunk 内每帧/每步的 select 方向分支
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
  // 预计算 inv_rd（6 次除法 → 3 次除法 + 3 次乘法）
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
  // entry_face：跨入当前 chunk 的面。首 chunk 用 normalize(-rd) 兜底（相机贴面/UB；
  // 按约定 UB 直接返回该 voxel 颜色，face_id 反推）；后续 chunk 由跨轴 + sign_v 更新。
  // face_index_from_normal 内联（函数调用开销，见 worklog）
  let en = normalize(-rd);
  let eax = abs(en.x);
  let eay = abs(en.y);
  let eaz = abs(en.z);
  var entry_face: u32 = 4u;
  if (eax >= max(eay, eaz)) {
    entry_face = select(0u, 1u, en.x >= 0.0);
  } else if (eay >= eaz) {
    entry_face = select(2u, 3u, en.y >= 0.0);
  } else {
    entry_face = select(4u, 5u, en.z >= 0.0);
  }
  for (var step_c: u32 = 0u; step_c < max_chunk_steps; step_c = step_c + 1u) {
    let t_exit_c = min(tmax_c.x, min(tmax_c.y, tmax_c.z));
    let t1 = min(t_exit_c, tl1);
    // ---- chunk 窗口定位（窗口外 = 空气，直接步进）----
    let rel = ci - index_origin;
    if (all(rel >= vec3<i32>(0)) && all(vec3<u32>(rel) < index_dims)) {
      let rel_u = vec3<u32>(rel);
      let index_addr = tree_base + rel_u.x + rel_u.y * CHUNK_INDEX_CAP
        + rel_u.z * (CHUNK_INDEX_CAP * CHUNK_INDEX_CAP);
      let entry = b_struct[index_addr];
      if (entry != 0u) {
        let chunk_base = tree_base + entry - 1u;
        let chunk_min = vec3<f32>(ci) * f32(CHUNK_SIZE);
        // P3 beam：chunk 内遍历从 max(chunk 入口, t_min) 开始，跳过 t_min 前的空空间
        let t0c = max(t_enter_c, t_min);
        let h = trace_chunk(chunk_base, chunk_min, ro, rd, sign_v,
                            t0c, t1, entry_face, depth_cap);
        if (h.hit) {
          // 统一法线计算：trace_chunk 内部已推好 face_id，调用方零分支
          let n_local = face_normal_from_index(h.face_id);
          let n_world = normalize(n_local.x * col0 + n_local.y * col1 + n_local.z * col2);
          // chunk 局部 v → grid 局部体素（ci 为 trace_chunk 命中时所在 chunk，未步进）
          let voxel = h.voxel + ci * i32(CHUNK_SIZE);
          return UnifiedHit(true, h.t, h.pal, n_world, h.face_id, voxel, obj_id);
        }
      }
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
  return miss;
}

// 构造统一 Grid：从 grid_descs[idx] 解码所有字段（主世界 idx=0，物体 idx≥1）。
// 主世界 = identity 变换（pos=0/rot=I/scale=1），物体 = 任意变换。
// chunk 步数上限由窗口 dims 估算（曼哈顿对角 ×3 + 余量）；物体 dims=(1,1,1)。
fn make_grid(idx: u32) -> Grid {
  let d = grid_descs[idx];
  let origin = vec3<i32>(d.index_origin_x, d.index_origin_y, d.index_origin_z);
  let dims = vec3<i32>(i32(d.index_dims_x), i32(d.index_dims_y), i32(d.index_dims_z));
  let chunk_budget = u32(dims.x + dims.y + dims.z) * 3u + 16u;
  // 主世界（idx=0，identity：局部=世界）：局部 AABB = 窗口 AABB（origin 可为负，
  // [0,256]³ 默认盒会把窗口绝大部分 slab 剔除）；物体 = 单 chunk 局部 [0,256]³。
  let is_world = idx == 0u;
  let l_mn = select(vec3<f32>(0.0), d.aabb_min.xyz, is_world);
  let l_mx = select(vec3<f32>(f32(CHUNK_SIZE)), d.aabb_max.xyz, is_world);
  return Grid(
    d.aabb_min.xyz, d.aabb_max.xyz,       // 世界 AABB
    l_mn, l_mx,                            // 局部 AABB
    chunk_budget,
    d.rot0.xyz, d.rot1.xyz, d.rot2.xyz,    // 旋转矩阵列
    d.pos_scale.xyz, d.pos_scale.w,        // pos + scale
    d.tree_base, d.palette_base,           // 数据源基址
    origin, vec3<u32>(d.index_dims_x, d.index_dims_y, d.index_dims_z),
    // obj_id 约定（镜像 cpu_reference_trace_volumes）：idx=0 主世界→-1；
    // idx≥1 物体→0 基 obj_id = idx-1（volume.rs list[0]=主世界, list[1..]=物体
    // obj_id 0..N-1）。下游按 obj_id+1 反查 grid_descs 取变换（如 p_voxel 归一化）。
    select(-1i, i32(idx) - 1i, idx > 0u),
  );
}


// ---- sRGB → linear 转换（palette/sky 均以 sRGB u8/255 存储，光照前必须转 linear） ----
fn srgb_channel_to_linear(c: f32) -> f32 {
  if (c <= 0.04045) {
    return c / 12.92;
  }
  return pow((c + 0.055) / 1.055, 2.4);
}

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(
    srgb_channel_to_linear(c.x),
    srgb_channel_to_linear(c.y),
    srgb_channel_to_linear(c.z),
  );
}

// ---- palette albedo 解包（sRGB → linear）----
fn palette_albedo(palette_base: u32, pal: u32) -> vec3<f32> {
  let w0 = b_palette[palette_base + pal * 2u];
  let srgb = vec3<f32>(
    f32(w0 & 0xFFu),
    f32((w0 >> 8u) & 0xFFu),
    f32((w0 >> 16u) & 0xFFu),
  ) / 255.0;
  return srgb_to_linear(srgb);
}

// 天空纯色（miss 像素输出 + 探针射线 miss 端点；Minecraft 白天平原 #78A7FF）
// sRGB → linear（与 palette_albedo 同空间；光照全程 linear，输出经 linear_to_srgb 还原）
fn sky_rgb() -> vec3<f32> {
  return srgb_to_linear(light_u.sky_color.xyz);
}


// 着色法线 = 命中面法线（`UnifiedHit.n`，trace_chunk 按进入面直接产出，已是世界系单位向量）。
//
// 为什么不用「逐体素隐式法线」（6 邻域占用差分）：1 体素厚的薄板（地板/墙，体素世界的主力
// 几何）的 ±x/±z 邻接都实心、±y 都空气，二值差分三项全部抵消 → 恒为零向量，任何**对称**
// 模板（±2、±4…）都一样退化 —— 因为薄板两侧往外都是空气。几何上薄板两侧本来就不存在唯一
// 的方向，逐面法线是唯一有定义的答案。
//
// 实测后果：旧实现退化时硬编码 (1,0,0)，所有薄板法线变成 +X → 垂直于 X 的薄墙只有 +X 那
// 侧碰巧正确，-X 那侧 8 个角探针全落在背面，被 DDGI 的 wn 背向剔除剔光（Probe 品红、GI
// 整面全黑）；地板则是一半探针被误用（漏光/发暗），太阳直光也按 +X 计算。
//
// 代价：放弃「同体素跨像素同色」（体素在棱边处会有逐面明暗差）。收益：每像素少 6 次
// sample_brickmap（一次 chunk 定位 + 最多 4 层树下钻）。

// 场景级命中：UnifiedHit + 命中 volume 的 palette 基址（dda_main 取 albedo 用）
struct SceneHit {
  uh: UnifiedHit,
  palette_base: u32,
}

// 主世界先跑 + 逐物体收缩 t_cap 取最近命中（beam 预 pass 与 dda_main 主射线共用）。
// 物体循环不被世界 miss 短路：天空背景前的物体必须可见。
fn trace_scene(origin: vec3<f32>, dir: vec3<f32>, t_cap: f32, t_min: f32, depth_cap: u32) -> SceneHit {
  var best_t = 1e+30;
  // P1：g0 只构造一次（旧代码 make_grid(0u) 调两次 = 144B GridDesc 双读）
  let g0 = make_grid(0u);
  var best = SceneHit(trace_grid(g0, origin, dir, t_cap, t_min, depth_cap), 0u);
  if (best.uh.hit) {
    best_t = best.uh.t;
    best.palette_base = g0.palette_base;
  }
  let n = g.grid_count;
  for (var i: u32 = 1u; i < n; i = i + 1u) {
    let gi = make_grid(i);
    let mhi = trace_grid(gi, origin, dir, min(best_t, t_cap), t_min, depth_cap);
    if (mhi.hit && mhi.t < best_t) {
      best_t = mhi.t;
      best.uh = mhi;
      best.palette_base = gi.palette_base;
    }
  }
  return best;
}

// 线性 → sRGB 转换
fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
  let l = clamp(c, vec3<f32>(0.0), vec3<f32>(1.0));
  return vec3<f32>(
    select(12.92 * l.r, pow(l.r, 1.0 / 2.4) * 1.055 - 0.055, l.r > 0.0031308),
    select(12.92 * l.g, pow(l.g, 1.0 / 2.4) * 1.055 - 0.055, l.g > 0.0031308),
    select(12.92 * l.b, pow(l.b, 1.0 / 2.4) * 1.055 - 0.055, l.b > 0.0031308)
  );
}

// ============================================================================
// P3 Beam 预 pass：低分辨率（全分辨率 / BEAM_DIV）**保守** trace，只输出最近命中 t。
//
// 保守性（#18 原文："低分辨率射线 never step far enough that voxels could get
// smaller than the minimum size required to always be hit by one low-res ray"）：
// 设 D_safe = 0.35/(BEAM_DIV·px_ang)。beam 射线只行进到 min(t_cap, D_safe)：
//   · 体素 v 在距离 d ≤ D_safe：v 投影双向宽 ≥ 2·BEAM_DIV px（含 45° 最坏菱形
//     + 旋转裕量）→ 全局 beam 网格必有一点落进投影，且距主像素 ≤ 6 px（3×3 邻域
//     半宽，见 dda_main）→ 邻域某条 beam 命中 v → min t ≤ d。✓
//   · 体素 v 在距离 d > D_safe：所有邻域 beam 返回 ≤ D_safe（在 D_safe 停下或命中
//     更前物体）→ 邻域 min ≤ D_safe < d → 主射线从 D_safe 起步继续行进命中 v。✓
// 输出与无 beam 逐像素一致。depth_cap=3（不限制八叉树深度，只限制行进距离）。
// miss 存 t_cap_beam（非 1e30、非 t_cap）：邻域 3×3 全 miss 时主射线起点 ≤ D_safe。
// ============================================================================
@compute @workgroup_size(8, 8, 1)
fn beam_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let beam_size = textureDimensions(beam_depth);
  if (gid.x >= beam_size.x || gid.y >= beam_size.y) { return; }
  // 低分辨率像素中心 → 全分辨率 NDC（用 beam_size 归一化，保证覆盖全视锥）
  let px = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5))
         / vec2<f32>(f32(beam_size.x), f32(beam_size.y));
  let uv = vec2<f32>(px.x * 2.0 - 1.0, 1.0 - px.y * 2.0);
  let near_ndc = vec4<f32>(uv.x, uv.y, 0.0, 1.0);
  let far_ndc  = vec4<f32>(uv.x, uv.y, 1.0, 1.0);
  let near_world_h = view_u.inv_view_proj * near_ndc;
  let far_world_h  = view_u.inv_view_proj * far_ndc;
  let near_world = near_world_h.xyz / near_world_h.w;
  let far_world  = far_world_h.xyz  / far_world_h.w;
  let dir = normalize(far_world - near_world);
  let origin = view_u.cam_pos_voxel.xyz;
  let t_cap = length(far_world - near_world);
  // D_safe = 0.35/(BEAM_DIV·px_ang)：体素在 d ≤ D 处屏幕投影双向宽 ≥ 2·BEAM_DIV px
  // （0.35 ≈ 1/(2√2)，覆盖 45° 最坏菱形 + 旋转裕量）→ 全局 beam 网格必有一点落进
  // 投影，且距主像素 ≤ 3×3 邻域半宽 6 px（见 dda_main）→ 邻域 min t ≤ 体素 t +
  // ~2.3（同体素侧入差，dda_main 已回退 BEAM_BACKOFF 补偿）。
  // 旧公式 BEAM_DIV/px_ang 超出保守距离 8 倍：远处亚间距体素 beam 漏记 + 主射线
  // 从错误 t_min 起步 → 穿墙（用户实测）。#18 原文即"低分辨率射线永不行进到
  // 体素小于最小必命中尺寸的距离"。
  let d_beam = 0.35 / (f32(BEAM_DIV) * max(view_u.lod.x, 1e-6));
  let t_cap_beam = min(t_cap, d_beam);

  // depth_cap=3：八叉树全深度（不限制深度，只限制行进距离 t_cap_beam）
  let h = trace_scene(origin, dir, t_cap_beam, 0.0, 3u);
  // miss 存 t_cap_beam（非 t_cap）：3×3 邻域全 miss 时主射线起点 ≤ d_beam，
  // d_beam 之外的几何由主射线完整 trace。旧 bug 存 t_cap（frustum 全长）时，
  // 全 miss 邻域把主射线起点推到 frustum 末端 → 整条射线假 miss（场景丢失）。
  let t = select(t_cap_beam, h.uh.t, h.uh.hit);
  textureStore(beam_depth, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(t, 0.0, 0.0, 0.0));
}

// ============================================================================
// DDA 主入口：R3-18 直光层 — 命中 → 直光着色（硬阴影 + sky 环境 + emissive），miss → 纯色天空
// ============================================================================
@compute @workgroup_size(8, 8, 1)
fn dda_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let size = textureDimensions(out_tex);
  if (gid.x >= size.x || gid.y >= size.y) { return; }
  let coord0 = vec2<i32>(i32(gid.x), i32(gid.y));

  // ---- 反投影：像素中心 (gid + 0.5) → NDC (u, v) ∈ [-1, 1] ----
  let px = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5)) / vec2<f32>(f32(size.x), f32(size.y));
  let uv = vec2<f32>(px.x * 2.0 - 1.0, 1.0 - px.y * 2.0);
  let near_ndc = vec4<f32>(uv.x, uv.y, 0.0, 1.0);
  let far_ndc  = vec4<f32>(uv.x, uv.y, 1.0, 1.0);
  let near_world_h = view_u.inv_view_proj * near_ndc;
  let far_world_h  = view_u.inv_view_proj * far_ndc;
  let near_world = near_world_h.xyz / near_world_h.w;
  let far_world  = far_world_h.xyz  / far_world_h.w;
  let diff_world = far_world - near_world;
  let frustum_length = length(diff_world);
  let dir_voxel = normalize(diff_world);
  let origin_voxel = view_u.cam_pos_voxel.xyz;

  // ---- 场景 trace（P3 beam 起点跳过空空间）+ unlit 逐体素法线着色 ----
  // P3 beam：取当前像素 3×3 beam 邻域的最小命中 t 作为起点，跳过空空间。
  // beam 像素 = floor(全分辨率像素 / BEAM_DIV)；邻域 clamp 到 beam 边界。
  // lod.z > 0.5（GATE_NO_BEAM=1）：关闭 beam，t_min=0。
  var t_min = 0.0;
  if (view_u.lod.z < 0.5) {
    let beam_size = textureDimensions(beam_depth);
    let bx = gid.x / BEAM_DIV;
    let by = gid.y / BEAM_DIV;
    t_min = 1e+30;
    // 3×3 对称邻域（旧 2×2 右下偏置：体素投影凸六边形中心可偏出窗）。
    // beam 有效距离（d_beam）内体素投影双向宽 ≥ 2·BEAM_DIV px → 最近 beam
    // 网格点距主像素 ≤ ~6 px（最坏 45° 菱形 ≈ 5.7 px）→ ±1 beam 格覆盖。
    for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
      for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
        let sx = clamp(i32(bx) + dx, 0, i32(beam_size.x) - 1);
        let sy = clamp(i32(by) + dy, 0, i32(beam_size.y) - 1);
        let bt = textureLoad(beam_depth, vec2<i32>(sx, sy)).x;
        t_min = min(t_min, bt);
      }
    }
    // P5：保守回退（BEAM_BACKOFF 注释）——邻域 beam 命中主射线首命中体素时可
    // 从侧面/远面入射，命中 t 晚 ~2.3 单位；直接以 min t 起步会把起点推过真实
    // 命中 → 薄壁/剪影穿墙。d_beam 之外 t_min ≤ d_beam < 首命中 t，本就安全。
    t_min = max(t_min - BEAM_BACKOFF, 0.0);
  }
  let best = trace_scene(origin_voxel, dir_voxel, frustum_length, t_min, 3u);
  var col = sky_rgb();
  if (best.uh.hit) {
    // ---- 逐体素着色（Douglas #22/#23：一体素一色）+ 逐面法线 ----
    // albedo/采样点/阴影射线体素锚定（同体素同色）；法线取命中面 —— 薄板不存在唯一的
    // 逐体素法线（见 voxel_normal 处注释），故按面着色。
    let alb = palette_albedo(best.palette_base, best.uh.pal);
    let n = best.uh.n;
    // 体素中心 → 世界系（主世界 identity 直等于体素中心；物体经旋转/缩放变换）
    let gg = make_grid(u32(best.uh.obj_id) + 1u);
    let vc = vec3<f32>(best.uh.voxel) + vec3<f32>(0.5);
    let p_voxel = gg.pos + vec3<f32>(dot(vc, gg.col0), dot(vc, gg.col1), dot(vc, gg.col2)) * gg.scale;
    // R3-18 直光硬阴影（#02/#17 形态）：1 条太阳射线，不通即阴影。
    // 射线原点 = 体素中心 + n×0.5（贴向空气侧邻域）；起点若落回自身体素
    // （对角 implicit normal 时可能）→ pre-check 自命中 → 视为无遮挡（self-hit 防护）
    let sun_dir = light_u.lights[0].kind_pos_dir.yzw;
    let ndl = max(dot(n, sun_dir), 0.0);
    var sun = 0.0;
    if (ndl > 0.0) {
      let sh = trace_scene(p_voxel + n * 0.5, sun_dir, 8192.0, 0.0, 3u);
      sun = select(
        1.0,
        0.0,
        sh.uh.hit
          && !(all(sh.uh.voxel == best.uh.voxel) && sh.uh.obj_id == best.uh.obj_id),
      );
    }
    let sky = light_u.sky_color.xyz;
    let sun_c = light_u.lights[0].color_intensity.xyz * light_u.lights[0].color_intensity.w;
    let gi_on = ddgi_u.misc.x > 0.5;
    var gi = vec3<f32>(0.0);
    if (gi_on) {
      // 只把点从「命中体素中心」推到体素表面（0.5 体素）；更远的、随 cell 尺寸缩放的外推
      // 在 ddgi_sample_lod 内部按该 LOD 的 cs 施加（DDGI_BIAS_CELLS）。
      gi = ddgi_sample(p_voxel + n * 0.5, n) * ddgi_u.params.z / DDGI_PI;
    }
    col = alb * (sun_c * ndl * sun + sky * DDGI_BASE_AMBIENT + gi);
    if (ddgi_u.params.y > 0.5) {
      if (ddgi_u.params.y < 1.5) {
        col = alb * sky * DDGI_BASE_AMBIENT * 0.15 + alb * gi * 8.0;
      } else if (ddgi_u.params.y < 2.5) {
        col = alb * vec3<f32>(clamp(ddgi_dbg_wsum * 0.125, 0.0, 1.0)) * 2.0;
      } else if (ddgi_u.params.y < 3.5) {
        // d = 「包含该像素的壳」所在 LOD + 1（纯几何包含，与采样是否成功无关）；
        // d = 0 表示没有任何 LOD 盒包含该像素 —— 这是覆盖率问题，不是剔除问题。
        // 4 级必须给出 4 种可分辨颜色（旧阈值把 LOD0 与 LOD1 都涂成红，无法定位）。
        let d = ddgi_dbg_dom;
        col = vec3<f32>(0.1, 0.8, 0.2);                          // 绿：未被任何 LOD 包含
        col = select(col, vec3<f32>(0.9, 0.2, 0.2), d > 0.5);    // 红：LOD0
        col = select(col, vec3<f32>(0.95, 0.85, 0.1), d > 1.5);  // 黄：LOD1
        col = select(col, vec3<f32>(0.2, 0.4, 0.9), d > 2.5);    // 蓝：LOD2
        col = select(col, vec3<f32>(0.9, 0.2, 0.9), d > 3.5);    // 品红：LOD3
      } else {
        // Probe：先看「数据有没有取到」，再看 8 个角是被哪道闸门剔掉的。
        //   浅绿 = 壳没给出数据，但**更粗一级 LOD 兜住了** → GI 有效，只是精度粗
        //          （相机滚动时新进入窗口的那条带属于这一类，不是异常）
        //   绿   = 权重和 > 0 且辐照度非 0 → 本壳直接正常
        //   白   = 权重和 > 0，但辐照度 ≈ 0 → 过闸的探针图集是空的（写入/寻址问题）
        //   青   = 无数据闸门：探针没进本帧 worklist（near=false）、年龄太小，或该方向纹素
        //          从未被写过 —— 都不能以 0 参与平均
        //   橙   = ENABLED 闸门（该 cell 全满 → bake 没放探针）
        //   品红 = 法线背向闸门（wn <= 0）
        //   红   = depth 遮挡闸门（wd <= 0）
        //   灰   = 角越界（clamp 路径不会出现）
        // 旧版在这里对 dom<0.5 直接涂蓝，把这份直方图整个短路掉了。
        if (ddgi_dbg_fb > 1.5) {
          col = vec3<f32>(0.55, 0.95, 0.35);
        } else if (ddgi_dbg_wsum >= 1e-4) {
          col = select(
            vec3<f32>(0.15, 0.85, 0.25),
            vec3<f32>(0.95, 0.95, 0.95),
            ddgi_dbg_zero > 0.5,
          );
        } else {
          let r_np = f32(ddgi_dbg_rej_out.x);
          let r_wn = f32(ddgi_dbg_rej_out.y);
          let r_wd = f32(ddgi_dbg_rej_out.z);
          let r_ia = f32(ddgi_dbg_rej_out.w);
          let rmax = max(r_ia, max(r_wd, max(r_wn, r_np)));
          if (rmax <= 0.0) {
            col = vec3<f32>(0.5, 0.5, 0.5);
          } else if (r_wd >= rmax) {
            col = vec3<f32>(0.9, 0.15, 0.15);
          } else if (r_wn >= rmax) {
            col = vec3<f32>(0.95, 0.15, 0.7);
          } else if (r_ia >= rmax) {
            col = vec3<f32>(0.1, 0.85, 0.95);
          } else {
            col = vec3<f32>(0.95, 0.6, 0.1);
          }
        }
      }
    }
  }
  textureStore(out_tex, coord0, vec4<f32>(linear_to_srgb(col), 1.0));
}


const DDGI_IRR_TEXELS: u32 = 8u;
const DDGI_DEPTH_TEXELS: u32 = 16u;
const DDGI_PROBES_PER_LAYER_AXIS: u32 = 16u;
const DDGI_PROBES_PER_LAYER: u32 = 256u;
const DDGI_LOD_COUNT: u32 = 4u;
// meta word = age(低 8bit) | ENABLED(烘焙放了探针) | ACTIVE(本 cell 或 6 邻接有体素/物体，
// 且不在更细 LOD 覆盖内)；0 = 无探针哨兵
const DDGI_META_ENABLED: u32 = 256u;
const DDGI_META_ACTIVE: u32 = 512u;
const DDGI_AGE_MAX: u32 = 255u;
const DDGI_NO_PROBE: u32 = 0xFFFFFFFFu;
const DDGI_FLAG_ENABLED: u32 = 1u;
const DDGI_FLAG_NO_SURFACES: u32 = 2u;
const DDGI_ALPHA: f32 = 0.06;
const DDGI_DEPTH_ALPHA: f32 = 0.10;
const DDGI_TEXEL_MIN_WEIGHT: f32 = 1e-4;
/// 前后判定（wn）的锐度：wn = clamp(dot(n,-dir)/此值, 0, 1)。0.2 ≈ 78° 起满权重，
/// 90° 处归零。只影响这一项的过渡宽度，不参与采样点偏移。
const DDGI_NORMAL_BIAS: f32 = 0.2;
/// 采样点沿法线的外推量 = 该 LOD 的 cell 边长 × 此系数（参考实现里的 normal_bias 语义）。
/// 必须随间距缩放：固定的 0.7 体素（1.4cm）相对 LOD0 的 64cm cell 等于贴在表面上，
/// 会让前后判定落在临界值上、整面被判「探针在背面」。
const DDGI_BIAS_CELLS: f32 = 0.1;
const DDGI_T_MAX: f32 = 8192.0;
/// 每帧射线总预算，由 `ddgi_seal` 均分给全部活跃探针（rpp = 预算/活跃数，钳 [1,256]）。
/// 必须与 Rust 侧 `DDGI_RAY_BUDGET` 一致：改小 → cast/collect 帧时按比例下降，
/// 代价是每探针样本变少（噪声靠 α=0.06 的时域混合吸收）。
const DDGI_RAY_BUDGET: u32 = 131072u;
const DDGI_PROBE_BUDGET: u32 = 4096u;
const DDGI_SHADOW_T_MAX: f32 = 8192.0;
const DDGI_SHADOW_BIAS: f32 = 0.5;
const DDGI_EMIT_GAIN: f32 = 4.0;
const DDGI_PI: f32 = 3.14159265;
const DDGI_HYSTERESIS: f32 = 0.85;
const DDGI_IRRAD_GAMMA: f32 = 1.0;
const DDGI_BIG_CHANGE: f32 = 0.2;
const DDGI_BRIGHTNESS: f32 = 1.0;
const DDGI_MIN_STEP: f32 = 0.0009765625;
const DDGI_CHANGE_DROP: f32 = 0.0;
const DDGI_DELTA_CLAMP: f32 = 0.25;
const DDGI_SKY_RADIANCE_SCALE: f32 = 1.0;
const DDGI_RAY_BIAS: f32 = 0.5;
const DDGI_BASE_AMBIENT: f32 = 0.2;

// 世界空间探针网格（每 LOD）：origin.xyz = 世界原点（voxel）、origin.w = cell 边长；
// dims.xyz = cell 维度、dims.w = 该 LOD 在全局 slot 数组中的起始下标。
struct DdgiLod {
  origin: vec4<i32>,
  dims: vec4<u32>,
};
struct DdgiUniform {
  lods: array<DdgiLod, 4>,
  params: vec4<f32>,
  misc: vec4<f32>,
  // 脏区（世界 voxel AABB）：dirty_min.xyz = min、w = 1 表示有效；dirty_max.xyz = max（不含）
  dirty_min: vec4<f32>,
  dirty_max: vec4<f32>,
};
@group(4) @binding(0) var<uniform> ddgi_u: DdgiUniform;
// 图集**采样侧**（cast 回读 GI、着色 ddgi_sample 用）。写入侧在 @group(5)：
// 同一纹理不能在同一 pass 内既作采样纹理又作存储纹理，故读写拆成两个 bind group。
@group(4) @binding(1) var ddgi_irr: texture_2d_array<f32>;
@group(4) @binding(2) var ddgi_depth: texture_2d_array<f32>;
// 3：烘焙输出（bake 写 / sort 读）；4：age/enabled（读写）
@group(4) @binding(3) var<storage, read_write> ddgi_cell: array<u32>;
@group(4) @binding(4) var<storage, read_write> ddgi_meta: array<u32>;
// BG4 binding 5：indirect args / 计数器合一 buffer（array<atomic<u32>>，word 布局）：
//   [0..16)  cast indirect args ×4 LOD（每 LOD 4 word：x, y, z, pad）
//   [16..32) collect indirect args ×4 LOD
//   [32..36) rpp（每探针射线数）×4 LOD
//   [36..40) 活跃探针计数器 ×4 LOD（CPU 每帧清零；sort atomicAdd；seal 读取）
@group(4) @binding(5) var<storage, read_write> ddgi_indirect: array<atomic<u32>>;
// worklist item = vec4(probe_pos_voxel.xyz, bitcast<f32>(age | lod<<24))；每 LOD 段起于 slot_base[lod]
@group(4) @binding(6) var<storage, read_write> ddgi_worklist: array<vec4<f32>>;
@group(4) @binding(7) var<storage, read_write> ddgi_slot_pos: array<vec4<f32>>;
// 8：每 slot 已烘焙的世界 cell 键（xyz）+ 有效标志（w）；滚动增量烘焙用
@group(4) @binding(8) var<storage, read_write> ddgi_cell_id: array<vec4<i32>>;
// 9：阶段二 cast 输出的射线样本：每样本 2 个 vec4 = (方向.xyz, 命中距离) / (辐亮度.xyz, 1)
@group(4) @binding(9) var<storage, read_write> ddgi_samples: array<vec4<f32>>;

// ===== @group(5)：collect 的图集写入侧（只被 ddgi_collect 使用）=====
// 只放两个存储纹理：其它缓冲（worklist / samples / indirect / uniform）全部复用 BG4 的绑定。
// 同一 buffer 若同时在两个 bind group 里以「只读 + 读写」两种方式绑定，wgpu 会判定使用冲突。
@group(5) @binding(0) var ddgi_irr_out: texture_storage_2d_array<rgba16float, write>;
@group(5) @binding(1) var ddgi_depth_out: texture_storage_2d_array<r32float, write>;

// ===== @group(6)：seal 的 dispatch 参数写入侧（只被 ddgi_seal 使用）=====
// [0..16) cast args ×4 LOD；[16..32) collect args ×4 LOD（每 LOD 4 word：x, y, z, pad，y = lod+1）。
// 必须与 ddgi_indirect 分开：该 buffer 在 cast/collect pass 里作 indirect 参数源，
// 若同时被绑成 storage，wgpu 会判 usage 冲突（STORAGE_READ_WRITE 是独占用法）。
@group(6) @binding(0) var<storage, read_write> ddgi_args: array<atomic<u32>>;

const DDGI_INDIR_CAST_BASE: u32 = 0u;
const DDGI_INDIR_COLL_BASE: u32 = 16u;
const DDGI_INDIR_RPP_BASE: u32 = 32u;
const DDGI_INDIR_COUNT_BASE: u32 = 36u;
// ddgi_indirect 另用 [0..10)：cast/collect 走「全 LOD 连续线程空间」的映射表。
// seal 算前缀和写这里，cast/collect 用二分（4 次循环）把全局 tid 还原成 (lod, probe, ray)。
// 这样 dispatch 只需 2 次、且不需要用 indirect 的 y 维传 LOD（那条路不可靠）。
const DDGI_INDIR_RAYBASE_BASE: u32 = 0u;  // ray_base[lod]：该级首线程在全局空间的位置
const DDGI_INDIR_COLLBASE_BASE: u32 = 4u; // coll_base[lod]
const DDGI_INDIR_RAY_TOTAL: u32 = 8u;     // 全局 cast 线程总数
const DDGI_INDIR_COLL_TOTAL: u32 = 9u;    // 全局 collect 线程总数
// 收敛跳帧参数（Douglas can_skip_update）：age 达阈值 + 距相机 > SKIP_DIST×spacing 才跳；
// 每 LOD 按周期强制刷新（帧号 + slot 错峰），兜住「offset 没变但光照变了」的编辑。
const DDGI_SKIP_AGE: vec4<u32> = vec4<u32>(32u, 24u, 16u, 12u);
const DDGI_REFRESH_PERIOD: vec4<u32> = vec4<u32>(64u, 96u, 128u, 160u);
/// can_skip_update 的「远离相机」半径（单位 = 本级 cell 边长）。LOD0 的 cell 是 32 体素
/// （64cm），所以 24 → 15.36m：这个半径外的探针一旦收敛（age ≥ SKIP_AGE）就只在
/// REFRESH_PERIOD 的强制刷新帧投线，近场刷新频率完全不变。
const DDGI_SKIP_DIST_CELLS: f32 = 24.0;
/// 采样门限：刚被（重）烘的探针（age 0/1）图集还没写全，跳过它，让那条带短暂由粗一级
/// LOD 顶替。只跳 1 帧即可 —— 第一次 collect（age 1）本来就用 `snap` 直接覆写所有被覆盖
/// 的纹素，且未被覆盖的纹素会被显式清零（见 collect），所以 age 2 起读数就是干净的。
const DDGI_MIN_SAMPLE_AGE: u32 = 2u;
/// collect 每探针的纹素线程数：64 irr（8×8）+ 256 depth（16×16）
const DDGI_COLLECT_THREADS: u32 = 320u;

// 射线样本区不再分段：seal 用的是**全局统一 rpp**，所以样本下标 = 全局射线编号
// （cast 的 tid 本身就是它），容量 = DDGI_RAY_BUDGET = ddgi_samples 的槽数。

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

fn ddgi_fibonacci(i: u32, n: u32) -> vec3<f32> {
  let golden = DDGI_PI * (3.0 - sqrt(5.0));
  let y = 1.0 - (2.0 * f32(i) + 1.0) / f32(n);
  let r = sqrt(max(1.0 - y * y, 0.0));
  let a = golden * f32(i);
  return vec3<f32>(r * cos(a), y, r * sin(a));
}
fn ddgi_random_quat(u1: f32, u2: f32, u3: f32) -> vec4<f32> {
  let s1 = sqrt(1.0 - u1);
  let s2 = sqrt(u1);
  let a2 = 2.0 * DDGI_PI * u2;
  let a3 = 2.0 * DDGI_PI * u3;
  return vec4<f32>(s1 * sin(a2), s1 * cos(a2), s2 * sin(a3), s2 * cos(a3));
}
fn ddgi_quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
  let t = 2.0 * cross(q.xyz, v);
  return v + q.w * t + cross(q.xyz, t);
}
fn ddgi_ray_dir(probe_id: u32, frame: u32, ray_index: u32, rays_total: u32) -> vec3<f32> {
  let base = ddgi_fibonacci(ray_index, max(rays_total, 1u));
  let seed = ddgi_ray_rand(probe_id, frame, 0u);
  let u = ddgi_rand2(seed);
  let u3 = f32(ddgi_pcg(seed) & 0xFFFFu) / 65536.0;
  return ddgi_quat_rotate(ddgi_random_quat(u.x, u.y, u3), base);
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
  // cell 边长的权威来源是 uniform（Rust 侧 DDGI_LOD_CELL_SIZES），不再硬编码 16<<lod
  return ddgi_u.lods[lod].origin.w;
}
fn ddgi_lod_count(lod: u32) -> u32 {
  let d = ddgi_u.lods[lod].dims;
  return d.x * d.y * d.z;
}
fn ddgi_lod_slot_base(lod: u32) -> u32 {
  return ddgi_u.lods[lod].dims.w;
}
// ---- 世界锚定的槽位映射 ----
// 槽位残差 = 世界 cell 号对网格维度取正模；槽位下标 = slot_base + 线性(残差)。
// 关键性质：**「槽位 ↔ 世界 cell」的身份与相机无关**。相机滚动只会让「新进入窗口的那条带」
// 换掉世界 cell（旧数据本来就该丢），其余槽位保持自己的世界身份 → 图集不再因为相机移动而
// 整体失效。改之前槽位是「相对相机窗口的格号」，滚一格就把整级所有槽位的世界 cell 全换掉
// → 整级图集变成旧位置的读数 → 深度判定成片失败（大片红）+ 下一帧重写（大片绿）= 动态闪烁。
fn ddgi_slot(lod: u32, wc: vec3<i32>) -> u32 {
  let d = ddgi_u.lods[lod].dims.xyz;
  let di = vec3<i32>(d);
  let r = ((wc % di) + di) % di;
  return ddgi_lod_slot_base(lod) + u32(r.x) + u32(r.y) * d.x + u32(r.z) * d.x * d.y;
}
/// 窗口原点（cell 单位）。origin.xyz 已按 cs 对齐，故整除精确。
fn ddgi_origin_cell(lod: u32) -> vec3<i32> {
  return ddgi_u.lods[lod].origin.xyz / ddgi_u.lods[lod].origin.w;
}
/// 槽位在本级内的下标 → 该槽位**当前**覆盖的世界 cell（环面映射的逆）
fn ddgi_slot_world_cell(lod: u32, idx: u32) -> vec3<i32> {
  let di = vec3<i32>(ddgi_u.lods[lod].dims.xyz);
  let o = ddgi_origin_cell(lod);
  let r = vec3<i32>(ddgi_slot_cell(lod, idx));
  return o + ((r - o) % di + di) % di;
}
/// 世界 cell 是否在本级窗口内
fn ddgi_cell_in_window(lod: u32, wc: vec3<i32>) -> bool {
  let c = wc - ddgi_origin_cell(lod);
  return all(c >= vec3<i32>(0)) && all(c < vec3<i32>(ddgi_u.lods[lod].dims.xyz));
}
fn ddgi_slot_cell(lod: u32, idx: u32) -> vec3<u32> {
  let d = ddgi_u.lods[lod].dims;
  return vec3<u32>(idx % d.x, (idx / d.x) % d.y, idx / (d.x * d.y));
}
fn ddgi_slot_lod_of(slot: u32) -> u32 {
  for (var l = DDGI_LOD_COUNT; l > 1u; l = l - 1u) {
    if (slot >= ddgi_lod_slot_base(l - 1u)) { return l - 1u; }
  }
  return 0u;
}
// 占位纹理索引（阶段二/三重做布局前仅用于越界保护）
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
// meta word = age(低 8bit) | ENABLED | ACTIVE；0 = 无探针
//   ENABLED：烘焙阶段放了探针（有 irr/depth 数据可采样）
//   ACTIVE ：本 cell 或 6 邻接有体素 / 物体，且未被更细 LOD 覆盖 —— 本帧需要投线
fn ddgi_meta_pack(age: u32, enabled: bool, is_active: bool) -> u32 {
  return (age & 0xFFu)
    | select(0u, DDGI_META_ENABLED, enabled)
    | select(0u, DDGI_META_ACTIVE, is_active);
}
fn ddgi_meta_age(p: u32) -> u32 {
  return p & 0xFFu;
}

fn ddgi_lum(c: vec3<f32>) -> f32 {
  return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}
fn ddgi_maxc(c: vec3<f32>) -> f32 {
  return max(c.x, max(c.y, c.z));
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

// ============================================================================
// 阶段一 seal：sort 之后独立 pass（pass 边界保证计数器可见）。4 线程各管一个 LOD：
// 读活跃探针数 → 算 rpp（固定预算/活跃数，clamp 8..256）→ 写 cast/collect indirect args。
// ============================================================================
@compute @workgroup_size(4, 1, 1)
fn ddgi_seal(@builtin(local_invocation_id) lid: vec3<u32>) {
  let lod = lid.x;
  // 每线程各自把 4 个 LOD 的活跃计数读一遍，独立算出前缀和 —— 计数都在全局内存里，
  // 不需要 workgroup 同步（thread 0 顺带写出总线程数与唯一一份 indirect args）。
  // rpp 是**全局统一**的：射线预算在全部活跃探针间均分（Douglas #23 原文 "dividing them
  // amongst all of the active probes"）。旧版按 LOD 权重 0.5/0.25/0.15/0.10 分摊预算，
  // 粗级探针拿到的射线数远低于平均（rpp≈3）→ 深度图 256 个纹素每帧只写到一小部分，
  // 遮挡判定(wd)随之失效。
  var total_active = 0u;
  for (var l = 0u; l < DDGI_LOD_COUNT; l = l + 1u) {
    total_active = total_active + atomicLoad(&ddgi_indirect[DDGI_INDIR_COUNT_BASE + l]);
  }
  // 下限 1：总活跃探针数 ≤ 槽数 65536 < 预算 262144，故 total_active×rpp 恒不超预算
  // （= ddgi_samples 的容量），无需再分段。
  let rpp = max(1u, min(256u, DDGI_RAY_BUDGET / max(total_active, 1u)));

  var ray_base = 0u;
  var coll_base = 0u;
  var total_ray = 0u;
  var total_coll = 0u;
  for (var l = 0u; l < DDGI_LOD_COUNT; l = l + 1u) {
    let cl = atomicLoad(&ddgi_indirect[DDGI_INDIR_COUNT_BASE + l]);
    let rc = select(0u, cl * rpp, cl > 0u);
    let cc = select(0u, cl * DDGI_COLLECT_THREADS, cl > 0u);
    if (l < lod) { ray_base = ray_base + rc; coll_base = coll_base + cc; }
    total_ray = total_ray + rc;
    total_coll = total_coll + cc;
  }
  atomicStore(&ddgi_indirect[DDGI_INDIR_RAYBASE_BASE + lod], ray_base);
  atomicStore(&ddgi_indirect[DDGI_INDIR_COLLBASE_BASE + lod], coll_base);
  atomicStore(&ddgi_indirect[DDGI_INDIR_RPP_BASE + lod], rpp);
  if (lod == 0u) {
    atomicStore(&ddgi_indirect[DDGI_INDIR_RAY_TOTAL], total_ray);
    atomicStore(&ddgi_indirect[DDGI_INDIR_COLL_TOTAL], total_coll);
    // 每探针 rpp 射线、每探针 320 纹素线程，WG 均为 64 → 两次 indirect dispatch，y=z=1
    let cb = DDGI_INDIR_CAST_BASE;
    atomicStore(&ddgi_args[cb + 0u], min((total_ray + 63u) / 64u, 65535u));
    atomicStore(&ddgi_args[cb + 1u], 1u);
    atomicStore(&ddgi_args[cb + 2u], 1u);
    let kb = DDGI_INDIR_COLL_BASE;
    atomicStore(&ddgi_args[kb + 0u], min((total_coll + 63u) / 64u, 65535u));
    atomicStore(&ddgi_args[kb + 1u], 1u);
    atomicStore(&ddgi_args[kb + 2u], 1u);
  }
}

// ============================================================================
// 阶段一 烘焙（嵌套级联：相机滚动补烘新入区 / 世界编辑时全量重烘）。
// 逐 cell 沿 4³ 分裂树 BFS 找「最大的全空叶」，探针放在其中心；同级里取更靠近
// cell 中心的空叶。全空 cell → 居中；全满 cell → 无探针。
// 输出 ddgi_cell[slot] = flags(bit0 探针存在 / bit1 cell 有体素) | cell 内归一化偏移。
// 每帧 sort 只读该记录，不再逐帧重算树 BFS。
// 增量：ddgi_cell_id[slot] 记录已烘焙的世界 cell 键，键未变的槽位直接跳过（续龄）；
// 相机滚动只影响发生滚动的那一级 LOD，其余级不动。
// ============================================================================

// 在 [cmin, cmin+cs) 内按 16³ → 4³ → 1³ 逐级找靠中心的空叶；返回 (p, found)。
// 三轮扫描惰性求值：先只找最近的空 16³，命中即返回；仅当整个 cell 都没有空 16³ 时才展开
// 4³、再兜底 1³。优先级与原实现一致（16³ 优于 4³），但避免了「每个混合 16³ 子块都展开 64
// 次 4³ 查询」——粗 LOD（cs=128 → n16=8）最坏是 512×64 次查询/cell，相机滚动触发 bake 时
// 这是主要开销。
fn ddgi_place_probe(g: Grid, cmin: vec3<i32>, cs: i32, center: vec3<f32>) -> vec4<f32> {
  let n16 = max(cs / 16, 1);
  // 轮 1：最近的「全空 16³ 子块」
  var b16_d2 = 1e30;
  var b16_p = vec3<f32>(0.0);
  var f16 = false;
  for (var k = 0; k < n16; k = k + 1) {
    for (var j = 0; j < n16; j = j + 1) {
      for (var i = 0; i < n16; i = i + 1) {
        let sub16 = cmin + vec3<i32>(i, j, k) * 16;
        if (ddgi_cell_state_sized(g, sub16, 16) == 0u) {
          let p = vec3<f32>(sub16) + vec3<f32>(8.0);
          let d2 = dot(p - center, p - center);
          if (d2 < b16_d2) { b16_d2 = d2; b16_p = p; f16 = true; }
        }
      }
    }
  }
  if (f16) { return vec4<f32>(b16_p, 1.0); }
  // 轮 2：无空 16³ → 在混合 16³ 内找最近的「全空 4³」
  var b4_d2 = 1e30;
  var b4_p = vec3<f32>(0.0);
  var f4 = false;
  for (var k = 0; k < n16; k = k + 1) {
    for (var j = 0; j < n16; j = j + 1) {
      for (var i = 0; i < n16; i = i + 1) {
        let sub16 = cmin + vec3<i32>(i, j, k) * 16;
        if (ddgi_cell_state_sized(g, sub16, 16) == 2u) {
          for (var kk = 0; kk < 4; kk = kk + 1) {
            for (var jj = 0; jj < 4; jj = jj + 1) {
              for (var ii = 0; ii < 4; ii = ii + 1) {
                let sub4 = sub16 + vec3<i32>(ii, jj, kk) * 4;
                if (ddgi_brick_state(g, sub4, 3u) == 0u) {
                  let p = vec3<f32>(sub4) + vec3<f32>(2.0);
                  let d2 = dot(p - center, p - center);
                  if (d2 < b4_d2) { b4_d2 = d2; b4_p = p; f4 = true; }
                }
              }
            }
          }
        }
      }
    }
  }
  if (f4) { return vec4<f32>(b4_p, 1.0); }
  // 兜底：mixed 16³ 内逐 1³ 找最近空体素（ddgi_leaf16）
  var b1_d2 = 1e30;
  var b1_p = vec3<f32>(0.0);
  var f1 = false;
  for (var k = 0; k < n16; k = k + 1) {
    for (var j = 0; j < n16; j = j + 1) {
      for (var i = 0; i < n16; i = i + 1) {
        let sub16 = cmin + vec3<i32>(i, j, k) * 16;
        if (ddgi_cell_state_sized(g, sub16, 16) == 2u) {
          let r = ddgi_leaf16(g, sub16, center);
          if (r.w > 0.5) {
            let d2 = dot(r.xyz - center, r.xyz - center);
            if (d2 < b1_d2) { b1_d2 = d2; b1_p = r.xyz; f1 = true; }
          }
        }
      }
    }
  }
  return vec4<f32>(b1_p, select(0.0, 1.0, f1));
}

@compute @workgroup_size(64)
fn ddgi_bake(@builtin(global_invocation_id) gid: vec3<u32>) {
  let slot = gid.x;
  if (slot >= u32(ddgi_u.misc.y)) { return; }
  let lod = ddgi_slot_lod_of(slot);
  let idx = slot - ddgi_lod_slot_base(lod);
  let cs = ddgi_u.lods[lod].origin.w;
  // 本槽位「当前」覆盖的世界 cell（世界锚定环面映射的逆）。
  let wc = ddgi_slot_world_cell(lod, idx);
  let cmin = wc * cs;
  // 增量判据：
  //   ① 相机滚动 → 只有「世界 cell 键变了」的槽位重烘（键相同即续用，保留 meta 续龄）；
  //   ② 世界编辑 → 脏区 AABB 内的 cell 强制重烘（脏区 = CPU 上传时改动 chunk 的合并包围盒，
  //      不再是整块清缓存，所以一次小编辑只重算受影响的 cell）。
  let cell_lo = vec3<f32>(cmin);
  let cell_hi = cell_lo + f32(cs);
  let dirty = ddgi_u.dirty_min.w > 0.5
    && all(cell_hi > ddgi_u.dirty_min.xyz)
    && all(cell_lo < ddgi_u.dirty_max.xyz);
  let wcell = cmin / cs;
  let prev = ddgi_cell_id[slot];
  if (!dirty && prev.w != 0 && all(prev.xyz == wcell)) { return; }
  ddgi_cell_id[slot] = vec4<i32>(wcell, 1);
  ddgi_meta[slot] = 0u; // 换 world cell / 落入脏区 → 重置 age/enabled/active（sort 当帧随后重建）
  let g = make_grid(0u);
  let st = ddgi_cell_state_sized(g, cmin, cs);
  // 物体（非主世界 volume）不在主世界 brick tree 里 → 单独判：cell 与任一物体相交就
  // 当作「有体素」，否则物体所在的 cell 会被判成纯空气（探针永不判活 → 图集恒 0）。
  let obj_hit = ddgi_box_hits_object(cell_lo, cell_hi);
  if (st == 1u) {
    // 全满：无探针，但仍标记「有体素」供邻接 cell 判活
    ddgi_cell[slot] = ddgi_rec_pack(0u, 1u, vec3<u32>(0u));
    return;
  }
  let occupied = select(0u, 1u, st != 0u || obj_hit);
  let center = vec3<f32>(cmin) + f32(cs) * 0.5;
  // 放置启发式（对齐 Douglas #23）：「取最大空子块的中心」——这保证探针与最近表面之间
  // **留出距离**（他说这正是深度/光照数据分辨率被充分利用的前提）。
  // 旧版加了一条捷径：只要 cell 中心不是实心就直接用中心 → 中心恰好落在薄壁旁的空侧时，
  // 探针就贴在表面上，前后判定落在临界值附近抖动，整面被判「探针在背面」而全剔掉。
  // 现在改为：中心必须落在一个**完全空的 16³ 子块**内才直接采用，否则交给
  // ddgi_place_probe 按「最大空叶 + 靠近中心」重新找。全空 cell 的最大空叶就是整格 →
  // 探针仍落在中心（与 Douglas「totally empty cell → probe right in the center」一致）。
  let center_v = vec3<i32>(center);
  let sub16 = cmin + ((center_v - cmin) / 16) * 16;
  let center_ok = sample_brickmap(g, center_v) == 0u
    && ddgi_cell_state_sized(g, sub16, 16) == 0u;
  var p = center;
  if (!center_ok) {
    let r = ddgi_place_probe(g, cmin, cs, center);
    if (r.w <= 0.5) { ddgi_cell[slot] = ddgi_rec_pack(0u, occupied, vec3<u32>(0u)); return; }
    p = r.xyz;
  }
  // 主世界的 place_probe 看不到物体体素：探针可能正落在物体内部（射线起点即实心，
  // 首命中 t≈0，辐亮度/深度全错）→ 推到物体 AABB 外。
  p = ddgi_push_out_of_objects(p);
  let u = clamp((p - vec3<f32>(cmin)) / f32(cs), vec3<f32>(0.0), vec3<f32>(1.0));
  let off_b = vec3<u32>(clamp(round(u * 255.0), vec3<f32>(0.0), vec3<f32>(255.0)));
  ddgi_cell[slot] = ddgi_rec_pack(1u, occupied, off_b);
}

// ============================================================================
// 阶段一 sort（Douglas pass #1：探针活跃性判定 + worklist 压缩）。
// 1D dispatch：每线程一个 (lod, 世界 cell) slot：
//   ① 读烘焙记录 ddgi_cell[slot]（探针是否存在 / 本 cell 是否有体素 / cell 内偏移）
//   ② 活跃 = 有探针 && (本 cell 有体素 || 6 邻接 cell 任一有体素 || 与非网格对齐物体 AABB 重叠)
//      且不落在更细 LOD 覆盖范围内（避免各 LOD 重复投线）
//   ③ can_skip_update：收敛 + 远离相机 + 非强制刷新帧 → 本帧不投线
//   ④ 幸存者 atomicAdd 进 per-LOD worklist（item = 探针世界坐标 + age/lod）
//   ⑤ 全 slot 写 meta（age/enabled）与 slot_pos
// ============================================================================
// ============================================================================
// 非主世界 volume（物体）辅助。
// 物体的体素只在 grid_descs[i≥1]（各自独立的 brick tree + 变换）里，主世界 brick tree
// 完全没有它们。旧版烘焙/判活只看 make_grid(0)，于是物体所在的 cell 被判成「纯空气」：
// 探针放在物体内部、且永不判活 → 图集恒 0 → 物体整面没有 GI（Domain 有壳色、
// Probe 全白 = 权重和>0 但辐照度≈0、GI 全黑）。
// ============================================================================
/// 世界 AABB [lo, hi) 是否与任一物体的世界 AABB 相交（cell 粒度判占用/判活）
fn ddgi_box_hits_object(lo: vec3<f32>, hi: vec3<f32>) -> bool {
  for (var i = 1u; i < g.grid_count; i = i + 1u) {
    let d = grid_descs[i];
    if (all(hi > d.aabb_min.xyz) && all(lo < d.aabb_max.xyz)) { return true; }
  }
  return false;
}
/// 把点推到所有物体 AABB 之外：沿「离得最近的那个面」推 1 voxel。
/// 探针落在物体内时射线起点就在实心里（首命中 t≈0）→ 该探针的辐亮度/深度全错。
fn ddgi_push_out_of_objects(p0: vec3<f32>) -> vec3<f32> {
  var p = p0;
  for (var i = 1u; i < g.grid_count; i = i + 1u) {
    let d = grid_descs[i];
    let mn = d.aabb_min.xyz;
    let mx = d.aabb_max.xyz;
    if (all(p >= mn) && all(p < mx)) {
      let dlo = p - mn;
      let dhi = mx - p;
      let near_lo = min(dlo, dhi);
      if (near_lo.x <= near_lo.y && near_lo.x <= near_lo.z) {
        p.x = select(mn.x - 1.0, mx.x + 1.0, dhi.x < dlo.x);
      } else if (near_lo.y <= near_lo.z) {
        p.y = select(mn.y - 1.0, mx.y + 1.0, dhi.y < dlo.y);
      } else {
        p.z = select(mn.z - 1.0, mx.z + 1.0, dhi.z < dlo.z);
      }
    }
  }
  return p;
}

/// 邻接 cell 是否有体素（cell 用**世界 cell 号**；不在本级窗口内视为无）
fn ddgi_neighbor_occupied(lod: u32, wc: vec3<i32>) -> bool {
  if (!ddgi_cell_in_window(lod, wc)) { return false; }
  return (ddgi_cell[ddgi_slot(lod, wc)] & DDGI_REC_OCCUPIED) != 0u;
}

@compute @workgroup_size(64)
fn ddgi_sort(@builtin(global_invocation_id) gid: vec3<u32>) {
  let slot = gid.x;
  if (slot >= u32(ddgi_u.misc.y)) { return; }
  let lod = ddgi_slot_lod_of(slot);
  let idx = slot - ddgi_lod_slot_base(lod);
  let rec = ddgi_cell[slot];
  let cs = ddgi_u.lods[lod].origin.w;
  let wc = ddgi_slot_world_cell(lod, idx);
  let cmin = wc * cs;

  let enabled = (rec & DDGI_REC_ENABLED) != 0u;
  if (!enabled) {
    ddgi_meta[slot] = 0u;
    ddgi_slot_pos[slot] = vec4<f32>(0.0, 0.0, 0.0, -1.0);
    return;
  }
  let off = vec3<f32>(ddgi_rec_off(rec)) / 255.0;
  let probe_pos = vec3<f32>(cmin) + off * f32(cs);

  // ---- 活跃判定：本 cell / 6 邻接 cell 有体素（含物体），或与非网格对齐物体 AABB 重叠 ----
  var near = (rec & DDGI_REC_OCCUPIED) != 0u;
  if (!near) {
    near = ddgi_neighbor_occupied(lod, wc + vec3<i32>(-1, 0, 0))
      || ddgi_neighbor_occupied(lod, wc + vec3<i32>(1, 0, 0))
      || ddgi_neighbor_occupied(lod, wc + vec3<i32>(0, -1, 0))
      || ddgi_neighbor_occupied(lod, wc + vec3<i32>(0, 1, 0))
      || ddgi_neighbor_occupied(lod, wc + vec3<i32>(0, 0, -1))
      || ddgi_neighbor_occupied(lod, wc + vec3<i32>(0, 0, 1));
  }
  if (!near) {
    // 物体：主世界 brick tree 里没有它的体素，必须用 grid_descs 的世界 AABB 判。
    // 旧版读的是从未被写入的 ddgi_objects（CPU 侧 u.params.w 恒 0）→ 这段是死代码，
    // 物体所在 cell 永不判活 → 图集恒 0 → 物体整面没有 GI。
    near = ddgi_box_hits_object(
      vec3<f32>(cmin),
      vec3<f32>(cmin) + vec3<f32>(f32(cs)),
    );
  }
  // ---- age 继承（同 slot 且 world cell 未变 → 续龄；换 cell 时 bake 已清零）----
  var age = ddgi_meta_age(ddgi_meta[slot]);
  // 判活条件严格对齐 Douglas #23：有探针 && (本 cell 或 6 邻接有体素 || 与物体 AABB 相交)。
  // 旧版额外加了「不落在更细一级 LOD 盒内」的排除，用来省重复投线 —— 但它的代价是：
  // 粗级的探针只要落点被细盒包住就永不投线 → 图集恒 0 → 壳边界附近整圈像素采到空数据
  // （Probe 青色 = 未判活）。字幕里没有这条规则，去掉。
  let is_active = near;

  if (is_active) {
    // ---- can_skip_update：收敛 + 远离相机 + 非强制刷新帧 → 本帧不投线 ----
    let frame = u32(ddgi_u.params.x);
    let converged = age >= DDGI_SKIP_AGE[lod];
    let far = length(probe_pos - view_u.cam_pos_voxel.xyz) > f32(cs) * DDGI_SKIP_DIST_CELLS;
    let due = (frame + slot) % DDGI_REFRESH_PERIOD[lod] == 0u;
    if (!(converged && far && !due)) {
      age = min(age + 1u, DDGI_AGE_MAX);
      let wslot = atomicAdd(&ddgi_indirect[DDGI_INDIR_COUNT_BASE + lod], 1u);
      // worklist 是**压缩后**的列表，它的下标（wslot）只是排名，不是 cell 下标。
      // 低 8 位 age、中 16 位「探针在本级网格里的 cell 下标」、高 8 位 lod ——
      // collect 必须用 cell 下标才能把数据写进正确的图集纹素。
      let packed = age | (idx << 8u) | (lod << 24u);
      ddgi_worklist[ddgi_lod_slot_base(lod) + wslot] =
        vec4<f32>(probe_pos, bitcast<f32>(packed));
    }
  }

  ddgi_meta[slot] = ddgi_meta_pack(age, true, is_active);
  ddgi_slot_pos[slot] = vec4<f32>(probe_pos, f32(lod));
}

fn ddgi_irr_fetch(id: u32, tx: i32, ty: i32) -> vec3<f32> {
  let c = ddgi_irr_coord(id, u32(tx), u32(ty));
  return textureLoad(ddgi_irr, vec2<i32>(vec2<u32>(c.y, c.z)), i32(c.x), 0).xyz;
}
fn ddgi_irr_sample(id: u32, d: vec3<f32>) -> vec3<f32> {
  let s = f32(DDGI_IRR_TEXELS);
  let e = ddgi_oct_encode(d) * 0.5 + vec2<f32>(0.5);
  let g2 = e * s - vec2<f32>(0.5);
  let g0 = floor(g2);
  let f = g2 - g0;
  let x0 = clamp(i32(g0.x), 0, 7);
  let y0 = clamp(i32(g0.y), 0, 7);
  let x1 = clamp(i32(g0.x) + 1, 0, 7);
  let y1 = clamp(i32(g0.y) + 1, 0, 7);
  let c00 = ddgi_irr_fetch(id, x0, y0);
  let c10 = ddgi_irr_fetch(id, x1, y0);
  let c01 = ddgi_irr_fetch(id, x0, y1);
  let c11 = ddgi_irr_fetch(id, x1, y1);
  return mix(mix(c00, c10, vec3<f32>(f.x)), mix(c01, c11, vec3<f32>(f.x)), vec3<f32>(f.y));
}
fn ddgi_depth_sample(id: u32, d: vec3<f32>) -> f32 {
  let s = f32(DDGI_DEPTH_TEXELS);
  let e = ddgi_oct_encode(d) * 0.5 + vec2<f32>(0.5);
  let g2 = floor(e * s);
  let x = clamp(i32(g2.x), 0, 15);
  let y = clamp(i32(g2.y), 0, 15);
  let c = ddgi_depth_coord(id, u32(x), u32(y));
  return textureLoad(ddgi_depth, vec2<i32>(vec2<u32>(c.y, c.z)), i32(c.x), 0).x;
}
// p 是否落在 LOD(lod) 的「壳」内：在 LOD(lod) 盒内、且不在更细一级 LOD 盒内。
// 嵌套级联下任意点恰好属于一级 → 每像素只采样一个 LOD（也保证只在探针真正活跃的区域取数）。
fn ddgi_lod_contains(lod: u32, p: vec3<f32>) -> bool {
  let L = ddgi_u.lods[lod];
  let o = vec3<f32>(L.origin.xyz);
  let ext = vec3<f32>(L.dims.xyz) * f32(L.origin.w);
  if (!(all(p >= o) && all(p < o + ext))) { return false; }
  if (lod == 0u) { return true; }
  let Lo = ddgi_u.lods[lod - 1u];
  let lo_f = vec3<f32>(Lo.origin.xyz);
  let hi_f = lo_f + vec3<f32>(Lo.dims.xyz) * f32(Lo.origin.w);
  return !(all(p >= lo_f) && all(p < hi_f));
}
// clamp_cells = true：把越界的世界 cell 号 clamp 进本级窗口（级联之外的退化采样）。
// 级联只覆盖相机周围有限体积；越界时若直接判空，相机上方/远方的表面会整片无 GI。
// clamp 之后用最靠近的可用探针给一个粗粒度估计 —— 空间上仍随位置变化，只是精度粗。
fn ddgi_sample_lod(p: vec3<f32>, n: vec3<f32>, lod: u32, clamp_cells: bool) -> vec4<f32> {
  let L = ddgi_u.lods[lod];
  let cs = f32(L.origin.w);
  // 探针的**标称位置在 cell 中心**（bake 取 center = cmin + cs*0.5；中心是实心时才搬到
  // 最近空叶），所以相邻两个探针位于 (c+0.5)·cs 与 (c+1.5)·cs —— 三线性插值必须在这两者
  // 之间做，即先把 cell 坐标**平移半格**再 floor。
  // 旧版直接 floor((p-o)/cs)：取到的是「包含 p 的 cell 与它 +1 的 cell」，这两个探针整体
  // 比 p 所在区间偏 +半格。后果按轴向最明显：对朝向 -x/-y/-z 的面，当 p 落在 cell 的
  // 前半格时，沿该轴的**两个候选探针都在 p 的正方向一侧** → 8 个角全部落在表面背面 →
  // wn 背向剔除把它们全剔光 → 整块按 cell 分块的无 GI 区域（Probe 品红、GI 全黑），
  // 且与法线正确与否无关（这就是换逐面法线后表现完全不变的原因）。
  // 采样点沿法线再外推一点（见 DDGI_BIAS_CELLS）：让采样点明确落在表面外侧，前后的
  // 判定不落在临界值上。调用处已用 0.5 体素把点从「命中体素中心」推到体素表面。
  let ps = p + n * (cs * DDGI_BIAS_CELLS);
  // 直接用**世界 cell 号**（不再经过窗口原点）：槽位是「世界 cell 对网格维度取模」，
  // 所以采样侧与烘焙/判活侧用的是同一套世界身份。
  let wc_f = ps / cs - vec3<f32>(0.5);
  let wc0 = vec3<i32>(floor(wc_f));
  let fr = wc_f - floor(wc_f);
  let o_cell = ddgi_origin_cell(lod);
  let di = vec3<i32>(L.dims.xyz);
  var total = vec3<f32>(0.0);
  var wsum = 0.0;
  ddgi_dbg_rej = vec4<u32>(0u, 0u, 0u, 0u);
  for (var iz = 0; iz < 2; iz = iz + 1) {
    for (var iy = 0; iy < 2; iy = iy + 1) {
      for (var ix = 0; ix < 2; ix = ix + 1) {
        var wc = wc0 + vec3<i32>(ix, iy, iz);
        if (clamp_cells) {
          wc = clamp(wc, o_cell, o_cell + di - vec3<i32>(1));
        } else if (!ddgi_cell_in_window(lod, wc)) {
          ddgi_dbg_rej.x = ddgi_dbg_rej.x + 1u;
          continue;
        }
        let slot = ddgi_slot(lod, wc);
        let mw = ddgi_meta[slot];
        if ((mw & DDGI_META_ENABLED) == 0u) { ddgi_dbg_rej.x = ddgi_dbg_rej.x + 1u; continue; }
        // 本帧没被判活（sort 的 near=false 或落在更细一级盒内）→ 该探针从不进 worklist，
        // 图集纹素恒为 0。它必须被排除：否则会以 0 辐照度 + 满权重参与平均，把整片 GI
        // 拉向黑。不能指望下面的 depth 闸门兜住它 —— 它连 depth 都是 0，会被判成「无遮挡」。
        if ((mw & DDGI_META_ACTIVE) == 0u) { ddgi_dbg_rej.w = ddgi_dbg_rej.w + 1u; continue; }
        // 年龄太小（刚换过世界 cell / 刚编辑）→ 图集里是旧位置或未覆盖的读数，跳过
        if (ddgi_meta_age(mw) < DDGI_MIN_SAMPLE_AGE) { ddgi_dbg_rej.w = ddgi_dbg_rej.w + 1u; continue; }
        let wx = select(1.0 - fr.x, fr.x, ix == 1);
        let wy = select(1.0 - fr.y, fr.y, iy == 1);
        let wz = select(1.0 - fr.z, fr.z, iz == 1);
        let wtri = wx * wy * wz;
        if (wtri <= 1e-6) { continue; }
        let probe = ddgi_slot_pos[slot].xyz;
        let to = ps - probe;
        let dist = length(to);
        let dir = to / max(dist, 1e-4);
        let wn = clamp(dot(n, -dir) / DDGI_NORMAL_BIAS, 0.0, 1.0);
        // 正常路径按标准 DDGI 剔除背向探针；退化路径（clamp_cells）放宽 wn。
        // 否则「整个面法线朝向不利」（探针全落在背面，例如薄墙对面那侧）会让整面全被剔除、
        // 表现为一整面没有 GI 而被周围有 GI 的面包围。遮挡仍由 depth map（wd）负责。
        if (!clamp_cells && wn <= 0.0) { ddgi_dbg_rej.y = ddgi_dbg_rej.y + 1u; continue; }
        let wn_w = select(wn, max(wn, 0.25), clamp_cells);
        let dtex = ddgi_depth_sample(slot, dir);
        // 该方向没有任何命中记录（纹素从未被写过，或刚被 collect 显式清零）→ **无数据**，
        // 不是「无遮挡」：判为不可见并计入无数据闸门。旧版把「无数据」当「无遮挡」→ 新鲜
        // 探针以 0 辐照度满权重参与平均 → 大片「白（辐照度≈0）↔绿」翻转（动态闪烁）。
        if (dtex <= 0.0) { ddgi_dbg_rej.w = ddgi_dbg_rej.w + 1u; continue; }
        // 遮挡判定：容差随距离放大（max(cs*0.25, dist*3%)），过渡带 ±1.0·bias。
        // 深度图只有 16×16 纹素（≈11°），dtex 是该纹素的余弦加权平均，且每帧射线方向重随机
        // → dtex 逐帧抖动；容差太紧（LOD0 仅 16cm）会让卡在阈值上的角逐帧翻转（静态闪烁）。
        let dep_bias = max(cs * 0.25, dist * 0.03);
        let wd = clamp((dtex - dist) / dep_bias + 1.0, 0.0, 1.0);
        if (wd <= 0.0) { ddgi_dbg_rej.z = ddgi_dbg_rej.z + 1u; continue; }
        let irr = ddgi_irr_sample(slot, n);
        // 该方向的辐照度纹素从未被写过 → 同样是「无数据」，不能以 0 参与平均（否则唯一被
        // 接受的角若是空的，平均值就是 0）。已写入的纹素恒 > 0：collect 里
        // radiance ≥ albedo·sky·DDGI_BASE_AMBIENT > 0，所以「≈0」可安全当作「无数据」。
        if (max(irr.x, max(irr.y, irr.z)) < DDGI_TEXEL_MIN_WEIGHT) {
          ddgi_dbg_rej.w = ddgi_dbg_rej.w + 1u;
          continue;
        }
        let w = wtri * wn_w * wd;
        total = total + irr * w;
        wsum = wsum + w;
      }
    }
  }
  let ok = wsum >= 1e-4;
  let avg = select(vec3<f32>(0.0), total / max(wsum, 1e-6), ok);
  // 权重和 > 0 但辐照度 ≈ 0 → 探针存在、过了全部闸门，只是图集纹素从未被 collect 写过。
  // 必须与「被剔除」区分开：两者的着色结果都是黑，但根因完全不同。
  ddgi_dbg_zero = select(0.0, 1.0, ok && max(max(abs(avg.x), abs(avg.y)), abs(avg.z)) < 1e-5);
  return vec4<f32>(select(vec3<f32>(0.0), total / wsum, ok), wsum);
}
var<private> ddgi_dbg_wsum: f32;
// 0 = 没有任何 LOD 盒包含该像素；lod+1 = 该像素落在这一级的「壳」内。
// 语义是**纯几何包含**，不被采样成功与否影响（旧版在 fallback 里把它清零，导致
// 「越界」和「壳内但无数据」在 Domain 模式里都是绿色，无法区分）。
var<private> ddgi_dbg_dom: f32;
// 1 = 本像素走了退化 fallback（clamp 到最粗一级）
var<private> ddgi_dbg_fb: f32;
// 1 = 权重和 > 0 但辐照度 ≈ 0（探针在、闸门过、图集空）
var<private> ddgi_dbg_zero: f32;
var<private> ddgi_dbg_rej: vec4<u32>;
var<private> ddgi_dbg_rej_out: vec4<u32>;
fn ddgi_sample(p: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
  var in_cascade = false;
  var shell_lod = DDGI_LOD_COUNT;
  for (var lod = 0u; lod < DDGI_LOD_COUNT; lod = lod + 1u) {
    if (ddgi_lod_contains(lod, p)) {
      in_cascade = true;
      shell_lod = lod;
      let r = ddgi_sample_lod(p, n, lod, false);
      ddgi_dbg_rej_out = ddgi_dbg_rej;
      ddgi_dbg_wsum = r.w;
      ddgi_dbg_dom = f32(lod) + 1.0;
      ddgi_dbg_fb = 0.0;
      if (r.w >= 1e-4) {
        return r.xyz;
      }
      // 嵌套级联下 p 至多落在一级的壳内，无需继续向上试
      break;
    }
  }
  // 壳内 8 个角全被剔除（或像素在级联之外）→ **逐级退到更粗的 LOD 再试**，而不是直接
  // clamp 到最粗一级的边界：
  //   ① 粗一级的探针间距更大、其标称位置与 p 的切平面关系不同，往往还有可用数据；
  //   ② clamp 到最粗一级的边界时，那圈边界探针远离几何、基本判不了活（图集恒 0）——
  //      这正是之前「整片发黑」的来源，而不是壳本身有问题。
  // 每级仍然照常过全部门（未投线的探针照样排除），所以不会因此漏光。
  // ddgi_dbg_dom 不再清零 —— 它如实标出「该像素根本不在任何级联盒内」（绿）。
  var start = 0u;
  if (in_cascade) { start = shell_lod + 1u; }
  for (var lod = start; lod < DDGI_LOD_COUNT; lod = lod + 1u) {
    let r = ddgi_sample_lod(p, n, lod, true);
    if (r.w >= 1e-4) {
      if (!in_cascade) {
        // 不在任何壳内：退化结果与剔除统计都归它
        ddgi_dbg_fb = 1.0;
        ddgi_dbg_wsum = r.w;
        ddgi_dbg_rej_out = ddgi_dbg_rej;
      } else {
        // 壳没给出数据，但更粗一级 LOD 兜住了 → GI 有效（只是精度粗）。单独标记，Probe
        // 模式用另一种颜色表示「粗级兜底」，而不是当成异常报警 —— 相机滚动时新进入窗口的
        // 那条带就是这种情况。
        ddgi_dbg_fb = 2.0;
        ddgi_dbg_wsum = r.w;
      }
      return r.xyz;
    }
  }
  // 全部失败：in_cascade 时**保留壳的剔除统计**（不被退化调用覆盖），否则 Probe 模式永远
  // 显示退化路径的直方图（那圈探针本来就全是 near=false），定位不到壳为什么失败。
  if (!in_cascade) {
    ddgi_dbg_fb = 1.0;
    ddgi_dbg_rej_out = ddgi_dbg_rej;
  }
  return vec3<f32>(0.0);
}

// ============================================================================
// 阶段二 cast（Douglas pass #2）：worklist 里的活跃探针各投 rpp 条射线。
// 命中体素 → 取该处上一帧 GI 乘 albedo 得出射辐亮度；命中天空 → 天空色。
// 输出 ddgi_samples（方向 + 命中距离）/（辐亮度）。
// 线程映射：全局 tid = ray_base[lod] + probe_idx × rpp + ray（ray_base 由 ddgi_seal 算）。
// 一次 dispatch 覆盖全部 LOD —— 不再依赖 indirect 的 y 维传 LOD。
// ============================================================================
@compute @workgroup_size(64)
fn ddgi_cast(@builtin(global_invocation_id) gid: vec3<u32>) {
  let tid = gid.x;
  if (tid >= atomicLoad(&ddgi_indirect[DDGI_INDIR_RAY_TOTAL])) { return; }
  // 全局 tid → LOD：ray_base 单调递增，取最后一个 base ≤ tid 的那级
  var lod = 0u;
  for (var l = 1u; l < DDGI_LOD_COUNT; l = l + 1u) {
    if (tid >= atomicLoad(&ddgi_indirect[DDGI_INDIR_RAYBASE_BASE + l])) { lod = l; }
  }
  let rpp = atomicLoad(&ddgi_indirect[DDGI_INDIR_RPP_BASE + lod]);
  if (rpp == 0u) { return; }
  let local = tid - atomicLoad(&ddgi_indirect[DDGI_INDIR_RAYBASE_BASE + lod]);
  let probe_idx = local / rpp;
  let ray = local % rpp;
  let slot = ddgi_lod_slot_base(lod) + probe_idx;
  let probe_pos = ddgi_worklist[slot].xyz;
  let frame = u32(ddgi_u.params.x);
  // RNG 种子用探针**自己的 cell 下标**（而非 worklist 排名）：排名每帧会变，
  // 会让同一个探针的射线方向逐帧跳变、叠加噪声。时间维的随机性由 frame 提供。
  let cell_idx = (bitcast<u32>(ddgi_worklist[slot].w) >> 8u) & 0xFFFFu;

  let dir = ddgi_ray_dir(ddgi_lod_slot_base(lod) + cell_idx, frame, ray, rpp);
  let sh = trace_scene(probe_pos + dir * DDGI_RAY_BIAS, dir, DDGI_T_MAX, 0.0, 3u);
  var radiance = sky_rgb();
  var dist = DDGI_T_MAX;
  if (sh.uh.hit) {
    dist = sh.uh.t;
    let alb = palette_albedo(sh.palette_base, sh.uh.pal);
    let hit_p = probe_pos + dir * dist;
    let n = sh.uh.n;
    // 入射辐照度 E → 出射辐亮度 L = albedo·E/π（与着色侧 col += albedo·E/π 同约定）。
    // 加一项 albedo·sky·BASE_AMBIENT 作下限，避免首帧 GI 全 0 时反馈回路死锁在 0。
    let gi = ddgi_sample(hit_p + n * 0.5, n);
    radiance = alb * (gi / DDGI_PI + sky_rgb() * DDGI_BASE_AMBIENT);
  }
  // 样本下标 = 全局射线编号（tid 本身就是「该 LOD 起始射线号 + 本探针号×rpp + 射线号」）
  let si = tid * 2u;
  ddgi_samples[si] = vec4<f32>(dir, dist);
  ddgi_samples[si + 1u] = vec4<f32>(radiance, 1.0);
}

// ============================================================================
// 阶段三 collect（Douglas pass #3）：射线样本按「样本方向 · 纹素方向」余弦加权积成图集。
// 每探针 DDGI_COLLECT_THREADS 个线程：texel < 64 → irradiance 8×8；否则 depth 16×16。
// 与上一帧做时域混合（hysteresis）；探针刚唤醒（age ≤ 1）时直接写入，避免残留旧世界位置的数据。
// 读 ddgi_irr/ddgi_depth（BG4 采样侧），写 ddgi_irr_out/ddgi_depth_out（BG5 写入侧）。
// ============================================================================
// 一次 dispatch 覆盖全部 LOD：全局 tid → coll_base[lod] + probe_idx × 320 + texel。
// ============================================================================
@compute @workgroup_size(64)
fn ddgi_collect(@builtin(global_invocation_id) gid: vec3<u32>) {
  let tid = gid.x;
  if (tid >= atomicLoad(&ddgi_indirect[DDGI_INDIR_COLL_TOTAL])) { return; }
  var lod = 0u;
  for (var l = 1u; l < DDGI_LOD_COUNT; l = l + 1u) {
    if (tid >= atomicLoad(&ddgi_indirect[DDGI_INDIR_COLLBASE_BASE + l])) { lod = l; }
  }
  let rpp = atomicLoad(&ddgi_indirect[DDGI_INDIR_RPP_BASE + lod]);
  if (rpp == 0u) { return; }
  let local = tid - atomicLoad(&ddgi_indirect[DDGI_INDIR_COLLBASE_BASE + lod]);
  let probe_idx = local / DDGI_COLLECT_THREADS;
  let texel = local % DDGI_COLLECT_THREADS;
  let slot = ddgi_lod_slot_base(lod) + probe_idx;
  let packed = bitcast<u32>(ddgi_worklist[slot].w);
  let age = packed & 0xFFu;
  // 图集纹素按「探针的 cell 下标」寻址，而 slot 只是压缩列表里的排名 ——
  // 用错会把整张图集写成错位置换（表现：大面积发黑、只有局部有 GI）。
  let atlas_slot = ddgi_lod_slot_base(lod) + ((packed >> 8u) & 0xFFFFu);
  // 该探针的首条射线在全局射线空间里的编号 = 本 LOD 起始射线号 + 探针号×rpp
  let si = atomicLoad(&ddgi_indirect[DDGI_INDIR_RAYBASE_BASE + lod]) + probe_idx * rpp;
  let snap = age <= 1u; // 刚唤醒 → 不做 hysteresis，直接写入

  if (texel < DDGI_IRR_TEXELS * DDGI_IRR_TEXELS) {
    // ---- irradiance：E = π · Σ(w·L) / Σw，w = max(0, dot(texel_dir, ray_dir)) ----
    let tx = texel % DDGI_IRR_TEXELS;
    let ty = texel / DDGI_IRR_TEXELS;
    let td = ddgi_oct_texel_dir(tx, ty, DDGI_IRR_TEXELS);
    var sum = vec3<f32>(0.0);
    var wsum = 0.0;
    for (var i = 0u; i < rpp; i = i + 1u) {
      let s = ddgi_samples[(si + i) * 2u];
      let w = max(0.0, dot(td, s.xyz));
      sum = sum + w * ddgi_samples[(si + i) * 2u + 1u].xyz;
      wsum = wsum + w;
    }
    if (wsum > 1e-5) {
      let irr_new = DDGI_PI * sum / wsum;
      let cc = ddgi_irr_coord(atlas_slot, tx, ty);
      let idx = vec2<i32>(vec2<u32>(cc.y, cc.z));
      let prev = textureLoad(ddgi_irr, idx, i32(cc.x), 0).xyz;
      let a = select(DDGI_ALPHA, 1.0, snap);
      textureStore(ddgi_irr_out, idx, i32(cc.x), vec4<f32>(mix(prev, irr_new, a), 1.0));
    } else if (snap) {
      // 新（重）烘的探针：本帧没有射线覆盖到的方向 → **显式清零**。
      // 不清零的话这里留着的是「该槽位上一任世界 cell」的读数（槽位是世界锚定环面映射，
      // 相机滚动时新进入窗口的格子会复用刚离开格子的槽位）→ 会被当成有效数据使用。
      // 清零后采样侧的两条「无数据」判据（dtex<=0 / irr≈0）才能精确识别它。
      let cc = ddgi_irr_coord(atlas_slot, tx, ty);
      textureStore(ddgi_irr_out, vec2<i32>(vec2<u32>(cc.y, cc.z)), i32(cc.x), vec4<f32>(0.0));
    }
  } else {
    // ---- depth：余弦加权平均命中距离（未命中 = T_MAX → 不产生遮挡）----
    let t = texel - DDGI_IRR_TEXELS * DDGI_IRR_TEXELS;
    let tx = t % DDGI_DEPTH_TEXELS;
    let ty = t / DDGI_DEPTH_TEXELS;
    let td = ddgi_oct_texel_dir(tx, ty, DDGI_DEPTH_TEXELS);
    var dsum = 0.0;
    var dwsum = 0.0;
    for (var i = 0u; i < rpp; i = i + 1u) {
      let s = ddgi_samples[(si + i) * 2u];
      let w = max(0.0, dot(td, s.xyz));
      dsum = dsum + w * s.w;
      dwsum = dwsum + w;
    }
    if (dwsum > 1e-5) {
      let dep_new = dsum / dwsum;
      let cc = ddgi_depth_coord(atlas_slot, tx, ty);
      let idx = vec2<i32>(vec2<u32>(cc.y, cc.z));
      let prev = textureLoad(ddgi_depth, idx, i32(cc.x), 0).x;
      let a = select(DDGI_DEPTH_ALPHA, 1.0, snap);
      // r32float 存储纹素的 store 值类型是 vec4<f32>（naga 校验要求），只取 .x 通道
      textureStore(ddgi_depth_out, idx, i32(cc.x), vec4<f32>(mix(prev, dep_new, a), 0.0, 0.0, 0.0));
    } else if (snap) {
      // 同 irradiance：新（重）烘探针未被覆盖的方向显式清零，避免残留上一任世界 cell 的深度
      let cc = ddgi_depth_coord(atlas_slot, tx, ty);
      textureStore(
        ddgi_depth_out,
        vec2<i32>(vec2<u32>(cc.y, cc.z)),
        i32(cc.x),
        vec4<f32>(0.0, 0.0, 0.0, 0.0),
      );
    }
  }
}

@compute @workgroup_size(64)
fn probe_viz_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dot_size = u32(view_u.probe_viz_params.y);
  let id = gid.x;
  // x = 世界空间探针总槽数（Rust 每帧写入）
  if (id >= u32(view_u.probe_viz_params.x)) {
    return;
  }
  let lod = ddgi_slot_lod_of(id);
  let sel = u32(view_u.probe_viz_params.z);
  if (sel != 0u && lod != sel - 1u) {
    return;
  }
  // 只可视化「活跃」探针（邻域检测 + 非更细 LOD 覆盖），而非所有烘焙探针
  if ((ddgi_meta[id] & DDGI_META_ACTIVE) == 0u) {
    return;
  }
  let pos = ddgi_slot_pos[id];
  let dot_col = vec4<f32>(1.0, 0.85, 0.0, 1.0);
  let clip = view_u.view_proj * vec4<f32>(pos.xyz, 1.0);
  if (clip.w <= 0.0) {
    return;
  }
  let ndc = clip.xyz / clip.w;
  if (ndc.x < -1.0 || ndc.x > 1.0 || ndc.y < -1.0 || ndc.y > 1.0 || ndc.z < 0.0 || ndc.z > 1.0) {
    return;
  }
  let to_probe = pos.xyz - view_u.cam_pos_voxel.xyz;
  let dist = length(to_probe);
  let hit = trace_scene(view_u.cam_pos_voxel.xyz, to_probe / dist, dist, 0.0, 3u);
  if (hit.uh.hit && hit.uh.t < dist - 0.5) {
    return;
  }
  let dims = vec2<f32>(textureDimensions(out_tex));
  let px = vec2<f32>((ndc.x * 0.5 + 0.5) * dims.x, (1.0 - (ndc.y * 0.5 + 0.5)) * dims.y);
  let ci = vec2<i32>(px);
  let half = i32(dot_size) / 2;
  for (var dy = -half; dy <= half; dy++) {
    for (var dx = -half; dx <= half; dx++) {
      let p = vec2<i32>(ci.x + dx, ci.y + dy);
      if (p.x >= 0 && p.y >= 0 && u32(p.x) < u32(dims.x) && u32(p.y) < u32(dims.y)) {
        textureStore(out_tex, vec2<u32>(p), dot_col);
      }
    }
  }
}


