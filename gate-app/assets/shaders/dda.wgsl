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
// @binding(4)/(5)：光照场（Douglas #15 AO fill + 「体素即光源」发光密度 ε 共用一张纹理）。
// 16-voxel cell 网格，**相机中心 + 世界锚定槽位**：纹素下标 = 世界 cell mod dims（见
// `light_field_uv`），相机滚动只换新进窗口的那条带，纹素身份不整幅失效。
// Rgba16Unorm：.rgb = 发光密度 ε、.a = AO fill；采样即硬件三线性插值
// （Douglas 原文："a single Hardware filtered texture read"）。
@group(1) @binding(4) var light_tex: texture_3d<f32>;
@group(1) @binding(5) var light_samp: sampler;

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

/// palette 条目的自发光强度（0..1）。word1 低字节（wire.rs `pack_palette_entry` 同布局）。
/// 发光体素是这个世界的**光源本体**：着色时 radiance 直出，不吃方向/阴影（见 lighting.rs 注释）。
fn palette_emissive(palette_base: u32, pal: u32) -> f32 {
  return f32(b_palette[palette_base + pal * 2u + 1u] & 0xFFu) / 255.0;
}

// 天空纯色（miss 像素输出 + 探针射线 miss 端点；Minecraft 白天平原 #78A7FF）
// sRGB → linear（与 palette_albedo 同空间；光照全程 linear，输出经 linear_to_srgb 还原）
fn sky_rgb() -> vec3<f32> {
  return srgb_to_linear(light_u.sky_color.xyz);
}

// ============================================================================
// 光照场（Douglas #15 的 AO fill + 「体素即光源」的发光密度 ε，共用一张 3D 纹理）。
//
// 形态与 DDGI 级联同构，因此是可流式的：
//   · cell = LIGHT_FIELD_CELL 体素（= Douglas #15 的 16³ 填充率网格），dims = 32³
//   · 原点 = align_down(相机 − dims/2·cell, cell)：相机恒在盒中心（偏差 < 1 cell）
//   · 槽位 = 世界 cell mod dims（**世界锚定**）：相机滚动只换「新进窗口那条带」的纹素，
//     其余纹素保持自己的世界身份 → 纹素不会因相机移动而整幅失效
// 与 Rust `brickmap/upload.rs::light_field_origin_cell` 是同一个式子，必须逐字一致。
// ============================================================================
const LIGHT_FIELD_CELL: f32 = 16.0;
const LIGHT_FIELD_DIM: i32 = 32;
/// 场窗口原点（世界 voxel，按 cell 对齐；相机恒在盒中心）。
fn light_field_origin_voxel() -> vec3<f32> {
  let half = f32(LIGHT_FIELD_DIM / 2) * LIGHT_FIELD_CELL;
  return floor((view_u.cam_pos_voxel.xyz - vec3<f32>(half)) / LIGHT_FIELD_CELL) * LIGHT_FIELD_CELL;
}
/// 世界 voxel 位置 → 光照场纹理 uv。
/// 连续 cell 坐标卷绕到 [0, dim) 后归一化。纹素 c 存的是**整个 cell** [16c, 16c+16) 的
/// 均值（CPU tally 如此），故采样位置取 `r` 本身：cell 中心（r = c+0.5）恰好落在纹素 c 的
/// 中心（硬件三线性在此取到纹素原值），cell 边界处才与相邻纹素各半 —— 与 tally 语义一致。
fn light_field_uv(p_voxel: vec3<f32>) -> vec3<f32> {
  let d = vec3<f32>(f32(LIGHT_FIELD_DIM));
  let cc = p_voxel / LIGHT_FIELD_CELL;
  let r = cc - floor(cc / d) * d;
  return r / d;
}
/// 点是否落在场窗口内。窗口外不采样：环面卷绕会把远处几何映到错误纹素。
fn light_field_contains(p_voxel: vec3<f32>) -> bool {
  let rel = p_voxel - light_field_origin_voxel();
  let s = vec3<f32>(f32(LIGHT_FIELD_DIM) * LIGHT_FIELD_CELL);
  return all(rel >= vec3<f32>(0.0)) && all(rel < s);
}
/// AO（Douglas #15）：只作用于**间接**部分（常量天光 + GI），直射太阳由阴影负责。
/// 模型：平面上以该点为中心的球（这里是 16³ 块）恰好一半实体 → 不压暗；
/// 地板与墙的夹角处 ≈ 3/4 实体 → 按超出 50% 的部分压暗。
const AO_GAIN: f32 = 2.0;
fn light_field_ao(p_voxel: vec3<f32>, obj_id: i32) -> f32 {
  if (obj_id >= 0) { return 1.0; }
  if (!light_field_contains(p_voxel)) { return 1.0; }
  let fill = textureSampleLevel(light_tex, light_samp, light_field_uv(p_voxel), 0.0).a;
  return clamp(1.0 - (fill - 0.5) * AO_GAIN, 0.0, 1.0);
}
/// 发光密度 ε（0..1，= cell 内 Σ 发光强度 / cell 体积）。
fn light_field_emit(p_voxel: vec3<f32>) -> vec3<f32> {
  return textureSampleLevel(light_tex, light_samp, light_field_uv(p_voxel), 0.0).rgb;
}
/// cast 射线**沿程累加**发光（「体素即光源」在 GI 传输侧的接法）。
///
/// 为什么必须按体积聚合：单个发光体素 2cm，射线在 32cm 探针间距下命中它的概率 ~3e-5
/// （rpp=10 时约每 3000 帧一次）—— 逐体素求交在采样上不可行。这里把 cell 的发光摊成
/// **密度** ε 并沿程积分 radiance += ε·ΔL（功率守恒：摊开面积不改变该点收到的期待辐照度，
/// 但方差降 256 倍）。只累加到几何首命中 t_max 为止 → 两侧遮挡依然正确，不引入漏光。
fn light_field_gather(origin: vec3<f32>, dir: vec3<f32>, t_max: f32) -> vec3<f32> {
  // 射线 × 包围球求交（球保守包含盒；用二次式而非 slab，避免 dir 分量为 0 的除零）
  let size = f32(LIGHT_FIELD_DIM) * LIGHT_FIELD_CELL;
  let center = light_field_origin_voxel() + vec3<f32>(size * 0.5);
  let radius = size * 0.8660254; // 半对角线
  let oc = origin - center;
  let b = dot(oc, dir);
  let disc = b * b - (dot(oc, oc) - radius * radius);
  if (disc <= 0.0) { return vec3<f32>(0.0); }
  let sq = sqrt(disc);
  let t_lo = max(-b - sq, 0.0);
  let t_hi = min(-b + sq, t_max);
  if (t_hi <= t_lo) { return vec3<f32>(0.0); }
  // 粗步进：步长 = 1 cell，头半步居中（2·半径/步长 ≈ 55 步上限）
  let step = LIGHT_FIELD_CELL;
  var t = t_lo + step * 0.5;
  var acc = vec3<f32>(0.0);
  var i = 0;
  loop {
    if (t >= t_hi || i >= 64) { break; }
    let p = origin + dir * t;
    if (light_field_contains(p)) {
      acc = acc + light_field_emit(p) * step;
    }
    t = t + step;
    i = i + 1;
  }
  return acc;
}


// 着色法线 = **逐体素（隐式）法线**：命中体素 6 邻域占据差分的负梯度；退化时退回面法线。
//
// 向外 = 占据度的负梯度，占据度用 `sample_brickmap != 0`（air/palette 0 = 非占据）。
// 效果：球/柱/斜面这类体素化曲面按 26 邻域方向的平滑法线着色（不再是一格一格的平面），
// 而薄板仍是面法线 —— 两边的极端都各得其所。
//
// 【退化回退为什么必须是面法线】1 体素厚的薄板（地板/墙 —— 体素世界的主力几何）的 ±x/±z
// 邻接全实心、±y 全空气，二值差分三项全部抵消 → 零向量；任何**对称**模板（±2、±4…）同样
// 退化，因为薄板两侧往外都是空气 —— 几何上本来就不存在唯一方向，面法线是唯一有定义的答案。
//
// 旧实现正是在这一步硬编码 (1,0,0)：薄墙只有 +X 那侧碰巧正确，-X 侧 8 个角探针全落背面、
// 被 DDGI 的 wn 闸门剔光（Probe 品红、整面全黑），地板则一半探针被误用（漏光/发暗），太阳
// 直光也按 +X 计算。**那次失败的是"退化回退"，不是隐式法线本身** —— 所以这次按面法线退。
//
// 代价：每像素 6 次 sample_brickmap（一次 chunk 定位 + 最多 4 层树下钻）。
fn voxel_normal(g: Grid, voxel: vec3<i32>, face_n: vec3<f32>) -> vec3<f32> {
  let x = vec3<i32>(1, 0, 0);
  let y = vec3<i32>(0, 1, 0);
  let z = vec3<i32>(0, 0, 1);
  // 占据度差分（- 侧减 + 侧）：实心一侧计数更大 → 差分**直接指向空气侧 = 向外法线**
  // （例：+X 面朝空气 → -X 是实心、+X 是空气 → d.x = 1-0 = +1 ✓）
  let d = vec3<f32>(
    f32(sample_brickmap(g, voxel - x) != 0u) - f32(sample_brickmap(g, voxel + x) != 0u),
    f32(sample_brickmap(g, voxel - y) != 0u) - f32(sample_brickmap(g, voxel + y) != 0u),
    f32(sample_brickmap(g, voxel - z) != 0u) - f32(sample_brickmap(g, voxel + z) != 0u),
  );
  let len = length(d);
  // 退化（薄板/实心内部）→ 退回面法线；否则归一到 26 邻域方向之一
  return select(face_n, d / len, len > 1e-3);
}

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
    // ---- 逐体素着色（Douglas #22/#23：一体素一色）+ 逐体素法线 ----
    // albedo/采样点/阴影射线体素锚定（同体素同色）；法线取**隐式逐体素法线**（退化退面法线，
    // 见 voxel_normal 注释）—— 它同时决定太阳 N·L、阴影射线/GI 采样点的外推方向、DDGI 方向采样。
    let alb = palette_albedo(best.palette_base, best.uh.pal);
    // 体素中心 → 世界系（主世界 identity 直等于体素中心；物体经旋转/缩放变换）
    let gg = make_grid(u32(best.uh.obj_id) + 1u);
    let vc = vec3<f32>(best.uh.voxel) + vec3<f32>(0.5);
    let p_voxel = gg.pos + vec3<f32>(dot(vc, gg.col0), dot(vc, gg.col1), dot(vc, gg.col2)) * gg.scale;
    let n = voxel_normal(gg, best.uh.voxel, best.uh.n);
    // 沿 n 的外推距离：**必须保证离开命中体素**。轴向法线时 0.5+eps 就够，但逐体素法线在
    // 棱边/圆角处是斜的，0.5·n 的分量 < 0.5 → 起点仍在体素内 → DDA 先命中所属体素的邻居
    // （厚墙的棱边被自己遮住 → 每条棱一圈暗边）。0.5/max|n| 正是「从中心沿 n 走多远离开
    // 单位立方体」，任何方向都保证落在空气侧。
    let n_off = 0.5 / max(max(abs(n.x), abs(n.y)), max(abs(n.z), 1e-3)) + SHADOW_SURFACE_EPS;
    // R3-18 直光硬阴影（#02/#17 形态）：1 条太阳射线，不通即阴影。
    // 射线原点 = 体素中心 + n×(0.5 + SHADOW_SURFACE_EPS)（贴向空气侧邻域）。
    // 那个 eps 是必需的：+n×0.5 恰好落在面平面上，对 -X/-Y/-Z 面坐标是整数 → DDA 的
    // floor 落回**体素自己** → t=0 自命中；而下面的自命中防护把「命中自己」当无遮挡 →
    // 这三向的面无论墙多厚都吃到太阳（封闭无光房间的明暗会完全跟着面朝向走）。
    // +X/+Y/+Z 面落在高侧边界，floor 到邻域空气格，所以旧代码只在这半边出错。
    let sun_dir = light_u.lights[0].kind_pos_dir.yzw;
    let ndl = max(dot(n, sun_dir), 0.0);
    var sun = 0.0;
    if (ndl > 0.0) {
      let sh = trace_scene(p_voxel + n * n_off, sun_dir, 8192.0, 0.0, 3u);
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
      // 只把点推到体素表面外侧（沿 n 走 n_off，保证离开命中体素，见其定义处注释）；
      // 更远的、随 cell 尺寸缩放的外推在 ddgi_sample_lod 内部按该 LOD 的 cs 施加
      // （DDGI_BIAS_CELLS）。
      gi = ddgi_sample(p_voxel + n * n_off, n) * ddgi_u.params.z / DDGI_PI;
    }
    // AO（Douglas #15）只作用于**常量天光**这一项。
    //
    // 【为什么不能乘在 gi 上】gi 是**已经带遮挡信息的全局光照** —— 探针射线本身就逐方向做了
    // 遮挡判定（wd/Chebyshev），再乘一个 16³ 粗粒度填充率的 AO 属于**重复计算**。它的症状
    // 很好认：AO 只乘间接项 → 关掉 DDGI 时 gi=0、只剩 0.05 常量天光，AO 作用在一个很小的
    // 量上（"不开 DDGI 就没斑"）；开着时 gi 是 0.2~0.4，同一个 AO 把它乘暗 → 圆柱/薄肋上
    // 出现**块状暗斑**，而且光照场是相机中心、cell 0.32m、随相机滚动重铺 → **微小视角变化
    // 就整片跳变**。这正是"光影斑点 + 暗部骤变"的主因。
    //
    // 常量天光没有任何遮挡信息（凹角与开阔面一样亮），AO 正是为它准备的 —— 所以只留这一项。
    let ao = light_field_ao(p_voxel, best.uh.obj_id);
    // 天光常量在两种情形下兜底：① 本像素不在任何级联盒内（ddgi_dbg_dom==0，无探针覆盖）；
    // ② 在级联内但**采样不可信/无数据**（见下）。级联内数据可信时一律由 GI 提供 ——
    // 无光室内的探针辐照度≈0 → 画面接近全黑；若无条件加常量，室内外就没有亮度差。
    //
    // 【为什么不能"采样失败就归零"】ddgi_sample 返回的是**归一化**平均（total/wsum）：
    // wsum 只决定"有几个探针参与了这次平均"，**不决定返回值的幅度**。于是 wsum 在阈值
    // 1e-4 附近时，结果会从"某一个边缘探针的满幅值"直接跳到 0（或跳到粗级 LOD 的值）。
    // 这一跳就是两个被反复报告的观感问题的共同根因：
    //   ① 薄几何/曲面上"8 角探针被剔空"的像素间接光整项归零 → **零星黑斑**
    //      （不开 DDGI 时 amb 是常量，所以看不到）；
    //   ② 相机稍微一动，这些像素就在「有 GI」与「纯黑」之间翻 → **阴影暗部骤变**。
    // 改成按置信度淡出到天光常量：探针数据正常 → conf≈1（室内依旧由 ≈0 的辐照度决定，
    // 仍然黑）；探针不可信 → conf→0（退化成常量天光，与 DDGI 关闭时的观感一致）。全程连续。
    //
    // 【判据必须是 cov（覆盖度），不是 wsum】cov = Σ wtri over「有可用数据的角」
    // （窗口/ENABLED/ACTIVE/age/深度纹素有记录），∈[0,1]；它回答的是"本点到底有没有探针数据"。
    // wsum = Σ wtri·wn·wd 额外乘了两道**照明**闸门，回答的是"有几个探针愿意给本点照明"。
    // 用 wsum 当置信度会把"该点本来就该黑"误判成"数据不可信"：室内凹角处相邻两堵墙互相
    // 遮挡，存活权重天然很低 → wsum 塌到阈值以下 → 把常量天光加回来 → **最该黑的墙角反而
    // 最亮**（室内墙角漏光）；cell 越粗（离相机越远）遮挡剔除越多 → 漏光越明显。见
    // ddgi_sample_lod 里 cov / wsum 的分界注释。
    // 阈值取 [0, 0.02]：**只覆盖"基本没有可用数据"的区间**。
    //
    // 【重要】conf 只用来**补天光常量**，绝不去缩放 GI 的幅度。曾经用它乘过 gi，结果：
    // 覆盖度低的像素被压暗 → 暗斑范围更大更明显。这个教训记在这里，别再犯。
    let conf = smoothstep(DDGI_CONF_LO, DDGI_CONF_HI, ddgi_dbg_cov);
    let amb_sky = sky * DDGI_SKY_AMBIENT;
    let in_casc = ddgi_dbg_dom > 0.5;
    let amb = select(amb_sky, amb_sky * (1.0 - conf), in_casc);
    col = alb * (sun_c * ndl * sun + amb * ao + gi);
    // 自发光体素：radiance 直出（不吃方向、不吃阴影）——「每个体素都能是光源」的着色侧。
    col = col + alb * palette_emissive(best.palette_base, best.uh.pal) * DDGI_EMIT_GAIN;
    if (ddgi_u.params.y > 0.5) {
      if (ddgi_u.params.y < 1.5) {
        col = alb * sky * DDGI_SKY_AMBIENT * 0.15 + alb * gi * 8.0;
      } else if (ddgi_u.params.y < 2.5) {
        // wsum 档 = 采样置信度的**灰度直方图**（1:1，不吃 albedo）。
        // wsum = Σ wtri·wn·wd 是 trilinear 权重和 ∈ [0,1]（开阔表面 ~0.3-0.6）→ 直接当灰度。
        // 旧版写成 `alb * clamp(wsum*0.125,0,1) * 2`：既乘了 albedo（深色材质即使覆盖正常
        // 也显示近黑）又整体偏低 16× —— 结果"哪里到底有没有数据"根本读不出来，
        // 是上一轮把 wsum 当成 GI 幅度缩放系数的误判成因之一。
        col = vec3<f32>(clamp(ddgi_dbg_wsum, 0.0, 1.0));
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
        //   亮黄 = 壳内 8 角被 wn 全剔（薄几何），用**同级**放宽 wn 兜住 → GI 有效（见
        //          ddgi_sample 壳内分支注释）；**不退到粗级**，所以不会跨墙漏光
        //   浅绿 = 壳没给出数据，但**更粗一级 LOD 兜住了** → GI 有效，只是精度粗
        //          （相机滚动时新进入窗口的那条带属于这一类，不是异常）；
        //          但粗级探针在远处、方向掠射时深度图判不出遮挡 → 封闭空间里它是漏光来源
        //   绿   = 权重和 > 0 且辐照度非 0 → 本壳直接正常
        //   白   = 权重和 > 0，但辐照度 ≈ 0 → 过闸的探针图集是空的（写入/寻址问题）
        //   青   = 无数据闸门：探针没进本帧 worklist（near=false）、年龄太小，或该方向纹素
        //          从未被写过 —— 都不能以 0 参与平均
        //   橙   = ENABLED 闸门（该 cell 全满 → bake 没放探针）
        //   品红 = 法线背向闸门（wn <= 0）
        //   红   = depth 遮挡闸门（wd <= 0）
        //   灰   = 角越界（clamp 路径不会出现）
        // 旧版在这里对 dom<0.5 直接涂蓝，把这份直方图整个短路掉了。
        if (ddgi_dbg_fb > 2.5) {
          col = vec3<f32>(0.98, 0.98, 0.35);
        } else if (ddgi_dbg_fb > 1.5) {
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


/// 辐照度图每探针 4×4（= 16 纹素）。
///
/// 【为什么不用原版的 8×8】曾对齐 DDGI 原版试过 8×8 / 16×16（见下面 DEPTH_TEXELS 注释）：
/// 实测**画质没有明显改善**，代价却是显存 ×4、collect 线程 ×4、帧轮换周期 ×4。故回退。
const DDGI_IRR_TEXELS: u32 = 4u;
/// 深度图每探针 8×8（= 64 纹素）。
///
/// 【为什么不用原版的 16×16】曾试过 16×16（每纹素 ~11°，现在 ~22°）：**实测漏光没有明显
/// 改善** —— 说明当时的漏光主因**不在深度角分辨率**，而在别处（射线方向未绑定纹素 / 借针
/// 跨墙 / 级联硬切，这三处都已单独修过）。而代价很实在：深度图集 64MB → 256MB、
/// collect 线程 80 → 320、帧轮换周期 ×4。故回退到 8×8。
const DDGI_DEPTH_TEXELS: u32 = 8u;
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
/// 时域混合系数（每帧向新采样靠拢的比例）。rpp 偏低时单帧估计方差大，亮区边界随噪声
/// 逐帧移动（静止相机也在脉动）。0.06→0.03 未见效 → 再降到 0.015（≈67 帧 / 1.1s）。
/// 这是**稳态**（探针年龄足够大）用的比例；年轻探针走 `ddgi_age_alpha` 的 1/age 快收敛，
/// 见该函数注释 —— 稳态逐帧脉动由这个值负责，所以不要为了"收敛快"去调大它。
const DDGI_ALPHA: f32 = 0.015;
/// 深度图时域系数（图集两通道：mean / std）。射线按「绑定纹素 + 帧轮换」分配后，
/// 每个纹素每 ⌈64/rpp⌉ 帧才拿到一条射线（rpp≈4 → 16 帧），单帧样本极少 →
/// 靠这个系数做跨帧滑动平均收敛（见 collect 的深度分支）。
const DDGI_DEPTH_ALPHA: f32 = 0.03;
/// Chebyshev 软遮挡里 std 的**绝对**下限（体素）。只用来防「std ≈ 0 时退化成刀锋」。
///
/// 【为什么不能是「占 mean 的比例」】旧值 = `mean × 0.02`，即下限随「探针到该方向表面的
/// 距离」线性增长。后果：探针离墙越远 soft 越大，跨墙角的 `wd = soft²/(soft² + delta²)`
/// 就越大 —— 而 `delta`（墙厚 + 采样点离墙的距离）与探针距离**无关**。于是远处/粗级探针的
/// wd 被系统性抬高；`ddgi_sample_lod` 末尾做的是**归一化**平均，只要有一个跨墙角通过，它
/// 就能独占整块结果 ⇒ 漏光，且「探针越远越明显」（对上实机：LOD0 轻、LOD1 重）。
/// DDGI 原版是 `variance/(variance + delta²)`，variance 取自深度方差、**没有距离相关下限**，
/// 深度稳定（方差≈0）时就是硬判定。这里改回同一口径：soft 由 std 主导，下限只防 0。
const DDGI_DEPTH_SOFT_MIN: f32 = 0.5;
/// 空间混合（探针间去噪）里"自身"的权重参照。邻居权重和按探针实际间距加权
/// （见 `ddgi_probe_blend_irr`），典型约 3（6 邻居 × ~0.5）→ 2.0 表示自身约占 40%。
const DDGI_BLEND_SELF: f32 = 2.0;
/// 探针放置的最小净距（体素）：候选空叶的**半宽**必须 ≥ 此值才接受。
/// Douglas #23："these heuristics ensure that the probes end up spaced as far apart as
/// possible with some distance between themselves and the nearest surface"。探针贴在表面上时
/// 半球被几何切掉一半 → 辐照度畸变、方差大，与相邻"自由"探针的差异被插值放大成探针晶格
/// 亮斑，贴面漏带也随之而来。
///
/// **取值 2.0（= 恢复 Douglas 的"下钻到子叶"）**：`ddgi_place_probe` 三级候选的准入条件是
/// 空叶半宽 8（16³ 子块）/ 2（4³ 子块）/ 0.5（1³ 体素）。曾经取 4.0 → 后两级（2 ≥ 4、0.5 ≥ 4
/// 全假）被整段关掉，只剩"整个 32cm 立方全空"这一级。后果：**任何离几何 < 32cm 的 cell 都
/// 没有探针**。而 LOD0 的 cell 本身就是 32cm —— 于是贴着墙的那一圈像素，壳内 8 个角永远
/// 一个有效探针都没有，只能走 `ddgi_sample` 的"退到更粗 LOD"兜底；兜底返回的是**归一化**
/// 平均（有一个探针入选就占满结果），粗级探针又可能落在墙外侧、看得见天空 → 封闭房间内壁
/// 被整片点亮成跟粗级 cell 对齐的条纹（用权重门槛去堵它只会把那一圈变成黑块 —— 试过，
/// 两个症状都更差）。
/// 取 2.0 ⇒ 4³ 那级恢复：贴墙 cell 会在墙内侧一个**全空 8cm 立方**的中心拿到探针（离表面
/// ≥4cm），壳内就有数据了，根本不需要粗级兜底；1³ 那级（0.5）仍关闭 —— 探针不会贴到面上。
/// 【保持 2.0 —— 提到 4.0 需要先改放置策略，见下】
///
/// Douglas 的原则确实是"探针必须离最近表面有距离"（"don't want probes right up against
/// walls… half of the probe's memory and half of the probe samples are being wasted"）。
/// 但**这个原则不能单独搬过来**，两次实测都失败：
///   ① 第一次（插值还按名义 cell 中心）：偏移变大 → 值被安放到错位置 → 连 LOD0 都出伪影；
///   ② 第二次（插值已按探针实际位置）：近处**又**变坏 —— 说明根因不在插值，而在**放置**。
///
/// 根因：我们的放置是「**每个 cell 塞一个探针**，取离 cell 中心最近、且满足净距的全空叶」。
/// 净距要求越严 → 能通过的 cell 越少 → **贴墙一圈的 cell 直接没有探针** → 那些像素可信的角
/// 变少 → 归一化平均被少数探针支配 → 近处变糊/出块。
/// 而 Douglas 的放置不是"逐 cell 填格"：他**选**一批彼此尽量远离、离表面有距离的位置，
/// 再把细级数据 **down-sample** 出粗级（见 DDGI 那集）。位置与刚性格点解耦，所以他能用大净距。
/// ⇒ 想恢复 4.0，必须先把放置改成"选择式 + 逐级 down-sample"（这是结构改动，不是常量）。
///
/// 1³ 那级（0.5）仍关闭 —— 探针不会贴到面上。
/// 【现在是**真正可调**的净距（体素）】
///
/// 语义：候选探针位置到最近固体的距离（`ddgi_probe_clearance`，1 体素粒度）必须 ≥ 本值。
/// 它只在第 2/3 档里**逐个候选点**过滤，**不再能一刀关掉整档** —— 密度由"第 1/2 档是否启用"
/// 决定，而它们现在是固定的；放不出探针的 cell 由 Step 1 的 cell→path 指向邻近探针兜住。
///
/// 历史（为什么曾经必须是 2.0）：那时各档比的是"空叶半宽"（8.0/2.0/0.5）这个代理，净距一调大
/// 就等于整档禁用 → 只有 100% 空气的 cell 有探针 → 密度崩塌 → 三角锯齿 + 黑斑。
/// 换成体素距离判据后，这个耦合被拆掉了，所以现在取 Douglas 的 4 体素（8cm）。
const DDGI_PROBE_MIN_CLEARANCE: f32 = 4.0;
// （原 DDGI_SHARE_K 已删）它把插值基的支撑半径放大到 K 个 cell，用来让 Step 1 共享过来
// 的探针拿到非零权重。代价是越过 cell 边界时"离开 stencil 的角"权重不归零却被丢弃 →
// 权重场在每个 cell 边界跳变（放射状三角锯齿）。现在插值基改用晶格名义三线性基，
// 共享探针天然拿到它所属角的权重，不需要放大支撑。
/// 邻居纹素的**覆盖度**下限（irr 图集第 4 通道 = 该纹素被写过多少次的时间累积，见 collect）。
/// 为什么需要它：空间混合（`ddgi_probe_blend_irr`）必须能拿**暗邻居**去稀释亮邻居，
/// 否则一颗跨墙拿到天光的探针没人稀释，就会在自己的投影处形成一个亮斑（室内墙角漏光）。
/// 而按**值**过滤是做不到的 —— `DDGI_CAST_FLOOR = 0` 之后，"真黑"与"没数据"的值都是 0。
/// 所以改按"这个纹素到底被写过没有"过滤，这个信号由第 4 通道承载。
const DDGI_TEXEL_MIN_COVERAGE: f32 = 0.5;
/// 前后判定（wn）的锐度：wn = clamp(dot(n,-dir)/此值, 0, 1)。
/// = 1.0 → 真正的余弦（60° 处 0.5、78° 处 0.2），与 Douglas/原版 DDGI 的「探针在体素
/// 前 vs 后」一致；= 0.2 时 78° 以内全部钳到满权重，会把**掠射探针**（墙角/凸边外侧那些
/// 看到开阔空间、天花板、外部天空的探针）按满权重混进来 —— 这正是「角落/接缝处柔和高亮」
/// 的来源（这类探针与着色点之间不穿几何，depth 闸门合理地放行，只能靠权重压）。
/// 注意：此值不改变拒绝集合（wn=0 恒为 dot<=0），只改相对权重 → 调它不会产生新黑区。
const DDGI_NORMAL_BIAS: f32 = 1.0;
/// 采样点沿法线的外推量 = 该 LOD 的 cell 边长 × 此系数（参考实现里的 normal_bias 语义）。
/// 必须随间距缩放：固定的 0.7 体素（1.4cm）相对 LOD0 的 64cm cell 等于贴在表面上，
/// 会让前后判定落在临界值上、整面被判「探针在背面」。
const DDGI_BIAS_CELLS: f32 = 0.1;
const DDGI_T_MAX: f32 = 8192.0;
/// 级联混合带宽度（以**当前级** cell 边长为单位）：像素距某个 LOD 盒边界的距离小于
/// `cs × 此值` 时，在该级与相邻级之间过渡，消除「跨过盒边界突然换一套光照」的硬边
/// （用户报告的「LOD0/LOD1 边界很明显」）。0.5 = 半格（LOD1 的 cs=32 体素 → 带宽 16 体素）。
/// 边界两侧恰好都是 0.5/0.5，所以跨边界连续 —— 推导见 ddgi_cascade_blend。
const DDGI_CASCADE_BLEND: f32 = 0.5;
/// 每帧射线总预算，由 `ddgi_seal` 均分给全部活跃探针（rpp = 预算/活跃数，钳 [1,256]）。
/// 必须与 Rust 侧 `DDGI_RAY_BUDGET` 一致。曾试过 262144 / 1048576 压"探针晶格亮斑"与
/// 深度闸门抖动：帧时涨了但问题没解决（根因是探针网格对贴缝尺度欠采样 + 深度角度均值偏差，
/// 不是射线数量），已退回 131072。
const DDGI_RAY_BUDGET: u32 = 131072u;
const DDGI_PROBE_BUDGET: u32 = 4096u;
const DDGI_SHADOW_T_MAX: f32 = 8192.0;
const DDGI_SHADOW_BIAS: f32 = 0.5;
/// 着色点从体素表面再外推的量（体素）。见 dda_main 阴影射线处注释：`+n×0.5` 恰好落在
/// 面平面上，负向面的坐标是整数 → DDA floor 落回自身体素 → 自命中被当成「无遮挡」→ 漏光。
/// 必须与 Rust `brickmap::dda::wgsl_consts::SHADOW_SURFACE_EPS` 一致（有对齐测试）。
const SHADOW_SURFACE_EPS: f32 = 0.03125;
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
/// 天光环境项：在该像素**没有任何级联探针覆盖**、或**采样置信度过低（无数据/仅剩边缘探针）**
/// 时兜底。数据可信的级联内像素一律由 DDGI 提供环境光（带遮挡）：无光室内 → 探针辐照度≈0
/// → 接近全黑。详见着色处 conf / amb / gi_eff 的推导（那里解释了为什么不能"失败就归零"）。
const DDGI_SKY_AMBIENT: f32 = 0.05;
/// 采样置信度区间：`conf = smoothstep(LO, HI, cov)`，**只用来在"基本没有可用数据"时
/// 补上天光常量**（绝不用来缩放 GI 幅度，理由见着色处注释）。
/// cov = Σ wtri over「有可用数据的角」（**不含** wn/wd 两道照明闸门）→ ∈ [0,1]。
/// HI=0.02 只覆盖"8 个角几乎都没有数据"这一档；只要有一个角有数据（cov ≥ ~0.25 量级）
/// 就完全不受影响 —— 这一点**不能**换成 wsum：凹角/远处粗级的存活权重天然低，用 wsum 会
/// 把"本该黑"判成"不可信"而把天光常量加回来（室内墙角漏光）。见着色处注释。
const DDGI_CONF_LO: f32 = 0.0;
const DDGI_CONF_HI: f32 = 0.02;
/// 探针射线命中时的辐亮度下限。**必须为 0**：任何非零下限都会被反馈回路放大成
/// ≈ alb·sky·FLOOR/(1-alb) 的"室内自发光"，封闭空间永远压不黑。
/// 首帧的种子由**能看到天空的射线**提供（miss → sky_rgb），所以不需要这个下限。
const DDGI_CAST_FLOOR: f32 = 0.0;

// 世界空间探针网格（每 LOD）：origin.xyz = 世界原点（voxel）、origin.w = cell 边长；
// dims.xyz = cell 维度、dims.w = 该 LOD 在全局 slot 数组中的起始下标。
struct DdgiLod {
  origin: vec4<i32>,
  dims: vec4<u32>,
};
struct DdgiUniform {
  lods: array<DdgiLod, 4>,
  // x=帧计数, y=调试模式(0..4), z=GI 增益, w=借针半径（0=关，见 ddgi_find_neighbor_probe）
  params: vec4<f32>,
  // x=GI 开关, y=总槽位数, z=Chebyshev std 信任系数（0..1，见采样侧 soft 计算）, w=未用
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
/// cell → slot 间接表（Step 1）：表长 = Σ 各级 dims 乘积，索引 = **绝对 slot 下标**（即
/// `ddgi_slot` 在 identity 下的返回值）。identity 状态与"没有这张表"逐位等价。
/// 用途：允许"本格放不出满足净距的探针"的 cell 指向**邻近 cell 的探针** —— 这样 Douglas 的
/// "探针必须离表面有距离"（否则半张深度/辐照度图都浪费在贴着的那一小片表面上）与"每个采样点
/// 都有 8 个可用角"就不再互斥（我们两次单独调净距都失败的根因就在这里）。
@group(4) @binding(10) var<storage, read_write> ddgi_cell_slot: array<u32>;

// ===== @group(5)：collect 的图集写入侧（只被 ddgi_collect 使用）=====
// 只放两个存储纹理：其它缓冲（worklist / samples / indirect / uniform）全部复用 BG4 的绑定。
// 同一 buffer 若同时在两个 bind group 里以「只读 + 读写」两种方式绑定，wgpu 会判定使用冲突。
@group(5) @binding(0) var ddgi_irr_out: texture_storage_2d_array<rgba16float, write>;
// depth 写入侧：.x = mean（绑定纹素的命中距离均值）、.y = std（该均值的跨帧标准差）。
// 与采样侧读法一一对应；存 std 而不是 mean² 是为了避免 f16 溢出/掉精度（mean² 可达 8192²）。
@group(5) @binding(1) var ddgi_depth_out: texture_storage_2d_array<rgba16float, write>;

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
/// collect 每探针的纹素线程数 = **从纹素数派生**（16 irr(4×4) + 64 depth(8×8) = 80）。
///
/// 【必须是派生值，不能写死】历史上它曾写死某个值而在纹素数变化后不同步：`ddgi_seal` 拿
/// 它算 collect 的 dispatch 规模，多出来的线程会落进 depth 分支并越出**本探针自己的**纹素块，
/// 把邻居探针的深度纹素改写（或越界作废）—— 邻居的 (mean,std) 每帧乱跳、`snap` 路径还会把
/// 它们写 0（采样侧判"无数据"）。症状是"静态闪烁/亮区伸缩"，且伪装成深度噪声、极难定位。
const DDGI_COLLECT_THREADS: u32 = DDGI_IRR_TEXELS * DDGI_IRR_TEXELS
  + DDGI_DEPTH_TEXELS * DDGI_DEPTH_TEXELS;

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
/// **纯算术**槽位（不经 cell→slot 间接表）。凡是要读「某个世界 cell **自己** 的烘焙记录」
/// 的地方（判定邻接占用、粗级继承细级探针位置）都必须用它 —— 间接表表达的是
/// "采样这个 cell 时该去哪读探针"，不是"这个 cell 的记录在哪"；走间接表会读到邻居的记录。
fn ddgi_slot_own(lod: u32, wc: vec3<i32>) -> u32 {
  let d = ddgi_u.lods[lod].dims.xyz;
  let di = vec3<i32>(d);
  let r = ((wc % di) + di) % di;
  return ddgi_lod_slot_base(lod) + u32(r.x) + u32(r.y) * d.x + u32(r.z) * d.x * d.y;
}
fn ddgi_slot(lod: u32, wc: vec3<i32>) -> u32 {
  // 经 cell→slot 间接表（见 binding(10) 注释）。identity 时 `ddgi_cell_slot[own] == own`，
  // 与"直接返回 own"逐位等价 —— 所以建立这条通路本身不改画面，是 Step 1 的第 1 小步。
  return ddgi_cell_slot[ddgi_slot_own(lod, wc)];
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
/// 时域混合系数按**探针年龄**给：历史越短越信当帧样本。
///
/// 为什么需要：老实现是「age ≤ 1 直写一帧原始估计，之后恒用 DDGI_ALPHA(0.015)」。那一帧
/// 原始估计只有 rpp 条射线（满活跃时 rpp≈4）→ 方差极大；而 0.015 的时间常数 ≈67 帧。
/// 合起来就是「先糊一帧、再花 1 秒慢慢擦」——刚加载、相机滚动带进新探针的那条带，正好都
/// 落在新探针上，于是感知到的就是"DDGI 收敛很慢"。
///
/// 换成 1/age 的**等权滑动平均**后：
///   · 年龄 n 的样本方差按 1/n 下降（统计上的最优等权平均）→ 前 5 帧就把噪声压掉一半以上，
///     而感知上的"收敛"恰恰发生在这几帧；
///   · 年龄 ≥ 1/DDGI_ALPHA（≈67）时自动切回 DDGI_ALPHA → **稳态逐帧脉动完全不变**
///     （静止画面不会因此变吵）；DDGI_ALPHA 想调也不会和这里打架。
/// age ≤ 1 时 1/age = 1 → 与原 `snap`（直写）行为逐位一致，所以新烘探针第一帧仍是精确覆写。
fn ddgi_age_alpha(age: u32, base: f32) -> f32 {
  return max(base, 1.0 / f32(max(age, 1u)));
}

// ddgi_update_texel：历史遗留的旧混合路径，**当前无人调用**（collect 直接用 mix(prev,new,a)）。
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
  // 粗级探针拿到的射线数远低于平均（rpp≈3）→ 深度图 64 个纹素每帧只写到一小部分，
  // 遮挡判定(wd)随之失效。
  var total_active = 0u;
  for (var l = 0u; l < DDGI_LOD_COUNT; l = l + 1u) {
    total_active = total_active + atomicLoad(&ddgi_indirect[DDGI_INDIR_COUNT_BASE + l]);
  }
  // 下限 1：总活跃探针数 ≤ 槽数 65536 < 预算 131072，故 total_active×rpp 恒不超预算
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
// 最小净距：每轮的候选还要过 DDGI_PROBE_MIN_CLEARANCE（空叶半宽 ≥ 该值）。默认 4 →
// 只有 16³ 那轮有效，4³/1³ 兜底被关掉（拿不到合格空叶的 cell 就不放探针，交给邻居覆盖）。
fn ddgi_place_probe(g: Grid, cmin: vec3<i32>, cs: i32, center: vec3<f32>) -> vec4<f32> {
  let n16 = max(cs / 16, 1);
  // ---- Step 2：粗级 LOD **优先继承细级探针的位置** ----
  // Douglas："I then down sample the generated data to pick out probe positions for higher
  // levels of detail or LODs." —— 粗级应该是细级的**低分辨率版本**，而不是同一片空间里的
  // 另一套独立数据；后者会让跨级边界像"换了一套光照"（LOD1 的锯齿/跳变就是这么来的）。
  //
  // 细级 cell 边长 = cs/2 ⇒ 本 cell 正好覆盖 `wc*2 + (0|1)³` 这 8 个细级 cell。取其中
  // 离本 cell 中心**最近**的那个已放置探针的位置，直接作为本格的探针。找不到（细级盒之外
  // 的那一圈）才退回下面原有的放置启发式。
  // `wc` 用 center 反推（floor，负数也正确），与 `ddgi_slot_world_cell` 的约定一致。
  if (cs > 16) {
    let my_lod = u32(log2(f32(cs / 16)));
    let flod = my_lod - 1u;
    let fcs = ddgi_lod_cell_size(flod);
    let wc = vec3<i32>(floor(center / f32(cs)));
    var best_d2 = 1e30;
    var best_p = vec3<f32>(0.0);
    var found = false;
    for (var k = 0; k < 2; k = k + 1) {
      for (var j = 0; j < 2; j = j + 1) {
        for (var i = 0; i < 2; i = i + 1) {
          let fwc = wc * 2 + vec3<i32>(i, j, k);
          // 读**细级本帧的 bake 输出**（ddgi_cell / ddgi_cell_id），而不是 ddgi_slot_pos：
          // 位置必须与细级**同一帧**自洽，否则相机滚动时粗级会继承到"上一个世界 cell 的
          // 探针位置"（可能落在实体内部）→ 粗级边界跳变/锯齿。bake 已拆成 per-LOD 的 4 个
          // pass（细→粗），pass 边界就是内存屏障，这里读到的正是本帧细级刚算出的结果。
          let fslot = ddgi_slot_own(flod, fwc);
          // 该槽位必须**当前确实覆盖 fwc**（细级窗口外 / 环面绕到别的世界 cell → 跳过），
          // 且本帧 bake 已写过它（cell_id.w != 0）。
          if (ddgi_cell_id[fslot].w == 0 || any(ddgi_cell_id[fslot].xyz != fwc)) { continue; }
          let frec = ddgi_cell[fslot];
          if ((frec & DDGI_REC_ENABLED) == 0u) { continue; }
          let fp = vec3<f32>(fwc * fcs) + vec3<f32>(ddgi_rec_off(frec)) / 255.0 * f32(fcs);
          let d2 = dot(fp - center, fp - center);
          if (d2 < best_d2) {
            best_d2 = d2;
            best_p = fp;
            found = true;
          }
        }
      }
    }
    if (found) {
      return vec4<f32>(best_p, 1.0);
    }
  }
  // 轮 1：最近的「全空 16³ 子块」
  var b16_d2 = 1e30;
  var b16_p = vec3<f32>(0.0);
  var found16 = false;
  for (var k = 0; k < n16; k = k + 1) {
    for (var j = 0; j < n16; j = j + 1) {
      for (var i = 0; i < n16; i = i + 1) {
        let sub16 = cmin + vec3<i32>(i, j, k) * 16;
        // 第 1 档：本 cell 里存在**整块全空**的 16³（半宽 8 ≥ 2）。这一档与净距常量解耦：
        // 档位准入用固定阈值，`DDGI_PROBE_MIN_CLEARANCE` 只表达"探针应离表面有距离"的原则，
        // 不再能一刀把整档关掉（那正是"净距 4.0 → 密度崩塌 → 三角锯齿/黑斑"的机制）。
        if (ddgi_cell_state_sized(g, sub16, 16) == 0u && 8.0 >= 2.0) {
          let p = vec3<f32>(sub16) + vec3<f32>(8.0);
          let d2 = dot(p - center, p - center);
          if (d2 < b16_d2) { b16_d2 = d2; b16_p = p; found16 = true; }
        }
      }
    }
  }
  if (found16) { return vec4<f32>(b16_p, 1.0); }
  // 轮 2：无空 16³ → 在混合 16³ 内找最近的「全空 4³」
  var b4_d2 = 1e30;
  var b4_p = vec3<f32>(0.0);
  var f4 = false;
  for (var k = 0; k < n16; k = k + 1) {
    for (var j = 0; j < n16; j = j + 1) {
      for (var i = 0; i < n16; i = i + 1) {
        let sub16 = cmin + vec3<i32>(i, j, k) * 16;
        // 第 2 档：全空 4³ 叶（半宽 2）—— 贴墙 cell 靠这一档拿到探针，**必须始终可用**
        // （它的存在与否直接决定近场探针密度；用固定阈值 2.0 准入，与净距常量解耦）。
        if (ddgi_cell_state_sized(g, sub16, 16) == 2u && 2.0 >= 2.0) {
          for (var kk = 0; kk < 4; kk = kk + 1) {
            for (var jj = 0; jj < 4; jj = jj + 1) {
              for (var ii = 0; ii < 4; ii = ii + 1) {
                let sub4 = sub16 + vec3<i32>(ii, jj, kk) * 4;
                if (ddgi_brick_state(g, sub4, 3u) == 0u) {
                  let p = vec3<f32>(sub4) + vec3<f32>(2.0);
                  // 真实距离判据（**体素粒度**，见 ddgi_probe_clearance）：取代"空叶半宽"代理。
                  // 低于净距的候选点直接不采用 —— 这个 cell 仍可由 Step 1 的 cell→slot 指向
                  // 邻近探针，所以不会留下空洞（这正是"净距"第一次成为可调量、而不是整档开关）。
                  if (ddgi_probe_clearance(g, p) < DDGI_PROBE_MIN_CLEARANCE) { continue; }
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
        // 第 3 档（全空 1³，半宽 0.5）：**第一次用真实距离判据**。紧贴表面的候选点会因
        // "对齐 16³ box 非全空"而返回 0 → 被净距挡住（这一档实际上仍几乎不触发）；
        // 只有当候选点周围真有 ≥ half 的空 box 时才会启用 —— 那正是"离表面有距离"的定义，
        // 而不是再靠调常量。第 1/2 档保持不变（它们决定近场密度，不能拿代理去卡）。
        if (ddgi_cell_state_sized(g, sub16, 16) == 2u) {
          let r = ddgi_leaf16(g, sub16, center);
          if (r.w > 0.5 && ddgi_probe_clearance(g, r.xyz) >= DDGI_PROBE_MIN_CLEARANCE) {
            let d2 = dot(r.xyz - center, r.xyz - center);
            if (d2 < b1_d2) { b1_d2 = d2; b1_p = r.xyz; f1 = true; }
          }
        }
      }
    }
  }
  return vec4<f32>(b1_p, select(0.0, 1.0, f1));
}

/// Step 1：本 cell 放不出满足净距的探针时，在**同一 LOD** 的 ±r 环面邻域里找最近的
/// **已放置探针**，返回它的 slot；找不到就返回自身（那时该 cell 仍无数据，采样照旧走兜底）。
///
/// 为什么需要：放置规则要求"探针离表面有距离"（Douglas），但我们的槽位原本与 cell 一一对应
/// ⇒ 净距一严，贴墙一圈的 cell 直接没有探针 ⇒ 那些像素可信的角变少 ⇒ 近处变糊/出块
/// （两次单独调净距失败的根因）。有了指向，位置与刚性格点解耦，两个目标才能同时成立。
///
/// 只在**自身放出了探针**（bake 记录 ENABLED）的邻域里选 —— 指向过的 cell 在 bake 里
/// ENABLED=0，所以天然不会形成"指向的指向"链。邻域用槽位线性下标算：x ±1、y ±dim.x、z ±dim.x·dim.y。
///
/// 半径 r 由 `params.w`（Debug 面板 Borrow 滑杆）运行时给定：**r=0 直接返回自身 = 关闭借针**，
/// 即 Douglas 原架构（无针 cell 的插值角直接缺席，由深度/法线闸门之外的归一化消化）。
/// 诊断用途：验证 ±2 借针不检查中间实体 → 粗级跨墙指向是不是室内墙角漏光的主因。
fn ddgi_find_neighbor_probe(slot: u32) -> u32 {
  let r = i32(ddgi_u.params.w + 0.5);
  if (r <= 0) { return slot; }
  // 各级 dims 相同（DDGI_LOD_DIMS），所以 slot → 本级的 base 可以直接算，不必依赖调用处的 lod 变量
  let d = ddgi_u.lods[0u].dims.xyz;
  let per_lod = d.x * d.y * d.z;
  let base = (slot / per_lod) * per_lod;
  let dim_xy = d.x * d.y;
  let local = slot - base;
  let lx0 = i32(local % d.x);
  let ly0 = i32((local / d.x) % d.y);
  let lz0 = i32(local / dim_xy);
  var best = slot;
  var best_d2 = 1e30;
  for (var k: i32 = -r; k <= r; k = k + 1) {
    for (var j: i32 = -r; j <= r; j = j + 1) {
      for (var i: i32 = -r; i <= r; i = i + 1) {
        if (i == 0 && j == 0 && k == 0) { continue; }
        let lx = (lx0 + i + i32(d.x) * 8) % i32(d.x);
        let ly = (ly0 + j + i32(d.y) * 8) % i32(d.y);
        let lz = (lz0 + k + i32(d.z) * 8) % i32(d.z);
        let nb = base + u32(lx) + u32(ly) * d.x + u32(lz) * dim_xy;
        // 判"邻域有没有探针"必须读 **bake 的输出**（ddgi_cell 的 ENABLED 位），不能读
        // ddgi_meta：meta 正是本 pass（sort）此刻在写的数组，同 pass 内读别人的 meta 无顺序
        // 保证（读到上一帧的值 → 指向会逐帧抖）。ddgi_cell 由 bake 写、sort 只读，稳定。
        // 顺带：被"指向"过的 cell 在 bake 里 ENABLED=0，所以天然不会形成"指向的指向"链。
        if ((ddgi_cell[nb] & DDGI_REC_ENABLED) == 0u) { continue; }
        let d2 = f32(i * i + j * j + k * k);
        if (d2 < best_d2) {
          best_d2 = d2;
          best = nb;
        }
      }
    }
  }
  return best;
}

/// 小尺度净空（体素）：沿 6 个轴向逐格外查，遇到固体即停 → 取六向**最小步数**（保守下界）。
/// 为什么要它：`ddgi_cell_state_sized` 最小只支持 16 体素（half 8），而第 1/2 档要区分的
/// 恰恰是 Douglas 那种量级（净距 2/4 体素 = 4/8cm）。只靠层级 box 会让"准入"退化成
/// "要求 ≥8 体素净空" ⇒ 探针密度崩塌（三角锯齿 + 黑斑的成因）。代价 6×max_steps 次点查询。
fn ddgi_axis_clearance(g: Grid, p: vec3<i32>, max_steps: i32) -> f32 {
  let dirs = array<vec3<i32>, 6>(
    vec3<i32>(1, 0, 0),
    vec3<i32>(-1, 0, 0),
    vec3<i32>(0, 1, 0),
    vec3<i32>(0, -1, 0),
    vec3<i32>(0, 0, 1),
    vec3<i32>(0, 0, -1),
  );
  var best = f32(max_steps);
  for (var d = 0; d < 6; d = d + 1) {
    var steps = max_steps;
    for (var i = 1; i <= max_steps; i = i + 1) {
      if (sample_brickmap(g, p + dirs[d] * i) != 0u) {
        steps = i - 1;
        break;
      }
    }
    best = min(best, f32(steps));
  }
  return best;
}

/// 探针到最近固体的**距离下界估计**（体素）：以 pos 为中心、边长 size 的**对齐 box** 全空 ⇒
/// 该 box 内无固体。从大往小试 size ∈ {256,128,64,32,16}，命中即返回 half = size/2。
///
/// 为什么需要：原来各档比的是"空叶半宽"（8.0 / 2.0 / 0.5），那是个**粗糙代理** ——
/// 净距一调大就等于把整档关掉（第 2 档 `2.0 >= 4.0` 恒假）⇒ 只有 100% 空气的 cell 有探针
/// ⇒ 密度崩塌 ⇒ 三角锯齿 + 黑斑。要真正落实 Douglas 的"探针离表面有距离"，判据必须是距离。
///
/// 代价：复用现成的层级查询 `ddgi_cell_state_sized`（子查询数 = (size/64)³ 量级，
/// 256→64 次、128→8、64→1）⇒ 通常 1~8 次查询，**不是** 9³ 邻域扫描那种 729 次。
/// 注意：box 必须对齐到 size（用位掩码，负数也正确），且 pos 要落在**半宽之内**才算数
/// （贴着 box 边的话"下界"就退化成 0，不能当距离用）。
fn ddgi_probe_clearance(g: Grid, pos: vec3<f32>) -> f32 {
  let pi = vec3<i32>(floor(pos));
  // ① 小尺度（1~8 体素）：轴向扫描，粒度 1 体素 —— 第 1/2 档要区分的正是这一档精度
  let small = ddgi_axis_clearance(g, pi, 8);
  if (small < 8.0) { return small; }
  // ② 大尺度（≥16 体素）：层级对齐 box，命中即返回 half（粗粒度下界）
  let sizes = array<i32, 5>(256, 128, 64, 32, 16);
  for (var s = 0; s < 5; s = s + 1) {
    let size = sizes[s];
    let half = size / 2;
    let lo = pi & vec3<i32>(~(size - 1)); // 对齐到下界（位掩码，负数同样正确）
    // pos 必须在 box 的**半宽以内**，否则"box 全空"给不出 half 这么强的下界
    let d = abs(pi - (lo + vec3<i32>(half)));
    if (any(d > vec3<i32>(half / 2))) { continue; }
    if (ddgi_cell_state_sized(g, lo, size) == 0u) {
      return f32(half);
    }
  }
  // 两个尺度都没给出更强结论 → 就用轴向扫描的结论（≥8 体素范围内 6 向无固体）
  return small;
}

/// 单级烘焙：lod 由**派发该 pass 时确定**（见下面 4 个入口）。idx = 该级内的槽位下标。
///
/// 为什么按级拆成 4 个 compute pass，而不是一次 dispatch 覆盖全部槽位：
/// Step 2 的「粗级继承细级探针位置」要读**本帧**细级的 bake 输出（ddgi_cell / ddgi_cell_id）。
/// 同一个 pass 内没有顺序保证（连续 dispatch 也不构成屏障），粗级可能读到细级尚未写入的值；
/// 只有 pass 边界才是内存屏障。所以按细→粗拆成 4 个 pass，逐级建立依赖。
fn ddgi_bake_one(lod: u32, idx: u32) {
  if (idx >= ddgi_lod_count(lod)) { return; }
  let slot = ddgi_lod_slot_base(lod) + idx;
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
  // 同时清掉"指向邻居"：本 slot 现在代表的世界 cell 变了，旧指向可能属于上一任 cell。
  // 不清的话，世界改变后的头 1~2 帧采样可能短暂指向不相干的探针（MIN_SAMPLE_AGE 只盖住
  // 大部分，不是全部）。指回自身 = 最保守的状态，随后本帧的放置结果会覆盖它。
  ddgi_cell_slot[slot] = slot;
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
  // 中心捷径也必须过最小净距：它所在的 16³ 子块全空，但中心本身可能贴在子块的某个面上
  // （子块外紧邻实体）→ 到最近表面的距离只有"到子块面的距离"。要求六个面都 ≥ CLEARANCE。
  let off = vec3<f32>(center_v - sub16);
  let face_lo = min(min(off.x, off.y), off.z);
  let face_hi = min(min(15.0 - off.x, 15.0 - off.y), 15.0 - off.z);
  let center_ok = sample_brickmap(g, center_v) == 0u
    && ddgi_cell_state_sized(g, sub16, 16) == 0u
    && min(face_lo, face_hi) >= DDGI_PROBE_MIN_CLEARANCE;
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

// 4 个 per-LOD 入口：Rust 侧按 lod 0→3（细→粗）各派发一个独立 compute pass，
// pass 边界即内存屏障 ⇒ 粗级能读到本帧细级刚写好的 ddgi_cell / ddgi_cell_id。
@compute @workgroup_size(64)
fn ddgi_bake0(@builtin(global_invocation_id) gid: vec3<u32>) { ddgi_bake_one(0u, gid.x); }
@compute @workgroup_size(64)
fn ddgi_bake1(@builtin(global_invocation_id) gid: vec3<u32>) { ddgi_bake_one(1u, gid.x); }
@compute @workgroup_size(64)
fn ddgi_bake2(@builtin(global_invocation_id) gid: vec3<u32>) { ddgi_bake_one(2u, gid.x); }
@compute @workgroup_size(64)
fn ddgi_bake3(@builtin(global_invocation_id) gid: vec3<u32>) { ddgi_bake_one(3u, gid.x); }

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
  // 必须用 ddgi_slot_own：这里问的是"那个 cell 自己有没有体素"，而 ddgi_slot 会把它
  // 重定向到邻近探针所在的槽位（读到的就是邻居的记录，判定会串位）。
  return (ddgi_cell[ddgi_slot_own(lod, wc)] & DDGI_REC_OCCUPIED) != 0u;
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
    // Step 1：本格放不出满足净距的探针 → 指向同 LOD 邻近**已放置**的探针（找不到则指回自身）。
    // 这样贴墙一圈不会留下空洞，而探针本身仍可以离表面足够远。搜索半径由 params.w
    // （Borrow 滑杆）控制，r=0 时直接指回自身 = 借针关闭，该插值角在采样时自然缺席。
    ddgi_cell_slot[slot] = ddgi_find_neighbor_probe(slot);
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
    // 旧版读的是从未被写入的 ddgi_objects（旧门控量 u.params.w 当时恒 0；注意该通道
    // 现在已改作借针半径，见 ddgi_find_neighbor_probe，与本判定无关）→ 那段是死代码，
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
  // 本格有自己的探针 → 指回自身（清掉上一帧可能留下的"指向邻居"）
  ddgi_cell_slot[slot] = slot;
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
  // 上界必须是**本探针自己的**最后一个纹素（DDGI_IRR_TEXELS-1），不能写死 7：
  // 常量从 8 降到 4 之后，`p.x*DDGI_IRR_TEXELS + tx` 里 tx 一旦取到 4 就跨进了
  // **相邻探针**的纹素 0（45° 宽的一个八面体楔形，插值权重可达 0.5）——
  // 于是整个楔形方向取到的是别人的辐照度，在表面上表现为**跟着探针投影走的大三角锯齿**。
  let tmax = i32(DDGI_IRR_TEXELS) - 1;
  let x0 = clamp(i32(g0.x), 0, tmax);
  let y0 = clamp(i32(g0.y), 0, tmax);
  let x1 = clamp(i32(g0.x) + 1, 0, tmax);
  let y1 = clamp(i32(g0.y) + 1, 0, tmax);
  let c00 = ddgi_irr_fetch(id, x0, y0);
  let c10 = ddgi_irr_fetch(id, x1, y0);
  let c01 = ddgi_irr_fetch(id, x0, y1);
  let c11 = ddgi_irr_fetch(id, x1, y1);
  return mix(mix(c00, c10, vec3<f32>(f.x)), mix(c01, c11, vec3<f32>(f.x)), vec3<f32>(f.y));
}
fn ddgi_depth_fetch(id: u32, x: i32, y: i32) -> vec2<f32> {
  let c = ddgi_depth_coord(id, u32(x), u32(y));
  return textureLoad(ddgi_depth, vec2<i32>(vec2<u32>(c.y, c.z)), i32(c.x), 0).xy;
}
// 返回 vec2(mean, std)：距离均值与该均值的标准差（图集 .x/.y）。
// 在原方向最近邻之外再取 2×2 邻域**取平均**：深度图每个纹素是 rpp 条射线估出来的，
// 射线方向逐帧重随机 → 单纹素读数逐帧抖动；4 点平均把噪声压 ~2×，同时抹掉最近邻在
// 纹素边界上的硬跳变。平均后的方差用 std 的平方近似（只用于软遮挡过渡宽度）。
// 全 0（无数据）纹素不参与；四个都无数据才回 0（采样侧按「无数据」剔除）。
fn ddgi_depth_sample(id: u32, d: vec3<f32>) -> vec2<f32> {
  let s = f32(DDGI_DEPTH_TEXELS);
  let e = ddgi_oct_encode(d) * 0.5 + vec2<f32>(0.5);
  let g2 = e * s - vec2<f32>(0.5);
  // 上界必须是**本探针自己的**最后一个纹素（DDGI_DEPTH_TEXELS-1），不能写死 15：
  // 常量从 16 降到 8 之后，`p.x*DDGI_DEPTH_TEXELS + tx` 里 tx 一旦取到 8 就跨进了
  // **相邻探针**的纹素 0 —— 2×2 平均会把这个"别人的深度"掺进来，让那一整个八面体楔形
  // 方向的 mean/std 变成错值 → Chebyshev 的 wd 在这个楔形里整体偏 0 或整体偏 1 →
  // 探针在楔形内外**成片入选/落选**，而归一化平均对集合变化敏感 → 表面上一块块
  // 「跟着探针投影走的三角亮/暗锯齿」。
  let tmax = i32(DDGI_DEPTH_TEXELS) - 1;
  let x0 = clamp(i32(floor(g2.x)), 0, tmax);
  let y0 = clamp(i32(floor(g2.y)), 0, tmax);
  let x1 = clamp(x0 + 1, 0, tmax);
  let y1 = clamp(y0 + 1, 0, tmax);
  let v00 = ddgi_depth_fetch(id, x0, y0);
  let v10 = ddgi_depth_fetch(id, x1, y0);
  let v01 = ddgi_depth_fetch(id, x0, y1);
  let v11 = ddgi_depth_fetch(id, x1, y1);
  var acc = vec2<f32>(0.0);
  var n = 0.0;
  if (v00.x > 0.0) { acc = acc + v00; n = n + 1.0; }
  if (v10.x > 0.0) { acc = acc + v10; n = n + 1.0; }
  if (v01.x > 0.0) { acc = acc + v01; n = n + 1.0; }
  if (v11.x > 0.0) { acc = acc + v11; n = n + 1.0; }
  return select(vec2<f32>(0.0), acc / max(n, 1.0), n > 0.0);
}
/// p 到 LOD(lod) 盒**外边界面**的距离：盒内为正、盒外为负（取六个面中最小者）。
/// 级联混合带的判据用它 —— 见 ddgi_cascade_blend。
fn ddgi_box_edge_dist(lod: u32, p: vec3<f32>) -> f32 {
  let L = ddgi_u.lods[lod];
  let lo = vec3<f32>(L.origin.xyz);
  let hi = lo + vec3<f32>(L.dims.xyz) * f32(L.origin.w);
  return min(
    min(min(p.x - lo.x, hi.x - p.x), min(p.y - lo.y, hi.y - p.y)),
    min(p.z - lo.z, hi.z - p.z),
  );
}
// p 是否落在 LOD(lod) 的「壳」内：在 LOD(lod) 盒内、且不在更细一级 LOD 盒内。
// 嵌套级联下任意点恰好属于一级 → **主采样**只有一个 LOD（保证只在探针真正活跃的区域取数）；
// 只有落在混合带内（见 ddgi_cascade_blend）时才额外采相邻一级做过渡。
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
// relax_normal = true：把背向探针闸门 wn 的下限抬到 0.25。**只在像素不在任何级联盒内时**
// 使用；壳内回退到更粗 LOD 时必须为 false，否则隔墙探针会被放进平均 → 漏光。
fn ddgi_sample_lod(p: vec3<f32>, n: vec3<f32>, lod: u32, clamp_cells: bool, relax_normal: bool) -> vec4<f32> {
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
  // 覆盖度：只累加「有可用数据」的角（见下面的分界注释）。与 wsum 的区别是本函数的核心 ——
  var cov = 0.0;
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
        // 插值基 = **晶格名义位置**上的三线性基（用上面的 `fr`），不是探针实际位置上的 tent。
        //
        // 【为什么必须用名义基】tent 的支撑一旦超过半格（旧 DDGI_SHARE_K = 2），越过一个
        // cell 边界时「离开 stencil 的那个角」权重仍有 ~0.5，但它已不再被枚举 → 这一份
        // 贡献被**整体丢弃**。两侧归一化后的结果因此相差 ~0.33·Δirr（相邻探针辐照度之差），
        // 也就是重量场在每个 cell 边界上有一个硬跳变：表现就是**跟着探针晶格走的放射状
        // 三角锯齿，周期 = cell 边长** ⇒ 粗级周期大 2×、离得稍远就格外显眼（wsum 档能直接
        // 看到同一图案，因为那是权重场本身在跳）。
        //
        // 名义三线性基的性质正好相反：8 个角上恒好和为 1，且在 cell 边界处该角的权重恰好
        // 归零 → 权重场连续、无跳变。探针的**实际位置**依旧参与 wn/wd（朝向与可见性必须用
        // 真实几何关系 —— 这正是 Douglas 说的那一步：
        //   "we fetch the probes' offsets within their cells and we compute additional blending
        //    factors based upon which probes have line of sight visibility to the voxel and which
        //    probes are in front of the voxel versus which probes are behind the voxel."
        // 而"哪个值算在哪个格点上"是插值的事，与探针在格内的偏移无关）。
        //
        // 顺带：Step 1 的"指向"共享过来的探针拿到的就是这个角的名义权重，不再需要放大
        // 支撑半径（DDGI_SHARE_K 已删）。
        let wl = vec3<f32>(
          select(1.0 - fr.x, fr.x, ix == 1),
          select(1.0 - fr.y, fr.y, iy == 1),
          select(1.0 - fr.z, fr.z, iz == 1),
        );
        let wtri = wl.x * wl.y * wl.z;
        if (wtri <= 1e-6) { continue; }
        let probe = ddgi_slot_pos[slot].xyz;
        // 【不再去重】旧的去重是在"支撑半径按探针实际位置算"时加的：那时同一个探针从多个角
        // 贡献的是**同一个权重**，累加等于把它的位置权重成倍放大。现在基是名义三线性基
        // （8 角和 = 1），同一份辐照度落在两个角上，本就该拿这两个角的权重之和 —— 那正是
        // "两个角的代表是同一个值"的正确插值结果。去重反而会把它压成其中一个角的基值，
        // 等于把一个共享探针又摆回了那个角点，晶格伪影会原样回来。
        let to = ps - probe;
        let dist = length(to);
        let dir = to / max(dist, 1e-4);
        // ================= 「这个角有没有可用数据」到此为止 =================
        // 上面过的闸门（窗口 / ENABLED / ACTIVE / age / wtri>0）加上下面这条「该方向深度纹素
        // 有记录」共同回答一个问题：**本点是"有探针数据"还是"根本没有可用的数据"**。
        // 天光常量兜底必须由这个问题决定（cov），**不能**由 wsum 决定：
        //   wn（探针在表面背面）/ wd（被墙挡住）是**照明决策** —— 它们把权重压到 0 是"这个
        //   探针不该给本点照明"，那是**正确结果、该点本来就该是暗的**，不是"数据不可信"。
        // 用 wsum 判的后果正是室内墙角漏光：凹角处相邻两堵墙互相遮挡，存活权重本来就低 →
        // wsum 塌到阈值以下 → 判成"不可信" → 把常量天光加回来 → 最该黑的地方反而最亮；
        // 而 cell 越粗（离相机越远）遮挡剔除越多 → 漏光越明显，完全对得上实机现象。
        let dtex = ddgi_depth_sample(slot, dir);
        // 该方向没有任何命中记录（纹素从未被写过，或刚被 collect 显式清零）→ **无数据**，
        // 不是「无遮挡」：判为不可见并计入无数据闸门。旧版把「无数据」当「无遮挡」→ 新鲜
        // 探针以 0 辐照度满权重参与平均 → 大片「白（辐照度≈0）↔绿」翻转（动态闪烁）。
        if (dtex.x <= 0.0) { ddgi_dbg_rej.w = ddgi_dbg_rej.w + 1u; continue; }
        // 到这里 = 本角有可用数据 → 计入覆盖度（无论它接下来会不会被 wn/wd 挡掉）
        cov = cov + wtri;
        // ============ 往下是照明决策（只影响 wsum，不影响 cov）============
        // 背向探针闸门：**两条路径一律剔除**（含放宽路径）。放宽（wn 下限 0.25）是为薄几何的
        // **掠射**探针准备的（探针在表面侧方，dot ≈ 0），不是为**背向**探针（探针在表面背后）。
        //
        // 【为什么必须先用未 clamp 的 dot 判定】`clamp(dot, 0, 1)` 把 dot≈0（掠射）与 dot<0
        // （背向）抹成同一个 0，放宽路径的 `max(wn, 0.25)` 于是把背向探针也拉进平均。而墙后/
        // 箱体内部的探针辐照度 ≈0（那里本来无光 —— 例如这个大圆柱是个罐体，背面就是罐内），
        // 一旦入选就把像素压成**暗斑**；哪几个角入选又随 cell 邻域摆动 → 斑块跟着晶格走、
        // 随相机跳变。这正是「GI 模式里也有暗斑」的来源。
        let wn_raw = dot(n, -dir) / DDGI_NORMAL_BIAS;
        if (wn_raw <= 0.0) {
          ddgi_dbg_rej.y = ddgi_dbg_rej.y + 1u;
          continue;
        }
        let wn = clamp(wn_raw, 0.0, 1.0);
        let wn_w = select(wn, max(wn, 0.25), relax_normal);
        // ---- 遮挡：Chebyshev 软判定（对齐 Majercik / RTXGI）----
        // 旧版是 `clamp((dtex-dist)/bias+1)` + `<=0 剔除` 的**刀锋**判定：dtex 抖一点，
        // 探针就在「入选/落选」之间翻 → 亮区边界伸缩（静止相机也闪）；而要把 bias 调大到
        // 不翻，就必然大面积漏光。Chebyshev 用 mean + 方差：**深度越不确定（std 越大）
        // 过渡越宽**，只有深度一致（真是一堵墙）时才接近硬判定 → 同时兼顾抗漏与抗闪。
        // 【前提】dtex.y 必须真的是"这个 mean 有多不确定"。旧版在 collect 侧存的是**本帧**
        // std，而该纹素每帧平均只摊到 ~0.1 条射线（大多数帧只有一条）→ 本帧 std 恒为 0 →
        // soft 永远塌到 0.02·mean 的刀锋，上面那句"过渡越宽"从来没生效过（静止也闪的根因）。
        // 现在 collect 存的是样本序列的 std（见那里的注释），这里不用改判据。
        // d_vis 减一个小偏置：贴在该深度图记录的那个面上的采样点，不应因浮点误差落到
        // mean 外侧被判成全额遮挡。**这个偏置不能大**：它等价于"允许点比最近表面再近
        // bias"，会在每处贴面/接缝制造一条 bias 宽的漏带（旧值 cs*0.1 = LOD0 3.2 体素，
        // 与深度均值偏大叠加就是可见亮带）。改成 cs*0.05，让 std/Softness 去吸收抖动。
        let d_vis = dist - max(cs * 0.02, dist * 0.005);
        let mean = dtex.x;
        var wd = 1.0;
        if (d_vis > mean) {
          // [诊断] std 项的信任系数（misc.z，默认 1）：拖到 0 则完全忽略 dtex.y、退成
          // 固定 2% 的硬判定；1 = 正常使用。射线绑定深度纹素后 dtex.y 已是「同一方向
          // 深度读数的跨帧噪声」（此前度量的是「20° 锥内几何变化的离散度」，见 cast 注释），
          // 该滑杆保留作 A/B：0 更能压漏光，但过渡带变窄、动态时更易闪。
          let soft = max(dtex.y * ddgi_u.misc.z, DDGI_DEPTH_SOFT_MIN);
          let delta = d_vis - mean;
          wd = (soft * soft) / (soft * soft + delta * delta);
        }
        // wd 极小 = 实质遮挡：保留统计与提前退出（不再做 0/1 硬剔除，避免临界翻转）
        if (wd <= 1e-3) { ddgi_dbg_rej.z = ddgi_dbg_rej.z + 1u; continue; }
        let irr = ddgi_irr_sample(slot, n);
        // 【旧闸门已删】这里曾经有一道 `max(irr) < 1e-4 → 剔除`，
        // 理由是"已写入的纹素恒 > 0（radiance ≥ albedo·sky·DDGI_CAST_FLOOR）"。但
        // **DDGI_CAST_FLOOR 后来被改成 0**（室内要能压黑），这个前提就不成立了：射线打在
        // 全黑表面 / 暗室里的**已写入**纹素，值本来就是 0 —— 用值阈值会把"真的暗"当成
        // "没数据"剔掉。
        //
        // 为什么它会造成**静态也闪 / 亮区边界伸缩**：纹素值每帧按 age 斜率做 alpha 混合，
        // 而 rpp 只有几条射线 → 值本来就在抖动。值在 1e-4 附近抖动时，该探针会逐帧在
        // "入选/落选"之间翻，而归一化平均（total/wsum）对**集合变化**很敏感（低 wsum 的
        // 像素尤其：1~2 个探针就占满结果）→ 画面闪、亮区边界伸缩。
        // Domain/Probe 档看不出来：它们只看几何归属与闸门直方图，不看这个连续值。
        //
        // "这个方向有没有被采样过"由**深度纹素**负责（未采样 → snap 清零 → 上面的
        // `dtex.x <= 0` 已剔除；命中/未命中都会写深度），所以这里直接用采样值即可：
        // 真的暗方向本来就该以 ~0 参与平均。
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
  // 覆盖度单独带出（返回值只有 4 个分量，且 .w 已被 wsum 占用）—— 着色侧用它决定
  // "要不要退回常量天光"，见 ddgi_dbg_cov 声明处与 ddgi_sample_lod 的分界注释。
  ddgi_dbg_cov = cov;
  return vec4<f32>(select(vec3<f32>(0.0), total / wsum, ok), wsum);
}
var<private> ddgi_dbg_wsum: f32;
// 「有可用数据的角」的名义权重覆盖度（Σ wtri，**不含** wn/wd 两道照明闸门）。
// 它是天光常量兜底的唯一判据 —— 见 ddgi_sample_lod 里 cov 的注释。
var<private> ddgi_dbg_cov: f32;
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

/// 级联混合：主级 `lod` 已给出结果 `main_c`，若 p 落在混合带内则与相邻级按权重混合，
/// 消除 LOD 盒边界处「跨过一步就换一套光照」的硬边（见 DDGI_CASCADE_BLEND）。
///
/// 两个方向各自独立判断；**边界两侧都恰好停在 0.5/0.5** → 跨边界连续：
///   ① **外侧**（p 刚离开更细一级 LOD(lod-1) 的盒）：与更细级混合。更细级在盒外，必须
///      `clamp_cells=true` 取边界格的数据；`relax_normal=false` —— **绝不放宽背向闸门**，
///      放宽会把隔墙探针拉进平均（漏光）。
///   ② **内侧**（p 靠近本级盒的外边界）：与更粗一级 LOD(lod+1) 混合。更粗级在 p 处本来
///      就在其盒内（盒严格嵌套）→ 正常采样，不 clamp、不放宽。
/// 权重 t = 0.5·(1 − d/W)：d=0（恰在边界）时 0.5，d ≥ W（出了混合带）时 0。
/// ⇒ 边界内侧 = 0.5·LOD(k) + 0.5·LOD(k+1)，外侧 = 0.5·LOD(k+1) + 0.5·clamp(LOD(k))，
///   两者近似相等 ⇒ 跨边界亮度连续。
/// 伙伴取不到数据（wsum < 1e-4）时退回纯主级 —— 混合只负责平滑，不改变「谁有数据」。
fn ddgi_cascade_blend(p: vec3<f32>, n: vec3<f32>, lod: u32, main_c: vec3<f32>) -> vec3<f32> {
  let W = f32(ddgi_u.lods[lod].origin.w) * DDGI_CASCADE_BLEND;
  if (W <= 0.0) { return main_c; }
  // 混合伙伴会覆写这批全局诊断量 → 先存主级的，返回前恢复（Probe/Domain 档才不被搅乱）
  let dbg_rej = ddgi_dbg_rej_out;
  let dbg_wsum = ddgi_dbg_wsum;
  let dbg_dom = ddgi_dbg_dom;
  let dbg_fb = ddgi_dbg_fb;
  let dbg_cov = ddgi_dbg_cov;
  let dbg_zero = ddgi_dbg_zero;
  var out = main_c;
  if (lod > 0u) {
    // ① 外侧：p 在 LOD(lod-1) 盒外（距离为正）且落在混合带内
    let d = -ddgi_box_edge_dist(lod - 1u, p);
    if (d >= 0.0 && d < W) {
      let rf = ddgi_sample_lod(p, n, lod - 1u, true, false);
      if (rf.w >= 1e-4) {
        out = mix(main_c, rf.xyz, 0.5 * (1.0 - d / W));
      }
    }
  }
  if (lod + 1u < DDGI_LOD_COUNT) {
    // ② 内侧：p 在 LOD(lod) 盒内（距离为正）且靠近其外边界
    let d = ddgi_box_edge_dist(lod, p);
    if (d >= 0.0 && d < W) {
      let rc = ddgi_sample_lod(p, n, lod + 1u, false, false);
      if (rc.w >= 1e-4) {
        out = mix(main_c, rc.xyz, 0.5 * (1.0 - d / W));
      }
    }
  }
  ddgi_dbg_rej_out = dbg_rej;
  ddgi_dbg_wsum = dbg_wsum;
  ddgi_dbg_dom = dbg_dom;
  ddgi_dbg_fb = dbg_fb;
  ddgi_dbg_cov = dbg_cov;
  ddgi_dbg_zero = dbg_zero;
  return out;
}

fn ddgi_sample(p: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
  var in_cascade = false;
  var shell_lod = DDGI_LOD_COUNT;
  for (var lod = 0u; lod < DDGI_LOD_COUNT; lod = lod + 1u) {
    if (ddgi_lod_contains(lod, p)) {
      in_cascade = true;
      shell_lod = lod;
      let r = ddgi_sample_lod(p, n, lod, false, false);
      ddgi_dbg_rej_out = ddgi_dbg_rej;
      ddgi_dbg_wsum = r.w;
      ddgi_dbg_dom = f32(lod) + 1.0;
      ddgi_dbg_fb = 0.0;
      if (r.w >= 1e-4) {
        // 主级成功 → 过一遍级联混合（不在混合带内则原样返回）
        return ddgi_cascade_blend(p, n, lod, r.xyz);
      }
      // 壳内 8 角被 wn 背向闸门**全剔**的最常见来源是薄几何（线缆/薄板/薄边/薄梁）：
      // 贴着细长物件时，最近的空叶都在**侧面**（上面/下面/两侧），dir 与命中面法线近乎
      // 垂直 → wn≈0 → 被 `wn <= 0` 硬剔 → 壳内一个角都不剩。
      // 对这种情形优先用**同一级**放宽 wn（下限 0.25）重试，而不是直接退到粗级 LOD：
      //   · 同级探针只在 32~64cm 外，16×16 深度图在这个尺度上可信 → 即使放宽 wn 也不会
      //     让隔墙探针漏进来；
      //   · 粗级探针在 1.3~2.6m 外、且多为掠射方向 → **超出深度图的角度分辨率** → 掠射
      //     方向上的薄墙在深度图里"看起来"在更远处 → wd 判不出遮挡。而 ddgi_sample 返回的是
      //     **归一化**平均（一个探针入选就占满结果）→ 一个隔墙粗级探针就能把整片点亮成
      //     跟着粗级 cell 走的亮条纹 —— 封闭空间里那些"细条"就是这么来的。
      //   · 仍是同一个 cell 邻域（不 clamp、不换级），位置连续、不产生块状跳变。
      //   · 放宽**只抬掠射探针的权重**（wn_raw > 0 但很小）；wn_raw <= 0 的背向探针在任何
      //     路径下都被剔除 —— 它们是墙后/箱体内部的探针（辐照度≈0），拉进来会把像素压成
      //     「跟着 cell 晶格走、随相机跳变」的暗斑（见 ddgi_sample_lod 的闸门注释）。
      let rej_shell = ddgi_dbg_rej; // 诊断用：保留未放宽时的剔除统计
      let r2 = ddgi_sample_lod(p, n, lod, false, true);
      if (r2.w >= 1e-4) {
        ddgi_dbg_wsum = r2.w;
        ddgi_dbg_fb = 3.0; // 3 = 同级放宽 wn 后成功（Probe 档亮黄）
        return ddgi_cascade_blend(p, n, lod, r2.xyz);
      }
      ddgi_dbg_rej = rej_shell;
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
  // clamp/放宽只留给「不在任何壳内」的退化路径：
  //   ① 壳内回退（in_cascade）：盒子严格嵌套 → p 必然也在更粗级的窗口内，无需 clamp；
  //      且**不放宽 wn** —— 否则背向/隔墙探针也会参与平均，正是漏光的来源之一。
  //   ② 壳外（!in_cascade）：越界 cell 必须 clamp 到最粗一级窗口才取得到数据，并放宽 wn。
  let relax_normal = !in_cascade;
  for (var lod = start; lod < DDGI_LOD_COUNT; lod = lod + 1u) {
    let r = ddgi_sample_lod(p, n, lod, true, relax_normal);
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
// **射线方向 = 绑定的八面体深度纹素方向 + 纹素内抖动**（纹素按 frame 轮换覆盖，见函数内注释）。
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

  // 【射线 ↔ 深度纹素绑定】每条射线固定归属一个八面体深度纹素，纹素按
  // (frame·rpp + ray) 轮换 → 每帧覆盖 rpp 个不同纹素、⌈N²/rpp⌉ 帧轮完一圈（N=8 → 64 个）；方向 =
  // 该纹素中心 + 纹素内抖动（抖动避免固定方向产生 banding，幅度恰好限制在纹素内）。
  //
  // 为什么必须绑定（旧版是 Fibonacci 球 + 每帧随机四元数**整体重旋**）：
  //   旧版 4 条射线撒在整球的 64 个纹素上，每个纹素每 ~16 帧才摊到 1 条、方向还完全
  //   随机 → collect 存进深度图的 mean/std 变成「该 20° 锥内**随机方向**命中距离的混合」，
  //   而不是 Chebyshev 需要的「**该方向**的最近距离 + 其噪声」：
  //     · mean 在墙角被远处命中抬高 → 采样点 dist > mean 不成立 → wd = 1 全额通过（硬漏光）
  //     · std 变成「锥内几何起伏」而非「同方向噪声」→ 墙角最大 → soft 巨大（软漏光）
  //   两者都随 LOD 变粗（cell 大、锥内起伏大）而加剧，正对上「LOD1 比 LOD0 漏得明显」。
  //   绑定后每个纹素的样本跨帧同属一个纹素方向 → mean/std 恢复物理含义，Chebyshev 成立。
  //   （DDGI 原版用 256 条低差异射线铺满 16×16 深度图 = 每纹素 1 条；这里在预算远小于
  //   纹素数时用「帧轮换」等价地保证「每条纹素最终拿到的是它自己的方向」。）
  let ndim = DDGI_DEPTH_TEXELS;
  let ntex = ndim * ndim;
  let tsel = (frame * rpp + ray) % ntex;
  let ttx = tsel % ndim;
  let tty = tsel / ndim;
  let jit = ddgi_rand2(ddgi_ray_rand(ddgi_lod_slot_base(lod) + cell_idx, frame, ray));
  let te = ((vec2<f32>(f32(ttx), f32(tty)) + jit) / f32(ndim)) * 2.0 - vec2<f32>(1.0);
  let dir = ddgi_oct_decode(te);
  let sh = trace_scene(probe_pos + dir * DDGI_RAY_BIAS, dir, DDGI_T_MAX, 0.0, 3u);
  var radiance = sky_rgb();
  var dist = DDGI_T_MAX;
  if (sh.uh.hit) {
    dist = sh.uh.t;
    let alb = palette_albedo(sh.palette_base, sh.uh.pal);
    let hit_p = probe_pos + dir * dist;
    // 与 dda_main 同一套逐体素法线（退化退面法线）：反弹辐照度的方向必须与着色侧一致，
    // 否则曲面上的间接光会与直射/环境光"对不上"。代价是每条 GI 射线 6 次 sample_brickmap
    // （射线数 ≈ 屏幕像素数的 1/7，可接受）。
    let n = voxel_normal(
      make_grid(u32(sh.uh.obj_id) + 1u),
      sh.uh.voxel,
      sh.uh.n,
    );
    // 入射辐照度 E → 出射辐亮度 L = albedo·E/π（与着色侧 col += albedo·E/π 同约定）。
    // 加一项 albedo·sky·CAST_FLOOR 作下限，避免首帧 GI 全 0 时反馈回路死锁在 0；
    // 该下限必须小，否则封闭房间的探针也被抬亮（室内外没有亮度差）。
    let gi = ddgi_sample(hit_p + n * 0.5, n);
    // 命中发光体素 → 出射辐亮度含**自身发射**（直出，不吃 GI/入射）：这是"体素即光源"
    // 往 GI 里注入光的关键接法（Douglas：命中体素取它的 lit color，发光体素的 lit color 含 emission）。
    radiance = alb * (gi / DDGI_PI + sky_rgb() * DDGI_CAST_FLOOR)
      + alb * palette_emissive(sh.palette_base, sh.uh.pal) * DDGI_EMIT_GAIN;
  }
  // 样本下标 = 全局射线编号（tid 本身就是「该 LOD 起始射线号 + 本探针号×rpp + 射线号」）
  let si = tid * 2u;
  // 沿程发光聚合（「体素即光源」）：叠加在命中辐亮度之上，累加到几何首命中（无命中 → T_MAX）。
  // 这是**间接**贡献（发光 cell 摊成的体积密度）；命中发光体素时的自身发射由上面的
  // palette_emissive 直出项负责，两者语义不重叠。
  radiance = radiance + light_field_gather(probe_pos + dir * DDGI_RAY_BIAS, dir, dist);
  ddgi_samples[si] = vec4<f32>(dir, dist);
  ddgi_samples[si + 1u] = vec4<f32>(radiance, 1.0);
}

// ---- 空间混合（RTXGI ProbeBlending 的简化）----
// 采样端 gi = Σ(irr·w)/Σw，w = wtri·wn·wd。`wn` 会把"探针落在背面"的那半边剔掉，于是
// 同一个 cell 内权重在「探针处 ≈1 ↔ cell 角 ≈0.5」之间变化；归一化后，加权平均会把
// **每个探针自己的辐照度**顶出来 → 与探针间距同周期的晶格亮斑（墙角/接缝亮带的来源）。
// 相邻探针的辐照度越接近，这个重分配越看不出来 → 把 6 个网格邻居的同名纹素混进来。
fn ddgi_irr_nbr(id: u32, tx: u32, ty: u32) -> vec4<f32> {
  let c = ddgi_irr_coord(id, tx, ty);
  return textureLoad(ddgi_irr, vec2<i32>(vec2<u32>(c.y, c.z)), i32(c.x), 0);
}
/// 6 个轴向邻居的同名纹素按探针**实际间距**加权求和；返回 (加权和, 权重和)。
/// 只接受同 LOD 网格内、本帧 ENABLED 且**该纹素有数据**的邻居。
///
/// 【为什么不能按值过滤】旧版是 `max(v) < 1e-4 → 跳过`。`DDGI_CAST_FLOOR` 改成 0 之后
/// "已写入但真黑"与"从未写过"的值**都是 0**，按值判会把**暗邻居整批踢掉** —— 而暗邻居
/// 恰恰是稀释亮邻居的唯一手段。后果：一颗落点在墙另一侧、看得到天光的探针，四周的暗邻居
/// 全部不参与平均 ⇒ 它完全不被稀释 ⇒ 在自己的投影处形成一个亮斑（室内墙角/接缝的亮边；
/// 粗级 cell 更大、更多邻居是实心格或未成型 ⇒ 更严重，对上"离越远越明显"）。
/// 现在按第 4 通道的**覆盖度**判"有没有数据"（见 DDGI_TEXEL_MIN_COVERAGE）。
fn ddgi_probe_blend_irr(lod: u32, pidx: u32, tx: u32, ty: u32) -> vec4<f32> {
  let dims = ddgi_u.lods[lod].dims.xyz;
  let base = ddgi_lod_slot_base(lod);
  let cell = ddgi_slot_cell(lod, pidx);
  let cs = f32(ddgi_u.lods[lod].origin.w);
  let self_pos = ddgi_slot_pos[base + pidx].xyz;
  var acc = vec3<f32>(0.0);
  var wacc = 0.0;
  for (var a = 0u; a < 3u; a = a + 1u) {
    let stride = select(select(1u, dims.x, a == 1u), dims.x * dims.y, a == 2u);
    let extent = select(select(dims.x, dims.y, a == 1u), dims.z, a == 2u);
    let coord = select(select(cell.x, cell.y, a == 1u), cell.z, a == 2u);
    for (var s = 0u; s < 2u; s = s + 1u) {
      let plus = s == 1u;
      if (plus && coord + 1u >= extent) { continue; }
      if (!plus && coord == 0u) { continue; }
      let nidx = select(pidx - stride, pidx + stride, plus);
      let nslot = base + nidx;
      if ((ddgi_meta[nslot] & DDGI_META_ENABLED) == 0u) { continue; }
      let v = ddgi_irr_nbr(nslot, tx, ty);
      if (v.w < DDGI_TEXEL_MIN_COVERAGE) { continue; }
      let npos = ddgi_slot_pos[nslot].xyz;
      let w = max(0.0, 1.0 - length(npos - self_pos) / (cs * 2.0));
      acc = acc + v.xyz * w;
      wacc = wacc + w;
    }
  }
  return vec4<f32>(acc, wacc);
}

// ============================================================================
// 阶段三 collect（Douglas pass #3）：射线样本积成图集。
//   · irradiance 4×4：按「样本方向 · 纹素方向」半球余弦加权（来自任何方向的射线都能贡献）。
//   · depth 8×8：**只吃绑定到本纹素的射线**（见 cast 的「射线 ↔ 深度纹素绑定」）→
//     mean/std 是「该方向最近距离」的均值与其噪声，Chebyshev 判定成立的前提。
// 每探针 DDGI_COLLECT_THREADS 个线程：texel < 16 → irradiance 4×4；否则 depth 8×8。
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
      // 先空间混合（和 6 邻居同名纹素），再时域混合 —— 见 ddgi_probe_blend_irr 注释
      let nb = ddgi_probe_blend_irr(lod, atlas_slot - ddgi_lod_slot_base(lod), tx, ty);
      let irr_s = (irr_new * DDGI_BLEND_SELF + nb.xyz) / (DDGI_BLEND_SELF + nb.w);
      let cc = ddgi_irr_coord(atlas_slot, tx, ty);
      let idx = vec2<i32>(vec2<u32>(cc.y, cc.z));
      let prev = textureLoad(ddgi_irr, idx, i32(cc.x), 0);
      // 年龄越小时向新样本靠拢越多（1/age 等权滑动平均）→ 新探针几帧内成型；
      // 年龄大了自动切回 DDGI_ALPHA，稳态脉动不变。见 ddgi_age_alpha。
      let a = ddgi_age_alpha(age, DDGI_ALPHA);
      // .w = 该纹素的**覆盖度**：每被写过一次就向 1 靠拢一次；从未写过则恒为 0。
      // 只有它能把"已写入但真黑"与"从未写过"分开（两者的值都是 0）—— 空间混合需要这个
      // 信号才能拿暗邻居稀释亮邻居，见 DDGI_TEXEL_MIN_COVERAGE / ddgi_probe_blend_irr。
      textureStore(
        ddgi_irr_out,
        idx,
        i32(cc.x),
        vec4<f32>(mix(prev.xyz, irr_s, a), mix(prev.w, 1.0, a)),
      );
    } else if (snap) {
      // 新（重）烘的探针：本帧没有射线覆盖到的方向 → **显式清零**。
      // 不清零的话这里留着的是「该槽位上一任世界 cell」的读数（槽位是世界锚定环面映射，
      // 相机滚动时新进入窗口的格子会复用刚离开格子的槽位）→ 会被当成有效数据使用。
      // 清零后采样侧的两条「无数据」判据（dtex<=0 / irr≈0）才能精确识别它。
      let cc = ddgi_irr_coord(atlas_slot, tx, ty);
      textureStore(ddgi_irr_out, vec2<i32>(vec2<u32>(cc.y, cc.z)), i32(cc.x), vec4<f32>(0.0));
    }
  } else {
    // ---- depth：本纹素绑定射线的**命中**距离统计 ----
    // 两条铁律：① 只统计**属于本纹素**的射线（绑定，见 cast）；② 只累计**命中**射线。
    // ② 的理由：早期版本把未命中（dist = DDGI_T_MAX = 8192）也加进均值，只要有射线看到
    // 天空，均值就被抬到数百且回落极慢 → dtex 系统性偏大 → 遮挡判定几乎恒为「可见」
    // → 深度闸门形同虚设 → 严重漏光。只累计命中后 dtex 逼近「该方向最近表面距离」；
    // 该方向一条都没命中 → 无遮挡物 → 不更新（保留上次命中均值；从未命中则恒 0 =
    // 无数据，采样侧按「无数据」剔除该角）。
    let t = texel - DDGI_IRR_TEXELS * DDGI_IRR_TEXELS;
    let tx = t % DDGI_DEPTH_TEXELS;
    let ty = t / DDGI_DEPTH_TEXELS;
    let frame = u32(ddgi_u.params.x);
    let ntex = DDGI_DEPTH_TEXELS * DDGI_DEPTH_TEXELS;
    var dsum = 0.0;
    var dsum2 = 0.0;
    var cnt = 0u;
    for (var i = 0u; i < rpp; i = i + 1u) {
      // 【与 cast 同一套绑定】本纹素只吃「属于它」的射线：(frame·rpp + i) % ntex == t。
      // 本帧可能对应 0 条（rpp < ntex 且未轮到）、1 条（rpp == ntex）或若干条（rpp > ntex）。
      // 这些射线的方向同属本纹素（只是纹素内抖动不同）→ 下面的 mean/mean2 就是
      // 「该方向最近表面距离」的均值与**噪声**方差，正是 Chebyshev 需要的量。
      // 旧版按 20° 锥权重把附近方向的射线也混进来，mean 被远处命中系统性抬高 →
      // 隔墙采样点被判「可见」→ 漏光（详见 cast 里的注释）。
      if ((frame * rpp + i) % ntex != t) { continue; }
      let s = ddgi_samples[(si + i) * 2u];
      // 未命中（dist = DDGI_T_MAX）不参与：本纹素存的是「该方向最近表面距离」，
      // 朝开阔方向的射线根本没有表面 → 不更新（保留上次命中均值；从未命中则恒为 0，
      // 采样侧按「无数据」剔除该角）。
      if (s.w < DDGI_T_MAX) {
        dsum = dsum + s.w;
        dsum2 = dsum2 + s.w * s.w;
        cnt = cnt + 1u;
      }
    }
    if (cnt > 0u) {
      let mean = dsum / f32(cnt);
      let mean2 = dsum2 / f32(cnt);
      let cc = ddgi_depth_coord(atlas_slot, tx, ty);
      let idx = vec2<i32>(vec2<u32>(cc.y, cc.z));
      let prev = textureLoad(ddgi_depth, idx, i32(cc.x), 0).xy;
      // 深度也走同样的年龄斜坡：新探针的深度图第一帧只摊到约 1 条射线/纹素，噪声极大，
      // 若沿用 0.03 的慢混合，遮挡判定会带着错误读数持续 ~1 秒（表现是"漏光/发暗慢慢退"）。
      let a = ddgi_age_alpha(age, DDGI_DEPTH_ALPHA);
      // 【为什么存的是「距离均值的跨帧序列 std」而不是本帧 std】
      // 绑定后每个纹素每 ⌈64/rpp⌉ 帧才拿到一条射线（rpp≈4 → 16 帧），绝大多数帧里
      // cnt 至多为 1 → 本帧 std 恒等于 0 → 采样侧
      // `soft = max(dtex.y, mean·0.02)` 就退化成 0.02·mean 的**刀锋**。
      // 而 mean 是单样本估计、跨帧方向抖动 → wd 在 ~1 与 ~0 之间翻 → 该角被
      // `wd <= 1e-3` 整块剔掉又放回 → wsum（以及跟着它走的光影）逐帧跳、亮区边界伸缩，
      // 相机静止也照样闪。Chebyshev 的过渡宽度本该由"深度读数有多不确定"决定，而不是
      // 一个恒定的 2%。
      // 做法：同时 EMA「距离」与「距离²」→ 存的是**落进本纹素的样本序列的 std**，
      // 即 mean 的真实不确定度尺度（同一纹素内抖动方向落在不同距离上的离散度）。
      // 注意 pm2 必须由上一帧的 (mean, std) 反推 —— 直接存 mean² 会在 f16 里溢出
      // （最远 8192 → 8192² = 6.7e7 ≫ f16 上限 65504）。中间量都是 f32，最后只存 std。
      let pm2 = prev.x * prev.x + prev.y * prev.y;
      let em = mix(prev.x, mean, a);
      let em2 = mix(pm2, mean2, a);
      let dep_std = sqrt(max(em2 - em * em, 0.0));
      // 存储纹素的 store 值类型是 vec4<f32>（naga 校验要求）：.x = mean、.y = std
      textureStore(
        ddgi_depth_out,
        idx,
        i32(cc.x),
        vec4<f32>(em, dep_std, 0.0, 0.0),
      );
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


