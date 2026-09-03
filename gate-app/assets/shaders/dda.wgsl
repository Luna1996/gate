// ============================================================================
// DDA Compute Shader：全屏逐像素两级 A&W 步进（cell 粗步 + cell 内细步）
// + brickmap 五步寻址
//
// 两级结构（P2.x 性能改造，Rust 参考 cpu_reference_dda_ray_two_level）：
//   粗级：cell（16 fine）粒度 A&W，每步 cell_occupied（寻址链 ①+②，2 load），
//         空 cell 一次跨越 16 fine——空气穿越访存 ~40× 低于单级 fine 步进。
//   细级：占用 cell 内有界 fine DDA（≤48 步），sample_brickmap 全链取 palette。
//
// BG0（与 Rust DdaViewUniform 144B + out tex rgba8unorm 对应）：
//   @group(0) @binding(0) = out storage tex（DDA 写入 rgba8unorm，linear RGB）
//   @group(0) @binding(1) = uniform DdaViewUniform（144B 对齐）
//
// BG1（与 Rust GpuBrickMap 资源 1:1 对应）：
//   @group(1) @binding(0) = b_struct: array<u32>（TileIndex + TileBitmaps + CellDirs + NodeStream 定长前缀 + 可变 node 区）
//   @group(1) @binding(1) = b_leaves: array<u32>（brick slab 池）
//   @group(1) @binding(2) = b_palette: array<u32>（palette 256 entries × 2 words = 512 words）
//   @group(1) @binding(3) = uniform BrickMapGlobals（scalar 字段 20×u32/i32 = 80B）
//
// BG2（Phase 3 OBJ→Volume 统一：GridDesc 数组，与 Rust GpuBrickMap.grid_descs_buf 1:1）：
//   @group(2) @binding(0) = grid_descs: array<GridDesc>（144B/entry，主世界 + 物体统一描述符）
//   shader `dda_main` 用 `arrayLength(&grid_descs)` 取 volume 数，遍历 trace_grid 无 kind 分支。
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

// palette 解包已并入 hit_mat(g.palette_base, pal)——多 volume 拼接后必须带
// palette_base 偏移，否则会读到别的 volume 的颜色。

// --- BG0：输出 + 视图 uniform（v5 single-pass per-pixel shade_hit）---
// @binding(0) = out storage write（rgba8unorm，linear RGB；ACES → sRGB 后输出）
// @binding(1) = uniform DdaViewUniform
@group(0) @binding(0) var out_tex: texture_storage_2d<rgba8unorm, write>;

struct DdaViewUniform {
  inv_view_proj: mat4x4<f32>,  // 64B
  cam_pos_fine: vec4<f32>,     // 16B，w=1
  debug_mode: vec4<f32>,       // 16B：x = 法向可视化，y = face 6 色诊断
}
@group(0) @binding(1) var<uniform> view_u: DdaViewUniform;

// --- BG1：brick map 全部数据 + globals ---
@group(1) @binding(0) var<storage, read> b_struct: array<u32>;
@group(1) @binding(1) var<storage, read> b_leaves: array<u32>;
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
  _pad0: u32,
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

// ---- BG3：光源池（P3.1；P3.2 发光元件并入，816B：48B header + 16×48B）----
struct LightGlobals {
  count: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
  ambient: vec4<f32>,      // rgb = 环境色（线性），w reserved
  exposure_pad: vec4<f32>, // x = 曝光系数
}
struct LightDesc {
  // x = kind（0 = 方向光 / 1 = 点光）；yzw = L 轴（指向光，已归一，方向光）或球心位置（fine，点光）
  kind_pos_dir: vec4<f32>,
  // rgb = 线性色，w = 强度（方向光无量纲；点光为米制衰减系数）
  color_intensity: vec4<f32>,
  // x = 方向光盘角半径（rad）/ 点光球半径（fine）
  shape: vec4<f32>,
}
struct LightPool {
  g: LightGlobals,
  lights: array<LightDesc, 8u>,
  sky_top: vec4<f32>,
  sky_horizon: vec4<f32>,
}
@group(3) @binding(0) var<uniform> light_u: LightPool;


// Douglas Brick Tree mask DDA 采样（1:1 复刻 devlog #17/#18）
// fine → chunk 窗口定位 → DFS 树 4 层 mask 遍历 → palette
// Phase 3 统一：从 Grid 参数读取 tree_base / index_origin / index_dims，
// 而非 BG1 globals（主世界 + 物体走同一路径，b_struct[tree_base + ...] 取树）。
fn sample_brickmap(g: Grid, fine: vec3<i32>) -> u32 {
  // ---- chunk 窗口定位 ----
  let m = ((fine % vec3<i32>(i32(CHUNK_SIZE))) + vec3<i32>(i32(CHUNK_SIZE))) % vec3<i32>(i32(CHUNK_SIZE));
  let chunk_i = (fine - m) / vec3<i32>(i32(CHUNK_SIZE));
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

  // ---- DFS 4 层 mask 遍历 ----
  let local = vec3<u32>(m);  // chunk 内 fine 坐标 0..255
  var node_addr = chunk_base;
  for (var level = 0u; level < MAX_LEVEL; level = level + 1u) {
    let mask_lo = b_struct[node_addr];
    let mask_hi = b_struct[node_addr + 1u];
    let palette = b_struct[node_addr + 2u];

    // 该层 4³ 子块索引：shift = 8 - (level+1)*2（level 0:>>6, 1:>>4, 2:>>2, 3:>>0）
    let shift = 8u - (level + 1u) * 2u;
    let cx = (local.x >> shift) & 3u;
    let cy = (local.y >> shift) & 3u;
    let cz = (local.z >> shift) & 3u;
    let child_idx = cz * 16u + cy * 4u + cx;  // z*16+y*4+x（与 child_linear_idx 一致）

    // mask 64-bit 拆两个 u32：child_idx < 32 → mask_lo, >= 32 → mask_hi
    var mask_word: u32 = mask_lo;
    var bit_in_word: u32 = child_idx;
    if (child_idx >= 32u) {
      mask_word = mask_hi;
      bit_in_word = child_idx - 32u;
    }
    let bit = 1u << bit_in_word;
    if ((mask_word & bit) == 0u) {
      return palette;  // uniform leaf（palette=0 = AIR）
    }

    // 分裂 → child offset = popcount(mask 中 child_idx 之前的 set bits)
    var popcount_below: u32;
    if (child_idx < 32u) {
      popcount_below = countOneBits(mask_lo & ((1u << bit_in_word) - 1u));
    } else {
      popcount_below = countOneBits(mask_lo) + countOneBits(mask_hi & ((1u << bit_in_word) - 1u));
    }
    let child_offset = b_struct[node_addr + NODE_FIXED_WORDS + popcount_below];
    node_addr = chunk_base + child_offset;  // chunk 内相对 → 绝对
  }
  // level 4（1³ leaf）：mask 全 0，直接读 palette
  return b_struct[node_addr + 2u];
}

// A&W：给定当前细格坐标分量与步进方向，返回下一格边界的坐标
// （WGSL 不支持函数内嵌套 fn 定义，必须放模块顶层——曾因嵌套导致整 shader 编译失败黑屏）
fn next_boundary(c: i32, s: i32) -> f32 {
  var v = f32(c);
  if (s >= 0) { v = v + 1.0; }
  return v;
}

// 粗级（cell = 16 fine）版 next_boundary：返回下一个粗边界的 fine 坐标
fn next_coarse_boundary(c: i32, s: i32) -> f32 {
  if (s >= 0) { return f32((c + 1) << 4); }
  return f32(c << 4);
}

// Brick Tree 占用检查：coarse cell（16³）内是否有非空体素
// 走 DFS 到 level 2（16³ brick），mask=0 → palette!=0 即占用；mask!=0 → 分裂=有内容
// Phase 3 统一：从 Grid 参数读取 tree_base / index_origin / index_dims。
fn cell_occupied(g: Grid, cc: vec3<i32>) -> bool {
  let fine = cc * 16;
  // ---- chunk 窗口定位 ----
  let m = ((fine % vec3<i32>(i32(CHUNK_SIZE))) + vec3<i32>(i32(CHUNK_SIZE))) % vec3<i32>(i32(CHUNK_SIZE));
  let chunk_i = (fine - m) / vec3<i32>(i32(CHUNK_SIZE));
  let origin = g.index_origin;
  let dims = g.index_dims;
  let rel = chunk_i - origin;
  if (any(rel < vec3<i32>(0))) { return false; }
  let rel_u = vec3<u32>(rel);
  if (any(rel_u >= dims)) { return false; }
  let index_addr = g.tree_base + rel_u.x + rel_u.y * CHUNK_INDEX_CAP + rel_u.z * (CHUNK_INDEX_CAP * CHUNK_INDEX_CAP);
  let entry = b_struct[index_addr];
  if (entry == 0u) { return false; }
  let chunk_base = g.tree_base + entry - 1u;

  // ---- DFS 到 level 2（16³ brick）----
  let local = vec3<u32>(m);
  var node_addr = chunk_base;
  for (var level = 0u; level < 2u; level = level + 1u) {
    let mask_lo = b_struct[node_addr];
    let mask_hi = b_struct[node_addr + 1u];
    let palette = b_struct[node_addr + 2u];
    let shift = 8u - (level + 1u) * 2u;
    let cx = (local.x >> shift) & 3u;
    let cy = (local.y >> shift) & 3u;
    let cz = (local.z >> shift) & 3u;
    let child_idx = cz * 16u + cy * 4u + cx;
    var mask_word: u32 = mask_lo;
    var bit_in_word: u32 = child_idx;
    if (child_idx >= 32u) {
      mask_word = mask_hi;
      bit_in_word = child_idx - 32u;
    }
    let bit = 1u << bit_in_word;
    if ((mask_word & bit) == 0u) {
      // uniform brick：palette!=0 即占用
      return palette != 0u;
    }
    // 分裂 → 下钻
    var popcount_below: u32;
    if (child_idx < 32u) {
      popcount_below = countOneBits(mask_lo & ((1u << bit_in_word) - 1u));
    } else {
      popcount_below = countOneBits(mask_lo) + countOneBits(mask_hi & ((1u << bit_in_word) - 1u));
    }
    let child_offset = b_struct[node_addr + NODE_FIXED_WORDS + popcount_below];
    node_addr = chunk_base + child_offset;
  }
  // level 2 brick 分裂了 → 有内容
  return true;
}

// ============================================================================
// P2.10 统一 DDA：trace_grid 一套管线同时服务主网格和逐物体 OBJ
// 以下函数严格对应 Rust `brickmap/obj.rs` CPU 参考实现（逐字镜像）。
// ============================================================================

// slab 法射线-AABB 求交，返回 (t_enter, t_exit)；平行且在外 → (1.0, 0.0) miss 哨兵
// 对应 obj.rs::slab_box
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
// 主世界和物体本质都是体素网格 DDA——用相同的 slab→coarse→fine 流程，
// 差异通过 Grid 参数化（tree_base + 变换矩阵 + chunk 窗口）。
// 零 kind 分支：所有 grid 从统一 b_struct[tree_base..] 读树、b_palette[palette_base..] 取色。
// ============================================================================

// 统一命中结构：hit/t/pal/n(世界空间法线)/face_id(0..5)/obj_id(-1=主世界,>=0=物体)
struct UnifiedHit {
  hit: bool,
  t: f32,
  pal: u32,
  n: vec3<f32>,        // 世界空间法线（光影用）
  face_id: u32,        // 命中面 0..5（与 face_index_from_normal 对齐）
  obj_id: i32,         // -1 = 主世界, >=0 = 物体索引
}

// 统一网格上下文——一套 DDA 跑所有网格（主世界 + 物体）
// 由 `make_grid(idx)` 从 `grid_descs[idx]` 构造；携带 tree_base/palette_base/
// index_origin/dims 作为数据源基址，trace_grid → fine_scan_cell → grid_sample_voxel
// 全链统一从 b_struct[tree_base + ...] / b_palette[palette_base + ...] 取数据。
struct Grid {
  w_mn: vec3<f32>,              // 世界 AABB min
  w_mx: vec3<f32>,              // 世界 AABB max
  l_min: vec3<f32>,             // 局部 AABB min（fine 坐标）= vec3(0.0)
  l_max: vec3<f32>,             // 局部 AABB max（fine 坐标）= vec3(256.0)
  cc_min: vec3<i32>,            // coarse cell 范围 min
  cc_max: vec3<i32>,            // coarse cell 范围 max
  max_coarse_steps: u32,
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

// cell 占用查询——统一入口，零 kind 分支
fn grid_cell_occupied(g: Grid, cc: vec3<i32>) -> bool {
  return cell_occupied(g, cc);
}

// 体素采样——统一入口，零 kind 分支
fn grid_sample_voxel(g: Grid, fc: vec3<i32>) -> u32 {
  return sample_brickmap(g, fc);
}

// 统一两级 DDA：slab→coarse→fine_scan_cell（完全通用，零 kind 分支）
// 所有网格（brickmap / obj / 未来任何体素 grid）走同一条路径
fn trace_grid(g: Grid, origin: vec3<f32>, dir: vec3<f32>, t_cap: f32) -> UnifiedHit {
  let miss = UnifiedHit(false, 0.0, 0u, vec3<f32>(0.0), 0u, g.obj_id);
  // ---- 世界 AABB 预剔除 ----
  let bx = slab_box(origin, dir, g.w_mn, g.w_mx, 0.0, t_cap);
  if (bx.y < max(bx.x, 0.0) || bx.x >= t_cap) { return miss; }
  let t_hi_cap = min(bx.y, t_cap);
  if (t_hi_cap <= max(bx.x, 0.0)) { return miss; }
  // ---- 局部变换 ----
  let wp = origin - g.pos;
  let ro = vec3<f32>(dot(wp, g.col0), dot(wp, g.col1), dot(wp, g.col2)) / g.scale;
  let rd = vec3<f32>(dot(dir, g.col0), dot(dir, g.col1), dot(dir, g.col2)) / g.scale;
  // ---- 局部 AABB slab ----
  let tl = slab_box(ro, rd, g.l_min, g.l_max, 0.0, t_hi_cap);
  if (tl.y < max(tl.x, 0.0)) { return miss; }
  let tl0 = max(tl.x, 0.0);
  let tl1 = min(tl.y, t_hi_cap);
  if (tl1 <= tl0) { return miss; }
  // ---- 两级 A&W ----
  var sign_v = vec3<i32>(1i);
  sign_v = select(sign_v, vec3<i32>(-1i), rd < vec3<f32>(0.0));
  var delta = vec3<f32>(1e+30);
  delta = select(delta, 1.0 / abs(rd), abs(rd) > vec3<f32>(1e-30));
  let delta_c = delta * 16.0;
  let start = ro + rd * tl0;
  var cc = vec3<i32>(vec3<i32>(floor(start)) >> vec3<u32>(4u));
  // cc clamp 统一对所有 grid 生效——越界时 cell_occupied / sample_voxel 自己过滤
  cc = clamp(cc, g.cc_min, g.cc_max);
  var tmax_c = vec3<f32>(1e+30);
  if (abs(rd.x) > 1e-30) { let t = (next_coarse_boundary(cc.x, sign_v.x) - start.x) / rd.x; tmax_c.x = max(t, 0.0); }
  if (abs(rd.y) > 1e-30) { let t = (next_coarse_boundary(cc.y, sign_v.y) - start.y) / rd.y; tmax_c.y = max(t, 0.0); }
  if (abs(rd.z) > 1e-30) { let t = (next_coarse_boundary(cc.z, sign_v.z) - start.z) / rd.z; tmax_c.z = max(t, 0.0); }
  let t_rel_max = tl1 - tl0;
  var t_in = 0.0;
  // entry_axis：跨入当前 cell 的入口面索引。首格 t_in=0 时用 normalize(-rd) 兜底
  // （相机贴面/UB；按用户约定 UB 直接返回该 voxel 颜色，face_id 仍按反方向反推）；
  // 后续格在 coarse 推进时由「刚跨过的轴 + sign_v」更新。
  var entry_axis: u32 = face_index_from_normal(normalize(-rd));
  for (var step_c: u32 = 0u; step_c < g.max_coarse_steps; step_c = step_c + 1u) {
    // 统一范围检查（所有 grid 都有 cc_min/cc_max）
    if (any(cc < g.cc_min) || any(cc > g.cc_max)) { break; }
    let t_out = min(tmax_c.x, min(tmax_c.y, tmax_c.z));
    if (grid_cell_occupied(g, cc)) {
      let hi = min(t_out, t_rel_max);
      let f = fine_scan_cell(g, start, rd, sign_v, delta, cc, t_in, hi, entry_axis);
      if (f.hit) {
        // 统一法线计算：fine_scan_cell 内部已推好 face_id，调用方零分支
        let n_local = face_normal_from_index(f.face_id);
        let n_world = normalize(n_local.x * g.col0 + n_local.y * g.col1 + n_local.z * g.col2);
        return UnifiedHit(true, tl0 + f.t, f.pal, n_world, f.face_id, g.obj_id);
      }
    }
    if (t_out >= t_rel_max) { break; }
    if (tmax_c.x <= tmax_c.y && tmax_c.x <= tmax_c.z) {
      t_in = tmax_c.x; tmax_c.x = tmax_c.x + delta_c.x; cc.x = cc.x + sign_v.x;
      entry_axis = select(0u, 1u, sign_v.x < 0);
    } else if (tmax_c.y <= tmax_c.z) {
      t_in = tmax_c.y; tmax_c.y = tmax_c.y + delta_c.y; cc.y = cc.y + sign_v.y;
      entry_axis = select(2u, 3u, sign_v.y < 0);
    } else {
      t_in = tmax_c.z; tmax_c.z = tmax_c.z + delta_c.z; cc.z = cc.z + sign_v.z;
      entry_axis = select(4u, 5u, sign_v.z < 0);
    }
  }
  return miss;
}

// 构造统一 Grid：从 grid_descs[idx] 解码所有字段（主世界 idx=0，物体 idx≥1）。
// 主世界 = identity 变换（pos=0/rot=I/scale=1），物体 = 任意变换。
// chunk 窗口 cc_min/cc_max 由 GridDesc.index_origin/dims 算出；物体 dims=(1,1,1)。
fn make_grid(idx: u32) -> Grid {
  let d = grid_descs[idx];
  let origin = vec3<i32>(d.index_origin_x, d.index_origin_y, d.index_origin_z);
  let dims = vec3<i32>(i32(d.index_dims_x), i32(d.index_dims_y), i32(d.index_dims_z));
  let cc_min = origin * 16;
  let cc_max = (origin + dims) * 16 - vec3<i32>(1i, 1i, 1i);
  let coarse_budget = (dims.x + dims.y + dims.z) * 16;
  let coarse_limit = u32(coarse_budget) * 3u;
  // 主世界（idx=0，identity：局部=世界）：局部 AABB = 窗口 AABB（origin 可为负，
  // [0,256]³ 默认盒会把窗口绝大部分 slab 剔除）；物体 = 单 chunk 局部 [0,256]³。
  let is_world = idx == 0u;
  let l_mn = select(vec3<f32>(0.0), d.aabb_min.xyz, is_world);
  let l_mx = select(vec3<f32>(f32(CHUNK_SIZE)), d.aabb_max.xyz, is_world);
  return Grid(
    d.aabb_min.xyz, d.aabb_max.xyz,       // 世界 AABB
    l_mn, l_mx,                            // 局部 AABB
    cc_min, cc_max,
    coarse_limit,
    d.rot0.xyz, d.rot1.xyz, d.rot2.xyz,    // 旋转矩阵列
    d.pos_scale.xyz, d.pos_scale.w,        // pos + scale
    d.tree_base, d.palette_base,           // 数据源基址
    origin, vec3<u32>(d.index_dims_x, d.index_dims_y, d.index_dims_z),
    select(-1i, i32(idx), idx > 0u),       // idx=0 → 主世界(-1)，idx≥1 → 物体
  );
}


// 统一细扫：任意 grid（brickmap / obj）通用。算法 1:1 原始 Akenine-Möller。
// 数据源唯一入口：grid_sample_voxel(g, fc)。
// 返回 t 为相对 t_lo 的偏移；face_id 0..5 = ±xyz 六面（命中面法线索引）
//   pre-check 命中（p 落在固体 voxel）：face_id = entry_axis（由调用方维护：
//   首格用 normalize(-rd) 兜底，非首格用 coarse 上一步跨轴；相机在体内时 UB，
//   直接返回该 voxel 颜色，face_id 用 normalize(-rd) 反推）
struct FineHit {
  hit: bool,
  t: f32,
  pal: u32,
  face_id: u32,
}
fn fine_scan_cell(g: Grid, ro: vec3<f32>, rd: vec3<f32>, sign: vec3<i32>, delta: vec3<f32>,
                  cc: vec3<i32>, t_lo: f32, t_hi: f32, entry_axis: u32) -> FineHit {
  if (t_hi <= t_lo) { return FineHit(false, 0.0, 0u, 0u); }
  let p = ro + rd * t_lo;
  let base = vec3<i32>(cc << vec3<u32>(4u));
  let fc0 = clamp(vec3<i32>(floor(p)), base, base + vec3<i32>(15));
  let pal0_init = grid_sample_voxel(g, fc0);
  if (pal0_init != 0u) { return FineHit(true, t_lo, pal0_init, entry_axis); }
  var fc = fc0;
  var tmax_f = vec3<f32>(1e+30);
  if (abs(rd.x) > 1e-30) {
    let t = (next_boundary(fc0.x, sign.x) - p.x) / rd.x;
    tmax_f.x = max(t, 0.0);
  }
  if (abs(rd.y) > 1e-30) {
    let t = (next_boundary(fc0.y, sign.y) - p.y) / rd.y;
    tmax_f.y = max(t, 0.0);
  }
  if (abs(rd.z) > 1e-30) {
    let t = (next_boundary(fc0.z, sign.z) - p.z) / rd.z;
    tmax_f.z = max(t, 0.0);
  }
  let span = t_hi - t_lo;
  var t_f = 0.0;
  for (var i_f: u32 = 0u; i_f < 48u; i_f = i_f + 1u) {
    if (t_f >= span) { break; }
    var face_id: u32 = 5u;
    if (tmax_f.x <= tmax_f.y && tmax_f.x <= tmax_f.z) {
      t_f = tmax_f.x;
      tmax_f.x = tmax_f.x + delta.x;
      fc.x = fc.x + sign.x;
      face_id = select(0u, 1u, sign.x < 0);
    } else if (tmax_f.y <= tmax_f.z) {
      t_f = tmax_f.y;
      tmax_f.y = tmax_f.y + delta.y;
      fc.y = fc.y + sign.y;
      face_id = select(2u, 3u, sign.y < 0);
    } else {
      t_f = tmax_f.z;
      tmax_f.z = tmax_f.z + delta.z;
      fc.z = fc.z + sign.z;
      face_id = select(4u, 5u, sign.z < 0);
    }
    let pal = grid_sample_voxel(g, fc);
    if (pal != 0u) { return FineHit(true, t_lo + t_f, pal, face_id); }
  }
  return FineHit(false, 0.0, 0u, 0u);
}

// 遮挡快路径（阴影射线）：[0, t_max) 内任一 volume 命中即 true。
// Phase 3 统一：遍历 grid_descs[0..arrayLength]，零 world/obj 分支。
fn scene_occluded(origin: vec3<f32>, dir: vec3<f32>, t_max: f32) -> bool {
  let n = arrayLength(&grid_descs);
  for (var i: u32 = 0u; i < n; i = i + 1u) {
    let mh = trace_grid(make_grid(i), origin, dir, t_max);
    if (mh.hit) { return true; }
  }
  return false;
}

// ============================================================================
// Douglas devlog #02 基础光影：方向光硬阴影 + sky 渐变环境光 + 发光体素 radiance 直出
// - 方向光：命中点向太阳投 1 条射线，不通即阴影（硬阴影，无锥采样）
// - sky 渐变环境光：按法线 y 混合天顶/地平线
// - 发光体素直出：albedo × emissive × GAIN，无方向性、不受阴影
// - 无点光源、无 Phong 高光
// ============================================================================

const SHADOW_BIAS: f32 = 0.5;
const SHADOW_DIR_T_MAX: f32 = 65536.0;
const EMISSIVE_EMIT_GAIN: f32 = 4.0;

// 命中点材质：palette 两 words 解包 albedo + roughness + emissive
// roughness 保留（palette 数据完整性），但 Douglas 方案高光已删除
struct HitMat {
  albedo: vec3<f32>,
  rough: f32,
  emissive: f32,
}
// Phase 3 统一：palette_base 参数替代 obj 分支——主世界 + 物体都从
// b_palette[palette_base + pal*2..] 取色，零 kind 分支。
fn hit_mat(palette_base: u32, pal: u32) -> HitMat {
  let w0 = b_palette[palette_base + pal * 2u];
  let w1 = b_palette[palette_base + pal * 2u + 1u];
  return HitMat(
    vec3<f32>(
      f32(w0 & 0xFFu),
      f32((w0 >> 8u) & 0xFFu),
      f32((w0 >> 16u) & 0xFFu),
    ) / 255.0,
    f32((w0 >> 24u) & 0xFFu) / 255.0,
    f32(w1 & 0xFFu) / 255.0,
  );
}

// ============================================================================
// P3.5 色调映射 + 颜色空间转换
// ============================================================================

// ACES Filmic 色调映射（简化版，Narkowicz 近似）
// 把 HDR 线性值压缩进 [0,1]，高光不截断、暗部不发黑
fn aces_tonemap(x: vec3<f32>) -> vec3<f32> {
  let a = 2.51;
  let b = 0.03;
  let c = 2.43;
  let d = 0.59;
  let e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
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
// sky(dir) → 线性色（渐变 + 太阳辉光晕）
// Douglas 方案：无软阴影太阳盘（方向光硬阴影 shape.x = 0），只保留渐变 + 轻微 glow
// ============================================================================
fn sky(dir: vec3<f32>) -> vec3<f32> {
  let d = normalize(dir);
  let h = clamp(d.y, 0.0, 1.0);
  let t = smoothstep(0.0, 0.35, h);
  var col: vec3<f32> = mix(light_u.sky_horizon.xyz, light_u.sky_top.xyz, t);
  if (light_u.g.count > 0u && light_u.lights[0].kind_pos_dir.x < 0.5) {
    let sdir = light_u.lights[0].kind_pos_dir.yzw;
    let cos_a = max(dot(d, sdir), 0.0);
    let sun_c = light_u.lights[0].color_intensity.xyz * light_u.lights[0].color_intensity.w;
    let glow = pow(max(cos_a, 0.0), 64.0) * 0.05 * select(1.0, 0.0, h > 0.0);
    col = col + sun_c * glow;
  }
  return col;
}

// ============================================================================
// Douglas devlog #22 implicit normals：GPU 运行时按邻域体素 occupancy 有限差分
// - 命中体素中心 + 6 方向采样 → 密度差 → 连续 per-voxel 法向
// - 不存储、不烘焙、随 DDA trace 实时算（比 Douglas 的 upload-time bake 更动态）
// - 主世界用 sample_brickmap（2~3 级寻址）；物体暂用 DDA 面法向（obj_id >= 0 跳过）
// Phase 3 统一：sample_brickmap 走 Grid 参数（g.tree_base + g.index_origin/dims）。
// ============================================================================
fn compute_implicit_normal(g: Grid, origin: vec3<f32>, dir: vec3<f32>, t: f32, dda_n: vec3<f32>) -> vec3<f32> {
  if (g.obj_id >= 0) { return dda_n; }  // 物体用 DDA 面法向
  let hit_pos = origin + dir * t;
  // dda_n 指向外部（射线来向）；沿 -dda_n 微偏 → 命中体素中心
  let hit_fc = vec3<i32>(floor(hit_pos - dda_n * 0.001));
  // 6 邻域 occupancy（1 = 实心，0 = 空气）
  let sx_n = f32(sample_brickmap(g, hit_fc + vec3<i32>(-1, 0, 0)) != 0u);
  let sx_p = f32(sample_brickmap(g, hit_fc + vec3<i32>( 1, 0, 0)) != 0u);
  let sy_n = f32(sample_brickmap(g, hit_fc + vec3<i32>( 0,-1, 0)) != 0u);
  let sy_p = f32(sample_brickmap(g, hit_fc + vec3<i32>( 0, 1, 0)) != 0u);
  let sz_n = f32(sample_brickmap(g, hit_fc + vec3<i32>( 0, 0,-1)) != 0u);
  let sz_p = f32(sample_brickmap(g, hit_fc + vec3<i32>( 0, 0, 1)) != 0u);
  let raw = vec3<f32>(sx_n - sx_p, sy_n - sy_p, sz_n - sz_p);
  let len2 = dot(raw, raw);
  return normalize(mix(dda_n, raw, 0.5));
}

// Douglas devlog #02 基础光影合成：
// 1. sky 渐变环境光（按法线 y）
// 2. 方向光硬阴影（命中点向太阳投 1 条射线，不通即阴影）
// 3. 发光体素 radiance 直出（无方向性、不受阴影）
// 无点光源、无 Phong 高光
// Per-pixel shading 入口（douglas #02/#22 基础光影）。
// Phase 3 统一：Grid 参数携带 palette_base（hit_mat 取色）+ obj_id（implicit normal 分流）。
fn shade_hit(g: Grid, origin: vec3<f32>, dir: vec3<f32>, t: f32, pal: u32, dda_n: vec3<f32>,
             shadow_t_max: f32) -> vec3<f32> {
  let n = compute_implicit_normal(g, origin, dir, t, dda_n);
  let p = origin + dir * t;
  let mat = hit_mat(g.palette_base, pal);
  let base = mat.albedo;

  // sky 渐变环境光
  let h = clamp(n.y, 0.0, 1.0);
  let sky_grad = mix(light_u.sky_horizon.xyz, light_u.sky_top.xyz, smoothstep(0.0, 0.35, h));
  var col: vec3<f32> = base * (light_u.g.ambient.xyz * 0.4 + sky_grad * 0.6);

  // 方向光硬阴影（Devlog #02：1 条射线，不通即阴影）
  if (light_u.g.count > 0u) {
    let ld = light_u.lights[0];
    if (ld.kind_pos_dir.x < 0.5) {
      let l_axis = ld.kind_pos_dir.yzw;
      let ndl = max(dot(n, l_axis), 0.0);
      if (ndl > 0.0) {
        let o = p + n * SHADOW_BIAS;
        let vis = select(0.0, 1.0, !scene_occluded(o, l_axis, SHADOW_DIR_T_MAX));
        let c = ld.color_intensity.xyz * ld.color_intensity.w;
        col = col + base * c * (ndl * vis);
      }
    }
  }

  // 发光体素 radiance 直出（Devlog #19 radiance 分支 3 的直接光版本）
  col = col + base * (mat.emissive * EMISSIVE_EMIT_GAIN);
  return col * light_u.g.exposure_pad.x;
}

// ============================================================================
// DDA 主入口：每个像素 = workgroup 内一个 invocation（8x8x1）
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
  let dir_fine = normalize(diff_world);
  let origin_fine = view_u.cam_pos_fine.xyz;

  // ---- trace_scene：遍历 grid_descs[0..N] 取最近命中（Phase 3 统一）----
  var best_t = 1e+30;
  var best_pal: u32 = 0u;
  var best_n = vec3<f32>(0.0);
  var best_face_id: u32 = 0u;
  var best_grid: Grid = make_grid(0u);  // 命中 volume 的 Grid（shade_hit 取色 + implicit normal 用）
  let n = arrayLength(&grid_descs);
  for (var i: u32 = 0u; i < n; i = i + 1u) {
    let cap = min(best_t, frustum_length);
    let mh = trace_grid(make_grid(i), origin_fine, dir_fine, cap);
    if (mh.hit && mh.t < best_t) {
      best_t = mh.t;
      best_pal = mh.pal;
      best_n = mh.n;
      best_face_id = mh.face_id;
      best_grid = make_grid(i);
    }
  }

  // ---- v5 single-pass 着色（per-pixel，无 hashmap）----
  if (best_t < 1e+29) {
    // debug_mode.x: implicit normal 可视化
    if (view_u.debug_mode.x > 0.5) {
      let n_implicit = compute_implicit_normal(best_grid, origin_fine, dir_fine, best_t, best_n);
      textureStore(out_tex, coord0, vec4<f32>(n_implicit * 0.5 + 0.5, 1.0));
      return;
    }
    // debug_mode.y: face 6 色 + sky=品红（G-buffer 状态图定位工具）
    if (view_u.debug_mode.y > 0.5) {
      textureStore(out_tex, coord0, vec4<f32>(face_color_from_index(best_face_id), 1.0));
      return;
    }
    // 正常着色：shade_hit（per-pixel 硬阴影 + emissive）→ ACES → sRGB
    var col = shade_hit(best_grid, origin_fine, dir_fine, best_t, best_pal, best_n, SHADOW_DIR_T_MAX);
    col = aces_tonemap(col);
    col = linear_to_srgb(col);
    textureStore(out_tex, coord0, vec4<f32>(col, 1.0));
  } else {
    // sky
    var col = sky(dir_fine);
    col = aces_tonemap(col);
    col = linear_to_srgb(col);
    textureStore(out_tex, coord0, vec4<f32>(col, 1.0));
  }
}

// ============================================================================
