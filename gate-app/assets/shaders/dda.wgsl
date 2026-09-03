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


// 统一网格上下文——一套 DDA 跑所有网格（主世界 + 物体）
// 由 `make_grid(idx)` 从 `grid_descs[idx]` 构造；携带 tree_base/palette_base/
// index_origin/dims 作为数据源基址，trace_grid → trace_chunk 统一从
// b_struct[tree_base + ...] / b_palette[palette_base + ...] 取数据。
struct Grid {
  w_mn: vec3<f32>,              // 世界 AABB min
  w_mx: vec3<f32>,              // 世界 AABB max
  l_min: vec3<f32>,             // 局部 AABB min（fine 坐标）= vec3(0.0)
  l_max: vec3<f32>,             // 局部 AABB max（fine 坐标）= vec3(256.0)
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
// fine → chunk 窗口定位 → DFS 树 4 层 mask 遍历 → palette
// Phase 3 统一：从 Grid 参数读取 tree_base / index_origin / index_dims，
// 而非 BG1 globals（主世界 + 物体走同一路径，b_struct[tree_base + ...] 取树）。
// 仅 implicit normals 6 邻域 occupancy 点查询用；主遍历走层次栈式 trace_chunk。
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

// ============================================================================
// 层次栈式 mask DDA（Douglas devlog #17 / sparse 64-tree 遍历）
//
// 旧两级 DDA 把树当点查询结构：每个 16³ 粗步从根重新 DFS（~9 load）、每个细步
// 再走 4 层（~13 load），树的层次连贯性全部丢弃。层次遍历把节点 mask 一次 load
// 进寄存器，在该节点 4³=64 个子块间做 A&W 步进——每步只查 mask bit（零 load）；
// bit=1（分裂）才压栈下钻，bit=0 uniform 子块整格跳过/整格命中。
// 空气穿越代价：空 chunk 1 load（窗口 entry=0）、有树空 chunk ~4 load（根节点），
// 与旧粗步 9 load/16³ 相比量级下降。
//
// 栈深 4（256→64→16→4→1）；chunk 间 256³ A&W 由 trace_grid 维护。
// CPU 逐字镜像：gate-render/src/brickmap/dda.rs `cpu_reference_dda_ray_tree`。
// ============================================================================

// 栈帧：一层分裂节点的 DDA 状态
struct TreeFrame {
  node_addr: u32,      // 节点绝对字址（b_struct）
  node_min: vec3<f32>, // 节点区域原点（局部 fine 坐标）
  cell: vec3<i32>,     // 当前子块坐标 0..3
  tmax: vec3<f32>,     // 到下一子块边界的 t（绝对，ro 系）
  t_enter: f32,        // 进入当前子块的 t
  t_exit: f32,         // 节点出口 t
  face: u32,           // 进入当前子块的面 0..5（±xyz）
}

// init_tree_frame 的函数体已直接展开进 trace_chunk 的两处调用点（根帧 + 下钻）——
// naga/驱动不内联 WGSL 函数，每次调用实测 ~µs 级开销（slab_box 内联 17ms→2.75ms 实证）

// 层次遍历命中记录：t 为 ro 系绝对 t；face_id 0..5 = ±xyz 六面（命中面法线索引）。
//   pre-check 命中（射线起点在固体 leaf 内，相机在体内 UB）：face_id 由调用方
//   用 normalize(-rd) 反推（首 chunk entry_face）。
struct FineHit {
  hit: bool,
  t: f32,
  pal: u32,
  face_id: u32,
}

// 单 chunk 内层次遍历。chunk_base = 根节点绝对字址；chunk_min = chunk 原点（局部 fine）。
// 射线段 [t0, t1]（ro 系绝对 t）；entry_face = 进入本 chunk 的面（首 chunk 由调用方
// 用 normalize(-rd) 兜底，后续 chunk 为跨 chunk 面）。
// 返回 FineHit（t 为 ro 系绝对 t）；走出 chunk 未命中 → hit=false。
fn trace_chunk(chunk_base: u32, chunk_min: vec3<f32>,
               ro: vec3<f32>, rd: vec3<f32>, sign_v: vec3<i32>, delta: vec3<f32>,
               t0: f32, t1: f32, entry_face: u32) -> FineHit {
  // 擦边退化（t0>=t1：射线只蹭到 chunk 边界）→ 无体素内部可穿过，直接 miss
  if (t0 >= t1) { return FineHit(false, 0.0, 0u, 0u); }
  // 注：曾试过 4 变量静态分支版（避免动态索引局部内存）——实测反而 2× 慢（59.8ms
  // vs 27.6ms），寄存器压力来自别处；数组栈版为实测最优，勿改回静态分支
  var stack: array<TreeFrame, 4u>;
  var depth: i32 = 0;
  // ---- 根帧初始化（init_tree_frame 展开内联，level 0 → sub 64）----
  {
    let p = ro + rd * t0;
    var cell = vec3<i32>(floor((p - chunk_min) / 64.0));
    cell = clamp(cell, vec3<i32>(0), vec3<i32>(3));
    let side = select(vec3<f32>(0.0), vec3<f32>(1.0), sign_v >= vec3<i32>(0));
    let boundary = chunk_min + (vec3<f32>(cell) + side) * 64.0;
    var tmax = vec3<f32>(1e+30);
    let axis_on = abs(rd) > vec3<f32>(1e-30);
    tmax = select(tmax, (boundary - ro) / rd, axis_on);
    tmax = max(tmax, vec3<f32>(t0));
    stack[0] = TreeFrame(chunk_base, chunk_min, cell, tmax, t0, t1, entry_face);
  }
  var f = stack[0];
  // 防挂死安全网：几何上界 = 对角射线穿越全分裂 chunk 的节点读数
  // （depth3 ≤768 体素 + depth2 ≤192 + depth1 ≤12 + 各级帧内重读 ≈ 25k 量级）；
  // 真实场景（uniform 盒体/地形/建筑，空气在高层整格跳过）每 chunk 仅几十~几百次。
  var budget: u32 = 65536u;
  loop {
    if (budget == 0u) { break; }
    budget = budget - 1u;
    let level = u32(depth);
    // 节点 fixed 字：mask_lo + mask_hi + palette
    let mask_lo = b_struct[f.node_addr];
    let mask_hi = b_struct[f.node_addr + 1u];
    let palette = b_struct[f.node_addr + 2u];
    // 当前子块的出口 t（三轴 tmax 最小值）；零厚度（== t_enter）= 擦边不入内部
    let cell_exit = min(min(f.tmax.x, f.tmax.y), f.tmax.z);
    if (mask_lo == 0u && mask_hi == 0u) {
      // 整节点 uniform（mask==0）：palette 直决，无需逐子块步进
      if (palette != 0u) { return FineHit(true, f.t_enter, palette, f.face); }
      // 整节点空气 → 跳过整个节点区域（弹栈，父帧推进）
      depth = depth - 1;
      if (depth < 0) { return FineHit(false, 0.0, 0u, 0u); }
      f = stack[depth];
    } else {
      let child_idx = u32(f.cell.z * 16 + f.cell.y * 4 + f.cell.x);
      var mask_word: u32 = mask_lo;
      var bit_in_word: u32 = child_idx;
      if (child_idx >= 32u) {
        mask_word = mask_hi;
        bit_in_word = child_idx - 32u;
      }
      let bit = 1u << bit_in_word;
      if ((mask_word & bit) == 0u) {
        // uniform 子块：颜色 = 父节点 palette（零额外 load）。
        // 零厚度擦边不入内部（与逐体素点查语义一致）
        if (palette != 0u && f.t_enter < cell_exit) {
          return FineHit(true, f.t_enter, palette, f.face);
        }
        // 空气子块 / 擦边 → 本帧推进一步（整子块跨越）
      } else {
        // 分裂 → popcount 定位紧凑 child offset
        var pop_below: u32;
        if (child_idx < 32u) {
          pop_below = countOneBits(mask_lo & ((1u << bit_in_word) - 1u));
        } else {
          pop_below = countOneBits(mask_lo) + countOneBits(mask_hi & ((1u << bit_in_word) - 1u));
        }
        let child_addr = chunk_base + b_struct[f.node_addr + NODE_FIXED_WORDS + pop_below];
        if (level < 3u) {
          // 下钻：子节点区域 = 当前子块；退化子块（出口==入口，零厚度擦边）不下钻
          if (cell_exit > f.t_enter) {
            let sub = 64u >> (level * 2u);
            let child_min = f.node_min + vec3<f32>(f.cell) * f32(sub);
            stack[depth] = f; // 寄存器态先落栈（弹出后要恢复的是已推进状态）
            depth = depth + 1;
            // ---- 子帧初始化（init_tree_frame 展开内联，sub = 64 >> (depth*2)）----
            let sub_c = 64u >> (u32(depth) * 2u);
            let cp = ro + rd * f.t_enter;
            var ccell = vec3<i32>(floor((cp - child_min) / f32(sub_c)));
            ccell = clamp(ccell, vec3<i32>(0), vec3<i32>(3));
            let cside = select(vec3<f32>(0.0), vec3<f32>(1.0), sign_v >= vec3<i32>(0));
            let cboundary = child_min + (vec3<f32>(ccell) + cside) * f32(sub_c);
            var ctmax = vec3<f32>(1e+30);
            let caxis_on = abs(rd) > vec3<f32>(1e-30);
            ctmax = select(ctmax, (cboundary - ro) / rd, caxis_on);
            ctmax = max(ctmax, vec3<f32>(f.t_enter));
            stack[depth] = TreeFrame(child_addr, child_min, ccell, ctmax,
                                     f.t_enter, min(cell_exit, f.t_exit), f.face);
            f = stack[depth];
            continue;
          }
        } else {
          // level 4 leaf（1³）：mask 必为 0，palette 即体素色；擦边不命中
          let leaf_pal = b_struct[child_addr + 2u];
          if (leaf_pal != 0u && f.t_enter < cell_exit) {
            return FineHit(true, f.t_enter, leaf_pal, f.face);
          }
          // 空气 leaf / 擦边 → 本帧推进一步
        }
      }
    }
    // ---- 推进：f 常驻寄存器前进一步；耗尽/跨界则弹栈连推，直到真正一步或走出 chunk ----
    loop {
      if (min(min(f.tmax.x, f.tmax.y), f.tmax.z) >= f.t_exit) {
        depth = depth - 1;
        if (depth < 0) { return FineHit(false, 0.0, 0u, 0u); }
        f = stack[depth];
        continue;
      }
      let sub = 64u >> (u32(depth) * 2u); // 该层子块边长：64/16/4/1
      if (f.tmax.x <= f.tmax.y && f.tmax.x <= f.tmax.z) {
        f.t_enter = f.tmax.x;
        f.tmax.x = f.tmax.x + delta.x * f32(sub);
        f.cell.x = f.cell.x + sign_v.x;
        f.face = select(0u, 1u, sign_v.x < 0);
      } else if (f.tmax.y <= f.tmax.z) {
        f.t_enter = f.tmax.y;
        f.tmax.y = f.tmax.y + delta.y * f32(sub);
        f.cell.y = f.cell.y + sign_v.y;
        f.face = select(2u, 3u, sign_v.y < 0);
      } else {
        f.t_enter = f.tmax.z;
        f.tmax.z = f.tmax.z + delta.z * f32(sub);
        f.cell.z = f.cell.z + sign_v.z;
        f.face = select(4u, 5u, sign_v.z < 0);
      }
      // 浮点边界（累计 tmax 与父帧 t_exit 差 1 ulp）：本步实际跨出了节点区域。
      // 禁止越界索引（cell=-1 会回绕成 bit 23 等错误子块）→ 视为节点耗尽，弹栈
      if (f.cell.x < 0 || f.cell.x > 3 ||
          f.cell.y < 0 || f.cell.y > 3 ||
          f.cell.z < 0 || f.cell.z > 3) {
        depth = depth - 1;
        if (depth < 0) { return FineHit(false, 0.0, 0u, 0u); }
        f = stack[depth];
        continue;
      }
      break;
    }
  }
  return FineHit(false, 0.0, 0u, 0u);
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
struct UnifiedHit {
  hit: bool,
  t: f32,
  pal: u32,
  n: vec3<f32>,        // 世界空间法线（光影用）
  face_id: u32,        // 命中面 0..5（与 face_index_from_normal 对齐）
  obj_id: i32,         // -1 = 主世界, >=0 = 物体索引
}

// 统一层次 DDA：slab→chunk 间 A&W→trace_chunk（完全通用，零 kind 分支）
// 所有网格（brickmap / obj / 未来任何体素 grid）走同一条路径
// 【性能】参数直接传 volume 索引（不传 15 字段 Grid 结构体）——WGSL 函数大结构体
// 按值传参在 naga/驱动下实测有 ~µs 级开销（slab_box 内联 17ms→2.75ms 实证）；
// slab 也直接内联（原 slab_box 调用版实测 17ms 纯调用开销）
fn trace_grid_idx(idx: u32, origin: vec3<f32>, dir: vec3<f32>, t_cap: f32) -> UnifiedHit {
  let d = grid_descs[idx];
  let w_mn = d.aabb_min.xyz;
  let w_mx = d.aabb_max.xyz;
  let is_world = idx == 0u;
  let l_min = select(vec3<f32>(0.0), d.aabb_min.xyz, is_world);
  let l_max = select(vec3<f32>(f32(CHUNK_SIZE)), d.aabb_max.xyz, is_world);
  let obj_id = select(-1i, i32(idx), idx > 0u);
  let index_origin = vec3<i32>(d.index_origin_x, d.index_origin_y, d.index_origin_z);
  let index_dims = vec3<u32>(d.index_dims_x, d.index_dims_y, d.index_dims_z);
  let tree_base = d.tree_base;
  let dims_i = vec3<i32>(i32(d.index_dims_x), i32(d.index_dims_y), i32(d.index_dims_z));
  let max_chunk_steps = u32(dims_i.x + dims_i.y + dims_i.z) * 3u + 16u;
  let col0 = d.rot0.xyz;
  let col1 = d.rot1.xyz;
  let col2 = d.rot2.xyz;
  let miss = UnifiedHit(false, 0.0, 0u, vec3<f32>(0.0), 0u, obj_id);
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
  if (bx_exit < max(bx_enter, 0.0) || bx_enter >= t_cap) { return miss; }
  let t_hi_cap = min(bx_exit, t_cap);
  if (t_hi_cap <= max(bx_enter, 0.0)) { return miss; }
  // ---- 局部变换 ----
  let wp = origin - d.pos_scale.xyz;
  let scale = d.pos_scale.w;
  let ro = vec3<f32>(dot(wp, d.rot0.xyz), dot(wp, d.rot1.xyz), dot(wp, d.rot2.xyz)) / scale;
  let rd = vec3<f32>(dot(dir, d.rot0.xyz), dot(dir, d.rot1.xyz), dot(dir, d.rot2.xyz)) / scale;
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
  let tl0 = max(tl_enter, 0.0);
  let tl1 = min(tl_exit, t_hi_cap);
  if (tl1 <= tl0) { return miss; }
  // debug_mode.z == 2（诊断）：跳过 chunk 步进层（二分 make_grid+slab vs 遍历成本）
  if (view_u.debug_mode.z > 1.5) { return miss; }
  // ---- chunk 间 A&W（256³ 一格）+ chunk 内层次 mask DDA（trace_chunk）----
  var sign_v = vec3<i32>(1i);
  sign_v = select(sign_v, vec3<i32>(-1i), rd < vec3<f32>(0.0));
  var delta = vec3<f32>(1e+30);
  delta = select(delta, 1.0 / abs(rd), abs(rd) > vec3<f32>(1e-30));
  let delta_c = delta * f32(CHUNK_SIZE);
  let start = ro + rd * tl0;
  var ci = vec3<i32>(floor(start / f32(CHUNK_SIZE)));
  // tmax_c：到下一 chunk 边界的 t（ro 系绝对）
  let side_c = select(vec3<f32>(0.0), vec3<f32>(1.0), sign_v >= vec3<i32>(0));
  var tmax_c = vec3<f32>(1e+30);
  let bnd_c = (vec3<f32>(ci) + side_c) * f32(CHUNK_SIZE);
  tmax_c = select(tmax_c, (bnd_c - ro) / rd, abs(rd) > vec3<f32>(1e-30));
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
        let h = trace_chunk(chunk_base, chunk_min, ro, rd, sign_v, delta,
                            t_enter_c, t1, entry_face);
        if (h.hit) {
          // 统一法线计算：trace_chunk 内部已推好 face_id，调用方零分支
          let n_local = face_normal_from_index(h.face_id);
          let n_world = normalize(n_local.x * col0 + n_local.y * col1 + n_local.z * col2);
          return UnifiedHit(true, h.t, h.pal, n_world, h.face_id, obj_id);
        }
      }
    }
    if (t_exit_c >= tl1) { break; }
    if (tmax_c.x <= tmax_c.y && tmax_c.x <= tmax_c.z) {
      t_enter_c = tmax_c.x; tmax_c.x = tmax_c.x + delta_c.x; ci.x = ci.x + sign_v.x;
      entry_face = select(0u, 1u, sign_v.x < 0);
    } else if (tmax_c.y <= tmax_c.z) {
      t_enter_c = tmax_c.y; tmax_c.y = tmax_c.y + delta_c.y; ci.y = ci.y + sign_v.y;
      entry_face = select(2u, 3u, sign_v.y < 0);
    } else {
      t_enter_c = tmax_c.z; tmax_c.z = tmax_c.z + delta_c.z; ci.z = ci.z + sign_v.z;
      entry_face = select(4u, 5u, sign_v.z < 0);
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
    select(-1i, i32(idx), idx > 0u),       // idx=0 → 主世界(-1)，idx≥1 → 物体
  );
}


// 遮挡快路径（阴影射线）：[0, t_max) 内任一 volume 命中即 true。
// Phase 3 统一：遍历 grid_descs[0..arrayLength]，零 world/obj 分支。
fn scene_occluded(origin: vec3<f32>, dir: vec3<f32>, t_max: f32) -> bool {
  let n = arrayLength(&grid_descs);
  for (var i: u32 = 0u; i < n; i = i + 1u) {
    let mh = trace_grid_idx(i, origin, dir, t_max);
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
  // debug_mode.w（诊断开关）：跳过 6 邻域点采样，直接用 DDA 面法线
  if (g.obj_id >= 0 || view_u.debug_mode.w > 0.5) { return dda_n; }
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

  // 方向光硬阴影（Devlog #02：1 条射线，不通即阴影）。
  // debug_mode.z（诊断开关）：跳过阴影射线（性能定位）
  if (light_u.g.count > 0u && view_u.debug_mode.z < 0.5) {
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

  // debug_mode.w > 1.5（诊断模式 2）：跳过全部 trace，直接天空色输出
  // （把 17ms 固定开销二分：反投影+输出 vs make_grid/trace_grid 框架）
  if (view_u.debug_mode.w > 1.5) {
    let px0 = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5)) / vec2<f32>(f32(size.x), f32(size.y));
    let uv0 = vec2<f32>(px0.x * 2.0 - 1.0, 1.0 - px0.y * 2.0);
    let nh = view_u.inv_view_proj * vec4<f32>(uv0.x, uv0.y, 1.0, 1.0);
    let fw = view_u.cam_pos_fine.xyz + (nh.xyz / nh.w - view_u.cam_pos_fine.xyz);
    var col0 = sky(normalize(fw - view_u.cam_pos_fine.xyz));
    col0 = linear_to_srgb(aces_tonemap(col0));
    textureStore(out_tex, coord0, vec4<f32>(col0, 1.0));
    return;
  }

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
  let n = arrayLength(&grid_descs);

  // debug_mode.w > 3.5（诊断模式 3）：只做 make_grid（读 GridDesc）不 trace
  // ——测量 make_grid 本身的成本（skip_chunkwalk 17ms 的最后嫌疑点）
  if (view_u.debug_mode.w > 3.5) {
    var sink = 0u;
    for (var i: u32 = 0u; i < n; i = i + 1u) {
      let gg = make_grid(i);
      sink = sink + u32(gg.scale) + u32(gg.w_mn.x) + u32(gg.col2.z);
    }
    var col1 = sky(dir_fine);
    col1 = col1 + vec3<f32>(f32(sink & 1u) * 0.001);
    col1 = linear_to_srgb(aces_tonemap(col1));
    textureStore(out_tex, coord0, vec4<f32>(col1, 1.0));
    return;
  }

  // ---- trace_scene：遍历 grid_descs[0..N] 取最近命中（Phase 3 统一）----
  var best_t = 1e+30;
  var best_pal: u32 = 0u;
  var best_n = vec3<f32>(0.0);
  var best_face_id: u32 = 0u;
  var best_grid: Grid = make_grid(0u);  // 命中 volume 的 Grid（shade_hit 取色 + implicit normal 用）
  for (var i: u32 = 0u; i < n; i = i + 1u) {
    let cap = min(best_t, frustum_length);
    let mh = trace_grid_idx(i, origin_fine, dir_fine, cap);
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
