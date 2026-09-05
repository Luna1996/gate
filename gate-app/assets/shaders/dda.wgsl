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
// BG3（R3-18 直光层：光照光池，与 Rust LightPoolUniform 464B 1:1）：
//   @group(3) @binding(0) = uniform LightPool（LightGlobals 48B + 8×LightDesc 384B + sky_top/horizon 32B）
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

// R3-18 直光层常量（与 Rust lighting.rs / wgsl_consts 镜像，单测防漂移）
const SHADOW_BIAS: f32 = 0.5;        // 阴影射线起点沿法线偏移（fine）
// 方向光阴影射线 t_max：场景 AABB 对角 ≈3118（[-256,-512,-256]~[1536,1536,1280]），
// 表面点沿任意方向的遮挡必在其内；65536 的空气段让每条阴影射线多空走 8×（性能）。
// 改世界尺度（GATE_TILES）时按对角线同步放大。
const SHADOW_DIR_T_MAX: f32 = 8192.0;
const EMISSIVE_EMIT_GAIN: f32 = 4.0;  // 发光体素 radiance 直出增益

// R3-10 DDGI 常量（与 Rust ddgi.rs 镜像，单测防漂移）
const DDGI_CELL: f32 = 16.0;            // 探针 cell 边长（fine）
const DDGI_IRR_TEXELS: u32 = 8u;        // 八面体 irradiance 边长（64 texel/探针）
const DDGI_DEPTH_TEXELS: u32 = 16u;     // 八面体 depth 边长（256 texel/探针）
const DDGI_RAYS_PER_PROBE: u32 = 64u;   // = 8×8，1 ray ↔ 1 irradiance texel
const DDGI_IRR_STRIDE: u32 = 64u;       // vec4 数/探针（8×8）
const DDGI_DEPTH_STRIDE: u32 = 256u;    // f32 数/探针（16×16）
// 采样权重（Rust ddgi.rs 常量镜像）
const DDGI_NORMAL_BIAS: f32 = 0.2;      // 前后权重锐度（Rohacek §3.2 锐利背面剔除）
const DDGI_DEPTH_BIAS: f32 = 4.0;       // 漏光 chevron 半宽 = cell × 0.25（Rohacek §3.3）

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
  inv_view_proj: mat4x4<f32>,  // 64B
  cam_pos_fine: vec4<f32>,     // 16B，w=1
  debug_mode: vec4<f32>,       // 16B：x = 法向可视化，y = face 6 色诊断
  lod: vec4<f32>,              // 16B：x = 像素角大小(rad)，y = LOD 早停开关
}
@group(0) @binding(1) var<uniform> view_u: DdaViewUniform;

// --- BG1：brick map 全部数据 + globals ---
@group(1) @binding(0) var<storage, read> b_struct: array<u32>;
@group(1) @binding(1) var<storage, read> b_leaves: array<u32>;
@group(1) @binding(2) var<storage, read> b_palette: array<u32>;

// --- BG5：逐体素直光可见性缓存（Douglas #19：每体素 1 条阴影射线 + hashmap 跨帧复用）---
struct VisCacheMeta {
  enabled: u32,       // 0 = 旁路（每像素直接投射）
  capacity_mask: u32, // 表容量 - 1（2 的幂）
  _pad0: u32,
  _pad1: u32,
}
@group(5) @binding(0) var<storage, read_write> vis_table: array<atomic<u32>>;
@group(5) @binding(1) var<uniform> vis_meta: VisCacheMeta;
// per-voxel implicit normal 表（Douglas #22：6 邻域 occupancy 差分，一体素一法线）
// slot 64bit = (hi: tag32, lo: oct_x16 | oct_y16)；hi==0 && lo==0 = 空
@group(5) @binding(2) var<storage, read_write> vis_norm: array<atomic<u32>>;

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

// --- BG3：光照光池（R3-18 直光层；与 Rust LightPoolUniform 逐字段镜像，464B）---
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
  sky_top: vec4<f32>,     // 天空天顶色（线性）
  sky_horizon: vec4<f32>, // 天空地平线色（线性）
}
@group(3) @binding(0) var<uniform> light_u: LightPool;

// --- BG4：DDGI 探针（R3-10；与 Rust DdgiMeta/buffer 布局逐字段镜像）---
//   @binding(0) = DdgiMeta uniform（64B）
//   @binding(1) = positions: vec4[probe_count]（xyz = 世界 fine 位置，w = active）
//   @binding(2) = cell_index: u32[cell_grid]（dense cell → probe id，u32::MAX = 无探针）
//   @binding(3) = irradiance: vec4[probe_count×64]（read_write；8×8 八面体/探针，EMA 累积）
//   @binding(4) = depth: f32[probe_count×256]（read_write；16×16 八面体/探针，EMA 累积）
// ddgi_update 写 3/4；dda_main（spike 4 采样接入后）只读 1..4；同一 BG4 两个 pass 复用。
struct DdgiMeta {
  probe_count: u32,
  active_count: u32,
  probes_this_frame: u32,
  cycle_base: u32,
  frame: u32,
  _pad0: u32,
  grid_origin: vec4<f32>,   // xyz = cell 网格原点（cell 单位）
  grid_dims: vec4<f32>,     // xyz = cell 网格 dims（cell 单位）
  cell_tmax_alpha: vec4<f32>, // x=cell(16), y=射线 t_max, z=EMA α, w reserved
}
@group(4) @binding(0) var<uniform> ddgi: DdgiMeta;
@group(4) @binding(1) var<storage, read> ddgi_pos: array<vec4<f32>>;
@group(4) @binding(2) var<storage, read> ddgi_cell: array<u32>;
@group(4) @binding(3) var<storage, read_write> ddgi_irr: array<vec4<f32>>;
@group(4) @binding(4) var<storage, read_write> ddgi_depth: array<f32>;
// cell_index 无探针哨兵（Rust NO_PROBE = u32::MAX）
const DDGI_NO_PROBE: u32 = 4294967295u;


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

  // ---- DFS 4 层 mask 遍历（wire v3：level 3 inline 4 体素/word，level 0-2 紧凑）----
  let local = vec3<u32>(m);  // chunk 内 fine 坐标 0..255
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

// ============================================================================
// Douglas 式整数体素层级自适应 DDA（octo_march_core: march_intersection_buffer）
//
// 旧 tmax 栈帧版为每层节点维护 tmax/cell/t_enter/t_exit（13 字 ×4 帧），弹栈必
// 重载节点头（3+2 LUT load）。本版状态只有：
//   v: vec3<i32>           chunk 局部 fine 整数体素坐标
//   bricks[4]: Brick       每层分裂节点的 addr/mask/pal（下钻载入、跨级跳复用）
//   level: u32             3=根(256³,子块64³) … 0=叶(4³,inline 1³)
// 边界距离按整数对齐每次重算（side_distance_for_ray，纯函数无状态）；brick 内
// DDA 沿最近边界轴整子块步进；跨出 brick 后用 firstTrailingBit 一次跳到最粗可行
// 层（步进轴新坐标的尾随零位 = 对齐 run 长度，纯整数 ALU、零节点加载），消除
// 逐层弹栈/重载/再下钻链。命中语义与旧版一致（统一子块色=节点 palette、
// level 0 inline、保守命中 t=子块入口）。
//
// CPU 逐字镜像：gate-render/src/brickmap/dda.rs `trace_chunk_cpu`
// （tree_traversal_equivalence_300_rays / tree_traversal_fuzz_2000_rays_multiscale
//   对暴力逐体素参考锁定 palette 严格一致 / t 容差 1.0 / 命中面法线反向）。
// ============================================================================

// 层级命中记录：t 为 ro 系绝对 t；face_id 0..5 = ±xyz 六面（命中面法线索引）。
//   pre-check 命中（射线起点在固体 leaf 内，相机在体内 UB）：face_id 由调用方
//   用 normalize(-rd) 反推（首 chunk entry_face）。
struct FineHit {
  hit: bool,
  t: f32,
  pal: u32,
  face_id: u32,
}

// brick 缓存槽：一层分裂节点常驻（对应 CPU BrickCpu / Douglas BrickMaskEntry）
struct Brick {
  addr: u32,  // 节点绝对字址（b_struct）
  mask: u64,  // 64bit 分裂掩码（bit=1 = 子块分裂；bit=0 = 统一子块，色=pal）
  pal: u32,   // 节点 palette（统一子块颜色，0=空气）
}

// 单 chunk 内层级遍历。chunk_base = 根节点绝对字址；chunk_min = chunk 原点（局部 fine）。
// 射线段 [t0, t1]（ro 系绝对 t）；entry_face = 进入本 chunk 的面（首 chunk 由调用方
// 用 normalize(-rd) 兜底，后续 chunk 为跨 chunk 面）。
// 返回 FineHit（t 为 ro 系绝对 t）；走出 chunk 未命中 → hit=false。
//
// depth_cap（beam 保守模式）：gate 深度 d=(3-level) 的分裂子块且 d>=depth_cap 时
// 返回子块入口 t（保守下界）；主/阴影 pass depth_cap=3 恒不触发（d<=2）。
fn trace_chunk(chunk_base: u32, chunk_min: vec3<f32>,
               ro: vec3<f32>, rd: vec3<f32>, sign_v: vec3<i32>,
               t0: f32, t1: f32, entry_face: u32, depth_cap: u32) -> FineHit {
  // 擦边退化（t0>=t1：射线只蹭到 chunk 边界）→ 无体素内部可穿过，直接 miss
  if (t0 >= t1) { return FineHit(false, 0.0, 0u, 0u); }
  // chunk 局部 fine 坐标（chunk 原点 = 0）；t 仍是 ro 系绝对 t
  let ro_c = ro - chunk_min;
  var bricks: array<Brick, 4u>;
  let r_ml = b_struct[chunk_base];
  let r_mh = b_struct[chunk_base + 1u];
  let r_pal = b_struct[chunk_base + 2u] & 0xFFu;
  bricks[3u] = Brick(chunk_base, (u64(r_mh) << 32u) | u64(r_ml), r_pal);
  var level: u32 = 3u;
  // 当前体素（chunk 局部 fine 整数坐标，0..255；跨出 chunk 的步进瞬态可达 -1/256）
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
    if (level > 3u) { return FineHit(false, 0.0, 0u, 0u); }
    // ---- traverse：从当前 level 下钻到 v 处内容（Douglas traverse_bit_set）----
    loop {
      let b = bricks[level];
      // WGSL 移位 RHS 必须 u32
      let sh = vec3<u32>(level * 2u);
      let cell = clamp((v >> sh) & vec3<i32>(3), vec3<i32>(0), vec3<i32>(3));
      let idx = u32(cell.z * 16 + cell.y * 4 + cell.x);
      if (level == 0u) {
        // 叶节点 inline palette：4 体素/word，低 2 位选字节
        // （到此必为 mask!=0 的 inline 叶：mask==0 统一叶在父层下钻快路径已处理）
        let w = b_struct[b.addr + NODE_FIXED_WORDS + (idx >> 2u)];
        let leaf_pal = (w >> ((idx & 3u) * 8u)) & 0xFFu;
        if (leaf_pal != 0u) { return FineHit(true, cur_t, leaf_pal, face); }
        break; // 空气 leaf
      }
      let bit = u64(1) << idx;
      if ((b.mask & bit) == u64(0)) {
        // 统一子块：颜色 = 节点 palette（0=空气）
        if (b.pal != 0u) { return FineHit(true, cur_t, b.pal, face); }
        break; // 空气统一子块
      }
      // depth_cap（beam 保守）：gate 深度 d=3-level 的分裂子块到达 cap → 子块入口 t
      let gd = 3u - level;
      if (gd >= depth_cap) { return FineHit(true, cur_t, b.pal, face); }
      // 分裂子块 → popcount 定位 child（用原始 mask，非 LUT eff）
      let below = b.mask & (bit - u64(1));
      let pop = countOneBits(u32(below)) + countOneBits(u32(below >> 32u));
      let child_addr = chunk_base + b_struct[b.addr + NODE_FIXED_WORDS + pop];
      let c_ml = b_struct[child_addr];
      let c_mh = b_struct[child_addr + 1u];
      let c_pal = b_struct[child_addr + 2u] & 0xFFu;
      let c_mask = (u64(c_mh) << 32u) | u64(c_ml);
      // 统一子节点快路径（旧 c_mask==0）：wire 任意层的分裂位都可能指向 3 字统一
      // 节点（mask=0，pal 直决；叶层统一节点无 inline 16 字，禁读 addr+3 之后）
      if (c_mask == u64(0)) {
        if (c_pal != 0u) { return FineHit(true, cur_t, c_pal, face); }
        break;
      }
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
        let ecell = clamp((v >> sh) & vec3<i32>(3), vec3<i32>(0), vec3<i32>(3));
        let entry_i = u32(ecell.z * 16 + ecell.y * 4 + ecell.x);
        let oct = select(0u, 1u, sign_v.x >= 0)
          | (select(0u, 1u, sign_v.y >= 0) << 1u)
          | (select(0u, 1u, sign_v.z >= 0) << 2u);
        let lut_base = min(oct * 128u + entry_i * 2u, 1022u);
        var rl = b_leaves[lut_base];
        var rh = b_leaves[lut_base + 1u];
        rl = select(rl, 0xFFFFFFFFu, lut_disable);
        rh = select(rh, 0xFFFFFFFFu, lut_disable);
        let reach = (u64(rh) << 32u) | u64(rl);
        if ((b.mask & reach) == u64(0)) {
          level = level + 1u;
          if (level > 3u) { return FineHit(false, 0.0, 0u, 0u); }
        }
      }
    }
    // ---- v 处为空气：当前 level brick 内 DDA（Douglas dda）----
    let log2 = level * 2u;
    let s = 1i << (level * 2u); // 子块边长 fine：1/4/16/64
    let s_mask = ~(s - 1i);    // 对齐掩码（two's complement: ~(s-1) = -s）
    // side_distance_for_ray：v 对齐到 s 的基址；正向 → 基址+s，负向 → 基址
    let base_v = v & vec3<i32>(s_mask);
    let boundary = vec3<f32>(base_v)
      + select(vec3<f32>(0.0), vec3<f32>(f32(s)), sign_v >= vec3<i32>(0));
    var side = vec3<f32>(1e+30);
    let axis_on = abs(rd) > vec3<f32>(1e-30);
    side = select(side, (boundary - ro_c) / rd, axis_on);
    side = max(side, vec3<f32>(cur_t));
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
      if (cur_t >= t1) { return FineHit(false, 0.0, 0u, 0u); } // 段内再无子块可入
      let old_cell = (v[mn] >> log2) & 3;
      // 沿 mn 轴整子块跨越
      v[mn] = v[mn] + sign_v[mn] * s;
      side[mn] = side[mn] + f32(s) / abs(rd[mn]);
      face = mn * 2u + select(1u, 0u, sign_v[mn] >= 0);
      // 跨出 brick（4 子块）？正向往 3→外、负向往 0→外
      let crossed = select(old_cell == 0, old_cell == 3, sign_v[mn] >= 0);
      if (crossed) { changed = true; break; }
      // 新子块内容：level 0 查 inline palette（mask!=0 inline 叶才有）；
      // level 1..3 = 分裂位或节点统一实体色
      let b = bricks[level];
      let sh2 = vec3<u32>(log2);
      let cell = clamp((v >> sh2) & vec3<i32>(3), vec3<i32>(0), vec3<i32>(3));
      let idx = u32(cell.z * 16 + cell.y * 4 + cell.x);
      var occ = false;
      if (level == 0u) {
        if (b.mask == u64(0)) {
          occ = b.pal != 0u;
        } else {
          let w = b_struct[b.addr + NODE_FIXED_WORDS + (idx >> 2u)];
          occ = ((w >> ((idx & 3u) * 8u)) & 0xFFu) != 0u;
        }
      } else {
        occ = ((b.mask & (u64(1) << idx)) != u64(0)) || (b.pal != 0u);
      }
      if (occ) { break; } // 有内容 → 回 traverse 下钻/命中
      // 空气子块 → 回 loop 顶重选 mn 轴继续
    }
    if (changed) {
      level = level + 1u;
      if (level > 3u) { return FineHit(false, 0.0, 0u, 0u); }
    }
    // ---- firstTrailingBit 层级自适应跨级跳（Douglas march 尾部）----
    // 步进轴新坐标的尾随零位 = 对齐 run 长度：正向 comp=对齐基址（tz 直接读），
    // 负向 comp=区域尾址（基址|~mask，+1 后读 tz）。tz>>1 = 可跨步的最粗 level。
    let positive = sign_v[step_axis] >= 0;
    let cur_log2 = level * 2u;
    let m: u32 = 0xFFFFFFFFu << cur_log2;
    let vmin_u = bitcast<u32>(v[step_axis]); // i32→u32 位环绕（负值公式自然处理）
    var comp: u32;
    if (positive) { comp = vmin_u & m; } else { comp = (vmin_u & m) | ~m; }
    let tz_i = firstTrailingBit(comp + select(1u, 0u, positive)); // 0 → -1
    var new_level: u32 = 0xFFFFFFFFu;
    if (tz_i >= 0) { new_level = u32(tz_i) >> 1u; }
    level = max(level, new_level);
    if (level > 3u) { return FineHit(false, 0.0, 0u, 0u); } // 跨出 chunk（tz≥8）
    // 对齐快照：v 钳到 cur_t 射线点所在的当前 level 区域，步进轴取精确边界整数
    // （其余轴按射线实际位置吸附，消除只沿单轴步进的漂移）
    let mi = bitcast<i32>(m);
    let base = v & vec3<i32>(mi);
    let p = ro_c + rd * cur_t;
    let region_max = base + vec3<i32>(bitcast<i32>(~m));
    v = clamp(vec3<i32>(floor(p)), base, region_max);
    v[step_axis] = bitcast<i32>(comp);
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
  var delta = vec3<f32>(1e+30);
  delta = select(delta, 1.0 / abs(rd), abs(rd) > vec3<f32>(1e-30));
  let delta_c = delta * f32(CHUNK_SIZE);
  let start = ro + rd * tl0;
  var ci = vec3<i32>(floor(start / f32(CHUNK_SIZE)));
  // tmax_c：到下一 chunk 边界的 t（ro 系绝对）
  var tmax_c = vec3<f32>(1e+30);
  let bnd_c = (vec3<f32>(ci) + side) * f32(CHUNK_SIZE);
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
        // P3 beam：chunk 内遍历从 max(chunk 入口, t_min) 开始，跳过 t_min 前的空空间
        let t0c = max(t_enter_c, t_min);
        let h = trace_chunk(chunk_base, chunk_min, ro, rd, sign_v,
                            t0c, t1, entry_face, depth_cap);
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
    select(-1i, i32(idx), idx > 0u),       // idx=0 → 主世界(-1)，idx≥1 → 物体
  );
}


// ---- palette albedo 解包 ----
fn palette_albedo(palette_base: u32, pal: u32) -> vec3<f32> {
  let w0 = b_palette[palette_base + pal * 2u];
  return vec3<f32>(
    f32(w0 & 0xFFu),
    f32((w0 >> 8u) & 0xFFu),
    f32((w0 >> 16u) & 0xFFu),
  ) / 255.0;
}

// palette emissive 解包（w1 低 8bit，Rust hit_mat 镜像）
fn palette_emissive(palette_base: u32, pal: u32) -> f32 {
  let w1 = b_palette[palette_base + pal * 2u + 1u];
  return f32(w1 & 0xFFu) / 255.0;
}

// ============================================================================
// R3-18 直光层（Douglas #02/#17/#23；CPU 镜像 lighting.rs::cpu_reference_sky/shade_hit）
//   final = albedo × (ambient×0.4 + sky_grad×0.6) + albedo × sun × NdotL × vis + emissive 直出
//   着色粒度 = 逐体素 flat（v5 决策）；阴影 = 命中点 1 条向太阳射线（硬阴影）
// ============================================================================

// 天空渐变 + 太阳盘光晕（miss 像素输出；CPU 镜像 cpu_reference_sky）
fn sky_color(dir: vec3<f32>) -> vec3<f32> {
  let d = normalize(dir);
  let h = clamp(d.y, 0.0, 1.0);
  let x = clamp(h / 0.35, 0.0, 1.0);
  let t = x * x * (3.0 - 2.0 * x);
  var col = mix(light_u.sky_horizon.xyz, light_u.sky_top.xyz, vec3<f32>(t));
  if (light_u.g.count > 0u && light_u.lights[0].kind_pos_dir.x < 0.5) {
    let sdir = light_u.lights[0].kind_pos_dir.yzw;
    let sun_c = light_u.lights[0].color_intensity.xyz * light_u.lights[0].color_intensity.w;
    let glow = pow(max(dot(d, sdir), 0.0), 64.0) * 0.05 * select(0.0, 1.0, h > 0.0);
    col = col + sun_c * glow;
  }
  return col;
}

// ============================================================================
// 逐体素直光可见性缓存（Douglas #19：每体素 1 条阴影射线 + hashmap 跨帧复用，
// 1660 Ti 实测省 1-2ms）。方向光 vis 视角无关 → 跨帧持久；编辑时整表清零
// （edit_generation，Rust 侧）。slot = tag30<<2 | state（0=空 1=遮挡 2=可见）。
// 语义：per-voxel vis 由受光面（朝太阳面）中心唯一一条射线判定，全面共享；
// ndl 仍按命中面法线（#22 隐式法线明暗不变）。
// ============================================================================

// key 折叠：obj_id(4bit) + 体素局部坐标三轴各 14bit（±8192）→ 32bit hash
fn vis_key(obj_id: i32, v: vec3<i32>) -> u32 {
  let o = (u32(obj_id) + 1u) & 15u;
  let x = u32(v.x) & 16383u;
  let y = u32(v.y) & 16383u;
  let z = u32(v.z) & 16383u;
  var h = (o * 0x9E3779B1u) ^ (x * 0x85EBCA6Bu) ^ (y * 0xC2B2AE35u) ^ (z * 0x27D4EB2Fu);
  h = h ^ (h >> 15u);
  h = h * 0x2C1B3C6Du;
  h = h ^ (h >> 12u);
  return h;
}

// 线性探测查询（读侧无原子——竞态最坏读到撕裂 tag → miss 重投射，无害）
fn vis_lookup(k: u32) -> u32 {
  let tag = k & 0x3FFFFFFFu;
  var idx = k & vis_meta.capacity_mask;
  for (var i = 0u; i < 4u; i = i + 1u) {
    let slot = atomicLoad(&vis_table[idx]);
    if (slot == 0u) { return 0u; }
    if ((slot >> 2u) == tag) { return slot & 3u; }
    idx = (idx + 1u) & vis_meta.capacity_mask;
  }
  return 0u;
}

// CAS 插入（两线程同 key 竞态：一方成功，另一方投射结果相同值，无害）
fn vis_insert(k: u32, state: u32) {
  let tag = k & 0x3FFFFFFFu;
  let val = (tag << 2u) | (state & 3u);
  var idx = k & vis_meta.capacity_mask;
  for (var i = 0u; i < 4u; i = i + 1u) {
    let r = atomicCompareExchangeWeak(&vis_table[idx], 0u, val);
    if (r.exchanged) { return; }
    if ((r.old_value >> 2u) == tag) { return; }
    idx = (idx + 1u) & vis_meta.capacity_mask;
  }
}

// 体素局部坐标：主世界 identity 直取；物体经 Grid 逆变换
fn vis_voxel_local(obj_id: i32, p_world: vec3<f32>) -> vec3<i32> {
  if (obj_id < 0) {
    return vec3<i32>(floor(p_world));
  }
  let gg = make_grid(u32(obj_id) + 1u);
  let wp = p_world - gg.pos;
  let lp = vec3<f32>(dot(wp, gg.col0), dot(wp, gg.col1), dot(wp, gg.col2)) / gg.scale;
  return vec3<i32>(floor(lp));
}

// ============================================================================
// per-voxel implicit normal（Douglas #22：6 邻域 occupancy 差分，一体素一法线
// → 一体素一色）。首算贵（6 次树点查）→ 64bit 表缓存跨帧复用，稳态 O(1)。
// ============================================================================

// 八面体方向 16bit 量化（DDGI oct 复用；normal 是单位向量）
fn vis_norm_encode(n: vec3<f32>) -> u32 {
  let e = oct_encode(n) * 0.5 + vec2<f32>(0.5);
  let q = vec2<u32>(clamp(e, vec2<f32>(0.0), vec2<f32>(1.0)) * 65535.0);
  return (q.x << 16u) | q.y;
}
fn vis_norm_decode(v: u32) -> vec3<f32> {
  let e = vec2<f32>(f32((v >> 16u) & 0xFFFFu), f32(v & 0xFFFFu)) / 65535.0 * 2.0 - vec2<f32>(1.0);
  return oct_decode(e);
}

// 6 邻域 occupancy 差分；退化（零向量）→ 回退 face normal（局部系）
fn implicit_normal_local(g: Grid, v: vec3<i32>, fallback: vec3<f32>) -> vec3<f32> {
  let px = select(0u, 1u, sample_brickmap(g, v + vec3<i32>(1, 0, 0)) != 0u);
  let nx = select(0u, 1u, sample_brickmap(g, v + vec3<i32>(-1, 0, 0)) != 0u);
  let py = select(0u, 1u, sample_brickmap(g, v + vec3<i32>(0, 1, 0)) != 0u);
  let ny = select(0u, 1u, sample_brickmap(g, v + vec3<i32>(0, -1, 0)) != 0u);
  let pz = select(0u, 1u, sample_brickmap(g, v + vec3<i32>(0, 0, 1)) != 0u);
  let nz = select(0u, 1u, sample_brickmap(g, v + vec3<i32>(0, 0, -1)) != 0u);
  let d = vec3<f32>(f32(nx) - f32(px), f32(ny) - f32(py), f32(nz) - f32(pz));
  let len = length(d);
  return select(fallback, d / len, len > 0.5);
}

// per-voxel normal 世界系（带缓存）；miss → 6 邻域差分（局部系）→ 世界系。
// fallback_face_id：差分零向量（孤立体素/全实心）时退回命中面法向。
fn vis_normal(obj_id: i32, v_local: vec3<i32>, fallback_face_id: u32) -> vec3<f32> {
  let k = vis_key(obj_id, v_local);
  let tag = k;
  let idx = k & vis_meta.capacity_mask;
  if (tag != 0u) {
    let hi = atomicLoad(&vis_norm[idx * 2u]);
    if (hi == tag) {
      let lo = atomicLoad(&vis_norm[idx * 2u + 1u]);
      return vis_norm_decode(lo);
    }
  }
  // miss：6 邻域差分
  let gg = make_grid(u32(obj_id) + 1u);
  let fd_local = face_normal_from_index(fallback_face_id);
  let n_local = implicit_normal_local(gg, v_local, fd_local);
  let n_world = normalize(n_local.x * gg.col0 + n_local.y * gg.col1 + n_local.z * gg.col2);
  if (tag != 0u) {
    let enc = vis_norm_encode(n_world);
    // 插入先 lo 后 hi：读侧 hi==tag 成立时 lo 必已写入（零撕裂）；
    // 反序会读到 (新 hi, 旧 lo=0) → decode 出错法线 1 帧
    atomicStore(&vis_norm[idx * 2u + 1u], enc);
    atomicStore(&vis_norm[idx * 2u], tag);
  }
  return n_world;
}

// 太阳遮挡判定（带缓存）。未算 → 从体素受光面中心投唯一一条阴影射线并插入。
fn vis_sun_blocked(obj_id: i32, p_world: vec3<f32>, n: vec3<f32>) -> bool {
  let l_axis = light_u.lights[0].kind_pos_dir.yzw;
  if (vis_meta.enabled == 0u) {
    // 旁路：沿用旧逐像素路径（命中点 + 面法线偏移投射线）
    let o = p_world + n * SHADOW_BIAS;
    let sh = trace_scene(o, l_axis, SHADOW_DIR_T_MAX, 0.0, 3u);
    return sh.uh.hit;
  }
  let v_local = vis_voxel_local(obj_id, p_world);
  let k = vis_key(obj_id, v_local);
  var st = vis_lookup(k);
  if (st == 0u) {
    let fd = -l_axis;
    let an = abs(fd);
    var nf = vec3<f32>(0.0);
    if (an.x >= max(an.y, an.z)) {
      nf = vec3<f32>(select(-1.0, 1.0, fd.x >= 0.0), 0.0, 0.0);
    } else if (an.y >= an.z) {
      nf = vec3<f32>(0.0, select(-1.0, 1.0, fd.y >= 0.0), 0.0);
    } else {
      nf = vec3<f32>(0.0, 0.0, select(-1.0, 1.0, fd.z >= 0.0));
    }
    // 局部受光面中心 → 世界
    let fc_local = vec3<f32>(v_local) + vec3<f32>(0.5) + nf * 0.5;
    var o_world: vec3<f32>;
    if (obj_id < 0) {
      o_world = fc_local + nf * SHADOW_BIAS;
    } else {
      let gg = make_grid(u32(obj_id) + 1u);
      let rows = mat3x3<f32>(
        vec3<f32>(gg.col0.x, gg.col1.x, gg.col2.x),
        vec3<f32>(gg.col0.y, gg.col1.y, gg.col2.y),
        vec3<f32>(gg.col0.z, gg.col1.z, gg.col2.z),
      );
      o_world = gg.pos + (rows * fc_local) * gg.scale + nf * SHADOW_BIAS;
    }
    let sh = trace_scene(o_world, l_axis, SHADOW_DIR_T_MAX, 0.0, 3u);
    st = select(2u, 1u, sh.uh.hit);
    vis_insert(k, st);
  }
  return st == 1u;
}

// 场景级命中：UnifiedHit + 命中 volume 的 palette 基址（shade_hit 取材质用）
struct SceneHit {
  uh: UnifiedHit,
  palette_base: u32,
}

// 主世界先跑 + 逐物体收缩 t_cap 取最近命中（主射线/阴影射线共用）。
// 物体循环不被世界 miss 短路：天空背景前的物体必须可见，阴影射线必须被物体遮挡。
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

// 直光着色（CPU 镜像 cpu_reference_shade_hit）：sky 渐变环境光 + 太阳硬阴影 + emissive 直出。
// 返回 pre-exposure 线性辐射度：DDGI probe 端点着色必须存 pre-exposure 值（曝光只在
// 最终合成施加一次，否则 EMA 累积会把曝光平方化）；dda_main 走 shade_hit 包装乘曝光。
fn shade_hit_linear(origin: vec3<f32>, dir: vec3<f32>, hit: SceneHit) -> vec3<f32> {
  let p = origin + dir * hit.uh.t;
  // per-voxel implicit normal（Douglas #22：6 邻域差分一体素一法线 → 一体素一色）。
  // 面法线（hit.uh.n）仅作差分退化 fallback 与 debug 可视化；跨面不再变明暗。
  let v_local = vis_voxel_local(hit.uh.obj_id, p);
  let n = vis_normal(hit.uh.obj_id, v_local, hit.uh.face_id);
  let base = palette_albedo(hit.palette_base, hit.uh.pal);
  let emissive = palette_emissive(hit.palette_base, hit.uh.pal);

  // sky 渐变环境光（按法线 y 混合地平线/天顶，smoothstep(0.35)）
  let h = clamp(n.y, 0.0, 1.0);
  let x = clamp(h / 0.35, 0.0, 1.0);
  let t_sky = x * x * (3.0 - 2.0 * x);
  let sky_grad = mix(light_u.sky_horizon.xyz, light_u.sky_top.xyz, vec3<f32>(t_sky));
  var col = base * (light_u.g.ambient.xyz * 0.4 + sky_grad * 0.6);

  // 方向光硬阴影（1 条射线，不通即阴影；无点光源/无软阴影/无 NEE）
  // 逐体素直光（Douglas #19）：vis 按体素受光面唯一射线判定 + 缓存复用
  if (light_u.g.count > 0u && light_u.lights[0].kind_pos_dir.x < 0.5) {
    let l_axis = light_u.lights[0].kind_pos_dir.yzw;
    let ndl = max(dot(n, l_axis), 0.0);
    if (ndl > 0.0) {
      let blocked = vis_sun_blocked(hit.uh.obj_id, p, n);
      let vis = select(1.0, 0.0, blocked);
      let sun_c = light_u.lights[0].color_intensity.xyz * light_u.lights[0].color_intensity.w;
      col = col + base * sun_c * (ndl * vis);
    }
  }

  // 发光体素 radiance 直出（无方向性、不受阴影）
  col = col + base * (emissive * EMISSIVE_EMIT_GAIN);
  // DDGI 间接光（R3-10 spike 4）：探针三线性采样的入射辐射度 × albedo。
  // probe 射线端点也走本函数 → 自动采样上一帧 DDGI（自闭环 = 无限反弹）；
  // 首帧 irradiance 为 0，随 EMA 收敛逐步填充。
  col = col + base * sample_ddgi(p, n);
  return col;
}

// 主射线着色：pre-exposure 线性辐射度 × 曝光（最终合成唯一曝光点）
fn shade_hit(origin: vec3<f32>, dir: vec3<f32>, hit: SceneHit) -> vec3<f32> {
  return shade_hit_linear(origin, dir, hit) * light_u.g.exposure_pad.x;
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
  let origin = view_u.cam_pos_fine.xyz;
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
// DDA 主入口：R3-18 直光层 — 命中 → 直光着色（硬阴影 + sky 环境 + emissive），miss → 天空渐变
// ============================================================================
@compute @workgroup_size(8, 8, 1)
fn dda_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let size = textureDimensions(out_tex);
  if (gid.x >= size.x || gid.y >= size.y) { return; }
  let coord0 = vec2<i32>(i32(gid.x), i32(gid.y));

  // debug_mode.w > 1.5（诊断模式）：跳过全部 trace，直接天空色输出
  if (view_u.debug_mode.w > 1.5) {
    textureStore(out_tex, coord0, vec4<f32>(linear_to_srgb(vec3<f32>(0.52, 0.80, 1.0)), 1.0));
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

  // debug_mode.w > 3.5（诊断模式）：只做 make_grid（读 GridDesc）不 trace
  if (view_u.debug_mode.w > 3.5) {
    let n = arrayLength(&grid_descs);
    var sink = 0u;
    for (var i: u32 = 0u; i < n; i = i + 1u) {
      let gg = make_grid(i);
      sink = sink + u32(gg.scale) + u32(gg.w_mn.x) + u32(gg.col2.z);
    }
    var col1 = vec3<f32>(0.52, 0.80, 1.0);
    col1 = col1 + vec3<f32>(f32(sink & 1u) * 0.001);
    textureStore(out_tex, coord0, vec4<f32>(linear_to_srgb(col1), 1.0));
    return;
  }

  // ---- 场景 trace + 直光着色 ----
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
  let best = trace_scene(origin_fine, dir_fine, frustum_length, t_min, 3u);
  if (view_u.debug_mode.w > 0.5 && view_u.debug_mode.w < 1.5) {
    let alb = palette_albedo(best.palette_base, best.uh.pal);
    textureStore(out_tex, coord0, vec4<f32>(linear_to_srgb(alb), 1.0));
    return;
  }
  if (best.uh.hit) {
    let col = shade_hit(origin_fine, dir_fine, best);
    textureStore(out_tex, coord0, vec4<f32>(linear_to_srgb(col), 1.0));
  } else {
    // 天空渐变 + 太阳盘光晕
    textureStore(out_tex, coord0, vec4<f32>(linear_to_srgb(sky_color(dir_fine)), 1.0));
  }
}

// ============================================================================
// R3-10 DDGI 射线更新 pass（Majercik 2019 §4；CPU 镜像 ddgi.rs 数学函数）
//
// 1 workgroup = 1 探针，workgroup_size = 64（= DDGI_RAYS_PER_PROBE）；
// 每帧 dispatch (probes_this_frame, 1, 1)，workgroup 环形轮转
// （probe_id = (cycle_base + wid) % probe_count），固定射线预算不随屏上探针数波动。
// 每线程：1 射线 trace_scene → 端点着色（sky / emissive 直出 / 直光 1-bounce）→
// EMA 写 8×8 irradiance texel（1 ray ↔ 1 texel，方向 = irradiance oct texel 中心）
// + 同方向 16×16 depth texel（漏光治理用，稀疏写；未写 texel 保持 tmax 远距初值）。
// ============================================================================

// 八面体编解码（Majercik 2019 §3；CPU 镜像 ddgi.rs oct_encode/oct_decode）
fn oct_sign_not_zero(x: f32) -> f32 {
  return select(-1.0, 1.0, x >= 0.0);
}
fn oct_encode(n: vec3<f32>) -> vec2<f32> {
  let d = n / (abs(n.x) + abs(n.y) + abs(n.z));
  if (d.z < 0.0) {
    return vec2<f32>(
      (1.0 - abs(d.y)) * oct_sign_not_zero(d.x),
      (1.0 - abs(d.x)) * oct_sign_not_zero(d.y),
    );
  }
  return vec2<f32>(d.x, d.y);
}
fn oct_decode(e: vec2<f32>) -> vec3<f32> {
  var n = vec3<f32>(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
  if (n.z < 0.0) {
    let ox = (1.0 - abs(n.y)) * oct_sign_not_zero(n.x);
    let oy = (1.0 - abs(n.x)) * oct_sign_not_zero(n.y);
    n.x = ox;
    n.y = oy;
  }
  return normalize(n);
}
// texel 中心 → 方向（边长 S）；ray 方向取自 irradiance 8×8 texel 中心
fn oct_texel_dir(tx: u32, ty: u32, s: u32) -> vec3<f32> {
  let e = (vec2<f32>(f32(tx), f32(ty)) + vec2<f32>(0.5)) / f32(s) * 2.0 - vec2<f32>(1.0);
  return oct_decode(e);
}
// 方向 → depth 16×16 texel 下标（同方向映射，稀疏写）
fn oct_texel_index(n: vec3<f32>, s: u32) -> vec2<u32> {
  let e = oct_encode(n) * 0.5 + vec2<f32>(0.5);
  let f = e * f32(s);
  return vec2<u32>(
    clamp(u32(i32(floor(f.x))), 0u, s - 1u),
    clamp(u32(i32(floor(f.y))), 0u, s - 1u),
  );
}

// ============================================================================
// DDGI 着色采样（Majercik 2019 §5 + Rohacek §3.2/§3.3）
// 最近 8 cell 三线性 × 锐利背面权重 × 漏光深度 chevron，归一化加权和。
// 返回探针方向到达 p 点的入射辐射度（pre-exposure）；调用方乘 albedo。
// ============================================================================

// irradiance 8×8 单 texel
fn ddgi_irr_fetch(id: u32, tx: u32, ty: u32) -> vec3<f32> {
  return ddgi_irr[id * DDGI_IRR_STRIDE + ty * DDGI_IRR_TEXELS + tx].rgb;
}
// irradiance 八面体双线性采样（边界硬钳；八面体折缝处轻微接缝，spike 5 精修候选）
fn ddgi_irr_sample(id: u32, d: vec3<f32>) -> vec3<f32> {
  let e = oct_encode(d) * 0.5 + vec2<f32>(0.5);
  let g = e * f32(DDGI_IRR_TEXELS) - vec2<f32>(0.5);
  let g0 = vec2<i32>(floor(g));
  let f = g - vec2<f32>(g0);
  let x0 = clamp(g0.x, 0, i32(DDGI_IRR_TEXELS) - 1);
  let y0 = clamp(g0.y, 0, i32(DDGI_IRR_TEXELS) - 1);
  let x1 = clamp(g0.x + 1, 0, i32(DDGI_IRR_TEXELS) - 1);
  let y1 = clamp(g0.y + 1, 0, i32(DDGI_IRR_TEXELS) - 1);
  let c00 = ddgi_irr_fetch(id, u32(x0), u32(y0));
  let c10 = ddgi_irr_fetch(id, u32(x1), u32(y0));
  let c01 = ddgi_irr_fetch(id, u32(x0), u32(y1));
  let c11 = ddgi_irr_fetch(id, u32(x1), u32(y1));
  return mix(
    mix(c00, c10, vec3<f32>(f.x)),
    mix(c01, c11, vec3<f32>(f.x)),
    vec3<f32>(f.y),
  );
}
// depth 16×16 最近邻（保守剔除：不跨 texel 混合深度）
fn ddgi_depth_sample(id: u32, d: vec3<f32>) -> f32 {
  let e = oct_encode(d) * 0.5 + vec2<f32>(0.5);
  let g = vec2<i32>(floor(e * f32(DDGI_DEPTH_TEXELS)));
  let g0 = clamp(g, vec2<i32>(0), vec2<i32>(i32(DDGI_DEPTH_TEXELS) - 1));
  return ddgi_depth[id * DDGI_DEPTH_STRIDE + u32(g0.y) * DDGI_DEPTH_TEXELS + u32(g0.x)];
}

// 世界点 p（表面法线 n）的 DDGI 间接入射辐射度；无探针/全剔除 → 0
fn sample_ddgi(p: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
  if (ddgi.probe_count == 0u) { return vec3<f32>(0.0); }
  let cell_f = p / vec3<f32>(DDGI_CELL);
  let c0 = floor(cell_f);
  let f = cell_f - c0;
  let dims = ddgi.grid_dims.xyz;
  var total = vec3<f32>(0.0);
  var wsum = 0.0;
  for (var iz = 0i; iz < 2; iz = iz + 1i) {
    for (var iy = 0i; iy < 2; iy = iy + 1i) {
      for (var ix = 0i; ix < 2; ix = ix + 1i) {
        let corner = c0 + vec3<f32>(f32(ix), f32(iy), f32(iz));
        let rel = corner - ddgi.grid_origin.xyz;
        if (any(rel < vec3<f32>(0.0)) || any(rel >= dims)) { continue; }
        let ru = vec3<u32>(rel);
        let ci = ru.x + ru.y * u32(dims.x) + ru.z * u32(dims.x) * u32(dims.y);
        let id = ddgi_cell[ci];
        if (id == DDGI_NO_PROBE) { continue; }
        // 三线性权重（corner=(c0+1) 取 f，否则 1-f）
        let wtri = select(1.0 - f.x, f.x, ix == 1i)
                 * select(1.0 - f.y, f.y, iy == 1i)
                 * select(1.0 - f.z, f.z, iz == 1i);
        if (wtri <= 1e-6) { continue; }
        let probe = ddgi_pos[id].xyz;
        let to = p - probe;
        let dist = length(to);
        let dir = to / max(dist, 1e-4);
        // 锐利背面剔除（Rohacek §3.2）：探针在表面后侧 → 权重 0（穿墙不漏光）
        let wn = clamp(dot(n, dir) / DDGI_NORMAL_BIAS, 0.0, 1.0);
        if (wn <= 0.0) { continue; }
        // 漏光 chevron（Rohacek §3.3 / Majercik §5）：
        // dtex = 探针沿 dir 到最近几何距离；墙在探针与 p 之间（dtex < dist）→ 降权/剔除
        let dtex = ddgi_depth_sample(id, dir);
        let wd = clamp((dtex - dist) / DDGI_DEPTH_BIAS + 0.5, 0.0, 1.0);
        if (wd <= 0.0) { continue; }
        let irr = ddgi_irr_sample(id, dir);
        let w = wtri * wn * wd;
        total = total + irr * w;
        wsum = wsum + w;
      }
    }
  }
  if (wsum < 1e-4) { return vec3<f32>(0.0); }
  return total / wsum;
}

// Fibonacci 球方向（Majercik 2019 §4；CPU 镜像 ddgi.rs fibonacci_dir）
// 当前射线方向直接取 oct texel 中心（1 ray ↔ 1 texel），本函数保留给后续
// 帧间相位旋转/蓝噪声抖动（R3-14）。
fn fibonacci_dir(i: u32, n: u32) -> vec3<f32> {
  let golden = 3.14159265359 * (3.0 - sqrt(5.0));
  let y = 1.0 - (2.0 * f32(i) + 1.0) / f32(n);
  let r = sqrt(max(1.0 - y * y, 0.0));
  let a = golden * f32(i);
  return vec3<f32>(r * cos(a), y, r * sin(a));
}

// probe 射线端点着色（ddgi.rs 模块头契约）：
//   sky 命中 → sky() 渐变；emissive 体素 → albedo × emissive × GAIN（gate 通电照亮暗室）；
//   普通命中 → 直光着色（太阳硬阴影 + sky 环境，1-bounce；pre-exposure）。
// 上一帧 DDGI 自闭环（无限反弹）在 spike 4 采样函数落地后接入。
fn probe_endpoint_shade(origin: vec3<f32>, rd: vec3<f32>, sh: SceneHit) -> vec3<f32> {
  let emissive = palette_emissive(sh.palette_base, sh.uh.pal);
  if (emissive > 0.0) {
    let albedo = palette_albedo(sh.palette_base, sh.uh.pal);
    return albedo * (emissive * EMISSIVE_EMIT_GAIN);
  }
  return shade_hit_linear(origin, rd, sh);
}

@compute @workgroup_size(DDGI_RAYS_PER_PROBE, 1, 1)
fn ddgi_update(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
) {
  if (ddgi.probe_count == 0u) { return; }
  let wg = wid.x;
  if (wg >= ddgi.probes_this_frame) { return; }
  let ray = gid.x;  // 0..63
  let probe_id = (ddgi.cycle_base + wg) % ddgi.probe_count;
  let origin = ddgi_pos[probe_id].xyz;

  // 射线方向 = irradiance 8×8 texel 中心（1 ray ↔ 1 texel）
  let tx = ray % DDGI_IRR_TEXELS;
  let ty = ray / DDGI_IRR_TEXELS;
  let rd = oct_texel_dir(tx, ty, DDGI_IRR_TEXELS);

  let t_max = ddgi.cell_tmax_alpha.y;
  let sh = trace_scene(origin, rd, t_max, 0.0, 3u);

  var radiance: vec3<f32>;
  var dist: f32 = t_max;
  if (sh.uh.hit) {
    dist = sh.uh.t;
    radiance = probe_endpoint_shade(origin, rd, sh);
  } else {
    radiance = sky_color(rd);
  }

  // ---- irradiance EMA（8×8，texel 与射线 1:1）----
  let irr_idx = probe_id * DDGI_IRR_STRIDE + ray;
  let old = ddgi_irr[irr_idx];
  let alpha = ddgi.cell_tmax_alpha.z;
  ddgi_irr[irr_idx] = vec4<f32>(mix(old.rgb, radiance, vec3<f32>(alpha)), 1.0);

  // ---- depth EMA（16×16，同方向 oct 映射；稀疏 texel 保持 tmax 初值）----
  let dt = oct_texel_index(rd, DDGI_DEPTH_TEXELS);
  let d_idx = probe_id * DDGI_DEPTH_STRIDE + dt.y * DDGI_DEPTH_TEXELS + dt.x;
  ddgi_depth[d_idx] = mix(ddgi_depth[d_idx], dist, alpha);
}

