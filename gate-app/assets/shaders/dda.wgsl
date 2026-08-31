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
// BG2（P2.10 MOV object pool，与 Rust GpuMovPool 1:1 对应）：
//   @group(2) @binding(0) = mov_struct: array<u32>（逐对象 [bitmap|dirs|node] 拼接）
//   @group(2) @binding(1) = mov_leaves: array<u32>（brick slab 池拼接）
//   @group(2) @binding(2) = mov_palette: array<u32>（逐对象 256×2w 拼接）
//   @group(2) @binding(3) = mov_descs: array<u32>（32 words/物体，f32 字段 bitcast）
//   @group(2) @binding(4) = uniform MovGlobals（count，16B）
//
// 顶部常量与 Rust `brickmap::dda::wgsl_consts` 完全一致（单测 TR-2.1 assert_eq 防漂移）。
// BrickMap 五步寻址链严格对应 Rust `gate-render/src/brickmap/view.rs::get_voxel`（逐段注释 L 号）。
// Slot 打包规则对应 Rust `gate-render/src/brickmap/wire.rs::encode_slot/unpack_slot_word/pack_palette_entry`。
// trace_object 严格对应 Rust `brickmap/mov.rs::cpu_reference_object_ray`（等价性单测锁死 CPU 侧）。
// ============================================================================

// --- 常量区（与 Rust wgsl_consts mod 字节对齐）---
const TILE_INDEX_CAP: u32 = 128u;
const TILE_CAP: u32 = 1024u;
const BITMAP_BASE: u32 = 2097152u;     // 128^3
const TILE_BITMAP_WORDS: u32 = 1024u;  // 32^3 / 32
const DIR_BASE: u32 = 3145728u;        // BITMAP_BASE + TILE_CAP*TILE_BITMAP_WORDS = 2097152+1048576
const CELL_DIR_WORDS: u32 = 32768u;    // 32^3
const HDR_UNIFORM_MASK: u32 = 0xFFu;
const HDR_HAS_L1: u32 = 256u;          // 1<<8
const HDR_HAS_L2: u32 = 512u;          // 1<<9
const HDR_HAS_L3: u32 = 1024u;         // 1<<10
const HDR_HAS_BRICK: u32 = 2048u;      // 1<<11
const BRICK_SLAB_WORDS: u32 = 1024u;   // 4096B / 4
const ST_L1_WORDS: u32 = 4u;
const ST_L2_WORDS: u32 = 32u;
const ST_L3_WORDS: u32 = 256u;
const ST_BRICK_PTR_WORDS: u32 = 2u;    // hdr + slab_ptr = 2 words
const TILE_SUB: u32 = 512u;            // tile 边长（fine units = 32 cells × 16 sub/cell）
const SUB_PER_CELL: u32 = 16u;         // 基元胞边长（fine units）
// MOV（P2.10）
const NODE_STREAM_BASE: u32 = 36700160u;  // 128³ + 1024×1024 + 1024×32768（wire.rs）
const MOV_DESC_WORDS: u32 = 32u;          // 128B / 4 = 32 words/物体 descriptor
const LOCAL_TILE_FINE: u32 = 512u;        // 物体局部 tile 边长（v1 每物体恰 1 tile）

// Slot 编码（wire.rs encode_slot / unpack_slot_word）
// tag: 最高 8bit (u16)，palette: 低 8bit
fn slot_pack(tag: u32, pal: u32) -> u32 {
  return ((tag & 0xFFu) << 8u) | (pal & 0xFFu);
}
fn slot_unpack(word: u32, half: u32) -> u32 {
  // half=0 → lower 16bit; half=1 → upper 16bit
  let shift = half * 16u;
  return (word >> shift) & 0xFFFFu;
}
fn slot_tag(slot: u32) -> u32 { return (slot >> 8u) & 0xFFu; }
fn slot_palette(slot: u32) -> u32 { return slot & 0xFFu; }

// palette 解包（wire.rs pack_palette_entry 反函数，palette_idx = 0 => AIR，调用方已提前返回）
// word_0 = color.r | color.g<<8 | color.b<<16 | roughness<<24
// 返回 sRGB 颜色分量 [0,255] → 外部再转 f32
fn palette_rgb_u8(pal_idx: u32) -> vec3<u32> {
  let w0 = b_palette[pal_idx * 2u];
  let r = w0 & 0xFFu;
  let g = (w0 >> 8u) & 0xFFu;
  let b = (w0 >> 16u) & 0xFFu;
  return vec3<u32>(r, g, b);
}

// Tag 常量（与 wire.rs SLOT_TAG_EMPTY/LEAF/BRANCH 一致；定义在 wire.rs：
// const SLOT_TAG_EMPTY = 0; SLOT_TAG_LEAF = 1; SLOT_TAG_BRANCH = 2）
const TAG_EMPTY: u32 = 0u;
const TAG_LEAF: u32 = 1u;
const TAG_BRANCH: u32 = 2u;

// --- BG0：输出 + 视图 uniform ---
@group(0) @binding(0) var out_tex: texture_storage_2d<rgba8unorm, write>;

struct DdaViewUniform {
  inv_view_proj: mat4x4<f32>,  // 64B
  cam_pos_fine: vec4<f32>,     // 16B，w=1
  _pad0: vec4<f32>,            // 16B（对齐到 144B，144=64+16+16+encase min）
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

// --- BG2：MOV object pool（P2.10）---
@group(2) @binding(0) var<storage, read> mov_struct: array<u32>;
@group(2) @binding(1) var<storage, read> mov_leaves: array<u32>;
@group(2) @binding(2) var<storage, read> mov_palette: array<u32>;
@group(2) @binding(3) var<storage, read> mov_descs: array<u32>;
struct MovGlobals {
  count: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}
@group(2) @binding(4) var<uniform> mov_g: MovGlobals;

// ============================================================================
// 五步寻址链 sample_brickmap(fine: vec3<i32>) -> u32 (palette_idx, 0=AIR)
// 严格对应 view.rs get_voxel L55-147：
//   ① TileIndex 查找（view.rs L59-66）
//   ② TileBitmaps 空位检测（view.rs L68-75）
//   ③ CellDirs node 绝对字偏移（view.rs L77-84）
//   ④ walk_node 四级槽（L1→L2→L3→BRICK）解包（view.rs L86-147）
// ============================================================================
fn sample_brickmap(fine: vec3<i32>) -> u32 {
  // ---- VoxelPos::from_fine(fine, MAX_LEVEL=4)：tile + cell + in_cell ----
  // 欧氏余数（负坐标正确落邻接 tile）：m = ((fine % 512) + 512) % 512 ∈ 0..511
  // tile = (fine - m)/512；cell = m/16（0..31）；in_cell = m - cell*16（0..15）
  let m = ((fine % vec3<i32>(i32(TILE_SUB))) + vec3<i32>(i32(TILE_SUB))) % vec3<i32>(i32(TILE_SUB));
  let tile_i = (fine - m) / vec3<i32>(i32(TILE_SUB));
  let cell_i = m / vec3<i32>(i32(SUB_PER_CELL));             // 0..31
  let in_cell = m - cell_i * vec3<i32>(i32(SUB_PER_CELL));   // 0..15
  let it = vec3<u32>(cell_i);                                // cell 0..31

  // ---- ① TileIndex：g.index_origin/dims windowing（view.rs L59-66 index_pos） ----
  let origin = vec3<i32>(g.index_origin_x, g.index_origin_y, g.index_origin_z);
  let dims = vec3<u32>(g.index_dims_x, g.index_dims_y, g.index_dims_z);
  let rel = tile_i - origin;
  if (any(rel < vec3<i32>(0))) { return 0u; }
  // vecN<T>(vecN<U>) 整体转换构造（WGSL 无 i32→u32 隐式转换，分量混型构造非法）；
  // rel >= 0 已在上方检查，转换安全
  let rel_u = vec3<u32>(rel);
  if (any(rel_u >= dims)) { return 0u; }
  let index_addr = rel_u.x + rel_u.y * TILE_INDEX_CAP + rel_u.z * (TILE_INDEX_CAP * TILE_INDEX_CAP);
  let slot_idx = b_struct[index_addr];      // 0 = empty tile
  if (slot_idx == 0u) { return 0u; }
  // slot_idx 是 1-based tile 槽（wire.rs slot of tile）
  let slot = slot_idx - 1u;

  // ---- ② TileBitmaps：bitmap[slot] 中 in_tile coarse bit（view.rs L68-75） ----
  // 位索引 = (it.z/8)*16 + (it.y/8)*4 + (it.x/8) = 4x4x4 coarse cell bit
  let coarse_x = it.x >> 3u;   // /8
  let coarse_y = it.y >> 3u;
  let coarse_z = it.z >> 3u;
  let bit_index = ((coarse_z << 4u) | (coarse_y << 2u) | coarse_x);  // 0..63? 不 4*4*4=64
  // 等等 4x4x4=64 bits? 但 in_cell_idx 一般 = (z/2)*(16*16)+(y/2)*16+(x/2) 在 tile
  // 让我们严格按 view.rs L71 bitmap_index_of 的实现：
  // view.rs L71-72: let i = cell.in_cell_idx(); let bit = 1u32 << (i & 31); b_struct[addr + (i >> 5)] & bit
  // cell.in_cell_idx() = z*1024 + y*32 + x（32³ cell index, 0-based in tile）
  let cell_in_tile = it.z * 1024u + it.y * 32u + it.x;  // 0..32767（z-major，与 coords.rs cell_index 一致）
  let bitmap_addr = BITMAP_BASE + slot * TILE_BITMAP_WORDS + (cell_in_tile >> 5u);
  let bitmap_word = b_struct[bitmap_addr];
  let bitmap_bit = 1u << (cell_in_tile & 31u);
  if ((bitmap_word & bitmap_bit) == 0u) { return 0u; }

  // ---- ③ CellDirs：node 绝对字偏移（view.rs L77-84） ----
  let dir_addr = DIR_BASE + slot * CELL_DIR_WORDS + cell_in_tile;
  let node_abs = b_struct[dir_addr];  // 绝对 word 偏移（相对于 b_struct[0]）
  if (node_abs == 0u) { return 0u; }

  // ---- ④ Walk Node：hdr + L1→L2→L3→BRICK（view.rs L86-147 walk_node） ----
  // Rust 侧 sub 是「cell 内分量坐标」IVec3（0..15，各分量独立），不是线性索引！
  // slot_at(sub: IVec3, axis) = sub.x + sub.y*axis + sub.z*axis²（x 最低位）
  //   L1: slot_at(sub >> 3, 2) → 2³=8 槽（ST_L1_WORDS=4 words × 2 slot/word）
  //   L2: slot_at(sub >> 2, 4) → 4³=64 槽（32 words）
  //   L3: slot_at(sub >> 1, 8) → 8³=512 槽（256 words）
  //   L4: slot_at(sub, 16)     → 16³=4096 byte brick（1024 words/slab）
  var p: u32 = node_abs;
  let hdr = b_struct[p];
  p = p + 1u;
  // L0 uniform: hdr&HDR_UNIFORM_MASK != 0 → palette
  let uniform = hdr & HDR_UNIFORM_MASK;
  if (uniform != 0u) {
    return uniform;
  }
  // L1 slot table（view.rs L88-101）
  if ((hdr & HDR_HAS_L1) == 0u) { return 0u; }

  let sub_u = vec3<u32>(in_cell);  // 0..15

  // L1: slot_at(sub >> 3, 2) = s.x + s.y*2 + s.z*4
  let s1 = sub_u >> vec3<u32>(3u);  // 各分量 0..1
  var idx = s1.x + s1.y * 2u + s1.z * 4u;
  let w_addr0 = p + (idx >> 1u);
  let w0 = b_struct[w_addr0];
  var slot_u = slot_unpack(w0, idx & 1u);
  var tag = slot_tag(slot_u);
  var pal = slot_palette(slot_u);
  if (tag == TAG_EMPTY) { return 0u; }
  if (tag == TAG_LEAF) { return pal; }
  // BRANCH → 加 L1 words
  p = p + ST_L1_WORDS;

  // L2: slot_at(sub >> 2, 4) = s.x + s.y*4 + s.z*16
  let s2 = sub_u >> vec3<u32>(2u);  // 0..3
  idx = s2.x + s2.y * 4u + s2.z * 16u;
  if ((hdr & HDR_HAS_L2) == 0u) { return 0u; }
  let w_addr1 = p + (idx >> 1u);
  let w1 = b_struct[w_addr1];
  slot_u = slot_unpack(w1, idx & 1u);
  tag = slot_tag(slot_u);
  pal = slot_palette(slot_u);
  if (tag == TAG_EMPTY) { return 0u; }
  if (tag == TAG_LEAF) { return pal; }
  p = p + ST_L2_WORDS;

  // L3: slot_at(sub >> 1, 8) = s.x + s.y*8 + s.z*64
  let s3 = sub_u >> vec3<u32>(1u);  // 0..7
  idx = s3.x + s3.y * 8u + s3.z * 64u;
  if ((hdr & HDR_HAS_L3) == 0u) { return 0u; }
  let w_addr2 = p + (idx >> 1u);
  let w2 = b_struct[w_addr2];
  slot_u = slot_unpack(w2, idx & 1u);
  tag = slot_tag(slot_u);
  pal = slot_palette(slot_u);
  if (tag == TAG_EMPTY) { return 0u; }
  if (tag == TAG_LEAF) { return pal; }
  p = p + ST_L3_WORDS;

  // L4 BRICK: view.rs L136-147
  // slab = b_struct[p] - 1；idx = slot_at(sub, 16) = s.x + s.y*16 + s.z*256
  // brick 存 bytes：word = b_leaves[slab*BRICK_SLAB_WORDS + (idx/4)]，pal = (word >> (idx%4)*8) & 0xFF
  if ((hdr & HDR_HAS_BRICK) == 0u) { return 0u; }
  let slab_minus_1 = b_struct[p];
  if (slab_minus_1 == 0u) { return 0u; }
  let slab = slab_minus_1 - 1u;
  let idx_l4 = sub_u.x + sub_u.y * 16u + sub_u.z * 256u;  // 0..4095
  let leaf_addr = slab * BRICK_SLAB_WORDS + (idx_l4 >> 2u);
  if (leaf_addr >= arrayLength(&b_leaves)) { return 0u; }
  let leaf_word = b_leaves[leaf_addr];
  let brick_pal = (leaf_word >> ((idx_l4 & 3u) << 3u)) & 0xFFu;
  return brick_pal;
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

// cell 级占用查询（寻址链 ①TileIndex + ②TileBitmaps；两级 DDA 粗步专用）。
// Rust BrickMapView::cell_occupied 的逐字翻译。cc 为 cell 坐标（1 单位 = 16 fine）。
// 语义：false ⇒ 该 cell 内全部 16³ fine 位置 sample 均为空。成本 2 次 load（全链 4~10 次）。
fn cell_occupied(cc: vec3<i32>) -> bool {
  let tile_i = cc >> vec3<u32>(5u);       // 32 cell / tile（i32 算术右移，负坐标正确；移位量须 u32）
  let it = vec3<u32>(cc & vec3<i32>(31)); // in-tile cell（欧氏余数；转 u32 供位运算）
  let origin = vec3<i32>(g.index_origin_x, g.index_origin_y, g.index_origin_z);
  let dims = vec3<u32>(g.index_dims_x, g.index_dims_y, g.index_dims_z);
  let rel = tile_i - origin;
  if (any(rel < vec3<i32>(0))) { return false; }
  let rel_u = vec3<u32>(rel);
  if (any(rel_u >= dims)) { return false; }
  let index_addr = rel_u.x + rel_u.y * TILE_INDEX_CAP + rel_u.z * (TILE_INDEX_CAP * TILE_INDEX_CAP);
  let slot_idx = b_struct[index_addr];    // 0 = empty tile
  if (slot_idx == 0u) { return false; }
  let slot = slot_idx - 1u;
  let cell_in_tile = it.z * 1024u + it.y * 32u + it.x;  // z-major，与 sample_brickmap 一致
  let bitmap_word = b_struct[BITMAP_BASE + slot * TILE_BITMAP_WORDS + (cell_in_tile >> 5u)];
  return ((bitmap_word >> (cell_in_tile & 31u)) & 1u) != 0u;
}

// ============================================================================
// P2.10 MOV：trace_scene() = 世界两级 DDA + 逐物体 trace_object，取最近命中
// 以下函数严格对应 Rust `brickmap/mov.rs` CPU 参考实现（逐字镜像）。
// ============================================================================

// slab 法射线-AABB 求交，返回 (t_enter, t_exit)；平行且在外 → (1.0, 0.0) miss 哨兵
// 对应 mov.rs::slab_box
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

// 物体 cell（0..31³）占用查询：bitmap 1 load。对应 mov.rs::obj_cell_occupied
fn mov_cell_occupied(bmp_base: u32, cc: vec3<i32>) -> bool {
  let it = vec3<u32>(cc);  // cc ∈ 0..31（入口保证非负）
  let ci = it.z * 1024u + it.y * 32u + it.x;
  let word = mov_struct[bmp_base + (ci >> 5u)];
  return ((word >> (ci & 31u)) & 1u) != 0u;
}

// 物体局部最细格采样（局部 fine 0..511³）。对应 mov.rs::obj_sample_voxel
// 逐字镜像 view.rs get_voxel ④ 链：基址换 descriptor（node_base + (abs - NODE_STREAM_BASE)、
// slab - 1 + leaves_base）
fn mov_sample_voxel(bmp_base: u32, dir_b: u32, node_b: u32, leaves_b: u32,
                    fine: vec3<i32>) -> u32 {
  let m = clamp(fine, vec3<i32>(0), vec3<i32>(511));
  let it = vec3<u32>(m);
  let ci = (it.z >> 4u) * 1024u + (it.y >> 4u) * 32u + (it.x >> 4u);
  let bmp = mov_struct[bmp_base + (ci >> 5u)];
  if (((bmp >> (ci & 31u)) & 1u) == 0u) { return 0u; }
  let abs_dir = mov_struct[dir_b + ci];
  if (abs_dir == 0u) { return 0u; }
  var p = node_b + (abs_dir - NODE_STREAM_BASE);
  let hdr = mov_struct[p];
  if ((hdr & HDR_UNIFORM_MASK) != 0u) { return hdr & HDR_UNIFORM_MASK; }
  p = p + 1u;
  let sub = it & vec3<u32>(15u);
  // L1（非 uniform 必有 l1）
  if ((hdr & HDR_HAS_L1) == 0u) { return 0u; }
  var si = (sub.x >> 3u) + (sub.y >> 3u) * 2u + (sub.z >> 3u) * 4u;
  var slot = slot_unpack(mov_struct[p + (si >> 1u)], si & 1u);
  var tag = slot_tag(slot);
  if (tag == TAG_EMPTY) { return 0u; }
  if (tag == TAG_LEAF) { return slot_palette(slot); }
  p = p + ST_L1_WORDS;
  // L2
  if ((hdr & HDR_HAS_L2) == 0u) { return 0u; }
  si = (sub.x >> 2u) + (sub.y >> 2u) * 4u + (sub.z >> 2u) * 16u;
  slot = slot_unpack(mov_struct[p + (si >> 1u)], si & 1u);
  tag = slot_tag(slot);
  if (tag == TAG_EMPTY) { return 0u; }
  if (tag == TAG_LEAF) { return slot_palette(slot); }
  p = p + ST_L2_WORDS;
  // L3
  if ((hdr & HDR_HAS_L3) == 0u) { return 0u; }
  si = (sub.x >> 1u) + (sub.y >> 1u) * 8u + (sub.z >> 1u) * 64u;
  slot = slot_unpack(mov_struct[p + (si >> 1u)], si & 1u);
  tag = slot_tag(slot);
  if (tag == TAG_EMPTY) { return 0u; }
  if (tag == TAG_LEAF) { return slot_palette(slot); }
  p = p + ST_L3_WORDS;
  // BRICK（slab 号 + leaves_base = pool slab 号）
  if ((hdr & HDR_HAS_BRICK) == 0u) { return 0u; }
  let slab_m1 = mov_struct[p];
  if (slab_m1 == 0u) { return 0u; }
  let slab = slab_m1 - 1u + leaves_b;
  let li = sub.x + sub.y * 16u + sub.z * 256u;
  let leaf_addr = slab * BRICK_SLAB_WORDS + (li >> 2u);
  if (leaf_addr >= arrayLength(&mov_leaves)) { return 0u; }
  let word = mov_leaves[leaf_addr];
  return (word >> ((li & 3u) << 3u)) & 0xFFu;
}

// 物体局部细扫：单粗 cell（16³ fine）内有界 fine DDA。对应 mov.rs::obj_fine_scan_cell
// 返回 (hit, t_rel_hit, pal)；t 为「相对 start」标尺（调用方 + tl0 还原全局）
struct FineHit {
  hit: bool,
  t: f32,
  pal: u32,
}
fn mov_fine_scan_cell(bmp_base: u32, dir_b: u32, node_b: u32, leaves_b: u32,
                      ro: vec3<f32>, rd: vec3<f32>, sign: vec3<i32>, delta: vec3<f32>,
                      cc: vec3<i32>, t_lo: f32, t_hi: f32) -> FineHit {
  if (t_hi <= t_lo) { return FineHit(false, 0.0, 0u); }
  let p = ro + rd * t_lo;
  let base = cc << vec3<u32>(4u);   // cell 起点 fine 坐标（移位量显式 u32 向量）
  // 入口 fine 胞 clamp 进本 cell（入口面浮点误差防护，同世界版）
  let fc0 = clamp(vec3<i32>(floor(p)), base, base + vec3<i32>(15));
  var fc = fc0;
  // fine tmax：相对 t_lo 的距离（自 cell 入口重算，非累加）
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
  // 初始 fine 胞采样
  let pal0 = mov_sample_voxel(bmp_base, dir_b, node_b, leaves_b, fc0);
  if (pal0 != 0u) { return FineHit(true, t_lo, pal0); }
  // 48 = 3 轴 × 16：斜穿 16³ cell 的步数上界
  for (var i_f: u32 = 0u; i_f < 48u; i_f = i_f + 1u) {
    if (t_f >= span) { break; }
    if (tmax_f.x <= tmax_f.y && tmax_f.x <= tmax_f.z) {
      t_f = tmax_f.x;
      tmax_f.x = tmax_f.x + delta.x;
      fc.x = fc.x + sign.x;
    } else if (tmax_f.y <= tmax_f.z) {
      t_f = tmax_f.y;
      tmax_f.y = tmax_f.y + delta.y;
      fc.y = fc.y + sign.y;
    } else {
      t_f = tmax_f.z;
      tmax_f.z = tmax_f.z + delta.z;
      fc.z = fc.z + sign.z;
    }
    let pal = mov_sample_voxel(bmp_base, dir_b, node_b, leaves_b, fc);
    if (pal != 0u) { return FineHit(true, t_lo + t_f, pal); }
  }
  return FineHit(false, 0.0, 0u);
}

// 单物体两级 DDA。对应 mov.rs::cpu_reference_object_ray：
// 世界 AABB 预剔除 → 局部变换（rd 不归一化，t 标尺不变）→ 局部 tile 盒 slab →
// cell 粗步（上限 96 = 3×32，cell 0..31³）+ fine 细步。返回 t 为全局标尺。
struct ObjHit {
  hit: bool,
  t: f32,
  pal: u32,
}
fn trace_object(idx: u32, origin: vec3<f32>, dir: vec3<f32>, t_cap: f32) -> ObjHit {
  // ---- descriptor 解码（32 words，f32 字段 bitcast）----
  let base = idx * MOV_DESC_WORDS;
  let obj_pos = vec3<f32>(
    bitcast<f32>(mov_descs[base]),
    bitcast<f32>(mov_descs[base + 1u]),
    bitcast<f32>(mov_descs[base + 2u]));
  let scale = bitcast<f32>(mov_descs[base + 3u]);
  // 旋转列（world = pos + R·(local·scale)）
  let col0 = vec3<f32>(
    bitcast<f32>(mov_descs[base + 4u]),
    bitcast<f32>(mov_descs[base + 5u]),
    bitcast<f32>(mov_descs[base + 6u]));
  let col1 = vec3<f32>(
    bitcast<f32>(mov_descs[base + 8u]),
    bitcast<f32>(mov_descs[base + 9u]),
    bitcast<f32>(mov_descs[base + 10u]));
  let col2 = vec3<f32>(
    bitcast<f32>(mov_descs[base + 12u]),
    bitcast<f32>(mov_descs[base + 13u]),
    bitcast<f32>(mov_descs[base + 14u]));
  let w_mn = vec3<f32>(
    bitcast<f32>(mov_descs[base + 16u]),
    bitcast<f32>(mov_descs[base + 17u]),
    bitcast<f32>(mov_descs[base + 18u]));
  let w_mx = vec3<f32>(
    bitcast<f32>(mov_descs[base + 20u]),
    bitcast<f32>(mov_descs[base + 21u]),
    bitcast<f32>(mov_descs[base + 22u]));
  let bmp_base = mov_descs[base + 24u];
  let dir_b = mov_descs[base + 25u];
  let node_b = mov_descs[base + 26u];
  let leaves_b = mov_descs[base + 27u];

  // ---- 世界 AABB 预剔除（几乎零成本）----
  let bx = slab_box(origin, dir, w_mn, w_mx, 0.0, t_cap);
  if (bx.y < max(bx.x, 0.0) || bx.x >= t_cap) { return ObjHit(false, 0.0, 0u); }
  let t_hi_cap = min(bx.y, t_cap);
  if (t_hi_cap <= max(bx.x, 0.0)) { return ObjHit(false, 0.0, 0u); }

  // ---- 局部变换：local = ((w - pos)·col_i) / scale（rd 不归一化）----
  let wp = origin - obj_pos;
  let ro = vec3<f32>(dot(wp, col0), dot(wp, col1), dot(wp, col2)) / scale;
  let rd = vec3<f32>(dot(dir, col0), dot(dir, col1), dot(dir, col2)) / scale;

  // ---- 局部 tile 盒 [0,512]³ slab ----
  let tl = slab_box(ro, rd, vec3<f32>(0.0), vec3<f32>(f32(LOCAL_TILE_FINE)), 0.0, t_hi_cap);
  if (tl.y < max(tl.x, 0.0)) { return ObjHit(false, 0.0, 0u); }
  let tl0 = max(tl.x, 0.0);
  let tl1 = min(tl.y, t_hi_cap);
  if (tl1 <= tl0) { return ObjHit(false, 0.0, 0u); }

  // ---- 两级 A&W（cell 0..31³）；全程「相对 start 的 t」标尺，返回时 +tl0 ----
  var sign_v = vec3<i32>(1i);
  sign_v = select(sign_v, vec3<i32>(-1i), rd < vec3<f32>(0.0));
  var delta = vec3<f32>(1e+30);
  delta = select(delta, 1.0 / abs(rd), abs(rd) > vec3<f32>(1e-30));
  let delta_c = delta * 16.0;
  let start = ro + rd * tl0;
  // cc = clamp(floor(start) >> 4, 0, 31)
  var cc = clamp(vec3<i32>(floor(start)) >> vec3<u32>(4u), vec3<i32>(0), vec3<i32>(31));
  // 粗 tmax：下一粗边界距离（相对 start）
  var tmax_c = vec3<f32>(1e+30);
  if (abs(rd.x) > 1e-30) {
    let t = (next_coarse_boundary(cc.x, sign_v.x) - start.x) / rd.x;
    tmax_c.x = max(t, 0.0);
  }
  if (abs(rd.y) > 1e-30) {
    let t = (next_coarse_boundary(cc.y, sign_v.y) - start.y) / rd.y;
    tmax_c.y = max(t, 0.0);
  }
  if (abs(rd.z) > 1e-30) {
    let t = (next_coarse_boundary(cc.z, sign_v.z) - start.z) / rd.z;
    tmax_c.z = max(t, 0.0);
  }
  let t_rel_max = tl1 - tl0;
  var t_in = 0.0;
  // 粗步上限 96 = 3×32：斜穿 32³ cell 的步数上界
  for (var step_c: u32 = 0u; step_c < 96u; step_c = step_c + 1u) {
    if (any(cc < vec3<i32>(0)) || any(cc > vec3<i32>(31))) { break; }
    let t_out = min(tmax_c.x, min(tmax_c.y, tmax_c.z));
    if (mov_cell_occupied(bmp_base, cc)) {
      let f = mov_fine_scan_cell(bmp_base, dir_b, node_b, leaves_b,
                                 start, rd, sign_v, delta, cc,
                                 t_in, min(t_out, t_rel_max));
      if (f.hit) { return ObjHit(true, tl0 + f.t, f.pal); }
    }
    if (t_out >= t_rel_max) { break; }
    // 粗级步进
    if (tmax_c.x <= tmax_c.y && tmax_c.x <= tmax_c.z) {
      t_in = tmax_c.x;
      tmax_c.x = tmax_c.x + delta_c.x;
      cc.x = cc.x + sign_v.x;
    } else if (tmax_c.y <= tmax_c.z) {
      t_in = tmax_c.y;
      tmax_c.y = tmax_c.y + delta_c.y;
      cc.y = cc.y + sign_v.y;
    } else {
      t_in = tmax_c.z;
      tmax_c.z = tmax_c.z + delta_c.z;
      cc.z = cc.z + sign_v.z;
    }
  }
  return ObjHit(false, 0.0, 0u);
}

// 物体 palette 解包（mov_palette 字基址 + pal_idx × 2 words）
fn mov_palette_rgb(pal_b: u32, pal_idx: u32) -> vec3<u32> {
  let w0 = mov_palette[pal_b + pal_idx * 2u];
  return vec3<u32>(w0 & 0xFFu, (w0 >> 8u) & 0xFFu, (w0 >> 16u) & 0xFFu);
}

// ============================================================================
// DDA 主入口：每个像素 = workgroup 内一个 invocation（8x8x1）
// ============================================================================
@compute @workgroup_size(8, 8, 1)
fn dda_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  // 画面尺寸：VIEW_SIZE (1280x720) 由 Rust Gradient 定义相同，这里不 hardcode，
  // 用 textureDimensions(out_tex) 拿精确尺寸。
  let size = textureDimensions(out_tex);
  if (gid.x >= size.x || gid.y >= size.y) { return; }
  let coord0 = vec2<i32>(i32(gid.x), i32(gid.y));
  // 背景色深蓝黑（命中时再覆盖）
  var col: vec3<f32> = vec3<f32>(0.05, 0.08, 0.12);

  // ---- 反投影：像素中心 (gid + 0.5) → NDC (u, v) ∈ [-1, 1] ----
  let px = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5)) / vec2<f32>(f32(size.x), f32(size.y));
  let uv = vec2<f32>(px.x * 2.0 - 1.0, 1.0 - px.y * 2.0);  // v Y 翻转与 Rust cpu_dda 一致
  let near_ndc = vec4<f32>(uv.x, uv.y, 0.0, 1.0);
  let far_ndc  = vec4<f32>(uv.x, uv.y, 1.0, 1.0);
  let near_world_h = view_u.inv_view_proj * near_ndc;
  let far_world_h  = view_u.inv_view_proj * far_ndc;
  let near_world = near_world_h.xyz / near_world_h.w;
  let far_world  = far_world_h.xyz  / far_world_h.w;
  let diff_world = far_world - near_world;
  // frustum_length = 视锥内这条射线 near→far 的 fine 距离（视锥外不可见，DDA 到这里就停）。
  // 随 CAM_FAR 变化（当前 65536 fine = 163.84m），用户取消"最远距离"后仍能看到整个场景。
  let frustum_length = length(diff_world);
  let dir_fine = normalize(diff_world);
  let origin_fine = view_u.cam_pos_fine.xyz;

  // ---- 1) slab 法求射线 vs brickmap tile-window AABB（fine 坐标）相交区间 ----
  // 与 Rust `cpu_reference_dda_ray_aabb_skip` 实现逐字对应，300 条随机射线已证明与直走 DDA 等价。
  // AABB = [index_origin..index_origin+index_dims] × 512 fine（每个 tile 512 fine）
  let tile_origin_fine = vec3<f32>(
    f32(g.index_origin_x) * 512.0,
    f32(g.index_origin_y) * 512.0,
    f32(g.index_origin_z) * 512.0,
  );
  let aabb_min = tile_origin_fine;
  let aabb_max = tile_origin_fine + vec3<f32>(
    f32(g.index_dims_x) * 512.0,
    f32(g.index_dims_y) * 512.0,
    f32(g.index_dims_z) * 512.0,
  );
  // t_enter / t_exit：自 origin_fine 量起的 fine 距离（全局标尺）
  var t_enter = 0.0;
  var t_exit  = frustum_length;
  var miss = false;
  // X 轴 slab
  {
    let d = dir_fine.x;
    if (abs(d) < 1e-30) {
      if (origin_fine.x < aabb_min.x || origin_fine.x > aabb_max.x) { miss = true; }
    } else {
      let t1 = (aabb_min.x - origin_fine.x) / d;
      let t2 = (aabb_max.x - origin_fine.x) / d;
      let lo = min(t1, t2);
      let hi = max(t1, t2);
      t_enter = max(t_enter, lo);
      t_exit  = min(t_exit,  hi);
    }
  }
  // Y 轴 slab
  {
    let d = dir_fine.y;
    if (abs(d) < 1e-30) {
      if (origin_fine.y < aabb_min.y || origin_fine.y > aabb_max.y) { miss = true; }
    } else {
      let t1 = (aabb_min.y - origin_fine.y) / d;
      let t2 = (aabb_max.y - origin_fine.y) / d;
      let lo = min(t1, t2);
      let hi = max(t1, t2);
      t_enter = max(t_enter, lo);
      t_exit  = min(t_exit,  hi);
    }
  }
  // Z 轴 slab
  {
    let d = dir_fine.z;
    if (abs(d) < 1e-30) {
      if (origin_fine.z < aabb_min.z || origin_fine.z > aabb_max.z) { miss = true; }
    } else {
      let t1 = (aabb_min.z - origin_fine.z) / d;
      let t2 = (aabb_max.z - origin_fine.z) / d;
      let lo = min(t1, t2);
      let hi = max(t1, t2);
      t_enter = max(t_enter, lo);
      t_exit  = min(t_exit,  hi);
    }
  }
  if (miss || t_exit < max(t_enter, 0.0)) {
    // 不相交 → 0 步 DDA，背景 return
    textureStore(out_tex, coord0, vec4<f32>(col, 1.0));
    return;
  }
  // 约束到视锥有效段 [0, frustum_length]
  let t_enter_clamped = max(t_enter, 0.0);
  let t_exit_clamped  = min(t_exit,  frustum_length);
  if (t_exit_clamped <= t_enter_clamped) {
    // 厚度为 0（擦边）或完全在视锥外
    textureStore(out_tex, coord0, vec4<f32>(col, 1.0));
    return;
  }

  // ---- 2) 起点推进到 start = origin + dir · t_enter_clamped（跳过前面空胞）
  // 之后全程用「相对标尺」：t_rel / tmax_* 都自 start 量起，和 Rust 单测 AABB-skip 版完全一致。
  let start_v = origin_fine + dir_fine * t_enter_clamped;
  let t_rel_max = t_exit_clamped - t_enter_clamped;

  // ---- 两级 Amanatides & Woo：粗级 cell（16 fine）步进 + 非空 cell 内 fine 细步 ----
  // Rust `cpu_reference_dda_ray_two_level` 的逐字翻译
  //（等价性单测 two_level_equivalence_300_rays 锁死：命中/palette 严格一致）。
  // 空气穿越成本：每 16 fine 一次 cell 占用查询（2 load），
  // 替代原单级版每 1 fine 一次全链 sample（4~10 load）——远距离射线 ~40× 访存降幅。
  var sign_x = 1i; var sign_y = 1i; var sign_z = 1i;
  if (dir_fine.x < 0.0) { sign_x = -1i; }
  if (dir_fine.y < 0.0) { sign_y = -1i; }
  if (dir_fine.z < 0.0) { sign_z = -1i; }
  var dx = 1e+30;
  if (abs(dir_fine.x) > 1e-30) { dx = 1.0 / abs(dir_fine.x); }
  var dy = 1e+30;
  if (abs(dir_fine.y) > 1e-30) { dy = 1.0 / abs(dir_fine.y); }
  var dz = 1e+30;
  if (abs(dir_fine.z) > 1e-30) { dz = 1.0 / abs(dir_fine.z); }
  // 粗 delta = fine delta × 16（f32 乘 2 的幂，精确）
  var dcx = dx * 16.0;
  var dcy = dy * 16.0;
  var dcz = dz * 16.0;

  // 粗 cell = floor(start_v) >> 4（算术右移 = floor 除法，负坐标正确）
  var cc_x = i32(floor(start_v.x)) >> 4;
  var cc_y = i32(floor(start_v.y)) >> 4;
  var cc_z = i32(floor(start_v.z)) >> 4;

  // 粗 tmax：下一个粗边界的距离（相对标尺，同 full 版公式，边界 ×16）
  var tmax_cx = 1e+30;
  if (abs(dir_fine.x) > 1e-30) {
    tmax_cx = (next_coarse_boundary(cc_x, sign_x) - start_v.x) / dir_fine.x;
    if (tmax_cx < 0.0) { tmax_cx = 0.0; }
  }
  var tmax_cy = 1e+30;
  if (abs(dir_fine.y) > 1e-30) {
    tmax_cy = (next_coarse_boundary(cc_y, sign_y) - start_v.y) / dir_fine.y;
    if (tmax_cy < 0.0) { tmax_cy = 0.0; }
  }
  var tmax_cz = 1e+30;
  if (abs(dir_fine.z) > 1e-30) {
    tmax_cz = (next_coarse_boundary(cc_z, sign_z) - start_v.z) / dir_fine.z;
    if (tmax_cz < 0.0) { tmax_cz = 0.0; }
  }

  var t_in = 0.0;                        // 当前粗 cell 入口 t（相对标尺，初始 cell = 0）
  var hit_pal: u32 = 0u;
  var hit = false;
  var world_hit_t = 1e+30;               // 世界命中的全局 t（trace_scene 最近比较用）

  // 粗级主循环：每步 = 占用查询 →（占用才）细级扫描 → 粗步进
  for (var step_c: u32 = 0u; step_c < 16384u; step_c = step_c + 1u) {
    let t_out = min(tmax_cx, min(tmax_cy, tmax_cz));
    // 占用查询先行：空 cell 整段跳过所有 fine 采样（性能核心）
    if (cell_occupied(vec3<i32>(cc_x, cc_y, cc_z))) {
      // ---- 细级：本 cell 区间 [t_in, min(t_out, t_rel_max)] 内有界 fine DDA ----
      let t_hi = min(t_out, t_rel_max);
      if (t_hi > t_in) {
        let p = start_v + dir_fine * t_in;
        let base_x = cc_x << 4;
        let base_y = cc_y << 4;
        let base_z = cc_z << 4;
        // 入口 fine 胞 = floor(p) clamp 进本 cell（入口面浮点误差防护；
        // clamp 掉的邻胞属前一粗 cell，其细扫已覆盖，不漏检）
        var fc_x = clamp(i32(floor(p.x)), base_x, base_x + 15);
        var fc_y = clamp(i32(floor(p.y)), base_y, base_y + 15);
        var fc_z = clamp(i32(floor(p.z)), base_z, base_z + 15);
        // fine tmax：相对 t_in 的距离（自 cell 入口重算，非累加）
        var tf_x = 1e+30;
        if (abs(dir_fine.x) > 1e-30) {
          tf_x = (next_boundary(fc_x, sign_x) - p.x) / dir_fine.x;
          if (tf_x < 0.0) { tf_x = 0.0; }
        }
        var tf_y = 1e+30;
        if (abs(dir_fine.y) > 1e-30) {
          tf_y = (next_boundary(fc_y, sign_y) - p.y) / dir_fine.y;
          if (tf_y < 0.0) { tf_y = 0.0; }
        }
        var tf_z = 1e+30;
        if (abs(dir_fine.z) > 1e-30) {
          tf_z = (next_boundary(fc_z, sign_z) - p.z) / dir_fine.z;
          if (tf_z < 0.0) { tf_z = 0.0; }
        }
        var t_f = 0.0;
        let span = t_hi - t_in;
        // 初始 fine 胞采样
        let pal0 = sample_brickmap(vec3<i32>(fc_x, fc_y, fc_z));
        if (pal0 != 0u) {
          hit_pal = pal0;
          hit = true;
          world_hit_t = t_enter_clamped + t_in;
        }
        // 48 = 3 轴 × 16：斜穿 16³ cell 的步数上界，几何上必在 span 内退出
        for (var i_f: u32 = 0u; !hit && i_f < 48u && t_f < span; i_f = i_f + 1u) {
          if (tf_x <= tf_y && tf_x <= tf_z) {
            t_f = tf_x;
            tf_x = tf_x + dx;
            fc_x = fc_x + sign_x;
          } else if (tf_y <= tf_z) {
            t_f = tf_y;
            tf_y = tf_y + dy;
            fc_y = fc_y + sign_y;
          } else {
            t_f = tf_z;
            tf_z = tf_z + dz;
            fc_z = fc_z + sign_z;
          }
          let pal = sample_brickmap(vec3<i32>(fc_x, fc_y, fc_z));
          if (pal != 0u) {
            hit_pal = pal;
            hit = true;
            world_hit_t = t_enter_clamped + t_in + t_f;
          }
        }
      }
    }
    if (hit || t_out >= t_rel_max) { break; }
    // ---- 粗级步进（整数增量，同 full 版约定）----
    if (tmax_cx <= tmax_cy && tmax_cx <= tmax_cz) {
      t_in = tmax_cx;
      tmax_cx = tmax_cx + dcx;
      cc_x = cc_x + sign_x;
    } else if (tmax_cy <= tmax_cz) {
      t_in = tmax_cy;
      tmax_cy = tmax_cy + dcy;
      cc_y = cc_y + sign_y;
    } else {
      t_in = tmax_cz;
      tmax_cz = tmax_cz + dcz;
      cc_z = cc_z + sign_z;
    }
  }

  // ---- trace_scene 合成（P2.10）：世界命中 → 逐物体（AABB 预剔除 + t_cap 剪枝）取最近 ----
  // 对应 mov.rs::cpu_reference_trace_scene；count=0 时循环零成本，行为与纯世界版一致。
  var best_t = world_hit_t;
  var best_pal = hit_pal;
  var best_obj: i32 = -1;  // -1 = 世界网格
  for (var i: u32 = 0u; i < mov_g.count; i = i + 1u) {
    // 已有更近命中时以之为剪枝上限（P3 阴影射线 / P9 GI 同一入口）
    let cap = min(best_t, frustum_length);
    let h = trace_object(i, origin_fine, dir_fine, cap);
    if (h.hit && h.t < best_t) {
      best_t = h.t;
      best_pal = h.pal;
      best_obj = i32(i);
    }
  }

  // ---- 颜色输出：命中 → palette sRGB 字节直接存（世界/物体各自 palette 区）；空 → 背景色 ----
  if (best_t < 1e+29) {
    var rgb_u8: vec3<u32>;
    if (best_obj < 0) {
      rgb_u8 = palette_rgb_u8(best_pal);
    } else {
      // descriptor word 28 = palette_base（mov_palette 内字基址）
      let pal_b = mov_descs[u32(best_obj) * MOV_DESC_WORDS + 28u];
      rgb_u8 = mov_palette_rgb(pal_b, best_pal);
    }
    col = vec3<f32>(rgb_u8) / 255.0;
  }
  textureStore(out_tex, coord0, vec4<f32>(col, 1.0));
}

// ---- entry point 命名与 Rust PipelineCache queue_compute 对应 ----
// Pipeline init（Task 5）会以 `entry_point="dda_main"` 创建 compute pipeline
