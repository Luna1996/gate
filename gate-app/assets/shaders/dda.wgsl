// ============================================================================
// DDA Compute Shader：全屏逐像素 A&W 步进 + brickmap 五步寻址
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
// 顶部常量与 Rust `brickmap::dda::wgsl_consts` 完全一致（单测 TR-2.1 assert_eq 防漂移）。
// BrickMap 五步寻址链严格对应 Rust `gate-render/src/brickmap/view.rs::get_voxel`（逐段注释 L 号）。
// Slot 打包规则对应 Rust `gate-render/src/brickmap/wire.rs::encode_slot/unpack_slot_word/pack_palette_entry`。
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

  // ---- Amanatides & Woo（相对标尺版） ----
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

  // start cell = floor(start_v)（相对起点所在的胞）
  let sc_x = i32(floor(start_v.x));
  let sc_y = i32(floor(start_v.y));
  let sc_z = i32(floor(start_v.z));

  // tmax_*：相对 start_v 的 fine 距离（相对标尺）
  var tmax_x = 1e+30;
  if (abs(dir_fine.x) > 1e-30) {
    tmax_x = (next_boundary(sc_x, sign_x) - start_v.x) / dir_fine.x;
    if (tmax_x < 0.0) { tmax_x = 0.0; }
  }
  var tmax_y = 1e+30;
  if (abs(dir_fine.y) > 1e-30) {
    tmax_y = (next_boundary(sc_y, sign_y) - start_v.y) / dir_fine.y;
    if (tmax_y < 0.0) { tmax_y = 0.0; }
  }
  var tmax_z = 1e+30;
  if (abs(dir_fine.z) > 1e-30) {
    tmax_z = (next_boundary(sc_z, sign_z) - start_v.z) / dir_fine.z;
    if (tmax_z < 0.0) { tmax_z = 0.0; }
  }

  var t_rel = 0.0;                       // 相对标尺：从 start_v 量起
  var cur_x = sc_x; var cur_y = sc_y; var cur_z = sc_z;
  var hit_pal: u32 = 0u;
  var hit = false;

  // 初始胞采样（start_v 对应的胞）
  let pal0 = sample_brickmap(vec3<i32>(cur_x, cur_y, cur_z));
  if (pal0 != 0u) {
    hit_pal = pal0;
    hit = true;
  }

  for (var i: u32 = 0u; i < 16384u; i = i + 1u) {
    if (hit || t_rel >= t_rel_max) { break; }
    if (tmax_x <= tmax_y && tmax_x <= tmax_z) {
      t_rel = tmax_x;
      tmax_x = tmax_x + dx;
      cur_x = cur_x + sign_x;
    } else if (tmax_y <= tmax_z) {
      t_rel = tmax_y;
      tmax_y = tmax_y + dy;
      cur_y = cur_y + sign_y;
    } else {
      t_rel = tmax_z;
      tmax_z = tmax_z + dz;
      cur_z = cur_z + sign_z;
    }
    let pal = sample_brickmap(vec3<i32>(cur_x, cur_y, cur_z));
    if (pal != 0u) {
      hit_pal = pal;
      hit = true;
    }
  }

  // ---- 颜色输出：命中 → palette sRGB 字节直接存；空 → 已初始化的背景色 ----
  if (hit) {
    let rgb_u8 = palette_rgb_u8(hit_pal);
    col = vec3<f32>(rgb_u8) / 255.0;
  }
  textureStore(out_tex, coord0, vec4<f32>(col, 1.0));
}

// ---- entry point 命名与 Rust PipelineCache queue_compute 对应 ----
// Pipeline init（Task 5）会以 `entry_point="dda_main"` 创建 compute pipeline
