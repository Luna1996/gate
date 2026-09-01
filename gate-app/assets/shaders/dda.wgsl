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

// 逐面光照注册表（P3.5d；Rust 镜像 gate-render/src/brickmap/face_light.rs，单测防漂移）
const FL_REG_SLOTS: u32 = 2097152u;       // 1<<21 槽（近景可见独立体素可逼近屏幕像素数 ~2M，余量防 probe 耗尽丢注册）
const FL_WORDS_PER_SLOT: u32 = 28u;
const FL_PROBE_MAX: u32 = 64u;
const FL_EPOCH_INVALID: u32 = 0xFFFFFFFFu; // 新占位槽的 epoch 值（作废残留，强制 fl_light 重算）
// slot 布局偏移：[0]status [1..3]xyz [4]obj_key [5]face_mask [6]epoch [7..24]light×6面 [25..27]MOV中心
const FL_OFF_OBJ: u32 = 4u;
const FL_OFF_MASK: u32 = 5u;
const FL_OFF_EPOCH: u32 = 6u;
const FL_OFF_LIGHT: u32 = 7u;
const FL_OFF_CENTER: u32 = 25u;

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

// --- BG0：G-buffer + 输出 + 视图 uniform（P3.5d 三段式管线共用）---
// @binding(0) = gbuffer storage read_write（rgba32uint: xyz 体素 + meta word）
//   dda_main 写入（textureStore），fl_composite_main 读取（textureLoad）
// @binding(1) = out storage write（rgba8unorm，合成 pass 最终输出；debug 模式主 pass 直写）
// @binding(2) = uniform DdaViewUniform
@group(0) @binding(0) var gbuf_tex: texture_storage_2d<rgba32uint, read_write>;
@group(0) @binding(1) var out_tex: texture_storage_2d<rgba8unorm, write>;

struct DdaViewUniform {
  inv_view_proj: mat4x4<f32>,  // 64B
  cam_pos_fine: vec4<f32>,     // 16B，w=1
  debug_mode: vec4<f32>,       // 16B：x = 1.0 → 法向向量可视化
  fl_epoch: vec4<f32>,         // 16B：x = 逐面光照 epoch（u32 as f32，2^24 内精确）
}
@group(0) @binding(2) var<uniform> view_u: DdaViewUniform;

// G-buffer meta word 打包
//   bits 0-7   pal
//   bits 8-15  objm（0xFF = 世界；否则 MOV 物体下标）
//   bits 16-18 face（0=-X 1=+X 2=-Y 3=+Y 4=-Z 5=+Z；MOV 为局部 face）
//   bit  19    is_sky
const GB_SKY_BIT: u32 = 1u << 19u;
const GB_OBJM_WORLD: u32 = 0xFFu;

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

// ---- BG4：逐面光照注册表（P3.5d；read_write，f32 字段 bitcast 存取）----
@group(4) @binding(0) var<storage, read_write> fl_table: array<atomic<u32>>;

// ============================================================================
// P3.5d 逐面光照：hash 注册表操作 + 面工具 + 面光照数学
// ============================================================================

// 整数混合（Rust face_light.rs fl_hash_u32 镜像）
fn fl_hash_u32(v: u32) -> u32 {
  var x = v;
  x = x ^ (x >> 16u);
  x = x * 0x7FEB352Du;
  x = x ^ (x >> 15u);
  x = x * 0x846CA68Bu;
  x = x ^ (x >> 16u);
  return x;
}
// key = (x, y, z, obj_key)（obj_key：0=世界，N+1=MOV N）
fn fl_hash(x: i32, y: i32, z: i32, obj_key: u32) -> u32 {
  var h = 0x811C9DC5u;
  h = fl_hash_u32(h ^ bitcast<u32>(x));
  h = fl_hash_u32(h ^ bitcast<u32>(y));
  h = fl_hash_u32(h ^ bitcast<u32>(z));
  h = fl_hash_u32(h ^ obj_key);
  return h;
}

// face f 法线（世界物体 = 世界轴向；MOV 光照时经旋转矩阵变换）
fn fl_face_normal(f: u32) -> vec3<f32> {
  if (f == 0u) { return vec3<f32>(-1.0, 0.0, 0.0); }
  if (f == 1u) { return vec3<f32>(1.0, 0.0, 0.0); }
  if (f == 2u) { return vec3<f32>(0.0, -1.0, 0.0); }
  if (f == 3u) { return vec3<f32>(0.0, 1.0, 0.0); }
  if (f == 4u) { return vec3<f32>(0.0, 0.0, -1.0); }
  return vec3<f32>(0.0, 0.0, 1.0);
}

// 命中面法线（±轴单位向量）→ face index
fn fl_face_index(n: vec3<f32>) -> u32 {
  if (n.x < -0.5) { return 0u; }
  if (n.x > 0.5) { return 1u; }
  if (n.y < -0.5) { return 2u; }
  if (n.y > 0.5) { return 3u; }
  if (n.z < -0.5) { return 4u; }
  return 5u;
}

// 可见面注册（dda_main 命中后调用）：开放寻址 + CAS 占位。
// 并发协议：CAS(0→1) 成功者独占槽位并写 key；后来者读 key 比对——
// 同 key → atomicOr face_mask；异 key → 下一槽。占位中（key 未写）被读到
// 旧尸体 key 时误判走下一槽，最坏产生重复同 key 槽（lookup 侧按 mask 位
// 过滤继续 probe，见 fl_lookup 注）。
//
// epoch 置 INVALID（v3.9.1 修复）：clear 保留 epoch/light 残留，但开放寻址的
// 槽位随注册竞争顺序漂移（同 key 每帧可能落不同槽）→ 槽内 epoch/light 可能是
// 其他体素的残留 → 「epoch 匹配跳过」会读到脏数据（随机亮暗噪点+闪烁）或漏算
// 新见面（黑块）。置 INVALID 强制 fl_light 恒重算，时间复用待槽位稳定化后恢复。
fn fl_register(voxel: vec3<i32>, obj_key: u32, face: u32, world_center: vec3<f32>) {
  let h = fl_hash(voxel.x, voxel.y, voxel.z, obj_key) % FL_REG_SLOTS;
  for (var probe: u32 = 0u; probe < FL_PROBE_MAX; probe = probe + 1u) {
    let idx = (h + probe) % FL_REG_SLOTS;
    let base = idx * FL_WORDS_PER_SLOT;
    let cas = atomicCompareExchangeWeak(&fl_table[base], 0u, 1u);
    if (cas.exchanged) {
      // 独占槽位：作废残留 epoch → 写 key + face
      atomicStore(&fl_table[base + FL_OFF_EPOCH], FL_EPOCH_INVALID);
      atomicStore(&fl_table[base + 1u], bitcast<u32>(voxel.x));
      atomicStore(&fl_table[base + 2u], bitcast<u32>(voxel.y));
      atomicStore(&fl_table[base + 3u], bitcast<u32>(voxel.z));
      atomicStore(&fl_table[base + FL_OFF_OBJ], obj_key);
      atomicOr(&fl_table[base + FL_OFF_MASK], 1u << face);
      if (obj_key != 0u) {
        // MOV：存世界体素中心（旋转/缩放后面中心 ≠ voxel+0.5）
        atomicStore(&fl_table[base + FL_OFF_CENTER], bitcast<u32>(world_center.x));
        atomicStore(&fl_table[base + FL_OFF_CENTER + 1u], bitcast<u32>(world_center.y));
        atomicStore(&fl_table[base + FL_OFF_CENTER + 2u], bitcast<u32>(world_center.z));
      }
      return;
    }
    // 已占用：比对 key（同 key 聚合 face）
    let kx = bitcast<i32>(atomicLoad(&fl_table[base + 1u]));
    let ky = bitcast<i32>(atomicLoad(&fl_table[base + 2u]));
    let kz = bitcast<i32>(atomicLoad(&fl_table[base + 3u]));
    if (kx == voxel.x && ky == voxel.y && kz == voxel.z
        && atomicLoad(&fl_table[base + FL_OFF_OBJ]) == obj_key) {
      atomicOr(&fl_table[base + FL_OFF_MASK], 1u << face);
      return;
    }
  }
  // probe 耗尽：丢弃（合成 pass 查不到 → 默认光照兜底）
}

// 查表取面光照（fl_composite_main 专用；未注册 = 中性灰兜底）。
// mask 位过滤（v3.9.1 修复）：注册并发写 key 的撕裂窗口可产生同 key 重复槽
// （face 分裂在两个槽）——命中匹配 key 但该槽 mask 不含 face 时继续 probe，
// 找含 face 的槽（各槽光照数学确定性一致，任一含 face 槽的值皆正确）。
fn fl_lookup(voxel: vec3<i32>, obj_key: u32, face: u32) -> vec3<f32> {
  let h = fl_hash(voxel.x, voxel.y, voxel.z, obj_key) % FL_REG_SLOTS;
  for (var probe: u32 = 0u; probe < FL_PROBE_MAX; probe = probe + 1u) {
    let idx = (h + probe) % FL_REG_SLOTS;
    let base = idx * FL_WORDS_PER_SLOT;
    if (atomicLoad(&fl_table[base]) == 0u) { break; }
    let kx = bitcast<i32>(atomicLoad(&fl_table[base + 1u]));
    let ky = bitcast<i32>(atomicLoad(&fl_table[base + 2u]));
    let kz = bitcast<i32>(atomicLoad(&fl_table[base + 3u]));
    if (kx == voxel.x && ky == voxel.y && kz == voxel.z
        && atomicLoad(&fl_table[base + FL_OFF_OBJ]) == obj_key) {
      let mask = atomicLoad(&fl_table[base + FL_OFF_MASK]);
      if ((mask & (1u << face)) != 0u) {
        let w = FL_OFF_LIGHT + face * 3u;
        return vec3<f32>(
          bitcast<f32>(atomicLoad(&fl_table[base + w])),
          bitcast<f32>(atomicLoad(&fl_table[base + w + 1u])),
          bitcast<f32>(atomicLoad(&fl_table[base + w + 2u])));
      }
      // 槽 mask 不含 face：撕裂重复槽，继续 probe
    }
  }
  return vec3<f32>(0.3);
}

// 面光照数学（原 shade_hit 环境部分逐面化；pre-exposure HDR；无高光/软阴影）
// n = 面法线（世界空间），v = 面中心（阴影射线起点基准）
fn face_light_math(n: vec3<f32>, v: vec3<f32>) -> vec3<f32> {
  // sky 渐变环境光（按面法线 y）
  let h = clamp(n.y, 0.0, 1.0);
  let sky_grad = mix(light_u.sky_horizon.xyz, light_u.sky_top.xyz, smoothstep(0.0, 0.35, h));
  var light = light_u.g.ambient.xyz * 0.4 + sky_grad * 0.6;

  // 方向光硬阴影（每面 1 条阴影射线；3.5d 软阴影半影后续换多样本平均）
  if (light_u.g.count > 0u) {
    let ld = light_u.lights[0];
    if (ld.kind_pos_dir.x < 0.5) {
      let l_axis = ld.kind_pos_dir.yzw;
      let ndl = max(dot(n, l_axis), 0.0);
      if (ndl > 0.0) {
        let o = v + n * SHADOW_BIAS;
        let vis = select(0.0, 1.0, !scene_occluded(o, l_axis, SHADOW_DIR_T_MAX));
        light = light + ld.color_intensity.xyz * ld.color_intensity.w * (ndl * vis);
      }
    }
  }
  return light;
}

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
  let tile_i = vec3<i32>(cc >> vec3<u32>(5u));       // 32 cell / tile（i32 算术右移，负坐标正确；移位量须 u32）
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
// 返回 (hit, t_rel_hit, pal, axis)；t 为「相对 start」标尺（调用方 + tl0 还原全局）；
// axis = 命中轴 0/1/2（法线用），3 = 起点即在体内
struct FineHit {
  hit: bool,
  t: f32,
  pal: u32,
  axis: u32,
}
fn mov_fine_scan_cell(bmp_base: u32, dir_b: u32, node_b: u32, leaves_b: u32,
                      ro: vec3<f32>, rd: vec3<f32>, sign: vec3<i32>, delta: vec3<f32>,
                      cc: vec3<i32>, t_lo: f32, t_hi: f32) -> FineHit {
  // 诊断开关：0 = 完整；1 = 初始胞采样 only（跳过 A&W）；2 = 全 skip（return miss）
  const MOV_FINE_DIAG: u32 = 0u;
  if (MOV_FINE_DIAG == 2u) { return FineHit(false, 0.0, 0u, 3u); }
  if (t_hi <= t_lo) { return FineHit(false, 0.0, 0u, 3u); }
  let p = ro + rd * t_lo;
  let base = vec3<i32>(cc << vec3<u32>(4u));
  let fc0 = clamp(vec3<i32>(floor(p)), base, base + vec3<i32>(15));
  // 初始 fine 胞采样（两种模式都跑）
  let pal0_init = mov_sample_voxel(bmp_base, dir_b, node_b, leaves_b, fc0);
  if (pal0_init != 0u) { return FineHit(true, t_lo, pal0_init, 3u); }
  if (MOV_FINE_DIAG == 1u) { return FineHit(false, 0.0, 0u, 3u); }
  // ---- 完整 A&W 细步循环（MOV_FINE_DIAG == 0）----
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
    var axis = 2u;
    if (tmax_f.x <= tmax_f.y && tmax_f.x <= tmax_f.z) {
      t_f = tmax_f.x;
      tmax_f.x = tmax_f.x + delta.x;
      fc.x = fc.x + sign.x;
      axis = 0u;
    } else if (tmax_f.y <= tmax_f.z) {
      t_f = tmax_f.y;
      tmax_f.y = tmax_f.y + delta.y;
      fc.y = fc.y + sign.y;
      axis = 1u;
    } else {
      t_f = tmax_f.z;
      tmax_f.z = tmax_f.z + delta.z;
      fc.z = fc.z + sign.z;
    }
    let pal = mov_sample_voxel(bmp_base, dir_b, node_b, leaves_b, fc);
    if (pal != 0u) { return FineHit(true, t_lo + t_f, pal, axis); }
  }
  return FineHit(false, 0.0, 0u, 3u);
}

// 单物体两级 DDA。对应 mov.rs::cpu_reference_object_ray：
// 世界 AABB 预剔除 → 局部变换（rd 不归一化，t 标尺不变）→ 局部 tile 盒 slab →
// cell 粗步（上限 96 = 3×32，cell 0..31³）+ fine 细步。返回 t 为全局标尺。
// n = 命中面法线（世界空间单位向量，指向射线来向；起点在体内时 = -dir 局部方向）
struct ObjHit {
  hit: bool,
  t: f32,
  pal: u32,
  n: vec3<f32>,
}
fn trace_object(idx: u32, origin: vec3<f32>, dir: vec3<f32>, t_cap: f32) -> ObjHit {
  // ---- descriptor 解码（32 words，f32 字段 bitcast）----
  let base = idx * MOV_DESC_WORDS;
  let obj_pos = vec3<f32>(
    f32(mov_descs[base]),
    f32(mov_descs[base + 1u]),
    f32(mov_descs[base + 2u]));
  let scale = f32(mov_descs[base + 3u]);
  let col0 = vec3<f32>(
    f32(mov_descs[base + 4u]),
    f32(mov_descs[base + 5u]),
    f32(mov_descs[base + 6u]));
  let col1 = vec3<f32>(
    f32(mov_descs[base + 8u]),
    f32(mov_descs[base + 9u]),
    f32(mov_descs[base + 10u]));
  let col2 = vec3<f32>(
    f32(mov_descs[base + 12u]),
    f32(mov_descs[base + 13u]),
    f32(mov_descs[base + 14u]));
  let w_mn = vec3<f32>(
    f32(mov_descs[base + 16u]),
    f32(mov_descs[base + 17u]),
    f32(mov_descs[base + 18u]));
  let w_mx = vec3<f32>(
    f32(mov_descs[base + 20u]),
    f32(mov_descs[base + 21u]),
    f32(mov_descs[base + 22u]));
  let bmp_base = mov_descs[base + 24u];
  let dir_b = mov_descs[base + 25u];
  let node_b = mov_descs[base + 26u];
  let leaves_b = mov_descs[base + 27u];

  // ---- 世界 AABB 预剔除（几乎零成本）----
  let bx = slab_box(origin, dir, w_mn, w_mx, 0.0, t_cap);
  if (bx.y < max(bx.x, 0.0) || bx.x >= t_cap) { return ObjHit(false, 0.0, 0u, vec3<f32>(0.0)); }
  let t_hi_cap = min(bx.y, t_cap);
  if (t_hi_cap <= max(bx.x, 0.0)) { return ObjHit(false, 0.0, 0u, vec3<f32>(0.0)); }
  // 诊断：slab_box 后直接 return miss（跳过局部变换 + A&W）
  // return ObjHit(false, 0.0, 0u, vec3<f32>(0.0));

  // ---- 局部变换：local = ((w - pos)·col_i) / scale（rd 不归一化）----
  let wp = origin - obj_pos;
  let ro = vec3<f32>(dot(wp, col0), dot(wp, col1), dot(wp, col2)) / scale;
  let rd = vec3<f32>(dot(dir, col0), dot(dir, col1), dot(dir, col2)) / scale;

  // ---- 局部 tile 盒 [0,512]³ slab ----
  let tl = slab_box(ro, rd, vec3<f32>(0.0), vec3<f32>(f32(LOCAL_TILE_FINE)), 0.0, t_hi_cap);
  if (tl.y < max(tl.x, 0.0)) { return ObjHit(false, 0.0, 0u, vec3<f32>(0.0)); }
  let tl0 = max(tl.x, 0.0);
  let tl1 = min(tl.y, t_hi_cap);
  if (tl1 <= tl0) { return ObjHit(false, 0.0, 0u, vec3<f32>(0.0)); }
  // 诊断：局部 slab 后直接 return miss（跳过 A&W）——第 2 级诊断
  // return ObjHit(false, 0.0, 0u, vec3<f32>(0.0));

  // ---- 两级 A&W（cell 0..31³）；全程「相对 start 的 t」标尺，返回时 +tl0 ----
  var sign_v = vec3<i32>(1i);
  sign_v = select(sign_v, vec3<i32>(-1i), rd < vec3<f32>(0.0));
  var delta = vec3<f32>(1e+30);
  delta = select(delta, 1.0 / abs(rd), abs(rd) > vec3<f32>(1e-30));
  let delta_c = delta * 16.0;
  let start = ro + rd * tl0;
  // cc = clamp(floor(start) >> 4, 0, 31)
  var cc = clamp(vec3<i32>(vec3<i32>(floor(start)) >> vec3<u32>(4u)), vec3<i32>(0), vec3<i32>(31));
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
      if (f.hit) {
        // 局部命中法线：-sign[axis] 单位轴；axis=3（起点在体内）→ -rd 方向
        //（CPU: -rd.normalize_or_zero()；rd 非零恒成立，normalize 即可）
        var n_local = -rd;
        if (f.axis == 0u) { n_local = vec3<f32>(f32(-sign_v.x), 0.0, 0.0); }
        else if (f.axis == 1u) { n_local = vec3<f32>(0.0, f32(-sign_v.y), 0.0); }
        else if (f.axis == 2u) { n_local = vec3<f32>(0.0, 0.0, f32(-sign_v.z)); }
        // 局部法线 → 世界：n_world = R · n_local（列线性组合），renormalize 消舍入
        let n_world = normalize(n_local.x * col0 + n_local.y * col1 + n_local.z * col2);
        return ObjHit(true, tl0 + f.t, f.pal, n_world);
      }
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
  return ObjHit(false, 0.0, 0u, vec3<f32>(0.0));
}

// 物体 palette 解包（mov_palette 字基址 + pal_idx × 2 words）
fn mov_palette_rgb(pal_b: u32, pal_idx: u32) -> vec3<u32> {
  let w0 = mov_palette[pal_b + pal_idx * 2u];
  return vec3<u32>(w0 & 0xFFu, (w0 >> 8u) & 0xFFu, (w0 >> 16u) & 0xFFu);
}

// ============================================================================
// P3.1 光照：trace_world 提取（主视线/阴影射线共用）+ scene_occluded 快路径
// ============================================================================

// 世界两级 DDA（原 dda_main 内联体提取，t_exit 初值参数化 = t_max）：
// - 主视线：t_max = frustum_length（行为与 P2.10 前完全一致）
// - 阴影射线：t_max = 段长；window AABB 剪裁保证出窗即停（无空域空走）
// 返回 t（自 origin 的全局标尺）、palette、命中面法线（世界空间，指向射线来向）。
struct WorldHit {
  t: f32,
  pal: u32,
  n: vec3<f32>,
}
fn trace_world(origin: vec3<f32>, dir: vec3<f32>, t_max: f32) -> WorldHit {
  let miss = WorldHit(1e+30, 0u, vec3<f32>(0.0));
  // ---- slab 法求射线 vs brickmap tile-window AABB（fine 坐标）相交区间 ----
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
  var t_enter = 0.0;
  var t_exit  = t_max;
  {
    // X 轴 slab
    let d = dir.x;
    if (abs(d) < 1e-30) {
      if (origin.x < aabb_min.x || origin.x > aabb_max.x) { return miss; }
    } else {
      let t1 = (aabb_min.x - origin.x) / d;
      let t2 = (aabb_max.x - origin.x) / d;
      t_enter = max(t_enter, min(t1, t2));
      t_exit  = min(t_exit,  max(t1, t2));
    }
    // Y 轴 slab
    {
      let d = dir.y;
      if (abs(d) < 1e-30) {
        if (origin.y < aabb_min.y || origin.y > aabb_max.y) { return miss; }
      } else {
        let t1 = (aabb_min.y - origin.y) / d;
        let t2 = (aabb_max.y - origin.y) / d;
        t_enter = max(t_enter, min(t1, t2));
        t_exit  = min(t_exit,  max(t1, t2));
      }
    }
    // Z 轴 slab
    {
      let d = dir.z;
      if (abs(d) < 1e-30) {
        if (origin.z < aabb_min.z || origin.z > aabb_max.z) { return miss; }
      } else {
        let t1 = (aabb_min.z - origin.z) / d;
        let t2 = (aabb_max.z - origin.z) / d;
        t_enter = max(t_enter, min(t1, t2));
        t_exit  = min(t_exit,  max(t1, t2));
      }
    }
  }
  if (t_exit < max(t_enter, 0.0)) { return miss; }
  let t_enter_clamped = max(t_enter, 0.0);
  let t_exit_clamped  = min(t_exit, t_max);
  if (t_exit_clamped <= t_enter_clamped) { return miss; }

  // ---- 起点推进 + 两级 A&W（相对标尺），与 cpu_reference_dda_ray_two_level 同构 ----
  let start_v = origin + dir * t_enter_clamped;
  let t_rel_max = t_exit_clamped - t_enter_clamped;
  var sign_v = vec3<i32>(1i);
  sign_v = select(sign_v, vec3<i32>(-1i), dir < vec3<f32>(0.0));
  var delta = vec3<f32>(1e+30);
  delta = select(delta, 1.0 / abs(dir), abs(dir) > vec3<f32>(1e-30));
  let delta_c = delta * 16.0;
  // 粗 cell = floor(start_v) >> 4（算术右移 = floor 除法，负坐标正确；不 clamp，出窗占用查询返 false）
  var cc = vec3<i32>(vec3<i32>(floor(start_v)) >> vec3<u32>(4u));
  var tmax_c = vec3<f32>(1e+30);
  // 下一粗边界（无 window 下限 clamp：负坐标 tile 合法，同内联版 next_coarse_boundary）
  {
    let b = (cc + select(vec3<i32>(1), vec3<i32>(0), dir < vec3<f32>(0.0))) << vec3<u32>(4);
    let tb = vec3<f32>(b) - start_v;
    let td = tb / dir;
    tmax_c = select(vec3<f32>(1e+30), max(td, vec3<f32>(0.0)), abs(dir) > vec3<f32>(1e-30));
  }
  var t_in = 0.0;
  for (var step_c: u32 = 0u; step_c < 16384u; step_c = step_c + 1u) {
    let t_out = min(tmax_c.x, min(tmax_c.y, tmax_c.z));
    if (cell_occupied(cc)) {
      let t_hi = min(t_out, t_rel_max);
      if (t_hi > t_in) {
        let p = start_v + dir * t_in;
        let base = vec3<i32>(cc << vec3<u32>(4));
        let fc0 = clamp(vec3<i32>(floor(p)), base, base + vec3<i32>(15));
        var fc = fc0;
        var tmax_f = vec3<f32>(1e+30);
        tmax_f = select(tmax_f, max((vec3<f32>(fc0 + select(vec3<i32>(1), vec3<i32>(0), dir < vec3<f32>(0.0))) - p) / dir, vec3<f32>(0.0)), abs(dir) > vec3<f32>(1e-30));
        var t_f = 0.0;
        let span = t_hi - t_in;
        let pal0 = sample_brickmap(fc0);
        if (pal0 != 0u) {
          // 起点即在体内：法线 = -dir（axis=3 语义）
          return WorldHit(t_enter_clamped + t_in, pal0, normalize(-dir));
        }
        for (var i_f: u32 = 0u; i_f < 48u && t_f < span; i_f = i_f + 1u) {
          var axis = 2u;
          if (tmax_f.x <= tmax_f.y && tmax_f.x <= tmax_f.z) {
            t_f = tmax_f.x;
            tmax_f.x = tmax_f.x + delta.x;
            fc.x = fc.x + sign_v.x;
            axis = 0u;
          } else if (tmax_f.y <= tmax_f.z) {
            t_f = tmax_f.y;
            tmax_f.y = tmax_f.y + delta.y;
            fc.y = fc.y + sign_v.y;
            axis = 1u;
          } else {
            t_f = tmax_f.z;
            tmax_f.z = tmax_f.z + delta.z;
            fc.z = fc.z + sign_v.z;
          }
          let pal = sample_brickmap(fc);
          if (pal != 0u) {
            // 面法线 = -sign[axis] 单位轴（射线朝 +axis 穿入 → 面在 -axis 侧）
            var n = vec3<f32>(0.0);
            if (axis == 0u) { n = vec3<f32>(f32(-sign_v.x), 0.0, 0.0); }
            else if (axis == 1u) { n = vec3<f32>(0.0, f32(-sign_v.y), 0.0); }
            else { n = vec3<f32>(0.0, 0.0, f32(-sign_v.z)); }
            return WorldHit(t_enter_clamped + t_in + t_f, pal, n);
          }
        }
      }
    }
    if (t_out >= t_rel_max) { break; }
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
  return miss;
}

// 遮挡快路径（阴影射线）：[0, t_max) 内任一命中即 true，无「最近」比较。
// 对应 mov.rs::cpu_reference_scene_occluded。
fn scene_occluded(origin: vec3<f32>, dir: vec3<f32>, t_max: f32) -> bool {
  let w = trace_world(origin, dir, t_max);
  if (w.t < t_max) { return true; }
  for (var i: u32 = 0u; i < mov_g.count; i = i + 1u) {
    let h = trace_object(i, origin, dir, t_max);
    if (h.hit && h.t < t_max) { return true; }
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
fn hit_mat(obj: i32, pal: u32) -> HitMat {
  var w0: u32;
  var w1: u32;
  if (obj < 0) {
    w0 = b_palette[pal * 2u];
    w1 = b_palette[pal * 2u + 1u];
  } else {
    let pal_b = mov_descs[u32(obj) * MOV_DESC_WORDS + 28u];
    w0 = mov_palette[pal_b + pal * 2u];
    w1 = mov_palette[pal_b + pal * 2u + 1u];
  }
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
// - 世界物体用 sample_brickmap（2~3 级寻址）；MOV 物体暂用 DDA 面法向
// ============================================================================
fn compute_implicit_normal(origin: vec3<f32>, dir: vec3<f32>, t: f32, dda_n: vec3<f32>, is_world: bool) -> vec3<f32> {
  if (!is_world) { return dda_n; }
  let hit_pos = origin + dir * t;
  // dda_n 指向外部（射线来向）；沿 -dda_n 微偏 → 命中体素中心
  let hit_fc = vec3<i32>(floor(hit_pos - dda_n * 0.001));
  // 6 邻域 occupancy（1 = 实心，0 = 空气）
  let sx_n = f32(sample_brickmap(hit_fc + vec3<i32>(-1, 0, 0)) != 0u);
  let sx_p = f32(sample_brickmap(hit_fc + vec3<i32>( 1, 0, 0)) != 0u);
  let sy_n = f32(sample_brickmap(hit_fc + vec3<i32>( 0,-1, 0)) != 0u);
  let sy_p = f32(sample_brickmap(hit_fc + vec3<i32>( 0, 1, 0)) != 0u);
  let sz_n = f32(sample_brickmap(hit_fc + vec3<i32>( 0, 0,-1)) != 0u);
  let sz_p = f32(sample_brickmap(hit_fc + vec3<i32>( 0, 0, 1)) != 0u);
  let raw = vec3<f32>(sx_n - sx_p, sy_n - sy_p, sz_n - sz_p);
  let len2 = dot(raw, raw);
  // 平坦区域（邻域对称，len2 很小）→ 回退 DDA 面法向
  if (len2 < 0.01) { return dda_n; }
  return normalize(raw);
}

// Douglas devlog #02 基础光影合成：
// 1. sky 渐变环境光（按法线 y）
// 2. 方向光硬阴影（命中点向太阳投 1 条射线，不通即阴影）
// 3. 发光体素 radiance 直出（无方向性、不受阴影）
// 无点光源、无 Phong 高光
fn shade_hit(origin: vec3<f32>, dir: vec3<f32>, t: f32, pal: u32, obj: i32, dda_n: vec3<f32>, shadow_t_max: f32) -> vec3<f32> {
  // Douglas devlog #22 implicit normals：邻域 occupancy 有限差分 → 连续法向
  let n = compute_implicit_normal(origin, dir, t, dda_n, obj < 0);
  let p = origin + dir * t;
  let mat = hit_mat(obj, pal);
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
  let size = textureDimensions(gbuf_tex);
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

  // ---- trace_scene：世界两级 DDA + 逐物体 trace_object 取最近 ----
  var best_t = 1e+30;
  var best_pal: u32 = 0u;
  var best_obj: i32 = -1;  // -1 = 世界网格
  var best_n = vec3<f32>(0.0);

  let w = trace_world(origin_fine, dir_fine, frustum_length);
  if (w.t < 1e+29) {
    best_t = w.t;
    best_pal = w.pal;
    best_n = w.n;
  }

  for (var i: u32 = 0u; i < mov_g.count; i = i + 1u) {
    let cap = min(best_t, frustum_length);
    let h = trace_object(i, origin_fine, dir_fine, cap);
    if (h.hit && h.t < best_t) {
      best_t = h.t;
      best_pal = h.pal;
      best_obj = i32(i);
      best_n = h.n;
    }
  }

  // ---- G-buffer 写入 + 可见面注册（P3.5d：移除 per-pixel shade_hit）----
  if (best_t < 1e+29) {
    // debug 模式：法向向量可视化，直接写 out_tex 绕过逐面管线
    if (view_u.debug_mode.x > 0.5) {
      let n_implicit = compute_implicit_normal(origin_fine, dir_fine, best_t, best_n, best_obj < 0);
      textureStore(out_tex, coord0, vec4<f32>(n_implicit * 0.5 + 0.5, 1.0));
      return;
    }
    // 命中：写 G-buffer + 注册可见面
    let hit_pos = origin_fine + dir_fine * best_t;
    let voxel = vec3<i32>(floor(hit_pos - best_n * 0.001));
    let obj_key = select(0u, u32(best_obj) + 1u, best_obj >= 0);
    let face = fl_face_index(best_n);
    let objm = select(GB_OBJM_WORLD, u32(best_obj), best_obj >= 0);
    let meta_w = u32(best_pal) | (objm << 8u) | (face << 16u);
    let world_center = vec3<f32>(voxel) + vec3<f32>(0.5);
    textureStore(gbuf_tex, coord0, vec4<u32>(
      bitcast<u32>(voxel.x),
      bitcast<u32>(voxel.y),
      bitcast<u32>(voxel.z),
      meta_w));
    fl_register(voxel, obj_key, face, world_center);
  } else {
    // 未命中：sky 标记（composite pass 重建方向算 sky()）
    textureStore(gbuf_tex, coord0, vec4<u32>(0u, 0u, 0u, GB_SKY_BIT));
  }
  // out_tex 不写——fl_composite_main 负责最终输出
}

// ============================================================================
// P3.5d 逐面光照管线：四个 compute entry point
//   dispatch 序：fl_clear_main → dda_main → fl_light_main → fl_composite_main
// ============================================================================

// ---- Pass 1：fl_clear_main — 清注册表 status + face_mask（epoch + light 保留）----
// 每线程 1 slot；workgroup 64 → dispatch ceil(FL_REG_SLOTS/64) workgroups。
// 时间复用（后置）：epoch/light 保留本可跨帧复用，但开放寻址槽位随注册竞争
// 顺序漂移 → fl_register 占位时置 FL_EPOCH_INVALID 作废残留，fl_light 恒重算。
// 恢复复用需槽位稳定化（同 key 恒同槽），见 TODO 3.5d 注。
@compute @workgroup_size(64, 1, 1)
fn fl_clear_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= FL_REG_SLOTS) { return; }
  let base = gid.x * FL_WORDS_PER_SLOT;
  atomicStore(&fl_table[base], 0u);                   // status = 0（空，供 fl_register CAS）
  atomicStore(&fl_table[base + 1u], 0u);              // key xyz+obj 清零（v3.9.3：残留 key 会让
  atomicStore(&fl_table[base + 2u], 0u);              // fl_register 聚合分支在「占位中」槽上误匹配
  atomicStore(&fl_table[base + 3u], 0u);              // → mask 写错槽 → 本体素 face 丢注册 → lookup
  atomicStore(&fl_table[base + FL_OFF_OBJ], 0u);      // 兜底 0.3 灰 → 单体素闪烁）
  atomicStore(&fl_table[base + FL_OFF_MASK], 0u);     // face_mask = 0（新帧重新收集）
  // epoch (word 6) + light (words 7-24) + center (words 25-27) 保留
}

// ---- Pass 3：fl_light_main — 逐注册体素逐面算光照 ----
// 每线程 1 slot；workgroup 64 → dispatch ceil(FL_REG_SLOTS/64) workgroups。
// 本帧注册槽 epoch = FL_EPOCH_INVALID → 复用分支恒不命中 = 恒重算（v3.9.1
// 正确性修复；时间复用待槽位稳定化后恢复）。MOV（obj_key != 0）本就恒重算。
// 光照结果 bitcast 存入 slot light 区（FL_OFF_LIGHT + face*3），算完写 cur_epoch。
@compute @workgroup_size(64, 1, 1)
fn fl_light_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= FL_REG_SLOTS) { return; }
  let base = gid.x * FL_WORDS_PER_SLOT;
  if (atomicLoad(&fl_table[base]) == 0u) { return; }  // 空槽跳过

  // 读 key
  let voxel = vec3<i32>(
    bitcast<i32>(atomicLoad(&fl_table[base + 1u])),
    bitcast<i32>(atomicLoad(&fl_table[base + 2u])),
    bitcast<i32>(atomicLoad(&fl_table[base + 3u])));
  let obj_key = atomicLoad(&fl_table[base + FL_OFF_OBJ]);
  let mask = atomicLoad(&fl_table[base + FL_OFF_MASK]);
  let cur_epoch = u32(view_u.fl_epoch.x);

  // 时间复用：世界体素 + epoch 匹配 → 跳过
  let stored_epoch = atomicLoad(&fl_table[base + FL_OFF_EPOCH]);
  if (obj_key == 0u && stored_epoch == cur_epoch) { return; }

  // 面中心（世界体素 = voxel+0.5；MOV = slot 内存储的世界中心）
  var center = vec3<f32>(f32(voxel.x) + 0.5, f32(voxel.y) + 0.5, f32(voxel.z) + 0.5);
  if (obj_key != 0u) {
    center = vec3<f32>(
      bitcast<f32>(atomicLoad(&fl_table[base + FL_OFF_CENTER])),
      bitcast<f32>(atomicLoad(&fl_table[base + FL_OFF_CENTER + 1u])),
      bitcast<f32>(atomicLoad(&fl_table[base + FL_OFF_CENTER + 2u])));
  }

  // 逐面算光照
  for (var f: u32 = 0u; f < 6u; f = f + 1u) {
    if ((mask & (1u << f)) == 0u) { continue; }
    let n = fl_face_normal(f);
    let v = center + n * 0.5;
    let light = face_light_math(n, v);
    let w = FL_OFF_LIGHT + f * 3u;
    atomicStore(&fl_table[base + w], bitcast<u32>(light.x));
    atomicStore(&fl_table[base + w + 1u], bitcast<u32>(light.y));
    atomicStore(&fl_table[base + w + 2u], bitcast<u32>(light.z));
  }

  // 更新 epoch
  atomicStore(&fl_table[base + FL_OFF_EPOCH], cur_epoch);
}

// ---- Pass 4：fl_composite_main — G-buffer + 面光照 → ACES → sRGB → out_tex ----
// 每像素 = workgroup 8×8×1 内一个 invocation（与 dda_main 对齐）。
// sky 像素：重建射线方向 → sky()；命中像素：albedo × fl_lookup(light) + emissive 直出。
@compute @workgroup_size(8, 8, 1)
fn fl_composite_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let size = textureDimensions(out_tex);
  if (gid.x >= size.x || gid.y >= size.y) { return; }
  let coord0 = vec2<i32>(i32(gid.x), i32(gid.y));

  // 读 G-buffer
  let g = textureLoad(gbuf_tex, coord0);
  let meta_w = g.a;
  var col: vec3<f32>;

  // debug_mode.y = G-buffer 状态图（v3.9.3 分割线定位）：sky=品红、命中=face 6 色。
  // 分割线两侧颜色分布直接区分「miss(sky 化)」vs「命中但光照异常」。
  if (view_u.debug_mode.y > 0.5) {
    if ((meta_w & GB_SKY_BIT) != 0u) {
      col = vec3<f32>(1.0, 0.0, 1.0);   // sky/miss = 品红
    } else {
      let face = (meta_w >> 16u) & 7u;
      // face: 0=-X 红 1=+X 绿 2=-Y 蓝 3=+Y 黄 4=-Z 青 5=+Z 紫（6/7 = 异常白）
      if (face == 0u) { col = vec3<f32>(1.0, 0.2, 0.2); }
      else if (face == 1u) { col = vec3<f32>(0.2, 1.0, 0.2); }
      else if (face == 2u) { col = vec3<f32>(0.2, 0.2, 1.0); }
      else if (face == 3u) { col = vec3<f32>(1.0, 1.0, 0.2); }
      else if (face == 4u) { col = vec3<f32>(0.2, 1.0, 1.0); }
      else if (face == 5u) { col = vec3<f32>(1.0, 0.2, 1.0); }
      else { col = vec3<f32>(1.0); }
    }
    textureStore(out_tex, coord0, vec4<f32>(col, 1.0));
    return;
  }

  if ((meta_w & GB_SKY_BIT) != 0u) {
    // sky：重建射线方向
    let px = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5)) / vec2<f32>(f32(size.x), f32(size.y));
    let uv = vec2<f32>(px.x * 2.0 - 1.0, 1.0 - px.y * 2.0);
    let near_h = view_u.inv_view_proj * vec4<f32>(uv.x, uv.y, 0.0, 1.0);
    let far_h  = view_u.inv_view_proj * vec4<f32>(uv.x, uv.y, 1.0, 1.0);
    let diff = (far_h.xyz / far_h.w) - (near_h.xyz / near_h.w);
    col = sky(normalize(diff));
  } else {
    // 命中：解包 meta
    let pal = meta_w & 0xFFu;
    let objm = (meta_w >> 8u) & 0xFFu;
    let face = (meta_w >> 16u) & 7u;
    let voxel = vec3<i32>(bitcast<i32>(g.x), bitcast<i32>(g.y), bitcast<i32>(g.z));
    let obj_key = select(0u, objm + 1u, objm != GB_OBJM_WORLD);
    let obj_i32 = select(-1, i32(objm), objm != GB_OBJM_WORLD);

    // albedo + emissive（hit_mat 内联）
    let mat = hit_mat(obj_i32, pal);
    let light = fl_lookup(voxel, obj_key, face);
    col = mat.albedo * light + mat.albedo * (mat.emissive * EMISSIVE_EMIT_GAIN);
    col = col * light_u.g.exposure_pad.x;
  }

  col = aces_tonemap(col);
  col = linear_to_srgb(col);
  textureStore(out_tex, coord0, vec4<f32>(col, 1.0));
}
