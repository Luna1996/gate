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

// 世界空间探针网格（每 LOD）：origin.xyz = 世界原点（voxel）、origin.w = cell 边长；
// dims.xyz = cell 维度、dims.w = 该 LOD 在全局 slot 数组中的起始下标。
struct DdgiLod {
  origin: vec4<i32>,
  dims: vec4<u32>,
};
// LOD0 的 chunk 段编址（见 DdgiUniform.chunk）
struct DdgiChunk {
  origin: vec4<i32>,
  dims: vec4<u32>,
};
struct DdgiUniform {
  lods: array<DdgiLod, 4>,
  // x=帧计数, y=调试模式(0..4), z=GI 增益, w=保留通道（旧借针半径，机制已删除），恒 0
  params: vec4<f32>,
  // x=GI 开关, y=总槽位数, z=Chebyshev std 信任系数（0..1，见采样侧 soft 计算）,
  // w=级联覆盖**之外**的天光兜底强度（见着色侧 amb_far / DdgiDebugSettings.far_ambient）
  misc: vec4<f32>,
  // 脏区（世界 voxel AABB）：dirty_min.xyz = min、w = 1 表示有效；dirty_max.xyz = max（不含）
  dirty_min: vec4<f32>,
  dirty_max: vec4<f32>,
  // LOD0 的 chunk 段编址（**只被 lod==0 读**；LOD1~3 走上面的 lods 世界网格）：
  //   origin.xyz = chunk 网格原点（chunk 坐标），origin.w = 每 chunk 每轴 cell 数（16）
  //   dims.xyz   = chunk 网格维度，dims.w = LOD0 已分配槽数（探针池高水位）
  chunk: DdgiChunk,
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
/// LOD0 的两张 chunk 段表 + 一段**无读者**的保留前缀。
///
/// 【旧「cell → slot 借针间接表」已删】它曾允许"本格放不出探针"的 cell 指向邻近 cell 的探针。
/// 放置规则改成 Douglas 的"最大的全空子块"后，格内只要有空体素就一定放得出探针，这个需求消失，
/// 因此 `ddgi_slot` 直接返回 `ddgi_slot_own`，前段 `[0, total_slots)` 不再被任何代码读取
/// （Rust 侧仍按 identity 铺一遍以保持下面的偏移不变，见 `DdgiGpu::cell_slot`）。
///
/// 【为什么复用这张 buffer 放 chunk 表】BG4 的 storage binding 已用满 8 个（WGSL 下限），
/// 不再新增 binding：`[total_slots, +num_chunks)` = chunk_base，之后到 `+lod0_slots`
/// = slot_chunk。偏移由 `misc.y` 与 chunk 维度算出，见 `ddgi_chunk_base_off`/`ddgi_slot_chunk_off`。
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
