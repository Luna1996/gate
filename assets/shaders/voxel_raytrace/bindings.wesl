// --- BG0：输出 + 视图 uniform ---
// out_tex：linear RGB，着色末尾转 sRGB。beam_depth：低分辨率最近命中 t（beam pass 写，主 pass 读）。
@group(0) @binding(0) var out_tex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var beam_depth: texture_storage_2d<r32float, read_write>;
/// @binding(3)：眼睛适应的只读视图，给 `dda_main` 取曝光系数；同 buffer 在 @group(7) @binding(1) 被 `eye_adapt_*` 读写。
@group(0) @binding(3) var<storage, read> eye_adapt_ro: array<u32>;

// beam 低分辨率边长 = 全分辨率 / BEAM_DIV。
const BEAM_DIV: u32 = 4u;

// beam 起点保守回退（世界单位）。
const BEAM_BACKOFF: f32 = 4.0;

struct DdaViewUniform {
  view_proj: mat4x4<f32>,       // 64B：世界→裁剪空间
  inv_view_proj: mat4x4<f32>,   // 64B
  cam_pos_voxel: vec4<f32>,      // 16B，w=1
  debug_mode: vec4<f32>,        // 16B：x = 法向可视化，y = face 6 色诊断
  lod: vec4<f32>,               // 16B：x = 像素角大小(rad)，y = LOD 早停开关
}
@group(0) @binding(1) var<uniform> view_u: DdaViewUniform;

// --- BG1：brick map 全部数据 + globals ---
@group(1) @binding(0) var<storage, read> b_struct: array<u32>;
@group(1) @binding(1) var<storage, read> b_leaves: array<u64>;
@group(1) @binding(2) var<storage, read> b_palette: array<u32>;

// BrickMapGlobals 标量镜像（字段顺序须与 Rust 一致）。
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
// @binding(4)/(5)：光照场（AO fill，Rgba16Unorm，仅 `.a` 有效、`.rgb` 恒 0，采样走硬件三线性）。
// 16-voxel cell 网格，相机中心 + 世界锚定槽位：纹素下标 = 世界 cell mod dims。
@group(1) @binding(4) var light_tex: texture_3d<f32>;
@group(1) @binding(5) var light_samp: sampler;

// --- BG2：GridDesc 数组（144B/entry，与 Rust GridDesc 字节一致）---
struct GridDesc {
  pos_scale: vec4<f32>,   // xyz = 位置，w = scale
  rot0: vec4<f32>,        // xyz = 旋转矩阵列 0，w = 0
  rot1: vec4<f32>,        // xyz = 旋转矩阵列 1，w = 0
  rot2: vec4<f32>,        // xyz = 旋转矩阵列 2，w = 0
  aabb_min: vec4<f32>,    // xyz = 世界 AABB min，w = 0
  aabb_max: vec4<f32>,    // xyz = 世界 AABB max，w = 0
  tree_base: u32,         // b_struct 内本 volume 树基址（含 chunk 窗口段）
  tree_depth: u32,        // 4（树最大分裂深度）
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

// --- BG3：光照光池（与 Rust LightPoolUniform 逐字段镜像，448B）---
// 只用 lights[0] = 方向光（kind=0）。
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
  sky_color: vec4<f32>,   // 天空纯色（miss 背景 + sky 环境光共用）
}
@group(3) @binding(0) var<uniform> light_u: LightPool;

// ===== GI（屏幕空间逐面 ReSTIR）=====
// uniform 与 Rust `GiUniform` 逐字段镜像（字节一致）。
struct GiUniform {
  // x = 保留（恒 0）、y = 保留（恒 0）、z = GI 增益、w = 保留（恒 0）
  params: vec4<f32>,
  // x = GI 开关、yzw = 保留（恒 0）
  misc: vec4<f32>,
  // x = 保留（恒 0）、y = GI 分辨率除数（1 = 全分辨率，2 = 半分辨率；整数值的 f32，只被
  // `gi_main` 用来把本 pass 的像素下标换成 beam 纹理下标）、
  // z = 世界几何相对「写入上一帧 reservoir 的那一帧」是否**逐位未变**（1 = 未变）⇒
  // 屏幕空间路径可跳过时域二次顶点键验证射线（见 `gi/screen.wesl` ① 与 `prepare_gi` 的 `world_rev`）、
  // w = 保留（恒 0）
  flags: vec4<f32>,
  // x = 自增帧号（精确 u32；yzw = 保留，恒 0）。
  // **所有整数帧逻辑只用本字段**（本帧的 RNG 种子混入、像素 hash）。帧号曾经以 f32 存在
  // `params.x` —— 超过 2^24 后 f32 无法表示连续整数，种子会偶发重复；`gpu.frame` 本就是 u32，
  // 这里逐字镜像、不再经 f32。
  seq: vec4<u32>,
  // 上一帧的相机矩阵（时域复用：把本帧主命中点重投影到上帧的 GI 网格）。
  // Rust `prepare_gi` 每帧把当前矩阵存下、把上一帧的写进这里 ⇒ 与上帧 `gi_main` 实际用的逐位一致。
  // 只用到 `prev_view_proj`（重投影三维点不需要逆矩阵）。
  prev_view_proj: mat4x4<f32>,
};
@group(4) @binding(0) var<uniform> gi_u: GiUniform;
/// 20/21：**屏幕空间逐面 ReSTIR** 的 reservoir 双缓冲。
/// 每 GI 像素一个 reservoir（`GI_RES_WORDS` 个 word，布局见 `gi/screen.wesl`）；
/// 20 = **本帧写入**（目标）、21 = **上一帧读出**（时域重投影 + 空间复用源）。
/// Rust 侧每帧 ping-pong 换绑同一对 buffer ⇒ 20 与 21 恒是不同块。
/// 20 里**只**有「新鲜 + 时域」的结果：空间复用的合并量绝不回写（见 `gi/screen.wesl` 文件头 ②）。
@group(4) @binding(20) var<storage, read_write> gi_res_cur: array<u32>;
@group(4) @binding(21) var<storage, read_write> gi_res_prev: array<u32>;

// ===== @group(5)：GI（网格 = 渲染分辨率 / gi_div）=====
// @binding(2)/(3)：GI 缓冲的写入侧（`gi_main` 用）；采样侧见 `@binding(4)/(5)`。
@group(5) @binding(2) var gi_out: texture_storage_2d<rgba16float, write>;
@group(5) @binding(3) var gi_cov_out: texture_storage_2d<rg32float, write>;
// @binding(4)/(5)：同一对纹理的采样侧（只被 `dda_main` 用）。
// rgba16f：rgb = gi·valid、a = valid；rg32f：r = cov·valid、g = valid。采样走 `light_samp` 双线性、按 valid 归一化。
@group(5) @binding(4) var gi_tex: texture_2d<f32>;
@group(5) @binding(5) var gi_cov_tex: texture_2d<f32>;
/// @binding(6)：**降噪导引**（`gi_main` 写；降噪两段 + `dda_main` 的几何感知上采样读）。
/// 每 GI 像素 `GI_DEN_GUIDE_WORDS` 个 word，布局见 `gi/consts.wesl`：
/// 面键 2（有效 = 键非零，**没有**独立有效位）+ 命中体素中心世界位置 3 + 法线 1
/// + 上帧重投影像素 1 + 平面检验容差 1。
/// 声明为 `read_write`：同一份 shader 模块里这个 var 在 `gi_main` 的 pipeline 里被写
/// （layout 也是 read_write），在 `dda_main` 的 pipeline 里只被读 —— **两边的 layout 都必须
/// 给 read_only = false**，否则与这里的声明不符（wgpu 会报绑定类型不匹配）。
@group(5) @binding(6) var<storage, read_write> gi_guide_out: array<u32>;

// ===== @group(0) @binding(10..)：GI 降噪（两个新 pass，见 `gi/denoise.wesl`）=====
// 降噪 pass 的 bound group 只有 group(0) 一份（`bindings.wesl` 的 binding 号在**同一个模块里唯一**
// ⇒ 不改动既有 0..3 的号段，另起 10..）。Rust 侧 `gi_den_*_layout` 与之逐号对应。
/// 10：降噪导引的**读侧**（`gi_main` 通过 @group(5) @binding(6) 写同一块 buffer）。
@group(0) @binding(10) var<storage, read> gi_guide: array<u32>;
/// 11：本帧原始 GI（`gi_out` 的采样视图，只读）：rgb = gi·valid、a = valid。
@group(0) @binding(11) var gi_raw_tex: texture_2d<f32>;
/// 12/13：时域历史双缓冲（每像素 `GI_DEN_HIST_WORDS` 个 word）。12 = 上帧读、13 = 本帧写。
@group(0) @binding(12) var<storage, read> gi_hist_prev: array<u32>;
@group(0) @binding(13) var<storage, read_write> gi_hist_cur: array<u32>;
/// 14/15：迭代 atrous 的输入（采样视图）/ 输出（存储视图）。Rust 每轮换绑 ⇒ 同一入口跑任意轮数。
@group(0) @binding(14) var gi_dn_src: texture_2d<f32>;
@group(0) @binding(15) var gi_dn_dst: texture_storage_2d<rgba16float, write>;
/// 16：时域累积的输出（atrous 第一轮的输入）。
@group(0) @binding(16) var gi_dn_tmp: texture_storage_2d<rgba16float, write>;
/// 17：每像素亮度 range 权重的 φ（= `GI_DEN_LUMA_K` × 本帧噪声 σ）。时域写、atrous 读。
/// 必须来自**本帧原始**噪声（不是滤波后的），否则迭代几次后 φ 退化成 0、跨面就开始糊。
@group(0) @binding(17) var<storage, read_write> gi_den_phi: array<f32>;

// ===== @group(7)：眼睛适应（自动曝光）——被 `dda_main`（读曝光）与 eye_adapt_* 两个入口使用 =====
/// `out_tex` 的采样视图（同一张图），统计曝光已应用之后的颜色。
@group(7) @binding(0) var out_tex_src: texture_2d<f32>;
/// 状态 + 直方图合一（word 布局，f32 一律 bitcast 存取）：[0] 曝光系数、[1] 已适应 EV、
/// [2] 备用（手动 EV 偏移）、[3] 帧间隔 dt（秒，CPU 每帧写）、[4] 直方图百分位平均 log2 亮度（调试）、
/// [5] 目标 EV（调试）、[8..72) 64 桶 log2 亮度直方图（本帧计数，`eye_adapt_update` 末尾清零）。
@group(7) @binding(1) var<storage, read_write> eye_adapt: array<atomic<u32>>;
