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
  lod: vec4<f32>,               // 16B：x = 像素角大小(rad)，y = 叶级 LOD 开关（阈值在 trace.wesl）
}
@group(0) @binding(1) var<uniform> view_u: DdaViewUniform;

// --- BG1：brick map 全部数据 + globals ---
@group(1) @binding(0) var<storage, read> b_struct: array<u32>;
@group(1) @binding(1) var<storage, read> b_leaves: array<u64>;  // 方向可达掩码 LUT（Rust 侧按 2 个 u32/项写入）
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
// @binding(4)：**线性采样器**（Rust 侧 desc：ClampToEdge ×3 + mipmap_filter = Nearest）。
// 唯一消费者是 GI 缓冲的采样侧（`gi_tex`，见 main.wesl 的几何感知上采样与玻璃的 GI 项）。
// PBR 贴图**不要复用它**（MT2-3 起）：triplanar 要 Repeat 平铺、要 Linear mipmap filter。
@group(1) @binding(4) var light_samp: sampler;

// --- BG1 追加（MT2-2，从 binding 5 起，不动 0..4）：全局材质资产表 + PBR 贴图集 ---
// 资产条目：与 Rust `wire.rs::MaterialAsset` **逐字段镜像**（8 × u32 = 32B，字段名与顺序都必须一致；
// storage array 的 stride 由 WGSL 对齐规则给出 = 4B 对齐 × 8 字段 = 32B，与 Rust 的 `repr(C)` 相同）。
// 语义（权威注释在 `wire.rs`）：`*_slot` = `texture_2d_array` 的层号，`0xFFFFFFFF`
// （= Rust `MATERIAL_SLOT_NONE`）表示"无贴图 ⇒ 走标量回退值"；其余 3 个字是标量回退值。
struct MaterialAsset {
  albedo_slot: u32,        // albedo 贴图层号；NONE ⇒ 取 albedo_rough 的颜色
  roughmetal_slot: u32,    // rough-metal 贴图层号（glTF ORM：G = Roughness、B = Metalness）；NONE ⇒ 取标量
  emissive_slot: u32,      // emissive 贴图层号（可选）；NONE ⇒ 只用标量强度
  transmission_slot: u32,  // transmission 贴图层号（可选）；NONE ⇒ 只用标量透射率
  height_slot: u32,        // 高度（位移）贴图层号 —— **仅 CPU 侧读**（MT6 的 CSG 位移），shader 不读
  albedo_rough: u32,       // 标量回退：color.rgb(sRGB) | roughness<<24（布局同平凡变体 word0）
  // 标量回退：emissive | metallic<<8 | specular<<16 | **displacement_amplitude<<24**。
  // 最高字节（bits 24..31 = D1 预留的"保留"字节）自 MT8-5 起有语义 = 位移幅度（**体素**，0 = 不位移）。
  // ⚠️ **CPU 侧专用**：位移发生在体素化那一刻（`gate-app` 解码高度图 → 普通 CPU 高度场 + 闭包 →
  // `gate_voxel`），产物就是普通体素 ⇒ **shader 不读这个字节**（本文件下面只解低三字节；
  // 权威注释与读写助手在 `wire.rs::MaterialAsset::displacement_amplitude`）。
  emissive_metal: u32,
  transmission_ior: u32,   // transmission | ior_x100<<16（IOR 是资产级物理基值，同时服务折射与 F0）
};
// 全局资产表：**所有 volume 共用一张**（不是 per-volume）—— palette 的 PBR 变体里 `asset: u16`
// 就是这张表的**全局下标**。表大小 = `MATERIAL_ASSET_SLOTS`（WESL 权威常量在 `common.wesl`）。
@group(1) @binding(5) var<storage, read> material_assets: array<MaterialAsset>;
// 采样 `pbr_albedo_rough` 的 rgb 后**必须**走 `srgb_to_linear()`：两张数组图都是 `Unorm` 格式、
// 采样不做 sRGB 转换（rgb 是 sRGB 编码的原始字节），与 `srgb_word_to_linear` 读 palette 字节后同样
// `srgb_to_linear()` 是**完全同一口径** ⇒ 平凡材质与 PBR 材质能共用同一个 BRDF 入口。
@group(1) @binding(6) var pbr_albedo_rough: texture_2d_array<f32>;  // Rgba8Unorm: rgb=albedo(sRGB 编码字节)、a=roughness
@group(1) @binding(7) var pbr_metal: texture_2d_array<f32>;         // R8Unorm: r=metalness
// @binding(8)：**PBR 贴图专用采样器**（MT2-3）。与 `light_samp` 分开是必须的 ——
// ① triplanar 要**三轴 Repeat**（GI 缓冲的采样器是 ClampToEdge，采样越界要钳住而不是绕回）；
// ② 要 **Linear mag/min/mipmap**（`light_samp` 的 mipmap_filter = Nearest，它没有 mip 链）。
// 权威 desc 在 Rust `pbr_texture.rs::create_pbr_sampler`：Repeat×3 / Linear(mag,min,mipmap) /
// anisotropy_clamp ≤ 8（设备不支持 `SAMPLER_ANISOTROPY` 时退化 1）/ lod_max_clamp = mip 链底层。
// 改那边的状态必须同改这里（占位与真身共用同一个 sampler 实例，见 `upload.rs::init_empty_gpu`）。
@group(1) @binding(8) var pbr_samp: sampler;

/// 9：**叶级 LOD 诊断计数器**（M0；`trace.wesl::LOD_DIAG` 为 1 时才被写）。
/// 为什么挂 BG1 而不是 BG0：BG1 是 dda / beam / gi / 光柱掩码**四个会调 trace 的 pass 共用**的那一份，
/// 而 BG0 有两份（完整版 + GI 瘦版）⇒ 挂 BG1 只需改一处 layout + 一处 bind group。
/// 槽位见 `trace.wesl::DIAG_*`；**只增不清**（CPU 侧读差值）⇒ 不需要清零，也没有"清零写 vs GPU 写"的竞态。
@group(1) @binding(9) var<storage, read_write> lod_diag: array<atomic<u32>>;

// **BG1 的 binding 号到此为止（0..9）**：MT8-3 曾在 10/11 加一对「反射缓存乒乓双缓冲」，
// 已按实测整体删除（负优化：用户判定"删除后画面不变、反而更快"）—— 见 `main.wesl` 文件头的
// MT4 注释块。**别再往 10 起加东西**以为这里是预留空位：那两块 buffer 与 `ReflEntry` 都已不存在。

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
  // 天体盘的外观（只被 `volumetric.wesl::sky_primary` 读；日月共用 lights[0] 这一个槽位，见 sky.rs）：
  // x = 盘的亮度倍数（乘 SUN_DISK_GAIN）；yzw reserved（恒 0）—— **光晕**不在这里，它是大气散射，走光柱 uniform 的「光晕」。
  shape: vec4<f32>,
}
struct LightGlobals {
  count: u32,
  // **镜面档位**（`REFL_TIER_*`，见 `main.wesl` 的 MT4 段）：0 = 关、1 = 太阳直射 + 自发光、
  // 2 = +逃逸天空走 `sky_primary`（镜像里有太阳盘）、3 = +反射命中点补阴影射线。
  // 原来是废弃的 `_pad0`（恒 0）⇒ 复用空闲填充，布局与字节数都不变（Rust 侧 `LightGlobals`
  // 同名字段，写侧在 `prepare_dda_bind_groups`）。
  refl_tier: u32,
  // **镜面嵌套层级**（0 / 1 / 2 / 4）：镜子里的镜子最多再反射几级（见 `main.wesl` 的 MT4-5 段）。
  // 同样复用废弃填充 `_pad1`。写侧 = `prepare_dda_bind_groups`（Rust `ReflectionSettings::nest`）。
  refl_nest: u32,
  // **「基础」开关位**（菜单「渲染/基础」，位编码 = 下面紧跟的 `BASE_FLAG_*`）：置位 = **算**。
  // bit0 = 直光阴影、bit1 = 隐式法相。同样复用废弃填充 `_pad2`。
  // 写侧 = `prepare_dda_bind_groups`（Rust `BaseSettings::flags`）。
  base_flags: u32,
  ambient: vec4<f32>,     // rgb = 环境色（线性）
  exposure_pad: vec4<f32>, // x = 曝光系数
}
struct LightPool {
  g: LightGlobals,
  lights: array<LightDesc, 8u>,
  sky_color: vec4<f32>,   // 天空纯色（miss 背景 + sky 环境光共用）
}
@group(3) @binding(0) var<uniform> light_u: LightPool;

// ---- 「基础」开关的位编码（权威在这里：它描述的是上面 `base_flags` 的布局）----
// 必须与 Rust `BaseSettings::FLAG_*`（`gate-render/src/lighting.rs`）一致。
/// bit0 **直光阴影**：置位 ⇒ 发射阴影射线（`sun_transmittance`）。
const BASE_FLAG_SHADOW: u32 = 1u;
/// bit1 **隐式法相**：置位 ⇒ 算邻域梯度法线；清零 ⇒ 不算它，并且**原色直出**。
const BASE_FLAG_IMPLICIT_NORMAL: u32 = 2u;

// ===== GI（屏幕空间逐面 ReSTIR）=====
// uniform 与 Rust `GiUniform` 逐字段镜像（字节一致）。
struct GiUniform {
  // x = **二次顶点太阳反弹**（1 = 开、0 = 关；菜单「渲染/RESTIR GI/太阳反弹」：
  // 关掉则二次顶点不做太阳 NEE，`sun_transmittance` 那条阴影射线整条不发）、
  // y = 保留（恒 0）、z = GI 增益、w = 保留（恒 0）
  params: vec4<f32>,
  // x = GI 开关、y = 每帧采样预算分摊 N、z = 误差驱动重分配档位、
  // w = **第二条弹射的倍数**（菜单「渲染/光照/二次弹射」）：0 = 关；1 = 每个候选都带；
  //     > 1 = 弹射分裂（每个候选以 `1/本值` 的概率带上、带上时乘本值 ⇒ 期望无偏，只有方差变大）。
  //     落点在 `gi/screen.wesl` 的候选循环（抽签）与 `gi/ray.wesl::gi_bounce2`（乘积）。
  misc: vec4<f32>,
  // x = 保留（恒 0）、y = GI 分辨率除数（1 = 全分辨率，2 = 半分辨率；整数值的 f32，只被
  // `gi_main` 用来把本 pass 的像素下标换成 beam 纹理下标）、
  // z = 世界的体素占据相对「写入上一帧 reservoir 的那一帧」是否**逐位未变**（1 = 未变）⇒
  // **本帧允许复用历史**（时域 + 空间两条；变过就整帧不复用，见 `gi/screen.wesl` 文件头 ⑥ 与
  // `prepare_gi` 的 `world_rev`）、
  // w = **降噪质量档**（0 = 关 / 1 = 低 / 2 = 中 / 3 = 高；菜单「渲染/RESTIR GI/降噪质量」）：
  // 本 pass 只用它区分「高」档（1/4 档新鲜候选数与 reservoir 记忆窗按它取，见 `gi/consts.wesl`）；
  // 档位影响的 pass 派发（时域/atrous 开关、核半径）由 Rust 决定。
  flags: vec4<f32>,
  // x = 自增帧号（精确 u32）、y = **二次顶点缓存的 epoch**（光照 / 几何的修订号；
  // Rust `prepare_gi` 逐项比对、变了才自增 ⇒ 参与 `gi_sec_slots` 槽里键的掩码，见 `gi/common.wesl`
  // 的「跨帧持久」段）、zw = 保留（恒 0）。
  // **所有整数帧逻辑只用 x**（本帧的 RNG 种子混入、像素 hash）。帧号曾经以 f32 存在
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

// ===== @group(4) @binding(1..5)：光柱（godray）的**两步链** =====
// 为什么挂在 group(4)：掩码 pass 的 group(0..3) 已经被 view/beam + brickmap 占满（与主 pass 同一套绑定），
// 而 blur 那两个 pass 只用 group(0) 一份 —— 于是**同一份 layout** 在两个 pipeline 里分别当 group(4)
// 与 group(0) 用（`BindGroupLayoutDescriptor` 不带组号，组号由 pipeline layout 数组的下标给）。
// 1..5 是 group(4) **空着的号**（0 = `gi_u`、20/21 = GI 的 reservoir）。
/// 光柱的参数 uniform（与 Rust `volumetric::FogUniform` 逐字段镜像，字节一致）。
struct FogUniform {
  /// x = **衰减**（模糊每步的权重乘它：`< 1` ⇒ 光柱离主体越远越淡）、yzw 保留（恒 0）。
  params: vec4<f32>,
  /// x = 保留（恒 0：网格分辨率除数已固定为 `volumetric.wesl::FOG_DIV`，不再是档位）、
  /// y = **集中度**（主体方向权重的指数：越大越集中在该天体附近，见 `volumetric.wesl`）、
  /// z = **强度**（乘在光柱上）、w = **朝向因子**（主体在画面背后 ⇒ 0，平滑淡出）。
  misc: vec4<f32>,
  /// x = **日月外观开关**（1 = 画、0 = 不画；`sky_primary` 见它清零就直接返回天光基色 ——
  /// **盘与光晕一起消失**）、y 保留（恒 0）、z = **天体盘角径**（rad：天空里那个盘的半径，日月共用；
  /// 它同时决定盘的软边缘与光晕的尺度）、w = **光晕强度**（盘外那圈拖尾的系数，随角径一起缩放）。
  /// 后三项只被 `sky_primary` 用（天空背景）；面板在「渲染/天空/日月」。
  misc2: vec4<f32>,
  /// xy = **天体的屏幕坐标**（0..1；可能落在画面外 —— 模糊的取样方向就是它）、zw 保留（恒 0）。
  misc3: vec4<f32>,
}
@group(4) @binding(1) var<uniform> fog_u: FogUniform;
/// 掩码 pass 只**写**中间靶 A（它不读任何纹理）⇒ 这一组只有这两个号。
@group(4) @binding(3) var fog_a_w: texture_storage_2d<rgba16float, write>;

// ===== @group(0) @binding(21..25)：光柱的**径向模糊**那两个 pass =====
// 为什么另起一批号：WGSL 把 `@group(N)` 写进声明里 ⇒ **同一批资源要按 pipeline layout 的组号
// 分别声明**（模糊那两个 pass 的 layout 只有 group(0) 一组，掩码 pass 的 0..3 已被 view/beam +
// brickmap 占满）。21..25 是这一组**空着的号**（10..20 是 GI 降噪、0..3 是主 pass 的 BG0）。
/// 模糊 pass 的参数 uniform（与 `fog_u` 是**同一块 buffer** 的另一个声明）。
@group(0) @binding(21) var<uniform> fog_u_blur: FogUniform;
/// 中间靶 A/B 的采样侧（只读；手写双线性用 `textureLoad`）。
@group(0) @binding(22) var fog_a_r_blur: texture_2d<f32>;
@group(0) @binding(24) var fog_b_r_blur: texture_2d<f32>;
/// 中间靶 A/B 的存储侧（写入）。
@group(0) @binding(23) var fog_a_w_blur: texture_storage_2d<rgba16float, write>;
@group(0) @binding(25) var fog_b_w_blur: texture_storage_2d<rgba16float, write>;


// ===== @group(5)：GI（网格 = 渲染分辨率 / gi_div）=====
// @binding(2)：GI 缓冲的写入侧（`gi_main` 用）；采样侧见 `@binding(4)`。
@group(5) @binding(2) var gi_out: texture_storage_2d<rgba16float, write>;
// @binding(4)：同一张纹理的采样侧（只被 `dda_main` 用）。
// rgba16f：rgb = gi·valid、a = valid。采样走 `light_samp` 双线性、按 valid 归一化。
// **覆盖度 cov 不再单独占一张纹理**：`gi_ss_main` 往 `gi_cov_out` 写的恒是「valid ? 1 : 0」，
// 于是 `cov` 与 `gi_tex.a` 是同一个场（`cov = s.a / max(s.a, 1e-4)` 恒为 0 或 1）。
@group(5) @binding(4) var gi_tex: texture_2d<f32>;
/// @binding(6)：**降噪导引**（`gi_main` 写；降噪两段 + `dda_main` 的几何感知上采样读，
/// 另外 `dda_main` 在 `gi_div = 1` 时也读它拿**着色法线**、省掉一次 `voxel_normal`）。
/// 每 GI 像素 `GI_DEN_GUIDE_WORDS` 个 word，布局见 `gi/consts.wesl`：**面键 2**
/// （有效 = 键非零，**没有**独立有效位）+ **着色法线 1**（八面体 f16x2）+ **上帧重投影像素 1**。
/// 「是不是同一个面」= 键（平面归属）+ 着色法线（相似度），两者都不需要世界位置 / 平面容差。
/// 声明为 `read_write`：同一份 shader 模块里这个 var 在 `gi_main` 的 pipeline 里被写
/// （layout 也是 read_write），在 `dda_main` 的 pipeline 里只被读 —— **两边的 layout 都必须
/// 给 read_only = false**，否则与这里的声明不符（wgpu 会报绑定类型不匹配）。
@group(5) @binding(6) var<storage, read_write> gi_guide_out: array<u32>;

/// @binding(8)：**帧内逐面去重表**（槽布局 = `gi/common.wesl` 的 `FACE_*`）。
/// 粒度 = **一个可见体素面**（体素 + 它朝向相机的那个面）⇒ 键 = `gi_key0`/`gi_key1`（体素 + 面 + 物体）。
/// 生命周期只有一帧：Rust 每帧在 `gi_main` 之前整块清空 ⇒ `FACE_W_FLAG == 0` = 空槽。
/// 三个消费者跨三个 dispatch（认领在 `gi_main`、逐面着色在 `dda_face_main`、查表在 `dda_main`）
/// ⇒ **dispatch 边界就是可见性边界**，不需要任何屏障 / 内存序 / 撕裂保护。
/// 声明成 atomic：认领走 `atomicCompareExchangeWeak`，其余读写走 `atomicLoad`/`atomicStore`
/// （relaxed ⇒ 与普通访存同价）。槽数由 Rust 按 GI 网格像素数给（`arrayLength` 读得到）。
@group(5) @binding(8) var<storage, read_write> face_slots: array<atomic<u32>>;

/// @binding(9)：**二次顶点按面缓存**（槽布局 = `gi/common.wesl` 的 `GI_SEC_*`）。
/// 键口径与 `face_slots` 相同（`gi_key0`/`gi_key1`），但填充者是 GI 射线自己（惰性、写同值 ⇒ 竞态
/// 幂等）：同一个面被成千上万条射线打到时只算一次二次顶点着色（含它那条太阳 NEE 阴影射线）。
/// **跨帧持久**（与 `face_slots` 不同）：不需要认领、**没有每帧清空**，槽里的键带 epoch 掩码
/// （uniform `gi_u.seq.y`）⇒ 光照 / 几何一变整张表自失效。算法与代价、撕裂窗口的
/// 处理都在 `gi/common.wesl` 的那两段说明里；这里只声明它只被 `gi_main` 使用 ⇒ 只出现在 group(5)。
@group(5) @binding(9) var<storage, read_write> gi_sec_slots: array<atomic<u32>>;

// ===== @group(6)：光柱（godray）的**读侧**（只给 `dda_main` 的 pipeline）=====
// 为什么单开一组而不是塞进 group(5)：group(5) 是「GI 网格」的号段，而光柱的网格尺寸、资源、
// 生命周期都与 GI 无关（分辨率档各自独立）⇒ 混在一组里会让两个模块的绑定互相牵制。这一组只有
// `dda_main` 用，其余入口都不引用它（未用的绑定会被 naga 剪掉，layout 里不必为它们留号）。
//   0 = 参数 uniform（**与 @group(4) @binding(1) 是同一块 buffer**：`dda_main` 的天空背景要走
//       `sky_primary`，而那条路径只有 group(6)。同一 buffer 两个声明，与 `gi_guide_out`/`gi_guide`
//       的先例一致）
//   1 = 光柱的采样视图（**中间靶 A** —— 模糊 B 的输出，见 @group(4) 的说明）
// 关掉光柱时 Rust 绑 1×1 的零纹理（uniform 照绑：太阳盘与开关无关）。
@group(6) @binding(0) var<uniform> fog_u_read: FogUniform;
@group(6) @binding(1) var fog_tex: texture_2d<f32>;

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
/// 雾的空间段复用同一个号：`fog_main` 通过 @group(4) @binding(6) 写同一块 buffer
/// （口径见 `volumetric.wesl`：雾用蓄水池的标准误，GI 用同面邻域的标准差）。
@group(0) @binding(17) var<storage, read_write> gi_den_phi: array<f32>;
/// 20：降噪 pass 的**小配置**（每帧由 Rust 按菜单「高质量降噪」写入）：
/// `[0]` = atrous 单轮核半径（`GI_DEN_ATROUS_R` 或 `GI_DEN_ATROUS_R_FAST`）。
/// 为什么另开一个小 buffer 而不是读 @group(4) 的 `gi_u`：降噪两个 pass 的 layout **只有 group(0)**
/// 一份（刻意 —— 不拉进 brickmap / 相机矩阵），而「bind group 必须从索引 0 起成前缀设置」
/// ⇒ 它们够不到 group(4)。加一个 4 字节 storage 是最省的接法。
@group(0) @binding(20) var<storage, read> gi_den_cfg: array<u32>;

/// 18/19：**① 逐面合并**（`main.wesl::gi_face_flatten`）的 group(0) 声明 —— 那个 pass 不拉整套
/// group(0..5)，只挂三件东西：导引（用上面的 binding 10）+ 逐面去重表（18，与 `@group(5) @binding(8)`
/// 是同一块 buffer）+ GI 写入侧（19，与 `@group(5) @binding(2)` 是同一张纹理）。
/// **同一资源在两个 group 各声明一次**是本工程既有手法（先例：`gi_guide_out`/`gi_guide`、
/// `fog_u`/`fog_u_blur`）；group(0) 的 18/19 在现有各 layout 里都空着。
/// ⚠️ 这两个名字与 group(5) 的两份是**两块绑定的同一资源**，查槽请用 `face_slot_of_flat`
/// （`face_slot_of` 读的是 group(5) 的声明，会把 group(5) 拖进这个只有瘦 group(0) 的 pass）。
@group(0) @binding(18) var<storage, read_write> face_slots_flat: array<atomic<u32>>;
@group(0) @binding(19) var gi_out_flat: texture_storage_2d<rgba16float, write>;

// ===== @group(7)：眼睛适应（自动曝光）——被 `dda_main`（读曝光）与 eye_adapt_* 两个入口使用 =====
/// `out_tex` 的采样视图（同一张图），统计曝光已应用之后的颜色。
@group(7) @binding(0) var out_tex_src: texture_2d<f32>;
/// 状态 + 直方图合一（word 布局，f32 一律 bitcast 存取）：[0] 曝光系数、[1] 已适应 EV、
/// [2] 备用（手动 EV 偏移）、[3] 帧间隔 dt（秒，CPU 每帧写）、[4] 直方图百分位平均 log2 亮度（调试）、
/// [5] 目标 EV（调试）、[8..72) 64 桶 log2 亮度直方图（本帧计数，`eye_adapt_update` 末尾清零）。
@group(7) @binding(1) var<storage, read_write> eye_adapt: array<atomic<u32>>;
