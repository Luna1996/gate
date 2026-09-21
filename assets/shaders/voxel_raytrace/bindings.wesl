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
// **只给光照场用**（Rust 侧 desc：ClampToEdge ×3 + mipmap_filter = Nearest，该纹理无 mip 链）。
// PBR 贴图**不要复用它**（MT2-3 起）：triplanar 要 Repeat 平铺、要 Linear mipmap filter，
// 而光照场是"有界网格 + 世界锚定环绕"⇒ 改成 Repeat 会让越界查询绕回网格另一侧。
@group(1) @binding(5) var light_samp: sampler;

// --- BG1 追加（MT2-2，从 binding 6 起，不动 0..5）：全局材质资产表 + PBR 贴图集 ---
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
@group(1) @binding(6) var<storage, read> material_assets: array<MaterialAsset>;
// 采样 `pbr_albedo_rough` 的 rgb 后**必须**走 `srgb_to_linear()`：两张数组图都是 `Unorm` 格式、
// 采样不做 sRGB 转换（rgb 是 sRGB 编码的原始字节），与 `palette_albedo` 读 palette 字节后同样
// `srgb_to_linear()` 是**完全同一口径** ⇒ 平凡材质与 PBR 材质能共用同一个 BRDF 入口。
@group(1) @binding(7) var pbr_albedo_rough: texture_2d_array<f32>;  // Rgba8Unorm: rgb=albedo(sRGB 编码字节)、a=roughness
@group(1) @binding(8) var pbr_metal: texture_2d_array<f32>;         // R8Unorm: r=metalness
// @binding(9)：**PBR 贴图专用采样器**（MT2-3）。与 `light_samp` 分开是必须的 ——
// ① triplanar 要**三轴 Repeat**（光照场是 ClampToEdge 的有界网格）；
// ② 要 **Linear mag/min/mipmap**（光照场 mipmap_filter = Nearest、纹理本身无 mip 链）。
// 权威 desc 在 Rust `pbr_texture.rs::create_pbr_sampler`：Repeat×3 / Linear(mag,min,mipmap) /
// anisotropy_clamp ≤ 8（设备不支持 `SAMPLER_ANISOTROPY` 时退化 1）/ lod_max_clamp = mip 链底层。
// 改那边的状态必须同改这里（占位与真身共用同一个 sampler 实例，见 `upload.rs::init_empty_gpu`）。
@group(1) @binding(9) var pbr_samp: sampler;

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
  // x = **二次顶点太阳反弹**（1 = 开、0 = 关；菜单「渲染/RESTIR GI/太阳反弹」：
  // 关掉则二次顶点不做太阳 NEE，`sun_transmittance` 那条阴影射线整条不发）、
  // y = 保留（恒 0）、z = GI 增益、w = 保留（恒 0）
  params: vec4<f32>,
  // x = GI 开关、yzw = 保留（恒 0）
  misc: vec4<f32>,
  // x = 保留（恒 0）、y = GI 分辨率除数（1 = 全分辨率，2 = 半分辨率；整数值的 f32，只被
  // `gi_main` 用来把本 pass 的像素下标换成 beam 纹理下标）、
  // z = 世界几何相对「写入上一帧 reservoir 的那一帧」是否**逐位未变**（1 = 未变）⇒
  // 屏幕空间路径可跳过时域二次顶点键验证射线（见 `gi/screen.wesl` ① 与 `prepare_gi` 的 `world_rev`）、
  // w = **降噪质量档**（0 = 关 / 1 = 低 / 2 = 中 / 3 = 高；菜单「渲染/RESTIR GI/降噪质量」）：
  // 本 pass 只用它区分「高」档（1/4 档新鲜候选数与 reservoir 记忆窗按它取，见 `gi/consts.wesl`）；
  // 档位影响的 pass 派发（时域/atrous 开关、核半径）由 Rust 决定。
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

// ===== @group(4) @binding(1..3)：体积散射（godray）的**写侧**（只给 `fog_main` 的 pipeline）=====
// 为什么挂在 group(4)：它的组生命周期与雾 pass 一致（每帧重建、不跨 pass），而 1/2/3 是这一组
// **空着的号**（0 = `gi_u`、20/21 = reservoir）⇒ 沿用「binding 号在模块内唯一」的约定，不新开 group。
// 雾自己的 pass 用不到 `gi_u`（它要的相机矩阵/帧号在自己的 uniform 里，见 `FogUniform`）。
/// 雾的参数 uniform（与 Rust `volumetric::FogUniform` 逐字段镜像，字节一致）。
/// `prev_view_proj` 必须与 GI 的 `gi_u.prev_view_proj` **同一条契约**（上一帧真正跑过本 pass 的那一份），
/// 但**各自独立**：GI 关掉时它的 `prev_view_proj` 会停在最后一次跑 GI 的帧，雾不能借用。
struct FogUniform {
  /// x = `σ_t`（消光系数，1/voxel：`1/σ_t` = 一个 e-folding 的距离）、
  /// y = **地面雾的高度衰减** `h`（1/voxel：密度沿视线 `exp(−h·dir.y·t)`，相机高度处 ρ = 1；
  /// `0` = 均匀介质）、z = Henyey-Greenstein 各向异性 `g`（前向 = 正）、
  /// w = 天光环境散射的系数（乘 `sky_color`）。
  params: vec4<f32>,
  /// x = 雾网格的分辨率除数（整数 f32，只被 `fog_main` 用来把本 pass 的像素下标换成 beam 纹理下标）、
  /// y = 太阳锥的角半径（rad，见 `fog_cone_dir`）、z = 阴影射线的距离上限（voxel，唯一的成本闸门）、
  /// w = 每帧每像素的采样数 `S`（1..`FOG_SAMPLE_MAX`）。
  misc: vec4<f32>,
  /// x = **光柱增益**（乘在太阳的散射项上）、y = **雾距**（积分的距离上限，voxel）、
  /// z = **太阳角径**（rad：同时是太阳盘的半径与阴影射线的锥角 —— 两者在物理上是同一个量）、
  /// w = **日晕强度**（太阳盘外那圈拖尾的系数）。
  /// y（雾距）是「不要变成起雾」的关键：没有它，一条到远平面的射线（天空）会积到 `e^{-σt·∞}=0`
  /// ⇒ 天空与远景被雾色 **100% 替换**（整幅画面像起大雾）。截断之后透射率不再下降 ⇒
  /// 远处仍是它自己的颜色，只有雾距内的那一段贡献散射。
  misc2: vec4<f32>,
  /// x = 自增帧号（精确 u32；yzw 恒 0）：抖动的每帧独立样本靠它。
  seq: vec4<u32>,
  /// 上一帧的相机矩阵（与 GI 同一套时域重投影契约）。
  prev_view_proj: mat4x4<f32>,
}
@group(4) @binding(1) var<uniform> fog_u: FogUniform;
/// 雾的写入侧：rgba16f，rgb = 线性散射辐亮度、a 恒 1（**没有「无效像素」**，与 GI 的
/// premultiplied-valid 不同 ⇒ 采样侧不做 valid 归一化）。
@group(4) @binding(2) var fog_out: texture_storage_2d<rgba16float, write>;
/// 雾的降噪导引写侧（@group(0) @binding(21) / @group(6) @binding(3) 是同一块 buffer 的另外两个号）。
@group(4) @binding(3) var<storage, read_write> fog_guide_w: array<u32>;


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
/// 幂等）：同一个面被成千上万条射线打到时只算一次二次顶点着色。生命周期同样只有一帧
/// （Rust 每帧在 `gi_main` 之前整块清空）；它只被 `gi_main` 使用 ⇒ 只出现在 group(5)。
@group(5) @binding(9) var<storage, read_write> gi_sec_slots: array<atomic<u32>>;

// ===== @group(6)：体积散射（godray）的**读侧**（只给 `dda_main` 的 pipeline，见 `volumetric.wesl::fog_upsample_joint`）=====
// 为什么单开一组而不是塞进 group(5)：group(5) 是「GI 网格」的号段，而雾的网格尺寸、资源、生命周期
// 都与 GI 无关（分辨率档各自独立）⇒ 混在一组里会让两个模块的绑定互相牵制。这一组只有 `dda_main` 用，
// 逐面 pass / beam / GI 三个入口都不引用它（未用的绑定会被 naga 剪掉，layout 里不必为它们留号）。
// 绑定号取 0/1/3，是为了让「同一块 buffer 的多个号」在两侧都读得出来：
//   0 = 雾的参数（**与 @group(4) @binding(1) 是同一块 buffer**：`dda_main` 的天空要用它取太阳盘/日晕的
//       参数，而那条路径只有 group(6)。同一 buffer 两个声明、两个访问类型，与 `gi_guide_out`/`gi_guide_t`
//       的先例一致）
//   1 = 雾的采样视图（最终结果：降噪链的输出，或关降噪时的原始 `fog_out`）
//   3 = 雾的降噪导引（**只读**：上采样要按面键筛 2×2 texel）
// 关掉雾时 Rust 绑 1×1 的零纹理 + 4 字节占位 buffer（参数 uniform 照绑：太阳盘与雾的开关无关）。
@group(6) @binding(0) var<uniform> fog_u_read: FogUniform;
@group(6) @binding(1) var fog_tex: texture_2d<f32>;
@group(6) @binding(3) var<storage, read> fog_guide_r: array<u32>;

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
/// 20：降噪 pass 的**小配置**（每帧由 Rust 按菜单「高质量降噪」写入）：
/// `[0]` = atrous 单轮核半径（`GI_DEN_ATROUS_R` 或 `GI_DEN_ATROUS_R_FAST`）。
/// 为什么另开一个小 buffer 而不是读 @group(4) 的 `gi_u`：降噪两个 pass 的 layout **只有 group(0)**
/// 一份（刻意 —— 不拉进 brickmap / 相机矩阵），而「bind group 必须从索引 0 起成前缀设置」
/// ⇒ 它们够不到 group(4)。加一个 4 字节 storage 是最省的接法。
@group(0) @binding(20) var<storage, read> gi_den_cfg: array<u32>;

// ===== @group(0) @binding(21..)：体积散射（godray）的**时域累积**（`volumetric.wesl::fog_den_temporal`）=====
// 与 GI 降噪同一套做法（同一个 group、同样的"另起号段"理由：这个 pass 的 bind group 也只有 group(0)）。
// **雾的空间段（迭代 atrous）不在这里** —— 它直接复用 GI 的 `gi_den_atrous*` 入口与 layout
// （binding 10/14/15/17/20），Rust 只把雾的资源绑到那几个号上（见 `gate-render/src/volumetric.rs`）。
/// 21：**雾的降噪导引**的读侧（`fog_main` 通过 @group(4) @binding(3) 写同一块 buffer，
/// `dda_main` 通过 @group(6) @binding(3) 读同一块 —— 三个号三个访问类型，与 `gi_guide_out` 的先例一致）。
/// word 布局与 GI 的导引**逐字同构**（`GI_DEN_G_*`）：键 2 + 面法线 1 + 上帧重投影像素 1
/// —— 复用 `gi_den_same_plane` 判据与 atrous 入口的前提就是这套布局。
@group(0) @binding(21) var<storage, read> fog_guide_t: array<u32>;
/// 22：本帧原始雾（`fog_out` 的采样视图）。rgb = 线性散射辐亮度，a 恒 1（每个像素都有雾）。
@group(0) @binding(22) var fog_raw_in: texture_2d<f32>;
/// 23/24：雾的时域历史双缓冲（每像素 `FOG_DEN_HIST_WORDS` 个 word，布局见 `volumetric.wesl`）。
@group(0) @binding(23) var<storage, read> fog_hist_in: array<u32>;
@group(0) @binding(24) var<storage, read_write> fog_hist_out: array<u32>;
/// 25：时域累积的输出（atrous 第一轮的输入，`fog_dn[0]`）。
@group(0) @binding(25) var fog_tmp_out: texture_storage_2d<rgba16float, write>;
/// 26：每像素亮度 range 权重的 φ。雾自己写（语义与 GI 的 `gi_den_phi` 相同：供 `gi_den_atrous` 读），
/// 但**口径不同**：雾用「帧间差」而不是「同面邻域的亮度标准差」（见 `volumetric.wesl` 的说明）。
@group(0) @binding(26) var<storage, read_write> fog_phi_w: array<f32>;

// ===== @group(7)：眼睛适应（自动曝光）——被 `dda_main`（读曝光）与 eye_adapt_* 两个入口使用 =====
/// `out_tex` 的采样视图（同一张图），统计曝光已应用之后的颜色。
@group(7) @binding(0) var out_tex_src: texture_2d<f32>;
/// 状态 + 直方图合一（word 布局，f32 一律 bitcast 存取）：[0] 曝光系数、[1] 已适应 EV、
/// [2] 备用（手动 EV 偏移）、[3] 帧间隔 dt（秒，CPU 每帧写）、[4] 直方图百分位平均 log2 亮度（调试）、
/// [5] 目标 EV（调试）、[8..72) 64 桶 log2 亮度直方图（本帧计数，`eye_adapt_update` 末尾清零）。
@group(7) @binding(1) var<storage, read_write> eye_adapt: array<atomic<u32>>;
