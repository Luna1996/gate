import package::bindings::{b_palette, light_u, material_assets, pbr_albedo_rough, pbr_metal, pbr_samp, view_u};

// --- 常量区（与 Rust wgsl_consts mod 字节对齐）---
// Brick Tree：256³ chunk，4³ 分裂因子，4 层（256→64→16→4→1）
const CHUNK_SIZE: u32 = 256u;
const MAX_LEVEL: u32 = 4u;
const NODE_FIXED_WORDS: u32 = 3u;    // mask_lo + mask_hi + palette
// b_struct Region ①：稠密 chunk 窗口。
const CHUNK_INDEX_CAP: u32 = 64u;

// --- 材质索引格式（16 位，与 Rust `gate_voxel::PALETTE_BITS` / `LEAF_*` 同源）---
// 节点第 3 字：bit0..15 = uniform 子块色；bit16..31 = **叶块代表值**（M2：块内加权众数槽号，
// 非叶节点恒 0；只有 `trace.wesl::leaf_lod_pal` 读它，其余所有 uniform 色读点都 `& 0xFFFF`）；
// 叶父层 inline 每字装 2 个体素（LEAF_VOXELS_PER_WORD）。
// 权威定义在 `wire.rs`，改位宽须同步全部读点。

// --- palette flags 位（与 Rust `gate_voxel::PaletteFlags` 同源；权威定义在 palette.rs）---
// bit0..3 = LOCKED / INPUT_PORT / OUTPUT_PORT / HOLOGRAM（着色不读），
// bit4 = IS_PBR（变体位）、bit5 = TRANSMISSIVE（DDA 热路径位，**写入侧维护**）、bit6..7 保留。
const FLAGS_IS_PBR: u32 = 1u << 4u;
const FLAGS_TRANSMISSIVE: u32 = 1u << 5u;
// ⚠️ **这两个是字节级掩码**（flags 字节 = `word1 >> 16 & 0xFF`）：只能与 `palette_flags(...)`
// 的结果、或自己先取出的 flags 字节相与。**直接与 palette 整字（`b_palette[i]`）相与会测到别的位**
// —— 平凡变体里那是 `emissive`/`transmission` 的位、PBR 变体里那是 `asset` 的位。

// --- 材质资产（PBR 变体的 `asset: u16` 所指的表）跨端常量 ---
// 权威值只写在这里：Rust 经 `wesl_consts.rs::material_consts` 解析同一份源码，缺一即启动 fail-fast。
// 只放 Rust 也需要知道的 `u32` 字面量；纯 shader 的常量（triplanar 采样尺度、混合锐度这类 f32）
// 留在用它的文件里即可（解析器只认 `u32` 字面量）。
/// 材质资产表的槽数上限：**全局一张表、所有 volume 共用**（`asset` 是全局下标，不是 per-volume 偏移），
/// 表大小 = 本值 × 32B/条目。PBR 变体的 `asset` 索引落在本表内。
const MATERIAL_ASSET_SLOTS: u32 = 1024u;
/// 贴图槽位（`texture_2d_array` 层）数目上限：`MaterialAsset` 的 `*_slot` 指向其中之一，
/// `0xFFFFFFFF`（`MATERIAL_SLOT_NONE`）表示"无贴图 ⇒ 走标量回退值"。
const MATERIAL_TEX_SLOTS: u32 = 64u;
/// 资产各 `*_slot` 的「无贴图」哨兵（= Rust `wire.rs::MATERIAL_SLOT_NONE`）：该通道退回
/// `albedo_rough` / `emissive_metal` / `transmission_ior` 里的标量回退值。
/// **0 不是哨兵**（层 0 可以是真实贴图）—— 判据只能是 `u32::MAX`。
const MATERIAL_SLOT_NONE: u32 = 0xFFFFFFFFu;

// 方向位掩码 LUT：3-bit 方向掩码（bit0=x>=0, bit1=y>=0, bit2=z>=0，共 8 种）→ side 与三轴步进 face id。
// 打包：bits0-2=side, bits8-15=face_x, bits16-23=face_y, bits24-31=face_z；face 规则 sign<0 穿对面。
const DIR_LUT: array<u32, 8u> = array<u32, 8u>(
  0x05030100u,
  0x05030001u,
  0x05020102u,
  0x05020003u,
  0x04030104u,
  0x04030005u,
  0x04020106u,
  0x04020007u,
);

// ---- face 索引（0..5 = -x,+x,-y,+y,-z,+z）与法线的转换 ----
fn face_normal_from_index(f: u32) -> vec3<f32> {
  if (f == 0u) { return vec3<f32>(-1.0, 0.0, 0.0); }
  if (f == 1u) { return vec3<f32>( 1.0, 0.0, 0.0); }
  if (f == 2u) { return vec3<f32>(0.0, -1.0, 0.0); }
  if (f == 3u) { return vec3<f32>(0.0,  1.0, 0.0); }
  if (f == 4u) { return vec3<f32>(0.0, 0.0, -1.0); }
  if (f == 5u) { return vec3<f32>(0.0, 0.0,  1.0); }
  return vec3<f32>(0.0);
}

// ---- sRGB → linear 转换（palette/sky 均以 sRGB u8/255 存储，光照前必须转 linear） ----
fn srgb_channel_to_linear(c: f32) -> f32 {
  if (c <= 0.04045) {
    return c / 12.92;
  }
  return pow((c + 0.055) / 1.055, 2.4);
}

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(
    srgb_channel_to_linear(c.x),
    srgb_channel_to_linear(c.y),
    srgb_channel_to_linear(c.z),
  );
}

// ---- palette albedo 解包（sRGB → linear）----
/// 从「`rgb` 三字节 sRGB 编码 + 高字节标量」这样的 word 里取出 rgb 并转 linear。
/// palette 纯色（平凡变体 word0）与 PBR 资产的标量回退值（`MaterialAsset::albedo_rough` **同布局**）
/// 共用它 ⇒「sRGB 编码字节 → linear」全工程只有一条口径（这正是平凡与 PBR 能共用同一个 BRDF
/// 入口的前提，见 MT3-2 / `docs/PLAN.md` 硬约束 1）。
fn srgb_word_to_linear(w: u32) -> vec3<f32> {
  let srgb = vec3<f32>(
    f32(w & 0xFFu),
    f32((w >> 8u) & 0xFFu),
    f32((w >> 16u) & 0xFFu),
  ) / 255.0;
  return srgb_to_linear(srgb);
}

/// palette 条目的 rgb（linear），供不走着色链的平色输出使用（相机陷在实心体素内的异常态）。
/// PBR 变体下 word0 的 rgb 是标量回退 albedo（布局同 `MaterialAsset::albedo_rough`）⇒ 两变体同一条读法。
fn palette_rgb_linear(palette_base: u32, pal: u32) -> vec3<f32> {
  return srgb_word_to_linear(b_palette[palette_base + pal * 2u]);
}

/// palette 条目的 flags 字节（word1 bit16..23，`flags << 16` 同布局）。
/// 变体位与介质位都从这里读，与 transmission 同在 word1 ⇒ DDA 内读它不额外 load 一个字
/// （`medium_of` 逐体素调用，成本敏感）。位的权威定义在 `gate_voxel::PaletteFlags`。
fn palette_flags(palette_base: u32, pal: u32) -> u32 {
  return (b_palette[palette_base + pal * 2u + 1u] >> 16u) & 0xFFu;
}

// 天空纯色（miss 像素与探针射线 miss 端点共用）。
// 输入 sRGB，返回 linear。
fn sky_rgb() -> vec3<f32> {
  return srgb_to_linear(light_u.sky_color.xyz);
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
// MT3 · 材质取值 + metallic-roughness 着色（**唯一着色入口**）
// `docs/PLAN.md` §4 MT3-1 ~ MT3-8；字节布局与 F0 规则的权威在 `§3 D1` 与 `wire.rs`。
//
// D1 的约束 1（"必须遵守的三条"第 1 条）：`IS_PBR` 标志**只决定参数从哪来**、**不决定走哪条
// 着色路径** ⇒ 全工程只有两个入口：
//   · `fetch_material(...)`：唯一取值入口（平凡 = 参数内联；PBR = 资产表 + triplanar 采样）；
//   · `brdf_reflected(...)`：唯一 BRDF 求值函数（两个变体 + 主 pass 与 GI 二次顶点共用）。
// 出现"平凡走一条、PBR 走另一条"的着色代码即违反硬约束 1 / 3。
// MT4-1 起多了一个**调用点**（不是新公式）：主 pass 的**镜面反射命中点**也用这两个函数着色
// （`main.wesl` 的 `dda_main`，以及 `trace_glass` 里原先那条独立近似 —— 它已按 MT4-1 的必做项
// 接进本入口）；三个着色点的差异只在 `SurfaceLight` 的字段（那是物理上该不同的部分）。
//
// MT8（`docs/PLAN.md §4b`：逐体素材质采样）在本入口上加了两个**输入量化器**（不是第二条取值路径）：
//   · `material_sample_pos`：纹理采样点 + 解析 LOD 量化到**体素格** ⇒ 一个体素面一个平色（MT8-1）；
//   · `face_view_dir`：视向量量化到**体素中心** ⇒ Fresnel / GGX 高光 / 反射方向逐面恒定
//     （MT8-2 曾这么做并回退；**2026-09-21 起只保留给反光/高光**，见本文件「视向量」那一段）。
// `material_sample_pos` 只有**一份**实现、且被四个着色点共用（硬约束 3：口径不分叉）。
//
// ⚠️ **MT8-2 全线量化被用户实测推翻过**（历史，必须读）：视向量量化到体素中心 ⇒ ① 掠射面的
// Fresnel 整面均匀变亮（画面里就是"几乎所有物体的轮廓都有一条白边"）；② 反射方向逐面恒定 ⇒
// 一个面只剩一个反射值 ⇒ **镜面图像在原理上不可能出现**（用户实测：光滑度拉到 100 也看不到镜面）。
// 2026-09-21 用户复查"还有不是逐体素着色的部分，重点调查反光、高光" ⇒ 这次**只**把反光/高光
// 按面量化（`face_view_dir`；本工程没有别的量读 `v`），上述两条代价**已知并接受**；
// 其中 ② 的实测发生在 MT8-3 陈旧缓存还在时（缓存已整体删除）⇒ 需人工重测，见 `face_view_dir`。
// MT8-1（`material_sample_pos`）**不受影响、一行未动** —— 用户已确认"现在材质是按逐体素着色的"。
// ============================================================================

/// **MT8-1 的"甲/乙"美学开关**（`docs/PLAN.md §4b` MT8-1 / §5 R14）。
/// `1u` = **甲**（严格 1 采样/体素：纹理采样点量化到体素中心 ⇒ 一个体素面一个平色，色块边界与几何
/// 台阶对齐）、`0u` = **乙**（退回 MT3 的逐屏幕像素连续采样，亚体素贴图细节保留）。
/// **默认取"甲"**：用户 2026-09-21 的决策（`§4b` 决策行"A = 甲"）—— 逐体素着色风格是体素引擎的身份，
/// 逐像素采样会把 2cm 体素面铺满 ~10 倍于它的贴图细节，观感退化成"贴了高分辨率贴图的网格模型"。
/// 这个选择的两个代价都在计划里登记了：
///   · **R12** 量化后远处（1 体素 < 1 像素）色块会逐帧闪烁 ⇒ 靠 `pbr_mip_lod` 的解析 LOD 把采样格
///     随像素足迹**一起变粗**（量级论证写在 `pbr_mip_lod` 的注释里）；
///   · **R14** 高光/反射随视点**逐面跳变** ⇒ "甲"的预期代价，不是 bug。
/// 编译期常量 ⇒ `0u` 时 `material_sample_pos` 里的 `select` 折成恒取连续点，整支零开销
/// （WESL 是运行期读盘编译：改这一行 + 重启即生效，不用 rebuild）。
///
/// **作用范围（别误解）**：本常量只切 MT8-1 的**采样点**量化。视向量的按面量化是另一个开关位
/// （`face_view_dir`，**没有**开关、恒生效），它只管反光/高光那一族量；依赖它做"逐位复用"的
/// MT8-3（按可见体素面复用反射）已被实测判为负优化并整体删除。
/// **本常量（MT8-1）不受那两次回退影响**：它只管"贴图采样点取在体素中心"这一件事。
/// **只影响有贴图的 PBR 变体**：平凡变体（palette 纯色）根本不调 `triplanar_sample` ⇒ 逐位不变
/// （取证见 `material_sample_pos` 的注释）。
const MATERIAL_FLAT_SHADING: u32 = 1u;

/// 一张贴图铺多大世界范围（米/张贴图）—— **MT3 的旋钮**（MT2-4 的 `PBR_DEBUG_TEX_WORLD_SCALE` 转正）。
/// triplanar 走**世界坐标**（与体素尺寸解耦）⇒ 本值与 `pbr_texture.rs::GPU_TEX_SIZE` 一起决定 texel 密度：
/// `texel/体素面 = GPU_TEX_SIZE / (本值 × MATERIAL_VOXEL_PER_METER)` = 128 / (2 × 50) = **1.28**（MT8-4 后）。
/// 依据 `docs/PLAN.md §7 决策记录`（2026-09-20「贴图分辨率档」）的核算：2cm 体素、1080p 下每体素面
/// 约 5~11 屏幕像素 ⇒ **每体素面 1.28 texel 已够**（MT8-1 之后一个体素面只取 1 个采样点，
/// 再多的 texel 也只是被丢掉的亚体素细节）。
/// 默认 2.0 = 那份核算的上沿（"偏软但绝不产生摩尔纹"）；嫌糊就调小（1.0 ⇒ 每体素面 2.56 texel）。
const MATERIAL_TEX_WORLD_SCALE: f32 = 2.0;
/// 1 米 = 多少体素（本工程**世界单位 = 体素**）—— 与 Rust `gate_app::consts::VOXEL_PER_METER` 同源。
const MATERIAL_VOXEL_PER_METER: f32 = 50.0;
/// triplanar 三轴混合的锐度 k：权重 = `pow(|n|, k)`（k 越大越接近"只取主轴投影"）。
/// MT3-1 的验收是"贴图在体素面上无接缝 / 无拉伸"，而本工程的着色法线只落在固定的几档角度上
/// （轴对齐 / 凸棱 45° / 凸角 54.7°，见 `world.wesl::voxel_normal`）：
///   · 在"各轴分量相等"的斜法线上 k **没有作用**（权重比恒为 1 : 1）⇒ 45° 与凸角都是干净的等权混合，
///     不会出现"三轴切换的硬边"（接缝的主要来源被结构性消掉）；
///   · k 只影响"某轴略占优"的中间角度（本工程几何给不出这种法线，但 MT6 位移后的邻域梯度可能给）
///     ⇒ 调大 = 更接近单轴取样（过渡带更硬）、调小 = 更软（斜面有轻微重影）。
/// 默认 4.0：既看不出接缝，又保留混合的平滑性。
const MATERIAL_TRIPLANAR_SHARPNESS: f32 = 4.0;
/// 粗糙度的数值下限：低于本值就与它同观感。两个理由：
/// ① `α = rough` 若为 0，`ggx_peak` 的 `α⁴ / (n·h²(α²−1)+1)²` 在 `n·h = 1` 处是 `0/0 = NaN`；
/// ② α 越小高光越窄，窄到一个像素以下时逐像素采样会闪烁（真正的镜面反射是 MT4 的反射射线）。
const MATERIAL_MIN_ROUGHNESS: f32 = 0.05;
/// 平凡变体的 IOR：**没有 IOR 字段**（D1 的字节布局里只有 PBR 变体存 IOR）⇒ 恒取 1.5
/// （= 改动前玻璃硬编码的那个值，MT3-8 逐位回归的锚点）。
const MATERIAL_DEFAULT_IOR: f32 = 1.5;
/// `fetch_material` 的 `debug_asset` 实参：**不强制任何资产**（走条目自己的变体）。
/// 值与 `main.wesl` 的 `PBR_DEBUG_ASSET` 的"关闭"值相同，但语义不同：那个是"总开关关着"，
/// 这个是"这个调用点不受调试通道影响"（GI 侧用它 ⇒ 调试通道只作用于主 pass）。
const MATERIAL_DEBUG_NONE: u32 = 0xFFFFFFFFu;

/// 材质参数（`docs/PLAN.md` §4 MT3-2）：**着色侧唯一的数据形态**。
/// 平凡变体与 PBR 变体产出的是同一个结构、差别只在字段从哪来 ⇒ 下游不可能出现两条着色路径。
struct MaterialParams {
  albedo: vec3<f32>,   // 已转 linear（palette 字节 / 贴图 rgb 都走 `srgb_to_linear`）
  roughness: f32,      // 0 = 镜面 … 1 = 完全粗糙
  metallic: f32,       // 0 = 电介质 … 1 = 金属（8 bit 连续量：贴图驱动下是磨损 / 锈迹的过渡）
  emissive: f32,       // 标量；沿用现有口径 `albedo · emissive · gain`（发光体素是光源本体，不吃方向 / 阴影）
  transmission: f32,   // 透射率：**着色不消费**（介质判定走 `FLAGS_TRANSMISSIVE` 位，见 `trace.wesl`）；
                       // **MT5-3 后仍无消费者** —— 介质吸收在 DDA 内直读 palette（成本理由见
                       // `trace.wesl::medium_of`），这里只是"如实带出"（PBR 变体取资产标量值）备用
  ior: f32,            // 资产级物理基值：同时服务玻璃折射（`trace_glass`）与电介质 F0（`dielectric_f0`）
  specular: f32,       // 槽级调制：**只乘性调制电介质 F0**（1.0 = 不调制），对金属无效（D1）
}

/// **MT8-1 · 采样点量化到体素格**（`docs/PLAN.md §4b` MT8-1）：把命中点的连续世界坐标换成**该体素
/// 中心**（`floor(p) + 0.5`）⇒ 同一体素面内所有像素取到**同一个**纹素 ⇒ albedo / roughness /
/// metallic / transmission 一个体素面一个平色，色块边界与几何台阶对齐（用户决策"A = 甲"）。
///
/// **为什么它"就是 1 采样/体素"**（量级，别凭感觉）：本工程**世界单位 = 体素**
/// （`MATERIAL_VOXEL_PER_METER = 50` ⇒ 2cm 体素）⇒ 世界空间里"整数面"**就是**体素边界，
/// `floor(p) + 0.5` 就是体素中心；而一张贴图铺 `MATERIAL_TEX_WORLD_SCALE = 2m = 100 体素`、
/// 贴图边长 `GPU_TEX_SIZE = 128`（MT8-4）⇒ 128/100 = **1.28 texel/体素**：一个体素面只落 1 个采样点，
/// 贴图分辨率与体素格**同量级** ⇒ 亚体素的贴图细节被有意丢弃（这就是"甲"的定义，不是精度损失）。
///
/// **坐标系前提（必须写明）**：量化在**世界空间**做，所以"世界整数 = 体素边界"只在该 volume 的变换为
/// identity 时成立 —— 这恰好是自洽的选择：triplanar 的 UV 本来就是世界坐标，量化格因而与世界 UV 格
/// 对齐。带 `VolumeTransform`（旋转/缩放/偏移）的物体上，量化格对齐的是**世界格**、不是物体格
/// （面内仍是平色，只是色块边界与物体自己的体素台阶错开）。当前 demo / castle 场景全是 identity
/// （`Volumes::add_object` 在应用里没有调用点）⇒ 无实际影响。
///
/// **法线轴分量**：三个轴一起 `floor + 0.5`。UV 只用两个切向轴（见 `triplanar_sample`），
/// 被丢掉的那个分量量化与否都不进 UV ⇒ 与"只量化两个切向轴"逐位等价（任务口径里的"法线轴不用管"）。
///
/// **平凡变体不经过这里**：调用点只有 `fetch_material_asset` 的"有贴图"分支（`fused != MATERIAL_SLOT_NONE`）
/// —— 量化一次、结果同时喂给 `triplanar_sample` 与 `pbr_mip_lod`（两侧口径一致）；
/// 平凡变体在 `fetch_material` 里走**参数内联**那条 `return`，从不触碰贴图路径（`material_ior` 同）。
fn material_sample_pos(p_world: vec3<f32>) -> vec3<f32> {
  // 编译期常量 ⇒ `MATERIAL_FLAT_SHADING = 0` 时 select 折成恒取 `p_world`（零开销，退回逐像素）。
  return select(p_world, floor(p_world) + vec3<f32>(0.5), MATERIAL_FLAT_SHADING != 0u);
}

/// 解析式 LOD（MT2-4 建立、MT3-1 抽成公共函数 —— **只有这一份**）。
/// compute pass 里**没有隐式 LOD**（`textureSample` 的导数版本只存在于 fragment stage）
/// ⇒ 只能用 `textureSampleLevel` + 自算 LOD。取"本像素在命中面处覆盖的贴图 texel 数"的 log2：
///   `dist(体素) × px_ang(rad) = 世界尺度(体素) → /50 = 米 → /WORLD_SCALE = 贴图数 → ×texel 边长 = texel 数`
/// 这是 triplanar 的标准解析 LOD 近似（与硬件在 fragment 里用屏幕导数算的同一量）。
/// **不给 0.0**（那样永远只采 mip0 ⇒ 远处必闪摩尔纹，正是 MT2-3 要消灭的现象）。
///
/// **MT8-1 起 `p_world` 先经 `material_sample_pos`**：这样同一体素面内的 LOD 也**恒定** ⇒
/// "逐面平色"是精确的而不是近似的（否则 mip 混合权重会随像素在面内微动，"平色"退化成"几乎平色"）。
/// 量化只把距离挪 ≤ 0.87 体素（体素中心 ↔ 面上命中点的最大距离），远小于 LOD 随距离的变化尺度
/// （`dLOD/ddist = 1/(dist·ln2)`）⇒ 对远处的 mip 选择没有可观察影响。
///
/// ---- **R12 核对（MT8-4，必读）：公式够不够抹平远处体素差异** ----
/// 量化到体素格后，远处"1 体素 < 1 像素"时相邻像素落到不同体素中心 ⇒ 若贴图还带亚体素细节就会
/// **逐帧色块闪烁**。本公式挡得住，量级论证（`GPU_TEX_SIZE = 128`、100 体素/张贴图 ⇒ 1.28 texel/体素）：
///   · 令 `texels` = 像素足迹的 texel 数，本函数取 `LOD = log2(texels)` ⇒ `2^LOD = texels`，
///     即 **mip 纹素的世界边长恰好 = 该像素的世界足迹**（教科书 LOD 的定义）；
///   · "体素投影 < 1 像素" ⇔ 足迹 > 1 体素 ⇔ `texels > 1.28` ⇔ **`LOD > log2(1.28) = 0.36`**
///     ⇒ 此刻 mip 纹素已 **≥ 1 体素**（`1.28 > 1` 正是这件事的意义：mip0 的纹素已 ≈ 0.78 体素，
///     mip1 的 2×2 盒 = 1.56 体素 > 1 体素）⇒ 相邻两个体素中心的采样落进同一个/相邻的 mip 纹素，
///     二者之差是"以体素为单位的盒式平均"之后的量 ⇒ **体素间的亚体素差异被抹掉**；
///   · 更远处的残余差异 ≤ `1 体素 / 足迹` × mip 级对比度（足迹 2 体素时 ~50%、16 体素时 ~6%），
///     单调衰减 + Linear mipmap 平滑 ⇒ 不会出现逐帧跳变的摩尔纹；剩下的"体素级色块抖动"就是"甲"
///     本身的分辨率（1 体素 < 1 像素时图像本来就是欠采样的），不是贴图混叠。
///   ⇒ **公式不用改**（MT2-4 建立、MT3 抽成唯一一份），MT8-4 只把贴图 1024 → 128：
///     `texels` 里的 `textureDimensions()` 自动跟随尺寸 ⇒ LOD 整体降 3（log2 8）。这不是"变糊了"，
///     而是"贴图不再 8× 过采样"（一个体素面本来就只取 1 个采样点）。
/// 已知的**既有**近似（MT2-4 就有，本次不改）：足迹只按**距离**算，没有按掠射角 `1/cosθ` 把足迹
/// 拉长（斜面上足迹是各向异性的）⇒ 掠射面略偏锐；缓解是采样器的 `anisotropy_clamp = 8`（MT2-3）。
/// `p` = **已量化的采样点**（= `material_sample_pos(p_world)`，与 `triplanar_sample` 共用同一个点）。
fn pbr_mip_lod(p: vec3<f32>) -> f32 {
  // MT8-1：距离从**体素中心**量 ⇒ 同一体素面内 LOD 恒定。
  let dist_voxel = length(p - view_u.cam_pos_voxel.xyz);
  let footprint_m = dist_voxel / MATERIAL_VOXEL_PER_METER * max(view_u.lod.x, 1e-6);
  let texels = footprint_m / MATERIAL_TEX_WORLD_SCALE * f32(textureDimensions(pbr_albedo_rough).x);
  return max(log2(max(texels, 1.0)), 0.0);
}

/// triplanar 的一次取值（三轴加权累加后的结果）。
struct TriplanarSample {
  albedo: vec3<f32>,   // linear
  roughness: f32,
  metallic: f32,
}

/// 单轴取样（三支共用这一份代码；权重在调用方乘）：
/// `albedo_rough` 一次采样同时给出 albedo（rgb，sRGB 编码 ⇒ 转 linear）与 roughness（a），
/// `metal` 再取一次 metalness（r）。`slot` 是**融合槽**（打包布局的后果，见 `fetch_material_asset`）。
fn triplanar_axis(slot: u32, uv: vec2<f32>, lod: f32) -> TriplanarSample {
  let s = textureSampleLevel(pbr_albedo_rough, pbr_samp, uv, i32(slot), lod);
  let met = textureSampleLevel(pbr_metal, pbr_samp, uv, i32(slot), lod).r;
  return TriplanarSample(srgb_to_linear(s.rgb), s.a, met);
}

/// triplanar 采样（MT3-1 / D3 = B）：按 `|n|` 三轴投影**加权混合**（不是 MT2-4 那个"只取主轴"的简化版）。
/// `p` = **已量化的采样点**（= `material_sample_pos(p_world)`）—— 因为量化后同一体素面内 UV 恒定：
///   · 三轴取样点、混合权重（权重只由 `n` 决定，本来就逐面恒定）、LOD（见 `pbr_mip_lod`）全都逐面恒定
///     ⇒ **albedo/roughness/metallic 一个体素面一个平色**；
///   · `MATERIAL_FLAT_SHADING = 0` 时 `material_sample_pos` 折成恒取连续命中点 ⇒ 退回 MT3 那条逐屏幕
///     像素的连续采样（亚体素细节保留、但一个体素面上会铺 1.28 texel/体素的细节 ⇒ MT3 的观感）。
/// 量化为什么等价于"只量化两个切向轴"、坐标系前提、平凡变体不受影响：见 `material_sample_pos`。
/// 三轴共用同一个世界尺度 ⇒ 贴图在三个方向上的 texel 密度一致（不会出现某一轴被拉伸）。
fn triplanar_sample(slot: u32, p: vec3<f32>, n: vec3<f32>, lod: f32) -> TriplanarSample {
  let w = pow(abs(n), vec3<f32>(MATERIAL_TRIPLANAR_SHARPNESS));
  let uv_scale = 1.0 / (MATERIAL_TEX_WORLD_SCALE * MATERIAL_VOXEL_PER_METER);
  let sx = triplanar_axis(slot, p.zy * uv_scale, lod);
  let sy = triplanar_axis(slot, p.xz * uv_scale, lod);
  let sz = triplanar_axis(slot, p.xy * uv_scale, lod);
  let inv = 1.0 / max(w.x + w.y + w.z, 1e-6);
  return TriplanarSample(
    (sx.albedo * w.x + sy.albedo * w.y + sz.albedo * w.z) * inv,
    (sx.roughness * w.x + sy.roughness * w.y + sz.roughness * w.z) * inv,
    (sx.metallic * w.x + sy.metallic * w.y + sz.metallic * w.z) * inv,
  );
}

/// PBR 变体槽级覆盖的解码：`0` = **不覆盖**（返回 -1.0 哨兵）、`1..255` = `(v−1)/254` ∈ [0, 1]
/// （`1` 因此能表达"完全镜面"这种 0 值，而不与"不覆盖"撞码 —— `docs/PLAN.md` D1）。
fn override_value(v: u32) -> f32 {
  return select(f32(v - 1u) / 254.0, -1.0, v == 0u);
}

/// 应用槽级覆盖：`ov < 0`（= 无覆盖哨兵）⇒ 用资产的值，否则用覆盖值。
fn apply_override(asset_value: f32, ov: f32) -> f32 {
  return select(ov, asset_value, ov < 0.0);
}

/// PBR 变体的取值路径（`IS_PBR = 1`，`docs/PLAN.md` D1）。
/// `asset` = palette word1 低 16 位（**全局**资产表下标）；`ov0`/`ov1` = 该条目的两个 word
/// （调试通道传 `0u`/`0u` = 全不覆盖，见 `fetch_material`）。
///
/// 贴图未覆盖的通道走**标量回退值**（`MaterialAsset` 的后三个字，布局与平凡变体逐位相同）。
/// `emissive` / `transmission` 贴图**不进 GPU**（MT2 的 GPU 侧只有 `albedo_rough` / `metal` 两组）
/// ⇒ 这两项一律走标量；`*_slot` 字段留给 MT7 的贴图集扩展（现在恒为 `MATERIAL_SLOT_NONE`）。
fn fetch_material_asset(asset: u32, ov0: u32, ov1: u32, p_world: vec3<f32>, n: vec3<f32>) -> MaterialParams {
  // `asset` 是 u16（恒 < 65536），但表大小是 `MATERIAL_ASSET_SLOTS` ⇒ 钳到表内：
  // 越界读在 wgpu 的 robust access 下会静默给 0（= "读到另一个材质"，比钳制更难查）。
  let a = material_assets[min(asset, MATERIAL_ASSET_SLOTS - 1u)];
  let sc_albedo = srgb_word_to_linear(a.albedo_rough);
  let sc_roughness = f32((a.albedo_rough >> 24u) & 0xFFu) / 255.0;
  let sc_emissive = f32(a.emissive_metal & 0xFFu) / 255.0;
  let sc_metallic = f32((a.emissive_metal >> 8u) & 0xFFu) / 255.0;
  let sc_specular = f32((a.emissive_metal >> 16u) & 0xFFu) / 255.0;
  let sc_transmission = f32(a.transmission_ior & 0xFFFFu) / 255.0;
  let ior = f32(a.transmission_ior >> 16u) / 100.0;
  // **融合槽**：本工程的数组图是通道打包的（一层里 rgb = albedo、a = roughness，见 `pbr_texture.rs`）
  // ⇒ `albedo_slot` 与 `roughmetal_slot` 指向同一层（默认资产表里恒相等）。只按一个槽取值
  // （先 `albedo_slot`、它 = NONE 则用 `roughmetal_slot`）：一次采样同时拿到 albedo / roughness / metal，
  // 不为"两槽指不同层"这种打包布局下**不存在**的配置留第二条代码路径。
  let fused = select(a.roughmetal_slot, a.albedo_slot, a.albedo_slot != MATERIAL_SLOT_NONE);
  var albedo = sc_albedo;
  var roughness = sc_roughness;
  var metallic = sc_metallic;
  if (fused != MATERIAL_SLOT_NONE) {
    // 采样点只量化一次：`triplanar_sample` 取 UV、`pbr_mip_lod` 量距离，两者共用同一个体素中心。
    let p = material_sample_pos(p_world);
    let ts = triplanar_sample(fused, p, n, pbr_mip_lod(p));
    albedo = ts.albedo;
    roughness = ts.roughness;
    metallic = ts.metallic;
  }
  return MaterialParams(
    albedo,
    apply_override(roughness, override_value(ov0 & 0xFFu)),
    apply_override(metallic, override_value((ov0 >> 8u) & 0xFFu)),
    apply_override(sc_emissive, override_value((ov0 >> 16u) & 0xFFu)),
    apply_override(sc_transmission, override_value((ov0 >> 24u) & 0xFFu)),
    ior,
    apply_override(sc_specular, override_value((ov1 >> 24u) & 0xFFu)),
  );
}

/// **材质取值的唯一入口**（MT3-2）。两条分支**只是参数来源不同，不是两条着色路径**（MT3-7）：
/// 产出的 `MaterialParams` 完全同型，之后一律交给 `brdf_reflected`。
/// * **平凡变体**（`IS_PBR = 0`）：参数**内联** —— albedo / roughness / metallic / emissive /
///   transmission 直接读 palette 的两个 word（逐位布局见 `wire.rs::pack_palette_entry`）；
///   `ior` 取默认 1.5、`specular` 取中性 1.0（不调制）—— 平凡变体根本没有这两个字段。
/// * **PBR 变体**（`IS_PBR = 1`）：word1 低 16 位索引全局资产表 → triplanar 采样；
///   槽级覆盖（word0 的 4 个 + word1 bit24..31 的 specular）按 D1 的语义生效。
///
/// `debug_asset`（= `main.wesl` 的 `PBR_DEBUG_ASSET`，**编译期常量**）：≠ `MATERIAL_DEBUG_NONE` 时
/// **强制所有表面用资产 N**。MT2-4 的调试语义**保留**，但改由本函数统一实现 ⇒ 调试通道与正式路径
/// **共享同一条代码**（MT2-4 那条"单独的 albedo 直出路径"已删）。这是人工"看到 triplanar 贴图 /
/// 看到金属 BRDF"的唯一入口：`0u` = `brick_wall_001`（贴图走向）、`10u` = `metal_plate`（金属 BRDF）。
/// 注意它**抢走整份材质**（含 emissive）⇒ 开启时发光体不再发光，这是调试视图的预期、不是回归。
/// GI 侧（`gi/ray.wesl`）传 `MATERIAL_DEBUG_NONE`：调试通道只作用于主 pass 的着色（与 MT2-4 一致）。
fn fetch_material(palette_base: u32, pal: u32, p_world: vec3<f32>, n: vec3<f32>,
                  debug_asset: u32) -> MaterialParams {
  if (debug_asset != MATERIAL_DEBUG_NONE) {
    return fetch_material_asset(debug_asset, 0u, 0u, p_world, n);
  }
  let w0 = b_palette[palette_base + pal * 2u];
  let w1 = b_palette[palette_base + pal * 2u + 1u];
  // flags 字节就在 `w1` 里（bit16..23）⇒ 直接取，不再多读一次 palette（同 `palette_flags`）
  if (((w1 >> 16u) & 0xFFu & FLAGS_IS_PBR) != 0u) {
    return fetch_material_asset(w1 & 0xFFFFu, w0, w1, p_world, n);
  }
  return MaterialParams(
    srgb_word_to_linear(w0),
    f32((w0 >> 24u) & 0xFFu) / 255.0,
    f32((w1 >> 24u) & 0xFFu) / 255.0,
    f32(w1 & 0xFFu) / 255.0,
    f32((w1 >> 8u) & 0xFFu) / 255.0,
    MATERIAL_DEFAULT_IOR,
    1.0,
  );
}

/// 材质的 IOR（MT3-8）：PBR 变体读资产的 `ior_x100`（`transmission_ior` 高 16 位，玻璃 1.5 → 150）；
/// 平凡变体**没有 IOR 字段** ⇒ 返回默认 1.5 ⇒ `trace_glass` 的折射与 Fresnel 与改动前一致
/// （这正是 MT3-8 的回归判据）。
/// 只读 palette + 资产表、**不采样贴图**（DDA 的玻璃循环里逐界面调用，成本敏感）。
fn material_ior(palette_base: u32, pal: u32) -> f32 {
  if ((palette_flags(palette_base, pal) & FLAGS_IS_PBR) == 0u) {
    return MATERIAL_DEFAULT_IOR;
  }
  let asset = b_palette[palette_base + pal * 2u + 1u] & 0xFFFFu;
  return f32(material_assets[min(asset, MATERIAL_ASSET_SLOTS - 1u)].transmission_ior >> 16u) / 100.0;
}

/// 电介质 F0 = `((IOR−1)/(IOR+1))²` × `specular` 调制 —— **全工程唯一的 F0 编码**
/// （`docs/PLAN.md` 硬约束 9 / D1「F0 的唯一来源规则」）。
/// 玻璃的 Fresnel（`trace_glass`）与 BRDF（`f0_of`）都从这里取；改动前那两份硬编码
/// （玻璃的 IOR 与 F0 各一个常量）已删除（MT3-8）。
/// `specular` 是**乘性调制**（语义同 glTF `KHR_materials_specular` 的 `specularFactor`：
/// 1.0 = 不调制 = 用 IOR 导出的物理值），**对金属无效**（金属的 F0 是 albedo，见 `f0_of`）。
fn dielectric_f0(ior: f32, specular: f32) -> f32 {
  let r = (ior - 1.0) / (ior + 1.0);
  return r * r * specular;
}

/// D1「F0 的唯一来源规则」落到一行：金属取 `albedo`、非金属取 `dielectric_f0`、
/// `metallic ∈ (0,1)` 时两条按 metallic **线性混合**（与 BRDF 的 `kD = 1 − metallic` 同一个权重
/// ⇒ 过渡无跳变）。
fn f0_of(m: MaterialParams) -> vec3<f32> {
  return mix(vec3<f32>(dielectric_f0(m.ior, m.specular)), m.albedo, m.metallic);
}

/// 峰值归一的 GGX 半程向量分布：`π·α²·D_ggx`（`α = roughness`），峰值恒为 1（`n·h = 1`）、
/// `α = 1` 时恒为 1（完全粗糙 = 各向同性）。
/// **为什么不用教科书里的 `D_ggx`**：`D_ggx` 带 `1/(π α²)` 因子，峰值随 α→0 发散 ⇒
/// "单向光 + 逐像素采样"下会炸出火点（rgba16f 的 GI 缓冲会直接溢出到 inf）。本工程整条着色链
/// **没有 1/π**（见 `brdf_reflected` 的能量约定），这里把 `π·α²` 一并折进函数、把高光**钳在 1 以内**：
/// 镜面项 ≤ `sun_c · F0`，与旧 Blinn-Phong（≤ `sun_c · albedo`）同量级 —— 这是"换模型不换亮度量级"的关键。
fn ggx_peak(n: vec3<f32>, h: vec3<f32>, roughness: f32) -> f32 {
  let a = max(roughness, MATERIAL_MIN_ROUGHNESS);
  let a2 = a * a;
  let ndh2 = clamp(dot(n, h) * dot(n, h), 0.0, 1.0);
  let d = ndh2 * (a2 - 1.0) + 1.0;
  return a2 * a2 / (d * d);
}

/// Schlick Fresnel 的一行式：`F0 + (1 − F0)·(1 − cosθ)⁵` —— **全工程唯一的 Fresnel 编码**
/// （与 `dielectric_f0` / `f0_of` 一起构成硬约束 9「F0 只有一个来源」的求值侧）。
/// 抽成函数是 MT4-1 的需要：反射权重也是一个 Fresnel（完美镜面下 h ≈ n ⇒ 余弦取 `v·n`），
/// 若在 `main.wesl` 里另写一遍就是**第二套 Fresnel 公式**（口径分叉，硬约束 3）。
/// `cos_theta` 传 `dot(h, v)`（直射高光，与改动前**逐位相同**：
/// `pow(clamp(1 − dot(h,v), 0, 1), 5)`）或 `dot(v, n)`（镜面反射的权重）。
fn fresnel_schlick(f0: vec3<f32>, cos_theta: f32) -> vec3<f32> {
  return f0 + (vec3<f32>(1.0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

// ============================================================================
// 视向量：**一份实现、两个名字**（`view_dir_to` = 公式；`face_view_dir` = 全工程唯一的锚点契约：
// **按体素面取**）。2026-09-21 用户复查"当前渲染链路里还有不是逐体素着色的部分，重点调查反光、高光"
// ⇒ 全部读 `v` 的量按面量化（MT8-2 曾这么做但被实测推翻，本次的代价与历史两次结论写在
// `face_view_dir` 的注释里 —— 改这里之前必须读完那两段）。
// ============================================================================

/// 视向量的**唯一实现**：`(射线起点 − 锚点)` 归一化；锚点由调用方给。
/// **全工程的调用点只有 `face_view_dir`**（锚点 = 命中体素的中心）—— 为什么要按面、代价是什么、
/// 历史两次实测结论都在它的注释里。
/// 退化兜底：起点与锚点重合（相机恰好落在命中体素中心、GI 起点与命中点重合）时用
/// `max(length, 1e-5)` 避免 NaN 扩散到整幅画面（与 `brdf_reflected` 里 `h` 的归一化同一手法）。
fn view_dir_to(ray_origin: vec3<f32>, hit_point: vec3<f32>) -> vec3<f32> {
  let d = ray_origin - hit_point;
  return d / max(length(d), 1e-5);
}

/// **逐面视向量**：第二个实参 = **命中体素的中心**（`world.wesl::voxel_center_world`；本工程世界单位
/// = 体素 ⇒ `floor(p)+0.5` 就是"这个面"的锚点）⇒ 同一体素面内的所有像素拿到**同一个** `v`。
///
/// ---- **为什么必须按面**（用户 2026-09-21 的复查结论）----
/// 本工程里**所有**读 `v` 的量都是反光/高光 —— 漫反射是 `kD·albedo·N·L`、环境与 GI 是
/// `(kD·albedo + F0)·(amb + gi)`，两者都不读 `v`（见 `brdf_reflected`）⇒ 视向量逐像素时，
/// 材质色是逐体素平色（MT8-1）而**光照却跨体素边界连续**：
///   · 太阳 GGX 高光 `h = normalize(sun_dir + v)` / `dot(h, v)`：平面上（同一 `n`）它是屏幕位置的
///     **连续函数** ⇒ 一条跨体素边界都不间断的平滑亮带；
///   · 环境反射方向 `reflect(-v, n)`：MT8-6 的抖动种子虽已按面，但基准方向逐像素时同一个面内仍各
///     采一个方向 ⇒ 反射内容连续（≈5.6° 的锥盖不住这个差异）；
///   · 反射的 Fresnel 权重 `dot(v, n)`：逐像素 ⇒ 掠射方向上一片单调变亮的连续光晕。
/// 按面之后这三项一个体素面一个值，与 MT8-1 的材质平色、MT8-6 的按面抖动种子**同格**。
///
/// ---- **调用点**（各传自己那条射线的起点 + 命中体素的中心）----
/// * 主 pass（`dda_main`）：相机眼位 → `p_voxel`（喂 GGX 高光 / `mirror` / 反射权重）；
/// * MT4 反射命中点的着色：`r_org` → 命中体素中心（该处 `sun_c = 0`、`sun_vis = 0` ⇒ `v` 不参与任何
///   项；按面只是为了"全工程一个口径"，不是那处有变化）；
/// * `trace_glass`：① 界面事件的 Fresnel 与反射方向 = `org` → 界面体素中心，② 玻璃反射命中点的
///   着色 = `r_org` → 命中体素中心。**`ci` 等光路几何量仍用精确方向**（那不是着色量，未动）；
/// * GI 二次顶点（`gi/ray.wesl`）：GI 射线起点 `org` → 命中体素中心。
///
/// ---- **已知代价（本档的预期行为，不是 bug）** ----
/// ① 掠射面**整面**变亮 —— 就是 MT8-2 时用户看到的"白边"（逐像素时它只是最外一列像素的渐变，
///    按面后整个面取面中心那个值）；
/// ② 反射变成**块状**（一个面一条方向 ⇒ 一个反射值）。
/// 当年"光滑度拉满也看不到镜面"的观察发生在 MT8-3 陈旧缓存（刷新 K=16 + 命中复用）还在时，
/// **该缓存已整体删除** ⇒ 本次应能看到"按体素格采样的镜面"，**待人工确认**（若仍看不到，说明
/// MT8-2 那条"逐面恒定 ⇒ 镜面不可能出现"的结论才是真的，那时必须回来重新决策）。
///
/// ---- **为什么锚点必须来自 DDA 的整数体素** ----
/// 命中点的法线轴分量正好落在整数面平面上，`floor(命中点)` 会因浮点误差取到相邻（通常是空气）
/// 体素 ⇒ 同一面内逐像素翻转。所以一律传 `voxel_center_world(网格, 命中体素)`，不许自己 floor。
fn face_view_dir(ray_origin: vec3<f32>, voxel_center: vec3<f32>) -> vec3<f32> {
  return view_dir_to(ray_origin, voxel_center);
}

/// 单点着色的光源项（主 pass 与 GI 二次顶点**同一套字段** —— 口径不分叉的前提）。
struct SurfaceLight {
  sun_dir: vec3<f32>,   // 归一化，指向光源
  sun_c: vec3<f32>,     // 太阳色 × 强度（线性）
  sun_vis: f32,         // 沿 sun_dir 的透过率（0 = 全遮 … 1 = 无遮）；调用方决定要不要发阴影射线
  /// 常量天光项（按各自的 `cov` 折算：主 pass 含 `mix(1, GI_AMBIENT_FLOOR, cov)`，
  /// GI 二次顶点拿全额 —— 二次顶点没有 GI 缓冲，这是**唯一**的项差异）。
  amb: vec3<f32>,
  gi: vec3<f32>,        // GI 辐照度（主 pass 有；二次顶点恒 0 —— 只算一次弹射）
}

/// **唯一的 BRDF 求值函数**（MT3-3 / MT3-7）：平凡变体与 PBR 变体、主 pass 与 GI 二次顶点全部走它。
/// `v` = 由着色点指向观察者（主 pass 与二次顶点都是 `-dir`：出射方向就是"回到上一个顶点"的方向）。
///
/// **能景约定（有意为之，别改成 1/π）**：本工程的直射漫反射一直是 `albedo · sun_c · N·L · vis`
/// —— **没有 1/π**（把"太阳辐照度"直接当"出射亮度"的量纲用）。MT3 保持这个口径：
///   · 漫反射 = `kD · albedo · sun_c · ndl · vis`，其中 **`kD = 1 − metallic`（择定）**：
///     ① 保证 metallic 0 → 1 时漫反射单调消隐、高光单调接管（过渡无跳变）；
///     ② 非金属下 `kD = 1` ⇒ 漫反射项与改动前**逐项相同**，MT1-4 的"平凡材质零回归"因此能保住。
///     （另一条候选 `(1−F)(1−metallic)` 会把电介质的漫反射压掉 4%，且多一次 Fresnel。）
///   · 高光 = `sun_c · F · D_peak · ndl · vis`（GGX 形状 + Schlick Fresnel，F0 驱动 / 粗糙度控宽 / 有界）。
///   · **引入 1/π 会把整场景压暗 π 倍** —— 那是本次没被要求的能量跳变，明确不做。
///   · 环境 / GI 项同样接上 metallic：`(kD · albedo + F0) · (amb + gi)`
///     —— 漫反射用同一套 `kD`、镜面用同一个 `F0`（不再引入第二套常量）。代价是电介质比改动前多了
///     `F0 · (amb + gi)`（F0 = 0.04 ⇒ 天光 / GI 约 +4%，物理上正是"环境光里那部分镜面反射"）。
///   · 电介质的高光从"albedo 量级"降到"F0 = 4% 量级"：这是 F0 唯一来源规则的直接后果
///     （旧 Blinn-Phong 拿 albedo 当高光色，物理上是错的）。
fn brdf_reflected(m: MaterialParams, n: vec3<f32>, v: vec3<f32>, l: SurfaceLight) -> vec3<f32> {
  let kd = 1.0 - m.metallic;
  let f0 = f0_of(m);
  var out = (kd * m.albedo + f0) * (l.amb + l.gi);
  let ndl = max(dot(n, l.sun_dir), 0.0);
  if (ndl > 0.0 && l.sun_vis > 0.0) {
    let hs = l.sun_dir + v;
    let h = hs / max(length(hs), 1e-5);
    let fres = fresnel_schlick(f0, dot(h, v));
    out = out + l.sun_c * (kd * m.albedo + fres * ggx_peak(n, h, m.roughness)) * (ndl * l.sun_vis);
  }
  return out;
}

/// 自发光项（MT3-6）：两个着色点共用**这一句**，沿用现有口径 `albedo · emissive · gain`
/// （`GI_EMIT_GAIN` 由调用方传，两个调用点都传同一个共享常量）。
/// **单独成函数而不是并进 `brdf_reflected`**：GI 侧要把**反射项**除以 π 转辐亮度，
/// 而自发光本来就是辐亮度、**不除**（见 `gi/ray.wesl` 文件头的约定）⇒ 两项必须分得开。
fn emissive_radiance(m: MaterialParams, gain: f32) -> vec3<f32> {
  return m.albedo * m.emissive * gain;
}
