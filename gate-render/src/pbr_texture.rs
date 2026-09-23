//! PBR 贴图基础设施（PLAN MT2-1 + MT2-1c）：把 `assets/textures/pbr/<id>/` 的 2D 贴图打包成 GPU `texture_2d_array`。
//!
//! **层号契约（与 MT1 的 `MaterialAsset` 对齐，不可单方面改）**：槽号 = 材质目录名在
//! `assets/textures/pbr/` 下按字典序排序后的下标；两张数组的**层号一一对应**
//! （第 i 层同属第 i 个材质）⇒ 资产侧只需存一个槽号。
//!
//! **通道打包（MT2-1c）**：源图是 Poly Haven 的 `Diffuse`（sRGB 基色）+ `arm`
//! （glTF ORM：R=AO / G=Roughness / B=Metalness），其中 AO 本工程不用（PLAN D1 属性清单：
//! 微观 AO 在体素粒度无意义，凹槽遮蔽由真实几何 + GI 提供）⇒ 直接丢弃，把它省下的宽度用来
//! 把 4 个有效通道挤进 1.25 张纹理：
//!
//! | 数组 | 格式 | 通道语义 | 单层 @128 |
//! |---|---|---|---|
//! | `albedo_rough` | `Rgba8Unorm` | rgb = albedo（**保持 sRGB 编码的原始字节**）、a = roughness（arm 的 G） | 64 KiB |
//! | `metal` | `R8Unorm` | r = metalness（arm 的 B） | 16 KiB |
//!
//! ⇒ **80 KiB/材质 @128**（MT2-1 的 `Rgba8UnormSrgb` + `Rgba8Unorm` 两张 4B/texel 数组是 128 KiB/材质），
//! 并随 [`GPU_TEX_SIZE`] 平方缩放；含 mip 链 ≈ 1.333× ⇒ **≈ 0.104 MiB/材质**（16 材质 ≈ 1.7 MiB）。
//!
//! **本文件负责"加载 + 打包 + mip 链 + 采样器 + 资产表内容"**：`texture_2d_array` 的占位资源与
//! 上传时机在 brickmap 侧 —— `upload.rs::init_empty_gpu` 建占位 buffer/纹理与采样器、`prepare`
//! 全量写一次资产表、`dda.rs` 把四者绑到 BG1 的 binding 5/6/7/8（MT2-2 / MT2-3）。
//! **mip 链与采样策略（MT2-3）就在本模块**：CPU 侧盒式平均生成完整链（base → 1×1，128 ⇒ 8 层），
//! 采样器见 [`create_pbr_sampler`]；`Image::data` 承载的是**全链数据**（见 `build_texture_arrays` 的说明）。
//! **不做 BC7/BC4 压缩**：MT8-4 把 [`GPU_TEX_SIZE`] 定成 128（= 体素格密度）后**预算问题消失** ——
//! 30 材质 ≈ 3 MiB，压缩（4:1）省下的 2 MiB 不值得引入离线 KTX2 工具链或新编码器依赖
//! ⇒ MT2-1c 留的那条 BC7 分支**取消**（见该常量的说明）。
//!
//! **height.png 完全不加载**：它只在 CPU 侧供 MT6 的 CSG 位移采样一次，位移产物就是普通体素 ⇒
//! 不进 GPU、不进本模块（见 PLAN D2 与硬约束 8）。
//!
//! 硬约束：**任何缺文件 / 尺寸不一致 / 格式不支持都只 `warn!` 后跳过该材质**（层号随后续材质顺延，
//! 保持 0..M-1 连续），**绝不 panic**。

use std::path::Path;

use bevy::{
  asset::RenderAssetUsages,
  image::{Image, ImageLoaderSettings},
  log::{info, warn},
  prelude::*,
  render::{
    extract_resource::ExtractResource,
    render_resource::{
      AddressMode, Extent3d, FilterMode, MipmapFilterMode, Sampler, SamplerDescriptor,
      TextureDimension, TextureFormat, TextureViewDescriptor, TextureViewDimension,
    },
    renderer::RenderDevice,
  },
};

use crate::brickmap::wire::{MATERIAL_SLOT_NONE, MaterialAsset};

/// `assets/` 下的 PBR 贴图集目录（相对路径，供 `AssetServer` 用；
/// 磁盘侧 = [`crate::paths::assets_dir`] 拼接 —— 绝不硬编码安装路径）。
pub const PBR_TEXTURE_DIR: &str = "textures/pbr";
/// 基色贴图后缀（Poly Haven `Diffuse`，sRGB）——打包后进 `albedo_rough` 的 **rgb**。
const ALBEDO_SUFFIX: &str = "_albedo.jpg";
/// 粗糙度 + 金属度贴图后缀（Poly Haven `arm`，glTF ORM 布局：R=AO / G=Roughness / B=Metalness）。
/// 打包时 **G → `albedo_rough` 的 a**、**B → `metal` 的 r**，R（AO）丢弃（见 PLAN D1 属性清单）。
const ROUGHMETAL_SUFFIX: &str = "_roughmetal.jpg";
/// 高度图后缀（MT8-6 起这里**只用它来判断"这个材质有没有高度图"**，以定默认位移幅度）。
/// 高度图本身仍**不进 GPU、不进本模块的数组图**（见模块头与硬约束 8）：它只在 `gate-app` 侧被
/// CSG 位移采样一次。这里查它是为了兑现"**没有高度图的材质不给非 0 位移幅度**"—— 否则 `gate-app`
/// 会去解一张不存在的图、白刷一条 `warn!`（见 [`has_height_map`]）。
const HEIGHT_SUFFIX: &str = "_height.png";
/// 槽号 ↔ 目录名日志每行条数（16 条一行太长，按行切）
const SLOT_LOG_PER_LINE: usize = 8;

/// MT2-4 调试通道的常量名与"关闭"值（权威定义在 `main.wesl`：`PBR_DEBUG_ASSET`）。
/// Rust 侧**不复刻它的值**，只按这个名字去 WESL 源码里读（见 [`log_pbr_debug_channel`]）。
const PBR_DEBUG_ASSET_CONST: &str = "PBR_DEBUG_ASSET";
/// `PBR_DEBUG_ASSET` 的"关闭"哨兵值（`u32::MAX`）——与 WESL 里那个字面量必须一致。
const PBR_DEBUG_ASSET_OFF: u32 = u32::MAX;
/// MT8-1 的"甲/乙"美学开关名（权威定义在 `common.wesl`：`MATERIAL_FLAT_SHADING`）——
/// 与上面同一个手法：Rust 只按名字去 WESL 源码读，不复刻它的值（见 [`log_flat_shading_switch`]）。
const MATERIAL_FLAT_SHADING_CONST: &str = "MATERIAL_FLAT_SHADING";

/// GPU 侧数组图**单层边长**（texel）——本模块唯一的显存预算旋钮，默认 **128**（MT8-4 定档）。
///
/// **为什么是 128（而不是原来的 1024）**：显存/带宽该由**真实采样密度**决定，而 MT8-1 之后
/// 一个体素面只取 **1 个**采样点（`MATERIAL_FLAT_SHADING = 1`，见 `common.wesl::material_sample_pos`）：
///   · 一张贴图铺 `MATERIAL_TEX_WORLD_SCALE = 2m`，而 `MATERIAL_VOXEL_PER_METER = 50` ⇒ **100 体素/张**；
///   · 128 / 100 = **1.28 texel/体素 ≥ 1** ⇒ 贴图分辨率与体素格**同量级**（再高的 texel 密度只是
///     "每体素面只取 1 次"时被丢掉的过采样）。1024 是 **8× 过采样**（每轴），纯浪费。
///   · 与 MT6 的高度图路径**恰好对齐**：`height_field::HEIGHT_DOWNSAMPLE = 8` 把 1k 高度图降到
///     **128 texel** 供位移采样（`gate-app/src/consts.rs`）⇒ 本次之后 albedo 贴图与高度场是
///     **同一张 texel 网格**（1:1），"凹凸与图案同相"这件事有了同一个分辨率基础。
///
/// **1k 素材不动、也不冲突（R13）**：磁盘上仍是 Poly Haven 的 1k（`assets/textures/pbr/<id>/`），
/// 加载时盒式降采样 `1024 → 128`（整除 8，`box_factor` 直接可用）后才上传 GPU
/// ⇒ **1k 是磁盘档位、128 是显存档位**；将来要换更高档素材（2k/4k：整除 8 不成立）需回头改
/// `box_factor` 的策略（例如改成 2048 → 128 的 16× 盒，同样整除）。
///
/// **30 个材质时的账（MT8-4 的正面收益）**：128 下 **≈ 0.104 MiB/材质**（含完整 mip 链）
/// ⇒ 16 材质 ≈ **1.7 MiB**、**30 材质 ≈ 3 MiB** ⇒ 远在 PLAN §MT2-1b 的 100MB 预算线之下
/// ⇒ **MT2-1c 当初留的"512 或 BC7 二选一"不再需要**：512 没必要（128 已够密度）、
/// **BC7/BC4 压缩取消**（省下的 ~2 MiB 不值得引入离线 KTX2 工具链或新编码器依赖，
/// 而且压缩会让"一个体素面一个平色"多一次解码误差 —— 降分辨率是更干净的那条路）。
///
/// **与 MT3 的世界尺度耦合**：一张贴图铺多大世界范围是 **MT3 的常量**（triplanar 走世界坐标，
/// 与体素尺寸解耦），两个常量一起决定 texel 密度 ⇒ **改一个必须回头看另一个**（这里调小 =
/// 密度变低，要靠 MT3 把铺贴范围也调小来补）。
///
/// **只支持"相同或整数倍缩小"**：实现只做 `src % dst == 0` 的**盒式（box）降采样**（纯 CPU、无新依赖）。
/// 放大或非整数比（如 1000 → 128）会 `warn!` 后跳过该材质——不 panic，也不做会改变色彩/能量分布的重采样。
pub const GPU_TEX_SIZE: u32 = 128;

// ============================================================================
// MT2-3 · 采样器与 mip 策略
// ============================================================================

/// mip 链的**最大层数**（含 base 层）：`ilog2(128) + 1 = 8`（128 → 64 → … → 1）。
/// 实际链长由 [`build_mip_chain`] 从实际尺寸算出（`GPU_TEX_SIZE` 改成非 2 的幂时链会更短），
/// 本常量只用来给采样器的 `lod_max_clamp` 定上界 —— **它随 `GPU_TEX_SIZE` 自动派生**
/// ⇒ MT8-4 把尺寸改成 128 后，钳位值同步从 10.0 变成 **7.0**，仍然**恰好覆盖到链底**
/// （mip 层下标 0..7，链底 = 7；写大了也无害，采样器只会钳到链底那一层）。
pub const PBR_MIP_LEVELS_MAX: u32 = GPU_TEX_SIZE.ilog2() + 1;

/// 采样器的各向异性上限（MT2-3 的"anisotropy 上限"，PLAN §5 R4）。
///
/// **wgpu 30 里 anisotropy 不再需要任何 `Features`**：它已从 `Features::SAMPLER_ANISOTROPY`
/// 移到 `DownlevelFlags::ANISOTROPIC_FILTERING`，且 wgpu-core 在不支持该 downlevel 标志时
/// **静默把 `anisotropy_clamp` 钳成 1**（唯一的硬校验是 `>= 1`）⇒ 这里写 8 **在任何后端都不会
/// panic**（不需要去开 feature，也不会因"设备不支持"而启动失败）。桌面 DX12/Vulkan 实测生效。
///
/// **为什么是 8 不是 16**：各向异性过滤的成本随 clamp 线性上升，而斜面上超过 8 之后的增益已不显著；
/// 本 pass 是 compute 里的贴图采样，取 8 是"够用且明确设了上限"。
pub const PBR_ANISOTROPY_CLAMP: u16 = 8;

/// 创建 PBR 贴图的**专用采样器**（MT2-3；与占位/真身共用同一个实例，见 `upload.rs::init_empty_gpu`）。
///
/// | 参数 | 值 | 为什么 |
/// |---|---|---|
/// | 三轴 address mode | `Repeat` | triplanar 必须平铺（世界坐标 UV 会越出 0..1） |
/// | mag / min filter | `Linear` | 近处不块状 |
/// | mipmap filter | `Linear` | 层间线性插值，避免 mip 边界出现硬跳 |
/// | `lod_min_clamp` | `0.0` | 允许采 base 层 |
/// | `lod_max_clamp` | [`PBR_MIP_LEVELS_MAX`] − 1 | **覆盖到 mip 链底**（128 → 7.0，MT8-4 后自动派生），远处不会停在中间层 |
/// | `anisotropy_clamp` | [`PBR_ANISOTROPY_CLAMP`] | 斜面（尤其地面）上的摩尔纹主要来自各向异性足迹 |
///
/// **为什么不复用 `light_samp`**（`upload.rs::init_empty_gpu` 给 GI 缓冲的那个）：那一份是
/// ClampToEdge 且 `mipmap_filter = Nearest`（GI 缓冲没有 mip 链）；改它的状态会连带改变
/// GI 的采样行为 ⇒ 另建一个。
/// 这个 desc 是**权威**：`bindings.wesl` 的 `@binding(8) pbr_samp` 注释指向这里，改一处必须同改另一处。
pub fn create_pbr_sampler(device: &RenderDevice) -> Sampler {
  device.create_sampler(&SamplerDescriptor {
    label: Some("gate_pbr_sampler"),
    address_mode_u: AddressMode::Repeat,
    address_mode_v: AddressMode::Repeat,
    address_mode_w: AddressMode::Repeat,
    mag_filter: FilterMode::Linear,
    min_filter: FilterMode::Linear,
    mipmap_filter: MipmapFilterMode::Linear,
    lod_min_clamp: 0.0,
    lod_max_clamp: (PBR_MIP_LEVELS_MAX - 1) as f32,
    anisotropy_clamp: PBR_ANISOTROPY_CLAMP,
    ..default()
  })
}

/// 已打包完成的 PBR 贴图集（main world Resource，**构建成功后才插入**）。
///
/// 消费方（MT2-2 的资产表 / bind group）用 [`Self::albedo_rough`] / [`Self::metal`] 取数组图句柄，
/// 用 [`Self::slot_of`] 把材质 id 解析成层号。
///
/// **通道语义（打包后的唯一口径）**：
///
/// | 数组 | 格式 | 通道语义 |
/// |---|---|---|
/// | `albedo_rough` | `Rgba8Unorm` | rgb = albedo（**sRGB 编码字节，采样不做转换**）、a = roughness（线性标量） |
/// | `metal` | `R8Unorm` | r = metalness（线性标量） |
///
/// ⚠️ `albedo_rough` 是 **`Rgba8Unorm` 而不是 `Rgba8UnormSrgb`** ⇒ 采样拿到的 rgb 就是 sRGB 编码值，
/// **着色侧必须自己 `srgb_to_linear()`**（与 `srgb_word_to_linear` 读 palette 字节 → `srgb_to_linear` 同一口径，
/// 这样平凡材质与 PBR 材质能共用同一个 BRDF 入口，见 PLAN D1 约束 1）。详见 `build_texture_arrays` 的注释。
///
/// **提取进 render world**（MT2-2）：整份资源随 `ExtractResourcePlugin` 拷进 render world，供
/// `upload.rs::prepare` 构建资产表、`dda.rs` 取 `GpuImage` 的视图绑 BG1 binding 6/7。
/// 句柄只是 Arc 计数 ⇒ 拷贝很便宜（`ids` 也才几十条）。
///
/// **mip（MT2-3）**：两张数组图各自带**完整 mip 链**（[`Self::mip_levels`]，128 时 8 层），
/// 采样走 `@group(1) @binding(8) pbr_samp`（[`create_pbr_sampler`]）。
#[derive(Resource, Clone, ExtractResource)]
#[extract_app(bevy::render::RenderApp)]
pub struct PbrTextureSet {
  /// 有序材质 id 列表：下标 = 槽号 = `texture_2d_array` 的层号（字典序）
  ids: Vec<String>,
  /// 打包层（`Rgba8Unorm`）：rgb = albedo（sRGB 编码字节）、a = roughness
  albedo_rough: Handle<Image>,
  /// 金属度层（`R8Unorm`）：r = metalness
  metal: Handle<Image>,
  /// 单层尺寸（所有层尺寸相同 = `GPU_TEX_SIZE` 见方，由构建期统一保证）
  size: UVec2,
  /// 层数 = 材质个数
  layers: u32,
  /// mip 链层数（含 base；两张数组图相同 —— 尺寸相同 ⇒ 链长相同）
  mip_levels: u32,
}

impl PbrTextureSet {
  /// 槽号 ↔ 材质 id 的有序表（下标即层号）。
  pub fn ids(&self) -> &[String] {
    &self.ids
  }

  /// 材质 id → 槽号（= 数组层号）；未收录返回 `None`。
  pub fn slot_of(&self, id: &str) -> Option<u32> {
    self.ids.iter().position(|x| x == id).map(|i| i as u32)
  }

  /// 打包层（`texture_2d_array`，`Rgba8Unorm`）句柄：rgb = albedo（sRGB 编码字节）、a = roughness。
  pub fn albedo_rough(&self) -> &Handle<Image> {
    &self.albedo_rough
  }

  /// 金属度层（`texture_2d_array`，`R8Unorm`）句柄：r = metalness。
  pub fn metal(&self) -> &Handle<Image> {
    &self.metal
  }

  /// 单层尺寸（宽 × 高，单位 texel）。
  pub fn size(&self) -> UVec2 {
    self.size
  }

  /// 层数（= 材质个数 = 槽号数量）。
  pub fn layers(&self) -> u32 {
    self.layers
  }

  /// mip 链层数（含 base 层；128 见方 ⇒ 8 层 = 128…1）。
  pub fn mip_levels(&self) -> u32 {
    self.mip_levels
  }

  /// 两张数组图各自的 GPU 字节数（**含完整 mip 链**：链上 texel 数 = `1 + 1/4 + 1/16 + …` ≈ 1.333×base）。
  /// 返回 `(albedo_rough, metal)`：前者 4B/texel、后者 1B/texel。
  pub fn bytes(&self) -> (u64, u64) {
    let px = mip_px_total(self.size.x) * self.layers as u64;
    (px * 4, px)
  }
}

// ============================================================================
// 全局材质资产表的**默认集**（MT2-2）
// ============================================================================

/// **临时 demo 默认值**：唯一存在的金属材质目录 id。MT3 要验收「metallic / roughness 的 BRDF 观感
/// 走向」，必须有个金属能用，所以给这一个槽位把 metallic 写死成 255。
/// **这不是隐蔽的魔法**：日志会打印它落在哪个槽号；真正的材质编写（资产 → 槽位映射、逐材质参数）
/// 属 **MT7**，届时这条默认值应当被真实的材质表取代。
pub const METAL_DEMO_ID: &str = "metal_plate";

/// **临时 demo 默认值**（MT8-5，与 [`METAL_DEMO_ID`] 同一手法、同一措辞）：唯一"**自带高度图 ⇒ CSG
/// 表面自动出凹凸**"的材质目录 id —— 就是 MT6 位移样例用的那张高度图
/// （磁盘侧 `assets/textures/pbr/<id>/<id>_height.png`）。
/// **这不是隐蔽的魔法**：启动日志会打印"这次的幅度是从材质资产读到的（值 = N）"，
/// 让人能一眼确认"改资产就改凹凸"；真正的材质编写（资产 → 幅度映射、逐材质参数）属 **MT7**，
/// 届时这条默认值应当被真实的材质表取代。背景见 `docs/PLAN.md` §4b MT8-5（决策 B = 甲）。
pub const DISPLACE_DEMO_ID: &str = "stone_wall_04";

/// **临时 demo 默认值**：[`DISPLACE_DEMO_ID`] 的位移幅度（**体素**，峰-峰；`0` = 不位移）。
/// `8` = MT6 实测的那一档（偏置 0.5 ⇒ 上下各 ±4）。
/// 上界约束：`bound = 幅度/2` 应 ≤ 块粒度（`fill_box` 的 4），否则逐体素壳层变厚、
/// 体素数 / 树规模上涨（`docs/PLAN.md` §5 R7、§8 MT6-6 的幅度上限建议）。
/// ⚠️ **别改它**：`gate-app::consts::DEMO_DISPLACE_HEIGHT_MAP` 与 MT6 / MT8-5 既有的一批对照记录
/// 都钉着 8（改了会破坏对照）。其余材质的幅度见 [`DISPLACE_TEX_DEMO_AMPLITUDE`]。
pub const DISPLACE_DEMO_AMPLITUDE: u8 = 8;

/// **临时 demo 默认值（MT8-6）**：除 [`DISPLACE_DEMO_ID`] 之外、**其余所有磁盘上有 `_height.png`
/// 的材质**的位移幅度（**体素**，峰-峰；`0` = 不位移）。`4` = 峰-峰 4 体素（偏置 0.5 ⇒ 上下各 ±2）。
/// 与 [`DISPLACE_DEMO_AMPLITUDE`] 同一手法、同一措辞：**临时 demo 默认值**，真正的材质编写
/// （逐材质的幅度）属 **MT7 的「资产表编辑」**，届时这条默认值应当被真实的材质表取代。
///
/// **为什么需要它（用户实测）**：反馈是"使用正方体、球体放置时，表面仍然没有材质凹凸"。
/// 笔触接线（MT8-5）本身是通的，卡点是**默认幅度只给了 `stone_wall_04` 一个材质**
/// ⇒ 用户随手选别的 PBR 资产（例如 `marble_cliff_03`）时幅度是 0，怎么写都没有凹凸。
/// `4`（±2）< 块粒度 4 ⇒ 只有表面一层 4³ 块退化为逐体素，体素数 / 树规模的代价最小；
/// 取它是"**随手选哪个带高度图的材质都能看出凹凸**"的一档（观感上仍是石头，不是毛刺）。
pub const DISPLACE_TEX_DEMO_AMPLITUDE: u8 = 4;

/// 位移幅度的**默认值规则**（MT8-5 的唯一来源；MT8-6 起 = "**有高度图就给非 0**"）：
/// 按材质 id 给出要写进 [`MaterialAsset::with_displacement_amplitude`] 的值。
/// 资产表（[`build_material_asset_table`]）与跨 crate 只读接口（[`displacement_amplitude_of`]）
/// **都调这一个函数** ⇒ "资产里存的值"与"`gate-app` 读到的值"不可能分叉。
///
/// - **没有高度图 ⇒ 恒 0**（哪怕 id 就是 [`DISPLACE_DEMO_ID`]）：幅度非 0 而高度图缺失时，
///   `gate-app` 侧会去解一张不存在的图 ⇒ `warn!` + 退回普通填充（不 panic，但每个这样的材质都
///   白刷一条告警）；"有没有高度图"是**磁盘事实**（`assets/textures/pbr/<id>/<id>_height.png`），
///   所以这里先查一次盘（[`has_height_map`]：纯 `is_file()`，两个调用点各查一次，代价可忽略）；
/// - **有高度图**：[`DISPLACE_DEMO_ID`] = [`DISPLACE_DEMO_AMPLITUDE`]（8，样例值**不动**）、
///   其余一律 [`DISPLACE_TEX_DEMO_AMPLITUDE`]（4）⇒ 用户随手选任何带高度图的材质都能出凹凸。
fn default_displacement_amplitude(id: &str) -> u8 {
  if !has_height_map(id) {
    return 0;
  }
  if id == DISPLACE_DEMO_ID { DISPLACE_DEMO_AMPLITUDE } else { DISPLACE_TEX_DEMO_AMPLITUDE }
}

/// `assets/textures/pbr/<id>/<id>{HEIGHT_SUFFIX}` 是否存在（**磁盘事实**，不缓存、不 panic）。
/// 路径口径与 `gate-app/src/height_field.rs::MaterialDisplace::load` 里那条**逐字相同**
/// （同一个 id、同一个后缀、同一个 `assets/` 根）⇒ 这里判"有"和那边真去解码的是同一个文件。
fn has_height_map(id: &str) -> bool {
  crate::paths::assets_dir()
    .join("textures")
    .join("pbr")
    .join(id)
    .join(format!("{id}{HEIGHT_SUFFIX}"))
    .is_file()
}

/// **只读接口（MT8-5 ③）**：按**材质 id** 查该材质的位移幅度（单位**体素**，`0` = 不位移）。
/// `gate-app` 在**场景构造**（`Startup` 的 CSG 体素化）里就要用它，见下面两条理由。
///
/// **为什么是按 id 的自由函数，而不是 `PbrTextureSet::displacement_amplitude(&self, slot: u32)`**：
/// 1. **时序**：`PbrTextureSet` 是贴图加载完成后（`Update` 里的 `finish_pbr_textures`）才插入的资源，
///    而位移必须在 `Startup` 发生（D2：位移是 CSG 体素化那一刻的产物）⇒ 那时资源还不存在，
///    按 `slot` 取**根本拿不到值**；
/// 2. **按 id 才是稳定口径**：槽位 = 目录名字典序下标，而"某个材质加载失败会让后续槽号顺延"
///    （见 `finish_pbr_textures` 的跳过逻辑）⇒ 下标不是稳定标识，id 才是（与 MT7 的材质下拉同口径）。
///
/// ⇒ 本函数内部用**与槽号定义完全同一套扫描 + 排序**（[`scan_material_dirs`]）把 id 映射到槽位，
/// 再给出"那一行会写进资产的值"。**不暴露整张资产表**（调用方只该按 id 问一个字节）。
///
/// 返回 `None` = 这个 id **不在材质目录集里**（未知材质 ⇒ 调用方不该位移，也不该拿它当"允许位移"的凭据；
/// 扫描失败同样收敛成 `None`，不 panic）；返回 `Some(0)` = 已收录但**不位移**（MT8-6 起 = 该材质磁盘上
/// **没有** `_height.png`；有高度图的材质恒为非 0，见 [`default_displacement_amplitude`]）。
pub fn displacement_amplitude_of(id: &str) -> Option<u8> {
  let root = crate::paths::assets_dir().join("textures").join("pbr");
  let ids = scan_material_dirs(&root).ok()?;
  ids.iter().any(|x| x == id).then(|| default_displacement_amplitude(id))
}

/// 默认资产的中性 albedo（sRGB 编码字节）——灰 128 与今天平凡材质的中性观感一致。
const DEFAULT_ALBEDO_SRGB: u8 = 128;
/// 默认资产的 roughness（0.5 = 中性）。
const DEFAULT_ROUGHNESS: u8 = 128;
/// 默认资产的 specular（语义同 glTF `KHR_materials_specular`：**乘性调制电介质 F0**，对金属无效）。
/// `255` ⇒ `(255-1)/254 = 1.0` = **中性、完全不调制**（这是 glTF 的默认语义）。
/// ⚠️ 曾经写成 128，那是把 `specularFactor` 当成"0.5 即中性"了 —— 会把电介质 F0 砍半（0.04 → 0.02），
/// 与注释自称的"中性"矛盾（MT3 实施时发现）。
const DEFAULT_SPECULAR: u8 = 255;
/// 默认资产的 IOR ×100（150 = 1.50 = 常见电介质，F0 = ((1.5−1)/(1.5+1))² = 0.04 = 今天玻璃的硬编码值）。
const DEFAULT_IOR_X100: u16 = 150;
/// demo 金属的 metallic 字节（255 = 完全金属）。
const METAL_DEMO_METALLIC: u8 = 255;

/// 构建**全局材质资产表的默认集**（MT2-2）：返回 `MATERIAL_ASSET_SLOTS` 条 [`MaterialAsset`]，
/// 按序号与贴图集槽位对齐。语义与取舍（写进代码而不是口头约定）：
///
/// - **贴图槽 0..`layers`**：`albedo_slot = roughmetal_slot = i`（两张数组图的第 i 层同属第 i 个材质，
///   见 `PbrTextureSet` 的层号契约）；`emissive_slot` / `transmission_slot` / `height_slot` = `NONE`。
/// - **其余槽位（`i >= layers`）与"没有贴图集"的情形**：五个 `*_slot` **全 `MATERIAL_SLOT_NONE`**。
///   **不选"全 0"**：`0` 是合法层号（层 0 有真实贴图）⇒ 用 0 会被当成"指向第 0 层的贴图"，
///   而哨兵只能是 `MATERIAL_SLOT_NONE`（`u32::MAX`）。
/// - **标量回退值**（无贴图 / 未覆盖时用）：中性灰 albedo（sRGB 128）、roughness 0.5、metallic 0、
///   specular 0.5（中性）、emissive 0、transmission 0、IOR 1.50 ⇒ 默认**电介质**，与今天的观感一致
///   （平凡变体默认本就是这套值）。
/// - **例外**：[`METAL_DEMO_ID`] 那一条的 metallic = 255（临时 demo 默认值，见该常量的说明）；
///   位移幅度（`emissive_metal` 的最高字节）**逐行按 [`default_displacement_amplitude`] 给**：
///   磁盘上有 `<id>_height.png` 的材质分别是 [`DISPLACE_DEMO_AMPLITUDE`]（`stone_wall_04`，样例值）
///   或 [`DISPLACE_TEX_DEMO_AMPLITUDE`]（其余），**没有高度图的恒 0 = 不位移**
///   —— 同样是**临时 demo 默认值**（MT8-6：让"随手选任何带高度图的材质"都能出凹凸），
///   见那几个常量的说明。**为何不在这里另做磁盘判断**：规则本身就在那个函数里，两处共用一条口径
///   （放在这里会让"资产表"与 `displacement_amplitude_of` 各判一次、可能分叉）。
pub fn build_material_asset_table(set: &PbrTextureSet) -> Vec<MaterialAsset> {
  let slots = crate::wesl_consts::material_consts().material_asset_slots as usize;
  let metal_demo = set.slot_of(METAL_DEMO_ID);
  // 标量回退字：`color.rgb(sRGB) | roughness<<24`（布局同平凡变体 word0，见 `wire.rs`）。
  let albedo_rough = DEFAULT_ALBEDO_SRGB as u32
    | (DEFAULT_ALBEDO_SRGB as u32) << 8
    | (DEFAULT_ALBEDO_SRGB as u32) << 16
    | (DEFAULT_ROUGHNESS as u32) << 24;
  (0..slots)
    .map(|i| {
      let i = i as u32;
      let textured = i < set.layers();
      let metallic = if metal_demo == Some(i) { METAL_DEMO_METALLIC as u32 } else { 0 };
      // MT8-5：位移幅度走 `default_displacement_amplitude(id)`（与 `displacement_amplitude_of`
      // 同一处规则）—— 不按 `slot_of` 比较，这样"目录集里的材质"与"资产表里的行"恒给同一个值，
      // 而且槽号顺延/加载失败都不会让两者分叉。写进去只用 `with_displacement_amplitude`，不手写移位。
      // MT8-6：该函数内部会查一次盘（有 `<id>_height.png` 才给非 0）⇒ 这里的值 = "磁盘上真有高度图"。
      let amplitude = set.ids().get(i as usize).map_or(0, |id| default_displacement_amplitude(id));
      MaterialAsset {
        albedo_slot: if textured { i } else { MATERIAL_SLOT_NONE },
        roughmetal_slot: if textured { i } else { MATERIAL_SLOT_NONE },
        emissive_slot: MATERIAL_SLOT_NONE,
        transmission_slot: MATERIAL_SLOT_NONE,
        height_slot: MATERIAL_SLOT_NONE,
        albedo_rough,
        // `emissive(0) | metallic<<8 | specular<<16`（最高字节留给位移幅度，见下面的链式调用）
        emissive_metal: metallic << 8 | (DEFAULT_SPECULAR as u32) << 16,
        // `transmission(0) | ior_x100<<16`
        transmission_ior: (DEFAULT_IOR_X100 as u32) << 16,
      }
      .with_displacement_amplitude(amplitude)
    })
    .collect()
}

/// 加载中的单条材质：两句柄已发出，等 `AssetServer` 就绪。
struct PendingMaterial {
  id: String,
  albedo: Handle<Image>,
  roughmetal: Handle<Image>,
}

/// 加载状态机（main world Resource）：`Startup` 扫描 + 发句柄一次，之后每个 `Update` 收敛一次。
#[derive(Resource, Default)]
struct PbrLoad {
  /// 扫描与句柄发出是否已完成（只做一次）
  started: bool,
  /// 仍在等就绪的材质
  pending: Vec<PendingMaterial>,
  /// **已就绪、等待最终构建**的材质。
  /// ⚠️ 必须是 Resource 字段而**不能**是 `finish_pbr_textures` 里的局部变量：加载是逐帧收敛的，
  /// 某一帧可能只有一部分材质就绪（其余还在途）⇒ 那一帧会 `return` 等下一帧，
  /// 局部变量会随 `return` 一起被丢弃；等到最后一批到齐时只剩最后一批在 `ready` 里，
  /// 表现为"**只加载了最后到齐的那几个材质，且不打任何 warn**"（时序相关，极易漏测）。
  /// ⚠️ 本表是**按加载完成顺序**追加的（与时序有关）⇒ 构建前必须重新按 id 排序，
  /// 才能兑现"槽号 = 目录名字典序下标"的契约（见 `build_texture_arrays`）。
  ready: Vec<PendingMaterial>,
  /// 已收尾（已建数组，或明确放弃 = 整体回落纯色）
  done: bool,
}

/// PBR 贴图集插件：负责扫描 `assets/textures/pbr/`、把每个材质的 albedo / roughmetal 打包成
/// `albedo_rough` + `metal` 两张 `texture_2d_array`，并在构建完成时打一条含层数 / 尺寸 / 显存估算 / 槽位表的日志。
/// 另外把 [`PbrTextureSet`] 提取进 render world（MT2-2：资产表与 BG1 binding 6/7 都在那边消费）。
pub struct PbrTexturesPlugin;

impl Plugin for PbrTexturesPlugin {
  fn build(&self, app: &mut App) {
    app
      .init_resource::<PbrLoad>()
      .add_systems(Startup, start_pbr_texture_load)
      .add_systems(Update, finish_pbr_textures)
      .add_plugins(
        bevy::render::extract_resource::ExtractResourcePlugin::<PbrTextureSet>::default(),
      );
  }
}

/// `Startup`：扫描贴图目录并按字典序发出全部加载请求（槽号 = 排序下标）。
///
/// **为什么全量加载**：PLAN 的 MT2-1 验收标准里写的是"只加载场景实际引用的材质"，但当前工程
/// **没有任何 PBR 材质被引用**（palette 里全是平凡材质，`IS_PBR` 位还没有任何写入方）⇒ 引用集是空集，
/// 全量加载是当前唯一有意义的行为；按引用加载留到 MT2-2/MT7（有资产表与引用之后）。
fn start_pbr_texture_load(asset_server: Res<AssetServer>, mut load: ResMut<PbrLoad>) {
  if load.started {
    return;
  }
  load.started = true;

  // 材质跨端常量的首个调用点：初始化即解析 WESL 权威值（缺失/写错就 fail-fast，MT1-3 的验收）。
  // 这里顺带用它校验层数上限 —— 否则常量只是"解析了但没人用"，上限写得比实际层数小不会被发现。
  let tex_slots = crate::wesl_consts::material_consts().material_tex_slots;

  let root = crate::paths::assets_dir().join("textures").join("pbr");
  let ids = match scan_material_dirs(&root) {
    Ok(ids) => ids,
    Err(e) => {
      warn!(
        target: "gate",
        "PBR 贴图集! 扫描 {} 失败（{e}）⇒ 不加载，材质回落纯色",
        root.display()
      );
      return;
    }
  };
  if ids.is_empty() {
    warn!(target: "gate", "PBR 贴图集! {} 无材质目录 ⇒ 不加载，材质回落纯色", root.display());
    return;
  }
  // 层数上限来自 WESL（`MATERIAL_TEX_SLOTS`）：超了就按字典序截断并 warn（截断是整表一致的，
  // 两张数组的层号仍一一对应）。截断而不是 panic：缺素材不该让引擎起不来。
  let ids: Vec<String> = if ids.len() as u32 > tex_slots {
    warn!(
      target: "gate",
      "PBR 贴图集! {} 个材质 > MATERIAL_TEX_SLOTS = {tex_slots}（texture_2d_array 层数上限）\
       ⇒ 只取字典序前 {tex_slots} 个；全量需上调 common.wesl 的 MATERIAL_TEX_SLOTS",
      ids.len()
    );
    ids.into_iter().take(tex_slots as usize).collect()
  } else {
    ids
  };

  for id in &ids {
    // 中间张（单材质单层）走 AssetServer → CPU 侧打包的原料，见 `load_jpg` 的说明。
    let albedo_path = format!("{PBR_TEXTURE_DIR}/{id}/{id}{ALBEDO_SUFFIX}");
    let roughmetal_path = format!("{PBR_TEXTURE_DIR}/{id}/{id}{ROUGHMETAL_SUFFIX}");
    load.pending.push(PendingMaterial {
      id: id.clone(),
      albedo: load_jpg(&asset_server, &albedo_path, true),
      roughmetal: load_jpg(&asset_server, &roughmetal_path, false),
    });
  }
  // MT2-4 的调试通道提示：放在**加载请求发出后、贴图集构建前** —— 即使后面构建失败（GpuImage 缺失等），
  // "调试开关是开着的"这一行也仍然在日志里（那种情况下它最该被看见）。槽号映射用这里的 `ids`。
  log_pbr_debug_channel(&ids);
  // MT8-1 的"甲/乙"开关同理由：它是 WESL 编译期常量，只能从画面上看效果 ⇒ 日志里留一行，
  // 人工验收（对照 Douglas 截图）时不会搞错自己在看哪一种美学。
  log_flat_shading_switch();
}

/// 扫描 `<pbr>/*/`：只收录**同时**含 albedo 与 roughmetal 的目录，按目录名字典序排序（= 槽号顺序）。
/// 缺文件的目录直接跳过（并 `warn!`）⇒ 层号对后续材质顺延，保持 0..M-1 连续。
fn scan_material_dirs(root: &Path) -> std::io::Result<Vec<String>> {
  let mut ids = Vec::new();
  for entry in std::fs::read_dir(root)? {
    let entry = entry?;
    if !entry.file_type()?.is_dir() {
      continue;
    }
    // 非 UTF-8 目录名无法进 id 表（槽号契约以目录名标识），跳过。
    let Some(id) = entry.file_name().to_str().map(str::to_owned) else {
      warn!(target: "gate", "PBR 贴图集! 目录 {:?} 名非 UTF-8 ⇒ 跳过", entry.file_name());
      continue;
    };
    let albedo = entry.path().join(format!("{id}{ALBEDO_SUFFIX}"));
    let roughmetal = entry.path().join(format!("{id}{ROUGHMETAL_SUFFIX}"));
    if !albedo.is_file() || !roughmetal.is_file() {
      warn!(
        target: "gate",
        "PBR 贴图集! 材质 `{id}` 缺文件（{} 或 {}）⇒ 跳过，后续槽号顺延",
        albedo.display(),
        roughmetal.display(),
      );
      continue;
    }
    ids.push(id);
  }
  ids.sort();
  Ok(ids)
}

/// 发出单张 jpg 的加载请求。
///
/// `is_srgb`：albedo = `true`（基色是 **sRGB 编码**字节）；roughmetal = `false`
/// （粗糙度 / 金属度是**线性标量**：把 arm 贴图标成 sRGB 格式，采样端会多做一次解码、把中间调整体压暗
/// —— 与 PLAN D1 的 ORM 布局一致）。
/// 注意：这两张中间张的字节**只被 [`pack_layer`] 原样搬运**（绝不采样、也不上传 GPU），
/// 所以这个开关实际只是把中间张标成 `Rgba8UnormSrgb` / `Rgba8Unorm` 两种格式；两者 [`rgba8_pixels`] 都收。
///
/// `asset_usage = MAIN_WORLD`：中间张**只是 CPU 侧打包的原料**，不需要 `GpuImage`。
/// 缺了这一句，32 张 1k 纹理会被各自上传成独立 2D 纹理，白白多占 ≈ 128MiB 显存。
fn load_jpg(asset_server: &AssetServer, path: &str, is_srgb: bool) -> Handle<Image> {
  asset_server
    .load_builder()
    .with_settings::<ImageLoaderSettings>(move |s| {
      s.is_srgb = is_srgb;
      s.asset_usage = RenderAssetUsages::MAIN_WORLD;
    })
    // `AssetPath<'static>`：`load` 要求 `'static`（句柄会活过本函数），故转成 owned path。
    .load(path.to_string())
}

/// `Update`：收敛加载状态；全部就绪后**只构建一次**并插入 [`PbrTextureSet`]（失败则整体回落纯色）。
fn finish_pbr_textures(
  mut commands: Commands,
  asset_server: Res<AssetServer>,
  mut images: ResMut<Assets<Image>>,
  mut load: ResMut<PbrLoad>,
) {
  if !load.started || load.done {
    return;
  }

  // 逐帧收敛：Failed 当场剔除（warn），Loaded 进 `load.ready`，其余（NotLoaded/Loading）留到下一次 Update。
  // `LoadState` 没有 `PartialEq`（`Failed` 带载荷）⇒ 用它的 `is_loaded` / `is_failed` 判定。
  // 注意：`ready` 是 Resource 字段（跨帧累积），**不能**换成局部变量 —— 理由见 `PbrLoad::ready` 的注释。
  {
    let PbrLoad { pending, ready, .. } = &mut *load;
    let mut still: Vec<PendingMaterial> = Vec::with_capacity(pending.len());
    for m in pending.drain(..) {
      let a = asset_server.load_state(m.albedo.id());
      let r = asset_server.load_state(m.roughmetal.id());
      if a.is_loaded() && r.is_loaded() {
        ready.push(m);
      } else if a.is_failed() || r.is_failed() {
        warn!(
          target: "gate",
          "PBR 贴图集! 材质 `{}` 加载失败（albedo={a:?} / roughmetal={r:?}）⇒ 跳过，后续槽号顺延",
          m.id,
        );
      } else {
        still.push(m);
      }
    }
    *pending = still;
    if !pending.is_empty() {
      return; // 还有在途的，等下一个 Update
    }
  }

  load.done = true;
  let Some(set) = build_texture_arrays(&mut images, &mut load.ready) else {
    warn!(
      target: "gate",
      "PBR 贴图集! 无可用材质（{} 候选全跳过）⇒ 不建数组图，材质回落纯色",
      load.ready.len(),
    );
    return;
  };
  log_texture_set(&set);
  commands.insert_resource(set);
}

/// CPU 侧打包：把每个材质的 albedo / roughmetal（arm）**降采样并通道打包**后依次写进两张多层数组图，
/// `Assets<Image>::add` 得到句柄。
///
/// 打包规则（见模块头注释的通道表）：
/// - `albedo_rough`：`rgb` = albedo 的 rgb 字节（**原样搬运**）、`a` = arm 的 G（roughness）；
/// - `metal`：`r` = arm 的 B（metalness）；arm 的 R（AO）丢弃。
///
/// 校验规则（全部 `warn!` + 跳过，不 panic）：
/// 1. 取不到 CPU 数据（已被提取 / 无 data）；
/// 2. 同一材质的 albedo 与 roughmetal 尺寸不同（打包要求两图逐像素对齐）；
/// 3. 源尺寸与 [`GPU_TEX_SIZE`] 不是"相同或整数倍缩小"关系（只支持盒式降采样，不放大、不做非整数重采样）；
/// 4. 纹素格式不是 8bit 无压缩 RGBA（本工程的中间张恒为 8bit RGBA）。
///
/// 被跳过的材质不占层号 ⇒ 返回表里的层号恒为 `0..M-1` 连续。
/// 所有层的输出尺寸恒为 `GPU_TEX_SIZE` 见方 ⇒ 不再需要 MT2-1 那条"跨材质参考尺寸一致"的校验。
fn build_texture_arrays(
  images: &mut Assets<Image>,
  ready: &mut [PendingMaterial],
) -> Option<PbrTextureSet> {
  let dst = GPU_TEX_SIZE;
  // 槽号契约 = **目录名字典序下标**（`scan_material_dirs` 的 `ids.sort()`、`common.wesl` 的
  // `MATERIAL_TEX_SLOTS` 注释、PLAN §8 都这么写）。而 `ready` 是**按加载完成顺序**累积的
  // ⇒ 这里必须重排，否则同一份素材两次启动可能给出不同的槽号表（槽号会随加载时序漂移，
  // 而 shader 的资产表、编辑器下拉、日志都依赖它稳定）。跳过某些材质时，槽号按"剩余者重新连续编号"。
  ready.sort_by(|a, b| a.id.cmp(&b.id));
  let mut ids: Vec<String> = Vec::new();
  let mut albedo_rough_data: Vec<u8> = Vec::new();
  let mut metal_data: Vec<u8> = Vec::new();
  // 两张数组图的 mip 链层数（下面的循环里由 `build_mip_chain` 给出；`ids` 非空时必 ≥ 1）
  let mut mip_levels: u32 = 0;

  for m in ready {
    let (Some(a), Some(r)) = (images.get(&m.albedo), images.get(&m.roughmetal)) else {
      warn!(
        target: "gate",
        "PBR 贴图集! 材质 `{}` 就绪图无 CPU 数据（已被提取？）⇒ 跳过，后续槽号顺延",
        m.id,
      );
      continue;
    };
    let (a_size, r_size) = (a.texture_descriptor.size, r.texture_descriptor.size);
    if a_size.width != r_size.width || a_size.height != r_size.height {
      warn!(
        target: "gate",
        "PBR 贴图集! 材质 `{}` albedo({}×{}) 与 roughmetal({}×{}) 尺寸不一致 ⇒ 跳过，后续槽号顺延",
        m.id, a_size.width, a_size.height, r_size.width, r_size.height,
      );
      continue;
    }
    let src = UVec2::new(a_size.width, a_size.height);
    let (Some(fx), Some(fy)) = (box_factor(src.x, dst), box_factor(src.y, dst)) else {
      warn!(
        target: "gate",
        "PBR 贴图集! 材质 `{}` 源尺寸 {}×{} 与 GPU_TEX_SIZE = {dst} 非「相同或整数倍缩小」⇒ 跳过\
         （只做盒式降采样，不放大），后续槽号顺延",
        m.id, src.x, src.y,
      );
      continue;
    };
    // 两个 `Option` 一起解构：任一侧不通过都不会有半截数据进缓冲。
    let (Some(ab), Some(rb)) = (rgba8_pixels(a, &m.id), rgba8_pixels(r, &m.id)) else {
      continue;
    };
    let (ar, mt) = pack_layer(&ab, &rb, src, fx, fy, dst);
    // MT2-3：每层再补一条**完整 mip 链**（base → 1×1，盒式 2×2 平均），
    // 于是层内布局 = `mip0, mip1, …`（见 `build_mip_chain` 与下面的数据序说明）。
    let (ar, mips) = build_mip_chain(ar, dst, 4);
    let (mt, mips_metal) = build_mip_chain(mt, dst, 1);
    // 两张数组图尺寸相同 ⇒ 链长必须相同（不同就是实现错了：会让某一张的 mip 高度对不上）。
    debug_assert_eq!(mips, mips_metal, "albedo_rough / metal 的 mip 链长不一致");
    mip_levels = mips;
    albedo_rough_data.extend_from_slice(&ar);
    metal_data.extend_from_slice(&mt);
    ids.push(m.id.clone());
  }

  let size = UVec2::splat(dst);
  if ids.is_empty() {
    return None;
  }
  let layers = ids.len() as u32;
  let extent = Extent3d { width: size.x, height: size.y, depth_or_array_layers: layers };
  // `depth_or_array_layers = N` + `TextureDimension::D2` = `texture_2d_array`（N 层同尺寸）。
  // **数据序（MT2-3 起带 mip，必须理解）**：wgpu `TextureDataOrder::LayerMajor`
  // （bevy `Image::data_order` 的默认值）= **外层 layer、内层 mip** ⇒ 内存里是
  // `layer0[mip0..mipN], layer1[mip0..mipN], …`（与上面对每个材质"先补链、再整体拼接"完全一致）。
  // wgpu `create_texture_with_data` 就是按这个顺序逐层逐 mip `&data[a..b]` 切片上传的
  // ⇒ **顺序或长度错了会直接越界切片 panic**（不会静默出错图）。
  // asset_usage = RENDER_WORLD：只给 GPU 用，提取后主世界副本自动释放（≈1.7MiB 不必在 host 留双份）。
  //
  // **为什么是 `Rgba8Unorm` 而不是 `Rgba8UnormSrgb`**（这一步很容易写错，务必理解）：
  // albedo 的字节是 jpg 里的 **sRGB 编码值**，而 `Rgba8Unorm` 采样**不做任何转换**
  // ⇒ 着色侧拿到的 rgb 就是 sRGB 编码值，再由 shader 里的 `srgb_to_linear()` 转 linear。
  // 这与 `srgb_word_to_linear`（palette 里的 color 也是 sRGB 字节 → 同样 `srgb_to_linear`）**完全同一口径**
  // ⇒ 平凡材质与 PBR 材质能共用同一个 BRDF 入口（PLAN D1 约束 1：着色公式唯一）。
  // **不在 CPU 侧把 albedo 转成 linear 再存**：8 bit 存 linear 会在暗部起色带（linear 的 8bit 量化
  // 在暗端步长过大），而存 sRGB 编码字节则暗部精度是充足的。
  // 选 `Rgba8UnormSrgb` 的写法也能跑（硬件只对 rgb 做 sRGB 解码、a 通道不受影响），但它会让
  // "rgb 已被硬件解码 / a 不解码"成为隐式约定：一旦有人在 shader 里再写一次 `srgb_to_linear()`
  // 就是双重解码（画面变暗且难以定位）。⇒ 显式用 `Rgba8Unorm` + shader 显式解码，全工程只有一条口径。
  //
  // ---- MT2-3：mip 数据必须整条链一起给，且**不能**用 `Image::new` ----
  // `Image::new` 的 `debug_assert` 是 `pixel_count(size) * pixel_size == data.len()`，而
  // `pixel_count(Extent3d)` **只按 `Extent3d` 算**（`bevy_image-0.20.0-rc.1/src/image.rs`：
  // `width * height * depth_or_array_layers`，**不含 mip**）⇒ `mip_level_count > 1` 且 data 是整条链时，
  // dev 构建下这条断言必炸。故改用 `Image::new_uninit` + 手工写 `mip_level_count` 与 `data`
  // （wgpu 那边**必须**拿到整条链：`create_texture_with_data` 会逐 mip 切片，只给 base 层会越界 panic）。
  let make_image = |data: Vec<u8>, format: TextureFormat, label: &'static str| {
    let mut img =
      Image::new_uninit(extent, TextureDimension::D2, format, RenderAssetUsages::RENDER_WORLD);
    img.texture_descriptor.label = Some(label);
    img.texture_descriptor.mip_level_count = mip_levels;
    img.data = Some(data);
    // 视图必须显式声明 `D2Array`：默认视图是单层 `D2`，将来以 `texture_2d_array` 绑定时会被 wgpu 拒掉。
    // （视图默认覆盖**全部 mip 层**，正是我们想要的：采样器要能采到链底。）
    img.texture_view_descriptor = Some(TextureViewDescriptor {
      label: Some(label),
      dimension: Some(TextureViewDimension::D2Array),
      ..default()
    });
    img
  };
  // `Rgba8Unorm`：rgb = albedo（sRGB 编码字节）、a = roughness。
  let albedo_rough_img =
    make_image(albedo_rough_data, TextureFormat::Rgba8Unorm, "gate_pbr_albedo_rough_array");
  // `R8Unorm`：metalness 是单通道线性标量，1B/texel（显存省下来的正是这里）。
  let metal_img = make_image(metal_data, TextureFormat::R8Unorm, "gate_pbr_metal_array");

  Some(PbrTextureSet {
    ids,
    albedo_rough: images.add(albedo_rough_img),
    metal: images.add(metal_img),
    size,
    layers,
    mip_levels,
  })
}

/// 一张（单层）纹理**含完整 mip 链**的 texel 数（128 见方 ⇒ 21845 ≈ 1.333 × 128²）。
/// 与 [`build_mip_chain`] 用同一套"逐级减半、遇 1 或奇数即停"的规则（改一处必须同改另一处）。
fn mip_px_total(mut size: u32) -> u64 {
  let mut px = 0u64;
  loop {
    px += u64::from(size) * u64::from(size);
    if size <= 1 || !size.is_multiple_of(2) {
      break;
    }
    size /= 2;
  }
  px
}

/// 由 base 层字节生成**完整 mip 链**（盒式 2×2 平均，降到 1×1），返回 `(全链连续字节, 层数)`。
///
/// **为什么必须自己生成**：wgpu 没有 mip 自动生成（无 `generate_mipmaps` 之类），bevy 的 `Image`
/// 只承载数据、不做降采样 ⇒ 容器侧就得把整条链算出来（MT2-3）。
///
/// **盒式平均**与 [`pack_layer`] 同一套口径、同一理由：不引新依赖、不引入振铃/能量漂移。
/// 通道**各自独立平均**（`channels` = 4 的 rgba 层与 = 1 的 metal 层共用本函数）；
/// rgb 是 sRGB 编码值，"在 sRGB 空间平均"与常规 mipmap 生成的取舍一致，本尺寸下差异不可见。
///
/// **输出顺序 = 层内 mip 连续**（`mip0 → mip1 → … → 1×1`），即 wgpu `TextureDataOrder::LayerMajor`
/// 对单层的展开；多层数组由调用方按层依次拼接（见 `build_texture_arrays`）。
/// 只对偶数尺寸减半，遇 1 或奇数即停 ⇒ `GPU_TEX_SIZE` 改成非 2 的幂时链会更短（不会错位）。
fn build_mip_chain(base: Vec<u8>, size: u32, channels: usize) -> (Vec<u8>, u32) {
  let mut out: Vec<u8> = Vec::new();
  let mut cur = base;
  let mut cur_size = size;
  let mut levels = 0u32;
  loop {
    // 每层自检：字节数必须严格等于「当前尺寸² × 通道数」。错一处，wgpu 上传时会按 mip
    // 逐层 `&data[a..b]` 切片、越界直接 panic（没有别的校验兜底）。
    debug_assert_eq!(
      cur.len(),
      (cur_size as usize) * (cur_size as usize) * channels,
      "mip 层 {levels} 字节数与 {cur_size}² × {channels} 通道不符"
    );
    out.extend_from_slice(&cur);
    levels += 1;
    if cur_size <= 1 || !cur_size.is_multiple_of(2) {
      break;
    }
    cur = halve_layer(&cur, cur_size, channels);
    cur_size /= 2;
  }
  (out, levels)
}

/// 盒式 2×2 平均降一层（`src` 必须是 `size² × channels` 字节、`size` 为偶数）。
/// `+2` 做四舍五入，避免逐级平均时整体亮度持续下偏（与 [`pack_layer`] 同一手法）。
fn halve_layer(src: &[u8], size: u32, channels: usize) -> Vec<u8> {
  let (src_size, dst) = (size as usize, (size / 2) as usize);
  let mut out = vec![0u8; dst * dst * channels];
  for y in 0..dst {
    for x in 0..dst {
      let (row0, row1) = (y * 2 * src_size * channels, (y * 2 + 1) * src_size * channels);
      let (c0, c1) = (x * 2 * channels, (x * 2 + 1) * channels);
      for c in 0..channels {
        let sum = u32::from(src[row0 + c0 + c])
          + u32::from(src[row0 + c1 + c])
          + u32::from(src[row1 + c0 + c])
          + u32::from(src[row1 + c1 + c]);
        out[(y * dst + x) * channels + c] = ((sum + 2) / 4) as u8;
      }
    }
  }
  out
}

/// 盒式降采样的整数倍因子（`src / dst`）：`1` = 尺寸相同（直接拷贝）。
/// 返回 `None` = 不支持（放大，或非整数比 ⇒ 平均盒盖不满源图，会出现采样相位漂移）。
fn box_factor(src: u32, dst: u32) -> Option<u32> {
  if src == dst {
    Some(1)
  } else if src > dst && src.is_multiple_of(dst) {
    Some(src / dst)
  } else {
    None
  }
}

/// 一次遍历同时完成**盒式降采样**与**通道打包**，产出两张层的字节：
///
/// - `rgba`（4B/texel）：`rgb` = albedo（**原样搬运，保持 sRGB 编码**）、`a` = arm 的 G（roughness）；
/// - `metal`（1B/texel）：`r` = arm 的 B（metalness）。arm 的 R（AO）丢弃。
///
/// **为什么用盒式平均**：`GPU_TEX_SIZE` 小于源图时，2×2（或 k×k）平均是唯一不需要新依赖、
/// 也不会引入振铃/能量漂移的做法；`fx == fy == 1` 时退化为纯拷贝（默认配置走的就是这条路）。
/// albedo 的 rgb 与 arm 的标量在这里是**同样的平均**：对 sRGB 编码值直接平均在数学上等价于
/// "在 sRGB 空间做平均"，与 mipmap 生成时的常规取舍一致，本尺寸下差异不可见。
///
/// 入参 `albedo` / `arm` 必须各为 `src.x × src.y × 4` 字节（由 [`rgba8_pixels`] 保证），
/// 输出长度恒为 `dst × dst × 4` 与 `dst × dst`。
fn pack_layer(
  albedo: &[u8],
  arm: &[u8],
  src: UVec2,
  fx: u32,
  fy: u32,
  dst: u32,
) -> (Vec<u8>, Vec<u8>) {
  let px = (dst as usize) * (dst as usize);
  let mut rgba = vec![0u8; px * 4];
  let mut metal = vec![0u8; px];
  // 平均盒内的纹素数（`1` = 直接拷贝）；`+ half` 做四舍五入，避免整体亮度下偏。
  let n = fx * fy;
  let half = n / 2;
  for y in 0..dst {
    for x in 0..dst {
      let (mut r, mut g, mut b, mut rough, mut met) = (0u32, 0u32, 0u32, 0u32, 0u32);
      for sy in y * fy..(y + 1) * fy {
        let row = (sy * src.x) as usize * 4;
        for sx in x * fx..(x + 1) * fx {
          let i = row + (sx as usize) * 4;
          r += albedo[i] as u32;
          g += albedo[i + 1] as u32;
          b += albedo[i + 2] as u32;
          rough += arm[i + 1] as u32; // arm 的 G = roughness
          met += arm[i + 2] as u32; // arm 的 B = metalness
        }
      }
      let o = (y * dst + x) as usize * 4;
      rgba[o] = ((r + half) / n) as u8;
      rgba[o + 1] = ((g + half) / n) as u8;
      rgba[o + 2] = ((b + half) / n) as u8;
      rgba[o + 3] = ((rough + half) / n) as u8;
      metal[(y * dst + x) as usize] = ((met + half) / n) as u8;
    }
  }
  (rgba, metal)
}

/// 取单层纹素的 RGBA8 副本（层内行主序、无行对齐填充）。
///
/// 中间张是 jpg 解码结果 ⇒ 恒为 8bit RGBA（见 `bevy_image::Image::from_dynamic`：8bit 的 RGB/RGBA
/// 都会补 α=255 后统一成 RGBA8；albedo 因 `is_srgb = true` 是 `Rgba8UnormSrgb`、arm 是 `Rgba8Unorm`，
/// 两者格式不同但**字节都是"文件里存的值"**，本函数只搬运字节、不做任何转换）。
/// 其余格式（压缩格式 / 16bit / 浮点）一律 `warn!` + 跳过 —— 宁可少一个材质，不可 panic（硬约束 1）。
fn rgba8_pixels(image: &Image, id: &str) -> Option<Vec<u8>> {
  let extent = image.texture_descriptor.size;
  if extent.depth_or_array_layers != 1 {
    warn!(
      target: "gate",
      "PBR 贴图集! 材质 `{id}` 中间张非单层（layers = {}）⇒ 跳过",
      extent.depth_or_array_layers,
    );
    return None;
  }
  let Some(data) = image.data.as_deref() else {
    warn!(target: "gate", "PBR 贴图集! 材质 `{id}` 中间张无 CPU 数据（已被提取？）⇒ 跳过");
    return None;
  };
  let px = (extent.width * extent.height) as usize;
  match image.texture_descriptor.format {
    TextureFormat::Rgba8Unorm | TextureFormat::Rgba8UnormSrgb => {
      if data.len() != px * 4 {
        warn!(
          target: "gate",
          "PBR 贴图集! 材质 `{id}` 中间张字节 {} ≠ {}×{}×4 ⇒ 跳过",
          data.len(), extent.width, extent.height,
        );
        return None;
      }
      Some(data.to_vec())
    }
    f => {
      warn!(
        target: "gate",
        "PBR 贴图集! 材质 `{id}` 中间张格式 {f:?} 非 Rgba8（只收 8bit 无压缩 RGBA）⇒ 跳过",
      );
      None
    }
  }
}

/// 构建完成日志：材质个数 / 两张数组图的尺寸与层数 / **mip 层数** / 各自与合计的显存估算 / 槽号 ↔ 目录名。
fn log_texture_set(set: &PbrTextureSet) {
  let (ar_bytes, m_bytes) = set.bytes();
  let mib = |b: u64| b as f64 / (1024.0 * 1024.0);
  let n = set.layers;
  let (w, h) = (set.size.x, set.size.y);
  let mips = set.mip_levels();
  info!(
    target: "gate",
    "PBR 贴图集: {n} 材质 → albedo_rough[{w}×{h}×{n}] Rgba8Unorm ≈ {:.2}MiB + metal[{w}×{h}×{n}] \
     R8Unorm ≈ {:.2}MiB = {:.2}MiB；mip_level_count = {mips}、GPU_TEX_SIZE = {}",
    mib(ar_bytes),
    mib(m_bytes),
    mib(ar_bytes + m_bytes),
    GPU_TEX_SIZE,
  );
  // 每材质的显存（打包后 **含 mip** ≈ 0.104MiB @128），用它把"30 材质"的账一次算清（MT8-4 的正面收益）。
  let per_mat = (ar_bytes + m_bytes) / n as u64;
  bevy::log::debug!(
    target: "gate",
    "PBR 显存: {:.3}MiB/材质（含完整 mip 链）@{w}；当前 {n} 材质 = {:.2}MiB，30 材质 ≈ {:.1}MiB",
    mib(per_mat),
    mib(ar_bytes + m_bytes),
    mib(per_mat * 30),
  );
  // MT2-4 的调试通道提示不在这里打：它在 `start_pbr_texture_load`（加载请求发出后）就打过了 ——
  // 那一处即使后面贴图集构建失败也仍然留在日志里。
  // MT8-5 / MT8-6：位移幅度（资产里的那个字节）在这里报一次 —— 它是"凹凸幅度"的唯一来源，按**槽位表
  // 同一口径**（id → 字典序槽号）打印 ⇒ 改规则 / 改常量后从这一行就能看出变化；同时它直接回答
  // "**该选哪个材质才有凹凸**"（用户实测反馈：选到 `marble_cliff_03` 之类幅度为 0 的材质怎么放都没凹凸）。
  let displaced: Vec<String> = set
    .ids()
    .iter()
    .enumerate()
    .map(|(slot, id)| (slot, id, default_displacement_amplitude(id)))
    .filter(|(_, _, amp)| *amp > 0)
    .map(|(slot, id, amp)| format!("{slot}={id}({amp})"))
    .collect();
  if displaced.is_empty() {
    warn!(
      target: "gate",
      "位移幅度: 资产表无任何非 0 幅度（这 {n} 个材质目录都无 `<id>_height.png`）⇒ 笔触与 CSG 不位移；\
       放回 assets/textures/pbr/<id>/ 即恢复"
    );
  } else {
    info!(
      target: "gate",
      "位移幅度: 非 0 槽位（`槽=id(幅度体素，峰-峰；偏置 0.5)`）= {} —— `{DISPLACE_DEMO_ID}` 是 {} 体素、\
       其余 {} 体素（`DISPLACE_TEX_DEMO_AMPLITUDE`）；无高度图者恒 0",
      displaced.join("  "),
      DISPLACE_DEMO_AMPLITUDE,
      DISPLACE_TEX_DEMO_AMPLITUDE,
    );
  }
  // 槽号 ↔ 目录名：一行太长 ⇒ 每行 `SLOT_LOG_PER_LINE` 条。
  for (row, chunk) in set.ids().chunks(SLOT_LOG_PER_LINE).enumerate() {
    let line = chunk
      .iter()
      .enumerate()
      .map(|(i, id)| format!("{}={id}", row * SLOT_LOG_PER_LINE + i))
      .collect::<Vec<_>>()
      .join("  ");
    info!(
      target: "gate",
      "PBR 槽位（字典序 = texture_2d_array 层号，albedo_rough / metal 同层）: {line}",
    );
  }
}

/// MT2-4 调试通道的启动提示 —— **唯一能让 Rust 侧"看见" `PBR_DEBUG_ASSET` 的途径**。
///
/// 为什么绕这一下：它是 WESL 源码里的编译期常量（`main.wesl`），而硬约束 1 要求跨端常量的
/// **权威只在 WESL**（Rust 不留副本）⇒ 这里直接读 `main.wesl` 源码、用
/// [`crate::wesl_consts::parse_u32_consts_in_source`] 抽出那个字面量（与 `wesl_consts.rs` 同一套解析，
/// 但**不**把它加进那边的 `REQUIRED`：它只是调试开关，缺了不起 fail-fast）。
///
/// 读不到文件 / 解析不出 / 就是关闭值 ⇒ 都只打一条 `info!`，**绝不 panic / warn**：
/// 调试开关不该影响启动。
fn log_pbr_debug_channel(ids: &[String]) {
  let path = crate::paths::dda_wesl_dir().join("main.wesl");
  let value = std::fs::read_to_string(&path).ok().and_then(|src| {
    crate::wesl_consts::parse_u32_consts_in_source(&src).get(PBR_DEBUG_ASSET_CONST).copied()
  });
  let Some(v) = value else {
    info!(
      target: "gate",
      "PBR 调试通道: 读不到 {} 的 {PBR_DEBUG_ASSET_CONST} ⇒ 按关闭（不影响渲染）",
      path.display(),
    );
    return;
  };
  if v == PBR_DEBUG_ASSET_OFF {
    info!(
      target: "gate",
      "PBR 调试通道: 关闭（PBR_DEBUG_ASSET = 0xFFFFFFFF）⇒ 不透明面走 palette 纯色（`srgb_word_to_linear`）；\
       要一键看贴图：把 main.wesl 顶部 `const PBR_DEBUG_ASSET: u32` 改成槽号再重启（0 = 字典序第一个）",
    );
  } else {
    let id = ids.get(v as usize).map_or("**超出已加载槽位数（会采到越界层）**", String::as_str);
    info!(
      target: "gate",
      "PBR 调试通道: 开启 PBR_DEBUG_ASSET = {v} → `{id}` ⇒ 所有不透明面改用该资产的整份材质\
       （triplanar 采样 + PBR 着色）；注意调试视图下自发光一并被替换（发光体熄灭，预期）",
    );
  }
}

/// MT8-1「逐体素材质采样」开关（`common.wesl::MATERIAL_FLAT_SHADING`）的启动提示。
/// 手法与 [`log_pbr_debug_channel`] 完全一致（读 WESL 源码、绝不影响启动），理由也一样：
/// 它是**编译期常量**、效果只体现在画面上，日志里留一行能让"我现在看的是甲还是乙"有据可查
/// （MT8 的验收就是人工对照 Douglas #22 的截图）。
fn log_flat_shading_switch() {
  let path = crate::paths::dda_wesl_dir().join("common.wesl");
  let value = std::fs::read_to_string(&path).ok().and_then(|src| {
    crate::wesl_consts::parse_u32_consts_in_source(&src).get(MATERIAL_FLAT_SHADING_CONST).copied()
  });
  match value {
    None => info!(
      target: "gate",
      "逐体素材质采样: 读不到 {} 的 {MATERIAL_FLAT_SHADING_CONST} ⇒ 按 `common.wesl` 的值执行（不影响渲染）",
      path.display(),
    ),
    Some(0) => info!(
      target: "gate",
      "逐体素材质采样: 关闭（MATERIAL_FLAT_SHADING = 0）= 美学「乙」：纹理按逐屏幕像素连续采样",
    ),
    Some(v) => info!(
      target: "gate",
      "逐体素材质采样: 开启（MATERIAL_FLAT_SHADING = {v}）= 美学「甲」：triplanar 采样点量化到体素中心\
       （一个体素面一个平色）；远处闪烁由 `pbr_mip_lod` 的解析 LOD 抑制",
    ),
  }
}
