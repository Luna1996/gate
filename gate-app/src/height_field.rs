//! MT6-2 · CPU 侧高度场：材质高度图（`assets/textures/pbr/<id>/<id>_height.png`）→ **普通 CPU 数组**。
//!
//! **为什么在 `gate-app` 而不在 `gate-voxel`**：`gate-voxel` 是纯逻辑 crate、零渲染依赖（硬约束 8），
//! 而高度图的解码要过 bevy 的 image 解码器 ⇒ 解码留在 app 侧，只把**普通闭包 + 普通数组**交给
//! `gate_voxel::fill_*_displaced`（见 `docs/PLAN.md` §3 D2 的"关键约束"）。
//!
//! **为什么不用 `AssetServer`**：场景构造（`scene::setup` / `build_demo_scene`）在 `Startup` 里**同步**跑，
//! 而 `AssetServer::load` 是异步的（CPU 数据要到后续 `Update` 才就绪）⇒ 这里用 `bevy_image` 的
//! `Image::from_buffer` **同步**解码（同一个解码器、同一个 `assets/` 目录）。
//! ⚠️ **两条路都需要 bevy 的 `png` feature**：它决定 bevy_image 是否**编译进** PNG 解码器
//! （`ImageFormat::Png` 本身就在 `#[cfg(feature = "png")]` 后面，`AssetServer` 也是靠它注册 `.png` 扩展名）
//! ⇒ 省不掉（本次在 `gate-app/Cargo.toml` 开了它，与 MT2 给 jpg 开 feature 同一手法；
//! 代价是 Cargo.lock 多了 PNG/zlib 解码链的传递依赖）。
//! 解码结果也不进 GPU（`RenderAssetUsages::MAIN_WORLD` 且从不注册成资产）：位移是**一次性**的 CPU 产物。
//!
//! **只解码真正要位移的那几个材质**（`docs/PLAN.md` §4 MT6-2）：全 16 个材质的 height 图约 20MB PNG、
//! 解码要几十上百毫秒，而位移只发生在场景构造期指定的那一个材质上 ⇒ 调用方按 id 调 [`HeightField::load_png`]
//! 即可（按槽号选材质的映射由 `gate_render::PbrTextureSet::slot_of` 给出，留给 MT7 的材质编辑接）。
//!
//! **采样语义（MT6-1 的"频率 / 缩放"落地处）**：见 [`HeightField::displace_fn`] 的表。

use std::path::Path;

use bevy::{
  asset::RenderAssetUsages,
  image::{CompressedImageFormats, Image, ImageSampler, ImageType},
  log::{info, warn},
  math::UVec2,
  render::render_resource::TextureFormat,
};
use glam::Vec3;

/// 载入时的**盒式降采样因子**：`1024 / 8 = 128 texel / 张贴图`。
///
/// **为什么必须降采样**（而不是直接用 1k 原图）：位移是在**体素粒度**上采样的，
/// 判据是"每个体素面至少覆盖 1 个 texel"——即 `texel/体素 = (边长 / 本因子) / 一张贴图铺的体素数`。
/// 本工程默认 128 / 100（`consts::DEMO_DISPLACE_TEX_SCALE`）= **1.28 texel/体素 ≥ 1** ✓；
/// 若直接用 1024（= 10.24 texel/体素），比体素还细的频率会被**点采样混叠**成"相邻体素高度互不相关"
/// ⇒ 位移退化成逐体素随机噪声（观感上是毛刺，不是石头）。这与 MT2-3 做 mip 的动机同源。
/// ⇒ **改 `DEMO_DISPLACE_TEX_SCALE` 时必须回头复核本值**（画面变毛刺就是它偏小了）。
/// 降采样本身是手写盒式平均（不引任何新依赖，与 MT2-1c 同一手法）；PNG 解码走 bevy 的 `image`
/// （需要 `png` feature，见本文件头的说明）。
pub const HEIGHT_DOWNSAMPLE: u32 = 8;

/// 位移的**偏置**：高度 = 0.5 的格点保持原位（`> 0.5` 沿法线外推、`< 0.5` 内缩）。
/// ⇒ 位移是**以原表面为中心的双向**位移：凹凸都在，体积基本守恒（`docs/PLAN.md` §3 D2 的"外推 / 内缩"）。
/// 取 0 = 只外推（体积只增不减）、取 1 = 只内缩；0.5 与 Douglas #22 截图里"石块表面凸起 + 缝隙凹进去"一致。
const DISPLACE_BIAS: f32 = 0.5;

/// 由幅度算"位移幅度上界"（`gate_voxel::Displace::bound`，单位 = 体素）。
/// 全法线方向的最大偏移 = `amplitude · max(bias, 1 − bias)`；`amplitude` 即**峰-峰幅度**。
pub fn displace_bound(amplitude: f32) -> f32 {
  amplitude.abs() * DISPLACE_BIAS.max(1.0 - DISPLACE_BIAS)
}

/// 已载入的高度场：行主序 `0..=1`（源图归一化），**已按 [`HEIGHT_DOWNSAMPLE`] 盒式降采样**。
#[derive(Debug, Clone)]
pub struct HeightField {
  /// 降采样后的尺寸（宽 × 高，texel）
  size: UVec2,
  data: Vec<f32>,
}

impl HeightField {
  /// 降采样后的尺寸（texel）
  pub fn size(&self) -> UVec2 {
    self.size
  }

  /// 取值区间 `(min, max)`（"1.0 之外还有多少余量"是判断"高度图是否真的用满动态范围"的依据）
  pub fn range(&self) -> (f32, f32) {
    self.data.iter().fold((f32::MAX, f32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)))
  }

  /// 同步读盘 + 解码 PNG → 高度场（`is_srgb = false`：高度是**线性数据**、不是颜色）。
  ///
  /// 收的纹素格式：8bit 灰度 / 16bit 灰度 / 8bit 灰度+alpha / 16bit 灰度+alpha / RGBA8 / RGBA16
  /// （Poly Haven 的 `height.png` 三种都出现过；一律取 **R 通道**当高度）。
  /// 其它格式 / 尺寸不是 [`HEIGHT_DOWNSAMPLE`] 整数倍 ⇒ `warn!` 后照常返回（不做降采样），**绝不 panic**。
  pub fn load_png(path: &Path) -> Result<Self, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("读 {} 失败: {e}", path.display()))?;
    let img = Image::from_buffer(
      &bytes,
      ImageType::Extension("png"),
      CompressedImageFormats::NONE,
      false, // is_srgb：高度是数据，不做 sRGB 解码（转了会把暗部整体拉亮，位移量就错了）
      ImageSampler::Default,
      RenderAssetUsages::MAIN_WORLD, // 只当 CPU 数组用：从不注册成资产 ⇒ 不上 GPU
    )
    .map_err(|e| format!("解码 {} 失败: {e}", path.display()))?;

    if img.texture_descriptor.size.depth_or_array_layers != 1 {
      return Err(format!(
        "{} 不是单层图（layers={}）",
        path.display(),
        img.texture_descriptor.size.depth_or_array_layers
      ));
    }
    let size = UVec2::new(img.width(), img.height());
    let format = img.texture_descriptor.format;
    let Some(src) = luma_of(&img) else {
      return Err(format!(
        "{} 的纹素格式 {format:?} 不支持（只收 8/16bit 的灰度或 RGBA）",
        path.display()
      ));
    };
    let (data, size) = downsample(src, size, HEIGHT_DOWNSAMPLE);
    let field = Self { size, data };
    let (lo, hi) = field.range();
    info!(
      target: "gate",
      "高度场（MT6-2）: {} → {}×{} texel（源 {}×{} {format:?}，盒式降采样 ÷{HEIGHT_DOWNSAMPLE}）；\
       取值范围 {lo:.3}..{hi:.3}",
      path.display(),
      size.x,
      size.y,
      img.width(),
      img.height(),
    );
    Ok(field)
  }

  /// 双线性采样（`u` / `v` 任意实数，内部按 **Repeat 平铺**）。
  ///
  /// 选**双线性**而不是最近邻：最近邻会把 texel 台阶（≈ 1.3 体素宽）原样搬进几何，
  /// 在斜面上呈现菱形锯齿；双线性是连续函数 ⇒ 位移曲面连续，只有量化到体素才出现台阶。
  /// **Repeat 与 GPU 侧 `pbr_samp` 的 `address_mode = Repeat` 同语义** ⇒ 位移的凹凸与
  /// albedo 的 triplanar 图案**同相**（`depth_slot` 那张图铺多大，凹凸就按多大铺）。
  pub fn sample(&self, u: f32, v: f32) -> f32 {
    let (w, h) = (self.size.x as f32, self.size.y as f32);
    // 半 texel 偏移：把 uv 的原点从 texel 角点挪到 texel 中心（否则 u=0 落在 0 与 -1 的中点）
    let x = u.rem_euclid(1.0) * w - 0.5;
    let y = v.rem_euclid(1.0) * h - 0.5;
    let (x0f, y0f) = (x.floor(), y.floor());
    let (fx, fy) = (x - x0f, y - y0f);
    let (x0, y0) = (x0f as i32, y0f as i32);
    let at = |ix: i32, iy: i32| -> f32 {
      let ix = ix.rem_euclid(self.size.x as i32) as usize;
      let iy = iy.rem_euclid(self.size.y as i32) as usize;
      self.data[iy * self.size.x as usize + ix]
    };
    let top = at(x0, y0) * (1.0 - fx) + at(x0 + 1, y0) * fx;
    let bot = at(x0, y0 + 1) * (1.0 - fx) + at(x0 + 1, y0 + 1) * fx;
    top * (1.0 - fy) + bot * fy
  }

  /// 生成 [`gate_voxel::DisplaceFn`]（MT6-1 的语义表就在这张表里定型）：
  ///
  /// | 项 | 取值 | 理由 |
  /// |---|---|---|
  /// | 采样缩放 `tex_scale_voxels` | 一张贴图铺多少**体素**（本工程 100 ≈ 2m @50 voxel/m） | 与 MT3 的 `MATERIAL_TEX_WORLD_SCALE`(= 2.0m) 同源 ⇒ 凹凸与贴图图案同相 |
  /// | UV 平面 | 由**法线**构造的切空间基（`tangent_basis`） | 轴对齐面上即"主轴投影"，与 triplanar 的三个投影一致 |
  /// | 幅度 `amplitude` | 峰-峰**体素数**（默认 8 ⇒ ±4 格） | 上限受"块粒度 ≤ 4 时只让表面一层退化"约束（见 `gate_voxel::Displace`） |
  /// | 方向 | `> 0` 沿法线外推、`< 0` 内缩（偏置 0.5 = 双向） | 凹凸都在，体积基本守恒 |
  /// | 返回值 | `(采样 − 0.5) · 幅度`，单位 = 体素 | `gate_voxel` 侧直接与有符号距离比较 |
  pub fn displace_fn<'a>(
    &'a self,
    amplitude: f32,
    tex_scale_voxels: f32,
  ) -> impl Fn(Vec3, Vec3) -> f32 + 'a {
    // 防 0 除（tex_scale 由常量给出，这里只是不让坏参数变成 inf）
    let scale = tex_scale_voxels.max(1e-3);
    move |p: Vec3, n: Vec3| {
      let (t0, t1) = tangent_basis(n);
      let u = p.dot(t0) / scale;
      let v = p.dot(t1) / scale;
      (self.sample(u, v) - DISPLACE_BIAS) * amplitude
    }
  }
}

/// 由单位外法线构造切空间基：取与法线最不平行的参考轴叉乘，避免退化（法线 ≈ ±该轴时叉积为 0）。
/// 与 MT3 的 triplanar 同思路 —— 轴对齐面上给出的正是"另外两轴"这两个 UV 方向。
fn tangent_basis(n: Vec3) -> (Vec3, Vec3) {
  let up = if n.y.abs() < 0.9 { Vec3::Y } else { Vec3::X };
  let t0 = up.cross(n).normalize_or_zero();
  let t1 = n.cross(t0).normalize_or_zero();
  (t0, t1)
}

/// 取 R 通道当高度（0..1）。长度与 `格式 × 尺寸` 不符 ⇒ `None`（不猜、不越界）。
fn luma_of(img: &Image) -> Option<Vec<f32>> {
  let data = img.data.as_deref()?;
  let px = (img.width() as usize) * (img.height() as usize);
  match img.texture_descriptor.format {
    TextureFormat::R8Unorm => gray8(data, px, 1),
    TextureFormat::Rg8Unorm => gray8(data, px, 2),
    TextureFormat::Rgba8Unorm | TextureFormat::Rgba8UnormSrgb => gray8(data, px, 4),
    TextureFormat::R16Unorm => gray16(data, px, 2),
    TextureFormat::Rg16Unorm => gray16(data, px, 4),
    TextureFormat::Rgba16Unorm => gray16(data, px, 8),
    _ => None,
  }
}

/// 每 `stride` 字节取第 1 个（R）当 8bit 高度
fn gray8(data: &[u8], px: usize, stride: usize) -> Option<Vec<f32>> {
  (data.len() == px * stride)
    .then(|| data.iter().step_by(stride).map(|&v| v as f32 / 255.0).collect())
}

/// 每 `stride` 字节取前 2 个（R）当 16bit 高度（小端，与 bevy 的 `cast_slice` 在同一台机器上同序）
fn gray16(data: &[u8], px: usize, stride: usize) -> Option<Vec<f32>> {
  (data.len() == px * stride).then(|| {
    data.chunks_exact(stride).map(|c| u16::from_le_bytes([c[0], c[1]]) as f32 / 65535.0).collect()
  })
}

/// 盒式 `factor × factor` 平均降采样（纯 CPU、零依赖，与 MT2-1c 的 `pack_layer` 同一手法）。
/// `factor <= 1` 或尺寸不整除 ⇒ `warn!` 后原样返回（不 panic，也不做会改变相位/能量的重采样）。
fn downsample(src: Vec<f32>, size: UVec2, factor: u32) -> (Vec<f32>, UVec2) {
  if factor <= 1 || !size.x.is_multiple_of(factor) || !size.y.is_multiple_of(factor) {
    warn!(
      target: "gate",
      "高度场：尺寸 {}×{} 与降采样因子 {factor} 不整除 ⇒ 跳过降采样（直接用原图，可能出逐体素毛刺）",
      size.x,
      size.y,
    );
    return (src, size);
  }
  let (dw, dh) = (size.x / factor, size.y / factor);
  let f = factor as usize;
  let mut out: Vec<f32> = Vec::with_capacity((dw * dh) as usize);
  for y in 0..dh {
    for x in 0..dw {
      let mut sum = 0.0f32;
      for sy in 0..factor {
        let row = (y * factor + sy) * size.x;
        for sx in 0..factor {
          sum += src[(row + x * factor + sx) as usize];
        }
      }
      out.push(sum / (f * f) as f32);
    }
  }
  (out, UVec2::new(dw, dh))
}
