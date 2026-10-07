use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use bevy::{
  asset::RenderAssetUsages,
  image::{CompressedImageFormats, Image, ImageSampler, ImageType},
  log::{info, warn},
  math::UVec2,
  prelude::Resource,
  render::render_resource::TextureFormat,
};
use glam::Vec3;

pub const HEIGHT_DOWNSAMPLE: u32 = 8;

const DISPLACE_BIAS: f32 = 0.5;

pub fn displace_bound(amplitude: f32) -> f32 {
  amplitude.abs() * DISPLACE_BIAS.max(1.0 - DISPLACE_BIAS)
}

#[derive(Debug, Clone)]
pub struct HeightField {
  size: UVec2,
  data: Vec<f32>,
}

impl HeightField {
  pub fn size(&self) -> UVec2 {
    self.size
  }

  pub fn range(&self) -> (f32, f32) {
    self.data.iter().fold((f32::MAX, f32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)))
  }

  pub fn load_png(path: &Path) -> Result<Self, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("读 {} 失败: {e}", path.display()))?;
    let img = Image::from_buffer(
      &bytes,
      ImageType::Extension("png"),
      CompressedImageFormats::NONE,
      false,
      ImageSampler::Default,
      RenderAssetUsages::MAIN_WORLD,
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
      "高度场 {} → {}×{} texel 源 {}×{} {format:?} ÷{HEIGHT_DOWNSAMPLE} 取值 {lo:.3}..{hi:.3}",
      path.display(),
      size.x,
      size.y,
      img.width(),
      img.height(),
    );
    Ok(field)
  }

  pub fn sample(&self, u: f32, v: f32) -> f32 {
    let (w, h) = (self.size.x as f32, self.size.y as f32);
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

  pub fn displace_fn<'a>(
    &'a self,
    amplitude: f32,
    tex_scale_voxels: f32,
  ) -> impl Fn(Vec3, Vec3) -> f32 + 'a {
    let scale = tex_scale_voxels.max(1e-3);
    move |p: Vec3, n: Vec3| {
      let (t0, t1) = tangent_basis(n);
      let u = p.dot(t0) / scale;
      let v = p.dot(t1) / scale;
      (self.sample(u, v) - DISPLACE_BIAS) * amplitude
    }
  }
}

fn tangent_basis(n: Vec3) -> (Vec3, Vec3) {
  let up = if n.y.abs() < 0.9 { Vec3::Y } else { Vec3::X };
  let t0 = up.cross(n).normalize_or_zero();
  let t1 = n.cross(t0).normalize_or_zero();
  (t0, t1)
}

pub struct MaterialDisplace {
  field: HeightField,
  amplitude: f32,
  tex_scale: f32,
}

impl MaterialDisplace {
  pub fn load(
    id: &str,
    tex_scale_voxels: f32,
    amplitude_override: Option<f32>,
  ) -> Result<Option<Self>, String> {
    let Some(asset_amplitude) = gate_render::pbr_texture::displacement_amplitude_of(id) else {
      info!(
        target: "gate",
        "位移源 材质 `{id}` 不在 PBR 材质目录集 → 不位移（普通 CSG）",
      );
      return Ok(None);
    };
    let (amplitude, from_asset) = match amplitude_override {
      Some(v) => (v, false),
      None => (asset_amplitude as f32, true),
    };
    if amplitude == 0.0 {
      info!(
        target: "gate",
        "位移源 材质 `{id}` 幅度 = 0（资产值）→ 不位移（普通 CSG）",
      );
      return Ok(None);
    }
    let path =
      crate::assets_dir().join("textures").join("pbr").join(id).join(format!("{id}_height.png"));
    let field = HeightField::load_png(&path)?;
    let bound = displace_bound(amplitude);
    let (lo, hi) = field.range();
    let size = field.size();
    let source = if from_asset {
      format!("幅度来自资产 id=`{id}`")
    } else {
      "幅度来自 consts::DEMO_DISPLACE_AMPLITUDE_OVERRIDE（实验旋钮）".to_string()
    };
    info!(
      target: "gate",
      "位移源 材质 `{id}` 幅度 = {amplitude} 体素（峰-峰，±{bound}）{source}；\
       高度图 {path} {w}×{h} texel {lo:.3}..{hi:.3}；铺 {tex_scale_voxels} 体素",
      path = path.display(),
      w = size.x,
      h = size.y,
    );
    Ok(Some(Self { field, amplitude, tex_scale: tex_scale_voxels }))
  }

  pub fn amplitude(&self) -> f32 {
    self.amplitude
  }

  pub fn bound(&self) -> f32 {
    displace_bound(self.amplitude)
  }

  pub fn field(&self) -> &HeightField {
    &self.field
  }

  pub fn displace_fn(&self) -> impl Fn(Vec3, Vec3) -> f32 + '_ {
    self.field.displace_fn(self.amplitude, self.tex_scale)
  }
}

#[derive(Resource, Default)]
pub struct MaterialDisplaceCache {
  entries: HashMap<String, Option<MaterialDisplace>>,
  decodes: usize,
  hits: usize,
}

impl MaterialDisplaceCache {
  pub fn get_or_load(&mut self, id: &str, tex_scale_voxels: f32) -> Option<&MaterialDisplace> {
    if self.entries.contains_key(id) {
      self.hits += 1;
      let t0 = Instant::now();
      let looked_up = t0.elapsed();
      let (decodes, hits) = (self.decodes, self.hits);
      let found = self.entries.get(id).and_then(Option::as_ref);
      info!(
        target: "gate",
        "位移缓存 材质 `{id}` 命中 #{hits}（累计解码 {decodes}）解码耗时 0，查表 {looked_up:?}"
      );
      return found;
    }
    let t0 = Instant::now();
    let loaded = match MaterialDisplace::load(id, tex_scale_voxels, None) {
      Ok(v) => v,
      Err(e) => {
        warn!(
          target: "gate",
          "位移缓存 材质 `{id}` 位移源不可用（{e}）⇒ 笔触按普通填充处理（不位移）；\
           放回 assets/textures/pbr/{id}/{id}_height.png 即恢复"
        );
        None
      }
    };
    let elapsed = t0.elapsed();
    self.decodes += 1;
    match &loaded {
      Some(md) => info!(
        target: "gate",
        "位移缓存 材质 `{id}` 首次解码 #{} 耗时 {elapsed:?} 幅度 {} 体素、铺 {tex_scale_voxels} 体素\
         ⇒ 此后落笔命中缓存、解码耗时 0",
        self.decodes,
        md.amplitude(),
      ),
      None => info!(
        target: "gate",
        "位移缓存 材质 `{id}` 不位移 #{} 耗时 {elapsed:?} ⇒ 记入缓存，此后落笔走普通填充",
        self.decodes,
      ),
    }
    self.entries.insert(id.to_string(), loaded);
    self.entries.get(id).and_then(Option::as_ref)
  }
}

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

fn gray8(data: &[u8], px: usize, stride: usize) -> Option<Vec<f32>> {
  (data.len() == px * stride)
    .then(|| data.iter().step_by(stride).map(|&v| v as f32 / 255.0).collect())
}

fn gray16(data: &[u8], px: usize, stride: usize) -> Option<Vec<f32>> {
  (data.len() == px * stride).then(|| {
    data.chunks_exact(stride).map(|c| u16::from_le_bytes([c[0], c[1]]) as f32 / 65535.0).collect()
  })
}

fn downsample(src: Vec<f32>, size: UVec2, factor: u32) -> (Vec<f32>, UVec2) {
  if factor <= 1 || !size.x.is_multiple_of(factor) || !size.y.is_multiple_of(factor) {
    warn!(
      target: "gate",
      "高度场 {}×{} 与降采样因子 {factor} 不整除 ⇒ 跳过降采样（用原图，可能出逐体素毛刺）",
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
