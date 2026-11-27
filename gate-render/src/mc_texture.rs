use bevy::{
  asset::RenderAssetUsages,
  image::Image,
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

pub const MC_TEX_SIZE: u32 = 32;
pub const MC_TEX_MIP_LEVELS: u32 = MC_TEX_SIZE.ilog2() + 1;
pub const MC_ASSET_BASE: u16 = 64;

#[derive(Resource, Clone, ExtractResource)]
#[extract_app(bevy::render::RenderApp)]
pub struct McTextureSet {
  ids: Vec<String>,
  albedo: Handle<Image>,
  size: u32,
  layers: u32,
  mip_levels: u32,
}

impl McTextureSet {
  pub fn new(ids: Vec<String>, albedo: Handle<Image>, layers: u32, mip_levels: u32) -> Self {
    Self { ids, albedo, size: MC_TEX_SIZE, layers, mip_levels }
  }

  pub fn ids(&self) -> &[String] {
    &self.ids
  }

  pub fn albedo(&self) -> &Handle<Image> {
    &self.albedo
  }

  pub fn size(&self) -> u32 {
    self.size
  }

  pub fn layers(&self) -> u32 {
    self.layers
  }

  pub fn mip_levels(&self) -> u32 {
    self.mip_levels
  }
}

pub fn create_mc_sampler(device: &RenderDevice) -> Sampler {
  device.create_sampler(&SamplerDescriptor {
    label: Some("gate_mc_sampler"),
    address_mode_u: AddressMode::Repeat,
    address_mode_v: AddressMode::Repeat,
    address_mode_w: AddressMode::Repeat,
    mag_filter: FilterMode::Nearest,
    min_filter: FilterMode::Linear,
    mipmap_filter: MipmapFilterMode::Linear,
    lod_min_clamp: 0.0,
    lod_max_clamp: (MC_TEX_MIP_LEVELS - 1) as f32,
    anisotropy_clamp: 1,
    ..default()
  })
}

pub fn texture_array_image(layers: &[Vec<u8>], size: u32) -> Image {
  let mut data = Vec::new();
  let mut mip_levels = 0;
  for src in layers {
    let (mips, levels) = build_mip_chain(src.clone(), size, 4);
    data.extend_from_slice(&mips);
    mip_levels = levels;
  }
  let extent = Extent3d { width: size, height: size, depth_or_array_layers: layers.len() as u32 };
  let mut img = Image::new_uninit(
    extent,
    TextureDimension::D2,
    TextureFormat::Rgba8Unorm,
    RenderAssetUsages::RENDER_WORLD,
  );
  img.texture_descriptor.label = Some("gate_mc_albedo_array");
  img.texture_descriptor.mip_level_count = mip_levels;
  img.data = Some(data);
  img.texture_view_descriptor = Some(TextureViewDescriptor {
    label: Some("gate_mc_albedo_array"),
    dimension: Some(TextureViewDimension::D2Array),
    ..default()
  });
  img
}

fn build_mip_chain(base: Vec<u8>, size: u32, channels: usize) -> (Vec<u8>, u32) {
  let mut out: Vec<u8> = Vec::new();
  let mut cur = base;
  let mut cur_size = size;
  let mut levels = 0u32;
  loop {
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
