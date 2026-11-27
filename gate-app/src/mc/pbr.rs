use std::sync::{Arc, OnceLock};

use bevy::image::Image;
use bevy::prelude::*;
use gate_render::{MC_TEX_MIP_LEVELS, MC_TEX_SIZE, McTextureSet, texture_array_image};

use super::assets::Assets as McAssets;

pub struct McTextureData {
  pub ids: Vec<String>,
  pub layers: Vec<Vec<u8>>,
}

static CACHE: OnceLock<Arc<McTextureData>> = OnceLock::new();

pub fn data(assets: &McAssets) -> Arc<McTextureData> {
  CACHE.get_or_init(|| Arc::new(load(assets))).clone()
}

pub fn install(images: &mut bevy::asset::Assets<Image>, commands: &mut Commands) {
  let Some(data) = CACHE.get() else { return };
  if data.layers.is_empty() {
    return;
  }
  let image = texture_array_image(&data.layers, MC_TEX_SIZE);
  let handle = images.add(image);
  info!(
    target: "gate",
    "MC PBR: {} 层 @ {MC_TEX_SIZE}²，asset base={}",
    data.layers.len(),
    gate_render::MC_ASSET_BASE,
  );
  commands.insert_resource(McTextureSet::new(
    data.ids.clone(),
    handle,
    data.layers.len() as u32,
    MC_TEX_MIP_LEVELS,
  ));
}

fn load(assets: &McAssets) -> McTextureData {
  let mut ids = assets.block_texture_paths();
  let cap = (gate_render::wesl_consts::material_consts().material_asset_slots as usize)
    .saturating_sub(gate_render::MC_ASSET_BASE as usize);
  if ids.len() > cap {
    bevy::log::warn!(
      target: "gate",
      "MC PBR! {} 张贴图 > 槽位上限 {cap}（MATERIAL_ASSET_SLOTS - MC_ASSET_BASE）⇒ 截断",
      ids.len()
    );
    ids.truncate(cap);
  }
  let mut layers = Vec::with_capacity(ids.len());
  for id in &ids {
    let Some(tex) = assets.texture(id) else {
      layers.push(vec![0; (MC_TEX_SIZE * MC_TEX_SIZE * 4) as usize]);
      continue;
    };
    layers.push(resize(tex.pixels(), tex.w, tex.h));
  }
  McTextureData { ids, layers }
}

fn resize(src: &[u8], w: u32, h: u32) -> Vec<u8> {
  let dst = MC_TEX_SIZE;
  let mut out = vec![0u8; (dst * dst * 4) as usize];
  if w == 0 || h == 0 {
    return out;
  }
  for y in 0..dst {
    let sy = y * h / dst;
    for x in 0..dst {
      let sx = x * w / dst;
      let si = ((sy * w + sx) * 4) as usize;
      let di = ((y * dst + x) * 4) as usize;
      if let Some(px) = src.get(si..si + 4) {
        out[di..di + 4].copy_from_slice(px);
      }
    }
  }
  out
}
