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

pub const PBR_TEXTURE_DIR: &str = "textures/pbr";
const ALBEDO_SUFFIX: &str = "_albedo.jpg";
const ROUGHMETAL_SUFFIX: &str = "_roughmetal.jpg";
const HEIGHT_SUFFIX: &str = "_height.png";
const SLOT_LOG_PER_LINE: usize = 8;

const PBR_DEBUG_ASSET_CONST: &str = "PBR_DEBUG_ASSET";
const PBR_DEBUG_ASSET_OFF: u32 = u32::MAX;
const MATERIAL_FLAT_SHADING_CONST: &str = "MATERIAL_FLAT_SHADING";

pub const GPU_TEX_SIZE: u32 = 128;

pub const PBR_MIP_LEVELS_MAX: u32 = GPU_TEX_SIZE.ilog2() + 1;

pub const PBR_ANISOTROPY_CLAMP: u16 = 8;

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

#[derive(Resource, Clone, ExtractResource)]
#[extract_app(bevy::render::RenderApp)]
pub struct PbrTextureSet {
  ids: Vec<String>,
  albedo_rough: Handle<Image>,
  metal: Handle<Image>,
  size: UVec2,
  layers: u32,
  mip_levels: u32,
}

impl PbrTextureSet {
  pub fn ids(&self) -> &[String] {
    &self.ids
  }

  pub fn slot_of(&self, id: &str) -> Option<u32> {
    self.ids.iter().position(|x| x == id).map(|i| i as u32)
  }

  pub fn albedo_rough(&self) -> &Handle<Image> {
    &self.albedo_rough
  }

  pub fn metal(&self) -> &Handle<Image> {
    &self.metal
  }

  pub fn size(&self) -> UVec2 {
    self.size
  }

  pub fn layers(&self) -> u32 {
    self.layers
  }

  pub fn mip_levels(&self) -> u32 {
    self.mip_levels
  }

  pub fn bytes(&self) -> (u64, u64) {
    let px = mip_px_total(self.size.x) * self.layers as u64;
    (px * 4, px)
  }
}

pub const METAL_DEMO_ID: &str = "metal_plate";

pub const DISPLACE_DEMO_ID: &str = "stone_wall_04";

pub const DISPLACE_DEMO_AMPLITUDE: u8 = 8;

pub const DISPLACE_TEX_DEMO_AMPLITUDE: u8 = 4;

fn default_displacement_amplitude(id: &str) -> u8 {
  if !has_height_map(id) {
    return 0;
  }
  if id == DISPLACE_DEMO_ID { DISPLACE_DEMO_AMPLITUDE } else { DISPLACE_TEX_DEMO_AMPLITUDE }
}

fn has_height_map(id: &str) -> bool {
  crate::paths::assets_dir()
    .join("textures")
    .join("pbr")
    .join(id)
    .join(format!("{id}{HEIGHT_SUFFIX}"))
    .is_file()
}

pub fn displacement_amplitude_of(id: &str) -> Option<u8> {
  let root = crate::paths::assets_dir().join("textures").join("pbr");
  let ids = scan_material_dirs(&root).ok()?;
  ids.iter().any(|x| x == id).then(|| default_displacement_amplitude(id))
}

pub fn material_ids() -> Vec<String> {
  let root = crate::paths::assets_dir().join(PBR_TEXTURE_DIR);
  scan_material_dirs(&root).unwrap_or_default()
}

const DEFAULT_ALBEDO_SRGB: u8 = 128;
const DEFAULT_ROUGHNESS: u8 = 128;
const DEFAULT_SPECULAR: u8 = 255;
const DEFAULT_IOR_X100: u16 = 150;
const METAL_DEMO_METALLIC: u8 = 255;

pub fn build_material_asset_table(set: &PbrTextureSet) -> Vec<MaterialAsset> {
  let slots = crate::wesl_consts::material_consts().material_asset_slots as usize;
  let metal_demo = set.slot_of(METAL_DEMO_ID);
  let albedo_rough = DEFAULT_ALBEDO_SRGB as u32
    | (DEFAULT_ALBEDO_SRGB as u32) << 8
    | (DEFAULT_ALBEDO_SRGB as u32) << 16
    | (DEFAULT_ROUGHNESS as u32) << 24;
  (0..slots)
    .map(|i| {
      let i = i as u32;
      let textured = i < set.layers();
      let metallic = if metal_demo == Some(i) { METAL_DEMO_METALLIC as u32 } else { 0 };
      let amplitude = set.ids().get(i as usize).map_or(0, |id| default_displacement_amplitude(id));
      MaterialAsset {
        albedo_slot: if textured { i } else { MATERIAL_SLOT_NONE },
        roughmetal_slot: if textured { i } else { MATERIAL_SLOT_NONE },
        emissive_slot: MATERIAL_SLOT_NONE,
        transmission_slot: MATERIAL_SLOT_NONE,
        height_slot: MATERIAL_SLOT_NONE,
        albedo_rough,
        emissive_metal: metallic << 8 | (DEFAULT_SPECULAR as u32) << 16,
        transmission_ior: (DEFAULT_IOR_X100 as u32) << 16,
      }
      .with_displacement_amplitude(amplitude)
    })
    .collect()
}

struct PendingMaterial {
  id: String,
  albedo: Handle<Image>,
  roughmetal: Handle<Image>,
}

#[derive(Resource, Default)]
struct PbrLoad {
  started: bool,
  pending: Vec<PendingMaterial>,
  ready: Vec<PendingMaterial>,
  done: bool,
}

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

fn start_pbr_texture_load(asset_server: Res<AssetServer>, mut load: ResMut<PbrLoad>) {
  if load.started {
    return;
  }
  load.started = true;

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
    let albedo_path = format!("{PBR_TEXTURE_DIR}/{id}/{id}{ALBEDO_SUFFIX}");
    let roughmetal_path = format!("{PBR_TEXTURE_DIR}/{id}/{id}{ROUGHMETAL_SUFFIX}");
    load.pending.push(PendingMaterial {
      id: id.clone(),
      albedo: load_jpg(&asset_server, &albedo_path, true),
      roughmetal: load_jpg(&asset_server, &roughmetal_path, false),
    });
  }
  log_pbr_debug_channel(&ids);
  log_flat_shading_switch();
}

fn scan_material_dirs(root: &Path) -> std::io::Result<Vec<String>> {
  let mut ids = Vec::new();
  for entry in std::fs::read_dir(root)? {
    let entry = entry?;
    if !entry.file_type()?.is_dir() {
      continue;
    }
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

fn load_jpg(asset_server: &AssetServer, path: &str, is_srgb: bool) -> Handle<Image> {
  asset_server
    .load_builder()
    .with_settings::<ImageLoaderSettings>(move |s| {
      s.is_srgb = is_srgb;
      s.asset_usage = RenderAssetUsages::MAIN_WORLD;
    })
    .load(path.to_string())
}

fn finish_pbr_textures(
  mut commands: Commands,
  asset_server: Res<AssetServer>,
  mut images: ResMut<Assets<Image>>,
  mut load: ResMut<PbrLoad>,
) {
  if !load.started || load.done {
    return;
  }

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
      return;
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

fn build_texture_arrays(
  images: &mut Assets<Image>,
  ready: &mut [PendingMaterial],
) -> Option<PbrTextureSet> {
  let dst = GPU_TEX_SIZE;
  ready.sort_by(|a, b| a.id.cmp(&b.id));
  let mut ids: Vec<String> = Vec::new();
  let mut albedo_rough_data: Vec<u8> = Vec::new();
  let mut metal_data: Vec<u8> = Vec::new();
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
    let (Some(ab), Some(rb)) = (rgba8_pixels(a, &m.id), rgba8_pixels(r, &m.id)) else {
      continue;
    };
    let (ar, mt) = pack_layer(&ab, &rb, src, fx, fy, dst);
    let (ar, mips) = build_mip_chain(ar, dst, 4);
    let (mt, mips_metal) = build_mip_chain(mt, dst, 1);
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
  //
  //
  let make_image = |data: Vec<u8>, format: TextureFormat, label: &'static str| {
    let mut img =
      Image::new_uninit(extent, TextureDimension::D2, format, RenderAssetUsages::RENDER_WORLD);
    img.texture_descriptor.label = Some(label);
    img.texture_descriptor.mip_level_count = mip_levels;
    img.data = Some(data);
    img.texture_view_descriptor = Some(TextureViewDescriptor {
      label: Some(label),
      dimension: Some(TextureViewDimension::D2Array),
      ..default()
    });
    img
  };
  let albedo_rough_img =
    make_image(albedo_rough_data, TextureFormat::Rgba8Unorm, "gate_pbr_albedo_rough_array");
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

fn build_mip_chain(base: Vec<u8>, size: u32, channels: usize) -> (Vec<u8>, u32) {
  let mut out: Vec<u8> = Vec::new();
  let mut cur = base;
  let mut cur_size = size;
  let mut levels = 0u32;
  loop {
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

fn box_factor(src: u32, dst: u32) -> Option<u32> {
  if src == dst {
    Some(1)
  } else if src > dst && src.is_multiple_of(dst) {
    Some(src / dst)
  } else {
    None
  }
}

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
          rough += arm[i + 1] as u32;
          met += arm[i + 2] as u32;
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
  let per_mat = (ar_bytes + m_bytes) / n as u64;
  bevy::log::debug!(
    target: "gate",
    "PBR 显存: {:.3}MiB/材质（含完整 mip 链）@{w}；当前 {n} 材质 = {:.2}MiB，30 材质 ≈ {:.1}MiB",
    mib(per_mat),
    mib(ar_bytes + m_bytes),
    mib(per_mat * 30),
  );
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
