use bevy::{
  asset::RenderAssetUsages,
  image::Image,
  prelude::*,
  render::{
    extract_resource::ExtractResource,
    render_resource::{Extent3d, TextureDimension, TextureFormat},
  },
};

pub const CLOUD_WEATHER_SIZE: u32 = 512;
pub const CLOUD_NOISE_SIZE: u32 = 96;

#[derive(Resource, Clone, ExtractResource)]
#[extract_app(bevy::render::RenderApp)]
pub struct CloudTextureSet {
  weather: Handle<Image>,
  noise: Handle<Image>,
}

impl CloudTextureSet {
  pub fn weather(&self) -> &Handle<Image> {
    &self.weather
  }

  pub fn noise(&self) -> &Handle<Image> {
    &self.noise
  }
}

pub struct CloudTexturePlugin;

impl Plugin for CloudTexturePlugin {
  fn build(&self, app: &mut App) {
    app.add_systems(Startup, create_cloud_textures).add_plugins(
      bevy::render::extract_resource::ExtractResourcePlugin::<CloudTextureSet>::default(),
    );
  }
}

fn create_cloud_textures(mut images: ResMut<Assets<Image>>, mut commands: Commands) {
  let t0 = std::time::Instant::now();
  let weather = weather_image(generate_weather(CLOUD_WEATHER_SIZE));
  let noise = noise_image(generate_noise(CLOUD_NOISE_SIZE));
  let weather = images.add(weather);
  let noise = images.add(noise);
  commands.insert_resource(CloudTextureSet { weather, noise });
  info!(
    target: "gate",
    "云纹理: 天气 {CLOUD_WEATHER_SIZE}² Rgba8Unorm + 3D 噪声 {CLOUD_NOISE_SIZE}³ Rgba8Unorm {:?}",
    t0.elapsed(),
  );
}

fn weather_image(data: Vec<u8>) -> Image {
  let mut image = Image::new_uninit(
    Extent3d { width: CLOUD_WEATHER_SIZE, height: CLOUD_WEATHER_SIZE, depth_or_array_layers: 1 },
    TextureDimension::D2,
    TextureFormat::Rgba8Unorm,
    RenderAssetUsages::RENDER_WORLD,
  );
  image.texture_descriptor.label = Some("gate_cloud_weather");
  image.data = Some(data);
  image
}

fn noise_image(data: Vec<u8>) -> Image {
  let mut image = Image::new_uninit(
    Extent3d {
      width: CLOUD_NOISE_SIZE,
      height: CLOUD_NOISE_SIZE,
      depth_or_array_layers: CLOUD_NOISE_SIZE,
    },
    TextureDimension::D3,
    TextureFormat::Rgba8Unorm,
    RenderAssetUsages::RENDER_WORLD,
  );
  image.texture_descriptor.label = Some("gate_cloud_noise_3d");
  image.data = Some(data);
  image
}

fn generate_weather(size: u32) -> Vec<u8> {
  let n = size as usize;
  let mut out = Vec::with_capacity(n * n * 4);
  let inv = 1.0 / size as f32;
  for z in 0..size {
    for x in 0..size {
      let u = x as f32 * inv;
      let v = z as f32 * inv;
      let cover = remap(fbm2(u, v, 3, 5, 101), 0.28, 0.72, 0.0, 1.0).clamp(0.0, 1.0);
      let kind = remap(fbm2(u + 0.37, v + 0.11, 4, 4, 211), 0.3, 0.7, 0.0, 1.0).clamp(0.0, 1.0);
      let rain = smoothstep(0.62, 0.86, fbm2(u + 0.71, v + 0.43, 2, 4, 307));
      let vary = remap(fbm2(u + 0.19, v + 0.67, 8, 3, 401), 0.25, 0.75, 0.0, 1.0).clamp(0.0, 1.0);
      out.extend_from_slice(&[
        (cover * 255.0 + 0.5) as u8,
        (kind * 255.0 + 0.5) as u8,
        (rain * 255.0 + 0.5) as u8,
        (vary * 255.0 + 0.5) as u8,
      ]);
    }
  }
  out
}

fn generate_noise(size: u32) -> Vec<u8> {
  let n = size as usize;
  let mut out = Vec::with_capacity(n * n * n * 4);
  let inv = 1.0 / size as f32;
  for z in 0..size {
    for y in 0..size {
      for x in 0..size {
        let (u, v, w) = (x as f32 * inv, y as f32 * inv, z as f32 * inv);
        let perlin = perlin_fbm(u, v, w, 4);
        let worley = worley_fbm(u, v, w, 4);
        let base = remap(perlin, worley, 1.0, 0.0, 1.0).clamp(0.0, 1.0);
        let detail = remap(worley, 0.0, 1.0, 0.0, 1.0).clamp(0.0, 1.0);
        out.extend_from_slice(&[
          (base * 255.0 + 0.5) as u8,
          (detail * 255.0 + 0.5) as u8,
          (perlin * 255.0 + 0.5) as u8,
          255u8,
        ]);
      }
    }
  }
  out
}

fn remap(v: f32, lo: f32, hi: f32, a: f32, b: f32) -> f32 {
  a + (v - lo) * (b - a) / (hi - lo).max(1e-6)
}

fn smoothstep(lo: f32, hi: f32, v: f32) -> f32 {
  let t = ((v - lo) / (hi - lo).max(1e-6)).clamp(0.0, 1.0);
  t * t * (3.0 - 2.0 * t)
}

fn fbm2(u: f32, v: f32, base: i32, octaves: u32, seed: u32) -> f32 {
  let mut sum = 0.0;
  let mut norm = 0.0;
  let mut amp = 0.5;
  let mut period = base;
  for o in 0..octaves {
    sum += amp * value2(u, v, period, seed.wrapping_add(o * 101));
    norm += amp;
    amp *= 0.5;
    period *= 2;
  }
  sum / norm
}

fn value2(u: f32, v: f32, period: i32, seed: u32) -> f32 {
  let x = u * period as f32;
  let y = v * period as f32;
  let (x0, y0) = (x.floor() as i32, y.floor() as i32);
  let (fx, fy) = (smooth(x - x0 as f32), smooth(y - y0 as f32));
  let n00 = lattice2(x0, y0, period, seed);
  let n10 = lattice2(x0 + 1, y0, period, seed);
  let n01 = lattice2(x0, y0 + 1, period, seed);
  let n11 = lattice2(x0 + 1, y0 + 1, period, seed);
  let top = n00 + (n10 - n00) * fx;
  let bot = n01 + (n11 - n01) * fx;
  top + (bot - top) * fy
}

fn lattice2(x: i32, y: i32, period: i32, seed: u32) -> f32 {
  hash2(x.rem_euclid(period), y.rem_euclid(period), seed)
}

fn perlin_fbm(u: f32, v: f32, w: f32, octaves: u32) -> f32 {
  let mut sum = 0.0;
  let mut norm = 0.0;
  let mut amp = 0.5;
  let mut period = 4;
  for o in 0..octaves {
    sum += amp * perlin3(u, v, w, period, 11 + o * 101);
    norm += amp;
    amp *= 0.5;
    period *= 2;
  }
  (sum / norm * 0.5 + 0.5).clamp(0.0, 1.0)
}

fn worley_fbm(u: f32, v: f32, w: f32, octaves: u32) -> f32 {
  let mut sum = 0.0;
  let mut norm = 0.0;
  let mut amp = 0.5;
  let mut period = 4;
  for o in 0..octaves {
    sum += amp * worley3(u, v, w, period, 23 + o * 101);
    norm += amp;
    amp *= 0.5;
    period *= 2;
  }
  (sum / norm).clamp(0.0, 1.0)
}

fn perlin3(u: f32, v: f32, w: f32, period: i32, seed: u32) -> f32 {
  let x = u * period as f32;
  let y = v * period as f32;
  let z = w * period as f32;
  let (x0, y0, z0) = (x.floor() as i32, y.floor() as i32, z.floor() as i32);
  let (fx, fy, fz) = (x - x0 as f32, y - y0 as f32, z - z0 as f32);
  let g = |ix: i32, iy: i32, iz: i32, ox: f32, oy: f32, oz: f32| {
    let h = hash3i(ix.rem_euclid(period), iy.rem_euclid(period), iz.rem_euclid(period), seed);
    let grad = GRADIENTS[(h & 15) as usize];
    grad[0] * ox + grad[1] * oy + grad[2] * oz
  };
  let n000 = g(x0, y0, z0, fx, fy, fz);
  let n100 = g(x0 + 1, y0, z0, fx - 1.0, fy, fz);
  let n010 = g(x0, y0 + 1, z0, fx, fy - 1.0, fz);
  let n110 = g(x0 + 1, y0 + 1, z0, fx - 1.0, fy - 1.0, fz);
  let n001 = g(x0, y0, z0 + 1, fx, fy, fz - 1.0);
  let n101 = g(x0 + 1, y0, z0 + 1, fx - 1.0, fy, fz - 1.0);
  let n011 = g(x0, y0 + 1, z0 + 1, fx, fy - 1.0, fz - 1.0);
  let n111 = g(x0 + 1, y0 + 1, z0 + 1, fx - 1.0, fy - 1.0, fz - 1.0);
  let (sx, sy, sz) = (smooth(fx), smooth(fy), smooth(fz));
  let x00 = n000 + (n100 - n000) * sx;
  let x10 = n010 + (n110 - n010) * sx;
  let x01 = n001 + (n101 - n001) * sx;
  let x11 = n011 + (n111 - n011) * sx;
  let y0 = x00 + (x10 - x00) * sy;
  let y1 = x01 + (x11 - x01) * sy;
  y0 + (y1 - y0) * sz
}

const GRADIENTS: [[f32; 3]; 16] = [
  [1.0, 1.0, 0.0],
  [-1.0, 1.0, 0.0],
  [1.0, -1.0, 0.0],
  [-1.0, -1.0, 0.0],
  [1.0, 0.0, 1.0],
  [-1.0, 0.0, 1.0],
  [1.0, 0.0, -1.0],
  [-1.0, 0.0, -1.0],
  [0.0, 1.0, 1.0],
  [0.0, -1.0, 1.0],
  [0.0, 1.0, -1.0],
  [0.0, -1.0, -1.0],
  [1.0, 1.0, 1.0],
  [-1.0, 1.0, -1.0],
  [-1.0, -1.0, 1.0],
  [1.0, -1.0, -1.0],
];

fn worley3(u: f32, v: f32, w: f32, period: i32, seed: u32) -> f32 {
  let x = u * period as f32;
  let y = v * period as f32;
  let z = w * period as f32;
  let (xi, yi, zi) = (x.floor() as i32, y.floor() as i32, z.floor() as i32);
  let mut min = f32::MAX;
  for dz in -1..=1 {
    for dy in -1..=1 {
      for dx in -1..=1 {
        let cx = xi + dx;
        let cy = yi + dy;
        let cz = zi + dz;
        let wx = cx.rem_euclid(period);
        let wy = cy.rem_euclid(period);
        let wz = cz.rem_euclid(period);
        let fx = cx as f32 + hash2(wx, wy ^ 0x45D9, seed ^ (wz as u32).wrapping_mul(0x9E37));
        let fy = cy as f32 + hash2(wy, wz ^ 0x27D4, seed ^ (wx as u32).wrapping_mul(0x85EB));
        let fz = cz as f32 + hash2(wz, wx ^ 0xC2B2, seed ^ (wy as u32).wrapping_mul(0x1656));
        let (ddx, ddy, ddz) = (x - fx, y - fy, z - fz);
        min = min.min(ddx * ddx + ddy * ddy + ddz * ddz);
      }
    }
  }
  min.sqrt().clamp(0.0, 1.0)
}

fn hash2(x: i32, y: i32, seed: u32) -> f32 {
  let mut h = (x as u32)
    .wrapping_mul(0x27D4_EB2D)
    .wrapping_add((y as u32).wrapping_mul(0x1656_67B1))
    .wrapping_add(seed.wrapping_mul(0x9E37_79B9));
  h ^= h >> 15;
  h = h.wrapping_mul(0x85EB_CA6B);
  h ^= h >> 13;
  h = h.wrapping_mul(0xC2B2_AE35);
  h ^= h >> 16;
  (h & 0xFFFF) as f32 / 65535.0
}

fn hash3i(x: i32, y: i32, z: i32, seed: u32) -> u32 {
  let mut h = (x as u32)
    .wrapping_mul(0x27D4_EB2D)
    .wrapping_add((y as u32).wrapping_mul(0x1656_67B1))
    .wrapping_add((z as u32).wrapping_mul(0x1B56_67B1))
    .wrapping_add(seed.wrapping_mul(0x9E37_79B9));
  h ^= h >> 15;
  h = h.wrapping_mul(0x85EB_CA6B);
  h ^= h >> 13;
  h = h.wrapping_mul(0xC2B2_AE35);
  h ^= h >> 16;
  h
}

fn smooth(t: f32) -> f32 {
  t * t * (3.0 - 2.0 * t)
}
