import package::bindings::{b_palette, light_u, material_assets, pbr_albedo_rough, pbr_metal, pbr_samp, view_u};

const CHUNK_SIZE: u32 = 256u;
const MAX_LEVEL: u32 = 4u;
const NODE_FIXED_WORDS: u32 = 3u;    const CHUNK_INDEX_CAP: u32 = 64u;
const INDEX_ENTRY_EMPTY: u32 = 0xFFFFFFFFu;
const FLAGS_IS_PBR: u32 = 1u << 4u;
const FLAGS_TRANSMISSIVE: u32 = 1u << 5u;
const MATERIAL_ASSET_SLOTS: u32 = 1024u;
const MATERIAL_TEX_SLOTS: u32 = 64u;
const MATERIAL_SLOT_NONE: u32 = 0xFFFFFFFFu;
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

fn face_normal_from_index(f: u32) -> vec3<f32> {
  if (f == 0u) { return vec3<f32>(-1.0, 0.0, 0.0); }
  if (f == 1u) { return vec3<f32>( 1.0, 0.0, 0.0); }
  if (f == 2u) { return vec3<f32>(0.0, -1.0, 0.0); }
  if (f == 3u) { return vec3<f32>(0.0,  1.0, 0.0); }
  if (f == 4u) { return vec3<f32>(0.0, 0.0, -1.0); }
  if (f == 5u) { return vec3<f32>(0.0, 0.0,  1.0); }
  return vec3<f32>(0.0);
}

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

fn srgb_word_to_linear(w: u32) -> vec3<f32> {
  let srgb = vec3<f32>(
    f32(w & 0xFFu),
    f32((w >> 8u) & 0xFFu),
    f32((w >> 16u) & 0xFFu),
  ) / 255.0;
  return srgb_to_linear(srgb);
}

fn palette_rgb_linear(palette_base: u32, pal: u32) -> vec3<f32> {
  return srgb_word_to_linear(b_palette[palette_base + pal * 2u]);
}

fn palette_flags(palette_base: u32, pal: u32) -> u32 {
  return (b_palette[palette_base + pal * 2u + 1u] >> 16u) & 0xFFu;
}

fn sky_rgb() -> vec3<f32> {
  return srgb_to_linear(light_u.sky_color.xyz);
}

fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
  let l = clamp(c, vec3<f32>(0.0), vec3<f32>(1.0));
  return vec3<f32>(
    select(12.92 * l.r, pow(l.r, 1.0 / 2.4) * 1.055 - 0.055, l.r > 0.0031308),
    select(12.92 * l.g, pow(l.g, 1.0 / 2.4) * 1.055 - 0.055, l.g > 0.0031308),
    select(12.92 * l.b, pow(l.b, 1.0 / 2.4) * 1.055 - 0.055, l.b > 0.0031308)
  );
}

const MATERIAL_FLAT_SHADING: u32 = 1u;
const MATERIAL_TEX_WORLD_SCALE: f32 = 2.0;
const MATERIAL_VOXEL_PER_METER: f32 = 50.0;
const MATERIAL_TRIPLANAR_SHARPNESS: f32 = 4.0;
const MATERIAL_MIN_ROUGHNESS: f32 = 0.05;
const MATERIAL_DEFAULT_IOR: f32 = 1.5;
const MATERIAL_DEBUG_NONE: u32 = 0xFFFFFFFFu;
struct MaterialParams {
  albedo: vec3<f32>,     roughness: f32,        metallic: f32,         emissive: f32,         transmission: f32,                                                   ior: f32,              specular: f32,       }

fn material_sample_pos(p_world: vec3<f32>) -> vec3<f32> {
    return select(p_world, floor(p_world) + vec3<f32>(0.5), MATERIAL_FLAT_SHADING != 0u);
}

fn pbr_mip_lod(p: vec3<f32>) -> f32 {
    let dist_voxel = length(p - view_u.cam_pos_voxel.xyz);
  let footprint_m = dist_voxel / MATERIAL_VOXEL_PER_METER * max(view_u.lod.x, 1e-6);
  let texels = footprint_m / MATERIAL_TEX_WORLD_SCALE * f32(textureDimensions(pbr_albedo_rough).x);
  return max(log2(max(texels, 1.0)), 0.0);
}

struct TriplanarSample {
  albedo: vec3<f32>,     roughness: f32,
  metallic: f32,
}

fn triplanar_axis(slot: u32, uv: vec2<f32>, lod: f32) -> TriplanarSample {
  let s = textureSampleLevel(pbr_albedo_rough, pbr_samp, uv, i32(slot), lod);
  let met = textureSampleLevel(pbr_metal, pbr_samp, uv, i32(slot), lod).r;
  return TriplanarSample(srgb_to_linear(s.rgb), s.a, met);
}

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

fn override_value(v: u32) -> f32 {
  return select(f32(v - 1u) / 254.0, -1.0, v == 0u);
}

fn apply_override(asset_value: f32, ov: f32) -> f32 {
  return select(ov, asset_value, ov < 0.0);
}

fn fetch_material_asset(asset: u32, ov0: u32, ov1: u32, p_world: vec3<f32>, n: vec3<f32>) -> MaterialParams {
      let a = material_assets[min(asset, MATERIAL_ASSET_SLOTS - 1u)];
  let sc_albedo = srgb_word_to_linear(a.albedo_rough);
  let sc_roughness = f32((a.albedo_rough >> 24u) & 0xFFu) / 255.0;
  let sc_emissive = f32(a.emissive_metal & 0xFFu) / 255.0;
  let sc_metallic = f32((a.emissive_metal >> 8u) & 0xFFu) / 255.0;
  let sc_specular = f32((a.emissive_metal >> 16u) & 0xFFu) / 255.0;
  let sc_transmission = f32(a.transmission_ior & 0xFFFFu) / 255.0;
  let ior = f32(a.transmission_ior >> 16u) / 100.0;
          let fused = select(a.roughmetal_slot, a.albedo_slot, a.albedo_slot != MATERIAL_SLOT_NONE);
  var albedo = sc_albedo;
  var roughness = sc_roughness;
  var metallic = sc_metallic;
  if (fused != MATERIAL_SLOT_NONE) {
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

fn fetch_material(palette_base: u32, pal: u32, p_world: vec3<f32>, n: vec3<f32>,
                  debug_asset: u32) -> MaterialParams {
  if (debug_asset != MATERIAL_DEBUG_NONE) {
    return fetch_material_asset(debug_asset, 0u, 0u, p_world, n);
  }
  let w0 = b_palette[palette_base + pal * 2u];
  let w1 = b_palette[palette_base + pal * 2u + 1u];
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

fn material_ior(palette_base: u32, pal: u32) -> f32 {
  if ((palette_flags(palette_base, pal) & FLAGS_IS_PBR) == 0u) {
    return MATERIAL_DEFAULT_IOR;
  }
  let asset = b_palette[palette_base + pal * 2u + 1u] & 0xFFFFu;
  return f32(material_assets[min(asset, MATERIAL_ASSET_SLOTS - 1u)].transmission_ior >> 16u) / 100.0;
}

fn dielectric_f0(ior: f32, specular: f32) -> f32 {
  let r = (ior - 1.0) / (ior + 1.0);
  return r * r * specular;
}

fn f0_of(m: MaterialParams) -> vec3<f32> {
  return mix(vec3<f32>(dielectric_f0(m.ior, m.specular)), m.albedo, m.metallic);
}

fn ggx_peak(n: vec3<f32>, h: vec3<f32>, roughness: f32) -> f32 {
  let a = max(roughness, MATERIAL_MIN_ROUGHNESS);
  let a2 = a * a;
  let ndh2 = clamp(dot(n, h) * dot(n, h), 0.0, 1.0);
  let d = ndh2 * (a2 - 1.0) + 1.0;
  return a2 * a2 / (d * d);
}

fn fresnel_schlick(f0: vec3<f32>, cos_theta: f32) -> vec3<f32> {
  return f0 + (vec3<f32>(1.0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

fn view_dir_to(ray_origin: vec3<f32>, hit_point: vec3<f32>) -> vec3<f32> {
  let d = ray_origin - hit_point;
  return d / max(length(d), 1e-5);
}

fn face_view_dir(ray_origin: vec3<f32>, voxel_center: vec3<f32>) -> vec3<f32> {
  return view_dir_to(ray_origin, voxel_center);
}

struct SurfaceLight {
  sun_dir: vec3<f32>,     sun_c: vec3<f32>,       sun_vis: f32,               amb: vec3<f32>,
  gi: vec3<f32>,        }

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

fn emissive_radiance(m: MaterialParams, gain: f32) -> vec3<f32> {
  return m.albedo * m.emissive * gain;
}
