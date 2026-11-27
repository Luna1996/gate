import package::bindings::{cloud_hist, cloud_hist_samp, cloud_noise, cloud_out, cloud_samp, cloud_u, cloud_weather, dda_hit, fog_a_r_blur, fog_a_w, fog_a_w_blur, fog_b_r_blur, fog_b_w_blur, fog_u, fog_u_blur, fog_u_read, gi_u, grid_descs, light_u, view_u};
import package::common::{hash_u32, sky_rgb};

const SUN_DISK_GAIN: f32 = 3.0;
const LIMB_POW: f32 = 0.5;
const HALO_T: f32 = 10.0;
const HALO_POW: f32 = 2.0;
const SUN_CONE_MAX: f32 = 0.06;
const FOG_DIV: u32 = 4u;
const GODRAY_TAPS: u32 = 12u;
const GODRAY_STEP_A: f32 = 0.35;
const GODRAY_STEP_B: f32 = 0.65;

fn sky_luma(c: vec3<f32>) -> f32 {
  return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn sky_stars(dir: vec3<f32>, night: f32) -> vec3<f32> {
  if (night <= 0.001) { return vec3<f32>(0.0); }
  let p = dir * 160.0;
  let cell = floor(p);
  let ip = vec3<i32>(cell);
  let h = hash_u32(u32(ip.x) * 73856093u + u32(ip.y) * 19349663u + u32(ip.z) * 83492791u);
  let pick = f32(h & 0xFFFFu) / 65535.0;
  if (pick > 0.0035) { return vec3<f32>(0.0); }
  let d = length(p - cell - vec3<f32>(0.5));
  let core = 1.0 - smoothstep(0.02, 0.34, d);
  let hot = f32((h >> 16u) & 0xFFu) / 255.0;
  let tint = mix(vec3<f32>(1.0, 0.84, 0.68), vec3<f32>(0.78, 0.88, 1.0), hot);
  let bright = (0.35 + 0.65 * hot) * night * smoothstep(0.02, 0.16, dir.y);
  return tint * core * bright * 0.22;
}

fn sky_ambient(dir: vec3<f32>) -> vec3<f32> {
  let c = sky_rgb();
  let luma = sky_luma(c);
  let day = smoothstep(0.0025, 0.12, luma);
  let dusk = smoothstep(0.002, 0.02, luma) * (1.0 - smoothstep(0.05, 0.15, luma));
  let warm = vec3<f32>(1.0, 0.97, 0.92);
  let dusk_col = vec3<f32>(0.35, 0.12, 0.05);
  let hor = mix(mix(c * 1.08, dusk_col, 0.22 * dusk), warm, 0.26 * day * day);
  let zen = mix(c * 0.72, c * vec3<f32>(0.68, 0.76, 0.92), day);
  return mix(hor, zen, smoothstep(-0.03, 0.55, dir.y));
}

fn sky_base(dir: vec3<f32>) -> vec3<f32> {
  let night = 1.0 - smoothstep(0.0015, 0.012, sky_luma(sky_rgb()));
  return sky_ambient(dir) + sky_stars(dir, night);
}

const SCATTER_GAIN: f32 = 0.02;

fn sky_scatter(dir: vec3<f32>) -> vec3<f32> {
  let s = light_u.g.exposure_pad;
  if (s.z <= 0.0) { return vec3<f32>(0.0); }
  let l = light_u.lights[0];
  let g = clamp(s.w, -0.95, 0.95);
  let cos_t = dot(dir, l.kind_pos_dir.yzw);
  let d = max(1.0 + g * g - 2.0 * g * cos_t, 1e-3);
  let lobe = (1.0 - g * g) / pow(d, 1.5);
  let horizon = smoothstep(-0.05, 0.22, dir.y);
  let tint = mix(sky_ambient(dir), l.color_intensity.xyz * l.color_intensity.w, 0.35);
  return tint * (s.z * max(lobe - 1.0, 0.0) * horizon * SCATTER_GAIN);
}

const CLOUD_WEATHER_SCALE_M: f32 = 30000.0;
const CLOUD_BASE_SCALE_M: f32 = 6000.0;
const CLOUD_DETAIL_SCALE_M: f32 = 900.0;
const CLOUD_SIGMA: f32 = 0.012;
const CLOUD_LOW_BASE_M: f32 = 1200.0;
const CLOUD_LOW_TOP_M: f32 = 4200.0;
const CLOUD_HIGH_BASE_M: f32 = 6500.0;
const CLOUD_HIGH_TOP_M: f32 = 10500.0;
const CLOUD_STEPS_LOW: u32 = 64u;
const CLOUD_STEPS_HIGH: u32 = 24u;
const CLOUD_LIGHT_STEPS: u32 = 4u;
const CLOUD_RANGE_M: f32 = 8000.0;
const CLOUD_STEP_M: f32 = 26.0;
const CLOUD_STEP_GROW: f32 = 1.04;

const CLOUD_WIND_MPS: vec2<f32> = vec2<f32>(45.0, 18.0);

fn cloud_weather_uv(p: vec3<f32>, time: f32, wind_scale: f32) -> vec2<f32> {
  return (p.xz + CLOUD_WIND_MPS * time * wind_scale) / CLOUD_WEATHER_SCALE_M;
}

fn cloud_density_raw(
  p: vec3<f32>, base: f32, top: f32, layer: u32, time: f32, cover_scale: f32, detail_on: bool
) -> f32 {
  let h = (p.y - base) / max(top - base, 1.0);
  if (h <= 0.0 || h >= 1.0) { return 0.0; }
  let wind_scale = select(1.0, 1.7, layer == 1u);
  let w = textureSampleLevel(cloud_weather, cloud_samp, cloud_weather_uv(p, time, wind_scale), 0.0);
  let cover = clamp(
    1.0 - exp(-w.r * cover_scale * 2.2 * select(1.0, 0.9, layer == 1u)),
    0.0,
    1.0,
  );
  if (cover <= 0.01) { return 0.0; }
  let kind = w.g;
  var prof = smoothstep(0.0, 0.06, h)
           * (1.0 - smoothstep(0.55 + 0.35 * (1.0 - kind), 1.0, h));
  if (layer == 1u) {
    prof = smoothstep(0.0, 0.18, h) * (1.0 - smoothstep(0.7, 1.0, h)) * 0.6;
  }
  let wind = vec3<f32>(CLOUD_WIND_MPS.x, 0.0, CLOUD_WIND_MPS.y) * time * wind_scale;
  let base_n =
    textureSampleLevel(cloud_noise, cloud_samp, (p + wind) / CLOUD_BASE_SCALE_M, 0.0);
  let cover_edge = 1.0 - cover;
  let shape = smoothstep(cover_edge, cover_edge + 0.12, base_n.r);
  if (shape <= 0.0 || !detail_on) { return shape * prof; }
  let detail =
    textureSampleLevel(cloud_noise, cloud_samp, (p + wind) / CLOUD_DETAIL_SCALE_M, 0.0);
  let eroded = smoothstep(cover_edge, cover_edge + 0.12, base_n.r - detail.g * 0.3);
  return eroded * prof;
}

fn cloud_density(p: vec3<f32>, base: f32, top: f32, layer: u32) -> f32 {
  return cloud_density_raw(p, base, top, layer, cloud_u.params.x, cloud_u.params.y, true);
}

fn cloud_shadow_layer(
  p: vec3<f32>, sun_dir: vec3<f32>, base: f32, top: f32, layer: u32, time: f32, cover: f32
) -> f32 {
  let a = (base - p.y) / sun_dir.y;
  let b = (top - p.y) / sun_dir.y;
  let near = max(min(a, b), 0.0);
  let far = min(max(a, b), near + 12000.0);
  if (far <= near) { return 0.0; }
  let n = 4u;
  let dt = (far - near) / f32(n);
  var od = 0.0;
  for (var i = 0u; i < n; i = i + 1u) {
    let q = p + sun_dir * (near + (f32(i) + 0.5) * dt);
    od = od + cloud_density_raw(q, base, top, layer, time, cover, false) * dt;
  }
  return od;
}

fn cloud_sun_shadow(p_voxel: vec3<f32>, sun_dir: vec3<f32>) -> f32 {
  if (sun_dir.y <= 0.0) { return 1.0; }
  let vpm = max(grid_descs[0].voxel_per_meter, 1e-3);
  let p = p_voxel / vpm;
  let time = fog_u_read.misc3.z;
  let cover = fog_u_read.misc3.w;
  var od = cloud_shadow_layer(p, sun_dir, CLOUD_LOW_BASE_M, CLOUD_LOW_TOP_M, 0u, time, cover);
  od = od + cloud_shadow_layer(p, sun_dir, CLOUD_HIGH_BASE_M, CLOUD_HIGH_TOP_M, 1u, time, cover);
  return exp(-od * CLOUD_SIGMA * 0.65);
}

fn cloud_phase(cos_t: f32) -> f32 {
  let g0 = 0.8;
  let d0 = max(1.0 + g0 * g0 - 2.0 * g0 * cos_t, 1e-3);
  let forward = (1.0 - g0 * g0) / pow(d0, 1.5);
  let g1 = -0.25;
  let d1 = max(1.0 + g1 * g1 - 2.0 * g1 * cos_t, 1e-3);
  let back = (1.0 - g1 * g1) / pow(d1, 1.5);
  return 0.25 + forward * 0.08 + back * 0.04;
}

fn cloud_light(p: vec3<f32>, sun_dir: vec3<f32>, base: f32, top: f32, layer: u32) -> f32 {
  var od = 0.0;
  var t = 0.0;
  for (var i = 0u; i < CLOUD_LIGHT_STEPS; i = i + 1u) {
    let step = 35.0 + f32(i) * 55.0;
    t = t + step;
    od = od + cloud_density(p + sun_dir * t, base, top, layer) * step;
  }
  let trans = exp(-od * CLOUD_SIGMA);
  return trans + 0.3 * (1.0 - trans);
}

fn cloud_march_layer(
  dir: vec3<f32>, sun_dir: vec3<f32>, sun_c: vec3<f32>, amb: vec3<f32>,
  base: f32, top: f32, steps: u32, layer: u32
) -> vec4<f32> {
  let vpm = max(grid_descs[0].voxel_per_meter, 1e-3);
  let cam = view_u.cam_pos_voxel.xyz / vpm;
  if (abs(dir.y) < 1e-4) {
    if (cam.y < base || cam.y > top) { return vec4<f32>(0.0); }
  }
  let t0 = (base - cam.y) / select(dir.y, sign(dir.y) * 1e-4, abs(dir.y) < 1e-4);
  let t1 = (top - cam.y) / select(dir.y, sign(dir.y) * 1e-4, abs(dir.y) < 1e-4);
  let near = max(min(t0, t1), 0.0);
  let far = min(max(t0, t1), near + CLOUD_RANGE_M);
  if (far <= near) { return vec4<f32>(0.0); }
  let base_step = select(CLOUD_STEP_M, 35.0, layer == 1u);
  let ratio = select(CLOUD_STEP_GROW, 1.12, layer == 1u);
  let jitter = fract(
    sin(dot(dir.xz, vec2<f32>(12.9898, 78.233)) + cloud_u.params.z * 0.618) * 43758.5453);
  var step = base_step;
  var t = near + jitter * base_step;
  var trans = 1.0;
  var acc = vec3<f32>(0.0);
  for (var i = 0u; i < steps; i = i + 1u) {
    let s = t + step * 0.5;
    if (s >= far) { break; }
    let p = cam + dir * s;
    let far_fade = 1.0 - smoothstep(CLOUD_RANGE_M * 0.7, CLOUD_RANGE_M, s);
    let d = cloud_density(p, base, top, layer) * far_fade;
    if (d > 0.002) {
      let light = cloud_light(p, sun_dir, base, top, layer);
      let phase = cloud_phase(dot(dir, sun_dir));
      let ext = exp(-d * CLOUD_SIGMA * step);
      let sun = min(sun_c, vec3<f32>(6.0));
      let rim = pow(max(dot(dir, sun_dir), 0.0), 6.0) * (1.0 - d);
      let scatter = sun * (light * phase * (1.0 - ext) + rim * 0.12)
                  + amb * 0.35 * (1.0 - ext);
      acc = acc + scatter * trans;
      trans = trans * ext;
      if (trans < 0.01) { break; }
    }
    t = t + step;
    step = min(step * ratio, far - t);
    if (step <= 1e-3) { break; }
  }
  return vec4<f32>(acc, 1.0 - trans);
}

fn cloud_volume(dir: vec3<f32>, sun_dir: vec3<f32>, sun_c: vec3<f32>) -> vec4<f32> {
  if (abs(dir.y) < 0.01) { return vec4<f32>(0.0); }
  let amb = sky_ambient(dir);
  let vpm = max(grid_descs[0].voxel_per_meter, 1e-3);
  let cam_y = view_u.cam_pos_voxel.y / vpm;
  let high = cloud_march_layer(
    dir, sun_dir, sun_c, amb, CLOUD_HIGH_BASE_M, CLOUD_HIGH_TOP_M, CLOUD_STEPS_HIGH, 1u);
  let low = cloud_march_layer(
    dir, sun_dir, sun_c, amb, CLOUD_LOW_BASE_M, CLOUD_LOW_TOP_M, CLOUD_STEPS_LOW, 0u);
  let h = smoothstep(0.02, 0.1, abs(dir.y));
  let high_a = high.a * h;
  let low_a = low.a * h;
  let high_rgb = high.rgb * h;
  let low_rgb = low.rgb * h;
  var c = vec3<f32>(0.0);
  var a = 0.0;
  if (cam_y > CLOUD_LOW_TOP_M) {
    c = c * (1.0 - low_a) + low_rgb;
    a = a * (1.0 - low_a) + low_a;
    c = c * (1.0 - high_a) + high_rgb;
    a = a * (1.0 - high_a) + high_a;
  } else {
    c = c * (1.0 - high_a) + high_rgb;
    a = a * (1.0 - high_a) + high_a;
    c = c * (1.0 - low_a) + low_rgb;
    a = a * (1.0 - low_a) + low_a;
  }
  return vec4<f32>(c, a);
}

fn cloud_render(gid: vec3<u32>) {
  let size = textureDimensions(cloud_out);
  if (gid.x >= size.x || gid.y >= size.y) { return; }
  let coord = vec2<i32>(i32(gid.x), i32(gid.y));
  let uv = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5))
         / vec2<f32>(f32(size.x), f32(size.y));
  let hsize = textureDimensions(dda_hit);
  let hc = vec2<i32>(
    clamp(i32(floor(uv.x * f32(hsize.x))), 0, i32(hsize.x) - 1),
    clamp(i32(floor(uv.y * f32(hsize.y))), 0, i32(hsize.y) - 1));
  let rec = textureLoad(dda_hit, hc);
  if ((rec.w & 1u) != 0u) {
    textureStore(cloud_out, coord, vec4<f32>(0.0));
    return;
  }

  let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
  let near_h = view_u.inv_view_proj * vec4<f32>(ndc.x, ndc.y, 0.0, 1.0);
  let far_h = view_u.inv_view_proj * vec4<f32>(ndc.x, ndc.y, 1.0, 1.0);
  let near_p = near_h.xyz / near_h.w;
  let far_p = far_h.xyz / far_h.w;
  let dir = normalize(far_p - near_p);

  let sun = light_u.lights[0];
  let sun_dir = sun.kind_pos_dir.yzw;
  let sun_c = sun.color_intensity.xyz * sun.color_intensity.w;
  let cur = cloud_volume(dir, sun_dir, sun_c);

  var hist = vec4<f32>(0.0);
  let cam = view_u.cam_pos_voxel.xyz;
  let probe = cam + dir * 20000.0;
  let prev_clip = cloud_u.prev_view_proj * vec4<f32>(probe, 1.0);
  if (prev_clip.w > 1e-4) {
    let prev_ndc = prev_clip.xyz / prev_clip.w;
    let prev_uv = vec2<f32>(prev_ndc.x * 0.5 + 0.5, 0.5 - prev_ndc.y * 0.5);
    if (all(prev_uv >= vec2<f32>(0.0)) && all(prev_uv <= vec2<f32>(1.0))) {
      hist = textureSampleLevel(cloud_hist, cloud_hist_samp, prev_uv, 0.0);
    }
  }
  let blend = select(0.0, 0.86, cloud_u.params.w > 0.5);
  textureStore(cloud_out, coord, mix(cur, hist, blend));
}

fn sky_primary(dir: vec3<f32>, sun_dir: vec3<f32>, sun_c: vec3<f32>) -> vec3<f32> {
  let base = sky_base(dir);
  if (fog_u_read.misc2.x <= 0.5) { return base + sky_scatter(dir); }
  let ang = clamp(fog_u_read.misc2.z, 1e-4, SUN_CONE_MAX);
  let cos_t = dot(dir, sun_dir);
  let t2 = (1.0 - cos_t) / max(1.0 - cos(ang), 1e-12);
  let limb = clamp(1.0 - t2, 0.0, 1.0);
  let disk = pow(limb, LIMB_POW);

  let hx = clamp(t2 / (HALO_T * HALO_T), 0.0, 1.0);
  let halo = pow(1.0 - hx, HALO_POW) * (1.0 - hx * hx * hx);
  let disk_gain = max(light_u.lights[0].shape.x, 0.0);
  return base + sky_scatter(dir)
       + sun_c * (disk * SUN_DISK_GAIN * disk_gain
                  + halo * max(fog_u_read.misc2.w, 0.0));
}


fn godray_pass_mask(gid: vec3<u32>) {
  let size = textureDimensions(fog_a_w);
  if (gid.x >= size.x || gid.y >= size.y) { return; }
  let coord = vec2<i32>(i32(gid.x), i32(gid.y));
  let px = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5))
         / vec2<f32>(f32(size.x), f32(size.y));
  let ndc = vec2<f32>(px.x * 2.0 - 1.0, 1.0 - px.y * 2.0);
  let near_world_h = view_u.inv_view_proj * vec4<f32>(ndc.x, ndc.y, 0.0, 1.0);
  let far_world_h = view_u.inv_view_proj * vec4<f32>(ndc.x, ndc.y, 1.0, 1.0);
  let near_world = near_world_h.xyz / near_world_h.w;
  let far_world = far_world_h.xyz / far_world_h.w;
  let diff_world = far_world - near_world;
  let dir_voxel = normalize(diff_world);

  
  
  
  let gdv = max(u32(gi_u.flags.y), 1u);
  let hsz = vec2<f32>(textureDimensions(dda_hit));
  let center_px = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5)) * f32(FOG_DIV);
  let hc = vec2<i32>(
    clamp(i32(floor(center_px.x / max(f32(gdv), 1.0))), 0, i32(hsz.x) - 1),
    clamp(i32(floor(center_px.y / max(f32(gdv), 1.0))), 0, i32(hsz.y) - 1));
  let rec = textureLoad(dda_hit, hc);

  let sun_dir = light_u.lights[0].kind_pos_dir.yzw;
  let sun_c = light_u.lights[0].color_intensity.xyz * light_u.lights[0].color_intensity.w;
  let sky = select(0.0, 1.0, (rec.w & 1u) == 0u);
  let w = pow(clamp(dot(dir_voxel, sun_dir), 0.0, 1.0), max(fog_u.misc.y, 1.0));
  let lit = sun_c * (sky * w * max(fog_u.misc.z, 0.0) * clamp(fog_u.misc.w, 0.0, 1.0));
  textureStore(fog_a_w, coord, vec4<f32>(lit, 1.0));
}

fn godray_sample_bilinear(tex: texture_2d<f32>, size: vec2<f32>, uv: vec2<f32>) -> vec3<f32> {
  let g = uv * size - vec2<f32>(0.5);
  let base = floor(g);
  let f = g - base;
  let hi = vec2<i32>(size) - vec2<i32>(1);
  var acc = vec3<f32>(0.0);
  for (var j: i32 = 0; j < 2; j = j + 1) {
    for (var i: i32 = 0; i < 2; i = i + 1) {
      let c = clamp(vec2<i32>(base) + vec2<i32>(i, j), vec2<i32>(0), hi);
      let bw = select(1.0 - f.x, f.x, i == 1) * select(1.0 - f.y, f.y, j == 1);
      acc = acc + textureLoad(tex, c, 0).rgb * bw;
    }
  }
  return acc;
}

fn godray_blur_accum(
  u: vec4<f32>,
  src: texture_2d<f32>,
  uv: vec2<f32>,
  step: f32,
) -> vec3<f32> {
  let to_sun = (u.xy - uv) * step;
  let decay = clamp(u.z, 0.05, 1.0);
  let src_size = vec2<f32>(textureDimensions(src));
  var acc = vec3<f32>(0.0);
  var wsum = 0.0;
  var w = 1.0;
  for (var i = 1u; i <= GODRAY_TAPS; i = i + 1u) {
    let p = uv + to_sun * (f32(i) / f32(GODRAY_TAPS));
    acc = acc + godray_sample_bilinear(src, src_size, p) * w;
    wsum = wsum + w;
    w = w * decay;
  }
  return acc / max(wsum, 1e-5);
}

fn godray_blur_params() -> vec4<f32> {
  return vec4<f32>(fog_u_blur.misc3.x, fog_u_blur.misc3.y, fog_u_blur.params.x, 0.0);
}

fn godray_pass_blur_a(gid: vec3<u32>) {
  let size = textureDimensions(fog_b_w_blur);
  if (gid.x >= size.x || gid.y >= size.y) { return; }
  let uv = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5))
         / vec2<f32>(f32(size.x), f32(size.y));
  let c = godray_blur_accum(godray_blur_params(), fog_a_r_blur, uv, GODRAY_STEP_A);
  textureStore(fog_b_w_blur, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(c, 1.0));
}

fn godray_pass_blur_b(gid: vec3<u32>) {
  let size = textureDimensions(fog_a_w_blur);
  if (gid.x >= size.x || gid.y >= size.y) { return; }
  let uv = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5))
         / vec2<f32>(f32(size.x), f32(size.y));
  let c = godray_blur_accum(godray_blur_params(), fog_b_r_blur, uv, GODRAY_STEP_B);
  textureStore(fog_a_w_blur, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(c, 1.0));
}
