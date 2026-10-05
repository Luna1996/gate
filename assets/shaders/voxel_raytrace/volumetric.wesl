import package::bindings::{BEAM_BACKOFF, BEAM_DIV, beam_depth, fog_a_r_blur, fog_a_w, fog_a_w_blur, fog_b_r_blur, fog_b_w_blur, fog_u, fog_u_blur, fog_u_read, light_u, view_u};
import package::common::{sky_rgb};
import package::trace::{make_grid};
import package::world::{world_cfg_full, world_raycast};

const SUN_DISK_GAIN: f32 = 3.0;
const LIMB_POW: f32 = 0.5;
const HALO_T: f32 = 10.0;
const HALO_POW: f32 = 2.0;
const SUN_CONE_MAX: f32 = 0.06;
const FOG_DIV: u32 = 4u;
const GODRAY_TAPS: u32 = 12u;
const GODRAY_STEP_A: f32 = 0.35;
const GODRAY_STEP_B: f32 = 0.65;

fn sky_primary(dir: vec3<f32>, sun_dir: vec3<f32>, sun_c: vec3<f32>) -> vec3<f32> {
  let base = sky_rgb();
      if (fog_u_read.misc2.x <= 0.5) { return base; }
  let ang = clamp(fog_u_read.misc2.z, 1e-4, SUN_CONE_MAX);
  let cos_t = dot(dir, sun_dir);
    let t2 = (1.0 - cos_t) / max(1.0 - cos(ang), 1e-12);
  let limb = clamp(1.0 - t2, 0.0, 1.0);
  let disk = pow(limb, LIMB_POW);
  let halo = pow(max(1.0 - t2 / (HALO_T * HALO_T), 0.0), HALO_POW);
  let disk_gain = max(light_u.lights[0].shape.x, 0.0);
  return base
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
  let frustum_length = length(diff_world);
  let dir_voxel = normalize(diff_world);
  let origin_voxel = view_u.cam_pos_voxel.xyz;

        var t_min = 0.0;
  if (view_u.lod.z < 0.5) {
    let beam_size = textureDimensions(beam_depth);
    let bx = gid.x / max(BEAM_DIV / FOG_DIV, 1u);
    let by = gid.y / max(BEAM_DIV / FOG_DIV, 1u);
    t_min = 1e+30;
    for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
      for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
        let sx = clamp(i32(bx) + dx, 0, i32(beam_size.x) - 1);
        let sy = clamp(i32(by) + dy, 0, i32(beam_size.y) - 1);
        t_min = min(t_min, textureLoad(beam_depth, vec2<i32>(sx, sy)).x);
      }
    }
    t_min = max(t_min - BEAM_BACKOFF, 0.0);
  }
  let h = world_raycast(make_grid(0u), origin_voxel, dir_voxel, frustum_length,
                        world_cfg_full(t_min, 3u));

  let sun_dir = light_u.lights[0].kind_pos_dir.yzw;
  let sun_c = light_u.lights[0].color_intensity.xyz * light_u.lights[0].color_intensity.w;
    let sky = select(0.0, 1.0, !h.hit);
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
