import package::bindings::{gi_sec_slots, light_u};
import package::common::{MATERIAL_DEBUG_NONE, SurfaceLight, brdf_reflected, emissive_radiance, fetch_material, sky_rgb};
import package::gi::common::{GI_SEC_WORDS, GI_SEC_W_K0, GI_SEC_W_K1, GI_SEC_W_RAD, gi_cos_hemisphere, gi_key0, gi_key1, gi_key_ok, gi_local_rev, gi_rand2, gi_sec_key_masked, gi_sec_slot_of};
import package::gi::consts::{GI_EMIT_GAIN, GI_PI, GI_SKY_AMBIENT, GI_SKY_RADIANCE_SCALE, GI_T_MAX, SHADOW_SURFACE_EPS};
import package::trace::{SECONDARY_REACH_CHUNKS, make_grid};
import package::world::{WorldHit, surface_offset, voxel_center_world, voxel_normal, world_trace_medium_scene};

const GI_REACH_T: f32 = SECONDARY_REACH_CHUNKS * 256.0;

fn sun_transmittance(origin: vec3<f32>, sun_dir: vec3<f32>, self_voxel: vec3<i32>, self_obj: i32) -> f32 {
        if (((u32(bitcast<i32>(origin.x * 64.0)) ^ u32(bitcast<i32>(sun_dir.x * 4096.0))) & 63u) == 0u) {
    atomicAdd(&lod_diag[15u], 64u);
  }
  var tint = vec3<f32>(1.0);
          let h = world_trace_medium_scene(origin, sun_dir, min(GI_T_MAX, GI_REACH_T), false, &tint);
  let through = (tint.r + tint.g + tint.b) / 3.0;
    if (!h.hit) { return through; }
  if (all(h.voxel == self_voxel) && h.obj_id == self_obj) { return 1.0; }
  return 0.0;
}

fn gi_face_shade(sh: WorldHit, org: vec3<f32>, sun_dir: vec3<f32>, sun_c: vec3<f32>,
                 sun_bounce: f32) -> vec3<f32> {
  let hg = make_grid(u32(sh.obj_id) + 1u);
  let hp = voxel_center_world(hg, sh.voxel);
  // WHY: 间接光是低频量。sun_bounce 关时 sun_vis=0，brdf_reflected 不走太阳项，法线只影响
  // 材质法线贴图的细节；用遍历给出的面法线即可，省掉 6 次分裂树占据采样。
  let hn = sh.n;
  let ndl = max(dot(hn, sun_dir), 0.0);
  var sun = 0.0;
    if (sun_bounce > 0.0 && ndl > 0.0) {
    let h_off = surface_offset(hg, hn, SHADOW_SURFACE_EPS);
    sun = sun_transmittance(hp + hn * h_off, sun_dir, sh.voxel, sh.obj_id);
  }
    let amb = light_u.sky_color.xyz * GI_SKY_AMBIENT;
                    let v = hn;
  let m = fetch_material(sh.palette_base, sh.pal, hp, hn, MATERIAL_DEBUG_NONE);
  return brdf_reflected(m, hn, v, SurfaceLight(sun_dir, sun_c, sun * sun_bounce, amb, vec3<f32>(0.0)))
         / GI_PI
       + emissive_radiance(m, GI_EMIT_GAIN);
}

fn gi_face_shade_cached(sh: WorldHit, org: vec3<f32>, sun_dir: vec3<f32>, sun_c: vec3<f32>,
                        sun_bounce: f32, epoch: u32, k0: u32, k1: u32, key_ok: bool) -> vec3<f32> {
  var l = vec3<f32>(0.0);
  var slot_ok = false;
  var sb = 0u;
  var mine = vec2<u32>(0u, 0u);
  if (key_ok) {
    let slot = gi_sec_slot_of(k0, k1, &slot_ok);
    if (slot_ok) {
      sb = slot * GI_SEC_WORDS;
                  let rev = select(0u, gi_local_rev(sh.voxel, sh.obj_id), sun_bounce > 0.5);
      mine = gi_sec_key_masked(k0, k1, epoch, sh.pal, rev);
      if (atomicLoad(&gi_sec_slots[sb + GI_SEC_W_K0]) == mine.x
          && atomicLoad(&gi_sec_slots[sb + GI_SEC_W_K1]) == mine.y) {
        let r = vec3<f32>(
          bitcast<f32>(atomicLoad(&gi_sec_slots[sb + GI_SEC_W_RAD])),
          bitcast<f32>(atomicLoad(&gi_sec_slots[sb + GI_SEC_W_RAD + 1u])),
          bitcast<f32>(atomicLoad(&gi_sec_slots[sb + GI_SEC_W_RAD + 2u])));
                        if (atomicLoad(&gi_sec_slots[sb + GI_SEC_W_K0]) == mine.x
            && atomicLoad(&gi_sec_slots[sb + GI_SEC_W_K1]) == mine.y) {
          return r;
        }
      }
    }
  }
  l = gi_face_shade(sh, org, sun_dir, sun_c, sun_bounce);
      if (slot_ok) {
    let c0 = atomicLoad(&gi_sec_slots[sb + GI_SEC_W_K0]);
    let c1 = atomicLoad(&gi_sec_slots[sb + GI_SEC_W_K1]);
    if (c0 != mine.x || c1 != mine.y) {
      atomicStore(&gi_sec_slots[sb + GI_SEC_W_RAD], bitcast<u32>(l.x));
      atomicStore(&gi_sec_slots[sb + GI_SEC_W_RAD + 1u], bitcast<u32>(l.y));
      atomicStore(&gi_sec_slots[sb + GI_SEC_W_RAD + 2u], bitcast<u32>(l.z));
      atomicStore(&gi_sec_slots[sb + GI_SEC_W_K0], mine.x);
      atomicStore(&gi_sec_slots[sb + GI_SEC_W_K1], mine.y);
    }
  }
  return l;
}

fn gi_bounce2_probe(org: vec3<f32>, dir: vec3<f32>, sun_dir: vec3<f32>, sun_c: vec3<f32>,
                    sun_bounce: f32, epoch: u32) -> vec3<f32> {
  var tint = vec3<f32>(1.0);
  let h = world_trace_medium_scene(org, dir, min(GI_T_MAX, GI_REACH_T), false, &tint);
  if (!h.hit) {
    return sky_rgb() * GI_SKY_RADIANCE_SCALE * tint;
  }
  let k0 = gi_key0(h.voxel, h.obj_id);
  let k1 = gi_key1(h.voxel, h.face_id, h.obj_id);
  return gi_face_shade_cached(h, org, sun_dir, sun_c, sun_bounce, epoch, k0, k1, gi_key_ok(h.voxel, h.obj_id))
         * tint;
}

fn gi_bounce2(sh: WorldHit, org: vec3<f32>, sun_dir: vec3<f32>, sun_c: vec3<f32>,
              sun_bounce: f32, epoch: u32, seed: u32, mult: f32) -> vec3<f32> {
  let hg = make_grid(u32(sh.obj_id) + 1u);
  let hp = voxel_center_world(hg, sh.voxel);
  let hn = voxel_normal(hg, sh.voxel, sh.face_id, sh.n);
  let m = fetch_material(sh.palette_base, sh.pal, hp, hn, MATERIAL_DEBUG_NONE);
  let off = surface_offset(hg, hn, SHADOW_SURFACE_EPS);
  let d2 = gi_cos_hemisphere(hn, gi_rand2(seed));
  let l2 = gi_bounce2_probe(hp + hn * off, d2, sun_dir, sun_c, sun_bounce, epoch);
    return (1.0 - m.metallic) * m.albedo * l2 * mult;
}

fn gi_ray_radiance(org: vec3<f32>, dir: vec3<f32>, sun_dir: vec3<f32>, sun_c: vec3<f32>,
                    sun_bounce: f32, epoch: u32, bounce2: f32, seed: u32) -> vec3<f32> {
      if (((u32(bitcast<i32>(org.z * 64.0)) ^ u32(bitcast<i32>(dir.x * 4096.0))) & 63u) == 0u) {
    atomicAdd(&lod_diag[14u], 64u);
  }
  var tint = vec3<f32>(1.0);
  let sh = world_trace_medium_scene(org, dir, min(GI_T_MAX, GI_REACH_T), false, &tint);
    if (!sh.hit) {
    return sky_rgb() * GI_SKY_RADIANCE_SCALE * tint;
  }
  let k0 = gi_key0(sh.voxel, sh.obj_id);
  let k1 = gi_key1(sh.voxel, sh.face_id, sh.obj_id);
  let key_ok = gi_key_ok(sh.voxel, sh.obj_id);
  var l = gi_face_shade_cached(sh, org, sun_dir, sun_c, sun_bounce, epoch, k0, k1, key_ok);
  if (bounce2 > 0.0) {
    l = l + gi_bounce2(sh, org, sun_dir, sun_c, sun_bounce, epoch, seed, bounce2);
  }
  return l * tint;
}