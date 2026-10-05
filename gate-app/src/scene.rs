use bevy::{image::Image, prelude::*};
use glam::{IVec3, Vec3};

use gate_render::{
  BrickMapBuilder, DdaCameraConfig, DdaImages, DebugNormals, OrbitCamera, UploadBudget, VIEW_SIZE,
  VoxelScene, create_dda_image,
};
use gate_voxel::{
  Displace, PaletteEntry, PaletteId, VolumeGrid, Volumes, draw_text, fill_box, fill_box_displaced,
  fill_bricks, fill_sphere,
};

use crate::{
  consts::{
    CAM_FAR, CAM_NEAR, DEMO_DISPLACE_AMPLITUDE_OVERRIDE, DEMO_DISPLACE_HEIGHT_MAP,
    DEMO_DISPLACE_SAMPLE, DEMO_DISPLACE_TEX_SCALE, EXT_VOXEL_HALF, EXT_VOXEL_X, EXT_VOXEL_Z, FOV_Y,
    START_CAMERA_SKY, STARTUP_DEMO_SCENE,
  },
  height_field::MaterialDisplace,
  mc, vox_scene,
};

pub(crate) const CUBE_IN_VOID: &str = "cube_in_void";

pub(crate) const INFINITE_CUBES: &str = "infinite_cubes";

pub(crate) fn setup(
  mut commands: Commands,
  mut images: ResMut<Assets<Image>>,
  config: Res<crate::config::Config>,
  mut stream: ResMut<crate::infinite_cubes::Streaming>,
  pbr: Option<Res<gate_render::PbrTextureSet>>,
) {
  let dda_handle = create_dda_image(&mut images);
  commands.spawn((Camera2d, Msaa::Off, UiPickingCamera));
  let theme = std::fs::read_to_string(gate_render::assets_dir().join("lighting/day_outdoor.ron"))
    .map_err(|e| format!("read: {e}"))
    .and_then(|s| gate_render::parse_lighting_ron(&s).map_err(|e| format!("ron: {e}")))
    .unwrap_or_else(|e| {
      bevy::log::warn!("lighting/day_outdoor.ron 加载失败 {e} → 回退内置默认主题");
      Default::default()
    });
  commands.insert_resource(theme);
  commands.insert_resource(DdaImages { target: dda_handle });
  commands.insert_resource(DebugNormals(0));

  let t0 = std::time::Instant::now();
  let mut grid = VolumeGrid::new();
  let mut cam_eye = Vec3::new(1., 0., 0.);
  let mut cam_target = Vec3::new(0., 0., 0.);
  let pbr_ids = pbr_asset_ids(pbr.as_deref());
  let mut start_eye = config.camera.as_ref().map_or(cam_eye, |p| p.to_orbit().eye());
  if STARTUP_DEMO_SCENE {
    paint_demo_palette(&mut grid);
    bevy::log::info!("STEP 1 palette {:?}", t0.elapsed());
    build_demo_scene(&mut grid);
    bevy::log::info!("STEP 2 build_demo_scene {:?}", t0.elapsed());
  } else {
    let name = crate::debug_menu::world_model_name(&crate::debug_menu::load_menu(&config))
      .unwrap_or_else(|| "nuke".to_string());
    (cam_eye, cam_target) = match name.as_str() {
      CUBE_IN_VOID => (Vec3::new(96.0, 64.0, 96.0), Vec3::ZERO),
      mc::MC_MAP => mc_camera(),
      _ => (Vec3::new(406.5, 339.5, 431.5), Vec3::new(551.5, 330.5, 359.5)),
    };
    start_eye = config.camera.as_ref().map_or(cam_eye, |p| p.to_orbit().eye());
    if name == mc::MC_MAP
      && let Some(spawn) = mc::spawn_eye()
      && start_eye.distance_squared(spawn.as_vec3()) > 25_000.0 * 25_000.0
    {
      bevy::log::info!("MC 地图：存档机位 {start_eye} 离出生点 {spawn} 过远 → 落在出生点");
      (cam_eye, cam_target) = mc_camera();
      start_eye = cam_eye;
    }
    let info = build_world(&mut grid, &name, &pbr_ids, start_eye.as_ivec3(), &mut stream)
      .unwrap_or_else(|e| panic!("{name} 加载失败: {e}"));
    bevy::log::info!(
      "STEP 2 world {name} instances={} written={} dropped={} aabb=[{}]-[{}] {:?}",
      info.instances_used,
      info.voxels_written,
      info.voxels_dropped,
      info.aabb_min,
      info.aabb_max,
      t0.elapsed(),
    );
  }
  grid.compact_all();
  bevy::log::info!("STEP 3 compact_all {:?}", t0.elapsed());

  let orbit = if START_CAMERA_SKY {
    OrbitCamera::from_eye(
      Vec3::new(cam_target.x, 320.0, cam_target.z),
      Vec3::new(cam_target.x, 5000.0, cam_target.z),
    )
  } else {
    OrbitCamera::from_eye(cam_eye, cam_target)
  };
  let saved = config.camera;
  let orbit = saved.as_ref().map_or(orbit, |p| p.to_orbit());
  commands.insert_resource(orbit);
  commands.insert_resource(DdaCameraConfig::from_orbit(
    &orbit,
    FOV_Y,
    VIEW_SIZE.x as f32 / VIEW_SIZE.y as f32,
    CAM_NEAR,
    CAM_FAR,
  ));
  commands.insert_resource(saved.map_or(crate::camera::CameraMode::default(), |p| p.mode));
  commands.insert_resource(crate::camera::FlyCamera {
    pos: orbit.eye(),
    speed: crate::consts::FLY_SPEED_DEFAULT,
    fast: false,
  });

  {
    let t1 = std::time::Instant::now();
    let mut words_per_chunk: Vec<(usize, _)> =
      grid.chunk_coords().map(|c| (grid.chunk(c).map(|t| t.len_words()).unwrap_or(0), c)).collect();
    let total_words: usize = words_per_chunk.iter().map(|(w, _)| *w).sum();
    words_per_chunk.sort_unstable_by_key(|(w, _)| std::cmp::Reverse(*w));
    bevy::log::info!(
      "TREE {}MB chunks={} top3={:?}",
      total_words * 4 / 1024 / 1024,
      words_per_chunk.len(),
      &words_per_chunk[..3.min(words_per_chunk.len())],
    );
    let bufs = BrickMapBuilder::build_full(&grid).buffers().clone();
    bevy::log::info!("STEP 4 diag build_full {:?}", t1.elapsed());
    let g = &bufs.globals;
    let n_chunks = grid.chunk_coords().count();
    bevy::log::info!(
      "BRICKMAP chunks={} origin=({},{},{}) dims=({},{},{}) AABB=[{},{},{}]-[{},{},{}]",
      n_chunks,
      g.index_origin_x,
      g.index_origin_y,
      g.index_origin_z,
      g.index_dims_x,
      g.index_dims_y,
      g.index_dims_z,
      g.index_origin_x * 256,
      g.index_origin_y * 256,
      g.index_origin_z * 256,
      (g.index_origin_x + g.index_dims_x as i32) * 256,
      (g.index_origin_y + g.index_dims_y as i32) * 256,
      (g.index_origin_z + g.index_dims_z as i32) * 256,
    );
  }

  let mut volumes = Volumes::new(grid);
  if volumes.main().attach_far() {
    if stream.has_custom_source() {
      crate::infinite_cubes::attach_far_levels_mc(&mut volumes, start_eye.as_ivec3());
    } else {
      crate::infinite_cubes::attach_far_levels(&mut volumes, &pbr_ids, start_eye.as_ivec3());
    }
  }

  commands.insert_resource(VoxelScene {
    volumes,
    demo_force_full_rebuild: true,
    interior_only_edit: false,
    edit_in_flight: false,
    residency_budget_bytes: 0,
  });
  commands
    .insert_resource(UploadBudget { max_bytes_per_frame: 4 * 1024 * 1024, incremental: true });
}

fn build_world(
  grid: &mut VolumeGrid,
  name: &str,
  pbr_ids: &[String],
  cam_eye: IVec3,
  stream: &mut crate::infinite_cubes::Streaming,
) -> Result<vox_scene::VoxSceneInfo, Box<dyn std::error::Error>> {
  if name == CUBE_IN_VOID {
    stream.set_source(None);
    return Ok(build_cube_in_void(grid));
  }
  if name == INFINITE_CUBES {
    stream.set_source(None);
    return Ok(build_infinite_cubes(grid, pbr_ids, cam_eye, stream.coarse_radius));
  }
  if name == mc::MC_MAP {
    let (info, city) = mc::build(grid, cam_eye)?;
    stream.set_source(Some(city));
    return Ok(info);
  }
  stream.set_source(None);
  let anchor = IVec3::new(EXT_VOXEL_HALF, 16, EXT_VOXEL_HALF);
  let path = gate_render::assets_dir().join("vox").join(format!("{name}.vox"));
  vox_scene::load_vox_scene(grid, &path, anchor)
}

fn mc_camera() -> (Vec3, Vec3) {
  let target =
    mc::spawn_eye().unwrap_or(IVec3::new(0, 400, 0)).as_vec3() + Vec3::new(0.0, 80.0, 0.0);
  (target + Vec3::new(-400.0, 400.0, -400.0), target)
}

pub(crate) fn pbr_asset_ids(pbr: Option<&gate_render::PbrTextureSet>) -> Vec<String> {
  pbr.map_or_else(gate_render::material_ids, |s| s.ids().to_vec())
}

pub(crate) fn pbr_asset_count(pbr: Option<&gate_render::PbrTextureSet>) -> usize {
  pbr.map_or_else(|| gate_render::material_ids().len(), |s| s.ids().len())
}

pub(crate) fn reload_world(
  scene: &mut VoxelScene,
  name: &str,
  pbr_ids: &[String],
  cam_eye: Option<IVec3>,
  stream: &mut crate::infinite_cubes::Streaming,
) -> Result<vox_scene::VoxSceneInfo, Box<dyn std::error::Error>> {
  let mut grid = VolumeGrid::new();
  let eye = cam_eye.unwrap_or(IVec3::new(EXT_VOXEL_HALF, 16, EXT_VOXEL_HALF));
  let info = build_world(&mut grid, name, pbr_ids, eye, stream)?;
  grid.compact_all();
  let mut volumes = Volumes::new(grid);
  if volumes.main().attach_far() {
    if stream.has_custom_source() {
      crate::infinite_cubes::attach_far_levels_mc(&mut volumes, eye);
    } else {
      crate::infinite_cubes::attach_far_levels(&mut volumes, pbr_ids, eye);
    }
  }
  scene.volumes = volumes;
  scene.demo_force_full_rebuild = true;
  Ok(info)
}

fn build_cube_in_void(grid: &mut VolumeGrid) -> vox_scene::VoxSceneInfo {
  const EDGE: i32 = 64;
  grid
    .palette_mut()
    .set(PaletteId(1), PaletteEntry { color: [255, 255, 255], ..Default::default() });
  let min = IVec3::splat(-EDGE / 2);
  let voxels_written = fill_box(grid, min, IVec3::splat(EDGE), 1);
  vox_scene::VoxSceneInfo {
    aabb_min: min,
    aabb_max: min + IVec3::splat(EDGE),
    instances_used: 1,
    voxels_written,
    voxels_dropped: 0,
  }
}

fn build_infinite_cubes(
  grid: &mut VolumeGrid,
  pbr_ids: &[String],
  center: IVec3,
  coarse_radius: i32,
) -> vox_scene::VoxSceneInfo {
  use crate::infinite_cubes;
  const WINDOW_CHUNKS: i32 = 32;
  let n_pbr = pbr_ids.len();
  for (id, entry) in infinite_cubes::material_slots(n_pbr) {
    grid.palette_mut().set(id, entry);
  }
  let _ = coarse_radius;
  let win = WINDOW_CHUNKS;
  let origin = center.div_euclid(IVec3::splat(gate_voxel::CHUNK_SIZE)) - IVec3::splat(win);
  grid.set_stream_window(Some((origin, IVec3::splat(win * 2))));
  grid.set_coverage_r(crate::mc::NEAR_COVER_CHUNKS as f32 * gate_voxel::CHUNK_SIZE as f32);
  grid.set_attach_far(true);
  let (lo, hi) = infinite_cubes::initial_box(center);
  let voxels_written =
    infinite_cubes::build_region(grid, lo, hi, n_pbr, gate_voxel::Detail::FULL) as usize;
  vox_scene::VoxSceneInfo {
    aabb_min: lo,
    aabb_max: hi,
    instances_used: 1,
    voxels_written,
    voxels_dropped: 0,
  }
}

fn paint_demo_palette(grid: &mut VolumeGrid) {
  let palette: &[(u16, [u8; 3], u8)] = &[
    (1, [86, 160, 70], 220),
    (2, [140, 108, 76], 200),
    (3, [240, 244, 248], 160),
    (4, [172, 176, 190], 210),
    (5, [52, 168, 72], 210),
    (6, [112, 72, 40], 200),
    (7, [64, 140, 232], 170),
    (8, [68, 230, 220], 150),
    (9, [200, 120, 240], 150),
    (10, [238, 76, 90], 160),
    (11, [248, 210, 72], 160),
    (12, [30, 32, 44], 220),
  ];
  let pal = grid.palette_mut();
  for &(idx, color, rough) in palette {
    pal.set(PaletteId(idx), PaletteEntry { color, roughness: rough, ..Default::default() });
  }
  let led =
    PaletteEntry { color: [255, 214, 156], roughness: 128, emissive: 255, ..Default::default() };
  pal.set(PaletteId(14), led);
}

fn terrain_h(x: i32, z: i32) -> i32 {
  let xf = x as f32;
  let zf = z as f32;
  let s1 = (xf * 0.0045).sin() * (zf * 0.0053).cos();
  let s2 = ((xf + zf) * 0.0020).sin() * 1.3;
  let corner_snow = |cx, cz| {
    let dx = xf - cx as f32;
    let dz = zf - cz as f32;
    let d = (dx * dx + dz * dz).sqrt().max(200.0);
    (600.0 / d).min(1.0) * 780.0
  };
  let sn = corner_snow(0, 0)
    + corner_snow(EXT_VOXEL_X - 1, 0)
    + corner_snow(0, EXT_VOXEL_Z - 1)
    + corner_snow(EXT_VOXEL_X - 1, EXT_VOXEL_Z - 1);
  let hf = 16.0 + s1 * 80.0 + s2 * 120.0 + sn;
  ((hf as i32).clamp(16, 1020)) & !3
}

fn dist_to_center(x: i32, z: i32) -> i32 {
  let dx = x - EXT_VOXEL_HALF;
  let dz = z - EXT_VOXEL_HALF;
  ((dx * dx + dz * dz) as f32).sqrt() as i32
}

fn in_river(x: i32, z: i32) -> bool {
  let t = z as f32 / EXT_VOXEL_Z as f32;
  let river_center = EXT_VOXEL_HALF as f32
    + (t * std::f32::consts::TAU).sin() * 700.0
    + ((t * 12.566).cos() * 180.0);
  let dx = (x as f32) - river_center;
  dx.abs() < 64.0
}

fn build_demo_scene(grid: &mut VolumeGrid) {
  let t0 = std::time::Instant::now();
  macro_rules! mark {
    ($name:expr) => {
      bevy::log::info!("SCENE {} {:?} chunks={}", $name, t0.elapsed(), grid.chunk_count())
    };
  }
  let mut z = 0i32;
  while z < EXT_VOXEL_Z {
    let mut x = 0i32;
    while x < EXT_VOXEL_X {
      let h = terrain_h(x, z);
      let dc = dist_to_center(x, z);
      let in_central_plaza = dc < 700;
      let in_r = in_river(x, z);
      if !in_central_plaza {
        let top = h;
        let snow_line = top - 40;
        if in_r {
          fill_box(grid, IVec3::new(x, 16, z), IVec3::new(16, 8, 16), 2);
          fill_box(grid, IVec3::new(x, 24, z), IVec3::new(16, 8, 16), 7);
        } else if top > 16 {
          let rock_h = (snow_line - 16).max(0).min(top - 16);
          if rock_h > 0 {
            fill_box(grid, IVec3::new(x, 16, z), IVec3::new(16, rock_h, 16), 2);
          }
          if top > 560 {
            let snow_h = top - snow_line;
            if snow_h > 0 {
              fill_box(grid, IVec3::new(x, snow_line, z), IVec3::new(16, snow_h, 16), 3);
            }
          } else {
            let grass_h = (top - 16).min(16);
            if grass_h > 0 && top - grass_h >= 16 {
              fill_box(grid, IVec3::new(x, top - grass_h, z), IVec3::new(16, grass_h, 16), 1);
            }
          }
        }
      } else {
      }
      x += 16;
    }
    z += 16;
  }
  mark!("(1) terrain columns");
  fill_bricks(grid, IVec3::new(0, 0, 0), IVec3::new(EXT_VOXEL_X, 16, EXT_VOXEL_Z), 16, 1);
  mark!("(1b) floor");

  fill_box(grid, IVec3::new(0, 16, EXT_VOXEL_HALF - 32), IVec3::new(EXT_VOXEL_X, 8, 32), 12);
  draw_text(grid, IVec3::new(96, 32, 256), "GATE ENGINE", 8);
  mark!("(2) road+text");

  {
    let mut n_planted = 0usize;
    let mut i = 0i32;
    while n_planted < 170 && i < 2000 {
      let x = (i * 211 + 83) % EXT_VOXEL_X;
      let z = ((i * 977 + 419) ^ 0xA53) % EXT_VOXEL_Z;
      let x = x.abs();
      let z = z.abs();
      let h = terrain_h(x, z);
      let ok = dist_to_center(x, z) > 900
        && !in_river(x, z)
        && !((EXT_VOXEL_HALF - 32)..=(EXT_VOXEL_HALF + 32)).contains(&z)
        && h < 360;
      if ok {
        fill_box(grid, IVec3::new(x, h, z), IVec3::new(32, 80, 32), 6);
        fill_sphere(grid, IVec3::new(x + 16, h + 80 + 64, z + 16), 80, 5);
        n_planted += 1;
      }
      i += 1;
    }
  }
  mark!("(4) forest");

  let crystal_clusters: &[(i32, i32, i32, u8, u8, i32)] =
    &[(480, 480, 90, 8, 9, 131), (4608, 640, 80, 10, 8, 251), (768, 4352, 70, 9, 10, 223)];
  for &(cx_, cz_, cnt, pa, pb, seed) in crystal_clusters.iter() {
    if cx_ >= EXT_VOXEL_X || cz_ >= EXT_VOXEL_Z {
      continue;
    }
    let h = terrain_h(cx_, cz_);
    let mut t = 0i32;
    let mut placed = 0usize;
    while placed < cnt as usize {
      let sx = ((t * 37 + seed) % 96) - 48;
      let sz = ((t * 131 + seed * 3) % 96) - 48;
      let hy = (t * 53) % 14;
      let pal = if (t & 1) == 0 { pa } else { pb };
      let px_ = cx_ + sx;
      let pz_ = cz_ + sz;
      if px_ >= 0 && pz_ >= 0 && px_ < EXT_VOXEL_X && pz_ < EXT_VOXEL_Z {
        fill_box(grid, IVec3::new(px_, h + hy, pz_), IVec3::new(1, 1, 1), pal);
        placed += 1;
      }
      t += 1;
    }
  }
  mark!("(5) crystals");

  fill_box(grid, IVec3::new(656, 64, 64), IVec3::new(32, 32, 32), 11);
  if 1264 + 32 <= EXT_VOXEL_X {
    fill_box(grid, IVec3::new(1264, 240, 240), IVec3::new(32, 32, 32), 7);
  }
  let t0 = gate_voxel::ChunkCoord::new(0, 0, 0);
  grid.set_state(0, 1, 0xDEAD);
  grid.set_state(1, 0, 0xBEEF);
  grid.set_comp(t0, 0, 0, 0, 0x1122);
  grid.set_comp(t0, 1, 0, 0, 0x3344);
  mark!("(6) hotspots+state");

  if DEMO_DISPLACE_SAMPLE {
    build_displace_sample(grid);
    mark!("(7) MT6 displacement sample");
  }
}

fn build_displace_sample(grid: &mut VolumeGrid) {
  let extent = IVec3::new(64, 48, 32);
  let plain = IVec3::new(480, 16, 380);
  let bumped = IVec3::new(480, 16, 420);

  let base_voxels = fill_box(grid, plain, extent, 4);

  let sample = match MaterialDisplace::load(
    DEMO_DISPLACE_HEIGHT_MAP,
    DEMO_DISPLACE_TEX_SCALE,
    DEMO_DISPLACE_AMPLITUDE_OVERRIDE,
  ) {
    Ok(Some(s)) => s,
    Ok(None) => {
      let n = fill_box(grid, bumped, extent, 4);
      bevy::log::warn!(
        target: "gate",
        "MT8-5 位移样例：材质 `{DEMO_DISPLACE_HEIGHT_MAP}` 幅度 = 0（资产值）⇒ 右台不位移\
         （普通 CSG，写入 {n} 体素）"
      );
      return;
    }
    Err(e) => {
      let n = fill_box(grid, bumped, extent, 4);
      bevy::log::warn!(
        target: "gate",
        "MT8-5 位移样例：材质 `{DEMO_DISPLACE_HEIGHT_MAP}` 高度图不可用（{e}）⇒ 右台普通 CSG\
         （写入 {n} 体素）；放回 assets/textures/pbr/{DEMO_DISPLACE_HEIGHT_MAP}/\
         {DEMO_DISPLACE_HEIGHT_MAP}_height.png 即恢复"
      );
      return;
    }
  };
  let f = sample.displace_fn();
  let bound = sample.bound();
  let t0 = std::time::Instant::now();
  let st = fill_box_displaced(grid, bumped, extent, 4, Some(Displace { f: &f, bound }));
  let (lo, hi) = sample.field().range();
  let size = sample.field().size();
  bevy::log::info!(
    target: "gate",
    "MT6/MT8-5 位移样例 基准台 @{plain}+{extent} 普通 CSG {base_voxels} 体素 | \
     位移台 @{bumped}+{extent} 高度图 {DEMO_DISPLACE_HEIGHT_MAP} {w}×{h} texel {lo:.3}..{hi:.3} \
     幅度 {} 体素 ±{bound} 铺 {DEMO_DISPLACE_TEX_SCALE} 体素 ⇒ {} 体素 整块 {}={} 壳层 {} {:?}",
    sample.amplitude(),
    st.voxels,
    st.whole_bricks,
    st.whole_bricks * 64,
    st.shell_voxels,
    t0.elapsed(),
    w = size.x,
    h = size.y,
  );
  log_sample_camera(plain, extent);
}

fn log_sample_camera(plain: IVec3, extent: IVec3) {
  let target = (plain + extent / 2).as_vec3();
  let eye = target + Vec3::new(150.0, 90.0, 150.0);
  let orbit = OrbitCamera::from_eye(eye, target);
  bevy::log::info!(
    target: "gate",
    "MT6 样例机位：mode=\"Fly\" eye=[{:.1},{:.1},{:.1}] yaw={:.4} pitch={:.4} distance={:.1}；\
     普通 CSG @[{},{},{}]+{extent}，位移版 @[{},{},{}]+{extent}",
    eye.x,
    eye.y,
    eye.z,
    orbit.yaw,
    orbit.pitch,
    orbit.distance,
    plain.x,
    plain.y,
    plain.z,
    plain.x,
    plain.y,
    plain.z + extent.z + 8,
  );
}
