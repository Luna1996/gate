//! 场景搭建：Startup 系统 setup（相机/资源/诊断）+ 运行期换世界 `reload_world` + demo 极限场景生成。
//! 启动场景与规模见 `consts`（`STARTUP_DEMO_SCENE` / `DEMO_TILES`）。

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
  vox_scene,
};

/// 程序化调试场景名（**不来自磁盘**，见 `build_cube_in_void`）：与 `assets/vox/*.vox` 的名字同列在
/// 「世界」页的模型下拉里，由 [`build_world`] 按名分发。
pub(crate) const CUBE_IN_VOID: &str = "cube_in_void";

pub(crate) fn setup(
  mut commands: Commands,
  mut images: ResMut<Assets<Image>>,
  config: Res<crate::config::Config>,
) {
  let dda_handle = create_dda_image(&mut images);
  // `UiPickingCamera`：UI 拾取（hover/press）只认挂了它的相机（见 gate_ui::pointer 的 `require_markers` 契约）
  commands.spawn((Camera2d, Msaa::Off, UiPickingCamera));
  // 光照主题 RON：同步读即可，缺失/解析失败回退内置默认主题。**它只提供首帧初值** ——
  // 之后每帧被 `gate_render::sky::apply_sky` 按时刻/年积日/纬度覆写（天上只有时间驱动这一条路）。
  let theme = std::fs::read_to_string(gate_render::assets_dir().join("lighting/day_outdoor.ron"))
    .map_err(|e| format!("read: {e}"))
    .and_then(|s| gate_render::parse_lighting_ron(&s).map_err(|e| format!("ron: {e}")))
    .unwrap_or_else(|e| {
      bevy::log::warn!("lighting/day_outdoor.ron 加载失败 {e} → 回退内置默认主题");
      Default::default()
    });
  commands.insert_resource(theme);
  commands.insert_resource(DdaImages { target: dda_handle });
  // 3 = unlit（跳过全部光照，albedo 直出）
  commands.insert_resource(DebugNormals(3));

  let t0 = std::time::Instant::now();
  let mut grid = VolumeGrid::new();
  let mut cam_eye = Vec3::new(1., 0., 0.);
  let mut cam_target = Vec3::new(0., 0., 0.);
  if STARTUP_DEMO_SCENE {
    paint_demo_palette(&mut grid);
    bevy::log::info!("STEP 1 palette {:?}", t0.elapsed());
    build_demo_scene(&mut grid);
    bevy::log::info!("STEP 2 build_demo_scene {:?}", t0.elapsed());
  } else {
    // 启动世界 = 「游戏/世界」页模型下拉的最终选中项（结构来自资产、选中项来自配置；读不到 → nuke）
    let name = crate::debug_menu::world_model_name(&crate::debug_menu::load_menu(&config))
      .unwrap_or_else(|| "nuke".to_string());
    let info = build_world(&mut grid, &name).unwrap_or_else(|e| panic!("{name} 加载失败: {e}"));
    // 默认机位：`cube_in_void` 的立方体（原点、边长 64）从斜上方看；其余 .vox 沿用既有读数
    (cam_eye, cam_target) = if name == CUBE_IN_VOID {
      (Vec3::new(96.0, 64.0, 96.0), Vec3::ZERO)
    } else {
      (Vec3::new(406.5, 339.5, 431.5), Vec3::new(551.5, 330.5, 359.5))
    };
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
  grid.compact_all(); // GC：回收编辑过程累积的废弃节点
  bevy::log::info!("STEP 3 compact_all {:?}", t0.elapsed());

  // 轨道相机为唯一相机状态源，DdaCameraConfig 由 `from_orbit` 生成（初始机位 = 场景中心俯视）。
  let orbit = if START_CAMERA_SKY {
    OrbitCamera::from_eye(
      Vec3::new(cam_target.x, 320.0, cam_target.z),
      Vec3::new(cam_target.x, 5000.0, cam_target.z),
    )
  } else {
    OrbitCamera::from_eye(cam_eye, cam_target)
  };
  // 上次退出时的姿态优先（`data/config.toml` 的 `[camera]` 节）；首启 / 缺该节 → 场景默认机位。
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
  // 幽灵飞行相机：起点 = 轨道眼位，切模式时视野原地不动（见 camera::sync_camera_mode_switch）。
  // 两种情况都必须与 `orbit.eye()` 一致，否则首帧的 `sync_camera_mode_switch` 会把眼位拽回去。
  commands.insert_resource(saved.map_or(crate::camera::CameraMode::default(), |p| p.mode));
  commands.insert_resource(crate::camera::FlyCamera {
    pos: orbit.eye(),
    speed: crate::consts::FLY_SPEED_DEFAULT,
    fast: false, // 缺省低速档
  });

  // 诊断：打印 brickmap globals
  {
    let t1 = std::time::Instant::now();
    // 树输出量诊断：serialize 总字数 + 最大 chunk
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

  // 物体 = 普通 VolumeGrid，经 `Volumes.add_object()` 注册变换，走与主世界相同的 dirty → builder → upload 路径。
  let volumes = Volumes::new(grid);

  commands.insert_resource(VoxelScene { volumes, demo_force_full_rebuild: true });
  commands
    .insert_resource(UploadBudget { max_bytes_per_frame: 4 * 1024 * 1024, incremental: true });
}

/// 按名字把主世界体素写进 `grid`（`setup` 与 `reload_world` 的唯一分发点）：
/// `cube_in_void` = 程序化调试场景（[`build_cube_in_void`]）；其余 = `assets/vox/<name>.vox`。
/// 不做 `compact_all` —— GC 时机由调用方定（`setup` 在场景构建后统一做一次）。
fn build_world(
  grid: &mut VolumeGrid,
  name: &str,
) -> Result<vox_scene::VoxSceneInfo, Box<dyn std::error::Error>> {
  if name == CUBE_IN_VOID {
    return Ok(build_cube_in_void(grid));
  }
  let anchor = IVec3::new(EXT_VOXEL_HALF, 16, EXT_VOXEL_HALF);
  let path = gate_render::assets_dir().join("vox").join(format!("{name}.vox"));
  vox_scene::load_vox_scene(grid, &path, anchor)
}

/// 运行期换世界（DebugMenu「游戏/世界/重载世界」）：按名字重建主世界（`cube_in_void` 见
/// [`build_cube_in_void`]，其余走 `assets/vox/<name>.vox`）。
/// 与 `setup` 同一套不变量：先 `compact_all`；`demo_force_full_rebuild` 触发全量重建 + 全量 GPU 上传。
/// 失败 → 原世界保持不变；相机不动（所有模型锚到同一 anchor）。
pub(crate) fn reload_world(
  scene: &mut VoxelScene,
  name: &str,
) -> Result<vox_scene::VoxSceneInfo, Box<dyn std::error::Error>> {
  let mut grid = VolumeGrid::new();
  let info = build_world(&mut grid, name)?;
  grid.compact_all();
  scene.volumes = Volumes::new(grid);
  scene.demo_force_full_rebuild = true;
  Ok(info)
}

/// 程序化调试场景（「世界」页模型下拉里的 `cube_in_void`）：**空场景** + 原点处的 64³ 实心立方体，
/// 槽 1 = `ffffff` 平凡材质。世界里除这一个立方体外没有任何东西，供"在简单场景里复现 bug"用。
/// 立方体居中于原点（占据 `[-32, 32)³`，边长 64 voxel = 1.28m @50 voxel/m）。
fn build_cube_in_void(grid: &mut VolumeGrid) -> vox_scene::VoxSceneInfo {
  const EDGE: i32 = 64;
  // 槽 1：颜色 ffffff 的平凡变体（`PaletteEntry::default` 的其余字段：不发光 / 不透射 / 非金属 / 非 PBR）
  grid.palette_mut().set(PaletteId(1), PaletteEntry { color: [255, 255, 255], ..Default::default() });
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

/// demo 调色板：1..=12 地形/岩石色 + 14 号 LED 灯柱（`PaletteEntry::default` + 逐字段赋值）。
/// （原 13 号「浮空岛岛底」随浮空岛一起删除；槽 13 现在是空的默认色。）
fn paint_demo_palette(grid: &mut VolumeGrid) {
  let palette: &[(u16, [u8; 3], u8)] = &[
    (1, [86, 160, 70], 220),   // 1 草地（L0 地面平原）
    (2, [140, 108, 76], 200),  // 2 山岩（山体主体）
    (3, [240, 244, 248], 160), // 3 雪峰（y > 山线顶）
    (4, [172, 176, 190], 210), // 4 石材（MT6 位移样例的石台）
    (5, [52, 168, 72], 210),   // 5 树叶（球冠）
    (6, [112, 72, 40], 200),   // 6 树干（细柱）
    (7, [64, 140, 232], 170),  // 7 河流蓝（L2 缠绕）
    (8, [68, 230, 220], 150),  // 8 水晶青（L4 高亮）
    (9, [200, 120, 240], 150), // 9 水晶紫（L4 高亮）
    (10, [238, 76, 90], 160),  // 10 水晶红（L4 高亮）
    (11, [248, 210, 72], 160), // 11 塔顶金（L4 高亮）
    (12, [30, 32, 44], 220),   // 12 道路/桥面（深灰）
  ];
  let pal = grid.palette_mut();
  for &(idx, color, rough) in palette {
    pal.set(PaletteId(idx), PaletteEntry { color, roughness: rough, ..Default::default() });
  }
  // 14 号 LED 灯柱（暖白满档发光）
  let led =
    PaletteEntry { color: [255, 214, 156], roughness: 128, emissive: 255, ..Default::default() };
  pal.set(PaletteId(14), led);
}

// 极限场景规模见 `consts`（世界边长 = tile 数 × 512 voxel；brickmap compute_window dims = (N+2, 5, N+2)）。

/// 正弦高度场（确定性，不用 rand）：
/// h(x,z) = 16 + A·sin(x·k1)·cos(z·k2) + 山体距 4 角的反比隆起
fn terrain_h(x: i32, z: i32) -> i32 {
  let xf = x as f32;
  let zf = z as f32;
  let s1 = (xf * 0.0045).sin() * (zf * 0.0053).cos(); // 基础波
  let s2 = ((xf + zf) * 0.0020).sin() * 1.3; // 对角长波
  // 四角雪峰：与四个角的距离反比
  let corner_snow = |cx, cz| {
    let dx = xf - cx as f32;
    let dz = zf - cz as f32;
    let d = (dx * dx + dz * dz).sqrt().max(200.0);
    (600.0 / d).min(1.0) * 780.0 // 山顶 ~800
  };
  let sn = corner_snow(0, 0)
    + corner_snow(EXT_VOXEL_X - 1, 0)
    + corner_snow(0, EXT_VOXEL_Z - 1)
    + corner_snow(EXT_VOXEL_X - 1, EXT_VOXEL_Z - 1);
  let hf = 16.0 + s1 * 80.0 + s2 * 120.0 + sn;
  // 4 对齐：高度层取整到 4³ 块（未对齐会退化为逐体素边缘）；阶梯化 4 级。
  ((hf as i32).clamp(16, 1020)) & !3
}

/// 离世界中心（= 中央广场圆心）的距离
fn dist_to_center(x: i32, z: i32) -> i32 {
  let dx = x - EXT_VOXEL_HALF;
  let dz = z - EXT_VOXEL_HALF;
  ((dx * dx + dz * dz) as f32).sqrt() as i32
}

/// 中央峡谷蜿蜒河（x,z 在河道走廊内返回 true）
fn in_river(x: i32, z: i32) -> bool {
  let t = z as f32 / EXT_VOXEL_Z as f32; // 0..1
  let river_center = EXT_VOXEL_HALF as f32 // 世界中线
    + (t * std::f32::consts::TAU).sin() * 700.0 // 正弦摆 ±700
    + ((t * 12.566).cos() * 180.0); // 次级摆幅
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
  // (1) 基础地形：按 16 voxel (L0) 步长采样高度场 → fill_box 铺柱（y=16..h 用 pal 2，h>560 顶部改 pal 3）
  let mut z = 0i32;
  while z < EXT_VOXEL_Z {
    let mut x = 0i32;
    while x < EXT_VOXEL_X {
      let h = terrain_h(x, z);
      let dc = dist_to_center(x, z);
      // 中央广场 (radius<700) 不开地形：只留全地图 L0 草地地板（y=0..16）⇒ 一片平坦广场。
      // （这里以前是「城堡基座，由 (3) 的天空之城接管」；浮空岛删除后它就是一个空广场。）
      let in_central_plaza = dc < 700;
      // 河道：y=16..24 填 palette 7，不叠加山岩
      let in_r = in_river(x, z);
      if !in_central_plaza {
        // 只填 16..h 山体（y=0..16 地面稍后全地图统一填）
        let top = h;
        let snow_line = top - 40; // 4 对齐（top 已对齐 4）
        if in_r {
          // 河床：4 对齐（岩 8 + 蓝 8 = y 16..32）
          fill_box(grid, IVec3::new(x, 16, z), IVec3::new(16, 8, 16), 2);
          fill_box(grid, IVec3::new(x, 24, z), IVec3::new(16, 8, 16), 7);
        } else if top > 16 {
          // 山体主体（palette 2 山岩）
          let rock_h = (snow_line - 16).max(0).min(top - 16);
          if rock_h > 0 {
            fill_box(grid, IVec3::new(x, 16, z), IVec3::new(16, rock_h, 16), 2);
          }
          // 雪顶（h > 560 时，顶部 40 voxel 改 palette 3 雪）
          if top > 560 {
            let snow_h = top - snow_line;
            if snow_h > 0 {
              fill_box(grid, IVec3::new(x, snow_line, z), IVec3::new(16, snow_h, 16), 3);
            }
          } else {
            // 低矮山坡顶部覆草（palette 1，最顶 16 voxel）
            let grass_h = (top - 16).min(16);
            if grass_h > 0 && top - grass_h >= 16 {
              fill_box(grid, IVec3::new(x, top - grass_h, z), IVec3::new(16, grass_h, 16), 1);
            }
          }
        }
      } else {
        // 中央广场：不铺山体，只靠稍后的全局 L0 草地地板（y=0..16）
      }
      x += 16;
    }
    z += 16;
  }
  mark!("(1) terrain columns");
  // 全地图 L0 基础地板（y=0..16，pal 1 草地）
  fill_bricks(grid, IVec3::new(0, 0, 0), IVec3::new(EXT_VOXEL_X, 16, EXT_VOXEL_Z), 16, 1);
  mark!("(1b) floor");

  // (2) 中央大道 + 入口大标语 "GATE ENGINE"
  // 中央大道（X 向，中线 ±32 宽）深灰铺路
  fill_box(grid, IVec3::new(0, 16, EXT_VOXEL_HALF - 32), IVec3::new(EXT_VOXEL_X, 8, 32), 12);
  // 大标语（L1，每像素 8³）
  draw_text(grid, IVec3::new(96, 32, 256), "GATE ENGINE", 8);
  mark!("(2) road+text");

  // (3) 【已删除】原「中央天空之城」：浮空岛倒锥 + 城墙/角楼/正殿/高塔/旗帜。
  //     删除原因：**浮空岛已过时** —— 场景现在从 `.vox` 读取（`vox_scene::load_vox_scene`），
  //     这座程序化天空之城只是早期的规模/精度展台，且它的岛底是半径最大 560 的**实心球**，
  //     会把中庭连默认机位一起包在固体里（`EDIT SELFTEST` 实测射线 t=0 命中相机自身）。
  //     ⇒ 下面是空出来的**中央广场**：`(1)` 里 `dc < 700` 不铺山体，只留全地图 L0 草地地板（y=0..16）。
  //     **编号保持不重排**，以对齐既有的 `SCENE (n) ...` 日志与本文件/计划里的引用。

  // (4) 森林：~160 棵确定性散点树（双素数线性同余，不用 rand）= 树干（L1）+ 叶球（L2，r=80）；
  // 选址：中央广场外、非河道、非大道、h<360。
  {
    let mut n_planted = 0usize;
    let mut i = 0i32;
    while n_planted < 170 && i < 2000 {
      // 确定性 2 互素线性同余 → (x,z) 伪散点
      let x = (i * 211 + 83) % EXT_VOXEL_X;
      let z = ((i * 977 + 419) ^ 0xA53) % EXT_VOXEL_Z;
      let x = x.abs();
      let z = z.abs();
      let h = terrain_h(x, z);
      // 广场保持空旷：MT6 的位移样例就摆在广场地面上，需要前方无遮挡
      let ok = dist_to_center(x, z) > 900
        && !in_river(x, z)
        && !((EXT_VOXEL_HALF - 32)..=(EXT_VOXEL_HALF + 32)).contains(&z) // 中央大道
        && h < 360;
      if ok {
        // 树干（L1）
        fill_box(grid, IVec3::new(x, h, z), IVec3::new(32, 80, 32), 6);
        // 树叶球（L2，r=80）
        fill_sphere(grid, IVec3::new(x + 16, h + 80 + 64, z + 16), 80, 5);
        n_planted += 1;
      }
      i += 1;
    }
  }
  mark!("(4) forest");

  // (5) 3 处 L4 水晶矿簇（每处 60~100 个 1³ 彩色小立方体，密集堆叠）
  let crystal_clusters: &[(i32, i32, i32, u8, u8, i32)] = &[
    // (cx, cz, count, pal_a, pal_b, seed)
    (480, 480, 90, 8, 9, 131),   // 左上 水晶青/紫
    (4608, 640, 80, 10, 8, 251), // 右上 水晶红/青
    (768, 4352, 70, 9, 10, 223), // 左下 水晶紫/红
  ];
  for &(cx_, cz_, cnt, pa, pb, seed) in crystal_clusters.iter() {
    if cx_ >= EXT_VOXEL_X || cz_ >= EXT_VOXEL_Z {
      continue; // 簇中心在世界外时跳过（避免 t 无限增长溢出）
    }
    let h = terrain_h(cx_, cz_);
    let mut t = 0i32;
    let mut placed = 0usize;
    while placed < cnt as usize {
      // 确定性位置抖动
      let sx = ((t * 37 + seed) % 96) - 48;
      let sz = ((t * 131 + seed * 3) % 96) - 48;
      let hy = (t * 53) % 14; // 高度 0..13 voxel
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

  // (6) 静态 L4 32³ 精度热点：tile(1,0,0) 金色 + tile(2,0,0) 蓝
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

  // (7) MT6-4 · 材质位移样例（开关见 `consts::DEMO_DISPLACE_SAMPLE`）
  if DEMO_DISPLACE_SAMPLE {
    build_displace_sample(grid);
    mark!("(7) MT6 displacement sample");
  }
}

/// MT6-4 + **MT8-5** · **材质位移样例**：一对同尺寸同材质的石台 —— 一座走既有 CSG（位移关闭）、
/// 一座**按材质资产里的位移幅度** + `consts::DEMO_DISPLACE_HEIGHT_MAP` 的高度图位移出**真实体素**凹凸，
/// 一眼可对照。
///
/// **MT8-5 的变化（决策 B = 甲）**：幅度**不再来自** `consts::DEMO_DISPLACE_AMPLITUDE`，
/// 而是由 [`MaterialDisplace::load`] 按材质 id 从**资产**里读（`MaterialAsset::emissive_metal`
/// 的 bits 24..31，`gate-render/src/pbr_texture.rs::DISPLACE_DEMO_AMPLITUDE`）⇒
/// **改资产里的幅度即改凹凸**；`consts::DEMO_DISPLACE_AMPLITUDE_OVERRIDE` 只剩"实验时压过资产"的
/// 地位（默认 `None` = 完全由资产决定）。**左台（对照）与位移语义都没动**：
/// 左台仍走普通 `fill_box`，右台仍是同一个 `fill_box_displaced` + 同一套高度图语义。
///
/// **位置**：中央广场地面（y=16 起）—— 浮空岛删除后广场是空的，这里一览无遗，
/// 不再需要"爬到塔帽顶、再换机位绕开实体"。
/// （历史：样例原先挂在西北角楼的塔帽顶，因为当时浮空岛倒锥是半径最大 560 的实心球、
/// 把中庭连默认机位一起包住；浮空岛删除后这个约束随之消失。）
///
/// **材质**：demo 调色板 4（石材，纯色）—— 凹凸**全部来自几何**，不靠任何贴图着色
/// （`docs/PLAN.md` §3 D2：不 bake 法线、不用法线贴图造假凹凸；凹槽的暗部由真实几何的 GI 遮蔽给出）。
///
/// **MT6-5 · 编辑语义（一次性产物）**：位移在这里一次性做完，产物就是普通体素（palette 4）——
/// 之后 `edit.rs` 的笔触按 `set_voxel` 覆盖它们（`EDIT[place]` = 普通体素、`EDIT[erase]` = 普通空格），
/// **不存在**"这块是位移出来的"这种状态，也就不存在"位移重算 / 位移与编辑打架"的路径。
/// （本工程也没有任何运行期重跑 CSG 的地方：`build_demo_scene` 只在 Startup 跑一次，
/// 「游戏/世界/重载世界」走的是 `vox_scene::load_vox_scene` 的另一条路径。）
fn build_displace_sample(grid: &mut VolumeGrid) {
  // 中央广场地面上，两座台子沿 Z 并排、留 8 体素间隙；避开中央大道（z∈[480,544]）与标语（z∈[256,304]）。
  let extent = IVec3::new(64, 48, 32);
  let plain = IVec3::new(480, 16, 380);
  let bumped = IVec3::new(480, 16, 420);

  // 基准台（位移关闭）：既有 CSG 快路径 —— 作为"位移前"的对照（同尺寸同材质，差别只有位移）
  let base_voxels = fill_box(grid, plain, extent, 4);

  // 位移源：**由材质资产驱动**（MT8-5）—— 按 id 读幅度，0 = 不位移；高度图同步解码
  // **只这一个**材质（MT6-2 的"按需"；AssetServer 是异步的，而场景构造在 Startup）。
  let sample = match MaterialDisplace::load(
    DEMO_DISPLACE_HEIGHT_MAP,
    DEMO_DISPLACE_TEX_SCALE,
    DEMO_DISPLACE_AMPLITUDE_OVERRIDE,
  ) {
    Ok(Some(s)) => s,
    // 幅度 = 0（资产值）或 id 不在材质目录集里 ⇒ 本材质这次不位移：右台退化成普通 CSG，
    // 其余场景一字不动（"位移只发生在有幅度的材质上"正是 MT8-5 的语义）。
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
      // 缺素材不该让引擎起不来：右台退化成普通 CSG，日志说清
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
  // MT6-1 的语义表在 `height_field::HeightField::displace_fn`：切空间 UV / Repeat 平铺
  // 双线性采样 / 偏置 0.5（双向，外推 + 内缩）。幅度来自资产（见上面 `MaterialDisplace::load` 的日志）。
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

/// 打印**建议机位**：把 `data/config.toml` 的 `[camera]` 节换成日志里这一组即可正对位移样例。
/// 为什么需要它：样例躺在**地面**上（y=16 起），而默认机位是低俯角平视（eye y≈310、pitch≈0.29），
/// 不一定正对它们 —— 给一组俯视机位比"自己找"省事。**不再是"必须换机位"**（浮空岛删除后
/// 广场上是空的，任何从上方看过去的机位都能看到，不存在被实体包住的问题）。
fn log_sample_camera(plain: IVec3, extent: IVec3) {
  let target = (plain + extent / 2).as_vec3();
  // 从样例的东南上方俯视（广场是平的草地，视线无遮挡）
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
