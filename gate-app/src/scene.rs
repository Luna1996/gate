//! 场景搭建：Startup 系统 setup（相机/资源/诊断）+ 运行期换世界 `reload_world` + demo 极限场景生成。
//! 启动场景与规模见 `consts`（`STARTUP_DEMO_SCENE` / `DEMO_TILES`）。

use bevy::{image::Image, prelude::*};
use glam::{IVec3, Vec3};

use gate_render::{
  BrickMapBuilder, DdaCameraConfig, DdaImages, DebugNormals, OrbitCamera, UploadBudget, VIEW_SIZE,
  VoxelScene, create_dda_image,
};
use gate_voxel::{
  PaletteEntry, PaletteId, VolumeGrid, Volumes, draw_text, fill_box, fill_bricks, fill_sphere,
};

use crate::{
  consts::{
    CAM_FAR, CAM_NEAR, EXT_VOXEL_HALF, EXT_VOXEL_X, EXT_VOXEL_Z, FOV_Y, START_CAMERA_SKY,
    STARTUP_DEMO_SCENE,
  },
  vox_scene,
};

pub(crate) fn setup(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
  let dda_handle = create_dda_image(&mut images);
  commands.spawn((Camera2d, Msaa::Off));
  // 光照主题 RON：一次性静态配置，同步读即可；缺失/解析失败回退内置默认主题
  let theme = std::fs::read_to_string(gate_render::assets_dir().join("lighting/day_outdoor.ron"))
    .map_err(|e| format!("read: {e}"))
    .and_then(|s| gate_render::parse_lighting_ron(&s).map_err(|e| format!("ron: {e}")))
    .unwrap_or_else(|e| {
      bevy::log::warn!("lighting/day_outdoor.ron 加载失败（{e}），回退内置默认主题");
      Default::default()
    });
  commands.insert_resource(theme);
  commands.insert_resource(DdaImages { target: dda_handle });
  // 3 = unlit（跳过全部光照，albedo 直出）
  commands.insert_resource(DebugNormals(3));

  let t0 = std::time::Instant::now();
  let mut grid = VolumeGrid::new();
  // DDGI 四级网格的锚点：世界 AABB（与相机无关；demo 场景无 AABB 时用默认值）。
  let mut ddgi_world_aabb = gate_render::ddgi::DdgiWorldAabb::default();
  let mut cam_eye = Vec3::new(1., 0., 0.);
  let mut cam_target = Vec3::new(0., 0., 0.);
  if STARTUP_DEMO_SCENE {
    paint_demo_palette(&mut grid);
    bevy::log::info!("STEP 1: palette done ({:?})", t0.elapsed());
    build_demo_scene(&mut grid);
    bevy::log::info!("STEP 2: build_demo_scene done ({:?})", t0.elapsed());
  } else {
    let anchor = IVec3::new(EXT_VOXEL_HALF, 16, EXT_VOXEL_HALF);
    let path = gate_render::assets_dir().join("vox/nuke.vox");
    let info = vox_scene::load_vox_scene(&mut grid, &path, anchor).expect("nuke.vox 加载失败");
    cam_eye = Vec3::new(406.5, 339.5, 431.5);
    cam_target = Vec3::new(551.5, 330.5, 359.5);
    bevy::log::info!(
      "VOX SCENE: instances={} written={} dropped={} aabb=[{}]-[{}]",
      info.instances_used,
      info.voxels_written,
      info.voxels_dropped,
      info.aabb_min,
      info.aabb_max,
    );
    bevy::log::info!("STEP 2: vox scene done ({:?})", t0.elapsed());
    ddgi_world_aabb = gate_render::ddgi::DdgiWorldAabb { min: info.aabb_min, max: info.aabb_max };
  }
  grid.compact_all(); // GC：回收编辑过程累积的废弃节点
  bevy::log::info!("STEP 3: compact_all done ({:?})", t0.elapsed());

  // DDGI LOD0 的 chunk 探针段分配集（内容驱动，见 `lod0_needed_chunks`）：只有有几何或几何贴着
  // chunk 边界（16 体素内）的 chunk 才领固定 4096 槽的段，空 chunk 不占槽位。
  let ddgi_lod0_chunks = gate_render::ddgi::DdgiLod0Chunks { chunks: lod0_needed_chunks(&grid) };

  // 轨道相机为唯一相机状态源，DdaCameraConfig 由 `from_orbit` 生成（初始机位 = 场景中心俯视）。
  let orbit = if START_CAMERA_SKY {
    OrbitCamera::from_eye(
      Vec3::new(cam_target.x, 320.0, cam_target.z),
      Vec3::new(cam_target.x, 5000.0, cam_target.z),
    )
  } else {
    OrbitCamera::from_eye(cam_eye, cam_target)
  };
  commands.insert_resource(orbit);
  commands.insert_resource(ddgi_world_aabb);
  commands.insert_resource(ddgi_lod0_chunks);
  commands.insert_resource(DdaCameraConfig::from_orbit(
    &orbit,
    FOV_Y,
    VIEW_SIZE.x as f32 / VIEW_SIZE.y as f32,
    CAM_NEAR,
    CAM_FAR,
  ));
  // 幽灵飞行相机（CameraMode 缺省 = Fly）：起点 = 轨道眼位，切模式时视野原地不动
  // （见 camera::sync_camera_mode_switch）。
  commands.insert_resource(crate::camera::CameraMode::default());
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
      "TREE SIZE: total={}MB chunks={} top3={:?}",
      total_words * 4 / 1024 / 1024,
      words_per_chunk.len(),
      &words_per_chunk[..3.min(words_per_chunk.len())],
    );
    let bufs = BrickMapBuilder::build_full(&grid).buffers().clone();
    bevy::log::info!("STEP 4: diag build_full done ({:?})", t1.elapsed());
    let g = &bufs.globals;
    let n_chunks = grid.chunk_coords().count();
    bevy::log::info!(
      "BRICKMAP DIAG: chunks={} origin=({},{},{}) dims=({},{},{})  AABB=[{},{},{}]-[{},{},{}]",
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

/// DDGI LOD0 需要探针段的 chunk 集合（chunk 坐标 = 世界 voxel / 256，见 `DdgiLod0Chunks`）。
/// 规则：① 自己有几何的 chunk 领一段；② 最外一层 16³ brick 非空时相邻 chunk 也领。
fn lod0_needed_chunks(grid: &VolumeGrid) -> Vec<IVec3> {
  use gate_voxel::BrickState;
  use std::collections::HashSet;
  // 每轴 16 个 16³ brick（256 / 16）；边界层 = 坐标 0 或 15
  const BRICKS: i32 = 16;
  let axis_offsets = |b: i32| -> [i32; 2] {
    if b == 0 {
      [-1, 0]
    } else if b == BRICKS - 1 {
      [1, 0]
    } else {
      [0, 0]
    }
  };
  let mut needed: HashSet<IVec3> = HashSet::new();
  for c in grid.chunk_coords() {
    let Some(tree) = grid.chunk(c) else { continue };
    if tree.is_empty() {
      continue;
    }
    needed.insert(c.0);
    for bx in 0..BRICKS {
      for by in 0..BRICKS {
        for bz in 0..BRICKS {
          if bx != 0
            && bx != BRICKS - 1
            && by != 0
            && by != BRICKS - 1
            && bz != 0
            && bz != BRICKS - 1
          {
            continue;
          }
          if tree.get_brick_state(bx * 16, by * 16, bz * 16, 2) == BrickState::Air {
            continue;
          }
          for dx in axis_offsets(bx) {
            for dy in axis_offsets(by) {
              for dz in axis_offsets(bz) {
                if dx == 0 && dy == 0 && dz == 0 {
                  continue;
                }
                needed.insert(c.0 + IVec3::new(dx, dy, dz));
              }
            }
          }
        }
      }
    }
  }
  let mut out: Vec<IVec3> = needed.into_iter().collect();
  out.sort_unstable_by_key(|c| (c.x, c.y, c.z));
  out
}

/// 运行期换世界（DebugMenu「游戏/世界/重载世界」）：按 `assets/vox/<name>.vox` 重建主世界。
/// 与 `setup` 同一套不变量：先 `compact_all` 再算 LOD0 chunk 集；`demo_force_full_rebuild` 触发全量
/// 重建 + 全量 GPU 上传。失败 → 原世界保持不变；相机不动（所有模型锚到同一 anchor）。
pub(crate) fn reload_world(
  scene: &mut VoxelScene,
  aabb: &mut gate_render::ddgi::DdgiWorldAabb,
  lod0: &mut gate_render::ddgi::DdgiLod0Chunks,
  name: &str,
) -> Result<vox_scene::VoxSceneInfo, Box<dyn std::error::Error>> {
  let anchor = IVec3::new(EXT_VOXEL_HALF, 16, EXT_VOXEL_HALF);
  let path = gate_render::assets_dir().join("vox").join(format!("{name}.vox"));
  let mut grid = VolumeGrid::new();
  let info = vox_scene::load_vox_scene(&mut grid, &path, anchor)?;
  grid.compact_all();
  let chunks = lod0_needed_chunks(&grid);
  scene.volumes = Volumes::new(grid);
  scene.demo_force_full_rebuild = true;
  aabb.min = info.aabb_min;
  aabb.max = info.aabb_max;
  lod0.chunks = chunks;
  Ok(info)
}

/// demo 调色板：1..=13 地形/建筑色 + 14 号 LED 灯柱（`PaletteEntry::default` + 逐字段赋值）。
fn paint_demo_palette(grid: &mut VolumeGrid) {
  let palette: &[(u16, [u8; 3], u8)] = &[
    (1, [86, 160, 70], 220),   // 1 草地（L0 地面平原）
    (2, [140, 108, 76], 200),  // 2 山岩（山体主体）
    (3, [240, 244, 248], 160), // 3 雪峰（y > 山线顶）
    (4, [172, 176, 190], 210), // 4 城堡石（墙/塔）
    (5, [52, 168, 72], 210),   // 5 树叶（球冠）
    (6, [112, 72, 40], 200),   // 6 树干（细柱）
    (7, [64, 140, 232], 170),  // 7 河流蓝（L2 缠绕）
    (8, [68, 230, 220], 150),  // 8 水晶青（L4 高亮）
    (9, [200, 120, 240], 150), // 9 水晶紫（L4 高亮）
    (10, [238, 76, 90], 160),  // 10 水晶红 / 旗帜（L4 高亮）
    (11, [248, 210, 72], 160), // 11 塔顶金
    (12, [30, 32, 44], 220),   // 12 道路/桥面（深灰）
    (13, [92, 118, 240], 180), // 13 浮空岛岛底
  ];
  let pal = grid.palette_mut();
  for &(idx, color, rough) in palette {
    let mut e = PaletteEntry::default();
    e.color = color;
    e.roughness = rough;
    pal.set(PaletteId(idx), e);
  }
  // 14 号 LED 灯柱（暖白满档发光）
  let mut led = PaletteEntry::default();
  led.color = [255, 214, 156];
  led.roughness = 128;
  led.emissive = 255;
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

/// 离中央堡（世界中心）的距离
fn dist_to_castle(x: i32, z: i32) -> i32 {
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
      bevy::log::info!("SCENE {}: {:?} chunks={}", $name, t0.elapsed(), grid.chunk_count())
    };
  }
  // (1) 基础地形：按 16 voxel (L0) 步长采样高度场 → fill_box 铺柱（y=16..h 用 pal 2，h>560 顶部改 pal 3）
  let mut z = 0i32;
  while z < EXT_VOXEL_Z {
    let mut x = 0i32;
    while x < EXT_VOXEL_X {
      let h = terrain_h(x, z);
      let dc = dist_to_castle(x, z);
      // 中央堡区 (radius<700) 不开地形，后面由城堡结构接管
      let in_castle_plate = dc < 700;
      // 河道：y=16..24 填 palette 7，不叠加山岩
      let in_r = in_river(x, z);
      if !in_castle_plate {
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
        // 城堡基座：y=0..16 草地平铺（稍后全局 L0 统一填），山体不叠加
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

  // (3) 中央天空之城（以世界中心为锚）：浮空岛底 → 城墙/角楼 → 正殿 → 高塔 → 旗帜
  let cx = EXT_VOXEL_HALF;
  let cz = EXT_VOXEL_HALF;
  // 浮空岛底（倒锥：按 y 降低半径收窄）——L0 级 16 步扫描
  let island_base_y = 128i32; // 岛底尖
  let island_top_y = 240i32; // 岛顶面（城墙在此升起）
  let top_r = 560i32; // 顶面岛半径
  let mut y = island_base_y;
  while y < island_top_y {
    let t = (y - island_base_y) as f32 / (island_top_y - island_base_y) as f32;
    let r = (t * t.sqrt() * top_r as f32) as i32 + 16;
    fill_sphere(grid, IVec3::new(cx, y, cz), r, 13);
    y += 16;
  }
  mark!("(3a) floating island");
  // 城墙平台（y=240..272）
  fill_bricks(grid, IVec3::new(cx - 512, 240, cz - 512), IVec3::new(1024, 32, 1024), 16, 4);
  // 四面城墙（y=272..336 = 64 高）
  // 北 Z=cz-512, 南 Z=cz+512-8
  fill_box(grid, IVec3::new(cx - 512, 272, cz - 512), IVec3::new(1024, 64, 8), 4);
  fill_box(grid, IVec3::new(cx - 512, 272, cz + 512 - 8), IVec3::new(1024, 64, 8), 4);
  fill_box(grid, IVec3::new(cx - 512, 272, cz - 512), IVec3::new(8, 64, 1024), 4);
  fill_box(grid, IVec3::new(cx + 512 - 8, 272, cz - 512), IVec3::new(8, 64, 1024), 4);
  // 四角角楼 96×96×160（从 y=272 起比城墙多高 96）
  for &(ox, oz) in &[(-512, -512), (512 - 96, -512), (-512, 512 - 96), (512 - 96, 512 - 96)] {
    let tx = cx + ox;
    let tz = cz + oz;
    fill_bricks(grid, IVec3::new(tx, 272, tz), IVec3::new(96, 160, 96), 16, 4);
    // 角楼顶金色 16
    fill_box(grid, IVec3::new(tx, 272 + 160, tz), IVec3::new(96, 16, 96), 11);
  }
  // 正殿（中心，y=336..464 = 128 高）
  fill_bricks(grid, IVec3::new(cx - 256, 336, cz - 256), IVec3::new(512, 128, 512), 16, 4);
  // 正殿正门（Z- 方向，挖一矩形门洞：clear_voxel）
  {
    // L1 每步 = 8 voxel；宽 96 → 12 步 × 高 96 → 12 步 × 深 8 → 1 步
    let e = 8i32; // L1 边长
    let mn = IVec3::new(cx - 48, 336, cz - 264);
    let ex = mn + IVec3::new(96, 96, 8);
    let mut z = mn.z.div_euclid(e) * e;
    while z < ex.z {
      let mut y = mn.y.div_euclid(e) * e;
      while y < ex.y {
        let mut x = mn.x.div_euclid(e) * e;
        while x < ex.x {
          grid.clear_voxel(gate_voxel::VoxelCoord::from_ivec3(IVec3::new(x, y, z)));
          x += e;
        }
        y += e;
      }
      z += e;
    }
  }
  // 高塔（y=464..720 = 256 高，底 96×96 上收顶）
  fill_bricks(grid, IVec3::new(cx - 48, 464, cz - 48), IVec3::new(96, 256, 96), 16, 4);
  // 塔顶平台 128×128×16
  fill_box(grid, IVec3::new(cx - 64, 720, cz - 64), IVec3::new(128, 16, 128), 11);
  // 金顶球（L2 r=64，塔顶 y=720+80=800）
  fill_sphere(grid, IVec3::new(cx, 800, cz), 64, 11);
  mark!("(3b) castle walls+towers");
  // 四角旗帜（L4 红飘带：从角楼顶 4 角斜向上拉出小立方体串）
  for &(ox, oz) in &[(-512, -512), (512 - 16, -512), (-512, 512 - 16), (512 - 16, 512 - 16)] {
    let fx = cx + ox + 4;
    let fz = cz + oz + 4;
    for s in 0..16i32 {
      fill_box(grid, IVec3::new(fx + s, 448 + s * 4, fz), IVec3::new(8, 8, 8), 10);
    }
  }

  // (4) 森林：~160 棵确定性散点树（双素数线性同余，不用 rand）= 树干（L1）+ 叶球（L2，r=80）；
  // 选址：城堡半径外、非河道、非大道、h<360。
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
      let ok = dist_to_castle(x, z) > 900
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
}
