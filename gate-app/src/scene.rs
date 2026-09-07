//! 场景搭建：Startup 系统setup（相机/资源/诊断）+ demo 极限场景生成。
//!
//! `GATE_SCENE=vox`（默认）→ MagicaVoxel nuke.vox（vox_scene 模块）；
//! `GATE_SCENE=demo` → 程序化极限场景（城堡/大道/森林/水晶矿，见 [`build_demo_scene`]）。

use std::path::Path;
use std::sync::LazyLock;

use bevy::{image::Image, prelude::*};
use glam::{IVec3, Vec3};

use gate_render::{
  BrickMapBuilder, DdaCameraConfig, DdaImages, DebugNormals, OrbitCamera, UploadBudget, VIEW_SIZE,
  VoxelScene, create_dda_image,
};
use gate_voxel::{
  PaletteEntry, Volumes, VolumeGrid, draw_text, fill_box, fill_bricks, fill_sphere,
};

use crate::{ASSETS_PATH, camera::{CAM_FAR, CAM_NEAR, FOV_Y}, vox_scene};

pub(crate) fn setup(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
  let dda_handle = create_dda_image(&mut images);
  commands.spawn((Camera2d, Msaa::Off));
  // ---- P3.1 光照主题：「暗色实验室」RON 加载（一次性静态配置，同步读足够；
  // 缺失/解析失败回退内置默认主题）----
  let theme = std::fs::read_to_string(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/assets/lighting/day_outdoor.ron"
  ))
  .map_err(|e| format!("read: {e}"))
  .and_then(|s| gate_render::parse_lighting_ron(&s).map_err(|e| format!("ron: {e}")))
  .unwrap_or_else(|e| {
    bevy::log::warn!("lighting/day_outdoor.ron 加载失败（{e}），回退内置默认主题");
    Default::default()
  });
  commands.insert_resource(theme);
  commands.insert_resource(DdaImages { target: dda_handle });
  // 3 = unlit（跳过全部光照，albedo 直出）；N 键循环切换已移除，固定为此模式
  commands.insert_resource(DebugNormals(3));

  // ---- 场景：GATE_SCENE=vox（默认）→ MagicaVoxel nuke.vox；=demo → 旧极限场景 ----
  let t0 = std::time::Instant::now();
  let mut grid = VolumeGrid::new();
  let mut cam_eye = Vec3::new(1., 0., 0.);
  let mut cam_target = Vec3::new(0., 0., 0.);
  match std::env::var("GATE_SCENE").as_deref() {
    Ok("demo") => {
      paint_demo_palette(&mut grid);
      bevy::log::info!("STEP 1: palette done ({:?})", t0.elapsed());
      build_demo_scene(&mut grid);
      bevy::log::info!("STEP 2: build_demo_scene done ({:?})", t0.elapsed());
    }
    _ => {
      let anchor = IVec3::new(*EXT_FINE_HALF, 16, *EXT_FINE_HALF);
      let path = Path::new(ASSETS_PATH).join("vox/nuke.vox");
      let info = vox_scene::load_vox_scene(&mut grid, &path, anchor)
        .expect("nuke.vox 加载失败");
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
    }
  }
  grid.compact_all(); // GC：回收编辑过程累积的废弃节点
  bevy::log::info!("STEP 3: compact_all done ({:?})", t0.elapsed());

  // P2.6：轨道相机为唯一相机状态源；DdaCameraConfig 由 from_orbit 生成
  // （初始机位：场景中心俯视；诊断：GATE_CAM=sky → 仰视天空，纯 miss 验证 GPU 负载）
  let orbit = if std::env::var("GATE_CAM").as_deref() == Ok("sky") {
    OrbitCamera::from_eye(
      Vec3::new(cam_target.x, 320.0, cam_target.z),
      Vec3::new(cam_target.x, 5000.0, cam_target.z),
    )
  } else {
    OrbitCamera::from_eye(cam_eye, cam_target)
  };
  commands.insert_resource(orbit);
  commands.insert_resource(DdaCameraConfig::from_orbit(
    &orbit,
    FOV_Y,
    VIEW_SIZE.x as f32 / VIEW_SIZE.y as f32,
    CAM_NEAR,
    CAM_FAR,
  ));

  // 诊断：打印 brickmap globals
  {
    let t1 = std::time::Instant::now();
    // 树输出量诊断：serialize 总字数 + 最大 chunk（build_full OOM 排查）
    let mut words_per_chunk: Vec<(usize, _)> = grid
      .chunk_coords()
      .map(|c| (grid.chunk(c).map(|t| t.len_words()).unwrap_or(0), c))
      .collect();
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

  // ---- Phase 3 OBJ→Volume 统一：物体作为 Volumes.list[1..N] 加入容器 ----
  // 旧 ObjScene/Pack_obj_pool/ObjObject 三段管道已删除；物体 = 普通的 VolumeGrid，
  // 通过 Volumes.add_object() 注册变换，走与主世界相同的 dirty → builder → upload 路径。
  let volumes = Volumes::new(grid);

  commands.insert_resource(VoxelScene {
    volumes,
    demo_force_full_rebuild: true,
  });
  commands.insert_resource(UploadBudget {
    max_bytes_per_frame: 4 * 1024 * 1024,
    incremental: true,
  });
}

/// demo 调色板（PaletteEntry._pad 私有 → 跨 crate 用 default + 逐字段赋值）
/// 极限场景用 14 色：草地 / 山岩 / 雪峰 / 城堡石 / 树叶 / 树干 / 河蓝 / 水晶青 / 水晶紫 / 水晶红 / 塔顶金 / 旗帜红
fn paint_demo_palette(grid: &mut VolumeGrid) {
  let palette: &[(u8, [u8; 3], u8)] = &[
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
    pal.set(idx, e);
  }
  // P3.2：14 号 LED 灯柱（暖白满档发光）
  let mut led = PaletteEntry::default();
  led.color = [255, 214, 156];
  led.roughness = 128;
  led.emissive = 255;
  pal.set(14, led);
}

// 极限场景版图：
// ─────────────────────────────────────────────────────────────────────
// 世界：tiles x∈[0..10] z∈[0..10] y∈[0..2] 共 10×3×10=300 实有 tile，
//       brickmap compute_window origin=(-1,-1,-1) dims=(12,5,12)=720 砖，
//       voxel AABB ≈ (-512,-512,-512)..(5632,2048,5632)，对角穿越 ≈9000 voxel。
// 内容分布（voxel 单位 0..5120 XY 平面，y 高度）：
//   · 基础 L0 地面（全地图高 16）+ 起伏正弦高度场山体
//   · 4 座雪峰（四角，y 到 800）· 中部峡谷蜿蜒河流 L2
//   · 160 棵散点树（L2 粗节节省体素）在非山非河格
//   · 正中央 "天空堡"：L1 岛底倒锥 + L1 城堡外城墙 + 4 座角楼 + L2 高塔 + L3 金顶
//   · 3 处水晶矿簇（L4 多色，给 DDA 穿越厚度内密集命中以展示引擎吞吐）
//   · 保留 tile(1,0,0) 每 120 帧 L4 黄↔青交替（验证 132KB/180µs 增量上传路径）
//   · 世界大标语 "GATE ENGINE" 立在入口大道
// ─────────────────────────────────────────────────────────────────────
// 世界规模（tile 数，1 tile = 512 voxel）：env `GATE_TILES` 可调。
// 默认 2 = 快速调试档（启动 ~几秒；正确性调试期默认小场景）；
// `GATE_TILES=10` = 完整压测场景（启动 ~55s，性能验收用）。
// 场景内所有结构性坐标（城堡/大道/河）均以 EXT_FINE_HALF 为锚，随规模等比成立。
static EXT_N_TILES: LazyLock<i32> = LazyLock::new(|| {
  std::env::var("GATE_TILES")
    .ok()
    .and_then(|s| s.parse::<i32>().ok())
    .unwrap_or(2)
    .clamp(2, 10)
});
static EXT_FINE_X: LazyLock<i32> = LazyLock::new(|| *EXT_N_TILES * 512);
static EXT_FINE_Z: LazyLock<i32> = LazyLock::new(|| *EXT_N_TILES * 512);
static EXT_FINE_HALF: LazyLock<i32> = LazyLock::new(|| *EXT_FINE_X / 2);

/// 正弦高度场（确定性，不用 rand）：
/// h(x,z) = 16 + A·sin(x·k1)·cos(z·k2) + 山体距 4 角的反比隆起
fn terrain_h(x: i32, z: i32) -> i32 {
  let xf = x as f32;
  let zf = z as f32;
  let s1 = (xf * 0.0045).sin() * (zf * 0.0053).cos(); // 基础波
  let s2 = ((xf + zf) * 0.0020).sin() * 1.3; // 对角长波
  // 四角雪峰：与 (0,0)/(5120,0)/(0,5120)/(5120,5120) 的距离反比
  let corner_snow = |cx, cz| {
    let dx = xf - cx as f32;
    let dz = zf - cz as f32;
    let d = (dx * dx + dz * dz).sqrt().max(200.0);
    (600.0 / d).min(1.0) * 780.0 // 山顶 ~800
  };
  let sn = corner_snow(0, 0)
    + corner_snow(*EXT_FINE_X - 1, 0)
    + corner_snow(0, *EXT_FINE_Z - 1)
    + corner_snow(*EXT_FINE_X - 1, *EXT_FINE_Z - 1);
  let hf = 16.0 + s1 * 80.0 + s2 * 120.0 + sn;
  // 4 对齐：fill_box 高度层全部 4³ 整块（非对齐会退化为逐体素边缘 →
  // 4³ Split 碎片化，树序列化输出曾膨胀到 1.9GB）。阶梯化 4 级符合像素风。
  ((hf as i32).clamp(16, 1020)) & !3
}

/// 离中央堡（世界中心）的距离
fn dist_to_castle(x: i32, z: i32) -> i32 {
  let dx = x - *EXT_FINE_HALF;
  let dz = z - *EXT_FINE_HALF;
  ((dx * dx + dz * dz) as f32).sqrt() as i32
}

/// 中央峡谷蜿蜒河（x,z 在河道走廊内返回 true）
fn in_river(x: i32, z: i32) -> bool {
  let t = z as f32 / *EXT_FINE_Z as f32; // 0..1
  let river_center = *EXT_FINE_HALF as f32 // 世界中线
    + (t * std::f32::consts::TAU).sin() * 700.0 // 正弦摆 ±700
    + ((t * 12.566).cos() * 180.0); // 次级摆幅
  let dx = (x as f32) - river_center;
  dx.abs() < 64.0
}

fn build_demo_scene(grid: &mut VolumeGrid) {
  let t0 = std::time::Instant::now();
  macro_rules! mark {
    ($name:expr) => {
      bevy::log::info!(
        "SCENE {}: {:?} chunks={}",
        $name,
        t0.elapsed(),
        grid.chunk_count()
      )
    };
  }
  // ================================================================
  //  (1) 基础地形：按 16 voxel (L0) 步长采样高度场 → fill_box 铺柱
  //      y=16..h 使用 palette 2 山岩；y>h-40 且 h>520 → palette 3 雪峰
  // ================================================================
  let mut z = 0i32;
  while z < *EXT_FINE_Z {
    let mut x = 0i32;
    while x < *EXT_FINE_X {
      let h = terrain_h(x, z);
      let dc = dist_to_castle(x, z);
      // 中央堡区 (radius<700) 不开地形，后面由城堡结构接管
      let in_castle_plate = dc < 700;
      // 河道：y=16..24 填 palette 7，不叠加山岩
      let in_r = in_river(x, z);
      if !in_castle_plate {
        // 先铺 L0 地面（所有格子 y=0..16 统一由稍后 L0 全地图填，这里只填 16..h 山体）
        let top = h;
        let snow_line = top - 40; // 4 对齐（top 已对齐 4）
        if in_r {
          // 河床：4 对齐（岩 8 + 蓝 8 = y 16..32，原 12+8 的 12 非对齐会碎片化）
          fill_box(grid, IVec3::new(x, 16, z), IVec3::new(16, 8, 16), 2);
          fill_box(grid, IVec3::new(x, 24, z), IVec3::new(16, 8, 16), 7);
        } else if top > 16 {
          // 山体主体（palette 2 山岩）
          let rock_h = (snow_line - 16).max(0).min(top - 16);
          if rock_h > 0 {
            fill_box(grid, IVec3::new(x, 16, z), IVec3::new(16, rock_h, 16), 2);
          }
          // 雪顶（h > 560 时，顶部 38 voxel 改 palette 3 雪）
          if top > 560 {
            let snow_h = top - snow_line;
            if snow_h > 0 {
              fill_box(
                grid,
                IVec3::new(x, snow_line, z),
                IVec3::new(16, snow_h, 16),
                3,
              );
            }
          } else {
            // 低矮山坡顶部覆草（palette 1，最顶 16 voxel）
            let grass_h = (top - 16).min(16);
            if grass_h > 0 && top - grass_h >= 16 {
              fill_box(
                grid,
                IVec3::new(x, top - grass_h, z),
                IVec3::new(16, grass_h, 16),
                1,
              );
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
  fill_bricks(
    grid,
    IVec3::new(0, 0, 0),
    IVec3::new(*EXT_FINE_X, 16, *EXT_FINE_Z),
    16,
    1,
  );
  mark!("(1b) floor");

  // ================================================================
  //  (2) 中央大道 + 入口大标语 "GATE ENGINE"
  // ================================================================
  // 中央大道（X 向，中线 ±32 宽）深灰铺路
  fill_box(
    grid,
    IVec3::new(0, 16, *EXT_FINE_HALF - 32),
    IVec3::new(*EXT_FINE_X, 8, 32),
    12,
  );
  // 大标语（L1，每像素 8³，放在入口 X=64 Y=32 Z=256 朝向 -Z）
  draw_text(grid, IVec3::new(96, 32, 256), "GATE ENGINE", 8);
  mark!("(2) road+text");

  // ================================================================
  //  (3) 中央天空之城（世界中心 ±520 方区）
  //     · 浮空岛底（倒锥状，L1 pal 13 深蓝灰）底 y=600 顶 y=240
  //     · 城墙平台 400×400×32（y=240..272 pal 4 石）
  //     · 四面城墙 8 宽 × 64 高 × 400 长（y=272..336）
  //     · 四角角楼 64×64×128
  //     · 正殿 192×128×192（y=336..464）
  //     · 正殿中心高塔 80×80×192（y=464..656）+ L3 金顶球
  //     · 四角旗帜（L4 红飘带形状）
  // ================================================================
  let cx = *EXT_FINE_HALF;
  let cz = *EXT_FINE_HALF;
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
  // 城墙平台（y=240..272，512×512×32）
  fill_bricks(
    grid,
    IVec3::new(cx - 512, 240, cz - 512),
    IVec3::new(1024, 32, 1024),
    16,
    4,
  );
  // 四面城墙（y=272..336 = 64 高）
  // 北 Z=cz-512, 南 Z=cz+512-8
  fill_box(
    grid,
    IVec3::new(cx - 512, 272, cz - 512),
    IVec3::new(1024, 64, 8),
    4,
  );
  fill_box(
    grid,
    IVec3::new(cx - 512, 272, cz + 512 - 8),
    IVec3::new(1024, 64, 8),
    4,
  );
  fill_box(
    grid,
    IVec3::new(cx - 512, 272, cz - 512),
    IVec3::new(8, 64, 1024),
    4,
  );
  fill_box(
    grid,
    IVec3::new(cx + 512 - 8, 272, cz - 512),
    IVec3::new(8, 64, 1024),
    4,
  );
  // 四角角楼 96×96×160（从 y=272 起比城墙多高 96）
  for &(ox, oz) in &[
    (-512, -512),
    (512 - 96, -512),
    (-512, 512 - 96),
    (512 - 96, 512 - 96),
  ] {
    let tx = cx + ox;
    let tz = cz + oz;
    fill_bricks(
      grid,
      IVec3::new(tx, 272, tz),
      IVec3::new(96, 160, 96),
      16,
      4,
    );
    // 角楼顶金色 16
    fill_box(
      grid,
      IVec3::new(tx, 272 + 160, tz),
      IVec3::new(96, 16, 96),
      11,
    );
  }
  // 正殿（中心，y=336..464 = 128 高，长 256×256）
  fill_bricks(
    grid,
    IVec3::new(cx - 256, 336, cz - 256),
    IVec3::new(512, 128, 512),
    16,
    4,
  );
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
  fill_bricks(
    grid,
    IVec3::new(cx - 48, 464, cz - 48),
    IVec3::new(96, 256, 96),
    16,
    4,
  );
  // 塔顶平台 128×128×16
  fill_box(
    grid,
    IVec3::new(cx - 64, 720, cz - 64),
    IVec3::new(128, 16, 128),
    11,
  );
  // 金顶球（L2 r=64，塔顶 y=720+80=800）
  fill_sphere(grid, IVec3::new(cx, 800, cz), 64, 11);
  mark!("(3b) castle walls+towers");
  // 四角旗帜（L4 红飘带：从角楼顶 4 角斜向上拉出小立方体串）
  for &(ox, oz) in &[
    (-512, -512),
    (512 - 16, -512),
    (-512, 512 - 16),
    (512 - 16, 512 - 16),
  ] {
    let fx = cx + ox + 4;
    let fz = cz + oz + 4;
    for s in 0..16i32 {
      fill_box(
        grid,
        IVec3::new(fx + s, 448 + s * 4, fz),
        IVec3::new(8, 8, 8),
        10,
      );
    }
  }

  // ================================================================
  //  (4) 森林：~160 棵确定性散点树（不用 rand，用 x 坐标做双素数步进）
  //     每棵：树干 L1 40×16×40 + 叶 L2 球 r=80
  //     不在城堡 800 半径内、不在河道上、不在大道上、高度 h<400（只种平原）
  // ================================================================
  {
    let mut n_planted = 0usize;
    let mut i = 0i32;
    while n_planted < 170 && i < 2000 {
      // 确定性 2 互素线性同余 → (x,z) 伪散点
      let x = (i * 211 + 83) % *EXT_FINE_X;
      let z = ((i * 977 + 419) ^ 0xA53) % *EXT_FINE_Z;
      let x = x.abs();
      let z = z.abs();
      let h = terrain_h(x, z);
      let ok = dist_to_castle(x, z) > 900
        && !in_river(x, z)
        && !((*EXT_FINE_HALF - 32)..=(*EXT_FINE_HALF + 32)).contains(&z) // 中央大道
        && h < 360;
      if ok {
        // 树干 （L1, 24 宽 16 宽 24 深 高 80）
        fill_box(grid, IVec3::new(x, h, z), IVec3::new(32, 80, 32), 6);
        // 树叶球（L2，r=80，中心在树干顶 + 80）
        fill_sphere(grid, IVec3::new(x + 16, h + 80 + 64, z + 16), 80, 5);
        n_planted += 1;
      }
      i += 1;
    }
  }
  mark!("(4) forest");

  // ================================================================
  //  (5) 3 处 L4 水晶矿簇（每处 60~100 个 1³ 彩色小立方体，密集堆叠，
  //      用来展示"高分辨率体素不拖垮 fps"——
  //      因为 DDA 命中就 break，不随地图体素数上升而变慢）
  // ================================================================
  let crystal_clusters: &[(i32, i32, i32, u8, u8, i32)] = &[
    // (cx, cz, count, pal_a, pal_b, seed)
    (480, 480, 90, 8, 9, 131),   // 左上（near 角雪峰脚）水晶青/紫
    (4608, 640, 80, 10, 8, 251), // 右上 水晶红/青（seed<255 安全）
    (768, 4352, 70, 9, 10, 223), // 左下 水晶紫/红
  ];
  for &(cx_, cz_, cnt, pa, pb, seed) in crystal_clusters.iter() {
    if cx_ >= *EXT_FINE_X || cz_ >= *EXT_FINE_Z {
      continue; // 簇中心在世界外（N 缩小时）跳过，避免 t 无限增长溢出
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
      if px_ >= 0 && pz_ >= 0 && px_ < *EXT_FINE_X && pz_ < *EXT_FINE_Z {
        fill_box(grid, IVec3::new(px_, h + hy, pz_), IVec3::new(1, 1, 1), pal);
        placed += 1;
      }
      t += 1;
    }
  }
  mark!("(5) crystals");

  // ================================================================
  //  (6) 静态 L4 32³ 精度热点：tile(1,0,0) 金色 32³ + tile(2,0,0) 蓝 32³
  //      （原「每 120 帧黄↔青交替」闪烁测试已移除，仅保留静态场景）
  // ================================================================
  fill_box(grid, IVec3::new(656, 64, 64), IVec3::new(32, 32, 32), 11);
  if 1264 + 32 <= *EXT_FINE_X {
    fill_box(grid, IVec3::new(1264, 240, 240), IVec3::new(32, 32, 32), 7);
  }
  let t0 = gate_voxel::ChunkCoord::new(0, 0, 0);
  grid.set_state(0, 1, 0xDEAD);
  grid.set_state(1, 0, 0xBEEF);
  grid.set_comp(t0, 0, 0, 0, 0x1122);
  grid.set_comp(t0, 1, 0, 0, 0x3344);
  mark!("(6) hotspots+state");
}
