use bevy::{
  asset::{AssetPlugin, LoadState},
  diagnostic::{DiagnosticPath, DiagnosticsStore, FrameTimeDiagnosticsPlugin},
  image::Image,
  input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll, MouseScrollUnit},
  log::LogPlugin,
  prelude::*,
  ui::{Checkable, Checked},
  window::{PresentMode, Window, WindowResized},
};
use glam::{Vec3, Vec4};

use gate_render::{
  DdaCameraConfig, DdaImages, GradientImages, GradientUniforms, OrbitCamera, UploadBudget,
  UploadCpuSampleChannel, VIEW_SIZE, VoxelScene, cpu_reference_dda_ray,
  cpu_reference_dda_ray_aabb_skip, create_dda_image, create_gradient_image,
};
use gate_ui::{
  ThemeFont, UiCtx, UiTheme,
  widgets::{
    PLOT_H, PLOT_W, PlotData, PlotDomain, RingList, SliderValue, UiClick, blank_plot_image, button,
    checkbox, color_of, label, label_muted, list, panel, plot, px, slider,
  },
};
use gate_voxel::{PaletteEntry, draw_text, fill_box, fill_sphere};

/// 以 crate 目录为锚的 assets 路径，F5 / 终端启动行为一致
pub const ASSETS_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/assets");

// ---- 相机参数（P2.6 from_orbit 使用；用户已取消"最远距离"限制）----
// CAM_FAR = 透视投影 far 面；dda.wgsl 内 DDA 射线 t_max 同步到此量级。
// 原 4000（10m）→ 现 65536（163.84m）足够 zoom-out 到整个 tile 场景（~1000 fine）
// 缩成屏幕 1% 像素仍可见。DIST_MAX 已删除，滚轮 zoom-out 距离本身无上限。
const FOV_Y: f32 = 60.0_f32.to_radians();
const CAM_NEAR: f32 = 1.0;
const CAM_FAR: f32 = 65536.0;
// ---- 输入灵敏度（spec FR-3；手感调整只改这里）----
const ROT_SPEED: f32 = 0.005; // rad/px（右键拖拽旋转）
const ZOOM_LOG_SPEED: f32 = 0.35; // /行（滚轮乘法缩放，各距离档手感一致）

fn main() {
  App::new()
    .add_plugins(
      DefaultPlugins
        .set(WindowPlugin {
          primary_window: Some(Window {
            resolution: VIEW_SIZE.into(),
            present_mode: PresentMode::AutoNoVsync,
            resizable: true, // 2.7a FR-5：解锁任意 resize（渲染目标 + aspect 由响应式系统跟随）
            ..default()
          }),
          ..default()
        })
        .set(AssetPlugin {
          file_path: ASSETS_PATH.into(),
          ..default()
        })
        .set(LogPlugin {
          // info 基线 + 两条定向屏蔽：
          // 1) wgpu_hal::vulkan::instance = off：屏蔽 wgpu 29.0.4 启动时的 VUID 错误——
          //    已知 bug（wgpu#9213/#9361，VUID-VkPresentInfoKHR-pImageIndices-01430 和
          //    VUID-vkAcquireNextImageKHR-semaphore-01286），仅首 1-2 帧的 swapchain
          //    时序异常，不影响渲染功能与画面正确性；wgpu 升级后自动恢复。
          // 2) wgpu_hal::vulkan::surface = off：同来源的 surface 层偶发错误。
          //    其余 wgpu/winit/bevy_render 的错误照常打印，避免掩盖真实问题。
          filter: "info,\
            wgpu_hal::vulkan::instance=off,\
            wgpu_hal::vulkan::surface=off"
            .into(),
          ..default()
        }),
    )
    .add_plugins((
      // FrameTimeDiagnostics：采集帧时数据写入 DiagnosticsStore（demo_fps_feed 读取）；
      // 日志侧的每秒打印由 LogDiagnosticsPlugin 负责——这里不装，因为游戏内性能面板
      // 已经有 FPS 折线图和实时 FPS 标签，每秒 info! 刷屏纯噪音。
      FrameTimeDiagnosticsPlugin::default(),
    ))
    // P2.7：渲染诊断（Bevy 0.19 非默认装配，仅 tracing-tracy feature 才自动加）——
    // 装配后 DiagnosticsRecorder 才存在，4×pass 的 time_span 才会记录 GPU/CPU 耗时；
    // 未装配时 gate-render 的 span 走 Option<&T> no-op，不影响渲染
    .add_plugins(bevy::render::diagnostic::RenderDiagnosticsPlugin)
    .add_plugins(gate_render::GateRenderPlugin)
    .add_plugins(gate_ui::GateUiPlugin)
    .add_systems(Startup, setup)
    .add_systems(
      Update,
      (
        orbit_camera_input,
        sync_anchor_camera.after(orbit_camera_input),
        edit_tile_every_120_frames,
        demo_ui_setup,
        demo_fps_feed,
        // debug_aabb_report,  // 诊断系统 224ms/2s build_full 阻塞 Update（极限场景 211 tiles 全 build），
                              // 导致 fps 基线从 73 拉到 20；已确认 Rust 侧 skip_hits=260/1024（体素构建正确）。
                              // 真需定位 WGSL 全黑时临时取消注释，跑完再注释掉。
      ),
    )
    .run();
}

fn setup(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
  let grad_handle = create_gradient_image(&mut images);
  let dda_handle = create_dda_image(&mut images);
  commands.spawn((Camera2d, Msaa::Off));
  commands.insert_resource(GradientImages {
    target: grad_handle,
  });
  commands.insert_resource(DdaImages { target: dda_handle });
  commands.insert_resource(GradientUniforms {
    size: VIEW_SIZE.as_vec2().extend(0.0).extend(0.0),
  });
  // P2.6：轨道相机为唯一相机状态源；DdaCameraConfig 由 from_orbit 生成
  // （极限场景：世界中心 2560,160,2560；eye 从 +X/+Z 45° 俯视距离 5200 fine 一览 10×10 大陆全境）
  let orbit = OrbitCamera::from_eye(
    Vec3::new(2560.0 + 3800.0, 2600.0, 2560.0 + 3800.0),
    Vec3::new(2560.0, 320.0, 2560.0),
  );
  commands.insert_resource(orbit);
  commands.insert_resource(DdaCameraConfig::from_orbit(
    &orbit,
    FOV_Y,
    VIEW_SIZE.x as f32 / VIEW_SIZE.y as f32,
    CAM_NEAR,
    CAM_FAR,
  ));

  // ---- demo scene：调色板 + 多分辨率混合极限场景 ----
  let mut grid = gate_voxel::TileGrid::new();
  paint_demo_palette(&mut grid);
  build_demo_scene(&mut grid);

  commands.insert_resource(VoxelScene {
    grid,
    demo_force_full_rebuild: true,
  });
  commands.insert_resource(UploadBudget {
    max_bytes_per_frame: 4 * 1024 * 1024,
    incremental: true,
  });

  // ---- 世界空间 UI 标注（极限场景 4 个兴趣点） ----
  // 天空堡金顶（正中央）
  gate_ui::world_anchor_label(
    &mut commands,
    "天空堡·金顶",
    Vec3::new(2560.0, 880.0, 2560.0),
    Color::srgb_u8(248, 210, 72),
  );
  // 入口大标语
  gate_ui::world_anchor_label(
    &mut commands,
    "入口大道 GATE ENGINE",
    Vec3::new(512.0, 96.0, 256.0),
    Color::srgb_u8(68, 230, 220),
  );
  // 西北角雪峰
  gate_ui::world_anchor_label(
    &mut commands,
    "西北雪峰",
    Vec3::new(480.0, 900.0, 480.0),
    Color::srgb_u8(240, 244, 248),
  );
  // 水晶矿区
  gate_ui::world_anchor_label(
    &mut commands,
    "青紫水晶矿区（L4 精细体素）",
    Vec3::new(480.0, 80.0, 480.0),
    Color::srgb_u8(200, 120, 240),
  );
}

/// demo 调色板（PaletteEntry._pad 私有 → 跨 crate 用 default + 逐字段赋值）
/// 极限场景用 14 色：草地 / 山岩 / 雪峰 / 城堡石 / 树叶 / 树干 / 河蓝 / 水晶青 / 水晶紫 / 水晶红 / 塔顶金 / 旗帜红
fn paint_demo_palette(grid: &mut gate_voxel::TileGrid) {
  let palette: &[(u8, [u8; 3], u8)] = &[
    ( 1, [ 86, 160,  70], 220), // 1 草地（L0 地面平原）
    ( 2, [140, 108,  76], 200), // 2 山岩（山体主体）
    ( 3, [240, 244, 248], 160), // 3 雪峰（y > 山线顶）
    ( 4, [172, 176, 190], 210), // 4 城堡石（墙/塔）
    ( 5, [ 52, 168,  72], 210), // 5 树叶（球冠）
    ( 6, [112,  72,  40], 200), // 6 树干（细柱）
    ( 7, [ 64, 140, 232], 170), // 7 河流蓝（L2 缠绕）
    ( 8, [ 68, 230, 220], 150), // 8 水晶青（L4 高亮）
    ( 9, [200, 120, 240], 150), // 9 水晶紫（L4 高亮）
    (10, [238,  76,  90], 160), // 10 水晶红 / 旗帜（L4 高亮）
    (11, [248, 210,  72], 160), // 11 塔顶金
    (12, [ 30,  32,  44], 220), // 12 道路/桥面（深灰）
    (13, [ 92, 118, 240], 180), // 13 浮空岛岛底
  ];
  let pal = grid.palette_mut();
  for &(idx, color, rough) in palette {
    let mut e = PaletteEntry::default();
    e.color = color;
    e.roughness = rough;
    pal.set(idx, e);
  }
}

// 极限场景版图：
// ─────────────────────────────────────────────────────────────────────
// 世界：tiles x∈[0..10] z∈[0..10] y∈[0..2] 共 10×3×10=300 实有 tile，
//       brickmap compute_window origin=(-1,-1,-1) dims=(12,5,12)=720 砖，
//       fine AABB ≈ (-512,-512,-512)..(5632,2048,5632)，对角穿越 ≈9000 fine。
// 内容分布（fine 单位 0..5120 XY 平面，y 高度）：
//   · 基础 L0 地面（全地图高 16）+ 起伏正弦高度场山体
//   · 4 座雪峰（四角，y 到 800）· 中部峡谷蜿蜒河流 L2
//   · 160 棵散点树（L2 粗节节省体素）在非山非河格
//   · 正中央 "天空堡"：L1 岛底倒锥 + L1 城堡外城墙 + 4 座角楼 + L2 高塔 + L3 金顶
//   · 3 处水晶矿簇（L4 多色，给 DDA 穿越厚度内密集命中以展示引擎吞吐）
//   · 保留 tile(1,0,0) 每 120 帧 L4 黄↔青交替（验证 132KB/180µs 增量上传路径）
//   · 世界大标语 "GATE ENGINE" 立在入口大道
// ─────────────────────────────────────────────────────────────────────
const EXT_N_TILES_X: i32 = 10;
const EXT_N_TILES_Z: i32 = 10;
const EXT_FINE_X: i32 = EXT_N_TILES_X * 512;
const EXT_FINE_Z: i32 = EXT_N_TILES_Z * 512;
const EXT_FINE_HALF: i32 = EXT_FINE_X / 2; // 2560

/// 正弦高度场（确定性，不用 rand）：
/// h(x,z) = 16 + A·sin(x·k1)·cos(z·k2) + 山体距 4 角的反比隆起
fn terrain_h(x: i32, z: i32) -> i32 {
  let xf = x as f32;
  let zf = z as f32;
  let s1 = (xf * 0.0045).sin() * (zf * 0.0053).cos();  // 基础波
  let s2 = ((xf + zf) * 0.0020).sin() * 1.3;             // 对角长波
  // 四角雪峰：与 (0,0)/(5120,0)/(0,5120)/(5120,5120) 的距离反比
  let corner_snow = |cx, cz| {
    let dx = xf - cx as f32;
    let dz = zf - cz as f32;
    let d = (dx * dx + dz * dz).sqrt().max(200.0);
    (600.0 / d).min(1.0) * 780.0 // 山顶 ~800
  };
  let sn = corner_snow(0, 0)
    + corner_snow(EXT_FINE_X - 1, 0)
    + corner_snow(0, EXT_FINE_Z - 1)
    + corner_snow(EXT_FINE_X - 1, EXT_FINE_Z - 1);
  let hf = 16.0 + s1 * 80.0 + s2 * 120.0 + sn;
  (hf as i32).clamp(16, 1020)
}

/// 离中央堡 (2560, 2560) 的距离
fn dist_to_castle(x: i32, z: i32) -> i32 {
  let dx = x - 2560;
  let dz = z - 2560;
  ((dx * dx + dz * dz) as f32).sqrt() as i32
}

/// 中央峡谷蜿蜒河（x,z 在河道走廊内返回 true）
fn in_river(x: i32, z: i32) -> bool {
  let t = z as f32 / EXT_FINE_Z as f32;     // 0..1
  let river_center = EXT_FINE_HALF as f32   // 中线 2560
    + (t * 6.28318).sin() * 700.0           // 正弦摆 ±700
    + ((t * 12.566).cos() * 180.0);         // 次级摆幅
  let dx = (x as f32) - river_center;
  dx.abs() < 64.0
}

fn build_demo_scene(grid: &mut gate_voxel::TileGrid) {
  // ================================================================
  //  (1) 基础地形：按 16 fine (L0) 步长采样高度场 → fill_box 铺柱
  //      y=16..h 使用 palette 2 山岩；y>h-40 且 h>520 → palette 3 雪峰
  // ================================================================
  let mut z = 0i32;
  while z < EXT_FINE_Z {
    let mut x = 0i32;
    while x < EXT_FINE_X {
      let h = terrain_h(x, z);
      let dc = dist_to_castle(x, z);
      // 中央堡区 (radius<700) 不开地形，后面由城堡结构接管
      let in_castle_plate = dc < 700;
      // 河道：y=16..24 填 palette 7，不叠加山岩
      let in_r = in_river(x, z);
      if !in_castle_plate {
        // 先铺 L0 地面（所有格子 y=0..16 统一由稍后 L0 全地图填，这里只填 16..h 山体）
        let top = h;
        let snow_line = top - 38;
        if in_r {
          // 河床：先填一层灰色岩，再加 8 fine 高河蓝
          fill_box(grid, IVec3::new(x, 16, z), IVec3::new(16, 12, 16), 0, 2);
          fill_box(grid, IVec3::new(x, 28, z), IVec3::new(16, 8, 16), 0, 7);
        } else if top > 16 {
          // 山体主体（palette 2 山岩）
          let rock_h = (snow_line - 16).max(0).min(top - 16);
          if rock_h > 0 {
            fill_box(grid, IVec3::new(x, 16, z), IVec3::new(16, rock_h, 16), 0, 2);
          }
          // 雪顶（h > 560 时，顶部 38 fine 改 palette 3 雪）
          if top > 560 {
            let snow_h = top - snow_line;
            if snow_h > 0 {
              fill_box(
                grid,
                IVec3::new(x, snow_line, z),
                IVec3::new(16, snow_h, 16),
                0,
                3,
              );
            }
          } else {
            // 低矮山坡顶部覆草（palette 1，最顶 16 fine）
            let grass_h = (top - 16).min(16);
            if grass_h > 0 && top - grass_h >= 16 {
              fill_box(
                grid,
                IVec3::new(x, top - grass_h, z),
                IVec3::new(16, grass_h, 16),
                0,
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
  // 全地图 L0 基础地板（y=0..16，pal 1 草地）——覆盖 0..10 tile 的 X/Z，Y=0..1
  fill_box(
    grid,
    IVec3::new(0, 0, 0),
    IVec3::new(EXT_FINE_X, 16, EXT_FINE_Z),
    0,
    1,
  );

  // ================================================================
  //  (2) 中央大道 + 入口大标语 "GATE ENGINE"
  // ================================================================
  // 中央大道（X 向，Z=2496..2528，宽 32 fine）深灰铺路
  fill_box(
    grid,
    IVec3::new(0, 16, 2496),
    IVec3::new(EXT_FINE_X, 8, 32),
    1,
    12,
  );
  // 大标语（L1，每像素 8³，放在入口 X=64 Y=32 Z=256 朝向 -Z）
  draw_text(grid, IVec3::new(96, 32, 256), "GATE ENGINE", 1, 8);

  // ================================================================
  //  (3) 中央天空之城（中央 2560±520 方区）
  //     · 浮空岛底（倒锥状，L1 pal 13 深蓝灰）底 y=600 顶 y=240
  //     · 城墙平台 400×400×32（y=240..272 pal 4 石）
  //     · 四面城墙 8 宽 × 64 高 × 400 长（y=272..336）
  //     · 四角角楼 64×64×128
  //     · 正殿 192×128×192（y=336..464）
  //     · 正殿中心高塔 80×80×192（y=464..656）+ L3 金顶球
  //     · 四角旗帜（L4 红飘带形状）
  // ================================================================
  let cx = 2560i32;
  let cz = 2560i32;
  // 浮空岛底（倒锥：按 y 降低半径收窄）——L0 级 16 步扫描
  let island_base_y = 128i32; // 岛底尖
  let island_top_y = 240i32; // 岛顶面（城墙在此升起）
  let top_r = 560i32;        // 顶面岛半径
  let mut y = island_base_y;
  while y < island_top_y {
    let t = (y - island_base_y) as f32 / (island_top_y - island_base_y) as f32;
    let r = (t * t.sqrt() * top_r as f32) as i32 + 16;
    fill_sphere(
      grid,
      IVec3::new(cx, y, cz),
      r,
      1,
      13,
    );
    y += 16;
  }
  // 城墙平台（y=240..272，512×512×32）
  fill_box(
    grid,
    IVec3::new(cx - 512, 240, cz - 512),
    IVec3::new(1024, 32, 1024),
    1,
    4,
  );
  // 四面城墙（y=272..336 = 64 高）
  // 北 Z=cz-512, 南 Z=cz+512-8
  fill_box(
    grid,
    IVec3::new(cx - 512, 272, cz - 512),
    IVec3::new(1024, 64, 8),
    1,
    4,
  );
  fill_box(
    grid,
    IVec3::new(cx - 512, 272, cz + 512 - 8),
    IVec3::new(1024, 64, 8),
    1,
    4,
  );
  fill_box(
    grid,
    IVec3::new(cx - 512, 272, cz - 512),
    IVec3::new(8, 64, 1024),
    1,
    4,
  );
  fill_box(
    grid,
    IVec3::new(cx + 512 - 8, 272, cz - 512),
    IVec3::new(8, 64, 1024),
    1,
    4,
  );
  // 四角角楼 96×96×160（从 y=272 起比城墙多高 96）
  for &(ox, oz) in &[(-512 + 0, -512 + 0), (512 - 96, -512 + 0), (-512 + 0, 512 - 96), (512 - 96, 512 - 96)] {
    let tx = cx + ox;
    let tz = cz + oz;
    fill_box(grid, IVec3::new(tx, 272, tz), IVec3::new(96, 160, 96), 1, 4);
    // 角楼顶金色 16
    fill_box(grid, IVec3::new(tx, 272 + 160, tz), IVec3::new(96, 16, 96), 1, 11);
  }
  // 正殿（中心，y=336..464 = 128 高，长 256×256）
  fill_box(
    grid,
    IVec3::new(cx - 256, 336, cz - 256),
    IVec3::new(512, 128, 512),
    1,
    4,
  );
  // 正殿正门（Z- 方向，挖一矩形门洞：clear_voxel）
  {
    // L1 每步 = 8 fine；宽 96 → 12 步 × 高 96 → 12 步 × 深 8 → 1 步
    let level: gate_voxel::Level = 1;
    let e = 8i32; // L1 边长
    let mn = IVec3::new(cx - 48, 336, cz - 264);
    let ex = mn + IVec3::new(96, 96, 8);
    let mut z = mn.z.div_euclid(e) * e;
    while z < ex.z {
      let mut y = mn.y.div_euclid(e) * e;
      while y < ex.y {
        let mut x = mn.x.div_euclid(e) * e;
        while x < ex.x {
          grid.clear_voxel(IVec3::new(x, y, z), level);
          x += e;
        }
        y += e;
      }
      z += e;
    }
  }
  // 高塔（y=464..720 = 256 高，底 96×96 上收顶）
  fill_box(grid, IVec3::new(cx - 48, 464, cz - 48), IVec3::new(96, 256, 96), 1, 4);
  // 塔顶平台 128×128×16
  fill_box(grid, IVec3::new(cx - 64, 720, cz - 64), IVec3::new(128, 16, 128), 1, 11);
  // 金顶球（L2 r=64，塔顶 y=720+80=800）
  fill_sphere(grid, IVec3::new(cx, 800, cz), 64, 2, 11);
  // 四角旗帜（L4 红飘带：从角楼顶 4 角斜向上拉出小立方体串）
  for &(ox, oz) in &[(-512 + 0, -512 + 0), (512 - 16, -512 + 0), (-512 + 0, 512 - 16), (512 - 16, 512 - 16)] {
    let fx = cx + ox + 4;
    let fz = cz + oz + 4;
    for s in 0..16i32 {
      fill_box(
        grid,
        IVec3::new(fx + s, 448 + s * 4, fz),
        IVec3::new(8, 8, 8),
        4,
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
      let x = (i * 211 + 83) % EXT_FINE_X;
      let z = ((i * 977 + 419) ^ 0xA53) % EXT_FINE_Z;
      let x = x.abs();
      let z = z.abs();
      let h = terrain_h(x, z);
      let ok = dist_to_castle(x, z) > 900
        && !in_river(x, z)
        && !(z >= 2496 && z <= 2528) // 中央大道
        && h < 360;
      if ok {
        // 树干 （L1, 24 宽 16 宽 24 深 高 80）
        fill_box(
          grid,
          IVec3::new(x, h, z),
          IVec3::new(32, 80, 32),
          1,
          6,
        );
        // 树叶球（L2，r=80，中心在树干顶 + 80）
        fill_sphere(
          grid,
          IVec3::new(x + 16, h + 80 + 64, z + 16),
          80,
          2,
          5,
        );
        n_planted += 1;
      }
      i += 1;
    }
  }

  // ================================================================
  //  (5) 3 处 L4 水晶矿簇（每处 60~100 个 1³ 彩色小立方体，密集堆叠，
  //      用来展示"高分辨率体素不拖垮 fps"——
  //      因为 DDA 命中就 break，不随地图体素数上升而变慢）
  // ================================================================
  let crystal_clusters: &[(i32, i32, i32, u8, u8, i32)] = &[
    // (cx, cz, count, pal_a, pal_b, seed)
    ( 480,  480, 90,  8,  9, 131), // 左上（near 角雪峰脚）水晶青/紫
    (4608,  640, 80, 10,  8, 251), // 右上 水晶红/青（seed<255 安全）
    ( 768, 4352, 70,  9, 10, 223), // 左下 水晶紫/红
  ];
  for &(cx_, cz_, cnt, pa, pb, seed) in crystal_clusters.iter() {
    let h = terrain_h(cx_, cz_);
    let mut t = 0i32;
    let mut placed = 0usize;
    while placed < cnt as usize {
      // 确定性位置抖动
      let sx = ((t * 37 + seed) % 96) - 48;
      let sz = ((t * 131 + seed * 3) % 96) - 48;
      let hy = ((t * 53) % 14); // 高度 0..13 fine
      let pal = if (t & 1) == 0 { pa } else { pb };
      let px_ = cx_ + sx;
      let pz_ = cz_ + sz;
      if px_ >= 0 && pz_ >= 0 && px_ < EXT_FINE_X && pz_ < EXT_FINE_Z {
        fill_box(
          grid,
          IVec3::new(px_, h + hy, pz_),
          IVec3::new(1, 1, 1),
          4,
          pal,
        );
        placed += 1;
      }
      t += 1;
    }
  }

  // ================================================================
  //  (6) 保留：tile(1,0,0) 内 L4 32³ 热点（每 120 帧黄↔青交替）
  //      tile(2,0,0) 中心 32³ 蓝 L4（旧 demo）
  //      → 保持增量上传特性展示（UPLOAD 132KB/180µs）
  // ================================================================
  fill_box(grid, IVec3::new(656, 64, 64), IVec3::new(32, 32, 32), 4, 11);
  fill_box(
    grid,
    IVec3::new(1264, 240, 240),
    IVec3::new(32, 32, 32),
    4,
    7,
  );
  let t0 = gate_voxel::TileCoord::new(0, 0, 0);
  grid.set_state(0, 1, 0xDEAD);
  grid.set_state(1, 0, 0xBEEF);
  grid.set_comp(t0, 0, 0x1122);
  grid.set_comp(t0, 1, 0x3344);
}

/// 轨道相机输入（P2.6 spec FR-3/FR-4）：
/// - 右键拖拽 = 旋转（yaw -= dx·ROT_SPEED, pitch += dy·ROT_SPEED）
/// - 中键拖拽 = 平移（right/up 正交基，PAN_PER_PX = distance·2tan(fov/2)/窗口物理高度，视觉 1:1）
/// - 滚轮 = 乘法缩放（distance *= exp(-line·0.35)）
/// - 拖拽类互斥（旋转 > 平移），滚轮可与拖拽共存
/// - 末尾同帧重建 DdaCameraConfig（from_orbit 唯一矩阵构造点）→ ≤1 帧生效
fn orbit_camera_input(
  mouse: Res<ButtonInput<MouseButton>>,
  motion: Res<AccumulatedMouseMotion>,
  scroll: Res<AccumulatedMouseScroll>,
  captured: Res<gate_ui::UiPointerCaptured>,
  windows: Query<&Window>,
  mut orbit: ResMut<OrbitCamera>,
  mut cfg: ResMut<DdaCameraConfig>,
) {
  // UI 指针捕获优先（2.7a FR-6）：hover/按下控件时吞掉拖拽/滚轮；
  // 但 cfg 重建在 gate 之外——resize 期间即便指针在 UI 上，aspect 也要跟上
  if !captured.0 {
    let delta = motion.delta;
    let rotating = mouse.pressed(MouseButton::Right);
    let panning = mouse.pressed(MouseButton::Middle);

    if rotating {
      orbit.yaw -= delta.x * ROT_SPEED;
      orbit.pitch += delta.y * ROT_SPEED;
      orbit.clamp();
    } else if panning {
      // 正交基：forward = eye→target；right = forward × Y；up = right × forward
      // （与 look_at_rh 的 xaxis/yaxis 同构；pitch ±89° clamp 保证 forward 不与 Y 共线）
      let (sin_yaw, cos_yaw) = orbit.yaw.sin_cos();
      let (sin_pitch, cos_pitch) = orbit.pitch.sin_cos();
      let forward = -Vec3::new(sin_yaw * cos_pitch, sin_pitch, cos_yaw * cos_pitch);
      let right = forward.cross(Vec3::Y).normalize();
      let up = right.cross(forward).normalize();
      let pan_per_px = orbit.distance * 2.0 * (FOV_Y / 2.0).tan() / window_height(&windows);
      orbit.target += (right * (-delta.x) + up * delta.y) * pan_per_px;
    }

    // 滚轮缩放独立于拖拽（spec：滚轮可与拖拽共存）；Line 单位，Pixel 按典型行高 16px 折算
    let lines = match scroll.unit {
      MouseScrollUnit::Line => scroll.delta.y,
      MouseScrollUnit::Pixel => scroll.delta.y / 16.0,
    };
    if lines != 0.0 {
      orbit.distance *= (-lines * ZOOM_LOG_SPEED).exp();
      orbit.clamp();
    }
  }

  // 每帧无条件重建矩阵（幂等；成本 = 一次 4×4 求逆，可忽略），省 dirty 标记。
  // aspect 读当前窗口物理尺寸（FR-5：resize 后 ≤1 帧生效，2.6 的恒定假设解除）
  let Ok(window) = windows.single() else { return };
  let size = UVec2::new(
    window.physical_width().max(1),
    window.physical_height().max(1),
  );
  *cfg = DdaCameraConfig::from_orbit(
    &orbit,
    FOV_Y,
    size.x as f32 / size.y as f32,
    CAM_NEAR,
    CAM_FAR,
  );
}

/// 世界空间 UI 相机镜像（gate-ui 不依赖 gate-render：通用资源传递投影矩阵）；
/// after(orbit_camera_input) 保证同帧拿到刚重建的 cfg
fn sync_anchor_camera(cfg: Res<DdaCameraConfig>, mut anchor_cam: ResMut<gate_ui::AnchorCamera>) {
  anchor_cam.view_proj = cfg.view_proj;
  anchor_cam.position_world = cfg.position_world;
}

/// 主窗口物理高度（pan_per_px 1:1 基准；无窗口时回退 VIEW_SIZE.y）
fn window_height(windows: &Query<&Window>) -> f32 {
  windows
    .single()
    .map(|w| w.physical_height().max(1) as f32)
    .unwrap_or(VIEW_SIZE.y as f32)
}

/// 每 120 帧改一次 tile (1,0,0) 触发增量上传（验证 UPLOAD[incremental] 日志）
/// 用 set_voxel 填 palette 交替 → 确保一定产生 DirtyEdit（而不是 clear 空胞 no-op）
/// 极限场景 palette：11 金 / 8 青（高对比肉眼可见、且不与森林树叶 5/树干 6 语义冲突）
fn edit_tile_every_120_frames(mut frame: Local<u64>, scene: Option<ResMut<VoxelScene>>) {
  *frame += 1;
  let Some(mut scene) = scene else { return };
  if *frame == 1 {
    scene.demo_force_full_rebuild = false;
  }
  if (*frame).is_multiple_of(120) {
    // 每 120 帧重写 tile(1,0,0) 内 32³ L4 热点（fine 656..688），palette 金↔青交替
    // 与 setup 热点重合同区域 → 必定产生 DirtyEdit → UPLOAD[incremental] ~ 132KB / ~180µs
    let pal = if (*frame / 120) % 2 == 1 { 11 } else { 8 };
    let origin = IVec3::new(656, 64, 64);
    let mut z = 0;
    while z < 32 {
      let mut y = 0;
      while y < 32 {
        let mut x = 0;
        while x < 32 {
          scene.grid.set_voxel(origin + IVec3::new(x, y, z), 4, pal);
          x += 1;
        }
        y += 1;
      }
      z += 1;
    }
  }
}

// ================= 2.7a demo 性能面板（FR-2/4/5/6 实机验收载体） =================

#[derive(Component)]
struct DemoFpsLabel;

/// 旧：滑杆数值标签（性能面板的滑杆）。用户已删除该面板，此组件保留以防
/// 未来恢复面板时重写；当前主程序不 spawn、测试也不引用。
#[allow(dead_code)]
#[derive(Component)]
struct DemoSliderLabel;

/// FPS 折线图 plot root marker；与 GPU 面板的 GpuPlot 分开，避免 Query::single_mut() 冲突
#[derive(Component)]
struct DemoFpsPlot;

/// 主题就绪后 spawn 一次（RON 成功或回退默认都会插入 UiTheme 资源）
///
/// 等待条件：主题字体资产已加载为 `LoadState::Loaded`。若 theme.font_path = None
/// 则用 SystemUi fallback（不等待，直接 spawn）。必须等字体到位再 spawn UI 标签，
/// 否则 TextPipeline 在 FiraMono-subset（Bevy 内置 default slot，不含 CJK）上生成
/// 字形缓存，即使之后字体 override default slot，已缓存的 atlas 条目仍回退 CJK
/// 字形为方框（parley 的字形请求不因为 Assets slot 变化而自动失效）。
fn demo_ui_setup(
  mut done: Local<bool>,
  theme: Option<Res<UiTheme>>,
  font: Option<Res<ThemeFont>>,
  server: Option<Res<AssetServer>>,
  mut commands: Commands,
  mut images: ResMut<Assets<Image>>,
) {
  if *done {
    return;
  }
  let Some(theme) = theme else {
    return;
  };
  // 字体就绪门控：theme.font_path=Some(_) 时必须等 ThemeFont.handle + Loaded，
  // 否则首次栅格会用 FiraMono-subset，中文/符号字符（fps 文本默认英文所以这里
  // 纯 ASCII 没问题，但统一遵守字体就绪判定避免以后换字符串时又出方框）。
  let font_handle = match (server.as_ref(), font.as_ref(), theme.font_path.as_ref()) {
    (Some(srv), Some(f), Some(_)) => match &f.handle {
      Some(h) if matches!(srv.load_state(h.id()), LoadState::Loaded) => Some(h.clone()),
      _ => return,
    },
    (_, _, None) => None,
    _ => return,
  };
  *done = true;
  let theme = theme.clone();
  let font = font_handle;
  let plot_handle = images.add(blank_plot_image(256, 64));

  commands.queue(move |world: &mut World| {
    let ctx = UiCtx::new(&theme, font.as_ref());
    let mut fps_label = Entity::PLACEHOLDER;
    let mut fps_plot_e = Entity::PLACEHOLDER;
    world
      .spawn(Node {
        position_type: PositionType::Absolute,
        left: px(8.0),
        top: px(8.0),
        row_gap: px(2.0),
        flex_direction: FlexDirection::Column,
        align_items: AlignItems::FlexStart,
        ..default()
      })
      .with_children(|root| {
        fps_label = label(&ctx, root, "fps --");
        fps_plot_e = plot(&ctx, root, plot_handle, 128, PlotDomain::Fixed(0.0, 120.0));
      });
    world.entity_mut(fps_plot_e).insert(DemoFpsPlot);
    world.entity_mut(fps_label).insert(DemoFpsLabel);
  });
}

/// 每 0.25s 统计帧率 → push 进 PlotData（Fixed 0..120 域）+ fps 标签
fn demo_fps_feed(
  mut acc_t: Local<f32>,
  mut frames: Local<u32>,
  time: Res<Time>,
  mut plots: Query<&mut PlotData, With<DemoFpsPlot>>,
  mut labels: Query<&mut Text, With<DemoFpsLabel>>,
) {
  *acc_t += time.delta_secs();
  *frames += 1;
  if *acc_t < 0.25 {
    return;
  }
  let fps = *frames as f32 / *acc_t;
  if let Ok(mut pd) = plots.single_mut() {
    pd.push(fps);
  }
  let txt = format!("fps {fps:.0}");
  if let Ok(mut t) = labels.single_mut()
    && t.0 != txt
  {
    t.0 = txt;
  }
  *acc_t = 0.0;
  *frames = 0;
}

/// 相机 zoom / 消失问题临时诊断（每 0.5s 输出）：
/// 对比 CPU 侧「无 AABB 直走 DDA」vs「AABB 跳步 DDA」在 32x32 视锥网格的命中差异。
/// WGSL 版 AABB 算法是 Rust `cpu_reference_dda_ray_aabb_skip` 的逐字翻译，
/// 所以只要这两套 CPU DDA 有差异，WGSL 版也必然出同样的 bug（反之亦然）。
fn debug_aabb_report(
  mut acc_t: Local<f32>,
  time: Res<Time>,
  scene: Res<VoxelScene>,
  orbit: Res<OrbitCamera>,
  cfg: Res<DdaCameraConfig>,
  windows: Query<&Window>,
) {
  *acc_t += time.delta_secs();
  if *acc_t < 2.0 { // 放宽到 2s 一次，避免影响 fps 基线评估
    return;
  }
  *acc_t = 0.0;
  // 用当前 orbit 重新构造（和实时 cfg 一致，aspect 读物理尺寸）
  let Ok(window) = windows.single() else { return };
  let aspect = (window.physical_width().max(1) as f32) / (window.physical_height().max(1) as f32);
  let live_cfg = DdaCameraConfig::from_orbit(&orbit, FOV_Y, aspect, CAM_NEAR, CAM_FAR);
  let eye = live_cfg.position_world;
  let distance = orbit.distance;
  let target = orbit.target;

  // 用 main-world 的 TileGrid 构建一份 CPU buffers（和 GPU 上传侧一致的 tile window）
  use gate_render::{BrickMapBuilder, BrickMapGlobals};
  let t0 = std::time::Instant::now();
  let builder = BrickMapBuilder::build_full(&scene.grid);
  let build_us = t0.elapsed().as_micros();
  let buffers = builder.buffers().clone();
  let g: BrickMapGlobals = buffers.globals;
  // BrickMapGlobals 中 AABB 是 tile 坐标，fine 坐标 AABB：
  //   [origin*512, (origin+dims)*512)
  let origin_fine_min = Vec3::new(
    (g.index_origin_x as f32) * 512.0,
    (g.index_origin_y as f32) * 512.0,
    (g.index_origin_z as f32) * 512.0,
  );
  let origin_fine_max = Vec3::new(
    ((g.index_origin_x + g.index_dims_x as i32) as f32) * 512.0,
    ((g.index_origin_y + g.index_dims_y as i32) as f32) * 512.0,
    ((g.index_origin_z + g.index_dims_z as i32) as f32) * 512.0,
  );
  let inv_vp = live_cfg.inv_view_proj;
  // 只跑 AABB-skip 版（16384 步，32×32 网格 = 53 万步 CPU，<1ms），用于判断：
  //   a) Rust 侧认为这帧视锥有多少射线命中 brickmap（如果 0 就说明 GPU 同样也 0）
  //   b) brickmap globals 实际取到的 tile window 是否覆盖世界 0..5120 fine
  let mut skip_hits = 0usize;
  let grid_w = 32usize;
  for iy in 0..grid_w {
    for ix in 0..grid_w {
      let u = (ix as f32 + 0.5) / (grid_w as f32) * 2.0 - 1.0;
      let v = 1.0 - ((iy as f32 + 0.5) / (grid_w as f32)) * 2.0;
      let near = inv_vp * Vec4::new(u, v, 0.0, 1.0);
      let far = inv_vp * Vec4::new(u, v, 1.0, 1.0);
      let near = near.truncate() / near.w;
      let far = far.truncate() / far.w;
      let diff = far - near;
      let frustum_len = diff.length();
      let dir = diff.normalize();
      let r = cpu_reference_dda_ray_aabb_skip(
        &buffers, eye, dir, frustum_len, 16384, origin_fine_min, origin_fine_max,
      );
      if r.is_some() { skip_hits += 1; }
    }
  }
  let total_px = grid_w * grid_w;
  bevy::log::info!(
    "AABB-DBG | dist={distance:.0} build_full={build_us}µs eye=({:.0},{:.0},{:.0}) target=({:.0},{:.0},{:.0})\n\
     AABB window(tile): origin=({},{},{}) dims=({},{},{})\n\
     AABB fine: min[{:.0},{:.0},{:.0}] max[{:.0},{:.0},{:.0}]\n\
     32x32 skip_hits={skip_hits}/{total_px}  (0 = GPU 同样全黑；>100 = GPU 也应该有像素)",
    eye.x, eye.y, eye.z, target.x, target.y, target.z,
    g.index_origin_x, g.index_origin_y, g.index_origin_z,
    g.index_dims_x, g.index_dims_y, g.index_dims_z,
    origin_fine_min.x, origin_fine_min.y, origin_fine_min.z,
    origin_fine_max.x, origin_fine_max.y, origin_fine_max.z,
  );
}

/// slider 值 → 标签跟随（旧性能面板保留函数，面板已删除）
#[allow(dead_code)]
fn demo_slider_label(
  sliders: Query<&SliderValue>,
  mut labels: Query<&mut Text, With<DemoSliderLabel>>,
) {
  let Ok(v) = sliders.single() else { return };
  let txt = format!("{:.0}", v.0);
  if let Ok(mut t) = labels.single_mut()
    && t.0 != txt
  {
    t.0 = txt;
  }
}

/// checkbox 翻转 → 切换**两张图**（FPS + GPU total）的折线图网格线（旧性能面板保留函数，面板已删除）
#[allow(dead_code)]
fn demo_checkbox_effect(
  mut prev: Local<Option<bool>>,
  checks: Query<Has<Checked>, With<Checkable>>,
  mut plots: Query<&mut PlotData>,
) {
  let Ok(now) = checks.single() else { return };
  if Some(now) != *prev {
    for mut pd in plots.iter_mut() {
      pd.grid = now;
    }
    *prev = Some(now);
  }
}

/// 窗口 resize 事件 → 环形列表回显（旧性能面板保留函数，面板已删除）
#[allow(dead_code)]
fn demo_resize_log(mut evr: MessageReader<WindowResized>, mut lists: Query<&mut RingList>) {
  for ev in evr.read() {
    let Ok(mut rl) = lists.single_mut() else {
      continue;
    };
    rl.push(format!("resize {}x{}", ev.width as u32, ev.height as u32));
  }
}

// ================= 2.7 GPU timings =================
// Spec: 从 DiagnosticsStore（由 bevy_render::sync_diagnostics 在 PreUpdate 写入 main world）
// 抽取 4×2 + 1×CPU 共 9 条实时耗时，填 GpuPassTimings 资源。
// Render 世界的 DiagnosticsStore 在本 App 有 RenderDiagnosticsPlugin（RenderPlugin 默认装配）时，
// 每帧 PreUpdate 阶段把 RenderDiagnosticsMutex 解包写入 DiagnosticsStore。
//
// 上传段的 CPU ms 不经过 DiagnosticsStore（OQ-2 选 A，GPU 拷贝在 submit 无法测），
// 而通过 Arc<Mutex<Option<UploadCpuSample>>> 共享通道读（见 gate-render upload.rs）。
// ----------------------------------------------------------------------------

/// 4×pass（GC/GB/DC/DB）+ 1×upload 的 pass-level GPU/CPU 耗时快照（main world 资源）
#[derive(Resource, Clone, Copy, Debug)]
pub struct GpuPassTimings {
  pub gc_gpu_ms: f32,
  pub gc_cpu_ms: f32,
  pub gb_gpu_ms: f32,
  pub gb_cpu_ms: f32,
  pub dc_gpu_ms: f32,
  pub dc_cpu_ms: f32,
  pub db_gpu_ms: f32,
  pub db_cpu_ms: f32,
  /// OQ-2 选 A：上传段的 elapsed_gpu 不支持，UI 显示"—"（NAN 哨兵）
  pub up_gpu_ms: f32,
  /// 上传段 CPU ms（来自 UploadCpuSampleChannel 的共享 sample）
  pub up_cpu_ms: f32,
  /// 4×pass 的 GPU 时间之和（任一 NAN 视为 0，即可能非完整Σ）
  pub total_gpu_ms: f32,
  /// timestamp_query feature 缺失时置 true，面板提示一次
  pub gpu_unsupported: bool,
  /// 每 0.5s +1（触发 UI plot push + 文本刷新判定）
  pub generation: u64,
}

impl Default for GpuPassTimings {
  fn default() -> Self {
    // NAN 哨兵 = 缺失值；UI 显示"—"
    let nan = f32::NAN;
    Self {
      gc_gpu_ms: nan,
      gc_cpu_ms: nan,
      gb_gpu_ms: nan,
      gb_cpu_ms: nan,
      dc_gpu_ms: nan,
      dc_cpu_ms: nan,
      db_gpu_ms: nan,
      db_cpu_ms: nan,
      up_gpu_ms: nan,
      up_cpu_ms: nan,
      total_gpu_ms: 0.0,
      gpu_unsupported: false,
      generation: 0,
    }
  }
}

/// 5 个渲染段名的 Diagnostic 路径表（与 gate-render 的 span name 严格对应，C2）
const SEGMENTS: [(&str, &str); 5] = [
  ("gate_gradient_compute", "GC"),
  ("gate_gradient_blit", "GB"),
  ("gate_dda_compute", "DC"),
  ("gate_dda_blit", "DB"),
  ("gate_brickmap_upload", "UP"),
];

/// 从 DiagnosticsStore 读一条 path 的 smoothed 值；缺失 → NAN
fn read_one(store: &DiagnosticsStore, seg: &str, field: &str) -> f32 {
  let path = DiagnosticPath::from_components(["render", seg, field]);
  store
    .get(&path)
    .and_then(|d| d.smoothed())
    .map(|v| v as f32)
    .unwrap_or(f32::NAN)
}

/// Update 阶段从 DiagnosticsStore + UploadCpuSampleChannel → GpuPassTimings（main world）。
/// 频率：每 0.5s 刷新一次 generation（0.5s 以下 Plot 会刷爆 CPU memcpy；be 2.7a fps_plot 是 0.25s）
/// 当前 UI 不再显示 GPU 面板，本函数保留给 headless 测试用（sync_populates_10_paths 等）。
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
fn sync_gpu_timings(
  mut timings: ResMut<GpuPassTimings>,
  store: Option<Res<DiagnosticsStore>>,
  upload: Option<Res<UploadCpuSampleChannel>>,
  mut acc: Local<f32>,
  mut seen_upload_gen: Local<u64>,
  time: Res<Time>,
) {
  *acc += time.delta_secs();
  let t0 = std::time::Instant::now();
  let threshold = 0.5;
  if let Some(store) = store {
    let ((gc_g, gc_c), (gb_g, gb_c), (dc_g, dc_c), (db_g, db_c), (up_g, up_c_raw)) = (
      (
        read_one(&store, SEGMENTS[0].0, "elapsed_gpu"),
        read_one(&store, SEGMENTS[0].0, "elapsed_cpu"),
      ),
      (
        read_one(&store, SEGMENTS[1].0, "elapsed_gpu"),
        read_one(&store, SEGMENTS[1].0, "elapsed_cpu"),
      ),
      (
        read_one(&store, SEGMENTS[2].0, "elapsed_gpu"),
        read_one(&store, SEGMENTS[2].0, "elapsed_cpu"),
      ),
      (
        read_one(&store, SEGMENTS[3].0, "elapsed_gpu"),
        read_one(&store, SEGMENTS[3].0, "elapsed_cpu"),
      ),
      (
        read_one(&store, SEGMENTS[4].0, "elapsed_gpu"),
        read_one(&store, SEGMENTS[4].0, "elapsed_cpu"),
      ),
    );
    timings.gc_gpu_ms = gc_g;
    timings.gc_cpu_ms = gc_c;
    timings.gb_gpu_ms = gb_g;
    timings.gb_cpu_ms = gb_c;
    timings.dc_gpu_ms = dc_g;
    timings.dc_cpu_ms = dc_c;
    timings.db_gpu_ms = db_g;
    timings.db_cpu_ms = db_c;
    timings.up_gpu_ms = up_g;
    if !up_c_raw.is_nan() {
      timings.up_cpu_ms = up_c_raw;
    }
    // gpu_unsupported = 4×render pass 的 elapsed_gpu 全 NAN 且 elapsed_cpu 至少一项有数
    let any_cpu = !gc_c.is_nan() || !gb_c.is_nan() || !dc_c.is_nan() || !db_c.is_nan();
    let all_gpu_nan = gc_g.is_nan() && gb_g.is_nan() && dc_g.is_nan() && db_g.is_nan();
    if any_cpu && all_gpu_nan {
      timings.gpu_unsupported = true;
    }
    // total_gpu_ms：非 NAN 求和；NAN 不计
    let mut total = 0.0f32;
    for v in [gc_g, gb_g, dc_g, db_g] {
      if !v.is_nan() {
        total += v;
      }
    }
    timings.total_gpu_ms = total;
  }
  // upload CPU ms 同步（若 channel 有新样本且 generation 变更则采用）
  if let Some(ch) = upload
    && let Ok(guard) = ch.0.lock()
    && let Some(s) = *guard
    && s.generation != *seen_upload_gen
  {
    timings.up_cpu_ms = s.cpu_ms;
    *seen_upload_gen = s.generation;
  }
  // 每 0.5s bump generation（Plot 每 0.5s push 一个样本 → 128 样本 ≈ 64s 覆盖）
  if *acc >= threshold {
    *acc -= threshold;
    timings.generation = timings.generation.wrapping_add(1);
    let _ = t0; // 避免未使用警告
  }
}

// ---- P2.7 GPU 面板 UI：demo_gpu_panel_setup / demo_gpu_refresh ----

/// Σ 总行文本（超预算 → 变红）标签 marker
#[derive(Component)]
struct GpuTotalLabel;
/// 每段 2 列（GPU ms / CPU ms）marker（SEGMENTS 顺序 × 2）。字段.0 = 0..10 索引
#[derive(Component)]
struct GpuCell(usize);
/// 折线图 plot marker（GPU total 历史，容量 128，域 Fixed 0..20）
#[derive(Component)]
struct GpuPlot;
/// DC 行容器（背景 accent_hover 色差，便于一眼找 DDA compute）
#[derive(Component)]
struct DcRowBg;
/// 6 行事件 RingList marker（|Δ|>2ms 回显、gpu_unsupported 一次性提示）
#[derive(Component)]
struct GpuEventList;

const SIXTY_FPS_BUDGET_MS: f32 = 16.7;
const EVENT_TICK_THRESHOLD_MS: f32 = 2.0;

/// gpu panel 构建（主题就绪后 spawn 一次，和 demo_ui_setup 同 trigger）
///
/// 与 demo_ui_setup 相同的字体就绪门槛：theme.font_path 存在时必须等
/// `LoadState::Loaded`，否则首次栅格用 FiraMono，CJK/希腊字母变方框。
fn demo_gpu_panel_setup(
  mut done: Local<bool>,
  theme: Option<Res<UiTheme>>,
  font: Option<Res<ThemeFont>>,
  server: Option<Res<AssetServer>>,
  mut commands: Commands,
  mut images: ResMut<Assets<Image>>,
) {
  if *done {
    return;
  }
  let Some(theme) = theme else { return };
  let font_handle = match (server.as_ref(), font.as_ref(), theme.font_path.as_ref()) {
    (Some(srv), Some(f), Some(_)) => match &f.handle {
      Some(h) if matches!(srv.load_state(h.id()), LoadState::Loaded) => Some(h.clone()),
      _ => return,
    },
    (_, _, None) => None,
    _ => return,
  };
  *done = true;
  let theme = theme.clone();
  let font = font_handle;
  let plot_handle = images.add(blank_plot_image(PLOT_W, PLOT_H));

  commands.queue(move |world: &mut World| {
    let ctx = UiCtx::new(&theme, font.as_ref());
    let mut total_label = Entity::PLACEHOLDER;
    let mut gpu_plot = Entity::PLACEHOLDER;
    let mut dc_row_bg = Entity::PLACEHOLDER;
    let mut event_list = Entity::PLACEHOLDER;
    // 5 段 × 2 列（gpu/cpu），顺序 = SEGMENTS[0..] × (gpu,cpu)
    let mut cells: [Entity; 10] = [Entity::PLACEHOLDER; 10];

    world
      .spawn(Node {
        position_type: PositionType::Absolute,
        right: Val::Percent(2.0),
        top: Val::Percent(5.0),
        width: Val::Percent(32.0),
        min_width: px(320.0),
        max_width: px(520.0),
        ..default()
      })
      .with_children(|root| {
        let panel_e = panel(&ctx, root);
        root.world_mut().entity_mut(panel_e).with_children(|p| {
          label(&ctx, p, "GPU 耗时面板 · 2.7");
          // Σ GPU 行（阈值 >16.7 → danger 色）
          total_label = label(&ctx, p, "Σ GPU —");
          p.world_mut().entity_mut(total_label).insert(GpuTotalLabel);

          // 折线图：total_gpu 历史（OQ-3 已答：Fixed 0..20ms，128 样本 ≈ 64s）
          let e = plot(&ctx, p, plot_handle, 128, PlotDomain::Fixed(0.0, 20.0));
          p.world_mut().entity_mut(e).insert(GpuPlot);
          gpu_plot = e;

          // 表头 + 5 行 3 列（段名 | GPU ms | CPU ms）
          // --- 表头（muted）
          world_mini_head(&ctx, p);
          // 5 段
          for (i, (seg, tag)) in SEGMENTS.iter().enumerate() {
            let cells_pair = build_row(&ctx, p, tag, seg, i == 2 /* DC row */, i == 2);
            cells[i * 2] = cells_pair.0;
            cells[i * 2 + 1] = cells_pair.1;
            if i == 2 {
              dc_row_bg = cells_pair.2;
            }
          }
          // 6 行 RingList（回显 Δ>2ms 与 unsupported 一次性提示）
          let e = list(&ctx, p, 6);
          p.world_mut().entity_mut(e).insert(GpuEventList);
          event_list = e;
        });
      });

    // 注册 marker（已在构建时插入，只需保留句柄供查询：cells/total_label/dc_row_bg）
    for (idx, c) in cells.iter().enumerate() {
      world.entity_mut(*c).insert(GpuCell(idx));
    }
    // 事件列表：一次性注入 "gpu_unsupported 提示" 标记，refresh 每帧读 timings
    world.entity_mut(event_list); // 占位：refresh 负责 push 文本
    world.entity_mut(total_label);
    world.entity_mut(gpu_plot);
    if dc_row_bg != Entity::PLACEHOLDER {
      world.entity_mut(dc_row_bg).insert(DcRowBg);
    }
  });
}

/// 小工具：表头（段 | GPU | CPU，muted）
fn world_mini_head(ctx: &UiCtx, r: &mut ChildSpawner) {
  r.spawn(Node {
    flex_direction: FlexDirection::Row,
    column_gap: px(ctx.theme.metrics.spacing.xs),
    width: Val::Percent(100.0),
    padding: UiRect::horizontal(px(ctx.theme.metrics.spacing.sm)),
    ..default()
  })
  .with_children(|row| {
    // 30% / 35% / 35% 三段
    row
      .spawn(Node {
        width: Val::Percent(30.0),
        ..default()
      })
      .with_children(|c| {
        label_muted(ctx, c, "段名");
      });
    row
      .spawn(Node {
        width: Val::Percent(35.0),
        ..default()
      })
      .with_children(|c| {
        label_muted(ctx, c, "GPU ms");
      });
    row
      .spawn(Node {
        width: Val::Percent(35.0),
        ..default()
      })
      .with_children(|c| {
        label_muted(ctx, c, "CPU ms");
      });
  });
}

/// 小工具：构建 3 列一行（tag | gpu_cell | cpu_cell），返回 (gpu_entity, cpu_entity, row_entity)
/// 若 accent=true（DC 行），整行背景 = accent_hover（低 alpha，半透 0x44 叠色）
fn build_row(
  ctx: &UiCtx,
  p: &mut ChildSpawner,
  tag: &str,
  _seg: &str,
  accent: bool,
  _is_dc: bool,
) -> (Entity, Entity, Entity) {
  let m = &ctx.theme.metrics;
  let c = &ctx.theme.colors;
  let mut gpu_e = Entity::PLACEHOLDER;
  let mut cpu_e = Entity::PLACEHOLDER;
  let row_bg_color = if accent {
    // accent_hover 半透明底；color_of 不暴露 alpha 接口 → 手动叠 0.15 alpha
    let base = color_of(&c.accent_hover);
    let mut srgba = base.to_srgba();
    srgba.alpha = 0.15;
    Color::Srgba(srgba)
  } else {
    Color::NONE
  };
  let row_e = p
    .spawn((
      Name::new("gpu-row"),
      Node {
        flex_direction: FlexDirection::Row,
        column_gap: px(m.spacing.xs),
        width: Val::Percent(100.0),
        padding: UiRect::vertical(px(m.spacing.xs)),
        border_radius: BorderRadius::all(Val::Px(m.corner_radius)),
        ..default()
      },
      BackgroundColor(row_bg_color),
    ))
    .with_children(|r| {
      // 30% 段名
      r.spawn(Node {
        width: Val::Percent(30.0),
        ..default()
      })
      .with_children(|c| {
        label(ctx, c, tag);
      });
      // 35% gpu
      let c1 = r
        .spawn(Node {
          width: Val::Percent(35.0),
          ..default()
        })
        .with_children(|c| {
          gpu_e = label(ctx, c, "—");
        })
        .id();
      let _ = c1;
      // 35% cpu
      let c2 = r
        .spawn(Node {
          width: Val::Percent(35.0),
          ..default()
        })
        .with_children(|c| {
          cpu_e = label(ctx, c, "—");
        })
        .id();
      let _ = c2;
    })
    .id();
  (gpu_e, cpu_e, row_e)
}

/// f32 → 短格式：NAN → "—"；否则 "x.xx"；阈值 >budget → danger 色文本
fn ms_fmt(v: f32) -> String {
  if v.is_nan() {
    "—".to_string()
  } else {
    format!("{v:>5.2}")
  }
}

/// 每 0.5s 刷新 GpuTotalLabel（文本 + 变色）、PlotData push、cells 文本、events 回显
fn demo_gpu_refresh(
  timings: Res<GpuPassTimings>,
  mut prev: Local<GpuPassTimings>,
  mut all_txt: Query<(Entity, &mut Text, Option<&GpuCell>, Option<&GpuTotalLabel>)>,
  mut plot_q: Query<&mut PlotData, With<GpuPlot>>,
  mut list_q: Query<&mut RingList, With<GpuEventList>>,
  mut warned: Local<bool>, // gpu_unsupported 一次性提示
) {
  // generation 变更才刷新文本/push plot（sync 每 0.5s bump）
  if timings.generation == prev.generation && !prev.total_gpu_ms.is_nan() {
    return;
  }
  // Δ 事件必须基于 *old prev* 对比，之后再更新 prev
  let old_prev = *prev;
  *prev = *timings;

  // 1) Σ GPU 行（预算 >16.7 → danger 红；此处只改文本，颜色切换留给 TextColor 单独 system，AC-7 非强制）
  let over = timings.total_gpu_ms > SIXTY_FPS_BUDGET_MS;
  let total_body = if over {
    format!(
      "Σ GPU {} ms (超过 60FPS 预算 16.7ms)",
      ms_fmt(timings.total_gpu_ms)
    )
  } else {
    format!("Σ GPU {} ms (预算 16.7ms)", ms_fmt(timings.total_gpu_ms))
  };

  // 2) cells 值表（10 = 5×2，顺序 gc_g/gc_c/gb_g/gb_c/dc_g/dc_c/db_g/db_c/up_g/up_c）
  let values = [
    timings.gc_gpu_ms,
    timings.gc_cpu_ms,
    timings.gb_gpu_ms,
    timings.gb_cpu_ms,
    timings.dc_gpu_ms,
    timings.dc_cpu_ms,
    timings.db_gpu_ms,
    timings.db_cpu_ms,
    timings.up_gpu_ms,
    timings.up_cpu_ms,
  ];

  // 统一遍历所有 Text：通过 marker 分派，避免 2 个 &mut Text 查询（Bevy B0001）
  for (_e, mut txt, cell, total) in all_txt.iter_mut() {
    if total.is_some() {
      if txt.0 != total_body {
        txt.0 = total_body.clone();
      }
      continue;
    }
    if let Some(c) = cell {
      let idx = c.0;
      if idx >= values.len() {
        continue;
      }
      let new = ms_fmt(values[idx]);
      if txt.0 != new {
        txt.0 = new;
      }
    }
  }
  let _ = values;
  let _ = total_body;

  // 3) Plot push：total_gpu_ms（若 NAN → push 0 占位，保证 UI 仍有数据显示）
  if let Ok(mut pd) = plot_q.single_mut() {
    let v = if timings.total_gpu_ms.is_nan() {
      0.0
    } else {
      timings.total_gpu_ms
    };
    pd.push(v);
  }

  // 4) events 回显：Δ>2ms（任何 gpu 值相对 prev 差超 2ms，只记 gpu 4 行 + up cpu）
  if let Ok(mut rl) = list_q.single_mut() {
    let now_vals = [
      ("GC", timings.gc_gpu_ms),
      ("GB", timings.gb_gpu_ms),
      ("DC", timings.dc_gpu_ms),
      ("DB", timings.db_gpu_ms),
      ("UP CPU", timings.up_cpu_ms),
    ];
    let prev_vals = [
      ("GC", old_prev.gc_gpu_ms),
      ("GB", old_prev.gb_gpu_ms),
      ("DC", old_prev.dc_gpu_ms),
      ("DB", old_prev.db_gpu_ms),
      ("UP CPU", old_prev.up_cpu_ms),
    ];
    for ((tag, nv), (_, pv)) in now_vals.iter().zip(prev_vals.iter()) {
      if !nv.is_nan() && !pv.is_nan() && (nv - pv).abs() > EVENT_TICK_THRESHOLD_MS {
        let sign = if nv > pv { "▲" } else { "▼" };
        rl.push(format!(
          "{sign} {tag} Δ {:.2}ms ({:.2} → {:.2})",
          (nv - pv).abs(),
          pv,
          nv
        ));
      }
    }
    // gpu_unsupported → 一次性提示（flip 后只 push 一次，warned = true）
    if timings.gpu_unsupported && !*warned {
      *warned = true;
      rl.push("! 未检测到 GPU timestamp（elapsed_gpu 缺失），面板仅显示 CPU ms");
    }
  }
}

#[cfg(test)]
mod p27_tests {
  use super::*;
  use bevy::diagnostic::{Diagnostic, DiagnosticMeasurement};
  use gate_render::{BrickMapBuilder, BrickMapGlobals, cpu_reference_dda_ray, cpu_reference_dda_ray_aabb_skip};
  use glam::{Mat4, Vec3, Vec4};
  use gate_voxel::PaletteEntry;

  /// GpuPassTimings::default() —— 字段为 NAN 或 0，GPU unsupported = false，gen=0
  #[test]
  fn defaults_are_sane() {
    let t = GpuPassTimings::default();
    assert!(t.gc_gpu_ms.is_nan());
    assert!(t.up_gpu_ms.is_nan());
    assert_eq!(t.total_gpu_ms, 0.0);
    assert!(!t.gpu_unsupported);
    assert_eq!(t.generation, 0);
  }

  /// headless：注入 DiagnosticsStore（10 条路径各 1.0f64），运行一次 sync_gpu_timings 后
  /// GpuPassTimings 每字段均 = 1.0，total = 4.0，gpu_unsupported = false
  #[test]
  fn sync_populates_10_paths() {
    let mut app = App::new();
    // 构建 DiagnosticsStore 资源（手动，不启动 RenderDiagnosticsPlugin）
    let mut store = DiagnosticsStore::default();
    for seg in SEGMENTS {
      for field in ["elapsed_gpu", "elapsed_cpu"] {
        let path = DiagnosticPath::from_components(["render", seg.0, field]);
        store.add(Diagnostic::new(path.clone()).with_suffix("ms"));
        let now = std::time::Instant::now();
        store
          .get_mut(&path)
          .unwrap()
          .add_measurement(DiagnosticMeasurement {
            time: now,
            value: 1.0,
          });
      }
    }
    app.insert_resource(store);
    app.insert_resource(GpuPassTimings::default());
    app.insert_resource(UploadCpuSampleChannel::default());
    app.insert_resource(Time::<()>::default());
    // Time 默认 delta = 0，需要一个假的 delta 好让 sync 不阻塞 generation 增加：
    // 由于 sync 里 acc<0.5 时仍会更新字段但不 bump gen，我们直接调用系统。
    app.update();
    // 手动执行一次：系统需要 Local，通过 app.world_mut().run_system 调用：
    let world = app.world_mut();
    // 先给 Time 一个 1s 累积：用 world.resource_mut::<Time>().update_with_instant() 不存在，
    // 替代：手动 insert 一个已累积的 Local（不行）。最简单：连续 update 60 次 × 假设默认
    // delta 0 → acc 永远不增，这里只测字段填充（generation 不触发也行）
    world.run_system_cached(sync_gpu_timings).unwrap();
    world.run_system_cached(sync_gpu_timings).unwrap(); // 第二次触发 update cycle
    let t = world.resource::<GpuPassTimings>();
    // 字段填充
    for (i, (gpu, cpu)) in [
      (t.gc_gpu_ms, t.gc_cpu_ms),
      (t.gb_gpu_ms, t.gb_cpu_ms),
      (t.dc_gpu_ms, t.dc_cpu_ms),
      (t.db_gpu_ms, t.db_cpu_ms),
    ]
    .iter()
    .enumerate()
    {
      assert!((gpu - 1.0).abs() < 1e-3, "seg {i} gpu mismatch {gpu}");
      assert!((cpu - 1.0).abs() < 1e-3, "seg {i} cpu mismatch {cpu}");
    }
    // 上传段 gpu：DiagnosticsStore 里有 elapsed_gpu = 1.0，但若 sync 代码正确也会读
    assert!(!t.up_gpu_ms.is_nan());
    assert!(
      (t.total_gpu_ms - 4.0).abs() < 1e-3,
      "total {} != 4.0",
      t.total_gpu_ms
    );
  }

  /// 缺失 DiagnosticsStore（或路径空）→ 字段保持 NAN 不崩溃；gpu_unsupported = false（因 no cpu 值）
  #[test]
  fn missing_paths_keeps_nan_no_panic() {
    let mut app = App::new();
    app.insert_resource(DiagnosticsStore::default());
    app.insert_resource(GpuPassTimings::default());
    app.insert_resource(UploadCpuSampleChannel::default());
    app.insert_resource(Time::<()>::default());
    let world = app.world_mut();
    world.run_system_cached(sync_gpu_timings).unwrap();
    let t = world.resource::<GpuPassTimings>();
    // 缺失 → NAN
    assert!(t.gc_gpu_ms.is_nan());
    assert!(t.dc_cpu_ms.is_nan());
    // cpu 与 gpu 都 NAN → gpu_unsupported 不置 true
    assert!(!t.gpu_unsupported);
  }

  /// elapsed_cpu 全部有数（4 项）但 gpu 全 NAN → gpu_unsupported = true（即 GPU timestamp 不支持）
  #[test]
  fn all_gpu_nan_with_cpu_marks_gpu_unsupported() {
    let mut app = App::new();
    let mut store = DiagnosticsStore::default();
    for seg in SEGMENTS {
      // 只写 elapsed_cpu，gpu 路径不写 → 读 NAN
      let path = DiagnosticPath::from_components(["render", seg.0, "elapsed_cpu"]);
      store.add(Diagnostic::new(path.clone()).with_suffix("ms"));
      let now = std::time::Instant::now();
      store
        .get_mut(&path)
        .unwrap()
        .add_measurement(DiagnosticMeasurement {
          time: now,
          value: 0.5,
        });
    }
    app.insert_resource(store);
    app.insert_resource(GpuPassTimings::default());
    app.insert_resource(UploadCpuSampleChannel::default());
    app.insert_resource(Time::<()>::default());
    let world = app.world_mut();
    world.run_system_cached(sync_gpu_timings).unwrap();
    let t = world.resource::<GpuPassTimings>();
    assert!(t.gpu_unsupported, "gpu_unsupported should flip");
    // total = 0（gpu 全 NAN）
    assert_eq!(t.total_gpu_ms, 0.0);
  }

  // ---------------- Task 3：GPU 面板 UI 相关 headless 测试 ----------------

  /// 注入 theme/font 资源 → 调用 setup 两次（第二次 done，不 panic），
  /// 验证 headless 世界至少有 1×GpuTotalLabel、1×GpuPlot、5×2 GpuCell、1×GpuEventList、1×DcRowBg
  #[test]
  fn layout_has_all_10_cells_and_markers() {
    use gate_ui::theme::default_theme;
    let mut app = App::new();
    // GateUiPlugin 太重（含 AssetServer 等），手动注入 theme+font 资源 + 资产
    app.init_resource::<Assets<Image>>();
    app.insert_resource(default_theme());
    app.insert_resource(ThemeFont {
      path: None,
      handle: None,
    });
    app.insert_resource(GpuPassTimings::default());
    // demo_gpu_panel_setup requires Commands + Assets<Image>; run once via run_system_cached
    let w = app.world_mut();
    w.run_system_cached(demo_gpu_panel_setup).unwrap();
    // 再跑一次 → done=true，应直接 return（不 panic）
    let w2 = app.world_mut();
    w2.run_system_cached(demo_gpu_panel_setup).unwrap();
    let world = app.world_mut();
    // 1×Σ 行标签
    assert_eq!(
      world
        .query_filtered::<(), With<GpuTotalLabel>>()
        .iter(world)
        .count(),
      1
    );
    // 1×Plot
    assert_eq!(
      world
        .query_filtered::<(), With<GpuPlot>>()
        .iter(world)
        .count(),
      1
    );
    // 10×GpuCell（gc,g/c | gb,g/c | dc,g/c | db,g/c | up,g/c）
    assert_eq!(
      world
        .query_filtered::<(), With<GpuCell>>()
        .iter(world)
        .count(),
      10
    );
    // 1×event RingList
    assert_eq!(
      world
        .query_filtered::<(), With<GpuEventList>>()
        .iter(world)
        .count(),
      1
    );
    // 1×DC row 背景色容器
    assert_eq!(
      world
        .query_filtered::<(), With<DcRowBg>>()
        .iter(world)
        .count(),
      1
    );
  }

  /// DC row (DcRowBg) 的 BackgroundColor 应有非 NONE 的 accent_hover 色（spec AC-6）
  #[test]
  fn dc_row_has_accent_background() {
    use gate_ui::theme::default_theme;
    let mut app = App::new();
    app.init_resource::<Assets<Image>>();
    app.insert_resource(default_theme());
    app.insert_resource(ThemeFont {
      path: None,
      handle: None,
    });
    app.insert_resource(GpuPassTimings::default());
    app
      .world_mut()
      .run_system_cached(demo_gpu_panel_setup)
      .unwrap();
    let world = app.world_mut();
    // DcRowBg entity 查 BackgroundColor.0
    let Ok((e, bg)) = world
      .query_filtered::<(Entity, &BackgroundColor), With<DcRowBg>>()
      .single(world)
    else {
      panic!("no DcRowBg");
    };
    assert!(e != Entity::PLACEHOLDER);
    let srgba = bg.0.to_srgba();
    assert!(
      srgba.alpha > 0.0,
      "DC row should have non-zero alpha accent bg (srgba.alpha={})",
      srgba.alpha
    );
  }

  /// timings.total_gpu_ms=20.0 >16.7 → refresh 后 Σ 行文本含 "超过 60FPS 预算"
  #[test]
  fn total_over_budget_shows_warning_text() {
    use gate_ui::theme::default_theme;
    let mut app = App::new();
    app.init_resource::<Assets<Image>>();
    app.insert_resource(default_theme());
    app.insert_resource(ThemeFont {
      path: None,
      handle: None,
    });
    let mut timings = GpuPassTimings::default();
    timings.generation = 1;
    timings.total_gpu_ms = 20.0; // 超预算
    app.insert_resource(timings);
    // spawn 面板 + Σ 行 label（手动，省跑 setup 整套）
    let total_e = app
      .world_mut()
      .spawn((Text::new("Σ GPU —".to_string()), GpuTotalLabel))
      .id();
    // Plot + cells + RingList 全部手动 spawn（refresh 用 marker 查询）
    app.world_mut().spawn((
      PlotData::new(128, PlotDomain::Fixed(0.0, 20.0), Color::WHITE, false),
      GpuPlot,
    ));
    for i in 0..10 {
      app
        .world_mut()
        .spawn((Text::new("—".to_string()), GpuCell(i)));
    }
    app.world_mut().spawn((RingList::new(6), GpuEventList));
    app.world_mut().run_system_cached(demo_gpu_refresh).unwrap();
    let txt = app.world().get::<Text>(total_e).expect("total label text");
    assert!(
      txt.0.contains("超过 60FPS 预算"),
      "Σ text should warn, got: {}",
      txt.0
    );
  }

  /// total_gpu_ms=8.0（<16.7）→ Σ 行不含 "超过"；PlotData len +1；Δ>2ms 触发 RingList 追加
  #[test]
  fn under_budget_and_plot_push_and_delta_event() {
    use gate_ui::theme::default_theme;
    let mut app = App::new();
    app.init_resource::<Assets<Image>>();
    app.insert_resource(default_theme());
    app.insert_resource(ThemeFont {
      path: None,
      handle: None,
    });
    let mut timings = GpuPassTimings::default();
    timings.generation = 1;
    timings.total_gpu_ms = 8.0;
    timings.dc_gpu_ms = 0.8; // 下次 bump 到 5.8，Δ=5.0 > 2ms → 触发
    timings.dc_cpu_ms = 0.5;
    timings.gc_gpu_ms = 0.8;
    app.insert_resource(timings);
    app
      .world_mut()
      .spawn((Text::new("Σ GPU —".to_string()), GpuTotalLabel));
    let plot_e = app
      .world_mut()
      .spawn((
        PlotData::new(128, PlotDomain::Fixed(0.0, 20.0), Color::WHITE, false),
        GpuPlot,
      ))
      .id();
    for i in 0..10 {
      app
        .world_mut()
        .spawn((Text::new("—".to_string()), GpuCell(i)));
    }
    let rl_e = app.world_mut().spawn((RingList::new(6), GpuEventList)).id();
    app.world_mut().run_system_cached(demo_gpu_refresh).unwrap();
    // 第二次调用：prev=默认（第一次的值写入 Local）。变更 timings：dc_gpu 从 0.8 → 5.8 （Δ=5.0>2ms）
    {
      let mut t = app.world_mut().resource_mut::<GpuPassTimings>();
      t.generation = 2;
      t.dc_gpu_ms = 5.8; // Δ = 5.8 - 0.8 = 5.0
    }
    app.world_mut().run_system_cached(demo_gpu_refresh).unwrap();
    // 1) Plot push 2 samples（2 gen changes +2）
    let pd = app.world().get::<PlotData>(plot_e).expect("plot");
    assert!(
      pd.len() >= 2,
      "PlotData should have ≥2 samples, got {}",
      pd.len()
    );
    // 2) Σ 行文本不含 "超过"
    let w = app.world_mut();
    let Ok(total_txt) = w.query_filtered::<&Text, With<GpuTotalLabel>>().single(&w) else {
      panic!("no total label");
    };
    let total_txt_str = total_txt.0.clone();
    assert!(
      !total_txt_str.contains("超过"),
      "Σ text should NOT warn for under budget: {}",
      total_txt_str
    );
    // 3) DC Δ=5.0 触发 RingList 追加条目
    let rl = app.world().get::<RingList>(rl_e).expect("ringlist");
    let items: Vec<_> = rl.items().collect();
    assert!(
      items.iter().any(|s| s.contains("DC") && s.contains("Δ")),
      "ringlist should have Δ event, got: {:?}",
      items
    );
  }

  // ========================= 新增：AABB 滚远消失问题 headless 复现 =========================
  //
  // 用真实 demo scene（build_demo_scene）+ 轨道相机 zoom-out（0 / 1 / 2 / 5 行滚轮）
  // 在 32x32 视锥网格上跑 CPU DDA：full 2M 步 vs AABB-skip 2048 步。
  // 若任一场景 diff_hit+diff_pal > 0 → WGSL shader 必然也有完全一样的 bug，
  // 因为 WGSL AABB 代码就是 `cpu_reference_dda_ray_aabb_skip` 的逐字翻译。
  //
  // 运行：cargo test -p gate-app demo_scene_aabb_zoom_out -- --nocapture

  fn scene_for_aabb_zoom_headless() -> (gate_voxel::TileGrid, BrickMapGlobals) {
    let mut grid = gate_voxel::TileGrid::new();
    paint(&mut grid);
    build_scene(&mut grid);
    let builder = BrickMapBuilder::build_full(&grid);
    (grid, builder.buffers().globals)
  }

  // 从 main.rs 的 paint_demo_palette / build_demo_scene 重命名 import（同名）——
  // 在 cfg(test) 里调用同名函数会优先选 super::，但函数在 test mod 外。
  // 所以我们直接再包一层：
  fn paint(grid: &mut gate_voxel::TileGrid) {
    let palette: &[(u8, [u8; 3], u8)] = &[
      (1, [118, 118, 126], 220),
      (2, [214, 64, 64], 180),
      (3, [72, 196, 96], 200),
      (4, [72, 120, 224], 180),
      (5, [240, 204, 64], 160),
      (6, [64, 216, 216], 160),
    ];
    let pal = grid.palette_mut();
    for &(idx, color, rough) in palette {
      let mut e = PaletteEntry::default();
      e.color = color;
      e.roughness = rough;
      pal.set(idx, e);
    }
  }
  fn build_scene(grid: &mut gate_voxel::TileGrid) {
    use gate_voxel::{draw_text, fill_box, fill_sphere};
    fill_box(grid, glam::IVec3::ZERO, glam::IVec3::new(512, 16, 512), 0, 1);
    for i in (0..512).step_by(128) {
      fill_box(grid, glam::IVec3::new(i, 16, 0), glam::IVec3::new(4, 4, 512), 3, 6);
      fill_box(grid, glam::IVec3::new(0, 16, i), glam::IVec3::new(512, 4, 4), 3, 6);
    }
    fill_box(grid, glam::IVec3::new(128, 16, 128), glam::IVec3::new(128, 256, 128), 1, 2);
    fill_sphere(grid, glam::IVec3::new(192, 320, 192), 64, 2, 5);
    fill_box(grid, glam::IVec3::new(256, 192, 144), glam::IVec3::new(96, 4, 32), 3, 2);
    fill_box(grid, glam::IVec3::new(352, 16, 128), glam::IVec3::new(64, 192, 64), 2, 6);
    fill_sphere(grid, glam::IVec3::new(384, 80, 320), 48, 3, 2);
    fill_sphere(grid, glam::IVec3::new(192, 48, 352), 24, 4, 4);
    draw_text(grid, glam::IVec3::new(64, 16, 448), "GATE", 1, 5);
    fill_box(grid, glam::IVec3::new(656, 64, 64), glam::IVec3::new(32, 32, 32), 4, 5);
    fill_box(grid, glam::IVec3::new(1264, 240, 240), glam::IVec3::new(32, 32, 32), 4, 4);
  }

  #[test]
  fn demo_scene_aabb_zoom_out_headless() {
    let (grid, g) = scene_for_aabb_zoom_headless();
    let builder = BrickMapBuilder::build_full(&grid);
    let buffers = builder.buffers().clone();
    // 与实际 setup 中 orbit eye/target 一致：eye(700,560,700) target(260,120,260)
    let base_eye = Vec3::new(700.0, 560.0, 700.0);
    let target = Vec3::new(260.0, 120.0, 260.0);
    let aabb_min = Vec3::new(
      g.index_origin_x as f32 * 512.0,
      g.index_origin_y as f32 * 512.0,
      g.index_origin_z as f32 * 512.0,
    );
    let aabb_max = Vec3::new(
      (g.index_origin_x + g.index_dims_x as i32) as f32 * 512.0,
      (g.index_origin_y + g.index_dims_y as i32) as f32 * 512.0,
      (g.index_origin_z + g.index_dims_z as i32) as f32 * 512.0,
    );
    // 初始 distance (700-260,560-120,700-260).len() ≈ 866 fine
    let base_offset = base_eye - target;
    let base_dist = base_offset.length();
    let base_dir = base_offset / base_dist;
    // 每个 "滚轮 -1 行"（往外滚）: distance *= exp(0.35) ≈ 1.419
    let zoom_factor_for_lines = |lines: i32| (lines as f32 * ZOOM_LOG_SPEED).exp();
    // 测试：0（初始）/ +2（两下滚出 distance ×≈2）/ +5（多滚几下 distance ×≈5.8）/ +10（×28，超远）/ +20（×815）
    for lines in [0i32, 2, 5, 10, 20] {
      let factor = zoom_factor_for_lines(lines);
      let distance = base_dist * factor;
      let eye = target + base_dir * distance;
      let aspect = 16.0 / 9.0;
      let proj = Mat4::perspective_rh(FOV_Y, aspect, CAM_NEAR, CAM_FAR);
      let view = Mat4::look_at_rh(eye, target, Vec3::Y);
      let vp = proj * view;
      let inv_vp = vp.inverse();
      let grid_w = 32usize;
      let mut full_hits = 0usize;
      let mut skip_hits = 0usize;
      let mut diff_hit = 0usize;
      let mut diff_pal = 0usize;
      for iy in 0..grid_w {
        for ix in 0..grid_w {
          let u = (ix as f32 + 0.5) / (grid_w as f32) * 2.0 - 1.0;
          let v = 1.0 - ((iy as f32 + 0.5) / (grid_w as f32)) * 2.0;
          let near = inv_vp * Vec4::new(u, v, 0.0, 1.0);
          let far  = inv_vp * Vec4::new(u, v, 1.0, 1.0);
          let near = near.truncate() / near.w;
          let far  = far.truncate() / far.w;
          let diff = far - near;
          let frustum_len = diff.length();
          let dir = diff.normalize();
          // 极限场景 AABB 尺寸：12×5×12 tile = (5632, 2560, 5632) fine，
          // 对角穿越 worst-case ~9000 fine；给 16384 留 2× 余量避免截断漏画。
          let aabb_max_steps: u32 = 16384;
          let r_full = cpu_reference_dda_ray(&buffers, eye, dir, frustum_len, 2_000_000);
          let r_skip = cpu_reference_dda_ray_aabb_skip(&buffers, eye, dir, frustum_len, aabb_max_steps, aabb_min, aabb_max);
          if r_full.is_some() { full_hits += 1; }
          if r_skip.is_some() { skip_hits += 1; }
          match (r_full, r_skip) {
            (Some((_, p1)), Some((_, p2))) if p1 != p2 => diff_pal += 1,
            (Some(_), None) | (None, Some(_)) => diff_hit += 1,
            _ => {}
          }
        }
      }
      println!(
        "ZOOM lines={lines:>3} distance={distance:>8.0} AABB fine min=({:.0},{:.0},{:.0}) max=({:.0},{:.0},{:.0})\n  \
         32x32 full_hits={full_hits}/{} skip_hits={skip_hits}/{} diff_hit={diff_hit} diff_pal={diff_pal}",
        aabb_min.x, aabb_min.y, aabb_min.z, aabb_max.x, aabb_max.y, aabb_max.z,
        grid_w*grid_w, grid_w*grid_w,
      );
      // 硬断言：Rust AABB skip 与 full DDA 在真实 demo scene + 真实 zoom-out 操作下必须逐像素一致
      assert!(
        diff_hit + diff_pal == 0,
        "zoom lines={lines} distance={distance}: full/skip MISMATCH {}/{} (hit+pal)",
        diff_hit + diff_pal, grid_w*grid_w
      );
    }
  }
}
