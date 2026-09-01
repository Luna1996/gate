use std::collections::VecDeque;

use bevy::{
    asset::{AssetPlugin, LoadState},
    image::Image,
    input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll, MouseScrollUnit},
    log::LogPlugin,
    prelude::*,
    window::{PresentMode, Window},
};
use glam::{Mat3, Vec3};

use gate_render::{
    BrickMapBuffers, BrickMapBuilder, DdaCameraConfig, DdaImages, MovObject, MovScene, OrbitCamera,
    UploadBudget, VIEW_SIZE, VoxelScene, create_dda_image, pack_mov_pool,
};
use gate_ui::{
    ThemeFont, UiCtx, UiTheme,
    widgets::{label, px},
};
use gate_voxel::{PaletteEntry, draw_text, fill_box, fill_sphere};

/// 以 crate 目录为锚的 assets 路径，F5 / 终端启动行为一致
pub const ASSETS_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/assets");

/// 程序侧日志落盘（方案 3）：LogPlugin custom_layer 追加无色文件层，
/// 每次启动截断重写（永远最新一轮）；默认 stderr 彩色层保留不变
pub const LOG_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/logs/latest.log");

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
                    // 日志落盘层（方案 3）：无 ANSI 色码，写 logs/latest.log；
                    // 默认 stderr 彩色层不受影响，终端 / LLDB log 照常有色输出
                    custom_layer: |_app| {
                        use bevy::log::BoxedLayer;
                        let path = std::path::Path::new(LOG_PATH);
                        std::fs::create_dir_all(path.parent().expect("LOG_PATH 必有父目录"))
                            .ok()?;
                        let file = std::fs::File::create(path).ok()?;
                        let (writer, guard) = tracing_appender::non_blocking(file);
                        // WorkerGuard 须活到进程退出；App 构建处无处安放，直接泄漏——
                        // worker 线程满缓冲 / 每 10ms 即落盘，进程退出由 OS 收尾，尾部丢失可忽略
                        std::mem::forget(guard);
                        // 本地时区时间戳（默认 SystemTime timer 是 UTC，+8 区看日志像"慢 8 小时"；
                        // Windows 下 current_local_offset 安全，此处仍在 main 早期单线程阶段，失败回退 UTC）
                        let offset =
                            time::UtcOffset::current_local_offset().unwrap_or(time::UtcOffset::UTC);
                        let timer = tracing_subscriber::fmt::time::OffsetTime::new(
                            offset,
                            time::format_description::well_known::Rfc3339,
                        );
                        Some(Box::new(
                            tracing_subscriber::fmt::layer()
                                .with_ansi(false)
                                .with_timer(timer)
                                .with_writer(writer),
                        ) as BoxedLayer)
                    },
                    ..default()
                }),
        )
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
                fps_line_feed,
            ),
        )
        .run();
}

fn setup(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    let dda_handle = create_dda_image(&mut images);
    commands.spawn((Camera2d, Msaa::Off));
    // ---- P3.1 光照主题：「暗色实验室」RON 加载（一次性静态配置，同步读足够；
    // 缺失/解析失败回退内置默认主题）----
    let theme = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/assets/lighting/dark_lab.ron"
    ))
    .map_err(|e| format!("read: {e}"))
    .and_then(|s| gate_render::parse_lighting_ron(&s).map_err(|e| format!("ron: {e}")))
    .unwrap_or_else(|e| {
        bevy::log::warn!("lighting/dark_lab.ron 加载失败（{e}），回退内置默认主题");
        Default::default()
    });
    commands.insert_resource(theme);
    commands.insert_resource(DdaImages { target: dda_handle });
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

    // ---- P2.10 MOV 硬编码验收场景：芯片预制件 ×3（遮挡/贴地/交叠/旋转/缩放）----
    let chip = build_chip_prefab();
    let chips = [
        // A 贴地：城堡平台顶面（y=272）上，identity / scale 1
        MovObject {
            buffers: &chip,
            pos: Vec3::new(2352.0, 272.0, 2352.0),
            rot: Mat3::IDENTITY,
            scale: 1.0,
        },
        // B 交叠：嵌入北城墙（z 2048..2056）——突出部遮挡墙 / 嵌入部被墙遮挡
        MovObject {
            buffers: &chip,
            pos: Vec3::new(2600.0, 260.0, 1980.0),
            rot: Mat3::IDENTITY,
            scale: 1.0,
        },
        // C 旋转 + 缩放：地面 yaw30° / scale 2，压中央大道
        MovObject {
            buffers: &chip,
            pos: Vec3::new(1600.0, 16.0, 2600.0),
            rot: Mat3::from_rotation_y(30.0_f32.to_radians()),
            scale: 2.0,
        },
    ];
    commands.insert_resource(MovScene {
        packed: std::sync::Arc::new(pack_mov_pool(&chips)),
        version: 1,
    });
    gate_ui::world_anchor_label(
        &mut commands,
        "P2.10 芯片·MOV",
        Vec3::new(2416.0, 380.0, 2416.0),
        Color::srgb_u8(68, 230, 220),
    );

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
    let t = z as f32 / EXT_FINE_Z as f32; // 0..1
    let river_center = EXT_FINE_HALF as f32   // 中线 2560
    + (t * std::f32::consts::TAU).sin() * 700.0 // 正弦摆 ±700
    + ((t * 12.566).cos() * 180.0); // 次级摆幅
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
    let top_r = 560i32; // 顶面岛半径
    let mut y = island_base_y;
    while y < island_top_y {
        let t = (y - island_base_y) as f32 / (island_top_y - island_base_y) as f32;
        let r = (t * t.sqrt() * top_r as f32) as i32 + 16;
        fill_sphere(grid, IVec3::new(cx, y, cz), r, 1, 13);
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
    for &(ox, oz) in &[
        (-512, -512),
        (512 - 96, -512),
        (-512, 512 - 96),
        (512 - 96, 512 - 96),
    ] {
        let tx = cx + ox;
        let tz = cz + oz;
        fill_box(grid, IVec3::new(tx, 272, tz), IVec3::new(96, 160, 96), 1, 4);
        // 角楼顶金色 16
        fill_box(
            grid,
            IVec3::new(tx, 272 + 160, tz),
            IVec3::new(96, 16, 96),
            1,
            11,
        );
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
    fill_box(
        grid,
        IVec3::new(cx - 48, 464, cz - 48),
        IVec3::new(96, 256, 96),
        1,
        4,
    );
    // 塔顶平台 128×128×16
    fill_box(
        grid,
        IVec3::new(cx - 64, 720, cz - 64),
        IVec3::new(128, 16, 128),
        1,
        11,
    );
    // 金顶球（L2 r=64，塔顶 y=720+80=800）
    fill_sphere(grid, IVec3::new(cx, 800, cz), 64, 2, 11);
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
        && !(2496..=2528).contains(&z) // 中央大道
        && h < 360;
            if ok {
                // 树干 （L1, 24 宽 16 宽 24 深 高 80）
                fill_box(grid, IVec3::new(x, h, z), IVec3::new(32, 80, 32), 1, 6);
                // 树叶球（L2，r=80，中心在树干顶 + 80）
                fill_sphere(grid, IVec3::new(x + 16, h + 80 + 64, z + 16), 80, 2, 5);
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
        (480, 480, 90, 8, 9, 131),   // 左上（near 角雪峰脚）水晶青/紫
        (4608, 640, 80, 10, 8, 251), // 右上 水晶红/青（seed<255 安全）
        (768, 4352, 70, 9, 10, 223), // 左下 水晶紫/红
    ];
    for &(cx_, cz_, cnt, pa, pb, seed) in crystal_clusters.iter() {
        let h = terrain_h(cx_, cz_);
        let mut t = 0i32;
        let mut placed = 0usize;
        while placed < cnt as usize {
            // 确定性位置抖动
            let sx = ((t * 37 + seed) % 96) - 48;
            let sz = ((t * 131 + seed * 3) % 96) - 48;
            let hy = (t * 53) % 14; // 高度 0..13 fine
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

// ================= P2.10 MOV 硬编码芯片预制件 =================
//
// 电子学级小网格（128×32×128 fine，独立 1-tile TileGrid，全链复用
// TileGrid/BrickMapBuilder）。含 L0 PCB / 引脚 / die + L4 走线（1-fine
// 细线，练习 brick slab 路径）。三枚实例见 setup()：贴地 / 交叠 / 旋转缩放。

fn build_chip_prefab() -> BrickMapBuffers {
    let mut g = gate_voxel::TileGrid::new();
    // 芯片自带 palette（物体独立 256 条；1 PCB 绿 / 2 走线青 / 3 引脚金 / 4 die 银灰）
    let pal = g.palette_mut();
    let chip_colors: &[(u8, [u8; 3], u8)] = &[
        (1, [26, 84, 52], 200),
        (2, [68, 230, 220], 150),
        (3, [248, 210, 72], 160),
        (4, [188, 196, 208], 170),
    ];
    for &(idx, color, rough) in chip_colors {
        let mut e = PaletteEntry::default();
        e.color = color;
        e.roughness = rough;
        pal.set(idx, e);
    }
    // PCB 基板（L0）+ 中央 die + 四边引脚
    fill_box(&mut g, IVec3::ZERO, IVec3::new(128, 16, 128), 0, 1);
    fill_box(&mut g, IVec3::new(48, 16, 48), IVec3::new(32, 16, 32), 0, 4);
    for s in 0..4 {
        fill_box(
            &mut g,
            IVec3::new(8 + s * 32, 16, 0),
            IVec3::new(16, 8, 8),
            0,
            3,
        );
        fill_box(
            &mut g,
            IVec3::new(8 + s * 32, 16, 120),
            IVec3::new(16, 8, 8),
            0,
            3,
        );
        fill_box(
            &mut g,
            IVec3::new(0, 16, 8 + s * 32),
            IVec3::new(8, 8, 16),
            0,
            3,
        );
        fill_box(
            &mut g,
            IVec3::new(120, 16, 8 + s * 32),
            IVec3::new(8, 8, 16),
            0,
            3,
        );
    }
    // L4 走线（1-fine 细线，PCB 上表面 y=32）
    for i in 0..7 {
        fill_box(
            &mut g,
            IVec3::new(16 + i * 16, 32, 16),
            IVec3::new(1, 1, 96),
            4,
            2,
        );
        fill_box(
            &mut g,
            IVec3::new(16, 32, 16 + i * 16),
            IVec3::new(96, 1, 1),
            4,
            2,
        );
    }
    BrickMapBuilder::build_full(&g).buffers().clone()
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

// ================= 左上角单行 FPS（CUR/AVG/MIN/MAX） =================
//
// 统计口径：滚动窗口 = 最近 5s 的逐帧 delta_secs：
// - CUR = 本刷新周期（0.25s）帧数 / 实际耗时
// - AVG = 窗口帧数 / 窗口总时长
// - MIN = 1 / 窗口最大帧耗时（最差一帧）
// - MAX = 1 / 窗口最小帧耗时（最好一帧）
// 数字固定 3 位宽（上限 999）→ 文本定长，UI 不抖动。

const FPS_REFRESH_SECS: f32 = 0.25;
const FPS_WINDOW_SECS: f32 = 5.0;

#[derive(Component)]
struct FpsText;

/// fps → 3 位宽显示值（上限 999，防 4 位数抖动）
fn fps3(v: f32) -> u32 {
    (v.round() as u32).min(999)
}

/// 主题就绪后 spawn 一次（RON 成功或回退默认都会插入 UiTheme 资源）
///
/// 等待条件：主题字体资产已加载为 `LoadState::Loaded`。若 theme.font_path = None
/// 则用 SystemUi fallback（不等待，直接 spawn）。必须等字体到位再 spawn UI 标签，
/// 否则 TextPipeline 在 FiraMono-subset（Bevy 内置 default slot，不含 CJK）上生成
/// 字形缓存，即使之后字体 override default slot，已缓存的 atlas 条目仍回退 CJK
/// 字形为方框（parley 的字形请求不因为 Assets slot 变化而自动失效）。
/// FPS 行纯 ASCII，但统一遵守字体就绪判定，避免以后换字符串时又出方框。
fn demo_ui_setup(
    mut done: Local<bool>,
    theme: Option<Res<UiTheme>>,
    font: Option<Res<ThemeFont>>,
    server: Option<Res<AssetServer>>,
    mut commands: Commands,
) {
    if *done {
        return;
    }
    let Some(theme) = theme else {
        return;
    };
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

    commands.queue(move |world: &mut World| {
        let ctx = UiCtx::new(&theme, font.as_ref());
        world
            .spawn(Node {
                position_type: PositionType::Absolute,
                left: px(8.0),
                top: px(8.0),
                ..default()
            })
            .with_children(|root| {
                let e = label(&ctx, root, "FPS: CUR ---, AVG ---, MIN ---, MAX ---");
                root.world_mut().entity_mut(e).insert(FpsText);
            });
    });
}

/// 每 0.25s 刷新一次左上角 FPS 行（文本只在变化时写入）
fn fps_line_feed(
    time: Res<Time>,
    mut q: Query<&mut Text, With<FpsText>>,
    mut window: Local<VecDeque<f32>>, // 逐帧 delta，按时间裁剪到 5s
    mut acc: Local<f32>,
    mut frames: Local<u32>,
) {
    let dt = time.delta_secs();
    window.push_back(dt);
    let mut sum = 0.0f32;
    for &d in window.iter() {
        sum += d;
    }
    while sum > FPS_WINDOW_SECS
        && let Some(old) = window.pop_front()
    {
        sum -= old;
    }
    *acc += dt;
    *frames += 1;
    if *acc < FPS_REFRESH_SECS {
        return;
    }
    let mut max_dt = 0.0f32;
    let mut min_dt = f32::MAX;
    for &d in window.iter() {
        max_dt = max_dt.max(d);
        min_dt = min_dt.min(d);
    }
    let cur = *frames as f32 / *acc;
    let avg = if sum > 0.0 {
        window.len() as f32 / sum
    } else {
        0.0
    };
    let min = if max_dt > 0.0 { 1.0 / max_dt } else { 0.0 };
    let max = if min_dt < f32::MAX && min_dt > 0.0 {
        1.0 / min_dt
    } else {
        0.0
    };
    let txt = format!(
        "FPS: CUR {:>3}, AVG {:>3}, MIN {:>3}, MAX {:>3}",
        fps3(cur),
        fps3(avg),
        fps3(min),
        fps3(max)
    );
    *acc = 0.0;
    *frames = 0;
    if let Ok(mut t) = q.single_mut()
        && t.0 != txt
    {
        t.0 = txt;
    }
}

#[cfg(test)]
mod aabb_zoom_tests {
    use super::*;
    use gate_render::{
        BrickMapBuilder, BrickMapGlobals, cpu_reference_dda_ray, cpu_reference_dda_ray_aabb_skip,
    };
    use gate_voxel::PaletteEntry;
    use glam::{Mat4, Vec3, Vec4};

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
        fill_box(
            grid,
            glam::IVec3::ZERO,
            glam::IVec3::new(512, 16, 512),
            0,
            1,
        );
        for i in (0..512).step_by(128) {
            fill_box(
                grid,
                glam::IVec3::new(i, 16, 0),
                glam::IVec3::new(4, 4, 512),
                3,
                6,
            );
            fill_box(
                grid,
                glam::IVec3::new(0, 16, i),
                glam::IVec3::new(512, 4, 4),
                3,
                6,
            );
        }
        fill_box(
            grid,
            glam::IVec3::new(128, 16, 128),
            glam::IVec3::new(128, 256, 128),
            1,
            2,
        );
        fill_sphere(grid, glam::IVec3::new(192, 320, 192), 64, 2, 5);
        fill_box(
            grid,
            glam::IVec3::new(256, 192, 144),
            glam::IVec3::new(96, 4, 32),
            3,
            2,
        );
        fill_box(
            grid,
            glam::IVec3::new(352, 16, 128),
            glam::IVec3::new(64, 192, 64),
            2,
            6,
        );
        fill_sphere(grid, glam::IVec3::new(384, 80, 320), 48, 3, 2);
        fill_sphere(grid, glam::IVec3::new(192, 48, 352), 24, 4, 4);
        draw_text(grid, glam::IVec3::new(64, 16, 448), "GATE", 1, 5);
        fill_box(
            grid,
            glam::IVec3::new(656, 64, 64),
            glam::IVec3::new(32, 32, 32),
            4,
            5,
        );
        fill_box(
            grid,
            glam::IVec3::new(1264, 240, 240),
            glam::IVec3::new(32, 32, 32),
            4,
            4,
        );
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
                    let far = inv_vp * Vec4::new(u, v, 1.0, 1.0);
                    let near = near.truncate() / near.w;
                    let far = far.truncate() / far.w;
                    let diff = far - near;
                    let frustum_len = diff.length();
                    let dir = diff.normalize();
                    // 极限场景 AABB 尺寸：12×5×12 tile = (5632, 2560, 5632) fine，
                    // 对角穿越 worst-case ~9000 fine；给 16384 留 2× 余量避免截断漏画。
                    let aabb_max_steps: u32 = 16384;
                    let r_full = cpu_reference_dda_ray(&buffers, eye, dir, frustum_len, 2_000_000);
                    let r_skip = cpu_reference_dda_ray_aabb_skip(
                        &buffers,
                        eye,
                        dir,
                        frustum_len,
                        aabb_max_steps,
                        aabb_min,
                        aabb_max,
                    );
                    if r_full.is_some() {
                        full_hits += 1;
                    }
                    if r_skip.is_some() {
                        skip_hits += 1;
                    }
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
                aabb_min.x,
                aabb_min.y,
                aabb_min.z,
                aabb_max.x,
                aabb_max.y,
                aabb_max.z,
                grid_w * grid_w,
                grid_w * grid_w,
            );
            // 硬断言：Rust AABB skip 与 full DDA 在真实 demo scene + 真实 zoom-out 操作下必须逐像素一致
            assert!(
                diff_hit + diff_pal == 0,
                "zoom lines={lines} distance={distance}: full/skip MISMATCH {}/{} (hit+pal)",
                diff_hit + diff_pal,
                grid_w * grid_w
            );
        }
    }
}
