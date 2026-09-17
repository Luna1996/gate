//! gate-app 的可调旋钮（本 crate 的模块都在 `src/` 下，故集中在本文件）：改这里即改默认值。
//!
//! 全是编译期常量：改值后重新编译。要给某项加 debug_menu 控件时，把它挪进 Bevy 资源再按
//! 菜单连线（做法见 `debug_menu.rs` 里的 `RenderScale` / `EyeAdaptSettings` / `DdgiDebugSettings`）。

/// 启动场景：true = 程序化 demo 场景；false = assets/vox/nuke.vox
pub const STARTUP_DEMO_SCENE: bool = false;
/// 启动相机朝天空（纯 miss 基准）
pub const START_CAMERA_SKY: bool = false;
/// demo 场景边长（tile 数，1 tile = 512 voxel）
pub const DEMO_TILES: i32 = 2;
/// 失焦窗口仍按 Continuous 跑帧（窗口启动即 focused=false）
pub const BENCH_UNFOCUSED: bool = false;
/// 相机自动绕目标旋转并平移（配 BENCH_UNFOCUSED 读移动中的逐 pass 帧）
pub const AUTO_ORBIT: bool = false;
/// 第 60 帧自动刷一次笔触（无鼠标走通编辑 → 增量上传链路）
pub const EDIT_SELFTEST: bool = false;

/// 垂直视场角（弧度）
pub const FOV_Y: f32 = 60.0_f32.to_radians();
/// 近裁剪面（voxel）
pub const CAM_NEAR: f32 = 1.0;
/// 远裁剪面（voxel）
pub const CAM_FAR: f32 = 65536.0;
/// 右键拖拽转向灵敏度（弧度/像素）
pub const ROT_SPEED: f32 = 0.005;
/// 滚轮缩放速率（对数档/格）
pub const ZOOM_LOG_SPEED: f32 = 0.35;
/// 飞行相机默认速度（voxel/s）
pub const FLY_SPEED_DEFAULT: f32 = 128.0;
/// 飞行相机加速档倍率
pub const FLY_SPEED_FAST_MUL: f32 = 2.0;

/// 笔触尺寸下限（体素）
pub const EDIT_SIZE_MIN: u32 = 1;
/// 编辑射线射程（体素）
pub const EDIT_REACH: f32 = 256.0;
/// 左键点击与拖拽转向的判定阈值（像素）
pub const DRAG_PX: f32 = 4.0;

/// 显示用换算：1 米 = 多少体素
pub const VOXEL_PER_METER: f32 = 50.0;
/// 相机读数刷新周期（秒）
pub const CAM_INFO_REFRESH_SECS: f32 = 0.25;
/// FPS 滚动窗口长度（秒）
pub const FPS_WINDOW_SECS: f32 = 1.0;
/// 半分辨率渲染倍数
pub const HALF_RES_FACTOR: u32 = 2;

/// 展示窗折线图画布宽度（像素）
pub const SHOWCASE_PLOT_W: u32 = 216;
/// 展示窗折线图画布高度（像素）
pub const SHOWCASE_PLOT_H: u32 = 48;
/// 展示窗折线图样本容量
pub const SHOWCASE_PLOT_CAP: usize = 128;

/// demo 场景世界边长（voxel）= tile 数 × 512
pub const EXT_VOXEL_X: i32 = DEMO_TILES * 512;
/// demo 场景世界边长（voxel）
pub const EXT_VOXEL_Z: i32 = DEMO_TILES * 512;
/// demo 场景世界中心（voxel）
pub const EXT_VOXEL_HALF: i32 = EXT_VOXEL_X / 2;
