pub const STARTUP_DEMO_SCENE: bool = false;
pub const START_CAMERA_SKY: bool = false;
pub const DEMO_TILES: i32 = 2;
pub const BENCH_UNFOCUSED: bool = false;
pub const AUTO_ORBIT: bool = false;

pub fn bench() -> bool {
  static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
  *ON.get_or_init(|| std::env::var("GATE_BENCH").is_ok_and(|v| !v.is_empty() && v != "0"))
}

pub fn bench_orbit() -> bool {
  static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
  *ON.get_or_init(|| {
    std::env::var("GATE_BENCH").is_ok_and(|v| v.to_ascii_lowercase().contains("orbit"))
  })
}

pub fn bench_fly() -> bool {
  static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
  *ON.get_or_init(|| {
    std::env::var("GATE_BENCH").is_ok_and(|v| v.to_ascii_lowercase().contains("fly"))
  })
}

pub const BENCH_FLY_SPEED: f32 = 2000.0;
pub const BENCH_FLY_TURN: f32 = 0.16;
pub const BENCH_FLY_YAW: f32 = 0.35;
pub const EDIT_SELFTEST: bool = false;

pub const DEFAULT_LOG_FILTER: &str = "info,wgpu=debug,wgpu_core=debug,\
  wgpu_hal::vulkan::instance=off,\
  wgpu_hal::vulkan::surface=off";

pub const FOV_Y: f32 = 60.0_f32.to_radians();
pub const CAM_NEAR: f32 = 1.0;
pub const CAM_FAR: f32 = 1_048_576.0;
pub const ROT_SPEED: f32 = 0.005;
pub const ZOOM_LOG_SPEED: f32 = 0.35;
pub const FLY_SPEED_DEFAULT: f32 = 128.0;
pub const FLY_SPEED_FAST_MUL: f32 = 2.0;

pub const CROSSHAIR_ARM: f32 = 8.0;
pub const CROSSHAIR_THICK: f32 = 1.0;
pub const CROSSHAIR_GAP: f32 = 3.0;

pub const EDIT_SIZE_MIN: u32 = 1;
pub const EDIT_REACH: f32 = f32::INFINITY;
pub const EDIT_REPEAT_SECS: f32 = 0.1;
pub const EDIT_BUDGET_MS: f32 = 1.0;

pub const VOXEL_PER_METER: f32 = 50.0;
pub const CAM_INFO_REFRESH_SECS: f32 = 0.25;
pub const FPS_WINDOW_SECS: f32 = 1.0;
pub const FPS_MAX_FRAMES_PER_TICK: u32 = 8;
pub const FPS_UPDATE_SECS: f32 = 0.2;

pub const SHOWCASE_PLOT_W: u32 = 216;
pub const SHOWCASE_PLOT_H: u32 = 48;
pub const SHOWCASE_PLOT_CAP: usize = 128;

pub const EXT_VOXEL_X: i32 = DEMO_TILES * 512;
pub const EXT_VOXEL_Z: i32 = DEMO_TILES * 512;
pub const EXT_VOXEL_HALF: i32 = EXT_VOXEL_X / 2;

pub const DEMO_DISPLACE_SAMPLE: bool = true;
pub const DEMO_DISPLACE_HEIGHT_MAP: &str = "stone_wall_04";
pub const DEMO_DISPLACE_AMPLITUDE_OVERRIDE: Option<f32> = None;
pub const DEMO_DISPLACE_TEX_SCALE: f32 = 100.0;

pub const EDIT_DISPLACE_SIZE_MAX: u32 = 32;
