//! 天象：**时间 → 太阳 / 月亮 / 天空色 / 光柱 / 天体盘**（每帧覆写静态主题 `assets/lighting/*.ron`）。
//!
//! 天上只有**一条路**：时刻/年积日/纬度决定一切（`assets/lighting/*.ron` 只提供首帧初值）。
//!
//! 三件事，口径各自独立：
//!   · **位置**：真实天文公式（赤纬 + 时角 + 纬度）给出太阳的高度角与方位角；
//!     月亮取**反日点**（`moon_dir = -sun_dir`）—— 这正是"日月不同时出现"的几何形式
//!     （`alt_moon = -alt_sun` 逐字成立）⇒ 两者可以**共用同一个方向光槽位**（`lights[0]`）。
//!   · **颜色 / 光柱 / 天体盘**：一张按**太阳高度角**排序的关键帧表（[`KEYS`]）线性插值出
//!     「主光色 / 主光强度 / 天空色 / 光柱三旋钮 / 盘角径 / 光晕 / 盘亮度」；
//!     负高度角那几帧描述的就是**月亮**。
//!   · **下发**：写 [`LightingTheme`]（主光 + 天空色）与 `FogSettings`（光柱 + 天体盘）——
//!     两者每帧分别被 `build_light_pool` 打进 BG3 光池、被 `volumetric::prepare_fog` 读进光柱 uniform
//!     ⇒ **不需要新的 uniform / bind group / pass**，光柱、GI、阴影、天空盘全部自动跟随。
//!
//! 覆写（菜单「渲染/天空」每组一个开关）：**关**（默认）⇒ 上面这些量每帧由时间曲线给，
//! 面板滑杆实时显示当前值；**开** ⇒ 用面板/资源里的值（`FogSettings` 的三个光柱旋钮与两个盘旋钮、
//! [`SkySettings`] 的三个颜色）。三组独立：只覆写你关心的那组，其余仍随时间走。
//! 面板上拖动某个滑杆会自动打开它那一组的覆写（`gate-app` 的观察者里做，避免"拖了没反应"）。
//!
//! 已知取舍（"日月不同时出现、共用一个方向光槽位"的代价，**有意接受**）：
//!   · **没有月相**：反日点的几何把月亮定死为**满月**（相位 = 日月夹角，而这里它恒为 180°）；
//!     要月相就得让月亮在白天也挂在天上，与"不同时出现"冲突；
//!   · 没有白道倾角（±5°）与交点的 18.6 年进动、没有均时差、没有月球轨道偏心率；
//!   · **天体盘的角径是夸张值**（≈ `1.2°`，真实太阳只有 `0.27°`）：`0.27°` 在 1080p 下约 5 像素，
//!     软边缘与光晕几乎没有像素可用；盘的大小纯粹是观感（光子柱是另一条链），所以按"看得清"来定，
//!     并在**低高度角时略放大**（落日大太阳，常见的镜头语言，不是物理）；
//!   · 天空仍是**常量色**（随高度角变，但不随视线方向变）⇒ GI 的 miss 辐亮度与玻璃/镜面反射的
//!     逃逸口径都不动（`sky_rgb()` 依旧是全工程唯一那条常量天光）。
//!
//! 面板（菜单「渲染/天空」，四个子页）在 `gate-app`，本模块只提供资源与推导。

use bevy::ecs::resource::Resource;
use bevy::ecs::system::Local;
use bevy::prelude::{Res, ResMut, Time};
use glam::Vec3;

use crate::lighting::{DirLightCfg, LightingTheme, SkyCfg};
use crate::volumetric::FogSettings;

/// 黄赤交角（度）：太阳赤纬的正弦振幅。
const AXIAL_TILT_DEG: f32 = 23.44;
/// 赤纬的相位锚：年积日 `81`（春分）赤纬 = 0、`172`（夏至）= `+AXIAL_TILT_DEG`、`355`（冬至）≈ `−AXIAL_TILT_DEG`。
const DECL_ANCHOR_DAY: f32 = 81.0;
/// 一昼夜的小时数（时角换算用）。
const DAY_HOURS: f32 = 24.0;
/// 月亮的盘亮度倍数（`DirLightCfg::disk_scale`，关键帧表的 `disk_scale` 列）：日月共用 WESL 的
/// `SUN_DISK_GAIN`，月亮再乘它 —— 夜里曝光会被拉到 `ev_max`（默认 `+3` 档 = `8×`），
/// 乘 `1.0` 的话盘面会直接撞白顶、又变成"第二个太阳"；乘 `0.6` 让它"亮而不白"。
/// 这个量**不**随时间变（是月亮自己的属性），故是常量而不是关键帧列。
const MOON_DISK_SCALE: f32 = 0.6;

/// 一个关键帧：**主天体**（白天 = 太阳、夜间 = 月亮）的全部外观 + 天空与光柱的旋钮。
///
/// 颜色字段的**空间不同**（沿用既有约定，不另立口径）：
///   · `light_color` 是**线性**色（经 `color_intensity` 直出，shader 不再解码）；
///   · `sky_color` 与 `assets/lighting/*.ron` 同口径（`common.wesl::sky_rgb` 里再做 `srgb_to_linear`）。
///
/// 表按 `elev_deg` **升序**（[`sample_key`] 依赖这一点）。
#[derive(Clone, Copy)]
struct SkyKey {
  /// **太阳**高度角（度）：整个表都以它排序 —— 它是"一天里的什么时候"的单调代理量。
  elev_deg: f32,
  /// 主光色（线性）。
  light_color: [f32; 3],
  /// 主光强度（太阳/月亮共用这一列）。
  light_intensity: f32,
  /// 天空色（`assets/lighting/*.ron` 口径）。
  sky_color: [f32; 3],
  /// 光柱**强度**（`FogSettings::strength`）。
  blur_strength: f32,
  /// 光柱**衰减**（`FogSettings::decay`）。
  blur_decay: f32,
  /// 光柱**集中度**（`FogSettings::focus`）。
  blur_focus: f32,
  /// 天体**盘角径**（度 → `FogSettings::sun_cone` 存弧度）。
  disk_radius_deg: f32,
  /// **光晕**强度（`FogSettings::halo`）。
  halo: f32,
  /// 盘的亮度倍数（`DirLightCfg::disk_scale`；太阳 `1`、月亮 [`MOON_DISK_SCALE`]）。
  disk_scale: f32,
}

/// 天象关键帧：**太阳高度角** → 主光 / 天空 / 光柱 / 天体盘。
///
/// 为什么按高度角而不是按时刻：高度角同时编码了时刻、季节与纬度（同一张表三个旋钮共用），
/// 且"日落该是什么颜色/多长的光柱"本来就是高度角的函数（大气路径长度）。
///
/// `elev_deg = 0` 那一帧的强度是 **0**：日月在这里交接，两边都给不出直射光 ⇒ 换天体的**接缝不可见**
/// （颜色、方向、盘都会跳，但强度是 0，跳的是"没有东西"）。
///
/// 光柱的三个旋钮随高度角走的就是"**太阳越低 ⇒ 光柱越长、越亮、越集中**"
/// （近地平线时大气路径最长、散射最烈，是光柱最好看的时刻）；月亮那几帧给的是克制的月光柱。
const KEYS: &[SkyKey] = &[
  // ---- 夜：主光 = 反日点的月亮 ----
  SkyKey {
    elev_deg: -90.0,
    light_color: [0.72, 0.80, 1.00],
    light_intensity: 0.045,
    sky_color: [0.030, 0.040, 0.090],
    blur_strength: 0.20,
    blur_decay: 0.55,
    blur_focus: 28.0,
    disk_radius_deg: 1.00,
    halo: 0.20,
    disk_scale: MOON_DISK_SCALE,
  },
  SkyKey {
    elev_deg: -18.0,
    light_color: [0.72, 0.80, 1.00],
    light_intensity: 0.035,
    sky_color: [0.050, 0.070, 0.140],
    blur_strength: 0.22,
    blur_decay: 0.55,
    blur_focus: 28.0,
    disk_radius_deg: 1.00,
    halo: 0.20,
    disk_scale: MOON_DISK_SCALE,
  },
  SkyKey {
    elev_deg: -12.0,
    light_color: [0.74, 0.80, 1.00],
    light_intensity: 0.025,
    sky_color: [0.080, 0.100, 0.200],
    blur_strength: 0.25,
    blur_decay: 0.58,
    blur_focus: 26.0,
    disk_radius_deg: 1.05,
    halo: 0.22,
    disk_scale: MOON_DISK_SCALE,
  },
  SkyKey {
    elev_deg: -6.0,
    light_color: [0.80, 0.78, 0.98],
    light_intensity: 0.012,
    sky_color: [0.180, 0.160, 0.300],
    blur_strength: 0.30,
    blur_decay: 0.62,
    blur_focus: 24.0,
    disk_radius_deg: 1.15,
    halo: 0.30,
    disk_scale: MOON_DISK_SCALE,
  },
  // 低空的天体一律偏红（太阳与月亮都过同一层大气），盘也略大、晕略强
  SkyKey {
    elev_deg: -2.0,
    light_color: [1.00, 0.62, 0.42],
    light_intensity: 0.002,
    sky_color: [0.420, 0.300, 0.340],
    blur_strength: 0.50,
    blur_decay: 0.70,
    blur_focus: 22.0,
    disk_radius_deg: 1.40,
    halo: 0.55,
    disk_scale: 0.45,
  },
  // 地平线：**交接帧**（强度 0，见上方说明）
  SkyKey {
    elev_deg: 0.0,
    light_color: [1.00, 0.42, 0.18],
    light_intensity: 0.0,
    sky_color: [0.500, 0.340, 0.300],
    blur_strength: 1.20,
    blur_decay: 0.80,
    blur_focus: 22.0,
    disk_radius_deg: 1.60,
    halo: 1.00,
    disk_scale: 1.0,
  },
  // ---- 昼：主光 = 太阳（金色时刻的光柱最盛）----
  SkyKey {
    elev_deg: 2.0,
    light_color: [1.00, 0.45, 0.20],
    light_intensity: 0.060,
    sky_color: [0.520, 0.400, 0.340],
    blur_strength: 1.80,
    blur_decay: 0.85,
    blur_focus: 22.0,
    disk_radius_deg: 1.60,
    halo: 1.20,
    disk_scale: 1.0,
  },
  SkyKey {
    elev_deg: 6.0,
    light_color: [1.00, 0.62, 0.36],
    light_intensity: 0.250,
    sky_color: [0.450, 0.500, 0.620],
    blur_strength: 1.30,
    blur_decay: 0.78,
    blur_focus: 21.0,
    disk_radius_deg: 1.50,
    halo: 1.00,
    disk_scale: 1.0,
  },
  SkyKey {
    elev_deg: 15.0,
    light_color: [1.00, 0.84, 0.66],
    light_intensity: 0.550,
    sky_color: [0.450, 0.580, 0.860],
    blur_strength: 0.85,
    blur_decay: 0.70,
    blur_focus: 20.0,
    disk_radius_deg: 1.35,
    halo: 0.70,
    disk_scale: 1.0,
  },
  SkyKey {
    elev_deg: 30.0,
    light_color: [1.00, 0.93, 0.82],
    light_intensity: 0.720,
    sky_color: [0.460, 0.620, 0.960],
    blur_strength: 0.60,
    blur_decay: 0.64,
    blur_focus: 19.0,
    disk_radius_deg: 1.25,
    halo: 0.55,
    disk_scale: 1.0,
  },
  // 天顶附近：≈ `assets/lighting/day_outdoor.ron` 那一组（静态主题与时间驱动的白天在此对齐）
  SkyKey {
    elev_deg: 60.0,
    light_color: [1.00, 0.96, 0.88],
    light_intensity: 0.800,
    sky_color: [0.470, 0.650, 1.000],
    blur_strength: 0.45,
    blur_decay: 0.60,
    blur_focus: 18.0,
    disk_radius_deg: 1.20,
    halo: 0.50,
    disk_scale: 1.0,
  },
  SkyKey {
    elev_deg: 90.0,
    light_color: [1.00, 0.97, 0.90],
    light_intensity: 0.820,
    sky_color: [0.470, 0.650, 1.000],
    blur_strength: 0.45,
    blur_decay: 0.60,
    blur_focus: 18.0,
    disk_radius_deg: 1.20,
    halo: 0.50,
    disk_scale: 1.0,
  },
];

/// 天象档位（菜单「渲染/天空/时间」+ 三组覆写开关）。
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct SkySettings {
  /// **时刻**（0..24，正午 = `12`）：与年积日、纬度一起决定主天体的高度角与方位角。
  pub hour: f32,
  /// **年积日**（1..365）：只用来算太阳赤纬。
  pub day_of_year: f32,
  /// **纬度**（度，北纬为正）：决定天顶方向与地轴的夹角。
  pub latitude_deg: f32,
  /// **自动流逝**：时刻随实时时间前进。
  pub auto: bool,
  /// 流逝速度（游戏小时 / 实时秒）：`0.1` ⇒ 一昼夜 `4` 分钟。
  pub hours_per_sec: f32,
  /// 覆写「径向模糊」组：`false` ⇒ 强度/衰减/集中度由 [`KEYS`] 每帧写进 `FogSettings`；
  /// `true` ⇒ 保持面板上的值不动。
  pub override_blur: bool,
  /// 覆写「天体盘」组：同上，管 `FogSettings::sun_cone`（角径）与 `halo`。
  pub override_disk: bool,
  /// 覆写「颜色」组：`true` ⇒ 用下面三个颜色代替 [`KEYS`] 算出来的主光色与天空色
  /// （**强度**仍由时间给）。
  pub override_colors: bool,
  /// 太阳色（**sRGB**，0..1，= 面板 HEX 的口径）：下发前转线性（与 `KEYS::light_color` 同空间）。
  pub sun_color: [f32; 3],
  /// 月亮色（**sRGB**，0..1），同上。
  pub moon_color: [f32; 3],
  /// 天空色（**sRGB**，0..1，= `assets/lighting/*.ron` 的口径，shader 侧再 `srgb_to_linear`）。
  pub sky_color: [f32; 3],
}

impl Default for SkySettings {
  /// 缺省 = **春分 · 正午 · 北纬 45°**：此时太阳正南、高度角 `45°`，落在 [`KEYS`] 的日间段里
  /// （主光色/强度与天空色都贴着 `assets/lighting/day_outdoor.ron` 那一组值）。
  /// 面板初值（`assets/ui/debug_menu.toml`）必须与这里逐项一致（三组覆写都默认关）。
  fn default() -> Self {
    Self {
      hour: 12.0,
      day_of_year: DECL_ANCHOR_DAY,
      latitude_deg: 45.0,
      auto: false,
      hours_per_sec: 0.1,
      override_blur: false,
      override_disk: false,
      override_colors: false,
      // 三个色的缺省 = 面板 HEX（`FFF5E6` / `C8D4FF` / `78A7FF`）的换算结果
      // （写成 `n / 255.0` 是为了让"哪个字节"一眼可查；`255/255` 就是 `1.0`）。
      sun_color: [1.0, 245.0 / 255.0, 230.0 / 255.0],
      moon_color: [200.0 / 255.0, 212.0 / 255.0, 1.0],
      sky_color: [120.0 / 255.0, 167.0 / 255.0, 1.0],
    }
  }
}

/// 一个天体（太阳或月亮）的位置。
#[derive(Clone, Copy, Debug)]
pub struct Celestial {
  /// 高度角（度，地平线 = 0，负值 = 在地平线以下）。
  pub alt_deg: f32,
  /// 单位方向（**指向该天体**）。世界坐标约定：`+X` 东、`+Y` 天顶、`+Z` 南。
  pub dir: Vec3,
}

/// 太阳的位置。`hour` = 地方太阳时（`12` = 正午 = 时角 0），`day_of_year` = 年积日，`latitude_deg` = 纬度（北正）。
///
/// 方位角：先算"自南起、向西为正"的 `atan2(sinH, cosH·sinφ − tanδ·cosφ)`（正午 = 0、下午为正），
/// 再换成"自北起、顺时针"，最后落到世界坐标（`+X` 东、`+Z` 南）。
/// 极区（`φ → ±90°`）不需要特判：`cosφ → 0` 时该式自然退化成"方位角 = 时角"，正是极昼/极夜该有的样子。
pub fn sun(hour: f32, day_of_year: f32, latitude_deg: f32) -> Celestial {
  use std::f32::consts::{PI, TAU};
  let phi = latitude_deg.to_radians();
  let dec = AXIAL_TILT_DEG.to_radians() * ((day_of_year - DECL_ANCHOR_DAY) * TAU / 365.0).sin();
  let h = (hour - 12.0) * (360.0 / DAY_HOURS).to_radians();
  let sin_alt = phi.sin() * dec.sin() + phi.cos() * dec.cos() * h.cos();
  let alt = sin_alt.asin();
  let cos_alt = alt.cos();
  let az_south = h.sin().atan2(h.cos() * phi.sin() - dec.tan() * phi.cos());
  let az_north = PI + az_south;
  Celestial {
    alt_deg: alt.to_degrees(),
    dir: Vec3::new(cos_alt * az_north.sin(), sin_alt, -cos_alt * az_north.cos()),
  }
}

/// 月亮的位置 = **反日点**（见文件头：这条约束是用"日月不同时出现"换来的）。
pub fn moon(hour: f32, day_of_year: f32, latitude_deg: f32) -> Celestial {
  let s = sun(hour, day_of_year, latitude_deg);
  Celestial { alt_deg: -s.alt_deg, dir: -s.dir }
}

/// 太阳的高度角（度）—— 菜单日志与 `gate-app` 的观察者要用它，故单列一个函数。
pub fn sun_altitude_deg(hour: f32, day_of_year: f32, latitude_deg: f32) -> f32 {
  sun(hour, day_of_year, latitude_deg).alt_deg
}

/// 按高度角在 [`KEYS`] 上线性插值（表外取两端）；`KEYS` 必须按 `elev_deg` 升序。
fn sample_key(alt_deg: f32) -> SkyKey {
  let (first, last) = (KEYS[0], KEYS[KEYS.len() - 1]);
  if alt_deg <= first.elev_deg {
    return first;
  }
  if alt_deg >= last.elev_deg {
    return last;
  }
  let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
  for w in KEYS.windows(2) {
    let (a, b) = (w[0], w[1]);
    if alt_deg <= b.elev_deg {
      let t = (alt_deg - a.elev_deg) / (b.elev_deg - a.elev_deg);
      return SkyKey {
        elev_deg: alt_deg,
        light_color: Vec3::from_array(a.light_color)
          .lerp(Vec3::from_array(b.light_color), t)
          .to_array(),
        light_intensity: lerp(a.light_intensity, b.light_intensity, t),
        sky_color: Vec3::from_array(a.sky_color).lerp(Vec3::from_array(b.sky_color), t).to_array(),
        blur_strength: lerp(a.blur_strength, b.blur_strength, t),
        blur_decay: lerp(a.blur_decay, b.blur_decay, t),
        blur_focus: lerp(a.blur_focus, b.blur_focus, t),
        disk_radius_deg: lerp(a.disk_radius_deg, b.disk_radius_deg, t),
        halo: lerp(a.halo, b.halo, t),
        disk_scale: lerp(a.disk_scale, b.disk_scale, t),
      };
    }
  }
  last
}

/// sRGB（0..1）→ 线性：与 shader 的 `srgb_to_linear` 同一条公式（面板 HEX 的太阳/月亮色要用它）。
fn srgb_to_linear(c: [f32; 3]) -> [f32; 3] {
  let f = |v: f32| {
    if v > 0.04045 { ((v + 0.055) / 1.055).powf(2.4) } else { v / 12.92 }
  };
  [f(c[0]), f(c[1]), f(c[2])]
}

pub struct SkyPlugin;

impl bevy::app::Plugin for SkyPlugin {
  fn build(&self, app: &mut bevy::app::App) {
    // main world 的 `Update`：写在 `ExtractSchedule` 之前 ⇒ 当帧的提取拿到的就是推导后的主题。
    app.init_resource::<SkySettings>().add_systems(bevy::app::Update, apply_sky);
  }
}

/// 每帧：推进时刻（若自动流逝）→ 算太阳/月亮 → 把结果写进 [`LightingTheme`] 与 `FogSettings`。
///
/// 三个覆写开关各管一组（见 [`SkySettings`]）：关着的组被本系统每帧覆写，开着的组原样不动。
fn apply_sky(
  time: Res<Time>,
  mut sky: ResMut<SkySettings>,
  mut fog: ResMut<FogSettings>,
  theme: Option<ResMut<LightingTheme>>,
  mut last_body: Local<u8>,
) {
  if sky.auto {
    sky.hour = (sky.hour + time.delta_secs() * sky.hours_per_sec.max(0.0)).rem_euclid(DAY_HOURS);
  }

  let sun_pos = sun(sky.hour, sky.day_of_year, sky.latitude_deg);
  // 主天体：白天 = 太阳、夜间 = 月亮（反日点）。两者的高度角互为相反数 ⇒ **恒有一个在地平线以上**。
  // 太阳高度角同时是整张关键帧表的横轴（它编码了"一天里的什么时候"），与谁是主光无关。
  let is_sun = sun_pos.alt_deg >= 0.0;
  let body = if is_sun { sun_pos } else { moon(sky.hour, sky.day_of_year, sky.latitude_deg) };
  let k = sample_key(sun_pos.alt_deg);

  // ---- 颜色（主光 + 天空色）：覆写开着就换成面板那三个色 ----
  let light_color = if sky.override_colors {
    srgb_to_linear(if is_sun { sky.sun_color } else { sky.moon_color })
  } else {
    k.light_color
  };
  let sky_color = if sky.override_colors { sky.sky_color } else { k.sky_color };

  // ---- 光柱 + 天体盘：覆写开着就不碰 FogSettings（面板值即真值），关着就按时间曲线写回 ----
  // 两组各管各的：只覆写其中一组时，另一组仍取时间曲线。
  let (mut strength, mut decay, mut focus) = (fog.strength(), fog.decay(), fog.focus());
  if !sky.override_blur {
    fog.strength = k.blur_strength;
    fog.decay = k.blur_decay;
    fog.focus = k.blur_focus;
    (strength, decay, focus) = (k.blur_strength, k.blur_decay, k.blur_focus);
  }
  let (mut radius_deg, mut halo) = (fog.sun_cone.max(0.0).to_degrees(), fog.halo.max(0.0));
  if !sky.override_disk {
    fog.sun_cone = k.disk_radius_deg.to_radians();
    fog.halo = k.halo;
    (radius_deg, halo) = (k.disk_radius_deg, k.halo);
  }

  if let Some(mut theme) = theme {
    theme.sun = Some(DirLightCfg {
      // `dir` = 光的**传播方向**（指向场景）⇒ 与"指向天体"反号（`build_light_pool` 同口径）。
      dir: (-body.dir).to_array(),
      // 未被渲染消费：天空里那个盘的半径走光柱 uniform 的「角径」（见 `volumetric.wesl`）。
      angular_radius_deg: 0.0,
      color: light_color,
      intensity: k.light_intensity,
      disk_scale: k.disk_scale,
    });
    theme.sky = Some(SkyCfg { color: sky_color });
  }

  // 换天体（太阳 ⇄ 月亮）时打一行：这是本系统唯一的信息量事件（每半天一次），
  // 逐帧的数值不进日志 —— 拖时刻/自动流逝时它会把日志冲爆。
  let body_id = if is_sun { 1u8 } else { 2u8 };
  if *last_body != body_id {
    *last_body = body_id;
    bevy::log::info!(
      target: "gate",
      "天象 → {}：高度角 {:.1}°（时刻 {:.2}h、年积日 {:.0}、纬度 {:.1}°）｜主光 线性色 \
       ({:.2}, {:.2}, {:.2}) 强度 {:.3}｜天空色（RON 口径）({:.2}, {:.2}, {:.2})｜光柱 \
       强度 {:.2} 衰减 {:.2} 集中度 {:.0}｜天体盘 角径 {:.2}° 光晕 {:.2} 盘 ×{:.2}",
      if is_sun { "太阳" } else { "月亮（反日点）" },
      body.alt_deg,
      sky.hour,
      sky.day_of_year,
      sky.latitude_deg,
      light_color[0],
      light_color[1],
      light_color[2],
      k.light_intensity,
      sky_color[0],
      sky_color[1],
      sky_color[2],
      strength,
      decay,
      focus,
      radius_deg,
      halo,
      k.disk_scale,
    );
  }
}
