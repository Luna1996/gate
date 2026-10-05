use bevy::ecs::resource::Resource;
use bevy::ecs::system::Local;
use bevy::prelude::{Res, ResMut, Time};
use glam::Vec3;

use crate::lighting::{DirLightCfg, LightingTheme, SkyCfg};
use crate::volumetric::FogSettings;

const AXIAL_TILT_DEG: f32 = 23.44;
const DECL_ANCHOR_DAY: f32 = 81.0;
const DAY_HOURS: f32 = 24.0;
const MOON_DISK_SCALE: f32 = 0.6;

#[derive(Clone, Copy)]
struct SkyKey {
  elev_deg: f32,
  light_color: [f32; 3],
  light_intensity: f32,
  sky_color: [f32; 3],
  blur_strength: f32,
  blur_decay: f32,
  blur_focus: f32,
  disk_radius_deg: f32,
  halo: f32,
  disk_scale: f32,
}

const KEYS: &[SkyKey] = &[
  SkyKey {
    elev_deg: -90.0,
    light_color: [0.72, 0.80, 1.00],
    light_intensity: 0.045,
    sky_color: [0.030, 0.040, 0.090],
    blur_strength: 0.20,
    blur_decay: 0.55,
    blur_focus: 28.0,
    disk_radius_deg: 2.00,
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
    disk_radius_deg: 2.00,
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
    disk_radius_deg: 2.10,
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
    disk_radius_deg: 2.30,
    halo: 0.30,
    disk_scale: MOON_DISK_SCALE,
  },
  SkyKey {
    elev_deg: -2.0,
    light_color: [1.00, 0.62, 0.42],
    light_intensity: 0.002,
    sky_color: [0.420, 0.300, 0.340],
    blur_strength: 0.50,
    blur_decay: 0.70,
    blur_focus: 22.0,
    disk_radius_deg: 2.80,
    halo: 0.55,
    disk_scale: 0.45,
  },
  SkyKey {
    elev_deg: 0.0,
    light_color: [1.00, 0.42, 0.18],
    light_intensity: 0.0,
    sky_color: [0.500, 0.340, 0.300],
    blur_strength: 1.20,
    blur_decay: 0.80,
    blur_focus: 22.0,
    disk_radius_deg: 3.20,
    halo: 1.00,
    disk_scale: 1.0,
  },
  SkyKey {
    elev_deg: 2.0,
    light_color: [1.00, 0.45, 0.20],
    light_intensity: 0.060,
    sky_color: [0.520, 0.400, 0.340],
    blur_strength: 1.80,
    blur_decay: 0.85,
    blur_focus: 22.0,
    disk_radius_deg: 3.20,
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
    disk_radius_deg: 3.00,
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
    disk_radius_deg: 2.70,
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
    disk_radius_deg: 2.50,
    halo: 0.55,
    disk_scale: 1.0,
  },
  SkyKey {
    elev_deg: 60.0,
    light_color: [1.00, 0.96, 0.88],
    light_intensity: 0.800,
    sky_color: [0.470, 0.650, 1.000],
    blur_strength: 0.45,
    blur_decay: 0.60,
    blur_focus: 18.0,
    disk_radius_deg: 2.40,
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
    disk_radius_deg: 2.40,
    halo: 0.50,
    disk_scale: 1.0,
  },
];

#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct SkySettings {
  pub hour: f32,
  pub day_of_year: f32,
  pub latitude_deg: f32,
  pub auto: bool,
  pub hours_per_sec: f32,
  pub override_blur: bool,
  pub override_disk: bool,
  pub override_colors: bool,
  pub sun_color: [f32; 3],
  pub moon_color: [f32; 3],
  pub sky_color: [f32; 3],
}

impl Default for SkySettings {
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
      sun_color: [1.0, 245.0 / 255.0, 230.0 / 255.0],
      moon_color: [200.0 / 255.0, 212.0 / 255.0, 1.0],
      sky_color: [120.0 / 255.0, 167.0 / 255.0, 1.0],
    }
  }
}

#[derive(Clone, Copy, Debug)]
pub struct Celestial {
  pub alt_deg: f32,
  pub dir: Vec3,
}

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

pub fn moon(hour: f32, day_of_year: f32, latitude_deg: f32) -> Celestial {
  let s = sun(hour, day_of_year, latitude_deg);
  Celestial { alt_deg: -s.alt_deg, dir: -s.dir }
}

pub fn sun_altitude_deg(hour: f32, day_of_year: f32, latitude_deg: f32) -> f32 {
  sun(hour, day_of_year, latitude_deg).alt_deg
}

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

fn srgb_to_linear(c: [f32; 3]) -> [f32; 3] {
  let f = |v: f32| {
    if v > 0.04045 { ((v + 0.055) / 1.055).powf(2.4) } else { v / 12.92 }
  };
  [f(c[0]), f(c[1]), f(c[2])]
}

pub struct SkyPlugin;

impl bevy::app::Plugin for SkyPlugin {
  fn build(&self, app: &mut bevy::app::App) {
    app.init_resource::<SkySettings>().add_systems(bevy::app::Update, apply_sky);
  }
}

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
  let is_sun = sun_pos.alt_deg >= 0.0;
  let body = if is_sun { sun_pos } else { moon(sky.hour, sky.day_of_year, sky.latitude_deg) };
  let k = sample_key(sun_pos.alt_deg);

  let light_color = if sky.override_colors {
    srgb_to_linear(if is_sun { sky.sun_color } else { sky.moon_color })
  } else {
    k.light_color
  };
  let sky_color = if sky.override_colors { sky.sky_color } else { k.sky_color };

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
      dir: (-body.dir).to_array(),
      angular_radius_deg: 0.0,
      color: light_color,
      intensity: k.light_intensity,
      disk_scale: k.disk_scale,
    });
    theme.sky = Some(SkyCfg { color: sky_color });
  }

  let body_id = if is_sun { 1u8 } else { 2u8 };
  if *last_body != body_id {
    *last_body = body_id;
    bevy::log::debug!(
      target: "gate",
      "天象 → {}：高度角 {:.1}°（时刻 {:.2}h、年积日 {:.0}、纬度 {:.1}°）｜主光 线性 ({:.2}, {:.2}, {:.2}) \
       强度 {:.3}｜天空色 (RON) ({:.2}, {:.2}, {:.2})｜光柱 强度 {:.2} 衰减 {:.2} 集中度 {:.0}｜\
       天体盘 角径 {:.2}° 光晕 {:.2} 盘 ×{:.2}",
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
