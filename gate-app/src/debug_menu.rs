use std::collections::VecDeque;

use bevy::prelude::*;
use bevy::window::{
  MonitorSelection, PresentMode, PrimaryWindow, Window, WindowMode, WindowPosition,
};
use rust_i18n::t;

use crate::camera::{CameraMode, FlyCamera};
use crate::config::Config;
use crate::consts::{
  CAM_INFO_REFRESH_SECS, EDIT_SIZE_MIN, FPS_MAX_FRAMES_PER_TICK, FPS_UPDATE_SECS, FPS_WINDOW_SECS,
  VOXEL_PER_METER,
};
use crate::edit::{
  BrushMaterial, BrushShape, EditSettings, metal_toggle_to_metallic, smooth_pct_to_roughness,
  transparency_pct_to_transmission,
};
use crate::showcase::ShowcaseRoot;
use gate_ui::widgets::{LabelConfig, LabelStyle, label, px};
use gate_ui::{
  DebugMenuRoot, MenuAction, MenuActionEvent, MenuFile, MenuNode, UiCtx, UiTranslator,
  parse_hex_color, spawn_debug_menu,
};
use std::sync::atomic::Ordering;

pub const MENU_TOML_PATH: &str = "ui/debug_menu.toml";

pub const WORLD_MODEL_PATH: &str = "game/world/model";
pub const WORLD_RELOAD_PATH: &str = "game/world/reload";
pub const WORLD_DUMP_PATH: &str = "game/world/dump";
pub const WORLD_LOD_PATH: &str = "game/world/lod";
const WORLD_FAR_LEVELS_PATH: &str = "game/world/far_levels";
const WORLD_VOL_TINT_PATH: &str = "game/world/vol_tint";
const WORLD_STREAM_PAUSE_PATH: &str = "game/world/stream_pause";
const WORLD_LOAD_RADIUS_PATH: &str = "game/world/load_radius";
const WORLD_UNLOAD_RADIUS_PATH: &str = "game/world/unload_radius";
const WORLD_COARSE_RADIUS_PATH: &str = "game/world/coarse_radius";
const WORLD_COARSE_HEIGHT_PATH: &str = "game/world/coarse_height";
const WORLD_MOUNT_WORDS_PATH: &str = "game/world/mount_words";
const WORLD_MOUNT_COUNT_PATH: &str = "game/world/mount_count";
const WORLD_REQUESTS_LOAD_PATH: &str = "game/world/requests_load";
const WORLD_REQUEST_MB_PATH: &str = "game/world/request_mb";
pub const EDIT_PBR_ASSET_PATH: &str = "game/edit/mat/pbr_asset";
const EDIT_OFFSET_PATH: &str = "game/edit/brush/offset";
const VIDEO_AA_PATH: &str = "video/aa";

const EDIT_MATERIAL_PATHS: [&str; 5] = [
  "game/edit/mat/color",
  "game/edit/mat/emissive",
  "game/edit/mat/alpha",
  "game/edit/mat/smooth",
  "game/edit/mat/metal",
];

const SKY_HOUR_PATH: &str = "render/sky/time/hour";
const BLUR_STRENGTH_PATH: &str = "render/sky/blur/strength";
const BLUR_DECAY_PATH: &str = "render/sky/blur/decay";
const BLUR_FOCUS_PATH: &str = "render/sky/blur/focus";
const BLUR_OVERRIDE_PATH: &str = "render/sky/blur/override";
const DISK_RADIUS_PATH: &str = "render/sky/disk/radius";
const DISK_HALO_PATH: &str = "render/sky/disk/halo";
const DISK_OVERRIDE_PATH: &str = "render/sky/disk/override";
const COLOR_OVERRIDE_PATH: &str = "render/sky/color/override";
const COLOR_SUN_PATH: &str = "render/sky/color/sun";
const COLOR_MOON_PATH: &str = "render/sky/color/moon";
const COLOR_SKY_PATH: &str = "render/sky/color/sky";

pub fn load_menu(config: &Config) -> MenuFile {
  let path = gate_render::assets_dir().join(MENU_TOML_PATH);
  let mut model = match std::fs::read_to_string(&path) {
    Ok(src) => match MenuFile::from_toml(&src) {
      Ok(m) => {
        debug!("debug menu ← {}", path.display());
        m
      }
      Err(e) => {
        warn!("debug menu TOML 解析失败 {e} → 菜单为空");
        MenuFile::default()
      }
    },
    Err(e) => {
      warn!("debug menu TOML 缺失 {e} → 菜单为空");
      MenuFile::default()
    }
  };
  apply_world_model_options(&mut model);
  let applied = model.apply_values(&config.menu);
  model.window = config.window.clone();
  model.sanitize();
  info!("debug menu config 命中 {applied}/{}", config.menu.len());
  model
}

fn apply_world_model_options(model: &mut MenuFile) {
  let mut options = crate::vox_scene::scan_vox_models();
  options.push(crate::scene::CUBE_IN_VOID.to_string());
  options.push(crate::scene::INFINITE_CUBES.to_string());
  options.push(crate::mc::MC_MAP.to_string());
  options.sort();
  options.dedup();
  let Some(MenuNode::Dropdown { options: opts, selected, .. }) =
    model.node_mut(&split(WORLD_MODEL_PATH))
  else {
    return;
  };
  let previous = opts.get(*selected).cloned();
  *selected = previous
    .as_ref()
    .and_then(|p| options.iter().position(|o| o == p))
    .or_else(|| options.iter().position(|o| o == "nuke"))
    .unwrap_or(0);
  *opts = options;
}

#[derive(Component)]
pub(crate) struct DebugUiRoot;

#[derive(Component)]
pub(crate) struct FpsOverlay;

#[derive(Component)]
pub(crate) struct FpsOverlayText;

#[derive(Resource, Default, Debug, Clone, Copy)]
pub struct FpsOverlayVisible(pub bool);

#[derive(Resource, Default)]
struct WindowedRestore(Option<(UVec2, Option<IVec2>)>);

#[derive(Resource, Default)]
pub struct FpsWindow {
  intervals: VecDeque<f32>,
  last: Option<(f32, u64)>,
}

pub(crate) fn spawn_debug_menu_ui(world: &mut World, ctx: &UiCtx) {
  let model = {
    let config = world.resource::<Config>();
    load_menu(config)
  };
  let translate = world.resource::<UiTranslator>().handle();
  let ctx = UiCtx::new(ctx.theme, ctx.font).with_icon_font(ctx.icon_font).with_translate(translate);
  let handle = spawn_debug_menu(world, &ctx, model);
  register_callbacks(world);
  apply_initial_state(world, handle.root);
  spawn_fps_overlay(world, &ctx);
}

pub(crate) fn sync_ui_locale(mut translator: ResMut<UiTranslator>, mut last: Local<String>) {
  let now = rust_i18n::locale().to_string();
  if now != *last {
    debug!("UI 语言 → {now}");
    *last = now;
    translator.bump();
  }
}

fn spawn_fps_overlay(world: &mut World, ctx: &UiCtx) {
  let visible = world.resource::<FpsOverlayVisible>().0;
  let mut e = world.spawn((
    Name::new("fps-overlay"),
    FpsOverlay,
    Node {
      position_type: PositionType::Absolute,
      right: px(6.0),
      top: px(6.0),
      padding: UiRect::axes(px(6.0), px(2.0)),
      ..default()
    },
    BackgroundColor(Color::BLACK),
    Visibility::Hidden,
  ));
  e.with_children(|p| {
    let h = label(
      ctx,
      p,
      LabelConfig { text: "(--, --, --, --)".into(), style: LabelStyle::Muted, ..default() },
    );
    p.world_mut().entity_mut(*h).insert((FpsOverlayText, TextColor(Color::WHITE)));
  });
  let root = e.id();
  if visible {
    world.entity_mut(root).insert(Visibility::Visible);
  }
}

fn apply_initial_state(world: &mut World, root: Entity) {
  let Some(model) = gate_ui::menu_model(world).cloned() else { return };
  let mut actions = Vec::new();
  collect_actions(&model.items, "", &mut actions);
  let n = actions.len();
  for (path, action) in actions {
    world.trigger(MenuActionEvent { entity: root, path, action });
  }
  info!(target: "gate", "debug menu 初值 {n} 项");
}

fn collect_actions(nodes: &[MenuNode], prefix: &str, out: &mut Vec<(String, MenuAction)>) {
  for n in nodes {
    let path = if prefix.is_empty() { n.id().to_string() } else { format!("{prefix}/{}", n.id()) };
    match n {
      MenuNode::SubMenu { .. } => collect_actions(n.children(), &path, out),
      MenuNode::Toggle { checked, .. } => out.push((path, MenuAction::Toggle(*checked))),
      MenuNode::SwitchGroup { selected, .. } | MenuNode::Dropdown { selected, .. } => {
        out.push((path, MenuAction::Select(*selected)));
      }
      MenuNode::Slider { value, .. } => out.push((path, MenuAction::Value(*value))),
      MenuNode::Input { fields, .. } => {
        if let Some(f) = fields.first() {
          out.push((path, MenuAction::Text(f.text.clone())));
        }
      }
      MenuNode::Color { hex, .. } => out.push((path, MenuAction::Text(hex.clone()))),
      MenuNode::Buttons { .. } | MenuNode::Text { .. } => {}
    }
  }
}

fn split(path: &str) -> Vec<String> {
  path.split('/').filter(|s| !s.is_empty()).map(|s| s.to_string()).collect()
}

fn register_callbacks(world: &mut World) {
  world.init_resource::<WindowedRestore>();
  world.add_observer(
    |ev: On<MenuActionEvent>,
     mut q_win: Query<&mut Window, With<PrimaryWindow>>,
     mut fps: ResMut<FpsOverlayVisible>,
     mut scale: ResMut<gate_render::RenderScale>,
     mut post: ResMut<gate_render::PostFxSettings>,
     mut restore: ResMut<WindowedRestore>| {
      if let ("video/scale", MenuAction::Select(i)) = (ev.path.as_str(), &ev.action) {
        scale.factor = gate_render::RenderScale::SCALE_CHOICES[(*i).min(3)];
        info!("渲染分辨率 → 1/{}", scale.factor);
        return;
      }
      let MenuAction::Toggle(on) = ev.action else { return };
      match ev.path.as_str() {
        "video/vsync" => {
          let Ok(mut win) = q_win.single_mut() else { return };
          win.present_mode = if on { PresentMode::Fifo } else { PresentMode::AutoNoVsync };
          info!("VSync → {:?}", win.present_mode);
        }
        "video/fps" => {
          fps.0 = on;
          info!("FPS 覆盖层 → {}", if on { "on" } else { "off" });
        }
        "video/fullscreen" => {
          let Ok(mut win) = q_win.single_mut() else { return };
          if on {
            restore.0 = Some((
              win.resolution.physical_size(),
              match win.position {
                WindowPosition::At(p) => Some(p),
                _ => None,
              },
            ));
            win.mode = WindowMode::BorderlessFullscreen(MonitorSelection::Current);
          } else {
            win.mode = WindowMode::Windowed;
            if let Some((size, pos)) = restore.0.take() {
              win.resolution.set_physical_resolution(size.x, size.y);
              if let Some(p) = pos {
                win.position = WindowPosition::At(p);
              }
            }
          }
          info!("全屏 → {:?}", win.mode);
        }
        "video/aa" => {
          post.fxaa = on;
          info!("抗锯齿 → {}", if on { "on" } else { "off" });
        }
        _ => {}
      }
    },
  );

  world.add_observer(
    |ev: On<MenuActionEvent>,
     mut gi: ResMut<gate_render::gi::GiSettings>,
     mut eye: ResMut<gate_render::EyeAdaptSettings>,
     mut refl: ResMut<gate_render::ReflectionSettings>,
     mut base: ResMut<gate_render::BaseSettings>,
     mut debug: ResMut<gate_render::DebugNormals>| {
      match (ev.path.as_str(), &ev.action) {
        ("render/debug_view", MenuAction::Select(i)) => {
          debug.0 = match *i {
            1 => 4,
            2 => 5,
            3 => 6,
            _ => 0,
          };
          info!("调试视图 → {}", ["关", "间接光", "阴影可见度", "归属检查"][(*i).min(3)]);
        }
        ("render/base/shadow", MenuAction::Toggle(on)) => {
          base.shadow = *on;
          info!("直光阴影 → {}", if *on { "on" } else { "off" });
        }
        ("render/base/normal", MenuAction::Toggle(on)) => {
          base.implicit_normal = *on;
          info!("隐式法相 → {}", if *on { "on" } else { "off（原色直出）" });
        }
        ("render/gi/enabled", MenuAction::Toggle(on)) => {
          gi.enabled = *on;
          info!("GI → {}", if *on { "on" } else { "off" });
        }
        ("render/gi/res", MenuAction::Select(i)) => {
          gi.gi_div = gate_render::gi::GiSettings::DIV_CHOICES[(*i).min(2)];
          info!("GI 分辨率 → 1/{}", gi.gi_div);
        }
        ("render/gi/denoise", MenuAction::Select(i)) => {
          let c = gate_render::wesl_consts::gi_consts();
          let t = (*i).min((gate_render::gi::GiSettings::DENOISE_TIERS - 1) as usize) as u32;
          gi.denoise = t;
          let (name, cand, k, r) = match t {
            0 => ("关", c.gi_ss_cand_n, c.gi_ss_m_cap_k, c.gi_den_atrous_r_fast),
            1 => ("低", c.gi_ss_cand_n, c.gi_ss_m_cap_k, c.gi_den_atrous_r_fast),
            2 => ("中", c.gi_ss_cand_n, c.gi_ss_m_cap_k, c.gi_den_atrous_r),
            _ => ("高", c.gi_ss_cand_n_hq, c.gi_ss_m_cap_k_hq, c.gi_den_atrous_r),
          };
          info!("GI 降噪质量 → {name} cand={cand} win={k} atrous_r={r}");
        }
        ("render/gi/sun_bounce", MenuAction::Toggle(on)) => {
          gi.sun_bounce = *on;
          info!("GI 二次顶点太阳反弹 → {}", if *on { "on" } else { "off" });
        }
        ("render/gi/wal", MenuAction::Toggle(on)) => {
          gi.wal = *on;
          info!("GI 世界累积（WAL）→ {}", if *on { "on" } else { "off" });
        }
        ("render/gi/depth", MenuAction::Select(i)) => {
          let t = (*i).min((gate_render::gi::GiSettings::BOUNCE2_TIERS - 1) as usize) as u32;
          gi.depth = t;
          let (name, mult) = match t {
            0 => ("关", 0.0),
            1 => ("稀疏", 4.0),
            _ => ("全", 1.0),
          };
          info!("GI 二次弹射 → {name} mult={mult}");
        }
        ("render/gi/interval", MenuAction::Select(i)) => {
          gi.share = gate_render::gi::GiSettings::SHARE_CHOICES[(*i).min(3)];
          let c = gate_render::wesl_consts::gi_consts();
          let base = if gi.tier() >= 3 { c.gi_ss_cand_n_hq } else { c.gi_ss_cand_n };
          info!("GI 分帧 → ÷{}（基准候选 {} ⇒ 每帧 {}）", gi.share, base, (base / gi.share).max(1));
        }
        ("render/gi/realloc", MenuAction::Select(i)) => {
          let t = (*i).min((gate_render::gi::GiSettings::REALLOC_TIERS - 1) as usize) as u32;
          gi.realloc = t;
          let (name, lo, hi) = match t {
            0 => ("关", 1.0, 1.0),
            1 => ("弱", 0.70, 1.40),
            2 => ("中", 0.50, 2.00),
            _ => ("强", 0.25, 4.00),
          };
          info!("GI 重分配 → {name} 窗已满×{lo} 窗未满×{hi}");
        }
        ("render/refl/tier", MenuAction::Select(i)) => {
          let t = (*i).min((gate_render::ReflectionSettings::TIERS - 1) as usize) as u32;
          refl.tier = t;
          let (name, _cost) = match t {
            0 => ("关", "不发反射射线（只有 F0·(amb+gi) 近似）"),
            1 => ("直射", "0 条额外射线：反射命中点带太阳直射 + 自发光"),
            2 => ("直射+天空", "0 条额外射线：镜像里再带上太阳盘 / 光晕"),
            _ => ("直射+天空+阴影", "每个触发像素 +1 次 DDA：反射命中点补太阳 NEE"),
          };
          info!("镜面档位 → {t} {name}");
        }
        ("render/refl/nest", MenuAction::Select(i)) => {
          let n = gate_render::ReflectionSettings::NEST_CHOICES[(*i).min(3)];
          refl.nest = n;
          info!("镜面嵌套 → {n} 层");
        }
        ("render/exposure/enabled", MenuAction::Toggle(on)) => {
          eye.enabled = *on;
          info!(target: "gate", "自动曝光 → {}", if *on { "on" } else { "off" });
        }
        ("render/exposure/ev_up", MenuAction::Value(v)) => eye.ev_max = *v,
        ("render/exposure/ev_dn", MenuAction::Value(v)) => eye.ev_min = *v,
        ("render/exposure/tau_up", MenuAction::Value(v)) => eye.tau_brighten = v.max(0.01),
        ("render/exposure/tau_dn", MenuAction::Value(v)) => eye.tau_darken = v.max(0.01),
        ("render/exposure/key", MenuAction::Value(v)) => eye.key = v.max(0.001),
        _ => {}
      }
    },
  );

  world.add_observer(
    |ev: On<MenuActionEvent>,
     mut fog: ResMut<gate_render::FogSettings>,
     mut sky: ResMut<gate_render::SkySettings>| {
      match (ev.path.as_str(), &ev.action) {
        ("render/sky/blur/enabled", MenuAction::Toggle(on)) => {
          fog.enabled = *on;
          info!("光柱 → {}", if *on { "on" } else { "off" });
        }
        ("render/sky/blur/override", MenuAction::Toggle(on)) => {
          sky.override_blur = *on;
          info!("径向模糊：覆写 → {}", if *on { "on" } else { "off" });
        }
        ("render/sky/blur/strength", MenuAction::Value(v)) => {
          fog.strength = v.max(0.0);
          info!("光柱强度 → {v:.2}");
        }
        ("render/sky/blur/decay", MenuAction::Value(v)) => {
          fog.decay = v.clamp(0.05, 1.0);
          info!("光柱衰减 → {:.2}", fog.decay);
        }
        ("render/sky/blur/focus", MenuAction::Value(v)) => {
          fog.focus = v.clamp(1.0, 64.0);
          info!("光柱集中度 → {:.0}", fog.focus);
        }
        ("render/sky/disk/enabled", MenuAction::Toggle(on)) => {
          fog.body = *on;
          info!("日月外观 → {}", if *on { "on" } else { "off" });
        }
        ("render/sky/disk/override", MenuAction::Toggle(on)) => {
          sky.override_disk = *on;
          info!("天体盘：覆写 → {}", if *on { "on" } else { "off" });
        }
        ("render/sky/disk/radius", MenuAction::Value(v)) => {
          fog.sun_cone = v.to_radians();
          info!("天体盘：角径 → {v:.2}°");
        }
        ("render/sky/disk/halo", MenuAction::Value(v)) => {
          fog.halo = *v;
          info!("天体盘：光晕 → {v:.1}");
        }
        _ => {}
      }
    },
  );

  world.add_observer(|ev: On<MenuActionEvent>, mut sky: ResMut<gate_render::SkySettings>| {
    match (ev.path.as_str(), &ev.action) {
      ("render/sky/time/hour", MenuAction::Value(v)) => {
        sky.hour = v.rem_euclid(24.0);
        info!("天象：时刻 → {:.2}h {}", sky.hour, sky_report(&sky));
      }
      ("render/sky/time/date", MenuAction::Value(v)) => {
        sky.day_of_year = *v;
        info!("天象：年积日 → {:.0} {}", sky.day_of_year, sky_report(&sky));
      }
      ("render/sky/time/lat", MenuAction::Value(v)) => {
        sky.latitude_deg = v.clamp(-90.0, 90.0);
        info!("天象：纬度 → {:.1}° {}", sky.latitude_deg, sky_report(&sky));
      }
      ("render/sky/time/auto", MenuAction::Toggle(on)) => {
        sky.auto = *on;
        info!("自动流逝 → {} {}", if *on { "on" } else { "off" }, day_cycle(sky.hours_per_sec));
      }
      ("render/sky/time/speed", MenuAction::Value(v)) => {
        sky.hours_per_sec = v.max(0.0);
        info!("流逝速度 → {v:.2} 游戏小时/秒 {}", day_cycle(sky.hours_per_sec));
      }
      _ => {}
    }
  });

  world.add_observer(|ev: On<MenuActionEvent>, mut sky: ResMut<gate_render::SkySettings>| {
    if let ("render/sky/color/override", MenuAction::Toggle(on)) = (ev.path.as_str(), &ev.action) {
      sky.override_colors = *on;
      info!("颜色：覆写 → {}", if *on { "on" } else { "off" });
      return;
    }
    let MenuAction::Text(hex) = &ev.action else { return };
    let Some(c) = hex_srgb01(hex) else {
      debug!(target: "gate", "颜色输入未成形 {hex:?} → 忽略");
      return;
    };
    let name = match ev.path.as_str() {
      "render/sky/color/sun" => {
        sky.sun_color = c;
        "太阳"
      }
      "render/sky/color/moon" => {
        sky.moon_color = c;
        "月亮"
      }
      "render/sky/color/sky" => {
        sky.sky_color = c;
        "天空"
      }
      _ => return,
    };
    info!("{name}色 → #{hex}");
  });

  world.add_observer(
    |ev: On<MenuActionEvent>, mut mode: ResMut<CameraMode>, mut fly: ResMut<FlyCamera>| match (
      ev.path.as_str(),
      &ev.action,
    ) {
      ("player/camera/mode", MenuAction::Select(i)) => {
        *mode = if *i == 0 { CameraMode::Orbit } else { CameraMode::Fly };
      }
      ("player/camera/speed", MenuAction::Value(v)) => {
        fly.speed = *v * VOXEL_PER_METER;
        info!("飞行速度 → {:.2} m/s {:.0} v/s", v, fly.speed);
      }
      _ => {}
    },
  );

  world.add_observer(
    |ev: On<MenuActionEvent>,
     mut edit: ResMut<EditSettings>,
     q_menu: Query<&gate_ui::DebugMenu>,
     mut q_offsets: Query<(&gate_ui::menu::MenuItem, &mut gate_ui::TextInputValue)>,
     pbr_set: Option<Res<gate_render::PbrTextureSet>>,
     mut q_show: Query<&mut Visibility, With<ShowcaseRoot>>| {
      match (ev.path.as_str(), &ev.action) {
        ("game/edit/brush/shape", MenuAction::Select(i)) => {
          edit.shape = if *i == 0 { BrushShape::Sphere } else { BrushShape::Cube };
          info!("笔触形状 → {:?}", edit.shape);
        }
        ("game/edit/brush/size", MenuAction::Text(t)) => {
          if let Ok(v) = t.trim().parse::<u32>() {
            edit.size = v.max(EDIT_SIZE_MIN);
            let auto = edit.size as f32;
            write_brush_offset(&mut edit, auto, &mut q_offsets);
            info!("笔触大小 → {} vx；偏移距离自动 → {:.1}", edit.size, auto);
          }
        }
        ("game/edit/brush/offset", MenuAction::Text(t)) => match t.trim().parse::<f32>() {
          Ok(v) => {
            edit.offset = v.max(0.0);
            info!("笔触偏移距离 → {:.1} vx", edit.offset);
          }
          Err(_) => debug!(target: "gate", "偏移距离未成形 {t:?} → 忽略"),
        },
        ("game/edit/mat/color", MenuAction::Text(t)) => match parse_hex_color(t) {
          Some([r, g, b, _]) => {
            edit.mat.color = [r, g, b];
            log_material(&edit.mat);
          }
          None => debug!(target: "gate", "笔触颜色未成形 {t:?} → 忽略"),
        },
        ("game/edit/mat/emissive", MenuAction::Value(v)) => {
          edit.mat.emissive = v.round().clamp(0.0, 255.0) as u8;
          log_material(&edit.mat);
        }
        ("game/edit/mat/alpha", MenuAction::Value(v)) => {
          edit.mat.transmission = transparency_pct_to_transmission(*v);
          log_material(&edit.mat);
        }
        ("game/edit/mat/smooth", MenuAction::Value(v)) => {
          edit.mat.roughness = smooth_pct_to_roughness(*v);
          log_material(&edit.mat);
        }
        ("game/edit/mat/pbr", MenuAction::Toggle(on)) => {
          edit.mat.pbr = *on;
          info!(
            target: "gate",
            "材质变体 → {}；{}",
            if *on { "PBR" } else { "平凡" },
            edit.mat.summary(),
          );
        }
        ("game/edit/mat/pbr_asset", MenuAction::Select(i)) => {
          let name = menu_pbr_asset(&q_menu).unwrap_or_else(|| "?".to_string());
          let (slot, how) = match pbr_set.as_deref().and_then(|s| s.slot_of(&name)) {
            Some(slot) => (slot, "按 PbrTextureSet 解析"),
            None => (*i as u32, "贴图集未就绪/未收录，回落选项下标"),
          };
          edit.mat.asset_slot = slot;
          info!(
            target: "gate",
            "材质 → PBR 资产槽 {slot}={name} {how}；{}",
            edit.mat.summary(),
          );
        }
        ("game/edit/mat/metal", MenuAction::Toggle(on)) => {
          edit.mat.metallic = metal_toggle_to_metallic(*on);
          log_material(&edit.mat);
        }
        ("ui/showcase", MenuAction::Toggle(on)) => {
          if let Ok(mut vis) = q_show.single_mut() {
            *vis = if *on { Visibility::Visible } else { Visibility::Hidden };
          }
        }
        _ => {}
      }
    },
  );

  world.add_observer(
    |ev: On<MenuActionEvent>,
     mut scene: ResMut<gate_render::VoxelScene>,
     mut dump: ResMut<gate_render::VoxelDumpRequest>,
     mut stream: ResMut<crate::infinite_cubes::Streaming>,
     mut base: ResMut<gate_render::BaseSettings>,
     pbr: Option<Res<gate_render::PbrTextureSet>>,
     cam: Option<Res<gate_render::DdaCameraConfig>>,
     q_menu: Query<&gate_ui::DebugMenu>| {
      match (ev.path.as_str(), &ev.action) {
        (WORLD_MODEL_PATH, MenuAction::Select(_)) => {
          let name = menu_world_model(&q_menu).unwrap_or_else(|| "?".to_string());
          info!("世界模型 → {name}");
        }
        (WORLD_RELOAD_PATH, MenuAction::Button(_)) => {
          let Some(name) = menu_world_model(&q_menu) else {
            warn!("重载世界：读不到模型选择（菜单未就绪）→ 忽略");
            return;
          };
          let t0 = std::time::Instant::now();
          match crate::scene::reload_world(
            &mut scene,
            &name,
            &crate::scene::pbr_asset_ids(pbr.as_deref()),
            cam.map(|c| c.position_world.as_ivec3()),
            &mut stream,
          ) {
            Ok(_info) => info!("世界重载 → {name} {:?}", t0.elapsed()),
            Err(e) => warn!("重载世界失败 {name}：{e} → 原世界不变"),
          }
        }
        (WORLD_DUMP_PATH, MenuAction::Button(_)) => {
          dump.arm();
          info!("数据转储 → 请求");
        }
        (WORLD_LOD_PATH, MenuAction::Button(i)) => match i {
          0 => {
            let city = stream
              .source()
              .and_then(|s| s.clone_as_any())
              .and_then(|a| a.downcast::<crate::mc::source::McCity>().ok());
            match city {
              Some(city) => crate::mc::start_lod_build(city),
              None => warn!("LOD 构建：当前世界不是 MC 地图（换到 mc_map 再点）→ 忽略"),
            }
          }
          _ => crate::mc::cancel_lod_build(),
        },
        (WORLD_STREAM_PAUSE_PATH, MenuAction::Toggle(on)) => {
          stream.paused = *on;
          info!("流式加载 → {}", if *on { "暂停" } else { "继续" });
        }
        (WORLD_LOAD_RADIUS_PATH, MenuAction::Value(v)) => stream.load_radius = v.round() as i32,
        (WORLD_UNLOAD_RADIUS_PATH, MenuAction::Value(v)) => stream.unload_radius = v.round() as i32,
        (WORLD_COARSE_RADIUS_PATH, MenuAction::Value(v)) => stream.coarse_radius = v.round() as i32,
        (WORLD_COARSE_HEIGHT_PATH, MenuAction::Value(v)) => stream.coarse_height = v.round() as i32,
        (WORLD_MOUNT_WORDS_PATH, MenuAction::Value(v)) => {
          stream.mount_words = (v.round().max(0.0) as usize) * 1024;
        }
        (WORLD_MOUNT_COUNT_PATH, MenuAction::Value(v)) => {
          stream.mount_count = v.round().max(1.0) as usize;
        }
        (WORLD_REQUESTS_LOAD_PATH, MenuAction::Toggle(on)) => {
          stream.requests_load = *on;
          info!("请求装载 → {}", if *on { "on" } else { "off（退回纯半径）" });
        }
        (WORLD_REQUEST_MB_PATH, MenuAction::Value(v)) => {
          stream.request_bytes = (v.round().max(0.0) as usize) * 1024 * 1024;
        }
        (WORLD_FAR_LEVELS_PATH, MenuAction::Value(v)) => {
          stream.far_levels_max = v.round().clamp(0.0, 3.0) as usize;
          info!("远场级上限 → L{}", stream.far_levels_max);
        }
        (WORLD_VOL_TINT_PATH, MenuAction::Toggle(on)) => {
          base.vol_tint = *on;
          info!("卷着色 → {}", if *on { "on" } else { "off" });
        }
        _ => {}
      }
    },
  );
}

pub(crate) fn world_model_name(model: &MenuFile) -> Option<String> {
  match model.node(&split(WORLD_MODEL_PATH)) {
    Some(MenuNode::Dropdown { options, selected, .. }) => options.get(*selected).cloned(),
    _ => None,
  }
}

fn menu_world_model(q_menu: &Query<&gate_ui::DebugMenu>) -> Option<String> {
  world_model_name(&q_menu.single().ok()?.model)
}

fn menu_pbr_asset(q_menu: &Query<&gate_ui::DebugMenu>) -> Option<String> {
  match q_menu.single().ok()?.model.node(&split(EDIT_PBR_ASSET_PATH)) {
    Some(MenuNode::Dropdown { options, selected, .. }) => options.get(*selected).cloned(),
    _ => None,
  }
}

fn log_material(mat: &BrushMaterial) {
  info!(target: "gate", "材质 → {}", mat.summary());
}

fn sky_report(sky: &gate_render::SkySettings) -> String {
  let alt = gate_render::sun_altitude_deg(sky.hour, sky.day_of_year, sky.latitude_deg);
  format!("太阳高度角 {alt:.1}°、主光 = {}", if alt >= 0.0 { "太阳" } else { "反日点的月亮" })
}

fn day_cycle(hours_per_sec: f32) -> String {
  if hours_per_sec <= 0.0 {
    "速度为 0 ⇒ 时刻不动".to_string()
  } else {
    format!("一昼夜 {:.1} 分钟", 24.0 / hours_per_sec / 60.0)
  }
}

fn hex_srgb01(hex: &str) -> Option<[f32; 3]> {
  let [r, g, b, _] = parse_hex_color(hex)?;
  Some([f32::from(r) / 255.0, f32::from(g) / 255.0, f32::from(b) / 255.0])
}

pub(crate) fn sync_sky_menu(
  sky: Res<gate_render::SkySettings>,
  fog: Res<gate_render::FogSettings>,
  mut commands: Commands,
  mut q_menu: Query<&mut gate_ui::DebugMenu>,
  mut q_sliders: Query<(&gate_ui::menu::MenuItem, &mut gate_ui::SliderValue)>,
  q_controls: Query<(
    Entity,
    &gate_ui::menu::MenuItem,
    Has<bevy::ui::Checked>,
    Has<gate_ui::widgets::UiDisabled>,
  )>,
  q_swatches: Query<(Entity, &gate_ui::menu::MenuColorSwatch, Has<gate_ui::widgets::UiDisabled>)>,
) {
  let mut want: Vec<(&str, f32)> = Vec::with_capacity(6);
  if sky.auto {
    want.push((SKY_HOUR_PATH, sky.hour));
  }
  if !sky.override_blur {
    want.push((BLUR_STRENGTH_PATH, fog.strength()));
    want.push((BLUR_DECAY_PATH, fog.decay()));
    want.push((BLUR_FOCUS_PATH, fog.focus()));
  }
  if !sky.override_disk {
    want.push((DISK_RADIUS_PATH, fog.sun_cone.max(0.0).to_degrees()));
    want.push((DISK_HALO_PATH, fog.halo.max(0.0)));
  }
  for (item, mut v) in &mut q_sliders {
    if let Some((_, value)) = want.iter().find(|(path, _)| *path == item.path)
      && v.0 != *value
    {
      v.0 = *value;
    }
  }

  let overrides = [
    (BLUR_OVERRIDE_PATH, sky.override_blur),
    (DISK_OVERRIDE_PATH, sky.override_disk),
    (COLOR_OVERRIDE_PATH, sky.override_colors),
  ];
  let disable = [
    (BLUR_STRENGTH_PATH, !sky.override_blur),
    (BLUR_DECAY_PATH, !sky.override_blur),
    (BLUR_FOCUS_PATH, !sky.override_blur),
    (DISK_RADIUS_PATH, !sky.override_disk),
    (DISK_HALO_PATH, !sky.override_disk),
    (COLOR_SUN_PATH, !sky.override_colors),
    (COLOR_MOON_PATH, !sky.override_colors),
    (COLOR_SKY_PATH, !sky.override_colors),
    (SKY_HOUR_PATH, sky.auto),
  ];
  for (e, item, checked, disabled) in &q_controls {
    if let Some((_, on)) = overrides.iter().find(|(path, _)| *path == item.path)
      && *on != checked
    {
      if *on {
        commands.entity(e).insert(bevy::ui::Checked);
      } else {
        commands.entity(e).remove::<bevy::ui::Checked>();
      }
    }
    if let Some((_, off)) = disable.iter().find(|(path, _)| *path == item.path)
      && disabled != *off
    {
      if *off {
        commands.entity(e).insert(gate_ui::widgets::UiDisabled);
      } else {
        commands.entity(e).remove::<gate_ui::widgets::UiDisabled>();
      }
    }
  }
  for (e, swatch, disabled) in &q_swatches {
    if let Some((_, off)) = disable.iter().find(|(path, _)| *path == swatch.path)
      && disabled != *off
    {
      if *off {
        commands.entity(e).insert(gate_ui::widgets::UiDisabled);
      } else {
        commands.entity(e).remove::<gate_ui::widgets::UiDisabled>();
      }
    }
  }

  let Ok(mut menu) = q_menu.single_mut() else { return };
  for (path, value) in &want {
    if let Some(MenuNode::Slider { value: cur, .. }) = menu.model.node_mut(&split(path)) {
      *cur = *value;
    }
  }
  for (path, on) in overrides {
    if let Some(MenuNode::Toggle { checked, .. }) = menu.model.node_mut(&split(path)) {
      *checked = on;
    }
  }
  for (path, disabled) in disable {
    if let Some(node) = menu.model.node_mut(&split(path)) {
      node.set_disabled(disabled);
    }
  }
}

fn write_brush_offset(
  edit: &mut EditSettings,
  offset: f32,
  q: &mut Query<(&gate_ui::menu::MenuItem, &mut gate_ui::TextInputValue)>,
) {
  edit.offset = offset.max(0.0);
  let text = format!("{:.1}", edit.offset);
  for (item, mut tv) in q.iter_mut() {
    if item.path == EDIT_OFFSET_PATH && tv.0 != text {
      tv.0 = text.clone();
    }
  }
}

pub(crate) fn sync_video_menu(
  scale: Res<gate_render::RenderScale>,
  mut commands: Commands,
  mut q_menu: Query<&mut gate_ui::DebugMenu>,
  q_aa: Query<(Entity, &gate_ui::menu::MenuItem, Has<gate_ui::widgets::UiDisabled>)>,
) {
  let want = scale.factor != 1;
  let Ok(mut menu) = q_menu.single_mut() else { return };
  if let Some(node) = menu.model.node_mut(&split(VIDEO_AA_PATH)) {
    node.set_disabled(want);
  }
  for (e, item, disabled) in &q_aa {
    if item.path != VIDEO_AA_PATH || disabled == want {
      continue;
    }
    if want {
      commands.entity(e).insert(gate_ui::widgets::UiDisabled);
    } else {
      commands.entity(e).remove::<gate_ui::widgets::UiDisabled>();
    }
  }
}

pub(crate) fn sync_edit_menu(
  edit: Res<EditSettings>,
  mut commands: Commands,
  mut q_menu: Query<&mut gate_ui::DebugMenu>,
  q_controls: Query<(Entity, &gate_ui::menu::MenuItem, Has<gate_ui::widgets::UiDisabled>)>,
  q_swatches: Query<(Entity, &gate_ui::menu::MenuColorSwatch, Has<gate_ui::widgets::UiDisabled>)>,
) {
  let want = edit.mat.pbr;
  let Ok(mut menu) = q_menu.single_mut() else { return };
  for path in EDIT_MATERIAL_PATHS {
    if let Some(node) = menu.model.node_mut(&split(path)) {
      node.set_disabled(want);
    }
  }
  for (e, item, disabled) in &q_controls {
    if !EDIT_MATERIAL_PATHS.contains(&item.path.as_str()) || disabled == want {
      continue;
    }
    if want {
      commands.entity(e).insert(gate_ui::widgets::UiDisabled);
    } else {
      commands.entity(e).remove::<gate_ui::widgets::UiDisabled>();
    }
  }
  for (e, swatch, disabled) in &q_swatches {
    if !EDIT_MATERIAL_PATHS.contains(&swatch.path.as_str()) || disabled == want {
      continue;
    }
    if want {
      commands.entity(e).insert(gate_ui::widgets::UiDisabled);
    } else {
      commands.entity(e).remove::<gate_ui::widgets::UiDisabled>();
    }
  }
}

pub(crate) fn camera_info_tick(
  time: Res<Time>,
  orbit: Res<gate_render::OrbitCamera>,
  mode: Res<CameraMode>,
  fly: Res<FlyCamera>,
  mut q_values: Query<(&gate_ui::MenuTextValue, &mut Text)>,
  mut acc: Local<f32>,
) {
  *acc += time.delta_secs();
  if *acc < CAM_INFO_REFRESH_SECS {
    return;
  }
  *acc = 0.0;
  let eye = match *mode {
    CameraMode::Orbit => orbit.eye(),
    CameraMode::Fly => fly.pos,
  };
  let pos = t!("menu.camera.pos.value", v = format!("({:.1}, {:.1}, {:.1})", eye.x, eye.y, eye.z))
    .to_string();
  let dir = t!(
    "menu.camera.dir.value",
    yaw = format!("{:.1}", orbit.yaw.to_degrees()),
    pitch = format!("{:.1}", orbit.pitch.to_degrees())
  )
  .to_string();
  for (value, mut text) in &mut q_values {
    let s = match value.path.as_str() {
      "player/camera/pos" => &pos,
      "player/camera/dir" => &dir,
      _ => continue,
    };
    if text.0 != *s {
      text.0 = s.clone();
    }
  }
}

pub(crate) fn lod_state_tick(
  time: Res<Time>,
  stream: Res<crate::infinite_cubes::Streaming>,
  mut q_values: Query<(&gate_ui::MenuTextValue, &mut Text)>,
  mut acc: Local<f32>,
) {
  *acc += time.delta_secs();
  if *acc < CAM_INFO_REFRESH_SECS {
    return;
  }
  *acc = 0.0;
  let city = stream
    .source()
    .and_then(|s| s.clone_as_any())
    .and_then(|a| a.downcast::<crate::mc::source::McCity>().ok());
  let s = match city {
    None => "—".to_string(),
    Some(city) => match city.lod() {
      Some(v) => t!(
        "menu.game.world.lod_state.ready",
        mb = format!("{:.0}", v.heap_bytes() as f64 / 1048576.0)
      )
      .to_string(),
      None => match crate::mc::lod_build_state() {
        (true, pct) => t!("menu.game.world.lod_state.building", pct = pct).to_string(),
        _ => t!("menu.game.world.lod_state.none").to_string(),
      },
    },
  };
  for (value, mut text) in &mut q_values {
    if value.path == "game/world/lod_state" && text.0 != s {
      text.0 = s.clone();
    }
  }
}

#[allow(clippy::type_complexity)]
pub(crate) fn fps_overlay_tick(
  time: Res<Time>,
  visible: Res<FpsOverlayVisible>,
  pace: Res<gate_render::profiler::FramePace>,
  mut window: ResMut<FpsWindow>,
  mut since_update: Local<f32>,
  mut q_root: Query<&mut Visibility, With<FpsOverlay>>,
  mut q_text: Query<&mut Text, With<FpsOverlayText>>,
) {
  if let Ok(mut vis) = q_root.single_mut() {
    let target = if visible.0 { Visibility::Visible } else { Visibility::Hidden };
    if *vis != target {
      *vis = target;
    }
  }
  if !visible.0 {
    return;
  }
  let now = time.elapsed_secs();
  let presented = pace.presented.load(Ordering::Relaxed);
  if let Some((t0, c0)) = window.last {
    let dt = now - t0;
    let dc = presented.saturating_sub(c0);
    if dc > 0 && dt > 0.0 {
      let per = dt / dc as f32;
      for _ in 0..dc.min(FPS_MAX_FRAMES_PER_TICK as u64) {
        window.intervals.push_back(per);
      }
      window.last = Some((now, presented));
    }
  } else {
    window.last = Some((now, presented));
  }
  let mut sum: f32 = window.intervals.iter().sum();
  while sum > FPS_WINDOW_SECS && window.intervals.len() > 1 {
    let old = window.intervals.pop_front().unwrap_or(0.0);
    sum -= old;
  }
  *since_update += time.delta_secs();
  if *since_update < FPS_UPDATE_SECS {
    return;
  }
  *since_update = 0.0;
  let cur = window.intervals.back().copied().map_or(0.0, rate_of);
  let avg = if sum > 0.0 { window.intervals.len() as f32 / sum } else { 0.0 };
  let min_dt = window.intervals.iter().cloned().fold(0.0f32, f32::max);
  let max_dt = window.intervals.iter().cloned().fold(f32::MAX, f32::min);
  let min = rate_of(min_dt);
  let max = if max_dt < f32::MAX { rate_of(max_dt) } else { 0.0 };
  let text = format!("({:>3}, {:>3}, {:>3}, {:>3})", fps3(cur), fps3(avg), fps3(min), fps3(max));
  bevy::log::debug!("FPS[1s] {avg:.0} 低 {min:.0} 高 {max:.0}（当前 {cur:.0}）");
  if let Ok(mut t) = q_text.single_mut()
    && t.0 != text
  {
    t.0 = text;
  }
}

fn rate_of(dt: f32) -> f32 {
  if dt > 0.0 { 1.0 / dt } else { 0.0 }
}

fn fps3(v: f32) -> u32 {
  (v.round() as u32).min(999)
}

pub(crate) fn debug_menu_toggle(
  keys: Res<ButtonInput<KeyCode>>,
  mut q: Query<&mut Visibility, With<DebugMenuRoot>>,
) {
  if keys.just_pressed(KeyCode::F3) {
    for mut vis in &mut q {
      *vis = if *vis == Visibility::Hidden { Visibility::Visible } else { Visibility::Hidden };
    }
  }
}
