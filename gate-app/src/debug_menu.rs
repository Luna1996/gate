//! DebugMenu：gate-app 的调试菜单（gate-ui 通用 menu 模块的第一个使用者）。
//! UI 结构与控件缺省值每次启动都从只读资源 `assets/ui/debug_menu.toml` 读；运行期改动过的值
//! 则来自 `data/config.toml`（见 `crate::config`），由本文件的 `load_menu` 合并。
//! 另含 `MenuActionEvent` → 各调试资源观察者、相机信息与 FPS 覆盖层刷新。

use std::collections::VecDeque;

use bevy::prelude::*;
use bevy::window::{
  MonitorSelection, PresentMode, PrimaryWindow, Window, WindowMode, WindowPosition,
};
use rust_i18n::t;

use gate_ui::widgets::{LabelConfig, LabelStyle, label, px};
use gate_ui::{
  DebugMenuRoot, MenuAction, MenuActionEvent, MenuFile, MenuNode, UiCtx, UiTranslator,
  parse_hex_color, spawn_debug_menu,
};

use crate::camera::{CameraMode, FlyCamera};
use crate::config::Config;
use crate::consts::{
  CAM_INFO_REFRESH_SECS, EDIT_SIZE_MIN, FPS_WINDOW_SECS, HALF_RES_FACTOR, VOXEL_PER_METER,
};
use crate::edit::{BrushShape, EditSettings, opacity_pct_to_transmission, smooth_pct_to_roughness};
use crate::showcase::ShowcaseRoot;

/// 菜单 TOML 相对 assets 目录的路径（UI 结构与控件缺省值的唯一来源）
pub const MENU_TOML_PATH: &str = "ui/debug_menu.toml";

/// 「世界」页模型下拉的节点路径（选项由 `apply_world_model_options` 按磁盘内容填）
pub const WORLD_MODEL_PATH: &str = "game/world/model";
/// 「世界」页「重载世界」按钮的节点路径（空 label 的按钮组 = 整行按钮）
pub const WORLD_RELOAD_PATH: &str = "game/world/reload";

/// 读 UI 结构与控件缺省值：每次都读只读资源 `assets/ui/debug_menu.toml`（结构与缺省值的唯一
/// 来源），再把 `data/config.toml` 里的值覆盖上去 —— 配置里没有的控件保留缺省值，
/// 配置里有而结构里没有的路径（控件被删了）直接忽略；随后按磁盘内容重填世界模型下拉的选项。
pub fn load_menu(config: &Config) -> MenuFile {
  let path = gate_render::assets_dir().join(MENU_TOML_PATH);
  let mut model = match std::fs::read_to_string(&path) {
    Ok(src) => match MenuFile::from_toml(&src) {
      Ok(m) => {
        info!("debug menu layout loaded from {}", path.display());
        m
      }
      Err(e) => {
        warn!("debug menu TOML 解析失败（{e}）；菜单为空");
        MenuFile::default()
      }
    },
    Err(e) => {
      warn!("debug menu TOML 缺失（{e}）；菜单为空");
      MenuFile::default()
    }
  };
  // 下拉选项取自磁盘扫描，必须先于配置值套用（配置里的选中项按名字找回）
  apply_world_model_options(&mut model);
  let applied = model.apply_values(&config.menu);
  // 窗口位置 / 收起 / 停留路径只存在配置里
  model.window = config.window.clone();
  model.sanitize();
  info!("debug menu 控件值：config 命中 {applied} / {} 项", config.menu.len());
  model
}

/// 把 `WORLD_MODEL_PATH` 下拉的选项换成 `assets/vox` 下实际存在的模型（见
/// `crate::vox_scene::scan_vox_models`）；选中项按名字找回，找不到 → `nuke` → 第一项。
fn apply_world_model_options(model: &mut MenuFile) {
  let options = crate::vox_scene::scan_vox_models();
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

/// UI 已生成的守卫标记（debug_menu_setup 的存在性守卫）
#[derive(Component)]
pub(crate) struct DebugUiRoot;

/// 右上角 FPS 覆盖层根
#[derive(Component)]
pub(crate) struct FpsOverlay;

/// FPS 覆盖层文本
#[derive(Component)]
pub(crate) struct FpsOverlayText;

/// FPS 覆盖层是否显示（`video/fps` 开关）
#[derive(Resource, Default, Debug, Clone, Copy)]
pub struct FpsOverlayVisible(pub bool);

/// 进无边框全屏前的窗口尺寸/位置（退出全屏时复原）；`None` = 当前不在全屏。
/// winit 多数平台会自己复原窗口，但位置在部分后端会丢，故显式存一份兜底。
#[derive(Resource, Default)]
struct WindowedRestore(Option<(UVec2, Option<IVec2>)>);

/// 每帧 delta 的 1s 滚动窗口（FPS 统计）
#[derive(Resource, Default)]
pub struct FpsWindow(VecDeque<f32>);

/// 建 DebugMenu + FPS 覆盖层 + 回调观察者；主题/字体就绪后由 `crate::debug_ui_setup` 调用一次。
pub(crate) fn spawn_debug_menu_ui(world: &mut World, ctx: &UiCtx) {
  let model = {
    let config = world.resource::<Config>();
    load_menu(config)
  };
  // 文案解析器取资源里的（语言切换只需 bump 版本，见 sync_ui_locale）
  let translate = world.resource::<UiTranslator>().handle();
  let ctx = UiCtx::new(ctx.theme, ctx.font).with_icon_font(ctx.icon_font).with_translate(translate);
  let handle = spawn_debug_menu(world, &ctx, model);
  // 顺序：先挂回调，再把最终控件值按节点重放成 `MenuActionEvent`（资源映射只存在于观察者里），
  // 最后建覆盖层（它读的 `FpsOverlayVisible` 已由初值重放写好）。
  register_callbacks(world);
  apply_initial_state(world, handle.root);
  spawn_fps_overlay(world, &ctx);
}

/// 语言切换后让 gate-ui 重解析全部 keyed 文本；解析闭包读 rust-i18n 当前 locale，故只需 bump 版本。
pub(crate) fn sync_ui_locale(mut translator: ResMut<UiTranslator>, mut last: Local<String>) {
  let now = rust_i18n::locale().to_string();
  if now != *last {
    info!("UI 语言 → {now}");
    *last = now;
    translator.bump();
  }
}

/// 右上角 FPS 覆盖层：黑底白字，格式 "(cur, avg, min, max)"，每帧更新
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

/// 用菜单模型的**最终控件值**初始化各调试资源：按节点重放成 `MenuActionEvent`
/// （「菜单值 → 资源」的映射只在 `register_callbacks` 的观察者里写一次）。
fn apply_initial_state(world: &mut World, root: Entity) {
  let Some(model) = gate_ui::menu_model(world).cloned() else { return };
  let mut actions = Vec::new();
  collect_actions(&model.items, "", &mut actions);
  let n = actions.len();
  for (path, action) in actions {
    world.trigger(MenuActionEvent { entity: root, path, action });
  }
  info!(target: "gate", "debug menu 初值已应用：{n} 项");
}

/// 递归收集「带初值」的节点 → (id 路径, 动作)；`SubMenu` 只递归下去，`Buttons` / `Text` 无初值。
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

/// 路径字符串 → id 段
fn split(path: &str) -> Vec<String> {
  path.split('/').filter(|s| !s.is_empty()).map(|s| s.to_string()).collect()
}

// 统一事件按 path 分派
fn register_callbacks(world: &mut World) {
  world.init_resource::<WindowedRestore>();
  world.add_observer(
    |ev: On<MenuActionEvent>,
     mut q_win: Query<&mut Window, With<PrimaryWindow>>,
     mut fps: ResMut<FpsOverlayVisible>,
     mut scale: ResMut<gate_render::RenderScale>,
     mut post: ResMut<gate_render::PostFxSettings>,
     mut restore: ResMut<WindowedRestore>| {
      let MenuAction::Toggle(on) = ev.action else { return };
      match ev.path.as_str() {
        "video/vsync" => {
          let Ok(mut win) = q_win.single_mut() else { return };
          win.present_mode = if on { PresentMode::Fifo } else { PresentMode::AutoNoVsync };
          info!("VSync {} → present_mode {:?}", if on { "on" } else { "off" }, win.present_mode);
        }
        "video/fps" => {
          fps.0 = on;
          info!("FPS 覆盖层 → {}", if on { "on" } else { "off" });
        }
        // 全屏：窗口 ↔ 无边框全屏（非独占，不动显示模式/刷新率）
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
        // 半分辨率渲染：只改降采样倍数，目标纹理由 responsive 系统下一帧重建
        "video/half_res" => {
          scale.factor = if on { HALF_RES_FACTOR } else { 1 };
          info!(
            "半分辨率渲染 → {}（渲染 {}x{} → 放大到窗口）",
            if on { "on" } else { "off" },
            scale.size.x,
            scale.size.y
          );
        }
        // 抗锯齿：FXAA 在最终 blit 里做（换 fragment 入口，见 PostFxSettings）
        "video/aa" => {
          post.fxaa = on;
          info!("抗锯齿（FXAA）→ {}", if on { "on" } else { "off" });
        }
        _ => {}
      }
    },
  );

  world.add_observer(
    |ev: On<MenuActionEvent>,
     mut gi: ResMut<gate_render::gi::GiSettings>,
     mut eye: ResMut<gate_render::EyeAdaptSettings>| {
      match (ev.path.as_str(), &ev.action) {
        ("render/gi/enabled", MenuAction::Toggle(on)) => {
          gi.enabled = *on;
          info!("GI → {}", if *on { "on" } else { "off" });
        }
        // 半分辨率开关：勾选 = 半分辨率、取消 = 全分辨率（**两档都跑 GI**，关掉不等于关 GI）
        ("render/gi/half", MenuAction::Toggle(on)) => {
          gi.gi_div = if *on { 2 } else { 1 };
          info!("GI 分辨率 → {}（网格边长 = 渲染分辨率 / {}）", if *on { "半分辨率" } else { "全分辨率" }, gi.gi_div);
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
    |ev: On<MenuActionEvent>, mut mode: ResMut<CameraMode>, mut fly: ResMut<FlyCamera>| match (
      ev.path.as_str(),
      &ev.action,
    ) {
      ("player/camera/mode", MenuAction::Select(i)) => {
        *mode = if *i == 0 { CameraMode::Orbit } else { CameraMode::Fly };
        info!("相机模式 → {:?}", *mode);
      }
      ("player/camera/speed", MenuAction::Value(v)) => {
        fly.speed = *v * VOXEL_PER_METER;
        info!("飞行速度 → {:.2} m/s（{:.0} v/s）", v, fly.speed);
      }
      _ => {}
    },
  );

  // 材质控件只改当前笔触材质，不碰调色板；材质 → 槽的分配在落笔时（`edit::material_slot`）。
  world.add_observer(
    |ev: On<MenuActionEvent>,
     mut edit: ResMut<EditSettings>,
     mut q_show: Query<&mut Visibility, With<ShowcaseRoot>>| {
      match (ev.path.as_str(), &ev.action) {
        ("game/edit/shape", MenuAction::Select(i)) => {
          edit.shape = if *i == 0 { BrushShape::Sphere } else { BrushShape::Cube };
          info!("笔触形状 → {:?}", edit.shape);
        }
        ("game/edit/size", MenuAction::Text(t)) => {
          if let Ok(v) = t.trim().parse::<u32>() {
            edit.size = v.max(EDIT_SIZE_MIN);
            // 跨度 = 2·size-1（saturating 防日志侧溢出）
            let span = edit.size.saturating_mul(2).saturating_sub(1);
            info!("笔触大小 → {} vx（跨度 {}）", edit.size, span);
          }
        }
        ("game/edit/color", MenuAction::Text(t)) => match parse_hex_color(t) {
          Some([r, g, b, _]) => {
            edit.mat.color = [r, g, b];
            info!(target: "gate", "笔触颜色 → {}", edit.mat.hex());
          }
          // 输入框是自由文本：解析失败（半截输入）就忽略，不打断输入
          None => debug!(target: "gate", "笔触颜色输入未成形 → {t:?}（忽略）"),
        },
        ("game/edit/emissive", MenuAction::Value(v)) => {
          edit.mat.emissive = v.round().clamp(0.0, 255.0) as u8;
          info!(target: "gate", "自发光 → {}", edit.mat.emissive);
        }
        ("game/edit/alpha", MenuAction::Value(v)) => {
          edit.mat.transmission = opacity_pct_to_transmission(*v);
          info!(target: "gate", "透明度 → {v:.0}%（透射率 {}）", edit.mat.transmission);
        }
        ("game/edit/smooth", MenuAction::Value(v)) => {
          edit.mat.roughness = smooth_pct_to_roughness(*v);
          info!(target: "gate", "光滑度 → {v:.0}%（粗糙度 {}）", edit.mat.roughness);
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

  // 下拉只改模型（退出时随存档持久化）；「重载世界」按钮才真正换世界（同步阻塞），失败只 warn、
  // 原世界不变；成功后全量重建 + GPU 上传由 VoxelScene.demo_force_full_rebuild 驱动。
  world.add_observer(
    |ev: On<MenuActionEvent>,
     mut scene: ResMut<gate_render::VoxelScene>,
     q_menu: Query<&gate_ui::DebugMenu>| {
      match (ev.path.as_str(), &ev.action) {
        (WORLD_MODEL_PATH, MenuAction::Select(_)) => {
          let name = menu_world_model(&q_menu).unwrap_or_else(|| "?".to_string());
          info!("世界模型 → {name}（点「重载世界」生效）");
        }
        (WORLD_RELOAD_PATH, MenuAction::Button(_)) => {
          let Some(name) = menu_world_model(&q_menu) else {
            warn!("重载世界：读不到模型下拉的选择（菜单未就绪），已忽略");
            return;
          };
          let t0 = std::time::Instant::now();
          match crate::scene::reload_world(&mut scene, &name) {
            Ok(info) => info!(
              "世界已重载：{name}.vox instances={} written={} dropped={} aabb=[{}]-[{}] ({:?})",
              info.instances_used,
              info.voxels_written,
              info.voxels_dropped,
              info.aabb_min,
              info.aabb_max,
              t0.elapsed(),
            ),
            Err(e) => warn!("重载世界失败（{name}.vox）：{e}（保持原世界不变）"),
          }
        }
        _ => {}
      }
    },
  );
}

/// 读「世界」页下拉当前选中的模型名（无该节点 / 选项为空 → None）；
/// 启动时由 `scene::setup` 用 `load_menu` 的模型读同一个值。
pub(crate) fn world_model_name(model: &MenuFile) -> Option<String> {
  match model.node(&split(WORLD_MODEL_PATH)) {
    Some(MenuNode::Dropdown { options, selected, .. }) => options.get(*selected).cloned(),
    _ => None,
  }
}

/// 从菜单组件里读选中的模型名（无菜单 → None）
fn menu_world_model(q_menu: &Query<&gate_ui::DebugMenu>) -> Option<String> {
  world_model_name(&q_menu.single().ok()?.model)
}

/// 纯文本行（相机位置/角度）的值：每 `CAM_INFO_REFRESH_SECS` 刷新一次。
/// 只改写值标签（`gate_ui::MenuTextValue`），左侧名称不动。
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
  // 数字先 format! 好再塞占位符：语言切换不改变数字列宽
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

/// 右上角 FPS：1s 窗口内统计 当前/平均/最低/最高，每帧更新
#[allow(clippy::type_complexity)]
pub(crate) fn fps_overlay_tick(
  time: Res<Time>,
  visible: Res<FpsOverlayVisible>,
  mut window: ResMut<FpsWindow>,
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
  let dt = time.delta_secs();
  if dt > 0.0 {
    window.0.push_back(dt);
  }
  let mut sum = 0.0f32;
  for &d in window.0.iter() {
    sum += d;
  }
  while sum > FPS_WINDOW_SECS
    && let Some(old) = window.0.pop_front()
  {
    sum -= old;
  }
  let mut min_dt = f32::MAX;
  let mut max_dt = 0.0f32;
  for &d in window.0.iter() {
    min_dt = min_dt.min(d);
    max_dt = max_dt.max(d);
  }
  let cur = if dt > 0.0 { 1.0 / dt } else { 0.0 };
  let avg = if sum > 0.0 { window.0.len() as f32 / sum } else { 0.0 };
  let min = if max_dt > 0.0 { 1.0 / max_dt } else { 0.0 };
  let max = if min_dt < f32::MAX && min_dt > 0.0 { 1.0 / min_dt } else { 0.0 };
  let text = format!("({:>3}, {:>3}, {:>3}, {:>3})", fps3(cur), fps3(avg), fps3(min), fps3(max));
  if let Ok(mut t) = q_text.single_mut()
    && t.0 != text
  {
    t.0 = text;
  }
}

/// fps → 3 位宽显示值（上限 999）
fn fps3(v: f32) -> u32 {
  (v.round() as u32).min(999)
}

/// F3 切换整个 DebugMenu 显隐（只影响菜单，不动 showcase）
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
