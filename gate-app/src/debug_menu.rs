//! DebugMenu：gate-app 的调试菜单（gate-ui 通用 menu 模块的第一个使用者）。
//! 本文件提供 `default_menu` 默认菜单树（= `assets/ui/debug_menu.toml`）、TOML 初值读写、
//! `MenuActionEvent` → 各调试资源观察者、相机信息与 FPS 覆盖层刷新。

use std::collections::VecDeque;

use bevy::prelude::*;
use bevy::window::{
  MonitorSelection, PresentMode, PrimaryWindow, Window, WindowMode, WindowPosition,
};
use rust_i18n::t;

use gate_ui::widgets::{LabelConfig, LabelStyle, label, px};
use gate_ui::{
  DebugMenuRoot, InputField, MenuAction, MenuActionEvent, MenuFile, MenuNode, UiCtx, UiTranslator,
  WindowState,
  menu::{
    buttons, color, dropdown, input, slider, sub_menu, switch_group, text, toggle, toggle_tip,
  },
  parse_hex_color, spawn_debug_menu,
};

use crate::camera::{CameraMode, FlyCamera};
use crate::edit::{
  BrushShape, EDIT_SIZE_MIN, EditSettings, opacity_pct_to_transmission, smooth_pct_to_roughness,
};
use crate::showcase::ShowcaseRoot;

/// 菜单 TOML 相对 assets 目录的路径（初值来源 + 退出时写回）
pub const MENU_TOML_PATH: &str = "ui/debug_menu.toml";

/// 「世界」页模型下拉的节点路径（选项由 `apply_world_model_options` 按磁盘内容填）
pub const WORLD_MODEL_PATH: &str = "game/world/model";
/// 「世界」页「重载世界」按钮的节点路径（空 label 的按钮组 = 整行按钮）
pub const WORLD_RELOAD_PATH: &str = "game/world/reload";

/// 1 m = 50 voxel（1 voxel = 2cm）：菜单速度用 m/s，资源用 voxel/s
pub const VOXEL_PER_METER: f32 = 50.0;

/// 相机信息纯文本行的刷新间隔（秒）
pub const CAM_INFO_REFRESH_SECS: f32 = 0.25;

/// FPS 统计窗口（秒）：当前 / 平均 / 最低 / 最高都在这个窗口内算
pub const FPS_WINDOW_SECS: f32 = 1.0;

/// DDGI 诊断模式选项的 i18n key（下标 = `DdgiDebugSettings.mode`，顺序与 WESL 一致）
pub const DDGI_MODE_KEYS: [&str; 5] = [
  "menu.render.ddgi.mode.normal",
  "menu.render.ddgi.mode.gi",
  "menu.render.ddgi.mode.wsum",
  "menu.render.ddgi.mode.domain",
  "menu.render.ddgi.mode.probe",
];

/// 探针绘制选项的 i18n key：无 / LOD0..3 / 全（下标 0 = 不绘制）
pub const PROBE_KEYS: [&str; 6] = [
  "menu.render.ddgi.probe.none",
  "menu.render.ddgi.probe.lod0",
  "menu.render.ddgi.probe.lod1",
  "menu.render.ddgi.probe.lod2",
  "menu.render.ddgi.probe.lod3",
  "menu.render.ddgi.probe.all",
];
/// 「全」对应的 `probe_viz_lod`（WESL 里 4 = All）
const PROBE_VIZ_LOD_ALL: f32 = 4.0;

/// 内置默认菜单树；与 `assets/ui/debug_menu.toml` 逐字一致（改这里就要同步资产文件）。
/// 文案字段一律写 i18n key（`menu.*`），经 gate-ui `UiTranslator` 解析；`id` 是与语言无关的回调路径段。
pub fn default_menu() -> MenuFile {
  MenuFile {
    window: WindowState::default(),
    items: vec![
      sub_menu(
        "video",
        "menu.video",
        vec![
          // 半分辨率渲染（3D 场景 1/2 分辨率 → blit 放大到整窗）；见 RenderScale.factor
          toggle_tip("half_res", "menu.video.half_res", false, "menu.video.half_res.tip"),
          toggle("fullscreen", "menu.video.fullscreen", false),
          toggle("vsync", "menu.video.vsync", true),
          // FXAA（最终 blit 的边缘抗锯齿）；见 PostFxSettings.fxaa
          toggle_tip("aa", "menu.video.aa", false, "menu.video.aa.tip"),
          toggle("fps", "menu.video.fps", false),
        ],
      ),
      sub_menu(
        "render",
        "menu.render",
        vec![
          sub_menu(
            "ddgi",
            "menu.render.ddgi",
            vec![
              toggle("enabled", "menu.render.ddgi.enabled", true),
              switch_group("mode", "menu.render.ddgi.mode", &DDGI_MODE_KEYS, 0),
              switch_group("probe", "menu.render.ddgi.probe", &PROBE_KEYS, 0),
              // 性能档：见 DdgiDebugSettings.gi_half_res
              toggle_tip(
                "gi_half",
                "menu.render.ddgi.gi_half",
                false,
                "menu.render.ddgi.gi_half.tip",
              ),
            ],
          ),
          sub_menu(
            "exposure",
            "menu.render.exposure",
            vec![
              toggle("enabled", "menu.render.exposure.enabled", true),
              slider(
                "ev_up",
                "menu.render.exposure.ev_up",
                3.0,
                0.0,
                12.0,
                0.25,
                2,
                Some("menu.render.exposure.ev_up.tip"),
              ),
              slider(
                "ev_dn",
                "menu.render.exposure.ev_dn",
                -3.0,
                -12.0,
                0.0,
                0.25,
                2,
                Some("menu.render.exposure.ev_dn.tip"),
              ),
              slider(
                "tau_up",
                "menu.render.exposure.tau_up",
                2.0,
                0.1,
                8.0,
                0.1,
                2,
                Some("menu.render.exposure.tau_up.tip"),
              ),
              slider(
                "tau_dn",
                "menu.render.exposure.tau_dn",
                1.0,
                0.1,
                8.0,
                0.1,
                2,
                Some("menu.render.exposure.tau_dn.tip"),
              ),
              slider(
                "key",
                "menu.render.exposure.key",
                0.18,
                0.02,
                0.5,
                0.01,
                3,
                Some("menu.render.exposure.key.tip"),
              ),
            ],
          ),
        ],
      ),
      sub_menu(
        "player",
        "menu.player",
        vec![sub_menu(
          "camera",
          "menu.player.camera",
          vec![
            text("pos", "menu.player.camera.pos"),
            text("dir", "menu.player.camera.dir"),
            switch_group(
              "mode",
              "menu.player.camera.mode",
              &["menu.player.camera.mode.orbit", "menu.player.camera.mode.fly"],
              1,
            ),
            slider("speed", "menu.player.camera.speed", 2.6, 0.32, 40.0, 0.1, 1, None),
          ],
        )],
      ),
      sub_menu(
        "game",
        "menu.game",
        vec![
          sub_menu(
            "edit",
            "menu.game.edit",
            vec![
              switch_group(
                "shape",
                "menu.game.edit.shape",
                &["menu.game.edit.shape.sphere", "menu.game.edit.shape.cube"],
                0,
              ),
              input(
                "size",
                "menu.game.edit.size",
                // 无上限：max 取 f32::MAX（仍按 min/step 归一）
                vec![InputField::number("", "3", 1.0, f32::MAX, 1.0, 0)],
              ),
              color("color", "menu.game.edit.color", "96989E"),
              slider("emissive", "menu.game.edit.emissive", 0.0, 0.0, 255.0, 1.0, 0, None),
              slider("alpha", "menu.game.edit.alpha", 100.0, 0.0, 100.0, 1.0, 0, None),
              slider("smooth", "menu.game.edit.smooth", 50.0, 0.0, 100.0, 1.0, 0, None),
            ],
          ),
          // 世界：第一行 = 模型下拉（选项 = assets/vox 下 .vox 文件名，启动时按磁盘内容重填）；第二行 = 重载世界
          sub_menu(
            "world",
            "menu.game.world",
            vec![
              dropdown("model", "menu.game.world.model", &["nuke"], 0),
              buttons("reload", "", &["menu.game.world.reload"]),
            ],
          ),
        ],
      ),
      sub_menu("ui", "menu.ui", vec![toggle("showcase", "menu.ui.showcase", false)]),
    ],
  }
}

/// 读菜单 TOML：优先可写数据目录里的存档，其次只读资源的初版（`assets/ui/debug_menu.toml`）；
/// 都缺失 / 解析失败 → 内置默认 + warn。
pub fn load_menu() -> MenuFile {
  let data = gate_render::data_dir().join(MENU_TOML_PATH);
  let asset = gate_render::assets_dir().join(MENU_TOML_PATH);
  let path = if data.is_file() { data } else { asset };
  let mut model = match std::fs::read_to_string(&path) {
    Ok(src) => match MenuFile::from_toml(&src) {
      Ok(mut m) => {
        m.sanitize();
        merge_defaults(&mut m, &default_menu());
        info!("debug menu loaded from {}", path.display());
        m
      }
      Err(e) => {
        warn!("debug menu TOML parse failed ({e}); using built-in defaults");
        let mut m = default_menu();
        m.sanitize();
        m
      }
    },
    Err(e) => {
      warn!("debug menu TOML missing ({e}); using built-in defaults");
      let mut m = default_menu();
      m.sanitize();
      m
    }
  };
  // 世界模型下拉选项来自磁盘扫描，最后统一填一次
  apply_world_model_options(&mut model);
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

/// 旧存档兼容：把默认树里「存档中不存在」的节点补进去，已存在的保留存档值；递归只在 SubMenu 内做。
fn merge_defaults(loaded: &mut MenuFile, dflt: &MenuFile) {
  merge_nodes(&mut loaded.items, &dflt.items);
}

fn merge_nodes(loaded: &mut Vec<MenuNode>, dflt: &[MenuNode]) {
  for (i, d) in dflt.iter().enumerate() {
    match loaded.iter_mut().find(|n| n.id() == d.id()) {
      Some(l) => {
        if let (MenuNode::SubMenu { children: lc, .. }, MenuNode::SubMenu { children: dc, .. }) =
          (&mut *l, d)
        {
          merge_nodes(lc, dc);
        }
      }
      // 新项按默认树里的位置插入（不是追加到末尾）
      None => loaded.insert(i.min(loaded.len()), d.clone()),
    }
  }
}

/// 把当前菜单状态写回 TOML（退出前调用）；写可写数据目录（`<安装根>/data/ui/debug_menu.toml`）
/// 而非 `assets/`。下次启动由 `load_menu` 优先读这份存档。
pub fn save_menu(model: &MenuFile) {
  let path = gate_render::data_dir().join(MENU_TOML_PATH);
  let Ok(src) = model.to_toml() else {
    warn!("debug menu serialize failed; not saved");
    return;
  };
  if let Some(dir) = path.parent()
    && let Err(e) = std::fs::create_dir_all(dir)
  {
    warn!("debug menu dir create failed ({e})");
    return;
  }
  match std::fs::write(&path, src) {
    Ok(()) => info!("debug menu saved to {}", path.display()),
    Err(e) => warn!("debug menu save failed ({e})"),
  }
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

/// 半分辨率渲染的降采样倍数（「视频/半分辨率」→ `RenderScale.factor`）
const HALF_RES_FACTOR: u32 = 2;

/// 进无边框全屏前的窗口尺寸/位置（退出全屏时复原）；`None` = 当前不在全屏。
/// winit 多数平台会自己复原窗口，但位置在部分后端会丢，故显式存一份兜底。
#[derive(Resource, Default)]
struct WindowedRestore(Option<(UVec2, Option<IVec2>)>);

/// 每帧 delta 的 1s 滚动窗口（FPS 统计）
#[derive(Resource, Default)]
pub struct FpsWindow(VecDeque<f32>);

/// 建 DebugMenu + FPS 覆盖层 + 回调观察者；主题/字体就绪后由 `crate::debug_ui_setup` 调用一次。
pub(crate) fn spawn_debug_menu_ui(world: &mut World, ctx: &UiCtx) {
  let model = load_menu();
  // 文案解析器取资源里的（语言切换只需 bump 版本，见 sync_ui_locale）
  let translate = world.resource::<UiTranslator>().handle();
  let ctx = UiCtx::new(ctx.theme, ctx.font).with_icon_font(ctx.icon_font).with_translate(translate);
  spawn_debug_menu(world, &ctx, model);
  apply_initial_state(world);
  spawn_fps_overlay(world, &ctx);
  register_callbacks(world);
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

/// 用菜单模型初始化各调试资源的初值（TOML = 所有调试常量的初值来源）
fn apply_initial_state(world: &mut World) {
  let model = match gate_ui::menu_model(world) {
    Some(m) => m.clone(),
    None => return,
  };
  let get_bool = |path: &str, dflt: bool| -> bool {
    match model.node(&split(path)) {
      Some(MenuNode::Toggle { checked, .. }) => *checked,
      _ => dflt,
    }
  };
  let get_sel = |path: &str| -> Option<usize> {
    match model.node(&split(path)) {
      Some(MenuNode::SwitchGroup { selected, .. }) => Some(*selected),
      _ => None,
    }
  };
  let get_val = |path: &str| -> Option<f32> {
    match model.node(&split(path)) {
      Some(MenuNode::Slider { value, .. }) => Some(*value),
      _ => None,
    }
  };
  let get_text = |path: &str| -> Option<String> {
    match model.node(&split(path)) {
      Some(MenuNode::Input { fields, .. }) => fields.first().map(|f| f.text.clone()),
      _ => None,
    }
  };
  // 颜色控件是独立节点类型（`color()` → `MenuNode::Color`），不是 `Input`；用 `get_text` 读它恒为 None。
  let get_color = |path: &str| -> Option<String> {
    match model.node(&split(path)) {
      Some(MenuNode::Color { hex, .. }) => Some(hex.clone()),
      _ => None,
    }
  };

  world.resource_mut::<FpsOverlayVisible>().0 = get_bool("video/fps", false);
  // 抗锯齿（FXAA）/ 半分辨率渲染（RenderScale.factor）写 main world 资源，render world 跟随。
  world.resource_mut::<gate_render::PostFxSettings>().fxaa = get_bool("video/aa", false);
  world.resource_mut::<gate_render::RenderScale>().factor =
    if get_bool("video/half_res", false) { HALF_RES_FACTOR } else { 1 };
  let mut q_win = world.query_filtered::<&mut Window, With<PrimaryWindow>>();
  if let Some(mut win) = q_win.iter_mut(world).next() {
    win.present_mode =
      if get_bool("video/vsync", true) { PresentMode::Fifo } else { PresentMode::AutoNoVsync };
    // 全屏：按存档进无边框全屏（窗口尺寸/位置由 winit 记住）
    win.mode = if get_bool("video/fullscreen", false) {
      WindowMode::BorderlessFullscreen(MonitorSelection::Current)
    } else {
      WindowMode::Windowed
    };
  }
  drop(q_win);

  {
    let stage = if get_bool("render/ddgi/enabled", true) {
      gate_render::ddgi::DdgiStage::FULL
    } else {
      gate_render::ddgi::DdgiStage::OFF
    };
    *world.resource_mut::<gate_render::ddgi::DdgiStage>() =
      gate_render::ddgi::DdgiStage::new(stage);
    let mut dbg = world.resource_mut::<gate_render::ddgi::DdgiDebugSettings>();
    if let Some(m) = get_sel("render/ddgi/mode") {
      dbg.mode = m as f32;
    }
    let probe = get_sel("render/ddgi/probe").unwrap_or(0);
    dbg.probe_viz = probe > 0;
    dbg.probe_viz_lod = if probe == 0 {
      0.0
    } else if probe + 1 >= PROBE_KEYS.len() {
      PROBE_VIZ_LOD_ALL
    } else {
      (probe - 1) as f32
    };
    // 性能档（默认关 = 逐像素精确路径）
    dbg.gi_half_res = get_bool("render/ddgi/gi_half", false);
  }
  {
    let mut eye = world.resource_mut::<gate_render::EyeAdaptSettings>();
    eye.enabled = get_bool("render/exposure/enabled", true);
    if let Some(v) = get_val("render/exposure/ev_up") {
      eye.ev_max = v;
    }
    if let Some(v) = get_val("render/exposure/ev_dn") {
      eye.ev_min = v;
    }
    if let Some(v) = get_val("render/exposure/tau_up") {
      eye.tau_brighten = v;
    }
    if let Some(v) = get_val("render/exposure/tau_dn") {
      eye.tau_darken = v;
    }
    if let Some(v) = get_val("render/exposure/key") {
      eye.key = v;
    }
  }
  {
    if let Some(i) = get_sel("player/camera/mode") {
      *world.resource_mut::<CameraMode>() =
        if i == 0 { CameraMode::Orbit } else { CameraMode::Fly };
    }
    if let Some(v) = get_val("player/camera/speed") {
      world.resource_mut::<FlyCamera>().speed = v * VOXEL_PER_METER;
    }
  }
  {
    let mut edit = world.resource_mut::<EditSettings>();
    if let Some(i) = get_sel("game/edit/shape") {
      edit.shape = if i == 0 { BrushShape::Sphere } else { BrushShape::Cube };
    }
    if let Some(t) = get_text("game/edit/size")
      && let Ok(v) = t.trim().parse::<u32>()
    {
      edit.size = v.max(EDIT_SIZE_MIN);
    }
    // 材质四参数；只写「当前笔触材质」，不碰调色板（材质 → 槽的分配在落笔时按内容去重）。
    if let Some(t) = get_color("game/edit/color")
      && let Some([r, g, b, _]) = parse_hex_color(&t)
    {
      edit.mat.color = [r, g, b];
    }
    if let Some(v) = get_val("game/edit/emissive") {
      edit.mat.emissive = v.round().clamp(0.0, 255.0) as u8;
    }
    if let Some(v) = get_val("game/edit/alpha") {
      edit.mat.transmission = opacity_pct_to_transmission(v);
    }
    if let Some(v) = get_val("game/edit/smooth") {
      edit.mat.roughness = smooth_pct_to_roughness(v);
    }
  }
  info!(
    target: "gate",
    "debug menu 初值已应用：fs={} aa={} half_res={} vsync={} fps={} ddgi={} gi_half={} cam={:?} speed={:.2}m/s shape={:?} size={}",
    get_bool("video/fullscreen", false),
    get_bool("video/aa", false),
    get_bool("video/half_res", false),
    get_bool("video/vsync", true),
    get_bool("video/fps", false),
    get_bool("render/ddgi/enabled", true),
    get_bool("render/ddgi/gi_half", false),
    world.resource::<CameraMode>(),
    world.resource::<FlyCamera>().speed / VOXEL_PER_METER,
    world.resource::<EditSettings>().shape,
    world.resource::<EditSettings>().size,
  );
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
     mut ddgi_stage: ResMut<gate_render::ddgi::DdgiStage>,
     mut ddgi_dbg: ResMut<gate_render::ddgi::DdgiDebugSettings>,
     mut eye: ResMut<gate_render::EyeAdaptSettings>| {
      match (ev.path.as_str(), &ev.action) {
        ("render/ddgi/enabled", MenuAction::Toggle(on)) => {
          *ddgi_stage = gate_render::ddgi::DdgiStage::new(if *on {
            gate_render::ddgi::DdgiStage::FULL
          } else {
            gate_render::ddgi::DdgiStage::OFF
          });
          info!("DDGI → stage {}", ddgi_stage.0);
        }
        ("render/ddgi/mode", MenuAction::Select(i)) => {
          ddgi_dbg.mode = *i as f32;
          let name = DDGI_MODE_KEYS.get(*i).map(|k| t!(*k).to_string()).unwrap_or_default();
          info!("DDGI 诊断模式 → {name}");
        }
        ("render/ddgi/probe", MenuAction::Select(i)) => {
          ddgi_dbg.probe_viz = *i > 0;
          ddgi_dbg.probe_viz_lod = if *i == 0 {
            0.0
          } else if *i + 1 >= PROBE_KEYS.len() {
            PROBE_VIZ_LOD_ALL
          } else {
            (*i - 1) as f32
          };
          let name = PROBE_KEYS.get(*i).map(|k| t!(*k).to_string()).unwrap_or_default();
          info!("探针绘制 → {name}");
        }
        // 性能档（关掉即回到逐像素精确路径）
        ("render/ddgi/gi_half", MenuAction::Toggle(on)) => {
          ddgi_dbg.gi_half_res = *on;
          info!("半分辨率 GI → {}", if *on { "on" } else { "off" });
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
  // 原世界不变；成功后全量重建 + GPU 上传 + DDGI 重烘焙由 VoxelScene.demo_force_full_rebuild 驱动。
  world.add_observer(
    |ev: On<MenuActionEvent>,
     mut scene: ResMut<gate_render::VoxelScene>,
     mut aabb: ResMut<gate_render::ddgi::DdgiWorldAabb>,
     mut lod0: ResMut<gate_render::ddgi::DdgiLod0Chunks>,
     q_menu: Query<&gate_ui::DebugMenu>| {
      match (ev.path.as_str(), &ev.action) {
        (WORLD_MODEL_PATH, MenuAction::Select(_)) => {
          let name = world_model_name(&q_menu).unwrap_or_else(|| "?".to_string());
          info!("世界模型 → {name}（点「重载世界」生效）");
        }
        (WORLD_RELOAD_PATH, MenuAction::Button(_)) => {
          let Some(name) = world_model_name(&q_menu) else {
            warn!("重载世界：读不到模型下拉的选择（菜单未就绪），已忽略");
            return;
          };
          let t0 = std::time::Instant::now();
          match crate::scene::reload_world(&mut scene, &mut aabb, &mut lod0, &name) {
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

/// 读「世界」页下拉当前选中的模型名（无菜单 / 无该节点 / 选项为空 → None）
fn world_model_name(q_menu: &Query<&gate_ui::DebugMenu>) -> Option<String> {
  let menu = q_menu.single().ok()?;
  match menu.model.node(&split(WORLD_MODEL_PATH)) {
    Some(MenuNode::Dropdown { options, selected, .. }) => options.get(*selected).cloned(),
    _ => None,
  }
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

/// 退出前把当前菜单状态写回 TOML（AppExit 那一帧执行一次）
pub(crate) fn save_menu_on_exit(
  mut exit: MessageReader<AppExit>,
  q_menu: Query<&gate_ui::DebugMenu>,
  mut saved: Local<bool>,
) {
  if *saved {
    return;
  }
  let mut quitting = false;
  for _ in exit.read() {
    quitting = true;
  }
  if !quitting {
    return;
  }
  *saved = true;
  if let Ok(menu) = q_menu.single() {
    // 当前停留路径也一并持久化
    let mut model = menu.model.clone();
    model.window.path = menu.path.clone();
    model.window.collapsed = menu.collapsed;
    save_menu(&model);
  }
}
