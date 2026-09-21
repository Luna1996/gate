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
use gate_voxel::{inverted_pct_to_override, ior_slider_to_x100, slider_to_override};

use crate::camera::{CameraMode, FlyCamera};
use crate::config::Config;
use crate::consts::{
  CAM_INFO_REFRESH_SECS, EDIT_SIZE_MIN, FPS_WINDOW_SECS, HALF_RES_FACTOR, VOXEL_PER_METER,
};
use crate::edit::{
  BrushMaterial, BrushShape, EditSettings, smooth_pct_to_roughness, transparency_pct_to_transmission,
};
use crate::showcase::ShowcaseRoot;

/// 菜单 TOML 相对 assets 目录的路径（UI 结构与控件缺省值的唯一来源）
pub const MENU_TOML_PATH: &str = "ui/debug_menu.toml";

/// 「世界」页模型下拉的节点路径（选项由 `apply_world_model_options` 按磁盘内容填）
pub const WORLD_MODEL_PATH: &str = "game/world/model";
/// 「世界」页「重载世界」按钮的节点路径（空 label 的按钮组 = 整行按钮）
pub const WORLD_RELOAD_PATH: &str = "game/world/reload";
/// 「编辑」页 PBR 资产下拉的节点路径（MT7-1；选项 = `assets/textures/pbr/` 的目录名）
pub const EDIT_PBR_ASSET_PATH: &str = "game/edit/pbr_asset";

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
        // GI 分辨率档（**每一档都跑 GI**，不是开关）：网格边长 = 渲染分辨率 / 除数。
        // 代价按 GI 像素数计 ⇒ 1/2 约是全分辨率的 1/4，1/4 再降 4 倍（GI 是低频信号，画质几乎无差）。
        ("render/gi/res", MenuAction::Select(i)) => {
          gi.gi_div = gate_render::gi::GiSettings::DIV_CHOICES[(*i).min(2)];
          info!(
            "GI 分辨率 → 渲染分辨率的 1/{}（GI 像素数为全分辨率的 1/{}）",
            gi.gi_div,
            gi.gi_div * gi.gi_div
          );
        }
        // 「降噪质量」档（**与分辨率档正交**：下面这些项目对所有分辨率档一视同仁）。
        // 每档只比上一档多一件事，成本单调递增。权威值都在 `gi/consts.wesl`，这里只选档；
        // 日志里的数值从 `gi_consts()` 取（不另抄一份，避免与 `.wesl` 漂移）：
        //   0 关：一条降噪 pass 都不跑（`dda_main` 直接采样原始 GI）；
        //   1 低（默认）：时域累积 + 5 轮 3×3（8 tap）atrous；
        //   2 中：atrous 换 5×5（24 tap）；
        //   3 高：再把每像素候选数翻倍（= GI 射线翻倍，全链最贵的一项）+ 记忆窗 20→32 帧。
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
          info!(
            "GI 降噪质量 → {}（每像素候选 {} 条、记忆窗 {} 帧、atrous 核半径 {}；与分辨率档正交）",
            name, cand, k, r
          );
        }
        ("render/gi/sun_bounce", MenuAction::Toggle(on)) => {
          gi.sun_bounce = *on;
          info!("GI 二次顶点太阳反弹 → {}", if *on { "on" } else { "off" });
        }
        // 「分帧」档（1/2/4/8）：把 GI 的采样预算摊到 N 帧上 —— 每帧只发 `基准候选数 ÷ N` 条射线，
        // 记忆窗同步 ×N。**帧时间是平的**（不是"集中到某一帧"）；窗内样本总数不变 ⇒ 稳态噪声不变，
        // 只有响应时间 ×N（`GI_SS_M_CAP_K` 帧 ≈ 0.53s ⇒ N=4 时 ≈ 2.1s）。
        // 整数条数约束：基准 4 条（降噪质量 低/中）时 N 只能到 4；真正 8 倍需要「降噪质量 = 高」。
        ("render/gi/interval", MenuAction::Select(i)) => {
          gi.share = gate_render::gi::GiSettings::SHARE_CHOICES[(*i).min(3)];
          let c = gate_render::wesl_consts::gi_consts();
          let base = if gi.tier() >= 3 { c.gi_ss_cand_n_hq } else { c.gi_ss_cand_n };
          info!(
            "GI 分帧 → 每帧预算 ÷{}（基准候选 {} 条 ⇒ 实际每帧 {} 条，窗内样本数为基准值；帧时间恒定）",
            gi.share,
            base,
            (base / gi.share).max(1)
          );
        }
        // 「采样重分配」档（关/温和/中/强）：按"这个像素已经有多干净"（上一帧 reservoir 的 M）
        // 把每帧射线预算挪过去 —— 收敛的少发、刚脏的多发；记忆窗同步放大 ⇒ 窗内样本数不变
        // （噪声不变、只有响应时间随像素变）⇒ 不会出现分配振荡。
        ("render/gi/realloc", MenuAction::Select(i)) => {
          let t = (*i).min((gate_render::gi::GiSettings::REALLOC_TIERS - 1) as usize) as u32;
          gi.realloc = t;
          let (name, lo, hi) = match t {
            0 => ("关", 1.0, 1.0),
            1 => ("温和", 0.70, 1.40),
            2 => ("中", 0.50, 2.00),
            _ => ("强", 0.25, 4.00),
          };
          info!("GI 采样重分配 → {}（收敛像素 ×{}、刚脏像素 ×{}）", name, lo, hi);
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
  // MT7-1：PBR 变体开关 / 资产槽 / metallic / IOR / specular 也走这里；每次改动都打一行
  // `材质 → …`（含两种变体的完整参数，见 `BrushMaterial::summary`）。
  world.add_observer(
    |ev: On<MenuActionEvent>,
     mut edit: ResMut<EditSettings>,
     q_menu: Query<&gate_ui::DebugMenu>,
     pbr_set: Option<Res<gate_render::PbrTextureSet>>,
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
            log_material(&edit.mat);
          }
          // 输入框是自由文本：解析失败（半截输入）就忽略，不打断输入
          None => debug!(target: "gate", "笔触颜色输入未成形 → {t:?}（忽略）"),
        },
        ("game/edit/emissive", MenuAction::Value(v)) => {
          // 平凡变体直接用这个字节；PBR 变体把它当**槽级覆盖**（最低档 = 不覆盖），
          // 两个编码同时维护 ⇒ 切变体不需要重算（编码见 gate_voxel::palette 的映射函数）。
          edit.mat.emissive = v.round().clamp(0.0, 255.0) as u8;
          edit.mat.emissive_ov = slider_to_override(*v, 0.0, 255.0);
          log_material(&edit.mat);
        }
        ("game/edit/alpha", MenuAction::Value(v)) => {
          // 「透明度」= 滑杆值越大越透明（见 `transparency_pct_to_transmission`）。
          // PBR 变体那一侧走 `slider_to_override`（**不是** `inverted_*`）：最低档 = 不覆盖，其余 1..100% 覆盖为 0.004..1.0
          edit.mat.transmission = transparency_pct_to_transmission(*v);
          edit.mat.transmission_ov = slider_to_override(*v, 0.0, 100.0);
          log_material(&edit.mat);
        }
        ("game/edit/smooth", MenuAction::Value(v)) => {
          edit.mat.roughness = smooth_pct_to_roughness(*v);
          edit.mat.roughness_ov = inverted_pct_to_override(*v);
          log_material(&edit.mat);
        }
        ("game/edit/pbr", MenuAction::Toggle(on)) => {
          edit.mat.pbr = *on;
          info!(
            target: "gate",
            "材质 → 变体切到 {}（IS_PBR）；{}",
            if *on { "PBR（资产贴图 + 槽级覆盖）" } else { "平凡（逐槽独立参数）" },
            edit.mat.summary(),
          );
        }
        ("game/edit/pbr_asset", MenuAction::Select(i)) => {
          // 下拉选项是**资产 id 文本**（不是 i18n key），选项顺序 = 目录字典序 = 槽号顺序。
          // 有 `PbrTextureSet` 就按 id 查**真实层号**（缺素材的目录会被跳过 ⇒ 层号顺延），
          // 资源还没就绪（贴图集是异步构建的）则回落选项下标 —— 两者在素材齐全时一致。
          let name = menu_pbr_asset(&q_menu).unwrap_or_else(|| "?".to_string());
          let (slot, how) = match pbr_set.as_deref().and_then(|s| s.slot_of(&name)) {
            Some(slot) => (slot, "按 PbrTextureSet 解析"),
            None => (*i as u32, "贴图集未就绪/未收录，回落选项下标"),
          };
          edit.mat.asset_slot = slot;
          info!(
            target: "gate",
            "材质 → PBR 资产槽 {slot} = {name}（{how}）；{}",
            edit.mat.summary(),
          );
        }
        ("game/edit/metal", MenuAction::Value(v)) => {
          edit.mat.metallic_ov = slider_to_override(*v, 0.0, 100.0);
          log_material(&edit.mat);
        }
        ("game/edit/ior", MenuAction::Value(v)) => {
          // IOR 是**资产级**物理基值（D1）：palette 槽里没有它，故只记录下来。
          edit.mat.ior_x100 = ior_slider_to_x100(*v);
          log_material(&edit.mat);
        }
        ("game/edit/spec", MenuAction::Value(v)) => {
          edit.mat.specular_ov = slider_to_override(*v, 0.0, 100.0);
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

/// 从菜单组件里读「编辑」页 PBR 资产下拉的选中项（= `assets/textures/pbr/` 的目录名，无菜单 → None）。
/// 下拉的选项是**纯文本 id**（不是 i18n key）⇒ 选中项本身就是资产 id，可直接交给
/// `PbrTextureSet::slot_of` 解析成真实槽号。
fn menu_pbr_asset(q_menu: &Query<&gate_ui::DebugMenu>) -> Option<String> {
  match q_menu.single().ok()?.model.node(&split(EDIT_PBR_ASSET_PATH)) {
    Some(MenuNode::Dropdown { options, selected, .. }) => options.get(*selected).cloned(),
    _ => None,
  }
}

/// 材质控件的统一日志（MT7-1 的验收：每次改动日志有 `材质 → …` 一行）。
fn log_material(mat: &BrushMaterial) {
  info!(target: "gate", "材质 → {}", mat.summary());
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
