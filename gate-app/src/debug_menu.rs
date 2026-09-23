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
  CAM_INFO_REFRESH_SECS, EDIT_SIZE_MIN, FPS_MAX_FRAMES_PER_TICK, FPS_UPDATE_SECS,
  FPS_WINDOW_SECS, VOXEL_PER_METER,
};
use crate::edit::{
  BrushMaterial, BrushShape, EditSettings, metal_toggle_to_metallic, smooth_pct_to_roughness,
  transparency_pct_to_transmission,
};
use crate::showcase::ShowcaseRoot;
use std::sync::atomic::Ordering;

/// 菜单 TOML 相对 assets 目录的路径（UI 结构与控件缺省值的唯一来源）
pub const MENU_TOML_PATH: &str = "ui/debug_menu.toml";

/// 「世界」页模型下拉的节点路径（选项由 `apply_world_model_options` 填：磁盘扫描 + 程序化场景）
pub const WORLD_MODEL_PATH: &str = "game/world/model";
/// 「世界」页「重载世界」按钮的节点路径（空 label 的按钮组 = 整行按钮）
pub const WORLD_RELOAD_PATH: &str = "game/world/reload";
/// 「世界」页「数据转储」按钮的节点路径（把 CPU/GPU 两份体素数据写进 `logs/`，见
/// `gate_render::VoxelDumpRequest`）
pub const WORLD_DUMP_PATH: &str = "game/world/dump";
/// 「编辑/材质」页 PBR 资产下拉的节点路径（MT7-1；选项 = `assets/textures/pbr/` 的目录名）
pub const EDIT_PBR_ASSET_PATH: &str = "game/edit/mat/pbr_asset";
/// 「编辑/笔触」页「偏移距离」输入框的节点路径：改「笔触大小」时要把自动值写回这个控件
/// （见 `write_brush_offset`）。与 `EditSettings::offset` 是同一个量的两面。
const EDIT_OFFSET_PATH: &str = "game/edit/brush/offset";
/// 「视频」页「抗锯齿」的节点路径：像素大小 ≠ 1 时整行置灰（`sync_video_menu` 按路径置禁用）。
const VIDEO_AA_PATH: &str = "video/aa";

/// 「编辑/材质」页受 **PBR 变体**影响的五个材质控件路径：PBR 模式下整行置灰
/// （`sync_edit_menu` 要按路径置禁用；观察者用字面量匹配）。逐一对照 `debug_menu.toml` 的 `edit/mat` 页。
/// ⚠️ 「编辑」页自己的控件（形状 / 大小 / 偏移距离）**不在**这张表里 —— 它们是笔触几何、与材质无关。
const EDIT_MATERIAL_PATHS: [&str; 5] = [
  "game/edit/mat/color",
  "game/edit/mat/emissive",
  "game/edit/mat/alpha",
  "game/edit/mat/smooth",
  "game/edit/mat/metal",
];

/// 「渲染/天空」各控件的节点路径：`sync_sky_menu` 要按路径查值，故集中在这里（观察者用字面量匹配）。
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

/// 读 UI 结构与控件缺省值：每次都读只读资源 `assets/ui/debug_menu.toml`（结构与缺省值的唯一
/// 来源），再把 `data/config.toml` 里的值覆盖上去 —— 配置里没有的控件保留缺省值，
/// 配置里有而结构里没有的路径（控件被删了）直接忽略；随后按磁盘内容重填世界模型下拉的选项。
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
  // 下拉选项取自磁盘扫描，必须先于配置值套用（配置里的选中项按名字找回）
  apply_world_model_options(&mut model);
  let applied = model.apply_values(&config.menu);
  // 窗口位置 / 收起 / 停留路径只存在配置里
  model.window = config.window.clone();
  model.sanitize();
  info!("debug menu config 命中 {applied}/{}", config.menu.len());
  model
}

/// 把 `WORLD_MODEL_PATH` 下拉的选项换成「`assets/vox` 下实际存在的模型（见
/// `crate::vox_scene::scan_vox_models`）+ 程序化调试场景 `crate::scene::CUBE_IN_VOID`」；
/// 选中项按名字找回，找不到 → `nuke` → 第一项。
fn apply_world_model_options(model: &mut MenuFile) {
  let mut options = crate::vox_scene::scan_vox_models();
  // 程序化场景不来自磁盘（见 `scene::build_cube_in_void`），与 .vox 名字同列在一个下拉里
  options.push(crate::scene::CUBE_IN_VOID.to_string());
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

/// **呈现帧**间隔的 1s 滚动窗口（FPS 统计）。
/// 存的是"每个被提交呈现的帧"的间隔，**不是**主循环的 delta —— 后者在 pipelined rendering 下
/// 会被显示成锯齿（主循环比渲染快得多，见 `gate_render::profiler::FramePace` 的说明）。
#[derive(Resource, Default)]
pub struct FpsWindow {
  /// 逐呈现帧的间隔（秒）
  intervals: VecDeque<f32>,
  /// 上次采样：墙钟秒数 + 呈现帧计数
  last: Option<(f32, u64)>,
}

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
    debug!("UI 语言 → {now}");
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
  info!(target: "gate", "debug menu 初值 {n} 项");
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
      // 「渲染分辨率」是 switch_group（`Select`），其余视频项是 `Toggle` ⇒ 先处理它再走开关。
      // 只改除数，目标纹理由 responsive 系统下一帧重建；上采样在 blit 里做整数块复制（不插值）。
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
        // 抗锯齿：FXAA 在最终 blit 里做（换 fragment 入口，见 PostFxSettings）
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
     mut base: ResMut<gate_render::BaseSettings>| {
      match (ev.path.as_str(), &ev.action) {
        // 「基础」两个开关（菜单「渲染/基础」，勾选 = **算**）：只改 uniform `base_flags`
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
        // GI 分辨率档（**每一档都跑 GI**，不是开关）：网格边长 = 渲染分辨率 / 除数。
        // 代价按 GI 像素数计 ⇒ 1/2 约是全分辨率的 1/4，1/4 再降 4 倍（GI 是低频信号，画质几乎无差）。
        ("render/gi/res", MenuAction::Select(i)) => {
          gi.gi_div = gate_render::gi::GiSettings::DIV_CHOICES[(*i).min(2)];
          info!("GI 分辨率 → 1/{}", gi.gi_div);
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
          info!("GI 降噪质量 → {name} cand={cand} win={k} atrous_r={r}");
        }
        ("render/gi/sun_bounce", MenuAction::Toggle(on)) => {
          gi.sun_bounce = *on;
          info!("GI 二次顶点太阳反弹 → {}", if *on { "on" } else { "off" });
        }
        // 「二次弹射」档（菜单「渲染/光照/二次弹射」）：在二次顶点上再发**一条**余弦射线，把"多一跳"
        // 的间接光（互反射 / 彩色渗色）叠回去 —— 关掉时 GI 只算一次弹射。
        // 档位 → uniform `gi_u.misc.w` 里的**倍数**（0 = 关、4 = 稀疏、1 = 全）：WESL 侧按它的**倒数**
        // 抽签、命中时乘它补齐期望 ⇒ 期望无偏、只有方差变大（见 `gi/screen.wesl` 的候选循环）。
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
        // 「分帧」档（1/2/4/8）：把 GI 的采样预算摊到 N 帧上 —— 每帧只发 `基准候选数 ÷ N` 条射线，
        // 记忆窗同步 ×N。**帧时间是平的**（不是"集中到某一帧"）；窗内样本总数不变 ⇒ 稳态噪声不变，
        // 只有响应时间 ×N（`GI_SS_M_CAP_K` 帧 ≈ 0.53s ⇒ N=4 时 ≈ 2.1s）。
        // 整数条数约束：基准 4 条（降噪质量 低/中）时 N 只能到 4；真正 8 倍需要「降噪质量 = 高」。
        ("render/gi/interval", MenuAction::Select(i)) => {
          gi.share = gate_render::gi::GiSettings::SHARE_CHOICES[(*i).min(3)];
          let c = gate_render::wesl_consts::gi_consts();
          let base = if gi.tier() >= 3 { c.gi_ss_cand_n_hq } else { c.gi_ss_cand_n };
          info!(
            "GI 分帧 → ÷{}（基准候选 {} ⇒ 每帧 {}）",
            gi.share,
            base,
            (base / gi.share).max(1)
          );
        }
        // 「重分配」档（关/弱/中/强）：按"这个像素的时域累积还在不在工作"（本帧有没有接到历史）
        // 把每帧射线预算挪过去 —— 接到的少发、没接到的多发；记忆窗同步放大 ⇒ 减发一侧窗内样本数不变
        // （响应变慢、稳态噪声不变）。代理量与候选数脱钩 ⇒ 不会出现分配振荡。
        ("render/gi/realloc", MenuAction::Select(i)) => {
          let t = (*i).min((gate_render::gi::GiSettings::REALLOC_TIERS - 1) as usize) as u32;
          gi.realloc = t;
          let (name, lo, hi) = match t {
            0 => ("关", 1.0, 1.0),
            1 => ("弱", 0.70, 1.40),
            2 => ("中", 0.50, 2.00),
            _ => ("强", 0.25, 4.00),
          };
          info!("GI 重分配 → {name} 有历史×{lo} 无历史×{hi}");
        }
        // 镜面档位（菜单「渲染/反射」）：**只改反射内容的着色口径**，不改逐面量化的粒度。
        // 档 0/1/2 都**不发额外射线**（1/2 只是把反射命中点按完整着色点算、逃逸天空改走 `sky_primary`）；
        // 只有档 3 在反射命中点补一条太阳 NEE 阴影射线（每个触发像素 +1 次 DDA 遍历）。
        // 档位 → uniform `LightGlobals::refl_tier`：菜单不直接写 uniform，写资源、由
        // `prepare_dda_bind_groups` 每帧搬（见 `gate-render/src/lighting.rs::ReflectionSettings`）。
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
        // 镜面嵌套层级（菜单「渲染/反射」）：允不允许"镜子里的镜子"再反射。
        // 取值是**枚举值**（`0 / 1 / 2 / 4`，不是档位号）⇒ 与 `GiSettings::DIV_CHOICES` 同一个手法。
        // 成本只在真的"镜子对着镜子"时才付：链条遇到非镜面（墙 / 地形）立刻断（见 `main.wesl` 的 MT4-5）。
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

  // 径向模糊（光柱）+ 天体盘（菜单「渲染/天空/径向模糊」「渲染/天空/天体盘」）：
  // 算法、成本模型、取舍全部写在 `gate-render/src/volumetric.rs` 与
  // `assets/shaders/voxel_raytrace/volumetric.wesl`；这里只做「档位 → 数值」的映射与一行日志。
  // **这两组的五个滑杆默认随时刻变化**（`sky.rs` 的关键帧表每帧写它们）：覆写**关**着时整行禁用
  // （不可交互、只实时显示时间算出来的值，见 `sync_sky_menu`）⇒ 想自己定值就先打开该组「覆写」。
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
        // 日月**外观**开关（与下面的「覆写」正交：这个决定画不画，覆写决定角径/光晕由谁给）。
        // 关掉 ⇒ 盘与光晕一起消失。**不受时间驱动** ⇒ `sky.rs::apply_sky` 不会覆写它
        //（它只写 sun_cone / halo）。
        ("render/sky/disk/enabled", MenuAction::Toggle(on)) => {
          fog.body = *on;
          info!("日月外观 → {}", if *on { "on" } else { "off" });
        }
        ("render/sky/disk/override", MenuAction::Toggle(on)) => {
          sky.override_disk = *on;
          info!("天体盘：覆写 → {}", if *on { "on" } else { "off" });
        }
        // 菜单给**度**、引擎吃弧度：全工程只有这一处换算，集中在这里。
        // 这一个量只影响天空里那个天体盘的半径（`volumetric.wesl::sky_primary`），日月共用；
        // 它同时决定盘外光晕的尺度（晕铺到角径的 10 倍）。
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

  // 时间（菜单「渲染/天空/时间」）：时刻 / 年积日 / 纬度 → 太阳与月亮的位置。
  // 推导、关键帧色板与全部取舍写在 `gate-render/src/sky.rs`；这里只把面板值写进资源，
  // 并顺手打一行**推导结果**（高度角 + 当前主天体）—— 推导本身每帧在 `sky::apply_sky` 里做。
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

  // 颜色（菜单「渲染/天空/颜色」）：太阳 / 月亮 / 天空三个色 + 覆写。
  // 覆写开着时用它们代替时间算出来的颜色 —— **强度**仍由时间给（`sky.rs` 只换色）。
  // 覆写关着时三个拾色器整行禁用（见 `sync_sky_menu`）；拾色器是自由文本输入（半截输入会中途
  // 失败）⇒ 解析失败就忽略，不打断输入。
  world.add_observer(
    |ev: On<MenuActionEvent>, mut sky: ResMut<gate_render::SkySettings>| {
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
    },
  );

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

  // 材质控件只改当前笔触材质，不碰调色板；材质 → 槽的分配在落笔时（`edit::material_slot`）。
  // MT7-1：PBR 变体开关 / 资产槽 / metallic / IOR / specular 也走这里；每次改动都打一行
  // `材质 → …`（含两种变体的完整参数，见 `BrushMaterial::summary`）。
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
            // **大小变了 ⇒ 偏移距离自动跟到"大小的一半"**（仍可自由输入，见 `write_brush_offset`）
            let auto = edit.size as f32;
            write_brush_offset(&mut edit, auto, &mut q_offsets);
            info!("笔触大小 → {} vx；偏移距离自动 → {:.1}", edit.size, auto);
          }
        }
        // 偏移距离：笔触几何中心相对命中体素的法线方向偏移（放置 `+`、摧毁 `−`）。
        // 自由输入 ⇒ **不回写控件**（半截输入会被下一帧重写、打断输入）；解析失败就忽略。
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
          // 输入框是自由文本：解析失败（半截输入）就忽略，不打断输入
          None => debug!(target: "gate", "笔触颜色未成形 {t:?} → 忽略"),
        },
        // 下面这七个材质控件只在**平凡变体**下生效：PBR 变体一开就整行置灰、不写进材质
        // （见 `sync_edit_menu`）。它们改的是笔触的平凡参数，切回平凡变体时原样还在。
        ("game/edit/mat/emissive", MenuAction::Value(v)) => {
          edit.mat.emissive = v.round().clamp(0.0, 255.0) as u8;
          log_material(&edit.mat);
        }
        ("game/edit/mat/alpha", MenuAction::Value(v)) => {
          // 「透明度」= 滑杆值越大越透明（见 `transparency_pct_to_transmission`）。
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
            "材质 → PBR 资产槽 {slot}={name} {how}；{}",
            edit.mat.summary(),
          );
        }
        ("game/edit/mat/metal", MenuAction::Toggle(on)) => {
          // 金属度是**二值**（见 `BrushMaterial::metallic`）⇒ 开关写 0 / 255。
          // 金属的 `F0 = albedo` 且 `kD = 0` ⇒ 这是平凡变体下"做镜面"的旋钮。
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

  // 下拉只改模型（退出时随存档持久化）；「重载世界」按钮才真正换世界（同步阻塞），失败只 warn、
  // 原世界不变；成功后全量重建 + GPU 上传由 VoxelScene.demo_force_full_rebuild 驱动。
  // 「数据转储」按钮与上面两个无关：它只置位一个请求位，转储在 render 侧下一帧做。
  world.add_observer(
    |ev: On<MenuActionEvent>,
     mut scene: ResMut<gate_render::VoxelScene>,
     mut dump: ResMut<gate_render::VoxelDumpRequest>,
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
          match crate::scene::reload_world(&mut scene, &name) {
            Ok(_info) => info!("世界重载 → {name} {:?}", t0.elapsed()),
            Err(e) => warn!("重载世界失败 {name}：{e} → 原世界不变"),
          }
        }
        // 数据转储：只置位请求，真正的转储（含 GPU readback 与落盘）在 render 侧下一帧做
        // （见 `gate_render::VoxelDumpRequest` 与 `upload::dump_voxel_buffers`）。
        (WORLD_DUMP_PATH, MenuAction::Button(_)) => {
          dump.arm();
          info!("数据转储 → 请求");
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

/// 「天象」那一行日志的推导部分：三个时间控件（时刻 / 年积日 / 纬度）都会改变主天体，
/// 于是统一用这一行汇报推导结果（色与强度由 `sky.rs` 的关键帧表给，日志里不重复抄）。
fn sky_report(sky: &gate_render::SkySettings) -> String {
  let alt = gate_render::sun_altitude_deg(sky.hour, sky.day_of_year, sky.latitude_deg);
  format!(
    "太阳高度角 {alt:.1}°、主光 = {}",
    if alt >= 0.0 { "太阳" } else { "反日点的月亮" }
  )
}

/// 「一昼夜多久」（流逝速度 = 游戏小时 / 实时秒）；速度 0 ⇒ 时刻不动。
fn day_cycle(hours_per_sec: f32) -> String {
  if hours_per_sec <= 0.0 {
    "速度为 0 ⇒ 时刻不动".to_string()
  } else {
    format!("一昼夜 {:.1} 分钟", 24.0 / hours_per_sec / 60.0)
  }
}

/// HEX 文本 → sRGB（0..1）。解析失败 = 半截输入 ⇒ `None`（调用方忽略，见颜色观察者）。
fn hex_srgb01(hex: &str) -> Option<[f32; 3]> {
  let [r, g, b, _] = parse_hex_color(hex)?;
  Some([f32::from(r) / 255.0, f32::from(g) / 255.0, f32::from(b) / 255.0])
}

/// 引擎 → 菜单（每帧）：把"引擎会自己改的那些控件"同步过去。
///   · **时刻**：**仅自动流逝时**回写（那一刻引擎在推进它）；自动关着时用户是唯一作者，
///     回写会把拖动结果当帧抹掉 —— 表现为"拖动后回弹"。与下面两组同理：引擎说了算的控件
///     不该同时可拖（那两组干脆是禁用的）；
///   · 「圣光」「日月」两组的**滑杆**：覆写关着时它们显示的就是时间算出来的值；
///   · 三个**「覆写」开关**的勾选态（真源是 `SkySettings` 的三个标志）；
///   · **禁用态**：覆写关着的那组滑杆 + 自动流逝开着时的「时刻」—— 值不是它说了算的
///     （控件不可交互、配色降亮；滑杆行还要连带降亮左侧名称）。
///
/// 写两处（都在这一帧内被消费）：模型值 —— `refresh_visuals` 每帧按它刷新滑杆右侧的数值文本，
/// 并据此重算行内文字/色块的降亮；控件侧 —— 只装/摘 `UiDisabled`，widget 的降亮配色每帧按它重算。
/// 「颜色」那三个拾色器不回写（它们永远是你的色，覆写只决定生不生效）。
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
  // (控件路径, 引擎当前值)：只列"引擎会自己改"的那些
  let mut want: Vec<(&str, f32)> = Vec::with_capacity(6);
  // 「时刻」只在**自动流逝**时由引擎推进（`sky.rs::apply_sky`）⇒ 仅那时回写。
  // 自动关着时 `sky.hour` 只由用户拖动改变，再回写就等于每帧撤销拖动（回弹）。
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

  // 覆写开关的勾选态（真源：`SkySettings` 的三个标志）
  let overrides = [
    (BLUR_OVERRIDE_PATH, sky.override_blur),
    (DISK_OVERRIDE_PATH, sky.override_disk),
    (COLOR_OVERRIDE_PATH, sky.override_colors),
  ];
  // 引擎说了算的控件 ⇒ 整行禁用（不可交互、配色降亮）。两种情形：
  //   · 覆写**关**着的组的滑杆：显示的是时间算出来的值，不是它能定的；
  //   · **自动流逝开着时的「时刻」**：`apply_sky` 每帧在推进它、`want` 每帧回写
  //     ⇒ 同样不许拖（拖了也当帧被覆盖）。自动关着时它由用户唯一拥有，故那时不禁用。
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
  // 界面侧**不重建页**：widget 的降亮配色本来就是**每帧**按 `Has<UiDisabled>` 从主题令牌重算的
  // ⇒ 装上/摘下这个组件就够了。重建会让整页重新 spawn，而滑杆值 / `Checked` 是延迟写入的
  // ⇒ 新控件有几帧呈"旧值"的样子（开关旋钮先跳到左边再弹回来）。
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
  // 颜色行的**色块**自己也带禁用态：调色板能不能展开只看它（`color_picker_system` 读 `UiDisabled`）
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
  // 模型侧照写：它是"整页重新 spawn"（导航进出那一页）时的渲染依据，必须与控件一致
  for (path, disabled) in disable {
    if let Some(node) = menu.model.node_mut(&split(path)) {
      node.set_disabled(disabled);
    }
  }
}

/// 把「偏移距离」写进两处：`EditSettings.offset`（**落笔真源**）+ 菜单里那个输入框的
/// `TextInputValue`（**界面真源** —— widget 每帧按它渲染，`menu_system` 再把控件值反向同步进模型
/// ⇒ 不需要重建页、也不打断正在输入的框）。
///
/// 调用点只有一处：**改「笔触大小」时自动置为 `size / 2`**。平时的自由输入走
/// `game/edit/brush/offset` 那条观察者分支（它只写资源、**不回写控件** —— 否则半截输入会被重写、
/// 打字被打断）。
fn write_brush_offset(
  edit: &mut EditSettings,
  offset: f32,
  q: &mut Query<(&gate_ui::menu::MenuItem, &mut gate_ui::TextInputValue)>,
) {
  edit.offset = offset.max(0.0);
  // 与 `debug_menu.toml` 里那个字段的 `decimals = 1` 对齐（`1.5` / `2.0`）
  let text = format!("{:.1}", edit.offset);
  for (item, mut tv) in q.iter_mut() {
    if item.path == EDIT_OFFSET_PATH && tv.0 != text {
      tv.0 = text.clone();
    }
  }
}

/// 「视频」页的「抗锯齿」在**像素大小 ≠ 1** 时整行置灰（不可交互、配色降亮）。
///
/// 理由：降采样档本身就是"大粒像素"的观感，FXAA 会把整数块边界抹糊，与那些档位的意图相反
/// ⇒ 那些档位下**一律不启用抗锯齿**（判据只有一处：`dda.rs::blit_dda_view`）。
///
/// **只置灰、不改值**：这个开关记的是"像素大小 = 1 时要不要抗锯齿"，切回 `1` 时它的状态立刻生效
/// —— 界面上的勾选态、落盘值都不动（与 `sync_sky_menu` 的"覆写关着时禁用整行"同一个语义）。
///
/// 与其他 sync 不同的地方：**不重建页**。开关的禁用配色是每帧算的 ⇒ 只装/摘 `UiDisabled`
/// （重建会让整页重新 spawn，而 `Checked` 延迟插入 ⇒ 开关旋钮会先跳到左边再弹回来）。
pub(crate) fn sync_video_menu(
  scale: Res<gate_render::RenderScale>,
  mut commands: Commands,
  mut q_menu: Query<&mut gate_ui::DebugMenu>,
  q_aa: Query<(Entity, &gate_ui::menu::MenuItem, Has<gate_ui::widgets::UiDisabled>)>,
) {
  let want = scale.factor != 1;
  // 模型侧照写：它是"整页重建"（导航进出那一页）时的渲染依据，必须与控件一致
  let Ok(mut menu) = q_menu.single_mut() else { return };
  if let Some(node) = menu.model.node_mut(&split(VIDEO_AA_PATH)) {
    node.set_disabled(want);
  }
  // 界面侧**不重建页**：开关的禁用态本来就是**每帧**算的（`toggle_switch_state_system` 读
  // `Has<UiDisabled>` 重算轨道/滑块配色，滑块位置仍按 `Checked`）⇒ 装上/摘下这个组件就够了。
  // 重建会让整页重新 spawn，而 `Checked` 是**延迟**插入的 ⇒ 新开关有几帧呈"关"的样子（旋钮跳到左边）。
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

/// 「编辑」页的材质控件在 **PBR 变体**下整行置灰（不可交互、配色降亮）。
///
/// 理由（**一刀切**）：PBR 的参数**全部来自材质资产与它的贴图** —— albedo / roughness / metallic
/// 来自 albedo + roughmetal 贴图，emissive / specular / IOR 是资产的标量 ⇒ 槽级一个旋钮都没有。
/// 因此包括「颜色」在内**五个控件一律置灰不生效**（PBR 的底色来自 albedo 贴图，不做 tint）：
/// 颜色 / 自发光 / 透明度 / 光滑度 / 金属度（见 [`EDIT_MATERIAL_PATHS`]）。
///
/// 不置灰的只有「笔触形状 / 笔触大小」（笔触几何，与材质无关）与「PBR 变体 / PBR 资产」本身。
/// 置灰不改值：切回平凡变体时，之前调好的平凡参数原样还在。
///
/// 与 `sync_sky_menu` 同一套手法：界面侧**不重建页**，只按路径装/摘 `UiDisabled` ——
/// widget 的降亮配色每帧按它重算；模型侧照写，供"整页重新 spawn"（导航进出那一页）时用。
pub(crate) fn sync_edit_menu(
  edit: Res<EditSettings>,
  mut commands: Commands,
  mut q_menu: Query<&mut gate_ui::DebugMenu>,
  q_controls: Query<(Entity, &gate_ui::menu::MenuItem, Has<gate_ui::widgets::UiDisabled>)>,
  q_swatches: Query<(Entity, &gate_ui::menu::MenuColorSwatch, Has<gate_ui::widgets::UiDisabled>)>,
) {
  let want = edit.mat.pbr;
  // 模型侧照写：它是"整页重新 spawn"（导航进出那一页）时的渲染依据，必须与控件一致
  let Ok(mut menu) = q_menu.single_mut() else { return };
  for path in EDIT_MATERIAL_PATHS {
    if let Some(node) = menu.model.node_mut(&split(path)) {
      node.set_disabled(want);
    }
  }
  // 界面侧只装/摘 `UiDisabled`：滑杆 / 输入框 / 开关的配色都由各自的系统每帧重算
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
  // 「颜色」那行的**色块**自己也带禁用态（调色板能不能展开只看它，见 `color_picker_system`）
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

/// 右上角 FPS：`(当前, 平均, 最低, 最高)`，每 `FPS_UPDATE_SECS` 刷新一次。
/// - 当前 = **最新这一帧**的帧率（1 ÷ 该帧间隔）
/// - 平均 = 近 `FPS_WINDOW_SECS` 的帧数 ÷ 该窗口时长
/// - 最低 / 最高 = 近 `FPS_WINDOW_SECS` 内**逐帧**帧率的最小 / 最大值
///
/// 间隔口径是**提交呈现的帧**（`gate_render::profiler::FramePace`），不是主循环的 delta ——
/// 本工程是 pipelined rendering，主循环可以比渲染快好几倍（见 `FramePace` 的说明）。
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
  // 只有**呈现帧数变了**才记一次；一次采样跨了多帧（主循环比渲染慢）就按帧均摊成多份 ——
  // 窗口里的统计口径始终是"每帧间隔"。上限 `FPS_MAX_FRAMES_PER_TICK` 防止（窗口拖动、
  // 菜单重载之类的）一次性长停顿把窗口灌满同一份间隔。
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
  // 四个值每 `FPS_UPDATE_SECS` 刷一次（`window` 仍然逐帧采样，见上）
  *since_update += time.delta_secs();
  if *since_update < FPS_UPDATE_SECS {
    return;
  }
  *since_update = 0.0;
  let cur = window.intervals.back().copied().map_or(0.0, rate_of);
  let avg = if sum > 0.0 { window.intervals.len() as f32 / sum } else { 0.0 };
  let min_dt = window.intervals.iter().cloned().fold(0.0f32, f32::max); // 最大间隔 ⇒ 最低帧率
  let max_dt = window.intervals.iter().cloned().fold(f32::MAX, f32::min); // 最小间隔 ⇒ 最高帧率
  let min = rate_of(min_dt);
  let max = if max_dt < f32::MAX { rate_of(max_dt) } else { 0.0 };
  let text = format!("({:>3}, {:>3}, {:>3}, {:>3})", fps3(cur), fps3(avg), fps3(min), fps3(max));
  if let Ok(mut t) = q_text.single_mut()
    && t.0 != text
  {
    t.0 = text;
  }
}

/// 帧间隔（秒）→ 帧率；间隔非正时记 0
fn rate_of(dt: f32) -> f32 {
  if dt > 0.0 { 1.0 / dt } else { 0.0 }
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
