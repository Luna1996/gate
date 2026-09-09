//! 左上角调试 overlay：FPS（CUR/AVG/MIN/MAX）+ 真实 GPU 帧时折线 + CPU 工作时长折线
//! （实测、不含 vsync/acquire 等待）+ 相机信息。
//!
//! 装配入口 [`demo_ui_setup`](crate::demo_ui_setup) 在主题/字体就绪后调用
//! [`spawn_debug_view`] 一次性生成；[`fps_line_feed`] 每帧喂统计数据。

use std::collections::VecDeque;

use bevy::prelude::*;
use bevy::window::{PresentMode, Window};

use gate_render::OrbitCamera;
use gate_ui::{
  UiCtx,
  widgets::{
    ButtonConfig, ButtonVariant, GridConfig, LabelConfig, LabelStyle, PanelSurface, PlotConfig,
    PlotData, PlotDomain, PlotLayout, SliderConfig, SliderValueChanged, TabConfig,
    ToggleSwitchConfig, ToggleSwitchToggled, UiClick, blank_plot_image, button, color_of, grid,
    grid_cell, label, plot, px, slider, tab_view, toggle_switch,
  },
};

use crate::showcase::ShowcaseRoot;

/// fps_line_feed 每 0.25s 写一行 CUR/AVG/MIN/MAX
pub(crate) const FPS_LOG_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/logs/fps.log");
/// 逐帧帧时日志：每帧一行
/// `elapsed,dt,trace,beam,ddgi_total,probe_viz,blit,cpu_total,seg_main,seg_pre,seg_acq,seg_prep,seg_graph,seg_submit`
///（-1 = 该来源未上线；cpu_total = 实测 CPU 工作时长（不含 seg_acq acquire/vblank
/// 等待段）；seg_* 为 gate-render 六段时间戳分解），0.25s 攒一批刷盘。
/// 帧时波动/周期尖刺定位用
pub(crate) const FRAME_TIME_LOG_PATH: &str =
  concat!(env!("CARGO_MANIFEST_DIR"), "/logs/frame_time.log");

// ================= 左上角单行 FPS（CUR/AVG/MIN/MAX） =================
//
// 统计口径：滚动窗口 = 最近 5s 的逐帧 delta_secs：
// - CUR = 本刷新周期（0.25s）帧数 / 实际耗时
// - AVG = 窗口帧数 / 窗口总时长
// - MIN = 1 / 窗口最大帧耗时（最差一帧）
// - MAX = 1 / 窗口最小帧耗时（最好一帧）
// 数字固定 3 位宽（上限 999）→ 文本定长，UI 不抖动。

pub(crate) const FPS_REFRESH_SECS: f32 = 0.25;
const FPS_WINDOW_SECS: f32 = 5.0;

/// debug-view 面板固定宽度（px；高度 auto 随当前 tab 页内容收缩）
const DEBUG_VIEW_W: f32 = 360.0;
/// 帧时长折线图：画布纹理尺寸（宽 ≈ FPS 文本宽；高 64px，1:1 显示）
const FRAME_PLOT_W: u32 = 320;
const FRAME_PLOT_H: u32 = 64;
/// 折线环形样本容量（≈ 画布像素宽，约 1 样本/px；60fps 下约 6s 窗口）
const FRAME_PLOT_CAP: usize = 360;
/// GPU span 诊断新鲜度上限（秒）：超过未更新的测量视为 span 已停录（如 DDGI 档位 0
/// 早退），残值不可用。GPU 时间戳回读延迟只有 1-2 帧，0.25s 余量充足
const GPU_DIAG_MAX_AGE: f32 = 0.25;

#[derive(Component)]
pub(crate) struct FpsText;

/// demo UI 已 spawn 的根标记：demo_ui_setup 的存在性守卫（查到即跳过），
/// 替代 Local<bool> done 平行状态——产物本身就是"是否已跑过"的真源
#[derive(Component)]
pub(crate) struct DemoUiRoot;

#[derive(Component)]
pub(crate) struct CamInfoText;

/// GPU 帧时折线标记（fps_line_feed 用它过滤 PlotData，避免命中 showcase 折线）
#[derive(Component)]
pub(crate) struct GpuPlot;

/// CPU 工作时长折线标记（gate-render 实测、不含 acquire/vblank 等待；与 GPU span 折线对照）
#[derive(Component)]
pub(crate) struct CpuPlot;

/// 「右上角面板」开关标记（ToggleSwitchToggled 观察者以此过滤事件）
#[derive(Component)]
struct ShowcaseVisibilityToggle;

/// 「垂直同步」开关标记（观察者写主窗口 Window.present_mode → bevy_render 重配 swapchain）
#[derive(Component)]
struct VsyncToggle;

/// DDGI 阶段档位滑杆标记（0-3 吸附档位；观察者写 gate_render::ddgi::DdgiStage）
#[derive(Component)]
struct DdgiStageSlider;

/// DDGI 阶段档位数值标签（滑杆右侧，实时显示档位名）
#[derive(Component)]
struct DdgiStageValueLabel;

/// DDGI 调试模式按钮标记（值 = 模式 0..4，互斥单选）
#[derive(Component)]
struct DdgiDebugModeBtn(u8);

/// DDGI 增益滑杆标记
#[derive(Component)]
struct DdgiGainSlider;

/// DDGI 增益数值标签（滑杆右侧，实时显示当前值）
#[derive(Component)]
struct DdgiGainValueLabel;

/// Probe Viz 开关标记（写 DdgiDebugSettings.probe_viz → 探针位置黄色方块可视化）
#[derive(Component)]
struct DdgiProbeVizToggle;

/// Probe Viz 层级滑杆标记（写 DdgiDebugSettings.probe_viz_lod）
#[derive(Component)]
struct DdgiProbeVizLodSlider;

/// Probe Viz 层级数值标签（滑杆右侧，实时显示当前层级名）
#[derive(Component)]
struct DdgiProbeVizLodValueLabel;

/// Probe Viz 层级名（滑杆 stop 序：0=All 全部, 1..=4=LOD0~3；
/// 新架构 = 4 级相机滚动 LOD 固定槽，无 base 世界级烘焙网格）
const DDGI_PROBE_VIZ_LODS: [&str; 5] = ["All", "LOD 0", "LOD 1", "LOD 2", "LOD 3"];

/// 调试模式名（按钮文本 + 日志用）
const DDGI_DEBUG_MODES: [&str; 5] = ["Normal", "GI", "wsum", "Domain", "Probe"];

/// DDGI 阶段档位名（slider stop 序：0=全关，1=①Active Probe，2=+②RayQuery
/// cast/update，3=+③voxel 着色采样完整 DDGI）
const DDGI_STAGES: [&str; 4] = ["Off", "Active", "Cast", "Full"];

/// fps → 3 位宽显示值（上限 999，防 4 位数抖动）
pub(crate) fn fps3(v: f32) -> u32 {
  (v.round() as u32).min(999)
}

/// 左上角 debug-view 面板：**固定宽度 360px、高度 auto**（随当前 Tab 页内容收缩）。
/// TabView（fit_content 自适应高度模式）两页：
/// - Stats：FPS 读数 / GPU 帧时折线 / CPU 工作时长折线 / 相机信息 / VSync / UI Showcase
/// - DDGI：阶段档位滑杆 / 调试模式按钮 / Gain 滑杆 / Probe Viz 开关 / LOD 滑杆
///   （全部控件合并在同一个 cell 内纵向排列）
///
/// 视觉：root 提供外框 + HUD 卡面底色；每页一个 grid（去外框避免双线，保留
/// 1px gap 填色格线），cell 用不透明 Card 面（grid_cell 约定：HUD 档半透明会露格线）。
/// 定位由本函数设置（PositionType::Absolute + 左上 8px 锚定）。
pub(crate) fn spawn_debug_view(world: &mut World, ctx: &UiCtx) {
  let c = &ctx.theme.colors;
  let m = &ctx.theme.metrics;
  // 帧时长折线图画布（透明背景，折线区域由画布 1px border 包围）：GPU / CPU 各一
  let gpu_plot_image = world
    .resource_mut::<Assets<Image>>()
    .add(blank_plot_image(FRAME_PLOT_W, FRAME_PLOT_H));
  let cpu_plot_image = world
    .resource_mut::<Assets<Image>>()
    .add(blank_plot_image(FRAME_PLOT_W, FRAME_PLOT_H));
  // 绝对定位根：定宽 + 高度 auto（TabView fit_content 随活动页收缩）；外框/底色本节点提供
  world
    .spawn((
      Name::new("debug-view"),
      DemoUiRoot,
      Node {
        position_type: PositionType::Absolute,
        left: px(8.0),
        top: px(8.0),
        width: px(DEBUG_VIEW_W),
        flex_direction: FlexDirection::Column,
        border: UiRect::all(px(m.border_width)),
        ..default()
      },
      BackgroundColor(color_of(&c.surface_card_hud)),
      BorderColor::all(color_of(&c.border)),
    ))
    .with_children(|root| {
      let tv = tab_view(
        ctx,
        root,
        TabConfig {
          tabs: vec!["Stats".into(), "DDGI".into()],
          active: 0,
          fit_content: true,
        },
      );
      // ============ Tab 0：Stats ============
      root
        .world_mut()
        .entity_mut(tv.contents[0])
        .with_children(|page| {
          let g = tab_page_grid(ctx, page);
          page.world_mut().entity_mut(g).with_children(|g| {
            // ---- FPS 关键读数（主文本色 text_primary） ----
            let c1 = grid_cell(ctx, g, PanelSurface::Card);
            g.world_mut().entity_mut(c1).with_children(|cell| {
              let e = label(
                ctx,
                cell,
                LabelConfig {
                  text: "FPS: CUR ---, AVG ---, MIN ---, MAX ---".into(),
                  ..default()
                },
              );
              cell
                .world_mut()
                .entity_mut(*e)
                .insert((FpsText, TextColor(color_of(&c.text_primary))));
            });
            // ---- GPU 帧时折线（强调蓝；muted 标题 + YAxis ms；宽度 Auto 撑满格子）----
            let c2 = grid_cell(ctx, g, PanelSurface::Card);
            g.world_mut()
              .entity_mut(c2)
              .insert(Node {
                flex_direction: FlexDirection::Column,
                row_gap: px(ctx.theme.metrics.spacing.sm),
                ..default()
              })
              .with_children(|cell| {
                label(
                  ctx,
                  cell,
                  LabelConfig {
                    text: "GPU".into(),
                    style: LabelStyle::Muted,
                    ..default()
                  },
                );
                let p = plot(
                  ctx,
                  cell,
                  PlotConfig {
                    layout: PlotLayout::YAxis,
                    image: gpu_plot_image,
                    capacity: FRAME_PLOT_CAP,
                    y_domain: PlotDomain::Auto,
                    line_color: color_of(&c.accent_text),
                    y_axis_width: Val::Auto,
                    canvas_width: Val::Auto,
                    unit: Some("ms"),
                    canvas_h: FRAME_PLOT_H as f32,
                  },
                );
                cell.world_mut().entity_mut(*p).insert(GpuPlot);
              });
            // ---- CPU 工作时长折线（warning 色；gate-render 实测 ms，不含 acquire 等待）----
            let c3 = grid_cell(ctx, g, PanelSurface::Card);
            g.world_mut()
              .entity_mut(c3)
              .insert(Node {
                flex_direction: FlexDirection::Column,
                row_gap: px(ctx.theme.metrics.spacing.sm),
                ..default()
              })
              .with_children(|cell| {
                label(
                  ctx,
                  cell,
                  LabelConfig {
                    text: "CPU".into(),
                    style: LabelStyle::Muted,
                    ..default()
                  },
                );
                let p = plot(
                  ctx,
                  cell,
                  PlotConfig {
                    layout: PlotLayout::YAxis,
                    image: cpu_plot_image,
                    capacity: FRAME_PLOT_CAP,
                    y_domain: PlotDomain::Auto,
                    line_color: color_of(&c.warning),
                    y_axis_width: Val::Auto,
                    canvas_width: Val::Auto,
                    unit: Some("ms"),
                    canvas_h: FRAME_PLOT_H as f32,
                  },
                );
                cell.world_mut().entity_mut(*p).insert(CpuPlot);
              });
            // ---- 相机当前位置/角度信息（说明文字 muted） ----
            let c4 = grid_cell(ctx, g, PanelSurface::Card);
            g.world_mut().entity_mut(c4).with_children(|cell| {
              let e = label(
                ctx,
                cell,
                LabelConfig {
                  text: "(---,---,---)->(---,---,---)".into(),
                  style: LabelStyle::Muted,
                },
              );
              cell.world_mut().entity_mut(*e).insert(CamInfoText);
            });
            // ---- 垂直同步开关（默认开 = Fifo；观察者写 Window.present_mode，
            //      bevy_render 检测变化自动重配 swapchain；关 = AutoNoVsync 不封顶） ----
            let c5 = grid_cell(ctx, g, PanelSurface::Card);
            g.world_mut().entity_mut(c5).with_children(|cell| {
              let t = toggle_switch(
                ctx,
                cell,
                ToggleSwitchConfig {
                  text: Some("VSync".into()),
                  checked: true, // 默认开，与启动 present_mode=Fifo 同步
                },
              );
              cell.world_mut().entity_mut(*t).insert(VsyncToggle);
            });
            // ---- 「右上角面板」显隐开关（默认隐藏；观察者写 ShowcaseRoot Visibility）----
            let c6 = grid_cell(ctx, g, PanelSurface::Card);
            g.world_mut().entity_mut(c6).with_children(|cell| {
              let t = toggle_switch(
                ctx,
                cell,
                ToggleSwitchConfig {
                  text: Some("UI Showcase".into()),
                  checked: false, // 默认隐藏 showcase，与 spawn_showcase 初始 Hidden 同步
                },
              );
              cell
                .world_mut()
                .entity_mut(*t)
                .insert(ShowcaseVisibilityToggle);
            });
          });
        });
      // ============ Tab 1：DDGI 调试组 ============
      root
        .world_mut()
        .entity_mut(tv.contents[1])
        .with_children(|page| {
          let g = tab_page_grid(ctx, page);
          page.world_mut().entity_mut(g).with_children(|g| {
            // 全部 DDGI 控件合并在同一个 cell 内纵向排列，子行间距 sm=4px：
            // 阶段档位滑杆 / 调试模式按钮行 / Gain 滑杆行 / Probe Viz 开关 / LOD 滑杆行
            let cell = grid_cell(ctx, g, PanelSurface::Card);
            g.world_mut()
              .entity_mut(cell)
              .insert(Node {
                flex_direction: FlexDirection::Column,
                row_gap: px(ctx.theme.metrics.spacing.sm),
                padding: UiRect::all(px(ctx.theme.metrics.spacing.sm)),
                ..default()
              })
              .with_children(|cell| {
            // -- 阶段档位滑杆（0=Off/1=Active/2=Cast/3=Full，step=1 整数吸附；
            //    默认 0 关，与 DdgiStage::default()=OFF 同步）--
            cell
              .spawn((
                Name::new("ddgi-stage-row"),
                Node {
                  flex_direction: FlexDirection::Row,
                  column_gap: px(ctx.theme.metrics.spacing.md),
                  align_items: AlignItems::Center,
                  ..default()
                },
              ))
              .with_children(|row| {
                label(
                  &ctx,
                  row,
                  LabelConfig {
                    text: "DDGI".into(),
                    style: LabelStyle::Muted,
                    ..default()
                  },
                );
                let s = slider(
                  &ctx,
                  row,
                  SliderConfig {
                    min: 0.0,
                    max: 3.0,
                    value: 0.0,
                    step: Some(1.0),
                  },
                );
                row.world_mut().entity_mut(*s).insert(DdgiStageSlider);
                let vl = label(
                  &ctx,
                  row,
                  LabelConfig {
                    text: "0 Off".into(),
                    style: LabelStyle::Muted,
                    ..default()
                  },
                );
                row
                  .world_mut()
                  .entity_mut(*vl)
                  .insert(DdgiStageValueLabel);
              });
            // -- 调试模式按钮行（5 按钮互斥单选；默认 mode=0 Normal = Primary）--
            cell
              .spawn((
                Name::new("ddgi-mode-row"),
                Node {
                  flex_direction: FlexDirection::Row,
                  column_gap: px(ctx.theme.metrics.spacing.sm),
                  align_items: AlignItems::Center,
                  ..default()
                },
              ))
              .with_children(|row| {
                for (i, name) in DDGI_DEBUG_MODES.iter().enumerate() {
                  let variant = if i == 0 {
                    ButtonVariant::Primary
                  } else {
                    ButtonVariant::Ghost
                  };
                  let b = button(
                    &ctx,
                    row,
                    ButtonConfig {
                      text: (*name).into(),
                      variant,
                    },
                  );
                  row
                    .world_mut()
                    .entity_mut(*b)
                    .insert(DdgiDebugModeBtn(i as u8));
                }
              });
            // -- Gain 滑杆行（0.1..4.0，默认 1.0；右侧实时数值）--
            cell
              .spawn((
                Name::new("ddgi-gain-row"),
                Node {
                  flex_direction: FlexDirection::Row,
                  column_gap: px(ctx.theme.metrics.spacing.md),
                  align_items: AlignItems::Center,
                  ..default()
                },
              ))
              .with_children(|row| {
                label(
                  &ctx,
                  row,
                  LabelConfig {
                    text: "Gain".into(),
                    style: LabelStyle::Muted,
                    ..default()
                  },
                );
                let s = slider(
                  &ctx,
                  row,
                  SliderConfig {
                    min: 0.1,
                    max: 4.0,
                    value: 1.0,
                    step: Some(0.1),
                  },
                );
                row.world_mut().entity_mut(*s).insert(DdgiGainSlider);
                let vl = label(
                  &ctx,
                  row,
                  LabelConfig {
                    text: "1.0".into(),
                    style: LabelStyle::Muted,
                    ..default()
                  },
                );
                row
                  .world_mut()
                  .entity_mut(*vl)
                  .insert(DdgiGainValueLabel);
              });
            // -- Probe Viz 开关（探针位置黄色方块可视化，默认关）--
            let t = toggle_switch(
              &ctx,
              cell,
              ToggleSwitchConfig {
                text: Some("Probe Viz".into()),
                checked: false,
              },
            );
            cell.world_mut().entity_mut(*t).insert(DdgiProbeVizToggle);
            // -- Probe Viz 层级滑杆行（0..=4 步进 1：All / LOD0~3）--
            cell
              .spawn((
                Name::new("ddgi-probe-lod-row"),
                Node {
                  flex_direction: FlexDirection::Row,
                  column_gap: px(ctx.theme.metrics.spacing.md),
                  align_items: AlignItems::Center,
                  ..default()
                },
              ))
              .with_children(|row| {
                label(
                  &ctx,
                  row,
                  LabelConfig {
                    text: "LOD".into(),
                    style: LabelStyle::Muted,
                    ..default()
                  },
                );
                let s = slider(
                  &ctx,
                  row,
                  SliderConfig {
                    min: 0.0,
                    max: 4.0,
                    value: 0.0,
                    step: Some(1.0),
                  },
                );
                row
                  .world_mut()
                  .entity_mut(*s)
                  .insert(DdgiProbeVizLodSlider);
                let vl = label(
                  &ctx,
                  row,
                  LabelConfig {
                    text: DDGI_PROBE_VIZ_LODS[0].into(),
                    style: LabelStyle::Muted,
                    ..default()
                  },
                );
                row
                  .world_mut()
                  .entity_mut(*vl)
                  .insert(DdgiProbeVizLodValueLabel);
              });
          });
        });
      });
    });

  // 「右上角面板」开关 → 切换 showcase 整体显隐（ShowcaseRoot 的 Visibility）
  world.add_observer(
    |ev: On<ToggleSwitchToggled>,
     q_toggle: Query<(), With<ShowcaseVisibilityToggle>>,
     mut q_root: Query<&mut Visibility, With<ShowcaseRoot>>| {
      if q_toggle.get(ev.entity).is_err() {
        return;
      }
      if let Ok(mut vis) = q_root.single_mut() {
        *vis = if ev.checked {
          Visibility::Visible
        } else {
          Visibility::Hidden
        };
      }
    },
  );

  // 「VSync」开关 → 切换主窗口 PresentMode（开 = Fifo 硬垂直同步，vblank 墙钟节拍
  // 稳定帧时；关 = AutoNoVsync 不封顶，测裸 GPU 吞吐）。bevy_winit 不处理此字段，
  // bevy_render 每帧 extract 检测 present_mode 变化后自动重配 swapchain
  world.add_observer(
    |ev: On<ToggleSwitchToggled>,
     q_toggle: Query<(), With<VsyncToggle>>,
     mut q_win: Query<&mut Window>| {
      if q_toggle.get(ev.entity).is_err() {
        return;
      }
      let Ok(mut win) = q_win.single_mut() else {
        return;
      };
      win.present_mode = if ev.checked {
        PresentMode::Fifo
      } else {
        PresentMode::AutoNoVsync
      };
      info!("VSync {} → present_mode {:?}", if ev.checked { "on" } else { "off" }, win.present_mode);
    },
  );

  // DDGI 阶段档位滑杆 → 写 DdgiStage 主世界资源（extract 每帧拷到渲染世界）：
  // 0=关 dispatch 早退；1=只跑 active probe 选择；2=+RayQuery cast/update；
  // 3=+voxel 着色采样完整 GI（trace shader misc.x gi 开关）。step=1 吸附，
  // round 后取整；同步刷新右侧 "N Name" 档位名标签
  world.add_observer(
    |ev: On<SliderValueChanged>,
     q_slider: Query<(), With<DdgiStageSlider>>,
     mut q_label: Query<&mut Text, With<DdgiStageValueLabel>>,
     mut ddgi: ResMut<gate_render::ddgi::DdgiStage>| {
      if q_slider.get(ev.entity).is_err() {
        return;
      }
      let v = ev.value.round() as u8;
      *ddgi = gate_render::ddgi::DdgiStage::new(v);
      if let Ok(mut t) = q_label.single_mut() {
        let i = (v as usize).min(DDGI_STAGES.len() - 1);
        t.0 = format!("{} {}", v, DDGI_STAGES[i]);
      }
      info!("DDGI stage → {} ({})", ddgi.0, DDGI_STAGES.get(v as usize).unwrap_or(&"?"));
    },
  );

  // 「Probe Viz」开关 → 写 DdgiDebugSettings.probe_viz（渲染世界 probe_viz_main dispatch）
  world.add_observer(
    |ev: On<ToggleSwitchToggled>,
     q_toggle: Query<(), With<DdgiProbeVizToggle>>,
     mut dbg: ResMut<gate_render::ddgi::DdgiDebugSettings>| {
      if q_toggle.get(ev.entity).is_err() {
        return;
      }
      dbg.probe_viz = ev.checked;
    },
  );

  // DDGI 调试模式按钮 → 写 DdgiDebugSettings.mode + 互斥切换按钮变体（选中=Primary）
  world.add_observer(
    |ev: On<UiClick>,
     q_btn: Query<&DdgiDebugModeBtn>,
     mut q_all: Query<(Entity, &mut ButtonVariant), With<DdgiDebugModeBtn>>,
     mut dbg: ResMut<gate_render::ddgi::DdgiDebugSettings>| {
      let Ok(btn) = q_btn.get(ev.entity) else {
        return;
      };
      let mode = btn.0 as f32;
      dbg.mode = mode;
      // 互斥：选中按钮 Primary，其余 Ghost
      for (e, mut var) in &mut q_all.iter_mut() {
        if e == ev.entity {
          *var = ButtonVariant::Primary;
        } else {
          *var = ButtonVariant::Ghost;
        }
      }
      let name = DDGI_DEBUG_MODES.get(btn.0 as usize).unwrap_or(&"?");
      info!("DDGI debug mode → {} ({})", mode, name);
    },
  );

  // DDGI 增益滑杆 → 写 DdgiDebugSettings.gain + 刷新右侧数值标签
  world.add_observer(
    |ev: On<SliderValueChanged>,
     q_slider: Query<(), With<DdgiGainSlider>>,
     mut q_label: Query<&mut Text, With<DdgiGainValueLabel>>,
     mut dbg: ResMut<gate_render::ddgi::DdgiDebugSettings>| {
      if q_slider.get(ev.entity).is_err() {
        return;
      }
      dbg.gain = ev.value;
      if let Ok(mut t) = q_label.single_mut() {
        t.0 = format!("{:.1}", ev.value);
      }
    },
  );

  // Probe Viz 层级滑杆 → 写 DdgiDebugSettings.probe_viz_lod + 刷新右侧层级名标签
  world.add_observer(
    |ev: On<SliderValueChanged>,
     q_slider: Query<(), With<DdgiProbeVizLodSlider>>,
     mut q_label: Query<&mut Text, With<DdgiProbeVizLodValueLabel>>,
     mut dbg: ResMut<gate_render::ddgi::DdgiDebugSettings>| {
      if q_slider.get(ev.entity).is_err() {
        return;
      }
      dbg.probe_viz_lod = ev.value;
      if let Ok(mut t) = q_label.single_mut() {
        let i = (ev.value.round() as usize).min(DDGI_PROBE_VIZ_LODS.len() - 1);
        t.0 = DDGI_PROBE_VIZ_LODS[i].into();
      }
    },
  );
}

/// Tab 页内网格：1 列 auto 行高 + 1px gap 填色格线。**去外框**——面板 root
/// 已提供外框，grid 自带全边框会叠成双线；容器 border 底色保留（填 gap 成格线）。
fn tab_page_grid(ctx: &UiCtx, page: &mut ChildSpawner) -> Entity {
  let g = grid(
    ctx,
    page,
    GridConfig {
      columns: 1,
      row_height: None,
    },
  );
  page.world_mut().entity_mut(*g).remove::<BorderColor>();
  if let Some(mut n) = page.world_mut().get_mut::<Node>(*g) {
    n.border = UiRect::DEFAULT;
  }
  *g
}

/// F3 切换左上角 debug overlay 显隐（默认显示；只影响本面板，不动 showcase）
pub(crate) fn debug_overlay_toggle(
  keys: Res<ButtonInput<KeyCode>>,
  mut q: Query<&mut Visibility, With<DemoUiRoot>>,
) {
  if keys.just_pressed(KeyCode::F3) {
    for mut vis in &mut q {
      *vis = if *vis == Visibility::Visible {
        Visibility::Hidden
      } else {
        Visibility::Visible
      };
    }
  }
}

/// 每 0.25s 刷新一次左上角 FPS 行 + 写一行到 logs/fps.log
///
/// 每帧（早于 0.25s 早退）压入两条折线环形缓冲，PlotData Changed → gate-ui 的
/// plot_redraw_system 自动光栅化重绘：
/// - GPU 折线：各 pass span 之和 ms（诊断未就绪跳过，绝不退回 dt）
/// - CPU 折线：gate-render 实测的本帧 CPU 工作时长 ms（两段纯 CPU 区间相加，
///   vsync/帧队列反压在 prepare_windows acquire 处的阻塞已排除；晚 1 帧，无值跳过）
#[allow(clippy::too_many_arguments)]
pub(crate) fn fps_line_feed(
  time: Res<Time>,
  store: Option<Res<bevy::diagnostic::DiagnosticsStore>>,
  cpu_report: Option<Res<gate_render::cpu_probe::CpuWorkReport>>,
  orbit: Res<OrbitCamera>,
  mut q: ParamSet<(
    Query<&mut Text, With<FpsText>>,
    Query<&mut Text, With<CamInfoText>>,
  )>,
  mut q_plot: ParamSet<(
    Query<&mut PlotData, With<GpuPlot>>,
    Query<&mut PlotData, With<CpuPlot>>,
  )>,
  mut window: Local<VecDeque<f32>>, // 逐帧 delta，按时间裁剪到 5s
  mut acc: Local<f32>,
  mut frames: Local<u32>,
  mut log_file: Local<Option<std::fs::File>>,
  mut ft_log: Local<Option<std::fs::File>>,
  mut ft_buf: Local<String>,
) {
  // 首次调用：创建/截断 fps.log + frame_time.log
  if log_file.is_none() {
    let path = std::path::Path::new(FPS_LOG_PATH);
    if let Some(parent) = path.parent() {
      std::fs::create_dir_all(parent).ok();
    }
    *log_file = std::fs::File::create(path).ok();
    *ft_log = std::fs::File::create(FRAME_TIME_LOG_PATH).ok();
  }
  let dt = time.delta_secs();
  // 折线图推真实 GPU 帧时 = trace+beam+ddgi+blit 四 pass span 之和（跨系统嵌套 span
  // 不可行：bevy_render open_spans 按 thread 分栈，并行 executor 下跨节点配对 panic）。
  // vsync 下 delta_secs 恒 ≈16.7（vblank 节拍），不反映真实工作量。
  // 诊断未就绪（全 -1）→ 跳过推入，绝不退回 delta（否则 vsync 下 16.7 混进 auto 域
  // 把真实曲线压在底部）。诊断约 0.7s 后上线，空白期折线图不动即可。
  //
  // **新鲜度门控**：`DiagnosticsStore` 中一个 span 诊断一旦被创建，其最近测量值会
  // 永久残留——span 不再录制（如 DDGI 档位 0 早退，gate_ddgi_total 不写时间戳）后
  // `Diagnostic::value()` 仍返回最后一次值。实测 DDGI 关闭后 51.4ms 冷启动残值被
  // 折线图当成当前值，800fps 下曲线仍贴在 52ms。GPU 回读延迟只有 1-2 帧（<33ms），
  // 超过 GPU_DIAG_MAX_AGE 未更新的测量一律视为失效（-1），杜绝残值。
  use std::fmt::Write as _;
  let now = std::time::Instant::now();
  let gpu_ms = |path: &'static str| -> f32 {
    store
      .as_deref()
      .and_then(|s| s.get_measurement(&bevy::diagnostic::DiagnosticPath::new(path)))
      .filter(|m| now.duration_since(m.time).as_secs_f32() < GPU_DIAG_MAX_AGE)
      .map(|m| m.value as f32)
      .unwrap_or(-1.0)
  };
  let parts = [
    gpu_ms("render/gate_dda_trace/elapsed_gpu"),
    gpu_ms("render/gate_beam/elapsed_gpu"),
    gpu_ms("render/gate_ddgi_total/elapsed_gpu"),
    gpu_ms("render/gate_probe_viz/elapsed_gpu"),
    gpu_ms("render/gate_dda_blit/elapsed_gpu"),
  ];
  let diag_ready = parts.iter().any(|&v| v >= 0.0);
  if diag_ready {
    let v: f32 = parts.iter().filter(|&&v| v >= 0.0).sum();
    if let Ok(mut plot) = q_plot.p0().single_mut() {
      plot.push(v);
    }
  }
  // CPU 折线：gate-render 实测的本帧 CPU 工作时长（ms）——主世界 First → render
  // 侧 acquire 前 + acquire 后 → RenderGraph Finish 两段纯 CPU 区间，vsync 下
  // prepare_windows 的 vblank/反压阻塞不包含在内（详见 gate-render cpu_probe 模块
  // 文档）。render 侧 Finish 集产出，主世界晚 1 帧 try_recv；无值跳过不补 dt
  //（dt 含 present 等待，vsync 下恒 16.7 会把真实曲线压平）。
  let mut cpu_timing: Option<gate_render::cpu_probe::CpuTiming> = None;
  if let Some(report) = cpu_report.as_deref()
    && let Ok(mut guard) = report.0.lock()
    && let Some(rx) = guard.as_mut()
  {
    while let Ok(v) = rx.try_recv() {
      cpu_timing = Some(v);
    }
  }
  if let Some(t) = cpu_timing
    && let Ok(mut plot) = q_plot.p1().single_mut()
  {
    plot.push(t.total);
  }
  window.push_back(dt);
  // 逐帧一行：elapsed,dt,5×gpu spans,cpu_total,seg_main,seg_pre,seg_acq,seg_prep,seg_graph,seg_submit
  //（-1 = 该来源未上线；seg_acq = acquire/vblank 等待段，不计入 cpu_total），随 0.25s 刷盘
  if diag_ready {
    let _ = write!(
      *ft_buf,
      "{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3}",
      time.elapsed_secs(),
      dt * 1000.0,
      parts[0],
      parts[1],
      parts[2],
      parts[3],
      parts[4],
    );
  } else {
    let _ = write!(*ft_buf, "{:.3},{:.3},-1,-1,-1,-1,-1", time.elapsed_secs(), dt * 1000.0);
  }
  if let Some(t) = cpu_timing {
    let _ = writeln!(
      *ft_buf,
      ",{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3}",
      t.total, t.seg_main, t.seg_pre, t.seg_acq, t.seg_prep, t.seg_graph, t.seg_submit
    );
  } else {
    let _ = writeln!(*ft_buf, ",-1,-1,-1,-1,-1,-1,-1");
  };
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
  let elapsed = time.elapsed_secs();
  *acc = 0.0;
  *frames = 0;
  if let Ok(mut t) = q.p0().single_mut()
    && t.0 != txt
  {
    t.0 = txt;
  }
  // 相机信息：眼位 -> 目标点
  let eye = orbit.eye();
  let tgt = orbit.target;
  let cam_txt = format!(
    "({:.1}, {:.1}, {:.1}) -> ({:.1}, {:.1}, {:.1})",
    eye.x, eye.y, eye.z, tgt.x, tgt.y, tgt.z
  );
  if let Some(mut t) = q.p1().iter_mut().next()
    && t.0 != cam_txt
  {
    t.0 = cam_txt;
  }
  // 写 fps.log：elapsed_secs,CUR,AVG,MIN,MAX
  use std::io::Write;
  if let Some(f) = log_file.as_mut() {
    let _ = writeln!(
      f,
      "{:.2},{},{},{},{}",
      elapsed,
      fps3(cur),
      fps3(avg),
      fps3(min),
      fps3(max)
    );
  }
  // 刷逐帧帧时批次（frame_time.log）
  if let Some(f) = ft_log.as_mut()
    && !ft_buf.is_empty()
  {
    let _ = f.write_all(ft_buf.as_bytes());
    ft_buf.clear();
  }
}
