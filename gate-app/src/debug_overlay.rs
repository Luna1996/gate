//! 左上角调试 overlay：FPS（CUR/AVG/MIN/MAX）+ 真实 GPU 帧时折线 + 相机信息。
//!
//! 装配入口 [`demo_ui_setup`](crate::demo_ui_setup) 在主题/字体就绪后调用
//! [`spawn_debug_view`] 一次性生成；[`fps_line_feed`] 每帧喂统计数据。

use std::collections::VecDeque;

use bevy::prelude::*;

use gate_render::OrbitCamera;
use gate_ui::{
  UiCtx,
  widgets::{
    ButtonConfig, ButtonVariant, GridConfig, LabelConfig, LabelStyle, PanelSurface, PlotConfig,
    PlotData, PlotDomain, PlotLayout, SliderConfig, SliderValueChanged, ToggleSwitchConfig,
    ToggleSwitchToggled, UiClick, blank_plot_image, button, color_of, grid, grid_cell, label, plot,
    px, slider, toggle_switch,
  },
};

use crate::showcase::ShowcaseRoot;

/// fps_line_feed 每 0.25s 写一行 CUR/AVG/MIN/MAX
pub(crate) const FPS_LOG_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/logs/fps.log");
/// 逐帧真实帧时日志：每帧一行 `elapsed_secs,frame_ms`（vsync 下为墙钟帧时间，
/// 0.25s 攒一批刷盘）。帧时波动/周期尖刺定位用
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

/// 帧时长折线图：画布纹理尺寸（宽 ≈ FPS 文本宽；高 64px，1:1 显示）
const FRAME_PLOT_W: u32 = 320;
const FRAME_PLOT_H: u32 = 64;
/// 折线环形样本容量（≈ 画布像素宽，约 1 样本/px；60fps 下约 6s 窗口）
const FRAME_PLOT_CAP: usize = 360;

#[derive(Component)]
pub(crate) struct FpsText;

/// demo UI 已 spawn 的根标记：demo_ui_setup 的存在性守卫（查到即跳过），
/// 替代 Local<bool> done 平行状态——产物本身就是"是否已跑过"的真源
#[derive(Component)]
pub(crate) struct DemoUiRoot;

#[derive(Component)]
pub(crate) struct CamInfoText;

/// 左上角 FPS 帧时折线标记（fps_line_feed 用它过滤 PlotData，避免命中 showcase 折线）
#[derive(Component)]
pub(crate) struct FpsPlot;

/// 「右上角面板」开关标记（ToggleSwitchToggled 观察者以此过滤事件）
#[derive(Component)]
struct ShowcaseVisibilityToggle;

/// 「DDGI 间接光」开关标记（观察者写 gate_render::ddgi::DdgiEnabled 主世界资源）
#[derive(Component)]
struct DdgiGiToggle;

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

/// Probe Viz 层级名（滑杆 stop 序：0=All, 1..=4=LOD0~3, 5=Base；
/// Douglas #23 = base 烘焙网格 + 4 LOD，视频切换的 LOD 0~3 即 4 个下采样级）
const DDGI_PROBE_VIZ_LODS: [&str; 6] = ["All", "LOD 0", "LOD 1", "LOD 2", "LOD 3", "Base"];

/// 调试模式名（按钮文本 + 日志用）
const DDGI_DEBUG_MODES: [&str; 5] = ["Normal", "GI", "wsum", "Domain", "Probe"];

/// fps → 3 位宽显示值（上限 999，防 4 位数抖动）
pub(crate) fn fps3(v: f32) -> u32 {
  (v.round() as u32).min(999)
}

/// 左上角 debug-view 面板（grid 布局 1 列 5 行：FPS 行 / 帧时长折线 / 相机信息 /
/// showcase 显隐开关 / DDGI 间接光开关），在 commands.queue 闭包内调用（直接操作 World）。
///
/// 视觉全部由 grid 提供：容器 border = 外框，1px gap = 行间格线，
/// cell 用不透明 Card 面盖出格子（grid_cell 约定：HUD 档半透明会露格线）。
/// 定位由本函数设置（PositionType::Absolute + 左上 8px 锚定）。
pub(crate) fn spawn_debug_view(world: &mut World, ctx: &UiCtx) {
  let c = &ctx.theme.colors;
  // 帧时长折线图画布纹理（透明背景，折线区域由画布 1px border 包围）
  let plot_image = world
    .resource_mut::<Assets<Image>>()
    .add(blank_plot_image(FRAME_PLOT_W, FRAME_PLOT_H));
  // 绝对定位根：只负责定位，视觉（外框/格线/格子底色）全部由 grid 提供
  world
    .spawn((
      Name::new("debug-view"),
      DemoUiRoot,
      Node {
        position_type: PositionType::Absolute,
        left: px(8.0),
        top: px(8.0),
        ..default()
      },
    ))
    .with_children(|root| {
      // 1 列 5 行：FPS / 折线 / 相机 / showcase 开关 / DDGI 开关，行间 1px 格线
      let g = grid(
        &ctx,
        root,
        GridConfig {
          columns: 1,
          row_height: None,
        },
      );
      root.world_mut().entity_mut(*g).with_children(|g| {
        // ---- 行 1：FPS 关键读数（主文本色 text_primary） ----
        let c1 = grid_cell(&ctx, g, PanelSurface::Card);
        g.world_mut().entity_mut(c1).with_children(|cell| {
          let e = label(
            &ctx,
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
        // ---- 行 2：帧时长（ms）折线，强调蓝折线 + 左侧纵轴标签（muted）、透明底、
        //      1px border 包围折线区域；宽度约束 Auto = 撑满格子剩余宽 ----
        let c2 = grid_cell(&ctx, g, PanelSurface::Card);
        g.world_mut().entity_mut(c2).with_children(|cell| {
          let fps_plot = plot(
            &ctx,
            cell,
            PlotConfig {
              layout: PlotLayout::YAxis,
              image: plot_image,
              capacity: FRAME_PLOT_CAP,
              y_domain: PlotDomain::Auto,
              line_color: color_of(&c.accent_text),
              y_axis_width: Val::Auto,
              canvas_width: Val::Auto,
              unit: None,
              canvas_h: FRAME_PLOT_H as f32,
            },
          );
          // 标记 FPS 折线，fps_line_feed 的查询靠它过滤（否则匹配不到/匹配多个）
          cell.world_mut().entity_mut(*fps_plot).insert(FpsPlot);
        });
        // ---- 行 3：相机当前位置/角度信息（说明文字 muted） ----
        let c3 = grid_cell(&ctx, g, PanelSurface::Card);
        g.world_mut().entity_mut(c3).with_children(|cell| {
          let e = label(
            &ctx,
            cell,
            LabelConfig {
              text: "(---,---,---)->(---,---,---)".into(),
              style: LabelStyle::Muted,
            },
          );
          cell.world_mut().entity_mut(*e).insert(CamInfoText);
        });
        // ---- 行 4：「右上角面板」显隐开关（默认开；观察者写 ShowcaseRoot Visibility） ----
        let c4 = grid_cell(&ctx, g, PanelSurface::Card);
        g.world_mut().entity_mut(c4).with_children(|cell| {
          let t = toggle_switch(
            &ctx,
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
        // ---- 行 5：DDGI 间接光开关（默认开；观察者写 DdgiEnabled 资源 → 渲染链路） ----
        let c5 = grid_cell(&ctx, g, PanelSurface::Card);
        g.world_mut().entity_mut(c5).with_children(|cell| {
          let t = toggle_switch(
            &ctx,
            cell,
            ToggleSwitchConfig {
              text: Some("DDGI GI".into()),
              checked: true, // 默认开，与 DdgiEnabled::default()=true 同步
            },
          );
          cell.world_mut().entity_mut(*t).insert(DdgiGiToggle);
        });
        // ---- 行 6：DDGI 调试模式（5 按钮互斥单选；写 DdgiDebugSettings.mode）----
        // 默认 mode=0（Normal）→ 第 0 个按钮 Primary（选中态），其余 Ghost。
        // 直接把 cell 设为 row 方向（不套中间容器，避免 grid cell 内 flex 宽度算异常）
        let c6 = grid_cell(&ctx, g, PanelSurface::Card);
        g.world_mut()
          .entity_mut(c6)
          .insert(Node {
            flex_direction: FlexDirection::Row,
            column_gap: px(4.0),
            align_items: AlignItems::Center,
            padding: UiRect::all(px(ctx.theme.metrics.spacing.sm)),
            ..default()
          })
          .with_children(|cell| {
            for (i, name) in DDGI_DEBUG_MODES.iter().enumerate() {
              let variant = if i == 0 {
                ButtonVariant::Primary
              } else {
                ButtonVariant::Ghost
              };
              let b = button(
                &ctx,
                cell,
                ButtonConfig {
                  text: (*name).into(),
                  variant,
                },
              );
              cell
                .world_mut()
                .entity_mut(*b)
                .insert(DdgiDebugModeBtn(i as u8));
            }
          });
        // ---- 行 7：DDGI 增益滑杆（0.1..4.0，默认 1.0；写 DdgiDebugSettings.gain）----
        let c7 = grid_cell(&ctx, g, PanelSurface::Card);
        g.world_mut()
          .entity_mut(c7)
          .insert(Node {
            flex_direction: FlexDirection::Row,
            column_gap: px(8.0),
            align_items: AlignItems::Center,
            padding: UiRect::all(px(ctx.theme.metrics.spacing.sm)),
            ..default()
          })
          .with_children(|cell| {
            label(
              &ctx,
              cell,
              LabelConfig {
                text: "Gain".into(),
                style: LabelStyle::Muted,
                ..default()
              },
            );
            let s = slider(
              &ctx,
              cell,
              SliderConfig {
                min: 0.1,
                max: 4.0,
                value: 1.0,
                step: Some(0.1),
              },
            );
            cell.world_mut().entity_mut(*s).insert(DdgiGainSlider);
            // 右侧实时数值标签
            let vl = label(
              &ctx,
              cell,
              LabelConfig {
                text: "1.0".into(),
                style: LabelStyle::Muted,
                ..default()
              },
            );
            cell
              .world_mut()
              .entity_mut(*vl)
              .insert(DdgiGainValueLabel);
          });
        // ---- 行 8：Probe Viz 开关（探针位置黄色方块可视化；写 DdgiDebugSettings.probe_viz）----
        let c8 = grid_cell(&ctx, g, PanelSurface::Card);
        g.world_mut().entity_mut(c8).with_children(|cell| {
          let t = toggle_switch(
            &ctx,
            cell,
            ToggleSwitchConfig {
              text: Some("Probe Viz".into()),
              checked: false,
            },
          );
          cell.world_mut().entity_mut(*t).insert(DdgiProbeVizToggle);
        });
        // ---- 行 9：Probe Viz 层级滑杆（0..=5 步进 1：All / LOD0~3 / Base；
        //      写 DdgiDebugSettings.probe_viz_lod）----
        let c9 = grid_cell(&ctx, g, PanelSurface::Card);
        g.world_mut()
          .entity_mut(c9)
          .insert(Node {
            flex_direction: FlexDirection::Row,
            column_gap: px(8.0),
            align_items: AlignItems::Center,
            padding: UiRect::all(px(ctx.theme.metrics.spacing.sm)),
            ..default()
          })
          .with_children(|cell| {
            label(
              &ctx,
              cell,
              LabelConfig {
                text: "LOD".into(),
                style: LabelStyle::Muted,
                ..default()
              },
            );
            let s = slider(
              &ctx,
              cell,
              SliderConfig {
                min: 0.0,
                max: 5.0,
                value: 0.0,
                step: Some(1.0),
              },
            );
            cell.world_mut().entity_mut(*s).insert(DdgiProbeVizLodSlider);
            // 右侧实时层级名标签
            let vl = label(
              &ctx,
              cell,
              LabelConfig {
                text: DDGI_PROBE_VIZ_LODS[0].into(),
                style: LabelStyle::Muted,
                ..default()
              },
            );
            cell
              .world_mut()
              .entity_mut(*vl)
              .insert(DdgiProbeVizLodValueLabel);
          });
      });
      // grid 默认 width: Percent(100) → 绝对定位根下改为收缩到内容宽（shrink-to-fit）
      if let Some(mut n) = root.world_mut().get_mut::<Node>(*g) {
        n.width = Val::Auto;
      }
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

  // 「DDGI GI」开关 → 写 DdgiEnabled 主世界资源（extract 每帧拷到渲染世界：
  // 关 = dispatch compute 链早退 + trace shader 跳过探针采样 gi=0）
  world.add_observer(
    |ev: On<ToggleSwitchToggled>,
     q_toggle: Query<(), With<DdgiGiToggle>>,
     mut ddgi: ResMut<gate_render::ddgi::DdgiEnabled>| {
      if q_toggle.get(ev.entity).is_err() {
        return;
      }
      ddgi.0 = ev.checked;
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
/// 每帧（早于 0.25s 早退）把真实 GPU 帧时 ms（gate_frame span）压入折线图环形缓冲：
/// PlotData Changed → gate-ui 的 plot_redraw_system 自动光栅化重绘。
#[allow(clippy::too_many_arguments)]
pub(crate) fn fps_line_feed(
  time: Res<Time>,
  store: Option<Res<bevy::diagnostic::DiagnosticsStore>>,
  orbit: Res<OrbitCamera>,
  mut q: ParamSet<(
    Query<&mut Text, With<FpsText>>,
    Query<&mut Text, With<CamInfoText>>,
  )>,
  mut q_plot: Query<&mut PlotData, With<FpsPlot>>,
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
  use std::fmt::Write as _;
  let gpu_ms = |path: &'static str| -> f32 {
    store
      .as_deref()
      .and_then(|s| s.get(&bevy::diagnostic::DiagnosticPath::new(path)))
      .and_then(|d| d.value())
      .map(|v| v as f32)
      .unwrap_or(-1.0)
  };
  let parts = [
    gpu_ms("render/gate_dda_trace/elapsed_gpu"),
    gpu_ms("render/gate_beam/elapsed_gpu"),
    gpu_ms("render/gate_ddgi_update/elapsed_gpu"),
    gpu_ms("render/gate_direct_light/elapsed_gpu"),
    gpu_ms("render/gate_dda_blit/elapsed_gpu"),
  ];
  let diag_ready = parts.iter().any(|&v| v >= 0.0);
  if diag_ready && let Ok(mut plot) = q_plot.single_mut() {
    let v: f32 = parts.iter().filter(|&&v| v >= 0.0).sum();
    plot.push(v);
  }
  window.push_back(dt);
  // 逐帧一行：elapsed,dt[,trace,beam,ddgi,blit]（-1 = 诊断未上线），随 0.25s 刷盘
  let mut line = format!("{:.3},{:.3}", time.elapsed_secs(), dt * 1000.0);
  if diag_ready {
    let _ = write!(
      line,
      ",{:.3},{:.3},{:.3},{:.3}",
      parts[0], parts[1], parts[2], parts[3]
    );
  }
  let _ = writeln!(*ft_buf, "{}", line);
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
