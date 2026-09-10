//! 左上角调试 overlay：FPS（CUR/AVG/MIN/MAX）读数 + 相机信息 + 渲染开关。
//!
//! 逐帧 GPU/CPU 剖析已改由 Tracy + wgpu-profiler 承担（`--features profile`，
//! Tracy GUI 看时间线），旧的自研折线/落盘帧时统计已删除。
//!
//! 装配入口 [`demo_ui_setup`](crate::demo_ui_setup) 在主题/字体就绪后调用
//! [`spawn_debug_view`] 一次性生成；[`fps_line_feed`] 每帧喂 FPS 统计。

use std::collections::VecDeque;

use bevy::prelude::*;
use bevy::window::{PresentMode, Window};

use gate_render::OrbitCamera;
use gate_ui::{
  UiCtx,
  widgets::{
    ButtonConfig, ButtonVariant, GridConfig, LabelConfig, LabelStyle, SliderConfig,
    SliderValueChanged, TabConfig, ToggleSwitchConfig, ToggleSwitchToggled, UiClick, button,
    color_of, grid, label, px, slider, tab_view, toggle_switch,
  },
};

use crate::showcase::ShowcaseRoot;

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

#[derive(Component)]
pub(crate) struct FpsText;

/// demo UI 已 spawn 的根标记：demo_ui_setup 的存在性守卫（查到即跳过），
/// 替代 Local<bool> done 平行状态——产物本身就是"是否已跑过"的真源
#[derive(Component)]
pub(crate) struct DemoUiRoot;

#[derive(Component)]
pub(crate) struct CamInfoText;

/// 「右上角面板」开关标记（ToggleSwitchToggled 观察者以此过滤事件）
#[derive(Component)]
struct ShowcaseVisibilityToggle;

/// 「垂直同步」开关标记（观察者写主窗口 Window.present_mode → bevy_render 重配 swapchain）
#[derive(Component)]
struct VsyncToggle;

#[derive(Component)]
struct DdgiStageSlider;

#[derive(Component)]
struct DdgiStageValueLabel;

#[derive(Component)]
struct DdgiDebugModeBtn(u8);

#[derive(Component)]
struct DdgiGainSlider;

#[derive(Component)]
struct DdgiGainValueLabel;

#[derive(Component)]
struct DdgiProbeVizToggle;

#[derive(Component)]
struct DdgiProbeVizLodSlider;

#[derive(Component)]
struct DdgiProbeVizLodValueLabel;

const DDGI_PROBE_VIZ_LODS: [&str; 5] = ["All", "LOD 0", "LOD 1", "LOD 2", "LOD 3"];

const DDGI_DEBUG_MODES: [&str; 5] = ["Normal", "GI", "wsum", "Domain", "Probe"];

const DDGI_STAGES: [&str; 4] = ["Off", "Active", "Cast", "Full"];

/// fps → 3 位宽显示值（上限 999，防 4 位数抖动）
pub(crate) fn fps3(v: f32) -> u32 {
  (v.round() as u32).min(999)
}

/// 左上角 debug-view 面板：**固定宽度 360px、高度 auto**（随当前 Tab 页内容收缩）。
/// TabView（fit_content 自适应高度模式）两页：
/// - Stats：FPS 读数 / 相机信息 / VSync / UI Showcase
///
/// 视觉：root 提供外框 + HUD 卡面底色；每页一个 grid（去外框/gap，行分割线由
/// cell bottom border 承担，避免右侧叠成 2px）。
/// 定位由本函数设置（PositionType::Absolute + 左上 8px 锚定）。
pub(crate) fn spawn_debug_view(world: &mut World, ctx: &UiCtx) {
  let c = &ctx.theme.colors;
  let m = &ctx.theme.metrics;
  // 绝对定位根：定宽 + 高度 auto（TabView fit_content 随活动页收缩）；外框/底色本节点提供
  world
    .spawn((
      Name::new("debug-view"),
      DemoUiRoot,
      Node {
        position_type: PositionType::Absolute,
        left: px(m.spacing.sm),
        top: px(m.spacing.sm),
        width: px(DEBUG_VIEW_W),
        flex_direction: FlexDirection::Column,
        border: UiRect::all(px(m.border_width)),
        ..default()
      },
      BackgroundColor(color_of(&c.surface_card_hud)),
      BorderColor::all(color_of(&c.border)),
    ))
    .with_children(|root| {
      // ---- FPS 关键读数（顶部，所有 Tab 可见；主文本色 text_primary） ----
      let fps_cell = root
        .spawn((
          Name::new("debug-fps"),
          Node {
            padding: UiRect::all(px(ctx.theme.metrics.spacing.sm)),
            border: UiRect::bottom(px(m.border_width)),
            ..default()
          },
          BackgroundColor(color_of(&c.surface_card)),
          BorderColor {
            bottom: color_of(&c.border),
            ..BorderColor::DEFAULT
          },
        ))
        .id();
      root.world_mut().entity_mut(fps_cell).with_children(|cell| {
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
      let tv = tab_view(
        ctx,
        root,
        TabConfig {
          tabs: vec!["Stats".into(), "DDGI".into()],
          active: 0,
          fit_content: true,
          ..default()
        },
      );
      // ============ Tab 0：Stats ============
      root
        .world_mut()
        .entity_mut(tv.contents[0])
        .with_children(|page| {
          let g = tab_page_grid(ctx, page);
          page.world_mut().entity_mut(g).with_children(|g| {
            // ---- 相机当前位置/角度信息（与 FPS 同档同色；两行行间距 = padding sm）----
            let c2 = tab_cell(ctx, g);
            g.world_mut().entity_mut(c2).with_children(|cell| {
              let e = label(
                ctx,
                cell,
                LabelConfig {
                  text: "CAMERA (---,---,---)\nTARGET (---,---,---)".into(),
                  ..default()
                },
              );
              cell.world_mut().entity_mut(*e).insert((
                CamInfoText,
                TextColor(color_of(&c.text_primary)),
                bevy::text::LineHeight::Px(
                  ctx.theme.metrics.font_size.md + ctx.theme.metrics.spacing.sm,
                ),
              ));
            });
            // ---- 垂直同步开关（默认开 = Fifo；观察者写 Window.present_mode，
            //      bevy_render 检测变化自动重配 swapchain；关 = AutoNoVsync 不封顶） ----
            let c5 = tab_cell(ctx, g);
            g.world_mut().entity_mut(c5).with_children(|cell| {
              let t = toggle_switch(
                ctx,
                cell,
                ToggleSwitchConfig {
                  text: Some("VSync".into()),
                  checked: true, // 默认开，与启动 present_mode=Fifo 同步
                  ..default()
                },
              );
              cell.world_mut().entity_mut(*t).insert(VsyncToggle);
            });
            // ---- 「右上角面板」显隐开关（默认隐藏；观察者写 ShowcaseRoot Visibility）----
            let c4 = tab_cell(ctx, g);
            g.world_mut().entity_mut(c4).with_children(|cell| {
              let t = toggle_switch(
                ctx,
                cell,
                ToggleSwitchConfig {
                  text: Some("UI Showcase".into()),
                  checked: false, // 默认隐藏 showcase，与 spawn_showcase 初始 Hidden 同步
                  ..default()
                },
              );
              cell
                .world_mut()
                .entity_mut(*t)
                .insert(ShowcaseVisibilityToggle);
            });
          });
        });
      root
        .world_mut()
        .entity_mut(tv.contents[1])
        .with_children(|page| {
          let g = tab_page_grid(ctx, page);
          page.world_mut().entity_mut(g).with_children(|g| {
            let cell = tab_cell(ctx, g);
            g.world_mut()
              .entity_mut(cell)
              .insert(Node {
                flex_direction: FlexDirection::Column,
                row_gap: px(ctx.theme.metrics.spacing.sm),
                padding: UiRect::all(px(ctx.theme.metrics.spacing.sm)),
                ..default()
              })
              .with_children(|cell| {
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
                        ..default()
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
                    row.world_mut().entity_mut(*vl).insert(DdgiStageValueLabel);
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
                          ..default()
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
                        ..default()
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
                    row.world_mut().entity_mut(*vl).insert(DdgiGainValueLabel);
                  });
                let t = toggle_switch(
                  &ctx,
                  cell,
                  ToggleSwitchConfig {
                    text: Some("Probe Viz".into()),
                    checked: false,
                    ..default()
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
                        ..default()
                      },
                    );
                    row.world_mut().entity_mut(*s).insert(DdgiProbeVizLodSlider);
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
      info!(
        "VSync {} → present_mode {:?}",
        if ev.checked { "on" } else { "off" },
        win.present_mode
      );
    },
  );

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
      info!(
        "DDGI stage → {} ({})",
        ddgi.0,
        DDGI_STAGES.get(v as usize).unwrap_or(&"?")
      );
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

/// Tab 页内网格：1 列 auto 行高。**去外框 + 去 gap 底色**——面板 root 已提供外框，
/// 行分割线由每个 cell 的 bottom border 承担（避免 grid BackgroundColor 在右侧溢出
/// 叠成 2px 边框）。
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
    n.row_gap = px(0.0);
    n.column_gap = px(0.0);
  }
  // 背景与 cell 一致（surface_card），行分割线改由 cell bottom border 画
  page
    .world_mut()
    .entity_mut(*g)
    .insert(BackgroundColor(color_of(&ctx.theme.colors.surface_card)));
  *g
}

/// Tab 页内 cell：surface_card 背景 + 1px 底部分割线（与 root 外框同色）。
/// 最后一行的底边线落在 root 内边框上方，不与外框叠加。
fn tab_cell(ctx: &UiCtx, parent: &mut ChildSpawner) -> Entity {
  let c = &ctx.theme.colors;
  let m = &ctx.theme.metrics;
  parent
    .spawn((
      Name::new("ui-grid-cell"),
      Node {
        padding: UiRect::all(px(m.spacing.sm)),
        border: UiRect::bottom(px(m.border_width)),
        ..default()
      },
      BackgroundColor(color_of(&c.surface_card)),
      BorderColor {
        bottom: color_of(&c.border),
        ..BorderColor::DEFAULT
      },
    ))
    .id()
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

/// 每 0.25s 刷新一次左上角 FPS 行（CUR/AVG/MIN/MAX）+ 相机位置行。
///
/// 逐帧 delta 压入 5s 滚动窗口；GPU/CPU 逐段剖析已交由 Tracy + wgpu-profiler
/// （`--features profile`，Tracy GUI 时间线），这里只留用户直视的 FPS 读数。
pub(crate) fn fps_line_feed(
  time: Res<Time>,
  orbit: Res<OrbitCamera>,
  mut q: ParamSet<(
    Query<&mut Text, With<FpsText>>,
    Query<&mut Text, With<CamInfoText>>,
  )>,
  mut window: Local<VecDeque<f32>>, // 逐帧 delta，按时间裁剪到 5s
  mut acc: Local<f32>,
  mut frames: Local<u32>,
) {
  let dt = time.delta_secs();
  window.push_back(dt);
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
  *acc = 0.0;
  *frames = 0;
  if let Ok(mut t) = q.p0().single_mut()
    && t.0 != txt
  {
    t.0 = txt;
  }
  // 相机信息：CAMERA / TARGET 两行
  let eye = orbit.eye();
  let tgt = orbit.target;
  let cam_txt = format!(
    "CAMERA: ({:>7.1}, {:>7.1}, {:>7.1})\nTARGET: ({:>7.1}, {:>7.1}, {:>7.1})",
    eye.x, eye.y, eye.z, tgt.x, tgt.y, tgt.z
  );
  if let Some(mut t) = q.p1().iter_mut().next()
    && t.0 != cam_txt
  {
    t.0 = cam_txt;
  }
}
