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

use crate::edit::{BrushShape, EDIT_MATERIALS, EDIT_SIZE_MAX, EDIT_SIZE_MIN, EditSettings};
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
struct DdgiProbeVizToggle;

#[derive(Component)]
struct DdgiProbeVizLodSlider;

#[derive(Component)]
struct DdgiProbeVizLodValueLabel;

/// 「Fly mode」开关标记（观察者写 [`crate::camera::CameraMode`]）
#[derive(Component)]
struct CameraModeToggle;

#[derive(Component)]
struct CameraSpeedSlider;

#[derive(Component)]
struct CameraSpeedValueLabel;

/// 笔触形状按钮（互斥；值 = 该按钮代表的形状）
#[derive(Component)]
struct EditShapeBtn(BrushShape);

#[derive(Component)]
struct EditSizeSlider;

#[derive(Component)]
struct EditSizeValueLabel;

#[derive(Component)]
struct EditMaterialSlider;

#[derive(Component)]
struct EditMaterialValueLabel;

/// 材质预览色块（只显示，不参与交互 —— button_state_system 会覆盖 Button 的背景色，
/// 所以这里用裸 Node + BackgroundColor）
#[derive(Component)]
struct EditMaterialSwatch;

const DDGI_PROBE_VIZ_LODS: [&str; 5] = ["All", "LOD 0", "LOD 1", "LOD 2", "LOD 3"];

const DDGI_DEBUG_MODES: [&str; 5] = ["Normal", "GI", "wsum", "Domain", "Probe"];

const DDGI_STAGES: [&str; 4] = ["Off", "Active", "Cast", "Full"];

/// 速度显示文本（voxel/s；1 voxel = 2cm）
fn speed_text(v: f32) -> String {
  format!("{} v/s", v.round() as i32)
}

/// 笔触跨度文本：size → (2N-1)³ 的边长（纯 ASCII，避免字体缺字形成方框）
fn brush_span_text(size: u32) -> String {
  format!("{size} vx (span {})", 2 * size.max(1) - 1)
}

/// 材质预览色
fn material_color(i: usize) -> Color {
  let c = EDIT_MATERIALS[i.min(EDIT_MATERIALS.len() - 1)].color;
  Color::srgb_u8(c[0], c[1], c[2])
}

/// 材质显示文本（名称 + 自发光）
fn material_text(i: usize) -> String {
  let m = &EDIT_MATERIALS[i.min(EDIT_MATERIALS.len() - 1)];
  format!("{} em{}", m.name, m.emissive)
}

/// fps → 3 位宽显示值（上限 999，防 4 位数抖动）
pub(crate) fn fps3(v: f32) -> u32 {
  (v.round() as u32).min(999)
}

/// 左上角 debug-view 面板：**固定宽度 360px、高度 auto**（随当前 Tab 页内容收缩）。
/// TabView（fit_content 自适应高度模式）三页：
/// - Stats：FPS 读数 / 相机信息 / VSync / UI Showcase
/// - DDGI：阶段档 / 调试模式 / Gain / Probe Viz / LOD
/// - Camera：轨道↔幽灵模式开关 / 飞行速度 / 操作说明
///
/// **所有控件的初始状态都从对应 Resource 读取**（UI 只是资源的视图）—— 缺省值只在
/// `DdgiStage::from_env` / `DdgiDebugSettings::default` / `camera` 里写一次，
/// 避免"资源默认变了、UI 还写着旧值"这类漂移。
///
/// 视觉：root 提供外框 + HUD 卡面底色；每页一个 grid（去外框/gap，行分割线由
/// cell bottom border 承担，避免右侧叠成 2px）。
/// 定位由本函数设置（PositionType::Absolute + 左上 8px 锚定）。
pub(crate) fn spawn_debug_view(world: &mut World, ctx: &UiCtx) {
  // ---- 初始状态：全部从资源读（见函数注释）----
  let ddgi_stage = world
    .get_resource::<gate_render::ddgi::DdgiStage>()
    .map_or(0, |s| s.0.min(DDGI_STAGES.len() as u8 - 1));
  let ddgi_dbg = world
    .get_resource::<gate_render::ddgi::DdgiDebugSettings>()
    .copied()
    .unwrap_or_default();
  let cam_fly = world
    .get_resource::<crate::camera::CameraMode>()
    .is_some_and(|m| *m == crate::camera::CameraMode::Fly);
  let fly_speed = world
    .get_resource::<crate::camera::FlyCamera>()
    .map_or(crate::camera::FLY_SPEED_DEFAULT, |f| f.speed)
    .clamp(crate::camera::FLY_SPEED_MIN, crate::camera::FLY_SPEED_MAX);
  let edit = world.get_resource::<EditSettings>().copied().unwrap_or_default();
  let edit_mat = edit.material.min(EDIT_MATERIALS.len() - 1);
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
          tabs: vec![
            "Stats".into(),
            "DDGI".into(),
            "Camera".into(),
            "Edit".into(),
          ],
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
      strip_last_cell_bottom(root.world_mut(), tv.contents[0]);
      root
        .world_mut()
        .entity_mut(tv.contents[1])
        .with_children(|page| {
          let g = tab_page_grid(ctx, page);
          page.world_mut().entity_mut(g).with_children(|g| {
            // -- DDGI 阶段滑杆行 --
            let cell = tab_cell(ctx, g);
            g.world_mut().entity_mut(cell).with_children(|cell| {
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
                    ctx,
                    row,
                    LabelConfig {
                      text: "DDGI".into(),
                      style: LabelStyle::Muted,
                      ..default()
                    },
                  );
                  let s = slider(
                    ctx,
                    row,
                    SliderConfig {
                      min: 0.0,
                      max: 3.0,
                      value: f32::from(ddgi_stage),
                      step: Some(1.0),
                      ..default()
                    },
                  );
                  row.world_mut().entity_mut(*s).insert(DdgiStageSlider);
                  let vl = label(
                    ctx,
                    row,
                    LabelConfig {
                      text: format!(
                        "{} {}",
                        ddgi_stage,
                        DDGI_STAGES[(ddgi_stage as usize).min(DDGI_STAGES.len() - 1)]
                      ),
                      style: LabelStyle::Muted,
                      ..default()
                    },
                  );
                  row.world_mut().entity_mut(*vl).insert(DdgiStageValueLabel);
                });
            });
            // -- 调试模式按钮行（5 按钮互斥单选；选中项 = DdgiDebugSettings.mode）--
            let cell = tab_cell(ctx, g);
            g.world_mut().entity_mut(cell).with_children(|cell| {
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
                  let active = ddgi_dbg.mode.round() as u8;
                  for (i, name) in DDGI_DEBUG_MODES.iter().enumerate() {
                    let variant = if i as u8 == active {
                      ButtonVariant::Primary
                    } else {
                      ButtonVariant::Ghost
                    };
                    let b = button(
                      ctx,
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
            });
            // -- Probe Viz 开关 --
            let cell = tab_cell(ctx, g);
            g.world_mut().entity_mut(cell).with_children(|cell| {
              let t = toggle_switch(
                ctx,
                cell,
                ToggleSwitchConfig {
                  text: Some("Probe Viz".into()),
                  checked: ddgi_dbg.probe_viz,
                  ..default()
                },
              );
              cell.world_mut().entity_mut(*t).insert(DdgiProbeVizToggle);
            });
            // -- Probe Viz 层级滑杆行（0..=4 步进 1：All / LOD0~3）--
            let cell = tab_cell(ctx, g);
            g.world_mut().entity_mut(cell).with_children(|cell| {
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
                    ctx,
                    row,
                    LabelConfig {
                      text: "LOD".into(),
                      style: LabelStyle::Muted,
                      ..default()
                    },
                  );
                  let s = slider(
                    ctx,
                    row,
                    SliderConfig {
                      min: 0.0,
                      max: 4.0,
                      value: ddgi_dbg.probe_viz_lod.clamp(0.0, 4.0),
                      step: Some(1.0),
                      ..default()
                    },
                  );
                  row.world_mut().entity_mut(*s).insert(DdgiProbeVizLodSlider);
                  let vl = label(
                    ctx,
                    row,
                    LabelConfig {
                      text: DDGI_PROBE_VIZ_LODS
                        [(ddgi_dbg.probe_viz_lod.round() as usize).min(DDGI_PROBE_VIZ_LODS.len() - 1)]
                      .into(),
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
      strip_last_cell_bottom(root.world_mut(), tv.contents[1]);
      // ============ Tab 2：Camera（轨道 ↔ 幽灵模式 + 飞行速度）============
      root
        .world_mut()
        .entity_mut(tv.contents[2])
        .with_children(|page| {
          let g = tab_page_grid(ctx, page);
          page.world_mut().entity_mut(g).with_children(|g| {
            // 相机模式开关：关 = 轨道（P2.6 原行为），开 = 幽灵飞行。**唯一切换入口**。
            let cell = tab_cell(ctx, g);
            g.world_mut().entity_mut(cell).with_children(|cell| {
              let t = toggle_switch(
                ctx,
                cell,
                ToggleSwitchConfig {
                  text: Some("Fly mode".into()),
                  checked: cam_fly,
                  ..default()
                },
              );
              cell.world_mut().entity_mut(*t).insert(CameraModeToggle);
            });
            // 飞行速度（voxel/s；只在 Fly 模式下生效，1 voxel = 2cm）
            let cell = tab_cell(ctx, g);
            g.world_mut().entity_mut(cell).with_children(|cell| {
              cell
                .spawn((
                  Name::new("cam-speed-row"),
                  Node {
                    flex_direction: FlexDirection::Row,
                    column_gap: px(ctx.theme.metrics.spacing.md),
                    align_items: AlignItems::Center,
                    ..default()
                  },
                ))
                .with_children(|row| {
                  label(
                    ctx,
                    row,
                    LabelConfig {
                      text: "Speed".into(),
                      style: LabelStyle::Muted,
                      ..default()
                    },
                  );
                  let s = slider(
                    ctx,
                    row,
                    SliderConfig {
                      min: crate::camera::FLY_SPEED_MIN,
                      max: crate::camera::FLY_SPEED_MAX,
                      value: fly_speed,
                      step: Some(crate::camera::FLY_SPEED_STEP),
                      ..default()
                    },
                  );
                  row.world_mut().entity_mut(*s).insert(CameraSpeedSlider);
                  let vl = label(
                    ctx,
                    row,
                    LabelConfig {
                      text: speed_text(fly_speed),
                      style: LabelStyle::Muted,
                      ..default()
                    },
                  );
                  row.world_mut().entity_mut(*vl).insert(CameraSpeedValueLabel);
                });
            });
          });
        });
      strip_last_cell_bottom(root.world_mut(), tv.contents[2]);
      // ============ Tab 3：Edit（体素编辑笔触：形状 / 大小 / 材质）============
      // 生效范围：**仅幽灵模式**（轨道模式左键仍是 recenter，见 camera::left_click_pick_recenter）。
      // 左键放置（只填空气）/ 右键擦除（挖空）；笔触以命中格为中心按形状展开。
      root
        .world_mut()
        .entity_mut(tv.contents[3])
        .with_children(|page| {
          let g = tab_page_grid(ctx, page);
          page.world_mut().entity_mut(g).with_children(|g| {
            // -- 形状：Sphere / Cube 互斥按钮 --
            let cell = tab_cell(ctx, g);
            g.world_mut().entity_mut(cell).with_children(|cell| {
              cell
                .spawn((
                  Name::new("edit-shape-row"),
                  Node {
                    flex_direction: FlexDirection::Row,
                    column_gap: px(ctx.theme.metrics.spacing.sm),
                    align_items: AlignItems::Center,
                    ..default()
                  },
                ))
                .with_children(|row| {
                  label(
                    ctx,
                    row,
                    LabelConfig {
                      text: "Shape".into(),
                      style: LabelStyle::Muted,
                      ..default()
                    },
                  );
                  for (shape, name) in [
                    (BrushShape::Sphere, "Sphere"),
                    (BrushShape::Cube, "Cube"),
                  ] {
                    let variant = if shape == edit.shape {
                      ButtonVariant::Primary
                    } else {
                      ButtonVariant::Ghost
                    };
                    let b = button(
                      ctx,
                      row,
                      ButtonConfig {
                        text: name.into(),
                        variant,
                        ..default()
                      },
                    );
                    row.world_mut().entity_mut(*b).insert(EditShapeBtn(shape));
                  }
                });
            });
            // -- 笔触大小（voxel；1 = 单格，N = (2N-1) 跨度）--
            let cell = tab_cell(ctx, g);
            g.world_mut().entity_mut(cell).with_children(|cell| {
              cell
                .spawn((
                  Name::new("edit-size-row"),
                  Node {
                    flex_direction: FlexDirection::Row,
                    column_gap: px(ctx.theme.metrics.spacing.md),
                    align_items: AlignItems::Center,
                    ..default()
                  },
                ))
                .with_children(|row| {
                  label(
                    ctx,
                    row,
                    LabelConfig {
                      text: "Size".into(),
                      style: LabelStyle::Muted,
                      ..default()
                    },
                  );
                  let s = slider(
                    ctx,
                    row,
                    SliderConfig {
                      min: EDIT_SIZE_MIN as f32,
                      max: EDIT_SIZE_MAX as f32,
                      value: edit.size as f32,
                      step: Some(1.0),
                      ..default()
                    },
                  );
                  row.world_mut().entity_mut(*s).insert(EditSizeSlider);
                  let vl = label(
                    ctx,
                    row,
                    LabelConfig {
                      text: brush_span_text(edit.size),
                      style: LabelStyle::Muted,
                      ..default()
                    },
                  );
                  row.world_mut().entity_mut(*vl).insert(EditSizeValueLabel);
                });
            });
            // -- 材质（预设表下标；右侧名称 + 自发光 + 预览色块）--
            let cell = tab_cell(ctx, g);
            g.world_mut().entity_mut(cell).with_children(|cell| {
              cell
                .spawn((
                  Name::new("edit-mat-row"),
                  Node {
                    flex_direction: FlexDirection::Row,
                    column_gap: px(ctx.theme.metrics.spacing.md),
                    align_items: AlignItems::Center,
                    ..default()
                  },
                ))
                .with_children(|row| {
                  label(
                    ctx,
                    row,
                    LabelConfig {
                      text: "Mat".into(),
                      style: LabelStyle::Muted,
                      ..default()
                    },
                  );
                  let s = slider(
                    ctx,
                    row,
                    SliderConfig {
                      min: 0.0,
                      max: (EDIT_MATERIALS.len() - 1) as f32,
                      value: edit_mat as f32,
                      step: Some(1.0),
                      ..default()
                    },
                  );
                  row.world_mut().entity_mut(*s).insert(EditMaterialSlider);
                  let vl = label(
                    ctx,
                    row,
                    LabelConfig {
                      text: material_text(edit_mat),
                      style: LabelStyle::Muted,
                      ..default()
                    },
                  );
                  row.world_mut().entity_mut(*vl).insert(EditMaterialValueLabel);
                  row.spawn((
                    Name::new("edit-mat-swatch"),
                    EditMaterialSwatch,
                    Node {
                      width: px(44.0),
                      height: px(m.font_size.md),
                      border: UiRect::all(px(m.border_width)),
                      ..default()
                    },
                    BackgroundColor(material_color(edit_mat)),
                    BorderColor::all(color_of(&c.border)),
                  ));
                });
            });
          });
        });
      strip_last_cell_bottom(root.world_mut(), tv.contents[3]);
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

  // 「Fly mode」开关 → 写 CameraMode（位置对齐由 camera::sync_camera_mode_switch 负责）
  world.add_observer(
    |ev: On<ToggleSwitchToggled>,
     q_toggle: Query<(), With<CameraModeToggle>>,
     mut mode: ResMut<crate::camera::CameraMode>| {
      if q_toggle.get(ev.entity).is_err() {
        return;
      }
      *mode = if ev.checked {
        crate::camera::CameraMode::Fly
      } else {
        crate::camera::CameraMode::Orbit
      };
    },
  );

  // 飞行速度滑杆 → 写 FlyCamera.speed + 刷新右侧数值标签
  world.add_observer(
    |ev: On<SliderValueChanged>,
     q_slider: Query<(), With<CameraSpeedSlider>>,
     mut q_label: Query<&mut Text, With<CameraSpeedValueLabel>>,
     mut fly: ResMut<crate::camera::FlyCamera>| {
      if q_slider.get(ev.entity).is_err() {
        return;
      }
      fly.speed =
        ev.value
          .clamp(crate::camera::FLY_SPEED_MIN, crate::camera::FLY_SPEED_MAX);
      if let Ok(mut t) = q_label.single_mut() {
        t.0 = speed_text(fly.speed);
      }
      info!("fly speed → {} v/s", fly.speed.round() as i32);
    },
  );

  // Edit：笔触形状按钮 → 写 EditSettings.shape + 互斥切换变体（选中=Primary）
  world.add_observer(
    |ev: On<UiClick>,
     q_btn: Query<&EditShapeBtn>,
     mut q_all: Query<(Entity, &mut ButtonVariant, &EditShapeBtn)>,
     mut settings: ResMut<EditSettings>| {
      let Ok(btn) = q_btn.get(ev.entity) else {
        return;
      };
      settings.shape = btn.0;
      for (_e, mut var, b) in &mut q_all {
        *var = if b.0 == btn.0 {
          ButtonVariant::Primary
        } else {
          ButtonVariant::Ghost
        };
      }
      info!("edit brush shape → {:?}", btn.0);
    },
  );

  // Edit：笔触大小滑杆 → 写 EditSettings.size + 刷新跨度标签
  world.add_observer(
    |ev: On<SliderValueChanged>,
     q_slider: Query<(), With<EditSizeSlider>>,
     mut q_label: Query<&mut Text, With<EditSizeValueLabel>>,
     mut settings: ResMut<EditSettings>| {
      if q_slider.get(ev.entity).is_err() {
        return;
      }
      let v = (ev.value.round() as u32).clamp(EDIT_SIZE_MIN, EDIT_SIZE_MAX);
      settings.size = v;
      if let Ok(mut t) = q_label.single_mut() {
        t.0 = brush_span_text(v);
      }
      info!("edit brush size → {v} vx");
    },
  );

  // Edit：材质滑杆 → 写 EditSettings.material + 刷新名称标签与预览色块
  world.add_observer(
    |ev: On<SliderValueChanged>,
     q_slider: Query<(), With<EditMaterialSlider>>,
     mut q_label: Query<&mut Text, With<EditMaterialValueLabel>>,
     mut q_swatch: Query<&mut BackgroundColor, With<EditMaterialSwatch>>,
     mut settings: ResMut<EditSettings>| {
      if q_slider.get(ev.entity).is_err() {
        return;
      }
      let i = (ev.value.round() as usize).min(EDIT_MATERIALS.len() - 1);
      settings.material = i;
      if let Ok(mut t) = q_label.single_mut() {
        t.0 = material_text(i);
      }
      if let Ok(mut bg) = q_swatch.single_mut() {
        bg.0 = material_color(i);
      }
      info!("edit material → {}", EDIT_MATERIALS[i].name);
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

/// 去掉某 Tab 页网格**最后一个 cell** 的 bottom 分割线。
///
/// root 外框在面板底边已画了 1px；末行 cell 若再画一条，两条同色线相邻叠加 = 视觉 2px。
/// 内部行分隔线仍由各 cell 的 bottom border 承担（见 [`tab_cell`]）。
/// page 的孩子只有一个（[`tab_page_grid`] 产出的 grid），grid 的孩子即各 cell。
fn strip_last_cell_bottom(world: &mut World, page: Entity) {
  let Some(grid) = world.get::<Children>(page).and_then(|c| c.first().copied()) else {
    return;
  };
  let Some(last) = world.get::<Children>(grid).and_then(|c| c.last().copied()) else {
    return;
  };
  if let Some(mut node) = world.get_mut::<Node>(last) {
    node.border.bottom = px(0.0);
  }
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
#[allow(clippy::type_complexity, clippy::too_many_arguments)] // Bevy system：ParamSet/资源逐一注入
pub(crate) fn fps_line_feed(
  time: Res<Time>,
  orbit: Res<OrbitCamera>,
  mode: Res<crate::camera::CameraMode>,
  fly: Res<crate::camera::FlyCamera>,
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
  // 相机信息两行：轨道模式 = 眼位 + 注视点；幽灵模式 = 飞行眼位 + 朝向角。
  // （幽灵模式下没有"轨道目标"，继续显示它只会给出误导读数。）
  let (eye, line2) = match *mode {
    crate::camera::CameraMode::Orbit => {
      let t = orbit.target;
      (
        orbit.eye(),
        format!("TARGET: ({:>7.1}, {:>7.1}, {:>7.1})", t.x, t.y, t.z),
      )
    }
    crate::camera::CameraMode::Fly => (
      fly.pos,
      format!(
        "DIR(deg): yaw {:>6.1}, pitch {:>5.1}",
        orbit.yaw.to_degrees(),
        orbit.pitch.to_degrees()
      ),
    ),
  };
  let cam_txt = format!(
    "CAMERA: ({:>7.1}, {:>7.1}, {:>7.1})\n{}",
    eye.x, eye.y, eye.z, line2
  );
  if let Some(mut t) = q.p1().iter_mut().next()
    && t.0 != cam_txt
  {
    t.0 = cam_txt;
  }
}
