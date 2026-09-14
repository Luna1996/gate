//! slider：SliderValue + clamp/step + 拖动 + 强调填充视觉（docs/ui-dark-theme.md §5.4）。
//!
//! 拖动逻辑复用 bevy_ui `RelativeCursorPosition`（ui_focus_system 自动更新）；
//! 归一化坐标以节点中心为原点（-0.5..0.5），映射到 0..1。
//! 视觉：4px 抬升表面轨道槽 + 强调填充段；16px 顶层表面滑块（拖拽放大到 18px）。

use std::ops::Deref;

use bevy::prelude::*;
use bevy::ui::{FocusPolicy, Interaction, RelativeCursorPosition};

use super::{UiCtx, UiDisabled, color_of, dim_color, px};

/// 滑杆根节点标记
#[derive(Component, Debug, Default)]
pub struct UiSlider;

/// 值域
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct SliderRange {
  pub min: f32,
  pub max: f32,
}

impl Default for SliderRange {
  fn default() -> Self {
    Self { min: 0.0, max: 1.0 }
  }
}

/// 步长（None = 连续）
#[derive(Component, Clone, Copy, Debug, PartialEq, Default)]
pub struct SliderStep(pub Option<f32>);

/// 当前值（clamp/step 后落在 [min, max]）
#[derive(Component, Clone, Copy, Debug, PartialEq, Default)]
pub struct SliderValue(pub f32);

/// 填充条子实体标记
#[derive(Component, Debug, Default)]
pub struct SliderFill;

/// 滑块子实体标记
#[derive(Component, Debug, Default)]
pub struct SliderThumb;

/// 滑杆句柄（Deref 到根实体 Entity）
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SliderHandle(pub Entity);

impl Deref for SliderHandle {
  type Target = Entity;
  fn deref(&self) -> &Entity {
    &self.0
  }
}

impl From<SliderHandle> for Entity {
  fn from(h: SliderHandle) -> Entity {
    h.0
  }
}

/// 滑杆配置（全部字段进 Config；Default = [0,1] 连续、初值 0、不禁用）
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SliderConfig {
  pub min: f32,
  pub max: f32,
  pub value: f32,
  pub step: Option<f32>,
  /// true = 禁用态：不响应拖动、配色暗一档（当前值仍显示）
  pub disabled: bool,
}

impl Default for SliderConfig {
  fn default() -> Self {
    Self {
      min: 0.0,
      max: 1.0,
      value: 0.0,
      step: None,
      disabled: false,
    }
  }
}

/// 滑杆值变化事件（仅用户拖动产生的变化触发；EntityEvent，target = 滑杆根实体）。
/// 组件 [`SliderValue`] 仍是真源，事件只是通知，主动读值可 Query。
#[derive(EntityEvent, Clone, Copy, Debug, PartialEq)]
pub struct SliderValueChanged {
  pub entity: Entity,
  pub value: f32,
}

const THUMB_SIZE: f32 = 16.0;
/// 拖拽中滑块放大尺寸
const THUMB_SIZE_DRAG: f32 = 18.0;
/// 根节点水平内边距 = 半 thumb 宽：thumb 行程首尾完全在根命中矩形内
const THUMB_INSET: f32 = THUMB_SIZE / 2.0;
const TRACK_HEIGHT: f32 = 4.0;

/// clamp + step 归一（纯函数）
pub fn clamp_step(value: f32, min: f32, max: f32, step: Option<f32>) -> f32 {
  let v = value.clamp(min, max);
  match step {
    Some(s) if s > 0.0 => min + ((v - min) / s).round() * s,
    _ => v,
  }
  .clamp(min, max)
}

/// 主题滑杆（横向；宽度由父容器 flex 决定，最小 120px）
///
/// 几何：根节点水平内边距 = 半 thumb 宽（[`THUMB_INSET`]），track 与 thumb 都在
/// 内容盒内——thumb 行程首尾各止于根边缘内 8px，**任何档位 thumb 都完整落在根
/// 命中区内**（旧几何 thumb 为根直接子、`left:0%/100% + 负 margin` 居中，首尾
/// 半幅溢出根命中矩形，端点滑块难点中）。thumb 是 track 子节点（与 fill 同一
/// 绝对定位包含块 → 位置天然对齐），`top:50%` 相对 4px track 高 + 负 margin
/// 半高实现垂直居中。
pub fn slider(ctx: &UiCtx, parent: &mut ChildSpawner, config: SliderConfig) -> SliderHandle {
  let c = &ctx.theme.colors;
  let m = &ctx.theme.metrics;
  let v = clamp_step(config.value, config.min, config.max, config.step);
  let norm = normalize(v, config.min, config.max);
  let track_bg = color_of(&c.surface_overlay);
  let fill_bg = color_of(&c.accent_fill_hover);
  let thumb_bg = color_of(&c.surface_top);
  let thumb_border = color_of(&c.border_strong);
  let (track_bg, fill_bg, thumb_bg, thumb_border) = if config.disabled {
    (
      dim_color(track_bg),
      dim_color(fill_bg),
      dim_color(thumb_bg),
      dim_color(thumb_border),
    )
  } else {
    (track_bg, fill_bg, thumb_bg, thumb_border)
  };
  let mut ec = parent.spawn((
    Name::new("ui-slider"),
    UiSlider,
    Interaction::default(),
    RelativeCursorPosition::default(),
    SliderRange {
      min: config.min,
      max: config.max,
    },
    SliderStep(config.step),
    SliderValue(v),
    Node {
      min_width: px(120.0),
      height: px(24.0),
      align_items: AlignItems::Center,
      // 半 thumb 水平内缩：thumb 行程端点不超出根命中矩形
      padding: UiRect::horizontal(px(THUMB_INSET)),
      ..default()
    },
    FocusPolicy::Block,
  ));
  ec.with_children(|root| {
    // 轨道槽（抬升表面 +1 阶：surface_elevated → surface_overlay，增强与填充对比）。
    // flex_grow 撑满内容盒（根宽 - 2*THUMB_INSET）；fill 与 thumb 均为其绝对
    // 定位子节点，包含块 = track 边框盒
    root
      .spawn((
        Name::new("ui-slider-track"),
        Node {
          flex_grow: 1.0,
          height: px(TRACK_HEIGHT),
          ..default()
        },
        BackgroundColor(track_bg),
      ))
      .with_children(|track| {
        // 填充段（accent_fill → accent_fill_hover，提亮一阶并与 thumb 面区分；
        // 原 accent_fill == surface_top == thumb 面色，导致填充与滑块同色不可辨）
        track.spawn((
          Name::new("ui-slider-fill"),
          SliderFill,
          Node {
            position_type: PositionType::Absolute,
            left: Val::ZERO,
            top: Val::ZERO,
            bottom: Val::ZERO,
            width: Val::Percent(norm * 100.0),
            ..default()
          },
          BackgroundColor(fill_bg),
        ));
        // 滑块（顶层表面 + 提亮边框；方形，无圆角；绝对定位：left 按值百分比，
        // margin 左/上各负半尺寸把滑块中心钉在定位点；top:50% 相对 track 高 4px
        // = 2px，配合 -半高 margin 实现垂直居中于 track 中心线）
        track.spawn((
          Name::new("ui-slider-thumb"),
          SliderThumb,
          Node {
            position_type: PositionType::Absolute,
            left: Val::Percent(norm * 100.0),
            top: Val::Percent(50.0),
            margin: UiRect {
              left: px(-THUMB_SIZE / 2.0),
              top: px(-THUMB_SIZE / 2.0),
              ..default()
            },
            width: px(THUMB_SIZE),
            height: px(THUMB_SIZE),
            border: UiRect::all(px(m.border_width)),
            ..default()
          },
          BackgroundColor(thumb_bg),
          BorderColor::all(thumb_border),
        ));
      });
  });
  if config.disabled {
    ec.insert(UiDisabled);
  }
  SliderHandle(ec.id())
}

fn normalize(v: f32, min: f32, max: f32) -> f32 {
  if (max - min).abs() < f32::EPSILON {
    0.0
  } else {
    ((v - min) / (max - min)).clamp(0.0, 1.0)
  }
}

/// 拖动：按下时把光标位置映射为值（每帧重算）；值变化时触发
/// [`SliderValueChanged`]（仅用户拖动，程序写 SliderValue 不触发）。
///
/// 映射基准 = 根节点**内容盒**（track 行程区 = 根宽 - 2×[`THUMB_INSET`]）：
/// 光标在两侧内边距带内即吸附 min/max（端点档位命中区与中段一样宽，不再只有
/// 1px 边界）。有 step 时经 [`clamp_step`] 逐档吸附（离散档位 slider）。
#[allow(clippy::type_complexity)] // Bevy system：多组件查询签名固有
pub fn slider_drag_system(
  mut commands: Commands,
  mut q: Query<(
    Entity,
    &Interaction,
    &RelativeCursorPosition,
    &ComputedNode,
    &SliderRange,
    &SliderStep,
    &mut SliderValue,
    Has<UiDisabled>,
  )>,
) {
  for (e, inter, rcp, node, range, step, mut val, disabled) in &mut q {
    if disabled || *inter != Interaction::Pressed {
      continue;
    }
    let Some(n) = rcp.normalized else { continue };
    // normalized 以根节点中心为原点（-0.5..0.5）；ComputedNode.size 与
    // padding（BorderRect：min_inset.x=左、max_inset.x=右）同为物理 px，
    // 比值运算单位自洽（不需要 inverse_scale_factor）
    let root_w = node.size().x;
    let cursor_x = (n.x + 0.5) * root_w;
    let pad_l = node.padding.min_inset.x;
    let pad_r = node.padding.max_inset.x;
    let content_w = root_w - pad_l - pad_r;
    let norm = if content_w > 0.0 {
      ((cursor_x - pad_l) / content_w).clamp(0.0, 1.0)
    } else {
      0.0
    };
    let target = range.min + norm * (range.max - range.min);
    let v = clamp_step(target, range.min, range.max, step.0);
    if (val.0 - v).abs() > f32::EPSILON {
      val.0 = v;
      commands.trigger(SliderValueChanged {
        entity: e,
        value: v,
      });
    }
  }
}

/// 视觉：填充宽度 + 滑块位置跟随 SliderValue；拖拽中滑块放大到 18px
///
/// Disabled 态：滑块不放大（恒为 16px），配色由 spawn 时降亮。
///
/// 注意：fill 与 thumb 都是 track 的子节点（slider root 的孙节点），不能只遍历
/// root 的直接 Children——需要下钻 track 的 Children 分别命中 SliderFill 与
/// SliderThumb。thumb 的 `top:50% + margin-top:-半高` 垂直居中也在此同步（尺寸
/// 随拖拽放大）。
#[allow(clippy::type_complexity)] // Bevy system：多组件查询签名固有
pub fn slider_visual_system(
  mut q_root: Query<
    (
      &SliderValue,
      &SliderRange,
      &Interaction,
      &Children,
      Has<UiDisabled>,
    ),
    With<UiSlider>,
  >,
  mut fills: Query<&mut Node, (With<SliderFill>, Without<SliderThumb>)>,
  mut thumbs: Query<&mut Node, With<SliderThumb>>,
  q_track_children: Query<&Children, Without<UiSlider>>,
) {
  for (val, range, inter, children, disabled) in &mut q_root {
    let pct = normalize(val.0, range.min, range.max) * 100.0;
    let thumb_size = if !disabled && *inter == Interaction::Pressed {
      THUMB_SIZE_DRAG
    } else {
      THUMB_SIZE
    };
    for child in children.iter() {
      // fill/thumb 都在 track 内部（孙节点）：下钻 track 的 Children
      if let Ok(tc) = q_track_children.get(child) {
        for sub in tc.iter() {
          if let Ok(mut node) = fills.get_mut(sub) {
            node.width = Val::Percent(pct);
          }
          if let Ok(mut node) = thumbs.get_mut(sub) {
            node.left = Val::Percent(pct);
            // top:50% 相对 track 高（4px）= 2px，margin 负半高把中心钉在 track
            // 中心线；拖拽放大时尺寸与负 margin 同步
            node.top = Val::Percent(50.0);
            node.width = px(thumb_size);
            node.height = px(thumb_size);
            node.margin = UiRect {
              left: px(-thumb_size / 2.0),
              top: px(-thumb_size / 2.0),
              ..default()
            };
          }
        }
      }
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::theme::default_theme;

  #[test]
  fn clamp_step_math() {
    assert_eq!(clamp_step(0.5, 0.0, 1.0, None), 0.5);
    assert_eq!(clamp_step(-1.0, 0.0, 1.0, None), 0.0, "clamp low");
    assert_eq!(clamp_step(2.0, 0.0, 1.0, None), 1.0, "clamp high");
    assert_eq!(clamp_step(0.44, 0.0, 1.0, Some(0.25)), 0.5, "step round");
    assert_eq!(
      clamp_step(7.0, 0.0, 10.0, Some(3.0)),
      6.0,
      "step aligned to min"
    );
    assert_eq!(clamp_step(5.0, 5.0, 5.0, None), 5.0, "degenerate range");
  }

  #[test]
  fn slider_spawn_structure() {
    let theme = default_theme();
    let ctx = UiCtx::new(&theme, None);
    let mut app = App::new();
    let root = app.world_mut().spawn_empty().id();
    let mut child = None;
    app.world_mut().entity_mut(root).with_children(|p| {
      child = Some(slider(
        &ctx,
        p,
        SliderConfig {
          min: 0.0,
          max: 100.0,
          value: 25.0,
          step: Some(5.0),
          ..default()
        },
      ));
    });
    let h = child.expect("slider spawned");
    let e = *h;
    let w = app.world();
    assert_eq!(
      w.get::<SliderValue>(e).unwrap().0,
      25.0,
      "initial value clamp/step"
    );
    let root_children = w.get::<Children>(e).unwrap();
    assert_eq!(
      root_children.len(),
      1,
      "root only has track (thumb moved under track)"
    );
    let track = root_children[0];
    let track_children = w.get::<Children>(track).unwrap();
    assert_eq!(track_children.len(), 2, "track holds fill + thumb");
    let has_fill = track_children
      .iter()
      .any(|c| w.get::<SliderFill>(c).is_some());
    let has_thumb = track_children
      .iter()
      .any(|c| w.get::<SliderThumb>(c).is_some());
    assert!(has_fill && has_thumb, "fill + thumb both under track");
    // 根节点水平内边距 = 半 thumb（thumb 行程端点不溢出命中区）
    let pad = w.get::<Node>(e).unwrap().padding;
    assert_eq!(pad.left, px(THUMB_INSET));
    assert_eq!(pad.right, px(THUMB_INSET));
  }

  #[test]
  fn slider_drag_updates_value_and_emits_event() {
    use std::sync::{Arc, Mutex};

    let mut app = App::new();
    app.add_systems(Update, slider_drag_system);
    let events = Arc::new(Mutex::new(Vec::<(Entity, f32)>::new()));
    let sink = events.clone();
    app.add_observer(move |ev: On<SliderValueChanged>| {
      sink.lock().unwrap().push((ev.entity, ev.value));
    });
    // ComputedNode：根宽 100px、无水平内边距 → cursor_x = (0.25+0.5)*100 = 75
    let computed = ComputedNode {
      size: Vec2::new(100.0, 24.0),
      ..Default::default()
    };
    let e = app
      .world_mut()
      .spawn((
        UiSlider,
        Interaction::Pressed,
        RelativeCursorPosition {
          cursor_over: true,
          normalized: Some(Vec2::new(0.25, 0.0)),
        },
        computed,
        SliderRange::default(),
        SliderStep(None),
        SliderValue(0.0),
      ))
      .id();
    app.update();
    // x=0.25（中心原点）→ cursor_x=75 → 内容盒 norm=0.75
    assert!((app.world().get::<SliderValue>(e).unwrap().0 - 0.75).abs() < 1e-6);
    assert_eq!(
      *events.lock().unwrap(),
      vec![(e, 0.75)],
      "value change emits SliderValueChanged"
    );

    // 未按下不更新，也不发事件
    app
      .world_mut()
      .get_mut::<Interaction>(e)
      .unwrap()
      .set_if_neq(Interaction::Hovered);
    app
      .world_mut()
      .get_mut::<RelativeCursorPosition>(e)
      .unwrap()
      .normalized = Some(Vec2::new(-0.5, 0.0));
    app.update();
    assert!((app.world().get::<SliderValue>(e).unwrap().0 - 0.75).abs() < 1e-6);
    assert_eq!(events.lock().unwrap().len(), 1, "no event when not pressed");

    // 水平内边距带（THUMB_INSET 半 thumb）：根宽 100px + padding 8px →
    // 内容盒 [8,92]。按下时光标在左 8px 内边距带（根左边缘外旧几何的点击
    // 死区）→ 吸附 min=0；右带 → 吸附 max=1（端点命中区与中段等宽）
    {
      let mut cn = app.world_mut().get_mut::<ComputedNode>(e).unwrap();
      cn.padding.min_inset.x = 8.0;
      cn.padding.max_inset.x = 8.0;
    }
    app
      .world_mut()
      .get_mut::<Interaction>(e)
      .unwrap()
      .set_if_neq(Interaction::Pressed);
    app
      .world_mut()
      .get_mut::<RelativeCursorPosition>(e)
      .unwrap()
      .normalized = Some(Vec2::new(-0.48, 0.0)); // cursor_x = 2px < 8px padding
    app.update();
    assert_eq!(
      app.world().get::<SliderValue>(e).unwrap().0,
      0.0,
      "left inset band clamps to min"
    );
    app
      .world_mut()
      .get_mut::<RelativeCursorPosition>(e)
      .unwrap()
      .normalized = Some(Vec2::new(0.48, 0.0)); // cursor_x = 98px > 92px content end
    app.update();
    assert_eq!(
      app.world().get::<SliderValue>(e).unwrap().0,
      1.0,
      "right inset band clamps to max"
    );
  }

  #[test]
  fn slider_visual_follows_value() {
    let mut app = App::new();
    app.add_systems(Update, slider_visual_system);
    let e = app
      .world_mut()
      .spawn((
        UiSlider,
        SliderValue(0.5),
        SliderRange::default(),
        Interaction::default(),
        Children::default(),
      ))
      .id();
    // 真实结构：root → track → fill + thumb（thumb 与 fill 同在 track 下）
    let track = app.world_mut().spawn((Children::default(),)).id();
    let fill = app.world_mut().spawn((SliderFill, Node::default())).id();
    let thumb = app.world_mut().spawn((SliderThumb, Node::default())).id();
    app.world_mut().entity_mut(track).add_child(fill);
    app.world_mut().entity_mut(track).add_child(thumb);
    app.world_mut().entity_mut(e).add_child(track);
    app.update();
    let fill_node = app.world().get::<Node>(fill).unwrap();
    assert!(matches!(fill_node.width, Val::Percent(p) if (p - 50.0).abs() < 1e-5));
    let thumb_node = app.world().get::<Node>(thumb).unwrap();
    assert!(matches!(thumb_node.left, Val::Percent(p) if (p - 50.0).abs() < 1e-5));
  }
}
