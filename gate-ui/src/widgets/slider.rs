//! slider：SliderValue + clamp/step + 拖动 + accent 填充视觉。
//!
//! 拖动逻辑复用 bevy_ui `RelativeCursorPosition`（ui_focus_system 自动更新）；
//! 归一化坐标以节点中心为原点（-0.5..0.5），映射到 0..1。

use bevy::prelude::*;
use bevy::ui::{FocusPolicy, Interaction, RelativeCursorPosition};

use super::{UiCtx, color_of, px};

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

const THUMB_SIZE: f32 = 14.0;
const TRACK_HEIGHT: f32 = 6.0;

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
#[allow(clippy::too_many_arguments)]
pub fn slider(
  ctx: &UiCtx,
  parent: &mut ChildSpawner,
  min: f32,
  max: f32,
  value: f32,
  step: Option<f32>,
) -> Entity {
  let c = &ctx.theme.colors;
  let v = clamp_step(value, min, max, step);
  let norm = normalize(v, min, max);
  parent
    .spawn((
      Name::new("ui-slider"),
      UiSlider,
      Interaction::default(),
      RelativeCursorPosition::default(),
      SliderRange { min, max },
      SliderStep(step),
      SliderValue(v),
      Node {
        min_width: px(120.0),
        height: px(24.0),
        align_items: AlignItems::Center,
        ..default()
      },
      FocusPolicy::Block,
    ))
    .with_children(|root| {
      // 轨道
      root
        .spawn((
          Name::new("ui-slider-track"),
          Node {
            flex_grow: 1.0,
            height: px(TRACK_HEIGHT),
            border_radius: BorderRadius::all(px(TRACK_HEIGHT / 2.0)),
            ..default()
          },
          BackgroundColor(color_of(&c.panel_border)),
        ))
        .with_children(|track| {
          // 填充
          track.spawn((
            Name::new("ui-slider-fill"),
            SliderFill,
            Node {
              position_type: PositionType::Absolute,
              left: Val::ZERO,
              top: Val::ZERO,
              bottom: Val::ZERO,
              width: Val::Percent(norm * 100.0),
              border_radius: BorderRadius::all(px(TRACK_HEIGHT / 2.0)),
              ..default()
            },
            BackgroundColor(color_of(&c.accent)),
          ));
        });
      // 滑块（绝对定位，margin 负半宽居中）
      root.spawn((
        Name::new("ui-slider-thumb"),
        SliderThumb,
        Node {
          position_type: PositionType::Absolute,
          left: Val::Percent(norm * 100.0),
          margin: UiRect::left(Val::Px(-THUMB_SIZE / 2.0)),
          width: px(THUMB_SIZE),
          height: px(THUMB_SIZE),
          border_radius: BorderRadius::MAX,
          ..default()
        },
        BackgroundColor(color_of(&c.text)),
        BorderColor::all(color_of(&c.accent)),
      ));
    })
    .id()
}

fn normalize(v: f32, min: f32, max: f32) -> f32 {
  if (max - min).abs() < f32::EPSILON {
    0.0
  } else {
    ((v - min) / (max - min)).clamp(0.0, 1.0)
  }
}

/// 拖动：按下时把光标归一化 x（-0.5..0.5 中心原点）映射为值（每帧重算）
pub fn slider_drag_system(
  mut q: Query<(
    &Interaction,
    &RelativeCursorPosition,
    &SliderRange,
    &SliderStep,
    &mut SliderValue,
  )>,
) {
  for (inter, rcp, range, step, mut val) in &mut q {
    if *inter != Interaction::Pressed {
      continue;
    }
    let Some(n) = rcp.normalized else { continue };
    let norm = (n.x + 0.5).clamp(0.0, 1.0);
    let target = range.min + norm * (range.max - range.min);
    let v = clamp_step(target, range.min, range.max, step.0);
    if (val.0 - v).abs() > f32::EPSILON {
      val.0 = v;
    }
  }
}

/// 视觉：填充宽度 + 滑块位置跟随 SliderValue
pub fn slider_visual_system(
  mut q_root: Query<(&SliderValue, &SliderRange, &Children), With<UiSlider>>,
  mut fills: Query<&mut Node, (With<SliderFill>, Without<SliderThumb>)>,
  mut thumbs: Query<&mut Node, With<SliderThumb>>,
) {
  for (val, range, children) in &mut q_root {
    let pct = normalize(val.0, range.min, range.max) * 100.0;
    for child in children.iter() {
      if let Ok(mut node) = fills.get_mut(child) {
        node.width = Val::Percent(pct);
      } else if let Ok(mut node) = thumbs.get_mut(child) {
        node.left = Val::Percent(pct);
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
      child = Some(slider(&ctx, p, 0.0, 100.0, 25.0, Some(5.0)));
    });
    let e = child.expect("slider spawned");
    let w = app.world();
    assert_eq!(
      w.get::<SliderValue>(e).unwrap().0,
      25.0,
      "initial value clamp/step"
    );
    let children = w.get::<Children>(e).unwrap();
    assert_eq!(children.len(), 2, "track + thumb");
  }

  #[test]
  fn slider_drag_updates_value() {
    let mut app = App::new();
    app.add_systems(Update, slider_drag_system);
    let e = app
      .world_mut()
      .spawn((
        UiSlider,
        Interaction::Pressed,
        RelativeCursorPosition {
          cursor_over: true,
          normalized: Some(Vec2::new(0.25, 0.0)),
        },
        SliderRange::default(),
        SliderStep(None),
        SliderValue(0.0),
      ))
      .id();
    app.update();
    // x=0.25（中心原点）→ norm=0.75
    assert!((app.world().get::<SliderValue>(e).unwrap().0 - 0.75).abs() < 1e-6);

    // 未按下不更新
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
        Children::default(),
      ))
      .id();
    let fill = app.world_mut().spawn((SliderFill, Node::default())).id();
    let thumb = app.world_mut().spawn((SliderThumb, Node::default())).id();
    app.world_mut().entity_mut(e).add_child(fill);
    app.world_mut().entity_mut(e).add_child(thumb);
    app.update();
    let fill_node = app.world().get::<Node>(fill).unwrap();
    assert!(matches!(fill_node.width, Val::Percent(p) if (p - 50.0).abs() < 1e-5));
    let thumb_node = app.world().get::<Node>(thumb).unwrap();
    assert!(matches!(thumb_node.left, Val::Percent(p) if (p - 50.0).abs() < 1e-5));
  }
}
