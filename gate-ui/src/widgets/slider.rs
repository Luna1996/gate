//! slider：SliderValue + clamp/step + 拖动 + 强调填充视觉（docs/ui-dark-theme.md §5.4）。
//!
//! 拖动逻辑复用 bevy_ui `RelativeCursorPosition`（ui_focus_system 自动更新）；
//! 归一化坐标以节点中心为原点（-0.5..0.5），映射到 0..1。
//! 视觉：4px 抬升表面轨道槽 + 强调填充段；16px 顶层表面滑块（拖拽放大到 18px）。

use std::ops::Deref;

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

/// 滑杆配置（全部字段进 Config；Default = [0,1] 连续、初值 0）
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SliderConfig {
  pub min: f32,
  pub max: f32,
  pub value: f32,
  pub step: Option<f32>,
}

impl Default for SliderConfig {
  fn default() -> Self {
    Self {
      min: 0.0,
      max: 1.0,
      value: 0.0,
      step: None,
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
pub fn slider(ctx: &UiCtx, parent: &mut ChildSpawner, config: SliderConfig) -> SliderHandle {
  let c = &ctx.theme.colors;
  let m = &ctx.theme.metrics;
  let v = clamp_step(config.value, config.min, config.max, config.step);
  let norm = normalize(v, config.min, config.max);
  let e = parent
    .spawn((
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
        ..default()
      },
      FocusPolicy::Block,
    ))
    .with_children(|root| {
      // 轨道槽（抬升表面 +1 阶：surface_elevated → surface_overlay，增强与填充对比）
      root
        .spawn((
          Name::new("ui-slider-track"),
          Node {
            flex_grow: 1.0,
            height: px(TRACK_HEIGHT),
            ..default()
          },
          BackgroundColor(color_of(&c.surface_overlay)),
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
            BackgroundColor(color_of(&c.accent_fill_hover)),
          ));
        });
      // 滑块（顶层表面 + 提亮边框；方形，无圆角；绝对定位，margin 负半宽居中）
      root.spawn((
        Name::new("ui-slider-thumb"),
        SliderThumb,
        Node {
          position_type: PositionType::Absolute,
          left: Val::Percent(norm * 100.0),
          margin: UiRect::left(Val::Px(-THUMB_SIZE / 2.0)),
          width: px(THUMB_SIZE),
          height: px(THUMB_SIZE),
          border: UiRect::all(px(m.border_width)),
          ..default()
        },
        BackgroundColor(color_of(&c.surface_top)),
        BorderColor::all(color_of(&c.border_strong)),
      ));
    })
    .id();
  SliderHandle(e)
}

fn normalize(v: f32, min: f32, max: f32) -> f32 {
  if (max - min).abs() < f32::EPSILON {
    0.0
  } else {
    ((v - min) / (max - min)).clamp(0.0, 1.0)
  }
}

/// 拖动：按下时把光标归一化 x（-0.5..0.5 中心原点）映射为值（每帧重算）；
/// 值变化时触发 [`SliderValueChanged`]（仅用户拖动，程序写 SliderValue 不触发）
pub fn slider_drag_system(
  mut commands: Commands,
  mut q: Query<(
    Entity,
    &Interaction,
    &RelativeCursorPosition,
    &SliderRange,
    &SliderStep,
    &mut SliderValue,
  )>,
) {
  for (e, inter, rcp, range, step, mut val) in &mut q {
    if *inter != Interaction::Pressed {
      continue;
    }
    let Some(n) = rcp.normalized else { continue };
    let norm = (n.x + 0.5).clamp(0.0, 1.0);
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
/// 注意：fill 是 track 的子节点（slider root 的孙节点），不能只遍历 root 的直接
/// Children——需要下钻 track 的 Children 才能命中 SliderFill。thumb 是 root 直接子。
pub fn slider_visual_system(
  mut q_root: Query<(&SliderValue, &SliderRange, &Interaction, &Children), With<UiSlider>>,
  mut fills: Query<&mut Node, (With<SliderFill>, Without<SliderThumb>)>,
  mut thumbs: Query<&mut Node, With<SliderThumb>>,
  q_track_children: Query<&Children, Without<UiSlider>>,
) {
  for (val, range, inter, children) in &mut q_root {
    let pct = normalize(val.0, range.min, range.max) * 100.0;
    let thumb_size = if *inter == Interaction::Pressed {
      THUMB_SIZE_DRAG
    } else {
      THUMB_SIZE
    };
    for child in children.iter() {
      // thumb 是 root 直接子
      if let Ok(mut node) = thumbs.get_mut(child) {
        node.left = Val::Percent(pct);
        node.width = px(thumb_size);
        node.height = px(thumb_size);
        node.margin = UiRect::left(Val::Px(-thumb_size / 2.0));
      }
      // fill 在 track 内部（孙节点）：下钻 track 的 Children 找 SliderFill
      if let Ok(tc) = q_track_children.get(child) {
        for fill_entity in tc.iter() {
          if let Ok(mut node) = fills.get_mut(fill_entity) {
            node.width = Val::Percent(pct);
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
    let children = w.get::<Children>(e).unwrap();
    assert_eq!(children.len(), 2, "track + thumb");
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
    // 真实结构：root → track → fill；root → thumb
    let track = app.world_mut().spawn((Children::default(),)).id();
    let fill = app.world_mut().spawn((SliderFill, Node::default())).id();
    let thumb = app.world_mut().spawn((SliderThumb, Node::default())).id();
    app.world_mut().entity_mut(track).add_child(fill);
    app.world_mut().entity_mut(e).add_child(track);
    app.world_mut().entity_mut(e).add_child(thumb);
    app.update();
    let fill_node = app.world().get::<Node>(fill).unwrap();
    assert!(matches!(fill_node.width, Val::Percent(p) if (p - 50.0).abs() < 1e-5));
    let thumb_node = app.world().get::<Node>(thumb).unwrap();
    assert!(matches!(thumb_node.left, Val::Percent(p) if (p - 50.0).abs() < 1e-5));
  }
}
