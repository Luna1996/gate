//! slider：SliderValue + clamp/step + 拖动 + 强调填充视觉。
//! 拖动复用 bevy_ui `RelativeCursorPosition`（归一化坐标以节点中心为原点，-0.5..0.5）。
//! 视觉：`TRACK_HEIGHT` 轨道槽 + 强调填充段 + `THUMB_SIZE` 顶层滑块（拖拽放大到 18px）。

use std::ops::Deref;

use bevy::prelude::*;
use bevy::ui::{FocusPolicy, Interaction, RelativeCursorPosition};

use super::{UiCtx, UiDisabled, color_of, dim_color, px};
use crate::widgets::consts::{THUMB_INSET, THUMB_SIZE, THUMB_SIZE_DRAG, TRACK_HEIGHT};

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

/// 滑块行程槽标记：track 内左右各内缩半 thumb 宽的绝对定位容器，
/// 滑块放在它里面 → `left: norm%` 的行程端点让滑块整块落在轨道槽两端之内
#[derive(Component, Debug, Default)]
pub struct SliderThumbSlot;

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
    Self { min: 0.0, max: 1.0, value: 0.0, step: None, disabled: false }
  }
}

/// 滑杆值变化事件（仅用户拖动产生的变化触发；EntityEvent，target = 滑杆根实体）。
/// 组件 `SliderValue` 仍是真源，事件只是通知，主动读值可 Query。
#[derive(EntityEvent, Clone, Copy, Debug, PartialEq)]
pub struct SliderValueChanged {
  pub entity: Entity,
  pub value: f32,
}

/// clamp + step 归一（纯函数）
pub fn clamp_step(value: f32, min: f32, max: f32, step: Option<f32>) -> f32 {
  let v = value.clamp(min, max);
  match step {
    Some(s) if s > 0.0 => min + ((v - min) / s).round() * s,
    _ => v,
  }
  .clamp(min, max)
}

/// 主题滑杆（横向；宽度由父容器 flex 决定，最小 120px）。
/// 根节点无水平内边距 → 轨道槽两端与同行其它控件左右边缘对齐；轨道槽内套「行程槽」
/// （绝对定位 + 左右各内缩半 thumb 宽），thumb 的 `left: norm%` 相对槽宽（= track 宽 - thumb 宽）。
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
    (dim_color(track_bg), dim_color(fill_bg), dim_color(thumb_bg), dim_color(thumb_border))
  } else {
    (track_bg, fill_bg, thumb_bg, thumb_border)
  };
  let mut ec = parent.spawn((
    Name::new("ui-slider"),
    UiSlider,
    Interaction::default(),
    RelativeCursorPosition::default(),
    SliderRange { min: config.min, max: config.max },
    SliderStep(config.step),
    SliderValue(v),
    Node { min_width: px(120.0), height: px(24.0), align_items: AlignItems::Center, ..default() },
    FocusPolicy::Block,
  ));
  ec.with_children(|root| {
    // flex_grow 撑满内容盒（根宽 - 2*THUMB_INSET）；fill 与 thumb 均以其为包含块
    root
      .spawn((
        Name::new("ui-slider-track"),
        Node { flex_grow: 1.0, height: px(TRACK_HEIGHT), ..default() },
        BackgroundColor(track_bg),
      ))
      .with_children(|track| {
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
        // 盒宽 = track 宽 - thumb 宽；滑块以它为包含块做 left: norm%（两端档位不出界）
        track
          .spawn((
            Name::new("ui-slider-thumb-slot"),
            SliderThumbSlot,
            Node {
              position_type: PositionType::Absolute,
              left: Val::ZERO,
              right: Val::ZERO,
              top: Val::ZERO,
              bottom: Val::ZERO,
              margin: UiRect::horizontal(px(THUMB_INSET)),
              ..default()
            },
          ))
          .with_children(|slot| {
            // margin 左/上各负半尺寸把滑块中心钉在定位点；top:50% 相对槽高（= track 高）
            slot.spawn((
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
  });
  if config.disabled {
    ec.insert(UiDisabled);
  }
  SliderHandle(ec.id())
}

fn normalize(v: f32, min: f32, max: f32) -> f32 {
  if (max - min).abs() < f32::EPSILON { 0.0 } else { ((v - min) / (max - min)).clamp(0.0, 1.0) }
}

/// 拖动：按下时把光标位置映射为值（每帧重算）；值变化时触发 `SliderValueChanged`（仅用户拖动）。
/// 映射基准 = 行程槽（根内容盒再各内缩半 thumb 宽）；光标落在槽外即吸附 min/max；有 step 时经 `clamp_step` 逐档吸附。
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
    // normalized 以节点中心为原点（-0.5..0.5）；size 与 padding（min_inset.x=左、max_inset.x=右）同为物理 px，比值单位自洽
    let root_w = node.size().x;
    let cursor_x = (n.x + 0.5) * root_w;
    let pad_l = node.padding.min_inset.x;
    let pad_r = node.padding.max_inset.x;
    // 行程槽 = 根内容盒再各内缩半 thumb 宽（与 visual 里 thumb 的行程一致）
    let travel_w = root_w - pad_l - pad_r - THUMB_SIZE;
    let norm = if travel_w > 0.0 {
      ((cursor_x - pad_l - THUMB_SIZE / 2.0) / travel_w).clamp(0.0, 1.0)
    } else {
      0.0
    };
    let target = range.min + norm * (range.max - range.min);
    let v = clamp_step(target, range.min, range.max, step.0);
    if (val.0 - v).abs() > f32::EPSILON {
      val.0 = v;
      commands.trigger(SliderValueChanged { entity: e, value: v });
    }
  }
}

/// 视觉：填充宽度 + 滑块位置跟随 SliderValue；拖拽中滑块放大到 18px。
/// Disabled：滑块恒为 16px（配色由 spawn 时降亮）。
/// 结构 root → track → (fill | thumb_slot → thumb)；thumb 的 `top:50%` + 负半高 margin 垂直居中也在此同步。
#[allow(clippy::type_complexity)] // Bevy system：多组件查询签名固有
pub fn slider_visual_system(
  mut q_root: Query<
    (&SliderValue, &SliderRange, &Interaction, &Children, Has<UiDisabled>),
    With<UiSlider>,
  >,
  mut fills: Query<&mut Node, (With<SliderFill>, Without<SliderThumb>)>,
  mut thumbs: Query<&mut Node, With<SliderThumb>>,
  q_track_children: Query<&Children, Without<UiSlider>>,
  q_slot_children: Query<&Children, With<SliderThumbSlot>>,
) {
  for (val, range, inter, children, disabled) in &mut q_root {
    let pct = normalize(val.0, range.min, range.max) * 100.0;
    let thumb_size =
      if !disabled && *inter == Interaction::Pressed { THUMB_SIZE_DRAG } else { THUMB_SIZE };
    for child in children.iter() {
      // fill 是 track 的子节点；thumb 在 track 里的行程槽下（再下一层）
      if let Ok(tc) = q_track_children.get(child) {
        for sub in tc.iter() {
          if let Ok(mut node) = fills.get_mut(sub) {
            node.width = Val::Percent(pct);
          }
          let Ok(sc) = q_slot_children.get(sub) else { continue };
          for thumb in sc.iter() {
            if let Ok(mut node) = thumbs.get_mut(thumb) {
              node.left = Val::Percent(pct);
              // margin 负半高把中心钉在 track 中心线；拖拽放大时尺寸与负 margin 同步
              node.top = Val::Percent(50.0);
              node.width = px(thumb_size);
              node.height = px(thumb_size);
              node.margin =
                UiRect { left: px(-thumb_size / 2.0), top: px(-thumb_size / 2.0), ..default() };
            }
          }
        }
      }
    }
  }
}
