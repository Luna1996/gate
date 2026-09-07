//! toggle_switch：滑动开关（docs/ui-dark-theme.md §5.3 选中态语义）。
//!
//! 复用 `Checked` presence 语义（与 checkbox 一致）：有 `Checked` = 开，无 = 关。
//!
//! 视觉：32×16 轨道（圆角 sm）+ 12×12 方形滑块（absolute，左 = 关 / 右 = 开）：
//! - 关：轨道 = 抬升表面 + 边框（hover 边框提亮），滑块 = 说明文字灰
//! - 开：轨道 = 强调填充，滑块 = 主文本色（强调填充上的最高对比）

use std::ops::Deref;

use bevy::prelude::*;
use bevy::ui::{FocusPolicy, Interaction};

use super::button::InteractionPrev;
use super::{UiCtx, color_of, px, spawn_label};
use crate::theme::UiTheme;
use bevy::ui::Checked;

/// 开关根节点标记
#[derive(Component, Debug, Default)]
pub struct ToggleSwitch;

/// 轨道标记（滑块定位/配色的查询目标）
#[derive(Component, Debug, Default)]
pub struct ToggleTrack;

/// 滑块标记
#[derive(Component, Debug, Default)]
pub struct ToggleKnob;

/// 轨道尺寸（px）
const TRACK_W: f32 = 32.0;
const TRACK_H: f32 = 16.0;

/// 开关句柄（Deref 到根实体 Entity）
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ToggleSwitchHandle(pub Entity);

impl Deref for ToggleSwitchHandle {
  type Target = Entity;
  fn deref(&self) -> &Entity {
    &self.0
  }
}

impl From<ToggleSwitchHandle> for Entity {
  fn from(h: ToggleSwitchHandle) -> Entity {
    h.0
  }
}

/// 开关配置（Default = 无文本、关）
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ToggleSwitchConfig {
  pub text: Option<String>,
  pub checked: bool,
}

/// 开关状态变化事件（用户点击翻转时触发；EntityEvent，target = 根实体）。
/// `Checked` 组件仍是真源，事件只是通知，主动读状态可 Query/Has。
#[derive(EntityEvent, Clone, Copy, Debug, PartialEq)]
pub struct ToggleSwitchToggled {
  pub entity: Entity,
  /// 翻转后的新状态
  pub checked: bool,
}

/// 滑动开关（可选文本标签）
pub fn toggle_switch(
  ctx: &UiCtx,
  parent: &mut ChildSpawner,
  config: ToggleSwitchConfig,
) -> ToggleSwitchHandle {
  let c = &ctx.theme.colors;
  let m = &ctx.theme.metrics;
  let mut ec = parent.spawn((
    Name::new("ui-toggle-switch"),
    ToggleSwitch,
    Interaction::default(),
    InteractionPrev::default(),
    Node {
      align_items: AlignItems::Center,
      column_gap: px(m.spacing.sm),
      ..default()
    },
    FocusPolicy::Block,
  ));
  ec.with_children(|root| {
    root
      .spawn((
        Name::new("ui-toggle-track"),
        ToggleTrack,
        Node {
          width: px(TRACK_W),
          height: px(TRACK_H),
          border: UiRect::all(px(m.border_width)),
          border_radius: BorderRadius::all(px(m.corner_radius_sm)),
          ..default()
        },
        BackgroundColor(color_of(&c.surface_elevated)),
        BorderColor::all(color_of(&c.border)),
      ))
      .with_children(|track| {
        track.spawn((
          Name::new("ui-toggle-knob"),
          ToggleKnob,
          Node {
            position_type: PositionType::Absolute,
            width: px(TRACK_H - m.border_width * 2.0),
            height: px(TRACK_H - m.border_width * 2.0),
            ..default()
          },
          BackgroundColor(color_of(&c.text_muted)),
        ));
      });
    if let Some(t) = config.text {
      spawn_label(ctx, root, t, m.font_size.md, color_of(&c.text_body));
    }
  });
  if config.checked {
    ec.insert(Checked);
  }
  ToggleSwitchHandle(ec.id())
}

/// 开关状态机查询集（type alias 满足 clippy::type_complexity）
type ToggleQuery = (
  Entity,
  &'static Interaction,
  &'static mut InteractionPrev,
  Has<Checked>,
  &'static Children,
);
type TrackData = (
  &'static mut BackgroundColor,
  &'static mut BorderColor,
  &'static Children,
);
type TrackFilter = (With<ToggleTrack>, Without<ToggleSwitch>);
type KnobData = (&'static mut Node, &'static mut BackgroundColor);
type KnobFilter = (With<ToggleKnob>, Without<ToggleTrack>);

/// 开关状态机：释放时翻转 `Checked` 并触发 [`ToggleSwitchToggled`]；
/// 轨道/滑块配色与滑块位置跟随状态（每帧重算）
pub fn toggle_switch_state_system(
  mut commands: Commands,
  theme: Option<Res<UiTheme>>,
  mut q: Query<ToggleQuery, With<ToggleSwitch>>,
  mut tracks: Query<TrackData, TrackFilter>,
  mut knobs: Query<KnobData, KnobFilter>,
) {
  let Some(theme) = theme else { return };
  let c = &theme.colors;
  let accent = color_of(&c.accent_fill);
  let elevated = color_of(&c.surface_elevated);
  let border = color_of(&c.border);
  let border_strong = color_of(&c.border_strong);
  let knob_off = color_of(&c.text_muted);
  let knob_on = color_of(&c.text_primary);
  let left_on = px(TRACK_W - TRACK_H);
  for (e, inter, mut prev, checked, children) in &mut q {
    // click = 按下并释放（与 button/checkbox 判定一致）
    if prev.0 == Interaction::Pressed && *inter == Interaction::Hovered {
      if checked {
        commands.entity(e).remove::<Checked>();
      } else {
        commands.entity(e).insert(Checked);
      }
      commands.trigger(ToggleSwitchToggled {
        entity: e,
        checked: !checked,
      });
    }
    prev.0 = *inter;
    let hovered = *inter == Interaction::Hovered;
    // 视觉：开 → 强调填充轨道 + 主色滑块；关 → 抬升轨道 + 灰滑块（hover 边框提亮）
    let (target_bg, target_border) = if checked {
      (accent, accent)
    } else if hovered {
      (elevated, border_strong)
    } else {
      (elevated, border)
    };
    let target_left = if checked { left_on } else { px(0.0) };
    let target_knob = if checked { knob_on } else { knob_off };
    for child in children.iter() {
      if let Ok((mut bg, mut bc, track_children)) = tracks.get_mut(child) {
        if bg.0 != target_bg {
          bg.0 = target_bg;
        }
        if bc.top != target_border {
          *bc = BorderColor::all(target_border);
        }
        for knob in track_children.iter() {
          if let Ok((mut node, mut knob_bg)) = knobs.get_mut(knob) {
            if node.left != target_left {
              node.left = target_left;
            }
            if knob_bg.0 != target_knob {
              knob_bg.0 = target_knob;
            }
          }
        }
      }
    }
  }
}
