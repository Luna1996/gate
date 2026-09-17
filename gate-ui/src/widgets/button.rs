//! button：四变体 + Interaction 状态机 + hover/pressed 视觉 + 按压缩放 + UiClick。
//! 变体：Primary（强调填充）/ Danger（破坏性）/ Secondary（抬升表面+边框）/
//! Ghost（透明底）。反馈：hover 提亮一档；pressed 填充压深 + `UiTransform` scale 0.98（绕节点中心）。

use std::ops::Deref;

use bevy::prelude::*;
use bevy::ui::widget::Button;
use bevy::ui::{FocusPolicy, Interaction, UiTransform};

use super::{UiCtx, UiDisabled, color_of, dim_color, px, spawn_label};
use crate::theme::UiTheme;

/// 按钮点击事件（EntityEvent，target = 按钮实体）。
/// 用法：`entity_mut(btn).observe(|_: On<UiClick>| ...)` 或 `app.add_observer(...)`。
#[derive(EntityEvent, Clone, Copy, Debug, PartialEq)]
pub struct UiClick {
  pub entity: Entity,
}

/// 上一帧 Interaction 状态（click = Pressed → Hovered 释放判定）
#[derive(Component, Clone, Copy, Debug, PartialEq, Default)]
pub struct InteractionPrev(pub Interaction);

/// 按钮变体（驱动状态机配色）
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ButtonVariant {
  #[default]
  Primary,
  Secondary,
  Ghost,
  Danger,
}

/// 按下时绕节点中心缩放到 98%
const PRESSED_SCALE: f32 = 0.98;

/// 按钮句柄（Deref 到根实体 Entity）
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ButtonHandle(pub Entity);

impl Deref for ButtonHandle {
  type Target = Entity;
  fn deref(&self) -> &Entity {
    &self.0
  }
}

impl From<ButtonHandle> for Entity {
  fn from(h: ButtonHandle) -> Entity {
    h.0
  }
}

/// 按钮配置（全部字段进 Config；Default = 空文本 + Primary、不禁用）
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ButtonConfig {
  pub text: String,
  pub variant: ButtonVariant,
  /// true = 禁用态：不响应点击、不缩放、配色暗一档
  pub disabled: bool,
}

/// 主题按钮（文本子标签；变体由 `ButtonConfig::variant` 决定）
pub fn button(ctx: &UiCtx, parent: &mut ChildSpawner, config: ButtonConfig) -> ButtonHandle {
  let m = &ctx.theme.metrics;
  let (bg, border, text_color) = variant_colors(ctx.theme, config.variant, Interaction::None);
  let (bg, border, text_color) = if config.disabled {
    (dim_color(bg), dim_color(border), dim_color(text_color))
  } else {
    (bg, border, text_color)
  };
  let mut ec = parent.spawn((
    Name::new("ui-button"),
    Button,
    config.variant,
    Interaction::default(),
    InteractionPrev::default(),
    Node {
      padding: UiRect {
        left: px(m.spacing.md),
        right: px(m.spacing.md),
        top: px(m.spacing.sm),
        bottom: px(m.spacing.sm),
      },
      border: UiRect::all(px(m.border_width)),
      align_items: AlignItems::Center,
      justify_content: JustifyContent::Center,
      ..default()
    },
    BackgroundColor(bg),
    BorderColor::all(border),
    UiTransform::default(),
    FocusPolicy::Block,
  ));
  if config.disabled {
    ec.insert(UiDisabled);
  }
  let e = ec
    .with_children(|b| {
      spawn_label(ctx, b, config.text, m.font_size.md, text_color);
    })
    .id();
  ButtonHandle(e)
}

/// 变体 × 交互态 → (背景, 边框, 文字) 配色
fn variant_colors(
  theme: &UiTheme,
  variant: ButtonVariant,
  inter: Interaction,
) -> (Color, Color, Color) {
  let c = &theme.colors;
  let none = Color::NONE;
  match variant {
    ButtonVariant::Primary => {
      let bg = match inter {
        Interaction::Pressed => color_of(&c.accent_fill_pressed),
        Interaction::Hovered => color_of(&c.accent_fill_hover),
        Interaction::None => color_of(&c.accent_fill),
      };
      (bg, bg, color_of(&c.text_primary))
    }
    ButtonVariant::Danger => {
      // danger 仅一档填充，hover 靠边框提亮区分
      let bg = color_of(&c.danger_fill);
      let border = match inter {
        Interaction::Hovered => color_of(&c.danger),
        _ => color_of(&c.danger_fill),
      };
      (bg, border, color_of(&c.text_primary))
    }
    ButtonVariant::Secondary => {
      let bg = match inter {
        Interaction::Pressed => color_of(&c.surface_card),
        Interaction::Hovered => color_of(&c.surface_overlay),
        Interaction::None => color_of(&c.surface_elevated),
      };
      let border = match inter {
        Interaction::None => color_of(&c.border),
        _ => color_of(&c.border_strong),
      };
      (bg, border, color_of(&c.text_body))
    }
    ButtonVariant::Ghost => {
      let bg = match inter {
        Interaction::Pressed => color_of(&c.surface_card),
        Interaction::Hovered => color_of(&c.surface_elevated),
        Interaction::None => none,
      };
      let text = match inter {
        Interaction::None => color_of(&c.text_muted),
        _ => color_of(&c.text_body),
      };
      (bg, none, text)
    }
  }
}

/// 按钮状态机查询集（type alias 满足 clippy::type_complexity）
type ButtonQuery = (
  Entity,
  &'static Interaction,
  &'static mut InteractionPrev,
  &'static mut BackgroundColor,
  &'static mut BorderColor,
  &'static mut UiTransform,
  &'static ButtonVariant,
  &'static Children,
  Has<UiDisabled>,
);

/// 按钮状态机：hover/pressed 视觉 + 按压缩放 + 释放触发 UiClick（每帧重算）；Disabled 态跳过
/// click、配色恒 None 态并降亮、不缩放。查询须带 `With<Button>`。
pub fn button_state_system(
  mut commands: Commands,
  theme: Option<Res<UiTheme>>,
  mut q: Query<ButtonQuery, With<Button>>,
  mut q_text: Query<&mut TextColor>,
) {
  let Some(theme) = theme else { return };
  for (e, inter, mut prev, mut bg, mut border, mut ui_t, variant, children, disabled) in &mut q {
    if disabled {
      let (target_bg, target_border, target_text) =
        variant_colors(&theme, *variant, Interaction::None);
      let target_bg = dim_color(target_bg);
      let target_border = dim_color(target_border);
      let target_text = dim_color(target_text);
      if bg.0 != target_bg {
        bg.0 = target_bg;
      }
      let target_border = BorderColor::all(target_border);
      if *border != target_border {
        *border = target_border;
      }
      if (ui_t.scale.x - 1.0).abs() > f32::EPSILON {
        ui_t.scale = Vec2::splat(1.0);
      }
      for child in children.iter() {
        if let Ok(mut tc) = q_text.get_mut(child)
          && tc.0 != target_text
        {
          tc.0 = target_text;
        }
      }
      // 重置 prev 避免 re-enable 瞬间误触发 click
      prev.0 = Interaction::None;
      continue;
    }
    // click = 按下后在按钮上释放（拖出释放视为取消）
    if prev.0 == Interaction::Pressed && *inter == Interaction::Hovered {
      commands.trigger(UiClick { entity: e });
    }
    prev.0 = *inter;
    let (target_bg, target_border, target_text) = variant_colors(&theme, *variant, *inter);
    if bg.0 != target_bg {
      bg.0 = target_bg;
    }
    let target_border = BorderColor::all(target_border);
    if *border != target_border {
      *border = target_border;
    }
    // 按压缩放（绕节点中心，bevy_ui 变换原点即节点中心）
    let target_scale = if *inter == Interaction::Pressed { PRESSED_SCALE } else { 1.0 };
    if (ui_t.scale.x - target_scale).abs() > f32::EPSILON {
      ui_t.scale = Vec2::splat(target_scale);
    }
    // 文本子标签颜色（按钮只有一个文本子节点）
    for child in children.iter() {
      if let Ok(mut tc) = q_text.get_mut(child)
        && tc.0 != target_text
      {
        tc.0 = target_text;
      }
    }
  }
}
