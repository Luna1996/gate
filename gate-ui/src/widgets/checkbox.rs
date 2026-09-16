//! checkbox：勾选切换 + 勾选视觉（docs/ui-dark-theme.md §5.3）。
//!
//! 复用 bevy_ui `Checked` / `Checkable`（presence 语义，自带 a11y 联动）：有 `Checked` = 勾选。
//! 视觉：16×16 盒子；未选中 = 抬升表面 + 边框（hover 边框提亮），选中 = 强调填充 + 对勾标记。

use std::ops::Deref;

use bevy::prelude::*;
use bevy::ui::{Checkable, Checked, FocusPolicy, Interaction};

use super::button::InteractionPrev;
use super::{UiCtx, UiDisabled, color_of, dim_color, px, spawn_label};
use crate::theme::UiTheme;

/// 勾选框视觉方块子实体标记
#[derive(Component, Debug, Default)]
pub struct CheckboxBox;

/// 盒子边长（px）
const BOX_SIZE: f32 = 16.0;

/// 勾选框句柄（Deref 到根实体 Entity）
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CheckboxHandle(pub Entity);

impl Deref for CheckboxHandle {
  type Target = Entity;
  fn deref(&self) -> &Entity {
    &self.0
  }
}

impl From<CheckboxHandle> for Entity {
  fn from(h: CheckboxHandle) -> Entity {
    h.0
  }
}

/// 勾选框配置（全部字段进 Config；Default = 无文本、未勾选、不禁用）
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CheckboxConfig {
  pub text: Option<String>,
  pub checked: bool,
  /// true = 禁用态：不响应点击翻转、配色暗一档（勾选状态仍可见）
  pub disabled: bool,
}

/// 勾选状态变化事件（用户点击翻转时触发；EntityEvent，target = 根实体）。
/// `Checked` 组件仍是真源，事件只是通知，主动读状态可 Query/Has。
#[derive(EntityEvent, Clone, Copy, Debug, PartialEq)]
pub struct CheckboxToggled {
  pub entity: Entity,
  /// 翻转后的新状态
  pub checked: bool,
}

/// 勾选框（可选文本标签）
pub fn checkbox(ctx: &UiCtx, parent: &mut ChildSpawner, config: CheckboxConfig) -> CheckboxHandle {
  let c = &ctx.theme.colors;
  let m = &ctx.theme.metrics;
  let box_bg = color_of(&c.surface_elevated);
  let box_border = color_of(&c.border);
  let (box_bg, box_border) =
    if config.disabled { (dim_color(box_bg), dim_color(box_border)) } else { (box_bg, box_border) };
  let mut ec = parent.spawn((
    Name::new("ui-checkbox"),
    Interaction::default(),
    InteractionPrev::default(),
    Checkable,
    Node { align_items: AlignItems::Center, column_gap: px(m.spacing.sm), ..default() },
    FocusPolicy::Block,
  ));
  ec.with_children(|root| {
    root
      .spawn((
        Name::new("ui-checkbox-box"),
        CheckboxBox,
        Node {
          width: px(BOX_SIZE),
          height: px(BOX_SIZE),
          border: UiRect::all(px(m.border_width)),
          border_radius: BorderRadius::all(px(m.corner_radius_sm)),
          align_items: AlignItems::Center,
          justify_content: JustifyContent::Center,
          ..default()
        },
        BackgroundColor(box_bg),
        BorderColor::all(box_border),
      ))
      .with_children(|box_node| {
        box_node.spawn((
          Name::new("ui-checkbox-mark"),
          Node { width: px(BOX_SIZE * 0.5), height: px(BOX_SIZE * 0.5), ..default() },
          // 对勾标记 = 主文本色（强调填充上的最高对比）
          BackgroundColor(color_of(&c.text_primary)),
          Visibility::Hidden,
        ));
      });
    if let Some(t) = config.text {
      let text_color = color_of(&c.text_body);
      let text_color = if config.disabled { dim_color(text_color) } else { text_color };
      spawn_label(ctx, root, t, m.font_size.md, text_color);
    }
  });
  if config.disabled {
    ec.insert(UiDisabled);
  }
  if config.checked {
    ec.insert(Checked);
  }
  CheckboxHandle(ec.id())
}

/// 勾选状态机查询集（type alias 满足 clippy::type_complexity）
type CheckboxQuery = (
  Entity,
  &'static Interaction,
  &'static mut InteractionPrev,
  Has<Checked>,
  &'static Children,
  Has<UiDisabled>,
);

/// 勾选状态机：释放时翻转 Checked 并触发 [`CheckboxToggled`]；
/// 盒子背景/边框/勾选标记跟随状态（每帧重算）
///
/// Disabled 态：跳过翻转逻辑，配色降亮一档，勾选标记仍按当前 checked 状态显示。
pub fn checkbox_state_system(
  mut commands: Commands,
  theme: Option<Res<UiTheme>>,
  mut q: Query<CheckboxQuery, With<Checkable>>,
  mut boxes: Query<(&mut BackgroundColor, &mut BorderColor, &Children), With<CheckboxBox>>,
  mut marks: Query<&mut Visibility, Without<CheckboxBox>>,
) {
  let Some(theme) = theme else { return };
  let c = &theme.colors;
  let accent = color_of(&c.accent_fill);
  let elevated = color_of(&c.surface_elevated);
  let border = color_of(&c.border);
  let border_strong = color_of(&c.border_strong);
  for (e, inter, mut prev, checked, children, disabled) in &mut q {
    if !disabled {
      // click = 按下并释放（与 button 判定一致）
      if prev.0 == Interaction::Pressed && *inter == Interaction::Hovered {
        if checked {
          commands.entity(e).remove::<Checked>();
        } else {
          commands.entity(e).insert(Checked);
        }
        commands.trigger(CheckboxToggled { entity: e, checked: !checked });
      }
    }
    prev.0 = *inter;
    let hovered = !disabled && *inter == Interaction::Hovered;
    // 视觉：选中 → 强调填充；未选中 → 抬升表面（hover 边框提亮）
    let (target_bg, target_border) = if checked {
      (accent, accent)
    } else if hovered {
      (elevated, border_strong)
    } else {
      (elevated, border)
    };
    let (target_bg, target_border) = if disabled {
      (dim_color(target_bg), dim_color(target_border))
    } else {
      (target_bg, target_border)
    };
    for child in children.iter() {
      if let Ok((mut bg, mut bc, box_children)) = boxes.get_mut(child) {
        if bg.0 != target_bg {
          bg.0 = target_bg;
        }
        if bc.top != target_border {
          *bc = BorderColor::all(target_border);
        }
        for mark in box_children.iter() {
          if let Ok(mut vis) = marks.get_mut(mark) {
            // 选中 = Inherited（跟随祖先显隐）；显式 Visible 会无视祖先
            // Hidden 强制可见 → 面板整体隐藏时勾选标记单独悬浮
            let target_vis = if checked { Visibility::Inherited } else { Visibility::Hidden };
            if *vis != target_vis {
              *vis = target_vis;
            }
          }
        }
      }
    }
  }
}
