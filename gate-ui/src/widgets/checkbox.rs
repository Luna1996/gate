//! checkbox：勾选切换 + 勾选视觉。
//!
//! 复用 bevy_ui `Checked` / `Checkable`（presence 语义，自带 a11y 联动）：
//! 有 `Checked` = 勾选，无 = 未勾选。

use bevy::prelude::*;
use bevy::ui::{Checkable, Checked, FocusPolicy, Interaction};

use super::button::InteractionPrev;
use super::{UiCtx, color_of, px, spawn_label};
use crate::theme::UiTheme;

/// 勾选框视觉方块子实体标记
#[derive(Component, Debug, Default)]
pub struct CheckboxBox;

/// 勾选框（可选文本标签）
pub fn checkbox(
  ctx: &UiCtx,
  parent: &mut ChildSpawner,
  text: Option<&str>,
  checked: bool,
) -> Entity {
  let c = &ctx.theme.colors;
  let m = &ctx.theme.metrics;
  let size = m.font_size.lg;
  let mut ec = parent.spawn((
    Name::new("ui-checkbox"),
    Interaction::default(),
    InteractionPrev::default(),
    Checkable,
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
        Name::new("ui-checkbox-box"),
        CheckboxBox,
        Node {
          width: px(size),
          height: px(size),
          border: UiRect::all(px(m.border_width)),
          border_radius: BorderRadius::all(px(m.corner_radius / 2.0)),
          align_items: AlignItems::Center,
          justify_content: JustifyContent::Center,
          ..default()
        },
        BackgroundColor(Color::NONE),
        BorderColor::all(color_of(&c.panel_border)),
      ))
      .with_children(|box_node| {
        box_node.spawn((
          Name::new("ui-checkbox-mark"),
          Node {
            width: px(size * 0.5),
            height: px(size * 0.5),
            border_radius: BorderRadius::all(px(m.corner_radius / 4.0)),
            ..default()
          },
          BackgroundColor(color_of(&c.accent)),
          Visibility::Hidden,
        ));
      });
    if let Some(t) = text {
      spawn_label(ctx, root, t.to_string(), m.font_size.md, color_of(&c.text));
    }
  });
  if checked {
    ec.insert(Checked);
  }
  ec.id()
}

/// 勾选状态机查询集（type alias 满足 clippy::type_complexity）
type CheckboxQuery = (
  Entity,
  &'static Interaction,
  &'static mut InteractionPrev,
  Has<Checked>,
  &'static Children,
);

/// 勾选状态机：释放时翻转 Checked；方块背景/勾选标记跟随状态（每帧重算）
pub fn checkbox_state_system(
  mut commands: Commands,
  theme: Option<Res<UiTheme>>,
  mut q: Query<CheckboxQuery, With<Checkable>>,
  mut boxes: Query<(&mut BackgroundColor, &Children), With<CheckboxBox>>,
  mut marks: Query<&mut Visibility, Without<CheckboxBox>>,
) {
  let Some(theme) = theme else { return };
  let c = &theme.colors;
  let accent = color_of(&c.accent);
  let none = Color::NONE;
  for (e, inter, mut prev, checked, children) in &mut q {
    // click = 按下并释放（与 button 判定一致）
    if prev.0 == Interaction::Pressed && *inter == Interaction::Hovered {
      if checked {
        commands.entity(e).remove::<Checked>();
      } else {
        commands.entity(e).insert(Checked);
      }
    }
    prev.0 = *inter;
    // 视觉：方块填充 = 勾选时 accent
    for child in children.iter() {
      if let Ok((mut bg, box_children)) = boxes.get_mut(child) {
        let target = if checked { accent } else { none };
        if bg.0 != target {
          bg.0 = target;
        }
        for mark in box_children.iter() {
          if let Ok(mut vis) = marks.get_mut(mark) {
            let target_vis = if checked {
              Visibility::Visible
            } else {
              Visibility::Hidden
            };
            if *vis != target_vis {
              *vis = target_vis;
            }
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
  fn checkbox_toggle_and_visual() {
    let theme = default_theme();
    let mut app = App::new();
    app.insert_resource(theme.clone());
    app.add_systems(Update, checkbox_state_system);

    let root = app
      .world_mut()
      .spawn((
        Interaction::None,
        InteractionPrev::default(),
        Checkable,
        Children::default(),
      ))
      .id();
    let box_e = app
      .world_mut()
      .spawn((CheckboxBox, BackgroundColor::default(), Children::default()))
      .id();
    let mark = app
      .world_mut()
      .spawn((Node::default(), Visibility::default()))
      .id();
    app.world_mut().entity_mut(box_e).add_child(mark);
    app.world_mut().entity_mut(root).add_child(box_e);

    // 初始未勾选：方块透明、标记隐藏
    app.update();
    assert_eq!(
      app.world().get::<BackgroundColor>(box_e).unwrap().0,
      Color::NONE
    );
    assert_eq!(
      *app.world().get::<Visibility>(mark).unwrap(),
      Visibility::Hidden
    );

    // 按下 → 释放：翻转为勾选
    app
      .world_mut()
      .get_mut::<Interaction>(root)
      .unwrap()
      .set_if_neq(Interaction::Pressed);
    app.update();
    app
      .world_mut()
      .get_mut::<Interaction>(root)
      .unwrap()
      .set_if_neq(Interaction::Hovered);
    app.update();
    assert!(
      app.world().get::<Checked>(root).is_some(),
      "release toggles on"
    );
    app.update();
    assert_eq!(
      app.world().get::<BackgroundColor>(box_e).unwrap().0,
      color_of(&theme.colors.accent)
    );
    assert_eq!(
      *app.world().get::<Visibility>(mark).unwrap(),
      Visibility::Visible
    );

    // 再点击一次：翻回未勾选
    app
      .world_mut()
      .get_mut::<Interaction>(root)
      .unwrap()
      .set_if_neq(Interaction::Pressed);
    app.update();
    app
      .world_mut()
      .get_mut::<Interaction>(root)
      .unwrap()
      .set_if_neq(Interaction::Hovered);
    app.update();
    assert!(
      app.world().get::<Checked>(root).is_none(),
      "release toggles off"
    );
    app.update();
    assert_eq!(
      app.world().get::<BackgroundColor>(box_e).unwrap().0,
      Color::NONE
    );
  }

  #[test]
  fn checkbox_spawn_structure() {
    let theme = default_theme();
    let ctx = UiCtx::new(&theme, None);
    let mut app = App::new();
    let root = app.world_mut().spawn_empty().id();
    let mut child = None;
    app.world_mut().entity_mut(root).with_children(|p| {
      child = Some(checkbox(&ctx, p, Some("opt"), false));
    });
    let e = child.expect("checkbox spawned");
    let w = app.world();
    assert!(w.get::<Checkable>(e).is_some());
    assert!(w.get::<Checked>(e).is_none(), "unchecked initial");
    let children = w.get::<Children>(e).unwrap();
    assert_eq!(children.len(), 2, "box + label");
  }
}
