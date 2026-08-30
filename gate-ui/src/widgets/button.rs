//! button：Interaction 状态机 + hover/pressed 视觉态 + UiClick 观察者事件。

use bevy::prelude::*;
use bevy::ui::widget::Button;
use bevy::ui::{FocusPolicy, Interaction};

use super::{UiCtx, color_of, px, spawn_label};
use crate::theme::UiTheme;

/// 按钮点击事件（EntityEvent，target = 按钮实体）。
/// 用法：`app.world_mut().entity_mut(btn).observe(|_: On<UiClick>| { ... })`
/// 或全局：`app.add_observer(|_: On<UiClick>| { ... })`
#[derive(EntityEvent, Clone, Copy, Debug, PartialEq)]
pub struct UiClick {
  pub entity: Entity,
}

/// 上一帧 Interaction 状态（click = Pressed → Hovered 释放判定）
#[derive(Component, Clone, Copy, Debug, PartialEq, Default)]
pub struct InteractionPrev(pub Interaction);

/// 主题按钮（文本子标签）
pub fn button(ctx: &UiCtx, parent: &mut ChildSpawner, text: &str) -> Entity {
  let c = &ctx.theme.colors;
  let m = &ctx.theme.metrics;
  parent
    .spawn((
      Name::new("ui-button"),
      Button,
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
        border_radius: BorderRadius::all(px(m.corner_radius)),
        align_items: AlignItems::Center,
        justify_content: JustifyContent::Center,
        ..default()
      },
      BackgroundColor(color_of(&c.accent)),
      BorderColor::all(color_of(&c.panel_border)),
      FocusPolicy::Block,
    ))
    .with_children(|b| {
      spawn_label(ctx, b, text.to_string(), m.font_size.md, color_of(&c.text));
    })
    .id()
}

/// 按钮状态机：hover/pressed 视觉 + 释放触发 UiClick（每帧重算）
///
/// 注意必须 With<Button>：0.19 中 Node require BackgroundColor，所有 UI 节点都带
/// 该组件；无过滤会把 checkbox 行等带 Interaction 的节点也刷成 accent 色。
pub fn button_state_system(
  mut commands: Commands,
  theme: Option<Res<UiTheme>>,
  mut q: Query<
    (
      Entity,
      &Interaction,
      &mut InteractionPrev,
      &mut BackgroundColor,
    ),
    With<Button>,
  >,
) {
  let Some(theme) = theme else { return };
  let c = &theme.colors;
  for (e, inter, mut prev, mut bg) in &mut q {
    // click = 在按钮上按下并释放（拖出后释放视为取消）
    if prev.0 == Interaction::Pressed && *inter == Interaction::Hovered {
      commands.trigger(UiClick { entity: e });
    }
    prev.0 = *inter;
    let target = match inter {
      Interaction::Pressed => color_of(&c.accent_pressed),
      Interaction::Hovered => color_of(&c.accent_hover),
      Interaction::None => color_of(&c.accent),
    };
    if bg.0 != target {
      bg.0 = target;
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::theme::default_theme;

  #[test]
  fn button_spawn_structure() {
    let theme = default_theme();
    let ctx = UiCtx::new(&theme, None);
    let mut app = App::new();
    let root = app.world_mut().spawn_empty().id();
    let mut child = None;
    app.world_mut().entity_mut(root).with_children(|p| {
      child = Some(button(&ctx, p, "OK"));
    });
    let e = child.expect("button spawned");
    let w = app.world();
    assert!(w.get::<Button>(e).is_some());
    assert!(w.get::<Interaction>(e).is_some());
    assert!(w.get::<InteractionPrev>(e).is_some());
    // 文本子标签
    let children = w.get::<Children>(e).expect("button has children");
    assert_eq!(children.len(), 1, "button contains one label");
  }

  #[test]
  fn button_click_on_release_and_hover_visual() {
    use std::sync::{Arc, Mutex};

    let theme = default_theme();
    let mut app = App::new();
    app.insert_resource(theme.clone());
    app.add_systems(Update, button_state_system);
    let clicks = Arc::new(Mutex::new(Vec::<Entity>::new()));
    let clicks_obs = clicks.clone();
    app.add_observer(move |click: On<UiClick>| {
      clicks_obs.lock().unwrap().push(click.entity);
    });

    let btn = app
      .world_mut()
      .spawn((
        Button,
        Interaction::None,
        InteractionPrev::default(),
        BackgroundColor::default(),
      ))
      .id();
    let bg_of = |app: &App| app.world().get::<BackgroundColor>(btn).unwrap().0;

    // hover：视觉切 accent_hover，无 click
    app
      .world_mut()
      .get_mut::<Interaction>(btn)
      .unwrap()
      .set_if_neq(Interaction::Hovered);
    app.update();
    assert!(clicks.lock().unwrap().is_empty());
    assert_eq!(bg_of(&app), color_of(&theme.colors.accent_hover));

    // 按下：视觉切 accent_pressed，无 click
    app
      .world_mut()
      .get_mut::<Interaction>(btn)
      .unwrap()
      .set_if_neq(Interaction::Pressed);
    app.update();
    assert!(clicks.lock().unwrap().is_empty());
    assert_eq!(bg_of(&app), color_of(&theme.colors.accent_pressed));

    // 释放（仍在按钮上）：触发 click，视觉回 accent_hover
    app
      .world_mut()
      .get_mut::<Interaction>(btn)
      .unwrap()
      .set_if_neq(Interaction::Hovered);
    app.update();
    assert_eq!(
      *clicks.lock().unwrap(),
      vec![btn],
      "release inside triggers UiClick"
    );

    // 按下后拖出释放：视为取消，不触发 click
    app
      .world_mut()
      .get_mut::<Interaction>(btn)
      .unwrap()
      .set_if_neq(Interaction::Pressed);
    app.update();
    app
      .world_mut()
      .get_mut::<Interaction>(btn)
      .unwrap()
      .set_if_neq(Interaction::None);
    app.update();
    assert_eq!(
      clicks.lock().unwrap().len(),
      1,
      "release outside does not click"
    );
  }
}
