use std::ops::Deref;

use bevy::picking::hover::Hovered;
use bevy::prelude::*;
use bevy::ui::Pressed;

use super::{UiCtx, UiDisabled, color_of, dim_color, px, spawn_label};
use crate::pointer::{UiInteract, UiInteractBundle, UiInteractPrev};
use crate::theme::UiTheme;

#[derive(Component, Debug)]
pub struct TabView {
  pub active: usize,
  pub count: usize,
  pub fit_content: bool,
}

#[derive(Component, Debug, Clone, Copy)]
pub struct TabButton {
  pub index: usize,
}

#[derive(Component, Debug, Clone, Copy)]
pub struct TabContent {
  pub index: usize,
}

#[derive(EntityEvent, Clone, Copy, Debug, PartialEq)]
pub struct TabChanged {
  pub entity: Entity,
  pub index: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TabViewHandle {
  pub entity: Entity,
  pub contents: Vec<Entity>,
}

impl Deref for TabViewHandle {
  type Target = Entity;
  fn deref(&self) -> &Entity {
    &self.entity
  }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TabConfig {
  pub tabs: Vec<String>,
  pub active: usize,
  pub fit_content: bool,
  pub disabled_tabs: Vec<usize>,
}

pub fn tab_view(ctx: &UiCtx, parent: &mut ChildSpawner, config: TabConfig) -> TabViewHandle {
  let c = &ctx.theme.colors;
  let m = &ctx.theme.metrics;
  let count = config.tabs.len();
  let active = config.active.min(count.saturating_sub(1));
  let mut content_entities = Vec::with_capacity(count);

  let root = parent
    .spawn((
      Name::new("ui-tab-view"),
      TabView { active, count, fit_content: config.fit_content },
      Node {
        flex_direction: FlexDirection::Column,
        width: Val::Percent(100.0),
        flex_grow: if config.fit_content { 0.0 } else { 1.0 },
        ..default()
      },
    ))
    .with_children(|root| {
      root
        .spawn((
          Name::new("ui-tab-bar"),
          Node {
            flex_direction: FlexDirection::Row,
            width: Val::Percent(100.0),
            border: UiRect::bottom(px(m.border_width)),
            ..default()
          },
          BorderColor::all(color_of(&c.border)),
        ))
        .with_children(|bar| {
          for (i, name) in config.tabs.iter().enumerate() {
            let is_active = i == active;
            let is_disabled = config.disabled_tabs.contains(&i);
            let mut tab_ec = bar.spawn((
              Name::new(format!("ui-tab-{i}")),
              TabButton { index: i },
              UiInteractBundle::default(),
              Node {
                padding: UiRect {
                  left: px(m.spacing.md),
                  right: px(m.spacing.md),
                  top: px(m.spacing.sm),
                  bottom: px(m.spacing.sm),
                },
                border: UiRect::bottom(px(if is_active { 2.0 } else { 0.0 })),
                ..default()
              },
              BackgroundColor(if is_active { color_of(&c.surface_elevated) } else { Color::NONE }),
              BorderColor::all(if is_active { color_of(&c.text_primary) } else { Color::NONE }),
            ));
            if is_disabled {
              tab_ec.insert(UiDisabled);
            }
            tab_ec.with_children(|tab| {
              spawn_label(
                ctx,
                tab,
                name.clone(),
                m.font_size.sm,
                if is_active {
                  color_of(&c.text_primary)
                } else if is_disabled {
                  dim_color(color_of(&c.text_muted))
                } else {
                  color_of(&c.text_muted)
                },
              );
            });
          }
        });
      root
        .spawn((
          Name::new("ui-tab-content-container"),
          Node {
            flex_direction: FlexDirection::Column,
            width: Val::Percent(100.0),
            flex_grow: if config.fit_content { 0.0 } else { 1.0 },
            overflow: Overflow::clip(),
            ..default()
          },
        ))
        .with_children(|container| {
          for i in 0..count {
            let is_active = i == active;
            let in_flow = config.fit_content && is_active;
            let e = container
              .spawn((
                Name::new(format!("ui-tab-content-{i}")),
                TabContent { index: i },
                Node {
                  position_type: if in_flow {
                    PositionType::Relative
                  } else {
                    PositionType::Absolute
                  },
                  top: px(0.0),
                  left: px(0.0),
                  width: Val::Percent(100.0),
                  height: if config.fit_content { Val::Auto } else { Val::Percent(100.0) },
                  flex_direction: FlexDirection::Column,
                  ..default()
                },
                if is_active { Visibility::Inherited } else { Visibility::Hidden },
              ))
              .id();
            content_entities.push(e);
          }
        });
    })
    .id();

  TabViewHandle { entity: root, contents: content_entities }
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn tab_view_system(
  mut commands: Commands,
  mut q_views: Query<(Entity, &mut TabView, &Children)>,
  mut q_tab: Query<(&TabButton, &Hovered, Has<Pressed>, &mut UiInteractPrev, Has<UiDisabled>)>,
  mut q_tab_node: Query<(&mut BackgroundColor, &mut BorderColor, &Children, Has<UiDisabled>)>,
  mut q_tab_text: Query<&mut TextColor>,
  q_children: Query<&Children>,
  mut q_content: Query<(&TabContent, &mut Visibility, &mut Node)>,
  theme: Option<Res<UiTheme>>,
) {
  let Some(theme) = theme else {
    return;
  };
  let c = &theme.colors;

  for (view_e, mut view, view_children) in &mut q_views {
    let Some(&bar) = view_children.first() else {
      continue;
    };
    let Some(&content_container) = view_children.get(1) else {
      continue;
    };
    let Ok(bar_children) = q_children.get(bar) else {
      continue;
    };

    for tab_e in bar_children.iter() {
      let Ok((tab_btn, hovered, pressed, mut prev, disabled)) = q_tab.get_mut(tab_e) else {
        continue;
      };
      let inter = UiInteract::of(hovered, pressed);
      if !disabled
        && prev.0 == UiInteract::Pressed
        && inter == UiInteract::Hovered
        && view.active != tab_btn.index
      {
        view.active = tab_btn.index;
        commands.trigger(TabChanged { entity: view_e, index: tab_btn.index });
      }
      prev.0 = inter;
    }

    for tab_e in bar_children.iter() {
      let Ok((tab_btn, hovered, pressed, _, _)) = q_tab.get(tab_e) else {
        continue;
      };
      let inter = UiInteract::of(hovered, pressed);
      let is_active = tab_btn.index == view.active;
      let Ok((mut bg, mut bc, tab_children, disabled)) = q_tab_node.get_mut(tab_e) else {
        continue;
      };
      let hovered = !disabled && inter.is_active();
      let target_bg = if is_active {
        color_of(&c.surface_elevated)
      } else if hovered {
        color_of(&c.surface_card)
      } else {
        Color::NONE
      };
      let target_bg =
        if disabled && target_bg != Color::NONE { dim_color(target_bg) } else { target_bg };
      bg.0 = target_bg;
      let target_border = if is_active { color_of(&c.text_primary) } else { Color::NONE };
      let target_border = if disabled && target_border != Color::NONE {
        dim_color(target_border)
      } else {
        target_border
      };
      *bc = BorderColor::all(target_border);
      let text_color = if is_active {
        color_of(&c.text_primary)
      } else if hovered {
        color_of(&c.text_body)
      } else {
        color_of(&c.text_muted)
      };
      let text_color = if disabled { dim_color(text_color) } else { text_color };
      for child in tab_children.iter() {
        if let Ok(mut tc) = q_tab_text.get_mut(child) {
          tc.0 = text_color;
        }
      }
    }

    let Ok(cc_children) = q_children.get(content_container) else {
      continue;
    };
    for content_e in cc_children.iter() {
      if let Ok((tc, mut vis, mut node)) = q_content.get_mut(content_e) {
        let is_active = tc.index == view.active;
        *vis = if is_active { Visibility::Inherited } else { Visibility::Hidden };
        if view.fit_content {
          node.position_type =
            if is_active { PositionType::Relative } else { PositionType::Absolute };
        }
      }
    }
  }
}
