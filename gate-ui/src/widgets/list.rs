use std::collections::VecDeque;
use std::ops::Deref;

use bevy::prelude::*;

use super::{UiCtx, color_of, px, spawn_label_cmd};
use crate::theme::{ThemeFont, UiTheme};

#[derive(Component, Clone, Debug)]
pub struct RingList {
  capacity: usize,
  items: VecDeque<String>,
}

impl RingList {
  pub fn new(capacity: usize) -> Self {
    Self { capacity, items: VecDeque::new() }
  }

  pub fn push(&mut self, text: impl Into<String>) -> Option<String> {
    self.items.push_back(text.into());
    if self.items.len() > self.capacity { self.items.pop_front() } else { None }
  }

  pub fn items(&self) -> impl Iterator<Item = &str> {
    self.items.iter().map(|s| s.as_str())
  }

  pub fn capacity(&self) -> usize {
    self.capacity
  }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ListHandle(pub Entity);

impl Deref for ListHandle {
  type Target = Entity;
  fn deref(&self) -> &Entity {
    &self.0
  }
}

impl From<ListHandle> for Entity {
  fn from(h: ListHandle) -> Entity {
    h.0
  }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ListConfig {
  pub capacity: usize,
}

impl Default for ListConfig {
  fn default() -> Self {
    Self { capacity: 8 }
  }
}

pub fn list(ctx: &UiCtx, parent: &mut ChildSpawner, config: ListConfig) -> ListHandle {
  let m = &ctx.theme.metrics;
  let e = parent
    .spawn((
      Name::new("ui-list"),
      RingList::new(config.capacity),
      Node { flex_direction: FlexDirection::Column, row_gap: px(m.spacing.xs), ..default() },
    ))
    .id();
  ListHandle(e)
}

pub fn ring_list_sync_system(
  mut commands: Commands,
  theme: Option<Res<UiTheme>>,
  theme_font: Option<Res<ThemeFont>>,
  mut q: Query<(Entity, &RingList), Changed<RingList>>,
) {
  let (Some(theme), Some(theme_font)) = (theme, theme_font) else {
    return;
  };
  let ctx = UiCtx::new(&theme, theme_font.handle.as_ref());
  let fs = &theme.metrics.font_size;
  let color = color_of(&theme.colors.text_body);
  for (e, ring) in &mut q {
    commands.entity(e).despawn_related::<Children>();
    for item in ring.items() {
      spawn_label_cmd(&ctx, &mut commands, e, item.to_string(), fs.sm, color);
    }
  }
}
