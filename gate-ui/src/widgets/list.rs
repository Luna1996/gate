//! list：固定容量环形列表（新条目顶替最旧条目）。
//!
//! 不做滚动裁剪：容量固定 + 同步重建子标签；渲染同步用 change 检测，只在脏时重建。

use std::collections::VecDeque;
use std::ops::Deref;

use bevy::prelude::*;

use super::{UiCtx, color_of, px, spawn_label_cmd};
use crate::theme::{ThemeFont, UiTheme};

/// 固定容量环形列表组件
#[derive(Component, Clone, Debug)]
pub struct RingList {
  capacity: usize,
  items: VecDeque<String>,
}

impl RingList {
  pub fn new(capacity: usize) -> Self {
    Self { capacity, items: VecDeque::new() }
  }

  /// 压入条目；超出容量时顶替最旧条目，返回被顶替者
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

/// 环形列表句柄（Deref 到根实体 Entity）
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

/// 环形列表配置（Default = 容量 8）
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ListConfig {
  pub capacity: usize,
}

impl Default for ListConfig {
  fn default() -> Self {
    Self { capacity: 8 }
  }
}

/// 环形列表容器（纵向排列）
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

/// 脏时同步：RingList changed → 重建文本子实体（条目顺序 = 自上而下）
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
