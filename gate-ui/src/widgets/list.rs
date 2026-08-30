//! list：固定容量环形列表（新条目顶替最旧条目）。
//!
//! v0 不做滚动裁剪（OQ-1 结论随 review 记录）：容量固定 + 同步重建子标签。
//! 逻辑（环形顶替）纯数据可 headless 单测；渲染同步用 change 检测只在脏时重建。

use std::collections::VecDeque;

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
    Self {
      capacity,
      items: VecDeque::new(),
    }
  }

  /// 压入条目；超出容量时顶替最旧条目，返回被顶替者
  pub fn push(&mut self, text: impl Into<String>) -> Option<String> {
    self.items.push_back(text.into());
    if self.items.len() > self.capacity {
      self.items.pop_front()
    } else {
      None
    }
  }

  pub fn items(&self) -> impl Iterator<Item = &str> {
    self.items.iter().map(|s| s.as_str())
  }

  pub fn capacity(&self) -> usize {
    self.capacity
  }
}

/// 环形列表容器（纵向排列）
pub fn list(ctx: &UiCtx, parent: &mut ChildSpawner, capacity: usize) -> Entity {
  let m = &ctx.theme.metrics;
  parent
    .spawn((
      Name::new("ui-list"),
      RingList::new(capacity),
      Node {
        flex_direction: FlexDirection::Column,
        row_gap: px(m.spacing.xs),
        ..default()
      },
    ))
    .id()
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
  let color = color_of(&theme.colors.text);
  for (e, ring) in &mut q {
    commands.entity(e).despawn_related::<Children>();
    for item in ring.items() {
      spawn_label_cmd(&ctx, &mut commands, e, item.to_string(), fs.sm, color);
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::theme::default_theme;

  #[test]
  fn ring_list_evicts_oldest() {
    let mut ring = RingList::new(3);
    assert_eq!(ring.push("a"), None);
    assert_eq!(ring.push("b"), None);
    assert_eq!(ring.push("c"), None);
    let items: Vec<_> = ring.items().collect();
    assert_eq!(items, ["a", "b", "c"]);

    assert_eq!(ring.push("d"), Some("a".to_string()), "oldest evicted");
    let items: Vec<_> = ring.items().collect();
    assert_eq!(items, ["b", "c", "d"]);
    assert_eq!(ring.capacity(), 3);
  }

  #[test]
  fn ring_list_sync_rebuilds_children() {
    let theme = default_theme();
    let mut app = App::new();
    app.insert_resource(theme);
    app.insert_resource(ThemeFont::default());
    app.add_systems(Update, ring_list_sync_system);

    let e = app.world_mut().spawn(RingList::new(2)).id();
    // 空列表首帧同步：无子
    app.update();
    assert!(app.world().get::<Children>(e).is_none());

    // push 两条 → 同步出两个子标签
    {
      let mut ring = app.world_mut().get_mut::<RingList>(e).unwrap();
      ring.push("one");
      ring.push("two");
    }
    app.update();
    let children = app.world().get::<Children>(e).expect("children synced");
    assert_eq!(children.len(), 2);
    let texts: Vec<String> = children
      .iter()
      .map(|c| app.world().get::<Text>(c).unwrap().0.clone())
      .collect();
    assert_eq!(texts, ["one", "two"]);

    // 顶替后重建：只保留最新两条
    {
      let mut ring = app.world_mut().get_mut::<RingList>(e).unwrap();
      ring.push("three");
    }
    app.update();
    let children = app.world().get::<Children>(e).unwrap();
    assert_eq!(children.len(), 2, "capacity bounded");
    let texts: Vec<String> = children
      .iter()
      .map(|c| app.world().get::<Text>(c).unwrap().0.clone())
      .collect();
    assert_eq!(texts, ["two", "three"]);
  }
}
