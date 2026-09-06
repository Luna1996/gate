//! scroll_view：可滚动视图（overflow 裁剪 + 鼠标滚轮纵向滚动）。
//!
//! bevy_ui 0.19 无内置滚动容器，自实现：
//! - viewport：`Overflow::clip()` 裁剪超出子节点，固定高度
//! - content：绝对定位，`top = -scroll_y` 实现纵向偏移
//! - 滚动量钳制在 `[-(content_h - viewport_h), 0]`
//!
//! 只响应光标悬停（Interaction::Hovered/Pressed）的 viewport，避免多个
//! scroll view 同时滚动。
//!
//! bevy 0.19 中 MouseWheel 是 `Message`（非 Event），用 `MessageReader` 读取。

use std::ops::Deref;

use bevy::ecs::message::MessageReader;
use bevy::input::mouse::{MouseScrollUnit, MouseWheel};
use bevy::prelude::*;
use bevy::ui::{ComputedNode, FocusPolicy, Interaction, Overflow};

use super::{UiCtx, color_of, px};

/// 滚动视图状态（挂在 viewport 节点上）
#[derive(Component, Debug, Default)]
pub struct ScrollView {
  /// 当前纵向滚动偏移（0 = 顶部；负值 = 向下滚）
  pub scroll_y: f32,
}

/// viewport 节点标记
#[derive(Component, Debug, Default)]
pub struct ScrollViewport;

/// content 节点标记（调用方在此实体上添加子节点）
#[derive(Component, Debug, Default)]
pub struct ScrollContent;

/// 每格滚轮滚动像素数
const SCROLL_SPEED: f32 = 24.0;

/// 滚动视图句柄：viewport 根实体 + content 实体（调用方在 content 上添加子节点）。
/// Deref 到 viewport 实体。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ScrollViewHandle {
  pub entity: Entity,
  pub content: Entity,
}

impl Deref for ScrollViewHandle {
  type Target = Entity;
  fn deref(&self) -> &Entity {
    &self.entity
  }
}

/// 滚动视图配置（`height`：viewport 高度，Val；Default = Val::Auto）
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScrollConfig {
  pub height: Val,
}

impl Default for ScrollConfig {
  fn default() -> Self {
    Self { height: Val::Auto }
  }
}

/// 创建可滚动视图。
///
/// `height`：viewport 高度（Val，可传 px/vh/percent）。
pub fn scroll_view(ctx: &UiCtx, parent: &mut ChildSpawner, config: ScrollConfig) -> ScrollViewHandle {
  let c = &ctx.theme.colors;
  let m = &ctx.theme.metrics;
  let mut content_e = None;
  let vp_e = parent
    .spawn((
      Name::new("ui-scroll-view"),
      ScrollView::default(),
      ScrollViewport,
      Interaction::default(),
      Node {
        width: Val::Percent(100.0),
        height: config.height,
        overflow: Overflow::clip(),
        border: UiRect::all(px(m.border_width)),
        ..default()
      },
      BackgroundColor(Color::NONE),
      BorderColor::all(color_of(&c.border)),
      FocusPolicy::Block,
    ))
    .with_children(|vp| {
      content_e = Some(
        vp
          .spawn((
            Name::new("ui-scroll-content"),
            ScrollContent,
            Node {
              position_type: PositionType::Absolute,
              top: Val::Px(0.0),
              left: Val::Px(0.0),
              right: Val::Px(0.0),
              flex_direction: FlexDirection::Column,
              row_gap: px(m.spacing.xs),
              padding: UiRect::all(px(m.spacing.sm)),
              ..default()
            },
          ))
          .id(),
      );
    })
    .id();
  ScrollViewHandle {
    entity: vp_e,
    content: content_e.expect("scroll content spawned"),
  }
}

/// 滚轮滚动系统：累加本帧 MouseWheel.y，更新悬停中 viewport 的 scroll_y
pub fn scroll_view_system(
  mut scroll_reader: MessageReader<MouseWheel>,
  mut q_vp: Query<
    (&mut ScrollView, &Interaction, &Children, &ComputedNode),
    With<ScrollViewport>,
  >,
  mut q_content: Query<(&mut Node, &ComputedNode), With<ScrollContent>>,
) {
  // 滚轮 y：正值 = 向上滚（winit/Windows 传统鼠标滚轮正向），内容应下移看上面的内容
  // → scroll_y 增大（趋向 0）。Pixel 单位（触控板）按典型行高 16px 折算成行，
  // 与 gate-app 相机缩放口径一致。
  let mut lines = 0.0;
  for ev in scroll_reader.read() {
    match ev.unit {
      MouseScrollUnit::Line => lines += ev.y,
      MouseScrollUnit::Pixel => lines += ev.y / 16.0,
    }
  }
  if lines == 0.0 {
    return;
  }
  let delta = lines * SCROLL_SPEED; // 滚轮上 → scroll_y 增大（内容下移）
  for (mut sv, inter, children, vp_computed) in &mut q_vp {
    // 只滚动光标悬停中的 viewport
    if *inter == Interaction::None {
      continue;
    }
    let vp_h = vp_computed.size.y;
    if vp_h <= 0.0 {
      continue;
    }
    for child in children.iter() {
      let Ok((mut content_node, content_computed)) = q_content.get_mut(child) else {
        continue;
      };
      let content_h = content_computed.size.y;
      sv.scroll_y += delta;
      let max_scroll = (content_h - vp_h).max(0.0);
      sv.scroll_y = sv.scroll_y.clamp(-max_scroll, 0.0);
      content_node.top = Val::Px(sv.scroll_y);
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::theme::default_theme;
  use bevy::ecs::message::Messages;
  use bevy::input::mouse::MouseScrollUnit;
  use bevy::input::touch::TouchPhase;

  #[test]
  fn scroll_view_spawn_structure() {
    let theme = default_theme();
    let ctx = UiCtx::new(&theme, None);
    let mut app = App::new();
    let root = app.world_mut().spawn_empty().id();
    let mut handle = None;
    app.world_mut().entity_mut(root).with_children(|p| {
      handle = Some(scroll_view(&ctx, p, ScrollConfig { height: px(100.0) }));
    });
    let h = handle.expect("scroll view spawned");
    let w = app.world();
    // content 有 ScrollContent 标记
    assert!(w.get::<ScrollContent>(h.content).is_some());
    // viewport = root 的第一个子节点（scroll_view 在 root 下创建 viewport）
    let vp = w.get::<Children>(root).unwrap()[0];
    assert_eq!(h.entity, vp, "handle.entity is the viewport");
    assert!(w.get::<ScrollView>(vp).is_some());
    assert!(w.get::<ScrollViewport>(vp).is_some());
    assert!(
      w.get::<Interaction>(vp).is_some(),
      "viewport has Interaction for hover detection"
    );
  }

  #[test]
  fn scroll_clamped_to_content_range() {
    let mut app = App::new();
    app.init_resource::<Messages<MouseWheel>>();
    app.add_systems(Update, scroll_view_system);

    let content = app
      .world_mut()
      .spawn((
        ScrollContent,
        Node {
          position_type: PositionType::Absolute,
          ..default()
        },
        ComputedNode {
          size: Vec2::new(0.0, 300.0), // content 高 300
          ..default()
        },
      ))
      .id();
    let vp = app
      .world_mut()
      .spawn((
        ScrollView::default(),
        ScrollViewport,
        Interaction::Hovered,
        Node {
          height: Val::Px(100.0), // viewport 高 100
          ..default()
        },
        ComputedNode {
          size: Vec2::new(0.0, 100.0),
          ..default()
        },
      ))
      .id();
    app.world_mut().entity_mut(vp).add_child(content);

    // 模拟向下滚一大段（dy 负 → delta 负 → scroll_y 减小，钳到 -max_scroll）
    app
      .world_mut()
      .resource_mut::<Messages<MouseWheel>>()
      .write(MouseWheel {
        unit: MouseScrollUnit::Line,
        x: 0.0,
        y: -100.0,
        phase: TouchPhase::Moved,
        window: Entity::PLACEHOLDER,
      });
    app.update();

    let sv = app.world().get::<ScrollView>(vp).unwrap();
    // content 300 - viewport 100 = 200 可滚；向下滚到底 → scroll_y ≈ -200
    assert!(
      (sv.scroll_y - (-200.0)).abs() < 1.0,
      "scroll down clamps to -200, got {}",
      sv.scroll_y
    );

    // content top 同步
    let content_node = app.world().get::<Node>(content).unwrap();
    assert_eq!(content_node.top, Val::Px(sv.scroll_y));

    // 再向上滚一大段（dy 正 → scroll_y 增大，钳回 0）
    app
      .world_mut()
      .resource_mut::<Messages<MouseWheel>>()
      .write(MouseWheel {
        unit: MouseScrollUnit::Line,
        x: 0.0,
        y: 100.0,
        phase: TouchPhase::Moved,
        window: Entity::PLACEHOLDER,
      });
    app.update();
    let sv = app.world().get::<ScrollView>(vp).unwrap();
    assert!(
      sv.scroll_y.abs() < 1.0,
      "scroll up clamps back to 0, got {}",
      sv.scroll_y
    );
  }
}
