//! scroll_view：可滚动视图（overflow 裁剪 + 鼠标滚轮纵向滚动）。
//!
//! bevy_ui 0.19 无内置滚动容器，自实现：
//! - viewport：`Overflow::clip()` 裁剪超出子节点，固定高度
//! - content：绝对定位，`top = -scroll_y` 实现纵向偏移
//! - 滚动量钳制在 `[-(content_h - viewport_h), 0]`
//!
//! 只响应光标悬停（Interaction::Hovered/Pressed）的 viewport。bevy 0.19 中 MouseWheel 是
//! `Message`（非 Event），用 `MessageReader` 读取。

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
pub fn scroll_view(
  ctx: &UiCtx,
  parent: &mut ChildSpawner,
  config: ScrollConfig,
) -> ScrollViewHandle {
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
        vp.spawn((
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
  ScrollViewHandle { entity: vp_e, content: content_e.expect("scroll content spawned") }
}

/// 滚轮滚动系统：累加本帧 MouseWheel.y，更新悬停中 viewport 的 scroll_y
pub fn scroll_view_system(
  mut scroll_reader: MessageReader<MouseWheel>,
  mut q_vp: Query<(&mut ScrollView, &Interaction, &Children, &ComputedNode), With<ScrollViewport>>,
  mut q_content: Query<(&mut Node, &ComputedNode), With<ScrollContent>>,
) {
  // 滚轮 y：正值 = 向上滚（winit/Windows 传统鼠标滚轮正向），scroll_y 增大（趋向 0）。
  // Pixel 单位（触控板）按典型行高 16px 折算成行。
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
