//! tooltip：任意控件可挂的悬浮提示（停留超时后显示）。用法：`entity_mut(*handle).insert(Tooltip::new("文案"))`。
//! 提示文案按 Markdown 渲染（见 `markdown`），内容变化才重建。
//! 命中判定：控件带 `Hovered` 用其状态（被上层遮挡时为 `false`），否则用 `ComputedNode::contains_point`。
//! 提示框全局唯一（懒创建）、绝对定位，位置在首次展示时钉住（同锚点内移动不跟随）。

use bevy::picking::Pickable;
use bevy::picking::hover::Hovered;
use bevy::prelude::*;
use bevy::ui::{ComputedNode, UiGlobalTransform};
use bevy::window::PrimaryWindow;

use super::markdown::{MarkdownConfig, MarkdownView, markdown_root, markdown_set_text};
use super::{color_of, px};
use crate::theme::UiTheme;
use crate::widgets::consts::{TOOLTIP_DELAY, TOOLTIP_MARGIN, TOOLTIP_MAX_W, TOOLTIP_OFFSET};

/// 提示框层深（高于一切面板）
const TOOLTIP_Z: i32 = 1000;

/// 悬浮提示文案（挂在任意可命中节点上）
#[derive(Component, Clone, Debug, PartialEq, Eq)]
pub struct Tooltip {
  pub text: String,
}

impl Tooltip {
  pub fn new(text: impl Into<String>) -> Self {
    Self { text: text.into() }
  }
}

/// 提示框根标记（全局唯一，懒创建）
#[derive(Component, Debug)]
pub struct TooltipLayer;

/// 提示框文本视图标记（挂 Markdown 根实体）
#[derive(Component, Debug)]
pub struct TooltipText;

/// 提示框容器实体缓存（懒创建）
#[derive(Resource, Default)]
pub struct TooltipLayerEntity(pub Option<Entity>);

/// 当前悬浮的提示实体与已累计时长
#[derive(Default)]
pub struct TooltipHoverState {
  entity: Option<Entity>,
  elapsed: f32,
  /// 已钉住的展示锚点与位置（逻辑 px，未做边界收束）；同锚点内移动不改位置，换锚点才重新取点
  pin: Option<(Entity, Vec2)>,
}

/// 悬浮判定 + 延时展示。每帧最多展示一个提示（命中节点中面积最小者 = 纵深最内层）。
#[allow(clippy::type_complexity, clippy::too_many_arguments)] // Bevy system：各 Query/Res 逐一注入
pub fn tooltip_system(
  mut commands: Commands,
  theme: Option<Res<UiTheme>>,
  windows: Query<&Window, With<PrimaryWindow>>,
  mut layer_res: ResMut<TooltipLayerEntity>,
  time: Res<Time>,
  mut state: Local<TooltipHoverState>,
  mut layer_q: Query<(&mut Node, Option<&ComputedNode>), With<TooltipLayer>>,
  view_q: Query<(Entity, &MarkdownView), With<TooltipText>>,
  owners: Query<(
    Entity,
    &Tooltip,
    Option<&Hovered>,
    &ComputedNode,
    &UiGlobalTransform,
    Option<&InheritedVisibility>,
  )>,
) {
  let Some(theme) = theme else { return };
  let Ok(window) = windows.single() else { return };
  let physical = window.physical_cursor_position();

  let mut best: Option<(Entity, &Tooltip)> = None;
  let mut best_area = f32::MAX;
  if let Some(cursor) = physical {
    for (e, tip, hover, node, transform, vis) in &owners {
      if !vis.is_some_and(|v| v.get()) || node.size() == Vec2::ZERO {
        continue;
      }
      let hovered = match hover {
        Some(h) => h.get(),
        None => node.contains_point(*transform, cursor),
      };
      if !hovered {
        continue;
      }
      let area = node.size().x * node.size().y;
      if area < best_area {
        best_area = area;
        best = Some((e, tip));
      }
    }
  }

  let Some((e, tip)) = best else {
    state.entity = None;
    state.elapsed = 0.0;
    state.pin = None;
    if let Ok((mut node, _)) = layer_q.single_mut() {
      node.display = Display::None;
    }
    return;
  };
  if state.entity != Some(e) {
    state.entity = Some(e);
    state.elapsed = 0.0;
  }
  state.elapsed += time.delta_secs();
  if state.elapsed < TOOLTIP_DELAY {
    return;
  }
  let text = tip.text.as_str();
  // 首次需要时创建（本帧查不到，下一帧起可写内容与位置）
  ensure_layer(&mut commands, &mut layer_res, &theme);
  // 文案变化才重建（重建走命令队列，新内容下一帧才参与布局）
  if let Ok((e, view)) = view_q.single()
    && view.source() != text
  {
    markdown_set_text(&mut commands, e, text);
  }
  // 光标物理坐标 → 逻辑（Node.left/top 为逻辑 px）
  let sf = window.scale_factor().max(f32::EPSILON);
  let cursor = physical.unwrap_or_default() / sf;
  let anchor = match state.pin {
    Some((pinned, p)) if pinned == e => p,
    _ => {
      let p = cursor + TOOLTIP_OFFSET;
      state.pin = Some((e, p));
      p
    }
  };
  let (w, h) = (window.width(), window.height());
  let size =
    layer_q.single_mut().ok().and_then(|(_, c)| c.map(|n| n.size() / sf)).unwrap_or_default();
  // 边界收束每帧都做：钉住的是未收束锚点，提示框尺寸要等布局一帧才量得到（勿把收束值写回 pin）
  let max_x = (w - size.x - TOOLTIP_MARGIN).max(TOOLTIP_MARGIN);
  let max_y = (h - size.y - TOOLTIP_MARGIN).max(TOOLTIP_MARGIN);
  if let Ok((mut node, _)) = layer_q.single_mut() {
    node.display = Display::Flex;
    node.left = px(anchor.x.min(max_x));
    node.top = px(anchor.y.min(max_y));
  }
}

/// 提示框懒创建（已存在且未被销毁则复用）
fn ensure_layer(commands: &mut Commands, res: &mut TooltipLayerEntity, theme: &UiTheme) -> Entity {
  if let Some(e) = res.0
    && commands.get_entity(e).is_ok()
  {
    return e;
  }
  let c = &theme.colors;
  let m = &theme.metrics;
  let e = commands
    .spawn((
      Name::new("ui-tooltip"),
      TooltipLayer,
      Node {
        position_type: PositionType::Absolute,
        display: Display::None,
        max_width: px(TOOLTIP_MAX_W),
        padding: UiRect {
          left: px(m.spacing.sm),
          right: px(m.spacing.sm),
          top: px(m.spacing.xs),
          bottom: px(m.spacing.xs),
        },
        border: UiRect::all(px(m.border_width)),
        ..default()
      },
      BackgroundColor(color_of(&c.surface_overlay)),
      BorderColor::all(color_of(&c.border_strong)),
      // 提示框自身不参与命中：不吃 hover、不挡下层
      Pickable::IGNORE,
      GlobalZIndex(TOOLTIP_Z),
    ))
    .with_children(|p| {
      // 内容为空，文案由 `markdown_set_text` 在首次展示时铺进去
      p.spawn((
        Name::new("ui-tooltip-text"),
        TooltipText,
        markdown_root(theme, &MarkdownConfig { text: String::new(), dense: true, max_width: None }),
      ));
    })
    .id();
  res.0 = Some(e);
  e
}
