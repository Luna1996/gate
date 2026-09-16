//! tooltip：任意控件可挂的悬浮提示（鼠标停留超过延时后显示）。
//!
//! 用法：`world.entity_mut(*handle).insert(Tooltip::new("文案"))` —— 不改变控件自身结构。
//! 命中判定二选一：控件带 `Interaction` 时用其状态（沿用 bevy_ui 的 Block 链，被上层控件
//! 遮住时为 `None`）；否则用 `ComputedNode::contains_point` 直接命中测试（label/panel 等
//! 无 Interaction 的节点也可挂）。
//!
//! 提示框是全局唯一实体（首次需要时懒创建），脱离布局——绝对定位：位置在**首次展示时钉住**，
//! 鼠标继续在同一个锚点控件内移动时不跟随（换锚点或离开后重新悬浮才重新取点）。

use bevy::prelude::*;
use bevy::ui::{ComputedNode, FocusPolicy, Interaction, UiGlobalTransform};
use bevy::window::PrimaryWindow;

use super::{color_of, px};
use crate::theme::UiTheme;

/// 悬浮多久后展示（秒）
pub const TOOLTIP_DELAY: f32 = 0.5;
/// 提示框相对光标的偏移（逻辑 px；y 为正 = 在光标下方）
pub const TOOLTIP_OFFSET: Vec2 = Vec2::new(14.0, 18.0);
/// 提示框与系统窗口边缘的最小留白（逻辑 px）
pub const TOOLTIP_MARGIN: f32 = 8.0;
/// 提示框最大宽度（逻辑 px，超出换行）
pub const TOOLTIP_MAX_W: f32 = 260.0;
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

/// 提示框文本标记
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
  /// 已钉住的展示锚点与位置（逻辑 px，**未做边界收束**）：
  /// 同一锚点内移动鼠标不改位置；换锚点 / 离开后重新悬浮才重新取点
  pin: Option<(Entity, Vec2)>,
}

/// 悬浮判定 + 延时展示。每帧最多展示一个提示（命中节点中面积最小者 = 纵深最内层）。
#[allow(clippy::type_complexity)] // Bevy system：多组件查询签名固有
pub fn tooltip_system(
  mut commands: Commands,
  theme: Option<Res<UiTheme>>,
  windows: Query<&Window, With<PrimaryWindow>>,
  mut layer_res: ResMut<TooltipLayerEntity>,
  time: Res<Time>,
  mut state: Local<TooltipHoverState>,
  mut layer_q: Query<(&mut Node, Option<&ComputedNode>), With<TooltipLayer>>,
  mut text_q: Query<&mut Text, With<TooltipText>>,
  owners: Query<(
    Entity,
    &Tooltip,
    Option<&Interaction>,
    &ComputedNode,
    &UiGlobalTransform,
    Option<&InheritedVisibility>,
  )>,
) {
  let Some(theme) = theme else { return };
  let Ok(window) = windows.single() else { return };
  let physical = window.physical_cursor_position();

  // ---- 命中判定：面积最小的命中节点即最内层 ----
  let mut best: Option<(Entity, &Tooltip)> = None;
  let mut best_area = f32::MAX;
  if let Some(cursor) = physical {
    for (e, tip, inter, node, transform, vis) in &owners {
      if !vis.is_some_and(|v| v.get()) || node.size() == Vec2::ZERO {
        continue;
      }
      let hovered = match inter {
        Some(i) => *i != Interaction::None,
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
  let text = tip.text.clone();
  // 首次需要时创建提示框实体（本帧查不到，下一帧起可写入文本与位置）
  ensure_layer(&mut commands, &mut layer_res, &theme);
  if let Ok(mut t) = text_q.single_mut()
    && t.0 != text
  {
    t.0 = text;
  }
  // 逻辑坐标（Node.left/top 为逻辑 px）：光标物理 → 逻辑
  let sf = window.scale_factor().max(f32::EPSILON);
  let cursor = physical.unwrap_or_default() / sf;
  // 展示位置钉在**首次展示**时的「光标 + 偏移」上：同一个锚点控件内继续移动鼠标时提示框不动，
  // 只有换锚点（或离开后重新悬浮）才重新取点
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
  // 边界收束每帧都做：钉住的是**未收束**的锚点，而提示框尺寸要等它布局一帧才量得到，
  // 所以贴右/下边的收束是在后续帧逐步收敛的（不能把收束后的值写回 pin，否则会越收越偏）
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
      FocusPolicy::Pass,
      GlobalZIndex(TOOLTIP_Z),
    ))
    .with_children(|p| {
      p.spawn((
        Name::new("ui-tooltip-text"),
        TooltipText,
        Text::new(String::new()),
        TextFont { font_size: bevy::text::FontSize::Px(m.font_size.sm), ..default() },
        TextColor(color_of(&c.text_primary)),
      ));
    })
    .id();
  res.0 = Some(e);
  e
}
