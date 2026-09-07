//! capture：UI 指针门控（FR-6）。
//!
//! 复刻 bevy_ui 0.19 `ui_focus_system` 的命中判定：光标落在任意「可见 + 非零
//! 尺寸 + 祖先裁剪区内」的 UI 节点上即视为 UI 捕获（每帧重算）；显式
//! [`FocusPolicy::Pass`] 的节点不捕获（交互穿透到下层，与引擎 focus 链同语义）。
//!
//! 旧实现只统计带 `Interaction` 组件的节点是否 Hovered——但面板背景、容器、label
//! 这类节点没有 `Interaction`（引擎 focus.rs 对它们只做命中/阻挡，不写状态），
//! 悬停这些区域时 [`UiPointerCaptured`] 仍为 false，相机旋转/滚轮缩放/左键拾取
//! 会穿透到 3D 场景。现改为直接对 UI 节点树做命中测试，规则与引擎一致。
//!
//! 轨道相机等游戏输入 system 开头检查 [`UiPointerCaptured`]，捕获时跳过一切
//! 鼠标操作（按键/滚轮/移动）。

use bevy::prelude::*;
use bevy::ui::{
  ComputedNode, FocusPolicy, Node, OverrideClip, UiGlobalTransform, clip_check_recursive,
};
use bevy::window::PrimaryWindow;

/// 指针是否被 UI 捕获（每帧重算：光标在任一可见 UI 节点上即为 true）
#[derive(Resource, Default, Debug, Clone, Copy, PartialEq)]
pub struct UiPointerCaptured(pub bool);

/// UI 节点树命中测试（每帧重算；set_if_neq 仅在翻转时触发 change 检测）
///
/// 判定条件与 bevy_ui `ui_focus_system` / picking backend 逐条对齐：
/// - `InheritedVisibility` 为真（隐藏的节点不可交互；无该组件的节点跳过）
/// - `ComputedNode.size != 0`（Display::None 布局为零尺寸）
/// - `ComputedNode::contains_point`（节点变换后的物理像素矩形；圆角按实际半径）
/// - `clip_check_recursive`（光标被任一 `Overflow::clip()` 祖先裁掉则不算命中）
/// - `FocusPolicy::Pass` 的节点本身不捕获（下层 Block 节点仍可命中）
pub fn ui_pointer_capture_system(
  windows: Query<&Window, With<PrimaryWindow>>,
  nodes: Query<(
    Entity,
    &ComputedNode,
    &UiGlobalTransform,
    Option<&InheritedVisibility>,
    Option<&FocusPolicy>,
  )>,
  clipping: Query<(&ComputedNode, &UiGlobalTransform, &Node)>,
  child_of: Query<&ChildOf, Without<OverrideClip>>,
  mut captured: ResMut<UiPointerCaptured>,
) {
  let mut over_ui = false;
  if let Ok(window) = windows.single()
    && let Some(cursor) = window.physical_cursor_position()
  {
    for (entity, node, transform, vis, focus) in &nodes {
      if !vis.is_some_and(|v| v.get()) {
        continue;
      }
      if node.size() == Vec2::ZERO {
        continue;
      }
      // FocusPolicy::Pass = 显式让交互穿透（本组件库目前全部用 Block，分支保真）
      if focus == Some(&FocusPolicy::Pass) {
        continue;
      }
      if node.contains_point(*transform, cursor)
        && clip_check_recursive(cursor, entity, &clipping, &child_of)
      {
        over_ui = true;
        break;
      }
    }
  }
  captured.set_if_neq(UiPointerCaptured(over_ui));
}

#[cfg(test)]
mod tests {
  use super::*;
  use bevy::math::DVec2;

  /// 构造一个居中在 (x,y)、尺寸 (w,h) 的可见 UI 节点（无父子关系 = 根节点，
  /// clip_check_recursive 无子节点查询直接通过）
  fn spawn_node(app: &mut App, x: f32, y: f32, w: f32, h: f32) -> Entity {
    app
      .world_mut()
      .spawn((
        ComputedNode {
          size: Vec2::new(w, h),
          ..default()
        },
        UiGlobalTransform::from_translation(Vec2::new(x, y)),
        // 显式可见：裸 App 无可见性传播系统，InheritedVisibility::default() = hidden；
        // 真实 app 里由 VisibilityPropagate 按 Visibility 组件每帧计算
        InheritedVisibility::VISIBLE,
      ))
      .id()
  }

  fn set_cursor(app: &mut App, pos: Option<(f32, f32)>) {
    let mut windows = app
      .world_mut()
      .query_filtered::<&mut Window, With<PrimaryWindow>>();
    let mut window = windows.single_mut(app.world_mut()).unwrap();
    window.set_physical_cursor_position(pos.map(|(x, y)| DVec2::new(x as f64, y as f64)));
  }

  fn captured(app: &mut App) -> bool {
    app.update();
    app.world().resource::<UiPointerCaptured>().0
  }

  #[test]
  fn capture_flips_with_hit_test() {
    let mut app = App::new();
    app.init_resource::<UiPointerCaptured>();
    app.add_systems(Update, ui_pointer_capture_system);
    app.world_mut().spawn((Window::default(), PrimaryWindow));

    // 节点：中心 (200,200)，尺寸 100×100 → 覆盖 [150,250]²
    let node = spawn_node(&mut app, 200.0, 200.0, 100.0, 100.0);

    // 无光标位置（指针不在窗口）→ 不捕获
    set_cursor(&mut app, None);
    assert!(!captured(&mut app), "no cursor → not captured");

    // 光标在节点矩形内 → 捕获
    set_cursor(&mut app, Some((210.0, 190.0)));
    assert!(captured(&mut app), "cursor over node → captured");

    // 光标在节点矩形外 → 不捕获
    set_cursor(&mut app, Some((50.0, 50.0)));
    assert!(!captured(&mut app), "cursor off node → not captured");

    // 隐藏节点 → 不捕获（即使光标在其矩形内）
    set_cursor(&mut app, Some((210.0, 190.0)));
    app
      .world_mut()
      .entity_mut(node)
      .insert(InheritedVisibility::HIDDEN);
    assert!(!captured(&mut app), "hidden node → not captured");
    app
      .world_mut()
      .entity_mut(node)
      .insert(InheritedVisibility::VISIBLE);
    assert!(captured(&mut app), "visible again → captured");

    // FocusPolicy::Pass 的节点本身不捕获
    app.world_mut().entity_mut(node).insert(FocusPolicy::Pass);
    assert!(!captured(&mut app), "Pass node does not capture");
  }
}
