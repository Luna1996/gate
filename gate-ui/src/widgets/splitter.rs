//! splitter：元素间分割线（docs/ui-dark-theme.md 边框令牌，1px 硬边）。
//! 按父容器主轴定向：Column/ColumnReverse → 水平线（1px 高，横向铺满）；Row/RowReverse → 垂直线
//! （1px 宽，纵向铺满）。交叉轴用 `AlignSelf::Stretch` 铺满；间距由调用方 gap 控制，本组件无 margin。

use bevy::prelude::*;

use super::{UiCtx, color_of, px};

/// 分割线节点标记
#[derive(Component, Debug, Default)]
pub struct Splitter;

/// 分割线：按父容器主轴方向自动决定横（Column 父）/竖（Row 父）
pub fn splitter(ctx: &UiCtx, parent: &mut ChildSpawner) -> Entity {
  let horizontal = parent.world().get::<Node>(parent.target_entity()).is_none_or(|n| {
    matches!(n.flex_direction, FlexDirection::Column | FlexDirection::ColumnReverse)
  });
  let (width, height, name) = if horizontal {
    (Val::Auto, px(1.0), "ui-splitter-h")
  } else {
    (px(1.0), Val::Auto, "ui-splitter-v")
  };
  let c = &ctx.theme.colors;
  parent
    .spawn((
      Name::new(name),
      Splitter,
      Node { width, height, align_self: AlignSelf::Stretch, flex_shrink: 0.0, ..default() },
      BackgroundColor(color_of(&c.border)),
    ))
    .id()
}
