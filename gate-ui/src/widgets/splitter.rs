//! splitter：元素间分割线（docs/ui-dark-theme.md 边框令牌，1px 硬边）。
//!
//! [`splitter`] 按父容器主轴方向自动定向：Column/ColumnReverse → 水平线（1px 高，横向铺满）；
//! Row/RowReverse（flex 默认）→ 垂直线（1px 宽，纵向铺满）。
//!
//! 交叉轴尺寸用 [`AlignSelf::Stretch`] 强制铺满：不依赖父容器显式高度，auto 高度的行/列里
//! 也能拉伸到兄弟元素等高（Val::Percent 在不定父级下会塌缩成 0）。间距由调用方容器的 gap
//! 控制，本组件不带 margin。

use bevy::prelude::*;

use super::{UiCtx, color_of, px};

/// 分割线节点标记
#[derive(Component, Debug, Default)]
pub struct Splitter;

/// 分割线：按父容器主轴方向自动决定横（Column 父）/竖（Row 父）
pub fn splitter(ctx: &UiCtx, parent: &mut ChildSpawner) -> Entity {
  let horizontal = parent
    .world()
    .get::<Node>(parent.target_entity())
    .is_none_or(|n| {
      matches!(
        n.flex_direction,
        FlexDirection::Column | FlexDirection::ColumnReverse
      )
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
      Node {
        width,
        height,
        align_self: AlignSelf::Stretch,
        flex_shrink: 0.0,
        ..default()
      },
      BackgroundColor(color_of(&c.border)),
    ))
    .id()
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::theme::default_theme;

  #[test]
  fn splitter_orients_by_parent_axis() {
    let theme = default_theme();
    let ctx = UiCtx::new(&theme, None);
    let mut app = App::new();
    let col_root = app
      .world_mut()
      .spawn(Node {
        flex_direction: FlexDirection::Column,
        ..default()
      })
      .id();
    let row_root = app
      .world_mut()
      .spawn(Node {
        flex_direction: FlexDirection::Row,
        ..default()
      })
      .id();
    // 父无 Node（未布局根）→ 回退为横线
    let bare_root = app.world_mut().spawn_empty().id();

    let mut in_col = None;
    let mut in_row = None;
    let mut in_bare = None;
    app.world_mut().entity_mut(col_root).with_children(|p| {
      in_col = Some(splitter(&ctx, p));
    });
    app.world_mut().entity_mut(row_root).with_children(|p| {
      in_row = Some(splitter(&ctx, p));
    });
    app.world_mut().entity_mut(bare_root).with_children(|p| {
      in_bare = Some(splitter(&ctx, p));
    });

    let w = app.world();
    let (in_col, in_row, in_bare) = (in_col.unwrap(), in_row.unwrap(), in_bare.unwrap());

    // Column 父 → 水平线：1px 高 + 交叉轴铺满
    let n = w.get::<Node>(in_col).unwrap();
    assert_eq!(n.height, px(1.0), "column parent yields horizontal line");
    assert_eq!(n.align_self, AlignSelf::Stretch);

    // Row 父 → 垂直线：1px 宽 + 交叉轴铺满
    let n = w.get::<Node>(in_row).unwrap();
    assert_eq!(n.width, px(1.0), "row parent yields vertical line");
    assert_eq!(n.align_self, AlignSelf::Stretch);

    // 父无 Node → 回退横线
    let n = w.get::<Node>(in_bare).unwrap();
    assert_eq!(n.height, px(1.0), "bare parent falls back to horizontal");

    // 全部用 border 令牌
    for e in [in_col, in_row, in_bare] {
      assert!(w.get::<Splitter>(e).is_some());
      assert_eq!(
        w.get::<BackgroundColor>(e).unwrap().0,
        color_of(&theme.colors.border),
        "uses border token"
      );
    }
  }
}
