//! capture：UI 指针门控（FR-6）。
//!
//! 任一 `Interaction ∈ {Hovered, Pressed}` → `UiPointerCaptured=true`（每帧重算）；
//! 轨道相机等游戏输入 system 开头检查，captured 时跳过旋转/平移/缩放（滚轮一并吞），
//! 避免 UI 交互穿透到场景操作。

use bevy::prelude::*;
use bevy::ui::Interaction;

/// 指针是否被 UI 捕获（每帧重算，与任一控件 hover/pressed 同步）
#[derive(Resource, Default, Debug, Clone, Copy, PartialEq)]
pub struct UiPointerCaptured(pub bool);

/// Interaction 全集扫描（每帧重算；set_if_neq 仅在翻转时触发 change 检测）
pub fn ui_pointer_capture_system(
  interactions: Query<&Interaction>,
  mut captured: ResMut<UiPointerCaptured>,
) {
  let any = interactions.iter().any(|i| *i != Interaction::None);
  captured.set_if_neq(UiPointerCaptured(any));
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn capture_flips_with_interaction() {
    let mut app = App::new();
    app.init_resource::<UiPointerCaptured>();
    app.add_systems(Update, ui_pointer_capture_system);

    let a = app.world_mut().spawn(Interaction::None).id();
    let b = app.world_mut().spawn(Interaction::None).id();

    app.update();
    assert!(
      !app.world().resource::<UiPointerCaptured>().0,
      "all None → not captured"
    );

    app
      .world_mut()
      .get_mut::<Interaction>(a)
      .unwrap()
      .set_if_neq(Interaction::Hovered);
    app.update();
    assert!(
      app.world().resource::<UiPointerCaptured>().0,
      "hover captures"
    );

    app
      .world_mut()
      .get_mut::<Interaction>(b)
      .unwrap()
      .set_if_neq(Interaction::Pressed);
    app.update();
    assert!(
      app.world().resource::<UiPointerCaptured>().0,
      "press captures"
    );

    // a 释放但 b 仍按下 → 仍捕获
    app
      .world_mut()
      .get_mut::<Interaction>(a)
      .unwrap()
      .set_if_neq(Interaction::None);
    app.update();
    assert!(app.world().resource::<UiPointerCaptured>().0);

    // 全部 None → 释放
    app
      .world_mut()
      .get_mut::<Interaction>(b)
      .unwrap()
      .set_if_neq(Interaction::None);
    app.update();
    assert!(
      !app.world().resource::<UiPointerCaptured>().0,
      "released when all None"
    );
  }
}
