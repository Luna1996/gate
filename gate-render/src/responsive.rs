//! 场景响应式：窗口 resize → 渲染目标纹理原地重建 + RenderScale 跟随。
//! 渲染内部分辨率 = 窗口物理像素 ÷ factor（渲染降采样倍数见 [`crate::consts::RENDER_SCALE`]）。

use bevy::prelude::*;
use bevy::render::render_resource::Extent3d;

use crate::brickmap::dda::{DdaImages, PostFxSettings, RenderScale};
use crate::consts::{MAX_DIM, MIN_DIM};

fn size_is_sane(s: UVec2) -> bool {
  (MIN_DIM..=MAX_DIM).contains(&s.x) && (MIN_DIM..=MAX_DIM).contains(&s.y)
}

/// 窗口物理尺寸 → 渲染目标尺寸（每轴 ÷ factor 向下取整，钳到合法下限）
fn render_size_for_window(full: UVec2, factor: u32) -> UVec2 {
  let f = factor.max(1);
  UVec2::new((full.x / f).max(MIN_DIM), (full.y / f).max(MIN_DIM))
}

/// 每帧对照主窗口物理尺寸；变化 → 原地重建 DDA 目标纹理并更新 RenderScale。
/// 退化尺寸（<64 或 >4096）：跳过本次 resize，warn 仅一次。
pub fn resize_render_targets(
  windows: Query<&Window>,
  dda: Option<Res<DdaImages>>,
  mut images: ResMut<Assets<Image>>,
  mut scale: ResMut<RenderScale>,
  mut warned: Local<bool>,
) {
  let Ok(window) = windows.single() else { return };
  let full = UVec2::new(window.physical_width(), window.physical_height());
  if !size_is_sane(full) {
    if !*warned {
      warn!(
        "degenerate window size {}x{}, resize skipped (keep {}x{})",
        full.x, full.y, scale.size.x, scale.size.y
      );
      *warned = true;
    }
    return;
  }
  *warned = false;

  let new_size = render_size_for_window(full, scale.factor);
  if new_size == scale.size {
    return;
  }
  info!(
    "render targets resized: {}x{} (window {}x{}, factor {})",
    new_size.x, new_size.y, full.x, full.y, scale.factor
  );
  let extent = Extent3d { width: new_size.x, height: new_size.y, depth_or_array_layers: 1 };
  for handle in dda.iter().map(|d| &d.target) {
    if let Some(mut img) = images.get_mut(handle) {
      img.resize(extent);
    }
  }
  scale.size = new_size;
}

pub struct ResponsivePlugin;

impl Plugin for ResponsivePlugin {
  fn build(&self, app: &mut App) {
    app
      .init_resource::<RenderScale>()
      .init_resource::<PostFxSettings>()
      .add_systems(Update, resize_render_targets);

    // 启动降采样倍数：只是初值，运行期以 RenderScale.factor 为准（菜单「视频/半分辨率」写它）
    let f = crate::consts::RENDER_SCALE;
    if f > 1 {
      app.world_mut().resource_mut::<RenderScale>().factor = f;
    }
  }
}
