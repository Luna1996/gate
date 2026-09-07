//! FR-5 场景响应式：窗口 resize → 渲染目标纹理原地重建 + RenderScale/Uniforms 跟随。
//!
//! 策略（spec FR-5 v0）：渲染内部分辨率跟随窗口物理像素。
//! 纹理用 `Assets<Image>::get_mut + resize`（Handle 不变，引用方零改动），
//! GpuImage 由资产管线在下一次 Prepare 自动按新描述符重建；
//! DDA shader 有越界剔除（textureDimensions），div_ceil 派发安全。
//!
//! 分辨率策略（2026-09-04 用户裁决纠正）：**默认全分辨率**（Douglas 最终画面 sharp =
//! 独显全速 + FXAA；1660 Ti 7ms 是全速数字）。降分辨率只是他的**集显降档路径**
//! （#17「体素世界低分辨率反而可爱」）——`GATE_RES_SCALE=2` 显式开启，勿默认。

use bevy::prelude::*;
use bevy::render::render_resource::Extent3d;

use crate::brickmap::dda::{DdaImages, RenderScale};

/// 合法窗口尺寸范围：下限防最小化/折叠矩形，上限防超 GPU 纹理上限
/// （实测 SetWindowPos 异常矩形曾上报 65496 物理高度——负值 u16 回绕）。
const MIN_DIM: u32 = 64;
const MAX_DIM: u32 = 4096;

/// 渲染分辨率 = 窗口物理像素 ÷ factor。默认 1（全分辨率，最终方案 sharp）；
/// `GATE_RES_SCALE=2` 开启集显降档路径（Douglas #17）
fn render_scale_factor() -> u32 {
  static FACTOR: std::sync::LazyLock<u32> = std::sync::LazyLock::new(|| {
    std::env::var("GATE_RES_SCALE")
      .ok()
      .and_then(|v| v.parse().ok())
      .filter(|&f| f >= 1)
      .unwrap_or(1)
  });
  *FACTOR
}

fn size_is_sane(s: UVec2) -> bool {
  (MIN_DIM..=MAX_DIM).contains(&s.x) && (MIN_DIM..=MAX_DIM).contains(&s.y)
}

/// 窗口物理尺寸 → 渲染目标尺寸（每轴 ÷ factor 向下取整，钳到合法下限）
fn render_size_for_window(full: UVec2) -> UVec2 {
  let f = render_scale_factor();
  UVec2::new((full.x / f).max(MIN_DIM), (full.y / f).max(MIN_DIM))
}

/// 每帧对照主窗口物理尺寸；变化 → 原地重建 DDA 目标纹理并更新 RenderScale
/// （aspect 侧由 gate-app orbit_camera_input 同帧跟随，用全窗尺寸——NDC 与渲染
/// 分辨率无关，反投影/拾取在全分辨率语义下保持正确）。
/// 退化尺寸（<64 或 >4096）：跳过本次 resize 保留上一组合法尺寸，warn 仅一次。
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
  let new_size = render_size_for_window(full);
  if new_size == scale.size {
    return;
  }
  info!(
    "render targets resized: {}x{} (window {}x{}, factor {})",
    new_size.x,
    new_size.y,
    full.x,
    full.y,
    render_scale_factor()
  );
  let extent = Extent3d {
    width: new_size.x,
    height: new_size.y,
    depth_or_array_layers: 1,
  };
  // target（out_tex）重建尺寸
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
      .add_systems(Update, resize_render_targets);
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  /// RenderScale 默认 = VIEW_SIZE（窗口初始分辨率，resize 前不触发重建）
  #[test]
  fn render_scale_defaults_to_view_size() {
    let s = RenderScale::default();
    assert_eq!(s.size, crate::brickmap::dda::VIEW_SIZE);
  }

  /// dispatch 数 = size.div_ceil(WORKGROUP)：非整除尺寸也要全屏覆盖
  #[test]
  fn dispatch_count_covers_non_multiple_sizes() {
    for size in [1280u32, 720, 1024, 769, 1] {
      let gx = size.div_ceil(crate::brickmap::dda::WORKGROUP_SIZE);
      assert!(gx * crate::brickmap::dda::WORKGROUP_SIZE >= size);
    }
  }

  /// 退化尺寸防御：65496（负高度 u16 回绕，实机实测）与 0/超上限必须拒绝
  #[test]
  fn degenerate_sizes_rejected() {
    assert!(size_is_sane(UVec2::new(1280, 720)));
    assert!(size_is_sane(UVec2::new(64, 64)));
    assert!(size_is_sane(UVec2::new(4096, 4096)));
    assert!(!size_is_sane(UVec2::new(65496, 720)));
    assert!(!size_is_sane(UVec2::new(0, 720)));
    assert!(!size_is_sane(UVec2::new(4097, 100)));
  }

  /// 分辨率策略：默认全分辨率（factor=1）；GATE_RES_SCALE=2 降档路径
  #[test]
  fn render_size_follows_factor() {
    assert_eq!(
      render_size_for_window(UVec2::new(1600, 900)),
      UVec2::new(1600, 900)
    );
    assert_eq!(
      render_size_for_window(UVec2::new(100, 100)),
      UVec2::new(100, 100)
    );
  }
}
