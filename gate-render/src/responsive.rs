//! FR-5 场景响应式：窗口 resize → 渲染目标纹理原地重建 + RenderScale/Uniforms 跟随。
//!
//! 渲染内部分辨率 = 窗口物理像素 ÷ factor（`GATE_RES_SCALE`，默认 1 即全分辨率）。纹理用
//! `Assets<Image>::get_mut + resize`（Handle 不变，引用方零改动），GpuImage 由资产管线在下
//! 一次 Prepare 按新描述符重建；DDA shader 有越界剔除（textureDimensions），div_ceil 派发安全。

use bevy::prelude::*;
use bevy::render::render_resource::Extent3d;

use crate::brickmap::dda::{DdaImages, PostFxSettings, RenderScale};

/// 合法窗口尺寸范围：下限防最小化/折叠矩形，上限防超 GPU 纹理上限
/// （SetWindowPos 收到负高度会按 u16 回绕上报成 65496）。
const MIN_DIM: u32 = 64;
const MAX_DIM: u32 = 4096;

/// 启动时的降采样倍数（`GATE_RES_SCALE`，默认 1 = 全分辨率）。
/// **只是初值**：运行期以 `RenderScale.factor` 为准（菜单「视频/半分辨率」写它）。
/// 非数字或 <1 一律按 1 处理。
fn render_scale_factor_from_env() -> u32 {
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
fn render_size_for_window(full: UVec2, factor: u32) -> UVec2 {
  let f = factor.max(1);
  UVec2::new((full.x / f).max(MIN_DIM), (full.y / f).max(MIN_DIM))
}

/// 每帧对照主窗口物理尺寸；变化 → 原地重建 DDA 目标纹理并更新 RenderScale
/// （aspect 侧由 gate-app orbit_camera_input 同帧按全窗尺寸跟随——NDC 与渲染分辨率无关，
/// 反投影/拾取在全分辨率语义下保持正确）。
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
  // `scale.factor` 由菜单「视频/半分辨率」写；它一变，这里算出的尺寸就变 ⇒ 下一帧原地重建。
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
    // 启动档位可由 `GATE_RES_SCALE` 覆盖（默认 1 = 全分辨率）；之后以菜单开关为准。
    let f = render_scale_factor_from_env();
    if f > 1 {
      app.world_mut().resource_mut::<RenderScale>().factor = f;
    }
  }
}
