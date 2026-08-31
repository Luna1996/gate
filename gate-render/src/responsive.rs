//! FR-5 场景响应式：窗口 resize → 渲染目标纹理原地重建 + RenderScale/Uniforms 跟随。
//!
//! 策略（spec FR-5 v0）：渲染内部分辨率跟随窗口物理像素。
//! 纹理用 `Assets<Image>::get_mut + resize`（Handle 不变，引用方零改动），
//! GpuImage 由资产管线在下一次 Prepare 自动按新描述符重建；
//! gradient/DDA shader 均有越界剔除（uniform 尺寸 / textureDimensions），div_ceil 派发安全。

use bevy::prelude::*;
use bevy::render::render_resource::Extent3d;

use crate::brickmap::dda::DdaImages;
use crate::gradient::{GradientImages, GradientUniforms, RenderScale};

/// 合法窗口尺寸范围：下限防最小化/折叠矩形，上限防超 GPU 纹理上限
/// （实测 SetWindowPos 异常矩形曾上报 65496 物理高度——负值 u16 回绕）。
const MIN_DIM: u32 = 64;
const MAX_DIM: u32 = 4096;

fn size_is_sane(s: UVec2) -> bool {
    (MIN_DIM..=MAX_DIM).contains(&s.x) && (MIN_DIM..=MAX_DIM).contains(&s.y)
}

/// 每帧对照主窗口物理尺寸；变化 → 重建 gradient + DDA 目标纹理并更新
/// RenderScale / GradientUniforms（aspect 侧由 gate-app orbit_camera_input 同帧跟随）。
/// 退化尺寸（<64 或 >4096）：跳过本次 resize 保留上一组合法尺寸，warn 仅一次。
pub fn resize_render_targets(
    windows: Query<&Window>,
    grad: Option<Res<GradientImages>>,
    dda: Option<Res<DdaImages>>,
    mut images: ResMut<Assets<Image>>,
    mut scale: ResMut<RenderScale>,
    mut uniforms: ResMut<GradientUniforms>,
    mut warned: Local<bool>,
) {
    let Ok(window) = windows.single() else { return };
    let new_size = UVec2::new(window.physical_width(), window.physical_height());
    if !size_is_sane(new_size) {
        if !*warned {
            warn!(
                "degenerate window size {}x{}, resize skipped (keep {}x{})",
                new_size.x, new_size.y, scale.size.x, scale.size.y
            );
            *warned = true;
        }
        return;
    }
    *warned = false;
    if new_size == scale.size {
        return;
    }
    info!(
        "render targets resized: {}x{} -> {}x{}",
        scale.size.x, scale.size.y, new_size.x, new_size.y
    );
    let extent = Extent3d {
        width: new_size.x,
        height: new_size.y,
        depth_or_array_layers: 1,
    };
    for handle in grad.iter().map(|g| &g.target) {
        if let Some(mut img) = images.get_mut(handle) {
            img.resize(extent);
        }
    }
    for handle in dda.iter().map(|d| &d.target) {
        if let Some(mut img) = images.get_mut(handle) {
            img.resize(extent);
        }
    }
    scale.size = new_size;
    uniforms.size = new_size.as_vec2().extend(0.0).extend(0.0);
}

pub struct ResponsivePlugin;

impl Plugin for ResponsivePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<RenderScale>()
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
        assert_eq!(s.size, crate::gradient::VIEW_SIZE);
    }

    /// dispatch 数 = size.div_ceil(WORKGROUP)：非整除尺寸也要全屏覆盖
    #[test]
    fn dispatch_count_covers_non_multiple_sizes() {
        for size in [1280u32, 720, 1024, 769, 1] {
            let gx = size.div_ceil(crate::gradient::WORKGROUP_SIZE);
            assert!(gx * crate::gradient::WORKGROUP_SIZE >= size);
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
}
