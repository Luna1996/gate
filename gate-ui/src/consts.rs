//! gate-ui 根目录级旋钮（主题 / 自适应缩放）：改这里即改默认值。
//!
//! 菜单容器的旋钮在 `menu/consts.rs`，widget 的旋钮在 `widgets/consts.rs`。
//! 主题颜色/度量令牌来自运行期资产 `assets/ui/theme.ron`，不在这里。

/// UiScale 自适应缩放参照的窗口高度（px）
pub const AUTOFIT_BASE_HEIGHT: f32 = 720.0;
