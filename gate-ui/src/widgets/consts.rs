//! widget 层的可调旋钮（尺寸 / 手感 / 提示）：改这里即改默认值。
//!
//! 需要主题令牌的量（边框宽、间距、字号）取自 `assets/ui/theme.ron`，不在这里。

use bevy::prelude::*;

/// Disabled 态降亮系数（线性 RGB 乘以它，alpha 不变）。全局唯一一处 —— 所有 widget 的禁用观感
/// 都由它定（接线见 `widgets::dim_color`）。取 `0.35`：`0.55` 那档与正常态区分度不够。
pub const DISABLED_DIM: f32 = 0.35;
/// 按下反馈缩放（绕节点中心）
pub const PRESSED_SCALE: f32 = 0.98;
/// 勾选框边长（px）
pub const BOX_SIZE: f32 = 16.0;
/// 下拉框展开/收起动画时长（秒）
pub const DROPDOWN_ANIM_SECS: f32 = 0.2;
/// 下拉框箭头图标字号（px）
pub const DROPDOWN_ARROW_SIZE: f32 = 12.0;
/// 下拉框滚轮滚动速度（每秒移动的选项数）
pub const DROPDOWN_SCROLL_SPEED: f32 = 5.0;
/// 折线画布默认宽度（px）
pub const PLOT_W: u32 = 256;
/// 折线画布默认高度（px）
pub const PLOT_H: u32 = 64;
/// 滚轮每格滚动像素数
pub const SCROLL_SPEED: f32 = 24.0;
/// 滑杆滑块边长（px）
pub const THUMB_SIZE: f32 = 16.0;
/// 滑杆拖拽中的滑块边长（px）
pub const THUMB_SIZE_DRAG: f32 = 18.0;
/// 滑杆轨道厚度（px）
pub const TRACK_HEIGHT: f32 = 6.0;
/// 滑杆精细档倍率（拖动中按住 Shift：光标位移对值的换算缩为它）
pub const SLIDER_FINE_SCALE: f32 = 0.1;
/// 开关轨道宽（px）
pub const TRACK_W: f32 = 32.0;
/// 开关轨道高（px）
pub const TRACK_H: f32 = 16.0;
/// 数字输入拖拽灵敏度（像素/步）
pub const NUMBER_DRAG_PX_PER_STEP: f32 = 4.0;
/// 输入框点击与拖拽的判定阈值（像素）
pub const DRAG_THRESHOLD_PX: f32 = 3.0;
/// 提示框展示延时（秒）
pub const TOOLTIP_DELAY: f32 = 0.5;
/// 提示框最大宽度（px，超出换行）
pub const TOOLTIP_MAX_W: f32 = 260.0;
/// 提示框相对光标的偏移 X（px）
pub const TOOLTIP_OFFSET_X: f32 = 14.0;
/// 提示框相对光标的偏移 Y（px，正 = 光标下方）
pub const TOOLTIP_OFFSET_Y: f32 = 18.0;
/// 提示框与屏幕边缘的最小留白（px）
pub const TOOLTIP_MARGIN: f32 = 8.0;
/// markdown marker 槽宽系数（等宽字体单字符前进宽 ≈ 0.6 em）
pub const MD_MONO_ADVANCE_EM: f32 = 0.6;
/// markdown 引用块左侧竖条宽（px）
pub const MD_QUOTE_BAR_W: f32 = 2.0;
/// markdown 分割线高度（px）
pub const MD_RULE_H: f32 = 1.0;

/// 滑块内缩 = 半径（轨道行程与命中范围按它对齐）
pub const THUMB_INSET: f32 = THUMB_SIZE / 2.0;

/// 提示框相对光标的偏移（逻辑 px）
pub const TOOLTIP_OFFSET: Vec2 = Vec2::new(TOOLTIP_OFFSET_X, TOOLTIP_OFFSET_Y);

/// 中间省略符（宽度计算也用它）
pub const ELLIPSIS: &str = "...";

/// 文本光标字形
pub const CARET_CHAR: char = '|';
