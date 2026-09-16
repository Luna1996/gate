//! DebugWindow 容器与菜单组件的可调常量（集中在此，改数值不用翻实现）。
//!
//! 这些是**布局/动画**常量（尺寸、时长），不是运行时可调状态——后者一律进
//! [`super::model::MenuFile`]（TOML 持久化）。

use bevy::prelude::*;

/// 标题栏高度（px）
pub const TITLE_BAR_H: f32 = 30.0;
/// 菜单页宽度（px）。各页等宽 → 窗口宽度 = 本值 + 2×边框，由内容唯一决定
pub const PAGE_W: f32 = 360.0;
/// 页面切换动画时长（秒）
pub const PAGE_ANIM_SECS: f32 = 0.2;
/// 默认列表项高度（px）
pub const ITEM_H: f32 = 30.0;
/// 「左|中|右」三分行的**左**列固定宽度（px；放名称，容得下 6 个中文字 ≈ 104px，
/// 更长的名称由 label 中间省略兜住；比右列宽）
pub const LEFT_COL_W: f32 = 80.0;
/// 「左|中|右」三分行的**右**列固定宽度（px；放数值 / 色块，比左列窄）
pub const RIGHT_COL_W: f32 = 50.0;
/// 列表项左右内边距（px）
pub const ITEM_PAD: f32 = 6.0;
/// 行内控件高度（px；需容身于 [`ITEM_H`]）
pub const CTRL_H: f32 = 24.0;
/// 按钮组相邻按钮间距（px；切换组为 0 = 无空隙并排）
pub const BUTTON_GAP: f32 = 2.0;
/// 标题栏图标字号（px）
pub const TITLE_ICON_SIZE: f32 = 12.0;
/// 窗口默认位置（逻辑 px，左上角锚定）
pub const DEFAULT_WINDOW_POS: Vec2 = Vec2::new(8.0, 8.0);
/// 窗口与系统窗口边缘的最小距离（逻辑 px）
pub const WINDOW_MARGIN: f32 = 8.0;
