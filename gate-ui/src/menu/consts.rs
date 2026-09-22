//! DebugWindow 容器与菜单组件的可调旋钮：改这里即改默认值。
//!
//! 布局/动画常量（尺寸、时长）；运行时可调的状态不在这里——后者进 `super::model::MenuFile`（TOML 持久化）。

use bevy::prelude::*;

/// 标题栏高度（px）
pub const TITLE_BAR_H: f32 = 30.0;
/// 菜单页宽度（px）＝ 窗口内容宽（各页等宽）
pub const PAGE_W: f32 = 360.0;
/// 页面切换动画时长（秒）
pub const PAGE_ANIM_SECS: f32 = 0.2;
/// 默认列表项高度（px）
pub const ITEM_H: f32 = 30.0;
/// 三分行左列宽度（px，放名称）
pub const LEFT_COL_W: f32 = 80.0;
/// 三分行右列宽度（px，放数值 / 色块）
pub const RIGHT_COL_W: f32 = 50.0;
/// 列表项左右内边距（px）
pub const ITEM_PAD: f32 = 6.0;
/// 行内控件高度（px，需容身于 ITEM_H）
pub const CTRL_H: f32 = 24.0;
/// 按钮组相邻按钮间距（px）
pub const BUTTON_GAP: f32 = 2.0;
/// 标题栏图标字号（px）
pub const TITLE_ICON_SIZE: f32 = 12.0;
/// 窗口与屏幕边缘的最小距离（逻辑 px）
pub const WINDOW_MARGIN: f32 = 8.0;
/// 调色板方阵边长（格数；第 0 行灰阶，其余每行一个色相）
pub const PALETTE_EDGE: usize = 16;
/// 调色板格子边长（px，正方形）
pub const PALETTE_CELL: f32 = 16.0;
/// 调色板格子选中框 / 悬停框宽（px）
pub const PALETTE_MARK_BORDER: f32 = 2.0;

/// 窗口默认位置（逻辑 px，左上角锚定）
pub const DEFAULT_WINDOW_POS: Vec2 = Vec2::new(8.0, 8.0);
