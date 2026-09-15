//! 通用 Menu 模块：DebugWindow 容器 + 可序列化菜单模型 + 8 种列表单项。
//!
//! 分两层用法：
//! 1. 本模块（gate-ui 级通用）：容器、通用组件、模型/TOML、交互驱动（[`menu_system`]）；
//! 2. 业务侧（如 gate-app）：给出 [`MenuFile`] 初始值（TOML 加载或代码构造），
//!    spawn 后用 [`menu_model`] 读回状态、用 [`MenuActionEvent`] 挂回调。
//!
//! 回调接口：所有交互汇总成 [`MenuActionEvent`]（`path` = 节点 id 路径 + [`MenuAction`]），
//! 调用方一个观察者按 path 分派即可——UI 重建不需要重挂回调。

pub mod consts;
pub mod items;
pub mod model;
pub mod window;

pub use consts::{
  BUTTON_GAP, CTRL_H, DEFAULT_WINDOW_POS, ITEM_H, ITEM_PAD, PAGE_ANIM_SECS, PAGE_W, SIDE_COL_W,
  TITLE_BAR_H, TITLE_ICON_SIZE, TITLE_ICON_W, WINDOW_MARGIN,
};
pub use items::{
  MenuColorSwatch, MenuItem, MenuOptionButton, MenuPressPrev, MenuRole, MenuSliderValue,
  MenuSubMenuRow, join_path,
};
pub use model::{
  InputField, MenuFile, MenuNode, WindowState, buttons, color, input, slider, sub_menu,
  switch_group, text, toggle, toggle_tip,
};
pub use window::{
  DebugMenu, DebugMenuHandle, DebugMenuRoot, MenuAction, MenuActionEvent, MenuDrag, MenuPage,
  MenuPager, MenuParts, menu_system, spawn_debug_menu,
};

use bevy::prelude::*;

/// 读回当前菜单模型（持久化 / 读初值；无菜单 → None）
pub fn menu_model(world: &mut World) -> Option<&MenuFile> {
  let mut q = world.query_filtered::<Entity, With<DebugMenu>>();
  let root = q.iter(world).next()?;
  world.get::<DebugMenu>(root).map(|m| &m.model)
}

/// 按节点 id 路径读开关状态
pub fn menu_toggle(world: &mut World, path: &str) -> Option<bool> {
  match menu_model(world)?.node(&window::split_path(path))? {
    MenuNode::Toggle { checked, .. } => Some(*checked),
    _ => None,
  }
}

/// 按节点 id 路径读切换组选中下标
pub fn menu_selected(world: &mut World, path: &str) -> Option<usize> {
  match menu_model(world)?.node(&window::split_path(path))? {
    MenuNode::SwitchGroup { selected, .. } => Some(*selected),
    _ => None,
  }
}

/// 按节点 id 路径读滑动条值
pub fn menu_value(world: &mut World, path: &str) -> Option<f32> {
  match menu_model(world)?.node(&window::split_path(path))? {
    MenuNode::Slider { value, .. } => Some(*value),
    _ => None,
  }
}

/// 按节点 id 路径读输入框第一个字段文本
pub fn menu_text<'a>(world: &'a mut World, path: &str) -> Option<&'a str> {
  match menu_model(world)?.node(&window::split_path(path))? {
    MenuNode::Input { fields, .. } => fields.first().map(|f| f.text.as_str()),
    MenuNode::Color { hex, .. } => Some(hex.as_str()),
    _ => None,
  }
}
