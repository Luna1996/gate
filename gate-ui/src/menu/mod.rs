//! 通用 Menu 模块：DebugWindow 容器 + 可序列化菜单模型 + 9 种列表单项。
//! 用法：业务侧给 `MenuFile` 初始值并 spawn，用 `menu_model` 读回状态、`MenuActionEvent` 挂回调；交互驱动为 `menu_system`。
//! 所有交互汇总成一个 `MenuActionEvent`（`path` = 节点 id 路径 + `MenuAction`），调用方按 path 分派，UI 重建无需重挂回调。

pub mod consts;
pub mod items;
pub mod model;
pub mod window;

pub use items::{
  MenuColorSwatch, MenuItem, MenuOptionButton, MenuPressPrev, MenuRole, MenuSliderValue,
  MenuSubMenuRow, MenuTextValue, join_path,
};
pub use model::{
  InputField, MenuFile, MenuNode, MenuValue, WindowState, buttons, color, dropdown, input, slider,
  sub_menu, switch_group, text, toggle, toggle_tip,
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
