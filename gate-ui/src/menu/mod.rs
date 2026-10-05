pub mod color_picker;
pub mod consts;
pub mod items;
pub mod model;
pub mod window;

pub use color_picker::{
  ColorPickerBackdrop, ColorPickerCell, ColorPickerOpen, ColorPickerPopup, color_picker_system,
  color_picker_visual_system, palette_color,
};
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

pub fn menu_model(world: &mut World) -> Option<&MenuFile> {
  let mut q = world.query_filtered::<Entity, With<DebugMenu>>();
  let root = q.iter(world).next()?;
  world.get::<DebugMenu>(root).map(|m| &m.model)
}

pub fn menu_toggle(world: &mut World, path: &str) -> Option<bool> {
  match menu_model(world)?.node(&window::split_path(path))? {
    MenuNode::Toggle { checked, .. } => Some(*checked),
    _ => None,
  }
}

pub fn menu_selected(world: &mut World, path: &str) -> Option<usize> {
  match menu_model(world)?.node(&window::split_path(path))? {
    MenuNode::SwitchGroup { selected, .. } => Some(*selected),
    _ => None,
  }
}

pub fn menu_value(world: &mut World, path: &str) -> Option<f32> {
  match menu_model(world)?.node(&window::split_path(path))? {
    MenuNode::Slider { value, .. } => Some(*value),
    _ => None,
  }
}

pub fn menu_text<'a>(world: &'a mut World, path: &str) -> Option<&'a str> {
  match menu_model(world)?.node(&window::split_path(path))? {
    MenuNode::Input { fields, .. } => fields.first().map(|f| f.text.as_str()),
    MenuNode::Color { hex, .. } => Some(hex.as_str()),
    _ => None,
  }
}
