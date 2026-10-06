use std::collections::BTreeMap;

use serde::{Deserialize, Serialize, Serializer};

use super::consts::DEFAULT_WINDOW_POS;
use crate::widgets::TextInputKind;

struct Rounded(f32);

impl Serialize for Rounded {
  fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
    s.serialize_f64(((self.0 as f64) * 1e6).round() / 1e6)
  }
}

fn ser_f32<S: Serializer>(v: &f32, s: S) -> Result<S::Ok, S::Error> {
  Rounded(*v).serialize(s)
}

fn ser_opt_f32<S: Serializer>(v: &Option<f32>, s: S) -> Result<S::Ok, S::Error> {
  match v {
    Some(x) => s.serialize_some(&Rounded(*x)),
    None => s.serialize_none(),
  }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Default)]
pub struct MenuFile {
  #[serde(default)]
  pub window: WindowState,
  #[serde(default)]
  pub items: Vec<MenuNode>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(default)]
pub struct WindowState {
  #[serde(serialize_with = "ser_f32")]
  pub x: f32,
  #[serde(serialize_with = "ser_f32")]
  pub y: f32,
  #[serde(default)]
  pub collapsed: bool,
  #[serde(default)]
  pub path: Vec<String>,
  #[serde(default)]
  pub pins: Vec<String>,
}

impl Default for WindowState {
  fn default() -> Self {
    Self {
      x: DEFAULT_WINDOW_POS.x,
      y: DEFAULT_WINDOW_POS.y,
      collapsed: false,
      path: Vec::new(),
      pins: Vec::new(),
    }
  }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct InputField {
  #[serde(default)]
  pub label: String,
  pub text: String,
  #[serde(default, serialize_with = "ser_opt_f32")]
  pub min: Option<f32>,
  #[serde(default, serialize_with = "ser_opt_f32")]
  pub max: Option<f32>,
  #[serde(default, serialize_with = "ser_opt_f32")]
  pub step: Option<f32>,
  #[serde(default)]
  pub decimals: u32,
}

impl InputField {
  pub fn text(label: impl Into<String>, text: impl Into<String>) -> Self {
    Self { label: label.into(), text: text.into(), min: None, max: None, step: None, decimals: 0 }
  }

  pub fn number(
    label: impl Into<String>,
    text: impl Into<String>,
    min: f32,
    max: f32,
    step: f32,
    decimals: u32,
  ) -> Self {
    Self {
      label: label.into(),
      text: text.into(),
      min: Some(min),
      max: Some(max),
      step: Some(step),
      decimals,
    }
  }

  pub fn kind(&self) -> TextInputKind {
    match self.min {
      Some(min) => TextInputKind::Number {
        min,
        max: self.max.unwrap_or(min),
        step: self.step.unwrap_or(0.0),
        decimals: self.decimals as usize,
      },
      None => TextInputKind::Text,
    }
  }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(untagged)]
pub enum MenuValue {
  Bool(bool),
  Number(f64),
  Text(String),
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MenuNode {
  SubMenu {
    #[serde(default)]
    id: String,
    label: String,
    #[serde(default)]
    children: Vec<MenuNode>,
    #[serde(default, skip_serializing_if = "is_false")]
    tabs: bool,
  },
  Buttons {
    #[serde(default)]
    id: String,
    label: String,
    items: Vec<String>,
  },
  Slider {
    #[serde(default)]
    id: String,
    label: String,
    #[serde(serialize_with = "ser_f32")]
    value: f32,
    #[serde(serialize_with = "ser_f32")]
    min: f32,
    #[serde(serialize_with = "ser_f32")]
    max: f32,
    #[serde(default, serialize_with = "ser_f32")]
    step: f32,
    #[serde(default = "default_decimals")]
    decimals: u32,
    #[serde(default)]
    tooltip: Option<String>,
    #[serde(default)]
    disabled: bool,
  },
  SwitchGroup {
    #[serde(default)]
    id: String,
    label: String,
    options: Vec<String>,
    selected: usize,
    #[serde(default)]
    tooltip: Option<String>,
  },
  Dropdown {
    #[serde(default)]
    id: String,
    label: String,
    options: Vec<String>,
    selected: usize,
  },
  Toggle {
    #[serde(default)]
    id: String,
    label: String,
    checked: bool,
    #[serde(default)]
    tooltip: Option<String>,
    #[serde(default)]
    disabled: bool,
  },
  Input {
    #[serde(default)]
    id: String,
    label: String,
    fields: Vec<InputField>,
  },
  Color {
    #[serde(default)]
    id: String,
    label: String,
    hex: String,
    #[serde(default)]
    disabled: bool,
  },
  Text {
    #[serde(default)]
    id: String,
    text: String,
  },
}

fn default_decimals() -> u32 {
  2
}

fn is_false(v: &bool) -> bool {
  !*v
}

fn pick<'a>(id: &'a str, label: &'a str) -> &'a str {
  if id.is_empty() { label } else { id }
}

impl MenuNode {
  pub fn id(&self) -> &str {
    match self {
      Self::SubMenu { id, label, .. }
      | Self::Buttons { id, label, .. }
      | Self::Slider { id, label, .. }
      | Self::SwitchGroup { id, label, .. }
      | Self::Dropdown { id, label, .. }
      | Self::Toggle { id, label, .. }
      | Self::Input { id, label, .. }
      | Self::Color { id, label, .. } => pick(id, label),
      Self::Text { id, text } => pick(id, text),
    }
  }

  pub fn label(&self) -> &str {
    match self {
      Self::SubMenu { label, .. }
      | Self::Buttons { label, .. }
      | Self::Slider { label, .. }
      | Self::SwitchGroup { label, .. }
      | Self::Dropdown { label, .. }
      | Self::Toggle { label, .. }
      | Self::Input { label, .. }
      | Self::Color { label, .. } => label,
      Self::Text { text, .. } => text,
    }
  }

  pub fn children(&self) -> &[MenuNode] {
    match self {
      Self::SubMenu { children, .. } => children,
      _ => &[],
    }
  }

  pub fn children_mut(&mut self) -> &mut Vec<MenuNode> {
    match self {
      Self::SubMenu { children, .. } => children,
      _ => unreachable!("children_mut 只对 SubMenu 有意义"),
    }
  }

  pub fn is_sub_menu(&self) -> bool {
    matches!(self, Self::SubMenu { .. })
  }

  pub fn tabs(&self) -> bool {
    matches!(self, Self::SubMenu { tabs: true, .. })
  }

  pub fn tooltip(&self) -> Option<&str> {
    match self {
      Self::Slider { tooltip, .. }
      | Self::Toggle { tooltip, .. }
      | Self::SwitchGroup { tooltip, .. } => tooltip.as_deref(),
      _ => None,
    }
  }

  pub fn value(&self) -> Option<MenuValue> {
    match self {
      Self::Toggle { checked, .. } => Some(MenuValue::Bool(*checked)),
      Self::Slider { value, .. } => Some(MenuValue::Number(f64::from(*value))),
      Self::SwitchGroup { options, selected, .. } | Self::Dropdown { options, selected, .. } => {
        options.get(*selected).cloned().map(MenuValue::Text)
      }
      Self::Input { fields, .. } => fields.first().map(|f| MenuValue::Text(f.text.clone())),
      Self::Color { hex, .. } => Some(MenuValue::Text(hex.clone())),
      Self::SubMenu { .. } | Self::Buttons { .. } | Self::Text { .. } => None,
    }
  }

  pub fn disabled(&self) -> bool {
    match self {
      Self::Slider { disabled, .. }
      | Self::Color { disabled, .. }
      | Self::Toggle { disabled, .. } => *disabled,
      _ => false,
    }
  }

  pub fn set_disabled(&mut self, v: bool) -> bool {
    let slot = match self {
      Self::Slider { disabled, .. }
      | Self::Color { disabled, .. }
      | Self::Toggle { disabled, .. } => disabled,
      _ => return false,
    };
    let changed = *slot != v;
    *slot = v;
    changed
  }

  pub fn apply_value(&mut self, v: &MenuValue) -> bool {
    match (self, v) {
      (Self::Toggle { checked, .. }, MenuValue::Bool(b)) => {
        *checked = *b;
        true
      }
      (Self::Slider { value, .. }, MenuValue::Number(n)) => {
        *value = *n as f32;
        true
      }
      (
        Self::SwitchGroup { options, selected, .. } | Self::Dropdown { options, selected, .. },
        MenuValue::Text(t),
      ) => match options.iter().position(|o| o == t) {
        Some(i) => {
          *selected = i;
          true
        }
        None => false,
      },
      (Self::Input { fields, .. }, MenuValue::Text(t)) => match fields.first_mut() {
        Some(f) => {
          f.text.clone_from(t);
          true
        }
        None => false,
      },
      (Self::Color { hex, .. }, MenuValue::Text(t)) => {
        hex.clone_from(t);
        true
      }
      _ => false,
    }
  }
}

impl MenuFile {
  pub fn node(&self, path: &[String]) -> Option<&MenuNode> {
    let (first, rest) = path.split_first()?;
    let mut cur = self.items.iter().find(|n| n.id() == first)?;
    for seg in rest {
      cur = cur.children().iter().find(|n| n.id() == seg)?;
    }
    Some(cur)
  }

  pub fn node_mut(&mut self, path: &[String]) -> Option<&mut MenuNode> {
    let (first, rest) = path.split_first()?;
    let mut cur = self.items.iter_mut().find(|n| n.id() == first)?;
    for seg in rest {
      cur = cur.children_mut().iter_mut().find(|n| n.id() == seg)?;
    }
    Some(cur)
  }

  pub fn children_of(&self, path: &[String]) -> &[MenuNode] {
    if path.is_empty() { &self.items } else { self.node(path).map(|n| n.children()).unwrap_or(&[]) }
  }

  pub fn to_toml(&self) -> Result<String, toml::ser::Error> {
    toml::to_string_pretty(self)
  }

  pub fn from_toml(src: &str) -> Result<Self, toml::de::Error> {
    toml::from_str(src)
  }

  pub fn values(&self) -> BTreeMap<String, MenuValue> {
    let mut out = BTreeMap::new();
    walk_values(&self.items, "", &mut out);
    out
  }

  pub fn apply_values(&mut self, values: &BTreeMap<String, MenuValue>) -> usize {
    let mut applied = 0;
    write_values(&mut self.items, "", values, &mut applied);
    applied
  }

  pub fn sanitize(&mut self) {
    for node in &mut self.items {
      sanitize_node(node);
    }
  }
}

fn sanitize_node(node: &mut MenuNode) {
  match node {
    MenuNode::SwitchGroup { options, selected, .. }
    | MenuNode::Dropdown { options, selected, .. } => {
      *selected = (*selected).min(options.len().saturating_sub(1));
    }
    MenuNode::Slider { value, min, max, step, .. } => {
      if *max < *min {
        std::mem::swap(min, max);
      }
      let v = value.clamp(*min, *max);
      *value = if *step > 0.0 { *min + ((v - *min) / *step).round() * *step } else { v };
      *value = value.clamp(*min, *max);
    }
    MenuNode::SubMenu { children, .. } => {
      for c in children.iter_mut() {
        sanitize_node(c);
      }
    }
    _ => {}
  }
}

fn id_path(prefix: &str, id: &str) -> String {
  if prefix.is_empty() { id.to_string() } else { format!("{prefix}/{id}") }
}

fn walk_values(nodes: &[MenuNode], prefix: &str, out: &mut BTreeMap<String, MenuValue>) {
  for node in nodes {
    let path = id_path(prefix, node.id());
    match node {
      MenuNode::SubMenu { children, .. } => walk_values(children, &path, out),
      _ => {
        if let Some(v) = node.value() {
          out.insert(path, v);
        }
      }
    }
  }
}

fn write_values(
  nodes: &mut [MenuNode],
  prefix: &str,
  values: &BTreeMap<String, MenuValue>,
  applied: &mut usize,
) {
  for node in nodes.iter_mut() {
    let path = id_path(prefix, node.id());
    match node {
      MenuNode::SubMenu { children, .. } => write_values(children, &path, values, applied),
      _ => {
        if let Some(v) = values.get(&path)
          && node.apply_value(v)
        {
          *applied += 1;
        }
      }
    }
  }
}

pub fn sub_menu(id: &str, label: &str, children: Vec<MenuNode>) -> MenuNode {
  MenuNode::SubMenu { id: id.into(), label: label.into(), children, tabs: false }
}

pub fn sub_menu_tabs(id: &str, label: &str, children: Vec<MenuNode>) -> MenuNode {
  MenuNode::SubMenu { id: id.into(), label: label.into(), children, tabs: true }
}

pub fn buttons(id: &str, label: &str, items: &[&str]) -> MenuNode {
  MenuNode::Buttons {
    id: id.into(),
    label: label.into(),
    items: items.iter().map(|s| (*s).to_string()).collect(),
  }
}

#[allow(clippy::too_many_arguments)]
pub fn slider(
  id: &str,
  label: &str,
  value: f32,
  min: f32,
  max: f32,
  step: f32,
  decimals: u32,
  tooltip: Option<&str>,
) -> MenuNode {
  MenuNode::Slider {
    id: id.into(),
    label: label.into(),
    value,
    min,
    max,
    step,
    decimals,
    tooltip: tooltip.map(|s| s.to_string()),
    disabled: false,
  }
}

pub fn switch_group(
  id: &str,
  label: &str,
  options: &[&str],
  selected: usize,
  tooltip: Option<&str>,
) -> MenuNode {
  MenuNode::SwitchGroup {
    id: id.into(),
    label: label.into(),
    options: options.iter().map(|s| (*s).to_string()).collect(),
    selected,
    tooltip: tooltip.map(|s| s.to_string()),
  }
}

pub fn dropdown(id: &str, label: &str, options: &[&str], selected: usize) -> MenuNode {
  MenuNode::Dropdown {
    id: id.into(),
    label: label.into(),
    options: options.iter().map(|s| (*s).to_string()).collect(),
    selected,
  }
}

pub fn toggle(id: &str, label: &str, checked: bool) -> MenuNode {
  MenuNode::Toggle { id: id.into(), label: label.into(), checked, tooltip: None, disabled: false }
}

pub fn toggle_tip(id: &str, label: &str, checked: bool, tooltip: &str) -> MenuNode {
  MenuNode::Toggle {
    id: id.into(),
    label: label.into(),
    checked,
    tooltip: Some(tooltip.to_string()),
    disabled: false,
  }
}

pub fn input(id: &str, label: &str, fields: Vec<InputField>) -> MenuNode {
  MenuNode::Input { id: id.into(), label: label.into(), fields }
}

pub fn color(id: &str, label: &str, hex: &str) -> MenuNode {
  MenuNode::Color { id: id.into(), label: label.into(), hex: hex.into(), disabled: false }
}

pub fn text(id: &str, content: &str) -> MenuNode {
  MenuNode::Text { id: id.into(), text: content.into() }
}
