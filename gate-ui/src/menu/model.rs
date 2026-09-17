//! 菜单模型：层级 + 控件状态的可序列化表示（TOML 持久化）。
//! 单靠一份 `MenuFile` 即可完整重建菜单 UI；唯一例外是控件回调，由调用方按节点 id 路径挂上（见 `MenuActionEvent`）。
//! 文案字段存 i18n key，渲染经 `UiTranslator` 解析；`id` 是与语言无关的回调路径段，须显式给出。

use serde::{Deserialize, Serialize, Serializer};

use super::consts::DEFAULT_WINDOW_POS;
use crate::widgets::TextInputKind;

/// 写盘用的浮点包装：f32 → 6 位小数的 f64
struct Rounded(f32);

impl Serialize for Rounded {
  fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
    s.serialize_f64(((self.0 as f64) * 1e6).round() / 1e6)
  }
}

/// f32 字段的序列化助手（配合 `#[serde(serialize_with = ...)]`）
fn ser_f32<S: Serializer>(v: &f32, s: S) -> Result<S::Ok, S::Error> {
  Rounded(*v).serialize(s)
}

/// Option<f32> 字段的序列化助手
fn ser_opt_f32<S: Serializer>(v: &Option<f32>, s: S) -> Result<S::Ok, S::Error> {
  match v {
    Some(x) => s.serialize_some(&Rounded(*x)),
    None => s.serialize_none(),
  }
}

/// 菜单持久化文件（TOML 顶层结构）
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Default)]
pub struct MenuFile {
  /// 窗口状态（位置/收起/停留路径）
  #[serde(default)]
  pub window: WindowState,
  /// 根节点的子项
  #[serde(default)]
  pub items: Vec<MenuNode>,
}

/// 窗口状态
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct WindowState {
  #[serde(serialize_with = "ser_f32")]
  pub x: f32,
  #[serde(serialize_with = "ser_f32")]
  pub y: f32,
  /// true = 只显示标题栏
  #[serde(default)]
  pub collapsed: bool,
  /// 停留节点路径（节点 id 序列；空 = 根）
  #[serde(default)]
  pub path: Vec<String>,
}

impl Default for WindowState {
  fn default() -> Self {
    Self { x: DEFAULT_WINDOW_POS.x, y: DEFAULT_WINDOW_POS.y, collapsed: false, path: Vec::new() }
  }
}

/// 输入框字段（`min` 有值 = 数字模式，否则纯文本）
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct InputField {
  /// 字段名前缀的 i18n key（空 = 无前缀）
  #[serde(default)]
  pub label: String,
  pub text: String,
  /// 数字模式下限（有值即数字模式）
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
  /// 纯文本字段
  pub fn text(label: impl Into<String>, text: impl Into<String>) -> Self {
    Self { label: label.into(), text: text.into(), min: None, max: None, step: None, decimals: 0 }
  }

  /// 数字字段（支持点击拖拽调值）
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

  /// 落到 widget 层的输入模式
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

/// 菜单节点（层级 + 控件状态）
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MenuNode {
  /// 子菜单：点击进入下级
  SubMenu {
    #[serde(default)]
    id: String,
    label: String,
    #[serde(default)]
    children: Vec<MenuNode>,
  },
  /// 按钮组（等宽并排；点击只上报，不改模型状态）
  Buttons {
    #[serde(default)]
    id: String,
    label: String,
    items: Vec<String>,
  },
  /// 滑动条（左名称 | 中滑杆 | 右数值）
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
    /// 步长；0 = 连续
    #[serde(default, serialize_with = "ser_f32")]
    step: f32,
    /// 数值显示小数位
    #[serde(default = "default_decimals")]
    decimals: u32,
    #[serde(default)]
    tooltip: Option<String>,
  },
  /// 切换组（无空隙并排，选中态持久化）
  SwitchGroup {
    #[serde(default)]
    id: String,
    label: String,
    options: Vec<String>,
    selected: usize,
  },
  /// 下拉框（左名称 | 中右下拉控件；选中态持久化）
  Dropdown {
    #[serde(default)]
    id: String,
    label: String,
    options: Vec<String>,
    selected: usize,
  },
  /// 开关项（左文字 | 右 toggle）
  Toggle {
    #[serde(default)]
    id: String,
    label: String,
    checked: bool,
    #[serde(default)]
    tooltip: Option<String>,
  },
  /// 输入框（一个或多个等宽输入框）
  Input {
    #[serde(default)]
    id: String,
    label: String,
    fields: Vec<InputField>,
  },
  /// 颜色选择器（左名称 | 中 HEX 输入 | 右色块）
  Color {
    #[serde(default)]
    id: String,
    label: String,
    hex: String,
  },
  /// 纯文本（内容可由调用方运行时改写）
  Text {
    #[serde(default)]
    id: String,
    text: String,
  },
}

fn default_decimals() -> u32 {
  2
}

/// 取显式 id，缺省回退 label（回调路径用 id，显示用 label）
fn pick<'a>(id: &'a str, label: &'a str) -> &'a str {
  if id.is_empty() { label } else { id }
}

impl MenuNode {
  /// 回调标识（路径段）
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

  /// 行内显示文案的 i18n key（各控件左侧文字；纯文本项为正文）
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

  /// 子节点（仅子菜单有）
  pub fn children(&self) -> &[MenuNode] {
    match self {
      Self::SubMenu { children, .. } => children,
      _ => &[],
    }
  }

  /// 子节点（可变）
  pub fn children_mut(&mut self) -> &mut Vec<MenuNode> {
    match self {
      Self::SubMenu { children, .. } => children,
      _ => unreachable!("children_mut 只对 SubMenu 有意义"),
    }
  }

  /// 是否可进入下级
  pub fn is_sub_menu(&self) -> bool {
    matches!(self, Self::SubMenu { .. })
  }

  /// 悬浮提示文案（没有则 None）
  pub fn tooltip(&self) -> Option<&str> {
    match self {
      Self::Slider { tooltip, .. } | Self::Toggle { tooltip, .. } => tooltip.as_deref(),
      _ => None,
    }
  }
}

impl MenuFile {
  /// 按 id 路径取节点（空路径 = None，根不是节点）
  pub fn node(&self, path: &[String]) -> Option<&MenuNode> {
    let (first, rest) = path.split_first()?;
    let mut cur = self.items.iter().find(|n| n.id() == first)?;
    for seg in rest {
      cur = cur.children().iter().find(|n| n.id() == seg)?;
    }
    Some(cur)
  }

  /// 按 id 路径取节点（可变）
  pub fn node_mut(&mut self, path: &[String]) -> Option<&mut MenuNode> {
    let (first, rest) = path.split_first()?;
    let mut cur = self.items.iter_mut().find(|n| n.id() == first)?;
    for seg in rest {
      cur = cur.children_mut().iter_mut().find(|n| n.id() == seg)?;
    }
    Some(cur)
  }

  /// 某路径下的子项列表（空路径 = 根）
  pub fn children_of(&self, path: &[String]) -> &[MenuNode] {
    if path.is_empty() { &self.items } else { self.node(path).map(|n| n.children()).unwrap_or(&[]) }
  }

  /// 序列化为 TOML（持久化写盘用）；f32 字段经 `Rounded` 输出 6 位小数。
  pub fn to_toml(&self) -> Result<String, toml::ser::Error> {
    toml::to_string_pretty(self)
  }

  /// 从 TOML 反序列化
  pub fn from_toml(src: &str) -> Result<Self, toml::de::Error> {
    toml::from_str(src)
  }

  /// 校验/纠偏：选中下标越界钳位（TOML 被手改后不至于 panic）
  pub fn sanitize(&mut self) {
    for node in &mut self.items {
      sanitize_node(node);
    }
  }
}

fn sanitize_node(node: &mut MenuNode) {
  match node {
    // 切换组 / 下拉框：选中下标越界钳位
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

/// 子菜单节点
pub fn sub_menu(id: &str, label: &str, children: Vec<MenuNode>) -> MenuNode {
  MenuNode::SubMenu { id: id.into(), label: label.into(), children }
}

/// 按钮组节点
pub fn buttons(id: &str, label: &str, items: &[&str]) -> MenuNode {
  MenuNode::Buttons {
    id: id.into(),
    label: label.into(),
    items: items.iter().map(|s| (*s).to_string()).collect(),
  }
}

/// 滑动条节点
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
  }
}

/// 切换组节点
pub fn switch_group(id: &str, label: &str, options: &[&str], selected: usize) -> MenuNode {
  MenuNode::SwitchGroup {
    id: id.into(),
    label: label.into(),
    options: options.iter().map(|s| (*s).to_string()).collect(),
    selected,
  }
}

/// 下拉框节点
pub fn dropdown(id: &str, label: &str, options: &[&str], selected: usize) -> MenuNode {
  MenuNode::Dropdown {
    id: id.into(),
    label: label.into(),
    options: options.iter().map(|s| (*s).to_string()).collect(),
    selected,
  }
}

/// 开关项节点
pub fn toggle(id: &str, label: &str, checked: bool) -> MenuNode {
  MenuNode::Toggle { id: id.into(), label: label.into(), checked, tooltip: None }
}

/// 带提示的开关项节点
pub fn toggle_tip(id: &str, label: &str, checked: bool, tooltip: &str) -> MenuNode {
  MenuNode::Toggle {
    id: id.into(),
    label: label.into(),
    checked,
    tooltip: Some(tooltip.to_string()),
  }
}

/// 输入框节点
pub fn input(id: &str, label: &str, fields: Vec<InputField>) -> MenuNode {
  MenuNode::Input { id: id.into(), label: label.into(), fields }
}

/// 颜色选择器节点
pub fn color(id: &str, label: &str, hex: &str) -> MenuNode {
  MenuNode::Color { id: id.into(), label: label.into(), hex: hex.into() }
}

/// 纯文本节点
pub fn text(id: &str, content: &str) -> MenuNode {
  MenuNode::Text { id: id.into(), text: content.into() }
}
