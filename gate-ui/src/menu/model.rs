//! 菜单模型：层级 + 控件状态的可序列化表示（TOML 持久化）。
//! 单靠一份 `MenuFile` 即可完整重建菜单 UI；唯一例外是控件回调，由调用方按节点 id 路径挂上（见 `MenuActionEvent`）。
//! 文案字段存 i18n key，渲染经 `UiTranslator` 解析；`id` 是与语言无关的回调路径段，须显式给出。

use std::collections::BTreeMap;

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

/// 窗口状态（缺字段回落 `Default`，便于手改配置文件）
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(default)]
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

/// 控件值的持久化表示（外部配置文件里 `[menu]` 表的值；无状态控件不产出）。
/// 选中态存**选项名**而非下标：切换组 = i18n key、下拉框 = 模型名，选项重排/增删后仍能找回。
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(untagged)]
pub enum MenuValue {
  Bool(bool),
  Number(f64),
  Text(String),
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
    /// **禁用态**：整行不可交互（配色降亮）。用于"值由别处决定、这里只是个显示/覆写值"的控件
    /// —— 例：天空页三组「覆写」关着时，下面那些滑杆就不是它说了算（见 `MenuNode::set_disabled`）。
    /// 只在 spawn 时生效（widget 配色是 spawn 时算的）⇒ 改它要重建那一页（`rebuild_menu_page`）。
    #[serde(default)]
    disabled: bool,
  },
  /// 切换组（无空隙并排，选中态持久化）
  SwitchGroup {
    #[serde(default)]
    id: String,
    label: String,
    options: Vec<String>,
    selected: usize,
    #[serde(default)]
    tooltip: Option<String>,
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
    /// **禁用态**：同 `Slider::disabled`（天空页「颜色」组的覆写关着时，三个拾色器不可改）
    #[serde(default)]
    disabled: bool,
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
      Self::Slider { tooltip, .. }
      | Self::Toggle { tooltip, .. }
      | Self::SwitchGroup { tooltip, .. } => tooltip.as_deref(),
      _ => None,
    }
  }

  /// 控件当前值（无状态控件 `SubMenu` / `Buttons` / `Text` → None）
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

  /// 本行是否处于**禁用态**（不可交互）。目前只有 `Slider` / `Color` 支持（其余节点恒 `false`）。
  pub fn disabled(&self) -> bool {
    match self {
      Self::Slider { disabled, .. } | Self::Color { disabled, .. } => *disabled,
      _ => false,
    }
  }

  /// 设置本行的禁用态；返回**是否发生了变化**（调用方据此决定要不要重建页面，
  /// 因为 widget 的配色是 spawn 时算的，光改模型不会重画）。
  pub fn set_disabled(&mut self, v: bool) -> bool {
    let slot = match self {
      Self::Slider { disabled, .. } | Self::Color { disabled, .. } => disabled,
      _ => return false,
    };
    let changed = *slot != v;
    *slot = v;
    changed
  }

  /// 用外部值覆盖控件状态：类型不符 / 选项不存在 → 忽略并返回 false。
  /// 滑杆只赋值不钳位，调用方随后 `MenuFile::sanitize` 归一化。
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

  /// 全树「id 路径 → 控件值」（持久化用；无状态控件不产出）
  pub fn values(&self) -> BTreeMap<String, MenuValue> {
    let mut out = BTreeMap::new();
    walk_values(&self.items, "", &mut out);
    out
  }

  /// 按 id 路径把外部值写回控件；配置里有、而这里没有的路径（控件已删）与类型不符的值一律忽略。
  /// 返回成功应用的条数；调用方随后 `sanitize` 归一化（滑杆钳位等）。
  pub fn apply_values(&mut self, values: &BTreeMap<String, MenuValue>) -> usize {
    let mut applied = 0;
    write_values(&mut self.items, "", values, &mut applied);
    applied
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

/// 节点 id 路径段拼接（根 = ""）
fn id_path(prefix: &str, id: &str) -> String {
  if prefix.is_empty() { id.to_string() } else { format!("{prefix}/{id}") }
}

/// 深度优先收集控件值（仅含带状态控件）
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

/// 深度优先套用外部值（子菜单只递归；节点自身的值由 `MenuNode::apply_value` 判定）
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
#[allow(clippy::too_many_arguments)] // 模型构造器：逐字段镜像 `MenuNode::Slider`，调用点按字段顺序可读
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

/// 切换组节点
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
  MenuNode::Color { id: id.into(), label: label.into(), hex: hex.into(), disabled: false }
}

/// 纯文本节点
pub fn text(id: &str, content: &str) -> MenuNode {
  MenuNode::Text { id: id.into(), text: content.into() }
}
