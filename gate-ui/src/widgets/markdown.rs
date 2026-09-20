//! markdown：CommonMark/GFM 子集 → bevy_ui 节点树（retained，源文本变化才重建）。
//!
//! 分层：comrak 解析 → 单遍遍历 AST 直接 spawn（不建中间 IR）。行内样式 = 同一文本块的多个
//! `TextSpan`（共享一次 shaping）；块级结构（标题/列表/引用/代码块/表格/分割线）= 独立节点。
//!
//! 性能：常驻零帧开销（`markdown_set_text` 先比对源串，相同直接返回）；解析选项是进程级单例
//! （含容器，只构造一次）；`Arena` 每次解析局部持有，解析完即可释放。
//!
//! 重建走命令队列（`markdown_set_text` 只入队，落地时才拿到 `&mut World`）：解析、主题与字体
//! 都在命令里取，调用方不需要持有 `UiCtx`，建树代码也只用 world 版 `ChildSpawner` 一条路径。
//!
//! 已知取舍（主题字体为单字重等宽 MapleMono Regular，`FontWeight`/`FontStyle` 无对应字面）：
//! 强调实际靠文字亮度档区分（`text_primary`）；行内代码用 `TextBackgroundColor` 底色、链接用
//! `Underline`、删除线用 `Strikethrough`（后三者按 section 实体取用）。图片不加载（展示 alt 文本）；
//! 未开扩展的语法（脚注/数学/元数据等）不渲染；表格无列宽测量预通道 → 等分。

use std::ops::Deref;
use std::sync::OnceLock;

use bevy::prelude::*;
use bevy::text::{FontStyle, FontWeight, Strikethrough, TextBackgroundColor, Underline};
use bevy::ui::Overflow;
use bevy::ui::widget::Label;

use comrak::nodes::{ListDelimType, ListType, Node as MdNode, NodeList, NodeValue};
use comrak::{Arena, Options, parse_document};

use super::{UiCtx, color_of, px};
use crate::theme::{ThemeFont, UiTheme};
use crate::widgets::consts::{MD_MONO_ADVANCE_EM, MD_QUOTE_BAR_W, MD_RULE_H};

/// Markdown 视图配置（Default = 空文本 + 常规间距 + 不限制宽度）
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MarkdownConfig {
  pub text: String,
  /// 紧凑模式：块间距取 xs 档（tooltip 等窄容器）
  pub dense: bool,
  /// 最大宽度（None = 跟随父级）
  pub max_width: Option<f32>,
}

/// 已渲染的 markdown 视图（`source` 是重建判据：相同则跳过重建）
#[derive(Component, Clone, Debug)]
pub struct MarkdownView {
  source: String,
  dense: bool,
}

impl MarkdownView {
  pub fn new(source: impl Into<String>, dense: bool) -> Self {
    Self { source: source.into(), dense }
  }

  pub fn source(&self) -> &str {
    &self.source
  }

  pub fn is_dense(&self) -> bool {
    self.dense
  }
}

/// markdown 句柄（Deref 到根实体 Entity）
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MarkdownHandle(pub Entity);

impl Deref for MarkdownHandle {
  type Target = Entity;
  fn deref(&self) -> &Entity {
    &self.0
  }
}

impl From<MarkdownHandle> for Entity {
  fn from(h: MarkdownHandle) -> Self {
    h.0
  }
}

/// markdown 根节点 bundle（不含 `Name`，供需要叠加自身标记组件的调用方复用）
pub(crate) fn markdown_root(theme: &UiTheme, config: &MarkdownConfig) -> impl Bundle {
  let gap = if config.dense { theme.metrics.spacing.xs } else { theme.metrics.spacing.sm };
  (
    MarkdownView::new(config.text.clone(), config.dense),
    Node {
      flex_direction: FlexDirection::Column,
      row_gap: px(gap),
      max_width: config.max_width.map_or(Val::Auto, px),
      ..default()
    },
  )
}

/// 渲染 markdown（块级列容器；文本变化时用 `markdown_set_text` 重建）
pub fn markdown(ctx: &UiCtx, parent: &mut ChildSpawner, config: MarkdownConfig) -> MarkdownHandle {
  let r = Renderer { ctx, dense: config.dense };
  let e = parent
    .spawn((Name::new("ui-markdown"), markdown_root(ctx.theme, &config)))
    .with_children(|p| r.document(p, &config.text))
    .id();
  MarkdownHandle(e)
}

/// 源文本变化才重建（相同直接返回）；判据见 `MarkdownView::source`，调用方应先自行比对以免每帧入队。
pub fn markdown_set_text(commands: &mut Commands, entity: Entity, text: impl Into<String>) {
  let text = text.into();
  commands.queue(move |world: &mut World| rebuild(world, entity, text));
}

/// 命令落地时重建：清空子节点 → 按新源重铺（解析 / 主题 / 字体都在这里取）
fn rebuild(world: &mut World, entity: Entity, text: String) {
  let Some(view) = world.get::<MarkdownView>(entity) else { return };
  if view.source == text {
    return;
  }
  let dense = view.dense;
  if !world.contains_resource::<UiTheme>() {
    return;
  }
  let font = world.get_resource::<ThemeFont>().and_then(|f| f.handle.clone());
  world.resource_scope::<UiTheme, _>(|world, theme| {
    let Ok(mut e) = world.get_entity_mut(entity) else { return };
    let ctx = UiCtx::new(&theme, font.as_ref());
    let r = Renderer { ctx: &ctx, dense };
    e.insert(MarkdownView { source: text.clone(), dense });
    e.despawn_children();
    e.with_children(|p| r.document(p, &text));
  });
}

/// 解析选项（进程级单例：只构造一次）；扩展按渲染能力开子集
fn options() -> &'static Options<'static> {
  static OPTIONS: OnceLock<Options<'static>> = OnceLock::new();
  OPTIONS.get_or_init(|| {
    let mut o = Options::default();
    o.extension.table = true;
    o.extension.strikethrough = true;
    o.extension.tasklist = true;
    o.extension.autolink = true;
    o
  })
}

/// 行内样式（沿子树下传；颜色已解析为主题色）
#[derive(Clone, Copy)]
struct InlineStyle {
  size: f32,
  color: Color,
  weight: FontWeight,
  style: FontStyle,
  bg: Option<Color>,
  underline: bool,
  strike: bool,
}

/// AST → UI 树（`dense` 决定块间距档）
struct Renderer<'a> {
  ctx: &'a UiCtx<'a>,
  dense: bool,
}

impl Renderer<'_> {
  /// 块间距（紧凑模式取 xs 档）
  fn gap(&self) -> f32 {
    let s = &self.ctx.theme.metrics.spacing;
    if self.dense { s.xs } else { s.sm }
  }

  /// 正文基准样式
  fn body(&self, size: f32) -> InlineStyle {
    InlineStyle {
      size,
      color: color_of(&self.ctx.theme.colors.text_body),
      weight: FontWeight::NORMAL,
      style: FontStyle::Normal,
      bg: None,
      underline: false,
      strike: false,
    }
  }

  fn strong(&self, st: InlineStyle) -> InlineStyle {
    InlineStyle {
      weight: FontWeight::BOLD,
      color: color_of(&self.ctx.theme.colors.text_primary),
      ..st
    }
  }

  fn emph(&self, st: InlineStyle) -> InlineStyle {
    InlineStyle {
      style: FontStyle::Italic,
      color: color_of(&self.ctx.theme.colors.text_primary),
      ..st
    }
  }

  fn strike(&self, st: InlineStyle) -> InlineStyle {
    InlineStyle { strike: true, color: color_of(&self.ctx.theme.colors.text_muted), ..st }
  }

  fn code(&self, st: InlineStyle) -> InlineStyle {
    InlineStyle {
      bg: Some(color_of(&self.ctx.theme.colors.surface_overlay)),
      color: color_of(&self.ctx.theme.colors.text_primary),
      ..st
    }
  }

  fn link(&self, st: InlineStyle) -> InlineStyle {
    InlineStyle { underline: true, color: color_of(&self.ctx.theme.colors.accent_text), ..st }
  }

  /// 文档根：块级子节点顺序铺进 `parent`
  fn document(&self, parent: &mut ChildSpawner, md: &str) {
    let arena = Arena::new();
    let root = parse_document(&arena, md, options());
    self.blocks(parent, root);
  }

  fn blocks(&self, parent: &mut ChildSpawner, node: MdNode<'_>) {
    for child in node.children() {
      self.block(parent, child);
    }
  }

  fn block(&self, parent: &mut ChildSpawner, node: MdNode<'_>) {
    match &node.data.borrow().value {
      NodeValue::Paragraph => self.paragraph(parent, node),
      NodeValue::Heading(h) => {
        let level = h.level;
        self.heading(parent, node, level);
      }
      NodeValue::BlockQuote => self.quote(parent, node),
      NodeValue::List(l) => {
        let list = *l;
        self.list(parent, node, list);
      }
      NodeValue::ThematicBreak => self.rule(parent),
      NodeValue::CodeBlock(cb) => {
        let literal = cb.literal.clone();
        self.code_block(parent, &literal);
      }
      NodeValue::HtmlBlock(h) => {
        let literal = h.literal.clone();
        self.code_block(parent, &literal);
      }
      NodeValue::Table(_) => self.table(parent, node),
      // 其余块（元数据 / 脚注 / 描述列表 / 容器指令等）：未开对应扩展或暂不渲染，
      // 但有子块时按块递归，避免整段内容丢失
      _ => self.blocks(parent, node),
    }
  }

  fn paragraph(&self, parent: &mut ChildSpawner, node: MdNode<'_>) {
    let size = self.ctx.theme.metrics.font_size.md;
    self.text_block(parent, node, self.body(size), "ui-md-text");
  }

  fn heading(&self, parent: &mut ChildSpawner, node: MdNode<'_>, level: u8) {
    let fs = &self.ctx.theme.metrics.font_size;
    let size = match level {
      1 | 2 => fs.lg,
      3 | 4 => fs.md,
      _ => fs.sm,
    };
    let mut st = self.body(size);
    st.color = color_of(&self.ctx.theme.colors.text_primary);
    st.weight = FontWeight::BOLD;
    self.text_block(parent, node, st, "ui-md-heading");
  }

  /// 引用块：左侧竖条 + 内缩列
  fn quote(&self, parent: &mut ChildSpawner, node: MdNode<'_>) {
    let m = &self.ctx.theme.metrics;
    parent
      .spawn((
        Name::new("ui-md-quote"),
        Node {
          flex_direction: FlexDirection::Column,
          row_gap: px(self.gap()),
          padding: UiRect { left: px(m.spacing.sm), ..default() },
          border: UiRect::left(px(MD_QUOTE_BAR_W)),
          ..default()
        },
        BorderColor::all(color_of(&self.ctx.theme.colors.border_strong)),
      ))
      .with_children(|p| self.blocks(p, node));
  }

  /// 分割线
  fn rule(&self, parent: &mut ChildSpawner) {
    parent.spawn((
      Name::new("ui-md-rule"),
      Node { width: Val::Percent(100.0), height: px(MD_RULE_H), ..default() },
      BackgroundColor(color_of(&self.ctx.theme.colors.border_strong)),
    ));
  }

  /// 代码块：底色嵌块 + 不换行文本（超出裁切）
  fn code_block(&self, parent: &mut ChildSpawner, literal: &str) {
    let m = &self.ctx.theme.metrics;
    let text = literal.strip_suffix('\n').unwrap_or(literal);
    let st = self.body(m.font_size.sm);
    parent
      .spawn((
        Name::new("ui-md-code"),
        Node {
          padding: UiRect::all(px(m.spacing.sm)),
          border: UiRect::all(px(m.border_width)),
          overflow: Overflow::clip(),
          ..default()
        },
        BackgroundColor(color_of(&self.ctx.theme.colors.surface_overlay)),
        BorderColor::all(color_of(&self.ctx.theme.colors.border_subtle)),
      ))
      .with_children(|p| {
        p.spawn(self.text_bundle(
          "ui-md-code-text",
          Node::default(),
          text.to_string(),
          st,
          TextLayout::no_wrap(),
        ));
      });
  }

  /// 列表：marker 槽定宽（同列表取最长 marker，等宽字体下正文左对齐）+ 内容列
  fn list(&self, parent: &mut ChildSpawner, node: MdNode<'_>, list: NodeList) {
    let size = self.ctx.theme.metrics.font_size.md;
    let items: Vec<(MdNode<'_>, String)> = node
      .children()
      .enumerate()
      .map(|(i, item)| (item, marker_text(item, list, i)))
      .collect();
    let max_marker = items.iter().map(|(_, m)| m.chars().count()).max().unwrap_or(1).max(1);
    let gutter = max_marker as f32 * MD_MONO_ADVANCE_EM * size;
    for (item, marker) in items {
      self.list_item(parent, item, marker, gutter, size);
    }
  }

  fn list_item(
    &self,
    parent: &mut ChildSpawner,
    item: MdNode<'_>,
    marker: String,
    gutter: f32,
    size: f32,
  ) {
    let m = &self.ctx.theme.metrics;
    let mut marker_st = self.body(size);
    marker_st.color = color_of(&self.ctx.theme.colors.text_muted);
    parent
      .spawn((Name::new("ui-md-item"), Node { column_gap: px(m.spacing.xs), ..default() }))
      .with_children(|row| {
        row.spawn(self.text_bundle(
          "ui-md-marker",
          Node { min_width: px(gutter), flex_shrink: 0.0, ..default() },
          marker,
          marker_st,
          TextLayout::justify(Justify::Right),
        ));
        row
          .spawn((
            Name::new("ui-md-item-body"),
            Node {
              flex_direction: FlexDirection::Column,
              row_gap: px(self.gap()),
              flex_grow: 1.0,
              flex_basis: px(0.0),
              ..default()
            },
          ))
          .with_children(|body| self.blocks(body, item));
      });
  }

  /// 表格：行/列 flex，列宽等分（无文本测量预通道）
  fn table(&self, parent: &mut ChildSpawner, node: MdNode<'_>) {
    let c = &self.ctx.theme.colors;
    let m = &self.ctx.theme.metrics;
    parent
      .spawn((
        Name::new("ui-md-table"),
        Node {
          flex_direction: FlexDirection::Column,
          width: Val::Percent(100.0),
          border: UiRect::all(px(m.border_width)),
          ..default()
        },
        BorderColor::all(color_of(&c.border)),
      ))
      .with_children(|t| {
        for row in node.children() {
          let header = matches!(&row.data.borrow().value, NodeValue::TableRow(true));
          let mut st = self.body(m.font_size.sm);
          st.color = color_of(if header { &c.text_primary } else { &c.text_body });
          t.spawn((
            Name::new(if header { "ui-md-head" } else { "ui-md-row" }),
            Node {
              padding: UiRect::all(px(m.spacing.xs)),
              border: UiRect::bottom(px(m.border_width)),
              ..default()
            },
            BackgroundColor(color_of(if header { &c.surface_elevated } else { &c.surface_card })),
            BorderColor::all(color_of(&c.border_subtle)),
          ))
          .with_children(|r| {
            for cell in row.children() {
              r.spawn((
                Name::new("ui-md-cell"),
                Node {
                  flex_direction: FlexDirection::Column,
                  flex_grow: 1.0,
                  flex_basis: px(0.0),
                  ..default()
                },
              ))
              .with_children(|cell_ui| self.text_block(cell_ui, cell, st, "ui-md-text"));
            }
          });
        }
      });
  }

  /// 一个文本块：`Text` 根（空串）+ 若干 `TextSpan` 子（行内样式共享同一文本块）
  fn text_block(
    &self,
    parent: &mut ChildSpawner,
    node: MdNode<'_>,
    st: InlineStyle,
    name: &'static str,
  ) {
    parent
      .spawn(self.text_bundle(name, Node::default(), String::new(), st, TextLayout::default()))
      .with_children(|p| self.inlines(p, node, st));
  }

  fn inlines(&self, parent: &mut ChildSpawner, node: MdNode<'_>, st: InlineStyle) {
    for child in node.children() {
      self.inline(parent, child, st);
    }
  }

  fn inline(&self, parent: &mut ChildSpawner, node: MdNode<'_>, st: InlineStyle) {
    match &node.data.borrow().value {
      NodeValue::Text(t) => self.span(parent, t, st),
      NodeValue::Code(code) => self.span(parent, &code.literal, self.code(st)),
      NodeValue::SoftBreak => self.span(parent, " ", st),
      NodeValue::LineBreak => self.span(parent, "\n", st),
      NodeValue::HtmlInline(html) => self.span(parent, html, st),
      NodeValue::Strong => self.inlines(parent, node, self.strong(st)),
      NodeValue::Emph => self.inlines(parent, node, self.emph(st)),
      NodeValue::Strikethrough => self.inlines(parent, node, self.strike(st)),
      NodeValue::Link(_) => self.inlines(parent, node, self.link(st)),
      // 图片不加载：只展示 alt 文本（无链接语义，不加下划线）
      NodeValue::Image(_) => self.inlines(parent, node, st),
      // 其余行内（强调变体 / 未开扩展的语法）：递归保住文字
      _ => self.inlines(parent, node, st),
    }
  }

  /// 行内片段（空串不建实体）
  fn span(&self, parent: &mut ChildSpawner, text: &str, st: InlineStyle) {
    if text.is_empty() {
      return;
    }
    let mut e = parent.spawn((
      Name::new("ui-md-span"),
      TextSpan::new(text),
      TextFont {
        font: self.ctx.font_source(),
        font_size: bevy::text::FontSize::Px(st.size),
        weight: st.weight,
        style: st.style,
        ..default()
      },
      TextColor(st.color),
    ));
    if let Some(bg) = st.bg {
      e.insert(TextBackgroundColor(bg));
    }
    if st.underline {
      e.insert(Underline);
    }
    if st.strike {
      e.insert(Strikethrough);
    }
  }

  /// 文本实体 bundle（`Label` + `Text` + 主题字体/颜色 + 给定布局）
  fn text_bundle(
    &self,
    name: &'static str,
    node: Node,
    text: String,
    st: InlineStyle,
    layout: TextLayout,
  ) -> impl Bundle {
    (
      Name::new(name),
      Label,
      node,
      Text::new(text),
      TextFont {
        font: self.ctx.font_source(),
        font_size: bevy::text::FontSize::Px(st.size),
        weight: st.weight,
        style: st.style,
        ..default()
      },
      TextColor(st.color),
      layout,
    )
  }
}

/// 列表项 marker 文本：任务项 `[x]`/`[ ]`、有序列表序号、无序列表圆点
fn marker_text(item: MdNode<'_>, list: NodeList, index: usize) -> String {
  match &item.data.borrow().value {
    NodeValue::TaskItem(t) => if t.symbol.is_some() { "[x]" } else { "[ ]" }.to_string(),
    _ => match list.list_type {
      ListType::Ordered => {
        let delim = match list.delimiter {
          ListDelimType::Paren => ')',
          ListDelimType::Period => '.',
        };
        format!("{}{delim}", list.start + index)
      }
      ListType::Bullet => "•".to_string(),
    },
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use bevy::ecs::system::RunSystemOnce;

  /// 无插件 World 里跑一遍渲染，返回 (挂载点, World)；主题作为 Resource 供重建路径取用
  fn render(md: &str) -> (Entity, World) {
    let mut world = World::new();
    world.insert_resource(UiTheme::default());
    let theme = UiTheme::default();
    let ctx = UiCtx::new(&theme, None);
    let mut root = world.spawn_empty();
    let id = root.id();
    root.with_children(|p| {
      markdown(&ctx, p, MarkdownConfig { text: md.to_string(), ..default() });
    });
    (id, world)
  }

  fn children(world: &World, e: Entity) -> Vec<Entity> {
    world.get::<Children>(e).map(|c| c.iter().collect()).unwrap_or_default()
  }

  fn name_of(world: &World, e: Entity) -> String {
    world.get::<Name>(e).map(|n| n.as_str().to_string()).unwrap_or_default()
  }

  /// 渲染后的 markdown 根实体
  fn md_root(world: &World, root: Entity) -> Entity {
    children(world, root)[0]
  }

  fn span_texts(world: &World, block: Entity) -> Vec<String> {
    children(world, block)
      .iter()
      .filter_map(|e| world.get::<TextSpan>(*e).map(|s| s.0.clone()))
      .collect()
  }

  fn span_with(world: &World, block: Entity, text: &str) -> Entity {
    children(world, block)
      .into_iter()
      .find(|e| world.get::<TextSpan>(*e).is_some_and(|s| s.0 == text))
      .expect("span not found")
  }

  #[test]
  fn paragraph_inline_styles() {
    let (root, world) = render("a **b** `c` [d](https://e) ~~f~~");
    let block = children(&world, md_root(&world, root))[0];
    assert_eq!(name_of(&world, block), "ui-md-text");
    assert_eq!(span_texts(&world, block), ["a ", "b", " ", "c", " ", "d", " ", "f"]);

    let colors = UiTheme::default().colors;
    let bold = span_with(&world, block, "b");
    assert_eq!(world.get::<TextColor>(bold).unwrap().0, color_of(&colors.text_primary));
    let code = span_with(&world, block, "c");
    assert!(world.get::<TextBackgroundColor>(code).is_some());
    let link = span_with(&world, block, "d");
    assert!(world.get::<Underline>(link).is_some());
    let strike = span_with(&world, block, "f");
    assert!(world.get::<Strikethrough>(strike).is_some());
  }

  #[test]
  fn block_kinds() {
    let (root, world) = render("# title\n\npara\n\n- a\n- b\n\n> q\n\n```\ncode\n```\n\n---\n");
    let blocks = children(&world, md_root(&world, root));
    let names: Vec<String> = blocks.iter().map(|e| name_of(&world, *e)).collect();
    assert_eq!(
      names,
      [
        "ui-md-heading",
        "ui-md-text",
        "ui-md-item",
        "ui-md-item",
        "ui-md-quote",
        "ui-md-code",
        "ui-md-rule"
      ]
    );
    // 标题字号取 lg 档
    let fs = UiTheme::default().metrics.font_size;
    assert_eq!(world.get::<TextFont>(blocks[0]).unwrap().font_size, bevy::text::FontSize::Px(fs.lg));
    // 引用块内只有一个段落
    assert_eq!(children(&world, blocks[4]).len(), 1);
    // 代码块文本不换行
    let code_text = children(&world, blocks[5])[0];
    assert_eq!(world.get::<Text>(code_text).unwrap().0, "code");
    assert_eq!(world.get::<TextLayout>(code_text).unwrap().linebreak, LineBreak::NoWrap);
  }

  #[test]
  fn ordered_and_bullet_markers() {
    let (root, world) = render("3. x\n4. y\n\n- z\n");
    let blocks = children(&world, md_root(&world, root));
    let marker = |item: Entity| {
      let m = children(&world, item)[0];
      world.get::<Text>(m).map(|t| t.0.clone()).unwrap_or_default()
    };
    assert_eq!(marker(blocks[0]), "3.");
    assert_eq!(marker(blocks[1]), "4.");
    assert_eq!(marker(blocks[2]), "•");
  }

  #[test]
  fn task_items_show_checkbox_marker() {
    let (root, world) = render("- [x] done\n- [ ] todo\n");
    let blocks = children(&world, md_root(&world, root));
    let marker = |item: Entity| {
      let m = children(&world, item)[0];
      world.get::<Text>(m).map(|t| t.0.clone()).unwrap_or_default()
    };
    assert_eq!(marker(blocks[0]), "[x]");
    assert_eq!(marker(blocks[1]), "[ ]");
  }

  #[test]
  fn table_rows_and_cells() {
    let (root, world) = render("| a | b |\n|---|---|\n| c | d |\n");
    let table = children(&world, md_root(&world, root))[0];
    assert_eq!(name_of(&world, table), "ui-md-table");
    let rows = children(&world, table);
    assert_eq!(rows.len(), 2);
    assert_eq!(name_of(&world, rows[0]), "ui-md-head");
    assert_eq!(name_of(&world, rows[1]), "ui-md-row");
    for row in rows {
      assert_eq!(children(&world, row).len(), 2);
    }
  }

  /// 重建：源相同时不动，变化时换掉整棵子树
  #[test]
  fn rebuild_only_on_change() {
    let (root, mut world) = render("a");
    let target = md_root(&world, root);
    let before = children(&world, target);

    world
      .run_system_once(move |mut commands: Commands| markdown_set_text(&mut commands, target, "a"))
      .unwrap();
    world.flush();
    assert_eq!(children(&world, target), before, "源相同不应重建");

    world
      .run_system_once(move |mut commands: Commands| {
        markdown_set_text(&mut commands, target, "**b**")
      })
      .unwrap();
    world.flush();
    assert_eq!(world.get::<MarkdownView>(target).unwrap().source(), "**b**");
    let block = children(&world, target)[0];
    assert_eq!(span_texts(&world, block), ["b"]);
  }
}
