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

#[derive(Clone, Debug, Default, PartialEq)]
pub struct MarkdownConfig {
  pub text: String,
  pub dense: bool,
  pub max_width: Option<f32>,
}

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

pub fn markdown(ctx: &UiCtx, parent: &mut ChildSpawner, config: MarkdownConfig) -> MarkdownHandle {
  let r = Renderer { ctx, dense: config.dense };
  let e = parent
    .spawn((Name::new("ui-markdown"), markdown_root(ctx.theme, &config)))
    .with_children(|p| r.document(p, &config.text))
    .id();
  MarkdownHandle(e)
}

pub fn markdown_set_text(commands: &mut Commands, entity: Entity, text: impl Into<String>) {
  let text = text.into();
  commands.queue(move |world: &mut World| rebuild(world, entity, text));
}

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

struct Renderer<'a> {
  ctx: &'a UiCtx<'a>,
  dense: bool,
}

impl Renderer<'_> {
  fn gap(&self) -> f32 {
    let s = &self.ctx.theme.metrics.spacing;
    if self.dense { s.xs } else { s.sm }
  }

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

  fn rule(&self, parent: &mut ChildSpawner) {
    parent.spawn((
      Name::new("ui-md-rule"),
      Node { width: Val::Percent(100.0), height: px(MD_RULE_H), ..default() },
      BackgroundColor(color_of(&self.ctx.theme.colors.border_strong)),
    ));
  }

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

  fn list(&self, parent: &mut ChildSpawner, node: MdNode<'_>, list: NodeList) {
    let size = self.ctx.theme.metrics.font_size.md;
    let items: Vec<(MdNode<'_>, String)> =
      node.children().enumerate().map(|(i, item)| (item, marker_text(item, list, i))).collect();
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
      NodeValue::Image(_) => self.inlines(parent, node, st),
      _ => self.inlines(parent, node, st),
    }
  }

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