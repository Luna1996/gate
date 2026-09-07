//! tab_view：标签页切换（tab bar + 内容区，选中态全灰度靠亮度差区分）。
//!
//! 结构：
//! ```text
//! root (TabView)
//! ├── tab_bar (横向 flex，每个 tab 带 TabButton{index})
//! │   ├── tab 0  ─── 子 Text
//! │   └── tab 1  ─── 子 Text
//! └── content_container
//!     ├── content 0 (TabContent{index:0})  ← 调用方在此加子节点
//!     └── content 1 (TabContent{index:1})
//! ```
//!
//! 切换由 `tab_view_system` 驱动：
//! - 点击 tab（Pressed→Hovered）→ 更新 `TabView.active`
//! - 选中 tab：背景 surface_elevated + 底部 2px 边框 + 文字 text_primary
//! - 未选中 tab：透明底 + 文字 text_muted（hover 时提亮）
//! - 选中 content：Inherited（跟随祖先显隐）；其余：Hidden

use std::ops::Deref;

use bevy::prelude::*;
use bevy::ui::{FocusPolicy, Interaction};

use super::button::InteractionPrev;
use super::{UiCtx, color_of, px, spawn_label};
use crate::theme::UiTheme;

/// 标签页状态（挂在 root 节点上）
#[derive(Component, Debug)]
pub struct TabView {
  pub active: usize,
  pub count: usize,
}

/// tab 按钮标记（挂在 tab 节点上，index = 对应 content 下标）
#[derive(Component, Debug, Clone, Copy)]
pub struct TabButton {
  pub index: usize,
}

/// tab 内容容器标记（挂在 content 节点上）
#[derive(Component, Debug, Clone, Copy)]
pub struct TabContent {
  pub index: usize,
}

/// 标签页切换事件（用户点击 tab 切换时触发；EntityEvent，target = root 实体）。
/// `TabView.active` 组件仍是真源，事件只是通知。
#[derive(EntityEvent, Clone, Copy, Debug, PartialEq)]
pub struct TabChanged {
  pub entity: Entity,
  /// 切换后选中的 tab 下标
  pub index: usize,
}

/// 标签页句柄：root 实体 + 各 tab 内容容器实体。
/// Deref 到 root 实体；调用方在 `contents[i]` 上添加该 tab 的子节点。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TabViewHandle {
  pub entity: Entity,
  pub contents: Vec<Entity>,
}

impl Deref for TabViewHandle {
  type Target = Entity;
  fn deref(&self) -> &Entity {
    &self.entity
  }
}

/// 标签页配置（全部字段进 Config；Default = 空 tabs、初始选中 0）
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TabConfig {
  /// 标签名列表
  pub tabs: Vec<String>,
  /// 初始选中下标（越界自动钳到最后一个）
  pub active: usize,
}

/// 创建标签页。
///
/// 调用方在返回的 [`TabViewHandle::contents`]`[i]` 上添加该 tab 的子节点。
pub fn tab_view(ctx: &UiCtx, parent: &mut ChildSpawner, config: TabConfig) -> TabViewHandle {
  let c = &ctx.theme.colors;
  let m = &ctx.theme.metrics;
  let count = config.tabs.len();
  let active = config.active.min(count.saturating_sub(1));
  let mut content_entities = Vec::with_capacity(count);

  let root = parent
    .spawn((
      Name::new("ui-tab-view"),
      TabView { active, count },
      Node {
        flex_direction: FlexDirection::Column,
        width: Val::Percent(100.0),
        // 撑满宿主剩余高度：flex 后尺寸视为 definite，子级 Val::Percent 高度才能解析。
        // 否则 root 高度 auto → content 容器塌缩 → 内部 scroll_view(Percent(100%)) 高 0，
        // Overflow::clip 会把全部内容裁掉（在固定高度面板内表现为"tab 里没有控件"）。
        flex_grow: 1.0,
        ..default()
      },
    ))
    .with_children(|root| {
      // ---- tab bar ----
      root
        .spawn((
          Name::new("ui-tab-bar"),
          Node {
            flex_direction: FlexDirection::Row,
            width: Val::Percent(100.0),
            border: UiRect::bottom(px(m.border_width)),
            ..default()
          },
          BorderColor::all(color_of(&c.border)),
        ))
        .with_children(|bar| {
          for (i, name) in config.tabs.iter().enumerate() {
            let is_active = i == active;
            bar
              .spawn((
                Name::new(format!("ui-tab-{i}")),
                TabButton { index: i },
                Interaction::default(),
                InteractionPrev::default(),
                Node {
                  padding: UiRect {
                    left: px(m.spacing.md),
                    right: px(m.spacing.md),
                    top: px(m.spacing.sm),
                    bottom: px(m.spacing.sm),
                  },
                  border: UiRect::bottom(px(if is_active { 2.0 } else { 0.0 })),
                  ..default()
                },
                BackgroundColor(if is_active {
                  color_of(&c.surface_elevated)
                } else {
                  Color::NONE
                }),
                BorderColor::all(if is_active {
                  color_of(&c.text_primary)
                } else {
                  Color::NONE
                }),
                FocusPolicy::Block,
              ))
              .with_children(|tab| {
                spawn_label(
                  ctx,
                  tab,
                  name.clone(),
                  m.font_size.sm,
                  if is_active {
                    color_of(&c.text_primary)
                  } else {
                    color_of(&c.text_muted)
                  },
                );
              });
          }
        });
      // ---- content container ----
      root
        .spawn((
          Name::new("ui-tab-content-container"),
          Node {
            flex_direction: FlexDirection::Column,
            width: Val::Percent(100.0),
            flex_grow: 1.0,
            // 页面绝对定位叠放，溢出由页内 scroll_view 裁剪，此处兜底
            overflow: Overflow::clip(),
            ..default()
          },
        ))
        .with_children(|container| {
          for i in 0..count {
            let e = container
              .spawn((
                Name::new(format!("ui-tab-content-{i}")),
                TabContent { index: i },
                // 绝对定位叠放：所有页面同位重叠、脱离流布局——
                // 隐藏页不占位，切页时内容位置不随页序漂移
                Node {
                  position_type: PositionType::Absolute,
                  top: px(0.0),
                  left: px(0.0),
                  width: Val::Percent(100.0),
                  height: Val::Percent(100.0),
                  flex_direction: FlexDirection::Column,
                  ..default()
                },
                if i == active {
                  // Inherited（非 Visible！）：Visible 会无视祖先强制可见，
                  // 面板被整体隐藏时页面会单独悬浮（propagate_recursive 遇
                  // Visible 直接置 true，覆盖父级 Hidden）
                  Visibility::Inherited
                } else {
                  Visibility::Hidden
                },
              ))
              .id();
            content_entities.push(e);
          }
        });
    })
    .id();

  TabViewHandle {
    entity: root,
    contents: content_entities,
  }
}

/// 标签页状态机：点击 tab 切换 active（触发 [`TabChanged`]）+ tab 视觉 + content 可见性
pub fn tab_view_system(
  mut commands: Commands,
  mut q_views: Query<(Entity, &mut TabView, &Children)>,
  mut q_tab: Query<(&TabButton, &Interaction, &mut InteractionPrev)>,
  mut q_tab_node: Query<(&mut BackgroundColor, &mut BorderColor, &Children)>,
  mut q_tab_text: Query<&mut TextColor>,
  q_children: Query<&Children>,
  mut q_content: Query<(&TabContent, &mut Visibility)>,
  theme: Option<Res<UiTheme>>,
) {
  let Some(theme) = theme else {
    return;
  };
  let c = &theme.colors;

  for (view_e, mut view, view_children) in &mut q_views {
    let Some(&bar) = view_children.get(0) else {
      continue;
    };
    let Some(&content_container) = view_children.get(1) else {
      continue;
    };
    let Ok(bar_children) = q_children.get(bar) else {
      continue;
    };

    // 阶段 1：检测点击，更新 active（需要 mut InteractionPrev）
    for tab_e in bar_children.iter() {
      let Ok((tab_btn, inter, mut prev)) = q_tab.get_mut(tab_e) else {
        continue;
      };
      if prev.0 == Interaction::Pressed
        && *inter == Interaction::Hovered
        && view.active != tab_btn.index
      {
        view.active = tab_btn.index;
        commands.trigger(TabChanged {
          entity: view_e,
          index: tab_btn.index,
        });
      }
      prev.0 = *inter;
    }

    // 阶段 2：更新 tab 视觉（只读 InteractionPrev，写 BackgroundColor/BorderColor/TextColor）
    for tab_e in bar_children.iter() {
      let Ok((tab_btn, inter, _)) = q_tab.get(tab_e) else {
        continue;
      };
      let is_active = tab_btn.index == view.active;
      let hovered = *inter != Interaction::None;
      let Ok((mut bg, mut bc, tab_children)) = q_tab_node.get_mut(tab_e) else {
        continue;
      };
      let target_bg = if is_active {
        color_of(&c.surface_elevated)
      } else if hovered {
        color_of(&c.surface_card)
      } else {
        Color::NONE
      };
      bg.0 = target_bg;
      let target_border = if is_active {
        color_of(&c.text_primary)
      } else {
        Color::NONE
      };
      *bc = BorderColor::all(target_border);
      let text_color = if is_active {
        color_of(&c.text_primary)
      } else if hovered {
        color_of(&c.text_body)
      } else {
        color_of(&c.text_muted)
      };
      for child in tab_children.iter() {
        if let Ok(mut tc) = q_tab_text.get_mut(child) {
          tc.0 = text_color;
        }
      }
    }

    // 阶段 3：更新 content 可见性
    let Ok(cc_children) = q_children.get(content_container) else {
      continue;
    };
    for content_e in cc_children.iter() {
      if let Ok((tc, mut vis)) = q_content.get_mut(content_e) {
        // 选中页 = Inherited（跟随祖先，面板整体隐藏时页面跟着隐藏）；
        // 显式 Visible 会无视祖先强制可见 → 页面单独悬浮
        *vis = if tc.index == view.active {
          Visibility::Inherited
        } else {
          Visibility::Hidden
        };
      }
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::theme::default_theme;

  fn cfg() -> TabConfig {
    TabConfig {
      tabs: vec!["a".into(), "b".into()],
      active: 0,
    }
  }

  #[test]
  fn tab_view_spawn_structure() {
    let theme = default_theme();
    let ctx = UiCtx::new(&theme, None);
    let mut app = App::new();
    let root = app.world_mut().spawn_empty().id();
    let mut res = None;
    app.world_mut().entity_mut(root).with_children(|p| {
      res = Some(tab_view(&ctx, p, cfg()));
    });
    let h = res.expect("tab_view spawned");
    let w = app.world();
    assert_eq!(h.contents.len(), 2, "two content containers");
    let view = w.get::<TabView>(*h).unwrap();
    assert_eq!(view.active, 0);
    assert_eq!(view.count, 2);
    assert_eq!(
      *w.get::<Visibility>(h.contents[0]).unwrap(),
      Visibility::Inherited,
      "active content inherits (Visible would ignore ancestor Hidden)"
    );
    assert_eq!(
      *w.get::<Visibility>(h.contents[1]).unwrap(),
      Visibility::Hidden
    );
  }

  #[test]
  fn tab_click_switches_active_and_emits_event() {
    use std::sync::{Arc, Mutex};

    let theme = default_theme();
    let mut app = App::new();
    app.insert_resource(theme.clone());
    app.add_systems(Update, tab_view_system);
    let changes = Arc::new(Mutex::new(Vec::<(Entity, usize)>::new()));
    let sink = changes.clone();
    app.add_observer(move |ev: On<TabChanged>| {
      sink.lock().unwrap().push((ev.entity, ev.index));
    });

    let root = app.world_mut().spawn_empty().id();
    let mut res = None;
    let ctx = UiCtx::new(&theme, None);
    app.world_mut().entity_mut(root).with_children(|p| {
      res = Some(tab_view(&ctx, p, cfg()));
    });
    let h = res.unwrap();

    let w = app.world();
    let view_children = w.get::<Children>(*h).unwrap();
    let bar = view_children[0];
    let bar_children = w.get::<Children>(bar).unwrap();
    let tab1 = bar_children[1];

    app
      .world_mut()
      .get_mut::<Interaction>(tab1)
      .unwrap()
      .set_if_neq(Interaction::Pressed);
    app.update();
    app
      .world_mut()
      .get_mut::<Interaction>(tab1)
      .unwrap()
      .set_if_neq(Interaction::Hovered);
    app.update();

    let w = app.world();
    assert_eq!(
      w.get::<TabView>(*h).unwrap().active,
      1,
      "active switched to 1"
    );
    assert_eq!(
      *changes.lock().unwrap(),
      vec![(*h, 1)],
      "switch emits TabChanged(1)"
    );
    assert_eq!(
      *w.get::<Visibility>(h.contents[0]).unwrap(),
      Visibility::Hidden,
      "content 0 hidden"
    );
    assert_eq!(
      *w.get::<Visibility>(h.contents[1]).unwrap(),
      Visibility::Inherited,
      "content 1 active and inheriting"
    );

    // 点击已选中的 tab：不重复发事件
    let changes_before = changes.lock().unwrap().len();
    app
      .world_mut()
      .get_mut::<Interaction>(tab1)
      .unwrap()
      .set_if_neq(Interaction::Pressed);
    app.update();
    app
      .world_mut()
      .get_mut::<Interaction>(tab1)
      .unwrap()
      .set_if_neq(Interaction::Hovered);
    app.update();
    assert_eq!(
      changes.lock().unwrap().len(),
      changes_before,
      "re-click active tab emits no event"
    );
  }
}
