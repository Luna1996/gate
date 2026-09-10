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
use super::{UiCtx, UiDisabled, color_of, dim_color, px, spawn_label};
use crate::theme::UiTheme;

/// 标签页状态（挂在 root 节点上）
#[derive(Component, Debug)]
pub struct TabView {
  pub active: usize,
  pub count: usize,
  /// true = 自适应高度模式：活动页流入布局撑开容器，切页面板高度随内容变化；
  /// false = 填充模式：root flex_grow 撑满宿主剩余高，页面绝对定位叠放
  pub fit_content: bool,
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

/// 标签页配置（全部字段进 Config；Default = 空 tabs、初始选中 0、填充模式、无禁用 tab）
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TabConfig {
  /// 标签名列表
  pub tabs: Vec<String>,
  /// 初始选中下标（越界自动钳到最后一个）
  pub active: usize,
  /// 高度模式：
  /// - false（默认）= 填充模式：root `flex_grow:1` 撑满宿主剩余高度，页面绝对定位
  ///   100% 高叠放（宿主必须有确定高度，页内用 scroll_view 滚动）；
  /// - true = 自适应高度：root/容器高度 auto，活动页 Relative 流入布局撑开容器，
  ///   隐藏页 Absolute 脱流不占位——切页时面板高度随活动页内容变化。
  ///   页内不可放 Percent(100%) 高的 scroll_view（auto 高度下解析为 0）
  pub fit_content: bool,
  /// 禁用的 tab 下标列表：这些 tab 不可点击切换、配色暗一档。
  /// 若当前 active 恰在禁用列表中，仍显示其内容（禁用只阻止切换，不强制切走）。
  pub disabled_tabs: Vec<usize>,
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
      TabView {
        active,
        count,
        fit_content: config.fit_content,
      },
      Node {
        flex_direction: FlexDirection::Column,
        width: Val::Percent(100.0),
        // 填充模式：撑满宿主剩余高度——flex 后尺寸视为 definite，子级 Val::Percent
        // 高度才能解析（否则 root 高度 auto → content 容器塌缩 → 内部
        // scroll_view(Percent(100%)) 高 0，Overflow::clip 裁掉全部内容）。
        // 自适应模式：不 grow，高度 auto 随活动页内容收缩。
        flex_grow: if config.fit_content { 0.0 } else { 1.0 },
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
            let is_disabled = config.disabled_tabs.contains(&i);
            let mut tab_ec = bar.spawn((
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
            ));
            if is_disabled {
              tab_ec.insert(UiDisabled);
            }
            tab_ec.with_children(|tab| {
              spawn_label(
                ctx,
                tab,
                name.clone(),
                m.font_size.sm,
                if is_active {
                  color_of(&c.text_primary)
                } else if is_disabled {
                  dim_color(color_of(&c.text_muted))
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
            // 填充模式 grow 撑满 tab bar 以下剩余高度；自适应模式 auto 随活动页
            flex_grow: if config.fit_content { 0.0 } else { 1.0 },
            // 页面绝对定位叠放，溢出由页内 scroll_view 裁剪，此处兜底
            overflow: Overflow::clip(),
            ..default()
          },
        ))
        .with_children(|container| {
          for i in 0..count {
            let is_active = i == active;
            // 填充模式：全部页面绝对定位叠放（同位重叠、脱离流布局，隐藏页不
            // 占位，切页内容位置不漂移），高度 100% 吃满容器。
            // 自适应模式：活动页 Relative 流入布局撑开容器高度；隐藏页 Absolute
            // 脱流不占位（无显式高度，随自身内容但不可见）。
            let in_flow = config.fit_content && is_active;
            let e = container
              .spawn((
                Name::new(format!("ui-tab-content-{i}")),
                TabContent { index: i },
                Node {
                  position_type: if in_flow {
                    PositionType::Relative
                  } else {
                    PositionType::Absolute
                  },
                  top: px(0.0),
                  left: px(0.0),
                  width: Val::Percent(100.0),
                  height: if config.fit_content {
                    Val::Auto
                  } else {
                    Val::Percent(100.0)
                  },
                  flex_direction: FlexDirection::Column,
                  ..default()
                },
                if is_active {
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
///
/// Disabled tab（带 [`UiDisabled`]）：跳过点击切换、配色降亮一档；若 active 恰为
/// 禁用 tab，仍显示其内容（禁用只阻止切换，不强制切走）。
pub fn tab_view_system(
  mut commands: Commands,
  mut q_views: Query<(Entity, &mut TabView, &Children)>,
  mut q_tab: Query<(
    &TabButton,
    &Interaction,
    &mut InteractionPrev,
    Has<UiDisabled>,
  )>,
  mut q_tab_node: Query<(
    &mut BackgroundColor,
    &mut BorderColor,
    &Children,
    Has<UiDisabled>,
  )>,
  mut q_tab_text: Query<&mut TextColor>,
  q_children: Query<&Children>,
  mut q_content: Query<(&TabContent, &mut Visibility, &mut Node)>,
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
      let Ok((tab_btn, inter, mut prev, disabled)) = q_tab.get_mut(tab_e) else {
        continue;
      };
      if !disabled
        && prev.0 == Interaction::Pressed
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
      let Ok((tab_btn, inter, _, _)) = q_tab.get(tab_e) else {
        continue;
      };
      let is_active = tab_btn.index == view.active;
      let Ok((mut bg, mut bc, tab_children, disabled)) = q_tab_node.get_mut(tab_e) else {
        continue;
      };
      let hovered = !disabled && *inter != Interaction::None;
      let target_bg = if is_active {
        color_of(&c.surface_elevated)
      } else if hovered {
        color_of(&c.surface_card)
      } else {
        Color::NONE
      };
      let target_bg = if disabled && target_bg != Color::NONE {
        dim_color(target_bg)
      } else {
        target_bg
      };
      bg.0 = target_bg;
      let target_border = if is_active {
        color_of(&c.text_primary)
      } else {
        Color::NONE
      };
      let target_border = if disabled && target_border != Color::NONE {
        dim_color(target_border)
      } else {
        target_border
      };
      *bc = BorderColor::all(target_border);
      let text_color = if is_active {
        color_of(&c.text_primary)
      } else if hovered {
        color_of(&c.text_body)
      } else {
        color_of(&c.text_muted)
      };
      let text_color = if disabled {
        dim_color(text_color)
      } else {
        text_color
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
      if let Ok((tc, mut vis, mut node)) = q_content.get_mut(content_e) {
        let is_active = tc.index == view.active;
        // 选中页 = Inherited（跟随祖先，面板整体隐藏时页面跟着隐藏）；
        // 显式 Visible 会无视祖先强制可见 → 页面单独悬浮
        *vis = if is_active {
          Visibility::Inherited
        } else {
          Visibility::Hidden
        };
        // 自适应模式：活动页必须流入布局（Relative）才能撑开容器高度；
        // 隐藏页脱流（Absolute）不占位。填充模式全部 Absolute，不动。
        if view.fit_content {
          node.position_type = if is_active {
            PositionType::Relative
          } else {
            PositionType::Absolute
          };
        }
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
      fit_content: false,
      ..default()
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

  #[test]
  fn fit_content_mode_layout_and_switch() {
    let theme = default_theme();
    let mut app = App::new();
    app.insert_resource(theme.clone());
    app.add_systems(Update, tab_view_system);

    let root = app.world_mut().spawn_empty().id();
    let mut res = None;
    let ctx = UiCtx::new(&theme, None);
    app.world_mut().entity_mut(root).with_children(|p| {
      res = Some(tab_view(
        &ctx,
        p,
        TabConfig {
          tabs: vec!["a".into(), "b".into()],
          active: 0,
          fit_content: true,
          ..default()
        },
      ));
    });
    let h = res.unwrap();
    let w = app.world();
    assert!(w.get::<TabView>(*h).unwrap().fit_content);
    assert_eq!(
      w.get::<Node>(*h).unwrap().flex_grow,
      0.0,
      "fit-content root does not grow (auto height)"
    );
    // 活动页流入布局（Relative + auto 高），隐藏页脱流（Absolute）
    assert_eq!(
      w.get::<Node>(h.contents[0]).unwrap().position_type,
      PositionType::Relative,
      "active page in flow"
    );
    assert_eq!(
      w.get::<Node>(h.contents[0]).unwrap().height,
      Val::Auto,
      "active page height auto"
    );
    assert_eq!(
      w.get::<Node>(h.contents[1]).unwrap().position_type,
      PositionType::Absolute,
      "inactive page out of flow"
    );

    // 切到 tab 1：位置类型随可见性翻转
    let bar = w.get::<Children>(*h).unwrap()[0];
    let tab1 = w.get::<Children>(bar).unwrap()[1];
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
      w.get::<Node>(h.contents[0]).unwrap().position_type,
      PositionType::Absolute,
      "page 0 now out of flow"
    );
    assert_eq!(
      w.get::<Node>(h.contents[1]).unwrap().position_type,
      PositionType::Relative,
      "page 1 now in flow"
    );
  }
}
