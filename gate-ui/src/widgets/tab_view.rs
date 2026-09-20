//! tab_view：标签页切换（tab bar + 内容区，选中态靠亮度差区分）。
//! 结构：root(TabView) → tab_bar(TabButton{index}) + content_container → content(TabContent{index})。
//! `tab_view_system` 驱动；选中 tab = surface_elevated 底 + 2px 底边框 + text_primary，其余透明 + text_muted。

use std::ops::Deref;

use bevy::picking::hover::Hovered;
use bevy::prelude::*;
use bevy::ui::Pressed;

use super::{UiCtx, UiDisabled, color_of, dim_color, px, spawn_label};
use crate::pointer::{UiInteract, UiInteractBundle, UiInteractPrev};
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

/// 标签页切换事件（用户点击 tab 切换时触发；EntityEvent，target = root 实体；`TabView.active` 仍是真源）。
#[derive(EntityEvent, Clone, Copy, Debug, PartialEq)]
pub struct TabChanged {
  pub entity: Entity,
  /// 切换后选中的 tab 下标
  pub index: usize,
}

/// 标签页句柄：root 实体 + 各 tab 内容容器实体；Deref 到 root，调用方在 `contents[i]` 上添加该 tab 的子节点。
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
  /// 高度模式：false（默认）= 填充模式，root `flex_grow:1` 撑满宿主剩余高度，页面绝对定位 100% 高叠放
  /// （宿主须有确定高度）；true = 自适应高度，活动页 Relative 流入撑开容器，隐藏页 Absolute 脱流不占位
  /// （页内不可放 Percent(100%) 高的 scroll_view，auto 高度下解析为 0）。
  pub fit_content: bool,
  /// 禁用的 tab 下标列表：这些 tab 不可点击切换、配色暗一档。
  /// 若当前 active 恰在禁用列表中，仍显示其内容（禁用只阻止切换，不强制切走）。
  pub disabled_tabs: Vec<usize>,
}

/// 创建标签页；调用方在返回的 `TabViewHandle::contents[i]` 上添加该 tab 的子节点。
pub fn tab_view(ctx: &UiCtx, parent: &mut ChildSpawner, config: TabConfig) -> TabViewHandle {
  let c = &ctx.theme.colors;
  let m = &ctx.theme.metrics;
  let count = config.tabs.len();
  let active = config.active.min(count.saturating_sub(1));
  let mut content_entities = Vec::with_capacity(count);

  let root = parent
    .spawn((
      Name::new("ui-tab-view"),
      TabView { active, count, fit_content: config.fit_content },
      Node {
        flex_direction: FlexDirection::Column,
        width: Val::Percent(100.0),
        // 填充模式 flex_grow=1：flex 后尺寸视为 definite，子级 Percent 高度才能解析；
        // 自适应模式不 grow，高度 auto 随活动页内容收缩。
        flex_grow: if config.fit_content { 0.0 } else { 1.0 },
        ..default()
      },
    ))
    .with_children(|root| {
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
              UiInteractBundle::default(),
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
              BackgroundColor(if is_active { color_of(&c.surface_elevated) } else { Color::NONE }),
              BorderColor::all(if is_active { color_of(&c.text_primary) } else { Color::NONE }),
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
            // 填充模式：全部页面绝对定位叠放（脱流、隐藏页不占位），高度 100% 吃满容器。
            // 自适应模式：活动页 Relative 流入布局撑开容器高度，隐藏页 Absolute 脱流不占位。
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
                  height: if config.fit_content { Val::Auto } else { Val::Percent(100.0) },
                  flex_direction: FlexDirection::Column,
                  ..default()
                },
                if is_active {
                  // Inherited（非 Visible）：Visible 会无视祖先强制可见（propagate_recursive 遇 Visible 直接置 true）
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

  TabViewHandle { entity: root, contents: content_entities }
}

/// 标签页状态机：点击 tab 切换 active（触发 `TabChanged`）+ tab 视觉 + content 可见性。
/// Disabled tab（带 `UiDisabled`）：跳过点击切换、配色降亮；active 恰为禁用 tab 时仍显示其内容。
#[allow(clippy::too_many_arguments, clippy::type_complexity)] // Bevy system：各 Query 逐一注入
pub fn tab_view_system(
  mut commands: Commands,
  mut q_views: Query<(Entity, &mut TabView, &Children)>,
  mut q_tab: Query<(&TabButton, &Hovered, Has<Pressed>, &mut UiInteractPrev, Has<UiDisabled>)>,
  mut q_tab_node: Query<(&mut BackgroundColor, &mut BorderColor, &Children, Has<UiDisabled>)>,
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
    let Some(&bar) = view_children.first() else {
      continue;
    };
    let Some(&content_container) = view_children.get(1) else {
      continue;
    };
    let Ok(bar_children) = q_children.get(bar) else {
      continue;
    };

    // 阶段 1：检测点击更新 active（需 mut UiInteractPrev）
    for tab_e in bar_children.iter() {
      let Ok((tab_btn, hovered, pressed, mut prev, disabled)) = q_tab.get_mut(tab_e) else {
        continue;
      };
      let inter = UiInteract::of(hovered, pressed);
      if !disabled
        && prev.0 == UiInteract::Pressed
        && inter == UiInteract::Hovered
        && view.active != tab_btn.index
      {
        view.active = tab_btn.index;
        commands.trigger(TabChanged { entity: view_e, index: tab_btn.index });
      }
      prev.0 = inter;
    }

    // 阶段 2：更新 tab 视觉（只读 UiInteractPrev）
    for tab_e in bar_children.iter() {
      let Ok((tab_btn, hovered, pressed, _, _)) = q_tab.get(tab_e) else {
        continue;
      };
      let inter = UiInteract::of(hovered, pressed);
      let is_active = tab_btn.index == view.active;
      let Ok((mut bg, mut bc, tab_children, disabled)) = q_tab_node.get_mut(tab_e) else {
        continue;
      };
      let hovered = !disabled && inter.is_active();
      let target_bg = if is_active {
        color_of(&c.surface_elevated)
      } else if hovered {
        color_of(&c.surface_card)
      } else {
        Color::NONE
      };
      let target_bg =
        if disabled && target_bg != Color::NONE { dim_color(target_bg) } else { target_bg };
      bg.0 = target_bg;
      let target_border = if is_active { color_of(&c.text_primary) } else { Color::NONE };
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
      let text_color = if disabled { dim_color(text_color) } else { text_color };
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
        // 选中页 = Inherited（跟随祖先）；显式 Visible 会无视祖先强制可见
        *vis = if is_active { Visibility::Inherited } else { Visibility::Hidden };
        // 自适应模式：活动页 Relative 流入布局撑开容器，隐藏页 Absolute 脱流；填充模式全 Absolute。
        if view.fit_content {
          node.position_type =
            if is_active { PositionType::Relative } else { PositionType::Absolute };
        }
      }
    }
  }
}
