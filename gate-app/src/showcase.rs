//! 右上角 gate-ui 组件展示窗：全部 widget 的功能演示面板。
//!
//! [`spawn_showcase`] 在主题/字体就绪后由 `demo_ui_setup` 一次性调用；
//! 交互（button/checkbox/slider）经观察者写入事件日志，[`showcase_demo_system`]
//! 每帧给演示折线喂正弦样本。

use bevy::prelude::*;

use gate_ui::{
  UiCtx,
  widgets::{
    ButtonConfig, ButtonVariant, CheckboxConfig, CheckboxToggled, GridConfig, LabelConfig,
    LabelStyle, ListConfig, PanelConfig, PanelSurface, PlotConfig, PlotData, PlotDomain,
    PlotLayout, RingList, ScrollConfig, SliderConfig, SliderValueChanged, TabConfig, TableConfig,
    UiClick, blank_plot_image, button, checkbox, color_of, grid, grid_cell, label, list, panel,
    plot, px, scroll_view, slider, splitter, tab_view, table,
  },
};

// ---- 展示窗交互标记 ----

#[derive(Component)]
struct ShowcaseButton(&'static str);

#[derive(Component)]
struct ShowcaseCheckbox {
  label: &'static str,
}

#[derive(Component)]
struct ShowcaseSliderValue;

#[derive(Component)]
pub(crate) struct ShowcasePlot;

#[derive(Component)]
struct ShowcaseLog;

/// 展示窗折线画布尺寸（透明纹理，面板底色透出）
const SHOWCASE_PLOT_W: u32 = 216;
const SHOWCASE_PLOT_H: u32 = 48;
/// 展示窗折线容量（60fps 下约 12s 窗口，0.1s 喂一个样本）
const SHOWCASE_PLOT_CAP: usize = 128;

/// 右上角展示面板根标记（debug overlay 的 toggle 以此定位整体显隐）
#[derive(Component)]
pub(crate) struct ShowcaseRoot;

/// 右上角 hud-showcase-panel（tabview：组件/数据 两页，每页包 scrollview），
/// 在 commands.queue 闭包内调用（直接操作 World）
pub(crate) fn spawn_showcase(world: &mut World, ctx: &UiCtx) {
  let c = &ctx.theme.colors;
  let m = &ctx.theme.metrics;
  let showcase_plot_image = world
    .resource_mut::<Assets<Image>>()
    .add(blank_plot_image(SHOWCASE_PLOT_W, SHOWCASE_PLOT_H));
  world
    .spawn((
      Name::new("hud-showcase-panel"),
      ShowcaseRoot,
      Node {
        position_type: PositionType::Absolute,
        right: px(8.0),
        top: px(8.0),
        width: px(280.0),
        // 限制高度，内容超出靠内部 scroll view 滚动
        height: Val::Vh(88.0),
        flex_direction: FlexDirection::Column,
        border: UiRect::all(px(m.border_width)),
        ..default()
      },
      BackgroundColor(color_of(&c.surface_card_hud)),
      BorderColor::all(color_of(&c.border)),
      // 默认隐藏（toggle 初始未勾选，两处状态一致；观察者按 ToggleSwitchToggled 翻转）
      Visibility::Hidden,
    ))
    .with_children(|root| {
      // 最外层 tabview：组件 / 数据 两个标签页，每页内容包在 scrollview 里
      let tv = tab_view(
        ctx,
        root,
        TabConfig {
          tabs: vec!["组件".into(), "数据".into()],
          active: 0,
        },
      );
      // ---- tab 0：组件展示 ----
      root
        .world_mut()
        .entity_mut(tv.contents[0])
        .with_children(|p| {
          let sv = scroll_view(
            ctx,
            p,
            ScrollConfig {
              height: Val::Percent(100.0),
            },
          );
          p.world_mut().entity_mut(sv.content).with_children(|root| {
            label(
              ctx,
              root,
              LabelConfig {
                text: "UI 组件展示".into(),
                style: LabelStyle::Title,
              },
            );
            label(
              ctx,
              root,
              LabelConfig {
                text: "gate-ui widget showcase".into(),
                style: LabelStyle::Muted,
              },
            );

            // ---- buttons：primary / secondary / ghost ----
            label(
              ctx,
              root,
              LabelConfig {
                text: "buttons".into(),
                style: LabelStyle::Muted,
              },
            );
            root
              .spawn((
                Name::new("showcase-button-row"),
                Node {
                  column_gap: px(m.spacing.xs),
                  ..default()
                },
              ))
              .with_children(|row| {
                let b1 = button(
                  ctx,
                  row,
                  ButtonConfig {
                    text: "primary".into(),
                    ..default()
                  },
                );
                row
                  .world_mut()
                  .entity_mut(*b1)
                  .insert(ShowcaseButton("primary"));
                let b2 = button(
                  ctx,
                  row,
                  ButtonConfig {
                    text: "secondary".into(),
                    variant: ButtonVariant::Secondary,
                  },
                );
                row
                  .world_mut()
                  .entity_mut(*b2)
                  .insert(ShowcaseButton("secondary"));
                let b3 = button(
                  ctx,
                  row,
                  ButtonConfig {
                    text: "ghost".into(),
                    variant: ButtonVariant::Ghost,
                  },
                );
                row
                  .world_mut()
                  .entity_mut(*b3)
                  .insert(ShowcaseButton("ghost"));
              });
            // ---- danger button（破坏性操作） ----
            let b4 = button(
              ctx,
              root,
              ButtonConfig {
                text: "danger / destructive".into(),
                variant: ButtonVariant::Danger,
              },
            );
            root
              .world_mut()
              .entity_mut(*b4)
              .insert(ShowcaseButton("danger"));

            // ---- checkbox ----
            label(
              ctx,
              root,
              LabelConfig {
                text: "checkbox".into(),
                style: LabelStyle::Muted,
              },
            );
            let cb1 = checkbox(
              ctx,
              root,
              CheckboxConfig {
                text: Some("选项 A（默认勾选）".into()),
                checked: true,
              },
            );
            root
              .world_mut()
              .entity_mut(*cb1)
              .insert(ShowcaseCheckbox { label: "A" });
            let cb2 = checkbox(
              ctx,
              root,
              CheckboxConfig {
                text: Some("选项 B".into()),
                checked: false,
              },
            );
            root
              .world_mut()
              .entity_mut(*cb2)
              .insert(ShowcaseCheckbox { label: "B" });

            // ---- slider：滑杆 + 实时值 ----
            label(
              ctx,
              root,
              LabelConfig {
                text: "slider（0..100, step 5）".into(),
                style: LabelStyle::Muted,
              },
            );
            root
              .spawn((
                Name::new("showcase-slider-row"),
                Node {
                  column_gap: px(m.spacing.sm),
                  align_items: AlignItems::Center,
                  ..default()
                },
              ))
              .with_children(|row| {
                let s = slider(
                  ctx,
                  row,
                  SliderConfig {
                    min: 0.0,
                    max: 100.0,
                    value: 40.0,
                    step: Some(5.0),
                  },
                );
                {
                  let w = row.world_mut();
                  if let Some(mut node) = w.get_mut::<Node>(*s) {
                    node.flex_grow = 1.0;
                  }
                }
                let v = label(
                  ctx,
                  row,
                  LabelConfig {
                    text: " 40.0".into(),
                    ..default()
                  },
                );
                row.world_mut().entity_mut(*v).insert(ShowcaseSliderValue);
              });

            // ---- plot：演示折线（正弦波喂入） ----
            label(
              ctx,
              root,
              LabelConfig {
                text: "plot（折线 + 纵轴标签）".into(),
                style: LabelStyle::Muted,
              },
            );
            let p = plot(
              ctx,
              root,
              PlotConfig {
                layout: PlotLayout::YAxis,
                image: showcase_plot_image,
                capacity: SHOWCASE_PLOT_CAP,
                y_domain: PlotDomain::Fixed(-1.0, 1.0),
                line_color: color_of(&c.accent_text),
                y_axis_width: Val::Auto,
                canvas_width: Val::Auto,
                unit: None,
                canvas_h: SHOWCASE_PLOT_H as f32,
              },
            );
            root.world_mut().entity_mut(*p).insert(ShowcasePlot);

            // ---- RingList：事件日志（L2 抬升嵌块内） ----
            label(
              ctx,
              root,
              LabelConfig {
                text: "list / RingList（事件日志，cap 6）".into(),
                style: LabelStyle::Muted,
              },
            );
            let inner_panel = panel(
              ctx,
              root,
              PanelConfig {
                surface: PanelSurface::Elevated,
              },
            );
            root
              .world_mut()
              .entity_mut(*inner_panel)
              .with_children(|inner| {
                let l = list(ctx, inner, ListConfig { capacity: 6 });
                let w = inner.world_mut();
                w.entity_mut(*l).insert(ShowcaseLog);
                // 种子日志在 spawn 处写入（组件真源初始化），无需系统里的 seeded 标志
                if let Some(mut ring) = w.get_mut::<RingList>(*l) {
                  ring.push("showcase ready");
                  ring.push("try every widget");
                }
              });

            // ---- 语义色文字 ----
            root
              .spawn((
                Name::new("showcase-semantic-row"),
                Node {
                  column_gap: px(m.spacing.sm),
                  ..default()
                },
              ))
              .with_children(|row| {
                label(
                  ctx,
                  row,
                  LabelConfig {
                    text: "success".into(),
                    style: LabelStyle::Success,
                  },
                );
                label(
                  ctx,
                  row,
                  LabelConfig {
                    text: "warning".into(),
                    style: LabelStyle::Warning,
                  },
                );
                label(
                  ctx,
                  row,
                  LabelConfig {
                    text: "danger".into(),
                    style: LabelStyle::Danger,
                  },
                );
                label(
                  ctx,
                  row,
                  LabelConfig {
                    text: "accent".into(),
                    style: LabelStyle::Accent,
                  },
                );
              });

            // ---- faint 档（仅 ≥18px 大号标签使用） ----
            label(
              ctx,
              root,
              LabelConfig {
                text: "faint: lg 18px only".into(),
                style: LabelStyle::FaintLg,
              },
            );

            // ---- splitter：分割线（按父容器方向自动横/竖） ----
            label(
              ctx,
              root,
              LabelConfig {
                text: "splitter（分割线）".into(),
                style: LabelStyle::Muted,
              },
            );
            // 父为 Column → 横线
            splitter(ctx, root);
            root
              .spawn((
                Name::new("showcase-splitter-row"),
                Node {
                  column_gap: px(m.spacing.sm),
                  ..default()
                },
              ))
              .with_children(|row| {
                label(
                  ctx,
                  row,
                  LabelConfig {
                    text: "左".into(),
                    ..default()
                  },
                );
                // 父为 Row → 竖线
                splitter(ctx, row);
                label(
                  ctx,
                  row,
                  LabelConfig {
                    text: "右".into(),
                    ..default()
                  },
                );
              });

            // ---- grid：格线连通（gap 填色 trick，border-collapse 等效） ----
            label(
              ctx,
              root,
              LabelConfig {
                text: "grid（3 列，格线十字连通）".into(),
                style: LabelStyle::Muted,
              },
            );
            let g = grid(
              ctx,
              root,
              GridConfig {
                columns: 3,
                row_height: Some(28.0),
              },
            );
            root.world_mut().entity_mut(*g).with_children(|g| {
              for (i, s) in ["A1", "A2", "A3", "B1", "B2", "B3"].iter().enumerate() {
                let surface = if i < 3 {
                  PanelSurface::Card
                } else {
                  PanelSurface::Elevated
                };
                let cell = grid_cell(ctx, g, surface);
                g.world_mut().entity_mut(cell).with_children(|cell| {
                  label(
                    ctx,
                    cell,
                    LabelConfig {
                      text: (*s).into(),
                      ..default()
                    },
                  );
                });
              }
            });

            // ---- scroll view 演示（固定高度，内容超出可滚轮滚动） ----
            label(
              ctx,
              root,
              LabelConfig {
                text: "scroll view（滚轮滚动）".into(),
                style: LabelStyle::Muted,
              },
            );
            let sv = scroll_view(ctx, root, ScrollConfig { height: px(120.0) });
            root
              .world_mut()
              .entity_mut(sv.content)
              .with_children(|svc| {
                for i in 0..16 {
                  label(
                    ctx,
                    svc,
                    LabelConfig {
                      text: format!("scroll row {i}"),
                      ..default()
                    },
                  );
                }
              });
          });
        });

      // ---- tab 1：数据展示（table） ----
      root
        .world_mut()
        .entity_mut(tv.contents[1])
        .with_children(|p| {
          let sv = scroll_view(
            ctx,
            p,
            ScrollConfig {
              height: Val::Percent(100.0),
            },
          );
          p.world_mut().entity_mut(sv.content).with_children(|root| {
            label(
              ctx,
              root,
              LabelConfig {
                text: "数据展示".into(),
                style: LabelStyle::Title,
              },
            );
            label(
              ctx,
              root,
              LabelConfig {
                text: "table（表头 + 斑马纹）".into(),
                style: LabelStyle::Muted,
              },
            );
            table(
              ctx,
              root,
              TableConfig {
                headers: ["name", "value", "status"]
                  .iter()
                  .map(|s| s.to_string())
                  .collect(),
                rows: [
                  ["fps", "60", "ok"],
                  ["frame", "16.7ms", "ok"],
                  ["mem", "1.2gb", "warn"],
                  ["gpu", "rtx 3070", "ok"],
                ]
                .iter()
                .map(|r| r.iter().map(|s| s.to_string()).collect())
                .collect(),
              },
            );
          });
        });
    });

  // 按钮点击 → 写事件日志（UiClick 观察者）。日志首条由 spawn 处播种。
  world.add_observer(
    |click: On<UiClick>,
     q_btn: Query<&ShowcaseButton>,
     mut q_log: Query<&mut RingList, bevy::prelude::With<ShowcaseLog>>| {
      let Ok(name) = q_btn.get(click.entity) else {
        return;
      };
      if let Ok(mut ring) = q_log.single_mut() {
        ring.push(format!("button: {}", name.0));
      }
    },
  );

  // checkbox 翻转 → 写事件日志（CheckboxToggled 观察者）
  world.add_observer(
    |ev: On<CheckboxToggled>,
     q_cb: Query<&ShowcaseCheckbox>,
     mut q_log: Query<&mut RingList, bevy::prelude::With<ShowcaseLog>>| {
      let Ok(cb) = q_cb.get(ev.entity) else {
        return;
      };
      if let Ok(mut ring) = q_log.single_mut() {
        ring.push(format!(
          "checkbox {}: {}",
          cb.label,
          if ev.checked { "on" } else { "off" }
        ));
      }
    },
  );

  // slider 值变化 → 实时值标签 + 事件日志（SliderValueChanged 观察者）
  world.add_observer(
    |ev: On<SliderValueChanged>,
     mut q_val: Query<&mut Text, With<ShowcaseSliderValue>>,
     mut q_log: Query<&mut RingList, bevy::prelude::With<ShowcaseLog>>| {
      if let Ok(mut t) = q_val.single_mut() {
        let s = format!("{:>5.1}", ev.value);
        if t.0 != s {
          t.0 = s;
        }
      }
      if let Ok(mut ring) = q_log.single_mut() {
        ring.push(format!("slider: {:.0}", ev.value));
      }
    },
  );
}

/// 展示窗动态行为：正弦波喂演示折线（button/checkbox/slider 交互由观察者写日志）
pub(crate) fn showcase_demo_system(
  time: Res<Time>,
  mut q_plot: Query<&mut PlotData, bevy::prelude::With<ShowcasePlot>>,
  mut plot_acc: Local<f32>,
) {
  // 演示折线：每 0.1s 喂一个正弦样本
  *plot_acc += time.delta_secs();
  if *plot_acc >= 0.1 {
    *plot_acc = 0.0;
    let v = (time.elapsed_secs() * 2.0).sin() * 0.8;
    if let Ok(mut plot) = q_plot.single_mut() {
      plot.push(v);
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use gate_ui::theme::default_theme;
  use gate_ui::widgets::TabContent;

  /// 沿 ChildOf 链向上回溯根
  fn ancestor_chain(world: &World, mut e: Entity) -> Vec<Entity> {
    let mut chain = vec![e];
    while let Some(parent) = world.get::<ChildOf>(e).map(|c| c.0) {
      chain.push(parent);
      e = parent;
    }
    chain
  }

  /// tab 页必须是面板子树内的 Inherited 节点（toggle 整体显隐的前提）：
  /// 层级正确 + 选中页不使用显式 Visible（bevy 里 Visible 无视祖先 Hidden）
  #[test]
  fn showcase_panel_hides_all_descendants() {
    let mut app = App::new();
    app.init_resource::<Assets<Image>>();
    let theme = default_theme();
    let ctx = UiCtx::new(&theme, None);
    spawn_showcase(app.world_mut(), &ctx);
    app.update();

    let panel = app
      .world_mut()
      .query_filtered::<Entity, With<ShowcaseRoot>>()
      .single(app.world())
      .expect("panel spawned");
    let plot = app
      .world_mut()
      .query_filtered::<Entity, With<ShowcasePlot>>()
      .single(app.world())
      .expect("showcase plot spawned");
    let tab_pages: Vec<Entity> = app
      .world_mut()
      .query_filtered::<Entity, With<TabContent>>()
      .iter(app.world())
      .collect();
    assert_eq!(tab_pages.len(), 2, "both tab pages spawned");

    // 页面与演示控件都在面板子树内
    let chain = ancestor_chain(app.world(), plot);
    assert!(
      chain.contains(&panel),
      "plot must be a descendant of the panel"
    );
    assert!(
      ancestor_chain(app.world(), tab_pages[0]).contains(&panel)
        && ancestor_chain(app.world(), tab_pages[1]).contains(&panel),
      "tab pages must be descendants of the panel"
    );

    // 回归契约：选中页必须 Visibility::Inherited（跟随祖先显隐）。
    // bevy 语义里显式 Visible 无视祖先 Hidden（propagate_recursive 直接置 true），
    // 面板被 toggle 隐藏时选中页会单独悬浮（tab 页面悬浮 bug）
    for (i, page) in tab_pages.iter().enumerate() {
      let vis = *app.world().get::<Visibility>(*page).unwrap();
      if i == 0 {
        assert_eq!(vis, Visibility::Inherited, "active page must inherit");
      } else {
        assert_eq!(vis, Visibility::Hidden, "inactive page must be hidden");
      }
    }
  }
}
