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

use rust_i18n::t;

use crate::consts::{SHOWCASE_PLOT_CAP, SHOWCASE_PLOT_H, SHOWCASE_PLOT_W};

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

#[derive(Component)]
pub(crate) struct ShowcaseRoot;

pub(crate) fn spawn_showcase(world: &mut World, ctx: &UiCtx) {
  let c = &ctx.theme.colors;
  let m = &ctx.theme.metrics;
  let showcase_plot_image =
    world.resource_mut::<Assets<Image>>().add(blank_plot_image(SHOWCASE_PLOT_W, SHOWCASE_PLOT_H));
  world
    .spawn((
      Name::new("hud-showcase-panel"),
      ShowcaseRoot,
      Node {
        position_type: PositionType::Absolute,
        right: px(8.0),
        top: px(8.0),
        width: px(280.0),
        height: Val::Vh(88.0),
        flex_direction: FlexDirection::Column,
        border: UiRect::all(px(m.border_width)),
        ..default()
      },
      BackgroundColor(color_of(&c.surface_card_hud)),
      BorderColor::all(color_of(&c.border)),
      Visibility::Hidden,
    ))
    .with_children(|root| {
      let tv = tab_view(
        ctx,
        root,
        TabConfig {
          tabs: vec![t!("showcase.tab.widgets").into(), t!("showcase.tab.data").into()],
          active: 0,
          fit_content: false,
          ..default()
        },
      );
      root.world_mut().entity_mut(tv.contents[0]).with_children(|p| {
        let sv = scroll_view(ctx, p, ScrollConfig { height: Val::Percent(100.0) });
        p.world_mut().entity_mut(sv.content).with_children(|root| {
          label(
            ctx,
            root,
            LabelConfig {
              text: t!("showcase.title").into(),
              style: LabelStyle::Title,
              ..default()
            },
          );
          label(
            ctx,
            root,
            LabelConfig {
              text: t!("showcase.subtitle").into(),
              style: LabelStyle::Muted,
              ..default()
            },
          );

          label(
            ctx,
            root,
            LabelConfig {
              text: t!("showcase.section.buttons").into(),
              style: LabelStyle::Muted,
              ..default()
            },
          );
          root
            .spawn((
              Name::new("showcase-button-row"),
              Node { column_gap: px(m.spacing.xs), ..default() },
            ))
            .with_children(|row| {
              let b1 = button(
                ctx,
                row,
                ButtonConfig { text: t!("showcase.button.primary").into(), ..default() },
              );
              row.world_mut().entity_mut(*b1).insert(ShowcaseButton("primary"));
              let b2 = button(
                ctx,
                row,
                ButtonConfig {
                  text: t!("showcase.button.secondary").into(),
                  variant: ButtonVariant::Secondary,
                  ..default()
                },
              );
              row.world_mut().entity_mut(*b2).insert(ShowcaseButton("secondary"));
              let b3 = button(
                ctx,
                row,
                ButtonConfig {
                  text: t!("showcase.button.ghost").into(),
                  variant: ButtonVariant::Ghost,
                  ..default()
                },
              );
              row.world_mut().entity_mut(*b3).insert(ShowcaseButton("ghost"));
            });
          let b4 = button(
            ctx,
            root,
            ButtonConfig {
              text: t!("showcase.button.danger").into(),
              variant: ButtonVariant::Danger,
              ..default()
            },
          );
          root.world_mut().entity_mut(*b4).insert(ShowcaseButton("danger"));

          label(
            ctx,
            root,
            LabelConfig {
              text: t!("showcase.section.checkbox").into(),
              style: LabelStyle::Muted,
              ..default()
            },
          );
          let cb1 = checkbox(
            ctx,
            root,
            CheckboxConfig {
              text: Some(t!("showcase.checkbox.a").into()),
              checked: true,
              ..default()
            },
          );
          root.world_mut().entity_mut(*cb1).insert(ShowcaseCheckbox { label: "A" });
          let cb2 = checkbox(
            ctx,
            root,
            CheckboxConfig {
              text: Some(t!("showcase.checkbox.b").into()),
              checked: false,
              ..default()
            },
          );
          root.world_mut().entity_mut(*cb2).insert(ShowcaseCheckbox { label: "B" });

          label(
            ctx,
            root,
            LabelConfig {
              text: t!("showcase.section.slider").into(),
              style: LabelStyle::Muted,
              ..default()
            },
          );
          root
            .spawn((
              Name::new("showcase-slider-row"),
              Node { column_gap: px(m.spacing.sm), align_items: AlignItems::Center, ..default() },
            ))
            .with_children(|row| {
              let s = slider(
                ctx,
                row,
                SliderConfig { min: 0.0, max: 100.0, value: 40.0, step: Some(5.0), ..default() },
              );
              {
                let w = row.world_mut();
                if let Some(mut node) = w.get_mut::<Node>(*s) {
                  node.flex_grow = 1.0;
                }
              }
              let v = label(ctx, row, LabelConfig { text: " 40.0".into(), ..default() });
              row.world_mut().entity_mut(*v).insert(ShowcaseSliderValue);
            });

          label(
            ctx,
            root,
            LabelConfig {
              text: t!("showcase.section.plot").into(),
              style: LabelStyle::Muted,
              ..default()
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

          label(
            ctx,
            root,
            LabelConfig {
              text: t!("showcase.section.list").into(),
              style: LabelStyle::Muted,
              ..default()
            },
          );
          let inner_panel = panel(ctx, root, PanelConfig { surface: PanelSurface::Elevated });
          root.world_mut().entity_mut(*inner_panel).with_children(|inner| {
            let l = list(ctx, inner, ListConfig { capacity: 6 });
            let w = inner.world_mut();
            w.entity_mut(*l).insert(ShowcaseLog);
            if let Some(mut ring) = w.get_mut::<RingList>(*l) {
              ring.push(t!("showcase.log.ready"));
              ring.push(t!("showcase.log.try_all"));
            }
          });

          root
            .spawn((
              Name::new("showcase-semantic-row"),
              Node { column_gap: px(m.spacing.sm), ..default() },
            ))
            .with_children(|row| {
              label(
                ctx,
                row,
                LabelConfig { text: "success".into(), style: LabelStyle::Success, ..default() },
              );
              label(
                ctx,
                row,
                LabelConfig { text: "warning".into(), style: LabelStyle::Warning, ..default() },
              );
              label(
                ctx,
                row,
                LabelConfig { text: "danger".into(), style: LabelStyle::Danger, ..default() },
              );
              label(
                ctx,
                row,
                LabelConfig { text: "accent".into(), style: LabelStyle::Accent, ..default() },
              );
            });

          label(
            ctx,
            root,
            LabelConfig {
              text: t!("showcase.section.faint").into(),
              style: LabelStyle::FaintLg,
              ..default()
            },
          );

          label(
            ctx,
            root,
            LabelConfig {
              text: t!("showcase.section.splitter").into(),
              style: LabelStyle::Muted,
              ..default()
            },
          );
          splitter(ctx, root);
          root
            .spawn((
              Name::new("showcase-splitter-row"),
              Node { column_gap: px(m.spacing.sm), ..default() },
            ))
            .with_children(|row| {
              label(
                ctx,
                row,
                LabelConfig { text: t!("showcase.splitter.left").into(), ..default() },
              );
              splitter(ctx, row);
              label(
                ctx,
                row,
                LabelConfig { text: t!("showcase.splitter.right").into(), ..default() },
              );
            });

          label(
            ctx,
            root,
            LabelConfig {
              text: t!("showcase.section.grid").into(),
              style: LabelStyle::Muted,
              ..default()
            },
          );
          let g = grid(ctx, root, GridConfig { columns: 3, row_height: Some(28.0) });
          root.world_mut().entity_mut(*g).with_children(|g| {
            for (i, s) in ["A1", "A2", "A3", "B1", "B2", "B3"].iter().enumerate() {
              let surface = if i < 3 { PanelSurface::Card } else { PanelSurface::Elevated };
              let cell = grid_cell(ctx, g, surface);
              g.world_mut().entity_mut(cell).with_children(|cell| {
                label(ctx, cell, LabelConfig { text: (*s).into(), ..default() });
              });
            }
          });

          label(
            ctx,
            root,
            LabelConfig {
              text: t!("showcase.section.scroll").into(),
              style: LabelStyle::Muted,
              ..default()
            },
          );
          let sv = scroll_view(ctx, root, ScrollConfig { height: px(120.0) });
          root.world_mut().entity_mut(sv.content).with_children(|svc| {
            for i in 0..16 {
              label(
                ctx,
                svc,
                LabelConfig { text: t!("showcase.scroll_row", i = i).to_string(), ..default() },
              );
            }
          });
        });
      });

      root.world_mut().entity_mut(tv.contents[1]).with_children(|p| {
        let sv = scroll_view(ctx, p, ScrollConfig { height: Val::Percent(100.0) });
        p.world_mut().entity_mut(sv.content).with_children(|root| {
          label(
            ctx,
            root,
            LabelConfig {
              text: t!("showcase.title.data").into(),
              style: LabelStyle::Title,
              ..default()
            },
          );
          label(
            ctx,
            root,
            LabelConfig {
              text: t!("showcase.section.table").into(),
              style: LabelStyle::Muted,
              ..default()
            },
          );
          table(
            ctx,
            root,
            TableConfig {
              headers: vec![
                t!("showcase.table.name").to_string(),
                t!("showcase.table.value").to_string(),
                t!("showcase.table.status").to_string(),
              ],
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

  world.add_observer(
    |ev: On<CheckboxToggled>,
     q_cb: Query<&ShowcaseCheckbox>,
     mut q_log: Query<&mut RingList, bevy::prelude::With<ShowcaseLog>>| {
      let Ok(cb) = q_cb.get(ev.entity) else {
        return;
      };
      if let Ok(mut ring) = q_log.single_mut() {
        ring.push(format!("checkbox {}: {}", cb.label, if ev.checked { "on" } else { "off" }));
      }
    },
  );

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

pub(crate) fn showcase_demo_system(
  time: Res<Time>,
  mut q_plot: Query<&mut PlotData, bevy::prelude::With<ShowcasePlot>>,
  mut plot_acc: Local<f32>,
) {
  *plot_acc += time.delta_secs();
  if *plot_acc >= 0.1 {
    *plot_acc = 0.0;
    let v = (time.elapsed_secs() * 2.0).sin() * 0.8;
    if let Ok(mut plot) = q_plot.single_mut() {
      plot.push(v);
    }
  }
}
