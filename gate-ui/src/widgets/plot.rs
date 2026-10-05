use std::collections::VecDeque;
use std::ops::Deref;

use bevy::asset::RenderAssetUsages;
use bevy::image::Image;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};

use super::{UiCtx, color_of, label_bundle, px};
use crate::widgets::consts::{PLOT_H, PLOT_W};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PlotDomain {
  Fixed(f32, f32),
  Auto,
}

#[derive(Component, Clone, Debug)]
pub struct PlotData {
  capacity: usize,
  samples: VecDeque<f32>,
  y_domain: PlotDomain,
  pub line_color: Color,
  pub unit: Option<&'static str>,
}

impl PlotData {
  pub fn new(capacity: usize, y_domain: PlotDomain, line_color: Color) -> Self {
    Self { capacity, samples: VecDeque::new(), y_domain, line_color, unit: None }
  }

  pub fn with_unit(mut self, unit: &'static str) -> Self {
    self.unit = Some(unit);
    self
  }

  pub fn push(&mut self, v: f32) -> Option<f32> {
    self.samples.push_back(v);
    if self.samples.len() > self.capacity { self.samples.pop_front() } else { None }
  }

  pub fn len(&self) -> usize {
    self.samples.len()
  }

  pub fn is_empty(&self) -> bool {
    self.samples.is_empty()
  }

  pub fn samples(&self) -> impl Iterator<Item = f32> + '_ {
    self.samples.iter().copied()
  }

  pub fn domain(&self) -> (f32, f32) {
    match self.y_domain {
      PlotDomain::Fixed(min, max) => (min, max),
      PlotDomain::Auto => {
        if self.samples.is_empty() {
          return (0.0, 1.0);
        }
        let mut min = f32::MAX;
        let mut max = f32::MIN;
        for &v in &self.samples {
          min = min.min(v);
          max = max.max(v);
        }
        if (max - min).abs() < f32::EPSILON { (min - 1.0, max + 1.0) } else { (min, max) }
      }
    }
  }
}

pub fn blank_plot_image(w: u32, h: u32) -> Image {
  Image::new(
    Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    TextureDimension::D2,
    vec![0; (w * h * 4) as usize],
    TextureFormat::Rgba8UnormSrgb,
    RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
  )
}

fn put_px(buf: &mut [u8], w: u32, h: u32, x: i64, y: i64, c: [u8; 4]) {
  if x < 0 || y < 0 || x >= w as i64 || y >= h as i64 {
    return;
  }
  let idx = ((y as u32 * w + x as u32) * 4) as usize;
  buf[idx..idx + 4].copy_from_slice(&c);
}

fn sample_xy(i: usize, n: usize, v: f32, domain: (f32, f32), w: u32, h: u32) -> (i64, i64) {
  let x = if n <= 1 {
    (w as i64 - 1) / 2
  } else {
    (i as f32 * (w as f32 - 1.0) / (n as f32 - 1.0)).round() as i64
  };
  let (dmin, dmax) = domain;
  let t = ((v - dmin) / (dmax - dmin)).clamp(0.0, 1.0);
  let y = ((1.0 - t) * (h as f32 - 1.0)).round() as i64;
  (x, y)
}

#[allow(clippy::too_many_arguments)]
fn draw_line(buf: &mut [u8], w: u32, h: u32, x0: i64, y0: i64, x1: i64, y1: i64, c: [u8; 4]) {
  let dx = (x1 - x0).abs();
  let sx = if x0 < x1 { 1 } else { -1 };
  let dy = -(y1 - y0).abs();
  let sy = if y0 < y1 { 1 } else { -1 };
  let mut err = dx + dy;
  let (mut x, mut y) = (x0, y0);
  loop {
    put_px(buf, w, h, x, y, c);
    if x == x1 && y == y1 {
      break;
    }
    let e2 = 2 * err;
    if e2 >= dy {
      err += dy;
      x += sx;
    }
    if e2 <= dx {
      err += dx;
      y += sy;
    }
  }
}

pub fn rasterize(
  buf: &mut [u8],
  w: u32,
  h: u32,
  samples: &[f32],
  domain: (f32, f32),
  line: [u8; 4],
) {
  buf.fill(0);
  if w == 0 || h == 0 || buf.len() < (w * h * 4) as usize {
    return;
  }
  let n = samples.len();
  if n == 0 {
    return;
  }
  if n == 1 {
    let (x, y) = sample_xy(0, 1, samples[0], domain, w, h);
    put_px(buf, w, h, x, y, line);
    return;
  }
  let (mut px0, mut py0) = sample_xy(0, n, samples[0], domain, w, h);
  put_px(buf, w, h, px0, py0, line);
  for (i, pair) in samples.windows(2).enumerate() {
    let (x1, y1) = sample_xy(i + 1, n, pair[1], domain, w, h);
    draw_line(buf, w, h, px0, py0, x1, y1, line);
    px0 = x1;
    py0 = y1;
  }
}

fn color_rgba(c: Color) -> [u8; 4] {
  c.to_srgba().to_u8_array()
}

#[derive(Component, Debug)]
pub struct PlotCanvas;

#[derive(Component, Debug)]
pub struct PlotExtents;

#[derive(Component, Debug)]
pub struct PlotYAxis;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PlotHandle(pub Entity);

impl Deref for PlotHandle {
  type Target = Entity;
  fn deref(&self) -> &Entity {
    &self.0
  }
}

impl From<PlotHandle> for Entity {
  fn from(h: PlotHandle) -> Entity {
    h.0
  }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PlotLayout {
  #[default]
  Plain,
  YAxis,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PlotConfig {
  pub layout: PlotLayout,
  pub image: Handle<Image>,
  pub capacity: usize,
  pub y_domain: PlotDomain,
  pub line_color: Color,
  pub unit: Option<&'static str>,
  pub canvas_h: f32,
  pub y_axis_width: Val,
  pub canvas_width: Val,
}

impl Default for PlotConfig {
  fn default() -> Self {
    Self {
      layout: PlotLayout::Plain,
      image: Handle::default(),
      capacity: 128,
      y_domain: PlotDomain::Auto,
      line_color: Color::WHITE,
      unit: None,
      canvas_h: PLOT_H as f32,
      y_axis_width: Val::Auto,
      canvas_width: Val::Auto,
    }
  }
}

pub fn plot(ctx: &UiCtx, parent: &mut ChildSpawner, config: PlotConfig) -> PlotHandle {
  let c = &ctx.theme.colors;
  let m = &ctx.theme.metrics;
  let canvas_border = (UiRect::all(px(m.border_width)), BorderColor::all(color_of(&c.border)));
  match config.layout {
    PlotLayout::Plain => {
      let canvas_width = match config.canvas_width {
        Val::Auto => px(PLOT_W as f32),
        w => w,
      };
      let e = parent
        .spawn((
          Name::new("ui-plot"),
          PlotData {
            unit: config.unit,
            ..PlotData::new(config.capacity, config.y_domain, config.line_color)
          },
          Node { flex_direction: FlexDirection::Column, row_gap: px(m.spacing.xs), ..default() },
        ))
        .with_children(|root| {
          root.spawn((
            Name::new("ui-plot-canvas"),
            PlotCanvas,
            ImageNode { image: config.image, ..default() },
            Node {
              width: canvas_width,
              height: px(config.canvas_h),
              border: canvas_border.0,
              ..default()
            },
            canvas_border.1,
          ));
          root.spawn((
            PlotExtents,
            label_bundle(ctx, String::new(), m.font_size.sm, color_of(&c.text_muted)),
          ));
        })
        .id();
      PlotHandle(e)
    }
    PlotLayout::YAxis => {
      let (canvas_width, canvas_grow) = match config.canvas_width {
        Val::Auto => (Val::Auto, 1.0),
        w => (w, 0.0),
      };
      let e = parent
        .spawn((
          Name::new("ui-plot-yaxis"),
          PlotData {
            unit: config.unit,
            ..PlotData::new(config.capacity, config.y_domain, config.line_color)
          },
          Node {
            flex_direction: FlexDirection::Row,
            column_gap: px(m.spacing.xs),
            align_items: AlignItems::Stretch,
            ..default()
          },
        ))
        .with_children(|root| {
          root
            .spawn((
              Name::new("ui-plot-yaxis-labels"),
              PlotYAxis,
              Node {
                flex_direction: FlexDirection::Column,
                justify_content: JustifyContent::SpaceBetween,
                align_items: AlignItems::FlexEnd,
                width: config.y_axis_width,
                ..default()
              },
            ))
            .with_children(|col| {
              for _ in 0..3 {
                col.spawn(label_bundle(
                  ctx,
                  String::new(),
                  m.font_size.sm,
                  color_of(&c.text_muted),
                ));
              }
            });
          root.spawn((
            Name::new("ui-plot-canvas"),
            PlotCanvas,
            ImageNode { image: config.image, ..default() },
            Node {
              flex_grow: canvas_grow,
              width: canvas_width,
              height: px(config.canvas_h),
              border: canvas_border.0,
              ..default()
            },
            canvas_border.1,
          ));
        })
        .id();
      PlotHandle(e)
    }
  }
}

pub fn plot_redraw_system(
  mut images: ResMut<Assets<Image>>,
  mut q_roots: Query<(Entity, &Children, &PlotData), Changed<PlotData>>,
  mut q_canvas: Query<&mut ImageNode>,
  q_yaxis: Query<&Children, With<PlotYAxis>>,
  mut q_text: Query<&mut Text>,
) {
  for (root_e, children, data) in &mut q_roots {
    let mut handle = None;
    for ch in children.iter() {
      if let Ok(node) = q_canvas.get_mut(ch) {
        handle = Some(node.image.clone());
        break;
      }
    }
    let Some(handle) = handle else { continue };
    let Some(mut image) = images.get_mut(&handle) else {
      continue;
    };
    let (w, h) = (image.width(), image.height());
    let Some(buf) = image.data.as_deref_mut() else {
      continue;
    };
    let domain = data.domain();
    let line = color_rgba(data.line_color);
    let samples: Vec<f32> = data.samples().collect();
    rasterize(buf, w, h, &samples, domain, line);

    let text = format!("{:.1}-{:.1}", domain.0, domain.1);
    for ch in children.iter() {
      if let Ok(mut t) = q_text.get_mut(ch) {
        t.set_if_neq(Text::new(text.clone()));
      }
    }
    let vals = [domain.1, (domain.0 + domain.1) * 0.5, domain.0];
    for ch in children.iter() {
      let Ok(labels) = q_yaxis.get(ch) else {
        continue;
      };
      for (i, label_e) in labels.iter().enumerate() {
        if let Ok(mut t) = q_text.get_mut(label_e) {
          let s = match data.unit {
            Some(u) => format!("{:05.1}{u}", vals[i]),
            None => format!("{:05.1}", vals[i]),
          };
          t.set_if_neq(Text::new(s));
        }
      }
    }
    let _ = root_e;
  }
}
