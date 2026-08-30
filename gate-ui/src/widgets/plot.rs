//! plot：折线图 widget（环形缓冲 + CPU 光栅化 + ImageNode）。
//!
//! 光栅化是纯函数（buf + 样本 → 像素），headless 单测断言像素；
//! 重绘走 change 检测（OQ-3 结论：数据每帧更新时等价"每帧重绘"，静止时零成本）。

use std::collections::VecDeque;

use bevy::asset::RenderAssetUsages;
use bevy::image::Image;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};

use super::{UiCtx, color_of, label_bundle, px};

/// 画布默认尺寸（v0 固定；面板宽高自适应重建后置）
pub const PLOT_W: u32 = 256;
pub const PLOT_H: u32 = 64;

/// Y 轴值域：Fixed(min, max) 或 Auto（按当前样本 min/max 推导）
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PlotDomain {
  Fixed(f32, f32),
  Auto,
}

/// 折线数据（固定容量环形缓冲，新样本顶替最旧样本）
#[derive(Component, Clone, Debug)]
pub struct PlotData {
  capacity: usize,
  samples: VecDeque<f32>,
  y_domain: PlotDomain,
  pub line_color: Color,
  pub grid: bool,
}

impl PlotData {
  pub fn new(capacity: usize, y_domain: PlotDomain, line_color: Color, grid: bool) -> Self {
    Self {
      capacity,
      samples: VecDeque::new(),
      y_domain,
      line_color,
      grid,
    }
  }

  /// 压入样本；超出容量时顶替最旧样本，返回被顶替者
  pub fn push(&mut self, v: f32) -> Option<f32> {
    self.samples.push_back(v);
    if self.samples.len() > self.capacity {
      self.samples.pop_front()
    } else {
      None
    }
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

  /// 当前有效值域：Fixed 原样返回；Auto 按样本推导（空 → (0,1)；单点/等值 → ±1 扩展）
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
        if (max - min).abs() < f32::EPSILON {
          (min - 1.0, max + 1.0)
        } else {
          (min, max)
        }
      }
    }
  }
}

/// 空白 rgba8 画布（plot spawn 前由调用方注册进 Assets<Image>）
pub fn blank_plot_image(w: u32, h: u32) -> Image {
  Image::new(
    Extent3d {
      width: w,
      height: h,
      depth_or_array_layers: 1,
    },
    TextureDimension::D2,
    vec![0; (w * h * 4) as usize],
    TextureFormat::Rgba8UnormSrgb,
    RenderAssetUsages::default(),
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

/// CPU 光栅化到 rgba8 缓冲（纯函数，先清零再画网格 + Bresenham 折线）
pub fn rasterize(
  buf: &mut [u8],
  w: u32,
  h: u32,
  samples: &[f32],
  domain: (f32, f32),
  line: [u8; 4],
  grid: bool,
) {
  buf.fill(0);
  if w == 0 || h == 0 || buf.len() < (w * h * 4) as usize {
    return;
  }
  let grid_c = [line[0], line[1], line[2], 40];
  if grid {
    for k in 1..4u32 {
      let y = (h * k / 4) as i64;
      for x in 0..w as i64 {
        put_px(buf, w, h, x, y, grid_c);
      }
      let x = (w * k / 4) as i64;
      for y in 0..h as i64 {
        put_px(buf, w, h, x, y, grid_c);
      }
    }
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

/// 画布实体标记（ImageNode 持有 plot 纹理）
#[derive(Component, Debug)]
pub struct PlotCanvas;

/// 极值文本实体标记（每次重绘更新 min-max）
#[derive(Component, Debug)]
pub struct PlotExtents;

/// 主题折线图：数据 + 画布（ImageNode）+ 极值文本。
/// `image_handle` 由调用方预先注册（`Assets<Image>::add(blank_plot_image(..))`）。
pub fn plot(
  ctx: &UiCtx,
  parent: &mut ChildSpawner,
  image_handle: Handle<Image>,
  capacity: usize,
  y_domain: PlotDomain,
) -> Entity {
  let c = &ctx.theme.colors;
  let m = &ctx.theme.metrics;
  parent
    .spawn((
      Name::new("ui-plot"),
      PlotData::new(capacity, y_domain, color_of(&c.accent), true),
      Node {
        flex_direction: FlexDirection::Column,
        row_gap: px(m.spacing.xs),
        ..default()
      },
    ))
    .with_children(|root| {
      root.spawn((
        Name::new("ui-plot-canvas"),
        PlotCanvas,
        ImageNode {
          image: image_handle,
          ..default()
        },
        Node {
          width: px(PLOT_W as f32),
          height: px(PLOT_H as f32),
          ..default()
        },
      ));
      root.spawn((
        PlotExtents,
        label_bundle(ctx, String::new(), m.font_size.sm, color_of(&c.text_muted)),
      ));
    })
    .id()
}

/// 重绘（OQ-3 结论）：PlotData 变更 → 光栅化写回 Image（Handle 不变）+ 更新极值文本
pub fn plot_redraw_system(
  mut images: ResMut<Assets<Image>>,
  mut q_roots: Query<(Entity, &Children, &PlotData), Changed<PlotData>>,
  mut q_canvas: Query<&mut ImageNode>,
  mut q_text: Query<&mut Text, With<PlotExtents>>,
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
    rasterize(buf, w, h, &samples, domain, line, data.grid);

    let text = format!("{:.1}-{:.1}", domain.0, domain.1);
    for ch in children.iter() {
      if let Ok(mut t) = q_text.get_mut(ch) {
        t.set_if_neq(Text::new(text.clone()));
      }
    }
    let _ = root_e;
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::theme::{ThemeFont, default_theme};

  #[test]
  fn plot_ring_evicts_oldest() {
    let mut p = PlotData::new(3, PlotDomain::Auto, Color::WHITE, false);
    assert_eq!(p.push(1.0), None);
    assert_eq!(p.push(2.0), None);
    assert_eq!(p.push(3.0), None);
    assert_eq!(p.push(4.0), Some(1.0), "oldest evicted");
    let s: Vec<f32> = p.samples().collect();
    assert_eq!(s, [2.0, 3.0, 4.0]);
  }

  #[test]
  fn plot_domain_fixed_and_auto() {
    let mut fixed = PlotData::new(8, PlotDomain::Fixed(0.0, 60.0), Color::WHITE, false);
    assert_eq!(fixed.domain(), (0.0, 60.0), "fixed domain untouched");
    fixed.push(100.0);
    assert_eq!(fixed.domain(), (0.0, 60.0));

    let mut auto = PlotData::new(8, PlotDomain::Auto, Color::WHITE, false);
    assert_eq!(auto.domain(), (0.0, 1.0), "empty auto domain");
    auto.push(5.0);
    assert_eq!(auto.domain(), (4.0, 6.0), "single sample expanded");
    auto.push(-2.0);
    assert_eq!(auto.domain(), (-2.0, 5.0), "min/max of samples");
  }

  #[test]
  fn plot_rasterize_pixel_assertions() {
    // 5×5 固定域 (0,1)，样本 [0,1]：完美对角线 (0,4)→(4,0)
    let mut buf = vec![0u8; 5 * 5 * 4];
    let line = [255u8, 0, 0, 255];
    rasterize(&mut buf, 5, 5, &[0.0, 1.0], (0.0, 1.0), line, true);

    let px = |x: u32, y: u32| -> [u8; 4] {
      let i = ((y * 5 + x) * 4) as usize;
      buf[i..i + 4].try_into().unwrap()
    };
    // 对角线中点命中线色
    assert_eq!(px(2, 2), line, "diagonal midpoint is line color");
    assert_eq!(px(1, 3), line);
    assert_eq!(px(3, 1), line);
    // 背景仍透明
    assert_eq!(px(0, 0), [0, 0, 0, 0], "background cleared");
    // 网格命中（k=1 横线 y=1 的 x=0 处：不在折线上，alpha=40）
    assert_eq!(px(0, 1)[3], 40, "grid line alpha");
  }

  #[test]
  fn plot_rasterize_auto_domain_clamps() {
    // 域 Auto(0,1) 内样本越界值被钳制：v=10 → t=1 → 顶行
    let mut buf = vec![0u8; 4 * 4 * 4];
    let line = [0u8, 255, 0, 255];
    rasterize(&mut buf, 4, 4, &[10.0], (0.0, 1.0), line, false);
    let i = 4usize; // 单样本：x 居中=1，y=0（顶行）→ (0*4+1)*4
    assert_eq!(buf[i..i + 4], line);
  }

  #[test]
  fn plot_redraw_updates_image_and_text() {
    let theme = default_theme();
    let mut app = App::new();
    app.init_resource::<Assets<Image>>();
    app.insert_resource(theme.clone());
    app.insert_resource(ThemeFont::default());
    app.add_systems(Update, plot_redraw_system);

    let handle = app
      .world_mut()
      .resource_mut::<Assets<Image>>()
      .add(blank_plot_image(PLOT_W, PLOT_H));
    let ctx = UiCtx::new(&theme, None);
    let mut plot_e = None;
    app.world_mut().spawn_empty().with_children(|p| {
      plot_e = Some(plot(
        &ctx,
        p,
        handle.clone(),
        64,
        PlotDomain::Fixed(0.0, 1.0),
      ));
    });
    let plot_e = plot_e.expect("plot spawned");

    // 首帧（空数据）：插入即 Changed，触发一次重绘（网格可见）
    app.update();
    let img = app
      .world()
      .resource::<Assets<Image>>()
      .get(&handle)
      .unwrap();
    let data = img.data.as_ref().unwrap();
    let i = ((PLOT_H / 4 * PLOT_W) * 4) as usize;
    assert_eq!(data[i + 3], 40, "first redraw paints grid");

    // push 两个样本 → 对角线出现
    {
      let mut data = app.world_mut().get_mut::<PlotData>(plot_e).unwrap();
      data.push(0.0);
      data.push(1.0);
    }
    app.update();
    let img = app
      .world()
      .resource::<Assets<Image>>()
      .get(&handle)
      .unwrap();
    let data = img.data.as_ref().unwrap();
    // 起点 (0,63) 不在网格线上 → 纯线色
    let i0 = ((63 * PLOT_W) * 4) as usize;
    assert_eq!(
      &data[i0..i0 + 4],
      &[137, 180, 250, 255],
      "accent line start"
    );
    // 对角线中点落 (128,31) 或 (128,32) 之一（Bresenham 取整），且该点可能叠网格线
    let mid_alpha = |y: u32| data[((y * PLOT_W + 128) * 4 + 3) as usize];
    assert!(
      mid_alpha(31) == 255 || mid_alpha(32) == 255,
      "diagonal midpoint painted with line color"
    );

    // 极值文本更新
    let children = app
      .world()
      .get::<Children>(plot_e)
      .unwrap()
      .iter()
      .collect::<Vec<_>>();
    let mut found = false;
    for ch in children {
      if let Some(t) = app.world().get::<Text>(ch)
        && t.0.contains("0.0-1.0")
      {
        found = true;
      }
    }
    assert!(found, "extents label updated");
  }
}
