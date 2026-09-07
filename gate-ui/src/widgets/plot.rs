//! plot：折线图 widget（环形缓冲 + CPU 光栅化 + ImageNode）。
//!
//! 光栅化是纯函数（buf + 样本 → 像素），headless 单测断言像素；
//! 重绘走 change 检测（OQ-3 结论：数据每帧更新时等价"每帧重绘"，静止时零成本）。

use std::collections::VecDeque;
use std::ops::Deref;

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
  /// 纵轴数值后缀（plot_yaxis 标签用），如 "ms"；None → 纯数字
  pub unit: Option<&'static str>,
}

impl PlotData {
  pub fn new(capacity: usize, y_domain: PlotDomain, line_color: Color) -> Self {
    Self {
      capacity,
      samples: VecDeque::new(),
      y_domain,
      line_color,
      unit: None,
    }
  }

  /// 链式设置纵轴单位后缀
  pub fn with_unit(mut self, unit: &'static str) -> Self {
    self.unit = Some(unit);
    self
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

/// 空白 rgba8 画布（plot spawn 前由调用方注册进 Assets<Image>）。
///
/// 必须带 `MAIN_WORLD`：bevy 0.19 中图像上传 GPU 后默认清空 CPU 端 `data`，
/// 而 `plot_redraw_system` 需要直接写 `image.data` 做 CPU 光栅化。
/// 缺此标志 → `image.data` 为 None → 重绘被跳过 → 折线图空白。
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

/// CPU 光栅化到 rgba8 缓冲（纯函数，先清零再画 Bresenham 折线）
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

/// 画布实体标记（ImageNode 持有 plot 纹理）
#[derive(Component, Debug)]
pub struct PlotCanvas;

/// 极值文本实体标记（每次重绘更新 min-max）
#[derive(Component, Debug)]
pub struct PlotExtents;

/// 左侧纵轴标签列标记（YAxis 布局 spawn；列内子实体按 spawn 顺序 = 上(max)/中/下(min)）
#[derive(Component, Debug)]
pub struct PlotYAxis;

/// 折线图句柄（Deref 到根实体 Entity；PlotData 就挂在该实体上）
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

/// 折线图布局档
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PlotLayout {
  /// 简单布局：画布（固定 PLOT_W 宽）+ 底部极值文本
  #[default]
  Plain,
  /// 纵轴布局：左侧纵轴标签列（max/mid/min）+ 画布（flex_grow 撑满宽度）
  YAxis,
}

/// 折线图配置（全部字段进 Config；Default 见字段说明）
#[derive(Clone, Debug, PartialEq)]
pub struct PlotConfig {
  /// 布局档（Default = Plain）
  pub layout: PlotLayout,
  /// 画布图像句柄，调用方预先注册 `Assets<Image>::add(blank_plot_image(..))`
  pub image: Handle<Image>,
  /// 样本环形缓冲容量（Default = 128）
  pub capacity: usize,
  /// Y 轴值域（Default = Auto）
  pub y_domain: PlotDomain,
  /// 折线颜色（须取自主题令牌；Default = 白，调用方总是显式覆盖）
  pub line_color: Color,
  /// 纵轴数值后缀（YAxis 标签用），如 "ms"；None → 纯数字
  pub unit: Option<&'static str>,
  /// 画布高度 px（Default = PLOT_H）
  pub canvas_h: f32,
  /// 纵轴标签列宽度（仅 YAxis 布局生效；Default = Auto 内容自适应，
  /// 可显式指定 Px/Percent 等布局约束）
  pub y_axis_width: Val,
  /// 画布宽度约束（Default = Auto：YAxis → flex_grow 撑满父容器剩余宽度，
  /// Plain → 固定 PLOT_W；显式 Px/Percent → 定宽，不参与 flex 拉伸）
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

/// 主题折线图：数据 + 画布（ImageNode）+ 标签，布局由 [`PlotConfig::layout`] 决定。
///
/// 画布（折线区域）一律带 1px 主题边框（`border` 令牌）包围；纹理透明，
/// 背景由父容器提供。
///
/// - [`PlotLayout::Plain`]：root(Column) = 画布（宽 = `canvas_width`，
///   Auto → 固定 PLOT_W）+ 极值文本
/// - [`PlotLayout::YAxis`]：root(Row) = 纵轴标签列（3 个右对齐标签，SpaceBetween
///   顶到画布上/中/下，数值在 `plot_redraw_system` 刷新，宽 = `y_axis_width`，
///   Auto → 内容自适应）+ 画布（宽 = `canvas_width`，Auto → `flex_grow = 1.0`
///   撑满父容器剩余宽度；高度固定 `canvas_h` px）
pub fn plot(ctx: &UiCtx, parent: &mut ChildSpawner, config: PlotConfig) -> PlotHandle {
  let c = &ctx.theme.colors;
  let m = &ctx.theme.metrics;
  let canvas_border = (
    UiRect::all(px(m.border_width)),
    BorderColor::all(color_of(&c.border)),
  );
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
              image: config.image,
              ..default()
            },
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
      // 画布宽度：Auto → flex_grow 撑满剩余；显式约束 → 定宽不拉伸
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
          // 纵轴标签列：高度随画布拉伸，SpaceBetween 分布 max/mid/min
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
          // 画布：宽度按 canvas_width 约束，高度固定
          root.spawn((
            Name::new("ui-plot-canvas"),
            PlotCanvas,
            ImageNode {
              image: config.image,
              ..default()
            },
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

/// 重绘（OQ-3 结论）：PlotData 变更 → 光栅化写回 Image（Handle 不变）+ 更新极值文本
pub fn plot_redraw_system(
  mut images: ResMut<Assets<Image>>,
  mut q_roots: Query<(Entity, &Children, &PlotData), Changed<PlotData>>,
  mut q_canvas: Query<&mut ImageNode>,
  q_yaxis: Query<&Children, With<PlotYAxis>>,
  // 单个 Text 查询覆盖两类标签：Plain 布局的极值文本（root 直接子节点）
  // 与 YAxis 布局的纵轴标签（纵轴列的子节点）；按层级定位，无实体重叠。
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
    // 左侧纵轴标签（plot_yaxis 变体）：列内子实体顺序 = 上(max)/中/下(min)。
    // 格式 000.0（5 位定宽 0 填充）+ 可选单位后缀，等宽字体下不抖动。
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

#[cfg(test)]
mod tests {
  use super::*;
  use crate::theme::{ThemeFont, default_theme};

  #[test]
  fn plot_ring_evicts_oldest() {
    let mut p = PlotData::new(3, PlotDomain::Auto, Color::WHITE);
    assert_eq!(p.push(1.0), None);
    assert_eq!(p.push(2.0), None);
    assert_eq!(p.push(3.0), None);
    assert_eq!(p.push(4.0), Some(1.0), "oldest evicted");
    let s: Vec<f32> = p.samples().collect();
    assert_eq!(s, [2.0, 3.0, 4.0]);
  }

  #[test]
  fn plot_domain_fixed_and_auto() {
    let mut fixed = PlotData::new(8, PlotDomain::Fixed(0.0, 60.0), Color::WHITE);
    assert_eq!(fixed.domain(), (0.0, 60.0), "fixed domain untouched");
    fixed.push(100.0);
    assert_eq!(fixed.domain(), (0.0, 60.0));

    let mut auto = PlotData::new(8, PlotDomain::Auto, Color::WHITE);
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
    rasterize(&mut buf, 5, 5, &[0.0, 1.0], (0.0, 1.0), line);

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
    // 无横线 indicator：非折线像素保持透明
    assert_eq!(px(0, 1), [0, 0, 0, 0], "no horizontal indicator line");
  }

  #[test]
  fn plot_rasterize_auto_domain_clamps() {
    // 域 Auto(0,1) 内样本越界值被钳制：v=10 → t=1 → 顶行
    let mut buf = vec![0u8; 4 * 4 * 4];
    let line = [0u8, 255, 0, 255];
    rasterize(&mut buf, 4, 4, &[10.0], (0.0, 1.0), line);
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
        PlotConfig {
          image: handle.clone(),
          capacity: 64,
          y_domain: PlotDomain::Fixed(0.0, 1.0),
          line_color: color_of(&theme.colors.accent_text),
          ..default()
        },
      ));
    });
    let plot_e = *plot_e.expect("plot spawned");

    // 首帧（空数据）：插入即 Changed，触发一次重绘；无横线 indicator → 保持透明
    app.update();
    let img = app
      .world()
      .resource::<Assets<Image>>()
      .get(&handle)
      .unwrap();
    let data = img.data.as_ref().unwrap();
    let i = ((PLOT_H / 4 * PLOT_W) * 4) as usize;
    assert_eq!(
      data[i + 3],
      0,
      "empty plot stays transparent (no indicator)"
    );

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
    // 起点 (0,63) → 纯线色（accent_text #FAFAFA）
    let i0 = ((63 * PLOT_W) * 4) as usize;
    assert_eq!(
      &data[i0..i0 + 4],
      &[250, 250, 250, 255],
      "accent_text line start"
    );
    // 对角线中点落 (128,31) 或 (128,32) 之一（Bresenham 取整）
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

  #[test]
  fn plot_yaxis_labels_track_domain() {
    let theme = default_theme();
    let mut app = App::new();
    app.init_resource::<Assets<Image>>();
    app.insert_resource(theme.clone());
    app.insert_resource(ThemeFont::default());
    app.add_systems(Update, plot_redraw_system);

    let handle = app
      .world_mut()
      .resource_mut::<Assets<Image>>()
      .add(blank_plot_image(64, 32));
    let ctx = UiCtx::new(&theme, None);
    let mut plot_e = None;
    app.world_mut().spawn_empty().with_children(|p| {
      plot_e = Some(plot(
        &ctx,
        p,
        PlotConfig {
          layout: PlotLayout::YAxis,
          image: handle,
          capacity: 64,
          y_domain: PlotDomain::Auto,
          line_color: Color::WHITE,
          unit: Some("ms"),
          canvas_h: 32.0,
          y_axis_width: Val::Auto,
          canvas_width: Val::Auto,
        },
      ));
    });
    let plot_e = *plot_e.expect("YAxis plot spawned");

    // 样本 2.0 / 10.0 → Auto 域 (2,10)：标签 上=10.0 中=6.0 下=2.0
    {
      let mut data = app.world_mut().get_mut::<PlotData>(plot_e).unwrap();
      data.push(2.0);
      data.push(10.0);
    }
    app.update();

    let root_children = app
      .world()
      .get::<Children>(plot_e)
      .unwrap()
      .iter()
      .collect::<Vec<_>>();
    let mut labels = vec![];
    for ch in root_children {
      if app.world().get::<PlotYAxis>(ch).is_some() {
        for l in app.world().get::<Children>(ch).unwrap().iter() {
          labels.push(app.world().get::<Text>(l).unwrap().0.clone());
        }
      }
    }
    assert_eq!(labels.len(), 3, "three y-axis labels");
    assert_eq!(labels[0], "010.0ms", "top label = max, 0-padded");
    assert_eq!(labels[1], "006.0ms", "mid label = mid, 0-padded");
    assert_eq!(labels[2], "002.0ms", "bottom label = min, 0-padded");
  }

  #[test]
  fn plot_yaxis_labels_no_unit_zero_padded() {
    let theme = default_theme();
    let mut app = App::new();
    app.init_resource::<Assets<Image>>();
    app.insert_resource(theme.clone());
    app.insert_resource(ThemeFont::default());
    app.add_systems(Update, plot_redraw_system);

    let handle = app
      .world_mut()
      .resource_mut::<Assets<Image>>()
      .add(blank_plot_image(64, 32));
    let ctx = UiCtx::new(&theme, None);
    let mut plot_e = None;
    app.world_mut().spawn_empty().with_children(|p| {
      plot_e = Some(plot(
        &ctx,
        p,
        PlotConfig {
          layout: PlotLayout::YAxis,
          image: handle,
          capacity: 64,
          y_domain: PlotDomain::Auto,
          line_color: Color::WHITE,
          unit: None,
          canvas_h: 32.0,
          y_axis_width: Val::Auto,
          canvas_width: Val::Auto,
        },
      ));
    });
    let plot_e = *plot_e.expect("YAxis plot spawned");

    {
      let mut data = app.world_mut().get_mut::<PlotData>(plot_e).unwrap();
      data.push(2.0);
      data.push(10.0);
    }
    app.update();

    let root_children = app
      .world()
      .get::<Children>(plot_e)
      .unwrap()
      .iter()
      .collect::<Vec<_>>();
    let mut labels = vec![];
    for ch in root_children {
      if app.world().get::<PlotYAxis>(ch).is_some() {
        for l in app.world().get::<Children>(ch).unwrap().iter() {
          labels.push(app.world().get::<Text>(l).unwrap().0.clone());
        }
      }
    }
    assert_eq!(labels[0], "010.0", "no unit suffix, 0-padded");
    assert_eq!(labels[1], "006.0");
    assert_eq!(labels[2], "002.0");
  }
}
