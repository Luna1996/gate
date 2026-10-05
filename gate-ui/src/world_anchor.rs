use crate::theme::ThemeFont;
use bevy::asset::{AssetServer, LoadState};
use bevy::log::debug;
use bevy::prelude::*;
use bevy::text::{FontSize, FontSource, TextColor};
use bevy::ui::widget::Label;

#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct AnchorCamera {
  pub view_proj: Mat4,
  pub position_world: Vec3,
}

impl Default for AnchorCamera {
  fn default() -> Self {
    Self { view_proj: Mat4::IDENTITY, position_world: Vec3::ZERO }
  }
}

#[derive(Component, Clone, Copy, Debug)]
pub struct WorldAnchor {
  pub pos_voxel: Vec3,
  pub visible: bool,
  pub scale_with_distance: bool,
  pub reference_distance: f32,
}

#[derive(Component, Debug)]
pub struct AnchorBaseFont(FontSize);

#[derive(Component, Clone, Debug)]
pub struct PendingAnchorText {
  pub text: String,
  pub color: Color,
  pub font_size: FontSize,
}

pub fn project_to_screen(view_proj: Mat4, pos: Vec3, screen: Vec2) -> Option<Vec2> {
  let clip = view_proj * pos.extend(1.0);
  if clip.w <= 1e-6 {
    return None;
  }
  let ndc = clip.truncate() / clip.w;
  if ndc.x.abs() > 1.0 || ndc.y.abs() > 1.0 {
    return None;
  }
  Some(Vec2::new((ndc.x * 0.5 + 0.5) * screen.x, (1.0 - (ndc.y * 0.5 + 0.5)) * screen.y))
}

pub fn anchor_distance_scale(distance: f32, reference_distance: f32) -> f32 {
  if distance <= 1e-6 {
    return 4.0;
  }
  (reference_distance / distance).clamp(0.25, 4.0)
}

type AnchorQuery = (
  Entity,
  &'static WorldAnchor,
  &'static mut Node,
  &'static mut Visibility,
  Option<&'static mut TextFont>,
  Option<&'static AnchorBaseFont>,
);

pub fn world_anchor_system(
  windows: Query<&Window>,
  cam: Option<Res<AnchorCamera>>,
  mut q: Query<AnchorQuery>,
  mut commands: Commands,
) {
  let Some(cam) = cam else { return };
  let Ok(window) = windows.single() else { return };
  let screen =
    Vec2::new(window.physical_width().max(1) as f32, window.physical_height().max(1) as f32);
  for (e, anchor, mut node, mut vis, tf, base) in &mut q {
    let mut show = anchor.visible;
    if show {
      match project_to_screen(cam.view_proj, anchor.pos_voxel, screen) {
        Some(px) => {
          let left = Val::Px(px.x);
          let top = Val::Px(px.y);
          if node.position_type != PositionType::Absolute {
            node.position_type = PositionType::Absolute;
          }
          if node.left != left {
            node.left = left;
          }
          if node.top != top {
            node.top = top;
          }
        }
        None => show = false,
      }
    }
    if *vis != vis_of(show) {
      *vis = vis_of(show);
    }

    if anchor.scale_with_distance
      && let Some(mut tf) = tf
    {
      let dist = anchor.pos_voxel.distance(cam.position_world);
      let s = anchor_distance_scale(dist, anchor.reference_distance);
      match base {
        Some(b) => {
          if let FontSize::Px(v) = b.0 {
            let target = FontSize::Px(v * s);
            if tf.font_size != target {
              tf.font_size = target;
            }
          }
        }
        None => {
          commands.entity(e).insert(AnchorBaseFont(tf.font_size));
        }
      }
    }
  }
}

fn vis_of(v: bool) -> Visibility {
  if v { Visibility::Inherited } else { Visibility::Hidden }
}

pub fn world_anchor_label(
  commands: &mut Commands,
  text: &str,
  pos_voxel: Vec3,
  color: Color,
) -> Entity {
  commands
    .spawn((
      Name::new("ui-world-anchor"),
      Label,
      WorldAnchor {
        pos_voxel,
        visible: true,
        scale_with_distance: true,
        reference_distance: 760.0,
      },
      Node::default(),
      PendingAnchorText { text: text.to_string(), color, font_size: FontSize::Px(14.0) },
    ))
    .id()
}

pub fn world_anchor_apply_text(
  mut commands: Commands,
  server: Option<Res<AssetServer>>,
  font: Option<Res<ThemeFont>>,
  mut pending: Query<(Entity, &PendingAnchorText), Without<Text>>,
) {
  let ready_handle: Option<FontSource> = match (server, font) {
    (Some(srv), Some(f)) => match &f.handle {
      Some(h) if matches!(srv.load_state(h.id()), LoadState::Loaded) => {
        Some(FontSource::Handle(h.clone()))
      }
      _ => None,
    },
    (_, None) => Some(FontSource::default()),
    _ => None,
  };
  let Some(font_source) = ready_handle else {
    return;
  };
  let mut count = 0usize;
  for (e, p) in &mut pending {
    commands
      .entity(e)
      .insert((
        Text::new(p.text.clone()),
        TextFont { font: font_source.clone(), font_size: p.font_size, ..default() },
        TextColor(p.color),
      ))
      .remove::<PendingAnchorText>();
    count += 1;
  }
  if count > 0 {
    debug!("world_anchor text → {count}");
  }
}
