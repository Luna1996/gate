//! world_anchor：世界空间 UI 投影锚定。
//! 每帧把锚点世界坐标经 view_proj 投影为屏幕像素，写入 UI 节点绝对定位（left/top）；越界 / 相机背后 → `Visibility::Hidden`。
//! 可选距离缩放作用于本实体 TextFont；投影矩阵经 `AnchorCamera` 由 app 侧同步（gate-ui 不依赖 gate-render）。

use crate::theme::ThemeFont;
use bevy::asset::{AssetServer, LoadState};
use bevy::log::debug;
use bevy::prelude::*;
use bevy::text::{FontSize, FontSource, TextColor};
use bevy::ui::widget::Label;

/// 投影相机镜像（app 侧每帧同步；Identity = 未同步，锚点将投影到无效位置）
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

/// 世界空间锚点（挂任意 UI 节点；距离缩放作用于本实体 TextFont）
#[derive(Component, Clone, Copy, Debug)]
pub struct WorldAnchor {
  /// 锚点世界坐标（voxel 单位，与渲染世界一致）
  pub pos_voxel: Vec3,
  /// 手动开关（投影系统之外的总闸）
  pub visible: bool,
  /// 近大远小（按 reference_distance / 距离 缩放 font_size）
  pub scale_with_distance: bool,
  /// 距离缩放基准：相机距锚点 = 该值时缩放系数 = 1.0
  pub reference_distance: f32,
}

/// 首次应用缩放时记录的基准字号（后续 = base × scale）
#[derive(Component, Debug)]
pub struct AnchorBaseFont(FontSize);

/// `world_anchor_label()` 生成的待应用文本（文本 + 颜色）。
/// Text/TextFont 延迟到主题字体就绪后由 `world_anchor_apply_text` 补插，避免 CJK 方框。
#[derive(Component, Clone, Debug)]
pub struct PendingAnchorText {
  pub text: String,
  pub color: Color,
  pub font_size: FontSize,
}

/// 世界坐标 → 屏幕像素（左上原点，y 向下）。
/// 返回 None = 视锥外（NDC 越界）或在相机背后（w ≤ 0）。
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

/// 距离缩放系数：reference_distance 处 = 1.0，近大远小，钳制 [0.25, 4.0]
pub fn anchor_distance_scale(distance: f32, reference_distance: f32) -> f32 {
  if distance <= 1e-6 {
    return 4.0;
  }
  (reference_distance / distance).clamp(0.25, 4.0)
}

/// 投影锚定查询集（type alias 满足 clippy::type_complexity）
type AnchorQuery = (
  Entity,
  &'static WorldAnchor,
  &'static mut Node,
  &'static mut Visibility,
  Option<&'static mut TextFont>,
  Option<&'static AnchorBaseFont>,
);

/// 投影锚定（每帧重算；gate-app 从 DdaCameraConfig 同步 AnchorCamera）
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

    // 可选距离缩放：无 TextFont 的节点跳过
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

/// 快捷 spawn：世界空间文本标注（绝对定位由系统每帧写入）。
/// 不立即插入 Text/TextFont，而是写入 `PendingAnchorText`，由 `world_anchor_apply_text` 字体就绪后补插。
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
      // 文本延迟到字体就绪后插入（见 world_anchor_apply_text）
      PendingAnchorText { text: text.to_string(), color, font_size: FontSize::Px(14.0) },
    ))
    .id()
}

/// 将 `PendingAnchorText` 转换为 `Text + TextFont + TextColor` 组件。
/// ThemeFont 已 Loaded → 用主题字体 Handle；不存在 / `font_path = None` → `FontSource::default()`；未就绪下一帧重试。
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
