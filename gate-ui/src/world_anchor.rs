//! world_anchor：世界空间 UI 投影锚定（FR-3）。
//!
//! 每帧把锚点世界坐标经 view_proj 投影到屏幕像素，写入 UI 节点绝对定位（left/top）；
//! NDC 越界 / 在相机背后 → `Visibility::Hidden`。可选距离缩放（作用于本实体 TextFont）。
//!
//! 依赖边界：gate-ui 不依赖 gate-render——投影矩阵经通用资源 [`AnchorCamera`]
//! 由 app 侧同步（gate-app 从 DdaCameraConfig 拷贝），投影数学只依赖 Mat4。
//! 遮挡检测（OQ-2，CPU DDA）v0 不做，后置 P7.3（需体素网格跨 crate 访问）。

use crate::theme::ThemeFont;
use bevy::asset::{AssetServer, LoadState};
use bevy::log::info;
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
    Self {
      view_proj: Mat4::IDENTITY,
      position_world: Vec3::ZERO,
    }
  }
}

/// 世界空间锚点（挂任意 UI 节点；v0 距离缩放作用于本实体 TextFont）
#[derive(Component, Clone, Copy, Debug)]
pub struct WorldAnchor {
  /// 锚点世界坐标（voxel 单位，与渲染世界一致）
  pub pos_fine: Vec3,
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
///
/// 不直接 spawn Text/TextFont 的原因：`world_anchor_label` 通常在 Startup set
/// 同步调用（例如 gate-app setup()），此时主题字体尚未异步加载，Assets<Font>
/// default slot 是 Bevy 内置 FiraMono（CJK 字形缺失 → 方框）。gate-ui 的
/// `world_anchor_apply_text` 系统在 ThemeFont Loaded 后把 Pending 记录里的
/// Text/TextFont 组件真正插入实体，保证首帧栅格即使用主题字体渲染 CJK 字形。
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
  Some(Vec2::new(
    (ndc.x * 0.5 + 0.5) * screen.x,
    (1.0 - (ndc.y * 0.5 + 0.5)) * screen.y,
  ))
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
  let screen = Vec2::new(
    window.physical_width().max(1) as f32,
    window.physical_height().max(1) as f32,
  );
  for (e, anchor, mut node, mut vis, tf, base) in &mut q {
    let mut show = anchor.visible;
    if show {
      match project_to_screen(cam.view_proj, anchor.pos_fine, screen) {
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

    // 可选距离缩放：作用于本实体 TextFont（无 TextFont 的节点静默跳过）
    if anchor.scale_with_distance
      && let Some(mut tf) = tf
    {
      let dist = anchor.pos_fine.distance(cam.position_world);
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
  if v {
    Visibility::Inherited
  } else {
    Visibility::Hidden
  }
}

/// 快捷 spawn：世界空间文本标注（绝对定位由系统每帧写入）
///
/// 字体来源：`FontSource::default()`（Assets<Font> 的 AssetId::default() slot）。
/// 为了保证 CJK 字形不变成方框，本函数**不会立即插入 Text/TextFont 组件**，而是
/// 写入 [`PendingAnchorText`]；`world_anchor_apply_text` 在主题字体
/// `LoadState::Loaded` + `ui_theme_font_install_default` 覆盖 default slot
/// 之后才把真正的文本组件插入实体。
pub fn world_anchor_label(
  commands: &mut Commands,
  text: &str,
  pos_fine: Vec3,
  color: Color,
) -> Entity {
  commands
    .spawn((
      Name::new("ui-world-anchor"),
      Label,
      WorldAnchor {
        pos_fine,
        visible: true,
        scale_with_distance: true,
        reference_distance: 760.0,
      },
      Node::default(),
      // 文本延迟到字体就绪后再插入（见 world_anchor_apply_text）
      PendingAnchorText {
        text: text.to_string(),
        color,
        font_size: FontSize::Px(14.0),
      },
    ))
    .id()
}

/// 将 [`PendingAnchorText`] 转换为 `Text + TextFont + TextColor` 组件。
///
/// 触发条件（任一满足，按优先级判断）：
/// 1. ThemeFont.handle 已存在 且 `LoadState::Loaded` → 直接用主题字体 Handle
///    （保证字形完整，与面板文本一致）；
/// 2. ThemeFont 不存在 或 theme.font_path = None → 立即 fallback 到
///    `FontSource::default()`（可能是 SystemUi/Bevy FiraMono，此时 CJK 风险
///    交给项目调用方配置 theme.font_path 兜底）。
///
/// 不满足条件 → 下一帧重试（Pending 组件继续保留）。
pub fn world_anchor_apply_text(
  mut commands: Commands,
  server: Option<Res<AssetServer>>,
  font: Option<Res<ThemeFont>>,
  mut pending: Query<(Entity, &PendingAnchorText), Without<Text>>,
) {
  // 1) 判断字体就绪态
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
  // 2) 对所有尚未应用文本（Without<Text>）的 Pending 实体插入组件
  let mut count = 0usize;
  for (e, p) in &mut pending {
    commands
      .entity(e)
      .insert((
        Text::new(p.text.clone()),
        TextFont {
          font: font_source.clone(),
          font_size: p.font_size,
          ..default()
        },
        TextColor(p.color),
      ))
      .remove::<PendingAnchorText>();
    count += 1;
  }
  if count > 0 {
    info!("world_anchor: applied text for {count} anchors (font_source ready)");
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  /// 相机 (5,5,5) 看原点：原点投影 = 屏幕中心
  #[test]
  fn anchor_projects_to_screen_center() {
    let view = Mat4::look_at_rh(Vec3::new(5.0, 5.0, 5.0), Vec3::ZERO, Vec3::Y);
    let proj = Mat4::perspective_rh(60.0_f32.to_radians(), 16.0 / 9.0, 1.0, 4000.0);
    let vp = proj * view;
    let screen = Vec2::new(1600.0, 900.0);
    let px = project_to_screen(vp, Vec3::ZERO, screen).expect("origin inside frustum");
    assert!((px.x - 800.0).abs() < 1.0, "x = {}", px.x);
    assert!((px.y - 450.0).abs() < 1.0, "y = {}", px.y);
  }

  /// 视锥外 / 相机背后 → None（system 侧置 Hidden）
  #[test]
  fn anchor_outside_frustum_hidden() {
    let view = Mat4::look_at_rh(Vec3::new(5.0, 5.0, 5.0), Vec3::ZERO, Vec3::Y);
    let proj = Mat4::perspective_rh(60.0_f32.to_radians(), 16.0 / 9.0, 1.0, 4000.0);
    let vp = proj * view;
    let screen = Vec2::new(1600.0, 900.0);
    // 相机背后（视线朝原点，(10,10,10) 在身后）
    assert_eq!(
      project_to_screen(vp, Vec3::new(10.0, 10.0, 10.0), screen),
      None
    );
    // 视锥外横向远偏
    assert_eq!(
      project_to_screen(vp, Vec3::new(100.0, 0.0, 0.0), screen),
      None
    );
  }

  /// 距离缩放：基准处 1.0、单调递减、钳制 [0.25, 4]
  #[test]
  fn distance_scale_monotonic_and_clamped() {
    let s = |d| anchor_distance_scale(d, 760.0);
    assert_eq!(s(760.0), 1.0);
    assert!(s(100.0) > s(200.0), "closer → larger");
    assert!(s(200.0) > s(760.0));
    assert!(s(760.0) > s(2000.0));
    assert_eq!(s(10.0), 4.0, "clamp max");
    assert_eq!(s(1.0e6), 0.25, "clamp min");
  }

  /// 已知 2D 标注：锚点像素随距离缩放后 font_size 单调（投影 → 缩放联动语义）
  #[test]
  fn scale_applied_to_font_size_range() {
    let base = 14.0_f32;
    for dist in [190.0, 760.0, 3040.0] {
      let s = anchor_distance_scale(dist, 760.0);
      let font = base * s;
      // scale 钳制 [0.25, 4] → font ∈ [3.5, 56]
      assert!((3.5..=56.0).contains(&font), "dist {dist} → font {font}");
    }
  }
}
