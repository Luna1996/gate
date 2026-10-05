use bevy::prelude::*;

pub const DISABLED_DIM: f32 = 0.35;
pub const PRESSED_SCALE: f32 = 0.98;
pub const BOX_SIZE: f32 = 16.0;
pub const DROPDOWN_ANIM_SECS: f32 = 0.2;
pub const DROPDOWN_ARROW_SIZE: f32 = 12.0;
pub const DROPDOWN_SCROLL_SPEED: f32 = 5.0;
pub const PLOT_W: u32 = 256;
pub const PLOT_H: u32 = 64;
pub const SCROLL_SPEED: f32 = 24.0;
pub const THUMB_SIZE: f32 = 16.0;
pub const THUMB_SIZE_DRAG: f32 = 18.0;
pub const TRACK_HEIGHT: f32 = 6.0;
pub const SLIDER_FINE_SCALE: f32 = 0.1;
pub const TRACK_W: f32 = 32.0;
pub const TRACK_H: f32 = 16.0;
pub const NUMBER_DRAG_PX_PER_STEP: f32 = 4.0;
pub const DRAG_THRESHOLD_PX: f32 = 3.0;
pub const TOOLTIP_DELAY: f32 = 0.5;
pub const TOOLTIP_MAX_W: f32 = 260.0;
pub const TOOLTIP_OFFSET_X: f32 = 14.0;
pub const TOOLTIP_OFFSET_Y: f32 = 18.0;
pub const TOOLTIP_MARGIN: f32 = 8.0;
pub const MD_MONO_ADVANCE_EM: f32 = 0.6;
pub const MD_QUOTE_BAR_W: f32 = 2.0;
pub const MD_RULE_H: f32 = 1.0;

pub const THUMB_INSET: f32 = THUMB_SIZE / 2.0;

pub const TOOLTIP_OFFSET: Vec2 = Vec2::new(TOOLTIP_OFFSET_X, TOOLTIP_OFFSET_Y);

pub const ELLIPSIS: &str = "...";

pub const CARET_CHAR: char = '|';
