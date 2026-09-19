//! gate-render 根目录级旋钮（光照 / 响应式 / profiler / DDGI）：改这里即改默认值。
//!
//! 全是编译期常量（改值后重新编译）。要给某项加 debug_menu 控件时，把它挪进 Bevy 资源
//! 再按菜单连线（做法见同目录的 `RenderScale` / `EyeAdaptSettings`）。brickmap 的旋钮在
//! `brickmap/consts.rs`；结构性常量（wire 布局、WGSL 镜像、WESL 权威值）不在这里。

use bevy::prelude::*;

/// 阴影射线起点沿法线的偏移（voxel，抑 acne）
pub const SHADOW_BIAS: f32 = 0.5;
/// 平行光阴影射线 t_max（voxel）
pub const SHADOW_DIR_T_MAX: f32 = 8192.0;
/// 自发光体素的辐亮度增益
pub const EMISSIVE_EMIT_GAIN: f32 = 4.0;

/// 窗口 / 渲染目标尺寸下限（像素）
pub const MIN_DIM: u32 = 64;
/// 窗口 / 渲染目标尺寸上限（像素）
pub const MAX_DIM: u32 = 4096;
/// 启动的渲染降采样倍数（运行期以菜单「视频/半分辨率」为准）
pub const RENDER_SCALE: u32 = 1;

/// GPU 逐 pass 日志周期（秒）
pub const REPORT_PERIOD_SECS: f32 = 2.0;

/// 启动 DDGI 阶段：0=Off 1=Active 2=Cast 3=Full
pub const DDGI_STAGE: u8 = 3;
/// DDGI 世界网格向外扩的量（voxel，六方向；决定 LOD1~3 的槽位预算）
pub const DDGI_GRID_MARGIN: i32 = 16;
/// 逐角几何遮挡求交（关掉画面漏光，仅量测 GPU 成本用）
pub const DDGI_OCCLUSION: bool = true;

/// 渲染目标初始尺寸（窗口按它开）
pub const VIEW_SIZE: UVec2 =
  UVec2::new(crate::brickmap::consts::VIEW_W, crate::brickmap::consts::VIEW_H);

/// DDGI 五级 LOD 的 cell 边长（voxel，逐级 ×2；长度 = `ddgi::DDGI_LODS`）。
/// LOD0（cell 4）不铺满：逐 16³ 砖在运行时认领；其余级按世界 AABB 铺满；`DDGI_CHUNK_LOD` 那级按 chunk 领段。
pub const DDGI_LOD_CELL_SIZES: [i32; 5] = [4, 16, 32, 64, 128];

/// 默认天空色（Minecraft 白天平原，线性空间 rgb）
pub const MINECRAFT_SKY: [f32; 3] = [120.0 / 255.0, 167.0 / 255.0, 1.0];
