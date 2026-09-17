//! brickmap 旋钮（DDA + 上传预算）：改这里即改默认值。
//!
//! 全是编译期常量（改值后重新编译）。要给某项加 debug_menu 控件时，把它挪进 Bevy 资源
//! 再按菜单连线（做法见 gate-render 的 `RenderScale` / `EyeAdaptSettings`）。

/// 渲染目标（= 窗口）初始宽度，像素
pub const VIEW_W: u32 = 1280;
/// 渲染目标（= 窗口）初始高度，像素
pub const VIEW_H: u32 = 720;
/// beam 预 pass 的 workgroup 边长（设备上限内越大越快）
pub const WORKGROUP_SIZE: u32 = 8;
/// 轨道相机俯仰角上限（弧度，= 89°）
pub const PITCH_LIMIT: f32 = 89.0_f32.to_radians();
/// 轨道相机最近距离（体素）
pub const DIST_MIN: f32 = 32.0;
/// beam 深度纹理边长 = 全分辨率 ÷ 本值
pub const BEAM_DIV: u32 = 4;
/// 垂直视场角（弧度）；须与 gate-app `consts::FOV_Y` 一致
pub const DDA_FOV_Y: f32 = 60.0_f32.to_radians();
/// CPU 参考遍历器的步数上限（与 WGSL `trace_chunk` 同值）
pub const CPU_TRACE_BUDGET: u32 = 65536;

/// 诊断：chunk 步进（关掉 → trace_grid 在局部 slab 后直接 miss）
pub const DDA_CHUNKWALK: bool = true;
/// 诊断：只出天空（跳过全部 trace）
pub const DDA_SKY_ONLY: bool = false;
/// 诊断：只建 grid 不 trace
pub const DDA_MAKEGRID_ONLY: bool = false;
/// 八叉树远场早停（LOD）
pub const DDA_LOD: bool = true;
/// beam 预 pass（关掉则主 pass 从 t=0 起步）
pub const DDA_BEAM: bool = true;
/// 方向可达掩码剔除（LUT）
pub const DDA_DIR_LUT: bool = true;
/// 自动曝光初值（运行期以 Eye 页开关为准）
pub const EYE_ADAPT: bool = true;

/// 单次绑定字节上限（超过走多缓冲路径）
pub const SINGLE_THRESHOLD_BYTES: u64 = (1 << 30) - 1;
/// 每 chunk 上传字节预算（`max_bytes_per_frame` 按它折算 chunk 数）
pub const PER_CHUNK_BYTES: usize = 256 * 1024;
/// `UploadBudget` 缺省的每帧上传字节数
pub const UPLOAD_BYTES_PER_FRAME: usize = 4 * 1024 * 1024;
/// 多缓冲切片的目标字节数
pub const BUFFER_SLICE_TARGET: u64 = 64 << 20;
/// 多缓冲切片的下限字节数
pub const BUFFER_SLICE_MIN: u64 = 4 << 20;
/// 缓冲扩容阈值（字节）
pub const BUFFER_GROW_BIG: u64 = 8 << 20;
/// 缓冲扩容预留量（字节）
pub const BUFFER_GROW_RESERVE: u64 = 32 << 20;
/// brickmap 显存占用超过它就 warn（仅提示）
pub const VRAM_WARN_BYTES: u64 = 2 << 30;
