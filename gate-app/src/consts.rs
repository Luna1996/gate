//! gate-app 的可调旋钮（本 crate 的模块都在 `src/` 下，故集中在本文件）：改这里即改默认值。
//!
//! 全是编译期常量：改值后重新编译。要给某项加 debug_menu 控件时，把它挪进 Bevy 资源再按
//! 菜单连线（做法见 `debug_menu.rs` 里的 `RenderScale` / `EyeAdaptSettings` / `GiSettings`）。
/// 启动场景：true = 程序化 demo 场景；false = assets/vox/nuke.vox
pub const STARTUP_DEMO_SCENE: bool = false;
/// 启动相机朝天空（纯 miss 基准）
pub const START_CAMERA_SKY: bool = false;
/// demo 场景边长（tile 数，1 tile = 512 voxel）
pub const DEMO_TILES: i32 = 2;
/// 失焦窗口仍按 Continuous 跑帧（窗口启动即 focused=false）
pub const BENCH_UNFOCUSED: bool = false;
/// 相机自动绕目标旋转并平移（配 BENCH_UNFOCUSED 读移动中的逐 pass 帧）
pub const AUTO_ORBIT: bool = false;

/// 基准开关：环境变量 `GATE_BENCH` 已设置且非 `0`（`GATE_BENCH=1` 即可）。
///
/// WHY：跑"逐 pass 帧"的取证要打开 [`BENCH_UNFOCUSED`]（窗口启动即失焦，不开就几乎不出帧），
/// 让它们只由常量控制就意味着每次取证都要改代码重编译。用环境变量把这两个开关变成运行期可开，
/// 与 `GATE_LOG`（日志过滤）同一套路；两个常量仍是默认值（正常游玩不触发）。
pub fn bench() -> bool {
  static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
  *ON.get_or_init(|| {
    std::env::var("GATE_BENCH").is_ok_and(|v| !v.is_empty() && v != "0")
  })
}

/// 是否**额外**打开自动绕行（相机持续转头 + 以 500 v/s 平移）：`GATE_BENCH` 的值里含 `orbit`。
///
/// WHY 与 [`bench`] 分开：`GATE_BENCH=static` 就是"相机不动、世界静止"的**对照**——
/// 没有这个对照，分不清某个 pass 的代价来自"世界每帧在变（历史/缓存被作废）"还是"它本身就贵"。
pub fn bench_orbit() -> bool {
  static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
  *ON.get_or_init(|| {
    std::env::var("GATE_BENCH").is_ok_and(|v| v.to_ascii_lowercase().contains("orbit"))
  })
}
/// 第 60 帧自动刷一次笔触（无鼠标走通编辑 → 增量上传链路）
pub const EDIT_SELFTEST: bool = false;

/// 日志过滤串的缺省值（喂给 bevy `LogPlugin.filter`）：info 基线 + 定向屏蔽 wgpu_hal::vulkan 的
/// instance/surface 层 VUID 报错。**取证时用环境变量 `GATE_LOG` 覆盖整串**（见 `main.rs`）——
/// `LogPlugin.filter` 是权威，`RUST_LOG` 不再被读，那些 `debug!`（`STREAM` / `RESID` / `WINDOW` /
/// `GI epoch`）默认不落盘。
pub const DEFAULT_LOG_FILTER: &str = "info,wgpu=debug,wgpu_core=debug,\
  wgpu_hal::vulkan::instance=off,\
  wgpu_hal::vulkan::surface=off";

/// 垂直视场角（弧度）
pub const FOV_Y: f32 = 60.0_f32.to_radians();
/// 近裁剪面（voxel）
pub const CAM_NEAR: f32 = 1.0;
/// 远裁剪面（voxel）
pub const CAM_FAR: f32 = 65536.0;
/// 右键拖拽转向灵敏度（弧度/像素）；自由模式锁定后的鼠标转头用同一个值
pub const ROT_SPEED: f32 = 0.005;
/// 滚轮缩放速率（对数档/格）
pub const ZOOM_LOG_SPEED: f32 = 0.35;
/// 飞行相机默认速度（voxel/s）
pub const FLY_SPEED_DEFAULT: f32 = 128.0;
/// 飞行相机加速档倍率
pub const FLY_SPEED_FAST_MUL: f32 = 2.0;

/// 准星：每条臂的长度（像素，自中心缺口外缘起算）
pub const CROSSHAIR_ARM: f32 = 8.0;
/// 准星：线宽（像素）
pub const CROSSHAIR_THICK: f32 = 1.0;
/// 准星：中心缺口半径（像素，缺口内不画）
pub const CROSSHAIR_GAP: f32 = 3.0;

/// 笔触尺寸下限（体素）
pub const EDIT_SIZE_MIN: u32 = 1;
/// 编辑射线射程（体素）：无界（`∞`）⇒ 只要射线能打到就有笔触。
/// 实际步进由 `gate_render::raycast` 截在各 volume 的占用窗口 AABB 内（世界之外恒为空气），故不会空转。
pub const EDIT_REACH: f32 = f32::INFINITY;
/// 按住左键（放置）/ 右键（擦除）的连发间隔（秒）：按下当帧先落一笔，之后按住每 `EDIT_REPEAT_SECS` 再落一笔。
/// 触发频率与帧率无关（余数结转）；`0` = 每帧触发（不节流），调大 = 更慢的连续笔触。
pub const EDIT_REPEAT_SECS: f32 = 0.1;
/// 普通笔触的单帧 CPU 预算（毫秒）：预算内做完就本帧出全形，做不完的块留到下一帧续做。
/// 调大 = 大笔触更快出形但单帧更重（更容易掉帧）；调小 = 更平滑但要更多帧。
///
/// 分帧的**尾成本已降到微秒级**（增量上传改成节点级重写之后：单格编辑实测 356B / 13.7µs），
/// 所以"预算越小越亏"这条不再成立 —— 现在瓶颈就是笔触本身的 CPU：
/// castle.vox 球实测 `size=61` ≈4.2ms、`size=33` ≈1.2ms、`size=17` ≈0.2ms（`size=121` ≈23ms）。
/// 取 1ms：单帧最多给编辑 1ms，大笔触分几帧出形（60fps 下 4ms 预算 = 一帧 40% 预算，必掉）。
pub const EDIT_BUDGET_MS: f32 = 1.0;

/// 显示用换算：1 米 = 多少体素
pub const VOXEL_PER_METER: f32 = 50.0;
/// 相机读数刷新周期（秒）
pub const CAM_INFO_REFRESH_SECS: f32 = 0.25;
/// FPS 滚动窗口长度（秒）
pub const FPS_WINDOW_SECS: f32 = 1.0;
/// 一次 FPS 采样最多折算成几个呈现帧（主循环比渲染慢时的一次长停顿按帧均摊；
/// 上限防止窗口被同一份间隔灌满 —— 见 `debug_menu::fps_overlay_tick`）
pub const FPS_MAX_FRAMES_PER_TICK: u32 = 8;
/// FPS 四值的刷新间隔（秒）：不是每帧刷新 —— 高帧率下每帧刷数字既看不清也没意义
pub const FPS_UPDATE_SECS: f32 = 0.2;

/// 展示窗折线图画布宽度（像素）
pub const SHOWCASE_PLOT_W: u32 = 216;
/// 展示窗折线图画布高度（像素）
pub const SHOWCASE_PLOT_H: u32 = 48;
/// 展示窗折线图样本容量
pub const SHOWCASE_PLOT_CAP: usize = 128;

/// demo 场景世界边长（voxel）= tile 数 × 512
pub const EXT_VOXEL_X: i32 = DEMO_TILES * 512;
/// demo 场景世界边长（voxel）
pub const EXT_VOXEL_Z: i32 = DEMO_TILES * 512;
/// demo 场景世界中心（voxel）
pub const EXT_VOXEL_HALF: i32 = EXT_VOXEL_X / 2;

// ---------------------------------------------------------------------------
// MT6 · 材质位移 → 真实体素几何（只在 `STARTUP_DEMO_SCENE = true` 的 demo 场景里生效）
// ---------------------------------------------------------------------------

/// demo 场景是否生成「材质位移」样例：一对同尺寸同材质的石台（一座 = 普通 CSG、一座 = 高度图位移）。
/// 位置/尺寸/写入体素数/建议机位见启动日志的 `MT6 位移样例:` 两行（**只认日志，不猜坐标**）。
pub const DEMO_DISPLACE_SAMPLE: bool = true;
/// 位移样例用的材质高度图 id（磁盘侧 `assets/textures/pbr/<id>/<id>_height.png`）。
/// 只解码这一个材质（MT6-2 的"按需"）：16 个全解没必要。
/// ⚠️ MT8-5 起**光有这个 id 不够**：`<id>` 在资产表里的 `displacement_amplitude`（见下）必须 ≠ 0，
/// 否则样例按"该材质不位移"处理（右台退化为普通 CSG，日志会说明）。
pub const DEMO_DISPLACE_HEIGHT_MAP: &str = "stone_wall_04";
/// **位移幅度的可选覆盖**（MT8-5）：`Some` 时**压过材质资产里的幅度**（实验/对照用），
/// `None`（默认值）⇒ **完全由资产决定**。
///
/// ⚠️ **资产才是唯一真源**：MT8-5 把幅度收进了 `MaterialAsset`（`emissive_metal` 的 bits 24..31，
/// 单位 = 体素、峰-峰，0 = 不位移；见 `gate-render/src/brickmap/wire.rs` 的字段文档与
/// `gate-render/src/pbr_texture.rs::DISPLACE_DEMO_AMPLITUDE`）⇒ **改那一处即改凹凸**，
/// 本常量不再是入口（这正是 MT8-5 的验收："`DEMO_DISPLACE_AMPLITUDE` 不再是唯一入口"）。
/// 留这个旋钮只为"不动资产、快速试一档"。
///
/// 幅度上界：`gate_voxel::Displace::bound`（= 幅度/2）应 ≤ 块粒度（`fill_box` = 4）
/// —— 这样只有表面一层 4³ 块退化为逐体素，内部仍整块写（`docs/PLAN.md` §5 R7）。
/// 调大（如 16）会让壳层变厚、体素数/树规模上涨，MT6-6 有实测对比。
pub const DEMO_DISPLACE_AMPLITUDE_OVERRIDE: Option<f32> = None;
/// **采样缩放**（MT6-1）：一张高度图铺多少**体素**（100 体素 ≈ 2m @50 voxel/m）。
/// 与 MT3 的 `MATERIAL_TEX_WORLD_SCALE`(= 2.0m) 同源 ⇒ 凹凸与 albedo 贴图图案同相。
/// 与 `height_field::HEIGHT_DOWNSAMPLE` 一起决定 texel/体素（= 128/100 ≈ 1.28，≥1 才不出毛刺）。
pub const DEMO_DISPLACE_TEX_SCALE: f32 = 100.0;

/// **笔触位移的尺寸上限**（体素，与 `EditSettings::size` 同量纲）：`size` 超过它 ⇒
/// 本次笔触**不位移**（走 `edit::apply_brush` 的原路径）+ `warn!`，日志里写明原因。
///
/// **为什么需要上限**（交互性能，实测量化）：位移的壳层是**逐体素**的
/// （`gate_voxel::fill_shape`：grain=4 的块级判定 + 壳层逐格 `set_voxel`），
/// 代价 ≈ **O(size²)**（表面积），而普通笔触是 brick 整块写。
/// `--release` 实测（`edit.rs::tests::displace_brush_cost_by_size`，空网格上一笔、
/// 幅度 8 / bound 4，2026-09-21；格式 = `不位移 → 位移`）：
///
/// | size | Cube | Sphere |
/// |---|---|---|
/// | 16 | 4.1ms → **18.5ms** | 3.4ms → **11.4ms** |
/// | 32 | 13.8ms → **66.1ms** | 9.8ms → **38.2ms** |
/// | 64 | 45.8ms → **300ms** | 41.5ms → **189ms** |
/// | 128 | 183ms → **1.22s** | 180ms → **713ms** |
/// | 512 | 2.56s → **23.2s** | 2.44s → **13.4s** |
///
/// （同机重跑一遍：16 Cube 位移 11.8ms —— 机器负载带来 ±40% 抖动，量级结论不变。）
/// ⇒ size 16 的位移 ≈ 一帧（16.7ms @60fps），32 ≈ 3~4 帧（一次点击无感），
/// 64 起就是"点一下卡 0.2~0.3s"，128 起 >0.7s、512 **>13s**（完全不可接受）。
/// **取舍**：上限取 32 —— 小笔触（用户当前用的 16）照常有凹凸，且一次落笔 ≤ 4 帧；
/// 超过就**不位移**并 `warn!`（宁可"大笔触没有凹凸"，也不要落笔把界面卡住）。
/// 想要更大的位移笔触：调大本常量并接受 O(size²) 的代价（数字在上表）。
pub const EDIT_DISPLACE_SIZE_MAX: u32 = 32;
