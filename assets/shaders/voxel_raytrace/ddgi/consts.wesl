/// 辐照度图每探针 4×4（= 16 纹素）。
const DDGI_IRR_TEXELS: u32 = 4u;
/// 深度图每探针 8×8（= 64 纹素；每纹素在八面体图上约 22° 角宽）。
const DDGI_DEPTH_TEXELS: u32 = 8u;
// 图集布局：每层 `DDGI_PROBES_PER_LAYER_AXIS²` 个探针，共 `DDGI_ATLAS_LAYERS` 层。
// 层内轴 / 层数 / worklist 下标位宽必须一起改（容量 = 层数 × 轴向² 须装进下标位宽）；Rust 侧由 `wesl_consts.rs` 启动时解析。
const DDGI_PROBES_PER_LAYER_AXIS: u32 = 40u;
/// 图集层数（纹理 array 层数；Rust 按它定显存与「槽位 → 层」映射）。
const DDGI_ATLAS_LAYERS: u32 = 272u;
const DDGI_PROBES_PER_LAYER: u32 = DDGI_PROBES_PER_LAYER_AXIS * DDGI_PROBES_PER_LAYER_AXIS;
const DDGI_LOD_COUNT: u32 = 4u;
// meta word = age(低 8bit) | ENABLED(放了探针) | ACTIVE(本 cell 或 6 邻接有体素/物体) | 邻接掩码(bit10-15)；0 = 无探针哨兵
const DDGI_META_ENABLED: u32 = 256u;
const DDGI_META_ACTIVE: u32 = 512u;
/// 邻接可见性掩码在 meta 里的位偏移（占 bit10-15，共 6 位）。
/// 位序：0=-X, 1=+X, 2=-Y, 3=+Y, 4=-Z, 5=+Z。
const DDGI_META_NBR_SHIFT: u32 = 10u;
/// 掩码取值域（6 位）。
const DDGI_META_NBR_MASK: u32 = 0x3Fu;
/// 「邻接可见性掩码已由本帧 sort 算过」的有效位（bit16）；未置位时消费侧按被挡处理。
const DDGI_META_NBR_VALID: u32 = 65536u;
const DDGI_AGE_MAX: u32 = 255u;
/// worklist `.w` 的位段打包：age(低 8 位) | cell 下标(19 位) | lod(2 位)。
/// 下标是本级内线性下标；19 位 = 524288 ≥ 图集容量（改层内轴 / 层数时须核对）。
const DDGI_WL_IDX_MASK: u32 = 0x7FFFFu;
const DDGI_WL_LOD_SHIFT: u32 = 27u;
const DDGI_NO_PROBE: u32 = 0xFFFFFFFFu;
/// 辐照度稳态混合系数 = 每次更新向新样本靠拢的比例（稳态窗口 ≈ 1/本值 次更新）。
const DDGI_ALPHA: f32 = 1.0 / 64.0;
/// 深度图时域系数（图集两通道：mean / std），是稳态地板。
const DDGI_DEPTH_ALPHA: f32 = 1.0 / 255.0;
/// 深度图「按纹素自己的更新次数收敛」的计数上限。
const DDGI_DEPTH_CONV_SAMPLES: f32 = 255.0;
/// 末级兜底 r4（"本点一侧的探针场插值"）启用的最小同侧余弦权重和 Σ wtri·wn。
const DDGI_FIELD_MIN_W: f32 = 0.05;
/// Chebyshev 软遮挡里 std 的绝对下限（体素）。
const DDGI_DEPTH_SOFT_MIN: f32 = 0.5;
/// 空间混合（探针间去噪）里"自身"的权重参照。
const DDGI_BLEND_SELF: f32 = 2.0;
/// 邻居纹素的覆盖度下限（irr 图集第 4 通道 = 该纹素被写过的时间累积）。
const DDGI_TEXEL_MIN_COVERAGE: f32 = 0.5;
/// 前后判定（wn）的锐度：wn = clamp(dot(n,-dir)/此值, 0, 1)。
const DDGI_NORMAL_BIAS: f32 = 1.0;
/// 采样点沿法线的外推量 = 该 LOD 的 cell 边长 × 此系数（normal_bias 语义）。
const DDGI_BIAS_CELLS: f32 = 0.1;
// ============================================================================
// 逐角几何遮挡（真·线段求交）参数：线段 = 采样点 cell 的 8 角 → 采样点，长度 ≤ cs·√3。
// ============================================================================
/// 线段终点偏置（体素）：从 `max_t = 线段长` 里扣除，停在采样点前。
const DDGI_OCCL_END_BIAS: f32 = 1.0;
/// 逐角遮挡的权重阈值 ε：角权重 w = wtri·wn_w·wd < ε 时丢弃。
const DDGI_OCCL_W_MIN: f32 = 0.005;
const DDGI_T_MAX: f32 = 8192.0;
/// 级联混合带宽度（以当前级 cell 边长为单位）：距 LOD 盒边界小于 `cs × 此值` 时在该级与相邻级间过渡。
const DDGI_CASCADE_BLEND: f32 = 1.0;
/// 「覆盖内 GI ↔ 覆盖外天光兜底」过渡带的宽度（以最粗级 cell 为单位）。
const DDGI_CASCADE_FADE_CELLS: f32 = 2.0;
/// 每帧射线总预算，由 `ddgi_seal` 均分给本帧的工作集（rpp = 预算 / 工作集，钳 [1,256]）。
/// 权威值在本文件，Rust 侧由 `wesl_consts.rs` 启动时解析。
const DDGI_RAY_BUDGET: u32 = 65536u;
const DDGI_SHADOW_T_MAX: f32 = 8192.0;
const DDGI_SHADOW_BIAS: f32 = 0.5;
/// 着色点从体素表面再外推的量（体素）。
/// 必须与 Rust `brickmap::dda::wgsl_consts::SHADOW_SURFACE_EPS` 一致。
const SHADOW_SURFACE_EPS: f32 = 0.03125;
/// 命中自发光体素时的辐亮度增益（albedo · emissive · 本值）；1.0 = 不额外放大。
const DDGI_EMIT_GAIN: f32 = 1.0;
const DDGI_PI: f32 = 3.14159265;
const DDGI_HYSTERESIS: f32 = 0.85;
const DDGI_IRRAD_GAMMA: f32 = 1.0;
const DDGI_BIG_CHANGE: f32 = 0.2;
const DDGI_BRIGHTNESS: f32 = 1.0;
const DDGI_MIN_STEP: f32 = 0.0009765625;
const DDGI_CHANGE_DROP: f32 = 0.0;
const DDGI_DELTA_CLAMP: f32 = 0.25;
/// GI 射线没打到几何时注入的天空辐亮度系数（`ddgi_cast` 的 miss 分支）；1.0 = `sky_rgb()` 原样注入。
/// 只作用于 cast，不影响采样侧的兜底取值。
const DDGI_SKY_RADIANCE_SCALE: f32 = 1.0;
// 世界之外全是天空（纯色环境）：向上与向下逃逸一律注入天空纯色。
const DDGI_RAY_BIAS: f32 = 0.5;
/// 天光环境项：该像素没有任何级联探针覆盖、或采样置信度过低时兜底。
const DDGI_SKY_AMBIENT: f32 = 0.05;
/// 天光常量的地板：`conf = 1`（有数据）时保留 k 倍这一项。
/// 必须写成 `mix(1.0, DDGI_AMBIENT_FLOOR, conf)`（语义 `1 - (1-k)·conf`）。
const DDGI_AMBIENT_FLOOR: f32 = 0.01;
/// 采样置信度区间：`conf = smoothstep(LO, HI, cov)`，只用来在无可用数据时补上天光常量。
const DDGI_CONF_LO: f32 = 0.0;
/// `cov` 的上界：conf = smoothstep(LO, HI, cov)。
const DDGI_CONF_HI: f32 = 0.5;
/// 探针射线命中时的辐亮度下限，必须为 0。
const DDGI_CAST_FLOOR: f32 = 0.0;

// 收敛跳帧阈值：`age` 达阈值才算「已收敛」。
const DDGI_SKIP_AGE: vec4<u32> = vec4<u32>(32u, 24u, 16u, 12u);
/// 换主（相机滚动）时把 age 打散到 `[1, N]` 的相位抖动周期。
/// 约束：必须 `1 + N - 1 < DDGI_SKIP_AGE`（最小 12）。
const DDGI_BAKE_STAGGER: u32 = 8u;
/// 每个 LOD 的强制刷新周期（帧，按槽位错峰）：`(frame + slot) % 周期 == 0` 的帧把该探针放进 worklist。
/// 四项必须全取奇数。
const DDGI_REFRESH_PERIOD: vec4<u32> = vec4<u32>(65u, 97u, 129u, 161u);
/// 「屏幕使用」粘滞窗口（帧）：着色侧把采样到的探针置为本值（幂等写入），`ddgi_sort` 每帧 -1。
/// 取值必须 ≥ 记账抽样的轮转周期（DDGI_USAGE_SCAN_MASK + 1）。
const DDGI_USAGE_STICKY: u32 = 32u;
/// 记账抽样的掩码：`((px + frame) & MASK) == 0 || ((py + frame) & MASK) == 0` 的像素才记账
/// （列带 ∪ 行带）。必须是 2ⁿ-1，且 `MASK+1 ≤ DDGI_USAGE_STICKY`。
const DDGI_USAGE_SCAN_MASK: u32 = 7u;
/// 采样门限：跳过 age=0 那一帧（判据 `age < MIN`，值必须 = 1）。
const DDGI_MIN_SAMPLE_AGE: u32 = 1u;
/// collect 每探针的纹素线程数 = 从纹素数派生（16 irr(4×4) + 64 depth(8×8) = 80）。
/// 必须是派生值（`ddgi_seal` 用它算 collect 的 dispatch 规模）。
const DDGI_COLLECT_THREADS: u32 = DDGI_IRR_TEXELS * DDGI_IRR_TEXELS
  + DDGI_DEPTH_TEXELS * DDGI_DEPTH_TEXELS;

// 样本下标 = 全局射线编号（cast 的 tid）；容量 = DDGI_RAY_BUDGET（ddgi_samples 槽数）。
