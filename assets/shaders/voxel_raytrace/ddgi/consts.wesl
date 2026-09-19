/// 辐照度图每探针 4×4（= 16 纹素）。
const DDGI_IRR_TEXELS: u32 = 4u;
/// 深度图每探针 8×8（= 64 纹素；每纹素在八面体图上约 22° 角宽）。
const DDGI_DEPTH_TEXELS: u32 = 8u;
// 图集布局：每层 `DDGI_PROBES_PER_LAYER_AXIS²` 个探针，共 `DDGI_ATLAS_LAYERS` 层。
// 层内轴 / 层数 / worklist 下标位宽必须一起改（容量 = 层数 × 轴向² 须装进下标位宽）；Rust 侧由 `wesl_consts.rs` 启动时解析。
const DDGI_PROBES_PER_LAYER_AXIS: u32 = 40u;
/// 图集层数（纹理 array 层数；Rust 按它定显存与「槽位 → 层」映射）。
/// 容量 = 层数 × `DDGI_PROBES_PER_LAYER`，必须装下五级槽位总和；当前账目（castle.vox）：
/// LOD0 细通道 1382 砖 × 64 = 88,448、LOD1~4 = 383,924 → 472,372 槽 ≤ 296 × 1600 = 473,600。
const DDGI_ATLAS_LAYERS: u32 = 296u;
const DDGI_PROBES_PER_LAYER: u32 = DDGI_PROBES_PER_LAYER_AXIS * DDGI_PROBES_PER_LAYER_AXIS;
const DDGI_LOD_COUNT: u32 = 5u;
/// 按 chunk 领段的那一级（其余级按世界 AABB 环面映射）：它的槽数是探针池高水位，不是空间盒 dims 乘积。
const DDGI_CHUNK_LOD: u32 = 1u;
// ---- LOD0（最细级）的段粒度：只给「粗级放不下探针」的瓦片领段 ----
// 段在运行时认领：`ddgi_sample` 把各级都试过仍拿不到有效样点时，认领该点所在的瓦片
// （见 `helpers.wesl::ddgi_lod0_claim`）。粒度取 8³（不是 16³）：一条 3 voxel 宽的过道只用到
// 瓦片里一两个 4³ cell，按 16³ 领段会把 4~8 倍容量浪费在墙上。
/// 每瓦片每轴的 LOD0 cell 数（8 / 4）。
const DDGI_LOD0_CELLS_PER_BRICK: i32 = 2;
/// 每 chunk 每轴的段粒度瓦片数（256 / 8）。
const DDGI_LOD0_BRICKS_PER_AXIS: i32 = 32;
/// 每瓦片的段长（槽）= `DDGI_LOD0_CELLS_PER_BRICK³`。
const DDGI_LOD0_SLOTS_PER_BRICK: u32 = 8u;
/// LOD0 认领状态（`ddgi_lod0_state`）的 word 布局：`[ALLOC]` = 段分配游标（= 下一个空闲段基址，
/// 也就是已领的槽数）；`[CLAIM, …)` = 每 chunk `bricks_per_chunk / 32` 个 word 的认领位图。
const DDGI_LOD0_STATE_ALLOC: u32 = 0u;
const DDGI_LOD0_STATE_CLAIM: u32 = 1u;
/// 认领门槛：沿表面法线往空气里退这么多 voxel，任一位置能摆下「十字档」才领段。
/// 挡掉树叶/栅栏这类多孔表面 —— 那里空气是 1~2 voxel 的碎片，粗级服务不了、细级也放不下探针，
/// 但数量极大，不挡会把细级预算吃光（探针全铺在树冠上）。也顺带保证 1~2 voxel 的墙缝不领段。
const DDGI_LOD0_CLAIM_BIAS: i32 = 3;
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
/// 「需要细级」的**几何**判据上界（voxel）：法线之外的两个轴上，至少有一个方向的空腔宽度
/// 落在 `[DDGI_PROBE_MIN_BLOCK, 本值]` 之间 ⇒ 认领该点所在的 8³ 瓦片。
///
/// 为什么用几何而不是"GI 值黑"：值黑会被瞬态主导（启动、编辑后探针重烘 ⇒ 到处都黑 ⇒ 一口气领掉
/// 一大片），而稳态下窄过道又常被更粗一级垫出一个高于阈值的值 ⇒ 永远不认领（"必须放体素才认领"）。
/// 几何判据与亮度、与瞬态都无关：开阔地形 / 大厅 / 宽沟的侧向空腔很长 ⇒ 不认领；
/// 1~2 voxel 的缝宽于上界之外（`DDGI_PROBE_MIN_BLOCK` 以下，十字档放不下）⇒ 也不认领。
/// 取 16 = **一个 chunk cell 的边长**：粗级格子连"和自己一样大的空腔"都表达不了，比它更窄的都该要细级。
/// （早先取 8 = 半个 cell，结果十几格宽的过道下半段被排除掉了 —— 那类空间粗级同样表达不了。）
/// 0 = 关闭认领。
const DDGI_CLAIM_CAVITY: i32 = 16;
/// 认领的**稀疏扫描**掩码：认领是按**瓦片**做的，不需要每个像素都跑几何判定（那最多要 2×(W+1) 次
/// 树查询）。用「着色点哈希 ^ 帧号」抽 1/(本值+1) 的像素做判定，并逐帧轮转 ⇒ 每个可见瓦片几帧内
/// 必然被覆盖到，而几何行走的总量降到可以忽略。
const DDGI_CLAIM_SCAN_MASK: u32 = 1023u;
/// 认领细级的**预热期**（帧）：启动阶段 brickmap 还在流式建起来，几何判据此刻不可靠；
/// 另外头几帧的探针也没投线。它只压启动瞬间，**不改变任何地点"有没有资格认领"**。
const DDGI_CLAIM_WARMUP_FRAMES: u32 = 90u;
/// 认领细级的**相机半径**（voxel）：只有相机附近（看得见的细节）才认领 LOD0 瓦片。
/// 细级只对近处可见细节有意义，远处/背对的区域领了也白占池子。已有探针不受影响
/// （变的只是"哪些地方有资格被认领"），池子消耗速度因此随视野可控。0 = 不限半径。
const DDGI_CLAIM_RADIUS: f32 = 192.0;
/// 「被几何完全包住」的极暗底（天光环境的倍数）：针对方位全被占满的点（草根、极窄缝）——
/// 那里的 0 不是"没有光"，而是二值可见性（深度占用/逐角求交）的产物，堆探针也救不回来。
/// 这里补一个**不参与 GI 传播**的常量辐射（不吃遮挡、不注入相邻表面，因此不会漏光）。
/// 只在 `cov` 高而 `wsum ≈ 0`（探针有数据但一个角都看不到）时生效 ⇒ 无光暗室（采样成功、值就是 0）
/// 不受影响，仍是纯黑。0 = 关闭。
const DDGI_DARK_FLOOR: f32 = 0.01;
/// 上面那个底生效所需的覆盖度下限（`ddgi_dbg_cov`）。
const DDGI_DARK_FLOOR_COV: f32 = 0.5;
/// 探针放置的最后一档块边长（voxel）：4³ 档要求对齐 4 voxel 网格，3 voxel 宽的过道只有这一档放得下。
/// 取 3 是下限——更窄的缝（1~2 voxel）不放探针，否则探针会把缝两侧的照明互相渗过去（漏光）。
const DDGI_PROBE_MIN_BLOCK: i32 = 3;
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
/// 上面那个外推量的绝对上限（voxel）：窄缝里 `cs × DDGI_BIAS_CELLS`（LOD0 = 1.6）会把采样点推进对面
/// 几何内部，使逐角可见性求交对全部 8 角都判被挡。
const DDGI_BIAS_MAX_VOXELS: f32 = 0.5;
// ============================================================================
// 逐角几何遮挡（真·线段求交）参数：线段 = 采样点 cell 的 8 角 → 采样点，长度 ≤ cs·√3。
// ============================================================================
/// 线段终点偏置（体素）：从 `max_t = 线段长` 里扣除，停在采样点前。
/// 必须 < 采样点到表面的距离（着色侧 n_off ≥ 0.5 + 本文件的 DDGI_BIAS_MAX_VOXELS ≥ 0，即 ≥ 0.5），
/// 否则探针背后的薄板/墙面会落在被扣掉的那一段里、被判成"可见"（漏光）。
const DDGI_OCCL_END_BIAS: f32 = 0.25;
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
/// 命中面的直射太阳项增益（`ddgi_cast`：`alb · sun_c · ndl · sun · 本值`）。
/// 这是 GI 唯一的**直射**光源 —— 没有它，任何看不到天空的表面（草丛根部、过道内墙、阴影地面）
/// 都只能自洽地维持全黑，加密探针也救不回来。语义与着色侧直射项一致（不含 1/π）。
/// 调小 = 反弹光更弱（更"噪点少但暗"），调大 = 间接光更亮（更容易过曝）。
const DDGI_CAST_SUN: f32 = 1.0;

/// LOD0↔chunk 级的接缝混合带宽度（以 LOD0 严格采样的覆盖度为单位）：
/// LOD0 是逐砖覆盖的，瓦片边界处 8 角里只有一部分在已认领的砖里 ⇒ 直接取用会出现明显接缝。
/// 覆盖度低于本值时按比例混到 chunk 级结果上，= 1 时纯 LOD0。
const DDGI_LOD0_SEAM_COV: f32 = 1.0;

/// 深度纹素总数 = 一条探针扫完整个八面体所需的射线数。
/// 收敛判据必须要求 `age × rpp ≥ 本值`（见 `ddgi_sort`）：投不满一圈时，图集里总有一批方向
/// 既没有深度记录、也没有辐照度覆盖度，采样侧的数据闸门会把那些角全部剔掉（整片 GI 变黑）。
const DDGI_DEPTH_SWEEP: u32 = DDGI_DEPTH_TEXELS * DDGI_DEPTH_TEXELS;

/// 收敛跳帧阈值：`age` 达阈值才算「已收敛」，是叠加在 `DDGI_DEPTH_SWEEP` 之上的最小投线次数
/// （用来做逐级收敛快慢的档位）。下标 = `ddgi_lod_param_idx(lod)`。
const DDGI_SKIP_AGE: vec4<u32> = vec4<u32>(32u, 24u, 16u, 12u);
/// 换主（相机滚动）时把 age 打散到 `[1, N]` 的相位抖动周期。
/// 约束：必须 `1 + N - 1 < DDGI_SKIP_AGE`（最小 12）。
const DDGI_BAKE_STAGGER: u32 = 8u;
/// 每个 LOD 的强制刷新周期（帧，按槽位错峰）：`(frame + slot) % 周期 == 0` 的帧把该探针放进 worklist。
/// 四项必须全取奇数。
const DDGI_REFRESH_PERIOD: vec4<u32> = vec4<u32>(65u, 97u, 129u, 161u);
/// 上面两张逐级表的索引：LOD0（最细级，探针最少、最需要收敛）与 LOD1 同档，LOD2~4 依次后移。
fn ddgi_lod_param_idx(lod: u32) -> u32 {
  return max(lod, 1u) - 1u;
}
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
