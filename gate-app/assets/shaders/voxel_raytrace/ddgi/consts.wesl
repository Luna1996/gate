/// 辐照度图每探针 4×4（= 16 纹素）。
/// 纹素数同时决定图集尺寸与 DDGI_COLLECT_THREADS 的一个加数（见该常量）。
const DDGI_IRR_TEXELS: u32 = 4u;
/// 深度图每探针 8×8（= 64 纹素；每纹素在八面体图上约 22° 角宽）。
/// 深度纹素与射线按「帧轮换」绑定：每个探针每帧只写 rpp 个纹素（见 collect 与 ddgi_cast）。
const DDGI_DEPTH_TEXELS: u32 = 8u;
// 图集布局：每层 `DDGI_PROBES_PER_LAYER_AXIS²` 个探针，共 `DDGI_ATLAS_LAYERS` 层。
//
// ️ **层内轴 / 层数 / worklist 下标位宽必须一起改**：容量 = 层数 × 轴向²，而下标位宽要装得
// 下容量。不一致会让图集 layer 内寻址错位，表现为大面积无 GI，且没有任何编译/运行时报错。
//
// **权威值在本文件**：Rust 侧由 `wesl_consts.rs` 启动时解析，不变量（容量 ≥ 实际总槽位、
// 容量 ≤ 下标位宽）由 `cargo test -p gate-render` 守着。
const DDGI_PROBES_PER_LAYER_AXIS: u32 = 40u;
/// 图集层数（纹理 array 层数；Rust 按它定显存与「槽位 → 层」映射）。
/// 容量 = 本值 × 1600。nuke.vox 下总槽位约 409732（LOD0 池 352256 + LOD1~3 的规则网格，
/// 世界网格向外扩 DDGI_GRID_MARGIN 后 LOD1~3 每轴 dims 至多 +1）⇒ 272 层 = 435200 槽。
/// 上限还受 worklist 的 19 位 cell 下标约束（524288），272×1600 远在其下。
const DDGI_ATLAS_LAYERS: u32 = 272u;
const DDGI_PROBES_PER_LAYER: u32 = DDGI_PROBES_PER_LAYER_AXIS * DDGI_PROBES_PER_LAYER_AXIS;
const DDGI_LOD_COUNT: u32 = 4u;
// meta word = age(低 8bit) | ENABLED(烘焙放了探针) | ACTIVE(本 cell 或 6 邻接有体素/物体，
// 且不在更细 LOD 覆盖内) | 邻接可见性掩码(bit10-15)；0 = 无探针哨兵
const DDGI_META_ENABLED: u32 = 256u;
const DDGI_META_ACTIVE: u32 = 512u;
/// 邻接可见性掩码在 meta 里的位偏移（占 bit10-15，共 6 位）。
/// 位序（相邻两位同轴，故「反向」= `bit ^ 1u`）：0=-X, 1=+X, 2=-Y, 3=+Y, 4=-Z, 5=+Z。
/// bit=1 = 从本探针沿该轴走 cs 不撞几何。由 `ddgi_sort` 每帧为活跃探针写入（6 条 cs 长射线），
/// `ddgi_probe_blend_irr` 读来门控空间混合 —— 见 `ddgi_nbr_vis_mask` 的说明。
const DDGI_META_NBR_SHIFT: u32 = 10u;
/// 掩码取值域（6 位）。
const DDGI_META_NBR_MASK: u32 = 0x3Fu;
/// 「邻接可见性掩码已由本帧 sort 算过」的有效位（bit16）；未置位时消费侧不做门控。
const DDGI_META_NBR_VALID: u32 = 65536u;
const DDGI_AGE_MAX: u32 = 255u;
/// worklist `.w` 的位段打包：age(低 8 位) | cell 下标(19 位) | lod(2 位)。
///
/// 下标是**本级内**的线性下标。19 位 = 524288 ≥ 图集容量（DDGI_ATLAS_LAYERS × 1600），够用；
/// 用 16 位会截断 LOD0 的下标（nuke 下 34 万 cell）⇒ 同一个图集纹素被多个远处探针反复覆写、
/// 下标超出者图集恒空 ⇒ 大片纯黑。该容量关系由
/// `wgsl_compile.rs::ddgi_worklist_pack_covers_atlas_capacity` 守着。
const DDGI_WL_IDX_MASK: u32 = 0x7FFFFu;
const DDGI_WL_LOD_SHIFT: u32 = 27u; // = 8(age) + 19(idx)
const DDGI_NO_PROBE: u32 = 0xFFFFFFFFu;
/// 辐照度**稳态**混合系数 = 每次更新向新样本靠拢的比例 ⇒ **稳态窗口 ≈ 1/本值 次更新**。
///
/// 只在探针年龄到上限（`age == DDGI_AGE_MAX`）之后生效；之前是 `ddgi_age_alpha` 的 1/age
/// 等权斜坡。窗口必须远大于方向轮换周期（64/rpp 帧），否则稳态残噪与拍频都会逐帧显形。
/// 世界编辑的快速跟随不靠它：`ddgi_sort` 见 `dirty_min.w` 会把 age 压回 1，约 30 帧重新收敛。
const DDGI_ALPHA: f32 = 1.0 / 3000.0;
/// 深度图时域系数（图集两通道：mean / std）。
/// **这是稳态地板**：纹素自己的更新次数还浅时走 1/该纹素的更新次数（等权平均），alpha 取两者
/// 较大者，见 `DDGI_DEPTH_CONV_SAMPLES` 与 collect 的深度分支。取 1/255 与 `DDGI_AGE_MAX` 同量级。
const DDGI_DEPTH_ALPHA: f32 = 1.0 / 255.0;
/// 深度图「按纹素自己的更新次数收敛」的计数上限（见 collect 的深度分支）。
///
/// 深度纹素每 ⌈64/rpp⌉ 帧才拿到 1 条样本，而 `ddgi_age_alpha` 的时钟是探针年龄（每帧 +1）——
/// age 会远早于纹素的样本数涨到收敛阈值。按纹素自己的计数后，前 N 次更新是 1/n 的等权平均
/// （与辐照度用 1/age 同思路），alpha 取「age 项」与 1/n 的较大者。
/// 取 255 与 `DDGI_AGE_MAX` 对齐：255 在 Rgba16Float（10 位尾数）里整数精确，计数不会因 f16
/// 舍入抖动；更长的窗口也意味着残噪更低、抖动频率远低于帧率。
const DDGI_DEPTH_CONV_SAMPLES: f32 = 255.0;
/// 末级兜底 r4（"本点一侧的探针场插值"）启用的**最小同侧余弦权重和** Σ wtri·wn
/// （见 ddgi_sample 的 r4）。
/// 0.05 ≈「一个 wtri=0.2 的角、wn≈0.25」：正好挡住只有绕角掠射角（wn≈0，其辐照度来自墙外/
/// 邻室，用它插值等于穿墙漏光）的情形，又不会挡掉薄几何/犄角旮旯（有明确同侧探针的）。
const DDGI_FIELD_MIN_W: f32 = 0.05;
/// Chebyshev 软遮挡里 std 的**绝对**下限（体素），只用来防「std ≈ 0 时退化成刀锋」。
///
/// 不能写成占 mean 的比例：那样下限随探针距离线性增长，而 `delta`（墙厚 + 采样点离墙的距离）
/// 与探针距离无关 ⇒ 远处/粗级探针的 wd 被系统性抬高，归一化平均下一个跨墙角就能独占结果 ⇒
/// 漏光且探针越远越明显。DDGI 原版是 `variance/(variance + delta²)`，没有距离相关下限。
/// 这里 soft 由 collect 存的样本序列 std 主导，本值只防 0。
const DDGI_DEPTH_SOFT_MIN: f32 = 0.5;
/// 空间混合（探针间去噪）里"自身"的权重参照。邻居权重和按探针实际间距加权
/// （见 `ddgi_probe_blend_irr`），典型约 3（6 邻居 × ~0.5）→ 2.0 表示自身约占 40%。
const DDGI_BLEND_SELF: f32 = 2.0;
/// 邻居纹素的**覆盖度**下限（irr 图集第 4 通道 = 该纹素被写过的时间累积，见 collect）。
/// 空间混合必须能拿暗邻居去稀释亮邻居（否则跨墙拿到天光的探针会在自己的投影处形成亮斑）；
/// 而 `DDGI_CAST_FLOOR = 0` 时"真黑"与"没数据"的值都是 0 ⇒ 只能按"被写过没有"过滤。
const DDGI_TEXEL_MIN_COVERAGE: f32 = 0.5;
/// 前后判定（wn）的锐度：wn = clamp(dot(n,-dir)/此值, 0, 1)。
/// 1.0 = 真正的余弦（60° 处 0.5、78° 处 0.2）；调小会把掠射探针（墙角/凸边外侧看到开阔空间
/// 的那些）按满权重混进来。
/// 此值不改变拒绝集合（wn=0 恒为 dot<=0），只改相对权重 ⇒ 调它不会产生新黑区。
const DDGI_NORMAL_BIAS: f32 = 1.0;
/// 采样点沿法线的外推量 = 该 LOD 的 cell 边长 × 此系数（参考实现里的 normal_bias 语义）。
/// 必须随间距缩放：固定体素量相对粗级 cell 等于贴在表面上，会让整面被判「探针在背面」。
const DDGI_BIAS_CELLS: f32 = 0.1;
// ============================================================================
// 逐角几何遮挡（真·线段求交）参数 —— 见 ddgi_sample_lod 调用点。
//
// wn 与 wd 都是代理量（前者纯方向，后者是探针自己那张深度图，检验的是"探针沿该方向的最近
// 表面"而非"探针→采样点这条线段"）⇒ 隔着墙/楼板/薄板的探针两道闸门都放行。这里补一层线段
// 求交把这类角剔除。线段 = 「采样点所在 cell 的 8 个角 → 采样点」，长度 ≤ cs·√3。
// 遮挡走 `world_raycast` 的完整层次 DDA（与主可见性 trace_scene 同一套遍历与实体判定），
// 线段多长就走到多长 ⇒ 不需要距离上限，也不需要固定步数（真挂死由 trace_chunk 的 65536 步
// 安全网兜底，超限返回 miss = 保守判可见）。
// ============================================================================
/// 线段终点偏置（体素）：从 `max_t = 线段长` 里扣除，停在采样点前。采样点已沿法线外推过
/// （dda_main 的 n_off ≥ 0.5 再叠 ddgi_sample_lod 的 DDGI_BIAS_CELLS·cs ≥ 1.6 体素@LOD0），
/// 留 1 体素既不会擦到接收面自身，也不会吃掉贴在接收面上的那面薄墙（穿墙点比终点早 ≥1.6 体素）。
const DDGI_OCCL_END_BIAS: f32 = 1.0;
/// 逐角遮挡的**权重阈值 ε**（丢弃语义）：角最终权重 `w = wtri·wn_w·wd < ε` ⇒ 当作 0 丢弃，
/// 不计入 total/wsum（跳过遮挡求交）。误差被 ε 界定，最坏只是少贡献一点弱光。
/// "不贡献"必须与"不检测"一致：跳过求交却保留权重等于绕过几何遮挡 ⇒ 接缝/薄墙弱权重角漏光。
/// A/B 档：0（= 无条件检测全部 8 角，正确但 +1.2ms）/ 0.005（默认）/ 0.01 / 0.02。
const DDGI_OCCL_W_MIN: f32 = 0.005;
const DDGI_T_MAX: f32 = 8192.0;
/// 级联混合带宽度（以当前级 cell 边长为单位）：距 LOD 盒边界小于 `cs × 此值` 时在该级与相邻级
/// 之间过渡，消除「跨过盒边界突然换一套光照」的硬边。
/// 取 1.0 = 论文的 "one grid cell"：过渡带就是一个格宽，不是越宽越好（太宽会让细级精度被粗级
/// 提前稀释）。边界两侧恰好都是 0.5/0.5 ⇒ 跨边界连续（推导见 ddgi_cascade_blend）。
const DDGI_CASCADE_BLEND: f32 = 1.0;
/// 「覆盖内 GI ↔ 覆盖外天光兜底」过渡带的宽度（以最粗级 cell 为单位）。
/// 着色侧用它把 `amb` 从"盒内由 conf 决定"平滑过渡到"盒外全量天光"：不可用布尔直接跳变，
/// 否则级联覆盖边界上会出现"有 GI / 无 GI"的硬边。
/// 2.0 = 最粗级两格（LOD3 的 cs = 128 体素 → 5.12m）。
const DDGI_CASCADE_FADE_CELLS: f32 = 2.0;
/// 每帧射线总预算，由 `ddgi_seal` 均分给本帧的工作集（rpp = 预算 / 工作集，钳 [1,256]）。
///
/// **权威值在本文件**：Rust 侧由 `wesl_consts.rs` 启动时解析（samples 缓冲容量按
/// `max(预算, 总槽位)` 定容，见 `ddgi::DdgiConsts::sample_bytes`）。
/// 工作集是屏幕相关的（见 DDGI_USAGE_STICKY）⇒ rpp 正比于本值：加预算使单帧估计方差 ↓√rpp、
/// 方向轮换周期 ↓1/rpp，代价是 cast 帧时随射线数线性上涨。
const DDGI_RAY_BUDGET: u32 = 65536u;
const DDGI_SHADOW_T_MAX: f32 = 8192.0;
const DDGI_SHADOW_BIAS: f32 = 0.5;
/// 着色点从体素表面再外推的量（体素）。见 dda_main 阴影射线处注释：`+n×0.5` 恰好落在
/// 面平面上，负向面的坐标是整数 → DDA floor 落回自身体素 → 自命中被当成「无遮挡」→ 漏光。
/// 必须与 Rust `brickmap::dda::wgsl_consts::SHADOW_SURFACE_EPS` 一致（有对齐测试）。
const SHADOW_SURFACE_EPS: f32 = 0.03125;
const DDGI_EMIT_GAIN: f32 = 4.0;
const DDGI_PI: f32 = 3.14159265;
const DDGI_HYSTERESIS: f32 = 0.85;
const DDGI_IRRAD_GAMMA: f32 = 1.0;
const DDGI_BIG_CHANGE: f32 = 0.2;
const DDGI_BRIGHTNESS: f32 = 1.0;
const DDGI_MIN_STEP: f32 = 0.0009765625;
const DDGI_CHANGE_DROP: f32 = 0.0;
const DDGI_DELTA_CLAMP: f32 = 0.25;
/// GI 射线没打到几何时注入的天空辐亮度系数（`ddgi_cast` 的 miss 分支）。
/// 它是**诊断开关**：探针回路里唯一的非零种子就是这条 miss 路径（命中几何的射线回读
/// `alb·gi/π`，而 `DDGI_CAST_FLOOR = 0`）⇒ 置 0 会让室外靠 GI 点亮的面明显变暗。
/// 默认 1.0 = 天空按 `sky_rgb()` 原样注入。它只作用于 cast，不影响采样侧的兜底取值。
const DDGI_SKY_RADIANCE_SCALE: f32 = 1.0;
// 世界之外全是天空（纯色环境，不分"地面/虚空"）：向上与向下逃逸一律注入天空纯色。
// 这是本引擎的世界模型；世界底部下方没有几何 ⇒ 那带探针天然拿到半球天空，是模型的结果。
const DDGI_RAY_BIAS: f32 = 0.5;
/// 天光环境项：该像素没有任何级联探针覆盖、或采样置信度过低（无数据/仅剩边缘探针）时兜底。
/// 数据可信的级联内像素一律由 DDGI 提供环境光（带遮挡）：无光室内 → 探针辐照度 ≈ 0。
const DDGI_SKY_AMBIENT: f32 = 0.05;
/// 天光常量的**地板**：`conf = 1`（有数据）时保留 k 倍这一项，而不是整个乘没。
/// 着色式 `col = alb·(sun + amb·ao + gi)`：无光室内三项同时为 0 ⇒ 恒 0 的黑区（auto exposure
/// 对它无效），本项是那份缺失的多次反弹照明的常量替身。室外 GI 是 0.2~0.4 量级，本项是
/// sky·0.05·k 量级 ⇒ 不抹平室内外差异；要更黑往 0 调，0 = 纯黑。
/// ⚠️ 必须写成 `mix(1.0, DDGI_AMBIENT_FLOOR, conf)`（语义 `1 - (1-k)·conf`）：conf=0 → 满额兜底，
/// conf=1 → k。写成 `mix(k, 1.0, conf)` 会让 conf=1 时返回 1.0、且结果与 k 无关。
const DDGI_AMBIENT_FLOOR: f32 = 0.01;
/// 采样置信度区间：`conf = smoothstep(LO, HI, cov)`，只用来在"基本没有可用数据"时补上天光常量，
/// 不缩放 GI 幅度。
/// cov 的语义是"本点周围有没有探针数据"，与朝向/可见性无关；不能换成 wsum（后者额外乘了照明
/// 闸门，凹角/远处粗级的存活权重天然低）。
const DDGI_CONF_LO: f32 = 0.0;
/// `cov` 的上界：conf = smoothstep(LO, HI, cov)。
/// 带宽必须给足：取值太小（如 0.02）等于"全有或全无"，amb 会在覆盖边界一步跳变 ⇒ 肉眼可见的
/// "有 GI / 无 GI"割裂。cov 满值约 1（8 个角的 wtri 之和），0.5 对应最后 1~2 个角失效的范围。
/// 不能换成 wsum：凹角/远处粗级的存活权重天然低，会"本该黑"的地方判成不可信而补回天光。
const DDGI_CONF_HI: f32 = 0.5;
/// 探针射线命中时的辐亮度下限。**必须为 0**：任何非零下限都会被反馈回路放大成
/// ≈ alb·sky·FLOOR/(1-alb) 的"室内自发光"，封闭空间永远压不黑。
/// 首帧的种子由**能看到天空的射线**提供（miss → sky_rgb），所以不需要这个下限。
const DDGI_CAST_FLOOR: f32 = 0.0;

// 收敛跳帧参数（Douglas can_skip_update）：`age` 达阈值才算"已收敛"（不再因为"刚成型"而
// 每帧投线）。工作集的三个来源 = DDGI_USAGE_STICKY（屏幕是否用到）∪ 本阈值（未收敛）
// ∪ DDGI_REFRESH_PERIOD（按槽位错峰的强制刷新，兜住"从没上过屏"的探针）。
const DDGI_SKIP_AGE: vec4<u32> = vec4<u32>(32u, 24u, 16u, 12u);
/// 换主（相机滚动）时把 age 打散到 `[1, N]` 的相位抖动周期：按槽位号把"当帧直写"错开到 N 帧，
/// 避免整圈 cell 同帧一起直写而在 LOD 分界线留下一条完整的浅灰带（槽位号连续 ⇒ 世界 cell 相邻
/// ⇒ 取模后相位分散）。N 越大越分散，代价是旧值的近似最多停留 N 帧。
/// 约束：必须 `1 + N - 1 < DDGI_SKIP_AGE`（最小 12），否则换主的探针会被判 converged 而停投线。
const DDGI_BAKE_STAGGER: u32 = 8u;
/// 每个 LOD 的强制刷新周期（帧，按槽位错峰）：`(frame + slot) % 周期 == 0` 的帧把该探针放进
/// worklist，兜住"从来没上过屏"的探针（也保证世界编辑后远场能跟上）。
///
/// ⚠️ **四项必须全取奇数**：图集是 ping-pong（BG4 采样侧 / BG5 写入侧，每帧翻转一次），帧号的
/// 奇偶决定这一帧读哪个缓冲、写哪个缓冲。周期取偶数时 `(frame + slot) % P == 0` 的解全是同一
/// 奇偶性的帧 ⇒ 这类探针永远只写同一个缓冲、另一个缓冲一次也不被刷新，而着色每帧轮换读两个
/// 缓冲 ⇒ 显示值逐帧在「当前值」和「一个非常陈旧的值」之间跳（`due` 刷新只落在这类探针上）。
/// 取奇数后相邻两次刷新落在不同奇偶性的帧上，两个缓冲都进入 EMA 链。
const DDGI_REFRESH_PERIOD: vec4<u32> = vec4<u32>(65u, 97u, 129u, 161u);
/// 「屏幕使用」粘滞窗口（帧）：着色侧把采样到的探针置为本值（幂等写入），`ddgi_sort` 每帧 -1。
/// 探针只要在最近 <本值> 帧里被屏幕采样过就留在工作集里 —— 与它的屏幕 footprint 大小无关。
/// 于是射线跟着像素走（不看的方向、墙后的探针自动出局），`rpp = 预算 / 工作集` 能上到 ~20。
/// 取值必须 ≥ 记账抽样的轮转周期（DDGI_USAGE_SCAN_MASK + 1）；32 帧（0.5s）留了 4× 余量。
const DDGI_USAGE_STICKY: u32 = 32u;
/// 记账抽样的掩码：`((px + frame) & MASK) == 0 || ((py + frame) & MASK) == 0` 的像素才记账
/// —— **列带 ∪ 行带**，必须取或：取与会只选中对角线像元，"屏幕 footprint 只有几像素"的探针
/// （薄几何 / 细缝）就永远进不了工作集。
/// 每帧约 2/(MASK+1) 的像素被采到（满值写入、幂等），且每个像素每 MASK+1 帧至少被采一次。
/// 约束：必须是 2ⁿ-1（掩码语义），且 `MASK+1 ≤ DDGI_USAGE_STICKY`（否则探针会先被每帧 -1 抹掉）。
const DDGI_USAGE_SCAN_MASK: u32 = 7u;
/// 采样门限：探针新生（首次烘焙）时 age 从 0 起步、图集还没写 ⇒ 跳过 age=0 那一帧。
/// **值必须 = 1**：判据是 `age < MIN`，MIN=1 恰好只跳 age=0。
/// 换主（bake 把 age 压到 1 并保留图集历史）与世界编辑都不重置 age，故都不会被这里跳过。
const DDGI_MIN_SAMPLE_AGE: u32 = 1u;
/// collect 每探针的纹素线程数 = 从纹素数派生（16 irr(4×4) + 64 depth(8×8) = 80）。
///
/// 必须是派生值，不能写死：`ddgi_seal` 拿它算 collect 的 dispatch 规模，与纹素数不同步会让
/// 多出的线程落进 depth 分支并越出本探针自己的纹素块，改写邻居探针的深度纹素。
const DDGI_COLLECT_THREADS: u32 = DDGI_IRR_TEXELS * DDGI_IRR_TEXELS
  + DDGI_DEPTH_TEXELS * DDGI_DEPTH_TEXELS;

// 样本下标 = 全局射线编号（cast 的 tid 本身就是它）：seal 用全局统一 rpp，样本区不分段；
// 容量 = DDGI_RAY_BUDGET = ddgi_samples 的槽数。
