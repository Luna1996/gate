/// 辐照度图每探针 4×4（= 16 纹素）。
///
/// 【为什么不用原版的 8×8】曾对齐 DDGI 原版试过 8×8 / 16×16（见下面 DEPTH_TEXELS 注释）：
/// 实测**画质没有明显改善**，代价却是显存 ×4、collect 线程 ×4、帧轮换周期 ×4。故回退。
const DDGI_IRR_TEXELS: u32 = 4u;
/// 深度图每探针 8×8（= 64 纹素）。
///
/// 【为什么不用原版的 16×16】曾试过 16×16（每纹素 ~11°，现在 ~22°）：**实测漏光没有明显
/// 改善** —— 说明当时的漏光主因**不在深度角分辨率**，而在别处（射线方向未绑定纹素 / 跨墙指向
/// / 级联硬切；其中跨墙指向对应的旧"借针"机制现已删除）。而代价很实在：深度图集 64MB → 256MB、
/// collect 线程 80 → 320、帧轮换周期 ×4。故回退到 8×8。
const DDGI_DEPTH_TEXELS: u32 = 8u;
// 图集布局：每层 `AXIS × AXIS` 个探针，共 `DDGI_ATLAS_LAYERS`(Rust 侧) 层。
//
// ⚠️ **必须与 Rust 的 `DDGI_ATLAS_PROBES_PER_LAYER_AXIS` 保持一致** —— 不一致会让
// 写图集与读图集的 layer 内寻址错位，表现为**大面积无 GI（按网格边界切开）**。
// 已踩过一次：Rust 改成 40 而这里还是 16，导致大部分探针读到空白。
// 现在有对齐测试守着：`wgsl_compile.rs::ddgi_atlas_layout_matches_rust_consts`。
const DDGI_PROBES_PER_LAYER_AXIS: u32 = 40u;
// 派生而非写死：写死的历史值 256(=16²) 就是上面那次错位的另一半原因。
const DDGI_PROBES_PER_LAYER: u32 = DDGI_PROBES_PER_LAYER_AXIS * DDGI_PROBES_PER_LAYER_AXIS;
const DDGI_LOD_COUNT: u32 = 4u;
// meta word = age(低 8bit) | ENABLED(烘焙放了探针) | ACTIVE(本 cell 或 6 邻接有体素/物体，
// 且不在更细 LOD 覆盖内) | 邻接可见性掩码(bit10-15)；0 = 无探针哨兵
const DDGI_META_ENABLED: u32 = 256u;
const DDGI_META_ACTIVE: u32 = 512u;
/// 邻接可见性掩码在 meta 里的位偏移（占 bit10-15，共 6 位）。
///
/// 位序（**相邻两位同轴**，故「反向」= `bit ^ 1u`）：
///   0 = -X, 1 = +X, 2 = -Y, 3 = +Y, 4 = -Z, 5 = +Z
/// bit=1 = 从本探针沿该轴走 cs 距离**不撞几何**（该方向与邻居之间是通的）。
/// 由 `ddgi_sort` 每帧为活跃探针写入（6 条 cs 长射线），`ddgi_probe_blend_irr` 读来门控
/// 空间混合 —— 见 `ddgi_nbr_vis_mask` 的说明。
const DDGI_META_NBR_SHIFT: u32 = 10u;
/// 掩码取值域（6 位）。
const DDGI_META_NBR_MASK: u32 = 0x3Fu;
/// 「邻接可见性掩码已由本帧 sort 算过」的有效位（bit16）。
/// 未置位时消费侧**不做门控**（退回改造前行为）—— 见 `ddgi_meta_with_nbr` 的说明。
const DDGI_META_NBR_VALID: u32 = 65536u;
const DDGI_AGE_MAX: u32 = 255u;
/// worklist `.w` 的位段打包：age(低 8 位) | cell 下标(19 位) | lod(2 位)。
///
/// 【为什么是 19 位而不是 16】cell 下标是**本级内**的线性下标。dims 改成按世界 AABB 算之后
/// （nuke 下 LOD0 = 122×39×72 = 342576 个 cell），16 位（上限 65535）装不下：
///   · 打包侧 `idx << 8` 会**撞进 lod 那几位**；
///   · collect 侧 `(packed >> 8) & 0xFFFFu` 把下标**截断**成 `idx mod 65536` →
///     `atlas_slot` 折回低 65536 个槽位 → 同一个图集纹素被多个远处探针反复覆写，
///     而下标 ≥ 65536 的探针**图集恒为空** → 采样 wsum>0 但辐照度≈0 → **大片纯黑**，
///     黑区边界恰好落在 LOD0 网格的第 65536 个 cell（dx·dy = 122×39 → z 第 14 格）。
///     这正是「部分区域正常、部分区域全黑」的直接成因。
/// 19 位 = 524288 ≥ 图集容量 `DDGI_ATLAS_LAYERS × AXIS²` = 409600，够用；该容量关系由
/// `wgsl_compile.rs::ddgi_worklist_pack_covers_atlas_capacity` 守着。
const DDGI_WL_IDX_MASK: u32 = 0x7FFFFu;
const DDGI_WL_LOD_SHIFT: u32 = 27u; // = 8(age) + 19(idx)
const DDGI_NO_PROBE: u32 = 0xFFFFFFFFu;
const DDGI_FLAG_ENABLED: u32 = 1u;
const DDGI_FLAG_NO_SURFACES: u32 = 2u;
/// 时域混合系数（每帧向新采样靠拢的比例）。rpp 偏低时单帧估计方差大，亮区边界随噪声
/// 逐帧移动（静止相机也在脉动）。0.06→0.03 未见效 → 再降到 0.015（≈67 帧 / 1.1s）。
/// 这是**稳态**（探针年龄足够大）用的比例；年轻探针走 `ddgi_age_alpha` 的 1/age 快收敛，
/// 见该函数注释 —— 稳态逐帧脉动由这个值负责，所以不要为了"收敛快"去调大它。
const DDGI_ALPHA: f32 = 0.015;
/// 深度图时域系数（图集两通道：mean / std）。射线按「绑定纹素 + 帧轮换」分配后，
/// 每个纹素每 ⌈64/rpp⌉ 帧才拿到一条射线（rpp≈4 → 16 帧），单帧样本极少 →
/// 靠这个系数做跨帧滑动平均收敛（见 collect 的深度分支）。
const DDGI_DEPTH_ALPHA: f32 = 0.03;
/// Chebyshev 软遮挡里 std 的**绝对**下限（体素）。只用来防「std ≈ 0 时退化成刀锋」。
///
/// 【为什么不能是「占 mean 的比例」】旧值 = `mean × 0.02`，即下限随「探针到该方向表面的
/// 距离」线性增长。后果：探针离墙越远 soft 越大，跨墙角的 `wd = soft²/(soft² + delta²)`
/// 就越大 —— 而 `delta`（墙厚 + 采样点离墙的距离）与探针距离**无关**。于是远处/粗级探针的
/// wd 被系统性抬高；`ddgi_sample_lod` 末尾做的是**归一化**平均，只要有一个跨墙角通过，它
/// 就能独占整块结果 ⇒ 漏光，且「探针越远越明显」（对上实机：LOD0 轻、LOD1 重）。
/// DDGI 原版是 `variance/(variance + delta²)`，variance 取自深度方差、**没有距离相关下限**，
/// 深度稳定（方差≈0）时就是硬判定。这里改回同一口径：soft 由 std 主导，下限只防 0。
const DDGI_DEPTH_SOFT_MIN: f32 = 0.5;
/// 空间混合（探针间去噪）里"自身"的权重参照。邻居权重和按探针实际间距加权
/// （见 `ddgi_probe_blend_irr`），典型约 3（6 邻居 × ~0.5）→ 2.0 表示自身约占 40%。
const DDGI_BLEND_SELF: f32 = 2.0;
// （原 DDGI_PROBE_MIN_CLEARANCE 已删）探针放置现在**只有 Douglas #23 的字幕规则**，没有独立的
// "净距门槛"。
//
// 【为什么移除净距硬门槛】我们曾要求"候选空叶半宽 / 探针到最近固体的距离 ≥ 4 体素"。它偏离了
// Douglas 的规则，也正是**墙体接缝漏光**的直接来源：一个 cell 里同时存在大片空腔与贴几何的小
// 空位时，硬门槛会把探针挤到贴墙的小空位（或干脆不放探针）→ 探针贴着墙面 → 深度图朝外方向记
// 的是墙 → 可见性判定失真 → 漏光。
//
// Douglas 的原话（#23 字幕）："it traverses downward until it finds a totally empty leaf in the
// [oc]tree and it does this in a breadth first manner so that one of the **largest** subnodes is
// selected and then it just places the probe at the center of that subnode… for a cell that was
// totally full of voxels, no probe would be generated." —— **只按"最大的全空子块"选点**，同级
// 并列时才靠 cell 中心，并没有"净距"这条额外门槛。现按此还原，见 `ddgi_place_probe`。
//
// 【为什么不会出现空洞】层级一直下探到 1³ 体素，所以**格内只要还有空体素，就一定放得出探针**
// （1³ 是保底档，不再有任何门槛把它挡掉）。既然格内必有探针，就不再需要把无针 cell 指向邻居。
// （原 DDGI_SHARE_K 已删）它把插值基的支撑半径放大到 K 个 cell。代价是越过 cell 边界时
// "离开 stencil 的角"权重不归零却被丢弃 → 权重场在每个 cell 边界跳变（放射状三角锯齿）。
// 现在插值基改用晶格名义三线性基，每个探针天然拿到它所属角的权重，不需要放大支撑。
/// 邻居纹素的**覆盖度**下限（irr 图集第 4 通道 = 该纹素被写过多少次的时间累积，见 collect）。
/// 为什么需要它：空间混合（`ddgi_probe_blend_irr`）必须能拿**暗邻居**去稀释亮邻居，
/// 否则一颗跨墙拿到天光的探针没人稀释，就会在自己的投影处形成一个亮斑（室内墙角漏光）。
/// 而按**值**过滤是做不到的 —— `DDGI_CAST_FLOOR = 0` 之后，"真黑"与"没数据"的值都是 0。
/// 所以改按"这个纹素到底被写过没有"过滤，这个信号由第 4 通道承载。
const DDGI_TEXEL_MIN_COVERAGE: f32 = 0.5;
/// 前后判定（wn）的锐度：wn = clamp(dot(n,-dir)/此值, 0, 1)。
/// = 1.0 → 真正的余弦（60° 处 0.5、78° 处 0.2），与 Douglas/原版 DDGI 的「探针在体素
/// 前 vs 后」一致；= 0.2 时 78° 以内全部钳到满权重，会把**掠射探针**（墙角/凸边外侧那些
/// 看到开阔空间、天花板、外部天空的探针）按满权重混进来 —— 这正是「角落/接缝处柔和高亮」
/// 的来源（这类探针与着色点之间不穿几何，depth 闸门合理地放行，只能靠权重压）。
/// 注意：此值不改变拒绝集合（wn=0 恒为 dot<=0），只改相对权重 → 调它不会产生新黑区。
const DDGI_NORMAL_BIAS: f32 = 1.0;
/// 采样点沿法线的外推量 = 该 LOD 的 cell 边长 × 此系数（参考实现里的 normal_bias 语义）。
/// 必须随间距缩放：固定的 0.7 体素（1.4cm）相对 LOD0 的 64cm cell 等于贴在表面上，
/// 会让前后判定落在临界值上、整面被判「探针在背面」。
const DDGI_BIAS_CELLS: f32 = 0.1;
// ============================================================================
// 逐角几何遮挡（真·线段求交）参数 —— 见 ddgi_sample_lod 调用点。
//
// 【为什么需要它】现有的两个可见性量都是**代理量**：
//   · wn 只看法线点积（纯方向，完全不知道探针与采样点之间隔着什么）；
//   · wd 是 Chebyshev，用**探针自己那张深度图**近似——它检验的是"探针沿该方向的最近
//     表面"，而不是"探针→采样点这条线段本身"（薄墙/掠射方向在深度图里分辨率不够）。
//   于是隔着墙/楼板/薄板的探针：wn>0 且 wd≈1，两道闸门都放行 → 高权重漏光（墙体接缝、
//   天花板与墙的交界处亮线）。这里补一层**线段求交**，把这类探针的权重直接置 0。
//
// 【为什么不设距离上限、也不设固定步数预算】遮挡线段就是「采样点所在 cell 的 8 个角 →
//   采样点」，长度 ≤ 一个 cell 对角线 = cs·√3（cs=16/32/64/128 → 28/55/111/222 体素）。
//   旧实现走 4 体素粗层线性步进 + 固定 48 体素可达(=12 步) + 超限**保守判可见**：LOD1/2/3
//   的对角线 55/111/222 全部 > 48 ⇒ 线段一开始就被判可见，粗级 LOD 的逐角遮挡**等于没做**
//   （这就是远处/粗级漏光更明显的原因）。现在遮挡改走 `world_raycast` 的完整层次 DDA
//   （与主可见性 trace_scene 同一套 trace_grid→trace_chunk 遍历、同一 palette!=0 实体判定）：
//   线段多长就精确走到多长，故不需要距离上限；层次 DDA 从当前格逐级下钻，cell 对角线线段只需
//   少量迭代，故也不需要固定步数上限。真挂死由 trace_chunk 自带的 65536 步安全网兜底
//   （超限返回 miss = 保守判可见，正常线段远不会触发）。
//   【薄墙一致性】同一套遍历/同一实体语义 ⇒ 遮挡对贴接收面的薄墙的判定与主 trace **逐字一致**，
//   不会像粗层格那样整块跳过而漏掉薄墙。
//
/// 线段终点偏置（体素）：从 `max_t = 线段长` 里扣除，停在采样点前。采样点已沿法线外推过
/// （dda_main 的 n_off ≥ 0.5 再叠 ddgi_sample_lod 的 DDGI_BIAS_CELLS·cs ≥ 1.6 体素@LOD0），
/// 留 1 体素既不会擦到接收面自身，也不会吃掉贴在接收面上的那面薄墙（穿墙点比终点早 ≥1.6 体素）。
const DDGI_OCCL_END_BIAS: f32 = 1.0;
/// 逐角遮挡的**权重阈值 ε**（**丢弃**语义）：角最终权重 `w = wtri·wn_w·wd < ε` ⇒ 把该角
/// 当作 0 **直接丢弃**（跳过遮挡求交、不计入 total/wsum）。误差被 ε 死死界定（最坏只是少
/// 贡献一点弱光），**结构上不可能漏光**。这与旧门槛相反（旧写法"权重低 ⇒ 不测，但权重照旧
/// 保留"✗：跳过的角仍以原权重参与平均 → 等于绕过几何遮挡 → 接缝/薄墙弱权重角照样漏光）。
/// A/B 档：0（= 无条件检测全部 8 角，正确但 +1.2ms）/ 0.005（默认）/ 0.01 / 0.02。
const DDGI_OCCL_W_MIN: f32 = 0.005;
const DDGI_T_MAX: f32 = 8192.0;
/// 级联混合带宽度（以**当前级** cell 边长为单位）：像素距某个 LOD 盒边界的距离小于
/// `cs × 此值` 时，在该级与相邻级之间过渡，消除「跨过盒边界突然换一套光照」的硬边
/// （用户报告的「LOD0/LOD1 边界很明显」）。
/// **取 1.0 = 论文的 "one grid cell"**：DDGI 2021（Scaling Probe-Based DDGI）处理相机锁定
/// volume 的级联过渡时明确写「tightening the transition region by **one grid cell** (along
/// each axis)」—— 过渡带就是一个格宽，**不是越宽越好**（太宽会让细级精度被粗级提前稀释，
/// 反而把近场做糊）。
/// 边界两侧恰好都是 0.5/0.5，所以跨边界连续 —— 推导见 ddgi_cascade_blend。
const DDGI_CASCADE_BLEND: f32 = 1.0;
/// 「覆盖内 GI ↔ 覆盖外天光兜底」过渡带的宽度（以**最粗级** cell 为单位）。
/// 着色侧用它把 `amb` 从"盒内由 conf 决定"**平滑过渡**到"盒外全量天光"。
/// 旧版用布尔 `in_casc` 直接 select：盒边界上 `amb` 从 `(1-conf)·sky` **硬跳**到 `sky`
/// → "有 GI / 无 GI"在级联覆盖边界处形成硬边（相机拉远必然能找到割裂机位）。
/// 2.0 = 最粗级两格（LOD3 的 cs=128 体素 → 5.12m）。
const DDGI_CASCADE_FADE_CELLS: f32 = 2.0;
/// 每帧射线总预算，由 `ddgi_seal` 均分给全部活跃探针（rpp = 预算/活跃数，钳 [1,256]）。
/// 必须与 Rust 侧 `DDGI_RAY_BUDGET` 一致。曾试过 262144 / 1048576 压"探针晶格亮斑"与
/// 深度闸门抖动：帧时涨了但问题没解决（根因是探针网格对贴缝尺度欠采样 + 深度角度均值偏差，
/// 不是射线数量），已退回 131072。
const DDGI_RAY_BUDGET: u32 = 65536u;
const DDGI_PROBE_BUDGET: u32 = 4096u;
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
const DDGI_SKY_RADIANCE_SCALE: f32 = 1.0;
const DDGI_RAY_BIAS: f32 = 0.5;
/// 天光环境项：在该像素**没有任何级联探针覆盖**、或**采样置信度过低（无数据/仅剩边缘探针）**
/// 时兜底。数据可信的级联内像素一律由 DDGI 提供环境光（带遮挡）：无光室内 → 探针辐照度≈0
/// → 接近全黑。详见着色处 conf / amb / gi_eff 的推导（那里解释了为什么不能"失败就归零"）。
const DDGI_SKY_AMBIENT: f32 = 0.05;
/// 采样置信度区间：`conf = smoothstep(LO, HI, cov)`，**只用来在"基本没有可用数据"时
/// 补上天光常量**（绝不用来缩放 GI 幅度，理由见着色处注释）。
/// cov = Σ wtri over「有可用数据的角」（**不含** wn/wd 两道照明闸门）→ ∈ [0,1]。
/// HI=0.02 只覆盖"8 个角几乎都没有数据"这一档；只要有一个角有数据（cov ≥ ~0.25 量级）
/// 就完全不受影响 —— 这一点**不能**换成 wsum：凹角/远处粗级的存活权重天然低，用 wsum 会
/// 把"本该黑"判成"不可信"而把天光常量加回来（室内墙角漏光）。见着色处注释。
const DDGI_CONF_LO: f32 = 0.0;
/// `cov` 的上界：conf = smoothstep(LO, HI, cov)。
///
/// 【为什么必须给足带宽】旧值 0.02 ≈ 「全有或全无」：8 个角里只要还有一个有数据就
/// conf=1（不补天光），等最后一个角失效才 conf=0（补满）—— 于是**覆盖边界上 amb 从 0
/// 一步跳到 amb_far**，那就是肉眼可见的"有 GI / 无 GI 割裂"。边界本身不可能消除
/// （覆盖必然有限），能做的只是让「有 GI ↔ 无 GI」的差**被摊平在过渡带里**。
/// 取 0.5：`cov` 的满值约 1（8 个角的 wtri 之和），所以过渡发生在「最后 1~2 个角失效」
/// 对应的空间范围内（约 1~2 个 cell 宽）。嫌边界还明显就继续往 1.0 提。
/// **不能**换成 wsum：凹角/远处粗级的存活权重天然低，用 wsum 会把"本该黑"判成"不可信"
/// 而把天光常量加回来（室内墙角漏光）。见着色处注释。
const DDGI_CONF_HI: f32 = 0.5;
/// 探针射线命中时的辐亮度下限。**必须为 0**：任何非零下限都会被反馈回路放大成
/// ≈ alb·sky·FLOOR/(1-alb) 的"室内自发光"，封闭空间永远压不黑。
/// 首帧的种子由**能看到天空的射线**提供（miss → sky_rgb），所以不需要这个下限。
const DDGI_CAST_FLOOR: f32 = 0.0;

// 收敛跳帧参数（Douglas can_skip_update）：age 达阈值 + 距相机 > SKIP_DIST×spacing 才跳；
// 每 LOD 按周期强制刷新（帧号 + slot 错峰），兜住「offset 没变但光照变了」的编辑。
const DDGI_SKIP_AGE: vec4<u32> = vec4<u32>(32u, 24u, 16u, 12u);
/// 换主（相机滚动）时把 age 打散到 `[1, N]` 的相位抖动周期：让"当帧直写"这个动作按**槽位号**
/// 错开到 N 帧，避免整圈 cell 在同一帧一起直写、从而在 LOD 分界线留下一条完整的浅灰带。
/// 原理：槽位号连续 ⇒ 世界 cell 相邻 ⇒ 取模后相位分散 ⇒ 每一帧只有 1/N 的换主探针在直写，
/// 且在空间上是散点、不成带状。见 `ddgi_bake_one` 的换主说明。
/// N 越大越分散，代价是单个探针"旧的近似值"最多停留 N 帧；8 是折中。
/// 必须让 `1 + N - 1 < DDGI_SKIP_AGE`（最小 12），否则换主的探针会被判 converged 而停止投线。
const DDGI_BAKE_STAGGER: u32 = 8u;
const DDGI_REFRESH_PERIOD: vec4<u32> = vec4<u32>(64u, 96u, 128u, 160u);
/// can_skip_update 的「远离相机」半径（单位 = 本级 cell 边长）。LOD0 的 cell 是 32 体素
/// （64cm），所以 24 → 15.36m：这个半径外的探针一旦收敛（age ≥ SKIP_AGE）就只在
/// REFRESH_PERIOD 的强制刷新帧投线，近场刷新频率完全不变。
const DDGI_SKIP_DIST_CELLS: f32 = 24.0;
/// 采样门限：探针**新生**（首次烘焙）时 age 从 0 起步、图集还没写 → 跳过 age=0 那一帧。
/// **值必须 = 1**：判据是 `age < MIN`，MIN=1 → 只跳 age=0。
/// 【注意「滚动换主」现在不走这条】换主时 bake 把 age 压到 1 并**保留图集历史**（见
/// ddgi_bake_one 的换主说明），所以相机滚动不会再让任何探针被跳过；需要跳的只剩"从未烘过"
/// 的 age=0。
/// 历史：旧值 2 配合"换主 age 归零 + collect 显式清零"曾使相机移动时整圈 cell 整片无数据
/// （那些角不计入 cov → conf=0 → 只剩很暗的天光兜底）→ 表现为"移动时 LOD 交汇处冒黑斑、
/// 几帧后收敛"。两头都改掉后这条链才断。
/// **这条门限不针对"世界编辑"**：编辑同一 world cell 时 bake 不重置 age（见 ddgi_bake_one），
/// 被波及的探针 age 仍然很大、不会被这里跳过 —— 否则那条带会突然退到粗级再切回来，正是编辑后
/// 光影闪烁的一半来源。
const DDGI_MIN_SAMPLE_AGE: u32 = 1u;
/// collect 每探针的纹素线程数 = **从纹素数派生**（16 irr(4×4) + 64 depth(8×8) = 80）。
///
/// 【必须是派生值，不能写死】历史上它曾写死某个值而在纹素数变化后不同步：`ddgi_seal` 拿
/// 它算 collect 的 dispatch 规模，多出来的线程会落进 depth 分支并越出**本探针自己的**纹素块，
/// 把邻居探针的深度纹素改写（或越界作废）—— 邻居的 (mean,std) 每帧乱跳、`snap` 路径还会把
/// 它们写 0（采样侧判"无数据"）。症状是"静态闪烁/亮区伸缩"，且伪装成深度噪声、极难定位。
const DDGI_COLLECT_THREADS: u32 = DDGI_IRR_TEXELS * DDGI_IRR_TEXELS
  + DDGI_DEPTH_TEXELS * DDGI_DEPTH_TEXELS;

// 射线样本区不再分段：seal 用的是**全局统一 rpp**，所以样本下标 = 全局射线编号
// （cast 的 tid 本身就是它），容量 = DDGI_RAY_BUDGET = ddgi_samples 的槽数。
