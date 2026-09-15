import package::bindings::{BEAM_BACKOFF, BEAM_DIV, DDGI_INDIR_CAST_BASE, DDGI_INDIR_COLLBASE_BASE, DDGI_INDIR_COLL_BASE, DDGI_INDIR_COLL_TOTAL, DDGI_INDIR_COUNT_BASE, DDGI_INDIR_RAYBASE_BASE, DDGI_INDIR_RAY_TOTAL, DDGI_INDIR_RPP_BASE, beam_depth, ddgi_args, ddgi_cell, ddgi_indirect, ddgi_meta, ddgi_samples, ddgi_slot_pos, ddgi_u, ddgi_worklist, eye_adapt, eye_adapt_ro, light_u, out_tex, out_tex_src, view_u};
import package::common::{linear_to_srgb, palette_albedo, palette_emissive, sky_rgb, srgb_to_linear};
import package::ddgi::bake::{ddgi_bake_one};
import package::ddgi::collect::{ddgi_collect_one};
import package::ddgi::consts::{DDGI_AGE_MAX, DDGI_CAST_FLOOR, DDGI_COLLECT_THREADS, DDGI_CONF_HI, DDGI_CONF_LO, DDGI_DEPTH_TEXELS, DDGI_EMIT_GAIN, DDGI_LOD_COUNT, DDGI_META_ACTIVE, DDGI_PI, DDGI_RAY_BIAS, DDGI_RAY_BUDGET, DDGI_REFRESH_PERIOD, DDGI_SKIP_AGE, DDGI_SKY_AMBIENT, DDGI_T_MAX, DDGI_USAGE_SCAN_MASK, DDGI_WL_IDX_MASK, DDGI_WL_LOD_SHIFT, SHADOW_SURFACE_EPS};
import package::ddgi::helpers::{DDGI_REC_ENABLED, DDGI_REC_OCCUPIED, ddgi_lod_slot_base, ddgi_meta_age, ddgi_meta_pack, ddgi_meta_with_nbr, ddgi_oct_decode, ddgi_rand2, ddgi_ray_rand, ddgi_rec_off, ddgi_slot_lod_of, ddgi_slot_world_cell};
import package::ddgi::sample::{ddgi_dbg_cov, ddgi_dbg_dom, ddgi_dbg_edge, ddgi_dbg_fb, ddgi_dbg_rej_out, ddgi_dbg_wsum, ddgi_dbg_zero, ddgi_sample, ddgi_usage_track};
import package::ddgi::sort::{ddgi_box_hits_object, ddgi_nbr_vis_mask, ddgi_neighbor_occupied};
import package::lightfield::{light_field_ao, light_field_gather};
import package::trace::{make_grid};
import package::world::{voxel_normal, world_cfg_full, world_raycast};

// ============================================================================
// DDA Compute Shader：全屏逐像素层次栈式 mask DDA（Douglas devlog #17 / 64-tree）
//
// 遍历结构（Rust 参考 cpu_reference_dda_ray_tree，dda.rs 逐字镜像）：
//   chunk 间：256³ 一格 A&W，窗口 entry=0 的空 chunk 只 1 load 直接跨越；
//   chunk 内：4 层栈帧（256→64→16→4→1），节点 mask 一次 load 进寄存器，
//             子块步进只查 bit（零 load）；bit=1 分裂才压栈下钻，
//             bit=0 uniform 子块整格跳过/整格命中。
// sample_brickmap（树点查询）仅保留给 implicit normals 6 邻域 occupancy。
//
// BG0（与 Rust DdaViewUniform 144B + out tex rgba8unorm 对应）：
//   @group(0) @binding(0) = out storage tex（DDA 写入 rgba8unorm，linear RGB）
//   @group(0) @binding(1) = uniform DdaViewUniform（144B 对齐）
//
// BG1（与 Rust GpuBrickMap 资源 1:1 对应）：
//   @group(1) @binding(0) = b_struct: array<u32>（TileIndex + TileBitmaps + CellDirs + NodeStream 定长前缀 + 可变 node 区）
//   @group(1) @binding(1) = b_leaves: array<u32>（P4 重定向：方向可达掩码 LUT，
//                          8 octant × 64 入口格 × 2 u32 = 1024 words = 4KB，
//                          CPU 端 wire::march_mask_lut_words 同布局）
//   @group(1) @binding(2) = b_palette: array<u32>（palette 256 entries × 2 words = 512 words）
//   @group(1) @binding(3) = uniform BrickMapGlobals（scalar 字段 20×u32/i32 = 80B）
//
// BG2（Phase 3 OBJ→Volume 统一：GridDesc 数组，与 Rust GpuBrickMap.grid_descs_buf 1:1）：
//   @group(2) @binding(0) = grid_descs: array<GridDesc>（144B/entry，主世界 + 物体统一描述符）
//   shader `dda_main` 用 `arrayLength(&grid_descs)` 取 volume 数，遍历 trace_grid 无 kind 分支。
//
// BG3（R3-18 直光层：光照光池，与 Rust LightPoolUniform 448B 1:1）：
//   @group(3) @binding(0) = uniform LightPool（LightGlobals 48B + 8×LightDesc 384B + sky_color 16B）
//   数据源 = assets/lighting/*.ron 主题（main world LightingTheme → ExtractResource → build_light_pool）
//
// 顶部常量与 Rust `brickmap::dda::wgsl_consts` 完全一致（单测 TR-2.1 assert_eq 防漂移）。
// BrickMap 五步寻址链严格对应 Rust `gate-render/src/brickmap/view.rs::get_voxel`（逐段注释 L 号）。
// Slot 打包规则对应 Rust `gate-render/src/brickmap/wire.rs::encode_slot/unpack_slot_word/pack_palette_entry`。
// trace_grid 统一 DDA（主网格 + 逐物体 GridDesc），OBJ 等价性单测锁死 CPU 侧。
// ============================================================================

// ============================================================================
// P3 Beam 预 pass：低分辨率（全分辨率 / BEAM_DIV）**保守** trace，只输出最近命中 t。
//
// 保守性（#18 原文："低分辨率射线 never step far enough that voxels could get
// smaller than the minimum size required to always be hit by one low-res ray"）：
// 设 D_safe = 0.35/(BEAM_DIV·px_ang)。beam 射线只行进到 min(t_cap, D_safe)：
//   · 体素 v 在距离 d ≤ D_safe：v 投影双向宽 ≥ 2·BEAM_DIV px（含 45° 最坏菱形
//     + 旋转裕量）→ 全局 beam 网格必有一点落进投影，且距主像素 ≤ 6 px（3×3 邻域
//     半宽，见 dda_main）→ 邻域某条 beam 命中 v → min t ≤ d。✓
//   · 体素 v 在距离 d > D_safe：所有邻域 beam 返回 ≤ D_safe（在 D_safe 停下或命中
//     更前物体）→ 邻域 min ≤ D_safe < d → 主射线从 D_safe 起步继续行进命中 v。✓
// 输出与无 beam 逐像素一致。depth_cap=3（不限制八叉树深度，只限制行进距离）。
// miss 存 t_cap_beam（非 1e30、非 t_cap）：邻域 3×3 全 miss 时主射线起点 ≤ D_safe。
// ============================================================================
@compute @workgroup_size(8, 8, 1)
fn beam_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let beam_size = textureDimensions(beam_depth);
  if (gid.x >= beam_size.x || gid.y >= beam_size.y) { return; }
  // 低分辨率像素中心 → 全分辨率 NDC（用 beam_size 归一化，保证覆盖全视锥）
  let px = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5))
         / vec2<f32>(f32(beam_size.x), f32(beam_size.y));
  let uv = vec2<f32>(px.x * 2.0 - 1.0, 1.0 - px.y * 2.0);
  let near_ndc = vec4<f32>(uv.x, uv.y, 0.0, 1.0);
  let far_ndc  = vec4<f32>(uv.x, uv.y, 1.0, 1.0);
  let near_world_h = view_u.inv_view_proj * near_ndc;
  let far_world_h  = view_u.inv_view_proj * far_ndc;
  let near_world = near_world_h.xyz / near_world_h.w;
  let far_world  = far_world_h.xyz  / far_world_h.w;
  let dir = normalize(far_world - near_world);
  let origin = view_u.cam_pos_voxel.xyz;
  let t_cap = length(far_world - near_world);
  // D_safe = 0.35/(BEAM_DIV·px_ang)：体素在 d ≤ D 处屏幕投影双向宽 ≥ 2·BEAM_DIV px
  // （0.35 ≈ 1/(2√2)，覆盖 45° 最坏菱形 + 旋转裕量）→ 全局 beam 网格必有一点落进
  // 投影，且距主像素 ≤ 3×3 邻域半宽 6 px（见 dda_main）→ 邻域 min t ≤ 体素 t +
  // ~2.3（同体素侧入差，dda_main 已回退 BEAM_BACKOFF 补偿）。
  // 旧公式 BEAM_DIV/px_ang 超出保守距离 8 倍：远处亚间距体素 beam 漏记 + 主射线
  // 从错误 t_min 起步 → 穿墙（用户实测）。#18 原文即"低分辨率射线永不行进到
  // 体素小于最小必命中尺寸的距离"。
  let d_beam = 0.35 / (f32(BEAM_DIV) * max(view_u.lod.x, 1e-6));
  let t_cap_beam = min(t_cap, d_beam);

  // depth_cap=3：八叉树全深度（不限制深度，只限制行进距离 t_cap_beam）
  let h = world_raycast(make_grid(0u), origin, dir, t_cap_beam, world_cfg_full(0.0, 3u));
  // miss 存 t_cap_beam（非 t_cap）：3×3 邻域全 miss 时主射线起点 ≤ d_beam，
  // d_beam 之外的几何由主射线完整 trace。旧 bug 存 t_cap（frustum 全长）时，
  // 全 miss 邻域把主射线起点推到 frustum 末端 → 整条射线假 miss（场景丢失）。
  let t = select(t_cap_beam, h.t, h.hit);
  textureStore(beam_depth, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(t, 0.0, 0.0, 0.0));
}

// ============================================================================
// DDA 主入口：R3-18 直光层 — 命中 → 直光着色（硬阴影 + sky 环境 + emissive），miss → 纯色天空
// ============================================================================
@compute @workgroup_size(8, 8, 1)
fn dda_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let size = textureDimensions(out_tex);
  if (gid.x >= size.x || gid.y >= size.y) { return; }
  let coord0 = vec2<i32>(i32(gid.x), i32(gid.y));

  // ---- 「屏幕使用」记账开关（工作集选择的唯一依据，见 DDGI_USAGE_STICKY）----
  // 采样像素 = "本帧轮到的那一列" ∪ "本帧轮到的那一行"（两条互相垂直的带）⇒ **每个像素每
  // MASK+1 帧至少被采到一次**（列、行各轮到一次），全局写入量摊到每帧约 2/(MASK+1) 的像素
  // （8 → 23%；而 `ddgi_usage_mark` 是满值写入、幂等，绝大多数只是"读一下、不写"）。
  // 【为什么不能取 AND（旧式）】那样选中的是 `x ≡ y ≡ -frame (mod 8)` 的**对角线像元**：
  // 它只占全屏的 1/8，其余 7/8 的像素**永远**进不了采样集。于是"屏幕 footprint 只有几个像素、
  // 且恰好不含对角像元"的探针（薄几何 / 细缝 / 小特征 = "细微处"）**一次也不会被记账** →
  // 永远进不了工作集 → 年龄冻在离开时的值（12~32）→ `ddgi_age_alpha` 据此给它 1/12~1/32 的
  // 大增益 → 每次零星刷新（`due` 错峰）就把图集值**跳 3%~8%** ⇒ 那些像素逐帧跳变。
  // 见 `ddgi_usage_mark` 的完整推导。cast 里的 GI 回读不设这个开关（恒 0）→ 不记账。
  let usage_frame = u32(ddgi_u.params.x);
  ddgi_usage_track = select(
    0u,
    1u,
    ((gid.x + usage_frame) & DDGI_USAGE_SCAN_MASK) == 0u
      || ((gid.y + usage_frame) & DDGI_USAGE_SCAN_MASK) == 0u,
  );

  // ---- 反投影：像素中心 (gid + 0.5) → NDC (u, v) ∈ [-1, 1] ----
  let px = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5)) / vec2<f32>(f32(size.x), f32(size.y));
  let uv = vec2<f32>(px.x * 2.0 - 1.0, 1.0 - px.y * 2.0);
  let near_ndc = vec4<f32>(uv.x, uv.y, 0.0, 1.0);
  let far_ndc  = vec4<f32>(uv.x, uv.y, 1.0, 1.0);
  let near_world_h = view_u.inv_view_proj * near_ndc;
  let far_world_h  = view_u.inv_view_proj * far_ndc;
  let near_world = near_world_h.xyz / near_world_h.w;
  let far_world  = far_world_h.xyz  / far_world_h.w;
  let diff_world = far_world - near_world;
  let frustum_length = length(diff_world);
  let dir_voxel = normalize(diff_world);
  let origin_voxel = view_u.cam_pos_voxel.xyz;

  // ---- 场景 trace（P3 beam 起点跳过空空间）+ unlit 逐体素法线着色 ----
  // P3 beam：取当前像素 3×3 beam 邻域的最小命中 t 作为起点，跳过空空间。
  // beam 像素 = floor(全分辨率像素 / BEAM_DIV)；邻域 clamp 到 beam 边界。
  // lod.z > 0.5（GATE_NO_BEAM=1）：关闭 beam，t_min=0。
  var t_min = 0.0;
  if (view_u.lod.z < 0.5) {
    let beam_size = textureDimensions(beam_depth);
    let bx = gid.x / BEAM_DIV;
    let by = gid.y / BEAM_DIV;
    t_min = 1e+30;
    // 3×3 对称邻域（旧 2×2 右下偏置：体素投影凸六边形中心可偏出窗）。
    // beam 有效距离（d_beam）内体素投影双向宽 ≥ 2·BEAM_DIV px → 最近 beam
    // 网格点距主像素 ≤ ~6 px（最坏 45° 菱形 ≈ 5.7 px）→ ±1 beam 格覆盖。
    for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
      for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
        let sx = clamp(i32(bx) + dx, 0, i32(beam_size.x) - 1);
        let sy = clamp(i32(by) + dy, 0, i32(beam_size.y) - 1);
        let bt = textureLoad(beam_depth, vec2<i32>(sx, sy)).x;
        t_min = min(t_min, bt);
      }
    }
    // P5：保守回退（BEAM_BACKOFF 注释）——邻域 beam 命中主射线首命中体素时可
    // 从侧面/远面入射，命中 t 晚 ~2.3 单位；直接以 min t 起步会把起点推过真实
    // 命中 → 薄壁/剪影穿墙。d_beam 之外 t_min ≤ d_beam < 首命中 t，本就安全。
    t_min = max(t_min - BEAM_BACKOFF, 0.0);
  }
  let best = world_raycast(make_grid(0u), origin_voxel, dir_voxel, frustum_length, world_cfg_full(t_min, 3u));
  var col = sky_rgb();
  if (best.hit) {
    // ---- 逐体素着色（Douglas #22/#23：一体素一色）+ 逐体素法线 ----
    // albedo/采样点/阴影射线体素锚定（同体素同色）；法线取**隐式逐体素法线**（退化退面法线，
    // 见 voxel_normal 注释）—— 它同时决定太阳 N·L、阴影射线/GI 采样点的外推方向、DDGI 方向采样。
    let alb = palette_albedo(best.palette_base, best.pal);
    // 体素中心 → 世界系（主世界 identity 直等于体素中心；物体经旋转/缩放变换）
    let gg = make_grid(u32(best.obj_id) + 1u);
    let vc = vec3<f32>(best.voxel) + vec3<f32>(0.5);
    let p_voxel = gg.pos + vec3<f32>(dot(vc, gg.col0), dot(vc, gg.col1), dot(vc, gg.col2)) * gg.scale;
    let n = voxel_normal(gg, best.voxel, best.n);
    // 沿 n 的外推距离：**必须保证离开命中体素**。轴向法线时 0.5+eps 就够，但逐体素法线在
    // 棱边/圆角处是斜的，0.5·n 的分量 < 0.5 → 起点仍在体素内 → DDA 先命中所属体素的邻居
    // （厚墙的棱边被自己遮住 → 每条棱一圈暗边）。0.5/max|n| 正是「从中心沿 n 走多远离开
    // 单位立方体」，任何方向都保证落在空气侧。
    let n_off = 0.5 / max(max(abs(n.x), abs(n.y)), max(abs(n.z), 1e-3)) + SHADOW_SURFACE_EPS;
    // R3-18 直光硬阴影（#02/#17 形态）：1 条太阳射线，不通即阴影。
    // 射线原点 = 体素中心 + n×(0.5 + SHADOW_SURFACE_EPS)（贴向空气侧邻域）。
    // 那个 eps 是必需的：+n×0.5 恰好落在面平面上，对 -X/-Y/-Z 面坐标是整数 → DDA 的
    // floor 落回**体素自己** → t=0 自命中；而下面的自命中防护把「命中自己」当无遮挡 →
    // 这三向的面无论墙多厚都吃到太阳（封闭无光房间的明暗会完全跟着面朝向走）。
    // +X/+Y/+Z 面落在高侧边界，floor 到邻域空气格，所以旧代码只在这半边出错。
    let sun_dir = light_u.lights[0].kind_pos_dir.yzw;
    let ndl = max(dot(n, sun_dir), 0.0);
    var sun = 0.0;
    if (ndl > 0.0) {
      let sh = world_raycast(make_grid(0u), p_voxel + n * n_off, sun_dir, 8192.0, world_cfg_full(0.0, 3u));
      sun = select(
        1.0,
        0.0,
        sh.hit
          && !(all(sh.voxel == best.voxel) && sh.obj_id == best.obj_id),
      );
    }
    let sky = light_u.sky_color.xyz;
    let sun_c = light_u.lights[0].color_intensity.xyz * light_u.lights[0].color_intensity.w;
    let gi_on = ddgi_u.misc.x > 0.5;
    var gi = vec3<f32>(0.0);
    if (gi_on) {
      // 只把点推到体素表面外侧（沿 n 走 n_off，保证离开命中体素，见其定义处注释）；
      // 更远的、随 cell 尺寸缩放的外推在 ddgi_sample_lod 内部按该 LOD 的 cs 施加
      // （DDGI_BIAS_CELLS）。
      gi = ddgi_sample(p_voxel + n * n_off, n) * ddgi_u.params.z / DDGI_PI;
    }
    // AO（Douglas #15）只作用于**常量天光**这一项。
    //
    // 【为什么不能乘在 gi 上】gi 是**已经带遮挡信息的全局光照** —— 探针射线本身就逐方向做了
    // 遮挡判定（wd/Chebyshev），再乘一个 16³ 粗粒度填充率的 AO 属于**重复计算**。它的症状
    // 很好认：AO 只乘间接项 → 关掉 DDGI 时 gi=0、只剩 0.05 常量天光，AO 作用在一个很小的
    // 量上（"不开 DDGI 就没斑"）；开着时 gi 是 0.2~0.4，同一个 AO 把它乘暗 → 圆柱/薄肋上
    // 出现**块状暗斑**，而且光照场是相机中心、cell 0.32m、随相机滚动重铺 → **微小视角变化
    // 就整片跳变**。这正是"光影斑点 + 暗部骤变"的主因。
    //
    // 常量天光没有任何遮挡信息（凹角与开阔面一样亮），AO 正是为它准备的 —— 所以只留这一项。
    let ao = light_field_ao(p_voxel, best.obj_id);
    // 天光常量在两种情形下兜底：① 本像素不在任何级联盒内（ddgi_dbg_dom==0，无探针覆盖）；
    // ② 在级联内但**采样不可信/无数据**（见下）。级联内数据可信时一律由 GI 提供 ——
    // 无光室内的探针辐照度≈0 → 画面接近全黑；若无条件加常量，室内外就没有亮度差。
    //
    // 【为什么不能"采样失败就归零"】ddgi_sample 返回的是**归一化**平均（total/wsum）：
    // wsum 只决定"有几个探针参与了这次平均"，**不决定返回值的幅度**。于是 wsum 在阈值
    // 1e-4 附近时，结果会从"某一个边缘探针的满幅值"直接跳到 0（或跳到粗级 LOD 的值）。
    // 这一跳就是两个被反复报告的观感问题的共同根因：
    //   ① 薄几何/曲面上"8 角探针被剔空"的像素间接光整项归零 → **零星黑斑**
    //      （不开 DDGI 时 amb 是常量，所以看不到）；
    //   ② 相机稍微一动，这些像素就在「有 GI」与「纯黑」之间翻 → **阴影暗部骤变**。
    // 改成按置信度淡出到天光常量：探针数据正常 → conf≈1（室内依旧由 ≈0 的辐照度决定，
    // 仍然黑）；探针不可信 → conf→0（退化成常量天光，与 DDGI 关闭时的观感一致）。全程连续。
    //
    // 【判据必须是 cov（覆盖度），不是 wsum】cov = Σ wtri over「有可用数据的角」
    // （窗口/ENABLED/ACTIVE/age/深度纹素有记录），∈[0,1]；它回答的是"本点到底有没有探针数据"。
    // wsum = Σ wtri·wn·wd 额外乘了两道**照明**闸门，回答的是"有几个探针愿意给本点照明"。
    // 用 wsum 当置信度会把"该点本来就该黑"误判成"数据不可信"：室内凹角处相邻两堵墙互相
    // 遮挡，存活权重天然很低 → wsum 塌到阈值以下 → 把常量天光加回来 → **最该黑的墙角反而
    // 最亮**（室内墙角漏光）；cell 越粗（离相机越远）遮挡剔除越多 → 漏光越明显。见
    // ddgi_sample_lod 里 cov / wsum 的分界注释。
    // 阈值取 [0, 0.02]：**只覆盖"基本没有可用数据"的区间**。
    //
    // 【重要】conf 只用来**补天光常量**，绝不去缩放 GI 的幅度。曾经用它乘过 gi，结果：
    // 覆盖度低的像素被压暗 → 暗斑范围更大更明显。这个教训记在这里，别再犯。
    let conf = smoothstep(DDGI_CONF_LO, DDGI_CONF_HI, ddgi_dbg_cov);
    let amb_sky = sky * DDGI_SKY_AMBIENT;
    // 【为什么拆成 amb_fill × (1-conf)】
    // `t_edge` 是**几何量**（离盒边界多远），`conf` 是**数据量**（本点有没有可用的 GI 数据）。
    // 旧式 `mix(amb_far, amb_sky*(1-conf), t_edge)` 把几何因子直接乘进了"有数据"那一项 ——
    // 于是**盒边界附近即便有 GI 数据也会被 fallback 稀释**。而相机拉远时整个场景恰好挤在盒
    // 边缘（t_edge 普遍很小）→ 全画面都在吃 amb_far → 拖 Far amb 全局变化、分界永远消不掉。
    // 现在分成两步：`t_edge` 只决定**用哪种兜底强度**（盒外 amb_far / 盒内 amb_sky），
    // **补多少完全由 conf 决定** → 有数据(conf=1)时 amb 恒为 0，**不受 Far amb 影响**。
    let t_edge = smoothstep(0.0, 1.0, ddgi_dbg_edge);
    let amb_far = sky * ddgi_u.misc.w;
    let amb = mix(amb_far, amb_sky, t_edge) * (1.0 - conf);
    col = alb * (sun_c * ndl * sun + amb * ao + gi);
    // 自发光体素：radiance 直出（不吃方向、不吃阴影）——「每个体素都能是光源」的着色侧。
    col = col + alb * palette_emissive(best.palette_base, best.pal) * DDGI_EMIT_GAIN;
    if (ddgi_u.params.y > 0.5) {
      if (ddgi_u.params.y < 1.5) {
        col = alb * sky * DDGI_SKY_AMBIENT * 0.15 + alb * gi * 8.0;
      } else if (ddgi_u.params.y < 2.5) {
        // wsum 档 = 采样置信度的**灰度直方图**（1:1，不吃 albedo）。
        // wsum = Σ wtri·wn·wd 是 trilinear 权重和 ∈ [0,1]（开阔表面 ~0.3-0.6）→ 直接当灰度。
        // 旧版写成 `alb * clamp(wsum*0.125,0,1) * 2`：既乘了 albedo（深色材质即使覆盖正常
        // 也显示近黑）又整体偏低 16× —— 结果"哪里到底有没有数据"根本读不出来，
        // 是上一轮把 wsum 当成 GI 幅度缩放系数的误判成因之一。
        col = vec3<f32>(clamp(ddgi_dbg_wsum, 0.0, 1.0));
      } else if (ddgi_u.params.y < 3.5) {
        // d = 「包含该像素的壳」所在 LOD + 1（纯几何包含，与采样是否成功无关）；
        // d = 0 表示没有任何 LOD 盒包含该像素 —— 这是覆盖率问题，不是剔除问题。
        // 4 级必须给出 4 种可分辨颜色（旧阈值把 LOD0 与 LOD1 都涂成红，无法定位）。
        let d = ddgi_dbg_dom;
        col = vec3<f32>(0.1, 0.8, 0.2);                          // 绿：未被任何 LOD 包含
        col = select(col, vec3<f32>(0.9, 0.2, 0.2), d > 0.5);    // 红：LOD0
        col = select(col, vec3<f32>(0.95, 0.85, 0.1), d > 1.5);  // 黄：LOD1
        col = select(col, vec3<f32>(0.2, 0.4, 0.9), d > 2.5);    // 蓝：LOD2
        col = select(col, vec3<f32>(0.9, 0.2, 0.9), d > 3.5);    // 品红：LOD3
      } else {
        // Probe：先看「数据有没有取到」，再看 8 个角是被哪道闸门剔掉的。
        //   青绿 = 所有路径都没给出值，改用**同侧探针场插值**（忽略可见性）兜住 → 见
        //          ddgi_sample 的 r4。这类点就是"犄角旮旯里的品红纯黑点"。
        //   紫   = 壳内 r1/r2 都失败后，**背向闸门放开**的末级兜底 r3' 成功 → GI 有效
        //          （线段求交确认"探针确实看得见本点"，见 ddgi_sample 的 r3' 说明）。
        //          这些像素在 r3' 之前就是"Probe 品红 + 纯黑点"。
        //   亮黄 = 壳内 8 角被 wn 全剔（薄几何），用**同级**放宽 wn 兜住 → GI 有效（见
        //          ddgi_sample 壳内分支注释）；**不退到粗级**，所以不会跨墙漏光
        //   浅绿 = 壳没给出数据，但**更粗一级 LOD 兜住了** → GI 有效，只是精度粗
        //          （相机滚动时新进入窗口的那条带属于这一类，不是异常）；
        //          但粗级探针在远处、方向掠射时深度图判不出遮挡 → 封闭空间里它是漏光来源
        //   绿   = 权重和 > 0 且辐照度非 0 → 本壳直接正常
        //   白   = 权重和 > 0，但辐照度 ≈ 0 → 过闸的探针图集是空的（写入/寻址问题）
        //   青   = 无数据闸门：探针没进本帧 worklist（near=false）、年龄太小，或该方向纹素
        //          从未被写过 —— 都不能以 0 参与平均
        //   橙   = ENABLED 闸门（该 cell 全满 → bake 没放探针）
        //   品红 = 法线背向闸门（wn <= 0）
        //   红   = depth 遮挡闸门（wd <= 0）
        //   灰   = 角越界（clamp 路径不会出现）
        // 旧版在这里对 dom<0.5 直接涂蓝，把这份直方图整个短路掉了。
        if (ddgi_dbg_fb > 4.5) {
          col = vec3<f32>(0.0, 0.6, 0.55);
        } else if (ddgi_dbg_fb > 3.5) {
          col = vec3<f32>(0.65, 0.35, 0.95);
        } else if (ddgi_dbg_fb > 2.5) {
          col = vec3<f32>(0.98, 0.98, 0.35);
        } else if (ddgi_dbg_fb > 1.5) {
          col = vec3<f32>(0.55, 0.95, 0.35);
        } else if (ddgi_dbg_wsum >= 1e-4) {
          col = select(
            vec3<f32>(0.15, 0.85, 0.25),
            vec3<f32>(0.95, 0.95, 0.95),
            ddgi_dbg_zero > 0.5,
          );
        } else {
          let r_np = f32(ddgi_dbg_rej_out.x);
          let r_wn = f32(ddgi_dbg_rej_out.y);
          let r_wd = f32(ddgi_dbg_rej_out.z);
          let r_ia = f32(ddgi_dbg_rej_out.w);
          let rmax = max(r_ia, max(r_wd, max(r_wn, r_np)));
          if (rmax <= 0.0) {
            col = vec3<f32>(0.5, 0.5, 0.5);
          } else if (r_wd >= rmax) {
            col = vec3<f32>(0.9, 0.15, 0.15);
          } else if (r_wn >= rmax) {
            col = vec3<f32>(0.95, 0.15, 0.7);
          } else if (r_ia >= rmax) {
            col = vec3<f32>(0.1, 0.85, 0.95);
          } else {
            col = vec3<f32>(0.95, 0.6, 0.1);
          }
        }
      }
    }
  }
  // 眼睛适应（自动曝光）：曝光在 sRGB 编码**之前**乘 —— 暗场提亮才用得上全部 8bit 动态范围
  // （放到编码之后乘会把暗部的量化台阶一起放大成色带）。见文件末尾 eye_adapt_* 两个入口。
  let exposure = bitcast<f32>(eye_adapt_ro[0u]);
  let exposed = col * select(1.0, exposure, exposure > 0.0);
  textureStore(out_tex, coord0, vec4<f32>(linear_to_srgb(exposed), 1.0));
}


// ============================================================================
// 阶段一 seal：sort 之后独立 pass（pass 边界保证计数器可见）。4 线程各管一个 LOD：
// 读活跃探针数 → 算 rpp（固定预算/活跃数，clamp 8..256）→ 写 cast/collect indirect args。
// ============================================================================
@compute @workgroup_size(4, 1, 1)
fn ddgi_seal(@builtin(local_invocation_id) lid: vec3<u32>) {
  let lod = lid.x;
  // 每线程各自把 4 个 LOD 的活跃计数读一遍，独立算出前缀和 —— 计数都在全局内存里，
  // 不需要 workgroup 同步（thread 0 顺带写出总线程数与唯一一份 indirect args）。
  // rpp 是**全局统一**的：射线预算在全部活跃探针间均分（Douglas #23 原文 "dividing them
  // amongst all of the active probes"）。旧版按 LOD 权重 0.5/0.25/0.15/0.10 分摊预算，
  // 粗级探针拿到的射线数远低于平均（rpp≈3）→ 深度图 64 个纹素每帧只写到一小部分，
  // 遮挡判定(wd)随之失效。
  var total_active = 0u;
  for (var l = 0u; l < DDGI_LOD_COUNT; l = l + 1u) {
    total_active = total_active + atomicLoad(&ddgi_indirect[DDGI_INDIR_COUNT_BASE + l]);
  }
  // 下限 1：rpp ≥ 1 是硬性的（每个活跃探针每帧至少 1 条射线）。因此
  // `total_ray = total_active × rpp ≤ max(total_active, DDGI_RAY_BUDGET)`，其上界是
  // **总槽位数**（活跃数 ≤ 槽数），而不再是预算 —— samples 缓冲按 `max(预算, 总槽位)` 定容，
  // 见 Rust `ddgi_sample_bytes`。旧版这里断言"活跃数 ≤ 槽数 65536 < 预算"，那个前提在
  // dims 改成按世界 AABB 算（39 万槽位）后已失效。
  let rpp = max(1u, min(256u, DDGI_RAY_BUDGET / max(total_active, 1u)));

  var ray_base = 0u;
  var coll_base = 0u;
  var total_ray = 0u;
  var total_coll = 0u;
  for (var l = 0u; l < DDGI_LOD_COUNT; l = l + 1u) {
    let cl = atomicLoad(&ddgi_indirect[DDGI_INDIR_COUNT_BASE + l]);
    let rc = select(0u, cl * rpp, cl > 0u);
    let cc = select(0u, cl * DDGI_COLLECT_THREADS, cl > 0u);
    if (l < lod) { ray_base = ray_base + rc; coll_base = coll_base + cc; }
    total_ray = total_ray + rc;
    total_coll = total_coll + cc;
  }
  atomicStore(&ddgi_indirect[DDGI_INDIR_RAYBASE_BASE + lod], ray_base);
  atomicStore(&ddgi_indirect[DDGI_INDIR_COLLBASE_BASE + lod], coll_base);
  atomicStore(&ddgi_indirect[DDGI_INDIR_RPP_BASE + lod], rpp);
  if (lod == 0u) {
    atomicStore(&ddgi_indirect[DDGI_INDIR_RAY_TOTAL], total_ray);
    atomicStore(&ddgi_indirect[DDGI_INDIR_COLL_TOTAL], total_coll);
    // 每探针 rpp 射线、每探针 320 纹素线程，WG 均为 64 → 两次 indirect dispatch，y=z=1
    let cb = DDGI_INDIR_CAST_BASE;
    atomicStore(&ddgi_args[cb + 0u], min((total_ray + 63u) / 64u, 65535u));
    atomicStore(&ddgi_args[cb + 1u], 1u);
    atomicStore(&ddgi_args[cb + 2u], 1u);
    let kb = DDGI_INDIR_COLL_BASE;
    atomicStore(&ddgi_args[kb + 0u], min((total_coll + 63u) / 64u, 65535u));
    atomicStore(&ddgi_args[kb + 1u], 1u);
    atomicStore(&ddgi_args[kb + 2u], 1u);
  }
}

// ============================================================================
// 阶段一 烘焙（嵌套级联：相机滚动补烘新入区 / 世界编辑时全量重烘）。
// 逐 cell 沿 4³ 分裂树 BFS 找「最大的全空叶」，探针放在其中心；同级里取更靠近
// cell 中心的空叶。全空 cell → 居中；全满 cell → 无探针。
// 输出 ddgi_cell[slot] = flags(bit0 探针存在 / bit1 cell 有体素) | cell 内归一化偏移。
// 每帧 sort 只读该记录，不再逐帧重算树 BFS。
// 增量：ddgi_cell_id[slot] 记录已烘焙的世界 cell 键，键未变的槽位直接跳过（续龄）；
// 相机滚动只影响发生滚动的那一级 LOD，其余级不动。
// ============================================================================

// 4 个 per-LOD 入口：Rust 侧按 lod 0→3（细→粗）各派发一个独立 compute pass，
// pass 边界即内存屏障 ⇒ 粗级能读到本帧细级刚写好的 ddgi_cell / ddgi_cell_id。
@compute @workgroup_size(64)
fn ddgi_bake0(@builtin(global_invocation_id) gid: vec3<u32>) { ddgi_bake_one(0u, gid.x); }
@compute @workgroup_size(64)
fn ddgi_bake1(@builtin(global_invocation_id) gid: vec3<u32>) { ddgi_bake_one(1u, gid.x); }
@compute @workgroup_size(64)
fn ddgi_bake2(@builtin(global_invocation_id) gid: vec3<u32>) { ddgi_bake_one(2u, gid.x); }
@compute @workgroup_size(64)
fn ddgi_bake3(@builtin(global_invocation_id) gid: vec3<u32>) { ddgi_bake_one(3u, gid.x); }

@compute @workgroup_size(64)
fn ddgi_sort(@builtin(global_invocation_id) gid: vec3<u32>) {
  let slot = gid.x;
  if (slot >= u32(ddgi_u.misc.y)) { return; }
  let lod = ddgi_slot_lod_of(slot);
  let idx = slot - ddgi_lod_slot_base(lod);
  let rec = ddgi_cell[slot];
  let cs = ddgi_u.lods[lod].origin.w;
  let wc = ddgi_slot_world_cell(lod, idx);
  let cmin = wc * cs;

  let enabled = (rec & DDGI_REC_ENABLED) != 0u;
  if (!enabled) {
    // 本格没有探针：清 meta 与 slot_pos 即可。
    // （旧的"借针"机制会在这里把本格指向邻近探针；放置规则改成 Douglas 的"最大的全空子块"后
    //   格内必有探针，重定向不再需要，相关间接表已删除。）
    ddgi_meta[slot] = 0u;
    ddgi_slot_pos[slot] = vec4<f32>(0.0, 0.0, 0.0, -1.0);
    return;
  }
  let off = vec3<f32>(ddgi_rec_off(rec)) / 255.0;
  let probe_pos = vec3<f32>(cmin) + off * f32(cs);

  // ---- 「本 cell 附近有没有几何」：活跃判定的输入（只对 enabled 的格计算）----
  var near = (rec & DDGI_REC_OCCUPIED) != 0u;
  if (!near) {
    near = ddgi_neighbor_occupied(lod, wc + vec3<i32>(-1, 0, 0))
      || ddgi_neighbor_occupied(lod, wc + vec3<i32>(1, 0, 0))
      || ddgi_neighbor_occupied(lod, wc + vec3<i32>(0, -1, 0))
      || ddgi_neighbor_occupied(lod, wc + vec3<i32>(0, 1, 0))
      || ddgi_neighbor_occupied(lod, wc + vec3<i32>(0, 0, -1))
      || ddgi_neighbor_occupied(lod, wc + vec3<i32>(0, 0, 1));
  }
  if (!near) {
    // 物体：主世界 brick tree 里没有它的体素，必须用 grid_descs 的世界 AABB 判。
    // 旧版读的是从未被写入的 ddgi_objects（那段是死代码）→ 物体所在 cell 永不判活
    // → 图集恒 0 → 物体整面没有 GI。
    near = ddgi_box_hits_object(
      vec3<f32>(cmin),
      vec3<f32>(cmin) + vec3<f32>(f32(cs)),
    );
  }

  // ---- age 继承（同 slot 且 world cell 未变 → 续龄；换 cell 时 bake 已清零）----
  var age = ddgi_meta_age(ddgi_meta[slot]);
  // ---- 世界编辑（脏区）→ 把年龄压回 1：**长窗口不牺牲编辑响应** ----
  // 稳态用的是 DDGI_ALPHA 的长窗口（≈3000 次更新 ≈ 50s），光靠它跟世界编辑要几十秒才显形。
  // 而编辑带**精确信号**：`dirty_min.w` 只在"本帧真的上传了世界数据"那一帧为 1（Rust 侧每帧
  // 覆盖写 `BrickMapDirty`，无脏区时 min == max ⇒ w = 0），所以这里既不会误触发、也不会常驻
  // 为真（常驻为真会让这些探针每帧走 alpha=1 直写，等于把时域平滑整个关掉）。
  // 压低 age ⇒ 该探针回到 `ddgi_age_alpha` 的 1/age 等权斜坡（约 30 帧重新收敛），并因
  // `!converged` 自动进入本帧 worklist（即使它不在"屏幕用到"的工作集里）—— 这正是
  // DDGI_ALPHA 的注释所依赖的那条快跟随通道（它一度在重构中丢失）。
  // 判据与 bake 的脏区判定**逐字一致**（"cell 的 voxel AABB 与脏区 AABB 相交"），
  // 保证"被重烘的 cell"与"被重置 age 的 cell"是同一批。
  let cell_lo = vec3<f32>(cmin);
  let cell_hi = cell_lo + f32(cs);
  if (ddgi_u.dirty_min.w > 0.5
      && all(cell_hi > ddgi_u.dirty_min.xyz)
      && all(cell_lo < ddgi_u.dirty_max.xyz)) {
    age = 1u;
  }
  // 判活条件严格对齐 Douglas #23：有探针 && (本 cell 或 6 邻接有体素 || 与物体 AABB 相交)。
  // 旧版额外加了「不落在更细一级 LOD 盒内」的排除，用来省重复投线 —— 但它的代价是：
  // 粗级的探针只要落点被细盒包住就永不投线 → 图集恒 0 → 壳边界附近整圈像素采到空数据
  // （Probe 青色 = 未判活）。字幕里没有这条规则，去掉。
  let is_active = near;

  if (is_active) {
    // ---- 本帧要不要投线：**按「屏幕会不会用到它」**，不再按距离猜 ----
    // 三个入选项（见 consts 的 DDGI_USAGE_STICKY）：
    //   ① `used`：上一帧被着色采样过（漏桶计数 > 0）—— 屏幕正在用它的数据；
    //   ② `!converged`：年龄不足（新探针 / 相机滚动换主）—— 让它尽快成型；
    //   ③ `due`：按槽位号错峰的强制刷新 —— 兜住"从没上过屏"的探针，避免它们的图集永远是空
    //      数据（相机转过去时那片区域才不会一开始全黑），也保证世界编辑后远场最终能跟上。
    // 记账在这里**读后衰减**（-1/帧）：本帧没被着色用到 → 下一帧就不再 >0。
    let usage = ddgi_cell_slot[slot];
    ddgi_cell_slot[slot] = select(0u, usage - 1u, usage > 0u);
    let used = usage > 0u;
    let frame = u32(ddgi_u.params.x);
    let converged = age >= DDGI_SKIP_AGE[lod];
    let due = (frame + slot) % DDGI_REFRESH_PERIOD[lod] == 0u;
    if (used || !converged || due) {
      age = min(age + 1u, DDGI_AGE_MAX);
      let wslot = atomicAdd(&ddgi_indirect[DDGI_INDIR_COUNT_BASE + lod], 1u);
      // worklist 是**压缩后**的列表，它的下标（wslot）只是排名，不是 cell 下标。
      // 低 8 位 age、中 19 位「探针在本级网格里的 cell 下标」、bit27 起 lod ——
      // collect 必须用 cell 下标才能把数据写进正确的图集纹素（见 DDGI_WL_IDX_MASK 的说明）。
      let packed = age | (idx << 8u) | (lod << DDGI_WL_LOD_SHIFT);
      ddgi_worklist[ddgi_lod_slot_base(lod) + wslot] =
        vec4<f32>(probe_pos, bitcast<f32>(packed));
    }
  }

  // ---- 邻接可见性掩码（空间混合门控）：只对**活跃**探针算（非活跃的不投线、图集不更新，
  // 也不会作为混来源被邻居采用 → 省 6 条射线/探针）。见 ddgi_nbr_vis_mask 的说明。
  // 非活跃路径不置 NBR_VALID → 消费侧按"未算过"处理（不做门控），不会误关混合。----
  var meta_word = ddgi_meta_pack(age, true, is_active);
  if (is_active) {
    meta_word = ddgi_meta_with_nbr(meta_word, ddgi_nbr_vis_mask(probe_pos, f32(cs)));
  }
  ddgi_meta[slot] = meta_word;
  ddgi_slot_pos[slot] = vec4<f32>(probe_pos, f32(lod));
}

// ============================================================================
// 阶段二 cast（Douglas pass #2）：worklist 里的活跃探针各投 rpp 条射线。
// **射线方向 = 绑定的八面体深度纹素方向 + 纹素内抖动**（纹素按 frame 轮换覆盖，见函数内注释）。
// 命中体素 → 取该处上一帧 GI 乘 albedo 得出射辐亮度；命中天空 → 天空色。
// 输出 ddgi_samples（方向 + 命中距离）/（辐亮度）。
// 线程映射：全局 tid = ray_base[lod] + probe_idx × rpp + ray（ray_base 由 ddgi_seal 算）。
// 一次 dispatch 覆盖全部 LOD —— 不再依赖 indirect 的 y 维传 LOD。
// ============================================================================
@compute @workgroup_size(64)
fn ddgi_cast(@builtin(global_invocation_id) gid: vec3<u32>) {
  let tid = gid.x;
  if (tid >= atomicLoad(&ddgi_indirect[DDGI_INDIR_RAY_TOTAL])) { return; }
  // 全局 tid → LOD：ray_base 单调递增，取最后一个 base ≤ tid 的那级
  var lod = 0u;
  for (var l = 1u; l < DDGI_LOD_COUNT; l = l + 1u) {
    if (tid >= atomicLoad(&ddgi_indirect[DDGI_INDIR_RAYBASE_BASE + l])) { lod = l; }
  }
  let rpp = atomicLoad(&ddgi_indirect[DDGI_INDIR_RPP_BASE + lod]);
  if (rpp == 0u) { return; }
  let local = tid - atomicLoad(&ddgi_indirect[DDGI_INDIR_RAYBASE_BASE + lod]);
  let probe_idx = local / rpp;
  let ray = local % rpp;
  let slot = ddgi_lod_slot_base(lod) + probe_idx;
  let probe_pos = ddgi_worklist[slot].xyz;
  let frame = u32(ddgi_u.params.x);
  // RNG 种子用探针**自己的 cell 下标**（而非 worklist 排名）：排名每帧会变，
  // 会让同一个探针的射线方向逐帧跳变、叠加噪声。时间维的随机性由 frame 提供。
  let cell_idx = (bitcast<u32>(ddgi_worklist[slot].w) >> 8u) & DDGI_WL_IDX_MASK;

  // 【射线 ↔ 深度纹素绑定】每条射线固定归属一个八面体深度纹素，纹素按
  // (frame·rpp + ray) 轮换 → 每帧覆盖 rpp 个不同纹素、⌈N²/rpp⌉ 帧轮完一圈（N=8 → 64 个）；方向 =
  // 该纹素中心 + 纹素内抖动（抖动避免固定方向产生 banding，幅度恰好限制在纹素内）。
  //
  // 为什么必须绑定（旧版是 Fibonacci 球 + 每帧随机四元数**整体重旋**）：
  //   旧版 4 条射线撒在整球的 64 个纹素上，每个纹素每 ~16 帧才摊到 1 条、方向还完全
  //   随机 → collect 存进深度图的 mean/std 变成「该 20° 锥内**随机方向**命中距离的混合」，
  //   而不是 Chebyshev 需要的「**该方向**的最近距离 + 其噪声」：
  //     · mean 在墙角被远处命中抬高 → 采样点 dist > mean 不成立 → wd = 1 全额通过（硬漏光）
  //     · std 变成「锥内几何起伏」而非「同方向噪声」→ 墙角最大 → soft 巨大（软漏光）
  //   两者都随 LOD 变粗（cell 大、锥内起伏大）而加剧，正对上「LOD1 比 LOD0 漏得明显」。
  //   绑定后每个纹素的样本跨帧同属一个纹素方向 → mean/std 恢复物理含义，Chebyshev 成立。
  //   （DDGI 原版用 256 条低差异射线铺满 16×16 深度图 = 每纹素 1 条；这里在预算远小于
  //   纹素数时用「帧轮换」等价地保证「每条纹素最终拿到的是它自己的方向」。）
  let ndim = DDGI_DEPTH_TEXELS;
  let ntex = ndim * ndim;
  let tsel = (frame * rpp + ray) % ntex;
  let ttx = tsel % ndim;
  let tty = tsel / ndim;
  let jit = ddgi_rand2(ddgi_ray_rand(ddgi_lod_slot_base(lod) + cell_idx, frame, ray));
  let te = ((vec2<f32>(f32(ttx), f32(tty)) + jit) / f32(ndim)) * 2.0 - vec2<f32>(1.0);
  let dir = ddgi_oct_decode(te);
  let sh = world_raycast(make_grid(0u), probe_pos + dir * DDGI_RAY_BIAS, dir, DDGI_T_MAX, world_cfg_full(0.0, 3u));
  var radiance = sky_rgb();
  var dist = DDGI_T_MAX;
  if (sh.hit) {
    dist = sh.t;
    let alb = palette_albedo(sh.palette_base, sh.pal);
    let hit_p = probe_pos + dir * dist;
    // 与 dda_main 同一套逐体素法线（退化退面法线）：反弹辐照度的方向必须与着色侧一致，
    // 否则曲面上的间接光会与直射/环境光"对不上"。代价是每条 GI 射线 6 次 world_solid
    // （射线数 ≈ 屏幕像素数的 1/7，可接受）。
    let n = voxel_normal(
      make_grid(u32(sh.obj_id) + 1u),
      sh.voxel,
      sh.n,
    );
    // 入射辐照度 E → 出射辐亮度 L = albedo·E/π（与着色侧 col += albedo·E/π 同约定）。
    // 加一项 albedo·sky·CAST_FLOOR 作下限，避免首帧 GI 全 0 时反馈回路死锁在 0；
    // 该下限必须小，否则封闭房间的探针也被抬亮（室内外没有亮度差）。
    let gi = ddgi_sample(hit_p + n * 0.5, n);
    // 命中发光体素 → 出射辐亮度含**自身发射**（直出，不吃 GI/入射）：这是"体素即光源"
    // 往 GI 里注入光的关键接法（Douglas：命中体素取它的 lit color，发光体素的 lit color 含 emission）。
    radiance = alb * (gi / DDGI_PI + sky_rgb() * DDGI_CAST_FLOOR)
      + alb * palette_emissive(sh.palette_base, sh.pal) * DDGI_EMIT_GAIN;
  }
  // 样本下标 = 全局射线编号（tid 本身就是「该 LOD 起始射线号 + 本探针号×rpp + 射线号」）
  let si = tid * 2u;
  // 沿程发光聚合（「体素即光源」）：叠加在命中辐亮度之上，累加到几何首命中（无命中 → T_MAX）。
  // 这是**间接**贡献（发光 cell 摊成的体积密度）；命中发光体素时的自身发射由上面的
  // palette_emissive 直出项负责，两者语义不重叠。
  radiance = radiance + light_field_gather(probe_pos + dir * DDGI_RAY_BIAS, dir, dist);
  ddgi_samples[si] = vec4<f32>(dir, dist);
  ddgi_samples[si + 1u] = vec4<f32>(radiance, 1.0);
}

@compute @workgroup_size(64)
fn ddgi_collect(@builtin(global_invocation_id) gid: vec3<u32>) {
  let total_coll = atomicLoad(&ddgi_indirect[DDGI_INDIR_COLL_TOTAL]);
  let stride = 65535u * 64u;
  var tid = gid.x;
  loop {
    if (tid >= total_coll) { break; }
    ddgi_collect_one(tid);
    tid = tid + stride;
  }
}

@compute @workgroup_size(64)
fn probe_viz_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dot_size = u32(view_u.probe_viz_params.y);
  let id = gid.x;
  // x = 世界空间探针总槽数（Rust 每帧写入）
  if (id >= u32(view_u.probe_viz_params.x)) {
    return;
  }
  let lod = ddgi_slot_lod_of(id);
  let sel = u32(view_u.probe_viz_params.z);
  if (sel != 0u && lod != sel - 1u) {
    return;
  }
  // 只可视化「活跃」探针（邻域检测 + 非更细 LOD 覆盖），而非所有烘焙探针
  if ((ddgi_meta[id] & DDGI_META_ACTIVE) == 0u) {
    return;
  }
  let pos = ddgi_slot_pos[id];
  let dot_col = vec4<f32>(1.0, 0.85, 0.0, 1.0);
  let clip = view_u.view_proj * vec4<f32>(pos.xyz, 1.0);
  if (clip.w <= 0.0) {
    return;
  }
  let ndc = clip.xyz / clip.w;
  if (ndc.x < -1.0 || ndc.x > 1.0 || ndc.y < -1.0 || ndc.y > 1.0 || ndc.z < 0.0 || ndc.z > 1.0) {
    return;
  }
  let to_probe = pos.xyz - view_u.cam_pos_voxel.xyz;
  let dist = length(to_probe);
  let hit = world_raycast(make_grid(0u), view_u.cam_pos_voxel.xyz, to_probe / dist, dist, world_cfg_full(0.0, 3u));
  if (hit.hit && hit.t < dist - 0.5) {
    return;
  }
  let dims = vec2<f32>(textureDimensions(out_tex));
  let px = vec2<f32>((ndc.x * 0.5 + 0.5) * dims.x, (1.0 - (ndc.y * 0.5 + 0.5)) * dims.y);
  let ci = vec2<i32>(px);
  let half = i32(dot_size) / 2;
  for (var dy = -half; dy <= half; dy++) {
    for (var dx = -half; dx <= half; dx++) {
      let p = vec2<i32>(ci.x + dx, ci.y + dy);
      if (p.x >= 0 && p.y >= 0 && u32(p.x) < u32(dims.x) && u32(p.y) < u32(dims.y)) {
        textureStore(out_tex, vec2<u32>(p), dot_col);
      }
    }
  }
}

// ============================================================================
// 眼睛适应（自动曝光）——结构对齐 UE/Unity 的 EyeAdaptation：
//   pass1 `eye_adapt_histogram`：out_tex 按 1/16 步长抽样 → sRGB 还原 → log2 亮度 → 64 桶直方图
//   pass2 `eye_adapt_update`：取 5%~95% 百分位区间求平均（剔除"一小块亮窗/太阳"这类离群值）
//                             → 反馈 + 分方向时间平滑 → 写曝光系数（下一帧 `dda_main` 用）
// 两个 pass 都极小：pass1 = 1 个 WG（256 线程覆盖全部 ~880 个抽样点），pass2 = 1 个线程。
// 【为什么统计"曝光之后"的颜色】UE 的直方图同样在当前曝光已应用的颜色上统计 —— 反馈环把
// "百分位平均亮度"压到 EYE_KEY（目标中灰），这就是"进暗处几秒后能看见"的机制本身。
// ============================================================================
/// log2 亮度的桶覆盖：[-12, +12] 共 24 档 / 64 桶 = 0.375 档每桶。8bit 量化远粗于桶宽，
/// 所以从 8bit 的 out_tex 统计完全够用。
const EYE_LUM_MIN_LOG2: f32 = -12.0;
const EYE_LUM_RANGE_LOG2: f32 = 24.0;
const EYE_HIST_BINS: u32 = 64u;
/// 活参数区起始字（= 状态/调试区 8 + 直方图 64）。**这些原来是编译期常量**，现在由
/// debug overlay 的「Eye」页实时调（Rust 侧 [`EyeAdaptSettings`] 变化才上传 20B）。
/// 下标 = `eye_p(i)` 的实参，顺序即语义，改 Rust 结构体时必须同步这里：
///   0 ev_max  1 ev_min  2 tau_brighten  3 tau_darken  4 key
const EYE_PARAM_BASE: u32 = 8u + EYE_HIST_BINS;

fn eye_p(i: u32) -> f32 {
  return bitcast<f32>(atomicLoad(&eye_adapt[EYE_PARAM_BASE + i]));
}

/// WG 内共享直方图。**必须是 workgroup 作用域的模块级变量** —— WGSL 规定原子类型只能出现在
/// storage / workgroup 地址空间，函数内的局部 var 不能是 atomic。
var<workgroup> eye_hist: array<atomic<u32>, 64>;

@compute @workgroup_size(256)
fn eye_adapt_histogram(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dims = vec2<i32>(textureDimensions(out_tex_src));
  // 先局部原子累加，最后一次性写全局（全局原子量压到 64/帧）。
  for (var i = 0u; i < EYE_HIST_BINS; i = i + 1u) { atomicStore(&eye_hist[i], 0u); }
  workgroupBarrier();
  // 抽样：x/y 各每 16 像素取一个中心点；256 线程按 grid-stride 覆盖全部抽样点。
  let stride = 16u;
  let axis = u32(dims.x / 16);
  let step = axis * u32(dims.y / 16);
  let bin_w = EYE_LUM_RANGE_LOG2 / f32(EYE_HIST_BINS);
  for (var k = gid.x; k < step; k = k + 256u) {
    let sx = i32((k % axis) * stride + stride / 2u);
    let sy = i32((k / axis) * stride + stride / 2u);
    if (sx >= dims.x || sy >= dims.y) { continue; }
    let c = textureLoad(out_tex_src, vec2<i32>(sx, sy), 0).rgb;
    let luma = dot(srgb_to_linear(c), vec3<f32>(0.2126, 0.7152, 0.0722));
    let b = u32(clamp((log2(max(luma, 1e-5)) - EYE_LUM_MIN_LOG2) / bin_w, 0.0, f32(EYE_HIST_BINS - 1u)));
    atomicAdd(&eye_hist[b], 1u);
  }
  workgroupBarrier();
  for (var i = gid.x; i < EYE_HIST_BINS; i = i + 256u) {
    let v = atomicLoad(&eye_hist[i]);
    if (v > 0u) { atomicAdd(&eye_adapt[8u + i], v); }
  }
}

@compute @workgroup_size(1)
fn eye_adapt_update() {
  var total = 0.0;
  for (var i = 0u; i < EYE_HIST_BINS; i = i + 1u) { total = total + f32(atomicLoad(&eye_adapt[8u + i])); }
  if (total > 0.0) {
    // 5%~95% 百分位裁剪：按累计计数取区间内的加权平均 log2 亮度
    let lo = total * 0.05;
    let hi = total * 0.95;
    let bin_w = EYE_LUM_RANGE_LOG2 / f32(EYE_HIST_BINS);
    var cum = 0.0;
    var acc = 0.0;
    var wsum = 0.0;
    for (var i = 0u; i < EYE_HIST_BINS; i = i + 1u) {
      let n = f32(atomicLoad(&eye_adapt[8u + i]));
      let c0 = cum;
      cum = cum + n;
      let w = max(0.0, min(cum, hi) - max(c0, lo));
      acc = acc + (EYE_LUM_MIN_LOG2 + (f32(i) + 0.5) * bin_w) * w;
      wsum = wsum + w;
    }
    if (wsum > 0.0) {
      let avg_log2 = acc / wsum;
      atomicStore(&eye_adapt[4u], bitcast<u32>(avg_log2));
      let ev_cur = bitcast<f32>(atomicLoad(&eye_adapt[1u]));
      // 反馈：把"当前（已含曝光）的平均亮度"往目标中灰拉 → 还差多少档
      let ev_want = clamp(ev_cur + (log2(eye_p(4u)) - avg_log2), eye_p(1u), eye_p(0u));
      atomicStore(&eye_adapt[5u], bitcast<u32>(ev_want));
      let dt = clamp(bitcast<f32>(atomicLoad(&eye_adapt[3u])), 0.0, 0.25);
      let tau = select(eye_p(3u), eye_p(2u), ev_want > ev_cur);
      // 与帧率无关的指数平滑
      let k = select(0.0, 1.0 - exp(-dt / max(tau, 1e-3)), dt > 0.0);
      let ev_new = mix(ev_cur, ev_want, k);
      atomicStore(&eye_adapt[1u], bitcast<u32>(ev_new));
      let manual = bitcast<f32>(atomicLoad(&eye_adapt[2u]));
      let exp_val = exp2(clamp(ev_new + manual, eye_p(1u), eye_p(0u)));
      atomicStore(&eye_adapt[0u], bitcast<u32>(exp_val));
    }
  }
  // 直方图清零，留给下一帧累加（本 pass 与 pass1 分属不同 pass，pass 边界即屏障）
  for (var i = 0u; i < EYE_HIST_BINS; i = i + 1u) { atomicStore(&eye_adapt[8u + i], 0u); }
}
