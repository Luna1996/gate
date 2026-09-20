import package::bindings::{b_leaves, b_palette, b_struct, g, grid_descs, view_u};
import package::brickmap::{Grid};
import package::common::{CHUNK_INDEX_CAP, CHUNK_SIZE, DIR_LUT, FLAGS_IS_PBR, FLAGS_TRANSMISSIVE, NODE_FIXED_WORDS, face_normal_from_index};

// 层级命中记录：t 为 ro 系绝对 t；face_id 0..5 = ±xyz 六面（命中面法线索引）。
// voxel = 命中固体体素 chunk 局部整数坐标，着色阶段直接消费，不要用「命中点 ± 法线半步」重建。
struct VoxelHit {
  hit: bool,
  t: f32,
  pal: u32,
  face_id: u32,
  voxel: vec3<i32>,
}

// 介质穿透（透明/玻璃）：透射材质按可穿过介质处理，在 DDA 内部一路走完。
// `stop_on_air = true`：遇空气停在出射面；false：空气也穿过（阴影射线）；`tint` 沿途累积透射色（乘性）。
// `last_pal` = 当前介质段材质（0 = 不在介质内）—— `medium_exit` 用它判"是否在介质段内"，
// **同时兼作单格透过率的记忆**（`last_tiny`）：几何上相邻的格子绝大多数是同一材质
// ⇒ 命中记忆时省掉 `medium_of` 的两次 palette load（本函数在 DDA 热循环里逐格调用）。
// **注意 `last_pal` 不再是"同段只染一次"的去重键**：MT5-3 前它兼任去重，那会让 `tint` 按**段**
// 染一次而**与厚度无关**（厚玻璃不比薄玻璃暗，不是 Beer-Lambert）—— 见 `pass_through_medium`。
// `penetrate` 是按值参数、不在本结构里。
//
// **距离记账不在本结构里**：`tint` 的累乘点只有一个（`medium_flush`），距离 = 相邻两次介质调用之间的
// **实测 t 跨度 × |rd|**，由 `trace_chunk` 用局部变量 `t_prev` 逐格喂进来 —— 所以本结构只保存
// "当前段的材质 + 该材质的单格透过率"，不需要再存"上一格的入口 t"（也就不会多一个需要跨文件同步的字段：
// 本结构由 `world.wesl` 按位置构造，字段数一变就得同时改那边）。
struct MediumRay {
  stop_on_air: bool,
  palette_base: u32,
  last_pal: u32,
  last_tiny: vec3<f32>,  // `last_pal` 材质的**单格（1 体素）**透过率（未按距离取幂）
  tint: vec3<f32>,
}

// brick 缓存槽：一层分裂节点常驻（镜像 CPU `BrickCpu`）。
struct Brick {
  addr: u32,  // 节点绝对字址（b_struct）
  mask: u64,  // 64bit 分裂掩码（bit=1 = 子块分裂；bit=0 = 统一子块，色 = pal）
  pal: u32,   // 节点 palette（统一子块颜色，0 = 空气）
}

/// 读 palette 槽的介质属性：`rgb` = **单格（1 体素）透过率**（乘性），`a` = 不透明度（a == 1 = 不透明，非介质）。
/// 布局与 `wire.rs::pack_palette_entry` 一致（一次读两个 word）。
/// **介质判定读 `flags` 的 `TRANSMISSIVE` 位**（bit5）而不是 `transmission` 字节：本函数在 DDA 内
/// **逐格**调用，且 PBR 变体里 bit8..15 已是 `asset` 的一部分（读它得不到透射率）。
/// flags 与 transmission 同在 word1 ⇒ 零额外读取；该位由写入侧维护（见 `wire.rs::pack_palette_entry`）。
///
/// **为什么这里直接读 palette、不走 `fetch_material`**（硬约束 1 的唯一入口规则）：本函数逐格调用，
/// 而 `fetch_material` 的 PBR 分支要 triplanar 采样（6 次纹理采样）= 逐格做不起；
/// 与 `common.wesl::material_ior`（"只读 palette + 资产表、不采样贴图"，同一个成本理由）同一条例外。
/// 这里取的是**介质透过率**（DDA 的物理量），不是着色 —— 着色仍然只有 `brdf_reflected` 一条路径。
fn medium_of(palette_base: u32, pal: u32) -> vec4<f32> {
  let w1 = b_palette[palette_base + pal * 2u + 1u];
  // 非介质：改动前的 `a = 1 − transmission/255` 在 transmission == 0 时恒等于 1.0，且调用方只用
  // `a >= 1.0` 判"不透明"（rgb 那条路径不消费）⇒ 这里直接给 a = 1.0，行为与改动前逐行等价。
  if ((w1 & FLAGS_TRANSMISSIVE) == 0u) { return vec4<f32>(1.0, 1.0, 1.0, 1.0); }
  if ((w1 & FLAGS_IS_PBR) != 0u) {
    // PBR 变体：w1 低 16 位是 `asset`，读不出透射率 ⇒ 只按 TRANSMISSIVE 位判"是介质"，
    // 单格透过率取中性（白 = 不吸收）。**PBR 变体的介质吸收色应当走材质资产** —— 这里**未做**：
    // 读资产要额外一次资产表 load（逐格调用，成本敏感）。
    // 注：MT7 之后**已有**写入方能产出 `IS_PBR + TRANSMISSIVE` 条目（编辑器笔触），
    // 所以这条现在是"**有真实用例但尚缺**"，属待补项，不是"无人使用所以延后"。
    return vec4<f32>(1.0, 1.0, 1.0, 0.0);
  }
  let w0 = b_palette[palette_base + pal * 2u];
  let srgb = vec3<f32>(f32(w0 & 0xFFu), f32((w0 >> 8u) & 0xFFu), f32((w0 >> 16u) & 0xFFu)) / 255.0;
  let a = 1.0 - f32((w1 >> 8u) & 0xFFu) / 255.0;
  // 单格透过率 = `mix(白, 吸收色, 密度)`：**密度 = `a` = 1 − transmission/255**（0 = 全透 = 白 = 不吸收，
  // 1 = 不透 = 吸收色）。吸收色取 `srgb·srgb`：palette 色是 **sRGB 编码**字节，平方是 γ≈2 的廉价
  // 线性化（沿用 MT5-3 之前的既有近似），并让吸收更饱和 ⇒ "厚玻璃更暗 / 彩色玻璃出色"更可辨。
  return vec4<f32>(mix(vec3<f32>(1.0), srgb * srgb, a), a);
}

/// **诊断开关（临时 A/B 用，默认必须 0u）**：`1u` ⇒ 介质吸收**整体关闭**（吸收因子恒为白）。
/// 它把两个候选成因**一刀切开**（人工验证配方）：
///   · 置 `1u` 后图案**消失** ⇒ 图案来自**吸收**（就是本次要修的东西：MT5-3 的"按格整数"指数
///     ⇒ `tint` 随格数跳变 ⇒ 等高层条纹）；
///   · 置 `1u` 后图案**仍在** ⇒ 图案与吸收无关，去查几何侧：`main.wesl::GLASS_MAX_BOUNCE`
///     （玻璃状态机的多界面/全反射路径）与 `main.wesl::PBR_REFLECTION_ENABLED`（MT4 反射射线）。
/// 编译期常量 ⇒ `0u` 时整个分支被折叠掉，正常路径零开销（WESL 运行期读盘编译：改这一行 + 重启即生效）。
const DIAG_DISABLE_MEDIUM_TINT: u32 = 0u;

/// 吸收因子：`tiny^dist`（`dist` = 该段介质的**实际穿行距离**，单位 = 体素、允许分数）。
///
/// **为什么不再用"按格整数"**（MT5-3 的 `tiny^cells`，`cells = 1 << (2*level)`）：DDA 的格是
/// **整数**体素块，而射线在格内的真实距离是**分数**（斜射时一格内可达 `4^level·√3`）。拿整数格数
/// 当指数有两个后果：
///   ① **阶梯**：`tint` 每穿一格跳一个 `tiny` 因子，而路径长度是连续增长的 ⇒ `tint` 呈**等高层**，
///      视觉上就是贴着几何的条带（用户报告的"同心方框状条纹"的候选成因之一）。
///      **可证伪点**：环的间距应 ≈ **一个 DDA 格**的路径长度（level 0 ⇒ 1 体素）；改变物体尺寸/视角
///      时环数应随之变化，而不是固定不变。
///   ② **量纲错**：格数 ≈ L1（曼哈顿）路径长度 —— 斜射时最多偏 `√3`（偏暗），与"穿行体素数"不是一回事。
/// 改用实测距离后，`tint` 是路径长度的**连续**函数（台阶消失）。
///
/// **为什么不用 `pow` / `exp`**（成本红线：本函数在 DDA 热循环里逐格调用，见 MT5-3 的注释）：
///   ① **精度上不需要**：距离拆成"整数体素（**精确**二进制幂）+ 分数余量（线性化）"后，线性化只作用在
///      **一个体素以内**（小数部分 `f < 1`）：`mix(1, tiny, f)` 与 `tiny^f` 之差 ≈ `(f·ln tiny)²/2`，
///      即 ≤ `(ln tiny)²/2`（`tiny = 0.9` ⇒ 0.5%；`tiny = 0.5` ⇒ 24%，而那里已是深暗区）；
///      整数部分走二进制幂是**精确**的（不是 `exp2(e·log2 tiny)` 那种超函数近似）。
///   ② **鲁棒性**：`exp`/`log2` 路线要多存一个 `ln tiny`，且 `tiny = 0`（纯黑 + 全不透）时
///      `log2(0) = -inf` 会在 `dist = 0` 处给出 `0 · -inf = NaN` 并扩散到整幅画面；本结构没有这个坏点。
///   ③ **成本**：常见情形（level 0、`dist ≈ 1`）只走 1 轮循环（1 次 `vec3` 乘）+ 1 次 `mix`；
///      最坏情形（level 3、`dist ≤ 64·√3 ≈ 110.9`）走 7 轮（≤6 次平方 + ≤6 次乘）——
///      而那一格覆盖 **64³ 个体素**，这点 ALU 与 DDA 自身的位运算/palette 读相比可忽略。
///      对照：MT5-3 的 `medium_cells_tint` 在最坏情形是 6 次 `vec3` 乘，两者同量级。
///
/// **与 MT5-3 的差异（有意，写明）**：**整数距离处逐位相同** —— 轴对齐射线走整数格时 `f = 0`
/// （`mix = 1`）且整数幂精确，结果与 MT5-3 的 `tiny^cells` 完全一致；差异只出现在
/// **斜射 / 分数格 / 实测距离 ≠ 格边长** 的情形 —— 那正是本次要修掉的量化台阶。
fn medium_dist_tint(tiny: vec3<f32>, dist: f32) -> vec3<f32> {
  if (DIAG_DISABLE_MEDIUM_TINT != 0u) { return vec3<f32>(1.0); }
  // 整数体素部分：二进制幂（精确）。上界 127 的论证：DDA 单格最大边长 64 体素（level 3），
  // 斜射时格内真实距离 ≤ 64·√3 ≈ 110.9 < 128 ⇒ 7 bit 足够（`clamp` 兼作负值/异常防御）。
  let fl = floor(dist);
  var n = u32(clamp(fl, 0.0, 127.0));
  var acc = vec3<f32>(1.0);
  var b = tiny;
  loop {
    if ((n & 1u) != 0u) { acc = acc * b; }
    n = n >> 1u;
    if (n == 0u) { break; }
    b = b * b;
  }
  // 分数部分（< 1 体素）：线性化 —— `mix(1, tiny, f)` = `tiny^f + O(f²·(ln tiny)²/2)`。
  return acc * mix(vec3<f32>(1.0), tiny, max(dist - fl, 0.0));
}

/// **唯一的 `tint` 累乘点**：结算 `(t_prev, t_now]` 这一段距离的吸收。
///
/// 距离 = `(t_now − t_prev) · |rd|`（体素）。**为什么用实测 t 跨度而不是格边长**：DDA 只在跨格边界时
/// 更新 t（见 `trace_chunk` 的步进），于是相邻两次介质调用之间的 t 跨度**就是**射线在该格内实际走过的
/// 距离（自动含斜射的 `1/|d_axis|` 因子与分数格）⇒ `tint` 对路径长度连续，也不再拿 L1 格数当欧氏距离。
///
/// **滞后一格的约定（有界近似，写明）**：结算发生在**下一次**介质调用处 —— 只有到那时才知道上一格的出口 t。
/// 于是每次结算用的是"上一格的材质"（`last_pal` / `last_tiny`）：**同一材质段内精确**（玻璃物体的绝大多数
/// 情形），材质交界处至多把一格的间隔记到相邻材质上（误差有界，且 ≤ 一格自身的吸收量）。
/// `last_pal == 0`（不在介质内：空气段）不结算 —— 那段距离不该吸收。
///
/// **跨 chunk 不丢距离**：`trace_chunk` 在"越过本 chunk 的 t1"时先用 `t1` 收尾（见那里的注释），
/// 下一个 chunk 的 `t_prev` 从 `t0 = t1` 起步 ⇒ 两段拼起来仍等于整段距离（不重不漏）。
fn medium_flush(mc: ptr<function, MediumRay>, t_now: f32, t_prev: f32, t_scale: f32) {
  if ((*mc).last_pal != 0u) {
    let dist = max(t_now - t_prev, 0.0) * t_scale;
    (*mc).tint = (*mc).tint * medium_dist_tint((*mc).last_tiny, dist);
  }
}

/// 可穿透介质 → 返回 true（调用方按介质继续步进）；不可穿透（`a >= 1`，含 `transmission == 0`）返回 false。
/// 只在 `penetrate = true` 时被调用 —— 而 `penetrate = true` 只有一条入口链
/// （`trace_glass`（主 pass 的玻璃状态机）与 `gi/ray.wesl` 的 `sun_transmittance` / 二次射线
/// 都经 `world_trace_medium` → `trace_grid` → 本函数）⇒ 硬约束 3「口径不分叉」由**只有这一份实现**保证。
///
/// 本函数**不再自己累乘 `tint`**：吸收由 `medium_flush` 按实测距离结算（见那里的注释），这里只做两件事：
///   ① 结算上一格；② 判本格是不是介质、并把它的单格透过率记进 `last_tiny`。
/// 距离相关（Beer-Lambert）的性质因此**只落在 `medium_dist_tint` 一个函数里**，没有第二处口径。
/// `t_prev`/`t_now`/`t_scale` 由调用方（`trace_chunk`）给：`t_now` = 本格的入口 t、`t_prev` = 上一格的入口 t、
/// `t_scale` = `|rd|`（⇒ `(t_now − t_prev)·t_scale` = 上一格内的实际穿行距离）。
fn pass_through_medium(mc: ptr<function, MediumRay>, pal: u32,
                       t_now: f32, t_prev: f32, t_scale: f32) -> bool {
  // ① 结算上一格：距离 = 本格入口 t − 上格入口 t（= 上一格内的实际穿行距离）。
  medium_flush(mc, t_now, t_prev, t_scale);
  // ② 材质不变 ⇒ 直接用记忆的单格透过率（省两次 palette load）；变了才重算。
  if ((*mc).last_pal != pal) {
    let m = medium_of((*mc).palette_base, pal);
    if (m.a >= 1.0) { return false; }
    (*mc).last_pal = pal;
    (*mc).last_tiny = m.rgb;
  }
  return true;
}

/// 介质段内撞到空气 = 出射面：返回 true（调用方带回 `last_pal`），只在 `penetrate = true` 时被调用。
fn medium_exit(mc: ptr<function, MediumRay>) -> bool {
  return (*mc).stop_on_air && ((*mc).last_pal != 0u);
}

// 单 chunk 内层级遍历（不透明版）：chunk_base = 根节点绝对字址，chunk_min = chunk 原点（局部 voxel）。
// [t0,t1] 与返回 t 为 ro 系绝对 t；entry_face = 进入面；depth_cap 下 gate 深度 d=3-level 达上限时返回子块入口 t（保守下界）。
fn trace_chunk_opaque(chunk_base: u32, chunk_min: vec3<f32>,
                      ro: vec3<f32>, rd: vec3<f32>, sign_v: vec3<i32>,
                      t0: f32, t1: f32, entry_face: u32, depth_cap: u32) -> VoxelHit {
  if (t0 >= t1) { return VoxelHit(false, 0.0, 0u, 0u, vec3<i32>(0)); }
  // chunk 局部 voxel 坐标（chunk 原点 = 0）；t 仍是 ro 系绝对 t
  let ro_c = ro - chunk_min;
  let inv_rd = 1.0 / rd;
  let abs_inv_rd = abs(inv_rd);
  let axis_on = abs(rd) > vec3<f32>(1e-30);
  let oct = select(0u, 1u, sign_v.x >= 0)
    | (select(0u, 1u, sign_v.y >= 0) << 1u)
    | (select(0u, 1u, sign_v.z >= 0) << 2u);
  let dir_side = select(vec3<f32>(0.0), vec3<f32>(1.0), sign_v >= vec3<i32>(0));
  let face_base = vec3<u32>(                     // 每轴两个面的 face id 基（0/1、2/3、4/5）
    select(1u, 0u, sign_v.x >= 0),
    select(1u, 0u, sign_v.y >= 0),
    select(1u, 0u, sign_v.z >= 0));
  var bricks: array<Brick, 4u>;
  let r_ml = b_struct[chunk_base];
  let r_mh = b_struct[chunk_base + 1u];
  let r_pal = b_struct[chunk_base + 2u] & 0xFFFFu;  // 节点字低 16 位 = uniform 色（16 位材质索引）
  bricks[3u] = Brick(chunk_base, (u64(r_mh) << 32u) | u64(r_ml), r_pal);
  var level: u32 = 3u;
  // 当前体素（chunk 局部 voxel 整数坐标 0..255；跨出 chunk 的步进瞬态可达 -1/256）
  let p0 = ro_c + rd * t0;
  var v = clamp(vec3<i32>(floor(p0)), vec3<i32>(0), vec3<i32>(255));
  var cur_t = t0;
  var face = entry_face;
  // 防挂死安全网：迭代上限。
  var budget: u32 = 65536u;
  let lut_disable = view_u.lod.w > 0.5;
  loop {
    if (budget == 0u) { break; }
    budget = budget - 1u;
    if (level > 3u) { return VoxelHit(false, 0.0, 0u, 0u, vec3<i32>(0)); }
    // ---- traverse：从当前 level 下钻到 v 处内容 ----
    loop {
      let b = bricks[level];
      // WGSL 移位 RHS 必须 u32
      let sh = vec3<u32>(level * 2u);
      let cell = vec3<i32>(v >> sh) & vec3<i32>(3);
      let idx = u32(cell.z * 16 + cell.y * 4 + cell.x);
      if (level == 0u) {
        // 叶节点 inline palette：bit=1（非空体素）才 load inline word 取色；每字 2 个体素（16 位索引）。
        if ((b.mask & (u64(1) << idx)) != u64(0)) {
          let w = b_struct[b.addr + NODE_FIXED_WORDS + (idx >> 1u)];
          let leaf_pal = (w >> ((idx & 1u) * 16u)) & 0xFFFFu;
          if (leaf_pal != 0u) { return VoxelHit(true, cur_t, leaf_pal, face, v); }
        }
        break;
      }
      let bit = u64(1) << idx;
      if ((b.mask & bit) == u64(0)) {
        // 统一子块：色 = 节点 palette（0 = 空气）
        if (b.pal != 0u) { return VoxelHit(true, cur_t, b.pal, face, v); }
        break;
      }
      // depth_cap（beam 保守）：gate 深度 d=3-level 的分裂子块到达 cap → 子块入口 t
      let gd = 3u - level;
      if (gd >= depth_cap) { return VoxelHit(true, cur_t, b.pal, face, v); }
      let below = b.mask & (bit - u64(1));
      let pop = countOneBits(u32(below)) + countOneBits(u32(below >> 32u));
      let child_addr = chunk_base + b_struct[b.addr + NODE_FIXED_WORDS + pop];
      let c_ml = b_struct[child_addr];
      let c_mh = b_struct[child_addr + 1u];
      let c_pw = b_struct[child_addr + 2u];
      let c_pal = c_pw & 0xFFFFu;  // 节点字低 16 位 = uniform 色
      let c_mask = (u64(c_mh) << 32u) | u64(c_ml);
      // 统一子节点快路径：wire 任意层的分裂位都可能指向 3 字统一节点（mask=0，禁读 addr+3 之后）
      if (c_mask == u64(0)) {
        if (c_pal != 0u) { return VoxelHit(true, cur_t, c_pal, face, v); }
        break;
      }
      level = level - 1u;
      bricks[level] = Brick(child_addr, c_mask, c_pal);
    }
    // ---- 整砖 LUT 跳过（仅 palette==0 节点安全）----
    {
      let b = bricks[level];
      if (b.pal == 0u) {
        let sh = vec3<u32>(level * 2u);
        let ecell = vec3<i32>(v >> sh) & vec3<i32>(3);
        let entry_i = u32(ecell.z * 16 + ecell.y * 4 + ecell.x);
        let lut_base = min(oct * 64u + entry_i, 511u);
        var reach = b_leaves[lut_base];
        reach = select(reach, ~u64(0), lut_disable);
        if ((b.mask & reach) == u64(0)) {
          level = level + 1u;
          if (level > 3u) { return VoxelHit(false, 0.0, 0u, 0u, vec3<i32>(0)); }
        }
      }
    }
    // ---- v 处为空气：当前 level brick 内 DDA ----
    let log2 = level * 2u;
    let s = 1i << (level * 2u); // 子块边长 voxel：1/4/16/64
    let s_mask = ~(s - 1i);
    // side_distance_for_ray：v 对齐到 s 的基址；正向 → 基址+s，负向 → 基址
    let base_v = v & vec3<i32>(s_mask);
    let boundary = vec3<f32>(base_v) + dir_side * f32(s);
    var side = vec3<f32>(1e+30);
    side = select(side, (boundary - ro_c) * inv_rd, axis_on);
    side = max(side, vec3<f32>(cur_t));
    let step_inc = f32(s) * abs_inv_rd;
    var step_axis: u32 = 0u;
    var changed = false;
    loop {
      var mn: u32;
      if (side.x <= side.y && side.x <= side.z) { mn = 0u; }
      else if (side.y <= side.z) { mn = 1u; }
      else { mn = 2u; }
      step_axis = mn;
      cur_t = side[mn];
      if (cur_t >= t1) { return VoxelHit(false, 0.0, 0u, 0u, vec3<i32>(0)); }
      let old_cell = (v[mn] >> log2) & 3;
      v[mn] = v[mn] + sign_v[mn] * s;
      side[mn] = side[mn] + step_inc[mn];
      face = mn * 2u + face_base[mn];
      let crossed = select(old_cell == 0, old_cell == 3, sign_v[mn] >= 0);
      if (crossed) { changed = true; break; }
      let b = bricks[level];
      let sh2 = vec3<u32>(log2);
      let cell = vec3<i32>(v >> sh2) & vec3<i32>(3);
      let idx = u32(cell.z * 16 + cell.y * 4 + cell.x);
      if (level == 0u) {
        if (b.mask != u64(0) && (b.mask & (u64(1) << idx)) != u64(0)) {
          let w = b_struct[b.addr + NODE_FIXED_WORDS + (idx >> 1u)];
          let dp = (w >> ((idx & 1u) * 16u)) & 0xFFFFu;
          if (dp != 0u) { return VoxelHit(true, cur_t, dp, face, v); }
        } else if (b.mask == u64(0) && b.pal != 0u) {
          return VoxelHit(true, cur_t, b.pal, face, v);
        }
      } else {
        let mb = (b.mask & (u64(1) << idx)) != u64(0);
        if (mb) { break; }
        if (b.pal != 0u) { return VoxelHit(true, cur_t, b.pal, face, v); }
      }
    }
    if (changed) {
      level = level + 1u;
      if (level > 3u) { return VoxelHit(false, 0.0, 0u, 0u, vec3<i32>(0)); }
    }
    // ---- firstTrailingBit 层级自适应跨级跳 ----
    // 步进轴新坐标的尾随零位 = 对齐 run 长度，tz>>1 = 可跨步的最粗 level。
    let positive = sign_v[step_axis] >= 0;
    let cur_log2 = level * 2u;
    let m: u32 = 0xFFFFFFFFu << cur_log2;
    let not_m = ~m;
    let vmin_u = u32(v[step_axis]);
    var comp: u32;
    if (positive) { comp = vmin_u & m; } else { comp = (vmin_u & m) | not_m; }
    let tz_i = firstTrailingBit(comp + select(1u, 0u, positive));
    var new_level: u32 = 0xFFFFFFFFu;
    if (tz_i >= 0) { new_level = u32(tz_i) >> 1u; }
    level = max(level, new_level);
    if (level > 3u) { return VoxelHit(false, 0.0, 0u, 0u, vec3<i32>(0)); }
    // 对齐快照：v 钳到 cur_t 处当前 level 区域，步进轴取精确边界整数。
    let mi = i32(m);
    let base = v & vec3<i32>(mi);
    let p = ro_c + rd * cur_t;
    let region_max = base + vec3<i32>(i32(not_m));
    v = clamp(vec3<i32>(floor(p)), base, region_max);
    v[step_axis] = i32(comp);
  }
  return VoxelHit(false, 0.0, 0u, 0u, vec3<i32>(0));
}

// 单 chunk 内层级遍历（介质穿透版）：调用点恒传穿透语义，无 `penetrate` 开关；mc = 介质穿透上下文。
// 命中 / 穿透 / 出射面判定只在两处唯一出口做。
// **介质吸收的距离记账**（`tint` 的唯一累乘点 = `medium_flush`）：本函数把"上一格的入口 t"记在 `t_prev`，
// 每次介质调用传 `(cur_t, t_prev, t_scale)` ⇒ 结算区间 = `(t_prev, cur_t]`，距离 = 该跨度 × |rd|。
// 三件事保证**不重不漏**（每段距离恰好结算一次）：
//   ① 每次跨格前 `t_prev = cur_t`（见内层 DDA 的第一步）；
//   ② 越过本 chunk 的 t1 时先用 `t1` 收尾（下半段由下一个 chunk 用它的 t0 = t1 起步接着算）；
//   ③ 段首（`last_pal == 0`）不结算 —— 那段在空气里。
fn trace_chunk(chunk_base: u32, chunk_min: vec3<f32>,
               ro: vec3<f32>, rd: vec3<f32>, sign_v: vec3<i32>,
               t0: f32, t1: f32, entry_face: u32, depth_cap: u32,
               mc: ptr<function, MediumRay>) -> VoxelHit {
  if (t0 >= t1) { return VoxelHit(false, 0.0, 0u, 0u, vec3<i32>(0)); }
  // 介质距离记账的两个局部量（函数作用域：要跨外层 loop 的迭代保持）：
  //   · `t_scale` = |rd| ⇒ `t` 跨度 × 它 = 该格内的**真实距离**（体素）。主世界 rd 是单位向量（= 1）；
  //     物体上 rd 已被变换（|rd| = 1/scale）⇒ 两种情形都得到"本 grid 体素"为单位的距离。
  //   · `t_prev` = **上一格的入口 t**（每次跨格前更新）⇒ 结算区间 = `(t_prev, cur_t]`（见 `medium_flush`）。
  //     从 `t0` 起步：跨 chunk 调用时 t0 就是上一个 chunk 的 t1 ⇒ 距离不重不漏。
  let t_scale = length(rd);
  var t_prev = t0;
  // chunk 局部 voxel 坐标（chunk 原点 = 0）；t 仍是 ro 系绝对 t
  let ro_c = ro - chunk_min;
  let inv_rd = 1.0 / rd;
  let abs_inv_rd = abs(inv_rd);
  let axis_on = abs(rd) > vec3<f32>(1e-30);
  let oct = select(0u, 1u, sign_v.x >= 0)
    | (select(0u, 1u, sign_v.y >= 0) << 1u)
    | (select(0u, 1u, sign_v.z >= 0) << 2u);
  let dir_side = select(vec3<f32>(0.0), vec3<f32>(1.0), sign_v >= vec3<i32>(0));
  let face_base = vec3<u32>(                     // 每轴两个面的 face id 基（0/1、2/3、4/5）
    select(1u, 0u, sign_v.x >= 0),
    select(1u, 0u, sign_v.y >= 0),
    select(1u, 0u, sign_v.z >= 0));
  var bricks: array<Brick, 4u>;
  let r_ml = b_struct[chunk_base];
  let r_mh = b_struct[chunk_base + 1u];
  let r_pal = b_struct[chunk_base + 2u] & 0xFFFFu;  // 节点字低 16 位 = uniform 色（16 位材质索引）
  bricks[3u] = Brick(chunk_base, (u64(r_mh) << 32u) | u64(r_ml), r_pal);
  var level: u32 = 3u;
  // 当前体素（chunk 局部 voxel 整数坐标 0..255；跨出 chunk 的步进瞬态可达 -1/256）
  let p0 = ro_c + rd * t0;
  var v = clamp(vec3<i32>(floor(p0)), vec3<i32>(0), vec3<i32>(255));
  var cur_t = t0;
  var face = entry_face;
  // 防挂死安全网：迭代上限。
  var budget: u32 = 65536u;
  let lut_disable = view_u.lod.w > 0.5;
  loop {
    if (budget == 0u) { break; }
    budget = budget - 1u;
    if (level > 3u) { return VoxelHit(false, 0.0, 0u, 0u, vec3<i32>(0)); }
    // ---- traverse：从当前 level 下钻到 v 处内容 ----
    var hit_pal = 0u;
    loop {
      let b = bricks[level];
      // WGSL 移位 RHS 必须 u32
      let sh = vec3<u32>(level * 2u);
      let cell = vec3<i32>(v >> sh) & vec3<i32>(3);
      let idx = u32(cell.z * 16 + cell.y * 4 + cell.x);
      if (level == 0u) {
        // 叶节点 inline palette：bit=1（非空体素）才 load inline word 取色；每字 2 个体素（16 位索引）。
        if ((b.mask & (u64(1) << idx)) != u64(0)) {
          let w = b_struct[b.addr + NODE_FIXED_WORDS + (idx >> 1u)];
          hit_pal = (w >> ((idx & 1u) * 16u)) & 0xFFFFu;
        }
        break;
      }
      let bit = u64(1) << idx;
      if ((b.mask & bit) == u64(0)) {
        // 统一子块：色 = 节点 palette（0 = 空气）
        hit_pal = b.pal;
        break;
      }
      // depth_cap（beam 保守）：gate 深度 d=3-level 的分裂子块到达 cap → 子块入口 t
      let gd = 3u - level;
      if (gd >= depth_cap) { return VoxelHit(true, cur_t, b.pal, face, v); }
      let below = b.mask & (bit - u64(1));
      let pop = countOneBits(u32(below)) + countOneBits(u32(below >> 32u));
      let child_addr = chunk_base + b_struct[b.addr + NODE_FIXED_WORDS + pop];
      let c_ml = b_struct[child_addr];
      let c_mh = b_struct[child_addr + 1u];
      let c_pw = b_struct[child_addr + 2u];
      let c_pal = c_pw & 0xFFFFu;  // 节点字低 16 位 = uniform 色
      let c_mask = (u64(c_mh) << 32u) | u64(c_ml);
      // 统一子节点快路径：wire 任意层的分裂位都可能指向 3 字统一节点（mask=0，禁读 addr+3 之后）
      if (c_mask == u64(0)) {
        hit_pal = c_pal;
        break;
      }
      level = level - 1u;
      bricks[level] = Brick(child_addr, c_mask, c_pal);
    }
    // ---- traverse 唯一出口：命中 / 穿透 / 出射面 ----
    // 距离记账：`pass_through_medium` 在开头用 `(cur_t − t_prev)` 结算**上一格**（见 `medium_flush`）。
    if (hit_pal != 0u) {
      if (!pass_through_medium(mc, hit_pal, cur_t, t_prev, t_scale)) {
        return VoxelHit(true, cur_t, hit_pal, face, v);
      }
    } else {
      // 空气格 = 本段介质的终点：先结算上一格的吸收，再判"是否到了出射面"。
      medium_flush(mc, cur_t, t_prev, t_scale);
      if (medium_exit(mc)) { return VoxelHit(true, cur_t, (*mc).last_pal, face, v); }
      (*mc).last_pal = 0u;
    }
    // ---- 整砖 LUT 跳过（仅 palette==0 节点安全）----
    {
      let b = bricks[level];
      if (b.pal == 0u) {
        let sh = vec3<u32>(level * 2u);
        let ecell = vec3<i32>(v >> sh) & vec3<i32>(3);
        let entry_i = u32(ecell.z * 16 + ecell.y * 4 + ecell.x);
        let lut_base = min(oct * 64u + entry_i, 511u);
        var reach = b_leaves[lut_base];
        reach = select(reach, ~u64(0), lut_disable);
        if ((b.mask & reach) == u64(0)) {
          level = level + 1u;
          if (level > 3u) { return VoxelHit(false, 0.0, 0u, 0u, vec3<i32>(0)); }
        }
      }
    }
    // ---- v 处为空气：当前 level brick 内 DDA ----
    let log2 = level * 2u;
    let s = 1i << (level * 2u); // 子块边长 voxel：1/4/16/64
    let s_mask = ~(s - 1i);
    // side_distance_for_ray：v 对齐到 s 的基址；正向 → 基址+s，负向 → 基址
    let base_v = v & vec3<i32>(s_mask);
    let boundary = vec3<f32>(base_v) + dir_side * f32(s);
    var side = vec3<f32>(1e+30);
    side = select(side, (boundary - ro_c) * inv_rd, axis_on);
    side = max(side, vec3<f32>(cur_t));
    let step_inc = f32(s) * abs_inv_rd;
    var step_axis: u32 = 0u;
    var changed = false;
    loop {
      var mn: u32;
      if (side.x <= side.y && side.x <= side.z) { mn = 0u; }
      else if (side.y <= side.z) { mn = 1u; }
      else { mn = 2u; }
      step_axis = mn;
      // 跨格前记下"上一格的入口 t"：上一格的实际穿行距离 = (新的 cur_t − t_prev)·|rd|（见 medium_flush）。
      t_prev = cur_t;
      cur_t = side[mn];
      if (cur_t >= t1) {
        // 越出本 chunk：把上一格到边界的距离先结算掉（`t_prev → t1`），其余部分由下一个 chunk 接着算
        // （它的 t0 就是本 chunk 的 t1，见函数头注释）⇒ 距离不重不漏、也不会把边界外的距离算进来。
        medium_flush(mc, t1, t_prev, t_scale);
        return VoxelHit(false, 0.0, 0u, 0u, vec3<i32>(0));
      }
      let old_cell = (v[mn] >> log2) & 3;
      v[mn] = v[mn] + sign_v[mn] * s;
      side[mn] = side[mn] + step_inc[mn];
      face = mn * 2u + face_base[mn];
      let crossed = select(old_cell == 0, old_cell == 3, sign_v[mn] >= 0);
      if (crossed) { changed = true; break; }
      let b = bricks[level];
      let sh2 = vec3<u32>(log2);
      let cell = vec3<i32>(v >> sh2) & vec3<i32>(3);
      let idx = u32(cell.z * 16 + cell.y * 4 + cell.x);
      hit_pal = 0u;
      if (level == 0u) {
        if (b.mask != u64(0) && (b.mask & (u64(1) << idx)) != u64(0)) {
          let w = b_struct[b.addr + NODE_FIXED_WORDS + (idx >> 1u)];
          hit_pal = (w >> ((idx & 1u) * 16u)) & 0xFFFFu;
        } else if (b.mask == u64(0)) {
          hit_pal = b.pal;
        }
      } else {
        if ((b.mask & (u64(1) << idx)) != u64(0)) { break; }
        hit_pal = b.pal;
      }
      if (hit_pal != 0u) {
        if (pass_through_medium(mc, hit_pal, cur_t, t_prev, t_scale)) { continue; }
        return VoxelHit(true, cur_t, hit_pal, face, v);
      }
      // 空气格：与 traverse 出口同一处理 —— 先结算上一格（本段介质到此结束），再判出射面。
      medium_flush(mc, cur_t, t_prev, t_scale);
      if (medium_exit(mc)) { return VoxelHit(true, cur_t, (*mc).last_pal, face, v); }
      (*mc).last_pal = 0u;
    }
    if (changed) {
      level = level + 1u;
      if (level > 3u) { return VoxelHit(false, 0.0, 0u, 0u, vec3<i32>(0)); }
    }
    // ---- firstTrailingBit 层级自适应跨级跳 ----
    // 步进轴新坐标的尾随零位 = 对齐 run 长度，tz>>1 = 可跨步的最粗 level。
    let positive = sign_v[step_axis] >= 0;
    let cur_log2 = level * 2u;
    let m: u32 = 0xFFFFFFFFu << cur_log2;
    let not_m = ~m;
    let vmin_u = u32(v[step_axis]);
    var comp: u32;
    if (positive) { comp = vmin_u & m; } else { comp = (vmin_u & m) | not_m; }
    let tz_i = firstTrailingBit(comp + select(1u, 0u, positive));
    var new_level: u32 = 0xFFFFFFFFu;
    if (tz_i >= 0) { new_level = u32(tz_i) >> 1u; }
    level = max(level, new_level);
    if (level > 3u) { return VoxelHit(false, 0.0, 0u, 0u, vec3<i32>(0)); }
    // 对齐快照：v 钳到 cur_t 处当前 level 区域，步进轴取精确边界整数。
    let mi = i32(m);
    let base = v & vec3<i32>(mi);
    let p = ro_c + rd * cur_t;
    let region_max = base + vec3<i32>(i32(not_m));
    v = clamp(vec3<i32>(floor(p)), base, region_max);
    v[step_axis] = i32(comp);
  }
  return VoxelHit(false, 0.0, 0u, 0u, vec3<i32>(0));
}

// 统一 DDA：主网格与逐物体 OBJ 共用一条管线，严格镜像 Rust `brickmap/dda.rs` CPU 参考实现。

// slab 法射线-AABB 求交，返回 (t_enter, t_exit)；平行且在外 → (1.0, 0.0) miss 哨兵。
fn slab_box(ro: vec3<f32>, rd: vec3<f32>, mn: vec3<f32>, mx: vec3<f32>, t0: f32, t1: f32) -> vec2<f32> {
  var t_enter = t0;
  var t_exit = t1;
  if (abs(rd.x) < 1e-30) {
    if (ro.x < mn.x || ro.x > mx.x) { return vec2<f32>(1.0, 0.0); }
  } else {
    let ta = (mn.x - ro.x) / rd.x;
    let tb = (mx.x - ro.x) / rd.x;
    t_enter = max(t_enter, min(ta, tb));
    t_exit = min(t_exit, max(ta, tb));
  }
  if (abs(rd.y) < 1e-30) {
    if (ro.y < mn.y || ro.y > mx.y) { return vec2<f32>(1.0, 0.0); }
  } else {
    let ta = (mn.y - ro.y) / rd.y;
    let tb = (mx.y - ro.y) / rd.y;
    t_enter = max(t_enter, min(ta, tb));
    t_exit = min(t_exit, max(ta, tb));
  }
  if (abs(rd.z) < 1e-30) {
    if (ro.z < mn.z || ro.z > mx.z) { return vec2<f32>(1.0, 0.0); }
  } else {
    let ta = (mn.z - ro.z) / rd.z;
    let tb = (mx.z - ro.z) / rd.z;
    t_enter = max(t_enter, min(ta, tb));
    t_exit = min(t_exit, max(ta, tb));
  }
  return vec2<f32>(t_enter, t_exit);
}

// 统一 DDA 核心：主世界与物体走同一条 slab→chunk 间 A&W→trace_chunk 流程，差异由 `Grid` 参数化。

// 统一命中结构：hit / t / pal / n（世界空间法线）/ face_id（0..5）/ obj_id（-1 = 主世界，>=0 = 物体）。
// voxel 为命中固体体素 grid 局部整数坐标（主世界 = 世界坐标），由 DDA 整数步进精确产出。
struct UnifiedHit {
  hit: bool,
  t: f32,
  pal: u32,
  n: vec3<f32>,        // 世界空间法线（光影用）
  face_id: u32,        // 命中面 0..5（与 face_index_from_normal 对齐）
  voxel: vec3<i32>,    // grid 局部命中体素坐标
  obj_id: i32,         // -1 = 主世界, >=0 = 物体索引
}

/// chunk 级命中 → grid 级命中：面法线换到世界系，体素坐标补 chunk 偏移（`ci` = 命中 chunk）。
fn grid_hit(h: VoxelHit, ci: vec3<i32>,
            col0: vec3<f32>, col1: vec3<f32>, col2: vec3<f32>, obj_id: i32) -> UnifiedHit {
  let n_local = face_normal_from_index(h.face_id);
  let n_world = normalize(n_local.x * col0 + n_local.y * col1 + n_local.z * col2);
  let voxel = h.voxel + ci * i32(CHUNK_SIZE);
  return UnifiedHit(true, h.t, h.pal, n_world, h.face_id, voxel, obj_id);
}

// 统一层次 DDA：slab→chunk 间 A&W→trace_chunk，`penetrate = false` 时一次都不碰 mc。
fn trace_grid(g: Grid, origin: vec3<f32>, dir: vec3<f32>, t_cap: f32, t_min: f32, depth_cap: u32,
              penetrate: bool, mc: ptr<function, MediumRay>) -> UnifiedHit {
  let w_mn = g.w_mn;
  let w_mx = g.w_mx;
  let l_min = g.l_min;
  let l_max = g.l_max;
  let index_origin = g.index_origin;
  let index_dims = g.index_dims;
  let tree_base = g.tree_base;
  let max_chunk_steps = g.max_chunk_steps;
  let col0 = g.col0;
  let col1 = g.col1;
  let col2 = g.col2;
  let obj_id = g.obj_id;
  let miss = UnifiedHit(false, 0.0, 0u, vec3<f32>(0.0), 0u, vec3<i32>(0), obj_id);
  // ---- 世界 AABB 预剔除（slab 内联）----
  var bx_enter = 0.0;
  var bx_exit = t_cap;
  if (abs(dir.x) < 1e-30) {
    if (origin.x < w_mn.x || origin.x > w_mx.x) { return miss; }
  } else {
    let ta = (w_mn.x - origin.x) / dir.x;
    let tb = (w_mx.x - origin.x) / dir.x;
    bx_enter = max(bx_enter, min(ta, tb));
    bx_exit = min(bx_exit, max(ta, tb));
  }
  if (abs(dir.y) < 1e-30) {
    if (origin.y < w_mn.y || origin.y > w_mx.y) { return miss; }
  } else {
    let ta = (w_mn.y - origin.y) / dir.y;
    let tb = (w_mx.y - origin.y) / dir.y;
    bx_enter = max(bx_enter, min(ta, tb));
    bx_exit = min(bx_exit, max(ta, tb));
  }
  if (abs(dir.z) < 1e-30) {
    if (origin.z < w_mn.z || origin.z > w_mx.z) { return miss; }
  } else {
    let ta = (w_mn.z - origin.z) / dir.z;
    let tb = (w_mx.z - origin.z) / dir.z;
    bx_enter = max(bx_enter, min(ta, tb));
    bx_exit = min(bx_exit, max(ta, tb));
  }
  let bx0 = max(bx_enter, 0.0);
  if (bx_exit < bx0 || bx_enter >= t_cap) { return miss; }
  let t_hi_cap = min(bx_exit, t_cap);
  let scale = g.scale;
  // 主世界（obj_id < 0）= identity 变换（make_grid 保证 pos=0/rot=I/scale=1），局部坐标直取世界坐标。
  var ro: vec3<f32>;
  var rd: vec3<f32>;
  var tl0: f32;
  var tl1: f32;
  if (obj_id < 0) {
    ro = origin;
    rd = dir;
    tl0 = bx0;
    tl1 = t_hi_cap;
  } else {
    // ---- 局部变换 ----
    let wp = origin - g.pos;
    ro = vec3<f32>(dot(wp, col0), dot(wp, col1), dot(wp, col2)) / scale;
    rd = vec3<f32>(dot(dir, col0), dot(dir, col1), dot(dir, col2)) / scale;
    // ---- 局部 AABB slab（内联）----
    var tl_enter = 0.0;
    var tl_exit = t_hi_cap;
    if (abs(rd.x) < 1e-30) {
      if (ro.x < l_min.x || ro.x > l_max.x) { return miss; }
    } else {
      let ta = (l_min.x - ro.x) / rd.x;
      let tb = (l_max.x - ro.x) / rd.x;
      tl_enter = max(tl_enter, min(ta, tb));
      tl_exit = min(tl_exit, max(ta, tb));
    }
    if (abs(rd.y) < 1e-30) {
      if (ro.y < l_min.y || ro.y > l_max.y) { return miss; }
    } else {
      let ta = (l_min.y - ro.y) / rd.y;
      let tb = (l_max.y - ro.y) / rd.y;
      tl_enter = max(tl_enter, min(ta, tb));
      tl_exit = min(tl_exit, max(ta, tb));
    }
    if (abs(rd.z) < 1e-30) {
      if (ro.z < l_min.z || ro.z > l_max.z) { return miss; }
    } else {
      let ta = (l_min.z - ro.z) / rd.z;
      let tb = (l_max.z - ro.z) / rd.z;
      tl_enter = max(tl_enter, min(ta, tb));
      tl_exit = min(tl_exit, max(ta, tb));
    }
    if (tl_exit < max(tl_enter, 0.0)) { return miss; }
    tl0 = max(tl_enter, 0.0);
    tl1 = min(tl_exit, t_hi_cap);
  }
  if (tl1 <= tl0) { return miss; }
  // t_min = 3×3 邻域低分辨率射线的最小命中距离，跳过 [tl0, t_min) 的空空间 traversal（保守下界）。
  tl0 = max(tl0, t_min);
  if (tl1 <= tl0) { return miss; }
  // debug_mode.z == 2（诊断）：跳过 chunk 步进层。
  if (view_u.debug_mode.z > 1.5) { return miss; }
  // ---- chunk 间 A&W（256³ 一格）+ chunk 内层次 mask DDA ----
  var sign_v = vec3<i32>(1i);
  sign_v = select(sign_v, vec3<i32>(-1i), rd < vec3<f32>(0.0));
  // 方向位掩码 LUT：一次查表拿到 side（boundary 基址）与三轴步进 face id。
  let dir_mask = u32(sign_v.x >= 0) | (u32(sign_v.y >= 0) << 1u) | (u32(sign_v.z >= 0) << 2u);
  let dir_word = DIR_LUT[dir_mask];
  let side = vec3<f32>(
    f32(dir_word & 1u),
    f32((dir_word >> 1u) & 1u),
    f32((dir_word >> 2u) & 1u)
  );
  let face_x = (dir_word >> 8u) & 0xFFu;
  let face_y = (dir_word >> 16u) & 0xFFu;
  let face_z = (dir_word >> 24u) & 0xFFu;
  let grid_inv_rd = 1.0 / rd;
  let grid_axis_on = abs(rd) > vec3<f32>(1e-30);
  let delta = select(vec3<f32>(1e+30), abs(grid_inv_rd), grid_axis_on);
  let delta_c = delta * f32(CHUNK_SIZE);
  let start = ro + rd * tl0;
  var ci = vec3<i32>(floor(start / f32(CHUNK_SIZE)));
  // tmax_c：到下一 chunk 边界的 t（ro 系绝对）
  let bnd_c = (vec3<f32>(ci) + side) * f32(CHUNK_SIZE);
  var tmax_c = select(vec3<f32>(1e+30), (bnd_c - ro) * grid_inv_rd, grid_axis_on);
  tmax_c = max(tmax_c, vec3<f32>(tl0));
  var t_enter_c = tl0;
  // entry_face = 跨入当前 chunk 的面（逆射线方向的主轴面）；首 chunk 用它兜底（相机贴面/UB）。
  var entry_face: u32 = 4u;
  let ad = abs(rd);
  if (ad.x >= max(ad.y, ad.z)) {
    entry_face = face_x;
  } else if (ad.y >= ad.z) {
    entry_face = face_y;
  } else {
    entry_face = face_z;
  }
  for (var step_c: u32 = 0u; step_c < max_chunk_steps; step_c = step_c + 1u) {
    let t_exit_c = min(tmax_c.x, min(tmax_c.y, tmax_c.z));
    let t1 = min(t_exit_c, tl1);
    // ---- chunk 窗口定位（窗口外 / entry==0 = 空气）----
    var air_chunk = true;
    let rel = ci - index_origin;
    if (all(rel >= vec3<i32>(0)) && all(vec3<u32>(rel) < index_dims)) {
      let rel_u = vec3<u32>(rel);
      let index_addr = tree_base + rel_u.x + rel_u.y * CHUNK_INDEX_CAP
        + rel_u.z * (CHUNK_INDEX_CAP * CHUNK_INDEX_CAP);
      let entry = b_struct[index_addr];
      if (entry != 0u) {
        air_chunk = false;
        let chunk_base = tree_base + entry - 1u;
        let chunk_min = vec3<f32>(ci) * f32(CHUNK_SIZE);
        // chunk 内遍历从 max(chunk 入口, t_min) 开始，跳过 t_min 前的空空间
        let t0c = max(t_enter_c, t_min);
        if (penetrate) {
          let h = trace_chunk(chunk_base, chunk_min, ro, rd, sign_v,
                              t0c, t1, entry_face, depth_cap, mc);
          if (h.hit) { return grid_hit(h, ci, col0, col1, col2, obj_id); }
        } else {
          let h = trace_chunk_opaque(chunk_base, chunk_min, ro, rd, sign_v,
                                     t0c, t1, entry_face, depth_cap);
          if (h.hit) { return grid_hit(h, ci, col0, col1, col2, obj_id); }
        }
      }
    }
    // 空气 chunk：介质段在此终止 ⇒ 出射面落在 chunk 边界（法线 = 跨入本 chunk 的面）。
    if (penetrate && air_chunk && medium_exit(mc)) {
      let nl = face_normal_from_index(entry_face);
      let nw = normalize(nl.x * col0 + nl.y * col1 + nl.z * col2);
      return UnifiedHit(true, t_enter_c, (*mc).last_pal, nw, entry_face, vec3<i32>(0), obj_id);
    }
    if (t_exit_c >= tl1) { break; }
    if (tmax_c.x <= tmax_c.y && tmax_c.x <= tmax_c.z) {
      t_enter_c = tmax_c.x; tmax_c.x = tmax_c.x + delta_c.x; ci.x = ci.x + sign_v.x;
      entry_face = face_x;
    } else if (tmax_c.y <= tmax_c.z) {
      t_enter_c = tmax_c.y; tmax_c.y = tmax_c.y + delta_c.y; ci.y = ci.y + sign_v.y;
      entry_face = face_y;
    } else {
      t_enter_c = tmax_c.z; tmax_c.z = tmax_c.z + delta_c.z; ci.z = ci.z + sign_v.z;
      entry_face = face_z;
    }
  }
  // 走完本 grid 时仍在介质段内 ⇒ 出射面落在 grid 出口（法线 = 最后一次跨入的面）。
  if (penetrate && medium_exit(mc)) {
    let nl = face_normal_from_index(entry_face);
    let nw = normalize(nl.x * col0 + nl.y * col1 + nl.z * col2);
    return UnifiedHit(true, tl1, (*mc).last_pal, nw, entry_face, vec3<i32>(0), obj_id);
  }
  return miss;
}

// 从 `grid_descs[idx]` 解码统一 `Grid`（主世界 idx=0 = identity 变换，物体 idx≥1 = 任意变换）。
// chunk 步数上限由窗口 dims 估算（曼哈顿对角 ×3 + 余量）；物体 dims=(1,1,1)。
fn make_grid(idx: u32) -> Grid {
  let d = grid_descs[idx];
  let origin = vec3<i32>(d.index_origin_x, d.index_origin_y, d.index_origin_z);
  let dims = vec3<i32>(i32(d.index_dims_x), i32(d.index_dims_y), i32(d.index_dims_z));
  let chunk_budget = u32(dims.x + dims.y + dims.z) * 3u + 16u;
  // 主世界（idx=0，identity）：局部 AABB = 窗口 AABB（origin 可为负）；物体 = 单 chunk 局部 [0,256]³。
  let is_world = idx == 0u;
  let l_mn = select(vec3<f32>(0.0), d.aabb_min.xyz, is_world);
  let l_mx = select(vec3<f32>(f32(CHUNK_SIZE)), d.aabb_max.xyz, is_world);
  // 变换段：主世界是 identity（Rust `VolumeTransform::IDENTITY`；`trace_grid` 的 `obj_id < 0` 分支
  // 早已依赖这条 —— 它把局部坐标直取世界坐标）⇒ 不读 `rot0..2` / `pos_scale` 这 4 个 vec4，
  // 直接给常量。这样下游的 `obj_id < 0` 分支（ro/rd 直取、`voxel_normal` 的 mat-vec 短路）
  // 也能整段常量折叠掉。
  var col0 = vec3<f32>(1.0, 0.0, 0.0);
  var col1 = vec3<f32>(0.0, 1.0, 0.0);
  var col2 = vec3<f32>(0.0, 0.0, 1.0);
  var pos = vec3<f32>(0.0);
  var scale = 1.0;
  if (!is_world) {
    col0 = d.rot0.xyz;      // 旋转矩阵列
    col1 = d.rot1.xyz;
    col2 = d.rot2.xyz;
    pos = d.pos_scale.xyz;  // 物体位置
    scale = d.pos_scale.w;
  }
  return Grid(
    d.aabb_min.xyz, d.aabb_max.xyz,       // 世界 AABB
    l_mn, l_mx,                            // 局部 AABB
    chunk_budget,
    col0, col1, col2,                      // 旋转矩阵列
    pos, scale,                            // pos + scale
    d.tree_base, d.palette_base,           // 数据源基址
    origin, vec3<u32>(d.index_dims_x, d.index_dims_y, d.index_dims_z),
    // obj_id 约定（镜像 Rust）：idx=0 主世界 → -1；idx≥1 物体 → idx-1。
    select(-1i, i32(idx) - 1i, idx > 0u),
  );
}
