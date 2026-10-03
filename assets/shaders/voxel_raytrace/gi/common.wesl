// GI 路径的**共用工具**：几何面键、位打包、RNG、余弦半球采样。
//
// 被三处共用 ⇒ 单独成模块，任何一侧都不许持有第二份实现（口径分叉会让画面跳变）：
//   · `gi/screen.wesl`  —— ReSTIR 的复用判据（主命中面键 + 着色法线）；
//   · `gi/ray.wesl`     —— 射线命中后算键（二次顶点缓存与复用判据都吃它）；
//   · `gi/denoise.wesl` —— 降噪导引的键与法线打包（与复用判据必须是同一套编码）。

import package::bindings::{face_slots, face_slots_flat, gi_region_rev, gi_sec_slots};
// 面键的编码原点取**该体素所属 volume** 的量化窗口原点（`gi_key_org`）⇒ 需要 `grid_descs` 与 `CHUNK_SIZE`。
import package::bindings::{grid_descs};
import package::common::{CHUNK_SIZE};

/// **面键可编码的坐标半径**（体素，可正负）：16 bit/轴 ⇒ 32768 体素 = 655 m（50 体素/米，
/// 体素没有固定物理边长，见 `README.md` §4）。
///
/// CONSTRAINT: 它量的坐标是**相对编码原点**（`gi_key_org`）的，故这个半径只需装下"窗口 + 次级射程"，
/// 而**不必**覆盖整个世界（MC 地图 ±5.6 km、L3 远场 ±10.5 km）—— 原点一减就把它折掉了。
///
/// CONSTRAINT: 越界不是"理论正确性"问题，是热路径的开关 —— `gi_key_ok` 判假时，时域复用 / 空间复用 /
/// 逐面认领**全部静默退化成"关"**，候选数按"无历史"那条贵路径发（8 条/像素；实测 `gate_gi`
/// 16 ms vs 关掉时 1.1 ms）。
///
/// REF: 加宽位域（16 → 20 bit/轴）试过一次，能编解码但对齐不齐，实测引入"玻璃变不透明 + 稳定噪点"
/// 的回归，已回退 ⇒ 改走"减去编码原点"这条路（`gi_key_org`）。
const GI_KEY_LIMIT: i32 = 32768;

/// **面键编码原点的量化步长**（体素）：8192 体素 = 163.84 m（50 体素/米）。
/// 它的唯一作用是把"窗口跟着相机平移"变成"基准每 163.84 m 才动一次"（见 `gi_key_org`）。
/// 取值的约束：`窗口边长 + 2·Q + GI 射程` 必须小于 [`GI_KEY_LIMIT`]，见 `gi_key_ok`。
/// Rust 侧 `gi::KEY_ORG_Q` 必须与它逐字相等（单测 `key_org_q_matches_wesl` 解析本行比对）。
const GI_KEY_ORG_Q: u32 = 8192u;

/// 面键的**编码原点**（世界体素）：主世界窗口的原点。三个键函数一律先减去它。
///
/// WHY 必须减：键只有 16 bit/轴（±32768 体素 = ±655 m），而世界远超这个半径（MC 地图 ±5.6 km、
/// L3 远场 ±10.5 km）⇒ 不减原点时 `gi_key_ok` 对**几乎每个像素**都返回假，于是时域复用 / 空间复用 /
/// 逐面认领**静默全关**：现象是**永不收敛 + 逐帧闪烁、静止也一样**，且与任何 GI 档位无关
/// （候选数退化成"无历史"那条贵路径的 8 条/像素，实测 `gate_gi` 16 ms vs 关掉时 1.1 ms）。
///
/// 取**窗口原点**（而不是逐帧相机位置）：所需范围只剩窗口半宽（32 chunk = 8192 体素）+ 次级射线行程，
/// 稳稳落在 ±32768 内，不必加宽位域。窗口原点跟着相机平移，故再按 [`GI_KEY_ORG_Q`] 量化 ——
/// 基准变的那一帧由 Rust 侧推 `wide_rev`（`prepare_gi`）把整链作废 ⇒ 参与比较的键必出自同一基准。
///
/// CONSTRAINT: 原点取**该体素所属 volume** 的窗口（`obj`：−1 = 主世界 ⇒ `grid_descs[0]`，≥0 = 物体 /
/// 远场级）。不能统一用主世界原点 —— 物体 / 远场级的 `h.voxel` 是它们**自己的网格局部坐标**，与主世界
/// 原点不同尺度，相减后必然越界（实测残下 1.4% 的 texel 因此拿不到 `pk_ok`，表现为远景物上的闪烁点）。
///
/// CONSTRAINT: 原点必须**量化**到 [`GI_KEY_ORG_Q`] 的整数倍后再用。流式窗口是**相机居中**的
/// （`stream_window`），相机每跨一个 chunk 它就平移一格 ⇒ 原点若逐格变，键的编码基准也跟着逐格变，
/// 于是每跨一格都要把整链作废（Rust 侧推 `wide_rev`）。正常移动即 ~15 次/s，静止外的画面根本没机会
/// 累积 ⇒ 一动就闪。量化后基准只在相机跨过 `GI_KEY_ORG_Q` 边界时变，作废频率降两个数量级。
/// 代价是键的取值域整体多出一个 `GI_KEY_ORG_Q` 的余量（`v` 落在 [0, 窗口边长 + Q)），仍远小于
/// [`GI_KEY_LIMIT`]。
fn gi_key_org(obj: i32) -> vec3<i32> {
  let d = grid_descs[u32(obj + 1)];
  let o = vec3<i32>(d.index_origin_x, d.index_origin_y, d.index_origin_z)
    * vec3<i32>(i32(CHUNK_SIZE));
  // 整除按截断（WGSL/i32 `/`），不要求 floor：只要"同一个 `o` 恒得同一个基准"即可，
  // 截断给出的余数落在 (−Q, Q) ⇒ 上界那一句同样成立。
  let q = vec3<i32>(i32(GI_KEY_ORG_Q));
  return (o / q) * q;
}

fn gi_key_ok(v_world: vec3<i32>, obj: i32) -> bool {
  let v = v_world - gi_key_org(obj);
  return all(v > vec3<i32>(-GI_KEY_LIMIT)) && all(v < vec3<i32>(GI_KEY_LIMIT));
}

/// 键 word 0：x、y 各 16 bit（+32768 偏移，**相对编码原点**）。
fn gi_key0(v_world: vec3<i32>, obj: i32) -> u32 {
  let v = v_world - gi_key_org(obj);
  return (u32(v.x + 32768) & 0xFFFFu) | ((u32(v.y + 32768) & 0xFFFFu) << 16u);
}

/// 键 word 1：z 16 bit（0..15）、面 3 bit（16..18）、物体 id 13 bit（19..31，+1 存，0 = 主世界）。
/// **面进键**是薄板 / 墙缝不漏光的根本：同一体素的两个面是两个不同的键（见 `gi/screen.wesl`）。
fn gi_key1(v_world: vec3<i32>, face: u32, obj: i32) -> u32 {
  let v = v_world - gi_key_org(obj);
  return (u32(v.z + 32768) & 0xFFFFu)
    | ((face & 7u) << 16u)
    | ((u32(obj + 1) & 0x1FFFu) << 19u);
}

/// `gi_key0` 的逆：从键还原**世界**体素坐标（面号 = `(k1 >> 16) & 7`，物体 = `(k1 >> 19) - 1`）。
///
/// CONSTRAINT: 必须加回 `gi_key_org(obj)` —— 键里存的是**相对编码原点**的坐标，不加回来得到的是
/// "离原点多远"，把它当世界坐标用会把着色点整体挪到 `GI_KEY_ORG_Q` 的整数倍之外（≥163.84 m）：
/// 阴影射线起点、triplanar 取样点、mip LOD 的距离输入**全部**落到别的地方
/// （现象：整个场景材质斑驳 + 阴影/自发光间接光全错）。
fn gi_unkey_voxel(k0: u32, k1: u32, obj: i32) -> vec3<i32> {
  return vec3<i32>(i32(k0 & 0xFFFFu) - 32768,
                   i32((k0 >> 16u) & 0xFFFFu) - 32768,
                   i32(k1 & 0xFFFFu) - 32768) + gi_key_org(obj);
}

// ============================================================================
// 区域修订表（`gi_region_rev`）：二次顶点缓存的**局部失效**
// ============================================================================
// 解决什么：`gi_sec_slots` 的值（太阳 NEE 的遮挡 + 天光 + 自发光）依赖「别的几何」，旧口径是
// **任何上传都作废整表**（`ShadeKey.geom` = world_rev）—— 流式世界里每帧成立 ⇒ 缓存等于不存在
// （见 `gi.rs::ShadeKey` 的 CONSTRAINT）。现在改成：**只有该面附近真的变过，这个面的槽才失效**。
//
// 布局：窗口（64³ chunk）按 8³ chunk 切区域；表 = 16³ = 4096 个 u32，**基址 = 该 volume 的量化键
// 原点**（`gi_key_org`，与面键同源 ⇒ 基准与键永远一致，一起被 `wide_rev` 作废）。
// Rust 侧每次上传把覆盖到的区域**及其 ±REACH 邻域** +1（膨胀在写侧做 ⇒ 读侧只付一次 load，
// REACH = ceil(太阳阴影行程 12 chunk / 8) = 2，定义在 `gi.rs::REGION_REACH`）。
// 取值 0 = 表未就绪 / 非主世界 ⇒ 不参与失效（退化为旧行为）。
const GI_REGION_CHUNKS: u32 = 8u;
const GI_REGION_TABLE: u32 = 16u;

/// 命中面 `(v_world, obj)` 的**区域修订号**（0 = 无 / 表未就绪 / 非主世界）。
///
/// CONSTRAINT: 与 Rust 侧 `gi.rs::region_cell()` 必须**逐字同源**（常量有单测
/// `region_consts_match_wesl` 解析本文件比对）：同样的"相对键原点 → chunk → 区域 → 钳制"。
/// 只用 u32 数学（键原点保证 `v − org ≥ 0`：org ≤ 窗口原点 ≤ 窗口内任何 chunk），
/// 所以除法即 floor、与 Rust 的 `/` 同口径；越界一律钳到表边界（保守：多失效、不漏失效）。
fn gi_local_rev(v_world: vec3<i32>, obj: i32) -> u32 {
  if (obj != -1) { return 0u; }
  if (arrayLength(&gi_region_rev) < GI_REGION_TABLE * GI_REGION_TABLE * GI_REGION_TABLE) {
    return 0u;
  }
  let rel = max(v_world - gi_key_org(obj), vec3<i32>(0));
  let chunk = vec3<u32>(rel) / vec3<u32>(CHUNK_SIZE);
  let cell = min(chunk / vec3<u32>(GI_REGION_CHUNKS), vec3<u32>(GI_REGION_TABLE - 1u));
  return gi_region_rev[cell.x + GI_REGION_TABLE * (cell.y + GI_REGION_TABLE * cell.z)];
}

/// 32 位整数混合（RNG 与哈希共用）。
/// 只用 `+ / * / >>`：本工程的 WESL 解析器不接受 `^`（异或）运算符，故按「乘-加-移位」混合。
fn gi_mix(x: u32) -> u32 {
  var h = x * 747796405u + 2891336453u;
  h = ((h >> ((h >> 28u) + 4u)) + h) * 277803737u;
  return (h >> 22u) + h;
}

/// [0,1)²。同一个 seed 恒定 ⇒ 每个像素的候选方向序列可复现（不逐帧抖动）。
fn gi_rand2(seed: u32) -> vec2<f32> {
  let a = gi_mix(seed);
  let b = gi_mix(a + 0x9E3779B9u);
  return vec2<f32>(f32(a), f32(b)) * (1.0 / 4294967296.0);
}

/// 世界法线 / 方向 → 一个 word（八面体编码 + f16x2）。单位向量解码误差 < 1°。
fn gi_pack_dir(n: vec3<f32>) -> u32 {
  let l = abs(n.x) + abs(n.y) + abs(n.z);
  var o = n.xy / max(l, 1e-6);
  if (n.z < 0.0) {
    o = (vec2<f32>(1.0) - abs(o.yx)) * select(vec2<f32>(-1.0), vec2<f32>(1.0), o >= vec2<f32>(0.0));
  }
  return pack2x16float(o);
}

fn gi_unpack_dir(p: u32) -> vec3<f32> {
  let o = unpack2x16float(p);
  var n = vec3<f32>(o.x, o.y, 1.0 - abs(o.x) - abs(o.y));
  if (n.z < 0.0) {
    n = vec3<f32>(
      (1.0 - abs(n.y)) * select(-1.0, 1.0, n.x >= 0.0),
      (1.0 - abs(n.x)) * select(-1.0, 1.0, n.y >= 0.0),
      n.z,
    );
  }
  return normalize(n);
}

/// 面法线的余弦加权半球方向：`u` = [0,1)²，`dir` 恒在 `n` 半球内。
/// 余弦 pdf 下 `f/p = albedo·L·cosθ·π/cosθ = π·L`（不含 albedo）⇒ RIS 权重取 `π·L`。
fn gi_cos_hemisphere(n: vec3<f32>, u: vec2<f32>) -> vec3<f32> {
  let r = sqrt(u.x);
  let phi = 6.2831853 * u.y;
  let axis = select(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 0.0), abs(n.z) > 0.9);
  let t = normalize(cross(axis, n));
  let b = cross(n, t);
  return normalize(t * (r * cos(phi)) + b * (r * sin(phi)) + n * sqrt(max(1.0 - u.x, 0.0)));
}

// ============================================================================
// 帧内逐面去重表（"同一个可见体素面 → 同一个槽"）—— `face_slots` 的唯一映射入口
// ============================================================================
// 要解决的问题：本工程的着色量子是**一个可见体素面**（体素 + 它被看见的那个面）—— 同一个体素的不同
// 可见面颜色不同（法线不同），同一个面内颜色相同。而主 pass 里除主射线之外的全部输入本来就只由这个
// 量子决定（材质 / 着色法线 / 视向量 / 阴影可见性 / 反射的起点与种子 / 玻璃状态机 —— 见
// `main.wesl` 的 `shade_face`），却被每个像素重算了一遍。一个面在屏幕上覆盖几百~几千像素时，
// 这份重复就是白付的。
//
// **量子 = `gi_key0`/`gi_key1`（体素坐标 + 面号 + 物体）**。面号必须进键：同一个体素最多 3 个可见面、
// 各面法线不同 ⇒ 着色不同。这是"体素被看见的那个面"，**不是**"按平面合并"（后者会把一整面墙的相邻
// 体素并成一个值 —— 那是降噪 `gi_same_surface` 的行为，与这里的粒度无关）。
//
// **为什么是哈希表，而不是"把结果写在体素中心投影的那个像素里"**（更省内存的直觉做法）：
//   ① 同一个体素最多 3 个可见面，而三个面共享**同一个体素中心** ⇒ 中心投影必然落在同一个像素
//      （只有射线真正进入的那个面能占住它）⇒ 至少 2 个面拿不到槽位，且是**必然冲突**不是偶发；
//   ② 体素中心的投影可能落在**被更近体素遮挡**的像素上（墙被柱子挡掉的那一块）⇒ 写入会覆盖
//      别人的值，自己的像素也读不回自己的值；
//   ③ 相机落在体素格线上沿轴看时，整列体素中心的投影**逐位重合**（引擎相机以体素为单位，这是常态）。
//   ⇒ 任何"用屏幕像素当存储位置"的方案都缺一个"不同量子 → 不同位置"的前提；哈希表只做
//      "键 → 槽"，撞键（槽被别的键占）退化成**该面逐像素内联着色** ⇒ 只损失收益、绝不损失正确性。
// ---- ① 逐面合并（本表同时承担"面级 reservoir"的角色）----
// 同一个可见体素面的**全部 texel 的 GI 射线起点/法线与外推量完全相同**（`gi_ss_main` 的 `org`），
// 只有种子不同 ⇒ 它们的候选是**同一个积分的 i.i.d. 样本** ⇒ 把它们各自的估计先按面平均，
// 方差 ÷ texel 数（这就是"面越大越干净"的来源，也是下面 `FACE_W_RAW*` 那对字段存在的理由）。
// 链路三段：
//   ① `gi_ss_main` 末尾把本 texel 的估计**定点累加**进 `FACE_W_RAW*`（与认领同一张表、同一个键）；
//   ② 新 pass `gi_face_flatten`（`main.wesl`）把该面的均值写回 `gi_out` —— 只有"被认领、且键逐位相等"
//      的 texel 会被改写，撞键 / 未认领的面照旧逐 texel（只损失收益，不损失正确性）；
//   ③ 降噪链拿到的输入因此是"整个面一个值"，时域累积的方差驱动权重与 atrous 的 range 权重都作用在
//      更干净的信号上（面内不再有逐 texel 的独立噪声）。
// 与 `FACE_W_GI*` 的分工：那对装**降噪之后**的值（`dda_face_accum` 写，供 `dda_face_main` 取逐面均值，
// 因为时域/atrous 之后面内仍可能有细微差异），这对装**降噪之前**的估计 ⇒ 两对互不覆盖。

/// 每槽字数。0 = 认领标志（CAS 目标）、1/2 = 面键、3 = 认领者 texel 下标、4 = palette 条目号、
/// 5..7 = **着色结果**（线性 rgb，曝光前；`dda_main` 与 `dda_face_main` 共用 `shade_face`）、
/// 8 = GI 累加计数、9..11 = GI 累加和（**降噪后**的值，定点，见 `GI_ACC_*`；`dda_face_accum` 写）、
/// 12 = 本帧**合并计数**、13..15 = 本帧合并和（**降噪前**的原始估计，定点）—— 见下面 ① 那段。
const FACE_WORDS: u32 = 16u;
const FACE_W_FLAG: u32 = 0u;
const FACE_W_K0: u32 = 1u;
const FACE_W_K1: u32 = 2u;
const FACE_W_TEXEL: u32 = 3u;
const FACE_W_PAL: u32 = 4u;
const FACE_W_COL: u32 = 5u;
const FACE_W_GI_N: u32 = 8u;
const FACE_W_GI: u32 = 9u;
/// ① **逐面合并**的两个字段（同一张表的另一对累加器，语义与上面那对**不重叠**）：
/// `12` = 本帧把估计并进来的 texel 数、`13..15` = 它们的估计和（定点）。
/// 为什么要有第二对：`FACE_W_GI*` 装的是**降噪之后**的值（供 `dda_face_main` 取逐面均值），
/// 而这里装的是**降噪之前**的原始估计 —— 它要在降噪链**之前**按面合并（`gi_face_flatten`），
/// 好让时域/atrous 的输入从一开始就是"整个面一个值"。
const FACE_W_RAW_N: u32 = 12u;
const FACE_W_RAW: u32 = 13u;

/// **逐面 GI 均值**的定点参数（WGSL 默认没有 float 原子 ⇒ 只能加整数）。
/// 每个 GI texel 把自己的 GI 值（`gi_tex.rgb`，已含增益）加上去，面着色时除以计数取平均。
/// 为什么必须做这一步：同一个面的所有 texel 的 GI 射线**起点都是该体素中心**（见 `gi_ss_main` 的
/// `org`）⇒ 它们是对**同一个积分**的独立估计 ⇒ 取平均是**无偏**的纯降噪（方差 ÷ 参与数）。
/// 不取平均而只取某一个 texel（认领者）的值时，该 texel 的降噪残差会被整面一起放大成
/// **相干闪烁**（原来逐像素各自独立、被眼睛平均掉），面越大越明显、关降噪时是整面雪块。
///
/// **`SCALE` 的下限由「暗处」定**（不是由亮度上限定）：8 位 sRGB 输出在近 0 处的斜率
/// `E(L) ≈ 12.92·L` ⇒ 量化步长 `1/SCALE` 在屏幕上的台阶 = `255 × 12.92 / SCALE` 个码值：
///   · 256  ⇒ **约 13 个码值/级**。室内 GI 低到 1e-3 量级时整面墙只剩 2~3 个可取值，
///     等值线变成一圈圈**彩虹状色带**（三通道各自进位 ⇒ 每一圈的色相都不同）；
///   · 4096 ⇒ 约 0.8 个码值/级，低于输出自身的 1 LSB ⇒ 不可见。
/// **上限由溢出定**：单条贡献 ≤ `CLAMP × SCALE`、参与数封顶 `MAX_N = 1024`
/// ⇒ 4096 × 255 × 1024 = 1.07e9（占 u32 的 25%；再翻一档 16384 就是 4.28e9，会溢出）。
/// 计数用 CAS 预留 ⇒ 和不会被并发冲过（见 `dda_face_accum`）。
/// ** ① 的原始估计合并（`FACE_W_RAW*`）用同一组参数**：它加的是"已经乘好增益、除好 π"的
/// `gi` 值（与 `gi_out.rgb` 同量纲），量级同样被 `CLAMP` 之外的部分钳住、参与数同样封顶 ⇒ 同一份溢出账。
const GI_ACC_SCALE: f32 = 4096.0;
const GI_ACC_CLAMP: f32 = 255.0;
const GI_ACC_MAX_N: u32 = 1024u;

/// 槽数（Rust 侧按 GI 网格像素数给；`arrayLength` 数的是 u32 元素）。
fn face_slot_count() -> u32 {
  return arrayLength(&face_slots) / FACE_WORDS;
}

/// 面键 → 槽下标。`ok = false` = 表未就绪（占位缓冲）/ 首帧 ⇒ 调用方一律走内联着色。
fn face_slot_of(k0: u32, k1: u32, ok: ptr<function, bool>) -> u32 {
  let n = face_slot_count();
  *ok = n >= 1u;
  if (!*ok) { return 0u; }
  // 槽数是 2 的幂 ⇒ 用位与取模（`gi_mix` 是本工程唯一的哈希，见文件头）。
  return gi_mix(k0 + gi_mix(k1 * 0x9E3779B1u)) & (n - 1u);
}

/// **双探针**：基槽之后的第二个候选槽（掩码回绕）。
///
/// WHY：哈希撞键时，从前那个面**整帧**退回"逐像素内联 `shade_face`"（含它那条阴影射线）——
/// 而实测（2026-09-28，MC 城市 / 1/2 档）**撞键的面占全屏 4.6% 的像素**，是一整块可回收的代价。
/// 认领侧（`gi_main`）在基槽 CAS 失败时**再试这一格**；读取侧（`dda_main` / `dda_face_main`）
/// 按"基槽 → 第二个槽"的顺序找**键逐位相等**的那一格。判据一个字没变（仍是同一个键、同一套
/// 校验），只是给撞键的面多一次落脚机会 ⇒ 那一小部分像素从"内联整条 `shade_face`"变成
/// "多读一条缓存行"。命中的绝大多数（基槽即中）**一次探针都不多付**。
fn face_slot_next(base: u32) -> u32 {
  return (base + 1u) & (face_slot_count() - 1u);
}

/// 槽里是否**确实**存放着这个键（认领标志置位 + 两个键 word 逐位相等）。
/// 三个字并行读出再判（不靠 `&&` 短路去串行化访存）—— 槽是 16 字 = 恰好一条 64B 缓存行。
fn face_slot_matches(slot: u32, k0: u32, k1: u32) -> bool {
  let fe = slot * FACE_WORDS;
  let flag = atomicLoad(&face_slots[fe + FACE_W_FLAG]);
  let s0 = atomicLoad(&face_slots[fe + FACE_W_K0]);
  let s1 = atomicLoad(&face_slots[fe + FACE_W_K1]);
  return flag != 0u && s0 == k0 && s1 == k1;
}

/// **面键 → 确实被本键认领的那个槽**（双探针：基槽 → 第二格；读 `@group(5) @binding(8)` 那份声明）。
/// `ok = false` = 表未就绪 / 两个候选槽都没有这个键（未认领 / 撞键）。
///
/// ⚠️ **所有读侧都必须走本函数（或 `_flat` 版）**：认领（`gi_main`）在基槽抢不到时会落到第二格，
/// 只探基槽的读者会把那些面当成"没被认领" ⇒ 逐面 GI 合并（`gi_ss_main` 的 `FACE_W_RAW*`）、
/// 逐面均值累加（`dda_face_accum`）、逐面平均（`gi_face_flatten`）**全部静默漏掉那个面**
/// ⇒ 症状：那些面拿不到 GI（`cov = 0`，只剩天光兜底）且降噪输入没有面内平均 ⇒ 斑驳噪点。
/// 命中基槽的绝大多数（基槽即中）**一次探针都不多付**。
fn face_slot_find(k0: u32, k1: u32, ok: ptr<function, bool>) -> u32 {
  var table_ok = false;
  let base = face_slot_of(k0, k1, &table_ok);
  *ok = false;
  if (!table_ok) { return 0u; }
  if (face_slot_matches(base, k0, k1)) {
    *ok = true;
    return base;
  }
  let second = face_slot_next(base);
  if (face_slot_matches(second, k0, k1)) {
    *ok = true;
    return second;
  }
  return 0u;
}

// ---- `_flat` 版：读 `@group(0) @binding(18)` 那份声明（给只有瘦 group(0) 的 pass 用）----
// 必须与上面那组**同一张表、同一个槽数、同一套判据**（`face_slot_of_flat` 的注释解释了为什么
// 不能混用：混用会让槽数不同 ⇒ 哈希取模不同 ⇒ 查到别的槽）。
/// 槽数（`face_slots_flat` 声明的 `arrayLength`）。
fn face_slot_count_flat() -> u32 {
  return arrayLength(&face_slots_flat) / FACE_WORDS;
}

/// 双探针的第二格（`_flat` 版）。
fn face_slot_next_flat(base: u32) -> u32 {
  return (base + 1u) & (face_slot_count_flat() - 1u);
}

/// 槽里是否确实存放着这个键（`_flat` 版）。
fn face_slot_matches_flat(slot: u32, k0: u32, k1: u32) -> bool {
  let fe = slot * FACE_WORDS;
  let flag = atomicLoad(&face_slots_flat[fe + FACE_W_FLAG]);
  let s0 = atomicLoad(&face_slots_flat[fe + FACE_W_K0]);
  let s1 = atomicLoad(&face_slots_flat[fe + FACE_W_K1]);
  return flag != 0u && s0 == k0 && s1 == k1;
}

/// 面键 → 确实被本键认领的那个槽（`_flat` 版；语义与 [`face_slot_find`] 逐字相同）。
fn face_slot_find_flat(k0: u32, k1: u32, ok: ptr<function, bool>) -> u32 {
  var table_ok = false;
  let base = face_slot_of_flat(k0, k1, &table_ok);
  *ok = false;
  if (!table_ok) { return 0u; }
  if (face_slot_matches_flat(base, k0, k1)) {
    *ok = true;
    return base;
  }
  let second = face_slot_next_flat(base);
  if (face_slot_matches_flat(second, k0, k1)) {
    *ok = true;
    return second;
  }
  return 0u;
}

/// **面键 → 槽里的着色结果 + 材质条目号**（`face_slots` 的唯一读入口；认领在 `gi_main`、填色在
/// `dda_face_main`；`pal` 由认领侧写）。命中条件 = 认领标志置位 + **两个键 word 逐位相等**。
/// `ok = false` 时调用方一律回落（覆盖 texel 回退 / 内联着色），`pal_out` 无意义。
///
/// `pal_out` 给「覆盖 texel 回退」做**同材质**准入（见 `main.wesl::dda_main`）：那个 texel 的面与
/// 本像素的面是**邻居**（不同体素），键相等证明不了材质相同 ⇒ 必须显式比 `pal`。
/// 只在同一 `obj` 内比较：`pal` 是**该 volume** 调色板的条目号，跨 volume 同号不同材质
/// （准入已由 `gi_den_same_plane` 的 `obj` 那一半保证）。
fn face_slot_color(k0: u32, k1: u32, pal_out: ptr<function, u32>, ok: ptr<function, bool>) -> vec3<f32> {
  *pal_out = 0u;
  let slot = face_slot_find(k0, k1, ok);
  if (!*ok) { return vec3<f32>(0.0); }
  let fe = slot * FACE_WORDS;
  *pal_out = atomicLoad(&face_slots[fe + FACE_W_PAL]);
  let c0 = atomicLoad(&face_slots[fe + FACE_W_COL]);
  let c1 = atomicLoad(&face_slots[fe + FACE_W_COL + 1u]);
  let c2 = atomicLoad(&face_slots[fe + FACE_W_COL + 2u]);
  return vec3<f32>(bitcast<f32>(c0), bitcast<f32>(c1), bitcast<f32>(c2));
}

// ============================================================================
// 二次顶点按面缓存（`gi_sec_slots`）—— 只服务 GI 射线的命中着色（`gi/ray.wesl`）
// ============================================================================
// 为什么能按面缓存：二次顶点着色里**除射线自身的介质透过率 `tint` 以外**的全部输入都只由"命中的
// 那个可见体素面"决定 —— 材质取值（`fetch_material` 内部把采样点与 mip LOD 距离都量化到体素中心）、
// 着色法线（`voxel_normal` 是 (voxel, face) 的纯函数）、视向量（锚点 = 体素中心）、太阳 NEE 的起点
// （体素中心沿面法线外推）—— 与主 pass 的 `p_voxel` 是同口径。
// 于是一条面被成千上万条 GI 射线打到时，着色只算一次；`tint` 留在调用方按**逐射线**乘。
//
// 与 `face_slots` 的分工：那张表按"**被看见的**面"由 `gi_main` 认领，存主 pass 的最终颜色与逐面
// GI 均值；这张表按"**被 GI 射线打到的**面"惰性填充（同一个面可能完全不可见），存二次顶点的入射
// 辐亮度。两者键口径相同（`gi_key0`/`gi_key1`），用途不重叠。
//
// **为什么不需要认领 / 两相**：填充者算出的值逐位相同 ⇒ 写同值 ⇒ 竞态幂等（不像主 pass 的查表那样
// 要求"先写后读"的可见性）。唯一风险窗口是"键已可见、辐亮度字尚未落地"（WGSL 没有全局 fence，
// 只有 workgroup 级的 `storageBarrier`）：那一条射线读到 0，等于少一个二次顶点样本 —— 属于随机
// 估计里的一次异常样本（每 texel 每帧 4~8 个样本，还有时域累积与降噪），不可见也不逐帧相干。
// 写入顺序刻意是「辐亮度 → 键」，配合"键不符 ⇒ 重算"的回退。
//
// ---- 跨帧持久（epoch 掩码键）：**这张表不再每帧清空** ----
// 槽里的键写成 `键 + mask(epoch, pal)`（见 `gi_sec_key_masked`），读侧用**同一个** `(epoch, pal)`
// 重算掩码再逐位比。epoch 一变，旧槽的掩码值与新期望恒不相等 ⇒ 整张表**自失效**（不需要任何清表，
// 省掉每帧的 `clear_buffer`，2K + 1/4 档那张表约 42MB/帧）。
// epoch 的构成 = `gi_face_shade` 真正读到的**全部**输入（Rust 侧 `prepare_gi` 逐项比对、变了才自增，
// 经 uniform `gi_u.seq.y` 下发；两个集合的差异写在那里，这里只说结论）：
//   · 天光色恒进；
//   · 太阳方向 / 颜色 / 强度**只在「太阳反弹」打开时**进 —— 关掉时 `sun_vis = 0`，
//     `brdf_reflected` 里整支太阳项被折掉 ⇒ 值真的与主光无关；
//   · 世界几何**只在遮挡项真的依赖它时才进**（同上，= 只有「太阳反弹」打开时）——
//     关掉时二次顶点不发阴影射线，值与**任何别的**体素无关 ⇒ **流式挂载 / 卸载、逐体素编辑
//     都不作废这张表**（它们只改别的体素，这里命中的那个面照旧）。逐面自己的材质由槽里的
//     `pal` 参与掩码覆盖（见 `gi_sec_key_masked`）。
//     CONSTRAINT: 少了"几何不进 epoch"这一条，epoch 会在流式世界里每帧 +1 ⇒ 整张表每帧自失效、
//     缓存等于不存在（2026-09-25 实测：720p/GI 1/2 档下 `gate_gi` 15.56 ms 里的大头正是这个）。
// 唯一没被覆盖的输入：`voxel_normal` 会探命中体素的 6 个邻居 ⇒ 邻居体素增删会微调二次顶点的
// 着色法线。它只由编辑 / 流式引起，且是"斜向梯度法线的一阶变化"，代价可接受（已知取舍）。
// 为什么可以跨帧留用：槽里的值是 `(命中的可见体素面, 材质, 上面那些光照量)` 的**纯函数** —— 它不含任何
// 逐像素 / 逐帧随机量（`gi_face_shade` 的采样点、法线、视向量、NEE 起点全部只由那个面决定），
// 所以同一 epoch 内任何时刻重算都逐位相同。
// 代价：撞键（两个面哈希到同一槽）从"只损失收益"变成"每帧互相覆写"（仍然只损失收益：都 miss）；
// 以及"不同面同时覆写同一槽"的撕裂窗口比原来大（原来只在首触帧可能发生）—— 读侧用**两次键校验**
// （键 → 辐亮度 → 键）把窗口收到几十个周期的量级，剩下的归入上面那类"一次异常样本"。
const GI_SEC_WORDS: u32 = 5u;
const GI_SEC_W_K0: u32 = 0u;
const GI_SEC_W_K1: u32 = 1u;
const GI_SEC_W_RAD: u32 = 2u;

/// **面键 → 槽下标**，与下面 `face_slot_of` **同一个哈希、同一张表**，但槽数取自 `@group(0)` 的
/// `face_slots_flat` 声明 —— 给 `gi_face_flatten` 这个"只有瘦 group(0)"的入口用。
///
/// ⚠️ **两个声明必须绑同一块 buffer**（Rust 侧就是这么绑的）：否则槽数不同 ⇒ 哈希取模不同 ⇒ 查到
/// 别的槽。**不要**把它换成 `face_slot_of`：那个函数读的是 `@group(5) @binding(8)` 的声明，
/// 会把 group(5) 拖进这个 pass 的 pipeline layout（实测会直接 fail：
/// `ResourceBinding { group: 5, binding: 8 } is not available in the pipeline layout`）。
fn face_slot_of_flat(k0: u32, k1: u32, ok: ptr<function, bool>) -> u32 {
  let n = arrayLength(&face_slots_flat) / FACE_WORDS;
  *ok = n >= 1u;
  if (!*ok) { return 0u; }
  return gi_mix(k0 + gi_mix(k1 * 0x9E3779B1u)) & (n - 1u);
}

/// 面键 → 二次顶点缓存的槽下标。`ok = false` = 表未就绪 ⇒ 该射线走内联着色。
/// 哈希只用**明文键**（与 epoch 无关）：同一面在同一槽里跨 epoch 复用，只是槽里的键要重新盖掩码。
fn gi_sec_slot_of(k0: u32, k1: u32, ok: ptr<function, bool>) -> u32 {
  let n = arrayLength(&gi_sec_slots) / GI_SEC_WORDS;
  *ok = n >= 1u;
  if (!*ok) { return 0u; }
  return gi_mix(k0 + gi_mix(k1 * 0x9E3779B1u)) & (n - 1u);
}

/// 槽里键的 **epoch 掩码**：`存储值 = 明文键 + 本函数`，读侧用同一个 `(epoch, pal)` 重算掩码再逐位比 ⇒
/// epoch 一变，旧槽恒不相等（整表自失效，不需要清表）。两个 word 用**不同的**混合式，避免
/// "两半同步平移"让别的键偶然对齐；`gi_mix` 是本文件现成的哈希（不另造）。
///
/// `pal` = 命中体素的 **palette 条目号**（`WorldHit::pal`），**必须进掩码**：同一个 (体素, 面, 物体)
/// 被重新刷上另一个材质时，键逐位不变（`gi_key0`/`gi_key1` 里没有材质），而值从材质来
/// ⇒ 不进掩码就会把旧材质的颜色一直挂在那个面上。它**不参与槽下标**（`gi_sec_slot_of` 只用明文键）
/// ⇒ 换材质是"同一个槽重新算"，不是"换一个槽"。
///
/// `rev` = **区域修订号**（`gi_local_rev`；太阳反弹关掉时调用方传 0）：把"该面附近几何变过"折进
/// 掩码 ⇒ 只有附近真的变过的槽才失效（局部化；见上面「区域修订表」那一段）。
/// `rev = 0` 时本函数与引入 rev 之前**逐位相同**（加法项为 0）。
fn gi_sec_key_masked(k0: u32, k1: u32, epoch: u32, pal: u32, rev: u32) -> vec2<u32> {
  return vec2<u32>(
    k0 + gi_mix(epoch * 0x9E3779B1u + rev * 0x85EBCA77u + 0x165667B1u),
    k1 + gi_mix(gi_mix(epoch + 0x85EBCA77u) + rev * 0x27D4EB2Fu + 0x27D4EB2Fu
                + pal * 0x9E3779B1u),
  );
}

// ============================================================================
// 像素邻域偏移（**空间复用**与**历史搜索**共用同一张 3×3 环表）
// ============================================================================
// 两处都要「在本像素周围找邻居」，共用一张表是必须的：两处各写一份的话，"哪个偏移排第几"会漂移，
// 而顺序是有语义的（十字优先 ⇒ 相机平移量小的时候第一个 tap 命中率最高，历史搜索靠它早退）。
// 顺序：+x、−x、+y、−y、++、+−、−+、−−。

/// 3×3 环上的第 `s` 个偏移（`s` = 0..8）。
fn gi_ring_offset(s: u32) -> vec2<i32> {
  if (s == 0u) { return vec2<i32>(1, 0); }
  if (s == 1u) { return vec2<i32>(-1, 0); }
  if (s == 2u) { return vec2<i32>(0, 1); }
  if (s == 3u) { return vec2<i32>(0, -1); }
  if (s == 4u) { return vec2<i32>(1, 1); }
  if (s == 5u) { return vec2<i32>(1, -1); }
  if (s == 6u) { return vec2<i32>(-1, 1); }
  return vec2<i32>(-1, -1);
}

/// 历史搜索的第 `s` 个候选偏移：`s = 0` = **正中**（重投影落点本身），`s = 1..9` = 3×3 环。
/// 正中必须排第一 —— 静止相机下它恒命中 ⇒ 顺序搜索的**第一个**候选就中，行为与"只认落点"逐位一致。
fn gi_hist_search_offset(s: u32) -> vec2<i32> {
  if (s == 0u) { return vec2<i32>(0, 0); }
  return gi_ring_offset(s - 1u);
}