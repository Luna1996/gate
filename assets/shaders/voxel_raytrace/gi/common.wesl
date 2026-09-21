// GI 路径的**共用工具**：几何面键、位打包、RNG、余弦半球采样。
//
// 被三处共用 ⇒ 单独成模块，任何一侧都不许持有第二份实现（口径分叉会让画面跳变）：
//   · `gi/screen.wesl`  —— ReSTIR 的复用判据（主命中面键 / 二次顶点键）与样本方向；
//   · `gi/ray.wesl`     —— 射线命中后算键（时域验证与复用都吃它）；
//   · `gi/denoise.wesl` —— 降噪导引的键与法线打包（与复用判据必须是同一套编码）。

import package::bindings::{face_slots, gi_sec_slots};

/// 键合法范围：13 bit 有符号偏移（voxel 坐标绝对值 < 4096）。
fn gi_key_ok(v: vec3<i32>) -> bool {
  return all(v >= vec3<i32>(-4096)) && all(v < vec3<i32>(4096));
}

/// 键 word 0：x、y 各 13 bit（+4096 偏移）。
fn gi_key0(v: vec3<i32>) -> u32 {
  return u32(v.x + 4096) | (u32(v.y + 4096) << 13u);
}

/// 键 word 1：z 13 bit、面 3 bit（13..15）、物体 id 16 bit（16..31，+1 存，0 = 主世界）。
/// **面进键**是薄板 / 墙缝不漏光的根本：同一体素的两个面是两个不同的键（见 `gi/screen.wesl`）。
fn gi_key1(v: vec3<i32>, face: u32, obj: i32) -> u32 {
  return u32(v.z + 4096) | ((face & 7u) << 13u) | (u32(obj + 1) << 16u);
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
/// 每槽字数。0 = 认领标志（CAS 目标）、1/2 = 面键、3 = 认领者 texel 下标、4 = palette 条目号、
/// 5..7 = **着色结果**（线性 rgb，曝光前；`dda_main` 与 `dda_face_main` 共用 `shade_face`）、
/// 8 = GI 累加计数、9..11 = GI 累加和（**定点**，见 `GI_ACC_*`）。
const FACE_WORDS: u32 = 12u;
const FACE_W_FLAG: u32 = 0u;
const FACE_W_K0: u32 = 1u;
const FACE_W_K1: u32 = 2u;
const FACE_W_TEXEL: u32 = 3u;
const FACE_W_PAL: u32 = 4u;
const FACE_W_COL: u32 = 5u;
const FACE_W_GI_N: u32 = 8u;
const FACE_W_GI: u32 = 9u;

/// **逐面 GI 均值**的定点参数（WGSL 默认没有 float 原子 ⇒ 只能加整数）。
/// 每个 GI texel 把自己的 GI 值（`gi_tex.rgb`，已含增益）加上去，面着色时除以计数取平均。
/// 为什么必须做这一步：同一个面的所有 texel 的 GI 射线**起点都是该体素中心**（见 `gi_ss_main` 的
/// `org`）⇒ 它们是对**同一个积分**的独立估计 ⇒ 取平均是**无偏**的纯降噪（方差 ÷ 参与数）。
/// 不取平均而只取某一个 texel（认领者）的值时，该 texel 的降噪残差会被整面一起放大成
/// **相干闪烁**（原来逐像素各自独立、被眼睛平均掉），面越大越明显、关降噪时是整面雪块。
/// 量纲：`SCALE` × 值 ≤ `CLAMP` ⇒ 单条贡献 ≤ 65280；参与数封顶 `MAX_N = 1024`
/// ⇒ 和 ≤ 6.7e7（远小于 u32 上限，且不会被并发冲过 —— 计数用 CAS 预留，见 `dda_face_accum`）。
const GI_ACC_SCALE: f32 = 256.0;
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

// ============================================================================
// 二次顶点按面缓存（`gi_sec_slots`）—— 只服务 GI 射线的命中着色（`gi/ray.wesl`）
// ============================================================================
// 为什么能按面缓存：二次顶点着色里**除射线自身的介质透过率 `tint` 以外**的全部输入都只由"命中的
// 那个可见体素面"决定 —— 材质取值（`fetch_material` 内部把采样点与 mip LOD 距离都量化到体素中心）、
// 着色法线（`voxel_normal` 是 (voxel, face) 的纯函数）、视向量（锚点 = 体素中心）、太阳 NEE 的起点
// （体素中心沿面法线外推）与 AO 采样点（体素中心）—— 后两者与主 pass 的 `p_voxel` 是同口径。
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
const GI_SEC_WORDS: u32 = 5u;
const GI_SEC_W_K0: u32 = 0u;
const GI_SEC_W_K1: u32 = 1u;
const GI_SEC_W_RAD: u32 = 2u;

/// 面键 → 二次顶点缓存的槽下标。`ok = false` = 表未就绪 ⇒ 该射线走内联着色。
fn gi_sec_slot_of(k0: u32, k1: u32, ok: ptr<function, bool>) -> u32 {
  let n = arrayLength(&gi_sec_slots) / GI_SEC_WORDS;
  *ok = n >= 1u;
  if (!*ok) { return 0u; }
  return gi_mix(k0 + gi_mix(k1 * 0x9E3779B1u)) & (n - 1u);
}