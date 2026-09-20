// GI 路径的**共用工具**：几何面键、位打包、RNG、余弦半球采样。
//
// 被三处共用 ⇒ 单独成模块，任何一侧都不许持有第二份实现（口径分叉会让画面跳变）：
//   · `gi/screen.wesl`  —— ReSTIR 的复用判据（主命中面键 / 二次顶点键）与样本方向；
//   · `gi/ray.wesl`     —— 射线命中后算键（时域验证与复用都吃它）；
//   · `gi/denoise.wesl` —— 降噪导引的键与法线打包（与复用判据必须是同一套编码）。

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