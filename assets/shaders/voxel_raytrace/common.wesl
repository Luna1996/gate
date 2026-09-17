import package::bindings::{b_palette, g, light_u};

// --- 常量区（与 Rust wgsl_consts mod 字节对齐）---
// Brick Tree：256³ chunk，4³ 分裂因子，4 层（256→64→16→4→1）
const CHUNK_SIZE: u32 = 256u;
const BRICK_FACTOR: u32 = 4u;
const MAX_LEVEL: u32 = 4u;
const NODE_FIXED_WORDS: u32 = 3u;    // mask_lo + mask_hi + palette
// b_struct Region ①：稠密 chunk 窗口。
const CHUNK_INDEX_CAP: u32 = 64u;
const CHUNK_INDEX_WORDS: u32 = 262144u;  // 64³
const TREE_BASE: u32 = 262144u;          // Region ② 起始

// --- 材质索引格式（16 位，与 Rust `gate_voxel::PALETTE_BITS` / `LEAF_*` 同源）---
// 节点第 3 字：bit0..15 = uniform 子块色，bit16..31 = LOD 子树多数色；叶父层 inline 每字装 2 个体素（LEAF_VOXELS_PER_WORD）。
// 权威定义在 `wire.rs`，改位宽须同步全部读点。

// 方向位掩码 LUT：3-bit 方向掩码（bit0=x>=0, bit1=y>=0, bit2=z>=0，共 8 种）→ side 与三轴步进 face id。
// 打包：bits0-2=side, bits8-15=face_x, bits16-23=face_y, bits24-31=face_z；face 规则 sign<0 穿对面。
const DIR_LUT: array<u32, 8u> = array<u32, 8u>(
  0x05030100u,
  0x05030001u,
  0x05020102u,
  0x05020003u,
  0x04030104u,
  0x04030005u,
  0x04020106u,
  0x04020007u,
);

// ---- face 索引（0..5 = -x,+x,-y,+y,-z,+z）与法线/诊断色的转换 ----
fn face_index_from_normal(n: vec3<f32>) -> u32 {
  let ax = abs(n.x); let ay = abs(n.y); let az = abs(n.z);
  if (ax >= max(ay, az)) { return select(0u, 1u, n.x >= 0.0); }
  if (ay >= az)          { return select(2u, 3u, n.y >= 0.0); }
  return select(4u, 5u, n.z >= 0.0);
}
fn face_normal_from_index(f: u32) -> vec3<f32> {
  if (f == 0u) { return vec3<f32>(-1.0, 0.0, 0.0); }
  if (f == 1u) { return vec3<f32>( 1.0, 0.0, 0.0); }
  if (f == 2u) { return vec3<f32>(0.0, -1.0, 0.0); }
  if (f == 3u) { return vec3<f32>(0.0,  1.0, 0.0); }
  if (f == 4u) { return vec3<f32>(0.0, 0.0, -1.0); }
  if (f == 5u) { return vec3<f32>(0.0, 0.0,  1.0); }
  return vec3<f32>(0.0);
}
fn face_color_from_index(f: u32) -> vec3<f32> {
  if (f == 0u) { return vec3<f32>(1.0, 0.2, 0.2); }
  if (f == 1u) { return vec3<f32>(0.2, 1.0, 0.2); }
  if (f == 2u) { return vec3<f32>(0.2, 0.2, 1.0); }
  if (f == 3u) { return vec3<f32>(1.0, 1.0, 0.2); }
  if (f == 4u) { return vec3<f32>(0.2, 1.0, 1.0); }
  if (f == 5u) { return vec3<f32>(1.0, 0.2, 1.0); }
  return vec3<f32>(1.0);
}

// ---- sRGB → linear 转换（palette/sky 均以 sRGB u8/255 存储，光照前必须转 linear） ----
fn srgb_channel_to_linear(c: f32) -> f32 {
  if (c <= 0.04045) {
    return c / 12.92;
  }
  return pow((c + 0.055) / 1.055, 2.4);
}

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(
    srgb_channel_to_linear(c.x),
    srgb_channel_to_linear(c.y),
    srgb_channel_to_linear(c.z),
  );
}

// ---- palette albedo 解包（sRGB → linear）----
fn palette_albedo(palette_base: u32, pal: u32) -> vec3<f32> {
  let w0 = b_palette[palette_base + pal * 2u];
  let srgb = vec3<f32>(
    f32(w0 & 0xFFu),
    f32((w0 >> 8u) & 0xFFu),
    f32((w0 >> 16u) & 0xFFu),
  ) / 255.0;
  return srgb_to_linear(srgb);
}

/// palette 条目的自发光强度（0..1），= word1 低字节（与 `pack_palette_entry` 同布局）。
/// 发光体素是光源本体：着色时 radiance 直出，不吃方向/阴影。
fn palette_emissive(palette_base: u32, pal: u32) -> f32 {
  return f32(b_palette[palette_base + pal * 2u + 1u] & 0xFFu) / 255.0;
}

/// palette 条目的粗糙度（0 = 镜面 … 1 = 完全粗糙），= word0 高字节（`roughness << 24` 同布局）。
/// 着色侧用来定高光指数：粗糙度越小高光越窄越亮。
fn palette_roughness(palette_base: u32, pal: u32) -> f32 {
  return f32((b_palette[palette_base + pal * 2u] >> 24u) & 0xFFu) / 255.0;
}

/// palette 条目的透射率（0 = 不透明 … 1 = 全透），= word1 次字节（`transmission << 8` 同布局）。
/// 着色侧用来做透射续行：primary 射线按 tau 连乘衰减后穿过体素继续走，太阳阴影射线同样按 tau 连乘。
fn palette_transmission(palette_base: u32, pal: u32) -> f32 {
  return f32((b_palette[palette_base + pal * 2u + 1u] >> 8u) & 0xFFu) / 255.0;
}

// 天空纯色（miss 像素与探针射线 miss 端点共用）。
// 输入 sRGB，返回 linear。
fn sky_rgb() -> vec3<f32> {
  return srgb_to_linear(light_u.sky_color.xyz);
}

// 线性 → sRGB 转换
fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
  let l = clamp(c, vec3<f32>(0.0), vec3<f32>(1.0));
  return vec3<f32>(
    select(12.92 * l.r, pow(l.r, 1.0 / 2.4) * 1.055 - 0.055, l.r > 0.0031308),
    select(12.92 * l.g, pow(l.g, 1.0 / 2.4) * 1.055 - 0.055, l.g > 0.0031308),
    select(12.92 * l.b, pow(l.b, 1.0 / 2.4) * 1.055 - 0.055, l.b > 0.0031308)
  );
}
