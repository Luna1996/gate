// WGSL: 全屏三角 blit，storage texture → ViewTarget（可降分辨率，uv 双线性上采样）。
// 两个 fragment 入口：`fs_main`（纯 blit）/ `fs_fxaa`（FXAA 抗锯齿，菜单「视频/抗锯齿」）。
// Rust 侧建两条 pipeline（同 layout、同 bind group），按开关选一条。

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var src_sampler: sampler;

struct VsOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
  let x = f32((vi << 1u) & 2u);
  let y = f32(vi & 2u);
  var out: VsOut;
  out.pos = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
  out.uv = vec2<f32>(x, y);
  return out;
}

/// 像素完美的整数块上采样：每个目标像素取**最近的源像素**，不做任何插值
/// （`uv` 在目标像素中心 = `(px+0.5)/dst_dim` ⇒ `floor(uv × src_dim)` 就是整数块映射：
/// `factor = 3` 时一个源像素正好铺 3×3 个目标像素；非整除的尺寸也只差最后一列/行的 1 像素）。
/// 走 `textureLoad` 而不是 sampler ⇒ 与滤波模式无关，降分辨率时永远不糊。
/// `factor = 1` 时逐位等于 1:1 拷贝。
@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
  let dim = textureDimensions(src_tex);
  let s = min(vec2<u32>(floor(in.uv * vec2<f32>(dim))), dim - vec2<u32>(1u));
  let c = textureLoad(src_tex, vec2<i32>(s), 0).rgb;
  // view target 是 sRGB：先做 sRGB→linear。
  return vec4<f32>(srgb_to_linear(c), 1.0);
}

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
  return select(
    c / 12.92,
    pow((c + vec3<f32>(0.055)) / vec3<f32>(1.055), vec3<f32>(2.4)),
    c > vec3<f32>(0.04045),
  );
}

// ============================================================================
// FXAA（NVIDIA FXAA 3.11）。
// 取值空间：输入为 sRGB 编码后的 LDR 缓冲（`dda_main` 的 `linear_to_srgb` 结果）。
// ============================================================================
/// 绝对下限：局部对比度低于它 → 判定不在边缘。
const FXAA_EDGE_MIN: f32 = 0.0312;
/// 相对阈值：局部对比度 / 邻域亮度峰值（越小处理越多边缘）。
const FXAA_EDGE_REL: f32 = 0.125;
/// 沿边缘搜索端点的迭代上限。
const FXAA_ITER: i32 = 12;
/// 亚像素偏移强度（0 = 关闭，1 = 全强度）。
const FXAA_SUBPIX: f32 = 0.75;

fn fxaa_luma(c: vec3<f32>) -> f32 {
  return sqrt(dot(c, vec3<f32>(0.299, 0.587, 0.114)));
}

/// 第 i 步的搜索跨度。
fn fxaa_quality(q: i32) -> f32 {
  switch (q) {
    case 5: { return 1.5; }
    case 6, 7, 8, 9: { return 2.0; }
    case 10: { return 4.0; }
    case 11: { return 8.0; }
    default: { return 1.0; }
  }
}

/// 取 `uv + off`（off 以源纹素为单位）。
fn fxaa_at(uv: vec2<f32>, off: vec2<f32>, texel: vec2<f32>) -> vec3<f32> {
  return textureSample(src_tex, src_sampler, uv + off * texel).rgb;
}

@fragment
fn fs_fxaa(in: VsOut) -> @location(0) vec4<f32> {
  let texel = 1.0 / vec2<f32>(textureDimensions(src_tex));
  let uv = in.uv;
  let center = fxaa_at(uv, vec2<f32>(0.0, 0.0), texel);
  let luma_m = fxaa_luma(center);
  let luma_n = fxaa_luma(fxaa_at(uv, vec2<f32>(0.0, 1.0), texel));
  let luma_s = fxaa_luma(fxaa_at(uv, vec2<f32>(0.0, -1.0), texel));
  let luma_w = fxaa_luma(fxaa_at(uv, vec2<f32>(-1.0, 0.0), texel));
  let luma_e = fxaa_luma(fxaa_at(uv, vec2<f32>(1.0, 0.0), texel));

  // ---- 1) 早退：局部对比度不够 → 不在边缘，原样输出 ----
  let luma_min = min(luma_m, min(min(luma_n, luma_s), min(luma_w, luma_e)));
  let luma_max = max(luma_m, max(max(luma_n, luma_s), max(luma_w, luma_e)));
  let luma_range = luma_max - luma_min;
  if (luma_range < max(FXAA_EDGE_MIN, luma_max * FXAA_EDGE_REL)) {
    return vec4<f32>(srgb_to_linear(center), 1.0);
  }

  // ---- 2) 3×3 亮度（补四个角），估计边缘是水平还是竖直 ----
  let luma_dl = fxaa_luma(fxaa_at(uv, vec2<f32>(-1.0, -1.0), texel));
  let luma_dr = fxaa_luma(fxaa_at(uv, vec2<f32>(1.0, -1.0), texel));
  let luma_ul = fxaa_luma(fxaa_at(uv, vec2<f32>(-1.0, 1.0), texel));
  let luma_ur = fxaa_luma(fxaa_at(uv, vec2<f32>(1.0, 1.0), texel));
  let luma_ns = luma_n + luma_s;
  let luma_we = luma_w + luma_e;
  let luma_left_corners = luma_dl + luma_ul;
  let luma_right_corners = luma_dr + luma_ur;
  let luma_down_corners = luma_dl + luma_dr;
  let luma_up_corners = luma_ul + luma_ur;
  // 水平/竖直方向的二阶梯度（绝对值之和）。
  let edge_horz = abs(-2.0 * luma_w + luma_left_corners)
    + abs(-2.0 * luma_m + luma_ns) * 2.0
    + abs(-2.0 * luma_e + luma_right_corners);
  let edge_vert = abs(-2.0 * luma_n + luma_up_corners)
    + abs(-2.0 * luma_m + luma_we) * 2.0
    + abs(-2.0 * luma_s + luma_down_corners);
  let is_horizontal = edge_horz >= edge_vert;
  // 边缘水平 → 沿竖直方向找端点，反之亦然。
  var step_len = select(texel.x, texel.y, is_horizontal);
  let luma_1 = select(luma_w, luma_s, is_horizontal);
  let luma_2 = select(luma_e, luma_n, is_horizontal);

  // ---- 3) 朝梯度更陡的一侧走 ----
  let grad_1 = luma_1 - luma_m;
  let grad_2 = luma_2 - luma_m;
  let is_1_steepest = abs(grad_1) >= abs(grad_2);
  let grad_scaled = 0.25 * max(abs(grad_1), abs(grad_2));
  var luma_local_avg = 0.0;
  if (is_1_steepest) {
    step_len = -step_len;
    luma_local_avg = 0.5 * (luma_1 + luma_m);
  } else {
    luma_local_avg = 0.5 * (luma_2 + luma_m);
  }

  // ---- 4) 沿边缘两侧探索直到亮度偏离局部均值超过局部梯度（= 到端点）。 ----
  var cur_uv = uv;
  var off = vec2<f32>(0.0, 0.0);
  if (is_horizontal) {
    cur_uv.y = cur_uv.y + step_len * 0.5;
    off.x = texel.x;
  } else {
    cur_uv.x = cur_uv.x + step_len * 0.5;
    off.y = texel.y;
  }
  var uv1 = cur_uv - off;
  var uv2 = cur_uv + off;
  var luma_end1 = fxaa_luma(fxaa_at(uv1, vec2<f32>(0.0, 0.0), texel)) - luma_local_avg;
  var luma_end2 = fxaa_luma(fxaa_at(uv2, vec2<f32>(0.0, 0.0), texel)) - luma_local_avg;
  var reached1 = abs(luma_end1) >= grad_scaled;
  var reached2 = abs(luma_end2) >= grad_scaled;
  var reached_both = reached1 && reached2;
  uv1 = select(uv1 - off, uv1, reached1);
  uv2 = select(uv2 + off, uv2, reached2);
  if (!reached_both) {
    for (var i = 2; i < FXAA_ITER; i = i + 1) {
      if (!reached1) {
        luma_end1 = fxaa_luma(fxaa_at(uv1, vec2<f32>(0.0, 0.0), texel)) - luma_local_avg;
      }
      if (!reached2) {
        luma_end2 = fxaa_luma(fxaa_at(uv2, vec2<f32>(0.0, 0.0), texel)) - luma_local_avg;
      }
      reached1 = abs(luma_end1) >= grad_scaled;
      reached2 = abs(luma_end2) >= grad_scaled;
      reached_both = reached1 && reached2;
      if (!reached1) {
        uv1 = uv1 - off * fxaa_quality(i);
      }
      if (!reached2) {
        uv2 = uv2 + off * fxaa_quality(i);
      }
      if (reached_both) {
        break;
      }
    }
  }

  // ---- 5) 取更近的一端做偏移，方向不对就不偏移 ----
  let distance1 = select(uv.y - uv1.y, uv.x - uv1.x, is_horizontal);
  let distance2 = select(uv2.y - uv.y, uv2.x - uv.x, is_horizontal);
  let is_dir1 = distance1 < distance2;
  let distance_final = min(distance1, distance2);
  let edge_thickness = distance1 + distance2;
  let luma_center_smaller = luma_m < luma_local_avg;
  let correct1 = (luma_end1 < 0.0) != luma_center_smaller;
  let correct2 = (luma_end2 < 0.0) != luma_center_smaller;
  let correct = select(correct2, correct1, is_dir1);
  let pixel_offset = -distance_final / edge_thickness + 0.5;
  var final_offset = select(0.0, pixel_offset, correct);

  // ---- 6) 亚像素修正：中心亮度偏离 3×3 均值越多补得越多。 ----
  let luma_avg = (1.0 / 12.0)
    * (2.0 * (luma_ns + luma_we) + luma_left_corners + luma_right_corners);
  let sub1 = clamp(abs(luma_avg - luma_m) / luma_range, 0.0, 1.0);
  let sub2 = (-2.0 * sub1 + 3.0) * sub1 * sub1;
  final_offset = max(final_offset, sub2 * sub2 * FXAA_SUBPIX);

  var final_uv = uv;
  if (is_horizontal) {
    final_uv.y = final_uv.y + final_offset * step_len;
  } else {
    final_uv.x = final_uv.x + final_offset * step_len;
  }
  let final_color = fxaa_at(final_uv, vec2<f32>(0.0, 0.0), texel);
  return vec4<f32>(srgb_to_linear(final_color), 1.0);
}
