// WGSL: 全屏三角 blit，storage texture → ViewTarget（P0.3 阶段 B）
// 渲染分辨率可低于窗口（Douglas #17 降分辨率策略）：uv 双线性采样上采样

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var src_sampler: sampler;

struct VsOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
  // 全屏三角形：vi 0/1/2 → 覆盖 [-1,3] NDC
  let x = f32((vi << 1u) & 2u);
  let y = f32(vi & 2u);
  var out: VsOut;
  out.pos = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
  out.uv = vec2<f32>(x, y);
  return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
  let c = textureSample(src_tex, src_sampler, in.uv).rgb;
  // view target 是 sRGB：硬件会把输出做 linear→sRGB 编码。
  // 先做 sRGB→linear 转换，两次抵消 → 屏幕像素 = storage texture 字节（渐变不被 gamma 提升）
  return vec4<f32>(srgb_to_linear(c), 1.0);
}

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
  return select(
    c / 12.92,
    pow((c + vec3<f32>(0.055)) / vec3<f32>(1.055), vec3<f32>(2.4)),
    c > vec3<f32>(0.04045),
  );
}
