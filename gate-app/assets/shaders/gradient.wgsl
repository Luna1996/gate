// WGSL: 全屏 compute 渐变（P0.3 spike 验收画面）
// 对应 Rust 侧 GradientUniforms { size: vec2<f32> }

@group(0) @binding(0) var out_tex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<uniform> viewport: vec4<f32>; // xy = size, zw 未用（对齐 16B）

@compute @workgroup_size(8, 8)
fn gradient(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = vec2<u32>(viewport.xy);
    if (gid.x >= size.x || gid.y >= size.y) {
        return;
    }
    let uv = vec2<f32>(gid.xy) / viewport.xy;
    // 暗色实验室基调：深蓝→洋红渐变，右下角压暗
    let c = vec3<f32>(0.05 + 0.5 * uv.x + 0.3 * uv.y, 0.08 + 0.15 * uv.y, 0.2 + 0.45 * (1.0 - uv.x));
    textureStore(out_tex, gid.xy, vec4<f32>(c, 1.0));
}
