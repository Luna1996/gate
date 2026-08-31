//! P3.1 WGSL 编译期校验：用 naga（bevy_render 同版本）parse + validate 全部 shader 资产。
//! shader 语法/类型错误在运行时才暴露（黑屏 + 日志），此测试把失败提前到 CI。
//!
//! 注意：wgpu 实际编译链 = naga → SPIR-V（DXC/FXC 再翻译），此处 validate 覆盖
//! naga 前端与验证器；平台后端差异仍由实机 F5 验收兜底。

use std::path::PathBuf;

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn compile_wgsl(rel_path: &str) {
    let path = manifest_dir().join("../gate-app/assets").join(rel_path);
    let src = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{rel_path} 读取失败: {e}"));
    let module = naga::front::wgsl::parse_str(&src)
        .unwrap_or_else(|e| panic!("{rel_path} naga parse 失败: {e:?}"));
    // validate 需要常量评估等能力：默认 Options 即可
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    let info = validator
        .validate(&module)
        .unwrap_or_else(|e| panic!("{rel_path} naga validate 失败: {e:?}"));
    // validate 通过即 shader 合法（entry point 集合非空由 dda_main/blit 的运行时装配保证）
    let _ = info;
}

#[test]
fn wgsl_shaders_parse_and_validate() {
    compile_wgsl("shaders/dda.wgsl");
    compile_wgsl("shaders/gradient.wgsl");
    compile_wgsl("shaders/blit.wgsl");
}
