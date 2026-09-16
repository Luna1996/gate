#!/usr/bin/env bash
# 打包发布：release 构建 → 组装便携目录（exe + assets）
#
# 产物形态（也就是 paths.rs 认的「便携发布」形态）：
#   dist/gate-<版本>-win64/
#     gate-app.exe            可执行文件（Linux 上无 .exe 后缀，脚本自动识别）
#     assets/                 运行期只读资源（字体 / shader（含 WESL 源）/ vox / theme.ron / ui）
#     logs/  data/            运行期自建的可写目录（不随包发，首次启动生成）
# 整个目录放哪都行：双击 exe 即可运行（paths.rs 按 exe 同目录找 assets）。
# 不压缩，需要 zip/7z 自己动手。
#
# 用法：bash package.sh        （Git Bash / MSYS2 / WSL / Linux 均可）

set -euo pipefail
cd "$(dirname "$0")"

version=$(sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -n 1)
name="gate-${version}-win64"
dist="dist/${name}"

echo "== cargo build --release -p gate-app =="
cargo build --release -p gate-app

bin="target/release/gate-app"
if [ -f "${bin}.exe" ]; then
  bin="${bin}.exe"
fi

echo "== 组装 ${dist} =="
rm -rf "$dist"
mkdir -p "$dist"
cp "$bin" "$dist/"
cp -r assets "$dist/assets"

# nuke.vox 被 .gitignore 排除（体积大），新克隆的仓库里可能没有 → 只提示，不阻断
if [ ! -f "${dist}/assets/vox/nuke.vox" ]; then
  echo "WARN: assets/vox/nuke.vox 缺失（GATE_SCENE=vox 默认场景需要它）"
fi

echo "OK: $dist"