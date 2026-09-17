#!/usr/bin/env bash
# 打包发布：release 构建 → 组装便携目录（exe + assets）
#
# 产物：dist/gate-<版本>-win64/（gate-app.exe + assets/；logs/、data/ 首次启动自建）
# 目录可放任意位置，exe 按自身目录找 assets（见 paths.rs）。不压缩，需 zip/7z 自行处理。
# 用法：bash package.sh（Git Bash / MSYS2 / WSL / Linux）

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

# nuke.vox 被 .gitignore 排除，新克隆的仓库里可能没有 → 只提示，不阻断
if [ ! -f "${dist}/assets/vox/nuke.vox" ]; then
  echo "WARN: assets/vox/nuke.vox 缺失（默认场景需要它；或把 consts.rs 的 STARTUP_DEMO_SCENE 改成 true）"
fi

echo "OK: $dist"