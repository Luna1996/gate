#!/usr/bin/env bash
# 打包发布：run-fast（Fat LTO）构建 → 组装便携目录（exe + assets + DLSS 运行库）
#
# 产物：dist/gate-<版本>-win64/（gate-app.exe + assets/ + nvngx_dlss.dll；logs/、data/ 首次启动自建）
# 目录可放任意位置，exe 按自身目录找 assets（见 paths.rs）。不压缩，需 zip/7z 自行处理。
# 用法：bash package.sh（Git Bash / MSYS2 / WSL / Linux）
# 其他 profile：GATE_PROFILE=balance bash package.sh
# 不带 DLSS：GATE_DLSS=0 bash package.sh

set -euo pipefail
cd "$(dirname "$0")"

version=$(sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -n 1)
name="gate-${version}-win64"
dist="dist/${name}"

profile="${GATE_PROFILE:-run-fast}"
dlss="${GATE_DLSS:-1}"

if [ "$dlss" = "1" ]; then
  # NGX 链接库只有 MSVC 格式 → DLSS 构建走 msvc target（默认构建仍用本机 gnu）。
  # 依赖自检/下载/工具集发现都收敛在 tools/build-dlss.ps1。
  echo "== tools/build-dlss.ps1 -Profile ${profile} =="
  powershell -NoProfile -ExecutionPolicy Bypass -File tools/build-dlss.ps1 -Profile "$profile"
  if [ $? -ne 0 ]; then exit 1; fi
  bin="target/x86_64-pc-windows-msvc/${profile}/gate-app.exe"
else
  echo "== cargo build --profile ${profile} -p gate-app =="
  cargo build --profile "$profile" -p gate-app
  bin="target/${profile}/gate-app"
  if [ -f "${bin}.exe" ]; then
    bin="${bin}.exe"
  fi
fi

echo "== 组装 ${dist} =="
rm -rf "$dist"
mkdir -p "$dist" "$dist/licenses"
cp "$bin" "$dist/"
cp -r assets "$dist/assets"

if [ "$dlss" = "1" ]; then
  # DLSS SDK §9.5：随应用分发需带运行时 DLL 与许可文本
  cp deps/DLSS/lib/Windows_x86_64/rel/nvngx_dlss.dll "$dist/"
  cp deps/DLSS/LICENSE.txt "$dist/licenses/DLSS-LICENSE.txt"
  if [ -f deps/DLSS/DISTRIBUTION.txt ]; then
    cp deps/DLSS/DISTRIBUTION.txt "$dist/licenses/"
  else
    echo "WARN: deps/DLSS/DISTRIBUTION.txt 缺失（需按 SDK 指南 §9.5 摘录分发文本一次）"
  fi
fi

# nuke.vox 被 .gitignore 排除，新克隆的仓库里可能没有 → 只提示，不阻断
if [ ! -f "${dist}/assets/vox/nuke.vox" ]; then
  echo "WARN: assets/vox/nuke.vox 缺失（默认场景需要它；或把 consts.rs 的 STARTUP_DEMO_SCENE 改成 true）"
fi

echo "OK: $dist"