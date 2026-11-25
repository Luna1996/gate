# DLSS 构建依赖自动获取（幂等）：deps/ 为 gitignored 缓存，
# 首次构建自动下载，之后全走缓存不联网。来源均为官方公开渠道：
#   - DLSS SDK v310.9.1（dlss_wgpu 6.0.1 的版本锁定要求）
#   - LLVM 23.1.6 官方便携包（仅取 bindgen 需要的 libclang）
param([switch]$Force)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$dlss = Join-Path $root "deps/DLSS"
$llvm = Join-Path $root "deps/LLVM"

$dlssTag = "v310.9.1"
$dlssBase = "https://raw.githubusercontent.com/NVIDIA/DLSS/$dlssTag"

$dlssFiles = @(
  "LICENSE.txt",
  "README.md",
  "doc/DLSS_Programming_Guide_Release.pdf",
  "include/nvsdk_ngx.h",
  "include/nvsdk_ngx_defs.h",
  "include/nvsdk_ngx_defs_dlssd.h",
  "include/nvsdk_ngx_defs_dlssg.h",
  "include/nvsdk_ngx_defs_vk.h",
  "include/nvsdk_ngx_helpers.h",
  "include/nvsdk_ngx_helpers_cuda.h",
  "include/nvsdk_ngx_helpers_d3d.h",
  "include/nvsdk_ngx_helpers_dlssd.h",
  "include/nvsdk_ngx_helpers_dlssd_cuda.h",
  "include/nvsdk_ngx_helpers_dlssd_d3d.h",
  "include/nvsdk_ngx_helpers_dlssd_vk.h",
  "include/nvsdk_ngx_helpers_dlssg.h",
  "include/nvsdk_ngx_helpers_dlssg_d3d.h",
  "include/nvsdk_ngx_helpers_dlssg_vk.h",
  "include/nvsdk_ngx_helpers_vk.h",
  "include/nvsdk_ngx_loader.h",
  "include/nvsdk_ngx_params.h",
  "include/nvsdk_ngx_params_dlssd.h",
  "include/nvsdk_ngx_params_dlssg.h",
  "include/nvsdk_ngx_standalone_common.h",
  "include/nvsdk_ngx_standalone_cuda.h",
  "include/nvsdk_ngx_vk.h",
  "lib/Windows_x86_64/x64/nvsdk_ngx_d.lib",
  "lib/Windows_x86_64/x64/nvsdk_ngx_s.lib",
  "lib/Windows_x86_64/rel/nvngx_dlss.dll"
)

$fetched = 0
foreach ($f in $dlssFiles) {
  $dst = Join-Path $dlss ($f -replace "/", "\")
  if (-not $Force -and (Test-Path $dst)) { continue }
  New-Item -ItemType Directory -Force -Path (Split-Path -Parent $dst) | Out-Null
  curl.exe -sL --retry 3 -o $dst "$dlssBase/$f"
  if (-not (Test-Path $dst) -or (Get-Item $dst).Length -lt 1024) {
    throw "DLSS 依赖下载失败：$f"
  }
  $fetched++
}
"DLSS SDK $dlssTag → $dlss（新增 $fetched 项）"

# ---- libclang（bindgen 解析 NGX 头文件用）
$libclang = Join-Path $llvm "bin/libclang.dll"
$clangVer = "23"
if (-not $Force -and (Test-Path $libclang)) {
  "libclang → 已存在，跳过"
} else {
  $tarUrl = "https://github.com/llvm/llvm-project/releases/download/llvmorg-23.1.6/clang+llvm-23.1.6-x86_64-pc-windows-msvc.tar.zst"
  $tmp = Join-Path $env:TEMP "clang+llvm-23.1.6-win64.tar.zst"
  if (-not (Test-Path $tmp)) {
    "下载 LLVM 便携包（263 MB，仅一次）…"
    curl.exe -sL --retry 3 -o $tmp $tarUrl
  }
  New-Item -ItemType Directory -Force -Path "$llvm/bin" | Out-Null
  New-Item -ItemType Directory -Force -Path "$llvm/lib/clang" | Out-Null
  # 只解出 bindgen 需要的两个成员；bsdtar 按模式匹配
  tar --zstd -xf $tmp -C $llvm "bin/libclang.dll" "lib/clang/$clangVer/include/*" 2>&1 | Out-Null
  if (-not (Test-Path $libclang)) {
    throw "libclang 解压失败（tar 不支持 zst？可手动安装 LLVM 后重跑）"
  }
  Remove-Item $tmp -Force -ErrorAction SilentlyContinue
  "libclang 23.1.6 → $llvm"
}

# ---- Vulkan 头文件（dlss_wgpu 构建脚本要求 VULKAN_SDK 指向含 Include/ 的目录）
# Vulkan-Headers 为 Apache-2.0，可自由分发；vendored 后新机器无需安装 Vulkan SDK。
$vkh = Join-Path $root "deps/Vulkan-Headers"
$vkhMark = Join-Path $vkh "Include/vulkan/vulkan_core.h"
if (-not $Force -and (Test-Path $vkhMark)) {
  "Vulkan-Headers → 已存在，跳过"
} else {
  $ver = "v1.4.321"
  $zip = Join-Path $env:TEMP "Vulkan-Headers-$ver.zip"
  curl.exe -sL --retry 3 -o $zip "https://github.com/KhronosGroup/Vulkan-Headers/archive/refs/tags/$ver.zip"
  $ex = Get-ChildItem $env:TEMP -Directory -Filter "Vulkan-Headers-1.4.321" -ErrorAction SilentlyContinue |
    Select-Object -First 1
  if (-not $ex) {
    Expand-Archive $zip -DestinationPath $env:TEMP
    $ex = Get-ChildItem $env:TEMP -Directory -Filter "Vulkan-Headers-1.4.321" |
      Select-Object -First 1
  }
  $src = Join-Path $ex.FullName "include/vulkan"
  New-Item -ItemType Directory -Force -Path "$vkh/Include" | Out-Null
  Copy-Item $src "$vkh/Include/vulkan" -Recurse -Force
  Copy-Item (Join-Path $ex.FullName "LICENSE.txt") $vkh -Force -ErrorAction SilentlyContinue
  Remove-Item $zip, $ex -Recurse -Force -ErrorAction SilentlyContinue
  if (-not (Test-Path $vkhMark)) { throw "Vulkan-Headers 获取失败" }
  "Vulkan-Headers $ver → $vkh"
}
"依赖就绪。"
