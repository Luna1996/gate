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

# 构建必需：NGX 头文件（bindgen）+ 导入库（链接）+ 运行时 DLL；
# LICENSE 为分发合规要求保留。doc/ 编程指南 6 MB 非构建依赖，不进必需清单。
$dlssFiles = @(
  "LICENSE.txt",
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
$dlssApi = "https://api.github.com/repos/NVIDIA/DLSS/contents"
foreach ($f in $dlssFiles) {
  $dst = Join-Path $dlss ($f -replace "/", "\")
  if (-not $Force -and (Test-Path $dst)) { continue }
  New-Item -ItemType Directory -Force -Path (Split-Path -Parent $dst) | Out-Null
  # 二进制（lib/）用 1 KB 下限防 HTML 错误页；头文件与文档本身可以很小
  $minBytes = if ($f -like "lib/*") { 1024 } else { 64 }
  $ok = $false
  foreach ($attempt in 1..2) {
    if ($attempt -eq 1) {
      # --speed-limit/--speed-time：连接停滞（<1 KB/s 持续 30 s）即中断，防 raw CDN 挂死
      curl.exe -sL --retry 3 --speed-limit 1024 --speed-time 30 -o $dst "$dlssBase/$f"
    } else {
      # raw CDN 对大文件（PDF / DLL）常断流：走 contents API 的 raw media type
      curl.exe -sL --retry 3 --speed-limit 1024 --speed-time 30 -H "Accept: application/vnd.github.raw" -o $dst "$dlssApi/$f`?ref=$dlssTag"
    }
    if (Test-Path $dst) { $ok = (Get-Item $dst).Length -ge $minBytes; if ($ok) { break } }
  }
  if (-not $ok) {
    throw "DLSS 依赖下载失败：$f"
  }
  $fetched++
}
"DLSS SDK $dlssTag → $dlss（新增 $fetched 项）"

# ---- libclang（bindgen 解析 NGX 头文件用）
# 优先复用系统 LLVM（bindgen 兼容 libclang 9+，NGX 头文件为纯 C 声明）；
# 无系统安装才下载 263 MB 便携包。libclang 按自身相对路径找资源目录 ⇒ 布局与便携包一致。
$libclang = Join-Path $llvm "bin/libclang.dll"
$clangVer = "23"
$sysLlvm = "C:\Program Files\LLVM"
$sysDll = Join-Path $sysLlvm "bin/libclang.dll"
if (-not $Force -and (Test-Path $libclang)) {
  "libclang → 已存在，跳过"
} elseif (Test-Path $sysDll) {
  $sysVer = (Get-ChildItem (Join-Path $sysLlvm "lib/clang") -Directory | Sort-Object Name -Descending | Select-Object -First 1).Name
  New-Item -ItemType Directory -Force -Path "$llvm/bin" | Out-Null
  New-Item -ItemType Directory -Force -Path "$llvm/lib/clang" | Out-Null
  Copy-Item $sysDll "$llvm/bin/" -Force
  Copy-Item (Join-Path $sysLlvm "lib/clang/$sysVer") "$llvm/lib/clang/" -Recurse -Force
  "libclang → 复用系统 LLVM $sysVer"
} else {
  $tarUrl = "https://github.com/llvm/llvm-project/releases/download/llvmorg-23.1.6/clang+llvm-23.1.6-x86_64-pc-windows-msvc.tar.zst"
  $tmp = Join-Path $env:TEMP "clang+llvm-23.1.6-win64.tar.zst"
  if (-not (Test-Path $tmp)) {
    "下载 LLVM 便携包（263 MB，仅一次）…"
    curl.exe -sL --retry 3 --speed-limit 4096 --speed-time 60 -o $tmp $tarUrl
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
