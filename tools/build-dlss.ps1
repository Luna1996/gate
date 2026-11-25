# DLSS 构建包装：依赖自检 → vcvars 自动发现 → MSVC target 构建 → 运行 DLL 落位。
# F5 默认路径；NGX 链接库只有 MSVC 格式，因此 DLSS 构建固定走 msvc target。
param(
  [ValidateSet("build-fast", "balance", "run-fast")]
  [string]$Profile = "build-fast"
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$target = "x86_64-pc-windows-msvc"

# 1) 依赖（幂等，缓存命中不联网；失败时子脚本 throw 向上传播终止）
& (Join-Path $PSScriptRoot "fetch-dlss-deps.ps1")

# 2) 定位 vcvars 与最新 MSVC 工具集版本
$vcRoots = @("C:\Program Files\Microsoft Visual Studio", "C:\Program Files (x86)\Microsoft Visual Studio")
$vcvars = Get-ChildItem $vcRoots -Recurse -Filter "vcvars64.bat" -ErrorAction SilentlyContinue |
  Where-Object { $_.FullName -match "VC\\Auxiliary\\Build" } |
  Sort-Object FullName -Descending | Select-Object -First 1
if (-not $vcvars) {
  throw "找不到 vcvars64.bat：需安装 VS BuildTools（含 C++ 工具集与 Windows SDK）"
}
$vcDir = $vcvars.FullName
for ($i = 0; $i -lt 3; $i++) { $vcDir = Split-Path -Parent $vcDir }   # → VC\
$toolset = (Get-ChildItem (Join-Path $vcDir "Tools\MSVC") -Directory |
  Sort-Object Name -Descending | Select-Object -First 1).Name
"MSVC 工具集：$toolset（$($vcvars.FullName)）"

# 3) 构建（vcvars 只影响子进程环境，不污染当前会话）
cmd /c "call ""$($vcvars.FullName)"" -vcvars_ver=$toolset && cargo build --profile $Profile -p gate-app --features dlss --target $target"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# 4) NGX 运行时 DLL 与 exe 同目录（缺失时 NGX 探测失败、自动回退普通 blit）
$exeDir = Join-Path $root "target/$target/$Profile"
Copy-Item (Join-Path $root "deps/DLSS/lib/Windows_x86_64/rel/nvngx_dlss.dll") $exeDir -Force
"OK: $exeDir/gate-app.exe"
