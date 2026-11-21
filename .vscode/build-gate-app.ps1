param(
  [Parameter(Mandatory = $true)]
  [ValidateSet('build-fast', 'balance', 'run-fast')]
  [string]$Profile
)

$ErrorActionPreference = 'Stop'
$workspace = Split-Path -Parent $PSScriptRoot
Set-Location $workspace

$dlssSdk = if ($env:DLSS_SDK) { $env:DLSS_SDK } else { 'C:\repo\DLSS' }
$vulkanSdk = if ($env:VULKAN_SDK) { $env:VULKAN_SDK } else { 'C:\Program Files\Vulkan' }
$llvmBin = if ($env:LIBCLANG_PATH) { $env:LIBCLANG_PATH } else { 'C:\Program Files\LLVM\bin' }

if (Test-Path $dlssSdk) {
  $env:DLSS_SDK = $dlssSdk
}
if (Test-Path $vulkanSdk) {
  $env:VULKAN_SDK = $vulkanSdk
}
if (Test-Path $llvmBin) {
  $env:LIBCLANG_PATH = $llvmBin
}

$features = @()
if (Test-Path (Join-Path $dlssSdk 'lib\Windows_x86_64\rel\nvngx_dlssd.dll')) {
  $features = @('--features', 'dlss')
} else {
  Write-Warning "DLSS SDK not found at $dlssSdk; building without the dlss feature"
}

& cargo build --profile $Profile -p gate-app @features
if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}

$dest = Join-Path $workspace "target\$Profile"
foreach ($dll in @('nvngx_dlss.dll', 'nvngx_dlssd.dll')) {
  $src = Join-Path $dlssSdk "lib\Windows_x86_64\rel\$dll"
  if (Test-Path $src) {
    Copy-Item $src (Join-Path $dest $dll) -Force
  }
}
