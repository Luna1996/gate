# 本地 CI 入口：与 .github/workflows/ci.yml 保持同一套检查
# 用法：pwsh scripts/ci.ps1  （或 ./scripts/ci.ps1）
$ErrorActionPreference = 'Stop'
$env:RUSTUP_TOOLCHAIN = 'stable'
Set-Location (Join-Path $PSScriptRoot '..')

Write-Host '== cargo fmt --check ==' -ForegroundColor Cyan
cargo fmt --all -- --check
if ($LASTEXITCODE -ne 0) { Write-Host 'FAIL: fmt' -ForegroundColor Red; exit 1 }

Write-Host '== cargo clippy -D warnings ==' -ForegroundColor Cyan
cargo clippy --workspace --all-targets -- -D warnings
if ($LASTEXITCODE -ne 0) { Write-Host 'FAIL: clippy' -ForegroundColor Red; exit 1 }

Write-Host '== cargo build ==' -ForegroundColor Cyan
cargo build --workspace
if ($LASTEXITCODE -ne 0) { Write-Host 'FAIL: build' -ForegroundColor Red; exit 1 }

Write-Host '== cargo test ==' -ForegroundColor Cyan
cargo test --workspace
if ($LASTEXITCODE -ne 0) { Write-Host 'FAIL: test' -ForegroundColor Red; exit 1 }

Write-Host 'CI PASSED' -ForegroundColor Green
