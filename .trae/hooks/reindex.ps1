# Stop / SessionStart hook：仅当工作区文件相对上次索引有变化时，才重建 codebase-memory 图谱。
# 指纹 = 仓库内（排除 target/data/logs/.git 等）所有文件的路径 + mtime + size；无变化直接退出。

$ErrorActionPreference = 'SilentlyContinue'

$repo = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$exe = 'C:\Users\WYF\AppData\Local\Programs\codebase-memory-mcp\codebase-memory-mcp.exe'

$skipDirs = @('.git', '.vscode', 'target', 'data', 'logs', '.codebase-memory', 'dist')
$skipRe = '\\(\.git|\.vscode|target|data|logs|\.codebase-memory|dist)\\'

$files = New-Object System.Collections.Generic.List[object]
Get-ChildItem -Path $repo -File -Force | ForEach-Object { $files.Add($_) }
foreach ($dir in Get-ChildItem -Path $repo -Directory -Force) {
  if ($skipDirs -contains $dir.Name) { continue }
  Get-ChildItem -Path $dir.FullName -Recurse -File -Force | ForEach-Object { $files.Add($_) }
}

$sb = New-Object System.Text.StringBuilder
foreach ($f in ($files | Where-Object { $_.FullName -notmatch $skipRe } | Sort-Object FullName)) {
  [void]$sb.Append($f.FullName).Append('|').Append($f.LastWriteTimeUtc.Ticks).Append('|').Append($f.Length).Append("`n")
}

$sha = [System.Security.Cryptography.SHA256]::Create()
$fp = [BitConverter]::ToString($sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($sb.ToString()))).Replace('-', '')

# 指纹存档放在工具缓存目录，不污染仓库
$stampDir = Join-Path $env:USERPROFILE '.cache\codebase-memory-mcp'
if (-not (Test-Path $stampDir)) { New-Item -ItemType Directory -Path $stampDir -Force | Out-Null }
$id = [BitConverter]::ToString($sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($repo.ToLowerInvariant()))).Replace('-', '').Substring(0, 12)
$stamp = Join-Path $stampDir "hook-fingerprint-$id.txt"

$prev = ''
if (Test-Path $stamp) { $prev = (Get-Content $stamp -Raw).Trim() }

if ($fp -ne $prev) {
  & $exe cli --quiet index_repository --repo-path $repo --mode full --persistence true | Out-Null
  if ($LASTEXITCODE -eq 0) { Set-Content -Path $stamp -Value $fp -NoNewline -Encoding ascii }
}

exit 0
