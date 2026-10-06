$ErrorActionPreference = 'SilentlyContinue'

try {
  $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
  $skillPath = Join-Path $scriptDir '..\skills\i-have-adhd\SKILL.md'

  if (-not (Test-Path -LiteralPath $skillPath -PathType Leaf)) { exit 0 }

  $lines = [System.IO.File]::ReadAllLines($skillPath)
  $bodyStart = 0

  if ($lines.Length -gt 0 -and $lines[0] -match '^---\s*$') {
    for ($i = 1; $i -lt $lines.Length; $i++) {
      if ($lines[$i] -match '^---\s*$') {
        $bodyStart = $i + 1
        break
      }
    }
  }

  $body = if ($bodyStart -lt $lines.Length) {
    [string]::Join([Environment]::NewLine, $lines[$bodyStart..($lines.Length - 1)])
  } else {
    ''
  }

  $banner = 'ADHD MODE ACTIVE (always-on). The ruleset below applies to every response. ' +
    '"stop adhd mode" turns it off for this session; remove this hook from .trae/hooks.json to turn always-on off for good.'

  [Console]::Out.Write($banner + "`n`n" + $body + "`n")
} catch {
  exit 0
}
