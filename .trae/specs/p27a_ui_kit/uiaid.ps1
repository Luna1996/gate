# 2.7a 实机验收辅助：窗口定位 / 截图 / resize / SendInput 拖拽 / 点击
# 用法：powershell -File uiaid.ps1 -Action shot -Path shot1.png
param(
  [string]$Action = "find",
  [string]$Path = "shot.png",
  [int]$X = 0, [int]$Y = 0, [int]$W = 0, [int]$H = 0,
  [int]$Dx = 0, [int]$Dy = 0,
  [string]$Button = "left"
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Drawing
Add-Type -AssemblyName System.Windows.Forms
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public static class W {
  [DllImport("user32.dll")] public static extern bool SetWindowPos(IntPtr h, IntPtr after, int x, int y, int w, int hgt, uint flags);
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
  [DllImport("user32.dll")] public static extern bool IsWindow(IntPtr h);
  [DllImport("user32.dll")] public static extern bool SetProcessDPIAware();
  [DllImport("user32.dll", SetLastError = true)] public static extern uint SendInput(uint n, INPUT[] inputs, int size);
  [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RECT r);
  [DllImport("user32.dll")] public static extern bool GetClientRect(IntPtr h, out RECT r);
  [StructLayout(LayoutKind.Sequential)] public struct RECT { public int L, T, R, B; }
  [StructLayout(LayoutKind.Sequential)] public struct INPUT { public uint type; public MOUSEINPUT mi; }
  [StructLayout(LayoutKind.Sequential)] public struct MOUSEINPUT { public int dx, dy; public uint mouseData, dwFlags, time; public IntPtr extraInfo; }
  public const uint MOVE = 0x0001, LEFTDOWN = 0x0002, LEFTUP = 0x0004, RIGHTDOWN = 0x0008, RIGHTUP = 0x0010;
  // 返回 int[] 规避 PS5.1 [ref] struct 写回坑（项目 memory 教训）
  public static int[] WinRect(IntPtr h) { RECT r; GetWindowRect(h, out r); return new int[] { r.L, r.T, r.R, r.B }; }
  public static int[] CliRect(IntPtr h) { RECT r; GetClientRect(h, out r); return new int[] { r.L, r.T, r.R, r.B }; }
  public static uint Send(uint flags, int dx, int dy) {
    var inp = new INPUT[1];
    inp[0].type = 0; inp[0].mi.dwFlags = flags; inp[0].mi.dx = dx; inp[0].mi.dy = dy; inp[0].mi.time = 0;
    return SendInput(1, inp, Marshal.SizeOf(typeof(INPUT)));
  }
  // 相对拖拽：按下 → 分步相对移动（每步 8px，WM_INPUT 供 Raw Input 捕获）→ 释放
  public static uint Drag(uint down, uint up, int dx, int dy) {
    Send(down, 0, 0);
    System.Threading.Thread.Sleep(60);
    int steps = Math.Max(Math.Abs(dx), Math.Abs(dy)) / 8 + 1;
    for (int i = 1; i <= steps; i++) {
      Send(MOVE, (dx * i) / steps - (dx * (i - 1)) / steps, (dy * i) / steps - (dy * (i - 1)) / steps);
      System.Threading.Thread.Sleep(8);
    }
    System.Threading.Thread.Sleep(60);
    Send(up, 0, 0);
    return 0;
  }
}
"@
[W]::SetProcessDPIAware() | Out-Null

function Get-AppWindow {
  $p = Get-Process gate-app -ErrorAction SilentlyContinue |
    Where-Object { $_.MainWindowHandle -ne 0 -and [W]::IsWindow($_.MainWindowHandle) } |
    Select-Object -First 1
  if ($p) { return $p.MainWindowHandle }
  return [IntPtr]::Zero
}

switch ($Action) {
  "find" {
    $h = Get-AppWindow
    if ($h -eq [IntPtr]::Zero) { Write-Output "NOWINDOW"; exit 1 }
    $wr = [W]::WinRect($h); $cr = [W]::CliRect($h)
    Write-Output ("handle={0} win={1},{2},{3},{4} client={5}x{6}" -f $h, $wr[0], $wr[1], $wr[2], $wr[3], ($cr[2] - $cr[0]), ($cr[3] - $cr[1]))
  }
  "shot" {
    $h = Get-AppWindow
    if ($h -eq [IntPtr]::Zero) { Write-Output "NOWINDOW"; exit 1 }
    [W]::SetForegroundWindow($h) | Out-Null
    Start-Sleep -Milliseconds 250
    $r = [W]::WinRect($h)
    $wd = $r[2] - $r[0]; $hg = $r[3] - $r[1]
    $bmp = New-Object System.Drawing.Bitmap($wd, $hg)
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.CopyFromScreen($r[0], $r[1], 0, 0, (New-Object System.Drawing.Size($wd, $hg)))
    $g.Dispose()
    $bmp.Save($Path, [System.Drawing.Imaging.ImageFormat]::Png)
    $bmp.Dispose()
    Write-Output ("saved {0} ({1}x{2})" -f $Path, $wd, $hg)
  }
  "resize" {
    $h = Get-AppWindow
    if ($h -eq [IntPtr]::Zero) { Write-Output "NOWINDOW"; exit 1 }
    [W]::SetWindowPos($h, [IntPtr]::Zero, $X, $Y, $W, $H, 0x0040) | Out-Null
    Start-Sleep -Milliseconds 600
    $cr = [W]::CliRect($h)
    Write-Output ("resized client={0}x{1}" -f ($cr[2] - $cr[0]), ($cr[3] - $cr[1]))
  }
  "move" {
    [System.Windows.Forms.Cursor]::Position = New-Object System.Drawing.Point($X, $Y)
    Start-Sleep -Milliseconds 120
    Write-Output ("cursor at {0},{1}" -f $X, $Y)
  }
  "drag" {
    $h = Get-AppWindow
    if ($h -eq [IntPtr]::Zero) { Write-Output "NOWINDOW"; exit 1 }
    [W]::SetForegroundWindow($h) | Out-Null
    Start-Sleep -Milliseconds 150
    if ($Button -eq "right") { [W]::Drag([W]::RIGHTDOWN, [W]::RIGHTUP, $Dx, $Dy) | Out-Null }
    else { [W]::Drag([W]::LEFTDOWN, [W]::LEFTUP, $Dx, $Dy) | Out-Null }
    Write-Output "dragged"
  }
  "click" {
    if ($Button -eq "right") { [W]::Send([W]::RIGHTDOWN, 0, 0) | Out-Null; Start-Sleep -Milliseconds 50; [W]::Send([W]::RIGHTUP, 0, 0) | Out-Null }
    else { [W]::Send([W]::LEFTDOWN, 0, 0) | Out-Null; Start-Sleep -Milliseconds 50; [W]::Send([W]::LEFTUP, 0, 0) | Out-Null }
    Write-Output "clicked"
  }
  default { Write-Output "unknown action"; exit 1 }
}
