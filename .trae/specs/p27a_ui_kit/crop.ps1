param([string]$In, [string]$Out, [int]$X, [int]$Y, [int]$W, [int]$H, [int]$Scale = 2)
Add-Type -AssemblyName System.Drawing
$src = [System.Drawing.Bitmap]::FromFile($In)
$rect = New-Object System.Drawing.Rectangle($X, $Y, $W, $H)
$crop = $src.Clone($rect, $src.PixelFormat)
$big = New-Object System.Drawing.Bitmap($crop, [int]($W * $Scale), [int]($H * $Scale))
$big.Save($Out, [System.Drawing.Imaging.ImageFormat]::Png)
$src.Dispose(); $crop.Dispose(); $big.Dispose()
Write-Output "cropped $Out"
