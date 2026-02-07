# Run all DPU RTL simulations: Phase 7 (primitives) + Layer0 patch (4 channels).
# Set OSS_CAD_PATH below to your OSS CAD Suite or Icarus install so iverilog/vvp are found.
# Then run: .\run_all_sims.ps1

param(
    [string]$OSS_CAD_PATH = $env:OSS_CAD_PATH
)

$ProjectRoot = $PSScriptRoot

# If OSS_CAD_PATH not set: try .oss_cad_path file (one line = path), then default empty
if ($OSS_CAD_PATH -eq "") {
    $pathFile = Join-Path $ProjectRoot ".oss_cad_path"
    if (Test-Path $pathFile) {
        $line = (Get-Content $pathFile | Where-Object { $_ -notmatch '^\s*#' -and $_.Trim() -ne '' } | Select-Object -First 1)
        if ($line) { $OSS_CAD_PATH = $line.Trim() }
    }
}
if ($OSS_CAD_PATH -eq "") {
    # Or set here: $OSS_CAD_PATH = "C:\project\upm\oss-cad-suite\oss-cad-suite"
    $OSS_CAD_PATH = ""
}

if ($OSS_CAD_PATH -ne "") {
    $binPath = Join-Path $OSS_CAD_PATH "bin"
    $libPath = Join-Path $OSS_CAD_PATH "lib"
    if (Test-Path $binPath) {
        $env:PATH = "$binPath;$libPath;$env:PATH"
        $env:OSS_CAD_PATH = $OSS_CAD_PATH
        Write-Host "[PATH] OSS CAD / Icarus: $OSS_CAD_PATH"
    }
}

Set-Location $ProjectRoot

Write-Host ""
Write-Host "========== Phase 7: AutoCheck primitives (mac_int8, leaky_relu, mult_shift_add) =========="
$r7 = 0
& python run_phase7_autocheck.py
if ($LASTEXITCODE -ne 0) { $r7 = 1 }

Write-Host ""
Write-Host "========== Layer0 patch (one pixel, 4 channels) =========="
$rL = 0
& python run_layer0_patch_check.py
if ($LASTEXITCODE -ne 0) { $rL = 1 }

Write-Host ""
if ($r7 -eq 0 -and $rL -eq 0) {
    Write-Host "ALL SIMULATIONS PASSED"
    exit 0
} else {
    Write-Host "SOME SIMULATIONS FAILED (Phase7 exit=$r7, Layer0 patch exit=$rL)"
    if ($OSS_CAD_PATH -eq "" -and $env:OSS_CAD_PATH -eq "") {
        Write-Host "Tip: set OSS_CAD_PATH in this script or in the environment to point to OSS CAD Suite or Icarus (bin must contain iverilog and vvp)."
    }
    exit 1
}
