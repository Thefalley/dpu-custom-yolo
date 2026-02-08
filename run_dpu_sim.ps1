# Run DPU RTL simulation with verilog-sim-py + OSS CAD Suite
# Usage: .\run_dpu_sim.ps1 [mac_int8 | leaky_relu | mult_shift_add | dpu_layer0_cmd_4x4]
# Before first run: set OSS_CAD_PATH below if OSS CAD is not inside verilog-sim-py

param(
    [Parameter(Position=0)]
    [ValidateSet("mac_int8", "leaky_relu", "mult_shift_add", "dpu_layer0_cmd_4x4")]
    [string]$Test = "mac_int8"
)

$ProjectRoot = $PSScriptRoot
$VerilogSimPy = Join-Path $ProjectRoot "verilog-sim-py"

# OSS CAD Suite: use this if you have it elsewhere (e.g. C:\project\upm\oss-cad-suite\oss-cad-suite)
# Otherwise leave empty to use verilog-sim-py's default (verilog-sim-py/oss-cad-suite/oss-cad-suite)
$OSS_CAD_PATH = ""
if ($OSS_CAD_PATH -ne "") {
    $env:PATH = "$OSS_CAD_PATH\bin;$OSS_CAD_PATH\lib;$env:PATH"
    Write-Host "[PATH] OSS CAD Suite: $OSS_CAD_PATH"
}

# Icarus NO soporta program/clocking -> usamos TB *_iv.sv (misma disciplina con tasks + @(posedge clk))
$Tests = @{
    mac_int8 = @{
        Rtl  = @("rtl/dpu/primitives/mac_int8.sv")
        Tb   = "rtl/tb/mac_int8_tb_iv.sv"
        Top  = "mac_int8_tb"
    }
    leaky_relu = @{
        Rtl  = @("rtl/dpu/primitives/leaky_relu.sv")
        Tb   = "rtl/tb/leaky_relu_tb_iv.sv"
        Top  = "leaky_relu_tb"
    }
    mult_shift_add = @{
        Rtl  = @("rtl/dpu/primitives/mult_shift_add.sv")
        Tb   = "rtl/tb/mult_shift_add_tb_iv.sv"
        Top  = "mult_shift_add_tb"
    }
    dpu_layer0_cmd_4x4 = @{
        Rtl  = @(
            "rtl/dpu/primitives/mac_int8.sv",
            "rtl/dpu/primitives/leaky_relu.sv",
            "rtl/dpu/primitives/requantize.sv",
            "rtl/dpu/layer0_engine.sv",
            "rtl/dpu/dpu_layer0_top.sv"
        )
        Tb   = "rtl/tb/dpu_layer0_cmd_tb_4x4.sv"
        Top  = "dpu_layer0_cmd_tb_4x4"
    }
}

$info = $Tests[$Test]
$tbPath   = Join-Path $ProjectRoot $info.Tb
$svPy     = Join-Path $VerilogSimPy "sv_simulator.py"

if (-not (Test-Path $svPy)) {
    Write-Error "verilog-sim-py not found. Clone it: git clone https://github.com/Thefalley/verilog-sim-py.git"
    exit 1
}
$rtlPaths = @()
foreach ($r in $info.Rtl) {
    $p = Join-Path $ProjectRoot $r
    if (-not (Test-Path $p)) { Write-Error "RTL not found: $p"; exit 1 }
    $rtlPaths += $p
}
if (-not (Test-Path $tbPath)) { Write-Error "TB not found: $tbPath"; exit 1 }

Set-Location $ProjectRoot
Write-Host "[RUN] Test: $Test | Top: $($info.Top)"
& python $svPy --no-wave $rtlPaths $tbPath --top $info.Top
exit $LASTEXITCODE
