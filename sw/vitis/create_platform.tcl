# ==============================================================================
# Vitis — Create Platform from XSA
# Creates a bare-metal (standalone) platform for the DPU hardware.
#
# Usage:
#   xsct create_platform.tcl
#   xsct create_platform.tcl /path/to/custom.xsa
#
# Prerequisites:
#   - Vitis 2022.2+ installed (xsct in PATH)
#   - XSA exported from Vivado (fpga/build_all.tcl or fpga/export_hw.tcl)
# ==============================================================================

set script_dir [file dirname [file normalize [info script]]]
set ws_dir     "$script_dir/workspace"
set proj_dir   [file normalize "$script_dir/../../fpga/vivado_project"]

# XSA path (default or command-line override)
if {$argc > 0} {
    set xsa_path [lindex $argv 0]
} else {
    set xsa_path "$proj_dir/dpu_yolov4_tiny.xsa"
}

if {![file exists $xsa_path]} {
    puts "ERROR: XSA not found at: $xsa_path"
    puts "Run fpga/build_all.tcl or fpga/export_hw.tcl first."
    exit 1
}

puts "============================================================"
puts "  Creating Vitis Platform"
puts "  XSA:       $xsa_path"
puts "  Workspace: $ws_dir"
puts "============================================================"

# Set workspace
setws $ws_dir

# Create platform
set pfm_name "dpu_platform"

# Remove old platform if exists
if {[llength [getprojects -type platform]] > 0} {
    deleteproject $pfm_name
}

platform create -name $pfm_name -hw $xsa_path -proc ps7_cortexa9_0 -os standalone

# Configure BSP
# Enable caches and FPU
bsp config stdin  ps7_uart_1
bsp config stdout ps7_uart_1

# Generate BSP and platform
platform generate

puts "============================================================"
puts "  Platform created: $pfm_name"
puts "  BSP generated for ps7_cortexa9_0 (standalone)"
puts "============================================================"
