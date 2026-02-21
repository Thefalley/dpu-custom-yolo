# ==============================================================================
# DPU Custom YOLOv4-tiny — Vivado Synthesis Script
# Target: Zynq-7020 (ZedBoard xc7z020clg484-1)
# Vivado version: 2022.2+
#
# Usage:
#   vivado -mode batch -source synth.tcl
#   or from Vivado GUI: Tools > Run Tcl Script
# ==============================================================================

# ------------------------------------------------------------------------------
# Project settings
# ------------------------------------------------------------------------------
set proj_name  "dpu_yolov4_tiny"
set proj_dir   "./vivado_project"
set part       "xc7z020clg484-1"       ;# ZedBoard Zynq-7020
set top_module "dpu_system_top"
set clk_mhz   100                       ;# Target clock (PS FCLK_CLK0)

# DPU parameters — full 36-layer YOLOv4-tiny (match golden model)
set H0         32
set W0         32
set MAX_CH     512
set NUM_LAYERS 36
set MAX_WBUF   2400000

# Derived paths
set script_dir [file dirname [file normalize [info script]]]
set rtl_dir    [file normalize "$script_dir/../rtl/dpu"]

# ------------------------------------------------------------------------------
# Create project
# ------------------------------------------------------------------------------
create_project $proj_name $proj_dir -part $part -force
set_property target_language Verilog [current_project]

# ------------------------------------------------------------------------------
# Add RTL sources
# ------------------------------------------------------------------------------
add_files [list \
    $rtl_dir/primitives/mac_int8.sv \
    $rtl_dir/primitives/leaky_relu.sv \
    $rtl_dir/primitives/requantize.sv \
    $rtl_dir/mac_array_32x32.sv \
    $rtl_dir/post_process_array.sv \
    $rtl_dir/maxpool_unit.sv \
    $rtl_dir/conv_engine_array.sv \
    $rtl_dir/dpu_top.sv \
    $rtl_dir/dpu_axi4_lite.sv \
    $rtl_dir/dpu_axi_dma.sv \
    $rtl_dir/dpu_system_top.sv \
]

set_property file_type SystemVerilog [get_files *.sv]

# ------------------------------------------------------------------------------
# Add constraints
# ------------------------------------------------------------------------------
add_files -fileset constrs_1 [file normalize "$script_dir/constraints.xdc"]

# ------------------------------------------------------------------------------
# Set top module and generics
# ------------------------------------------------------------------------------
set_property top $top_module [current_fileset]
set_property generic "H0=$H0 W0=$W0 MAX_CH=$MAX_CH NUM_LAYERS=$NUM_LAYERS MAX_WBUF=$MAX_WBUF" [current_fileset]

# ------------------------------------------------------------------------------
# Synthesis settings
# ------------------------------------------------------------------------------
set_property strategy Flow_PerfOptimized_high [get_runs synth_1]
set_property STEPS.SYNTH_DESIGN.ARGS.FLATTEN_HIERARCHY rebuilt [get_runs synth_1]
set_property STEPS.SYNTH_DESIGN.ARGS.RETIMING on [get_runs synth_1]

# Force BRAM for large memories (weight_buf, fmap_a, fmap_b)
# Vivado should infer these automatically, but we add directives as safety
set_property STEPS.SYNTH_DESIGN.ARGS.MORE_OPTIONS {-ram_style block} [get_runs synth_1]

# ------------------------------------------------------------------------------
# Implementation settings
# ------------------------------------------------------------------------------
set_property strategy Performance_ExplorePostRoutePhysOpt [get_runs impl_1]

# ------------------------------------------------------------------------------
# Run synthesis
# ------------------------------------------------------------------------------
puts "===== Starting Synthesis ====="
puts "Parameters: H0=$H0 W0=$W0 MAX_CH=$MAX_CH NUM_LAYERS=$NUM_LAYERS"
launch_runs synth_1 -jobs 4
wait_on_run synth_1

if {[get_property STATUS [get_runs synth_1]] != "synth_design Complete!"} {
    puts "ERROR: Synthesis failed!"
    exit 1
}

open_run synth_1

# Report utilization
report_utilization -file $proj_dir/utilization_synth.rpt
report_timing_summary -file $proj_dir/timing_synth.rpt
puts "Synthesis utilization report: $proj_dir/utilization_synth.rpt"

# ------------------------------------------------------------------------------
# Run implementation + bitstream
# ------------------------------------------------------------------------------
puts "===== Starting Implementation ====="
launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1

if {[get_property STATUS [get_runs impl_1]] != "write_bitstream Complete!"} {
    puts "ERROR: Implementation/bitstream failed!"
    exit 1
}

# Reports
open_run impl_1
report_utilization -file $proj_dir/utilization_impl.rpt
report_timing_summary -file $proj_dir/timing_impl.rpt
report_power -file $proj_dir/power_impl.rpt

# Check timing
set wns [get_property STATS.WNS [get_runs impl_1]]
puts "===== Done ====="
puts "Utilization: $proj_dir/utilization_impl.rpt"
puts "Timing:      $proj_dir/timing_impl.rpt (WNS = ${wns} ns)"
puts "Power:       $proj_dir/power_impl.rpt"
puts "Bitstream:   $proj_dir/${proj_name}.runs/impl_1/${top_module}.bit"

if {$wns < 0} {
    puts "WARNING: Timing not met (WNS = ${wns} ns). Consider reducing clock."
}
