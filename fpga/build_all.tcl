# ==============================================================================
# DPU YOLOv4-tiny — Master Build Script
# Runs the complete FPGA build flow:
#   1. Create Vivado project + synthesis + implementation + bitstream
#   2. Create Zynq block design
#   3. Export hardware (XSA) for Vitis
#
# Usage:
#   vivado -mode batch -source build_all.tcl
#
# Prerequisites:
#   - Vivado 2022.2+ installed
#   - Run from the fpga/ directory
# ==============================================================================

set script_dir [file dirname [file normalize [info script]]]
set proj_dir   "$script_dir/vivado_project"

puts "============================================================"
puts "  DPU YOLOv4-tiny — Full FPGA Build"
puts "  Target: ZedBoard (xc7z020clg484-1)"
puts "  Design: 36-layer YOLOv4-tiny (H0=32, W0=32)"
puts "============================================================"

# ------------------------------------------------------------------------------
# Step 1: Synthesis + Implementation + Bitstream
# ------------------------------------------------------------------------------
puts "\n===== Step 1: Synthesis + Implementation ====="
source $script_dir/synth.tcl
close_project

# ------------------------------------------------------------------------------
# Step 2: Block Design (Zynq PS-PL)
# ------------------------------------------------------------------------------
puts "\n===== Step 2: Block Design ====="
source $script_dir/create_block_design.tcl

# ------------------------------------------------------------------------------
# Step 3: Re-synthesize with block design wrapper as top
# ------------------------------------------------------------------------------
puts "\n===== Step 3: Re-run with BD wrapper ====="
open_project $proj_dir/dpu_yolov4_tiny.xpr

# Set block design wrapper as top
set bd_wrapper [get_files -of_objects [get_filesets sources_1] -filter {NAME =~ *dpu_bd_wrapper*}]
if {[llength $bd_wrapper] > 0} {
    set_property top dpu_bd_wrapper [current_fileset]
    update_compile_order -fileset sources_1

    # Re-run synth + impl with BD wrapper
    reset_run synth_1
    launch_runs synth_1 -jobs 4
    wait_on_run synth_1

    launch_runs impl_1 -to_step write_bitstream -jobs 4
    wait_on_run impl_1

    puts "Bitstream generated with block design wrapper"
} else {
    puts "WARNING: BD wrapper not found. Using standalone DPU top."
}

# ------------------------------------------------------------------------------
# Step 4: Export Hardware (XSA for Vitis)
# ------------------------------------------------------------------------------
puts "\n===== Step 4: Export Hardware ====="
set xsa_path "$proj_dir/dpu_yolov4_tiny.xsa"
write_hw_platform -fixed -include_bit -force -file $xsa_path
puts "Hardware exported: $xsa_path"

close_project

puts "\n============================================================"
puts "  BUILD COMPLETE"
puts "  Bitstream: $proj_dir/dpu_yolov4_tiny.runs/impl_1/*.bit"
puts "  XSA:       $xsa_path"
puts "============================================================"
