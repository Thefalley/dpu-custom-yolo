# ==============================================================================
# Program FPGA via JTAG — Vivado Hardware Manager
# Programs the ZedBoard with the DPU bitstream.
#
# Usage:
#   vivado -mode batch -source program_fpga.tcl
#   vivado -mode batch -source program_fpga.tcl -tclargs /path/to/custom.bit
#
# Prerequisites:
#   - ZedBoard connected via JTAG (USB)
#   - Vivado Hardware Server running (auto-started)
# ==============================================================================

set script_dir [file dirname [file normalize [info script]]]
set proj_dir   "$script_dir/vivado_project"

# Bitstream path (default or command-line override)
if {$argc > 0} {
    set bit_file [lindex $argv 0]
} else {
    # Find bitstream from project
    set bit_file [glob -nocomplain $proj_dir/dpu_yolov4_tiny.runs/impl_1/*.bit]
    if {[llength $bit_file] == 0} {
        puts "ERROR: No bitstream found. Run build_all.tcl first."
        exit 1
    }
    set bit_file [lindex $bit_file 0]
}

puts "============================================================"
puts "  Programming FPGA"
puts "  Bitstream: $bit_file"
puts "============================================================"

# Connect to hardware
open_hw_manager
connect_hw_server -allow_non_jtag

# Find target (ZedBoard = Zynq xc7z020)
open_hw_target

set hw_device [get_hw_devices xc7z020_1]
if {[llength $hw_device] == 0} {
    puts "ERROR: Zynq device not found. Check JTAG connection."
    close_hw_target
    disconnect_hw_server
    close_hw_manager
    exit 1
}

current_hw_device $hw_device
set_property PROGRAM.FILE $bit_file $hw_device

# Program
puts "Programming..."
program_hw_devices $hw_device

puts "FPGA programmed successfully!"

# Clean up
close_hw_target
disconnect_hw_server
close_hw_manager
