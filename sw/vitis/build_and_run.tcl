# ==============================================================================
# Vitis — Build, Program, and Run on ZedBoard
# Complete automated deployment: FPGA + ELF -> JTAG -> Run
#
# Usage:
#   xsct build_and_run.tcl
#   xsct build_and_run.tcl --program-only    # Skip build, just program
#   xsct build_and_run.tcl --build-only      # Skip programming
#
# Prerequisites:
#   - create_platform.tcl and create_app.tcl already run
#   - ZedBoard connected via JTAG
# ==============================================================================

set script_dir [file dirname [file normalize [info script]]]
set ws_dir     "$script_dir/workspace"
set app_name   "dpu_inference"
set pfm_name   "dpu_platform"

# Parse args
set do_build   1
set do_program 1
if {$argc > 0} {
    if {[lindex $argv 0] == "--program-only"} { set do_build 0 }
    if {[lindex $argv 0] == "--build-only"}   { set do_program 0 }
}

# Set workspace
setws $ws_dir

puts "============================================================"
puts "  DPU Deployment to ZedBoard"
puts "  Build: [expr {$do_build ? "YES" : "SKIP"}]"
puts "  Program: [expr {$do_program ? "YES" : "SKIP"}]"
puts "============================================================"

# ---- Build ----
if {$do_build} {
    puts "\n===== Building Platform ====="
    platform generate -name $pfm_name

    puts "\n===== Building Application ====="
    app build -name $app_name
}

# ---- Program & Run ----
if {$do_program} {
    # Find bitstream and ELF
    set bit_file [glob -nocomplain $ws_dir/$pfm_name/hw/*.bit]
    set elf_file "$ws_dir/$app_name/Debug/$app_name.elf"

    if {[llength $bit_file] == 0} {
        puts "ERROR: Bitstream not found in platform."
        exit 1
    }
    set bit_file [lindex $bit_file 0]

    if {![file exists $elf_file]} {
        puts "ERROR: ELF not found: $elf_file"
        exit 1
    }

    puts "\n===== Programming ZedBoard ====="
    puts "  Bitstream: $bit_file"
    puts "  ELF:       $elf_file"

    # Connect to target
    connect

    # Find Zynq target
    set targets_list [targets]
    puts "Available targets: $targets_list"

    # Select APU (ARM Cortex-A9 #0)
    targets -set -filter {name =~ "ARM*#0"}

    # Reset system
    rst -system

    # Program FPGA
    puts "Programming FPGA..."
    fpga $bit_file

    # Configure PS7 init (from XSA)
    # This sets up DDR, MIO, clocks etc
    set ps7_init [glob -nocomplain $ws_dir/$pfm_name/hw/*ps7_init.tcl]
    if {[llength $ps7_init] > 0} {
        source [lindex $ps7_init 0]
        ps7_init
        ps7_post_config
    }

    # Download and run ELF
    puts "Downloading ELF..."
    dow $elf_file

    puts "Starting execution..."
    con

    # Wait for completion (poll UART for "=== Done ===")
    puts "\n===== Application Running ====="
    puts "Monitor UART output (115200 baud) for results."
    puts "Use 'stop' in xsct to halt, 'con' to continue."

    # Give it some time to run
    after 10000

    # Stop and read output
    stop
    puts "\n===== Execution stopped ====="
}

puts "\n============================================================"
puts "  Deployment complete."
puts "============================================================"
