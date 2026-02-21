# ==============================================================================
# Export Hardware Platform (XSA) for Vitis
# Run after build_all.tcl or when you want to re-export.
#
# Usage:
#   vivado -mode batch -source export_hw.tcl
# ==============================================================================

set script_dir [file dirname [file normalize [info script]]]
set proj_dir   "$script_dir/vivado_project"
set xsa_path   "$proj_dir/dpu_yolov4_tiny.xsa"

# Open project
open_project $proj_dir/dpu_yolov4_tiny.xpr

# Make sure implementation is complete
if {[get_property STATUS [get_runs impl_1]] != "write_bitstream Complete!"} {
    puts "Implementation not complete. Running..."
    launch_runs impl_1 -to_step write_bitstream -jobs 4
    wait_on_run impl_1
}

# Export XSA (includes bitstream)
write_hw_platform -fixed -include_bit -force -file $xsa_path

puts "Hardware platform exported: $xsa_path"
close_project
