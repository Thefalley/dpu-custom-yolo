# ==============================================================================
# Vitis — Create Application Project
# Creates the bare-metal DPU inference application.
#
# Usage:
#   xsct create_app.tcl
#
# Prerequisites:
#   - Run create_platform.tcl first
#   - sw/hal/ and sw/driver/ and sw/app/ source files present
# ==============================================================================

set script_dir [file dirname [file normalize [info script]]]
set ws_dir     "$script_dir/workspace"
set sw_dir     [file normalize "$script_dir/.."]

puts "============================================================"
puts "  Creating DPU Application"
puts "============================================================"

# Set workspace
setws $ws_dir

# Application project
set app_name "dpu_inference"
set pfm_name "dpu_platform"

# Remove old app if exists
catch { deleteproject $app_name }

# Create application (Empty C project)
app create -name $app_name -platform $pfm_name \
           -proc ps7_cortexa9_0 -os standalone \
           -template {Empty Application(C)}

# Import source files
# HAL
importsources -name $app_name -path $sw_dir/hal -soft-link
# Driver
importsources -name $app_name -path $sw_dir/driver -soft-link
# Application
importsources -name $app_name -path $sw_dir/app -soft-link

# Add include paths
app config -name $app_name -add include-path $sw_dir/hal
app config -name $app_name -add include-path $sw_dir/driver

# Optimization
app config -name $app_name -set compiler-optimization {Optimize most (-O3)}

# Build
puts "Building application..."
app build -name $app_name

set elf_path "$ws_dir/$app_name/Debug/$app_name.elf"
if {[file exists $elf_path]} {
    puts "============================================================"
    puts "  Application built: $elf_path"
    puts "============================================================"
} else {
    puts "WARNING: ELF not found at expected path."
    puts "Check build output for errors."
}
