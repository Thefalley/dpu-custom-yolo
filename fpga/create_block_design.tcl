# ==============================================================================
# DPU Zynq Block Design — Creates PS-PL interconnect
# Target: Zynq-7020 (ZedBoard), Vivado 2022.2+
#
# Run AFTER synth.tcl creates the project:
#   vivado -mode batch -source create_block_design.tcl
#
# Creates a block design with:
#   - Zynq PS (FCLK_CLK0=100MHz, GP0, HP0)
#   - DPU system top (AXI4-Lite slave on GP0, 36-layer)
#   - AXI DMA (MM2S->DPU S_AXIS, S2MM<-DPU M_AXIS, on HP0)
#   - Interrupt concatenation -> PS IRQ_F2P
# ==============================================================================

set script_dir [file dirname [file normalize [info script]]]

# Open existing project
open_project $script_dir/vivado_project/dpu_yolov4_tiny.xpr

# Delete old block design if it exists
if {[llength [get_bd_designs -quiet dpu_bd]] > 0} {
    delete_bd_design dpu_bd
}

# Create block design
create_bd_design "dpu_bd"

# ==== Zynq Processing System ====
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 ps7
# Configure PS
set_property -dict [list \
    CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {100} \
    CONFIG.PCW_USE_S_AXI_HP0 {1} \
    CONFIG.PCW_USE_FABRIC_INTERRUPT {1} \
    CONFIG.PCW_IRQ_F2P_INTR {1} \
] [get_bd_cells ps7]
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
    -config {make_external "FIXED_IO, DDR" } [get_bd_cells ps7]

# ==== DPU System Top (RTL module) ====
create_bd_cell -type module -reference dpu_system_top dpu_sys

# ==== AXI Interconnect for GP0 -> DPU AXI-Lite + DMA control ====
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_gp0_ic
set_property CONFIG.NUM_MI {2} [get_bd_cells axi_gp0_ic]

# Connect GP0 -> interconnect
connect_bd_intf_net [get_bd_intf_pins ps7/M_AXI_GP0] \
                    [get_bd_intf_pins axi_gp0_ic/S00_AXI]
# M00 -> DPU AXI-Lite
connect_bd_intf_net [get_bd_intf_pins axi_gp0_ic/M00_AXI] \
                    [get_bd_intf_pins dpu_sys/s_axi]

# ==== AXI DMA (Simple DMA, no scatter-gather) ====
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma
set_property -dict [list \
    CONFIG.c_include_sg {0} \
    CONFIG.c_sg_include_stscntrl_strm {0} \
    CONFIG.c_mm2s_burst_size {16} \
    CONFIG.c_s2mm_burst_size {16} \
    CONFIG.c_m_axi_mm2s_data_width {32} \
    CONFIG.c_m_axi_s2mm_data_width {32} \
    CONFIG.c_m_axis_mm2s_tdata_width {32} \
    CONFIG.c_s_axis_s2mm_tdata_width {32} \
] [get_bd_cells axi_dma]

# DMA MM2S -> DPU S_AXIS (data in)
connect_bd_intf_net [get_bd_intf_pins axi_dma/M_AXIS_MM2S] \
                    [get_bd_intf_pins dpu_sys/s_axis]
# DPU M_AXIS -> DMA S2MM (data out)
connect_bd_intf_net [get_bd_intf_pins dpu_sys/m_axis] \
                    [get_bd_intf_pins axi_dma/S_AXIS_S2MM]

# DMA control port on GP0 interconnect (M01)
connect_bd_intf_net [get_bd_intf_pins axi_gp0_ic/M01_AXI] \
                    [get_bd_intf_pins axi_dma/S_AXI_LITE]

# ==== HP0 Interconnect for DMA memory access ====
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_hp0_ic
set_property CONFIG.NUM_SI {2} [get_bd_cells axi_hp0_ic]
connect_bd_intf_net [get_bd_intf_pins axi_dma/M_AXI_MM2S] \
                    [get_bd_intf_pins axi_hp0_ic/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_dma/M_AXI_S2MM] \
                    [get_bd_intf_pins axi_hp0_ic/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_hp0_ic/M00_AXI] \
                    [get_bd_intf_pins ps7/S_AXI_HP0]

# ==== Clocks and Resets ====
connect_bd_net [get_bd_pins ps7/FCLK_CLK0] \
    [get_bd_pins dpu_sys/aclk] \
    [get_bd_pins axi_gp0_ic/ACLK] [get_bd_pins axi_gp0_ic/S00_ACLK] \
    [get_bd_pins axi_gp0_ic/M00_ACLK] [get_bd_pins axi_gp0_ic/M01_ACLK] \
    [get_bd_pins axi_hp0_ic/ACLK] [get_bd_pins axi_hp0_ic/S00_ACLK] \
    [get_bd_pins axi_hp0_ic/S01_ACLK] [get_bd_pins axi_hp0_ic/M00_ACLK] \
    [get_bd_pins axi_dma/s_axi_lite_aclk] [get_bd_pins axi_dma/m_axi_mm2s_aclk] \
    [get_bd_pins axi_dma/m_axi_s2mm_aclk]

# Processor system reset
create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 ps_reset
connect_bd_net [get_bd_pins ps7/FCLK_CLK0] [get_bd_pins ps_reset/slowest_sync_clk]
connect_bd_net [get_bd_pins ps7/FCLK_RESET0_N] [get_bd_pins ps_reset/ext_reset_in]

connect_bd_net [get_bd_pins ps_reset/peripheral_aresetn] \
    [get_bd_pins dpu_sys/aresetn] \
    [get_bd_pins axi_gp0_ic/ARESETN] [get_bd_pins axi_gp0_ic/S00_ARESETN] \
    [get_bd_pins axi_gp0_ic/M00_ARESETN] [get_bd_pins axi_gp0_ic/M01_ARESETN] \
    [get_bd_pins axi_hp0_ic/ARESETN] [get_bd_pins axi_hp0_ic/S00_ARESETN] \
    [get_bd_pins axi_hp0_ic/S01_ARESETN] [get_bd_pins axi_hp0_ic/M00_ARESETN] \
    [get_bd_pins axi_dma/axi_resetn]

# ==== Interrupt ====
create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 irq_concat
set_property CONFIG.NUM_PORTS {3} [get_bd_cells irq_concat]
connect_bd_net [get_bd_pins dpu_sys/irq]            [get_bd_pins irq_concat/In0]
connect_bd_net [get_bd_pins axi_dma/mm2s_introut]   [get_bd_pins irq_concat/In1]
connect_bd_net [get_bd_pins axi_dma/s2mm_introut]   [get_bd_pins irq_concat/In2]
connect_bd_net [get_bd_pins irq_concat/dout]         [get_bd_pins ps7/IRQ_F2P]

# ==== Address mapping ====
assign_bd_address
# DPU control registers
set_property range 256 [get_bd_addr_segs {ps7/Data/SEG_dpu_sys_reg0}]
set_property offset 0x43C00000 [get_bd_addr_segs {ps7/Data/SEG_dpu_sys_reg0}]

# ==== Validate and save ====
validate_bd_design
save_bd_design
generate_target all [get_files dpu_bd.bd]
make_wrapper -files [get_files dpu_bd.bd] -top

puts "Block design created: dpu_bd"
puts "DPU AXI-Lite base:  0x43C0_0000"
puts "AXI DMA base:       (auto-assigned)"
close_project
