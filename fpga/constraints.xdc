# ==============================================================================
# DPU Custom YOLOv4-tiny — Timing & Pin Constraints
# Target: Zynq-7020 (ZedBoard xc7z020clg484-1)
#
# For PS-PL integration, clock comes from PS FCLK_CLK0.
# No external pin constraints needed when using Zynq block design.
# This file provides timing constraints only.
# ==============================================================================

# ------------------------------------------------------------------------------
# Clock definition (PS FCLK_CLK0 at 100 MHz)
# ------------------------------------------------------------------------------
create_clock -period 10.000 -name aclk [get_ports aclk]

# Clock uncertainty for setup/hold
set_clock_uncertainty -setup 0.5 [get_clocks aclk]
set_clock_uncertainty -hold  0.1 [get_clocks aclk]

# ------------------------------------------------------------------------------
# Input/Output delays (AXI interface from PS)
# Assume 2ns max from PS GP port
# ------------------------------------------------------------------------------
set_input_delay  -clock aclk -max 2.0 [get_ports s_axi_*]
set_input_delay  -clock aclk -min 0.5 [get_ports s_axi_*]
set_output_delay -clock aclk -max 2.0 [get_ports s_axi_*]
set_output_delay -clock aclk -min 0.5 [get_ports s_axi_*]

# AXI-Stream (DMA)
set_input_delay  -clock aclk -max 2.0 [get_ports s_axis_*]
set_input_delay  -clock aclk -min 0.5 [get_ports s_axis_*]
set_output_delay -clock aclk -max 2.0 [get_ports m_axis_*]
set_output_delay -clock aclk -min 0.5 [get_ports m_axis_*]

# IRQ output
set_output_delay -clock aclk -max 2.0 [get_ports irq]
set_output_delay -clock aclk -min 0.5 [get_ports irq]

# Reset (async, but synchronize internally)
set_false_path -from [get_ports aresetn]

# ------------------------------------------------------------------------------
# Area / Pblock constraints (optional, uncomment if needed)
# ------------------------------------------------------------------------------
# create_pblock pblock_dpu
# resize_pblock pblock_dpu -add CLOCKREGION_X0Y0:CLOCKREGION_X1Y1
# add_cells_to_pblock pblock_dpu [get_cells u_dpu]

# ------------------------------------------------------------------------------
# Memory inference hints
# ------------------------------------------------------------------------------
# Weight buffer (144KB) -> BRAM
# Feature map buffers (64KB x 2) -> BRAM
# These should be automatically inferred as BRAM by Vivado.
# If not, use:
# set_property RAM_STYLE block [get_cells u_dpu/u_dpu/weight_buf_reg]
# set_property RAM_STYLE block [get_cells u_dpu/u_dpu/fmap_a_reg]
# set_property RAM_STYLE block [get_cells u_dpu/u_dpu/fmap_b_reg]
