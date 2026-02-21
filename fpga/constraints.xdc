# ==============================================================================
# DPU Custom YOLOv4-tiny — Timing & Pin Constraints
# Target: Zynq-7020 (ZedBoard xc7z020clg484-1)
# Design: Full 36-layer YOLOv4-tiny (H0=32, W0=32, MAX_CH=512)
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
# Memory inference — BRAM directives
# ------------------------------------------------------------------------------
# Weight buffer (~2.4 MB) — may exceed on-chip BRAM capacity on Zynq-7020
# (140 BRAM36K = 630 KB). Consider external memory or weight streaming.
# For now, let Vivado decide optimal mapping.
#
# Feature map buffers (64 KB x 2 = 128 KB) -> BRAM
# set_property RAM_STYLE block [get_cells -hierarchical -filter {NAME =~ *fmap_a*}]
# set_property RAM_STYLE block [get_cells -hierarchical -filter {NAME =~ *fmap_b*}]

# ------------------------------------------------------------------------------
# Area / Pblock constraints (optional, uncomment if needed)
# ------------------------------------------------------------------------------
# create_pblock pblock_dpu
# resize_pblock pblock_dpu -add CLOCKREGION_X0Y0:CLOCKREGION_X1Y1
# add_cells_to_pblock pblock_dpu [get_cells u_dpu]
