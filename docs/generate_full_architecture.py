#!/usr/bin/env python3
"""
Generate a comprehensive DPU architecture diagram showing the full system:
  - Zynq SoC (PS + PL)
  - AXI interconnect
  - DPU system top (AXI-Lite + DMA + core)
  - DPU core internals (FSM, memories, engine, maxpool)
  - Conv engine pipeline (MAC array + post-process)
  - 18-layer dataflow
  - Software stack
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ArrowStyle
import numpy as np

# =============================================================================
# Drawing helpers
# =============================================================================
def box(ax, x, y, w, h, label, color='#4A90D9', text_color='white',
        fontsize=8, sublabel=None, border_color=None, lw=1.5, alpha=0.95,
        zorder=2, fontstyle='normal', sublabel_fs=None, label_y_offset=0):
    if border_color is None:
        border_color = color
    if sublabel_fs is None:
        sublabel_fs = max(5, fontsize - 2)
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                        facecolor=color, edgecolor=border_color,
                        linewidth=lw, alpha=alpha, zorder=zorder)
    ax.add_patch(b)
    ty = y + h / 2 + label_y_offset
    if sublabel:
        ty = y + h * 0.62 + label_y_offset
        ax.text(x + w / 2, y + h * 0.30, sublabel, ha='center', va='center',
                fontsize=sublabel_fs, color=text_color, alpha=0.85, zorder=zorder+1,
                fontfamily='monospace', fontstyle=fontstyle)
    ax.text(x + w / 2, ty, label, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color=text_color, zorder=zorder+1)
    return b


def arrow(ax, x1, y1, x2, y2, color='#444', lw=1.2, style='->', label=None,
          label_offset=(0.03, 0), fontsize=6, connectionstyle=None, linestyle='-',
          zorder=1):
    props = dict(arrowstyle=style, color=color, lw=lw, linestyle=linestyle)
    if connectionstyle:
        props['connectionstyle'] = connectionstyle
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=props, zorder=zorder)
    if label:
        mx = (x1 + x2) / 2 + label_offset[0]
        my = (y1 + y2) / 2 + label_offset[1]
        ax.text(mx, my, label, fontsize=fontsize, color=color, ha='center', va='center',
                fontweight='bold', zorder=zorder+1,
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.8))


def section_bg(ax, x, y, w, h, color, border_color, lw=2, alpha=0.15, zorder=0):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                        facecolor=color, edgecolor=border_color,
                        linewidth=lw, alpha=alpha, zorder=zorder)
    ax.add_patch(b)
    return b


def section_label(ax, x, y, text, color='#333', fontsize=10):
    ax.text(x, y, text, fontsize=fontsize, fontweight='bold', color=color,
            va='center', ha='left', zorder=5)


# =============================================================================
# Main diagram
# =============================================================================
def create_full_architecture():
    fig, ax = plt.subplots(1, 1, figsize=(28, 20))
    ax.set_xlim(-0.5, 27.5)
    ax.set_ylim(-1.5, 18.5)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor('#FAFBFC')

    # =====================================================================
    # TITLE
    # =====================================================================
    ax.text(13.5, 18.1, 'DPU Custom YOLOv4-tiny — Full System Architecture',
            ha='center', va='center', fontsize=22, fontweight='bold', color='#1a1a2e')
    ax.text(13.5, 17.6, 'INT8 Fixed-Point | 32x32 MAC Array (1024 MACs/cyc) | 18-Layer Backbone | '
            '256-bit Wide Memory | AXI4 Zynq Interface',
            ha='center', va='center', fontsize=9, color='#666')

    # =====================================================================
    # 1. SOFTWARE STACK (top-left)
    # =====================================================================
    section_bg(ax, -0.3, 14.2, 6.8, 3.0, '#E8EAF6', '#5C6BC0', alpha=0.2)
    section_label(ax, 0.0, 16.9, 'SOFTWARE (ARM Cortex-A9)', '#3949AB', 11)

    box(ax, 0.0, 16.0, 6.2, 0.7, 'Application (main.c)',
        color='#3949AB', sublabel='Load image -> Run inference -> Read output', fontsize=8)
    box(ax, 0.0, 15.1, 6.2, 0.7, 'Driver (dpu_driver.c)',
        color='#5C6BC0', sublabel='load_weights / run_inference / read_output', fontsize=8)
    box(ax, 0.0, 14.3, 6.2, 0.6, 'HAL (dpu_hal.c)',
        color='#7986CB', sublabel='MMIO reg read/write | PIO cmd', fontsize=7)

    arrow(ax, 3.1, 16.0, 3.1, 15.8, color='#3949AB', lw=1)
    arrow(ax, 3.1, 15.1, 3.1, 14.9, color='#5C6BC0', lw=1)

    # =====================================================================
    # 2. ZYNQ PS (top-center)
    # =====================================================================
    section_bg(ax, 7.3, 14.2, 5.5, 3.0, '#FFF3E0', '#E65100', alpha=0.15)
    section_label(ax, 7.6, 16.9, 'ZYNQ-7020 PS', '#E65100', 11)

    box(ax, 7.5, 15.8, 2.3, 0.8, 'ARM Cortex-A9',
        color='#E65100', sublabel='Dual-core 667MHz', fontsize=8)
    box(ax, 10.2, 15.8, 2.3, 0.8, 'DDR3 Memory',
        color='#BF360C', sublabel='512 MB', fontsize=8)
    box(ax, 7.5, 14.5, 2.3, 0.65, 'M_AXI_GP0',
        color='#F57C00', sublabel='Control path', fontsize=7)
    box(ax, 10.2, 14.5, 2.3, 0.65, 'S_AXI_HP0',
        color='#F57C00', sublabel='Data path (DMA)', fontsize=7)

    # SW -> PS
    arrow(ax, 6.2, 15.4, 7.5, 15.4, color='#3949AB', lw=1.5, label='ABI')
    # PS internal
    arrow(ax, 9.8, 16.2, 10.2, 16.2, color='#E65100', lw=1)
    arrow(ax, 8.65, 15.8, 8.65, 15.15, color='#E65100', lw=1)
    arrow(ax, 11.35, 15.8, 11.35, 15.15, color='#E65100', lw=1)

    # =====================================================================
    # 3. POST-PROCESSING (top-right)
    # =====================================================================
    section_bg(ax, 14.0, 14.2, 5.5, 3.0, '#F3E5F5', '#7B1FA2', alpha=0.15)
    section_label(ax, 14.3, 16.9, 'POST-PROCESSING (Python/SW)', '#7B1FA2', 11)

    box(ax, 14.3, 16.0, 4.9, 0.65, 'Detection Head',
        color='#7B1FA2', sublabel='Conv1x1 + Upsample + Concat (6 layers)', fontsize=8)
    box(ax, 14.3, 15.2, 2.3, 0.6, 'YOLO Decode',
        color='#9C27B0', sublabel='Grid + Anchors + Sigmoid', fontsize=7)
    box(ax, 16.9, 15.2, 2.3, 0.6, 'NMS',
        color='#AB47BC', sublabel='IoU threshold 0.45', fontsize=7)
    box(ax, 14.3, 14.4, 4.9, 0.6, 'Output: Bounding Boxes + Classes + Confidence',
        color='#CE93D8', text_color='#4A148C', fontsize=7, sublabel='80 COCO classes')

    arrow(ax, 16.75, 16.0, 16.75, 15.8, color='#7B1FA2', lw=1)
    arrow(ax, 16.6, 15.2, 16.9, 15.4, color='#9C27B0', lw=1)
    arrow(ax, 16.75, 15.2, 16.75, 15.0, color='#AB47BC', lw=1)

    # =====================================================================
    # 4. FPGA INFO (top far-right)
    # =====================================================================
    section_bg(ax, 20.2, 14.2, 7.0, 3.0, '#E8F5E9', '#2E7D32', alpha=0.15)
    section_label(ax, 20.5, 16.9, 'FPGA TARGET', '#2E7D32', 11)

    info_lines = [
        ('Device:', 'Zynq xc7z020clg484-1 (ZedBoard)'),
        ('Clock:', '100 MHz (PS FCLK_CLK0)'),
        ('MACs:', '1024/cycle (32x32 INT8)'),
        ('Memory:', '272 KB BRAM (weight+fmap+patch)'),
        ('Perf 16x16:', '~1.2M cycles (12 ms)'),
        ('Perf 416x416:', '~15M cycles (150 ms est.)'),
        ('Total MACs:', '1.52 billion (416x416)'),
        ('Interface:', 'AXI4-Lite + AXI-Stream DMA'),
    ]
    for idx, (k, v) in enumerate(info_lines):
        yy = 16.35 - idx * 0.30
        ax.text(20.6, yy, k, fontsize=7, color='#1B5E20', fontweight='bold', va='center')
        ax.text(22.2, yy, v, fontsize=7, color='#333', fontfamily='monospace', va='center')

    # =====================================================================
    # 5. PL REGION — dpu_system_top (main area)
    # =====================================================================
    pl_bg = FancyBboxPatch((-0.3, 0.0), 27.3, 13.8, boxstyle="round,pad=0.15",
                            facecolor='#E3F2FD', edgecolor='#0D47A1',
                            linewidth=3, alpha=0.12, zorder=0)
    ax.add_patch(pl_bg)
    ax.text(13.5, 13.45, 'PROGRAMMABLE LOGIC — dpu_system_top.sv',
            ha='center', fontsize=13, fontweight='bold', color='#0D47A1')

    # ----- AXI4-Lite Slave -----
    box(ax, 0.2, 11.8, 3.5, 1.2, 'AXI4-Lite Slave',
        color='#0D47A1', fontsize=10,
        sublabel='Register Map (0x00-0x34)\nCMD | ADDR | WDATA | STATUS | PERF | IRQ')
    # Arrow from PS GP0 down to AXI-Lite
    arrow(ax, 8.65, 14.5, 2.0, 13.0, color='#F57C00', lw=2, label='GP0')

    # ----- AXI DMA Engine -----
    box(ax, 4.2, 11.8, 3.2, 1.2, 'AXI-Stream DMA',
        color='#01579B', fontsize=10,
        sublabel='S_AXIS (in) | M_AXIS (out)\nByte unpacking + PIO bridge')
    # Arrow from PS HP0 to DMA
    arrow(ax, 11.35, 14.5, 5.8, 13.0, color='#F57C00', lw=2, label='HP0')

    # ----- IRQ -----
    box(ax, 7.8, 12.2, 1.4, 0.6, 'IRQ',
        color='#C62828', fontsize=7, sublabel='done | reload')
    arrow(ax, 9.2, 12.5, 10.0, 14.5, color='#C62828', lw=1.2, label='IRQ_F2P',
          connectionstyle='arc3,rad=-0.2')

    # ----- PIO Arbiter -----
    box(ax, 3.0, 10.5, 3.5, 0.8, 'PIO Arbiter',
        color='#37474F', fontsize=8, sublabel='DMA priority | else AXI-Lite')
    arrow(ax, 2.0, 11.8, 4.0, 11.3, color='#0D47A1', lw=1.5, label='PIO')
    arrow(ax, 5.8, 11.8, 5.0, 11.3, color='#01579B', lw=1.5, label='PIO')

    # =====================================================================
    # 6. DPU CORE — dpu_top.sv
    # =====================================================================
    dpu_bg = FancyBboxPatch((0.0, 0.3), 26.5, 9.8, boxstyle="round,pad=0.12",
                             facecolor='#E8F5E9', edgecolor='#1B5E20',
                             linewidth=2.5, alpha=0.15, zorder=0)
    ax.add_patch(dpu_bg)
    ax.text(13.25, 9.75, 'dpu_top.sv — 18-Layer YOLOv4-tiny Sequencer',
            ha='center', fontsize=12, fontweight='bold', color='#1B5E20')

    # Arrow arbiter -> dpu_top
    arrow(ax, 4.75, 10.5, 4.75, 10.1, color='#37474F', lw=2,
          label='cmd_valid/ready/type/addr/data')

    # ----- PIO Interface -----
    box(ax, 0.3, 8.5, 3.0, 1.0, 'PIO Command\nDecoder',
        color='#6A1B9A', fontsize=8, sublabel='cmd_type 0-6')

    # ----- Layer Sequencer FSM -----
    box(ax, 3.7, 8.5, 4.0, 1.0, 'Layer Sequencer FSM',
        color='#263238', fontsize=9, sublabel='20 states | run_all + reload_req')

    # ----- Perf Counters -----
    box(ax, 8.1, 8.5, 2.5, 1.0, 'Performance\nCounters',
        color='#1B5E20', fontsize=8, sublabel='layer_cycles[0:17]\nperf_total_cycles')

    # ----- Layer Descriptors -----
    box(ax, 11.0, 8.5, 2.8, 1.0, 'Layer Descriptors',
        color='#455A64', fontsize=8, sublabel='ld_type/c_in/c_out/scale[0:17]\n16 fields x 18 layers')

    # Arrows between control blocks
    arrow(ax, 3.3, 9.0, 3.7, 9.0, color='#333', lw=1.5)
    arrow(ax, 7.7, 9.0, 8.1, 9.0, color='#333', lw=1.5)
    arrow(ax, 10.6, 9.0, 11.0, 9.0, color='#333', lw=1)

    # =====================================================================
    # 7. MEMORY SUBSYSTEM
    # =====================================================================
    mem_bg = FancyBboxPatch((0.3, 5.5), 13.3, 2.6, boxstyle="round,pad=0.1",
                             facecolor='#FFF8E1', edgecolor='#F57F17',
                             linewidth=2, alpha=0.25, zorder=0)
    ax.add_patch(mem_bg)
    ax.text(7.0, 7.85, 'Memory Subsystem', ha='center', fontsize=10,
            fontweight='bold', color='#F57F17')

    box(ax, 0.5, 5.7, 2.4, 1.5, 'fmap_a',
        color='#FF8F00', fontsize=9, sublabel='64 KB\n(ping)', label_y_offset=0.15)
    box(ax, 3.1, 5.7, 2.4, 1.5, 'fmap_b',
        color='#FF8F00', fontsize=9, sublabel='64 KB\n(pong)', label_y_offset=0.15)
    box(ax, 5.7, 5.7, 2.6, 1.5, 'weight_buf',
        color='#EF6C00', fontsize=9, sublabel='144 KB\n256-bit wide', label_y_offset=0.15)
    box(ax, 8.5, 5.7, 1.4, 1.5, 'bias_buf',
        color='#E65100', fontsize=8, sublabel='1 KB\n256x32b', label_y_offset=0.15)
    box(ax, 10.1, 5.7, 1.7, 1.5, 'patch_buf',
        color='#BF360C', fontsize=8, sublabel='1152 B\n256-bit wide', label_y_offset=0.15)
    box(ax, 12.0, 5.7, 1.4, 1.5, 'save_l*',
        color='#D84315', fontsize=8, sublabel='skip\nconn x4', label_y_offset=0.15)

    # FSM -> memories
    arrow(ax, 5.7, 8.5, 4.0, 7.2, color='#263238', lw=1.5)
    arrow(ax, 5.7, 8.5, 7.0, 7.2, color='#263238', lw=1.2)

    # =====================================================================
    # 8. CONV ENGINE ARRAY
    # =====================================================================
    eng_bg = FancyBboxPatch((14.2, 0.5), 12.0, 7.5, boxstyle="round,pad=0.1",
                             facecolor='#E8F5E9', edgecolor='#2E7D32',
                             linewidth=2.5, alpha=0.2, zorder=0)
    ax.add_patch(eng_bg)
    ax.text(20.2, 7.65, 'conv_engine_array.sv — Cout/Cin Tiling Controller',
            ha='center', fontsize=11, fontweight='bold', color='#1B5E20')

    # Engine FSM states (compact)
    fsm_states = [
        ('INIT',    14.5, 6.7, '#78909C', 0.9),
        ('BIAS\nLD', 15.6, 6.7, '#607D8B', 0.9),
        ('WEIGHT\nLOAD', 16.7, 6.7, '#E67E22', 1.1),
        ('ACT\nLOAD', 18.0, 6.7, '#E67E22', 0.9),
        ('MAC\nFIRE', 19.1, 6.7, '#2E7D32', 0.9),
        ('POST\nPROC', 20.2, 6.7, '#1565C0', 1.0),
        ('OUT', 21.4, 6.7, '#8E44AD', 0.7),
    ]
    for label, sx, sy, col, sw in fsm_states:
        box(ax, sx, sy, sw, 0.65, label, color=col, fontsize=6)

    for idx in range(len(fsm_states) - 1):
        x1 = fsm_states[idx][1] + fsm_states[idx][4]
        x2 = fsm_states[idx + 1][1]
        y = fsm_states[idx][2] + 0.32
        arrow(ax, x1, y, x2, y, color='#555', lw=1)

    # Tiling loop arrows
    ax.annotate('', xy=(16.7, 6.7), xytext=(20.0, 6.7),
                arrowprops=dict(arrowstyle='->', color='#C62828', lw=1.2,
                                connectionstyle='arc3,rad=0.5', linestyle='--'), zorder=1)
    ax.text(18.3, 6.35, 'next cin_tile', fontsize=5.5, color='#C62828',
            ha='center', fontweight='bold')

    ax.annotate('', xy=(15.6, 7.35), xytext=(21.4, 7.6),
                arrowprops=dict(arrowstyle='->', color='#C62828', lw=1.2,
                                connectionstyle='arc3,rad=-0.2', linestyle='--'), zorder=1)
    ax.text(18.5, 7.7, 'next cout_tile', fontsize=5.5, color='#C62828',
            ha='center', fontweight='bold')

    # ----- MAC Array -----
    mac_bg = FancyBboxPatch((14.5, 3.0), 5.5, 3.2, boxstyle="round,pad=0.08",
                             facecolor='#C8E6C9', edgecolor='#2E7D32',
                             linewidth=2, alpha=0.4, zorder=0)
    ax.add_patch(mac_bg)
    ax.text(17.25, 5.9, 'mac_array_32x32.sv', ha='center', fontsize=10,
            fontweight='bold', color='#1B5E20')
    ax.text(17.25, 5.55, '1024 INT8xINT8 -> INT32 MACs per cycle', ha='center',
            fontsize=7, color='#2E7D32')

    # Draw MAC grid
    for r in range(8):
        for c in range(8):
            rect = plt.Rectangle((14.8 + c * 0.5, 3.2 + r * 0.28), 0.45, 0.23,
                                  facecolor='#66BB6A', edgecolor='#388E3C',
                                  linewidth=0.3, alpha=0.7, zorder=3)
            ax.add_patch(rect)
    ax.text(16.8, 3.05, '32 rows x 32 cols', fontsize=6, color='#1B5E20', ha='center')

    # Weight arrow in
    arrow(ax, 13.6, 6.5, 14.5, 5.0, color='#E67E22', lw=2.5,
          label='256-bit\nweights', fontsize=6)
    # Act arrow in
    arrow(ax, 13.6, 6.0, 14.5, 4.2, color='#BF360C', lw=2,
          label='256-bit\nactivations', fontsize=6)

    # ----- Post-Process Array -----
    pp_bg = FancyBboxPatch((20.5, 3.0), 5.3, 3.2, boxstyle="round,pad=0.08",
                            facecolor='#E3F2FD', edgecolor='#1565C0',
                            linewidth=2, alpha=0.35, zorder=0)
    ax.add_patch(pp_bg)
    ax.text(23.15, 5.9, 'post_process_array.sv', ha='center', fontsize=10,
            fontweight='bold', color='#0D47A1')
    ax.text(23.15, 5.55, '32 parallel lanes', ha='center', fontsize=7, color='#1565C0')

    # Three processing stages
    box(ax, 20.7, 4.2, 1.3, 1.0, '+ Bias\nINT32', color='#1976D2', fontsize=7)
    box(ax, 22.2, 4.2, 1.3, 1.0, 'Leaky\nReLU', color='#1976D2', fontsize=7,
        sublabel='x>=0 ? x : x>>3')
    box(ax, 23.7, 4.2, 1.3, 1.0, 'Requant\nINT8', color='#1976D2', fontsize=7,
        sublabel='(x*scale)>>16')

    # Arrows between PP stages
    arrow(ax, 22.0, 4.7, 22.2, 4.7, color='#1565C0', lw=1.5)
    arrow(ax, 23.5, 4.7, 23.7, 4.7, color='#1565C0', lw=1.5)

    # MAC -> PP
    arrow(ax, 20.0, 4.7, 20.5, 4.7, color='#333', lw=2, label='acc[31:0] x32')

    # PP -> output (to fmap)
    arrow(ax, 25.0, 4.7, 25.8, 4.7, color='#8E44AD', lw=2)
    ax.text(26.0, 4.7, 'INT8\nx 32', fontsize=7, color='#8E44AD', va='center',
            fontweight='bold')

    # Output back to memory
    arrow(ax, 25.8, 5.2, 13.4, 7.0, color='#8E44AD', lw=1.5, linestyle='--',
          label='write output to fmap', fontsize=6,
          connectionstyle='arc3,rad=-0.15')

    # =====================================================================
    # 9. MAXPOOL UNIT
    # =====================================================================
    box(ax, 0.5, 2.8, 3.0, 1.5, 'maxpool_unit.sv',
        color='#6A1B9A', fontsize=9, sublabel='2x2 stride-2\n4-way max compare')
    arrow(ax, 1.7, 5.7, 2.0, 4.3, color='#6A1B9A', lw=1.5, label='fmap')
    arrow(ax, 2.0, 2.8, 2.0, 1.5, color='#6A1B9A', lw=1, label='out', fontsize=6)
    # Maxpool output back to fmap
    ax.annotate('', xy=(3.5, 5.7), xytext=(3.5, 2.8),
                arrowprops=dict(arrowstyle='->', color='#6A1B9A', lw=1.2, linestyle='--'),
                zorder=1)

    # =====================================================================
    # 10. 18-LAYER DATAFLOW (bottom strip)
    # =====================================================================
    flow_bg = FancyBboxPatch((0.0, -1.3), 26.5, 1.8, boxstyle="round,pad=0.08",
                              facecolor='#F5F5F5', edgecolor='#9E9E9E',
                              linewidth=1.5, alpha=0.4, zorder=0)
    ax.add_patch(flow_bg)
    ax.text(13.25, 0.3, '18-Layer Backbone Dataflow', ha='center',
            fontsize=9, fontweight='bold', color='#333')

    layer_colors = {
        'C3': '#2E7D32',   # Conv3x3
        'C1': '#1565C0',   # Conv1x1
        'RS': '#E65100',   # Route Split
        'RC': '#BF360C',   # Route Concat
        'MP': '#6A1B9A',   # MaxPool
    }
    layers_short = [
        (0, 'C3', '3->32\ns2'),   (1, 'C3', '32->64\ns2'),
        (2, 'C3', '64->64'),      (3, 'RS', '64->32'),
        (4, 'C3', '32->32'),      (5, 'C3', '32->32'),
        (6, 'RC', '32->64'),      (7, 'C1', '64->64'),
        (8, 'RC', '64->128'),     (9, 'MP', '128\ns2'),
        (10, 'C3', '128->128'),   (11, 'RS', '128->64'),
        (12, 'C3', '64->64'),     (13, 'C3', '64->64'),
        (14, 'RC', '64->128'),    (15, 'C1', '128->128'),
        (16, 'RC', '128->256'),   (17, 'MP', '256\ns2'),
    ]

    lw_box = 1.35
    lh_box = 0.95
    for lid, ltype, desc in layers_short:
        lx = 0.3 + lid * 1.46
        ly = -1.15
        box(ax, lx, ly, lw_box, lh_box, f'L{lid}', color=layer_colors[ltype],
            fontsize=7, sublabel=desc, sublabel_fs=5, lw=1)

    # Main chain arrows
    for i in range(17):
        x1 = 0.3 + i * 1.46 + lw_box
        x2 = 0.3 + (i + 1) * 1.46
        y = -1.15 + lh_box / 2
        arrow(ax, x1, y, x2, y, color='#666', lw=0.8)

    # Skip connections
    skip_style = dict(arrowstyle='->', color='#FF6F00', lw=1.2, linestyle='--',
                      connectionstyle='arc3,rad=-0.5')
    # L2 save -> L8
    l2x = 0.3 + 2 * 1.46 + lw_box / 2
    l8x = 0.3 + 8 * 1.46 + lw_box / 2
    ax.annotate('', xy=(l8x, -1.15), xytext=(l2x, -1.15),
                arrowprops=skip_style, zorder=1)
    # L4 save -> L6
    l4x = 0.3 + 4 * 1.46 + lw_box / 2
    l6x = 0.3 + 6 * 1.46 + lw_box / 2
    ax.annotate('', xy=(l6x, -1.15), xytext=(l4x, -1.15),
                arrowprops=dict(**{**skip_style, 'connectionstyle': 'arc3,rad=-0.6'}), zorder=1)
    # L10 save -> L16
    l10x = 0.3 + 10 * 1.46 + lw_box / 2
    l16x = 0.3 + 16 * 1.46 + lw_box / 2
    ax.annotate('', xy=(l16x, -1.15), xytext=(l10x, -1.15),
                arrowprops=skip_style, zorder=1)
    # L12 save -> L14
    l12x = 0.3 + 12 * 1.46 + lw_box / 2
    l14x = 0.3 + 14 * 1.46 + lw_box / 2
    ax.annotate('', xy=(l14x, -1.15), xytext=(l12x, -1.15),
                arrowprops=dict(**{**skip_style, 'connectionstyle': 'arc3,rad=-0.6'}), zorder=1)

    # Input label
    ax.text(-0.2, -0.68, 'IN\n3xHxW', fontsize=6, ha='center', va='center',
            fontweight='bold', color='#333')
    arrow(ax, 0.0, -0.68, 0.3, -0.68, color='#333', lw=1.5)
    # Output label
    ax.text(26.5, -0.68, 'OUT\n256xH/16\nxW/16', fontsize=6, ha='center', va='center',
            fontweight='bold', color='#333')
    arrow(ax, 25.9, -0.68, 26.2, -0.68, color='#333', lw=1.5)

    # =====================================================================
    # 11. LEGEND
    # =====================================================================
    legend_items = [
        ('Conv 3x3', '#2E7D32'),
        ('Conv 1x1', '#1565C0'),
        ('Route Split', '#E65100'),
        ('Route Concat', '#BF360C'),
        ('MaxPool', '#6A1B9A'),
        ('Skip Connection', '#FF6F00'),
        ('AXI Control', '#0D47A1'),
        ('AXI Data/DMA', '#01579B'),
        ('Memory', '#EF6C00'),
        ('Post-Process', '#1976D2'),
        ('Software', '#3949AB'),
        ('IRQ', '#C62828'),
    ]
    for i, (label, color) in enumerate(legend_items):
        col = i % 6
        row = i // 6
        lx = 14.5 + col * 2.15
        ly = 9.2 + (1 - row) * 0.35
        rect = plt.Rectangle((lx, ly), 0.2, 0.2, facecolor=color, edgecolor='none',
                               alpha=0.9, zorder=5)
        ax.add_patch(rect)
        ax.text(lx + 0.28, ly + 0.1, label, fontsize=6, va='center', color='#333', zorder=5)

    # =====================================================================
    # SAVE
    # =====================================================================
    plt.tight_layout(pad=0.5)
    out_path = 'C:/project/dpu-custom-yolo/docs/dpu_full_architecture.png'
    plt.savefig(out_path, dpi=220, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"[OK] {out_path}")


if __name__ == '__main__':
    create_full_architecture()
