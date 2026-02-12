#!/usr/bin/env python3
"""
Generate DPU architecture diagrams as PNG files.
1. Module hierarchy diagram
2. Dataflow graph (18-layer YOLOv4-tiny)
3. Conv engine internal pipeline
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


def draw_rounded_box(ax, x, y, w, h, label, color='#4A90D9', text_color='white',
                     fontsize=9, alpha=0.95, sublabel=None, border_color=None):
    """Draw a rounded rectangle with centered label."""
    if border_color is None:
        border_color = color
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                          facecolor=color, edgecolor=border_color,
                          linewidth=1.5, alpha=alpha, zorder=2)
    ax.add_patch(box)
    ty = y + h / 2
    if sublabel:
        ty = y + h * 0.6
        ax.text(x + w / 2, y + h * 0.3, sublabel, ha='center', va='center',
                fontsize=fontsize - 2, color=text_color, alpha=0.8, zorder=3,
                fontfamily='monospace')
    ax.text(x + w / 2, ty, label, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color=text_color, zorder=3)
    return box


def draw_arrow(ax, x1, y1, x2, y2, color='#333333', lw=1.5, style='->', label=None):
    """Draw an arrow between two points."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw),
                zorder=1)
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx + 0.02, my, label, fontsize=7, color=color, ha='left', va='center')


# =========================================================================
# DIAGRAM 1: Module Hierarchy
# =========================================================================
def create_hierarchy_diagram():
    fig, ax = plt.subplots(1, 1, figsize=(18, 14))
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor('#F8F9FA')

    # Title
    ax.text(5, 9.7, 'DPU Custom YOLOv4-tiny - Module Hierarchy',
            ha='center', va='center', fontsize=18, fontweight='bold', color='#1a1a2e')
    ax.text(5, 9.35, 'INT8 Fixed-Point | 32x32 MAC Array | 18 Layers | 256-bit Wide Memory',
            ha='center', va='center', fontsize=10, color='#555555')

    # === Top level: dpu_top ===
    top_box = FancyBboxPatch((0.3, 0.3), 9.4, 8.7, boxstyle="round,pad=0.15",
                              facecolor='#E8EEF6', edgecolor='#2C3E50',
                              linewidth=3, alpha=0.5, zorder=0)
    ax.add_patch(top_box)
    ax.text(5, 8.65, 'dpu_top.sv', ha='center', va='center',
            fontsize=14, fontweight='bold', color='#2C3E50')
    ax.text(5, 8.35, 'PIO Interface | FSM Sequencer | Ping-Pong Feature Maps',
            ha='center', va='center', fontsize=8, color='#555555')

    # === PIO Interface (top left) ===
    draw_rounded_box(ax, 0.6, 7.2, 2.8, 0.9, 'PIO Command Interface',
                     color='#8E44AD', sublabel='cmd_type 0-6 | run_all')

    # === FSM Block (top center) ===
    draw_rounded_box(ax, 3.7, 7.2, 3.1, 0.9, 'Layer Sequencer FSM',
                     color='#2C3E50', sublabel='S_IDLE..S_ALL_DONE (20 states)')

    # === Performance counters (top right) ===
    draw_rounded_box(ax, 7.1, 7.2, 2.3, 0.9, 'Perf Counters',
                     color='#27AE60', sublabel='layer_cycles[0:17]')

    # === Memory subsystem (row 2) ===
    mem_box = FancyBboxPatch((0.6, 5.2), 8.8, 1.6, boxstyle="round,pad=0.1",
                              facecolor='#FFF3E0', edgecolor='#E67E22',
                              linewidth=2, alpha=0.5, zorder=0)
    ax.add_patch(mem_box)
    ax.text(5, 6.55, 'Memory Subsystem', ha='center', va='center',
            fontsize=10, fontweight='bold', color='#E67E22')

    draw_rounded_box(ax, 0.8, 5.4, 1.8, 0.9, 'fmap_a / fmap_b',
                     color='#E67E22', sublabel='64KB x 2 (ping-pong)')
    draw_rounded_box(ax, 2.8, 5.4, 1.8, 0.9, 'weight_buf',
                     color='#E67E22', sublabel='144KB | 256b wide')
    draw_rounded_box(ax, 4.8, 5.4, 1.4, 0.9, 'bias_buf',
                     color='#E67E22', sublabel='256 x 32b')
    draw_rounded_box(ax, 6.4, 5.4, 1.4, 0.9, 'patch_buf',
                     color='#E67E22', sublabel='1152B | 256b wide')
    draw_rounded_box(ax, 8.0, 5.4, 1.2, 0.9, 'save_l*',
                     color='#D35400', sublabel='skip conn')

    # === Conv Engine Array (row 3, main block) ===
    eng_box = FancyBboxPatch((0.6, 1.6), 6.0, 3.2, boxstyle="round,pad=0.12",
                              facecolor='#E8F5E9', edgecolor='#27AE60',
                              linewidth=2.5, alpha=0.4, zorder=0)
    ax.add_patch(eng_box)
    ax.text(3.6, 4.5, 'conv_engine_array.sv', ha='center', va='center',
            fontsize=12, fontweight='bold', color='#1B5E20')
    ax.text(3.6, 4.2, 'Cout/Cin Tiling Controller | Wide Memory Reads',
            ha='center', va='center', fontsize=8, color='#2E7D32')

    # MAC array (inside engine)
    mac_box = FancyBboxPatch((0.9, 2.0), 3.0, 1.8, boxstyle="round,pad=0.1",
                              facecolor='#C8E6C9', edgecolor='#2E7D32',
                              linewidth=2, alpha=0.6, zorder=0)
    ax.add_patch(mac_box)
    draw_rounded_box(ax, 1.0, 2.8, 2.8, 0.8, 'mac_array_32x32.sv',
                     color='#2E7D32', sublabel='1024 MACs | INT8xINT8->INT32')

    # Draw grid representation
    for i in range(5):
        for j in range(5):
            rect = plt.Rectangle((1.1 + j * 0.15, 2.15 + i * 0.12), 0.12, 0.09,
                                  facecolor='#66BB6A', edgecolor='#388E3C',
                                  linewidth=0.5, alpha=0.8, zorder=3)
            ax.add_patch(rect)
    ax.text(2.2, 2.3, '32x32', fontsize=7, color='#1B5E20', ha='center', fontweight='bold')

    # Post-process array (inside engine)
    draw_rounded_box(ax, 4.2, 2.8, 2.2, 0.8, 'post_process_array.sv',
                     color='#1565C0', sublabel='32 lanes parallel')
    draw_rounded_box(ax, 4.3, 2.05, 0.6, 0.5, 'Bias+',
                     color='#1976D2', fontsize=7)
    draw_rounded_box(ax, 5.0, 2.05, 0.65, 0.5, 'LeakyReLU',
                     color='#1976D2', fontsize=7)
    draw_rounded_box(ax, 5.75, 2.05, 0.55, 0.5, 'Requant',
                     color='#1976D2', fontsize=7)

    # Arrows inside engine
    draw_arrow(ax, 2.4, 2.8, 4.2, 3.2, color='#2E7D32', lw=2, label='acc[31:0]')
    draw_arrow(ax, 4.9, 2.05, 5.0, 2.05, color='#1565C0', lw=1)
    draw_arrow(ax, 5.55, 2.3, 5.75, 2.3, color='#1565C0', lw=1)

    # === MaxPool unit (right side) ===
    draw_rounded_box(ax, 7.0, 2.8, 2.5, 1.0, 'maxpool_unit.sv',
                     color='#9C27B0', sublabel='2x2 stride-2 | 4-way max')

    # === Layer descriptor ROM (right) ===
    draw_rounded_box(ax, 7.0, 1.6, 2.5, 0.9, 'Layer Descriptors',
                     color='#607D8B', sublabel='ld_type/c_in/c_out/scale[0:17]')

    # === Arrows between major blocks ===
    # PIO -> FSM
    draw_arrow(ax, 3.4, 7.65, 3.7, 7.65, color='#333', lw=2)
    # FSM -> memories
    draw_arrow(ax, 5.25, 7.2, 5.0, 6.8, color='#333', lw=1.5, label='addr/data')
    # Memories -> engine (wide reads)
    draw_arrow(ax, 3.7, 5.4, 3.6, 4.8, color='#E67E22', lw=2.5, label='256-bit wide')
    # Engine -> fmap (output)
    draw_arrow(ax, 0.8, 4.8, 0.8, 6.1, color='#27AE60', lw=1.5, label='out_data')
    # FSM -> engine
    draw_arrow(ax, 4.5, 7.2, 3.6, 4.8, color='#2C3E50', lw=1.5, label='start')
    # FSM -> maxpool
    draw_arrow(ax, 6.5, 7.2, 8.25, 3.8, color='#2C3E50', lw=1)
    # Maxpool -> fmap
    draw_arrow(ax, 7.0, 3.3, 1.7, 6.1, color='#9C27B0', lw=1)
    # fmap -> patch_buf (patch loading)
    draw_arrow(ax, 1.7, 5.4, 6.4, 5.4, color='#D35400', lw=1, label='patch load (1B/cyc)')

    # === Legend ===
    legend_items = [
        ('PIO Interface', '#8E44AD'),
        ('Control/FSM', '#2C3E50'),
        ('Memory', '#E67E22'),
        ('Compute (MAC)', '#2E7D32'),
        ('Post-Process', '#1565C0'),
        ('MaxPool', '#9C27B0'),
        ('Config', '#607D8B'),
        ('Performance', '#27AE60'),
    ]
    for i, (label, color) in enumerate(legend_items):
        x = 0.7 + (i % 4) * 2.4
        y = 0.05 + (1 - i // 4) * 0.35
        rect = plt.Rectangle((x, y), 0.2, 0.2, facecolor=color, edgecolor='none', alpha=0.9)
        ax.add_patch(rect)
        ax.text(x + 0.28, y + 0.1, label, fontsize=7, va='center', color='#333')

    plt.tight_layout()
    plt.savefig('C:/project/dpu-custom-yolo/docs/dpu_hierarchy.png', dpi=200,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print("[OK] dpu_hierarchy.png")


# =========================================================================
# DIAGRAM 2: 18-Layer Dataflow Graph
# =========================================================================
def create_dataflow_diagram():
    fig, ax = plt.subplots(1, 1, figsize=(20, 12))
    ax.set_xlim(-1, 21)
    ax.set_ylim(-1.5, 8.5)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor('#F8F9FA')

    ax.text(10, 8.1, 'YOLOv4-tiny 18-Layer Dataflow (DPU)',
            ha='center', va='center', fontsize=18, fontweight='bold', color='#1a1a2e')
    ax.text(10, 7.7, 'Input: 3x16x16 INT8 | Output: 256x1x1 INT8 | H0=16, W0=16',
            ha='center', va='center', fontsize=10, color='#555555')

    # Layer definitions
    layers = [
        # (id, type, cin, cout, hout, wout, x_pos, y_pos, color)
        (0,  'Conv3x3\ns2', 3, 32, 8, 8),
        (1,  'Conv3x3\ns2', 32, 64, 4, 4),
        (2,  'Conv3x3', 64, 64, 4, 4),
        (3,  'Route\nSplit', 64, 32, 4, 4),
        (4,  'Conv3x3', 32, 32, 4, 4),
        (5,  'Conv3x3', 32, 32, 4, 4),
        (6,  'Route\nConcat', 32, 64, 4, 4),
        (7,  'Conv1x1', 64, 64, 4, 4),
        (8,  'Route\nConcat', 64, 128, 4, 4),
        (9,  'MaxPool\ns2', 128, 128, 2, 2),
        (10, 'Conv3x3', 128, 128, 2, 2),
        (11, 'Route\nSplit', 128, 64, 2, 2),
        (12, 'Conv3x3', 64, 64, 2, 2),
        (13, 'Conv3x3', 64, 64, 2, 2),
        (14, 'Route\nConcat', 64, 128, 2, 2),
        (15, 'Conv1x1', 128, 128, 2, 2),
        (16, 'Route\nConcat', 128, 256, 2, 2),
        (17, 'MaxPool\ns2', 256, 256, 1, 1),
    ]

    type_colors = {
        'Conv3x3': '#2E7D32', 'Conv3x3\ns2': '#1B5E20',
        'Conv1x1': '#1565C0',
        'Route\nSplit': '#E65100', 'Route\nConcat': '#BF360C',
        'MaxPool\ns2': '#6A1B9A',
    }

    # Position layers in a flow layout
    # Main chain goes left to right, with skip connections
    positions = {}
    # Row 1 (top): layers 0-9
    for i in range(10):
        x = 1.0 + i * 2.0
        y = 5.5
        positions[i] = (x, y)

    # Row 2 (bottom): layers 10-17
    for i in range(10, 18):
        x = 1.0 + (i - 10) * 2.0
        y = 2.5
        positions[i] = (x, y)

    # Draw layer boxes
    bw, bh = 1.4, 1.3
    for lid, ltype, cin, cout, hout, wout in layers:
        x, y = positions[lid]
        color = type_colors.get(ltype, '#555555')
        draw_rounded_box(ax, x - bw/2, y - bh/2, bw, bh, f'L{lid}',
                         color=color, fontsize=11,
                         sublabel=f'{ltype}\n{cout}x{hout}x{wout}')

    # Draw main chain arrows
    for i in range(9):
        x1, y1 = positions[i]
        x2, y2 = positions[i + 1]
        draw_arrow(ax, x1 + bw/2, y1, x2 - bw/2, y2, color='#333', lw=1.5)

    # L9 -> L10 (row change: down)
    x1, y1 = positions[9]
    x2, y2 = positions[10]
    draw_arrow(ax, x1, y1 - bh/2, x1, y2 + bh/2 + 0.3, color='#333', lw=1.5)
    draw_arrow(ax, x1 - 0.3, y2 + bh/2 + 0.3, x2 + bw/2, y2 + bh/2 + 0.3,
               color='#333', lw=1.5, style='-')
    draw_arrow(ax, x2 + bw/2, y2 + bh/2 + 0.3, x2, y2 + bh/2, color='#333', lw=1.5)

    for i in range(10, 17):
        x1, y1 = positions[i]
        x2, y2 = positions[i + 1]
        draw_arrow(ax, x1 + bw/2, y1, x2 - bw/2, y2, color='#333', lw=1.5)

    # Skip connections (curved arrows)
    skip_connections = [
        (5, 6, 'L5+L4_save', '#FF6F00'),     # L5 output + L4 save -> L6
        (2, 8, 'L2_save', '#FF6F00'),          # L2 save -> L8
        (4, 6, 'L4_save', '#FF8F00'),           # L4 save -> L6
        (13, 14, 'L13+L12_save', '#FF6F00'),   # L13 + L12 save -> L14
        (12, 14, 'L12_save', '#FF8F00'),         # L12 save -> L14
        (10, 16, 'L10_save', '#FF6F00'),        # L10 save -> L16
    ]

    # Draw skip: L2 save -> L8
    x2s, y2s = positions[2]
    x8, y8 = positions[8]
    ax.annotate('', xy=(x8, y8 + bh/2), xytext=(x2s, y2s - bh/2),
                arrowprops=dict(arrowstyle='->', color='#FF6F00', lw=2,
                                connectionstyle='arc3,rad=0.3', linestyle='--'))
    ax.text((x2s + x8)/2, y2s - bh/2 - 0.25, 'save_l2', fontsize=7, color='#FF6F00',
            ha='center', fontweight='bold')

    # Draw skip: L4 save -> L6
    x4, y4 = positions[4]
    x6, y6 = positions[6]
    ax.annotate('', xy=(x6, y6 - bh/2), xytext=(x4, y4 - bh/2),
                arrowprops=dict(arrowstyle='->', color='#FF8F00', lw=1.5,
                                connectionstyle='arc3,rad=-0.4', linestyle='--'))
    ax.text((x4 + x6)/2, y4 - bh/2 - 0.35, 'save_l4', fontsize=7, color='#FF8F00',
            ha='center', fontweight='bold')

    # Draw skip: L10 save -> L16
    x10, y10 = positions[10]
    x16, y16 = positions[16]
    ax.annotate('', xy=(x16, y16 - bh/2), xytext=(x10, y10 - bh/2),
                arrowprops=dict(arrowstyle='->', color='#FF6F00', lw=2,
                                connectionstyle='arc3,rad=-0.3', linestyle='--'))
    ax.text((x10 + x16)/2, y10 - bh/2 - 0.35, 'save_l10', fontsize=7, color='#FF6F00',
            ha='center', fontweight='bold')

    # Draw skip: L12 save -> L14
    x12, y12 = positions[12]
    x14, y14 = positions[14]
    ax.annotate('', xy=(x14, y14 - bh/2), xytext=(x12, y12 - bh/2),
                arrowprops=dict(arrowstyle='->', color='#FF8F00', lw=1.5,
                                connectionstyle='arc3,rad=-0.4', linestyle='--'))
    ax.text((x12 + x14)/2, y12 - bh/2 - 0.35, 'save_l12', fontsize=7, color='#FF8F00',
            ha='center', fontweight='bold')

    # Input arrow
    x0, y0 = positions[0]
    ax.annotate('Input\n3x16x16', xy=(x0 - bw/2, y0), xytext=(x0 - bw/2 - 1.2, y0),
                arrowprops=dict(arrowstyle='->', color='#333', lw=2),
                fontsize=9, ha='center', va='center', fontweight='bold')

    # Output arrow
    x17, y17 = positions[17]
    ax.annotate('Output\n256x1x1', xy=(x17 + bw/2 + 1.2, y17), xytext=(x17 + bw/2, y17),
                arrowprops=dict(arrowstyle='->', color='#333', lw=2),
                fontsize=9, ha='center', va='center', fontweight='bold')

    # Legend
    legend_items = [
        ('Conv 3x3', '#2E7D32'), ('Conv 1x1', '#1565C0'),
        ('Route (split/concat)', '#BF360C'), ('MaxPool 2x2', '#6A1B9A'),
        ('Skip connection (save)', '#FF6F00'),
    ]
    for i, (label, color) in enumerate(legend_items):
        x = 1.0 + i * 3.8
        y = 0.3
        rect = plt.Rectangle((x, y), 0.25, 0.25, facecolor=color, edgecolor='none', alpha=0.9)
        ax.add_patch(rect)
        ax.text(x + 0.35, y + 0.12, label, fontsize=8, va='center', color='#333')

    # Cycle counts annotation
    ax.text(10, -0.5,
            'Performance (16x16 input): Total ~1.2M cycles | Conv 99.8% | Route/MaxPool <0.2%',
            ha='center', fontsize=9, color='#555', style='italic')

    plt.tight_layout()
    plt.savefig('C:/project/dpu-custom-yolo/docs/dpu_dataflow.png', dpi=200,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print("[OK] dpu_dataflow.png")


# =========================================================================
# DIAGRAM 3: Conv Engine Pipeline Detail
# =========================================================================
def create_engine_pipeline_diagram():
    fig, ax = plt.subplots(1, 1, figsize=(18, 10))
    ax.set_xlim(-0.5, 16.5)
    ax.set_ylim(-0.5, 9)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor('#F8F9FA')

    ax.text(8, 8.7, 'Conv Engine Array - Internal Pipeline',
            ha='center', va='center', fontsize=16, fontweight='bold', color='#1a1a2e')
    ax.text(8, 8.3, '32x32 MAC Array | Cout/Cin Tiling | 256-bit Wide Memory | 3-Stage Post-Process',
            ha='center', va='center', fontsize=9, color='#555555')

    # === FSM states flow (top) ===
    states = [
        ('INIT', 0.5, 7.2, '#78909C'),
        ('BIAS\nLOAD', 2.2, 7.2, '#607D8B'),
        ('WEIGHT\nLOAD', 4.2, 7.2, '#E67E22'),
        ('ACT\nLOAD', 6.2, 7.2, '#E67E22'),
        ('MAC\nFIRE', 8.2, 7.2, '#2E7D32'),
        ('POST\nPROC', 10.5, 7.2, '#1565C0'),
        ('OUTPUT', 13.0, 7.2, '#8E44AD'),
    ]
    sw, sh = 1.4, 0.9
    for label, x, y, color in states:
        draw_rounded_box(ax, x, y, sw, sh, label, color=color, fontsize=8)

    # Arrows between states
    for i in range(len(states) - 1):
        x1 = states[i][1] + sw
        x2 = states[i+1][1]
        y = states[i][2] + sh/2
        draw_arrow(ax, x1, y, x2, y, color='#333', lw=1.5)

    # Loop back arrow (MAC -> WEIGHT for next cin/kpos tile)
    ax.annotate('', xy=(4.2, 7.2), xytext=(8.2 + sw/2, 7.2),
                arrowprops=dict(arrowstyle='->', color='#C62828', lw=1.5,
                                connectionstyle='arc3,rad=0.4', linestyle='--'))
    ax.text(6.5, 6.7, 'next cin_tile\nor kpos', fontsize=7, color='#C62828',
            ha='center', fontweight='bold')

    # Loop back: OUTPUT -> BIAS for next Cout tile
    ax.annotate('', xy=(2.2, 8.1), xytext=(13.0 + sw/2, 8.1),
                arrowprops=dict(arrowstyle='->', color='#C62828', lw=1.5,
                                connectionstyle='arc3,rad=-0.25', linestyle='--'))
    ax.text(8, 8.0, 'next cout_tile', fontsize=7, color='#C62828',
            ha='center', fontweight='bold')

    # === Weight loading detail ===
    draw_rounded_box(ax, 0.5, 4.5, 4.5, 2.2, '', color='#FFF3E0',
                     border_color='#E67E22', text_color='#333')
    ax.text(2.75, 6.4, 'Weight Loading (32B/cycle)', ha='center', fontsize=10,
            fontweight='bold', color='#E67E22')

    # Show weight matrix
    ax.text(1.0, 5.9, 'weight_buf [co][kpos][cin]', fontsize=8, color='#555',
            fontfamily='monospace')

    # Weight matrix grid
    for r in range(4):
        for c in range(8):
            rect = plt.Rectangle((1.0 + c * 0.4, 4.7 + r * 0.25), 0.35, 0.2,
                                  facecolor='#FFE0B2' if r < 3 else '#FFCC80',
                                  edgecolor='#E67E22', linewidth=0.5, zorder=3)
            ax.add_patch(rect)
    ax.text(2.6, 4.6, '32 rows x 32 cols (one row per cycle)', fontsize=7,
            color='#E67E22', ha='center')
    # Wide read arrow
    draw_arrow(ax, 4.6, 5.2, 5.2, 5.2, color='#E67E22', lw=2.5, label='256b')

    # === Activation loading detail ===
    draw_rounded_box(ax, 5.4, 4.5, 2.5, 2.2, '', color='#E8F5E9',
                     border_color='#2E7D32', text_color='#333')
    ax.text(6.65, 6.4, 'Act Loading (32B)', ha='center', fontsize=10,
            fontweight='bold', color='#2E7D32')

    # Activation vector
    for c in range(8):
        rect = plt.Rectangle((5.6 + c * 0.27, 5.0), 0.22, 0.8,
                              facecolor='#C8E6C9', edgecolor='#2E7D32',
                              linewidth=0.5, zorder=3)
        ax.add_patch(rect)
    ax.text(6.65, 4.7, '32 values in 1 cycle', fontsize=7, color='#2E7D32', ha='center')
    draw_arrow(ax, 7.6, 5.4, 8.2, 5.4, color='#2E7D32', lw=2, label='256b')

    # === MAC Array ===
    mac_x, mac_y = 8.5, 3.8
    mac_w, mac_h = 3.5, 3.0
    draw_rounded_box(ax, mac_x, mac_y, mac_w, mac_h, '',
                     color='#E8F5E9', border_color='#1B5E20')
    ax.text(mac_x + mac_w/2, mac_y + mac_h - 0.3, 'mac_array_32x32',
            ha='center', fontsize=11, fontweight='bold', color='#1B5E20')

    # Draw 32x32 grid (simplified)
    gx, gy = mac_x + 0.3, mac_y + 0.3
    gw, gh = 2.9, 2.0
    for r in range(8):
        for c in range(8):
            rect = plt.Rectangle((gx + c * gw/8, gy + r * gh/8),
                                  gw/8 - 0.02, gh/8 - 0.02,
                                  facecolor='#66BB6A', edgecolor='#388E3C',
                                  linewidth=0.3, alpha=0.8, zorder=3)
            ax.add_patch(rect)

    ax.text(mac_x + mac_w/2, mac_y + 0.15,
            '1024 MACs/cycle | INT8 x INT8 -> INT32 accumulate',
            fontsize=7, color='#1B5E20', ha='center')

    # Weights arrow into MAC
    draw_arrow(ax, 5.0, 5.6, mac_x, 5.6, color='#E67E22', lw=2)
    # Act arrow into MAC (broadcast)
    ax.text(mac_x + mac_w/2, mac_y + mac_h + 0.15, 'activations broadcast',
            fontsize=7, color='#2E7D32', ha='center')

    # === Post-process pipeline ===
    pp_x = 12.5
    draw_rounded_box(ax, pp_x, 5.5, 3.5, 1.3, '', color='#E3F2FD',
                     border_color='#1565C0')
    ax.text(pp_x + 1.75, 6.5, 'Post-Process Pipeline (3 stages)',
            ha='center', fontsize=10, fontweight='bold', color='#1565C0')

    stages = [('+ Bias', pp_x + 0.15), ('Leaky\nReLU', pp_x + 1.2), ('Requant\nINT8', pp_x + 2.3)]
    for label, sx in stages:
        draw_rounded_box(ax, sx, 5.7, 0.9, 0.8, label, color='#1976D2', fontsize=8)

    draw_arrow(ax, pp_x + 1.05, 6.1, pp_x + 1.2, 6.1, color='#1565C0', lw=1.5)
    draw_arrow(ax, pp_x + 2.1, 6.1, pp_x + 2.3, 6.1, color='#1565C0', lw=1.5)

    # MAC -> post-process
    draw_arrow(ax, mac_x + mac_w, 5.8, pp_x, 6.1, color='#333', lw=2, label='acc[31:0] x 32')

    # Output
    draw_arrow(ax, pp_x + 3.5, 6.1, pp_x + 4.2, 6.1, color='#8E44AD', lw=2)
    ax.text(pp_x + 4.3, 6.1, 'out_data[0:31]\nINT8 x 32 lanes',
            fontsize=8, color='#8E44AD', va='center', fontweight='bold')

    # === Tiling info box ===
    draw_rounded_box(ax, 0.5, 0.5, 15.5, 2.8, '', color='#FAFAFA',
                     border_color='#9E9E9E', text_color='#333')
    ax.text(8.25, 3.0, 'Tiling Strategy (Output-Stationary)', ha='center',
            fontsize=12, fontweight='bold', color='#333')

    tiling_text = [
        'Cout tiles: process 32 output channels at a time (1-8 tiles depending on layer)',
        'Cin tiles: process 32 input channels at a time, accumulate partial sums',
        'Kernel positions: for 3x3 conv, iterate 9 positions; for 1x1, single position',
        '',
        'Layer 10 (128in, 128out, 3x3): 4 Cout_tiles x (9 kpos x 4 Cin_tiles) = 144 array invocations per pixel',
        'Each invocation: 32 cycles (weight rows) + 2 cycles (act) + 2 cycles (MAC) = 36 cycles',
    ]
    for i, line in enumerate(tiling_text):
        ax.text(1.0, 2.6 - i * 0.35, line, fontsize=8, color='#333',
                fontfamily='monospace' if i >= 4 else 'sans-serif')

    plt.tight_layout()
    plt.savefig('C:/project/dpu-custom-yolo/docs/dpu_engine_pipeline.png', dpi=200,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print("[OK] dpu_engine_pipeline.png")


if __name__ == '__main__':
    import os
    os.makedirs('C:/project/dpu-custom-yolo/docs', exist_ok=True)
    create_hierarchy_diagram()
    create_dataflow_diagram()
    create_engine_pipeline_diagram()
    print("\nAll diagrams generated in docs/")
