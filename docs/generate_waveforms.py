#!/usr/bin/env python3
"""
Generate clean timing/waveform diagrams as PNGs.
1. AXI4-Lite write transaction
2. MAC array compute cycle
3. Full layer execution timeline
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def draw_clock(ax, y, x_start, x_end, period=1.0, label='clk'):
    """Draw a clock waveform."""
    x = x_start
    xs, ys = [x], [y]
    while x < x_end:
        xs.extend([x, x, x + period/2, x + period/2])
        ys.extend([y, y + 0.8, y + 0.8, y])
        x += period
    ax.plot(xs, ys, color='#333', lw=1.5, zorder=3)
    ax.text(x_start - 0.3, y + 0.4, label, fontsize=8, ha='right', va='center',
            fontweight='bold', fontfamily='monospace')


def draw_signal(ax, y, transitions, label='', color='#1565C0', height=0.7):
    """
    Draw a digital signal.
    transitions: list of (x_start, x_end, value_str, level)
        level: 0=low, 1=high, 0.5=data (hatched box)
    """
    ax.text(transitions[0][0] - 0.3, y + height/2, label, fontsize=8,
            ha='right', va='center', fontweight='bold', fontfamily='monospace')

    for x1, x2, val_str, level in transitions:
        if level == 0:
            ax.plot([x1, x2], [y, y], color=color, lw=1.5, zorder=3)
        elif level == 1:
            ax.plot([x1, x2], [y + height, y + height], color=color, lw=1.5, zorder=3)
            ax.fill_between([x1, x2], y, y + height, color=color, alpha=0.15, zorder=1)
        elif level == 0.5:
            # Data/bus value (box with text)
            rect = plt.Rectangle((x1, y), x2 - x1, height,
                                  facecolor=color, alpha=0.2, edgecolor=color, lw=1.2, zorder=2)
            ax.add_patch(rect)
            ax.text((x1 + x2) / 2, y + height / 2, val_str, fontsize=7,
                    ha='center', va='center', color=color, fontweight='bold', zorder=4)
            # Transition marks
            ax.plot([x1, x1 + 0.1], [y, y + height/2], color=color, lw=1, zorder=3)
            ax.plot([x1, x1 + 0.1], [y + height, y + height/2], color=color, lw=1, zorder=3)

        # Transitions between segments
        if level in (0, 1):
            prev_y = y if level == 0 else y + height
            ax.plot([x1, x1], [y, y + height], color=color, lw=0.8, alpha=0.5, zorder=2)


# =============================================================================
# DIAGRAM 1: AXI4-Lite Write Transaction
# =============================================================================
def create_axi_waveform():
    fig, ax = plt.subplots(figsize=(18, 8))
    ax.set_xlim(-3, 18)
    ax.set_ylim(-1, 14)
    ax.axis('off')
    fig.patch.set_facecolor('#FAFBFC')

    ax.text(7.5, 13.5, 'AXI4-Lite Write Transaction — PIO write_byte', fontsize=16,
            ha='center', fontweight='bold', color='#1a1a2e')
    ax.text(7.5, 12.8, 'Writing one byte to DPU weight buffer through AXI register interface',
            ha='center', fontsize=9, color='#666')

    # Cycle numbers
    for i in range(16):
        ax.text(i + 0.5, 12.2, str(i), fontsize=7, ha='center', va='center',
                color='#999', fontfamily='monospace')

    # Clock
    draw_clock(ax, 11.0, 0, 16, period=1.0, label='aclk')

    # AXI Write Address
    draw_signal(ax, 9.5, [
        (0, 2, '', 0), (2, 4, '0x00', 0.5), (4, 6, '', 0),
        (6, 8, '0x04', 0.5), (8, 10, '', 0), (10, 12, '0x08', 0.5), (12, 16, '', 0),
    ], label='awaddr', color='#0D47A1')

    draw_signal(ax, 8.3, [
        (0, 2, '', 0), (2, 4, '', 1), (4, 6, '', 0),
        (6, 8, '', 1), (8, 10, '', 0), (10, 12, '', 1), (12, 16, '', 0),
    ], label='awvalid', color='#0D47A1')

    draw_signal(ax, 7.1, [
        (0, 3, '', 0), (3, 4, '', 1), (4, 7, '', 0),
        (7, 8, '', 1), (8, 11, '', 0), (11, 12, '', 1), (12, 16, '', 0),
    ], label='awready', color='#2E7D32')

    # AXI Write Data
    draw_signal(ax, 5.9, [
        (0, 2, '', 0), (2, 4, 'CMD=0', 0.5), (4, 6, '', 0),
        (6, 8, 'ADDR', 0.5), (8, 10, '', 0), (10, 12, 'DATA', 0.5), (12, 16, '', 0),
    ], label='wdata', color='#E65100')

    draw_signal(ax, 4.7, [
        (0, 2, '', 0), (2, 4, '', 1), (4, 6, '', 0),
        (6, 8, '', 1), (8, 10, '', 0), (10, 12, '', 1), (12, 16, '', 0),
    ], label='wvalid', color='#E65100')

    # Write Response
    draw_signal(ax, 3.5, [
        (0, 3.5, '', 0), (3.5, 4.5, '', 1), (4.5, 7.5, '', 0),
        (7.5, 8.5, '', 1), (8.5, 11.5, '', 0), (11.5, 12.5, '', 1), (12.5, 16, '', 0),
    ], label='bvalid', color='#C62828')

    # PIO internal state
    draw_signal(ax, 2.0, [
        (0, 12, 'IDLE', 0.5), (12, 13, 'REQ', 0.5), (13, 16, 'IDLE', 0.5),
    ], label='pio_state', color='#6A1B9A')

    # DPU cmd_valid
    draw_signal(ax, 0.8, [
        (0, 12, '', 0), (12, 13, '', 1), (13, 16, '', 0),
    ], label='cmd_valid', color='#2E7D32')

    # Annotations
    ax.annotate('1. Write CMD\n(cmd_type=0)', xy=(3, 9.8), fontsize=7, ha='center',
                color='#0D47A1', fontweight='bold')
    ax.annotate('2. Write ADDR', xy=(7, 9.8), fontsize=7, ha='center',
                color='#0D47A1', fontweight='bold')
    ax.annotate('3. Write DATA\n(triggers PIO)', xy=(11, 9.8), fontsize=7, ha='center',
                color='#E65100', fontweight='bold')
    ax.annotate('PIO fires', xy=(12.5, 1.5), xytext=(14, 0.3),
                arrowprops=dict(arrowstyle='->', color='#C62828'), fontsize=8,
                color='#C62828', fontweight='bold')

    plt.tight_layout()
    out = 'C:/project/dpu-custom-yolo/docs/waveform_axi_write.png'
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"[OK] {out}")


# =============================================================================
# DIAGRAM 2: MAC Array Compute Cycle
# =============================================================================
def create_mac_waveform():
    fig, ax = plt.subplots(figsize=(18, 9))
    ax.set_xlim(-4, 22)
    ax.set_ylim(-1, 15)
    ax.axis('off')
    fig.patch.set_facecolor('#FAFBFC')

    ax.text(9, 14.5, 'Conv Engine Array — One Tile Computation', fontsize=16,
            ha='center', fontweight='bold', color='#1a1a2e')
    ax.text(9, 13.8, '32 weight rows loaded (1 cycle each), 32 activations broadcast, 1-cycle MAC fire',
            ha='center', fontsize=9, color='#666')

    for i in range(20):
        ax.text(i + 0.5, 13.2, str(i), fontsize=6, ha='center', color='#999', fontfamily='monospace')

    draw_clock(ax, 12.0, 0, 20, period=1.0, label='clk')

    # Engine state
    draw_signal(ax, 10.5, [
        (0, 1, 'BIAS', 0.5), (1, 2, 'W0', 0.5), (2, 3, 'W1', 0.5),
        (3, 4, '...', 0.5), (4, 5, 'W31', 0.5),
        (5, 6, 'ACT', 0.5), (6, 7, 'LATCH', 0.5),
        (7, 8, 'MAC', 0.5), (8, 9, 'MAC', 0.5),
        (9, 15, '... next kpos ...', 0.5),
        (15, 17, 'POST', 0.5), (17, 19, 'OUT', 0.5), (19, 20, '', 0),
    ], label='state', color='#263238')

    # Weight read address
    draw_signal(ax, 9.0, [
        (0, 1, '', 0),
        (1, 2, 'r0', 0.5), (2, 3, 'r1', 0.5), (3, 4, '...', 0.5), (4, 5, 'r31', 0.5),
        (5, 20, '', 0),
    ], label='wt_addr', color='#E67E22')

    # Weight data (256-bit wide)
    draw_signal(ax, 7.5, [
        (0, 1.5, '', 0),
        (1.5, 2.5, '32B', 0.5), (2.5, 3.5, '32B', 0.5), (3.5, 4.5, '...', 0.5), (4.5, 5.5, '32B', 0.5),
        (5.5, 20, '', 0),
    ], label='wt_data\n(256b)', color='#E67E22')

    # Activation load
    draw_signal(ax, 6.0, [
        (0, 5, '', 0), (5, 6, '32B', 0.5), (6, 7, 'latch', 0.5), (7, 20, '', 0),
    ], label='act_data\n(256b)', color='#2E7D32')

    # MAC fire
    draw_signal(ax, 4.5, [
        (0, 7, '', 0), (7, 9, '', 1), (9, 20, '', 0),
    ], label='mac_fire', color='#C62828')

    # Accumulator
    draw_signal(ax, 3.0, [
        (0, 7, '0', 0.5), (7, 9, 'partial', 0.5),
        (9, 15, 'accumulate...', 0.5), (15, 17, 'final', 0.5), (17, 20, '', 0),
    ], label='acc[31:0]', color='#6A1B9A')

    # Output valid
    draw_signal(ax, 1.5, [
        (0, 17, '', 0), (17, 19, '', 1), (19, 20, '', 0),
    ], label='out_valid', color='#1565C0')

    # Output data
    draw_signal(ax, 0.0, [
        (0, 17, '', 0), (17, 19, 'INT8 x32', 0.5), (19, 20, '', 0),
    ], label='out_data', color='#1565C0')

    # Annotations with arrows
    ax.annotate('32 rows\n(1 cycle each)', xy=(3, 10.3), xytext=(3, 9.0),
                fontsize=7, ha='center', color='#E67E22', fontweight='bold')
    ax.annotate('32 activations\nbroadcast', xy=(5.5, 7.0), xytext=(8, 6.0),
                arrowprops=dict(arrowstyle='->', color='#2E7D32'),
                fontsize=7, color='#2E7D32', fontweight='bold')
    ax.annotate('1024 MACs\nin 2 cycles', xy=(8, 5.2), xytext=(10, 4.2),
                arrowprops=dict(arrowstyle='->', color='#C62828'),
                fontsize=7, color='#C62828', fontweight='bold')

    # Timing summary box
    ax.text(15, 2.5, 'Tile Timing:\n'
            '  Weight load: 32 cycles\n'
            '  Act load:     2 cycles\n'
            '  MAC fire:     2 cycles\n'
            '  Total:       36 cycles/tile',
            fontsize=8, fontfamily='monospace', color='#333',
            bbox=dict(boxstyle='round', facecolor='#E8F5E9', edgecolor='#2E7D32', alpha=0.8))

    plt.tight_layout()
    out = 'C:/project/dpu-custom-yolo/docs/waveform_mac_cycle.png'
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"[OK] {out}")


# =============================================================================
# DIAGRAM 3: Full Layer Execution Timeline
# =============================================================================
def create_layer_timeline():
    fig, ax = plt.subplots(figsize=(20, 8))
    ax.set_xlim(-1, 22)
    ax.set_ylim(-0.5, 7)
    ax.axis('off')
    fig.patch.set_facecolor('#FAFBFC')

    ax.text(10.5, 6.7, 'Full 18-Layer Inference Timeline (run_all mode)', fontsize=16,
            ha='center', fontweight='bold', color='#1a1a2e')
    ax.text(10.5, 6.2, 'DPU pauses before conv layers for weight reload, auto-advances route/maxpool',
            ha='center', fontsize=9, color='#666')

    # Layer timing data (approximate proportional widths)
    # Based on measured cycles: total ~1.2M, conv layers dominate
    layers_timeline = [
        # (label, type, relative_width, color)
        ('L0\nConv\ns2', 'conv', 1.6, '#2E7D32'),
        ('L1\nConv\ns2', 'conv', 1.2, '#2E7D32'),
        ('L2\nConv', 'conv', 1.8, '#2E7D32'),
        ('R3', 'route', 0.1, '#E65100'),
        ('L4\nConv', 'conv', 0.6, '#388E3C'),
        ('L5\nConv', 'conv', 0.6, '#388E3C'),
        ('R6', 'route', 0.1, '#E65100'),
        ('L7\nC1x1', 'conv', 0.3, '#1565C0'),
        ('R8', 'route', 0.1, '#BF360C'),
        ('M9', 'maxpool', 0.15, '#6A1B9A'),
        ('L10\nConv', 'conv', 3.0, '#1B5E20'),
        ('R11', 'route', 0.1, '#E65100'),
        ('L12\nConv', 'conv', 1.2, '#388E3C'),
        ('L13\nConv', 'conv', 1.2, '#388E3C'),
        ('R14', 'route', 0.1, '#BF360C'),
        ('L15\nC1x1', 'conv', 0.6, '#1565C0'),
        ('R16', 'route', 0.1, '#BF360C'),
        ('M17', 'maxpool', 0.15, '#6A1B9A'),
    ]

    # Draw timeline
    y_main = 3.5
    x = 0.5
    total_w = sum(l[2] for l in layers_timeline)
    scale = 20.0 / total_w

    for label, ltype, w, color in layers_timeline:
        sw = w * scale
        rect = plt.Rectangle((x, y_main), sw, 1.5, facecolor=color, alpha=0.8,
                               edgecolor='white', linewidth=1.5, zorder=3)
        ax.add_patch(rect)
        if sw > 0.4:
            ax.text(x + sw / 2, y_main + 0.75, label, fontsize=6, ha='center', va='center',
                    color='white', fontweight='bold', zorder=4)

        # Reload pause marker for conv layers
        if ltype == 'conv':
            ax.plot([x, x], [y_main + 1.5, y_main + 2.0], color='#C62828', lw=1.5, zorder=5)
            ax.plot([x - 0.1, x + 0.1], [y_main + 2.0, y_main + 2.0],
                    color='#C62828', lw=1.5, zorder=5)

        x += sw

    # Reload label
    ax.text(10.5, y_main + 2.3, 'reload_req pauses (weight load from host)',
            ha='center', fontsize=8, color='#C62828', fontweight='bold')

    # Time axis
    ax.annotate('', xy=(20.5, y_main - 0.3), xytext=(0.5, y_main - 0.3),
                arrowprops=dict(arrowstyle='->', color='#333', lw=2))
    ax.text(10.5, y_main - 0.6, 'Time (~1.2M cycles @ 100 MHz = 12.2 ms for 16x16 input)',
            ha='center', fontsize=9, color='#333')

    # Cycle breakdown
    ax.text(0.5, 1.5, 'Cycle Breakdown:', fontsize=10, fontweight='bold', color='#333')
    breakdown = [
        ('Conv 3x3 layers (0,1,2,4,5,10,12,13)', '1,180,000 cyc', '97.1%', '#2E7D32'),
        ('Conv 1x1 layers (7,15)',                 '34,000 cyc',    '2.8%',  '#1565C0'),
        ('Route layers (3,6,8,11,14,16)',          '800 cyc',       '0.07%', '#E65100'),
        ('MaxPool layers (9,17)',                   '500 cyc',       '0.04%', '#6A1B9A'),
    ]
    for i, (desc, cyc, pct, color) in enumerate(breakdown):
        y = 1.0 - i * 0.35
        rect = plt.Rectangle((0.5, y), 0.25, 0.2, facecolor=color, edgecolor='none', zorder=3)
        ax.add_patch(rect)
        ax.text(0.9, y + 0.1, f'{desc}', fontsize=8, va='center', color='#333')
        ax.text(16, y + 0.1, f'{cyc}', fontsize=8, va='center', color='#333',
                fontfamily='monospace', ha='right')
        ax.text(17, y + 0.1, f'{pct}', fontsize=8, va='center', color=color,
                fontweight='bold')

    plt.tight_layout()
    out = 'C:/project/dpu-custom-yolo/docs/waveform_layer_timeline.png'
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"[OK] {out}")


if __name__ == '__main__':
    create_axi_waveform()
    create_mac_waveform()
    create_layer_timeline()
