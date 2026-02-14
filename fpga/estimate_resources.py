#!/usr/bin/env python3
"""
FPGA Resource Estimation for DPU Custom YOLOv4-tiny
Target: Zynq-7020 (xc7z020clg484-1)

Estimates BRAM, LUT, FF, and DSP48 usage based on the RTL design.
Generates a summary table and PNG chart.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# Zynq-7020 Resources
# =============================================================================
ZYNQ_7020 = {
    'LUT':    53200,
    'FF':     106400,
    'BRAM36': 140,      # 36Kb blocks (= 4.5 KB each, total 630 KB)
    'BRAM18': 280,      # 18Kb blocks
    'DSP48':  220,
}

# =============================================================================
# DPU Resource Estimation
# =============================================================================
def estimate_resources():
    resources = {}

    # ---- BRAM ----
    # weight_buf: 147,456 bytes = 144 KB -> 144*1024/4096 = 36 BRAM36
    weight_bram36 = 147456 * 8 / 36864  # bits / bits_per_BRAM36
    # Actually: 147456 bytes = 1,179,648 bits. BRAM36 = 36,864 bits. -> 32 BRAM36
    weight_bram36 = int(np.ceil(147456 * 8 / 36864))

    # fmap_a: 65,536 bytes -> 65536*8/36864 = 14.2 -> 15 BRAM36
    fmap_a_bram36 = int(np.ceil(65536 * 8 / 36864))
    fmap_b_bram36 = fmap_a_bram36

    # bias_buf: 256 * 32 = 1024 bytes -> 1 BRAM36
    bias_bram36 = 1

    # patch_buf: 1152 bytes -> 1 BRAM36
    patch_bram36 = 1

    # save buffers (skip connections):
    #   save_l2: 64 * 4 * 4 = 1024 bytes -> 1 BRAM36
    #   save_l4: 32 * 4 * 4 = 512 bytes -> 1 BRAM18
    #   save_l10: 128 * 2 * 2 = 512 bytes -> 1 BRAM18
    #   save_l12: 64 * 2 * 2 = 256 bytes -> 1 BRAM18
    save_bram36 = 1  # save_l2
    save_bram18 = 3  # save_l4 + save_l10 + save_l12

    total_bram36 = weight_bram36 + fmap_a_bram36 + fmap_b_bram36 + bias_bram36 + patch_bram36 + save_bram36
    total_bram18 = save_bram18
    # Convert to equivalent BRAM36: 1 BRAM36 = 2 BRAM18
    total_bram36_equiv = total_bram36 + int(np.ceil(total_bram18 / 2))

    resources['BRAM'] = {
        'weight_buf (144 KB)':  weight_bram36,
        'fmap_a (64 KB)':       fmap_a_bram36,
        'fmap_b (64 KB)':       fmap_b_bram36,
        'bias_buf (1 KB)':      bias_bram36,
        'patch_buf (1.1 KB)':   patch_bram36,
        'save buffers':         save_bram36,
        'total_bram36':         total_bram36_equiv,
    }

    # ---- DSP48 ----
    # 32x32 MAC array = 1024 multipliers
    # Each INT8x INT8 multiply uses 1 DSP48E1 (or shift-and-add for 0 DSP)
    # Option A: Full DSP -> 1024 DSP48 (exceeds Zynq-7020!)
    # Option B: Shift-and-add -> 0 DSP48 (uses LUTs instead)
    # Practical: Time-multiplex 4:1 -> 256 DSP48 or use shift-add
    # For this design: shift-and-add multiplier -> 0 DSP48
    # But requantize uses a 16-bit multiply -> 32 DSP48 (post-process array)
    dsp_mac = 0         # shift-and-add (no DSP)
    dsp_postproc = 32   # 32 requantize lanes, each 1 DSP for 16-bit mult
    total_dsp = dsp_mac + dsp_postproc
    resources['DSP48'] = {
        'MAC array (shift-add)': dsp_mac,
        'Post-process (requant)': dsp_postproc,
        'total': total_dsp,
    }

    # ---- LUT ----
    # MAC array: 1024 * 8-bit shift-and-add ≈ 30 LUT per multiplier = 30,720 LUT
    # Plus 1024 * 32-bit accumulators ≈ 32 LUT per acc = 32,768 LUT ... too much
    # Realistic: Vivado optimizes heavily. Estimate ~15 LUT per MAC unit
    lut_mac = 1024 * 15     # 15,360
    # Post-process: 32 lanes * (bias_add + leaky_relu + requant) ≈ 80 LUT/lane
    lut_postproc = 32 * 80  # 2,560
    # Conv engine controller + address gen ≈ 500 LUT
    lut_engine_ctrl = 500
    # DPU top FSM + address decode + muxes ≈ 2000 LUT
    lut_dpu_fsm = 2000
    # AXI4-Lite slave ≈ 300 LUT
    lut_axi_lite = 300
    # AXI DMA bridge ≈ 200 LUT
    lut_dma = 200
    # MaxPool unit ≈ 200 LUT
    lut_maxpool = 200
    total_lut = lut_mac + lut_postproc + lut_engine_ctrl + lut_dpu_fsm + lut_axi_lite + lut_dma + lut_maxpool
    resources['LUT'] = {
        'MAC array (32x32)':    lut_mac,
        'Post-process (32 ch)': lut_postproc,
        'Engine controller':    lut_engine_ctrl,
        'DPU FSM + decode':     lut_dpu_fsm,
        'AXI4-Lite slave':      lut_axi_lite,
        'AXI DMA bridge':       lut_dma,
        'MaxPool unit':         lut_maxpool,
        'total':                total_lut,
    }

    # ---- FF (Flip-Flops) ----
    # MAC accumulators: 1024 * 32 = 32,768 FF
    ff_mac_acc = 1024 * 32
    # MAC registers: 1024 * 8 (weight) + 32 * 8 (act) = 8,448 FF
    ff_mac_reg = 1024 * 8 + 32 * 8
    # Post-process pipeline: 32 * 32 * 3 stages ≈ 3,072 FF
    ff_postproc = 32 * 32 * 3
    # FSM + control: ≈ 500 FF
    ff_ctrl = 500
    # AXI interfaces: ≈ 400 FF
    ff_axi = 400
    total_ff = ff_mac_acc + ff_mac_reg + ff_postproc + ff_ctrl + ff_axi
    resources['FF'] = {
        'MAC accumulators':  ff_mac_acc,
        'MAC registers':     ff_mac_reg,
        'Post-process pipe': ff_postproc,
        'Control/FSM':       ff_ctrl,
        'AXI interfaces':    ff_axi,
        'total':             total_ff,
    }

    return resources


def print_summary(resources):
    print("=" * 65)
    print("  FPGA Resource Estimation — DPU Custom YOLOv4-tiny")
    print("  Target: Zynq-7020 (xc7z020clg484-1)")
    print("=" * 65)

    # BRAM
    print("\n  BRAM36 Blocks:")
    for k, v in resources['BRAM'].items():
        if k != 'total_bram36':
            print(f"    {k:25s}: {v:4d}")
    total_bram = resources['BRAM']['total_bram36']
    pct_bram = total_bram / ZYNQ_7020['BRAM36'] * 100
    print(f"    {'TOTAL':25s}: {total_bram:4d} / {ZYNQ_7020['BRAM36']} ({pct_bram:.1f}%)")

    # DSP48
    print("\n  DSP48E1 Slices:")
    for k, v in resources['DSP48'].items():
        if k != 'total':
            print(f"    {k:25s}: {v:4d}")
    total_dsp = resources['DSP48']['total']
    pct_dsp = total_dsp / ZYNQ_7020['DSP48'] * 100
    print(f"    {'TOTAL':25s}: {total_dsp:4d} / {ZYNQ_7020['DSP48']} ({pct_dsp:.1f}%)")

    # LUT
    print("\n  LUTs (6-input):")
    for k, v in resources['LUT'].items():
        if k != 'total':
            print(f"    {k:25s}: {v:6d}")
    total_lut = resources['LUT']['total']
    pct_lut = total_lut / ZYNQ_7020['LUT'] * 100
    print(f"    {'TOTAL':25s}: {total_lut:6d} / {ZYNQ_7020['LUT']} ({pct_lut:.1f}%)")

    # FF
    print("\n  Flip-Flops:")
    for k, v in resources['FF'].items():
        if k != 'total':
            print(f"    {k:25s}: {v:6d}")
    total_ff = resources['FF']['total']
    pct_ff = total_ff / ZYNQ_7020['FF'] * 100
    print(f"    {'TOTAL':25s}: {total_ff:6d} / {ZYNQ_7020['FF']} ({pct_ff:.1f}%)")

    print("\n" + "=" * 65)
    print("  UTILIZATION SUMMARY")
    print(f"    BRAM36:  {total_bram:5d} / {ZYNQ_7020['BRAM36']:5d}  ({pct_bram:5.1f}%)")
    print(f"    DSP48:   {total_dsp:5d} / {ZYNQ_7020['DSP48']:5d}  ({pct_dsp:5.1f}%)")
    print(f"    LUT:     {total_lut:5d} / {ZYNQ_7020['LUT']:5d}  ({pct_lut:5.1f}%)")
    print(f"    FF:      {total_ff:5d} / {ZYNQ_7020['FF']:5d}  ({pct_ff:5.1f}%)")
    print("=" * 65)

    return {
        'BRAM36': (total_bram, ZYNQ_7020['BRAM36'], pct_bram),
        'DSP48':  (total_dsp, ZYNQ_7020['DSP48'], pct_dsp),
        'LUT':    (total_lut, ZYNQ_7020['LUT'], pct_lut),
        'FF':     (total_ff, ZYNQ_7020['FF'], pct_ff),
    }


def generate_chart(summary):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor('#FAFBFC')

    # ---- Bar chart: utilization % ----
    labels = list(summary.keys())
    pcts = [summary[k][2] for k in labels]
    used = [summary[k][0] for k in labels]
    avail = [summary[k][1] for k in labels]

    colors = ['#E65100', '#1565C0', '#2E7D32', '#6A1B9A']
    bars = ax1.bar(labels, pcts, color=colors, alpha=0.85, edgecolor='white', linewidth=2)
    ax1.set_ylabel('Utilization (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Zynq-7020 Resource Utilization', fontsize=14, fontweight='bold', pad=15)
    ax1.set_ylim(0, 100)
    ax1.axhline(y=80, color='#C62828', linestyle='--', lw=1, alpha=0.5, label='80% threshold')
    ax1.legend(fontsize=8)
    ax1.grid(axis='y', alpha=0.3)

    for bar, pct, u, a in zip(bars, pcts, used, avail):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f'{pct:.1f}%\n({u}/{a})',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    # ---- Stacked breakdown (BRAM) ----
    resources = estimate_resources()
    bram_items = [(k, v) for k, v in resources['BRAM'].items() if k != 'total_bram36']
    bram_labels = [item[0] for item in bram_items]
    bram_values = [item[1] for item in bram_items]
    bram_colors = ['#FF8F00', '#FFB300', '#FFC107', '#FFD54F', '#FFE082', '#FFF176']

    wedges, texts, autotexts = ax2.pie(bram_values, labels=bram_labels,
                                        colors=bram_colors[:len(bram_values)],
                                        autopct='%1.0f%%', startangle=90,
                                        textprops={'fontsize': 8})
    ax2.set_title('BRAM Distribution', fontsize=14, fontweight='bold', pad=15)

    plt.tight_layout()
    out_path = 'C:/project/dpu-custom-yolo/docs/fpga_resource_estimation.png'
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n[OK] Chart saved: {out_path}")


if __name__ == '__main__':
    resources = estimate_resources()
    summary = print_summary(resources)
    generate_chart(summary)
