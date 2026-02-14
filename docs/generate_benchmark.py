#!/usr/bin/env python3
"""
Benchmark comparison: DPU Custom YOLOv4-tiny vs commercial/academic DPUs.
Generates a comparison table as PNG.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def create_benchmark_table():
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.axis('off')
    fig.patch.set_facecolor('#FAFBFC')

    ax.text(0.5, 0.97, 'DPU Benchmark Comparison', transform=ax.transAxes,
            ha='center', va='top', fontsize=20, fontweight='bold', color='#1a1a2e')
    ax.text(0.5, 0.935, 'Custom DPU vs Commercial/Academic AI Accelerators',
            transform=ax.transAxes, ha='center', va='top', fontsize=11, color='#666')

    # Data
    headers = [
        'Feature',
        'This Work\n(DPU Custom)',
        'Xilinx DPU\n(DPUCZDX8G)',
        'Google\nEdge TPU',
        'ARM\nEthos-U55',
        'NVIDIA DLA\n(Orin Nano)',
    ]

    rows = [
        ['Target Model',    'YOLOv4-tiny',   'ResNet/YOLO',    'MobileNet/SSD',  'MobileNet',     'YOLO/ResNet'],
        ['Precision',       'INT8',           'INT8/INT16',     'INT8',           'INT8/INT16',    'INT8/FP16'],
        ['MAC Units',       '1,024',          '4,096',          '~8,000 (est)',   '256',           '2x 1,024'],
        ['MACs/cycle',      '1,024',          '4,096',          '~4,000',         '256',           '2,048'],
        ['Frequency',       '100 MHz',        '300 MHz',        '500 MHz',        '400 MHz',       '600 MHz'],
        ['Peak TOPS',       '0.1',            '1.2',            '4.0',            '0.1',           '1.2'],
        ['Memory (on-chip)','272 KB BRAM',    '~1 MB URAM',    'Fixed (8 MB)',   '256 KB SRAM',   'Shared L2'],
        ['Memory Bandwidth','3.2 GB/s',       '19.2 GB/s',     '16 GB/s',        '3.2 GB/s',      '34 GB/s'],
        ['Power (est.)',    '0.5-1 W',        '5-15 W',        '2 W',            '0.05-0.5 W',   '7-15 W'],
        ['TOPS/W',          '0.1-0.2',        '0.1-0.2',       '2.0',            '0.2-2.0',       '0.1-0.2'],
        ['Host Interface',  'AXI4-Lite+Stream','AXI4',         'USB/PCIe',       'AXI/AHB',       'NVDLA AXI'],
        ['Programmable',    'Full RTL source', 'Vitis AI tools', 'Fixed function', 'Vela compiler', 'SDK/TRT'],
        ['Open Source',     'YES',            'Partial (DPU IP)','No',            'Partial',       'Yes (HW)'],
        ['FPGA Target',     'Zynq-7020',      'Zynq US+/Versal','ASIC',          'ASIC',          'ASIC'],
        ['YOLOv4-tiny FPS', '~6.5 (16x16)\n~0.4 (416x416)',
                                              '~30-100',       '~25 (SSD)',      '~2-5',          '~60-120'],
    ]

    col_colors = ['#E8EAF6', '#C5E1A5', '#BBDEFB', '#FFECB3', '#F8BBD0', '#D1C4E9']
    row_colors_alt = ['#FFFFFF', '#F5F5F5']

    # Create table
    n_cols = len(headers)
    n_rows = len(rows)

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc='center',
        cellLoc='center',
        bbox=[0.0, 0.02, 1.0, 0.87],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)

    # Style headers
    for j in range(n_cols):
        cell = table[0, j]
        cell.set_facecolor(col_colors[j])
        cell.set_text_props(fontweight='bold', fontsize=9)
        cell.set_height(0.07)
        cell.set_edgecolor('#BDBDBD')

    # Style data cells
    for i in range(n_rows):
        for j in range(n_cols):
            cell = table[i + 1, j]
            if j == 0:
                cell.set_facecolor('#ECEFF1')
                cell.set_text_props(fontweight='bold', fontsize=8)
            elif j == 1:
                cell.set_facecolor('#E8F5E9')
                cell.set_text_props(fontweight='bold', fontsize=8, color='#1B5E20')
            else:
                cell.set_facecolor(row_colors_alt[i % 2])
                cell.set_text_props(fontsize=8)
            cell.set_edgecolor('#E0E0E0')
            cell.set_height(0.052)

    # Highlight "This Work" column header
    table[0, 1].set_facecolor('#66BB6A')
    table[0, 1].set_text_props(fontweight='bold', fontsize=9, color='white')

    # Footer notes
    ax.text(0.5, 0.005,
            'Notes: Peak TOPS = MACs/cycle x Frequency x 2 (multiply+add) / 10^12. '
            'Power estimates are typical for the target platform. '
            'FPS values depend on input resolution and model variant.',
            transform=ax.transAxes, ha='center', fontsize=7, color='#888',
            style='italic')

    plt.tight_layout()
    out_path = 'C:/project/dpu-custom-yolo/docs/benchmark_comparison.png'
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"[OK] {out_path}")


def create_performance_chart():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor('#FAFBFC')

    # ---- Chart 1: Peak TOPS ----
    names =  ['This Work', 'Xilinx DPU', 'Edge TPU', 'Ethos-U55', 'NVIDIA DLA']
    tops =   [0.1,          1.2,           4.0,        0.1,          1.2]
    colors = ['#2E7D32',   '#1565C0',    '#E65100',  '#6A1B9A',   '#C62828']

    bars = ax1.barh(names, tops, color=colors, alpha=0.85, edgecolor='white', height=0.6)
    ax1.set_xlabel('Peak TOPS (INT8)', fontsize=11, fontweight='bold')
    ax1.set_title('Peak Throughput Comparison', fontsize=13, fontweight='bold')
    ax1.set_xlim(0, 5)
    ax1.grid(axis='x', alpha=0.3)
    for bar, val in zip(bars, tops):
        ax1.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                f'{val} TOPS', va='center', fontsize=9, fontweight='bold')

    # ---- Chart 2: Efficiency (TOPS/W) ----
    topsw = [0.15,         0.15,          2.0,        1.0,          0.15]
    bars2 = ax2.barh(names, topsw, color=colors, alpha=0.85, edgecolor='white', height=0.6)
    ax2.set_xlabel('TOPS/W (Energy Efficiency)', fontsize=11, fontweight='bold')
    ax2.set_title('Energy Efficiency Comparison', fontsize=13, fontweight='bold')
    ax2.set_xlim(0, 2.5)
    ax2.grid(axis='x', alpha=0.3)
    for bar, val in zip(bars2, topsw):
        ax2.text(bar.get_width() + 0.03, bar.get_y() + bar.get_height() / 2,
                f'{val} TOPS/W', va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    out_path = 'C:/project/dpu-custom-yolo/docs/benchmark_charts.png'
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"[OK] {out_path}")


if __name__ == '__main__':
    create_benchmark_table()
    create_performance_chart()
