"""
PHASE 4B: Multiplier Architecture Exploration
==============================================
Compare DSP-based vs LUT-based (shift-and-add) multiplication.

Target: ZedBoard (Zynq-7020)

Two approaches:
1. DSP-based: Use DSP48E1 slices (fast, limited quantity)
2. LUT-based: Use shift-and-add with LUTs (slower, more available)
3. Hybrid: Mix of both (parameterizable)
"""

import numpy as np
import json
from dataclasses import dataclass
from typing import List, Dict
import math

# =============================================================================
# ZEDBOARD (ZYNQ-7020) SPECIFICATIONS
# =============================================================================

ZEDBOARD_SPECS = {
    'name': 'ZedBoard (Zynq-7020)',
    'dsp48e1': 220,           # DSP slices
    'luts': 53200,            # 6-input LUTs
    'ffs': 106400,            # Flip-flops
    'bram_36kb': 140,         # 36Kb BRAM blocks
    'bram_kb': 140 * 36 // 8, # Total BRAM in KB = 630 KB
    'max_freq_mhz': 200,      # Realistic max frequency
}

print("=" * 80)
print("PHASE 4B: MULTIPLIER ARCHITECTURE EXPLORATION")
print("=" * 80)
print(f"\nTarget FPGA: {ZEDBOARD_SPECS['name']}")
print(f"  DSP48E1:  {ZEDBOARD_SPECS['dsp48e1']}")
print(f"  LUTs:     {ZEDBOARD_SPECS['luts']:,}")
print(f"  FFs:      {ZEDBOARD_SPECS['ffs']:,}")
print(f"  BRAM:     {ZEDBOARD_SPECS['bram_kb']} KB")


# =============================================================================
# MULTIPLIER IMPLEMENTATIONS
# =============================================================================

@dataclass
class MultiplierConfig:
    """Multiplier implementation configuration"""
    name: str
    type: str              # 'dsp', 'lut', 'hybrid'

    # Resource usage per multiplier
    dsps_per_mult: float
    luts_per_mult: int
    ffs_per_mult: int

    # Timing
    latency_cycles: int    # Pipeline latency
    throughput: float      # Multiplies per cycle (1.0 = fully pipelined)

    # Frequency impact
    max_freq_mhz: int

    def __str__(self):
        return f"{self.name}: {self.dsps_per_mult} DSP, {self.luts_per_mult} LUT, {self.latency_cycles} cycles"


# =============================================================================
# OPTION 1: DSP-BASED MULTIPLICATION
# =============================================================================

DSP_MULTIPLIER = MultiplierConfig(
    name="DSP48E1 (Native)",
    type="dsp",
    dsps_per_mult=1.0,      # 1 DSP per INT8x8 MAC
    luts_per_mult=20,       # Routing and control
    ffs_per_mult=30,        # Pipeline registers
    latency_cycles=3,       # DSP pipeline depth
    throughput=1.0,         # Fully pipelined
    max_freq_mhz=200,       # Can run at high frequency
)

# With DSP packing: 2 INT8x8 per DSP (using 18x25 mode)
DSP_PACKED_MULTIPLIER = MultiplierConfig(
    name="DSP48E1 (Packed 2x)",
    type="dsp",
    dsps_per_mult=0.5,      # 2 INT8x8 per DSP!
    luts_per_mult=40,       # Extra packing logic
    ffs_per_mult=50,
    latency_cycles=4,       # Extra cycle for packing
    throughput=1.0,
    max_freq_mhz=180,       # Slightly lower due to packing
)


# =============================================================================
# OPTION 2: LUT-BASED SHIFT-AND-ADD MULTIPLICATION
# =============================================================================

# Method 1: Fully parallel array multiplier
LUT_PARALLEL_MULTIPLIER = MultiplierConfig(
    name="LUT Array (Parallel)",
    type="lut",
    dsps_per_mult=0,
    luts_per_mult=64,       # 8x8 array of AND gates + adders
    ffs_per_mult=48,        # Pipeline registers
    latency_cycles=4,       # Adder tree depth
    throughput=1.0,         # Fully pipelined
    max_freq_mhz=150,       # Lower frequency due to LUT delays
)

# Method 2: Serial shift-and-add (8 cycles per multiply)
LUT_SERIAL_MULTIPLIER = MultiplierConfig(
    name="LUT Serial (Shift-Add)",
    type="lut",
    dsps_per_mult=0,
    luts_per_mult=24,       # Shifter + adder + control
    ffs_per_mult=40,        # Accumulator + state
    latency_cycles=8,       # 8 bits = 8 cycles
    throughput=0.125,       # 1 multiply per 8 cycles
    max_freq_mhz=200,       # Can run fast (simple logic)
)

# Method 3: Booth's algorithm (radix-4, 4 cycles)
LUT_BOOTH_MULTIPLIER = MultiplierConfig(
    name="LUT Booth Radix-4",
    type="lut",
    dsps_per_mult=0,
    luts_per_mult=40,       # Booth encoder + adder
    ffs_per_mult=50,
    latency_cycles=4,       # 8 bits / 2 bits per cycle = 4
    throughput=0.25,        # 1 multiply per 4 cycles
    max_freq_mhz=180,
)

# Method 4: Hybrid pipelined (2-stage, parallel with sharing)
LUT_PIPELINED_MULTIPLIER = MultiplierConfig(
    name="LUT Pipelined (2-stage)",
    type="lut",
    dsps_per_mult=0,
    luts_per_mult=48,       # Partial products + adder tree
    ffs_per_mult=64,        # Pipeline registers
    latency_cycles=2,       # 2-stage pipeline
    throughput=1.0,         # Fully pipelined after latency
    max_freq_mhz=160,
)


# =============================================================================
# OPTION 3: HYBRID APPROACH
# =============================================================================

def create_hybrid_config(dsp_ratio: float, name: str) -> MultiplierConfig:
    """
    Create hybrid configuration with given DSP/LUT ratio.

    dsp_ratio: 0.0 = all LUT, 1.0 = all DSP
    """
    # Blend resources
    dsps = dsp_ratio * DSP_MULTIPLIER.dsps_per_mult
    luts = int((1 - dsp_ratio) * LUT_PIPELINED_MULTIPLIER.luts_per_mult +
               dsp_ratio * DSP_MULTIPLIER.luts_per_mult)
    ffs = int((1 - dsp_ratio) * LUT_PIPELINED_MULTIPLIER.ffs_per_mult +
              dsp_ratio * DSP_MULTIPLIER.ffs_per_mult)

    # Frequency limited by slower component
    freq = int(dsp_ratio * DSP_MULTIPLIER.max_freq_mhz +
               (1 - dsp_ratio) * LUT_PIPELINED_MULTIPLIER.max_freq_mhz)

    return MultiplierConfig(
        name=name,
        type="hybrid",
        dsps_per_mult=dsps,
        luts_per_mult=luts,
        ffs_per_mult=ffs,
        latency_cycles=3,
        throughput=1.0,
        max_freq_mhz=freq,
    )


HYBRID_50_50 = create_hybrid_config(0.5, "Hybrid 50% DSP / 50% LUT")
HYBRID_25_75 = create_hybrid_config(0.25, "Hybrid 25% DSP / 75% LUT")


# =============================================================================
# ALL MULTIPLIER OPTIONS
# =============================================================================

ALL_MULTIPLIERS = [
    DSP_MULTIPLIER,
    DSP_PACKED_MULTIPLIER,
    LUT_PARALLEL_MULTIPLIER,
    LUT_SERIAL_MULTIPLIER,
    LUT_BOOTH_MULTIPLIER,
    LUT_PIPELINED_MULTIPLIER,
    HYBRID_50_50,
    HYBRID_25_75,
]


# =============================================================================
# MAC ARRAY ANALYSIS
# =============================================================================

@dataclass
class MACArrayConfig:
    """MAC array configuration"""
    name: str
    multiplier: MultiplierConfig
    mac_rows: int
    mac_cols: int
    freq_mhz: int

    @property
    def total_macs(self) -> int:
        return self.mac_rows * self.mac_cols

    @property
    def effective_macs_per_cycle(self) -> float:
        return self.total_macs * self.multiplier.throughput

    @property
    def dsps_used(self) -> int:
        return int(self.total_macs * self.multiplier.dsps_per_mult)

    @property
    def luts_used(self) -> int:
        # MAC array LUTs + accumulator + control overhead
        mac_luts = self.total_macs * self.multiplier.luts_per_mult
        acc_luts = self.mac_rows * 50  # 32-bit accumulators
        ctrl_luts = int(mac_luts * 0.2)  # 20% overhead
        return mac_luts + acc_luts + ctrl_luts

    @property
    def ffs_used(self) -> int:
        mac_ffs = self.total_macs * self.multiplier.ffs_per_mult
        acc_ffs = self.mac_rows * 32  # Accumulators
        ctrl_ffs = int(mac_ffs * 0.2)
        return mac_ffs + acc_ffs + ctrl_ffs

    @property
    def tops(self) -> float:
        return self.effective_macs_per_cycle * self.freq_mhz * 1e6 / 1e12


def calculate_max_array_size(mult: MultiplierConfig, specs: dict) -> tuple:
    """
    Calculate maximum MAC array size for given multiplier on target FPGA.
    """
    # Resource limits
    max_dsps = specs['dsp48e1']
    max_luts = int(specs['luts'] * 0.7)  # Reserve 30% for other logic
    max_ffs = int(specs['ffs'] * 0.7)

    # Calculate max MACs based on each resource
    if mult.dsps_per_mult > 0:
        max_macs_dsp = int(max_dsps / mult.dsps_per_mult)
    else:
        max_macs_dsp = float('inf')

    max_macs_lut = int(max_luts / (mult.luts_per_mult * 1.3))  # Include overhead
    max_macs_ff = int(max_ffs / (mult.ffs_per_mult * 1.3))

    max_macs = min(max_macs_dsp, max_macs_lut, max_macs_ff)

    # Find largest square array that fits
    side = int(math.sqrt(max_macs))
    # Round to power of 2 or nice number
    nice_sizes = [4, 8, 12, 16, 20, 24, 32, 48, 64]
    side = max([s for s in nice_sizes if s <= side], default=4)

    return side, side, max_macs


def analyze_multiplier_options():
    """Analyze all multiplier options on ZedBoard"""

    print("\n" + "=" * 80)
    print("[1] MULTIPLIER IMPLEMENTATIONS COMPARISON")
    print("=" * 80)

    print(f"\n{'Multiplier':<30} {'DSP':<6} {'LUTs':<8} {'FFs':<8} {'Lat':<6} {'Thru':<8} {'Freq':<8}")
    print("-" * 80)

    for mult in ALL_MULTIPLIERS:
        print(f"{mult.name:<30} {mult.dsps_per_mult:<6.1f} {mult.luts_per_mult:<8} "
              f"{mult.ffs_per_mult:<8} {mult.latency_cycles:<6} {mult.throughput:<8.2f} {mult.max_freq_mhz:<8}")

    return ALL_MULTIPLIERS


def analyze_mac_arrays():
    """Analyze MAC arrays with different multipliers on ZedBoard"""

    print("\n" + "=" * 80)
    print("[2] MAC ARRAY CONFIGURATIONS FOR ZEDBOARD")
    print("=" * 80)

    configs = []

    # Key multipliers to analyze
    key_multipliers = [
        DSP_MULTIPLIER,
        DSP_PACKED_MULTIPLIER,
        LUT_PIPELINED_MULTIPLIER,
        LUT_PARALLEL_MULTIPLIER,
        HYBRID_50_50,
    ]

    for mult in key_multipliers:
        rows, cols, max_macs = calculate_max_array_size(mult, ZEDBOARD_SPECS)

        # Create configuration
        config = MACArrayConfig(
            name=f"{mult.name} ({rows}x{cols})",
            multiplier=mult,
            mac_rows=rows,
            mac_cols=cols,
            freq_mhz=min(mult.max_freq_mhz, ZEDBOARD_SPECS['max_freq_mhz']),
        )

        configs.append(config)

        print(f"\n--- {mult.name} ---")
        print(f"  Max Array Size:    {rows}x{cols} = {rows*cols} MACs")
        print(f"  Effective MACs:    {config.effective_macs_per_cycle:.0f}/cycle")
        print(f"  Frequency:         {config.freq_mhz} MHz")
        print(f"  Peak TOPS:         {config.tops:.4f}")
        print(f"  Resources Used:")
        print(f"    DSPs:  {config.dsps_used:4} / {ZEDBOARD_SPECS['dsp48e1']} ({100*config.dsps_used/ZEDBOARD_SPECS['dsp48e1']:.1f}%)")
        print(f"    LUTs:  {config.luts_used:5} / {ZEDBOARD_SPECS['luts']} ({100*config.luts_used/ZEDBOARD_SPECS['luts']:.1f}%)")
        print(f"    FFs:   {config.ffs_used:5} / {ZEDBOARD_SPECS['ffs']} ({100*config.ffs_used/ZEDBOARD_SPECS['ffs']:.1f}%)")

    return configs


# =============================================================================
# YOLO PERFORMANCE ESTIMATION
# =============================================================================

# YOLOv4-tiny total MACs
YOLOV4_TINY_MACS = 3_453_938_176  # ~3.45 GMACs


def estimate_yolo_performance(configs: List[MACArrayConfig]):
    """Estimate YOLOv4-tiny performance for each configuration"""

    print("\n" + "=" * 80)
    print("[3] YOLOV4-TINY PERFORMANCE ESTIMATION")
    print("=" * 80)

    print(f"\nNetwork: YOLOv4-tiny")
    print(f"Total MACs: {YOLOV4_TINY_MACS:,} ({YOLOV4_TINY_MACS/1e9:.2f} GMACs)")

    results = []

    print(f"\n{'Configuration':<40} {'MACs/cyc':<12} {'Freq':<8} {'Cycles':<15} {'Time':<10} {'FPS':<8}")
    print("-" * 100)

    for config in configs:
        # Effective MACs per cycle (considering throughput)
        eff_macs = config.effective_macs_per_cycle

        # Ideal cycles (perfect utilization)
        ideal_cycles = YOLOV4_TINY_MACS / eff_macs

        # Add overhead for tiling, memory, etc. (~10%)
        total_cycles = ideal_cycles * 1.1

        # Time in milliseconds
        time_ms = total_cycles / (config.freq_mhz * 1e6) * 1000

        # FPS
        fps = 1000 / time_ms

        results.append({
            'name': config.name,
            'multiplier': config.multiplier.name,
            'array_size': f"{config.mac_rows}x{config.mac_cols}",
            'eff_macs': eff_macs,
            'freq_mhz': config.freq_mhz,
            'cycles': int(total_cycles),
            'time_ms': time_ms,
            'fps': fps,
            'dsps': config.dsps_used,
            'luts': config.luts_used,
            'tops': config.tops,
        })

        print(f"{config.name:<40} {eff_macs:<12.0f} {config.freq_mhz:<8} {int(total_cycles):<15,} {time_ms:<10.2f} {fps:<8.2f}")

    return results


def detailed_comparison(results: List[Dict]):
    """Detailed comparison of approaches"""

    print("\n" + "=" * 80)
    print("[4] DETAILED COMPARISON")
    print("=" * 80)

    # Sort by FPS
    sorted_results = sorted(results, key=lambda x: x['fps'], reverse=True)

    print(f"\n{'Rank':<6} {'Configuration':<35} {'FPS':<10} {'DSPs':<8} {'LUTs':<10} {'Efficiency':<12}")
    print("-" * 90)

    for i, r in enumerate(sorted_results, 1):
        efficiency = r['fps'] / max(r['dsps'], 1)  # FPS per DSP (or per unit if no DSP)
        lut_eff = r['fps'] / (r['luts'] / 1000)  # FPS per K-LUTs
        print(f"{i:<6} {r['name']:<35} {r['fps']:<10.2f} {r['dsps']:<8} {r['luts']:<10,} {efficiency:<12.3f}")

    # Best options analysis
    print("\n" + "=" * 80)
    print("[5] ANALYSIS BY CATEGORY")
    print("=" * 80)

    # Best DSP-based
    dsp_results = [r for r in results if 'DSP' in r['multiplier']]
    if dsp_results:
        best_dsp = max(dsp_results, key=lambda x: x['fps'])
        print(f"\n  BEST DSP-BASED:")
        print(f"    {best_dsp['name']}")
        print(f"    FPS: {best_dsp['fps']:.2f}")
        print(f"    DSPs: {best_dsp['dsps']}")

    # Best LUT-based
    lut_results = [r for r in results if 'LUT' in r['multiplier']]
    if lut_results:
        best_lut = max(lut_results, key=lambda x: x['fps'])
        print(f"\n  BEST LUT-BASED (for ASIC compatibility):")
        print(f"    {best_lut['name']}")
        print(f"    FPS: {best_lut['fps']:.2f}")
        print(f"    LUTs: {best_lut['luts']:,}")
        print(f"    DSPs: {best_lut['dsps']} (zero!)")

    # Best hybrid
    hybrid_results = [r for r in results if 'Hybrid' in r['multiplier']]
    if hybrid_results:
        best_hybrid = max(hybrid_results, key=lambda x: x['fps'])
        print(f"\n  BEST HYBRID:")
        print(f"    {best_hybrid['name']}")
        print(f"    FPS: {best_hybrid['fps']:.2f}")
        print(f"    DSPs: {best_hybrid['dsps']}, LUTs: {best_hybrid['luts']:,}")

    return sorted_results


def asic_considerations():
    """Analyze ASIC implementation considerations"""

    print("\n" + "=" * 80)
    print("[6] ASIC IMPLEMENTATION CONSIDERATIONS")
    print("=" * 80)

    print("""
    LUT-BASED MULTIPLIER FOR ASIC:
    ==============================

    Advantages:
    -----------
    1. PORTABLE: No vendor-specific DSP blocks
       - Direct translation to standard cells
       - Same RTL works on any ASIC process

    2. SCALABLE: Can instantiate any number of multipliers
       - Not limited by DSP count
       - Limited only by area budget

    3. CUSTOMIZABLE: Can optimize for specific use case
       - INT8 x INT8 only (no need for wider multipliers)
       - Can share partial product logic

    4. POWER EFFICIENT: In ASIC, shift-add can be very efficient
       - No overhead of general-purpose DSP
       - Can use clock gating per multiplier

    Shift-and-Add Implementation:
    -----------------------------

    For INT8 x INT8:

        A x B = sum( (A x B[i]) << i )   for i = 0 to 7

    Hardware:
        +-------+     +-------+     +-------+
        |  B[0] |---->| A×1   |---->|       |
        +-------+     +-------+     |       |
        +-------+     +-------+     |  ADD  |---> Product
        |  B[1] |---->| A×2   |---->|  TREE |
        +-------+     +-------+     |       |
           ...           ...        |       |
        +-------+     +-------+     |       |
        |  B[7] |---->| A×128 |---->|       |
        +-------+     +-------+     +-------+

    Where A×2^n is just A shifted left by n bits.

    Pipeline Options:
    -----------------

    1. FULLY PARALLEL (1 cycle, max area):
       - All 8 partial products computed simultaneously
       - Adder tree combines them
       - ~64 LUTs equivalent per multiplier

    2. SERIAL (8 cycles, min area):
       - 1 partial product per cycle
       - Accumulate result
       - ~24 LUTs equivalent per multiplier
       - 8x more multipliers possible, but 8x slower each

    3. RADIX-4 BOOTH (4 cycles, balanced):
       - 2 bits per cycle
       - ~40 LUTs equivalent per multiplier
       - Good balance of area and speed

    RECOMMENDATION FOR ASIC:
    ========================

    Use PIPELINED PARALLEL approach:
    - Fully parallel partial products
    - 2-stage pipelined adder tree
    - ~48 LUTs equivalent
    - Throughput: 1 multiply per cycle (after 2-cycle latency)

    This translates well to ASIC:
    - Predictable timing
    - Regular structure
    - Easy to synthesize
    - Good for place-and-route
    """)


def generate_recommendations(results: List[Dict]):
    """Generate final recommendations"""

    print("\n" + "=" * 80)
    print("[7] RECOMMENDATIONS")
    print("=" * 80)

    # Find best of each type
    dsp_results = [r for r in results if 'DSP' in r['multiplier'] and r['dsps'] > 0]
    lut_results = [r for r in results if r['dsps'] == 0]

    best_dsp = max(dsp_results, key=lambda x: x['fps']) if dsp_results else None
    best_lut = max(lut_results, key=lambda x: x['fps']) if lut_results else None

    print(f"""
    FOR ZEDBOARD (Zynq-7020):
    =========================

    OPTION A: DSP-BASED (Maximum Performance)
    -----------------------------------------
    Configuration: {best_dsp['name'] if best_dsp else 'N/A'}
    Performance:   {best_dsp['fps']:.2f} FPS
    Resources:     {best_dsp['dsps']} DSPs, {best_dsp['luts']:,} LUTs

    Pros: Fastest, proven approach
    Cons: Limited by DSP count, not ASIC-portable

    OPTION B: LUT-BASED (ASIC Compatible)
    -------------------------------------
    Configuration: {best_lut['name'] if best_lut else 'N/A'}
    Performance:   {best_lut['fps']:.2f} FPS
    Resources:     0 DSPs, {best_lut['luts']:,} LUTs

    Pros: ASIC-portable, more scalable
    Cons: Lower frequency, higher LUT usage

    PERFORMANCE COMPARISON:
    -----------------------
    DSP approach:  {best_dsp['fps']:.2f} FPS
    LUT approach:  {best_lut['fps']:.2f} FPS
    Difference:    LUT is {((best_lut['fps']/best_dsp['fps'])-1)*100:.1f}% faster than DSP (on ZedBoard)

    RECOMMENDATION:
    ---------------
    """)

    if best_lut and best_lut['fps'] >= 10:
        print(f"""    For ASIC development path, the LUT-based approach at {best_lut['fps']:.1f} FPS
    is viable and worth considering:

    1. Start with LUT-based design on FPGA for verification
    2. Same RTL can be synthesized directly to ASIC
    3. No DSP blocks to replace/remap
    4. Performance gap ({((best_dsp['fps']/best_lut['fps'])-1)*100:.1f}%) may be acceptable

    For pure FPGA deployment, use DSP-based for maximum performance.
    """)
    else:
        print(f"""    The LUT-based approach is too slow ({best_lut['fps']:.1f} FPS).
    Consider:
    1. Hybrid approach with some DSPs
    2. Smaller YOLO model (YOLOv3-tiny is simpler)
    3. Lower resolution input
    """)


def save_results(results: List[Dict]):
    """Save results to JSON"""

    output = {
        'phase': '4b',
        'target': ZEDBOARD_SPECS,
        'multiplier_options': [
            {
                'name': m.name,
                'type': m.type,
                'dsps_per_mult': m.dsps_per_mult,
                'luts_per_mult': m.luts_per_mult,
                'latency': m.latency_cycles,
                'throughput': m.throughput,
                'max_freq': m.max_freq_mhz,
            }
            for m in ALL_MULTIPLIERS
        ],
        'performance_results': results,
        'recommendations': {
            'fpga_deployment': 'DSP-based for max performance',
            'asic_path': 'LUT-based (pipelined) for portability',
        }
    }

    with open('phase4b_multiplier_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 80)
    print("PHASE 4B COMPLETE")
    print("=" * 80)
    print("Results saved to: phase4b_multiplier_results.json")


def main():
    """Main function"""

    # Analyze multiplier options
    multipliers = analyze_multiplier_options()

    # Analyze MAC arrays
    mac_configs = analyze_mac_arrays()

    # Estimate YOLO performance
    results = estimate_yolo_performance(mac_configs)

    # Detailed comparison
    sorted_results = detailed_comparison(results)

    # ASIC considerations
    asic_considerations()

    # Recommendations
    generate_recommendations(results)

    # Save results
    save_results(results)


if __name__ == "__main__":
    main()
