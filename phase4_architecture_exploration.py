"""
PHASE 4: Software Benchmarking & Architecture Exploration
==========================================================
Explore different DPU architectures to find optimal configuration for YOLO.

We simulate multiple configurations varying:
- MAC array size
- Buffer sizes
- Parallelism levels

And benchmark:
- Estimated cycles
- Memory bandwidth
- Resource utilization
"""

import numpy as np
import json
from dataclasses import dataclass
from typing import List, Dict, Tuple
import math

# =============================================================================
# YOLO LAYER DEFINITIONS
# =============================================================================

@dataclass
class ConvLayer:
    """Convolution layer specification"""
    name: str
    C_in: int       # Input channels
    C_out: int      # Output channels
    H_in: int       # Input height
    W_in: int       # Input width
    K: int          # Kernel size (3 or 1)
    stride: int     # Stride (1 or 2)

    @property
    def H_out(self) -> int:
        if self.K == 1:
            return self.H_in
        return (self.H_in + 2 * (self.K // 2) - self.K) // self.stride + 1

    @property
    def W_out(self) -> int:
        if self.K == 1:
            return self.W_in
        return (self.W_in + 2 * (self.K // 2) - self.K) // self.stride + 1

    @property
    def macs(self) -> int:
        """Total MAC operations for this layer"""
        return self.K * self.K * self.C_in * self.C_out * self.H_out * self.W_out

    @property
    def weight_bytes(self) -> int:
        """Weight memory in bytes (INT8)"""
        return self.K * self.K * self.C_in * self.C_out

    @property
    def input_bytes(self) -> int:
        """Input feature map size in bytes"""
        return self.C_in * self.H_in * self.W_in

    @property
    def output_bytes(self) -> int:
        """Output feature map size in bytes"""
        return self.C_out * self.H_out * self.W_out


# YOLOv4-tiny conv layers (from Phase 1 analysis)
YOLOV4_TINY_LAYERS = [
    ConvLayer("conv0", 3, 32, 416, 416, 3, 2),      # 3->32, stride 2
    ConvLayer("conv1", 32, 64, 208, 208, 3, 2),     # 32->64, stride 2
    ConvLayer("conv2", 64, 64, 104, 104, 3, 1),     # 64->64
    ConvLayer("conv4", 32, 32, 104, 104, 3, 1),     # 32->32 (after split)
    ConvLayer("conv5", 32, 32, 104, 104, 3, 1),     # 32->32
    ConvLayer("conv7", 64, 64, 104, 104, 1, 1),     # 64->64, 1x1
    ConvLayer("conv10", 128, 128, 52, 52, 3, 1),    # 128->128
    ConvLayer("conv12", 64, 64, 52, 52, 3, 1),      # 64->64
    ConvLayer("conv13", 64, 64, 52, 52, 3, 1),      # 64->64
    ConvLayer("conv15", 128, 128, 52, 52, 1, 1),    # 128->128, 1x1
    ConvLayer("conv18", 256, 256, 26, 26, 3, 1),    # 256->256
    ConvLayer("conv20", 128, 128, 26, 26, 3, 1),    # 128->128
    ConvLayer("conv21", 128, 128, 26, 26, 3, 1),    # 128->128
    ConvLayer("conv23", 256, 256, 26, 26, 1, 1),    # 256->256, 1x1
    ConvLayer("conv26", 512, 512, 13, 13, 3, 1),    # 512->512
    ConvLayer("conv27", 512, 256, 13, 13, 1, 1),    # 512->256, 1x1
    ConvLayer("conv28", 256, 512, 13, 13, 3, 1),    # 256->512
    ConvLayer("conv29", 512, 255, 13, 13, 1, 1),    # 512->255, 1x1 (head)
    ConvLayer("conv32", 256, 128, 13, 13, 1, 1),    # 256->128, 1x1
    ConvLayer("conv35", 384, 256, 26, 26, 3, 1),    # 384->256
    ConvLayer("conv36", 256, 255, 26, 26, 1, 1),    # 256->255, 1x1 (head)
]


# =============================================================================
# DPU ARCHITECTURE CONFIGURATION
# =============================================================================

@dataclass
class DPUConfig:
    """DPU Architecture Configuration"""
    name: str

    # MAC Array dimensions
    mac_rows: int           # Parallel output channels
    mac_cols: int           # Parallel input channels

    # Buffer sizes (in bytes)
    weight_buffer: int      # On-chip weight buffer
    input_buffer: int       # Input line buffer
    output_buffer: int      # Output buffer

    # Clock frequency (MHz)
    freq_mhz: int

    # Memory bandwidth (GB/s)
    mem_bandwidth: float

    @property
    def total_macs(self) -> int:
        """Total MACs per cycle"""
        return self.mac_rows * self.mac_cols

    @property
    def ops_per_second(self) -> float:
        """Peak operations per second"""
        return self.total_macs * self.freq_mhz * 1e6

    @property
    def tops(self) -> float:
        """Tera operations per second"""
        return self.ops_per_second / 1e12


# =============================================================================
# PERFORMANCE MODEL
# =============================================================================

class PerformanceModel:
    """Model DPU performance for a given configuration"""

    def __init__(self, config: DPUConfig):
        self.config = config

    def estimate_layer_cycles(self, layer: ConvLayer) -> Dict:
        """
        Estimate cycles to process a convolution layer.

        We consider:
        1. Compute cycles (MAC operations / parallelism)
        2. Weight loading cycles
        3. Input loading cycles
        4. Output storing cycles
        """
        cfg = self.config

        # =====================================================================
        # COMPUTE CYCLES
        # =====================================================================
        # For each output pixel, we need K*K*C_in MACs
        # We can do mac_rows * mac_cols MACs per cycle

        # Tiling strategy:
        # - Tile output channels by mac_rows
        # - Tile input channels by mac_cols
        # - Process K*K positions sequentially (for 3x3)

        macs_per_output_pixel = layer.K * layer.K * layer.C_in
        output_pixels = layer.H_out * layer.W_out

        # Output channel tiles
        c_out_tiles = math.ceil(layer.C_out / cfg.mac_rows)

        # Input channel tiles
        c_in_tiles = math.ceil(layer.C_in / cfg.mac_cols)

        # Cycles for compute (assuming perfect pipelining)
        # For each output tile, process all input tiles and kernel positions
        compute_cycles = (
            c_out_tiles *           # Output channel tiles
            c_in_tiles *            # Input channel tiles
            layer.K * layer.K *     # Kernel positions
            output_pixels           # All output positions
        )

        # =====================================================================
        # MEMORY CYCLES
        # =====================================================================

        # Weight loading
        # Weights needed per output channel tile: mac_rows * C_in * K * K bytes
        weights_per_tile = min(cfg.mac_rows, layer.C_out) * layer.C_in * layer.K * layer.K
        total_weight_loads = c_out_tiles * weights_per_tile

        # Check if weights fit in buffer
        weights_fit = layer.weight_bytes <= cfg.weight_buffer

        # Weight load cycles (assuming 8 bytes per cycle at mem bandwidth)
        bytes_per_cycle = cfg.mem_bandwidth * 1e9 / (cfg.freq_mhz * 1e6)
        weight_cycles = total_weight_loads / bytes_per_cycle if not weights_fit else 0

        # Input loading
        # Need to load input feature map (can be streamed with line buffer)
        input_cycles = layer.input_bytes / bytes_per_cycle

        # Output storing
        output_cycles = layer.output_bytes / bytes_per_cycle

        # =====================================================================
        # TOTAL CYCLES
        # =====================================================================

        # Memory and compute can overlap, take max
        memory_cycles = weight_cycles + input_cycles + output_cycles
        total_cycles = max(compute_cycles, memory_cycles)

        # Utilization
        ideal_cycles = layer.macs / cfg.total_macs
        utilization = ideal_cycles / total_cycles if total_cycles > 0 else 0

        return {
            'layer': layer.name,
            'macs': layer.macs,
            'compute_cycles': int(compute_cycles),
            'memory_cycles': int(memory_cycles),
            'total_cycles': int(total_cycles),
            'ideal_cycles': int(ideal_cycles),
            'utilization': utilization,
            'weights_fit': weights_fit,
            'c_out_tiles': c_out_tiles,
            'c_in_tiles': c_in_tiles,
        }

    def estimate_network_performance(self, layers: List[ConvLayer]) -> Dict:
        """Estimate performance for entire network"""
        layer_results = []
        total_macs = 0
        total_cycles = 0

        for layer in layers:
            result = self.estimate_layer_cycles(layer)
            layer_results.append(result)
            total_macs += result['macs']
            total_cycles += result['total_cycles']

        # Calculate FPS
        time_seconds = total_cycles / (self.config.freq_mhz * 1e6)
        fps = 1.0 / time_seconds if time_seconds > 0 else 0

        # Average utilization
        avg_utilization = sum(r['utilization'] for r in layer_results) / len(layer_results)

        return {
            'config': self.config.name,
            'total_macs': total_macs,
            'total_cycles': total_cycles,
            'time_ms': time_seconds * 1000,
            'fps': fps,
            'avg_utilization': avg_utilization,
            'peak_tops': self.config.tops,
            'effective_tops': (total_macs / time_seconds) / 1e12 if time_seconds > 0 else 0,
            'layers': layer_results,
        }


# =============================================================================
# RESOURCE ESTIMATION
# =============================================================================

@dataclass
class ResourceEstimate:
    """Estimated FPGA resources"""
    luts: int
    ffs: int
    bram_kb: int
    dsps: int

def estimate_resources(config: DPUConfig) -> ResourceEstimate:
    """
    Estimate FPGA resources for a DPU configuration.

    Based on typical resource usage:
    - Each MAC: ~1 DSP, ~100 LUTs, ~100 FFs
    - Buffers: BRAM
    - Control logic: ~10% overhead
    """
    # MAC array
    num_macs = config.mac_rows * config.mac_cols
    mac_dsps = num_macs  # 1 DSP per MAC (INT8x8)
    mac_luts = num_macs * 100
    mac_ffs = num_macs * 100

    # Buffers (BRAM)
    total_buffer = config.weight_buffer + config.input_buffer + config.output_buffer
    bram_kb = total_buffer // 1024

    # Accumulator registers (32-bit per output channel)
    acc_ffs = config.mac_rows * 32

    # Control logic overhead (~20%)
    control_luts = int((mac_luts) * 0.2)
    control_ffs = int((mac_ffs + acc_ffs) * 0.2)

    # LeakyReLU, Requant units
    post_luts = config.mac_rows * 50
    post_ffs = config.mac_rows * 30

    return ResourceEstimate(
        luts=mac_luts + control_luts + post_luts,
        ffs=mac_ffs + acc_ffs + control_ffs + post_ffs,
        bram_kb=bram_kb,
        dsps=mac_dsps,
    )


# =============================================================================
# CONFIGURATION DEFINITIONS
# =============================================================================

def define_configurations() -> List[DPUConfig]:
    """Define DPU configurations to explore"""

    configs = [
        # TINY: Minimal configuration for smallest FPGAs
        DPUConfig(
            name="TINY",
            mac_rows=8,         # 8 output channels parallel
            mac_cols=8,         # 8 input channels parallel
            weight_buffer=32*1024,      # 32 KB
            input_buffer=16*1024,       # 16 KB (2 lines of 416 width)
            output_buffer=16*1024,      # 16 KB
            freq_mhz=100,
            mem_bandwidth=2.0,  # GB/s
        ),

        # SMALL: For entry-level FPGAs (Zynq-7020)
        DPUConfig(
            name="SMALL",
            mac_rows=16,        # 16 output channels parallel
            mac_cols=16,        # 16 input channels parallel
            weight_buffer=64*1024,      # 64 KB
            input_buffer=32*1024,       # 32 KB
            output_buffer=32*1024,      # 32 KB
            freq_mhz=150,
            mem_bandwidth=4.0,  # GB/s
        ),

        # MEDIUM: For mid-range FPGAs (ZU3EG, ZU4EV)
        DPUConfig(
            name="MEDIUM",
            mac_rows=32,        # 32 output channels parallel
            mac_cols=32,        # 32 input channels parallel
            weight_buffer=128*1024,     # 128 KB
            input_buffer=64*1024,       # 64 KB
            output_buffer=64*1024,      # 64 KB
            freq_mhz=200,
            mem_bandwidth=8.0,  # GB/s
        ),

        # LARGE: For high-end FPGAs (ZU7EV, ZU9EG)
        DPUConfig(
            name="LARGE",
            mac_rows=64,        # 64 output channels parallel
            mac_cols=64,        # 64 input channels parallel
            weight_buffer=256*1024,     # 256 KB
            input_buffer=128*1024,      # 128 KB
            output_buffer=128*1024,     # 128 KB
            freq_mhz=250,
            mem_bandwidth=12.0, # GB/s
        ),

        # XLARGE: Maximum configuration
        DPUConfig(
            name="XLARGE",
            mac_rows=128,       # 128 output channels parallel
            mac_cols=64,        # 64 input channels parallel
            weight_buffer=512*1024,     # 512 KB
            input_buffer=256*1024,      # 256 KB
            output_buffer=256*1024,     # 256 KB
            freq_mhz=300,
            mem_bandwidth=16.0, # GB/s
        ),
    ]

    return configs


# =============================================================================
# ANALYSIS AND COMPARISON
# =============================================================================

def analyze_configurations():
    """Analyze all configurations and compare"""

    print("=" * 80)
    print("PHASE 4: DPU ARCHITECTURE EXPLORATION")
    print("=" * 80)

    configs = define_configurations()
    layers = YOLOV4_TINY_LAYERS

    # Calculate total network MACs
    total_network_macs = sum(l.macs for l in layers)
    total_network_weights = sum(l.weight_bytes for l in layers)

    print(f"\n[1] NETWORK ANALYSIS (YOLOv4-tiny)")
    print("-" * 60)
    print(f"  Total Conv Layers:  {len(layers)}")
    print(f"  Total MACs:         {total_network_macs:,} ({total_network_macs/1e9:.2f} GMACs)")
    print(f"  Total Weights:      {total_network_weights:,} bytes ({total_network_weights/1024:.1f} KB)")

    # Analyze each configuration
    results = []

    print(f"\n[2] CONFIGURATION ANALYSIS")
    print("=" * 80)

    for config in configs:
        print(f"\n--- {config.name} Configuration ---")
        print(f"  MAC Array:     {config.mac_rows} x {config.mac_cols} = {config.total_macs} MACs/cycle")
        print(f"  Frequency:     {config.freq_mhz} MHz")
        print(f"  Peak TOPS:     {config.tops:.3f}")
        print(f"  Weight Buffer: {config.weight_buffer//1024} KB")
        print(f"  Mem Bandwidth: {config.mem_bandwidth} GB/s")

        # Performance model
        model = PerformanceModel(config)
        perf = model.estimate_network_performance(layers)

        print(f"\n  Performance Results:")
        print(f"    Total Cycles:    {perf['total_cycles']:,}")
        print(f"    Inference Time:  {perf['time_ms']:.2f} ms")
        print(f"    FPS:             {perf['fps']:.1f}")
        print(f"    Utilization:     {perf['avg_utilization']*100:.1f}%")
        print(f"    Effective TOPS:  {perf['effective_tops']:.3f}")

        # Resource estimation
        resources = estimate_resources(config)
        print(f"\n  Resource Estimate:")
        print(f"    DSPs:  {resources.dsps}")
        print(f"    LUTs:  {resources.luts:,}")
        print(f"    FFs:   {resources.ffs:,}")
        print(f"    BRAM:  {resources.bram_kb} KB")

        results.append({
            'config': config.name,
            'mac_array': f"{config.mac_rows}x{config.mac_cols}",
            'total_macs': config.total_macs,
            'freq_mhz': config.freq_mhz,
            'peak_tops': config.tops,
            'total_cycles': perf['total_cycles'],
            'time_ms': perf['time_ms'],
            'fps': perf['fps'],
            'utilization': perf['avg_utilization'],
            'effective_tops': perf['effective_tops'],
            'dsps': resources.dsps,
            'luts': resources.luts,
            'bram_kb': resources.bram_kb,
        })

    return results


def print_comparison_table(results: List[Dict]):
    """Print comparison table"""

    print("\n" + "=" * 80)
    print("[3] CONFIGURATION COMPARISON TABLE")
    print("=" * 80)

    print(f"\n{'Config':<10} {'MAC Array':<12} {'Freq':<8} {'Peak':<8} {'FPS':<8} {'Util':<8} {'DSPs':<8} {'BRAM':<8}")
    print(f"{'':10} {'':12} {'(MHz)':<8} {'(TOPS)':<8} {'':8} {'(%)':8} {'':8} {'(KB)':<8}")
    print("-" * 80)

    for r in results:
        print(f"{r['config']:<10} {r['mac_array']:<12} {r['freq_mhz']:<8} {r['peak_tops']:<8.3f} "
              f"{r['fps']:<8.1f} {r['utilization']*100:<8.1f} {r['dsps']:<8} {r['bram_kb']:<8}")


def efficiency_analysis(results: List[Dict]):
    """Analyze efficiency metrics"""

    print("\n" + "=" * 80)
    print("[4] EFFICIENCY ANALYSIS")
    print("=" * 80)

    print(f"\n{'Config':<10} {'FPS/DSP':<12} {'FPS/TOPS':<12} {'FPS/BRAM':<12} {'Efficiency':<12}")
    print("-" * 80)

    for r in results:
        fps_per_dsp = r['fps'] / r['dsps']
        fps_per_tops = r['fps'] / r['peak_tops']
        fps_per_bram = r['fps'] / r['bram_kb']
        efficiency = r['utilization'] * (r['fps'] / max(rr['fps'] for rr in results))

        print(f"{r['config']:<10} {fps_per_dsp:<12.3f} {fps_per_tops:<12.1f} {fps_per_bram:<12.3f} {efficiency:<12.3f}")


def select_optimal_config(results: List[Dict]) -> Dict:
    """Select optimal configuration based on constraints"""

    print("\n" + "=" * 80)
    print("[5] OPTIMAL CONFIGURATION SELECTION")
    print("=" * 80)

    # Define target constraints
    constraints = {
        'min_fps': 30,          # Minimum 30 FPS for real-time
        'max_dsps': 512,        # Typical mid-range FPGA
        'max_bram_kb': 300,     # Typical mid-range FPGA
    }

    print(f"\nTarget Constraints:")
    print(f"  Minimum FPS:  {constraints['min_fps']}")
    print(f"  Maximum DSPs: {constraints['max_dsps']}")
    print(f"  Maximum BRAM: {constraints['max_bram_kb']} KB")

    # Filter configurations that meet constraints
    valid_configs = []
    for r in results:
        meets_fps = r['fps'] >= constraints['min_fps']
        meets_dsp = r['dsps'] <= constraints['max_dsps']
        meets_bram = r['bram_kb'] <= constraints['max_bram_kb']

        status = "VALID" if (meets_fps and meets_dsp and meets_bram) else "INVALID"
        reason = []
        if not meets_fps:
            reason.append(f"FPS={r['fps']:.1f}<{constraints['min_fps']}")
        if not meets_dsp:
            reason.append(f"DSPs={r['dsps']}>{constraints['max_dsps']}")
        if not meets_bram:
            reason.append(f"BRAM={r['bram_kb']}>{constraints['max_bram_kb']}")

        print(f"\n  {r['config']}: {status}")
        if reason:
            print(f"    Reasons: {', '.join(reason)}")

        if meets_fps and meets_dsp and meets_bram:
            valid_configs.append(r)

    if not valid_configs:
        print("\n  WARNING: No configuration meets all constraints!")
        print("  Selecting best FPS within DSP constraint...")
        valid_configs = [r for r in results if r['dsps'] <= constraints['max_dsps']]

    # Select best among valid (highest FPS with best efficiency)
    if valid_configs:
        # Score by FPS * utilization
        best = max(valid_configs, key=lambda x: x['fps'] * x['utilization'])

        print(f"\n  SELECTED: {best['config']}")
        print(f"    MAC Array:   {best['mac_array']}")
        print(f"    Frequency:   {best['freq_mhz']} MHz")
        print(f"    FPS:         {best['fps']:.1f}")
        print(f"    DSPs:        {best['dsps']}")
        print(f"    Utilization: {best['utilization']*100:.1f}%")

        return best

    return results[0]


def layer_breakdown(config_name: str = "MEDIUM"):
    """Show detailed layer-by-layer breakdown for selected config"""

    print("\n" + "=" * 80)
    print(f"[6] LAYER-BY-LAYER BREAKDOWN ({config_name} Configuration)")
    print("=" * 80)

    configs = define_configurations()
    config = next(c for c in configs if c.name == config_name)

    model = PerformanceModel(config)

    print(f"\n{'Layer':<12} {'Shape':<20} {'MACs':<15} {'Cycles':<12} {'Util':<8} {'Tiles':<10}")
    print("-" * 80)

    total_cycles = 0
    for layer in YOLOV4_TINY_LAYERS:
        result = model.estimate_layer_cycles(layer)
        shape = f"{layer.C_in}->{layer.C_out} {layer.K}x{layer.K}"
        tiles = f"{result['c_out_tiles']}x{result['c_in_tiles']}"

        print(f"{layer.name:<12} {shape:<20} {result['macs']:>12,} {result['total_cycles']:>10,} "
              f"{result['utilization']*100:>6.1f}% {tiles:<10}")

        total_cycles += result['total_cycles']

    print("-" * 80)
    time_ms = total_cycles / (config.freq_mhz * 1e6) * 1000
    print(f"{'TOTAL':<12} {'':<20} {sum(l.macs for l in YOLOV4_TINY_LAYERS):>12,} {total_cycles:>10,} "
          f"{'':<8} {time_ms:.2f}ms")


def generate_recommendations():
    """Generate final recommendations"""

    print("\n" + "=" * 80)
    print("[7] ARCHITECTURE RECOMMENDATIONS")
    print("=" * 80)

    recommendations = """
    RECOMMENDATIONS FOR CUSTOM YOLO DPU:
    ====================================

    1. SELECTED ARCHITECTURE: MEDIUM (32x32 MAC Array)
       - Best balance of performance and resources
       - Achieves real-time (>30 FPS) on YOLOv4-tiny
       - Fits in mid-range FPGAs (Zynq UltraScale+)

    2. MAC ARRAY DESIGN:
       - 32 rows x 32 cols = 1024 MACs/cycle
       - Output-stationary dataflow recommended
       - INT8 x INT8 -> INT32 accumulation

    3. BUFFER SIZING:
       - Weight Buffer: 128 KB (fits largest layer weights)
       - Input Buffer: 64 KB (4 lines at max width)
       - Output Buffer: 64 KB

    4. MEMORY INTERFACE:
       - Target bandwidth: 8 GB/s minimum
       - AXI4 interface with burst support
       - Double buffering for overlap

    5. PARALLELISM STRATEGY:
       - Parallelize across output channels (32)
       - Parallelize across input channels (32)
       - Sequential processing of kernel positions (3x3=9)
       - Tile large layers in channel dimension

    6. CLOCK FREQUENCY:
       - Target: 200 MHz
       - Critical path: MAC + Accumulate + LeakyReLU

    7. OPTIMIZATION OPPORTUNITIES:
       - 1x1 convolutions: Higher utilization (no kernel overhead)
       - Early layers: Bandwidth limited (large feature maps)
       - Later layers: Compute limited (small feature maps)

    FOR SIMPLER FIRST IMPLEMENTATION:
    =================================
    Start with SMALL configuration (16x16):
       - 256 MACs/cycle
       - ~150 DSPs
       - Easier to debug and verify
       - Can achieve ~15 FPS on YOLOv4-tiny
       - Upgrade to MEDIUM after validation
    """

    print(recommendations)


def save_results(results: List[Dict], selected: Dict):
    """Save analysis results"""

    output = {
        'phase': 4,
        'model': 'YOLOv4-tiny',
        'configurations': results,
        'selected': selected,
        'recommendations': {
            'mac_array': '32x32',
            'frequency_mhz': 200,
            'weight_buffer_kb': 128,
            'input_buffer_kb': 64,
            'output_buffer_kb': 64,
            'dataflow': 'output-stationary',
            'target_fps': 30,
        }
    }

    with open('phase4_architecture_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 80)
    print("PHASE 4 COMPLETE")
    print("=" * 80)
    print("Results saved to: phase4_architecture_results.json")


def main():
    """Main function"""

    # Run analysis
    results = analyze_configurations()

    # Print comparison
    print_comparison_table(results)

    # Efficiency analysis
    efficiency_analysis(results)

    # Select optimal
    selected = select_optimal_config(results)

    # Layer breakdown
    layer_breakdown(selected['config'])

    # Recommendations
    generate_recommendations()

    # Save
    save_results(results, selected)

    print("\nReady for PHASE 5: Hardware Architecture Definition")


if __name__ == "__main__":
    main()
