/**
 * DPU Driver — High-level API for YOLOv4-tiny inference
 *
 * Provides layer-aware loading and execution:
 *   - Load weights/biases for all 18 layers
 *   - Load input image (INT8 CHW)
 *   - Run full inference (run_all with weight reload)
 *   - Read output feature maps
 */

#ifndef DPU_DRIVER_H
#define DPU_DRIVER_H

#include "../hal/dpu_hal.h"

/* --------------------------------------------------------------------------
 * Layer descriptor (matches golden model)
 * -------------------------------------------------------------------------- */
typedef struct {
    uint8_t  type;      /* 0=Conv3x3, 1=Conv1x1, 2=RouteSplit, 3=RouteConcat, 4=MaxPool */
    uint16_t c_in;
    uint16_t c_out;
    uint16_t h_in, w_in;
    uint16_t h_out, w_out;
    uint8_t  stride;
    uint16_t scale;     /* requantization scale (Q8.8 or similar) */
} dpu_layer_desc_t;

/* --------------------------------------------------------------------------
 * Driver context
 * -------------------------------------------------------------------------- */
typedef struct {
    dpu_hal_t hal;
    dpu_layer_desc_t layers[DPU_NUM_LAYERS];
    uint32_t weight_offsets[DPU_NUM_LAYERS]; /* byte offset in weight_buf per layer */
    uint32_t weight_sizes[DPU_NUM_LAYERS];   /* byte count per layer */
    uint32_t bias_offsets[DPU_NUM_LAYERS];
    uint32_t bias_sizes[DPU_NUM_LAYERS];
} dpu_driver_t;

/**
 * Initialize driver and program layer descriptors.
 * @param drv    Driver context (caller-allocated)
 * @param base   DPU AXI base address (e.g. 0x43C00000)
 */
void dpu_driver_init(dpu_driver_t *drv, uintptr_t base);

/**
 * Load layer descriptors and scales to hardware.
 * Must be called after dpu_driver_init and before inference.
 */
void dpu_driver_load_descriptors(dpu_driver_t *drv);

/**
 * Load weights for all 18 layers via PIO.
 * @param weights  Flat array of all layer weights (concatenated, cin-contiguous layout)
 * @param total_bytes  Total size in bytes
 */
void dpu_driver_load_weights(dpu_driver_t *drv,
                             const uint8_t *weights, uint32_t total_bytes);

/**
 * Load biases for all 18 layers via PIO.
 * @param biases   Array of int32_t biases (concatenated)
 * @param total_count  Total number of bias values
 */
void dpu_driver_load_biases(dpu_driver_t *drv,
                            const int32_t *biases, uint32_t total_count);

/**
 * Load input image to fmap_a.
 * @param input  INT8 image in CHW format (C x H0 x W0)
 * @param size   Number of bytes (should be C_in * H0 * W0)
 */
void dpu_driver_load_input(dpu_driver_t *drv,
                           const int8_t *input, uint32_t size);

/**
 * Run full 18-layer inference.
 * Handles weight reload pauses automatically.
 * @return Total compute cycles
 */
uint32_t dpu_driver_run_inference(dpu_driver_t *drv);

/**
 * Read output feature map from the DPU.
 * @param output  Buffer to receive INT8 output (C_out * H_out * W_out)
 * @param size    Number of bytes to read
 */
void dpu_driver_read_output(dpu_driver_t *drv,
                            int8_t *output, uint32_t size);

/**
 * Run single layer (for debugging).
 * @param layer_id  Layer index (0-17)
 */
void dpu_driver_run_layer(dpu_driver_t *drv, uint8_t layer_id);

/**
 * Get current layer being executed.
 */
uint8_t dpu_driver_current_layer(const dpu_driver_t *drv);

/**
 * Get performance cycle count.
 */
uint32_t dpu_driver_perf_cycles(const dpu_driver_t *drv);

#endif /* DPU_DRIVER_H */
