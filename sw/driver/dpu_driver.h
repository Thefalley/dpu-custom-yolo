/**
 * DPU Driver — High-level API for YOLOv4-tiny inference
 *
 * Provides layer-aware loading and execution:
 *   - Load weights/biases for all 36 layers (21 conv layers)
 *   - Load input image (INT8 CHW)
 *   - Run full inference (run_all with weight reload)
 *   - Read dual detection outputs (layer 29 + layer 35)
 */

#ifndef DPU_DRIVER_H
#define DPU_DRIVER_H

#include "../hal/dpu_hal.h"

/* --------------------------------------------------------------------------
 * Layer types (matches RTL encoding)
 * -------------------------------------------------------------------------- */
#define DPU_LAYER_CONV3X3       0
#define DPU_LAYER_CONV1X1       1
#define DPU_LAYER_ROUTE_SPLIT   2
#define DPU_LAYER_ROUTE_CONCAT  3
#define DPU_LAYER_MAXPOOL       4
#define DPU_LAYER_ROUTE_SAVE    5
#define DPU_LAYER_UPSAMPLE      6
#define DPU_LAYER_CONV1X1_LIN   7   /* Conv 1x1 without ReLU (detection output) */

/* --------------------------------------------------------------------------
 * Layer descriptor (matches golden model)
 * -------------------------------------------------------------------------- */
typedef struct {
    uint8_t  type;      /* Layer type (see defines above) */
    uint16_t c_in;
    uint16_t c_out;
    uint16_t h_in, w_in;
    uint16_t h_out, w_out;
    uint8_t  stride;
    uint16_t scale;     /* requantization scale */
} dpu_layer_desc_t;

/* Detection output info */
typedef struct {
    uint8_t  layer_idx;     /* Which internal layer (29 or 35) */
    uint16_t channels;      /* 255 = 3 * (5 + 80) */
    uint16_t grid_h;
    uint16_t grid_w;
} dpu_detection_head_t;

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
    dpu_detection_head_t det_heads[2];       /* Two YOLO detection heads */
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
 * Load weights for all conv layers via PIO.
 * @param weights  Flat array of all layer weights (concatenated, cin-contiguous layout)
 * @param total_bytes  Total size in bytes
 */
void dpu_driver_load_weights(dpu_driver_t *drv,
                             const uint8_t *weights, uint32_t total_bytes);

/**
 * Load biases for all conv layers via PIO.
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
 * Run full 36-layer inference.
 * Handles weight reload pauses automatically.
 * @return Total compute cycles
 */
uint32_t dpu_driver_run_inference(dpu_driver_t *drv);

/**
 * Read output feature map from the DPU.
 * @param output  Buffer to receive INT8 output
 * @param size    Number of bytes to read
 */
void dpu_driver_read_output(dpu_driver_t *drv,
                            int8_t *output, uint32_t size);

/**
 * Run single layer (for debugging).
 * @param layer_id  Layer index (0-35)
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
